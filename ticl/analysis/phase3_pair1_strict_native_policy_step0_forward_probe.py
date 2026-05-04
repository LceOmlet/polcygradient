import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ticl.analysis.critic_free_single_env_audit import (
    _build_audit_env_cfg,
    _make_deterministic_batch_plan,
)
from ticl.analysis.phase2_guardrail import DEFAULT_PHASE2_SUMMARY, assert_phase2_green
from ticl.analysis.phase3_multi_env_optimization_audit import (
    _bind_vec_env_to_fixed_suite,
    _clone_suite_h_list,
    _default_device,
    _load_required_suite,
    _resolve_train_profile,
)
from ticl.analysis.phase3_pair1_train_update_ab_compare_pack import (
    PAIR1_CHECKPOINT_PATH,
    PAIR1_TRAIN_SUITE_PATH,
    _load_pair1_preflight,
)
from ticl.analysis.phase3_phase2_one_env_fixed_suite_collect_drift_probe import (
    _compare_tensors_with_options,
    _flatten_tensor_tree,
    _max_compare_diff,
)
from ticl.model_builder import load_model
from ticl.priors.environment_prior import EnvironmentPrior
from ticl.sb3_recurrent_ppo import _array_digest_payload, build_recurrent_ppo


DEFAULT_OUTPUT_JSON = (
    "/home/chen/RLPFN/artifacts/phase3_pair1_strict_native_policy_step0_forward_probe.json"
)

PHASE2_EQ_N_STEPS = 256
PHASE2_EQ_BATCH_SIZE = 256
PHASE2_EQ_SINGLE_EVAL_POS = 64

_RAW_TENSOR_KEYS = (
    "obs_full",
    "reward_token_y",
    "encoded_token",
    "rwkv_input_token",
    "rwkv_forward_token_arg",
    "rwkv_forward_hidden",
    "latent_pi",
    "latent_vf",
    "action_distribution_mean",
    "returned_action_mean",
    "value_logits",
    "returned_values",
    "action_dim_per_sample",
)


def _clone_tree(node: Any) -> Any:
    if torch.is_tensor(node):
        return node.detach().cpu().clone()
    if isinstance(node, np.ndarray):
        return np.array(node, copy=True)
    if isinstance(node, dict):
        return {key: _clone_tree(value) for key, value in node.items()}
    if isinstance(node, list):
        return [_clone_tree(value) for value in node]
    if isinstance(node, tuple):
        return tuple(_clone_tree(value) for value in node)
    return copy.deepcopy(node)


def _to_jsonable(node: Any) -> Any:
    if node is None:
        return None
    if torch.is_tensor(node):
        return node.detach().cpu().tolist()
    if isinstance(node, np.ndarray):
        return node.tolist()
    if isinstance(node, (np.floating, np.integer, np.bool_)):
        return node.item()
    if isinstance(node, dict):
        return {str(key): _to_jsonable(value) for key, value in node.items()}
    if isinstance(node, (list, tuple)):
        return [_to_jsonable(value) for value in node]
    return node


def _tensor_tree_compare(left: Any, right: Any) -> dict[str, Any]:
    flat_left = _flatten_tensor_tree(left)
    flat_right = _flatten_tensor_tree(right)
    keys_left = set(flat_left.keys())
    keys_right = set(flat_right.keys())
    common_keys = sorted(keys_left & keys_right)
    leaf_summaries = {}
    max_abs_diff = 0.0
    exact_all = bool(keys_left == keys_right)
    for key in common_keys:
        summary = _compare_tensors_with_options(flat_left[key], flat_right[key], include_values=False)
        leaf_summaries[key] = summary
        max_abs_diff = max(max_abs_diff, _max_compare_diff(summary))
        exact_all = exact_all and bool(summary["exact_match"])
    return {
        "leaf_count_left": int(len(flat_left)),
        "leaf_count_right": int(len(flat_right)),
        "missing_in_left": sorted(keys_right - keys_left),
        "missing_in_right": sorted(keys_left - keys_right),
        "exact_match": bool(exact_all),
        "max_abs_diff": float(max_abs_diff),
        "leaf_summaries": leaf_summaries,
    }


def _tensor_tree_digest_summary(node: Any) -> dict[str, Any]:
    flat = _flatten_tensor_tree(node)
    return {
        str(key): _array_digest_payload(value.detach().cpu().numpy() if torch.is_tensor(value) else value)
        for key, value in flat.items()
    }


def _resolve_action_dims(obs_full: torch.Tensor, action_t: torch.Tensor, env_info: dict[str, Any]) -> torch.Tensor:
    action_dims = env_info.get("action_dim_per_sample", None)
    if action_dims is None:
        return torch.full(
            (int(obs_full.shape[0]),),
            int(action_t.shape[-1]),
            device=obs_full.device,
            dtype=torch.long,
        )
    if torch.is_tensor(action_dims):
        return action_dims.to(device=obs_full.device, dtype=torch.long).reshape(-1)
    return torch.as_tensor(action_dims, device=obs_full.device, dtype=torch.long).reshape(-1)


def _capture_digest_summary(capture: dict[str, Any]) -> dict[str, Any]:
    tensor_digests = {
        key: _array_digest_payload(
            capture[key].detach().cpu().numpy() if torch.is_tensor(capture[key]) else np.asarray(capture[key])
        )
        for key in _RAW_TENSOR_KEYS
    }
    return {
        "step_idx": int(capture["step_idx"]),
        "path_flags": dict(capture["path_flags"]),
        "tensor_digests": tensor_digests,
        "init_state_digests": _tensor_tree_digest_summary(capture["init_state"]),
        "state_arg_digests": _tensor_tree_digest_summary(capture["state_arg"]),
        "next_state_digests": _tensor_tree_digest_summary(capture["next_state"]),
    }


def _capture_first_policy_step_subpath(
    *,
    checkpoint_path: str,
    train_suite_path: str,
    phase2_summary_path: str,
    device: str | None,
) -> dict[str, Any]:
    assert_phase2_green(phase2_summary_path)
    checkpoint_path = str(Path(checkpoint_path).expanduser().resolve())
    train_suite_path = str(Path(train_suite_path).expanduser().resolve())
    device_obj = torch.device(str(device or _default_device()))
    suite = _load_required_suite(train_suite_path)

    started_at = time.perf_counter()
    load_model.cache_clear()
    model, config = load_model(checkpoint_path, device=device_obj, verbose=False)
    env_cfg = _build_audit_env_cfg(config["prior"]["environment"])
    prior = EnvironmentPrior(copy.deepcopy(env_cfg))
    prior._sample_batch_hypers = lambda batch_n: _clone_suite_h_list(suite)
    prior._sample_single_eval_pos = lambda n_samples_arg, rng_seed=None: int(PHASE2_EQ_SINGLE_EVAL_POS)

    optimizer_cfg = dict(config.get("optimizer", {}))
    profile_cfg = _resolve_train_profile(
        optimizer_cfg=optimizer_cfg,
        train_profile="phase2_shared_backbone_contract",
        batch_size=int(PHASE2_EQ_BATCH_SIZE),
        n_epochs=1,
        learning_rate=0.0002,
        target_kl=0.03,
    )
    algo, callback, vec_env = build_recurrent_ppo(
        model=model,
        env_prior=prior,
        device=str(device_obj),
        num_features=int(config["prior"]["num_features"]),
        n_envs=int(suite["batch_size"]),
        n_steps=int(PHASE2_EQ_N_STEPS),
        learning_rate=float(profile_cfg["learning_rate"]),
        batch_size=int(profile_cfg["batch_size"]),
        n_epochs=int(profile_cfg["n_epochs"]),
        gamma=float(optimizer_cfg.get("ppo_gamma", 1.0)),
        gae_lambda=float(optimizer_cfg.get("ppo_gae_lambda", 0.95)),
        clip_range=optimizer_cfg.get("ppo_clip_range", 0.2),
        clip_range_vf=None,
        normalize_advantage=bool(profile_cfg["normalize_advantage"]),
        actor_gae_space=str(profile_cfg["actor_gae_space"]),
        actor_baseline_mode=str(profile_cfg["actor_baseline_mode"]),
        actor_objective_mode=str(profile_cfg["actor_objective_mode"]),
        strict_fixed_env_mode=True,
        env_rng_seeds=[int(v) for v in suite["env_seeds"]],
        rollout_rng_seeds=[int(v) for v in suite["rollout_seeds"]],
        deterministic_actor_sampling=True,
        deterministic_batch_plan=True,
        strict_native_rollout=True,
        actor_objective_runtime_current_suite_name=None,
        reset_env_state_at_sep=bool(profile_cfg["reset_env_state_at_sep"]),
        separate_value_backbone=bool(profile_cfg["separate_value_backbone"]),
        restore_validation_policy_state=bool(profile_cfg["restore_validation_policy_state"]),
        runtime_normalized_q_value_weight_override=profile_cfg["runtime_normalized_q_value_weight_override"],
        runtime_next_state_flow_matching_weight_override=profile_cfg["runtime_next_state_flow_matching_weight_override"],
        ent_coef=float(optimizer_cfg.get("ppo_ent_coef", 0.0)),
        vf_coef=float(profile_cfg["vf_coef"]),
        max_grad_norm=float(optimizer_cfg.get("ppo_max_grad_norm", 0.5)),
        target_kl=profile_cfg["target_kl"],
        verbose=0,
    )
    _bind_vec_env_to_fixed_suite(vec_env, suite=suite, single_eval_pos=int(PHASE2_EQ_SINGLE_EVAL_POS))
    _make_deterministic_batch_plan(algo.rollout_buffer)
    algo.ep_info_buffer = []
    algo.ep_success_buffer = []
    algo.policy.reset_rollout_cache(vec_env.num_envs)
    algo._last_obs = vec_env.reset()
    algo._last_episode_starts = np.ones((vec_env.num_envs,), dtype=bool)
    algo._last_lstm_states = algo.policy._dummy_states(vec_env.num_envs)
    callback.init_callback(algo)
    algo._update_current_progress_remaining(0, int(PHASE2_EQ_N_STEPS))

    record: dict[str, Any] = {}
    policy = algo.policy
    rwkv_core = policy.rlpfn_model.rwkv_core
    mlp_extractor = policy.mlp_extractor
    original_make = policy.make_vectorized_rollout_step_fn
    original_encode_train_token = policy.rlpfn_model._encode_train_token
    original_cast_token_for_rwkv_core = policy.rlpfn_model._cast_token_for_rwkv_core
    original_init_state = rwkv_core.init_state
    original_forward_step = rwkv_core.forward_step
    original_forward_actor = mlp_extractor.forward_actor
    original_forward_critic = mlp_extractor.forward_critic
    original_get_action_dist = policy._get_action_dist_from_latent
    original_value_logits_from_latent = policy._value_logits_from_latent

    def _wrapped_encode_train_token(x_token, y_token):
        out = original_encode_train_token(x_token, y_token)
        record.setdefault("reward_token_y", y_token.detach().cpu().clone())
        record.setdefault("encoded_token", out.detach().cpu().clone())
        return out

    def _wrapped_cast_token_for_rwkv_core(token):
        out = original_cast_token_for_rwkv_core(token)
        record.setdefault("rwkv_input_token", out.detach().cpu().clone())
        return out

    def _wrapped_init_state(batch_size: int, *, device: torch.device, dtype: torch.dtype):
        state = original_init_state(batch_size, device=device, dtype=dtype)
        record.setdefault("init_state", _clone_tree(state))
        return state

    def _wrapped_forward_step(token, state=None):
        record.setdefault(
            "path_flags",
            {
                "batch_size": int(token.shape[0]),
                "token_device": str(token.device),
                "token_dtype": str(token.dtype),
                "token_is_cuda": bool(token.is_cuda),
                "core_training": bool(rwkv_core.training),
                "torch_grad_enabled": bool(torch.is_grad_enabled()),
                "force_native_eval_forward_step": bool(getattr(rwkv_core, "force_native_eval_forward_step", False)),
            },
        )
        record.setdefault("rwkv_forward_token_arg", token.detach().cpu().clone())
        record.setdefault("state_arg", _clone_tree(state))
        hidden, next_state = original_forward_step(token, state)
        record.setdefault("rwkv_forward_hidden", hidden.detach().cpu().clone())
        record.setdefault("next_state", _clone_tree(next_state))
        return hidden, next_state

    def _wrapped_forward_actor(hidden):
        out = original_forward_actor(hidden)
        record.setdefault("latent_pi", out.detach().cpu().clone())
        return out

    def _wrapped_forward_critic(hidden):
        out = original_forward_critic(hidden)
        record.setdefault("latent_vf", out.detach().cpu().clone())
        return out

    def _wrapped_get_action_dist(latent_pi):
        out = original_get_action_dist(latent_pi)
        record.setdefault("action_distribution_mean", out.distribution.mean.detach().cpu().clone())
        return out

    def _wrapped_value_logits_from_latent(latent_vf):
        out = original_value_logits_from_latent(latent_vf)
        record.setdefault("value_logits", out.detach().cpu().clone())
        return out

    def _wrapped_make():
        base = original_make()

        def _rec(obs_t, action_t, reward_t, reward_mask_t, cache, step_idx, env_info):
            out = base(obs_t, action_t, reward_t, reward_mask_t, cache, step_idx, env_info)
            if "first_policy_step" in record:
                return out
            actor_out, _ = out
            obs_full = policy._build_obs_from_rollout_step_inputs(
                obs_t.detach(),
                action_t.detach(),
                reward_t.detach(),
                reward_mask_t.detach(),
                env_info,
            )
            action_dims = _resolve_action_dims(obs_full, action_t.detach(), env_info)
            record["first_policy_step"] = {
                "step_idx": int(step_idx),
                "obs_full": obs_full.detach().cpu().clone(),
                "reward_token_y": record["reward_token_y"],
                "encoded_token": record["encoded_token"],
                "rwkv_input_token": record["rwkv_input_token"],
                "rwkv_forward_token_arg": record["rwkv_forward_token_arg"],
                "rwkv_forward_hidden": record["rwkv_forward_hidden"],
                "latent_pi": record["latent_pi"],
                "latent_vf": record["latent_vf"],
                "action_distribution_mean": record["action_distribution_mean"],
                "returned_action_mean": actor_out["action_mean"].detach().cpu().clone(),
                "value_logits": record["value_logits"],
                "returned_values": actor_out["values"].detach().cpu().clone(),
                "action_dim_per_sample": action_dims.detach().cpu().clone(),
                "init_state": record["init_state"],
                "state_arg": record["state_arg"],
                "next_state": record["next_state"],
                "path_flags": dict(record["path_flags"]),
            }
            return out

        for name in (
            "_policy_actor_log_prob_fn",
            "_policy_actor_score_fn",
            "_policy_actor_decomp_stats_fn",
            "_fit_action_dim_fn",
            "_clear_buffers",
            "_model_ref",
            "_policy_actor_sample_fn",
        ):
            if hasattr(base, name):
                setattr(_rec, name, getattr(base, name))
        return _rec

    policy.rlpfn_model._encode_train_token = _wrapped_encode_train_token
    policy.rlpfn_model._cast_token_for_rwkv_core = _wrapped_cast_token_for_rwkv_core
    rwkv_core.init_state = _wrapped_init_state
    rwkv_core.forward_step = _wrapped_forward_step
    mlp_extractor.forward_actor = _wrapped_forward_actor
    mlp_extractor.forward_critic = _wrapped_forward_critic
    policy._get_action_dist_from_latent = _wrapped_get_action_dist
    policy._value_logits_from_latent = _wrapped_value_logits_from_latent
    policy.make_vectorized_rollout_step_fn = _wrapped_make
    try:
        ok = algo.collect_rollouts(vec_env, callback, algo.rollout_buffer, n_rollout_steps=algo.n_steps)
        if not ok:
            raise RuntimeError("collect_rollouts returned False")
    finally:
        policy.make_vectorized_rollout_step_fn = original_make
        policy.rlpfn_model._encode_train_token = original_encode_train_token
        policy.rlpfn_model._cast_token_for_rwkv_core = original_cast_token_for_rwkv_core
        rwkv_core.init_state = original_init_state
        rwkv_core.forward_step = original_forward_step
        mlp_extractor.forward_actor = original_forward_actor
        mlp_extractor.forward_critic = original_forward_critic
        policy._get_action_dist_from_latent = original_get_action_dist
        policy._value_logits_from_latent = original_value_logits_from_latent
        vec_env.close()
    if "first_policy_step" not in record:
        raise RuntimeError("Failed to capture first strict-native policy step.")
    return {
        "config": {
            "checkpoint_path": checkpoint_path,
            "train_suite_path": train_suite_path,
            "device": str(device_obj),
            "n_steps": int(PHASE2_EQ_N_STEPS),
            "batch_size": int(PHASE2_EQ_BATCH_SIZE),
            "single_eval_pos": int(PHASE2_EQ_SINGLE_EVAL_POS),
            "strict_fixed_env_mode": True,
            "deterministic_actor_sampling": True,
            "deterministic_batch_plan": True,
            "strict_native_rollout": True,
            "actor_objective_mode": str(profile_cfg["actor_objective_mode"]),
        },
        "runtime": {
            "elapsed_wall_time_sec": float(time.perf_counter() - started_at),
            "rollout_wall_time_sec": float(getattr(algo, "_rwkv_last_rollout_wall_time_sec", 0.0)),
        },
        "capture": record["first_policy_step"],
    }


def _summarize_forward_compare(primary_capture: dict[str, Any], repeat_capture: dict[str, Any]) -> dict[str, Any]:
    primary = primary_capture["capture"]
    repeat = repeat_capture["capture"]
    compare = {
        key: _compare_tensors_with_options(primary[key], repeat[key], include_values=False)
        for key in _RAW_TENSOR_KEYS
    }
    compare["step_idx_primary"] = int(primary["step_idx"])
    compare["step_idx_repeat"] = int(repeat["step_idx"])
    compare["path_flags_primary"] = dict(primary["path_flags"])
    compare["path_flags_repeat"] = dict(repeat["path_flags"])
    compare["path_flags_exact_match"] = bool(primary["path_flags"] == repeat["path_flags"])
    compare["init_state_compare"] = _tensor_tree_compare(primary["init_state"], repeat["init_state"])
    compare["state_arg_compare"] = _tensor_tree_compare(primary["state_arg"], repeat["state_arg"])
    compare["next_state_compare"] = _tensor_tree_compare(primary["next_state"], repeat["next_state"])
    return compare


def _localize_forward_drift(compare: dict[str, Any]) -> dict[str, Any]:
    obs_full_diff = _max_compare_diff(compare["obs_full"])
    reward_token_diff = _max_compare_diff(compare["reward_token_y"])
    encoded_token_diff = _max_compare_diff(compare["encoded_token"])
    rwkv_input_diff = _max_compare_diff(compare["rwkv_input_token"])
    rwkv_token_arg_diff = _max_compare_diff(compare["rwkv_forward_token_arg"])
    init_state_diff = _max_compare_diff(compare["init_state_compare"])
    state_arg_diff = _max_compare_diff(compare["state_arg_compare"])
    hidden_diff = _max_compare_diff(compare["rwkv_forward_hidden"])
    latent_pi_diff = _max_compare_diff(compare["latent_pi"])
    latent_vf_diff = _max_compare_diff(compare["latent_vf"])
    action_dist_diff = _max_compare_diff(compare["action_distribution_mean"])
    action_mean_diff = _max_compare_diff(compare["returned_action_mean"])
    value_logits_diff = _max_compare_diff(compare["value_logits"])
    returned_values_diff = _max_compare_diff(compare["returned_values"])
    if obs_full_diff > 0.0:
        stage = "obs_full_path"
        reason = "strict-native step-0 drift already appears in obs_full before reward-token encoding"
    elif reward_token_diff > 0.0:
        stage = "reward_token_path"
        reason = "obs_full matches, but reward-token extraction already diverges"
    elif encoded_token_diff > 0.0:
        stage = "encoded_token_path"
        reason = "reward-token path matches, but _encode_train_token already diverges"
    elif rwkv_input_diff > 0.0 or rwkv_token_arg_diff > 0.0:
        stage = "rwkv_input_cast_path"
        reason = "encoded token matches, but RWKV core input token diverges before forward_step"
    elif not bool(compare["path_flags_exact_match"]):
        stage = "rwkv_path_flags"
        reason = "token/state tensors may match, but forward_step dtype/device/native-path flags diverge"
    elif init_state_diff > 0.0 or state_arg_diff > 0.0:
        stage = "rwkv_cache_state_path"
        reason = "RWKV token matches, but init/cache state diverges before forward_step"
    elif hidden_diff > 0.0:
        stage = "rwkv_forward_step_path"
        reason = "token and cache match, but rwkv_core.forward_step hidden diverges"
    elif latent_pi_diff > 0.0 or latent_vf_diff > 0.0:
        stage = "latent_head_path"
        reason = "RWKV hidden matches, but actor/critic latent heads diverge"
    elif action_dist_diff > 0.0 or action_mean_diff > 0.0 or value_logits_diff > 0.0 or returned_values_diff > 0.0:
        stage = "distribution_or_value_head_path"
        reason = "latents match, but policy/value output heads diverge"
    else:
        stage = "no_step0_forward_drift"
        reason = "all requested strict-native step-0 forward-subpath tensors match exactly"
    return {
        "stage": stage,
        "reason": reason,
        "obs_full_max_abs_diff": float(obs_full_diff),
        "reward_token_y_max_abs_diff": float(reward_token_diff),
        "encoded_token_max_abs_diff": float(encoded_token_diff),
        "rwkv_input_token_max_abs_diff": float(rwkv_input_diff),
        "rwkv_forward_token_arg_max_abs_diff": float(rwkv_token_arg_diff),
        "init_state_max_abs_diff": float(init_state_diff),
        "state_arg_max_abs_diff": float(state_arg_diff),
        "rwkv_forward_hidden_max_abs_diff": float(hidden_diff),
        "latent_pi_max_abs_diff": float(latent_pi_diff),
        "latent_vf_max_abs_diff": float(latent_vf_diff),
        "action_distribution_mean_max_abs_diff": float(action_dist_diff),
        "returned_action_mean_max_abs_diff": float(action_mean_diff),
        "value_logits_max_abs_diff": float(value_logits_diff),
        "returned_values_max_abs_diff": float(returned_values_diff),
    }


def build_pair1_strict_native_policy_step0_forward_probe(
    *,
    checkpoint_path: str = PAIR1_CHECKPOINT_PATH,
    train_suite_path: str = PAIR1_TRAIN_SUITE_PATH,
    phase2_summary_path: str = DEFAULT_PHASE2_SUMMARY,
    device: str | None = None,
) -> dict[str, Any]:
    preflight = _load_pair1_preflight()
    primary = _capture_first_policy_step_subpath(
        checkpoint_path=checkpoint_path,
        train_suite_path=train_suite_path,
        phase2_summary_path=phase2_summary_path,
        device=device,
    )
    repeat = _capture_first_policy_step_subpath(
        checkpoint_path=checkpoint_path,
        train_suite_path=train_suite_path,
        phase2_summary_path=phase2_summary_path,
        device=device,
    )
    compare = _summarize_forward_compare(primary, repeat)
    localized = _localize_forward_drift(compare)
    return {
        "checkpoint_path": str(Path(checkpoint_path).expanduser().resolve()),
        "train_suite_path": str(Path(train_suite_path).expanduser().resolve()),
        "phase2_summary_path": str(Path(phase2_summary_path).expanduser().resolve()),
        "device": str(device or _default_device()),
        "preflight_contract": {
            "path": preflight.get("path"),
            "baseline_mode": preflight.get("baseline", {}).get("actor_objective_mode"),
            "override_mode": preflight.get("override", {}).get("actor_objective_mode"),
            "unexpected_diff": dict(preflight.get("unexpected_diff", {})),
        },
        "primary_runtime": dict(primary["runtime"]),
        "repeat_runtime": dict(repeat["runtime"]),
        "primary_capture_summary": _capture_digest_summary(primary["capture"]),
        "repeat_capture_summary": _capture_digest_summary(repeat["capture"]),
        "step0_forward_compare": compare,
        "localized_drift": localized,
        "strict_native_step0_same_contract_stable": bool(localized["stage"] == "no_step0_forward_drift"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=str, default=PAIR1_CHECKPOINT_PATH)
    parser.add_argument("--train-suite-path", type=str, default=PAIR1_TRAIN_SUITE_PATH)
    parser.add_argument("--phase2-summary-path", type=str, default=DEFAULT_PHASE2_SUMMARY)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-json", type=str, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output_path}")

    result = build_pair1_strict_native_policy_step0_forward_probe(
        checkpoint_path=args.checkpoint_path,
        train_suite_path=args.train_suite_path,
        phase2_summary_path=args.phase2_summary_path,
        device=args.device,
    )
    output_path.write_text(json.dumps(_to_jsonable(result), indent=2, sort_keys=True))
    print(json.dumps(result["localized_drift"], indent=2, sort_keys=True))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
