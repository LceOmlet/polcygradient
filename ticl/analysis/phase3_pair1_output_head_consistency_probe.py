import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ticl.analysis.critic_free_single_env_audit import _build_audit_env_cfg
from ticl.analysis.phase2_guardrail import DEFAULT_PHASE2_SUMMARY, assert_phase2_green
from ticl.analysis.phase3_multi_env_optimization_audit import (
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
from ticl.sb3_recurrent_ppo import (
    _array_digest_payload,
    build_recurrent_ppo,
    build_validation_recurrent_ppo_policy,
)


DEFAULT_OUTPUT_JSON = (
    "/home/chen/RLPFN/artifacts/phase3_pair1_output_head_consistency_probe.json"
)

PHASE2_EQ_N_STEPS = 256
PHASE2_EQ_BATCH_SIZE = 256
PHASE2_EQ_SINGLE_EVAL_POS = 64

_ACTION_HEAD_KEYS = (
    "action_net.weight",
    "action_net.bias",
    "log_std",
)
_VALUE_HEAD_KEYS = (
    "value_net.weight",
    "value_net.bias",
    "value_bardist.borders",
    "value_bardist.losses_per_bucket",
)


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


def _filter_head_state(state_dict: dict[str, Any], keys: tuple[str, ...]) -> dict[str, torch.Tensor]:
    filtered = {}
    for key in keys:
        if key not in state_dict:
            continue
        value = state_dict[key]
        if torch.is_tensor(value):
            filtered[str(key)] = value.detach().cpu().clone()
        else:
            filtered[str(key)] = torch.as_tensor(value).detach().cpu().clone()
    return filtered


def _digest_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, Any]:
    return {
        str(key): _array_digest_payload(value.detach().cpu().numpy())
        for key, value in state_dict.items()
    }


def _make_fixed_latent(rows: int, cols: int, *, device: torch.device, dtype: torch.dtype, scale: float) -> torch.Tensor:
    latent = torch.linspace(
        -float(scale),
        float(scale),
        steps=int(rows * cols),
        device=device,
        dtype=torch.float32,
    ).reshape(int(rows), int(cols))
    return latent.to(dtype=dtype)


def _capture_head_replay(policy) -> dict[str, torch.Tensor]:
    action_param = next(policy.action_net.parameters())
    value_param = next(policy.value_net.parameters())
    latent_pi = _make_fixed_latent(
        3,
        int(policy.action_net.in_features),
        device=action_param.device,
        dtype=action_param.dtype,
        scale=0.75,
    )
    latent_vf = _make_fixed_latent(
        3,
        int(policy.value_net.in_features),
        device=value_param.device,
        dtype=value_param.dtype,
        scale=0.5,
    )
    distribution = policy._get_action_dist_from_latent(latent_pi)
    action_mean = distribution.distribution.mean.detach().cpu().clone()
    value_logits = policy._value_logits_from_latent(latent_vf).detach().cpu().clone()
    values = policy._value_from_logits(value_logits.to(device=value_param.device, dtype=value_param.dtype)).detach().cpu().clone()
    return {
        "latent_pi_input": latent_pi.detach().cpu().clone(),
        "latent_vf_input": latent_vf.detach().cpu().clone(),
        "action_distribution_mean": action_mean,
        "value_logits": value_logits,
        "returned_values": values,
    }


def _build_pair1_algo_once(
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
    policy = algo.policy
    policy_state = {str(key): value.detach().cpu().clone() for key, value in policy.state_dict().items()}
    saved_validation_state = getattr(model, "_validation_ppo_policy_state", None)
    saved_validation_state = (
        {str(key): value.detach().cpu().clone() for key, value in saved_validation_state.items()}
        if isinstance(saved_validation_state, dict)
        else None
    )
    validation_policy = None
    validation_replay = None
    if isinstance(saved_validation_state, dict):
        validation_policy = build_validation_recurrent_ppo_policy(
            model=model,
            env_cfg=copy.deepcopy(env_cfg),
            device=str(device_obj),
            num_features=int(config["prior"]["num_features"]),
            policy_state_dict=saved_validation_state,
        )
        validation_replay = _capture_head_replay(validation_policy)

    current_replay = _capture_head_replay(policy)
    result = {
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
            "restore_validation_policy_state": bool(profile_cfg["restore_validation_policy_state"]),
        },
        "runtime": {
            "elapsed_wall_time_sec": float(time.perf_counter() - started_at),
        },
        "saved_validation_state_present": bool(isinstance(saved_validation_state, dict)),
        "saved_validation_state_keys": [] if saved_validation_state is None else sorted(saved_validation_state.keys()),
        "action_head_state": _filter_head_state(policy_state, _ACTION_HEAD_KEYS),
        "value_head_state": _filter_head_state(policy_state, _VALUE_HEAD_KEYS),
        "saved_action_head_state": {} if saved_validation_state is None else _filter_head_state(saved_validation_state, _ACTION_HEAD_KEYS),
        "saved_value_head_state": {} if saved_validation_state is None else _filter_head_state(saved_validation_state, _VALUE_HEAD_KEYS),
        "current_head_replay": current_replay,
        "validation_head_replay": validation_replay,
    }
    if validation_policy is not None:
        validation_policy.to("cpu")
    vec_env.close()
    return result


def _summarize_probe(primary: dict[str, Any], repeat: dict[str, Any]) -> dict[str, Any]:
    compare = {
        "action_head_primary_vs_repeat": _tensor_tree_compare(primary["action_head_state"], repeat["action_head_state"]),
        "value_head_primary_vs_repeat": _tensor_tree_compare(primary["value_head_state"], repeat["value_head_state"]),
        "current_replay_primary_vs_repeat": _tensor_tree_compare(primary["current_head_replay"], repeat["current_head_replay"]),
        "saved_action_head_primary_vs_repeat": _tensor_tree_compare(primary["saved_action_head_state"], repeat["saved_action_head_state"]),
        "saved_value_head_primary_vs_repeat": _tensor_tree_compare(primary["saved_value_head_state"], repeat["saved_value_head_state"]),
        "saved_state_key_lists_equal": bool(primary["saved_validation_state_keys"] == repeat["saved_validation_state_keys"]),
    }
    if primary["validation_head_replay"] is not None and repeat["validation_head_replay"] is not None:
        compare["validation_replay_primary_vs_repeat"] = _tensor_tree_compare(
            primary["validation_head_replay"],
            repeat["validation_head_replay"],
        )
    compare["primary_current_vs_saved_action_head"] = _tensor_tree_compare(
        primary["action_head_state"],
        primary["saved_action_head_state"],
    )
    compare["primary_current_vs_saved_value_head"] = _tensor_tree_compare(
        primary["value_head_state"],
        primary["saved_value_head_state"],
    )
    compare["repeat_current_vs_saved_action_head"] = _tensor_tree_compare(
        repeat["action_head_state"],
        repeat["saved_action_head_state"],
    )
    compare["repeat_current_vs_saved_value_head"] = _tensor_tree_compare(
        repeat["value_head_state"],
        repeat["saved_value_head_state"],
    )
    if primary["validation_head_replay"] is not None:
        compare["primary_current_replay_vs_saved_validation_replay"] = _tensor_tree_compare(
            primary["current_head_replay"],
            primary["validation_head_replay"],
        )
    if repeat["validation_head_replay"] is not None:
        compare["repeat_current_replay_vs_saved_validation_replay"] = _tensor_tree_compare(
            repeat["current_head_replay"],
            repeat["validation_head_replay"],
        )
    return compare


def _localize(compare: dict[str, Any], primary: dict[str, Any], repeat: dict[str, Any]) -> dict[str, Any]:
    action_head_repeat_diff = _max_compare_diff(compare["action_head_primary_vs_repeat"])
    value_head_repeat_diff = _max_compare_diff(compare["value_head_primary_vs_repeat"])
    replay_repeat_diff = _max_compare_diff(compare["current_replay_primary_vs_repeat"])
    primary_action_vs_saved = _max_compare_diff(compare["primary_current_vs_saved_action_head"])
    primary_value_vs_saved = _max_compare_diff(compare["primary_current_vs_saved_value_head"])
    repeat_action_vs_saved = _max_compare_diff(compare["repeat_current_vs_saved_action_head"])
    repeat_value_vs_saved = _max_compare_diff(compare["repeat_current_vs_saved_value_head"])
    restore_flag = bool(primary["config"]["restore_validation_policy_state"])
    saved_state_present = bool(primary["saved_validation_state_present"] and repeat["saved_validation_state_present"])

    if (
        saved_state_present
        and (not restore_flag)
        and (action_head_repeat_diff > 0.0 or value_head_repeat_diff > 0.0)
        and (
            primary_action_vs_saved > 0.0
            or primary_value_vs_saved > 0.0
            or repeat_action_vs_saved > 0.0
            or repeat_value_vs_saved > 0.0
        )
    ):
        stage = "unrestored_validation_head_state"
        reason = (
            "checkpoint already carries saved PPO head state, but the guarded contract builds with "
            "restore_validation_policy_state=False, so repeated builds keep fresh random action/value heads"
        )
    elif action_head_repeat_diff > 0.0 or value_head_repeat_diff > 0.0:
        stage = "repeated_build_head_parameter_drift"
        reason = "repeated guarded builds already disagree at action/value head parameters"
    elif replay_repeat_diff > 0.0:
        stage = "repeated_build_head_output_drift_without_param_drift"
        reason = "head parameters match, but direct head replay on fixed latent still diverges"
    else:
        stage = "head_consistent"
        reason = "repeated guarded builds keep action/value head parameters and direct outputs consistent"
    return {
        "stage": stage,
        "reason": reason,
        "checkpoint_has_saved_validation_policy_state": bool(saved_state_present),
        "contract_restore_validation_policy_state": bool(restore_flag),
        "action_head_primary_vs_repeat_max_abs_diff": float(action_head_repeat_diff),
        "value_head_primary_vs_repeat_max_abs_diff": float(value_head_repeat_diff),
        "current_replay_primary_vs_repeat_max_abs_diff": float(replay_repeat_diff),
        "primary_current_vs_saved_action_head_max_abs_diff": float(primary_action_vs_saved),
        "primary_current_vs_saved_value_head_max_abs_diff": float(primary_value_vs_saved),
        "repeat_current_vs_saved_action_head_max_abs_diff": float(repeat_action_vs_saved),
        "repeat_current_vs_saved_value_head_max_abs_diff": float(repeat_value_vs_saved),
    }


def build_pair1_output_head_consistency_probe(
    *,
    checkpoint_path: str = PAIR1_CHECKPOINT_PATH,
    train_suite_path: str = PAIR1_TRAIN_SUITE_PATH,
    phase2_summary_path: str = DEFAULT_PHASE2_SUMMARY,
    device: str | None = None,
) -> dict[str, Any]:
    preflight = _load_pair1_preflight()
    primary = _build_pair1_algo_once(
        checkpoint_path=checkpoint_path,
        train_suite_path=train_suite_path,
        phase2_summary_path=phase2_summary_path,
        device=device,
    )
    repeat = _build_pair1_algo_once(
        checkpoint_path=checkpoint_path,
        train_suite_path=train_suite_path,
        phase2_summary_path=phase2_summary_path,
        device=device,
    )
    compare = _summarize_probe(primary, repeat)
    localized = _localize(compare, primary, repeat)
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
        "primary_head_digest_summary": {
            "action_head": _digest_state_dict(primary["action_head_state"]),
            "value_head": _digest_state_dict(primary["value_head_state"]),
            "saved_action_head": _digest_state_dict(primary["saved_action_head_state"]),
            "saved_value_head": _digest_state_dict(primary["saved_value_head_state"]),
        },
        "repeat_head_digest_summary": {
            "action_head": _digest_state_dict(repeat["action_head_state"]),
            "value_head": _digest_state_dict(repeat["value_head_state"]),
            "saved_action_head": _digest_state_dict(repeat["saved_action_head_state"]),
            "saved_value_head": _digest_state_dict(repeat["saved_value_head_state"]),
        },
        "head_consistency_compare": compare,
        "localized_drift": localized,
        "repeated_guarded_build_output_heads_consistent": bool(localized["stage"] == "head_consistent"),
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

    result = build_pair1_output_head_consistency_probe(
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
