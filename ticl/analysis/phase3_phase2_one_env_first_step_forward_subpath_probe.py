import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ticl.analysis.phase2_guardrail import DEFAULT_PHASE2_SUMMARY, assert_phase2_green
from ticl.analysis.phase3_phase2_milestone_contract_gradient_compare import (
    DEFAULT_CHECKPOINT_PATH,
    _build_fixed_env_h,
    _build_phase2_algo,
    _build_phase3_contract_algo,
    _default_device,
    _param_diff_summary,
    _signature_diff,
    _sync_last_obs,
)
from ticl.analysis.phase3_phase2_one_env_fixed_suite_collect_drift_probe import (
    _compare_tensors,
    _max_compare_diff,
)
from ticl.analysis.critic_free_single_env_audit import _build_audit_env_cfg
from ticl.model_builder import load_model


DEFAULT_OUTPUT_JSON = (
    "/home/chen/RLPFN/artifacts/phase3_phase2_one_env_first_step_forward_subpath_probe.json"
)


def _clone_tree(node: Any) -> Any:
    if torch.is_tensor(node):
        return node.detach().clone()
    if isinstance(node, np.ndarray):
        return np.array(node, copy=True)
    if isinstance(node, dict):
        return {key: _clone_tree(value) for key, value in node.items()}
    if isinstance(node, list):
        return [_clone_tree(value) for value in node]
    if isinstance(node, tuple):
        return tuple(_clone_tree(value) for value in node)
    return copy.deepcopy(node)


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


def _capture_first_step_forward_subpath(algo, callback, vec_env, *, train_env_seed: int) -> dict[str, Any]:
    record: dict[str, Any] = {}
    original_make = algo.policy.make_vectorized_rollout_step_fn
    policy = algo.policy
    model = policy.rlpfn_model
    rwkv_core = model.rwkv_core
    mlp_extractor = policy.mlp_extractor

    original_encode_train_token = model._encode_train_token
    original_cast_token_for_rwkv_core = model._cast_token_for_rwkv_core
    original_forward_step = rwkv_core.forward_step
    original_forward_actor = mlp_extractor.forward_actor
    original_forward_critic = mlp_extractor.forward_critic
    original_get_action_dist = policy._get_action_dist_from_latent
    original_value_logits_from_latent = policy._value_logits_from_latent

    def _wrapped_encode_train_token(x_token, y_token):
        out = original_encode_train_token(x_token, y_token)
        record.setdefault("encoded_token", out.detach().cpu())
        return out

    def _wrapped_cast_token_for_rwkv_core(token):
        out = original_cast_token_for_rwkv_core(token)
        record.setdefault("rwkv_input_token", out.detach().cpu())
        return out

    def _wrapped_forward_step(token, cache):
        hidden, next_cache = original_forward_step(token, cache)
        record.setdefault("rwkv_forward_hidden", hidden.detach().cpu())
        return hidden, next_cache

    def _wrapped_forward_actor(hidden):
        out = original_forward_actor(hidden)
        record.setdefault("latent_pi", out.detach().cpu())
        return out

    def _wrapped_forward_critic(hidden):
        out = original_forward_critic(hidden)
        record.setdefault("latent_vf", out.detach().cpu())
        return out

    def _wrapped_get_action_dist(latent_pi):
        out = original_get_action_dist(latent_pi)
        record.setdefault("action_distribution_mean", out.distribution.mean.detach().cpu())
        return out

    def _wrapped_value_logits_from_latent(latent_vf):
        out = original_value_logits_from_latent(latent_vf)
        record.setdefault("value_logits", out.detach().cpu())
        return out

    model._encode_train_token = _wrapped_encode_train_token
    model._cast_token_for_rwkv_core = _wrapped_cast_token_for_rwkv_core
    rwkv_core.forward_step = _wrapped_forward_step
    mlp_extractor.forward_actor = _wrapped_forward_actor
    mlp_extractor.forward_critic = _wrapped_forward_critic
    policy._get_action_dist_from_latent = _wrapped_get_action_dist
    policy._value_logits_from_latent = _wrapped_value_logits_from_latent

    def _wrapped_make():
        base = original_make()

        def _rec(obs_t, action_t, reward_t, reward_mask_t, cache, step_idx, env_info):
            out = base(obs_t, action_t, reward_t, reward_mask_t, cache, step_idx, env_info)
            if "first_forward_subpath" in record:
                return out

            actor_out, _ = out
            action_dims = _resolve_action_dims(
                policy._build_obs_from_rollout_step_inputs(
                    obs_t.detach(),
                    action_t.detach(),
                    reward_t.detach(),
                    reward_mask_t.detach(),
                    env_info,
                ),
                action_t.detach(),
                env_info,
            )

            record["first_forward_subpath"] = {
                "step_idx": int(step_idx),
                "encoded_token": record["encoded_token"],
                "rwkv_input_token": record["rwkv_input_token"],
                "rwkv_forward_hidden": record["rwkv_forward_hidden"],
                "latent_pi": record["latent_pi"],
                "latent_vf": record["latent_vf"],
                "action_distribution_mean": record["action_distribution_mean"],
                "masked_action_mean": actor_out["action_mean"].detach().cpu(),
                "value_logits": record["value_logits"],
                "returned_action_mean": actor_out["action_mean"].detach().cpu(),
                "returned_action_std": actor_out["action_std"].detach().cpu(),
                "returned_value_logits": actor_out["value_logits"].detach().cpu(),
                "returned_values": actor_out["values"].detach().cpu(),
                "action_dim_per_sample": action_dims.detach().cpu(),
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

    algo.policy.make_vectorized_rollout_step_fn = _wrapped_make
    try:
        _sync_last_obs(algo, callback, vec_env, train_env_seed=int(train_env_seed))
        assert algo.collect_rollouts(vec_env, callback, algo.rollout_buffer, n_rollout_steps=algo.n_steps)
    finally:
        algo.policy.make_vectorized_rollout_step_fn = original_make
        model._encode_train_token = original_encode_train_token
        model._cast_token_for_rwkv_core = original_cast_token_for_rwkv_core
        rwkv_core.forward_step = original_forward_step
        mlp_extractor.forward_actor = original_forward_actor
        mlp_extractor.forward_critic = original_forward_critic
        policy._get_action_dist_from_latent = original_get_action_dist
        policy._value_logits_from_latent = original_value_logits_from_latent
    if "first_forward_subpath" not in record:
        raise RuntimeError("Failed to capture first forward subpath during trusted one-env collect.")
    return record["first_forward_subpath"]


def _summarize_forward_compare(phase2_capture: dict[str, Any], phase3_capture: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "encoded_token",
        "rwkv_input_token",
        "rwkv_forward_hidden",
        "latent_pi",
        "latent_vf",
        "action_distribution_mean",
        "masked_action_mean",
        "value_logits",
        "returned_action_mean",
        "returned_action_std",
        "returned_value_logits",
        "returned_values",
        "action_dim_per_sample",
    )
    compare = {
        key: _compare_tensors(phase2_capture[key], phase3_capture[key])
        for key in keys
    }
    compare["step_idx_phase2"] = int(phase2_capture["step_idx"])
    compare["step_idx_phase3"] = int(phase3_capture["step_idx"])
    return compare


def _localize_forward_drift(compare: dict[str, Any]) -> dict[str, Any]:
    encoded_token_diff = _max_compare_diff(compare["encoded_token"])
    rwkv_input_token_diff = _max_compare_diff(compare["rwkv_input_token"])
    hidden_diff = _max_compare_diff(compare["rwkv_forward_hidden"])
    latent_pi_diff = _max_compare_diff(compare["latent_pi"])
    latent_vf_diff = _max_compare_diff(compare["latent_vf"])
    action_mean_diff = _max_compare_diff(compare["action_distribution_mean"])
    value_logits_diff = _max_compare_diff(compare["value_logits"])
    if encoded_token_diff > 0.0:
        stage = "encoded_token_path"
        reason = "the first forward subpath already diverges at _encode_train_token"
    elif rwkv_input_token_diff > 0.0:
        stage = "rwkv_input_cast_path"
        reason = "encoded token matches, but _cast_token_for_rwkv_core already diverges"
    elif hidden_diff > 0.0:
        stage = "rwkv_forward_step_path"
        reason = "RWKV input token matches, but rwkv_core.forward_step hidden already diverges"
    elif latent_pi_diff > 0.0 or latent_vf_diff > 0.0:
        stage = "latent_head_path"
        reason = "RWKV hidden matches, but actor/critic latents diverge in mlp_extractor"
    elif action_mean_diff > 0.0 or value_logits_diff > 0.0:
        stage = "distribution_or_value_head_path"
        reason = "latents match, but distribution/value-head outputs diverge"
    else:
        stage = "no_forward_subpath_drift"
        reason = "all requested first-step forward subpath tensors match exactly"
    return {
        "stage": stage,
        "reason": reason,
        "encoded_token_max_abs_diff": float(encoded_token_diff),
        "rwkv_input_token_max_abs_diff": float(rwkv_input_token_diff),
        "rwkv_forward_hidden_max_abs_diff": float(hidden_diff),
        "latent_pi_max_abs_diff": float(latent_pi_diff),
        "latent_vf_max_abs_diff": float(latent_vf_diff),
        "action_distribution_mean_max_abs_diff": float(action_mean_diff),
        "value_logits_max_abs_diff": float(value_logits_diff),
    }


def run_phase3_phase2_one_env_first_step_forward_subpath_probe(
    *,
    checkpoint_path: str,
    device: str | None = None,
    frozen_h_seed: int = 12345,
    train_env_seed: int = 2020,
    train_rollout_seed: int = 4040,
    single_eval_pos: int = 64,
    n_steps: int = 256,
    batch_size: int = 256,
    n_epochs: int = 1,
    learning_rate: float = 2e-4,
    target_kl: float = 0.03,
    phase2_summary_path: str = DEFAULT_PHASE2_SUMMARY,
) -> dict[str, Any]:
    phase2_summary = assert_phase2_green(phase2_summary_path)
    checkpoint_path = str(Path(checkpoint_path).expanduser().resolve())
    device_obj = torch.device(str(device or _default_device()))
    args = {
        "build_seed": 4040,
        "frozen_h_seed": int(frozen_h_seed),
        "train_env_seed": int(train_env_seed),
        "train_rollout_seed": int(train_rollout_seed),
        "single_eval_pos": int(single_eval_pos),
        "n_steps": int(n_steps),
        "batch_size": int(batch_size),
        "n_epochs": int(n_epochs),
        "learning_rate": float(learning_rate),
        "target_kl": float(target_kl),
    }

    load_model.cache_clear()
    _, config = load_model(checkpoint_path, device=device_obj, verbose=False)
    env_cfg = _build_audit_env_cfg(config["prior"]["environment"])
    frozen_h = _build_fixed_env_h(prior_cfg=env_cfg, frozen_h_seed=int(frozen_h_seed))

    phase2_bundle = _build_phase2_algo(
        checkpoint_path=checkpoint_path,
        device_obj=device_obj,
        frozen_h=frozen_h,
        args=args,
    )
    phase3_bundle = _build_phase3_contract_algo(
        checkpoint_path=checkpoint_path,
        device_obj=device_obj,
        frozen_h=frozen_h,
        args=args,
    )
    try:
        initial_param_diff = _param_diff_summary(
            phase2_bundle["algo"].policy,
            phase3_bundle["algo"].policy,
        )
        phase2_capture = _capture_first_step_forward_subpath(
            phase2_bundle["algo"],
            phase2_bundle["callback"],
            phase2_bundle["vec_env"],
            train_env_seed=int(train_env_seed),
        )
        phase3_capture = _capture_first_step_forward_subpath(
            phase3_bundle["algo"],
            phase3_bundle["callback"],
            phase3_bundle["vec_env"],
            train_env_seed=int(train_env_seed),
        )
    finally:
        phase2_bundle["vec_env"].close()
        phase3_bundle["vec_env"].close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    compare = _summarize_forward_compare(phase2_capture, phase3_capture)
    localization = _localize_forward_drift(compare)
    return {
        "checkpoint_path": checkpoint_path,
        "device": str(device_obj),
        "phase2_summary_path": str(Path(phase2_summary_path).expanduser().resolve()),
        "phase2_summary_status": str(phase2_summary.get("status", "")),
        "args": args,
        "phase2_signature": phase2_bundle["signature"],
        "phase3_signature": phase3_bundle["signature"],
        "signature_diff": _signature_diff(phase2_bundle["signature"], phase3_bundle["signature"]),
        "initial_policy_param_diff": initial_param_diff,
        "compare": compare,
        "forward_subpath_localization": localization,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", type=str, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--frozen-h-seed", type=int, default=12345)
    parser.add_argument("--train-env-seed", type=int, default=2020)
    parser.add_argument("--train-rollout-seed", type=int, default=4040)
    parser.add_argument("--single-eval-pos", type=int, default=64)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument("--phase2-summary-path", type=str, default=DEFAULT_PHASE2_SUMMARY)
    parser.add_argument("--output-json", type=str, default=DEFAULT_OUTPUT_JSON)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = run_phase3_phase2_one_env_first_step_forward_subpath_probe(
        checkpoint_path=args.checkpoint_path,
        device=args.device,
        frozen_h_seed=args.frozen_h_seed,
        train_env_seed=args.train_env_seed,
        train_rollout_seed=args.train_rollout_seed,
        single_eval_pos=args.single_eval_pos,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        learning_rate=args.learning_rate,
        target_kl=args.target_kl,
        phase2_summary_path=args.phase2_summary_path,
    )
    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
