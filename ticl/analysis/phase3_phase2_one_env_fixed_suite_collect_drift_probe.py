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
from ticl.model_builder import load_model
from ticl.analysis.critic_free_single_env_audit import _build_audit_env_cfg


DEFAULT_OUTPUT_JSON = (
    "/home/chen/RLPFN/artifacts/phase3_phase2_one_env_fixed_suite_collect_drift_probe.json"
)

_ENV_INFO_CAPTURE_KEYS = (
    "action_dim_per_sample",
    "obs_dim_per_sample",
    "state_dim_per_sample",
    "obs_slot_dim_per_sample",
    "action_slot_dim_per_sample",
    "terminal_reset_enabled",
    "init_state_std",
    "init_action_std",
)

_NUMERIC_RESET_FIELDS = (
    "_state_t",
    "_action_t",
    "_reward_t",
    "_reward_mask_t",
    "_terminal_t",
    "_single_eval_pos",
)

_BUFFER_STEP0_FIELDS = (
    "observations",
    "actions",
    "rewards",
    "episode_starts",
    "values",
    "log_probs",
    "objective_masks",
    "next_states",
    "next_state_masks",
    "action_masks",
)


def _to_cpu_tensor(x: Any) -> torch.Tensor:
    if torch.is_tensor(x):
        return x.detach().cpu()
    return torch.as_tensor(x)


def _to_jsonable(x: Any) -> Any:
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().cpu().tolist()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    return x


def _compare_tensors(a: Any, b: Any) -> dict[str, Any]:
    return _compare_tensors_with_options(a, b, include_values=True)


def _compare_tensors_with_options(a: Any, b: Any, *, include_values: bool) -> dict[str, Any]:
    a_t = _to_cpu_tensor(a)
    b_t = _to_cpu_tensor(b)
    if tuple(a_t.shape) != tuple(b_t.shape):
        raise ValueError(f"Tensor shape mismatch: {tuple(a_t.shape)} vs {tuple(b_t.shape)}")
    exact_match = bool(torch.equal(a_t, b_t))
    a_f = a_t.to(dtype=torch.float32)
    b_f = b_t.to(dtype=torch.float32)
    max_abs_diff = 0.0 if a_f.numel() == 0 else float((a_f - b_f).abs().max().item())
    result = {
        "shape": list(a_t.shape),
        "dtype_phase2": str(a_t.dtype),
        "dtype_phase3": str(b_t.dtype),
        "exact_match": exact_match,
        "max_abs_diff": max_abs_diff,
    }
    if include_values:
        result["phase2"] = _to_jsonable(a_t)
        result["phase3"] = _to_jsonable(b_t)
    return result


def _flatten_tensor_tree(node: Any, *, prefix: str = "root") -> dict[str, torch.Tensor]:
    flat: dict[str, torch.Tensor] = {}
    if torch.is_tensor(node) or isinstance(node, np.ndarray):
        flat[str(prefix)] = _to_cpu_tensor(node)
        return flat
    if isinstance(node, (float, int, bool, np.floating, np.integer, np.bool_)):
        flat[str(prefix)] = _to_cpu_tensor(node)
        return flat
    if isinstance(node, dict):
        for key in sorted(node.keys(), key=str):
            flat.update(_flatten_tensor_tree(node[key], prefix=f"{prefix}.{key}"))
        return flat
    if isinstance(node, (list, tuple)):
        for idx, item in enumerate(node):
            flat.update(_flatten_tensor_tree(item, prefix=f"{prefix}[{idx}]"))
        return flat
    if node is None:
        return flat
    if isinstance(node, (str, bytes)):
        return flat
    raise TypeError(f"Unsupported tensor tree leaf type: {type(node)!r}")


def _max_compare_diff(compare: dict[str, Any]) -> float:
    return float(compare.get("max_abs_diff", 0.0))


def _capture_reset_and_first_step(algo, callback, vec_env, *, train_env_seed: int) -> dict[str, Any]:
    record: dict[str, Any] = {}
    original_make = algo.policy.make_vectorized_rollout_step_fn

    def _wrapped_make():
        base = original_make()

        def _rec(obs_t, action_t, reward_t, reward_mask_t, cache, step_idx, env_info):
            out = base(obs_t, action_t, reward_t, reward_mask_t, cache, step_idx, env_info)
            if "first_policy_step" not in record:
                actor_out, _ = out
                env_info_record = {}
                for key in _ENV_INFO_CAPTURE_KEYS:
                    if key in env_info:
                        env_info_record[str(key)] = _to_cpu_tensor(env_info[key])
                obs_full = algo.policy._build_obs_from_rollout_step_inputs(
                    obs_t,
                    action_t,
                    reward_t,
                    reward_mask_t,
                    env_info,
                )
                record["first_policy_step"] = {
                    "step_idx": int(step_idx),
                    "obs_t": _to_cpu_tensor(obs_t),
                    "action_t": _to_cpu_tensor(action_t),
                    "reward_t": _to_cpu_tensor(reward_t),
                    "reward_mask_t": _to_cpu_tensor(reward_mask_t),
                    "obs_full": _to_cpu_tensor(obs_full),
                    "cache_inputs": _flatten_tensor_tree(cache),
                    "env_info": env_info_record,
                    "env_info_all_tensors": _flatten_tensor_tree(env_info, prefix="env_info"),
                    "actor_outputs": {
                        "action_mean": _to_cpu_tensor(actor_out["action_mean"]),
                        "action_std": _to_cpu_tensor(actor_out["action_std"]),
                        "values": _to_cpu_tensor(actor_out["values"]),
                        "value_logits": _to_cpu_tensor(actor_out["value_logits"]),
                    },
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
        record["reset"] = {
            "last_obs": _to_cpu_tensor(algo._last_obs),
            "last_episode_starts": _to_cpu_tensor(algo._last_episode_starts),
            "reset_infos": copy.deepcopy(getattr(vec_env, "reset_infos", None)),
            "internal": {
                field: _to_cpu_tensor(getattr(vec_env, field))
                for field in _NUMERIC_RESET_FIELDS
            },
        }
        assert algo.collect_rollouts(vec_env, callback, algo.rollout_buffer, n_rollout_steps=algo.n_steps)
        rollout_buffer = algo.rollout_buffer
        record["buffer_step0"] = {
            str(field): _to_cpu_tensor(getattr(rollout_buffer, field)[0])
            for field in _BUFFER_STEP0_FIELDS
        }
    finally:
        algo.policy.make_vectorized_rollout_step_fn = original_make
    return record


def _summarize_capture_compare(phase2_capture: dict[str, Any], phase3_capture: dict[str, Any]) -> dict[str, Any]:
    reset_compare = {
        "last_obs": _compare_tensors(
            phase2_capture["reset"]["last_obs"],
            phase3_capture["reset"]["last_obs"],
        ),
        "last_episode_starts": _compare_tensors(
            phase2_capture["reset"]["last_episode_starts"],
            phase3_capture["reset"]["last_episode_starts"],
        ),
        "internal": {
            key: _compare_tensors(
                phase2_capture["reset"]["internal"][key],
                phase3_capture["reset"]["internal"][key],
            )
            for key in sorted(phase2_capture["reset"]["internal"].keys())
        },
        "reset_infos_phase2": _to_jsonable(phase2_capture["reset"]["reset_infos"]),
        "reset_infos_phase3": _to_jsonable(phase3_capture["reset"]["reset_infos"]),
        "reset_infos_exact_match": bool(
            phase2_capture["reset"]["reset_infos"] == phase3_capture["reset"]["reset_infos"]
        ),
    }
    first_step_compare = {
        "step_idx_phase2": int(phase2_capture["first_policy_step"]["step_idx"]),
        "step_idx_phase3": int(phase3_capture["first_policy_step"]["step_idx"]),
        "inputs": {
            key: _compare_tensors(
                phase2_capture["first_policy_step"][key],
                phase3_capture["first_policy_step"][key],
            )
            for key in ("obs_t", "action_t", "reward_t", "reward_mask_t", "obs_full")
        },
        "cache_inputs": {
            key: _compare_tensors_with_options(
                phase2_capture["first_policy_step"]["cache_inputs"][key],
                phase3_capture["first_policy_step"]["cache_inputs"][key],
                include_values=False,
            )
            for key in sorted(phase2_capture["first_policy_step"]["cache_inputs"].keys())
        },
        "env_info": {
            key: _compare_tensors(
                phase2_capture["first_policy_step"]["env_info"][key],
                phase3_capture["first_policy_step"]["env_info"][key],
            )
            for key in sorted(phase2_capture["first_policy_step"]["env_info"].keys())
        },
        "env_info_all_tensors": {
            key: _compare_tensors_with_options(
                phase2_capture["first_policy_step"]["env_info_all_tensors"][key],
                phase3_capture["first_policy_step"]["env_info_all_tensors"][key],
                include_values=False,
            )
            for key in sorted(phase2_capture["first_policy_step"]["env_info_all_tensors"].keys())
        },
        "actor_outputs": {
            key: _compare_tensors(
                phase2_capture["first_policy_step"]["actor_outputs"][key],
                phase3_capture["first_policy_step"]["actor_outputs"][key],
            )
            for key in sorted(phase2_capture["first_policy_step"]["actor_outputs"].keys())
        },
    }
    buffer_step0_compare = {
        key: _compare_tensors(
            phase2_capture["buffer_step0"][key],
            phase3_capture["buffer_step0"][key],
        )
        for key in sorted(phase2_capture["buffer_step0"].keys())
    }
    return {
        "reset_compare": reset_compare,
        "first_policy_step_compare": first_step_compare,
        "buffer_step0_compare": buffer_step0_compare,
    }


def _localize_drift(compare: dict[str, Any]) -> dict[str, Any]:
    reset_numeric_diffs = [
        _max_compare_diff(compare["reset_compare"]["last_obs"]),
        _max_compare_diff(compare["reset_compare"]["last_episode_starts"]),
        *[
            _max_compare_diff(item)
            for item in compare["reset_compare"]["internal"].values()
        ],
    ]
    first_step_input_diffs = [
        _max_compare_diff(item)
        for item in compare["first_policy_step_compare"]["inputs"].values()
    ]
    first_step_cache_diffs = [
        _max_compare_diff(item)
        for item in compare["first_policy_step_compare"]["cache_inputs"].values()
    ]
    first_step_env_info_diffs = [
        _max_compare_diff(item)
        for item in compare["first_policy_step_compare"]["env_info"].values()
    ]
    first_step_env_info_all_diffs = [
        _max_compare_diff(item)
        for item in compare["first_policy_step_compare"]["env_info_all_tensors"].values()
    ]
    first_step_actor_diffs = [
        _max_compare_diff(item)
        for item in compare["first_policy_step_compare"]["actor_outputs"].values()
    ]
    buffer_step0_diffs = [
        _max_compare_diff(item)
        for item in compare["buffer_step0_compare"].values()
    ]
    reset_max = max(reset_numeric_diffs) if reset_numeric_diffs else 0.0
    first_input_max = max(first_step_input_diffs) if first_step_input_diffs else 0.0
    first_cache_max = max(first_step_cache_diffs) if first_step_cache_diffs else 0.0
    first_env_info_max = max(first_step_env_info_diffs) if first_step_env_info_diffs else 0.0
    first_env_info_all_max = max(first_step_env_info_all_diffs) if first_step_env_info_all_diffs else 0.0
    first_actor_max = max(first_step_actor_diffs) if first_step_actor_diffs else 0.0
    buffer_max = max(buffer_step0_diffs) if buffer_step0_diffs else 0.0
    obs_full_diff = _max_compare_diff(compare["first_policy_step_compare"]["inputs"]["obs_full"])
    base_input_diff = max(first_input_max, first_cache_max, first_env_info_max, first_env_info_all_max)
    if base_input_diff > 0.0:
        stage = "first_policy_step_path"
        reason = (
            "the strict collect path has already diverged by the first policy step "
            "(step inputs, cache inputs, or env_info tensors differ)"
        )
    elif obs_full_diff > 0.0:
        stage = "obs_builder_path"
        reason = "_build_obs_from_rollout_step_inputs already diverges on the first policy step"
    elif first_actor_max > 0.0:
        stage = "model_forward_path"
        reason = (
            "first policy-step inputs and obs_full match, but actor outputs already diverge "
            "inside the policy forward path"
        )
    elif reset_max > 0.0:
        stage = "vec_env_reset_side_path"
        reason = (
            "vec_env.reset() outputs differ, but the strict collect path reaches the first policy step "
            "with identical step inputs and actor outputs"
        )
    elif buffer_max > 0.0:
        stage = "first_env_transition_or_step_sink"
        reason = "reset and first policy-step actor outputs match, but first stored rollout step differs"
    else:
        stage = "after_first_rollout_step"
        reason = "reset, first policy-step, and first stored rollout step all match exactly"
    return {
        "stage": stage,
        "reason": reason,
        "reset_numeric_max_abs_diff": float(reset_max),
        "first_policy_step_input_max_abs_diff": float(first_input_max),
        "first_policy_step_cache_input_max_abs_diff": float(first_cache_max),
        "first_policy_step_env_info_max_abs_diff": float(first_env_info_max),
        "first_policy_step_env_info_all_tensor_max_abs_diff": float(first_env_info_all_max),
        "first_policy_step_obs_full_max_abs_diff": float(obs_full_diff),
        "first_policy_step_actor_output_max_abs_diff": float(first_actor_max),
        "buffer_step0_max_abs_diff": float(buffer_max),
        "reset_info_metadata_exact_match": bool(compare["reset_compare"]["reset_infos_exact_match"]),
    }


def run_phase3_phase2_one_env_fixed_suite_collect_drift_probe(
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
        phase2_capture = _capture_reset_and_first_step(
            phase2_bundle["algo"],
            phase2_bundle["callback"],
            phase2_bundle["vec_env"],
            train_env_seed=int(train_env_seed),
        )
        phase3_capture = _capture_reset_and_first_step(
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

    compare = _summarize_capture_compare(phase2_capture, phase3_capture)
    localization = _localize_drift(compare)
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
        "drift_localization": localization,
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
    report = run_phase3_phase2_one_env_fixed_suite_collect_drift_probe(
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
