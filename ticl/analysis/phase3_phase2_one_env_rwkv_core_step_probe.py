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
    _compare_tensors_with_options,
    _flatten_tensor_tree,
    _max_compare_diff,
)
from ticl.model_builder import load_model


DEFAULT_OUTPUT_JSON = "/home/chen/RLPFN/artifacts/phase3_phase2_one_env_rwkv_core_step_probe.json"


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


def _tensor_tree_compare(a: Any, b: Any) -> dict[str, Any]:
    flat_a = _flatten_tensor_tree(a)
    flat_b = _flatten_tensor_tree(b)
    keys_a = set(flat_a.keys())
    keys_b = set(flat_b.keys())
    common_keys = sorted(keys_a & keys_b)
    leaf_summaries = {}
    max_abs_diff = 0.0
    exact_all = True
    for key in common_keys:
        summary = _compare_tensors_with_options(flat_a[key], flat_b[key], include_values=False)
        leaf_summaries[key] = summary
        max_abs_diff = max(max_abs_diff, _max_compare_diff(summary))
        exact_all = exact_all and bool(summary["exact_match"])
    return {
        "leaf_count_phase2": int(len(flat_a)),
        "leaf_count_phase3": int(len(flat_b)),
        "missing_in_phase2": sorted(keys_b - keys_a),
        "missing_in_phase3": sorted(keys_a - keys_b),
        "exact_match": bool(exact_all and keys_a == keys_b),
        "max_abs_diff": float(max_abs_diff),
        "leaf_summaries": leaf_summaries,
    }


def _build_official_flat_state_snapshot(core, *, state: Any, token: torch.Tensor) -> list[torch.Tensor]:
    token_bf16 = token.to(dtype=torch.bfloat16)
    if state is None:
        flat_state = core._init_official_eval_state(device=token_bf16.device, dtype=token_bf16.dtype)
    elif core._is_official_eval_state(state):
        flat_state = [
            t.to(
                device=token_bf16.device,
                dtype=core._official_eval_state_tensor_dtype(i, token_dtype=token_bf16.dtype),
            ).contiguous()
            for i, t in enumerate(state)
        ]
    else:
        flat_state = [
            t.to(
                device=token_bf16.device,
                dtype=core._official_eval_state_tensor_dtype(i, token_dtype=token_bf16.dtype),
            ).contiguous()
            for i, t in enumerate(core._flatten_official_eval_state(state))
        ]
    return [t.detach().clone() for t in flat_state]


def _run_native_core_step(core, *, token: torch.Tensor, state: Any) -> torch.Tensor:
    if state is None:
        state = core.init_state(int(token.shape[0]), device=token.device, dtype=token.dtype)
    x = token.detach().clone()
    state = _clone_tree(state)
    v_first = None
    autocast_enabled = bool(token.is_cuda)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
        for block, block_state in zip(core.blocks, state):
            x, _, v_first = block.forward_step(x, block_state, v_first)
        x = core.ln_out(x)
    return x.detach().clone()


def _official_weight_snapshot(core, *, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu()
        for key, value in core._build_official_eval_weights(device=device).items()
    }


def _capture_first_core_step(algo, callback, vec_env, *, train_env_seed: int) -> dict[str, Any]:
    record: dict[str, Any] = {}
    core = algo.policy.rlpfn_model.rwkv_core
    original_make = algo.policy.make_vectorized_rollout_step_fn
    original_init_state = core.init_state
    original_forward_step = core.forward_step

    def _wrapped_init_state(batch_size: int, *, device: torch.device, dtype: torch.dtype):
        state = original_init_state(batch_size, device=device, dtype=dtype)
        if "init_state" not in record:
            record["init_state"] = _clone_tree(state)
            record["init_state_request"] = {
                "batch_size": int(batch_size),
                "device": str(device),
                "dtype": str(dtype),
            }
        return state

    def _wrapped_forward_step(token: torch.Tensor, state=None):
        if "core_step" in record:
            return original_forward_step(token, state)
        state_snapshot = _clone_tree(state)
        path_flags = {
            "batch_size": int(token.shape[0]),
            "token_device": str(token.device),
            "token_dtype": str(token.dtype),
            "token_is_cuda": bool(token.is_cuda),
            "core_training": bool(core.training),
            "torch_grad_enabled": bool(torch.is_grad_enabled()),
            "uses_official_eval_path": bool(
                int(token.shape[0]) == 1
                and bool(token.is_cuda)
                and (not bool(core.training))
                and (not bool(torch.is_grad_enabled()))
            ),
            "official_eval_core_cached_before": bool(core._official_eval_core is not None),
        }
        official_flat_state = _build_official_flat_state_snapshot(
            core,
            state=state_snapshot,
            token=token.detach(),
        )
        hidden, next_state = original_forward_step(token, state)
        native_hidden = _run_native_core_step(
            core,
            token=token.detach(),
            state=state_snapshot,
        )
        record["core_step"] = {
            "path_flags": path_flags | {
                "official_eval_core_cached_after": bool(core._official_eval_core is not None),
                "hidden_dtype": str(hidden.dtype),
                "hidden_device": str(hidden.device),
            },
            "token_arg": token.detach().cpu(),
            "state_arg": state_snapshot,
            "official_flat_state_input": [t.detach().cpu() for t in official_flat_state],
            "actual_hidden": hidden.detach().cpu(),
            "native_replay_hidden": native_hidden.detach().cpu(),
            "next_state": _clone_tree(next_state),
            "actual_vs_native_hidden": _compare_tensors_with_options(
                hidden.detach().cpu(),
                native_hidden.detach().cpu(),
                include_values=False,
            ),
        }
        return hidden, next_state

    core.init_state = _wrapped_init_state
    core.forward_step = _wrapped_forward_step
    algo.policy.make_vectorized_rollout_step_fn = original_make
    try:
        _sync_last_obs(algo, callback, vec_env, train_env_seed=int(train_env_seed))
        assert algo.collect_rollouts(vec_env, callback, algo.rollout_buffer, n_rollout_steps=algo.n_steps)
    finally:
        core.init_state = original_init_state
        core.forward_step = original_forward_step
        algo.policy.make_vectorized_rollout_step_fn = original_make
    if "core_step" not in record:
        raise RuntimeError("Failed to capture first RWKV core forward_step.")
    record["official_weight_snapshot"] = _official_weight_snapshot(core, device=algo.device)
    return record


def _summarize_core_compare(phase2_capture: dict[str, Any], phase3_capture: dict[str, Any]) -> dict[str, Any]:
    core2 = phase2_capture["core_step"]
    core3 = phase3_capture["core_step"]
    return {
        "path_flags_phase2": core2["path_flags"],
        "path_flags_phase3": core3["path_flags"],
        "path_flags_exact_match": bool(core2["path_flags"] == core3["path_flags"]),
        "init_state_request_phase2": phase2_capture.get("init_state_request"),
        "init_state_request_phase3": phase3_capture.get("init_state_request"),
        "init_state_compare": _tensor_tree_compare(
            phase2_capture.get("init_state"),
            phase3_capture.get("init_state"),
        ),
        "token_arg": _compare_tensors(core2["token_arg"], core3["token_arg"]),
        "state_arg_compare": _tensor_tree_compare(core2["state_arg"], core3["state_arg"]),
        "official_flat_state_input_compare": _tensor_tree_compare(
            core2["official_flat_state_input"],
            core3["official_flat_state_input"],
        ),
        "official_weight_snapshot_compare": _tensor_tree_compare(
            phase2_capture["official_weight_snapshot"],
            phase3_capture["official_weight_snapshot"],
        ),
        "actual_hidden": _compare_tensors(core2["actual_hidden"], core3["actual_hidden"]),
        "native_replay_hidden": _compare_tensors(core2["native_replay_hidden"], core3["native_replay_hidden"]),
        "phase2_actual_vs_native_hidden": core2["actual_vs_native_hidden"],
        "phase3_actual_vs_native_hidden": core3["actual_vs_native_hidden"],
        "next_state_compare": _tensor_tree_compare(core2["next_state"], core3["next_state"]),
    }


def _localize_core_step_drift(compare: dict[str, Any]) -> dict[str, Any]:
    token_diff = _max_compare_diff(compare["token_arg"])
    init_state_diff = _max_compare_diff(compare["init_state_compare"])
    state_arg_diff = _max_compare_diff(compare["state_arg_compare"])
    official_flat_state_diff = _max_compare_diff(compare["official_flat_state_input_compare"])
    official_weight_diff = _max_compare_diff(compare["official_weight_snapshot_compare"])
    actual_hidden_diff = _max_compare_diff(compare["actual_hidden"])
    native_hidden_diff = _max_compare_diff(compare["native_replay_hidden"])
    if token_diff > 0.0:
        stage = "core_input_token"
        reason = "RWKV core input token differs before forward_step"
    elif init_state_diff > 0.0 or state_arg_diff > 0.0:
        stage = "core_cache_state"
        reason = "RWKV core state/cache differs before forward_step"
    elif official_flat_state_diff > 0.0:
        stage = "official_eval_state_cast"
        reason = "raw cache matches, but official eval flat-state conversion differs"
    elif official_weight_diff > 0.0:
        stage = "official_eval_weight_snapshot"
        reason = "token/state match, but official eval weight snapshots differ"
    elif native_hidden_diff > 0.0:
        stage = "native_core_replay"
        reason = "token/state/weights match, but native core replay differs"
    elif actual_hidden_diff > 0.0:
        stage = "official_eval_kernel_path"
        reason = "token/state/weights/native replay match, but actual official eval forward_step differs"
    else:
        stage = "no_core_step_drift"
        reason = "RWKV core token, cache/state, weights, native replay, and actual hidden all match"
    return {
        "stage": stage,
        "reason": reason,
        "token_arg_max_abs_diff": float(token_diff),
        "init_state_max_abs_diff": float(init_state_diff),
        "state_arg_max_abs_diff": float(state_arg_diff),
        "official_flat_state_input_max_abs_diff": float(official_flat_state_diff),
        "official_weight_snapshot_max_abs_diff": float(official_weight_diff),
        "native_replay_hidden_max_abs_diff": float(native_hidden_diff),
        "actual_hidden_max_abs_diff": float(actual_hidden_diff),
        "phase2_actual_vs_native_hidden_max_abs_diff": _max_compare_diff(
            compare["phase2_actual_vs_native_hidden"]
        ),
        "phase3_actual_vs_native_hidden_max_abs_diff": _max_compare_diff(
            compare["phase3_actual_vs_native_hidden"]
        ),
        "path_flags_exact_match": bool(compare["path_flags_exact_match"]),
    }


def run_phase3_phase2_one_env_rwkv_core_step_probe(
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
    wall_t0 = time.perf_counter()
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
        phase2_capture = _capture_first_core_step(
            phase2_bundle["algo"],
            phase2_bundle["callback"],
            phase2_bundle["vec_env"],
            train_env_seed=int(train_env_seed),
        )
        phase3_capture = _capture_first_core_step(
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

    compare = _summarize_core_compare(phase2_capture, phase3_capture)
    localization = _localize_core_step_drift(compare)
    return {
        "checkpoint_path": checkpoint_path,
        "device": str(device_obj),
        "elapsed_wall_time_sec": float(time.perf_counter() - wall_t0),
        "phase2_summary_path": str(Path(phase2_summary_path).expanduser().resolve()),
        "phase2_summary_status": str(phase2_summary.get("status", "")),
        "args": args,
        "phase2_signature": phase2_bundle["signature"],
        "phase3_signature": phase3_bundle["signature"],
        "signature_diff": _signature_diff(phase2_bundle["signature"], phase3_bundle["signature"]),
        "initial_policy_param_diff": initial_param_diff,
        "compare": compare,
        "core_step_localization": localization,
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
    report = run_phase3_phase2_one_env_rwkv_core_step_probe(
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
