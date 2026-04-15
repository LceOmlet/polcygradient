import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

import torch

from ticl.analysis.critic_free_single_env_audit import _build_audit_env_cfg
from ticl.analysis.phase2_guardrail import DEFAULT_PHASE2_SUMMARY, assert_phase2_green
from ticl.analysis.phase3_phase2_milestone_contract_gradient_compare import (
    DEFAULT_CHECKPOINT_PATH,
    _build_fixed_env_h,
    _build_phase2_algo,
    _build_phase3_contract_algo,
    _collect_and_measure_main_loss,
    _default_device,
    _flatten_policy_grad,
    _max_abs_diff_tensor,
    _param_diff_summary,
    _signature_diff,
)
from ticl.analysis.phase3_phase2_one_env_fixed_suite_collect_drift_probe import (
    _compare_tensors,
    _max_compare_diff,
)
from ticl.model_builder import load_model


DEFAULT_OUTPUT_JSON = (
    "/home/chen/RLPFN/artifacts/phase3_phase2_official_eval_vs_native_contract_probe.json"
)
DEFAULT_MILESTONE_COMPARE_JSON = (
    "/home/chen/RLPFN/artifacts/phase3_phase2_milestone_contract_gradient_compare.json"
)
DEFAULT_CORE_STEP_JSON = (
    "/home/chen/RLPFN/artifacts/phase3_phase2_one_env_rwkv_core_step_probe.json"
)


def _clone_tree(node: Any) -> Any:
    if torch.is_tensor(node):
        return node.detach().clone()
    if isinstance(node, dict):
        return {key: _clone_tree(value) for key, value in node.items()}
    if isinstance(node, list):
        return [_clone_tree(value) for value in node]
    if isinstance(node, tuple):
        return tuple(_clone_tree(value) for value in node)
    return copy.deepcopy(node)


def _unflatten_official_eval_state_for_native(core, state, *, token: torch.Tensor):
    if not core._is_official_eval_state(state):
        return state
    batch_size = int(token.shape[0])
    if batch_size != 1:
        raise RuntimeError(
            "Official eval flat state can only be converted back to native state for batch_size=1."
        )
    nested_state = []
    for layer_idx in range(int(core.nlayers)):
        att_x_prev = state[3 * layer_idx].to(device=token.device, dtype=token.dtype).unsqueeze(0)
        att_kv = state[3 * layer_idx + 1].to(device=token.device, dtype=torch.float32).unsqueeze(0)
        ffn_x_prev = state[3 * layer_idx + 2].to(device=token.device, dtype=token.dtype).unsqueeze(0)
        nested_state.append((att_x_prev.contiguous(), att_kv.contiguous(), ffn_x_prev.contiguous()))
    return nested_state


def _native_forward_step(core, token: torch.Tensor, state=None):
    if token.ndim != 2:
        raise ValueError(f"RWKV7Core.forward_step expects (B, C), got {tuple(token.shape)}")
    batch_size = int(token.shape[0])
    if state is None:
        state = core.init_state(batch_size, device=token.device, dtype=token.dtype)
    else:
        state = _unflatten_official_eval_state_for_native(core, state, token=token)
    x = token
    new_state = []
    v_first = None
    autocast_enabled = bool(token.is_cuda)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
        for block, block_state in zip(core.blocks, state):
            x, block_state_next, v_first = block.forward_step(x, block_state, v_first)
            new_state.append(block_state_next)
        x = core.ln_out(x)
    return x, new_state


def _install_strict_native_forward_step(policy):
    core = policy.rlpfn_model.rwkv_core
    original_forward_step = core.forward_step
    stats: dict[str, Any] = {
        "call_count": 0,
        "official_eval_state_seen_count": 0,
        "first_call": None,
    }

    def _wrapped_forward_step(token: torch.Tensor, state=None):
        call_idx = int(stats["call_count"])
        state_is_official_eval = bool(core._is_official_eval_state(state))
        if state_is_official_eval:
            stats["official_eval_state_seen_count"] = int(stats["official_eval_state_seen_count"]) + 1
        would_use_official_eval_path = bool(
            int(token.shape[0]) == 1
            and bool(token.is_cuda)
            and (not bool(core.training))
            and (not bool(torch.is_grad_enabled()))
        )
        hidden, next_state = _native_forward_step(core, token, state)
        if call_idx == 0:
            stats["first_call"] = {
                "forced_native_path": True,
                "would_use_official_eval_path": would_use_official_eval_path,
                "state_is_official_eval": state_is_official_eval,
                "batch_size": int(token.shape[0]),
                "token_device": str(token.device),
                "token_dtype": str(token.dtype),
                "token_is_cuda": bool(token.is_cuda),
                "core_training": bool(core.training),
                "torch_grad_enabled": bool(torch.is_grad_enabled()),
                "official_eval_core_cached_after": bool(core._official_eval_core is not None),
                "hidden_dtype": str(hidden.dtype),
                "hidden_device": str(hidden.device),
                "token_arg": token.detach().cpu(),
                "hidden": hidden.detach().cpu(),
            }
        stats["call_count"] = call_idx + 1
        return hidden, next_state

    core.forward_step = _wrapped_forward_step

    def _restore() -> None:
        core.forward_step = original_forward_step

    return _restore, stats


def _strip_first_call_tensors(stats: dict[str, Any]) -> dict[str, Any]:
    first_call = dict(stats.get("first_call") or {})
    first_call.pop("token_arg", None)
    first_call.pop("hidden", None)
    return {
        "call_count": int(stats.get("call_count", 0)),
        "official_eval_state_seen_count": int(stats.get("official_eval_state_seen_count", 0)),
        "first_call": first_call,
    }


def _compare_measure_outputs(phase2_measure: dict[str, Any], phase3_measure: dict[str, Any]) -> dict[str, Any]:
    grad_a = phase2_measure.pop("grad_vector")
    grad_b = phase3_measure.pop("grad_vector")
    rollout_a = phase2_measure.pop("rollout_tensors")
    rollout_b = phase3_measure.pop("rollout_tensors")
    rollout_diff = {
        key: _max_abs_diff_tensor(rollout_a[key], rollout_b[key])
        for key in sorted(rollout_a.keys())
    }
    scalar_diff = {
        key: float(abs(float(phase2_measure[key]) - float(phase3_measure[key])))
        for key in sorted(phase2_measure.keys())
    }
    grad_compare = {
        "cosine_similarity": float(
            torch.nn.functional.cosine_similarity(
                grad_a.unsqueeze(0),
                grad_b.unsqueeze(0),
                dim=1,
                eps=1e-8,
            ).item()
        ),
        "l2_delta_norm": float(torch.linalg.vector_norm(grad_b - grad_a).item()),
        "phase2_grad_norm": float(torch.linalg.vector_norm(grad_a).item()),
        "phase3_grad_norm": float(torch.linalg.vector_norm(grad_b).item()),
        "max_abs_grad_delta": _max_abs_diff_tensor(grad_a, grad_b),
    }
    return {
        "phase2_measure": phase2_measure,
        "phase3_measure": phase3_measure,
        "rollout_tensor_max_abs_diff": rollout_diff,
        "scalar_measure_abs_diff": scalar_diff,
        "grad_compare": grad_compare,
    }


def _load_json_if_exists(path: str) -> dict[str, Any] | None:
    json_path = Path(path).expanduser()
    if not json_path.exists():
        return None
    return json.loads(json_path.read_text(encoding="utf-8"))


def _reference_summary(
    *,
    milestone_compare_path: str,
    core_step_path: str,
) -> dict[str, Any]:
    milestone = _load_json_if_exists(milestone_compare_path)
    core_step = _load_json_if_exists(core_step_path)
    rollout_diff = (milestone or {}).get("rollout_tensor_max_abs_diff", {})
    return {
        "milestone_compare_path": str(Path(milestone_compare_path).expanduser().resolve()),
        "core_step_path": str(Path(core_step_path).expanduser().resolve()),
        "default_first_hidden_max_abs_diff": (
            (core_step or {}).get("core_step_localization", {}).get("actual_hidden_max_abs_diff")
        ),
        "default_rollout_max_abs_diff": max((float(v) for v in rollout_diff.values()), default=None),
        "default_old_values_max_abs_diff": rollout_diff.get("old_values"),
        "default_returns_max_abs_diff": rollout_diff.get("returns"),
        "default_advantages_max_abs_diff": rollout_diff.get("advantages"),
        "default_grad_max_abs_delta": (
            (milestone or {}).get("grad_compare", {}).get("max_abs_grad_delta")
        ),
        "default_grad_l2_delta_norm": (
            (milestone or {}).get("grad_compare", {}).get("l2_delta_norm")
        ),
    }


def _compare_reduction(reference: dict[str, Any], native_report: dict[str, Any]) -> dict[str, Any]:
    first_hidden_diff = float(_max_compare_diff(native_report["first_call_hidden_compare"]))
    rollout_max = max(
        (float(v) for v in native_report["native_measure_compare"]["rollout_tensor_max_abs_diff"].values()),
        default=0.0,
    )
    grad_max = float(native_report["native_measure_compare"]["grad_compare"]["max_abs_grad_delta"])

    def _ratio(new_value: float, old_value: Any) -> float | None:
        if old_value is None:
            return None
        old_f = float(old_value)
        if old_f == 0.0:
            return None
        return float(new_value / old_f)

    return {
        "native_first_hidden_max_abs_diff": first_hidden_diff,
        "native_rollout_max_abs_diff": float(rollout_max),
        "native_grad_max_abs_delta": grad_max,
        "first_hidden_ratio_vs_default": _ratio(
            first_hidden_diff,
            reference.get("default_first_hidden_max_abs_diff"),
        ),
        "rollout_max_ratio_vs_default": _ratio(
            rollout_max,
            reference.get("default_rollout_max_abs_diff"),
        ),
        "grad_max_ratio_vs_default": _ratio(
            grad_max,
            reference.get("default_grad_max_abs_delta"),
        ),
        "first_hidden_closed_exactly": bool(first_hidden_diff == 0.0),
        "rollout_tensors_closed_exactly": bool(rollout_max == 0.0),
        "gradient_closed_exactly": bool(grad_max == 0.0),
    }


def run_phase3_phase2_official_eval_vs_native_contract_probe(
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
    milestone_compare_path: str = DEFAULT_MILESTONE_COMPARE_JSON,
    core_step_path: str = DEFAULT_CORE_STEP_JSON,
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
    phase2_restore = None
    phase3_restore = None
    try:
        initial_param_diff = _param_diff_summary(
            phase2_bundle["algo"].policy,
            phase3_bundle["algo"].policy,
        )
        phase2_restore, phase2_native_stats = _install_strict_native_forward_step(
            phase2_bundle["algo"].policy
        )
        phase3_restore, phase3_native_stats = _install_strict_native_forward_step(
            phase3_bundle["algo"].policy
        )
        phase2_measure = _collect_and_measure_main_loss(
            phase2_bundle["algo"],
            phase2_bundle["callback"],
            phase2_bundle["vec_env"],
            train_env_seed=int(train_env_seed),
        )
        phase3_measure = _collect_and_measure_main_loss(
            phase3_bundle["algo"],
            phase3_bundle["callback"],
            phase3_bundle["vec_env"],
            train_env_seed=int(train_env_seed),
        )
    finally:
        if phase2_restore is not None:
            phase2_restore()
        if phase3_restore is not None:
            phase3_restore()
        phase2_bundle["vec_env"].close()
        phase3_bundle["vec_env"].close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    first2 = phase2_native_stats["first_call"]
    first3 = phase3_native_stats["first_call"]
    if first2 is None or first3 is None:
        raise RuntimeError("Strict native probe did not capture first core forward_step on both sides.")
    native_measure_compare = _compare_measure_outputs(phase2_measure, phase3_measure)
    reference = _reference_summary(
        milestone_compare_path=milestone_compare_path,
        core_step_path=core_step_path,
    )
    native_report = {
        "first_call_token_compare": _compare_tensors(first2["token_arg"], first3["token_arg"]),
        "first_call_hidden_compare": _compare_tensors(first2["hidden"], first3["hidden"]),
        "phase2_native_stats": _strip_first_call_tensors(phase2_native_stats),
        "phase3_native_stats": _strip_first_call_tensors(phase3_native_stats),
        "native_measure_compare": native_measure_compare,
    }
    reduction = _compare_reduction(reference, native_report)
    return {
        "audit_entry": "phase3_phase2_official_eval_vs_native_contract_probe",
        "checkpoint_path": checkpoint_path,
        "device": str(device_obj),
        "elapsed_wall_time_sec": float(time.perf_counter() - wall_t0),
        "phase2_summary_path": str(Path(phase2_summary_path).expanduser().resolve()),
        "phase2_guardrail_passed": bool(phase2_summary["validation"]["all_checks_pass"]),
        "contract_args": dict(args),
        "phase2_signature": phase2_bundle["signature"],
        "phase3_signature": phase3_bundle["signature"],
        "signature_diff": _signature_diff(phase2_bundle["signature"], phase3_bundle["signature"]),
        "initial_param_diff": initial_param_diff,
        "default_reference": reference,
        **native_report,
        "reduction_vs_default_reference": reduction,
        "conclusions": {
            "signature_exact_match": len(
                _signature_diff(phase2_bundle["signature"], phase3_bundle["signature"])
            )
            == 0,
            "initial_policy_state_exact_match": float(initial_param_diff["policy_param_max_abs_diff"]) == 0.0,
            "strict_native_first_hidden_exact_match": bool(reduction["first_hidden_closed_exactly"]),
            "strict_native_rollout_tensors_exact_match": bool(reduction["rollout_tensors_closed_exactly"]),
            "strict_native_gradient_exact_match": bool(reduction["gradient_closed_exactly"]),
        },
        "recommendation": {
            "next_focus": "guard_or_bypass_official_eval_shortcut_only_if_native_closes_contract",
            "reason": (
                "This probe isolates whether the remaining Phase2-vs-current one-env drift is caused by "
                "the batch=1 no-grad official eval shortcut. It does not test or modify Phase3 objective semantics."
            ),
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_path", type=str, nargs="?", default=DEFAULT_CHECKPOINT_PATH)
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
    parser.add_argument("--milestone-compare-path", type=str, default=DEFAULT_MILESTONE_COMPARE_JSON)
    parser.add_argument("--core-step-path", type=str, default=DEFAULT_CORE_STEP_JSON)
    parser.add_argument("--output-json", type=str, default=DEFAULT_OUTPUT_JSON)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = run_phase3_phase2_official_eval_vs_native_contract_probe(
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
        milestone_compare_path=args.milestone_compare_path,
        core_step_path=args.core_step_path,
    )
    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
