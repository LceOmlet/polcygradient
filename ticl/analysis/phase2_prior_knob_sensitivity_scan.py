import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from ticl.analysis.phase2_alpha_temporal_control_probe import measure_alpha_temporal_control
from ticl.analysis.phase2_suffix_state_identity_probe import run_phase2_suffix_state_identity_probe


DEFAULT_SCREENING_CSV = (
    "/home/chen/RLPFN/artifacts/"
    "phase2_prior_gym_control_feature_audit_gain07_all_openloop_0428/"
    "screening_rows.csv"
)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return None if (math.isnan(value) or math.isinf(value)) else value
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _read_csv_rows(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _stats(values: list[Any]) -> dict[str, Any]:
    arr = np.asarray([v for v in (_finite_float(v) for v in values) if v is not None], dtype=np.float64)
    if arr.size == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "q10": None,
            "q50": None,
            "q90": None,
            "min": None,
            "max": None,
        }
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "q10": float(np.quantile(arr, 0.10)),
        "q50": float(np.quantile(arr, 0.50)),
        "q90": float(np.quantile(arr, 0.90)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _parse_scalar(text: str) -> Any:
    token = str(text).strip()
    lower = token.lower()
    if lower in {"true", "yes", "y"}:
        return True
    if lower in {"false", "no", "n"}:
        return False
    if lower in {"none", "null"}:
        return None
    try:
        if any(ch in token for ch in (".", "e", "E")):
            return float(token)
        return int(token)
    except ValueError:
        return token


def _parse_case(text: str) -> dict[str, Any]:
    raw = str(text).strip()
    if raw in {"", "baseline", "none"}:
        return {"name": "baseline", "overrides": {}}
    name = None
    body = raw
    if ":" in raw:
        name, body = raw.split(":", 1)
        name = name.strip()
    overrides = {}
    for piece in body.split(","):
        item = piece.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"case item must be key=value, got {item!r} in {text!r}")
        key, value = item.split("=", 1)
        overrides[key.strip()] = _parse_scalar(value)
    if name is None:
        name = "_".join(f"{k}{str(v).replace('.', 'p')}" for k, v in sorted(overrides.items()))
    return {"name": str(name), "overrides": overrides}


def _parse_cases(raw: str) -> list[dict[str, Any]]:
    cases = [_parse_case(piece) for piece in str(raw).split(";") if piece.strip()]
    if not cases:
        cases = [{"name": "baseline", "overrides": {}}]
    names = [case["name"] for case in cases]
    if len(set(names)) != len(names):
        raise ValueError(f"case names must be unique, got {names}")
    return cases


def _select_stratified(rows: list[dict[str, Any]], *, count: int, seed: int, key: str) -> list[dict[str, Any]]:
    valid = [row for row in rows if _finite_float(row.get(key)) is not None]
    valid.sort(key=lambda row: float(row[key]))
    if int(count) <= 0 or int(count) >= len(valid):
        return list(valid)
    rng = np.random.default_rng(int(seed))
    bucket_count = min(4, int(count), len(valid))
    buckets = np.array_split(np.asarray(valid, dtype=object), bucket_count)
    selected = []
    remaining = int(count)
    for bucket_idx, bucket in enumerate(buckets):
        take = max(1, int(round(int(count) / bucket_count)))
        if bucket_idx == bucket_count - 1:
            take = remaining
        take = min(take, len(bucket))
        remaining -= take
        indices = sorted(rng.choice(len(bucket), size=take, replace=False).tolist())
        selected.extend([bucket[int(i)] for i in indices])
    return selected[: int(count)]


def _identity_curve(report: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for item in report.get("horizon_diagnostics") or []:
        h = int(item["horizon_step"])
        out[f"G{h}_bias_corrected_identity"] = float(item["bias_corrected_identity_score"])
        out[f"G{h}_signal_to_noise"] = float(item["signal_to_noise"])
    return out


def _run_one(
    *,
    source_row: dict[str, Any],
    case: dict[str, Any],
    device: str,
    control_rollout_count: int,
    control_n_steps: int,
    action_delta: float,
    identity_reference_state_count: int,
    identity_continuation_repeats: int,
    identity_n_steps: int,
    identity_anchor_step: int,
) -> dict[str, Any]:
    frozen_path = str(source_row["full_frozen_h_json"])
    overrides = dict(case["overrides"])
    control = measure_alpha_temporal_control(
        checkpoint_path=None,
        device=str(device),
        from_scratch=True,
        from_scratch_model_type="rlpfn",
        build_seed=4040,
        train_env_seed=2020,
        single_eval_pos=64,
        n_steps=int(control_n_steps),
        rollout_count=int(control_rollout_count),
        rollout_seed_start=800000,
        sampled_policy=True,
        fixed_frozen_h_json=frozen_path,
        reference_state_inertia_enabled_override=None,
        terminal_reset_enabled_override=None,
        terminal_reset_count_target_override=None,
        action_delta=float(action_delta),
        behavior_policy="unit_gaussian_iid",
        frozen_h_overrides=overrides,
    )
    identity = run_phase2_suffix_state_identity_probe(
        checkpoint_path=None,
        device=str(device),
        from_scratch=True,
        from_scratch_model_type="rlpfn",
        fixed_frozen_h_json=frozen_path,
        frozen_h_seed=int(float(source_row["frozen_h_seed"])),
        train_env_seed=2020,
        train_rollout_seed=4040,
        single_eval_pos=64,
        anchor_step=int(identity_anchor_step),
        n_steps=int(identity_n_steps),
        batch_size=max(256, int(identity_reference_state_count) * int(identity_continuation_repeats)),
        behavior_policy="unit_gaussian",
        reference_state_count=int(identity_reference_state_count),
        continuation_repeats=int(identity_continuation_repeats),
        reference_rollout_seed_start=5000,
        continuation_rollout_seed_start=10000,
        sampled_policy=True,
        discount_gamma=0.99,
        horizon_diagnostics_steps="2,3,4,5",
        frozen_h_overrides=overrides,
        print_progress_every_repeat=False,
    )
    row: dict[str, Any] = {
        "case": str(case["name"]),
        "overrides": overrides,
        "frozen_h_seed": int(float(source_row["frozen_h_seed"])),
        "full_frozen_h_json": frozen_path,
        "source_topology_reward_state_input_gain_fraction": _finite_float(
            source_row.get("topology_reward_state_input_gain_fraction")
        ),
        "source_anchor_obs_value_std": _finite_float(source_row.get("anchor_obs_value_std")),
        "source_anchor_reward_std": _finite_float(source_row.get("anchor_reward_std")),
        "source_state_action_sensitivity_to_step1_drift_ratio": _finite_float(
            source_row.get("state_action_sensitivity_to_step1_drift_ratio")
        ),
        "source_reward_action_sensitivity_per_unit": _finite_float(
            source_row.get("mean_abs_reward_effective_t1_action_sensitivity_per_action_unit")
        ),
        "mean_l2_s_t1_minus_s_t": control.get("mean_l2_s_t1_minus_s_t"),
        "mean_l2_s_t1_action_sensitivity_per_action_unit": control.get(
            "mean_l2_s_t1_action_sensitivity_per_action_unit"
        ),
        "state_action_sensitivity_to_step1_drift_ratio": control.get("state_action_sensitivity_to_step1_drift_ratio"),
        "mean_abs_reward_effective_t1_action_sensitivity_per_action_unit": control.get(
            "mean_abs_reward_effective_t1_action_sensitivity_per_action_unit"
        ),
        "discounted_effective_return_std": (control.get("discounted_effective_return") or {}).get("std"),
        "sampled_env_noise_std": (control.get("sampled_env_snapshot") or {}).get("noise_std"),
        "sampled_env_init_std": (control.get("sampled_env_snapshot") or {}).get("init_std"),
        "identity_score": (identity.get("aggregate") or {}).get("identity_score"),
    }
    row.update(_identity_curve(identity))
    return _json_safe(row)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [
        "mean_l2_s_t1_minus_s_t",
        "mean_l2_s_t1_action_sensitivity_per_action_unit",
        "state_action_sensitivity_to_step1_drift_ratio",
        "mean_abs_reward_effective_t1_action_sensitivity_per_action_unit",
        "discounted_effective_return_std",
        "identity_score",
        "G2_bias_corrected_identity",
        "G3_bias_corrected_identity",
        "G4_bias_corrected_identity",
        "G5_bias_corrected_identity",
    ]
    case_names = sorted({str(row["case"]) for row in rows})
    by_case = {}
    for case in case_names:
        case_rows = [row for row in rows if str(row["case"]) == case]
        by_case[case] = {
            "row_count": int(len(case_rows)),
            "stats": {key: _stats([row.get(key) for row in case_rows]) for key in keys},
        }
    baseline = by_case.get("baseline")
    if baseline is not None:
        for case, payload in by_case.items():
            if case == "baseline":
                continue
            deltas = {}
            for key in keys:
                case_mean = _finite_float((payload["stats"].get(key) or {}).get("mean"))
                base_mean = _finite_float((baseline["stats"].get(key) or {}).get("mean"))
                deltas[f"{key}_mean_delta_vs_baseline"] = (
                    None if case_mean is None or base_mean is None else float(case_mean - base_mean)
                )
            payload["mean_deltas_vs_baseline"] = deltas
    return {"case_summary": by_case}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Paired prior frozen_h knob scan for Gym-alignment features.")
    parser.add_argument("--screening-csv", type=str, default=DEFAULT_SCREENING_CSV)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--cases", type=str, default="baseline;noise_std_0p03:noise_std=0.03;noise_std_0p05:noise_std=0.05")
    parser.add_argument("--env-count", type=int, default=12)
    parser.add_argument("--sample-seed", type=int, default=20260428)
    parser.add_argument("--stratify-key", type=str, default="state_action_sensitivity_to_step1_drift_ratio")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--control-rollout-count", type=int, default=64)
    parser.add_argument("--control-n-steps", type=int, default=32)
    parser.add_argument("--action-delta", type=float, default=0.1)
    parser.add_argument("--identity-reference-state-count", type=int, default=16)
    parser.add_argument("--identity-continuation-repeats", type=int, default=16)
    parser.add_argument("--identity-anchor-step", type=int, default=128)
    parser.add_argument("--identity-n-steps", type=int, default=133)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    rows = _read_csv_rows(args.screening_csv)
    selected = _select_stratified(
        rows,
        count=int(args.env_count),
        seed=int(args.sample_seed),
        key=str(args.stratify_key),
    )
    cases = _parse_cases(args.cases)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "rows.jsonl"
    out_rows = []
    with rows_path.open("w", encoding="utf-8") as handle:
        for source_idx, source_row in enumerate(selected, start=1):
            for case_idx, case in enumerate(cases, start=1):
                print(
                    f"[prior-knob-scan] env {source_idx}/{len(selected)} "
                    f"case {case_idx}/{len(cases)} seed={source_row.get('frozen_h_seed')} case={case['name']}",
                    flush=True,
                )
                out = _run_one(
                    source_row=source_row,
                    case=case,
                    device=str(args.device),
                    control_rollout_count=int(args.control_rollout_count),
                    control_n_steps=int(args.control_n_steps),
                    action_delta=float(args.action_delta),
                    identity_reference_state_count=int(args.identity_reference_state_count),
                    identity_continuation_repeats=int(args.identity_continuation_repeats),
                    identity_n_steps=int(args.identity_n_steps),
                    identity_anchor_step=int(args.identity_anchor_step),
                )
                out_rows.append(out)
                handle.write(json.dumps(_json_safe(out), sort_keys=True) + "\n")
                handle.flush()
    summary = {
        "audit_entry": "phase2_prior_knob_sensitivity_scan",
        "screening_csv": str(Path(args.screening_csv).expanduser().resolve()),
        "selected_frozen_h_seeds": [int(float(row["frozen_h_seed"])) for row in selected],
        "cases": cases,
        "settings": {
            "env_count": int(args.env_count),
            "stratify_key": str(args.stratify_key),
            "device": str(args.device),
            "control_rollout_count": int(args.control_rollout_count),
            "control_n_steps": int(args.control_n_steps),
            "identity_reference_state_count": int(args.identity_reference_state_count),
            "identity_continuation_repeats": int(args.identity_continuation_repeats),
            "identity_anchor_step": int(args.identity_anchor_step),
            "identity_n_steps": int(args.identity_n_steps),
        },
        **summarize(out_rows),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(_json_safe(summary), sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(_json_safe(summary), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
