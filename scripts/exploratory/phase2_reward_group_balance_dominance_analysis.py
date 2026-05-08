#!/usr/bin/env python
"""Dominance analysis for exploratory reward-path group balancing.

This script is intentionally outside the exact-SCM implementation path.  It
does not sample environments and does not train PPO.  It only re-reads the
Gym-near topology audit/probe artifacts and asks whether the observed
topology failures are dominated by the reward-path width-bias pattern that
the reward-group balancing wrapper targets.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from phase2_reward_group_balance_runtime import balanced_gain_summary_from_raw


DEFAULT_FAILURE_AUDIT_CSV = (
    "/home/chen/RLPFN/artifacts/"
    "phase2_exploratory_gym_near_topology_failure_audit_n8192_0506/"
    "gym_near_topology_failure_audit.csv"
)
DEFAULT_FAILURE_AUDIT_REPORT = (
    "/home/chen/RLPFN/artifacts/"
    "phase2_exploratory_gym_near_topology_failure_audit_n8192_0506/"
    "failure_audit_report.json"
)
DEFAULT_ROOT_CAUSE = (
    "/home/chen/RLPFN/artifacts/"
    "phase2_exploratory_gym_near_topology_failure_audit_n8192_0506/"
    "root_cause_dim_gain_analysis.json"
)
DEFAULT_DIM_NORMALIZED = (
    "/home/chen/RLPFN/artifacts/"
    "phase2_exploratory_gym_near_topology_failure_audit_n8192_0506/"
    "dim_normalized_gain_analysis.json"
)
DEFAULT_BALANCE_PROBE_REPORT = (
    "/home/chen/RLPFN/artifacts/"
    "phase2_exploratory_reward_group_balance_probe_s3_a16_n0125_n64_0506/"
    "reward_group_balance_probe_report.json"
)
DEFAULT_Q90_PROGRESS = (
    "/home/chen/RLPFN/artifacts/"
    "phase2_exploratory_reward_group_balance_q90_fixedlist_n64_s512_u20_e4_advnorm_0506/"
    "prior_progress.jsonl"
)
DEFAULT_FULL_PROGRESS = (
    "/home/chen/RLPFN/artifacts/"
    "phase2_exploratory_reward_group_balance_full_fixedlist_n64_s512_u4_e4_advnorm_0506/"
    "prior_progress.jsonl"
)
DEFAULT_OUTPUT_DIR = (
    "/home/chen/RLPFN/artifacts/"
    "phase2_exploratory_reward_group_balance_dominance_0506"
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _load_json(path: str | Path) -> dict[str, Any]:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _as_float(row: dict[str, str], key: str, default: float = float("nan")) -> float:
    try:
        value = float(row.get(key, default))
    except Exception:
        value = float(default)
    return value if math.isfinite(value) else float(default)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _topology_reasons(
    *,
    state_gain: float,
    action_gain: float,
    ratio: float,
    state_gain_min: float,
    action_gain_min: float,
    ratio_max: float,
) -> list[str]:
    reasons: list[str] = []
    if (not math.isfinite(state_gain)) or state_gain < state_gain_min:
        reasons.append("state_gain_low")
    if (not math.isfinite(action_gain)) or action_gain < action_gain_min:
        reasons.append("action_gain_low")
    if (not math.isfinite(ratio)) or ratio > ratio_max:
        reasons.append("state_action_ratio_high")
    return reasons


def _stats(values: list[float]) -> dict[str, Any]:
    arr = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=np.float64)
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "q10": float(np.quantile(arr, 0.10)),
        "q25": float(np.quantile(arr, 0.25)),
        "median": float(np.quantile(arr, 0.50)),
        "q75": float(np.quantile(arr, 0.75)),
        "q90": float(np.quantile(arr, 0.90)),
        "max": float(np.max(arr)),
    }


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2 or float(np.std(x)) <= 0.0 or float(np.std(y)) <= 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _summarize_progress(path: str | Path) -> dict[str, Any]:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        return {"path": str(p), "exists": False}
    rows: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            rows.append(row)
    out: dict[str, Any] = {"path": str(p), "exists": True, "updates": int(len(rows))}
    if not rows:
        return out
    first = rows[0]
    last = rows[-1]
    first_stats = (((first.get("pack_summary") or {}).get("stats") or {}))
    last_stats = (((last.get("pack_summary") or {}).get("stats") or {}))
    for key in ("raw_reward_return_mean", "reward_return_mean", "done_count"):
        first_v = first_stats.get(key, None)
        last_v = last_stats.get(key, None)
        if isinstance(first_v, (int, float)) and isinstance(last_v, (int, float)):
            out[f"{key}_first"] = float(first_v)
            out[f"{key}_last"] = float(last_v)
            out[f"{key}_delta"] = float(last_v - first_v)
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    audit_report = _load_json(args.failure_audit_report)
    root_cause = _load_json(args.root_cause_json)
    dim_normalized = _load_json(args.dim_normalized_json)
    balance_probe = _load_json(args.balance_probe_report)

    thresholds = {
        "state_gain_min": float(args.state_gain_min),
        "action_gain_min": float(args.action_gain_min),
        "state_to_action_ratio_max": float(args.state_to_action_ratio_max),
    }
    if audit_report:
        health = ((audit_report.get("contract") or {}).get("health_criteria") or {})
        thresholds = {
            "state_gain_min": float(health.get("state_gain_min", thresholds["state_gain_min"])),
            "action_gain_min": float(health.get("action_gain_min", thresholds["action_gain_min"])),
            "state_to_action_ratio_max": float(
                health.get("state_to_action_ratio_max", thresholds["state_to_action_ratio_max"])
            ),
        }

    rows_by_rule: dict[str, list[dict[str, str]]] = {}
    with Path(args.failure_audit_csv).expanduser().resolve().open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows_by_rule.setdefault(str(row.get("rule", "")), []).append(row)

    summary_by_rule: dict[str, Any] = {}
    for rule, rows in sorted(rows_by_rule.items()):
        original_pass = []
        balanced_pass = []
        original_reason_counter: Counter[str] = Counter()
        balanced_reason_counter: Counter[str] = Counter()
        original_combo_counter: Counter[str] = Counter()
        balanced_combo_counter: Counter[str] = Counter()
        fixed_fail_indices = []
        still_fail_indices = []
        balanced_values = {
            "state_gain": [],
            "action_gain": [],
            "noise_gain": [],
            "state_to_action_ratio": [],
        }
        dim_state_action_ratio = []
        dim_noise_action_ratio = []
        expected_action_share = []
        actual_action_gain = []
        actual_ratio = []

        for idx, row in enumerate(rows):
            state_gain = _as_float(row, "reward_state_input_gain_fraction")
            action_gain = _as_float(row, "reward_action_input_gain_fraction")
            noise_gain = _as_float(row, "reward_noise_input_gain_fraction")
            ratio = _as_float(row, "reward_state_to_action_gain_ratio")
            orig_reasons = _topology_reasons(
                state_gain=state_gain,
                action_gain=action_gain,
                ratio=ratio,
                state_gain_min=thresholds["state_gain_min"],
                action_gain_min=thresholds["action_gain_min"],
                ratio_max=thresholds["state_to_action_ratio_max"],
            )
            orig_pass = (not orig_reasons) and _as_bool(row.get("topology_pass", True))
            original_pass.append(orig_pass)
            original_reason_counter.update(orig_reasons)
            original_combo_counter.update([";".join(orig_reasons) if orig_reasons else "pass"])

            balanced = balanced_gain_summary_from_raw(
                state_gain=state_gain,
                action_gain=action_gain,
                noise_gain=noise_gain,
                state_scale=float(args.state_scale),
                action_scale=float(args.action_scale),
                noise_scale=float(args.noise_scale),
            )
            b_state = float(balanced["reward_state_input_gain_fraction"])
            b_action = float(balanced["reward_action_input_gain_fraction"])
            b_noise = float(balanced["reward_noise_input_gain_fraction"])
            b_ratio = float(balanced["reward_state_to_action_gain_ratio"])
            b_reasons = _topology_reasons(
                state_gain=b_state,
                action_gain=b_action,
                ratio=b_ratio,
                state_gain_min=thresholds["state_gain_min"],
                action_gain_min=thresholds["action_gain_min"],
                ratio_max=thresholds["state_to_action_ratio_max"],
            )
            b_pass = not b_reasons
            balanced_pass.append(b_pass)
            balanced_reason_counter.update(b_reasons)
            balanced_combo_counter.update([";".join(b_reasons) if b_reasons else "pass"])
            if (not orig_pass) and b_pass:
                fixed_fail_indices.append(idx)
            if (not orig_pass) and (not b_pass):
                still_fail_indices.append(idx)

            balanced_values["state_gain"].append(b_state)
            balanced_values["action_gain"].append(b_action)
            balanced_values["noise_gain"].append(b_noise)
            balanced_values["state_to_action_ratio"].append(b_ratio)

            state_dim = max(0.0, _as_float(row, "state_dim", 0.0))
            action_dim = max(0.0, _as_float(row, "action_dim", 0.0))
            noise_dim = max(0.0, _as_float(row, "noise_dim", 0.0))
            denom = max(state_dim + action_dim + noise_dim, 1e-12)
            expected_action_share.append(action_dim / denom)
            dim_state_action_ratio.append(state_dim / max(action_dim, 1e-12))
            dim_noise_action_ratio.append(noise_dim / max(action_dim, 1e-12))
            actual_action_gain.append(action_gain)
            actual_ratio.append(ratio)

        n = len(rows)
        orig_pass_n = int(sum(bool(v) for v in original_pass))
        balanced_pass_n = int(sum(bool(v) for v in balanced_pass))
        original_fail_n = int(n - orig_pass_n)
        fixed_fail_n = int(len(fixed_fail_indices))
        still_fail_n = int(len(still_fail_indices))
        gated_pass_n = int(orig_pass_n + fixed_fail_n)
        pass_preserved_n = int(sum(bool(o) and bool(b) for o, b in zip(original_pass, balanced_pass)))
        original_pass_lost_n = int(sum(bool(o) and (not bool(b)) for o, b in zip(original_pass, balanced_pass)))

        summary_by_rule[rule] = {
            "n": int(n),
            "original_pass_n": orig_pass_n,
            "original_pass_rate": float(orig_pass_n / max(n, 1)),
            "balanced_pass_n": balanced_pass_n,
            "balanced_pass_rate": float(balanced_pass_n / max(n, 1)),
            "pass_rate_lift_x": (
                float(balanced_pass_n / orig_pass_n)
                if orig_pass_n > 0
                else None
            ),
            "gated_original_keep_plus_balance_repair_pass_n": gated_pass_n,
            "gated_original_keep_plus_balance_repair_pass_rate": float(gated_pass_n / max(n, 1)),
            "gated_pass_rate_lift_x": (
                float(gated_pass_n / orig_pass_n)
                if orig_pass_n > 0
                else None
            ),
            "original_fail_n": original_fail_n,
            "original_fail_fixed_by_balance_n": fixed_fail_n,
            "original_fail_fixed_by_balance_fraction": float(fixed_fail_n / max(original_fail_n, 1)),
            "original_fail_still_fail_n": still_fail_n,
            "original_pass_preserved_by_balance_n": pass_preserved_n,
            "original_pass_preserved_by_balance_fraction": float(pass_preserved_n / max(orig_pass_n, 1)),
            "original_pass_lost_by_balance_n": original_pass_lost_n,
            "original_failure_reason_counts": dict(original_reason_counter),
            "balanced_failure_reason_counts": dict(balanced_reason_counter),
            "original_failure_combo_counts": dict(original_combo_counter),
            "balanced_failure_combo_counts": dict(balanced_combo_counter),
            "balanced_metric_stats": {key: _stats(vals) for key, vals in balanced_values.items()},
            "width_bias_correlations_from_rows": {
                "action_gain_vs_expected_action_share": _pearson(actual_action_gain, expected_action_share),
                "ratio_vs_dim_state_action_ratio": _pearson(actual_ratio, dim_state_action_ratio),
            },
            "dim_ratio_stats": {
                "expected_action_share": _stats(expected_action_share),
                "dim_state_action_ratio": _stats(dim_state_action_ratio),
                "dim_noise_action_ratio": _stats(dim_noise_action_ratio),
            },
        }

    report = {
        "analysis_entry": "phase2_reward_group_balance_dominance_analysis",
        "exploratory_only": True,
        "contract": {
            "does_not_modify_exact_scm": True,
            "does_not_modify_default_prior_generator": True,
            "does_not_run_ppo_training": True,
            "environment_side_only": True,
            "dimension_features_are_filter_coordinates_not_generator_knobs": True,
            "topology_criteria": thresholds,
            "balance_profile": {
                "state_scale": float(args.state_scale),
                "action_scale": float(args.action_scale),
                "noise_scale": float(args.noise_scale),
            },
        },
        "input_artifacts": {
            "failure_audit_csv": str(Path(args.failure_audit_csv).expanduser().resolve()),
            "failure_audit_report": str(Path(args.failure_audit_report).expanduser().resolve()),
            "root_cause_json": str(Path(args.root_cause_json).expanduser().resolve()),
            "dim_normalized_json": str(Path(args.dim_normalized_json).expanduser().resolve()),
            "balance_probe_report": str(Path(args.balance_probe_report).expanduser().resolve()),
        },
        "summary_by_rule": summary_by_rule,
        "supporting_existing_reports": {
            "root_cause_correlations": {
                rule: (data or {}).get("correlations", {})
                for rule, data in (root_cause or {}).items()
            },
            "dim_normalized_selected": {
                rule: {
                    "all": ((data or {}).get("all") or {}),
                    "fail": ((data or {}).get("fail") or {}),
                    "pass": ((data or {}).get("pass") or {}),
                }
                for rule, data in (dim_normalized or {}).items()
            },
            "balance_probe_sampling": balance_probe.get("sampling", {}),
            "balance_probe_transition_diversity": balance_probe.get("transition_diversity_probe", {}),
            "balance_probe_selected_summary_balanced": balance_probe.get("selected_summary_by_rule_balanced", {}),
            "balance_probe_selected_summary_original": balance_probe.get("selected_summary_by_rule_original", {}),
        },
        "short_training_support_only": {
            "q90_fixedlist": _summarize_progress(args.q90_progress_jsonl),
            "full_fixedlist": _summarize_progress(args.full_progress_jsonl),
        },
        "interpretation": {
            "dominance_question_answer": (
                "The topology-health rejection is dominated by width-biased reward-path gain allocation "
                "if original failures are mostly fixed by the same group scaling profile and if action "
                "gain/ratio track dimension-derived expected shares. This report quantifies both; it does "
                "not claim that every downstream generalization issue is fixed."
            ),
            "anti_regression_selection_contract": (
                "For a candidate generator-level wrapper, do not blanket-rebalance all environments. "
                "Keep already healthy candidates original, apply reward-path balancing only to width-bias "
                "failures, and reject candidates that still fail. The gated pass-rate fields quantify this "
                "anti-regression contract."
            ),
            "remaining_nonclaims": [
                "No default exact-SCM generator replacement is implied.",
                "No long-run generalization claim is made here.",
                "Dynamics/done distribution and optimization amplitude remain separate exploratory axes.",
            ],
        },
    }
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "dominance_report.json"
    report_path.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failure-audit-csv", type=str, default=DEFAULT_FAILURE_AUDIT_CSV)
    parser.add_argument("--failure-audit-report", type=str, default=DEFAULT_FAILURE_AUDIT_REPORT)
    parser.add_argument("--root-cause-json", type=str, default=DEFAULT_ROOT_CAUSE)
    parser.add_argument("--dim-normalized-json", type=str, default=DEFAULT_DIM_NORMALIZED)
    parser.add_argument("--balance-probe-report", type=str, default=DEFAULT_BALANCE_PROBE_REPORT)
    parser.add_argument("--q90-progress-jsonl", type=str, default=DEFAULT_Q90_PROGRESS)
    parser.add_argument("--full-progress-jsonl", type=str, default=DEFAULT_FULL_PROGRESS)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--state-scale", type=float, default=3.0)
    parser.add_argument("--action-scale", type=float, default=16.0)
    parser.add_argument("--noise-scale", type=float, default=0.125)
    parser.add_argument("--state-gain-min", type=float, default=0.7)
    parser.add_argument("--action-gain-min", type=float, default=0.06)
    parser.add_argument("--state-to-action-ratio-max", type=float, default=10.0)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps(_json_safe({
        "output_dir": str(Path(args.output_dir).expanduser().resolve()),
        "summary_by_rule": {
            rule: {
                "original_pass_rate": data["original_pass_rate"],
                "balanced_pass_rate": data["balanced_pass_rate"],
                "gated_pass_rate": data["gated_original_keep_plus_balance_repair_pass_rate"],
                "original_fail_fixed_by_balance_fraction": data["original_fail_fixed_by_balance_fraction"],
            }
            for rule, data in report["summary_by_rule"].items()
        },
    }), sort_keys=True))


if __name__ == "__main__":
    main()
