#!/usr/bin/env python
"""Offline mechanism audit for Gym-near prior mismatch.

This exploratory report is deliberately outside the exact-SCM/default prior
path.  It does not sample environments and does not train PPO.  It separates
surface mismatches from mechanism-level evidence so that obvious, stable
distribution bugs can be repaired without shrinking diversity along unrelated
dimensions.
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
DEFAULT_TRIAGE_REPORT = (
    "/home/chen/RLPFN/artifacts/"
    "phase2_exploratory_obvious_mismatch_triage_0506/"
    "obvious_mismatch_triage_report.json"
)
DEFAULT_GENERALIZATION_REPORT = (
    "/home/chen/RLPFN/artifacts/"
    "phase2_exploratory_fitmodel_generalization_mismatch_argument_0506/"
    "fitmodel_generalization_mismatch_argument_report.json"
)
DEFAULT_OUTPUT_DIR = (
    "/home/chen/RLPFN/artifacts/"
    "phase2_exploratory_distribution_mismatch_mechanism_audit_0506"
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


def _linear_r2(xs: list[float], ys: list[float]) -> float | None:
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3 or float(np.var(x)) <= 0.0 or float(np.var(y)) <= 0.0:
        return None
    coeff = np.polyfit(x, y, deg=1)
    pred = coeff[0] * x + coeff[1]
    ss_res = float(np.sum(np.square(y - pred)))
    ss_tot = float(np.sum(np.square(y - np.mean(y))))
    return float(1.0 - ss_res / max(ss_tot, 1e-12))


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


def _row_features(row: dict[str, str]) -> dict[str, float]:
    state_dim = max(0.0, _as_float(row, "state_dim", 0.0))
    action_dim = max(0.0, _as_float(row, "action_dim", 0.0))
    noise_dim = max(0.0, _as_float(row, "noise_dim", 0.0))
    denom = max(state_dim + action_dim + noise_dim, 1e-12)
    return {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "noise_dim": noise_dim,
        "expected_state_share": state_dim / denom,
        "expected_action_share": action_dim / denom,
        "expected_noise_share": noise_dim / denom,
        "dim_state_action_ratio": state_dim / max(action_dim, 1e-12),
        "dim_noise_action_ratio": noise_dim / max(action_dim, 1e-12),
    }


def _summarize_rule(rows: list[dict[str, str]], args: argparse.Namespace) -> dict[str, Any]:
    original_pass = []
    balanced_pass = []
    original_reasons: Counter[str] = Counter()
    balanced_reasons: Counter[str] = Counter()
    original_combos: Counter[str] = Counter()
    balanced_combos: Counter[str] = Counter()
    expected_action_share = []
    dim_state_action_ratio = []
    action_gain = []
    ratio = []
    gain_residual_to_width = []
    fixed_rows: list[dict[str, str]] = []
    still_fail_rows: list[dict[str, str]] = []

    for row in rows:
        state_gain = _as_float(row, "reward_state_input_gain_fraction")
        action_gain_v = _as_float(row, "reward_action_input_gain_fraction")
        noise_gain = _as_float(row, "reward_noise_input_gain_fraction")
        ratio_v = _as_float(row, "reward_state_to_action_gain_ratio")
        reasons = _topology_reasons(
            state_gain=state_gain,
            action_gain=action_gain_v,
            ratio=ratio_v,
            state_gain_min=float(args.state_gain_min),
            action_gain_min=float(args.action_gain_min),
            ratio_max=float(args.state_to_action_ratio_max),
        )
        orig_pass = (not reasons) and _as_bool(row.get("topology_pass", True))
        original_pass.append(orig_pass)
        original_reasons.update(reasons)
        original_combos.update([";".join(reasons) if reasons else "pass"])

        balanced = balanced_gain_summary_from_raw(
            state_gain=state_gain,
            action_gain=action_gain_v,
            noise_gain=noise_gain,
            state_scale=float(args.state_scale),
            action_scale=float(args.action_scale),
            noise_scale=float(args.noise_scale),
        )
        b_reasons = _topology_reasons(
            state_gain=float(balanced["reward_state_input_gain_fraction"]),
            action_gain=float(balanced["reward_action_input_gain_fraction"]),
            ratio=float(balanced["reward_state_to_action_gain_ratio"]),
            state_gain_min=float(args.state_gain_min),
            action_gain_min=float(args.action_gain_min),
            ratio_max=float(args.state_to_action_ratio_max),
        )
        b_pass = not b_reasons
        balanced_pass.append(b_pass)
        balanced_reasons.update(b_reasons)
        balanced_combos.update([";".join(b_reasons) if b_reasons else "pass"])
        if (not orig_pass) and b_pass:
            fixed_rows.append(row)
        if (not orig_pass) and (not b_pass):
            still_fail_rows.append(row)

        f = _row_features(row)
        expected_action_share.append(float(f["expected_action_share"]))
        dim_state_action_ratio.append(float(f["dim_state_action_ratio"]))
        action_gain.append(action_gain_v)
        ratio.append(ratio_v)
        gain_residual_to_width.append(action_gain_v - float(f["expected_action_share"]))

    n = len(rows)
    orig_pass_n = int(sum(original_pass))
    fixed_n = int(len(fixed_rows))
    still_fail_n = int(len(still_fail_rows))
    gated_n = int(orig_pass_n + fixed_n)
    mechanism_stability = {
        "action_gain_width_corr": _pearson(action_gain, expected_action_share),
        "action_gain_width_r2_linear": _linear_r2(expected_action_share, action_gain),
        "ratio_dim_ratio_corr": _pearson(ratio, dim_state_action_ratio),
        "ratio_dim_ratio_r2_linear": _linear_r2(dim_state_action_ratio, ratio),
        "residual_action_gain_minus_expected_action_share": _stats(gain_residual_to_width),
    }
    return {
        "n": int(n),
        "original_pass_n": orig_pass_n,
        "original_pass_rate": float(orig_pass_n / max(n, 1)),
        "original_fail_n": int(n - orig_pass_n),
        "original_fail_fixed_by_balance_n": fixed_n,
        "original_fail_fixed_by_balance_fraction": float(fixed_n / max(n - orig_pass_n, 1)),
        "balanced_still_fail_n": still_fail_n,
        "gated_pass_n": gated_n,
        "gated_pass_rate": float(gated_n / max(n, 1)),
        "original_failure_reason_counts": dict(original_reasons),
        "balanced_failure_reason_counts": dict(balanced_reasons),
        "original_failure_combo_counts": dict(original_combos),
        "balanced_failure_combo_counts": dict(balanced_combos),
        "mechanism_stability": mechanism_stability,
        "fixed_by_balance_dim_stats": {
            "expected_action_share": _stats([_row_features(row)["expected_action_share"] for row in fixed_rows]),
            "dim_state_action_ratio": _stats([_row_features(row)["dim_state_action_ratio"] for row in fixed_rows]),
            "action_dim": _stats([_row_features(row)["action_dim"] for row in fixed_rows]),
            "state_dim": _stats([_row_features(row)["state_dim"] for row in fixed_rows]),
            "noise_dim": _stats([_row_features(row)["noise_dim"] for row in fixed_rows]),
        },
        "still_fail_after_balance_dim_stats": {
            "expected_action_share": _stats([_row_features(row)["expected_action_share"] for row in still_fail_rows]),
            "dim_state_action_ratio": _stats([_row_features(row)["dim_state_action_ratio"] for row in still_fail_rows]),
            "action_dim": _stats([_row_features(row)["action_dim"] for row in still_fail_rows]),
            "state_dim": _stats([_row_features(row)["state_dim"] for row in still_fail_rows]),
            "noise_dim": _stats([_row_features(row)["noise_dim"] for row in still_fail_rows]),
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows_by_rule: dict[str, list[dict[str, str]]] = {}
    with Path(args.failure_audit_csv).expanduser().resolve().open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows_by_rule.setdefault(str(row.get("rule", "")), []).append(row)

    triage = _load_json(args.triage_report)
    generalization = _load_json(args.generalization_report)
    summary_by_rule = {
        rule: _summarize_rule(rows, args)
        for rule, rows in sorted(rows_by_rule.items())
    }

    findings = []
    for rule, data in summary_by_rule.items():
        corr = (data.get("mechanism_stability") or {}).get("action_gain_width_corr")
        fixed_fraction = data.get("original_fail_fixed_by_balance_fraction")
        if corr is not None and corr >= 0.98 and fixed_fraction >= 0.35:
            findings.append(
                {
                    "rule": rule,
                    "finding": "reward_path_width_bias_is_stable_mechanism",
                    "certainty": "high",
                    "evidence": {
                        "action_gain_width_corr": corr,
                        "fixed_fraction": fixed_fraction,
                        "original_pass_rate": data.get("original_pass_rate"),
                        "gated_pass_rate": data.get("gated_pass_rate"),
                    },
                    "repair_scope": (
                        "Use reward-path-only gated balancing.  Keep original healthy candidates, "
                        "balance only topology failures explained by low action gain/high state-action ratio, "
                        "and reject still-failed candidates."
                    ),
                }
            )
    findings.extend(
        [
            {
                "finding": "action_obs_state_dimensions_are_diagnostic_coordinates_not_repair_knobs",
                "certainty": "high",
                "reason": (
                    "The actionable mechanism is width-biased reward sensitivity allocation inside the reward path. "
                    "Changing action_dim/obs_dim/state_dim directly would shrink support and can damage diversity."
                ),
            },
            {
                "finding": "done_rate_horizon_shift_remains_surface_mismatch",
                "certainty": "medium",
                "reason": (
                    "Existing logs show a done-rate gap, but the mechanism could be termination thresholds, dynamics, "
                    "rollout horizon, or selected topology.  It should be guarded and measured after reward-path repair, "
                    "not repaired blindly."
                ),
            },
            {
                "finding": "survival_reward_absence_is_real_mismatch_but_not_yet_primary_repair",
                "certainty": "medium",
                "reason": (
                    "Gym validation contains survival-style tasks while current prior fixed lists have no survival component. "
                    "However, survival improves return only when termination is action/state sensitive; otherwise it is mostly "
                    "a baseline-like constant.  Add only as a later exploratory subset test with component logging."
                ),
            },
        ]
    )

    report = {
        "analysis_entry": "phase2_distribution_mismatch_mechanism_audit",
        "exploratory_only": True,
        "contract": {
            "does_not_modify_exact_scm": True,
            "does_not_modify_default_prior_generator": True,
            "does_not_sample_environments": True,
            "does_not_run_ppo_training": True,
            "separates_surface_features_from_repair_mechanisms": True,
            "dimension_features_are_filter_coordinates_not_generator_knobs": True,
        },
        "input_artifacts": {
            "failure_audit_csv": str(Path(args.failure_audit_csv).expanduser().resolve()),
            "triage_report": str(Path(args.triage_report).expanduser().resolve()),
            "generalization_report": str(Path(args.generalization_report).expanduser().resolve()),
        },
        "balance_profile": {
            "state_scale": float(args.state_scale),
            "action_scale": float(args.action_scale),
            "noise_scale": float(args.noise_scale),
        },
        "summary_by_rule": summary_by_rule,
        "supporting_context": {
            "triage_top_finding": (triage.get("findings") or [{}])[0] if triage else {},
            "generalization_claim_assessment": generalization.get("claim_assessment", {}) if generalization else {},
        },
        "findings": findings,
        "decision_guidance": {
            "safe_primary_repair_now": "gated_reward_path_balancing",
            "do_not_repair_by": ["action_dim_sampler", "obs_dim_sampler", "state_dim_sampler"],
            "guards_before_milestone": [
                "already healthy candidates remain unbalanced",
                "balanced candidates pass the same topology health criteria",
                "selected fixed list keeps reward/state/done diversity non-collapsed",
                "short fixed-list PPO health is not worse than current milestone",
            ],
            "survival_next_step": (
                "Only after the primary repair is stable, run an exploratory fixed-list survival subset probe "
                "with raw return, episode length, reward_env, survival component, value target scale, and advantage scale."
            ),
        },
    }

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "distribution_mismatch_mechanism_audit_report.json"
    report_path.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Distribution Mismatch Mechanism Audit",
        "",
        "Exploratory only. No exact-SCM/default prior mutation, no sampling, no PPO training.",
        "",
        "## Main Findings",
    ]
    for item in findings:
        lines.append(f"- {item['finding']}: {item.get('certainty', 'na')}")
    lines.extend(["", "## Rule Summary"])
    for rule, data in summary_by_rule.items():
        mech = data["mechanism_stability"]
        lines.append(
            "- "
            f"{rule}: original_pass={data['original_pass_rate']:.6f}, "
            f"gated_pass={data['gated_pass_rate']:.6f}, "
            f"fixed_fraction={data['original_fail_fixed_by_balance_fraction']:.6f}, "
            f"action_gain_width_corr={mech['action_gain_width_corr']:.6f}"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "Primary repair candidate: gated reward-path balancing. Keep original healthy candidates unchanged; balance only failed candidates; reject still-failed candidates.",
            "Survival is a real mismatch but should remain a later exploratory subset test, because its advantage usefulness depends on action-sensitive termination.",
        ]
    )
    md_path = out_dir / "distribution_mismatch_mechanism_audit_summary.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failure-audit-csv", type=str, default=DEFAULT_FAILURE_AUDIT_CSV)
    parser.add_argument("--triage-report", type=str, default=DEFAULT_TRIAGE_REPORT)
    parser.add_argument("--generalization-report", type=str, default=DEFAULT_GENERALIZATION_REPORT)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--state-scale", type=float, default=3.0)
    parser.add_argument("--action-scale", type=float, default=16.0)
    parser.add_argument("--noise-scale", type=float, default=0.125)
    parser.add_argument("--state-gain-min", type=float, default=0.7)
    parser.add_argument("--action-gain-min", type=float, default=0.06)
    parser.add_argument("--state-to-action-ratio-max", type=float, default=10.0)
    args = parser.parse_args()
    report = run(args)
    print(
        json.dumps(
            _json_safe(
                {
                    "output_dir": str(Path(args.output_dir).expanduser().resolve()),
                    "summary": {
                        rule: {
                            "original_pass_rate": data["original_pass_rate"],
                            "gated_pass_rate": data["gated_pass_rate"],
                            "fixed_fraction": data["original_fail_fixed_by_balance_fraction"],
                            "action_gain_width_corr": data["mechanism_stability"]["action_gain_width_corr"],
                        }
                        for rule, data in report["summary_by_rule"].items()
                    },
                }
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
