#!/usr/bin/env python
"""Offline argument report for fit_model Gym-generalization mismatch.

The goal is deliberately narrower than a fit_model long run: assemble the
existing evidence for whether the suspected reward-path distribution mismatch
is an important, well-localized cause of poor Gym validation.  This script
does not sample environments, does not train PPO, and does not modify the
default prior generator or exact-SCM implementation.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_FITMODEL_PRIOR_PROGRESS = (
    "/home/chen/RLPFN/health/ppo_pack_runner/"
    "rlpfn_05_04_2026_09_00_32/prior_progress.jsonl"
)
DEFAULT_FITMODEL_LOG = "/home/chen/RLPFN/health/log/rlpfn_05_04_2026_09_00_32.log"
DEFAULT_GYM_PROGRESS = (
    "/home/chen/RLPFN/artifacts/"
    "phase2_exploratory_gym_validation_baseline_n64_s512_u40_e4_advnorm_0506/"
    "gym_progress.jsonl"
)
DEFAULT_FAILURE_AUDIT_REPORT = (
    "/home/chen/RLPFN/artifacts/"
    "phase2_exploratory_gym_near_topology_failure_audit_n8192_0506/"
    "failure_audit_report.json"
)
DEFAULT_DOMINANCE_REPORT = (
    "/home/chen/RLPFN/artifacts/"
    "phase2_exploratory_reward_group_balance_dominance_0506/"
    "dominance_report.json"
)
DEFAULT_BALANCE_PROBE_REPORT = (
    "/home/chen/RLPFN/artifacts/"
    "phase2_exploratory_reward_group_balance_probe_s3_a16_n0125_n64_0506/"
    "reward_group_balance_probe_report.json"
)
DEFAULT_OUTPUT_DIR = (
    "/home/chen/RLPFN/artifacts/"
    "phase2_exploratory_fitmodel_generalization_mismatch_argument_0506"
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
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


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path).expanduser().resolve()
    rows: list[dict[str, Any]] = []
    if not p.exists():
        return rows
    with p.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def _finite_values(values: list[Any]) -> np.ndarray:
    out = []
    for value in values:
        try:
            v = float(value)
        except Exception:
            continue
        if math.isfinite(v):
            out.append(v)
    return np.asarray(out, dtype=np.float64)


def _stats(values: list[Any]) -> dict[str, Any]:
    arr = _finite_values(values)
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "q01": float(np.quantile(arr, 0.01)),
        "q10": float(np.quantile(arr, 0.10)),
        "q25": float(np.quantile(arr, 0.25)),
        "q50": float(np.quantile(arr, 0.50)),
        "q75": float(np.quantile(arr, 0.75)),
        "q90": float(np.quantile(arr, 0.90)),
        "q99": float(np.quantile(arr, 0.99)),
        "max": float(np.max(arr)),
    }


def _collector_vectors(rows: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        collector = row.get("collector") or {}
        row_values = collector.get(key) or []
        if isinstance(row_values, list):
            values.extend(row_values)
    return values


def _pack_stat_series(rows: list[dict[str, Any]], key: str) -> list[float]:
    values = []
    for row in rows:
        value = (((row.get("pack_summary") or {}).get("stats") or {}).get(key))
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return values


def _fraction(mask: np.ndarray) -> float | None:
    if mask.size == 0:
        return None
    return float(np.mean(mask.astype(np.float64)))


def _distribution_overlap(
    *,
    action_dims: list[Any],
    obs_dims: list[Any],
    gym_feature_range: dict[str, Any],
) -> dict[str, Any]:
    a = _finite_values(action_dims)
    o = _finite_values(obs_dims)
    n = min(int(a.size), int(o.size))
    if n <= 0:
        return {"n": 0}
    a = a[:n]
    o = o[:n]
    action_min = float(gym_feature_range.get("action_min", 1))
    action_max = float(gym_feature_range.get("action_max", 17))
    obs_min = float(gym_feature_range.get("obs_min", 2))
    obs_max = float(gym_feature_range.get("obs_max", 348))
    action_q90 = float(math.ceil(float(gym_feature_range.get("action_q90", action_max))))
    obs_q90 = float(math.ceil(float(gym_feature_range.get("obs_q90", obs_max))))
    action_values = {float(v) for v in gym_feature_range.get("action_values", [])}
    obs_values = {float(v) for v in gym_feature_range.get("obs_values", [])}
    return {
        "n": int(n),
        "action_in_gym_full_range_fraction": _fraction((a >= action_min) & (a <= action_max)),
        "obs_in_gym_full_range_fraction": _fraction((o >= obs_min) & (o <= obs_max)),
        "joint_action_obs_in_gym_full_range_fraction": _fraction(
            (a >= action_min) & (a <= action_max) & (o >= obs_min) & (o <= obs_max)
        ),
        "action_in_gym_q90_range_fraction": _fraction((a >= action_min) & (a <= action_q90)),
        "obs_in_gym_q90_range_fraction": _fraction((o >= obs_min) & (o <= obs_q90)),
        "joint_action_obs_in_gym_q90_range_fraction": _fraction(
            (a >= action_min) & (a <= action_q90) & (o >= obs_min) & (o <= obs_q90)
        ),
        "action_exactly_in_gym_validation_values_fraction": (
            _fraction(np.asarray([float(v) in action_values for v in a], dtype=bool))
            if action_values
            else None
        ),
        "obs_exactly_in_gym_validation_values_fraction": (
            _fraction(np.asarray([float(v) in obs_values for v in o], dtype=bool))
            if obs_values
            else None
        ),
    }


def _summarize_progress_distribution(
    rows: list[dict[str, Any]],
    gym_feature_range: dict[str, Any],
    *,
    include_last_update: bool = True,
) -> dict[str, Any]:
    action_dims = _collector_vectors(rows, "action_dims")
    obs_dims = _collector_vectors(rows, "obs_dims")
    state_dims = _collector_vectors(rows, "state_dims")
    out = {
        "rows": int(len(rows)),
        "env_samples": int(len(action_dims)),
        "dims": {
            "action_dim": _stats(action_dims),
            "obs_dim": _stats(obs_dims),
            "state_dim": _stats(state_dims),
        },
        "gym_feature_overlap": _distribution_overlap(
            action_dims=action_dims,
            obs_dims=obs_dims,
            gym_feature_range=gym_feature_range,
        ),
        "pack_stats_over_updates": {
            "raw_reward_return_mean": _stats(_pack_stat_series(rows, "raw_reward_return_mean")),
            "training_reward_return_mean": _stats(_pack_stat_series(rows, "reward_return_mean")),
            "done_count": _stats(_pack_stat_series(rows, "done_count")),
        },
    }
    gain_keys = (
        "reward_state_input_gain_fraction",
        "reward_action_input_gain_fraction",
        "reward_noise_input_gain_fraction",
        "reward_state_to_action_gain_ratio",
    )
    gain_summary = {}
    for key in gain_keys:
        vals = _collector_vectors(rows, key)
        if vals:
            gain_summary[key] = _stats(vals)
    if gain_summary:
        out["reward_topology"] = gain_summary
    if rows and include_last_update:
        out["last_update"] = _summarize_progress_distribution(
            [rows[-1]],
            gym_feature_range,
            include_last_update=False,
        )
    return out


def _parse_fitmodel_validation_log(path: str | Path) -> dict[str, Any]:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        return {"exists": False, "path": str(p)}
    mean_returns = []
    per_env: dict[int, dict[str, float]] = {}
    mean_re = re.compile(r"^Epoch\s+(\d+)\s+mean_return_all\s+([-+0-9.eE]+)")
    env_re = re.compile(r"^Epoch\s+(\d+)\s+return_(.+)_return_mean\s+([-+0-9.eE]+)")
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        m = mean_re.match(line)
        if m:
            mean_returns.append({"epoch": int(m.group(1)), "mean_return_all": float(m.group(2))})
            continue
        m = env_re.match(line)
        if m:
            epoch = int(m.group(1))
            per_env.setdefault(epoch, {})[m.group(2)] = float(m.group(3))
    last_epoch = max(per_env) if per_env else None
    return {
        "exists": True,
        "path": str(p),
        "mean_return_all_series": mean_returns,
        "last_mean_return_all": mean_returns[-1] if mean_returns else None,
        "last_per_env_return_epoch": last_epoch,
        "last_per_env_return_mean": None if last_epoch is None else per_env[last_epoch],
    }


def _rule_summary_from_dominance(dominance: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for rule, data in ((dominance.get("summary_by_rule") or {}).items()):
        out[rule] = {
            "original_pass_rate": data.get("original_pass_rate"),
            "balanced_pass_rate": data.get("balanced_pass_rate"),
            "pass_rate_lift_x": data.get("pass_rate_lift_x"),
            "original_fail_fixed_by_balance_fraction": data.get("original_fail_fixed_by_balance_fraction"),
            "original_fail_still_fail_n": data.get("original_fail_still_fail_n"),
            "width_bias_correlations_from_rows": data.get("width_bias_correlations_from_rows"),
            "original_failure_reason_counts": data.get("original_failure_reason_counts"),
            "balanced_failure_reason_counts": data.get("balanced_failure_reason_counts"),
        }
    return out


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    fit = report["fitmodel_training_prior_distribution"]
    validation = report["fitmodel_gym_validation_symptom"]
    dom = report["reward_path_mismatch_dominance"]
    lines = [
        "# fit_model Generalization Mismatch Argument",
        "",
        "This is an exploratory offline argument report. It does not train PPO, sample environments, or modify exact SCM.",
        "",
        "## Current Evidence",
        "",
        f"- fit_model prior env samples read: `{fit.get('env_samples')}` from `{fit.get('rows')}` updates.",
        f"- fit_model action dim mean/q50/q90: `{fit['dims']['action_dim'].get('mean'):.3g}` / `{fit['dims']['action_dim'].get('q50'):.3g}` / `{fit['dims']['action_dim'].get('q90'):.3g}`.",
        f"- Gym baseline action dim mean/q50/q90: `{report['gym_validation_training_distribution']['dims']['action_dim'].get('mean'):.3g}` / `{report['gym_validation_training_distribution']['dims']['action_dim'].get('q50'):.3g}` / `{report['gym_validation_training_distribution']['dims']['action_dim'].get('q90'):.3g}`.",
        f"- fit_model joint action/obs in Gym q90 range fraction: `{fit['gym_feature_overlap'].get('joint_action_obs_in_gym_q90_range_fraction'):.3g}`.",
        f"- fit_model last validation mean_return_all: `{(validation.get('last_mean_return_all') or {}).get('mean_return_all')}`.",
        "",
        "## Reward-Path Dominance",
        "",
    ]
    for rule, data in dom.items():
        lines.extend([
            f"- `{rule}`: original pass `{data.get('original_pass_rate'):.4g}`, balanced pass `{data.get('balanced_pass_rate'):.4g}`, "
            f"fixed failed fraction `{data.get('original_fail_fixed_by_balance_fraction'):.4g}`.",
        ])
    lines.extend([
        "",
        "## Causal Claim Status",
        "",
        "- Strongly supported: the current healthy training prior is shifted away from Gym action/obs support, especially action dimension.",
        "- Strongly supported: within Gym-near dimensions, topology failure is dominated by reward action-path weakness and state/action ratio induced by width-biased reward paths.",
        "- Supported but not sufficient: reward-path balancing repairs a large fraction of Gym-near topology failures without changing dimensions or state/terminal path.",
        "- Not yet proven: this mismatch is sufficient to explain fit_model Gym generalization failure.",
        "- Not yet proven: this exact repair is necessary; another repair that restores healthy Gym-near support could also work.",
        "",
        "## Minimal Next Evidence Before Long Fit",
        "",
        "- Offline fixed-distribution audit: compare current healthy prior vs balanced healthy prior against Gym on action/obs/reward/done/state-drift statistics.",
        "- Small ablation only after the audit: same fit_model/PPO path, only generator changed, then compare Gym validation.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    failure_audit = _load_json(args.failure_audit_report)
    dominance = _load_json(args.dominance_report)
    balance_probe = _load_json(args.balance_probe_report)
    gym_feature_range = failure_audit.get("gym_feature_range") or balance_probe.get("gym_feature_range") or {}

    fit_rows = _load_jsonl(args.fitmodel_prior_progress)
    gym_rows = _load_jsonl(args.gym_progress)

    report = {
        "analysis_entry": "phase2_fitmodel_generalization_mismatch_argument",
        "exploratory_only": True,
        "contract": {
            "does_not_modify_exact_scm": True,
            "does_not_modify_default_prior_generator": True,
            "does_not_sample_environments": True,
            "does_not_run_ppo_training": True,
            "separates_environment_distribution_argument_from_ppo_implementation": True,
        },
        "input_artifacts": {
            "fitmodel_prior_progress": str(Path(args.fitmodel_prior_progress).expanduser().resolve()),
            "fitmodel_log": str(Path(args.fitmodel_log).expanduser().resolve()),
            "gym_progress": str(Path(args.gym_progress).expanduser().resolve()),
            "failure_audit_report": str(Path(args.failure_audit_report).expanduser().resolve()),
            "dominance_report": str(Path(args.dominance_report).expanduser().resolve()),
            "balance_probe_report": str(Path(args.balance_probe_report).expanduser().resolve()),
        },
        "gym_feature_range": gym_feature_range,
        "fitmodel_training_prior_distribution": _summarize_progress_distribution(fit_rows, gym_feature_range),
        "gym_validation_training_distribution": _summarize_progress_distribution(gym_rows, gym_feature_range),
        "fitmodel_gym_validation_symptom": _parse_fitmodel_validation_log(args.fitmodel_log),
        "reward_path_mismatch_dominance": _rule_summary_from_dominance(dominance),
        "balance_probe_support": {
            "sampling": balance_probe.get("sampling", {}),
            "transition_diversity_probe": balance_probe.get("transition_diversity_probe", {}),
            "selected_summary_by_rule_original": balance_probe.get("selected_summary_by_rule_original", {}),
            "selected_summary_by_rule_balanced": balance_probe.get("selected_summary_by_rule_balanced", {}),
        },
        "claim_assessment": {
            "sufficient_for_fitmodel_generalization_failure": {
                "status": "not_proven",
                "reason": (
                    "Existing evidence proves a strong environment-distribution mismatch and a targeted "
                    "repair for topology health, but does not yet eliminate other validation mismatches "
                    "or show a fit_model validation improvement under the isolated generator intervention."
                ),
            },
            "necessary_for_fitmodel_generalization_failure": {
                "status": "not_proven",
                "reason": (
                    "The repair is not shown to be unique. Any mechanism that restores healthy Gym-near "
                    "support while preserving diversity could in principle address the same mismatch."
                ),
            },
            "important_dominant_environment_mismatch": {
                "status": "strongly_supported",
                "reason": (
                    "The fit_model training prior is concentrated at much higher action dimensions than Gym, "
                    "while Gym-near candidates fail topology health mostly through reward action-path weakness "
                    "and state/action ratio; the balancing counterfactual repairs a large fraction of those failures."
                ),
            },
            "safe_to_replace_milestone_generator": {
                "status": "no",
                "reason": (
                    "This is still exploratory. The repaired generator-level path must first preserve diversity "
                    "and pass offline distribution checks, then undergo isolated fit_model validation."
                ),
            },
        },
        "alternative_explanations_not_yet_eliminated": [
            "obs/state dynamics distribution mismatch",
            "done/horizon distribution mismatch",
            "reward scale and VecNorm trajectory mismatch",
            "action bound and saturation distribution mismatch",
            "Gym validation policy-state or context-length effects",
            "multi-environment interference unrelated to reward topology",
        ],
        "recommended_non_longrun_next_steps": [
            "Build an offline fixed-distribution audit comparing current healthy prior, balanced Gym-near prior, and Gym validation on action/obs/reward/done/state-drift.",
            "Promote reward-path balancing only as an exploratory generator-level candidate with gated semantics: already healthy keep original; width-bias failed balance; still-failed reject.",
            "Before long fit_model, require the candidate generator to improve Gym-near support without collapsing action/obs/state/reward/done diversity.",
        ],
    }

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "fitmodel_generalization_mismatch_argument_report.json"
    md_path = out_dir / "fitmodel_generalization_mismatch_argument_summary.md"
    json_path.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(md_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fitmodel-prior-progress", type=str, default=DEFAULT_FITMODEL_PRIOR_PROGRESS)
    parser.add_argument("--fitmodel-log", type=str, default=DEFAULT_FITMODEL_LOG)
    parser.add_argument("--gym-progress", type=str, default=DEFAULT_GYM_PROGRESS)
    parser.add_argument("--failure-audit-report", type=str, default=DEFAULT_FAILURE_AUDIT_REPORT)
    parser.add_argument("--dominance-report", type=str, default=DEFAULT_DOMINANCE_REPORT)
    parser.add_argument("--balance-probe-report", type=str, default=DEFAULT_BALANCE_PROBE_REPORT)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    report = run(args)
    fit = report["fitmodel_training_prior_distribution"]
    print(json.dumps(_json_safe({
        "output_dir": str(Path(args.output_dir).expanduser().resolve()),
        "fitmodel_action_dim_mean": fit["dims"]["action_dim"].get("mean"),
        "fitmodel_action_dim_q50": fit["dims"]["action_dim"].get("q50"),
        "fitmodel_joint_gym_q90_overlap": fit["gym_feature_overlap"].get(
            "joint_action_obs_in_gym_q90_range_fraction"
        ),
        "claim": report["claim_assessment"]["important_dominant_environment_mismatch"]["status"],
        "sufficient": report["claim_assessment"]["sufficient_for_fitmodel_generalization_failure"]["status"],
    }), sort_keys=True))


if __name__ == "__main__":
    main()
