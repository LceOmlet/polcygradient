#!/usr/bin/env python
"""Triage obvious environment-distribution mismatches before long fit_model runs.

This is an offline exploratory report.  It intentionally favors high-signal
coarse evidence over expensive analysis: find large mismatches, state whether
their repair is predictable, and spell out diversity/health/optimizable guard
rails.  It does not sample environments or train PPO.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_FIT_PRIOR = "/home/chen/RLPFN/health/ppo_pack_runner/rlpfn_05_04_2026_09_00_32/prior_progress.jsonl"
DEFAULT_GYM_BASELINE = (
    "/home/chen/RLPFN/artifacts/"
    "phase2_exploratory_gym_validation_baseline_n64_s512_u40_e4_advnorm_0506/gym_progress.jsonl"
)
DEFAULT_Q90_BALANCED = (
    "/home/chen/RLPFN/artifacts/"
    "phase2_exploratory_reward_group_balance_q90_fixedlist_n64_s512_u20_e4_advnorm_0506/"
    "prior_progress.jsonl"
)
DEFAULT_FULL_BALANCED = (
    "/home/chen/RLPFN/artifacts/"
    "phase2_exploratory_reward_group_balance_full_fixedlist_n64_s512_u4_e4_advnorm_0506/"
    "prior_progress.jsonl"
)
DEFAULT_FIT_LOG = "/home/chen/RLPFN/health/log/rlpfn_05_04_2026_09_00_32.log"
DEFAULT_ARGUMENT_REPORT = (
    "/home/chen/RLPFN/artifacts/"
    "phase2_exploratory_fitmodel_generalization_mismatch_argument_0506/"
    "fitmodel_generalization_mismatch_argument_report.json"
)
DEFAULT_DOMINANCE_REPORT = (
    "/home/chen/RLPFN/artifacts/"
    "phase2_exploratory_reward_group_balance_dominance_0506/"
    "dominance_report.json"
)
DEFAULT_OUTPUT_DIR = (
    "/home/chen/RLPFN/artifacts/"
    "phase2_exploratory_obvious_mismatch_triage_0506"
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
    if not p.exists():
        return []
    rows = []
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


def _finite(values: list[Any]) -> np.ndarray:
    out = []
    for value in values:
        try:
            x = float(value)
        except Exception:
            continue
        if math.isfinite(x):
            out.append(x)
    return np.asarray(out, dtype=np.float64)


def _stats(values: list[Any]) -> dict[str, Any]:
    arr = _finite(values)
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "q10": float(np.quantile(arr, 0.10)),
        "q50": float(np.quantile(arr, 0.50)),
        "q90": float(np.quantile(arr, 0.90)),
        "max": float(np.max(arr)),
    }


def _collector_values(rows: list[dict[str, Any]], key: str) -> list[Any]:
    vals = []
    for row in rows:
        cur = (row.get("collector") or {}).get(key)
        if isinstance(cur, list):
            vals.extend(cur)
    return vals


def _pack_stat_values(rows: list[dict[str, Any]], key: str) -> list[Any]:
    vals = []
    for row in rows:
        cur = ((row.get("pack_summary") or {}).get("stats") or {}).get(key)
        if isinstance(cur, (int, float)):
            vals.append(float(cur))
    return vals


def _last_pack_stat(rows: list[dict[str, Any]], key: str) -> float | None:
    if not rows:
        return None
    cur = ((rows[-1].get("pack_summary") or {}).get("stats") or {}).get(key)
    return float(cur) if isinstance(cur, (int, float)) and math.isfinite(float(cur)) else None


def _done_rate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rates = []
    for row in rows:
        stats = ((row.get("pack_summary") or {}).get("stats") or {})
        done = stats.get("done_count")
        shapes = ((row.get("pack_summary") or {}).get("shapes") or {})
        done_shape = shapes.get("dones")
        if isinstance(done, (int, float)) and isinstance(done_shape, list) and len(done_shape) >= 2:
            denom = max(1, int(done_shape[0]) * int(done_shape[1]))
            rates.append(float(done) / float(denom))
    return _stats(rates)


def _summarize_pack(name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {
        "name": name,
        "updates": int(len(rows)),
        "action_dim": _stats(_collector_values(rows, "action_dims")),
        "obs_dim": _stats(_collector_values(rows, "obs_dims")),
        "state_dim": _stats(_collector_values(rows, "state_dims")),
        "done_rate": _done_rate(rows),
        "raw_reward_return_mean": _stats(_pack_stat_values(rows, "raw_reward_return_mean")),
        "training_reward_return_mean": _stats(_pack_stat_values(rows, "reward_return_mean")),
        "last_raw_reward_return_mean": _last_pack_stat(rows, "raw_reward_return_mean"),
        "last_training_reward_return_mean": _last_pack_stat(rows, "reward_return_mean"),
    }
    gain_keys = (
        "reward_state_input_gain_fraction",
        "reward_action_input_gain_fraction",
        "reward_noise_input_gain_fraction",
        "reward_state_to_action_gain_ratio",
    )
    gain_summary = {}
    for key in gain_keys:
        vals = _collector_values(rows, key)
        if vals:
            gain_summary[key] = _stats(vals)
    if gain_summary:
        out["reward_topology"] = gain_summary
    survival_enabled = _collector_values(rows, "survival_reward_enabled")
    survival_weight = _collector_values(rows, "survival_reward_weight")
    if survival_enabled:
        enabled = _finite(survival_enabled)
        out["survival"] = {
            "enabled_fraction": float(np.mean(enabled != 0.0)) if enabled.size else None,
            "weight": _stats(survival_weight),
        }
    return out


def _parse_validation_survival(path: str | Path) -> dict[str, Any]:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        return {"exists": False, "path": str(p)}
    pat = re.compile(r"^Epoch\s+(\d+)\s+return_(.+?)_(return|len|reward_survive)_mean\s+([-+0-9.eE]+)")
    by: dict[tuple[int, str], dict[str, float]] = {}
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        m = pat.match(line)
        if not m:
            continue
        epoch = int(m.group(1))
        env_id = str(m.group(2))
        metric = str(m.group(3))
        by.setdefault((epoch, env_id), {})[metric] = float(m.group(4))
    if not by:
        return {"exists": True, "path": str(p), "envs": {}}
    last_epoch = max(epoch for epoch, _env in by)
    envs = {}
    ratios = []
    for (epoch, env_id), data in sorted(by.items()):
        if epoch != last_epoch or env_id.endswith("_explore_rollout"):
            continue
        ret = data.get("return")
        surv = data.get("reward_survive", 0.0)
        length = data.get("len")
        ratio = None
        if isinstance(ret, (int, float)) and abs(float(ret)) > 1e-9:
            ratio = float(surv) / float(ret)
            ratios.append(ratio)
        envs[env_id] = {
            "return_mean": ret,
            "len_mean": length,
            "reward_survive_mean": surv,
            "survival_to_return_ratio": ratio,
            "has_survival_component": bool(abs(float(surv or 0.0)) > 1e-9),
        }
    return {
        "exists": True,
        "path": str(p),
        "last_epoch": int(last_epoch),
        "env_count": int(len(envs)),
        "survival_env_count": int(sum(1 for d in envs.values() if d["has_survival_component"])),
        "survival_env_fraction": float(
            sum(1 for d in envs.values() if d["has_survival_component"]) / max(len(envs), 1)
        ),
        "survival_to_return_ratio_stats": _stats(ratios),
        "envs": envs,
    }


def _mismatch_findings(
    *,
    fit: dict[str, Any],
    gym: dict[str, Any],
    q90: dict[str, Any],
    full: dict[str, Any],
    argument: dict[str, Any],
    dominance: dict[str, Any],
    validation_survival: dict[str, Any],
) -> list[dict[str, Any]]:
    fit_action_q50 = fit["action_dim"].get("q50")
    gym_action_q50 = gym["action_dim"].get("q50")
    fit_q90_overlap = (((argument.get("fitmodel_training_prior_distribution") or {}).get("gym_feature_overlap") or {}).get("joint_action_obs_in_gym_q90_range_fraction"))
    findings = [
        {
            "id": "action_dim_support_shift",
            "severity": "very_high",
            "certainty": "very_high",
            "evidence": {
                "fit_action_dim_q50": fit_action_q50,
                "gym_action_dim_q50": gym_action_q50,
                "fit_joint_action_obs_gym_q90_overlap": fit_q90_overlap,
                "fit_action_dim_mean": fit["action_dim"].get("mean"),
                "gym_action_dim_mean": gym["action_dim"].get("mean"),
            },
            "repair": "Do not change dimension sampler knobs. Use gated reward-path balancing so Gym-near low-action-dim candidates can pass topology health, then reject still-failed candidates.",
            "predictable_target": "reward_action_input_gain_fraction and reward_state_to_action_gain_ratio after repair",
            "regression_guards": [
                "post-repair topology pass must improve",
                "action/obs/state/noise diversity must not collapse",
                "short fixed-list PPO raw return must not regress versus unbalanced selected lists",
            ],
        },
        {
            "id": "healthy_selection_width_bias",
            "severity": "very_high",
            "certainty": "very_high",
            "evidence": {
                "fit_reward_action_gain_mean": ((fit.get("reward_topology") or {}).get("reward_action_input_gain_fraction") or {}).get("mean"),
                "fit_reward_ratio_q50": ((fit.get("reward_topology") or {}).get("reward_state_to_action_gain_ratio") or {}).get("q50"),
                "dominance_by_rule": {
                    rule: {
                        "original_pass_rate": data.get("original_pass_rate"),
                        "balanced_pass_rate": data.get("balanced_pass_rate"),
                        "fixed_failed_fraction": data.get("original_fail_fixed_by_balance_fraction"),
                    }
                    for rule, data in ((dominance.get("summary_by_rule") or {}).items())
                },
            },
            "repair": "Partition-aware, reward-path-only balancing with already-healthy keep-original semantics.",
            "predictable_target": "topology pass/yield among Gym-near candidates",
            "regression_guards": [
                "already-healthy candidates should not be force-balanced",
                "state/terminal path must remain original",
                "balanced candidates must still pass finite reward/state/done diversity probes",
            ],
        },
        {
            "id": "done_rate_horizon_shift",
            "severity": "medium_high",
            "certainty": "medium",
            "evidence": {
                "fit_done_rate_mean": fit["done_rate"].get("mean"),
                "gym_done_rate_mean": gym["done_rate"].get("mean"),
                "q90_balanced_done_rate_mean": q90["done_rate"].get("mean"),
                "full_balanced_done_rate_mean": full["done_rate"].get("mean"),
            },
            "repair": "Prefer selecting/balancing Gym-near healthy candidates and then checking done-rate distribution before touching terminal reset defaults.",
            "predictable_target": "done_rate and first_done distribution",
            "regression_guards": [
                "do not tune terminal reset to match Gym exactly yet",
                "reject candidates with pathological early termination or no termination",
            ],
        },
        {
            "id": "survival_reward_absent_in_prior",
            "severity": "medium_high",
            "certainty": "high_for_mismatch_low_for_repair_effect",
            "evidence": {
                "fit_prior_survival_enabled_fraction": (fit.get("survival") or {}).get("enabled_fraction"),
                "q90_balanced_survival_enabled_fraction": (q90.get("survival") or {}).get("enabled_fraction"),
                "gym_validation_survival_env_fraction": validation_survival.get("survival_env_fraction"),
                "gym_validation_survival_env_count": validation_survival.get("survival_env_count"),
                "gym_validation_env_count": validation_survival.get("env_count"),
            },
            "repair": "Do not add to milestone yet. First test exploratory survival on a fixed list with terminal-sensitive candidates, small positive weights, and raw-vs-training return/component logs.",
            "predictable_target": "episode length/return correlation and survival component contribution",
            "regression_guards": [
                "survival must improve raw return through longer successful episodes, not just add a constant reward offset",
                "advantage/value targets must remain finite and not be dominated by survival",
                "reward_env diversity must remain nonzero",
            ],
        },
    ]
    return findings


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Obvious Mismatch Triage",
        "",
        "Exploratory offline report. No environment sampling, no PPO training, no generator mutation.",
        "",
        "## Main Findings",
    ]
    for finding in report["findings"]:
        lines.extend([
            "",
            f"### {finding['id']}",
            f"- severity: `{finding['severity']}`; certainty: `{finding['certainty']}`",
            f"- repair: {finding['repair']}",
            f"- predictable target: `{finding['predictable_target']}`",
        ])
    lines.extend([
        "",
        "## Survival Position",
        "",
        report["survival_argument"]["conclusion"],
        "",
        "## Recommended Order",
    ])
    for item in report["recommended_order"]:
        lines.append(f"- {item}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    fit_rows = _load_jsonl(args.fit_prior_progress)
    gym_rows = _load_jsonl(args.gym_progress)
    q90_rows = _load_jsonl(args.q90_balanced_progress)
    full_rows = _load_jsonl(args.full_balanced_progress)
    argument = _load_json(args.argument_report)
    dominance = _load_json(args.dominance_report)
    fit = _summarize_pack("fitmodel_current_prior", fit_rows)
    gym = _summarize_pack("gym_validation_pack_baseline", gym_rows)
    q90 = _summarize_pack("balanced_gym_q90_fixedlist", q90_rows)
    full = _summarize_pack("balanced_gym_full_fixedlist", full_rows)
    validation_survival = _parse_validation_survival(args.fitmodel_log)

    findings = _mismatch_findings(
        fit=fit,
        gym=gym,
        q90=q90,
        full=full,
        argument=argument,
        dominance=dominance,
        validation_survival=validation_survival,
    )
    report = {
        "analysis_entry": "phase2_obvious_mismatch_triage",
        "exploratory_only": True,
        "contract": {
            "does_not_modify_exact_scm": True,
            "does_not_modify_default_prior_generator": True,
            "does_not_sample_environments": True,
            "does_not_run_ppo_training": True,
            "focus": "obvious large mismatches and robust minimal repairs",
        },
        "input_artifacts": {
            "fit_prior_progress": str(Path(args.fit_prior_progress).expanduser().resolve()),
            "gym_progress": str(Path(args.gym_progress).expanduser().resolve()),
            "q90_balanced_progress": str(Path(args.q90_balanced_progress).expanduser().resolve()),
            "full_balanced_progress": str(Path(args.full_balanced_progress).expanduser().resolve()),
            "fitmodel_log": str(Path(args.fitmodel_log).expanduser().resolve()),
            "argument_report": str(Path(args.argument_report).expanduser().resolve()),
            "dominance_report": str(Path(args.dominance_report).expanduser().resolve()),
        },
        "pack_summaries": {
            "fitmodel_current_prior": fit,
            "gym_validation_pack_baseline": gym,
            "balanced_gym_q90_fixedlist": q90,
            "balanced_gym_full_fixedlist": full,
        },
        "validation_survival_summary": validation_survival,
        "findings": findings,
        "survival_argument": {
            "conclusion": (
                "Survival is a real Gym-vs-prior mismatch: current prior and balanced fixed lists have "
                "survival enabled fraction 0, while the fit_model Gym validation log has survival reward "
                "components in a large subset of MuJoCo balance tasks.  However survival is not yet a safe "
                "milestone change: its PPO effect is only useful when termination is action/state sensitive. "
                "A constant survival reward with weak termination dependence can become a scale/baseline "
                "artifact.  Treat it as a separate exploratory candidate after reward-path balancing and "
                "done-rate guards are stable."
            ),
            "ppo_advantage_mechanism": (
                "If done probability depends on action/state, per-alive-step survival reward changes Monte "
                "Carlo/GAE returns: actions that postpone termination receive more future survival reward, so "
                "advantages can train survival behavior.  If termination is independent of policy, the term "
                "is mostly a constant offset and value baseline absorbs it."
            ),
            "minimum_test_before_accepting": [
                "fixed-list prior smoke with survival enabled on terminal-sensitive candidates",
                "compare raw return delta, episode length delta, reward_env return, and survival component",
                "verify advantage/value target scale remains finite and not dominated by survival",
            ],
        },
        "recommended_order": [
            "Primary repair: gated reward-path balancing for Gym-near failed candidates, preserving already-healthy candidates.",
            "Offline guard audit: compare fit prior, balanced prior, and Gym on action/obs/topology/done/reward/state drift.",
            "Only if still obviously mismatched: exploratory survival-on subset test with component logging.",
            "Set a new milestone only after health, diversity, and short fixed-list optimizability improve without collapse.",
        ],
    }
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "obvious_mismatch_triage_report.json"
    md_path = out_dir / "obvious_mismatch_triage_summary.md"
    json_path.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(md_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-prior-progress", type=str, default=DEFAULT_FIT_PRIOR)
    parser.add_argument("--gym-progress", type=str, default=DEFAULT_GYM_BASELINE)
    parser.add_argument("--q90-balanced-progress", type=str, default=DEFAULT_Q90_BALANCED)
    parser.add_argument("--full-balanced-progress", type=str, default=DEFAULT_FULL_BALANCED)
    parser.add_argument("--fitmodel-log", type=str, default=DEFAULT_FIT_LOG)
    parser.add_argument("--argument-report", type=str, default=DEFAULT_ARGUMENT_REPORT)
    parser.add_argument("--dominance-report", type=str, default=DEFAULT_DOMINANCE_REPORT)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    report = run(args)
    fit = report["pack_summaries"]["fitmodel_current_prior"]
    gym = report["pack_summaries"]["gym_validation_pack_baseline"]
    print(json.dumps(_json_safe({
        "output_dir": str(Path(args.output_dir).expanduser().resolve()),
        "fit_action_dim_q50": fit["action_dim"].get("q50"),
        "gym_action_dim_q50": gym["action_dim"].get("q50"),
        "fit_done_rate_mean": fit["done_rate"].get("mean"),
        "gym_done_rate_mean": gym["done_rate"].get("mean"),
        "survival_env_fraction": report["validation_survival_summary"].get("survival_env_fraction"),
        "top_finding": report["findings"][0]["id"] if report["findings"] else None,
    }), sort_keys=True))


if __name__ == "__main__":
    main()
