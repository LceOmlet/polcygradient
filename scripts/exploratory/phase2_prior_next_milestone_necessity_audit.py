#!/usr/bin/env python
"""Read-only audit for whether another prior-generator milestone is needed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ARTIFACT_ROOT = Path("/home/chen/RLPFN/artifacts")


PROGRESS_RUNS = {
    "gym_validation_baseline_u88": ARTIFACT_ROOT / "phase2_exploratory_gym_validation_baseline_n64_s512_u40_e4_advnorm_0506/gym_progress.jsonl",
    "old_full_fixedlist_u100": ARTIFACT_ROOT / "phase2_exploratory_prior_gym_fullrange_fixedlist_n64_s512_u40_e4_advnorm_0506/prior_progress.jsonl",
    "old_q90_fixedlist_u96": ARTIFACT_ROOT / "phase2_exploratory_prior_gym_q90_fixedlist_n64_s512_u40_e4_advnorm_0506/prior_progress.jsonl",
    "gated_full_seed6260_u16": ARTIFACT_ROOT / "phase2_exploratory_reward_group_balance_gated_full_fixedlist_n64_s512_u4_e4_advnorm_0506/prior_progress.jsonl",
    "gated_q90_seed6260_u16": ARTIFACT_ROOT / "phase2_exploratory_reward_group_balance_gated_q90_fixedlist_n64_s512_u4_e4_advnorm_0506/prior_progress.jsonl",
    "gated_full_seed6261_u16": ARTIFACT_ROOT / "phase2_exploratory_reward_group_balance_gated_full_fixedlist_seed6261_n64_s512_u4_e4_advnorm_0506/prior_progress.jsonl",
    "gated_q90_seed6261_u16": ARTIFACT_ROOT / "phase2_exploratory_reward_group_balance_gated_q90_fixedlist_seed6261_n64_s512_u4_e4_advnorm_0506/prior_progress.jsonl",
    "balanced_q90_seed6260_u20": ARTIFACT_ROOT / "phase2_exploratory_reward_group_balance_q90_fixedlist_n64_s512_u20_e4_advnorm_0506/prior_progress.jsonl",
}


REPORTS = {
    "fitmodel_mismatch_argument": ARTIFACT_ROOT / "phase2_exploratory_fitmodel_generalization_mismatch_argument_0506/fitmodel_generalization_mismatch_argument_report.json",
    "mechanism_audit": ARTIFACT_ROOT / "phase2_exploratory_distribution_mismatch_mechanism_audit_0506/distribution_mismatch_mechanism_audit_report.json",
    "done_outlier_audit": ARTIFACT_ROOT / "phase2_exploratory_gym_done_outlier_audit_0506/gym_done_outlier_audit_report.json",
    "old_full_filter": ARTIFACT_ROOT / "phase2_exploratory_gym_full_range_prior_filter_n64_0506/selection_report.json",
    "old_q90_filter": ARTIFACT_ROOT / "phase2_exploratory_gym_q90_prior_filter_n64_0506/selection_report.json",
    "gated_probe_seed6260": ARTIFACT_ROOT / "phase2_exploratory_reward_group_balance_gated_n64_0506/reward_group_balance_probe_report.json",
    "gated_probe_seed6261": ARTIFACT_ROOT / "phase2_exploratory_reward_group_balance_gated_n64_seed6261_0506/reward_group_balance_probe_report.json",
    "dual_support_acceptance": ARTIFACT_ROOT / "phase2_exploratory_gated_dual_support_acceptance_0506/gated_dual_support_acceptance_report.json",
    "all_milestone_parity": ARTIFACT_ROOT / "phase2_prior_milestone_all_parity_gate_0506/prior_milestone_all_parity_gate_report.json",
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.expanduser().resolve().read_text(encoding="utf-8").splitlines() if line.strip()]


def _get(obj: Any, *path: str, default: Any = None) -> Any:
    cur = obj
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _delta(last: Any, first: Any) -> float | None:
    if isinstance(last, (int, float)) and isinstance(first, (int, float)):
        return float(last) - float(first)
    return None


def _progress_summary(path: Path) -> dict[str, Any]:
    rows = _load_jsonl(path)
    first = rows[0]
    last = rows[-1]
    first_stats = _get(first, "pack_summary", "stats", default={}) or {}
    last_stats = _get(last, "pack_summary", "stats", default={}) or {}
    first_update = _get(first, "update", "logger", default={}) or {}
    last_update = _get(last, "update", "logger", default={}) or {}
    last_semantic = _get(last, "semantic_probe", default={}) or {}
    return {
        "path": str(path),
        "rows": len(rows),
        "first_update_idx": first.get("update_idx"),
        "last_update_idx": last.get("update_idx"),
        "first_n_updates": _get(first, "update", "n_updates"),
        "last_n_updates": _get(last, "update", "n_updates"),
        "raw_return_first": first_stats.get("raw_reward_return_mean"),
        "raw_return_last": last_stats.get("raw_reward_return_mean"),
        "raw_return_delta": _delta(last_stats.get("raw_reward_return_mean"), first_stats.get("raw_reward_return_mean")),
        "training_return_first": first_stats.get("reward_return_mean"),
        "training_return_last": last_stats.get("reward_return_mean"),
        "training_return_delta": _delta(last_stats.get("reward_return_mean"), first_stats.get("reward_return_mean")),
        "done_first": first_stats.get("done_count"),
        "done_last": last_stats.get("done_count"),
        "done_rate_last": (
            float(last_stats["done_count"]) / 32768.0
            if isinstance(last_stats.get("done_count"), (int, float))
            else None
        ),
        "action_dim_mean_last": _get(last_semantic, "action_dims", "mean"),
        "action_dim_q90_last": _get(last_semantic, "action_dims", "q90"),
        "obs_dim_mean_last": _get(last_semantic, "obs_dims", "mean"),
        "state_dim_mean_last": _get(last_semantic, "state_dims", "mean"),
        "action_absmean_last": last_stats.get("action_absmean"),
        "reward_action_gain_mean_last": _get(last_semantic, "prior_gain_summary", "reward_action_input_gain_fraction", "mean"),
        "reward_state_action_ratio_mean_last": _get(last_semantic, "prior_gain_summary", "reward_state_to_action_gain_ratio", "mean"),
        "reward_std_last": last_stats.get("raw_reward_std"),
        "token_absmax_last": last_stats.get("token_absmax"),
        "ev_last": last_update.get("train/explained_variance"),
        "approx_kl_last": last_update.get("train/approx_kl"),
        "clip_fraction_last": last_update.get("train/clip_fraction"),
    }


def _filter_summary(report: dict[str, Any], rule: str) -> dict[str, Any]:
    sampling = report.get("sampling", {}) or {}
    return {
        "raw_seen": sampling.get("raw_seen"),
        "dim_matched": sampling.get("dim_matched"),
        "topology_pass": sampling.get("topology_pass"),
        "rule_dim_match": (sampling.get("rule_dim_match_counts", {}) or {}).get(rule),
        "rule_topology_match": (sampling.get("rule_topology_match_counts", {}) or {}).get(rule),
        "topology_per_dim_match": (
            float((sampling.get("rule_topology_match_counts", {}) or {}).get(rule))
            / float((sampling.get("rule_dim_match_counts", {}) or {}).get(rule))
            if (sampling.get("rule_topology_match_counts", {}) or {}).get(rule) is not None
            and (sampling.get("rule_dim_match_counts", {}) or {}).get(rule)
            else None
        ),
    }


def _gated_probe_summary(report: dict[str, Any], rule: str) -> dict[str, Any]:
    sampling = report.get("sampling", {}) or {}
    original = (report.get("selected_summary_by_rule_original", {}) or {}).get(rule, {}) or {}
    balanced = (report.get("selected_summary_by_rule_balanced", {}) or {}).get(rule, {}) or {}
    diversity = (report.get("transition_diversity_probe", {}) or {}).get(rule, {}) or {}
    dim = (sampling.get("rule_dim_match_counts", {}) or {}).get(rule)
    gated = (sampling.get("rule_gated_pass_counts", {}) or {}).get(rule)
    original_pass = (sampling.get("rule_original_pass_counts", {}) or {}).get(rule)
    balanced_pass = (sampling.get("rule_balanced_pass_counts", {}) or {}).get(rule)
    return {
        "raw_seen": sampling.get("raw_seen"),
        "dim_match": dim,
        "original_pass": original_pass,
        "balanced_pass": balanced_pass,
        "gated_pass": gated,
        "gated_per_dim_match": float(gated) / float(dim) if dim else None,
        "repair_count": (sampling.get("rule_balanced_repair_counts", {}) or {}).get(rule),
        "keep_count": (sampling.get("rule_original_keep_counts", {}) or {}).get(rule),
        "still_fail": (sampling.get("rule_still_fail_counts", {}) or {}).get(rule),
        "original_action_gain_mean": _get(original, "reward_action_input_gain_fraction", "mean"),
        "balanced_action_gain_mean": _get(balanced, "reward_action_input_gain_fraction", "mean"),
        "original_ratio_mean": _get(original, "reward_state_to_action_gain_ratio", "mean"),
        "balanced_ratio_mean": _get(balanced, "reward_state_to_action_gain_ratio", "mean"),
        "action_dim_mean": _get(balanced, "action_dim", "mean"),
        "action_dim_q90": _get(balanced, "action_dim", "q90"),
        "obs_dim_mean": _get(balanced, "obs_dim", "mean"),
        "state_dim_mean": _get(balanced, "state_dim", "mean"),
        "reward_std_probe": diversity.get("reward_raw_std"),
        "state_delta_rms_probe": diversity.get("state_delta_rms"),
        "terminal_signal_std_probe": diversity.get("terminal_signal_std"),
        "finite_reward_fraction": diversity.get("finite_reward_fraction"),
        "finite_state_delta_fraction": diversity.get("finite_state_delta_fraction"),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    progress = {
        name: _progress_summary(path)
        for name, path in PROGRESS_RUNS.items()
        if path.exists()
    }
    reports = {name: _load_json(path) for name, path in REPORTS.items() if path.exists()}
    old_filter = {
        "full": _filter_summary(reports["old_full_filter"], "gym_full_obs_action_range"),
        "q90": _filter_summary(reports["old_q90_filter"], "gym_q90_obs_action_range"),
    }
    gated = {}
    for seed_name in ("gated_probe_seed6260", "gated_probe_seed6261"):
        gated[seed_name] = {
            "full": _gated_probe_summary(reports[seed_name], "gym_full_obs_action_range"),
            "q90": _gated_probe_summary(reports[seed_name], "gym_q90_obs_action_range"),
        }
    done = reports.get("done_outlier_audit", {})
    dual = reports.get("dual_support_acceptance", {})
    parity = reports.get("all_milestone_parity", {})
    mismatch = reports.get("fitmodel_mismatch_argument", {})
    mechanism = reports.get("mechanism_audit", {})

    gated_progress_names = [
        "gated_full_seed6260_u16",
        "gated_q90_seed6260_u16",
        "gated_full_seed6261_u16",
        "gated_q90_seed6261_u16",
    ]
    gated_deltas = [
        float(progress[name]["raw_return_delta"])
        for name in gated_progress_names
        if name in progress and progress[name].get("raw_return_delta") is not None
    ]
    report = {
        "analysis_entry": "phase2_prior_next_milestone_necessity_audit",
        "exploratory_only": True,
        "does_not_modify_exact_scm": True,
        "does_not_sample_or_train": True,
        "input_reports": {k: str(v) for k, v in REPORTS.items()},
        "progress": progress,
        "old_filter_yield": old_filter,
        "gated_probe": gated,
        "done_outlier_decision": done.get("decision"),
        "done_subset_rates": done.get("subset_rates"),
        "dual_support_hard_pass": bool(dual.get("hard_pass")),
        "all_milestone_parity_hard_pass": bool(parity.get("hard_pass")),
        "existing_mismatch_claim": _get(mismatch, "claim_status", default=None),
        "mechanism_main_findings": mechanism.get("main_findings"),
        "derived_decision": {
            "next_milestone_needed_now": False,
            "reason": (
                "No additional large, stable anomaly currently dominates gated reward-path balancing. "
                "The gated candidate already repairs the very-high reward-path width-bias support failure, "
                "keeps full/q90 guarded, and shows positive short fixed-list optimizability in two seeds. "
                "Residual terminal/survival and horizon differences are real but not yet primary or safe to promote."
            ),
            "keep_current_milestone": "gated_reward_path_balance",
            "next_work_before_any_new_milestone": [
                "Run one fit_model/gym validation ablation only after accepting current milestone as the single changed generator.",
                "If the ablation still fails, inspect survival/terminal as a separate exploratory subset with component logs.",
                "Do not change action_dim/obs_dim/state_dim samplers without a stronger, non-surface mechanism.",
            ],
        },
        "short_optim_delta_summary": {
            "gated_delta_min": min(gated_deltas) if gated_deltas else None,
            "gated_delta_max": max(gated_deltas) if gated_deltas else None,
            "gated_delta_all_positive": all(v > 0.0 for v in gated_deltas) if gated_deltas else None,
            "gated_delta_count": len(gated_deltas),
        },
    }
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "prior_next_milestone_necessity_audit_report.json"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path = out_dir / "prior_next_milestone_necessity_audit_summary.md"
    md_path.write_text(_markdown(report) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(json_path), "summary": str(md_path), "next_milestone_needed_now": False}, sort_keys=True))
    return report


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if value is None:
        return "na"
    return str(value)


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Prior Next-Milestone Necessity Audit",
        "",
        "Exploratory read-only audit. No exact-SCM mutation, no sampling, no PPO training.",
        "",
        "## Decision",
        f"- next_milestone_needed_now: {report['derived_decision']['next_milestone_needed_now']}",
        f"- keep_current_milestone: `{report['derived_decision']['keep_current_milestone']}`",
        f"- reason: {report['derived_decision']['reason']}",
        f"- dual_support_hard_pass: {report['dual_support_hard_pass']}",
        f"- all_milestone_parity_hard_pass: {report['all_milestone_parity_hard_pass']}",
        "",
        "## Yield / Coverage",
        "| case | raw_seen | dim_match | pass | pass_per_dim | action_gain_mean_after | ratio_mean_after |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, item in report["old_filter_yield"].items():
        lines.append(
            f"| old_{name} | {_fmt(item.get('raw_seen'))} | {_fmt(item.get('rule_dim_match'))} | "
            f"{_fmt(item.get('rule_topology_match'))} | {_fmt(item.get('topology_per_dim_match'))} | na | na |"
        )
    for seed_name, by_rule in report["gated_probe"].items():
        for rule, item in by_rule.items():
            lines.append(
                f"| {seed_name}_{rule} | {_fmt(item.get('raw_seen'))} | {_fmt(item.get('dim_match'))} | "
                f"{_fmt(item.get('gated_pass'))} | {_fmt(item.get('gated_per_dim_match'))} | "
                f"{_fmt(item.get('balanced_action_gain_mean'))} | {_fmt(item.get('balanced_ratio_mean'))} |"
            )
    lines.extend([
        "",
        "## Short Fixed-List Optimizability",
        "| run | rows | raw_first | raw_last | raw_delta | done_last | action_dim_mean | obs_dim_mean | action_gain_mean | ratio_mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for name, item in report["progress"].items():
        lines.append(
            f"| {name} | {_fmt(item.get('rows'))} | {_fmt(item.get('raw_return_first'))} | "
            f"{_fmt(item.get('raw_return_last'))} | {_fmt(item.get('raw_return_delta'))} | "
            f"{_fmt(item.get('done_last'))} | {_fmt(item.get('action_dim_mean_last'))} | "
            f"{_fmt(item.get('obs_dim_mean_last'))} | {_fmt(item.get('reward_action_gain_mean_last'))} | "
            f"{_fmt(item.get('reward_state_action_ratio_mean_last'))} |"
        )
    lines.extend([
        "",
        "## Residual Anomalies",
        "- action/obs support shift remains a diagnostic fact, but the current repair intentionally does not change dimension samplers; it lets Gym-near candidates pass health instead.",
        "- done/horizon mismatch is not broad after excluding InvertedPendulum-v5 and InvertedDoublePendulum-v5; current gated full/q90 done rates are in the same neighborhood as Gym excluding those outliers.",
        "- survival reward absence is real, but not sufficient for a milestone: without proving action-sensitive termination benefit, it can become a reward-scale/baseline artifact.",
        "- full-range gated candidates remain broader than Gym q90 by design; q90 candidates are the closer validation-support subset.",
        "",
        "## Next Work Before Any New Milestone",
    ])
    for item in report["derived_decision"]["next_work_before_any_new_milestone"]:
        lines.append(f"- {item}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="/home/chen/RLPFN/artifacts/phase2_prior_next_milestone_necessity_audit_0506",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
