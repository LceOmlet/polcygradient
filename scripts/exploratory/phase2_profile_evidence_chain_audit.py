#!/usr/bin/env python
"""Summarize the profile evidence chain for a candidate prior milestone.

This is a read-only audit.  It connects the artifacts that matter for the
profile mainline:

source pool -> cached/cheap labels -> profile selector -> health/diversity
smoke -> pack-to-fit_model parity -> selected-list reference-label sanity.

It does not sample environments, train PPO, or change exact SCM / fit_model.
Raw return positivity is not used as a gate.
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


ART = Path("/home/chen/RLPFN/artifacts")
RULES = ("gym_full_obs_action_range", "gym_q90_obs_action_range")
AXES = (
    "locomotion_witness",
    "low_energy_not_ctrl_only",
    "sign_or_phase_sensitive",
    "state_conditioned_energy_injection",
)

DEFAULT_SOURCE_REPORT = (
    ART
    / "phase2_profile_online_candidate_pool_source384_seed9702_0512"
    / "source_pool"
    / "reward_group_balance_probe_report.json"
)
DEFAULT_CHUNKED_LABEL_REPORT = (
    ART
    / "phase2_profile_chunked_label_source384_seed9702_cheap_obs2_c128_0512"
    / "profile_chunked_label_contract.json"
)
DEFAULT_SELECTOR_REPORT = (
    ART
    / "phase2_profile_band_selector_source384_seed9702_cheap_obs2_v2d_0512"
    / "profile_band_selector_report.json"
)
DEFAULT_SMOKE_REPORT = (
    ART
    / "phase2_profile_integrated_health_diversity_smoke_source384_seed9702_cheap_obs2_v2d_0512"
    / "profile_integrated_health_diversity_smoke.json"
)
DEFAULT_PARITY_REPORT = (
    ART
    / "phase2_profile_source384_v2d_pack_fit_model_parity_n4_s16_0512"
    / "profile_v2e_pack_fit_model_parity_report.json"
)
DEFAULT_REFERENCE_LABEL_DIR = ART / "phase2_profile_selected64_reference_labels_obs4_seed9702_0512"
DEFAULT_OUTPUT_DIR = ART / "phase2_profile_source384_v2d_evidence_chain_0512"


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    csv_path = Path(path).expanduser().resolve()
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if isinstance(value, Path):
        return str(value)
    return value


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _rate(count: int, n: int) -> float | None:
    return float(count) / float(n) if n else None


def _action_counts(action_report: dict[str, Any]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for item in action_report.get("lists", []):
        rule = str(item.get("label", ""))
        rows = item.get("classification", {}).get("per_env", []) or []
        out[rule] = {
            "n": len(rows),
            "low_energy_not_ctrl_only": sum(_safe_bool(row.get("low_energy_not_ctrl_only")) for row in rows),
            "sign_or_phase_sensitive": sum(_safe_bool(row.get("sign_or_phase_sensitive_present")) for row in rows),
            "state_conditioned_energy_injection": sum(
                _safe_bool(row.get("state_conditioned_energy_injection_witness_present"))
                or _safe_bool(row.get("state_conditioned_env_reward_witness_present"))
                for row in rows
            ),
        }
    return out


def _locomotion_counts(rows: list[dict[str, str]]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {rule: {"n": 0, "locomotion_witness": 0} for rule in RULES}
    for row in rows:
        rule = str(row.get("scope") or row.get("rule") or row.get("label"))
        if rule not in out:
            continue
        out[rule]["n"] += 1
        out[rule]["locomotion_witness"] += int(_safe_bool(row.get("sustained_locomotion_temporal_witness")))
    return out


def _feedback_counts(rows: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {
        rule: {"n": 0, "strict_feedback_count": 0, "profile_counts": Counter(), "gain_counts": Counter()}
        for rule in RULES
    }
    for row in rows:
        rule = str(row.get("rule") or row.get("scope") or row.get("label"))
        if rule not in out:
            continue
        out[rule]["n"] += 1
        strict = _safe_bool(row.get("strict_feedback_candidate"))
        out[rule]["strict_feedback_count"] += int(strict)
        if strict:
            out[rule]["profile_counts"][str(row.get("profile", ""))] += 1
            gain = row.get("best_gain") or row.get("strict_feedback_gain") or row.get("gain")
            out[rule]["gain_counts"][str(gain)] += 1
    for rule in RULES:
        out[rule]["profile_counts"] = dict(out[rule]["profile_counts"])
        out[rule]["gain_counts"] = dict(out[rule]["gain_counts"])
    return out


def _label_source_index_check(chunked_report: dict[str, Any]) -> dict[str, Any]:
    feedback_path = chunked_report.get("merged_outputs", {}).get("feedback_near_best", {}).get("path")
    rows = _read_csv(feedback_path) if feedback_path else []
    by_rule: dict[str, list[int]] = {rule: [] for rule in RULES}
    for row in rows:
        rule = str(row.get("rule"))
        if rule in by_rule:
            by_rule[rule].append(_safe_int(row.get("source_index"), -1))
    out: dict[str, Any] = {}
    for rule, values in by_rule.items():
        expected = list(range(int(chunked_report.get("source_counts", {}).get(rule, 0))))
        got = sorted(values)
        out[rule] = {
            "n": len(values),
            "min": min(values) if values else None,
            "max": max(values) if values else None,
            "contiguous_expected_range": got == expected,
        }
    return out


def _selector_axis_summary(selector_report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for rule in RULES:
        payload = selector_report.get("rules", {}).get(rule, {})
        band = payload.get("band_details", {})
        selected = payload.get("selected_summary", {})
        out[rule] = {
            "guard_pass": bool(payload.get("guard_pass")),
            "profile_guard_pass": bool(payload.get("profile_guard_pass")),
            "joint_profile_entropy_norm": selected.get("joint_profile_entropy_norm"),
            "joint_profile_active_cells": selected.get("joint_profile_active_cells"),
            "joint_profile_dominant_rate": selected.get("joint_profile_dominant_rate"),
            "axes": {
                axis: {
                    "source_count": band.get(axis, {}).get("source_count"),
                    "source_rate": band.get(axis, {}).get("source_rate"),
                    "selected_count": band.get(axis, {}).get("selected_count"),
                    "selected_rate": band.get(axis, {}).get("selected_rate"),
                    "min_count": band.get(axis, {}).get("min_count"),
                    "max_count": band.get(axis, {}).get("max_count"),
                    "passed": bool(band.get(axis, {}).get("passed")),
                }
                for axis in AXES
            },
            "strict_feedback": band.get("strict_feedback_subprofile_branch", {}),
            "pendulum_settle_yield_repair": band.get("pendulum_settle_yield_repair_branch", {}),
        }
    return out


def _future_basin_cached_label_summary(path_text: str | Path | None) -> dict[str, Any]:
    if path_text is None or str(path_text).strip() == "":
        return {
            "available": False,
            "hard_pass": True,
            "read": "No future-basin cached-label adapter was provided.",
        }
    path = Path(path_text).expanduser().resolve()
    if not path.exists():
        return {
            "available": False,
            "hard_pass": False,
            "read": f"Configured future-basin cached-label adapter does not exist: {path}",
        }
    report = _read_json(path)
    summaries = report.get("summaries", {})
    pendulum_ready = all(
        int(summaries.get(rule, {}).get("pendulum_label_available_count", 0)) >= 64 for rule in RULES
    )
    # MountainCar is explicitly coverage-only at this stage; absence is a note,
    # not a selector readiness failure.
    mountaincar_available = any(
        int(summaries.get(rule, {}).get("mountaincar_label_available_count", 0)) > 0 for rule in RULES
    )
    return {
        "available": True,
        "hard_pass": bool(pendulum_ready),
        "path": str(path),
        "summaries": summaries,
        "read": report.get("read", {}),
        "pendulum_label_available_for_selected_pool": bool(pendulum_ready),
        "mountaincar_guard_coverage_available": bool(mountaincar_available),
        "mountaincar_quota_ready": False,
    }


def _reference_label_summary(reference_dir: Path) -> dict[str, Any]:
    merged = reference_dir / "merged_labels"
    action = _action_counts(_read_json(merged / "action_mode_report.json"))
    locomotion = _locomotion_counts(_read_csv(merged / "locomotion_full_temporal_per_env.csv"))
    feedback = _feedback_counts(_read_csv(merged / "feedback_near_best_gain_per_env.csv"))
    out: dict[str, Any] = {}
    for rule in RULES:
        n = max(action.get(rule, {}).get("n", 0), locomotion.get(rule, {}).get("n", 0), feedback.get(rule, {}).get("n", 0))
        out[rule] = {
            "n": n,
            "locomotion_witness": locomotion.get(rule, {}).get("locomotion_witness", 0),
            "low_energy_not_ctrl_only": action.get(rule, {}).get("low_energy_not_ctrl_only", 0),
            "sign_or_phase_sensitive": action.get(rule, {}).get("sign_or_phase_sensitive", 0),
            "state_conditioned_energy_injection": action.get(rule, {}).get("state_conditioned_energy_injection", 0),
            "strict_feedback_count": feedback.get(rule, {}).get("strict_feedback_count", 0),
            "strict_feedback_profile_counts": feedback.get(rule, {}).get("profile_counts", {}),
            "strict_feedback_gain_counts": feedback.get(rule, {}).get("gain_counts", {}),
        }
        for axis in AXES + ("strict_feedback_count",):
            out[rule][f"{axis}_rate"] = _rate(int(out[rule][axis]), n)
    return out


def _cheap_reference_compare(selector_report: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    warnings: list[str] = []
    hard_failures: list[str] = []
    for rule in RULES:
        selected = selector_report.get("rules", {}).get(rule, {}).get("selected_summary", {})
        n = int(selected.get("n", 64))
        rule_out: dict[str, Any] = {"n": n, "axes": {}}
        for axis in AXES:
            cheap_count = int(selected.get(f"{axis}_count", 0))
            ref_count = int(reference.get(rule, {}).get(axis, 0))
            diff = ref_count - cheap_count
            warn = abs(diff) > 8
            if warn:
                warnings.append(f"{rule}:{axis}:cheap={cheap_count},reference={ref_count}")
            # If the reference detector sees more examples than the cheap
            # detector, the cheap label is conservative.  That is not semantic
            # corruption.  A hard failure is a non-low-energy axis collapsing
            # under the stronger selected-list reference check.
            if axis != "low_energy_not_ctrl_only" and diff < -8:
                hard_failures.append(f"{rule}:{axis}:cheap={cheap_count},reference={ref_count}")
            rule_out["axes"][axis] = {
                "cheap_selected_count": cheap_count,
                "reference_selected_count": ref_count,
                "diff_reference_minus_cheap": diff,
                "warn_abs_diff_gt_8": warn,
            }
        cheap_strict = int(selected.get("strict_feedback_count", 0))
        ref_strict = int(reference.get(rule, {}).get("strict_feedback_count", 0))
        rule_out["strict_feedback"] = {
            "cheap_selected_count": cheap_strict,
            "reference_selected_count": ref_strict,
            "diff_reference_minus_cheap": ref_strict - cheap_strict,
        }
        out[rule] = rule_out
    return {
        "rules": out,
        "warnings": warnings,
        "hard_failures": hard_failures,
        "low_energy_calibration_warning": any("low_energy_not_ctrl_only" in warning for warning in warnings),
        "hard_pass": not hard_failures,
    }


def _diversity_compare(source_report: dict[str, Any], smoke_report: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    source_div = source_report.get("transition_diversity_probe", {})
    for rule in RULES:
        src = source_div.get(rule, {})
        smk = smoke_report.get("scopes", {}).get(rule, {}).get("transition_diversity", {})
        rule_out: dict[str, Any] = {"ratios": {}, "hard_pass": True}
        for key in ("reward_raw_std", "state_delta_rms", "terminal_signal_std"):
            src_val = _safe_float(src.get(key), math.nan)
            smk_val = _safe_float(smk.get(key), math.nan)
            ratio = smk_val / src_val if math.isfinite(src_val) and src_val > 0 else math.nan
            pass_ratio = math.isfinite(ratio) and 0.5 <= ratio <= 2.5
            rule_out["ratios"][key] = {"source": src_val, "selected": smk_val, "ratio": ratio, "pass": pass_ratio}
            rule_out["hard_pass"] = bool(rule_out["hard_pass"] and pass_ratio)
        dim = smoke_report.get("scopes", {}).get(rule, {}).get("dimension_support", {})
        rule_out["dimension_support"] = dim
        rule_out["dimension_support_pass"] = all(int(dim.get(key, 0)) >= 4 for key in ("action_dim_unique", "obs_dim_unique", "state_dim_unique"))
        rule_out["hard_pass"] = bool(rule_out["hard_pass"] and rule_out["dimension_support_pass"])
        out[rule] = rule_out
    return out


def _parity_summary(parity_report: dict[str, Any]) -> dict[str, Any]:
    cases = []
    for case in parity_report.get("cases", []):
        cases.append(
            {
                "case": case.get("case"),
                "rule": case.get("rule"),
                "hard_pass": bool(case.get("hard_pass")),
                "collection_hard_pass": bool(case.get("collection_parity", {}).get("hard_pass")),
                "update_hard_pass": bool(case.get("update_parity", {}).get("hard_pass")),
                "n_envs": case.get("n_envs"),
                "n_steps": case.get("n_steps"),
                "selected_profile_version_counts": case.get("selected_profile_version_counts"),
                "selected_pendulum_settle_yield_repair_count": case.get(
                    "selected_pendulum_settle_yield_repair_count"
                ),
            }
        )
    return {"hard_pass": bool(parity_report.get("hard_pass")) and all(item["hard_pass"] for item in cases), "cases": cases}


def _repair_smoke_summary(selector_report: dict[str, Any], smoke_report: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    any_repair = False
    hard_pass = True
    for rule in RULES:
        band = selector_report.get("rules", {}).get(rule, {}).get("band_details", {})
        target = int(band.get("pendulum_settle_yield_repair_branch", {}).get("target_count", 0) or 0)
        scope = smoke_report.get("scopes", {}).get(rule, {})
        checks = scope.get("checks", {})
        any_repair = bool(any_repair or target > 0)
        rule_pass = True
        if target > 0:
            rule_pass = bool(
                scope.get("hard_pass")
                and checks.get("repair_count_matches")
                and checks.get("repair_is_prior_internal")
                and checks.get("repair_gap_positive")
            )
        hard_pass = bool(hard_pass and rule_pass)
        out[rule] = {
            "target_count": target,
            "expected_repair_count": scope.get("expected_repair_count"),
            "materialized_repair_count": scope.get("materialized_repair_count"),
            "repair_count_matches": checks.get("repair_count_matches"),
            "repair_is_prior_internal": checks.get("repair_is_prior_internal"),
            "repair_gap_positive": checks.get("repair_gap_positive"),
            "gap_active": scope.get("settle_gap", {}).get("gap_active", {}),
            "hard_pass": rule_pass,
        }
    return {"enabled": any_repair, "hard_pass": hard_pass, "rules": out}


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Profile Evidence Chain Audit",
        "",
        "This audit is read-only. It uses profile quality, health/diversity, and parity evidence; raw positive return is not a gate.",
        "",
        "## Verdict",
        "",
        f"- hard pass: `{report['hard_pass']}`",
        f"- current best tested pool: `{report['verdict']['best_tested_pool']}`",
        f"- default path requirement: `{report['verdict']['default_path_requirement']}`",
        f"- milestone readiness: `{report['verdict']['milestone_readiness']}`",
        "",
        "## Evidence Gates",
        "",
        "| gate | pass | note |",
        "|---|---:|---|",
    ]
    for gate in report["gates"]:
        lines.append(f"| {gate['name']} | `{gate['pass']}` | {gate['note']} |")
    lines.extend(["", "## Profile Coverage", ""])
    for rule in RULES:
        lines.extend([f"### {rule}", "", "| axis | source | selected | band pass |", "|---|---:|---:|---:|"])
        axes = report["profile_coverage"][rule]["axes"]
        for axis in AXES:
            item = axes[axis]
            lines.append(
                f"| `{axis}` | {item['source_count']} ({item['source_rate']:.3f}) | "
                f"{item['selected_count']} ({item['selected_rate']:.3f}) | `{item['passed']}` |"
            )
        strict = report["profile_coverage"][rule]["strict_feedback"]
        lines.append(
            f"| `strict_feedback_subprofile_branch` | {strict.get('source_strict_count')} | "
            f"{strict.get('selected_strict_count')} | `{strict.get('passed')}` |"
        )
        repair = report["profile_coverage"][rule].get("pendulum_settle_yield_repair", {})
        if repair:
            lines.append(
                f"| `pendulum_settle_yield_repair_branch` | {repair.get('eligible_count')} | "
                f"{repair.get('selected_count')}/{repair.get('target_count')} | `{repair.get('passed')}` |"
            )
        summary = report["profile_coverage"][rule]
        lines.append(
            f"\nJoint profile: active `{summary['joint_profile_active_cells']}`, "
            f"dominant rate `{summary['joint_profile_dominant_rate']:.3f}`, "
            f"entropy `{summary['joint_profile_entropy_norm']:.3f}`.\n"
        )
    if report.get("repair_smoke", {}).get("enabled"):
        lines.extend(["## Pendulum Repair Smoke", "", "| rule | target | materialized | internal | gap positive | gap active mean |", "|---|---:|---:|---:|---:|---:|"])
        for rule, item in report["repair_smoke"]["rules"].items():
            gap_mean = item.get("gap_active", {}).get("mean")
            lines.append(
                f"| {rule} | {item.get('target_count')} | {item.get('materialized_repair_count')} | "
                f"`{item.get('repair_is_prior_internal')}` | `{item.get('repair_gap_positive')}` | {gap_mean} |"
            )
    lines.extend(["## Diversity Vs Source Pool", "", "| rule | reward std ratio | state delta ratio | terminal ratio | dim pass |", "|---|---:|---:|---:|---:|"])
    for rule in RULES:
        item = report["diversity_vs_source"][rule]
        ratios = item["ratios"]
        lines.append(
            f"| {rule} | {ratios['reward_raw_std']['ratio']:.3f} | "
            f"{ratios['state_delta_rms']['ratio']:.3f} | "
            f"{ratios['terminal_signal_std']['ratio']:.3f} | `{item['dimension_support_pass']}` |"
        )
    lines.extend(["", "## Cheap Vs Reference Labels", ""])
    lines.append(f"- hard pass excluding low-energy calibration warning: `{report['cheap_reference_consistency']['hard_pass']}`")
    lines.append(f"- low-energy calibration warning: `{report['cheap_reference_consistency']['low_energy_calibration_warning']}`")
    for warning in report["cheap_reference_consistency"]["warnings"]:
        lines.append(f"- warning: `{warning}`")
    for failure in report["cheap_reference_consistency"]["hard_failures"]:
        lines.append(f"- hard failure: `{failure}`")
    lines.extend(["", "## Remaining Work", ""])
    for item in report["verdict"]["remaining_work"]:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_report = _read_json(args.source_report)
    chunked_report = _read_json(args.chunked_label_report)
    selector_report = _read_json(args.selector_report)
    smoke_report = _read_json(args.smoke_report)
    parity_report = _read_json(args.parity_report)
    reference = _reference_label_summary(Path(args.reference_label_dir))

    source_counts = source_report.get("selected_count_by_rule", {})
    source_pass = all(int(source_counts.get(rule, 0)) >= int(args.target_per_rule) for rule in RULES)
    source_efficiency_pass = float(args.source_wall_sec) <= float(args.source_wall_budget_sec)
    labels_cached_required = float(args.label_wall_sec) > 10.0 * max(1.0, float(args.source_wall_sec))
    label_source_index = _label_source_index_check(chunked_report)
    label_index_pass = all(item.get("contiguous_expected_range") for item in label_source_index.values())
    label_row_pass = all(int(chunked_report.get("source_counts", {}).get(rule, 0)) == int(args.target_per_rule) for rule in RULES)
    profile_coverage = _selector_axis_summary(selector_report)
    selector_pass = all(profile_coverage[rule]["guard_pass"] and profile_coverage[rule]["profile_guard_pass"] for rule in RULES)
    smoke_pass = bool(smoke_report.get("hard_pass"))
    diversity = _diversity_compare(source_report, smoke_report)
    diversity_pass = all(item["hard_pass"] for item in diversity.values())
    parity = _parity_summary(parity_report)
    repair_smoke = _repair_smoke_summary(selector_report, smoke_report)
    future_basin = _future_basin_cached_label_summary(args.future_basin_label_report)
    reference_compare = _cheap_reference_compare(selector_report, reference)

    gates = [
        {
            "name": "source pool yield",
            "pass": source_pass,
            "note": f"selected {source_counts}; raw_seen={source_report.get('sampling', {}).get('raw_seen')}",
        },
        {
            "name": "source pool time",
            "pass": source_efficiency_pass,
            "note": f"{args.source_wall_sec:.2f}s <= {args.source_wall_budget_sec:.2f}s budget",
        },
        {
            "name": "chunked label source_index merge",
            "pass": label_index_pass,
            "note": f"{label_source_index}",
        },
        {
            "name": "chunked label coverage",
            "pass": label_row_pass,
            "note": f"source_counts={chunked_report.get('source_counts')}; cached_required={labels_cached_required}",
        },
        {
            "name": "profile selector bands",
            "pass": selector_pass,
            "note": "full/q90 guard_pass and profile_guard_pass",
        },
        {
            "name": "integrated health/diversity smoke",
            "pass": smoke_pass and diversity_pass,
            "note": "finite transition stats and no diversity collapse vs source pool",
        },
        {
            "name": "pack-to-fit_model parity",
            "pass": parity["hard_pass"],
            "note": f"{len(parity['cases'])} cases",
        },
        {
            "name": "future-basin cached labels",
            "pass": future_basin["hard_pass"],
            "note": (
                "Pendulum labels available; MountainCar remains coverage-only"
                if future_basin.get("available")
                else str(future_basin.get("read"))
            ),
        },
        {
            "name": "prior-internal Pendulum repair smoke",
            "pass": repair_smoke["hard_pass"],
            "note": "disabled" if not repair_smoke["enabled"] else "repair count/internal/gap checks",
        },
        {
            "name": "selected64 reference-label sanity",
            "pass": reference_compare["hard_pass"],
            "note": "obs4 selected-list sanity; low-energy warning is tracked separately",
        },
    ]
    hard_pass = all(item["pass"] for item in gates)
    report = {
        "analysis_entry": "phase2_profile_evidence_chain_audit",
        "contract": {
            "read_only": True,
            "profile_quality_first": True,
            "raw_positive_return_not_a_gate": True,
            "does_not_sample_environments": True,
            "does_not_train": True,
            "does_not_modify_exact_scm": True,
            "does_not_modify_fit_model": True,
        },
        "inputs": {
            "source_report": str(Path(args.source_report).expanduser().resolve()),
            "chunked_label_report": str(Path(args.chunked_label_report).expanduser().resolve()),
            "selector_report": str(Path(args.selector_report).expanduser().resolve()),
            "smoke_report": str(Path(args.smoke_report).expanduser().resolve()),
            "parity_report": str(Path(args.parity_report).expanduser().resolve()),
            "reference_label_dir": str(Path(args.reference_label_dir).expanduser().resolve()),
            "future_basin_label_report": (
                ""
                if str(args.future_basin_label_report).strip() == ""
                else str(Path(args.future_basin_label_report).expanduser().resolve())
            ),
            "source_wall_sec": float(args.source_wall_sec),
            "label_wall_sec": float(args.label_wall_sec),
            "reference_label_wall_sec": float(args.reference_label_wall_sec),
        },
        "hard_pass": hard_pass,
        "outputs": {
            "json": str(Path(args.output_dir).expanduser().resolve() / "profile_evidence_chain_audit.json"),
            "markdown": str(Path(args.output_dir).expanduser().resolve() / "profile_evidence_chain_audit.md"),
        },
        "gates": gates,
        "source_pool": {
            "sampling": source_report.get("sampling", {}),
            "selected_count_by_rule": source_counts,
            "efficiency": {
                "source_wall_sec": float(args.source_wall_sec),
                "source_rss_kib": int(args.source_rss_kib),
                "source_efficiency_pass": source_efficiency_pass,
            },
        },
        "label_contract": {
            "source_counts": chunked_report.get("source_counts", {}),
            "chunk_count": chunked_report.get("chunk_count"),
            "merged_outputs": chunked_report.get("merged_outputs", {}),
            "source_index_check": label_source_index,
            "efficiency": {
                "label_wall_sec": float(args.label_wall_sec),
                "label_rss_kib": int(args.label_rss_kib),
                "reference_label_wall_sec": float(args.reference_label_wall_sec),
                "reference_rss_kib": int(args.reference_rss_kib),
                "cached_required_for_default_path": labels_cached_required,
            },
        },
        "profile_coverage": profile_coverage,
        "diversity_vs_source": diversity,
        "future_basin_cached_labels": future_basin,
        "repair_smoke": repair_smoke,
        "cheap_reference_consistency": reference_compare,
        "reference_labels": reference,
        "parity": parity,
        "verdict": {
            "best_tested_pool": "384 per rule source pool + cached/chunked labels + profile selector",
            "default_path_requirement": (
                "source generation can be online; profile labels must be cached or guarded cheaply, "
                "not recomputed as a full grid on every fit_model batch"
            ),
            "milestone_readiness": (
                "candidate-ready only if selector bands, integrated smoke, pack-to-fit parity, "
                "and prior-internal repair wiring all pass; low-energy remains protected/monitoring, not strengthened"
            ),
            "remaining_work": [
                "Calibrate low_energy_not_ctrl_only cheap-vs-reference drift before using it as a stronger quota.",
                "Keep MountainCar future-basin as coverage/guard telemetry until same-pool labels are cheap and stable.",
                "Pendulum settle repair may enter only as prior-internal materialization driven by prior-derived cached labels.",
                "Do not increase pool beyond 384/rule unless a new profile axis makes quota feasibility tight again.",
            ],
        },
    }
    out_dir = Path(args.output_dir).expanduser().resolve()
    _write_json(out_dir / "profile_evidence_chain_audit.json", report)
    (out_dir / "profile_evidence_chain_audit.md").write_text(_markdown(report), encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", default=str(DEFAULT_SOURCE_REPORT))
    parser.add_argument("--chunked-label-report", default=str(DEFAULT_CHUNKED_LABEL_REPORT))
    parser.add_argument("--selector-report", default=str(DEFAULT_SELECTOR_REPORT))
    parser.add_argument("--smoke-report", default=str(DEFAULT_SMOKE_REPORT))
    parser.add_argument("--parity-report", default=str(DEFAULT_PARITY_REPORT))
    parser.add_argument("--future-basin-label-report", default="")
    parser.add_argument("--reference-label-dir", default=str(DEFAULT_REFERENCE_LABEL_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--target-per-rule", type=int, default=384)
    parser.add_argument("--source-wall-sec", type=float, default=34.17)
    parser.add_argument("--source-rss-kib", type=int, default=1_748_600)
    parser.add_argument("--label-wall-sec", type=float, default=2314.36)
    parser.add_argument("--label-rss-kib", type=int, default=1_933_424)
    parser.add_argument("--reference-label-wall-sec", type=float, default=573.37)
    parser.add_argument("--reference-rss-kib", type=int, default=1_903_484)
    parser.add_argument("--source-wall-budget-sec", type=float, default=60.0)
    return parser


def main() -> None:
    report = run(build_parser().parse_args())
    print(
        json.dumps(
            _json_safe({"hard_pass": report["hard_pass"], "outputs": report["outputs"]}),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
