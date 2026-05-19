#!/usr/bin/env python
"""Audit profile-mainline complexity and profile-mode definition pressure.

Read-only consolidation.  This script does not run PPO, does not train, and
does not sample environments.  It answers two questions:

1. Is the profile-to-fit_model mainline becoming too complex to trust?
2. Which profile modes are sufficiently defined, and which mode should be fixed
   next without over-optimizing a single mode?
"""

from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path
from typing import Any


ART = Path("/home/chen/RLPFN/artifacts")
REPO = Path(__file__).resolve().parents[2]

DEFAULT_OUTPUT_DIR = ART / "phase2_profile_complexity_mode_inventory_audit_0511"
DEFAULT_GOVERNANCE_JSON = ART / "phase2_profile_governance_table_v2c_0511/profile_governance_table.json"
DEFAULT_MODE_TABLE_JSON = ART / "phase2_mode_coverage_table_v2e_0511/mode_coverage_table_v2e.json"
DEFAULT_SUFFICIENCY_JSON = (
    ART
    / "phase2_profile_fit_model_sufficiency_audit_0511/profile_fit_model_sufficiency_audit.json"
)
DEFAULT_FEEDBACK_BUDGETED_JSON = (
    ART
    / "phase2_profile_online_candidate_pool_feedback_support_generator_seed9602_0511"
    / "feedback_budgeted_support_guard/feedback_budgeted_support_guard.json"
)
DEFAULT_FEEDBACK_CONTROL_JSON = (
    ART / "phase2_feedback_subprofile_control_audit_0511/feedback_subprofile_control_audit.json"
)
DEFAULT_FEEDBACK_BROADEN_JSON = (
    ART / "phase2_feedback_source_pool_broaden_audit_0511/feedback_source_pool_broaden_audit.json"
)
DEFAULT_FEEDBACK_HISTORY_JSON = (
    ART / "phase2_feedback_history_recall_audit_0511/feedback_history_recall_audit.json"
)
DEFAULT_GENERATOR_GAP_JSON = (
    ART
    / "phase2_profile_v2e_generator_level_gap_audit_0511/profile_v2e_generator_level_gap_audit.json"
)
DEFAULT_MAINLINE_FILES = (
    REPO / "ticl/cli_parsing.py",
    REPO / "ticl/model_configs.py",
    REPO / "ticl/rlpfn_maintained_path.py",
    REPO / "ticl/train.py",
)
DEFAULT_PROFILE_FILES = (
    REPO / "scripts/exploratory/phase2_profile_online_selector_preflight.py",
    REPO / "scripts/exploratory/phase2_profile_chunked_label_contract.py",
    REPO / "scripts/exploratory/phase2_profile_band_selector.py",
    REPO / "scripts/exploratory/phase2_feedback_budgeted_support_guard.py",
    REPO / "scripts/exploratory/phase2_profile_fit_model_sufficiency_audit.py",
)

MODE_ID_TO_KEY = {
    "low_energy_weak_action_optimum": "low_energy_not_ctrl_only",
    "sign_phase_sensitive_action": "sign_or_phase_sensitive",
    "state_conditioned_energy_injection": "state_conditioned_energy_injection",
    "locomotion_sustained_energy_tradeoff": "locomotion_witness",
    "context_identifiable_closed_loop_settle_feedback": "strict_feedback_candidate",
    "mountaincar_upfront_investment_future_basin": None,
    "terminal_survival_action_sensitive": None,
}


def _read_json_optional(path: str | Path | None) -> Any:
    if path is None or str(path).strip() == "":
        return None
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        return None
    return json.loads(resolved.read_text(encoding="utf-8"))


def _read_text_optional(path: str | Path | None) -> str:
    if path is None or str(path).strip() == "":
        return ""
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        return ""
    return resolved.read_text(encoding="utf-8", errors="replace")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def _file_metrics(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    text = _read_text_optional(resolved)
    lines = text.splitlines()
    nonempty = [line for line in lines if line.strip()]
    metrics: dict[str, Any] = {
        "path": str(resolved),
        "available": bool(text),
        "line_count": len(lines),
        "nonempty_line_count": len(nonempty),
        "function_count": 0,
        "class_count": 0,
        "max_function_lines": 0,
        "parse_ok": False,
    }
    if not text:
        return metrics
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return metrics
    metrics["parse_ok"] = True
    function_lengths = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            metrics["function_count"] += 1
            end = getattr(node, "end_lineno", node.lineno)
            function_lengths.append(max(1, int(end) - int(node.lineno) + 1))
        elif isinstance(node, ast.ClassDef):
            metrics["class_count"] += 1
    metrics["max_function_lines"] = max(function_lengths or [0])
    return metrics


def _complexity_audit(
    *,
    mainline_files: list[str | Path],
    profile_files: list[str | Path],
    sufficiency: dict[str, Any],
    generator_gap: dict[str, Any],
) -> dict[str, Any]:
    mainline_metrics = [_file_metrics(path) for path in mainline_files]
    profile_metrics = [_file_metrics(path) for path in profile_files]
    suff_checks = sufficiency.get("checks", {}) if isinstance(sufficiency, dict) else {}
    parity_pass = bool(suff_checks.get("pack_to_fit_model_numeric_parity"))
    startup_pass = bool(suff_checks.get("fit_model_fixed_csv_startup_pass"))
    chunk_pass = bool(suff_checks.get("chunked_label_path_ready"))
    all_critical_pass = bool(parity_pass and startup_pass and chunk_pass)
    train_metrics = next((item for item in mainline_metrics if item["path"].endswith("/ticl/train.py")), {})
    large_train_function = int(train_metrics.get("max_function_lines", 0) or 0) > 1000
    profile_script_count = sum(1 for item in profile_metrics if item.get("available"))
    generator_ready = bool(
        (generator_gap.get("generator_level_read", {}) or {}).get("generator_level_selector_ready")
    )

    risk_points = []
    if large_train_function:
        risk_points.append(
            "fixed-list startup touches a very large train orchestration function; local tests can miss branch coupling"
        )
    if profile_script_count >= 5:
        risk_points.append(
            "profile logic is spread across several exploratory scripts; contracts must stay machine-readable"
        )
    if not generator_ready:
        risk_points.append(
            "some profile labels still originate from offline detectors; generator-level parity must be preserved explicitly"
        )
    if not all_critical_pass:
        risk_points.append("one or more fit_model/profile parity gates are missing")

    mitigations = []
    if parity_pass:
        mitigations.append("pack-to-fit_model numeric parity hard-passes on full/q90 smoke")
    if startup_pass:
        mitigations.append("fit_model fixed CSV startup smoke hard-passes")
    if chunk_pass:
        mitigations.append("chunked label path and selector semantic parity hard-pass")
    if bool(suff_checks.get("explicit_no_silent_fallback")):
        mitigations.append("no-silent-fallback contract is explicit")

    risk_score = 0.0
    risk_score += 0.25 if large_train_function else 0.05
    risk_score += 0.20 if profile_script_count >= 5 else 0.05
    risk_score += 0.20 if not generator_ready else 0.05
    risk_score += 0.25 if not all_critical_pass else 0.05
    risk_score = min(1.0, risk_score)
    if all_critical_pass and risk_score > 0.55:
        risk_score = 0.55
    return {
        "mainline_files": mainline_metrics,
        "profile_files": profile_metrics,
        "large_train_function": large_train_function,
        "profile_script_count": profile_script_count,
        "risk_score": risk_score,
        "risk_level": "high" if risk_score >= 0.75 else "moderate" if risk_score >= 0.4 else "low",
        "risk_points": risk_points,
        "mitigations": mitigations,
        "read": (
            "complexity is controlled by parity/checklist gates, but the train.py integration should stay thin"
            if all_critical_pass
            else "complexity is not yet controlled because critical parity/checklist gates are missing"
        ),
        "recommendations": [
            "keep fit_model mainline changes limited to CLI/config/loading; keep detectors in read-only scripts",
            "do not add new mode-specific branches to train.py; add a bounded artifact/CSV contract instead",
            "for any mode-definition change, require selector parity plus pack-to-fit_model parity before long runs",
            "promote cache-by-frozen_h fingerprint before attempting 2048-scale label execution",
        ],
    }


def _merge_mode_rows(governance: dict[str, Any], mode_table: dict[str, Any]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for row in governance.get("axis_rows", []) or []:
        axis_id = str(row.get("axis_id") or "")
        if axis_id:
            by_id[axis_id] = dict(row)
    for row in mode_table.get("rows", []) or []:
        mode_id = str(row.get("mode_id") or "")
        if not mode_id:
            continue
        merged = dict(by_id.get(mode_id, {}))
        merged.update({f"mode_table_{key}": value for key, value in row.items()})
        merged.setdefault("axis_id", mode_id)
        by_id[mode_id] = merged
    return [by_id[key] for key in sorted(by_id)]


def _feedback_signals(
    *,
    feedback_budgeted: dict[str, Any],
    feedback_control: dict[str, Any],
    feedback_broaden: dict[str, Any],
    feedback_history: dict[str, Any],
) -> dict[str, Any]:
    budget_rules = feedback_budgeted.get("rules", {}) or {}
    min_candidate = min(
        [int((budget_rules.get(rule, {}) or {}).get("candidate_count", 0) or 0) for rule in budget_rules]
        or [0]
    )
    control_read = feedback_control.get("read", {}) or {}
    broaden_read = feedback_broaden.get("read", {}) or {}
    history_read = feedback_history.get("read", {}) or {}
    return {
        "budgeted_min_candidate_count": min_candidate,
        "budgeted_rules": budget_rules,
        "subprofile_is_pressure_point": bool(control_read.get("feedback_subprofile_is_pressure_point")),
        "single_existing_scalar_knob_sufficient": bool(control_read.get("single_existing_scalar_knob_sufficient")),
        "strict_source_pool_collapse_confirmed": bool(broaden_read.get("strict_source_pool_collapse_confirmed")),
        "core_or_weak_relaxation_sufficient": bool(broaden_read.get("core_or_weak_relaxation_sufficient")),
        "broad_feedback_memory_supported": bool(history_read.get("broad_feedback_memory_supported")),
        "narrow_official_trainable_feedback_memory_partly_supported": bool(
            history_read.get("narrow_official_trainable_feedback_memory_partly_supported")
        ),
    }


def _mode_sufficiency(row: dict[str, Any], feedback: dict[str, Any]) -> dict[str, Any]:
    mode_id = str(row.get("axis_id") or row.get("mode_table_mode_id") or "")
    trust = str(row.get("trust_status") or row.get("mode_table_status") or "")
    definition = str(row.get("definition") or "")
    profile_key = row.get("profile_key")
    if profile_key in (None, ""):
        profile_key = MODE_ID_TO_KEY.get(mode_id)

    sufficient_for_selector = False
    sufficient_for_gym = False
    pressure = 0.35
    known_gaps: list[str] = []
    next_action = "preserve and monitor"

    if mode_id == "low_energy_weak_action_optimum":
        sufficient_for_selector = True
        pressure = 0.20
        next_action = "preserve band; do not tune as a feedback fix"
    elif mode_id == "sign_phase_sensitive_action":
        sufficient_for_selector = True
        pressure = 0.45
        known_gaps.append("necessary broad witness, too coarse for closed-loop feedback quality")
        next_action = "keep protected; only refine through feedback/closed-loop subtype"
    elif mode_id == "state_conditioned_energy_injection":
        sufficient_for_selector = True
        pressure = 0.60
        known_gaps.append("can include immediate-reward shortcuts; delayed future-basin subtype remains separate")
        next_action = "preserve as broad band; repair MountainCar/future-basin separately"
    elif mode_id == "locomotion_sustained_energy_tradeoff":
        sufficient_for_selector = True
        pressure = 0.50
        known_gaps.append("quota-ready only with joint-profile guard; historically scarce in milestone2")
        next_action = "preserve raised band and joint-cell cap; avoid locomotion-only repair"
    elif mode_id == "context_identifiable_closed_loop_settle_feedback":
        pressure = 0.92
        sufficient_for_selector = bool(feedback.get("budgeted_min_candidate_count", 0) >= 8)
        known_gaps.extend(
            [
                "budgeted startup guard is narrow decay/gain=1 support, not closed-loop settle sufficiency",
                "strict source pool collapse and gain/profile dominance are confirmed",
                "broad feedback memory is not supported by history audit",
            ]
        )
        if feedback.get("single_existing_scalar_knob_sufficient"):
            known_gaps.append("a simple scalar knob may exist, but this conflicts with current control audit expectation")
        next_action = (
            "one bounded repair pass: subprofile/gain guard plus Pendulum/Reacher settle anchor, then stop and re-rank"
        )
    elif mode_id == "mountaincar_upfront_investment_future_basin":
        pressure = 0.88
        known_gaps.append("annotation-only and not quota-ready; Gym anchoring/fixed-list health guard still missing")
        next_action = "after one bounded feedback pass, anchor detector on Gym and prior fixed-list health"
    elif mode_id == "terminal_survival_action_sensitive":
        pressure = 0.62
        known_gaps.append("diagnostic-only; done mismatch was outlier-concentrated and not a clean reward-shaping candidate")
        next_action = "keep diagnostic; do not fold into current profile milestone"

    if "not sufficient" in trust:
        sufficient_for_gym = False
    elif "diagnostic" in trust or "not quota" in trust or "annotation" in trust:
        sufficient_for_gym = False
    elif sufficient_for_selector and mode_id in {
        "low_energy_weak_action_optimum",
        "locomotion_sustained_energy_tradeoff",
    }:
        sufficient_for_gym = False
        known_gaps.append("selector witness is not a full Gym success proof")
    elif sufficient_for_selector:
        sufficient_for_gym = False

    return {
        "mode_id": mode_id,
        "profile_key": profile_key,
        "gym_targets": row.get("gym_targets") or row.get("mode_table_gym_targets"),
        "definition": definition or row.get("mode_table_status"),
        "trust_status": trust,
        "selector_sufficient": sufficient_for_selector,
        "gym_semantic_sufficient": sufficient_for_gym,
        "pressure_score": pressure,
        "known_gaps": known_gaps,
        "recommended_next_action": next_action,
        "mode_table_status": row.get("mode_table_status"),
        "mode_table_decision": row.get("mode_table_decision"),
        "mode_table_full_read": row.get("mode_table_full_read"),
        "mode_table_q90_read": row.get("mode_table_q90_read"),
    }


def _mode_inventory(
    *,
    governance: dict[str, Any],
    mode_table: dict[str, Any],
    feedback_budgeted: dict[str, Any],
    feedback_control: dict[str, Any],
    feedback_broaden: dict[str, Any],
    feedback_history: dict[str, Any],
) -> dict[str, Any]:
    feedback = _feedback_signals(
        feedback_budgeted=feedback_budgeted,
        feedback_control=feedback_control,
        feedback_broaden=feedback_broaden,
        feedback_history=feedback_history,
    )
    rows = [_mode_sufficiency(row, feedback) for row in _merge_mode_rows(governance, mode_table)]
    rows.sort(key=lambda item: (-float(item["pressure_score"]), str(item["mode_id"])))
    max_pressure = rows[0] if rows else None
    high_pressure = [row for row in rows if float(row["pressure_score"]) >= 0.75]
    return {
        "feedback_signals": feedback,
        "rows": rows,
        "highest_pressure_mode": max_pressure,
        "high_pressure_modes": high_pressure,
        "read": (
            "feedback is the highest active selector pressure; MountainCar is the largest unclosed Gym-mode pressure"
            if rows
            and max_pressure
            and max_pressure.get("mode_id") == "context_identifiable_closed_loop_settle_feedback"
            else "mode pressure should be re-ranked before choosing another single-mode repair"
        ),
        "anti_overfit_plan": [
            "spend at most one bounded iteration on the highest active pressure mode",
            "define pass/fail by cross-seed support, selector parity, and pack-to-fit_model parity, not by a single delta metric",
            "after that iteration, re-run this inventory before touching another mode",
            "do not add hard quotas for annotation-only modes until Gym anchoring and fixed-list health pass",
        ],
    }


def build_report(
    *,
    governance_json: str | Path,
    mode_table_json: str | Path,
    sufficiency_json: str | Path,
    feedback_budgeted_json: str | Path,
    feedback_control_json: str | Path,
    feedback_broaden_json: str | Path,
    feedback_history_json: str | Path,
    generator_gap_json: str | Path,
    mainline_files: list[str | Path],
    profile_files: list[str | Path],
) -> dict[str, Any]:
    governance = _read_json_optional(governance_json) or {}
    mode_table = _read_json_optional(mode_table_json) or {}
    sufficiency = _read_json_optional(sufficiency_json) or {}
    feedback_budgeted = _read_json_optional(feedback_budgeted_json) or {}
    feedback_control = _read_json_optional(feedback_control_json) or {}
    feedback_broaden = _read_json_optional(feedback_broaden_json) or {}
    feedback_history = _read_json_optional(feedback_history_json) or {}
    generator_gap = _read_json_optional(generator_gap_json) or {}
    complexity = _complexity_audit(
        mainline_files=mainline_files,
        profile_files=profile_files,
        sufficiency=sufficiency,
        generator_gap=generator_gap,
    )
    inventory = _mode_inventory(
        governance=governance,
        mode_table=mode_table,
        feedback_budgeted=feedback_budgeted,
        feedback_control=feedback_control,
        feedback_broaden=feedback_broaden,
        feedback_history=feedback_history,
    )
    highest = inventory.get("highest_pressure_mode") or {}
    return {
        "analysis_entry": "phase2_profile_complexity_mode_inventory_audit",
        "contract": {
            "read_only_existing_artifacts": True,
            "does_not_run_ppo": True,
            "does_not_train": True,
            "does_not_sample_environments": True,
            "does_not_modify_exact_scm": True,
            "does_not_modify_fit_model": True,
            "does_not_modify_ppo": True,
        },
        "inputs": {
            "governance_json": str(Path(governance_json).expanduser().resolve()),
            "mode_table_json": str(Path(mode_table_json).expanduser().resolve()),
            "sufficiency_json": str(Path(sufficiency_json).expanduser().resolve()),
            "feedback_budgeted_json": str(Path(feedback_budgeted_json).expanduser().resolve()),
            "feedback_control_json": str(Path(feedback_control_json).expanduser().resolve()),
            "feedback_broaden_json": str(Path(feedback_broaden_json).expanduser().resolve()),
            "feedback_history_json": str(Path(feedback_history_json).expanduser().resolve()),
            "generator_gap_json": str(Path(generator_gap_json).expanduser().resolve()),
            "mainline_files": [str(Path(path).expanduser().resolve()) for path in mainline_files],
            "profile_files": [str(Path(path).expanduser().resolve()) for path in profile_files],
        },
        "complexity": complexity,
        "mode_inventory": inventory,
        "decision": {
            "mainline_complexity_acceptance": (
                "acceptable_with_guards"
                if complexity["risk_level"] in {"low", "moderate"}
                else "too_risky_without_refactor"
            ),
            "next_mode_to_fix": highest.get("mode_id"),
            "next_mode_reason": highest.get("known_gaps", []),
            "do_not_overfix_single_mode": True,
        },
    }


def _markdown(report: dict[str, Any]) -> str:
    complexity = report["complexity"]
    inventory = report["mode_inventory"]
    highest = inventory.get("highest_pressure_mode") or {}
    lines = [
        "# Profile Complexity / Mode Inventory Audit",
        "",
        f"- complexity risk: `{complexity['risk_level']}` (`{complexity['risk_score']:.2f}`)",
        f"- complexity read: {complexity['read']}",
        f"- next mode to fix: `{highest.get('mode_id')}`",
        f"- inventory read: {inventory['read']}",
        "",
        "## Complexity Risks",
        "",
    ]
    for item in complexity["risk_points"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Mitigations", ""])
    for item in complexity["mitigations"]:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Mode Inventory",
            "",
            "| mode | selector sufficient | gym sufficient | pressure | next |",
            "|---|---|---|---:|---|",
        ]
    )
    for row in inventory["rows"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['mode_id']}`",
                    f"`{row['selector_sufficient']}`",
                    f"`{row['gym_semantic_sufficient']}`",
                    f"{float(row['pressure_score']):.2f}",
                    str(row["recommended_next_action"]),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Highest Pressure Gaps", ""])
    for item in highest.get("known_gaps", []):
        lines.append(f"- {item}")
    lines.extend(["", "## Anti-overfit Plan", ""])
    for item in inventory["anti_overfit_plan"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Files", ""])
    for key, value in report.get("outputs", {}).items():
        lines.append(f"- `{key}`: `{value}`")
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(
        governance_json=args.governance_json,
        mode_table_json=args.mode_table_json,
        sufficiency_json=args.sufficiency_json,
        feedback_budgeted_json=args.feedback_budgeted_json,
        feedback_control_json=args.feedback_control_json,
        feedback_broaden_json=args.feedback_broaden_json,
        feedback_history_json=args.feedback_history_json,
        generator_gap_json=args.generator_gap_json,
        mainline_files=args.mainline_file,
        profile_files=args.profile_file,
    )
    json_path = out_dir / "profile_complexity_mode_inventory_audit.json"
    md_path = out_dir / "profile_complexity_mode_inventory_audit.md"
    report["outputs"] = {"json": str(json_path), "markdown": str(md_path)}
    json_path.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report["outputs"], sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--governance-json", default=str(DEFAULT_GOVERNANCE_JSON))
    parser.add_argument("--mode-table-json", default=str(DEFAULT_MODE_TABLE_JSON))
    parser.add_argument("--sufficiency-json", default=str(DEFAULT_SUFFICIENCY_JSON))
    parser.add_argument("--feedback-budgeted-json", default=str(DEFAULT_FEEDBACK_BUDGETED_JSON))
    parser.add_argument("--feedback-control-json", default=str(DEFAULT_FEEDBACK_CONTROL_JSON))
    parser.add_argument("--feedback-broaden-json", default=str(DEFAULT_FEEDBACK_BROADEN_JSON))
    parser.add_argument("--feedback-history-json", default=str(DEFAULT_FEEDBACK_HISTORY_JSON))
    parser.add_argument("--generator-gap-json", default=str(DEFAULT_GENERATOR_GAP_JSON))
    parser.add_argument("--mainline-file", action="append", default=[str(path) for path in DEFAULT_MAINLINE_FILES])
    parser.add_argument("--profile-file", action="append", default=[str(path) for path in DEFAULT_PROFILE_FILES])
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
