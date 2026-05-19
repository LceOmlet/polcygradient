#!/usr/bin/env python
"""Forecast remaining work to sufficient profile/Gym coverage.

Read-only consolidation.  This script separates three different notions that
are easy to conflate:

* fit_model/profile engineering coverage,
* selector-mode coverage,
* Gym semantic coverage.

It then proposes the next bounded repair step without over-optimizing one mode.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


ART = Path("/home/chen/RLPFN/artifacts")
DEFAULT_OUTPUT_DIR = ART / "phase2_profile_coverage_closure_forecast_0511"
DEFAULT_SUFFICIENCY_JSON = (
    ART / "phase2_profile_fit_model_sufficiency_audit_0511/profile_fit_model_sufficiency_audit.json"
)
DEFAULT_INVENTORY_JSON = (
    ART / "phase2_profile_complexity_mode_inventory_audit_0511/profile_complexity_mode_inventory_audit.json"
)
DEFAULT_PENDULUM_JSON = (
    ART / "phase2_pendulum_settle_anchor_preflight_0511/pendulum_settle_anchor_preflight.json"
)
DEFAULT_MOUNTAINCAR_ANCHOR_JSON = (
    ART / "phase2_gym_mountaincar_future_basin_anchor_0511/gym_mountaincar_future_basin_anchor_report.json"
)
DEFAULT_MOUNTAINCAR_DETECTOR_JSON = (
    ART / "phase2_mountaincar_future_basin_detector_v1_0511/mountaincar_future_basin_detector_report.json"
)
DEFAULT_FEEDBACK_BRANCH_JSON = (
    ART / "phase2_feedback_strict_subprofile_branch_v1_n32_0511/feedback_strict_subprofile_branch_report.json"
)

MODE_WEIGHTS = {
    "context_identifiable_closed_loop_settle_feedback": 0.22,
    "mountaincar_upfront_investment_future_basin": 0.18,
    "state_conditioned_energy_injection": 0.14,
    "locomotion_sustained_energy_tradeoff": 0.14,
    "terminal_survival_action_sensitive": 0.12,
    "sign_phase_sensitive_action": 0.11,
    "low_energy_weak_action_optimum": 0.09,
}

SEMANTIC_BASE_BY_MODE = {
    "low_energy_weak_action_optimum": 0.72,
    "sign_phase_sensitive_action": 0.55,
    "state_conditioned_energy_injection": 0.48,
    "locomotion_sustained_energy_tradeoff": 0.62,
    "context_identifiable_closed_loop_settle_feedback": 0.40,
    "mountaincar_upfront_investment_future_basin": 0.38,
    "terminal_survival_action_sensitive": 0.25,
}


def _read_json_optional(path: str | Path | None) -> Any:
    if path is None or str(path).strip() == "":
        return None
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        return None
    return json.loads(resolved.read_text(encoding="utf-8"))


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


def _weighted_average(items: list[tuple[float, float]]) -> float:
    denom = sum(weight for weight, _value in items)
    if denom <= 0:
        return 0.0
    return sum(weight * value for weight, value in items) / denom


def _inventory_rows(inventory: dict[str, Any]) -> list[dict[str, Any]]:
    return list(((inventory.get("mode_inventory", {}) or {}).get("rows", []) or []))


def _selector_mode_coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_mode = {str(row.get("mode_id")): row for row in rows}
    weighted = []
    per_mode = {}
    for mode, weight in MODE_WEIGHTS.items():
        row = by_mode.get(mode, {})
        value = 1.0 if row.get("selector_sufficient") is True else 0.0
        weighted.append((weight, value))
        per_mode[mode] = {"weight": weight, "selector_sufficient": bool(value), "credit": weight * value}
    score = _weighted_average(weighted)
    return {"score": score, "remaining": max(0.0, 1.0 - score), "per_mode": per_mode}


def _semantic_mode_coverage(
    rows: list[dict[str, Any]],
    *,
    pendulum: dict[str, Any],
    mountaincar_anchor: dict[str, Any],
    mountaincar_detector: dict[str, Any],
    feedback_branch: dict[str, Any],
) -> dict[str, Any]:
    by_mode = {str(row.get("mode_id")): row for row in rows}
    pendulum_read = ((pendulum.get("current_evidence", {}) or {}).get("current_read", {}) or {})
    mountaincar_anchor_read = mountaincar_anchor.get("read", {}) or {}
    mountaincar_detector_read = mountaincar_detector.get("read", {}) or {}
    branch_read = feedback_branch.get("read", {}) or {}

    per_mode: dict[str, Any] = {}
    weighted = []
    for mode, weight in MODE_WEIGHTS.items():
        row = by_mode.get(mode, {})
        value = float(SEMANTIC_BASE_BY_MODE.get(mode, 0.0))
        blockers: list[str] = []
        if mode == "context_identifiable_closed_loop_settle_feedback":
            if branch_read.get("branch_control_reduces_constant_gain2_dominance") is True:
                value += 0.08
            if "mode_is_not_clean_enough" in pendulum_read:
                blockers.append(str(pendulum_read["mode_is_not_clean_enough"]))
            if "pendulum_failure_location" in pendulum_read:
                blockers.append(str(pendulum_read["pendulum_failure_location"]))
        elif mode == "mountaincar_upfront_investment_future_basin":
            if mountaincar_anchor_read.get("gym_anchor_pass") is True:
                value += 0.12
            if mountaincar_detector_read.get("detector_candidate_exists") is True:
                value += 0.05
            if mountaincar_anchor_read.get("quota_ready") is not True:
                blockers.append(str(mountaincar_anchor_read.get("why_not_quota_ready", "not quota-ready")))
        elif mode == "terminal_survival_action_sensitive":
            blockers.append("diagnostic-only; do not fold into the current profile milestone")
        for gap in row.get("known_gaps", []) or []:
            if gap not in blockers:
                blockers.append(str(gap))
        value = max(0.0, min(1.0, value))
        weighted.append((weight, value))
        per_mode[mode] = {
            "weight": weight,
            "semantic_score": value,
            "selector_sufficient": bool(row.get("selector_sufficient")),
            "gym_semantic_sufficient": bool(row.get("gym_semantic_sufficient")),
            "blockers": blockers,
        }
    score = _weighted_average(weighted)
    return {"score": score, "remaining": max(0.0, 1.0 - score), "per_mode": per_mode}


def _coverage_buckets(
    *,
    sufficiency: dict[str, Any],
    inventory: dict[str, Any],
    selector_mode: dict[str, Any],
    semantic_mode: dict[str, Any],
) -> dict[str, Any]:
    suff_score = float((sufficiency.get("score", {}) or {}).get("score", 0.0) or 0.0)
    complexity = inventory.get("complexity", {}) or {}
    risk_score = float(complexity.get("risk_score", 1.0) or 1.0)
    anti_corruption = max(0.0, min(1.0, 1.0 - 0.5 * risk_score))
    scale_ready = 0.70
    remaining_work = sufficiency.get("remaining_work", []) or []
    if any("2048-scale" in str(item) for item in remaining_work):
        scale_ready -= 0.15
    if any("cache-by" in str(item) for item in remaining_work):
        scale_ready -= 0.10
    scale_ready = max(0.0, min(1.0, scale_ready))

    buckets = {
        "engineering_pipeline": {
            "weight": 0.30,
            "score": suff_score,
            "read": "fit_model/profile contract coverage",
        },
        "selector_mode_coverage": {
            "weight": 0.20,
            "score": selector_mode["score"],
            "read": "profile modes that can be represented in selector guards",
        },
        "gym_semantic_coverage": {
            "weight": 0.30,
            "score": semantic_mode["score"],
            "read": "mode definitions anchored enough to claim Gym-like semantic coverage",
        },
        "scale_execution_readiness": {
            "weight": 0.10,
            "score": scale_ready,
            "read": "2048-scale chunk/cache execution readiness",
        },
        "anti_corruption_controls": {
            "weight": 0.10,
            "score": anti_corruption,
            "read": "complexity risk after parity/checklist mitigations",
        },
    }
    overall = _weighted_average([(item["weight"], item["score"]) for item in buckets.values()])
    return {"overall_score": overall, "remaining": max(0.0, 1.0 - overall), "buckets": buckets}


def _remaining_milestones(semantic_mode: dict[str, Any]) -> list[dict[str, Any]]:
    feedback_score = semantic_mode["per_mode"]["context_identifiable_closed_loop_settle_feedback"]["semantic_score"]
    mountaincar_score = semantic_mode["per_mode"]["mountaincar_upfront_investment_future_basin"]["semantic_score"]
    terminal_score = semantic_mode["per_mode"]["terminal_survival_action_sensitive"]["semantic_score"]
    return [
        {
            "id": "feedback_closed_loop_settle_closure",
            "priority": 1,
            "current_score": feedback_score,
            "target_score": 0.72,
            "estimated_coverage_gain": max(0.0, 0.22 * (0.72 - feedback_score)),
            "bounded_next_step": (
                "Run one feedback closure pass: subprofile/gain guard plus Pendulum/Reacher settle anchor; "
                "require selector parity and pack-to-fit_model parity."
            ),
            "stop_condition": "if cross-seed support slack or settle anchor fails, stop feedback work and re-rank",
        },
        {
            "id": "mountaincar_future_basin_quota_readiness",
            "priority": 2,
            "current_score": mountaincar_score,
            "target_score": 0.70,
            "estimated_coverage_gain": max(0.0, 0.18 * (0.70 - mountaincar_score)),
            "bounded_next_step": (
                "Add prior fixed-list health/diversity validation and selector integration for the anchored "
                "MountainCar detector."
            ),
            "stop_condition": "do not quota if candidate rate remains too sparse or health/diversity fails",
        },
        {
            "id": "terminal_survival_scope_decision",
            "priority": 3,
            "current_score": terminal_score,
            "target_score": 0.50,
            "estimated_coverage_gain": max(0.0, 0.12 * (0.50 - terminal_score)),
            "bounded_next_step": "Decide explicit exclusion vs diagnostic guard; do not mix survival reward into v2 profile.",
            "stop_condition": "if no clean Gym anchor emerges, keep diagnostic-only",
        },
        {
            "id": "scale_cache_execution",
            "priority": 4,
            "current_score": 0.45,
            "target_score": 0.75,
            "estimated_coverage_gain": 0.03,
            "bounded_next_step": "Implement/use frozen_h fingerprint cache and run a partial multi-chunk execution smoke.",
            "stop_condition": "if cache misses dominate, fix cache before larger 2048 run",
        },
        {
            "id": "longer_cross_seed_trainability",
            "priority": 5,
            "current_score": 0.40,
            "target_score": 0.65,
            "estimated_coverage_gain": 0.025,
            "bounded_next_step": "After semantic closure, run short cross-seed fit_model probes before any long train.",
            "stop_condition": "if pack parity or early trainability regresses, revert to contract/debug stage",
        },
    ]


def build_report(
    *,
    sufficiency_json: str | Path,
    inventory_json: str | Path,
    pendulum_json: str | Path,
    mountaincar_anchor_json: str | Path,
    mountaincar_detector_json: str | Path,
    feedback_branch_json: str | Path,
) -> dict[str, Any]:
    sufficiency = _read_json_optional(sufficiency_json) or {}
    inventory = _read_json_optional(inventory_json) or {}
    pendulum = _read_json_optional(pendulum_json) or {}
    mountaincar_anchor = _read_json_optional(mountaincar_anchor_json) or {}
    mountaincar_detector = _read_json_optional(mountaincar_detector_json) or {}
    feedback_branch = _read_json_optional(feedback_branch_json) or {}
    rows = _inventory_rows(inventory)
    selector_mode = _selector_mode_coverage(rows)
    semantic_mode = _semantic_mode_coverage(
        rows,
        pendulum=pendulum,
        mountaincar_anchor=mountaincar_anchor,
        mountaincar_detector=mountaincar_detector,
        feedback_branch=feedback_branch,
    )
    coverage = _coverage_buckets(
        sufficiency=sufficiency,
        inventory=inventory,
        selector_mode=selector_mode,
        semantic_mode=semantic_mode,
    )
    milestones = _remaining_milestones(semantic_mode)
    sufficient_threshold = 0.78
    return {
        "analysis_entry": "phase2_profile_coverage_closure_forecast",
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
            "sufficiency_json": str(Path(sufficiency_json).expanduser().resolve()),
            "inventory_json": str(Path(inventory_json).expanduser().resolve()),
            "pendulum_json": str(Path(pendulum_json).expanduser().resolve()),
            "mountaincar_anchor_json": str(Path(mountaincar_anchor_json).expanduser().resolve()),
            "mountaincar_detector_json": str(Path(mountaincar_detector_json).expanduser().resolve()),
            "feedback_branch_json": str(Path(feedback_branch_json).expanduser().resolve()),
        },
        "coverage": coverage,
        "selector_mode_coverage": selector_mode,
        "gym_semantic_mode_coverage": semantic_mode,
        "remaining_milestones": milestones,
        "forecast": {
            "distance_to_full_coverage": coverage["remaining"],
            "distance_to_sufficient_threshold": max(0.0, sufficient_threshold - coverage["overall_score"]),
            "likely_bounded_iterations": 3,
            "next_step": milestones[0],
            "sufficient_coverage_threshold": sufficient_threshold,
            "read": (
                "engineering coverage is complete enough to protect the path; "
                "sufficient Gym-mode coverage still needs feedback closure, MountainCar quota readiness, "
                "and scale/cache execution."
            ),
        },
    }


def _markdown(report: dict[str, Any]) -> str:
    coverage = report["coverage"]
    forecast = report["forecast"]
    lines = [
        "# Profile Coverage Closure Forecast",
        "",
        f"- overall coverage forecast: `{coverage['overall_score']:.3f}`",
        f"- distance to sufficient threshold: `{forecast['distance_to_sufficient_threshold']:.3f}`",
        f"- distance to full coverage: `{forecast['distance_to_full_coverage']:.3f}`",
        f"- likely bounded iterations: `{forecast['likely_bounded_iterations']}`",
        f"- read: {forecast['read']}",
        "",
        "## Buckets",
        "",
        "| bucket | weight | score | read |",
        "|---|---:|---:|---|",
    ]
    for name, item in coverage["buckets"].items():
        lines.append(f"| `{name}` | {item['weight']:.2f} | {item['score']:.3f} | {item['read']} |")
    lines.extend(["", "## Remaining Milestones", "", "| priority | milestone | current | target | est gain | next |", "|---:|---|---:|---:|---:|---|"])
    for item in report["remaining_milestones"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(item["priority"]),
                    f"`{item['id']}`",
                    f"{float(item['current_score']):.3f}",
                    f"{float(item['target_score']):.3f}",
                    f"{float(item['estimated_coverage_gain']):.3f}",
                    item["bounded_next_step"],
                ]
            )
            + " |"
        )
    next_step = forecast["next_step"]
    lines.extend(
        [
            "",
            "## Immediate Next Step",
            "",
            f"- `{next_step['id']}`",
            f"- {next_step['bounded_next_step']}",
            f"- stop condition: {next_step['stop_condition']}",
            "",
            "## Files",
            "",
        ]
    )
    for key, value in report.get("outputs", {}).items():
        lines.append(f"- `{key}`: `{value}`")
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(
        sufficiency_json=args.sufficiency_json,
        inventory_json=args.inventory_json,
        pendulum_json=args.pendulum_json,
        mountaincar_anchor_json=args.mountaincar_anchor_json,
        mountaincar_detector_json=args.mountaincar_detector_json,
        feedback_branch_json=args.feedback_branch_json,
    )
    json_path = out_dir / "profile_coverage_closure_forecast.json"
    md_path = out_dir / "profile_coverage_closure_forecast.md"
    report["outputs"] = {"json": str(json_path), "markdown": str(md_path)}
    json_path.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report["outputs"], sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sufficiency-json", default=str(DEFAULT_SUFFICIENCY_JSON))
    parser.add_argument("--inventory-json", default=str(DEFAULT_INVENTORY_JSON))
    parser.add_argument("--pendulum-json", default=str(DEFAULT_PENDULUM_JSON))
    parser.add_argument("--mountaincar-anchor-json", default=str(DEFAULT_MOUNTAINCAR_ANCHOR_JSON))
    parser.add_argument("--mountaincar-detector-json", default=str(DEFAULT_MOUNTAINCAR_DETECTOR_JSON))
    parser.add_argument("--feedback-branch-json", default=str(DEFAULT_FEEDBACK_BRANCH_JSON))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
