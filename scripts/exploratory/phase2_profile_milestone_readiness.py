#!/usr/bin/env python
"""Assess readiness of the profile-governance line for a new milestone.

Exploratory/read-only.  This script turns the current profile governance
artifacts into a sufficiency checklist.  It is deliberately conservative:
passing short-run trainability does not complete a condition unless the profile
axis itself is trusted and maintainable.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


ART = Path("/home/chen/RLPFN/artifacts")
DEFAULT_OUTPUT_DIR = ART / "phase2_profile_milestone_readiness_0511"
DEFAULT_GOVERNANCE_JSON = ART / "phase2_profile_governance_table_v2c_0511/profile_governance_table.json"
DEFAULT_BAND_SELECTOR_JSON = (
    ART / "phase2_profile_band_selector_v2c_centered_protected_feedback_safe_0511/profile_band_selector_report.json"
)
DEFAULT_FEEDBACK_CONTROL_JSON = (
    ART / "phase2_feedback_subprofile_control_audit_0511/feedback_subprofile_control_audit.json"
)
DEFAULT_FEEDBACK_BRANCH_JSON = (
    ART / "phase2_feedback_strict_subprofile_branch_v1_n32_0511/feedback_strict_subprofile_branch_report.json"
)
DEFAULT_FEEDBACK_BRANCH_SMOKE_JSON = (
    ART
    / "phase2_feedback_strict_subprofile_branch_v1_n32_sidecar_summary_seed9512_0511"
    / "pendulum_like_sidecar_delta_summary_report.json"
)
DEFAULT_MOUNTAINCAR_DETECTOR_JSON = (
    ART / "phase2_mountaincar_future_basin_detector_v1_0511/mountaincar_future_basin_detector_report.json"
)
DEFAULT_MOUNTAINCAR_ANCHOR_JSON = (
    ART / "phase2_gym_mountaincar_future_basin_anchor_0511/gym_mountaincar_future_basin_anchor_report.json"
)
DEFAULT_INTEGRATED_SMOKE_JSON = (
    ART
    / "phase2_profile_band_v2c_centered_protected_feedback_safe_delta_smoke_summary_0511"
    / "locomotion_guarded_smoke_summary.json"
)
DEFAULT_INTEGRATED_XSEED_SMOKE_JSON = (
    ART
    / "phase2_profile_band_v2c_centered_protected_feedback_safe_xseed_delta_smoke_summary_0511"
    / "locomotion_guarded_smoke_summary.json"
)


SUFFICIENCY_ITEMS = [
    {
        "id": "protected_milestones_preserved",
        "weight": 0.10,
        "definition": "milestone1 and milestone2 remain protected and are not reinterpreted as profile repairs",
    },
    {
        "id": "unified_profile_controller_exists",
        "weight": 0.15,
        "definition": "one reusable profile-band selector replaces one-wrapper-per-mode exploration",
    },
    {
        "id": "broad_profile_axes_quota_ready",
        "weight": 0.15,
        "definition": "low-energy, sign/phase, state-conditioned, and locomotion axes have clear trust roles and bands",
    },
    {
        "id": "joint_profile_balance_guarded",
        "weight": 0.10,
        "definition": "full/q90 candidates have high active-cell count and no dominant joint-cell collapse",
    },
    {
        "id": "health_diversity_guards_pass",
        "weight": 0.10,
        "definition": "candidate profile changes preserve state/reward/done/action diversity and raw/env trainability guards",
    },
    {
        "id": "feedback_strict_subprofile_control_ready",
        "weight": 0.18,
        "definition": "Pendulum-like strict feedback subprofile/gain collapse has a simple maintainable control and health guard",
    },
    {
        "id": "mountaincar_future_basin_detector_ready",
        "weight": 0.10,
        "definition": "upfront-investment/future-basin mode has a trusted detector before entering profile quotas",
    },
    {
        "id": "profile_candidate_cross_seed_or_longer_smoke",
        "weight": 0.07,
        "definition": "the resulting unified candidate is checked beyond one short fixed-list smoke",
    },
    {
        "id": "fit_model_parity_migration_ready",
        "weight": 0.05,
        "definition": "candidate milestone can be parity-migrated to fit_model without silent path divergence",
    },
]


def _read_json(path: str | Path) -> Any:
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    return value


def _score_item(
    item_id: str,
    governance: dict[str, Any] | None,
    selector: dict[str, Any] | None,
    feedback: dict[str, Any] | None,
    feedback_branch: dict[str, Any] | None = None,
    feedback_branch_smoke: dict[str, Any] | None = None,
    mountaincar_detector: dict[str, Any] | None = None,
    mountaincar_anchor: dict[str, Any] | None = None,
    integrated_smoke: dict[str, Any] | None = None,
    integrated_xseed_smoke: dict[str, Any] | None = None,
) -> tuple[float, str, list[str]]:
    gov = governance or {}
    sel = selector or {}
    fb = feedback or {}
    branch = feedback_branch or {}
    smoke = feedback_branch_smoke or {}
    mountaincar = mountaincar_detector or {}
    mc_anchor = mountaincar_anchor or {}
    int_smoke = integrated_smoke or {}
    int_xseed = integrated_xseed_smoke or {}
    versions = gov.get("profile_versions", {})
    axis_rows = gov.get("axis_rows", [])

    if item_id == "protected_milestones_preserved":
        ok = (
            versions.get("milestone1_sampled_topology_conditioned", {}).get("status") == "protected_milestone"
            and versions.get("milestone2_gated_reward_path_balance", {}).get("status") == "protected_milestone"
            and gov.get("contract", {}).get("does_not_modify_exact_scm") is True
        )
        return (1.0 if ok else 0.0, "protected" if ok else "missing", [])

    if item_id == "unified_profile_controller_exists":
        ok = (
            sel.get("analysis_entry") == "phase2_profile_band_selector"
            and sel.get("contract", {}).get("selection_wrapper_only") is True
            and sel.get("contract", {}).get("raw_delta_not_used_in_selection") is True
        )
        return (1.0 if ok else 0.0, "selector_ready" if ok else "selector_missing", [])

    if item_id == "broad_profile_axes_quota_ready":
        roles = {row["axis_id"]: row.get("trust_status") for row in axis_rows}
        ready = {
            "low_energy_weak_action_optimum": roles.get("low_energy_weak_action_optimum"),
            "sign_phase_sensitive_action": roles.get("sign_phase_sensitive_action"),
            "state_conditioned_energy_injection": roles.get("state_conditioned_energy_injection"),
            "locomotion_sustained_energy_tradeoff": roles.get("locomotion_sustained_energy_tradeoff"),
        }
        # State-conditioned is still broad/diagnostic, so this is not a full 1.
        score = 0.75 if all(ready.values()) else 0.0
        gaps = ["state_conditioned axis remains broad/diagnostic, not fully quota-ready"] if score else list(ready)
        return (score, "mostly_ready_with_one_broad_axis", gaps)

    if item_id == "joint_profile_balance_guarded":
        push = versions.get("profile_balanced_push2", {})
        profile = push.get("profile", {})
        full = profile.get("full", {})
        q90 = profile.get("q90", {})
        ok = (
            full.get("joint_active_cells", 0) >= 14
            and q90.get("joint_active_cells", 0) >= 14
            and float(full.get("joint_max_cell_share", 1.0)) <= 0.20
            and float(q90.get("joint_max_cell_share", 1.0)) <= 0.20
        )
        return (1.0 if ok else 0.4, "push2_joint_profile_guard_pass" if ok else "joint_profile_needs_work", [])

    if item_id == "health_diversity_guards_pass":
        integrated_ok = (
            int_smoke.get("analysis_entry") == "phase2_locomotion_guarded_smoke_summary"
            and int_smoke.get("overall", {}).get("candidate_sufficient_for_smoke") is True
            and all(
                scope.get("diversity_guard", {}).get("pass") is True
                and scope.get("sufficiency_read", {}).get("pass") is True
                for scope in (int_smoke.get("scopes") or {}).values()
            )
        )
        integrated_xseed_ok = (
            int_xseed.get("analysis_entry") == "phase2_locomotion_guarded_smoke_summary"
            and int_xseed.get("overall", {}).get("candidate_sufficient_for_smoke") is True
            and all(
                scope.get("diversity_guard", {}).get("pass") is True
                and scope.get("sufficiency_read", {}).get("pass") is True
                for scope in (int_xseed.get("scopes") or {}).values()
            )
        )
        if integrated_ok and integrated_xseed_ok:
            return (
                0.95,
                "two_seed_integrated_full_q90_delta_smoke_pass",
                ["still short smoke", "MountainCar prior fixed-list guard still missing"],
            )
        if integrated_ok:
            return (
                0.90,
                "integrated_full_q90_smoke_pass",
                ["still single seed/short smoke", "MountainCar prior fixed-list guard still missing"],
            )
        if int_smoke.get("analysis_entry") == "phase2_locomotion_guarded_smoke_summary":
            return (
                0.55,
                "integrated_full_q90_delta_smoke_failed",
                [
                    "post-pre delta mean or diversity guard is not met",
                    "delta-positive fraction is monitoring-only and not a gate",
                    "MountainCar prior fixed-list guard still missing",
                ],
            )
        push = versions.get("profile_balanced_push2", {})
        guards = push.get("guards", {})
        ok = all(guards.get(scope, {}).get("diversity_guard_pass") is True for scope in ("full", "q90"))
        smoke_overall = smoke.get("overall", {})
        smoke_raw = (smoke_overall.get("raw_reward_sum_delta") or {}).get("positive_fraction")
        smoke_env = (smoke_overall.get("reward_env_sum_delta") or {}).get("positive_fraction")
        feedback_branch_smoke_ok = (
            smoke.get("analysis_entry") == "phase2_pendulum_like_sidecar_delta_summary"
            and float(smoke_raw or 0.0) >= 0.60
            and float(smoke_env or 0.0) >= 0.60
        )
        # This is only for the current push2 profile, not future feedback/MountainCar changes.
        if ok and feedback_branch_smoke_ok:
            return (
                0.72,
                "current_candidate_plus_feedback_branch_smoke_pass",
                ["guards still not run for integrated full/q90+feedback candidate", "MountainCar prior guard still missing"],
            )
        return (0.65 if ok else 0.0, "current_candidate_guard_pass_only" if ok else "guard_missing", ["guards not yet run for feedback-subprofile or MountainCar changes"])

    if item_id == "feedback_strict_subprofile_control_ready":
        control = versions.get("strict_feedback_profile_gain_frontier", {}).get("subprofile", {}).get("control_audit", {})
        pressure = control.get("read", {}).get("feedback_subprofile_is_pressure_point") is True
        simple = control.get("simple_existing_scalar_knob_sufficient") is True
        branch_ok = (
            branch.get("analysis_entry") == "phase2_feedback_subprofile_branch_selector"
            and branch.get("selection", {}).get("guard_pass") is True
            and branch.get("read", {}).get("branch_control_reduces_constant_gain2_dominance") is True
        )
        smoke_overall = smoke.get("overall", {})
        smoke_ok = (
            smoke.get("analysis_entry") == "phase2_pendulum_like_sidecar_delta_summary"
            and float((smoke_overall.get("raw_reward_sum_delta") or {}).get("positive_fraction") or 0.0) >= 0.60
            and float((smoke_overall.get("reward_env_sum_delta") or {}).get("positive_fraction") or 0.0) >= 0.60
        )
        integrated_selector_ok = (
            sel.get("analysis_entry") == "phase2_profile_band_selector"
            and (sel.get("config") or {}).get("strict_feedback_subprofile_branch", {}).get("enabled") is True
            and all(
                (payload.get("band_details") or {})
                .get("strict_feedback_subprofile_branch", {})
                .get("passed")
                is True
                for payload in (sel.get("rules") or {}).values()
            )
            and all(payload.get("guard_pass") is True for payload in (sel.get("rules") or {}).values())
        )
        integrated_smoke_ok = (
            int_smoke.get("analysis_entry") == "phase2_locomotion_guarded_smoke_summary"
            and int_smoke.get("overall", {}).get("candidate_sufficient_for_smoke") is True
        )
        integrated_xseed_smoke_ok = (
            int_xseed.get("analysis_entry") == "phase2_locomotion_guarded_smoke_summary"
            and int_xseed.get("overall", {}).get("candidate_sufficient_for_smoke") is True
        )
        if (
            pressure
            and branch_ok
            and smoke_ok
            and integrated_selector_ok
            and integrated_smoke_ok
            and integrated_xseed_smoke_ok
        ):
            return (
                0.86,
                "integrated_full_q90_static_and_two_seed_smoke_pass",
                [
                    "source pool still imposes constant/gain-2.0 lower bound; generator-level or mixture control needed for stronger balance",
                    "MountainCar mode is not integrated into profile quotas yet",
                ],
            )
        if pressure and branch_ok and smoke_ok and integrated_selector_ok and integrated_smoke_ok:
            return (
                0.82,
                "integrated_full_q90_static_and_smoke_pass",
                [
                    "source pool still imposes constant/gain-2.0 lower bound; generator-level or mixture control needed for stronger balance",
                    "still needs cross-seed or longer guard before milestone",
                ],
            )
        if pressure and branch_ok and smoke_ok and integrated_selector_ok:
            if int_smoke.get("analysis_entry") == "phase2_locomotion_guarded_smoke_summary":
                return (
                    0.62,
                    "integrated_full_q90_static_pass_but_delta_smoke_failed",
                    [
                        "static profile branch is integrated, but full/q90 delta/diversity smoke is not sufficient",
                        "source pool still imposes constant/gain-2.0 lower bound; generator-level or mixture control may be needed",
                    ],
                )
            return (
                0.74,
                "integrated_full_q90_static_guard_plus_branch_smoke_pass",
                [
                    "integrated full/q90 smoke is running or pending",
                    "source pool still imposes constant/gain-2.0 lower bound; generator-level or mixture control needed for stronger balance",
                ],
            )
        if pressure and branch_ok and smoke_ok:
            return (
                0.58,
                "branch_control_materialized_short_smoke_pass",
                [
                    "only n32 and one smoke seed so far",
                    "not integrated into full/q90 profile selector",
                    "selection-only boundary means n64 needs generator-level or mixture control",
                ],
            )
        if pressure and branch_ok:
            return (
                0.45,
                "branch_control_materialized_needs_smoke",
                ["trusted pack smoke not yet passed", "not integrated into full/q90 profile selector"],
            )
        if pressure and not simple:
            return (0.25, "pressure_confirmed_but_control_missing", ["explicit subprofile branch/control still needed", "selection-only and scalar-knob fixes are insufficient"])
        if pressure and simple:
            return (0.65, "candidate_knob_found_needs_health_guard", ["health guard not run for knob-controlled candidate"])
        return (0.0, "not_assessed", ["feedback subprofile audit missing"])

    if item_id == "mountaincar_future_basin_detector_ready":
        axis = next((row for row in axis_rows if row["axis_id"] == "mountaincar_upfront_investment_future_basin"), {})
        if mountaincar.get("analysis_entry") == "phase2_mountaincar_future_basin_detector":
            quota_ready = mountaincar.get("read", {}).get("quota_ready") is True
            anchor_ok = (
                mc_anchor.get("analysis_entry") == "phase2_gym_mountaincar_future_basin_anchor"
                and mc_anchor.get("read", {}).get("gym_anchor_pass") is True
            )
            if quota_ready:
                return (0.60, "detector_quota_ready_needs_guard", ["fixed-list health/diversity guard still required"])
            if anchor_ok:
                return (
                    0.45,
                    "candidate_detector_gym_anchored_not_quota_ready",
                    ["needs prior fixed-list health/diversity validation", "needs selector integration before quota"],
                )
            return (
                0.25,
                "candidate_detector_defined_not_quota_ready",
                ["needs Gym-side MountainCar anchoring", "needs fixed-list health/diversity validation before quota"],
            )
        if axis.get("trust_status") == "missing_detector":
            return (0.0, "missing_detector", ["must define trusted upfront-investment/future-basin detector"])
        return (0.5, "detector_partial", ["detector still needs profile/health validation"])

    if item_id == "profile_candidate_cross_seed_or_longer_smoke":
        if (
            integrated_smoke
            and integrated_smoke.get("overall", {}).get("candidate_sufficient_for_smoke") is True
            and integrated_xseed_smoke
            and integrated_xseed_smoke.get("overall", {}).get("candidate_sufficient_for_smoke") is True
        ):
            return (
                0.65,
                "two_seed_integrated_full_q90_delta_smoke_pass",
                ["still needs longer fixed-list check only if this becomes a milestone candidate"],
            )
        if integrated_smoke and integrated_smoke.get("overall", {}).get("candidate_sufficient_for_smoke") is True:
            return (
                0.35,
                "single_integrated_full_q90_smoke_pass",
                ["needs cross-seed and/or slightly longer integrated fixed-list check"],
            )
        if integrated_smoke and integrated_smoke.get("analysis_entry") == "phase2_locomotion_guarded_smoke_summary":
            return (
                0.15,
                "integrated_short_delta_smoke_failed",
                ["cross-seed is not the blocker until the single-seed delta/diversity guard is repaired"],
            )
        if feedback_branch_smoke and feedback_branch_smoke.get("analysis_entry"):
            return (
                0.25,
                "single_short_smoke_plus_feedback_branch_smoke",
                ["still needs cross-seed and/or slightly longer integrated fixed-list check"],
            )
        return (0.2, "single_short_smoke_only", ["needs cross-seed and/or slightly longer fixed-list check after feedback/MountainCar decisions"])

    if item_id == "fit_model_parity_migration_ready":
        return (0.0, "not_started", ["only after a candidate milestone exists"])

    raise KeyError(item_id)


def build_report_from_loaded(
    governance: dict[str, Any] | None,
    selector: dict[str, Any] | None,
    feedback: dict[str, Any] | None,
    feedback_branch: dict[str, Any] | None = None,
    feedback_branch_smoke: dict[str, Any] | None = None,
    mountaincar_detector: dict[str, Any] | None = None,
    mountaincar_anchor: dict[str, Any] | None = None,
    integrated_smoke: dict[str, Any] | None = None,
    integrated_xseed_smoke: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rows = []
    weighted = 0.0
    for item in SUFFICIENCY_ITEMS:
        score, status, gaps = _score_item(
            item["id"],
            governance,
            selector,
            feedback,
            feedback_branch,
            feedback_branch_smoke,
            mountaincar_detector,
            mountaincar_anchor,
            integrated_smoke,
            integrated_xseed_smoke,
        )
        weighted += float(item["weight"]) * score
        rows.append({**item, "score": score, "status": status, "gaps": gaps})
    completion = weighted / sum(float(item["weight"]) for item in SUFFICIENCY_ITEMS)
    if completion < 0.50:
        band = "early_middle"
    elif completion < 0.70:
        band = "middle_not_milestone_ready"
    elif completion < 0.85:
        band = "near_candidate_requires_validation"
    else:
        band = "milestone_candidate_ready_for_parity"
    return {
        "analysis_entry": "phase2_profile_milestone_readiness",
        "contract": {
            "exploratory_only": True,
            "read_only_existing_artifacts": True,
            "does_not_sample_environments": True,
            "does_not_train": True,
            "does_not_modify_exact_scm": True,
            "does_not_modify_fit_model": True,
            "does_not_modify_ppo": True,
        },
        "inputs": {},
        "weighted_completion": completion,
        "completion_band": band,
        "sufficiency_items": rows,
        "necessary_next_items": [
            "continue profile-pressure repair while keeping post-pre delta mean and diversity as guards",
            "validate MountainCar upfront-investment/future-basin detector on prior fixed lists before any quota",
            "only after those guards pass, decide whether this is close enough for a third candidate milestone and fit_model parity migration",
        ],
    }


def build_report(
    governance_path: str | Path,
    selector_path: str | Path,
    feedback_path: str | Path,
    feedback_branch_path: str | Path,
    feedback_branch_smoke_path: str | Path,
    mountaincar_detector_path: str | Path,
    mountaincar_anchor_path: str | Path,
    integrated_smoke_path: str | Path,
    integrated_xseed_smoke_path: str | Path,
) -> dict[str, Any]:
    governance = _read_json(governance_path)
    selector = _read_json(selector_path)
    feedback = _read_json(feedback_path)
    feedback_branch = _read_json(feedback_branch_path)
    feedback_branch_smoke = _read_json(feedback_branch_smoke_path)
    mountaincar_detector = _read_json(mountaincar_detector_path)
    mountaincar_anchor = _read_json(mountaincar_anchor_path)
    integrated_smoke = _read_json(integrated_smoke_path)
    integrated_xseed_smoke = _read_json(integrated_xseed_smoke_path)
    report = build_report_from_loaded(
        governance,
        selector,
        feedback,
        feedback_branch,
        feedback_branch_smoke,
        mountaincar_detector,
        mountaincar_anchor,
        integrated_smoke,
        integrated_xseed_smoke,
    )
    report["inputs"] = {
        "governance_json": str(Path(governance_path).expanduser().resolve()),
        "selector_json": str(Path(selector_path).expanduser().resolve()),
        "feedback_control_json": str(Path(feedback_path).expanduser().resolve()),
        "feedback_branch_json": str(Path(feedback_branch_path).expanduser().resolve()),
        "feedback_branch_smoke_json": str(Path(feedback_branch_smoke_path).expanduser().resolve()),
        "mountaincar_detector_json": str(Path(mountaincar_detector_path).expanduser().resolve()),
        "mountaincar_anchor_json": str(Path(mountaincar_anchor_path).expanduser().resolve()),
        "integrated_smoke_json": str(Path(integrated_smoke_path).expanduser().resolve()),
        "integrated_xseed_smoke_json": str(
            Path(integrated_xseed_smoke_path).expanduser().resolve()
        ),
    }
    return report


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Profile Milestone Readiness",
        "",
        "Exploratory/read-only readiness assessment.  Scores estimate progress toward sufficient conditions; they are not acceptance criteria by themselves.",
        "",
        f"- weighted completion: `{100.0 * report['weighted_completion']:.1f}%`",
        f"- completion band: `{report['completion_band']}`",
        "",
        "| item | weight | score | status | main gaps |",
        "|---|---:|---:|---|---|",
    ]
    for row in report["sufficiency_items"]:
        lines.append(
            "| {id} | {w:.2f} | {s:.2f} | {status} | {gaps} |".format(
                id=row["id"],
                w=float(row["weight"]),
                s=float(row["score"]),
                status=row["status"],
                gaps="; ".join(row["gaps"]),
            )
        )
    lines.extend(["", "## Necessary Next Items", ""])
    for item in report["necessary_next_items"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Sources", ""])
    for key, value in report["inputs"].items():
        lines.append(f"- `{key}`: `{value}`")
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(
        args.governance_json,
        args.selector_json,
        args.feedback_control_json,
        args.feedback_branch_json,
        args.feedback_branch_smoke_json,
        args.mountaincar_detector_json,
        args.mountaincar_anchor_json,
        args.integrated_smoke_json,
        args.integrated_xseed_smoke_json,
    )
    json_path = out_dir / "profile_milestone_readiness.json"
    md_path = out_dir / "profile_milestone_readiness.md"
    report["outputs"] = {"json": str(json_path), "markdown": str(md_path)}
    json_path.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report["outputs"], indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--governance-json", default=str(DEFAULT_GOVERNANCE_JSON))
    parser.add_argument("--selector-json", default=str(DEFAULT_BAND_SELECTOR_JSON))
    parser.add_argument("--feedback-control-json", default=str(DEFAULT_FEEDBACK_CONTROL_JSON))
    parser.add_argument("--feedback-branch-json", default=str(DEFAULT_FEEDBACK_BRANCH_JSON))
    parser.add_argument("--feedback-branch-smoke-json", default=str(DEFAULT_FEEDBACK_BRANCH_SMOKE_JSON))
    parser.add_argument("--mountaincar-detector-json", default=str(DEFAULT_MOUNTAINCAR_DETECTOR_JSON))
    parser.add_argument("--mountaincar-anchor-json", default=str(DEFAULT_MOUNTAINCAR_ANCHOR_JSON))
    parser.add_argument("--integrated-smoke-json", default=str(DEFAULT_INTEGRATED_SMOKE_JSON))
    parser.add_argument(
        "--integrated-xseed-smoke-json",
        default=str(DEFAULT_INTEGRATED_XSEED_SMOKE_JSON),
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
