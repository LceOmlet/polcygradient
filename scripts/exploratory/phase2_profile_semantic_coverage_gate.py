#!/usr/bin/env python
"""Gate profile semantic coverage separately from engineering parity.

This is a read-only consolidation.  It deliberately treats PPO/fit_model
numeric parity, fixed-list profile smoke, and Gym-mode semantic coverage as
separate gates so a passing fixed-list candidate cannot be mistaken for a
fully covered generator-level milestone.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ART = Path("/home/chen/RLPFN/artifacts")
DEFAULT_OUTPUT_DIR = ART / "phase2_profile_semantic_coverage_gate_0512"
DEFAULT_SUFFICIENCY_JSON = (
    ART / "phase2_profile_fit_model_sufficiency_audit_0511/profile_fit_model_sufficiency_audit.json"
)
DEFAULT_V2E_READINESS_JSON = (
    ART / "phase2_profile_v2e_readiness_forecast_0511/profile_v2e_readiness_forecast.json"
)
DEFAULT_FEEDBACK_CLOSURE_JSON = (
    ART / "phase2_feedback_closure_preflight_0511/feedback_closure_preflight.json"
)
DEFAULT_FEEDBACK_SLACK_JSON = (
    ART / "phase2_feedback_support_slack_audit_decay_g1_obs4_xseed_0512/feedback_support_slack_audit.json"
)
DEFAULT_PENDULUM_ANCHOR_JSON = (
    ART / "phase2_pendulum_settle_anchor_preflight_0511/pendulum_settle_anchor_preflight.json"
)
DEFAULT_PENDULUM_PRIOR_JSON = (
    ART / "phase2_pendulum_settle_prior_coverage_audit_0511/pendulum_settle_prior_coverage_audit.json"
)
DEFAULT_MOUNTAINCAR_ANCHOR_JSON = (
    ART / "phase2_gym_mountaincar_future_basin_anchor_0511/gym_mountaincar_future_basin_anchor_report.json"
)
DEFAULT_MOUNTAINCAR_PRIOR_JSON = (
    ART / "phase2_profile_next_pressure_audit_v2d_prior_coverage_0511/profile_next_pressure_audit.json"
)


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def _read_json_optional(path: str | Path | None) -> Any:
    if path is None or str(path).strip() == "":
        return None
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        return None
    return _read_json(resolved)


def _feedback_slack_read(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return {
            "available": False,
            "has_selector_minimum": False,
            "has_slack": False,
            "gain_diversity_guard_pass": False,
            "read": "no feedback slack audit available",
        }
    decision = payload.get("decision", {}) or {}
    groups = payload.get("groups", {}) or {}
    ready = bool(decision.get("ready_to_clear_slack_blocker"))
    best_group = decision.get("best_group")
    if best_group is None and decision.get("ready_groups"):
        best_group = decision.get("ready_groups", [None])[0]
    if best_group is None and decision.get("near_but_not_slack_groups"):
        best_group = decision.get("near_but_not_slack_groups", [None])[0]
    if best_group is None and groups:
        best_group = next(iter(groups))
    group = groups.get(best_group, {}) if best_group else {}
    return {
        "available": True,
        "best_group": best_group,
        "has_selector_minimum": bool(group.get("all_meet_selector_minimum")),
        "has_slack": bool(group.get("all_meet_slack_target")),
        "gain_diversity_guard_pass": bool(group.get("gain_diversity_guard_pass")),
        "ready_to_clear_slack_blocker": ready,
        "read": decision.get("read", ""),
    }


def _mountaincar_read(anchor: dict[str, Any], prior: dict[str, Any]) -> dict[str, Any]:
    anchor_read = anchor.get("read", {}) or {}
    prior_guard = prior.get("mountaincar_prior_fixedlist_guard", {}) or {}
    gym_anchor_pass = bool(anchor_read.get("gym_anchor_pass", anchor_read.get("anchor_pass")))
    return {
        "gym_anchor_pass": gym_anchor_pass,
        "prior_guard_pass": bool(prior_guard.get("guard_pass")),
        "prior_scope_summaries": prior_guard.get("scope_summaries", {}),
        "read": prior_guard.get(
            "read",
            "MountainCar future-basin remains detector/coverage only until fixed-list guard passes.",
        ),
    }


def build_report(
    *,
    sufficiency_json: str | Path,
    v2e_readiness_json: str | Path,
    feedback_closure_json: str | Path,
    feedback_slack_json: str | Path | None,
    pendulum_anchor_json: str | Path,
    pendulum_prior_json: str | Path,
    mountaincar_anchor_json: str | Path,
    mountaincar_prior_json: str | Path,
) -> dict[str, Any]:
    sufficiency = _read_json(sufficiency_json)
    v2e = _read_json(v2e_readiness_json)
    feedback = _read_json(feedback_closure_json)
    feedback_slack = _feedback_slack_read(_read_json_optional(feedback_slack_json))
    pendulum_anchor = _read_json(pendulum_anchor_json)
    pendulum_prior = _read_json(pendulum_prior_json)
    mountaincar = _mountaincar_read(_read_json(mountaincar_anchor_json), _read_json(mountaincar_prior_json))

    engineering_closed = bool(
        sufficiency.get("score", {}).get("pipeline_score", 0.0) >= 1.0
        or sufficiency.get("decision", {}).get("seal_as_new_engineering_milestone_candidate")
    )
    fixed_list_profile_closed = bool(
        v2e.get("readiness", {}).get("candidate_profile_readiness_estimate", 0.0) >= 0.90
        and v2e.get("smoke_read", {}).get("candidate_smoke_guard_pass")
        and v2e.get("v2e_numeric_parity_read", {}).get("hard_pass")
    )
    generator_level_closed = bool(
        v2e.get("generator_level_gap_read", {}).get("generator_level_selector_ready")
        and v2e.get("fit_model_support_read", {}).get("fit_model_startup_option_present")
    )

    feedback_closed = bool(feedback.get("decision", {}).get("ready_to_claim_feedback_closure"))
    pendulum_anchor_closed = bool(pendulum_anchor.get("read", {}).get("closed_loop_settle_anchor_pass"))
    pendulum_prior_closed = bool(pendulum_prior.get("read", {}).get("quota_ready"))
    mountaincar_closed = bool(mountaincar["gym_anchor_pass"] and mountaincar["prior_guard_pass"])

    semantic_closed = bool(
        feedback_closed
        and pendulum_anchor_closed
        and pendulum_prior_closed
        and mountaincar_closed
    )
    sufficient_coverage_closed = bool(
        engineering_closed
        and fixed_list_profile_closed
        and generator_level_closed
        and semantic_closed
    )

    blockers: list[str] = []
    if not generator_level_closed:
        blockers.append("generator_level_selector_not_ready")
    if not feedback_closed:
        blockers.append("feedback_closure_not_ready")
    if not feedback_slack["ready_to_clear_slack_blocker"]:
        blockers.append("budgeted_feedback_startup_slack_not_safe")
    if not pendulum_anchor_closed:
        blockers.append("pendulum_closed_loop_settle_anchor_failed")
    if not pendulum_prior_closed:
        blockers.append("pendulum_settle_prior_coverage_sparse")
    if not mountaincar_closed:
        blockers.append("mountaincar_future_basin_prior_guard_sparse")

    return {
        "analysis_entry": "phase2_profile_semantic_coverage_gate",
        "contract": {
            "read_only_existing_artifacts": True,
            "does_not_sample_environments": True,
            "does_not_train": True,
            "does_not_modify_exact_scm": True,
            "does_not_modify_fit_model": True,
            "does_not_modify_ppo": True,
            "profile_quality_first": True,
            "delta_sign_fraction_not_a_gate": True,
        },
        "inputs": {
            "sufficiency_json": str(Path(sufficiency_json).expanduser().resolve()),
            "v2e_readiness_json": str(Path(v2e_readiness_json).expanduser().resolve()),
            "feedback_closure_json": str(Path(feedback_closure_json).expanduser().resolve()),
            "feedback_slack_json": None
            if feedback_slack_json is None
            else str(Path(feedback_slack_json).expanduser().resolve()),
            "pendulum_anchor_json": str(Path(pendulum_anchor_json).expanduser().resolve()),
            "pendulum_prior_json": str(Path(pendulum_prior_json).expanduser().resolve()),
            "mountaincar_anchor_json": str(Path(mountaincar_anchor_json).expanduser().resolve()),
            "mountaincar_prior_json": str(Path(mountaincar_prior_json).expanduser().resolve()),
        },
        "gates": {
            "engineering_parity_closed": engineering_closed,
            "fixed_list_profile_closed": fixed_list_profile_closed,
            "generator_level_closed": generator_level_closed,
            "semantic_gym_mode_closed": semantic_closed,
            "sufficient_coverage_closed": sufficient_coverage_closed,
        },
        "reads": {
            "engineering": {
                "pipeline_score": sufficiency.get("score", {}).get("pipeline_score"),
                "seal_as_candidate": sufficiency.get("decision", {}).get(
                    "seal_as_new_engineering_milestone_candidate"
                ),
            },
            "fixed_list_profile": {
                "readiness_estimate": v2e.get("readiness", {}).get("candidate_profile_readiness_estimate"),
                "smoke_guard_pass": v2e.get("smoke_read", {}).get("candidate_smoke_guard_pass"),
                "numeric_parity_hard_pass": v2e.get("v2e_numeric_parity_read", {}).get("hard_pass"),
            },
            "generator_level": v2e.get("generator_level_gap_read", {}),
            "feedback": {
                "closure_ready": feedback_closed,
                "closure_blockers": feedback.get("decision", {}).get("blockers", []),
                "slack": feedback_slack,
            },
            "pendulum_settle": {
                "anchor_pass": pendulum_anchor_closed,
                "anchor_interpretation": pendulum_anchor.get("read", {}).get("interpretation"),
                "prior_quota_ready": pendulum_prior_closed,
                "prior_min_settle_count": pendulum_prior.get("read", {}).get("min_settle_count"),
                "prior_max_settle_count": pendulum_prior.get("read", {}).get("max_settle_count"),
                "prior_mean_settle_rate": pendulum_prior.get("read", {}).get("mean_settle_rate"),
                "prior_interpretation": pendulum_prior.get("read", {}).get("interpretation"),
            },
            "mountaincar_future_basin": mountaincar,
        },
        "decision": {
            "blockers": blockers,
            "can_claim_third_candidate_milestone": sufficient_coverage_closed,
            "read": (
                "Engineering and fixed-list profile evidence are strong, but semantic Gym-mode coverage "
                "and generator-level startup are not closed."
                if not sufficient_coverage_closed
                else "All profile semantic coverage gates are closed."
            ),
            "next_steps": [
                "Implement a bounded online candidate-pool selector with cheap guards or cached labels; no fixed-list-only fit_model startup.",
                "Design a cheap prior-side sustained closed-loop settle guard; validate it without full feedback grid in the startup path.",
                "Keep MountainCar future-basin as an annotation until fixed-list prior guard and diversity checks pass.",
                "Use delta means only as health monitors; do not optimize selector quotas for raw-positive fraction.",
            ],
        },
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Profile Semantic Coverage Gate",
        "",
        "Read-only hard gate separating engineering parity from semantic Gym-mode coverage.",
        "",
        "## Verdict",
        "",
        f"- engineering parity closed: `{report['gates']['engineering_parity_closed']}`",
        f"- fixed-list profile closed: `{report['gates']['fixed_list_profile_closed']}`",
        f"- generator-level closed: `{report['gates']['generator_level_closed']}`",
        f"- semantic Gym-mode closed: `{report['gates']['semantic_gym_mode_closed']}`",
        f"- sufficient coverage closed: `{report['gates']['sufficient_coverage_closed']}`",
        f"- read: {report['decision']['read']}",
        "",
        "## Blockers",
        "",
    ]
    for blocker in report["decision"]["blockers"]:
        lines.append(f"- `{blocker}`")
    lines.extend(["", "## Key Reads", ""])
    pend = report["reads"]["pendulum_settle"]
    lines.append(
        "- Pendulum settle prior coverage: "
        f"quota=`{pend['prior_quota_ready']}`, min/max=`{pend['prior_min_settle_count']}`/"
        f"`{pend['prior_max_settle_count']}`, mean_rate=`{pend['prior_mean_settle_rate']}`"
    )
    lines.append(
        "- Pendulum anchor: "
        f"pass=`{pend['anchor_pass']}`, read={pend['anchor_interpretation']}"
    )
    fb = report["reads"]["feedback"]
    lines.append(
        "- Feedback closure: "
        f"ready=`{fb['closure_ready']}`, slack_ready=`{fb['slack']['ready_to_clear_slack_blocker']}`"
    )
    mc = report["reads"]["mountaincar_future_basin"]
    lines.append(
        "- MountainCar future-basin: "
        f"gym_anchor=`{mc['gym_anchor_pass']}`, prior_guard=`{mc['prior_guard_pass']}`"
    )
    lines.extend(["", "## Next Steps", ""])
    for item in report["decision"]["next_steps"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Files", ""])
    for key, value in report.get("outputs", {}).items():
        lines.append(f"- `{key}`: `{value}`")
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(
        sufficiency_json=args.sufficiency_json,
        v2e_readiness_json=args.v2e_readiness_json,
        feedback_closure_json=args.feedback_closure_json,
        feedback_slack_json=args.feedback_slack_json,
        pendulum_anchor_json=args.pendulum_anchor_json,
        pendulum_prior_json=args.pendulum_prior_json,
        mountaincar_anchor_json=args.mountaincar_anchor_json,
        mountaincar_prior_json=args.mountaincar_prior_json,
    )
    json_path = out_dir / "profile_semantic_coverage_gate.json"
    md_path = out_dir / "profile_semantic_coverage_gate.md"
    report["outputs"] = {"json": str(json_path), "markdown": str(md_path)}
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report["outputs"], sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sufficiency-json", default=str(DEFAULT_SUFFICIENCY_JSON))
    parser.add_argument("--v2e-readiness-json", default=str(DEFAULT_V2E_READINESS_JSON))
    parser.add_argument("--feedback-closure-json", default=str(DEFAULT_FEEDBACK_CLOSURE_JSON))
    parser.add_argument("--feedback-slack-json", default=str(DEFAULT_FEEDBACK_SLACK_JSON))
    parser.add_argument("--pendulum-anchor-json", default=str(DEFAULT_PENDULUM_ANCHOR_JSON))
    parser.add_argument("--pendulum-prior-json", default=str(DEFAULT_PENDULUM_PRIOR_JSON))
    parser.add_argument("--mountaincar-anchor-json", default=str(DEFAULT_MOUNTAINCAR_ANCHOR_JSON))
    parser.add_argument("--mountaincar-prior-json", default=str(DEFAULT_MOUNTAINCAR_PRIOR_JSON))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
