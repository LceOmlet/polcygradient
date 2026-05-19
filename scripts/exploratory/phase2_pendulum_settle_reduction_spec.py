#!/usr/bin/env python
"""Pendulum settle problem reduction spec.

This is a read-only governance artifact.  It reduces the current Pendulum
profile blocker to the smallest mechanism that is still semantically aligned
with the Gym Pendulum failure: a controllable quadratic settle factor.

The point is not to add another detector or quota.  The point is to define the
mechanism, its necessary/sufficient conditions, and the exact role of future
tests before touching the Exact SCM generator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ART = Path("/home/chen/RLPFN/artifacts")
DEFAULT_OUTPUT_DIR = ART / "phase2_pendulum_settle_reduction_spec_0512"
DEFAULT_CLOSURE_REGISTER_JSON = ART / "phase2_profile_closure_register_0512/profile_closure_register.json"
DEFAULT_REWARD_RELEVANT_AUDIT_JSON = (
    ART
    / "phase2_pendulum_reward_relevant_state_settle_audit_0512"
    / "pendulum_reward_relevant_state_settle_audit.json"
)


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def build_report(
    *,
    closure_register_json: str | Path,
    reward_relevant_audit_json: str | Path,
) -> dict[str, Any]:
    closure = _read_json(closure_register_json)
    reward_relevant = _read_json(reward_relevant_audit_json)
    disallowed = {str(item.get("name")): item for item in closure.get("disallowed_repairs", [])}
    allowed = {str(item.get("name")): item for item in closure.get("allowed_repairs", [])}
    return {
        "analysis_entry": "phase2_pendulum_settle_reduction_spec",
        "contract": {
            "read_only_existing_artifacts": True,
            "does_not_sample_environments": True,
            "does_not_train": True,
            "does_not_modify_exact_scm": True,
            "does_not_modify_fit_model": True,
            "does_not_modify_ppo": True,
            "profile_quality_first": True,
            "test_is_only_for_falsification": True,
            "not_a_raw_delta_gate": True,
        },
        "inputs": {
            "closure_register_json": str(Path(closure_register_json).expanduser().resolve()),
            "reward_relevant_audit_json": str(Path(reward_relevant_audit_json).expanduser().resolve()),
        },
        "current_evidence": {
            "reward_relevant_state_guard_ready": bool(
                reward_relevant.get("decision", {}).get("cheap_suffix_state_settle_guard_ready")
            ),
            "reward_relevant_state_min_count": int(reward_relevant.get("decision", {}).get("min_count", 0)),
            "rejected_repairs": {
                name: disallowed.get(name, {}).get("reason")
                for name in (
                    "raw_delta_or_raw_positive_quota",
                    "time_core_only_pendulum_quota",
                    "existing_signature_formula_quota",
                    "suffix_state_settle_quota_now",
                    "reward_relevant_state_settle_quota_now",
                    "simple_reference_inertia_highway_knob",
                )
                if name in disallowed
            },
            "still_allowed_repair": allowed.get("cheap_clean_pendulum_settle_guard", {}),
        },
        "reduction_decision": {
            "should_simplify_problem": True,
            "should_define_new_reduction_before_more_tests": True,
            "should_implement_generator_now": False,
            "why": (
                "Existing audits already falsify threshold-only, detector-only, scalar-knob, and simple "
                "inertia/highway repairs.  The next reliable step is to define a minimal mechanism whose "
                "semantics are necessary and sufficient for Pendulum-like settle before running more tests."
            ),
        },
        "mechanism": {
            "id": "controllable_quadratic_settle_factor",
            "name": "controllable quadratic settle factor",
            "definition": {
                "state_subspace": "z = [q, v], where q is a reward-relevant phase/position error and v is its velocity/momentum",
                "dynamics": "z_{t+1} = A z_t + B a_t + eps_t",
                "reward": "r_t = c - z_{t+1}^T Q z_{t+1} - lambda * ||a_t||^2",
                "controller_witness": "a_t = -K z_t, e.g. a small PD/LQR feedback controller",
                "basin": "high reward means q≈0 and v≈0; the action should decrease after the state enters the basin",
            },
            "why_simpler_than_conditional_basin": [
                "no discrete gate",
                "no exponential basin shape",
                "no special-case reward trigger",
                "analytic controllability and reward-state alignment checks are available",
                "the same factor can be labelled cheaply without a full feedback grid",
            ],
            "why_natural": [
                "local Pendulum dynamics are well approximated by phase error plus angular velocity",
                "many continuous-control stabilization tasks reduce to damped second-order control",
                "the profile target is closed-loop settling, not a Pendulum-specific reward hack",
            ],
        },
        "necessary_conditions": [
            {
                "id": "controllable_subspace",
                "condition": "B is nonzero and [B, A B] has rank 2, or paired action counterfactuals change future z",
                "failure_means": "the policy cannot use action to move into the reward basin",
            },
            {
                "id": "reward_state_alignment",
                "condition": "Q weights the same q/v subspace controlled by B",
                "failure_means": "reward asks for a basin that action cannot reliably reach",
            },
            {
                "id": "bounded_control_feasibility",
                "condition": "the witness controller reaches the basin without persistent action saturation",
                "failure_means": "the task becomes a saturation trick, not a settle mode",
            },
            {
                "id": "context_identifiability",
                "condition": "short context reveals action sign/gain and the q/v reward basin direction",
                "failure_means": "fit_model cannot infer the needed feedback law from context",
            },
            {
                "id": "profile_non_dominance",
                "condition": "the factor is a bounded subprofile and does not collapse broad profile bands",
                "failure_means": "the repair hurts profile diversity even if it helps Pendulum",
            },
        ],
        "sufficient_conditions_for_candidate": [
            {
                "id": "analytic_witness",
                "condition": "a bounded PD/LQR witness beats zero/random/open-loop on full and suffix reward",
            },
            {
                "id": "settle_witness",
                "condition": "witness prefix action magnitude exceeds suffix action magnitude and suffix reward-relevant drift contracts",
            },
            {
                "id": "guard_count_safe",
                "condition": "full and q90 fixed lists have enough clean guarded examples without loosening into non-worsening",
            },
            {
                "id": "health_diversity_safe",
                "condition": "state/reward/done/action diversity and existing protected profile bands do not regress",
            },
            {
                "id": "startup_safe",
                "condition": "fit_model-scale generation uses a cheap analytic/cached label, never a full feedback grid",
            },
        ],
        "minimal_test_plan": [
            {
                "stage": "analytic_preflight",
                "purpose": "verify controllability, reward-state alignment, action bound feasibility, and context identifiability without PPO",
                "stop_if_fails": True,
            },
            {
                "stage": "counterfactual_witness",
                "purpose": "compare zero/random/open-loop/PD on identical seeds to prove action-sensitive basin entry and suffix settle",
                "stop_if_fails": True,
            },
            {
                "stage": "profile_guard",
                "purpose": "measure full/q90 count-safe support and protected broad-profile diversity",
                "stop_if_fails": True,
            },
            {
                "stage": "short_pack_smoke",
                "purpose": "monitor PPO optimizability and delta mean only after the structural conditions pass",
                "stop_if_fails": False,
            },
        ],
        "decision": {
            "next_action": (
                "Implement only an exploratory wrapper/specimen generator for this factor, run analytic and "
                "counterfactual preflights first, and do not modify the milestone generator until the sufficient "
                "conditions above pass."
            ),
            "closed_now": False,
            "reason_not_closed": (
                "The simplified mechanism is defined, but no generated candidate has yet passed analytic, "
                "counterfactual, profile, and diversity guards."
            ),
        },
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Pendulum Settle Reduction Spec",
        "",
        "This report reduces the Pendulum profile blocker to the smallest reliable mechanism hypothesis.",
        "",
        "## Decision",
        "",
    ]
    for key, value in report["reduction_decision"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Mechanism", ""])
    mechanism = report["mechanism"]
    lines.append(f"- id: `{mechanism['id']}`")
    for key, value in mechanism["definition"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Necessary Conditions", ""])
    for item in report["necessary_conditions"]:
        lines.append(f"- `{item['id']}`: {item['condition']} Failure: {item['failure_means']}")
    lines.extend(["", "## Sufficient Conditions For Candidate", ""])
    for item in report["sufficient_conditions_for_candidate"]:
        lines.append(f"- `{item['id']}`: {item['condition']}")
    lines.extend(["", "## Minimal Test Plan", ""])
    for item in report["minimal_test_plan"]:
        lines.append(
            f"- `{item['stage']}`: {item['purpose']} stop_if_fails=`{item['stop_if_fails']}`"
        )
    lines.extend(["", "## Current Evidence", ""])
    lines.append(f"- reward-relevant state guard ready: `{report['current_evidence']['reward_relevant_state_guard_ready']}`")
    lines.append(f"- reward-relevant state min count: `{report['current_evidence']['reward_relevant_state_min_count']}`")
    for key, value in report["current_evidence"]["rejected_repairs"].items():
        lines.append(f"- rejected `{key}`: {value}")
    lines.extend(["", "## Files", ""])
    for key, value in report.get("outputs", {}).items():
        lines.append(f"- `{key}`: `{value}`")
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(
        closure_register_json=args.closure_register_json,
        reward_relevant_audit_json=args.reward_relevant_audit_json,
    )
    json_path = out_dir / "pendulum_settle_reduction_spec.json"
    md_path = out_dir / "pendulum_settle_reduction_spec.md"
    report["outputs"] = {"json": str(json_path), "markdown": str(md_path)}
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report["outputs"], sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--closure-register-json", default=str(DEFAULT_CLOSURE_REGISTER_JSON))
    parser.add_argument("--reward-relevant-audit-json", default=str(DEFAULT_REWARD_RELEVANT_AUDIT_JSON))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
