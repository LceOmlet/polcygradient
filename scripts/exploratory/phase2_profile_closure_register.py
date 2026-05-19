#!/usr/bin/env python
"""Register remaining profile-coverage blockers and allowed repairs.

This is a read-only governance artifact.  It turns the current pile of profile
audits into a small decision table so the next work stays on the profile
coverage mainline instead of drifting toward raw-delta tuning or expensive
full-grid probes in the default generator path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ART = Path("/home/chen/RLPFN/artifacts")
DEFAULT_OUTPUT_DIR = ART / "phase2_profile_closure_register_0512"
DEFAULT_CLOSURE_PLAN_JSON = (
    ART / "phase2_profile_closure_progress_plan_0512/profile_closure_progress_plan.json"
)
DEFAULT_LIGHT_GUARD_JSON = (
    ART / "phase2_pendulum_settle_light_guard_audit_0512/pendulum_settle_light_guard_audit.json"
)
DEFAULT_H_FEATURE_JSON = (
    ART / "phase2_pendulum_settle_h_feature_relation_audit_0512/pendulum_settle_h_feature_relation_audit.json"
)
DEFAULT_GUARD_SEARCH_JSON = (
    ART / "phase2_pendulum_settle_guard_search_0512/pendulum_settle_guard_search.json"
)
DEFAULT_STATE_LABEL_JSON = (
    ART / "phase2_pendulum_settle_state_label_preflight_0512/pendulum_settle_state_label_preflight.json"
)
DEFAULT_STATE_BLOCKER_JSON = (
    ART / "phase2_pendulum_settle_state_blocker_decomposition_0512/pendulum_settle_state_blocker_decomposition.json"
)
DEFAULT_REWARD_RELEVANT_STATE_JSON = (
    ART
    / "phase2_pendulum_reward_relevant_state_settle_audit_0512"
    / "pendulum_reward_relevant_state_settle_audit.json"
)
DEFAULT_REFERENCE_INERTIA_HIGHWAY_JSON = (
    ART
    / "phase2_pendulum_settle_state_label_inertia_alpha025_highway050_n32_s128_0512"
    / "pendulum_settle_state_label_preflight.json"
)
DEFAULT_FEEDBACK_POOL_JSON = (
    ART / "phase2_feedback_source_pool_broaden_audit_0511/feedback_source_pool_broaden_audit.json"
)
DEFAULT_FEEDBACK_CONTROL_JSON = (
    ART / "phase2_feedback_subprofile_control_audit_0511/feedback_subprofile_control_audit.json"
)
DEFAULT_ONLINE_PREFLIGHT_JSON = (
    ART / "phase2_profile_online_selector_preflight_0511/profile_online_selector_preflight.json"
)


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def _safe_get(payload: dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = payload
    for key in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    return default if cur is None else cur


def build_report(
    *,
    closure_plan_json: str | Path,
    light_guard_json: str | Path,
    h_feature_json: str | Path,
    guard_search_json: str | Path,
    state_label_json: str | Path,
    state_blocker_json: str | Path,
    reward_relevant_state_json: str | Path,
    reference_inertia_highway_json: str | Path,
    feedback_pool_json: str | Path,
    feedback_control_json: str | Path,
    online_preflight_json: str | Path,
) -> dict[str, Any]:
    closure = _read_json(closure_plan_json)
    light = _read_json(light_guard_json)
    h_feature = _read_json(h_feature_json)
    guard_search = _read_json(guard_search_json)
    state_label = _read_json(state_label_json)
    state_blocker = _read_json(state_blocker_json)
    reward_relevant_state = _read_json(reward_relevant_state_json)
    reference_inertia_highway = _read_json(reference_inertia_highway_json)
    pool = _read_json(feedback_pool_json)
    control = _read_json(feedback_control_json)
    online = _read_json(online_preflight_json)

    progress = closure.get("progress", {})
    diagnostics = closure.get("diagnostics", {})
    stage_ready = diagnostics.get("planned_stage_ready", {})
    feedback_read = pool.get("read", {})
    feedback_control_read = control.get("read", {})
    feedback_budget = online.get("feedback_eval_budget", {})
    if not feedback_budget:
        feedback_budget = _safe_get(online, "bounded_generator_plan", "feedback_eval_budget", default={})

    disallowed_repairs = [
        {
            "name": "raw_delta_or_raw_positive_quota",
            "reason": "raw delta is a monitor for health, not a profile selector objective",
            "evidence": "closure contract marks delta_sign_fraction_not_a_gate",
        },
        {
            "name": "time_core_only_pendulum_quota",
            "reason": "wide but impure; it over-admits non-settle cases",
            "evidence": light.get("decision", {}).get("broad_but_impure_guards", []),
        },
        {
            "name": "single_existing_h_scalar_bias",
            "reason": "no stable existing h split explains clean settle yield",
            "evidence": h_feature.get("decision", {}).get("read"),
        },
        {
            "name": "existing_signature_formula_quota",
            "reason": "bounded formulas over current signature fields are either clean and sparse or broad and impure",
            "evidence": {
                "cheap_formula_guard_ready": guard_search.get("decision", {}).get("cheap_formula_guard_ready"),
                "best_formula": guard_search.get("decision", {}).get("best_formula"),
                "read": guard_search.get("decision", {}).get("read"),
            },
        },
        {
            "name": "suffix_state_settle_quota_now",
            "reason": "the new suffix-state settle label is semantically closer to Pendulum but has zero count-safe support",
            "evidence": {
                "ready": state_label.get("decision", {}).get("cheap_suffix_state_settle_guard_ready"),
                "min_count": state_label.get("decision", {}).get("min_count"),
                "read": state_label.get("decision", {}).get("read"),
            },
        },
        {
            "name": "loose_state_non_worsening_quota",
            "reason": "count-safe only after weakening settle into non-worsening, which is not the Pendulum closed-loop settle anchor",
            "evidence": {
                "loose_count_safe": state_blocker.get("decision", {}).get("loose_non_worsening_count_safe_under_grid"),
                "strict_contraction_count_safe": state_blocker.get("decision", {}).get("strict_contraction_count_safe_under_grid"),
                "selector_threshold_repair_allowed": state_blocker.get("decision", {}).get("selector_threshold_repair_allowed"),
            },
        },
        {
            "name": "reward_relevant_state_settle_quota_now",
            "reason": (
                "reward-relevant state drift is a better offline audit than full-state drift, but it still has "
                "zero count-safe support; this rejects the hypothesis that the blocker was only irrelevant-state noise"
            ),
            "evidence": {
                "ready": reward_relevant_state.get("decision", {}).get("cheap_suffix_state_settle_guard_ready"),
                "min_count": reward_relevant_state.get("decision", {}).get("min_count"),
                "read": reward_relevant_state.get("decision", {}).get("read"),
            },
        },
        {
            "name": "simple_reference_inertia_highway_knob",
            "reason": (
                "a valid vectorized-runtime screen with reference inertia, alpha=0.25, and state_highway_lambda=0.5 "
                "still leaves strict full/q90 settle support at zero; this simple transition knob does not solve "
                "the state-basin blocker"
            ),
            "evidence": {
                "ready": reference_inertia_highway.get("decision", {}).get("cheap_suffix_state_settle_guard_ready"),
                "min_count": reference_inertia_highway.get("decision", {}).get("min_count"),
                "read": reference_inertia_highway.get("decision", {}).get("read"),
            },
        },
        {
            "name": "full_feedback_grid_in_fit_model_startup",
            "reason": "full feedback grid is an offline audit; default startup must use cheap guard or cached/chunked labels",
            "evidence": {
                "default_label_mode": feedback_budget.get("label_mode"),
                "default_eval_units": feedback_budget.get("eval_units"),
                "blocked_by_default": feedback_budget.get("execution_blocked_by_default"),
                "read": feedback_budget.get("read", "feedback budget must remain bounded"),
            },
        },
        {
            "name": "mountaincar_quota_now",
            "reason": "Gym anchor exists, but prior detector support is still sparse; keep as annotation/guard",
            "evidence": _safe_get(closure, "diagnostics", "mountaincar_progress_parts", default={}),
        },
    ]

    allowed_repairs = [
        {
            "name": "bounded_generator_level_selector",
            "status": "necessary",
            "definition": (
                "sample milestone2 candidates, attach cheap/cached profile labels in chunks, select by profile bands, "
                "and fail explicitly when support is insufficient"
            ),
            "close_condition": (
                "generator_startup_option true, no silent fallback, selector path pack-to-fit_model parity passes"
            ),
            "current_gap": {
                "stage_ready": stage_ready,
                "generator_progress_parts": diagnostics.get("generator_progress_parts", {}),
            },
        },
        {
            "name": "cheap_clean_pendulum_settle_guard",
            "status": "necessary",
            "definition": (
                "a prior-side label for sustained closed-loop settle: action is phase/state-conditioned, control decays or settles, "
                "and suffix state/reward remains improved without requiring the full feedback grid"
            ),
            "close_condition": (
                "guard is clean enough and count-safe across full/q90; Pendulum anchor reproduced; prior quota-ready"
            ),
            "current_gap": {
                "light_guard": light.get("decision", {}),
                "h_feature": h_feature.get("decision", {}),
                "guard_search": guard_search.get("decision", {}),
                "suffix_state_label": state_label.get("decision", {}),
                "state_blocker": state_blocker.get("decision", {}),
                "reward_relevant_state": reward_relevant_state.get("decision", {}),
                "reference_inertia_highway": reference_inertia_highway.get("decision", {}),
            },
        },
        {
            "name": "feedback_source_pool_subprofile_governance",
            "status": "keep_as_profile_guard",
            "definition": (
                "use temporal-union and near-best gain annotations to prevent constant/gain=2.0 collapse, "
                "without treating broad feedback count as a milestone target"
            ),
            "close_condition": (
                "profile/gain dominance is reduced under the unified selector while broad profile bands and diversity guards hold"
            ),
            "current_gap": {
                "source_pool_read": feedback_read,
                "single_scalar_control": feedback_control_read.get("single_existing_scalar_knob_sufficient"),
            },
        },
        {
            "name": "mountaincar_future_basin_detector",
            "status": "annotation_after_main_blockers",
            "definition": (
                "detect upfront control investment followed by future-basin reward/state improvement; do not quota until stable prior support exists"
            ),
            "close_condition": "stable full/q90 prior support plus diversity guard; not required for immediate profile closure",
            "current_gap": _safe_get(closure, "diagnostics", "mountaincar_progress_parts", default={}),
        },
    ]

    readiness = {
        "weighted_progress": _safe_get(progress, "weighted_progress", default=0.0),
        "remaining_to_threshold": _safe_get(progress, "remaining_to_threshold", default=1.0),
        "can_claim_sufficient_coverage": bool(_safe_get(progress, "can_claim_sufficient_coverage", default=False)),
        "largest_remaining_blocker": _safe_get(closure, "decision", "largest_remaining_blocker"),
        "second_blocker": _safe_get(closure, "decision", "second_blocker"),
    }

    return {
        "analysis_entry": "phase2_profile_closure_register",
        "contract": {
            "read_only_existing_artifacts": True,
            "does_not_sample_environments": True,
            "does_not_train": True,
            "does_not_modify_exact_scm": True,
            "does_not_modify_fit_model": True,
            "does_not_modify_ppo": True,
            "profile_quality_first": True,
            "delta_positive_not_a_gate": True,
            "full_feedback_grid_not_default_path": True,
        },
        "inputs": {
            "closure_plan_json": str(Path(closure_plan_json).expanduser().resolve()),
            "light_guard_json": str(Path(light_guard_json).expanduser().resolve()),
            "h_feature_json": str(Path(h_feature_json).expanduser().resolve()),
            "guard_search_json": str(Path(guard_search_json).expanduser().resolve()),
            "state_label_json": str(Path(state_label_json).expanduser().resolve()),
            "state_blocker_json": str(Path(state_blocker_json).expanduser().resolve()),
            "reward_relevant_state_json": str(Path(reward_relevant_state_json).expanduser().resolve()),
            "reference_inertia_highway_json": str(Path(reference_inertia_highway_json).expanduser().resolve()),
            "feedback_pool_json": str(Path(feedback_pool_json).expanduser().resolve()),
            "feedback_control_json": str(Path(feedback_control_json).expanduser().resolve()),
            "online_preflight_json": str(Path(online_preflight_json).expanduser().resolve()),
        },
        "readiness": readiness,
        "disallowed_repairs": disallowed_repairs,
        "allowed_repairs": allowed_repairs,
        "decision": {
            "closed_now": False,
            "why_not_closed": (
                "Fixed-list profile and engineering parity are closed, but generator-level selector startup and "
                "Pendulum sustained-settle semantics are not closed."
            ),
            "next_actions_in_order": [
                "Keep full feedback grid offline; use only cheap/cached/chunked labels in default generator startup.",
                "Repair Pendulum settle as a generator/yield problem: existing scalar h bias, existing field formulas, suffix-state label quota, and loose state-threshold quotas are all rejected for now.",
                "Treat reward-relevant state-settle as an offline falsification result, not a quota: it shows the blocker is not merely full-state detector noise.",
                "Do not retry simple reference-inertia/state-highway knobs unless a new mechanism hypothesis changes the semantics.",
                "Wire the bounded generator-level selector with explicit no-fallback failure only after the semantic blocker is resolved.",
                "Rerun selector-path pack-to-fit_model parity only if startup code changes.",
                "Only then decide whether MountainCar needs quota or remains annotation.",
            ],
            "complexity_safety": (
                "Use one selector orchestrator with label providers; do not add multiple milestone-specific generator branches."
            ),
        },
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Profile Closure Register",
        "",
        "Read-only register of allowed and disallowed repairs for profile semantic coverage.",
        "",
        "## Readiness",
        "",
    ]
    for key, value in report["readiness"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Disallowed Repairs", ""])
    for item in report["disallowed_repairs"]:
        lines.append(f"- `{item['name']}`: {item['reason']} Evidence: `{item['evidence']}`")
    lines.extend(["", "## Allowed Repairs", ""])
    for item in report["allowed_repairs"]:
        lines.append(f"### `{item['name']}`")
        lines.append(f"- status: `{item['status']}`")
        lines.append(f"- definition: {item['definition']}")
        lines.append(f"- close condition: {item['close_condition']}")
        lines.append("")
    lines.extend(["## Decision", ""])
    for key, value in report["decision"].items():
        if isinstance(value, list):
            lines.append(f"- `{key}`:")
            for entry in value:
                lines.append(f"  - {entry}")
        else:
            lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Files", ""])
    for key, value in report.get("outputs", {}).items():
        lines.append(f"- `{key}`: `{value}`")
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(
        closure_plan_json=args.closure_plan_json,
        light_guard_json=args.light_guard_json,
        h_feature_json=args.h_feature_json,
        guard_search_json=args.guard_search_json,
        state_label_json=args.state_label_json,
        state_blocker_json=args.state_blocker_json,
        reward_relevant_state_json=args.reward_relevant_state_json,
        reference_inertia_highway_json=args.reference_inertia_highway_json,
        feedback_pool_json=args.feedback_pool_json,
        feedback_control_json=args.feedback_control_json,
        online_preflight_json=args.online_preflight_json,
    )
    json_path = out_dir / "profile_closure_register.json"
    md_path = out_dir / "profile_closure_register.md"
    report["outputs"] = {"json": str(json_path), "markdown": str(md_path)}
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report["outputs"], sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--closure-plan-json", default=str(DEFAULT_CLOSURE_PLAN_JSON))
    parser.add_argument("--light-guard-json", default=str(DEFAULT_LIGHT_GUARD_JSON))
    parser.add_argument("--h-feature-json", default=str(DEFAULT_H_FEATURE_JSON))
    parser.add_argument("--guard-search-json", default=str(DEFAULT_GUARD_SEARCH_JSON))
    parser.add_argument("--state-label-json", default=str(DEFAULT_STATE_LABEL_JSON))
    parser.add_argument("--state-blocker-json", default=str(DEFAULT_STATE_BLOCKER_JSON))
    parser.add_argument("--reward-relevant-state-json", default=str(DEFAULT_REWARD_RELEVANT_STATE_JSON))
    parser.add_argument("--reference-inertia-highway-json", default=str(DEFAULT_REFERENCE_INERTIA_HIGHWAY_JSON))
    parser.add_argument("--feedback-pool-json", default=str(DEFAULT_FEEDBACK_POOL_JSON))
    parser.add_argument("--feedback-control-json", default=str(DEFAULT_FEEDBACK_CONTROL_JSON))
    parser.add_argument("--online-preflight-json", default=str(DEFAULT_ONLINE_PREFLIGHT_JSON))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
