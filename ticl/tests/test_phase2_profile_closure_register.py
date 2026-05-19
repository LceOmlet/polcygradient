import json
from pathlib import Path

from scripts.exploratory import phase2_profile_closure_register as register


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_register_rejects_raw_delta_and_full_grid_default(tmp_path: Path):
    closure = _write_json(
        tmp_path / "closure.json",
        {
            "progress": {
                "weighted_progress": 0.74,
                "remaining_to_threshold": 0.16,
                "can_claim_sufficient_coverage": False,
            },
            "decision": {
                "largest_remaining_blocker": "pendulum_settle_semantics",
                "second_blocker": "generator_level_selector",
            },
            "diagnostics": {
                "planned_stage_ready": {"source_pool": True},
                "generator_progress_parts": {"generator_startup_option": False},
                "mountaincar_progress_parts": {"prior_guard_pass": False},
            },
        },
    )
    light = _write_json(
        tmp_path / "light.json",
        {"decision": {"broad_but_impure_guards": ["time_core"], "cheap_guard_ready": False}},
    )
    h_feature = _write_json(
        tmp_path / "h.json",
        {"decision": {"strict_current_simple_h_knob_ready": False, "read": "no stable h split"}},
    )
    guard_search = _write_json(
        tmp_path / "guard_search.json",
        {
            "decision": {
                "cheap_formula_guard_ready": False,
                "best_formula": "a AND b",
                "read": "No simple bounded formula",
            }
        },
    )
    state_label = _write_json(
        tmp_path / "state_label.json",
        {
            "decision": {
                "cheap_suffix_state_settle_guard_ready": False,
                "min_count": 0,
                "read": "not count-safe",
            }
        },
    )
    state_blocker = _write_json(
        tmp_path / "state_blocker.json",
        {
            "decision": {
                "loose_non_worsening_count_safe_under_grid": True,
                "strict_contraction_count_safe_under_grid": False,
                "selector_threshold_repair_allowed": False,
            }
        },
    )
    reward_relevant_state = _write_json(
        tmp_path / "reward_relevant_state.json",
        {
            "decision": {
                "cheap_suffix_state_settle_guard_ready": False,
                "min_count": 0,
                "read": "not count-safe",
            }
        },
    )
    reference_inertia_highway = _write_json(
        tmp_path / "reference_inertia_highway.json",
        {
            "decision": {
                "cheap_suffix_state_settle_guard_ready": False,
                "min_count": 0,
                "read": "not count-safe",
            }
        },
    )
    feedback_pool = _write_json(
        tmp_path / "pool.json",
        {"read": {"temporal_union_repairs_profile": True, "temporal_union_repairs_gain": False}},
    )
    feedback_control = _write_json(
        tmp_path / "control.json",
        {"read": {"single_existing_scalar_knob_sufficient": False}},
    )
    online = _write_json(
        tmp_path / "online.json",
        {"feedback_eval_budget": {"read": "full grid over budget"}},
    )

    report = register.build_report(
        closure_plan_json=closure,
        light_guard_json=light,
        h_feature_json=h_feature,
        guard_search_json=guard_search,
        state_label_json=state_label,
        state_blocker_json=state_blocker,
        reward_relevant_state_json=reward_relevant_state,
        reference_inertia_highway_json=reference_inertia_highway,
        feedback_pool_json=feedback_pool,
        feedback_control_json=feedback_control,
        online_preflight_json=online,
    )

    assert report["contract"]["delta_positive_not_a_gate"] is True
    assert report["contract"]["full_feedback_grid_not_default_path"] is True
    names = {item["name"] for item in report["disallowed_repairs"]}
    assert "raw_delta_or_raw_positive_quota" in names
    assert "full_feedback_grid_in_fit_model_startup" in names
    assert "existing_signature_formula_quota" in names
    assert "suffix_state_settle_quota_now" in names
    assert "loose_state_non_worsening_quota" in names
    assert "reward_relevant_state_settle_quota_now" in names
    assert "simple_reference_inertia_highway_knob" in names
    allowed = {item["name"]: item for item in report["allowed_repairs"]}
    assert allowed["cheap_clean_pendulum_settle_guard"]["status"] == "necessary"
    assert allowed["bounded_generator_level_selector"]["status"] == "necessary"
    assert report["decision"]["closed_now"] is False
