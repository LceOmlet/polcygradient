import json
from pathlib import Path

from scripts.exploratory import phase2_profile_closure_progress_plan as plan


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_closure_progress_keeps_semantic_blockers_separate(tmp_path: Path):
    semantic = _write_json(
        tmp_path / "semantic.json",
        {
            "gates": {
                "sufficient_coverage_closed": False,
                "generator_level_closed": False,
            },
            "reads": {
                "feedback": {
                    "slack": {"ready_to_clear_slack_blocker": False},
                },
                "pendulum_settle": {
                    "anchor_interpretation": "anchor defined",
                    "anchor_pass": False,
                    "prior_max_settle_count": 8,
                    "prior_quota_ready": False,
                },
                "mountaincar_future_basin": {
                    "gym_anchor_pass": True,
                    "prior_scope_summaries": {"gym_full_obs_action_range": {}},
                    "prior_guard_pass": False,
                },
            },
        },
    )
    online = _write_json(
        tmp_path / "online.json",
        {
            "existing_evidence": {
                "source_pool": {"source_ready_for_candidate_multiplier": True},
                "selector": {"hard_pass": True},
            },
            "bounded_generator_plan": {
                "stage_plan": [{"name": "source_pool", "ready": True}],
            },
        },
    )
    fit = _write_json(
        tmp_path / "fit.json",
        {
            "preflight": {
                "fit_model_2048_scale_projection": {
                    "chunked_probe": {"memory_safe": True},
                    "naive_all_at_once": {"memory_safe": False},
                }
            }
        },
    )
    light = _write_json(
        tmp_path / "light.json",
        {"decision": {"cheap_guard_ready": False}},
    )
    h_feature = _write_json(
        tmp_path / "h_feature.json",
        {"decision": {"strict_current_simple_h_knob_ready": False}},
    )
    guard_search = _write_json(
        tmp_path / "guard_search.json",
        {"decision": {"cheap_formula_guard_ready": False}},
    )
    state_label = _write_json(
        tmp_path / "state_label.json",
        {"decision": {"cheap_suffix_state_settle_guard_ready": False}},
    )
    state_blocker = _write_json(
        tmp_path / "state_blocker.json",
        {"decision": {"selector_threshold_repair_allowed": False}},
    )
    reward_relevant_state = _write_json(
        tmp_path / "reward_relevant_state.json",
        {"decision": {"cheap_suffix_state_settle_guard_ready": False}},
    )
    selector_parity = _write_json(
        tmp_path / "selector_parity.json",
        {"hard_pass": False, "analysis_entry": "test"},
    )

    report = plan.build_report(
        semantic_gate_json=semantic,
        online_preflight_json=online,
        fit_sufficiency_json=fit,
        pendulum_light_guard_json=light,
        pendulum_h_feature_json=h_feature,
        pendulum_guard_search_json=guard_search,
        pendulum_state_label_json=state_label,
        pendulum_state_blocker_json=state_blocker,
        pendulum_reward_relevant_state_json=reward_relevant_state,
        selector_path_parity_json=selector_parity,
    )

    assert report["contract"]["delta_sign_fraction_not_a_gate"] is True
    assert report["progress"]["components"]["engineering_parity"]["status"] == "closed"
    assert report["progress"]["components"]["fixed_list_profile"]["status"] == "closed"
    assert report["progress"]["components"]["generator_level_selector"]["progress"] < 1.0
    assert report["diagnostics"]["generator_progress_parts"]["selector_path_pack_fit_parity"] is False
    assert report["progress"]["components"]["pendulum_settle_semantics"]["progress"] < 1.0
    assert report["diagnostics"]["pendulum_progress_parts"]["existing_h_knob_ready"] is False
    assert report["diagnostics"]["pendulum_progress_parts"]["existing_signature_formula_ready"] is False
    assert report["diagnostics"]["pendulum_progress_parts"]["suffix_state_label_ready"] is False
    assert report["diagnostics"]["pendulum_progress_parts"]["reward_relevant_state_label_ready"] is False
    assert report["diagnostics"]["pendulum_progress_parts"]["state_threshold_repair_allowed"] is False
    assert report["decision"]["largest_remaining_blocker"] == "pendulum_settle_semantics"
    assert report["progress"]["can_claim_sufficient_coverage"] is False


def test_closure_progress_requires_semantic_gate_even_if_numeric_progress_high(tmp_path: Path):
    semantic = _write_json(
        tmp_path / "semantic.json",
        {
            "gates": {
                "sufficient_coverage_closed": False,
                "generator_level_closed": True,
            },
            "reads": {
                "feedback": {"slack": {"ready_to_clear_slack_blocker": True}},
                "pendulum_settle": {
                    "anchor_interpretation": "anchor defined",
                    "anchor_pass": True,
                    "prior_max_settle_count": 12,
                    "prior_quota_ready": True,
                },
                "mountaincar_future_basin": {
                    "gym_anchor_pass": True,
                    "prior_scope_summaries": {"gym_full_obs_action_range": {}},
                    "prior_guard_pass": True,
                },
            },
        },
    )
    online = _write_json(
        tmp_path / "online.json",
        {
            "existing_evidence": {
                "source_pool": {"source_ready_for_candidate_multiplier": True},
                "selector": {"hard_pass": True},
            },
            "bounded_generator_plan": {"stage_plan": []},
        },
    )
    fit = _write_json(
        tmp_path / "fit.json",
        {
            "preflight": {
                "fit_model_2048_scale_projection": {
                    "chunked_probe": {"memory_safe": True},
                    "naive_all_at_once": {"memory_safe": False},
                }
            }
        },
    )
    light = _write_json(
        tmp_path / "light.json",
        {"decision": {"cheap_guard_ready": True}},
    )
    h_feature = _write_json(
        tmp_path / "h_feature.json",
        {"decision": {"strict_current_simple_h_knob_ready": True}},
    )
    guard_search = _write_json(
        tmp_path / "guard_search.json",
        {"decision": {"cheap_formula_guard_ready": True}},
    )
    state_label = _write_json(
        tmp_path / "state_label.json",
        {"decision": {"cheap_suffix_state_settle_guard_ready": True}},
    )
    state_blocker = _write_json(
        tmp_path / "state_blocker.json",
        {"decision": {"selector_threshold_repair_allowed": True}},
    )
    reward_relevant_state = _write_json(
        tmp_path / "reward_relevant_state.json",
        {"decision": {"cheap_suffix_state_settle_guard_ready": True}},
    )
    selector_parity = _write_json(
        tmp_path / "selector_parity.json",
        {"hard_pass": True, "analysis_entry": "test"},
    )

    report = plan.build_report(
        semantic_gate_json=semantic,
        online_preflight_json=online,
        fit_sufficiency_json=fit,
        pendulum_light_guard_json=light,
        pendulum_h_feature_json=h_feature,
        pendulum_guard_search_json=guard_search,
        pendulum_state_label_json=state_label,
        pendulum_state_blocker_json=state_blocker,
        pendulum_reward_relevant_state_json=reward_relevant_state,
        selector_path_parity_json=selector_parity,
    )

    assert report["progress"]["weighted_progress"] >= report["progress"]["close_enough_threshold"]
    assert report["progress"]["can_claim_sufficient_coverage"] is False


def test_closure_progress_consumes_cheap_milestone3_as_separate_track(tmp_path: Path):
    semantic = _write_json(
        tmp_path / "semantic.json",
        {
            "gates": {
                "sufficient_coverage_closed": False,
                "generator_level_closed": False,
            },
            "reads": {
                "feedback": {"slack": {"ready_to_clear_slack_blocker": False}},
                "pendulum_settle": {
                    "anchor_interpretation": "anchor defined",
                    "anchor_pass": False,
                    "prior_max_settle_count": 8,
                    "prior_quota_ready": False,
                },
                "mountaincar_future_basin": {
                    "gym_anchor_pass": True,
                    "prior_scope_summaries": {"gym_full_obs_action_range": {}},
                    "prior_guard_pass": False,
                },
            },
        },
    )
    online = _write_json(
        tmp_path / "online.json",
        {
            "existing_evidence": {
                "source_pool": {"source_ready_for_candidate_multiplier": True},
                "selector": {"hard_pass": True},
            },
            "bounded_generator_plan": {"stage_plan": []},
        },
    )
    fit = _write_json(
        tmp_path / "fit.json",
        {
            "preflight": {
                "fit_model_2048_scale_projection": {
                    "chunked_probe": {"memory_safe": True},
                    "naive_all_at_once": {"memory_safe": False},
                }
            }
        },
    )
    false_decision = {"decision": {"cheap_guard_ready": False}}
    light = _write_json(tmp_path / "light.json", false_decision)
    h_feature = _write_json(tmp_path / "h_feature.json", {"decision": {"strict_current_simple_h_knob_ready": False}})
    guard_search = _write_json(tmp_path / "guard_search.json", {"decision": {"cheap_formula_guard_ready": False}})
    state_label = _write_json(tmp_path / "state_label.json", {"decision": {"cheap_suffix_state_settle_guard_ready": False}})
    state_blocker = _write_json(tmp_path / "state_blocker.json", {"decision": {"selector_threshold_repair_allowed": False}})
    reward_relevant_state = _write_json(
        tmp_path / "reward_relevant_state.json",
        {"decision": {"cheap_suffix_state_settle_guard_ready": False}},
    )
    selector_parity = _write_json(tmp_path / "selector_parity.json", {"hard_pass": True, "analysis_entry": "test"})
    cheap = _write_json(
        tmp_path / "cheap.json",
        {
            "analysis_entry": "phase2_profile_milestone3_cheap_closure_preflight",
            "score": {"weighted_progress": 1.0, "milestone_threshold": 0.9, "remaining_to_threshold": 0.0},
            "decision": {"ready_to_land_milestone3_cheap": True, "hard_blockers": []},
        },
    )

    report = plan.build_report(
        semantic_gate_json=semantic,
        online_preflight_json=online,
        fit_sufficiency_json=fit,
        pendulum_light_guard_json=light,
        pendulum_h_feature_json=h_feature,
        pendulum_guard_search_json=guard_search,
        pendulum_state_label_json=state_label,
        pendulum_state_blocker_json=state_blocker,
        pendulum_reward_relevant_state_json=reward_relevant_state,
        selector_path_parity_json=selector_parity,
        cheap_milestone3_preflight_json=cheap,
    )

    tracks = report["decision"]["milestone_tracks"]
    assert tracks["broad_sufficient_coverage"]["can_claim"] is False
    assert tracks["cheap_milestone3"]["consumed_preflight"] is True
    assert tracks["cheap_milestone3"]["ready_to_land"] is True
    assert tracks["cheap_milestone3"]["remaining_to_threshold"] == 0.0
    assert report["diagnostics"]["cheap_milestone3_preflight_read"]["analysis_entry"] == (
        "phase2_profile_milestone3_cheap_closure_preflight"
    )
