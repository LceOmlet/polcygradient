import json
from pathlib import Path

from scripts.exploratory import phase2_profile_semantic_coverage_gate as gate


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _base_inputs(tmp_path: Path) -> dict[str, Path]:
    paths = {
        "sufficiency": tmp_path / "sufficiency.json",
        "v2e": tmp_path / "v2e.json",
        "feedback": tmp_path / "feedback.json",
        "slack": tmp_path / "slack.json",
        "pendulum_anchor": tmp_path / "pendulum_anchor.json",
        "pendulum_prior": tmp_path / "pendulum_prior.json",
        "mountaincar_anchor": tmp_path / "mountaincar_anchor.json",
        "mountaincar_prior": tmp_path / "mountaincar_prior.json",
    }
    _write_json(
        paths["sufficiency"],
        {
            "score": {"pipeline_score": 1.0},
            "decision": {"seal_as_new_engineering_milestone_candidate": True},
        },
    )
    _write_json(
        paths["v2e"],
        {
            "readiness": {"candidate_profile_readiness_estimate": 0.91},
            "smoke_read": {"candidate_smoke_guard_pass": True},
            "v2e_numeric_parity_read": {"hard_pass": True},
            "generator_level_gap_read": {"generator_level_selector_ready": False},
            "fit_model_support_read": {"fit_model_startup_option_present": False},
        },
    )
    _write_json(
        paths["feedback"],
        {
            "decision": {
                "ready_to_claim_feedback_closure": False,
                "blockers": ["budgeted_startup_support_slack"],
            }
        },
    )
    _write_json(
        paths["slack"],
        {
            "decision": {"ready_to_clear_slack_blocker": False, "best_group": "obs4"},
            "groups": {
                "obs4": {
                    "all_meet_selector_minimum": True,
                    "all_meet_slack_target": False,
                    "gain_diversity_guard_pass": True,
                }
            },
        },
    )
    _write_json(
        paths["pendulum_anchor"],
        {"read": {"closed_loop_settle_anchor_pass": False, "interpretation": "missing settle"}},
    )
    _write_json(
        paths["pendulum_prior"],
        {
            "read": {
                "quota_ready": False,
                "min_settle_count": 3,
                "max_settle_count": 8,
                "mean_settle_rate": 0.074,
                "interpretation": "too sparse",
            }
        },
    )
    _write_json(paths["mountaincar_anchor"], {"read": {"gym_anchor_pass": True}})
    _write_json(
        paths["mountaincar_prior"],
        {
            "mountaincar_prior_fixedlist_guard": {
                "guard_pass": False,
                "scope_summaries": {},
                "read": "sparse",
            }
        },
    )
    return paths


def _build(paths: dict[str, Path]) -> dict:
    return gate.build_report(
        sufficiency_json=paths["sufficiency"],
        v2e_readiness_json=paths["v2e"],
        feedback_closure_json=paths["feedback"],
        feedback_slack_json=paths["slack"],
        pendulum_anchor_json=paths["pendulum_anchor"],
        pendulum_prior_json=paths["pendulum_prior"],
        mountaincar_anchor_json=paths["mountaincar_anchor"],
        mountaincar_prior_json=paths["mountaincar_prior"],
    )


def test_semantic_gate_blocks_fixed_list_only_success(tmp_path: Path):
    report = _build(_base_inputs(tmp_path))

    assert report["gates"]["engineering_parity_closed"] is True
    assert report["gates"]["fixed_list_profile_closed"] is True
    assert report["gates"]["generator_level_closed"] is False
    assert report["gates"]["semantic_gym_mode_closed"] is False
    assert report["gates"]["sufficient_coverage_closed"] is False
    assert "generator_level_selector_not_ready" in report["decision"]["blockers"]
    assert "pendulum_settle_prior_coverage_sparse" in report["decision"]["blockers"]
    assert report["contract"]["delta_sign_fraction_not_a_gate"] is True


def test_semantic_gate_passes_only_when_all_semantic_and_generator_gates_pass(tmp_path: Path):
    paths = _base_inputs(tmp_path)
    _write_json(
        paths["v2e"],
        {
            "readiness": {"candidate_profile_readiness_estimate": 0.91},
            "smoke_read": {"candidate_smoke_guard_pass": True},
            "v2e_numeric_parity_read": {"hard_pass": True},
            "generator_level_gap_read": {"generator_level_selector_ready": True},
            "fit_model_support_read": {"fit_model_startup_option_present": True},
        },
    )
    _write_json(paths["feedback"], {"decision": {"ready_to_claim_feedback_closure": True}})
    _write_json(
        paths["slack"],
        {
            "decision": {"ready_to_clear_slack_blocker": True, "best_group": "obs4"},
            "groups": {
                "obs4": {
                    "all_meet_selector_minimum": True,
                    "all_meet_slack_target": True,
                    "gain_diversity_guard_pass": True,
                }
            },
        },
    )
    _write_json(
        paths["pendulum_anchor"],
        {"read": {"closed_loop_settle_anchor_pass": True, "interpretation": "passed"}},
    )
    _write_json(
        paths["pendulum_prior"],
        {
            "read": {
                "quota_ready": True,
                "min_settle_count": 12,
                "max_settle_count": 16,
                "mean_settle_rate": 0.2,
                "interpretation": "ready",
            }
        },
    )
    _write_json(paths["mountaincar_prior"], {"mountaincar_prior_fixedlist_guard": {"guard_pass": True}})

    report = _build(paths)

    assert report["gates"]["sufficient_coverage_closed"] is True
    assert report["decision"]["can_claim_third_candidate_milestone"] is True
    assert report["decision"]["blockers"] == []
