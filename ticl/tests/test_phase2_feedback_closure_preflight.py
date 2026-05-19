import json
from pathlib import Path

from scripts.exploratory import phase2_feedback_closure_preflight as preflight


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _base_payloads(tmp_path: Path) -> dict[str, Path]:
    paths = {
        "coverage": tmp_path / "coverage.json",
        "sufficiency": tmp_path / "sufficiency.json",
        "branch": tmp_path / "branch.json",
        "smoke": tmp_path / "smoke.json",
        "budgeted": tmp_path / "budgeted.json",
        "pendulum": tmp_path / "pendulum.json",
        "reacher": tmp_path / "reacher.json",
    }
    _write_json(
        paths["coverage"],
        {"forecast": {"next_step": {"estimated_coverage_gain": 0.05}}},
    )
    _write_json(
        paths["sufficiency"],
        {
            "checks": {
                "selector_profile_guard_pass": True,
                "chunked_selector_semantic_parity": True,
                "pack_to_fit_model_numeric_parity": True,
            }
        },
    )
    _write_json(
        paths["branch"],
        {
            "read": {"branch_control_reduces_constant_gain2_dominance": True},
            "selection": {"guard_pass": True},
        },
    )
    _write_json(
        paths["smoke"],
        {
            "overall": {
                "raw_reward_sum_delta": {"positive_fraction": 0.65, "mean": 1.0},
                "reward_env_sum_delta": {"positive_fraction": 0.62, "mean": 1.0},
            }
        },
    )
    _write_json(
        paths["budgeted"],
        {
            "rules": {
                "gym_full_obs_action_range": {"candidate_count": 8},
                "gym_q90_obs_action_range": {"candidate_count": 10},
            }
        },
    )
    _write_json(
        paths["pendulum"],
        {
            "current_evidence": {
                "current_read": {
                    "mode_is_not_clean_enough": "not clean",
                    "pendulum_failure_location": "needs closed-loop settling",
                }
            }
        },
    )
    _write_json(paths["reacher"], {})
    return paths


def _build(paths: dict[str, Path]) -> dict:
    return preflight.build_report(
        coverage_json=paths["coverage"],
        sufficiency_json=paths["sufficiency"],
        branch_json=paths["branch"],
        branch_smoke_json=paths["smoke"],
        budgeted_json=paths["budgeted"],
        pendulum_json=paths["pendulum"],
        reacher_anchor_json=paths["reacher"],
    )


def test_feedback_closure_preflight_blocks_missing_settle_anchors(tmp_path: Path):
    report = _build(_base_payloads(tmp_path))

    assert report["evidence"]["branch_guard_pass"] is True
    assert report["evidence"]["sidecar"]["pass"] is True
    assert report["evidence"]["budgeted_support"]["has_selector_minimum"] is True
    assert report["evidence"]["budgeted_support"]["has_cross_seed_slack"] is False
    assert report["decision"]["ready_to_claim_feedback_closure"] is False
    assert "pendulum_closed_loop_settle_anchor" in report["decision"]["blockers"]
    assert "reacher_closed_loop_settle_anchor" in report["decision"]["blockers"]


def test_feedback_closure_preflight_passes_when_anchors_and_slack_pass(tmp_path: Path):
    paths = _base_payloads(tmp_path)
    _write_json(
        paths["budgeted"],
        {
            "rules": {
                "gym_full_obs_action_range": {"candidate_count": 12},
                "gym_q90_obs_action_range": {"candidate_count": 13},
            }
        },
    )
    _write_json(
        paths["pendulum"],
        {
            "current_evidence": {
                "current_read": {
                    "mode_is_not_clean_enough": "not clean",
                    "pendulum_failure_location": "needs closed-loop settling",
                }
            },
            "read": {"closed_loop_settle_anchor_pass": True},
        },
    )
    _write_json(paths["reacher"], {"read": {"closed_loop_settle_anchor_pass": True}})

    report = _build(paths)

    assert report["decision"]["ready_to_claim_feedback_closure"] is True
    assert report["decision"]["blockers"] == []
    assert report["score"]["score"] >= 0.78


def test_feedback_closure_preflight_reads_support_slack_audit_shape(tmp_path: Path):
    paths = _base_payloads(tmp_path)
    _write_json(
        paths["budgeted"],
        {
            "analysis_entry": "phase2_feedback_support_slack_audit",
            "groups": {
                "decay_plus_early_obs2_g1_2": {
                    "all_rules_present": True,
                    "all_meet_selector_minimum": True,
                    "all_meet_slack_target": True,
                    "gain_diversity_guard_pass": False,
                    "by_rule": {
                        "gym_full_obs_action_range": {"counts": [16, 19], "min": 16},
                        "gym_q90_obs_action_range": {"counts": [18, 21], "min": 18},
                    },
                }
            },
            "decision": {
                "ready_to_clear_slack_blocker": False,
                "read": "gain dominance blocks slack closure",
            },
        },
    )

    report = _build(paths)

    support = report["evidence"]["budgeted_support"]
    assert support["source"] == "support_slack_audit"
    assert support["has_selector_minimum"] is True
    assert support["has_cross_seed_slack"] is False
    assert support["gain_diversity_guard_pass"] is False
    assert "budgeted_startup_support_slack" in report["decision"]["blockers"]
