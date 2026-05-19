import json
from pathlib import Path

from scripts.exploratory import phase2_profile_coverage_closure_forecast as forecast


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _inventory() -> dict:
    rows = []
    for mode in forecast.MODE_WEIGHTS:
        rows.append(
            {
                "mode_id": mode,
                "selector_sufficient": mode
                not in {"mountaincar_upfront_investment_future_basin", "terminal_survival_action_sensitive"},
                "gym_semantic_sufficient": False,
                "known_gaps": ["gap"] if mode == "context_identifiable_closed_loop_settle_feedback" else [],
            }
        )
    return {
        "complexity": {"risk_score": 0.55},
        "mode_inventory": {"rows": rows},
    }


def _sufficiency() -> dict:
    return {
        "score": {"score": 1.0},
        "remaining_work": [
            "2048-scale profile labels are projected/chunked but not yet fully executed end-to-end",
            "cache-by-frozen_h fingerprint is a contract",
        ],
    }


def _pendulum() -> dict:
    return {
        "current_evidence": {
            "current_read": {
                "mode_is_not_clean_enough": "not clean",
                "pendulum_failure_location": "needs sustained closed-loop settling",
            }
        }
    }


def _mountaincar_anchor() -> dict:
    return {"read": {"gym_anchor_pass": True, "quota_ready": False, "why_not_quota_ready": "needs health"}}


def _mountaincar_detector() -> dict:
    return {"read": {"detector_candidate_exists": True, "quota_ready": False}}


def _feedback_branch() -> dict:
    return {"read": {"branch_control_reduces_constant_gain2_dominance": True}}


def _fixture(tmp_path: Path) -> dict[str, Path]:
    return {
        "sufficiency": _write_json(tmp_path / "sufficiency.json", _sufficiency()),
        "inventory": _write_json(tmp_path / "inventory.json", _inventory()),
        "pendulum": _write_json(tmp_path / "pendulum.json", _pendulum()),
        "mountaincar_anchor": _write_json(tmp_path / "mountaincar_anchor.json", _mountaincar_anchor()),
        "mountaincar_detector": _write_json(tmp_path / "mountaincar_detector.json", _mountaincar_detector()),
        "feedback_branch": _write_json(tmp_path / "feedback_branch.json", _feedback_branch()),
    }


def _build(paths: dict[str, Path]) -> dict:
    return forecast.build_report(
        sufficiency_json=paths["sufficiency"],
        inventory_json=paths["inventory"],
        pendulum_json=paths["pendulum"],
        mountaincar_anchor_json=paths["mountaincar_anchor"],
        mountaincar_detector_json=paths["mountaincar_detector"],
        feedback_branch_json=paths["feedback_branch"],
    )


def test_forecast_separates_engineering_and_gym_coverage(tmp_path: Path):
    report = _build(_fixture(tmp_path))

    buckets = report["coverage"]["buckets"]
    assert buckets["engineering_pipeline"]["score"] == 1.0
    assert buckets["gym_semantic_coverage"]["score"] < buckets["engineering_pipeline"]["score"]
    assert report["forecast"]["next_step"]["id"] == "feedback_closed_loop_settle_closure"
    assert report["forecast"]["likely_bounded_iterations"] == 3
    assert report["coverage"]["overall_score"] < 1.0


def test_selector_mode_coverage_counts_missing_annotation_modes(tmp_path: Path):
    report = _build(_fixture(tmp_path))

    per_mode = report["selector_mode_coverage"]["per_mode"]
    assert per_mode["context_identifiable_closed_loop_settle_feedback"]["selector_sufficient"] is True
    assert per_mode["mountaincar_upfront_investment_future_basin"]["selector_sufficient"] is False
    assert report["selector_mode_coverage"]["score"] < 1.0
