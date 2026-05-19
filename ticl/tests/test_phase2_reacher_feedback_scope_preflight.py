import json
from pathlib import Path

from scripts.exploratory import phase2_reacher_feedback_scope_preflight as preflight


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _official_dir(tmp_path: Path) -> Path:
    root = tmp_path / "Reacher-v5"
    root.mkdir(parents=True, exist_ok=True)
    for name in ("final_model.zip", "vecnormalize.pkl", "summary.json"):
        (root / name).write_text("{}", encoding="utf-8")
    return root


def test_reacher_scope_marks_secondary_target_out_of_scope(tmp_path: Path):
    governance = _write_json(
        tmp_path / "governance.json",
        {
            "axis_rows": [
                {
                    "axis_id": "context_identifiable_closed_loop_settle_feedback",
                    "gym_targets": ["Pendulum-v1", "stabilization-style tasks"],
                    "trust_status": "pressure_point_subprofile_not_single_quota",
                }
            ]
        },
    )

    report = preflight.build_report(governance_json=governance, official_reacher_dir=_official_dir(tmp_path))

    assert report["read"]["out_of_scope_for_current_closure"] is True
    assert report["read"]["closed_loop_settle_anchor_pass"] is True
    assert report["read"]["blockers"] == []


def test_reacher_scope_blocks_when_reacher_is_explicit_hard_target(tmp_path: Path):
    governance = _write_json(
        tmp_path / "governance.json",
        {
            "axis_rows": [
                {
                    "axis_id": "context_identifiable_closed_loop_settle_feedback",
                    "gym_targets": ["Pendulum-v1", "Reacher-v5"],
                    "trust_status": "hard_gym_target",
                }
            ]
        },
    )

    report = preflight.build_report(governance_json=governance, official_reacher_dir=_official_dir(tmp_path))

    assert report["read"]["out_of_scope_for_current_closure"] is False
    assert report["read"]["closed_loop_settle_anchor_pass"] is False
    assert report["read"]["blockers"] == ["reacher_anchor_required_but_missing"]
