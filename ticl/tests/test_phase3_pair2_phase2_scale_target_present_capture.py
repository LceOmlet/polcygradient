from ticl.analysis.phase3_pair2_phase2_scale_target_present_capture import _build_capture_summary


def test_build_capture_summary_with_match() -> None:
    selector_trace = {
        "summary": {
            "target_appeared": True,
            "first_match": {"epoch_idx": 1, "outer_batch_idx": 7, "target_match_count": 4},
        }
    }
    snapshot = {
        "summary": {"outer_batch_idx": 7, "objective_total": 64},
        "snapshot_target_selector": {
            "target_env_index": 12,
            "target_objective_episode_index": 10,
            "target_objective_position_start": 10,
            "target_objective_position_end": 13,
            "target_outer_batch_idx": None,
        },
        "snapshot_target_match_count": 4,
    }
    summary = _build_capture_summary(selector_trace=selector_trace, snapshot=snapshot)
    assert summary["target_appeared"] is True
    assert summary["snapshot_written"] is True
    assert summary["same_hit_relationship_preserved"] is True
    assert summary["snapshot_outer_batch_idx"] == 7


def test_build_capture_summary_without_match() -> None:
    selector_trace = {"summary": {"target_appeared": False, "first_match": None}}
    summary = _build_capture_summary(selector_trace=selector_trace, snapshot=None)
    assert summary["target_appeared"] is False
    assert summary["snapshot_written"] is False
    assert summary["same_hit_relationship_preserved"] is False
    assert summary["snapshot_outer_batch_idx"] is None
