from ticl.analysis import phase3_pair2_guarded_target_present_selector_trace as trace_mod


def test_build_arm_summary_flags_stale_position_range() -> None:
    selector_trace = {
        "source_path": "/tmp/selector_trace.json",
        "completed": True,
        "summary": {
            "row_count": 1,
            "target_appeared": False,
            "matched_outer_batches": [],
            "first_match": None,
        },
        "rows": [
            {
                "epoch_idx": 1,
                "outer_batch_idx": 7,
                "total_outer_batches": 16,
                "objective_total": 64,
                "objective_env_indices_present": [12],
                "target_status": "target_position_range_absent",
                "target_env_index": 12,
                "target_env_objective_token_count": 64,
                "target_env_objective_episode_indices_present": [4],
                "target_env_objective_episode_spans": [
                    {
                        "objective_episode_index": 4,
                        "token_count": 4,
                        "objective_position_min": 10,
                        "objective_position_max": 13,
                    }
                ],
                "target_objective_episode_index": 4,
                "target_episode_token_count": 4,
                "target_episode_positions_present": [10, 11, 12, 13],
                "target_objective_position_start": 15,
                "target_objective_position_end": 18,
                "target_match_count": 0,
                "target_match_positions": [],
            }
        ],
    }
    target_selector = {
        "target_outer_batch_idx": None,
        "target_env_index": 12,
        "target_objective_episode_index": 4,
        "target_objective_position_start": 15,
        "target_objective_position_end": 18,
    }

    summary = trace_mod._build_arm_summary(
        mode_label="baseline",
        actor_objective_mode_override=None,
        actor_objective_runtime_current_suite_name=None,
        selector_trace=selector_trace,
        snapshot=None,
        elapsed_wall_time_sec=10.0,
        last_update_wall_time_sec=9.0,
        total_env_steps=4096,
        target_selector=target_selector,
    )

    assert summary["capture"]["target_appeared"] is False
    assert summary["selector_trace"]["target_absence_reason"] == "target_position_range_absent"
    assert summary["selector_trace"]["selector_match_bug_suspected"] is False
    assert summary["selector_trace"]["canonical_present_selector"]["candidate_position_start"] == 10
    assert summary["selector_trace"]["canonical_present_selector"]["candidate_position_end"] == 13
    assert summary["selector_trace"]["canonical_selector_preserves_env_and_episode_identity"] is True
    assert summary["selector_trace"]["canonical_selector_requires_position_change"] is True


def test_build_arm_summary_flags_selector_match_bug_suspected() -> None:
    selector_trace = {
        "source_path": "/tmp/selector_trace.json",
        "completed": True,
        "summary": {
            "row_count": 1,
            "target_appeared": False,
            "matched_outer_batches": [],
            "first_match": None,
        },
        "rows": [
            {
                "epoch_idx": 1,
                "outer_batch_idx": 7,
                "total_outer_batches": 16,
                "objective_total": 64,
                "objective_env_indices_present": [12],
                "target_status": "selector_unmatched",
                "target_env_index": 12,
                "target_env_objective_token_count": 64,
                "target_env_objective_episode_indices_present": [0],
                "target_env_objective_episode_spans": [
                    {
                        "objective_episode_index": 0,
                        "token_count": 4,
                        "objective_position_min": 18,
                        "objective_position_max": 21,
                    }
                ],
                "target_objective_episode_index": 0,
                "target_episode_token_count": 4,
                "target_episode_positions_present": [18, 19, 20, 21],
                "target_objective_position_start": 18,
                "target_objective_position_end": 21,
                "target_match_count": 0,
                "target_match_positions": [],
            }
        ],
    }
    target_selector = {
        "target_outer_batch_idx": None,
        "target_env_index": 12,
        "target_objective_episode_index": 0,
        "target_objective_position_start": 18,
        "target_objective_position_end": 21,
    }

    summary = trace_mod._build_arm_summary(
        mode_label="baseline",
        actor_objective_mode_override=None,
        actor_objective_runtime_current_suite_name=None,
        selector_trace=selector_trace,
        snapshot=None,
        elapsed_wall_time_sec=10.0,
        last_update_wall_time_sec=9.0,
        total_env_steps=4096,
        target_selector=target_selector,
    )

    assert summary["capture"]["target_appeared"] is False
    assert summary["selector_trace"]["target_absence_reason"] == "selector_match_bug_suspected"
    assert summary["selector_trace"]["selector_match_bug_suspected"] is True
    assert summary["selector_trace"]["requested_position_range_present_outer_batches"] == [7]


def test_build_arm_summary_flags_target_episode_absent() -> None:
    selector_trace = {
        "source_path": "/tmp/selector_trace.json",
        "completed": True,
        "summary": {
            "row_count": 1,
            "target_appeared": False,
            "matched_outer_batches": [],
            "first_match": None,
        },
        "rows": [
            {
                "epoch_idx": 1,
                "outer_batch_idx": 13,
                "total_outer_batches": 16,
                "objective_total": 192,
                "objective_env_indices_present": [12],
                "target_status": "target_episode_absent",
                "target_env_index": 12,
                "target_env_objective_token_count": 192,
                "target_env_objective_episode_indices_present": [3, 4, 5],
                "target_env_objective_episode_spans": [
                    {
                        "objective_episode_index": 3,
                        "token_count": 3,
                        "objective_position_min": 0,
                        "objective_position_max": 2,
                    }
                ],
                "target_objective_episode_index": 0,
                "target_episode_token_count": 0,
                "target_episode_positions_present": [],
                "target_objective_position_start": 18,
                "target_objective_position_end": 21,
                "target_match_count": 0,
                "target_match_positions": [],
            }
        ],
    }
    target_selector = {
        "target_outer_batch_idx": None,
        "target_env_index": 12,
        "target_objective_episode_index": 0,
        "target_objective_position_start": 18,
        "target_objective_position_end": 21,
    }

    summary = trace_mod._build_arm_summary(
        mode_label="baseline",
        actor_objective_mode_override=None,
        actor_objective_runtime_current_suite_name=None,
        selector_trace=selector_trace,
        snapshot=None,
        elapsed_wall_time_sec=10.0,
        last_update_wall_time_sec=9.0,
        total_env_steps=4096,
        target_selector=target_selector,
    )

    assert summary["capture"]["target_appeared"] is False
    assert summary["selector_trace"]["target_absence_reason"] == "target_episode_absent"
    assert summary["selector_trace"]["env_present_outer_batches"] == [13]
    assert summary["selector_trace"]["episode_present_outer_batches"] == []


def test_build_pair2_guarded_target_present_selector_trace_report_detects_stale_selector_without_runtime_drift(
    monkeypatch,
) -> None:
    stale_trace = {
        "source_path": "/tmp/selector_trace.json",
        "completed": True,
        "summary": {
            "row_count": 1,
            "target_appeared": False,
            "matched_outer_batches": [],
            "first_match": None,
        },
        "rows": [
            {
                "epoch_idx": 1,
                "outer_batch_idx": 7,
                "total_outer_batches": 16,
                "objective_total": 64,
                "objective_env_indices_present": [12],
                "target_status": "target_position_range_absent",
                "target_env_index": 12,
                "target_env_objective_token_count": 64,
                "target_env_objective_episode_indices_present": [0],
                "target_env_objective_episode_spans": [
                    {
                        "objective_episode_index": 0,
                        "token_count": 4,
                        "objective_position_min": 10,
                        "objective_position_max": 13,
                    }
                ],
                "target_objective_episode_index": 0,
                "target_episode_token_count": 4,
                "target_episode_positions_present": [10, 11, 12, 13],
                "target_objective_position_start": 18,
                "target_objective_position_end": 21,
                "target_match_count": 0,
                "target_match_positions": [],
            }
        ],
    }

    def _fake_run_single_arm_capture(**kwargs):
        return {
            "runtime": {
                "elapsed_wall_time_sec": 12.0,
                "last_update_wall_time_sec": 10.0,
                "total_env_steps": 4096,
            },
            "selector_trace": stale_trace,
            "snapshot": None,
            "train_history": [{"outer_epoch": 1}],
        }

    monkeypatch.setattr(trace_mod, "_run_single_arm_capture", _fake_run_single_arm_capture)
    monkeypatch.setattr(
        trace_mod,
        "_load_pair2_preflight",
        lambda: {
            "preflight_passed": True,
            "contract_args": {},
            "contract_diff": {"allowed_diff": {}, "unexpected_diff": {}},
        },
    )
    monkeypatch.setattr(
        trace_mod,
        "assert_phase2_green",
        lambda path: {"trusted_contract": {"phase2_green": True}},
    )

    report = trace_mod.build_pair2_guarded_target_present_selector_trace_report()

    assert report["conclusions"]["selector_trace_completed_in_both_arms"] is True
    assert report["conclusions"]["selector_row_layout_identical_across_arms"] is True
    assert report["conclusions"]["target_present_in_neither_arm"] is True
    assert report["conclusions"]["env12_enters_trusted_outer_batch"] is True
    assert report["conclusions"]["objective_episode_present_for_env12"] is True
    assert report["conclusions"]["requested_selector_range_present_for_env12"] is False
    assert report["conclusions"]["requested_selector_position_range_looks_stale"] is True
    assert report["conclusions"]["runtime_mode_changes_selector_membership"] is False
    assert report["recommendation"]["next_focus"] == "repair_selector_definition"
