import numpy as np

from ticl.sb3_recurrent_ppo import (
    _build_objective_position_metadata,
    _build_train_outer_batch_selector_trace_row,
    _resolve_train_outer_batch_snapshot_target_mask,
)


def test_build_objective_position_metadata_tracks_env_step_and_objective_local_positions():
    episode_starts = np.asarray(
        [
            [1.0, 1.0],
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 0.0],
        ],
        dtype=np.float32,
    )
    objective_masks = np.asarray(
        [
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )

    metadata = _build_objective_position_metadata(
        episode_starts=episode_starts,
        objective_masks=objective_masks,
    )

    assert np.array_equal(
        metadata["env_indices"],
        np.asarray(
            [
                [0, 1],
                [0, 1],
                [0, 1],
                [0, 1],
                [0, 1],
            ],
            dtype=np.int64,
        ),
    )
    assert np.array_equal(
        metadata["step_indices"],
        np.asarray(
            [
                [0, 0],
                [1, 1],
                [2, 2],
                [3, 3],
                [4, 4],
            ],
            dtype=np.int64,
        ),
    )
    assert np.array_equal(
        metadata["objective_episode_indices"],
        np.asarray(
            [
                [0, 0],
                [0, 0],
                [0, 1],
                [1, 1],
                [1, 1],
            ],
            dtype=np.int64,
        ),
    )
    assert np.array_equal(
        metadata["objective_episode_positions"],
        np.asarray(
            [
                [0, -1],
                [1, 0],
                [-1, 0],
                [0, 1],
                [1, -1],
            ],
            dtype=np.int64,
        ),
    )
    assert np.array_equal(
        metadata["objective_global_positions"],
        np.asarray(
            [
                [0, -1],
                [1, 0],
                [-1, 1],
                [2, 2],
                [3, -1],
            ],
            dtype=np.int64,
        ),
    )


def test_resolve_train_outer_batch_snapshot_target_mask_matches_requested_env_segment():
    objective_mask = np.asarray([1, 1, 1, 1, 1, 0], dtype=bool)
    target_mask = _resolve_train_outer_batch_snapshot_target_mask(
        objective_mask=objective_mask,
        flat_env_indices=np.asarray([0, 12, 12, 12, 12, 12], dtype=np.int64),
        flat_objective_episode_indices=np.asarray([2, 0, 0, 0, 1, -1], dtype=np.int64),
        flat_objective_episode_positions=np.asarray([10, 18, 19, 25, 0, -1], dtype=np.int64),
        target_env_index=12,
        target_objective_episode_index=0,
        target_objective_position_start=18,
        target_objective_position_end=21,
    )

    assert np.array_equal(
        target_mask,
        np.asarray([False, True, True, False, False, False]),
    )


def test_resolve_train_outer_batch_snapshot_target_mask_returns_none_without_segment_selector():
    objective_mask = np.asarray([1, 0, 1], dtype=bool)
    target_mask = _resolve_train_outer_batch_snapshot_target_mask(
        objective_mask=objective_mask,
        flat_env_indices=np.asarray([0, 0, 1], dtype=np.int64),
        flat_objective_episode_indices=np.asarray([0, -1, 1], dtype=np.int64),
        flat_objective_episode_positions=np.asarray([0, -1, 3], dtype=np.int64),
    )

    assert target_mask is None


def test_build_train_outer_batch_selector_trace_row_reports_target_position_range_absent():
    row = _build_train_outer_batch_selector_trace_row(
        objective_mask=np.asarray([1, 1, 1, 0], dtype=bool),
        flat_env_indices=np.asarray([12, 12, 7, 12], dtype=np.int64),
        flat_objective_episode_indices=np.asarray([0, 0, 0, -1], dtype=np.int64),
        flat_objective_episode_positions=np.asarray([10, 11, 2, -1], dtype=np.int64),
        epoch_idx=1,
        outer_batch_idx=3,
        total_outer_batches=8,
        selector={
            "target_outer_batch_idx": None,
            "target_env_index": 12,
            "target_objective_episode_index": 0,
            "target_objective_position_start": 18,
            "target_objective_position_end": 21,
        },
    )

    assert row["target_status"] == "target_position_range_absent"
    assert row["target_env_objective_token_count"] == 2
    assert row["target_episode_token_count"] == 2
    assert row["target_match_count"] == 0
    assert row["target_env_objective_episode_spans"] == [
        {
            "objective_episode_index": 0,
            "token_count": 2,
            "objective_position_min": 10,
            "objective_position_max": 11,
        }
    ]


def test_build_train_outer_batch_selector_trace_row_reports_target_present():
    row = _build_train_outer_batch_selector_trace_row(
        objective_mask=np.asarray([1, 1, 1, 1], dtype=bool),
        flat_env_indices=np.asarray([0, 12, 12, 12], dtype=np.int64),
        flat_objective_episode_indices=np.asarray([1, 0, 0, 1], dtype=np.int64),
        flat_objective_episode_positions=np.asarray([4, 18, 19, 0], dtype=np.int64),
        epoch_idx=1,
        outer_batch_idx=2,
        total_outer_batches=8,
        selector={
            "target_outer_batch_idx": None,
            "target_env_index": 12,
            "target_objective_episode_index": 0,
            "target_objective_position_start": 18,
            "target_objective_position_end": 21,
        },
    )

    assert row["target_status"] == "target_present"
    assert row["target_match_count"] == 2
    assert row["target_match_positions"] == [18, 19]
