import numpy as np

from ticl.analysis.phase3_pair2_train_segment_identity_probe import (
    _build_env_objective_identity_rows,
    _summarize_rows_by_key,
)


def test_build_env_objective_identity_rows_distinguishes_raw_vs_compressed_episode_identity():
    episode_starts = np.asarray(
        [
            [1.0],
            [0.0],
            [1.0],
            [0.0],
            [1.0],
            [0.0],
            [1.0],
            [0.0],
        ],
        dtype=np.float32,
    )
    objective_masks = np.asarray(
        [
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [1.0],
            [1.0],
            [1.0],
            [1.0],
        ],
        dtype=np.float32,
    )

    rows = _build_env_objective_identity_rows(
        episode_starts=episode_starts,
        objective_masks=objective_masks,
        env_index=0,
    )

    assert rows == [
        {
            "global_step": 4,
            "episode_start": 1,
            "raw_objective_episode_index": 2,
            "raw_objective_episode_position": 0,
            "raw_objective_global_position": 0,
            "compressed_objective_episode_index": 0,
            "compressed_objective_episode_position": 0,
        },
        {
            "global_step": 5,
            "episode_start": 0,
            "raw_objective_episode_index": 2,
            "raw_objective_episode_position": 1,
            "raw_objective_global_position": 1,
            "compressed_objective_episode_index": 0,
            "compressed_objective_episode_position": 1,
        },
        {
            "global_step": 6,
            "episode_start": 1,
            "raw_objective_episode_index": 3,
            "raw_objective_episode_position": 0,
            "raw_objective_global_position": 2,
            "compressed_objective_episode_index": 1,
            "compressed_objective_episode_position": 0,
        },
        {
            "global_step": 7,
            "episode_start": 0,
            "raw_objective_episode_index": 3,
            "raw_objective_episode_position": 1,
            "raw_objective_global_position": 3,
            "compressed_objective_episode_index": 1,
            "compressed_objective_episode_position": 1,
        },
    ]


def test_summarize_rows_by_key_groups_episode_spans():
    rows = [
        {"global_step": 10, "raw_objective_episode_index": 3, "raw_objective_episode_position": 0},
        {"global_step": 11, "raw_objective_episode_index": 3, "raw_objective_episode_position": 1},
        {"global_step": 15, "raw_objective_episode_index": 4, "raw_objective_episode_position": 0},
    ]

    spans = _summarize_rows_by_key(
        rows,
        episode_key="raw_objective_episode_index",
        position_key="raw_objective_episode_position",
    )

    assert spans == [
        {
            "episode_index": 3,
            "token_count": 2,
            "global_step_min": 10,
            "global_step_max": 11,
            "position_min": 0,
            "position_max": 1,
        },
        {
            "episode_index": 4,
            "token_count": 1,
            "global_step_min": 15,
            "global_step_max": 15,
            "position_min": 0,
            "position_max": 0,
        },
    ]
