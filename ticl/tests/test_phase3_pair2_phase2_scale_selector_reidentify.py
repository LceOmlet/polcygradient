from ticl.analysis.phase3_pair2_phase2_scale_selector_reidentify import (
    _candidate_rows,
    _pick_canonical_selector,
)


def test_candidate_rows_extract_present_episode_span() -> None:
    trace = {
        "rows": [
            {
                "epoch_idx": 1,
                "outer_batch_idx": 13,
                "target_env_index": 12,
                "target_status": "target_position_range_absent",
                "target_env_objective_token_count": 192,
                "target_env_objective_episode_indices_present": [8, 9, 10],
                "target_objective_episode_index": 10,
                "target_episode_token_count": 8,
                "target_episode_positions_present": list(range(8)),
            }
        ]
    }
    rows = _candidate_rows(trace)
    assert len(rows) == 1
    assert rows[0]["candidate_position_start"] == 0
    assert rows[0]["candidate_position_end"] == 7
    assert rows[0]["candidate_token_count"] == 8


def test_pick_canonical_selector_prefers_earliest_outer_batch() -> None:
    candidates = [
        {
            "epoch_idx": 1,
            "outer_batch_idx": 14,
            "target_env_index": 12,
            "target_objective_episode_index": 10,
            "candidate_position_start": 0,
            "candidate_position_end": 7,
            "candidate_token_count": 8,
        },
        {
            "epoch_idx": 1,
            "outer_batch_idx": 13,
            "target_env_index": 12,
            "target_objective_episode_index": 10,
            "candidate_position_start": 2,
            "candidate_position_end": 5,
            "candidate_token_count": 4,
        },
    ]
    selected = _pick_canonical_selector(candidates)
    assert selected["outer_batch_idx"] == 13
