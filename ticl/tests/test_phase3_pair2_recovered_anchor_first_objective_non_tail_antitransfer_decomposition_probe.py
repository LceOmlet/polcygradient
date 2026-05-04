from ticl.analysis.phase3_pair2_recovered_anchor_first_objective_non_tail_antitransfer_decomposition_probe import (
    _build_post_gap_delta_rows,
    _mask_overlap_summary,
    _summarize_delta_group,
)


def test_build_post_gap_delta_rows_uses_post_vs_zero_contract():
    baseline = {
        "zero_control": {
            "heldout": {
                "full_return_per_env": [1.0, 2.0],
                "suffix_return_per_env": [0.5, 1.5],
            }
        },
        "post": {
            "heldout": {
                "full_return_per_env": [3.0, 1.0],
                "suffix_return_per_env": [2.0, 0.0],
            }
        },
    }
    override = {
        "post": {
            "heldout": {
                "full_return_per_env": [4.0, 0.0],
                "suffix_return_per_env": [2.5, -0.5],
            }
        }
    }

    rows = _build_post_gap_delta_rows(
        baseline_report=baseline,
        override_report=override,
        split="heldout",
    )

    assert rows == [
        {
            "env_index": 0,
            "baseline_post_vs_zero_full": 2.0,
            "override_post_vs_zero_full": 3.0,
            "full_gap_delta": 1.0,
            "baseline_post_vs_zero_suffix": 1.5,
            "override_post_vs_zero_suffix": 2.0,
            "suffix_gap_delta": 0.5,
        },
        {
            "env_index": 1,
            "baseline_post_vs_zero_full": -1.0,
            "override_post_vs_zero_full": -2.0,
            "full_gap_delta": -1.0,
            "baseline_post_vs_zero_suffix": -1.5,
            "override_post_vs_zero_suffix": -2.0,
            "suffix_gap_delta": -0.5,
        },
    ]


def test_summarize_delta_group_can_show_spillover_dominance():
    rows = [
        {"env_index": 0, "full_gap_delta": -0.08, "suffix_gap_delta": -0.10},
        {"env_index": 15, "full_gap_delta": 0.00, "suffix_gap_delta": 0.00},
        {"env_index": 3, "full_gap_delta": 0.09, "suffix_gap_delta": 0.09},
        {"env_index": 4, "full_gap_delta": 0.06, "suffix_gap_delta": 0.05},
    ]

    target_group = _summarize_delta_group(rows, env_indices=[0, 15])
    complement_group = _summarize_delta_group(rows, env_indices=[3, 4])

    assert target_group["suffix_gap_delta_sum"] < 0.0
    assert complement_group["suffix_gap_delta_sum"] > abs(target_group["suffix_gap_delta_sum"])


def test_mask_overlap_summary_detects_semantic_runtime_drift():
    objective_mask = [True, True, True, True, True]
    env_indices = [0, 13, 15, 0, 13]
    objective_episode_indices = [0, 0, 2, 0, 0]
    raw_actor_advantages = [-3.0, -2.0, -1.0, 1.0, -4.0]
    semantic_mask = [True, True, True, False, True]
    runtime_mask = [True, False, False, False, False]

    summary = _mask_overlap_summary(
        semantic_mask=semantic_mask,
        runtime_mask=runtime_mask,
        objective_mask=objective_mask,
        env_indices=env_indices,
        objective_episode_indices=objective_episode_indices,
        raw_actor_advantages=raw_actor_advantages,
    )

    assert summary["runtime_selector_matches_semantic_family_exact"] is False
    assert summary["semantic_env_indices"] == [0, 13, 15]
    assert summary["runtime_env_indices"] == [0]
    assert summary["runtime_token_share_of_semantic"] == 0.25
    assert summary["overlap_negative_share_of_semantic"] > 0.0
    assert summary["overlap_negative_share_of_semantic"] < 1.0
