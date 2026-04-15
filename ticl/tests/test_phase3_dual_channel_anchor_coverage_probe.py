import json

from ticl.analysis.phase3_dual_channel_anchor_coverage_probe import (
    build_dual_channel_anchor_coverage_probe,
)


def test_dual_channel_anchor_coverage_probe_recovers_first_episode_non_tail_anchors(tmp_path):
    def write(name: str, payload: dict) -> str:
        path = tmp_path / name
        path.write_text(json.dumps(payload))
        return str(path)

    distance = write(
        "distance.json",
        {
            "env_rows": [
                {
                    "env_index": 0,
                    "env_seed": 10,
                    "rollout_seed": 20,
                    "objective_terminal_reset_count": 2,
                    "objective_token_count": 100,
                    "pre_vs_zero_suffix_gap": 5.0,
                    "post_reset_tail_dist_1": 4.0,
                    "post_reset_tail_dist_2": 0.0,
                    "post_reset_tail_dist_3": 0.0,
                    "post_reset_tail_dist_4": 0.0,
                },
                {
                    "env_index": 1,
                    "env_seed": 11,
                    "rollout_seed": 21,
                    "objective_terminal_reset_count": 0,
                    "objective_token_count": 100,
                    "pre_vs_zero_suffix_gap": 4.0,
                    "post_reset_tail_dist_1": 0.0,
                    "post_reset_tail_dist_2": 0.0,
                    "post_reset_tail_dist_3": 0.0,
                    "post_reset_tail_dist_4": 0.0,
                },
                {
                    "env_index": 2,
                    "env_seed": 12,
                    "rollout_seed": 22,
                    "objective_terminal_reset_count": 1,
                    "objective_token_count": 100,
                    "pre_vs_zero_suffix_gap": 3.0,
                    "post_reset_tail_dist_1": 0.0,
                    "post_reset_tail_dist_2": 0.0,
                    "post_reset_tail_dist_3": 0.0,
                    "post_reset_tail_dist_4": 0.0,
                },
                {
                    "env_index": 3,
                    "env_seed": 13,
                    "rollout_seed": 23,
                    "objective_terminal_reset_count": 1,
                    "objective_token_count": 100,
                    "pre_vs_zero_suffix_gap": 2.0,
                    "post_reset_tail_dist_1": 0.0,
                    "post_reset_tail_dist_2": 0.0,
                    "post_reset_tail_dist_3": 0.0,
                    "post_reset_tail_dist_4": 0.0,
                },
            ]
        },
    )
    tail = write(
        "tail.json",
        {
            "env_rows": [
                {
                    "env_index": 0,
                    "base_positive_mass": 10.0,
                    "tail_len_rows": {
                        "8": {
                            "first_episode_only": {"removed_positive_mass": 4.0},
                            "first_terminal_tail": {"removed_positive_mass": 1.0},
                        }
                    },
                },
                {
                    "env_index": 1,
                    "base_positive_mass": 5.0,
                    "tail_len_rows": {
                        "8": {
                            "first_episode_only": {"removed_positive_mass": 4.0},
                            "first_terminal_tail": {"removed_positive_mass": 0.5},
                        }
                    },
                },
                {
                    "env_index": 2,
                    "base_positive_mass": 0.2,
                    "tail_len_rows": {
                        "8": {
                            "first_episode_only": {"removed_positive_mass": 0.05},
                            "first_terminal_tail": {"removed_positive_mass": 0.01},
                        }
                    },
                },
                {
                    "env_index": 3,
                    "base_positive_mass": 0.0,
                    "tail_len_rows": {
                        "8": {
                            "first_episode_only": {"removed_positive_mass": 0.0},
                            "first_terminal_tail": {"removed_positive_mass": 0.0},
                        }
                    },
                },
            ]
        },
    )

    result = build_dual_channel_anchor_coverage_probe(
        [
            {
                "suite_name": "synthetic",
                "distance_probe_json": distance,
                "tail_mask_probe_json": tail,
            }
        ],
        strong_gain_min_pre_gap=1.0,
        coverage_share_threshold=0.2,
    )

    assert result["strong_gain_anchor_count"] == 4
    assert result["baseline_reset_tail_summary"]["count"] == 1
    assert result["dual_channel_summary"]["count"] == 2
    assert result["fused_sum_summary"]["count"] == 2
    assert result["recovered_anchor_summary"]["count"] == 1
    assert result["residual_missed_summary"]["count"] == 2
    assert result["conclusions"]["dual_channel_improves_anchor_coverage"] is True
    assert result["conclusions"]["dual_channel_recovers_nonreset_first_episode_anchors"] is True
    assert result["conclusions"]["residual_missed_anchors_remain"] is True
    assert result["conclusions"]["residual_missed_anchors_have_low_first_episode_non_tail_signal"] is True
    assert result["conclusions"]["residual_missed_anchors_have_low_or_zero_positive_mass"] is True
