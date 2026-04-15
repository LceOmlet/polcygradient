import json

from ticl.analysis.phase3_positive_regime_anchor_structure_probe import (
    build_positive_regime_anchor_structure_probe,
)


def test_positive_regime_anchor_structure_probe_separates_captured_and_missed(tmp_path):
    def write(name: str, payload: dict) -> str:
        path = tmp_path / name
        path.write_text(json.dumps(payload))
        return str(path)

    zero = write(
        "zero.json",
        {
            "heldout_suite": {
                "metrics": {
                    "suffix_return_mean": 0.0,
                    "suffix_return_per_env": [0.0, 0.0, 0.0],
                    "full_return_per_env": [0.0, 0.0, 0.0],
                }
            }
        },
    )
    ppo = write(
        "ppo.json",
        {
            "heldout_suite": {
                "metrics": {
                    "suffix_return_mean": 1.0,
                    "suffix_return_per_env": [5.0, 3.0, 2.0],
                    "full_return_per_env": [10.0, 8.0, 4.0],
                }
            }
        },
    )
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
                    "pre_vs_zero_suffix_gap": 3.0,
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
                    "pre_vs_zero_suffix_gap": 0.2,
                    "post_reset_tail_dist_1": 0.2,
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
                            "post_reset_terminal_tail": {"removed_positive_mass": 4.0},
                            "post_reset_only": {"removed_positive_mass": 6.0},
                            "first_episode_only": {"removed_positive_mass": 4.0},
                            "terminal_tail": {"removed_positive_mass": 7.0},
                        }
                    },
                },
                {
                    "env_index": 1,
                    "base_positive_mass": 3.0,
                    "tail_len_rows": {
                        "8": {
                            "post_reset_terminal_tail": {"removed_positive_mass": 0.0},
                            "post_reset_only": {"removed_positive_mass": 0.0},
                            "first_episode_only": {"removed_positive_mass": 2.4},
                            "terminal_tail": {"removed_positive_mass": 0.3},
                        }
                    },
                },
                {
                    "env_index": 2,
                    "base_positive_mass": 4.0,
                    "tail_len_rows": {
                        "8": {
                            "post_reset_terminal_tail": {"removed_positive_mass": 1.0},
                            "post_reset_only": {"removed_positive_mass": 1.0},
                            "first_episode_only": {"removed_positive_mass": 2.0},
                            "terminal_tail": {"removed_positive_mass": 1.0},
                        }
                    },
                },
            ]
        },
    )

    result = build_positive_regime_anchor_structure_probe(
        [
            {
                "suite_name": "synthetic",
                "zero_json": zero,
                "ppo_json": ppo,
                "distance_probe_json": distance,
                "tail_mask_probe_json": tail,
            }
        ],
        anchor_min_pre_gap=1.0,
        captured_min_mass_share_within_suite=0.05,
    )

    assert result["strong_gain_anchor_count"] == 2
    assert result["captured_summary"]["count"] == 1
    assert result["missed_summary"]["count"] == 1
    assert result["conclusions"]["captured_gain_anchors_exist"] is True
    assert result["conclusions"]["proxy_missed_gain_anchors_exist"] is True
    assert result["conclusions"]["missed_gain_anchors_still_positive_on_full_return"] is True
    assert result["conclusions"]["captured_anchors_are_more_reset_heavy"] is True
    assert result["conclusions"]["captured_anchors_are_more_post_reset_tail_weighted"] is True
    assert result["conclusions"]["missed_anchors_are_more_first_episode_skewed"] is True
    assert result["conclusions"]["current_proxy_underweights_real_gain_anchors"] is True
