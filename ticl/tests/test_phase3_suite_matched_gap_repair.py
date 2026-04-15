import json

from ticl.analysis.phase3_suite_matched_gap_repair import repair_artifact


def test_repair_distance_probe_rewrites_pre_gap_and_summaries(tmp_path):
    def write(name: str, payload: dict) -> str:
        path = tmp_path / name
        path.write_text(json.dumps(payload))
        return str(path)

    artifact = write(
        "distance.json",
        {
            "audit_entry": "phase3_post_reset_tail_distance_probe",
            "env_rows": [
                {
                    "env_index": 0,
                    "objective_terminal_reset_count": 1,
                    "pre_vs_zero_suffix_gap": 0.0,
                    "post_reset_tail_dist_1": 1.0,
                    "post_reset_tail_dist_2": 0.0,
                    "post_reset_tail_dist_3": 0.0,
                    "post_reset_tail_dist_4": 0.0,
                    "post_reset_tail_dist_5": 0.0,
                    "post_reset_tail_dist_6": 0.0,
                    "post_reset_tail_dist_7": 0.0,
                    "post_reset_tail_dist_8": 0.0,
                    "post_reset_early_k1": 0.0,
                    "post_reset_early_k2": 0.0,
                    "post_reset_early_k4": 0.0,
                    "post_reset_late_k1": 1.0,
                    "post_reset_late_k2": 1.0,
                    "post_reset_late_k4": 1.0,
                },
                {
                    "env_index": 1,
                    "objective_terminal_reset_count": 0,
                    "pre_vs_zero_suffix_gap": 0.0,
                    "post_reset_tail_dist_1": 0.0,
                    "post_reset_tail_dist_2": 0.0,
                    "post_reset_tail_dist_3": 0.0,
                    "post_reset_tail_dist_4": 0.0,
                    "post_reset_tail_dist_5": 0.0,
                    "post_reset_tail_dist_6": 0.0,
                    "post_reset_tail_dist_7": 0.0,
                    "post_reset_tail_dist_8": 0.0,
                    "post_reset_early_k1": 0.0,
                    "post_reset_early_k2": 0.0,
                    "post_reset_early_k4": 0.0,
                    "post_reset_late_k1": 0.0,
                    "post_reset_late_k2": 0.0,
                    "post_reset_late_k4": 0.0,
                },
            ],
            "summaries": {},
        },
    )
    zero = write(
        "zero.json",
        {
            "heldout_suite": {
                "metrics": {
                    "suffix_return_per_env": [0.0, 1.0],
                }
            }
        },
    )
    ppo = write(
        "ppo.json",
        {
            "heldout_suite": {
                "metrics": {
                    "suffix_return_per_env": [2.0, 1.5],
                }
            }
        },
    )

    repaired = repair_artifact(artifact_json=artifact, zero_json=zero, ppo_json=ppo)
    assert repaired["repair_metadata"]["mismatch_count"] == 2
    assert repaired["env_rows"][0]["pre_vs_zero_suffix_gap"] == 2.0
    assert repaired["env_rows"][1]["pre_vs_zero_suffix_gap"] == 0.5
    assert repaired["summaries"]["post_reset_tail_dist_1"]["corr_vs_pre_suffix_gap"] > 0.0


def test_repair_tailmask_probe_rewrites_pre_gap_and_summaries(tmp_path):
    def write(name: str, payload: dict) -> str:
        path = tmp_path / name
        path.write_text(json.dumps(payload))
        return str(path)

    artifact = write(
        "tailmask.json",
        {
            "audit_entry": "phase3_terminal_tail_mask_probe",
            "env_rows": [
                {
                    "env_index": 0,
                    "objective_terminal_reset_count": 1,
                    "pre_vs_zero_suffix_gap": 0.0,
                    "raw_value_return_corr": 0.0,
                    "base_positive_mass": 2.0,
                    "tail_len_rows": {
                        "1": {
                            "terminal_tail": {"remaining_positive_mass": 1.0, "removed_positive_mass": 1.0},
                            "first_terminal_tail": {"remaining_positive_mass": 1.0, "removed_positive_mass": 1.0},
                            "post_reset_terminal_tail": {"remaining_positive_mass": 1.0, "removed_positive_mass": 1.0},
                            "first_episode_only": {"remaining_positive_mass": 1.0, "removed_positive_mass": 1.0},
                            "post_reset_only": {"remaining_positive_mass": 1.0, "removed_positive_mass": 1.0},
                        },
                        "2": {
                            "terminal_tail": {"remaining_positive_mass": 1.0, "removed_positive_mass": 1.0},
                            "first_terminal_tail": {"remaining_positive_mass": 1.0, "removed_positive_mass": 1.0},
                            "post_reset_terminal_tail": {"remaining_positive_mass": 1.0, "removed_positive_mass": 1.0},
                            "first_episode_only": {"remaining_positive_mass": 1.0, "removed_positive_mass": 1.0},
                            "post_reset_only": {"remaining_positive_mass": 1.0, "removed_positive_mass": 1.0},
                        },
                        "4": {
                            "terminal_tail": {"remaining_positive_mass": 1.0, "removed_positive_mass": 1.0},
                            "first_terminal_tail": {"remaining_positive_mass": 1.0, "removed_positive_mass": 1.0},
                            "post_reset_terminal_tail": {"remaining_positive_mass": 1.0, "removed_positive_mass": 1.0},
                            "first_episode_only": {"remaining_positive_mass": 1.0, "removed_positive_mass": 1.0},
                            "post_reset_only": {"remaining_positive_mass": 1.0, "removed_positive_mass": 1.0},
                        },
                        "8": {
                            "terminal_tail": {"remaining_positive_mass": 1.0, "removed_positive_mass": 1.0},
                            "first_terminal_tail": {"remaining_positive_mass": 1.0, "removed_positive_mass": 1.0},
                            "post_reset_terminal_tail": {"remaining_positive_mass": 1.0, "removed_positive_mass": 1.0},
                            "first_episode_only": {"remaining_positive_mass": 1.0, "removed_positive_mass": 1.0},
                            "post_reset_only": {"remaining_positive_mass": 1.0, "removed_positive_mass": 1.0},
                        },
                    },
                },
                {
                    "env_index": 1,
                    "objective_terminal_reset_count": 0,
                    "pre_vs_zero_suffix_gap": 0.0,
                    "raw_value_return_corr": 0.0,
                    "base_positive_mass": 0.0,
                    "tail_len_rows": {
                        "1": {
                            "terminal_tail": {"remaining_positive_mass": 0.0, "removed_positive_mass": 0.0},
                            "first_terminal_tail": {"remaining_positive_mass": 0.0, "removed_positive_mass": 0.0},
                            "post_reset_terminal_tail": {"remaining_positive_mass": 0.0, "removed_positive_mass": 0.0},
                            "first_episode_only": {"remaining_positive_mass": 0.0, "removed_positive_mass": 0.0},
                            "post_reset_only": {"remaining_positive_mass": 0.0, "removed_positive_mass": 0.0},
                        },
                        "2": {
                            "terminal_tail": {"remaining_positive_mass": 0.0, "removed_positive_mass": 0.0},
                            "first_terminal_tail": {"remaining_positive_mass": 0.0, "removed_positive_mass": 0.0},
                            "post_reset_terminal_tail": {"remaining_positive_mass": 0.0, "removed_positive_mass": 0.0},
                            "first_episode_only": {"remaining_positive_mass": 0.0, "removed_positive_mass": 0.0},
                            "post_reset_only": {"remaining_positive_mass": 0.0, "removed_positive_mass": 0.0},
                        },
                        "4": {
                            "terminal_tail": {"remaining_positive_mass": 0.0, "removed_positive_mass": 0.0},
                            "first_terminal_tail": {"remaining_positive_mass": 0.0, "removed_positive_mass": 0.0},
                            "post_reset_terminal_tail": {"remaining_positive_mass": 0.0, "removed_positive_mass": 0.0},
                            "first_episode_only": {"remaining_positive_mass": 0.0, "removed_positive_mass": 0.0},
                            "post_reset_only": {"remaining_positive_mass": 0.0, "removed_positive_mass": 0.0},
                        },
                        "8": {
                            "terminal_tail": {"remaining_positive_mass": 0.0, "removed_positive_mass": 0.0},
                            "first_terminal_tail": {"remaining_positive_mass": 0.0, "removed_positive_mass": 0.0},
                            "post_reset_terminal_tail": {"remaining_positive_mass": 0.0, "removed_positive_mass": 0.0},
                            "first_episode_only": {"remaining_positive_mass": 0.0, "removed_positive_mass": 0.0},
                            "post_reset_only": {"remaining_positive_mass": 0.0, "removed_positive_mass": 0.0},
                        },
                    },
                },
            ],
            "summaries": {},
        },
    )
    zero = write(
        "zero.json",
        {
            "heldout_suite": {
                "metrics": {
                    "suffix_return_per_env": [0.0, 1.0],
                }
            }
        },
    )
    ppo = write(
        "ppo.json",
        {
            "heldout_suite": {
                "metrics": {
                    "suffix_return_per_env": [3.0, 1.5],
                }
            }
        },
    )

    repaired = repair_artifact(artifact_json=artifact, zero_json=zero, ppo_json=ppo)
    assert repaired["repair_metadata"]["mismatch_count"] == 2
    assert repaired["env_rows"][0]["pre_vs_zero_suffix_gap"] == 3.0
    assert repaired["env_rows"][1]["pre_vs_zero_suffix_gap"] == 0.5
    assert repaired["summaries"]["8"]["post_reset_only"]["corr_vs_pre_suffix_gap"] > 0.0
