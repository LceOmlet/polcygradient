import json

from ticl.analysis.phase3_regime_mixed_update_probe import build_regime_mixed_update_probe


def test_regime_mixed_update_probe_detects_conflict(tmp_path):
    def write(name: str, payload: dict) -> str:
        path = tmp_path / name
        path.write_text(json.dumps(payload))
        return str(path)

    zero_pos = write(
        "zero_pos.json",
        {"heldout_suite": {"metrics": {"suffix_return_mean": 0.0, "full_return_mean": 0.0}}},
    )
    ppo_pos = write(
        "ppo_pos.json",
        {"heldout_suite": {"metrics": {"suffix_return_mean": 1.0, "full_return_mean": 1.0}}},
    )
    dist_pos = write(
        "dist_pos.json",
        {
            "env_rows": [
                {
                    "env_index": 0,
                    "objective_terminal_reset_count": 1,
                    "pre_vs_zero_suffix_gap": -0.5,
                    "post_reset_tail_dist_1": 2.0,
                    "post_reset_tail_dist_2": 1.0,
                    "post_reset_tail_dist_3": 0.0,
                    "post_reset_tail_dist_4": 0.0,
                },
                {
                    "env_index": 1,
                    "objective_terminal_reset_count": 0,
                    "pre_vs_zero_suffix_gap": 0.1,
                    "post_reset_tail_dist_1": 0.0,
                    "post_reset_tail_dist_2": 0.0,
                    "post_reset_tail_dist_3": 0.0,
                    "post_reset_tail_dist_4": 0.0,
                },
            ]
        },
    )
    zero_nonpos = write(
        "zero_nonpos.json",
        {"heldout_suite": {"metrics": {"suffix_return_mean": 1.0, "full_return_mean": 1.0}}},
    )
    ppo_nonpos = write(
        "ppo_nonpos.json",
        {"heldout_suite": {"metrics": {"suffix_return_mean": 0.5, "full_return_mean": 0.5}}},
    )
    dist_nonpos = write(
        "dist_nonpos.json",
        {
            "env_rows": [
                {
                    "env_index": 0,
                    "objective_terminal_reset_count": 1,
                    "pre_vs_zero_suffix_gap": 0.4,
                    "post_reset_tail_dist_1": 1.0,
                    "post_reset_tail_dist_2": 0.0,
                    "post_reset_tail_dist_3": 0.0,
                    "post_reset_tail_dist_4": 0.0,
                },
                {
                    "env_index": 1,
                    "objective_terminal_reset_count": 0,
                    "pre_vs_zero_suffix_gap": -0.1,
                    "post_reset_tail_dist_1": 0.0,
                    "post_reset_tail_dist_2": 0.0,
                    "post_reset_tail_dist_3": 0.0,
                    "post_reset_tail_dist_4": 0.0,
                },
            ]
        },
    )

    result = build_regime_mixed_update_probe(
        [
            {
                "suite_name": "pos",
                "zero_json": zero_pos,
                "ppo_json": ppo_pos,
                "distance_probe_json": dist_pos,
            },
            {
                "suite_name": "nonpos",
                "zero_json": zero_nonpos,
                "ppo_json": ppo_nonpos,
                "distance_probe_json": dist_nonpos,
            },
        ]
    )

    assert result["suite_count"] == 2
    assert result["conclusions"]["reset_tracking_stable_across_regimes"] is True
    assert result["conclusions"]["pre_gap_alignment_flips_by_regime"] is True
    assert result["conclusions"]["regime_split_visible_in_raw_correlation"] is True
    assert result["conclusions"]["pooled_sign_hides_regime_split"] is True
    assert result["conclusions"]["positive_regime_dominates_pooled_update"] is True
    assert result["conclusions"]["mixed_update_conflict_present"] is True
