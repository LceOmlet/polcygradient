import json

from ticl.analysis.phase3_regime_env_concentration_probe import build_regime_env_concentration_probe


def test_regime_env_concentration_probe_detects_drowning(tmp_path):
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
                    "env_seed": 10,
                    "rollout_seed": 20,
                    "objective_terminal_reset_count": 2,
                    "objective_token_count": 100,
                    "pre_vs_zero_suffix_gap": -1.0,
                    "post_reset_tail_dist_1": 7.903580227831925,
                    "post_reset_tail_dist_2": 0.0,
                    "post_reset_tail_dist_3": 0.0,
                    "post_reset_tail_dist_4": 0.0,
                },
                {
                    "env_index": 1,
                    "env_seed": 11,
                    "rollout_seed": 21,
                    "objective_terminal_reset_count": 1,
                    "objective_token_count": 100,
                    "pre_vs_zero_suffix_gap": 0.0,
                    "post_reset_tail_dist_1": 7.301065192264159,
                    "post_reset_tail_dist_2": 0.0,
                    "post_reset_tail_dist_3": 0.0,
                    "post_reset_tail_dist_4": 0.0,
                },
                {
                    "env_index": 2,
                    "env_seed": 12,
                    "rollout_seed": 22,
                    "objective_terminal_reset_count": 0,
                    "objective_token_count": 100,
                    "pre_vs_zero_suffix_gap": 0.0,
                    "post_reset_tail_dist_1": 5.034366512630339,
                    "post_reset_tail_dist_2": 0.0,
                    "post_reset_tail_dist_3": 0.0,
                    "post_reset_tail_dist_4": 0.0,
                },
                {
                    "env_index": 3,
                    "env_seed": 13,
                    "rollout_seed": 23,
                    "objective_terminal_reset_count": 0,
                    "objective_token_count": 100,
                    "pre_vs_zero_suffix_gap": -1.0,
                    "post_reset_tail_dist_1": 0.8279955630837121,
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
                    "env_seed": 30,
                    "rollout_seed": 40,
                    "objective_terminal_reset_count": 2,
                    "objective_token_count": 100,
                    "pre_vs_zero_suffix_gap": -0.1672156298328913,
                    "post_reset_tail_dist_1": 2.338922928994947,
                    "post_reset_tail_dist_2": 0.0,
                    "post_reset_tail_dist_3": 0.0,
                    "post_reset_tail_dist_4": 0.0,
                },
                {
                    "env_index": 1,
                    "env_seed": 31,
                    "rollout_seed": 41,
                    "objective_terminal_reset_count": 1,
                    "objective_token_count": 100,
                    "pre_vs_zero_suffix_gap": -0.2857160759072189,
                    "post_reset_tail_dist_1": 1.7331217272981085,
                    "post_reset_tail_dist_2": 0.0,
                    "post_reset_tail_dist_3": 0.0,
                    "post_reset_tail_dist_4": 0.0,
                },
                {
                    "env_index": 2,
                    "env_seed": 32,
                    "rollout_seed": 42,
                    "objective_terminal_reset_count": 1,
                    "objective_token_count": 100,
                    "pre_vs_zero_suffix_gap": -0.4323731335216162,
                    "post_reset_tail_dist_1": 1.2236940375725665,
                    "post_reset_tail_dist_2": 0.0,
                    "post_reset_tail_dist_3": 0.0,
                    "post_reset_tail_dist_4": 0.0,
                },
                {
                    "env_index": 3,
                    "env_seed": 33,
                    "rollout_seed": 43,
                    "objective_terminal_reset_count": 0,
                    "objective_token_count": 100,
                    "pre_vs_zero_suffix_gap": -0.6899601715481356,
                    "post_reset_tail_dist_1": 1.0223185324676018,
                    "post_reset_tail_dist_2": 0.0,
                    "post_reset_tail_dist_3": 0.0,
                    "post_reset_tail_dist_4": 0.0,
                },
            ]
        },
    )

    result = build_regime_env_concentration_probe(
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
    assert result["conclusions"]["nonpositive_local_signal_positive"] is True
    assert result["conclusions"]["nonpositive_signal_flips_under_global_centering"] is True
    assert result["conclusions"]["positive_regime_mass_share_exceeds_nonpositive"] is True
    assert result["conclusions"]["positive_regime_abs_pooled_cov_dominates_nonpositive"] is True
    assert result["conclusions"]["positive_regime_few_envs_dominate_mass"] is True
    assert result["conclusions"]["largest_positive_suite_dominates_regime_mass"] is True
    assert result["conclusions"]["nonpositive_signal_drowned_by_mass_and_centering"] is True
