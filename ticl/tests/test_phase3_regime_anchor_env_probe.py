import json

from ticl.analysis.phase3_regime_anchor_env_probe import build_regime_anchor_env_probe


def test_regime_anchor_env_probe_flags_sparse_pair1_mass_and_pair2_proxy_outlier(tmp_path):
    def write(name: str, payload: dict) -> str:
        path = tmp_path / name
        path.write_text(json.dumps(payload))
        return str(path)

    pair1_zero = write(
        "pair1_zero.json",
        {
            "heldout_suite": {
                "summary": {"fingerprint": "pair1", "suite_seed": 1},
                "metrics": {
                    "rollout_backend": "serial",
                    "batch_size": 3,
                    "n_samples": 8,
                    "single_eval_pos": 64,
                    "suffix_horizon": 8,
                    "full_return_per_env": [0.0, 0.0, 0.0],
                    "suffix_return_per_env": [0.0, 0.0, 0.0],
                    "full_return_nonfinite_count": 0,
                    "suffix_return_nonfinite_count": 0,
                    "reward_step_nonfinite_count": 0,
                },
            }
        },
    )
    pair1_ppo = write(
        "pair1_ppo.json",
        {
            "heldout_suite": {
                "summary": {"fingerprint": "pair1", "suite_seed": 1},
                "metrics": {
                    "rollout_backend": "serial",
                    "batch_size": 3,
                    "n_samples": 8,
                    "single_eval_pos": 64,
                    "suffix_horizon": 8,
                    "full_return_per_env": [5.0, 4.0, 20.0],
                    "suffix_return_per_env": [5.0, 4.0, 20.0],
                    "full_return_nonfinite_count": 0,
                    "suffix_return_nonfinite_count": 0,
                    "reward_step_nonfinite_count": 0,
                },
            }
        },
    )
    pair1_distance = write(
        "pair1_distance.json",
        {
            "env_rows": [
                {
                    "env_index": 0,
                    "env_seed": 10,
                    "rollout_seed": 20,
                    "objective_terminal_reset_count": 2,
                    "objective_token_count": 100,
                    "pre_vs_zero_suffix_gap": 5.0,
                    "post_reset_tail_dist_1": 6.0,
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
                    "pre_vs_zero_suffix_gap": 4.0,
                    "post_reset_tail_dist_1": 4.0,
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
                    "pre_vs_zero_suffix_gap": 20.0,
                    "post_reset_tail_dist_1": 0.0,
                    "post_reset_tail_dist_2": 0.0,
                    "post_reset_tail_dist_3": 0.0,
                    "post_reset_tail_dist_4": 0.0,
                },
            ]
        },
    )

    pair2_zero = write(
        "pair2_zero.json",
        {
            "heldout_suite": {
                "summary": {"fingerprint": "pair2", "suite_seed": 2},
                "metrics": {
                    "rollout_backend": "serial",
                    "batch_size": 4,
                    "n_samples": 16,
                    "single_eval_pos": 1975,
                    "suffix_horizon": 8,
                    "full_return_per_env": [-10.0] * 13,
                    "suffix_return_per_env": [-10.0] * 13,
                    "full_return_nonfinite_count": 0,
                    "suffix_return_nonfinite_count": 0,
                    "reward_step_nonfinite_count": 0,
                },
            }
        },
    )
    pair2_ppo = write(
        "pair2_ppo.json",
        {
            "heldout_suite": {
                "summary": {"fingerprint": "pair2", "suite_seed": 2},
                "metrics": {
                    "rollout_backend": "serial",
                    "batch_size": 4,
                    "n_samples": 16,
                    "single_eval_pos": 1975,
                    "suffix_horizon": 8,
                    "full_return_per_env": [-10.0] * 13,
                    "suffix_return_per_env": [-10.0] * 13,
                    "full_return_nonfinite_count": 0,
                    "suffix_return_nonfinite_count": 0,
                    "reward_step_nonfinite_count": 0,
                },
            }
        },
    )
    pair2_zero_payload = json.loads((tmp_path / "pair2_zero.json").read_text())
    pair2_zero_payload["heldout_suite"]["metrics"]["full_return_per_env"][0] = -10.0
    pair2_zero_payload["heldout_suite"]["metrics"]["full_return_per_env"][1] = -8.0
    pair2_zero_payload["heldout_suite"]["metrics"]["full_return_per_env"][12] = -7.0
    pair2_zero_payload["heldout_suite"]["metrics"]["full_return_per_env"][3] = -6.0
    pair2_zero_payload["heldout_suite"]["metrics"]["suffix_return_per_env"][0] = -10.0
    pair2_zero_payload["heldout_suite"]["metrics"]["suffix_return_per_env"][1] = -8.0
    pair2_zero_payload["heldout_suite"]["metrics"]["suffix_return_per_env"][12] = -7.0
    pair2_zero_payload["heldout_suite"]["metrics"]["suffix_return_per_env"][3] = -6.0
    (tmp_path / "pair2_zero.json").write_text(json.dumps(pair2_zero_payload))

    pair2_ppo_payload = json.loads((tmp_path / "pair2_ppo.json").read_text())
    pair2_ppo_payload["heldout_suite"]["metrics"]["full_return_per_env"][0] = -11.0
    pair2_ppo_payload["heldout_suite"]["metrics"]["full_return_per_env"][1] = -7.0
    pair2_ppo_payload["heldout_suite"]["metrics"]["full_return_per_env"][12] = 14.0
    pair2_ppo_payload["heldout_suite"]["metrics"]["full_return_per_env"][3] = -5.5
    pair2_ppo_payload["heldout_suite"]["metrics"]["suffix_return_per_env"][0] = -11.0
    pair2_ppo_payload["heldout_suite"]["metrics"]["suffix_return_per_env"][1] = -7.0
    pair2_ppo_payload["heldout_suite"]["metrics"]["suffix_return_per_env"][12] = 14.0
    pair2_ppo_payload["heldout_suite"]["metrics"]["suffix_return_per_env"][3] = -5.5
    (tmp_path / "pair2_ppo.json").write_text(json.dumps(pair2_ppo_payload))
    pair2_distance = write(
        "pair2_distance.json",
        {
            "env_rows": [
                {
                    "env_index": 0,
                    "env_seed": 30,
                    "rollout_seed": 40,
                    "objective_terminal_reset_count": 2,
                    "objective_token_count": 100,
                    "pre_vs_zero_suffix_gap": -1.0,
                    "post_reset_tail_dist_1": 1.5,
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
                    "pre_vs_zero_suffix_gap": 1.0,
                    "post_reset_tail_dist_1": 0.2,
                    "post_reset_tail_dist_2": 0.0,
                    "post_reset_tail_dist_3": 0.0,
                    "post_reset_tail_dist_4": 0.0,
                },
                {
                    "env_index": 12,
                    "env_seed": 32,
                    "rollout_seed": 42,
                    "objective_terminal_reset_count": 1,
                    "objective_token_count": 100,
                    "pre_vs_zero_suffix_gap": 21.0,
                    "post_reset_tail_dist_1": 0.01,
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
                    "pre_vs_zero_suffix_gap": 0.5,
                    "post_reset_tail_dist_1": 0.0,
                    "post_reset_tail_dist_2": 0.0,
                    "post_reset_tail_dist_3": 0.0,
                    "post_reset_tail_dist_4": 0.0,
                },
            ]
        },
    )

    result = build_regime_anchor_env_probe(
        pair1_zero_json=pair1_zero,
        pair1_ppo_json=pair1_ppo,
        pair1_distance_json=pair1_distance,
        pair2_zero_json=pair2_zero,
        pair2_ppo_json=pair2_ppo,
        pair2_distance_json=pair2_distance,
    )

    assert result["conclusions"]["pair1_mass_concentration_is_sparse_reset_active"] is True
    assert result["conclusions"]["pair1_mass_proxy_not_exhaustive_for_gain"] is True
    assert result["conclusions"]["pair2_env12_metric_outlier_unlikely"] is True
    assert result["conclusions"]["pair2_env12_is_proxy_outlier"] is True
    assert result["pair2_anchor"]["env12"]["env_index"] == 12
