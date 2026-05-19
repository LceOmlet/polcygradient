import numpy as np

from ticl.analysis.phase2_gym_prior_pack_training_runner import (
    _accumulate_per_env_obs_stats,
    _finalize_per_env_obs_stats,
    _new_per_env_obs_stats,
    _next_prior_obs_reward_token_after_step,
    _per_env_semantic_probe_rows_from_pack,
    _set_exact_scm_family_source,
)


def test_next_prior_obs_reward_token_clears_after_done():
    scaled = np.asarray([1.5, -2.0, 0.25], dtype=np.float32)
    dones = np.asarray([False, True, False], dtype=bool)

    np.testing.assert_allclose(
        _next_prior_obs_reward_token_after_step(scaled, dones),
        np.asarray([1.5, 0.0, 0.25], dtype=np.float32),
    )


def test_exact_scm_cost_markov_runner_keeps_rejected_terminal_independent_explicit():
    env_cfg = {}

    resolved = _set_exact_scm_family_source(env_cfg, "gym_lowtail_cost_markov")

    assert resolved == "gym_lowtail_cost_markov"
    assert env_cfg["exact_scm_gym_lowtail_family_prob"] == 0.25
    assert env_cfg["exact_scm_gym_lowtail_terminal_override_enabled"] is True

    env_cfg = {}
    resolved = _set_exact_scm_family_source(env_cfg, "gym_lowtail_potential_markov")

    assert resolved == "gym_lowtail_potential_markov"
    assert env_cfg["exact_scm_gym_lowtail_family_prob"] == 0.25
    assert env_cfg["exact_scm_gym_lowtail_terminal_override_enabled"] is True
    assert env_cfg["exact_scm_gym_lowtail_reward_linear_weight"] == 0.0
    assert env_cfg["exact_scm_gym_lowtail_reward_potential_delta_weight"] == 1.0
    assert env_cfg["exact_scm_gym_lowtail_reward_state_cost"] == 0.02

    env_cfg = {}
    resolved = _set_exact_scm_family_source(env_cfg, "gym_lowtail_potential_cost_markov")

    assert resolved == "gym_lowtail_potential_cost_markov"
    assert env_cfg["exact_scm_gym_lowtail_family_prob"] == 0.25
    assert env_cfg["exact_scm_gym_lowtail_terminal_override_enabled"] is True
    assert env_cfg["exact_scm_gym_lowtail_reward_linear_weight"] == 0.0
    assert env_cfg["exact_scm_gym_lowtail_reward_potential_delta_weight"] == 0.5
    assert env_cfg["exact_scm_gym_lowtail_reward_state_cost"] == 0.5
    assert env_cfg["exact_scm_gym_lowtail_reward_ctrl_cost"] == 0.5

    env_cfg = {}
    resolved = _set_exact_scm_family_source(
        env_cfg,
        "gym_lowtail_cost_markov_terminal_independent",
    )

    assert resolved == "gym_lowtail_cost_markov_terminal_independent"
    assert env_cfg["exact_scm_gym_lowtail_family_prob"] == 0.25
    assert env_cfg["exact_scm_gym_lowtail_terminal_override_enabled"] is False

    env_cfg = {}
    resolved = _set_exact_scm_family_source(env_cfg, "gym_lowtail_reward_transform_chain_preimage")

    assert resolved == "gym_lowtail_reward_transform_chain_preimage"
    assert env_cfg["exact_scm_gym_lowtail_state_obs_rank_preimage_enabled"] is True
    assert env_cfg["exact_scm_gym_lowtail_state_obs_rank_preimage_rank"] == 6
    assert env_cfg["exact_scm_gym_lowtail_reward_distribution_head_enabled"] is True
    assert env_cfg["exact_scm_gym_lowtail_shared_energy_terminal_enabled"] is True
    assert env_cfg["reinforce_reward_transform"] == "tanh"
    assert env_cfg["reward_dropout_enabled"] is False

    env_cfg = {}
    resolved = _set_exact_scm_family_source(env_cfg, "gym_lowtail_clock_preimage")

    assert resolved == "gym_lowtail_clock_preimage"
    assert env_cfg["exact_scm_gym_lowtail_state_obs_rank_preimage_enabled"] is True
    assert env_cfg["exact_scm_gym_lowtail_state_obs_rank_preimage_rank"] == 6
    assert env_cfg["exact_scm_gym_lowtail_state_obs_rank_preimage_innovation_scale"] == 1.5
    assert env_cfg["exact_scm_gym_lowtail_clock_preimage_enabled"] is True
    assert env_cfg["exact_scm_gym_lowtail_family_prob"] == 1.0
    assert env_cfg["exact_scm_gym_lowtail_action_gain"] == 0.25
    assert env_cfg["exact_scm_gym_lowtail_noise_gain"] == 0.10
    assert env_cfg["exact_scm_gym_lowtail_reward_linear_weight"] == 0.03
    assert env_cfg["exact_scm_gym_lowtail_reward_potential_delta_weight"] == 0.0
    assert env_cfg["exact_scm_gym_lowtail_reward_state_cost"] == 0.0
    assert env_cfg["exact_scm_gym_lowtail_reward_ctrl_cost"] == 0.0
    assert env_cfg["exact_scm_gym_lowtail_reward_distribution_head_enabled"] is True
    assert env_cfg["exact_scm_gym_lowtail_reward_distribution_head_scale"] == 0.0
    assert env_cfg["exact_scm_gym_lowtail_reward_distribution_head_vol_scale"] == 0.0
    assert env_cfg["reinforce_reward_transform"] == "tanh"

    env_cfg = {}
    resolved = _set_exact_scm_family_source(env_cfg, "gym_lowtail_paired_env_response_preimage")

    assert resolved == "gym_lowtail_paired_env_response_preimage"
    assert env_cfg["exact_scm_gym_lowtail_family_prob"] == 1.0
    assert env_cfg["exact_scm_gym_lowtail_state_obs_rank_preimage_enabled"] is True
    assert env_cfg["exact_scm_gym_lowtail_state_obs_temporal_preimage_enabled"] is True
    assert env_cfg["exact_scm_gym_lowtail_state_obs_rank_preimage_rank"] == 6
    assert env_cfg["exact_scm_gym_lowtail_reward_distribution_head_enabled"] is True
    assert env_cfg["exact_scm_gym_lowtail_shared_energy_terminal_enabled"] is False
    assert env_cfg["exact_scm_gym_lowtail_paired_response_preimage_enabled"] is True
    assert env_cfg["reinforce_reward_transform"] == "tanh"

    env_cfg = {}
    resolved = _set_exact_scm_family_source(env_cfg, "gym_lowtail_markov_carrier_preimage")

    assert resolved == "gym_lowtail_markov_carrier_preimage"
    assert env_cfg["exact_scm_gym_lowtail_family_prob"] == 1.0
    assert env_cfg["exact_scm_gym_lowtail_state_obs_rank_preimage_enabled"] is True
    assert env_cfg["exact_scm_gym_lowtail_state_obs_temporal_preimage_enabled"] is True
    assert env_cfg["exact_scm_gym_lowtail_state_obs_hetero_scale_preimage_enabled"] is True
    assert env_cfg["exact_scm_gym_lowtail_markov_carrier_preimage_enabled"] is True
    assert env_cfg["exact_scm_gym_lowtail_state_obs_rank_preimage_rank"] == 6
    assert env_cfg["exact_scm_gym_lowtail_reward_distribution_head_enabled"] is True
    assert env_cfg["exact_scm_gym_lowtail_paired_response_preimage_enabled"] is True
    assert env_cfg["reinforce_reward_transform"] == "tanh"

    env_cfg = {}
    resolved = _set_exact_scm_family_source(env_cfg, "gym_lowtail_paired_obs_hetero_preimage")

    assert resolved == "gym_lowtail_paired_obs_hetero_preimage"
    assert env_cfg["exact_scm_gym_lowtail_family_prob"] == 1.0
    assert env_cfg["exact_scm_gym_lowtail_state_obs_rank_preimage_enabled"] is True
    assert env_cfg["exact_scm_gym_lowtail_state_obs_temporal_preimage_enabled"] is True
    assert env_cfg["exact_scm_gym_lowtail_state_obs_hetero_scale_preimage_enabled"] is True
    assert env_cfg["exact_scm_gym_lowtail_state_obs_rank_preimage_rank"] == 6
    assert env_cfg["exact_scm_gym_lowtail_reward_distribution_head_enabled"] is True
    assert env_cfg["exact_scm_gym_lowtail_paired_response_preimage_enabled"] is True
    assert env_cfg["reinforce_reward_transform"] == "tanh"


def test_semantic_sidecar_reward_token_relation_respects_done_reset():
    n_steps = 4
    n_envs = 1
    num_features = 6
    obs_slot_dim = 2
    action_slot_dim = 1
    tokens = np.zeros((n_steps, n_envs, num_features), dtype=np.float32)
    rewards = np.asarray([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32)
    dones = np.asarray([[False], [True], [False], [False]], dtype=bool)
    tokens[:, 0, obs_slot_dim] = np.asarray([0.0, 1.0, 0.0, 3.0], dtype=np.float32)
    pack = {
        "tokens": tokens,
        "actions": np.zeros((n_steps, n_envs, action_slot_dim), dtype=np.float32),
        "action_masks": np.ones((n_steps, n_envs, action_slot_dim), dtype=np.float32),
        "rewards": rewards,
        "raw_rewards": rewards.copy(),
        "dones": dones,
        "truncated": np.zeros_like(dones),
        "next_state_targets": np.zeros((n_steps, n_envs, obs_slot_dim), dtype=np.float32),
        "next_state_masks": np.ones((n_steps, n_envs, obs_slot_dim), dtype=np.float32),
    }

    rows = _per_env_semantic_probe_rows_from_pack(
        arm="prior",
        update_idx=0,
        pack=pack,
        meta={"obs_dims": [obs_slot_dim], "obs_slot_dims": [obs_slot_dim], "action_dims": [action_slot_dim]},
        obs_slot_dim=obs_slot_dim,
        action_slot_dim=action_slot_dim,
    )

    assert rows[0]["input_reward_token_prev_training_reward_mae"] == 0.0
    assert rows[0]["input_reward_token_done_reset_absmean"] == 0.0
    assert rows[0]["input_reward_token_done_reset_nonzero_fraction"] == 0.0


def test_per_env_obs_stats_trace_pre_normalized_active_dims_only():
    acc = _new_per_env_obs_stats(n_envs=2, obs_slot_dim=3)
    _accumulate_per_env_obs_stats(
        acc,
        np.asarray([[1.0, 3.0, 100.0], [2.0, 0.0, 8.0]], dtype=np.float32),
        obs_dims=[2, 3],
        obs_slot_dim=3,
    )
    _accumulate_per_env_obs_stats(
        acc,
        np.asarray([[3.0, 7.0, 200.0], [4.0, 0.0, 10.0]], dtype=np.float32),
        obs_dims=[2, 3],
        obs_slot_dim=3,
    )

    meta = _finalize_per_env_obs_stats(acc, obs_dims=[2, 3], obs_slot_dim=3, prefix="pre_norm_obs")

    assert meta["pre_norm_obs_active_value_count"] == [4, 6]
    np.testing.assert_allclose(meta["pre_norm_obs_per_dim_std_mean"][0], 1.5)
    np.testing.assert_allclose(meta["pre_norm_obs_per_dim_std_q90"][0], 1.9)
    np.testing.assert_allclose(meta["pre_norm_obs_value_absmax"][0], 7.0)
    np.testing.assert_allclose(meta["pre_norm_obs_per_dim_std_mean"][1], 2.0 / 3.0)


def test_semantic_sidecar_includes_pre_norm_obs_meta():
    n_steps = 3
    n_envs = 1
    num_features = 6
    obs_slot_dim = 2
    action_slot_dim = 1
    pack = {
        "tokens": np.zeros((n_steps, n_envs, num_features), dtype=np.float32),
        "actions": np.zeros((n_steps, n_envs, action_slot_dim), dtype=np.float32),
        "action_masks": np.ones((n_steps, n_envs, action_slot_dim), dtype=np.float32),
        "rewards": np.zeros((n_steps, n_envs), dtype=np.float32),
        "raw_rewards": np.zeros((n_steps, n_envs), dtype=np.float32),
        "dones": np.zeros((n_steps, n_envs), dtype=bool),
        "truncated": np.zeros((n_steps, n_envs), dtype=bool),
        "next_state_targets": np.zeros((n_steps, n_envs, obs_slot_dim), dtype=np.float32),
        "next_state_masks": np.ones((n_steps, n_envs, obs_slot_dim), dtype=np.float32),
    }
    pack["tokens"][:, 0, :obs_slot_dim] = np.asarray([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]], dtype=np.float32)

    rows = _per_env_semantic_probe_rows_from_pack(
        arm="prior",
        update_idx=0,
        pack=pack,
        meta={
            "obs_dims": [obs_slot_dim],
            "obs_slot_dims": [obs_slot_dim],
            "action_dims": [action_slot_dim],
            "pre_norm_obs_value_std": [2.5],
            "pre_norm_obs_per_dim_std_q90": [3.5],
            "obs_norm_state_rms_per_dim_mean": [4.5],
            "exact_scm_gym_lowtail_family_enabled": [1.0],
            "exact_scm_gym_lowtail_family_active": [1.0],
            "exact_scm_gym_lowtail_reward_potential_delta_weight": [1.0],
            "exact_scm_gym_lowtail_reward_state_cost": [1.0],
            "exact_scm_gym_lowtail_reward_ctrl_cost": [1.0],
        },
        obs_slot_dim=obs_slot_dim,
        action_slot_dim=action_slot_dim,
    )

    assert rows[0]["pre_norm_obs_value_std"] == 2.5
    assert rows[0]["pre_norm_obs_per_dim_std_q90"] == 3.5
    assert rows[0]["obs_norm_state_rms_per_dim_mean"] == 4.5
    assert rows[0]["exact_scm_gym_lowtail_family_enabled"] == 1.0
    assert rows[0]["exact_scm_gym_lowtail_family_active"] == 1.0
    assert rows[0]["exact_scm_gym_lowtail_reward_potential_delta_weight"] == 1.0
    assert rows[0]["exact_scm_gym_lowtail_reward_state_cost"] == 1.0
    assert rows[0]["exact_scm_gym_lowtail_reward_ctrl_cost"] == 1.0
    expected_delta = np.sqrt(12.5)
    np.testing.assert_allclose(rows[0]["obs_step_l2_delta_mean"], expected_delta)
    np.testing.assert_allclose(
        rows[0]["obs_step_l2_delta_to_rms_ratio"],
        expected_delta / np.sqrt(125.0 / 6.0),
    )
