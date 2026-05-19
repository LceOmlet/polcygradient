import numpy as np
from gymnasium import spaces

from ticl.analysis.phase2_gym_prior_pack_isomorphism_sentinel import (
    _clear_rollout_buffer_payload_refs,
    _fill_existing_rollout_buffer_from_pack,
)
from ticl.sb3_recurrent_ppo import MaskedRecurrentRolloutBuffer


def test_fill_existing_rollout_buffer_from_pack_bulk_materializes_fields():
    n_steps = 5
    n_envs = 3
    num_features = 7
    action_dim = 4
    next_state_dim = 6
    single_eval_pos = 2

    rng = np.random.default_rng(123)
    tokens = rng.normal(size=(n_steps, n_envs, num_features)).astype(np.float32)
    actions = rng.normal(size=(n_steps, n_envs, action_dim)).astype(np.float32)
    action_masks = rng.integers(0, 2, size=(n_steps, n_envs, action_dim)).astype(np.float32)
    rewards = rng.normal(size=(n_steps, n_envs)).astype(np.float32)
    episode_starts = np.zeros((n_steps, n_envs), dtype=np.float32)
    episode_starts[0] = 1.0
    dones = np.zeros((n_steps, n_envs), dtype=bool)
    dones[-1, 1] = True
    truncated = np.zeros_like(dones)
    next_states = rng.normal(size=(n_steps, n_envs, next_state_dim)).astype(np.float32)
    next_state_masks = rng.integers(0, 2, size=(n_steps, n_envs, next_state_dim)).astype(np.float32)
    values = rng.normal(size=(n_steps, n_envs)).astype(np.float32)
    log_probs = rng.normal(size=(n_steps, n_envs)).astype(np.float32)
    last_values = rng.normal(size=(n_envs,)).astype(np.float32)

    buffer = MaskedRecurrentRolloutBuffer(
        buffer_size=n_steps,
        observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(num_features,), dtype=np.float32),
        action_space=spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32),
        hidden_state_shape=(n_steps, 1, n_envs, 1),
        device="cpu",
        gae_lambda=0.95,
        gamma=0.97,
        n_envs=n_envs,
        next_state_dim=next_state_dim,
    )

    summary = _fill_existing_rollout_buffer_from_pack(
        buffer,
        {
            "tokens": tokens,
            "actions": actions,
            "action_masks": action_masks,
            "rewards": rewards,
            "episode_starts": episode_starts,
            "dones": dones,
            "truncated": truncated,
            "next_state_targets": next_states,
            "next_state_masks": next_state_masks,
        },
        num_features=num_features,
        action_slot_dim=action_dim,
        next_state_target_dim=next_state_dim,
        single_eval_pos=single_eval_pos,
        values_by_step_env=values,
        log_probs_by_step_env=log_probs,
        last_values_by_env=last_values,
        compute_returns=True,
    )

    assert summary["buffer_full"] is True
    assert buffer.full is True
    assert buffer.pos == n_steps
    np.testing.assert_allclose(buffer.observations, tokens)
    np.testing.assert_allclose(buffer.actions, actions)
    np.testing.assert_allclose(buffer.rewards, rewards)
    np.testing.assert_allclose(buffer.episode_starts, episode_starts)
    np.testing.assert_allclose(buffer.values, values)
    np.testing.assert_allclose(buffer.log_probs, log_probs)
    np.testing.assert_allclose(buffer.action_masks, action_masks)
    np.testing.assert_allclose(buffer.next_states, next_states)
    np.testing.assert_allclose(buffer.next_state_masks, next_state_masks)
    np.testing.assert_allclose(buffer.objective_masks[:single_eval_pos], 0.0)
    np.testing.assert_allclose(buffer.objective_masks[single_eval_pos:], 1.0)
    assert np.isfinite(buffer.advantages).all()
    assert np.isfinite(buffer.returns).all()


def test_fill_existing_rollout_buffer_from_pack_default_path_resets_flat_storage():
    n_steps = 5
    n_envs = 3
    num_features = 7
    action_dim = 4
    single_eval_pos = 2

    rng = np.random.default_rng(321)
    tokens = rng.normal(size=(n_steps, n_envs, num_features)).astype(np.float32)
    actions = rng.normal(size=(n_steps, n_envs, action_dim)).astype(np.float32)
    rewards = rng.normal(size=(n_steps, n_envs)).astype(np.float32)
    episode_starts = np.zeros((n_steps, n_envs), dtype=np.float32)
    episode_starts[0] = 1.0
    values = rng.normal(size=(n_steps, n_envs)).astype(np.float32)
    log_probs = rng.normal(size=(n_steps, n_envs)).astype(np.float32)
    last_values = rng.normal(size=(n_envs,)).astype(np.float32)

    buffer = MaskedRecurrentRolloutBuffer(
        buffer_size=n_steps,
        observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(num_features,), dtype=np.float32),
        action_space=spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32),
        hidden_state_shape=(n_steps, 1, n_envs, 1),
        device="cpu",
        gae_lambda=0.95,
        gamma=0.97,
        n_envs=n_envs,
        next_state_dim=0,
    )
    buffer.reset()
    buffer._ensure_generator_ready()
    assert tuple(buffer.observations.shape) == (n_steps * n_envs, num_features)

    summary = _fill_existing_rollout_buffer_from_pack(
        buffer,
        {
            "tokens": tokens,
            "actions": actions,
            "action_masks": np.ones((n_steps, n_envs, action_dim), dtype=np.float32),
            "rewards": rewards,
            "episode_starts": episode_starts,
            "dones": np.zeros((n_steps, n_envs), dtype=bool),
            "truncated": np.zeros((n_steps, n_envs), dtype=bool),
        },
        num_features=num_features,
        action_slot_dim=action_dim,
        next_state_target_dim=0,
        single_eval_pos=single_eval_pos,
        values_by_step_env=values,
        log_probs_by_step_env=log_probs,
        last_values_by_env=last_values,
        compute_returns=True,
        summary_mode="light",
        allow_missing_next_state_targets=True,
    )

    assert summary["take_ownership"] is False
    assert summary["buffer_full"] is True
    np.testing.assert_allclose(buffer.observations, tokens)
    np.testing.assert_allclose(buffer.actions, actions)
    np.testing.assert_allclose(buffer.rewards, rewards)
    np.testing.assert_allclose(buffer.episode_starts, episode_starts)
    np.testing.assert_allclose(buffer.values, values)
    np.testing.assert_allclose(buffer.log_probs, log_probs)
    assert np.isfinite(buffer.advantages).all()
    assert np.isfinite(buffer.returns).all()


def test_fill_existing_rollout_buffer_from_pack_can_take_ownership():
    n_steps = 5
    n_envs = 3
    num_features = 7
    action_dim = 4
    next_state_dim = 6
    single_eval_pos = 2

    rng = np.random.default_rng(456)
    tokens = rng.normal(size=(n_steps, n_envs, num_features)).astype(np.float32)
    actions = rng.normal(size=(n_steps, n_envs, action_dim)).astype(np.float32)
    action_masks = rng.integers(0, 2, size=(n_steps, n_envs, action_dim)).astype(np.float32)
    rewards = rng.normal(size=(n_steps, n_envs)).astype(np.float32)
    episode_starts = np.zeros((n_steps, n_envs), dtype=np.float32)
    episode_starts[0] = 1.0
    dones = np.zeros((n_steps, n_envs), dtype=bool)
    truncated = np.zeros_like(dones)
    next_states = rng.normal(size=(n_steps, n_envs, next_state_dim)).astype(np.float32)
    next_state_masks = rng.integers(0, 2, size=(n_steps, n_envs, next_state_dim)).astype(np.float32)
    values = rng.normal(size=(n_steps, n_envs)).astype(np.float32)
    log_probs = rng.normal(size=(n_steps, n_envs)).astype(np.float32)
    last_values = rng.normal(size=(n_envs,)).astype(np.float32)
    expected = {
        "tokens": tokens.copy(),
        "actions": actions.copy(),
        "action_masks": action_masks.copy(),
        "rewards": rewards.copy(),
        "episode_starts": episode_starts.copy(),
        "next_states": next_states.copy(),
        "next_state_masks": next_state_masks.copy(),
    }
    pack = {
        "tokens": tokens,
        "actions": actions,
        "action_masks": action_masks,
        "rewards": rewards,
        "episode_starts": episode_starts,
        "dones": dones,
        "truncated": truncated,
        "next_state_targets": next_states,
        "next_state_masks": next_state_masks,
    }

    buffer = MaskedRecurrentRolloutBuffer(
        buffer_size=n_steps,
        observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(num_features,), dtype=np.float32),
        action_space=spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32),
        hidden_state_shape=(n_steps, 1, n_envs, 1),
        device="cpu",
        gae_lambda=0.95,
        gamma=0.97,
        n_envs=n_envs,
        next_state_dim=next_state_dim,
    )

    summary = _fill_existing_rollout_buffer_from_pack(
        buffer,
        pack,
        num_features=num_features,
        action_slot_dim=action_dim,
        next_state_target_dim=next_state_dim,
        single_eval_pos=single_eval_pos,
        values_by_step_env=values,
        log_probs_by_step_env=log_probs,
        last_values_by_env=last_values,
        compute_returns=True,
        summary_mode="light",
        take_ownership=True,
    )

    assert summary["take_ownership"] is True
    assert set(summary["transferred_keys"]) >= {
        "tokens",
        "actions",
        "action_masks",
        "rewards",
        "episode_starts",
        "next_state_targets",
        "next_state_masks",
    }
    for key in ["tokens", "actions", "action_masks", "rewards", "episode_starts", "next_state_targets", "next_state_masks"]:
        assert key not in pack
    np.testing.assert_allclose(buffer.observations, expected["tokens"])
    np.testing.assert_allclose(buffer.actions, expected["actions"])
    np.testing.assert_allclose(buffer.rewards, expected["rewards"])
    np.testing.assert_allclose(buffer.episode_starts, expected["episode_starts"])
    np.testing.assert_allclose(buffer.action_masks, expected["action_masks"])
    np.testing.assert_allclose(buffer.next_states, expected["next_states"])
    np.testing.assert_allclose(buffer.next_state_masks, expected["next_state_masks"])
    np.testing.assert_allclose(buffer.values, values)
    np.testing.assert_allclose(buffer.log_probs, log_probs)
    assert np.isfinite(buffer.advantages).all()
    assert np.isfinite(buffer.returns).all()


def test_take_ownership_fill_can_reuse_released_rollout_buffer():
    n_steps = 5
    n_envs = 3
    num_features = 7
    action_dim = 4
    single_eval_pos = 2

    rng = np.random.default_rng(789)
    values = rng.normal(size=(n_steps, n_envs)).astype(np.float32)
    log_probs = rng.normal(size=(n_steps, n_envs)).astype(np.float32)
    last_values = rng.normal(size=(n_envs,)).astype(np.float32)
    episode_starts = np.zeros((n_steps, n_envs), dtype=np.float32)
    episode_starts[0] = 1.0
    pack = {
        "tokens": rng.normal(size=(n_steps, n_envs, num_features)).astype(np.float32),
        "actions": rng.normal(size=(n_steps, n_envs, action_dim)).astype(np.float32),
        "action_masks": rng.integers(0, 2, size=(n_steps, n_envs, action_dim)).astype(np.float32),
        "rewards": rng.normal(size=(n_steps, n_envs)).astype(np.float32),
        "episode_starts": episode_starts,
        "dones": np.zeros((n_steps, n_envs), dtype=bool),
        "truncated": np.zeros((n_steps, n_envs), dtype=bool),
        "last_values": last_values,
    }

    buffer = MaskedRecurrentRolloutBuffer(
        buffer_size=n_steps,
        observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(num_features,), dtype=np.float32),
        action_space=spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32),
        hidden_state_shape=(n_steps, 1, n_envs, 1),
        device="cpu",
        gae_lambda=0.95,
        gamma=0.97,
        n_envs=n_envs,
        next_state_dim=0,
    )
    _clear_rollout_buffer_payload_refs(buffer)
    assert buffer.episode_starts is None

    summary = _fill_existing_rollout_buffer_from_pack(
        buffer,
        pack,
        num_features=num_features,
        action_slot_dim=action_dim,
        next_state_target_dim=0,
        single_eval_pos=single_eval_pos,
        values_by_step_env=values,
        log_probs_by_step_env=log_probs,
        last_values_by_env=last_values,
        compute_returns=True,
        summary_mode="light",
        take_ownership=True,
        allow_missing_next_state_targets=True,
    )

    assert summary["buffer_full"] is True
    assert buffer.episode_starts is not None
    np.testing.assert_allclose(buffer.episode_starts, episode_starts)
    assert np.isfinite(buffer.advantages).all()
    assert np.isfinite(buffer.returns).all()
