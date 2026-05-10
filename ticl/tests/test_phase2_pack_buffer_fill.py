import numpy as np
from gymnasium import spaces

from ticl.analysis.phase2_gym_prior_pack_isomorphism_sentinel import _fill_existing_rollout_buffer_from_pack
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
