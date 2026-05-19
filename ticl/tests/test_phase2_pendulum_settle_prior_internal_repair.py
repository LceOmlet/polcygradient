import copy

import pytest
import torch

from ticl.priors.environment_prior import EnvironmentPrior


def _base_env_cfg(**overrides):
    cfg = {
        "family": "scm",
        "action_dim": 1,
        "state_dim": 2,
        "obs_dim": 2,
        "noise_dim": 1,
        "zero_pad_dim": 0,
        "obs_slot_dim": 2,
        "action_slot_dim": 1,
        "num_layers": 2,
        "prior_mlp_hidden_dim": 8,
        "prior_mlp_activations": "tanh",
        "init_std": 0.1,
        "noise_std": 0.0,
        "alpha": 1.0,
        "init_state_std": 0.1,
        "init_action_std": 0.0,
        "state_noise_std": 0.0,
        "action_noise_train_std": 0.0,
        "action_noise_eval_std": 0.0,
        "reward_scale": 1.0,
        "reward_clip": 100.0,
        "state_clip": 100.0,
        "state_output_scale": 1.0,
        "state_full_rms_enabled": False,
        "state_highway_enabled": False,
        "reference_state_inertia_enabled": True,
        "ctrl_reward_weight": 0.0,
        "ctrl_reward_enable_prob": 0.0,
        "survival_reward_weight": 0.0,
        "survival_reward_enable_prob": 0.0,
        "terminal_reset_enabled": False,
        "reward_dropout_enabled": False,
        "reinforce_action_transform": "none",
        "reinforce_reward_transform": "none",
        "pendulum_settle_yield_dt": 0.10,
        "pendulum_settle_yield_damping": 0.90,
        "pendulum_settle_yield_spring": 0.0,
        "pendulum_settle_yield_position_coupling": 0.0,
        "pendulum_settle_yield_action_controls_position": False,
        "pendulum_settle_yield_action_gain": 0.50,
        "pendulum_settle_yield_action_bound": 1.0,
        "pendulum_settle_yield_state_mix": 1.0,
        "pendulum_settle_yield_reward_mix": 1.0,
        "pendulum_settle_yield_reward_bias": 1.0,
        "pendulum_settle_yield_q_weight": 1.0,
        "pendulum_settle_yield_v_weight": 0.25,
        "pendulum_settle_yield_ctrl_weight": 0.02,
    }
    cfg.update(overrides)
    return cfg


def _transition_reward(env, *, q, v, action):
    x = torch.zeros((1, int(env["env_input_dim"])), dtype=torch.float32)
    x[0, 0] = float(q)
    x[0, 1] = float(v)
    x[0, int(env["env_action_start"])] = float(action)
    state_next, reward_next = env["transition_generator"](x)
    return state_next, reward_next


def test_prior_internal_settle_repair_is_default_off():
    prior = EnvironmentPrior(_base_env_cfg())
    h = prior._sample_batch_hypers(1)[0]
    env = prior._sample_environment_once(h, device="cpu", rng_seed=0)

    assert env["pendulum_settle_yield_repair_enabled"] is False
    assert env["pendulum_settle_yield_repair_active"] is False
    assert env["pendulum_settle_yield_repair_standalone_generator"] is False
    assert getattr(env["transition_generator"], "_pendulum_settle_yield_repair_enabled", False) is False


def test_prior_internal_settle_repair_prob_enables_existing_slot_patch():
    prior = EnvironmentPrior(
        _base_env_cfg(
            pendulum_settle_yield_repair_enabled=False,
            pendulum_settle_yield_repair_prob=1.0,
        )
    )
    h = prior._sample_batch_hypers(1)[0]
    env = prior._sample_environment_once(h, device="cpu", rng_seed=0)

    assert env["pendulum_settle_yield_repair_enabled"] is True
    assert env["pendulum_settle_yield_repair_active"] is True
    assert env["pendulum_settle_yield_repair_uses_existing_state_action_slots"] is True
    assert env["pendulum_settle_yield_repair_standalone_generator"] is False
    assert env["pendulum_settle_yield_repair_reward_path_summary_valid"] is False
    assert "reward_state_input_gain_fraction" not in env
    assert getattr(env["transition_generator"], "_pendulum_settle_yield_repair_standalone_generator") is False


def test_prior_internal_settle_repair_rewards_stabilizing_feedback_action():
    prior = EnvironmentPrior(_base_env_cfg(pendulum_settle_yield_repair_enabled=True))
    h = prior._sample_batch_hypers(1)[0]
    env = prior._sample_environment_once(h, device="cpu", rng_seed=0)

    state_stabilize, reward_stabilize = _transition_reward(env, q=0.0, v=1.0, action=-1.0)
    state_destabilize, reward_destabilize = _transition_reward(env, q=0.0, v=1.0, action=1.0)

    assert state_stabilize[0, 1].abs() < state_destabilize[0, 1].abs()
    assert reward_stabilize.item() > reward_destabilize.item() + 0.20


def test_prior_internal_settle_repair_spring_couples_state_to_velocity():
    prior = EnvironmentPrior(
        _base_env_cfg(
            pendulum_settle_yield_repair_enabled=True,
            pendulum_settle_yield_damping=1.0,
            pendulum_settle_yield_spring=0.50,
            pendulum_settle_yield_action_gain=0.0,
            pendulum_settle_yield_ctrl_weight=0.0,
        )
    )
    h = prior._sample_batch_hypers(1)[0]
    env = prior._sample_environment_once(h, device="cpu", rng_seed=0)

    state_next, reward_next = _transition_reward(env, q=1.0, v=0.0, action=0.0)

    assert state_next[0, 0].item() == pytest.approx(1.0)
    assert state_next[0, 1].item() == pytest.approx(-0.50)
    assert reward_next.item() == pytest.approx(-0.0625)
    assert getattr(env["transition_generator"], "_pendulum_settle_yield_repair_spring_coupling") is True


def test_prior_internal_settle_repair_position_coupling_makes_action_affect_next_position():
    prior_explicit = EnvironmentPrior(
        _base_env_cfg(
            pendulum_settle_yield_repair_enabled=True,
            pendulum_settle_yield_dt=0.10,
            pendulum_settle_yield_damping=1.0,
            pendulum_settle_yield_spring=0.0,
            pendulum_settle_yield_position_coupling=0.0,
            pendulum_settle_yield_action_gain=1.0,
            pendulum_settle_yield_ctrl_weight=0.0,
        )
    )
    prior_coupled = EnvironmentPrior(
        _base_env_cfg(
            pendulum_settle_yield_repair_enabled=True,
            pendulum_settle_yield_dt=0.10,
            pendulum_settle_yield_damping=1.0,
            pendulum_settle_yield_spring=0.0,
            pendulum_settle_yield_position_coupling=1.0,
            pendulum_settle_yield_action_gain=1.0,
            pendulum_settle_yield_ctrl_weight=0.0,
        )
    )
    h_explicit = prior_explicit._sample_batch_hypers(1)[0]
    h_coupled = prior_coupled._sample_batch_hypers(1)[0]
    env_explicit = prior_explicit._sample_environment_once(h_explicit, device="cpu", rng_seed=0)
    env_coupled = prior_coupled._sample_environment_once(h_coupled, device="cpu", rng_seed=0)

    state_explicit, reward_explicit = _transition_reward(env_explicit, q=1.0, v=0.0, action=-1.0)
    state_coupled, reward_coupled = _transition_reward(env_coupled, q=1.0, v=0.0, action=-1.0)

    assert state_explicit[0, 0].item() == pytest.approx(1.0)
    assert state_coupled[0, 0].item() < state_explicit[0, 0].item()
    assert reward_coupled.item() > reward_explicit.item()
    assert getattr(env_coupled["transition_generator"], "_pendulum_settle_yield_repair_position_coupling") is True


def test_prior_internal_settle_repair_position_action_control_is_directly_maintainable():
    prior = EnvironmentPrior(
        _base_env_cfg(
            pendulum_settle_yield_repair_enabled=True,
            pendulum_settle_yield_dt=0.10,
            pendulum_settle_yield_damping=0.90,
            pendulum_settle_yield_spring=0.0,
            pendulum_settle_yield_position_coupling=0.0,
            pendulum_settle_yield_action_controls_position=True,
            pendulum_settle_yield_action_gain=1.0,
            pendulum_settle_yield_ctrl_weight=0.0,
        )
    )
    h = prior._sample_batch_hypers(1)[0]
    env = prior._sample_environment_once(h, device="cpu", rng_seed=0)

    state_push, reward_push = _transition_reward(env, q=1.0, v=0.0, action=-1.0)
    state_keep, reward_keep = _transition_reward(env, q=0.0, v=0.0, action=0.0)
    state_wrong, reward_wrong = _transition_reward(env, q=1.0, v=0.0, action=1.0)

    assert state_push[0, 0].item() < 1.0
    assert state_keep[0, 0].item() == pytest.approx(0.0)
    assert state_keep[0, 1].item() == pytest.approx(0.0)
    assert reward_keep.item() == pytest.approx(1.0)
    assert reward_push.item() > reward_wrong.item()
    assert getattr(env["transition_generator"], "_pendulum_settle_yield_repair_action_controls_position") is True


def test_prior_internal_settle_repair_family_coarse_batch_is_per_sample():
    prior = EnvironmentPrior(_base_env_cfg())
    h_list = prior._sample_batch_hypers(2)
    h_list[0] = copy.deepcopy(h_list[0])
    h_list[1] = copy.deepcopy(h_list[1])
    h_list[0]["pendulum_settle_yield_repair_enabled"] = True
    h_list[1]["pendulum_settle_yield_repair_enabled"] = False

    env = prior._sample_environment_family_coarse_batch(
        h_list,
        device="cpu",
        rng_seeds=[11, 12],
        build_policy_generator=False,
    )

    assert env["pendulum_settle_yield_repair_enabled"].tolist() == [True, False]
    assert env["pendulum_settle_yield_repair_active"].tolist() == [True, False]

    x_pos = torch.zeros((2, int(env["env_input_dim"])), dtype=torch.float32)
    x_neg = torch.zeros_like(x_pos)
    x_pos[:, 1] = 1.0
    x_neg[:, 1] = 1.0
    action_start = int(env["state_dim"])
    x_pos[:, action_start] = 1.0
    x_neg[:, action_start] = -1.0

    _, reward_pos = env["transition_generator"](x_pos)
    _, reward_neg = env["transition_generator"](x_neg)

    assert reward_neg[0].item() > reward_pos[0].item() + 0.20


def test_prior_internal_settle_repair_runs_vectorized_rollout_smoke():
    prior = EnvironmentPrior(
        _base_env_cfg(
            batch_parallel_backend="torch_vectorized",
            batch_shared_environment=False,
            pendulum_settle_yield_repair_prob=1.0,
        )
    )

    x, y, _ = prior.get_batch(
        batch_size=2,
        n_samples=6,
        num_features=6,
        device="cpu",
        single_eval_pos=3,
    )

    assert x.shape == (6, 2, 6)
    assert y.shape == (6, 2)
    assert torch.isfinite(x).all()
    assert torch.isfinite(y).all()
    assert all(info["state_dim"] == 2 for info in prior.last_runtime_info)
