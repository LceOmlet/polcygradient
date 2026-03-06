import random
import numpy as np
import torch
from torch import nn

from ticl.model_configs import get_prior_config
from ticl.priors.environment_prior import EnvironmentPrior


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def test_environment_prior_state_highway_postprocess_toggle_semantics():
    prior = EnvironmentPrior({})
    state_prev = torch.tensor([[0.2, -0.3], [0.1, 0.4]], dtype=torch.float32)
    state_next_raw = torch.tensor([[9.0, -9.0], [0.5, -0.5]], dtype=torch.float32)
    state_clip = torch.tensor([8.0, 8.0], dtype=torch.float32)

    baseline = torch.tanh(torch.clamp(state_next_raw, -8.0, 8.0))

    out_off = prior._apply_state_postprocess(
        state_next_raw=state_next_raw,
        state_prev=state_prev,
        state_clip=state_clip,
        state_highway_enabled=False,
        state_highway_lambda=0.9,
    )
    assert torch.allclose(out_off, baseline)

    out_l0 = prior._apply_state_postprocess(
        state_next_raw=state_next_raw,
        state_prev=state_prev,
        state_clip=state_clip,
        state_highway_enabled=True,
        state_highway_lambda=0.0,
    )
    assert torch.allclose(out_l0, baseline)

    out_l1 = prior._apply_state_postprocess(
        state_next_raw=state_next_raw,
        state_prev=state_prev,
        state_clip=state_clip,
        state_highway_enabled=True,
        state_highway_lambda=1.0,
    )
    assert torch.allclose(out_l1, state_prev)


def test_environment_prior_state_highway_enabled_rollout_smoke():
    _seed_everything(123)
    config = get_prior_config()
    env_cfg = dict(config["prior"]["environment"])
    env_cfg["family"] = {"distribution": "meta_choice", "choice_values": ["scm"]}
    env_cfg["state_dim"] = {"distribution": "uniform_int", "min": 6, "max": 6}
    env_cfg["obs_dim"] = {"distribution": "uniform_int", "min": 4, "max": 4}
    env_cfg["action_dim"] = {"distribution": "uniform_int", "min": 3, "max": 3}
    env_cfg["noise_dim"] = {"distribution": "uniform_int", "min": 5, "max": 5}
    env_cfg["zero_pad_dim"] = {"distribution": "uniform_int", "min": 2, "max": 2}
    env_cfg["state_highway_enabled"] = True
    env_cfg["state_highway_lambda"] = 0.25
    prior = EnvironmentPrior(env_cfg)

    x, y, _ = prior.get_batch(
        batch_size=3,
        n_samples=16,
        num_features=24,
        device="cpu",
        single_eval_pos=8,
    )

    assert torch.isfinite(x).all()
    assert torch.isfinite(y).all()
    assert len(prior.last_runtime_info) == 3
    for row in prior.last_runtime_info:
        assert bool(row["state_highway_enabled"])
        assert abs(float(row["state_highway_lambda"]) - 0.25) < 1e-6


def test_environment_prior_shapes_and_finite():
    _seed_everything(42)
    config = get_prior_config()
    prior = EnvironmentPrior(config["prior"]["environment"])

    x, y, y_ = prior.get_batch(
        batch_size=5,
        n_samples=64,
        num_features=32,
        device="cpu",
        single_eval_pos=24,
    )

    assert x.shape == (64, 5, 32)
    assert y.shape == (64, 5)
    assert y_.shape == (64, 5)
    assert torch.isfinite(x).all()
    assert torch.isfinite(y).all()
    assert torch.isfinite(y_).all()


def test_environment_prior_dim_ranges_and_obs_subset_state():
    _seed_everything(123)
    config = get_prior_config()
    prior = EnvironmentPrior(config["prior"]["environment"])

    prior.get_batch(
        batch_size=8,
        n_samples=40,
        num_features=16,
        device="cpu",
        single_eval_pos=15,
    )

    assert len(prior.last_runtime_info) == 8
    for row in prior.last_runtime_info:
        assert 1 <= int(row["action_dim"]) <= 30
        assert 1 <= int(row["obs_dim"]) <= 400
        assert 1 <= int(row["state_dim"]) <= 400
        assert int(row["obs_dim"]) <= int(row["state_dim"])
        assert int(row["noise_dim"]) >= 1
        assert int(row["zero_pad_dim"]) >= 0


def test_environment_prior_coverage_stats_exist():
    _seed_everything(7)
    config = get_prior_config()
    prior = EnvironmentPrior(config["prior"]["environment"])

    prior.get_batch(
        batch_size=3,
        n_samples=20,
        num_features=12,
        device="cpu",
        single_eval_pos=10,
    )
    coverage = prior.get_last_coverage()
    assert "reward_min" in coverage
    assert "reward_max" in coverage
    assert "reward_std_mean" in coverage
    assert "state_abs_max" in coverage
    assert coverage["reward_min"] <= coverage["reward_max"]
    assert coverage["state_abs_max"] >= 0.0


def test_environment_prior_rollout_with_policy_is_differentiable():
    _seed_everything(11)
    config = get_prior_config()
    env_cfg = dict(config["prior"]["environment"])
    # Fix dims for deterministic policy head shape in this unit test.
    env_cfg["family"] = {"distribution": "meta_choice", "choice_values": ["scm"]}
    env_cfg["state_dim"] = {"distribution": "uniform_int", "min": 6, "max": 6}
    env_cfg["obs_dim"] = {"distribution": "uniform_int", "min": 4, "max": 4}
    env_cfg["action_dim"] = {"distribution": "uniform_int", "min": 3, "max": 3}
    env_cfg["noise_dim"] = {"distribution": "uniform_int", "min": 5, "max": 5}
    env_cfg["zero_pad_dim"] = {"distribution": "uniform_int", "min": 2, "max": 2}

    prior = EnvironmentPrior(env_cfg)

    class TinyPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Linear(4 + 3 + 1, 3)

        def step(self, obs_t, action_t, reward_t, cache, step_idx, env_info):
            del cache, step_idx, env_info
            return self.net(torch.cat([obs_t, action_t, reward_t], dim=-1))

    policy = TinyPolicy()
    rollout = prior.rollout_with_policy(
        policy_step_fn=policy.step,
        batch_size=3,
        n_samples=25,
        num_features=16,
        device="cpu",
        single_eval_pos=12,
    )
    loss, _ = prior.policy_gradient_loss_from_rewards(rollout["rewards"])
    loss.backward()
    grad_sum = sum(p.grad.abs().sum() for p in policy.parameters() if p.grad is not None)
    assert torch.isfinite(rollout["x"]).all()
    assert torch.isfinite(rollout["rewards"]).all()
    assert float(grad_sum) > 0.0


def test_environment_prior_tbptt_policy_gradient_stats_include_reward_range():
    _seed_everything(20260304)
    config = get_prior_config()
    env_cfg = dict(config["prior"]["environment"])
    env_cfg["family"] = {"distribution": "meta_choice", "choice_values": ["scm"]}
    env_cfg["state_dim"] = {"distribution": "uniform_int", "min": 6, "max": 6}
    env_cfg["obs_dim"] = {"distribution": "uniform_int", "min": 4, "max": 4}
    env_cfg["action_dim"] = {"distribution": "uniform_int", "min": 3, "max": 3}
    env_cfg["noise_dim"] = {"distribution": "uniform_int", "min": 5, "max": 5}
    env_cfg["zero_pad_dim"] = {"distribution": "uniform_int", "min": 2, "max": 2}
    prior = EnvironmentPrior(env_cfg)

    class TinyPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Linear(4 + 3 + 1, 3)

        def step(self, obs_t, action_t, reward_t, cache, step_idx, env_info):
            del cache, step_idx, env_info
            return self.net(torch.cat([obs_t, action_t, reward_t], dim=-1))

    policy = TinyPolicy()
    loss, rollout, stats = prior.rollout_policy_gradient_loss(
        policy_step_fn=policy.step,
        batch_size=3,
        n_samples=12,
        num_features=24,
        device="cpu",
        single_eval_pos=6,
        tbptt_window=4,
        normalize=False,
        collect_x=False,
    )

    assert torch.isfinite(loss)
    assert torch.isfinite(rollout["rewards"]).all()
    assert "reward_min" in stats
    assert "reward_max" in stats
    assert "reward_abs_max" in stats
    assert "reward_clip_hit_share" in stats
    assert "reward_norm_clip_hit_share" in stats
    assert float(stats["reward_abs_max"]) >= 0.0


def test_environment_prior_aev2_switch_exposes_method_only_stats():
    _seed_everything(20260305)
    config = get_prior_config()
    env_cfg = dict(config["prior"]["environment"])
    env_cfg["family"] = {"distribution": "meta_choice", "choice_values": ["scm"]}
    env_cfg["state_dim"] = {"distribution": "uniform_int", "min": 6, "max": 6}
    env_cfg["obs_dim"] = {"distribution": "uniform_int", "min": 4, "max": 4}
    env_cfg["action_dim"] = {"distribution": "uniform_int", "min": 3, "max": 3}
    env_cfg["noise_dim"] = {"distribution": "uniform_int", "min": 5, "max": 5}
    env_cfg["zero_pad_dim"] = {"distribution": "uniform_int", "min": 2, "max": 2}
    env_cfg["anti_explosion_vanishing_v2_enabled"] = True
    env_cfg["anti_explosion_vanishing_v2_lambda"] = 0.1
    env_cfg["anti_explosion_vanishing_v2_gain_lo"] = 0.85
    env_cfg["anti_explosion_vanishing_v2_gain_hi"] = 1.15
    prior = EnvironmentPrior(env_cfg)

    class TinyPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Linear(4 + 3 + 1, 3)

        def step(self, obs_t, action_t, reward_t, cache, step_idx, env_info):
            del cache, step_idx, env_info
            return self.net(torch.cat([obs_t, action_t, reward_t], dim=-1))

    policy = TinyPolicy()
    loss, rollout, stats = prior.rollout_policy_gradient_loss(
        policy_step_fn=policy.step,
        batch_size=3,
        n_samples=16,
        num_features=24,
        device="cpu",
        single_eval_pos=8,
        tbptt_window=4,
        normalize=False,
        collect_x=False,
    )
    loss.backward()
    grad_sum = sum(p.grad.abs().sum() for p in policy.parameters() if p.grad is not None)

    assert torch.isfinite(loss)
    assert torch.isfinite(rollout["rewards"]).all()
    assert float(grad_sum) > 0.0
    assert int(stats.get("aev2_enabled", 0)) == 1
    assert "aev2_penalty" in stats
    assert "aev2_loss_add" in stats
    assert "aev2_gain_mean" in stats
    assert "aev2_gain_std" in stats
    assert "aev2_gain_min" in stats
    assert "aev2_gain_max" in stats
    assert torch.isfinite(stats["aev2_penalty"])
    assert torch.isfinite(stats["aev2_loss_add"])
    assert torch.isfinite(stats["aev2_gain_mean"])
    assert torch.isfinite(stats["aev2_gain_std"])


def test_environment_prior_aev3_switch_exposes_method_only_stats():
    _seed_everything(20260305 + 1)
    config = get_prior_config()
    env_cfg = dict(config["prior"]["environment"])
    env_cfg["family"] = {"distribution": "meta_choice", "choice_values": ["scm"]}
    env_cfg["state_dim"] = {"distribution": "uniform_int", "min": 6, "max": 6}
    env_cfg["obs_dim"] = {"distribution": "uniform_int", "min": 4, "max": 4}
    env_cfg["action_dim"] = {"distribution": "uniform_int", "min": 3, "max": 3}
    env_cfg["noise_dim"] = {"distribution": "uniform_int", "min": 5, "max": 5}
    env_cfg["zero_pad_dim"] = {"distribution": "uniform_int", "min": 2, "max": 2}
    env_cfg["anti_explosion_vanishing_v2_enabled"] = False
    env_cfg["anti_explosion_vanishing_v3_enabled"] = True
    env_cfg["anti_explosion_vanishing_v3_lambda_drift"] = 0.02
    env_cfg["anti_explosion_vanishing_v3_lambda_tail"] = 0.05
    env_cfg["anti_explosion_vanishing_v3_gain_lo"] = 0.85
    env_cfg["anti_explosion_vanishing_v3_gain_hi"] = 1.15
    env_cfg["anti_explosion_vanishing_v3_tail_tau"] = 0.02
    env_cfg["anti_explosion_vanishing_v3_eps"] = 1e-6
    prior = EnvironmentPrior(env_cfg)

    class TinyPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Linear(4 + 3 + 1, 3)

        def step(self, obs_t, action_t, reward_t, cache, step_idx, env_info):
            del cache, step_idx, env_info
            return self.net(torch.cat([obs_t, action_t, reward_t], dim=-1))

    policy = TinyPolicy()
    loss, rollout, stats = prior.rollout_policy_gradient_loss(
        policy_step_fn=policy.step,
        batch_size=3,
        n_samples=16,
        num_features=24,
        device="cpu",
        single_eval_pos=8,
        tbptt_window=4,
        normalize=False,
        collect_x=False,
    )
    loss.backward()
    grad_sum = sum(p.grad.abs().sum() for p in policy.parameters() if p.grad is not None)

    assert torch.isfinite(loss)
    assert torch.isfinite(rollout["rewards"]).all()
    assert float(grad_sum) > 0.0
    assert int(stats.get("aev3_enabled", 0)) == 1
    assert "aev3_penalty" in stats
    assert "aev3_penalty_drift" in stats
    assert "aev3_penalty_tail" in stats
    assert "aev3_loss_add" in stats
    assert "aev3_log_gain_mean" in stats
    assert "aev3_log_gain_std" in stats
    assert "aev3_tail_low_share" in stats
    assert "aev3_tail_high_share" in stats
    assert "aev3_gain_mean" in stats
    assert "aev3_gain_std" in stats
    assert "aev3_gain_min" in stats
    assert "aev3_gain_max" in stats
    assert torch.isfinite(stats["aev3_penalty"])
    assert torch.isfinite(stats["aev3_penalty_drift"])
    assert torch.isfinite(stats["aev3_penalty_tail"])
    assert torch.isfinite(stats["aev3_log_gain_mean"])
    assert torch.isfinite(stats["aev3_gain_mean"])
    assert "aev2_penalty" not in stats


def test_environment_prior_aev4_switch_exposes_method_only_stats():
    _seed_everything(20260305 + 2)
    config = get_prior_config()
    env_cfg = dict(config["prior"]["environment"])
    env_cfg["family"] = {"distribution": "meta_choice", "choice_values": ["scm"]}
    env_cfg["state_dim"] = {"distribution": "uniform_int", "min": 6, "max": 6}
    env_cfg["obs_dim"] = {"distribution": "uniform_int", "min": 4, "max": 4}
    env_cfg["action_dim"] = {"distribution": "uniform_int", "min": 3, "max": 3}
    env_cfg["noise_dim"] = {"distribution": "uniform_int", "min": 5, "max": 5}
    env_cfg["zero_pad_dim"] = {"distribution": "uniform_int", "min": 2, "max": 2}
    env_cfg["anti_explosion_vanishing_v2_enabled"] = False
    env_cfg["anti_explosion_vanishing_v3_enabled"] = False
    env_cfg["anti_explosion_vanishing_v4_enabled"] = True
    env_cfg["anti_explosion_vanishing_v4_lambda_drift"] = 0.08
    env_cfg["anti_explosion_vanishing_v4_lambda_tail"] = 0.25
    env_cfg["anti_explosion_vanishing_v4_gain_lo"] = 0.94
    env_cfg["anti_explosion_vanishing_v4_gain_hi"] = 1.07
    env_cfg["anti_explosion_vanishing_v4_tail_tau"] = 0.016
    env_cfg["anti_explosion_vanishing_v4_eps"] = 1e-6
    env_cfg["anti_explosion_vanishing_v4_highway_ratio"] = 0.25
    env_cfg["anti_explosion_vanishing_v4_update_scale"] = 0.12
    env_cfg["anti_explosion_vanishing_v4_update_clip"] = 0.0
    prior = EnvironmentPrior(env_cfg)

    class TinyPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Linear(4 + 3 + 1, 3)

        def step(self, obs_t, action_t, reward_t, cache, step_idx, env_info):
            del cache, step_idx, env_info
            return self.net(torch.cat([obs_t, action_t, reward_t], dim=-1))

    policy = TinyPolicy()
    loss, rollout, stats = prior.rollout_policy_gradient_loss(
        policy_step_fn=policy.step,
        batch_size=3,
        n_samples=16,
        num_features=24,
        device="cpu",
        single_eval_pos=8,
        tbptt_window=4,
        normalize=False,
        collect_x=False,
    )
    loss.backward()
    grad_sum = sum(p.grad.abs().sum() for p in policy.parameters() if p.grad is not None)

    assert torch.isfinite(loss)
    assert torch.isfinite(rollout["rewards"]).all()
    assert float(grad_sum) > 0.0
    assert int(stats.get("aev4_enabled", 0)) == 1
    assert "aev4_penalty" in stats
    assert "aev4_penalty_drift" in stats
    assert "aev4_penalty_tail" in stats
    assert "aev4_loss_add" in stats
    assert "aev4_log_gain_mean" in stats
    assert "aev4_log_gain_std" in stats
    assert "aev4_tail_low_share" in stats
    assert "aev4_tail_high_share" in stats
    assert "aev4_gain_mean" in stats
    assert "aev4_gain_std" in stats
    assert "aev4_gain_min" in stats
    assert "aev4_gain_max" in stats
    assert "aev4_update_rms_mean" in stats
    assert "aev4_update_rms_std" in stats
    assert "aev4_clip_hit_share" in stats
    assert torch.isfinite(stats["aev4_penalty"])
    assert torch.isfinite(stats["aev4_penalty_drift"])
    assert torch.isfinite(stats["aev4_penalty_tail"])
    assert torch.isfinite(stats["aev4_log_gain_mean"])
    assert torch.isfinite(stats["aev4_gain_mean"])
    assert "aev2_penalty" not in stats
    assert "aev3_penalty" not in stats


def test_environment_prior_aev5_switch_exposes_method_only_stats():
    _seed_everything(20260305 + 3)
    config = get_prior_config()
    env_cfg = dict(config["prior"]["environment"])
    env_cfg["family"] = {"distribution": "meta_choice", "choice_values": ["scm"]}
    env_cfg["state_dim"] = {"distribution": "uniform_int", "min": 6, "max": 6}
    env_cfg["obs_dim"] = {"distribution": "uniform_int", "min": 4, "max": 4}
    env_cfg["action_dim"] = {"distribution": "uniform_int", "min": 3, "max": 3}
    env_cfg["noise_dim"] = {"distribution": "uniform_int", "min": 5, "max": 5}
    env_cfg["zero_pad_dim"] = {"distribution": "uniform_int", "min": 2, "max": 2}
    env_cfg["anti_explosion_vanishing_v2_enabled"] = False
    env_cfg["anti_explosion_vanishing_v3_enabled"] = False
    env_cfg["anti_explosion_vanishing_v4_enabled"] = False
    env_cfg["anti_explosion_vanishing_v5_enabled"] = True
    env_cfg["anti_explosion_vanishing_v5_target_std"] = 0.25
    env_cfg["anti_explosion_vanishing_v5_scale_lo"] = 0.5
    env_cfg["anti_explosion_vanishing_v5_scale_hi"] = 4.0
    env_cfg["anti_explosion_vanishing_v5_eps"] = 1e-6
    prior = EnvironmentPrior(env_cfg)

    class TinyPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Linear(4 + 3 + 1, 3)

        def step(self, obs_t, action_t, reward_t, cache, step_idx, env_info):
            del cache, step_idx, env_info
            return self.net(torch.cat([obs_t, action_t, reward_t], dim=-1))

    policy = TinyPolicy()
    loss, rollout, stats = prior.rollout_policy_gradient_loss(
        policy_step_fn=policy.step,
        batch_size=3,
        n_samples=16,
        num_features=24,
        device="cpu",
        single_eval_pos=8,
        tbptt_window=4,
        normalize=False,
        collect_x=False,
    )
    loss.backward()
    grad_sum = sum(p.grad.abs().sum() for p in policy.parameters() if p.grad is not None)

    assert torch.isfinite(loss)
    assert torch.isfinite(rollout["rewards"]).all()
    assert float(grad_sum) > 0.0
    assert int(stats.get("aev5_enabled", 0)) == 1
    assert "objective_with_aev5" in stats
    assert "aev5_loss_mul" in stats
    assert "aev5_scale" in stats
    assert "aev5_scale_raw" in stats
    assert "aev5_reward_std_ref" in stats
    assert torch.isfinite(stats["objective_with_aev5"])
    assert torch.isfinite(stats["aev5_loss_mul"])
    assert torch.isfinite(stats["aev5_scale"])
    assert torch.isfinite(stats["aev5_scale_raw"])
    assert torch.isfinite(stats["aev5_reward_std_ref"])
    assert "aev2_penalty" not in stats
    assert "aev3_penalty" not in stats
    assert "aev4_penalty" not in stats


def test_environment_prior_rollout_with_policy_clips_rewards():
    _seed_everything(101)
    config = get_prior_config()
    env_cfg = dict(config["prior"]["environment"])
    env_cfg["family"] = {"distribution": "meta_choice", "choice_values": ["scm"]}
    env_cfg["state_dim"] = {"distribution": "uniform_int", "min": 6, "max": 6}
    env_cfg["obs_dim"] = {"distribution": "uniform_int", "min": 4, "max": 4}
    env_cfg["action_dim"] = {"distribution": "uniform_int", "min": 3, "max": 3}
    env_cfg["noise_dim"] = {"distribution": "uniform_int", "min": 5, "max": 5}
    env_cfg["zero_pad_dim"] = {"distribution": "uniform_int", "min": 0, "max": 0}
    env_cfg["reward_scale"] = 100.0
    env_cfg["reward_clip"] = 0.1
    env_cfg["reward_dropout_enabled"] = False
    env_cfg["reward_dropout_randomize"] = False
    prior = EnvironmentPrior(env_cfg)

    class ZeroPolicy(nn.Module):
        def step(self, obs_t, action_t, reward_t, reward_mask_t, cache, step_idx, env_info):
            del obs_t, reward_t, reward_mask_t, cache, step_idx, env_info
            return torch.zeros_like(action_t)

    rollout = prior.rollout_with_policy(
        policy_step_fn=ZeroPolicy().step,
        batch_size=4,
        n_samples=12,
        num_features=16,
        device="cpu",
        single_eval_pos=6,
        collect_x=False,
    )
    rewards = rollout["rewards"]
    assert torch.isfinite(rewards).all()
    assert float(rewards.abs().max()) <= 0.1000001


def test_policy_gradient_reward_stats_include_clip_observability():
    _seed_everything(202)
    config = get_prior_config()
    env_cfg = dict(config["prior"]["environment"])
    env_cfg["reward_clip"] = 10.0
    prior = EnvironmentPrior(env_cfg)

    rewards = torch.tensor(
        [
            [10.0, 0.0, -10.0],
            [5.0, -10.0, 10.0],
        ],
        dtype=torch.float32,
    )
    loss, stats = prior.policy_gradient_loss_from_rewards(
        rewards,
        normalize=True,
        clip=0.5,
        detach_stats=True,
    )

    assert torch.isfinite(loss)
    assert float(stats["reward_min"]) == -10.0
    assert float(stats["reward_max"]) == 10.0
    assert float(stats["reward_abs_max"]) == 10.0
    assert float(stats["reward_clip_hit_share"]) > 0.0
    assert float(stats["reward_norm_clip_hit_share"]) > 0.0


def test_environment_prior_reward_dropout_mask_is_in_token():
    _seed_everything(3)
    config = get_prior_config()
    env_cfg = dict(config["prior"]["environment"])
    env_cfg["family"] = {"distribution": "meta_choice", "choice_values": ["scm"]}
    env_cfg["state_dim"] = {"distribution": "uniform_int", "min": 8, "max": 8}
    env_cfg["obs_dim"] = {"distribution": "uniform_int", "min": 4, "max": 4}
    env_cfg["action_dim"] = {"distribution": "uniform_int", "min": 2, "max": 2}
    env_cfg["noise_dim"] = {"distribution": "uniform_int", "min": 3, "max": 3}
    env_cfg["zero_pad_dim"] = {"distribution": "uniform_int", "min": 0, "max": 0}
    env_cfg["obs_slot_dim"] = 10
    env_cfg["action_slot_dim"] = 4
    env_cfg["reward_dropout_enabled"] = True
    env_cfg["reward_dropout_randomize"] = False
    env_cfg["reward_dropout_ratio"] = 1.0
    env_cfg["reward_dropout_impute_zero"] = True

    prior = EnvironmentPrior(env_cfg)
    x, y, _ = prior.get_batch(
        batch_size=2,
        n_samples=16,
        num_features=16,  # 10 + 1 + 1 + 4
        device="cpu",
        single_eval_pos=8,
    )

    reward_mask_col = 10 + 1
    assert x.shape == (16, 2, 16)
    # t=0 placeholder mask is observed, afterwards rewards are fully dropped.
    assert torch.all(x[0, :, reward_mask_col] == 1)
    assert torch.all(x[1:, :, reward_mask_col] == 0)
    # With ratio=1 and zero imputation, training rewards are zeroed.
    assert torch.all(y == 0)


def test_environment_prior_rollout_with_policy_propagates_reward_dropout_mask():
    _seed_everything(19)
    config = get_prior_config()
    env_cfg = dict(config["prior"]["environment"])
    env_cfg["family"] = {"distribution": "meta_choice", "choice_values": ["scm"]}
    env_cfg["state_dim"] = {"distribution": "uniform_int", "min": 7, "max": 7}
    env_cfg["obs_dim"] = {"distribution": "uniform_int", "min": 4, "max": 4}
    env_cfg["action_dim"] = {"distribution": "uniform_int", "min": 2, "max": 2}
    env_cfg["noise_dim"] = {"distribution": "uniform_int", "min": 3, "max": 3}
    env_cfg["zero_pad_dim"] = {"distribution": "uniform_int", "min": 0, "max": 0}
    env_cfg["reward_dropout_enabled"] = True
    env_cfg["reward_dropout_randomize"] = False
    env_cfg["reward_dropout_ratio"] = 1.0
    env_cfg["reward_dropout_impute_zero"] = True

    prior = EnvironmentPrior(env_cfg)

    class MaskRecorderPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Linear(4 + 2 + 1 + 1, 2)
            self.reward_masks = []

        def step(self, obs_t, action_t, reward_t, reward_mask_t, cache, step_idx, env_info):
            del cache, step_idx, env_info
            self.reward_masks.append(reward_mask_t.detach().clone())
            return self.net(torch.cat([obs_t, action_t, reward_t, reward_mask_t], dim=-1))

    policy = MaskRecorderPolicy()
    rollout = prior.rollout_with_policy(
        policy_step_fn=policy.step,
        batch_size=1,
        n_samples=12,
        num_features=16,
        device="cpu",
        single_eval_pos=6,
        collect_x=False,
    )

    masks = torch.cat(policy.reward_masks, dim=0).reshape(-1)
    expected = torch.zeros_like(masks)
    expected[0] = 1.0  # r0 placeholder mask
    assert torch.allclose(masks, expected)
    assert torch.all(rollout["rewards"] == 0)


def test_environment_prior_reproducible_with_fixed_seed():
    seed = 2026
    _seed_everything(seed)
    config = get_prior_config()
    env_cfg = dict(config["prior"]["environment"])
    env_cfg["batch_vectorized_strict_rng_match"] = True
    prior_a = EnvironmentPrior(env_cfg)
    xa, ya, _ = prior_a.get_batch(
        batch_size=4,
        n_samples=20,
        num_features=24,
        device="cpu",
        single_eval_pos=10,
    )

    _seed_everything(seed)
    prior_b = EnvironmentPrior(env_cfg)
    xb, yb, _ = prior_b.get_batch(
        batch_size=4,
        n_samples=20,
        num_features=24,
        device="cpu",
        single_eval_pos=10,
    )

    assert torch.allclose(xa, xb)
    assert torch.allclose(ya, yb)


def test_rollout_collect_x_toggle_keeps_reward_and_grad_identical():
    _seed_everything(77)
    config = get_prior_config()
    env_cfg = dict(config["prior"]["environment"])
    env_cfg["family"] = {"distribution": "meta_choice", "choice_values": ["scm"]}
    env_cfg["state_dim"] = {"distribution": "uniform_int", "min": 6, "max": 6}
    env_cfg["obs_dim"] = {"distribution": "uniform_int", "min": 4, "max": 4}
    env_cfg["action_dim"] = {"distribution": "uniform_int", "min": 3, "max": 3}
    env_cfg["noise_dim"] = {"distribution": "uniform_int", "min": 5, "max": 5}
    env_cfg["zero_pad_dim"] = {"distribution": "uniform_int", "min": 2, "max": 2}
    prior = EnvironmentPrior(env_cfg)

    class TinyPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Linear(4 + 3 + 1, 3)

        def step(self, obs_t, action_t, reward_t, cache, step_idx, env_info):
            del cache, step_idx, env_info
            return self.net(torch.cat([obs_t, action_t, reward_t], dim=-1))

    policy_a = TinyPolicy()
    policy_b = TinyPolicy()
    policy_b.load_state_dict(policy_a.state_dict())

    _seed_everything(1234)
    rollout_a = prior.rollout_with_policy(
        policy_step_fn=policy_a.step,
        batch_size=3,
        n_samples=24,
        num_features=16,
        device="cpu",
        single_eval_pos=12,
        collect_x=True,
    )
    loss_a, _ = prior.policy_gradient_loss_from_rewards(rollout_a["rewards"])
    loss_a.backward()
    grads_a = [p.grad.detach().clone() for p in policy_a.parameters()]

    _seed_everything(1234)
    rollout_b = prior.rollout_with_policy(
        policy_step_fn=policy_b.step,
        batch_size=3,
        n_samples=24,
        num_features=16,
        device="cpu",
        single_eval_pos=12,
        collect_x=False,
    )
    loss_b, _ = prior.policy_gradient_loss_from_rewards(rollout_b["rewards"])
    loss_b.backward()
    grads_b = [p.grad.detach().clone() for p in policy_b.parameters()]

    assert torch.allclose(rollout_a["rewards"], rollout_b["rewards"])
    assert torch.allclose(loss_a.detach(), loss_b.detach())
    for ga, gb in zip(grads_a, grads_b):
        assert torch.allclose(ga, gb, atol=1e-6, rtol=1e-5)


def test_environment_prior_parallel_get_batch_matches_serial():
    config = get_prior_config()
    env_cfg_serial = dict(config["prior"]["environment"])
    env_cfg_serial["family"] = {"distribution": "meta_choice", "choice_values": ["scm"]}
    env_cfg_serial["state_dim"] = {"distribution": "uniform_int", "min": 12, "max": 12}
    env_cfg_serial["obs_dim"] = {"distribution": "uniform_int", "min": 10, "max": 10}
    env_cfg_serial["action_dim"] = {"distribution": "uniform_int", "min": 4, "max": 4}
    env_cfg_serial["noise_dim"] = {"distribution": "uniform_int", "min": 7, "max": 7}
    env_cfg_serial["zero_pad_dim"] = {"distribution": "uniform_int", "min": 3, "max": 3}
    env_cfg_serial["batch_parallel_backend"] = "python_thread"
    env_cfg_serial["batch_parallel_workers"] = 1
    env_cfg_serial["batch_vectorized_strict_rng_match"] = True

    env_cfg_parallel = dict(env_cfg_serial)
    env_cfg_parallel["batch_parallel_workers"] = 4

    _seed_everything(20260227)
    prior_serial = EnvironmentPrior(env_cfg_serial)
    x_serial, y_serial, _ = prior_serial.get_batch(
        batch_size=8,
        n_samples=32,
        num_features=64,
        device="cpu",
        single_eval_pos=16,
    )

    _seed_everything(20260227)
    prior_parallel = EnvironmentPrior(env_cfg_parallel)
    x_parallel, y_parallel, _ = prior_parallel.get_batch(
        batch_size=8,
        n_samples=32,
        num_features=64,
        device="cpu",
        single_eval_pos=16,
    )

    assert torch.allclose(x_serial, x_parallel)
    assert torch.allclose(y_serial, y_parallel)


def _make_shared_env_cfg_for_backend_compare():
    config = get_prior_config()
    env_cfg = dict(config["prior"]["environment"])
    env_cfg["family"] = {"distribution": "meta_choice", "choice_values": ["scm"]}
    env_cfg["state_dim"] = {"distribution": "uniform_int", "min": 8, "max": 8}
    env_cfg["obs_dim"] = {"distribution": "uniform_int", "min": 6, "max": 6}
    env_cfg["action_dim"] = {"distribution": "uniform_int", "min": 3, "max": 3}
    env_cfg["noise_dim"] = {"distribution": "uniform_int", "min": 4, "max": 4}
    env_cfg["zero_pad_dim"] = {"distribution": "uniform_int", "min": 2, "max": 2}
    env_cfg["batch_shared_environment"] = True
    # Make env dynamics deterministic so backend A/B can be compared exactly.
    env_cfg["init_state_std"] = 0.0
    env_cfg["init_action_std"] = 0.0
    env_cfg["state_noise_std"] = 0.0
    env_cfg["action_noise_train_std"] = 0.0
    env_cfg["action_noise_eval_std"] = 0.0
    env_cfg["reward_dropout_enabled"] = False
    env_cfg["init_std"] = 0.0
    env_cfg["noise_std"] = 0.0
    return env_cfg


def test_environment_prior_torch_vectorized_matches_python_thread_in_deterministic_shared_env():
    env_cfg_thread = _make_shared_env_cfg_for_backend_compare()
    env_cfg_thread["batch_parallel_backend"] = "python_thread"
    env_cfg_thread["batch_parallel_workers"] = 4

    env_cfg_vec = dict(env_cfg_thread)
    env_cfg_vec["batch_parallel_backend"] = "torch_vectorized"

    _seed_everything(20260227)
    prior_thread = EnvironmentPrior(env_cfg_thread)
    x_thread, y_thread, _ = prior_thread.get_batch(
        batch_size=6,
        n_samples=24,
        num_features=40,
        device="cpu",
        single_eval_pos=12,
    )

    _seed_everything(20260227)
    prior_vec = EnvironmentPrior(env_cfg_vec)
    x_vec, y_vec, _ = prior_vec.get_batch(
        batch_size=6,
        n_samples=24,
        num_features=40,
        device="cpu",
        single_eval_pos=12,
    )

    assert torch.allclose(x_thread, x_vec)
    assert torch.allclose(y_thread, y_vec)


def test_environment_prior_torch_vectorized_without_shared_env_avoids_thread_pool():
    config = get_prior_config()
    env_cfg = dict(config["prior"]["environment"])
    env_cfg["family"] = {"distribution": "meta_choice", "choice_values": ["scm"]}
    env_cfg["state_dim"] = {"distribution": "uniform_int", "min": 7, "max": 7}
    env_cfg["obs_dim"] = {"distribution": "uniform_int", "min": 5, "max": 5}
    env_cfg["action_dim"] = {"distribution": "uniform_int", "min": 3, "max": 3}
    env_cfg["noise_dim"] = {"distribution": "uniform_int", "min": 4, "max": 4}
    env_cfg["zero_pad_dim"] = {"distribution": "uniform_int", "min": 1, "max": 1}
    env_cfg["batch_parallel_backend"] = "torch_vectorized"
    env_cfg["batch_shared_environment"] = False
    env_cfg["batch_parallel_workers"] = 8

    prior = EnvironmentPrior(env_cfg)

    def _fail_thread_pool(*args, **kwargs):
        raise AssertionError("torch_vectorized backend should not use ThreadPoolExecutor")

    prior._get_rollout_executor = _fail_thread_pool
    x, y, _ = prior.get_batch(
        batch_size=5,
        n_samples=16,
        num_features=24,
        device="cpu",
        single_eval_pos=8,
    )

    assert x.shape == (16, 5, 24)
    assert y.shape == (16, 5)
    assert torch.isfinite(x).all()
    assert torch.isfinite(y).all()


def test_environment_prior_torch_vectorized_without_shared_env_uses_distinct_env_initializations():
    _seed_everything(20260227)
    config = get_prior_config()
    env_cfg = dict(config["prior"]["environment"])
    env_cfg["family"] = {"distribution": "meta_choice", "choice_values": ["scm"]}
    env_cfg["state_dim"] = {"distribution": "uniform_int", "min": 8, "max": 8}
    env_cfg["obs_dim"] = {"distribution": "uniform_int", "min": 6, "max": 6}
    env_cfg["action_dim"] = {"distribution": "uniform_int", "min": 3, "max": 3}
    env_cfg["noise_dim"] = {"distribution": "uniform_int", "min": 4, "max": 4}
    env_cfg["zero_pad_dim"] = {"distribution": "uniform_int", "min": 2, "max": 2}
    env_cfg["num_layers"] = {"distribution": "uniform_int", "min": 3, "max": 3}
    env_cfg["prior_mlp_hidden_dim"] = {"distribution": "uniform_int", "min": 16, "max": 16}
    env_cfg["prior_mlp_activations"] = "tanh"
    env_cfg["init_state_std"] = 0.0
    env_cfg["init_action_std"] = 0.0
    env_cfg["state_noise_std"] = 0.0
    env_cfg["action_noise_train_std"] = 0.0
    env_cfg["action_noise_eval_std"] = 0.0
    env_cfg["reward_dropout_enabled"] = False
    env_cfg["noise_std"] = 0.0
    env_cfg["batch_parallel_backend"] = "torch_vectorized"
    env_cfg["batch_shared_environment"] = False

    prior = EnvironmentPrior(env_cfg)
    _, y, _ = prior.get_batch(
        batch_size=6,
        n_samples=12,
        num_features=40,
        device="cpu",
        single_eval_pos=6,
    )

    # Distinct env initialization should produce non-identical reward trajectories
    # across at least one pair of batch columns in this deterministic setup.
    equal_cols = [torch.allclose(y[:, 0], y[:, b]) for b in range(1, y.shape[1])]
    assert not all(equal_cols)


def _manual_sampled_h(*, family, state_dim, obs_dim, action_dim, noise_dim, zero_pad_dim, num_layers=3, gp_rff_features=64):
    h = {
        "family": family,
        "state_dim": state_dim,
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "noise_dim": noise_dim,
        "zero_pad_dim": zero_pad_dim,
        "obs_slot_dim": 12,
        "action_slot_dim": 4,
        "alpha": 0.2,
        "init_state_std": 0.0,
        "init_action_std": 0.0,
        "state_noise_std": 0.0,
        "action_noise_train_std": 0.0,
        "action_noise_eval_std": 0.0,
        "reward_scale": 1.0,
        "state_clip": 8.0,
        "reward_dropout_enabled": False,
        "reward_dropout_randomize": False,
        "reward_dropout_ratio": 0.0,
        "reward_dropout_ratio_min": 0.0,
        "reward_dropout_ratio_max": 0.0,
        "reward_dropout_impute_zero": True,
        "num_layers": num_layers,
        "prior_mlp_hidden_dim": 16,
        "prior_mlp_activations": "tanh",
        "init_std": 0.05,
        "noise_std": 0.0,
        "lengthscale": 1.0,
        "outputscale": 1.0,
        "noise": 0.0,
        "gp_rff_features": gp_rff_features,
    }
    return h


def test_environment_prior_torch_vectorized_preserves_heterogeneous_structure_semantics():
    _seed_everything(20260227)
    config = get_prior_config()
    env_cfg = dict(config["prior"]["environment"])
    env_cfg["batch_parallel_backend"] = "torch_vectorized"
    env_cfg["batch_shared_environment"] = False
    prior = EnvironmentPrior(env_cfg)

    sampled = [
        _manual_sampled_h(
            family="scm",
            state_dim=8,
            obs_dim=6,
            action_dim=3,
            noise_dim=4,
            zero_pad_dim=2,
            num_layers=3,
        ),
        _manual_sampled_h(
            family="scm",
            state_dim=11,
            obs_dim=7,
            action_dim=3,
            noise_dim=4,
            zero_pad_dim=2,
            num_layers=2,
        ),
        _manual_sampled_h(
            family="gp",
            state_dim=7,
            obs_dim=5,
            action_dim=2,
            noise_dim=3,
            zero_pad_dim=1,
            gp_rff_features=32,
        ),
        _manual_sampled_h(
            family="scm",
            state_dim=8,
            obs_dim=6,
            action_dim=3,
            noise_dim=4,
            zero_pad_dim=2,
            num_layers=3,
        ),
    ]

    prior._sample_batch_hypers = lambda batch_size: sampled[:batch_size]
    x, y, _ = prior.get_batch(
        batch_size=4,
        n_samples=10,
        num_features=24,
        device="cpu",
        single_eval_pos=5,
    )

    assert x.shape == (10, 4, 24)
    assert y.shape == (10, 4)
    assert len(prior.last_runtime_info) == 4
    for idx, info in enumerate(prior.last_runtime_info):
        assert info["family"] == sampled[idx]["family"]
        assert int(info["state_dim"]) == sampled[idx]["state_dim"]
        assert int(info["obs_dim"]) == min(sampled[idx]["obs_dim"], sampled[idx]["state_dim"])
        assert int(info["action_dim"]) == sampled[idx]["action_dim"]
        assert int(info["noise_dim"]) == sampled[idx]["noise_dim"]
        assert int(info["zero_pad_dim"]) == sampled[idx]["zero_pad_dim"]

    signatures = {
        (row["family"], int(row["state_dim"]), int(row["obs_dim"]), int(row["action_dim"]), int(row["noise_dim"]), int(row["zero_pad_dim"]))
        for row in prior.last_runtime_info
    }
    assert len(signatures) >= 3


def test_environment_prior_torch_vectorized_matches_serial_semantics_with_identical_h_and_seeds():
    sampled = [
        _manual_sampled_h(
            family="scm",
            state_dim=8,
            obs_dim=6,
            action_dim=3,
            noise_dim=4,
            zero_pad_dim=2,
            num_layers=3,
        ),
        _manual_sampled_h(
            family="scm",
            state_dim=11,
            obs_dim=7,
            action_dim=3,
            noise_dim=4,
            zero_pad_dim=2,
            num_layers=2,
        ),
        _manual_sampled_h(
            family="gp",
            state_dim=7,
            obs_dim=5,
            action_dim=2,
            noise_dim=3,
            zero_pad_dim=1,
            gp_rff_features=32,
        ),
        _manual_sampled_h(
            family="scm",
            state_dim=8,
            obs_dim=6,
            action_dim=3,
            noise_dim=4,
            zero_pad_dim=2,
            num_layers=3,
        ),
    ]
    env_seeds = [11, 23, 37, 41]
    rollout_seeds = [101, 211, 307, 401]

    config = get_prior_config()
    env_cfg = dict(config["prior"]["environment"])
    env_cfg["batch_shared_environment"] = False
    env_cfg["batch_parallel_workers"] = 1
    env_cfg["batch_vectorized_strict_rng_match"] = True

    env_cfg_serial = dict(env_cfg)
    env_cfg_serial["batch_parallel_backend"] = "python_thread"
    prior_serial = EnvironmentPrior(env_cfg_serial)
    prior_serial._sample_batch_hypers = lambda batch_size: sampled[:batch_size]

    serial_seed_calls = [env_seeds, rollout_seeds]

    def _serial_seed_sampler(batch_size):
        vals = serial_seed_calls.pop(0)
        return vals[:batch_size]

    prior_serial._sample_seed_list = _serial_seed_sampler
    x_serial, y_serial, _ = prior_serial.get_batch(
        batch_size=4,
        n_samples=14,
        num_features=24,
        device="cpu",
        single_eval_pos=7,
    )
    info_serial = prior_serial.last_runtime_info

    env_cfg_vec = dict(env_cfg)
    env_cfg_vec["batch_parallel_backend"] = "torch_vectorized"
    prior_vec = EnvironmentPrior(env_cfg_vec)
    prior_vec._sample_batch_hypers = lambda batch_size: sampled[:batch_size]

    vec_seed_calls = [env_seeds, rollout_seeds]

    def _vec_seed_sampler(batch_size):
        vals = vec_seed_calls.pop(0)
        return vals[:batch_size]

    prior_vec._sample_seed_list = _vec_seed_sampler
    x_vec, y_vec, _ = prior_vec.get_batch(
        batch_size=4,
        n_samples=14,
        num_features=24,
        device="cpu",
        single_eval_pos=7,
    )
    info_vec = prior_vec.last_runtime_info

    assert torch.allclose(x_serial, x_vec)
    assert torch.allclose(y_serial, y_vec)
    assert len(info_serial) == len(info_vec)
    for serial_row, vec_row in zip(info_serial, info_vec):
        assert serial_row.keys() == vec_row.keys()
        for key in serial_row:
                s_val = serial_row[key]
                v_val = vec_row[key]
                if isinstance(s_val, float):
                    assert np.isclose(s_val, v_val, rtol=5e-6, atol=2e-7)
                else:
                    assert s_val == v_val


class _TinyMaskPolicy(nn.Module):
    def __init__(self, obs_dim_cap, action_dim_cap):
        super().__init__()
        self.obs_dim_cap = int(obs_dim_cap)
        self.action_dim_cap = int(action_dim_cap)
        self.net = nn.Sequential(
            nn.Linear(self.obs_dim_cap + self.action_dim_cap + 2, 24),
            nn.Tanh(),
            nn.Linear(24, self.action_dim_cap),
        )

    def step(self, obs_t, action_t, reward_t, reward_mask_t, cache, step_idx, env_info):
        del cache, step_idx
        if obs_t.shape[-1] > self.obs_dim_cap:
            obs_slot = obs_t[:, : self.obs_dim_cap]
        elif obs_t.shape[-1] < self.obs_dim_cap:
            obs_pad = torch.zeros(
                (obs_t.shape[0], self.obs_dim_cap - obs_t.shape[-1]),
                device=obs_t.device,
                dtype=obs_t.dtype,
            )
            obs_slot = torch.cat([obs_t, obs_pad], dim=-1)
        else:
            obs_slot = obs_t

        if action_t.shape[-1] > self.action_dim_cap:
            action_slot = action_t[:, : self.action_dim_cap]
        elif action_t.shape[-1] < self.action_dim_cap:
            action_pad = torch.zeros(
                (action_t.shape[0], self.action_dim_cap - action_t.shape[-1]),
                device=action_t.device,
                dtype=action_t.dtype,
            )
            action_slot = torch.cat([action_t, action_pad], dim=-1)
        else:
            action_slot = action_t

        inp = torch.cat([obs_slot, action_slot, reward_t, reward_mask_t], dim=-1)
        out = self.net(inp)
        action_dim = int(env_info["action_dim"])
        return out[:, :action_dim]


def test_environment_prior_rollout_with_policy_torch_vectorized_matches_serial_semantics_with_identical_h_and_seeds():
    sampled = [
        _manual_sampled_h(
            family="scm",
            state_dim=9,
            obs_dim=6,
            action_dim=3,
            noise_dim=4,
            zero_pad_dim=2,
            num_layers=3,
        ),
        _manual_sampled_h(
            family="gp",
            state_dim=7,
            obs_dim=5,
            action_dim=2,
            noise_dim=3,
            zero_pad_dim=1,
            gp_rff_features=32,
        ),
        _manual_sampled_h(
            family="scm",
            state_dim=9,
            obs_dim=6,
            action_dim=3,
            noise_dim=4,
            zero_pad_dim=2,
            num_layers=3,
        ),
        _manual_sampled_h(
            family="scm",
            state_dim=11,
            obs_dim=7,
            action_dim=3,
            noise_dim=4,
            zero_pad_dim=2,
            num_layers=2,
        ),
    ]
    env_seeds = [11, 23, 37, 41]
    rollout_seeds = [101, 211, 307, 401]

    config = get_prior_config()
    env_cfg = dict(config["prior"]["environment"])
    env_cfg["batch_shared_environment"] = False
    env_cfg["batch_parallel_workers"] = 1
    env_cfg["batch_vectorized_strict_rng_match"] = True

    _seed_everything(20260228)
    policy_serial = _TinyMaskPolicy(obs_dim_cap=7, action_dim_cap=3)
    policy_vec = _TinyMaskPolicy(obs_dim_cap=7, action_dim_cap=3)
    policy_vec.load_state_dict(policy_serial.state_dict())

    env_cfg_serial = dict(env_cfg)
    env_cfg_serial["batch_parallel_backend"] = "python_thread"
    prior_serial = EnvironmentPrior(env_cfg_serial)
    prior_serial._sample_batch_hypers = lambda batch_size: sampled[:batch_size]
    serial_seed_calls = [env_seeds, rollout_seeds]
    prior_serial._sample_seed_list = lambda batch_size: serial_seed_calls.pop(0)[:batch_size]
    rollout_serial = prior_serial.rollout_with_policy(
        policy_step_fn=policy_serial.step,
        batch_size=4,
        n_samples=14,
        num_features=32,
        device="cpu",
        single_eval_pos=7,
        collect_x=True,
    )
    loss_serial, _ = prior_serial.policy_gradient_loss_from_rewards(rollout_serial["rewards"])
    loss_serial.backward()
    grads_serial = [p.grad.detach().clone() for p in policy_serial.parameters()]

    env_cfg_vec = dict(env_cfg)
    env_cfg_vec["batch_parallel_backend"] = "torch_vectorized"
    prior_vec = EnvironmentPrior(env_cfg_vec)
    prior_vec._sample_batch_hypers = lambda batch_size: sampled[:batch_size]
    vec_seed_calls = [env_seeds, rollout_seeds]
    prior_vec._sample_seed_list = lambda batch_size: vec_seed_calls.pop(0)[:batch_size]
    rollout_vec = prior_vec.rollout_with_policy(
        policy_step_fn=policy_vec.step,
        batch_size=4,
        n_samples=14,
        num_features=32,
        device="cpu",
        single_eval_pos=7,
        collect_x=True,
    )
    loss_vec, _ = prior_vec.policy_gradient_loss_from_rewards(rollout_vec["rewards"])
    loss_vec.backward()
    grads_vec = [p.grad.detach().clone() for p in policy_vec.parameters()]

    assert torch.allclose(rollout_serial["x"], rollout_vec["x"])
    assert torch.allclose(rollout_serial["rewards"], rollout_vec["rewards"])
    # Tiny reduction-order differences around 1e-7 reward-scale can flip sign
    # when objective is numerically close to zero.
    assert torch.allclose(loss_serial.detach(), loss_vec.detach(), atol=2e-6, rtol=1e-5)
    assert len(grads_serial) == len(grads_vec)
    for g_serial, g_vec in zip(grads_serial, grads_vec):
        assert torch.allclose(g_serial, g_vec, atol=1e-6, rtol=1e-5)

    info_serial = rollout_serial["info"]
    info_vec = rollout_vec["info"]
    assert len(info_serial) == len(info_vec)
    for serial_row, vec_row in zip(info_serial, info_vec):
        assert serial_row.keys() == vec_row.keys()
        for key in serial_row:
            s_val = serial_row[key]
            v_val = vec_row[key]
            if isinstance(s_val, float):
                assert np.isclose(s_val, v_val, rtol=5e-6, atol=2e-7)
            else:
                assert s_val == v_val


def test_environment_prior_rollout_with_policy_family_grouping_batches_policy_calls():
    sampled = [
        _manual_sampled_h(
            family="scm",
            state_dim=9,
            obs_dim=6,
            action_dim=3,
            noise_dim=4,
            zero_pad_dim=2,
            num_layers=3,
        ),
        _manual_sampled_h(
            family="scm",
            state_dim=11,
            obs_dim=7,
            action_dim=3,
            noise_dim=4,
            zero_pad_dim=2,
            num_layers=2,
        ),
        _manual_sampled_h(
            family="gp",
            state_dim=8,
            obs_dim=5,
            action_dim=2,
            noise_dim=3,
            zero_pad_dim=1,
            gp_rff_features=32,
        ),
        _manual_sampled_h(
            family="gp",
            state_dim=7,
            obs_dim=5,
            action_dim=2,
            noise_dim=3,
            zero_pad_dim=1,
            gp_rff_features=64,
        ),
    ]

    config = get_prior_config()
    env_cfg = dict(config["prior"]["environment"])
    env_cfg["batch_parallel_backend"] = "torch_vectorized"
    env_cfg["batch_vectorized_grouping"] = "family"
    env_cfg["batch_vectorized_strict_rng_match"] = False
    env_cfg["batch_shared_environment"] = False
    prior = EnvironmentPrior(env_cfg)
    prior._sample_batch_hypers = lambda batch_size: sampled[:batch_size]

    class CountingPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = []

        def step(self, obs_t, action_t, reward_t, reward_mask_t, cache, step_idx, env_info):
            del reward_t, reward_mask_t, cache, step_idx, env_info
            self.calls.append(int(obs_t.shape[0]))
            return torch.zeros_like(action_t)

    policy = CountingPolicy()
    n_samples = 13
    rollout = prior.rollout_with_policy(
        policy_step_fn=policy.step,
        batch_size=4,
        n_samples=n_samples,
        num_features=32,
        device="cpu",
        single_eval_pos=7,
        collect_x=False,
    )

    assert rollout["rewards"].shape == (n_samples, 4)
    # Family grouping keeps slot-homogeneous samples in one outer rollout group;
    # inner structure grouping preserves transition semantics.
    assert len(policy.calls) == n_samples
    assert set(policy.calls) == {4}


def test_environment_prior_rollout_with_policy_family_grouping_matches_serial_in_deterministic_setup():
    sampled = [
        _manual_sampled_h(
            family="scm",
            state_dim=9,
            obs_dim=6,
            action_dim=3,
            noise_dim=4,
            zero_pad_dim=2,
            num_layers=3,
        ),
        _manual_sampled_h(
            family="gp",
            state_dim=7,
            obs_dim=5,
            action_dim=2,
            noise_dim=3,
            zero_pad_dim=1,
            gp_rff_features=32,
        ),
        _manual_sampled_h(
            family="scm",
            state_dim=11,
            obs_dim=7,
            action_dim=3,
            noise_dim=4,
            zero_pad_dim=2,
            num_layers=2,
        ),
        _manual_sampled_h(
            family="gp",
            state_dim=8,
            obs_dim=5,
            action_dim=2,
            noise_dim=3,
            zero_pad_dim=1,
            gp_rff_features=64,
        ),
    ]
    for h in sampled:
        # Remove random function-weight effects so A/B comparison is invariant
        # to environment construction order across grouping strategies.
        h["init_std"] = 0.0
        h["outputscale"] = 0.0
        h["noise"] = 0.0

    config = get_prior_config()
    env_cfg = dict(config["prior"]["environment"])
    env_cfg["batch_shared_environment"] = False
    env_cfg["batch_parallel_workers"] = 1
    env_cfg["batch_vectorized_strict_rng_match"] = False

    _seed_everything(20260301)
    policy_serial = _TinyMaskPolicy(obs_dim_cap=7, action_dim_cap=3)
    policy_vec = _TinyMaskPolicy(obs_dim_cap=7, action_dim_cap=3)
    policy_vec.load_state_dict(policy_serial.state_dict())

    env_cfg_serial = dict(env_cfg)
    env_cfg_serial["batch_parallel_backend"] = "python_thread"
    prior_serial = EnvironmentPrior(env_cfg_serial)
    prior_serial._sample_batch_hypers = lambda batch_size: sampled[:batch_size]
    rollout_serial = prior_serial.rollout_with_policy(
        policy_step_fn=policy_serial.step,
        batch_size=4,
        n_samples=12,
        num_features=32,
        device="cpu",
        single_eval_pos=6,
        collect_x=True,
    )
    loss_serial, _ = prior_serial.policy_gradient_loss_from_rewards(rollout_serial["rewards"])
    loss_serial.backward()
    grads_serial = [p.grad.detach().clone() for p in policy_serial.parameters()]

    env_cfg_vec = dict(env_cfg)
    env_cfg_vec["batch_parallel_backend"] = "torch_vectorized"
    env_cfg_vec["batch_vectorized_grouping"] = "family"
    prior_vec = EnvironmentPrior(env_cfg_vec)
    prior_vec._sample_batch_hypers = lambda batch_size: sampled[:batch_size]
    rollout_vec = prior_vec.rollout_with_policy(
        policy_step_fn=policy_vec.step,
        batch_size=4,
        n_samples=12,
        num_features=32,
        device="cpu",
        single_eval_pos=6,
        collect_x=True,
    )
    loss_vec, _ = prior_vec.policy_gradient_loss_from_rewards(rollout_vec["rewards"])
    loss_vec.backward()
    grads_vec = [p.grad.detach().clone() for p in policy_vec.parameters()]

    assert torch.allclose(rollout_serial["x"], rollout_vec["x"], atol=1e-6, rtol=1e-5)
    assert torch.allclose(rollout_serial["rewards"], rollout_vec["rewards"], atol=1e-6, rtol=1e-5)
    assert torch.allclose(loss_serial.detach(), loss_vec.detach(), atol=1e-6, rtol=1e-5)
    assert len(grads_serial) == len(grads_vec)
    for g_serial, g_vec in zip(grads_serial, grads_vec):
        assert torch.allclose(g_serial, g_vec, atol=1e-6, rtol=1e-5)


def test_environment_prior_rollout_with_policy_family_grouping_avoids_per_sample_env_sampling():
    sampled = [
        _manual_sampled_h(
            family="scm",
            state_dim=9,
            obs_dim=6,
            action_dim=3,
            noise_dim=4,
            zero_pad_dim=2,
            num_layers=3,
        ),
        _manual_sampled_h(
            family="scm",
            state_dim=11,
            obs_dim=7,
            action_dim=3,
            noise_dim=4,
            zero_pad_dim=2,
            num_layers=2,
        ),
        _manual_sampled_h(
            family="gp",
            state_dim=8,
            obs_dim=5,
            action_dim=2,
            noise_dim=3,
            zero_pad_dim=1,
            gp_rff_features=32,
        ),
        _manual_sampled_h(
            family="gp",
            state_dim=7,
            obs_dim=5,
            action_dim=2,
            noise_dim=3,
            zero_pad_dim=1,
            gp_rff_features=64,
        ),
    ]
    config = get_prior_config()
    env_cfg = dict(config["prior"]["environment"])
    env_cfg["batch_parallel_backend"] = "torch_vectorized"
    env_cfg["batch_vectorized_grouping"] = "family"
    env_cfg["batch_vectorized_strict_rng_match"] = False
    env_cfg["batch_shared_environment"] = False
    prior = EnvironmentPrior(env_cfg)
    prior._sample_batch_hypers = lambda batch_size: sampled[:batch_size]

    def _forbidden_single_env_sample(*args, **kwargs):
        del args, kwargs
        raise AssertionError("family-group rollout should not fall back to per-sample _sample_environment")

    prior._sample_environment = _forbidden_single_env_sample

    class ZeroPolicy(nn.Module):
        def step(self, obs_t, action_t, reward_t, reward_mask_t, cache, step_idx, env_info):
            del obs_t, reward_t, reward_mask_t, cache, step_idx, env_info
            return torch.zeros_like(action_t)

    rollout = prior.rollout_with_policy(
        policy_step_fn=ZeroPolicy().step,
        batch_size=4,
        n_samples=10,
        num_features=32,
        device="cpu",
        single_eval_pos=5,
        collect_x=False,
    )
    assert rollout["rewards"].shape == (10, 4)


def test_environment_prior_rollout_with_policy_family_grouping_uses_coarse_subgroups():
    sampled = [
        _manual_sampled_h(
            family="scm",
            state_dim=9,
            obs_dim=6,
            action_dim=3,
            noise_dim=4,
            zero_pad_dim=2,
            num_layers=2,
        ),
        _manual_sampled_h(
            family="scm",
            state_dim=11,
            obs_dim=7,
            action_dim=3,
            noise_dim=4,
            zero_pad_dim=2,
            num_layers=2,
        ),
        _manual_sampled_h(
            family="scm",
            state_dim=8,
            obs_dim=5,
            action_dim=3,
            noise_dim=4,
            zero_pad_dim=2,
            num_layers=3,
        ),
        _manual_sampled_h(
            family="gp",
            state_dim=7,
            obs_dim=5,
            action_dim=2,
            noise_dim=3,
            zero_pad_dim=1,
            gp_rff_features=32,
        ),
    ]
    config = get_prior_config()
    env_cfg = dict(config["prior"]["environment"])
    env_cfg["batch_parallel_backend"] = "torch_vectorized"
    env_cfg["batch_vectorized_grouping"] = "family"
    env_cfg["batch_vectorized_strict_rng_match"] = False
    env_cfg["batch_shared_environment"] = False
    prior = EnvironmentPrior(env_cfg)
    prior._sample_batch_hypers = lambda batch_size: sampled[:batch_size]

    call_sizes = []
    orig_coarse = prior._sample_environment_family_coarse_batch

    def _count_coarse(h_list, device, rng_seeds=None):
        call_sizes.append(len(h_list))
        return orig_coarse(h_list=h_list, device=device, rng_seeds=rng_seeds)

    prior._sample_environment_family_coarse_batch = _count_coarse

    class ZeroPolicy(nn.Module):
        def step(self, obs_t, action_t, reward_t, reward_mask_t, cache, step_idx, env_info):
            del obs_t, reward_t, reward_mask_t, cache, step_idx, env_info
            return torch.zeros_like(action_t)

    rollout = prior.rollout_with_policy(
        policy_step_fn=ZeroPolicy().step,
        batch_size=4,
        n_samples=8,
        num_features=32,
        device="cpu",
        single_eval_pos=4,
        collect_x=False,
    )
    assert rollout["rewards"].shape == (8, 4)
    # Coarse family subgrouping: scm + gp -> 2 groups.
    assert sorted(call_sizes) == [1, 3]


def test_environment_prior_lipschitz_scm_keeps_rollout_grads_finite_under_extreme_init():
    _seed_everything(20260303)
    cfg = dict(get_prior_config()["prior"]["environment"])
    cfg["batch_parallel_backend"] = "torch_vectorized"
    cfg["family"] = {"distribution": "meta_choice", "choice_values": ["scm"]}
    cfg["state_dim"] = {"distribution": "uniform_int", "min": 6, "max": 6}
    cfg["obs_dim"] = {"distribution": "uniform_int", "min": 4, "max": 4}
    cfg["action_dim"] = {"distribution": "uniform_int", "min": 3, "max": 3}
    cfg["noise_dim"] = {"distribution": "uniform_int", "min": 5, "max": 5}
    cfg["zero_pad_dim"] = {"distribution": "uniform_int", "min": 0, "max": 0}
    cfg["num_layers"] = {"distribution": "uniform_int", "min": 4, "max": 4}
    cfg["prior_mlp_hidden_dim"] = {"distribution": "uniform_int", "min": 48, "max": 48}
    cfg["prior_mlp_activations"] = "relu"
    cfg["init_std"] = 1000.0
    cfg["noise_std"] = 0.0
    cfg["lipschitz_enforce"] = True
    cfg["lipschitz_weight_fro_norm_max"] = 1.0
    cfg["lipschitz_gp_outputscale_max"] = 1.0
    prior = EnvironmentPrior(cfg)

    class TinyPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Linear(4 + 3 + 1, 3)

        def step(self, obs_t, action_t, reward_t, cache, step_idx, env_info):
            del cache, step_idx, env_info
            return self.net(torch.cat([obs_t, action_t, reward_t], dim=-1))

    policy = TinyPolicy()
    rollout = prior.rollout_with_policy(
        policy_step_fn=policy.step,
        batch_size=2,
        n_samples=32,
        num_features=16,
        device="cpu",
        single_eval_pos=16,
        collect_x=False,
    )
    loss, _ = prior.policy_gradient_loss_from_rewards(rollout["rewards"])
    loss.backward()
    assert torch.isfinite(rollout["rewards"]).all()
    for p in policy.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()


def test_environment_prior_lipschitz_gp_keeps_rollout_grads_finite_under_tiny_lengthscale():
    _seed_everything(20260303)
    cfg = dict(get_prior_config()["prior"]["environment"])
    cfg["batch_parallel_backend"] = "torch_vectorized"
    cfg["family"] = {"distribution": "meta_choice", "choice_values": ["gp"]}
    cfg["state_dim"] = {"distribution": "uniform_int", "min": 6, "max": 6}
    cfg["obs_dim"] = {"distribution": "uniform_int", "min": 4, "max": 4}
    cfg["action_dim"] = {"distribution": "uniform_int", "min": 3, "max": 3}
    cfg["noise_dim"] = {"distribution": "uniform_int", "min": 5, "max": 5}
    cfg["zero_pad_dim"] = {"distribution": "uniform_int", "min": 0, "max": 0}
    cfg["lengthscale"] = 1e-5
    cfg["outputscale"] = 8.0
    cfg["noise"] = 0.0
    cfg["gp_rff_features"] = 128
    cfg["lipschitz_enforce"] = True
    cfg["lipschitz_weight_fro_norm_max"] = 1.0
    cfg["lipschitz_gp_outputscale_max"] = 1.0
    prior = EnvironmentPrior(cfg)

    class TinyPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Linear(4 + 3 + 1, 3)

        def step(self, obs_t, action_t, reward_t, cache, step_idx, env_info):
            del cache, step_idx, env_info
            return self.net(torch.cat([obs_t, action_t, reward_t], dim=-1))

    policy = TinyPolicy()
    rollout = prior.rollout_with_policy(
        policy_step_fn=policy.step,
        batch_size=2,
        n_samples=32,
        num_features=16,
        device="cpu",
        single_eval_pos=16,
        collect_x=False,
    )
    loss, _ = prior.policy_gradient_loss_from_rewards(rollout["rewards"])
    loss.backward()
    assert torch.isfinite(rollout["rewards"]).all()
    for p in policy.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()
