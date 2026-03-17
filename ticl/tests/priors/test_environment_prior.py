import os
import random
import math
import numpy as np
import pytest
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


def test_environment_prior_clear_rollout_artifacts_resets_all_cached_slots():
    prior = EnvironmentPrior({})
    prior.last_runtime_info = [{"reward_min": -1.0}]
    prior.last_rollout_profile = {"policy_wall_ms": 1.0}
    prior.last_rollout_v2 = {"enabled": 1}
    prior.last_rollout_v3 = {"enabled": 1}
    prior.last_rollout_v4 = {"enabled": 1}
    prior.last_rollout_v5_next = {"enabled": 1}
    prior.last_rollout_lipschitz_audit = {"enabled": 1}

    prior.clear_rollout_artifacts()

    assert prior.last_runtime_info == []
    assert prior.last_rollout_profile is None
    assert prior.last_rollout_v2 is None
    assert prior.last_rollout_v3 is None
    assert prior.last_rollout_v4 is None
    assert prior.last_rollout_v5_next is None
    assert prior.last_rollout_lipschitz_audit is None


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


def test_environment_prior_constrained_dim_sampling_is_stable_per_h_and_respects_budget():
    _seed_everything(321)
    prior = EnvironmentPrior({})
    h = {
        "action_dim": 17,
        "state_dim": 273,
        "obs_dim": 399,
        "noise_dim": 61,
        "zero_pad_dim": 123,
        "constrained_dim_sampling_enabled": True,
        "constrained_dim_sampling_total_budget": 400,
    }

    dims_a = prior._sample_dims(h)
    dims_b = prior._sample_dims(h)

    assert dims_a == dims_b
    state_dim, obs_dim, action_dim, noise_dim, zero_pad_dim = dims_a
    assert action_dim == 17
    assert 1 <= state_dim <= 400
    assert 1 <= obs_dim <= state_dim
    assert noise_dim >= 0
    assert zero_pad_dim >= 0
    assert state_dim + noise_dim + zero_pad_dim == 400


def test_environment_prior_constrained_dim_sampling_rollout_reports_budgeted_dims():
    _seed_everything(456)
    env_cfg = dict(get_prior_config()["prior"]["environment"])
    env_cfg["family"] = {"distribution": "meta_choice", "choice_values": ["scm"]}
    env_cfg["constrained_dim_sampling_enabled"] = True
    env_cfg["constrained_dim_sampling_total_budget"] = 400
    prior = EnvironmentPrior(env_cfg)

    prior.get_batch(
        batch_size=6,
        n_samples=12,
        num_features=32,
        device="cpu",
        single_eval_pos=6,
    )

    assert len(prior.last_runtime_info) == 6
    for row in prior.last_runtime_info:
        state_dim = int(row["state_dim"])
        obs_dim = int(row["obs_dim"])
        noise_dim = int(row["noise_dim"])
        zero_pad_dim = int(row["zero_pad_dim"])
        assert 1 <= obs_dim <= state_dim <= 400
        assert noise_dim >= 0
        assert zero_pad_dim >= 0
        assert state_dim + noise_dim + zero_pad_dim == 400


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


def test_environment_prior_policy_gradient_stats_include_reference_semantics_summary():
    _seed_everything(20260309)
    config = get_prior_config()
    env_cfg = dict(config["prior"]["environment"])
    env_cfg["family"] = {"distribution": "meta_choice", "choice_values": ["scm"]}
    env_cfg["state_dim"] = {"distribution": "uniform_int", "min": 6, "max": 6}
    env_cfg["obs_dim"] = {"distribution": "uniform_int", "min": 4, "max": 4}
    env_cfg["action_dim"] = {"distribution": "uniform_int", "min": 3, "max": 3}
    env_cfg["noise_dim"] = {"distribution": "uniform_int", "min": 5, "max": 5}
    env_cfg["zero_pad_dim"] = {"distribution": "uniform_int", "min": 2, "max": 2}
    env_cfg["strict_joint_transition_enabled"] = True
    env_cfg["batch_parallel_backend"] = "torch_vectorized"
    env_cfg["batch_vectorized_grouping"] = "family"
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
        n_samples=10,
        num_features=24,
        device="cpu",
        single_eval_pos=5,
        normalize=False,
        collect_x=False,
    )

    assert torch.isfinite(loss)
    assert torch.isfinite(rollout["rewards"]).all()
    assert int(stats["rollout_env_count"]) == 3
    assert int(stats["rollout_reference_semantics_count"]) == 3
    assert int(stats["rollout_strict_joint_transition_count"]) == 3
    assert float(stats["rollout_reference_semantics_share"]) == pytest.approx(1.0)
    assert float(stats["rollout_strict_joint_transition_share"]) == pytest.approx(1.0)
    assert int(stats["rollout_exact_scm_count"]) == 3
    assert int(stats["rollout_exact_gp_count"]) == 0
    assert int(stats["rollout_legacy_scm_count"]) == 0
    assert int(stats["rollout_legacy_gp_count"]) == 0
    assert stats["rollout_transition_reference_mode"] == "scm_exact"


def test_environment_prior_envgen_checkpoint_preserves_rollout_semantics():
    _seed_everything(20260306)
    config = get_prior_config()
    env_cfg = dict(config["prior"]["environment"])
    env_cfg["batch_parallel_backend"] = "torch_vectorized"
    env_cfg["batch_vectorized_grouping"] = "family"
    env_cfg["batch_vectorized_strict_rng_match"] = False
    env_cfg["family"] = {"distribution": "meta_choice", "choice_values": ["scm"]}
    env_cfg["state_dim"] = {"distribution": "uniform_int", "min": 6, "max": 6}
    env_cfg["obs_dim"] = {"distribution": "uniform_int", "min": 4, "max": 4}
    env_cfg["action_dim"] = {"distribution": "uniform_int", "min": 3, "max": 3}
    env_cfg["noise_dim"] = {"distribution": "uniform_int", "min": 5, "max": 5}
    env_cfg["zero_pad_dim"] = {"distribution": "uniform_int", "min": 2, "max": 2}
    env_cfg["num_layers"] = {"distribution": "uniform_int", "min": 4, "max": 4}
    env_cfg["prior_mlp_hidden_dim"] = {"distribution": "uniform_int", "min": 32, "max": 32}
    env_cfg["noise_std"] = 0.05
    sampled = [
        _manual_sampled_h(
            family="scm",
            state_dim=6,
            obs_dim=4,
            action_dim=3,
            noise_dim=5,
            zero_pad_dim=2,
            num_layers=4,
        )
        for _ in range(3)
    ]

    class TinyPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(4 + 3 + 1, 32),
                nn.Tanh(),
                nn.Linear(32, 3),
            )

        def step(self, obs_t, action_t, reward_t, cache, step_idx, env_info):
            del cache, step_idx, env_info
            return self.net(torch.cat([obs_t, action_t, reward_t], dim=-1))

    base_policy = TinyPolicy()
    base_state = {
        k: v.detach().clone()
        for k, v in base_policy.state_dict().items()
    }
    test_device = "cuda" if torch.cuda.is_available() else "cpu"

    env_backup = {
        "TICL_POLICY_FUSED_TRANSITION_GENERATOR": os.environ.get("TICL_POLICY_FUSED_TRANSITION_GENERATOR"),
        "TICL_POLICY_ENVGEN_CHECKPOINT": os.environ.get("TICL_POLICY_ENVGEN_CHECKPOINT"),
        "TICL_POLICY_ENVGEN_CHECKPOINT_REENTRANT": os.environ.get("TICL_POLICY_ENVGEN_CHECKPOINT_REENTRANT"),
    }

    def _run(checkpoint_enabled):
        _seed_everything(20260306)
        os.environ["TICL_POLICY_FUSED_TRANSITION_GENERATOR"] = "1"
        os.environ["TICL_POLICY_ENVGEN_CHECKPOINT"] = "1" if checkpoint_enabled else "0"
        os.environ["TICL_POLICY_ENVGEN_CHECKPOINT_REENTRANT"] = "0"
        prior = EnvironmentPrior(env_cfg)
        prior._sample_batch_hypers = lambda batch_size: sampled[:batch_size]
        policy = TinyPolicy()
        policy.load_state_dict(base_state)
        policy.to(test_device)
        loss, rollout, stats = prior.rollout_policy_gradient_loss(
            policy_step_fn=policy.step,
            batch_size=3,
            n_samples=10,
            num_features=24,
            device=test_device,
            single_eval_pos=5,
            normalize=False,
            collect_x=True,
        )
        loss.backward()
        grads = [p.grad.detach().clone() for p in policy.parameters()]
        return {
            "loss": loss.detach().clone(),
            "x": rollout["x"].detach().clone(),
            "rewards": rollout["rewards"].detach().clone(),
            "objective": stats["objective"].detach().clone(),
            "reward_mean": stats["reward_mean"].detach().clone(),
            "transition_fused_enabled": int(stats.get("rollout_transition_fused_enabled", 0) or 0),
            "transition_checkpoint_enabled": int(stats.get("rollout_transition_checkpoint_enabled", 0) or 0),
            "transition_checkpoint_calls": int(stats.get("rollout_transition_checkpoint_call_count", 0) or 0),
            "grads": grads,
        }

    try:
        baseline = _run(False)
        checkpointed = _run(True)
    finally:
        for key, value in env_backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    assert torch.allclose(baseline["loss"], checkpointed["loss"], atol=1e-6, rtol=1e-6)
    assert torch.allclose(baseline["x"], checkpointed["x"], atol=1e-6, rtol=1e-6)
    assert torch.allclose(baseline["rewards"], checkpointed["rewards"], atol=1e-6, rtol=1e-6)
    assert torch.allclose(baseline["objective"], checkpointed["objective"], atol=1e-6, rtol=1e-6)
    assert torch.allclose(baseline["reward_mean"], checkpointed["reward_mean"], atol=1e-6, rtol=1e-6)
    for grad_base, grad_ckpt in zip(baseline["grads"], checkpointed["grads"]):
        assert torch.allclose(grad_base, grad_ckpt, atol=1e-6, rtol=1e-6)
    if int(checkpointed["transition_checkpoint_enabled"]) > 0:
        assert baseline["transition_checkpoint_enabled"] == 0
        assert checkpointed["transition_checkpoint_calls"] > 0


def test_environment_prior_fused_transition_dual_packed_matches_cat_semantics():
    _seed_everything(20260306)
    config = get_prior_config()
    env_cfg = dict(config["prior"]["environment"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env_backup = {
        "TICL_POLICY_ENVGEN_CHECKPOINT": os.environ.get("TICL_POLICY_ENVGEN_CHECKPOINT"),
        "TICL_POLICY_ENVGEN_CHECKPOINT_REENTRANT": os.environ.get("TICL_POLICY_ENVGEN_CHECKPOINT_REENTRANT"),
    }

    try:
        os.environ["TICL_POLICY_ENVGEN_CHECKPOINT"] = "1"
        os.environ["TICL_POLICY_ENVGEN_CHECKPOINT_REENTRANT"] = "0"
        for family in ("scm", "gp"):
            _seed_everything(20260306)
            prior = EnvironmentPrior(env_cfg)
            sampled = [
                _manual_sampled_h(
                    family=family,
                    state_dim=6,
                    obs_dim=4,
                    action_dim=3,
                    noise_dim=5,
                    zero_pad_dim=2,
                    num_layers=4,
                    gp_rff_features=32,
                )
                for _ in range(2)
            ]
            batch_size = len(sampled)
            total_dim = 6 + 4 + 3 + 5 + 2
            in_dims = torch.full((batch_size,), total_dim, device=device, dtype=torch.long)
            state_dims = torch.full((batch_size,), 6, device=device, dtype=torch.long)
            if family == "scm":
                transition_fn = prior._build_scm_hetero_transition_batch_fn(
                    in_dims=in_dims,
                    state_dims=state_dims,
                    h_list=sampled,
                    device=device,
                    depth_values=[4] * batch_size,
                    activation_names=["tanh"] * batch_size,
                    input_mask=None,
                )
            else:
                transition_fn = prior._build_gp_hetero_transition_batch_fn(
                    in_dims=in_dims,
                    state_dims=state_dims,
                    h_list=sampled,
                    device=device,
                    input_mask=None,
                )

            x_base = torch.randn((batch_size, total_dim), device=device, dtype=torch.float32, requires_grad=True)
            x_dual_base = x_base.detach().clone().requires_grad_(True)
            baseline_state, baseline_reward = transition_fn(x_base)
            baseline_loss = baseline_state.sum() + baseline_reward.sum()
            baseline_loss.backward()

            dual_input = torch.cat([x_dual_base, x_dual_base], dim=0)
            packed_state, packed_reward = transition_fn(dual_input, x_is_dual_packed=True)
            packed_loss = packed_state.sum() + packed_reward.sum()
            packed_loss.backward()

            assert torch.allclose(baseline_state, packed_state, atol=1e-6, rtol=1e-6)
            assert torch.allclose(baseline_reward, packed_reward, atol=1e-6, rtol=1e-6)
            assert torch.allclose(x_base.grad, x_dual_base.grad, atol=1e-6, rtol=1e-6)
    finally:
        for key, value in env_backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_environment_prior_scm_hidden_fused_transition_matches_legacy_dual_semantics():
    _seed_everything(20260307)
    config = get_prior_config()
    env_cfg = dict(config["prior"]["environment"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env_backup = {
        "TICL_POLICY_ENVGEN_CHECKPOINT": os.environ.get("TICL_POLICY_ENVGEN_CHECKPOINT"),
        "TICL_POLICY_ENVGEN_CHECKPOINT_REENTRANT": os.environ.get("TICL_POLICY_ENVGEN_CHECKPOINT_REENTRANT"),
        "TICL_POLICY_FUSED_TRANSITION_SCM_HIDDEN_FUSED": os.environ.get("TICL_POLICY_FUSED_TRANSITION_SCM_HIDDEN_FUSED"),
        "TICL_POLICY_ENVGEN_RAGGED_AFFINE": os.environ.get("TICL_POLICY_ENVGEN_RAGGED_AFFINE"),
    }

    def _build_input_mask(sampled, device):
        state_dims = [int(h["state_dim"]) for h in sampled]
        obs_dims = [int(h["obs_dim"]) for h in sampled]
        action_dims = [int(h["action_dim"]) for h in sampled]
        noise_dims = [int(h["noise_dim"]) for h in sampled]
        zero_dims = [int(h["zero_pad_dim"]) for h in sampled]
        max_state = max(state_dims)
        max_obs = max(obs_dims)
        max_action = max(action_dims)
        max_noise = max(noise_dims)
        max_zero = max(zero_dims)
        in_cap = max_state + max_obs + max_action + max_noise + max_zero
        input_mask = torch.zeros((len(sampled), in_cap), device=device, dtype=torch.float32)
        obs_start = max_state
        action_start = max_state + max_obs
        noise_start = action_start + max_action
        zero_start = noise_start + max_noise
        for bi, h in enumerate(sampled):
            state_dim = int(h["state_dim"])
            obs_dim = int(h["obs_dim"])
            action_dim = int(h["action_dim"])
            noise_dim = int(h["noise_dim"])
            zero_dim = int(h["zero_pad_dim"])
            input_mask[bi, :state_dim] = 1.0
            input_mask[bi, obs_start: obs_start + obs_dim] = 1.0
            input_mask[bi, action_start: action_start + action_dim] = 1.0
            input_mask[bi, noise_start: noise_start + noise_dim] = 1.0
            input_mask[bi, zero_start: zero_start + zero_dim] = 1.0
        return input_mask

    def _make_generators(base_seed, batch_size):
        generators = []
        for idx in range(batch_size):
            g = torch.Generator(device=device)
            g.manual_seed(int(base_seed + idx))
            generators.append(g)
        return generators

    def _run(hidden_fused_enabled, dual_packed):
        os.environ["TICL_POLICY_ENVGEN_CHECKPOINT"] = "1"
        os.environ["TICL_POLICY_ENVGEN_CHECKPOINT_REENTRANT"] = "0"
        os.environ["TICL_POLICY_FUSED_TRANSITION_SCM_HIDDEN_FUSED"] = "1" if hidden_fused_enabled else "0"
        os.environ["TICL_POLICY_ENVGEN_RAGGED_AFFINE"] = "0"
        _seed_everything(20260307)
        prior = EnvironmentPrior(env_cfg)
        sampled = [
            _manual_sampled_h(
                family="scm",
                state_dim=6,
                obs_dim=4,
                action_dim=3,
                noise_dim=5,
                zero_pad_dim=2,
                num_layers=4,
            ),
            _manual_sampled_h(
                family="scm",
                state_dim=5,
                obs_dim=3,
                action_dim=2,
                noise_dim=4,
                zero_pad_dim=1,
                num_layers=3,
            ),
        ]
        for h in sampled:
            h["noise_std"] = 0.05
        input_mask = _build_input_mask(sampled, device=device)
        in_dims = input_mask.sum(dim=1).to(dtype=torch.long)
        state_dims = torch.tensor([int(h["state_dim"]) for h in sampled], device=device, dtype=torch.long)
        transition_fn = prior._build_scm_hetero_transition_batch_fn(
            in_dims=in_dims,
            state_dims=state_dims,
            h_list=sampled,
            device=device,
            depth_values=[4, 3],
            activation_names=["tanh", "relu"],
            input_mask=input_mask,
        )
        batch_size = len(sampled)
        in_cap = int(input_mask.shape[1])
        if dual_packed:
            x_input = torch.randn((2 * batch_size, in_cap), device=device, dtype=torch.float32, requires_grad=True)
        else:
            x_input = torch.randn((batch_size, in_cap), device=device, dtype=torch.float32, requires_grad=True)
        grad_state = torch.randn((batch_size, int(state_dims.max().item())), device=device, dtype=torch.float32)
        grad_reward = torch.randn((batch_size, 1), device=device, dtype=torch.float32)
        generators = _make_generators(9900, batch_size)
        state_out, reward_out = transition_fn(
            x_input,
            generators_for_noise=generators,
            x_is_dual_packed=dual_packed,
        )
        loss = (state_out * grad_state).sum() + (reward_out * grad_reward).sum()
        loss.backward()
        return {
            "state": state_out.detach().cpu(),
            "reward": reward_out.detach().cpu(),
            "grad": x_input.grad.detach().cpu(),
            "hidden_fused": bool(getattr(transition_fn, "_scm_hidden_fused_specialized", False)),
        }

    try:
        baseline = _run(False, False)
        fused = _run(True, False)
        assert fused["hidden_fused"]
        assert torch.allclose(baseline["state"], fused["state"], atol=1e-6, rtol=1e-6)
        assert torch.allclose(baseline["reward"], fused["reward"], atol=1e-6, rtol=1e-6)
        assert torch.allclose(baseline["grad"], fused["grad"], atol=1e-6, rtol=1e-6)

        baseline_dual = _run(False, True)
        fused_dual = _run(True, True)
        assert fused_dual["hidden_fused"]
        assert torch.allclose(baseline_dual["state"], fused_dual["state"], atol=1e-6, rtol=1e-6)
        assert torch.allclose(baseline_dual["reward"], fused_dual["reward"], atol=1e-6, rtol=1e-6)
        assert torch.allclose(baseline_dual["grad"], fused_dual["grad"], atol=1e-6, rtol=1e-6)
    finally:
        for key, value in env_backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_environment_prior_scm_hidden_fused_transition_falls_back_when_temp_budget_is_tight():
    _seed_everything(20260308)
    config = get_prior_config()
    env_cfg = dict(config["prior"]["environment"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env_backup = {
        "TICL_POLICY_FUSED_TRANSITION_SCM_HIDDEN_FUSED": os.environ.get("TICL_POLICY_FUSED_TRANSITION_SCM_HIDDEN_FUSED"),
        "TICL_POLICY_SCM_HIDDEN_FUSED_MAX_TEMP_MB": os.environ.get("TICL_POLICY_SCM_HIDDEN_FUSED_MAX_TEMP_MB"),
    }
    try:
        os.environ["TICL_POLICY_FUSED_TRANSITION_SCM_HIDDEN_FUSED"] = "1"
        os.environ["TICL_POLICY_SCM_HIDDEN_FUSED_MAX_TEMP_MB"] = "0.01"
        prior = EnvironmentPrior(env_cfg)
        sampled = [
            _manual_sampled_h(
                family="scm",
                state_dim=32,
                obs_dim=16,
                action_dim=8,
                noise_dim=8,
                zero_pad_dim=4,
                num_layers=4,
            ),
            _manual_sampled_h(
                family="scm",
                state_dim=24,
                obs_dim=12,
                action_dim=8,
                noise_dim=8,
                zero_pad_dim=4,
                num_layers=4,
            ),
        ]
        for h in sampled:
            h["prior_mlp_hidden_dim"] = 4096
        input_mask = torch.ones((len(sampled), 68), device=device, dtype=torch.float32)
        in_dims = input_mask.sum(dim=1).to(dtype=torch.long)
        state_dims = torch.tensor([int(h["state_dim"]) for h in sampled], device=device, dtype=torch.long)
        transition_fn = prior._build_scm_hetero_transition_batch_fn(
            in_dims=in_dims,
            state_dims=state_dims,
            h_list=sampled,
            device=device,
            depth_values=[4, 4],
            activation_names=["tanh", "tanh"],
            input_mask=input_mask,
        )
        assert not bool(getattr(transition_fn, "_scm_hidden_fused_specialized", False))
    finally:
        for key, value in env_backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_environment_prior_gp_input_rff_fused_transition_matches_legacy_dual_semantics():
    _seed_everything(20260307)
    config = get_prior_config()
    env_cfg = dict(config["prior"]["environment"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env_backup = {
        "TICL_POLICY_ENVGEN_CHECKPOINT": os.environ.get("TICL_POLICY_ENVGEN_CHECKPOINT"),
        "TICL_POLICY_ENVGEN_CHECKPOINT_REENTRANT": os.environ.get("TICL_POLICY_ENVGEN_CHECKPOINT_REENTRANT"),
        "TICL_POLICY_FUSED_TRANSITION_GP_INPUT_FUSED": os.environ.get("TICL_POLICY_FUSED_TRANSITION_GP_INPUT_FUSED"),
        "TICL_POLICY_ENVGEN_RAGGED_AFFINE": os.environ.get("TICL_POLICY_ENVGEN_RAGGED_AFFINE"),
    }

    def _build_input_mask(sampled, device):
        state_dims = [int(h["state_dim"]) for h in sampled]
        obs_dims = [int(h["obs_dim"]) for h in sampled]
        action_dims = [int(h["action_dim"]) for h in sampled]
        noise_dims = [int(h["noise_dim"]) for h in sampled]
        zero_dims = [int(h["zero_pad_dim"]) for h in sampled]
        max_state = max(state_dims)
        max_obs = max(obs_dims)
        max_action = max(action_dims)
        max_noise = max(noise_dims)
        max_zero = max(zero_dims)
        in_cap = max_state + max_obs + max_action + max_noise + max_zero
        input_mask = torch.zeros((len(sampled), in_cap), device=device, dtype=torch.float32)
        obs_start = max_state
        action_start = max_state + max_obs
        noise_start = action_start + max_action
        zero_start = noise_start + max_noise
        for bi, h in enumerate(sampled):
            state_dim = int(h["state_dim"])
            obs_dim = int(h["obs_dim"])
            action_dim = int(h["action_dim"])
            noise_dim = int(h["noise_dim"])
            zero_dim = int(h["zero_pad_dim"])
            input_mask[bi, :state_dim] = 1.0
            input_mask[bi, obs_start: obs_start + obs_dim] = 1.0
            input_mask[bi, action_start: action_start + action_dim] = 1.0
            input_mask[bi, noise_start: noise_start + noise_dim] = 1.0
            input_mask[bi, zero_start: zero_start + zero_dim] = 1.0
        return input_mask

    def _make_generators(base_seed, batch_size):
        generators = []
        for idx in range(batch_size):
            g = torch.Generator(device=device)
            g.manual_seed(int(base_seed + idx))
            generators.append(g)
        return generators

    def _run(input_fused_enabled, dual_packed):
        os.environ["TICL_POLICY_ENVGEN_CHECKPOINT"] = "1"
        os.environ["TICL_POLICY_ENVGEN_CHECKPOINT_REENTRANT"] = "0"
        os.environ["TICL_POLICY_FUSED_TRANSITION_GP_INPUT_FUSED"] = "1" if input_fused_enabled else "0"
        os.environ["TICL_POLICY_ENVGEN_RAGGED_AFFINE"] = "0"
        _seed_everything(20260307)
        prior = EnvironmentPrior(env_cfg)
        sampled = [
            _manual_sampled_h(
                family="gp",
                state_dim=6,
                obs_dim=4,
                action_dim=3,
                noise_dim=5,
                zero_pad_dim=2,
                num_layers=4,
                gp_rff_features=32,
            ),
            _manual_sampled_h(
                family="gp",
                state_dim=5,
                obs_dim=3,
                action_dim=2,
                noise_dim=4,
                zero_pad_dim=1,
                num_layers=3,
                gp_rff_features=24,
            ),
        ]
        for h in sampled:
            h["noise"] = 0.01
        input_mask = _build_input_mask(sampled, device=device)
        in_dims = input_mask.sum(dim=1).to(dtype=torch.long)
        state_dims = torch.tensor([int(h["state_dim"]) for h in sampled], device=device, dtype=torch.long)
        transition_fn = prior._build_gp_hetero_transition_batch_fn(
            in_dims=in_dims,
            state_dims=state_dims,
            h_list=sampled,
            device=device,
            input_mask=input_mask,
        )
        batch_size = len(sampled)
        in_cap = int(input_mask.shape[1])
        value_gen = torch.Generator(device=device)
        value_gen.manual_seed(2026030701)
        if dual_packed:
            x_input = torch.randn(
                (2 * batch_size, in_cap),
                device=device,
                dtype=torch.float32,
                generator=value_gen,
                requires_grad=True,
            )
        else:
            x_input = torch.randn(
                (batch_size, in_cap),
                device=device,
                dtype=torch.float32,
                generator=value_gen,
                requires_grad=True,
            )
        grad_state = torch.randn(
            (batch_size, int(state_dims.max().item())),
            device=device,
            dtype=torch.float32,
            generator=value_gen,
        )
        grad_reward = torch.randn((batch_size, 1), device=device, dtype=torch.float32, generator=value_gen)
        generators = _make_generators(10100, batch_size)
        state_out, reward_out = transition_fn(
            x_input,
            generators_for_noise=generators,
            x_is_dual_packed=dual_packed,
        )
        loss = (state_out * grad_state).sum() + (reward_out * grad_reward).sum()
        loss.backward()
        return {
            "state": state_out.detach().cpu(),
            "reward": reward_out.detach().cpu(),
            "grad": x_input.grad.detach().cpu(),
            "gp_input_fused": bool(getattr(transition_fn, "_gp_input_rff_fused", False)),
        }

    try:
        baseline = _run(False, False)
        fused = _run(True, False)
        assert fused["gp_input_fused"]
        assert torch.allclose(baseline["state"], fused["state"], atol=1e-6, rtol=1e-6)
        assert torch.allclose(baseline["reward"], fused["reward"], atol=1e-6, rtol=1e-6)
        assert torch.allclose(baseline["grad"], fused["grad"], atol=1e-6, rtol=1e-6)

        baseline_dual = _run(False, True)
        fused_dual = _run(True, True)
        assert fused_dual["gp_input_fused"]
        assert torch.allclose(baseline_dual["state"], fused_dual["state"], atol=1e-6, rtol=1e-6)
        assert torch.allclose(baseline_dual["reward"], fused_dual["reward"], atol=1e-6, rtol=1e-6)
        assert torch.allclose(baseline_dual["grad"], fused_dual["grad"], atol=1e-6, rtol=1e-6)
    finally:
        for key, value in env_backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_environment_prior_gp_output_projection_fused_transition_matches_legacy_dual_semantics():
    _seed_everything(20260307)
    config = get_prior_config()
    env_cfg = dict(config["prior"]["environment"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env_backup = {
        "TICL_POLICY_ENVGEN_CHECKPOINT": os.environ.get("TICL_POLICY_ENVGEN_CHECKPOINT"),
        "TICL_POLICY_ENVGEN_CHECKPOINT_REENTRANT": os.environ.get("TICL_POLICY_ENVGEN_CHECKPOINT_REENTRANT"),
        "TICL_POLICY_FUSED_TRANSITION_GP_INPUT_FUSED": os.environ.get("TICL_POLICY_FUSED_TRANSITION_GP_INPUT_FUSED"),
        "TICL_POLICY_FUSED_TRANSITION_GP_OUTPUT_FUSED": os.environ.get("TICL_POLICY_FUSED_TRANSITION_GP_OUTPUT_FUSED"),
        "TICL_POLICY_ENVGEN_RAGGED_AFFINE": os.environ.get("TICL_POLICY_ENVGEN_RAGGED_AFFINE"),
    }

    def _build_input_mask(sampled, device):
        state_dims = [int(h["state_dim"]) for h in sampled]
        obs_dims = [int(h["obs_dim"]) for h in sampled]
        action_dims = [int(h["action_dim"]) for h in sampled]
        noise_dims = [int(h["noise_dim"]) for h in sampled]
        zero_dims = [int(h["zero_pad_dim"]) for h in sampled]
        max_state = max(state_dims)
        max_obs = max(obs_dims)
        max_action = max(action_dims)
        max_noise = max(noise_dims)
        max_zero = max(zero_dims)
        in_cap = max_state + max_obs + max_action + max_noise + max_zero
        input_mask = torch.zeros((len(sampled), in_cap), device=device, dtype=torch.float32)
        obs_start = max_state
        action_start = max_state + max_obs
        noise_start = action_start + max_action
        zero_start = noise_start + max_noise
        for bi, h in enumerate(sampled):
            state_dim = int(h["state_dim"])
            obs_dim = int(h["obs_dim"])
            action_dim = int(h["action_dim"])
            noise_dim = int(h["noise_dim"])
            zero_dim = int(h["zero_pad_dim"])
            input_mask[bi, :state_dim] = 1.0
            input_mask[bi, obs_start: obs_start + obs_dim] = 1.0
            input_mask[bi, action_start: action_start + action_dim] = 1.0
            input_mask[bi, noise_start: noise_start + noise_dim] = 1.0
            input_mask[bi, zero_start: zero_start + zero_dim] = 1.0
        return input_mask

    def _make_generators(base_seed, batch_size):
        generators = []
        for idx in range(batch_size):
            g = torch.Generator(device=device)
            g.manual_seed(int(base_seed + idx))
            generators.append(g)
        return generators

    def _run(output_fused_enabled, dual_packed):
        os.environ["TICL_POLICY_ENVGEN_CHECKPOINT"] = "1"
        os.environ["TICL_POLICY_ENVGEN_CHECKPOINT_REENTRANT"] = "0"
        os.environ["TICL_POLICY_FUSED_TRANSITION_GP_INPUT_FUSED"] = "0"
        os.environ["TICL_POLICY_FUSED_TRANSITION_GP_OUTPUT_FUSED"] = "1" if output_fused_enabled else "0"
        os.environ["TICL_POLICY_ENVGEN_RAGGED_AFFINE"] = "0"
        _seed_everything(20260307)
        prior = EnvironmentPrior(env_cfg)
        sampled = [
            _manual_sampled_h(
                family="gp",
                state_dim=6,
                obs_dim=4,
                action_dim=3,
                noise_dim=5,
                zero_pad_dim=2,
                num_layers=4,
                gp_rff_features=32,
            ),
            _manual_sampled_h(
                family="gp",
                state_dim=5,
                obs_dim=3,
                action_dim=2,
                noise_dim=4,
                zero_pad_dim=1,
                num_layers=3,
                gp_rff_features=24,
            ),
        ]
        for h in sampled:
            h["noise"] = 0.01
        input_mask = _build_input_mask(sampled, device=device)
        in_dims = input_mask.sum(dim=1).to(dtype=torch.long)
        state_dims = torch.tensor([int(h["state_dim"]) for h in sampled], device=device, dtype=torch.long)
        transition_fn = prior._build_gp_hetero_transition_batch_fn(
            in_dims=in_dims,
            state_dims=state_dims,
            h_list=sampled,
            device=device,
            input_mask=input_mask,
        )
        batch_size = len(sampled)
        in_cap = int(input_mask.shape[1])
        if dual_packed:
            x_input = torch.randn((2 * batch_size, in_cap), device=device, dtype=torch.float32, requires_grad=True)
        else:
            x_input = torch.randn((batch_size, in_cap), device=device, dtype=torch.float32, requires_grad=True)
        grad_state = torch.randn((batch_size, int(state_dims.max().item())), device=device, dtype=torch.float32)
        grad_reward = torch.randn((batch_size, 1), device=device, dtype=torch.float32)
        generators = _make_generators(10100, batch_size)
        state_out, reward_out = transition_fn(
            x_input,
            generators_for_noise=generators,
            x_is_dual_packed=dual_packed,
        )
        loss = (state_out * grad_state).sum() + (reward_out * grad_reward).sum()
        loss.backward()
        return {
            "state": state_out.detach().cpu(),
            "reward": reward_out.detach().cpu(),
            "grad": x_input.grad.detach().cpu(),
            "gp_output_fused": bool(getattr(transition_fn, "_gp_output_projection_fused", False)),
        }

    try:
        baseline = _run(False, False)
        fused = _run(True, False)
        assert fused["gp_output_fused"]
        assert torch.allclose(baseline["state"], fused["state"], atol=1e-6, rtol=1e-6)
        assert torch.allclose(baseline["reward"], fused["reward"], atol=1e-6, rtol=1e-6)
        assert torch.allclose(baseline["grad"], fused["grad"], atol=1e-6, rtol=1e-6)

        baseline_dual = _run(False, True)
        fused_dual = _run(True, True)
        assert fused_dual["gp_output_fused"]
        assert torch.allclose(baseline_dual["state"], fused_dual["state"], atol=1e-6, rtol=1e-6)
        assert torch.allclose(baseline_dual["reward"], fused_dual["reward"], atol=1e-6, rtol=1e-6)
        assert torch.allclose(baseline_dual["grad"], fused_dual["grad"], atol=1e-6, rtol=1e-6)
    finally:
        for key, value in env_backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_environment_prior_gp_output_subgraph_fused_transition_matches_legacy_dual_semantics():
    _seed_everything(20260307)
    config = get_prior_config()
    env_cfg = dict(config["prior"]["environment"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env_backup = {
        "TICL_POLICY_ENVGEN_CHECKPOINT": os.environ.get("TICL_POLICY_ENVGEN_CHECKPOINT"),
        "TICL_POLICY_ENVGEN_CHECKPOINT_REENTRANT": os.environ.get("TICL_POLICY_ENVGEN_CHECKPOINT_REENTRANT"),
        "TICL_POLICY_FUSED_TRANSITION_GP_INPUT_FUSED": os.environ.get("TICL_POLICY_FUSED_TRANSITION_GP_INPUT_FUSED"),
        "TICL_POLICY_FUSED_TRANSITION_GP_OUTPUT_FUSED": os.environ.get("TICL_POLICY_FUSED_TRANSITION_GP_OUTPUT_FUSED"),
        "TICL_POLICY_FUSED_TRANSITION_GP_OUTPUT_SUBGRAPH": os.environ.get("TICL_POLICY_FUSED_TRANSITION_GP_OUTPUT_SUBGRAPH"),
        "TICL_POLICY_ENVGEN_RAGGED_AFFINE": os.environ.get("TICL_POLICY_ENVGEN_RAGGED_AFFINE"),
    }

    def _build_input_mask(sampled, device):
        state_dims = [int(h["state_dim"]) for h in sampled]
        obs_dims = [int(h["obs_dim"]) for h in sampled]
        action_dims = [int(h["action_dim"]) for h in sampled]
        noise_dims = [int(h["noise_dim"]) for h in sampled]
        zero_dims = [int(h["zero_pad_dim"]) for h in sampled]
        max_state = max(state_dims)
        max_obs = max(obs_dims)
        max_action = max(action_dims)
        max_noise = max(noise_dims)
        max_zero = max(zero_dims)
        in_cap = max_state + max_obs + max_action + max_noise + max_zero
        input_mask = torch.zeros((len(sampled), in_cap), device=device, dtype=torch.float32)
        obs_start = max_state
        action_start = max_state + max_obs
        noise_start = action_start + max_action
        zero_start = noise_start + max_noise
        for bi, h in enumerate(sampled):
            state_dim = int(h["state_dim"])
            obs_dim = int(h["obs_dim"])
            action_dim = int(h["action_dim"])
            noise_dim = int(h["noise_dim"])
            zero_dim = int(h["zero_pad_dim"])
            input_mask[bi, :state_dim] = 1.0
            input_mask[bi, obs_start: obs_start + obs_dim] = 1.0
            input_mask[bi, action_start: action_start + action_dim] = 1.0
            input_mask[bi, noise_start: noise_start + noise_dim] = 1.0
            input_mask[bi, zero_start: zero_start + zero_dim] = 1.0
        return input_mask

    def _make_generators(base_seed, batch_size):
        generators = []
        for idx in range(batch_size):
            g = torch.Generator(device=device)
            g.manual_seed(int(base_seed + idx))
            generators.append(g)
        return generators

    def _run(subgraph_fused_enabled, dual_packed):
        os.environ["TICL_POLICY_ENVGEN_CHECKPOINT"] = "1"
        os.environ["TICL_POLICY_ENVGEN_CHECKPOINT_REENTRANT"] = "0"
        os.environ["TICL_POLICY_FUSED_TRANSITION_GP_INPUT_FUSED"] = "0"
        os.environ["TICL_POLICY_FUSED_TRANSITION_GP_OUTPUT_FUSED"] = "0"
        os.environ["TICL_POLICY_FUSED_TRANSITION_GP_OUTPUT_SUBGRAPH"] = "1" if subgraph_fused_enabled else "0"
        os.environ["TICL_POLICY_ENVGEN_RAGGED_AFFINE"] = "0"
        _seed_everything(20260307)
        prior = EnvironmentPrior(env_cfg)
        sampled = [
            _manual_sampled_h(
                family="gp",
                state_dim=6,
                obs_dim=4,
                action_dim=3,
                noise_dim=5,
                zero_pad_dim=2,
                num_layers=4,
                gp_rff_features=32,
            ),
            _manual_sampled_h(
                family="gp",
                state_dim=5,
                obs_dim=3,
                action_dim=2,
                noise_dim=4,
                zero_pad_dim=1,
                num_layers=3,
                gp_rff_features=24,
            ),
        ]
        for h in sampled:
            h["noise"] = 0.01
        input_mask = _build_input_mask(sampled, device=device)
        in_dims = input_mask.sum(dim=1).to(dtype=torch.long)
        state_dims = torch.tensor([int(h["state_dim"]) for h in sampled], device=device, dtype=torch.long)
        transition_fn = prior._build_gp_hetero_transition_batch_fn(
            in_dims=in_dims,
            state_dims=state_dims,
            h_list=sampled,
            device=device,
            input_mask=input_mask,
        )
        batch_size = len(sampled)
        in_cap = int(input_mask.shape[1])
        if dual_packed:
            x_input = torch.randn((2 * batch_size, in_cap), device=device, dtype=torch.float32, requires_grad=True)
        else:
            x_input = torch.randn((batch_size, in_cap), device=device, dtype=torch.float32, requires_grad=True)
        grad_state = torch.randn((batch_size, int(state_dims.max().item())), device=device, dtype=torch.float32)
        grad_reward = torch.randn((batch_size, 1), device=device, dtype=torch.float32)
        generators = _make_generators(10100, batch_size)
        state_out, reward_out = transition_fn(
            x_input,
            generators_for_noise=generators,
            x_is_dual_packed=dual_packed,
        )
        loss = (state_out * grad_state).sum() + (reward_out * grad_reward).sum()
        loss.backward()
        return {
            "state": state_out.detach().cpu(),
            "reward": reward_out.detach().cpu(),
            "grad": x_input.grad.detach().cpu(),
            "gp_output_subgraph_fused": bool(getattr(transition_fn, "_gp_output_subgraph_fused", False)),
        }

    try:
        baseline = _run(False, False)
        fused = _run(True, False)
        assert fused["gp_output_subgraph_fused"]
        assert torch.allclose(baseline["state"], fused["state"], atol=1e-6, rtol=1e-6)
        assert torch.allclose(baseline["reward"], fused["reward"], atol=1e-6, rtol=1e-6)
        assert torch.allclose(baseline["grad"], fused["grad"], atol=1e-6, rtol=1e-6)

        baseline_dual = _run(False, True)
        fused_dual = _run(True, True)
        assert fused_dual["gp_output_subgraph_fused"]
        assert torch.allclose(baseline_dual["state"], fused_dual["state"], atol=1e-6, rtol=1e-6)
        assert torch.allclose(baseline_dual["reward"], fused_dual["reward"], atol=1e-6, rtol=1e-6)
        assert torch.allclose(baseline_dual["grad"], fused_dual["grad"], atol=1e-6, rtol=1e-6)
    finally:
        for key, value in env_backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for GP RFF fused transition semantics")
def test_environment_prior_gp_rff_fused_transition_matches_legacy_dual_semantics():
    _seed_everything(20260307)
    config = get_prior_config()
    env_cfg = dict(config["prior"]["environment"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env_backup = {
        "TICL_POLICY_ENVGEN_CHECKPOINT": os.environ.get("TICL_POLICY_ENVGEN_CHECKPOINT"),
        "TICL_POLICY_ENVGEN_CHECKPOINT_REENTRANT": os.environ.get("TICL_POLICY_ENVGEN_CHECKPOINT_REENTRANT"),
        "TICL_POLICY_FUSED_TRANSITION_GP_INPUT_FUSED": os.environ.get("TICL_POLICY_FUSED_TRANSITION_GP_INPUT_FUSED"),
        "TICL_POLICY_FUSED_TRANSITION_GP_OUTPUT_FUSED": os.environ.get("TICL_POLICY_FUSED_TRANSITION_GP_OUTPUT_FUSED"),
        "TICL_POLICY_FUSED_TRANSITION_GP_OUTPUT_SUBGRAPH": os.environ.get("TICL_POLICY_FUSED_TRANSITION_GP_OUTPUT_SUBGRAPH"),
        "TICL_POLICY_FUSED_TRANSITION_GP_RFF_FUSED": os.environ.get("TICL_POLICY_FUSED_TRANSITION_GP_RFF_FUSED"),
        "TICL_POLICY_FUSED_TRANSITION_GP_PACKED_ENV_INPUT": os.environ.get(
            "TICL_POLICY_FUSED_TRANSITION_GP_PACKED_ENV_INPUT"
        ),
        "TICL_POLICY_FUSED_TRANSITION_GP_SHARED_FIRST_PROJ": os.environ.get(
            "TICL_POLICY_FUSED_TRANSITION_GP_SHARED_FIRST_PROJ"
        ),
        "TICL_POLICY_ENVGEN_RAGGED_AFFINE": os.environ.get("TICL_POLICY_ENVGEN_RAGGED_AFFINE"),
    }

    def _build_input_mask(sampled, device):
        state_dims = [int(h["state_dim"]) for h in sampled]
        obs_dims = [int(h["obs_dim"]) for h in sampled]
        action_dims = [int(h["action_dim"]) for h in sampled]
        noise_dims = [int(h["noise_dim"]) for h in sampled]
        zero_dims = [int(h["zero_pad_dim"]) for h in sampled]
        max_state = max(state_dims)
        max_obs = max(obs_dims)
        max_action = max(action_dims)
        max_noise = max(noise_dims)
        max_zero = max(zero_dims)
        in_cap = max_state + max_obs + max_action + max_noise + max_zero
        input_mask = torch.zeros((len(sampled), in_cap), device=device, dtype=torch.float32)
        obs_start = max_state
        action_start = max_state + max_obs
        noise_start = action_start + max_action
        zero_start = noise_start + max_noise
        for bi, h in enumerate(sampled):
            state_dim = int(h["state_dim"])
            obs_dim = int(h["obs_dim"])
            action_dim = int(h["action_dim"])
            noise_dim = int(h["noise_dim"])
            zero_dim = int(h["zero_pad_dim"])
            input_mask[bi, :state_dim] = 1.0
            input_mask[bi, obs_start: obs_start + obs_dim] = 1.0
            input_mask[bi, action_start: action_start + action_dim] = 1.0
            input_mask[bi, noise_start: noise_start + noise_dim] = 1.0
            input_mask[bi, zero_start: zero_start + zero_dim] = 1.0
        return input_mask

    def _make_generators(base_seed, batch_size):
        generators = []
        for idx in range(batch_size):
            g = torch.Generator(device=device)
            g.manual_seed(int(base_seed + idx))
            generators.append(g)
        return generators

    def _run(rff_fused_enabled, dual_packed, packed_env_input_enabled=False, shared_first_proj_enabled=False):
        os.environ["TICL_POLICY_ENVGEN_CHECKPOINT"] = "1"
        os.environ["TICL_POLICY_ENVGEN_CHECKPOINT_REENTRANT"] = "0"
        os.environ["TICL_POLICY_FUSED_TRANSITION_GP_INPUT_FUSED"] = "0"
        os.environ["TICL_POLICY_FUSED_TRANSITION_GP_OUTPUT_FUSED"] = "0"
        os.environ["TICL_POLICY_FUSED_TRANSITION_GP_OUTPUT_SUBGRAPH"] = "0"
        os.environ["TICL_POLICY_FUSED_TRANSITION_GP_RFF_FUSED"] = "1" if rff_fused_enabled else "0"
        os.environ["TICL_POLICY_FUSED_TRANSITION_GP_PACKED_ENV_INPUT"] = "1" if packed_env_input_enabled else "0"
        os.environ["TICL_POLICY_FUSED_TRANSITION_GP_SHARED_FIRST_PROJ"] = "1" if shared_first_proj_enabled else "0"
        os.environ["TICL_POLICY_ENVGEN_RAGGED_AFFINE"] = "0"
        _seed_everything(20260307)
        prior = EnvironmentPrior(env_cfg)
        sampled = [
            _manual_sampled_h(
                family="gp",
                state_dim=6,
                obs_dim=4,
                action_dim=3,
                noise_dim=5,
                zero_pad_dim=2,
                num_layers=4,
                gp_rff_features=32,
            ),
            _manual_sampled_h(
                family="gp",
                state_dim=5,
                obs_dim=3,
                action_dim=2,
                noise_dim=4,
                zero_pad_dim=1,
                num_layers=3,
                gp_rff_features=24,
            ),
        ]
        for h in sampled:
            h["noise"] = 0.01
        input_mask = _build_input_mask(sampled, device=device)
        in_dims = input_mask.sum(dim=1).to(dtype=torch.long)
        state_dims = torch.tensor([int(h["state_dim"]) for h in sampled], device=device, dtype=torch.long)
        transition_fn = prior._build_gp_hetero_transition_batch_fn(
            in_dims=in_dims,
            state_dims=state_dims,
            h_list=sampled,
            device=device,
            input_mask=input_mask,
        )
        batch_size = len(sampled)
        in_cap = int(input_mask.shape[1])
        value_gen = torch.Generator(device=device)
        value_gen.manual_seed(2026030701)
        if dual_packed:
            x_input = torch.randn(
                (2 * batch_size, in_cap),
                device=device,
                dtype=torch.float32,
                generator=value_gen,
                requires_grad=True,
            )
        else:
            x_input = torch.randn(
                (batch_size, in_cap),
                device=device,
                dtype=torch.float32,
                generator=value_gen,
                requires_grad=True,
            )
        grad_state = torch.randn(
            (batch_size, int(state_dims.max().item())),
            device=device,
            dtype=torch.float32,
            generator=value_gen,
        )
        grad_reward = torch.randn((batch_size, 1), device=device, dtype=torch.float32, generator=value_gen)
        generators = _make_generators(10100, batch_size)
        state_out, reward_out = transition_fn(
            x_input,
            generators_for_noise=generators,
            x_is_dual_packed=dual_packed,
        )
        loss = (state_out * grad_state).sum() + (reward_out * grad_reward).sum()
        loss.backward()
        return {
            "state": state_out.detach().cpu(),
            "reward": reward_out.detach().cpu(),
            "grad": x_input.grad.detach().cpu(),
            "gp_rff_fused": bool(getattr(transition_fn, "_gp_rff_fused", False)),
            "gp_packed_env_input": bool(getattr(transition_fn, "_prefers_packed_env_input", False)),
            "gp_shared_first_proj_fused": bool(getattr(transition_fn, "_gp_shared_first_proj_fused", False)),
        }

    try:
        baseline = _run(False, False)
        fused = _run(True, False)
        assert fused["gp_rff_fused"]
        assert torch.allclose(baseline["state"], fused["state"], atol=1e-6, rtol=1e-6)
        assert torch.allclose(baseline["reward"], fused["reward"], atol=1e-6, rtol=1e-6)
        assert torch.allclose(baseline["grad"], fused["grad"], atol=1e-6, rtol=1e-6)

        baseline_dual = _run(False, True)
        fused_dual = _run(True, True)
        assert fused_dual["gp_rff_fused"]
        assert torch.allclose(baseline_dual["state"], fused_dual["state"], atol=1e-6, rtol=1e-6)
        assert torch.allclose(baseline_dual["reward"], fused_dual["reward"], atol=1e-6, rtol=1e-6)
        assert torch.allclose(baseline_dual["grad"], fused_dual["grad"], atol=1e-6, rtol=1e-6)

        packed = _run(False, False, True)
        assert packed["gp_packed_env_input"]
        assert torch.allclose(baseline["state"], packed["state"], atol=1e-6, rtol=1e-6)
        assert torch.allclose(baseline["reward"], packed["reward"], atol=1e-6, rtol=1e-6)
        assert torch.allclose(baseline["grad"], packed["grad"], atol=1e-6, rtol=1e-6)

        shared = _run(False, False, False, True)
        assert shared["gp_shared_first_proj_fused"]
        assert torch.allclose(baseline["state"], shared["state"], atol=1e-6, rtol=1e-6)
        assert torch.allclose(baseline["reward"], shared["reward"], atol=1e-6, rtol=1e-6)
        assert torch.allclose(baseline["grad"], shared["grad"], atol=1e-6, rtol=1e-6)

        shared_dual = _run(False, True, False, True)
        assert shared_dual["gp_shared_first_proj_fused"]
        assert torch.allclose(baseline_dual["state"], shared_dual["state"], atol=1e-6, rtol=1e-6)
        assert torch.allclose(baseline_dual["reward"], shared_dual["reward"], atol=1e-6, rtol=1e-6)
        assert torch.allclose(baseline_dual["grad"], shared_dual["grad"], atol=1e-6, rtol=1e-6)

        packed_dual = _run(False, True, True)
        assert packed_dual["gp_packed_env_input"]
        assert torch.allclose(baseline_dual["state"], packed_dual["state"], atol=1e-6, rtol=1e-6)
        assert torch.allclose(baseline_dual["reward"], packed_dual["reward"], atol=1e-6, rtol=1e-6)
        assert torch.allclose(baseline_dual["grad"], packed_dual["grad"], atol=1e-6, rtol=1e-6)
    finally:
        for key, value in env_backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for GP projection timing profile")
def test_environment_prior_mixed_family_gp_projection_profile_survives_async_rollout():
    _seed_everything(20260307)
    env_backup = {
        "TICL_POLICY_ENVGEN_CHECKPOINT": os.environ.get("TICL_POLICY_ENVGEN_CHECKPOINT"),
        "TICL_POLICY_ENVGEN_CHECKPOINT_REENTRANT": os.environ.get("TICL_POLICY_ENVGEN_CHECKPOINT_REENTRANT"),
        "TICL_POLICY_FUSED_TRANSITION_SCM_HIDDEN_FUSED": os.environ.get(
            "TICL_POLICY_FUSED_TRANSITION_SCM_HIDDEN_FUSED"
        ),
        "TICL_POLICY_FUSED_TRANSITION_GP_INPUT_FUSED": os.environ.get("TICL_POLICY_FUSED_TRANSITION_GP_INPUT_FUSED"),
        "TICL_POLICY_FUSED_TRANSITION_GP_OUTPUT_FUSED": os.environ.get("TICL_POLICY_FUSED_TRANSITION_GP_OUTPUT_FUSED"),
        "TICL_POLICY_FUSED_TRANSITION_GP_OUTPUT_SUBGRAPH": os.environ.get(
            "TICL_POLICY_FUSED_TRANSITION_GP_OUTPUT_SUBGRAPH"
        ),
        "TICL_POLICY_FUSED_TRANSITION_GP_RFF_FUSED": os.environ.get("TICL_POLICY_FUSED_TRANSITION_GP_RFF_FUSED"),
        "TICL_PROFILE_GP_PROJECTION_TIMING": os.environ.get("TICL_PROFILE_GP_PROJECTION_TIMING"),
        "TICL_PROFILE_ROLLOUT_TIMING": os.environ.get("TICL_PROFILE_ROLLOUT_TIMING"),
        "TICL_POLICY_TRANSITION_INNER_GROUPING": os.environ.get("TICL_POLICY_TRANSITION_INNER_GROUPING"),
        "TICL_POLICY_TRANSITION_INNER_MIN_BUCKET": os.environ.get("TICL_POLICY_TRANSITION_INNER_MIN_BUCKET"),
    }

    try:
        os.environ["TICL_POLICY_ENVGEN_CHECKPOINT"] = "1"
        os.environ["TICL_POLICY_ENVGEN_CHECKPOINT_REENTRANT"] = "0"
        os.environ["TICL_POLICY_FUSED_TRANSITION_SCM_HIDDEN_FUSED"] = "1"
        os.environ["TICL_POLICY_FUSED_TRANSITION_GP_INPUT_FUSED"] = "0"
        os.environ["TICL_POLICY_FUSED_TRANSITION_GP_OUTPUT_FUSED"] = "0"
        os.environ["TICL_POLICY_FUSED_TRANSITION_GP_OUTPUT_SUBGRAPH"] = "0"
        os.environ["TICL_POLICY_FUSED_TRANSITION_GP_RFF_FUSED"] = "0"
        os.environ["TICL_PROFILE_GP_PROJECTION_TIMING"] = "1"
        os.environ["TICL_PROFILE_ROLLOUT_TIMING"] = "1"
        os.environ["TICL_POLICY_TRANSITION_INNER_GROUPING"] = "family"
        os.environ["TICL_POLICY_TRANSITION_INNER_MIN_BUCKET"] = "0"

        cfg = dict(get_prior_config()["prior"]["environment"])
        cfg["batch_parallel_backend"] = "torch_vectorized"
        cfg["batch_shared_environment"] = False
        cfg["batch_vectorized_grouping"] = "family"
        prior = EnvironmentPrior(cfg)
        sampled = [
            _manual_sampled_h(
                family="scm",
                state_dim=8,
                obs_dim=6,
                action_dim=4,
                noise_dim=3,
                zero_pad_dim=0,
                num_layers=4,
            ),
            _manual_sampled_h(
                family="scm",
                state_dim=7,
                obs_dim=5,
                action_dim=3,
                noise_dim=2,
                zero_pad_dim=1,
                num_layers=3,
            ),
            _manual_sampled_h(
                family="gp",
                state_dim=6,
                obs_dim=4,
                action_dim=3,
                noise_dim=5,
                zero_pad_dim=2,
                gp_rff_features=48,
            ),
            _manual_sampled_h(
                family="gp",
                state_dim=5,
                obs_dim=3,
                action_dim=2,
                noise_dim=4,
                zero_pad_dim=1,
                gp_rff_features=32,
            ),
        ]

        class ZeroPolicy(nn.Module):
            def step(self, obs_t, action_t, reward_t, cache, step_idx, env_info):
                del obs_t, reward_t, cache, step_idx, env_info
                return torch.zeros_like(action_t)

        prior.rollout_with_policy(
            policy_step_fn=ZeroPolicy().step,
            batch_size=len(sampled),
            n_samples=12,
            num_features=64,
            device="cuda",
            single_eval_pos=6,
            collect_x=False,
            h_list_override=sampled,
        )
        rollout_profile = prior.last_rollout_profile
        assert isinstance(rollout_profile, dict)
        assert int(rollout_profile.get("transition_gp_profile_group_count", 0)) > 0
        assert int(rollout_profile.get("transition_gp_profile_sync_group_count", 0)) > 0
        assert int(rollout_profile.get("transition_gp_projection_call_count", 0)) > 0
        assert float(rollout_profile.get("transition_gp_first_projection_wall_ms", 0.0)) > 0.0
    finally:
        for key, value in env_backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_environment_prior_ragged_affine_matches_dense_hetero_batch_semantics():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for ragged affine test")
    env_backup = {
        "TICL_POLICY_ENVGEN_RAGGED_AFFINE": os.environ.get("TICL_POLICY_ENVGEN_RAGGED_AFFINE"),
        "TICL_POLICY_ENVGEN_BMM": os.environ.get("TICL_POLICY_ENVGEN_BMM"),
        "TICL_POLICY_ENVGEN_CHECKPOINT": os.environ.get("TICL_POLICY_ENVGEN_CHECKPOINT"),
    }

    def _build_input_mask(sampled, device, in_cap):
        mask = torch.zeros((len(sampled), in_cap), device=device, dtype=torch.float32)
        state_start = 0
        obs_start = 8
        action_start = 12
        noise_start = 16
        zero_start = 21
        for bi, h in enumerate(sampled):
            state_dim = int(h["state_dim"])
            obs_dim = int(h["obs_dim"])
            action_dim = int(h["action_dim"])
            noise_dim = int(h["noise_dim"])
            zero_dim = int(h["zero_pad_dim"])
            mask[bi, state_start: state_start + state_dim] = 1.0
            mask[bi, obs_start: obs_start + obs_dim] = 1.0
            mask[bi, action_start: action_start + action_dim] = 1.0
            mask[bi, noise_start: noise_start + noise_dim] = 1.0
            mask[bi, zero_start: zero_start + zero_dim] = 1.0
        return mask

    def _run(family, ragged_enabled):
        os.environ["TICL_POLICY_ENVGEN_RAGGED_AFFINE"] = "1" if ragged_enabled else "0"
        os.environ["TICL_POLICY_ENVGEN_BMM"] = "1"
        os.environ["TICL_POLICY_ENVGEN_CHECKPOINT"] = "0"
        _seed_everything(20260307)
        prior = EnvironmentPrior({})
        sampled = [
            _manual_sampled_h(
                family=family,
                state_dim=6,
                obs_dim=4,
                action_dim=3,
                noise_dim=5,
                zero_pad_dim=2,
                num_layers=4,
                gp_rff_features=32,
            ),
            _manual_sampled_h(
                family=family,
                state_dim=5,
                obs_dim=3,
                action_dim=2,
                noise_dim=4,
                zero_pad_dim=1,
                num_layers=3,
                gp_rff_features=24,
            ),
            _manual_sampled_h(
                family=family,
                state_dim=7,
                obs_dim=2,
                action_dim=4,
                noise_dim=3,
                zero_pad_dim=0,
                num_layers=5,
                gp_rff_features=40,
            ),
        ]
        in_cap = 24
        input_mask = _build_input_mask(sampled, device="cuda", in_cap=in_cap)
        in_dims = input_mask.sum(dim=1).to(dtype=torch.long)
        out_dims = torch.tensor([int(h["state_dim"]) for h in sampled], device="cuda", dtype=torch.long)
        if family == "scm":
            fn = prior._build_scm_hetero_batch_fn(
                in_dims=in_dims,
                out_dims=out_dims,
                h_list=sampled,
                device="cuda",
                depth_values=[4, 3, 5],
                activation_names=["tanh", "relu", "identity"],
                generators=None,
                input_mask=input_mask,
            )
        else:
            fn = prior._build_gp_hetero_batch_fn(
                in_dims=in_dims,
                out_dims=out_dims,
                h_list=sampled,
                device="cuda",
                generators=None,
                input_mask=input_mask,
            )
        x = torch.randn((len(sampled), in_cap), device="cuda", dtype=torch.float32, requires_grad=True)
        grad_weight = torch.randn((len(sampled), int(out_dims.max().item())), device="cuda", dtype=torch.float32)
        y = fn(x)
        loss = (y * grad_weight).sum()
        loss.backward()
        return {
            "y": y.detach().cpu(),
            "grad": x.grad.detach().cpu(),
            "ragged_active": bool(prior.envgen_ragged_affine),
        }

    try:
        dense = _run("scm", False)
        ragged = _run("scm", True)
        if not ragged["ragged_active"]:
            pytest.skip("Triton ragged affine unavailable in this environment")
        assert torch.allclose(dense["y"], ragged["y"], atol=2e-5, rtol=1e-5)
        assert torch.allclose(dense["grad"], ragged["grad"], atol=2e-5, rtol=1e-5)

        dense = _run("gp", False)
        ragged = _run("gp", True)
        assert ragged["ragged_active"]
        assert torch.allclose(dense["y"], ragged["y"], atol=2e-5, rtol=1e-5)
        assert torch.allclose(dense["grad"], ragged["grad"], atol=2e-5, rtol=1e-5)
    finally:
        for key, value in env_backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


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


def test_environment_prior_aev5_next_switch_exposes_method_only_stats():
    _seed_everything(20260305 + 31)
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
    env_cfg["anti_explosion_vanishing_v5_enabled"] = False
    env_cfg["anti_explosion_vanishing_v5_next_enabled"] = True
    env_cfg["anti_explosion_vanishing_v5_next_state_gain_lo"] = 0.985
    env_cfg["anti_explosion_vanishing_v5_next_state_gain_hi"] = 1.035
    env_cfg["anti_explosion_vanishing_v5_next_state_rms_lo"] = 4e-3
    env_cfg["anti_explosion_vanishing_v5_next_state_rms_hi"] = 9e-2
    env_cfg["anti_explosion_vanishing_v5_next_loss_target_std"] = 0.25
    env_cfg["anti_explosion_vanishing_v5_next_loss_scale_lo"] = 0.5
    env_cfg["anti_explosion_vanishing_v5_next_loss_scale_hi"] = 4.0
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
    assert int(stats.get("aev5_next_enabled", 0)) == 1
    assert "objective_with_aev5_next" in stats
    assert "aev5_next_loss_mul" in stats
    assert "aev5_next_scale" in stats
    assert "aev5_next_scale_raw" in stats
    assert "aev5_next_reward_std_ref" in stats
    assert "aev5_next_gain_mean" in stats
    assert "aev5_next_update_rms_mean" in stats
    assert "aev5_next_high_clip_share" in stats
    assert "aev5_next_corridor_trigger_share" in stats
    assert "aev5_next_bias_thermostat_abs_offset" in stats
    assert torch.isfinite(stats["objective_with_aev5_next"])
    assert torch.isfinite(stats["aev5_next_loss_mul"])
    assert torch.isfinite(stats["aev5_next_scale"])
    assert torch.isfinite(stats["aev5_next_scale_raw"])
    assert torch.isfinite(stats["aev5_next_reward_std_ref"])
    assert torch.isfinite(stats["aev5_next_gain_mean"])
    assert torch.isfinite(stats["aev5_next_update_rms_mean"])
    assert torch.isfinite(stats["aev5_next_corridor_trigger_share"])
    assert torch.isfinite(stats["aev5_next_bias_thermostat_abs_offset"])
    assert "aev5_enabled" not in stats
    assert "aev2_penalty" not in stats
    assert "aev3_penalty" not in stats
    assert "aev4_penalty" not in stats


def test_environment_prior_aev5_next_state_update_enforces_high_side_bounds():
    cfg = {
        "enabled": True,
        "state_gain_lo": 0.985,
        "state_gain_hi": 1.035,
        "state_rms_lo": 4e-3,
        "state_rms_hi": 9e-2,
        "state_reward_gate": 0.05,
        "state_low_boost_cap": 1.5,
        "eps": 1e-6,
        "detach_reference": True,
    }
    state_prev = torch.zeros(2, 4, dtype=torch.float32)
    state_next_post = torch.tensor(
        [
            [1.0, -1.0, 1.0, -1.0],
            [0.5, -0.5, 0.5, -0.5],
        ],
        dtype=torch.float32,
    )
    prev_delta = torch.full((2, 4), 0.05, dtype=torch.float32)
    reward_next = torch.tensor([0.2, 0.2], dtype=torch.float32)

    state_next, aux = EnvironmentPrior._apply_aev5_next_state_update(
        state_prev=state_prev,
        state_next_post=state_next_post,
        prev_delta=prev_delta,
        reward_next=reward_next,
        aev5_next_cfg=cfg,
    )

    assert torch.isfinite(state_next).all()
    assert aux is not None
    assert torch.isfinite(aux["gain"]).all()
    assert torch.isfinite(aux["update_rms"]).all()
    assert float(aux["gain"].max()) <= cfg["state_gain_hi"] + 1e-5
    assert float(aux["update_rms"].max()) <= cfg["state_rms_hi"] + 1e-6
    assert float(aux["high_clip"].max()) == 1.0


def test_environment_prior_aev5_next_state_update_recovers_low_side_when_reachable():
    cfg = {
        "enabled": True,
        "state_gain_lo": 0.6,
        "state_gain_hi": 10.0,
        "state_rms_lo": 0.05,
        "state_rms_hi": 10.0,
        "state_reward_gate": 0.05,
        "state_low_boost_cap": 4.0,
        "eps": 1e-6,
        "detach_reference": True,
    }
    state_prev = torch.zeros(1, 4, dtype=torch.float32)
    state_next_post = torch.tensor([[0.02, 0.02, 0.02, 0.02]], dtype=torch.float32)
    prev_delta = torch.tensor([[0.1, 0.1, 0.1, 0.1]], dtype=torch.float32)
    reward_next = torch.tensor([0.2], dtype=torch.float32)

    state_next, aux = EnvironmentPrior._apply_aev5_next_state_update(
        state_prev=state_prev,
        state_next_post=state_next_post,
        prev_delta=prev_delta,
        reward_next=reward_next,
        aev5_next_cfg=cfg,
    )

    assert torch.isfinite(state_next).all()
    assert aux is not None
    assert float(aux["high_clip"].item()) == 0.0
    assert float(aux["low_active"].item()) == 1.0
    assert float(aux["low_boost"].item()) == 1.0
    assert float(aux["gain"].item()) >= cfg["state_gain_lo"] - 1e-4
    assert float(aux["update_rms"].item()) >= cfg["state_rms_lo"] - 1e-4


def test_environment_prior_aev5_next_state_update_low_side_respects_cap_when_unreachable():
    cfg = {
        "enabled": True,
        "state_gain_lo": 0.6,
        "state_gain_hi": 10.0,
        "state_rms_lo": 0.05,
        "state_rms_hi": 10.0,
        "state_reward_gate": 0.05,
        "state_low_boost_cap": 2.0,
        "eps": 1e-6,
        "detach_reference": True,
    }
    state_prev = torch.zeros(1, 4, dtype=torch.float32)
    state_next_post = torch.tensor([[1e-3, 1e-3, 1e-3, 1e-3]], dtype=torch.float32)
    prev_delta = torch.tensor([[0.1, 0.1, 0.1, 0.1]], dtype=torch.float32)
    reward_next = torch.tensor([0.2], dtype=torch.float32)

    _, aux = EnvironmentPrior._apply_aev5_next_state_update(
        state_prev=state_prev,
        state_next_post=state_next_post,
        prev_delta=prev_delta,
        reward_next=reward_next,
        aev5_next_cfg=cfg,
    )

    assert aux is not None
    assert float(aux["high_clip"].item()) == 0.0
    assert float(aux["low_active"].item()) == 1.0
    assert float(aux["low_boost"].item()) == 1.0
    assert float(aux["scale"].item()) <= cfg["state_low_boost_cap"] + 1e-6
    assert float(aux["gain"].item()) < cfg["state_gain_lo"]
    assert float(aux["update_rms"].item()) < cfg["state_rms_lo"]


def test_environment_prior_aev5_next_tbptt_aggregates_loss_scale():
    _seed_everything(20260305 + 32)
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
    env_cfg["anti_explosion_vanishing_v5_enabled"] = False
    env_cfg["anti_explosion_vanishing_v5_next_enabled"] = True
    env_cfg["anti_explosion_vanishing_v5_next_loss_target_std"] = 1e-3
    env_cfg["anti_explosion_vanishing_v5_next_loss_scale_lo"] = 1e-2
    env_cfg["anti_explosion_vanishing_v5_next_loss_scale_hi"] = 1.0
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

    assert torch.isfinite(loss)
    assert torch.isfinite(rollout["rewards"]).all()
    assert float(stats["aev5_next_loss_mul"]) < 0.5
    assert torch.isfinite(stats["aev5_next_scale_raw"])
    assert float(stats["aev5_next_scale"]) == float(stats["aev5_next_loss_mul"])
    assert not torch.allclose(
        stats["objective_with_aev5_next"],
        stats["objective"],
        atol=1e-6,
        rtol=1e-5,
    )


def test_environment_prior_aev5_next_neutral_matches_disabled_semantics():
    def _run(*, enabled):
        _seed_everything(20260305 + 33)
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
        env_cfg["anti_explosion_vanishing_v5_enabled"] = False
        env_cfg["anti_explosion_vanishing_v5_next_enabled"] = bool(enabled)
        env_cfg["anti_explosion_vanishing_v5_next_state_gain_lo"] = 1e-6
        env_cfg["anti_explosion_vanishing_v5_next_state_gain_hi"] = 1e6
        env_cfg["anti_explosion_vanishing_v5_next_state_rms_lo"] = 1e-8
        env_cfg["anti_explosion_vanishing_v5_next_state_rms_hi"] = 1e6
        env_cfg["anti_explosion_vanishing_v5_next_state_reward_gate"] = 1e9
        env_cfg["anti_explosion_vanishing_v5_next_state_low_boost_cap"] = 1.0
        env_cfg["anti_explosion_vanishing_v5_next_loss_target_std"] = 1.0
        env_cfg["anti_explosion_vanishing_v5_next_loss_scale_lo"] = 1.0
        env_cfg["anti_explosion_vanishing_v5_next_loss_scale_hi"] = 1.0
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
        grads = [p.grad.detach().clone() for p in policy.parameters() if p.grad is not None]
        stats = {k: (v.detach().clone() if torch.is_tensor(v) else v) for k, v in stats.items()}
        return loss.detach().clone(), rollout["rewards"].detach().clone(), stats, grads

    loss_off, rewards_off, stats_off, grads_off = _run(enabled=False)
    loss_on, rewards_on, stats_on, grads_on = _run(enabled=True)

    assert torch.allclose(loss_off, loss_on, atol=1e-6, rtol=1e-5)
    assert torch.allclose(rewards_off, rewards_on, atol=1e-6, rtol=1e-5)
    assert torch.allclose(stats_off["objective"], stats_on["objective"], atol=1e-6, rtol=1e-5)
    assert torch.allclose(stats_off["reward_mean"], stats_on["reward_mean"], atol=1e-6, rtol=1e-5)
    assert torch.allclose(stats_off["reward_std"], stats_on["reward_std"], atol=1e-6, rtol=1e-5)
    assert torch.allclose(stats_on["objective_with_aev5_next"], stats_off["objective"], atol=1e-6, rtol=1e-5)
    assert abs(float(stats_on["aev5_next_loss_mul"]) - 1.0) < 1e-6
    assert len(grads_off) == len(grads_on)
    for g_off, g_on in zip(grads_off, grads_on):
        assert torch.allclose(g_off, g_on, atol=1e-6, rtol=1e-5)


def test_environment_prior_aev5_next_detached_thermostat_preserves_gradient_direction():
    prior = EnvironmentPrior({})
    rewards_base = torch.tensor(
        [
            [0.0, 2.0, -2.0],
            [1.0, -1.0, 3.0],
        ],
        dtype=torch.float32,
    )

    rewards_off = rewards_base.clone().requires_grad_(True)
    loss_off, stats_off = prior.policy_gradient_loss_from_rewards(
        rewards_off,
        normalize=False,
        discount=1.0,
        detach_stats=True,
        aev5_cfg={"enabled": False},
        aev5_next_cfg={"enabled": False},
    )
    loss_off.backward()
    grad_off = rewards_off.grad.detach().clone()

    rewards_on = rewards_base.clone().requires_grad_(True)
    loss_on, stats_on = prior.policy_gradient_loss_from_rewards(
        rewards_on,
        normalize=False,
        discount=1.0,
        detach_stats=True,
        aev5_cfg={"enabled": False},
        aev5_next_cfg={
            "enabled": True,
            "loss_target_std": 0.25,
            "loss_scale_lo": 0.5,
            "loss_scale_hi": 4.0,
            "eps": 1e-6,
            "detach_reference": True,
        },
    )
    loss_on.backward()
    grad_on = rewards_on.grad.detach().clone()

    scale = float(stats_on["aev5_next_loss_mul"])
    assert scale > 0.0
    assert torch.allclose(grad_on, grad_off * scale, atol=1e-7, rtol=1e-6)
    assert float(stats_on["objective_with_aev5_next"]) == pytest.approx(
        float(stats_on["objective"]) * scale,
        rel=1e-6,
        abs=1e-7,
    )


@pytest.mark.parametrize(("family", "family_cfg"), [
    ("scm", {
        "num_layers": {"distribution": "uniform_int", "min": 4, "max": 4},
        "prior_mlp_hidden_dim": {"distribution": "uniform_int", "min": 48, "max": 48},
        "prior_mlp_activations": "relu",
        "init_std": 1000.0,
        "noise_std": 0.0,
    }),
    ("gp", {
        "lengthscale": 1e-5,
        "outputscale": 8.0,
        "noise": 0.0,
        "gp_rff_features": 128,
    }),
])
def test_environment_prior_lipschitz_audit_exposes_rollout_stats(family, family_cfg):
    _seed_everything(20260308 + (0 if family == "scm" else 1))
    env_cfg = dict(get_prior_config()["prior"]["environment"])
    env_cfg["batch_parallel_backend"] = "torch_vectorized"
    env_cfg["batch_vectorized_grouping"] = "family"
    env_cfg["family"] = {"distribution": "meta_choice", "choice_values": [family]}
    env_cfg["state_dim"] = {"distribution": "uniform_int", "min": 6, "max": 6}
    env_cfg["obs_dim"] = {"distribution": "uniform_int", "min": 4, "max": 4}
    env_cfg["action_dim"] = {"distribution": "uniform_int", "min": 3, "max": 3}
    env_cfg["noise_dim"] = {"distribution": "uniform_int", "min": 5, "max": 5}
    env_cfg["zero_pad_dim"] = {"distribution": "uniform_int", "min": 0, "max": 0}
    env_cfg["lipschitz_enforce"] = True
    env_cfg["lipschitz_weight_fro_norm_max"] = 1.0
    env_cfg["lipschitz_gp_outputscale_max"] = 1.0
    env_cfg.update(family_cfg)
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
        num_features=16,
        device="cpu",
        single_eval_pos=6,
        tbptt_window=4,
        normalize=False,
        collect_x=False,
    )

    assert torch.isfinite(loss)
    assert isinstance(rollout.get("lipschitz_audit", None), dict)
    assert int(stats.get("lipschitz_audit_enabled", 0)) == 1
    assert float(stats["lipschitz_matrix_clip_share"]) > 0.0
    assert float(stats["lipschitz_matrix_tail_rel_mean"]) > 0.0
    assert float(stats["lipschitz_matrix_projection_rel_mean"]) > 0.0
    assert float(stats["lipschitz_matrix_projection_rel_max"]) >= float(stats["lipschitz_matrix_projection_rel_mean"])
    if family == "gp":
        assert float(stats["lipschitz_outputscale_clip_share"]) > 0.0
        assert float(stats["lipschitz_outputscale_projection_rel_mean"]) > 0.0


def test_environment_prior_aev5_next_bias_audit_exposes_runtime_offsets():
    _seed_everything(20260311)
    env_cfg = dict(get_prior_config()["prior"]["environment"])
    env_cfg["family"] = {"distribution": "meta_choice", "choice_values": ["scm"]}
    env_cfg["state_dim"] = {"distribution": "uniform_int", "min": 6, "max": 6}
    env_cfg["obs_dim"] = {"distribution": "uniform_int", "min": 4, "max": 4}
    env_cfg["action_dim"] = {"distribution": "uniform_int", "min": 3, "max": 3}
    env_cfg["noise_dim"] = {"distribution": "uniform_int", "min": 5, "max": 5}
    env_cfg["zero_pad_dim"] = {"distribution": "uniform_int", "min": 0, "max": 0}
    env_cfg["num_layers"] = {"distribution": "uniform_int", "min": 4, "max": 4}
    env_cfg["prior_mlp_hidden_dim"] = {"distribution": "uniform_int", "min": 48, "max": 48}
    env_cfg["init_std"] = 8.0
    env_cfg["noise_std"] = 0.0
    env_cfg["anti_explosion_vanishing_v5_enabled"] = False
    env_cfg["anti_explosion_vanishing_v5_next_enabled"] = True
    env_cfg["anti_explosion_vanishing_v5_next_state_gain_lo"] = 0.985
    env_cfg["anti_explosion_vanishing_v5_next_state_gain_hi"] = 0.75
    env_cfg["anti_explosion_vanishing_v5_next_state_rms_lo"] = 1e-4
    env_cfg["anti_explosion_vanishing_v5_next_state_rms_hi"] = 1e-3
    env_cfg["anti_explosion_vanishing_v5_next_loss_target_std"] = 1e-3
    env_cfg["anti_explosion_vanishing_v5_next_loss_scale_lo"] = 1e-2
    env_cfg["anti_explosion_vanishing_v5_next_loss_scale_hi"] = 1.0
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
        num_features=16,
        device="cpu",
        single_eval_pos=6,
        tbptt_window=4,
        normalize=False,
        collect_x=False,
    )

    assert torch.isfinite(loss)
    assert torch.isfinite(rollout["rewards"]).all()
    assert torch.isfinite(stats["aev5_next_bias_thermostat_abs_offset"])
    assert torch.isfinite(stats["aev5_next_bias_state_trigger_share"])
    assert torch.isfinite(stats["aev5_next_corridor_trigger_share"])
    assert float(stats["aev5_next_bias_thermostat_abs_offset"]) > 0.0
    assert float(stats["aev5_next_bias_state_trigger_share"]) > 0.0
    assert float(stats["aev5_next_corridor_trigger_share"]) == pytest.approx(
        float(stats["aev5_next_bias_state_trigger_share"]),
        rel=1e-6,
        abs=1e-6,
    )


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


def test_environment_prior_reinforce_loss_uses_total_return_objective():
    _seed_everything(203)
    config = get_prior_config()
    env_cfg = dict(config["prior"]["environment"])
    prior = EnvironmentPrior(env_cfg)

    rewards = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        dtype=torch.float32,
    )
    log_probs = torch.tensor(
        [
            [0.5, -0.25],
            [0.1, 0.2],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )

    loss, stats = prior.reinforce_loss_from_rewards(
        rewards=rewards,
        log_probs=log_probs,
        discount=1.0,
    )

    assert torch.isfinite(loss)
    assert float(stats["objective"]) == 5.0
    assert float(stats["reinforce_return_mean"]) == 5.0
    assert stats["reinforce_baseline_mode"] == "leave_one_out"
    assert float(stats["reward_nonfinite_share"]) == 0.0
    assert float(stats["reinforce_return_nonfinite_share"]) == 0.0
    assert float(stats["reinforce_log_prob_nonfinite_share"]) == 0.0
    assert float(stats["reinforce_adv_nonfinite_share"]) == 0.0

    loss.backward()
    assert log_probs.grad is not None
    assert torch.isfinite(log_probs.grad).all()


def test_environment_prior_first_policy_gradient_loss_uses_mean_reward_objective():
    prior = EnvironmentPrior({})
    rewards = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 5.0],
        ],
        dtype=torch.float32,
    )

    loss, stats = prior.first_policy_gradient_loss_from_rewards(rewards)

    assert torch.isfinite(loss)
    assert float(stats["objective"]) == pytest.approx(float(rewards.mean()))
    assert float(loss) == pytest.approx(-float(rewards.mean()))


def _alpha_grad_unit_normalize_manual(values, valid_mask, delta):
    mask_f = valid_mask.to(device=values.device, dtype=values.dtype)
    denom = ((values.square() * mask_f).sum(dim=-1, keepdim=True)).sqrt() + float(delta)
    normalized = values / denom
    return torch.where(valid_mask, normalized, torch.zeros_like(normalized))


def _alpha_grad_manual_mix(
    g0,
    g1,
    valid_mask,
    *,
    variance_eps=1e-6,
    unit_grad_enabled=True,
    unit_grad_delta=1e-6,
):
    if unit_grad_enabled:
        g0_source = _alpha_grad_unit_normalize_manual(g0, valid_mask, unit_grad_delta)
        g1_source = _alpha_grad_unit_normalize_manual(g1, valid_mask, unit_grad_delta)
    else:
        g0_source = torch.where(valid_mask, g0, torch.zeros_like(g0))
        g1_source = torch.where(valid_mask, g1, torch.zeros_like(g1))
    mask_f = valid_mask.to(dtype=g0.dtype)
    count = mask_f.sum(dim=1).clamp_min(1.0)
    mu0 = (g0_source * mask_f).sum(dim=1) / count
    mu1 = (g1_source * mask_f).sum(dim=1) / count
    v0 = (((g0_source - mu0.unsqueeze(1)) ** 2) * mask_f).sum(dim=1) / count
    v1 = (((g1_source - mu1.unsqueeze(1)) ** 2) * mask_f).sum(dim=1) / count
    alpha = v0 / (v0 + v1 + float(variance_eps))
    alpha = torch.where(count > 0, alpha, torch.zeros_like(alpha))
    grad = ((1.0 - alpha).unsqueeze(1) * g0) + (alpha.unsqueeze(1) * g1)
    grad = torch.where(valid_mask, grad, torch.zeros_like(grad))
    return grad, alpha, v0, v1


def test_environment_prior_alpha_grad_matches_manual_action_space_mixing():
    prior = EnvironmentPrior({"discount": 1.0})
    action_mean = torch.tensor(
        [
            [[0.2, -0.3], [0.4, 0.1], [-0.2, 0.5]],
            [[0.6, -0.4], [0.1, 0.7], [0.3, -0.6]],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    action_mask = torch.tensor(
        [
            [[True, True], [True, True], [True, False]],
            [[True, True], [True, True], [True, False]],
        ],
        dtype=torch.bool,
    )
    rewards = (
        (1.5 * action_mean[..., 0])
        - (0.25 * action_mean[..., 1])
    )
    log_probs = (
        (0.7 * action_mean[..., 0])
        + (0.2 * action_mean[..., 1])
    )
    loss, stats = prior.alpha_grad_loss_from_rollout_tensors(
        rewards=rewards,
        log_probs=log_probs,
        action_mean=action_mean,
        action_mask=action_mask,
        discount=1.0,
        variance_eps=1e-6,
    )
    grad_actual = torch.autograd.grad(loss, action_mean, retain_graph=True)[0]

    first_loss, _ = prior.first_policy_gradient_loss_from_rewards(rewards)
    reinforce_loss, reinforce_stats = prior.reinforce_loss_from_rewards(
        rewards=rewards,
        log_probs=log_probs,
        discount=1.0,
    )
    g1 = torch.autograd.grad(first_loss, action_mean, retain_graph=True)[0].detach()
    g0 = torch.autograd.grad(reinforce_loss, action_mean, retain_graph=True)[0].detach()
    valid = action_mask & torch.isfinite(g0) & torch.isfinite(g1)
    grad_expected, _, _, _ = _alpha_grad_manual_mix(
        g0,
        g1,
        valid,
        variance_eps=1e-6,
        unit_grad_enabled=True,
        unit_grad_delta=1e-6,
    )

    assert torch.allclose(grad_actual, grad_expected, atol=1e-6, rtol=1e-5)
    assert float(stats["objective"]) == pytest.approx(float(reinforce_stats["objective"]))
    assert int(stats["alpha_grad_enabled"]) == 1
    assert int(stats["alpha_grad_unit_grad_enabled"]) == 1
    assert float(stats["alpha_grad_valid_share"]) > 0.0


def test_environment_prior_alpha_grad_group_traces_match_dense_manual_gradient():
    prior = EnvironmentPrior({"discount": 1.0, "alpha_grad_local_coordinate_enabled": False})
    action_mean_dense = torch.tensor(
        [
            [[0.2, -0.3], [0.4, 0.1], [-0.2, 0.5]],
            [[0.6, -0.4], [0.1, 0.7], [0.3, -0.6]],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    action_mask = torch.tensor(
        [
            [[True, True], [True, True], [True, False]],
            [[True, True], [True, True], [True, False]],
        ],
        dtype=torch.bool,
    )

    rewards_dense = (1.5 * action_mean_dense[..., 0]) - (0.25 * action_mean_dense[..., 1])
    log_probs_dense = (0.7 * action_mean_dense[..., 0]) + (0.2 * action_mean_dense[..., 1])
    loss_dense, _ = prior.alpha_grad_loss_from_rollout_tensors(
        rewards=rewards_dense,
        log_probs=log_probs_dense,
        action_mean=action_mean_dense,
        action_mask=action_mask,
        discount=1.0,
        variance_eps=1e-6,
    )
    grad_dense = torch.autograd.grad(loss_dense, action_mean_dense, retain_graph=True)[0]

    group0_roots = (
        action_mean_dense.detach()[0, [0, 2]].clone().requires_grad_(),
        action_mean_dense.detach()[1, [0, 2]].clone().requires_grad_(),
    )
    group1_roots = (
        action_mean_dense.detach()[0, [1]].clone().requires_grad_(),
        action_mean_dense.detach()[1, [1]].clone().requires_grad_(),
    )
    action_mean_grouped = (
        {
            "indices": (0, 2),
            "action_mean_roots": group0_roots,
            "action_mask": action_mask[:, [0, 2]],
        },
        {
            "indices": (1,),
            "action_mean_roots": group1_roots,
            "action_mask": action_mask[:, [1]],
        },
    )
    action_mean_rebuilt = torch.zeros_like(action_mean_dense)
    action_mean_rebuilt[0, [0, 2]] = group0_roots[0]
    action_mean_rebuilt[1, [0, 2]] = group0_roots[1]
    action_mean_rebuilt[0, [1]] = group1_roots[0]
    action_mean_rebuilt[1, [1]] = group1_roots[1]
    rewards_grouped = (1.5 * action_mean_rebuilt[..., 0]) - (0.25 * action_mean_rebuilt[..., 1])
    log_probs_grouped = (0.7 * action_mean_rebuilt[..., 0]) + (0.2 * action_mean_rebuilt[..., 1])
    loss_grouped, stats_grouped = prior.alpha_grad_loss_from_rollout_tensors(
        rewards=rewards_grouped,
        log_probs=log_probs_grouped,
        action_mean=action_mean_grouped,
        action_mask=action_mask,
        discount=1.0,
        variance_eps=1e-6,
    )
    grad_group0 = torch.autograd.grad(loss_grouped, group0_roots, retain_graph=True)
    grad_group1 = torch.autograd.grad(loss_grouped, group1_roots, retain_graph=True)

    assert int(stats_grouped["alpha_grad_enabled"]) == 1
    assert int(stats_grouped["alpha_grad_local_coordinate_enabled"]) == 0
    assert torch.allclose(grad_group0[0], grad_dense[0, [0, 2]], atol=1e-6, rtol=1e-5)
    assert torch.allclose(grad_group0[1], grad_dense[1, [0, 2]], atol=1e-6, rtol=1e-5)
    assert torch.allclose(grad_group1[0], grad_dense[0, [1]], atol=1e-6, rtol=1e-5)
    assert torch.allclose(grad_group1[1], grad_dense[1, [1]], atol=1e-6, rtol=1e-5)


def test_environment_prior_alpha_grad_group_traces_local_coordinate_matches_manual_group_local_gradient():
    prior = EnvironmentPrior({"discount": 1.0})
    action_mean_dense = torch.tensor(
        [
            [[0.2, -0.3], [0.4, 0.1], [-0.2, 0.5]],
            [[0.6, -0.4], [0.1, 0.7], [0.3, -0.6]],
        ],
        dtype=torch.float32,
    )
    action_mask = torch.tensor(
        [
            [[True, True], [True, True], [True, False]],
            [[True, True], [True, True], [True, False]],
        ],
        dtype=torch.bool,
    )
    group0_roots = (
        action_mean_dense[0, [0, 2]].clone().requires_grad_(),
        action_mean_dense[1, [0, 2]].clone().requires_grad_(),
    )
    group1_roots = (
        action_mean_dense[0, [1]].clone().requires_grad_(),
        action_mean_dense[1, [1]].clone().requires_grad_(),
    )
    action_mean_grouped = (
        {
            "indices": (0, 2),
            "action_mean_roots": group0_roots,
            "action_mask": action_mask[:, [0, 2]],
        },
        {
            "indices": (1,),
            "action_mean_roots": group1_roots,
            "action_mask": action_mask[:, [1]],
        },
    )
    action_mean_rebuilt = torch.zeros_like(action_mean_dense)
    action_mean_rebuilt[0, [0, 2]] = group0_roots[0]
    action_mean_rebuilt[1, [0, 2]] = group0_roots[1]
    action_mean_rebuilt[0, [1]] = group1_roots[0]
    action_mean_rebuilt[1, [1]] = group1_roots[1]
    rewards_grouped = (1.5 * action_mean_rebuilt[..., 0]) - (0.25 * action_mean_rebuilt[..., 1])
    log_probs_grouped = (0.7 * action_mean_rebuilt[..., 0]) + (0.2 * action_mean_rebuilt[..., 1])
    loss_grouped, stats_grouped = prior.alpha_grad_loss_from_rollout_tensors(
        rewards=rewards_grouped,
        log_probs=log_probs_grouped,
        action_mean=action_mean_grouped,
        action_mask=action_mask,
        discount=1.0,
        variance_eps=1e-6,
    )
    grad_group0 = torch.autograd.grad(loss_grouped, group0_roots, retain_graph=True)
    grad_group1 = torch.autograd.grad(loss_grouped, group1_roots, retain_graph=True)

    first_loss, _ = prior.first_policy_gradient_loss_from_rewards(rewards_grouped)
    reinforce_loss, _ = prior.reinforce_loss_from_rewards(
        rewards=rewards_grouped,
        log_probs=log_probs_grouped,
        discount=1.0,
    )
    g1_group0 = tuple(x.detach() for x in torch.autograd.grad(first_loss, group0_roots, retain_graph=True))
    g1_group1 = tuple(x.detach() for x in torch.autograd.grad(first_loss, group1_roots, retain_graph=True))
    g0_group0 = tuple(x.detach() for x in torch.autograd.grad(reinforce_loss, group0_roots, retain_graph=True))
    g0_group1 = tuple(x.detach() for x in torch.autograd.grad(reinforce_loss, group1_roots, retain_graph=True))

    expected_group0 = []
    expected_group1 = []
    for t_idx in range(len(group0_roots)):
        valid0 = action_mask[t_idx, [0, 2]] & torch.isfinite(g0_group0[t_idx]) & torch.isfinite(g1_group0[t_idx])
        valid1 = action_mask[t_idx, [1]] & torch.isfinite(g0_group1[t_idx]) & torch.isfinite(g1_group1[t_idx])
        grad0, _, _, _ = _alpha_grad_manual_mix(
            g0_group0[t_idx].unsqueeze(0),
            g1_group0[t_idx].unsqueeze(0),
            valid0.unsqueeze(0),
            variance_eps=1e-6,
            unit_grad_enabled=True,
            unit_grad_delta=1e-6,
        )
        grad1, _, _, _ = _alpha_grad_manual_mix(
            g0_group1[t_idx].unsqueeze(0),
            g1_group1[t_idx].unsqueeze(0),
            valid1.unsqueeze(0),
            variance_eps=1e-6,
            unit_grad_enabled=True,
            unit_grad_delta=1e-6,
        )
        expected_group0.append(grad0.squeeze(0))
        expected_group1.append(grad1.squeeze(0))

    assert int(stats_grouped["alpha_grad_enabled"]) == 1
    assert int(stats_grouped["alpha_grad_local_coordinate_enabled"]) == 1
    assert int(stats_grouped["alpha_grad_unit_grad_enabled"]) == 1
    assert torch.allclose(grad_group0[0], expected_group0[0], atol=1e-6, rtol=1e-5)
    assert torch.allclose(grad_group0[1], expected_group0[1], atol=1e-6, rtol=1e-5)
    assert torch.allclose(grad_group1[0], expected_group1[0], atol=1e-6, rtol=1e-5)
    assert torch.allclose(grad_group1[1], expected_group1[1], atol=1e-6, rtol=1e-5)


def test_environment_prior_alpha_grad_log_prob_score_matches_autograd_g0_path():
    prior = EnvironmentPrior({"discount": 1.0})
    action_mean = torch.tensor(
        [
            [[0.2, -0.3], [0.4, 0.1], [-0.2, 0.5]],
            [[0.6, -0.4], [0.1, 0.7], [0.3, -0.6]],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    action_mask = torch.tensor(
        [
            [[True, True], [True, True], [True, False]],
            [[True, True], [True, True], [True, False]],
        ],
        dtype=torch.bool,
    )
    action_std = torch.tensor([[0.7, 0.5, 0.4], [0.7, 0.5, 0.4]], dtype=torch.float32)
    eps = torch.tensor(
        [
            [[0.5, -0.25], [0.2, 0.1], [-0.3, 0.8]],
            [[-0.1, 0.2], [0.4, -0.6], [0.7, -0.5]],
        ],
        dtype=torch.float32,
    )
    pre_tanh_action = action_mean + (eps * action_std.unsqueeze(-1))
    action = torch.tanh(pre_tanh_action)
    rewards = (1.5 * action_mean[..., 0]) - (0.25 * action_mean[..., 1])
    log_probs = torch.stack(
        [
            EnvironmentPrior._squashed_gaussian_log_prob(
                pre_tanh_action[t].detach(),
                action_mean[t],
                action_std[t],
                action=action[t].detach(),
                mask=action_mask[t],
            )
            for t in range(action_mean.shape[0])
        ],
        dim=0,
    )
    log_prob_score = EnvironmentPrior._reinforce_log_prob_score_wrt_action_mean(
        pre_tanh_action.detach(),
        action_mean.detach(),
        action_std.unsqueeze(-1),
        mask=action_mask,
    )

    loss_ref, _ = prior.alpha_grad_loss_from_rollout_tensors(
        rewards=rewards,
        log_probs=log_probs,
        action_mean=action_mean,
        action_mask=action_mask,
        discount=1.0,
        variance_eps=1e-6,
    )
    grad_ref = torch.autograd.grad(loss_ref, action_mean, retain_graph=True)[0]

    loss_score, _ = prior.alpha_grad_loss_from_rollout_tensors(
        rewards=rewards,
        log_probs=log_probs,
        action_mean=action_mean,
        action_mask=action_mask,
        log_prob_score=log_prob_score,
        discount=1.0,
        variance_eps=1e-6,
    )
    grad_score = torch.autograd.grad(loss_score, action_mean)[0]

    assert torch.allclose(grad_score, grad_ref, atol=1e-6, rtol=1e-5)


def test_environment_prior_first_policy_gradient_shares_reinforce_rollout_rewards():
    _seed_everything(204)
    config = get_prior_config()
    env_cfg = dict(config["prior"]["environment"])
    env_cfg["family"] = {"distribution": "meta_choice", "choice_values": ["scm"]}
    env_cfg["batch_parallel_backend"] = "python_thread"
    env_cfg["batch_parallel_workers"] = 1
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
    h_list = prior._sample_batch_hypers(3)
    h_list_reinforce = [dict(h) for h in h_list]
    h_list_first = [dict(h) for h in h_list]
    for h_group in (h_list_reinforce, h_list_first):
        for h in h_group:
            h["reward_dropout_enabled"] = False
            h["reward_dropout_randomize"] = False
            h["reward_dropout_ratio"] = 0.0
            h["action_noise_train_std"] = 0.2
            h["action_noise_eval_std"] = 0.15
    env_seeds = [1201, 1202, 1203]
    rollout_seeds = [2201, 2202, 2203]

    rollout_reinforce = prior.rollout_with_policy(
        policy_step_fn=policy.step,
        batch_size=3,
        n_samples=12,
        num_features=24,
        device="cpu",
        single_eval_pos=6,
        collect_x=True,
        h_list_override=h_list_reinforce,
        env_seeds_override=env_seeds,
        rollout_seeds_override=rollout_seeds,
        policy_objective_kind="reinforce",
    )
    rollout_first = prior.rollout_with_policy(
        policy_step_fn=policy.step,
        batch_size=3,
        n_samples=12,
        num_features=24,
        device="cpu",
        single_eval_pos=6,
        collect_x=True,
        h_list_override=h_list_first,
        env_seeds_override=env_seeds,
        rollout_seeds_override=rollout_seeds,
        policy_objective_kind="first_policy_gradient",
    )

    assert torch.allclose(rollout_reinforce["rewards"], rollout_first["rewards"], atol=1e-6, rtol=1e-6)
    assert torch.allclose(rollout_reinforce["x"], rollout_first["x"], atol=1e-6, rtol=1e-6)
    assert isinstance(rollout_reinforce["reinforce"], dict)
    assert torch.is_tensor(rollout_reinforce["reinforce"]["log_probs"])
    assert rollout_first["reinforce"] is None


def test_environment_prior_alpha_grad_shares_reinforce_rollout_rewards_and_trace():
    _seed_everything(2042)
    config = get_prior_config()
    env_cfg = dict(config["prior"]["environment"])
    env_cfg["family"] = {"distribution": "meta_choice", "choice_values": ["scm"]}
    env_cfg["batch_parallel_backend"] = "python_thread"
    env_cfg["batch_parallel_workers"] = 1
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
    h_list = prior._sample_batch_hypers(3)
    for h in h_list:
        h["reward_dropout_enabled"] = False
        h["reward_dropout_randomize"] = False
        h["reward_dropout_ratio"] = 0.0
        h["action_noise_train_std"] = 0.2
        h["action_noise_eval_std"] = 0.15
    env_seeds = [5201, 5202, 5203]
    rollout_seeds = [6201, 6202, 6203]

    rollout_reinforce = prior.rollout_with_policy(
        policy_step_fn=policy.step,
        batch_size=3,
        n_samples=12,
        num_features=24,
        device="cpu",
        single_eval_pos=6,
        collect_x=True,
        h_list_override=[dict(h) for h in h_list],
        env_seeds_override=env_seeds,
        rollout_seeds_override=rollout_seeds,
        policy_objective_kind="reinforce",
    )
    rollout_alpha = prior.rollout_with_policy(
        policy_step_fn=policy.step,
        batch_size=3,
        n_samples=12,
        num_features=24,
        device="cpu",
        single_eval_pos=6,
        collect_x=True,
        h_list_override=[dict(h) for h in h_list],
        env_seeds_override=env_seeds,
        rollout_seeds_override=rollout_seeds,
        policy_objective_kind="alpha_grad",
        _policy_collect_action_trace=True,
    )

    assert torch.allclose(rollout_reinforce["rewards"], rollout_alpha["rewards"], atol=1e-6, rtol=1e-6)
    assert torch.allclose(rollout_reinforce["x"], rollout_alpha["x"], atol=1e-6, rtol=1e-6)
    assert isinstance(rollout_alpha["reinforce"], dict)
    assert torch.is_tensor(rollout_alpha["reinforce"]["log_probs"])
    assert isinstance(rollout_alpha["policy_trace"], dict)
    assert torch.is_tensor(rollout_alpha["policy_trace"]["action_mean"])
    assert torch.is_tensor(rollout_alpha["policy_trace"]["action_mask"])


def test_environment_prior_first_policy_gradient_family_grouping_shares_reinforce_rollout_rewards():
    _seed_everything(2041)
    config = get_prior_config()
    env_cfg = dict(config["prior"]["environment"])
    env_cfg["family"] = {"distribution": "meta_choice", "choice_values": ["scm"]}
    env_cfg["batch_parallel_backend"] = "torch_vectorized"
    env_cfg["batch_vectorized_grouping"] = "family"
    env_cfg["state_dim"] = {"distribution": "uniform_int", "min": 6, "max": 6}
    env_cfg["obs_dim"] = {"distribution": "uniform_int", "min": 4, "max": 4}
    env_cfg["action_dim"] = {"distribution": "uniform_int", "min": 3, "max": 3}
    env_cfg["noise_dim"] = {"distribution": "uniform_int", "min": 5, "max": 5}
    env_cfg["zero_pad_dim"] = {"distribution": "uniform_int", "min": 2, "max": 2}
    prior = EnvironmentPrior(dict(env_cfg))

    class TinyPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Linear(4 + 3 + 1, 3)

        def step(self, obs_t, action_t, reward_t, cache, step_idx, env_info):
            del cache, step_idx, env_info
            return self.net(torch.cat([obs_t, action_t, reward_t], dim=-1))

    policy = TinyPolicy()
    h_list = prior._sample_batch_hypers(3)
    h_list_reinforce = [dict(h) for h in h_list]
    h_list_first = [dict(h) for h in h_list]
    for h_group in (h_list_reinforce, h_list_first):
        for h in h_group:
            h["reward_dropout_enabled"] = False
            h["reward_dropout_randomize"] = False
            h["reward_dropout_ratio"] = 0.0
            h["action_noise_train_std"] = 0.2
            h["action_noise_eval_std"] = 0.15
    env_seeds = [3201, 3202, 3203]
    rollout_seeds = [4201, 4202, 4203]

    rollout_reinforce = prior.rollout_with_policy(
        policy_step_fn=policy.step,
        batch_size=3,
        n_samples=12,
        num_features=24,
        device="cpu",
        single_eval_pos=6,
        collect_x=True,
        h_list_override=h_list_reinforce,
        env_seeds_override=env_seeds,
        rollout_seeds_override=rollout_seeds,
        policy_objective_kind="reinforce",
    )
    rollout_first = prior.rollout_with_policy(
        policy_step_fn=policy.step,
        batch_size=3,
        n_samples=12,
        num_features=24,
        device="cpu",
        single_eval_pos=6,
        collect_x=True,
        h_list_override=h_list_first,
        env_seeds_override=env_seeds,
        rollout_seeds_override=rollout_seeds,
        policy_objective_kind="first_policy_gradient",
    )

    assert torch.allclose(rollout_reinforce["rewards"], rollout_first["rewards"], atol=1e-6, rtol=1e-6)
    assert torch.allclose(rollout_reinforce["x"], rollout_first["x"], atol=1e-6, rtol=1e-6)
    assert isinstance(rollout_reinforce["reinforce"], dict)
    assert torch.is_tensor(rollout_reinforce["reinforce"]["log_probs"])
    assert rollout_first["reinforce"] is None


def test_environment_prior_first_policy_gradient_rollout_gradients_are_finite():
    _seed_everything(205)
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
        collect_x=False,
        policy_objective_kind="first_policy_gradient",
    )

    loss.backward()
    grad_sum = sum(p.grad.abs().sum() for p in policy.parameters() if p.grad is not None)
    assert torch.isfinite(loss)
    assert torch.isfinite(rollout["rewards"]).all()
    assert float(stats["objective"]) == pytest.approx(float(rollout["rewards"].mean().detach()))
    assert float(grad_sum) > 0.0
    assert all(torch.isfinite(p.grad).all() for p in policy.parameters() if p.grad is not None)


def test_environment_prior_joint_first_and_reinforce_losses_are_finite():
    _seed_everything(206)
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
    result = prior.rollout_joint_policy_gradient_losses(
        policy_step_fn=policy.step,
        batch_size=2,
        n_samples=12,
        num_features=24,
        device="cpu",
        single_eval_pos=6,
        collect_x=False,
    )

    total_loss = (
        result["first_policy_gradient"]["loss"]
        + (0.01 * result["reinforce"]["loss"])
    )
    total_loss.backward()
    grad_sum = sum(p.grad.abs().sum() for p in policy.parameters() if p.grad is not None)

    assert torch.isfinite(result["first_policy_gradient"]["loss"])
    assert torch.isfinite(result["reinforce"]["loss"])
    assert isinstance(result["rollout"]["reinforce"], dict)
    assert torch.is_tensor(result["rollout"]["reinforce"]["log_probs"])
    assert torch.isfinite(result["rollout"]["rewards"]).all()
    assert float(grad_sum) > 0.0


def test_environment_prior_alpha_grad_rollout_gradients_are_finite():
    _seed_everything(207)
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
        collect_x=False,
        policy_objective_kind="alpha_grad",
    )

    loss.backward()
    grad_sum = sum(p.grad.abs().sum() for p in policy.parameters() if p.grad is not None)
    assert torch.isfinite(loss)
    assert torch.isfinite(rollout["rewards"]).all()
    assert int(stats["alpha_grad_enabled"]) == 1
    assert float(grad_sum) > 0.0
    assert all(torch.isfinite(p.grad).all() for p in policy.parameters() if p.grad is not None)


def test_environment_prior_alpha_grad_tbptt_multi_group_rollout_gradients_are_finite():
    _seed_everything(208)
    config = get_prior_config()
    env_cfg = dict(config["prior"]["environment"])
    env_cfg["family"] = {"distribution": "meta_choice", "choice_values": ["scm"]}
    env_cfg["state_dim"] = {"distribution": "uniform_int", "min": 6, "max": 6}
    env_cfg["obs_dim"] = {"distribution": "uniform_int", "min": 4, "max": 4}
    env_cfg["action_dim"] = {"distribution": "uniform_int", "min": 3, "max": 3}
    env_cfg["noise_dim"] = {"distribution": "uniform_int", "min": 5, "max": 5}
    env_cfg["zero_pad_dim"] = {"distribution": "uniform_int", "min": 2, "max": 2}
    env_cfg["obs_slot_dim"] = {"distribution": "meta_choice", "choice_values": [400, 401, 402]}
    env_cfg["action_slot_dim"] = {"distribution": "meta_choice", "choice_values": [30, 31, 32]}
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
        batch_size=6,
        n_samples=12,
        num_features=24,
        device="cpu",
        single_eval_pos=6,
        collect_x=False,
        policy_objective_kind="alpha_grad",
        tbptt_window=4,
    )

    loss.backward()
    grad_sum = sum(p.grad.abs().sum() for p in policy.parameters() if p.grad is not None)
    assert torch.isfinite(loss)
    assert torch.isfinite(rollout["rewards"]).all()
    assert int(stats["alpha_grad_enabled"]) == 1
    assert float(grad_sum) > 0.0
    assert all(torch.isfinite(p.grad).all() for p in policy.parameters() if p.grad is not None)


def test_environment_prior_alpha_grad_tbptt_nonfinal_family_window_gradients_are_finite():
    _seed_everything(2081)
    config = get_prior_config()
    env_cfg = dict(config["prior"]["environment"])
    env_cfg["family"] = {"distribution": "meta_choice", "choice_values": ["scm"]}
    env_cfg["state_dim"] = {"distribution": "uniform_int", "min": 6, "max": 6}
    env_cfg["obs_dim"] = {"distribution": "uniform_int", "min": 4, "max": 4}
    env_cfg["action_dim"] = {"distribution": "uniform_int", "min": 3, "max": 3}
    env_cfg["noise_dim"] = {"distribution": "uniform_int", "min": 5, "max": 5}
    env_cfg["zero_pad_dim"] = {"distribution": "uniform_int", "min": 2, "max": 2}
    env_cfg["obs_slot_dim"] = {"distribution": "meta_choice", "choice_values": [400, 401, 402]}
    env_cfg["action_slot_dim"] = {"distribution": "meta_choice", "choice_values": [30, 31, 32]}
    env_cfg["batch_parallel_backend"] = "torch_vectorized"
    env_cfg["batch_vectorized_grouping"] = "family"
    prior = EnvironmentPrior(env_cfg)

    class TinyPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Linear(4 + 3 + 1, 3)

        def step(self, obs_t, action_t, reward_t, cache, step_idx, env_info):
            del cache, step_idx, env_info
            return self.net(torch.cat([obs_t[:, :4], action_t[:, :3], reward_t], dim=-1))

    policy = TinyPolicy()
    loss, rollout, stats = prior.rollout_policy_gradient_loss(
        policy_step_fn=policy.step,
        batch_size=6,
        n_samples=9,
        num_features=24,
        device="cpu",
        single_eval_pos=4,
        collect_x=False,
        policy_objective_kind="alpha_grad",
        tbptt_window=8,
    )

    loss.backward()
    grad_sum = sum(p.grad.abs().sum() for p in policy.parameters() if p.grad is not None)
    assert torch.isfinite(loss)
    assert int(stats["alpha_grad_enabled"]) == 1
    assert float(grad_sum) > 0.0
    assert all(torch.isfinite(p.grad).all() for p in policy.parameters() if p.grad is not None)
    assert tuple(rollout["rewards"].shape) == (0, 6)


def test_environment_prior_reinforce_stats_report_nonfinite_shares():
    prior = EnvironmentPrior()
    rewards = torch.tensor(
        [
            [1.0, float("nan")],
            [float("inf"), 4.0],
        ],
        dtype=torch.float32,
    )
    log_probs = torch.tensor(
        [
            [0.5, -0.25],
            [0.1, float("inf")],
        ],
        dtype=torch.float32,
    )

    _, stats = prior.reinforce_loss_from_rewards(
        rewards=rewards,
        log_probs=log_probs,
        discount=1.0,
    )

    assert float(stats["reward_nonfinite_share"]) > 0.0
    assert float(stats["reward_nan_share"]) > 0.0
    assert float(stats["reward_inf_share"]) > 0.0
    assert float(stats["reinforce_return_nonfinite_share"]) > 0.0
    assert float(stats["reinforce_log_prob_nonfinite_share"]) > 0.0


def test_environment_prior_reinforce_tanh_reward_transform_is_bounded_and_monotone():
    prior = EnvironmentPrior(
        {
            "reinforce_reward_transform": "tanh",
            "reinforce_reward_tanh_c": 2.0,
            "reinforce_reward_tanh_bound": 10.0,
        }
    )
    rewards = torch.tensor(
        [
            [-100.0, -2.0, 0.0, 2.0, 100.0],
        ],
        dtype=torch.float32,
    )
    log_probs = torch.zeros_like(rewards, requires_grad=True)

    transformed, meta = prior._transform_reinforce_rewards(rewards)
    assert meta["mode"] == "tanh"
    assert float(transformed.abs().max()) <= 10.0 + 1e-6
    diffs = transformed[:, 1:] - transformed[:, :-1]
    assert torch.all(diffs >= 0)

    loss, stats = prior.reinforce_loss_from_rewards(
        rewards=transformed,
        log_probs=log_probs,
        discount=1.0,
        baseline_mode="zero",
    )

    assert torch.isfinite(loss)
    assert float(stats["reinforce_reward_used_abs_max"]) <= 10.0 + 1e-6


def test_environment_prior_action_rms_masks_inactive_dims():
    action_raw = torch.tensor(
        [
            [3.0, 4.0, 9.0],
            [5.0, -2.0, 7.0],
        ],
        dtype=torch.float32,
    )
    mask = torch.tensor(
        [
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )

    transformed = EnvironmentPrior._transform_reinforce_action(
        action_raw,
        mode="rms",
        rms_eps=1e-6,
        mask=mask,
    )

    assert torch.allclose(transformed[0, 2], torch.tensor(0.0))
    assert torch.allclose(transformed[1, 0], torch.tensor(0.0))
    assert torch.allclose(transformed[1, 2], torch.tensor(0.0))
    rms_0 = torch.sqrt(transformed[0, :2].square().mean())
    rms_1 = torch.sqrt(transformed[1, 1:2].square().mean())
    assert math.isclose(float(rms_0), 1.0, rel_tol=1e-5, abs_tol=1e-5)
    assert math.isclose(float(rms_1), 1.0, rel_tol=1e-5, abs_tol=1e-5)


def test_environment_prior_state_grad_clip_norm_clips_per_row_global_norm():
    state_next = torch.tensor(
        [
            [1.0, -2.0, 3.0],
            [4.0, 0.0, -3.0],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    state_clipped = EnvironmentPrior._clip_tensor_grad_by_global_norm(
        state_next,
        max_norm=2.0,
    )
    incoming = torch.tensor(
        [
            [6.0, 8.0, 0.0],
            [0.0, -9.0, -12.0],
        ],
        dtype=torch.float32,
    )

    loss = (state_clipped * incoming).sum()
    loss.backward()

    row_norms = state_next.grad.norm(dim=-1)
    assert torch.allclose(
        row_norms,
        torch.tensor([2.0, 2.0], dtype=torch.float32),
        atol=1e-6,
        rtol=1e-6,
    )


def test_environment_prior_action_grad_clip_value_clamps_elementwise():
    action_next = torch.tensor(
        [
            [1.0, -2.0, 3.0],
            [4.0, 0.0, -3.0],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    action_clipped = EnvironmentPrior._clip_tensor_grad_by_value(
        action_next,
        max_abs=2.5,
    )
    incoming = torch.tensor(
        [
            [6.0, -8.0, 0.0],
            [0.0, -9.0, 12.0],
        ],
        dtype=torch.float32,
    )

    loss = (action_clipped * incoming).sum()
    loss.backward()

    assert torch.allclose(
        action_next.grad,
        torch.tensor(
            [
                [2.5, -2.5, 0.0],
                [0.0, -2.5, 2.5],
            ],
            dtype=torch.float32,
        ),
        atol=1e-6,
        rtol=1e-6,
    )


def test_environment_prior_reference_scm_partition_aligns_autocast_dtype_before_index_copy():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for reference SCM autocast partition regression")

    _seed_everything(20260311)
    cfg = dict(get_prior_config()["prior"]["environment"])
    prior = EnvironmentPrior(cfg)
    prior.reference_scm_partition_max_bytes = 1

    h_list = prior._sample_batch_hypers(2)
    for h in h_list:
        h["family"] = "scm"

    state_dims = torch.tensor([int(h["state_dim"]) for h in h_list], dtype=torch.long)
    obs_dims = torch.tensor([int(h["obs_dim"]) for h in h_list], dtype=torch.long)
    action_dims = torch.tensor([int(h["action_dim"]) for h in h_list], dtype=torch.long)
    noise_dims = torch.tensor([int(h["noise_dim"]) for h in h_list], dtype=torch.long)
    zero_pad_dims = torch.tensor([int(h["zero_pad_dim"]) for h in h_list], dtype=torch.long)
    in_dims = state_dims + obs_dims + action_dims + noise_dims + zero_pad_dims

    transition_fn = prior._build_reference_scm_joint_transition_padded_batch_fn(
        in_dims,
        state_dims,
        h_list,
        device=torch.device("cuda"),
        generators=None,
        input_mask=None,
        _allow_partition=True,
    )

    x = torch.randn((2, int(in_dims.max().item())), device="cuda", dtype=torch.float32)
    autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    with torch.autocast(device_type="cuda", dtype=autocast_dtype):
        state_out, reward_out = transition_fn(x)

    assert state_out.dtype == torch.float32
    assert reward_out.dtype == torch.float32
    assert torch.isfinite(state_out).all()
    assert torch.isfinite(reward_out).all()


def test_environment_prior_rollout_applies_action_rms_before_scm_input():
    _seed_everything(20260310)
    cfg = dict(get_prior_config()["prior"]["environment"])
    cfg["family"] = {"distribution": "meta_choice", "choice_values": ["scm"]}
    cfg["batch_parallel_backend"] = "torch_vectorized"
    cfg["state_dim"] = {"distribution": "uniform_int", "min": 4, "max": 4}
    cfg["obs_dim"] = {"distribution": "uniform_int", "min": 2, "max": 2}
    cfg["action_dim"] = {"distribution": "uniform_int", "min": 2, "max": 2}
    cfg["noise_dim"] = {"distribution": "uniform_int", "min": 2, "max": 2}
    cfg["zero_pad_dim"] = {"distribution": "uniform_int", "min": 0, "max": 0}
    cfg["obs_slot_dim"] = 4
    cfg["action_slot_dim"] = 2
    cfg["strict_joint_transition_enabled"] = True
    cfg["reinforce_action_transform"] = "rms"
    cfg["reinforce_action_rms_eps"] = 1e-6
    prior = EnvironmentPrior(cfg)

    class FixedPolicy:
        def step(self, obs_t, action_t, reward_t, cache, step_idx, env_info):
            batch = obs_t.shape[0]
            del action_t, reward_t, cache, step_idx, env_info
            return torch.tensor([[3.0, 4.0]], dtype=torch.float32).expand(batch, -1)

    rollout = prior.rollout_with_policy(
        policy_step_fn=FixedPolicy().step,
        batch_size=2,
        n_samples=3,
        num_features=8,
        device="cpu",
        single_eval_pos=1,
        collect_x=True,
    )
    x = rollout["x"]
    action_start = int(cfg["obs_slot_dim"]) + 2
    action_tokens = x[1, :, action_start: action_start + 2]
    expected = torch.tensor([3.0, 4.0], dtype=torch.float32)
    expected = expected / torch.sqrt(expected.square().mean())
    assert torch.allclose(action_tokens, expected.unsqueeze(0).expand_as(action_tokens), atol=1e-6, rtol=1e-6)


def test_environment_prior_state_full_rms_only_downscales_active_dims():
    state_next = torch.tensor(
        [
            [3.0, 4.0, 100.0],
            [0.2, 0.0, 100.0],
        ],
        dtype=torch.float32,
    )
    state_mask = torch.tensor(
        [
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    enabled = torch.tensor([True, False])

    transformed = EnvironmentPrior._apply_state_full_rms(
        state_next,
        enabled=enabled,
        target=1.0,
        state_mask=state_mask,
        eps=1e-6,
    )

    rms_0 = torch.sqrt(transformed[0, :2].square().mean())
    rms_1 = torch.sqrt(transformed[1, :2].square().mean())
    scale_0 = float(torch.sqrt(state_next[0, :2].square().mean()))
    assert math.isclose(float(rms_0), 1.0, rel_tol=1e-5, abs_tol=1e-5)
    assert math.isclose(
        float(rms_1),
        float(torch.sqrt(state_next[1, :2].square().mean())),
        rel_tol=1e-5,
        abs_tol=1e-5,
    )
    assert math.isclose(
        float(transformed[0, 2]),
        float(state_next[0, 2] / scale_0),
        rel_tol=1e-5,
        abs_tol=1e-5,
    )
    assert torch.allclose(transformed[1, 2], state_next[1, 2])


def test_environment_prior_squashed_gaussian_log_prob_matches_clean_score_function_gradient():
    action_mean = torch.tensor([0.3], dtype=torch.float32, requires_grad=True)
    action_std = 0.7
    eps = torch.tensor([0.5], dtype=torch.float32)
    pre_tanh_action = action_mean + (eps * action_std)
    action = torch.tanh(pre_tanh_action)

    log_prob = EnvironmentPrior._squashed_gaussian_log_prob(
        pre_tanh_action.detach(),
        action_mean,
        action_std,
        action=action.detach(),
    )
    (grad_mean,) = torch.autograd.grad(log_prob, action_mean)
    expected = (pre_tanh_action.detach() - action_mean.detach()) / (action_std ** 2)
    assert torch.allclose(grad_mean, expected, atol=1e-6, rtol=1e-6)


def test_environment_prior_reinforce_log_prob_score_matches_autograd_gradient():
    action_mean = torch.tensor([[0.3, -0.2]], dtype=torch.float32, requires_grad=True)
    action_std = torch.tensor([0.7], dtype=torch.float32)
    eps = torch.tensor([[0.5, -0.25]], dtype=torch.float32)
    mask = torch.tensor([[True, False]], dtype=torch.bool)
    pre_tanh_action = action_mean + (eps * action_std.unsqueeze(-1))
    action = torch.tanh(pre_tanh_action)

    log_prob = EnvironmentPrior._squashed_gaussian_log_prob(
        pre_tanh_action.detach(),
        action_mean,
        action_std,
        action=action.detach(),
        mask=mask,
    )
    (grad_mean,) = torch.autograd.grad(log_prob, action_mean)
    score = EnvironmentPrior._reinforce_log_prob_score_wrt_action_mean(
        pre_tanh_action.detach(),
        action_mean.detach(),
        action_std,
        mask=mask,
    )
    assert torch.allclose(score, grad_mean, atol=1e-6, rtol=1e-6)


def test_environment_prior_pack_env_input_scales_only_state_block():
    state_t = torch.tensor([[4.0, -8.0]], dtype=torch.float32)
    obs_t = torch.tensor([[1.0]], dtype=torch.float32)
    action_t = torch.tensor([[0.25, -0.5]], dtype=torch.float32)
    noise_t = torch.tensor([[2.0]], dtype=torch.float32)
    zero_pad_t = torch.tensor([[0.0, 0.0]], dtype=torch.float32)

    packed = EnvironmentPrior._pack_env_input(
        state_t,
        obs_t,
        action_t,
        noise_t,
        zero_pad_t,
        reference_semantics_enabled=False,
        state_input_scale=4.0,
    )
    assert torch.allclose(
        packed,
        torch.tensor([[1.0, -2.0, 1.0, 0.25, -0.5, 2.0, 0.0, 0.0]], dtype=torch.float32),
    )

    packed_ref = EnvironmentPrior._pack_env_input(
        state_t,
        obs_t,
        action_t,
        noise_t,
        zero_pad_t,
        reference_semantics_enabled=True,
        state_input_scale=4.0,
    )
    assert torch.allclose(
        packed_ref,
        torch.tensor([[1.0, -2.0, 0.25, -0.5, 2.0, 0.0, 0.0]], dtype=torch.float32),
    )


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
        "pre_sample_weights": False,
        "prior_mlp_dropout_prob": 0.0,
        "prior_mlp_scale_weights_sqrt": True,
        "block_wise_dropout": False,
        "y_is_effect": True,
        "sort_features": False,
        "in_clique": False,
        "random_feature_rotation": False,
        "lengthscale": 1.0,
        "outputscale": 1.0,
        "noise": 0.0,
        "gp_rff_features": gp_rff_features,
    }
    return h


def _reference_scm_apply_weight_init(
    weight,
    *,
    init_std,
    activation_name,
    standard_init_enabled,
    prior_mlp_dropout_prob,
    block_wise_dropout,
    prior_mlp_scale_weights_sqrt,
    generator,
):
    base_std = EnvironmentPrior._scm_linear_init_std(
        weight.shape[0],
        weight.shape[1],
        activation_name=activation_name,
        standard_init_enabled=standard_init_enabled,
        init_std=init_std,
    )
    if block_wise_dropout:
        weight.zero_()
        n_blocks = int(
            torch.randint(
                1,
                int(math.ceil(math.sqrt(min(weight.shape[0], weight.shape[1])))) + 1,
                (1,),
                generator=generator,
            ).item()
        )
        block_h = max(1, weight.shape[0] // n_blocks)
        block_w = max(1, weight.shape[1] // n_blocks)
        keep_prob = float((n_blocks * block_h * block_w) / max(1, weight.numel()))
        denom = keep_prob ** (0.5 if prior_mlp_scale_weights_sqrt else 1.0)
        block_std = float(base_std) / max(denom, 1e-12)
        for block_idx in range(n_blocks):
            h_start = block_h * block_idx
            h_end = min(weight.shape[0], block_h * (block_idx + 1))
            w_start = block_w * block_idx
            w_end = min(weight.shape[1], block_w * (block_idx + 1))
            if h_end <= h_start or w_end <= w_start:
                continue
            weight[h_start:h_end, w_start:w_end] = (
                torch.randn((h_end - h_start, w_end - w_start), generator=generator) * block_std
            )
        return

    dropout_prob = float(min(0.99, max(0.0, prior_mlp_dropout_prob)))
    denom = 1.0 - (dropout_prob ** (0.5 if prior_mlp_scale_weights_sqrt else 1.0))
    init_scale = float(base_std) / max(denom, 1e-12)
    weight.copy_(torch.randn(weight.shape, generator=generator) * init_scale)
    if dropout_prob > 0:
        keep_mask = torch.bernoulli(
            torch.zeros_like(weight) + (1.0 - dropout_prob),
            generator=generator,
        )
        weight.mul_(keep_mask)


def _reference_scm_init_bias(bias, fan_in, *, generator):
    bound = 1.0 / math.sqrt(max(1, int(fan_in)))
    bias.copy_((torch.rand(bias.shape, generator=generator) * (2.0 * bound)) - bound)


def _reference_scm_joint_eval(x, h, *, generator):
    state_dim = int(h["state_dim"])
    reward_dim = 1
    in_dim = int(x.shape[-1])
    depth = max(2, int(h["num_layers"]))
    hidden_dim = max(int(h["prior_mlp_hidden_dim"]), reward_dim + 2 * state_dim)
    activation = EnvironmentPrior._resolve_activation(h["prior_mlp_activations"])
    init_std = float(h["init_std"])
    noise_std = float(h["noise_std"])
    pre_sample_weights = bool(h.get("pre_sample_weights", False))
    prior_mlp_dropout_prob = float(h.get("prior_mlp_dropout_prob", 0.0))
    block_wise_dropout = bool(h.get("block_wise_dropout", False))
    prior_mlp_scale_weights_sqrt = bool(h.get("prior_mlp_scale_weights_sqrt", True))
    y_is_effect = bool(h.get("y_is_effect", True))
    sort_features = bool(h.get("sort_features", False))
    in_clique = bool(h.get("in_clique", False))
    random_feature_rotation = bool(h.get("random_feature_rotation", False))
    standard_init_enabled = bool(h.get("scm_standard_linear_init_enabled", False))
    activation_name = EnvironmentPrior._activation_name(h["prior_mlp_activations"])

    first_weight = torch.empty((in_dim, hidden_dim), dtype=x.dtype)
    _reference_scm_apply_weight_init(
        first_weight,
        init_std=init_std,
        activation_name=activation_name,
        standard_init_enabled=standard_init_enabled,
        prior_mlp_dropout_prob=0.0,
        block_wise_dropout=block_wise_dropout,
        prior_mlp_scale_weights_sqrt=prior_mlp_scale_weights_sqrt,
        generator=generator,
    )
    first_bias = torch.empty((hidden_dim,), dtype=x.dtype)
    _reference_scm_init_bias(first_bias, in_dim, generator=generator)

    hidden_weights = []
    hidden_biases = []
    hidden_noise_cfgs = []
    for _ in range(depth - 1):
        weight = torch.empty((hidden_dim, hidden_dim), dtype=x.dtype)
        _reference_scm_apply_weight_init(
            weight,
            init_std=init_std,
            activation_name=activation_name,
            standard_init_enabled=standard_init_enabled,
            prior_mlp_dropout_prob=prior_mlp_dropout_prob,
            block_wise_dropout=block_wise_dropout,
            prior_mlp_scale_weights_sqrt=prior_mlp_scale_weights_sqrt,
            generator=generator,
        )
        bias = torch.empty((hidden_dim,), dtype=x.dtype)
        _reference_scm_init_bias(bias, hidden_dim, generator=generator)
        hidden_weights.append(weight)
        hidden_biases.append(bias)
        if pre_sample_weights and noise_std > 0:
            hidden_noise_cfgs.append(torch.abs(torch.normal(torch.zeros((1, hidden_dim)), noise_std, generator=generator)))
        else:
            hidden_noise_cfgs.append(float(noise_std))

    outputs_flat_dim = (depth - 1) * hidden_dim
    if in_clique:
        clique_hi = max(0, outputs_flat_dim - reward_dim - state_dim)
        clique_start = int(torch.randint(0, clique_hi + 1, (1,), generator=generator).item())
        selection = clique_start + torch.randperm(reward_dim + state_dim, generator=generator)
    else:
        selection = torch.randperm(outputs_flat_dim - 1, generator=generator)
    if y_is_effect:
        reward_index = torch.tensor([outputs_flat_dim - reward_dim], dtype=torch.long)
    else:
        reward_index = selection[:reward_dim].to(dtype=torch.long)
    state_indices = selection[reward_dim: reward_dim + state_dim].to(dtype=torch.long)
    if sort_features:
        state_indices, _ = torch.sort(state_indices)
    rotation_shift = 0
    if random_feature_rotation and state_dim > 0:
        rotation_shift = int(torch.randint(0, state_dim, (1,), generator=generator).item())

    z = x @ first_weight + first_bias
    outputs = []
    for weight, bias, noise_cfg in zip(hidden_weights, hidden_biases, hidden_noise_cfgs):
        z = activation(z)
        z = z @ weight + bias
        if isinstance(noise_cfg, float):
            if noise_cfg > 0:
                z = z + torch.randn(z.shape, generator=generator) * noise_cfg
        else:
            z = z + torch.randn(z.shape, generator=generator) * noise_cfg
        outputs.append(z)
    outputs_flat = torch.cat(outputs, dim=-1)
    state = outputs_flat.index_select(-1, state_indices)
    if rotation_shift > 0 and state.shape[-1] > 0:
        rotate_idx = (torch.arange(state.shape[-1], dtype=torch.long) + rotation_shift) % state.shape[-1]
        state = state.index_select(-1, rotate_idx)
    reward = outputs_flat.index_select(-1, reward_index)
    return state, reward


def _fork_generator_like_builder(seed):
    build_generator = torch.Generator(device="cpu")
    build_generator.manual_seed(int(seed))
    child_seed = int(torch.randint(0, 2**31 - 1, (1,), generator=build_generator).item())
    child = torch.Generator(device="cpu")
    child.manual_seed(child_seed)
    return child


def _reference_gp_kernel(x1, x2, *, lengthscale, outputscale):
    diff = (x1[:, None, :] - x2[None, :, :]) / max(float(lengthscale), 1e-12)
    sq_dist = torch.sum(diff * diff, dim=-1)
    return float(outputscale) * torch.exp(-0.5 * sq_dist)


def _reference_gp_first_call_eval(x, h, *, seed):
    state_dim = int(h["state_dim"])
    out_dim = state_dim + 1
    outputscale = float(h["outputscale"])
    lengthscale = float(h["lengthscale"])
    noise_variance = float(h["noise"])
    latent_generator = _fork_generator_like_builder(seed)
    cov = _reference_gp_kernel(x.to(dtype=torch.float64), x.to(dtype=torch.float64), lengthscale=lengthscale, outputscale=outputscale)
    cov = 0.5 * (cov + cov.transpose(0, 1))
    if float(cov.abs().max().item()) <= 1e-14:
        latent = torch.zeros((x.shape[0], out_dim), dtype=torch.float64)
    else:
        chol = torch.linalg.cholesky(cov + (1e-10 * torch.eye(cov.shape[0], dtype=cov.dtype)))
        eps = torch.randn((x.shape[0], out_dim), dtype=torch.float64, generator=latent_generator)
        latent = chol @ eps
    y = latent
    if noise_variance > 0.0:
        y = y + (torch.randn((x.shape[0], out_dim), dtype=torch.float64) * math.sqrt(noise_variance))
    y = y.to(dtype=torch.float32)
    return y[:, :state_dim], y[:, state_dim: state_dim + 1]

def _coarse_batch_reference_active_index(env_batch, sample_idx):
    sample_idx = int(sample_idx)
    max_state_dim = int(env_batch["state_dim"])
    max_obs_input_dim = int(env_batch.get("env_obs_input_dim", 0))
    max_action_dim = int(env_batch["action_dim"])
    max_noise_dim = int(env_batch["noise_dim"])
    state_dim = int(env_batch["state_dim_per_sample"][sample_idx].item())
    obs_input_dim = int(env_batch["env_obs_input_dim_per_sample"][sample_idx].item())
    action_dim = int(env_batch["action_dim_per_sample"][sample_idx].item())
    noise_dim = int(env_batch["noise_dim_per_sample"][sample_idx].item())
    zero_pad_dim = int(env_batch["zero_pad_dim_per_sample"][sample_idx].item())

    obs_start = max_state_dim
    action_start = max_state_dim + max_obs_input_dim
    noise_start = action_start + max_action_dim
    zero_start = noise_start + max_noise_dim

    parts = []
    if state_dim > 0:
        parts.append(torch.arange(state_dim, dtype=torch.long))
    if obs_input_dim > 0:
        parts.append(obs_start + torch.arange(obs_input_dim, dtype=torch.long))
    if action_dim > 0:
        parts.append(action_start + torch.arange(action_dim, dtype=torch.long))
    if noise_dim > 0:
        parts.append(noise_start + torch.arange(noise_dim, dtype=torch.long))
    if zero_pad_dim > 0:
        parts.append(zero_start + torch.arange(zero_pad_dim, dtype=torch.long))
    if not parts:
        return torch.empty((0,), dtype=torch.long)
    return torch.cat(parts, dim=0)


@pytest.mark.parametrize("family", ["scm", "gp"])
def test_environment_prior_strict_joint_transition_builds_single_generator(family):
    _seed_everything(20260309)
    prior = EnvironmentPrior({})
    h = _manual_sampled_h(
        family=family,
        state_dim=7,
        obs_dim=5,
        action_dim=3,
        noise_dim=4,
        zero_pad_dim=2,
        num_layers=3,
        gp_rff_features=32,
    )
    h["strict_joint_transition_enabled"] = True

    env = prior._sample_environment(h, device="cpu", rng_seed=123)

    assert callable(env["transition_generator"])
    assert env["x_generator"] is None
    assert env["y_generator"] is None
    assert callable(env["policy_generator"])

    in_dim = int(env["env_input_dim"])
    env_in = torch.zeros((2, in_dim), dtype=torch.float32)
    x_next, reward = env["transition_generator"](env_in)

    assert x_next.shape == (2, int(env["state_dim"]))
    assert reward.shape == (2, 1)
    assert torch.isfinite(x_next).all()
    assert torch.isfinite(reward).all()
    assert bool(getattr(env["transition_generator"], "_applies_output_tanh", True)) is False


def test_environment_prior_strict_reference_scm_joint_transition_matches_reference_builder():
    _seed_everything(20260309)
    prior = EnvironmentPrior({})
    h = _manual_sampled_h(
        family="scm",
        state_dim=5,
        obs_dim=3,
        action_dim=2,
        noise_dim=4,
        zero_pad_dim=1,
        num_layers=4,
    )
    h["strict_joint_transition_enabled"] = True
    h["scm_standard_linear_init_enabled"] = False
    h["noise_std"] = 0.0
    h["pre_sample_weights"] = False
    h["prior_mlp_dropout_prob"] = 0.0
    h["block_wise_dropout"] = False
    h["random_feature_rotation"] = False

    seed = 321
    env = prior._sample_environment(h, device="cpu", rng_seed=seed)
    assert bool(env["reference_semantics_enabled"]) is True

    in_dim = int(env["env_input_dim"])
    env_in = torch.linspace(-1.0, 1.0, steps=3 * in_dim, dtype=torch.float32).reshape(3, in_dim)
    state_actual, reward_actual = env["transition_generator"](env_in)

    ref_generator = torch.Generator(device="cpu")
    ref_generator.manual_seed(seed)
    state_ref, reward_ref = _reference_scm_joint_eval(env_in, h, generator=ref_generator)

    assert torch.allclose(state_actual, state_ref, atol=1e-6, rtol=1e-6)
    assert torch.allclose(reward_actual, reward_ref, atol=1e-6, rtol=1e-6)


def test_environment_prior_strict_reference_scm_joint_transition_matches_reference_builder_with_standard_linear_init():
    _seed_everything(20260309)
    prior = EnvironmentPrior({})
    h = _manual_sampled_h(
        family="scm",
        state_dim=5,
        obs_dim=3,
        action_dim=2,
        noise_dim=4,
        zero_pad_dim=1,
        num_layers=4,
    )
    h["strict_joint_transition_enabled"] = True
    h["scm_standard_linear_init_enabled"] = True
    h["noise_std"] = 0.0
    h["pre_sample_weights"] = False
    h["prior_mlp_dropout_prob"] = 0.0
    h["block_wise_dropout"] = False
    h["random_feature_rotation"] = False

    env = prior._sample_environment(h, device="cpu", rng_seed=321)
    in_dim = int(env["env_input_dim"])
    env_in = torch.linspace(-1.0, 1.0, steps=3 * in_dim, dtype=torch.float32).reshape(3, in_dim)
    state_actual, reward_actual = env["transition_generator"](env_in)

    ref_generator = torch.Generator(device="cpu")
    ref_generator.manual_seed(321)
    state_ref, reward_ref = _reference_scm_joint_eval(env_in, h, generator=ref_generator)

    assert torch.allclose(state_actual, state_ref, atol=1e-6, rtol=1e-6)
    assert torch.allclose(reward_actual, reward_ref, atol=1e-6, rtol=1e-6)


def test_environment_prior_prefix_grouped_noise_matches_rowwise_on_homogeneous_prefix_case():
    scale = torch.tensor(
        [
            [0.10, 0.20, 0.30, 0.40],
            [0.50, 0.60, 0.70, 0.80],
            [0.90, 1.00, 1.10, 1.20],
        ],
        dtype=torch.float32,
    )

    _seed_everything(20260309)
    actual = EnvironmentPrior._sample_prefix_grouped_scaled_noise_without_generators(
        scale,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    _seed_everything(20260309)
    expected = torch.zeros_like(scale)
    for bi in range(scale.shape[0]):
        active = torch.nonzero(scale[bi] > 0, as_tuple=False).squeeze(1)
        expected[bi, active] = torch.randn((int(active.numel()),), dtype=torch.float32) * scale[bi, active]

    assert torch.allclose(actual, expected, atol=1e-7, rtol=0.0)


def test_environment_prior_prefix_grouped_noise_plan_matches_runtime_helper():
    scale = torch.tensor(
        [
            [0.10, 0.20, 0.30, 0.00, 0.00],
            [0.40, 0.50, 0.00, 0.00, 0.00],
            [0.60, 0.70, 0.80, 0.90, 0.00],
            [0.00, 0.00, 0.00, 0.00, 0.00],
        ],
        dtype=torch.float32,
    )

    plan = EnvironmentPrior._build_prefix_grouped_scale_plan(
        scale,
        device=torch.device("cpu"),
    )
    assert plan is not None

    _seed_everything(20260309)
    actual = EnvironmentPrior._sample_prefix_grouped_scaled_noise_without_generators(
        scale,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    _seed_everything(20260309)
    expected = EnvironmentPrior._sample_prefix_grouped_scaled_noise_with_plan(
        scale,
        plan,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert torch.allclose(actual, expected, atol=1e-7, rtol=0.0)


def test_environment_prior_strict_reference_scm_hidden_noise_packed_flag_is_noop_with_generators(monkeypatch):
    _seed_everything(20260309)
    h_list = [
        _manual_sampled_h(
            family="scm",
            state_dim=6,
            obs_dim=4,
            action_dim=3,
            noise_dim=2,
            zero_pad_dim=1,
            num_layers=4,
        ),
        _manual_sampled_h(
            family="scm",
            state_dim=8,
            obs_dim=5,
            action_dim=2,
            noise_dim=3,
            zero_pad_dim=2,
            num_layers=5,
        ),
    ]
    for h in h_list:
        h["strict_joint_transition_enabled"] = True
        h["noise_std"] = 0.15
        h["pre_sample_weights"] = False
        h["prior_mlp_dropout_prob"] = 0.0
        h["block_wise_dropout"] = False
        h["random_feature_rotation"] = False
        h["init_std"] = 0.05
        h["prior_mlp_hidden_dim"] = 16

    in_dims = [
        int(h["state_dim"]) + int(h["obs_dim"]) + int(h["action_dim"]) + int(h["noise_dim"]) + int(h["zero_pad_dim"])
        for h in h_list
    ]
    state_dims = [int(h["state_dim"]) for h in h_list]
    in_cap = max(in_dims)
    x = torch.linspace(-0.5, 0.75, steps=len(h_list) * in_cap, dtype=torch.float32).reshape(len(h_list), in_cap)

    def _build(flag: str):
        _seed_everything(20260309)
        monkeypatch.setenv("TICL_POLICY_REFERENCE_SCM_HIDDEN_NOISE_PACKED", flag)
        prior = EnvironmentPrior({})
        return prior._build_reference_scm_joint_transition_padded_batch_fn(
            in_dims,
            state_dims,
            h_list,
            torch.device("cpu"),
            generators=None,
            input_mask=None,
        )

    fn_off = _build("0")
    fn_on = _build("1")

    def _noise_generators():
        out = []
        for seed in (123, 456):
            g = torch.Generator(device="cpu")
            g.manual_seed(seed)
            out.append(g)
        return out

    state_off, reward_off = fn_off(x, generators_for_noise=_noise_generators())
    state_on, reward_on = fn_on(x, generators_for_noise=_noise_generators())

    assert torch.allclose(state_off, state_on, atol=0.0, rtol=0.0)
    assert torch.allclose(reward_off, reward_on, atol=0.0, rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for fused affine semantics check")
def test_environment_prior_strict_reference_scm_hidden_affine_fused_matches_eager_on_cuda(monkeypatch):
    _seed_everything(20260309)
    h_list = [
        _manual_sampled_h(
            family="scm",
            state_dim=16,
            obs_dim=10,
            action_dim=4,
            noise_dim=6,
            zero_pad_dim=2,
            num_layers=4,
        ),
        _manual_sampled_h(
            family="scm",
            state_dim=24,
            obs_dim=18,
            action_dim=6,
            noise_dim=8,
            zero_pad_dim=4,
            num_layers=5,
        ),
    ]
    for h in h_list:
        h["strict_joint_transition_enabled"] = True
        h["noise_std"] = 0.0
        h["pre_sample_weights"] = False
        h["prior_mlp_dropout_prob"] = 0.0
        h["block_wise_dropout"] = False
        h["random_feature_rotation"] = False
        h["init_std"] = 0.05
        h["prior_mlp_hidden_dim"] = 64

    in_dims = [
        int(h["state_dim"]) + int(h["obs_dim"]) + int(h["action_dim"]) + int(h["noise_dim"]) + int(h["zero_pad_dim"])
        for h in h_list
    ]
    state_dims = [int(h["state_dim"]) for h in h_list]
    in_cap = max(in_dims)
    x = torch.linspace(-0.75, 0.85, steps=len(h_list) * in_cap, dtype=torch.float32, device="cuda").reshape(len(h_list), in_cap)

    def _build(flag: str):
        _seed_everything(20260309)
        monkeypatch.setenv("TICL_POLICY_REFERENCE_SCM_HIDDEN_AFFINE_FUSED", flag)
        prior = EnvironmentPrior({})
        return prior._build_reference_scm_joint_transition_padded_batch_fn(
            in_dims,
            state_dims,
            h_list,
            torch.device("cuda"),
            generators=None,
            input_mask=None,
        )

    fn_off = _build("0")
    fn_on = _build("1")

    state_off, reward_off = fn_off(x, generators_for_noise=None)
    state_on, reward_on = fn_on(x, generators_for_noise=None)

    assert torch.allclose(state_off, state_on, atol=1e-6, rtol=1e-6)
    assert torch.allclose(reward_off, reward_on, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for fused update semantics check")
def test_environment_prior_strict_reference_scm_hidden_update_fused_matches_unfused_on_cuda(monkeypatch):
    _seed_everything(20260309)
    h_list = [
        _manual_sampled_h(
            family="scm",
            state_dim=16,
            obs_dim=10,
            action_dim=4,
            noise_dim=6,
            zero_pad_dim=2,
            num_layers=4,
        ),
        _manual_sampled_h(
            family="scm",
            state_dim=24,
            obs_dim=18,
            action_dim=6,
            noise_dim=8,
            zero_pad_dim=4,
            num_layers=5,
        ),
    ]
    for h in h_list:
        h["strict_joint_transition_enabled"] = True
        h["noise_std"] = 0.05
        h["pre_sample_weights"] = False
        h["prior_mlp_dropout_prob"] = 0.0
        h["block_wise_dropout"] = False
        h["random_feature_rotation"] = False
        h["init_std"] = 0.05
        h["prior_mlp_hidden_dim"] = 64

    in_dims = [
        int(h["state_dim"]) + int(h["obs_dim"]) + int(h["action_dim"]) + int(h["noise_dim"]) + int(h["zero_pad_dim"])
        for h in h_list
    ]
    state_dims = [int(h["state_dim"]) for h in h_list]
    in_cap = max(in_dims)
    x = torch.linspace(-0.75, 0.85, steps=len(h_list) * in_cap, dtype=torch.float32, device="cuda").reshape(len(h_list), in_cap)

    def _build(flag: str):
        _seed_everything(20260309)
        monkeypatch.setenv("TICL_POLICY_REFERENCE_SCM_HIDDEN_AFFINE_FUSED", "1")
        monkeypatch.setenv("TICL_POLICY_REFERENCE_SCM_HIDDEN_UPDATE_FUSED", flag)
        prior = EnvironmentPrior({})
        return prior._build_reference_scm_joint_transition_padded_batch_fn(
            in_dims,
            state_dims,
            h_list,
            torch.device("cuda"),
            generators=None,
            input_mask=None,
        )

    fn_off = _build("0")
    fn_on = _build("1")

    torch.manual_seed(12345)
    state_off, reward_off = fn_off(x, generators_for_noise=None)
    torch.manual_seed(12345)
    state_on, reward_on = fn_on(x, generators_for_noise=None)

    assert torch.allclose(state_off, state_on, atol=1e-6, rtol=1e-6)
    assert torch.allclose(reward_off, reward_on, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for grad-fused update semantics check")
def test_environment_prior_strict_reference_scm_hidden_update_grad_fused_matches_unfused_on_cuda(monkeypatch):
    _seed_everything(20260312)
    h_list = [
        _manual_sampled_h(
            family="scm",
            state_dim=12,
            obs_dim=8,
            action_dim=4,
            noise_dim=5,
            zero_pad_dim=2,
            num_layers=4,
        ),
        _manual_sampled_h(
            family="scm",
            state_dim=10,
            obs_dim=7,
            action_dim=3,
            noise_dim=4,
            zero_pad_dim=3,
            num_layers=5,
        ),
    ]
    h_list[0]["prior_mlp_activations"] = "relu"
    h_list[1]["prior_mlp_activations"] = "identity"
    for h in h_list:
        h["strict_joint_transition_enabled"] = True
        h["noise_std"] = 0.0
        h["pre_sample_weights"] = False
        h["prior_mlp_dropout_prob"] = 0.0
        h["block_wise_dropout"] = False
        h["random_feature_rotation"] = False
        h["init_std"] = 0.05
        h["prior_mlp_hidden_dim"] = 64

    in_dims = [
        int(h["state_dim"]) + int(h["obs_dim"]) + int(h["action_dim"]) + int(h["noise_dim"]) + int(h["zero_pad_dim"])
        for h in h_list
    ]
    state_dims = [int(h["state_dim"]) for h in h_list]
    in_cap = max(in_dims)
    x_base = torch.linspace(-0.5, 0.7, steps=len(h_list) * in_cap, dtype=torch.float32, device="cuda").reshape(len(h_list), in_cap)

    def _build(flag: str):
        _seed_everything(20260312)
        monkeypatch.setenv("TICL_POLICY_REFERENCE_SCM_HIDDEN_AFFINE_FUSED", "1")
        monkeypatch.setenv("TICL_POLICY_REFERENCE_SCM_HIDDEN_UPDATE_FUSED", "1")
        monkeypatch.setenv("TICL_POLICY_REFERENCE_SCM_HIDDEN_UPDATE_SAMPLE_FUSED", "1")
        monkeypatch.setenv("TICL_POLICY_REFERENCE_SCM_HIDDEN_UPDATE_GRAD_FUSED", flag)
        prior = EnvironmentPrior({})
        return prior._build_reference_scm_joint_transition_padded_batch_fn(
            in_dims,
            state_dims,
            h_list,
            torch.device("cuda"),
            generators=None,
            input_mask=None,
        )

    fn_off = _build("0")
    fn_on = _build("1")

    x_off = x_base.clone().requires_grad_(True)
    state_off, reward_off = fn_off(x_off, generators_for_noise=None)
    loss_off = state_off.sum() + reward_off.sum()
    grad_off = torch.autograd.grad(loss_off, x_off)[0]

    x_on = x_base.clone().requires_grad_(True)
    state_on, reward_on = fn_on(x_on, generators_for_noise=None)
    loss_on = state_on.sum() + reward_on.sum()
    grad_on = torch.autograd.grad(loss_on, x_on)[0]

    assert torch.allclose(state_off, state_on, atol=1e-5, rtol=1e-5)
    assert torch.allclose(reward_off, reward_on, atol=1e-5, rtol=1e-5)
    assert torch.allclose(grad_off, grad_on, atol=2e-5, rtol=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for rowwise noise plan semantics check")
def test_environment_prior_rowwise_scaled_noise_with_plan_matches_unplanned_on_cuda():
    device = "cuda"
    dtype = torch.float32
    prior = EnvironmentPrior(get_prior_config()["prior"]["environment"])
    scale = torch.zeros((4, 9), device=device, dtype=dtype)
    scale[0, :3] = torch.tensor([0.5, 1.0, 0.25], device=device, dtype=dtype)
    scale[1, 4:7] = torch.tensor([0.2, 0.4, 0.6], device=device, dtype=dtype)
    scale[2, :5] = 0.3
    scale[3, :2] = torch.tensor([1.0, 2.0], device=device, dtype=dtype)

    plan = prior._build_rowwise_scaled_noise_plan(scale, device=device)

    seeds = [1701, 1702, 1703, 1704]
    generators_a = []
    generators_b = []
    for seed in seeds:
        g_a = torch.Generator(device=device)
        g_a.manual_seed(seed)
        generators_a.append(g_a)
        g_b = torch.Generator(device=device)
        g_b.manual_seed(seed)
        generators_b.append(g_b)

    eps_ref = prior._sample_rowwise_scaled_noise(
        scale,
        generators=generators_a,
        device=device,
        dtype=dtype,
    )
    eps_plan = prior._sample_rowwise_scaled_noise_with_plan(
        scale,
        plan,
        generators=generators_b,
        device=device,
        dtype=dtype,
    )

    assert eps_ref is not None
    assert eps_plan is not None
    assert torch.allclose(eps_plan, eps_ref, atol=0.0, rtol=0.0)

    tail_a = torch.stack(
        [torch.randn((11,), device=device, dtype=dtype, generator=g) for g in generators_a],
        dim=0,
    )
    tail_b = torch.stack(
        [torch.randn((11,), device=device, dtype=dtype, generator=g) for g in generators_b],
        dim=0,
    )
    assert torch.allclose(tail_a, tail_b, atol=0.0, rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for multilayer fused semantics check")
def test_environment_prior_strict_reference_scm_hidden_multilayer_fused_matches_unfused_on_cuda(monkeypatch):
    _seed_everything(20260309)
    h_list = [
        _manual_sampled_h(
            family="scm",
            state_dim=16,
            obs_dim=10,
            action_dim=4,
            noise_dim=6,
            zero_pad_dim=2,
            num_layers=4,
        ),
        _manual_sampled_h(
            family="scm",
            state_dim=24,
            obs_dim=18,
            action_dim=6,
            noise_dim=8,
            zero_pad_dim=4,
            num_layers=5,
        ),
    ]
    for h in h_list:
        h["strict_joint_transition_enabled"] = True
        h["noise_std"] = 0.05
        h["pre_sample_weights"] = False
        h["prior_mlp_dropout_prob"] = 0.0
        h["block_wise_dropout"] = False
        h["random_feature_rotation"] = False
        h["init_std"] = 0.05
        h["prior_mlp_hidden_dim"] = 64

    in_dims = [
        int(h["state_dim"]) + int(h["obs_dim"]) + int(h["action_dim"]) + int(h["noise_dim"]) + int(h["zero_pad_dim"])
        for h in h_list
    ]
    state_dims = [int(h["state_dim"]) for h in h_list]
    in_cap = max(in_dims)
    x = torch.linspace(-0.75, 0.85, steps=len(h_list) * in_cap, dtype=torch.float32, device="cuda").reshape(len(h_list), in_cap)

    def _build(flag: str):
        _seed_everything(20260309)
        monkeypatch.setenv("TICL_POLICY_REFERENCE_SCM_HIDDEN_AFFINE_FUSED", "1")
        monkeypatch.setenv("TICL_POLICY_REFERENCE_SCM_HIDDEN_UPDATE_FUSED", "1")
        monkeypatch.setenv("TICL_POLICY_REFERENCE_SCM_HIDDEN_UPDATE_SAMPLE_FUSED", "1")
        monkeypatch.setenv("TICL_POLICY_REFERENCE_SCM_HIDDEN_MULTILAYER_FUSED", flag)
        prior = EnvironmentPrior({})
        return prior._build_reference_scm_joint_transition_padded_batch_fn(
            in_dims,
            state_dims,
            h_list,
            torch.device("cuda"),
            generators=None,
            input_mask=None,
        )

    fn_off = _build("0")
    fn_on = _build("1")

    torch.manual_seed(12345)
    state_off, reward_off = fn_off(x, generators_for_noise=None)
    torch.manual_seed(12345)
    state_on, reward_on = fn_on(x, generators_for_noise=None)

    assert torch.allclose(state_off, state_on, atol=1e-6, rtol=1e-6)
    assert torch.allclose(reward_off, reward_on, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_environment_prior_strict_reference_scm_full_multilayer_fused_matches_unfused_on_cuda(monkeypatch):
    _seed_everything(20260309)
    h_list = [
        _manual_sampled_h(
            family="scm",
            state_dim=18,
            obs_dim=12,
            action_dim=5,
            noise_dim=7,
            zero_pad_dim=3,
            num_layers=4,
        ),
        _manual_sampled_h(
            family="scm",
            state_dim=24,
            obs_dim=18,
            action_dim=6,
            noise_dim=8,
            zero_pad_dim=4,
            num_layers=5,
        ),
    ]
    for h in h_list:
        h["strict_joint_transition_enabled"] = True
        h["noise_std"] = 0.05
        h["pre_sample_weights"] = False
        h["prior_mlp_dropout_prob"] = 0.0
        h["block_wise_dropout"] = False
        h["random_feature_rotation"] = False
        h["init_std"] = 0.05
        h["prior_mlp_hidden_dim"] = 64

    in_dims = [
        int(h["state_dim"]) + int(h["obs_dim"]) + int(h["action_dim"]) + int(h["noise_dim"]) + int(h["zero_pad_dim"])
        for h in h_list
    ]
    state_dims = [int(h["state_dim"]) for h in h_list]
    packed_input_cap = max(in_dims)
    x = torch.linspace(
        -0.75,
        0.85,
        steps=len(h_list) * packed_input_cap,
        dtype=torch.float32,
        device="cuda",
    ).reshape(len(h_list), packed_input_cap)

    def _build(flag: str):
        _seed_everything(20260309)
        monkeypatch.setenv("TICL_POLICY_REFERENCE_SCM_HIDDEN_AFFINE_FUSED", "1")
        monkeypatch.setenv("TICL_POLICY_REFERENCE_SCM_HIDDEN_UPDATE_FUSED", "1")
        monkeypatch.setenv("TICL_POLICY_REFERENCE_SCM_HIDDEN_UPDATE_SAMPLE_FUSED", "1")
        monkeypatch.setenv("TICL_POLICY_REFERENCE_SCM_HIDDEN_MULTILAYER_FUSED", "1")
        monkeypatch.setenv("TICL_POLICY_REFERENCE_SCM_FULL_MULTILAYER_FUSED", flag)
        prior = EnvironmentPrior({})
        return prior._build_reference_scm_joint_transition_padded_batch_fn(
            in_dims,
            state_dims,
            h_list,
            torch.device("cuda"),
            generators=None,
            input_mask=None,
        )

    build_generators_off = [torch.Generator(device="cuda").manual_seed(100 + i) for i in range(len(h_list))]
    build_generators_on = [torch.Generator(device="cuda").manual_seed(100 + i) for i in range(len(h_list))]
    _seed_everything(20260310)
    monkeypatch.setenv("TICL_POLICY_REFERENCE_SCM_HIDDEN_AFFINE_FUSED", "1")
    monkeypatch.setenv("TICL_POLICY_REFERENCE_SCM_HIDDEN_UPDATE_FUSED", "1")
    monkeypatch.setenv("TICL_POLICY_REFERENCE_SCM_HIDDEN_UPDATE_SAMPLE_FUSED", "1")
    monkeypatch.setenv("TICL_POLICY_REFERENCE_SCM_HIDDEN_MULTILAYER_FUSED", "1")
    monkeypatch.setenv("TICL_POLICY_REFERENCE_SCM_FULL_MULTILAYER_FUSED", "1")
    monkeypatch.setenv("TICL_POLICY_REFERENCE_SCM_PARTITION_MAX_BYTES", "0")
    prior_off = EnvironmentPrior({})
    fn_off = prior_off._build_reference_scm_joint_transition_padded_batch_fn(
        in_dims,
        state_dims,
        h_list,
        torch.device("cuda"),
        generators=build_generators_off,
        input_mask=None,
    )
    _seed_everything(20260310)
    monkeypatch.setenv("TICL_POLICY_REFERENCE_SCM_PARTITION_MAX_BYTES", "1")
    prior_on = EnvironmentPrior({})
    fn_on = prior_on._build_reference_scm_joint_transition_padded_batch_fn(
        in_dims,
        state_dims,
        h_list,
        torch.device("cuda"),
        generators=build_generators_on,
        input_mask=None,
    )
    noise_generators_off = [torch.Generator(device="cuda").manual_seed(200 + i) for i in range(len(h_list))]
    noise_generators_on = [torch.Generator(device="cuda").manual_seed(200 + i) for i in range(len(h_list))]
    state_off, reward_off = fn_off(x, generators_for_noise=noise_generators_off, x_input_is_packed=True)
    state_on, reward_on = fn_on(x, generators_for_noise=noise_generators_on, x_input_is_packed=True)

    assert torch.allclose(state_off, state_on, atol=1e-6, rtol=1e-6)
    assert torch.allclose(reward_off, reward_on, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_environment_prior_strict_reference_scm_full_multilayer_fused_flag_is_noop_with_generators(monkeypatch):
    _seed_everything(20260309)
    h_list = [
        _manual_sampled_h(
            family="scm",
            state_dim=18,
            obs_dim=12,
            action_dim=5,
            noise_dim=7,
            zero_pad_dim=3,
            num_layers=4,
        ),
        _manual_sampled_h(
            family="scm",
            state_dim=24,
            obs_dim=18,
            action_dim=6,
            noise_dim=8,
            zero_pad_dim=4,
            num_layers=5,
        ),
    ]
    for h in h_list:
        h["strict_joint_transition_enabled"] = True
        h["noise_std"] = 0.05
        h["pre_sample_weights"] = False
        h["prior_mlp_dropout_prob"] = 0.0
        h["block_wise_dropout"] = False
        h["random_feature_rotation"] = False
        h["init_std"] = 0.05
        h["prior_mlp_hidden_dim"] = 64

    in_dims = [
        int(h["state_dim"]) + int(h["obs_dim"]) + int(h["action_dim"]) + int(h["noise_dim"]) + int(h["zero_pad_dim"])
        for h in h_list
    ]
    state_dims = [int(h["state_dim"]) for h in h_list]
    packed_input_cap = max(in_dims)
    x = torch.linspace(
        -0.75,
        0.85,
        steps=len(h_list) * packed_input_cap,
        dtype=torch.float32,
        device="cuda",
    ).reshape(len(h_list), packed_input_cap)

    def _build(flag: str):
        _seed_everything(20260309)
        monkeypatch.setenv("TICL_POLICY_REFERENCE_SCM_HIDDEN_AFFINE_FUSED", "1")
        monkeypatch.setenv("TICL_POLICY_REFERENCE_SCM_HIDDEN_UPDATE_FUSED", "1")
        monkeypatch.setenv("TICL_POLICY_REFERENCE_SCM_HIDDEN_UPDATE_SAMPLE_FUSED", "1")
        monkeypatch.setenv("TICL_POLICY_REFERENCE_SCM_HIDDEN_MULTILAYER_FUSED", "1")
        monkeypatch.setenv("TICL_POLICY_REFERENCE_SCM_FULL_MULTILAYER_FUSED", flag)
        prior = EnvironmentPrior({})
        return prior._build_reference_scm_joint_transition_padded_batch_fn(
            in_dims,
            state_dims,
            h_list,
            torch.device("cuda"),
            generators=None,
            input_mask=None,
        )

    fn_off = _build("0")
    fn_on = _build("1")
    generators = [torch.Generator(device="cuda").manual_seed(11 + i) for i in range(len(h_list))]

    state_off, reward_off = fn_off(x, generators_for_noise=generators, x_input_is_packed=True)
    generators = [torch.Generator(device="cuda").manual_seed(11 + i) for i in range(len(h_list))]
    state_on, reward_on = fn_on(x, generators_for_noise=generators, x_input_is_packed=True)

    assert torch.equal(state_off, state_on)
    assert torch.equal(reward_off, reward_on)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_environment_prior_strict_reference_scm_partitioned_builder_matches_unpartitioned_on_cuda(monkeypatch):
    _seed_everything(20260310)
    h_list = []
    specs = [
        dict(state_dim=18, obs_dim=12, action_dim=5, noise_dim=7, zero_pad_dim=3, num_layers=4, prior_mlp_hidden_dim=64),
        dict(state_dim=24, obs_dim=18, action_dim=6, noise_dim=8, zero_pad_dim=4, num_layers=5, prior_mlp_hidden_dim=96),
        dict(state_dim=32, obs_dim=20, action_dim=8, noise_dim=10, zero_pad_dim=2, num_layers=6, prior_mlp_hidden_dim=80),
        dict(state_dim=16, obs_dim=10, action_dim=4, noise_dim=6, zero_pad_dim=5, num_layers=3, prior_mlp_hidden_dim=48),
    ]
    for spec in specs:
        hidden_dim = int(spec.pop("prior_mlp_hidden_dim"))
        h = _manual_sampled_h(family="scm", **spec)
        h["strict_joint_transition_enabled"] = True
        h["noise_std"] = 0.05
        h["pre_sample_weights"] = False
        h["prior_mlp_dropout_prob"] = 0.0
        h["block_wise_dropout"] = False
        h["random_feature_rotation"] = False
        h["init_std"] = 0.05
        h["prior_mlp_hidden_dim"] = hidden_dim
        h_list.append(h)

    in_dims = [
        int(h["state_dim"]) + int(h["obs_dim"]) + int(h["action_dim"]) + int(h["noise_dim"]) + int(h["zero_pad_dim"])
        for h in h_list
    ]
    state_dims = [int(h["state_dim"]) for h in h_list]
    packed_input_cap = max(in_dims)
    x = torch.linspace(
        -0.75,
        0.85,
        steps=len(h_list) * packed_input_cap,
        dtype=torch.float32,
        device="cuda",
    ).reshape(len(h_list), packed_input_cap)

    _seed_everything(20260310)
    monkeypatch.setenv("TICL_POLICY_REFERENCE_SCM_HIDDEN_AFFINE_FUSED", "1")
    monkeypatch.setenv("TICL_POLICY_REFERENCE_SCM_HIDDEN_UPDATE_FUSED", "1")
    monkeypatch.setenv("TICL_POLICY_REFERENCE_SCM_HIDDEN_UPDATE_SAMPLE_FUSED", "1")
    monkeypatch.setenv("TICL_POLICY_REFERENCE_SCM_HIDDEN_MULTILAYER_FUSED", "1")
    monkeypatch.setenv("TICL_POLICY_REFERENCE_SCM_FULL_MULTILAYER_FUSED", "1")
    monkeypatch.setenv("TICL_POLICY_REFERENCE_SCM_PARTITION_MAX_BYTES", "0")
    prior_off = EnvironmentPrior({})
    build_generators_off = [torch.Generator(device="cuda").manual_seed(100 + i) for i in range(len(h_list))]
    fn_off = prior_off._build_reference_scm_joint_transition_padded_batch_fn(
        in_dims,
        state_dims,
        h_list,
        torch.device("cuda"),
        generators=build_generators_off,
        input_mask=None,
    )

    _seed_everything(20260310)
    monkeypatch.setenv("TICL_POLICY_REFERENCE_SCM_PARTITION_MAX_BYTES", "1")
    prior_on = EnvironmentPrior({})
    build_generators_on = [torch.Generator(device="cuda").manual_seed(100 + i) for i in range(len(h_list))]
    fn_on = prior_on._build_reference_scm_joint_transition_padded_batch_fn(
        in_dims,
        state_dims,
        h_list,
        torch.device("cuda"),
        generators=build_generators_on,
        input_mask=None,
    )

    noise_generators_off = [torch.Generator(device="cuda").manual_seed(200 + i) for i in range(len(h_list))]
    noise_generators_on = [torch.Generator(device="cuda").manual_seed(200 + i) for i in range(len(h_list))]
    state_off, reward_off = fn_off(x, generators_for_noise=noise_generators_off, x_input_is_packed=True)
    state_on, reward_on = fn_on(x, generators_for_noise=noise_generators_on, x_input_is_packed=True)

    assert torch.allclose(state_off, state_on, atol=1e-6, rtol=1e-6)
    assert torch.allclose(reward_off, reward_on, atol=1e-6, rtol=1e-6)


def test_environment_prior_strict_reference_scm_family_coarse_batch_matches_reference_builder():
    _seed_everything(20260309)
    prior = EnvironmentPrior({})
    h_list = [
        _manual_sampled_h(
            family="scm",
            state_dim=5,
            obs_dim=3,
            action_dim=2,
            noise_dim=4,
            zero_pad_dim=1,
            num_layers=4,
        ),
        _manual_sampled_h(
            family="scm",
            state_dim=7,
            obs_dim=4,
            action_dim=3,
            noise_dim=2,
            zero_pad_dim=0,
            num_layers=3,
        ),
        _manual_sampled_h(
            family="scm",
            state_dim=6,
            obs_dim=5,
            action_dim=2,
            noise_dim=3,
            zero_pad_dim=2,
            num_layers=5,
        ),
    ]
    seeds = [321, 654, 987]
    for idx, h in enumerate(h_list):
        h["strict_joint_transition_enabled"] = True
        h["noise_std"] = 0.0
        h["pre_sample_weights"] = False
        h["prior_mlp_dropout_prob"] = 0.0
        h["block_wise_dropout"] = False
        h["random_feature_rotation"] = False
        h["init_std"] = 0.03 + (0.01 * idx)
        h["prior_mlp_hidden_dim"] = 10 + (2 * idx)

    env_batch = prior._sample_environment_family_coarse_batch(
        h_list=h_list,
        device="cpu",
        rng_seeds=seeds,
        build_x_generator=False,
        build_y_generator=False,
        build_policy_generator=False,
    )
    assert bool(getattr(env_batch["transition_generator"], "_reference_scm_vectorized", False))

    env_in = torch.linspace(
        -1.25,
        1.1,
        steps=len(h_list) * int(env_batch["env_input_dim"]),
        dtype=torch.float32,
    ).reshape(len(h_list), int(env_batch["env_input_dim"]))
    state_actual, reward_actual = env_batch["transition_generator"](env_in)

    for bi, (seed, h) in enumerate(zip(seeds, h_list)):
        active_idx = _coarse_batch_reference_active_index(env_batch, bi)
        env_in_compact = env_in[bi: bi + 1].index_select(1, active_idx)
        ref_generator = torch.Generator(device="cpu")
        ref_generator.manual_seed(int(seed))
        state_ref, reward_ref = _reference_scm_joint_eval(env_in_compact, h, generator=ref_generator)
        state_dim = int(h["state_dim"])
        assert torch.allclose(state_actual[bi: bi + 1, :state_dim], state_ref, atol=1e-6, rtol=1e-6)
        assert torch.allclose(reward_actual[bi: bi + 1], reward_ref, atol=1e-6, rtol=1e-6)
        if state_actual.shape[1] > state_dim:
            assert torch.allclose(state_actual[bi, state_dim:], torch.zeros_like(state_actual[bi, state_dim:]))


def test_environment_prior_strict_reference_scm_family_coarse_batch_packed_input_matches_dense():
    _seed_everything(20260309)
    prior = EnvironmentPrior({})
    h_list = [
        _manual_sampled_h(
            family="scm",
            state_dim=5,
            obs_dim=3,
            action_dim=2,
            noise_dim=4,
            zero_pad_dim=2,
            num_layers=4,
        ),
        _manual_sampled_h(
            family="scm",
            state_dim=7,
            obs_dim=4,
            action_dim=3,
            noise_dim=2,
            zero_pad_dim=1,
            num_layers=3,
        ),
    ]
    seeds = [321, 654]
    for idx, h in enumerate(h_list):
        h["strict_joint_transition_enabled"] = True
        h["noise_std"] = 0.0
        h["pre_sample_weights"] = False
        h["prior_mlp_dropout_prob"] = 0.0
        h["block_wise_dropout"] = False
        h["random_feature_rotation"] = False
        h["init_std"] = 0.03 + (0.01 * idx)
        h["prior_mlp_hidden_dim"] = 10 + (2 * idx)

    env_batch = prior._sample_environment_family_coarse_batch(
        h_list=h_list,
        device="cpu",
        rng_seeds=seeds,
        build_x_generator=False,
        build_y_generator=False,
        build_policy_generator=False,
    )
    transition = env_batch["transition_generator"]
    assert bool(getattr(transition, "_prefers_packed_env_input", False))

    env_in = torch.linspace(
        -1.25,
        1.1,
        steps=len(h_list) * int(env_batch["env_input_dim"]),
        dtype=torch.float32,
    ).reshape(len(h_list), int(env_batch["env_input_dim"]))
    packed_cap = int(getattr(transition, "_packed_input_cap"))
    packed_in = torch.zeros((len(h_list), packed_cap), dtype=torch.float32)
    for bi in range(len(h_list)):
        active_idx = _coarse_batch_reference_active_index(env_batch, bi)
        take = min(int(active_idx.numel()), packed_cap)
        if take > 0:
            packed_in[bi, :take] = env_in[bi].index_select(0, active_idx[:take])

    state_dense, reward_dense = transition(env_in)
    state_packed, reward_packed = transition(packed_in, x_input_is_packed=True)

    assert torch.allclose(state_dense, state_packed, atol=1e-6, rtol=1e-6)
    assert torch.allclose(reward_dense, reward_packed, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for exact SCM layer compile semantics")
def test_environment_prior_strict_reference_scm_layer_compile_matches_eager_and_logs(capsys, monkeypatch):
    _seed_everything(20260309)
    monkeypatch.setenv("TICL_POLICY_REFERENCE_SCM_LAYER_COMPILE", "0")
    prior_eager = EnvironmentPrior({})

    monkeypatch.setenv("TICL_POLICY_REFERENCE_SCM_LAYER_COMPILE", "1")
    monkeypatch.setenv("TICL_POLICY_REFERENCE_SCM_LAYER_COMPILE_LOG", "1")
    monkeypatch.setenv("TICL_POLICY_REFERENCE_SCM_LAYER_COMPILE_BACKEND", "inductor")
    monkeypatch.setenv("TICL_POLICY_REFERENCE_SCM_LAYER_COMPILE_MODE", "reduce-overhead")
    monkeypatch.setenv("TICL_POLICY_REFERENCE_SCM_LAYER_COMPILE_FULLGRAPH", "0")
    monkeypatch.setenv("TICL_POLICY_REFERENCE_SCM_LAYER_COMPILE_DYNAMIC", "0")
    prior_compiled = EnvironmentPrior({})

    h_list = [
        _manual_sampled_h(
            family="scm",
            state_dim=5,
            obs_dim=3,
            action_dim=2,
            noise_dim=4,
            zero_pad_dim=1,
            num_layers=4,
        ),
        _manual_sampled_h(
            family="scm",
            state_dim=7,
            obs_dim=4,
            action_dim=3,
            noise_dim=2,
            zero_pad_dim=0,
            num_layers=3,
        ),
    ]
    seeds = [321, 654]
    for idx, h in enumerate(h_list):
        h["strict_joint_transition_enabled"] = True
        h["noise_std"] = 0.0
        h["pre_sample_weights"] = False
        h["prior_mlp_dropout_prob"] = 0.0
        h["block_wise_dropout"] = False
        h["random_feature_rotation"] = False
        h["prior_mlp_hidden_dim"] = 12 + (2 * idx)

    env_eager = prior_eager._sample_environment_family_coarse_batch(
        h_list=h_list,
        device="cuda",
        rng_seeds=seeds,
        build_x_generator=False,
        build_y_generator=False,
        build_policy_generator=False,
    )
    env_compiled = prior_compiled._sample_environment_family_coarse_batch(
        h_list=h_list,
        device="cuda",
        rng_seeds=seeds,
        build_x_generator=False,
        build_y_generator=False,
        build_policy_generator=False,
    )
    in_dim = int(env_eager["env_input_dim"])
    env_in = torch.linspace(
        -1.0,
        1.0,
        steps=len(h_list) * in_dim,
        device="cuda",
        dtype=torch.float32,
    ).reshape(len(h_list), in_dim)
    state_eager, reward_eager = env_eager["transition_generator"](env_in)
    state_compiled, reward_compiled = env_compiled["transition_generator"](env_in)

    assert torch.allclose(state_eager, state_compiled, atol=1e-6, rtol=1e-6)
    assert torch.allclose(reward_eager, reward_compiled, atol=1e-6, rtol=1e-6)

    h_list_2 = [
        _manual_sampled_h(
            family="scm",
            state_dim=6,
            obs_dim=4,
            action_dim=2,
            noise_dim=3,
            zero_pad_dim=1,
            num_layers=5,
        ),
        _manual_sampled_h(
            family="scm",
            state_dim=8,
            obs_dim=5,
            action_dim=2,
            noise_dim=3,
            zero_pad_dim=1,
            num_layers=4,
        ),
        _manual_sampled_h(
            family="scm",
            state_dim=9,
            obs_dim=5,
            action_dim=3,
            noise_dim=2,
            zero_pad_dim=1,
            num_layers=4,
        ),
    ]
    for idx, h in enumerate(h_list_2):
        h["strict_joint_transition_enabled"] = True
        h["noise_std"] = 0.0
        h["pre_sample_weights"] = False
        h["prior_mlp_dropout_prob"] = 0.0
        h["block_wise_dropout"] = False
        h["random_feature_rotation"] = False
        h["prior_mlp_hidden_dim"] = 16 + idx
    env_compiled_2 = prior_compiled._sample_environment_family_coarse_batch(
        h_list=h_list_2,
        device="cuda",
        rng_seeds=[777, 888, 999],
        build_x_generator=False,
        build_y_generator=False,
        build_policy_generator=False,
    )
    env_in_2 = torch.linspace(
        -1.25,
        1.25,
        steps=len(h_list_2) * int(env_compiled_2["env_input_dim"]),
        device="cuda",
        dtype=torch.float32,
    ).reshape(len(h_list_2), int(env_compiled_2["env_input_dim"]))
    env_compiled_2["transition_generator"](env_in_2)

    logs = capsys.readouterr().out
    assert "[envgen-compile] kind=reference_scm_layer" in logs
    assert "[envgen-compile-done] kind=reference_scm_layer" in logs
    assert "[envgen-compile-recompile] kind=reference_scm_layer" in logs
    assert "[envgen-compile-warn] kind=reference_scm_layer" in logs


def test_environment_prior_strict_reference_gp_joint_transition_matches_exact_first_call():
    _seed_everything(20260309)
    prior = EnvironmentPrior({})
    h = _manual_sampled_h(
        family="gp",
        state_dim=4,
        obs_dim=2,
        action_dim=2,
        noise_dim=3,
        zero_pad_dim=1,
        gp_rff_features=32,
    )
    h["strict_joint_transition_enabled"] = True
    h["reference_gp_forward_mode"] = "exact"
    h["outputscale"] = 1.3
    h["lengthscale"] = 0.7
    h["noise"] = 0.0

    seed = 777
    env = prior._sample_environment(h, device="cpu", rng_seed=seed)
    env_in = torch.linspace(-0.8, 0.9, steps=3 * int(env["env_input_dim"]), dtype=torch.float32).reshape(3, int(env["env_input_dim"]))
    state_actual, reward_actual = env["transition_generator"](env_in)
    state_ref, reward_ref = _reference_gp_first_call_eval(env_in, h, seed=seed)

    assert torch.allclose(state_actual, state_ref, atol=1e-6, rtol=1e-6)
    assert torch.allclose(reward_actual, reward_ref, atol=1e-6, rtol=1e-6)


def test_environment_prior_strict_reference_gp_family_coarse_batch_matches_reference_builder():
    _seed_everything(20260309)
    prior = EnvironmentPrior({})
    h_list = [
        _manual_sampled_h(
            family="gp",
            state_dim=4,
            obs_dim=2,
            action_dim=2,
            noise_dim=3,
            zero_pad_dim=1,
            gp_rff_features=32,
        ),
        _manual_sampled_h(
            family="gp",
            state_dim=6,
            obs_dim=3,
            action_dim=2,
            noise_dim=4,
            zero_pad_dim=0,
            gp_rff_features=64,
        ),
    ]
    seeds = [777, 888]
    for idx, h in enumerate(h_list):
        h["strict_joint_transition_enabled"] = True
        h["reference_gp_forward_mode"] = "exact"
        h["outputscale"] = 1.1 + (0.2 * idx)
        h["lengthscale"] = 0.7 + (0.1 * idx)
        h["noise"] = 0.0

    env_batch = prior._sample_environment_family_coarse_batch(
        h_list=h_list,
        device="cpu",
        rng_seeds=seeds,
        build_x_generator=False,
        build_y_generator=False,
        build_policy_generator=False,
    )
    assert bool(getattr(env_batch["transition_generator"], "_reference_gp_vectorized", False))
    env_in = torch.linspace(
        -0.8,
        0.9,
        steps=len(h_list) * int(env_batch["env_input_dim"]),
        dtype=torch.float32,
    ).reshape(len(h_list), int(env_batch["env_input_dim"]))
    state_actual, reward_actual = env_batch["transition_generator"](env_in)

    for bi, (seed, h) in enumerate(zip(seeds, h_list)):
        active_idx = _coarse_batch_reference_active_index(env_batch, bi)
        env_in_compact = env_in[bi: bi + 1].index_select(1, active_idx)
        state_ref, reward_ref = _reference_gp_first_call_eval(env_in_compact, h, seed=seed)
        state_dim = int(h["state_dim"])
        assert torch.allclose(state_actual[bi: bi + 1, :state_dim], state_ref, atol=1e-6, rtol=1e-6)
        assert torch.allclose(reward_actual[bi: bi + 1], reward_ref, atol=1e-6, rtol=1e-6)
        if state_actual.shape[1] > state_dim:
            assert torch.allclose(state_actual[bi, state_dim:], torch.zeros_like(state_actual[bi, state_dim:]))


def test_environment_prior_strict_reference_gp_family_coarse_batch_repeats_identical_query_without_noise():
    _seed_everything(20260309)
    prior = EnvironmentPrior({})
    h_list = [
        _manual_sampled_h(
            family="gp",
            state_dim=5,
            obs_dim=3,
            action_dim=2,
            noise_dim=4,
            zero_pad_dim=1,
            gp_rff_features=32,
        ),
        _manual_sampled_h(
            family="gp",
            state_dim=6,
            obs_dim=4,
            action_dim=2,
            noise_dim=2,
            zero_pad_dim=0,
            gp_rff_features=64,
        ),
    ]
    seeds = [1701, 1702]
    for h in h_list:
        h["strict_joint_transition_enabled"] = True
        h["reference_gp_forward_mode"] = "exact"
        h["noise"] = 0.0

    env_batch = prior._sample_environment_family_coarse_batch(
        h_list=h_list,
        device="cpu",
        rng_seeds=seeds,
        build_x_generator=False,
        build_y_generator=False,
        build_policy_generator=False,
    )
    env_in = torch.linspace(
        -0.4,
        0.6,
        steps=len(h_list) * int(env_batch["env_input_dim"]),
        dtype=torch.float32,
    ).reshape(len(h_list), int(env_batch["env_input_dim"]))

    state_first, reward_first = env_batch["transition_generator"](env_in)
    state_second, reward_second = env_batch["transition_generator"](env_in)

    assert torch.allclose(state_first, state_second, atol=1e-6, rtol=1e-6)
    assert torch.allclose(reward_first, reward_second, atol=1e-6, rtol=1e-6)


def test_environment_prior_strict_reference_gp_family_coarse_batch_multistep_matches_slow_builder():
    _seed_everything(20260309)
    prior = EnvironmentPrior({})
    h_list = []
    seeds = []
    for i in range(4):
        h = _manual_sampled_h(
            family="gp",
            state_dim=4 + i,
            obs_dim=2 + (i % 2),
            action_dim=2,
            noise_dim=3 + (i % 2),
            zero_pad_dim=i % 2,
            gp_rff_features=32,
        )
        h["strict_joint_transition_enabled"] = True
        h["reference_gp_forward_mode"] = "exact"
        h["lengthscale"] = 0.7 + (0.1 * i)
        h["outputscale"] = 1.0 + (0.1 * i)
        h["noise"] = 0.0
        h_list.append(h)
        seeds.append(300 + i)

    env_batch = prior._sample_environment_family_coarse_batch(
        h_list=h_list,
        device="cpu",
        rng_seeds=seeds,
        build_x_generator=False,
        build_y_generator=False,
        build_policy_generator=False,
    )
    fast_fn = env_batch["transition_generator"]
    state_cap = int(env_batch["state_dim"])

    active_indices = [_coarse_batch_reference_active_index(env_batch, i) for i in range(len(h_list))]
    slow_fns = []
    for i, h in enumerate(h_list):
        g = torch.Generator(device="cpu")
        g.manual_seed(seeds[i])
        slow_fns.append(
            prior._build_reference_gp_joint_transition_fn(
                int(active_indices[i].numel()),
                int(h["state_dim"]),
                h,
                "cpu",
                generator=g,
            )
        )

    def _slow_step(x_full):
        state_out = torch.zeros((len(h_list), state_cap), dtype=x_full.dtype)
        reward_out = torch.zeros((len(h_list), 1), dtype=x_full.dtype)
        for bi, fn in enumerate(slow_fns):
            x_compact = x_full[bi: bi + 1].index_select(1, active_indices[bi])
            state_b, reward_b = fn(x_compact)
            state_out[bi, :int(h_list[bi]["state_dim"])] = state_b.squeeze(0)
            reward_out[bi, 0] = reward_b.reshape(())
        return state_out, reward_out

    for step in range(6):
        env_in = torch.linspace(
            -1.0 + (0.05 * step),
            0.9 - (0.03 * step),
            steps=len(h_list) * int(env_batch["env_input_dim"]),
            dtype=torch.float32,
        ).reshape(len(h_list), int(env_batch["env_input_dim"]))
        state_fast, reward_fast = fast_fn(env_in)
        state_slow, reward_slow = _slow_step(env_in)
        assert torch.allclose(state_fast, state_slow, atol=2e-5, rtol=1e-5)
        assert torch.allclose(reward_fast, reward_slow, atol=5e-6, rtol=1e-5)


def test_environment_prior_strict_reference_gp_joint_transition_repeats_identical_query_without_noise():
    _seed_everything(20260309)
    prior = EnvironmentPrior({})
    h = _manual_sampled_h(
        family="gp",
        state_dim=3,
        obs_dim=2,
        action_dim=1,
        noise_dim=2,
        zero_pad_dim=1,
        gp_rff_features=16,
    )
    h["strict_joint_transition_enabled"] = True
    h["reference_gp_forward_mode"] = "exact"
    h["outputscale"] = 0.9
    h["lengthscale"] = 1.2
    h["noise"] = 0.0

    env = prior._sample_environment(h, device="cpu", rng_seed=123)
    env_in = torch.linspace(-0.4, 0.5, steps=int(env["env_input_dim"]), dtype=torch.float32).reshape(1, -1)
    state_first, reward_first = env["transition_generator"](env_in)
    state_second, reward_second = env["transition_generator"](env_in)

    assert torch.allclose(state_first, state_second, atol=1e-6, rtol=1e-6)
    assert torch.allclose(reward_first, reward_second, atol=1e-6, rtol=1e-6)


def test_environment_prior_reference_gp_fixed_cost_joint_transition_repeats_identical_query_without_noise():
    _seed_everything(20260309)
    prior = EnvironmentPrior({})
    h = _manual_sampled_h(
        family="gp",
        state_dim=3,
        obs_dim=2,
        action_dim=1,
        noise_dim=2,
        zero_pad_dim=1,
        gp_rff_features=256,
    )
    h["strict_joint_transition_enabled"] = True
    h["reference_gp_forward_mode"] = "fixed_cost"
    h["outputscale"] = 0.9
    h["lengthscale"] = 1.2
    h["noise"] = 0.0

    env = prior._sample_environment(h, device="cpu", rng_seed=123)
    assert bool(getattr(env["transition_generator"], "_reference_gp_fixed_cost", False))
    env_in = torch.linspace(-0.4, 0.5, steps=int(env["env_input_dim"]), dtype=torch.float32).reshape(1, -1)
    state_first, reward_first = env["transition_generator"](env_in)
    state_second, reward_second = env["transition_generator"](env_in)

    assert torch.allclose(state_first, state_second, atol=1e-6, rtol=1e-6)
    assert torch.allclose(reward_first, reward_second, atol=1e-6, rtol=1e-6)


def test_environment_prior_reference_gp_fixed_cost_family_coarse_batch_matches_single_builders():
    _seed_everything(20260309)
    prior = EnvironmentPrior({})
    h_list = [
        _manual_sampled_h(
            family="gp",
            state_dim=4,
            obs_dim=2,
            action_dim=2,
            noise_dim=3,
            zero_pad_dim=1,
            gp_rff_features=128,
        ),
        _manual_sampled_h(
            family="gp",
            state_dim=6,
            obs_dim=3,
            action_dim=2,
            noise_dim=4,
            zero_pad_dim=0,
            gp_rff_features=192,
        ),
    ]
    seeds = [1777, 1888]
    for idx, h in enumerate(h_list):
        h["strict_joint_transition_enabled"] = True
        h["reference_gp_forward_mode"] = "fixed_cost"
        h["outputscale"] = 1.1 + (0.2 * idx)
        h["lengthscale"] = 0.7 + (0.1 * idx)
        h["noise"] = 0.0

    env_batch = prior._sample_environment_family_coarse_batch(
        h_list=h_list,
        device="cpu",
        rng_seeds=seeds,
        build_x_generator=False,
        build_y_generator=False,
        build_policy_generator=False,
    )
    assert bool(getattr(env_batch["transition_generator"], "_reference_gp_fixed_cost", False))
    env_in = torch.linspace(
        -0.8,
        0.9,
        steps=len(h_list) * int(env_batch["env_input_dim"]),
        dtype=torch.float32,
    ).reshape(len(h_list), int(env_batch["env_input_dim"]))
    state_actual, reward_actual = env_batch["transition_generator"](env_in)

    for bi, (seed, h) in enumerate(zip(seeds, h_list)):
        env_single = prior._sample_environment(h, device="cpu", rng_seed=seed)
        active_idx = _coarse_batch_reference_active_index(env_batch, bi)
        env_in_compact = env_in[bi: bi + 1].index_select(1, active_idx)
        state_ref, reward_ref = env_single["transition_generator"](env_in_compact)
        state_dim = int(h["state_dim"])
        assert torch.allclose(state_actual[bi: bi + 1, :state_dim], state_ref, atol=1e-6, rtol=1e-6)
        assert torch.allclose(reward_actual[bi: bi + 1], reward_ref, atol=1e-6, rtol=1e-6)
        if state_actual.shape[1] > state_dim:
            assert torch.allclose(state_actual[bi, state_dim:], torch.zeros_like(state_actual[bi, state_dim:]))


def test_environment_prior_reference_gp_fixed_cost_reward_moments_match_fast_gp_kernel():
    _seed_everything(20260309)
    prior = EnvironmentPrior({})
    h = _manual_sampled_h(
        family="gp",
        state_dim=1,
        obs_dim=1,
        action_dim=1,
        noise_dim=1,
        zero_pad_dim=0,
        gp_rff_features=2048,
    )
    h["strict_joint_transition_enabled"] = True
    h["reference_gp_forward_mode"] = "fixed_cost"
    h["outputscale"] = 1.25
    h["lengthscale"] = 0.7
    h["noise"] = 0.0

    probe_env = prior._sample_environment(h, device="cpu", rng_seed=0)
    in_dim = int(probe_env["env_input_dim"])
    x = torch.linspace(-0.9, 0.8, steps=4 * in_dim, dtype=torch.float32).reshape(4, in_dim)

    reward_draws = []
    for seed in range(96):
        env = prior._sample_environment(h, device="cpu", rng_seed=1000 + seed)
        _, reward = env["transition_generator"](x)
        reward_draws.append(reward.squeeze(-1))
    reward_draws = torch.stack(reward_draws, dim=0)
    reward_centered = reward_draws - reward_draws.mean(dim=0, keepdim=True)
    cov_emp = (reward_centered.transpose(0, 1) @ reward_centered) / float(reward_draws.shape[0])
    cov_ref = _reference_gp_kernel(
        x.to(dtype=torch.float64),
        x.to(dtype=torch.float64),
        lengthscale=float(h["lengthscale"]),
        outputscale=float(h["outputscale"]),
    ).to(dtype=torch.float32)

    assert torch.allclose(reward_draws.mean(dim=0), torch.zeros(4, dtype=torch.float32), atol=0.20, rtol=0.0)
    assert torch.allclose(cov_emp, cov_ref, atol=0.20, rtol=0.20)


def test_environment_prior_strict_reference_semantics_override_env_fields_and_serial_rollout(monkeypatch):
    _seed_everything(20260309)
    prior = EnvironmentPrior({})
    captured = {}

    def _constant_policy_builder(in_dim, out_dim, h, device, generator=None, apply_output_tanh=True):
        del h, generator

        def _fn(x, generator=None):
            del generator
            return torch.zeros((x.shape[0], out_dim), device=device, dtype=torch.float32)

        _fn._applies_output_tanh = bool(apply_output_tanh)
        return _fn

    def _constant_transition_builder(in_dim, state_dim, h, device, generator=None):
        del h, generator

        def _fn(x, generator=None):
            del generator
            captured["serial_env_input_dim"] = int(x.shape[-1])
            captured["serial_env_in"] = x.detach().clone()
            state = torch.full((x.shape[0], state_dim), 2.0, device=device, dtype=torch.float32)
            reward = torch.full((x.shape[0], 1), 3.0, device=device, dtype=torch.float32)
            return state, reward

        _fn._applies_output_tanh = False
        return _fn

    monkeypatch.setattr(prior, "_build_scm_fn", _constant_policy_builder)
    monkeypatch.setattr(prior, "_build_scm_joint_transition_fn", _constant_transition_builder)
    monkeypatch.setattr(prior, "_build_reference_scm_joint_transition_fn", _constant_transition_builder)

    h = _manual_sampled_h(
        family="scm",
        state_dim=3,
        obs_dim=2,
        action_dim=1,
        noise_dim=1,
        zero_pad_dim=1,
        num_layers=3,
    )
    h["strict_joint_transition_enabled"] = True
    h["alpha"] = 0.2
    h["state_noise_std"] = 1.0
    h["reward_scale"] = 7.0
    h["reward_clip"] = 0.1
    h["state_clip"] = 0.1
    h["state_input_scale_enabled"] = True
    h["state_input_scale"] = 4.0
    h["state_highway_enabled"] = True
    h["state_highway_lambda"] = 1.0
    h["reward_dropout_enabled"] = True
    h["reward_dropout_randomize"] = False
    h["reward_dropout_ratio"] = 1.0
    h["reward_dropout_impute_zero"] = True

    env = prior._sample_environment(h, device="cpu", rng_seed=123)
    assert bool(env["reference_semantics_enabled"]) is True
    assert float(env["alpha"]) == 1.0
    assert float(env["state_noise_std"]) == 0.0
    assert float(env["reward_scale"]) == 1.0
    assert math.isinf(float(env["reward_clip"]))
    assert math.isinf(float(env["state_clip"]))
    assert bool(env["reward_dropout_enabled"]) is False
    assert float(env["reward_dropout_ratio"]) == 0.0
    assert bool(env["state_input_scale_enabled"]) is True
    assert float(env["state_input_scale"]) == 4.0
    assert int(env["env_input_dim"]) == 6
    assert env["env_obs_start"] is None
    assert int(env["env_action_start"]) == 3
    assert int(env["env_noise_start"]) == 4

    x, y, info = prior._rollout_single(
        env=env,
        n_samples=2,
        num_features=8,
        single_eval_pos=1,
        device="cpu",
        collect_x=True,
        collect_runtime_info=True,
        rng_seed=456,
    )

    assert captured["serial_env_input_dim"] == 6
    assert torch.allclose(y, torch.full_like(y, 3.0))
    assert torch.allclose(x[1, :2], torch.tensor([2.0, 2.0]))
    assert torch.allclose(captured["serial_env_in"][0, :3], torch.full((3,), 0.5))
    assert float(info["reward_drop_frac_realized"]) == 0.0


def test_environment_prior_strict_reference_semantics_shared_vectorized_rollout(monkeypatch):
    _seed_everything(20260309)
    prior = EnvironmentPrior({})
    captured = {}

    def _constant_policy_batch_builder(in_dim, out_dim, h_list, device, generators=None, apply_output_tanh=True):
        del h_list, generators

        def _fn(x, generators_for_noise=None, generator=None):
            del generators_for_noise, generator
            return torch.zeros((x.shape[0], out_dim), device=device, dtype=torch.float32)

        _fn._applies_output_tanh = bool(apply_output_tanh)
        return _fn

    def _constant_transition_batch_builder(in_dim, state_dim, h_list, device, generators=None):
        del h_list, generators

        def _fn(x, generators_for_noise=None):
            del generators_for_noise
            captured["shared_env_input_dim"] = int(x.shape[-1])
            captured["shared_env_in"] = x.detach().clone()
            state = torch.full((x.shape[0], state_dim), 2.0, device=device, dtype=torch.float32)
            reward = torch.full((x.shape[0], 1), 3.0, device=device, dtype=torch.float32)
            return state, reward

        _fn._applies_output_tanh = False
        return _fn

    monkeypatch.setattr(prior, "_build_scm_batch_fn", _constant_policy_batch_builder)
    monkeypatch.setattr(prior, "_build_scm_joint_transition_batch_fn", _constant_transition_batch_builder)
    monkeypatch.setattr(prior, "_build_reference_scm_joint_transition_batch_fn", _constant_transition_batch_builder)

    h = _manual_sampled_h(
        family="scm",
        state_dim=3,
        obs_dim=2,
        action_dim=1,
        noise_dim=1,
        zero_pad_dim=1,
        num_layers=3,
    )
    h["strict_joint_transition_enabled"] = True
    h["alpha"] = 0.2
    h["state_noise_std"] = 1.0
    h["reward_scale"] = 7.0
    h["reward_clip"] = 0.1
    h["state_clip"] = 0.1
    h["state_input_scale_enabled"] = True
    h["state_input_scale"] = 4.0
    h["reward_dropout_enabled"] = True
    h["reward_dropout_randomize"] = False
    h["reward_dropout_ratio"] = 1.0

    env = prior._sample_environment_batch([h, h], device="cpu", rng_seeds=[11, 23])
    assert bool(env["reference_semantics_enabled"]) is True
    assert int(env["env_input_dim"]) == 6
    assert env["env_obs_start"] is None
    assert int(env["env_action_start"]) == 3
    assert int(env["env_noise_start"]) == 4

    x, y, info = prior._rollout_shared_env_vectorized(
        env=env,
        batch_size=2,
        n_samples=2,
        num_features=8,
        single_eval_pos=1,
        device="cpu",
        collect_x=True,
        rng_seeds=[101, 211],
    )

    assert captured["shared_env_input_dim"] == 6
    assert torch.allclose(y, torch.full_like(y, 3.0))
    assert torch.allclose(x[1, :, :2], torch.full((2, 2), 2.0))
    assert torch.allclose(captured["shared_env_in"][:, :3], torch.full((2, 3), 0.5))
    assert len(info) == 2
    assert all(float(row["reward_drop_frac_realized"]) == 0.0 for row in info)


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

    assert torch.allclose(rollout_serial["x"], rollout_vec["x"], atol=1e-6, rtol=1e-5)
    assert torch.allclose(rollout_serial["rewards"], rollout_vec["rewards"], atol=1e-6, rtol=1e-5)
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
        common_keys = set(serial_row.keys()) & set(vec_row.keys())
        assert common_keys
        for key in common_keys:
            s_val = serial_row[key]
            v_val = vec_row[key]
            if isinstance(s_val, float):
                assert np.isclose(s_val, v_val, rtol=5e-6, atol=2e-7)
            else:
                assert s_val == v_val


def test_environment_prior_rollout_with_policy_strict_joint_transition_matches_serial_and_vectorized_with_identical_h_and_seeds():
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
            state_dim=7,
            obs_dim=5,
            action_dim=2,
            noise_dim=3,
            zero_pad_dim=1,
            num_layers=2,
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
    for h in sampled:
        h["strict_joint_transition_enabled"] = True
    env_seeds = [11, 23, 37, 41]
    rollout_seeds = [101, 211, 307, 401]

    config = get_prior_config()
    env_cfg = dict(config["prior"]["environment"])
    env_cfg["batch_shared_environment"] = False
    env_cfg["batch_parallel_workers"] = 1
    env_cfg["batch_vectorized_strict_rng_match"] = True

    _seed_everything(20260309)
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

    # Strict joint-reference SCM semantics removes several legacy squash/mix
    # steps, so serial and vectorized paths now differ only by small
    # floating-point ordering effects inside the fused joint transition.
    assert torch.allclose(rollout_serial["x"], rollout_vec["x"], atol=3e-4, rtol=1e-4)
    assert torch.allclose(rollout_serial["rewards"], rollout_vec["rewards"], atol=3e-4, rtol=1e-4)
    assert torch.allclose(loss_serial.detach(), loss_vec.detach(), atol=3e-4, rtol=1e-4)
    assert len(grads_serial) == len(grads_vec)
    for g_serial, g_vec in zip(grads_serial, grads_vec):
        assert torch.allclose(g_serial, g_vec, atol=1e-3, rtol=1e-3)


def test_environment_prior_reference_rollout_does_not_force_strict_seed_without_strict_rng_match():
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
    ]
    for h in sampled:
        h["strict_joint_transition_enabled"] = True

    config = get_prior_config()
    env_cfg = dict(config["prior"]["environment"])
    env_cfg["batch_shared_environment"] = False
    env_cfg["batch_parallel_backend"] = "torch_vectorized"
    env_cfg["batch_vectorized_grouping"] = "family"
    env_cfg["batch_vectorized_strict_rng_match"] = False

    prior = EnvironmentPrior(env_cfg)
    prior._sample_batch_hypers = lambda batch_size: sampled[:batch_size]

    def _unexpected_seed_list(batch_size):
        raise AssertionError(f"_sample_seed_list should not be called for batch_size={batch_size}")

    prior._sample_seed_list = _unexpected_seed_list
    policy = _TinyMaskPolicy(obs_dim_cap=7, action_dim_cap=3)
    prior.rollout_with_policy(
        policy_step_fn=policy.step,
        batch_size=2,
        n_samples=10,
        num_features=32,
        device="cpu",
        single_eval_pos=5,
        collect_x=False,
    )
    rollout_profile = prior.last_rollout_profile
    assert isinstance(rollout_profile, dict)
    assert rollout_profile.get("noise_mode", None) != "strict_seed"


def test_environment_prior_reference_rollout_overrides_keep_strict_seed_mode():
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
    ]
    for h in sampled:
        h["strict_joint_transition_enabled"] = True

    config = get_prior_config()
    env_cfg = dict(config["prior"]["environment"])
    env_cfg["batch_shared_environment"] = False
    env_cfg["batch_parallel_backend"] = "torch_vectorized"
    env_cfg["batch_vectorized_grouping"] = "family"
    env_cfg["batch_vectorized_strict_rng_match"] = False

    prior = EnvironmentPrior(env_cfg)
    policy = _TinyMaskPolicy(obs_dim_cap=7, action_dim_cap=3)
    prior.rollout_with_policy(
        policy_step_fn=policy.step,
        batch_size=2,
        n_samples=10,
        num_features=32,
        device="cpu",
        single_eval_pos=5,
        collect_x=False,
        h_list_override=sampled,
        env_seeds_override=[11, 23],
        rollout_seeds_override=[101, 211],
    )
    rollout_profile = prior.last_rollout_profile
    assert isinstance(rollout_profile, dict)
    assert rollout_profile.get("noise_mode", None) == "strict_seed"


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


def test_environment_prior_rollout_with_policy_family_grouping_matches_serial_in_deterministic_setup_with_strict_joint_transition():
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
            state_dim=7,
            obs_dim=5,
            action_dim=2,
            noise_dim=3,
            zero_pad_dim=1,
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
            action_dim=2,
            noise_dim=3,
            zero_pad_dim=1,
            num_layers=3,
        ),
    ]
    for h in sampled:
        h["strict_joint_transition_enabled"] = True
        h["init_std"] = 0.0
        h["outputscale"] = 0.0
        h["noise"] = 0.0

    config = get_prior_config()
    env_cfg = dict(config["prior"]["environment"])
    env_cfg["batch_shared_environment"] = False
    env_cfg["batch_parallel_workers"] = 1
    env_cfg["batch_vectorized_strict_rng_match"] = False
    env_seeds = [11, 23, 37, 41]
    rollout_seeds = [101, 211, 307, 401]

    _seed_everything(20260309)
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
        h_list_override=sampled,
        env_seeds_override=env_seeds,
        rollout_seeds_override=rollout_seeds,
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
        h_list_override=sampled,
        env_seeds_override=env_seeds,
        rollout_seeds_override=rollout_seeds,
    )
    loss_vec, _ = prior_vec.policy_gradient_loss_from_rewards(rollout_vec["rewards"])
    loss_vec.backward()
    grads_vec = [p.grad.detach().clone() for p in policy_vec.parameters()]

    # Under reference SCM semantics, bias initialization remains stochastic even
    # when init_std=0, so family-group equivalence must be checked under matched
    # per-column env/rollout RNG streams rather than implicit global-RNG order.
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

    call_sizes = []

    class ZeroPolicy(nn.Module):
        def step(self, obs_t, action_t, reward_t, reward_mask_t, cache, step_idx, env_info):
            del obs_t, reward_t, reward_mask_t, cache, step_idx, env_info
            return torch.zeros_like(action_t)

    env_backup = {
        "TICL_POLICY_TRANSITION_INNER_GROUPING": os.environ.get("TICL_POLICY_TRANSITION_INNER_GROUPING"),
        "TICL_POLICY_TRANSITION_INNER_MIN_BUCKET": os.environ.get("TICL_POLICY_TRANSITION_INNER_MIN_BUCKET"),
    }
    try:
        os.environ["TICL_POLICY_TRANSITION_INNER_GROUPING"] = "family"
        prior = EnvironmentPrior(env_cfg)
        prior._sample_batch_hypers = lambda batch_size: sampled[:batch_size]
        orig_coarse = prior._sample_environment_family_coarse_batch

        def _count_coarse(h_list, device, rng_seeds=None, **kwargs):
            call_sizes.append(len(h_list))
            return orig_coarse(h_list=h_list, device=device, rng_seeds=rng_seeds, **kwargs)

        prior._sample_environment_family_coarse_batch = _count_coarse
        rollout = prior.rollout_with_policy(
            policy_step_fn=ZeroPolicy().step,
            batch_size=4,
            n_samples=8,
            num_features=32,
            device="cpu",
            single_eval_pos=4,
            collect_x=False,
        )
    finally:
        if env_backup is None:
            os.environ.pop("TICL_POLICY_TRANSITION_INNER_GROUPING", None)
        else:
            os.environ["TICL_POLICY_TRANSITION_INNER_GROUPING"] = env_backup
    assert rollout["rewards"].shape == (8, 4)
    # Coarse family subgrouping: scm + gp -> 2 groups.
    assert sorted(call_sizes) == [1, 3]


def test_environment_prior_rollout_with_policy_family_grouping_uses_transition_structure_buckets():
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

    call_sizes = []
    env_backup = {
        "TICL_POLICY_TRANSITION_INNER_GROUPING": os.environ.get("TICL_POLICY_TRANSITION_INNER_GROUPING"),
        "TICL_POLICY_TRANSITION_INNER_MIN_BUCKET": os.environ.get("TICL_POLICY_TRANSITION_INNER_MIN_BUCKET"),
    }

    class ZeroPolicy(nn.Module):
        def step(self, obs_t, action_t, reward_t, reward_mask_t, cache, step_idx, env_info):
            del obs_t, reward_t, reward_mask_t, cache, step_idx, env_info
            return torch.zeros_like(action_t)

    try:
        os.environ["TICL_POLICY_TRANSITION_INNER_GROUPING"] = "structure"
        os.environ["TICL_POLICY_TRANSITION_INNER_MIN_BUCKET"] = "0"
        prior = EnvironmentPrior(env_cfg)
        prior._sample_batch_hypers = lambda batch_size: sampled[:batch_size]
        orig_coarse = prior._sample_environment_family_coarse_batch

        def _count_coarse(h_list, device, rng_seeds=None, **kwargs):
            call_sizes.append(len(h_list))
            return orig_coarse(h_list=h_list, device=device, rng_seeds=rng_seeds, **kwargs)

        prior._sample_environment_family_coarse_batch = _count_coarse
        rollout = prior.rollout_with_policy(
            policy_step_fn=ZeroPolicy().step,
            batch_size=4,
            n_samples=8,
            num_features=32,
            device="cpu",
            single_eval_pos=4,
            collect_x=False,
        )
    finally:
        for key, value in env_backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    assert rollout["rewards"].shape == (8, 4)
    assert sorted(call_sizes) == [1, 1, 1, 1]


def test_environment_prior_transition_inner_grouping_defaults_to_family_with_min_bucket_zero():
    env_backup = {
        "TICL_POLICY_TRANSITION_INNER_GROUPING": os.environ.get("TICL_POLICY_TRANSITION_INNER_GROUPING"),
        "TICL_POLICY_TRANSITION_INNER_MIN_BUCKET": os.environ.get("TICL_POLICY_TRANSITION_INNER_MIN_BUCKET"),
    }
    try:
        os.environ.pop("TICL_POLICY_TRANSITION_INNER_GROUPING", None)
        os.environ.pop("TICL_POLICY_TRANSITION_INNER_MIN_BUCKET", None)
        prior = EnvironmentPrior(get_prior_config()["prior"]["environment"])
        assert prior.transition_inner_grouping == "family"
        assert prior.transition_inner_min_bucket == 0
    finally:
        for key, value in env_backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_environment_prior_transition_structure_bucketing_matches_family_bucket_semantics():
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
        h["init_std"] = 0.0
        h["outputscale"] = 0.0
        h["noise"] = 0.0

    config = get_prior_config()
    env_cfg = dict(config["prior"]["environment"])
    env_cfg["batch_parallel_backend"] = "torch_vectorized"
    env_cfg["batch_vectorized_grouping"] = "family"
    env_cfg["batch_vectorized_strict_rng_match"] = False
    env_cfg["batch_shared_environment"] = False

    def _run(inner_grouping):
        env_backup = os.environ.get("TICL_POLICY_TRANSITION_INNER_GROUPING")
        try:
            os.environ["TICL_POLICY_TRANSITION_INNER_GROUPING"] = inner_grouping
            prior = EnvironmentPrior(env_cfg)
            prior._sample_batch_hypers = lambda batch_size: sampled[:batch_size]
            policy = _TinyMaskPolicy(obs_dim_cap=7, action_dim_cap=3)
            rollout = prior.rollout_with_policy(
                policy_step_fn=policy.step,
                batch_size=4,
                n_samples=12,
                num_features=32,
                device="cpu",
                single_eval_pos=6,
                collect_x=True,
            )
            loss, _ = prior.policy_gradient_loss_from_rewards(rollout["rewards"])
            loss.backward()
            grads = [p.grad.detach().clone() for p in policy.parameters()]
            return rollout, loss.detach().clone(), grads
        finally:
            if env_backup is None:
                os.environ.pop("TICL_POLICY_TRANSITION_INNER_GROUPING", None)
            else:
                os.environ["TICL_POLICY_TRANSITION_INNER_GROUPING"] = env_backup

    _seed_everything(20260307)
    rollout_family, loss_family, grads_family = _run("family")
    _seed_everything(20260307)
    rollout_structure, loss_structure, grads_structure = _run("structure")

    assert torch.allclose(rollout_family["x"], rollout_structure["x"], atol=1e-6, rtol=1e-5)
    assert torch.allclose(rollout_family["rewards"], rollout_structure["rewards"], atol=1e-6, rtol=1e-5)
    assert torch.allclose(loss_family, loss_structure, atol=1e-6, rtol=1e-5)
    for g_family, g_structure in zip(grads_family, grads_structure):
        assert torch.allclose(g_family, g_structure, atol=1e-6, rtol=1e-5)


def test_environment_prior_transition_min_bucket_merges_small_buckets_back_to_family():
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

    call_sizes = []
    env_backup = {
        "TICL_POLICY_TRANSITION_INNER_GROUPING": os.environ.get("TICL_POLICY_TRANSITION_INNER_GROUPING"),
        "TICL_POLICY_TRANSITION_INNER_MIN_BUCKET": os.environ.get("TICL_POLICY_TRANSITION_INNER_MIN_BUCKET"),
    }

    class ZeroPolicy(nn.Module):
        def step(self, obs_t, action_t, reward_t, reward_mask_t, cache, step_idx, env_info):
            del obs_t, reward_t, reward_mask_t, cache, step_idx, env_info
            return torch.zeros_like(action_t)

    try:
        os.environ["TICL_POLICY_TRANSITION_INNER_GROUPING"] = "structure"
        os.environ["TICL_POLICY_TRANSITION_INNER_MIN_BUCKET"] = "2"
        prior = EnvironmentPrior(env_cfg)
        prior._sample_batch_hypers = lambda batch_size: sampled[:batch_size]
        orig_coarse = prior._sample_environment_family_coarse_batch

        def _count_coarse(h_list, device, rng_seeds=None, **kwargs):
            call_sizes.append(len(h_list))
            return orig_coarse(h_list=h_list, device=device, rng_seeds=rng_seeds, **kwargs)

        prior._sample_environment_family_coarse_batch = _count_coarse
        rollout = prior.rollout_with_policy(
            policy_step_fn=ZeroPolicy().step,
            batch_size=4,
            n_samples=8,
            num_features=32,
            device="cpu",
            single_eval_pos=4,
            collect_x=False,
        )
    finally:
        for key, value in env_backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    assert rollout["rewards"].shape == (8, 4)
    assert sorted(call_sizes) == [1, 3]


def test_environment_prior_balanced_transition_bucket_groups_split_family_into_few_buckets():
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
            state_dim=15,
            obs_dim=10,
            action_dim=3,
            noise_dim=4,
            zero_pad_dim=2,
            num_layers=4,
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
            family="scm",
            state_dim=18,
            obs_dim=11,
            action_dim=3,
            noise_dim=4,
            zero_pad_dim=2,
            num_layers=5,
        ),
    ]
    prior = EnvironmentPrior(dict(get_prior_config()["prior"]["environment"]))
    family_group = list(enumerate(sampled))
    groups = prior._build_balanced_transition_bucket_groups(
        family_group=family_group,
        bucket_count=2,
    )
    group_sizes = sorted(len(group) for group in groups.values())
    assert sum(group_sizes) == len(sampled)
    assert len(group_sizes) == 2
    assert max(group_sizes) <= 3
    assert min(group_sizes) >= 1


def test_environment_prior_family_coarse_batch_transition_only_build_falls_back_without_fused_transition():
    _seed_everything(20260307)
    cfg = dict(get_prior_config()["prior"]["environment"])
    sampled = [
        _manual_sampled_h(
            family="scm",
            state_dim=6,
            obs_dim=4,
            action_dim=3,
            noise_dim=5,
            zero_pad_dim=2,
            num_layers=3,
        ),
        _manual_sampled_h(
            family="scm",
            state_dim=5,
            obs_dim=3,
            action_dim=2,
            noise_dim=4,
            zero_pad_dim=1,
            num_layers=2,
        ),
    ]
    prior = EnvironmentPrior(cfg)
    env_batch = prior._sample_environment_family_coarse_batch(
        h_list=sampled,
        device="cpu",
        prefer_transition_only=True,
        build_policy_generator=False,
    )
    assert callable(env_batch["x_generator"])
    assert callable(env_batch["y_generator"])
    assert env_batch["policy_generator"] is None
    assert env_batch["transition_generator"] is None
    build_profile = env_batch["_build_profile"]
    assert int(build_profile["transition_only_build_enabled"]) == 0
    assert int(build_profile["non_transition_generator_build_count"]) == 2
    assert int(build_profile["non_transition_generator_skip_count"]) == 1


def test_environment_prior_family_coarse_batch_skipped_generators_preserve_cpu_rng_state():
    _seed_everything(20260307)
    cfg = dict(get_prior_config()["prior"]["environment"])
    sampled = [
        _manual_sampled_h(
            family="scm",
            state_dim=6,
            obs_dim=4,
            action_dim=3,
            noise_dim=5,
            zero_pad_dim=2,
            num_layers=3,
        ),
        _manual_sampled_h(
            family="gp",
            state_dim=5,
            obs_dim=3,
            action_dim=2,
            noise_dim=4,
            zero_pad_dim=1,
        ),
    ]
    prior = EnvironmentPrior(cfg)

    torch.manual_seed(20260307)
    cpu_rng_before = torch.random.get_rng_state()
    env_full = prior._sample_environment_family_coarse_batch(
        h_list=[sampled[0]],
        device="cpu",
        build_policy_generator=True,
        preserve_skipped_generator_rng=True,
    )
    cpu_rng_after_full = torch.random.get_rng_state()

    torch.random.set_rng_state(cpu_rng_before.clone())
    env_skip = prior._sample_environment_family_coarse_batch(
        h_list=[sampled[0]],
        device="cpu",
        build_x_generator=False,
        build_y_generator=False,
        build_policy_generator=False,
        preserve_skipped_generator_rng=True,
    )
    cpu_rng_after_skip = torch.random.get_rng_state()

    assert callable(env_full["x_generator"])
    assert callable(env_full["y_generator"])
    assert callable(env_full["policy_generator"])
    assert env_skip["x_generator"] is None
    assert env_skip["y_generator"] is None
    assert env_skip["policy_generator"] is None
    assert env_skip["transition_generator"] is None
    assert torch.equal(cpu_rng_after_full, cpu_rng_after_skip)
    build_profile = env_skip["_build_profile"]
    assert int(build_profile["transition_only_build_enabled"]) == 0
    assert int(build_profile["skipped_non_transition_generator_count"]) == 3
    assert int(build_profile["skipped_non_transition_rng_preserve_count"]) == 3
    assert int(build_profile["non_transition_generator_build_count"]) == 0
    assert int(build_profile["non_transition_generator_skip_count"]) == 3

    # Also probe the GP path because its init consumes mixed rand/randn draws.
    torch.manual_seed(20260308)
    cpu_rng_before = torch.random.get_rng_state()
    prior._sample_environment_family_coarse_batch(
        h_list=[sampled[1]],
        device="cpu",
        build_policy_generator=True,
        preserve_skipped_generator_rng=True,
    )
    cpu_rng_after_full = torch.random.get_rng_state()
    torch.random.set_rng_state(cpu_rng_before.clone())
    prior._sample_environment_family_coarse_batch(
        h_list=[sampled[1]],
        device="cpu",
        build_x_generator=False,
        build_y_generator=False,
        build_policy_generator=False,
        preserve_skipped_generator_rng=True,
    )
    cpu_rng_after_skip = torch.random.get_rng_state()
    assert torch.equal(cpu_rng_after_full, cpu_rng_after_skip)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for transition-only env build")
def test_environment_prior_family_coarse_batch_transition_only_build_skips_non_transition_generators_on_cuda():
    _seed_everything(20260307)
    torch.cuda.empty_cache()
    cfg = dict(get_prior_config()["prior"]["environment"])
    sampled = [
        _manual_sampled_h(
            family="scm",
            state_dim=6,
            obs_dim=4,
            action_dim=3,
            noise_dim=5,
            zero_pad_dim=2,
            num_layers=3,
        ),
        _manual_sampled_h(
            family="scm",
            state_dim=5,
            obs_dim=3,
            action_dim=2,
            noise_dim=4,
            zero_pad_dim=1,
            num_layers=2,
        ),
    ]
    prior = EnvironmentPrior(cfg)
    try:
        env_batch = prior._sample_environment_family_coarse_batch(
            h_list=sampled,
            device="cuda",
            prefer_transition_only=True,
            build_policy_generator=False,
        )
    except (torch.OutOfMemoryError, torch.AcceleratorError):
        pytest.skip("CUDA OOM while probing transition-only env build")
    assert env_batch["x_generator"] is None
    assert env_batch["y_generator"] is None
    assert env_batch["policy_generator"] is None
    assert callable(env_batch["transition_generator"])
    build_profile = env_batch["_build_profile"]
    assert int(build_profile["transition_only_build_enabled"]) == 1
    assert int(build_profile["non_transition_generator_build_count"]) == 0
    assert int(build_profile["non_transition_generator_skip_count"]) == 3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for fused-transition env build")
def test_environment_prior_family_coarse_batch_auto_elision_preserves_cuda_rng_state():
    _seed_everything(20260307)
    torch.cuda.empty_cache()
    cfg = dict(get_prior_config()["prior"]["environment"])
    sampled = [
        _manual_sampled_h(
            family="scm",
            state_dim=6,
            obs_dim=4,
            action_dim=3,
            noise_dim=5,
            zero_pad_dim=2,
            num_layers=3,
        ),
        _manual_sampled_h(
            family="scm",
            state_dim=5,
            obs_dim=3,
            action_dim=2,
            noise_dim=4,
            zero_pad_dim=1,
            num_layers=2,
        ),
    ]
    prior = EnvironmentPrior(cfg)
    device = torch.device("cuda")

    torch.manual_seed(20260307)
    torch.cuda.manual_seed_all(20260307)
    cpu_rng_before = torch.random.get_rng_state()
    cuda_rng_before = torch.cuda.get_rng_state(device=device)
    try:
        env_full = prior._sample_environment_family_coarse_batch(
            h_list=sampled,
            device=device,
            build_policy_generator=True,
            preserve_skipped_generator_rng=True,
        )
    except (torch.OutOfMemoryError, torch.AcceleratorError):
        pytest.skip("CUDA OOM while probing full family env build")
    cpu_rng_after_full = torch.random.get_rng_state()
    cuda_rng_after_full = torch.cuda.get_rng_state(device=device)

    torch.random.set_rng_state(cpu_rng_before.clone())
    torch.cuda.set_rng_state(cuda_rng_before.clone(), device=device)
    try:
        env_skip = prior._sample_environment_family_coarse_batch(
            h_list=sampled,
            device=device,
            build_x_generator=False,
            build_y_generator=False,
            build_policy_generator=False,
            preserve_skipped_generator_rng=True,
        )
    except (torch.OutOfMemoryError, torch.AcceleratorError):
        pytest.skip("CUDA OOM while probing elided family env build")
    cpu_rng_after_skip = torch.random.get_rng_state()
    cuda_rng_after_skip = torch.cuda.get_rng_state(device=device)

    assert callable(env_full["transition_generator"])
    assert callable(env_skip["transition_generator"])
    assert env_skip["x_generator"] is None
    assert env_skip["y_generator"] is None
    assert env_skip["policy_generator"] is None
    assert torch.equal(cpu_rng_after_full, cpu_rng_after_skip)
    assert torch.equal(cuda_rng_after_full, cuda_rng_after_skip)
    build_profile = env_skip["_build_profile"]
    assert int(build_profile["transition_only_build_enabled"]) == 0
    assert int(build_profile["skipped_non_transition_generator_count"]) == 3
    assert int(build_profile["skipped_non_transition_rng_preserve_count"]) == 3
    assert int(build_profile["non_transition_generator_build_count"]) == 0
    assert int(build_profile["non_transition_generator_skip_count"]) == 3


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


def _install_constant_terminal_builders(prior, monkeypatch):
    def _constant_policy_builder(in_dim, out_dim, h, device, generator=None, apply_output_tanh=True):
        del in_dim, h, generator, apply_output_tanh

        def _fn(x, generators_for_noise=None, generator=None):
            del generators_for_noise, generator
            return torch.zeros((x.shape[0], out_dim), device=device, dtype=torch.float32)

        _fn._applies_output_tanh = False
        return _fn

    def _constant_policy_batch_builder(in_dim, out_dim, h_list, device, generators=None, apply_output_tanh=True):
        del in_dim, h_list, generators, apply_output_tanh

        def _fn(x, generators_for_noise=None, generator=None):
            del generators_for_noise, generator
            return torch.zeros((x.shape[0], out_dim), device=device, dtype=torch.float32)

        _fn._applies_output_tanh = False
        return _fn

    def _constant_transition_builder(in_dim, state_dim, h, device, generator=None):
        del in_dim, h, generator

        def _fn(x, generators_for_noise=None, generator=None):
            del generators_for_noise, generator
            state = torch.full((x.shape[0], state_dim), 2.0, device=device, dtype=torch.float32)
            reward = torch.zeros((x.shape[0], 1), device=device, dtype=torch.float32)
            terminal_signal = torch.full((x.shape[0], 1), 5.0, device=device, dtype=torch.float32)
            return state, reward, terminal_signal

        _fn._applies_output_tanh = False
        return _fn

    def _constant_transition_batch_builder(in_dim, state_dim, h_list, device, generators=None):
        del in_dim, h_list, generators

        def _fn(x, generators_for_noise=None, generator=None, **kwargs):
            del generators_for_noise, generator, kwargs
            state = torch.full((x.shape[0], state_dim), 2.0, device=device, dtype=torch.float32)
            reward = torch.zeros((x.shape[0], 1), device=device, dtype=torch.float32)
            terminal_signal = torch.full((x.shape[0], 1), 5.0, device=device, dtype=torch.float32)
            return state, reward, terminal_signal

        _fn._applies_output_tanh = False
        return _fn

    monkeypatch.setattr(prior, "_build_scm_fn", _constant_policy_builder)
    monkeypatch.setattr(prior, "_build_scm_batch_fn", _constant_policy_batch_builder)
    monkeypatch.setattr(prior, "_build_scm_joint_transition_fn", _constant_transition_builder)
    monkeypatch.setattr(prior, "_build_scm_joint_transition_batch_fn", _constant_transition_batch_builder)
    monkeypatch.setattr(prior, "_build_reference_scm_joint_transition_fn", _constant_transition_builder)
    monkeypatch.setattr(prior, "_build_reference_scm_joint_transition_batch_fn", _constant_transition_batch_builder)


def _zero_policy_step(obs_t, action_t, reward_t, reward_mask_t, cache, step_idx, env_info):
    del obs_t, reward_t, reward_mask_t, cache, step_idx, env_info
    return torch.zeros_like(action_t)


def _make_terminal_sampled_h_list():
    specs = [
        dict(state_dim=3, obs_dim=2, action_dim=2, noise_dim=1, zero_pad_dim=0, num_layers=2),
        dict(state_dim=5, obs_dim=3, action_dim=2, noise_dim=1, zero_pad_dim=0, num_layers=3),
        dict(state_dim=4, obs_dim=2, action_dim=2, noise_dim=1, zero_pad_dim=0, num_layers=2),
        dict(state_dim=6, obs_dim=4, action_dim=2, noise_dim=1, zero_pad_dim=0, num_layers=3),
    ]
    sampled = []
    for spec in specs:
        h = _manual_sampled_h(family="scm", **spec)
        h["strict_joint_transition_enabled"] = True
        h["obs_slot_dim"] = 4
        h["action_slot_dim"] = 2
        h["init_state_std"] = 0.0
        h["init_action_std"] = 0.0
        h["alpha"] = 1.0
        h["reward_scale"] = 1.0
        h["reward_clip"] = 10.0
        h["state_full_rms_enabled"] = False
        h["state_noise_std"] = 0.0
        h["action_noise_train_std"] = 0.0
        h["action_noise_eval_std"] = 0.0
        h["terminal_reset_enabled"] = True
        h["terminal_reset_count_target"] = 4
        h["terminal_bonus_tanh_c"] = 10.0
        h["terminal_bonus_scale_min"] = 3.0
        h["terminal_bonus_scale_max"] = 3.0
        h["reinforce_reward_transform"] = "none"
        sampled.append(h)
    return sampled


def test_environment_prior_rollout_policy_gradient_loss_reports_terminal_count_stats(monkeypatch):
    _seed_everything(20260313)
    sampled = _make_terminal_sampled_h_list()[:1]
    sampled[0]["action_noise_train_std"] = 0.1
    sampled[0]["action_noise_eval_std"] = 0.1
    prior = EnvironmentPrior(dict(get_prior_config()["prior"]["environment"], batch_parallel_backend="python_thread"))
    _install_constant_terminal_builders(prior, monkeypatch)

    loss, rollout, stats = prior.rollout_policy_gradient_loss(
        policy_step_fn=_zero_policy_step,
        batch_size=1,
        n_samples=4,
        num_features=9,
        device="cpu",
        single_eval_pos=2,
        collect_x=True,
        h_list_override=sampled,
        env_seeds_override=[11],
        rollout_seeds_override=[101],
        policy_objective_kind="first_policy_gradient",
    )

    del loss
    assert isinstance(rollout.get("terminal_stats", None), dict)
    assert float(stats["terminal_count_mean"]) == 4.0
    assert float(stats["terminal_count_min"]) == 4.0
    assert float(stats["terminal_count_max"]) == 4.0
    assert float(stats["terminal_count_target_mean"]) == 4.0
    assert float(stats["terminal_count_target_min"]) == 4.0
    assert float(stats["terminal_count_target_max"]) == 4.0


def test_environment_prior_rollout_with_policy_structure_grouping_matches_serial_with_terminal_reset(monkeypatch):
    _seed_everything(20260313)
    sampled = _make_terminal_sampled_h_list()
    env_seeds = [11, 23, 37, 41]
    rollout_seeds = [101, 211, 307, 401]

    config = get_prior_config()
    env_cfg = dict(config["prior"]["environment"])
    env_cfg["batch_shared_environment"] = False
    env_cfg["batch_parallel_workers"] = 1
    env_cfg["batch_vectorized_strict_rng_match"] = False

    prior_serial = EnvironmentPrior(dict(env_cfg, batch_parallel_backend="python_thread"))
    prior_vec = EnvironmentPrior(dict(env_cfg, batch_parallel_backend="torch_vectorized", batch_vectorized_grouping="structure"))
    _install_constant_terminal_builders(prior_serial, monkeypatch)
    _install_constant_terminal_builders(prior_vec, monkeypatch)
    prior_serial._sample_batch_hypers = lambda batch_size: sampled[:batch_size]
    prior_vec._sample_batch_hypers = lambda batch_size: sampled[:batch_size]

    rollout_serial = prior_serial.rollout_with_policy(
        policy_step_fn=_zero_policy_step,
        batch_size=4,
        n_samples=4,
        num_features=9,
        device="cpu",
        single_eval_pos=2,
        collect_x=True,
        h_list_override=sampled,
        env_seeds_override=env_seeds,
        rollout_seeds_override=rollout_seeds,
    )
    rollout_vec = prior_vec.rollout_with_policy(
        policy_step_fn=_zero_policy_step,
        batch_size=4,
        n_samples=4,
        num_features=9,
        device="cpu",
        single_eval_pos=2,
        collect_x=True,
        h_list_override=sampled,
        env_seeds_override=env_seeds,
        rollout_seeds_override=rollout_seeds,
    )

    expected_bonus = 3.0 * math.tanh(5.0 / 10.0)
    assert torch.allclose(rollout_serial["x"], rollout_vec["x"], atol=1.5e-2, rtol=1e-6)
    assert torch.allclose(rollout_serial["rewards"], rollout_vec["rewards"], atol=1.5e-2, rtol=1e-6)
    assert torch.allclose(rollout_vec["rewards"], torch.full_like(rollout_vec["rewards"], expected_bonus), atol=1e-6, rtol=1e-6)
    assert torch.allclose(rollout_vec["x"][:, :, :2], torch.zeros_like(rollout_vec["x"][:, :, :2]), atol=1e-6, rtol=1e-6)
    terminal_col = int(sampled[0]["obs_slot_dim"]) + 2
    assert torch.all(rollout_vec["x"][0, :, terminal_col] == 0.0)
    assert torch.all(rollout_vec["x"][1:, :, terminal_col] == 1.0)
    assert isinstance(rollout_vec.get("terminal_stats", None), dict)
    assert float(rollout_vec["terminal_stats"]["terminal_count_mean"]) == 4.0


def test_environment_prior_rollout_with_policy_family_grouping_matches_serial_with_terminal_reset(monkeypatch):
    _seed_everything(20260313)
    base_h = _make_terminal_sampled_h_list()[0]
    sampled = [dict(base_h) for _ in range(4)]
    env_seeds = [11, 23, 37, 41]
    rollout_seeds = [101, 211, 307, 401]

    config = get_prior_config()
    env_cfg = dict(config["prior"]["environment"])
    env_cfg["batch_shared_environment"] = False
    env_cfg["batch_parallel_workers"] = 1
    env_cfg["batch_vectorized_strict_rng_match"] = False

    prior_serial = EnvironmentPrior(dict(env_cfg, batch_parallel_backend="python_thread"))
    prior_vec = EnvironmentPrior(dict(env_cfg, batch_parallel_backend="torch_vectorized", batch_vectorized_grouping="family"))
    _install_constant_terminal_builders(prior_serial, monkeypatch)
    _install_constant_terminal_builders(prior_vec, monkeypatch)
    prior_serial._sample_batch_hypers = lambda batch_size: sampled[:batch_size]
    prior_vec._sample_batch_hypers = lambda batch_size: sampled[:batch_size]
    prior_vec._sample_environment_family_coarse_batch = (
        lambda h_list, device, rng_seeds=None, **_: prior_vec._sample_environment_batch(
            h_list=h_list,
            device=device,
            rng_seeds=rng_seeds,
        )
    )

    rollout_serial = prior_serial.rollout_with_policy(
        policy_step_fn=_zero_policy_step,
        batch_size=4,
        n_samples=4,
        num_features=9,
        device="cpu",
        single_eval_pos=2,
        collect_x=True,
        h_list_override=sampled,
        env_seeds_override=env_seeds,
        rollout_seeds_override=rollout_seeds,
    )
    rollout_vec = prior_vec.rollout_with_policy(
        policy_step_fn=_zero_policy_step,
        batch_size=4,
        n_samples=4,
        num_features=9,
        device="cpu",
        single_eval_pos=2,
        collect_x=True,
        h_list_override=sampled,
        env_seeds_override=env_seeds,
        rollout_seeds_override=rollout_seeds,
    )

    expected_bonus = 3.0 * math.tanh(5.0 / 10.0)
    assert torch.allclose(rollout_serial["x"], rollout_vec["x"], atol=1.5e-2, rtol=1e-6)
    assert torch.allclose(rollout_serial["rewards"], rollout_vec["rewards"], atol=1.5e-2, rtol=1e-6)
    assert torch.allclose(rollout_vec["rewards"], torch.full_like(rollout_vec["rewards"], expected_bonus), atol=1e-6, rtol=1e-6)
    assert torch.allclose(rollout_vec["x"][:, :, :2], torch.zeros_like(rollout_vec["x"][:, :, :2]), atol=1e-6, rtol=1e-6)
    terminal_col = int(sampled[0]["obs_slot_dim"]) + 2
    assert torch.all(rollout_vec["x"][0, :, terminal_col] == 0.0)
    assert torch.all(rollout_vec["x"][1:, :, terminal_col] == 1.0)
    assert isinstance(rollout_vec.get("terminal_stats", None), dict)
    assert float(rollout_vec["terminal_stats"]["terminal_count_mean"]) == 4.0
    assert float(rollout_vec["terminal_stats"]["terminal_count_min"]) == 4.0
    assert float(rollout_vec["terminal_stats"]["terminal_count_max"]) == 4.0
    assert float(rollout_vec["terminal_stats"]["terminal_count_target_mean"]) == 4.0


def test_environment_prior_rollout_policy_gradient_loss_family_reports_terminal_count_stats(monkeypatch):
    _seed_everything(20260314)
    base_h = _make_terminal_sampled_h_list()[0]
    sampled = [dict(base_h) for _ in range(4)]
    for h in sampled:
        h["action_noise_train_std"] = 0.1
        h["action_noise_eval_std"] = 0.1

    config = get_prior_config()
    env_cfg = dict(config["prior"]["environment"])
    env_cfg["batch_shared_environment"] = False
    env_cfg["batch_parallel_workers"] = 1
    env_cfg["batch_vectorized_strict_rng_match"] = False

    prior = EnvironmentPrior(dict(env_cfg, batch_parallel_backend="torch_vectorized", batch_vectorized_grouping="family"))
    _install_constant_terminal_builders(prior, monkeypatch)
    prior._sample_batch_hypers = lambda batch_size: sampled[:batch_size]
    prior._sample_environment_family_coarse_batch = (
        lambda h_list, device, rng_seeds=None, **_: prior._sample_environment_batch(
            h_list=h_list,
            device=device,
            rng_seeds=rng_seeds,
        )
    )

    loss, rollout, stats = prior.rollout_policy_gradient_loss(
        policy_step_fn=_zero_policy_step,
        batch_size=4,
        n_samples=4,
        num_features=9,
        device="cpu",
        single_eval_pos=2,
        collect_x=True,
        h_list_override=sampled,
        env_seeds_override=[11, 23, 37, 41],
        rollout_seeds_override=[101, 211, 307, 401],
        policy_objective_kind="first_policy_gradient",
    )

    del loss
    assert isinstance(rollout.get("terminal_stats", None), dict)
    assert float(stats["terminal_count_mean"]) == 4.0
    assert float(stats["terminal_count_min"]) == 4.0
    assert float(stats["terminal_count_max"]) == 4.0
    assert float(stats["terminal_count_target_mean"]) == 4.0


def test_environment_prior_alpha_grad_family_tbptt_reports_terminal_count_stats(monkeypatch):
    _seed_everything(20260314)
    base_h = _make_terminal_sampled_h_list()[0]
    sampled = [dict(base_h) for _ in range(4)]
    for h in sampled:
        h["action_noise_train_std"] = 0.1
        h["action_noise_eval_std"] = 0.1

    config = get_prior_config()
    env_cfg = dict(config["prior"]["environment"])
    env_cfg["batch_shared_environment"] = False
    env_cfg["batch_parallel_workers"] = 1
    env_cfg["batch_vectorized_strict_rng_match"] = False

    prior = EnvironmentPrior(dict(env_cfg, batch_parallel_backend="torch_vectorized", batch_vectorized_grouping="family"))
    _install_constant_terminal_builders(prior, monkeypatch)
    prior._sample_batch_hypers = lambda batch_size: sampled[:batch_size]
    prior._sample_environment_family_coarse_batch = (
        lambda h_list, device, rng_seeds=None, **_: prior._sample_environment_batch(
            h_list=h_list,
            device=device,
            rng_seeds=rng_seeds,
        )
    )

    loss, rollout, stats = prior.rollout_policy_gradient_loss(
        policy_step_fn=_zero_policy_step,
        batch_size=4,
        n_samples=4,
        num_features=9,
        device="cpu",
        single_eval_pos=2,
        collect_x=False,
        tbptt_window=2,
        h_list_override=sampled,
        env_seeds_override=[11, 23, 37, 41],
        rollout_seeds_override=[101, 211, 307, 401],
        policy_objective_kind="alpha_grad",
    )

    del loss
    assert isinstance(rollout.get("terminal_stats", None), dict)
    assert float(rollout["terminal_stats"]["terminal_count_mean"]) == 4.0
    assert float(rollout["terminal_stats"]["terminal_count_target_mean"]) == 4.0
    assert float(stats["terminal_count_mean"]) == 4.0
    assert float(stats["terminal_count_target_mean"]) == 4.0
    assert int(stats["alpha_grad_enabled"]) == 1


def test_environment_prior_reinforce_family_tbptt_reports_one_hop_replay_stats():
    _seed_everything(2082)
    config = get_prior_config()
    env_cfg = dict(config["prior"]["environment"])
    env_cfg["family"] = {"distribution": "meta_choice", "choice_values": ["scm"]}
    env_cfg["batch_parallel_backend"] = "torch_vectorized"
    env_cfg["batch_vectorized_grouping"] = "family"
    env_cfg["batch_vectorized_strict_rng_match"] = False
    env_cfg["pg_one_hop_replay_enabled"] = True
    env_cfg["alpha_grad_one_hop_replay_enabled"] = True
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
            return self.net(torch.cat([obs_t[:, :4], action_t[:, :3], reward_t], dim=-1))

    policy = TinyPolicy()
    h_list = prior._sample_batch_hypers(4)
    for h in h_list:
        h["reward_dropout_enabled"] = False
        h["reward_dropout_randomize"] = False
        h["reward_dropout_ratio"] = 0.0
        h["action_noise_train_std"] = 0.2
        h["action_noise_eval_std"] = 0.15
    env_seeds = prior._sample_seed_list(4)
    rollout_seeds = prior._sample_seed_list(4)

    loss, _, stats = prior.rollout_policy_gradient_loss(
        policy_step_fn=policy.step,
        batch_size=4,
        n_samples=6,
        num_features=24,
        device="cpu",
        single_eval_pos=2,
        collect_x=False,
        tbptt_window=2,
        h_list_override=h_list,
        env_seeds_override=env_seeds,
        rollout_seeds_override=rollout_seeds,
        policy_objective_kind="reinforce",
    )

    assert torch.isfinite(loss)
    assert int(stats["reinforce_enabled"]) == 1
    assert int(stats["pg_one_hop_replay_enabled"]) == 1
    assert int(stats["pg_bridge_replay_count"]) >= 1


def test_environment_prior_reinforce_family_tbptt_clamps_replay_depth_to_safe_one_hop():
    _seed_everything(2083)
    config = get_prior_config()
    env_cfg = dict(config["prior"]["environment"])
    env_cfg["family"] = {"distribution": "meta_choice", "choice_values": ["scm"]}
    env_cfg["batch_parallel_backend"] = "torch_vectorized"
    env_cfg["batch_vectorized_grouping"] = "family"
    env_cfg["batch_vectorized_strict_rng_match"] = False
    env_cfg["pg_one_hop_replay_enabled"] = True
    env_cfg["alpha_grad_one_hop_replay_enabled"] = True
    env_cfg["pg_replay_window_depth"] = 2
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
            return self.net(torch.cat([obs_t[:, :4], action_t[:, :3], reward_t], dim=-1))

    policy = TinyPolicy()
    h_list = prior._sample_batch_hypers(4)
    for h in h_list:
        h["reward_dropout_enabled"] = False
        h["reward_dropout_randomize"] = False
        h["reward_dropout_ratio"] = 0.0
        h["action_noise_train_std"] = 0.2
        h["action_noise_eval_std"] = 0.15

    loss, _, stats = prior.rollout_policy_gradient_loss(
        policy_step_fn=policy.step,
        batch_size=4,
        n_samples=8,
        num_features=24,
        device="cpu",
        single_eval_pos=4,
        collect_x=False,
        tbptt_window=2,
        h_list_override=h_list,
        policy_objective_kind="reinforce",
    )

    assert torch.isfinite(loss)
    assert int(stats["pg_one_hop_replay_enabled"]) == 1
    assert int(stats["pg_replay_window_depth"]) == 1
    assert int(stats["pg_bridge_replay_count"]) >= 1


def test_environment_prior_reinforce_family_tbptt_markov_adjacent_future_replay_reports_sampled_bridges():
    _seed_everything(2084)
    config = get_prior_config()
    env_cfg = dict(config["prior"]["environment"])
    env_cfg["family"] = {"distribution": "meta_choice", "choice_values": ["scm"]}
    env_cfg["batch_parallel_backend"] = "torch_vectorized"
    env_cfg["batch_vectorized_grouping"] = "family"
    env_cfg["batch_vectorized_strict_rng_match"] = False
    env_cfg["pg_one_hop_replay_enabled"] = True
    env_cfg["alpha_grad_one_hop_replay_enabled"] = True
    env_cfg["pg_markov_adjacent_replay_enabled"] = True
    env_cfg["pg_markov_adjacent_replay_sample_prob"] = 1.0
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
            return self.net(torch.cat([obs_t[:, :4], action_t[:, :3], reward_t], dim=-1))

    policy = TinyPolicy()
    h_list = prior._sample_batch_hypers(4)
    for h in h_list:
        h["reward_dropout_enabled"] = False
        h["reward_dropout_randomize"] = False
        h["reward_dropout_ratio"] = 0.0
        h["action_noise_train_std"] = 0.2
        h["action_noise_eval_std"] = 0.15

    loss, _, stats = prior.rollout_policy_gradient_loss(
        policy_step_fn=policy.step,
        batch_size=4,
        n_samples=8,
        num_features=24,
        device="cpu",
        single_eval_pos=2,
        collect_x=False,
        tbptt_window=2,
        h_list_override=h_list,
        policy_objective_kind="reinforce",
    )

    assert torch.isfinite(loss)
    assert int(stats["pg_markov_adjacent_replay_enabled"]) == 1
    assert float(stats["pg_markov_adjacent_replay_sample_prob"]) == 1.0
    assert int(stats["pg_markov_adjacent_bridge_sampled_count"]) == 2
    assert int(stats["pg_bridge_replay_count"]) == 3


def test_environment_prior_reinforce_family_tbptt_skips_bridge_when_phase_boundary_is_in_first_window():
    _seed_everything(2085)
    config = get_prior_config()
    env_cfg = dict(config["prior"]["environment"])
    env_cfg["family"] = {"distribution": "meta_choice", "choice_values": ["scm"]}
    env_cfg["batch_parallel_backend"] = "torch_vectorized"
    env_cfg["batch_vectorized_grouping"] = "family"
    env_cfg["batch_vectorized_strict_rng_match"] = False
    env_cfg["pg_one_hop_replay_enabled"] = True
    env_cfg["alpha_grad_one_hop_replay_enabled"] = True
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
            return self.net(torch.cat([obs_t[:, :4], action_t[:, :3], reward_t], dim=-1))

    policy = TinyPolicy()
    h_list = prior._sample_batch_hypers(4)
    for h in h_list:
        h["reward_dropout_enabled"] = False
        h["reward_dropout_randomize"] = False
        h["reward_dropout_ratio"] = 0.0
        h["action_noise_train_std"] = 0.2
        h["action_noise_eval_std"] = 0.15

    loss, _, stats = prior.rollout_policy_gradient_loss(
        policy_step_fn=policy.step,
        batch_size=4,
        n_samples=8,
        num_features=24,
        device="cpu",
        single_eval_pos=1,
        collect_x=False,
        tbptt_window=4,
        h_list_override=h_list,
        policy_objective_kind="reinforce",
    )

    assert torch.isfinite(loss)
    assert int(stats["pg_one_hop_replay_enabled"]) == 1
    assert int(stats["pg_bridge_replay_count"]) == 0


def test_environment_prior_one_hop_replay_leaves_skip_paged_prefix():
    k_prefix = torch.randn(2, 3, 5, 7)
    v_prefix = torch.randn(2, 3, 5, 7)
    k_page = torch.randn(2, 3, 2, 7)
    v_page = torch.randn(2, 3, 2, 7)
    cache = {
        "cache_mode": "paged",
        "k_prefix": k_prefix,
        "v_prefix": v_prefix,
        "k_pages": [k_page],
        "v_pages": [v_page],
        "valid_len": 7,
        "prefix_base_len": 5,
        "tail_frozen": True,
    }

    replay_leaves = EnvironmentPrior._policy_cache_replay_leaves(cache)
    assert len(replay_leaves) == 2
    assert tuple(replay_leaves[0].shape) == tuple(k_page.shape)
    assert tuple(replay_leaves[1].shape) == tuple(v_page.shape)
    assert torch.allclose(replay_leaves[0], k_page)
    assert torch.allclose(replay_leaves[1], v_page)

    replay_grad_targets = EnvironmentPrior._enable_policy_cache_grad(cache)
    assert len(replay_grad_targets) == 2
    assert tuple(replay_grad_targets[0].shape) == tuple(k_page.shape)
    assert tuple(replay_grad_targets[1].shape) == tuple(v_page.shape)
    assert torch.allclose(replay_grad_targets[0], k_page)
    assert torch.allclose(replay_grad_targets[1], v_page)
    assert replay_grad_targets[0].requires_grad
    assert replay_grad_targets[1].requires_grad
    assert not k_prefix.requires_grad
    assert not v_prefix.requires_grad


def test_environment_prior_one_hop_replay_output_leaves_match_tbptt_detach_tail():
    k_page0 = torch.randn(2, 3, 4, 7)
    v_page0 = torch.randn(2, 3, 4, 7)
    k_page1 = torch.randn(2, 3, 4, 7)
    v_page1 = torch.randn(2, 3, 4, 7)
    cache_live = {
        "cache_mode": "paged",
        "k_pages": [k_page0, k_page1],
        "v_pages": [v_page0, v_page1],
        "valid_len": 6,
        "prefix_base_len": 0,
        "prefix_pages": 0,
        "tail_frozen": False,
    }
    cache_detached = EnvironmentPrior._detach_policy_cache(cache_live, clone_tensors=False)

    output_leaves = EnvironmentPrior._policy_cache_replay_leaves(cache_live, emulate_tbptt_detach=True)
    input_leaves = EnvironmentPrior._policy_cache_replay_leaves(cache_detached)

    assert len(output_leaves) == len(input_leaves) == 2
    assert output_leaves[0].shape == input_leaves[0].shape
    assert output_leaves[1].shape == input_leaves[1].shape
    assert tuple(output_leaves[0].shape) == (2, 3, 2, 7)
    assert tuple(output_leaves[1].shape) == (2, 3, 2, 7)


def test_environment_prior_one_hop_boundary_eta_keeps_only_active_fields():
    state_t = torch.randn(2, 4, requires_grad=True)
    action_t = torch.randn(2, 5, requires_grad=True)
    reward_t = torch.randn(2, requires_grad=True)
    reward_mask_t = torch.randn(2, requires_grad=True)
    terminal_t = torch.randn(2, requires_grad=True)
    cache_leaf = torch.randn(2, 3, requires_grad=True)

    loss = (
        (state_t ** 2).sum()
        + (action_t[:, :2] ** 2).sum()
        + (reward_t ** 2).sum()
        + (cache_leaf ** 2).sum()
    )
    boundary = {
        "state_t": state_t,
        "action_t": action_t,
        "action_replay_dim": 2,
        "reward_t": reward_t,
    }

    eta = EnvironmentPrior._alpha_tbptt_boundary_eta_from_loss(loss, boundary, (cache_leaf,))

    assert set(eta.keys()) == {"state_t", "action_t", "reward_t", "cache_leaves"}
    assert tuple(eta["action_t"].shape) == (2, 2)
    assert len(eta["cache_leaves"]) == 1
    assert torch.allclose(eta["state_t"], 2.0 * state_t.detach())
    assert torch.allclose(eta["action_t"], 2.0 * action_t[:, :2].detach())
    assert torch.allclose(eta["reward_t"], 2.0 * reward_t.detach())
    assert torch.allclose(eta["cache_leaves"][0], 2.0 * cache_leaf.detach())
    assert "reward_mask_t" not in eta
    assert "terminal_t" not in eta


def test_environment_prior_reinforce_one_hop_supports_immediate_tbptt_backward():
    _seed_everything(2084)
    config = get_prior_config()
    env_cfg = dict(config["prior"]["environment"])
    env_cfg["family"] = {"distribution": "meta_choice", "choice_values": ["scm"]}
    env_cfg["batch_parallel_backend"] = "torch_vectorized"
    env_cfg["batch_vectorized_grouping"] = "family"
    env_cfg["batch_vectorized_strict_rng_match"] = False
    env_cfg["pg_one_hop_replay_enabled"] = True
    env_cfg["alpha_grad_one_hop_replay_enabled"] = True
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
            return self.net(torch.cat([obs_t[:, :4], action_t[:, :3], reward_t], dim=-1))

    policy = TinyPolicy()
    policy.zero_grad(set_to_none=True)
    h_list = prior._sample_batch_hypers(4)
    for h in h_list:
        h["reward_dropout_enabled"] = False
        h["reward_dropout_randomize"] = False
        h["reward_dropout_ratio"] = 0.0
        h["action_noise_train_std"] = 0.2
        h["action_noise_eval_std"] = 0.15

    loss, _, stats = prior.rollout_policy_gradient_loss(
        policy_step_fn=policy.step,
        batch_size=4,
        n_samples=6,
        num_features=24,
        device="cpu",
        single_eval_pos=2,
        collect_x=False,
        tbptt_window=2,
        tbptt_loss_sink=lambda loss_root: loss_root.backward(),
        h_list_override=h_list,
        policy_objective_kind="reinforce",
    )

    assert float(loss.detach()) == 0.0
    grads = [p.grad for p in policy.parameters() if p.grad is not None]
    assert grads
    assert all(torch.isfinite(g).all() for g in grads)
    assert int(stats["pg_one_hop_replay_enabled"]) == 1
    assert int(stats["pg_bridge_replay_count"]) >= 1


@pytest.mark.parametrize("objective_kind", ["reinforce", "alpha_grad"])
def test_environment_prior_family_tbptt_uses_safe_runner_when_one_hop_disabled(monkeypatch, objective_kind):
    _seed_everything(2083)
    config = get_prior_config()
    env_cfg = dict(config["prior"]["environment"])
    env_cfg["family"] = {"distribution": "meta_choice", "choice_values": ["scm"]}
    env_cfg["batch_parallel_backend"] = "torch_vectorized"
    env_cfg["batch_vectorized_grouping"] = "family"
    env_cfg["batch_vectorized_strict_rng_match"] = False
    env_cfg["pg_one_hop_replay_enabled"] = False
    env_cfg["alpha_grad_one_hop_replay_enabled"] = False
    env_cfg["state_dim"] = {"distribution": "uniform_int", "min": 6, "max": 6}
    env_cfg["obs_dim"] = {"distribution": "uniform_int", "min": 4, "max": 4}
    env_cfg["action_dim"] = {"distribution": "uniform_int", "min": 3, "max": 3}
    env_cfg["noise_dim"] = {"distribution": "uniform_int", "min": 5, "max": 5}
    env_cfg["zero_pad_dim"] = {"distribution": "uniform_int", "min": 2, "max": 2}
    prior = EnvironmentPrior(env_cfg)

    def _forbidden_special_runner(*args, **kwargs):
        raise AssertionError("one-hop special runner should stay disabled on the safe default path")

    monkeypatch.setattr(prior, "_rollout_alpha_grad_family_tbptt_windowed_loss", _forbidden_special_runner)

    class TinyPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Linear(4 + 3 + 1, 3)

        def step(self, obs_t, action_t, reward_t, cache, step_idx, env_info):
            del cache, step_idx, env_info
            return self.net(torch.cat([obs_t[:, :4], action_t[:, :3], reward_t], dim=-1))

    policy = TinyPolicy()
    h_list = prior._sample_batch_hypers(4)
    for h in h_list:
        h["reward_dropout_enabled"] = False
        h["reward_dropout_randomize"] = False
        h["reward_dropout_ratio"] = 0.0
        h["action_noise_train_std"] = 0.2
        h["action_noise_eval_std"] = 0.15
    env_seeds = prior._sample_seed_list(4)
    rollout_seeds = prior._sample_seed_list(4)

    loss, _, stats = prior.rollout_policy_gradient_loss(
        policy_step_fn=policy.step,
        batch_size=4,
        n_samples=6,
        num_features=24,
        device="cpu",
        single_eval_pos=2,
        collect_x=False,
        tbptt_window=2,
        h_list_override=h_list,
        env_seeds_override=env_seeds,
        rollout_seeds_override=rollout_seeds,
        policy_objective_kind=objective_kind,
    )

    assert torch.isfinite(loss)
    if objective_kind == "reinforce":
        assert int(stats["reinforce_enabled"]) == 1
    else:
        assert int(stats["alpha_grad_enabled"]) == 1


@pytest.mark.parametrize("objective_kind", ["reinforce", "alpha_grad"])
def test_environment_prior_family_tbptt_uses_safe_runner_when_one_hop_enabled(monkeypatch, objective_kind):
    _seed_everything(2084)
    config = get_prior_config()
    env_cfg = dict(config["prior"]["environment"])
    env_cfg["family"] = {"distribution": "meta_choice", "choice_values": ["scm"]}
    env_cfg["batch_parallel_backend"] = "torch_vectorized"
    env_cfg["batch_vectorized_grouping"] = "family"
    env_cfg["batch_vectorized_strict_rng_match"] = False
    env_cfg["pg_one_hop_replay_enabled"] = True
    env_cfg["alpha_grad_one_hop_replay_enabled"] = True
    env_cfg["state_dim"] = {"distribution": "uniform_int", "min": 6, "max": 6}
    env_cfg["obs_dim"] = {"distribution": "uniform_int", "min": 4, "max": 4}
    env_cfg["action_dim"] = {"distribution": "uniform_int", "min": 3, "max": 3}
    env_cfg["noise_dim"] = {"distribution": "uniform_int", "min": 5, "max": 5}
    env_cfg["zero_pad_dim"] = {"distribution": "uniform_int", "min": 2, "max": 2}
    prior = EnvironmentPrior(env_cfg)

    def _forbidden_special_runner(*args, **kwargs):
        raise AssertionError("one-hop replay should graft onto the safe TBPTT path")

    monkeypatch.setattr(prior, "_rollout_alpha_grad_family_tbptt_windowed_loss", _forbidden_special_runner)

    class TinyPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Linear(4 + 3 + 1, 3)

        def step(self, obs_t, action_t, reward_t, cache, step_idx, env_info):
            del cache, step_idx, env_info
            return self.net(torch.cat([obs_t[:, :4], action_t[:, :3], reward_t], dim=-1))

    policy = TinyPolicy()
    h_list = prior._sample_batch_hypers(4)
    for h in h_list:
        h["reward_dropout_enabled"] = False
        h["reward_dropout_randomize"] = False
        h["reward_dropout_ratio"] = 0.0
        h["action_noise_train_std"] = 0.2
        h["action_noise_eval_std"] = 0.15
    env_seeds = prior._sample_seed_list(4)
    rollout_seeds = prior._sample_seed_list(4)

    loss, _, stats = prior.rollout_policy_gradient_loss(
        policy_step_fn=policy.step,
        batch_size=4,
        n_samples=6,
        num_features=24,
        device="cpu",
        single_eval_pos=2,
        collect_x=False,
        tbptt_window=2,
        h_list_override=h_list,
        env_seeds_override=env_seeds,
        rollout_seeds_override=rollout_seeds,
        policy_objective_kind=objective_kind,
    )

    assert torch.isfinite(loss)
    assert int(stats["pg_one_hop_replay_enabled"]) == 1
    if objective_kind == "reinforce":
        assert int(stats["reinforce_enabled"]) == 1
    else:
        assert int(stats["alpha_grad_enabled"]) == 1
        assert int(stats["alpha_grad_one_hop_replay_enabled"]) == 1
