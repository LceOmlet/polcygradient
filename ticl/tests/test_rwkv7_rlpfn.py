from copy import deepcopy
import random
import sys
import types

import numpy as np
import torch

from ticl.model_builder import get_model
from ticl.model_configs import get_model_default_config
from ticl.priors.environment_prior import EnvironmentPrior
from ticl.rl_validation import evaluate_rlpfn_on_gym_envs
from ticl.train import _build_policy_step_fn


def _seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _build_rwkv7_rlpfn_config():
    cfg = deepcopy(get_model_default_config("rlpfn"))
    cfg["transformer"]["backbone"] = "rwkv7"
    cfg["optimizer"]["rl_objective"] = "reinforce"
    return cfg


def _build_small_exact_scm_env_cfg():
    cfg = _build_rwkv7_rlpfn_config()
    env_cfg = deepcopy(cfg["prior"]["environment"])
    env_cfg.update(
        {
            "state_dim": {"distribution": "uniform_int", "min": 8, "max": 8},
            "obs_dim": {"distribution": "uniform_int", "min": 6, "max": 6},
            "action_dim": {"distribution": "uniform_int", "min": 3, "max": 3},
            "noise_dim": {"distribution": "uniform_int", "min": 2, "max": 2},
            "zero_pad_dim": {"distribution": "uniform_int", "min": 0, "max": 0},
            "reward_dropout_randomize": False,
            "reward_dropout_ratio_min": 0.0,
            "reward_dropout_ratio_max": 0.0,
            "action_noise_train_std": 0.05,
            "action_noise_eval_std": 0.03,
        }
    )
    return cfg, env_cfg


class _FakeBox:
    def __init__(self, low, high):
        self.low = np.asarray(low, dtype=np.float32)
        self.high = np.asarray(high, dtype=np.float32)


class _ScriptedEnv:
    def __init__(self, rollout_lengths, rollout_rewards, obs_dim=400, action_dim=30):
        self._rollout_lengths = [int(x) for x in rollout_lengths]
        self._rollout_rewards = [float(x) for x in rollout_rewards]
        self._obs_dim = int(obs_dim)
        self._rollout_idx = -1
        self._step_idx = 0
        self.action_space = _FakeBox(
            low=-np.ones((int(action_dim),), dtype=np.float32),
            high=np.ones((int(action_dim),), dtype=np.float32),
        )

    def reset(self, seed=None):
        del seed
        self._rollout_idx += 1
        self._step_idx = 0
        obs = np.full((self._obs_dim,), float(self._rollout_idx), dtype=np.float32)
        return obs, {}

    def step(self, action):
        del action
        reward = self._rollout_rewards[self._rollout_idx]
        self._step_idx += 1
        terminated = bool(self._step_idx >= self._rollout_lengths[self._rollout_idx])
        obs = np.full(
            (self._obs_dim,),
            float(self._rollout_idx) + 0.1 * float(self._step_idx),
            dtype=np.float32,
        )
        return obs, reward, terminated, False, {}

    def close(self):
        return None


def _install_fake_gym(monkeypatch, env_factory):
    gym_mod = types.ModuleType("gymnasium")
    spaces_mod = types.ModuleType("gymnasium.spaces")
    spaces_mod.Box = _FakeBox
    gym_mod.spaces = spaces_mod
    gym_mod.make = lambda env_name: env_factory(env_name)
    monkeypatch.setitem(sys.modules, "gymnasium", gym_mod)
    monkeypatch.setitem(sys.modules, "gymnasium.spaces", spaces_mod)


def _assemble_split_token(obs_t, action_t, reward_t, reward_mask_t, phase_t, terminal_t, *, obs_dim: int, action_dim: int):
    batch_size = int(obs_t.shape[0])
    obs_slot_dim = int(obs_dim - 2 - int(phase_t is not None) - int(terminal_t is not None))
    x_token = torch.zeros((1, batch_size, obs_dim + action_dim), dtype=obs_t.dtype)
    x_row = x_token[0]
    obs_copy = min(int(obs_t.shape[-1]), obs_slot_dim)
    if obs_copy > 0:
        x_row[:, :obs_copy] = obs_t[:, :obs_copy]
    x_row[:, obs_slot_dim] = reward_t.reshape(batch_size)
    x_row[:, obs_slot_dim + 1] = reward_mask_t.reshape(batch_size)
    if phase_t is not None:
        x_row[:, obs_slot_dim + 2] = phase_t.reshape(batch_size)
    if terminal_t is not None:
        x_row[:, obs_slot_dim + 3] = terminal_t.reshape(batch_size)
    action_start = obs_slot_dim + 2 + int(phase_t is not None) + int(terminal_t is not None)
    action_copy = min(int(action_t.shape[-1]), action_dim)
    if action_copy > 0:
        x_row[:, action_start: action_start + action_copy] = action_t[:, :action_copy]
    y_token = reward_t.reshape(1, batch_size)
    return x_token, y_token


def test_rwkv7_rlpfn_builder_has_correct_action_head_and_5m_budget():
    cfg = _build_rwkv7_rlpfn_config()
    _, model, *_ = get_model(cfg, device="cpu", should_train=False, verbose=False)

    param_count = sum(p.numel() for p in model.parameters())
    assert type(model).__name__ == "RWKV7PFN"
    assert 4_800_000 <= param_count <= 5_600_000
    assert model.policy_action_head_required()
    assert model.has_correct_policy_action_head()
    assert int(model.policy_action_dim) == 30


def test_rwkv7_rlpfn_split_policy_step_matches_generic_policy_step():
    _seed_everything(7)
    cfg = _build_rwkv7_rlpfn_config()
    cfg["transformer"]["emsize"] = 64
    cfg["transformer"]["nlayers"] = 2
    cfg["transformer"]["rwkv_head_size"] = 64
    _, model, *_ = get_model(cfg, device="cpu", should_train=False, verbose=False)
    model.eval()

    obs = torch.randn(2, 400)
    action = torch.randn(2, 30)
    reward = torch.randn(2, 1)
    reward_mask = torch.ones(2, 1)
    phase = torch.tensor([[0.0], [1.0]], dtype=torch.float32)
    terminal = torch.tensor([[0.0], [1.0]], dtype=torch.float32)

    x_token, y_token = _assemble_split_token(
        obs,
        action,
        reward,
        reward_mask,
        phase,
        terminal,
        obs_dim=int(model.encoder.obs_dim),
        action_dim=int(model.encoder.action_dim),
    )
    out_generic, state_generic = model.forward_policy_step(x_token, y_token, kv_cache=None)
    out_split, state_split = model.forward_policy_step_split(
        obs,
        action,
        reward,
        reward_mask,
        phase_t=phase,
        terminal_t=terminal,
        kv_cache=None,
    )

    assert torch.allclose(out_generic, out_split, atol=1e-6, rtol=1e-6)
    assert len(state_generic) == len(state_split) == int(cfg["transformer"]["nlayers"])
    for generic_layer, split_layer in zip(state_generic, state_split):
        for generic_tensor, split_tensor in zip(generic_layer, split_layer):
            assert torch.allclose(generic_tensor, split_tensor, atol=1e-6, rtol=1e-6)


def test_rwkv7_rlpfn_policy_step_reuses_state_cache():
    _seed_everything(11)
    cfg = _build_rwkv7_rlpfn_config()
    cfg["transformer"]["emsize"] = 64
    cfg["transformer"]["nlayers"] = 2
    cfg["transformer"]["rwkv_head_size"] = 64
    _, model, *_ = get_model(cfg, device="cpu", should_train=False, verbose=False)
    model.eval()
    step_fn = _build_policy_step_fn(
        model,
        num_features=int(cfg["prior"]["num_features"]),
        max_cache_len=16,
        kv_cache_mode="immutable",
        kv_cache_page_size=None,
        allow_grad_mutable_cache=False,
        pg_torch_compile=False,
    )

    env_info = {
        "obs_slot_dim": 400,
        "action_slot_dim": 30,
        "action_dim": 30,
        "phase_t": torch.tensor([[0.0], [1.0]], dtype=torch.float32),
        "terminal_t": torch.tensor([[0.0], [0.0]], dtype=torch.float32),
    }
    obs_1 = torch.randn(2, 400)
    action_1 = torch.randn(2, 30)
    reward_1 = torch.randn(2, 1)
    reward_mask = torch.ones(2, 1)
    obs_2 = torch.randn(2, 400)
    action_2 = torch.randn(2, 30)
    reward_2 = torch.randn(2, 1)

    _, cache_1 = step_fn(obs_1, action_1, reward_1, reward_mask, None, 0, env_info)
    _, cache_with_history = step_fn(obs_2, action_2, reward_2, reward_mask, cache_1, 1, env_info)
    _, cache_without_history = step_fn(obs_2, action_2, reward_2, reward_mask, None, 1, env_info)

    assert isinstance(cache_1, list)
    assert len(cache_1) == int(cfg["transformer"]["nlayers"])
    cache_1_nonzero = any(float(t.detach().abs().sum()) > 0.0 for layer_state in cache_1 for t in layer_state)
    cache_diverged = any(
        not torch.allclose(t_hist, t_fresh)
        for layer_hist, layer_fresh in zip(cache_with_history, cache_without_history)
        for t_hist, t_fresh in zip(layer_hist, layer_fresh)
    )
    assert cache_1_nonzero
    assert cache_diverged


def test_rwkv7_rlpfn_exact_scm_reinforce_rollout_backward_is_finite():
    _seed_everything(123)
    cfg, env_cfg = _build_small_exact_scm_env_cfg()
    prior = EnvironmentPrior(env_cfg)
    _, model, *_ = get_model(cfg, device="cpu", should_train=False, verbose=False)
    model.train()
    step_fn = _build_policy_step_fn(
        model,
        num_features=int(cfg["prior"]["num_features"]),
        max_cache_len=16,
        kv_cache_mode="immutable",
        kv_cache_page_size=None,
        allow_grad_mutable_cache=False,
        pg_torch_compile=False,
    )

    _seed_everything(515151)
    h_list = prior._sample_batch_hypers(2)
    for h in h_list:
        h["reward_dropout_enabled"] = False
        h["reward_dropout_randomize"] = False
        h["reward_dropout_ratio"] = 0.0
        h["action_noise_train_std"] = 0.05
        h["action_noise_eval_std"] = 0.03

    loss, rollout, stats = prior.rollout_policy_gradient_loss(
        policy_step_fn=step_fn,
        batch_size=2,
        n_samples=8,
        num_features=int(cfg["prior"]["num_features"]),
        device="cpu",
        single_eval_pos=4,
        collect_x=False,
        h_list_override=[dict(h) for h in h_list],
        env_seeds_override=[17, 29],
        rollout_seeds_override=[101, 211],
        policy_objective_kind="reinforce",
    )
    loss.backward()

    policy_grads = [p.grad.detach() for p in model.policy_action_head.parameters() if p.grad is not None]
    backbone_grads = [p.grad.detach() for p in model.rwkv_core.parameters() if p.grad is not None]

    assert torch.isfinite(loss).item()
    assert torch.isfinite(rollout["rewards"]).all().item()
    assert torch.isfinite(stats["objective"]).item()
    assert policy_grads
    assert backbone_grads
    assert all(torch.isfinite(g).all().item() for g in policy_grads)
    assert all(torch.isfinite(g).all().item() for g in backbone_grads)
    assert sum(float(g.abs().sum()) for g in policy_grads) > 0.0
    assert any(float(g.abs().sum()) > 0.0 for g in backbone_grads)


def test_rwkv7_rlpfn_official_cuda_sequence_forward_runs():
    if not torch.cuda.is_available():
        return

    _seed_everything(321)
    cfg = _build_rwkv7_rlpfn_config()
    cfg["transformer"]["emsize"] = 64
    cfg["transformer"]["nlayers"] = 2
    cfg["transformer"]["rwkv_head_size"] = 64
    _, model, *_ = get_model(cfg, device="cuda", should_train=False, verbose=False)
    model = model.cuda().eval()

    seq_len = 15
    batch_size = 2
    num_features = int(cfg["prior"]["num_features"])
    x = torch.randn(seq_len, batch_size, num_features, device="cuda")
    y = torch.randn(seq_len, batch_size, device="cuda")

    out = model((x, y), single_eval_pos=7)
    assert out.is_cuda
    assert tuple(out.shape) == (seq_len - 7, batch_size, int(model.policy_action_dim))


def test_rwkv7_rlpfn_validation_uses_split_policy_step_state_cache(monkeypatch):
    _install_fake_gym(
        monkeypatch,
        lambda env_name: _ScriptedEnv([2, 2], [1.0, 1.0]),
    )
    cfg = _build_rwkv7_rlpfn_config()
    cfg["transformer"]["emsize"] = 64
    cfg["transformer"]["nlayers"] = 2
    cfg["transformer"]["rwkv_head_size"] = 64
    _, model, *_ = get_model(cfg, device="cpu", should_train=False, verbose=False)
    model.eval()

    import ticl.rl_validation as rl_validation_mod

    def _forbid_old_scoring(*args, **kwargs):
        raise AssertionError("RWKV validation must not fall back to legacy candidate scoring")

    monkeypatch.setattr(rl_validation_mod, "_score_candidate_action_jobs", _forbid_old_scoring)

    split_calls = {"count": 0}
    split_cache_seen = []

    original_split = model.forward_policy_step_split

    def _wrapped_split(
        self,
        obs_t,
        action_t,
        reward_t,
        reward_mask_t,
        phase_t=None,
        terminal_t=None,
        kv_cache=None,
        max_cache_len=None,
        kv_cache_mode="auto",
        kv_cache_page_size=None,
        allow_grad_mutable_cache=False,
        allow_grad_inplace_paged_cache=False,
    ):
        split_calls["count"] += 1
        split_cache_seen.append(kv_cache is not None)
        return original_split(
            obs_t,
            action_t,
            reward_t,
            reward_mask_t,
            phase_t=phase_t,
            terminal_t=terminal_t,
            kv_cache=kv_cache,
            max_cache_len=max_cache_len,
            kv_cache_mode=kv_cache_mode,
            kv_cache_page_size=kv_cache_page_size,
            allow_grad_mutable_cache=allow_grad_mutable_cache,
            allow_grad_inplace_paged_cache=allow_grad_inplace_paged_cache,
        )

    def _forbid_generic(
        self,
        x_token,
        y_token,
        kv_cache=None,
        max_cache_len=None,
        kv_cache_mode="auto",
        kv_cache_page_size=None,
        allow_grad_mutable_cache=False,
        allow_grad_inplace_paged_cache=False,
    ):
        raise AssertionError("RWKV validation should use split policy-step fastpath, not generic policy_step")

    monkeypatch.setattr(model, "forward_policy_step_split", types.MethodType(_wrapped_split, model))
    monkeypatch.setattr(model, "forward_policy_step", types.MethodType(_forbid_generic, model))

    val_cfg = {
        "device": "cpu",
        "prior": {
            "num_features": int(cfg["prior"]["num_features"]),
            "environment": {
                "obs_slot_dim": 400,
                "action_slot_dim": 30,
                "terminal_reset_enabled": True,
                "init_action_std": 0.0,
                "action_noise_train_std": 0.0,
                "action_noise_eval_std": 0.0,
                "reinforce_action_transform": "none",
                "reinforce_reward_transform": "none",
            },
        },
        "optimizer": {
            "pg_kv_cache_mode": "auto",
            "pg_kv_cache_page_size": None,
        },
        "orchestration": {
            "rl_validate_envs": "DummyEnv-vRWKV",
            "rl_validate_episodes": 1,
            "rl_validate_max_steps": 8,
            "rl_validate_action_candidates": 1,
            "rl_validate_seed": 1,
            "rl_validate_context_lower_bound": 1,
        },
    }

    mean_ret, per_env = evaluate_rlpfn_on_gym_envs(model=model, config=val_cfg)

    assert np.isfinite(mean_ret)
    assert per_env["DummyEnv-vRWKV"]["return"] == 2.0
    assert split_calls["count"] > 0
    assert split_cache_seen[0] is False
    assert any(split_cache_seen[1:])
