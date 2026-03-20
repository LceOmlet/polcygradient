import numpy as np
import pytest
import sys
import types
import time

import torch

from ticl.models.encoders import Linear
from ticl.models.tabpfn import TabPFN
from ticl.rl_validation import _score_candidate_actions, evaluate_rlpfn_on_gym_envs


def _build_small_causal_model():
    model = TabPFN(
        n_out=1,
        n_features=12,
        emsize=32,
        nhead=1,
        nhid_factor=2,
        nlayers=1,
        y_encoder_layer=Linear(1, emsize=32),
        classification_task=False,
        y_encoder="linear",
        x_encoder_type="single",
        single_eval_causal=True,
    )
    model.eval()
    return model


def _build_small_split_rlpfn_like_model():
    model = TabPFN(
        n_out=1,
        n_features=9,
        emsize=32,
        nhead=1,
        nhid_factor=2,
        nlayers=1,
        y_encoder_layer=Linear(1, emsize=32),
        classification_task=False,
        y_encoder="linear",
        x_encoder_type="split_obs_action",
        x_obs_dim=8,
        x_action_dim=1,
        single_eval_causal=True,
    )
    model.eval()
    return model


def test_score_candidate_actions_bootstrap_step_no_prefix_no_crash():
    model = _build_small_causal_model()
    candidates = np.array(
        [
            [0.0, 0.0],
            [0.2, -0.1],
            [-0.3, 0.4],
        ],
        dtype=np.float32,
    )
    scores = _score_candidate_actions(
        model=model,
        device="cpu",
        x_hist=[],
        y_hist=[],
        obs=np.zeros((4,), dtype=np.float32),
        prev_reward=0.0,
        action_candidates=candidates,
        obs_slot_dim=8,
        action_slot_dim=2,
        num_features=12,
    )

    assert scores.shape == (3,)
    assert np.all(np.isfinite(scores))
    assert np.allclose(scores, 0.0)


class _FakeBox:
    def __init__(self, low, high):
        self.low = np.asarray(low, dtype=np.float32)
        self.high = np.asarray(high, dtype=np.float32)


class _ScriptedEnv:
    def __init__(self, rollout_lengths, rollout_rewards, obs_dim=4, action_dim=1):
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


class _PhaseRecordingModel:
    def __init__(self, phase_idx, terminal_idx):
        self.training = False
        self.phase_calls = []
        self.terminal_calls = []
        self.call_widths = []
        self.phase_idx = int(phase_idx)
        self.terminal_idx = int(terminal_idx)

    def eval(self):
        self.training = False
        return self

    def train(self, mode=True):
        self.training = bool(mode)
        return self

    def __call__(self, inputs, single_eval_pos=None):
        del single_eval_pos
        x_tensor, _ = inputs
        self.call_widths.append(int(x_tensor.shape[1]))
        last_token = x_tensor[-1, 0]
        self.phase_calls.append(float(last_token[self.phase_idx].item()))
        self.terminal_calls.append(float(last_token[self.terminal_idx].item()))
        return torch.zeros((1, x_tensor.shape[1], 1), dtype=x_tensor.dtype, device=x_tensor.device)


class _PolicyStepRecordingModel(torch.nn.Module):
    def __init__(self, obs_total_dim, action_dim=1):
        super().__init__()
        self.x_encoder_type = "split_obs_action"
        self.encoder = types.SimpleNamespace(obs_dim=int(obs_total_dim), action_dim=int(action_dim))
        self.action_dim = int(action_dim)
        self.reward_mask_calls = []
        self.phase_calls = []
        self.terminal_calls = []
        self.cache_seen = []

    def forward_policy_step(
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
        del y_token
        del max_cache_len
        del kv_cache_mode
        del kv_cache_page_size
        del allow_grad_mutable_cache
        del allow_grad_inplace_paged_cache
        out = torch.zeros((1, x_token.shape[1], self.action_dim), dtype=x_token.dtype, device=x_token.device)
        return out, kv_cache

    def forward_policy_step_split(
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
        del obs_t
        del action_t
        del reward_t
        del max_cache_len
        del kv_cache_mode
        del kv_cache_page_size
        del allow_grad_mutable_cache
        del allow_grad_inplace_paged_cache
        self.reward_mask_calls.append(float(reward_mask_t.reshape(-1)[0].item()))
        self.phase_calls.append(float(phase_t.reshape(-1)[0].item()) if phase_t is not None else float("nan"))
        self.terminal_calls.append(float(terminal_t.reshape(-1)[0].item()) if terminal_t is not None else float("nan"))
        self.cache_seen.append(kv_cache is not None)
        batch_size = int(reward_mask_t.shape[0])
        out = torch.zeros((1, batch_size, self.action_dim), dtype=reward_mask_t.dtype, device=reward_mask_t.device)
        return out, {"step": len(self.reward_mask_calls)}


class _WrappedPolicyModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


def _install_fake_gym(monkeypatch, env_factory):
    gym_mod = types.ModuleType("gymnasium")
    spaces_mod = types.ModuleType("gymnasium.spaces")
    spaces_mod.Box = _FakeBox
    gym_mod.spaces = spaces_mod
    gym_mod.make = lambda env_name: env_factory(env_name)
    monkeypatch.setitem(sys.modules, "gymnasium", gym_mod)
    monkeypatch.setitem(sys.modules, "gymnasium.spaces", spaces_mod)


def test_evaluate_rlpfn_on_gym_envs_switches_to_eval_rollout_after_context_threshold(monkeypatch):
    _install_fake_gym(
        monkeypatch,
        lambda env_name: _ScriptedEnv([3, 4], [1.0, 2.0]),
    )
    model = _PhaseRecordingModel(phase_idx=6, terminal_idx=7)
    cfg = {
        "device": "cpu",
        "prior": {
            "num_features": 9,
            "environment": {
                "obs_slot_dim": 4,
                "action_slot_dim": 1,
                "terminal_reset_enabled": True,
            },
        },
        "orchestration": {
            "rl_validate_envs": "DummyEnv-v0",
            "rl_validate_episodes": 1,
            "rl_validate_max_steps": 32,
            "rl_validate_action_candidates": 1,
            "rl_validate_seed": 1,
            "rl_validate_context_lower_bound": 5,
        },
    }

    mean_ret, per_env = evaluate_rlpfn_on_gym_envs(model=model, config=cfg)

    assert mean_ret == 8.0
    assert per_env["DummyEnv-v0"]["return_mean"] == 8.0
    assert per_env["DummyEnv-v0"]["len_mean"] == 4.0
    assert per_env["DummyEnv-v0"]["context_len_before_eval_mean"] == 3.0
    assert per_env["DummyEnv-v0"]["explore_rollout_count_mean"] == 1.0
    assert per_env["DummyEnv-v0"]["explore_rollout_len_mean"] == 3.0
    assert model.phase_calls == [0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    assert model.terminal_calls == [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]


def test_evaluate_rlpfn_on_gym_envs_policy_step_path_reuses_cache_and_reports_single_return(monkeypatch):
    _install_fake_gym(
        monkeypatch,
        lambda env_name: _ScriptedEnv([3, 4], [1.0, 2.0]),
    )
    model = _PolicyStepRecordingModel(obs_total_dim=8, action_dim=1)
    cfg = {
        "device": "cpu",
        "prior": {
            "num_features": 9,
            "environment": {
                "obs_slot_dim": 4,
                "action_slot_dim": 1,
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
            "rl_validate_envs": "DummyEnv-vPolicy",
            "rl_validate_episodes": 1,
            "rl_validate_max_steps": 32,
            "rl_validate_action_candidates": 1,
            "rl_validate_seed": 1,
            "rl_validate_context_lower_bound": 5,
        },
    }

    mean_ret, per_env = evaluate_rlpfn_on_gym_envs(model=model, config=cfg)

    assert mean_ret == 8.0
    assert per_env["DummyEnv-vPolicy"]["return_mean"] == 8.0
    assert per_env["DummyEnv-vPolicy"]["return"] == 8.0
    assert per_env["DummyEnv-vPolicy"]["len"] == 4.0
    assert model.reward_mask_calls == [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    assert model.phase_calls == [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    assert model.terminal_calls == [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    assert model.cache_seen == [False, True, True, True, True, True, True]


def test_evaluate_rlpfn_on_gym_envs_small_rlpfn_model_uses_policy_step_validation(monkeypatch):
    _install_fake_gym(
        monkeypatch,
        lambda env_name: _ScriptedEnv([2, 2], [1.0, 1.0]),
    )
    model = _build_small_split_rlpfn_like_model()

    import ticl.train as train_mod
    import ticl.rl_validation as rl_validation_mod

    original_builder = train_mod._build_policy_step_fn
    builder_calls = {"count": 0}

    def _spy_builder(*args, **kwargs):
        builder_calls["count"] += 1
        return original_builder(*args, **kwargs)

    def _forbid_old_scoring(*args, **kwargs):
        raise AssertionError("old candidate-scoring validation path should not run for RLPFN policy-step models")

    monkeypatch.setattr(train_mod, "_build_policy_step_fn", _spy_builder)
    monkeypatch.setattr(rl_validation_mod, "_score_candidate_action_jobs", _forbid_old_scoring)

    cfg = {
        "device": "cpu",
        "prior": {
            "num_features": 9,
            "environment": {
                "obs_slot_dim": 4,
                "action_slot_dim": 1,
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
            "rl_validate_envs": "DummyEnv-vSplit",
            "rl_validate_episodes": 1,
            "rl_validate_max_steps": 8,
            "rl_validate_action_candidates": 1,
            "rl_validate_seed": 1,
            "rl_validate_context_lower_bound": 1,
        },
    }

    t0 = time.perf_counter()
    mean_ret, per_env = evaluate_rlpfn_on_gym_envs(model=model, config=cfg)
    elapsed_s = time.perf_counter() - t0

    assert builder_calls["count"] == 1
    assert np.isfinite(mean_ret)
    assert per_env["DummyEnv-vSplit"]["return"] == 2.0
    assert elapsed_s >= 0.0


def test_evaluate_rlpfn_on_gym_envs_paged_policy_step_validation_sets_cache_capacity(monkeypatch):
    _install_fake_gym(
        monkeypatch,
        lambda env_name: _ScriptedEnv([2, 2], [1.0, 1.0]),
    )
    model = _build_small_split_rlpfn_like_model()

    import ticl.train as train_mod

    original_builder = train_mod._build_policy_step_fn
    builder_kwargs = {}

    def _spy_builder(*args, **kwargs):
        builder_kwargs.update(kwargs)
        return original_builder(*args, **kwargs)

    monkeypatch.setattr(train_mod, "_build_policy_step_fn", _spy_builder)

    cfg = {
        "device": "cpu",
        "prior": {
            "num_features": 9,
            "environment": {
                "obs_slot_dim": 4,
                "action_slot_dim": 1,
                "terminal_reset_enabled": True,
                "init_action_std": 0.0,
                "action_noise_train_std": 0.0,
                "action_noise_eval_std": 0.0,
                "reinforce_action_transform": "none",
                "reinforce_reward_transform": "none",
            },
        },
        "optimizer": {
            "pg_kv_cache_mode": "paged",
            "pg_kv_cache_page_size": 8,
        },
        "orchestration": {
            "rl_validate_envs": "DummyEnv-vPaged",
            "rl_validate_episodes": 1,
            "rl_validate_max_steps": 8,
            "rl_validate_action_candidates": 1,
            "rl_validate_seed": 1,
            "rl_validate_context_lower_bound": 1,
        },
    }

    mean_ret, per_env = evaluate_rlpfn_on_gym_envs(model=model, config=cfg)

    assert builder_kwargs["kv_cache_mode"] == "paged"
    assert builder_kwargs["kv_cache_page_size"] == 8
    assert builder_kwargs["max_cache_len"] == 17
    assert np.isfinite(mean_ret)
    assert per_env["DummyEnv-vPaged"]["return"] == 2.0


def test_evaluate_rlpfn_on_gym_envs_requires_policy_action_head_for_split_rlpfn(monkeypatch):
    _install_fake_gym(
        monkeypatch,
        lambda env_name: _ScriptedEnv([2, 2], [1.0, 1.0]),
    )
    model = _build_small_split_rlpfn_like_model()
    model.policy_action_head = None

    import ticl.rl_validation as rl_validation_mod

    def _forbid_old_scoring(*args, **kwargs):
        raise AssertionError("validation must fail loudly instead of falling back to legacy candidate scoring")

    monkeypatch.setattr(rl_validation_mod, "_score_candidate_action_jobs", _forbid_old_scoring)

    cfg = {
        "device": "cpu",
        "prior": {
            "num_features": 9,
            "environment": {
                "obs_slot_dim": 4,
                "action_slot_dim": 1,
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
            "rl_validate_envs": "DummyEnv-vSplit",
            "rl_validate_episodes": 1,
            "rl_validate_max_steps": 8,
            "rl_validate_action_candidates": 1,
            "rl_validate_seed": 1,
            "rl_validate_context_lower_bound": 1,
        },
    }

    with pytest.raises(ValueError, match="policy_action_head"):
        evaluate_rlpfn_on_gym_envs(model=model, config=cfg)


def test_evaluate_rlpfn_on_gym_envs_wrapped_split_model_uses_policy_step_validation(monkeypatch):
    _install_fake_gym(
        monkeypatch,
        lambda env_name: _ScriptedEnv([2, 2], [1.0, 1.0]),
    )
    wrapped_model = _WrappedPolicyModel(_build_small_split_rlpfn_like_model())

    import ticl.rl_validation as rl_validation_mod

    def _forbid_old_scoring(*args, **kwargs):
        raise AssertionError("wrapped split model validation should not fall back to legacy candidate scoring")

    monkeypatch.setattr(rl_validation_mod, "_score_candidate_action_jobs", _forbid_old_scoring)

    cfg = {
        "device": "cpu",
        "prior": {
            "num_features": 9,
            "environment": {
                "obs_slot_dim": 4,
                "action_slot_dim": 1,
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
            "rl_validate_envs": "DummyEnv-vSplit",
            "rl_validate_episodes": 1,
            "rl_validate_max_steps": 8,
            "rl_validate_action_candidates": 1,
            "rl_validate_seed": 1,
            "rl_validate_context_lower_bound": 1,
        },
    }

    mean_ret, per_env = evaluate_rlpfn_on_gym_envs(model=wrapped_model, config=cfg)

    assert np.isfinite(mean_ret)
    assert per_env["DummyEnv-vSplit"]["return"] == 2.0


def test_evaluate_rlpfn_on_gym_envs_uses_mean_explore_rollout_length_for_threshold(monkeypatch):
    _install_fake_gym(
        monkeypatch,
        lambda env_name: _ScriptedEnv([2, 4, 3], [1.0, 1.5, 5.0]),
    )
    model = _PhaseRecordingModel(phase_idx=6, terminal_idx=7)
    cfg = {
        "device": "cpu",
        "prior": {
            "num_features": 9,
            "environment": {
                "obs_slot_dim": 4,
                "action_slot_dim": 1,
                "terminal_reset_enabled": True,
            },
        },
        "orchestration": {
            "rl_validate_envs": "DummyEnv-v1",
            "rl_validate_episodes": 1,
            "rl_validate_max_steps": 32,
            "rl_validate_action_candidates": 1,
            "rl_validate_seed": 1,
            "rl_validate_context_lower_bound": 6,
        },
    }

    mean_ret, per_env = evaluate_rlpfn_on_gym_envs(model=model, config=cfg)

    assert mean_ret == 15.0
    assert per_env["DummyEnv-v1"]["return_mean"] == 15.0
    assert per_env["DummyEnv-v1"]["len_mean"] == 3.0
    assert per_env["DummyEnv-v1"]["context_len_before_eval_mean"] == 6.0
    assert per_env["DummyEnv-v1"]["explore_rollout_count_mean"] == 2.0
    assert per_env["DummyEnv-v1"]["explore_rollout_len_mean"] == 3.0
    assert model.phase_calls == [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    assert model.terminal_calls == [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]


def test_evaluate_rlpfn_on_gym_envs_batches_across_envs_when_cap_allows(monkeypatch):
    _install_fake_gym(
        monkeypatch,
        lambda env_name: _ScriptedEnv([2, 2], [1.0, 1.0]),
    )
    model = _PhaseRecordingModel(phase_idx=6, terminal_idx=7)
    cfg = {
        "device": "cpu",
        "prior": {
            "num_features": 9,
            "environment": {
                "obs_slot_dim": 4,
                "action_slot_dim": 1,
                "terminal_reset_enabled": True,
            },
        },
        "orchestration": {
            "rl_validate_envs": "DummyEnv-vA,DummyEnv-vB",
            "rl_validate_episodes": 1,
            "rl_validate_max_steps": 8,
            "rl_validate_action_candidates": 1,
            "rl_validate_seed": 1,
            "rl_validate_context_lower_bound": 1,
            "rl_validate_max_parallel_columns": 2,
        },
    }

    mean_ret, per_env = evaluate_rlpfn_on_gym_envs(model=model, config=cfg)

    assert np.isfinite(mean_ret)
    assert per_env["DummyEnv-vA"]["return_mean"] == 2.0
    assert per_env["DummyEnv-vB"]["return_mean"] == 2.0
    assert max(model.call_widths) == 2


def test_evaluate_rlpfn_on_gym_envs_respects_parallel_column_cap(monkeypatch):
    _install_fake_gym(
        monkeypatch,
        lambda env_name: _ScriptedEnv([2, 2], [1.0, 1.0]),
    )
    model = _PhaseRecordingModel(phase_idx=6, terminal_idx=7)
    cfg = {
        "device": "cpu",
        "prior": {
            "num_features": 9,
            "environment": {
                "obs_slot_dim": 4,
                "action_slot_dim": 1,
                "terminal_reset_enabled": True,
            },
        },
        "orchestration": {
            "rl_validate_envs": "DummyEnv-vC,DummyEnv-vD",
            "rl_validate_episodes": 1,
            "rl_validate_max_steps": 8,
            "rl_validate_action_candidates": 1,
            "rl_validate_seed": 1,
            "rl_validate_context_lower_bound": 1,
            "rl_validate_max_parallel_columns": 1,
        },
    }

    mean_ret, per_env = evaluate_rlpfn_on_gym_envs(model=model, config=cfg)

    assert np.isfinite(mean_ret)
    assert per_env["DummyEnv-vC"]["return_mean"] == 2.0
    assert per_env["DummyEnv-vD"]["return_mean"] == 2.0
    assert max(model.call_widths) == 1
