import numpy as np
import pytest
import sys
import types
import time

import torch

from ticl.models.encoders import Linear
from ticl.models.tabpfn import TabPFN
from ticl.rl_validation import (
    _concat_cache_batch_dim,
    _score_candidate_actions,
    _slice_cache_batch_dim,
    evaluate_rlpfn_on_gym_envs,
)


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


def test_validation_cache_helpers_normalize_official_eval_flat_cache():
    flat_cache_0 = [
        torch.randn(32),
        torch.randn(2, 4, 4),
        torch.randn(32),
        torch.randn(32),
        torch.randn(2, 4, 4),
        torch.randn(32),
    ]
    flat_cache_1 = [
        torch.randn(32),
        torch.randn(2, 4, 4),
        torch.randn(32),
        torch.randn(32),
        torch.randn(2, 4, 4),
        torch.randn(32),
    ]

    sliced = _slice_cache_batch_dim(flat_cache_0, 0)
    assert isinstance(sliced, list)
    assert len(sliced) == 2
    assert sliced[0][0].shape == (1, 32)
    assert sliced[0][1].shape == (1, 2, 4, 4)
    assert sliced[0][2].shape == (1, 32)

    merged = _concat_cache_batch_dim([flat_cache_0, flat_cache_1])
    assert isinstance(merged, list)
    assert len(merged) == 2
    assert merged[0][0].shape == (2, 32)
    assert merged[0][1].shape == (2, 2, 4, 4)
    assert merged[0][2].shape == (2, 32)


def test_score_candidate_actions_raise_after_candidate_scoring_removal():
    model = _build_small_causal_model()
    candidates = np.array(
        [
            [0.0, 0.0],
            [0.2, -0.1],
            [-0.3, 0.4],
        ],
        dtype=np.float32,
    )
    with pytest.raises(RuntimeError, match="candidate-action scoring has been removed"):
        _score_candidate_actions(
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


class _FakeBox:
    def __init__(self, low, high):
        self.low = np.asarray(low, dtype=np.float32)
        self.high = np.asarray(high, dtype=np.float32)


class _ShapeOnlyActionSpace:
    def __init__(self, shape):
        self.shape = tuple(shape)


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


class _ActionRecordingEnv(_ScriptedEnv):
    def __init__(self, rollout_lengths, rollout_rewards, action_log, obs_dim=4, action_dim=1):
        super().__init__(rollout_lengths, rollout_rewards, obs_dim=obs_dim, action_dim=action_dim)
        self._action_log = action_log

    def step(self, action):
        self._action_log.append(float(np.asarray(action, dtype=np.float32).reshape(-1)[0]))
        return super().step(action)


class _FallbackBoundsEnv(_ActionRecordingEnv):
    def __init__(self, rollout_lengths, rollout_rewards, action_log, obs_dim=4, action_dim=1):
        super().__init__(rollout_lengths, rollout_rewards, action_log, obs_dim=obs_dim, action_dim=action_dim)
        self.action_space = _ShapeOnlyActionSpace((int(action_dim),))


class _RewardInfoEnv(_ScriptedEnv):
    def __init__(self, rollout_lengths, rollout_rewards, rollout_infos, obs_dim=4, action_dim=1):
        super().__init__(rollout_lengths, rollout_rewards, obs_dim=obs_dim, action_dim=action_dim)
        self._rollout_infos = [dict(info) for info in rollout_infos]

    def step(self, action):
        obs, reward, terminated, truncated, _ = super().step(action)
        return obs, reward, terminated, truncated, dict(self._rollout_infos[self._rollout_idx])


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


class _FixedActionPolicyModel(torch.nn.Module):
    def __init__(self, obs_total_dim, action_dim=1, action_value=0.25):
        super().__init__()
        self.x_encoder_type = "split_obs_action"
        self.encoder = types.SimpleNamespace(obs_dim=int(obs_total_dim), action_dim=int(action_dim))
        self.action_dim = int(action_dim)
        self.action_value = float(action_value)

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
        del x_token
        del y_token
        del max_cache_len
        del kv_cache_mode
        del kv_cache_page_size
        del allow_grad_mutable_cache
        del allow_grad_inplace_paged_cache
        batch_size = int(x_token.shape[1])
        out = torch.full((1, batch_size, self.action_dim), self.action_value, dtype=x_token.dtype, device=x_token.device)
        return out, {"step": 1 if kv_cache is None else int(kv_cache.get("step", 0)) + 1}

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
        del phase_t
        del terminal_t
        del max_cache_len
        del kv_cache_mode
        del kv_cache_page_size
        del allow_grad_mutable_cache
        del allow_grad_inplace_paged_cache
        batch_size = int(reward_mask_t.shape[0])
        out = torch.full(
            (1, batch_size, self.action_dim),
            self.action_value,
            dtype=reward_mask_t.dtype,
            device=reward_mask_t.device,
        )
        return out, {"step": 1 if kv_cache is None else int(kv_cache.get("step", 0)) + 1}


class _MinimalValidationModel(torch.nn.Module):
    pass


class _FakePPOValidationPolicy:
    def __init__(self, *, action_dim=1, action_mean=0.5, action_std=0.25):
        self.action_dim = int(action_dim)
        self.action_mean = float(action_mean)
        self.action_std = float(action_std)
        self.batch_sizes = []
        self.phase_batches = []
        self.sample_calls = 0

    def eval(self):
        return self

    def to(self, device):
        del device
        return self

    def make_vectorized_rollout_step_fn(self):
        policy = self

        def _step_fn(obs_t, action_t, reward_t, reward_mask_t, cache, step_idx, env_info):
            del obs_t
            del action_t
            del reward_t
            del reward_mask_t
            del step_idx
            batch_size = int(env_info["phase_t"].shape[0])
            policy.batch_sizes.append(batch_size)
            policy.phase_batches.append(env_info["phase_t"].detach().cpu().reshape(-1).tolist())
            action_mean = torch.full(
                (batch_size, policy.action_dim),
                policy.action_mean,
                dtype=torch.float32,
                device=env_info["phase_t"].device,
            )
            action_std = torch.full_like(action_mean, policy.action_std)
            next_cache = {"step": torch.arange(batch_size, device=action_mean.device, dtype=torch.float32).reshape(-1, 1)}
            if cache is not None and isinstance(cache, dict) and "step" in cache:
                next_cache["step"] = cache["step"] + 1.0
            return {
                "action_mean": action_mean,
                "action_std": action_std,
                "values": torch.zeros((batch_size, 1), dtype=action_mean.dtype, device=action_mean.device),
            }, next_cache

        def _sample_fn(actor_outputs, noise):
            policy.sample_calls += 1
            return actor_outputs["action_mean"] + (noise * actor_outputs["action_std"])

        _step_fn._policy_actor_sample_fn = _sample_fn
        return _step_fn


class _FakeSyncVectorEnv:
    def __init__(self, env_fns, copy=True, observation_mode="same"):
        del observation_mode
        self.copy = bool(copy)
        self.envs = [fn() for fn in env_fns]
        self.num_envs = int(len(self.envs))
        self._observations = None
        self._rewards = np.zeros((self.num_envs,), dtype=np.float32)
        self._terminations = np.zeros((self.num_envs,), dtype=np.bool_)
        self._truncations = np.zeros((self.num_envs,), dtype=np.bool_)
        self._autoreset_envs = np.zeros((self.num_envs,), dtype=np.bool_)
        self.step_calls = 0

    def reset(self, *, seed=None, options=None):
        del options
        if seed is None:
            seed = [None] * self.num_envs
        elif isinstance(seed, int):
            seed = [seed + i for i in range(self.num_envs)]
        observations = []
        infos = {}
        for idx, (env, env_seed) in enumerate(zip(self.envs, seed)):
            obs, info = env.reset(seed=env_seed)
            observations.append(np.asarray(obs, dtype=np.float32))
            for key, value in dict(info).items():
                infos.setdefault(key, np.empty((self.num_envs,), dtype=object))
                infos.setdefault(f"_{key}", np.zeros((self.num_envs,), dtype=np.bool_))
                infos[key][idx] = value
                infos[f"_{key}"][idx] = True
        self._observations = np.stack(observations, axis=0)
        self._rewards.fill(0.0)
        self._terminations.fill(False)
        self._truncations.fill(False)
        self._autoreset_envs.fill(False)
        return np.copy(self._observations) if self.copy else self._observations, infos

    def step(self, actions):
        self.step_calls += 1
        observations = []
        infos = {}
        for idx, (env, action) in enumerate(zip(self.envs, np.asarray(actions))):
            if self._autoreset_envs[idx]:
                obs, info = env.reset(seed=None)
                reward = 0.0
                terminated = False
                truncated = False
            else:
                obs, reward, terminated, truncated, info = env.step(action)
            observations.append(np.asarray(obs, dtype=np.float32))
            self._rewards[idx] = float(reward)
            self._terminations[idx] = bool(terminated)
            self._truncations[idx] = bool(truncated)
            for key, value in dict(info).items():
                infos.setdefault(key, np.empty((self.num_envs,), dtype=object))
                infos.setdefault(f"_{key}", np.zeros((self.num_envs,), dtype=np.bool_))
                infos[key][idx] = value
                infos[f"_{key}"][idx] = True
        self._observations = np.stack(observations, axis=0)
        self._autoreset_envs = np.logical_or(self._terminations, self._truncations)
        return (
            np.copy(self._observations) if self.copy else self._observations,
            np.copy(self._rewards),
            np.copy(self._terminations),
            np.copy(self._truncations),
            infos,
        )

    def close(self):
        for env in self.envs:
            env.close()


def _install_fake_gym(monkeypatch, env_factory, *, with_vector=False):
    gym_mod = types.ModuleType("gymnasium")
    spaces_mod = types.ModuleType("gymnasium.spaces")
    spaces_mod.Box = _FakeBox
    gym_mod.Env = object
    gym_mod.spaces = spaces_mod
    gym_mod.make = lambda env_name: env_factory(env_name)
    if bool(with_vector):
        vector_mod = types.ModuleType("gymnasium.vector")
        vector_mod.SyncVectorEnv = _FakeSyncVectorEnv
        gym_mod.vector = vector_mod
        monkeypatch.setitem(sys.modules, "gymnasium.vector", vector_mod)
    monkeypatch.setitem(sys.modules, "gymnasium", gym_mod)
    monkeypatch.setitem(sys.modules, "gymnasium.spaces", spaces_mod)


def test_evaluate_rlpfn_on_gym_envs_switches_to_eval_rollout_after_context_threshold(monkeypatch):
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
            "rl_validate_envs": "DummyEnv-v0",
            "rl_validate_episodes": 1,
            "rl_validate_max_steps": 32,
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
    assert model.reward_mask_calls == [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    assert model.phase_calls == [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    assert model.terminal_calls == [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    assert model.cache_seen == [False, True, True, True, True, True, True]


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


def test_evaluate_rlpfn_on_gym_envs_requires_ppo_or_policy_step_validation_backend(monkeypatch):
    _install_fake_gym(
        monkeypatch,
        lambda env_name: _ScriptedEnv([2, 3], [1.0, 2.0]),
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
            "rl_validate_envs": "DummyEnv-vRewardTerms",
            "rl_validate_episodes": 1,
            "rl_validate_max_steps": 32,
            "rl_validate_seed": 1,
            "rl_validate_context_lower_bound": 3,
        },
    }

    with pytest.raises(RuntimeError, match="candidate-action scoring has been removed"):
        evaluate_rlpfn_on_gym_envs(model=model, config=cfg)


def test_evaluate_rlpfn_on_gym_envs_reports_reward_term_breakdown_for_policy_step_path(monkeypatch):
    _install_fake_gym(
        monkeypatch,
        lambda env_name: _RewardInfoEnv(
            [2, 3],
            [1.0, 2.0],
            [
                {"reward_forward": 0.25, "reward_ctrl": -0.10},
                {"reward_forward": 0.50, "reward_ctrl": -0.20},
            ],
        ),
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
            "rl_validate_envs": "DummyEnv-vRewardTermsPolicy",
            "rl_validate_episodes": 1,
            "rl_validate_max_steps": 32,
            "rl_validate_action_candidates": 1,
            "rl_validate_seed": 1,
            "rl_validate_context_lower_bound": 3,
        },
    }

    mean_ret, per_env = evaluate_rlpfn_on_gym_envs(model=model, config=cfg)

    assert mean_ret == 6.0
    assert per_env["DummyEnv-vRewardTermsPolicy"]["return"] == 6.0
    assert per_env["DummyEnv-vRewardTermsPolicy"]["reward_forward"] == pytest.approx(1.5)
    assert per_env["DummyEnv-vRewardTermsPolicy"]["reward_forward_mean"] == pytest.approx(1.5)
    assert per_env["DummyEnv-vRewardTermsPolicy"]["reward_ctrl"] == pytest.approx(-0.6)
    assert per_env["DummyEnv-vRewardTermsPolicy"]["reward_ctrl_mean"] == pytest.approx(-0.6)


def test_evaluate_rlpfn_on_gym_envs_does_not_apply_prior_rms_action_transform_to_real_env(monkeypatch):
    action_log = []
    _install_fake_gym(
        monkeypatch,
        lambda env_name: _ActionRecordingEnv([1, 1], [0.0, 0.0], action_log, obs_dim=4, action_dim=1),
    )
    model = _FixedActionPolicyModel(obs_total_dim=8, action_dim=1, action_value=0.25)
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
                # Validation should ignore this synthetic-prior-only transform.
                "reinforce_action_transform": "rms",
                "reinforce_reward_transform": "none",
            },
        },
        "optimizer": {
            "pg_kv_cache_mode": "auto",
            "pg_kv_cache_page_size": None,
        },
        "orchestration": {
            "rl_validate_envs": "DummyEnv-vAction",
            "rl_validate_episodes": 1,
            "rl_validate_max_steps": 8,
            "rl_validate_action_candidates": 1,
            "rl_validate_seed": 1,
            "rl_validate_context_lower_bound": 1,
        },
    }

    mean_ret, per_env = evaluate_rlpfn_on_gym_envs(model=model, config=cfg)

    assert np.isfinite(mean_ret)
    assert per_env["DummyEnv-vAction"]["return"] == 0.0
    assert len(action_log) == 2
    assert action_log == pytest.approx([0.25, 0.25], abs=1e-6)


def test_evaluate_rlpfn_on_gym_envs_falls_back_to_clip_bounds_when_env_has_no_low_high(monkeypatch):
    action_log = []
    _install_fake_gym(
        monkeypatch,
        lambda env_name: _FallbackBoundsEnv([1, 1], [0.0, 0.0], action_log, obs_dim=4, action_dim=1),
    )
    model = _FixedActionPolicyModel(obs_total_dim=8, action_dim=1, action_value=25.0)
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
            "rl_validate_envs": "DummyEnv-vFallbackBounds",
            "rl_validate_episodes": 1,
            "rl_validate_max_steps": 8,
            "rl_validate_action_candidates": 1,
            "rl_validate_seed": 1,
            "rl_validate_context_lower_bound": 1,
        },
    }

    mean_ret, per_env = evaluate_rlpfn_on_gym_envs(model=model, config=cfg)

    assert np.isfinite(mean_ret)
    assert per_env["DummyEnv-vFallbackBounds"]["return"] == 0.0
    assert per_env["DummyEnv-vFallbackBounds"]["fallback_action_bounds"] == 1
    assert len(action_log) == 2
    assert action_log == pytest.approx([10.0, 10.0], abs=1e-6)


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
            "rl_validate_envs": "DummyEnv-v1",
            "rl_validate_episodes": 1,
            "rl_validate_max_steps": 32,
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
    assert model.reward_mask_calls == [1.0] * 9
    assert model.phase_calls == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    assert model.terminal_calls == [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    assert model.cache_seen == [False, True, True, True, True, True, True, True, True]


def test_evaluate_rlpfn_on_gym_envs_ppo_validation_uses_sb3_policy_math_and_batches(monkeypatch):
    action_logs = {"DummyEnv-vPPOA": [], "DummyEnv-vPPOB": []}
    _install_fake_gym(
        monkeypatch,
        lambda env_name: _ActionRecordingEnv(
            [2, 2],
            [0.0, 0.0],
            action_logs[env_name],
            obs_dim=4,
            action_dim=1,
        ),
    )
    model = _MinimalValidationModel()
    fake_policy = _FakePPOValidationPolicy(action_dim=1, action_mean=0.5, action_std=0.25)
    model.__dict__["_validation_sb3_policy_live"] = fake_policy

    import ticl.train as train_mod
    import ticl.rl_validation as rl_validation_mod

    def _forbid_fastpath(*args, **kwargs):
        raise AssertionError("PPO validation must not build the generic policy_step fastpath")

    def _forbid_old_scoring(*args, **kwargs):
        raise AssertionError("PPO validation must not fall back to candidate-action scoring")

    monkeypatch.setattr(train_mod, "_build_policy_step_fn", _forbid_fastpath)
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
            "rl_objective": "ppo",
        },
        "orchestration": {
            "rl_validate_envs": "DummyEnv-vPPOA,DummyEnv-vPPOB",
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
    assert per_env["DummyEnv-vPPOA"]["return_mean"] == 0.0
    assert per_env["DummyEnv-vPPOB"]["return_mean"] == 0.0
    assert max(fake_policy.batch_sizes) == 2
    assert any(any(float(phase) < 1.0 for phase in batch) for batch in fake_policy.phase_batches)
    assert any(any(float(phase) >= 1.0 for phase in batch) for batch in fake_policy.phase_batches)
    assert fake_policy.sample_calls == 2
    assert action_logs["DummyEnv-vPPOA"][-2:] == pytest.approx([0.5, 0.5], abs=1e-6)
    assert action_logs["DummyEnv-vPPOB"][-2:] == pytest.approx([0.5, 0.5], abs=1e-6)


def test_evaluate_rlpfn_on_gym_envs_ppo_validation_vectorizes_same_env_episodes(monkeypatch):
    _install_fake_gym(
        monkeypatch,
        lambda env_name: _ActionRecordingEnv([2, 2], [0.0, 0.0], [], obs_dim=4, action_dim=1),
        with_vector=True,
    )
    model = _MinimalValidationModel()
    fake_policy = _FakePPOValidationPolicy(action_dim=1, action_mean=0.5, action_std=0.25)
    model.__dict__["_validation_sb3_policy_live"] = fake_policy

    import ticl.rl_validation as rl_validation_mod

    original_advance_single = rl_validation_mod._advance_validation_policy_state_with_action
    single_calls = {"count": 0}
    vector_step_calls = {"count": 0}

    def _spy_single(*args, **kwargs):
        single_calls["count"] += 1
        return original_advance_single(*args, **kwargs)

    original_vector = rl_validation_mod._advance_validation_state_actions_vectorized

    def _spy_vector(*args, **kwargs):
        vector_step_calls["count"] += 1
        return original_vector(*args, **kwargs)

    monkeypatch.setattr(rl_validation_mod, "_advance_validation_policy_state_with_action", _spy_single)
    monkeypatch.setattr(rl_validation_mod, "_advance_validation_state_actions_vectorized", _spy_vector)

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
            "rl_objective": "ppo",
        },
        "orchestration": {
            "rl_validate_envs": "DummyEnv-vVectorPPO",
            "rl_validate_episodes": 2,
            "rl_validate_max_steps": 8,
            "rl_validate_action_candidates": 1,
            "rl_validate_seed": 1,
            "rl_validate_context_lower_bound": 1,
            "rl_validate_max_parallel_columns": 2,
        },
    }

    mean_ret, per_env = evaluate_rlpfn_on_gym_envs(model=model, config=cfg)

    assert np.isfinite(mean_ret)
    assert per_env["DummyEnv-vVectorPPO"]["return_mean"] == 0.0
    assert vector_step_calls["count"] > 0
    assert single_calls["count"] == 0


def test_evaluate_rlpfn_on_gym_envs_ppo_validation_rebuilds_saved_policy(monkeypatch):
    action_log = []
    _install_fake_gym(
        monkeypatch,
        lambda env_name: _ActionRecordingEnv([2, 2], [0.0, 0.0], action_log, obs_dim=4, action_dim=1),
    )
    model = _MinimalValidationModel()
    model.__dict__["_validation_ppo_policy_state"] = {"dummy_weight": torch.tensor([1.0])}

    import ticl.train as train_mod
    import ticl.rl_validation as rl_validation_mod

    fake_policy = _FakePPOValidationPolicy(action_dim=1, action_mean=0.25, action_std=0.1)
    builder_calls = {}

    def _fake_builder(*, model, env_cfg, device, num_features, policy_state_dict):
        builder_calls["device"] = device
        builder_calls["num_features"] = int(num_features)
        builder_calls["policy_state_dict"] = policy_state_dict
        return fake_policy

    def _forbid_fastpath(*args, **kwargs):
        raise AssertionError("rebuilt PPO validation policy must bypass generic policy_step validation")

    def _forbid_old_scoring(*args, **kwargs):
        raise AssertionError("rebuilt PPO validation policy must not use candidate-action scoring")

    fake_ppo_mod = types.ModuleType("ticl.sb3_recurrent_ppo")
    fake_ppo_mod.build_validation_recurrent_ppo_policy = _fake_builder
    monkeypatch.setitem(sys.modules, "ticl.sb3_recurrent_ppo", fake_ppo_mod)
    monkeypatch.setattr(train_mod, "_build_policy_step_fn", _forbid_fastpath)
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
            "rl_objective": "ppo",
        },
        "orchestration": {
            "rl_validate_envs": "DummyEnv-vSavedPPO",
            "rl_validate_episodes": 1,
            "rl_validate_max_steps": 8,
            "rl_validate_action_candidates": 1,
            "rl_validate_seed": 1,
            "rl_validate_context_lower_bound": 1,
        },
    }

    mean_ret, per_env = evaluate_rlpfn_on_gym_envs(model=model, config=cfg)

    assert np.isfinite(mean_ret)
    assert per_env["DummyEnv-vSavedPPO"]["return_mean"] == 0.0
    assert builder_calls["device"] == "cpu"
    assert builder_calls["num_features"] == 9
    assert "dummy_weight" in builder_calls["policy_state_dict"]
    assert action_log[-2:] == pytest.approx([0.25, 0.25], abs=1e-6)


def test_evaluate_rlpfn_on_gym_envs_ppo_validation_builds_and_caches_fresh_policy_by_default(monkeypatch):
    action_log = []
    _install_fake_gym(
        monkeypatch,
        lambda env_name: _ActionRecordingEnv([2, 2], [0.0, 0.0], action_log, obs_dim=4, action_dim=1),
    )
    model = _MinimalValidationModel()

    import ticl.train as train_mod
    import ticl.rl_validation as rl_validation_mod

    fake_policy = _FakePPOValidationPolicy(action_dim=1, action_mean=0.75, action_std=0.05)
    builder_calls = {"count": 0}

    def _fake_builder(*, model, env_cfg, device, num_features, policy_state_dict):
        del model
        del env_cfg
        builder_calls["count"] += 1
        builder_calls["device"] = device
        builder_calls["num_features"] = int(num_features)
        builder_calls["policy_state_dict"] = policy_state_dict
        return fake_policy

    def _forbid_fastpath(*args, **kwargs):
        raise AssertionError("PPO validation default path must not build the generic policy_step fastpath")

    def _forbid_old_scoring(*args, **kwargs):
        raise AssertionError("PPO validation default path must not fall back to candidate-action scoring")

    fake_ppo_mod = types.ModuleType("ticl.sb3_recurrent_ppo")
    fake_ppo_mod.build_validation_recurrent_ppo_policy = _fake_builder
    monkeypatch.setitem(sys.modules, "ticl.sb3_recurrent_ppo", fake_ppo_mod)
    monkeypatch.setattr(train_mod, "_build_policy_step_fn", _forbid_fastpath)
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
            "rl_objective": "ppo",
        },
        "orchestration": {
            "rl_validate_envs": "DummyEnv-vFreshPPO",
            "rl_validate_episodes": 1,
            "rl_validate_max_steps": 8,
            "rl_validate_action_candidates": 1,
            "rl_validate_seed": 1,
            "rl_validate_context_lower_bound": 1,
        },
    }

    mean_ret, per_env = evaluate_rlpfn_on_gym_envs(model=model, config=cfg)
    mean_ret_2, per_env_2 = evaluate_rlpfn_on_gym_envs(model=model, config=cfg)

    assert np.isfinite(mean_ret)
    assert np.isfinite(mean_ret_2)
    assert per_env["DummyEnv-vFreshPPO"]["return_mean"] == 0.0
    assert per_env_2["DummyEnv-vFreshPPO"]["return_mean"] == 0.0
    assert builder_calls["count"] == 1
    assert builder_calls["device"] == "cpu"
    assert builder_calls["num_features"] == 9
    assert builder_calls["policy_state_dict"] is None
    assert model.__dict__["_validation_sb3_policy_live"] is fake_policy
    assert len(action_log) == 8
    assert action_log[-2:] == pytest.approx([0.75, 0.75], abs=1e-6)
    assert fake_policy.sample_calls >= 2
