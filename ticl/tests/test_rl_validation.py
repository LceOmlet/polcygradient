import numpy as np
import sys
import types

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
        last_token = x_tensor[-1, 0]
        self.phase_calls.append(float(last_token[self.phase_idx].item()))
        self.terminal_calls.append(float(last_token[self.terminal_idx].item()))
        return torch.zeros((1, x_tensor.shape[1], 1), dtype=x_tensor.dtype, device=x_tensor.device)


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
