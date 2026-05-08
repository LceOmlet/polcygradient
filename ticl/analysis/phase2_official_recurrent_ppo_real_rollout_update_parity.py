#!/usr/bin/env python3
"""Minimal real rollout/update parity for the PPO-only contract.

This probe intentionally avoids prior/gym environment construction.  It uses a
scripted Box VecEnv only to exercise the official SB3 VecNormalize update order,
then compares the resulting rollout buffers, GAE/value targets, PPO scalar
losses, and clipped gradients against the local MaskedRecurrentPPO path with all
custom objectives disabled and objective_mask == official valid mask.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch as th
from gymnasium import spaces


def _patch_cv2_for_vendored_sb3() -> None:
    try:
        import cv2  # type: ignore
    except Exception:
        return
    if hasattr(cv2, "ocl"):
        return

    class _OpenCLShim:
        @staticmethod
        def setUseOpenCL(flag: bool) -> None:
            del flag

    cv2.ocl = _OpenCLShim()  # type: ignore[attr-defined]


_patch_cv2_for_vendored_sb3()

from stable_baselines3.common.running_mean_std import RunningMeanStd  # noqa: E402
from stable_baselines3.common.utils import explained_variance  # noqa: E402
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvObs, VecEnvStepReturn  # noqa: E402
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize  # noqa: E402
from sb3_contrib.common.recurrent.buffers import RecurrentRolloutBuffer  # noqa: E402
from sb3_contrib.common.recurrent.type_aliases import RNNStates  # noqa: E402
from sb3_contrib.ppo_recurrent.ppo_recurrent import RecurrentPPO  # noqa: E402

from ticl.analysis.phase2_gym_prior_pack_training_runner import (  # noqa: E402
    _apply_observation_normalization_if_enabled,
)
from ticl.sb3_recurrent_ppo import (  # noqa: E402
    MaskedRecurrentFlatBatchSamples,
    MaskedRecurrentPPO,
    MaskedRecurrentRolloutBuffer,
    MaskedRecurrentRolloutBufferSamples,
    _sb3_vecnormalize_reward_step,
)


class _CaptureLogger:
    def __init__(self) -> None:
        self.rows: dict[str, Any] = {}

    def record(self, key: str, value: Any, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        if isinstance(value, th.Tensor):
            value = value.detach().cpu().item()
        if isinstance(value, np.generic):
            value = value.item()
        self.rows[str(key)] = value


class _ScriptedVecEnv(VecEnv):
    def __init__(self, *, obs_seq: np.ndarray, rewards: np.ndarray, dones: np.ndarray, action_dim: int) -> None:
        self.obs_seq = np.asarray(obs_seq, dtype=np.float32)
        self.rewards = np.asarray(rewards, dtype=np.float32)
        self.dones = np.asarray(dones, dtype=bool)
        self._step_idx = 0
        self._pending_actions: np.ndarray | None = None
        n_envs = int(self.rewards.shape[1])
        obs_dim = int(self.obs_seq.shape[-1])
        super().__init__(
            num_envs=n_envs,
            observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32),
            action_space=spaces.Box(low=-1.0, high=1.0, shape=(int(action_dim),), dtype=np.float32),
        )

    def reset(self) -> VecEnvObs:
        self._step_idx = 0
        return self.obs_seq[0].copy()

    def step_async(self, actions: np.ndarray) -> None:
        self._pending_actions = np.asarray(actions, dtype=np.float32)

    def step_wait(self) -> VecEnvStepReturn:
        del self._pending_actions
        if self._step_idx >= int(self.rewards.shape[0]):
            raise RuntimeError("Scripted VecEnv exhausted.")
        obs = self.obs_seq[self._step_idx + 1].copy()
        rewards = self.rewards[self._step_idx].copy()
        dones = self.dones[self._step_idx].copy()
        infos = [{} for _ in range(int(self.num_envs))]
        self._step_idx += 1
        return obs, rewards, dones, infos

    def close(self) -> None:
        return None

    def get_attr(self, attr_name: str, indices=None) -> list[Any]:
        del indices
        if attr_name == "render_mode":
            return [None for _ in range(int(self.num_envs))]
        raise AttributeError(attr_name)

    def set_attr(self, attr_name: str, value: Any, indices=None) -> None:
        del attr_name, value, indices
        return None

    def env_method(self, method_name: str, *method_args: Any, indices=None, **method_kwargs: Any) -> list[Any]:
        del method_name, method_args, indices, method_kwargs
        return [None for _ in range(int(self.num_envs))]

    def env_is_wrapped(self, wrapper_class: type, indices=None) -> list[bool]:
        del wrapper_class, indices
        return [False for _ in range(int(self.num_envs))]


class _ObsNormAlgo:
    def __init__(self, obs_dim: int, *, clip_obs: float, epsilon: float) -> None:
        self._rwkv_sb3_observation_normalization_enabled = True
        self._rwkv_sb3_observation_normalization_clip = float(clip_obs)
        self._rwkv_sb3_observation_normalization_epsilon = float(epsilon)
        self._rwkv_sb3_observation_normalization_mean = np.zeros((int(obs_dim),), dtype=np.float64)
        self._rwkv_sb3_observation_normalization_var = np.ones((int(obs_dim),), dtype=np.float64)
        self._rwkv_sb3_observation_normalization_count = np.full((int(obs_dim),), 1e-4, dtype=np.float64)


class _OneBatchBuffer:
    def __init__(self, sample: Any, *, values_np: np.ndarray, returns_np: np.ndarray, mask_np: np.ndarray) -> None:
        self._sample = sample
        self.buffer_size = int(values_np.reshape(-1).shape[0])
        self.n_envs = 1
        self.values = values_np.reshape(-1, 1).astype(np.float32)
        self.returns = returns_np.reshape(-1, 1).astype(np.float32)
        self.objective_masks = mask_np.reshape(-1, 1).astype(np.float32)

    def get(self, batch_size: int):
        del batch_size
        yield self._sample

    def get_gpu_flat(self, batch_size: int):
        del batch_size
        yield self._sample


class _ParametricPolicy:
    def __init__(self, *, lr: float) -> None:
        self.param = th.nn.Parameter(th.tensor([0.17, -0.23, 0.31], dtype=th.float32))
        self.optimizer = th.optim.SGD([self.param], lr=float(lr))

    def parameters(self):
        return [self.param]

    def set_training_mode(self, mode: bool) -> None:
        del mode

    def _compute(self, observations: th.Tensor, actions: th.Tensor) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        obs = observations.to(dtype=th.float32).reshape(int(observations.shape[0]), -1)
        act = actions.to(dtype=th.float32).reshape(int(actions.shape[0]), -1)
        obs_feat = (0.13 * obs[:, 0]) + (0.01 * obs.sum(dim=1))
        act_feat = 0.04 * act.sum(dim=1)
        values = 0.15 + 0.20 * obs_feat - 0.05 * act_feat + self.param[0] * (0.30 + 0.02 * obs_feat)
        log_prob = -0.40 + 0.05 * obs_feat + 0.07 * act_feat + self.param[1] * (0.20 - 0.03 * act_feat)
        entropy = 0.70 + 0.02 * obs_feat.abs() + self.param[2] * (0.10 + 0.01 * obs_feat)
        return values, log_prob, entropy

    def evaluate_actions(self, observations, actions, lstm_states, episode_starts):
        del lstm_states, episode_starts
        return self._compute(observations, actions)

    def evaluate_actions_with_hidden(
        self,
        observations,
        actions,
        lstm_states,
        episode_starts,
        *,
        action_masks=None,
    ):
        del lstm_states, episode_starts, action_masks
        values, log_prob, entropy = self._compute(observations, actions)
        return {"value_logits": values, "values": values, "log_prob": log_prob, "entropy": entropy}

    def evaluate_actions_with_hidden_flat(
        self,
        obs,
        actions,
        *,
        seq_lengths,
        action_masks=None,
        include_normalized_q_logits: bool = False,
        flow_matching_xt=None,
        flow_matching_t=None,
    ):
        del seq_lengths, action_masks, include_normalized_q_logits, flow_matching_xt, flow_matching_t
        values, log_prob, entropy = self._compute(obs, actions)
        return {"value_logits": values, "values": values, "log_prob": log_prob, "entropy": entropy}

    def compute_value_loss_from_logits(self, *, value_logits, targets, objective_mask, value_target_bucket_idx=None):
        del value_target_bucket_idx
        mask = objective_mask.to(device=value_logits.device, dtype=th.bool).reshape(-1)
        return th.mean(((targets.reshape(-1) - value_logits.reshape(-1)) ** 2)[mask])


def _max_abs(a: Any, b: Any) -> float:
    return float(np.max(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))


def _tensor_max_abs(a: th.Tensor, b: th.Tensor) -> float:
    return float((a.detach().cpu().to(dtype=th.float64) - b.detach().cpu().to(dtype=th.float64)).abs().max().item())


def _finite_float(value: Any) -> float:
    return float(np.asarray(value, dtype=np.float64).reshape(()))


def _scripted_data(seed: int, *, n_steps: int, n_envs: int, obs_dim: int, act_dim: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    obs_seq = rng.normal(loc=0.0, scale=1.0, size=(n_steps + 1, n_envs, obs_dim)).astype(np.float32)
    obs_seq += np.linspace(-0.4, 0.6, n_steps + 1, dtype=np.float32).reshape(-1, 1, 1)
    rewards = rng.normal(loc=0.2, scale=1.3, size=(n_steps, n_envs)).astype(np.float32)
    rewards[1::3] += np.asarray([0.5, -0.25, 0.75], dtype=np.float32)[:n_envs].reshape(1, -1)
    dones = np.zeros((n_steps, n_envs), dtype=bool)
    if n_steps >= 5 and n_envs >= 2:
        dones[2, 1] = True
    if n_steps >= 7:
        dones[5, 0] = True
    if n_steps >= 8 and n_envs >= 3:
        dones[6, 2] = True
    actions = rng.normal(loc=0.0, scale=0.35, size=(n_steps, n_envs, act_dim)).astype(np.float32)
    values = rng.normal(loc=-0.1, scale=0.65, size=(n_steps, n_envs)).astype(np.float32)
    log_probs = rng.normal(loc=-0.45, scale=0.25, size=(n_steps, n_envs)).astype(np.float32)
    last_values = rng.normal(loc=0.05, scale=0.55, size=(n_envs,)).astype(np.float32)
    return {
        "obs_seq": obs_seq,
        "rewards": rewards,
        "dones": dones,
        "actions": actions,
        "values": values,
        "log_probs": log_probs,
        "last_values": last_values,
    }


def _collect_buffers(seed: int) -> dict[str, Any]:
    n_steps, n_envs, obs_dim, act_dim = 9, 3, 5, 2
    gamma, gae_lambda = 0.97, 0.83
    clip_obs, clip_reward, epsilon = 10.0, 10.0, 1e-8
    data = _scripted_data(seed, n_steps=n_steps, n_envs=n_envs, obs_dim=obs_dim, act_dim=act_dim)
    obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
    act_space = spaces.Box(low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32)
    hidden_state_shape = (n_steps, 1, n_envs, 4)
    official_buffer = RecurrentRolloutBuffer(
        n_steps,
        obs_space,
        act_space,
        hidden_state_shape,
        device="cpu",
        gae_lambda=gae_lambda,
        gamma=gamma,
        n_envs=n_envs,
    )
    custom_buffer = MaskedRecurrentRolloutBuffer(
        n_steps,
        obs_space,
        act_space,
        hidden_state_shape,
        device="cpu",
        gae_lambda=gae_lambda,
        gamma=gamma,
        n_envs=n_envs,
        next_state_dim=obs_dim,
    )
    custom_buffer._value_target_space = "raw"
    custom_buffer._actor_gae_space = "raw"
    custom_buffer._actor_baseline_mode = "learned"
    custom_buffer._actor_objective_mode = "tokenwise"
    custom_buffer._enable_q_target_cache = False
    custom_buffer.set_deterministic_batch_plan(True)

    venv = _ScriptedVecEnv(
        obs_seq=data["obs_seq"],
        rewards=data["rewards"],
        dones=data["dones"],
        action_dim=act_dim,
    )
    vecnorm = VecNormalize(
        venv,
        training=True,
        norm_obs=True,
        norm_reward=True,
        clip_obs=clip_obs,
        clip_reward=clip_reward,
        gamma=gamma,
        epsilon=epsilon,
    )
    obs_algo = _ObsNormAlgo(obs_dim, clip_obs=clip_obs, epsilon=epsilon)
    ret_rms = RunningMeanStd(shape=())
    running_returns = np.zeros((n_envs,), dtype=np.float32)
    obs_dims = [obs_dim] * n_envs

    official_obs = vecnorm.reset()
    custom_obs, _ = _apply_observation_normalization_if_enabled(
        obs_algo,
        data["obs_seq"][0],
        obs_dims=obs_dims,
        obs_slot_dim=obs_dim,
    )
    episode_starts = np.ones((n_envs,), dtype=np.float32)
    dummy_state = th.zeros((1, n_envs, 4), dtype=th.float32)
    lstm_states = RNNStates((dummy_state, dummy_state), (dummy_state, dummy_state))

    vecnorm_diffs: dict[str, float] = {
        "reset_obs_max_abs_diff": _max_abs(official_obs, custom_obs),
        "obs_step_max_abs_diff": 0.0,
        "reward_step_max_abs_diff": 0.0,
        "running_returns_max_abs_diff": 0.0,
    }

    for step in range(n_steps):
        action = data["actions"][step]
        value_t = th.as_tensor(data["values"][step], dtype=th.float32)
        log_prob_t = th.as_tensor(data["log_probs"][step], dtype=th.float32)
        official_buffer.add(
            np.asarray(official_obs, dtype=np.float32),
            action,
            np.zeros((n_envs,), dtype=np.float32),
            episode_starts,
            value_t,
            log_prob_t,
            lstm_states=lstm_states,
        )
        custom_buffer.add(
            np.asarray(custom_obs, dtype=np.float32),
            action,
            np.zeros((n_envs,), dtype=np.float32),
            episode_starts,
            value_t,
            log_prob_t,
            lstm_states=lstm_states,
            action_masks=np.ones((n_envs, act_dim), dtype=np.float32),
            next_states=data["obs_seq"][step + 1],
            next_state_masks=np.ones((n_envs, obs_dim), dtype=np.float32),
            objective_masks=np.ones((n_envs,), dtype=np.float32),
        )

        vecnorm.step_async(action)
        official_next_obs, official_reward, done, _infos = vecnorm.step_wait()
        custom_reward, running_returns = _sb3_vecnormalize_reward_step(
            data["rewards"][step],
            running_returns=running_returns,
            ret_rms=ret_rms,
            dones=done,
            gamma=gamma,
            epsilon=epsilon,
            clip_reward=clip_reward,
        )
        custom_next_obs, _ = _apply_observation_normalization_if_enabled(
            obs_algo,
            data["obs_seq"][step + 1],
            obs_dims=obs_dims,
            obs_slot_dim=obs_dim,
        )

        official_buffer.rewards[step] = np.asarray(official_reward, dtype=np.float32)
        custom_buffer.rewards[step] = np.asarray(custom_reward, dtype=np.float32)

        vecnorm_diffs["obs_step_max_abs_diff"] = max(
            vecnorm_diffs["obs_step_max_abs_diff"],
            _max_abs(official_next_obs, custom_next_obs),
        )
        vecnorm_diffs["reward_step_max_abs_diff"] = max(
            vecnorm_diffs["reward_step_max_abs_diff"],
            _max_abs(official_reward, custom_reward),
        )
        vecnorm_diffs["running_returns_max_abs_diff"] = max(
            vecnorm_diffs["running_returns_max_abs_diff"],
            _max_abs(vecnorm.returns, running_returns),
        )
        official_obs = official_next_obs
        custom_obs = custom_next_obs
        episode_starts = np.asarray(done, dtype=np.float32)

    assert isinstance(vecnorm.obs_rms, RunningMeanStd)
    last_values_t = th.as_tensor(data["last_values"], dtype=th.float32)
    official_buffer.compute_returns_and_advantage(last_values_t, data["dones"][-1])
    custom_buffer.compute_returns_and_advantage(last_values_t, data["dones"][-1])

    vecnorm_diffs.update(
        {
            "obs_rms_mean_max_abs_diff": _max_abs(vecnorm.obs_rms.mean, obs_algo._rwkv_sb3_observation_normalization_mean),
            "obs_rms_var_max_abs_diff": _max_abs(vecnorm.obs_rms.var, obs_algo._rwkv_sb3_observation_normalization_var),
            "obs_rms_count_max_abs_diff": _max_abs(
                np.full((obs_dim,), float(vecnorm.obs_rms.count), dtype=np.float64),
                obs_algo._rwkv_sb3_observation_normalization_count,
            ),
            "ret_rms_mean_abs_diff": abs(_finite_float(vecnorm.ret_rms.mean) - _finite_float(ret_rms.mean)),
            "ret_rms_var_abs_diff": abs(_finite_float(vecnorm.ret_rms.var) - _finite_float(ret_rms.var)),
            "ret_rms_count_abs_diff": abs(float(vecnorm.ret_rms.count) - float(ret_rms.count)),
        }
    )
    rollout_diffs = {
        "buffer_observations_max_abs_diff": _max_abs(official_buffer.observations, custom_buffer.observations),
        "buffer_rewards_max_abs_diff": _max_abs(official_buffer.rewards, custom_buffer.rewards),
        "returns_value_targets_max_abs_diff": _max_abs(official_buffer.returns, custom_buffer.returns),
        "advantages_max_abs_diff": _max_abs(official_buffer.advantages, custom_buffer.advantages),
        "actor_advantages_max_abs_diff": _max_abs(official_buffer.advantages, custom_buffer.actor_advantages),
        "old_values_max_abs_diff": _max_abs(official_buffer.values, custom_buffer.values),
        "old_log_probs_max_abs_diff": _max_abs(official_buffer.log_probs, custom_buffer.log_probs),
    }
    ppo_side_ev = {
        "official_formula_ev": float(
            explained_variance(official_buffer.values.flatten(), official_buffer.returns.flatten())
        ),
        "custom_formula_ev": float(
            explained_variance(custom_buffer.values.flatten(), custom_buffer.returns.flatten())
        ),
        "values_max_abs_diff": _max_abs(official_buffer.values, custom_buffer.values),
        "returns_value_targets_max_abs_diff": _max_abs(official_buffer.returns, custom_buffer.returns),
        "formula_ev_abs_diff": abs(
            float(explained_variance(official_buffer.values.flatten(), official_buffer.returns.flatten()))
            - float(explained_variance(custom_buffer.values.flatten(), custom_buffer.returns.flatten()))
        ),
        "values_source": "PPO rollout buffer old_values; not environment-side reward.",
        "targets_source": "PPO rollout buffer returns from official GAE/value-target compute_returns_and_advantage.",
    }
    environment_side = {
        "action_buffer_max_abs_diff": _max_abs(official_buffer.actions, custom_buffer.actions),
        "obs_to_ppo_buffer_max_abs_diff": rollout_diffs["buffer_observations_max_abs_diff"],
        "reward_to_ppo_buffer_max_abs_diff": rollout_diffs["buffer_rewards_max_abs_diff"],
        "vecnormalize_obs_step_max_abs_diff": vecnorm_diffs["obs_step_max_abs_diff"],
        "vecnormalize_reward_step_max_abs_diff": vecnorm_diffs["reward_step_max_abs_diff"],
        "action_contract": "Environment side consumes actions only; PPO side consumes the resulting normalized obs/reward.",
    }
    return {
        "official_buffer": official_buffer,
        "custom_buffer": custom_buffer,
        "vecnorm": vecnorm_diffs,
        "rollout": rollout_diffs,
        "environment_side_contract": environment_side,
        "ppo_side_ev_contract": ppo_side_ev,
        "config": {
            "n_steps": n_steps,
            "n_envs": n_envs,
            "obs_dim": obs_dim,
            "act_dim": act_dim,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "clip_obs": clip_obs,
            "clip_reward": clip_reward,
            "epsilon": epsilon,
        },
    }


def _make_official_algo(sample: Any, policy: _ParametricPolicy, *, cfg: dict[str, Any], logger: _CaptureLogger):
    algo = SimpleNamespace()
    algo.policy = policy
    algo.rollout_buffer = _OneBatchBuffer(
        sample,
        values_np=sample.old_values.detach().cpu().numpy(),
        returns_np=sample.returns.detach().cpu().numpy(),
        mask_np=sample.mask.detach().cpu().numpy(),
    )
    algo.action_space = spaces.Box(low=-1.0, high=1.0, shape=(int(sample.actions.shape[-1]),), dtype=np.float32)
    algo.n_epochs = int(cfg["n_epochs"])
    algo.batch_size = int(sample.actions.shape[0])
    algo.clip_range = lambda progress: float(cfg["clip_range"])
    algo.clip_range_vf = None
    algo._current_progress_remaining = 1.0
    algo.normalize_advantage = bool(cfg["normalize_advantage"])
    algo.ent_coef = float(cfg["ent_coef"])
    algo.vf_coef = float(cfg["vf_coef"])
    algo.target_kl = cfg["target_kl"]
    algo.verbose = 0
    algo.max_grad_norm = float(cfg["max_grad_norm"])
    algo._n_updates = 0
    algo.logger = logger
    algo._update_learning_rate = lambda optimizer: None
    return algo


def _make_custom_algo(sample: Any, policy: _ParametricPolicy, *, cfg: dict[str, Any], logger: _CaptureLogger):
    algo = SimpleNamespace()
    algo.policy = policy
    algo.rollout_buffer = _OneBatchBuffer(
        sample,
        values_np=sample.old_values.detach().cpu().numpy(),
        returns_np=sample.returns.detach().cpu().numpy(),
        mask_np=sample.objective_masks.detach().cpu().numpy(),
    )
    algo.n_epochs = int(cfg["n_epochs"])
    algo.batch_size = int(sample.actions.shape[0])
    algo.n_envs = 1
    algo.n_steps = int(sample.actions.shape[0])
    algo.clip_range = lambda progress: float(cfg["clip_range"])
    algo.clip_range_vf = None
    algo._current_progress_remaining = 1.0
    algo.normalize_advantage = bool(cfg["normalize_advantage"])
    algo.ent_coef = float(cfg["ent_coef"])
    algo.vf_coef = float(cfg["vf_coef"])
    algo.target_kl = cfg["target_kl"]
    algo.verbose = 0
    algo.max_grad_norm = float(cfg["max_grad_norm"])
    algo._n_updates = 0
    algo.device = th.device("cpu")
    algo.logger = logger
    algo._rwkv_policy_loss_coef = 1.0
    algo._rwkv_aux_q_weight = 0.0
    algo._rwkv_aux_flow_weight = 0.0
    algo._rwkv_reset_env_state_at_sep = False
    algo._update_learning_rate = lambda optimizer: None
    algo._progress_log_every = lambda: 1000000
    algo._progress_log_min_interval_sec = lambda: 1e9
    algo._progress_logging_enabled = lambda: False
    algo._emit_progress_log = lambda msg: None
    algo._append_train_outer_batch_selector_trace_row_flat = lambda **kwargs: None
    algo._resolve_sequence_subbatch_size = lambda rollout_data: int(
        getattr(rollout_data, "n_seq", rollout_data.lstm_states.pi[0].shape[1])
    )
    algo._resolve_train_outer_batch_snapshot_match_mask_flat = lambda **kwargs: None
    algo._write_train_outer_batch_snapshot_flat = lambda **kwargs: None
    algo._finalize_train_outer_batch_selector_trace = lambda: None
    return algo


def _official_sample_from_buffer(buffer: RecurrentRolloutBuffer, *, seed: int):
    np.random.seed(seed)
    return next(buffer.get(buffer.buffer_size * buffer.n_envs))


def _custom_padded_from_official(sample: Any) -> MaskedRecurrentRolloutBufferSamples:
    n = int(sample.observations.shape[0])
    action_dim = int(sample.actions.shape[-1])
    obs_dim = int(sample.observations.shape[-1])
    return MaskedRecurrentRolloutBufferSamples(
        observations=sample.observations,
        actions=sample.actions,
        old_values=sample.old_values,
        old_log_prob=sample.old_log_prob,
        advantages=sample.advantages,
        returns=sample.returns,
        objective_masks=sample.mask,
        rollout_return_means=th.zeros((n,), dtype=th.float32),
        rollout_return_stds=th.ones((n,), dtype=th.float32),
        lstm_states=sample.lstm_states,
        episode_starts=sample.episode_starts,
        mask=sample.mask,
        action_masks=th.ones((n, action_dim), dtype=th.float32),
        next_states=th.zeros((n, obs_dim), dtype=th.float32),
        next_state_masks=th.zeros((n, obs_dim), dtype=th.float32),
        actor_advantages=sample.advantages,
        normalized_q_targets=None,
        value_target_bucket_idx=None,
    )


def _custom_flat_from_official(sample: Any) -> MaskedRecurrentFlatBatchSamples:
    n = int(sample.observations.shape[0])
    action_dim = int(sample.actions.shape[-1])
    obs_dim = int(sample.observations.shape[-1])
    n_seq = int(sample.lstm_states.pi[0].shape[1])
    if n_seq <= 0 or n % n_seq != 0:
        raise ValueError(f"Cannot infer flat sequence layout from official sample shape: n={n}, n_seq={n_seq}")
    max_len = int(n // n_seq)
    return MaskedRecurrentFlatBatchSamples(
        observations=sample.observations,
        actions=sample.actions,
        old_values=sample.old_values,
        old_log_prob=sample.old_log_prob,
        advantages=sample.advantages,
        returns=sample.returns,
        objective_masks=sample.mask,
        rollout_return_means=th.zeros((n,), dtype=th.float32),
        rollout_return_stds=th.ones((n,), dtype=th.float32),
        lstm_states=sample.lstm_states,
        episode_starts=sample.episode_starts,
        action_masks=th.ones((n, action_dim), dtype=th.float32),
        next_states=th.zeros((n, obs_dim), dtype=th.float32),
        next_state_masks=th.zeros((n, obs_dim), dtype=th.float32),
        seq_start_indices=np.arange(0, n, max_len, dtype=np.int64),
        seq_lengths=np.full((n_seq,), max_len, dtype=np.int64),
        actor_advantages=sample.advantages,
        normalized_q_targets=None,
        value_target_bucket_idx=None,
        flat_batch_indices=np.arange(n, dtype=np.int64),
        flat_env_indices=np.zeros((n,), dtype=np.int64),
        flat_step_indices=np.arange(n, dtype=np.int64),
        flat_objective_episode_indices=np.zeros((n,), dtype=np.int64),
        flat_objective_episode_positions=np.arange(n, dtype=np.int64),
        flat_objective_global_positions=np.arange(n, dtype=np.int64),
    )


def _run_train_pair(official_sample: Any, custom_sample: Any, *, normalize_advantage: bool, seed: int) -> dict[str, Any]:
    del seed
    cfg = {
        "n_epochs": 1,
        "clip_range": 0.2,
        "normalize_advantage": bool(normalize_advantage),
        "ent_coef": 0.03,
        "vf_coef": 0.5,
        "target_kl": None,
        "max_grad_norm": 0.5,
    }
    official_logger = _CaptureLogger()
    official_policy = _ParametricPolicy(lr=0.01)
    official_initial = official_policy.param.detach().clone()
    official_algo = _make_official_algo(official_sample, official_policy, cfg=cfg, logger=official_logger)
    official_formula_ev = float(
        explained_variance(official_algo.rollout_buffer.values.flatten(), official_algo.rollout_buffer.returns.flatten())
    )
    RecurrentPPO.train(official_algo)  # type: ignore[arg-type]

    custom_logger = _CaptureLogger()
    custom_policy = _ParametricPolicy(lr=0.01)
    custom_initial = custom_policy.param.detach().clone()
    custom_algo = _make_custom_algo(custom_sample, custom_policy, cfg=cfg, logger=custom_logger)
    custom_formula_ev = float(
        explained_variance(custom_algo.rollout_buffer.values.flatten(), custom_algo.rollout_buffer.returns.flatten())
    )
    MaskedRecurrentPPO.train(custom_algo)  # type: ignore[arg-type]

    keys = [
        "train/policy_gradient_loss",
        "train/value_loss",
        "train/entropy_loss",
        "train/approx_kl",
        "train/clip_fraction",
        "train/loss",
        "train/explained_variance",
    ]
    out: dict[str, Any] = {}
    for key in keys:
        official_value = float(official_logger.rows[key])
        custom_value = float(custom_logger.rows[key])
        label = key.replace("train/", "")
        out[f"{label}_official"] = official_value
        out[f"{label}_custom"] = custom_value
        out[f"{label}_abs_diff"] = abs(official_value - custom_value)
    out["grad_max_abs_diff"] = _tensor_max_abs(official_policy.param.grad, custom_policy.param.grad)
    out["grad_l2_official"] = float(th.linalg.vector_norm(official_policy.param.grad.detach()).cpu())
    out["grad_l2_custom"] = float(th.linalg.vector_norm(custom_policy.param.grad.detach()).cpu())
    out["ppo_side_ev_formula_official"] = official_formula_ev
    out["ppo_side_ev_formula_custom"] = custom_formula_ev
    out["ppo_side_ev_formula_abs_diff"] = abs(official_formula_ev - custom_formula_ev)
    out["official_logger_ev_formula_abs_diff"] = abs(float(official_logger.rows["train/explained_variance"]) - official_formula_ev)
    out["custom_logger_ev_formula_abs_diff"] = abs(float(custom_logger.rows["train/explained_variance"]) - custom_formula_ev)
    out["param_delta_max_abs_diff"] = _tensor_max_abs(
        official_policy.param.detach() - official_initial,
        custom_policy.param.detach() - custom_initial,
    )
    out["final_param_max_abs_diff"] = _tensor_max_abs(official_policy.param.detach(), custom_policy.param.detach())
    return out


def _max_report_diff(report: dict[str, Any]) -> float:
    max_diff = 0.0
    for section in (
        "vecnormalize",
        "rollout_buffer",
        "environment_side_contract",
        "ppo_side_ev_contract",
        "official_update_similarity_contract",
        "train",
    ):
        obj = report.get(section)
        if isinstance(obj, dict):
            iterator = obj.values()
        else:
            iterator = []
        for value in iterator:
            if isinstance(value, dict):
                for key, inner_value in value.items():
                    if key.endswith("_diff") or key.endswith("max_abs_diff") or key.endswith("abs_diff"):
                        max_diff = max(max_diff, float(inner_value))
            elif isinstance(value, (float, int)):
                pass
    for section_name in ("vecnormalize", "rollout_buffer"):
        for key, value in report.get(section_name, {}).items():
            if key.endswith("_diff") or key.endswith("max_abs_diff") or key.endswith("abs_diff"):
                max_diff = max(max_diff, float(value))
    return float(max_diff)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/chen/RLPFN/artifacts/phase2_official_recurrent_ppo_real_rollout_update_parity_0501"),
    )
    parser.add_argument("--seed", type=int, default=20260501)
    parser.add_argument("--tol", type=float, default=3e-6)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    collected = _collect_buffers(args.seed)
    official_sample = _official_sample_from_buffer(collected["official_buffer"], seed=args.seed + 17)
    padded_sample = _custom_padded_from_official(official_sample)
    flat_sample = _custom_flat_from_official(official_sample)
    train = {
        "padded_no_adv_norm": _run_train_pair(
            official_sample,
            padded_sample,
            normalize_advantage=False,
            seed=args.seed + 101,
        ),
        "padded_adv_norm": _run_train_pair(
            official_sample,
            padded_sample,
            normalize_advantage=True,
            seed=args.seed + 102,
        ),
        "flat_no_adv_norm": _run_train_pair(
            official_sample,
            flat_sample,
            normalize_advantage=False,
            seed=args.seed + 103,
        ),
        "flat_adv_norm": _run_train_pair(
            official_sample,
            flat_sample,
            normalize_advantage=True,
            seed=args.seed + 104,
        ),
    }
    official_update_similarity_contract = {
        "metrics": {
            "train/explained_variance": "SB3 RecurrentPPO value-target explained variance.",
            "train/approx_kl": "SB3 RecurrentPPO reverse-KL approximation for policy update size.",
            "train/clip_fraction": "SB3 RecurrentPPO fraction of policy ratios outside the clip range.",
        },
        "note": "SB3/RecurrentPPO does not provide an official Pearson-correlation training diagnostic.",
        "per_train_case_abs_diffs": {
            name: {
                "explained_variance_abs_diff": float(row["explained_variance_abs_diff"]),
                "approx_kl_abs_diff": float(row["approx_kl_abs_diff"]),
                "clip_fraction_abs_diff": float(row["clip_fraction_abs_diff"]),
            }
            for name, row in train.items()
        },
    }
    report = {
        "seed": int(args.seed),
        "tolerance": float(args.tol),
        "config": collected["config"],
        "vecnormalize": collected["vecnorm"],
        "rollout_buffer": collected["rollout"],
        "environment_side_contract": collected["environment_side_contract"],
        "ppo_side_ev_contract": collected["ppo_side_ev_contract"],
        "official_update_similarity_contract": official_update_similarity_contract,
        "train": train,
        "scope": {
            "compared": [
                "official VecNormalize reset/step_wait obs RMS update order on Box observations",
                "official VecNormalize reward returns/ret_rms update order",
                "normalized observations and rewards stored in rollout buffer",
                "GAE advantages and TD(lambda) returns/value targets",
                "PPO policy loss, value MSE target, entropy loss, reverse-KL approximation, clip fraction",
                "official explained_variance on PPO value targets",
                "post-clip parameter gradients and optimizer parameter delta",
                "official recurrent padded sample path and local flat replay path",
            ],
            "held_equal": [
                "objective_mask equals official valid/padding mask",
                "clip_range_vf=None, matching the clean Phase-2 runner and SB3 default",
                "auxiliary Q/flow losses disabled",
                "custom actor/value outputs identical to official by construction",
                "single minibatch, one epoch",
                "all observation dimensions active; variable-dimension pad masking is a bridge contract, not PPO core",
            ],
            "not_covered": [
                "prior/gym environment bridge construction",
                "RWKV backbone numerical equality beyond the separate outer-path contract",
                "value bar-distribution head calibration beyond MSE target contract",
                "multi-update optimizer drift",
                "non-default clip_range_vf because the clean runner pins clip_range_vf=None",
            ],
        },
    }
    max_diff = _max_report_diff(report)
    report["max_abs_diff"] = max_diff
    report["passed"] = bool(math.isfinite(max_diff) and max_diff <= float(args.tol))
    out_path = args.output_dir / "real_rollout_update_parity_report.json"
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
