#!/usr/bin/env python3
"""Parity checks for the PPO core against sb3-contrib RecurrentPPO.

This probe deliberately removes the environment bridge, prior generator, and
RWKV backbone from the question.  It compares:

1. GAE/returns from the official RecurrentRolloutBuffer vs our masked buffer
   when the custom objective mask is identical to the official valid mask.
2. PPO train-step scalar math from official RecurrentPPO.train vs our
   MaskedRecurrentPPO.train on the same synthetic rollout batch.

If this fails, the PPO core is not trustworthy.  If this passes, remaining
risks are outside the official PPO core: data collection, objective masking,
RWKV policy evaluation, normalization plumbing, or environment construction.
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

from sb3_contrib.common.recurrent.buffers import RecurrentRolloutBuffer  # noqa: E402
from sb3_contrib.ppo_recurrent.ppo_recurrent import RecurrentPPO  # noqa: E402

from ticl.sb3_recurrent_ppo import (  # noqa: E402
    MaskedRecurrentFlatBatchSamples,
    MaskedRecurrentPPO,
    MaskedRecurrentRolloutBuffer,
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
        self.rows[key] = value


class _OneBatchBuffer:
    def __init__(self, sample: Any, *, values_np: np.ndarray, returns_np: np.ndarray, mask_np: np.ndarray) -> None:
        self._sample = sample
        self.buffer_size = int(values_np.shape[0])
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


class _OfficialFakePolicy:
    def __init__(self, new_values: th.Tensor, new_log_prob: th.Tensor, entropy: th.Tensor) -> None:
        self.param = th.nn.Parameter(th.zeros(()))
        self.optimizer = th.optim.SGD([self.param], lr=0.0)
        self._new_values = new_values
        self._new_log_prob = new_log_prob
        self._entropy = entropy

    def parameters(self):
        return [self.param]

    def set_training_mode(self, mode: bool) -> None:
        del mode

    def evaluate_actions(self, observations, actions, lstm_states, episode_starts):
        del observations, actions, lstm_states, episode_starts
        zero = self.param * 0.0
        return self._new_values + zero, self._new_log_prob + zero, self._entropy + zero


class _CustomFakePolicy(_OfficialFakePolicy):
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
        del obs, actions, seq_lengths, action_masks, include_normalized_q_logits, flow_matching_xt, flow_matching_t
        zero = self.param * 0.0
        values = self._new_values + zero
        return {
            "value_logits": values,
            "values": values,
            "log_prob": self._new_log_prob + zero,
            "entropy": self._entropy + zero,
        }

    def compute_value_loss_from_logits(self, *, value_logits, targets, objective_mask, value_target_bucket_idx=None):
        del value_target_bucket_idx
        mask = objective_mask.to(device=value_logits.device, dtype=th.bool).reshape(-1)
        return th.mean(((targets.reshape(-1) - value_logits.reshape(-1)) ** 2)[mask])


def _make_official_fake_algo(sample: Any, policy: _OfficialFakePolicy, *, cfg: dict[str, Any], logger: _CaptureLogger):
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


def _make_custom_fake_algo(sample: MaskedRecurrentFlatBatchSamples, policy: _CustomFakePolicy, *, cfg: dict[str, Any], logger: _CaptureLogger):
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
    algo._resolve_sequence_subbatch_size = lambda rollout_data: int(rollout_data.n_seq)
    algo._resolve_train_outer_batch_snapshot_match_mask_flat = lambda **kwargs: None
    algo._write_train_outer_batch_snapshot_flat = lambda **kwargs: None
    algo._finalize_train_outer_batch_selector_trace = lambda: None
    return algo


def _max_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))


def _check_gae(seed: int, *, n_steps: int, n_envs: int, obs_dim: int, act_dim: int, gamma: float, gae_lambda: float) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
    act_space = spaces.Box(low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32)
    hidden_state_shape = (n_steps, 1, n_envs, 4)

    official = RecurrentRolloutBuffer(
        n_steps,
        obs_space,
        act_space,
        hidden_state_shape,
        device="cpu",
        gae_lambda=gae_lambda,
        gamma=gamma,
        n_envs=n_envs,
    )
    custom = MaskedRecurrentRolloutBuffer(
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

    rewards = rng.normal(loc=0.1, scale=1.7, size=(n_steps, n_envs)).astype(np.float32)
    values = rng.normal(loc=-0.2, scale=1.1, size=(n_steps, n_envs)).astype(np.float32)
    episode_starts = np.zeros((n_steps, n_envs), dtype=np.float32)
    episode_starts[0, :] = 1.0
    for env_i in range(n_envs):
        for step in rng.choice(np.arange(1, n_steps), size=max(1, n_steps // 5), replace=False):
            episode_starts[int(step), env_i] = 1.0
    dones = rng.integers(0, 2, size=(n_envs,), dtype=np.int64).astype(bool)
    last_values = th.as_tensor(rng.normal(size=(n_envs,)).astype(np.float32))

    for buf in (official, custom):
        buf.rewards[:] = rewards
        buf.values[:] = values
        buf.episode_starts[:] = episode_starts
    custom.objective_masks[:] = 1.0
    custom._value_target_space = "raw"
    custom._actor_gae_space = "raw"
    custom._actor_baseline_mode = "learned"
    custom._actor_gae_lambda_override = None
    custom._actor_objective_mode = "tokenwise"

    official.compute_returns_and_advantage(last_values, dones)
    custom.compute_returns_and_advantage(last_values, dones)

    return {
        "advantages_max_abs_diff": _max_abs(official.advantages, custom.advantages),
        "actor_advantages_max_abs_diff": _max_abs(official.advantages, custom.actor_advantages),
        "returns_max_abs_diff": _max_abs(official.returns, custom.returns),
    }


def _build_train_samples(seed: int, *, n: int, obs_dim: int, act_dim: int) -> tuple[Any, MaskedRecurrentFlatBatchSamples, dict[str, th.Tensor]]:
    rng = np.random.default_rng(seed)
    obs = th.as_tensor(rng.normal(size=(n, obs_dim)).astype(np.float32))
    actions = th.as_tensor(rng.normal(scale=0.4, size=(n, act_dim)).astype(np.float32))
    old_values = th.as_tensor(rng.normal(size=(n,)).astype(np.float32))
    returns = th.as_tensor(rng.normal(loc=0.2, scale=1.3, size=(n,)).astype(np.float32))
    advantages = th.as_tensor(rng.normal(loc=-0.1, scale=1.4, size=(n,)).astype(np.float32))
    old_log_prob = th.as_tensor(rng.normal(loc=-0.5, scale=0.7, size=(n,)).astype(np.float32))
    new_log_prob = old_log_prob + th.as_tensor(rng.normal(loc=0.02, scale=0.18, size=(n,)).astype(np.float32))
    new_values = old_values + th.as_tensor(rng.normal(loc=0.0, scale=0.35, size=(n,)).astype(np.float32))
    entropy = th.as_tensor(rng.uniform(0.2, 1.5, size=(n,)).astype(np.float32))
    mask_np = np.ones((n,), dtype=np.float32)
    mask_np[rng.choice(np.arange(n), size=max(1, n // 7), replace=False)] = 0.0
    mask = th.as_tensor(mask_np, dtype=th.float32)
    episode_starts = th.zeros((n,), dtype=th.float32)
    episode_starts[0] = 1.0
    action_masks = th.ones((n, act_dim), dtype=th.float32)

    lstm_states = SimpleNamespace(
        pi=(th.zeros((1, 1, 4)), th.zeros((1, 1, 4))),
        vf=(th.zeros((1, 1, 4)), th.zeros((1, 1, 4))),
    )
    official_sample = SimpleNamespace(
        observations=obs,
        actions=actions,
        old_values=old_values,
        old_log_prob=old_log_prob,
        advantages=advantages,
        returns=returns,
        lstm_states=lstm_states,
        episode_starts=episode_starts,
        mask=mask,
    )
    custom_sample = MaskedRecurrentFlatBatchSamples(
        observations=obs,
        actions=actions,
        old_values=old_values,
        old_log_prob=old_log_prob,
        advantages=advantages,
        returns=returns,
        objective_masks=mask,
        rollout_return_means=th.zeros((n,), dtype=th.float32),
        rollout_return_stds=th.ones((n,), dtype=th.float32),
        lstm_states=lstm_states,
        episode_starts=episode_starts,
        action_masks=action_masks,
        next_states=th.zeros((n, obs_dim), dtype=th.float32),
        next_state_masks=th.zeros((n, obs_dim), dtype=th.float32),
        seq_start_indices=np.asarray([0], dtype=np.int64),
        seq_lengths=np.asarray([n], dtype=np.int64),
        actor_advantages=advantages,
        normalized_q_targets=None,
        value_target_bucket_idx=None,
        flat_batch_indices=np.arange(n, dtype=np.int64),
        flat_env_indices=np.zeros((n,), dtype=np.int64),
        flat_step_indices=np.arange(n, dtype=np.int64),
        flat_objective_episode_indices=np.zeros((n,), dtype=np.int64),
        flat_objective_episode_positions=np.arange(n, dtype=np.int64),
        flat_objective_global_positions=np.arange(n, dtype=np.int64),
    )
    outputs = {"new_values": new_values, "new_log_prob": new_log_prob, "entropy": entropy, "mask": mask}
    return official_sample, custom_sample, outputs


def _check_train(seed: int, *, normalize_advantage: bool) -> dict[str, Any]:
    cfg = {
        "n_epochs": 1,
        "clip_range": 0.2,
        "normalize_advantage": normalize_advantage,
        "ent_coef": 0.03,
        "vf_coef": 0.5,
        "target_kl": None,
        "max_grad_norm": 0.5,
    }
    official_sample, custom_sample, outputs = _build_train_samples(seed, n=41, obs_dim=5, act_dim=3)

    official_logger = _CaptureLogger()
    official_policy = _OfficialFakePolicy(outputs["new_values"], outputs["new_log_prob"], outputs["entropy"])
    official_algo = _make_official_fake_algo(official_sample, official_policy, cfg=cfg, logger=official_logger)
    RecurrentPPO.train(official_algo)  # type: ignore[arg-type]

    custom_logger = _CaptureLogger()
    custom_policy = _CustomFakePolicy(outputs["new_values"], outputs["new_log_prob"], outputs["entropy"])
    custom_algo = _make_custom_fake_algo(custom_sample, custom_policy, cfg=cfg, logger=custom_logger)
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
    diffs = {}
    for key in keys:
        official_value = float(official_logger.rows[key])
        custom_value = float(custom_logger.rows[key])
        diffs[f"{key.replace('train/', '')}_official"] = official_value
        diffs[f"{key.replace('train/', '')}_custom"] = custom_value
        diffs[f"{key.replace('train/', '')}_abs_diff"] = abs(official_value - custom_value)
    return diffs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/phase2_official_recurrent_ppo_core_parity_0501"))
    parser.add_argument("--seed", type=int, default=20260501)
    parser.add_argument("--tol", type=float, default=2e-6)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "seed": int(args.seed),
        "tolerance": float(args.tol),
        "gae": _check_gae(args.seed, n_steps=17, n_envs=5, obs_dim=4, act_dim=2, gamma=0.97, gae_lambda=0.83),
        "train_no_adv_norm": _check_train(args.seed + 1, normalize_advantage=False),
        "train_adv_norm": _check_train(args.seed + 2, normalize_advantage=True),
        "scope": {
            "compared": [
                "GAE advantages",
                "TD(lambda) returns",
                "PPO clipped policy loss",
                "value MSE loss without clip_range_vf",
                "entropy loss",
                "reverse-KL approximation",
                "clip fraction",
                "official explained_variance on PPO value targets",
            ],
            "held_equal": [
                "objective mask equals official valid/padding mask",
                "single minibatch, one epoch",
                "clip_range_vf disabled",
                "auxiliary Q/flow losses disabled",
                "policy outputs identical by construction",
            ],
            "not_covered": [
                "RWKV policy forward/evaluate_actions implementation",
                "pack/rollout collection bridge",
                "VecNormalize online plumbing",
                "custom objective masks different from official mask",
                "value distribution/bar head",
                "multi-minibatch optimizer side effects",
            ],
        },
    }
    max_diff = 0.0
    for section in ("gae", "train_no_adv_norm", "train_adv_norm"):
        for key, value in result[section].items():
            if key.endswith("_abs_diff") or key.endswith("max_abs_diff"):
                max_diff = max(max_diff, float(value))
    result["max_abs_diff"] = float(max_diff)
    result["passed"] = bool(math.isfinite(max_diff) and max_diff <= float(args.tol))
    out_path = args.output_dir / "parity_report.json"
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
