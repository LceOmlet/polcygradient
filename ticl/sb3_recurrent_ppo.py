"""
Strict PPO bridge for RWKV + EnvironmentPrior with minimal variable-action support.

Batched-stateful rollout policy evaluation reuses vendored official RWKV
per-sample step operators under torch.vmap, while training-time policy
evaluation uses the official sequence training path.

Custom code here is limited to:
- EnvironmentPrior <-> gymnasium environment adaptation.
- Observation/token plumbing.
- RWKV state/cache adaptation to the recurrent policy API.
- Minimal batched-state wrapper around vendored official step operators.
- Minimal continuous-action masking bridge so PPO can ignore inactive action
  dimensions while still using fixed-width official SB3 Box policies.
- Minimal supervised aux heads on top of the same PPO hidden states; PPO
  clip/GAE/value mathematics remain owned by official SB3.
"""

from __future__ import annotations

import copy
import math
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, NamedTuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def prepare_sb3_runtime() -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


def _ppo_autocast_context(*, autocast_dtype, device):
    if autocast_dtype is None:
        return nullcontext()
    try:
        device_type = torch.device(device).type
    except Exception:
        device_type = "cuda" if "cuda" in str(device) else "cpu"
    return torch.amp.autocast(device_type=device_type, dtype=autocast_dtype)


prepare_sb3_runtime()

try:  # pragma: no cover - import-time compatibility shim
    import cv2  # type: ignore

    if not hasattr(cv2, "ocl"):
        class _OpenCLShim:
            @staticmethod
            def setUseOpenCL(flag):
                del flag
                return None

        cv2.ocl = _OpenCLShim()
except Exception:
    pass

from gymnasium import Env, spaces  # noqa: E402
from sb3_contrib.common.maskable.utils import get_action_masks, is_masking_supported  # noqa: E402
from sb3_contrib.common.recurrent.buffers import RecurrentRolloutBuffer  # noqa: E402
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy  # noqa: E402
from sb3_contrib.common.recurrent.type_aliases import RNNStates  # noqa: E402
from sb3_contrib.ppo_recurrent.ppo_recurrent import RecurrentPPO  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402
from stable_baselines3.common.distributions import DiagGaussianDistribution  # noqa: E402
from stable_baselines3.common.type_aliases import Schedule  # noqa: E402
from stable_baselines3.common.utils import explained_variance, obs_as_tensor  # noqa: E402
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvIndices, VecEnvObs, VecEnvStepReturn  # noqa: E402

from ticl.priors.environment_prior import EnvironmentPrior  # noqa: E402
from ticl.rlpfn_maintained_path import resolve_rlpfn_token_layout  # noqa: E402


STRICT_OFFICIAL_RWKV_PPO_PREFIX = "Strict official RWKV PPO requires"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"{STRICT_OFFICIAL_RWKV_PPO_PREFIX} {message}")


def _resolve_fixed_scalar(spec: Any, *, name: str) -> float:
    if isinstance(spec, bool):
        return float(spec)
    if isinstance(spec, (int, float)):
        return float(spec)
    if isinstance(spec, dict):
        if "value" in spec:
            return float(spec["value"])
        if "min" in spec and "max" in spec and float(spec["min"]) == float(spec["max"]):
            return float(spec["min"])
    raise ValueError(
        f"{STRICT_OFFICIAL_RWKV_PPO_PREFIX} {name} to be fixed, got dynamic spec {spec!r}"
    )


def _resolve_fixed_int(spec: Any, *, name: str) -> int:
    value = _resolve_fixed_scalar(spec, name=name)
    if float(value) != float(int(value)):
        raise ValueError(
            f"{STRICT_OFFICIAL_RWKV_PPO_PREFIX} {name} to be integer-valued, got {value!r}"
        )
    return int(value)


def _clone_tree(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, list):
        return [_clone_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_clone_tree(v) for v in value)
    if isinstance(value, dict):
        return {k: _clone_tree(v) for k, v in value.items()}
    return copy.deepcopy(value)


def _zero_batch_rows(value, row_mask: torch.Tensor):
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.ndim == 0:
            return value
        if int(value.shape[0]) != int(row_mask.shape[0]):
            raise RuntimeError(
                "RWKV PPO rollout state row mask does not match cached state batch dimension."
            )
        if not bool(row_mask.any().item()):
            return value
        out = value.clone()
        mask = row_mask
        while mask.ndim < out.ndim:
            mask = mask.unsqueeze(-1)
        return out.masked_fill(mask, 0)
    if isinstance(value, list):
        return [_zero_batch_rows(v, row_mask) for v in value]
    if isinstance(value, tuple):
        return tuple(_zero_batch_rows(v, row_mask) for v in value)
    if isinstance(value, dict):
        return {k: _zero_batch_rows(v, row_mask) for k, v in value.items()}
    return copy.deepcopy(value)


def _append_text_log_line(path, line: str) -> None:
    if path is None:
        return
    path_str = str(path).strip()
    if path_str == "":
        return
    try:
        with open(path_str, "a", encoding="utf-8") as fh:
            fh.write(str(line))
            fh.write("\n")
    except Exception:
        return


class _NullCallback(BaseCallback):
    def _on_step(self) -> bool:
        return True


class _PlaceholderLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_layers = 1
        self.hidden_size = 1

    def forward(self, *args, **kwargs):
        raise RuntimeError("_PlaceholderLSTM must not be executed.")


class MaskedRecurrentRolloutBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    lstm_states: RNNStates
    episode_starts: torch.Tensor
    mask: torch.Tensor
    action_masks: torch.Tensor
    next_states: torch.Tensor
    next_state_masks: torch.Tensor
    normalized_q_targets: Optional[torch.Tensor] = None


class MaskedRecurrentFlatBatchSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    lstm_states: RNNStates
    episode_starts: torch.Tensor
    action_masks: torch.Tensor
    next_states: torch.Tensor
    next_state_masks: torch.Tensor
    seq_start_indices: np.ndarray
    seq_lengths: np.ndarray
    normalized_q_targets: Optional[torch.Tensor] = None

    @property
    def n_seq(self) -> int:
        return int(len(self.seq_start_indices))

    @property
    def max_seq_len(self) -> int:
        return int(max(self.seq_lengths.tolist(), default=[0]))


def _resolve_padded_sequence_length(flat_steps: int, *, n_seq: int) -> int:
    flat_steps = int(flat_steps)
    n_seq = int(n_seq)
    if n_seq <= 0:
        raise ValueError(f"n_seq must be positive, got {n_seq}")
    if flat_steps % n_seq != 0:
        raise ValueError(
            f"Flat padded batch of {flat_steps} steps must divide evenly by n_seq={n_seq}"
        )
    return int(flat_steps // n_seq)


def _slice_padded_sequence_tensor(
    tensor: torch.Tensor,
    *,
    n_seq: int,
    start_seq: int,
    end_seq: int,
) -> torch.Tensor:
    seq_len = _resolve_padded_sequence_length(int(tensor.shape[0]), n_seq=n_seq)
    sliced = tensor.reshape((int(n_seq), int(seq_len), *tensor.shape[1:]))[int(start_seq) : int(end_seq)]
    return sliced.reshape((-1, *tensor.shape[1:]))


def _slice_masked_rollout_sequence_batch(
    rollout_data: MaskedRecurrentRolloutBufferSamples,
    *,
    start_seq: int,
    end_seq: int,
) -> MaskedRecurrentRolloutBufferSamples:
    n_seq = int(rollout_data.lstm_states.pi[0].shape[1])
    if not (0 <= int(start_seq) < int(end_seq) <= n_seq):
        raise ValueError(
            f"Invalid sequence slice [{int(start_seq)}, {int(end_seq)}) for n_seq={n_seq}"
        )
    pi_states = (
        rollout_data.lstm_states.pi[0][:, int(start_seq) : int(end_seq)].contiguous(),
        rollout_data.lstm_states.pi[1][:, int(start_seq) : int(end_seq)].contiguous(),
    )
    vf_states = (
        rollout_data.lstm_states.vf[0][:, int(start_seq) : int(end_seq)].contiguous(),
        rollout_data.lstm_states.vf[1][:, int(start_seq) : int(end_seq)].contiguous(),
    )
    return MaskedRecurrentRolloutBufferSamples(
        observations=_slice_padded_sequence_tensor(
            rollout_data.observations,
            n_seq=n_seq,
            start_seq=int(start_seq),
            end_seq=int(end_seq),
        ),
        actions=_slice_padded_sequence_tensor(
            rollout_data.actions,
            n_seq=n_seq,
            start_seq=int(start_seq),
            end_seq=int(end_seq),
        ),
        old_values=_slice_padded_sequence_tensor(
            rollout_data.old_values,
            n_seq=n_seq,
            start_seq=int(start_seq),
            end_seq=int(end_seq),
        ),
        old_log_prob=_slice_padded_sequence_tensor(
            rollout_data.old_log_prob,
            n_seq=n_seq,
            start_seq=int(start_seq),
            end_seq=int(end_seq),
        ),
        advantages=_slice_padded_sequence_tensor(
            rollout_data.advantages,
            n_seq=n_seq,
            start_seq=int(start_seq),
            end_seq=int(end_seq),
        ),
        returns=_slice_padded_sequence_tensor(
            rollout_data.returns,
            n_seq=n_seq,
            start_seq=int(start_seq),
            end_seq=int(end_seq),
        ),
        lstm_states=RNNStates(pi_states, vf_states),
        episode_starts=_slice_padded_sequence_tensor(
            rollout_data.episode_starts,
            n_seq=n_seq,
            start_seq=int(start_seq),
            end_seq=int(end_seq),
        ),
        mask=_slice_padded_sequence_tensor(
            rollout_data.mask,
            n_seq=n_seq,
            start_seq=int(start_seq),
            end_seq=int(end_seq),
        ),
        action_masks=_slice_padded_sequence_tensor(
            rollout_data.action_masks,
            n_seq=n_seq,
            start_seq=int(start_seq),
            end_seq=int(end_seq),
        ),
        next_states=_slice_padded_sequence_tensor(
            rollout_data.next_states,
            n_seq=n_seq,
            start_seq=int(start_seq),
            end_seq=int(end_seq),
        ),
        next_state_masks=_slice_padded_sequence_tensor(
            rollout_data.next_state_masks,
            n_seq=n_seq,
            start_seq=int(start_seq),
            end_seq=int(end_seq),
        ),
    )


def _slice_flat_sequence_batch_to_padded(
    flat_batch: MaskedRecurrentFlatBatchSamples,
    *,
    start_seq: int,
    end_seq: int,
):
    n_seq = int(flat_batch.n_seq)
    if not (0 <= int(start_seq) < int(end_seq) <= n_seq):
        raise ValueError(
            f"Invalid sequence slice [{int(start_seq)}, {int(end_seq)}) for n_seq={n_seq}"
        )
    flat_start = int(flat_batch.seq_start_indices[int(start_seq)])
    flat_end = (
        int(flat_batch.seq_start_indices[int(end_seq)])
        if int(end_seq) < n_seq
        else int(flat_batch.observations.shape[0])
    )
    local_seq_starts = (
        np.asarray(flat_batch.seq_start_indices[int(start_seq) : int(end_seq)], dtype=np.int64) - int(flat_start)
    )
    observations_padded, valid = MaskedRecurrentRolloutBuffer._pad_device_tensor(
        flat_batch.observations[flat_start:flat_end],
        seq_start_indices=local_seq_starts,
    )
    actions_padded, _ = MaskedRecurrentRolloutBuffer._pad_device_tensor(
        flat_batch.actions[flat_start:flat_end],
        seq_start_indices=local_seq_starts,
    )
    old_values_padded, _ = MaskedRecurrentRolloutBuffer._pad_device_tensor(
        flat_batch.old_values[flat_start:flat_end],
        seq_start_indices=local_seq_starts,
    )
    old_log_prob_padded, _ = MaskedRecurrentRolloutBuffer._pad_device_tensor(
        flat_batch.old_log_prob[flat_start:flat_end],
        seq_start_indices=local_seq_starts,
    )
    returns_padded, _ = MaskedRecurrentRolloutBuffer._pad_device_tensor(
        flat_batch.returns[flat_start:flat_end],
        seq_start_indices=local_seq_starts,
    )
    episode_starts_padded, _ = MaskedRecurrentRolloutBuffer._pad_device_tensor(
        flat_batch.episode_starts[flat_start:flat_end],
        seq_start_indices=local_seq_starts,
    )
    action_masks_padded, _ = MaskedRecurrentRolloutBuffer._pad_device_tensor(
        flat_batch.action_masks[flat_start:flat_end],
        seq_start_indices=local_seq_starts,
    )
    next_states_padded, _ = MaskedRecurrentRolloutBuffer._pad_device_tensor(
        flat_batch.next_states[flat_start:flat_end],
        seq_start_indices=local_seq_starts,
    )
    next_state_masks_padded, _ = MaskedRecurrentRolloutBuffer._pad_device_tensor(
        flat_batch.next_state_masks[flat_start:flat_end],
        seq_start_indices=local_seq_starts,
    )
    n_sub_seq = int(end_seq - start_seq)
    max_length = int(observations_padded.shape[1])
    padded_batch_size = int(n_sub_seq * max_length)
    pi_states = (
        flat_batch.lstm_states.pi[0][:, int(start_seq) : int(end_seq)].contiguous(),
        flat_batch.lstm_states.pi[1][:, int(start_seq) : int(end_seq)].contiguous(),
    )
    vf_states = (
        flat_batch.lstm_states.vf[0][:, int(start_seq) : int(end_seq)].contiguous(),
        flat_batch.lstm_states.vf[1][:, int(start_seq) : int(end_seq)].contiguous(),
    )
    return (
        MaskedRecurrentRolloutBufferSamples(
            observations=observations_padded.reshape((padded_batch_size, *flat_batch.observations.shape[1:])),
            actions=actions_padded.reshape((padded_batch_size, *flat_batch.actions.shape[1:])),
            old_values=old_values_padded.reshape((padded_batch_size,)),
            old_log_prob=old_log_prob_padded.reshape((padded_batch_size,)),
            advantages=returns_padded.new_zeros((padded_batch_size,)),
            returns=returns_padded.reshape((padded_batch_size,)),
            lstm_states=RNNStates(pi_states, vf_states),
            episode_starts=episode_starts_padded.reshape((padded_batch_size,)),
            mask=valid.reshape((padded_batch_size,)).to(device=returns_padded.device, dtype=returns_padded.dtype),
            action_masks=action_masks_padded.reshape((padded_batch_size, flat_batch.action_masks.shape[-1])),
            next_states=next_states_padded.reshape((padded_batch_size, flat_batch.next_states.shape[-1])),
            next_state_masks=next_state_masks_padded.reshape((padded_batch_size, flat_batch.next_state_masks.shape[-1])),
        ),
        flat_start,
        flat_end,
        local_seq_starts,
    )


class MaskedRecurrentRolloutBuffer(RecurrentRolloutBuffer):
    action_masks: np.ndarray
    next_states: np.ndarray
    next_state_masks: np.ndarray

    def __init__(self, *args, next_state_dim: int, **kwargs):
        self.next_state_dim = int(next_state_dim)
        super().__init__(*args, **kwargs)

    def reset(self):
        enable_q_target_cache = bool(getattr(self, "_enable_q_target_cache", False))
        if not isinstance(self.action_space, spaces.Box):
            raise ValueError("MaskedRecurrentRolloutBuffer only supports Box action spaces.")
        self.mask_dims = int(np.prod(self.action_space.shape, dtype=np.int64))
        self.action_masks = np.ones((self.buffer_size, self.n_envs, self.mask_dims), dtype=np.float32)
        self.next_states = np.zeros((self.buffer_size, self.n_envs, self.next_state_dim), dtype=np.float32)
        self.next_state_masks = np.zeros((self.buffer_size, self.n_envs, self.next_state_dim), dtype=np.float32)
        self._enable_q_target_cache = enable_q_target_cache
        self._global_normalized_q_targets_flat = None
        self._global_q_target_seq_ids_flat = None
        self._global_q_target_seq_lengths = None
        super().reset()

    def clear_device_cache(self) -> None:
        return None

    def _ensure_generator_ready(self) -> None:
        if self.generator_ready:
            return
        for tensor in ["hidden_states_pi", "cell_states_pi", "hidden_states_vf", "cell_states_vf"]:
            self.__dict__[tensor] = self.__dict__[tensor].swapaxes(1, 2)

        for tensor in [
            "observations",
            "actions",
            "values",
            "log_probs",
            "advantages",
            "returns",
            "hidden_states_pi",
            "cell_states_pi",
            "hidden_states_vf",
            "cell_states_vf",
            "episode_starts",
            "action_masks",
            "next_states",
            "next_state_masks",
        ]:
            self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
        self.generator_ready = True

    def _resolve_batch_size(self, batch_size: Optional[int]) -> int:
        if batch_size is None:
            return int(self.buffer_size * self.n_envs)
        return int(batch_size)

    def _build_env_change_flat(self) -> np.ndarray:
        env_change = np.zeros(self.buffer_size * self.n_envs).reshape(self.buffer_size, self.n_envs)
        env_change[0, :] = 1.0
        return self.swap_and_flatten(env_change)

    @staticmethod
    def _normalize_sequence_fragment_like_reinforce(
        values: torch.Tensor,
        *,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        mean = values.mean()
        centered = values - mean
        std = torch.sqrt(centered.square().mean().clamp_min(float(eps)))
        return (values - mean) / std

    def _ensure_global_q_target_cache(self) -> None:
        if (
            self._global_normalized_q_targets_flat is not None
            and self._global_q_target_seq_ids_flat is not None
            and self._global_q_target_seq_lengths is not None
        ):
            return
        self._ensure_generator_ready()
        total_steps = int(self.buffer_size * self.n_envs)
        batch_inds = np.arange(total_steps, dtype=np.int64)
        env_change = self._build_env_change_flat()
        seq_start_indices, _, _ = self._create_sequencers(batch_inds, env_change)
        seq_start_indices = np.asarray(seq_start_indices, dtype=np.int64)
        seq_end_exclusive = np.empty_like(seq_start_indices)
        seq_end_exclusive[:-1] = seq_start_indices[1:]
        seq_end_exclusive[-1] = total_steps
        returns_flat = torch.as_tensor(self.returns, dtype=torch.float32).reshape(-1)
        normalized_q_targets_flat = torch.empty_like(returns_flat)
        seq_ids_flat = np.empty((total_steps,), dtype=np.int64)
        seq_lengths = np.empty((len(seq_start_indices),), dtype=np.int64)
        for seq_id, (seq_start, seq_end) in enumerate(zip(seq_start_indices.tolist(), seq_end_exclusive.tolist())):
            seq_start = int(seq_start)
            seq_end = int(seq_end)
            seq_values = returns_flat[seq_start:seq_end]
            normalized_q_targets_flat[seq_start:seq_end] = self._normalize_sequence_fragment_like_reinforce(seq_values)
            seq_ids_flat[seq_start:seq_end] = int(seq_id)
            seq_lengths[int(seq_id)] = int(seq_end - seq_start)
        self._global_normalized_q_targets_flat = normalized_q_targets_flat.cpu().numpy().astype(np.float32, copy=False)
        self._global_q_target_seq_ids_flat = seq_ids_flat
        self._global_q_target_seq_lengths = seq_lengths

    def _get_cached_flat_normalized_q_targets(
        self,
        *,
        batch_inds: np.ndarray,
        seq_start_indices: np.ndarray,
        seq_lengths: np.ndarray,
        returns_flat: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        self._ensure_global_q_target_cache()
        batch_inds = np.asarray(batch_inds, dtype=np.int64)
        seq_start_indices = np.asarray(seq_start_indices, dtype=np.int64)
        seq_lengths = np.asarray(seq_lengths, dtype=np.int64)
        cached_targets = torch.as_tensor(
            self._global_normalized_q_targets_flat[batch_inds],
            device=returns_flat.device,
            dtype=returns_flat.dtype,
        ).contiguous()
        batch_seq_ids = np.asarray(self._global_q_target_seq_ids_flat[batch_inds], dtype=np.int64)
        global_seq_lengths = np.asarray(self._global_q_target_seq_lengths, dtype=np.int64)
        for local_seq_idx, seq_start in enumerate(seq_start_indices.tolist()):
            seq_start = int(seq_start)
            seq_len = int(seq_lengths[int(local_seq_idx)])
            seq_end = seq_start + seq_len
            local_seq_ids = batch_seq_ids[seq_start:seq_end]
            global_seq_id = int(local_seq_ids[0])
            if np.all(local_seq_ids == global_seq_id) and seq_len == int(global_seq_lengths[global_seq_id]):
                continue
            cached_targets[seq_start:seq_end] = self._normalize_sequence_fragment_like_reinforce(
                returns_flat[seq_start:seq_end],
                eps=eps,
            ).to(dtype=cached_targets.dtype)
        return cached_targets

    def _iter_batch_plan(self, batch_size: Optional[int] = None):
        assert self.full, "Rollout buffer must be full before sampling from it"
        self._ensure_generator_ready()
        batch_size = self._resolve_batch_size(batch_size)

        split_index = np.random.randint(self.buffer_size * self.n_envs)
        indices = np.arange(self.buffer_size * self.n_envs)
        indices = np.concatenate((indices[split_index:], indices[:split_index]))

        env_change = self._build_env_change_flat()

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            batch_inds = indices[start_idx : start_idx + batch_size]
            yield batch_inds, env_change
            start_idx += batch_size

    @staticmethod
    def _pad_device_tensor(flat_tensor: torch.Tensor, *, seq_start_indices: np.ndarray):
        n_seq = int(len(seq_start_indices))
        if n_seq <= 0:
            raise ValueError("PPO padded sequence batch requires at least one sequence.")
        seq_start_indices = np.asarray(seq_start_indices, dtype=np.int64)
        seq_end_exclusive = np.empty_like(seq_start_indices)
        seq_end_exclusive[:-1] = seq_start_indices[1:]
        seq_end_exclusive[-1] = int(flat_tensor.shape[0])
        seq_starts = torch.as_tensor(seq_start_indices, device=flat_tensor.device, dtype=torch.long)
        seq_ends = torch.as_tensor(seq_end_exclusive, device=flat_tensor.device, dtype=torch.long)
        seq_lengths = seq_ends - seq_starts
        max_length = int(seq_lengths.max().detach().cpu())
        pos = torch.arange(max_length, device=flat_tensor.device, dtype=torch.long)
        gather_idx = seq_starts.unsqueeze(1) + pos.unsqueeze(0)
        safe_idx = torch.minimum(gather_idx, (seq_ends - 1).unsqueeze(1))
        padded = flat_tensor.index_select(0, safe_idx.reshape(-1)).reshape(
            n_seq,
            max_length,
            *flat_tensor.shape[1:],
        )
        valid = pos.unsqueeze(0) < seq_lengths.unsqueeze(1)
        if flat_tensor.ndim == 1:
            padded = padded * valid.to(dtype=flat_tensor.dtype)
        else:
            valid_expanded = valid
            while valid_expanded.ndim < padded.ndim:
                valid_expanded = valid_expanded.unsqueeze(-1)
            padded = padded * valid_expanded.to(dtype=padded.dtype)
        return padded, valid

    def add(
        self,
        *args,
        action_masks: Optional[np.ndarray] = None,
        next_states: Optional[np.ndarray] = None,
        next_state_masks: Optional[np.ndarray] = None,
        **kwargs,
    ) -> None:
        if action_masks is not None:
            self.action_masks[self.pos] = np.asarray(action_masks, dtype=np.float32).reshape((self.n_envs, self.mask_dims))
        if next_states is not None:
            self.next_states[self.pos] = np.asarray(next_states, dtype=np.float32).reshape((self.n_envs, self.next_state_dim))
        if next_state_masks is not None:
            self.next_state_masks[self.pos] = np.asarray(next_state_masks, dtype=np.float32).reshape(
                (self.n_envs, self.next_state_dim)
            )
        super().add(*args, **kwargs)

    def get(self, batch_size: Optional[int] = None):
        for batch_inds, env_change in self._iter_batch_plan(batch_size):
            yield self._get_samples(batch_inds, env_change)

    def _get_samples(self, batch_inds: np.ndarray, env_change: np.ndarray, env=None) -> MaskedRecurrentRolloutBufferSamples:
        self.seq_start_indices, self.pad, self.pad_and_flatten = self._create_sequencers(batch_inds, env_change)

        n_seq = len(self.seq_start_indices)
        max_length = self.pad(self.actions[batch_inds]).shape[1]
        padded_batch_size = n_seq * max_length
        lstm_states_pi = (
            self.hidden_states_pi[batch_inds][self.seq_start_indices].swapaxes(0, 1),
            self.cell_states_pi[batch_inds][self.seq_start_indices].swapaxes(0, 1),
        )
        lstm_states_vf = (
            self.hidden_states_vf[batch_inds][self.seq_start_indices].swapaxes(0, 1),
            self.cell_states_vf[batch_inds][self.seq_start_indices].swapaxes(0, 1),
        )
        lstm_states_pi = (self.to_torch(lstm_states_pi[0]).contiguous(), self.to_torch(lstm_states_pi[1]).contiguous())
        lstm_states_vf = (self.to_torch(lstm_states_vf[0]).contiguous(), self.to_torch(lstm_states_vf[1]).contiguous())

        return MaskedRecurrentRolloutBufferSamples(
            observations=self.pad(self.observations[batch_inds]).reshape((padded_batch_size, *self.obs_shape)),
            actions=self.pad(self.actions[batch_inds]).reshape((padded_batch_size, *self.actions.shape[1:])),
            old_values=self.pad_and_flatten(self.values[batch_inds]),
            old_log_prob=self.pad_and_flatten(self.log_probs[batch_inds]),
            advantages=self.pad_and_flatten(self.advantages[batch_inds]),
            returns=self.pad_and_flatten(self.returns[batch_inds]),
            lstm_states=RNNStates(lstm_states_pi, lstm_states_vf),
            episode_starts=self.pad_and_flatten(self.episode_starts[batch_inds]),
            mask=self.pad_and_flatten(np.ones_like(self.returns[batch_inds])),
            action_masks=self.pad(self.action_masks[batch_inds]).reshape((padded_batch_size, self.mask_dims)),
            next_states=self.pad(self.next_states[batch_inds]).reshape((padded_batch_size, self.next_state_dim)),
            next_state_masks=self.pad(self.next_state_masks[batch_inds]).reshape((padded_batch_size, self.next_state_dim)),
        )

    def get_gpu(self, batch_size: Optional[int] = None):
        for batch_inds, env_change in self._iter_batch_plan(batch_size):
            yield self._get_samples_gpu(batch_inds, env_change)

    def get_gpu_flat(self, batch_size: Optional[int] = None):
        for batch_inds, env_change in self._iter_batch_plan(batch_size):
            yield self._get_flat_batch_gpu(batch_inds, env_change)

    def _get_samples_gpu(
        self,
        batch_inds: np.ndarray,
        env_change: np.ndarray,
    ) -> MaskedRecurrentRolloutBufferSamples:
        self.seq_start_indices, _, _ = self._create_sequencers(batch_inds, env_change)
        n_seq = int(len(self.seq_start_indices))
        observations_flat = self.to_torch(self.observations[batch_inds]).contiguous()
        actions_flat = self.to_torch(self.actions[batch_inds]).contiguous()
        values_flat = self.to_torch(self.values[batch_inds]).reshape(-1).contiguous()
        log_probs_flat = self.to_torch(self.log_probs[batch_inds]).reshape(-1).contiguous()
        advantages_flat = self.to_torch(self.advantages[batch_inds]).reshape(-1).contiguous()
        returns_flat = self.to_torch(self.returns[batch_inds]).reshape(-1).contiguous()
        episode_starts_flat = self.to_torch(self.episode_starts[batch_inds]).reshape(-1).contiguous()
        action_masks_flat = self.to_torch(self.action_masks[batch_inds]).contiguous()
        next_states_flat = self.to_torch(self.next_states[batch_inds]).contiguous()
        next_state_masks_flat = self.to_torch(self.next_state_masks[batch_inds]).contiguous()
        hidden_states_pi_flat = self.to_torch(self.hidden_states_pi[batch_inds]).contiguous()
        cell_states_pi_flat = self.to_torch(self.cell_states_pi[batch_inds]).contiguous()
        hidden_states_vf_flat = self.to_torch(self.hidden_states_vf[batch_inds]).contiguous()
        cell_states_vf_flat = self.to_torch(self.cell_states_vf[batch_inds]).contiguous()
        seq_start_indices = np.asarray(self.seq_start_indices, dtype=np.int64)
        seq_end_exclusive = np.empty_like(seq_start_indices)
        seq_end_exclusive[:-1] = seq_start_indices[1:]
        seq_end_exclusive[-1] = int(len(batch_inds))
        seq_lengths = (seq_end_exclusive - seq_start_indices).astype(np.int64, copy=False)
        normalized_q_targets_flat = None
        if bool(getattr(self, "_enable_q_target_cache", False)):
            normalized_q_targets_flat = self._get_cached_flat_normalized_q_targets(
                batch_inds=batch_inds,
                seq_start_indices=seq_start_indices,
                seq_lengths=seq_lengths,
                returns_flat=returns_flat,
            )

        observations_padded, valid = self._pad_device_tensor(
            observations_flat,
            seq_start_indices=self.seq_start_indices,
        )
        actions_padded, _ = self._pad_device_tensor(
            actions_flat,
            seq_start_indices=self.seq_start_indices,
        )
        values_padded, _ = self._pad_device_tensor(
            values_flat,
            seq_start_indices=self.seq_start_indices,
        )
        log_probs_padded, _ = self._pad_device_tensor(
            log_probs_flat,
            seq_start_indices=self.seq_start_indices,
        )
        advantages_padded, _ = self._pad_device_tensor(
            advantages_flat,
            seq_start_indices=self.seq_start_indices,
        )
        returns_padded, _ = self._pad_device_tensor(
            returns_flat,
            seq_start_indices=self.seq_start_indices,
        )
        episode_starts_padded, _ = self._pad_device_tensor(
            episode_starts_flat,
            seq_start_indices=self.seq_start_indices,
        )
        action_masks_padded, _ = self._pad_device_tensor(
            action_masks_flat,
            seq_start_indices=self.seq_start_indices,
        )
        next_states_padded, _ = self._pad_device_tensor(
            next_states_flat,
            seq_start_indices=self.seq_start_indices,
        )
        next_state_masks_padded, _ = self._pad_device_tensor(
            next_state_masks_flat,
            seq_start_indices=self.seq_start_indices,
        )
        normalized_q_targets_padded = None
        if normalized_q_targets_flat is not None:
            normalized_q_targets_padded, _ = self._pad_device_tensor(
                normalized_q_targets_flat,
                seq_start_indices=self.seq_start_indices,
            )

        max_length = int(observations_padded.shape[1])
        padded_batch_size = int(n_seq * max_length)
        seq_start_indices_t = torch.as_tensor(self.seq_start_indices, device=self.device, dtype=torch.long)
        lstm_states_pi = (
            hidden_states_pi_flat.index_select(0, seq_start_indices_t).swapaxes(0, 1).contiguous(),
            cell_states_pi_flat.index_select(0, seq_start_indices_t).swapaxes(0, 1).contiguous(),
        )
        lstm_states_vf = (
            hidden_states_vf_flat.index_select(0, seq_start_indices_t).swapaxes(0, 1).contiguous(),
            cell_states_vf_flat.index_select(0, seq_start_indices_t).swapaxes(0, 1).contiguous(),
        )

        return MaskedRecurrentRolloutBufferSamples(
            observations=observations_padded.reshape((padded_batch_size, *self.obs_shape)),
            actions=actions_padded.reshape((padded_batch_size, *self.actions.shape[1:])),
            old_values=values_padded.reshape((padded_batch_size,)),
            old_log_prob=log_probs_padded.reshape((padded_batch_size,)),
            advantages=advantages_padded.reshape((padded_batch_size,)),
            returns=returns_padded.reshape((padded_batch_size,)),
            lstm_states=RNNStates(lstm_states_pi, lstm_states_vf),
            episode_starts=episode_starts_padded.reshape((padded_batch_size,)),
            mask=valid.reshape((padded_batch_size,)).to(device=self.device, dtype=returns_flat.dtype),
            action_masks=action_masks_padded.reshape((padded_batch_size, self.mask_dims)),
            next_states=next_states_padded.reshape((padded_batch_size, self.next_state_dim)),
            next_state_masks=next_state_masks_padded.reshape((padded_batch_size, self.next_state_dim)),
            normalized_q_targets=(
                None
                if normalized_q_targets_padded is None
                else normalized_q_targets_padded.reshape((padded_batch_size,))
            ),
        )

    def _get_flat_batch_gpu(
        self,
        batch_inds: np.ndarray,
        env_change: np.ndarray,
    ) -> MaskedRecurrentFlatBatchSamples:
        self.seq_start_indices, _, _ = self._create_sequencers(batch_inds, env_change)
        seq_start_indices = np.asarray(self.seq_start_indices, dtype=np.int64)
        seq_end_exclusive = np.empty_like(seq_start_indices)
        seq_end_exclusive[:-1] = seq_start_indices[1:]
        seq_end_exclusive[-1] = int(len(batch_inds))
        seq_lengths = (seq_end_exclusive - seq_start_indices).astype(np.int64, copy=False)

        observations_flat = self.to_torch(self.observations[batch_inds]).contiguous()
        actions_flat = self.to_torch(self.actions[batch_inds]).contiguous()
        values_flat = self.to_torch(self.values[batch_inds]).reshape(-1).contiguous()
        log_probs_flat = self.to_torch(self.log_probs[batch_inds]).reshape(-1).contiguous()
        advantages_flat = self.to_torch(self.advantages[batch_inds]).reshape(-1).contiguous()
        returns_flat = self.to_torch(self.returns[batch_inds]).reshape(-1).contiguous()
        episode_starts_flat = self.to_torch(self.episode_starts[batch_inds]).reshape(-1).contiguous()
        action_masks_flat = self.to_torch(self.action_masks[batch_inds]).contiguous()
        next_states_flat = self.to_torch(self.next_states[batch_inds]).contiguous()
        next_state_masks_flat = self.to_torch(self.next_state_masks[batch_inds]).contiguous()
        n_seq = int(len(seq_start_indices))
        dummy_lstm_state = torch.zeros((1, n_seq, 1), device=self.device, dtype=torch.float32)
        normalized_q_targets_flat = None
        if bool(getattr(self, "_enable_q_target_cache", False)):
            normalized_q_targets_flat = self._get_cached_flat_normalized_q_targets(
                batch_inds=batch_inds,
                seq_start_indices=seq_start_indices,
                seq_lengths=seq_lengths,
                returns_flat=returns_flat,
            )

        return MaskedRecurrentFlatBatchSamples(
            observations=observations_flat,
            actions=actions_flat,
            old_values=values_flat,
            old_log_prob=log_probs_flat,
            advantages=advantages_flat,
            returns=returns_flat,
            lstm_states=RNNStates(
                (dummy_lstm_state, dummy_lstm_state),
                (dummy_lstm_state, dummy_lstm_state),
            ),
            episode_starts=episode_starts_flat,
            action_masks=action_masks_flat,
            next_states=next_states_flat,
            next_state_masks=next_state_masks_flat,
            seq_start_indices=seq_start_indices,
            seq_lengths=seq_lengths,
            normalized_q_targets=normalized_q_targets_flat,
        )

    def _create_sequencers(self, batch_inds: np.ndarray, env_change: np.ndarray):
        from sb3_contrib.common.recurrent.buffers import create_sequencers

        return create_sequencers(self.episode_starts[batch_inds], env_change[batch_inds], self.device)


class MaskedRecurrentPPO(RecurrentPPO):
    rollout_buffer: MaskedRecurrentRolloutBuffer

    def _setup_model(self) -> None:
        super()._setup_model()
        lstm = self.policy.lstm_actor
        hidden_state_buffer_shape = (self.n_steps, lstm.num_layers, self.n_envs, lstm.hidden_size)
        next_state_dim = int(getattr(self.env, "next_state_target_dim", 1))
        self.rollout_buffer = MaskedRecurrentRolloutBuffer(
            self.n_steps,
            self.observation_space,
            self.action_space,
            hidden_state_buffer_shape,
            self.device,
            next_state_dim=next_state_dim,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )

    def _progress_logging_enabled(self) -> bool:
        explicit = getattr(self, "_rwkv_progress_log_enabled", None)
        if explicit is not None:
            return bool(explicit)
        return bool(self.verbose >= 1)

    def _progress_log_every(self) -> int:
        value = getattr(self, "_rwkv_progress_log_every", 1)
        try:
            return max(1, int(value))
        except Exception:
            return 1

    def _progress_log_min_interval_sec(self) -> float:
        value = getattr(self, "_rwkv_progress_log_min_interval_sec", 10.0)
        try:
            return max(0.0, float(value))
        except Exception:
            return 10.0

    def _progress_log_file(self):
        return getattr(self, "_rwkv_progress_log_file", None)

    def _emit_progress_log(self, message: str) -> None:
        if not self._progress_logging_enabled():
            return
        stamped = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(stamped)
        _append_text_log_line(self._progress_log_file(), stamped)

    @staticmethod
    def _normalize_masked_returns_like_reinforce(
        returns_flat: torch.Tensor,
        valid_mask_flat: torch.Tensor,
        *,
        n_seq: int,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        if returns_flat.ndim != 1 or valid_mask_flat.ndim != 1:
            raise ValueError("returns and valid masks must be flattened 1D tensors.")
        n_seq = int(max(1, n_seq))
        total = int(returns_flat.numel())
        if total % n_seq != 0:
            raise ValueError(f"Flattened PPO rollout size {total} must divide by n_seq={n_seq}.")
        seq_len = total // n_seq
        returns_seq = returns_flat.reshape((n_seq, seq_len)).swapaxes(0, 1)
        valid_seq = valid_mask_flat.reshape((n_seq, seq_len)).swapaxes(0, 1).to(
            device=returns_flat.device,
            dtype=returns_flat.dtype,
        )
        counts = valid_seq.sum(dim=0, keepdim=True).clamp_min(1.0)
        mean = (returns_seq * valid_seq).sum(dim=0, keepdim=True) / counts
        centered = (returns_seq - mean) * valid_seq
        std = torch.sqrt((centered.square().sum(dim=0, keepdim=True) / counts).clamp_min(float(eps)))
        normalized = ((returns_seq - mean) / std) * valid_seq
        return normalized.swapaxes(0, 1).reshape(-1)

    @staticmethod
    def _normalize_flat_sequence_returns_like_reinforce(
        returns_flat: torch.Tensor,
        *,
        seq_start_indices: np.ndarray,
        seq_lengths: np.ndarray,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        if returns_flat.ndim != 1:
            raise ValueError("returns_flat must be a flattened 1D tensor.")
        seq_start_indices = np.asarray(seq_start_indices, dtype=np.int64)
        seq_lengths = np.asarray(seq_lengths, dtype=np.int64)
        n_seq = int(len(seq_start_indices))
        if n_seq <= 0:
            raise ValueError("Flat PPO sequence normalization requires at least one sequence.")
        if int(seq_lengths.shape[0]) != n_seq:
            raise ValueError("seq_lengths must align with seq_start_indices.")
        seq_end_exclusive = np.empty_like(seq_start_indices)
        seq_end_exclusive[:-1] = seq_start_indices[1:]
        seq_end_exclusive[-1] = int(returns_flat.shape[0])
        expected_lengths = seq_end_exclusive - seq_start_indices
        if not np.array_equal(expected_lengths, seq_lengths):
            raise ValueError("seq_lengths must exactly match the flat PPO sequence boundaries.")

        seq_starts = torch.as_tensor(seq_start_indices, device=returns_flat.device, dtype=torch.long)
        seq_lens = torch.as_tensor(seq_lengths, device=returns_flat.device, dtype=torch.long)
        seq_ends = seq_starts + seq_lens
        max_length = int(seq_lens.max().detach().cpu())
        pos = torch.arange(max_length, device=returns_flat.device, dtype=torch.long)
        gather_idx = seq_starts.unsqueeze(1) + pos.unsqueeze(0)
        safe_idx = torch.minimum(gather_idx, (seq_ends - 1).unsqueeze(1))
        valid = pos.unsqueeze(0) < seq_lens.unsqueeze(1)
        padded = returns_flat.index_select(0, safe_idx.reshape(-1)).reshape(n_seq, max_length)
        valid_f = valid.to(device=returns_flat.device, dtype=returns_flat.dtype)
        counts = valid_f.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean = (padded * valid_f).sum(dim=1, keepdim=True) / counts
        centered = (padded - mean) * valid_f
        std = torch.sqrt((centered.square().sum(dim=1, keepdim=True) / counts).clamp_min(float(eps)))
        normalized_padded = ((padded - mean) / std) * valid_f
        normalized_flat = torch.zeros_like(returns_flat)
        normalized_flat.index_copy_(0, gather_idx[valid], normalized_padded[valid])
        return normalized_flat

    @staticmethod
    def _pad_flat_sequence_values(
        flat_tensor: torch.Tensor,
        *,
        seq_start_indices: np.ndarray,
    ) -> torch.Tensor:
        padded, _ = MaskedRecurrentRolloutBuffer._pad_device_tensor(
            flat_tensor,
            seq_start_indices=seq_start_indices,
        )
        return padded.reshape((-1, *flat_tensor.shape[1:]))

    def _compute_aux_losses(
        self,
        rollout_data: MaskedRecurrentRolloutBufferSamples,
        *,
        eval_outputs: dict[str, torch.Tensor],
        n_seq: Optional[int] = None,
        normalized_q_targets: Optional[torch.Tensor] = None,
        q_target_denom: Optional[torch.Tensor] = None,
        flow_matching_xt: Optional[torch.Tensor] = None,
        flow_matching_t: Optional[torch.Tensor] = None,
        flow_matching_dx: Optional[torch.Tensor] = None,
        flow_denom: Optional[torch.Tensor] = None,
    ):
        env_prior = getattr(self, "_rwkv_env_prior", None)
        if env_prior is None:
            return None
        q_weight = float(getattr(self, "_rwkv_aux_q_weight", 0.0))
        flow_weight = float(getattr(self, "_rwkv_aux_flow_weight", 0.0))
        if q_weight <= 0.0 and flow_weight <= 0.0:
            return None

        policy = self.policy
        model_ref = getattr(policy, "rlpfn_model", None)
        if model_ref is None:
            raise RuntimeError("PPO auxiliary losses require OfficialRWKVRecurrentPPOPolicy.rlpfn_model.")

        hidden = eval_outputs["hidden"]
        masked_actions = eval_outputs["actions"]
        valid_mask = rollout_data.mask > 1e-8
        if n_seq is None:
            n_seq = int(rollout_data.lstm_states.pi[0].shape[1])
        aux_loss = torch.zeros((), device=hidden.device, dtype=hidden.dtype)
        aux_stats = {}
        aux_outputs = None

        if q_weight > 0.0 or flow_weight > 0.0:
            flow_xt = flow_matching_xt
            flow_t = flow_matching_t
            flow_dx = flow_matching_dx
            if flow_weight > 0.0 and (flow_xt is None or flow_t is None or flow_dx is None):
                flow_xt, flow_t, flow_dx = env_prior._sample_condot_flow_matching_path(
                    rollout_data.next_states.detach()
                )
            aux_outputs = policy.decode_aux_from_hidden(
                hidden,
                actions=masked_actions,
                flow_matching_xt=flow_xt,
                flow_matching_t=flow_t,
                n_seq=int(n_seq),
            )

        if q_weight > 0.0:
            if not hasattr(model_ref, "get_normalized_q_value_bardist"):
                raise RuntimeError("PPO normalized Q aux requires model.get_normalized_q_value_bardist().")
            normalized_q_logits = aux_outputs.get("normalized_q_logits", None)
            if normalized_q_logits is None:
                raise RuntimeError("PPO normalized Q aux expected normalized_q_logits from model.")
            q_targets = normalized_q_targets
            if q_targets is None:
                q_targets = self._normalize_masked_returns_like_reinforce(
                    rollout_data.returns.detach(),
                    valid_mask.detach(),
                    n_seq=n_seq,
                    eps=1e-6,
                )
            bardist = model_ref.get_normalized_q_value_bardist()
            q_loss_all = bardist(
                normalized_q_logits.to(dtype=torch.float32),
                q_targets.to(device=normalized_q_logits.device, dtype=torch.float32),
            )
            if q_target_denom is None:
                q_target_denom = valid_mask.sum().clamp_min(1).to(
                    device=normalized_q_logits.device,
                    dtype=torch.float32,
                )
            valid_mask_f = valid_mask.to(device=q_loss_all.device, dtype=q_loss_all.dtype)
            q_loss_num = (q_loss_all * valid_mask_f).sum()
            q_loss = q_loss_num / q_target_denom.to(device=q_loss_num.device, dtype=q_loss_num.dtype)
            aux_loss = aux_loss + (float(q_weight) * q_loss.to(dtype=aux_loss.dtype))
            aux_stats["normalized_q_value_loss"] = q_loss.detach()

        if flow_weight > 0.0:
            if not hasattr(model_ref, "has_next_state_flow_head") or not bool(model_ref.has_next_state_flow_head()):
                raise RuntimeError("PPO state-flow aux requires model next_state_flow_head.")
            next_state_flow = aux_outputs.get("next_state_flow", None)
            if next_state_flow is None:
                raise RuntimeError("PPO state-flow aux expected next_state_flow from model.")
            flow_mask = rollout_data.next_state_masks.to(
                device=next_state_flow.device,
                dtype=next_state_flow.dtype,
            ) * valid_mask.to(device=next_state_flow.device, dtype=next_state_flow.dtype).unsqueeze(-1)
            if flow_denom is None:
                flow_denom = flow_mask.sum().clamp_min(1.0)
            flow_loss_num = (((next_state_flow - flow_dx.to(dtype=next_state_flow.dtype)) ** 2) * flow_mask).sum()
            flow_loss = flow_loss_num / flow_denom.to(device=flow_loss_num.device, dtype=flow_loss_num.dtype)
            aux_loss = aux_loss + (float(flow_weight) * flow_loss.to(dtype=aux_loss.dtype))
            aux_stats["next_state_flow_matching_loss"] = flow_loss.detach()

        return aux_loss, aux_stats

    def _compute_aux_losses_flat(
        self,
        *,
        hidden: torch.Tensor,
        actions: torch.Tensor,
        seq_lengths: np.ndarray,
        normalized_q_logits: Optional[torch.Tensor] = None,
        next_state_flow: Optional[torch.Tensor] = None,
        normalized_q_targets: Optional[torch.Tensor] = None,
        q_target_denom: Optional[torch.Tensor] = None,
        flow_matching_xt: Optional[torch.Tensor] = None,
        flow_matching_t: Optional[torch.Tensor] = None,
        flow_matching_dx: Optional[torch.Tensor] = None,
        next_states: Optional[torch.Tensor] = None,
        next_state_masks: Optional[torch.Tensor] = None,
        flow_denom: Optional[torch.Tensor] = None,
    ):
        env_prior = getattr(self, "_rwkv_env_prior", None)
        if env_prior is None:
            return None
        q_weight = float(getattr(self, "_rwkv_aux_q_weight", 0.0))
        flow_weight = float(getattr(self, "_rwkv_aux_flow_weight", 0.0))
        if q_weight <= 0.0 and flow_weight <= 0.0:
            return None

        policy = self.policy
        model_ref = getattr(policy, "rlpfn_model", None)
        if model_ref is None:
            raise RuntimeError("PPO auxiliary losses require OfficialRWKVRecurrentPPOPolicy.rlpfn_model.")

        aux_loss = torch.zeros((), device=hidden.device, dtype=hidden.dtype)
        aux_stats = {}
        aux_outputs = None

        need_q_decode = bool(q_weight > 0.0 and normalized_q_logits is None)
        need_flow_decode = bool(flow_weight > 0.0 and next_state_flow is None)
        if need_q_decode or need_flow_decode:
            flow_xt = flow_matching_xt
            flow_t = flow_matching_t
            flow_dx = flow_matching_dx
            if flow_weight > 0.0 and (flow_xt is None or flow_t is None or flow_dx is None):
                if next_states is None:
                    raise RuntimeError("Flat PPO flow aux requires next_states when flow targets are not provided.")
                flow_xt, flow_t, flow_dx = env_prior._sample_condot_flow_matching_path(
                    next_states.detach()
                )
            aux_outputs = policy.decode_aux_from_hidden(
                hidden,
                actions=actions,
                flow_matching_xt=flow_xt,
                flow_matching_t=flow_t,
                seq_lengths=seq_lengths,
            )

        if q_weight > 0.0:
            if not hasattr(model_ref, "get_normalized_q_value_bardist"):
                raise RuntimeError("PPO normalized Q aux requires model.get_normalized_q_value_bardist().")
            if normalized_q_logits is None and aux_outputs is not None:
                normalized_q_logits = aux_outputs.get("normalized_q_logits", None)
            if normalized_q_logits is None:
                raise RuntimeError("PPO normalized Q aux expected normalized_q_logits from model.")
            q_targets = normalized_q_targets
            if q_targets is None:
                raise RuntimeError("Flat PPO normalized Q aux requires precomputed normalized_q_targets.")
            bardist = model_ref.get_normalized_q_value_bardist()
            q_loss_all = bardist(
                normalized_q_logits.to(dtype=torch.float32),
                q_targets.to(device=normalized_q_logits.device, dtype=torch.float32),
            )
            if q_target_denom is None:
                q_target_denom = q_loss_all.new_tensor(float(max(1, int(q_loss_all.numel()))))
            q_loss_num = q_loss_all.sum()
            q_loss = q_loss_num / q_target_denom.to(device=q_loss_num.device, dtype=q_loss_num.dtype)
            aux_loss = aux_loss + (float(q_weight) * q_loss.to(dtype=aux_loss.dtype))
            aux_stats["normalized_q_value_loss"] = q_loss.detach()

        if flow_weight > 0.0:
            if not hasattr(model_ref, "has_next_state_flow_head") or not bool(model_ref.has_next_state_flow_head()):
                raise RuntimeError("PPO state-flow aux requires model next_state_flow_head.")
            if next_state_flow is None and aux_outputs is not None:
                next_state_flow = aux_outputs.get("next_state_flow", None)
            if next_state_flow is None:
                raise RuntimeError("PPO state-flow aux expected next_state_flow from model.")
            if next_state_masks is None or flow_matching_dx is None:
                raise RuntimeError("Flat PPO flow aux requires next_state_masks and flow_matching_dx.")
            flow_mask = next_state_masks.to(
                device=next_state_flow.device,
                dtype=next_state_flow.dtype,
            )
            if flow_denom is None:
                flow_denom = flow_mask.sum().clamp_min(1.0)
            flow_loss_num = (((next_state_flow - flow_matching_dx.to(dtype=next_state_flow.dtype)) ** 2) * flow_mask).sum()
            flow_loss = flow_loss_num / flow_denom.to(device=flow_loss_num.device, dtype=flow_loss_num.dtype)
            aux_loss = aux_loss + (float(flow_weight) * flow_loss.to(dtype=aux_loss.dtype))
            aux_stats["next_state_flow_matching_loss"] = flow_loss.detach()

        return aux_loss, aux_stats

    def _resolve_sequence_subbatch_size(
        self,
        rollout_data: MaskedRecurrentRolloutBufferSamples | MaskedRecurrentFlatBatchSamples,
    ) -> int:
        if isinstance(rollout_data, MaskedRecurrentFlatBatchSamples):
            n_seq = int(rollout_data.n_seq)
            seq_len = int(rollout_data.max_seq_len)
        else:
            n_seq = int(rollout_data.lstm_states.pi[0].shape[1])
            padded_batch = int(rollout_data.observations.shape[0])
            seq_len = _resolve_padded_sequence_length(padded_batch, n_seq=n_seq)
        return int(
            _resolve_official_ppo_sequence_batch_capacity(
                model=getattr(self.policy, "rlpfn_model", None),
                seq_len=seq_len,
                total_batch=n_seq,
            )
        )

    def collect_rollouts(
        self,
        env,
        callback: BaseCallback,
        rollout_buffer,
        n_rollout_steps: int,
    ) -> bool:
        assert isinstance(rollout_buffer, MaskedRecurrentRolloutBuffer), "RolloutBuffer doesn't support masked recurrent PPO"
        assert self._last_obs is not None, "No previous observation was provided"
        rollout_wall_t0 = time.perf_counter()
        if self._progress_logging_enabled():
            self._emit_progress_log(
                "[ppo-rollout-start] "
                f"steps={int(n_rollout_steps)} envs={int(env.num_envs)} "
                f"total_env_steps={int(n_rollout_steps) * int(env.num_envs)}"
            )
        self.policy.set_training_mode(False)
        if not isinstance(env, EnvironmentPriorPPOBatchVecEnv):
            raise ValueError("Strict RWKV PPO rollout requires EnvironmentPriorPPOBatchVecEnv.")
        if not is_masking_supported(env):
            raise ValueError("Environment does not expose action_masks() for variable-action PPO.")

        rollout_buffer.reset()
        callback.on_rollout_start()
        env_prior = self._rwkv_env_prior
        env_prior.clear_rollout_artifacts()
        policy_step_fn = self.policy.make_vectorized_rollout_step_fn()
        single_eval_pos = int(env_prior._sample_single_eval_pos(int(n_rollout_steps), None))
        h_list = env_prior._sample_batch_hypers(int(env.num_envs))
        episode_returns = np.zeros((env.num_envs,), dtype=np.float32)
        episode_lengths = np.zeros((env.num_envs,), dtype=np.int32)
        dummy_lstm_states = self.policy._dummy_states(env.num_envs)
        dones_np = None
        rollout_step_count = 0
        action_masks_np = None
        next_state_masks_np = None
        rollout_continue = True

        def _ppo_step_sink(step_payload):
            nonlocal dones_np
            nonlocal rollout_step_count
            nonlocal action_masks_np
            nonlocal next_state_masks_np
            nonlocal rollout_continue

            obs_np = step_payload["obs"].detach().cpu().numpy()
            actions_np = step_payload["action"].detach().cpu().numpy()
            rewards_np = step_payload["reward"].detach().cpu().numpy()
            starts_np = step_payload["episode_starts"].detach().cpu().numpy().astype(np.float32, copy=False)
            values_t = step_payload["values"]
            log_probs_t = step_payload["log_probs"]
            if action_masks_np is None:
                action_masks_np = step_payload["action_mask"].detach().cpu().numpy().astype(np.float32, copy=False)
            if next_state_masks_np is None:
                next_state_masks_np = (
                    step_payload["next_state_mask"].detach().cpu().numpy().astype(np.float32, copy=False)
                )
            next_states_np = step_payload["next_state"].detach().cpu().numpy()
            next_state_target_dim = int(rollout_buffer.next_state_dim)
            if int(next_states_np.shape[-1]) != next_state_target_dim:
                padded_next_states_np = np.zeros((env.num_envs, next_state_target_dim), dtype=np.float32)
                state_copy = int(min(int(next_states_np.shape[-1]), next_state_target_dim))
                if state_copy > 0:
                    padded_next_states_np[:, :state_copy] = next_states_np[:, :state_copy]
                next_states_np = padded_next_states_np
            if int(next_state_masks_np.shape[-1]) != next_state_target_dim:
                padded_next_state_masks_np = np.zeros((env.num_envs, next_state_target_dim), dtype=np.float32)
                state_copy = int(min(int(next_state_masks_np.shape[-1]), next_state_target_dim))
                if state_copy > 0:
                    padded_next_state_masks_np[:, :state_copy] = next_state_masks_np[:, :state_copy]
                next_state_masks_np = padded_next_state_masks_np
            dones_np = step_payload["dones"].detach().cpu().numpy().astype(bool, copy=False)

            episode_returns[:] = episode_returns + rewards_np.astype(np.float32, copy=False)
            episode_lengths[:] = episode_lengths + 1
            infos_t = [{} for _ in range(env.num_envs)]
            done_indices = np.nonzero(dones_np)[0].tolist()
            for env_idx in done_indices:
                infos_t[int(env_idx)]["episode"] = {
                    "r": float(episode_returns[int(env_idx)]),
                    "l": int(episode_lengths[int(env_idx)]),
                }
                episode_returns[int(env_idx)] = 0.0
                episode_lengths[int(env_idx)] = 0

            self.num_timesteps += env.num_envs
            callback.update_locals(locals())
            if not callback.on_step():
                rollout_continue = False
                return False
            self._update_info_buffer(infos_t, dones_np)

            rollout_buffer.add(
                obs_np,
                actions_np,
                rewards_np,
                starts_np,
                values_t,
                log_probs_t,
                lstm_states=dummy_lstm_states,
                action_masks=action_masks_np,
                next_states=next_states_np,
                next_state_masks=next_state_masks_np,
            )
            rollout_step_count += 1
            return True

        with torch.no_grad():
            env_prior._rollout_family_group_vectorized_with_policy(
                h_list,
                policy_step_fn,
                n_rollout_steps,
                self.policy.num_features,
                single_eval_pos,
                self.device,
                collect_x=False,
                collect_runtime_info=False,
                store_rewards=False,
                policy_objective_kind="reinforce",
                _policy_collect_log_probs=True,
                _policy_detach_action_in_env=False,
                _policy_force_no_grad=True,
                _policy_collect_ppo_trace=True,
                _policy_ppo_step_sink=_ppo_step_sink,
            )

        ppo_trace = getattr(env_prior, "last_rollout_ppo_trace", None)
        if not isinstance(ppo_trace, dict):
            raise RuntimeError("PPO rollout expected final rollout trace, but EnvironmentPrior did not provide it.")
        final_obs = ppo_trace.get("final_obs", None)
        final_terminal = ppo_trace.get("final_terminal", None)
        if not all(torch.is_tensor(x) for x in (final_obs, final_terminal)):
            raise RuntimeError("PPO rollout final trace omitted required tensors.")
        if not rollout_continue:
            return False
        if rollout_step_count != int(n_rollout_steps):
            raise RuntimeError(
                f"PPO rollout step sink emitted {rollout_step_count} steps, expected {int(n_rollout_steps)}."
            )
        if dones_np is None:
            raise RuntimeError("PPO rollout produced zero steps.")
        final_obs = final_obs.detach()
        final_terminal = final_terminal.detach().to(dtype=torch.bool)

        with torch.no_grad():
            final_value_steps = self.policy.evaluate_rollout_values(
                final_obs.to(device=self.device).unsqueeze(0),
                final_terminal.to(device=self.device, dtype=torch.float32).unsqueeze(0),
            )
        last_values = final_value_steps.reshape(-1)
        self._last_obs = final_obs.detach().cpu().numpy().astype(np.float32, copy=False)
        self._last_episode_starts = final_terminal.detach().cpu().numpy().astype(bool, copy=False)
        self._last_lstm_states = dummy_lstm_states

        rollout_buffer.compute_returns_and_advantage(
            last_values=last_values,
            dones=self._last_episode_starts,
        )
        callback.on_rollout_end()
        rollout_wall_s = float(time.perf_counter() - rollout_wall_t0)
        setattr(self, "_rwkv_last_rollout_wall_time_sec", rollout_wall_s)
        if self._progress_logging_enabled():
            fps = float(int(rollout_step_count) * int(env.num_envs)) / max(1e-9, rollout_wall_s)
            self._emit_progress_log(
                "[ppo-rollout-end] "
                f"steps={int(rollout_step_count)}/{int(n_rollout_steps)} "
                f"envs={int(env.num_envs)} elapsed_s={rollout_wall_s:.1f} fps={fps:.1f}"
            )
        return True

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)
        grad_scaler = getattr(self, "_rwkv_grad_scaler", None)
        autocast_dtype = getattr(self, "_rwkv_autocast_dtype", None)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        q_aux_losses = []
        flow_aux_losses = []
        continue_training = True
        train_wall_t0 = time.perf_counter()
        total_transitions = int(self.rollout_buffer.buffer_size * self.rollout_buffer.n_envs)
        total_outer_batches = int(max(1, math.ceil(float(total_transitions) / float(max(1, int(self.batch_size))))))
        progress_every = self._progress_log_every()
        progress_min_interval_sec = self._progress_log_min_interval_sec()
        last_progress_t = train_wall_t0 - progress_min_interval_sec
        outer_batches_total_processed = 0
        subbatches_total_processed = 0
        if self._progress_logging_enabled():
            self._emit_progress_log(
                "[ppo-train-start] "
                f"epochs={int(self.n_epochs)} outer_batches_per_epoch~={int(total_outer_batches)} "
                f"batch_size={int(self.batch_size)} n_envs={int(self.n_envs)} n_steps={int(self.n_steps)}"
            )

        buffer_getter_flat = getattr(self.rollout_buffer, "get_gpu_flat", None)
        buffer_getter_padded = getattr(self.rollout_buffer, "get_gpu", None)
        if not callable(buffer_getter_padded):
            buffer_getter_padded = self.rollout_buffer.get

        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            outer_batch_idx = 0
            if callable(buffer_getter_flat):
                rollout_iterator = buffer_getter_flat(self.batch_size)
            else:
                rollout_iterator = buffer_getter_padded(self.batch_size)
            for rollout_data in rollout_iterator:
                outer_batch_idx += 1
                outer_batches_total_processed += 1
                if isinstance(rollout_data, MaskedRecurrentFlatBatchSamples):
                    valid_total = rollout_data.returns.new_tensor(float(max(1, int(rollout_data.returns.numel()))))
                    n_seq_total = int(rollout_data.n_seq)
                    seq_subbatch_size = self._resolve_sequence_subbatch_size(rollout_data)
                    subbatch_count = int(max(1, math.ceil(float(n_seq_total) / float(max(1, int(seq_subbatch_size))))))
                    if self._progress_logging_enabled() and (
                        outer_batch_idx == 1
                        or outer_batch_idx == total_outer_batches
                        or ((outer_batch_idx - 1) % progress_every) == 0
                    ):
                        self._emit_progress_log(
                            "[ppo-train-batch] "
                            f"epoch={int(epoch + 1)}/{int(self.n_epochs)} "
                            f"outer_batch={int(outer_batch_idx)}/{int(total_outer_batches)} "
                            f"seq={int(n_seq_total)} subbatches={int(subbatch_count)} "
                            f"elapsed_s={float(time.perf_counter() - train_wall_t0):.1f}"
                        )

                    advantages = rollout_data.advantages
                    if self.normalize_advantage:
                        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                    self.policy.optimizer.zero_grad()
                    policy_num_total = rollout_data.returns.new_zeros(())
                    value_num_total = rollout_data.returns.new_zeros(())
                    entropy_num_total = rollout_data.returns.new_zeros(())
                    clip_num_total = rollout_data.returns.new_zeros(())
                    approx_kl_num_total = rollout_data.returns.new_zeros(())
                    loss_total = rollout_data.returns.new_zeros(())
                    q_aux_loss_total = rollout_data.returns.new_zeros(())
                    flow_aux_loss_total = rollout_data.returns.new_zeros(())
                    q_targets_full = None
                    flow_xt_full = flow_t_full = flow_dx_full = None
                    flow_denom_total = None
                    if float(getattr(self, "_rwkv_aux_q_weight", 0.0)) > 0.0:
                        q_targets_full = rollout_data.normalized_q_targets
                        if q_targets_full is None:
                            q_targets_full = self._normalize_flat_sequence_returns_like_reinforce(
                                rollout_data.returns.detach(),
                                seq_start_indices=rollout_data.seq_start_indices,
                                seq_lengths=rollout_data.seq_lengths,
                                eps=1e-6,
                            )
                    if float(getattr(self, "_rwkv_aux_flow_weight", 0.0)) > 0.0:
                        env_prior = getattr(self, "_rwkv_env_prior", None)
                        if env_prior is None:
                            raise RuntimeError("PPO flow aux requires bound EnvironmentPrior.")
                        flow_xt_full, flow_t_full, flow_dx_full = env_prior._sample_condot_flow_matching_path(
                            rollout_data.next_states.detach()
                        )
                        flow_denom_total = rollout_data.next_state_masks.sum().clamp_min(1.0)

                    subbatch_idx = 0
                    for start_seq in range(0, n_seq_total, seq_subbatch_size):
                        subbatch_idx += 1
                        subbatches_total_processed += 1
                        end_seq = min(n_seq_total, start_seq + seq_subbatch_size)
                        flat_start = int(rollout_data.seq_start_indices[int(start_seq)])
                        flat_end = (
                            int(rollout_data.seq_start_indices[int(end_seq)])
                            if int(end_seq) < n_seq_total
                            else int(rollout_data.observations.shape[0])
                        )
                        sub_seq_lengths = np.asarray(
                            rollout_data.seq_lengths[int(start_seq) : int(end_seq)],
                            dtype=np.int64,
                        )
                        sub_advantages = advantages[flat_start:flat_end]
                        sub_actions = rollout_data.actions[flat_start:flat_end]
                        sub_old_log_prob = rollout_data.old_log_prob[flat_start:flat_end]
                        sub_old_values = rollout_data.old_values[flat_start:flat_end]
                        sub_returns = rollout_data.returns[flat_start:flat_end]
                        sub_action_masks = rollout_data.action_masks[flat_start:flat_end]
                        sub_q_targets = None
                        sub_flow_xt = sub_flow_t = sub_flow_dx = None
                        if q_targets_full is not None:
                            sub_q_targets = q_targets_full[flat_start:flat_end]
                        if flow_xt_full is not None:
                            sub_flow_xt = flow_xt_full[flat_start:flat_end]
                            sub_flow_t = flow_t_full[flat_start:flat_end]
                            sub_flow_dx = flow_dx_full[flat_start:flat_end]
                        with _ppo_autocast_context(autocast_dtype=autocast_dtype, device=self.device):
                            eval_outputs = self.policy.evaluate_actions_with_hidden_flat(
                                rollout_data.observations[flat_start:flat_end],
                                sub_actions,
                                seq_lengths=sub_seq_lengths,
                                action_masks=sub_action_masks,
                                include_normalized_q_logits=(sub_q_targets is not None),
                                flow_matching_xt=sub_flow_xt,
                                flow_matching_t=sub_flow_t,
                            )
                            values = eval_outputs["values"].flatten()
                            log_prob = eval_outputs["log_prob"]
                            entropy = eval_outputs["entropy"]

                            ratio = torch.exp(log_prob - sub_old_log_prob)
                            policy_loss_1 = sub_advantages * ratio
                            policy_loss_2 = sub_advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                            clipped_objective = torch.min(policy_loss_1, policy_loss_2)
                            policy_num = clipped_objective.sum()
                            policy_loss = -(policy_num / valid_total.to(device=policy_num.device, dtype=policy_num.dtype))

                            if self.clip_range_vf is None:
                                values_pred = values
                            else:
                                values_pred = sub_old_values + torch.clamp(
                                    values - sub_old_values,
                                    -clip_range_vf,
                                    clip_range_vf,
                                )
                            value_errors = (sub_returns - values_pred) ** 2
                            value_num = value_errors.sum()
                            value_loss = value_num / valid_total.to(device=value_num.device, dtype=value_num.dtype)

                            if entropy is None:
                                entropy_terms = -log_prob
                            else:
                                entropy_terms = entropy
                            entropy_num = entropy_terms.sum()
                            entropy_loss = -(entropy_num / valid_total.to(device=entropy_num.device, dtype=entropy_num.dtype))

                            main_loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                            subbatch_loss = main_loss

                            if q_targets_full is not None or flow_xt_full is not None:
                                aux_loss_info = self._compute_aux_losses_flat(
                                    hidden=eval_outputs["hidden"],
                                    actions=eval_outputs["actions"],
                                    seq_lengths=sub_seq_lengths,
                                    normalized_q_logits=eval_outputs.get("normalized_q_logits", None),
                                    next_state_flow=eval_outputs.get("next_state_flow", None),
                                    normalized_q_targets=sub_q_targets,
                                    q_target_denom=valid_total,
                                    flow_matching_xt=sub_flow_xt,
                                    flow_matching_t=sub_flow_t,
                                    flow_matching_dx=sub_flow_dx,
                                    next_states=rollout_data.next_states[flat_start:flat_end],
                                    next_state_masks=rollout_data.next_state_masks[flat_start:flat_end],
                                    flow_denom=flow_denom_total,
                                )
                                if aux_loss_info is not None:
                                    aux_loss, aux_stats = aux_loss_info
                                    subbatch_loss = subbatch_loss + aux_loss
                            else:
                                aux_loss_info = None

                        policy_num_total = policy_num_total + policy_num.detach()
                        value_num_total = value_num_total + value_num.detach()
                        entropy_num_total = entropy_num_total + entropy_num.detach()
                        if aux_loss_info is not None:
                            _, aux_stats = aux_loss_info
                            if "normalized_q_value_loss" in aux_stats:
                                q_aux_loss_total = q_aux_loss_total + aux_stats["normalized_q_value_loss"].to(
                                    device=q_aux_loss_total.device,
                                    dtype=q_aux_loss_total.dtype,
                                )
                            if "next_state_flow_matching_loss" in aux_stats:
                                flow_aux_loss_total = flow_aux_loss_total + aux_stats["next_state_flow_matching_loss"].to(
                                    device=flow_aux_loss_total.device,
                                    dtype=flow_aux_loss_total.dtype,
                                )

                        if grad_scaler is None:
                            subbatch_loss.backward()
                        else:
                            grad_scaler.scale(subbatch_loss).backward()
                        loss_total = loss_total + subbatch_loss.detach()
                        now = time.perf_counter()
                        if self._progress_logging_enabled() and (
                            subbatch_idx == 1
                            or subbatch_idx == subbatch_count
                            or (now - last_progress_t) >= progress_min_interval_sec
                        ):
                            self._emit_progress_log(
                                "[ppo-train-progress] "
                                f"epoch={int(epoch + 1)}/{int(self.n_epochs)} "
                                f"outer_batch={int(outer_batch_idx)}/{int(total_outer_batches)} "
                                f"subbatch={int(subbatch_idx)}/{int(subbatch_count)} "
                                f"seq_done={int(end_seq)}/{int(n_seq_total)} "
                                f"elapsed_s={float(now - train_wall_t0):.1f}"
                            )
                            last_progress_t = now

                        with torch.no_grad():
                            log_ratio = log_prob - sub_old_log_prob
                            approx_kl_terms = (torch.exp(log_ratio) - 1) - log_ratio
                            approx_kl_num = approx_kl_terms.sum()
                            approx_kl_num_total = approx_kl_num_total + approx_kl_num.detach()
                            clip_num = (torch.abs(ratio - 1) > clip_range).to(dtype=ratio.dtype).sum()
                            clip_num_total = clip_num_total + clip_num.detach()
                else:
                    mask = rollout_data.mask > 1e-8
                    valid_total = mask.sum().clamp_min(1).to(device=rollout_data.returns.device, dtype=rollout_data.returns.dtype)
                    n_seq_total = int(rollout_data.lstm_states.pi[0].shape[1])
                    seq_subbatch_size = self._resolve_sequence_subbatch_size(rollout_data)
                    subbatch_count = int(max(1, math.ceil(float(n_seq_total) / float(max(1, int(seq_subbatch_size))))))
                    if self._progress_logging_enabled() and (
                        outer_batch_idx == 1
                        or outer_batch_idx == total_outer_batches
                        or ((outer_batch_idx - 1) % progress_every) == 0
                    ):
                        self._emit_progress_log(
                            "[ppo-train-batch] "
                            f"epoch={int(epoch + 1)}/{int(self.n_epochs)} "
                            f"outer_batch={int(outer_batch_idx)}/{int(total_outer_batches)} "
                            f"seq={int(n_seq_total)} subbatches={int(subbatch_count)} "
                            f"elapsed_s={float(time.perf_counter() - train_wall_t0):.1f}"
                        )

                    advantages = rollout_data.advantages
                    if self.normalize_advantage:
                        advantages = (advantages - advantages[mask].mean()) / (advantages[mask].std() + 1e-8)
                    self.policy.optimizer.zero_grad()
                    policy_num_total = rollout_data.returns.new_zeros(())
                    value_num_total = rollout_data.returns.new_zeros(())
                    entropy_num_total = rollout_data.returns.new_zeros(())
                    clip_num_total = rollout_data.returns.new_zeros(())
                    approx_kl_num_total = rollout_data.returns.new_zeros(())
                    loss_total = rollout_data.returns.new_zeros(())
                    q_aux_loss_total = rollout_data.returns.new_zeros(())
                    flow_aux_loss_total = rollout_data.returns.new_zeros(())
                    q_targets_full = None
                    flow_xt_full = flow_t_full = flow_dx_full = None
                    flow_denom_total = None
                    if float(getattr(self, "_rwkv_aux_q_weight", 0.0)) > 0.0:
                        q_targets_full = rollout_data.normalized_q_targets
                        if q_targets_full is None:
                            q_targets_full = self._normalize_masked_returns_like_reinforce(
                                rollout_data.returns.detach(),
                                mask.detach(),
                                n_seq=n_seq_total,
                                eps=1e-6,
                            )
                    if float(getattr(self, "_rwkv_aux_flow_weight", 0.0)) > 0.0:
                        env_prior = getattr(self, "_rwkv_env_prior", None)
                        if env_prior is None:
                            raise RuntimeError("PPO flow aux requires bound EnvironmentPrior.")
                        flow_xt_full, flow_t_full, flow_dx_full = env_prior._sample_condot_flow_matching_path(
                            rollout_data.next_states.detach()
                        )
                        flow_mask_full = rollout_data.next_state_masks.to(
                            device=rollout_data.next_states.device,
                            dtype=rollout_data.next_states.dtype,
                        ) * mask.to(device=rollout_data.next_states.device, dtype=rollout_data.next_states.dtype).unsqueeze(-1)
                        flow_denom_total = flow_mask_full.sum().clamp_min(1.0)

                    subbatch_idx = 0
                    for start_seq in range(0, n_seq_total, seq_subbatch_size):
                        subbatch_idx += 1
                        subbatches_total_processed += 1
                        end_seq = min(n_seq_total, start_seq + seq_subbatch_size)
                        sub_rollout_data = _slice_masked_rollout_sequence_batch(
                            rollout_data,
                            start_seq=start_seq,
                            end_seq=end_seq,
                        )
                        sub_mask = sub_rollout_data.mask > 1e-8
                        sub_mask_f = sub_mask.to(device=sub_rollout_data.returns.device, dtype=sub_rollout_data.returns.dtype)
                        sub_advantages = _slice_padded_sequence_tensor(
                            advantages,
                            n_seq=n_seq_total,
                            start_seq=start_seq,
                            end_seq=end_seq,
                        )
                        sub_q_targets = None if q_targets_full is None else _slice_padded_sequence_tensor(
                            q_targets_full,
                            n_seq=n_seq_total,
                            start_seq=start_seq,
                            end_seq=end_seq,
                        )
                        sub_flow_xt = None if flow_xt_full is None else _slice_padded_sequence_tensor(
                            flow_xt_full,
                            n_seq=n_seq_total,
                            start_seq=start_seq,
                            end_seq=end_seq,
                        )
                        sub_flow_t = None if flow_t_full is None else _slice_padded_sequence_tensor(
                            flow_t_full,
                            n_seq=n_seq_total,
                            start_seq=start_seq,
                            end_seq=end_seq,
                        )
                        sub_flow_dx = None if flow_dx_full is None else _slice_padded_sequence_tensor(
                            flow_dx_full,
                            n_seq=n_seq_total,
                            start_seq=start_seq,
                            end_seq=end_seq,
                        )
                        with _ppo_autocast_context(autocast_dtype=autocast_dtype, device=self.device):
                            eval_outputs = self.policy.evaluate_actions_with_hidden(
                                sub_rollout_data.observations,
                                sub_rollout_data.actions,
                                sub_rollout_data.lstm_states,
                                sub_rollout_data.episode_starts,
                                action_masks=sub_rollout_data.action_masks,
                            )
                            values = eval_outputs["values"].flatten()
                            log_prob = eval_outputs["log_prob"]
                            entropy = eval_outputs["entropy"]

                            ratio = torch.exp(log_prob - sub_rollout_data.old_log_prob)
                            policy_loss_1 = sub_advantages * ratio
                            policy_loss_2 = sub_advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                            clipped_objective = torch.min(policy_loss_1, policy_loss_2)
                            policy_num = (clipped_objective * sub_mask_f.to(device=clipped_objective.device, dtype=clipped_objective.dtype)).sum()
                            policy_loss = -(policy_num / valid_total.to(device=policy_num.device, dtype=policy_num.dtype))

                            if self.clip_range_vf is None:
                                values_pred = values
                            else:
                                values_pred = sub_rollout_data.old_values + torch.clamp(
                                    values - sub_rollout_data.old_values,
                                    -clip_range_vf,
                                    clip_range_vf,
                                )
                            value_errors = (sub_rollout_data.returns - values_pred) ** 2
                            value_num = (value_errors * sub_mask_f.to(device=value_errors.device, dtype=value_errors.dtype)).sum()
                            value_loss = value_num / valid_total.to(device=value_num.device, dtype=value_num.dtype)

                            if entropy is None:
                                entropy_terms = -log_prob
                            else:
                                entropy_terms = entropy
                            entropy_num = (
                                entropy_terms * sub_mask_f.to(device=entropy_terms.device, dtype=entropy_terms.dtype)
                            ).sum()
                            entropy_loss = -(entropy_num / valid_total.to(device=entropy_num.device, dtype=entropy_num.dtype))

                            main_loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                            subbatch_loss = main_loss

                            if q_targets_full is not None or flow_xt_full is not None:
                                aux_loss_info = self._compute_aux_losses(
                                    sub_rollout_data,
                                    eval_outputs=eval_outputs,
                                    n_seq=int(end_seq - start_seq),
                                    normalized_q_targets=sub_q_targets,
                                    q_target_denom=valid_total,
                                    flow_matching_xt=sub_flow_xt,
                                    flow_matching_t=sub_flow_t,
                                    flow_matching_dx=sub_flow_dx,
                                    flow_denom=flow_denom_total,
                                )
                                if aux_loss_info is not None:
                                    aux_loss, aux_stats = aux_loss_info
                                    subbatch_loss = subbatch_loss + aux_loss
                            else:
                                aux_loss_info = None

                        policy_num_total = policy_num_total + policy_num.detach()
                        value_num_total = value_num_total + value_num.detach()
                        entropy_num_total = entropy_num_total + entropy_num.detach()
                        if aux_loss_info is not None:
                            _, aux_stats = aux_loss_info
                            if "normalized_q_value_loss" in aux_stats:
                                q_aux_loss_total = q_aux_loss_total + aux_stats["normalized_q_value_loss"].to(
                                    device=q_aux_loss_total.device,
                                    dtype=q_aux_loss_total.dtype,
                                )
                            if "next_state_flow_matching_loss" in aux_stats:
                                flow_aux_loss_total = flow_aux_loss_total + aux_stats["next_state_flow_matching_loss"].to(
                                    device=flow_aux_loss_total.device,
                                    dtype=flow_aux_loss_total.dtype,
                                )

                        if grad_scaler is None:
                            subbatch_loss.backward()
                        else:
                            grad_scaler.scale(subbatch_loss).backward()
                        loss_total = loss_total + subbatch_loss.detach()
                        now = time.perf_counter()
                        if self._progress_logging_enabled() and (
                            subbatch_idx == 1
                            or subbatch_idx == subbatch_count
                            or (now - last_progress_t) >= progress_min_interval_sec
                        ):
                            self._emit_progress_log(
                                "[ppo-train-progress] "
                                f"epoch={int(epoch + 1)}/{int(self.n_epochs)} "
                                f"outer_batch={int(outer_batch_idx)}/{int(total_outer_batches)} "
                                f"subbatch={int(subbatch_idx)}/{int(subbatch_count)} "
                                f"seq_done={int(end_seq)}/{int(n_seq_total)} "
                                f"elapsed_s={float(now - train_wall_t0):.1f}"
                            )
                            last_progress_t = now

                        with torch.no_grad():
                            log_ratio = log_prob - sub_rollout_data.old_log_prob
                            approx_kl_terms = (torch.exp(log_ratio) - 1) - log_ratio
                            approx_kl_num = (
                                approx_kl_terms * sub_mask_f.to(device=approx_kl_terms.device, dtype=approx_kl_terms.dtype)
                            ).sum()
                            approx_kl_num_total = approx_kl_num_total + approx_kl_num.detach()
                            clip_num = (
                                (torch.abs(ratio - 1) > clip_range).to(dtype=ratio.dtype)
                                * sub_mask_f.to(device=ratio.device, dtype=ratio.dtype)
                            ).sum()
                            clip_num_total = clip_num_total + clip_num.detach()

                policy_loss_scalar = -(policy_num_total / valid_total.to(device=policy_num_total.device, dtype=policy_num_total.dtype))
                value_loss_scalar = value_num_total / valid_total.to(device=value_num_total.device, dtype=value_num_total.dtype)
                entropy_loss_scalar = -(entropy_num_total / valid_total.to(device=entropy_num_total.device, dtype=entropy_num_total.dtype))
                approx_kl_div = (
                    approx_kl_num_total / valid_total.to(device=approx_kl_num_total.device, dtype=approx_kl_num_total.dtype)
                ).detach()
                clip_fraction = (
                    clip_num_total / valid_total.to(device=clip_num_total.device, dtype=clip_num_total.dtype)
                ).detach()
                pg_losses.append(policy_loss_scalar.detach())
                value_losses.append(value_loss_scalar.detach())
                entropy_losses.append(entropy_loss_scalar.detach())
                clip_fractions.append(clip_fraction)
                approx_kl_divs.append(approx_kl_div)
                if float(getattr(self, "_rwkv_aux_q_weight", 0.0)) > 0.0:
                    q_aux_losses.append(q_aux_loss_total.detach())
                if float(getattr(self, "_rwkv_aux_flow_weight", 0.0)) > 0.0:
                    flow_aux_losses.append(flow_aux_loss_total.detach())

                if self.target_kl is not None and float(approx_kl_div.cpu()) > 1.5 * self.target_kl:
                    self.policy.optimizer.zero_grad(set_to_none=True)
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {float(approx_kl_div.cpu()):.2f}")
                    break

                if grad_scaler is not None:
                    grad_scaler.unscale_(self.policy.optimizer)
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                if grad_scaler is None:
                    self.policy.optimizer.step()
                else:
                    grad_scaler.step(self.policy.optimizer)
                    grad_scaler.update()
                loss = loss_total
                if self._progress_logging_enabled() and (
                    outer_batch_idx == total_outer_batches
                    or (outer_batch_idx % progress_every) == 0
                ):
                    self._emit_progress_log(
                        "[ppo-train-batch-done] "
                        f"epoch={int(epoch + 1)}/{int(self.n_epochs)} "
                        f"outer_batch={int(outer_batch_idx)}/{int(total_outer_batches)} "
                        f"elapsed_s={float(time.perf_counter() - train_wall_t0):.1f}"
                    )

            if not continue_training:
                break

        self._n_updates += self.n_epochs
        update_wall_time_sec = float(time.perf_counter() - train_wall_t0)
        setattr(self, "_rwkv_last_update_wall_time_sec", update_wall_time_sec)
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        def _mean_tensor_scalar(values):
            if not values:
                return float("nan")
            return float(torch.stack(values).mean().detach().cpu())

        self.logger.record("train/entropy_loss", _mean_tensor_scalar(entropy_losses))
        self.logger.record("train/policy_gradient_loss", _mean_tensor_scalar(pg_losses))
        self.logger.record("train/value_loss", _mean_tensor_scalar(value_losses))
        self.logger.record("train/approx_kl", _mean_tensor_scalar(approx_kl_divs))
        self.logger.record("train/clip_fraction", _mean_tensor_scalar(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/update_wall_time_sec", update_wall_time_sec)
        self.logger.record("train/outer_batches", int(outer_batches_total_processed))
        self.logger.record("train/subbatches", int(subbatches_total_processed))
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", torch.exp(self.policy.log_std).mean().item())
        if q_aux_losses:
            self.logger.record("train/normalized_q_value_loss", _mean_tensor_scalar(q_aux_losses))
        if flow_aux_losses:
            self.logger.record("train/next_state_flow_matching_loss", _mean_tensor_scalar(flow_aux_losses))
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
        if self._progress_logging_enabled():
            self._emit_progress_log(
                "[ppo-train-end] "
                f"epochs={int(self.n_epochs)} outer_batches={int(outer_batches_total_processed)} "
                f"subbatches={int(subbatches_total_processed)} elapsed_s={update_wall_time_sec:.1f}"
            )
        if hasattr(self.rollout_buffer, "clear_device_cache"):
            self.rollout_buffer.clear_device_cache()


def _target_model(model):
    if hasattr(model, "module"):
        raise ValueError(
            f"{STRICT_OFFICIAL_RWKV_PPO_PREFIX} single-process non-DDP model ownership in PPO mode."
        )
    return model


def _validate_official_rwkv_ppo_boundary(
    *,
    model,
    env_prior: EnvironmentPrior,
    device,
    num_features: int,
):
    target_model = _target_model(model)
    env_cfg = env_prior.config
    layout = resolve_rlpfn_token_layout(env_cfg, num_features=num_features)

    _require("cuda" in str(device), "CUDA device for official RWKV train/eval paths.")
    _require(torch.cuda.is_available(), "CUDA runtime availability.")
    _require(
        str(getattr(target_model, "x_encoder_type", "")).strip().lower() == "split_obs_action",
        "RWKV split_obs_action encoder.",
    )
    _require(
        bool(getattr(target_model, "single_eval_causal", False)),
        "single_eval_causal=True on the RWKV model.",
    )
    _require(
        int(getattr(target_model, "emsize", 0)) > 0,
        "RWKV model emsize metadata.",
    )

    action_slot_dim = int(layout["action_slot_dim"])
    obs_slot_dim = int(layout["obs_slot_dim"])
    _require(
        int(getattr(target_model, "policy_action_dim", action_slot_dim) or action_slot_dim) == action_slot_dim,
        "policy action width to match action_slot_dim exactly.",
    )
    _require(
        int(getattr(target_model.encoder, "obs_dim", layout["x_obs_dim"])) == int(layout["x_obs_dim"]),
        "RWKV encoder obs width to match maintained token layout.",
    )
    _require(
        int(getattr(target_model.encoder, "action_dim", layout["x_action_dim"])) == int(layout["x_action_dim"]),
        "RWKV encoder action width to match maintained token layout.",
    )
    _require(
        int(num_features) == int(layout["default_num_features"]),
        "prior.num_features to match maintained token layout exactly.",
    )

    action_dim_upper_bound = int(env_prior._resolve_dim_upper_bound(env_cfg.get("action_dim", None), default=0))
    _require(
        int(action_dim_upper_bound) > 0,
        "environment.action_dim to resolve to a positive upper bound.",
    )
    _require(
        int(action_dim_upper_bound) <= int(action_slot_dim),
        "environment.action_dim upper bound to be <= action_slot_dim; "
        "policy outputs stay fixed-width and inactive dimensions are masked.",
    )

    for key in ("action_noise_train_std", "action_noise_eval_std"):
        fixed_noise = float(_resolve_fixed_scalar(env_cfg.get(key, 0.0), name=f"environment.{key}"))
        _require(
            abs(float(fixed_noise)) <= 1e-12,
            f"environment.{key}=0.0; extra post-policy action noise would break official PPO old/new semantics.",
        )

    _require(
        str(env_prior._resolve_reinforce_action_transform(env_cfg)).strip().lower() == "none",
        "environment.reinforce_action_transform='none'.",
    )

    q_weight = float(env_prior._resolve_normalized_q_value_weight(env_cfg))
    fm_weight = float(env_prior._resolve_next_state_flow_matching_weight(env_cfg))
    next_state_target_dim = int(
        max(
            int(getattr(target_model, "next_state_flow_dim", 0) or 0),
            int(env_prior._resolve_dim_upper_bound(env_cfg.get("state_dim", None), default=0)),
        )
    )
    if q_weight > 0.0:
        _require(
            hasattr(target_model, "has_normalized_q_value_head")
            and bool(target_model.has_normalized_q_value_head()),
            "normalized_q_value_head enabled on the RWKV model when PPO normalized_q_value_weight>0.",
        )
    if fm_weight > 0.0:
        _require(
            hasattr(target_model, "has_next_state_flow_head")
            and bool(target_model.has_next_state_flow_head()),
            "next_state_flow_head enabled on the RWKV model when PPO next_state_flow_matching_weight>0.",
        )
        _require(
            next_state_target_dim > 0,
            "next_state_flow target width to resolve to a positive value.",
        )

    return {
        "target_model": target_model,
        "layout": layout,
        "obs_slot_dim": obs_slot_dim,
        "action_dim": int(action_slot_dim),
        "action_dim_upper_bound": int(action_dim_upper_bound),
        "q_weight": float(q_weight),
        "fm_weight": float(fm_weight),
        "next_state_target_dim": int(next_state_target_dim),
    }


def resolve_official_ppo_update_batch_size(
    *,
    model,
    n_envs: int,
    n_steps: int,
    configured_batch_size=None,
) -> int:
    total_transitions = int(max(1, int(n_envs) * int(n_steps)))
    if configured_batch_size is not None:
        resolved = int(configured_batch_size)
        if resolved <= 0:
            raise ValueError(
                "Official RecurrentPPO update batch_size must be positive, "
                f"got {resolved}."
            )
        if resolved > total_transitions:
            raise ValueError(
                "Official RecurrentPPO update batch_size cannot exceed one rollout worth of transitions "
                f"({total_transitions}), got {resolved}."
            )
        return int(resolved)
    return int(
        max(
            1,
            min(
                total_transitions,
                int(n_steps) * _resolve_official_ppo_sequence_batch_capacity(
                    model=model,
                    seq_len=int(n_steps),
                    total_batch=int(n_envs),
                ),
            ),
        )
    )


def extract_validation_recurrent_ppo_policy_state(policy) -> dict[str, torch.Tensor]:
    if not isinstance(policy, OfficialRWKVRecurrentPPOPolicy):
        raise TypeError(
            "extract_validation_recurrent_ppo_policy_state expects OfficialRWKVRecurrentPPOPolicy, "
            f"got {type(policy)!r}."
        )
    state = {}
    for key, value in policy.state_dict().items():
        if str(key).startswith("rlpfn_model."):
            continue
        state[str(key)] = value.detach().cpu()
    return state


def build_validation_recurrent_ppo_policy(
    *,
    model,
    env_cfg: dict[str, Any],
    device,
    num_features: int,
    policy_state_dict: Optional[dict[str, torch.Tensor]] = None,
):
    target_model = _target_model(model)
    layout = resolve_rlpfn_token_layout(env_cfg, num_features=num_features)
    obs_slot_dim = int(layout["obs_slot_dim"])
    action_dim = int(layout["action_slot_dim"])
    observation_space = spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(int(num_features),),
        dtype=np.float32,
    )
    action_space = spaces.Box(
        low=-1.0,
        high=1.0,
        shape=(int(action_dim),),
        dtype=np.float32,
    )
    policy = OfficialRWKVRecurrentPPOPolicy(
        observation_space=observation_space,
        action_space=action_space,
        lr_schedule=(lambda _: 0.0),
        rlpfn_model=target_model,
        num_features=int(num_features),
        obs_slot_dim=int(obs_slot_dim),
        net_arch=[],
    )
    if policy_state_dict is not None:
        missing, unexpected = policy.load_state_dict(policy_state_dict, strict=False)
        missing_filtered = [str(key) for key in missing if not str(key).startswith("rlpfn_model.")]
        unexpected_filtered = [str(key) for key in unexpected if not str(key).startswith("rlpfn_model.")]
        if missing_filtered or unexpected_filtered:
            raise ValueError(
                "Failed to restore saved PPO validation policy state. "
                f"missing={missing_filtered}, unexpected={unexpected_filtered}"
            )
    policy.to(device)
    policy.eval()
    return policy


def _resolve_official_ppo_sequence_batch_capacity(
    *,
    model,
    seq_len: int,
    total_batch: int,
) -> int:
    resolve_chunk = getattr(model, "resolve_replay_batch_chunk_size", None)
    if callable(resolve_chunk):
        seq_batch = int(resolve_chunk(seq_len=int(seq_len), total_batch=int(total_batch)))
    else:
        seq_batch = int(total_batch)
    seq_batch = int(max(1, min(int(total_batch), int(seq_batch))))
    # The flat PPO update path now reuses the same memory-safe RWKV replay
    # chunking discipline as REINFORCE: actor/value/Q stay flat and flow keeps
    # only the minimal length-bucket regroup. Reuse the exact RWKV replay chunk
    # capacity instead of a conservative historical shrink factor, which would
    # otherwise under-fill the device and depress update-phase utilization.
    return int(seq_batch)


class EnvironmentPriorPPOGymEnv(Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        env_prior: EnvironmentPrior,
        device,
        num_features: int,
        n_steps: int,
        action_dim: int,
        obs_slot_dim: int,
        action_slot_dim: int,
        next_state_target_dim: int,
    ):
        super().__init__()
        self.env_prior = env_prior
        self.device = torch.device(device)
        self.num_features = int(num_features)
        self.n_steps = int(n_steps)
        self.action_dim = int(action_dim)
        self.obs_slot_dim = int(obs_slot_dim)
        self.action_slot_dim = int(action_slot_dim)
        self.next_state_target_dim = int(next_state_target_dim)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_features,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.action_dim,),
            dtype=np.float32,
        )
        self._generator: Optional[torch.Generator] = None
        self._env = None
        self._state_t = None
        self._action_t = None
        self._reward_t = None
        self._reward_mask_t = None
        self._terminal_t = None
        self._terminal_signal_history = None
        self._zero_pad_t = None
        self._step_idx = 0
        self._single_eval_pos = 1
        self._env_action_dim = int(self.action_dim)

    def _build_next_state_target(self, state_t: torch.Tensor):
        state_dim = int(state_t.shape[-1])
        if state_dim > self.next_state_target_dim:
            raise RuntimeError(
                f"Sampled PPO env state_dim={state_dim} exceeds next_state_target_dim={self.next_state_target_dim}."
            )
        target = torch.zeros((self.next_state_target_dim,), device=self.device, dtype=state_t.dtype)
        mask = torch.zeros((self.next_state_target_dim,), device=self.device, dtype=state_t.dtype)
        if state_dim > 0:
            target[:state_dim] = state_t
            mask[:state_dim] = 1.0
        return target, mask

    def _build_obs_token(self) -> np.ndarray:
        assert self._env is not None
        token = torch.zeros((self.num_features,), device=self.device, dtype=torch.float32)
        obs_dim = int(self._env["obs_dim"])
        obs_t = self._state_t[:obs_dim]
        token_obs_cap = min(obs_dim, self.obs_slot_dim, self.num_features)
        if token_obs_cap > 0:
            token[:token_obs_cap] = obs_t[:token_obs_cap]
        reward_idx = self.obs_slot_dim
        mask_idx = self.obs_slot_dim + 1
        phase_idx = self.obs_slot_dim + 2
        terminal_enabled = bool(self._env.get("terminal_reset_enabled", False))
        terminal_idx = self.obs_slot_dim + 3
        action_start = self.obs_slot_dim + 3 + int(terminal_enabled)
        action_copy = min(self._env_action_dim, self.action_slot_dim, max(0, self.num_features - action_start))
        if reward_idx < self.num_features:
            token[reward_idx] = self._reward_t
        if mask_idx < self.num_features:
            token[mask_idx] = self._reward_mask_t
        if phase_idx < self.num_features:
            token[phase_idx] = float(self._step_idx >= self._single_eval_pos)
        if terminal_enabled and terminal_idx < self.num_features:
            token[terminal_idx] = self._terminal_t
        if action_copy > 0:
            token[action_start : action_start + action_copy] = self._action_t[:action_copy]
        return token.detach().cpu().numpy().astype(np.float32, copy=False)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None):
        super().reset(seed=seed)
        del options
        if seed is None:
            seed = int(self.np_random.integers(0, 2**31 - 1, dtype=np.int64))
        self._generator = torch.Generator(device=self.device)
        self._generator.manual_seed(int(seed))
        h = self.env_prior._sample_batch_hypers(1)[0]
        self._env = self.env_prior._sample_environment(h=h, device=self.device, rng_seed=int(seed))
        self._env_action_dim = int(self._env["action_dim"])
        if self._env_action_dim <= 0 or self._env_action_dim > self.action_dim:
            raise RuntimeError(
                f"Variable-action PPO expected sampled action_dim in [1, {self.action_dim}], got {self._env_action_dim}."
            )
        self._single_eval_pos = int(self.env_prior._sample_single_eval_pos(self.n_steps, None))
        state_dim = int(self._env["state_dim"])
        zero_pad_dim = int(self._env["zero_pad_dim"])
        init_state_std = float(self._env.get("init_state_std", 0.0))
        init_action_std = float(self._env.get("init_action_std", 0.0))
        self._state_t = torch.randn(
            (state_dim,),
            device=self.device,
            dtype=torch.float32,
            generator=self._generator,
        ) * init_state_std
        self._action_t = torch.zeros((self.action_dim,), device=self.device, dtype=torch.float32)
        if self._env_action_dim > 0:
            self._action_t[: self._env_action_dim] = torch.randn(
                (self._env_action_dim,),
                device=self.device,
                dtype=torch.float32,
                generator=self._generator,
            ) * init_action_std
        self._reward_t = torch.zeros((), device=self.device, dtype=torch.float32)
        self._reward_mask_t = torch.ones((), device=self.device, dtype=torch.float32)
        self._terminal_t = torch.zeros((), device=self.device, dtype=torch.float32)
        self._terminal_signal_history = (
            torch.zeros((self.n_steps, 1), device=self.device, dtype=torch.float32)
            if bool(self._env.get("terminal_reset_enabled", False))
            else None
        )
        self._zero_pad_t = torch.zeros((zero_pad_dim,), device=self.device, dtype=torch.float32)
        self._step_idx = 0
        return self._build_obs_token(), {"single_eval_pos": int(self._single_eval_pos)}

    def action_masks(self) -> np.ndarray:
        mask = np.zeros((self.action_dim,), dtype=np.float32)
        mask[: int(max(0, min(self.action_dim, self._env_action_dim)))] = 1.0
        return mask

    def step(self, action):
        assert self._env is not None and self._generator is not None
        state_dim = int(self._env["state_dim"])
        obs_dim = int(self._env["obs_dim"])
        noise_dim = int(self._env["noise_dim"])
        reference_semantics_enabled = bool(self._env.get("reference_semantics_enabled", False))
        transition_generator = self._env.get("transition_generator", None)
        state_input_scale = self._env.get("state_input_scale", 1.0)

        action_t = torch.as_tensor(action, device=self.device, dtype=torch.float32).reshape(self.action_dim)
        action_t = torch.clamp(action_t, min=-1.0, max=1.0)
        action_mask_t = torch.zeros((self.action_dim,), device=self.device, dtype=torch.float32)
        action_mask_t[: self._env_action_dim] = 1.0
        action_t = action_t * action_mask_t
        action_env_t = action_t[: self._env_action_dim]
        obs_t = self._state_t[:obs_dim]
        if noise_dim > 0:
            noise_t = torch.randn(
                (noise_dim,),
                device=self.device,
                dtype=torch.float32,
                generator=self._generator,
            )
        else:
            noise_t = torch.zeros((0,), device=self.device, dtype=torch.float32)

        env_in = self.env_prior._pack_env_input(
            self._state_t,
            obs_t,
            action_env_t,
            noise_t,
            self._zero_pad_t,
            reference_semantics_enabled=reference_semantics_enabled,
            state_input_scale=state_input_scale,
        ).unsqueeze(0)

        terminal_signal_next = None
        terminal_bonus_base_next = None
        if callable(transition_generator):
            transition_out = transition_generator(env_in, generator=self._generator)
            x_next, reward_unit, terminal_signal_next, terminal_bonus_base_next = self.env_prior._unpack_transition_output(
                transition_out
            )
            x_next = x_next.squeeze(0)
            reward_next_raw = float(self._env["reward_scale"]) * reward_unit.reshape(())
        else:
            x_next = self._env["x_generator"](env_in, generator=self._generator).squeeze(0)
            reward_next_raw = float(self._env["reward_scale"]) * self._env["y_generator"](
                env_in,
                generator=self._generator,
            ).reshape(())

        aux_reward_next = self.env_prior._exact_scm_aux_reward_terms(
            action_t,
            ctrl_weight=self._env.get("ctrl_reward_weight", 0.0),
            ctrl_enabled=self._env.get("ctrl_reward_enabled", False),
            survival_weight=self._env.get("survival_reward_weight", 0.0),
            survival_enabled=self._env.get("survival_reward_enabled", False),
        )
        reward_next = self.env_prior._compose_exact_scm_reward(
            reward_next_raw,
            aux_reward=aux_reward_next,
            reward_clip=self._env.get("reward_clip", 10.0),
            mode=self._env.get("reinforce_reward_transform", "none"),
            rms_eps=self._env.get("reinforce_reward_rms_eps", 1e-6),
            tanh_c=self._env.get("reinforce_reward_tanh_c", 1.0),
            tanh_bound=self._env.get("reinforce_reward_tanh_bound", 2.0),
        )
        reward_mask_next = torch.ones((), device=self.device, dtype=torch.float32)
        reward_dropout_ratio = float(self._env.get("reward_dropout_ratio", 0.0))
        if bool(self._env.get("reward_dropout_enabled", False)) and reward_dropout_ratio > 0.0:
            draw = torch.rand((), device=self.device, dtype=torch.float32, generator=self._generator)
            if bool(draw < reward_dropout_ratio):
                reward_mask_next = torch.zeros((), device=self.device, dtype=torch.float32)
                if bool(self._env.get("reward_dropout_impute_zero", True)):
                    reward_next = torch.zeros_like(reward_next)

        alpha = float(self._env["alpha"])
        state_next = (1.0 - alpha) * self._state_t + alpha * x_next
        state_noise_std = float(self._env.get("state_noise_std", 0.0))
        if state_noise_std > 0.0:
            state_next = state_next + torch.randn(
                (state_dim,),
                device=self.device,
                dtype=torch.float32,
                generator=self._generator,
            ) * state_noise_std
        if not reference_semantics_enabled:
            state_next = self.env_prior._apply_state_postprocess(
                state_next_raw=state_next,
                state_prev=self._state_t,
                state_clip=self._env.get("state_clip", 8.0),
                state_highway_enabled=self._env.get("state_highway_enabled", False),
                state_highway_lambda=self._env.get("state_highway_lambda", 0.0),
            )
        state_next = self.env_prior._apply_state_full_rms(
            state_next,
            enabled=self._env.get("state_full_rms_enabled", False),
            target=self._env.get("state_full_rms_target", 1.0),
        )

        if bool(self._env.get("terminal_reset_enabled", False)):
            reset_prob = self.env_prior._terminal_reset_prob_from_count(
                self._env.get("terminal_reset_count_target", 0.0),
                self.n_steps,
            )
            bonus_scale_draw = torch.rand((), device=self.device, dtype=torch.float32, generator=self._generator)
            terminal_draw = torch.rand((), device=self.device, dtype=torch.float32, generator=self._generator)
            reset_state = torch.randn(
                (state_dim,),
                device=self.device,
                dtype=torch.float32,
                generator=self._generator,
            ) * float(self._env.get("init_state_std", 0.0))
            state_next, reward_next, terminal_next = self.env_prior._apply_terminal_reset_step(
                state_next=state_next.unsqueeze(0),
                reward_next=reward_next.reshape(1),
                terminal_draw=terminal_draw.reshape(1),
                reset_prob=reset_prob,
                bonus_scale_draw=bonus_scale_draw.reshape(1),
                bonus_scale_min=self._env.get("terminal_bonus_scale_min", 1.0),
                bonus_scale_max=self._env.get("terminal_bonus_scale_max", 2.0),
                bonus_tanh_c=self._env.get("terminal_bonus_tanh_c", 10.0),
                reset_state=reset_state.unsqueeze(0),
                enabled=True,
                terminal_signal=None if terminal_signal_next is None else terminal_signal_next.reshape(1, 1),
                terminal_bonus_base=None if terminal_bonus_base_next is None else terminal_bonus_base_next.reshape(1),
                terminal_signal_history=self._terminal_signal_history,
                history_index=self._step_idx,
            )
            state_next = state_next.squeeze(0)
            reward_next = reward_next.reshape(())
            terminal_next = terminal_next.reshape(())
        else:
            terminal_next = torch.zeros((), device=self.device, dtype=reward_next.dtype)

        self._state_t = state_next
        self._action_t = action_t
        self._reward_t = reward_next.reshape(())
        self._reward_mask_t = reward_mask_next.reshape(())
        self._terminal_t = terminal_next.reshape(())

        self._step_idx += 1
        truncated = bool(self._step_idx >= self.n_steps)
        obs_next = self._build_obs_token()
        next_state_target, next_state_mask = self._build_next_state_target(self._state_t)
        info = {
            "single_eval_pos": int(self._single_eval_pos),
            "terminal_flag": float(self._terminal_t.detach().cpu().item()),
            "next_state_target": next_state_target.detach().cpu().numpy().astype(np.float32, copy=False),
            "next_state_mask": next_state_mask.detach().cpu().numpy().astype(np.float32, copy=False),
        }
        return obs_next, float(self._reward_t.detach().cpu().item()), False, truncated, info


class EnvironmentPriorPPOBatchVecEnv(VecEnv):
    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        env_prior: EnvironmentPrior,
        device,
        num_features: int,
        n_steps: int,
        num_envs: int,
        action_dim: int,
        obs_slot_dim: int,
        action_slot_dim: int,
        next_state_target_dim: int,
    ):
        self.env_prior = env_prior
        self.device = torch.device(device)
        self.num_features = int(num_features)
        self.n_steps = int(n_steps)
        self.action_dim = int(action_dim)
        self.obs_slot_dim = int(obs_slot_dim)
        self.action_slot_dim = int(action_slot_dim)
        self.next_state_target_dim = int(next_state_target_dim)
        self.render_mode = None
        self._strict_rng_match = bool(self.env_prior._resolve_batch_vectorized_strict_rng_match())
        observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_features,),
            dtype=np.float32,
        )
        action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.action_dim,),
            dtype=np.float32,
        )
        super().__init__(int(num_envs), observation_space, action_space)
        self._pending_actions: Optional[np.ndarray] = None
        self._h_list = None
        self._env = None
        self._rollout_generators = None
        self._state_t = None
        self._action_t = None
        self._reward_t = None
        self._reward_mask_t = None
        self._terminal_t = None
        self._terminal_signal_history = None
        self._zero_pad_t = None
        self._single_eval_pos = None
        self._single_eval_pos_np = None
        self._action_masks_np = None
        self._step_idx = 0
        self._episode_returns = np.zeros((self.num_envs,), dtype=np.float32)
        self._episode_lengths = np.zeros((self.num_envs,), dtype=np.int32)

    def _vector_action_masks(self) -> np.ndarray:
        if self._action_masks_np is None:
            action_dims = self._env["action_dim_per_sample"].detach().cpu().numpy().astype(np.int64, copy=True)
            action_positions = np.arange(self.action_dim, dtype=np.int64)[None, :]
            self._action_masks_np = (action_positions < action_dims[:, None]).astype(np.float32, copy=False)
        return self._action_masks_np

    def _build_next_state_targets(self) -> tuple[np.ndarray, np.ndarray]:
        state_dims = self._env["state_dim_per_sample"]
        cap = int(min(int(self._env["state_dim"]), self.next_state_target_dim))
        target = torch.zeros((self.num_envs, self.next_state_target_dim), device=self.device, dtype=torch.float32)
        mask = torch.zeros_like(target)
        if cap > 0:
            target[:, :cap] = self._state_t[:, :cap]
            cols = torch.arange(cap, device=self.device, dtype=torch.long).unsqueeze(0)
            valid = cols < state_dims.unsqueeze(1)
            mask[:, :cap] = valid.to(dtype=target.dtype)
        return (
            target.detach().cpu().numpy().astype(np.float32, copy=False),
            mask.detach().cpu().numpy().astype(np.float32, copy=False),
        )

    def _build_obs_batch(self) -> np.ndarray:
        token = torch.zeros((self.num_envs, self.num_features), device=self.device, dtype=torch.float32)
        obs_dims = self._env["obs_dim_per_sample"].to(device=self.device, dtype=torch.long)
        obs_slot_dims = self._env["obs_slot_dim_per_sample"].to(device=self.device, dtype=torch.long)
        terminal_enabled = self._env["terminal_reset_enabled"].to(device=self.device, dtype=torch.bool)
        reward_idx = obs_slot_dims
        mask_idx = obs_slot_dims + 1
        phase_idx = obs_slot_dims + 2
        terminal_idx = obs_slot_dims + 3
        action_start = obs_slot_dims + 3 + terminal_enabled.to(dtype=torch.long)
        rows = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

        obs_cap = torch.minimum(obs_dims, obs_slot_dims)
        obs_write_cap = int(min(int(self._env["obs_dim"]), self.num_features))
        if obs_write_cap > 0:
            cols = torch.arange(obs_write_cap, device=self.device, dtype=torch.long).unsqueeze(0)
            valid = cols < obs_cap.unsqueeze(1)
            token[:, :obs_write_cap] = self._state_t[:, :obs_write_cap] * valid.to(dtype=token.dtype)

        valid = reward_idx < self.num_features
        if bool(valid.any().item()):
            token[rows[valid], reward_idx[valid]] = self._reward_t[valid]
        valid = mask_idx < self.num_features
        if bool(valid.any().item()):
            token[rows[valid], mask_idx[valid]] = self._reward_mask_t[valid]
        valid = phase_idx < self.num_features
        if bool(valid.any().item()):
            phase_value = (self._step_idx >= self._single_eval_pos).to(dtype=token.dtype)
            token[rows[valid], phase_idx[valid]] = phase_value[valid]
        valid = terminal_enabled & (terminal_idx < self.num_features)
        if bool(valid.any().item()):
            token[rows[valid], terminal_idx[valid]] = self._terminal_t[valid]

        action_dims = self._env["action_dim_per_sample"].to(device=self.device, dtype=torch.long)
        action_slot_dims = self._env["action_slot_dim_per_sample"].to(device=self.device, dtype=torch.long)
        action_cap = torch.minimum(action_dims, action_slot_dims)
        action_write_cap = int(min(self.action_dim, self.num_features))
        if action_write_cap > 0:
            action_pos = torch.arange(action_write_cap, device=self.device, dtype=torch.long).unsqueeze(0)
            dst_cols = action_start.unsqueeze(1) + action_pos
            valid = (action_pos < action_cap.unsqueeze(1)) & (dst_cols < self.num_features)
            if bool(valid.any().item()):
                token[rows.unsqueeze(1).expand_as(dst_cols)[valid], dst_cols[valid]] = self._action_t[:, :action_write_cap][valid]

        return token.detach().cpu().numpy().astype(np.float32, copy=False)

    def _sample_seed_list(self) -> list[int]:
        seeds: list[int] = []
        for env_idx in range(self.num_envs):
            seed = self._seeds[env_idx]
            if seed is None:
                seed = int(np.random.randint(0, 2**31 - 1, dtype=np.int64))
            seeds.append(int(seed))
        self._reset_seeds()
        return seeds

    def _full_reset_batch(self, *, seeds: Optional[list[int]] = None) -> np.ndarray:
        if seeds is None:
            if self._strict_rng_match:
                seeds = self._sample_seed_list()
            else:
                self._reset_seeds()
                seeds = None
        else:
            seeds = [int(s) for s in seeds]
            self._reset_seeds()
        self._h_list = self.env_prior._sample_batch_hypers(self.num_envs)
        family_values = {self.env_prior._normalize_family(h.get("family", "scm")) for h in self._h_list}
        if len(family_values) != 1:
            raise RuntimeError(
                "Strict PPO batched VecEnv currently requires a fixed environment family across the batch. "
                f"Got families={sorted(family_values)}."
            )
        self._env = self.env_prior._sample_environment_family_coarse_batch(
            self._h_list,
            device=self.device,
            rng_seeds=seeds,
            build_policy_generator=False,
        )
        self._rollout_generators = (
            self.env_prior._make_generators_from_seeds(seeds, self.num_envs, self.device)
            if self._strict_rng_match and seeds is not None
            else None
        )
        state_dim = int(self._env["state_dim"])
        zero_pad_dim = int(self._env["zero_pad_dim"])
        self._state_t = self.env_prior._stack_randn_with_generators(
            self._rollout_generators,
            (self.num_envs, state_dim),
            device=self.device,
            dtype=torch.float32,
        ) * self._env["init_state_std"][:, None]
        self._action_t = torch.zeros((self.num_envs, self.action_dim), device=self.device, dtype=torch.float32)
        init_action = self.env_prior._stack_randn_with_generators(
            self._rollout_generators,
            (self.num_envs, self.action_dim),
            device=self.device,
            dtype=torch.float32,
        ) * self._env["init_action_std"][:, None]
        self._action_t.copy_(init_action * torch.as_tensor(self._vector_action_masks(), device=self.device, dtype=torch.float32))
        self._reward_t = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
        self._reward_mask_t = torch.ones((self.num_envs,), device=self.device, dtype=torch.float32)
        self._terminal_t = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
        self._terminal_signal_history = (
            torch.zeros((self.n_steps, self.num_envs), device=self.device, dtype=torch.float32)
            if bool(self._env["terminal_reset_enabled"].any().item())
            else None
        )
        self._zero_pad_t = torch.zeros((self.num_envs, zero_pad_dim), device=self.device, dtype=torch.float32)
        self._single_eval_pos = torch.tensor(
            [int(self.env_prior._sample_single_eval_pos(self.n_steps, None)) for _ in range(self.num_envs)],
            device=self.device,
            dtype=torch.long,
        )
        self._single_eval_pos_np = self._single_eval_pos.detach().cpu().numpy().astype(np.int64, copy=True)
        self._action_masks_np = None
        self._step_idx = 0
        self._episode_returns.fill(0.0)
        self._episode_lengths.fill(0)
        self.reset_infos = [{"single_eval_pos": int(v)} for v in self._single_eval_pos_np.tolist()]
        return self._build_obs_batch()

    def reset(self) -> VecEnvObs:
        self._reset_options()
        return self._full_reset_batch()

    def step_async(self, actions: np.ndarray) -> None:
        self._pending_actions = np.asarray(actions, dtype=np.float32).reshape((self.num_envs, self.action_dim))

    def step_wait(self) -> VecEnvStepReturn:
        if self._pending_actions is None:
            raise RuntimeError("step_async() must be called before step_wait().")
        action_t = torch.as_tensor(self._pending_actions, device=self.device, dtype=torch.float32)
        action_t = torch.clamp(action_t, min=-1.0, max=1.0)
        action_mask_t = torch.as_tensor(self._vector_action_masks(), device=self.device, dtype=torch.float32)
        action_t = action_t * action_mask_t

        obs_dim = int(self._env["obs_dim"])
        noise_dim = int(self._env["noise_dim"])
        env_total_dim = int(self._env["env_input_dim"])
        env_obs_input_dim = int(self._env["env_obs_input_dim"])
        env_action_start = int(self._env["action_dim"]) + int(self._env["state_dim"]) + int(env_obs_input_dim)  # overwritten below
        env_layout_action_start = int(self._env["state_dim"] + self._env["env_obs_input_dim"])
        env_action_start = int(env_layout_action_start)
        env_noise_start = int(env_action_start + self._env["action_dim"])

        obs_t = self._state_t[:, :obs_dim]
        noise_t = self.env_prior._stack_randn_with_generators(
            self._rollout_generators,
            (self.num_envs, noise_dim),
            device=self.device,
            dtype=torch.float32,
        ) if noise_dim > 0 else torch.zeros((self.num_envs, 0), device=self.device, dtype=torch.float32)

        env_in = torch.zeros((self.num_envs, env_total_dim), device=self.device, dtype=torch.float32)
        env_in[:, : int(self._env["state_dim"])] = self.env_prior._scale_state_env_input(
            self._state_t,
            self._env.get("state_input_scale", 1.0),
        )
        if env_obs_input_dim > 0:
            env_in[:, int(self._env["state_dim"]) : int(self._env["state_dim"]) + env_obs_input_dim] = obs_t[:, :env_obs_input_dim]
        env_in[:, env_action_start : env_action_start + self.action_dim] = action_t
        if noise_dim > 0:
            env_in[:, env_noise_start : env_noise_start + noise_dim] = noise_t

        transition_generator = self._env.get("transition_generator", None)
        terminal_signal_next = None
        terminal_bonus_base_next = None
        if callable(transition_generator):
            transition_out = transition_generator(env_in, generators_for_noise=self._rollout_generators)
            x_next, reward_unit, terminal_signal_next, terminal_bonus_base_next = self.env_prior._unpack_transition_output(
                transition_out
            )
            reward_next_raw = self._env["reward_scale"] * reward_unit.reshape(self.num_envs)
        else:
            x_next = self._env["x_generator"](env_in, generators_for_noise=self._rollout_generators)
            reward_next_raw = self._env["reward_scale"] * self._env["y_generator"](
                env_in,
                generators_for_noise=self._rollout_generators,
            ).reshape(self.num_envs)

        aux_reward_next = self.env_prior._exact_scm_aux_reward_terms(
            action_t,
            action_mask=action_mask_t,
            ctrl_weight=self._env.get("ctrl_reward_weight", 0.0),
            ctrl_enabled=self._env.get("ctrl_reward_enabled", False),
            survival_weight=self._env.get("survival_reward_weight", 0.0),
            survival_enabled=self._env.get("survival_reward_enabled", False),
        )
        reward_next = self.env_prior._compose_exact_scm_reward(
            reward_next_raw,
            aux_reward=aux_reward_next,
            reward_clip=self._env.get("reward_clip", 10.0),
            mode=self._env.get("reinforce_reward_transform", "none"),
            rms_eps=self._env.get("reinforce_reward_rms_eps", 1e-6),
            tanh_c=self._env.get("reinforce_reward_tanh_c", 1.0),
            tanh_bound=self._env.get("reinforce_reward_tanh_bound", 2.0),
        )
        reward_mask_next = torch.ones((self.num_envs,), device=self.device, dtype=torch.float32)
        dropout_enabled = self._env.get("reward_dropout_enabled", False)
        dropout_ratio = self._env.get("reward_dropout_ratio", 0.0)
        if bool(torch.any(dropout_enabled).item()) and bool(torch.any(dropout_ratio > 0).item()):
            drop_draw = self.env_prior._stack_rand_with_generators(
                self._rollout_generators,
                (self.num_envs,),
                device=self.device,
                dtype=torch.float32,
            )
            drop_mask = dropout_enabled & (drop_draw < dropout_ratio)
            reward_mask_next = torch.where(drop_mask, torch.zeros_like(reward_mask_next), reward_mask_next)
            impute_mask = drop_mask & self._env.get("reward_dropout_impute_zero", False)
            reward_next = torch.where(impute_mask, torch.zeros_like(reward_next), reward_next)

        state_next = (1.0 - self._env["alpha"][:, None]) * self._state_t + self._env["alpha"][:, None] * x_next
        if bool(torch.any(self._env["state_noise_std"] > 0).item()):
            state_noise = self.env_prior._stack_randn_with_generators(
                self._rollout_generators,
                (self.num_envs, int(self._env["state_dim"])),
                device=self.device,
                dtype=torch.float32,
            )
            state_next = state_next + state_noise * self._env["state_noise_std"][:, None]
        state_post = self.env_prior._apply_state_postprocess(
            state_next_raw=state_next,
            state_prev=self._state_t,
            state_clip=self._env.get("state_clip", 8.0),
            state_highway_enabled=self._env.get("state_highway_enabled", False),
            state_highway_lambda=self._env.get("state_highway_lambda", 0.0),
        )
        reference_mask = self._env.get("reference_semantics_enabled", False)
        if torch.is_tensor(reference_mask):
            reference_mask = reference_mask.to(device=self.device, dtype=torch.bool)
            state_next = torch.where(reference_mask.unsqueeze(-1), state_next, state_post)
        elif bool(reference_mask):
            pass
        else:
            state_next = state_post
        state_next = self.env_prior._apply_state_full_rms(
            state_next,
            enabled=self._env.get("state_full_rms_enabled", False),
            target=self._env.get("state_full_rms_target", 1.0),
        )

        if bool(torch.any(self._env.get("terminal_reset_enabled", False)).item()):
            reset_prob = self.env_prior._terminal_reset_prob_from_count(
                self._env.get("terminal_reset_count_target", 0.0),
                self.n_steps,
            )
            bonus_scale_draw = self.env_prior._stack_rand_with_generators(
                self._rollout_generators,
                (self.num_envs,),
                device=self.device,
                dtype=torch.float32,
            )
            terminal_draw = self.env_prior._stack_rand_with_generators(
                self._rollout_generators,
                (self.num_envs,),
                device=self.device,
                dtype=torch.float32,
            )
            reset_state = self.env_prior._stack_randn_with_generators(
                self._rollout_generators,
                (self.num_envs, int(self._env["state_dim"])),
                device=self.device,
                dtype=torch.float32,
            ) * self._env["init_state_std"][:, None]
            state_next, reward_next, terminal_next = self.env_prior._apply_terminal_reset_step(
                state_next=state_next,
                reward_next=reward_next,
                terminal_draw=terminal_draw,
                reset_prob=reset_prob,
                bonus_scale_draw=bonus_scale_draw,
                bonus_scale_min=self._env.get("terminal_bonus_scale_min", 1.0),
                bonus_scale_max=self._env.get("terminal_bonus_scale_max", 2.0),
                bonus_tanh_c=self._env.get("terminal_bonus_tanh_c", 10.0),
                reset_state=reset_state,
                enabled=self._env.get("terminal_reset_enabled", False),
                terminal_signal=terminal_signal_next,
                terminal_bonus_base=terminal_bonus_base_next,
                terminal_signal_history=self._terminal_signal_history,
                history_index=self._step_idx,
            )
        else:
            terminal_next = torch.zeros((self.num_envs,), device=self.device, dtype=reward_next.dtype)

        self._state_t = state_next
        self._action_t = action_t
        self._reward_t = reward_next
        self._reward_mask_t = reward_mask_next
        self._terminal_t = terminal_next.to(dtype=reward_next.dtype)
        self._step_idx += 1

        obs_next = self._build_obs_batch()
        next_state_target, next_state_mask = self._build_next_state_targets()
        rewards_np = self._reward_t.detach().cpu().numpy().astype(np.float32, copy=False)
        terminal_flags_np = self._terminal_t.detach().cpu().numpy().astype(np.float32, copy=False)
        single_eval_pos_np = self._single_eval_pos_np
        if single_eval_pos_np is None:
            single_eval_pos_np = self._single_eval_pos.detach().cpu().numpy().astype(np.int64, copy=True)
            self._single_eval_pos_np = single_eval_pos_np
        dones = np.full((self.num_envs,), bool(self._step_idx >= self.n_steps), dtype=bool)
        self._episode_returns += rewards_np
        self._episode_lengths += 1
        infos = [
            {
                "single_eval_pos": int(single_eval_pos_np[env_idx]),
                "terminal_flag": float(terminal_flags_np[env_idx]),
                "next_state_target": next_state_target[env_idx],
                "next_state_mask": next_state_mask[env_idx],
                "TimeLimit.truncated": bool(dones[env_idx]),
            }
            for env_idx in range(self.num_envs)
        ]

        if bool(dones.any()):
            terminal_obs = obs_next.copy()
            for env_idx in range(self.num_envs):
                infos[env_idx]["episode"] = {
                    "r": float(self._episode_returns[env_idx]),
                    "l": int(self._episode_lengths[env_idx]),
                }
                infos[env_idx]["terminal_observation"] = terminal_obs[env_idx].copy()
            obs_next = self._full_reset_batch()

        self._pending_actions = None
        return obs_next, rewards_np.copy(), dones, infos

    def close(self) -> None:
        return None

    def get_images(self):
        return [None for _ in range(self.num_envs)]

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> list[Any]:
        target_indices = list(self._get_indices(indices))
        if attr_name == "render_mode":
            return [None for _ in target_indices]
        if attr_name == "action_masks":
            return [self.action_masks for _ in target_indices]
        values = []
        for _ in target_indices:
            if not hasattr(self, attr_name):
                raise AttributeError(attr_name)
            values.append(getattr(self, attr_name))
        return values

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        for _ in self._get_indices(indices):
            setattr(self, attr_name, value)

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> list[Any]:
        target_indices = list(self._get_indices(indices))
        if method_name == "action_masks":
            masks = self.action_masks()
            return [masks[idx] for idx in target_indices]
        if not hasattr(self, method_name):
            raise AttributeError(method_name)
        method = getattr(self, method_name)
        return [method(*method_args, **method_kwargs) for _ in target_indices]

    def env_is_wrapped(self, wrapper_class: type[Env], indices: VecEnvIndices = None) -> list[bool]:
        del wrapper_class
        return [False for _ in self._get_indices(indices)]

    def action_masks(self) -> np.ndarray:
        return self._vector_action_masks()


class OfficialRWKVRecurrentPPOPolicy(RecurrentActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        *,
        rlpfn_model,
        num_features: int,
        obs_slot_dim: int,
        net_arch=None,
        **kwargs,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=[] if net_arch is None else net_arch,
            lstm_hidden_size=int(getattr(rlpfn_model, "emsize")),
            enable_critic_lstm=False,
            shared_lstm=False,
            **kwargs,
        )
        self.rlpfn_model = rlpfn_model
        self.num_features = int(num_features)
        self.obs_slot_dim = int(obs_slot_dim)
        self.lstm_actor = _PlaceholderLSTM()
        self.lstm_critic = None
        self.critic = nn.Identity()
        self.lstm_hidden_state_shape = (1, 1, 1)
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
        self._rollout_kv_cache = None
        self._rollout_cache_batch_size = None

    def _coerce_action_masks(self, action_masks, batch_size: int, *, device: torch.device) -> Optional[torch.Tensor]:
        if action_masks is None:
            return None
        if torch.is_tensor(action_masks):
            mask = action_masks.to(device=device, dtype=torch.float32)
        else:
            mask = torch.as_tensor(action_masks, device=device, dtype=torch.float32)
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        if tuple(mask.shape) != (int(batch_size), int(self.action_space.shape[0])):
            raise ValueError(
                "action_masks must match batch/action shape, "
                f"got {tuple(mask.shape)} and expected {(int(batch_size), int(self.action_space.shape[0]))}"
            )
        return mask

    @staticmethod
    def _masked_action_tensor(actions: torch.Tensor, action_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if action_mask is None:
            return actions
        return actions * action_mask.to(device=actions.device, dtype=actions.dtype)

    def _masked_diag_gaussian_stats(
        self,
        distribution,
        actions: torch.Tensor,
        action_mask: Optional[torch.Tensor],
    ):
        if action_mask is None:
            return distribution.log_prob(actions), distribution.entropy()
        if not isinstance(self.action_dist, DiagGaussianDistribution):
            raise RuntimeError("Variable-action PPO currently supports only DiagGaussianDistribution.")
        mask_f = action_mask.to(device=actions.device, dtype=actions.dtype)
        per_dim_log_prob = distribution.distribution.log_prob(actions)
        per_dim_entropy = distribution.distribution.entropy()
        log_prob = (per_dim_log_prob * mask_f).sum(dim=1)
        entropy = (per_dim_entropy * mask_f).sum(dim=1)
        return log_prob, entropy

    @staticmethod
    def _flat_to_time_major_sequence(tensor: torch.Tensor, *, n_seq: int) -> torch.Tensor:
        seq_len = _resolve_padded_sequence_length(int(tensor.shape[0]), n_seq=int(n_seq))
        return tensor.reshape((int(n_seq), int(seq_len), *tensor.shape[1:])).swapaxes(0, 1)

    @staticmethod
    def _time_major_to_flat_sequence(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.swapaxes(0, 1).reshape((-1, *tensor.shape[2:]))

    def decode_aux_from_hidden(
        self,
        hidden: torch.Tensor,
        *,
        actions: torch.Tensor,
        flow_matching_xt: Optional[torch.Tensor] = None,
        flow_matching_t: Optional[torch.Tensor] = None,
        n_seq: Optional[int] = None,
        seq_lengths: Optional[np.ndarray] = None,
    ):
        outputs = {}
        if hasattr(self.rlpfn_model, "has_normalized_q_value_head") and bool(self.rlpfn_model.has_normalized_q_value_head()):
            outputs["normalized_q_logits"] = self.rlpfn_model._decode_normalized_q_value_logits(
                hidden,
                action=actions,
            )
        if (
            flow_matching_xt is not None
            and flow_matching_t is not None
            and hasattr(self.rlpfn_model, "has_next_state_flow_head")
            and bool(self.rlpfn_model.has_next_state_flow_head())
        ):
            hidden_flow = hidden
            action_flow = actions
            flow_xt = flow_matching_xt
            flow_t = flow_matching_t
            flatten_output = False
            if hidden_flow.ndim == 2:
                if seq_lengths is not None:
                    seq_lengths = np.asarray(seq_lengths, dtype=np.int64)
                    if int(seq_lengths.sum()) != int(hidden_flow.shape[0]):
                        raise ValueError(
                            "decode_aux_from_hidden seq_lengths must sum to the flattened hidden length, "
                            f"got sum={int(seq_lengths.sum())} and hidden={tuple(hidden_flow.shape)}"
                        )
                    next_state_flow = self._decode_flow_from_hidden_flat_packed(
                        hidden_flow,
                        action_flow,
                        flow_xt,
                        flow_t,
                        seq_lengths=seq_lengths,
                    )
                else:
                    if n_seq is None:
                        raise ValueError(
                            "decode_aux_from_hidden requires n_seq or seq_lengths when decoding flow from flattened hidden."
                        )
                    hidden_flow = self._flat_to_time_major_sequence(hidden_flow, n_seq=int(n_seq))
                    action_flow = self._flat_to_time_major_sequence(action_flow, n_seq=int(n_seq))
                    flow_xt = self._flat_to_time_major_sequence(flow_xt, n_seq=int(n_seq))
                    flow_t = self._flat_to_time_major_sequence(flow_t, n_seq=int(n_seq))
                    flatten_output = True
                    next_state_flow = self.rlpfn_model._decode_next_state_flow(
                        hidden_flow,
                        flow_xt,
                        flow_t,
                        action=action_flow,
                    )
            else:
                next_state_flow = self.rlpfn_model._decode_next_state_flow(
                    hidden_flow,
                    flow_xt,
                    flow_t,
                    action=action_flow,
                )
            if flatten_output:
                next_state_flow = self._time_major_to_flat_sequence(next_state_flow)
            outputs["next_state_flow"] = next_state_flow
        return outputs

    def _decode_flow_from_hidden_flat_exact_buckets(
        self,
        hidden_flow: torch.Tensor,
        action_flow: torch.Tensor,
        flow_xt: torch.Tensor,
        flow_t: torch.Tensor,
        *,
        seq_lengths: np.ndarray,
    ) -> torch.Tensor:
        seq_lengths = np.asarray(seq_lengths, dtype=np.int64)
        if int(seq_lengths.sum()) != int(hidden_flow.shape[0]):
            raise ValueError(
                "seq_lengths must sum to flattened hidden length, "
                f"got sum={int(seq_lengths.sum())} and hidden={tuple(hidden_flow.shape)}"
            )
        seq_start_indices = np.zeros((len(seq_lengths),), dtype=np.int64)
        if len(seq_lengths) > 1:
            seq_start_indices[1:] = np.cumsum(seq_lengths[:-1], dtype=np.int64)
        next_state_flow = torch.empty_like(flow_xt)
        flow_buckets: dict[int, list[tuple[int, int]]] = {}
        for seq_idx, (seq_start, seq_len) in enumerate(zip(seq_start_indices.tolist(), seq_lengths.tolist())):
            seq_len = int(seq_len)
            if seq_len <= 0:
                raise ValueError("decode_aux_from_hidden seq_lengths must all be positive.")
            flow_buckets.setdefault(seq_len, []).append((int(seq_idx), int(seq_start)))
        for seg_len, bucket in flow_buckets.items():
            hidden_bucket = torch.stack(
                [hidden_flow[int(seq_start) : int(seq_start) + int(seg_len)] for _, seq_start in bucket],
                dim=1,
            )
            action_bucket = torch.stack(
                [action_flow[int(seq_start) : int(seq_start) + int(seg_len)] for _, seq_start in bucket],
                dim=1,
            )
            flow_xt_bucket = torch.stack(
                [flow_xt[int(seq_start) : int(seq_start) + int(seg_len)] for _, seq_start in bucket],
                dim=1,
            )
            flow_t_bucket = torch.stack(
                [flow_t[int(seq_start) : int(seq_start) + int(seg_len)] for _, seq_start in bucket],
                dim=1,
            )
            next_state_flow_bucket = self.rlpfn_model._decode_next_state_flow(
                hidden_bucket,
                flow_xt_bucket,
                flow_t_bucket,
                action=action_bucket,
            )
            for bucket_idx, (_, seq_start) in enumerate(bucket):
                seq_start = int(seq_start)
                seq_end = seq_start + int(seg_len)
                next_state_flow[seq_start:seq_end] = next_state_flow_bucket[:, bucket_idx]
        return next_state_flow

    def _decode_flow_from_hidden_flat_packed(
        self,
        hidden_flow: torch.Tensor,
        action_flow: torch.Tensor,
        flow_xt: torch.Tensor,
        flow_t: torch.Tensor,
        *,
        seq_lengths: np.ndarray,
    ) -> torch.Tensor:
        seq_lengths = np.asarray(seq_lengths, dtype=np.int64)
        if int(seq_lengths.sum()) != int(hidden_flow.shape[0]):
            raise ValueError(
                "seq_lengths must sum to flattened hidden length, "
                f"got sum={int(seq_lengths.sum())} and hidden={tuple(hidden_flow.shape)}"
            )
        seq_start_indices = np.zeros((len(seq_lengths),), dtype=np.int64)
        if len(seq_lengths) > 1:
            seq_start_indices[1:] = np.cumsum(seq_lengths[:-1], dtype=np.int64)

        flow_head = getattr(self.rlpfn_model, "next_state_flow_head", None)
        if flow_head is None:
            raise RuntimeError("next_state_flow_head is not initialized")
        action_proj = getattr(self.rlpfn_model, "next_state_flow_action_proj", None)
        if action_proj is None:
            raise RuntimeError("next_state_flow_head action conditioning is not initialized")
        if tuple(hidden_flow.shape[:-1]) != tuple(action_flow.shape[:-1]):
            raise ValueError(
                "next_state_flow_head action_query must match hidden leading dims, "
                f"got {tuple(action_flow.shape)} for hidden {tuple(hidden_flow.shape)}"
            )
        if int(action_flow.shape[-1]) != int(action_proj.in_features):
            raise ValueError(
                f"next_state_flow_head action_query last dim must be {int(action_proj.in_features)}, "
                f"got {tuple(action_flow.shape)}"
            )
        proj_param = next(action_proj.parameters())
        action_cond = action_proj(action_flow.to(device=hidden_flow.device, dtype=proj_param.dtype))
        if action_cond.dtype != hidden_flow.dtype:
            action_cond = action_cond.to(dtype=hidden_flow.dtype)
        conditioned_hidden_flow = hidden_flow + action_cond

        def _resolve_pack_capacity(seq_len: int, total_batch: int) -> int:
            if flow_head is not None and hasattr(flow_head, "resolve_replay_batch_chunk_size"):
                batch_chunk_size = int(flow_head.resolve_replay_batch_chunk_size(seq_len=int(seq_len), total_batch=int(total_batch)))
            else:
                batch_chunk_size = int(total_batch)
            return int(max(1, min(int(total_batch), batch_chunk_size)))

        order = sorted(range(len(seq_lengths)), key=lambda idx: int(seq_lengths[int(idx)]), reverse=True)
        next_state_flow = torch.empty_like(flow_xt)
        cursor = 0
        while cursor < len(order):
            first_seq_idx = int(order[cursor])
            pack_max_len = int(seq_lengths[first_seq_idx])
            remaining = int(len(order) - cursor)
            pack_capacity = _resolve_pack_capacity(seq_len=pack_max_len, total_batch=remaining)
            pack_seq_indices = [int(idx) for idx in order[cursor : cursor + pack_capacity]]
            pack_size = int(len(pack_seq_indices))

            hidden_pack = torch.zeros(
                (pack_max_len, pack_size, int(conditioned_hidden_flow.shape[-1])),
                device=conditioned_hidden_flow.device,
                dtype=conditioned_hidden_flow.dtype,
            )
            flow_xt_pack = torch.zeros(
                (pack_max_len, pack_size, int(flow_xt.shape[-1])),
                device=flow_xt.device,
                dtype=flow_xt.dtype,
            )
            flow_t_pack = torch.zeros(
                (pack_max_len, pack_size, int(flow_t.shape[-1])),
                device=flow_t.device,
                dtype=flow_t.dtype,
            )
            for pack_col, seq_idx in enumerate(pack_seq_indices):
                seg_start = int(seq_start_indices[int(seq_idx)])
                seg_len = int(seq_lengths[int(seq_idx)])
                seg_end = seg_start + seg_len
                hidden_pack[:seg_len, pack_col].copy_(conditioned_hidden_flow[seg_start:seg_end])
                flow_xt_pack[:seg_len, pack_col].copy_(flow_xt[seg_start:seg_end])
                flow_t_pack[:seg_len, pack_col].copy_(flow_t[seg_start:seg_end])

            next_state_flow_pack = self.rlpfn_model._decode_next_state_flow(
                hidden_pack,
                flow_xt_pack,
                flow_t_pack,
                action=None,
            )
            for pack_col, seq_idx in enumerate(pack_seq_indices):
                seg_start = int(seq_start_indices[int(seq_idx)])
                seg_len = int(seq_lengths[int(seq_idx)])
                seg_end = seg_start + seg_len
                next_state_flow[seg_start:seg_end].copy_(next_state_flow_pack[:seg_len, pack_col])
            cursor += pack_size
        return next_state_flow

    def _dummy_states(self, batch_size: int) -> RNNStates:
        zeros = torch.zeros((1, int(batch_size), 1), device=self.device, dtype=torch.float32)
        return RNNStates((zeros, zeros), (zeros, zeros))

    def reset_rollout_cache(self, n_envs: int) -> None:
        self._rollout_kv_cache = None
        self._rollout_cache_batch_size = int(n_envs)

    def _reward_tokens_from_obs(self, obs: torch.Tensor) -> torch.Tensor:
        return obs[..., self.obs_slot_dim]

    def _encode_obs_tokens(self, obs: torch.Tensor) -> torch.Tensor:
        y = self._reward_tokens_from_obs(obs)
        return self.rlpfn_model.encode_train_sequence_tokens(obs, y)

    def _official_rollout_hidden(
        self,
        obs: torch.Tensor,
        episode_starts: torch.Tensor,
        *,
        mutate_cache: bool,
    ) -> torch.Tensor:
        if obs.ndim != 2:
            raise ValueError(f"PPO rollout expects obs with shape (B, F), got {tuple(obs.shape)}")
        if not obs.is_cuda:
            raise RuntimeError("Strict official RWKV PPO rollout requires CUDA observations.")
        batch_size = int(obs.shape[0])
        if self._rollout_cache_batch_size != batch_size:
            self.reset_rollout_cache(batch_size)
        episode_flags = episode_starts.reshape(batch_size).to(device=obs.device, dtype=torch.bool)
        y = self._reward_tokens_from_obs(obs).reshape(1, batch_size)
        token = self.rlpfn_model._encode_train_token(obs.unsqueeze(0), y)
        token = self.rlpfn_model._cast_token_for_rwkv_core(token)

        if self._rollout_kv_cache is None:
            current_cache = self.rlpfn_model.rwkv_core.init_state(
                batch_size,
                device=token.device,
                dtype=token.dtype,
            )
        else:
            current_cache = self._rollout_kv_cache if mutate_cache else _clone_tree(self._rollout_kv_cache)
        current_cache = _zero_batch_rows(current_cache, episode_flags)
        hidden, next_cache = self.rlpfn_model.rwkv_core.forward_step(token[0], current_cache)
        if mutate_cache:
            self._rollout_kv_cache = next_cache
        return hidden

    def _official_sequence_hidden(
        self,
        obs: torch.Tensor,
        episode_starts: torch.Tensor,
        *,
        n_seq: int,
    ) -> torch.Tensor:
        if obs.ndim != 2:
            raise ValueError(f"PPO train obs must have shape (B, F), got {tuple(obs.shape)}")
        if int(obs.shape[0]) % int(n_seq) != 0:
            raise ValueError(
                f"PPO train batch size {int(obs.shape[0])} must divide evenly by n_seq={int(n_seq)}"
            )
        seq_obs = obs.reshape((int(n_seq), -1, self.num_features)).swapaxes(0, 1)
        seq_episode_starts = episode_starts.reshape((int(n_seq), -1)).swapaxes(0, 1)
        segment_buckets: dict[int, list[tuple[int, int, torch.Tensor]]] = {}
        for seq_idx in range(int(n_seq)):
            obs_one = seq_obs[:, seq_idx : seq_idx + 1]
            starts_one = seq_episode_starts[:, seq_idx]
            seg_start = 0
            for t in range(1, int(obs_one.shape[0])):
                if bool(starts_one[t].item()):
                    segment = obs_one[seg_start:t]
                    if int(segment.shape[0]) > 0:
                        seg_len = int(segment.shape[0])
                        segment_buckets.setdefault(seg_len, []).append((seq_idx, seg_start, segment))
                    seg_start = t
            tail = obs_one[seg_start:]
            if int(tail.shape[0]) > 0:
                seg_len = int(tail.shape[0])
                segment_buckets.setdefault(seg_len, []).append((seq_idx, seg_start, tail))
        if len(segment_buckets) == 0:
            raise RuntimeError("Encountered empty PPO sequence while evaluating RWKV policy.")

        total_seq_len = int(seq_obs.shape[0])

        def _forward_bucket_obs(bucket_obs: torch.Tensor, *, seg_len: int) -> torch.Tensor:
            total_batch = int(bucket_obs.shape[1])
            resolve_chunk = getattr(self.rlpfn_model, "resolve_replay_batch_chunk_size", None)
            if callable(resolve_chunk):
                batch_chunk_size = int(resolve_chunk(seq_len=int(seg_len), total_batch=total_batch))
            else:
                batch_chunk_size = total_batch
            batch_chunk_size = int(max(1, min(total_batch, batch_chunk_size)))
            if batch_chunk_size >= total_batch:
                tokens = self._encode_obs_tokens(bucket_obs)
                tokens = self.rlpfn_model._cast_token_for_rwkv_core(tokens)
                return self.rlpfn_model.rwkv_core.forward_tokens_sequence_only(tokens)
            hidden_bucket = None
            for start in range(0, total_batch, batch_chunk_size):
                end = min(total_batch, start + batch_chunk_size)
                tokens_chunk = self._encode_obs_tokens(bucket_obs[:, start:end])
                tokens_chunk = self.rlpfn_model._cast_token_for_rwkv_core(tokens_chunk)
                hidden_chunk = self.rlpfn_model.rwkv_core.forward_tokens_sequence_only(tokens_chunk)
                if hidden_bucket is None:
                    hidden_bucket = torch.empty(
                        (int(seg_len), total_batch, int(hidden_chunk.shape[-1])),
                        device=hidden_chunk.device,
                        dtype=hidden_chunk.dtype,
                    )
                hidden_bucket[:, start:end].copy_(hidden_chunk)
            if hidden_bucket is None:
                raise RuntimeError("RWKV PPO sequence bucket produced no hidden states.")
            return hidden_bucket

        if len(segment_buckets) == 1:
            seg_len, bucket = next(iter(segment_buckets.items()))
            if (
                int(seg_len) == total_seq_len
                and int(len(bucket)) == int(n_seq)
                and all(int(seg_start) == 0 for _, seg_start, _ in bucket)
            ):
                bucket_obs = torch.cat([segment for _, _, segment in bucket], dim=1)
                hidden_bucket = _forward_bucket_obs(bucket_obs, seg_len=int(seg_len))
                return hidden_bucket.swapaxes(0, 1).reshape(-1, hidden_bucket.shape[-1])

        hidden_flat = None
        for seg_len, bucket in segment_buckets.items():
            bucket_obs = torch.cat([segment for _, _, segment in bucket], dim=1)
            hidden_bucket = _forward_bucket_obs(bucket_obs, seg_len=int(seg_len))
            if hidden_flat is None:
                hidden_flat = torch.empty(
                    (int(obs.shape[0]), int(hidden_bucket.shape[-1])),
                    device=hidden_bucket.device,
                    dtype=hidden_bucket.dtype,
                )
            for bucket_idx, (seq_idx, seg_start, _) in enumerate(bucket):
                seg_end = int(seg_start) + int(seg_len)
                flat_start = int(seq_idx) * total_seq_len + int(seg_start)
                flat_end = int(seq_idx) * total_seq_len + seg_end
                hidden_flat[flat_start:flat_end] = hidden_bucket[:, bucket_idx]
        if hidden_flat is None:
            raise RuntimeError("PPO sequence replay produced no hidden states.")
        return hidden_flat

    def _official_sequence_hidden_flat(
        self,
        obs: torch.Tensor,
        *,
        seq_lengths: np.ndarray,
    ) -> torch.Tensor:
        if obs.ndim != 2:
            raise ValueError(f"PPO flat train obs must have shape (B, F), got {tuple(obs.shape)}")
        seq_lengths = np.asarray(seq_lengths, dtype=np.int64)
        n_seq = int(len(seq_lengths))
        if n_seq <= 0:
            raise ValueError("PPO flat train batch requires at least one sequence.")
        if int(seq_lengths.sum()) != int(obs.shape[0]):
            raise ValueError(
                f"PPO flat train batch size {int(obs.shape[0])} must equal sum(seq_lengths)={int(seq_lengths.sum())}"
            )
        return self._official_sequence_hidden_flat_packed(obs, seq_lengths=seq_lengths)

    def _official_sequence_hidden_flat_exact_buckets(
        self,
        obs: torch.Tensor,
        *,
        seq_lengths: np.ndarray,
    ) -> torch.Tensor:
        if obs.ndim != 2:
            raise ValueError(f"PPO flat train obs must have shape (B, F), got {tuple(obs.shape)}")
        seq_lengths = np.asarray(seq_lengths, dtype=np.int64)
        n_seq = int(len(seq_lengths))
        if n_seq <= 0:
            raise ValueError("PPO flat train batch requires at least one sequence.")
        if int(seq_lengths.sum()) != int(obs.shape[0]):
            raise ValueError(
                f"PPO flat train batch size {int(obs.shape[0])} must equal sum(seq_lengths)={int(seq_lengths.sum())}"
            )
        seq_start_indices = np.zeros((n_seq,), dtype=np.int64)
        if n_seq > 1:
            seq_start_indices[1:] = np.cumsum(seq_lengths[:-1], dtype=np.int64)
        segment_buckets: dict[int, list[tuple[int, int]]] = {}
        for seq_idx, (seg_start, seg_len) in enumerate(zip(seq_start_indices.tolist(), seq_lengths.tolist())):
            seg_len = int(seg_len)
            if seg_len <= 0:
                raise RuntimeError("Encountered empty PPO flat sequence while evaluating RWKV policy.")
            segment_buckets.setdefault(seg_len, []).append((int(seq_idx), int(seg_start)))

        def _forward_bucket_obs(bucket_obs: torch.Tensor, *, seg_len: int) -> torch.Tensor:
            total_batch = int(bucket_obs.shape[1])
            resolve_chunk = getattr(self.rlpfn_model, "resolve_replay_batch_chunk_size", None)
            if callable(resolve_chunk):
                batch_chunk_size = int(resolve_chunk(seq_len=int(seg_len), total_batch=total_batch))
            else:
                batch_chunk_size = total_batch
            batch_chunk_size = int(max(1, min(total_batch, batch_chunk_size)))
            if batch_chunk_size >= total_batch:
                tokens = self._encode_obs_tokens(bucket_obs)
                tokens = self.rlpfn_model._cast_token_for_rwkv_core(tokens)
                return self.rlpfn_model.rwkv_core.forward_tokens_sequence_only(tokens)
            hidden_bucket = None
            for start in range(0, total_batch, batch_chunk_size):
                end = min(total_batch, start + batch_chunk_size)
                tokens_chunk = self._encode_obs_tokens(bucket_obs[:, start:end])
                tokens_chunk = self.rlpfn_model._cast_token_for_rwkv_core(tokens_chunk)
                hidden_chunk = self.rlpfn_model.rwkv_core.forward_tokens_sequence_only(tokens_chunk)
                if hidden_bucket is None:
                    hidden_bucket = torch.empty(
                        (int(seg_len), total_batch, int(hidden_chunk.shape[-1])),
                        device=hidden_chunk.device,
                        dtype=hidden_chunk.dtype,
                    )
                hidden_bucket[:, start:end].copy_(hidden_chunk)
            if hidden_bucket is None:
                raise RuntimeError("RWKV PPO flat sequence bucket produced no hidden states.")
            return hidden_bucket

        if len(segment_buckets) == 1:
            seg_len, bucket = next(iter(segment_buckets.items()))
            if int(len(bucket)) == n_seq and all(int(seg_start) == int(seq_idx) * int(seg_len) for seq_idx, seg_start in bucket):
                bucket_obs = torch.stack(
                    [obs[int(seg_start) : int(seg_start) + int(seg_len)] for _, seg_start in bucket],
                    dim=1,
                )
                hidden_bucket = _forward_bucket_obs(bucket_obs, seg_len=int(seg_len))
                return hidden_bucket.swapaxes(0, 1).reshape(-1, hidden_bucket.shape[-1])

        hidden_flat = None
        for seg_len, bucket in segment_buckets.items():
            bucket_obs = torch.stack(
                [obs[int(seg_start) : int(seg_start) + int(seg_len)] for _, seg_start in bucket],
                dim=1,
            )
            hidden_bucket = _forward_bucket_obs(bucket_obs, seg_len=int(seg_len))
            if hidden_flat is None:
                hidden_flat = torch.empty(
                    (int(obs.shape[0]), int(hidden_bucket.shape[-1])),
                    device=hidden_bucket.device,
                    dtype=hidden_bucket.dtype,
                )
            for bucket_idx, (_, seg_start) in enumerate(bucket):
                seg_start = int(seg_start)
                seg_end = seg_start + int(seg_len)
                hidden_flat[seg_start:seg_end] = hidden_bucket[:, bucket_idx]
        if hidden_flat is None:
            raise RuntimeError("PPO flat sequence replay produced no hidden states.")
        return hidden_flat

    def _official_sequence_hidden_flat_packed(
        self,
        obs: torch.Tensor,
        *,
        seq_lengths: np.ndarray,
    ) -> torch.Tensor:
        if obs.ndim != 2:
            raise ValueError(f"PPO flat train obs must have shape (B, F), got {tuple(obs.shape)}")
        seq_lengths = np.asarray(seq_lengths, dtype=np.int64)
        n_seq = int(len(seq_lengths))
        if n_seq <= 0:
            raise ValueError("PPO flat train batch requires at least one sequence.")
        if int(seq_lengths.sum()) != int(obs.shape[0]):
            raise ValueError(
                f"PPO flat train batch size {int(obs.shape[0])} must equal sum(seq_lengths)={int(seq_lengths.sum())}"
            )
        seq_start_indices = np.zeros((n_seq,), dtype=np.int64)
        if n_seq > 1:
            seq_start_indices[1:] = np.cumsum(seq_lengths[:-1], dtype=np.int64)

        def _resolve_pack_capacity(seq_len: int, total_batch: int) -> int:
            resolve_chunk = getattr(self.rlpfn_model, "resolve_replay_batch_chunk_size", None)
            if callable(resolve_chunk):
                batch_chunk_size = int(resolve_chunk(seq_len=int(seq_len), total_batch=int(total_batch)))
            else:
                batch_chunk_size = int(total_batch)
            return int(max(1, min(int(total_batch), batch_chunk_size)))

        def _forward_pack(pack_seq_indices: list[int]) -> torch.Tensor:
            pack_size = int(len(pack_seq_indices))
            pack_lengths = seq_lengths[np.asarray(pack_seq_indices, dtype=np.int64)]
            pack_max_len = int(pack_lengths.max())
            bucket_obs = torch.zeros(
                (pack_max_len, pack_size, int(obs.shape[-1])),
                device=obs.device,
                dtype=obs.dtype,
            )
            for col, seq_idx in enumerate(pack_seq_indices):
                seg_start = int(seq_start_indices[int(seq_idx)])
                seg_len = int(seq_lengths[int(seq_idx)])
                bucket_obs[:seg_len, col].copy_(obs[seg_start : seg_start + seg_len])
            tokens = self._encode_obs_tokens(bucket_obs)
            tokens = self.rlpfn_model._cast_token_for_rwkv_core(tokens)
            return self.rlpfn_model.rwkv_core.forward_tokens_sequence_only(tokens)

        if np.all(seq_lengths == int(seq_lengths[0])):
            seq_len = int(seq_lengths[0])
            bucket_obs = obs.reshape((n_seq, seq_len, self.num_features)).swapaxes(0, 1).contiguous()
            total_batch = int(bucket_obs.shape[1])
            batch_chunk_size = _resolve_pack_capacity(seq_len=seq_len, total_batch=total_batch)
            if batch_chunk_size >= total_batch:
                hidden_bucket = _forward_pack(list(range(n_seq)))
                return hidden_bucket.swapaxes(0, 1).reshape(-1, hidden_bucket.shape[-1])

        order = sorted(range(n_seq), key=lambda idx: int(seq_lengths[int(idx)]), reverse=True)
        hidden_flat = None
        cursor = 0
        while cursor < n_seq:
            first_seq_idx = int(order[cursor])
            pack_max_len = int(seq_lengths[first_seq_idx])
            remaining = int(n_seq - cursor)
            pack_capacity = _resolve_pack_capacity(seq_len=pack_max_len, total_batch=remaining)
            pack_seq_indices = [int(idx) for idx in order[cursor : cursor + pack_capacity]]
            hidden_pack = _forward_pack(pack_seq_indices)
            if hidden_flat is None:
                hidden_flat = torch.empty(
                    (int(obs.shape[0]), int(hidden_pack.shape[-1])),
                    device=hidden_pack.device,
                    dtype=hidden_pack.dtype,
                )
            for pack_col, seq_idx in enumerate(pack_seq_indices):
                seg_start = int(seq_start_indices[int(seq_idx)])
                seg_len = int(seq_lengths[int(seq_idx)])
                hidden_flat[seg_start : seg_start + seg_len].copy_(hidden_pack[:seg_len, pack_col])
            cursor += len(pack_seq_indices)
        if hidden_flat is None:
            raise RuntimeError("PPO flat sequence replay produced no hidden states.")
        return hidden_flat

    def _official_sequence_hidden_and_aux_flat_packed(
        self,
        obs: torch.Tensor,
        *,
        seq_lengths: np.ndarray,
        actions: Optional[torch.Tensor] = None,
        include_normalized_q_logits: bool = False,
        flow_matching_xt: Optional[torch.Tensor] = None,
        flow_matching_t: Optional[torch.Tensor] = None,
    ):
        replay_hidden_and_aux_fn = getattr(
            self.rlpfn_model,
            "replay_hidden_and_aux_sequence_outputs_from_train_tokens",
            None,
        )
        if not callable(replay_hidden_and_aux_fn):
            hidden = self._official_sequence_hidden_flat_packed(obs, seq_lengths=seq_lengths)
            outputs = {"hidden": hidden}
            if include_normalized_q_logits or (flow_matching_xt is not None and flow_matching_t is not None):
                aux_outputs = self.decode_aux_from_hidden(
                    hidden,
                    actions=actions,
                    flow_matching_xt=flow_matching_xt,
                    flow_matching_t=flow_matching_t,
                    seq_lengths=seq_lengths,
                )
                outputs.update(aux_outputs)
            return outputs

        if obs.ndim != 2:
            raise ValueError(f"PPO flat train obs must have shape (B, F), got {tuple(obs.shape)}")
        seq_lengths = np.asarray(seq_lengths, dtype=np.int64)
        n_seq = int(len(seq_lengths))
        if n_seq <= 0:
            raise ValueError("PPO flat train batch requires at least one sequence.")
        if int(seq_lengths.sum()) != int(obs.shape[0]):
            raise ValueError(
                f"PPO flat train batch size {int(obs.shape[0])} must equal sum(seq_lengths)={int(seq_lengths.sum())}"
            )
        need_q = bool(include_normalized_q_logits)
        need_flow = flow_matching_xt is not None or flow_matching_t is not None
        if need_q or need_flow:
            if actions is None:
                raise ValueError("PPO fused hidden+aux replay requires actions when aux outputs are requested.")
            if tuple(actions.shape[:1]) != tuple(obs.shape[:1]):
                raise ValueError(
                    f"PPO fused hidden+aux replay actions must match obs batch, got {tuple(actions.shape)} vs {tuple(obs.shape)}"
                )
        if need_flow:
            if flow_matching_xt is None or flow_matching_t is None:
                raise ValueError("PPO fused hidden+aux replay requires both flow_matching_xt and flow_matching_t.")
            if tuple(flow_matching_xt.shape[:1]) != tuple(obs.shape[:1]):
                raise ValueError(
                    "PPO fused hidden+aux replay flow_matching_xt must match obs batch, "
                    f"got {tuple(flow_matching_xt.shape)} vs {tuple(obs.shape)}"
                )
            if tuple(flow_matching_t.shape[:1]) != tuple(obs.shape[:1]):
                raise ValueError(
                    "PPO fused hidden+aux replay flow_matching_t must match obs batch, "
                    f"got {tuple(flow_matching_t.shape)} vs {tuple(obs.shape)}"
                )

        seq_start_indices = np.zeros((n_seq,), dtype=np.int64)
        if n_seq > 1:
            seq_start_indices[1:] = np.cumsum(seq_lengths[:-1], dtype=np.int64)

        def _resolve_pack_capacity(seq_len: int, total_batch: int) -> int:
            resolve_chunk = getattr(self.rlpfn_model, "resolve_replay_batch_chunk_size", None)
            if callable(resolve_chunk):
                batch_chunk_size = int(resolve_chunk(seq_len=int(seq_len), total_batch=int(total_batch)))
            else:
                batch_chunk_size = int(total_batch)
            return int(max(1, min(int(total_batch), batch_chunk_size)))

        def _forward_pack(pack_seq_indices: list[int]):
            pack_size = int(len(pack_seq_indices))
            pack_lengths = seq_lengths[np.asarray(pack_seq_indices, dtype=np.int64)]
            pack_max_len = int(pack_lengths.max())
            bucket_obs = torch.zeros(
                (pack_max_len, pack_size, int(obs.shape[-1])),
                device=obs.device,
                dtype=obs.dtype,
            )
            action_pack = None
            if need_q or need_flow:
                action_pack = torch.zeros(
                    (pack_max_len, pack_size, int(actions.shape[-1])),
                    device=actions.device,
                    dtype=actions.dtype,
                )
            flow_xt_pack = None
            flow_t_pack = None
            if need_flow:
                flow_xt_pack = torch.zeros(
                    (pack_max_len, pack_size, int(flow_matching_xt.shape[-1])),
                    device=flow_matching_xt.device,
                    dtype=flow_matching_xt.dtype,
                )
                flow_t_pack = torch.zeros(
                    (pack_max_len, pack_size, int(flow_matching_t.shape[-1])),
                    device=flow_matching_t.device,
                    dtype=flow_matching_t.dtype,
                )
            for col, seq_idx in enumerate(pack_seq_indices):
                seg_start = int(seq_start_indices[int(seq_idx)])
                seg_len = int(seq_lengths[int(seq_idx)])
                seg_end = seg_start + seg_len
                bucket_obs[:seg_len, col].copy_(obs[seg_start:seg_end])
                if action_pack is not None:
                    action_pack[:seg_len, col].copy_(actions[seg_start:seg_end])
                if flow_xt_pack is not None:
                    flow_xt_pack[:seg_len, col].copy_(flow_matching_xt[seg_start:seg_end])
                    flow_t_pack[:seg_len, col].copy_(flow_matching_t[seg_start:seg_end])
            train_tokens = self._encode_obs_tokens(bucket_obs)
            return replay_hidden_and_aux_fn(
                train_tokens,
                eval_start=0,
                include_normalized_q_logits=bool(need_q),
                q_action_query=action_pack if need_q else None,
                flow_action_query=action_pack if need_flow else None,
                flow_matching_xt=flow_xt_pack,
                flow_matching_t=flow_t_pack,
            )

        if np.all(seq_lengths == int(seq_lengths[0])):
            seq_len = int(seq_lengths[0])
            total_batch = int(n_seq)
            batch_chunk_size = _resolve_pack_capacity(seq_len=seq_len, total_batch=total_batch)
            if batch_chunk_size >= total_batch:
                outputs = _forward_pack(list(range(n_seq)))
                flat_outputs = {
                    "hidden": outputs["hidden"].swapaxes(0, 1).reshape(-1, int(outputs["hidden"].shape[-1])),
                }
                if "normalized_q_logits" in outputs:
                    flat_outputs["normalized_q_logits"] = outputs["normalized_q_logits"].swapaxes(0, 1).reshape(
                        -1,
                        int(outputs["normalized_q_logits"].shape[-1]),
                    )
                if "next_state_flow" in outputs:
                    flat_outputs["next_state_flow"] = outputs["next_state_flow"].swapaxes(0, 1).reshape(
                        -1,
                        int(outputs["next_state_flow"].shape[-1]),
                    )
                return flat_outputs

        order = sorted(range(n_seq), key=lambda idx: int(seq_lengths[int(idx)]), reverse=True)
        hidden_flat = None
        q_logits_flat = None
        next_state_flow_flat = None
        cursor = 0
        while cursor < n_seq:
            first_seq_idx = int(order[cursor])
            pack_max_len = int(seq_lengths[first_seq_idx])
            remaining = int(n_seq - cursor)
            pack_capacity = _resolve_pack_capacity(seq_len=pack_max_len, total_batch=remaining)
            pack_seq_indices = [int(idx) for idx in order[cursor : cursor + pack_capacity]]
            pack_outputs = _forward_pack(pack_seq_indices)
            hidden_pack = pack_outputs["hidden"]
            if hidden_flat is None:
                hidden_flat = torch.empty(
                    (int(obs.shape[0]), int(hidden_pack.shape[-1])),
                    device=hidden_pack.device,
                    dtype=hidden_pack.dtype,
                )
            if "normalized_q_logits" in pack_outputs and q_logits_flat is None:
                q_logits_pack = pack_outputs["normalized_q_logits"]
                q_logits_flat = torch.empty(
                    (int(obs.shape[0]), int(q_logits_pack.shape[-1])),
                    device=q_logits_pack.device,
                    dtype=q_logits_pack.dtype,
                )
            if "next_state_flow" in pack_outputs and next_state_flow_flat is None:
                flow_pack = pack_outputs["next_state_flow"]
                next_state_flow_flat = torch.empty(
                    (int(obs.shape[0]), int(flow_pack.shape[-1])),
                    device=flow_pack.device,
                    dtype=flow_pack.dtype,
                )
            for pack_col, seq_idx in enumerate(pack_seq_indices):
                seg_start = int(seq_start_indices[int(seq_idx)])
                seg_len = int(seq_lengths[int(seq_idx)])
                seg_end = seg_start + seg_len
                hidden_flat[seg_start:seg_end].copy_(hidden_pack[:seg_len, pack_col])
                if q_logits_flat is not None and "normalized_q_logits" in pack_outputs:
                    q_logits_flat[seg_start:seg_end].copy_(pack_outputs["normalized_q_logits"][:seg_len, pack_col])
                if next_state_flow_flat is not None and "next_state_flow" in pack_outputs:
                    next_state_flow_flat[seg_start:seg_end].copy_(pack_outputs["next_state_flow"][:seg_len, pack_col])
            cursor += len(pack_seq_indices)

        if hidden_flat is None:
            raise RuntimeError("PPO fused hidden+aux replay produced no hidden states.")
        outputs = {"hidden": hidden_flat}
        if q_logits_flat is not None:
            outputs["normalized_q_logits"] = q_logits_flat
        if next_state_flow_flat is not None:
            outputs["next_state_flow"] = next_state_flow_flat
        return outputs

    def _build_obs_from_rollout_step_inputs(
        self,
        obs_t: torch.Tensor,
        action_t: torch.Tensor,
        reward_t: torch.Tensor,
        reward_mask_t: torch.Tensor,
        env_info: dict[str, Any],
    ) -> torch.Tensor:
        def _vector_from_env_info(key: str, *, dtype: torch.dtype) -> torch.Tensor:
            value = env_info[key]
            if torch.is_tensor(value):
                out = value.to(device=obs_t.device, dtype=dtype)
            else:
                out = torch.full((batch_size,), value, device=obs_t.device, dtype=dtype)
            return out

        batch_size = int(obs_t.shape[0])
        obs = torch.zeros((batch_size, self.num_features), device=obs_t.device, dtype=obs_t.dtype)
        obs_slot_dims = _vector_from_env_info("obs_slot_dim", dtype=torch.long)
        action_slot_dims = _vector_from_env_info("action_slot_dim", dtype=torch.long)
        action_dims = _vector_from_env_info("action_dim_per_sample", dtype=torch.long)
        terminal_enabled = _vector_from_env_info("terminal_reset_enabled", dtype=torch.bool)
        reward_idx = obs_slot_dims
        mask_idx = obs_slot_dims + 1
        phase_idx = obs_slot_dims + 2
        terminal_idx = obs_slot_dims + 3
        action_start = obs_slot_dims + 3 + terminal_enabled.to(dtype=torch.long)
        rows = torch.arange(batch_size, device=obs_t.device, dtype=torch.long)

        obs_cap = torch.minimum(
            _vector_from_env_info("obs_dim", dtype=torch.long),
            obs_slot_dims,
        )
        obs_write_cap = int(min(int(obs_t.shape[-1]), self.num_features))
        if obs_write_cap > 0:
            cols = torch.arange(obs_write_cap, device=obs_t.device, dtype=torch.long).unsqueeze(0)
            valid = cols < obs_cap.unsqueeze(1)
            obs[:, :obs_write_cap] = obs_t[:, :obs_write_cap] * valid.to(dtype=obs.dtype)

        valid = reward_idx < self.num_features
        if bool(valid.any().item()):
            obs[rows[valid], reward_idx[valid]] = reward_t.reshape(batch_size)[valid].to(dtype=obs.dtype)
        valid = mask_idx < self.num_features
        if bool(valid.any().item()):
            obs[rows[valid], mask_idx[valid]] = reward_mask_t.reshape(batch_size)[valid].to(dtype=obs.dtype)
        phase_t = env_info.get("phase_t", None)
        if phase_t is not None:
            phase_scalar = phase_t.reshape(batch_size).to(device=obs.device, dtype=obs.dtype)
            valid = phase_idx < self.num_features
            if bool(valid.any().item()):
                obs[rows[valid], phase_idx[valid]] = phase_scalar[valid]
        terminal_t = env_info.get("terminal_t", None)
        if terminal_t is not None:
            terminal_scalar = terminal_t.reshape(batch_size).to(device=obs.device, dtype=obs.dtype)
            valid = terminal_enabled & (terminal_idx < self.num_features)
            if bool(valid.any().item()):
                obs[rows[valid], terminal_idx[valid]] = terminal_scalar[valid]

        action_cap = torch.minimum(action_dims, action_slot_dims)
        action_write_cap = int(min(int(action_t.shape[-1]), self.num_features))
        if action_write_cap > 0:
            pos = torch.arange(action_write_cap, device=obs.device, dtype=torch.long).unsqueeze(0)
            dst_cols = action_start.unsqueeze(1) + pos
            valid = (pos < action_cap.unsqueeze(1)) & (dst_cols < self.num_features)
            if bool(valid.any().item()):
                obs[rows.unsqueeze(1).expand_as(dst_cols)[valid], dst_cols[valid]] = action_t[:, :action_write_cap][valid]
        return obs

    def make_vectorized_rollout_step_fn(self):
        policy = self

        def policy_step_fn(
            obs_t,
            action_t,
            reward_t,
            reward_mask_t,
            cache,
            step_idx,
            env_info,
        ):
            del step_idx
            obs_full = policy._build_obs_from_rollout_step_inputs(
                obs_t,
                action_t,
                reward_t,
                reward_mask_t,
                env_info,
            )
            token_y = policy._reward_tokens_from_obs(obs_full).reshape(1, int(obs_full.shape[0]))
            token = policy.rlpfn_model._encode_train_token(obs_full.unsqueeze(0), token_y)
            token = policy.rlpfn_model._cast_token_for_rwkv_core(token)
            if cache is None:
                cache = policy.rlpfn_model.rwkv_core.init_state(
                    int(obs_full.shape[0]),
                    device=token.device,
                    dtype=token.dtype,
                )
            hidden, next_cache = policy.rlpfn_model.rwkv_core.forward_step(token[0], cache)
            latent_pi = policy.mlp_extractor.forward_actor(hidden)
            latent_vf = policy.mlp_extractor.forward_critic(hidden)
            distribution = policy._get_action_dist_from_latent(latent_pi)
            action_mean = distribution.distribution.mean
            action_std = distribution.distribution.stddev
            return {
                "action_mean": action_mean,
                "action_std": action_std,
                "values": policy.value_net(latent_vf),
            }, next_cache

        policy_step_fn._policy_actor_sample_fn = (
            lambda actor_outputs, noise: actor_outputs["action_mean"] + (noise * actor_outputs["action_std"])
        )
        return policy_step_fn

    def evaluate_rollout_values(
        self,
        obs_steps: torch.Tensor,
        episode_starts_steps: torch.Tensor,
    ) -> torch.Tensor:
        if obs_steps.ndim != 3:
            raise ValueError(f"obs_steps must have shape (T, B, F), got {tuple(obs_steps.shape)}")
        if episode_starts_steps.shape != obs_steps.shape[:2]:
            raise ValueError(
                "episode_starts_steps must match (T, B), "
                f"got {tuple(episode_starts_steps.shape)} vs {tuple(obs_steps.shape[:2])}"
            )
        n_seq = int(obs_steps.shape[1])
        obs_flat = obs_steps.swapaxes(0, 1).reshape(-1, int(obs_steps.shape[-1]))
        starts_flat = episode_starts_steps.swapaxes(0, 1).reshape(-1)
        hidden = self._official_sequence_hidden(obs_flat, starts_flat, n_seq=n_seq)
        latent_vf = self.mlp_extractor.forward_critic(hidden)
        values = self.value_net(latent_vf).reshape(n_seq, int(obs_steps.shape[0])).swapaxes(0, 1)
        return values

    def forward(
        self,
        obs: torch.Tensor,
        lstm_states: RNNStates,
        episode_starts: torch.Tensor,
        deterministic: bool = False,
        action_masks=None,
    ):
        del lstm_states
        if self.training or torch.is_grad_enabled():
            raise RuntimeError("PPO rollout forward must stay on the official no-grad eval path.")
        hidden = self._official_rollout_hidden(obs, episode_starts, mutate_cache=True)
        latent_pi = self.mlp_extractor.forward_actor(hidden)
        latent_vf = self.mlp_extractor.forward_critic(hidden)
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        action_mask_t = self._coerce_action_masks(action_masks, int(obs.shape[0]), device=obs.device)
        actions = distribution.get_actions(deterministic=deterministic)
        actions = self._masked_action_tensor(actions, action_mask_t)
        log_prob, _ = self._masked_diag_gaussian_stats(distribution, actions, action_mask_t)
        actions = actions.reshape((-1, *self.action_space.shape))
        return actions, values, log_prob, self._dummy_states(obs.shape[0])

    def predict_values(self, obs: torch.Tensor, lstm_states, episode_starts: torch.Tensor) -> torch.Tensor:
        del lstm_states
        hidden = self._official_rollout_hidden(obs, episode_starts, mutate_cache=False)
        latent_vf = self.mlp_extractor.forward_critic(hidden)
        return self.value_net(latent_vf)

    def evaluate_actions_with_hidden(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        lstm_states: RNNStates,
        episode_starts: torch.Tensor,
        action_masks=None,
    ):
        n_seq = int(lstm_states.pi[0].shape[1])
        hidden = self._official_sequence_hidden(obs, episode_starts, n_seq=n_seq)
        latent_pi = self.mlp_extractor.forward_actor(hidden)
        latent_vf = self.mlp_extractor.forward_critic(hidden)
        distribution = self._get_action_dist_from_latent(latent_pi)
        action_mask_t = self._coerce_action_masks(action_masks, int(obs.shape[0]), device=obs.device)
        actions = self._masked_action_tensor(actions, action_mask_t)
        log_prob, entropy = self._masked_diag_gaussian_stats(distribution, actions, action_mask_t)
        values = self.value_net(latent_vf)
        return {
            "values": values,
            "log_prob": log_prob,
            "entropy": entropy,
            "hidden": hidden,
            "actions": actions,
            "action_masks": action_mask_t,
        }

    def evaluate_actions_with_hidden_flat(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        *,
        seq_lengths: np.ndarray,
        action_masks=None,
        include_normalized_q_logits: bool = False,
        flow_matching_xt: Optional[torch.Tensor] = None,
        flow_matching_t: Optional[torch.Tensor] = None,
    ):
        fused_aux_requested = bool(include_normalized_q_logits or (flow_matching_xt is not None and flow_matching_t is not None))
        if fused_aux_requested:
            fused_outputs = self._official_sequence_hidden_and_aux_flat_packed(
                obs,
                seq_lengths=seq_lengths,
                actions=actions,
                include_normalized_q_logits=bool(include_normalized_q_logits),
                flow_matching_xt=flow_matching_xt,
                flow_matching_t=flow_matching_t,
            )
            hidden = fused_outputs["hidden"]
        else:
            fused_outputs = None
            hidden = self._official_sequence_hidden_flat(obs, seq_lengths=seq_lengths)
        latent_pi = self.mlp_extractor.forward_actor(hidden)
        latent_vf = self.mlp_extractor.forward_critic(hidden)
        distribution = self._get_action_dist_from_latent(latent_pi)
        action_mask_t = self._coerce_action_masks(action_masks, int(obs.shape[0]), device=obs.device)
        actions = self._masked_action_tensor(actions, action_mask_t)
        log_prob, entropy = self._masked_diag_gaussian_stats(distribution, actions, action_mask_t)
        values = self.value_net(latent_vf)
        outputs = {
            "values": values,
            "log_prob": log_prob,
            "entropy": entropy,
            "hidden": hidden,
            "actions": actions,
            "action_masks": action_mask_t,
        }
        if fused_outputs is not None:
            if "normalized_q_logits" in fused_outputs:
                outputs["normalized_q_logits"] = fused_outputs["normalized_q_logits"]
            if "next_state_flow" in fused_outputs:
                outputs["next_state_flow"] = fused_outputs["next_state_flow"]
        return outputs

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        lstm_states: RNNStates,
        episode_starts: torch.Tensor,
        action_masks=None,
    ):
        outputs = self.evaluate_actions_with_hidden(
            obs,
            actions,
            lstm_states,
            episode_starts,
            action_masks=action_masks,
        )
        return outputs["values"], outputs["log_prob"], outputs["entropy"]


def build_recurrent_ppo(
    *,
    model,
    env_prior: EnvironmentPrior,
    device,
    num_features: int,
    n_envs: int,
    n_steps: int,
    learning_rate,
    batch_size: int,
    n_epochs: int,
    gamma: float,
    gae_lambda: float,
    clip_range,
    clip_range_vf,
    normalize_advantage: bool,
    ent_coef: float,
    vf_coef: float,
    max_grad_norm: float,
    target_kl,
    verbose: int = 0,
):
    boundary = _validate_official_rwkv_ppo_boundary(
        model=model,
        env_prior=env_prior,
        device=device,
        num_features=num_features,
    )
    target_model = boundary["target_model"]
    obs_slot_dim = int(boundary["obs_slot_dim"])
    action_dim = int(boundary["action_dim"])
    resolved_batch_size = resolve_official_ppo_update_batch_size(
        model=target_model,
        n_envs=int(n_envs),
        n_steps=int(n_steps),
        configured_batch_size=batch_size,
    )

    vec_env = EnvironmentPriorPPOBatchVecEnv(
        env_prior=env_prior,
        device=device,
        num_features=num_features,
        n_steps=int(n_steps),
        num_envs=int(n_envs),
        action_dim=action_dim,
        obs_slot_dim=obs_slot_dim,
        action_slot_dim=int(boundary["layout"]["action_slot_dim"]),
        next_state_target_dim=int(boundary["next_state_target_dim"]),
    )
    algo = MaskedRecurrentPPO(
        policy=OfficialRWKVRecurrentPPOPolicy,
        env=vec_env,
        learning_rate=learning_rate,
        n_steps=int(n_steps),
        batch_size=resolved_batch_size,
        n_epochs=int(n_epochs),
        gamma=float(gamma),
        gae_lambda=float(gae_lambda),
        clip_range=clip_range,
        clip_range_vf=clip_range_vf,
        normalize_advantage=bool(normalize_advantage),
        ent_coef=float(ent_coef),
        vf_coef=float(vf_coef),
        max_grad_norm=float(max_grad_norm),
        target_kl=target_kl,
        verbose=int(verbose),
        device=device,
        policy_kwargs={
            "rlpfn_model": target_model,
            "num_features": int(num_features),
            "obs_slot_dim": int(obs_slot_dim),
            "net_arch": [],
        },
    )
    callback = _NullCallback()
    algo._rwkv_env_prior = env_prior
    algo._rwkv_aux_q_weight = float(boundary["q_weight"])
    algo._rwkv_aux_flow_weight = float(boundary["fm_weight"])
    if isinstance(algo.rollout_buffer, MaskedRecurrentRolloutBuffer):
        algo.rollout_buffer._enable_q_target_cache = float(boundary["q_weight"]) > 0.0
    return algo, callback, vec_env
