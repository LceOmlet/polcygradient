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
import hashlib
import json
import math
import os
import re
import sys
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, NamedTuple, Optional, Sequence

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


def _flat_numeric_summary_no_materialize(flat: np.ndarray) -> dict[str, float | int | bool]:
    chunk_elems = 1_000_000
    total = int(flat.size)
    sum_v = 0.0
    sumsq_v = 0.0
    min_v = None
    max_v = None
    finite = True
    nan_count = 0
    posinf_count = 0
    neginf_count = 0
    for start in range(0, total, int(chunk_elems)):
        chunk = flat[start : start + int(chunk_elems)]
        chunk64 = chunk.astype(np.float64, copy=False)
        sum_v += float(np.sum(chunk64, dtype=np.float64))
        sumsq_v += float(np.sum(np.square(chunk64), dtype=np.float64))
        chunk_min = float(np.min(chunk64))
        chunk_max = float(np.max(chunk64))
        min_v = chunk_min if min_v is None else float(np.minimum(min_v, chunk_min))
        max_v = chunk_max if max_v is None else float(np.maximum(max_v, chunk_max))
        if np.issubdtype(flat.dtype, np.floating):
            nan_count += int(np.isnan(chunk).sum())
            posinf_count += int(np.isposinf(chunk).sum())
            neginf_count += int(np.isneginf(chunk).sum())
        if not bool(np.isfinite(chunk64).all()):
            finite = False
    mean_v = float(sum_v / max(1, total))
    var_v = float((sumsq_v / max(1, total)) - mean_v * mean_v)
    if math.isfinite(var_v) and var_v < 0.0 and abs(var_v) < 1e-9:
        var_v = 0.0
    std_v = float(math.sqrt(var_v)) if var_v >= 0.0 else float("nan")
    return {
        "min": float(min_v if min_v is not None else 0.0),
        "max": float(max_v if max_v is not None else 0.0),
        "mean": mean_v,
        "std": std_v,
        "finite": bool(finite),
        "nan_count": int(nan_count),
        "posinf_count": int(posinf_count),
        "neginf_count": int(neginf_count),
    }


def _array_digest_payload(array: Any, *, sample_size: int = 8) -> dict[str, Any]:
    arr = np.ascontiguousarray(np.asarray(array))
    flat = arr.reshape(-1)
    digest = hashlib.sha256()
    digest.update(memoryview(arr.view(np.uint8).reshape(-1)))
    payload: dict[str, Any] = {
        "shape": [int(v) for v in arr.shape],
        "dtype": str(arr.dtype),
        "sha256": digest.hexdigest(),
        "size": int(flat.size),
    }
    if flat.size <= 0:
        return payload
    sample = flat[: int(sample_size)]
    if np.issubdtype(arr.dtype, np.bool_):
        payload["sample"] = sample.astype(np.int64, copy=False).tolist()
        payload["true_count"] = int(flat.astype(np.int64, copy=False).sum())
        return payload
    if np.issubdtype(arr.dtype, np.integer):
        payload["sample"] = sample.astype(np.int64, copy=False).tolist()
        payload["min"] = int(flat.min())
        payload["max"] = int(flat.max())
        payload["sum"] = int(flat.astype(np.int64, copy=False).sum())
        return payload
    sample_float = sample.astype(np.float64, copy=False)
    numeric = _flat_numeric_summary_no_materialize(flat)
    payload["sample"] = sample_float.tolist()
    payload["min"] = float(numeric["min"])
    payload["max"] = float(numeric["max"])
    payload["mean"] = float(numeric["mean"])
    payload["std"] = float(numeric["std"])
    if np.issubdtype(arr.dtype, np.floating):
        payload["nan_count"] = int(numeric["nan_count"])
        payload["posinf_count"] = int(numeric["posinf_count"])
        payload["neginf_count"] = int(numeric["neginf_count"])
    return payload


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
from stable_baselines3.common.running_mean_std import RunningMeanStd  # noqa: E402
from stable_baselines3.common.type_aliases import Schedule  # noqa: E402
from stable_baselines3.common.utils import explained_variance, obs_as_tensor  # noqa: E402
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvIndices, VecEnvObs, VecEnvStepReturn  # noqa: E402

from ticl.models.tabpfn_bar_distribution import make_standardized_full_support_bar_distribution  # noqa: E402
from ticl.models.tabpfn_regressor_vendor import TabPFNOfficialRegressorHarness  # noqa: E402
from ticl.priors.environment_prior import EnvironmentPrior  # noqa: E402
from ticl.rlpfn_maintained_path import resolve_rlpfn_token_layout  # noqa: E402


STRICT_OFFICIAL_RWKV_PPO_PREFIX = "Strict official RWKV PPO requires"
PPO_VALUE_TARGET_NORM_EPS = 1e-6
CROSS_ROLLOUT_RUNNING_EMA_RMS_SPACE = "cross_rollout_running_ema_rms"
CROSS_ROLLOUT_RUNNING_EMA_RMS_ALPHA = 0.01
PPO_VALUE_BARDIST_VALUE_RANGE = 5.0


def build_rlpfn_obs_token_batch_from_components(
    *,
    obs_t: torch.Tensor,
    action_t: torch.Tensor,
    reward_t: torch.Tensor,
    reward_mask_t: torch.Tensor,
    env_info: dict[str, Any],
    num_features: int,
) -> torch.Tensor:
    """Build RLPFN PPO observation tokens from explicit rollout components."""

    def _vector_from_env_info(key: str, *, dtype: torch.dtype) -> torch.Tensor:
        value = env_info[key]
        if torch.is_tensor(value):
            out = value.to(device=obs_t.device, dtype=dtype)
        else:
            out = torch.full((batch_size,), value, device=obs_t.device, dtype=dtype)
        if out.ndim == 0:
            out = out.reshape(1).expand(batch_size)
        if int(out.shape[0]) != int(batch_size):
            raise ValueError(
                f"env_info[{key!r}] must be scalar or batch-sized ({batch_size}), got shape={tuple(out.shape)}"
            )
        return out.reshape(batch_size)

    if obs_t.ndim != 2:
        raise ValueError(f"obs_t must be 2D [batch, obs_dim], got shape={tuple(obs_t.shape)}")
    if action_t.ndim != 2:
        raise ValueError(f"action_t must be 2D [batch, action_dim], got shape={tuple(action_t.shape)}")
    batch_size = int(obs_t.shape[0])
    if int(action_t.shape[0]) != batch_size:
        raise ValueError(
            f"action_t batch size must match obs_t ({batch_size}), got shape={tuple(action_t.shape)}"
        )
    reward_t = reward_t.reshape(batch_size)
    reward_mask_t = reward_mask_t.reshape(batch_size)

    obs = torch.zeros((batch_size, int(num_features)), device=obs_t.device, dtype=obs_t.dtype)
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
    obs_write_cap = int(min(int(obs_t.shape[-1]), int(num_features)))
    if obs_write_cap > 0:
        cols = torch.arange(obs_write_cap, device=obs_t.device, dtype=torch.long).unsqueeze(0)
        valid = cols < obs_cap.unsqueeze(1)
        obs[:, :obs_write_cap] = obs_t[:, :obs_write_cap] * valid.to(dtype=obs.dtype)

    valid = reward_idx < int(num_features)
    if bool(valid.any().item()):
        obs[rows[valid], reward_idx[valid]] = reward_t[valid].to(dtype=obs.dtype)
    valid = mask_idx < int(num_features)
    if bool(valid.any().item()):
        obs[rows[valid], mask_idx[valid]] = reward_mask_t[valid].to(dtype=obs.dtype)
    phase_t = env_info.get("phase_t", None)
    if phase_t is not None:
        if torch.is_tensor(phase_t):
            phase_scalar = phase_t.to(device=obs.device, dtype=obs.dtype)
        else:
            phase_scalar = torch.as_tensor(phase_t, device=obs.device, dtype=obs.dtype)
        if phase_scalar.ndim == 0:
            phase_scalar = phase_scalar.reshape(1).expand(batch_size)
        phase_scalar = phase_scalar.reshape(batch_size)
        valid = phase_idx < int(num_features)
        if bool(valid.any().item()):
            obs[rows[valid], phase_idx[valid]] = phase_scalar[valid]
    terminal_t = env_info.get("terminal_t", None)
    if terminal_t is not None:
        if torch.is_tensor(terminal_t):
            terminal_scalar = terminal_t.to(device=obs.device, dtype=obs.dtype)
        else:
            terminal_scalar = torch.as_tensor(terminal_t, device=obs.device, dtype=obs.dtype)
        if terminal_scalar.ndim == 0:
            terminal_scalar = terminal_scalar.reshape(1).expand(batch_size)
        terminal_scalar = terminal_scalar.reshape(batch_size)
        valid = terminal_enabled & (terminal_idx < int(num_features))
        if bool(valid.any().item()):
            obs[rows[valid], terminal_idx[valid]] = terminal_scalar[valid]

    action_cap = torch.minimum(action_dims, action_slot_dims)
    action_write_cap = int(min(int(action_t.shape[-1]), int(num_features)))
    if action_write_cap > 0:
        pos = torch.arange(action_write_cap, device=obs.device, dtype=torch.long).unsqueeze(0)
        dst_cols = action_start.unsqueeze(1) + pos
        valid = (pos < action_cap.unsqueeze(1)) & (dst_cols < int(num_features))
        if bool(valid.any().item()):
            obs[rows.unsqueeze(1).expand_as(dst_cols)[valid], dst_cols[valid]] = action_t[:, :action_write_cap][valid]
    return obs


def _make_ppo_value_bardist():
    # PPO critic targets live in the same standardized value space as the
    # dedicated normalized-Q head after rollout-level return normalization.
    return make_standardized_full_support_bar_distribution(
        value_range=PPO_VALUE_BARDIST_VALUE_RANGE,
    )


class _ValuePathBatchNormAdapter(nn.Module):
    """Thin critic-only geometry adapter for value-path hidden states."""

    def __init__(
        self,
        num_features: int,
        *,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        use_batch_stats_in_eval: bool = False,
    ) -> None:
        super().__init__()
        self.use_batch_stats_in_eval = bool(use_batch_stats_in_eval)
        self.norm = nn.BatchNorm1d(
            int(num_features),
            eps=float(eps),
            momentum=float(momentum),
            affine=bool(affine),
            track_running_stats=bool(track_running_stats),
        )

    def forward(self, latent_vf: torch.Tensor) -> torch.Tensor:
        if latent_vf.ndim != 2:
            raise ValueError(
                "_ValuePathBatchNormAdapter expects a rank-2 latent tensor, "
                f"got shape {tuple(latent_vf.shape)}"
            )
        use_batch_stats = int(latent_vf.shape[0]) > 1 and (
            bool(self.training) or bool(self.use_batch_stats_in_eval)
        )
        running_mean = self.norm.running_mean
        running_var = self.norm.running_var
        if use_batch_stats and not bool(self.training):
            running_mean = None if running_mean is None else running_mean.detach().clone()
            running_var = None if running_var is None else running_var.detach().clone()
        return F.batch_norm(
            latent_vf,
            running_mean,
            running_var,
            self.norm.weight,
            self.norm.bias,
            use_batch_stats,
            self.norm.momentum,
            self.norm.eps,
        )


class _ValuePathFrozenZScoreAdapter(nn.Module):
    """Frozen per-feature z-score adapter fitted from rollout value-path latents."""

    def __init__(
        self,
        num_features: int,
        *,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.num_features = int(num_features)
        self.eps = float(eps)
        self.register_buffer("mean", torch.zeros((self.num_features,), dtype=torch.float32))
        self.register_buffer("std", torch.ones((self.num_features,), dtype=torch.float32))
        self.register_buffer("_is_fitted", torch.zeros((1,), dtype=torch.bool))

    def reset_statistics(self) -> None:
        with torch.no_grad():
            self.mean.zero_()
            self.std.fill_(1.0)
            self._is_fitted.zero_()

    def fit_from_latent(
        self,
        latent_vf: torch.Tensor,
        *,
        objective_mask: Optional[torch.Tensor] = None,
    ) -> None:
        if latent_vf.ndim != 2 or int(latent_vf.shape[-1]) != int(self.num_features):
            raise ValueError(
                "_ValuePathFrozenZScoreAdapter.fit_from_latent expects latent shape "
                f"(N, {int(self.num_features)}), got {tuple(latent_vf.shape)}"
            )
        with torch.no_grad():
            stats_latent = latent_vf.detach().to(device=self.mean.device, dtype=torch.float32)
            if objective_mask is not None:
                mask = objective_mask.reshape(-1).to(device=stats_latent.device, dtype=torch.bool)
                if tuple(mask.shape[:1]) != tuple(stats_latent.shape[:1]):
                    raise ValueError(
                        "objective_mask must match latent leading dim, "
                        f"got {tuple(mask.shape)} for latent {tuple(stats_latent.shape)}"
                    )
                if bool(mask.any().item()):
                    stats_latent = stats_latent[mask]
            if int(stats_latent.shape[0]) <= 0:
                self.reset_statistics()
                return
            mean = stats_latent.mean(dim=0)
            std = stats_latent.std(dim=0, unbiased=False).clamp_min(float(self.eps))
            self.mean.copy_(mean)
            self.std.copy_(std)
            self._is_fitted.fill_(True)

    def forward(self, latent_vf: torch.Tensor) -> torch.Tensor:
        if latent_vf.ndim != 2:
            raise ValueError(
                "_ValuePathFrozenZScoreAdapter expects a rank-2 latent tensor, "
                f"got shape {tuple(latent_vf.shape)}"
            )
        if not bool(self._is_fitted.item()):
            return latent_vf
        mean = self.mean.to(device=latent_vf.device, dtype=latent_vf.dtype)
        std = self.std.to(device=latent_vf.device, dtype=latent_vf.dtype).clamp_min(float(self.eps))
        return (latent_vf - mean) / std


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


def _normalize_optional_seed_spec(
    seed_spec: Optional[int | Sequence[int]],
    *,
    n_envs: int,
    name: str,
) -> Optional[list[int]]:
    if seed_spec is None:
        return None
    if isinstance(seed_spec, (int, np.integer)):
        values = [int(seed_spec)]
    else:
        values = [int(v) for v in seed_spec]
    if len(values) == 1 and int(n_envs) > 1:
        values = values * int(n_envs)
    if len(values) != int(n_envs):
        raise ValueError(
            f"{STRICT_OFFICIAL_RWKV_PPO_PREFIX} {name} to have length 1 or n_envs={int(n_envs)}, "
            f"got length={len(values)}."
        )
    return values


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


def _masked_mean_std(
    values: torch.Tensor,
    mask: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    mask_f = mask.to(device=values.device, dtype=values.dtype)
    count = mask_f.sum().clamp_min(1.0)
    mean = (values * mask_f).sum() / count
    centered = (values - mean) * mask_f
    std = torch.sqrt((centered.square().sum() / count).clamp_min(float(eps)))
    return mean, std


def _normalize_advantages_with_mask(
    advantages: torch.Tensor,
    mask: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    if advantages.numel() == 0:
        return advantages
    if not bool(mask.any()):
        return torch.zeros_like(advantages)
    # Match sb3-contrib RecurrentPPO exactly:
    # advantages = (advantages - advantages[mask].mean()) / (advantages[mask].std() + 1e-8)
    # torch.std() defaults to the unbiased/sample estimator, unlike _masked_mean_std().
    valid_advantages = advantages[mask.to(device=advantages.device, dtype=torch.bool)]
    return (advantages - valid_advantages.mean()) / (valid_advantages.std() + float(eps))


def _resolve_actor_advantages(
    rollout_data: MaskedRecurrentRolloutBufferSamples | MaskedRecurrentFlatBatchSamples,
) -> torch.Tensor:
    actor_advantages = getattr(rollout_data, "actor_advantages", None)
    if actor_advantages is None:
        return rollout_data.advantages
    return actor_advantages


def _resolve_effective_actor_gae_space(
    actor_gae_space: str,
    *,
    normalize_advantage: bool,
    actor_baseline_mode: str,
) -> str:
    requested = str(actor_gae_space).strip().lower()
    if requested == "normalized" and not bool(normalize_advantage):
        baseline_mode = str(actor_baseline_mode).strip().lower()
        if baseline_mode == "learned":
            # With unnormalized PPO advantages, keeping the learned actor update
            # in rollout-normalized value space shrinks policy updates relative
            # to the raw-space zero baseline. Resolve to raw to keep the actor
            # objective in the same effective reward space.
            return "raw"
    return requested


def _resolve_ppo_space_contract(
    *,
    space_contract: str,
    value_target_space: Optional[str],
    actor_gae_space: Optional[str],
    normalize_advantage: bool,
    actor_baseline_mode: str,
    allow_mixed_space_contract: bool,
) -> dict:
    contract = str(space_contract).strip().lower()
    if contract not in {"raw", "normalized", "reward_normalized", CROSS_ROLLOUT_RUNNING_EMA_RMS_SPACE}:
        raise ValueError(
            f"Unsupported PPO space_contract: {space_contract}. "
            "Expected 'raw', 'normalized', 'reward_normalized', or "
            f"'{CROSS_ROLLOUT_RUNNING_EMA_RMS_SPACE}'."
        )

    requested_value_target_space = contract if value_target_space is None else str(value_target_space).strip().lower()
    requested_actor_gae_space = contract if actor_gae_space is None else str(actor_gae_space).strip().lower()

    if requested_value_target_space not in {"raw", "normalized", "reward_normalized", CROSS_ROLLOUT_RUNNING_EMA_RMS_SPACE}:
        raise ValueError(
            f"Unsupported PPO value_target_space: {requested_value_target_space}. "
            "Expected 'raw', 'normalized', 'reward_normalized', or "
            f"'{CROSS_ROLLOUT_RUNNING_EMA_RMS_SPACE}'."
        )
    if requested_actor_gae_space not in {
        "normalized",
        "raw",
        "normalized_sep_synced",
        "reward_normalized",
        CROSS_ROLLOUT_RUNNING_EMA_RMS_SPACE,
    }:
        raise ValueError(
            f"Unsupported PPO actor_gae_space: {requested_actor_gae_space}. "
            "Expected 'normalized', 'normalized_sep_synced', 'reward_normalized', "
            f"'{CROSS_ROLLOUT_RUNNING_EMA_RMS_SPACE}', or 'raw'."
        )

    if not bool(allow_mixed_space_contract):
        if requested_value_target_space != contract or requested_actor_gae_space != contract:
            raise ValueError(
                "Mixed PPO reward/value space contracts are disabled. "
                f"space_contract={contract!r} requires "
                f"value_target_space={contract!r} and actor_gae_space={contract!r}."
            )
        effective_actor_gae_space = contract
        alignment = "aligned"
    else:
        effective_actor_gae_space = _resolve_effective_actor_gae_space(
            requested_actor_gae_space,
            normalize_advantage=bool(normalize_advantage),
            actor_baseline_mode=actor_baseline_mode,
        )
        alignment = "aligned" if requested_value_target_space == effective_actor_gae_space else "mixed"

    return {
        "space_contract": contract,
        "value_target_space_requested": requested_value_target_space,
        "value_target_space_effective": requested_value_target_space,
        "actor_gae_space_requested": requested_actor_gae_space,
        "actor_gae_space_effective": effective_actor_gae_space,
        "alignment": alignment,
    }


def _explained_variance_with_mask(
    predictions: np.ndarray | torch.Tensor,
    targets: np.ndarray | torch.Tensor,
    *,
    mask: Optional[np.ndarray | torch.Tensor] = None,
) -> float:
    predictions_np = np.asarray(predictions).reshape(-1)
    targets_np = np.asarray(targets).reshape(-1)
    if mask is None:
        return explained_variance(predictions_np, targets_np)
    mask_np = np.asarray(mask).reshape(-1) > 1e-8
    if not bool(np.any(mask_np)):
        return float("nan")
    return explained_variance(predictions_np[mask_np], targets_np[mask_np])


def _masked_flat_np(
    values: np.ndarray | torch.Tensor,
    mask: Optional[np.ndarray | torch.Tensor],
) -> np.ndarray:
    values_np = np.asarray(values, dtype=np.float32).reshape(-1)
    if mask is None:
        return values_np
    mask_np = np.asarray(mask).reshape(-1) > 1e-8
    if not bool(np.any(mask_np)):
        return values_np[:0]
    return values_np[mask_np]


def _safe_float_stat(value: float | np.floating) -> float:
    value_f = float(value)
    if not math.isfinite(value_f):
        return float("nan")
    return value_f


def _masked_array_summary(
    values: np.ndarray | torch.Tensor,
    *,
    mask: Optional[np.ndarray | torch.Tensor],
    prefix: str,
) -> dict[str, float]:
    masked = _masked_flat_np(values, mask)
    if int(masked.size) <= 0:
        return {
            f"{prefix}_mean": float("nan"),
            f"{prefix}_std": float("nan"),
            f"{prefix}_absmax": float("nan"),
        }
    return {
        f"{prefix}_mean": _safe_float_stat(np.mean(masked, dtype=np.float64)),
        f"{prefix}_std": _safe_float_stat(np.std(masked, dtype=np.float64)),
        f"{prefix}_absmax": _safe_float_stat(np.max(np.abs(masked))),
    }


def _array_quantile_summary(
    values: np.ndarray | torch.Tensor,
    *,
    prefix: str,
) -> dict[str, float]:
    values_np = np.asarray(values, dtype=np.float32).reshape(-1)
    values_np = values_np[np.isfinite(values_np)]
    if int(values_np.size) <= 0:
        return {
            f"{prefix}_min": float("nan"),
            f"{prefix}_p01": float("nan"),
            f"{prefix}_p10": float("nan"),
            f"{prefix}_p50": float("nan"),
            f"{prefix}_p90": float("nan"),
            f"{prefix}_p99": float("nan"),
            f"{prefix}_max": float("nan"),
        }
    quantiles = np.quantile(values_np, [0.01, 0.10, 0.50, 0.90, 0.99])
    return {
        f"{prefix}_min": _safe_float_stat(np.min(values_np)),
        f"{prefix}_p01": _safe_float_stat(quantiles[0]),
        f"{prefix}_p10": _safe_float_stat(quantiles[1]),
        f"{prefix}_p50": _safe_float_stat(quantiles[2]),
        f"{prefix}_p90": _safe_float_stat(quantiles[3]),
        f"{prefix}_p99": _safe_float_stat(quantiles[4]),
        f"{prefix}_max": _safe_float_stat(np.max(values_np)),
    }


def _sampled_array_summary(
    values: np.ndarray | torch.Tensor,
    *,
    prefix: str,
    sample_size: int = 65536,
) -> dict[str, float]:
    values_np = np.asarray(values, dtype=np.float32).reshape(-1)
    total = int(values_np.size)
    if total <= 0:
        return {
            f"{prefix}_sample_mean": float("nan"),
            f"{prefix}_sample_std": float("nan"),
            f"{prefix}_sample_absmax": float("nan"),
            f"{prefix}_sample_finite_fraction": float("nan"),
        }
    take = int(min(total, max(1, int(sample_size))))
    if take == total:
        sample = values_np
    else:
        # A fixed stride can alias with row-major token width at large
        # n_steps*n_envs*features, repeatedly sampling the same feature column.
        # Linspace keeps the diagnostic deterministic without that column bias.
        sample_indices = np.linspace(0, total - 1, num=take, dtype=np.int64)
        sample = values_np[sample_indices]
    finite = np.isfinite(sample)
    if not bool(np.any(finite)):
        return {
            f"{prefix}_sample_mean": float("nan"),
            f"{prefix}_sample_std": float("nan"),
            f"{prefix}_sample_absmax": float("nan"),
            f"{prefix}_sample_finite_fraction": 0.0,
        }
    finite_sample = sample[finite]
    return {
        f"{prefix}_sample_mean": _safe_float_stat(np.mean(finite_sample, dtype=np.float64)),
        f"{prefix}_sample_std": _safe_float_stat(np.std(finite_sample, dtype=np.float64)),
        f"{prefix}_sample_absmax": _safe_float_stat(np.max(np.abs(finite_sample))),
        f"{prefix}_sample_finite_fraction": _safe_float_stat(float(finite.mean())),
    }


def _masked_corrcoef_np(
    x: np.ndarray | torch.Tensor,
    y: np.ndarray | torch.Tensor,
    mask: Optional[np.ndarray | torch.Tensor],
) -> float:
    x_m = _masked_flat_np(x, mask)
    y_m = _masked_flat_np(y, mask)
    if int(x_m.size) <= 1 or int(y_m.size) <= 1:
        return 0.0
    if int(x_m.size) != int(y_m.size):
        raise ValueError(f"corr inputs must have the same masked size, got {int(x_m.size)} vs {int(y_m.size)}")
    x_m = x_m - float(np.mean(x_m, dtype=np.float64))
    y_m = y_m - float(np.mean(y_m, dtype=np.float64))
    denom = float(np.linalg.norm(x_m) * np.linalg.norm(y_m))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(x_m, y_m) / denom)


def _discounted_returns_from_rewards(
    rewards: np.ndarray,
    *,
    episode_starts: np.ndarray,
    dones: np.ndarray,
    gamma: float,
) -> np.ndarray:
    rewards = np.asarray(rewards, dtype=np.float32)
    episode_starts = np.asarray(episode_starts, dtype=np.float32)
    dones = np.asarray(dones, dtype=bool).reshape((-1,))
    if rewards.ndim != 2:
        raise ValueError(f"rewards must have shape (T, B), got {tuple(rewards.shape)}")
    if episode_starts.shape != rewards.shape:
        raise ValueError(
            "episode_starts must match rewards shape, "
            f"got {tuple(episode_starts.shape)} vs {tuple(rewards.shape)}"
        )
    if int(dones.shape[0]) != int(rewards.shape[1]):
        raise ValueError(f"dones must have shape (B,), got {tuple(dones.shape)} vs B={int(rewards.shape[1])}")
    gamma = float(gamma)
    returns = np.zeros_like(rewards, dtype=np.float32)
    next_return = np.zeros((rewards.shape[1],), dtype=np.float32)
    for step in reversed(range(int(rewards.shape[0]))):
        if step == int(rewards.shape[0]) - 1:
            next_non_terminal = 1.0 - dones.astype(np.float32, copy=False)
        else:
            next_non_terminal = 1.0 - episode_starts[step + 1].astype(np.float32, copy=False)
        next_return = rewards[step] + (gamma * next_non_terminal * next_return)
        returns[step] = next_return
    return returns


def _rollout_return_norm_stats_from_raw_returns(
    raw_returns: np.ndarray,
    *,
    eps: float = PPO_VALUE_TARGET_NORM_EPS,
) -> tuple[np.ndarray, np.ndarray]:
    raw_returns = np.asarray(raw_returns, dtype=np.float32)
    if raw_returns.ndim != 2:
        raise ValueError(f"raw_returns must have shape (T, B), got {tuple(raw_returns.shape)}")
    means = raw_returns.mean(axis=0, dtype=np.float64).astype(np.float32, copy=False)
    stds = raw_returns.std(axis=0, dtype=np.float64).astype(np.float32, copy=False)
    stds = np.maximum(stds, float(eps))
    return means, stds


def _cross_rollout_running_ema_rms_stats_from_raw_returns(
    raw_returns: np.ndarray,
    *,
    previous_mean_square: np.ndarray | None,
    alpha: float = CROSS_ROLLOUT_RUNNING_EMA_RMS_ALPHA,
    eps: float = PPO_VALUE_TARGET_NORM_EPS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return zero means and cross-rollout EMA RMS scales for raw returns.

    This value space intentionally does not subtract a rollout-local mean:
    preserving absolute deterioration across rollouts is the point of the
    diagnostic contract.
    """
    raw_returns = np.asarray(raw_returns, dtype=np.float32)
    if raw_returns.ndim != 2:
        raise ValueError(f"raw_returns must have shape (T, B), got {tuple(raw_returns.shape)}")
    mean_square = np.mean(np.square(raw_returns.astype(np.float32, copy=False)), axis=0, dtype=np.float64).astype(
        np.float32,
        copy=False,
    )
    mean_square = np.maximum(mean_square, float(eps) * float(eps))
    alpha_f = float(alpha)
    if not (0.0 < alpha_f <= 1.0):
        raise ValueError(f"cross-rollout running EMA RMS alpha must be in (0, 1], got {alpha_f}")
    if previous_mean_square is None:
        running_mean_square = mean_square
    else:
        prev = np.asarray(previous_mean_square, dtype=np.float32)
        if prev.shape != mean_square.shape:
            running_mean_square = mean_square
        else:
            running_mean_square = ((1.0 - alpha_f) * prev) + (alpha_f * mean_square)
    running_mean_square = np.maximum(running_mean_square.astype(np.float32, copy=False), float(eps) * float(eps))
    means = np.zeros_like(running_mean_square, dtype=np.float32)
    rms = np.sqrt(running_mean_square).astype(np.float32, copy=False)
    return means, rms, running_mean_square


def _sb3_vecnormalize_reward_step(
    rewards: np.ndarray,
    *,
    running_returns: np.ndarray,
    ret_rms: RunningMeanStd,
    dones: np.ndarray,
    gamma: float,
    epsilon: float,
    clip_reward: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply SB3 VecNormalize reward scaling to one vectorized reward step."""
    rewards_np = np.asarray(rewards, dtype=np.float32)
    running_returns_np = np.asarray(running_returns, dtype=np.float32)
    if running_returns_np.shape != rewards_np.shape:
        running_returns_np = np.zeros_like(rewards_np, dtype=np.float32)
    dones_np = np.asarray(dones, dtype=bool)
    running_returns_np = (running_returns_np * float(gamma)) + rewards_np
    ret_rms.update(running_returns_np)
    scaled = rewards_np / np.sqrt(ret_rms.var + float(epsilon))
    scaled = np.clip(scaled, -float(clip_reward), float(clip_reward)).astype(np.float32, copy=False)
    running_returns_np[dones_np] = 0.0
    return scaled, running_returns_np.astype(np.float32, copy=False)


def _rollout_return_norm_stats_from_raw_returns_with_mask(
    raw_returns: np.ndarray,
    *,
    mask: np.ndarray,
    eps: float = PPO_VALUE_TARGET_NORM_EPS,
    fallback_means: np.ndarray | None = None,
    fallback_stds: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    raw_returns = np.asarray(raw_returns, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.float32) > 1e-8
    if raw_returns.ndim != 2:
        raise ValueError(f"raw_returns must have shape (T, B), got {tuple(raw_returns.shape)}")
    if mask.shape != raw_returns.shape:
        raise ValueError(
            f"mask must match raw_returns shape, got {tuple(mask.shape)} vs {tuple(raw_returns.shape)}"
        )
    del mask, fallback_means, fallback_stds
    # Value-space statistics must be derived from the whole rollout rather than
    # an objective-only subset to avoid path-dependent leakage.
    return _rollout_return_norm_stats_from_raw_returns(raw_returns, eps=eps)


def _reward_norm_stats_from_rewards_with_mask(
    rewards: np.ndarray,
    *,
    mask: np.ndarray,
    eps: float = PPO_VALUE_TARGET_NORM_EPS,
) -> tuple[np.ndarray, np.ndarray]:
    rewards = np.asarray(rewards, dtype=np.float32)
    if rewards.ndim != 2:
        raise ValueError(f"rewards must have shape (T, B), got {tuple(rewards.shape)}")
    mask = np.asarray(mask, dtype=np.float32)
    if mask.shape != rewards.shape:
        raise ValueError(f"mask must match rewards shape, got {tuple(mask.shape)} vs {tuple(rewards.shape)}")
    # Reward/value space statistics must be computed from the whole rollout to avoid
    # path-dependent leakage. The mask is retained only for API compatibility.
    means = rewards.mean(axis=0, dtype=np.float64).astype(np.float32, copy=False)
    stds = rewards.std(axis=0, dtype=np.float64).astype(np.float32, copy=False)
    stds = np.maximum(stds, float(eps))
    return means, stds


def _episode_starts_synced_to_objective_mask(
    episode_starts: np.ndarray,
    objective_masks: np.ndarray,
) -> np.ndarray:
    episode_starts = np.asarray(episode_starts, dtype=np.float32)
    objective_masks = np.asarray(objective_masks, dtype=np.float32)
    if episode_starts.shape != objective_masks.shape:
        raise ValueError(
            f"episode_starts must match objective_masks shape, got {tuple(episode_starts.shape)} vs {tuple(objective_masks.shape)}"
        )
    adjusted = np.array(episode_starts, copy=True, dtype=np.float32)
    objective_mask_bool = objective_masks > 1e-8
    for env_idx in range(int(adjusted.shape[1])):
        valid_steps = np.flatnonzero(objective_mask_bool[:, env_idx])
        if int(valid_steps.size) == 0:
            continue
        adjusted[int(valid_steps[0]), env_idx] = 1.0
    return adjusted


def _compute_suffix_scores_from_objective_mask(
    rewards: np.ndarray,
    objective_masks: np.ndarray,
) -> np.ndarray:
    rewards = np.asarray(rewards, dtype=np.float32)
    objective_masks = np.asarray(objective_masks, dtype=np.float32)
    return (rewards * (objective_masks > 1e-8).astype(np.float32, copy=False)).sum(
        axis=0,
        dtype=np.float32,
    )


def _drop_early_objective_bucket(
    objective_masks: np.ndarray,
    *,
    fraction: float = 0.25,
) -> np.ndarray:
    adjusted = np.array(objective_masks, copy=True, dtype=np.float32)
    objective_mask_bool = adjusted > 1e-8
    for env_idx in range(int(adjusted.shape[1])):
        valid_steps = np.flatnonzero(objective_mask_bool[:, env_idx])
        if int(valid_steps.size) == 0:
            continue
        drop_count = max(1, int(math.ceil(float(valid_steps.size) * float(fraction))))
        adjusted[valid_steps[:drop_count], env_idx] = 0.0
    return adjusted


def _keep_only_post_first_terminal_objective_mask(
    objective_masks: np.ndarray,
    *,
    episode_starts: np.ndarray,
) -> np.ndarray:
    adjusted = np.zeros_like(np.asarray(objective_masks, dtype=np.float32))
    objective_mask_bool = np.asarray(objective_masks, dtype=np.float32) > 1e-8
    episode_starts_bool = np.asarray(episode_starts, dtype=np.float32) > 1e-8
    if objective_mask_bool.shape != episode_starts_bool.shape:
        raise ValueError(
            f"objective_masks and episode_starts must match, got {tuple(objective_mask_bool.shape)} vs {tuple(episode_starts_bool.shape)}"
        )
    for env_idx in range(int(adjusted.shape[1])):
        valid_steps = np.flatnonzero(objective_mask_bool[:, env_idx])
        if int(valid_steps.size) <= 1:
            continue
        subsequent_steps = valid_steps[1:]
        subsequent_starts = episode_starts_bool[subsequent_steps, env_idx]
        if not bool(np.any(subsequent_starts)):
            continue
        first_start_idx = int(np.flatnonzero(subsequent_starts)[0])
        kept_steps = subsequent_steps[first_start_idx:]
        adjusted[kept_steps, env_idx] = 1.0
    return adjusted


def _parse_boundary_local_actor_objective_mode(actor_objective_mode: str) -> tuple[str, int] | None:
    actor_objective_mode = str(actor_objective_mode).strip().lower()
    match = re.match(r"^tokenwise_drop_first_(\d+)_suffix_steps$", actor_objective_mode)
    if match is not None:
        return ("drop_first_k", max(1, int(match.group(1))))
    match = re.match(r"^tokenwise_linear_ramp_first_(\d+)_suffix_steps$", actor_objective_mode)
    if match is not None:
        return ("linear_ramp_first_k", max(1, int(match.group(1))))
    return None


def _build_objective_position_metadata(
    episode_starts: np.ndarray,
    objective_masks: np.ndarray,
) -> dict[str, np.ndarray]:
    episode_starts_bool = np.asarray(episode_starts, dtype=np.float32) > 1e-8
    objective_mask_bool = np.asarray(objective_masks, dtype=np.float32) > 1e-8
    if episode_starts_bool.shape != objective_mask_bool.shape:
        raise ValueError(
            f"episode_starts and objective_masks must match, got {tuple(episode_starts_bool.shape)} vs {tuple(objective_mask_bool.shape)}"
        )
    if episode_starts_bool.ndim != 2:
        raise ValueError(
            f"objective position metadata expects 2D [steps, envs] arrays, got ndim={episode_starts_bool.ndim}"
        )

    n_steps, n_envs = episode_starts_bool.shape
    env_indices = np.broadcast_to(
        np.arange(n_envs, dtype=np.int64).reshape((1, n_envs)),
        (n_steps, n_envs),
    ).copy()
    step_indices = np.broadcast_to(
        np.arange(n_steps, dtype=np.int64).reshape((n_steps, 1)),
        (n_steps, n_envs),
    ).copy()
    objective_episode_indices = np.full((n_steps, n_envs), -1, dtype=np.int64)
    objective_episode_positions = np.full((n_steps, n_envs), -1, dtype=np.int64)
    objective_global_positions = np.full((n_steps, n_envs), -1, dtype=np.int64)

    for env_idx in range(int(n_envs)):
        objective_episode_index = -1
        objective_episode_position = -1
        objective_global_position = -1
        for step_idx in range(int(n_steps)):
            if bool(episode_starts_bool[step_idx, env_idx]):
                objective_episode_index += 1
                objective_episode_position = -1
            objective_episode_indices[step_idx, env_idx] = int(objective_episode_index)
            if not bool(objective_mask_bool[step_idx, env_idx]):
                continue
            objective_episode_position += 1
            objective_global_position += 1
            objective_episode_positions[step_idx, env_idx] = int(objective_episode_position)
            objective_global_positions[step_idx, env_idx] = int(objective_global_position)

    return {
        "env_indices": env_indices,
        "step_indices": step_indices,
        "objective_episode_indices": objective_episode_indices,
        "objective_episode_positions": objective_episode_positions,
        "objective_global_positions": objective_global_positions,
    }


def _resolve_train_outer_batch_snapshot_target_mask(
    *,
    objective_mask: np.ndarray,
    flat_env_indices: np.ndarray,
    flat_objective_episode_indices: np.ndarray,
    flat_objective_episode_positions: np.ndarray,
    target_env_index: int | None = None,
    target_objective_episode_index: int | None = None,
    target_objective_position_start: int | None = None,
    target_objective_position_end: int | None = None,
) -> np.ndarray | None:
    objective_mask_bool = np.asarray(objective_mask, dtype=bool).reshape(-1)
    if not bool(np.any(objective_mask_bool)):
        return None
    candidate = np.array(objective_mask_bool, copy=True)
    has_segment_selector = False

    if target_env_index is not None:
        has_segment_selector = True
        candidate &= np.asarray(flat_env_indices, dtype=np.int64).reshape(-1) == int(target_env_index)
    if target_objective_episode_index is not None:
        has_segment_selector = True
        candidate &= (
            np.asarray(flat_objective_episode_indices, dtype=np.int64).reshape(-1)
            == int(target_objective_episode_index)
        )
    if target_objective_position_start is not None:
        has_segment_selector = True
        candidate &= (
            np.asarray(flat_objective_episode_positions, dtype=np.int64).reshape(-1)
            >= int(target_objective_position_start)
        )
    if target_objective_position_end is not None:
        has_segment_selector = True
        candidate &= (
            np.asarray(flat_objective_episode_positions, dtype=np.int64).reshape(-1)
            <= int(target_objective_position_end)
        )
    if not has_segment_selector:
        return None
    return candidate


def _build_train_outer_batch_selector_trace_row(
    *,
    objective_mask: np.ndarray,
    flat_env_indices: np.ndarray,
    flat_objective_episode_indices: np.ndarray,
    flat_objective_episode_positions: np.ndarray,
    epoch_idx: int,
    outer_batch_idx: int,
    total_outer_batches: int,
    selector: dict[str, int | None],
) -> dict[str, Any]:
    objective_mask_bool = np.asarray(objective_mask, dtype=bool).reshape(-1)
    flat_env_indices_np = np.asarray(flat_env_indices, dtype=np.int64).reshape(-1)
    flat_objective_episode_indices_np = np.asarray(
        flat_objective_episode_indices,
        dtype=np.int64,
    ).reshape(-1)
    flat_objective_episode_positions_np = np.asarray(
        flat_objective_episode_positions,
        dtype=np.int64,
    ).reshape(-1)

    objective_total = int(objective_mask_bool.sum())
    objective_env_indices_present = sorted(
        int(v)
        for v in np.unique(flat_env_indices_np[objective_mask_bool]).tolist()
    ) if objective_total > 0 else []

    target_env_index = selector.get("target_env_index", None)
    target_objective_episode_index = selector.get("target_objective_episode_index", None)
    target_objective_position_start = selector.get("target_objective_position_start", None)
    target_objective_position_end = selector.get("target_objective_position_end", None)

    if target_env_index is None:
        target_env_mask = np.array(objective_mask_bool, copy=True)
    else:
        target_env_mask = objective_mask_bool & (flat_env_indices_np == int(target_env_index))
    target_env_objective_token_count = int(target_env_mask.sum())

    target_env_episode_indices_present = sorted(
        int(v)
        for v in np.unique(flat_objective_episode_indices_np[target_env_mask]).tolist()
    ) if target_env_objective_token_count > 0 else []

    target_env_episode_spans: list[dict[str, int]] = []
    for episode_idx in target_env_episode_indices_present:
        episode_mask = target_env_mask & (flat_objective_episode_indices_np == int(episode_idx))
        positions = flat_objective_episode_positions_np[episode_mask]
        if positions.size <= 0:
            continue
        target_env_episode_spans.append(
            {
                "objective_episode_index": int(episode_idx),
                "token_count": int(positions.size),
                "objective_position_min": int(positions.min()),
                "objective_position_max": int(positions.max()),
            }
        )

    if target_objective_episode_index is None:
        target_episode_mask = np.array(target_env_mask, copy=True)
    else:
        target_episode_mask = target_env_mask & (
            flat_objective_episode_indices_np == int(target_objective_episode_index)
        )
    target_episode_token_count = int(target_episode_mask.sum())
    target_episode_positions_present = sorted(
        int(v)
        for v in np.unique(flat_objective_episode_positions_np[target_episode_mask]).tolist()
    ) if target_episode_token_count > 0 else []

    target_match_mask = _resolve_train_outer_batch_snapshot_target_mask(
        objective_mask=objective_mask_bool,
        flat_env_indices=flat_env_indices_np,
        flat_objective_episode_indices=flat_objective_episode_indices_np,
        flat_objective_episode_positions=flat_objective_episode_positions_np,
        target_env_index=target_env_index,
        target_objective_episode_index=target_objective_episode_index,
        target_objective_position_start=target_objective_position_start,
        target_objective_position_end=target_objective_position_end,
    )
    target_match_count = 0 if target_match_mask is None else int(np.asarray(target_match_mask, dtype=bool).sum())
    target_match_positions = (
        sorted(
            int(v)
            for v in np.unique(flat_objective_episode_positions_np[np.asarray(target_match_mask, dtype=bool)]).tolist()
        )
        if target_match_count > 0
        else []
    )

    if objective_total <= 0:
        target_status = "no_objective_tokens"
    elif target_env_index is not None and target_env_objective_token_count <= 0:
        target_status = "target_env_absent"
    elif target_objective_episode_index is not None and target_episode_token_count <= 0:
        target_status = "target_episode_absent"
    elif target_match_count > 0:
        target_status = "target_present"
    elif target_objective_position_start is not None or target_objective_position_end is not None:
        target_status = "target_position_range_absent"
    else:
        target_status = "selector_unmatched"

    return {
        "epoch_idx": int(epoch_idx),
        "outer_batch_idx": int(outer_batch_idx),
        "total_outer_batches": int(total_outer_batches),
        "objective_total": int(objective_total),
        "objective_env_indices_present": objective_env_indices_present,
        "target_status": str(target_status),
        "target_env_index": None if target_env_index is None else int(target_env_index),
        "target_env_objective_token_count": int(target_env_objective_token_count),
        "target_env_objective_episode_indices_present": target_env_episode_indices_present,
        "target_env_objective_episode_spans": target_env_episode_spans,
        "target_objective_episode_index": None
        if target_objective_episode_index is None
        else int(target_objective_episode_index),
        "target_episode_token_count": int(target_episode_token_count),
        "target_episode_positions_present": target_episode_positions_present,
        "target_objective_position_start": None
        if target_objective_position_start is None
        else int(target_objective_position_start),
        "target_objective_position_end": None
        if target_objective_position_end is None
        else int(target_objective_position_end),
        "target_match_count": int(target_match_count),
        "target_match_positions": target_match_positions,
    }


ENV12_MID_EPISODE_EXTENSION_BLOCK_SCALE = 0.7731768424154796
ENV12_MID_EPISODE_EXTENSION_BLOCK_TARGET_ENV_INDEX = 12
# The stable semantic anchor is env12 objective-global positions 18..21.
# Under the current trusted guarded pair2 contract, that anchor maps to
# raw objective episode 4, positions 15..18; see:
# /home/chen/RLPFN/artifacts/phase3_pair2_guarded_selector_reidentified.json
ENV12_MID_EPISODE_EXTENSION_BLOCK_OBJECTIVE_EPISODE_INDEX = 4
ENV12_MID_EPISODE_EXTENSION_BLOCK_POSITION_START = 15
ENV12_MID_EPISODE_EXTENSION_BLOCK_POSITION_END = 18
ENV12_MID_EPISODE_EXTENSION_BLOCK_TARGET_SUITE_NAME = "pair2"
ENV12_SELECTOR_MATCHED_OBJECTIVE_EPISODE_SCALE = ENV12_MID_EPISODE_EXTENSION_BLOCK_SCALE
ENV12_SELECTOR_MATCHED_OBJECTIVE_EPISODE_TARGET_ENV_INDEX = 12
ENV12_SELECTOR_MATCHED_OBJECTIVE_EPISODE_INDEX = 4
ENV12_SELECTOR_MATCHED_OBJECTIVE_EPISODE_TARGET_SUITE_NAME = "pair2"
ENV12_HIGH_MASS_OBJECTIVE_EPISODE_FAMILY_SCALE = ENV12_MID_EPISODE_EXTENSION_BLOCK_SCALE
ENV12_HIGH_MASS_OBJECTIVE_EPISODE_FAMILY_TARGET_ENV_INDEX = 12
ENV12_HIGH_MASS_OBJECTIVE_EPISODE_FAMILY_INDICES = (4, 7, 11)
ENV12_HIGH_MASS_OBJECTIVE_EPISODE_FAMILY_TARGET_SUITE_NAME = "pair2"
PAIR2_RECOVERED_ANCHOR_FIRST_OBJECTIVE_NON_TAIL_FAMILY_SCALE = ENV12_MID_EPISODE_EXTENSION_BLOCK_SCALE
PAIR2_RECOVERED_ANCHOR_FIRST_OBJECTIVE_NON_TAIL_FAMILY_TARGET_ENV_INDICES = (0, 15)
PAIR2_RECOVERED_ANCHOR_FIRST_OBJECTIVE_NON_TAIL_FAMILY_OBJECTIVE_EPISODE_INDEX = 0
PAIR2_RECOVERED_ANCHOR_FIRST_OBJECTIVE_NON_TAIL_FAMILY_TAIL_LEN = 8
PAIR2_RECOVERED_ANCHOR_FIRST_OBJECTIVE_NON_TAIL_FAMILY_TARGET_SUITE_NAME = "pair2"


def _env12_mid_episode_extension_branch_specific_scale_mask(
    objective_masks: np.ndarray,
    *,
    episode_starts: np.ndarray,
) -> np.ndarray:
    adjusted = np.ones_like(np.asarray(objective_masks, dtype=np.float32))
    objective_mask_bool = np.asarray(objective_masks, dtype=np.float32) > 1e-8
    episode_starts_bool = np.asarray(episode_starts, dtype=np.float32) > 1e-8
    if objective_mask_bool.shape != episode_starts_bool.shape:
        raise ValueError(
            f"objective_masks and episode_starts must match, got {tuple(objective_mask_bool.shape)} vs {tuple(episode_starts_bool.shape)}"
        )
    for env_idx in range(int(adjusted.shape[1])):
        if int(env_idx) != int(ENV12_MID_EPISODE_EXTENSION_BLOCK_TARGET_ENV_INDEX):
            continue
        objective_episode_index = -1
        objective_position = -1
        for step in range(int(adjusted.shape[0])):
            if episode_starts_bool[step, env_idx]:
                objective_episode_index += 1
                objective_position = -1
            if not bool(objective_mask_bool[step, env_idx]):
                continue
            objective_position += 1
            if (
                objective_episode_index == ENV12_MID_EPISODE_EXTENSION_BLOCK_OBJECTIVE_EPISODE_INDEX
                and ENV12_MID_EPISODE_EXTENSION_BLOCK_POSITION_START
                <= objective_position
                <= ENV12_MID_EPISODE_EXTENSION_BLOCK_POSITION_END
            ):
                adjusted[step, env_idx] = np.float32(ENV12_MID_EPISODE_EXTENSION_BLOCK_SCALE)
    return adjusted


def _env12_selector_matched_objective_episode_scale_mask(
    objective_masks: np.ndarray,
    *,
    episode_starts: np.ndarray,
) -> np.ndarray:
    adjusted = np.ones_like(np.asarray(objective_masks, dtype=np.float32))
    objective_mask_bool = np.asarray(objective_masks, dtype=np.float32) > 1e-8
    episode_starts_bool = np.asarray(episode_starts, dtype=np.float32) > 1e-8
    if objective_mask_bool.shape != episode_starts_bool.shape:
        raise ValueError(
            f"objective_masks and episode_starts must match, got {tuple(objective_mask_bool.shape)} vs {tuple(episode_starts_bool.shape)}"
        )
    for env_idx in range(int(adjusted.shape[1])):
        if int(env_idx) != int(ENV12_SELECTOR_MATCHED_OBJECTIVE_EPISODE_TARGET_ENV_INDEX):
            continue
        objective_episode_index = -1
        for step in range(int(adjusted.shape[0])):
            if episode_starts_bool[step, env_idx]:
                objective_episode_index += 1
            if not bool(objective_mask_bool[step, env_idx]):
                continue
            if objective_episode_index == ENV12_SELECTOR_MATCHED_OBJECTIVE_EPISODE_INDEX:
                adjusted[step, env_idx] = np.float32(ENV12_SELECTOR_MATCHED_OBJECTIVE_EPISODE_SCALE)
    return adjusted


def _env12_high_mass_objective_episode_family_scale_mask(
    objective_masks: np.ndarray,
    *,
    episode_starts: np.ndarray,
) -> np.ndarray:
    adjusted = np.ones_like(np.asarray(objective_masks, dtype=np.float32))
    objective_mask_bool = np.asarray(objective_masks, dtype=np.float32) > 1e-8
    episode_starts_bool = np.asarray(episode_starts, dtype=np.float32) > 1e-8
    if objective_mask_bool.shape != episode_starts_bool.shape:
        raise ValueError(
            f"objective_masks and episode_starts must match, got {tuple(objective_mask_bool.shape)} vs {tuple(episode_starts_bool.shape)}"
        )
    target_episode_indices = {
        int(v) for v in ENV12_HIGH_MASS_OBJECTIVE_EPISODE_FAMILY_INDICES
    }
    for env_idx in range(int(adjusted.shape[1])):
        if int(env_idx) != int(ENV12_HIGH_MASS_OBJECTIVE_EPISODE_FAMILY_TARGET_ENV_INDEX):
            continue
        objective_episode_index = -1
        for step in range(int(adjusted.shape[0])):
            if episode_starts_bool[step, env_idx]:
                objective_episode_index += 1
            if not bool(objective_mask_bool[step, env_idx]):
                continue
            if objective_episode_index in target_episode_indices:
                adjusted[step, env_idx] = np.float32(ENV12_HIGH_MASS_OBJECTIVE_EPISODE_FAMILY_SCALE)
    return adjusted


def _pair2_recovered_anchor_first_objective_non_tail_family_scale_mask(
    objective_masks: np.ndarray,
    *,
    episode_starts: np.ndarray,
) -> np.ndarray:
    adjusted = np.ones_like(np.asarray(objective_masks, dtype=np.float32))
    objective_mask_bool = np.asarray(objective_masks, dtype=np.float32) > 1e-8
    episode_starts_bool = np.asarray(episode_starts, dtype=np.float32) > 1e-8
    if objective_mask_bool.shape != episode_starts_bool.shape:
        raise ValueError(
            f"objective_masks and episode_starts must match, got {tuple(objective_mask_bool.shape)} vs {tuple(episode_starts_bool.shape)}"
        )
    target_env_indices = {
        int(v) for v in PAIR2_RECOVERED_ANCHOR_FIRST_OBJECTIVE_NON_TAIL_FAMILY_TARGET_ENV_INDICES
    }
    target_objective_episode_index = int(
        PAIR2_RECOVERED_ANCHOR_FIRST_OBJECTIVE_NON_TAIL_FAMILY_OBJECTIVE_EPISODE_INDEX
    )
    tail_len = int(PAIR2_RECOVERED_ANCHOR_FIRST_OBJECTIVE_NON_TAIL_FAMILY_TAIL_LEN)
    for env_idx in range(int(adjusted.shape[1])):
        if int(env_idx) not in target_env_indices:
            continue
        objective_episode_index = -1
        objective_position = -1
        first_episode_steps: list[int] = []
        first_episode_positions: list[int] = []
        for step in range(int(adjusted.shape[0])):
            if episode_starts_bool[step, env_idx]:
                objective_episode_index += 1
                objective_position = -1
            if not bool(objective_mask_bool[step, env_idx]):
                continue
            objective_position += 1
            if int(objective_episode_index) == int(target_objective_episode_index):
                first_episode_steps.append(int(step))
                first_episode_positions.append(int(objective_position))
        if not first_episode_positions:
            continue
        non_tail_end = int(max(first_episode_positions)) - int(tail_len)
        if non_tail_end < 0:
            continue
        for step, objective_position in zip(first_episode_steps, first_episode_positions):
            if int(objective_position) <= int(non_tail_end):
                adjusted[step, env_idx] = np.float32(
                    PAIR2_RECOVERED_ANCHOR_FIRST_OBJECTIVE_NON_TAIL_FAMILY_SCALE
                )
    return adjusted


def _resolve_runtime_scoped_actor_objective_mode(
    actor_objective_mode: str,
    *,
    current_suite_name: str | None = None,
) -> str:
    actor_objective_mode = str(actor_objective_mode).strip().lower()
    suite_name = str(current_suite_name or "").strip().lower()
    if actor_objective_mode in {
        "tokenwise_scale_env12_mid_episode_extension_block",
        "tokenwise_scale_env12_selector_matched_objective_episode",
        "tokenwise_scale_env12_high_mass_objective_episode_family",
        "tokenwise_scale_pair2_recovered_anchor_first_objective_non_tail_family",
    } and suite_name != ENV12_MID_EPISODE_EXTENSION_BLOCK_TARGET_SUITE_NAME:
        return "tokenwise"
    return actor_objective_mode


def _is_supported_actor_objective_mode(actor_objective_mode: str) -> bool:
    actor_objective_mode = str(actor_objective_mode).strip().lower()
    if actor_objective_mode in {
        "tokenwise",
        "trajectory_suffix_return",
        "tokenwise_suffix_return_correction",
        "tokenwise_drop_q1_early",
        "tokenwise_post_first_terminal_only",
        "tokenwise_scale_env12_mid_episode_extension_block",
        "tokenwise_scale_env12_selector_matched_objective_episode",
        "tokenwise_scale_env12_high_mass_objective_episode_family",
        "tokenwise_scale_pair2_recovered_anchor_first_objective_non_tail_family",
        "tokenwise_shift_q1_contract",
    }:
        return True
    return _parse_boundary_local_actor_objective_mode(actor_objective_mode) is not None


def _apply_actor_objective_postprocess(
    actor_advantages: np.ndarray,
    *,
    rewards: np.ndarray,
    objective_masks: np.ndarray,
    episode_starts: np.ndarray,
    actor_objective_mode: str,
) -> np.ndarray:
    actor_objective_mode = str(actor_objective_mode).strip().lower()
    actor_advantages = np.asarray(actor_advantages, dtype=np.float32)
    if actor_objective_mode == "tokenwise":
        return actor_advantages
    if actor_objective_mode == "tokenwise_suffix_return_correction":
        suffix_scores = _compute_suffix_scores_from_objective_mask(rewards, objective_masks)
        if int(actor_advantages.shape[1]) > 1:
            centered_scores = suffix_scores - float(suffix_scores.mean())
            denom = max(float(suffix_scores.std()), 1e-6)
        else:
            centered_scores = suffix_scores
            denom = max(abs(float(suffix_scores[0])), 1.0)
        correction = 1.0 + np.tanh(centered_scores / denom)
        return (actor_advantages * correction.reshape((1, int(actor_advantages.shape[1])))).astype(
            np.float32,
            copy=False,
        )
    if actor_objective_mode == "tokenwise_drop_q1_early":
        adjusted = np.array(actor_advantages, copy=True, dtype=np.float32)
        adjusted_masks = _drop_early_objective_bucket(objective_masks, fraction=0.25)
        dropped_mask = (np.asarray(objective_masks, dtype=np.float32) > 1e-8) & ~(adjusted_masks > 1e-8)
        adjusted[dropped_mask] = 0.0
        return adjusted
    if actor_objective_mode == "tokenwise_post_first_terminal_only":
        adjusted = np.array(actor_advantages, copy=True, dtype=np.float32)
        kept_masks = _keep_only_post_first_terminal_objective_mask(
            objective_masks,
            episode_starts=episode_starts,
        )
        dropped_mask = (np.asarray(objective_masks, dtype=np.float32) > 1e-8) & ~(kept_masks > 1e-8)
        adjusted[dropped_mask] = 0.0
        return adjusted
    if actor_objective_mode == "tokenwise_scale_env12_mid_episode_extension_block":
        adjusted = np.array(actor_advantages, copy=True, dtype=np.float32)
        scaling_mask = _env12_mid_episode_extension_branch_specific_scale_mask(
            objective_masks,
            episode_starts=episode_starts,
        )
        adjusted *= scaling_mask.astype(np.float32, copy=False)
        return adjusted
    if actor_objective_mode == "tokenwise_scale_env12_selector_matched_objective_episode":
        adjusted = np.array(actor_advantages, copy=True, dtype=np.float32)
        scaling_mask = _env12_selector_matched_objective_episode_scale_mask(
            objective_masks,
            episode_starts=episode_starts,
        )
        adjusted *= scaling_mask.astype(np.float32, copy=False)
        return adjusted
    if actor_objective_mode == "tokenwise_scale_env12_high_mass_objective_episode_family":
        adjusted = np.array(actor_advantages, copy=True, dtype=np.float32)
        scaling_mask = _env12_high_mass_objective_episode_family_scale_mask(
            objective_masks,
            episode_starts=episode_starts,
        )
        adjusted *= scaling_mask.astype(np.float32, copy=False)
        return adjusted
    if actor_objective_mode == "tokenwise_scale_pair2_recovered_anchor_first_objective_non_tail_family":
        adjusted = np.array(actor_advantages, copy=True, dtype=np.float32)
        scaling_mask = _pair2_recovered_anchor_first_objective_non_tail_family_scale_mask(
            objective_masks,
            episode_starts=episode_starts,
        )
        adjusted *= scaling_mask.astype(np.float32, copy=False)
        return adjusted
    boundary_local = _parse_boundary_local_actor_objective_mode(actor_objective_mode)
    if boundary_local is not None:
        mode_kind, step_count = boundary_local
        adjusted = np.array(actor_advantages, copy=True, dtype=np.float32)
        objective_mask_bool = np.asarray(objective_masks, dtype=np.float32) > 1e-8
        for env_idx in range(int(adjusted.shape[1])):
            valid_steps = np.flatnonzero(objective_mask_bool[:, env_idx])
            if int(valid_steps.size) == 0:
                continue
            local_count = min(int(step_count), int(valid_steps.size))
            early_steps = valid_steps[:local_count]
            if mode_kind == "drop_first_k":
                adjusted[early_steps, env_idx] = 0.0
                continue
            if mode_kind == "linear_ramp_first_k":
                ramp = (np.arange(local_count, dtype=np.float32) + 1.0) / float(local_count)
                adjusted[early_steps, env_idx] *= ramp
                continue
            raise ValueError(f"Unsupported boundary-local actor objective mode kind: {mode_kind}")
        return adjusted
    raise ValueError(f"Unsupported PPO actor objective postprocess mode: {actor_objective_mode}")


def _recover_raw_from_value_space(
    value_space_tensor: torch.Tensor,
    *,
    value_means: torch.Tensor | np.ndarray | float,
    value_stds: torch.Tensor | np.ndarray | float,
    eps: float = PPO_VALUE_TARGET_NORM_EPS,
) -> torch.Tensor:
    targets = value_space_tensor.to(dtype=torch.float32)
    if torch.is_tensor(value_means):
        means = value_means.to(device=targets.device, dtype=targets.dtype)
    else:
        means = torch.as_tensor(value_means, device=targets.device, dtype=targets.dtype)
    if torch.is_tensor(value_stds):
        stds = value_stds.to(device=targets.device, dtype=targets.dtype)
    else:
        stds = torch.as_tensor(value_stds, device=targets.device, dtype=targets.dtype)
    while means.ndim < targets.ndim:
        means = means.unsqueeze(-1)
    while stds.ndim < targets.ndim:
        stds = stds.unsqueeze(-1)
    if int(means.numel()) == 1:
        means = means.expand_as(targets)
    if int(stds.numel()) == 1:
        stds = stds.expand_as(targets)
    return means + stds.clamp_min(float(eps)) * targets


class _RolloutScalarAccumulator:
    def __init__(self) -> None:
        self.sum = 0.0
        self.sumsq = 0.0
        self.count = 0

    def add(self, values: np.ndarray | list[float] | float) -> None:
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        if int(arr.size) <= 0:
            return
        self.sum += float(arr.sum())
        self.sumsq += float(np.square(arr).sum())
        self.count += int(arr.size)

    def mean_std(self) -> tuple[Optional[float], Optional[float]]:
        if int(self.count) <= 0:
            return None, None
        mean = float(self.sum / float(self.count))
        var = max(0.0, (self.sumsq / float(self.count)) - (mean ** 2))
        return mean, float(math.sqrt(var))


class _TorchRolloutScalarAccumulator:
    def __init__(self, *, device: torch.device) -> None:
        self.device = device
        self.sum = torch.zeros((), device=device, dtype=torch.float64)
        self.sumsq = torch.zeros((), device=device, dtype=torch.float64)
        self.count = torch.zeros((), device=device, dtype=torch.int64)

    def add(self, values: torch.Tensor | np.ndarray | list[float] | float) -> None:
        if torch.is_tensor(values):
            arr = values.to(device=self.device, dtype=torch.float64).reshape(-1)
        else:
            arr = torch.as_tensor(values, device=self.device, dtype=torch.float64).reshape(-1)
        if int(arr.numel()) <= 0:
            return
        self.sum = self.sum + arr.sum()
        self.sumsq = self.sumsq + arr.square().sum()
        self.count = self.count + int(arr.numel())

    def mean_std(self) -> tuple[Optional[float], Optional[float]]:
        count = int(self.count.detach().cpu().item())
        if count <= 0:
            return None, None
        sum_v = float(self.sum.detach().cpu().item())
        sumsq_v = float(self.sumsq.detach().cpu().item())
        mean = float(sum_v / float(count))
        var = max(0.0, (sumsq_v / float(count)) - (mean ** 2))
        return mean, float(math.sqrt(var))


class _PpoRolloutLogAccumulator:
    _COMPONENT_KEYS = (
        "reward",
        "reward_env",
        "reward_ctrl",
        "reward_survival",
        "reward_terminal_bonus",
    )

    def __init__(self, n_envs: int, *, device: Optional[torch.device] = None) -> None:
        self.n_envs = int(n_envs)
        self._torch_device = device
        accum_cls = (
            (lambda: _TorchRolloutScalarAccumulator(device=device))
            if device is not None
            else _RolloutScalarAccumulator
        )
        self._objective_step_stats = {
            key: accum_cls()
            for key in self._COMPONENT_KEYS
        }
        self._objective_episode_stats = {
            key: accum_cls()
            for key in self._COMPONENT_KEYS
        }
        self._full_episode_stats = {
            key: accum_cls()
            for key in self._COMPONENT_KEYS
        }
        self._objective_length_stats = accum_cls()
        self._full_length_stats = accum_cls()
        if device is None:
            self._objective_current = {
                key: np.zeros((self.n_envs,), dtype=np.float64)
                for key in self._COMPONENT_KEYS
            }
            self._full_current = {
                key: np.zeros((self.n_envs,), dtype=np.float64)
                for key in self._COMPONENT_KEYS
            }
            self._objective_lengths = np.zeros((self.n_envs,), dtype=np.int64)
            self._full_lengths = np.zeros((self.n_envs,), dtype=np.int64)
        else:
            self._objective_current = {
                key: torch.zeros((self.n_envs,), device=device, dtype=torch.float64)
                for key in self._COMPONENT_KEYS
            }
            self._full_current = {
                key: torch.zeros((self.n_envs,), device=device, dtype=torch.float64)
                for key in self._COMPONENT_KEYS
            }
            self._objective_lengths = torch.zeros((self.n_envs,), device=device, dtype=torch.int64)
            self._full_lengths = torch.zeros((self.n_envs,), device=device, dtype=torch.int64)

    def observe_step(
        self,
        *,
        reward: torch.Tensor | np.ndarray,
        reward_env: torch.Tensor | np.ndarray,
        reward_ctrl: torch.Tensor | np.ndarray,
        reward_survival: torch.Tensor | np.ndarray,
        reward_terminal_bonus: torch.Tensor | np.ndarray,
        objective_mask: torch.Tensor | np.ndarray,
        dones: torch.Tensor | np.ndarray,
    ) -> None:
        if self._torch_device is not None:
            self._observe_step_torch(
                reward=reward,
                reward_env=reward_env,
                reward_ctrl=reward_ctrl,
                reward_survival=reward_survival,
                reward_terminal_bonus=reward_terminal_bonus,
                objective_mask=objective_mask,
                dones=dones,
            )
            return
        values = {
            "reward": np.asarray(reward, dtype=np.float64).reshape((self.n_envs,)),
            "reward_env": np.asarray(reward_env, dtype=np.float64).reshape((self.n_envs,)),
            "reward_ctrl": np.asarray(reward_ctrl, dtype=np.float64).reshape((self.n_envs,)),
            "reward_survival": np.asarray(reward_survival, dtype=np.float64).reshape((self.n_envs,)),
            "reward_terminal_bonus": np.asarray(reward_terminal_bonus, dtype=np.float64).reshape((self.n_envs,)),
        }
        objective_mask_np = np.asarray(objective_mask, dtype=bool).reshape((self.n_envs,))
        dones_np = np.asarray(dones, dtype=bool).reshape((self.n_envs,))

        for key, arr in values.items():
            self._full_current[key] += arr
        self._full_lengths += 1

        if bool(np.any(objective_mask_np)):
            active_idx = np.nonzero(objective_mask_np)[0]
            for key, arr in values.items():
                self._objective_step_stats[key].add(arr[active_idx])
                self._objective_current[key][active_idx] += arr[active_idx]
            self._objective_lengths[active_idx] += 1

        for env_idx in np.nonzero(dones_np)[0].tolist():
            self._flush_full_episode(int(env_idx))
            self._flush_objective_episode(int(env_idx))

    def _observe_step_torch(
        self,
        *,
        reward: torch.Tensor | np.ndarray,
        reward_env: torch.Tensor | np.ndarray,
        reward_ctrl: torch.Tensor | np.ndarray,
        reward_survival: torch.Tensor | np.ndarray,
        reward_terminal_bonus: torch.Tensor | np.ndarray,
        objective_mask: torch.Tensor | np.ndarray,
        dones: torch.Tensor | np.ndarray,
    ) -> None:
        device = self._torch_device
        assert device is not None
        values = {
            "reward": torch.as_tensor(reward, device=device, dtype=torch.float64).reshape((self.n_envs,)),
            "reward_env": torch.as_tensor(reward_env, device=device, dtype=torch.float64).reshape((self.n_envs,)),
            "reward_ctrl": torch.as_tensor(reward_ctrl, device=device, dtype=torch.float64).reshape((self.n_envs,)),
            "reward_survival": torch.as_tensor(reward_survival, device=device, dtype=torch.float64).reshape((self.n_envs,)),
            "reward_terminal_bonus": torch.as_tensor(
                reward_terminal_bonus,
                device=device,
                dtype=torch.float64,
            ).reshape((self.n_envs,)),
        }
        objective_is_scalar = not torch.is_tensor(objective_mask) and np.asarray(objective_mask).ndim == 0
        objective_active = bool(objective_mask) if objective_is_scalar else None
        objective_mask_t = torch.as_tensor(objective_mask, device=device, dtype=torch.bool)
        if int(objective_mask_t.numel()) == 1:
            objective_mask_t = objective_mask_t.expand((self.n_envs,))
        objective_mask_t = objective_mask_t.reshape((self.n_envs,))
        dones_t = torch.as_tensor(dones, device=device, dtype=torch.bool).reshape((self.n_envs,))

        for key, arr in values.items():
            self._full_current[key] = self._full_current[key] + arr
        self._full_lengths = self._full_lengths + 1

        if objective_active is True or (objective_active is None and bool(objective_mask_t.any())):
            for key, arr in values.items():
                active_vals = arr[objective_mask_t]
                self._objective_step_stats[key].add(active_vals)
                self._objective_current[key][objective_mask_t] = (
                    self._objective_current[key][objective_mask_t] + active_vals
                )
            self._objective_lengths[objective_mask_t] = self._objective_lengths[objective_mask_t] + 1

        done_idx = torch.nonzero(dones_t, as_tuple=False).reshape(-1)
        self._flush_full_episode_torch(done_idx)
        self._flush_objective_episode_torch(done_idx)

    def finalize(self) -> dict[str, Optional[float]]:
        if self._torch_device is not None:
            all_idx = torch.arange(self.n_envs, device=self._torch_device, dtype=torch.long)
            self._flush_objective_episode_torch(all_idx)
        else:
            for env_idx in range(self.n_envs):
                self._flush_objective_episode(env_idx)
        stats: dict[str, Optional[float]] = {}
        for key, accum in self._objective_step_stats.items():
            mean, std = accum.mean_std()
            stats[f"{key}_mean"] = mean
            stats[f"{key}_std"] = std
        self._populate_episode_stats(
            stats,
            prefix="",
            episode_stats=self._objective_episode_stats,
            length_stats=self._objective_length_stats,
        )
        self._populate_episode_stats(
            stats,
            prefix="full_",
            episode_stats=self._full_episode_stats,
            length_stats=self._full_length_stats,
        )
        return stats

    def _flush_full_episode(self, env_idx: int) -> None:
        episode_len = int(self._full_lengths[env_idx])
        if episode_len <= 0:
            return
        self._full_length_stats.add(float(episode_len))
        for key, current in self._full_current.items():
            self._full_episode_stats[key].add(float(current[env_idx]))
            current[env_idx] = 0.0
        self._full_lengths[env_idx] = 0

    def _flush_objective_episode(self, env_idx: int) -> None:
        episode_len = int(self._objective_lengths[env_idx])
        if episode_len <= 0:
            return
        self._objective_length_stats.add(float(episode_len))
        for key, current in self._objective_current.items():
            self._objective_episode_stats[key].add(float(current[env_idx]))
            current[env_idx] = 0.0
        self._objective_lengths[env_idx] = 0

    def _flush_full_episode_torch(self, env_idx_t: torch.Tensor) -> None:
        if self._torch_device is None or int(env_idx_t.numel()) <= 0:
            return
        valid_mask = self._full_lengths.index_select(0, env_idx_t) > 0
        if not bool(valid_mask.any()):
            return
        env_idx_t = env_idx_t[valid_mask]
        episode_lens = self._full_lengths.index_select(0, env_idx_t)
        self._full_length_stats.add(episode_lens)
        for key, current in self._full_current.items():
            self._full_episode_stats[key].add(current.index_select(0, env_idx_t))
            current.index_fill_(0, env_idx_t, 0.0)
        self._full_lengths.index_fill_(0, env_idx_t, 0)

    def _flush_objective_episode_torch(self, env_idx_t: torch.Tensor) -> None:
        if self._torch_device is None or int(env_idx_t.numel()) <= 0:
            return
        valid_mask = self._objective_lengths.index_select(0, env_idx_t) > 0
        if not bool(valid_mask.any()):
            return
        env_idx_t = env_idx_t[valid_mask]
        episode_lens = self._objective_lengths.index_select(0, env_idx_t)
        self._objective_length_stats.add(episode_lens)
        for key, current in self._objective_current.items():
            self._objective_episode_stats[key].add(current.index_select(0, env_idx_t))
            current.index_fill_(0, env_idx_t, 0.0)
        self._objective_lengths.index_fill_(0, env_idx_t, 0)

    @staticmethod
    def _populate_episode_stats(
        stats: dict[str, Optional[float]],
        *,
        prefix: str,
        episode_stats: dict[str, _RolloutScalarAccumulator],
        length_stats: _RolloutScalarAccumulator,
    ) -> None:
        reward_mean, reward_std = episode_stats["reward"].mean_std()
        length_mean, length_std = length_stats.mean_std()
        if prefix:
            stats[f"{prefix}ep_rew_mean"] = reward_mean
            stats[f"{prefix}ep_rew_std"] = reward_std
            stats[f"{prefix}ep_len_mean"] = length_mean
            stats[f"{prefix}ep_len_std"] = length_std
        else:
            stats["ep_rew_mean"] = reward_mean
            stats["ep_rew_std"] = reward_std
            stats["ep_len_mean"] = length_mean
            stats["ep_len_std"] = length_std
        for key, accum in episode_stats.items():
            mean, std = accum.mean_std()
            if prefix:
                stats[f"{prefix}{key}_return_mean"] = mean
                stats[f"{prefix}{key}_return_std"] = std
            else:
                stats[f"{key}_return_mean"] = mean
                stats[f"{key}_return_std"] = std


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
    objective_masks: torch.Tensor
    rollout_return_means: torch.Tensor
    rollout_return_stds: torch.Tensor
    lstm_states: RNNStates
    episode_starts: torch.Tensor
    mask: torch.Tensor
    action_masks: torch.Tensor
    next_states: torch.Tensor
    next_state_masks: torch.Tensor
    actor_advantages: Optional[torch.Tensor] = None
    normalized_q_targets: Optional[torch.Tensor] = None
    value_target_bucket_idx: Optional[torch.Tensor] = None


class MaskedRecurrentFlatBatchSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    old_values: Optional[torch.Tensor]
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    objective_masks: torch.Tensor
    rollout_return_means: Optional[torch.Tensor]
    rollout_return_stds: Optional[torch.Tensor]
    lstm_states: RNNStates
    episode_starts: torch.Tensor
    action_masks: torch.Tensor
    next_states: Optional[torch.Tensor]
    next_state_masks: Optional[torch.Tensor]
    seq_start_indices: np.ndarray
    seq_lengths: np.ndarray
    actor_advantages: Optional[torch.Tensor] = None
    normalized_q_targets: Optional[torch.Tensor] = None
    value_target_bucket_idx: Optional[torch.Tensor] = None
    flat_batch_indices: Optional[np.ndarray] = None
    flat_env_indices: Optional[np.ndarray] = None
    flat_step_indices: Optional[np.ndarray] = None
    flat_objective_episode_indices: Optional[np.ndarray] = None
    flat_objective_episode_positions: Optional[np.ndarray] = None
    flat_objective_global_positions: Optional[np.ndarray] = None

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
        objective_masks=_slice_padded_sequence_tensor(
            rollout_data.objective_masks,
            n_seq=n_seq,
            start_seq=int(start_seq),
            end_seq=int(end_seq),
        ),
        rollout_return_means=_slice_padded_sequence_tensor(
            rollout_data.rollout_return_means,
            n_seq=n_seq,
            start_seq=int(start_seq),
            end_seq=int(end_seq),
        ),
        rollout_return_stds=_slice_padded_sequence_tensor(
            rollout_data.rollout_return_stds,
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
        actor_advantages=(
            None
            if rollout_data.actor_advantages is None
            else _slice_padded_sequence_tensor(
                rollout_data.actor_advantages,
                n_seq=n_seq,
                start_seq=int(start_seq),
                end_seq=int(end_seq),
            )
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
    objective_masks_padded, _ = MaskedRecurrentRolloutBuffer._pad_device_tensor(
        flat_batch.objective_masks[flat_start:flat_end],
        seq_start_indices=local_seq_starts,
    )
    rollout_return_means_padded, _ = MaskedRecurrentRolloutBuffer._pad_device_tensor(
        flat_batch.rollout_return_means[flat_start:flat_end],
        seq_start_indices=local_seq_starts,
    )
    rollout_return_stds_padded, _ = MaskedRecurrentRolloutBuffer._pad_device_tensor(
        flat_batch.rollout_return_stds[flat_start:flat_end],
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
            value_target_bucket_idx=MaskedRecurrentRolloutBuffer._pad_device_tensor(
                flat_batch.value_target_bucket_idx[flat_start:flat_end],
                seq_start_indices=local_seq_starts,
            )[0].reshape((padded_batch_size,)),
            objective_masks=objective_masks_padded.reshape((padded_batch_size,)),
            rollout_return_means=rollout_return_means_padded.reshape((padded_batch_size,)),
            rollout_return_stds=rollout_return_stds_padded.reshape((padded_batch_size,)),
            lstm_states=RNNStates(pi_states, vf_states),
            episode_starts=episode_starts_padded.reshape((padded_batch_size,)),
            mask=valid.reshape((padded_batch_size,)).to(device=returns_padded.device, dtype=returns_padded.dtype),
            action_masks=action_masks_padded.reshape((padded_batch_size, flat_batch.action_masks.shape[-1])),
            next_states=next_states_padded.reshape((padded_batch_size, flat_batch.next_states.shape[-1])),
            next_state_masks=next_state_masks_padded.reshape((padded_batch_size, flat_batch.next_state_masks.shape[-1])),
            actor_advantages=(
                None
                if flat_batch.actor_advantages is None
                else MaskedRecurrentRolloutBuffer._pad_device_tensor(
                    flat_batch.actor_advantages[flat_start:flat_end],
                    seq_start_indices=local_seq_starts,
                )[0].reshape((padded_batch_size,))
            ),
        ),
        flat_start,
        flat_end,
        local_seq_starts,
    )


class MaskedRecurrentRolloutBuffer(RecurrentRolloutBuffer):
    action_masks: np.ndarray
    next_states: np.ndarray
    next_state_masks: np.ndarray
    objective_masks: np.ndarray
    rollout_return_means: np.ndarray
    rollout_return_stds: np.ndarray
    actor_advantages: np.ndarray
    value_target_bucket_idx: np.ndarray

    def __init__(self, *args, next_state_dim: int, **kwargs):
        self.next_state_dim = int(next_state_dim)
        self._actor_gae_space = "normalized"
        self._actor_baseline_mode = "learned"
        self._actor_gae_lambda_override = None
        self._value_target_space = "normalized"
        self._deterministic_batch_plan = False
        self._cross_rollout_running_ema_rms_alpha = CROSS_ROLLOUT_RUNNING_EMA_RMS_ALPHA
        self._cross_rollout_running_ema_rms_mean_square = None
        super().__init__(*args, **kwargs)

    def reset(self):
        enable_q_target_cache = bool(getattr(self, "_enable_q_target_cache", False))
        if not isinstance(self.action_space, spaces.Box):
            raise ValueError("MaskedRecurrentRolloutBuffer only supports Box action spaces.")
        self.mask_dims = int(np.prod(self.action_space.shape, dtype=np.int64))
        self.action_masks = np.ones((self.buffer_size, self.n_envs, self.mask_dims), dtype=np.float32)
        self.next_states = np.zeros((self.buffer_size, self.n_envs, self.next_state_dim), dtype=np.float32)
        self.next_state_masks = np.zeros((self.buffer_size, self.n_envs, self.next_state_dim), dtype=np.float32)
        self.objective_masks = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.rollout_return_means = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.rollout_return_stds = np.ones((self.buffer_size, self.n_envs), dtype=np.float32)
        self.rollout_raw_returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.actor_advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.value_target_bucket_idx = np.full((self.buffer_size, self.n_envs), -1, dtype=np.int64)
        self._enable_q_target_cache = enable_q_target_cache
        self._global_normalized_q_targets_flat = None
        self._global_q_target_seq_ids_flat = None
        self._global_q_target_seq_lengths = None
        self._global_value_target_bucket_idx_flat = None
        self._flat_env_indices = None
        self._flat_step_indices = None
        self._flat_objective_episode_indices = None
        self._flat_objective_episode_positions = None
        self._flat_objective_global_positions = None
        super().reset()

    def clear_device_cache(self) -> None:
        return None

    def _ensure_generator_ready(self) -> None:
        if self.generator_ready:
            return
        position_metadata = _build_objective_position_metadata(
            self.episode_starts,
            self.objective_masks,
        )
        self._flat_env_indices = self.swap_and_flatten(position_metadata["env_indices"]).reshape(-1).astype(
            np.int64,
            copy=False,
        )
        self._flat_step_indices = self.swap_and_flatten(position_metadata["step_indices"]).reshape(-1).astype(
            np.int64,
            copy=False,
        )
        self._flat_objective_episode_indices = self.swap_and_flatten(
            position_metadata["objective_episode_indices"]
        ).reshape(-1).astype(np.int64, copy=False)
        self._flat_objective_episode_positions = self.swap_and_flatten(
            position_metadata["objective_episode_positions"]
        ).reshape(-1).astype(np.int64, copy=False)
        self._flat_objective_global_positions = self.swap_and_flatten(
            position_metadata["objective_global_positions"]
        ).reshape(-1).astype(np.int64, copy=False)
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
            "objective_masks",
            "rollout_return_means",
            "rollout_return_stds",
            "actor_advantages",
            "value_target_bucket_idx",
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

    def set_deterministic_batch_plan(self, enabled: bool = True) -> None:
        self._deterministic_batch_plan = bool(enabled)

    @staticmethod
    def _normalize_sequence_fragment_with_mask_like_reinforce(
        values: torch.Tensor,
        mask: torch.Tensor,
        *,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        if values.ndim != 1 or mask.ndim != 1:
            raise ValueError("values and mask must be flattened 1D tensors.")
        if values.shape != mask.shape:
            raise ValueError("values and mask must have identical shapes.")
        if not bool(mask.any()):
            return torch.zeros_like(values)
        mean, std = _masked_mean_std(values, mask, eps=eps)
        normalized = (values - mean) / std
        return normalized * mask.to(device=values.device, dtype=values.dtype)

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
        returns_flat = self._get_flat_raw_returns_tensor(dtype=torch.float32)
        objective_masks_flat = torch.as_tensor(self.objective_masks, dtype=torch.bool).reshape(-1)
        normalized_q_targets_flat = self._normalize_sequence_fragment_with_mask_like_reinforce(
            returns_flat,
            objective_masks_flat,
        )
        seq_ids_flat = np.empty((total_steps,), dtype=np.int64)
        seq_lengths = np.empty((len(seq_start_indices),), dtype=np.int64)
        for seq_id, (seq_start, seq_end) in enumerate(zip(seq_start_indices.tolist(), seq_end_exclusive.tolist())):
            seq_start = int(seq_start)
            seq_end = int(seq_end)
            seq_ids_flat[seq_start:seq_end] = int(seq_id)
            seq_lengths[int(seq_id)] = int(seq_end - seq_start)
        self._global_normalized_q_targets_flat = normalized_q_targets_flat.cpu().numpy().astype(np.float32, copy=False)
        self._global_q_target_seq_ids_flat = seq_ids_flat
        self._global_q_target_seq_lengths = seq_lengths

    def get_flat_position_metadata(self, batch_inds: np.ndarray) -> dict[str, np.ndarray]:
        self._ensure_generator_ready()
        batch_inds = np.asarray(batch_inds, dtype=np.int64)
        return {
            "flat_batch_indices": batch_inds.astype(np.int64, copy=True),
            "flat_env_indices": np.asarray(self._flat_env_indices[batch_inds], dtype=np.int64),
            "flat_step_indices": np.asarray(self._flat_step_indices[batch_inds], dtype=np.int64),
            "flat_objective_episode_indices": np.asarray(
                self._flat_objective_episode_indices[batch_inds],
                dtype=np.int64,
            ),
            "flat_objective_episode_positions": np.asarray(
                self._flat_objective_episode_positions[batch_inds],
                dtype=np.int64,
            ),
            "flat_objective_global_positions": np.asarray(
                self._flat_objective_global_positions[batch_inds],
                dtype=np.int64,
            ),
        }

    def _get_flat_raw_returns_tensor(
        self,
        *,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        self._ensure_generator_ready()
        raw_returns_flat = torch.as_tensor(self.rollout_raw_returns, dtype=dtype, device=device).reshape(-1)
        return raw_returns_flat.reshape(-1)

    def _get_flat_raw_value_targets_tensor(
        self,
        *,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        self._ensure_generator_ready()
        targets_flat = torch.as_tensor(self.returns, dtype=dtype, device=device).reshape(-1)
        means_flat = torch.as_tensor(self.rollout_return_means, dtype=dtype, device=device).reshape(-1)
        stds_flat = torch.as_tensor(self.rollout_return_stds, dtype=dtype, device=device).reshape(-1)
        return _recover_raw_from_value_space(
            targets_flat,
            value_means=means_flat,
            value_stds=stds_flat,
        ).reshape(-1)

    def _cache_value_target_bucket_idx(self) -> None:
        borders = getattr(self, "_value_target_bardist_borders", None)
        if borders is None:
            self.value_target_bucket_idx.fill(-1)
            self._global_value_target_bucket_idx_flat = None
            return
        borders = np.asarray(borders, dtype=np.float32).reshape(-1)
        if borders.ndim != 1 or int(borders.shape[0]) < 2:
            raise ValueError(f"Invalid cached value borders: {tuple(borders.shape)}")
        returns_flat = np.asarray(self.returns, dtype=np.float32).reshape(-1)
        target_sample = np.searchsorted(borders, returns_flat, side="left") - 1
        target_sample[returns_flat == borders[0]] = 0
        target_sample[returns_flat == borders[-1]] = int(borders.shape[0]) - 2
        target_sample = np.clip(
            target_sample,
            0,
            int(borders.shape[0]) - 2,
        )
        self.value_target_bucket_idx = target_sample.reshape(self.buffer_size, self.n_envs).astype(np.int64, copy=False)
        self._global_value_target_bucket_idx_flat = self.value_target_bucket_idx.reshape(-1).astype(np.int64, copy=False)

    def _get_flat_raw_values_tensor(
        self,
        *,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        self._ensure_generator_ready()
        values_flat = torch.as_tensor(self.values, dtype=dtype, device=device).reshape(-1)
        means_flat = torch.as_tensor(self.rollout_return_means, dtype=dtype, device=device).reshape(-1)
        stds_flat = torch.as_tensor(self.rollout_return_stds, dtype=dtype, device=device).reshape(-1)
        return _recover_raw_from_value_space(
            values_flat,
            value_means=means_flat,
            value_stds=stds_flat,
        ).reshape(-1)

    def compute_returns_and_advantage(
        self,
        last_values: torch.Tensor,
        dones: np.ndarray,
    ) -> None:
        actor_objective_mode = str(getattr(self, "_actor_objective_mode", "tokenwise")).strip().lower()
        actor_objective_mode = _resolve_runtime_scoped_actor_objective_mode(
            actor_objective_mode,
            current_suite_name=getattr(self, "_actor_objective_runtime_current_suite_name", None),
        )
        if actor_objective_mode == "tokenwise_shift_q1_contract":
            self.objective_masks[:] = _drop_early_objective_bucket(self.objective_masks, fraction=0.25)
            actor_objective_mode = "tokenwise"
        value_target_space = str(getattr(self, "_value_target_space", "raw")).strip().lower()
        if value_target_space not in {"raw", "normalized", "reward_normalized", CROSS_ROLLOUT_RUNNING_EMA_RMS_SPACE}:
            raise ValueError(
                f"Unsupported PPO value target space: {value_target_space}. "
                "Expected 'raw', 'normalized', 'reward_normalized', or "
                f"'{CROSS_ROLLOUT_RUNNING_EMA_RMS_SPACE}'."
            )
        last_values_np = last_values.clone().cpu().numpy().flatten().astype(np.float32, copy=False)
        rollout_raw_returns = _discounted_returns_from_rewards(
            self.rewards,
            episode_starts=self.episode_starts,
            dones=dones,
            gamma=self.gamma,
        )
        self.rollout_raw_returns[:] = np.asarray(rollout_raw_returns, dtype=np.float32)
        rollout_raw_return_means_np = rollout_raw_returns.mean(axis=0, dtype=np.float64).astype(np.float32, copy=False)
        rollout_raw_return_means_np_full, rollout_raw_return_stds_np_full = _rollout_return_norm_stats_from_raw_returns(
            rollout_raw_returns,
            eps=PPO_VALUE_TARGET_NORM_EPS,
        )
        reward_norm_means_np, reward_norm_stds_np = _reward_norm_stats_from_rewards_with_mask(
            self.rewards,
            mask=self.objective_masks,
            eps=PPO_VALUE_TARGET_NORM_EPS,
        )
        reward_norm_rewards = (
            self.rewards - reward_norm_means_np.reshape((1, self.n_envs))
        ) / reward_norm_stds_np.reshape((1, self.n_envs))
        reward_norm_returns = _discounted_returns_from_rewards(
            reward_norm_rewards,
            episode_starts=self.episode_starts,
            dones=dones,
            gamma=self.gamma,
        )
        if value_target_space == "normalized":
            rollout_return_means_np, rollout_return_stds_np = _rollout_return_norm_stats_from_raw_returns(
                rollout_raw_returns,
                eps=PPO_VALUE_TARGET_NORM_EPS,
            )
            rollout_return_mean_over_std = rollout_return_means_np / rollout_return_stds_np
            last_gae_lam = np.zeros((self.n_envs,), dtype=np.float32)
            for step in reversed(range(self.buffer_size)):
                if step == self.buffer_size - 1:
                    next_non_terminal = 1.0 - dones.astype(np.float32, copy=False)
                    next_values = last_values_np
                else:
                    next_non_terminal = 1.0 - self.episode_starts[step + 1].astype(np.float32, copy=False)
                    next_values = self.values[step + 1]
                curr_values = self.values[step]
                reward_norm = (
                    self.rewards[step] / rollout_return_stds_np
                    + ((self.gamma * next_non_terminal) - 1.0) * rollout_return_mean_over_std
                )
                gamma_eff = self.gamma * next_non_terminal
                delta = reward_norm + gamma_eff * next_values - curr_values
                last_gae_lam = delta + gamma_eff * self.gae_lambda * last_gae_lam
                self.advantages[step] = last_gae_lam
        elif value_target_space == "reward_normalized":
            rollout_return_means_np = rollout_raw_return_means_np_full
            rollout_return_stds_np = rollout_raw_return_stds_np_full
            last_gae_lam = np.zeros((self.n_envs,), dtype=np.float32)
            for step in reversed(range(self.buffer_size)):
                if step == self.buffer_size - 1:
                    next_non_terminal = 1.0 - dones.astype(np.float32, copy=False)
                    next_values = last_values_np
                else:
                    next_non_terminal = 1.0 - self.episode_starts[step + 1].astype(np.float32, copy=False)
                    next_values = self.values[step + 1]
                curr_values = self.values[step]
                reward_scaled = reward_norm_rewards[step]
                gamma_eff = self.gamma * next_non_terminal
                delta = reward_scaled + gamma_eff * next_values - curr_values
                last_gae_lam = delta + gamma_eff * self.gae_lambda * last_gae_lam
                self.advantages[step] = last_gae_lam
        elif value_target_space == CROSS_ROLLOUT_RUNNING_EMA_RMS_SPACE:
            rollout_return_means_np, rollout_return_stds_np, running_mean_square_np = (
                _cross_rollout_running_ema_rms_stats_from_raw_returns(
                    rollout_raw_returns,
                    previous_mean_square=getattr(self, "_cross_rollout_running_ema_rms_mean_square", None),
                    alpha=float(getattr(self, "_cross_rollout_running_ema_rms_alpha", CROSS_ROLLOUT_RUNNING_EMA_RMS_ALPHA)),
                    eps=PPO_VALUE_TARGET_NORM_EPS,
                )
            )
            self._cross_rollout_running_ema_rms_mean_square = np.asarray(
                running_mean_square_np,
                dtype=np.float32,
            ).copy()
            last_gae_lam = np.zeros((self.n_envs,), dtype=np.float32)
            for step in reversed(range(self.buffer_size)):
                if step == self.buffer_size - 1:
                    next_non_terminal = 1.0 - dones.astype(np.float32, copy=False)
                    next_values = last_values_np
                else:
                    next_non_terminal = 1.0 - self.episode_starts[step + 1].astype(np.float32, copy=False)
                    next_values = self.values[step + 1]
                curr_values = self.values[step]
                reward_scaled = self.rewards[step] / rollout_return_stds_np
                gamma_eff = self.gamma * next_non_terminal
                delta = reward_scaled + gamma_eff * next_values - curr_values
                last_gae_lam = delta + gamma_eff * self.gae_lambda * last_gae_lam
                self.advantages[step] = last_gae_lam
        else:
            rollout_return_means_np = np.zeros((self.n_envs,), dtype=np.float32)
            rollout_return_stds_np = np.ones((self.n_envs,), dtype=np.float32)
            last_gae_lam = np.zeros((self.n_envs,), dtype=np.float32)
            for step in reversed(range(self.buffer_size)):
                if step == self.buffer_size - 1:
                    next_non_terminal = 1.0 - dones.astype(np.float32, copy=False)
                    next_values = last_values_np
                else:
                    next_non_terminal = 1.0 - self.episode_starts[step + 1].astype(np.float32, copy=False)
                    next_values = self.values[step + 1]
                curr_values = self.values[step]
                gamma_eff = self.gamma * next_non_terminal
                delta = self.rewards[step] + gamma_eff * next_values - curr_values
                last_gae_lam = delta + gamma_eff * self.gae_lambda * last_gae_lam
                self.advantages[step] = last_gae_lam
        # Repeat per-env rollout stats across steps so any sampled sub-batch can
        # reconstruct raw-space values/returns without extra rollout context.
        if np.asarray(rollout_return_means_np).shape == self.rewards.shape:
            self.rollout_return_means[:] = np.asarray(rollout_return_means_np, dtype=np.float32)
        else:
            self.rollout_return_means[:] = np.asarray(rollout_return_means_np, dtype=np.float32).reshape((1, self.n_envs))
        if np.asarray(rollout_return_stds_np).shape == self.rewards.shape:
            self.rollout_return_stds[:] = np.asarray(rollout_return_stds_np, dtype=np.float32)
        else:
            self.rollout_return_stds[:] = np.asarray(rollout_return_stds_np, dtype=np.float32).reshape((1, self.n_envs))
        rollout_return_means_full_np = np.asarray(self.rollout_return_means, dtype=np.float32)
        rollout_return_stds_full_np = np.asarray(self.rollout_return_stds, dtype=np.float32)
        rollout_return_means_env_np = rollout_return_means_full_np[-1]
        rollout_return_stds_env_np = rollout_return_stds_full_np[-1]
        self.returns = self.advantages + self.values
        self._cache_value_target_bucket_idx()
        if actor_objective_mode == "trajectory_suffix_return":
            suffix_scores = _compute_suffix_scores_from_objective_mask(self.rewards, self.objective_masks)
            repeated_scores = np.broadcast_to(
                suffix_scores.reshape((1, self.n_envs)),
                (self.buffer_size, self.n_envs),
            ).astype(np.float32, copy=False)
            self.actor_advantages[:] = repeated_scores
            return
        if not _is_supported_actor_objective_mode(actor_objective_mode) or actor_objective_mode in {
            "trajectory_suffix_return",
            "tokenwise_shift_q1_contract",
        }:
            raise ValueError(
                f"Unsupported PPO actor objective mode: {actor_objective_mode}"
            )
        actor_baseline_mode = str(getattr(self, "_actor_baseline_mode", "learned")).strip().lower()
        actor_gae_space = str(getattr(self, "_actor_gae_space", "normalized")).strip().lower()
        actor_reward_stds_np = None
        actor_reward_means_np = None
        actor_reward_norm_returns = None
        if actor_gae_space == "reward_normalized":
            actor_reward_means_np = reward_norm_means_np
            actor_reward_stds_np = reward_norm_stds_np
            actor_reward_norm_returns = reward_norm_returns.astype(np.float32, copy=False)
        if actor_baseline_mode == "zero":
            if actor_gae_space == "reward_normalized":
                self.actor_advantages[:] = actor_reward_norm_returns.astype(np.float32, copy=False)
            else:
                self.actor_advantages[:] = rollout_raw_returns.astype(np.float32, copy=False)
            return
        if actor_baseline_mode == "rollout_mean":
            if actor_gae_space == "reward_normalized":
                self.actor_advantages[:] = (
                    actor_reward_norm_returns
                    - actor_reward_norm_returns.mean(axis=0, keepdims=True)
                ).astype(np.float32, copy=False)
            else:
                self.actor_advantages[:] = (
                    rollout_raw_returns - rollout_raw_return_means_np.reshape((1, self.n_envs))
                ).astype(np.float32, copy=False)
            return
        if actor_baseline_mode != "learned":
            raise ValueError(
                f"Unsupported PPO actor baseline mode: {actor_baseline_mode}"
            )
        actor_gae_space = str(getattr(self, "_actor_gae_space", "normalized")).strip().lower()
        if actor_gae_space == "reward_normalized":
            if value_target_space == "reward_normalized":
                self.actor_advantages[:] = self.advantages
                self.actor_advantages[:] = _apply_actor_objective_postprocess(
                    self.actor_advantages,
                    rewards=self.rewards,
                    objective_masks=self.objective_masks,
                    episode_starts=self.episode_starts,
                    actor_objective_mode=actor_objective_mode,
                )
                return
            raise ValueError(
                "actor_gae_space='reward_normalized' requires value_target_space='reward_normalized' "
                "to avoid path-dependent reward/value normalization leakage."
            )
        if actor_gae_space == CROSS_ROLLOUT_RUNNING_EMA_RMS_SPACE:
            if value_target_space == CROSS_ROLLOUT_RUNNING_EMA_RMS_SPACE:
                self.actor_advantages[:] = self.advantages
                self.actor_advantages[:] = _apply_actor_objective_postprocess(
                    self.actor_advantages,
                    rewards=self.rewards,
                    objective_masks=self.objective_masks,
                    episode_starts=self.episode_starts,
                    actor_objective_mode=actor_objective_mode,
                )
                return
            raise ValueError(
                f"actor_gae_space='{CROSS_ROLLOUT_RUNNING_EMA_RMS_SPACE}' requires "
                f"value_target_space='{CROSS_ROLLOUT_RUNNING_EMA_RMS_SPACE}' "
                "to avoid path-dependent reward/value normalization leakage."
            )
        if actor_gae_space == "normalized":
            actor_gae_lambda_override = getattr(self, "_actor_gae_lambda_override", None)
            if actor_gae_lambda_override is None:
                self.actor_advantages[:] = self.advantages
            else:
                if value_target_space != "normalized":
                    raise ValueError(
                        "actor_gae_lambda_override with actor_gae_space='normalized' "
                        "requires value_target_space='normalized'."
                    )
                actor_gae_lambda = float(actor_gae_lambda_override)
                if not math.isfinite(actor_gae_lambda) or actor_gae_lambda < 0.0 or actor_gae_lambda > 1.0:
                    raise ValueError(
                        f"actor_gae_lambda_override must be in [0, 1], got {actor_gae_lambda_override!r}."
                    )
                actor_last_gae_lam = np.zeros((self.n_envs,), dtype=np.float32)
                for step in reversed(range(self.buffer_size)):
                    if step == self.buffer_size - 1:
                        next_non_terminal = 1.0 - dones.astype(np.float32, copy=False)
                        next_values = last_values_np
                    else:
                        next_non_terminal = 1.0 - self.episode_starts[step + 1].astype(np.float32, copy=False)
                        next_values = self.values[step + 1]
                    curr_values = self.values[step]
                    reward_norm = (
                        self.rewards[step] / rollout_return_stds_np
                        + ((self.gamma * next_non_terminal) - 1.0) * rollout_return_mean_over_std
                    )
                    gamma_eff = self.gamma * next_non_terminal
                    delta = reward_norm + gamma_eff * next_values - curr_values
                    actor_last_gae_lam = delta + gamma_eff * actor_gae_lambda * actor_last_gae_lam
                    self.actor_advantages[step] = actor_last_gae_lam
            self.actor_advantages[:] = _apply_actor_objective_postprocess(
                self.actor_advantages,
                rewards=self.rewards,
                objective_masks=self.objective_masks,
                episode_starts=self.episode_starts,
                actor_objective_mode=actor_objective_mode,
            )
            return
        if actor_gae_space == "normalized_sep_synced":
            actor_episode_starts = _episode_starts_synced_to_objective_mask(
                self.episode_starts,
                self.objective_masks,
            )
            actor_rollout_raw_returns = _discounted_returns_from_rewards(
                self.rewards,
                episode_starts=actor_episode_starts,
                dones=dones,
                gamma=self.gamma,
            )
            actor_return_means_np, actor_return_stds_np = _rollout_return_norm_stats_from_raw_returns(
                actor_rollout_raw_returns,
                eps=PPO_VALUE_TARGET_NORM_EPS,
            )
            values_raw = (
                rollout_return_means_full_np
                + rollout_return_stds_full_np * self.values.astype(np.float32, copy=False)
            )
            last_values_raw = (
                rollout_return_means_env_np
                + rollout_return_stds_env_np * last_values_np.astype(np.float32, copy=False)
            )
            actor_values = (
                values_raw - actor_return_means_np.reshape((1, self.n_envs))
            ) / actor_return_stds_np.reshape((1, self.n_envs))
            actor_last_values = (
                last_values_raw - actor_return_means_np
            ) / actor_return_stds_np
            actor_return_mean_over_std = actor_return_means_np / actor_return_stds_np
            last_gae_lam_actor = np.zeros((self.n_envs,), dtype=np.float32)
            for step in reversed(range(self.buffer_size)):
                if step == self.buffer_size - 1:
                    next_non_terminal = 1.0 - dones.astype(np.float32, copy=False)
                    next_values = actor_last_values
                else:
                    next_non_terminal = 1.0 - actor_episode_starts[step + 1].astype(np.float32, copy=False)
                    next_values = actor_values[step + 1]
                curr_values = actor_values[step]
                reward_norm = (
                    self.rewards[step] / actor_return_stds_np
                    + ((self.gamma * next_non_terminal) - 1.0) * actor_return_mean_over_std
                )
                gamma_eff = self.gamma * next_non_terminal
                delta = reward_norm + gamma_eff * next_values - curr_values
                last_gae_lam_actor = delta + gamma_eff * self.gae_lambda * last_gae_lam_actor
                self.actor_advantages[step] = last_gae_lam_actor
            self.actor_advantages[:] = _apply_actor_objective_postprocess(
                self.actor_advantages,
                rewards=self.rewards,
                objective_masks=self.objective_masks,
                episode_starts=actor_episode_starts,
                actor_objective_mode=actor_objective_mode,
            )
            return
        if actor_gae_space != "raw":
            raise ValueError(f"Unsupported PPO actor GAE space: {actor_gae_space}")

        actor_gae_lambda = self.gae_lambda
        actor_gae_lambda_override = getattr(self, "_actor_gae_lambda_override", None)
        if actor_gae_lambda_override is not None:
            actor_gae_lambda = float(actor_gae_lambda_override)
            if not math.isfinite(actor_gae_lambda) or actor_gae_lambda < 0.0 or actor_gae_lambda > 1.0:
                raise ValueError(
                    f"actor_gae_lambda_override must be in [0, 1], got {actor_gae_lambda_override!r}."
                )
        values_raw = (
            rollout_return_means_full_np
            + rollout_return_stds_full_np * self.values.astype(np.float32, copy=False)
        )
        last_values_raw = (
            rollout_return_means_env_np
            + rollout_return_stds_env_np * last_values_np.astype(np.float32, copy=False)
        )
        last_gae_lam_raw = np.zeros((self.n_envs,), dtype=np.float32)
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones.astype(np.float32, copy=False)
                next_values_raw = last_values_raw
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1].astype(np.float32, copy=False)
                next_values_raw = values_raw[step + 1]
            curr_values_raw = values_raw[step]
            gamma_eff = self.gamma * next_non_terminal
            delta_raw = self.rewards[step] + gamma_eff * next_values_raw - curr_values_raw
            last_gae_lam_raw = delta_raw + gamma_eff * actor_gae_lambda * last_gae_lam_raw
            self.actor_advantages[step] = last_gae_lam_raw
        self.actor_advantages[:] = _apply_actor_objective_postprocess(
            self.actor_advantages,
            rewards=self.rewards,
            objective_masks=self.objective_masks,
            episode_starts=self.episode_starts,
            actor_objective_mode=actor_objective_mode,
        )

    def _get_cached_flat_normalized_q_targets(
        self,
        *,
        batch_inds: np.ndarray,
        seq_start_indices: np.ndarray,
        seq_lengths: np.ndarray,
        returns_flat: torch.Tensor,
        objective_masks_flat: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        self._ensure_global_q_target_cache()
        batch_inds = np.asarray(batch_inds, dtype=np.int64)
        seq_start_indices = np.asarray(seq_start_indices, dtype=np.int64)
        seq_lengths = np.asarray(seq_lengths, dtype=np.int64)
        objective_masks_flat = objective_masks_flat.to(device=returns_flat.device, dtype=torch.bool).reshape(-1)
        cached_targets = torch.as_tensor(
            self._global_normalized_q_targets_flat[batch_inds],
            device=returns_flat.device,
            dtype=returns_flat.dtype,
        ).contiguous()
        del seq_start_indices, seq_lengths, returns_flat, objective_masks_flat, eps
        return cached_targets

    def _iter_batch_plan(self, batch_size: Optional[int] = None):
        assert self.full, "Rollout buffer must be full before sampling from it"
        self._ensure_generator_ready()
        batch_size = self._resolve_batch_size(batch_size)

        total = self.buffer_size * self.n_envs
        if bool(getattr(self, "_deterministic_batch_plan", False)):
            indices = np.arange(total, dtype=np.int64)
        else:
            split_index = np.random.randint(total)
            indices = np.arange(total, dtype=np.int64)
            indices = np.concatenate((indices[split_index:], indices[:split_index]))

        env_change = self._build_env_change_flat()

        start_idx = 0
        while start_idx < total:
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

    @staticmethod
    def _to_float32_numpy_fast(array) -> np.ndarray:
        if isinstance(array, np.ndarray):
            if array.dtype == np.float32:
                return array
            return array.astype(np.float32, copy=False)
        return np.asarray(array, dtype=np.float32)

    def add(
        self,
        *args,
        action_masks: Optional[np.ndarray] = None,
        next_states: Optional[np.ndarray] = None,
        next_state_masks: Optional[np.ndarray] = None,
        objective_masks: Optional[np.ndarray] = None,
        **kwargs,
    ) -> None:
        if len(args) >= 2:
            obs_arg = args[0]
            action_arg = self._to_float32_numpy_fast(args[1])
            if action_arg.ndim == 1:
                action_arg = action_arg.reshape(self.n_envs, -1)
            elif action_arg.ndim > 2:
                action_arg = action_arg.reshape(self.n_envs, -1)
            if int(action_arg.shape[0]) != int(self.n_envs):
                raise ValueError(
                    f"action batch mismatch: expected n_envs={int(self.n_envs)}, got shape={tuple(action_arg.shape)}"
                )
            if int(action_arg.shape[-1]) == int(self.action_dim):
                args = (obs_arg, action_arg, *args[2:])
            else:
                padded_action = np.zeros((self.n_envs, self.action_dim), dtype=np.float32)
                copy_dim = int(min(int(action_arg.shape[-1]), int(self.action_dim)))
                if copy_dim > 0:
                    padded_action[:, :copy_dim] = action_arg[:, :copy_dim]
                args = (obs_arg, padded_action, *args[2:])
        if action_masks is not None:
            action_masks_np = self._to_float32_numpy_fast(action_masks)
            if action_masks_np.ndim == 1:
                action_masks_np = action_masks_np.reshape(self.n_envs, -1)
            elif action_masks_np.ndim > 2:
                action_masks_np = action_masks_np.reshape(self.n_envs, -1)
            if int(action_masks_np.shape[0]) != int(self.n_envs):
                raise ValueError(
                    f"action_masks batch mismatch: expected n_envs={int(self.n_envs)}, got shape={tuple(action_masks_np.shape)}"
                )
            if int(action_masks_np.shape[-1]) == int(self.mask_dims):
                self.action_masks[self.pos] = action_masks_np
            else:
                padded_action_masks = np.zeros((self.n_envs, self.mask_dims), dtype=np.float32)
                copy_dim = int(min(int(action_masks_np.shape[-1]), int(self.mask_dims)))
                if copy_dim > 0:
                    padded_action_masks[:, :copy_dim] = action_masks_np[:, :copy_dim]
                self.action_masks[self.pos] = padded_action_masks
        if next_states is not None:
            next_states_np = self._to_float32_numpy_fast(next_states)
            if next_states_np.shape == (self.n_envs, self.next_state_dim):
                self.next_states[self.pos] = next_states_np
            else:
                self.next_states[self.pos] = next_states_np.reshape((self.n_envs, self.next_state_dim))
        if next_state_masks is not None:
            next_state_masks_np = self._to_float32_numpy_fast(next_state_masks)
            if next_state_masks_np.shape == (self.n_envs, self.next_state_dim):
                self.next_state_masks[self.pos] = next_state_masks_np
            else:
                self.next_state_masks[self.pos] = next_state_masks_np.reshape((self.n_envs, self.next_state_dim))
        if objective_masks is not None:
            objective_masks_np = self._to_float32_numpy_fast(objective_masks)
            if objective_masks_np.shape == (self.n_envs,):
                self.objective_masks[self.pos] = objective_masks_np
            else:
                self.objective_masks[self.pos] = objective_masks_np.reshape((self.n_envs,))
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
            value_target_bucket_idx=self.pad_and_flatten(self.value_target_bucket_idx[batch_inds]),
            objective_masks=self.pad_and_flatten(self.objective_masks[batch_inds]),
            rollout_return_means=self.pad_and_flatten(self.rollout_return_means[batch_inds]),
            rollout_return_stds=self.pad_and_flatten(self.rollout_return_stds[batch_inds]),
            lstm_states=RNNStates(lstm_states_pi, lstm_states_vf),
            episode_starts=self.pad_and_flatten(self.episode_starts[batch_inds]),
            mask=self.pad_and_flatten(np.ones_like(self.returns[batch_inds])),
            action_masks=self.pad(self.action_masks[batch_inds]).reshape((padded_batch_size, self.mask_dims)),
            next_states=self.pad(self.next_states[batch_inds]).reshape((padded_batch_size, self.next_state_dim)),
            next_state_masks=self.pad(self.next_state_masks[batch_inds]).reshape((padded_batch_size, self.next_state_dim)),
            actor_advantages=self.pad_and_flatten(self.actor_advantages[batch_inds]),
        )

    def get_gpu(self, batch_size: Optional[int] = None):
        for batch_inds, env_change in self._iter_batch_plan(batch_size):
            yield self._get_samples_gpu(batch_inds, env_change)

    def get_gpu_flat(
        self,
        batch_size: Optional[int] = None,
        *,
        include_old_values: bool = True,
        include_value_affine: bool = True,
        include_aux_tensors: bool = True,
        include_position_metadata: bool = True,
    ):
        for batch_inds, env_change in self._iter_batch_plan(batch_size):
            yield self._get_flat_batch_gpu(
                batch_inds,
                env_change,
                include_old_values=include_old_values,
                include_value_affine=include_value_affine,
                include_aux_tensors=include_aux_tensors,
                include_position_metadata=include_position_metadata,
            )

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
        actor_advantages_flat = self.to_torch(self.actor_advantages[batch_inds]).reshape(-1).contiguous()
        returns_flat = self.to_torch(self.returns[batch_inds]).reshape(-1).contiguous()
        value_target_bucket_idx_flat = self.to_torch(self.value_target_bucket_idx[batch_inds]).reshape(-1).contiguous()
        objective_masks_flat = self.to_torch(self.objective_masks[batch_inds]).reshape(-1).contiguous()
        rollout_return_means_flat = self.to_torch(self.rollout_return_means[batch_inds]).reshape(-1).contiguous()
        rollout_return_stds_flat = self.to_torch(self.rollout_return_stds[batch_inds]).reshape(-1).contiguous()
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
                objective_masks_flat=objective_masks_flat,
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
        actor_advantages_padded, _ = self._pad_device_tensor(
            actor_advantages_flat,
            seq_start_indices=self.seq_start_indices,
        )
        returns_padded, _ = self._pad_device_tensor(
            returns_flat,
            seq_start_indices=self.seq_start_indices,
        )
        value_target_bucket_idx_padded, _ = self._pad_device_tensor(
            value_target_bucket_idx_flat,
            seq_start_indices=self.seq_start_indices,
        )
        objective_masks_padded, _ = self._pad_device_tensor(
            objective_masks_flat,
            seq_start_indices=self.seq_start_indices,
        )
        rollout_return_means_padded, _ = self._pad_device_tensor(
            rollout_return_means_flat,
            seq_start_indices=self.seq_start_indices,
        )
        rollout_return_stds_padded, _ = self._pad_device_tensor(
            rollout_return_stds_flat,
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
            value_target_bucket_idx=value_target_bucket_idx_padded.reshape((padded_batch_size,)),
            objective_masks=objective_masks_padded.reshape((padded_batch_size,)),
            rollout_return_means=rollout_return_means_padded.reshape((padded_batch_size,)),
            rollout_return_stds=rollout_return_stds_padded.reshape((padded_batch_size,)),
            lstm_states=RNNStates(lstm_states_pi, lstm_states_vf),
            episode_starts=episode_starts_padded.reshape((padded_batch_size,)),
            mask=valid.reshape((padded_batch_size,)).to(device=self.device, dtype=returns_flat.dtype),
            action_masks=action_masks_padded.reshape((padded_batch_size, self.mask_dims)),
            next_states=next_states_padded.reshape((padded_batch_size, self.next_state_dim)),
            next_state_masks=next_state_masks_padded.reshape((padded_batch_size, self.next_state_dim)),
            actor_advantages=actor_advantages_padded.reshape((padded_batch_size,)),
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
        *,
        include_old_values: bool = True,
        include_value_affine: bool = True,
        include_aux_tensors: bool = True,
        include_position_metadata: bool = True,
    ) -> MaskedRecurrentFlatBatchSamples:
        self.seq_start_indices, _, _ = self._create_sequencers(batch_inds, env_change)
        seq_start_indices = np.asarray(self.seq_start_indices, dtype=np.int64)
        seq_end_exclusive = np.empty_like(seq_start_indices)
        seq_end_exclusive[:-1] = seq_start_indices[1:]
        seq_end_exclusive[-1] = int(len(batch_inds))
        seq_lengths = (seq_end_exclusive - seq_start_indices).astype(np.int64, copy=False)

        observations_flat = self.to_torch(self.observations[batch_inds]).contiguous()
        actions_flat = self.to_torch(self.actions[batch_inds]).contiguous()
        values_flat = (
            self.to_torch(self.values[batch_inds]).reshape(-1).contiguous()
            if bool(include_old_values)
            else None
        )
        log_probs_flat = self.to_torch(self.log_probs[batch_inds]).reshape(-1).contiguous()
        advantages_flat = self.to_torch(self.advantages[batch_inds]).reshape(-1).contiguous()
        actor_advantages_flat = self.to_torch(self.actor_advantages[batch_inds]).reshape(-1).contiguous()
        returns_flat = self.to_torch(self.returns[batch_inds]).reshape(-1).contiguous()
        value_target_bucket_idx_flat = self.to_torch(self.value_target_bucket_idx[batch_inds]).reshape(-1).contiguous()
        objective_masks_flat = self.to_torch(self.objective_masks[batch_inds]).reshape(-1).contiguous()
        rollout_return_means_flat = (
            self.to_torch(self.rollout_return_means[batch_inds]).reshape(-1).contiguous()
            if bool(include_value_affine)
            else None
        )
        rollout_return_stds_flat = (
            self.to_torch(self.rollout_return_stds[batch_inds]).reshape(-1).contiguous()
            if bool(include_value_affine)
            else None
        )
        episode_starts_flat = self.to_torch(self.episode_starts[batch_inds]).reshape(-1).contiguous()
        action_masks_flat = self.to_torch(self.action_masks[batch_inds]).contiguous()
        next_states_flat = (
            self.to_torch(self.next_states[batch_inds]).contiguous()
            if bool(include_aux_tensors)
            else None
        )
        next_state_masks_flat = (
            self.to_torch(self.next_state_masks[batch_inds]).contiguous()
            if bool(include_aux_tensors)
            else None
        )
        n_seq = int(len(seq_start_indices))
        dummy_lstm_state = torch.zeros((1, n_seq, 1), device=self.device, dtype=torch.float32)
        normalized_q_targets_flat = None
        if bool(getattr(self, "_enable_q_target_cache", False)):
            normalized_q_targets_flat = self._get_cached_flat_normalized_q_targets(
                batch_inds=batch_inds,
                seq_start_indices=seq_start_indices,
                seq_lengths=seq_lengths,
                returns_flat=returns_flat,
                objective_masks_flat=objective_masks_flat,
            )
        position_metadata = (
            self.get_flat_position_metadata(batch_inds)
            if bool(include_position_metadata)
            else {
                "flat_batch_indices": None,
                "flat_env_indices": None,
                "flat_step_indices": None,
                "flat_objective_episode_indices": None,
                "flat_objective_episode_positions": None,
                "flat_objective_global_positions": None,
            }
        )

        return MaskedRecurrentFlatBatchSamples(
            observations=observations_flat,
            actions=actions_flat,
            old_values=values_flat,
            old_log_prob=log_probs_flat,
            advantages=advantages_flat,
            returns=returns_flat,
            value_target_bucket_idx=value_target_bucket_idx_flat,
            objective_masks=objective_masks_flat,
            rollout_return_means=rollout_return_means_flat,
            rollout_return_stds=rollout_return_stds_flat,
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
            actor_advantages=actor_advantages_flat,
            normalized_q_targets=normalized_q_targets_flat,
            flat_batch_indices=position_metadata["flat_batch_indices"],
            flat_env_indices=position_metadata["flat_env_indices"],
            flat_step_indices=position_metadata["flat_step_indices"],
            flat_objective_episode_indices=position_metadata["flat_objective_episode_indices"],
            flat_objective_episode_positions=position_metadata["flat_objective_episode_positions"],
            flat_objective_global_positions=position_metadata["flat_objective_global_positions"],
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
        if bool(getattr(self.policy, "has_value_bardist", lambda: False)()):
            self.rollout_buffer._value_target_bardist_borders = (
                self.policy.get_value_bardist().borders.detach().cpu().numpy().astype(np.float32, copy=False)
            )

    def _should_capture_train_outer_batch_snapshot(self) -> bool:
        snapshot_path = getattr(self, "_rwkv_train_outer_batch_snapshot_path", None)
        return snapshot_path is not None and not bool(
            getattr(self, "_rwkv_train_outer_batch_snapshot_written", False)
        )

    def _should_capture_rollout_collect_snapshot(self) -> bool:
        snapshot_path = getattr(self, "_rwkv_rollout_collect_snapshot_path", None)
        return snapshot_path is not None and not bool(
            getattr(self, "_rwkv_rollout_collect_snapshot_written", False)
        )

    def _should_capture_train_outer_batch_selector_trace(self) -> bool:
        return getattr(self, "_rwkv_train_outer_batch_selector_trace_path", None) is not None

    def _get_train_outer_batch_snapshot_selector(self) -> dict[str, int | None]:
        return {
            "target_outer_batch_idx": getattr(
                self,
                "_rwkv_train_outer_batch_snapshot_target_outer_batch_idx",
                None,
            ),
            "target_env_index": getattr(
                self,
                "_rwkv_train_outer_batch_snapshot_target_env_index",
                None,
            ),
            "target_objective_episode_index": getattr(
                self,
                "_rwkv_train_outer_batch_snapshot_target_objective_episode_index",
                None,
            ),
            "target_objective_position_start": getattr(
                self,
                "_rwkv_train_outer_batch_snapshot_target_objective_position_start",
                None,
            ),
            "target_objective_position_end": getattr(
                self,
                "_rwkv_train_outer_batch_snapshot_target_objective_position_end",
                None,
            ),
        }

    def _resolve_train_outer_batch_snapshot_match_mask_flat(
        self,
        *,
        rollout_data: MaskedRecurrentFlatBatchSamples,
        objective_mask: torch.Tensor,
        outer_batch_idx: int,
    ) -> np.ndarray | None:
        if not self._should_capture_train_outer_batch_snapshot():
            return None
        selector = self._get_train_outer_batch_snapshot_selector()
        target_outer_batch_idx = selector["target_outer_batch_idx"]
        if target_outer_batch_idx is not None and int(outer_batch_idx) != int(target_outer_batch_idx):
            return None

        objective_mask_np = objective_mask.detach().to(dtype=torch.bool).cpu().numpy().reshape(-1)
        target_mask = _resolve_train_outer_batch_snapshot_target_mask(
            objective_mask=objective_mask_np,
            flat_env_indices=rollout_data.flat_env_indices,
            flat_objective_episode_indices=rollout_data.flat_objective_episode_indices,
            flat_objective_episode_positions=rollout_data.flat_objective_episode_positions,
            target_env_index=selector["target_env_index"],
            target_objective_episode_index=selector["target_objective_episode_index"],
            target_objective_position_start=selector["target_objective_position_start"],
            target_objective_position_end=selector["target_objective_position_end"],
        )
        if target_mask is None:
            return objective_mask_np
        if not bool(np.any(target_mask)):
            return None
        return target_mask

    def _write_train_outer_batch_snapshot(self, payload: dict[str, Any]) -> None:
        snapshot_path = getattr(self, "_rwkv_train_outer_batch_snapshot_path", None)
        if snapshot_path is None:
            return
        resolved = Path(str(snapshot_path)).expanduser().resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(json.dumps(payload, indent=2, sort_keys=True))
        self._rwkv_train_outer_batch_snapshot_written = True
        self._rwkv_last_train_outer_batch_snapshot_path = str(resolved)

    def _write_rollout_collect_snapshot(self, payload: dict[str, Any]) -> None:
        snapshot_path = getattr(self, "_rwkv_rollout_collect_snapshot_path", None)
        if snapshot_path is None:
            return
        resolved = Path(str(snapshot_path)).expanduser().resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(json.dumps(payload, indent=2, sort_keys=True))
        self._rwkv_rollout_collect_snapshot_written = True
        self._rwkv_last_rollout_collect_snapshot_path = str(resolved)

    def _write_train_outer_batch_selector_trace(self) -> None:
        trace_path = getattr(self, "_rwkv_train_outer_batch_selector_trace_path", None)
        if trace_path is None:
            return
        resolved = Path(str(trace_path)).expanduser().resolve()
        rows = list(getattr(self, "_rwkv_train_outer_batch_selector_trace_rows", []) or [])
        matched_rows = [
            {
                "epoch_idx": int(row["epoch_idx"]),
                "outer_batch_idx": int(row["outer_batch_idx"]),
                "target_match_count": int(row["target_match_count"]),
            }
            for row in rows
            if int(row.get("target_match_count", 0)) > 0
        ]
        payload = {
            "trace_entry": "phase3_train_outer_batch_selector_trace",
            "selector": {
                key: (None if value is None else int(value))
                for key, value in self._get_train_outer_batch_snapshot_selector().items()
            },
            "summary": {
                "row_count": int(len(rows)),
                "target_appeared": bool(matched_rows),
                "matched_outer_batches": matched_rows,
                "first_match": None if not matched_rows else dict(matched_rows[0]),
            },
            "completed": bool(getattr(self, "_rwkv_train_outer_batch_selector_trace_completed", False)),
            "rows": rows,
        }
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(json.dumps(payload, indent=2, sort_keys=True))
        self._rwkv_last_train_outer_batch_selector_trace_path = str(resolved)

    def _append_train_outer_batch_selector_trace_row_flat(
        self,
        *,
        rollout_data: MaskedRecurrentFlatBatchSamples,
        objective_mask: torch.Tensor,
        epoch_idx: int,
        outer_batch_idx: int,
        total_outer_batches: int,
    ) -> None:
        if not self._should_capture_train_outer_batch_selector_trace():
            return
        metadata_fields = {
            "flat_env_indices": rollout_data.flat_env_indices,
            "flat_objective_episode_indices": rollout_data.flat_objective_episode_indices,
            "flat_objective_episode_positions": rollout_data.flat_objective_episode_positions,
        }
        missing = [name for name, value in metadata_fields.items() if value is None]
        if missing:
            raise ValueError(
                "Train outer-batch selector trace requires flat batch metadata, missing "
                f"{missing!r}"
            )
        objective_mask_np = objective_mask.detach().to(dtype=torch.bool).cpu().numpy().reshape(-1)
        row = _build_train_outer_batch_selector_trace_row(
            objective_mask=objective_mask_np,
            flat_env_indices=np.asarray(metadata_fields["flat_env_indices"], dtype=np.int64),
            flat_objective_episode_indices=np.asarray(
                metadata_fields["flat_objective_episode_indices"],
                dtype=np.int64,
            ),
            flat_objective_episode_positions=np.asarray(
                metadata_fields["flat_objective_episode_positions"],
                dtype=np.int64,
            ),
            epoch_idx=int(epoch_idx),
            outer_batch_idx=int(outer_batch_idx),
            total_outer_batches=int(total_outer_batches),
            selector=self._get_train_outer_batch_snapshot_selector(),
        )
        rows = getattr(self, "_rwkv_train_outer_batch_selector_trace_rows", None)
        if rows is None:
            rows = []
            self._rwkv_train_outer_batch_selector_trace_rows = rows
        rows.append(row)
        self._write_train_outer_batch_selector_trace()

    def _finalize_train_outer_batch_selector_trace(self) -> None:
        if not self._should_capture_train_outer_batch_selector_trace():
            return
        self._rwkv_train_outer_batch_selector_trace_completed = True
        self._write_train_outer_batch_selector_trace()

    def _maybe_capture_train_outer_batch_snapshot_flat(
        self,
        *,
        rollout_data: MaskedRecurrentFlatBatchSamples,
        objective_mask: torch.Tensor,
        pre_normalization_actor_advantages: np.ndarray,
        post_normalization_advantages: np.ndarray,
        ratio_snapshot: np.ndarray,
        clipped_objective_snapshot: np.ndarray,
        clip_range: float,
        outer_batch_idx: int,
        epoch_idx: int,
        matched_target_mask: np.ndarray | None = None,
    ) -> None:
        if not self._should_capture_train_outer_batch_snapshot():
            return
        metadata_fields = {
            "flat_batch_indices": rollout_data.flat_batch_indices,
            "flat_env_indices": rollout_data.flat_env_indices,
            "flat_step_indices": rollout_data.flat_step_indices,
            "flat_objective_episode_indices": rollout_data.flat_objective_episode_indices,
            "flat_objective_episode_positions": rollout_data.flat_objective_episode_positions,
            "flat_objective_global_positions": rollout_data.flat_objective_global_positions,
        }
        missing = [name for name, value in metadata_fields.items() if value is None]
        if missing:
            raise ValueError(
                "Train outer-batch snapshot requires flat batch metadata, missing "
                f"{missing!r}"
            )
        if rollout_data.old_values is None:
            raise ValueError("Train outer-batch snapshot requires old_values in the flat batch sample.")
        objective_mask_np = objective_mask.detach().to(dtype=torch.bool).cpu().numpy().reshape(-1)
        payload = {
            "snapshot_entry": "phase3_train_outer_batch_snapshot",
            "summary": {
                "outer_batch_idx": int(outer_batch_idx),
                "epoch_idx": int(epoch_idx),
                "batch_size_flat": int(len(pre_normalization_actor_advantages)),
                "objective_total": int(objective_mask_np.sum()),
                "actor_objective_mode": str(getattr(self, "_rwkv_actor_objective_mode", "")),
                "actor_objective_runtime_current_suite_name": str(
                    getattr(self.rollout_buffer, "_actor_objective_runtime_current_suite_name", "")
                ),
                "normalize_advantage": bool(self.normalize_advantage),
                "clip_range": float(clip_range),
            },
            "snapshot_target_selector": {
                key: (None if value is None else int(value))
                for key, value in self._get_train_outer_batch_snapshot_selector().items()
            },
            "snapshot_target_match_count": 0
            if matched_target_mask is None
            else int(np.asarray(matched_target_mask, dtype=bool).sum()),
            "flat_batch_indices": np.asarray(
                metadata_fields["flat_batch_indices"],
                dtype=np.int64,
            ).tolist(),
            "flat_env_indices": np.asarray(
                metadata_fields["flat_env_indices"],
                dtype=np.int64,
            ).tolist(),
            "flat_step_indices": np.asarray(
                metadata_fields["flat_step_indices"],
                dtype=np.int64,
            ).tolist(),
            "flat_objective_episode_indices": np.asarray(
                metadata_fields["flat_objective_episode_indices"],
                dtype=np.int64,
            ).tolist(),
            "flat_objective_episode_positions": np.asarray(
                metadata_fields["flat_objective_episode_positions"],
                dtype=np.int64,
            ).tolist(),
            "flat_objective_global_positions": np.asarray(
                metadata_fields["flat_objective_global_positions"],
                dtype=np.int64,
            ).tolist(),
            "actions": np.asarray(
                rollout_data.actions.detach().to(dtype=torch.float32).cpu().numpy(),
                dtype=np.float32,
            ).tolist(),
            "objective_mask": objective_mask_np.astype(np.int64, copy=False).tolist(),
            "episode_starts": np.asarray(
                rollout_data.episode_starts.detach().to(dtype=torch.int64).cpu().numpy(),
                dtype=np.int64,
            ).tolist(),
            "old_values": np.asarray(
                rollout_data.old_values.detach().to(dtype=torch.float32).cpu().numpy(),
                dtype=np.float32,
            ).tolist(),
            "old_log_prob": np.asarray(
                rollout_data.old_log_prob.detach().to(dtype=torch.float32).cpu().numpy(),
                dtype=np.float32,
            ).tolist(),
            "returns": np.asarray(
                rollout_data.returns.detach().to(dtype=torch.float32).cpu().numpy(),
                dtype=np.float32,
            ).tolist(),
            "pre_normalization_actor_advantages": np.asarray(
                pre_normalization_actor_advantages,
                dtype=np.float32,
            ).tolist(),
            "post_normalization_advantages": np.asarray(
                post_normalization_advantages,
                dtype=np.float32,
            ).tolist(),
            "ratio": np.asarray(ratio_snapshot, dtype=np.float32).tolist(),
            "clipped_objective": np.asarray(clipped_objective_snapshot, dtype=np.float32).tolist(),
            "seq_start_indices": np.asarray(rollout_data.seq_start_indices, dtype=np.int64).tolist(),
            "seq_lengths": np.asarray(rollout_data.seq_lengths, dtype=np.int64).tolist(),
        }
        if matched_target_mask is not None:
            payload["snapshot_target_match_mask"] = np.asarray(
                matched_target_mask,
                dtype=np.int64,
            ).tolist()
        self._write_train_outer_batch_snapshot(payload)

    def _maybe_capture_rollout_collect_snapshot(
        self,
        *,
        rollout_buffer: RecurrentRolloutBuffer,
        n_rollout_steps: int,
        single_eval_pos: int,
    ) -> None:
        if not self._should_capture_rollout_collect_snapshot():
            return
        if not isinstance(rollout_buffer, MaskedRecurrentRolloutBuffer):
            raise ValueError(
                "Rollout collect snapshot requires MaskedRecurrentRolloutBuffer, got "
                f"{type(rollout_buffer)!r}"
            )
        position_metadata = _build_objective_position_metadata(
            rollout_buffer.episode_starts,
            rollout_buffer.objective_masks,
        )
        first_objective_step = int(
            min(
                max(0, int(single_eval_pos)),
                max(0, int(rollout_buffer.buffer_size) - 1),
            )
        )
        payload = {
            "snapshot_entry": "phase3_rollout_collect_snapshot",
            "summary": {
                "buffer_size": int(rollout_buffer.buffer_size),
                "n_envs": int(rollout_buffer.n_envs),
                "n_rollout_steps": int(n_rollout_steps),
                "single_eval_pos": int(single_eval_pos),
                "first_objective_step": int(first_objective_step),
                "actor_objective_mode": str(getattr(self, "_rwkv_actor_objective_mode", "")),
                "actor_baseline_mode": str(getattr(rollout_buffer, "_actor_baseline_mode", "")),
                "actor_gae_space": str(getattr(rollout_buffer, "_actor_gae_space", "")),
                "normalize_advantage": bool(self.normalize_advantage),
                "objective_total": int((np.asarray(rollout_buffer.objective_masks) > 1e-8).sum()),
                "episode_start_total": int((np.asarray(rollout_buffer.episode_starts) > 1e-8).sum()),
            },
            "array_digests": {
                "observations": _array_digest_payload(rollout_buffer.observations),
                "actions": _array_digest_payload(rollout_buffer.actions),
                "rewards": _array_digest_payload(rollout_buffer.rewards),
                "values": _array_digest_payload(rollout_buffer.values),
                "log_probs": _array_digest_payload(rollout_buffer.log_probs),
                "returns": _array_digest_payload(rollout_buffer.returns),
                "advantages": _array_digest_payload(rollout_buffer.advantages),
                "actor_advantages": _array_digest_payload(rollout_buffer.actor_advantages),
                "objective_masks": _array_digest_payload(rollout_buffer.objective_masks),
                "episode_starts": _array_digest_payload(rollout_buffer.episode_starts),
                "rollout_return_means": _array_digest_payload(rollout_buffer.rollout_return_means),
                "rollout_return_stds": _array_digest_payload(rollout_buffer.rollout_return_stds),
                "position_env_indices": _array_digest_payload(position_metadata["env_indices"]),
                "position_step_indices": _array_digest_payload(position_metadata["step_indices"]),
                "position_objective_episode_indices": _array_digest_payload(
                    position_metadata["objective_episode_indices"]
                ),
                "position_objective_episode_positions": _array_digest_payload(
                    position_metadata["objective_episode_positions"]
                ),
                "position_objective_global_positions": _array_digest_payload(
                    position_metadata["objective_global_positions"]
                ),
            },
            "step_slices": {
                "step0": {
                    "rewards": np.asarray(rollout_buffer.rewards[0], dtype=np.float32).tolist(),
                    "values": np.asarray(rollout_buffer.values[0], dtype=np.float32).tolist(),
                    "log_probs": np.asarray(rollout_buffer.log_probs[0], dtype=np.float32).tolist(),
                    "returns": np.asarray(rollout_buffer.returns[0], dtype=np.float32).tolist(),
                    "actor_advantages": np.asarray(rollout_buffer.actor_advantages[0], dtype=np.float32).tolist(),
                    "objective_masks": np.asarray(rollout_buffer.objective_masks[0], dtype=np.int64).tolist(),
                    "episode_starts": np.asarray(rollout_buffer.episode_starts[0], dtype=np.int64).tolist(),
                },
                "first_objective_step": {
                    "step_idx": int(first_objective_step),
                    "rewards": np.asarray(rollout_buffer.rewards[first_objective_step], dtype=np.float32).tolist(),
                    "values": np.asarray(rollout_buffer.values[first_objective_step], dtype=np.float32).tolist(),
                    "log_probs": np.asarray(rollout_buffer.log_probs[first_objective_step], dtype=np.float32).tolist(),
                    "returns": np.asarray(rollout_buffer.returns[first_objective_step], dtype=np.float32).tolist(),
                    "actor_advantages": np.asarray(
                        rollout_buffer.actor_advantages[first_objective_step],
                        dtype=np.float32,
                    ).tolist(),
                    "objective_masks": np.asarray(
                        rollout_buffer.objective_masks[first_objective_step],
                        dtype=np.int64,
                    ).tolist(),
                    "episode_starts": np.asarray(
                        rollout_buffer.episode_starts[first_objective_step],
                        dtype=np.int64,
                    ).tolist(),
                },
            },
        }
        self._write_rollout_collect_snapshot(payload)

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
        log_file = self._progress_log_file()
        if log_file is None or str(log_file).strip() == "":
            return
        try:
            log_path = str(log_file)
            log_dir = os.path.dirname(log_path)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(stamped + "\n")
        except Exception as exc:
            print(f"[ppo-progress-log-warning] failed to write {log_file!r}: {exc}")

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
        del n_seq
        valid_mask_flat_f = valid_mask_flat.to(device=returns_flat.device, dtype=returns_flat.dtype)
        mean, std = _masked_mean_std(returns_flat, valid_mask_flat_f, eps=eps)
        normalized = (returns_flat - mean) / (std + float(eps))
        return normalized * valid_mask_flat_f

    @staticmethod
    def _normalize_flat_sequence_returns_like_reinforce(
        returns_flat: torch.Tensor,
        *,
        seq_start_indices: np.ndarray,
        seq_lengths: np.ndarray,
        valid_mask_flat: Optional[torch.Tensor] = None,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        if returns_flat.ndim != 1:
            raise ValueError("returns_flat must be a flattened 1D tensor.")
        del seq_start_indices, seq_lengths
        valid_mask_flat_t = (
            torch.ones_like(returns_flat, dtype=torch.bool)
            if valid_mask_flat is None
            else valid_mask_flat.to(device=returns_flat.device, dtype=torch.bool)
        )
        if valid_mask_flat_t.shape != returns_flat.shape:
            raise ValueError("valid_mask_flat must match returns_flat when provided.")
        valid_f = valid_mask_flat_t.to(device=returns_flat.device, dtype=returns_flat.dtype)
        mean, std = _masked_mean_std(returns_flat, valid_f, eps=eps)
        normalized = (returns_flat - mean) / (std + float(eps))
        return normalized * valid_f

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
        objective_mask: Optional[torch.Tensor] = None,
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
        if objective_mask is None:
            objective_mask = valid_mask & (rollout_data.objective_masks > 1e-8)
        else:
            objective_mask = objective_mask.to(device=hidden.device, dtype=torch.bool)
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
                raw_returns = _recover_raw_from_value_space(
                    rollout_data.returns.detach(),
                    value_means=rollout_data.rollout_return_means.detach(),
                    value_stds=rollout_data.rollout_return_stds.detach(),
                )
                q_targets = self._normalize_masked_returns_like_reinforce(
                    raw_returns,
                    objective_mask.detach(),
                    n_seq=n_seq,
                    eps=1e-6,
                )
            bardist = model_ref.get_normalized_q_value_bardist()
            q_loss_all = bardist(
                normalized_q_logits.to(dtype=torch.float32),
                q_targets.to(device=normalized_q_logits.device, dtype=torch.float32),
            )
            if q_target_denom is None:
                q_target_denom = objective_mask.sum().clamp_min(1).to(
                    device=normalized_q_logits.device,
                    dtype=torch.float32,
                )
            objective_mask_f = objective_mask.to(device=q_loss_all.device, dtype=q_loss_all.dtype)
            q_loss_num = (q_loss_all * objective_mask_f).sum()
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
        objective_mask: Optional[torch.Tensor] = None,
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
            if objective_mask is None:
                objective_mask = torch.ones_like(q_targets, dtype=torch.bool, device=q_targets.device)
            else:
                objective_mask = objective_mask.to(device=q_targets.device, dtype=torch.bool).reshape(-1)
            bardist = model_ref.get_normalized_q_value_bardist()
            q_loss_all = bardist(
                normalized_q_logits.to(dtype=torch.float32),
                q_targets.to(device=normalized_q_logits.device, dtype=torch.float32),
            )
            if q_target_denom is None:
                q_target_denom = objective_mask.sum().clamp_min(1).to(
                    device=q_loss_all.device,
                    dtype=q_loss_all.dtype,
                )
            objective_mask_f = objective_mask.to(device=q_loss_all.device, dtype=q_loss_all.dtype)
            q_loss_num = (q_loss_all * objective_mask_f).sum()
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

    def _resolve_collect_rollout_rng_channels(
        self,
        env: "EnvironmentPriorPPOBatchVecEnv",
    ) -> tuple[Optional[list[int]], Optional[list[int]]]:
        strict_fixed_env_mode = bool(getattr(self, "_rwkv_strict_fixed_env_mode", False))
        env_seed_spec = getattr(self, "_rwkv_env_rng_seeds", None)
        rollout_seed_spec = getattr(self, "_rwkv_rollout_rng_seeds", None)
        if not strict_fixed_env_mode and env_seed_spec is None and rollout_seed_spec is None:
            self._rwkv_last_collect_rollout_env_rng_seeds = None
            self._rwkv_last_collect_rollout_rollout_rng_seeds = None
            return None, None

        env_rng_seeds = (
            [int(v) for v in env_seed_spec]
            if env_seed_spec is not None
            else [int(v) for v in env._sample_seed_list()]
        )
        env_prior = getattr(self, "_rwkv_env_prior", None)
        if env_prior is None:
            raise RuntimeError("Strict fixed-env rollout mode requires algo._rwkv_env_prior.")
        rollout_rng_seeds = (
            [int(v) for v in rollout_seed_spec]
            if rollout_seed_spec is not None
            else [int(v) for v in env_prior._sample_seed_list(int(env.num_envs))]
        )
        self._rwkv_last_collect_rollout_env_rng_seeds = tuple(env_rng_seeds)
        self._rwkv_last_collect_rollout_rollout_rng_seeds = tuple(rollout_rng_seeds)
        return env_rng_seeds, rollout_rng_seeds

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
        # This strict collect path bypasses VecEnv.step_wait and performs the
        # environment rollout directly through EnvironmentPrior below. The
        # transition generator cached by the mandatory SB3 reset is therefore
        # dead weight during collect_rollouts; keep cheap metadata for masks and
        # observations, but release heavy exact-SCM closures before building the
        # rollout batch.
        if isinstance(getattr(env, "_env", None), dict):
            env._env["transition_generator"] = None
            env._env["policy_generator"] = None
        policy_step_fn = self.policy.make_vectorized_rollout_step_fn()
        if bool(getattr(self, "_rwkv_deterministic_actor_sampling", False)):
            policy_step_fn._policy_actor_sample_fn = (  # type: ignore[attr-defined]
                lambda actor_outputs, noise: actor_outputs["action_mean"]
            )
        env_rng_seeds, rollout_rng_seeds = self._resolve_collect_rollout_rng_channels(env)
        single_eval_pos = int(env_prior._sample_single_eval_pos(int(n_rollout_steps), None))
        sep_state_reset_enabled = bool(getattr(env_prior, "_rwkv_reset_env_state_at_sep_keep_actor_history", False))
        if bool(env_prior._resolve_reward_topology_conditioned_sampling_enabled()):
            h_list, _accepted_env, env_rng_seeds = (
                env_prior._sample_batch_hypers_and_environment_family_coarse_batch_conditioned(
                    int(env.num_envs),
                    device=self.device,
                    rng_seeds=env_rng_seeds,
                    build_policy_generator=False,
                    return_final_env=False,
                )
            )
            _accepted_env = None
        else:
            h_list = env_prior._sample_batch_hypers(int(env.num_envs))
        # PPO optimizes only the suffix after single_eval_pos, so rollout logs
        # should be aligned to that same suffix. We still keep separate explicit
        # full-episode diagnostics under full_* keys for debugging.
        rollout_log_accumulator = _PpoRolloutLogAccumulator(int(env.num_envs), device=self.device)
        episode_returns = np.zeros((env.num_envs,), dtype=np.float32)
        episode_lengths = np.zeros((env.num_envs,), dtype=np.int32)
        dummy_lstm_states = self.policy._dummy_states(env.num_envs)
        dones_np = None
        rollout_step_count = 0
        action_masks_np = None
        next_state_masks_np = None
        rollout_continue = True
        rollout_cpu_marshal_wall_s = 0.0
        rollout_buffer_add_wall_s = 0.0

        def _ppo_step_sink(step_payload):
            nonlocal dones_np
            nonlocal rollout_step_count
            nonlocal action_masks_np
            nonlocal next_state_masks_np
            nonlocal rollout_continue
            nonlocal rollout_cpu_marshal_wall_s
            nonlocal rollout_buffer_add_wall_s

            cpu_marshal_t0 = time.perf_counter()
            obs_np = step_payload["obs"].detach().cpu().numpy()
            actions_np = step_payload["action"].detach().cpu().numpy()
            rewards_np = step_payload["reward"].detach().cpu().numpy()
            starts_np = step_payload["episode_starts"].detach().cpu().numpy().astype(np.float32, copy=False)
            dones_np = step_payload["dones"].detach().cpu().numpy().astype(bool, copy=False)
            rewards_for_buffer_np = rewards_np.astype(np.float32, copy=False)
            if bool(getattr(self, "_rwkv_sb3_reward_normalization_enabled", False)):
                ret_rms = getattr(self, "_rwkv_sb3_reward_normalization_ret_rms", None)
                if not isinstance(ret_rms, RunningMeanStd):
                    ret_rms = RunningMeanStd(shape=())
                    self._rwkv_sb3_reward_normalization_ret_rms = ret_rms
                running_returns = np.asarray(
                    getattr(self, "_rwkv_sb3_reward_normalization_returns", np.zeros_like(rewards_for_buffer_np)),
                    dtype=np.float32,
                )
                rewards_for_buffer_np, running_returns = _sb3_vecnormalize_reward_step(
                    rewards_for_buffer_np,
                    running_returns=running_returns,
                    ret_rms=ret_rms,
                    dones=dones_np,
                    gamma=float(self.gamma),
                    epsilon=float(getattr(self, "_rwkv_sb3_reward_normalization_epsilon", 1e-8)),
                    clip_reward=float(getattr(self, "_rwkv_sb3_reward_normalization_clip", 10.0)),
                )
                self._rwkv_sb3_reward_normalization_returns = running_returns
                self._rwkv_last_sb3_reward_normalization_stats = {
                    "ret_rms_mean": float(np.asarray(ret_rms.mean).reshape(())),
                    "ret_rms_var": float(np.asarray(ret_rms.var).reshape(())),
                    "ret_rms_count": float(ret_rms.count),
                    "reward_scaled_mean": float(np.mean(rewards_for_buffer_np, dtype=np.float64)),
                    "reward_scaled_std": float(np.std(rewards_for_buffer_np, dtype=np.float64)),
                    "reward_scaled_clip_fraction": float(
                        np.mean(
                            np.abs(rewards_for_buffer_np)
                            >= (float(getattr(self, "_rwkv_sb3_reward_normalization_clip", 10.0)) - 1e-6)
                        )
                    ),
                }
            value_logits_t = step_payload["value_logits"]
            log_probs_t = step_payload["log_probs"]
            step_idx = int(step_payload["step"])
            objective_active = int(step_idx) >= int(single_eval_pos)
            values_t = self.policy._value_from_logits(value_logits_t)
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
            rollout_cpu_marshal_wall_s += float(time.perf_counter() - cpu_marshal_t0)

            episode_returns[:] = episode_returns + rewards_np.astype(np.float32, copy=False)
            episode_lengths[:] = episode_lengths + 1
            rollout_log_accumulator.observe_step(
                reward=step_payload["reward"],
                reward_env=step_payload["reward_env"],
                reward_ctrl=step_payload["reward_ctrl"],
                reward_survival=step_payload["reward_survival"],
                reward_terminal_bonus=step_payload["reward_terminal_bonus"],
                objective_mask=objective_active,
                dones=step_payload["dones"],
            )
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

            buffer_add_t0 = time.perf_counter()
            rollout_buffer.add(
                obs_np,
                actions_np,
                rewards_for_buffer_np,
                starts_np,
                values_t,
                log_probs_t,
                lstm_states=dummy_lstm_states,
                next_states=next_states_np,
            )
            rollout_buffer_add_wall_s += float(time.perf_counter() - buffer_add_t0)
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
                env_rng_seeds=env_rng_seeds,
                rollout_rng_seeds=rollout_rng_seeds,
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
        setattr(
            self,
            "_rwkv_last_sep_state_reset_count",
            int(env.num_envs if sep_state_reset_enabled and int(single_eval_pos) < int(n_rollout_steps) else 0),
        )
        final_obs = final_obs.detach()
        final_terminal = final_terminal.detach().to(dtype=torch.bool)

        if rollout_step_count > 0:
            if action_masks_np is None:
                raise RuntimeError("PPO rollout expected cached action masks for buffer backfill.")
            if next_state_masks_np is None:
                raise RuntimeError("PPO rollout expected cached next-state masks for buffer backfill.")
            action_masks_np = np.asarray(action_masks_np, dtype=np.float32).reshape((env.num_envs, -1))
            action_mask_batch = np.zeros((1, env.num_envs, int(rollout_buffer.mask_dims)), dtype=np.float32)
            action_copy_dim = int(min(int(action_masks_np.shape[-1]), int(rollout_buffer.mask_dims)))
            if action_copy_dim > 0:
                action_mask_batch[0, :, :action_copy_dim] = action_masks_np[:, :action_copy_dim]
            rollout_buffer.action_masks[:rollout_step_count] = action_mask_batch

            next_state_masks_np = np.asarray(next_state_masks_np, dtype=np.float32).reshape((env.num_envs, -1))
            next_state_mask_batch = np.zeros((1, env.num_envs, int(rollout_buffer.next_state_dim)), dtype=np.float32)
            next_state_copy_dim = int(min(int(next_state_masks_np.shape[-1]), int(rollout_buffer.next_state_dim)))
            if next_state_copy_dim > 0:
                next_state_mask_batch[0, :, :next_state_copy_dim] = next_state_masks_np[:, :next_state_copy_dim]
            rollout_buffer.next_state_masks[:rollout_step_count] = next_state_mask_batch
            rollout_buffer.objective_masks[:rollout_step_count] = 0.0
            suffix_start = int(min(int(single_eval_pos), int(rollout_step_count)))
            if suffix_start < int(rollout_step_count):
                rollout_buffer.objective_masks[suffix_start:rollout_step_count] = 1.0
            value_path_adapter_impl = str(getattr(self.policy, "value_path_adapter_impl", "none")).strip().lower()
            if value_path_adapter_impl in {"frozen_zscore", "batchnorm_batchstats"}:
                obs_steps_t = torch.as_tensor(
                    rollout_buffer.observations[:rollout_step_count],
                    device=self.device,
                    dtype=torch.float32,
                )
                episode_starts_steps_t = torch.as_tensor(
                    rollout_buffer.episode_starts[:rollout_step_count],
                    device=self.device,
                    dtype=torch.float32,
                )
                objective_masks_steps_t = torch.as_tensor(
                    rollout_buffer.objective_masks[:rollout_step_count],
                    device=self.device,
                    dtype=torch.float32,
                )
                if value_path_adapter_impl == "frozen_zscore":
                    self.policy.fit_value_path_adapter_from_rollout_steps(
                        obs_steps_t,
                        episode_starts_steps_t,
                        objective_masks_steps_t,
                    )
                with torch.no_grad():
                    rollout_value_steps = self.policy.evaluate_rollout_values(
                        obs_steps_t,
                        episode_starts_steps_t,
                    )
                rollout_buffer.values[:rollout_step_count] = (
                    rollout_value_steps.detach().cpu().numpy().reshape((rollout_step_count, env.num_envs))
                )

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
        self._maybe_capture_rollout_collect_snapshot(
            rollout_buffer=rollout_buffer,
            n_rollout_steps=int(n_rollout_steps),
            single_eval_pos=int(single_eval_pos),
        )
        rollout_reward_stats = rollout_log_accumulator.finalize()
        setattr(self, "_rwkv_last_reward_component_stats", rollout_reward_stats)
        rollout_profile = dict(getattr(env_prior, "last_rollout_profile", {}) or {})
        setattr(self, "_rwkv_last_rollout_profile", rollout_profile)
        logger_obj = getattr(self, "_logger", None)
        if logger_obj is not None:
            for key, value in rollout_reward_stats.items():
                if value is not None and math.isfinite(float(value)):
                    logger_obj.record(f"rollout/{key}", float(value))
            sb3_reward_norm_stats = getattr(self, "_rwkv_last_sb3_reward_normalization_stats", None)
            if isinstance(sb3_reward_norm_stats, dict):
                for key, value in sb3_reward_norm_stats.items():
                    if value is not None and math.isfinite(float(value)):
                        logger_obj.record(f"rollout/sb3_reward_normalization_{key}", float(value))
        callback.on_rollout_end()
        rollout_wall_s = float(time.perf_counter() - rollout_wall_t0)
        setattr(self, "_rwkv_last_rollout_wall_time_sec", rollout_wall_s)
        setattr(self, "_rwkv_last_rollout_cpu_marshal_wall_time_sec", float(rollout_cpu_marshal_wall_s))
        setattr(self, "_rwkv_last_rollout_buffer_add_wall_time_sec", float(rollout_buffer_add_wall_s))
        if self._progress_logging_enabled():
            fps = float(int(rollout_step_count) * int(env.num_envs)) / max(1e-9, rollout_wall_s)
            self._emit_progress_log(
                "[ppo-rollout-end] "
                f"steps={int(rollout_step_count)}/{int(n_rollout_steps)} "
                f"envs={int(env.num_envs)} elapsed_s={rollout_wall_s:.1f} fps={fps:.1f}"
            )
            policy_cuda_ms = rollout_profile.get("policy_cuda_ms", None)
            transition_cuda_ms = rollout_profile.get("transition_cuda_ms", None)
            policy_wall_ms = rollout_profile.get("policy_wall_ms", None)
            transition_wall_ms = rollout_profile.get("transition_wall_ms", None)
            if (
                policy_cuda_ms is not None
                or transition_cuda_ms is not None
                or policy_wall_ms is not None
                or transition_wall_ms is not None
            ):
                breakdown_parts = []
                if policy_cuda_ms is not None:
                    breakdown_parts.append(f"policy_cuda_ms={float(policy_cuda_ms):.2f}")
                if transition_cuda_ms is not None:
                    breakdown_parts.append(f"transition_cuda_ms={float(transition_cuda_ms):.2f}")
                rollout_cuda_total_ms = 0.0
                if policy_cuda_ms is not None:
                    rollout_cuda_total_ms += float(policy_cuda_ms)
                if transition_cuda_ms is not None:
                    rollout_cuda_total_ms += float(transition_cuda_ms)
                if rollout_cuda_total_ms > 0.0 and policy_cuda_ms is not None:
                    breakdown_parts.append(
                        f"policy_cuda_share={float(policy_cuda_ms) / max(1e-9, rollout_cuda_total_ms):.3f}"
                    )
                if policy_wall_ms is not None:
                    breakdown_parts.append(f"policy_wall_ms={float(policy_wall_ms):.2f}")
                if transition_wall_ms is not None:
                    breakdown_parts.append(f"transition_wall_ms={float(transition_wall_ms):.2f}")
                rollout_wall_total_ms = 0.0
                if policy_wall_ms is not None:
                    rollout_wall_total_ms += float(policy_wall_ms)
                if transition_wall_ms is not None:
                    rollout_wall_total_ms += float(transition_wall_ms)
                if rollout_wall_total_ms > 0.0 and policy_wall_ms is not None:
                    breakdown_parts.append(
                        f"policy_wall_share={float(policy_wall_ms) / max(1e-9, rollout_wall_total_ms):.3f}"
                    )
                for key in (
                    "transition_setup_wall_ms",
                    "transition_family_build_wall_ms",
                    "transition_generator_build_wall_ms",
                    "transition_gp_shared_build_wall_ms",
                    "transition_group_wall_ms",
                    "transition_group_launch_wall_ms",
                    "transition_group_sync_wall_ms",
                    "token_pack_wall_ms",
                    "action_sample_wall_ms",
                    "transition_prep_wall_ms",
                    "transition_env_pack_wall_ms",
                    "transition_state_update_wall_ms",
                    "transition_noise_wall_ms",
                    "transition_fused_wall_ms",
                    "transition_fused_launch_wall_ms",
                    "transition_fused_group_count",
                    "transition_checkpoint_call_count",
                    "transition_only_build_group_count",
                    "transition_only_skipped_generator_count",
                    "skipped_non_transition_rng_preserve_count",
                    "transition_group_count",
                    "transition_family_group_count",
                    "transition_inner_grouping_structure_enabled",
                    "transition_bucket_max_batch",
                    "transition_work_fill_ratio",
                ):
                    value = rollout_profile.get(key, None)
                    if value is None:
                        continue
                    if isinstance(value, bool):
                        breakdown_parts.append(f"{key}={int(value)}")
                    elif isinstance(value, int):
                        breakdown_parts.append(f"{key}={int(value)}")
                    else:
                        breakdown_parts.append(f"{key}={float(value):.3f}")
                if breakdown_parts:
                    self._emit_progress_log("[ppo-rollout-breakdown] " + " ".join(breakdown_parts))
            self._emit_progress_log(
                "[ppo-rollout-cpu] "
                f"marshal_s={float(rollout_cpu_marshal_wall_s):.3f} "
                f"buffer_add_s={float(rollout_buffer_add_wall_s):.3f} "
                f"marshal_share={float(rollout_cpu_marshal_wall_s) / max(1e-9, rollout_wall_s):.3f} "
                f"buffer_add_share={float(rollout_buffer_add_wall_s) / max(1e-9, rollout_wall_s):.3f}"
            )
        return True

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        has_value_bardist = bool(getattr(self.policy, "has_value_bardist", lambda: False)())
        if has_value_bardist and self.clip_range_vf is not None:
            raise RuntimeError("PPO bar-distribution value head does not support clip_range_vf; set ppo_clip_range_vf=None.")
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)
        value_bardist = self.policy.get_value_bardist() if has_value_bardist else None
        policy_loss_coef = float(getattr(self, "_rwkv_policy_loss_coef", 1.0))
        grad_scaler = getattr(self, "_rwkv_grad_scaler", None)
        autocast_dtype = getattr(self, "_rwkv_autocast_dtype", None)
        exact_value_only_mode = (
            float(policy_loss_coef) == 0.0
            and float(self.ent_coef) == 0.0
            and float(getattr(self, "_rwkv_aux_q_weight", 0.0)) == 0.0
            and float(getattr(self, "_rwkv_aux_flow_weight", 0.0)) == 0.0
        )

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        q_aux_losses = []
        flow_aux_losses = []
        continue_training = True
        train_wall_t0 = time.perf_counter()
        total_transitions = int(self.rollout_buffer.buffer_size * self.rollout_buffer.n_envs)
        total_outer_batches = int(max(1, math.ceil(float(total_transitions) / float(max(1, int(self.batch_size))))))
        probe_max_outer_batches = int(os.environ.get("TICL_PPO_TRAIN_PROBE_MAX_OUTER_BATCHES_EXIT", "0") or "0")
        probe_total_outer_batches = int(max(1, int(self.n_epochs) * int(total_outer_batches)))
        progress_every = self._progress_log_every()
        progress_min_interval_sec = self._progress_log_min_interval_sec()
        last_progress_t = train_wall_t0 - progress_min_interval_sec
        outer_batches_total_processed = 0
        subbatches_total_processed = 0
        loss = torch.zeros((), device=self.device, dtype=torch.float32)
        train_profile_enabled = bool(int(os.environ.get("TICL_PPO_TRAIN_UNIT_PROFILE_ENABLED", "0") or "0"))
        train_profile_sync = bool(int(os.environ.get("TICL_PPO_TRAIN_UNIT_PROFILE_SYNC", "1") or "1"))
        legacy_full_flat_batch = bool(int(os.environ.get("TICL_PPO_TRAIN_LEGACY_FULL_FLAT_BATCH", "0") or "0"))
        legacy_eval_values = bool(int(os.environ.get("TICL_PPO_TRAIN_LEGACY_EVAL_VALUES", "0") or "0"))
        train_profile: dict[str, float] = {}

        def _train_profile_now() -> float:
            if train_profile_enabled and train_profile_sync:
                try:
                    if torch.device(self.device).type == "cuda":
                        torch.cuda.synchronize(self.device)
                except Exception:
                    pass
            return float(time.perf_counter())

        def _train_profile_add(name: str, start_t: float) -> None:
            if not train_profile_enabled:
                return
            elapsed = float(_train_profile_now() - float(start_t))
            train_profile[name] = float(train_profile.get(name, 0.0) + elapsed)

        def _emit_train_profile(tag: str) -> None:
            if not train_profile_enabled or not self._progress_logging_enabled():
                return
            payload = {
                key: round(float(value), 6)
                for key, value in sorted(train_profile.items())
            }
            payload["outer_batches"] = int(outer_batches_total_processed)
            payload["subbatches"] = int(subbatches_total_processed)
            self._emit_progress_log(f"[ppo-train-unit-profile] tag={tag} {json.dumps(payload, sort_keys=True)}")

        if self._progress_logging_enabled():
            self._emit_progress_log(
                "[ppo-train-start] "
                f"epochs={int(self.n_epochs)} outer_batches_per_epoch~={int(total_outer_batches)} "
                f"batch_size={int(self.batch_size)} n_envs={int(self.n_envs)} n_steps={int(self.n_steps)}"
            )

        def _maybe_exit_train_probe() -> None:
            if int(probe_max_outer_batches) <= 0:
                return
            if int(outer_batches_total_processed) < int(probe_max_outer_batches):
                return
            elapsed_s = float(time.perf_counter() - train_wall_t0)
            projected_update_s = elapsed_s * float(probe_total_outer_batches) / max(
                1,
                int(outer_batches_total_processed),
            )
            if self._progress_logging_enabled():
                self._emit_progress_log(
                    "[ppo-train-probe-exit] "
                    f"outer_batches={int(outer_batches_total_processed)}/{int(probe_total_outer_batches)} "
                    f"subbatches={int(subbatches_total_processed)} "
                    f"elapsed_s={elapsed_s:.1f} "
                    f"projected_update_s={projected_update_s:.1f}"
                )
            _emit_train_profile("probe_exit")
            raise SystemExit(0)

        buffer_getter_flat = getattr(self.rollout_buffer, "get_gpu_flat", None)
        buffer_getter_padded = getattr(self.rollout_buffer, "get_gpu", None)
        if not callable(buffer_getter_padded):
            buffer_getter_padded = self.rollout_buffer.get
        should_capture_selector_trace = getattr(
            self,
            "_should_capture_train_outer_batch_selector_trace",
            lambda: False,
        )
        should_capture_snapshot = getattr(
            self,
            "_should_capture_train_outer_batch_snapshot",
            lambda: False,
        )
        needs_debug_flat_metadata = bool(
            should_capture_selector_trace()
            or should_capture_snapshot()
        )
        needs_q_aux_tensors = bool(float(getattr(self, "_rwkv_aux_q_weight", 0.0)) > 0.0)
        needs_flow_aux_tensors = bool(float(getattr(self, "_rwkv_aux_flow_weight", 0.0)) > 0.0)
        include_eval_values = bool(legacy_eval_values)

        for epoch in range(self.n_epochs):
            if getattr(self.policy, "value_path_adapter_impl", "none") == "frozen_zscore":
                obs_steps_t = torch.as_tensor(
                    self.rollout_buffer.observations,
                    device=self.device,
                    dtype=torch.float32,
                )
                episode_starts_steps_t = torch.as_tensor(
                    self.rollout_buffer.episode_starts,
                    device=self.device,
                    dtype=torch.float32,
                )
                objective_masks_steps_t = torch.as_tensor(
                    self.rollout_buffer.objective_masks,
                    device=self.device,
                    dtype=torch.float32,
                )
                if obs_steps_t.ndim == 2:
                    obs_steps_t = obs_steps_t.unsqueeze(1)
                if episode_starts_steps_t.ndim == 1:
                    episode_starts_steps_t = episode_starts_steps_t.unsqueeze(1)
                if objective_masks_steps_t.ndim == 1:
                    objective_masks_steps_t = objective_masks_steps_t.unsqueeze(1)
                self.policy.fit_value_path_adapter_from_rollout_steps(
                    obs_steps_t,
                    episode_starts_steps_t,
                    objective_masks_steps_t,
                )
            approx_kl_divs = []
            outer_batch_idx = 0
            if callable(buffer_getter_flat):
                if isinstance(self.rollout_buffer, MaskedRecurrentRolloutBuffer):
                    rollout_iterator = buffer_getter_flat(
                        self.batch_size,
                        include_old_values=bool(legacy_full_flat_batch or needs_debug_flat_metadata),
                        include_value_affine=bool(legacy_full_flat_batch or needs_q_aux_tensors),
                        include_aux_tensors=bool(legacy_full_flat_batch or needs_flow_aux_tensors),
                        include_position_metadata=bool(legacy_full_flat_batch or needs_debug_flat_metadata),
                    )
                else:
                    rollout_iterator = buffer_getter_flat(self.batch_size)
            else:
                rollout_iterator = buffer_getter_padded(self.batch_size)
            rollout_iterator = iter(rollout_iterator)
            while True:
                profile_t = _train_profile_now() if train_profile_enabled else 0.0
                try:
                    rollout_data = next(rollout_iterator)
                except StopIteration:
                    break
                _train_profile_add("rollout_iterator_next_s", profile_t)
                outer_batch_idx += 1
                outer_batches_total_processed += 1
                if isinstance(rollout_data, MaskedRecurrentFlatBatchSamples):
                    profile_t = _train_profile_now() if train_profile_enabled else 0.0
                    objective_mask = rollout_data.objective_masks > 1e-8
                    objective_total = int(objective_mask.sum().detach().cpu())
                    _train_profile_add("flat_objective_mask_s", profile_t)
                    profile_t = _train_profile_now() if train_profile_enabled else 0.0
                    self._append_train_outer_batch_selector_trace_row_flat(
                        rollout_data=rollout_data,
                        objective_mask=objective_mask,
                        epoch_idx=int(epoch + 1),
                        outer_batch_idx=int(outer_batch_idx),
                        total_outer_batches=int(total_outer_batches),
                    )
                    _train_profile_add("flat_selector_trace_s", profile_t)
                    valid_total = objective_mask.to(
                        device=rollout_data.returns.device,
                        dtype=rollout_data.returns.dtype,
                    ).sum().clamp_min(1.0)
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

                    if objective_total <= 0:
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
                        _maybe_exit_train_probe()
                        continue

                    advantages = None
                    snapshot_target_mask = None
                    snapshot_requested = False
                    if not exact_value_only_mode:
                        profile_t = _train_profile_now() if train_profile_enabled else 0.0
                        advantages = _resolve_actor_advantages(rollout_data)
                        snapshot_target_mask = self._resolve_train_outer_batch_snapshot_match_mask_flat(
                            rollout_data=rollout_data,
                            objective_mask=objective_mask,
                            outer_batch_idx=int(outer_batch_idx),
                        )
                        snapshot_requested = snapshot_target_mask is not None
                        if snapshot_requested:
                            pre_normalization_actor_advantages_snapshot = (
                                advantages.detach().to(dtype=torch.float32).cpu().numpy().copy()
                            )
                        if self.normalize_advantage:
                            advantages = _normalize_advantages_with_mask(advantages, objective_mask, eps=1e-8)
                        _train_profile_add("flat_advantage_norm_s", profile_t)
                        if snapshot_requested:
                            post_normalization_advantages_snapshot = (
                                advantages.detach().to(dtype=torch.float32).cpu().numpy().copy()
                            )
                            ratio_snapshot = np.full(
                                (int(advantages.shape[0]),),
                                np.nan,
                                dtype=np.float32,
                            )
                            clipped_objective_snapshot = np.full(
                                (int(advantages.shape[0]),),
                                np.nan,
                                dtype=np.float32,
                            )
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
                        profile_t = _train_profile_now() if train_profile_enabled else 0.0
                        q_targets_full = rollout_data.normalized_q_targets
                        if q_targets_full is None:
                            if rollout_data.rollout_return_means is None or rollout_data.rollout_return_stds is None:
                                raise RuntimeError("Flat PPO Q aux requires rollout return affine tensors.")
                            raw_returns_full = _recover_raw_from_value_space(
                                rollout_data.returns.detach(),
                                value_means=rollout_data.rollout_return_means.detach(),
                                value_stds=rollout_data.rollout_return_stds.detach(),
                            )
                            q_targets_full = self._normalize_flat_sequence_returns_like_reinforce(
                                raw_returns_full,
                                seq_start_indices=rollout_data.seq_start_indices,
                                seq_lengths=rollout_data.seq_lengths,
                                valid_mask_flat=objective_mask.detach(),
                                eps=1e-6,
                            )
                        _train_profile_add("flat_q_target_prep_s", profile_t)
                    if float(getattr(self, "_rwkv_aux_flow_weight", 0.0)) > 0.0:
                        profile_t = _train_profile_now() if train_profile_enabled else 0.0
                        env_prior = getattr(self, "_rwkv_env_prior", None)
                        if env_prior is None:
                            raise RuntimeError("PPO flow aux requires bound EnvironmentPrior.")
                        if rollout_data.next_states is None or rollout_data.next_state_masks is None:
                            raise RuntimeError("Flat PPO flow aux requires next-state tensors.")
                        flow_xt_full, flow_t_full, flow_dx_full = env_prior._sample_condot_flow_matching_path(
                            rollout_data.next_states.detach()
                        )
                        flow_denom_total = rollout_data.next_state_masks.sum().clamp_min(1.0)
                        _train_profile_add("flat_flow_target_prep_s", profile_t)

                    subbatch_idx = 0
                    for start_seq in range(0, n_seq_total, seq_subbatch_size):
                        profile_t = _train_profile_now() if train_profile_enabled else 0.0
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
                        sub_actions = rollout_data.actions[flat_start:flat_end]
                        sub_returns = rollout_data.returns[flat_start:flat_end]
                        sub_value_target_bucket_idx = getattr(rollout_data, "value_target_bucket_idx", None)
                        if sub_value_target_bucket_idx is not None:
                            sub_value_target_bucket_idx = sub_value_target_bucket_idx[flat_start:flat_end]
                        sub_objective_mask = rollout_data.objective_masks[flat_start:flat_end] > 1e-8
                        sub_objective_f = sub_objective_mask.to(
                            device=rollout_data.returns.device,
                            dtype=rollout_data.returns.dtype,
                        )
                        sub_action_masks = rollout_data.action_masks[flat_start:flat_end]
                        sub_q_targets = None
                        sub_flow_xt = sub_flow_t = sub_flow_dx = None
                        if q_targets_full is not None:
                            sub_q_targets = q_targets_full[flat_start:flat_end]
                        if flow_xt_full is not None:
                            sub_flow_xt = flow_xt_full[flat_start:flat_end]
                            sub_flow_t = flow_t_full[flat_start:flat_end]
                            sub_flow_dx = flow_dx_full[flat_start:flat_end]
                        aux_loss_info = None
                        _train_profile_add("flat_subbatch_slice_s", profile_t)
                        eval_kwargs = {}
                        if (
                            not include_eval_values
                            and bool(getattr(self.policy, "supports_skipping_eval_values", False))
                        ):
                            eval_kwargs["include_values"] = False
                        profile_t = _train_profile_now() if train_profile_enabled else 0.0
                        with _ppo_autocast_context(autocast_dtype=autocast_dtype, device=self.device):
                            eval_outputs = self.policy.evaluate_actions_with_hidden_flat(
                                rollout_data.observations[flat_start:flat_end],
                                sub_actions,
                                seq_lengths=sub_seq_lengths,
                                action_masks=sub_action_masks,
                                include_normalized_q_logits=(sub_q_targets is not None),
                                flow_matching_xt=sub_flow_xt,
                                flow_matching_t=sub_flow_t,
                                **eval_kwargs,
                            )
                            value_logits = eval_outputs["value_logits"]
                            value_loss = self.policy.compute_value_loss_from_logits(
                                value_logits=value_logits,
                                targets=sub_returns.to(device=value_logits.device, dtype=torch.float32),
                                objective_mask=sub_objective_mask,
                                value_target_bucket_idx=sub_value_target_bucket_idx,
                            )
                            sub_value_total = sub_objective_f.to(
                                device=value_loss.device,
                                dtype=value_loss.dtype,
                            ).sum()
                            value_num = value_loss * sub_value_total
                            value_loss_for_objective = value_num / valid_total.to(
                                device=value_loss.device,
                                dtype=value_loss.dtype,
                            )

                            if exact_value_only_mode:
                                policy_num = rollout_data.returns.new_zeros(())
                                entropy_num = rollout_data.returns.new_zeros(())
                                subbatch_loss = self.vf_coef * value_loss_for_objective
                            else:
                                sub_advantages = advantages[flat_start:flat_end]
                                sub_old_log_prob = rollout_data.old_log_prob[flat_start:flat_end]
                                log_prob = eval_outputs["log_prob"]
                                entropy = eval_outputs["entropy"]

                                ratio = torch.exp(log_prob - sub_old_log_prob)
                                policy_loss_1 = sub_advantages * ratio
                                policy_loss_2 = sub_advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                                clipped_objective = torch.min(policy_loss_1, policy_loss_2)
                                if snapshot_requested:
                                    ratio_snapshot[flat_start:flat_end] = (
                                        ratio.detach().to(dtype=torch.float32).cpu().numpy()
                                    )
                                    clipped_objective_snapshot[flat_start:flat_end] = (
                                        clipped_objective.detach().to(dtype=torch.float32).cpu().numpy()
                                    )
                                policy_num = (
                                    clipped_objective
                                    * sub_objective_f.to(
                                        device=clipped_objective.device,
                                        dtype=clipped_objective.dtype,
                                    )
                                ).sum()
                                policy_loss = -(
                                    policy_num
                                    / valid_total.to(device=policy_num.device, dtype=policy_num.dtype)
                                )

                                if entropy is None:
                                    entropy_terms = -log_prob
                                else:
                                    entropy_terms = entropy
                                entropy_num = (
                                    entropy_terms
                                    * sub_objective_f.to(
                                        device=entropy_terms.device,
                                        dtype=entropy_terms.dtype,
                                    )
                                ).sum()
                                entropy_loss = -(
                                    entropy_num
                                    / valid_total.to(device=entropy_num.device, dtype=entropy_num.dtype)
                                )

                                main_loss = (
                                    policy_loss_coef * policy_loss
                                    + self.ent_coef * entropy_loss
                                    + self.vf_coef * value_loss_for_objective
                                )
                                subbatch_loss = main_loss

                            if q_targets_full is not None or flow_xt_full is not None:
                                aux_loss_info = self._compute_aux_losses_flat(
                                    hidden=eval_outputs["hidden"],
                                    actions=eval_outputs["actions"],
                                    seq_lengths=sub_seq_lengths,
                                    normalized_q_logits=eval_outputs.get("normalized_q_logits", None),
                                    next_state_flow=eval_outputs.get("next_state_flow", None),
                                    normalized_q_targets=sub_q_targets,
                                    objective_mask=sub_objective_mask,
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
                        _train_profile_add("flat_forward_loss_s", profile_t)

                        profile_t = _train_profile_now() if train_profile_enabled else 0.0
                        if grad_scaler is None:
                            subbatch_loss.backward()
                        else:
                            grad_scaler.scale(subbatch_loss).backward()
                        _train_profile_add("flat_backward_s", profile_t)
                        loss_total = loss_total + subbatch_loss.detach()

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

                        if not exact_value_only_mode:
                            profile_t = _train_profile_now() if train_profile_enabled else 0.0
                            with torch.no_grad():
                                log_ratio = log_prob - sub_old_log_prob
                                approx_kl_terms = (torch.exp(log_ratio) - 1) - log_ratio
                                approx_kl_num = (
                                    approx_kl_terms
                                    * sub_objective_f.to(device=approx_kl_terms.device, dtype=approx_kl_terms.dtype)
                                ).sum()
                                approx_kl_num_total = approx_kl_num_total + approx_kl_num.detach()
                                clip_num = (
                                    (torch.abs(ratio - 1) > clip_range).to(dtype=ratio.dtype)
                                    * sub_objective_f.to(device=ratio.device, dtype=ratio.dtype)
                                ).sum()
                                clip_num_total = clip_num_total + clip_num.detach()
                            _train_profile_add("flat_metric_kl_clip_s", profile_t)
                    if snapshot_requested:
                        self._maybe_capture_train_outer_batch_snapshot_flat(
                            rollout_data=rollout_data,
                            objective_mask=objective_mask,
                            pre_normalization_actor_advantages=pre_normalization_actor_advantages_snapshot,
                            post_normalization_advantages=post_normalization_advantages_snapshot,
                            ratio_snapshot=ratio_snapshot,
                            clipped_objective_snapshot=clipped_objective_snapshot,
                            clip_range=float(clip_range),
                            outer_batch_idx=int(outer_batch_idx),
                            epoch_idx=int(epoch + 1),
                            matched_target_mask=snapshot_target_mask,
                        )
                else:
                    mask = rollout_data.mask > 1e-8
                    objective_mask = mask & (rollout_data.objective_masks > 1e-8)
                    objective_total = int(objective_mask.sum().detach().cpu())
                    valid_total = objective_mask.sum().clamp_min(1).to(
                        device=rollout_data.returns.device,
                        dtype=rollout_data.returns.dtype,
                    )
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

                    if objective_total <= 0:
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
                        _maybe_exit_train_probe()
                        continue

                    advantages = _resolve_actor_advantages(rollout_data)
                    if self.normalize_advantage:
                        advantages = _normalize_advantages_with_mask(advantages, objective_mask, eps=1e-8)
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
                            raw_returns_full = _recover_raw_from_value_space(
                                rollout_data.returns.detach(),
                                value_means=rollout_data.rollout_return_means.detach(),
                                value_stds=rollout_data.rollout_return_stds.detach(),
                            )
                            q_targets_full = self._normalize_masked_returns_like_reinforce(
                                raw_returns_full,
                                objective_mask.detach(),
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
                        sub_objective_mask = sub_mask & (sub_rollout_data.objective_masks > 1e-8)
                        sub_mask_f = sub_objective_mask.to(
                            device=sub_rollout_data.returns.device,
                            dtype=sub_rollout_data.returns.dtype,
                        )
                        sub_advantages = _slice_padded_sequence_tensor(
                            advantages,
                            n_seq=n_seq_total,
                            start_seq=start_seq,
                            end_seq=end_seq,
                        )
                        sub_value_target_bucket_idx = None
                        if getattr(rollout_data, "value_target_bucket_idx", None) is not None:
                            sub_value_target_bucket_idx = _slice_padded_sequence_tensor(
                                rollout_data.value_target_bucket_idx,
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
                            eval_kwargs = {}
                            if (
                                not include_eval_values
                                and bool(getattr(self.policy, "supports_skipping_eval_values", False))
                            ):
                                eval_kwargs["include_values"] = False
                            eval_outputs = self.policy.evaluate_actions_with_hidden(
                                sub_rollout_data.observations,
                                sub_rollout_data.actions,
                                sub_rollout_data.lstm_states,
                                sub_rollout_data.episode_starts,
                                action_masks=sub_rollout_data.action_masks,
                                **eval_kwargs,
                            )
                            value_logits = eval_outputs["value_logits"]
                            log_prob = eval_outputs["log_prob"]
                            entropy = eval_outputs["entropy"]

                            ratio = torch.exp(log_prob - sub_rollout_data.old_log_prob)
                            policy_loss_1 = sub_advantages * ratio
                            policy_loss_2 = sub_advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                            clipped_objective = torch.min(policy_loss_1, policy_loss_2)
                            policy_num = (clipped_objective * sub_mask_f.to(device=clipped_objective.device, dtype=clipped_objective.dtype)).sum()
                            policy_loss = -(policy_num / valid_total.to(device=policy_num.device, dtype=policy_num.dtype))

                            value_loss = self.policy.compute_value_loss_from_logits(
                                value_logits=value_logits,
                                targets=sub_rollout_data.returns.to(device=value_logits.device, dtype=torch.float32),
                                objective_mask=sub_objective_mask,
                                value_target_bucket_idx=sub_value_target_bucket_idx,
                            )
                            sub_value_total = sub_mask_f.to(
                                device=value_loss.device,
                                dtype=value_loss.dtype,
                            ).sum()
                            value_num = value_loss * sub_value_total
                            value_loss_for_objective = value_num / valid_total.to(
                                device=value_loss.device,
                                dtype=value_loss.dtype,
                            )

                            if entropy is None:
                                entropy_terms = -log_prob
                            else:
                                entropy_terms = entropy
                            entropy_num = (
                                entropy_terms * sub_mask_f.to(device=entropy_terms.device, dtype=entropy_terms.dtype)
                            ).sum()
                            entropy_loss = -(entropy_num / valid_total.to(device=entropy_num.device, dtype=entropy_num.dtype))

                            main_loss = (
                                policy_loss_coef * policy_loss
                                + self.ent_coef * entropy_loss
                                + self.vf_coef * value_loss_for_objective
                            )
                            subbatch_loss = main_loss

                            if q_targets_full is not None or flow_xt_full is not None:
                                aux_loss_info = self._compute_aux_losses(
                                    sub_rollout_data,
                                    eval_outputs=eval_outputs,
                                    n_seq=int(end_seq - start_seq),
                                    normalized_q_targets=sub_q_targets,
                                    objective_mask=sub_objective_mask,
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

                loss = loss_total
                if self.target_kl is not None and float(approx_kl_div.cpu()) > 1.5 * self.target_kl:
                    self.policy.optimizer.zero_grad(set_to_none=True)
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {float(approx_kl_div.cpu()):.2f}")
                    break

                profile_t = _train_profile_now() if train_profile_enabled else 0.0
                if grad_scaler is not None:
                    grad_scaler.unscale_(self.policy.optimizer)
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                if grad_scaler is None:
                    self.policy.optimizer.step()
                else:
                    grad_scaler.step(self.policy.optimizer)
                    grad_scaler.update()
                _train_profile_add("optimizer_step_s", profile_t)
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
                _maybe_exit_train_probe()

            if not continue_training:
                break

        self._finalize_train_outer_batch_selector_trace()
        self._n_updates += self.n_epochs
        update_wall_time_sec = float(time.perf_counter() - train_wall_t0)
        setattr(self, "_rwkv_last_update_wall_time_sec", update_wall_time_sec)
        _emit_train_profile("train_end")
        objective_masks_np = getattr(self.rollout_buffer, "objective_masks", None)
        value_space_values_flat = self.rollout_buffer.values.flatten()
        value_space_targets_flat = self.rollout_buffer.returns.flatten()
        explained_var_value_target = _explained_variance_with_mask(
            value_space_values_flat,
            value_space_targets_flat,
            mask=None,
        )
        rollout_quality: dict[str, float] = {}
        objective_mask_flat = (
            np.asarray(objective_masks_np).reshape(-1) > 1e-8
            if objective_masks_np is not None
            else None
        )
        objective_total = (
            int(objective_mask_flat.sum())
            if objective_mask_flat is not None
            else int(np.asarray(value_space_targets_flat).reshape(-1).size)
        )
        rollout_quality["objective_total"] = float(objective_total)
        rollout_quality.update(
            _masked_array_summary(
                value_space_values_flat,
                mask=objective_masks_np,
                prefix="value_space_value",
            )
        )
        rollout_quality.update(
            _masked_array_summary(
                value_space_targets_flat,
                mask=objective_masks_np,
                prefix="value_space_target",
            )
        )
        if isinstance(self.rollout_buffer, MaskedRecurrentRolloutBuffer):
            rollout_return_stds = np.asarray(self.rollout_buffer.rollout_return_stds, dtype=np.float32)
            rollout_return_means = np.asarray(self.rollout_buffer.rollout_return_means, dtype=np.float32)
            if rollout_return_stds.ndim == 2 and int(rollout_return_stds.shape[0]) > 0:
                rollout_quality.update(
                    _array_quantile_summary(
                        rollout_return_stds[-1],
                        prefix="value_scale_std",
                    )
                )
            if rollout_return_means.ndim == 2 and int(rollout_return_means.shape[0]) > 0:
                rollout_quality.update(
                    _array_quantile_summary(
                        rollout_return_means[-1],
                        prefix="value_scale_mean",
                    )
                )
            rollout_quality.update(
                _sampled_array_summary(
                    self.rollout_buffer.observations,
                    prefix="input_observation",
                )
            )
            rollout_quality.update(
                _sampled_array_summary(
                    self.rollout_buffer.actions,
                    prefix="input_action",
                )
            )
            rollout_quality.update(
                _sampled_array_summary(
                    self.rollout_buffer.rewards,
                    prefix="input_reward",
                )
            )
            rollout_quality.update(
                _sampled_array_summary(
                    self.rollout_buffer.advantages,
                    prefix="input_advantage",
                )
            )
            rollout_quality.update(
                _sampled_array_summary(
                    self.rollout_buffer.actor_advantages,
                    prefix="input_actor_advantage",
                )
            )

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
        self.logger.record("train/explained_variance", explained_var_value_target)
        for metric_key, metric_value in rollout_quality.items():
            self.logger.record(f"train/{metric_key}", metric_value)
        self.logger.record("train/update_wall_time_sec", update_wall_time_sec)
        self.logger.record("train/outer_batches", int(outer_batches_total_processed))
        self.logger.record("train/subbatches", int(subbatches_total_processed))
        self.logger.record("train/policy_weight", float(policy_loss_coef))
        self.logger.record("train/entropy_weight", float(self.ent_coef))
        self.logger.record("train/value_weight", float(self.vf_coef))
        self.logger.record("train/normalized_q_value_weight", float(getattr(self, "_rwkv_aux_q_weight", 0.0)))
        self.logger.record("train/next_state_flow_matching_weight", float(getattr(self, "_rwkv_aux_flow_weight", 0.0)))
        self.logger.record("train/sep_state_reset_enabled", float(bool(getattr(self, "_rwkv_reset_env_state_at_sep", False))))
        self.logger.record(
            "train/sep_state_reset_count",
            float(int(getattr(self, "_rwkv_last_sep_state_reset_count", 0) or 0)),
        )
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
    _require(
        q_weight <= 0.0,
        "normalized_q_value_weight=0.0; PPO path must not include non-official auxiliary Q losses.",
    )
    _require(
        fm_weight <= 0.0,
        "next_state_flow_matching_weight=0.0; PPO path must not include non-official auxiliary flow losses.",
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


_VALIDATION_POLICY_HEAD_STATE_PREFIXES = (
    "action_net.",
    "log_std",
    "value_net.",
    "value_bardist.",
    "value_vendor_head.",
    "value_path_adapter.",
)


def _filter_validation_policy_head_state(
    policy_state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    filtered = {}
    for key, value in policy_state_dict.items():
        key_str = str(key)
        if any(
            key_str == prefix or key_str.startswith(prefix)
            for prefix in _VALIDATION_POLICY_HEAD_STATE_PREFIXES
        ):
            filtered[key_str] = value
    return filtered


def build_validation_recurrent_ppo_policy(
    *,
    model,
    env_cfg: dict[str, Any],
    device,
    num_features: int,
    policy_state_dict: Optional[dict[str, torch.Tensor]] = None,
    strict_native_rollout: bool = False,
):
    target_model = _target_model(model)
    separate_value_backbone = bool(
        policy_state_dict is not None
        and any(str(key).startswith("value_rlpfn_model.") for key in policy_state_dict.keys())
    )
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
        separate_value_backbone=bool(separate_value_backbone),
        net_arch=[],
    )
    if policy_state_dict is not None:
        _restore_saved_recurrent_ppo_policy_state(policy, policy_state_dict)
    _set_force_native_eval_forward_step_on_policy(policy, bool(strict_native_rollout))
    policy.to(device)
    policy.eval()
    return policy


def _set_force_native_eval_forward_step(model_ref, enabled: bool) -> None:
    core = getattr(model_ref, "rwkv_core", None)
    if core is None:
        return
    setattr(core, "force_native_eval_forward_step", bool(enabled))
    if bool(enabled):
        setattr(core, "_official_eval_core", None)
        setattr(core, "_official_eval_core_device", None)


def _set_force_native_eval_forward_step_on_policy(policy, enabled: bool) -> None:
    _set_force_native_eval_forward_step(getattr(policy, "rlpfn_model", None), bool(enabled))
    value_model = getattr(policy, "value_rlpfn_model", None)
    if value_model is not None:
        _set_force_native_eval_forward_step(value_model, bool(enabled))


def _restore_saved_recurrent_ppo_policy_state(
    policy: "OfficialRWKVRecurrentPPOPolicy",
    policy_state_dict: dict[str, torch.Tensor],
) -> None:
    current_state = policy.state_dict()
    filtered_policy_state_dict = {}
    dropped_shape_mismatch = set()
    for key, value in policy_state_dict.items():
        current_value = current_state.get(str(key), None)
        if current_value is not None and tuple(current_value.shape) != tuple(value.shape):
            dropped_shape_mismatch.add(str(key))
            continue
        filtered_policy_state_dict[str(key)] = value
    missing, unexpected = policy.load_state_dict(filtered_policy_state_dict, strict=False)
    legacy_scalar_value_head = "value_bardist.borders" not in policy_state_dict
    mismatched_value_head = any(
        str(key).startswith("value_net.")
        or str(key).startswith("value_bardist.")
        or str(key).startswith("value_vendor_head.")
        or str(key).startswith("value_path_adapter.")
        for key in dropped_shape_mismatch
    )
    missing_filtered = [str(key) for key in missing if not str(key).startswith("rlpfn_model.")]
    if legacy_scalar_value_head or mismatched_value_head:
        missing_filtered = [
            str(key)
            for key in missing_filtered
            if not (
                str(key).startswith("value_net.")
                or str(key).startswith("value_bardist.")
                or str(key).startswith("value_vendor_head.")
                or str(key).startswith("value_path_adapter.")
            )
        ]
    missing_filtered = [str(key) for key in missing_filtered if str(key) not in dropped_shape_mismatch]
    unexpected_filtered = [str(key) for key in unexpected if not str(key).startswith("rlpfn_model.")]
    if missing_filtered or unexpected_filtered:
        raise ValueError(
            "Failed to restore saved PPO validation policy state. "
            f"missing={missing_filtered}, unexpected={unexpected_filtered}"
        )


def _restore_saved_recurrent_ppo_policy_head_state(
    policy: "OfficialRWKVRecurrentPPOPolicy",
    policy_state_dict: dict[str, torch.Tensor],
) -> None:
    filtered_head_state = _filter_validation_policy_head_state(policy_state_dict)
    if not filtered_head_state:
        raise ValueError("Saved PPO validation policy state does not contain action/value head parameters.")
    _restore_saved_recurrent_ppo_policy_state(policy, filtered_head_state)


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
        if int(self.next_state_target_dim) <= 0:
            target = torch.zeros((0,), device=self.device, dtype=state_t.dtype)
            mask = torch.zeros((0,), device=self.device, dtype=state_t.dtype)
            return target, mask
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
        transition_generator = self.env_prior._require_transition_generator(self._env)
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
        transition_out = transition_generator(env_in, generator=self._generator)
        x_next, reward_unit, terminal_signal_next, terminal_bonus_base_next = self.env_prior._unpack_transition_output(
            transition_out
        )
        x_next = x_next.squeeze(0)
        reward_next_raw = float(self._env["reward_scale"]) * reward_unit.reshape(())

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

        sep_state_reset_applied = bool(
            getattr(self.env_prior, "_rwkv_reset_env_state_at_sep_keep_actor_history", False)
        ) and int(self._step_idx + 1) == int(self._single_eval_pos)
        if sep_state_reset_applied:
            state_next = torch.randn(
                (state_dim,),
                device=self.device,
                dtype=torch.float32,
                generator=self._generator,
            ) * float(self._env.get("init_state_std", 0.0))

        self._state_t = state_next
        self._action_t = action_t
        self._reward_t = reward_next.reshape(())
        self._reward_mask_t = reward_mask_next.reshape(())
        self._terminal_t = terminal_next.reshape(())

        self._step_idx += 1
        terminated = bool(float(self._terminal_t.detach().cpu().item()) > 1e-8)
        truncated = bool(self._step_idx >= self.n_steps)
        obs_next = self._build_obs_token()
        next_state_target, next_state_mask = self._build_next_state_target(self._state_t)
        info = {
            "step_idx": int(self._step_idx),
            "single_eval_pos": int(self._single_eval_pos),
            "terminal_flag": float(self._terminal_t.detach().cpu().item()),
            "sep_state_reset": bool(sep_state_reset_applied),
            "next_state_target": next_state_target.detach().cpu().numpy().astype(np.float32, copy=False),
            "next_state_mask": next_state_mask.detach().cpu().numpy().astype(np.float32, copy=False),
            "TimeLimit.truncated": bool(truncated and not terminated),
        }
        return obs_next, float(self._reward_t.detach().cpu().item()), terminated, truncated, info


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
        self._last_next_state_target_np = None
        self._last_next_state_mask_np = None
        self._last_dones_np = None
        self._last_terminated_np = None
        self._last_truncated_np = None
        self._last_terminal_observation_np = None
        self._compact_pack_step_infos = False
        self._step_idx = 0
        self._episode_returns = np.zeros((self.num_envs,), dtype=np.float32)
        self._episode_lengths = np.zeros((self.num_envs,), dtype=np.int32)
        self._direct_collect_reset_stub = False

    def enable_direct_collect_reset_stub(self, enabled: bool = True) -> None:
        self._direct_collect_reset_stub = bool(enabled)

    def release_current_exact_scm_batch(self) -> dict[str, Any]:
        """Release GPU-resident exact-SCM state after a pack has been collected.

        The canonical pack runner updates PPO from the already-materialized
        rollout buffer, so the live transition generator and rollout state are
        dead weight during ``algo.train()``.  Keeping this as an explicit method
        avoids changing rollout semantics while giving the runner a trusted
        place to lower the update-time memory peak.
        """

        had_env = isinstance(getattr(self, "_env", None), dict)
        if had_env:
            self._env["transition_generator"] = None
            self._env["policy_generator"] = None
        self._pending_actions = None
        self._h_list = None
        self._env = None
        self._rollout_generators = None
        self._initial_state_t = None
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
        self._last_next_state_target_np = None
        self._last_next_state_mask_np = None
        self._last_dones_np = None
        self._last_terminated_np = None
        self._last_truncated_np = None
        self._last_terminal_observation_np = None
        clear_artifacts = getattr(self.env_prior, "clear_rollout_artifacts", None)
        if callable(clear_artifacts):
            clear_artifacts()
        return {"released_exact_scm_batch": bool(had_env)}

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

    def _direct_collect_reset_obs_stub(self, *, seeds: Optional[list[int]] = None) -> np.ndarray:
        # MaskedRecurrentPPO.collect_rollouts bypasses VecEnv.step_wait and
        # samples the real exact-SCM batch itself. SB3 still requires a reset()
        # before learning starts, so return a correctly shaped placeholder
        # instead of building a full transition generator that direct collect
        # immediately discards.
        if seeds is None:
            if self._strict_rng_match:
                self._sample_seed_list()
            else:
                self._reset_seeds()
        else:
            self._reset_seeds()
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
        self._single_eval_pos_np = np.zeros((self.num_envs,), dtype=np.int64)
        self._action_masks_np = None
        self._last_next_state_target_np = None
        self._last_next_state_mask_np = None
        self._last_dones_np = None
        self._last_terminated_np = None
        self._last_truncated_np = None
        self._last_terminal_observation_np = None
        self._step_idx = 0
        self._episode_returns.fill(0.0)
        self._episode_lengths.fill(0)
        self.reset_infos = [{"single_eval_pos": 0} for _ in range(self.num_envs)]
        return np.zeros((self.num_envs, self.num_features), dtype=np.float32)

    def _full_reset_batch(self, *, seeds: Optional[list[int]] = None) -> np.ndarray:
        if bool(getattr(self, "_direct_collect_reset_stub", False)):
            return self._direct_collect_reset_obs_stub(seeds=seeds)
        if seeds is None:
            if self._strict_rng_match:
                seeds = self._sample_seed_list()
            else:
                self._reset_seeds()
                seeds = None
        else:
            seeds = [int(s) for s in seeds]
            self._reset_seeds()
        actual_env_seeds = seeds
        if bool(self.env_prior._resolve_reward_topology_conditioned_sampling_enabled()):
            self._h_list, self._env, actual_env_seeds = (
                self.env_prior._sample_batch_hypers_and_environment_family_coarse_batch_conditioned(
                    self.num_envs,
                    device=self.device,
                    rng_seeds=seeds,
                    build_policy_generator=False,
                )
            )
        else:
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
        accepted_env_seeds = self._env.get("reward_topology_conditioned_env_rng_seed")
        if not torch.is_tensor(accepted_env_seeds):
            accepted_env_seeds = self._env.get("reward_state_input_gain_fraction_conditioned_env_rng_seed")
        if not torch.is_tensor(accepted_env_seeds):
            accepted_env_seeds = self._env.get("reward_state_input_gain_fraction_rejection_env_rng_seed")
        if torch.is_tensor(accepted_env_seeds):
            actual_env_seeds = [
                int(v)
                for v in accepted_env_seeds.detach().cpu().reshape(-1).tolist()
            ]
        self._rollout_generators = (
            self.env_prior._make_generators_from_seeds(actual_env_seeds, self.num_envs, self.device)
            if self._strict_rng_match and actual_env_seeds is not None
            else None
        )
        state_dim = int(self._env["state_dim"])
        zero_pad_dim = int(self._env["zero_pad_dim"])
        self._initial_state_t = self.env_prior._resolve_env_initial_state_batch(
            self._env,
            batch_size=self.num_envs,
            state_dim=state_dim,
            device=self.device,
            dtype=torch.float32,
            generators=self._rollout_generators,
        )
        self._state_t = self._initial_state_t.clone()
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
        self._last_next_state_target_np = None
        self._last_next_state_mask_np = None
        self._last_dones_np = None
        self._last_terminated_np = None
        self._last_truncated_np = None
        self._last_terminal_observation_np = None
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

        transition_generator = self.env_prior._require_transition_generator(self._env)
        terminal_signal_next = None
        terminal_bonus_base_next = None
        transition_out = transition_generator(env_in, generators_for_noise=self._rollout_generators)
        x_next, reward_unit, terminal_signal_next, terminal_bonus_base_next = self.env_prior._unpack_transition_output(
            transition_out
        )
        reward_next_raw = self._env["reward_scale"] * reward_unit.reshape(self.num_envs)

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
            state_next, reward_next, terminal_next = self.env_prior._apply_terminal_reset_step(
                state_next=state_next,
                reward_next=reward_next,
                terminal_draw=terminal_draw,
                reset_prob=reset_prob,
                bonus_scale_draw=bonus_scale_draw,
                bonus_scale_min=self._env.get("terminal_bonus_scale_min", 1.0),
                bonus_scale_max=self._env.get("terminal_bonus_scale_max", 2.0),
                bonus_tanh_c=self._env.get("terminal_bonus_tanh_c", 10.0),
                reset_state=self._initial_state_t,
                enabled=self._env.get("terminal_reset_enabled", False),
                terminal_signal=terminal_signal_next,
                terminal_bonus_base=terminal_bonus_base_next,
                terminal_signal_history=self._terminal_signal_history,
                history_index=self._step_idx,
            )
        else:
            terminal_next = torch.zeros((self.num_envs,), device=self.device, dtype=reward_next.dtype)

        reward_return = reward_next
        audit_force_episode_reset_at_sep = bool(
            getattr(self.env_prior, "_audit_force_episode_reset_at_sep", False)
        )
        audit_reset_env_state_at_sep_keep_actor_history = bool(
            getattr(self.env_prior, "_audit_reset_env_state_at_sep_keep_actor_history", False)
        )
        reset_env_state_at_sep_keep_actor_history = bool(
            getattr(self.env_prior, "_rwkv_reset_env_state_at_sep_keep_actor_history", False)
        )
        sep_forced_reset = np.zeros((self.num_envs,), dtype=bool)
        sep_state_only_reset = np.zeros((self.num_envs,), dtype=bool)
        next_step_idx = int(self._step_idx + 1)
        if (
            audit_force_episode_reset_at_sep
            or audit_reset_env_state_at_sep_keep_actor_history
            or reset_env_state_at_sep_keep_actor_history
        ):
            single_eval_pos_np = self._single_eval_pos_np
            if single_eval_pos_np is None:
                single_eval_pos_np = self._single_eval_pos.detach().cpu().numpy().astype(np.int64, copy=True)
                self._single_eval_pos_np = single_eval_pos_np
            sep_mask = (single_eval_pos_np == next_step_idx)
            if audit_force_episode_reset_at_sep:
                sep_forced_reset = sep_mask
            if audit_reset_env_state_at_sep_keep_actor_history or reset_env_state_at_sep_keep_actor_history:
                sep_state_only_reset = sep_mask
        state_store = state_next
        action_store = action_t
        reward_store = reward_next
        reward_mask_store = reward_mask_next
        terminal_store = terminal_next.to(dtype=reward_next.dtype)
        if bool(sep_forced_reset.any()) or bool(sep_state_only_reset.any()):
            reset_mask_t = torch.as_tensor(sep_forced_reset, device=self.device, dtype=torch.bool)
            state_only_reset_mask_t = torch.as_tensor(sep_state_only_reset, device=self.device, dtype=torch.bool)
            reset_state = self._initial_state_t
            state_store = state_next.clone()
            if bool(state_only_reset_mask_t.any()):
                state_store[state_only_reset_mask_t] = reset_state[state_only_reset_mask_t]
            if bool(reset_mask_t.any()):
                reset_action = self.env_prior._stack_randn_with_generators(
                    self._rollout_generators,
                    (self.num_envs, self.action_dim),
                    device=self.device,
                    dtype=torch.float32,
                ) * self._env["init_action_std"][:, None]
                reset_action = reset_action * torch.as_tensor(
                    self._vector_action_masks(),
                    device=self.device,
                    dtype=torch.float32,
                )
                action_store = action_t.clone()
                reward_store = reward_next.clone()
                reward_mask_store = reward_mask_next.clone()
                terminal_store = terminal_store.clone()
                state_store[reset_mask_t] = reset_state[reset_mask_t]
                action_store[reset_mask_t] = reset_action[reset_mask_t]
                reward_store[reset_mask_t] = 0.0
                reward_mask_store[reset_mask_t] = 1.0
                terminal_store[reset_mask_t] = 0.0

        self._state_t = state_store
        self._action_t = action_store
        self._reward_t = reward_store
        self._reward_mask_t = reward_mask_store
        self._terminal_t = terminal_store
        self._step_idx += 1

        obs_next = self._build_obs_batch()
        next_state_target, next_state_mask = self._build_next_state_targets()
        self._last_next_state_target_np = next_state_target
        self._last_next_state_mask_np = next_state_mask
        rewards_np = reward_return.detach().cpu().numpy().astype(np.float32, copy=False)
        terminal_flags_np = self._terminal_t.detach().cpu().numpy().astype(np.float32, copy=False)
        single_eval_pos_np = self._single_eval_pos_np
        if single_eval_pos_np is None:
            single_eval_pos_np = self._single_eval_pos.detach().cpu().numpy().astype(np.int64, copy=True)
            self._single_eval_pos_np = single_eval_pos_np
        terminated = np.logical_or(terminal_flags_np > 1e-8, sep_forced_reset)
        truncated = np.full((self.num_envs,), bool(self._step_idx >= self.n_steps), dtype=bool)
        dones = np.logical_or(terminated, truncated)
        self._last_dones_np = dones
        self._last_terminated_np = terminated
        self._last_truncated_np = truncated
        self._last_terminal_observation_np = None
        self._episode_returns += rewards_np
        self._episode_lengths += 1
        compact_infos = bool(getattr(self, "_compact_pack_step_infos", False))
        if compact_infos:
            infos = [{} for _ in range(self.num_envs)]
        else:
            infos = [
                {
                    "step_idx": int(self._step_idx),
                    "single_eval_pos": int(single_eval_pos_np[env_idx]),
                    "terminal_flag": float(terminal_flags_np[env_idx]),
                    "sep_forced_reset": bool(sep_forced_reset[env_idx]),
                    "sep_state_reset": bool(sep_state_only_reset[env_idx]),
                    "next_state_target": next_state_target[env_idx],
                    "next_state_mask": next_state_mask[env_idx],
                    "TimeLimit.truncated": bool(truncated[env_idx] and not terminated[env_idx]),
                }
                for env_idx in range(self.num_envs)
            ]

        if bool(dones.any()):
            terminal_obs = obs_next.copy()
            self._last_terminal_observation_np = terminal_obs
            for env_idx in range(self.num_envs):
                if not bool(dones[env_idx]):
                    continue
                if not compact_infos:
                    infos[env_idx]["episode"] = {
                        "r": float(self._episode_returns[env_idx]),
                        "l": int(self._episode_lengths[env_idx]),
                    }
                    if bool(truncated[env_idx]):
                        infos[env_idx]["terminal_observation"] = terminal_obs[env_idx].copy()
                self._episode_returns[env_idx] = 0.0
                self._episode_lengths[env_idx] = 0
            defer_final_time_limit_reset = bool(
                getattr(self, "_defer_time_limit_reset_at_rollout_end", False)
                and bool(truncated.all())
                and int(self._step_idx) >= int(self.n_steps)
            )
            if bool(truncated.any()) and not defer_final_time_limit_reset:
                obs_next = self._full_reset_batch()
                self._last_next_state_target_np = next_state_target
                self._last_next_state_mask_np = next_state_mask
                self._last_dones_np = dones
                self._last_terminated_np = terminated
                self._last_truncated_np = truncated
                self._last_terminal_observation_np = terminal_obs

        self._pending_actions = None
        return obs_next, rewards_np.copy(), dones, infos

    def close(self) -> None:
        self.release_current_exact_scm_batch()
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
    supports_skipping_eval_values = True

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        *,
        rlpfn_model,
        num_features: int,
        obs_slot_dim: int,
        separate_value_backbone: bool = False,
        value_head_impl: str = "legacy_bar",
        value_path_adapter_impl: str = "none",
        value_head_mlp_hidden_dim: int = 256,
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
        self.separate_value_backbone = bool(separate_value_backbone)
        self.value_rlpfn_model = copy.deepcopy(rlpfn_model) if self.separate_value_backbone else None
        self.num_features = int(num_features)
        self.obs_slot_dim = int(obs_slot_dim)
        self.value_head_impl = str(value_head_impl).strip().lower()
        if self.value_head_impl not in {"legacy_bar", "scalar_linear", "scalar_mlp", "vendor_official"}:
            raise ValueError(
                f"Unsupported PPO value_head_impl={value_head_impl!r}. "
                "Expected 'legacy_bar', 'scalar_linear', 'scalar_mlp', or 'vendor_official'."
            )
        self.value_head_mlp_hidden_dim = int(value_head_mlp_hidden_dim)
        if self.value_head_mlp_hidden_dim <= 0:
            raise ValueError(
                f"value_head_mlp_hidden_dim must be positive, got {self.value_head_mlp_hidden_dim}."
            )
        self.value_path_adapter_impl = str(value_path_adapter_impl).strip().lower()
        if self.value_path_adapter_impl not in {"none", "batchnorm", "batchnorm_batchstats", "frozen_zscore"}:
            raise ValueError(
                f"Unsupported PPO value_path_adapter_impl={value_path_adapter_impl!r}. "
                "Expected 'none', 'batchnorm', 'batchnorm_batchstats', or 'frozen_zscore'."
            )
        self.value_bardist = _make_ppo_value_bardist() if self.value_head_impl == "legacy_bar" else None
        self.value_vendor_head = (
            TabPFNOfficialRegressorHarness(
                emsize=int(self.mlp_extractor.latent_dim_vf),
                num_buckets=int(_make_ppo_value_bardist().num_bars),
                value_range=float(PPO_VALUE_BARDIST_VALUE_RANGE),
                ignore_nan_targets=True,
            )
            if self.value_head_impl == "vendor_official"
            else None
        )
        value_out_dim = int(self.get_value_bardist().num_bars) if self.has_value_bardist() else 1
        if self.value_path_adapter_impl == "batchnorm":
            self.value_path_adapter = _ValuePathBatchNormAdapter(self.mlp_extractor.latent_dim_vf)
        elif self.value_path_adapter_impl == "batchnorm_batchstats":
            self.value_path_adapter = _ValuePathBatchNormAdapter(
                self.mlp_extractor.latent_dim_vf,
                use_batch_stats_in_eval=True,
            )
        elif self.value_path_adapter_impl == "frozen_zscore":
            self.value_path_adapter = _ValuePathFrozenZScoreAdapter(self.mlp_extractor.latent_dim_vf)
        else:
            self.value_path_adapter = nn.Identity()
        if self.value_head_impl in {"legacy_bar", "scalar_linear"}:
            self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, int(value_out_dim))
        elif self.value_head_impl == "scalar_mlp":
            self.value_net = nn.Sequential(
                nn.Linear(self.mlp_extractor.latent_dim_vf, int(self.value_head_mlp_hidden_dim)),
                nn.Tanh(),
                nn.Linear(int(self.value_head_mlp_hidden_dim), int(value_out_dim)),
            )
        else:
            self.value_net = None
        if self.ortho_init:
            if isinstance(self.value_net, nn.Linear):
                self.value_net.apply(lambda module: self.init_weights(module, gain=1))
            elif isinstance(self.value_net, nn.Sequential):
                linear_layers = [module for module in self.value_net if isinstance(module, nn.Linear)]
                for idx, layer in enumerate(linear_layers):
                    gain = float(np.sqrt(2.0)) if idx < (len(linear_layers) - 1) else 1.0
                    self.init_weights(layer, gain=gain)
        self.lstm_actor = _PlaceholderLSTM()
        self.lstm_critic = None
        self.critic = nn.Identity()
        self.lstm_hidden_state_shape = (1, 1, 1)
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
        self._rollout_kv_cache = None
        self._rollout_cache_batch_size = None
        self._value_rollout_kv_cache = None
        self._value_rollout_cache_batch_size = None

    def has_value_bardist(self) -> bool:
        if bool(self.value_head_impl == "vendor_official"):
            return isinstance(getattr(self, "value_vendor_head", None), nn.Module)
        return isinstance(getattr(self, "value_bardist", None), nn.Module)

    def get_value_bardist(self):
        if not self.has_value_bardist():
            raise RuntimeError("PPO value_bardist is not initialized.")
        if bool(self.value_head_impl == "vendor_official"):
            return self.value_vendor_head.znorm_space_bardist_
        return self.value_bardist

    def has_value_vendor_head(self) -> bool:
        return isinstance(getattr(self, "value_vendor_head", None), TabPFNOfficialRegressorHarness)

    def get_value_head_trainable_parameters(self) -> list[torch.nn.Parameter]:
        params: list[torch.nn.Parameter] = []
        if self.has_value_vendor_head():
            params.extend(list(self.value_vendor_head.parameters()))
        elif isinstance(getattr(self, "value_net", None), nn.Module):
            params.extend(list(self.value_net.parameters()))
        if isinstance(getattr(self, "value_path_adapter", None), nn.Module):
            params.extend(
                [param for param in self.value_path_adapter.parameters() if bool(param.requires_grad)]
            )
        return params

    def _adapt_value_latent(self, latent_vf: torch.Tensor) -> torch.Tensor:
        return self.value_path_adapter(latent_vf)

    @torch.no_grad()
    def fit_value_path_adapter_from_rollout_steps(
        self,
        obs_steps: torch.Tensor,
        episode_starts_steps: torch.Tensor,
        objective_masks_steps: torch.Tensor,
    ) -> bool:
        if self.value_path_adapter_impl != "frozen_zscore":
            return False
        if obs_steps.ndim != 3:
            raise ValueError(f"obs_steps must have shape (T, B, F), got {tuple(obs_steps.shape)}")
        if tuple(episode_starts_steps.shape) != tuple(obs_steps.shape[:2]):
            raise ValueError(
                "episode_starts_steps must match obs_steps (T, B), "
                f"got {tuple(episode_starts_steps.shape)} vs {tuple(obs_steps.shape[:2])}"
            )
        if tuple(objective_masks_steps.shape) != tuple(obs_steps.shape[:2]):
            raise ValueError(
                "objective_masks_steps must match obs_steps (T, B), "
                f"got {tuple(objective_masks_steps.shape)} vs {tuple(obs_steps.shape[:2])}"
            )
        obs_steps = obs_steps.to(device=self.device, dtype=torch.float32)
        episode_starts_steps = episode_starts_steps.to(device=self.device, dtype=torch.float32)
        objective_masks_steps = objective_masks_steps.to(device=self.device, dtype=torch.float32)
        n_seq = int(obs_steps.shape[1])
        obs_flat = obs_steps.swapaxes(0, 1).reshape(-1, int(obs_steps.shape[-1]))
        starts_flat = episode_starts_steps.swapaxes(0, 1).reshape(-1)
        objective_flat = objective_masks_steps.swapaxes(0, 1).reshape(-1) > 1e-8
        hidden = self._value_sequence_hidden(
            obs_flat,
            starts_flat,
            n_seq=n_seq,
        )
        latent_vf = self.mlp_extractor.forward_critic(hidden)
        self.value_path_adapter.fit_from_latent(latent_vf, objective_mask=objective_flat)
        return True

    def _value_logits_from_latent(self, latent_vf: torch.Tensor) -> torch.Tensor:
        latent_vf = self._adapt_value_latent(latent_vf)
        if self.has_value_vendor_head():
            value_logits = self.value_vendor_head.forward_logits(latent_vf)
            expected_last_dim = int(self.value_vendor_head.num_buckets)
        else:
            value_param = next(self.value_net.parameters())
            if latent_vf.dtype != value_param.dtype:
                latent_vf = latent_vf.to(dtype=value_param.dtype)
            value_logits = self.value_net(latent_vf)
            expected_last_dim = int(self.get_value_bardist().num_bars) if self.value_bardist is not None else 1
        if int(value_logits.shape[-1]) != expected_last_dim:
            raise RuntimeError(
                "PPO value_net emitted unexpected value-head width, "
                f"got {tuple(value_logits.shape)} for expected last dim {expected_last_dim}."
            )
        return value_logits

    @staticmethod
    def _coerce_value_affine_tensor(
        value: Optional[torch.Tensor | np.ndarray | float],
        *,
        reference: torch.Tensor,
        default: float,
        name: str,
    ) -> torch.Tensor:
        if value is None:
            return torch.full_like(reference, float(default))
        if torch.is_tensor(value):
            out = value.to(device=reference.device, dtype=reference.dtype)
        else:
            out = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
        while out.ndim < reference.ndim:
            out = out.unsqueeze(-1)
        if tuple(out.shape) == tuple(reference.shape):
            return out
        if int(out.numel()) == 1:
            return out.expand_as(reference)
        raise ValueError(
            f"{name} must broadcast to {tuple(reference.shape)}, got {tuple(out.shape)}"
        )

    def _value_from_logits(
        self,
        value_logits: torch.Tensor,
        *,
        value_means: Optional[torch.Tensor | np.ndarray | float] = None,
        value_stds: Optional[torch.Tensor | np.ndarray | float] = None,
    ) -> torch.Tensor:
        if self.has_value_vendor_head():
            value_mean_scaled = self.value_vendor_head.predict_mean(
                logits=value_logits.to(dtype=torch.float32),
                use_raw_space=False,
            ).unsqueeze(-1)
            means = self._coerce_value_affine_tensor(
                value_means,
                reference=value_mean_scaled,
                default=0.0,
                name="value_means",
            )
            stds = self._coerce_value_affine_tensor(
                value_stds,
                reference=value_mean_scaled,
                default=1.0,
                name="value_stds",
            ).clamp_min(PPO_VALUE_TARGET_NORM_EPS)
            return means + stds * value_mean_scaled
        if self.value_bardist is None:
            value_scaled = value_logits.to(dtype=torch.float32)
            if value_scaled.ndim == 1:
                value_scaled = value_scaled.unsqueeze(-1)
            means = self._coerce_value_affine_tensor(
                value_means,
                reference=value_scaled,
                default=0.0,
                name="value_means",
            )
            stds = self._coerce_value_affine_tensor(
                value_stds,
                reference=value_scaled,
                default=1.0,
                name="value_stds",
            ).clamp_min(PPO_VALUE_TARGET_NORM_EPS)
            return means + stds * value_scaled
        value_mean_scaled = self.value_bardist.mean(value_logits.to(dtype=torch.float32)).unsqueeze(-1)
        means = self._coerce_value_affine_tensor(
            value_means,
            reference=value_mean_scaled,
            default=0.0,
            name="value_means",
        )
        stds = self._coerce_value_affine_tensor(
            value_stds,
            reference=value_mean_scaled,
            default=1.0,
            name="value_stds",
        ).clamp_min(PPO_VALUE_TARGET_NORM_EPS)
        return means + stds * value_mean_scaled

    def compute_value_loss_from_logits(
        self,
        *,
        value_logits: torch.Tensor,
        targets: torch.Tensor,
        objective_mask: torch.Tensor,
        value_target_bucket_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        objective_mask = objective_mask.reshape(-1).to(device=targets.device, dtype=torch.bool)
        if self.has_value_vendor_head():
            if not bool(objective_mask.any().item()):
                return value_logits.new_zeros(())
            logits_flat = value_logits.reshape(-1, int(value_logits.shape[-1]))
            targets_flat = targets.reshape(-1).to(device=value_logits.device, dtype=torch.float32)
            selected_logits = logits_flat[objective_mask.to(device=value_logits.device)].unsqueeze(0)
            selected_targets = targets_flat[objective_mask.to(device=value_logits.device)].unsqueeze(0)
            return self.value_vendor_head.compute_regression_loss(
                logits=selected_logits.to(dtype=torch.float32),
                targets_BQ=selected_targets,
            )
        value_bardist = self.get_value_bardist() if self.has_value_bardist() else None
        if value_bardist is not None and value_target_bucket_idx is not None:
            value_errors = value_bardist.nll_from_bucket_idx(
                value_logits.to(dtype=torch.float32),
                value_target_bucket_idx,
            )
        elif value_bardist is not None:
            value_errors = value_bardist(
                value_logits.to(dtype=torch.float32),
                targets.to(device=value_logits.device, dtype=torch.float32),
            )
        else:
            value_preds = self._value_from_logits(value_logits).flatten()
            value_errors = F.mse_loss(value_preds, targets, reduction="none")
        objective_f = objective_mask.to(device=value_errors.device, dtype=value_errors.dtype)
        value_num = (value_errors.reshape(-1) * objective_f).sum()
        valid_total = objective_f.sum().clamp_min(1.0)
        return value_num / valid_total.to(device=value_num.device, dtype=value_num.dtype)

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

    @contextmanager
    def _value_backbone_context(self):
        if not bool(getattr(self, "separate_value_backbone", False)):
            yield
            return
        actor_model = self.rlpfn_model
        actor_cache = self._rollout_kv_cache
        actor_cache_batch_size = self._rollout_cache_batch_size
        self.rlpfn_model = self.value_rlpfn_model
        self._rollout_kv_cache = self._value_rollout_kv_cache
        self._rollout_cache_batch_size = self._value_rollout_cache_batch_size
        try:
            yield
        finally:
            self._value_rollout_kv_cache = self._rollout_kv_cache
            self._value_rollout_cache_batch_size = self._rollout_cache_batch_size
            self.rlpfn_model = actor_model
            self._rollout_kv_cache = actor_cache
            self._rollout_cache_batch_size = actor_cache_batch_size

    def _value_rollout_hidden(
        self,
        obs: torch.Tensor,
        episode_starts: torch.Tensor,
        *,
        mutate_cache: bool,
    ) -> torch.Tensor:
        if not bool(getattr(self, "separate_value_backbone", False)):
            return self._official_rollout_hidden(obs, episode_starts, mutate_cache=mutate_cache)
        with self._value_backbone_context():
            return self._official_rollout_hidden(obs, episode_starts, mutate_cache=mutate_cache)

    def _value_sequence_hidden(
        self,
        obs: torch.Tensor,
        episode_starts: torch.Tensor,
        *,
        n_seq: int,
    ) -> torch.Tensor:
        if not bool(getattr(self, "separate_value_backbone", False)):
            return self._official_sequence_hidden(obs, episode_starts, n_seq=n_seq)
        with self._value_backbone_context():
            return self._official_sequence_hidden(obs, episode_starts, n_seq=n_seq)

    def _value_sequence_hidden_flat(
        self,
        obs: torch.Tensor,
        *,
        seq_lengths: np.ndarray,
    ) -> torch.Tensor:
        if not bool(getattr(self, "separate_value_backbone", False)):
            return self._official_sequence_hidden_flat(obs, seq_lengths=seq_lengths)
        with self._value_backbone_context():
            return self._official_sequence_hidden_flat(obs, seq_lengths=seq_lengths)

    def _should_share_train_value_hidden(self) -> bool:
        if bool(getattr(self, "separate_value_backbone", False)):
            return False
        if bool(int(os.environ.get("TICL_PPO_TRAIN_LEGACY_SEPARATE_VALUE_HIDDEN", "0") or "0")):
            return False
        return bool(int(os.environ.get("TICL_PPO_TRAIN_SHARE_VALUE_SEQUENCE_HIDDEN", "1") or "1"))

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
        self._value_rollout_kv_cache = None
        self._value_rollout_cache_batch_size = int(n_envs)

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
        if self._rollout_kv_cache is None:
            current_cache = None
        else:
            current_cache = self._rollout_kv_cache if mutate_cache else _clone_tree(self._rollout_kv_cache)
        hidden, next_cache = self._official_rollout_hidden_from_cache(
            obs,
            episode_flags,
            cache=current_cache,
        )
        if mutate_cache:
            self._rollout_kv_cache = next_cache
        return hidden

    def _official_rollout_hidden_from_cache(
        self,
        obs: torch.Tensor,
        episode_starts: torch.Tensor,
        *,
        cache,
    ) -> tuple[torch.Tensor, Any]:
        if obs.ndim != 2:
            raise ValueError(f"PPO rollout expects obs with shape (B, F), got {tuple(obs.shape)}")
        batch_size = int(obs.shape[0])
        episode_flags = episode_starts.reshape(batch_size).to(device=obs.device, dtype=torch.bool)
        y = self._reward_tokens_from_obs(obs).reshape(1, batch_size)
        token = self.rlpfn_model._encode_train_token(obs.unsqueeze(0), y)
        token = self.rlpfn_model._cast_token_for_rwkv_core(token)
        current_cache = cache
        is_official_eval_state = getattr(self.rlpfn_model.rwkv_core, "_is_official_eval_state", None)
        if (
            current_cache is not None
            and callable(is_official_eval_state)
            and bool(is_official_eval_state(current_cache))
        ):
            if batch_size != 1:
                raise RuntimeError(
                    "RWKV PPO official-eval rollout cache only supports batch_size=1 in the current contract."
                )
            if bool(episode_flags.any().item()):
                current_cache = None
        if current_cache is None:
            current_cache = self.rlpfn_model.rwkv_core.init_state(
                batch_size,
                device=token.device,
                dtype=token.dtype,
            )
        elif not (
            callable(is_official_eval_state)
            and bool(is_official_eval_state(current_cache))
        ):
            current_cache = _zero_batch_rows(current_cache, episode_flags)
        hidden, next_cache = self.rlpfn_model.rwkv_core.forward_step(token[0], current_cache)
        return hidden, next_cache

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
        return build_rlpfn_obs_token_batch_from_components(
            obs_t=obs_t,
            action_t=action_t,
            reward_t=reward_t,
            reward_mask_t=reward_mask_t,
            env_info=env_info,
            num_features=int(self.num_features),
        )

    def _rollout_actor_outputs_from_obs_cache(
        self,
        obs_full: torch.Tensor,
        *,
        cache,
        episode_starts: torch.Tensor,
        action_dims: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], Any]:
        batch_size = int(obs_full.shape[0])
        actor_cache = cache
        value_cache = None
        if bool(getattr(self, "separate_value_backbone", False)) and isinstance(cache, dict):
            actor_cache = cache.get("actor")
            value_cache = cache.get("value")
        if actor_cache is None and bool(getattr(self, "separate_value_backbone", False)):
            value_cache = None
        hidden, next_actor_cache = self._official_rollout_hidden_from_cache(
            obs_full,
            episode_starts,
            cache=actor_cache,
        )
        latent_pi = self.mlp_extractor.forward_actor(hidden)
        if bool(getattr(self, "separate_value_backbone", False)):
            with self._value_backbone_context():
                value_hidden, next_value_cache = self._official_rollout_hidden_from_cache(
                    obs_full,
                    episode_starts,
                    cache=value_cache,
                )
            latent_vf = self.mlp_extractor.forward_critic(value_hidden)
            next_cache = {
                "actor": next_actor_cache,
                "value": next_value_cache,
            }
        else:
            latent_vf = self.mlp_extractor.forward_critic(hidden)
            next_cache = next_actor_cache
        distribution = self._get_action_dist_from_latent(latent_pi)
        action_mean_full = distribution.distribution.mean
        action_std_full = distribution.distribution.stddev
        max_action_dim = (
            int(action_dims.max().item())
            if int(action_dims.numel()) > 0
            else int(action_mean_full.shape[-1])
        )
        max_action_dim = int(max(1, min(int(action_mean_full.shape[-1]), max_action_dim)))
        action_mean = action_mean_full[:, :max_action_dim].clone()
        action_std = action_std_full[:, :max_action_dim].clone()
        if batch_size > 0:
            action_positions = torch.arange(
                max_action_dim,
                device=action_mean.device,
                dtype=torch.long,
            ).unsqueeze(0)
            action_valid = action_positions < action_dims.unsqueeze(1)
            action_mean.masked_fill_(~action_valid, 0.0)
            action_std.masked_fill_(~action_valid, 0.0)
        value_logits = self._value_logits_from_latent(latent_vf)
        return (
            {
                "action_mean": action_mean,
                "action_std": action_std,
                "values": self._value_from_logits(value_logits),
                "value_logits": value_logits,
            },
            next_cache,
        )

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
            prebuilt_obs = env_info.get("rollout_obs_row", None)
            if prebuilt_obs is None:
                obs_full = policy._build_obs_from_rollout_step_inputs(
                    obs_t,
                    action_t,
                    reward_t,
                    reward_mask_t,
                    env_info,
                )
            elif torch.is_tensor(prebuilt_obs):
                obs_full = prebuilt_obs.to(device=obs_t.device, dtype=obs_t.dtype)
            else:
                obs_full = torch.as_tensor(prebuilt_obs, device=obs_t.device, dtype=obs_t.dtype)
            if obs_full.ndim != 2 or int(obs_full.shape[0]) != int(obs_t.shape[0]):
                raise ValueError(
                    "rollout_obs_row must have shape (batch, features), "
                    f"got {tuple(obs_full.shape)} for batch={int(obs_t.shape[0])}"
                )
            if int(obs_full.shape[-1]) != int(policy.num_features):
                raise ValueError(
                    f"rollout_obs_row feature dim must equal num_features={int(policy.num_features)}, "
                    f"got {int(obs_full.shape[-1])}"
                )
            action_dims = env_info.get("action_dim_per_sample", None)
            if action_dims is None:
                action_dims = torch.full(
                    (int(obs_full.shape[0]),),
                    int(action_t.shape[-1]),
                    device=obs_full.device,
                    dtype=torch.long,
                )
            elif torch.is_tensor(action_dims):
                action_dims = action_dims.to(device=obs_full.device, dtype=torch.long).reshape(-1)
            else:
                action_dims = torch.as_tensor(action_dims, device=obs_full.device, dtype=torch.long).reshape(-1)
            batch_size = int(obs_full.shape[0])
            rollout_episode_starts = env_info.get("rollout_episode_starts", None)
            if rollout_episode_starts is None:
                rollout_episode_starts = torch.zeros((batch_size,), device=obs_full.device, dtype=torch.float32)
            elif torch.is_tensor(rollout_episode_starts):
                rollout_episode_starts = rollout_episode_starts.to(device=obs_full.device, dtype=torch.float32)
            else:
                rollout_episode_starts = torch.as_tensor(
                    rollout_episode_starts,
                    device=obs_full.device,
                    dtype=torch.float32,
                )
            actor_outputs, next_cache = policy._rollout_actor_outputs_from_obs_cache(
                obs_full,
                cache=cache,
                episode_starts=rollout_episode_starts.reshape(batch_size),
                action_dims=action_dims,
            )
            return actor_outputs, next_cache

        policy_step_fn._policy_actor_sample_fn = (
            lambda actor_outputs, noise: actor_outputs["action_mean"] + (noise * actor_outputs["action_std"])
        )
        return policy_step_fn

    def evaluate_rollout_values(
        self,
        obs_steps: torch.Tensor,
        episode_starts_steps: torch.Tensor,
        *,
        value_means: Optional[torch.Tensor | np.ndarray] = None,
        value_stds: Optional[torch.Tensor | np.ndarray] = None,
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
        def _flatten_value_affine(value, *, name: str):
            if value is None:
                return None
            if torch.is_tensor(value):
                out = value.to(device=obs_steps.device, dtype=torch.float32)
            else:
                out = torch.as_tensor(value, device=obs_steps.device, dtype=torch.float32)
            if out.ndim == 0:
                return out
            if tuple(out.shape) == tuple(obs_steps.shape[:2]):
                return out.swapaxes(0, 1).reshape(-1)
            if out.ndim == 2 and int(out.shape[0]) == 1 and int(out.shape[1]) == int(obs_steps.shape[1]):
                return out.reshape(-1)
            if out.ndim == 1 and int(out.shape[0]) == int(obs_flat.shape[0]):
                return out
            raise ValueError(
                f"{name} must have shape (), (T, B), (1, B), or (T*B,), got {tuple(out.shape)}"
            )
        hidden = self._value_sequence_hidden(obs_flat, starts_flat, n_seq=n_seq)
        latent_vf = self.mlp_extractor.forward_critic(hidden)
        value_logits = self._value_logits_from_latent(latent_vf)
        values = self._value_from_logits(
            value_logits,
            value_means=_flatten_value_affine(value_means, name="value_means"),
            value_stds=_flatten_value_affine(value_stds, name="value_stds"),
        ).reshape(n_seq, int(obs_steps.shape[0])).swapaxes(0, 1)
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
        # Legacy Phase-2 PPO used a distinct value-path RWKV forward even when
        # actor and critic shared parameters.  Keep that graph topology for
        # fixed-env numeric parity; "separate_value_backbone" only controls
        # whether the parameters are copied, not whether the value pass is
        # fused with the actor pass.
        value_hidden = self._value_rollout_hidden(obs, episode_starts, mutate_cache=True)
        latent_vf = self.mlp_extractor.forward_critic(value_hidden)
        value_logits = self._value_logits_from_latent(latent_vf)
        values = self._value_from_logits(value_logits)
        distribution = self._get_action_dist_from_latent(latent_pi)
        action_mask_t = self._coerce_action_masks(action_masks, int(obs.shape[0]), device=obs.device)
        actions = distribution.get_actions(deterministic=deterministic)
        actions = self._masked_action_tensor(actions, action_mask_t)
        log_prob, _ = self._masked_diag_gaussian_stats(distribution, actions, action_mask_t)
        actions = actions.reshape((-1, *self.action_space.shape))
        return actions, values, log_prob, self._dummy_states(obs.shape[0])

    def predict_values(self, obs: torch.Tensor, lstm_states, episode_starts: torch.Tensor) -> torch.Tensor:
        del lstm_states
        hidden = self._value_rollout_hidden(obs, episode_starts, mutate_cache=False)
        latent_vf = self.mlp_extractor.forward_critic(hidden)
        value_logits = self._value_logits_from_latent(latent_vf)
        return self._value_from_logits(value_logits)

    def evaluate_actions_with_hidden(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        lstm_states: RNNStates,
        episode_starts: torch.Tensor,
        action_masks=None,
        include_values: bool = True,
    ):
        n_seq = int(lstm_states.pi[0].shape[1])
        hidden = self._official_sequence_hidden(obs, episode_starts, n_seq=n_seq)
        value_hidden = (
            hidden
            if self._should_share_train_value_hidden()
            else self._value_sequence_hidden(obs, episode_starts, n_seq=n_seq)
        )
        latent_pi = self.mlp_extractor.forward_actor(hidden)
        latent_vf = self.mlp_extractor.forward_critic(value_hidden)
        distribution = self._get_action_dist_from_latent(latent_pi)
        action_mask_t = self._coerce_action_masks(action_masks, int(obs.shape[0]), device=obs.device)
        actions = self._masked_action_tensor(actions, action_mask_t)
        log_prob, entropy = self._masked_diag_gaussian_stats(distribution, actions, action_mask_t)
        value_logits = self._value_logits_from_latent(latent_vf)
        values = self._value_from_logits(value_logits) if bool(include_values) else None
        return {
            "values": values,
            "value_logits": value_logits,
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
        include_values: bool = True,
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
        value_hidden = (
            hidden
            if self._should_share_train_value_hidden()
            else self._value_sequence_hidden_flat(obs, seq_lengths=seq_lengths)
        )
        latent_pi = self.mlp_extractor.forward_actor(hidden)
        latent_vf = self.mlp_extractor.forward_critic(value_hidden)
        distribution = self._get_action_dist_from_latent(latent_pi)
        action_mask_t = self._coerce_action_masks(action_masks, int(obs.shape[0]), device=obs.device)
        actions = self._masked_action_tensor(actions, action_mask_t)
        log_prob, entropy = self._masked_diag_gaussian_stats(distribution, actions, action_mask_t)
        value_logits = self._value_logits_from_latent(latent_vf)
        values = self._value_from_logits(value_logits) if bool(include_values) else None
        outputs = {
            "values": values,
            "value_logits": value_logits,
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
    reset_env_state_at_sep: bool = False,
    separate_value_backbone: bool = False,
    strict_fixed_env_mode: bool = False,
    env_rng_seeds: Optional[int | Sequence[int]] = None,
    rollout_rng_seeds: Optional[int | Sequence[int]] = None,
    deterministic_actor_sampling: bool = False,
    deterministic_batch_plan: bool = False,
    strict_native_rollout: bool = False,
    restore_validation_policy_state: bool = False,
    restore_validation_policy_head_state: bool = False,
    value_head_impl: str = "legacy_bar",
    value_path_adapter_impl: str = "none",
    value_head_mlp_hidden_dim: int = 256,
    space_contract: str = "raw",
    value_target_space: Optional[str] = "raw",
    actor_gae_space: Optional[str] = "raw",
    allow_mixed_space_contract: bool = False,
    sb3_reward_normalization_enabled: bool = False,
    sb3_reward_normalization_clip: float = 10.0,
    sb3_reward_normalization_epsilon: float = 1e-8,
    actor_baseline_mode: str = "learned",
    actor_gae_lambda_override: Optional[float] = None,
    actor_objective_mode: str = "tokenwise",
    actor_objective_runtime_current_suite_name: Optional[str] = None,
    runtime_normalized_q_value_weight_override=None,
    runtime_next_state_flow_matching_weight_override=None,
    verbose: int = 0,
):
    actor_baseline_mode = str(actor_baseline_mode).strip().lower()
    if actor_baseline_mode != "learned":
        raise ValueError(
            "Only official-style PPO actor advantages are enabled: "
            f"actor_baseline_mode must be 'learned', got {actor_baseline_mode!r}."
        )
    if actor_gae_lambda_override is not None:
        raise ValueError(
            "Only official-style PPO actor advantages are enabled: "
            "actor_gae_lambda_override must be None."
        )
    if str(space_contract).strip().lower() != "raw":
        raise ValueError(
            "Only official SB3-style scaling is enabled: space_contract must be 'raw'. "
            "Use VecNormalize-style observation/reward normalization instead."
        )
    if value_target_space is not None and str(value_target_space).strip().lower() != "raw":
        raise ValueError(
            "Only official SB3-style value targets are enabled: value_target_space must be None or 'raw'."
        )
    if actor_gae_space is not None and str(actor_gae_space).strip().lower() != "raw":
        raise ValueError(
            "Only official SB3-style actor advantages are enabled: actor_gae_space must be None or 'raw'."
        )
    if bool(allow_mixed_space_contract):
        raise ValueError("Mixed PPO reward/value/actor spaces are disabled.")
    actor_objective_mode = str(actor_objective_mode).strip().lower()
    if actor_objective_mode != "tokenwise":
        raise ValueError(
            "Only official-style PPO tokenwise actor objectives are enabled: "
            f"actor_objective_mode must be 'tokenwise', got {actor_objective_mode!r}."
        )
    if actor_objective_mode in {
        "tokenwise_scale_env12_mid_episode_extension_block",
        "tokenwise_scale_env12_selector_matched_objective_episode",
        "tokenwise_scale_env12_high_mass_objective_episode_family",
        "tokenwise_scale_pair2_recovered_anchor_first_objective_non_tail_family",
    }:
        if not bool(strict_fixed_env_mode):
            raise ValueError(
                f"{actor_objective_mode} requires strict_fixed_env_mode=True "
                "so the targeted pair2 env columns remain stable."
            )
        if str(actor_objective_runtime_current_suite_name or "").strip().lower() != "pair2":
            raise ValueError(
                f"{actor_objective_mode} is reserved for pair2 runtime scope and "
                "requires actor_objective_runtime_current_suite_name='pair2'."
            )
        required_max_env_index = (
            max(PAIR2_RECOVERED_ANCHOR_FIRST_OBJECTIVE_NON_TAIL_FAMILY_TARGET_ENV_INDICES)
            if actor_objective_mode == "tokenwise_scale_pair2_recovered_anchor_first_objective_non_tail_family"
            else int(ENV12_MID_EPISODE_EXTENSION_BLOCK_TARGET_ENV_INDEX)
        )
        if int(n_envs) <= int(required_max_env_index):
            raise ValueError(
                f"{actor_objective_mode} requires n_envs to include the target pair2 env indices, "
                f"got n_envs={int(n_envs)}."
            )
    space_contract_cfg = _resolve_ppo_space_contract(
        space_contract=space_contract,
        value_target_space=value_target_space,
        actor_gae_space=actor_gae_space,
        normalize_advantage=bool(normalize_advantage),
        actor_baseline_mode=actor_baseline_mode,
        allow_mixed_space_contract=bool(allow_mixed_space_contract),
    )
    value_target_space = str(space_contract_cfg["value_target_space_effective"])
    actor_gae_space = str(space_contract_cfg["actor_gae_space_requested"])
    effective_actor_gae_space = str(space_contract_cfg["actor_gae_space_effective"])
    value_head_impl = str(value_head_impl).strip().lower()
    if value_head_impl not in {"legacy_bar", "scalar_linear", "scalar_mlp", "vendor_official"}:
        raise ValueError(
            f"Unsupported PPO value_head_impl: {value_head_impl}. "
            "Expected 'legacy_bar', 'scalar_linear', 'scalar_mlp', or 'vendor_official'."
        )
    value_head_mlp_hidden_dim = int(value_head_mlp_hidden_dim)
    if value_head_mlp_hidden_dim <= 0:
        raise ValueError(
            f"value_head_mlp_hidden_dim must be positive, got {value_head_mlp_hidden_dim}."
        )
    value_path_adapter_impl = str(value_path_adapter_impl).strip().lower()
    if value_path_adapter_impl not in {"none", "batchnorm", "batchnorm_batchstats", "frozen_zscore"}:
        raise ValueError(
            f"Unsupported PPO value_path_adapter_impl: {value_path_adapter_impl}. "
            "Expected 'none', 'batchnorm', 'batchnorm_batchstats', or 'frozen_zscore'."
        )
    if clip_range_vf is not None:
        raise ValueError("PPO bar-distribution value head requires ppo_clip_range_vf=None.")
    sb3_reward_normalization_clip = float(sb3_reward_normalization_clip)
    if sb3_reward_normalization_clip <= 0:
        raise ValueError(
            f"sb3_reward_normalization_clip must be positive, got {sb3_reward_normalization_clip}."
        )
    sb3_reward_normalization_epsilon = float(sb3_reward_normalization_epsilon)
    if sb3_reward_normalization_epsilon <= 0:
        raise ValueError(
            f"sb3_reward_normalization_epsilon must be positive, got {sb3_reward_normalization_epsilon}."
        )
    effective_env_prior = env_prior
    if (
        runtime_normalized_q_value_weight_override is not None
        or runtime_next_state_flow_matching_weight_override is not None
    ):
        effective_env_prior = copy.deepcopy(env_prior)
        if runtime_normalized_q_value_weight_override is not None:
            effective_env_prior.config["normalized_q_value_weight"] = float(
                runtime_normalized_q_value_weight_override
            )
        if runtime_next_state_flow_matching_weight_override is not None:
            effective_env_prior.config["next_state_flow_matching_weight"] = float(
                runtime_next_state_flow_matching_weight_override
            )
    boundary = _validate_official_rwkv_ppo_boundary(
        model=model,
        env_prior=effective_env_prior,
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

    next_state_storage_dim = (
        int(boundary["next_state_target_dim"])
        if float(boundary["fm_weight"]) > 0.0
        else 0
    )
    vec_env = EnvironmentPriorPPOBatchVecEnv(
        env_prior=effective_env_prior,
        device=device,
        num_features=num_features,
        n_steps=int(n_steps),
        num_envs=int(n_envs),
        action_dim=action_dim,
        obs_slot_dim=obs_slot_dim,
        action_slot_dim=int(boundary["layout"]["action_slot_dim"]),
        next_state_target_dim=int(next_state_storage_dim),
    )
    vec_env.full_next_state_target_dim = int(boundary["next_state_target_dim"])
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
            "separate_value_backbone": bool(separate_value_backbone),
            "value_head_impl": str(value_head_impl),
            "value_path_adapter_impl": str(value_path_adapter_impl),
            "value_head_mlp_hidden_dim": int(value_head_mlp_hidden_dim),
            "net_arch": [],
        },
    )
    if bool(restore_validation_policy_state) and bool(restore_validation_policy_head_state):
        raise ValueError(
            "restore_validation_policy_state and restore_validation_policy_head_state cannot both be enabled."
        )
    if bool(restore_validation_policy_head_state):
        if not (
            bool(strict_fixed_env_mode)
            and bool(deterministic_actor_sampling)
            and bool(deterministic_batch_plan)
            and bool(strict_native_rollout)
        ):
            raise ValueError(
                "restore_validation_policy_head_state is reserved for trusted compare / strict regression mode "
                "and requires strict_fixed_env_mode, deterministic_actor_sampling, deterministic_batch_plan, "
                "and strict_native_rollout to all be True."
            )
        saved_policy_state = getattr(model, "_validation_ppo_policy_state", None)
        if not isinstance(saved_policy_state, dict):
            raise ValueError(
                "restore_validation_policy_head_state=True requires model._validation_ppo_policy_state to be present."
            )
        _restore_saved_recurrent_ppo_policy_head_state(algo.policy, saved_policy_state)
    elif bool(restore_validation_policy_state):
        saved_policy_state = getattr(model, "_validation_ppo_policy_state", None)
        if not isinstance(saved_policy_state, dict):
            raise ValueError(
                "restore_validation_policy_state=True requires model._validation_ppo_policy_state to be present."
            )
        _restore_saved_recurrent_ppo_policy_state(algo.policy, saved_policy_state)

    _set_force_native_eval_forward_step_on_policy(algo.policy, bool(strict_native_rollout))

    callback = _NullCallback()
    setattr(effective_env_prior, "_rwkv_reset_env_state_at_sep_keep_actor_history", bool(reset_env_state_at_sep))
    algo._rwkv_env_prior = effective_env_prior
    algo._rwkv_aux_q_weight = float(boundary["q_weight"])
    algo._rwkv_aux_flow_weight = float(boundary["fm_weight"])
    algo._rwkv_full_next_state_target_dim = int(boundary["next_state_target_dim"])
    algo._rwkv_next_state_storage_dim = int(next_state_storage_dim)
    algo._rwkv_reset_env_state_at_sep = bool(reset_env_state_at_sep)
    algo._rwkv_strict_fixed_env_mode = bool(
        strict_fixed_env_mode or env_rng_seeds is not None or rollout_rng_seeds is not None
    )
    algo._rwkv_env_rng_seeds = _normalize_optional_seed_spec(
        env_rng_seeds,
        n_envs=int(n_envs),
        name="env_rng_seeds",
    )
    algo._rwkv_rollout_rng_seeds = _normalize_optional_seed_spec(
        rollout_rng_seeds,
        n_envs=int(n_envs),
        name="rollout_rng_seeds",
    )
    algo._rwkv_last_collect_rollout_env_rng_seeds = None
    algo._rwkv_last_collect_rollout_rollout_rng_seeds = None
    algo._rwkv_sb3_reward_normalization_enabled = bool(sb3_reward_normalization_enabled)
    algo._rwkv_sb3_reward_normalization_clip = float(sb3_reward_normalization_clip)
    algo._rwkv_sb3_reward_normalization_epsilon = float(sb3_reward_normalization_epsilon)
    algo._rwkv_sb3_reward_normalization_returns = np.zeros((int(n_envs),), dtype=np.float32)
    algo._rwkv_sb3_reward_normalization_ret_rms = RunningMeanStd(shape=())
    algo._rwkv_last_sb3_reward_normalization_stats = None
    algo._rwkv_deterministic_actor_sampling = bool(deterministic_actor_sampling)
    algo._rwkv_deterministic_batch_plan = bool(deterministic_batch_plan)
    algo._rwkv_strict_native_rollout = bool(strict_native_rollout)
    algo._rwkv_restore_validation_policy_head_state = bool(restore_validation_policy_head_state)
    algo._rwkv_rollout_collect_snapshot_path = None
    algo._rwkv_rollout_collect_snapshot_written = False
    algo._rwkv_last_rollout_collect_snapshot_path = None
    algo._rwkv_train_outer_batch_snapshot_path = None
    algo._rwkv_train_outer_batch_snapshot_written = False
    algo._rwkv_last_train_outer_batch_snapshot_path = None
    algo._rwkv_train_outer_batch_selector_trace_path = None
    algo._rwkv_train_outer_batch_selector_trace_rows = []
    algo._rwkv_train_outer_batch_selector_trace_completed = False
    algo._rwkv_last_train_outer_batch_selector_trace_path = None
    algo._rwkv_train_outer_batch_snapshot_target_outer_batch_idx = None
    algo._rwkv_train_outer_batch_snapshot_target_env_index = None
    algo._rwkv_train_outer_batch_snapshot_target_objective_episode_index = None
    algo._rwkv_train_outer_batch_snapshot_target_objective_position_start = None
    algo._rwkv_train_outer_batch_snapshot_target_objective_position_end = None
    if isinstance(algo.rollout_buffer, MaskedRecurrentRolloutBuffer):
        algo.rollout_buffer._enable_q_target_cache = float(boundary["q_weight"]) > 0.0
        algo.rollout_buffer._actor_gae_space = effective_actor_gae_space
        algo.rollout_buffer._actor_baseline_mode = actor_baseline_mode
        algo.rollout_buffer._actor_gae_lambda_override = actor_gae_lambda_override
        algo.rollout_buffer._value_target_space = value_target_space
        algo.rollout_buffer._actor_objective_mode = actor_objective_mode
        algo.rollout_buffer._actor_objective_runtime_current_suite_name = (
            None
            if actor_objective_runtime_current_suite_name is None
            else str(actor_objective_runtime_current_suite_name).strip().lower()
        )
        algo.rollout_buffer.set_deterministic_batch_plan(bool(deterministic_batch_plan))
    algo._rwkv_requested_actor_gae_space = actor_gae_space
    algo._rwkv_actor_gae_space = effective_actor_gae_space
    algo._rwkv_actor_gae_lambda_override = actor_gae_lambda_override
    algo._rwkv_value_target_space = value_target_space
    space_contract = {
        "space_contract": str(space_contract_cfg["space_contract"]),
        "value_target_space_requested": str(space_contract_cfg["value_target_space_requested"]),
        "value_target_space_effective": str(space_contract_cfg["value_target_space_effective"]),
        "actor_gae_space_requested": str(space_contract_cfg["actor_gae_space_requested"]),
        "actor_gae_space_effective": str(space_contract_cfg["actor_gae_space_effective"]),
        "actor_baseline_mode": str(actor_baseline_mode),
        "actor_gae_lambda_override": (
            None if actor_gae_lambda_override is None else float(actor_gae_lambda_override)
        ),
        "normalize_advantage": bool(normalize_advantage),
        "allow_mixed_space_contract": bool(allow_mixed_space_contract),
        "alignment": str(space_contract_cfg["alignment"]),
        "sb3_reward_normalization_enabled": bool(sb3_reward_normalization_enabled),
        "sb3_reward_normalization_clip": (
            float(sb3_reward_normalization_clip) if bool(sb3_reward_normalization_enabled) else None
        ),
        "sb3_reward_normalization_epsilon": (
            float(sb3_reward_normalization_epsilon) if bool(sb3_reward_normalization_enabled) else None
        ),
        "cross_rollout_running_ema_rms_alpha": (
            float(CROSS_ROLLOUT_RUNNING_EMA_RMS_ALPHA)
            if (
                value_target_space == CROSS_ROLLOUT_RUNNING_EMA_RMS_SPACE
                or effective_actor_gae_space == CROSS_ROLLOUT_RUNNING_EMA_RMS_SPACE
            )
            else None
        ),
    }
    algo._rwkv_space_contract = dict(space_contract)
    if isinstance(algo.rollout_buffer, MaskedRecurrentRolloutBuffer):
        algo.rollout_buffer._space_contract = dict(space_contract)
    algo._rwkv_actor_objective_mode = actor_objective_mode
    algo._rwkv_separate_value_backbone = bool(separate_value_backbone)
    return algo, callback, vec_env
