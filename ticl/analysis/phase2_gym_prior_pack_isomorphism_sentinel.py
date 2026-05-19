import argparse
import copy
import ctypes
import gc
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ticl.analysis.critic_free_single_env_audit import (  # noqa: E402
    _apply_boundary_contract_mode_flags,
    _apply_sep_state_reset_flag,
    _seed_all,
)
from ticl.analysis.phase2_legacy_bar_longrun_probe import (  # noqa: E402
    _load_full_frozen_h,
    _sanitize_loaded_frozen_h_for_batch_use,
)
from ticl.analysis.critic_value_fit_probe import _clone_h_repeated  # noqa: E402
from ticl.config_utils import str2bool  # noqa: E402
from ticl.model_builder import get_model  # noqa: E402
from ticl.model_configs import get_model_default_config  # noqa: E402
from ticl.priors.environment_prior import EnvironmentPrior  # noqa: E402
from ticl.rlpfn_maintained_path import resolve_rlpfn_token_layout, validate_rlpfn_maintained_path_config  # noqa: E402
from ticl.sb3_recurrent_ppo import (  # noqa: E402
    EnvironmentPriorPPOBatchVecEnv,
    MaskedRecurrentRolloutBuffer,
    RNNStates,
    _array_digest_payload,
    build_recurrent_ppo,
    build_rlpfn_obs_token_batch_from_components,
)
from stable_baselines3.common.logger import configure as configure_logger  # noqa: E402
from gymnasium import spaces  # noqa: E402


DEFAULT_FROZEN_H_JSON = (
    "/home/chen/RLPFN/artifacts/"
    "phase2_longrun_vendor_official_normalized_fixedenv_gain09011_initstate1_vf01_shared_outer5000/"
    "full_frozen_h_gain09011_init_state_std1.json"
)
DEFAULT_OUTPUT_JSON = (
    "/home/chen/RLPFN/artifacts/phase2_prior_gym_alignment_mainline_0428/"
    "gym_prior_pack_isomorphism_sentinel_0428.json"
)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if torch.is_tensor(value):
        return _json_safe(value.detach().cpu().numpy())
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


def _box_bounds(space) -> tuple[np.ndarray, np.ndarray, int]:
    low = np.asarray(space.low, dtype=np.float32).reshape(-1)
    high = np.asarray(space.high, dtype=np.float32).reshape(-1)
    if low.shape != high.shape or low.ndim != 1 or low.size <= 0:
        raise ValueError(f"Expected flat Box action space, got low={low.shape}, high={high.shape}")
    if not np.all(np.isfinite(low)) or not np.all(np.isfinite(high)):
        raise ValueError("Gym action space bounds must be finite for normalized action bridging.")
    return low, high, int(low.size)


def _normalize_to_box_action(norm_action: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    active = np.asarray(norm_action, dtype=np.float32).reshape(-1)
    return (low + (active + 1.0) * 0.5 * (high - low)).astype(np.float32, copy=False)


def _token_env_info(
    *,
    batch_size: int,
    obs_dim: int,
    action_dim: int,
    obs_slot_dim: int,
    action_slot_dim: int,
    terminal_token_enabled: bool,
    phase_t: np.ndarray,
    terminal_t: np.ndarray,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        "obs_slot_dim": torch.full((batch_size,), int(obs_slot_dim), device=device, dtype=torch.long),
        "action_slot_dim": torch.full((batch_size,), int(action_slot_dim), device=device, dtype=torch.long),
        "action_dim_per_sample": torch.full((batch_size,), int(action_dim), device=device, dtype=torch.long),
        "terminal_reset_enabled": torch.full((batch_size,), bool(terminal_token_enabled), device=device, dtype=torch.bool),
        "obs_dim": torch.full((batch_size,), int(obs_dim), device=device, dtype=torch.long),
        "phase_t": torch.as_tensor(phase_t, device=device, dtype=torch.float32).reshape(batch_size),
        "terminal_t": torch.as_tensor(terminal_t, device=device, dtype=torch.float32).reshape(batch_size),
    }


def _chunked_numeric_stats(array: Any, *, empty_mean: float = 0.0) -> dict[str, Any]:
    arr = np.asarray(array)
    flat = np.ravel(arr, order="K")
    if int(flat.size) <= 0:
        return {
            "shape": [int(v) for v in arr.shape],
            "finite": True,
            "mean": float(empty_mean),
            "std": 0.0,
            "absmax": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
    chunk_elems = 1_000_000
    total = 0
    sum_v = 0.0
    sumsq_v = 0.0
    sumabs_v = 0.0
    min_v = None
    max_v = None
    absmax_v = None
    finite = True
    for start in range(0, int(flat.size), int(chunk_elems)):
        chunk = flat[start : start + int(chunk_elems)]
        chunk64 = chunk.astype(np.float64, copy=False)
        total += int(chunk64.size)
        sum_v += float(np.sum(chunk64, dtype=np.float64))
        sumsq_v += float(np.sum(np.square(chunk64), dtype=np.float64))
        sumabs_v += float(np.sum(np.abs(chunk64), dtype=np.float64))
        chunk_min = float(np.min(chunk64))
        chunk_max = float(np.max(chunk64))
        chunk_absmax = float(np.max(np.abs(chunk64)))
        min_v = chunk_min if min_v is None else float(np.minimum(min_v, chunk_min))
        max_v = chunk_max if max_v is None else float(np.maximum(max_v, chunk_max))
        absmax_v = chunk_absmax if absmax_v is None else float(np.maximum(absmax_v, chunk_absmax))
        if not bool(np.isfinite(chunk64).all()):
            finite = False
    mean_v = float(sum_v / max(1, int(total)))
    var_v = float((sumsq_v / max(1, int(total))) - mean_v * mean_v)
    if math.isfinite(var_v) and var_v < 0.0 and abs(var_v) < 1e-9:
        var_v = 0.0
    std_v = float(math.sqrt(var_v)) if var_v >= 0.0 else float("nan")
    return {
        "shape": [int(v) for v in arr.shape],
        "finite": bool(finite),
        "mean": mean_v,
        "std": std_v,
        "absmax": float(absmax_v if absmax_v is not None else 0.0),
        "absmean": float(sumabs_v / max(1, int(total))),
        "min": float(min_v if min_v is not None else 0.0),
        "max": float(max_v if max_v is not None else 0.0),
    }


def _summarize_pack(pack: dict[str, np.ndarray], *, label: str) -> dict[str, Any]:
    reward_stats = _chunked_numeric_stats(pack["rewards"])
    raw_reward_stats = _chunked_numeric_stats(pack.get("raw_rewards", pack["rewards"]))
    token_stats = _chunked_numeric_stats(pack["tokens"])
    action_stats = _chunked_numeric_stats(pack["actions"])
    episode_starts = np.asarray(pack["episode_starts"], dtype=np.float32)
    dones = np.asarray(pack["dones"], dtype=bool)
    truncated = np.asarray(pack["truncated"], dtype=bool)
    rewards_arr = np.asarray(pack["rewards"], dtype=np.float64)
    raw_rewards_arr = np.asarray(pack.get("raw_rewards", pack["rewards"]), dtype=np.float64)
    reward_returns = rewards_arr.sum(axis=0) if rewards_arr.ndim >= 2 else rewards_arr.reshape(-1)
    raw_reward_returns = raw_rewards_arr.sum(axis=0) if raw_rewards_arr.ndim >= 2 else raw_rewards_arr.reshape(-1)
    reward_return_stats = _chunked_numeric_stats(reward_returns)
    raw_reward_return_stats = _chunked_numeric_stats(raw_reward_returns)
    digests = {
        "tokens": _array_digest_payload(pack["tokens"]),
        "actions": _array_digest_payload(pack["actions"]),
        "rewards": _array_digest_payload(pack["rewards"]),
        "episode_starts": _array_digest_payload(pack["episode_starts"]),
        "dones": _array_digest_payload(pack["dones"]),
        "truncated": _array_digest_payload(pack["truncated"]),
    }
    if "raw_rewards" in pack:
        digests["raw_rewards"] = _array_digest_payload(pack["raw_rewards"])
    return {
        "label": str(label),
        "shapes": {
            key: [int(v) for v in np.asarray(value).shape]
            for key, value in pack.items()
            if isinstance(value, np.ndarray)
        },
        "digests": digests,
        "stats": {
            "reward_mean": float(reward_stats["mean"]),
            "reward_std": float(reward_stats["std"]),
            "reward_return_mean": float(reward_return_stats["mean"]),
            "reward_return_std": float(reward_return_stats["std"]),
            "raw_reward_mean": float(raw_reward_stats["mean"]),
            "raw_reward_std": float(raw_reward_stats["std"]),
            "raw_reward_return_mean": float(raw_reward_return_stats["mean"]),
            "raw_reward_return_std": float(raw_reward_return_stats["std"]),
            "token_absmax": float(token_stats["absmax"]),
            "action_absmean": float(action_stats["absmean"]),
            "episode_start_count": int((episode_starts > 0.5).sum()),
            "done_count": int(dones.sum()),
            "truncated_count": int(truncated.sum()),
        },
    }


def _array_light_payload(array: Any, *, sample_size: int = 8) -> dict[str, Any]:
    arr = np.asarray(array)
    payload: dict[str, Any] = {
        "shape": [int(v) for v in arr.shape],
        "dtype": str(arr.dtype),
        "size": int(arr.size),
    }
    if int(arr.size) <= 0:
        return payload
    sample = arr.reshape(-1)[: int(sample_size)]
    if np.issubdtype(arr.dtype, np.bool_):
        payload["sample"] = sample.astype(np.int64, copy=False).tolist()
    elif np.issubdtype(arr.dtype, np.integer):
        payload["sample"] = sample.astype(np.int64, copy=False).tolist()
    elif np.issubdtype(arr.dtype, np.number):
        payload["sample"] = sample.astype(np.float64, copy=False).tolist()
    else:
        payload["sample"] = sample.tolist()
    return payload


def _summarize_pack_light(pack: dict[str, np.ndarray], *, label: str) -> dict[str, Any]:
    rewards = np.asarray(pack["rewards"], dtype=np.float32)
    raw_rewards = np.asarray(pack.get("raw_rewards", pack["rewards"]), dtype=np.float32)
    episode_starts = np.asarray(pack["episode_starts"], dtype=np.float32)
    dones = np.asarray(pack["dones"], dtype=bool)
    truncated = np.asarray(pack["truncated"], dtype=bool)
    reward_returns = rewards.sum(axis=0) if rewards.ndim >= 2 else rewards.reshape(-1)
    raw_reward_returns = raw_rewards.sum(axis=0) if raw_rewards.ndim >= 2 else raw_rewards.reshape(-1)
    return {
        "label": str(label),
        "summary_mode": "light",
        "shapes": {
            key: [int(v) for v in np.asarray(value).shape]
            for key, value in pack.items()
            if isinstance(value, np.ndarray)
        },
        "samples": {
            key: _array_light_payload(pack[key])
            for key in ("tokens", "actions", "action_masks", "rewards", "episode_starts", "dones", "truncated")
            if key in pack
        },
        "stats": {
            "reward_mean": float(np.mean(rewards, dtype=np.float64)),
            "reward_std": float(np.std(rewards, dtype=np.float64)),
            "reward_return_mean": float(np.mean(reward_returns, dtype=np.float64)),
            "reward_return_std": float(np.std(reward_returns, dtype=np.float64)),
            "raw_reward_mean": float(np.mean(raw_rewards, dtype=np.float64)),
            "raw_reward_std": float(np.std(raw_rewards, dtype=np.float64)),
            "raw_reward_return_mean": float(np.mean(raw_reward_returns, dtype=np.float64)),
            "raw_reward_return_std": float(np.std(raw_reward_returns, dtype=np.float64)),
            "episode_start_count": int((episode_starts > 0.5).sum()),
            "done_count": int(dones.sum()),
            "truncated_count": int(truncated.sum()),
        },
    }


def _finite_stats(array: np.ndarray) -> dict[str, Any]:
    stats = _chunked_numeric_stats(array)
    return {
        "shape": stats["shape"],
        "finite": bool(stats["finite"]),
        "mean": float(stats["mean"]),
        "std": float(stats["std"]),
        "absmax": float(stats["absmax"]),
    }


def _summarize_rollout_buffer(buffer: MaskedRecurrentRolloutBuffer) -> dict[str, Any]:
    n_steps = int(buffer.buffer_size)
    n_envs = int(buffer.n_envs)
    seq_lengths_list: list[int] = []
    starts_bool = np.asarray(buffer.episode_starts, dtype=np.float32) > 0.5
    if starts_bool.ndim == 1:
        starts_bool = starts_bool.reshape((n_steps, n_envs))
    for env_idx in range(n_envs):
        starts = np.nonzero(starts_bool[:, env_idx])[0].astype(np.int64)
        if starts.size <= 0 or int(starts[0]) != 0:
            starts = np.concatenate([np.asarray([0], dtype=np.int64), starts])
        ends = np.concatenate([starts[1:], np.asarray([n_steps], dtype=np.int64)])
        seq_lengths_list.extend([int(max(0, end - start)) for start, end in zip(starts, ends)])
    seq_lengths = np.asarray([v for v in seq_lengths_list if int(v) > 0], dtype=np.int64)
    return {
        "buffer_full": bool(buffer.full),
        "pos": int(buffer.pos),
        "objective_mask_sum": float(np.asarray(buffer.objective_masks).sum()),
        "episode_start_count": int((np.asarray(buffer.episode_starts) > 0.5).sum()),
        "sequence_count": int(seq_lengths.size),
        "sequence_length_min": int(seq_lengths.min()) if seq_lengths.size else 0,
        "sequence_length_max": int(seq_lengths.max()) if seq_lengths.size else 0,
        "sequence_length_mean": float(np.mean(seq_lengths)) if seq_lengths.size else 0.0,
        "returns": _finite_stats(buffer.returns),
        "advantages": _finite_stats(buffer.advantages),
        "actor_advantages": _finite_stats(buffer.actor_advantages),
        "action_masks": _finite_stats(buffer.action_masks),
        "next_states": _finite_stats(buffer.next_states),
        "next_state_masks": _finite_stats(buffer.next_state_masks),
        "digests": {
            "observations": _array_digest_payload(buffer.observations),
            "actions": _array_digest_payload(buffer.actions),
            "rewards": _array_digest_payload(buffer.rewards),
            "episode_starts": _array_digest_payload(buffer.episode_starts),
            "objective_masks": _array_digest_payload(buffer.objective_masks),
            "returns": _array_digest_payload(buffer.returns),
            "advantages": _array_digest_payload(buffer.advantages),
            "action_masks": _array_digest_payload(buffer.action_masks),
            "next_states": _array_digest_payload(buffer.next_states),
            "next_state_masks": _array_digest_payload(buffer.next_state_masks),
        },
    }


def _summarize_rollout_buffer_light(
    buffer: MaskedRecurrentRolloutBuffer,
    *,
    timings: dict[str, float] | None = None,
) -> dict[str, Any]:
    return {
        "buffer_full": bool(buffer.full),
        "pos": int(buffer.pos),
        "buffer_size": int(buffer.buffer_size),
        "n_envs": int(buffer.n_envs),
        "objective_mask_sum": float(np.asarray(buffer.objective_masks, dtype=np.float32).sum()),
        "episode_start_count": int((np.asarray(buffer.episode_starts, dtype=np.float32) > 0.5).sum()),
        "summary_mode": "light",
        "timings": {} if timings is None else {str(k): float(v) for k, v in timings.items()},
    }


def _trim_host_allocator_for_pack_bridge() -> None:
    gc.collect()
    try:
        libc = ctypes.CDLL("libc.so.6")
        malloc_trim = getattr(libc, "malloc_trim", None)
        if malloc_trim is not None:
            malloc_trim(0)
    except Exception:
        pass


def _clear_rollout_buffer_payload_refs(buffer: MaskedRecurrentRolloutBuffer) -> None:
    for name in [
        "observations",
        "actions",
        "rewards",
        "returns",
        "episode_starts",
        "values",
        "log_probs",
        "advantages",
        "hidden_states_pi",
        "cell_states_pi",
        "hidden_states_vf",
        "cell_states_vf",
        "objective_masks",
        "rollout_return_means",
        "rollout_return_stds",
        "rollout_raw_returns",
        "actor_advantages",
        "value_target_bucket_idx",
        "action_masks",
        "next_states",
        "next_state_masks",
        "_global_normalized_q_targets_flat",
        "_global_q_target_seq_ids_flat",
        "_global_q_target_seq_lengths",
        "_global_value_target_bucket_idx_flat",
        "_flat_env_indices",
        "_flat_step_indices",
        "_flat_objective_episode_indices",
        "_flat_objective_episode_positions",
        "_flat_objective_global_positions",
    ]:
        if hasattr(buffer, name):
            setattr(buffer, name, None)
    buffer.seq_start_indices = None
    buffer.seq_end_indices = None
    buffer.generator_ready = False
    buffer.pos = 0
    buffer.full = False


def _fill_existing_rollout_buffer_from_pack(
    buffer: MaskedRecurrentRolloutBuffer,
    pack: dict[str, np.ndarray],
    *,
    num_features: int,
    action_slot_dim: int,
    next_state_target_dim: int,
    single_eval_pos: int,
    values_by_step_env: np.ndarray | None = None,
    log_probs_by_step_env: np.ndarray | None = None,
    last_values_by_env: np.ndarray | None = None,
    compute_returns: bool = True,
    summary_mode: str = "full",
    take_ownership: bool = False,
    allow_missing_next_state_targets: bool = False,
) -> dict[str, Any]:
    fill_t0 = time.perf_counter()
    timings: dict[str, float] = {}
    tokens = np.asarray(pack["tokens"], dtype=np.float32)
    actions = np.asarray(pack["actions"], dtype=np.float32)
    rewards = np.asarray(pack["rewards"], dtype=np.float32)
    episode_starts = np.asarray(pack["episode_starts"], dtype=np.float32)
    n_steps, n_envs, token_dim = tokens.shape
    if int(buffer.buffer_size) != int(n_steps) or int(buffer.n_envs) != int(n_envs):
        raise ValueError(
            "pack shape must match the target rollout buffer, got "
            f"pack={(int(n_steps), int(n_envs))} buffer={(int(buffer.buffer_size), int(buffer.n_envs))}"
        )
    if int(token_dim) != int(num_features):
        raise ValueError(f"pack token dim={token_dim} does not match num_features={num_features}")
    if tuple(actions.shape[:2]) != (int(n_steps), int(n_envs)):
        raise ValueError("actions must share [steps, envs] shape with tokens")
    if int(actions.shape[-1]) != int(action_slot_dim):
        raise ValueError(f"action dim={actions.shape[-1]} does not match action_slot_dim={action_slot_dim}")
    buffer_next_state_dim = int(getattr(buffer, "next_state_dim", int(next_state_target_dim)))
    pack_next_states = pack.get("next_state_targets")
    pack_next_state_masks = pack.get("next_state_masks")
    if buffer_next_state_dim > 0:
        if pack_next_states is None or pack_next_state_masks is None:
            if bool(allow_missing_next_state_targets):
                buffer_next_state_dim = 0
                buffer.next_state_dim = 0
            else:
                raise ValueError("pack omitted next_state targets but rollout buffer requires them")
        elif int(np.asarray(pack_next_states).shape[-1]) != int(buffer_next_state_dim):
            raise ValueError(
                "next_state target dim mismatch: "
                f"{np.asarray(pack_next_states).shape[-1]} vs buffer={int(buffer_next_state_dim)}"
            )
    values_arr = np.zeros((int(n_steps), int(n_envs)), dtype=np.float32)
    log_probs_arr = np.zeros((int(n_steps), int(n_envs)), dtype=np.float32)
    if values_by_step_env is not None:
        values_arr[:] = np.asarray(values_by_step_env, dtype=np.float32).reshape((int(n_steps), int(n_envs)))
    if log_probs_by_step_env is not None:
        log_probs_arr[:] = np.asarray(log_probs_by_step_env, dtype=np.float32).reshape((int(n_steps), int(n_envs)))

    device_obj = torch.device(buffer.device)
    phase_t0 = time.perf_counter()
    objective_masks = np.zeros((int(n_steps), int(n_envs)), dtype=np.float32)
    suffix_start = int(min(int(single_eval_pos), int(n_steps)))
    if suffix_start < int(n_steps):
        objective_masks[suffix_start:] = 1.0
    timings["objective_mask_s"] = float(time.perf_counter() - phase_t0)

    def _assign_step_env_last_dim(dst: np.ndarray, src: np.ndarray, *, name: str) -> None:
        src_arr = np.asarray(src, dtype=np.float32)
        if tuple(src_arr.shape) == tuple(dst.shape):
            dst[:] = src_arr
            return
        if src_arr.ndim != dst.ndim or tuple(src_arr.shape[:-1]) != tuple(dst.shape[:-1]):
            raise ValueError(f"{name} shape mismatch: source={tuple(src_arr.shape)} target={tuple(dst.shape)}")
        dst.fill(0.0)
        copy_dim = int(min(int(src_arr.shape[-1]), int(dst.shape[-1])))
        if copy_dim > 0:
            dst[..., :copy_dim] = src_arr[..., :copy_dim]

    target_tokens_shape = (int(buffer.buffer_size), int(buffer.n_envs), int(num_features))
    if tuple(tokens.shape) != tuple(target_tokens_shape):
        raise ValueError(f"tokens shape mismatch: source={tuple(tokens.shape)} target={tuple(target_tokens_shape)}")
    target_rewards_shape = (int(buffer.buffer_size), int(buffer.n_envs))
    if tuple(rewards.shape) != tuple(target_rewards_shape):
        raise ValueError(f"rewards shape mismatch: source={tuple(rewards.shape)} target={tuple(target_rewards_shape)}")
    target_episode_starts_shape = (int(buffer.buffer_size), int(buffer.n_envs))
    if tuple(episode_starts.shape) != tuple(target_episode_starts_shape):
        raise ValueError(
            f"episode_starts shape mismatch: source={tuple(episode_starts.shape)} "
            f"target={tuple(target_episode_starts_shape)}"
        )
    transferred_keys: list[str] = []
    copied_keys: list[str] = []

    def _take_array(key: str, arr: np.ndarray, *, expected_shape: tuple[int, ...], name: str) -> np.ndarray:
        if tuple(arr.shape) != tuple(expected_shape):
            raise ValueError(f"{name} shape mismatch: source={tuple(arr.shape)} target={tuple(expected_shape)}")
        pack.pop(key, None)
        out = np.ascontiguousarray(arr, dtype=np.float32)
        if out is arr:
            transferred_keys.append(key)
        else:
            copied_keys.append(key)
        return out

    phase_t0 = time.perf_counter()
    if bool(take_ownership):
        _clear_rollout_buffer_payload_refs(buffer)
        _trim_host_allocator_for_pack_bridge()
        timings["reset_s"] = float(time.perf_counter() - phase_t0)
        phase_t0 = time.perf_counter()
        buffer._lazy_step_env_flatten = True
        buffer._lazy_step_env_unflattened = False
        buffer.mask_dims = int(np.prod(buffer.action_space.shape, dtype=np.int64))
        buffer.observations = _take_array("tokens", tokens, expected_shape=(int(n_steps), int(n_envs), int(num_features)), name="tokens")
        if tuple(actions.shape) == (int(n_steps), int(n_envs), int(buffer.action_dim)):
            buffer.actions = _take_array(
                "actions",
                actions,
                expected_shape=(int(n_steps), int(n_envs), int(buffer.action_dim)),
                name="actions",
            )
        else:
            buffer.actions = np.zeros((int(n_steps), int(n_envs), int(buffer.action_dim)), dtype=np.float32)
            _assign_step_env_last_dim(buffer.actions, actions, name="actions")
            pack.pop("actions", None)
            copied_keys.append("actions")
        buffer.rewards = _take_array("rewards", rewards, expected_shape=(int(n_steps), int(n_envs)), name="rewards")
        buffer.episode_starts = _take_array(
            "episode_starts",
            episode_starts,
            expected_shape=(int(n_steps), int(n_envs)),
            name="episode_starts",
        )
        buffer.values = np.ascontiguousarray(values_arr, dtype=np.float32)
        buffer.log_probs = np.ascontiguousarray(log_probs_arr, dtype=np.float32)
        buffer.advantages = np.zeros((int(n_steps), int(n_envs)), dtype=np.float32)
        buffer.returns = np.zeros((int(n_steps), int(n_envs)), dtype=np.float32)
        action_masks_arr = np.asarray(pack["action_masks"], dtype=np.float32)
        if tuple(action_masks_arr.shape) == (int(n_steps), int(n_envs), int(buffer.mask_dims)):
            buffer.action_masks = _take_array(
                "action_masks",
                action_masks_arr,
                expected_shape=(int(n_steps), int(n_envs), int(buffer.mask_dims)),
                name="action_masks",
            )
        else:
            buffer.action_masks = np.ones((int(n_steps), int(n_envs), int(buffer.mask_dims)), dtype=np.float32)
            _assign_step_env_last_dim(buffer.action_masks, action_masks_arr, name="action_masks")
            pack.pop("action_masks", None)
            copied_keys.append("action_masks")
        if buffer_next_state_dim > 0:
            next_states_arr = np.asarray(pack_next_states, dtype=np.float32)
            next_state_masks_arr = np.asarray(pack_next_state_masks, dtype=np.float32)
            buffer.next_states = _take_array(
                "next_state_targets",
                next_states_arr,
                expected_shape=(int(n_steps), int(n_envs), int(buffer_next_state_dim)),
                name="next_states",
            )
            buffer.next_state_masks = _take_array(
                "next_state_masks",
                next_state_masks_arr,
                expected_shape=(int(n_steps), int(n_envs), int(buffer_next_state_dim)),
                name="next_state_masks",
            )
        else:
            buffer.next_states = np.zeros((int(n_steps), int(n_envs), 0), dtype=np.float32)
            buffer.next_state_masks = np.zeros((int(n_steps), int(n_envs), 0), dtype=np.float32)
            pack.pop("next_state_targets", None)
            pack.pop("next_state_masks", None)
        buffer.objective_masks = objective_masks
        buffer.rollout_return_means = np.zeros((int(n_steps), int(n_envs)), dtype=np.float32)
        buffer.rollout_return_stds = np.ones((int(n_steps), int(n_envs)), dtype=np.float32)
        buffer.rollout_raw_returns = np.zeros((int(n_steps), int(n_envs)), dtype=np.float32)
        buffer.actor_advantages = np.zeros((int(n_steps), int(n_envs)), dtype=np.float32)
        buffer.value_target_bucket_idx = np.full((int(n_steps), int(n_envs)), -1, dtype=np.int64)
        buffer.hidden_states_pi = np.zeros(buffer.hidden_state_shape, dtype=np.float32)
        buffer.cell_states_pi = np.zeros(buffer.hidden_state_shape, dtype=np.float32)
        buffer.hidden_states_vf = np.zeros(buffer.hidden_state_shape, dtype=np.float32)
        buffer.cell_states_vf = np.zeros(buffer.hidden_state_shape, dtype=np.float32)
    else:
        phase_t0 = time.perf_counter()
        buffer.reset()
        buffer._lazy_step_env_flatten = False
        buffer._lazy_step_env_unflattened = False
        timings["reset_s"] = float(time.perf_counter() - phase_t0)
        if tuple(tokens.shape) != tuple(buffer.observations.shape):
            raise ValueError(f"tokens shape mismatch: source={tuple(tokens.shape)} target={tuple(buffer.observations.shape)}")
        if tuple(rewards.shape) != tuple(buffer.rewards.shape):
            raise ValueError(f"rewards shape mismatch: source={tuple(rewards.shape)} target={tuple(buffer.rewards.shape)}")
        if tuple(episode_starts.shape) != tuple(buffer.episode_starts.shape):
            raise ValueError(
                f"episode_starts shape mismatch: source={tuple(episode_starts.shape)} "
                f"target={tuple(buffer.episode_starts.shape)}"
            )
        phase_t0 = time.perf_counter()
        buffer.observations[:] = tokens
        _assign_step_env_last_dim(buffer.actions, actions, name="actions")
        buffer.rewards[:] = rewards
        buffer.episode_starts[:] = episode_starts
        buffer.values[:] = values_arr
        buffer.log_probs[:] = log_probs_arr
        _assign_step_env_last_dim(buffer.action_masks, np.asarray(pack["action_masks"], dtype=np.float32), name="action_masks")
        if buffer_next_state_dim > 0:
            _assign_step_env_last_dim(buffer.next_states, np.asarray(pack_next_states, dtype=np.float32), name="next_states")
            _assign_step_env_last_dim(
                buffer.next_state_masks,
                np.asarray(pack_next_state_masks, dtype=np.float32),
                name="next_state_masks",
            )
        elif bool(allow_missing_next_state_targets):
            buffer.next_states = np.zeros((int(n_steps), int(n_envs), 0), dtype=np.float32)
            buffer.next_state_masks = np.zeros((int(n_steps), int(n_envs), 0), dtype=np.float32)
        buffer.objective_masks[:] = objective_masks
    buffer.pos = int(buffer.buffer_size)
    buffer.full = True
    buffer.generator_ready = False
    buffer.seq_start_indices = None
    buffer.seq_end_indices = None
    timings["array_assign_s"] = float(time.perf_counter() - phase_t0)
    if bool(compute_returns):
        phase_t0 = time.perf_counter()
        dones = np.asarray(pack["dones"][-1], dtype=bool)
        if last_values_by_env is None:
            last_values_arr = np.zeros((int(n_envs),), dtype=np.float32)
        else:
            last_values_arr = np.asarray(last_values_by_env, dtype=np.float32).reshape((int(n_envs),))
        buffer.compute_returns_and_advantage(
            torch.as_tensor(last_values_arr, device=device_obj, dtype=torch.float32),
            dones,
        )
        timings["compute_returns_s"] = float(time.perf_counter() - phase_t0)
    summary_mode_norm = str(summary_mode).strip().lower()
    timings["pre_summary_total_s"] = float(time.perf_counter() - fill_t0)
    if summary_mode_norm in {"light", "minimal", "none", "off", "false", "0"}:
        timings["total_s"] = float(time.perf_counter() - fill_t0)
        summary = _summarize_rollout_buffer_light(buffer, timings=timings)
        summary["take_ownership"] = bool(take_ownership)
        summary["transferred_keys"] = list(transferred_keys)
        summary["copied_keys"] = list(copied_keys)
        return summary
    phase_t0 = time.perf_counter()
    summary = _summarize_rollout_buffer(buffer)
    timings["summary_s"] = float(time.perf_counter() - phase_t0)
    timings["total_s"] = float(time.perf_counter() - fill_t0)
    summary["summary_mode"] = "full"
    summary["timings"] = timings
    summary["take_ownership"] = bool(take_ownership)
    summary["transferred_keys"] = list(transferred_keys)
    summary["copied_keys"] = list(copied_keys)
    return summary


def _materialize_pack_rollout_buffer(
    pack: dict[str, np.ndarray],
    *,
    num_features: int,
    action_slot_dim: int,
    next_state_target_dim: int,
    single_eval_pos: int,
    gamma: float,
    gae_lambda: float,
    value_target_space: str,
    actor_gae_space: str,
) -> tuple[MaskedRecurrentRolloutBuffer, dict[str, Any]]:
    tokens = np.asarray(pack["tokens"], dtype=np.float32)
    actions = np.asarray(pack["actions"], dtype=np.float32)
    rewards = np.asarray(pack["rewards"], dtype=np.float32)
    episode_starts = np.asarray(pack["episode_starts"], dtype=np.float32)
    n_steps, n_envs, token_dim = tokens.shape
    if int(token_dim) != int(num_features):
        raise ValueError(f"pack token dim={token_dim} does not match num_features={num_features}")
    if tuple(actions.shape[:2]) != (int(n_steps), int(n_envs)):
        raise ValueError("actions must share [steps, envs] shape with tokens")
    if tuple(rewards.shape) != (int(n_steps), int(n_envs)):
        raise ValueError("rewards must share [steps, envs] shape with tokens")
    if tuple(episode_starts.shape) != (int(n_steps), int(n_envs)):
        raise ValueError("episode_starts must share [steps, envs] shape with tokens")

    buffer = MaskedRecurrentRolloutBuffer(
        buffer_size=int(n_steps),
        observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(int(num_features),), dtype=np.float32),
        action_space=spaces.Box(low=-1.0, high=1.0, shape=(int(action_slot_dim),), dtype=np.float32),
        hidden_state_shape=(int(n_steps), 1, int(n_envs), 1),
        device="cpu",
        gae_lambda=float(gae_lambda),
        gamma=float(gamma),
        n_envs=int(n_envs),
        next_state_dim=int(next_state_target_dim),
    )
    buffer._value_target_space = str(value_target_space)
    buffer._actor_gae_space = str(actor_gae_space)
    buffer.reset()
    lstm_states = RNNStates(
        pi=(
            torch.zeros((1, int(n_envs), 1), dtype=torch.float32),
            torch.zeros((1, int(n_envs), 1), dtype=torch.float32),
        ),
        vf=(
            torch.zeros((1, int(n_envs), 1), dtype=torch.float32),
            torch.zeros((1, int(n_envs), 1), dtype=torch.float32),
        ),
    )
    values = torch.zeros((int(n_envs),), dtype=torch.float32)
    log_probs = torch.zeros((int(n_envs),), dtype=torch.float32)
    objective_masks = np.zeros((int(n_steps), int(n_envs)), dtype=np.float32)
    suffix_start = int(min(int(single_eval_pos), int(n_steps)))
    if suffix_start < int(n_steps):
        objective_masks[suffix_start:] = 1.0
    for step_idx in range(int(n_steps)):
        buffer.add(
            tokens[step_idx],
            actions[step_idx],
            rewards[step_idx],
            episode_starts[step_idx],
            values,
            log_probs,
            lstm_states=lstm_states,
            action_masks=np.asarray(pack["action_masks"][step_idx], dtype=np.float32),
            next_states=(
                None
                if int(next_state_target_dim) <= 0
                else np.asarray(pack_next_states[step_idx], dtype=np.float32)
            ),
            next_state_masks=(
                None
                if int(next_state_target_dim) <= 0
                else np.asarray(pack_next_state_masks[step_idx], dtype=np.float32)
            ),
            objective_masks=objective_masks[step_idx],
        )
    dones = np.asarray(pack["dones"][-1], dtype=bool)
    buffer.compute_returns_and_advantage(torch.zeros((int(n_envs),), dtype=torch.float32), dones)
    seq_lengths_list: list[int] = []
    starts_bool = np.asarray(buffer.episode_starts, dtype=np.float32) > 0.5
    for env_idx in range(int(n_envs)):
        starts = np.nonzero(starts_bool[:, env_idx])[0].astype(np.int64)
        if starts.size <= 0 or int(starts[0]) != 0:
            starts = np.concatenate([np.asarray([0], dtype=np.int64), starts])
        ends = np.concatenate([starts[1:], np.asarray([int(n_steps)], dtype=np.int64)])
        seq_lengths_list.extend([int(max(0, end - start)) for start, end in zip(starts, ends)])
    seq_lengths = np.asarray([v for v in seq_lengths_list if int(v) > 0], dtype=np.int64)
    summary = {
        "buffer_full": bool(buffer.full),
        "pos": int(buffer.pos),
        "objective_mask_sum": float(buffer.objective_masks.sum()),
        "episode_start_count": int((buffer.episode_starts > 0.5).sum()),
        "sequence_count": int(seq_lengths.size),
        "sequence_length_min": int(seq_lengths.min()) if seq_lengths.size else 0,
        "sequence_length_max": int(seq_lengths.max()) if seq_lengths.size else 0,
        "sequence_length_mean": float(np.mean(seq_lengths)) if seq_lengths.size else 0.0,
        "returns": _finite_stats(buffer.returns),
        "advantages": _finite_stats(buffer.advantages),
        "actor_advantages": _finite_stats(buffer.actor_advantages),
        "action_masks": _finite_stats(buffer.action_masks),
        "next_states": _finite_stats(buffer.next_states),
        "next_state_masks": _finite_stats(buffer.next_state_masks),
        "digests": {
            "observations": _array_digest_payload(buffer.observations),
            "actions": _array_digest_payload(buffer.actions),
            "rewards": _array_digest_payload(buffer.rewards),
            "episode_starts": _array_digest_payload(buffer.episode_starts),
            "objective_masks": _array_digest_payload(buffer.objective_masks),
            "returns": _array_digest_payload(buffer.returns),
            "advantages": _array_digest_payload(buffer.advantages),
            "action_masks": _array_digest_payload(buffer.action_masks),
            "next_states": _array_digest_payload(buffer.next_states),
            "next_state_masks": _array_digest_payload(buffer.next_state_masks),
        },
    }
    return buffer, summary


def _collect_gym_pack(
    *,
    gym_env_id: str,
    n_envs: int,
    n_steps: int,
    seed: int,
    num_features: int,
    obs_slot_dim: int,
    action_slot_dim: int,
    next_state_target_dim: int,
    single_eval_pos: int,
    terminal_token_enabled: bool,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    import gymnasium as gym
    from gymnasium import spaces

    envs = [gym.make(str(gym_env_id)) for _ in range(int(n_envs))]
    try:
        for env in envs:
            if not isinstance(env.observation_space, spaces.Box):
                raise ValueError(f"Gym observation space must be Box, got {env.observation_space!r}")
            if not isinstance(env.action_space, spaces.Box):
                raise ValueError(f"Gym action space must be Box, got {env.action_space!r}")
        obs_dim = int(np.asarray(envs[0].observation_space.shape, dtype=np.int64).prod())
        low, high, action_dim = _box_bounds(envs[0].action_space)
        for env in envs[1:]:
            if int(np.asarray(env.observation_space.shape, dtype=np.int64).prod()) != obs_dim:
                raise ValueError("All Gym envs must share observation dimension.")
            env_low, env_high, env_action_dim = _box_bounds(env.action_space)
            if env_action_dim != action_dim or not np.allclose(env_low, low) or not np.allclose(env_high, high):
                raise ValueError("All Gym envs must share action dimension and bounds.")
        if int(obs_dim) > int(obs_slot_dim):
            raise ValueError(f"Gym obs_dim={obs_dim} exceeds obs_slot_dim={obs_slot_dim}.")
        if int(action_dim) > int(action_slot_dim):
            raise ValueError(f"Gym action_dim={action_dim} exceeds action_slot_dim={action_slot_dim}.")

        rng = np.random.default_rng(int(seed))
        obs_rows = []
        for env_idx, env in enumerate(envs):
            obs, _info = env.reset(seed=int(seed) + int(env_idx))
            obs_rows.append(np.asarray(obs, dtype=np.float32).reshape(-1))
        obs_t = np.stack(obs_rows, axis=0).astype(np.float32, copy=False)
        prev_action = np.zeros((int(n_envs), int(action_slot_dim)), dtype=np.float32)
        prev_reward = np.zeros((int(n_envs),), dtype=np.float32)
        prev_reward_mask = np.ones((int(n_envs),), dtype=np.float32)
        prev_terminal = np.zeros((int(n_envs),), dtype=np.float32)
        episode_starts_t = np.ones((int(n_envs),), dtype=np.float32)

        tokens = np.zeros((int(n_steps), int(n_envs), int(num_features)), dtype=np.float32)
        actions = np.zeros((int(n_steps), int(n_envs), int(action_slot_dim)), dtype=np.float32)
        action_masks = np.zeros_like(actions)
        rewards = np.zeros((int(n_steps), int(n_envs)), dtype=np.float32)
        episode_starts = np.zeros((int(n_steps), int(n_envs)), dtype=np.float32)
        dones = np.zeros((int(n_steps), int(n_envs)), dtype=bool)
        terminated = np.zeros_like(dones)
        truncated = np.zeros_like(dones)
        next_state_targets = np.zeros((int(n_steps), int(n_envs), int(next_state_target_dim)), dtype=np.float32)
        next_state_masks = np.zeros_like(next_state_targets)

        for step_idx in range(int(n_steps)):
            phase = np.full((int(n_envs),), float(step_idx >= int(single_eval_pos)), dtype=np.float32)
            env_info = _token_env_info(
                batch_size=int(n_envs),
                obs_dim=int(obs_dim),
                action_dim=int(action_dim),
                obs_slot_dim=int(obs_slot_dim),
                action_slot_dim=int(action_slot_dim),
                terminal_token_enabled=bool(terminal_token_enabled),
                phase_t=phase,
                terminal_t=prev_terminal,
                device=device,
            )
            token_t = build_rlpfn_obs_token_batch_from_components(
                obs_t=torch.as_tensor(obs_t, device=device, dtype=torch.float32),
                action_t=torch.as_tensor(prev_action, device=device, dtype=torch.float32),
                reward_t=torch.as_tensor(prev_reward, device=device, dtype=torch.float32),
                reward_mask_t=torch.as_tensor(prev_reward_mask, device=device, dtype=torch.float32),
                env_info=env_info,
                num_features=int(num_features),
            )
            tokens[step_idx] = token_t.detach().cpu().numpy().astype(np.float32, copy=False)
            episode_starts[step_idx] = episode_starts_t
            active_actions = np.clip(
                rng.standard_normal((int(n_envs), int(action_dim))).astype(np.float32),
                -1.0,
                1.0,
            )
            action_pad = np.zeros((int(n_envs), int(action_slot_dim)), dtype=np.float32)
            action_pad[:, : int(action_dim)] = active_actions
            mask_pad = np.zeros_like(action_pad)
            mask_pad[:, : int(action_dim)] = 1.0
            actions[step_idx] = action_pad
            action_masks[step_idx] = mask_pad

            next_obs_rows = []
            next_episode_starts = np.zeros((int(n_envs),), dtype=np.float32)
            next_terminal_token = np.zeros((int(n_envs),), dtype=np.float32)
            for env_idx, env in enumerate(envs):
                gym_action = _normalize_to_box_action(active_actions[env_idx], low, high).reshape(env.action_space.shape)
                next_obs, reward, term, trunc, _info = env.step(gym_action)
                horizon_trunc = bool(step_idx + 1 >= int(n_steps))
                done = bool(term or trunc or horizon_trunc)
                rewards[step_idx, env_idx] = float(reward)
                terminated[step_idx, env_idx] = bool(term)
                truncated[step_idx, env_idx] = bool((trunc or horizon_trunc) and not term)
                dones[step_idx, env_idx] = bool(done)
                next_state = np.asarray(next_obs, dtype=np.float32).reshape(-1)
                copy_dim = int(min(int(obs_dim), int(next_state_target_dim)))
                next_state_targets[step_idx, env_idx, :copy_dim] = next_state[:copy_dim]
                next_state_masks[step_idx, env_idx, :copy_dim] = 1.0
                if done and step_idx + 1 < int(n_steps):
                    reset_obs, _reset_info = env.reset(seed=int(seed) + 100000 + step_idx * int(n_envs) + env_idx)
                    next_obs_rows.append(np.asarray(reset_obs, dtype=np.float32).reshape(-1))
                    next_episode_starts[env_idx] = 1.0
                    next_terminal_token[env_idx] = 0.0
                else:
                    next_obs_rows.append(next_state)
                    next_terminal_token[env_idx] = float(term)
            obs_t = np.stack(next_obs_rows, axis=0).astype(np.float32, copy=False)
            prev_action = action_pad
            prev_reward = rewards[step_idx].copy()
            prev_reward_mask.fill(1.0)
            prev_terminal = next_terminal_token
            episode_starts_t = next_episode_starts

        return (
            {
                "tokens": tokens,
                "actions": actions,
                "action_masks": action_masks,
                "rewards": rewards,
                "episode_starts": episode_starts,
                "dones": dones,
                "terminated": terminated,
                "truncated": truncated,
                "next_state_targets": next_state_targets,
                "next_state_masks": next_state_masks,
            },
            {
                "gym_env_id": str(gym_env_id),
                "obs_dim": int(obs_dim),
                "action_dim": int(action_dim),
                "action_low_min": float(np.min(low)),
                "action_high_max": float(np.max(high)),
                "terminal_token_enabled": bool(terminal_token_enabled),
            },
        )
    finally:
        for env in envs:
            env.close()


def _collect_prior_pack(
    *,
    frozen_h: dict[str, Any],
    n_envs: int,
    n_steps: int,
    seed: int,
    num_features: int,
    obs_slot_dim: int,
    action_slot_dim: int,
    next_state_target_dim: int,
    env_cfg: dict[str, Any],
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    prior = EnvironmentPrior(copy.deepcopy(env_cfg))
    prior._sample_batch_hypers = lambda batch_n, _h=frozen_h: _clone_h_repeated(_h, int(batch_n))
    prior._sample_single_eval_pos = lambda n_samples_arg, rng_seed=None: 64
    vec_env = EnvironmentPriorPPOBatchVecEnv(
        env_prior=prior,
        device=device,
        num_features=int(num_features),
        n_steps=int(n_steps),
        num_envs=int(n_envs),
        action_dim=int(action_slot_dim),
        obs_slot_dim=int(obs_slot_dim),
        action_slot_dim=int(action_slot_dim),
        next_state_target_dim=int(next_state_target_dim),
    )
    try:
        vec_env.seed(int(seed))
        obs = vec_env.reset()
        rng = np.random.default_rng(int(seed) + 17)
        tokens = np.zeros((int(n_steps), int(n_envs), int(num_features)), dtype=np.float32)
        actions = np.zeros((int(n_steps), int(n_envs), int(action_slot_dim)), dtype=np.float32)
        action_masks = np.zeros_like(actions)
        rewards = np.zeros((int(n_steps), int(n_envs)), dtype=np.float32)
        episode_starts = np.zeros((int(n_steps), int(n_envs)), dtype=np.float32)
        dones = np.zeros((int(n_steps), int(n_envs)), dtype=bool)
        terminated = np.zeros_like(dones)
        truncated = np.zeros_like(dones)
        next_state_targets = np.zeros((int(n_steps), int(n_envs), int(next_state_target_dim)), dtype=np.float32)
        next_state_masks = np.zeros_like(next_state_targets)
        episode_starts_t = np.ones((int(n_envs),), dtype=np.float32)
        for step_idx in range(int(n_steps)):
            masks = vec_env.action_masks().astype(np.float32, copy=False)
            action = np.clip(rng.standard_normal((int(n_envs), int(action_slot_dim))).astype(np.float32), -1.0, 1.0)
            action *= masks
            tokens[step_idx] = np.asarray(obs, dtype=np.float32)
            actions[step_idx] = action
            action_masks[step_idx] = masks
            episode_starts[step_idx] = episode_starts_t
            vec_env.step_async(action)
            obs, reward, done, infos = vec_env.step_wait()
            rewards[step_idx] = np.asarray(reward, dtype=np.float32)
            dones[step_idx] = np.asarray(done, dtype=bool)
            truncated[step_idx] = np.asarray([bool(info.get("TimeLimit.truncated", False)) for info in infos], dtype=bool)
            terminated[step_idx] = dones[step_idx] & ~truncated[step_idx]
            for env_idx, info in enumerate(infos):
                target = np.asarray(info.get("next_state_target"), dtype=np.float32).reshape(-1)
                mask = np.asarray(info.get("next_state_mask"), dtype=np.float32).reshape(-1)
                copy_dim = int(min(target.shape[0], int(next_state_target_dim)))
                next_state_targets[step_idx, env_idx, :copy_dim] = target[:copy_dim]
                next_state_masks[step_idx, env_idx, :copy_dim] = mask[:copy_dim]
            episode_starts_t = dones[step_idx].astype(np.float32, copy=False)
        return (
            {
                "tokens": tokens,
                "actions": actions,
                "action_masks": action_masks,
                "rewards": rewards,
                "episode_starts": episode_starts,
                "dones": dones,
                "terminated": terminated,
                "truncated": truncated,
                "next_state_targets": next_state_targets,
                "next_state_masks": next_state_masks,
            },
            {
                "state_dim": int(frozen_h.get("state_dim")),
                "obs_dim": int(frozen_h.get("obs_dim")),
                "action_dim": int(frozen_h.get("action_dim")),
                "terminal_token_enabled": bool(frozen_h.get("terminal_reset_enabled", False)),
            },
        )
    finally:
        vec_env.close()


def _policy_param_snapshot(algo) -> dict[str, torch.Tensor]:
    return {
        str(name): param.detach().cpu().to(dtype=torch.float32).clone()
        for name, param in algo.policy.named_parameters()
        if bool(param.requires_grad)
    }


def _param_delta_summary(algo, before: dict[str, torch.Tensor]) -> dict[str, Any]:
    delta_sq = 0.0
    after_sq = 0.0
    max_abs_delta = 0.0
    missing = []
    for name, param in algo.policy.named_parameters():
        if not bool(param.requires_grad):
            continue
        if str(name) not in before:
            missing.append(str(name))
            continue
        after = param.detach().cpu().to(dtype=torch.float32)
        delta = after - before[str(name)]
        delta_sq += float(torch.sum(delta * delta).item())
        after_sq += float(torch.sum(after * after).item())
        max_abs_delta = max(max_abs_delta, float(torch.max(torch.abs(delta)).item()) if delta.numel() else 0.0)
    return {
        "param_delta_l2": float(math.sqrt(max(delta_sq, 0.0))),
        "param_after_l2": float(math.sqrt(max(after_sq, 0.0))),
        "param_delta_absmax": float(max_abs_delta),
        "missing_trainable_params": missing,
        "finite": bool(math.isfinite(delta_sq) and math.isfinite(after_sq) and math.isfinite(max_abs_delta)),
    }


def _grad_summary(algo) -> dict[str, Any]:
    total_sq = 0.0
    max_abs = 0.0
    count = 0
    finite = True
    for param in algo.policy.parameters():
        grad = param.grad
        if grad is None:
            continue
        grad_f = grad.detach().to(dtype=torch.float32)
        finite = bool(finite and torch.isfinite(grad_f).all().detach().cpu())
        total_sq += float(torch.sum(grad_f * grad_f).detach().cpu().item())
        max_abs = max(max_abs, float(torch.max(torch.abs(grad_f)).detach().cpu().item()) if grad_f.numel() else 0.0)
        count += 1
    return {
        "grad_tensor_count": int(count),
        "grad_l2": float(math.sqrt(max(total_sq, 0.0))),
        "grad_absmax": float(max_abs),
        "finite": bool(finite and math.isfinite(total_sq) and math.isfinite(max_abs)),
    }


def _numeric_logger_summary(algo) -> tuple[dict[str, Any], bool]:
    values = dict(getattr(algo.logger, "name_to_value", {}) or {})
    out: dict[str, Any] = {}
    finite = True
    for key, value in values.items():
        safe_value = _json_safe(value)
        out[str(key)] = safe_value
        if isinstance(safe_value, (int, float)):
            finite = bool(finite and math.isfinite(float(safe_value)))
    return out, bool(finite)


def _build_one_update_algo(
    *,
    cfg: dict[str, Any],
    env_cfg: dict[str, Any],
    frozen_h: dict[str, Any],
    device_obj: torch.device,
    num_features: int,
    n_envs: int,
    n_steps: int,
    build_seed: int,
):
    _seed_all(int(build_seed))
    _, model, _, _ = get_model(copy.deepcopy(cfg), device=str(device_obj), should_train=False, verbose=False)
    model.to(device_obj)
    model.train()
    prior = EnvironmentPrior(copy.deepcopy(env_cfg))
    prior._sample_batch_hypers = lambda batch_n, _h=frozen_h: _clone_h_repeated(_h, int(batch_n))
    prior._sample_single_eval_pos = lambda n_samples_arg, rng_seed=None: 64
    _apply_boundary_contract_mode_flags(prior, "normal")
    _apply_sep_state_reset_flag(prior, True)
    algo, _callback, vec_env = build_recurrent_ppo(
        model=model,
        env_prior=prior,
        device=str(device_obj),
        num_features=int(num_features),
        n_envs=int(n_envs),
        n_steps=int(n_steps),
        learning_rate=float(cfg["optimizer"].get("learning_rate", 2e-4)),
        batch_size=int(n_steps) * int(n_envs),
        n_epochs=1,
        gamma=float(cfg["optimizer"].get("ppo_gamma", 0.98)),
        gae_lambda=float(cfg["optimizer"].get("ppo_gae_lambda", 0.90)),
        clip_range=cfg["optimizer"].get("ppo_clip_range", 0.2),
        clip_range_vf=None,
        normalize_advantage=False,
        ent_coef=float(cfg["optimizer"].get("ppo_ent_coef", 0.0)),
        vf_coef=0.1,
        max_grad_norm=float(cfg["optimizer"].get("ppo_max_grad_norm", 0.5)),
        target_kl=0.03,
        reset_env_state_at_sep=True,
        separate_value_backbone=False,
        strict_fixed_env_mode=False,
        deterministic_actor_sampling=False,
        deterministic_batch_plan=True,
        strict_native_rollout=False,
        restore_validation_policy_state=False,
        restore_validation_policy_head_state=False,
        value_head_impl="vendor_official",
        value_path_adapter_impl="none",
        value_head_mlp_hidden_dim=256,
        space_contract="raw",
        value_target_space="raw",
        actor_gae_space="raw",
        allow_mixed_space_contract=False,
        sb3_reward_normalization_enabled=False,
        actor_baseline_mode="learned",
        actor_objective_mode="tokenwise",
        runtime_normalized_q_value_weight_override=0.0,
        runtime_next_state_flow_matching_weight_override=0.0,
        verbose=0,
    )
    algo._rwkv_aux_q_weight = 0.0
    algo._rwkv_aux_flow_weight = 0.0
    algo._rwkv_policy_loss_coef = 1.0
    algo.rollout_buffer._enable_q_target_cache = False
    algo.rollout_buffer.set_deterministic_batch_plan(True)
    algo.set_logger(configure_logger(folder=None, format_strings=[]))
    return algo, vec_env


def _old_policy_stats_for_pack(
    algo,
    pack: dict[str, np.ndarray],
    *,
    num_features: int,
    action_slot_dim: int,
    next_state_target_dim: int,
    single_eval_pos: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    buffer = algo.rollout_buffer
    _fill_existing_rollout_buffer_from_pack(
        buffer,
        pack,
        num_features=int(num_features),
        action_slot_dim=int(action_slot_dim),
        next_state_target_dim=int(next_state_target_dim),
        single_eval_pos=int(single_eval_pos),
        compute_returns=False,
    )
    total = int(buffer.buffer_size) * int(buffer.n_envs)
    buffer.set_deterministic_batch_plan(True)
    rollout_data = next(buffer.get_gpu_flat(total))
    with torch.no_grad():
        eval_outputs = algo.policy.evaluate_actions_with_hidden_flat(
            rollout_data.observations,
            rollout_data.actions,
            seq_lengths=rollout_data.seq_lengths,
            action_masks=rollout_data.action_masks,
        )
    values_flat = eval_outputs["values"].detach().to(dtype=torch.float32).cpu().numpy().reshape(-1)
    log_probs_flat = eval_outputs["log_prob"].detach().to(dtype=torch.float32).cpu().numpy().reshape(-1)
    values_by_step_env = np.zeros((int(buffer.buffer_size), int(buffer.n_envs)), dtype=np.float32)
    log_probs_by_step_env = np.zeros_like(values_by_step_env)
    for idx, step_idx, env_idx in zip(
        range(int(values_flat.shape[0])),
        np.asarray(rollout_data.flat_step_indices, dtype=np.int64).reshape(-1),
        np.asarray(rollout_data.flat_env_indices, dtype=np.int64).reshape(-1),
    ):
        values_by_step_env[int(step_idx), int(env_idx)] = values_flat[int(idx)]
        log_probs_by_step_env[int(step_idx), int(env_idx)] = log_probs_flat[int(idx)]
    summary = {
        "source": "current_policy_evaluate_actions_with_hidden_flat",
        "values": _finite_stats(values_by_step_env),
        "log_probs": _finite_stats(log_probs_by_step_env),
        "seq_count": int(len(rollout_data.seq_lengths)),
        "seq_length_min": int(np.min(rollout_data.seq_lengths)) if len(rollout_data.seq_lengths) else 0,
        "seq_length_max": int(np.max(rollout_data.seq_lengths)) if len(rollout_data.seq_lengths) else 0,
    }
    return values_by_step_env, log_probs_by_step_env, summary


def _run_one_update_smoke_arm(
    *,
    label: str,
    cfg: dict[str, Any],
    env_cfg: dict[str, Any],
    frozen_h: dict[str, Any],
    pack: dict[str, np.ndarray],
    device_obj: torch.device,
    num_features: int,
    action_slot_dim: int,
    next_state_target_dim: int,
    single_eval_pos: int,
    build_seed: int,
) -> dict[str, Any]:
    algo, vec_env = _build_one_update_algo(
        cfg=cfg,
        env_cfg=env_cfg,
        frozen_h=frozen_h,
        device_obj=device_obj,
        num_features=int(num_features),
        n_envs=int(pack["tokens"].shape[1]),
        n_steps=int(pack["tokens"].shape[0]),
        build_seed=int(build_seed),
    )
    try:
        old_values, old_log_probs, old_policy_summary = _old_policy_stats_for_pack(
            algo,
            pack,
            num_features=int(num_features),
            action_slot_dim=int(action_slot_dim),
            next_state_target_dim=int(next_state_target_dim),
            single_eval_pos=int(single_eval_pos),
        )
        buffer_summary = _fill_existing_rollout_buffer_from_pack(
            algo.rollout_buffer,
            pack,
            num_features=int(num_features),
            action_slot_dim=int(action_slot_dim),
            next_state_target_dim=int(next_state_target_dim),
            single_eval_pos=int(single_eval_pos),
            values_by_step_env=old_values,
            log_probs_by_step_env=old_log_probs,
            compute_returns=True,
        )
        before = _policy_param_snapshot(algo)
        train_exception = None
        try:
            algo.train()
        except Exception as exc:  # pragma: no cover - sentinel reports failures instead of hiding them
            train_exception = repr(exc)
        logger_values, logger_finite = _numeric_logger_summary(algo)
        param_delta = _param_delta_summary(algo, before)
        grads = _grad_summary(algo)
        update_pass = (
            train_exception is None
            and bool(logger_finite)
            and bool(param_delta["finite"])
            and bool(grads["finite"])
            and float(param_delta["param_delta_l2"]) > 0.0
            and int(getattr(algo, "_n_updates", 0)) >= 1
        )
        return {
            "label": str(label),
            "hard_pass": bool(update_pass),
            "train_exception": train_exception,
            "n_updates": int(getattr(algo, "_n_updates", 0)),
            "old_policy_summary": old_policy_summary,
            "buffer_summary_before_update": buffer_summary,
            "logger_finite": bool(logger_finite),
            "logger": logger_values,
            "param_delta": param_delta,
            "grad_summary": grads,
            "space_contract": dict(getattr(algo, "_rwkv_space_contract", {}) or {}),
            "deterministic_actor_sampling": bool(getattr(algo, "_rwkv_deterministic_actor_sampling", False)),
            "deterministic_batch_plan": bool(getattr(algo, "_rwkv_deterministic_batch_plan", False)),
            "strict_native_rollout": bool(getattr(algo, "_rwkv_strict_native_rollout", False)),
        }
    finally:
        vec_env.close()
        del algo
        if torch.cuda.is_available() and device_obj.type == "cuda":
            torch.cuda.empty_cache()


def _run_one_update_smoke(
    *,
    cfg: dict[str, Any],
    env_cfg: dict[str, Any],
    frozen_h: dict[str, Any],
    gym_pack: dict[str, np.ndarray],
    prior_pack: dict[str, np.ndarray],
    device_obj: torch.device,
    num_features: int,
    action_slot_dim: int,
    next_state_target_dim: int,
    single_eval_pos: int,
    build_seed: int,
) -> dict[str, Any]:
    gym_update = _run_one_update_smoke_arm(
        label="gym",
        cfg=cfg,
        env_cfg=env_cfg,
        frozen_h=frozen_h,
        pack=gym_pack,
        device_obj=device_obj,
        num_features=int(num_features),
        action_slot_dim=int(action_slot_dim),
        next_state_target_dim=int(next_state_target_dim),
        single_eval_pos=int(single_eval_pos),
        build_seed=int(build_seed),
    )
    prior_update = _run_one_update_smoke_arm(
        label="prior",
        cfg=cfg,
        env_cfg=env_cfg,
        frozen_h=frozen_h,
        pack=prior_pack,
        device_obj=device_obj,
        num_features=int(num_features),
        action_slot_dim=int(action_slot_dim),
        next_state_target_dim=int(next_state_target_dim),
        single_eval_pos=int(single_eval_pos),
        build_seed=int(build_seed),
    )
    return {
        "hard_pass": bool(gym_update["hard_pass"] and prior_update["hard_pass"]),
        "contract": {
            "old_policy_stats": "computed_by_current_policy_before_update",
            "update_epochs": 1,
            "batch_size": int(gym_pack["tokens"].shape[0]) * int(gym_pack["tokens"].shape[1]),
            "target_kl": 0.03,
            "normalize_advantage": False,
            "vf_coef": 0.1,
            "value_head_impl": "vendor_official",
            "space_contract": "raw",
            "aux_losses_disabled": True,
        },
        "gym": gym_update,
        "prior": prior_update,
    }


def run_sentinel(
    *,
    gym_env_id: str,
    frozen_h_json: str,
    output_json: str,
    device: str,
    n_envs: int,
    n_steps: int,
    seed: int,
    single_eval_pos: int,
    terminal_token_enabled: bool,
    materialize_buffer: bool,
    one_update_smoke: bool,
) -> dict[str, Any]:
    device_obj = torch.device(device)
    cfg = copy.deepcopy(get_model_default_config("rlpfn"))
    validate_rlpfn_maintained_path_config(cfg)
    env_cfg = copy.deepcopy(cfg["prior"]["environment"])
    num_features = int(cfg["prior"]["num_features"])
    layout = resolve_rlpfn_token_layout(env_cfg, num_features=num_features)
    frozen_h, frozen_h_path = _load_full_frozen_h(frozen_h_json)
    frozen_h = _sanitize_loaded_frozen_h_for_batch_use(frozen_h)
    obs_slot_dim = int(layout["obs_slot_dim"])
    action_slot_dim = int(layout["action_slot_dim"])
    dim_probe_prior = EnvironmentPrior(copy.deepcopy(env_cfg))
    next_state_target_dim = int(
        max(
            int(frozen_h.get("state_dim", 1)),
            int(dim_probe_prior._resolve_dim_upper_bound(env_cfg.get("state_dim", None), default=0)),
            1,
        )
    )
    if int(single_eval_pos) >= int(n_steps):
        raise ValueError(f"single_eval_pos={single_eval_pos} must be < n_steps={n_steps}.")

    gym_pack, gym_meta = _collect_gym_pack(
        gym_env_id=gym_env_id,
        n_envs=int(n_envs),
        n_steps=int(n_steps),
        seed=int(seed),
        num_features=int(num_features),
        obs_slot_dim=int(obs_slot_dim),
        action_slot_dim=int(action_slot_dim),
        next_state_target_dim=int(next_state_target_dim),
        single_eval_pos=int(single_eval_pos),
        terminal_token_enabled=bool(terminal_token_enabled),
        device=device_obj,
    )
    prior_pack, prior_meta = _collect_prior_pack(
        frozen_h=frozen_h,
        n_envs=int(n_envs),
        n_steps=int(n_steps),
        seed=int(seed),
        num_features=int(num_features),
        obs_slot_dim=int(obs_slot_dim),
        action_slot_dim=int(action_slot_dim),
        next_state_target_dim=int(next_state_target_dim),
        env_cfg=env_cfg,
        device=device_obj,
    )

    comparable_keys = [
        "tokens",
        "actions",
        "action_masks",
        "rewards",
        "episode_starts",
        "dones",
        "terminated",
        "truncated",
        "next_state_targets",
        "next_state_masks",
    ]
    shape_equal = {
        key: tuple(gym_pack[key].shape) == tuple(prior_pack[key].shape)
        for key in comparable_keys
    }
    pack_hard_pass = (
        all(shape_equal.values())
        and int(gym_pack["tokens"].shape[0]) == int(n_steps)
        and int(prior_pack["tokens"].shape[0]) == int(n_steps)
        and int(gym_pack["tokens"].shape[-1]) == int(num_features)
        and int(prior_pack["tokens"].shape[-1]) == int(num_features)
        and int(gym_pack["actions"].shape[-1]) == int(action_slot_dim)
        and int(prior_pack["actions"].shape[-1]) == int(action_slot_dim)
    )
    buffer_shape_equal: dict[str, bool] = {}
    gym_buffer_summary = None
    prior_buffer_summary = None
    one_update_summary = None
    if bool(materialize_buffer):
        gamma = float(cfg["optimizer"].get("ppo_gamma", 0.98))
        gae_lambda = float(cfg["optimizer"].get("ppo_gae_lambda", 0.90))
        _gym_buffer, gym_buffer_summary = _materialize_pack_rollout_buffer(
            gym_pack,
            num_features=int(num_features),
            action_slot_dim=int(action_slot_dim),
            next_state_target_dim=int(next_state_target_dim),
            single_eval_pos=int(single_eval_pos),
            gamma=gamma,
            gae_lambda=gae_lambda,
            value_target_space="raw",
            actor_gae_space="raw",
        )
        _prior_buffer, prior_buffer_summary = _materialize_pack_rollout_buffer(
            prior_pack,
            num_features=int(num_features),
            action_slot_dim=int(action_slot_dim),
            next_state_target_dim=int(next_state_target_dim),
            single_eval_pos=int(single_eval_pos),
            gamma=gamma,
            gae_lambda=gae_lambda,
            value_target_space="raw",
            actor_gae_space="raw",
        )
        for key in (
            "returns",
            "advantages",
            "actor_advantages",
            "action_masks",
            "next_states",
            "next_state_masks",
        ):
            buffer_shape_equal[key] = (
                tuple(gym_buffer_summary[key]["shape"])
                == tuple(prior_buffer_summary[key]["shape"])
            )
        buffer_finite = all(
            bool(summary[key]["finite"])
            for summary in (gym_buffer_summary, prior_buffer_summary)
            for key in ("returns", "advantages", "actor_advantages", "action_masks", "next_states", "next_state_masks")
        )
        buffer_hard_pass = all(buffer_shape_equal.values()) and bool(buffer_finite)
    else:
        buffer_hard_pass = True
    if bool(one_update_smoke):
        one_update_summary = _run_one_update_smoke(
            cfg=cfg,
            env_cfg=env_cfg,
            frozen_h=frozen_h,
            gym_pack=gym_pack,
            prior_pack=prior_pack,
            device_obj=device_obj,
            num_features=int(num_features),
            action_slot_dim=int(action_slot_dim),
            next_state_target_dim=int(next_state_target_dim),
            single_eval_pos=int(single_eval_pos),
            build_seed=int(seed),
        )
        one_update_hard_pass = bool(one_update_summary["hard_pass"])
    else:
        one_update_hard_pass = True
    hard_pass = bool(pack_hard_pass and buffer_hard_pass and one_update_hard_pass)
    report = {
        "audit_entry": "phase2_gym_prior_pack_isomorphism_sentinel",
        "hard_pass": bool(hard_pass),
        "pack_hard_pass": bool(pack_hard_pass),
        "buffer_hard_pass": bool(buffer_hard_pass),
        "one_update_hard_pass": bool(one_update_hard_pass),
        "contract": {
            "n_steps": int(n_steps),
            "n_envs": int(n_envs),
            "context_window_fixed": int(n_steps) == 256,
            "single_eval_pos": int(single_eval_pos),
            "num_features": int(num_features),
            "obs_slot_dim": int(obs_slot_dim),
            "action_slot_dim": int(action_slot_dim),
            "terminal_token_enabled": bool(terminal_token_enabled),
            "gym_nonterminating_rollouts_truncate_at_horizon": True,
            "training_actor_sampling_required": "stochastic",
            "buffer_materialized": bool(materialize_buffer),
            "buffer_value_target_space": "raw",
            "buffer_actor_gae_space": "raw",
            "one_update_smoke": bool(one_update_smoke),
        },
        "sources": {
            "frozen_h_json": str(frozen_h_path),
            "gym_env_id": str(gym_env_id),
        },
        "shape_equal": shape_equal,
        "buffer_shape_equal": buffer_shape_equal,
        "gym": {
            "metadata": gym_meta,
            "summary": _summarize_pack(gym_pack, label="gym"),
            "buffer_summary": gym_buffer_summary,
        },
        "prior": {
            "metadata": prior_meta,
            "summary": _summarize_pack(prior_pack, label="prior"),
            "buffer_summary": prior_buffer_summary,
        },
        "one_update_smoke": one_update_summary,
        "blocking_notes": [
            "This sentinel verifies pack/schema and rollout-buffer materialization; it intentionally does not bypass MaskedRecurrentPPO.collect_rollouts.",
            "When enabled, one_update_smoke verifies that both canonical packs can be consumed by one current-policy PPO update, but it is still not a long-training outcome guarantee.",
            "Sequence counts may differ because true environment terminations differ; this is reported but not treated as a pack schema failure.",
        ],
    }
    out_path = Path(output_json).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Gym-vs-prior canonical 256-step pack isomorphism sentinel.")
    parser.add_argument("--gym-env-id", type=str, default="Ant-v5")
    parser.add_argument("--fixed-frozen-h-json", type=str, default=DEFAULT_FROZEN_H_JSON)
    parser.add_argument("--output-json", type=str, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--seed", type=int, default=4040)
    parser.add_argument("--single-eval-pos", type=int, default=64)
    parser.add_argument("--terminal-token-enabled", type=str2bool, default=True)
    parser.add_argument("--materialize-buffer", type=str2bool, default=True)
    parser.add_argument("--one-update-smoke", type=str2bool, default=False)
    args = parser.parse_args(argv)
    report = run_sentinel(
        gym_env_id=args.gym_env_id,
        frozen_h_json=args.fixed_frozen_h_json,
        output_json=args.output_json,
        device=args.device,
        n_envs=int(args.n_envs),
        n_steps=int(args.n_steps),
        seed=int(args.seed),
        single_eval_pos=int(args.single_eval_pos),
        terminal_token_enabled=bool(args.terminal_token_enabled),
        materialize_buffer=bool(args.materialize_buffer),
        one_update_smoke=bool(args.one_update_smoke),
    )
    print(json.dumps(_json_safe({"hard_pass": report["hard_pass"], "output_json": args.output_json}), sort_keys=True))


if __name__ == "__main__":
    main()
