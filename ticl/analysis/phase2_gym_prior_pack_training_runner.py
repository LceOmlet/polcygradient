import argparse
import csv
import copy
import ctypes
import gc
import hashlib
import json
import math
import os
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
from ticl.analysis.critic_value_fit_probe import _clone_h_repeated  # noqa: E402
from ticl.analysis.phase2_gym_prior_pack_isomorphism_sentinel import (  # noqa: E402
    DEFAULT_FROZEN_H_JSON,
    _box_bounds,
    _fill_existing_rollout_buffer_from_pack,
    _finite_stats,
    _json_safe,
    _normalize_to_box_action,
    _param_delta_summary,
    _policy_param_snapshot,
    _summarize_pack,
    _summarize_pack_light,
    _token_env_info,
)
from ticl.analysis.phase2_legacy_bar_longrun_probe import (  # noqa: E402
    _load_full_frozen_h,
    _sanitize_loaded_frozen_h_for_batch_use,
)
from ticl.config_utils import str2bool  # noqa: E402
from ticl.model_builder import get_model  # noqa: E402
from ticl.model_configs import get_model_default_config  # noqa: E402
from ticl.priors.environment_prior import EnvironmentPrior  # noqa: E402
from ticl.rlpfn_maintained_path import resolve_rlpfn_token_layout, validate_rlpfn_maintained_path_config  # noqa: E402
from ticl.sb3_recurrent_ppo import (  # noqa: E402
    EnvironmentPriorPPOBatchVecEnv,
    _PpoRolloutLogAccumulator,
    _sb3_vecnormalize_reward_step,
    build_recurrent_ppo,
    build_rlpfn_obs_token_batch_from_components,
)
from stable_baselines3.common.logger import configure as configure_logger  # noqa: E402
from stable_baselines3.common.running_mean_std import RunningMeanStd  # noqa: E402
from ticl.analysis.phase2_gated_reward_path_milestone import (  # noqa: E402
    GATED_REWARD_PATH_MILESTONE,
    GATED_REWARD_PATH_MILESTONES,
    GATED_REWARD_PATH_TERMINAL_COVERAGE_MILESTONE,
    install_gated_reward_path_balance_milestone,
)


DEFAULT_OUTPUT_DIR = (
    "/home/chen/RLPFN/artifacts/"
    "phase2_gym_prior_pack_training_runner_0428"
)

UPDATE_PACK_CACHE_FORMAT = "rlpfn_update_pack_cache_v1"


def _env_truthy(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name, "").strip()
    if not value:
        return None
    return Path(value).expanduser()


def _pack_phase_logging_enabled() -> bool:
    return _env_truthy("TICL_PPO_PACK_PHASE_LOG_ENABLED", default=_env_truthy("TICL_PPO_PHASE_LOG_ENABLED", False))


def _pack_phase_log(message: str) -> None:
    if bool(_pack_phase_logging_enabled()):
        print(f"[ppo-pack-phase] {message}", flush=True)


def _torch_load_local(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _pack_cache_array_shape(payload: dict[str, Any], key: str) -> tuple[int, ...]:
    value = payload.get(key)
    if value is None:
        raise ValueError(f"Update pack cache missing {key!r}.")
    return tuple(int(v) for v in np.asarray(value).shape)


def _validate_update_pack_cache(
    *,
    path: Path,
    payload: dict[str, Any],
    arm: str,
    n_steps: int,
    n_envs: int,
    num_features: int,
    action_slot_dim: int,
    next_state_target_dim: int,
    single_eval_pos: int,
    store_next_state_targets: bool,
) -> None:
    if not isinstance(payload, dict) or payload.get("format") != UPDATE_PACK_CACHE_FORMAT:
        raise ValueError(f"{path} is not a {UPDATE_PACK_CACHE_FORMAT} cache.")
    dimensions = dict(payload.get("dimensions") or {})
    expected = {
        "arm": str(arm),
        "n_steps": int(n_steps),
        "n_envs": int(n_envs),
        "num_features": int(num_features),
        "action_slot_dim": int(action_slot_dim),
        "next_state_target_dim": int(next_state_target_dim),
        "single_eval_pos": int(single_eval_pos),
        "store_next_state_targets": bool(store_next_state_targets),
    }
    mismatches = {
        key: {"expected": value, "actual": dimensions.get(key)}
        for key, value in expected.items()
        if dimensions.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Update pack cache dimension mismatch for {path}: {mismatches}")
    pack = payload.get("pack")
    if not isinstance(pack, dict):
        raise ValueError(f"Update pack cache {path} missing pack dict.")
    if _pack_cache_array_shape(pack, "tokens") != (int(n_steps), int(n_envs), int(num_features)):
        raise ValueError(f"Update pack cache {path} tokens shape does not match current run.")
    if _pack_cache_array_shape(pack, "actions") != (int(n_steps), int(n_envs), int(action_slot_dim)):
        raise ValueError(f"Update pack cache {path} actions shape does not match current run.")
    if _pack_cache_array_shape(payload, "values") != (int(n_steps), int(n_envs)):
        raise ValueError(f"Update pack cache {path} values shape does not match current run.")
    if _pack_cache_array_shape(payload, "log_probs") != (int(n_steps), int(n_envs)):
        raise ValueError(f"Update pack cache {path} log_probs shape does not match current run.")
    if bool(store_next_state_targets):
        expected_next = (int(n_steps), int(n_envs), int(next_state_target_dim))
        if _pack_cache_array_shape(pack, "next_state_targets") != expected_next:
            raise ValueError(f"Update pack cache {path} next_state_targets shape does not match current run.")
        if _pack_cache_array_shape(pack, "next_state_masks") != expected_next:
            raise ValueError(f"Update pack cache {path} next_state_masks shape does not match current run.")


def _save_update_pack_cache(
    *,
    path: Path,
    arm: str,
    update_idx: int,
    env_seed: int,
    n_steps: int,
    n_envs: int,
    num_features: int,
    action_slot_dim: int,
    next_state_target_dim: int,
    single_eval_pos: int,
    store_next_state_targets: bool,
    pack: dict[str, np.ndarray],
    values: np.ndarray,
    log_probs: np.ndarray,
    meta: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": UPDATE_PACK_CACHE_FORMAT,
        "arm": str(arm),
        "update_idx": int(update_idx),
        "env_seed": int(env_seed),
        "dimensions": {
            "arm": str(arm),
            "n_steps": int(n_steps),
            "n_envs": int(n_envs),
            "num_features": int(num_features),
            "action_slot_dim": int(action_slot_dim),
            "next_state_target_dim": int(next_state_target_dim),
            "single_eval_pos": int(single_eval_pos),
            "store_next_state_targets": bool(store_next_state_targets),
        },
        "pack": pack,
        "values": np.asarray(values, dtype=np.float32),
        "log_probs": np.asarray(log_probs, dtype=np.float32),
        "meta": copy.deepcopy(meta),
    }
    torch.save(payload, path, pickle_protocol=4)
    print(
        "[ppo-update-pack-cache-save]"
        f" path={path}"
        f" bytes={path.stat().st_size}"
        f" update_idx={int(update_idx)}"
        f" n_steps={int(n_steps)}"
        f" n_envs={int(n_envs)}",
        flush=True,
    )


def _load_update_pack_cache(
    *,
    path: Path,
    arm: str,
    update_idx: int,
    n_steps: int,
    n_envs: int,
    num_features: int,
    action_slot_dim: int,
    next_state_target_dim: int,
    single_eval_pos: int,
    store_next_state_targets: bool,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, dict[str, Any]]:
    payload = _torch_load_local(path)
    _validate_update_pack_cache(
        path=path,
        payload=payload,
        arm=str(arm),
        n_steps=int(n_steps),
        n_envs=int(n_envs),
        num_features=int(num_features),
        action_slot_dim=int(action_slot_dim),
        next_state_target_dim=int(next_state_target_dim),
        single_eval_pos=int(single_eval_pos),
        store_next_state_targets=bool(store_next_state_targets),
    )
    pack = payload["pack"]
    values = np.asarray(payload["values"], dtype=np.float32)
    log_probs = np.asarray(payload["log_probs"], dtype=np.float32)
    meta = copy.deepcopy(payload.get("meta") or {})
    meta["update_pack_cache"] = {
        "loaded": True,
        "path": str(path),
        "source_update_idx": int(payload.get("update_idx", -1)),
        "current_update_idx": int(update_idx),
    }
    print(
        "[ppo-update-pack-cache-load]"
        f" path={path}"
        f" bytes={path.stat().st_size}"
        f" source_update_idx={int(payload.get('update_idx', -1))}"
        f" current_update_idx={int(update_idx)}",
        flush=True,
    )
    return pack, values, log_probs, meta


def _trim_host_allocator_after_pack_release() -> None:
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        return


def _release_prior_vec_env_before_update(vec_env: EnvironmentPriorPPOBatchVecEnv | None) -> dict[str, Any]:
    if vec_env is None:
        return {"enabled": False, "reason": "missing_vec_env"}
    release_fn = getattr(vec_env, "release_current_exact_scm_batch", None)
    if not callable(release_fn):
        return {"enabled": False, "reason": "release_method_unavailable"}
    summary = release_fn()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"enabled": True, **dict(summary or {})}


def _release_policy_rollout_cache_before_update(algo: Any) -> dict[str, Any]:
    policy = getattr(algo, "policy", None)
    reset_fn = getattr(policy, "reset_rollout_cache", None)
    if not callable(reset_fn):
        return {"enabled": False, "reason": "reset_rollout_cache_unavailable"}

    device = getattr(algo, "device", None)

    def _cuda_memory(prefix: str) -> dict[str, float]:
        if device is None or not torch.cuda.is_available():
            return {}
        try:
            return {
                f"{prefix}_cuda_alloc_gib": float(torch.cuda.memory_allocated(device)) / float(1024**3),
                f"{prefix}_cuda_reserved_gib": float(torch.cuda.memory_reserved(device)) / float(1024**3),
            }
        except Exception:
            return {}

    summary: dict[str, Any] = {"enabled": True}
    summary.update(_cuda_memory("before"))
    batch_size = getattr(policy, "_rollout_cache_batch_size", None)
    if batch_size is None:
        batch_size = getattr(algo, "n_envs", 0)
    reset_fn(int(batch_size))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    summary.update(_cuda_memory("after"))
    return summary


def _normalize_env_id_list(value: str, n_envs: int) -> list[str]:
    ids = [part.strip() for part in str(value).split(",") if part.strip()]
    if not ids:
        raise ValueError("At least one Gym env id is required.")
    return [ids[idx % len(ids)] for idx in range(int(n_envs))]


def _optional_float(value: str | float | int | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "none", "null"}:
            return None
        return float(text)
    return float(value)


def _load_fixed_frozen_h_list_from_csv(
    csv_path: str | Path,
    *,
    rule: str | None,
    limit: int | None,
) -> tuple[list[dict[str, Any]], list[str], list[int]]:
    path = Path(csv_path).expanduser().resolve()
    rule_norm = None if rule is None or str(rule).strip() == "" else str(rule).strip()
    selected: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "full_frozen_h_json" not in (reader.fieldnames or []):
            raise ValueError(f"{path} must contain a full_frozen_h_json column.")
        for row in reader:
            if rule_norm is not None and str(row.get("rule", "")).strip() != rule_norm:
                continue
            selected.append(row)
            if limit is not None and int(limit) > 0 and int(len(selected)) >= int(limit):
                break
    h_list: list[dict[str, Any]] = []
    h_paths: list[str] = []
    env_seeds: list[int] = []
    for idx, row in enumerate(selected):
        h_path = str(row.get("full_frozen_h_json", "")).strip()
        if not h_path:
            raise ValueError(f"Empty full_frozen_h_json in {path} at selected row {idx}.")
        h, resolved = _load_full_frozen_h(h_path)
        h_list.append(_sanitize_loaded_frozen_h_for_batch_use(h))
        h_paths.append(str(resolved))
        seed_value = row.get("seed", "")
        try:
            env_seed = int(float(seed_value))
        except (TypeError, ValueError):
            env_seed = int(idx)
        env_seeds.append(env_seed)
    return h_list, h_paths, env_seeds


def _logger_values(algo) -> dict[str, Any]:
    return {str(k): _json_safe(v) for k, v in dict(getattr(algo.logger, "name_to_value", {}) or {}).items()}


def _stable_digest(value: Any) -> str:
    payload = json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _vector_stats(values: Any) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "q01": None,
            "q10": None,
            "q50": None,
            "q90": None,
            "q99": None,
            "max": None,
            "absmax": None,
        }
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "q01": float(np.quantile(arr, 0.01)),
        "q10": float(np.quantile(arr, 0.10)),
        "q50": float(np.quantile(arr, 0.50)),
        "q90": float(np.quantile(arr, 0.90)),
        "q99": float(np.quantile(arr, 0.99)),
        "max": float(np.max(arr)),
        "absmax": float(np.max(np.abs(arr))),
    }


def _safe_corrcoef(x: Any, y: Any) -> float | None:
    x_arr = np.asarray(x, dtype=np.float64).reshape(-1)
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if int(mask.sum()) < 3:
        return None
    x0 = x_arr[mask] - float(np.mean(x_arr[mask]))
    y0 = y_arr[mask] - float(np.mean(y_arr[mask]))
    denom = float(np.linalg.norm(x0) * np.linalg.norm(y0))
    if denom <= 1e-12:
        return None
    return float(np.dot(x0, y0) / denom)


def _basic_stats(values: Any) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0, "mean": None, "std": None, "min": None, "q10": None, "q50": None, "q90": None, "max": None, "absmax": None}
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "q10": float(np.quantile(arr, 0.10)),
        "q50": float(np.quantile(arr, 0.50)),
        "q90": float(np.quantile(arr, 0.90)),
        "max": float(np.max(arr)),
        "absmax": float(np.max(np.abs(arr))),
    }


def _empty_flat_stats() -> dict[str, Any]:
    return {
        "count": 0,
        "sum": 0.0,
        "sumsq": 0.0,
        "min": None,
        "max": None,
    }


def _accumulate_flat_stats(acc: dict[str, Any], values: Any) -> None:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return
    acc["count"] = int(acc["count"]) + int(arr.size)
    acc["sum"] = float(acc["sum"]) + float(np.sum(arr))
    acc["sumsq"] = float(acc["sumsq"]) + float(np.sum(np.square(arr)))
    min_v = float(np.min(arr))
    max_v = float(np.max(arr))
    acc["min"] = min_v if acc["min"] is None else min(float(acc["min"]), min_v)
    acc["max"] = max_v if acc["max"] is None else max(float(acc["max"]), max_v)


def _finalize_flat_stats(acc: dict[str, Any]) -> dict[str, Any]:
    count = int(acc.get("count", 0))
    if count <= 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "absmax": None,
        }
    mean = float(acc["sum"]) / float(count)
    var = max(float(acc["sumsq"]) / float(count) - mean * mean, 0.0)
    min_v = float(acc["min"])
    max_v = float(acc["max"])
    return {
        "count": int(count),
        "mean": float(mean),
        "std": float(math.sqrt(var)),
        "min": min_v,
        "max": max_v,
        "absmax": float(max(abs(min_v), abs(max_v))),
    }


def _obs_active_mask(obs_dims: list[int], obs_slot_dim: int) -> np.ndarray:
    mask = np.zeros((len(obs_dims), int(obs_slot_dim)), dtype=bool)
    for env_idx, obs_dim in enumerate(obs_dims):
        active_dim = int(max(0, min(int(obs_dim), int(obs_slot_dim))))
        if active_dim > 0:
            mask[env_idx, :active_dim] = True
    return mask


def _ensure_obs_norm_state(algo, obs_slot_dim: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    obs_slot_dim = int(obs_slot_dim)
    mean = getattr(algo, "_rwkv_sb3_observation_normalization_mean", None)
    var = getattr(algo, "_rwkv_sb3_observation_normalization_var", None)
    count = getattr(algo, "_rwkv_sb3_observation_normalization_count", None)
    if (
        not isinstance(mean, np.ndarray)
        or not isinstance(var, np.ndarray)
        or not isinstance(count, np.ndarray)
        or tuple(mean.shape) != (obs_slot_dim,)
        or tuple(var.shape) != (obs_slot_dim,)
        or tuple(count.shape) != (obs_slot_dim,)
    ):
        mean = np.zeros((obs_slot_dim,), dtype=np.float64)
        var = np.ones((obs_slot_dim,), dtype=np.float64)
        count = np.full((obs_slot_dim,), 1e-4, dtype=np.float64)
        algo._rwkv_sb3_observation_normalization_mean = mean
        algo._rwkv_sb3_observation_normalization_var = var
        algo._rwkv_sb3_observation_normalization_count = count
    return mean, var, count


def _masked_running_mean_var_update(
    *,
    mean: np.ndarray,
    var: np.ndarray,
    count: np.ndarray,
    obs: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    obs64 = np.asarray(obs, dtype=np.float64)
    mask_bool = np.asarray(mask, dtype=bool)
    for dim_idx in range(int(obs64.shape[1])):
        active = mask_bool[:, dim_idx]
        batch_count = int(active.sum())
        if batch_count <= 0:
            continue
        values = obs64[active, dim_idx]
        batch_mean = float(np.mean(values))
        batch_var = float(np.var(values))
        old_count = float(count[dim_idx])
        old_mean = float(mean[dim_idx])
        old_var = float(var[dim_idx])
        total_count = old_count + float(batch_count)
        delta = batch_mean - old_mean
        new_mean = old_mean + delta * float(batch_count) / total_count
        m_a = old_var * old_count
        m_b = batch_var * float(batch_count)
        m2 = m_a + m_b + delta * delta * old_count * float(batch_count) / total_count
        mean[dim_idx] = new_mean
        var[dim_idx] = max(m2 / total_count, 1e-12)
        count[dim_idx] = total_count
    return mean, var, count


def _apply_observation_normalization_if_enabled(
    algo,
    obs: np.ndarray,
    obs_dims: list[int],
    obs_slot_dim: int,
    stats_acc: dict[str, dict[str, Any]] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not bool(getattr(algo, "_rwkv_sb3_observation_normalization_enabled", False)):
        return np.asarray(obs, dtype=np.float32), {"enabled": False}
    obs_arr = np.asarray(obs, dtype=np.float32)
    if obs_arr.ndim != 2 or obs_arr.shape[1] < int(obs_slot_dim):
        raise ValueError(
            "Observation normalization expects a 2D obs prefix array with obs_slot_dim columns; "
            f"got shape={tuple(obs_arr.shape)}, obs_slot_dim={int(obs_slot_dim)}"
        )
    prefix = obs_arr[:, : int(obs_slot_dim)].astype(np.float32, copy=True)
    mask = _obs_active_mask(obs_dims, int(obs_slot_dim))
    mean, var, count = _ensure_obs_norm_state(algo, int(obs_slot_dim))
    if stats_acc is not None:
        _accumulate_flat_stats(stats_acc["raw_active_obs"], prefix[mask])
    _masked_running_mean_var_update(mean=mean, var=var, count=count, obs=prefix, mask=mask)
    epsilon = float(getattr(algo, "_rwkv_sb3_observation_normalization_epsilon", 1e-8))
    clip_obs = float(getattr(algo, "_rwkv_sb3_observation_normalization_clip", 10.0))
    norm_prefix = (prefix.astype(np.float64) - mean.reshape(1, -1)) / np.sqrt(var.reshape(1, -1) + epsilon)
    norm_prefix = np.clip(norm_prefix, -clip_obs, clip_obs).astype(np.float32)
    norm_prefix = np.where(mask, norm_prefix, 0.0).astype(np.float32, copy=False)
    out = obs_arr.copy()
    out[:, : int(obs_slot_dim)] = norm_prefix
    if stats_acc is not None:
        _accumulate_flat_stats(stats_acc["normalized_active_obs"], norm_prefix[mask])
    active_counts = count[count > 1e-4]
    stats = {
        "enabled": True,
        "clip_obs": float(clip_obs),
        "epsilon": float(epsilon),
        "active_dim_count": int(active_counts.size),
        "rms_count_min": float(np.min(active_counts)) if active_counts.size else None,
        "rms_count_max": float(np.max(active_counts)) if active_counts.size else None,
        "rms_mean": _finite_stats(mean[count > 1e-4]),
        "rms_var": _finite_stats(var[count > 1e-4]),
    }
    return out.astype(np.float32, copy=False), stats


def _apply_observation_normalization_snapshot(
    algo,
    obs: np.ndarray,
    obs_dims: list[int],
    obs_slot_dim: int,
) -> np.ndarray:
    """Normalize state slots with the current RMS without updating statistics."""
    if not bool(getattr(algo, "_rwkv_sb3_observation_normalization_enabled", False)):
        return np.asarray(obs, dtype=np.float32)
    obs_arr = np.asarray(obs, dtype=np.float32)
    prefix = obs_arr[:, : int(obs_slot_dim)].astype(np.float32, copy=True)
    mask = _obs_active_mask(obs_dims, int(obs_slot_dim))
    mean, var, _count = _ensure_obs_norm_state(algo, int(obs_slot_dim))
    epsilon = float(getattr(algo, "_rwkv_sb3_observation_normalization_epsilon", 1e-8))
    clip_obs = float(getattr(algo, "_rwkv_sb3_observation_normalization_clip", 10.0))
    norm_prefix = (prefix.astype(np.float64) - mean.reshape(1, -1)) / np.sqrt(var.reshape(1, -1) + epsilon)
    norm_prefix = np.clip(norm_prefix, -clip_obs, clip_obs).astype(np.float32)
    norm_prefix = np.where(mask, norm_prefix, 0.0).astype(np.float32, copy=False)
    out = obs_arr.copy()
    out[:, : int(obs_slot_dim)] = norm_prefix
    return out.astype(np.float32, copy=False)


def _new_obs_norm_stats_acc() -> dict[str, dict[str, Any]]:
    return {
        "raw_active_obs": _empty_flat_stats(),
        "normalized_active_obs": _empty_flat_stats(),
    }


def _finalize_obs_norm_stats_acc(acc: dict[str, dict[str, Any]], last_stats: dict[str, Any]) -> dict[str, Any]:
    return {
        **last_stats,
        "raw_active_obs": _finalize_flat_stats(acc["raw_active_obs"]),
        "normalized_active_obs": _finalize_flat_stats(acc["normalized_active_obs"]),
    }


def _tensor_vector(env: dict[str, Any], key: str) -> list[float] | None:
    value = env.get(key, None)
    if torch.is_tensor(value):
        return [float(v) for v in value.detach().cpu().reshape(-1).tolist()]
    if isinstance(value, np.ndarray):
        return [float(v) for v in value.reshape(-1).tolist()]
    return None


def _tensor_int_vector(env: dict[str, Any], key: str) -> list[int] | None:
    values = _tensor_vector(env, key)
    if values is None:
        return None
    return [int(round(float(v))) for v in values]


def _dim_list(meta: dict[str, Any], key: str, fallback: int, n_envs: int) -> list[int]:
    values = meta.get(key, None)
    if isinstance(values, list) and values:
        out = [int(v) for v in values]
    else:
        out = [int(fallback)] * int(n_envs)
    if len(out) < int(n_envs):
        out.extend([int(fallback)] * (int(n_envs) - len(out)))
    return [int(max(0, v)) for v in out[: int(n_envs)]]


def _meta_list_value(
    meta: dict[str, Any],
    key: str,
    env_idx: int,
    default: Any,
) -> Any:
    values = meta.get(key, None)
    if isinstance(values, list) and int(env_idx) < len(values):
        return values[int(env_idx)]
    return default


def _reward_component_stats(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "sum": None,
            "mean": None,
            "std": None,
            "min": None,
            "q10": None,
            "q50": None,
            "q90": None,
            "max": None,
            "positive_fraction": None,
            "q90_gt0": None,
        }
    q90 = float(np.quantile(arr, 0.90))
    return {
        "sum": float(np.sum(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "q10": float(np.quantile(arr, 0.10)),
        "q50": float(np.quantile(arr, 0.50)),
        "q90": q90,
        "max": float(np.max(arr)),
        "positive_fraction": float(np.mean(arr > 0.0)),
        "q90_gt0": bool(q90 > 0.0),
    }


def _reward_component_arrays_from_pack(
    *,
    arm: str,
    reward: np.ndarray,
    active_actions: np.ndarray,
    active_action_masks: np.ndarray,
    meta: dict[str, Any],
    env_idx: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    reward_arr = np.asarray(reward, dtype=np.float64).reshape(-1)
    if str(arm) != "prior":
        return {}, {
            "enabled": False,
            "reason": "component logging currently applies to exact-SCM prior arms only",
        }
    action_arr = np.asarray(active_actions, dtype=np.float64)
    mask_arr = np.asarray(active_action_masks, dtype=np.float64)
    if action_arr.ndim != 2 or mask_arr.shape != action_arr.shape:
        return {}, {
            "enabled": False,
            "reason": "missing or shape-mismatched action/action_mask arrays",
        }

    ctrl_weight = float(_meta_list_value(meta, "ctrl_reward_weight", env_idx, 0.0) or 0.0)
    ctrl_enabled = bool(_meta_list_value(meta, "ctrl_reward_enabled", env_idx, False))
    survival_weight = float(_meta_list_value(meta, "survival_reward_weight", env_idx, 0.0) or 0.0)
    survival_enabled = bool(_meta_list_value(meta, "survival_reward_enabled", env_idx, False))
    reward_transform = str(_meta_list_value(meta, "reinforce_reward_transform", env_idx, "unknown"))
    reward_clip = float(_meta_list_value(meta, "reward_clip", env_idx, float("nan")) or float("nan"))

    denom = np.maximum(np.sum(mask_arr, axis=-1), 1.0)
    mean_sq = np.sum(np.square(action_arr * mask_arr), axis=-1) / denom
    ctrl = (-ctrl_weight * mean_sq) if ctrl_enabled else np.zeros_like(reward_arr)
    survival = (
        np.full_like(reward_arr, survival_weight, dtype=np.float64)
        if survival_enabled
        else np.zeros_like(reward_arr, dtype=np.float64)
    )
    aux = ctrl + survival
    reward_env = reward_arr - aux
    contract_exact = reward_transform == "none" and (not math.isfinite(reward_clip) or reward_clip >= 10.0)
    return {
        "reward_env": reward_env,
        "reward_ctrl": ctrl,
        "reward_survival": survival,
        "reward_aux": aux,
    }, {
        "enabled": True,
        "contract": (
            "reward_env_component = raw_total_reward - ctrl_reward - survival_reward. "
            "This is the exact base env reward after exact-SCM clip/transform; for current "
            "none-transform runs it equals clipped raw env reward."
        ),
        "contract_exact_for_current_none_transform_runs": bool(contract_exact),
        "reward_env_component_kind": "base_after_clip_transform",
        "ctrl_enabled": bool(ctrl_enabled),
        "ctrl_weight": float(ctrl_weight),
        "survival_enabled": bool(survival_enabled),
        "survival_weight": float(survival_weight),
        "reinforce_reward_transform": reward_transform,
        "reward_clip": float(reward_clip) if math.isfinite(reward_clip) else None,
    }


def _rollout_reward_stats_from_pack(
    *,
    arm: str,
    pack: dict[str, np.ndarray],
    meta: dict[str, Any],
    single_eval_pos: int,
) -> dict[str, Any]:
    raw_reward = np.asarray(pack.get("raw_rewards", pack["rewards"]), dtype=np.float64)
    dones = np.asarray(pack["dones"], dtype=bool)
    if raw_reward.ndim != 2:
        return {"enabled": False, "reason": f"expected [steps, envs] rewards, got {list(raw_reward.shape)}"}
    n_steps, n_envs = int(raw_reward.shape[0]), int(raw_reward.shape[1])
    actions = np.asarray(pack.get("actions"), dtype=np.float64)
    action_masks = np.asarray(pack.get("action_masks"), dtype=np.float64)
    ctrl = np.zeros_like(raw_reward, dtype=np.float64)
    survival = np.zeros_like(raw_reward, dtype=np.float64)
    terminal_bonus = np.zeros_like(raw_reward, dtype=np.float64)

    if (
        str(arm) == "prior"
        and actions.ndim == 3
        and action_masks.shape == actions.shape
        and tuple(actions.shape[:2]) == (n_steps, n_envs)
    ):
        ctrl_weight = np.asarray(
            [_meta_list_value(meta, "ctrl_reward_weight", env_idx, 0.0) or 0.0 for env_idx in range(n_envs)],
            dtype=np.float64,
        ).reshape((1, n_envs))
        ctrl_enabled = np.asarray(
            [bool(_meta_list_value(meta, "ctrl_reward_enabled", env_idx, False)) for env_idx in range(n_envs)],
            dtype=bool,
        ).reshape((1, n_envs))
        survival_weight = np.asarray(
            [_meta_list_value(meta, "survival_reward_weight", env_idx, 0.0) or 0.0 for env_idx in range(n_envs)],
            dtype=np.float64,
        ).reshape((1, n_envs))
        survival_enabled = np.asarray(
            [bool(_meta_list_value(meta, "survival_reward_enabled", env_idx, False)) for env_idx in range(n_envs)],
            dtype=bool,
        ).reshape((1, n_envs))
        denom = np.maximum(np.sum(action_masks, axis=-1), 1.0)
        mean_sq = np.sum(np.square(actions * action_masks), axis=-1) / denom
        ctrl = np.where(ctrl_enabled, -ctrl_weight * mean_sq, 0.0)
        survival = np.where(
            np.broadcast_to(survival_enabled, (n_steps, n_envs)),
            np.broadcast_to(survival_weight, (n_steps, n_envs)),
            0.0,
        )
    reward_env = raw_reward - ctrl - survival - terminal_bonus

    acc = _PpoRolloutLogAccumulator(n_envs, device=None)
    suffix_start = int(min(max(int(single_eval_pos), 0), n_steps))
    for step_idx in range(n_steps):
        objective_mask = np.full((n_envs,), bool(step_idx >= suffix_start), dtype=bool)
        acc.observe_step(
            reward=raw_reward[step_idx],
            reward_env=reward_env[step_idx],
            reward_ctrl=ctrl[step_idx],
            reward_survival=survival[step_idx],
            reward_terminal_bonus=terminal_bonus[step_idx],
            objective_mask=objective_mask,
            dones=dones[step_idx],
        )
    stats = acc.finalize()
    stats["enabled"] = True
    stats["source"] = "trusted_pack_raw_rewards"
    stats["single_eval_pos"] = int(single_eval_pos)
    return stats


def _add_prior_reward_component_meta_from_h_list(
    meta: dict[str, Any],
    h_list: list[dict[str, Any]] | None,
    n_envs: int,
) -> None:
    if not isinstance(h_list, list) or len(h_list) < int(n_envs):
        return

    def _enabled(h: dict[str, Any], draw_key: str, prob_key: str) -> bool:
        return float(h.get(draw_key, 1.0)) < float(h.get(prob_key, 0.0))

    meta["ctrl_reward_weight"] = [float(h.get("ctrl_reward_weight", 0.0) or 0.0) for h in h_list[: int(n_envs)]]
    meta["ctrl_reward_enabled"] = [
        _enabled(h, "_ctrl_reward_enable_u", "ctrl_reward_enable_prob") for h in h_list[: int(n_envs)]
    ]
    meta["survival_reward_weight"] = [
        float(h.get("survival_reward_weight", 0.0) or 0.0) for h in h_list[: int(n_envs)]
    ]
    meta["survival_reward_enabled"] = [
        _enabled(h, "_survival_reward_enable_u", "survival_reward_enable_prob")
        for h in h_list[: int(n_envs)]
    ]
    meta["terminal_reset_enabled"] = [
        bool(h.get("terminal_reset_enabled", False)) for h in h_list[: int(n_envs)]
    ]
    meta["terminal_reset_count_target"] = [
        float(h.get("terminal_reset_count_target", 0.0) or 0.0) for h in h_list[: int(n_envs)]
    ]
    for key in (
        "gated_terminal_coverage_bucket",
        "gated_terminal_coverage_applied",
        "gated_terminal_coverage_original_target",
        "gated_terminal_coverage_target",
    ):
        if any(key in h for h in h_list[: int(n_envs)]):
            meta[key] = [h.get(key) for h in h_list[: int(n_envs)]]
    meta["reinforce_reward_transform"] = [
        str(h.get("reinforce_reward_transform", "none")) for h in h_list[: int(n_envs)]
    ]
    meta["reward_clip"] = [float(h.get("reward_clip", 10.0) or 10.0) for h in h_list[: int(n_envs)]]


def _semantic_probe_from_pack(
    *,
    arm: str,
    pack: dict[str, np.ndarray],
    meta: dict[str, Any],
    obs_slot_dim: int,
    action_slot_dim: int,
) -> dict[str, Any]:
    """Read-only probes for Gym-vs-Prior comparability.

    This intentionally consumes only the already-collected canonical pack and
    collector metadata, so it cannot perturb rollout RNG, policy sampling, or
    PPO update semantics.
    """
    tokens = np.asarray(pack["tokens"], dtype=np.float32)
    actions = np.asarray(pack["actions"], dtype=np.float32)
    action_masks = np.asarray(pack["action_masks"], dtype=np.float32)
    rewards = np.asarray(pack["rewards"], dtype=np.float32)
    raw_rewards = np.asarray(pack.get("raw_rewards", pack["rewards"]), dtype=np.float32)
    rewards_are_sb3_normalized = bool(pack.get("rewards_are_sb3_normalized", False))
    dones = np.asarray(pack["dones"], dtype=bool)
    truncated = np.asarray(pack["truncated"], dtype=bool)
    if "next_state_targets" not in pack or "next_state_masks" not in pack:
        raise ValueError("semantic probe requires next_state_targets/masks in the collected pack")
    next_states = np.asarray(pack["next_state_targets"], dtype=np.float32)
    next_state_masks = np.asarray(pack["next_state_masks"], dtype=np.float32)

    n_steps = int(tokens.shape[0])
    n_envs = int(tokens.shape[1])
    obs_dims = _dim_list(meta, "obs_dims", int(obs_slot_dim), n_envs)
    obs_slot_dims = _dim_list(meta, "obs_slot_dims", int(obs_slot_dim), n_envs)
    action_dims = _dim_list(meta, "action_dims", int(action_slot_dim), n_envs)
    state_dims = _dim_list(meta, "state_dims", int(next_states.shape[-1]), n_envs)

    obs_values: list[np.ndarray] = []
    obs_per_env_dim_std: list[float] = []
    obs_per_env_dim_rms: list[float] = []
    reward_token_values: list[np.ndarray] = []
    action_active_values: list[np.ndarray] = []
    action_temporal_std: list[float] = []
    action_temporal_delta_absmean: list[float] = []
    next_state_rms: list[float] = []
    next_state_drift_l2: list[float] = []
    next_state_drift_to_norm_ratio: list[float] = []

    for env_idx in range(n_envs):
        obs_dim = int(min(obs_dims[env_idx], tokens.shape[-1]))
        if obs_dim > 0:
            obs = tokens[:, env_idx, :obs_dim]
            obs_values.append(obs.reshape(-1))
            obs_per_env_dim_std.append(float(np.mean(np.std(obs, axis=0))))
            obs_per_env_dim_rms.append(float(np.mean(np.sqrt(np.mean(np.square(obs), axis=0)))))

        reward_idx = int(obs_slot_dims[env_idx])
        if 0 <= reward_idx < int(tokens.shape[-1]):
            reward_token_values.append(tokens[:, env_idx, reward_idx].reshape(-1))

        action_dim = int(min(action_dims[env_idx], actions.shape[-1]))
        if action_dim > 0:
            active = actions[:, env_idx, :action_dim]
            action_active_values.append(active.reshape(-1))
            per_dim_std = np.std(active, axis=0)
            action_temporal_std.extend([float(v) for v in per_dim_std.tolist()])
            if n_steps > 1:
                action_temporal_delta_absmean.append(float(np.mean(np.abs(np.diff(active, axis=0)))))

        state_dim = int(min(state_dims[env_idx], next_states.shape[-1]))
        if state_dim > 0:
            state = next_states[:, env_idx, :state_dim]
            mask = next_state_masks[:, env_idx, :state_dim] > 0.5
            valid_state = state[mask]
            if valid_state.size:
                next_state_rms.append(float(np.sqrt(np.mean(np.square(valid_state)))))
            if n_steps > 1:
                valid_pair = mask[1:] & mask[:-1]
                delta = (state[1:] - state[:-1]) * valid_pair
                valid_counts = np.maximum(valid_pair.sum(axis=1), 1)
                l2 = np.sqrt(np.sum(np.square(delta), axis=1) / valid_counts)
                l2 = l2[np.isfinite(l2)]
                if l2.size:
                    drift_mean = float(np.mean(l2))
                    next_state_drift_l2.append(drift_mean)
                    state_norm = float(np.sqrt(np.mean(np.square(valid_state)))) if valid_state.size else 0.0
                    next_state_drift_to_norm_ratio.append(drift_mean / max(state_norm, 1e-12))

    masked_actions = actions * action_masks
    unit_clipped_masked_actions = np.clip(masked_actions, -1.0, 1.0) * action_masks
    action_norm = np.sqrt(np.sum(np.square(masked_actions), axis=-1))
    unit_clipped_action_norm = np.sqrt(np.sum(np.square(unit_clipped_masked_actions), axis=-1))
    action_delta_norm = (
        np.sqrt(np.sum(np.square(np.diff(actions * action_masks, axis=0)), axis=-1))
        if n_steps > 1
        else np.zeros((0, n_envs), dtype=np.float32)
    )
    active_action_concat = np.concatenate(action_active_values) if action_active_values else np.asarray([], dtype=np.float32)
    raw_unit_clip_excess = np.maximum(np.abs(active_action_concat) - 1.0, 0.0)
    per_env_training_returns = np.sum(rewards, axis=0)
    per_env_training_reward_std = np.std(rewards, axis=0)
    per_env_raw_returns = np.sum(raw_rewards, axis=0)
    per_env_raw_reward_std = np.std(raw_rewards, axis=0)
    first_done = []
    for env_idx in range(n_envs):
        done_idx = np.nonzero(dones[:, env_idx])[0]
        first_done.append(int(done_idx[0]) if done_idx.size else None)

    prior_gain_keys = [
        "reward_state_input_gain_fraction",
        "reward_action_input_gain_fraction",
        "reward_noise_input_gain_fraction",
        "reward_state_to_noise_gain_ratio",
        "reward_state_to_action_gain_ratio",
        "reward_state_input_gain_fraction_conditioned_accepted_gain",
        "reward_action_input_gain_fraction_conditioned_accepted_gain",
        "reward_state_to_action_gain_ratio_conditioned_accepted_ratio",
        "reward_state_input_gain_fraction_conditioned_attempt",
        "reward_topology_conditioned_attempt",
        "reward_topology_conditioned_total_candidate_draws",
    ]
    prior_terminal_keys = [
        "terminal_reset_enabled",
        "terminal_reset_count_target",
        "gated_terminal_coverage_bucket",
        "gated_terminal_coverage_applied",
        "gated_terminal_coverage_original_target",
        "gated_terminal_coverage_target",
    ]
    prior_gain_summary = {
        key: _vector_stats(meta[key])
        for key in prior_gain_keys
        if isinstance(meta.get(key), list)
    }

    return {
        "probe_version": 1,
        "probe_contract": (
            "read-only semantic probe from canonical pack; no policy calls, no env stepping, "
            "no RNG consumption, no PPO semantic mutation"
        ),
        "limitations": [
            "Gym next_state is observation-space next obs; Prior next_state is prior state target under the shared boundary",
        ],
        "arm": str(arm),
        "n_envs": int(n_envs),
        "n_steps": int(n_steps),
        "obs_dims": _vector_stats(obs_dims),
        "action_dims": _vector_stats(action_dims),
        "state_dims": _vector_stats(state_dims),
        "obs_only": {
            "values": _vector_stats(np.concatenate(obs_values) if obs_values else []),
            "per_env_mean_dim_std": _vector_stats(obs_per_env_dim_std),
            "per_env_mean_dim_rms": _vector_stats(obs_per_env_dim_rms),
        },
        "model_input_prev_reward_token": _vector_stats(
            np.concatenate(reward_token_values) if reward_token_values else []
        ),
        "reward_scale": {
            "pack_rewards_are_sb3_normalized": bool(rewards_are_sb3_normalized),
            "reward_sum_field": "training_reward_sum",
            "raw_reward_sum_field": "raw_reward_sum",
        },
        "training_reward": {
            "all_steps": _vector_stats(rewards.reshape(-1)),
            "per_env_return": _vector_stats(per_env_training_returns),
            "per_env_reward_std": _vector_stats(per_env_training_reward_std),
        },
        "raw_reward": {
            "all_steps": _vector_stats(raw_rewards.reshape(-1)),
            "per_env_return": _vector_stats(per_env_raw_returns),
            "per_env_reward_std": _vector_stats(per_env_raw_reward_std),
        },
        "episode_dynamics": {
            "done_count": int(dones.sum()),
            "truncated_count": int(truncated.sum()),
            "first_done_step": first_done,
            "first_done_step_stats": _vector_stats([v for v in first_done if v is not None]),
        },
        "action_distribution": {
            "active_values": _vector_stats(np.concatenate(action_active_values) if action_active_values else []),
            "active_abs_values": _vector_stats(
                np.abs(np.concatenate(action_active_values)) if action_active_values else []
            ),
            "unit_clipped_active_abs_values": _vector_stats(
                np.abs(np.concatenate([np.clip(v, -1.0, 1.0).reshape(-1) for v in action_active_values]))
                if action_active_values
                else []
            ),
            "per_dim_temporal_std": _vector_stats(action_temporal_std),
            "per_env_temporal_delta_absmean": _vector_stats(action_temporal_delta_absmean),
            "action_norm": _vector_stats(action_norm.reshape(-1)),
            "unit_clipped_action_norm": _vector_stats(unit_clipped_action_norm.reshape(-1)),
            "raw_policy_action_unit_clip_saturation_fraction": (
                float(np.mean(np.abs(active_action_concat) > 1.0)) if active_action_concat.size else None
            ),
            "raw_policy_action_unit_clip_excess_absmean": (
                float(np.mean(raw_unit_clip_excess)) if raw_unit_clip_excess.size else None
            ),
            "open_loop_constant_dim_fraction_std_lt_0p05": (
                float(np.mean(np.asarray(action_temporal_std, dtype=np.float64) < 0.05))
                if action_temporal_std
                else None
            ),
            "open_loop_constant_dim_fraction_std_lt_0p10": (
                float(np.mean(np.asarray(action_temporal_std, dtype=np.float64) < 0.10))
                if action_temporal_std
                else None
            ),
        },
        "state_or_obs_drift": {
            "next_state_rms": _vector_stats(next_state_rms),
            "step_l2_delta": _vector_stats(next_state_drift_l2),
            "step_l2_delta_to_state_rms_ratio": _vector_stats(next_state_drift_to_norm_ratio),
        },
        "prior_gain_summary": prior_gain_summary,
    }


def _per_env_semantic_probe_rows_from_pack(
    *,
    arm: str,
    update_idx: int,
    pack: dict[str, np.ndarray],
    meta: dict[str, Any],
    obs_slot_dim: int,
    action_slot_dim: int,
) -> list[dict[str, Any]]:
    """Return one read-only semantic probe row per environment column."""
    tokens = np.asarray(pack["tokens"], dtype=np.float32)
    actions = np.asarray(pack["actions"], dtype=np.float32)
    action_masks = np.asarray(pack["action_masks"], dtype=np.float32)
    rewards = np.asarray(pack["rewards"], dtype=np.float32)
    dones = np.asarray(pack["dones"], dtype=bool)
    truncated = np.asarray(pack["truncated"], dtype=bool)
    if "next_state_targets" not in pack or "next_state_masks" not in pack:
        raise ValueError("semantic probe sidecar requires next_state_targets/masks in the collected pack")
    next_states = np.asarray(pack["next_state_targets"], dtype=np.float32)
    next_state_masks = np.asarray(pack["next_state_masks"], dtype=np.float32)
    raw_rewards = np.asarray(pack.get("raw_rewards", pack["rewards"]), dtype=np.float32)
    rewards_are_sb3_normalized = bool(pack.get("rewards_are_sb3_normalized", False))
    n_steps = int(tokens.shape[0])
    n_envs = int(tokens.shape[1])
    obs_dims = _dim_list(meta, "obs_dims", int(obs_slot_dim), n_envs)
    obs_slot_dims = _dim_list(meta, "obs_slot_dims", int(obs_slot_dim), n_envs)
    action_dims = _dim_list(meta, "action_dims", int(action_slot_dim), n_envs)
    state_dims = _dim_list(meta, "state_dims", int(next_states.shape[-1]), n_envs)
    gym_env_ids = meta.get("gym_env_ids", None)
    env_group_seeds = meta.get("env_group_seeds", None)
    prior_gain_keys = (
        "reward_state_input_gain_fraction",
        "reward_action_input_gain_fraction",
        "reward_noise_input_gain_fraction",
        "reward_state_to_noise_gain_ratio",
        "reward_state_to_action_gain_ratio",
        "reward_state_input_gain_fraction_conditioned_accepted_gain",
        "reward_action_input_gain_fraction_conditioned_accepted_gain",
        "reward_state_to_action_gain_ratio_conditioned_accepted_ratio",
        "reward_state_input_gain_fraction_conditioned_attempt",
        "reward_topology_conditioned_attempt",
        "reward_topology_conditioned_total_candidate_draws",
    )
    prior_terminal_keys = (
        "terminal_reset_enabled",
        "terminal_reset_count_target",
        "gated_terminal_coverage_bucket",
        "gated_terminal_coverage_applied",
        "gated_terminal_coverage_original_target",
        "gated_terminal_coverage_target",
    )
    rows: list[dict[str, Any]] = []
    for env_idx in range(n_envs):
        obs_dim = int(min(obs_dims[env_idx], tokens.shape[-1]))
        obs = tokens[:, env_idx, :obs_dim] if obs_dim > 0 else np.zeros((n_steps, 0), dtype=np.float64)
        reward_idx = int(obs_slot_dims[env_idx])
        reward_token = (
            tokens[:, env_idx, reward_idx]
            if 0 <= reward_idx < int(tokens.shape[-1])
            else np.zeros((n_steps,), dtype=np.float64)
        )
        action_dim = int(min(action_dims[env_idx], actions.shape[-1]))
        active_actions = (
            actions[:, env_idx, :action_dim]
            if action_dim > 0
            else np.zeros((n_steps, 0), dtype=np.float64)
        )
        unit_clipped_active_actions = np.clip(active_actions, -1.0, 1.0)
        active_action_masks = (
            action_masks[:, env_idx, :action_dim]
            if action_dim > 0
            else np.zeros((n_steps, 0), dtype=np.float64)
        )
        action_norm = np.sqrt(np.sum(np.square(active_actions), axis=-1)) if action_dim > 0 else np.zeros((n_steps,))
        unit_clipped_action_norm = (
            np.sqrt(np.sum(np.square(unit_clipped_active_actions), axis=-1))
            if action_dim > 0
            else np.zeros((n_steps,))
        )
        action_delta_norm = (
            np.sqrt(np.sum(np.square(np.diff(active_actions, axis=0)), axis=-1))
            if n_steps > 1 and action_dim > 0
            else np.zeros((0,), dtype=np.float64)
        )
        per_dim_temporal_std = np.std(active_actions, axis=0) if action_dim > 0 else np.zeros((0,), dtype=np.float64)
        active_action_values = (
            active_actions[active_action_masks > 0.5]
            if active_actions.size and active_action_masks.size
            else active_actions.reshape(-1)
        )
        raw_unit_clip_excess = np.maximum(np.abs(active_action_values) - 1.0, 0.0)
        state_dim = int(min(state_dims[env_idx], next_states.shape[-1]))
        state = next_states[:, env_idx, :state_dim] if state_dim > 0 else np.zeros((n_steps, 0), dtype=np.float64)
        mask = next_state_masks[:, env_idx, :state_dim] > 0.5 if state_dim > 0 else np.zeros_like(state, dtype=bool)
        valid_state = state[mask]
        next_state_rms = float(np.sqrt(np.mean(np.square(valid_state)))) if valid_state.size else None
        step_l2_delta = np.zeros((0,), dtype=np.float64)
        if n_steps > 1 and state_dim > 0:
            valid_pair = mask[1:] & mask[:-1]
            delta = (state[1:] - state[:-1]) * valid_pair
            valid_counts = np.maximum(valid_pair.sum(axis=1), 1)
            step_l2_delta = np.sqrt(np.sum(np.square(delta), axis=1) / valid_counts)
            step_l2_delta = step_l2_delta[np.isfinite(step_l2_delta)]
        done_idx = np.nonzero(dones[:, env_idx])[0]
        reward = rewards[:, env_idx]
        raw_reward = raw_rewards[:, env_idx]
        reward_components, reward_component_meta = _reward_component_arrays_from_pack(
            arm=str(arm),
            reward=raw_reward,
            active_actions=active_actions,
            active_action_masks=active_action_masks,
            meta=meta,
            env_idx=int(env_idx),
        )
        gym_normalized_bounds = meta.get("gym_normalized_executed_action_l2_upper_bounds", None)
        gym_native_bounds = meta.get("gym_native_action_l2_upper_bounds", None)
        gym_normalized_bound = (
            float(gym_normalized_bounds[env_idx])
            if isinstance(gym_normalized_bounds, list) and env_idx < len(gym_normalized_bounds)
            else None
        )
        gym_native_bound = (
            float(gym_native_bounds[env_idx])
            if isinstance(gym_native_bounds, list) and env_idx < len(gym_native_bounds)
            else None
        )
        raw_action_norm_mean = float(np.mean(action_norm))
        unit_clipped_action_norm_mean = float(np.mean(unit_clipped_action_norm))
        action_value_absmax = float(np.max(np.abs(active_actions))) if active_actions.size else None
        row: dict[str, Any] = {
            "arm": str(arm),
            "update_idx": int(update_idx),
            "env_idx": int(env_idx),
            "env_group_digest": meta.get("env_group_digest"),
            "env_group_seed": int(meta.get("env_group_seed", 0)),
            "env_group_env_seed": (
                int(env_group_seeds[env_idx])
                if isinstance(env_group_seeds, list) and env_idx < len(env_group_seeds)
                else None
            ),
            "gym_env_id": (
                str(gym_env_ids[env_idx])
                if isinstance(gym_env_ids, list) and env_idx < len(gym_env_ids)
                else None
            ),
            "obs_dim": int(obs_dims[env_idx]),
            "obs_slot_dim": int(obs_slot_dims[env_idx]),
            "action_dim": int(action_dims[env_idx]),
            "state_dim": int(state_dims[env_idx]),
            "first_done_step": int(done_idx[0]) if done_idx.size else None,
            "done_count": int(dones[:, env_idx].sum()),
            "truncated_count": int(truncated[:, env_idx].sum()),
            "reward_sum_scale": "sb3_normalized" if rewards_are_sb3_normalized else "raw",
            "reward_sum_is_sb3_normalized": bool(rewards_are_sb3_normalized),
            "training_reward_sum": float(np.sum(reward)),
            "training_reward_mean": float(np.mean(reward)),
            "training_reward_std": float(np.std(reward)),
            "training_reward_min": float(np.min(reward)),
            "training_reward_q10": float(np.quantile(reward, 0.10)),
            "training_reward_q50": float(np.quantile(reward, 0.50)),
            "training_reward_q90": float(np.quantile(reward, 0.90)),
            "training_reward_max": float(np.max(reward)),
            "raw_reward_sum": float(np.sum(raw_reward)),
            "raw_reward_mean": float(np.mean(raw_reward)),
            "raw_reward_std": float(np.std(raw_reward)),
            "raw_reward_min": float(np.min(raw_reward)),
            "raw_reward_q10": float(np.quantile(raw_reward, 0.10)),
            "raw_reward_q50": float(np.quantile(raw_reward, 0.50)),
            "raw_reward_q90": float(np.quantile(raw_reward, 0.90)),
            "raw_reward_max": float(np.max(raw_reward)),
            "reward_sum": float(np.sum(reward)),
            "reward_mean": float(np.mean(reward)),
            "reward_std": float(np.std(reward)),
            "reward_min": float(np.min(reward)),
            "reward_q10": float(np.quantile(reward, 0.10)),
            "reward_q50": float(np.quantile(reward, 0.50)),
            "reward_q90": float(np.quantile(reward, 0.90)),
            "reward_max": float(np.max(reward)),
            "input_reward_token_mean": float(np.mean(reward_token)),
            "input_reward_token_std": float(np.std(reward_token)),
            "input_reward_token_absmax": float(np.max(np.abs(reward_token))) if reward_token.size else 0.0,
            "obs_value_mean": float(np.mean(obs)) if obs.size else None,
            "obs_value_std": float(np.std(obs)) if obs.size else None,
            "obs_value_absmax": float(np.max(np.abs(obs))) if obs.size else None,
            "obs_per_dim_std_mean": float(np.mean(np.std(obs, axis=0))) if obs.size else None,
            "obs_per_dim_std_q90": float(np.quantile(np.std(obs, axis=0), 0.90)) if obs.size else None,
            "obs_per_dim_rms_mean": (
                float(np.mean(np.sqrt(np.mean(np.square(obs), axis=0)))) if obs.size else None
            ),
            "action_value_mean": float(np.mean(active_actions)) if active_actions.size else None,
            "action_value_std": float(np.std(active_actions)) if active_actions.size else None,
            "action_value_absmean": float(np.mean(np.abs(active_actions))) if active_actions.size else None,
            "action_value_absmax": action_value_absmax,
            "action_norm_mean": raw_action_norm_mean,
            "action_norm_std": float(np.std(action_norm)),
            "unit_clipped_action_value_absmean": (
                float(np.mean(np.abs(unit_clipped_active_actions))) if unit_clipped_active_actions.size else None
            ),
            "unit_clipped_action_value_absmax": (
                float(np.max(np.abs(unit_clipped_active_actions))) if unit_clipped_active_actions.size else None
            ),
            "unit_clipped_action_norm_mean": unit_clipped_action_norm_mean,
            "unit_clipped_action_norm_std": float(np.std(unit_clipped_action_norm)),
            "gym_normalized_executed_action_l2_upper_bound": gym_normalized_bound,
            "gym_native_action_l2_upper_bound": gym_native_bound,
            "gym_action_norm_bound_source": (
                "runner clips policy action to [-1,1] before Gym Box mapping"
                if gym_normalized_bound is not None
                else None
            ),
            "raw_policy_action_norm_mean_to_gym_bound": (
                raw_action_norm_mean / max(float(gym_normalized_bound), 1e-12)
                if gym_normalized_bound is not None
                else None
            ),
            "unit_clipped_action_norm_mean_to_gym_bound": (
                unit_clipped_action_norm_mean / max(float(gym_normalized_bound), 1e-12)
                if gym_normalized_bound is not None
                else None
            ),
            "raw_policy_action_absmax_to_unit_clip": (
                action_value_absmax if action_value_absmax is not None else None
            ),
            "raw_policy_action_unit_clip_saturation_fraction": (
                float(np.mean(np.abs(active_action_values) > 1.0)) if active_action_values.size else None
            ),
            "raw_policy_action_unit_clip_excess_absmean": (
                float(np.mean(raw_unit_clip_excess)) if raw_unit_clip_excess.size else None
            ),
            "action_temporal_delta_norm_mean": float(np.mean(action_delta_norm)) if action_delta_norm.size else None,
            "action_per_dim_temporal_std_mean": (
                float(np.mean(per_dim_temporal_std)) if per_dim_temporal_std.size else None
            ),
            "action_per_dim_temporal_std_q10": (
                float(np.quantile(per_dim_temporal_std, 0.10)) if per_dim_temporal_std.size else None
            ),
            "action_per_dim_temporal_std_q50": (
                float(np.quantile(per_dim_temporal_std, 0.50)) if per_dim_temporal_std.size else None
            ),
            "action_per_dim_temporal_std_q90": (
                float(np.quantile(per_dim_temporal_std, 0.90)) if per_dim_temporal_std.size else None
            ),
            "action_constant_dim_fraction_std_lt_0p05": (
                float(np.mean(per_dim_temporal_std < 0.05)) if per_dim_temporal_std.size else None
            ),
            "action_constant_dim_fraction_std_lt_0p10": (
                float(np.mean(per_dim_temporal_std < 0.10)) if per_dim_temporal_std.size else None
            ),
            "next_state_rms": next_state_rms,
            "next_state_step_l2_delta_mean": float(np.mean(step_l2_delta)) if step_l2_delta.size else None,
            "next_state_step_l2_delta_q90": float(np.quantile(step_l2_delta, 0.90)) if step_l2_delta.size else None,
            "next_state_step_l2_delta_to_rms_ratio": (
                float(np.mean(step_l2_delta)) / max(float(next_state_rms), 1e-12)
                if step_l2_delta.size and next_state_rms is not None
                else None
            ),
            "reward_component_logging": reward_component_meta,
        }
        for component_name, component_values in reward_components.items():
            stats = _reward_component_stats(component_values)
            for stat_key, stat_value in stats.items():
                row[f"{component_name}_{stat_key}"] = stat_value
        for key in prior_gain_keys:
            values = meta.get(key)
            if isinstance(values, list) and env_idx < len(values):
                row[key] = values[env_idx]
        for key in prior_terminal_keys:
            values = meta.get(key)
            if isinstance(values, list) and env_idx < len(values):
                row[key] = values[env_idx]
        rows.append(row)
    return rows


def _set_gain_conditioned_sampling(env_cfg: dict[str, Any], gain_min: float) -> None:
    env_cfg["reward_state_input_gain_fraction_conditioned_sampling_enabled"] = True
    env_cfg["reward_state_input_gain_fraction_conditioned_min"] = float(gain_min)
    env_cfg["reward_state_input_gain_fraction_rejection_min"] = 0.0


def _set_topology_conditioned_sampling(
    env_cfg: dict[str, Any],
    *,
    state_gain_min: float,
    action_gain_min: float,
    state_to_action_ratio_max: float,
    max_attempts: int,
) -> None:
    env_cfg["reward_topology_conditioned_sampling_enabled"] = True
    env_cfg["reward_state_input_gain_fraction_conditioned_min"] = float(state_gain_min)
    env_cfg["reward_action_input_gain_fraction_conditioned_min"] = float(action_gain_min)
    env_cfg["reward_state_to_action_gain_ratio_conditioned_max"] = float(state_to_action_ratio_max)
    env_cfg["reward_topology_conditioned_sampling_max_attempts"] = int(max_attempts)
    env_cfg["reward_state_input_gain_fraction_rejection_min"] = 0.0


def _build_algo(
    *,
    cfg: dict[str, Any],
    env_cfg: dict[str, Any],
    model_override=None,
    frozen_h: dict[str, Any] | None,
    frozen_h_list: list[dict[str, Any]] | None,
    frozen_h_list_env_seeds: list[int] | None,
    prior_mode: str,
    device_obj: torch.device,
    num_features: int,
    n_envs: int,
    n_steps: int,
    build_seed: int,
    fixed_single_eval_pos: int | None = 64,
    sb3_reward_normalization_enabled: bool,
    sb3_observation_normalization_enabled: bool,
    sb3_observation_normalization_clip: float,
    sb3_observation_normalization_epsilon: float,
    gain_min: float,
    topology_state_gain_min: float,
    topology_action_gain_min: float,
    topology_state_to_action_ratio_max: float,
    topology_max_attempts: int,
    fixed_env_group_across_updates: bool,
    ppo_learning_rate: float,
    ppo_batch_size: int | None,
    ppo_n_epochs: int,
    ppo_vf_coef: float,
    ppo_normalize_advantage: bool,
    ppo_target_kl: float | None,
):
    _seed_all(int(build_seed))
    if model_override is None:
        _, model, _, _ = get_model(copy.deepcopy(cfg), device=str(device_obj), should_train=False, verbose=False)
    else:
        model = model_override
    model.to(device_obj)
    model.train()

    prior_cfg = copy.deepcopy(env_cfg)
    prior_mode_norm = str(prior_mode).strip().lower()
    if prior_mode_norm == "sampled_gain":
        _set_gain_conditioned_sampling(prior_cfg, float(gain_min))
    elif prior_mode_norm == "sampled_topology":
        _set_topology_conditioned_sampling(
            prior_cfg,
            state_gain_min=float(topology_state_gain_min),
            action_gain_min=float(topology_action_gain_min),
            state_to_action_ratio_max=float(topology_state_to_action_ratio_max),
            max_attempts=int(topology_max_attempts),
        )
    elif prior_mode_norm in GATED_REWARD_PATH_MILESTONES:
        _set_topology_conditioned_sampling(
            prior_cfg,
            state_gain_min=float(topology_state_gain_min),
            action_gain_min=float(topology_action_gain_min),
            state_to_action_ratio_max=float(topology_state_to_action_ratio_max),
            max_attempts=int(topology_max_attempts),
        )
    prior = EnvironmentPrior(prior_cfg)
    if prior_mode_norm in GATED_REWARD_PATH_MILESTONES:
        install_gated_reward_path_balance_milestone(
            prior,
            state_gain_min=float(topology_state_gain_min),
            action_gain_min=float(topology_action_gain_min),
            state_to_action_ratio_max=float(topology_state_to_action_ratio_max),
            max_attempts=int(topology_max_attempts),
            terminal_count_coverage_enabled=(
                prior_mode_norm == GATED_REWARD_PATH_TERMINAL_COVERAGE_MILESTONE
            ),
        )
    if prior_mode_norm == "fixed_frozen":
        if frozen_h is None:
            raise ValueError("prior_mode=fixed_frozen requires a frozen_h.")
        prior._sample_batch_hypers = lambda batch_n, _h=frozen_h: _clone_h_repeated(_h, int(batch_n))
    elif prior_mode_norm == "fixed_frozen_list":
        if frozen_h_list is None:
            raise ValueError("prior_mode=fixed_frozen_list requires a frozen_h_list.")
        fixed_h_list = [copy.deepcopy(h) for h in frozen_h_list]
        if int(len(fixed_h_list)) != int(n_envs):
            raise ValueError(
                "fixed_frozen_list requires n_envs to match the frozen-h list length; "
                f"n_envs={int(n_envs)} list_len={int(len(fixed_h_list))}"
            )
        if frozen_h_list_env_seeds is None:
            fixed_env_seeds = [int(build_seed) + int(env_idx) for env_idx in range(int(n_envs))]
        else:
            fixed_env_seeds = [int(v) for v in frozen_h_list_env_seeds]
            if int(len(fixed_env_seeds)) != int(n_envs):
                raise ValueError(
                    "fixed_frozen_list env seed length must match n_envs; "
                    f"n_envs={int(n_envs)} seed_len={int(len(fixed_env_seeds))}"
                )
        prior._phase2_fixed_env_group_h_list = fixed_h_list
        prior._phase2_fixed_env_group_env_seeds = fixed_env_seeds
        prior.config["reward_state_input_gain_fraction_conditioned_sampling_enabled"] = False
        prior.config["reward_topology_conditioned_sampling_enabled"] = False

        def _sample_fixed_frozen_list(batch_n, _fixed_h_list=fixed_h_list):
            if int(batch_n) != len(_fixed_h_list):
                raise ValueError(
                    "fixed frozen-h environment group requires the same n_envs across all updates; "
                    f"requested={int(batch_n)}, fixed={len(_fixed_h_list)}"
                )
            return [copy.deepcopy(h) for h in _fixed_h_list]

        prior._sample_batch_hypers = _sample_fixed_frozen_list
    elif prior_mode_norm in {"sampled_gain", "sampled_topology", *GATED_REWARD_PATH_MILESTONES}:
        if bool(fixed_env_group_across_updates):
            base_seeds = [int(build_seed) + int(env_idx) for env_idx in range(int(n_envs))]
            fixed_h_list, _unused_env, fixed_env_seeds = (
                prior._sample_batch_hypers_and_environment_family_coarse_batch_conditioned(
                    int(n_envs),
                    device=device_obj,
                    rng_seeds=base_seeds,
                    build_policy_generator=False,
                    return_final_env=False,
                )
            )
            fixed_h_list = [copy.deepcopy(h) for h in fixed_h_list]
            fixed_env_seeds = [int(seed_value) for seed_value in fixed_env_seeds]
            prior.config["reward_state_input_gain_fraction_conditioned_sampling_enabled"] = False
            prior.config["reward_topology_conditioned_sampling_enabled"] = False
            prior._phase2_fixed_env_group_h_list = fixed_h_list
            prior._phase2_fixed_env_group_env_seeds = fixed_env_seeds

            def _sample_fixed_group(batch_n, _fixed_h_list=fixed_h_list):
                if int(batch_n) != len(_fixed_h_list):
                    raise ValueError(
                        f"fixed {prior_mode_norm} environment group requires the same n_envs across all updates; "
                        f"requested={int(batch_n)}, fixed={len(_fixed_h_list)}"
                    )
                return [copy.deepcopy(h) for h in _fixed_h_list]

            prior._sample_batch_hypers = _sample_fixed_group
    else:
        raise ValueError(
            "prior_mode must be 'fixed_frozen', 'fixed_frozen_list', 'sampled_gain', "
            "'sampled_topology', 'gated_reward_path_balance', or "
            "'gated_reward_path_balance_terminal_coverage'."
        )
    prior._phase2_original_sample_single_eval_pos = prior._sample_single_eval_pos
    if fixed_single_eval_pos is not None:
        prior._sample_single_eval_pos = (
            lambda n_samples_arg, rng_seed=None, _value=int(fixed_single_eval_pos): int(_value)
        )
    _apply_boundary_contract_mode_flags(prior, "normal")
    _apply_sep_state_reset_flag(prior, True)

    algo, _callback, vec_env = build_recurrent_ppo(
        model=model,
        env_prior=prior,
        device=str(device_obj),
        num_features=int(num_features),
        n_envs=int(n_envs),
        n_steps=int(n_steps),
        learning_rate=float(ppo_learning_rate),
        batch_size=(
            int(n_steps) * int(n_envs)
            if ppo_batch_size is None
            else int(ppo_batch_size)
        ),
        n_epochs=int(ppo_n_epochs),
        gamma=float(cfg["optimizer"].get("ppo_gamma", 0.98)),
        gae_lambda=float(cfg["optimizer"].get("ppo_gae_lambda", 0.90)),
        clip_range=cfg["optimizer"].get("ppo_clip_range", 0.2),
        clip_range_vf=None,
        normalize_advantage=bool(ppo_normalize_advantage),
        ent_coef=float(cfg["optimizer"].get("ppo_ent_coef", 0.0)),
        vf_coef=float(ppo_vf_coef),
        max_grad_norm=float(cfg["optimizer"].get("ppo_max_grad_norm", 0.5)),
        target_kl=ppo_target_kl,
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
        sb3_reward_normalization_enabled=bool(sb3_reward_normalization_enabled),
        actor_objective_mode="tokenwise",
        verbose=0,
    )
    algo._rwkv_aux_q_weight = 0.0
    algo._rwkv_aux_flow_weight = 0.0
    algo._rwkv_policy_loss_coef = 1.0
    algo._rwkv_sb3_observation_normalization_enabled = bool(sb3_observation_normalization_enabled)
    algo._rwkv_sb3_observation_normalization_clip = float(sb3_observation_normalization_clip)
    algo._rwkv_sb3_observation_normalization_epsilon = float(sb3_observation_normalization_epsilon)
    algo._rwkv_sb3_observation_normalization_mean = np.zeros((int(num_features),), dtype=np.float64)
    algo._rwkv_sb3_observation_normalization_var = np.ones((int(num_features),), dtype=np.float64)
    algo._rwkv_sb3_observation_normalization_count = np.full((int(num_features),), 1e-4, dtype=np.float64)
    algo.rollout_buffer._enable_q_target_cache = False
    algo.rollout_buffer.set_deterministic_batch_plan(True)
    train_outer_snapshot_path = os.environ.get("TICL_PPO_TRAIN_OUTER_BATCH_SNAPSHOT_PATH", "").strip()
    if train_outer_snapshot_path:
        algo._rwkv_train_outer_batch_snapshot_path = train_outer_snapshot_path
        algo._rwkv_train_outer_batch_snapshot_written = False
        target_outer = os.environ.get("TICL_PPO_TRAIN_OUTER_BATCH_SNAPSHOT_TARGET_OUTER_BATCH", "").strip()
        if target_outer:
            algo._rwkv_train_outer_batch_snapshot_target_outer_batch_idx = int(target_outer)
        target_env = os.environ.get("TICL_PPO_TRAIN_OUTER_BATCH_SNAPSHOT_TARGET_ENV", "").strip()
        if target_env:
            algo._rwkv_train_outer_batch_snapshot_target_env_index = int(target_env)
        target_episode = os.environ.get("TICL_PPO_TRAIN_OUTER_BATCH_SNAPSHOT_TARGET_OBJECTIVE_EPISODE", "").strip()
        if target_episode:
            algo._rwkv_train_outer_batch_snapshot_target_objective_episode_index = int(target_episode)
        target_pos_start = os.environ.get("TICL_PPO_TRAIN_OUTER_BATCH_SNAPSHOT_TARGET_POSITION_START", "").strip()
        if target_pos_start:
            algo._rwkv_train_outer_batch_snapshot_target_objective_position_start = int(target_pos_start)
        target_pos_end = os.environ.get("TICL_PPO_TRAIN_OUTER_BATCH_SNAPSHOT_TARGET_POSITION_END", "").strip()
        if target_pos_end:
            algo._rwkv_train_outer_batch_snapshot_target_objective_position_end = int(target_pos_end)
    train_outer_trace_path = os.environ.get("TICL_PPO_TRAIN_OUTER_BATCH_SELECTOR_TRACE_PATH", "").strip()
    if train_outer_trace_path:
        algo._rwkv_train_outer_batch_selector_trace_path = train_outer_trace_path
        algo._rwkv_train_outer_batch_selector_trace_rows = []
        algo._rwkv_train_outer_batch_selector_trace_completed = False
    algo.set_logger(configure_logger(folder=None, format_strings=[]))
    return algo, vec_env


def _set_prior_single_eval_pos_for_update(prior: EnvironmentPrior, n_steps: int, single_eval_pos: int | None) -> int:
    original = getattr(prior, "_phase2_original_sample_single_eval_pos", None)
    if not callable(original):
        original = prior._sample_single_eval_pos
        prior._phase2_original_sample_single_eval_pos = original
    if single_eval_pos is None:
        resolved = int(original(int(n_steps), None))
    else:
        resolved = int(single_eval_pos)
    resolved = int(max(1, min(int(n_steps) - 1, int(resolved))))
    prior._sample_single_eval_pos = lambda n_samples_arg, rng_seed=None, _value=resolved: int(_value)
    return int(resolved)


def _policy_step(
    algo,
    obs_np: np.ndarray,
    starts_np: np.ndarray,
    masks_np: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    obs_t = torch.as_tensor(obs_np, device=algo.device, dtype=torch.float32)
    starts_t = torch.as_tensor(starts_np, device=algo.device, dtype=torch.float32)
    masks_t = torch.as_tensor(masks_np, device=algo.device, dtype=torch.float32)
    dummy_states = algo.policy._dummy_states(int(obs_t.shape[0]))
    with torch.no_grad():
        actions_t, values_t, log_probs_t, _ = algo.policy(
            obs_t,
            dummy_states,
            starts_t,
            deterministic=False,
            action_masks=masks_t,
        )
    return (
        actions_t.detach().cpu().numpy().astype(np.float32, copy=False),
        values_t.detach().cpu().numpy().reshape(-1).astype(np.float32, copy=False),
        log_probs_t.detach().cpu().numpy().reshape(-1).astype(np.float32, copy=False),
    )


def _predict_terminal_values_from_tokens(
    algo,
    token_np: np.ndarray,
    episode_starts_np: np.ndarray | None = None,
) -> np.ndarray:
    token_t = torch.as_tensor(token_np, device=algo.device, dtype=torch.float32)
    if episode_starts_np is None:
        starts_np = np.zeros((int(token_t.shape[0]),), dtype=np.float32)
    else:
        starts_np = np.asarray(episode_starts_np, dtype=np.float32).reshape((int(token_t.shape[0]),))
    starts_t = torch.as_tensor(starts_np, device=algo.device, dtype=torch.float32)
    dummy_states = algo.policy._dummy_states(int(token_t.shape[0]))
    with torch.no_grad():
        values_t = algo.policy.predict_values(token_t, dummy_states, starts_t)
    return values_t.detach().cpu().numpy().reshape(-1).astype(np.float32, copy=False)


def _normalize_reward_step_online(
    algo,
    raw_rewards: np.ndarray,
    dones: np.ndarray,
) -> np.ndarray:
    raw = np.asarray(raw_rewards, dtype=np.float32).reshape(-1)
    done_arr = np.asarray(dones, dtype=bool).reshape(raw.shape)
    if not bool(getattr(algo, "_rwkv_sb3_reward_normalization_enabled", False)):
        return raw.astype(np.float32, copy=True)
    ret_rms = getattr(algo, "_rwkv_sb3_reward_normalization_ret_rms", None)
    if not isinstance(ret_rms, RunningMeanStd):
        ret_rms = RunningMeanStd(shape=())
        algo._rwkv_sb3_reward_normalization_ret_rms = ret_rms
    running_returns = np.asarray(
        getattr(algo, "_rwkv_sb3_reward_normalization_returns", np.zeros_like(raw, dtype=np.float32)),
        dtype=np.float32,
    )
    if running_returns.shape != raw.shape:
        running_returns = np.zeros_like(raw, dtype=np.float32)
    scaled, running_returns = _sb3_vecnormalize_reward_step(
        raw,
        running_returns=running_returns,
        ret_rms=ret_rms,
        dones=done_arr,
        gamma=float(algo.gamma),
        epsilon=float(getattr(algo, "_rwkv_sb3_reward_normalization_epsilon", 1e-8)),
        clip_reward=float(getattr(algo, "_rwkv_sb3_reward_normalization_clip", 10.0)),
    )
    algo._rwkv_sb3_reward_normalization_returns = running_returns
    return np.asarray(scaled, dtype=np.float32).reshape(raw.shape)


def _collect_prior_policy_pack(
    *,
    algo,
    vec_env: EnvironmentPriorPPOBatchVecEnv,
    n_steps: int,
    seed: int,
    num_features: int,
    obs_slot_dim: int,
    action_slot_dim: int,
    next_state_target_dim: int,
    store_next_state_targets: bool = True,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, dict[str, Any]]:
    collect_t0 = time.perf_counter()
    collect_log_every = int(os.environ.get("TICL_PPO_PACK_COLLECT_LOG_EVERY_STEPS", "256"))
    n_envs = int(vec_env.num_envs)
    fixed_env_seeds = getattr(vec_env.env_prior, "_phase2_fixed_env_group_env_seeds", None)
    env_group_seeds = (
        [int(seed_value) for seed_value in fixed_env_seeds]
        if fixed_env_seeds is not None
        else [int(seed) + int(env_idx) for env_idx in range(n_envs)]
    )
    reset_t0 = time.perf_counter()
    obs = vec_env._full_reset_batch(seeds=env_group_seeds)
    reset_elapsed_s = float(time.perf_counter() - reset_t0)
    _pack_phase_log(
        "collect_prior_reset_done "
        f"elapsed_s={reset_elapsed_s:.3f} "
        f"total_s={float(time.perf_counter() - collect_t0):.3f}"
    )
    obs_dims = [int(obs_slot_dim)] * n_envs
    obs_slot_dims = [int(obs_slot_dim)] * n_envs
    if isinstance(getattr(vec_env, "_env", None), dict):
        env_batch_for_dims = vec_env._env
        obs_dims_meta = _tensor_int_vector(env_batch_for_dims, "obs_dim_per_sample")
        obs_slot_dims_meta = _tensor_int_vector(env_batch_for_dims, "obs_slot_dim_per_sample")
        if obs_dims_meta is not None:
            obs_dims = obs_dims_meta[:n_envs]
        if obs_slot_dims_meta is not None:
            obs_slot_dims = obs_slot_dims_meta[:n_envs]
    algo.policy.reset_rollout_cache(n_envs)
    algo.policy.set_training_mode(False)
    episode_starts_t = np.ones((n_envs,), dtype=np.float32)
    obs_norm_acc = _new_obs_norm_stats_acc()
    obs_norm_last_stats: dict[str, Any] = {"enabled": bool(getattr(algo, "_rwkv_sb3_observation_normalization_enabled", False))}

    tokens = np.zeros((int(n_steps), n_envs, int(num_features)), dtype=np.float32)
    actions = np.zeros((int(n_steps), n_envs, int(action_slot_dim)), dtype=np.float32)
    action_masks = np.zeros_like(actions)
    rewards = np.zeros((int(n_steps), n_envs), dtype=np.float32)
    raw_rewards = np.zeros_like(rewards)
    episode_starts = np.zeros_like(rewards)
    dones = np.zeros((int(n_steps), n_envs), dtype=bool)
    terminated = np.zeros_like(dones)
    truncated = np.zeros_like(dones)
    original_vec_next_state_dim = int(getattr(vec_env, "next_state_target_dim", int(next_state_target_dim)))
    vec_env.next_state_target_dim = int(next_state_target_dim if bool(store_next_state_targets) else 0)
    if bool(store_next_state_targets):
        next_state_targets = np.zeros((int(n_steps), n_envs, int(next_state_target_dim)), dtype=np.float32)
        next_state_masks = np.zeros_like(next_state_targets)
    else:
        next_state_targets = None
        next_state_masks = None
    values = np.zeros_like(rewards)
    log_probs = np.zeros_like(rewards)

    original_defer_time_limit_reset = getattr(vec_env, "_defer_time_limit_reset_at_rollout_end", False)
    original_compact_pack_step_infos = getattr(vec_env, "_compact_pack_step_infos", False)
    vec_env._defer_time_limit_reset_at_rollout_end = True
    vec_env._compact_pack_step_infos = True
    collect_probe_max_steps = int(os.environ.get("TICL_PPO_COLLECT_PROBE_MAX_STEPS_EXIT", "0") or "0")
    phase_acc = {
        "mask": 0.0,
        "obs_norm": 0.0,
        "policy": 0.0,
        "env_step": 0.0,
        "reward": 0.0,
        "next_state": 0.0,
        "obs_post": 0.0,
    }
    for step_idx in range(int(n_steps)):
        step_t0 = time.perf_counter()
        phase_t0 = time.perf_counter()
        masks = vec_env.action_masks().astype(np.float32, copy=False)
        phase_acc["mask"] += float(time.perf_counter() - phase_t0)
        phase_t0 = time.perf_counter()
        obs_for_policy, obs_norm_last_stats = _apply_observation_normalization_if_enabled(
            algo,
            np.asarray(obs, dtype=np.float32),
            obs_dims=obs_dims,
            obs_slot_dim=int(obs_slot_dim),
            stats_acc=obs_norm_acc,
        )
        phase_acc["obs_norm"] += float(time.perf_counter() - phase_t0)
        phase_t0 = time.perf_counter()
        action, value, log_prob = _policy_step(algo, obs_for_policy, episode_starts_t, masks)
        phase_acc["policy"] += float(time.perf_counter() - phase_t0)
        tokens[step_idx] = np.asarray(obs_for_policy, dtype=np.float32)
        actions[step_idx] = action
        action_masks[step_idx] = masks
        episode_starts[step_idx] = episode_starts_t
        values[step_idx] = value
        log_probs[step_idx] = log_prob
        phase_t0 = time.perf_counter()
        vec_env.step_async(action)
        obs, reward, done, infos = vec_env.step_wait()
        phase_acc["env_step"] += float(time.perf_counter() - phase_t0)
        phase_t0 = time.perf_counter()
        done_arr = np.asarray(done, dtype=bool)
        raw_reward_arr = np.asarray(reward, dtype=np.float32)
        truncated_cached = getattr(vec_env, "_last_truncated_np", None)
        if truncated_cached is not None and tuple(np.asarray(truncated_cached).shape) == (n_envs,):
            truncated_arr = np.asarray(truncated_cached, dtype=bool)
        else:
            truncated_arr = np.asarray([bool(info.get("TimeLimit.truncated", False)) for info in infos], dtype=bool)
        scaled_reward_arr = _normalize_reward_step_online(algo, raw_reward_arr, done_arr)
        if bool(truncated_arr.any()):
            terminal_tokens = np.asarray(obs, dtype=np.float32).copy()
            terminal_observations = getattr(vec_env, "_last_terminal_observation_np", None)
            if (
                terminal_observations is not None
                and np.asarray(terminal_observations).ndim == 2
                and int(np.asarray(terminal_observations).shape[0]) == n_envs
            ):
                terminal_obs_arr = np.asarray(terminal_observations, dtype=np.float32)
                copy_dim = int(min(int(terminal_obs_arr.shape[1]), int(num_features)))
                terminal_tokens[truncated_arr, :copy_dim] = terminal_obs_arr[truncated_arr, :copy_dim]
            else:
                for env_idx, info in enumerate(infos):
                    if bool(truncated_arr[env_idx]) and info.get("terminal_observation") is not None:
                        terminal_observation = np.asarray(info["terminal_observation"], dtype=np.float32).reshape(-1)
                        copy_dim = int(min(int(terminal_observation.shape[0]), int(num_features)))
                        terminal_tokens[env_idx, :copy_dim] = terminal_observation[:copy_dim]
            terminal_tokens[:, int(obs_slot_dim)] = scaled_reward_arr
            terminal_tokens = _apply_observation_normalization_snapshot(
                algo,
                terminal_tokens,
                obs_dims=obs_dims,
                obs_slot_dim=int(obs_slot_dim),
            )
            terminal_values = _predict_terminal_values_from_tokens(
                algo,
                terminal_tokens,
                np.zeros((n_envs,), dtype=np.float32),
            )
            scaled_reward_arr[truncated_arr] = (
                scaled_reward_arr[truncated_arr] + float(algo.gamma) * terminal_values[truncated_arr]
            )
        rewards[step_idx] = scaled_reward_arr
        raw_rewards[step_idx] = raw_reward_arr
        dones[step_idx] = done_arr
        truncated[step_idx] = truncated_arr
        terminated_cached = getattr(vec_env, "_last_terminated_np", None)
        if terminated_cached is not None and tuple(np.asarray(terminated_cached).shape) == (n_envs,):
            terminated[step_idx] = np.asarray(terminated_cached, dtype=bool)
        else:
            terminated[step_idx] = dones[step_idx] & ~truncated[step_idx]
        phase_acc["reward"] += float(time.perf_counter() - phase_t0)
        phase_t0 = time.perf_counter()
        if bool(store_next_state_targets):
            batch_next_state_targets = getattr(vec_env, "_last_next_state_target_np", None)
            batch_next_state_masks = getattr(vec_env, "_last_next_state_mask_np", None)
            if (
                batch_next_state_targets is not None
                and batch_next_state_masks is not None
                and tuple(np.asarray(batch_next_state_targets).shape) == (n_envs, int(next_state_target_dim))
                and tuple(np.asarray(batch_next_state_masks).shape) == (n_envs, int(next_state_target_dim))
            ):
                next_state_targets[step_idx] = np.asarray(batch_next_state_targets, dtype=np.float32)
                next_state_masks[step_idx] = np.asarray(batch_next_state_masks, dtype=np.float32)
            else:
                for env_idx, info in enumerate(infos):
                    target = np.asarray(info.get("next_state_target"), dtype=np.float32).reshape(-1)
                    mask = np.asarray(info.get("next_state_mask"), dtype=np.float32).reshape(-1)
                    copy_dim = int(min(target.shape[0], int(next_state_target_dim)))
                    next_state_targets[step_idx, env_idx, :copy_dim] = target[:copy_dim]
                    next_state_masks[step_idx, env_idx, :copy_dim] = mask[:copy_dim]
        phase_acc["next_state"] += float(time.perf_counter() - phase_t0)
        phase_t0 = time.perf_counter()
        obs = np.asarray(obs, dtype=np.float32).copy()
        if not bool(truncated_arr.any()) and int(obs.shape[1]) > int(obs_slot_dim):
            obs[:, int(obs_slot_dim)] = scaled_reward_arr
        episode_starts_t = dones[step_idx].astype(np.float32, copy=False)
        phase_acc["obs_post"] += float(time.perf_counter() - phase_t0)
        if (
            bool(_pack_phase_logging_enabled())
            and int(collect_log_every) > 0
            and ((int(step_idx) + 1) % int(collect_log_every) == 0 or int(step_idx) + 1 == int(n_steps))
        ):
            _pack_phase_log(
                "collect_prior_progress "
                f"step={int(step_idx) + 1}/{int(n_steps)} "
                f"step_s={float(time.perf_counter() - step_t0):.3f} "
                f"elapsed_s={float(time.perf_counter() - collect_t0):.1f} "
                f"avg_mask_s={float(phase_acc['mask'] / float(step_idx + 1)):.4f} "
                f"avg_obs_norm_s={float(phase_acc['obs_norm'] / float(step_idx + 1)):.4f} "
                f"avg_policy_s={float(phase_acc['policy'] / float(step_idx + 1)):.4f} "
                f"avg_env_step_s={float(phase_acc['env_step'] / float(step_idx + 1)):.4f} "
                f"avg_reward_s={float(phase_acc['reward'] / float(step_idx + 1)):.4f} "
                f"avg_next_state_s={float(phase_acc['next_state'] / float(step_idx + 1)):.4f} "
                f"avg_obs_post_s={float(phase_acc['obs_post'] / float(step_idx + 1)):.4f}"
            )
        if int(collect_probe_max_steps) > 0 and int(step_idx) + 1 >= int(collect_probe_max_steps):
            elapsed_s = float(time.perf_counter() - collect_t0)
            measured_steps = int(step_idx) + 1
            step_elapsed_s = max(0.0, elapsed_s - float(reset_elapsed_s))
            projected_s = float(reset_elapsed_s) + (
                float(int(n_steps)) * step_elapsed_s / max(1, measured_steps)
            )
            vec_env._defer_time_limit_reset_at_rollout_end = original_defer_time_limit_reset
            vec_env._compact_pack_step_infos = original_compact_pack_step_infos
            _pack_phase_log(
                "collect_prior_probe_exit "
                f"step={measured_steps}/{int(n_steps)} "
                f"elapsed_s={elapsed_s:.1f} "
                f"projected_collect_s={projected_s:.1f} "
                f"avg_policy_s={float(phase_acc['policy'] / float(measured_steps)):.4f} "
                f"avg_env_step_s={float(phase_acc['env_step'] / float(measured_steps)):.4f} "
                f"avg_obs_norm_s={float(phase_acc['obs_norm'] / float(measured_steps)):.4f}"
            )
            raise SystemExit(0)

    final_obs_for_value = _apply_observation_normalization_snapshot(
        algo,
        np.asarray(obs, dtype=np.float32),
        obs_dims=obs_dims,
        obs_slot_dim=int(obs_slot_dim),
    )
    last_values = _predict_terminal_values_from_tokens(algo, final_obs_for_value, episode_starts_t)
    vec_env._defer_time_limit_reset_at_rollout_end = original_defer_time_limit_reset
    vec_env._compact_pack_step_infos = original_compact_pack_step_infos
    _pack_phase_log(
        "collect_prior_done "
        f"elapsed_s={float(time.perf_counter() - collect_t0):.1f} "
        f"n_steps={int(n_steps)} n_envs={int(n_envs)}"
    )

    pack = {
        "tokens": tokens,
        "actions": actions,
        "action_masks": action_masks,
        "rewards": rewards,
        "raw_rewards": raw_rewards,
        "rewards_are_sb3_normalized": bool(getattr(algo, "_rwkv_sb3_reward_normalization_enabled", False)),
        "episode_starts": episode_starts,
        "dones": dones,
        "terminated": terminated,
        "truncated": truncated,
        "next_state_targets_stored": bool(store_next_state_targets),
        "last_values": last_values,
    }
    if bool(store_next_state_targets):
        pack["next_state_targets"] = next_state_targets
        pack["next_state_masks"] = next_state_masks
    vec_env.next_state_target_dim = int(original_vec_next_state_dim)
    meta = {
        "collector": "prior_policy_pack",
        "env_seed": int(seed),
        "env_group_seed": int(seed),
        "env_group_seeds": [int(v) for v in env_group_seeds],
        "env_group_digest": _stable_digest(
            {
                "h_list": getattr(vec_env, "_h_list", None),
                "env_group_seeds": [int(v) for v in env_group_seeds],
            }
        ),
        "n_envs": int(n_envs),
        "obs_normalization": _finalize_obs_norm_stats_acc(obs_norm_acc, obs_norm_last_stats),
    }
    _add_prior_reward_component_meta_from_h_list(
        meta,
        getattr(vec_env.env_prior, "_phase2_fixed_env_group_h_list", None),
        int(n_envs),
    )
    if isinstance(getattr(vec_env, "_env", None), dict):
        env_batch = vec_env._env
        for key in (
            "obs_dim_per_sample",
            "obs_slot_dim_per_sample",
            "action_dim_per_sample",
            "action_slot_dim_per_sample",
            "state_dim_per_sample",
        ):
            meta_values = _tensor_int_vector(env_batch, key)
            if meta_values is not None:
                public_key = {
                    "obs_dim_per_sample": "obs_dims",
                    "obs_slot_dim_per_sample": "obs_slot_dims",
                    "action_dim_per_sample": "action_dims",
                    "action_slot_dim_per_sample": "action_slot_dims",
                    "state_dim_per_sample": "state_dims",
                }[key]
                meta[public_key] = meta_values
        for key in (
            "reward_state_input_gain_fraction",
            "reward_action_input_gain_fraction",
            "reward_noise_input_gain_fraction",
            "reward_state_to_noise_gain_ratio",
            "reward_state_to_action_gain_ratio",
            "reward_state_input_gain_fraction_conditioned_accepted_gain",
            "reward_state_input_gain_fraction_conditioned_attempt",
        ):
            meta_values = _tensor_vector(env_batch, key)
            if meta_values is not None:
                meta[key] = meta_values
        for key in (
            "ctrl_reward_weight",
            "ctrl_reward_enabled",
            "survival_reward_weight",
            "survival_reward_enabled",
            "reward_clip",
            "terminal_reset_enabled",
            "terminal_reset_count_target",
            "gated_terminal_coverage_applied",
            "gated_terminal_coverage_original_target",
            "gated_terminal_coverage_target",
        ):
            if key not in meta:
                meta_values = _tensor_vector(env_batch, key)
                if meta_values is not None:
                    meta[key] = meta_values
    return pack, values, log_probs, meta


def _collect_gym_policy_pack(
    *,
    algo,
    gym_env_ids: list[str],
    n_steps: int,
    seed: int,
    num_features: int,
    obs_slot_dim: int,
    action_slot_dim: int,
    next_state_target_dim: int,
    single_eval_pos: int,
    terminal_token_enabled: bool,
    device_obj: torch.device,
    store_next_state_targets: bool = True,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, dict[str, Any]]:
    import gymnasium as gym
    from gymnasium import spaces

    envs = [gym.make(env_id) for env_id in gym_env_ids]
    try:
        n_envs = int(len(envs))
        obs_dims: list[int] = []
        action_dims: list[int] = []
        lows: list[np.ndarray] = []
        highs: list[np.ndarray] = []
        obs_rows = []
        for env_idx, env in enumerate(envs):
            if not isinstance(env.observation_space, spaces.Box):
                raise ValueError(f"Gym observation space must be Box, got {env.observation_space!r}")
            if not isinstance(env.action_space, spaces.Box):
                raise ValueError(f"Gym action space must be Box, got {env.action_space!r}")
            obs_dim = int(np.asarray(env.observation_space.shape, dtype=np.int64).prod())
            low, high, action_dim = _box_bounds(env.action_space)
            if obs_dim > int(obs_slot_dim):
                raise ValueError(f"{gym_env_ids[env_idx]} obs_dim={obs_dim} exceeds obs_slot_dim={obs_slot_dim}.")
            if action_dim > int(action_slot_dim):
                raise ValueError(
                    f"{gym_env_ids[env_idx]} action_dim={action_dim} exceeds action_slot_dim={action_slot_dim}."
                )
            obs, _info = env.reset(seed=int(seed) + int(env_idx))
            obs_rows.append(np.asarray(obs, dtype=np.float32).reshape(-1))
            obs_dims.append(obs_dim)
            action_dims.append(action_dim)
            lows.append(low)
            highs.append(high)

        algo.policy.reset_rollout_cache(n_envs)
        algo.policy.set_training_mode(False)
        obs_t = np.zeros((n_envs, int(obs_slot_dim)), dtype=np.float32)
        for env_idx, obs in enumerate(obs_rows):
            obs_t[env_idx, : obs.shape[0]] = obs
        prev_action = np.zeros((n_envs, int(action_slot_dim)), dtype=np.float32)
        prev_reward = np.zeros((n_envs,), dtype=np.float32)
        prev_reward_mask = np.ones((n_envs,), dtype=np.float32)
        prev_terminal = np.zeros((n_envs,), dtype=np.float32)
        episode_starts_t = np.ones((n_envs,), dtype=np.float32)
        obs_norm_acc = _new_obs_norm_stats_acc()
        obs_norm_last_stats: dict[str, Any] = {
            "enabled": bool(getattr(algo, "_rwkv_sb3_observation_normalization_enabled", False))
        }

        tokens = np.zeros((int(n_steps), n_envs, int(num_features)), dtype=np.float32)
        actions = np.zeros((int(n_steps), n_envs, int(action_slot_dim)), dtype=np.float32)
        action_masks = np.zeros_like(actions)
        rewards = np.zeros((int(n_steps), n_envs), dtype=np.float32)
        raw_rewards = np.zeros_like(rewards)
        episode_starts = np.zeros_like(rewards)
        dones = np.zeros((int(n_steps), n_envs), dtype=bool)
        terminated = np.zeros_like(dones)
        truncated = np.zeros_like(dones)
        if bool(store_next_state_targets):
            next_state_targets = np.zeros((int(n_steps), n_envs, int(next_state_target_dim)), dtype=np.float32)
            next_state_masks = np.zeros_like(next_state_targets)
        else:
            next_state_targets = None
            next_state_masks = None
        values = np.zeros_like(rewards)
        log_probs = np.zeros_like(rewards)

        for step_idx in range(int(n_steps)):
            obs_t_for_policy, obs_norm_last_stats = _apply_observation_normalization_if_enabled(
                algo,
                obs_t,
                obs_dims=obs_dims,
                obs_slot_dim=int(obs_slot_dim),
                stats_acc=obs_norm_acc,
            )
            phase = np.full((n_envs,), float(step_idx >= int(single_eval_pos)), dtype=np.float32)
            env_info = _token_env_info(
                batch_size=n_envs,
                obs_dim=int(obs_slot_dim),
                action_dim=int(action_slot_dim),
                obs_slot_dim=int(obs_slot_dim),
                action_slot_dim=int(action_slot_dim),
                terminal_token_enabled=bool(terminal_token_enabled),
                phase_t=phase,
                terminal_t=prev_terminal,
                device=device_obj,
            )
            token_t = build_rlpfn_obs_token_batch_from_components(
                obs_t=torch.as_tensor(obs_t_for_policy, device=device_obj, dtype=torch.float32),
                action_t=torch.as_tensor(prev_action, device=device_obj, dtype=torch.float32),
                reward_t=torch.as_tensor(prev_reward, device=device_obj, dtype=torch.float32),
                reward_mask_t=torch.as_tensor(prev_reward_mask, device=device_obj, dtype=torch.float32),
                env_info=env_info,
                num_features=int(num_features),
            ).detach().cpu().numpy().astype(np.float32, copy=False)
            mask_pad = np.zeros((n_envs, int(action_slot_dim)), dtype=np.float32)
            for env_idx, action_dim in enumerate(action_dims):
                mask_pad[env_idx, : int(action_dim)] = 1.0
            action, value, log_prob = _policy_step(algo, token_t, episode_starts_t, mask_pad)
            tokens[step_idx] = token_t
            actions[step_idx] = action
            action_masks[step_idx] = mask_pad
            episode_starts[step_idx] = episode_starts_t
            values[step_idx] = value
            log_probs[step_idx] = log_prob

            next_obs_rows = []
            next_episode_starts = np.zeros((n_envs,), dtype=np.float32)
            next_terminal_token = np.zeros((n_envs,), dtype=np.float32)
            raw_reward_step = np.zeros((n_envs,), dtype=np.float32)
            done_step = np.zeros((n_envs,), dtype=bool)
            terminal_obs_step = np.zeros((n_envs, int(obs_slot_dim)), dtype=np.float32)
            for env_idx, env in enumerate(envs):
                active = np.clip(action[env_idx, : action_dims[env_idx]], -1.0, 1.0)
                gym_action = _normalize_to_box_action(active, lows[env_idx], highs[env_idx]).reshape(env.action_space.shape)
                next_obs, reward, term, trunc, _info = env.step(gym_action)
                horizon_trunc = bool(step_idx + 1 >= int(n_steps))
                done = bool(term or trunc or horizon_trunc)
                raw_reward_step[env_idx] = float(reward)
                done_step[env_idx] = bool(done)
                terminated[step_idx, env_idx] = bool(term)
                truncated[step_idx, env_idx] = bool((trunc or horizon_trunc) and not term)
                dones[step_idx, env_idx] = bool(done)
                next_state = np.asarray(next_obs, dtype=np.float32).reshape(-1)
                terminal_obs_step[env_idx, : min(next_state.shape[0], int(obs_slot_dim))] = next_state[: int(obs_slot_dim)]
                if bool(store_next_state_targets):
                    copy_dim = int(min(next_state.shape[0], int(next_state_target_dim)))
                    next_state_targets[step_idx, env_idx, :copy_dim] = next_state[:copy_dim]
                    next_state_masks[step_idx, env_idx, :copy_dim] = 1.0
                if done and step_idx + 1 < int(n_steps):
                    reset_obs, _reset_info = env.reset(seed=int(seed) + 100000 + step_idx * n_envs + env_idx)
                    next_obs_rows.append(np.asarray(reset_obs, dtype=np.float32).reshape(-1))
                    next_episode_starts[env_idx] = 1.0
                    next_terminal_token[env_idx] = 0.0
                else:
                    next_obs_rows.append(next_state)
                    next_terminal_token[env_idx] = float(term)
            scaled_reward_step = _normalize_reward_step_online(algo, raw_reward_step, done_step)
            if bool(truncated[step_idx].any()):
                terminal_obs_norm = _apply_observation_normalization_snapshot(
                    algo,
                    terminal_obs_step,
                    obs_dims=obs_dims,
                    obs_slot_dim=int(obs_slot_dim),
                )
                phase_next = np.full(
                    (n_envs,),
                    float((int(step_idx) + 1) >= int(single_eval_pos)),
                    dtype=np.float32,
                )
                terminal_info = _token_env_info(
                    batch_size=n_envs,
                    obs_dim=int(obs_slot_dim),
                    action_dim=int(action_slot_dim),
                    obs_slot_dim=int(obs_slot_dim),
                    action_slot_dim=int(action_slot_dim),
                    terminal_token_enabled=bool(terminal_token_enabled),
                    phase_t=phase_next,
                    terminal_t=terminated[step_idx].astype(np.float32, copy=False),
                    device=device_obj,
                )
                terminal_tokens = build_rlpfn_obs_token_batch_from_components(
                    obs_t=torch.as_tensor(terminal_obs_norm, device=device_obj, dtype=torch.float32),
                    action_t=torch.as_tensor(action, device=device_obj, dtype=torch.float32),
                    reward_t=torch.as_tensor(scaled_reward_step, device=device_obj, dtype=torch.float32),
                    reward_mask_t=torch.ones((n_envs,), device=device_obj, dtype=torch.float32),
                    env_info=terminal_info,
                    num_features=int(num_features),
                ).detach().cpu().numpy().astype(np.float32, copy=False)
                terminal_values = _predict_terminal_values_from_tokens(
                    algo,
                    terminal_tokens,
                    np.zeros((n_envs,), dtype=np.float32),
                )
                trunc_mask = truncated[step_idx]
                scaled_reward_step[trunc_mask] = (
                    scaled_reward_step[trunc_mask] + float(algo.gamma) * terminal_values[trunc_mask]
                )
            rewards[step_idx] = scaled_reward_step
            raw_rewards[step_idx] = raw_reward_step
            obs_t.fill(0.0)
            for env_idx, next_obs in enumerate(next_obs_rows):
                obs_t[env_idx, : min(next_obs.shape[0], int(obs_slot_dim))] = next_obs[: int(obs_slot_dim)]
            prev_action = action
            prev_reward = np.where(dones[step_idx], 0.0, scaled_reward_step).astype(np.float32, copy=False)
            prev_reward_mask.fill(1.0)
            prev_terminal = next_terminal_token
            episode_starts_t = next_episode_starts

        final_obs_for_policy = _apply_observation_normalization_snapshot(
            algo,
            obs_t,
            obs_dims=obs_dims,
            obs_slot_dim=int(obs_slot_dim),
        )
        final_phase = np.full((n_envs,), float(int(n_steps) >= int(single_eval_pos)), dtype=np.float32)
        final_env_info = _token_env_info(
            batch_size=n_envs,
            obs_dim=int(obs_slot_dim),
            action_dim=int(action_slot_dim),
            obs_slot_dim=int(obs_slot_dim),
            action_slot_dim=int(action_slot_dim),
            terminal_token_enabled=bool(terminal_token_enabled),
            phase_t=final_phase,
            terminal_t=prev_terminal,
            device=device_obj,
        )
        final_tokens = build_rlpfn_obs_token_batch_from_components(
            obs_t=torch.as_tensor(final_obs_for_policy, device=device_obj, dtype=torch.float32),
            action_t=torch.as_tensor(prev_action, device=device_obj, dtype=torch.float32),
            reward_t=torch.as_tensor(prev_reward, device=device_obj, dtype=torch.float32),
            reward_mask_t=torch.as_tensor(prev_reward_mask, device=device_obj, dtype=torch.float32),
            env_info=final_env_info,
            num_features=int(num_features),
        ).detach().cpu().numpy().astype(np.float32, copy=False)
        last_values = _predict_terminal_values_from_tokens(algo, final_tokens, episode_starts_t)

        pack = {
            "tokens": tokens,
            "actions": actions,
            "action_masks": action_masks,
            "rewards": rewards,
            "raw_rewards": raw_rewards,
            "rewards_are_sb3_normalized": bool(getattr(algo, "_rwkv_sb3_reward_normalization_enabled", False)),
            "episode_starts": episode_starts,
            "dones": dones,
            "terminated": terminated,
            "truncated": truncated,
            "next_state_targets_stored": bool(store_next_state_targets),
            "last_values": last_values,
        }
        if bool(store_next_state_targets):
            pack["next_state_targets"] = next_state_targets
            pack["next_state_masks"] = next_state_masks
        meta = {
            "collector": "gym_policy_pack",
            "gym_env_ids": list(gym_env_ids),
            "env_seed": int(seed),
            "env_group_seed": int(seed),
            "env_group_digest": _stable_digest(
                {
                    "gym_env_ids": list(gym_env_ids),
                    "reset_seeds": [int(seed) + int(env_idx) for env_idx in range(n_envs)],
                    "obs_dims": [int(v) for v in obs_dims],
                    "action_dims": [int(v) for v in action_dims],
                }
            ),
            "obs_dims": [int(v) for v in obs_dims],
            "action_dims": [int(v) for v in action_dims],
            "gym_normalized_executed_action_l2_upper_bounds": [
                float(np.sqrt(float(action_dim))) for action_dim in action_dims
            ],
            "gym_native_action_l2_upper_bounds": [
                float(np.linalg.norm(np.maximum(np.abs(low), np.abs(high))))
                for low, high in zip(lows, highs)
            ],
            "obs_normalization": _finalize_obs_norm_stats_acc(obs_norm_acc, obs_norm_last_stats),
        }
        return pack, values, log_probs, meta
    finally:
        for env in envs:
            env.close()


def _apply_reward_normalization_if_enabled(algo, pack: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if bool(pack.get("rewards_are_sb3_normalized", False)):
        ret_rms = getattr(algo, "_rwkv_sb3_reward_normalization_ret_rms", None)
        rewards = np.asarray(pack["rewards"], dtype=np.float32)
        raw_rewards = np.asarray(pack.get("raw_rewards", pack["rewards"]), dtype=np.float32)
        stats: dict[str, Any] = {
            "enabled": bool(getattr(algo, "_rwkv_sb3_reward_normalization_enabled", False)),
            "already_applied_online": True,
            "raw_reward": _finite_stats(raw_rewards),
            "buffer_reward": _finite_stats(rewards),
        }
        if isinstance(ret_rms, RunningMeanStd):
            stats.update(
                {
                    "ret_rms_mean": float(np.asarray(ret_rms.mean).reshape(())),
                    "ret_rms_var": float(np.asarray(ret_rms.var).reshape(())),
                    "ret_rms_count": float(ret_rms.count),
                }
            )
        return pack, stats
    if not bool(getattr(algo, "_rwkv_sb3_reward_normalization_enabled", False)):
        return pack, {"enabled": False}
    out = dict(pack)
    rewards = np.asarray(pack["rewards"], dtype=np.float32)
    dones = np.asarray(pack["dones"], dtype=bool)
    scaled = np.zeros_like(rewards, dtype=np.float32)
    ret_rms = getattr(algo, "_rwkv_sb3_reward_normalization_ret_rms", None)
    if not isinstance(ret_rms, RunningMeanStd):
        ret_rms = RunningMeanStd(shape=())
        algo._rwkv_sb3_reward_normalization_ret_rms = ret_rms
    running_returns = np.asarray(
        getattr(algo, "_rwkv_sb3_reward_normalization_returns", np.zeros((rewards.shape[1],), dtype=np.float32)),
        dtype=np.float32,
    )
    for step_idx in range(int(rewards.shape[0])):
        scaled[step_idx], running_returns = _sb3_vecnormalize_reward_step(
            rewards[step_idx],
            running_returns=running_returns,
            ret_rms=ret_rms,
            dones=dones[step_idx],
            gamma=float(algo.gamma),
            epsilon=float(getattr(algo, "_rwkv_sb3_reward_normalization_epsilon", 1e-8)),
            clip_reward=float(getattr(algo, "_rwkv_sb3_reward_normalization_clip", 10.0)),
        )
    algo._rwkv_sb3_reward_normalization_returns = running_returns
    out["rewards"] = scaled
    stats = {
        "enabled": True,
        "ret_rms_mean": float(np.asarray(ret_rms.mean).reshape(())),
        "ret_rms_var": float(np.asarray(ret_rms.var).reshape(())),
        "ret_rms_count": float(ret_rms.count),
        "raw_reward": _finite_stats(rewards),
        "buffer_reward": _finite_stats(scaled),
    }
    return out, stats


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_json_safe(row), sort_keys=True) + "\n")


def _save_runner_checkpoint(
    *,
    algo,
    path: Path,
    row: dict[str, Any],
    include_optimizer: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ret_rms = getattr(algo, "_rwkv_sb3_reward_normalization_ret_rms", None)
    reward_norm = {
        "enabled": bool(getattr(algo, "_rwkv_sb3_reward_normalization_enabled", False)),
        "returns": np.asarray(
            getattr(algo, "_rwkv_sb3_reward_normalization_returns", np.asarray([], dtype=np.float32)),
            dtype=np.float32,
        ),
    }
    if isinstance(ret_rms, RunningMeanStd):
        reward_norm.update(
            {
                "ret_rms_mean": np.asarray(ret_rms.mean),
                "ret_rms_var": np.asarray(ret_rms.var),
                "ret_rms_count": float(ret_rms.count),
            }
        )
    obs_norm = {
        "enabled": bool(getattr(algo, "_rwkv_sb3_observation_normalization_enabled", False)),
        "mean": np.asarray(
            getattr(algo, "_rwkv_sb3_observation_normalization_mean", np.asarray([], dtype=np.float64)),
            dtype=np.float64,
        ),
        "var": np.asarray(
            getattr(algo, "_rwkv_sb3_observation_normalization_var", np.asarray([], dtype=np.float64)),
            dtype=np.float64,
        ),
        "count": np.asarray(
            getattr(algo, "_rwkv_sb3_observation_normalization_count", np.asarray([], dtype=np.float64)),
            dtype=np.float64,
        ),
        "clip": float(getattr(algo, "_rwkv_sb3_observation_normalization_clip", 10.0)),
        "epsilon": float(getattr(algo, "_rwkv_sb3_observation_normalization_epsilon", 1e-8)),
    }
    payload: dict[str, Any] = {
        "checkpoint_format": "phase2_gym_prior_pack_training_runner_policy_debug_v1",
        "arm": str(row.get("arm")),
        "update_idx": int(row.get("update_idx", -1)),
        "n_updates": int(getattr(algo, "_n_updates", 0)),
        "policy_state_dict": algo.policy.state_dict(),
        "reward_norm": reward_norm,
        "obs_norm": obs_norm,
        "row_summary": _json_safe(
            {
                "collector": row.get("collector"),
                "fixed_env_group_across_updates": row.get("fixed_env_group_across_updates"),
                "semantic_probe": row.get("semantic_probe"),
                "bridge_reward_norm": (row.get("bridge") or {}).get("reward_norm"),
                "update_logger": (row.get("update") or {}).get("logger"),
                "param_delta": (row.get("update") or {}).get("param_delta"),
            }
        ),
    }
    optimizer = getattr(algo.policy, "optimizer", None)
    if bool(include_optimizer) and optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, path)


def _dump_fixed_prior_group_if_available(
    *,
    vec_env: EnvironmentPriorPPOBatchVecEnv,
    output_dir: Path,
    arm: str,
) -> dict[str, Any] | None:
    if str(arm) != "prior":
        return None
    fixed_h_list = getattr(vec_env.env_prior, "_phase2_fixed_env_group_h_list", None)
    fixed_env_seeds = getattr(vec_env.env_prior, "_phase2_fixed_env_group_env_seeds", None)
    if fixed_h_list is None:
        return None
    dump_dir = output_dir / "fixed_env_group"
    dump_dir.mkdir(parents=True, exist_ok=True)
    h_list_path = dump_dir / "prior_fixed_h_list.json"
    seeds_path = dump_dir / "prior_fixed_env_seeds.json"
    h_payload = [copy.deepcopy(h) for h in fixed_h_list]
    seed_payload = [] if fixed_env_seeds is None else [int(v) for v in fixed_env_seeds]
    h_list_path.write_text(json.dumps(_json_safe(h_payload), sort_keys=True) + "\n", encoding="utf-8")
    seeds_path.write_text(json.dumps(_json_safe(seed_payload), sort_keys=True) + "\n", encoding="utf-8")
    return {
        "fixed_h_list_path": str(h_list_path),
        "fixed_env_seeds_path": str(seeds_path),
        "fixed_h_count": int(len(h_payload)),
        "fixed_env_seed_count": int(len(seed_payload)),
        "fixed_group_digest": _stable_digest(
            {
                "h_list": h_payload,
                "env_group_seeds": seed_payload,
            }
        ),
    }


def _run_update_from_pack(
    *,
    algo,
    pack: dict[str, np.ndarray],
    values: np.ndarray,
    log_probs: np.ndarray,
    num_features: int,
    action_slot_dim: int,
    next_state_target_dim: int,
    single_eval_pos: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    bridge_t0 = time.perf_counter()
    buffer_pack, reward_norm_stats = _apply_reward_normalization_if_enabled(algo, pack)
    _pack_phase_log(f"reward_normalization_done elapsed_s={float(time.perf_counter() - bridge_t0):.3f}")
    fill_t0 = time.perf_counter()
    buffer_summary = _fill_existing_rollout_buffer_from_pack(
        algo.rollout_buffer,
        buffer_pack,
        num_features=int(num_features),
        action_slot_dim=int(action_slot_dim),
        next_state_target_dim=int(next_state_target_dim),
        single_eval_pos=int(single_eval_pos),
        values_by_step_env=values,
        log_probs_by_step_env=log_probs,
        last_values_by_env=pack.get("last_values"),
        compute_returns=True,
        summary_mode=os.environ.get("TICL_PPO_ROLLOUT_BUFFER_SUMMARY_MODE", "light"),
    )
    _pack_phase_log(
        "rollout_buffer_fill_done "
        f"elapsed_s={float(time.perf_counter() - fill_t0):.3f} "
        f"summary_mode={buffer_summary.get('summary_mode')} "
        f"timings={json.dumps(_json_safe(buffer_summary.get('timings', {})), sort_keys=True)}"
    )
    before = _policy_param_snapshot(algo)
    _pack_phase_log(f"param_snapshot_done elapsed_s={float(time.perf_counter() - bridge_t0):.3f}")
    exception = None
    train_t0 = time.perf_counter()
    try:
        algo.train()
    except Exception as exc:  # pragma: no cover
        exception = repr(exc)
    _pack_phase_log(f"algo_train_returned elapsed_s={float(time.perf_counter() - train_t0):.1f}")
    update = {
        "exception": exception,
        "n_updates": int(getattr(algo, "_n_updates", 0)),
        "logger": _logger_values(algo),
        "param_delta": _param_delta_summary(algo, before),
    }
    return update, {"buffer_summary": buffer_summary, "reward_norm": reward_norm_stats}


def _run_pack_update_once(
    *,
    arm: str,
    algo,
    vec_env: EnvironmentPriorPPOBatchVecEnv,
    gym_env_ids_arg: str,
    output_dir: Path,
    update_idx: int,
    env_seed: int,
    fixed_env_group_across_updates: bool,
    fixed_group_dump: dict[str, Any] | None,
    device_obj: torch.device,
    num_features: int,
    obs_slot_dim: int,
    action_slot_dim: int,
    next_state_target_dim: int,
    n_envs: int,
    n_steps: int,
    single_eval_pos: int | None,
    terminal_token_enabled: bool,
    semantic_probes_enabled: bool,
    semantic_probe_sidecar_enabled: bool,
    semantic_probe_sidecar_every: int,
    checkpoint_every: int,
    checkpoint_include_optimizer: bool,
    force_checkpoint: bool,
    progress_path: Path | None,
) -> tuple[dict[str, Any], str]:
    run_t0 = time.perf_counter()
    sidecar_dir = output_dir / "semantic_sidecars"
    checkpoint_dir = output_dir / "checkpoints"
    effective_single_eval_pos = (
        _set_prior_single_eval_pos_for_update(vec_env.env_prior, int(n_steps), single_eval_pos)
        if str(arm) == "prior"
        else int(max(1, min(int(n_steps) - 1, int(single_eval_pos if single_eval_pos is not None else 1))))
    )
    store_next_state_targets = bool(
        semantic_probes_enabled
        or float(getattr(algo, "_rwkv_aux_flow_weight", 0.0)) > 0.0
        or int(getattr(algo.rollout_buffer, "next_state_dim", 0)) > 0
    )
    update_pack_load_path = _env_path("TICL_PPO_UPDATE_PACK_LOAD_PATH")
    update_pack_save_path = _env_path("TICL_PPO_UPDATE_PACK_SAVE_PATH")
    if update_pack_load_path is not None:
        phase_t0 = time.perf_counter()
        _pack_phase_log(f"load_pack_start path={update_pack_load_path}")
        pack, values, log_probs, meta = _load_update_pack_cache(
            path=update_pack_load_path,
            arm=str(arm),
            update_idx=int(update_idx),
            n_steps=int(n_steps),
            n_envs=int(n_envs),
            num_features=int(num_features),
            action_slot_dim=int(action_slot_dim),
            next_state_target_dim=int(next_state_target_dim),
            single_eval_pos=int(effective_single_eval_pos),
            store_next_state_targets=bool(store_next_state_targets),
        )
        _pack_phase_log(f"load_pack_done elapsed_s={float(time.perf_counter() - phase_t0):.1f}")
    elif str(arm) == "gym":
        phase_t0 = time.perf_counter()
        _pack_phase_log(f"collect_gym_start update_idx={int(update_idx)}")
        pack, values, log_probs, meta = _collect_gym_policy_pack(
            algo=algo,
            gym_env_ids=_normalize_env_id_list(gym_env_ids_arg, int(n_envs)),
            n_steps=int(n_steps),
            seed=int(env_seed),
            num_features=int(num_features),
            obs_slot_dim=int(obs_slot_dim),
            action_slot_dim=int(action_slot_dim),
            next_state_target_dim=int(next_state_target_dim),
            single_eval_pos=int(effective_single_eval_pos),
            terminal_token_enabled=bool(terminal_token_enabled),
            device_obj=device_obj,
            store_next_state_targets=bool(store_next_state_targets),
        )
        _pack_phase_log(f"collect_gym_done elapsed_s={float(time.perf_counter() - phase_t0):.1f}")
    elif str(arm) == "prior":
        phase_t0 = time.perf_counter()
        _pack_phase_log(
            f"collect_prior_start update_idx={int(update_idx)} n_steps={int(n_steps)} n_envs={int(n_envs)}"
        )
        pack, values, log_probs, meta = _collect_prior_policy_pack(
            algo=algo,
            vec_env=vec_env,
            n_steps=int(n_steps),
            seed=int(env_seed),
            num_features=int(num_features),
            obs_slot_dim=int(obs_slot_dim),
            action_slot_dim=int(action_slot_dim),
            next_state_target_dim=int(next_state_target_dim),
            store_next_state_targets=bool(store_next_state_targets),
        )
        _pack_phase_log(f"collect_prior_phase_done elapsed_s={float(time.perf_counter() - phase_t0):.1f}")
    else:
        raise ValueError(f"Unsupported arm={arm!r}")
    meta["single_eval_pos"] = int(effective_single_eval_pos)
    meta["next_state_targets_stored"] = bool(store_next_state_targets)
    meta["rollout_buffer_next_state_dim"] = int(getattr(algo.rollout_buffer, "next_state_dim", 0))
    if update_pack_save_path is not None:
        phase_t0 = time.perf_counter()
        _pack_phase_log(f"save_pack_start path={update_pack_save_path}")
        _save_update_pack_cache(
            path=update_pack_save_path,
            arm=str(arm),
            update_idx=int(update_idx),
            env_seed=int(env_seed),
            n_steps=int(n_steps),
            n_envs=int(n_envs),
            num_features=int(num_features),
            action_slot_dim=int(action_slot_dim),
            next_state_target_dim=int(next_state_target_dim),
            single_eval_pos=int(effective_single_eval_pos),
            store_next_state_targets=bool(store_next_state_targets),
            pack=pack,
            values=values,
            log_probs=log_probs,
            meta=meta,
        )
        _pack_phase_log(f"save_pack_done elapsed_s={float(time.perf_counter() - phase_t0):.1f}")
        if _env_truthy("TICL_PPO_UPDATE_PACK_SAVE_EXIT", default=False):
            print(
                f"[ppo-update-pack-cache-save-exit] path={update_pack_save_path} update_idx={int(update_idx)}",
                flush=True,
            )
            raise SystemExit(0)
    if str(arm) == "prior":
        phase_t0 = time.perf_counter()
        meta["pre_update_env_release"] = _release_prior_vec_env_before_update(vec_env)
        _pack_phase_log(f"prior_env_release_done elapsed_s={float(time.perf_counter() - phase_t0):.3f}")
    phase_t0 = time.perf_counter()
    meta["pre_update_policy_rollout_cache_release"] = _release_policy_rollout_cache_before_update(algo)
    _pack_phase_log(f"policy_rollout_cache_release_done elapsed_s={float(time.perf_counter() - phase_t0):.3f}")
    phase_t0 = time.perf_counter()
    semantic_probe = (
        _semantic_probe_from_pack(
            arm=str(arm),
            pack=pack,
            meta=meta,
            obs_slot_dim=int(obs_slot_dim),
            action_slot_dim=int(action_slot_dim),
        )
        if bool(semantic_probes_enabled)
        else {"enabled": False}
    )
    _pack_phase_log(f"semantic_probe_done elapsed_s={float(time.perf_counter() - phase_t0):.3f}")
    phase_t0 = time.perf_counter()
    reward_component_summary = _rollout_reward_stats_from_pack(
        arm=str(arm),
        pack=pack,
        meta=meta,
        single_eval_pos=int(effective_single_eval_pos),
    )
    _pack_phase_log(f"reward_summary_done elapsed_s={float(time.perf_counter() - phase_t0):.3f}")
    semantic_sidecar_path = None
    if (
        bool(semantic_probes_enabled)
        and bool(semantic_probe_sidecar_enabled)
        and int(semantic_probe_sidecar_every) > 0
        and int(update_idx) % int(semantic_probe_sidecar_every) == 0
    ):
        phase_t0 = time.perf_counter()
        semantic_sidecar_path = sidecar_dir / f"{arm}_update_{int(update_idx):06d}_envs.jsonl"
        _write_jsonl(
            semantic_sidecar_path,
            _per_env_semantic_probe_rows_from_pack(
                arm=str(arm),
                update_idx=int(update_idx),
                pack=pack,
                meta=meta,
                obs_slot_dim=int(obs_slot_dim),
                action_slot_dim=int(action_slot_dim),
            ),
        )
        _pack_phase_log(f"semantic_sidecar_done elapsed_s={float(time.perf_counter() - phase_t0):.3f}")
    phase_t0 = time.perf_counter()
    update, bridge = _run_update_from_pack(
        algo=algo,
        pack=pack,
        values=values,
        log_probs=log_probs,
        num_features=int(num_features),
        action_slot_dim=int(action_slot_dim),
        next_state_target_dim=int(next_state_target_dim),
        single_eval_pos=int(effective_single_eval_pos),
    )
    _pack_phase_log(
        "update_from_pack_done "
        f"elapsed_s={float(time.perf_counter() - phase_t0):.1f} "
        f"total_s={float(time.perf_counter() - run_t0):.1f}"
    )
    pack_summary_mode = str(os.environ.get("TICL_PPO_PACK_SUMMARY_MODE", "light")).strip().lower()
    pack_summary_t0 = time.perf_counter()
    pack_summary = (
        _summarize_pack(pack, label=str(arm))
        if pack_summary_mode in {"full", "complete", "digest"}
        else _summarize_pack_light(pack, label=str(arm))
    )
    _pack_phase_log(
        "pack_summary_done "
        f"elapsed_s={float(time.perf_counter() - pack_summary_t0):.3f} "
        f"summary_mode={pack_summary.get('summary_mode', 'full')}"
    )
    row = {
        "arm": str(arm),
        "update_idx": int(update_idx),
        "env_seed": int(env_seed),
        "single_eval_pos": int(effective_single_eval_pos),
        "fixed_env_group_across_updates": bool(fixed_env_group_across_updates),
        "fixed_group_dump": fixed_group_dump,
        "collector": meta,
        "pack_summary": pack_summary,
        "reward_component_summary": reward_component_summary,
        "semantic_probe": semantic_probe,
        "semantic_probe_sidecar_path": None if semantic_sidecar_path is None else str(semantic_sidecar_path),
        "old_values": _finite_stats(values),
        "old_log_probs": _finite_stats(log_probs),
        "bridge": bridge,
        "update": update,
    }
    checkpoint_path = None
    if int(checkpoint_every) > 0 and (
        int(update_idx) % int(checkpoint_every) == 0
        or bool(force_checkpoint)
        or update.get("exception") is not None
    ):
        checkpoint_path = checkpoint_dir / f"{arm}_update_{int(update_idx):06d}.pt"
        _save_runner_checkpoint(
            algo=algo,
            path=checkpoint_path,
            row=row,
            include_optimizer=bool(checkpoint_include_optimizer),
        )
        row["checkpoint_path"] = str(checkpoint_path)
    if progress_path is not None:
        phase_t0 = time.perf_counter()
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        with progress_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(_json_safe(row), sort_keys=True) + "\n")
        _pack_phase_log(f"progress_row_write_done elapsed_s={float(time.perf_counter() - phase_t0):.3f}")
    del pack, values, log_probs
    phase_t0 = time.perf_counter()
    _trim_host_allocator_after_pack_release()
    _pack_phase_log(
        "pack_release_trim_done "
        f"elapsed_s={float(time.perf_counter() - phase_t0):.3f} "
        f"total_s={float(time.perf_counter() - run_t0):.1f}"
    )
    return row, str(meta.get("env_group_digest", ""))


def _run_arm(
    *,
    arm: str,
    cfg: dict[str, Any],
    env_cfg: dict[str, Any],
    model_override=None,
    frozen_h: dict[str, Any] | None,
    frozen_h_list: list[dict[str, Any]] | None,
    frozen_h_list_env_seeds: list[int] | None,
    prior_mode: str,
    gym_env_ids_arg: str,
    output_dir: Path,
    device_obj: torch.device,
    num_features: int,
    obs_slot_dim: int,
    action_slot_dim: int,
    next_state_target_dim: int,
    n_envs: int,
    n_steps: int,
    updates: int,
    seed: int,
    single_eval_pos: int,
    terminal_token_enabled: bool,
    sb3_reward_normalization_enabled: bool,
    sb3_observation_normalization_enabled: bool,
    sb3_observation_normalization_clip: float,
    sb3_observation_normalization_epsilon: float,
    gain_min: float,
    topology_state_gain_min: float,
    topology_action_gain_min: float,
    topology_state_to_action_ratio_max: float,
    topology_max_attempts: int,
    fixed_env_group_across_updates: bool,
    semantic_probes_enabled: bool,
    semantic_probe_sidecar_enabled: bool,
    semantic_probe_sidecar_every: int,
    checkpoint_every: int,
    checkpoint_include_optimizer: bool,
    ppo_learning_rate: float,
    ppo_batch_size: int | None,
    ppo_n_epochs: int,
    ppo_vf_coef: float,
    ppo_normalize_advantage: bool,
    ppo_target_kl: float | None,
) -> dict[str, Any]:
    algo, vec_env = _build_algo(
        cfg=cfg,
        env_cfg=env_cfg,
        model_override=model_override,
        frozen_h=frozen_h,
        frozen_h_list=frozen_h_list,
        frozen_h_list_env_seeds=frozen_h_list_env_seeds,
        prior_mode=str(prior_mode),
        device_obj=device_obj,
        num_features=int(num_features),
        n_envs=int(n_envs),
        n_steps=int(n_steps),
        build_seed=int(seed),
        sb3_reward_normalization_enabled=bool(sb3_reward_normalization_enabled),
        sb3_observation_normalization_enabled=bool(sb3_observation_normalization_enabled),
        sb3_observation_normalization_clip=float(sb3_observation_normalization_clip),
        sb3_observation_normalization_epsilon=float(sb3_observation_normalization_epsilon),
        gain_min=float(gain_min),
        topology_state_gain_min=float(topology_state_gain_min),
        topology_action_gain_min=float(topology_action_gain_min),
        topology_state_to_action_ratio_max=float(topology_state_to_action_ratio_max),
        topology_max_attempts=int(topology_max_attempts),
        fixed_env_group_across_updates=bool(fixed_env_group_across_updates),
        ppo_learning_rate=float(ppo_learning_rate),
        ppo_batch_size=ppo_batch_size,
        ppo_n_epochs=int(ppo_n_epochs),
        ppo_vf_coef=float(ppo_vf_coef),
        ppo_normalize_advantage=bool(ppo_normalize_advantage),
        ppo_target_kl=ppo_target_kl,
    )
    progress_path = output_dir / f"{arm}_progress.jsonl"
    fixed_group_dump = _dump_fixed_prior_group_if_available(
        vec_env=vec_env,
        output_dir=output_dir,
        arm=str(arm),
    )
    rows: list[dict[str, Any]] = []
    env_group_digests: list[str] = []
    try:
        for update_idx in range(int(updates)):
            env_seed = int(seed) if bool(fixed_env_group_across_updates) else int(seed) + 1000 * int(update_idx)
            row, digest = _run_pack_update_once(
                arm=str(arm),
                algo=algo,
                vec_env=vec_env,
                gym_env_ids_arg=str(gym_env_ids_arg),
                output_dir=output_dir,
                update_idx=int(update_idx),
                env_seed=int(env_seed),
                fixed_env_group_across_updates=bool(fixed_env_group_across_updates),
                fixed_group_dump=fixed_group_dump,
                device_obj=device_obj,
                num_features=int(num_features),
                obs_slot_dim=int(obs_slot_dim),
                action_slot_dim=int(action_slot_dim),
                next_state_target_dim=int(next_state_target_dim),
                n_envs=int(n_envs),
                n_steps=int(n_steps),
                single_eval_pos=int(single_eval_pos),
                terminal_token_enabled=bool(terminal_token_enabled),
                semantic_probes_enabled=bool(semantic_probes_enabled),
                semantic_probe_sidecar_enabled=bool(semantic_probe_sidecar_enabled),
                semantic_probe_sidecar_every=int(semantic_probe_sidecar_every),
                checkpoint_every=int(checkpoint_every),
                checkpoint_include_optimizer=bool(checkpoint_include_optimizer),
                force_checkpoint=(int(update_idx) == int(updates) - 1),
                progress_path=progress_path,
            )
            if digest:
                env_group_digests.append(digest)
            rows.append(row)
        env_group_fixed_pass = (
            len(set(env_group_digests)) <= 1 if bool(fixed_env_group_across_updates) else True
        )
        return {
            "arm": str(arm),
            "hard_pass": all(row["update"]["exception"] is None for row in rows) and bool(env_group_fixed_pass),
            "env_group_fixed_pass": bool(env_group_fixed_pass),
            "env_group_digest_first": env_group_digests[0] if env_group_digests else None,
            "env_group_digest_unique_count": int(len(set(env_group_digests))),
            "progress_path": str(progress_path),
            "updates": int(len(rows)),
            "last_row": rows[-1] if rows else None,
        }
    finally:
        vec_env.close()
        del algo
        if torch.cuda.is_available() and device_obj.type == "cuda":
            torch.cuda.empty_cache()


def run_runner(
    *,
    arms: str,
    output_dir: str,
    device: str,
    n_envs: int,
    n_steps: int,
    updates: int,
    seed: int,
    gym_env_ids: str,
    prior_mode: str,
    fixed_frozen_h_json: str,
    fixed_frozen_h_list_csv: str | None,
    fixed_frozen_h_list_rule: str | None,
    fixed_frozen_h_list_limit: int | None,
    single_eval_pos: int,
    terminal_token_enabled: bool,
    sb3_reward_normalization_enabled: bool,
    sb3_observation_normalization_enabled: bool,
    sb3_observation_normalization_clip: float,
    sb3_observation_normalization_epsilon: float,
    gain_min: float,
    topology_state_gain_min: float,
    topology_action_gain_min: float,
    topology_state_to_action_ratio_max: float,
    topology_max_attempts: int,
    fixed_env_group_across_updates: bool,
    semantic_probes_enabled: bool,
    semantic_probe_sidecar_enabled: bool,
    semantic_probe_sidecar_every: int,
    checkpoint_every: int,
    checkpoint_include_optimizer: bool,
    ppo_learning_rate: float,
    ppo_batch_size: int | None,
    ppo_n_epochs: int,
    ppo_vf_coef: float,
    ppo_normalize_advantage: bool,
    ppo_target_kl: float | None,
) -> dict[str, Any]:
    device_obj = torch.device(device)
    cfg = copy.deepcopy(get_model_default_config("rlpfn"))
    validate_rlpfn_maintained_path_config(cfg)
    env_cfg = copy.deepcopy(cfg["prior"]["environment"])
    num_features = int(cfg["prior"]["num_features"])
    layout = resolve_rlpfn_token_layout(env_cfg, num_features=num_features)
    obs_slot_dim = int(layout["obs_slot_dim"])
    action_slot_dim = int(layout["action_slot_dim"])
    frozen_h = None
    frozen_h_path = None
    frozen_h_list = None
    frozen_h_list_paths = None
    frozen_h_list_env_seeds = None
    if str(prior_mode).strip().lower() == "fixed_frozen":
        frozen_h, frozen_h_path = _load_full_frozen_h(fixed_frozen_h_json)
        frozen_h = _sanitize_loaded_frozen_h_for_batch_use(frozen_h)
    elif str(prior_mode).strip().lower() == "fixed_frozen_list":
        if not fixed_frozen_h_list_csv:
            raise ValueError("prior_mode=fixed_frozen_list requires --fixed-frozen-h-list-csv.")
        frozen_h_list, frozen_h_list_paths, frozen_h_list_env_seeds = _load_fixed_frozen_h_list_from_csv(
            fixed_frozen_h_list_csv,
            rule=fixed_frozen_h_list_rule,
            limit=fixed_frozen_h_list_limit,
        )
        if not frozen_h_list:
            raise ValueError("fixed_frozen_list CSV produced an empty frozen_h list.")
    dim_probe_prior = EnvironmentPrior(copy.deepcopy(env_cfg))
    next_state_target_dim = int(
        max(
            int((frozen_h or {}).get("state_dim", 1)),
            *[int((h or {}).get("state_dim", 1)) for h in (frozen_h_list or [])],
            int(dim_probe_prior._resolve_dim_upper_bound(env_cfg.get("state_dim", None), default=0)),
            1,
        )
    )
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    arm_list = [part.strip().lower() for part in str(arms).split(",") if part.strip()]
    summaries = []
    for arm in arm_list:
        summaries.append(
            _run_arm(
                arm=arm,
                cfg=cfg,
                env_cfg=env_cfg,
                frozen_h=frozen_h,
                frozen_h_list=frozen_h_list,
                frozen_h_list_env_seeds=frozen_h_list_env_seeds,
                prior_mode=str(prior_mode),
                gym_env_ids_arg=str(gym_env_ids),
                output_dir=out_dir,
                device_obj=device_obj,
                num_features=int(num_features),
                obs_slot_dim=int(obs_slot_dim),
                action_slot_dim=int(action_slot_dim),
                next_state_target_dim=int(next_state_target_dim),
                n_envs=int(n_envs),
                n_steps=int(n_steps),
                updates=int(updates),
                seed=int(seed),
                single_eval_pos=int(single_eval_pos),
                terminal_token_enabled=bool(terminal_token_enabled),
                sb3_reward_normalization_enabled=bool(sb3_reward_normalization_enabled),
                sb3_observation_normalization_enabled=bool(sb3_observation_normalization_enabled),
                sb3_observation_normalization_clip=float(sb3_observation_normalization_clip),
                sb3_observation_normalization_epsilon=float(sb3_observation_normalization_epsilon),
                gain_min=float(gain_min),
                topology_state_gain_min=float(topology_state_gain_min),
                topology_action_gain_min=float(topology_action_gain_min),
                topology_state_to_action_ratio_max=float(topology_state_to_action_ratio_max),
                topology_max_attempts=int(topology_max_attempts),
                fixed_env_group_across_updates=bool(fixed_env_group_across_updates),
                semantic_probes_enabled=bool(semantic_probes_enabled),
                semantic_probe_sidecar_enabled=bool(semantic_probe_sidecar_enabled),
                semantic_probe_sidecar_every=int(semantic_probe_sidecar_every),
                checkpoint_every=int(checkpoint_every),
                checkpoint_include_optimizer=bool(checkpoint_include_optimizer),
                ppo_learning_rate=float(ppo_learning_rate),
                ppo_batch_size=ppo_batch_size,
                ppo_n_epochs=int(ppo_n_epochs),
                ppo_vf_coef=float(ppo_vf_coef),
                ppo_normalize_advantage=bool(ppo_normalize_advantage),
                ppo_target_kl=ppo_target_kl,
            )
        )
    report = {
        "audit_entry": "phase2_gym_prior_pack_training_runner",
        "hard_pass": all(bool(summary["hard_pass"]) for summary in summaries),
        "contract": {
            "batch_semantics": (
                "one fixed environment group reused across all updates; "
                "one rollout column per env; one PPO update per packed batch"
                if bool(fixed_env_group_across_updates)
                else "one environment group per update; one rollout column per env; one PPO update per packed batch"
            ),
            "fixed_env_group_across_updates": bool(fixed_env_group_across_updates),
            "n_envs": int(n_envs),
            "n_steps": int(n_steps),
            "single_eval_pos": int(single_eval_pos),
            "num_features": int(num_features),
            "obs_slot_dim": int(obs_slot_dim),
            "action_slot_dim": int(action_slot_dim),
            "next_state_target_dim": int(next_state_target_dim),
            "training_actor_sampling": "stochastic",
            "batch_size": int(n_envs) * int(n_steps),
            "ppo_update_batch_size": (
                int(n_envs) * int(n_steps)
                if ppo_batch_size is None
                else int(ppo_batch_size)
            ),
            "ppo_n_epochs": int(ppo_n_epochs),
            "ppo_learning_rate": float(ppo_learning_rate),
            "ppo_vf_coef": float(ppo_vf_coef),
            "ppo_normalize_advantage": bool(ppo_normalize_advantage),
            "ppo_target_kl": None if ppo_target_kl is None else float(ppo_target_kl),
            "sb3_reward_normalization_enabled": bool(sb3_reward_normalization_enabled),
            "sb3_observation_normalization_enabled": bool(sb3_observation_normalization_enabled),
            "sb3_observation_normalization_clip": (
                float(sb3_observation_normalization_clip)
                if bool(sb3_observation_normalization_enabled)
                else None
            ),
            "sb3_observation_normalization_epsilon": (
                float(sb3_observation_normalization_epsilon)
                if bool(sb3_observation_normalization_enabled)
                else None
            ),
            "semantic_probes_enabled": bool(semantic_probes_enabled),
            "semantic_probe_sidecar_enabled": bool(semantic_probe_sidecar_enabled),
            "semantic_probe_sidecar_every": int(semantic_probe_sidecar_every),
            "checkpoint_every": int(checkpoint_every),
            "checkpoint_include_optimizer": bool(checkpoint_include_optimizer),
            "semantic_probe_groups": [
                "obs_only_scale",
                "model_input_prev_reward_token_scale",
                "per_env_raw_return_tail",
                "masked_action_distribution",
                "action_temporal_constancy",
                "state_or_obs_drift",
                "prior_reward_gain_metadata",
            ],
            "prior_mode": str(prior_mode),
            "topology_state_gain_min": float(topology_state_gain_min),
            "topology_action_gain_min": float(topology_action_gain_min),
            "topology_state_to_action_ratio_max": float(topology_state_to_action_ratio_max),
            "topology_max_attempts": int(topology_max_attempts),
            "frozen_h_json": None if frozen_h_path is None else str(frozen_h_path),
            "fixed_frozen_h_list_csv": None if fixed_frozen_h_list_csv is None else str(fixed_frozen_h_list_csv),
            "fixed_frozen_h_list_rule": None if fixed_frozen_h_list_rule is None else str(fixed_frozen_h_list_rule),
            "fixed_frozen_h_list_limit": None if fixed_frozen_h_list_limit is None else int(fixed_frozen_h_list_limit),
            "fixed_frozen_h_list_paths": None if frozen_h_list_paths is None else [str(p) for p in frozen_h_list_paths],
            "gym_env_ids": str(gym_env_ids),
        },
        "arms": summaries,
    }
    (out_dir / "summary.json").write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Gym-pack vs Prior-pack PPO runner using the canonical bridge.")
    parser.add_argument("--arms", type=str, default="gym,prior")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--updates", type=int, default=1)
    parser.add_argument("--seed", type=int, default=4040)
    parser.add_argument("--gym-env-ids", type=str, default="Ant-v5")
    parser.add_argument(
        "--prior-mode",
        type=str,
        default="fixed_frozen",
        choices=[
            "fixed_frozen",
            "fixed_frozen_list",
            "sampled_gain",
            "sampled_topology",
            GATED_REWARD_PATH_MILESTONE,
            GATED_REWARD_PATH_TERMINAL_COVERAGE_MILESTONE,
        ],
    )
    parser.add_argument("--fixed-frozen-h-json", type=str, default=DEFAULT_FROZEN_H_JSON)
    parser.add_argument("--fixed-frozen-h-list-csv", type=str, default=None)
    parser.add_argument("--fixed-frozen-h-list-rule", type=str, default=None)
    parser.add_argument("--fixed-frozen-h-list-limit", type=int, default=None)
    parser.add_argument("--single-eval-pos", type=int, default=64)
    parser.add_argument("--terminal-token-enabled", type=str2bool, default=True)
    parser.add_argument("--sb3-reward-normalization-enabled", type=str2bool, default=True)
    parser.add_argument("--sb3-observation-normalization-enabled", type=str2bool, default=True)
    parser.add_argument("--sb3-observation-normalization-clip", type=float, default=10.0)
    parser.add_argument("--sb3-observation-normalization-epsilon", type=float, default=1e-8)
    parser.add_argument("--gain-min", type=float, default=0.7)
    parser.add_argument("--topology-state-gain-min", type=float, default=0.7)
    parser.add_argument("--topology-action-gain-min", type=float, default=0.06)
    parser.add_argument("--topology-state-to-action-ratio-max", type=float, default=15.0)
    parser.add_argument("--topology-max-attempts", type=int, default=512)
    parser.add_argument("--fixed-env-group-across-updates", type=str2bool, default=True)
    parser.add_argument("--semantic-probes-enabled", type=str2bool, default=True)
    parser.add_argument("--semantic-probe-sidecar-enabled", type=str2bool, default=False)
    parser.add_argument("--semantic-probe-sidecar-every", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--checkpoint-include-optimizer", type=str2bool, default=False)
    parser.add_argument("--ppo-learning-rate", type=float, default=2e-4)
    parser.add_argument("--ppo-batch-size", type=int, default=None)
    parser.add_argument("--ppo-n-epochs", type=int, default=1)
    parser.add_argument("--ppo-vf-coef", type=float, default=0.1)
    parser.add_argument("--ppo-normalize-advantage", type=str2bool, default=False)
    parser.add_argument("--ppo-target-kl", type=_optional_float, default=0.03)
    args = parser.parse_args(argv)
    report = run_runner(
        arms=args.arms,
        output_dir=args.output_dir,
        device=args.device,
        n_envs=int(args.n_envs),
        n_steps=int(args.n_steps),
        updates=int(args.updates),
        seed=int(args.seed),
        gym_env_ids=args.gym_env_ids,
        prior_mode=args.prior_mode,
        fixed_frozen_h_json=args.fixed_frozen_h_json,
        fixed_frozen_h_list_csv=args.fixed_frozen_h_list_csv,
        fixed_frozen_h_list_rule=args.fixed_frozen_h_list_rule,
        fixed_frozen_h_list_limit=args.fixed_frozen_h_list_limit,
        single_eval_pos=int(args.single_eval_pos),
        terminal_token_enabled=bool(args.terminal_token_enabled),
        sb3_reward_normalization_enabled=bool(args.sb3_reward_normalization_enabled),
        sb3_observation_normalization_enabled=bool(args.sb3_observation_normalization_enabled),
        sb3_observation_normalization_clip=float(args.sb3_observation_normalization_clip),
        sb3_observation_normalization_epsilon=float(args.sb3_observation_normalization_epsilon),
        gain_min=float(args.gain_min),
        topology_state_gain_min=float(args.topology_state_gain_min),
        topology_action_gain_min=float(args.topology_action_gain_min),
        topology_state_to_action_ratio_max=float(args.topology_state_to_action_ratio_max),
        topology_max_attempts=int(args.topology_max_attempts),
        fixed_env_group_across_updates=bool(args.fixed_env_group_across_updates),
        semantic_probes_enabled=bool(args.semantic_probes_enabled),
        semantic_probe_sidecar_enabled=bool(args.semantic_probe_sidecar_enabled),
        semantic_probe_sidecar_every=int(args.semantic_probe_sidecar_every),
        checkpoint_every=int(args.checkpoint_every),
        checkpoint_include_optimizer=bool(args.checkpoint_include_optimizer),
        ppo_learning_rate=float(args.ppo_learning_rate),
        ppo_batch_size=args.ppo_batch_size,
        ppo_n_epochs=int(args.ppo_n_epochs),
        ppo_vf_coef=float(args.ppo_vf_coef),
        ppo_normalize_advantage=bool(args.ppo_normalize_advantage),
        ppo_target_kl=args.ppo_target_kl,
    )
    print(json.dumps(_json_safe({"hard_pass": report["hard_pass"], "output_dir": args.output_dir}), sort_keys=True))


if __name__ == "__main__":
    main()
