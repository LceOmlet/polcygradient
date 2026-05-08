#!/usr/bin/env python
"""Exploratory reward-path-only group balancing helpers.

This module is deliberately outside the exact-SCM implementation path.  It
monkey-patches EnvironmentPrior only inside the current Python process when
explicitly requested by an exploratory script.  The wrapper keeps the original
transition state/terminal path and recomputes only the reward path with scaled
state/action/noise input groups.
"""

from __future__ import annotations

import copy
import inspect
import math
from contextlib import ExitStack, contextmanager
from typing import Any, Callable

import torch


BALANCE_ENABLED_KEY = "exploratory_reward_group_balance_enabled"
BALANCE_STATE_SCALE_KEY = "exploratory_reward_group_state_scale"
BALANCE_ACTION_SCALE_KEY = "exploratory_reward_group_action_scale"
BALANCE_NOISE_SCALE_KEY = "exploratory_reward_group_noise_scale"
BALANCE_PROFILE_KEY = "exploratory_reward_group_balance_profile"


def balanced_gain_summary_from_raw(
    *,
    state_gain: float,
    action_gain: float,
    noise_gain: float,
    state_scale: float,
    action_scale: float,
    noise_scale: float,
) -> dict[str, float]:
    """Return reward topology fractions after group scaling."""

    s = max(0.0, float(state_gain)) * max(0.0, float(state_scale))
    a = max(0.0, float(action_gain)) * max(0.0, float(action_scale))
    n = max(0.0, float(noise_gain)) * max(0.0, float(noise_scale))
    total = max(float(s + a + n), 1e-12)
    return {
        "reward_state_input_gain_fraction": float(s / total),
        "reward_action_input_gain_fraction": float(a / total),
        "reward_noise_input_gain_fraction": float(n / total),
        "reward_state_to_noise_gain_ratio": float(s / max(n, 1e-12)),
        "reward_state_to_action_gain_ratio": float(s / max(a, 1e-12)),
    }


def _as_bool(value: Any) -> bool:
    if torch.is_tensor(value):
        if value.numel() != 1:
            return bool(torch.any(value != 0).item())
        return bool(value.detach().cpu().reshape(()).item())
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "off", "none"}
    return bool(value)


def _scalar_from_h(h: dict[str, Any], key: str, default: float) -> float:
    value = h.get(key, default)
    if torch.is_tensor(value):
        value = value.detach().cpu().reshape(-1)[0].item()
    try:
        value_f = float(value)
    except Exception:
        value_f = float(default)
    if not math.isfinite(value_f):
        value_f = float(default)
    return float(value_f)


def h_balance_enabled(h: dict[str, Any]) -> bool:
    return _as_bool(h.get(BALANCE_ENABLED_KEY, False))


def annotate_h_with_reward_group_balance(
    h: dict[str, Any],
    *,
    state_scale: float,
    action_scale: float,
    noise_scale: float,
    profile: str,
) -> dict[str, Any]:
    out = copy.deepcopy(h)
    out[BALANCE_ENABLED_KEY] = True
    out[BALANCE_STATE_SCALE_KEY] = float(state_scale)
    out[BALANCE_ACTION_SCALE_KEY] = float(action_scale)
    out[BALANCE_NOISE_SCALE_KEY] = float(noise_scale)
    out[BALANCE_PROFILE_KEY] = str(profile)
    return out


def balance_scales_from_h_list(h_list: list[dict[str, Any]]) -> tuple[list[float], list[float], list[float], list[str]]:
    state_scales = [_scalar_from_h(h, BALANCE_STATE_SCALE_KEY, 1.0) for h in h_list]
    action_scales = [_scalar_from_h(h, BALANCE_ACTION_SCALE_KEY, 1.0) for h in h_list]
    noise_scales = [_scalar_from_h(h, BALANCE_NOISE_SCALE_KEY, 1.0) for h in h_list]
    profiles = [str(h.get(BALANCE_PROFILE_KEY, "none")) for h in h_list]
    return state_scales, action_scales, noise_scales, profiles


def _closure_nonlocals(fn: Callable[..., Any]) -> dict[str, Any]:
    try:
        return dict(inspect.getclosurevars(fn).nonlocals)
    except Exception:
        return {}


def _unwrap_reference_core(fn: Callable[..., Any]) -> Callable[..., Any] | None:
    """Find the inner reference SCM padded transition function."""

    seen: set[int] = set()
    current = fn
    for _ in range(8):
        if id(current) in seen:
            return None
        seen.add(id(current))
        nonlocals = _closure_nonlocals(current)
        if "_get_runtime_cached" in nonlocals:
            return current
        next_fn = (
            nonlocals.get("transition_fn_full")
            or nonlocals.get("transition_core")
            or nonlocals.get("transition_fn")
        )
        if callable(next_fn):
            current = next_fn
            continue
        return None
    return None


def _reference_core_storage(fn: Callable[..., Any]) -> dict[str, Any] | None:
    core = _unwrap_reference_core(fn)
    if core is None:
        return None
    nonlocals = _closure_nonlocals(core)
    getter = nonlocals.get("_get_runtime_cached")
    if not callable(getter):
        return None
    getter_nonlocals = _closure_nonlocals(getter)
    needed = ("first_weight", "first_weight_packed", "runtime_tensor_cache")
    if not all(key in getter_nonlocals for key in needed):
        return None
    return {
        "core": core,
        "first_weight": getter_nonlocals["first_weight"],
        "first_weight_packed": getter_nonlocals["first_weight_packed"],
        "runtime_tensor_cache": getter_nonlocals["runtime_tensor_cache"],
    }


def _infer_env_batch_size(env: dict[str, Any], h_list: list[dict[str, Any]]) -> int:
    for key in (
        "state_dim_per_sample",
        "env_obs_input_dim_per_sample",
        "action_dim_per_sample",
        "noise_dim_per_sample",
        "zero_pad_dim_per_sample",
        "env_rng_seeds",
    ):
        value = env.get(key, None)
        if torch.is_tensor(value) and value.ndim > 0:
            return int(value.shape[0])
        if isinstance(value, (list, tuple)):
            return int(len(value))
    return int(len(h_list))


def _select_partition_value(value: Any, indices: list[int], batch_size: int) -> Any:
    if torch.is_tensor(value):
        if value.ndim > 0 and int(value.shape[0]) == batch_size:
            idx = torch.as_tensor(indices, device=value.device, dtype=torch.long)
            return value.index_select(0, idx)
        return value
    if isinstance(value, tuple) and len(value) == batch_size:
        return tuple(value[int(i)] for i in indices)
    if isinstance(value, list) and len(value) == batch_size:
        return [value[int(i)] for i in indices]
    return value


def _max_dim_from_subset(env: dict[str, Any], vector_key: str, scalar_key: str, default: int = 0) -> int:
    value = env.get(vector_key, None)
    if value is None:
        value = env.get(scalar_key, default)
    if torch.is_tensor(value):
        flat = value.detach().reshape(-1)
        if int(flat.numel()) > 0:
            return int(flat.max().item())
        return int(default)
    if isinstance(value, (list, tuple)) and value:
        return int(max(int(v) for v in value))
    try:
        return int(value)
    except Exception:
        return int(default)


def _select_env_subset(env: dict[str, Any], indices: list[int], batch_size: int) -> dict[str, Any]:
    sub_env = {
        key: _select_partition_value(value, indices, batch_size)
        for key, value in env.items()
    }
    dim_key_pairs = (
        ("state_dim_per_sample", "state_dim"),
        ("env_obs_input_dim_per_sample", "env_obs_input_dim"),
        ("action_dim_per_sample", "action_dim"),
        ("noise_dim_per_sample", "noise_dim"),
        ("zero_pad_dim_per_sample", "zero_pad_dim"),
    )
    for vector_key, scalar_key in dim_key_pairs:
        if vector_key in sub_env:
            sub_env[scalar_key] = max(1, _max_dim_from_subset(sub_env, vector_key, scalar_key, 0))
    if all(key in sub_env for key in ("state_dim", "env_obs_input_dim", "action_dim", "noise_dim")):
        sub_env["env_input_dim"] = int(sub_env["state_dim"]) + int(sub_env["env_obs_input_dim"]) + int(sub_env["action_dim"]) + int(sub_env["noise_dim"])
    return sub_env


def _transition_balance_available(transition: Callable[..., Any]) -> bool:
    if _reference_core_storage(transition) is not None:
        return True
    partition_sub_index_lists = getattr(
        transition,
        "_reference_scm_partition_sub_index_lists",
        None,
    )
    partition_sub_transition_fns = getattr(
        transition,
        "_reference_scm_partition_sub_transition_fns",
        None,
    )
    if partition_sub_index_lists is None or partition_sub_transition_fns is None:
        return False
    if len(partition_sub_index_lists) != len(partition_sub_transition_fns):
        return False
    return all(_transition_balance_available(sub_fn) for sub_fn in partition_sub_transition_fns)


def _clone_rng_seeds(rng_seeds: Any) -> list[int] | None:
    if rng_seeds is None:
        return None
    if torch.is_tensor(rng_seeds):
        return [int(v) for v in rng_seeds.detach().cpu().reshape(-1).tolist()]
    try:
        return [int(v) for v in rng_seeds]
    except TypeError:
        return None


def _extract_rng_seeds_from_sample_call(args: tuple[Any, ...], kwargs: dict[str, Any]) -> list[int] | None:
    if "rng_seeds" in kwargs:
        return _clone_rng_seeds(kwargs.get("rng_seeds"))
    if len(args) >= 1:
        return _clone_rng_seeds(args[0])
    return None


def _dim_tensor(env: dict[str, Any], key: str, scalar_key: str, batch_size: int, device: torch.device) -> torch.Tensor:
    value = env.get(key, None)
    if value is None:
        value = env.get(scalar_key, 0)
    if torch.is_tensor(value):
        t = value.detach().to(device=device, dtype=torch.long).reshape(-1)
        if int(t.numel()) == batch_size:
            return t
        if int(t.numel()) == 1:
            return t.expand(batch_size)
        raise ValueError(f"{key} must be scalar or batch-sized")
    return torch.full((batch_size,), int(value), device=device, dtype=torch.long)


def _scale_matrix_for_padded_env(
    env: dict[str, Any],
    *,
    width: int,
    state_scales: list[float],
    action_scales: list[float],
    noise_scales: list[float],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    batch_size = len(state_scales)
    out = torch.ones((batch_size, int(width)), device=device, dtype=dtype)
    state_dims = _dim_tensor(env, "state_dim_per_sample", "state_dim", batch_size, device).clamp_min(0)
    obs_input_dims = _dim_tensor(env, "env_obs_input_dim_per_sample", "env_obs_input_dim", batch_size, device).clamp_min(0)
    action_dims = _dim_tensor(env, "action_dim_per_sample", "action_dim", batch_size, device).clamp_min(0)
    noise_dims = _dim_tensor(env, "noise_dim_per_sample", "noise_dim", batch_size, device).clamp_min(0)
    state_cap = int(env.get("state_dim", int(state_dims.max().item() if state_dims.numel() else 0)))
    obs_cap = int(env.get("env_obs_input_dim", int(obs_input_dims.max().item() if obs_input_dims.numel() else 0)))
    action_cap = int(env.get("action_dim", int(action_dims.max().item() if action_dims.numel() else 0)))
    noise_cap = int(env.get("noise_dim", int(noise_dims.max().item() if noise_dims.numel() else 0)))
    action_start = state_cap + obs_cap
    noise_start = action_start + action_cap
    max_width = int(width)
    for bi in range(batch_size):
        s_i = int(state_dims[bi].item())
        a_i = int(action_dims[bi].item())
        n_i = int(noise_dims[bi].item())
        if s_i > 0:
            out[bi, :min(s_i, max_width)] = float(state_scales[bi])
        if a_i > 0 and action_start < max_width:
            out[bi, action_start:min(action_start + a_i, max_width)] = float(action_scales[bi])
        if n_i > 0 and noise_start < max_width:
            out[bi, noise_start:min(noise_start + n_i, max_width)] = float(noise_scales[bi])
    return out


def _scale_matrix_for_packed_env(
    env: dict[str, Any],
    *,
    width: int,
    state_scales: list[float],
    action_scales: list[float],
    noise_scales: list[float],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    batch_size = len(state_scales)
    out = torch.ones((batch_size, int(width)), device=device, dtype=dtype)
    state_dims = _dim_tensor(env, "state_dim_per_sample", "state_dim", batch_size, device).clamp_min(0)
    obs_input_dims = _dim_tensor(env, "env_obs_input_dim_per_sample", "env_obs_input_dim", batch_size, device).clamp_min(0)
    action_dims = _dim_tensor(env, "action_dim_per_sample", "action_dim", batch_size, device).clamp_min(0)
    noise_dims = _dim_tensor(env, "noise_dim_per_sample", "noise_dim", batch_size, device).clamp_min(0)
    max_width = int(width)
    for bi in range(batch_size):
        cursor = 0
        s_i = int(state_dims[bi].item())
        o_i = int(obs_input_dims[bi].item())
        a_i = int(action_dims[bi].item())
        n_i = int(noise_dims[bi].item())
        if s_i > 0:
            out[bi, cursor:min(cursor + s_i, max_width)] = float(state_scales[bi])
        cursor += s_i + o_i
        if a_i > 0 and cursor < max_width:
            out[bi, cursor:min(cursor + a_i, max_width)] = float(action_scales[bi])
        cursor += a_i
        if n_i > 0 and cursor < max_width:
            out[bi, cursor:min(cursor + n_i, max_width)] = float(noise_scales[bi])
    return out


@contextmanager
def _preserve_transition_rng(args: tuple[Any, ...], kwargs: dict[str, Any]):
    generators = []
    for key in ("generator", "generators_for_noise"):
        value = kwargs.get(key, None)
        if isinstance(value, torch.Generator):
            generators.append(value)
        elif isinstance(value, (list, tuple)):
            generators.extend([g for g in value if isinstance(g, torch.Generator)])
    states = [g.get_state() for g in generators]
    cpu_state = torch.random.get_rng_state()
    cuda_states: dict[int, torch.Tensor] = {}
    for obj in list(args) + list(kwargs.values()):
        if torch.is_tensor(obj) and obj.device.type == "cuda":
            idx = int(obj.device.index or 0)
            if idx not in cuda_states:
                cuda_states[idx] = torch.cuda.get_rng_state(device=obj.device)
    try:
        yield
    finally:
        for g, state in zip(generators, states):
            g.set_state(state)
        torch.random.set_rng_state(cpu_state)
        for idx, state in cuda_states.items():
            torch.cuda.set_rng_state(state, device=torch.device(f"cuda:{idx}"))


@contextmanager
def _temporarily_scaled_first_layer(
    storage: dict[str, Any],
    env: dict[str, Any],
    *,
    state_scales: list[float],
    action_scales: list[float],
    noise_scales: list[float],
):
    first_weight = storage["first_weight"]
    first_weight_packed = storage.get("first_weight_packed")
    cache = storage["runtime_tensor_cache"]
    if not torch.is_tensor(first_weight) or first_weight.ndim != 3:
        raise RuntimeError("reward group balancing requires vectorized reference SCM first_weight")
    fw_old = first_weight.detach().clone()
    fwp_old = first_weight_packed.detach().clone() if torch.is_tensor(first_weight_packed) else None
    with torch.no_grad():
        scale = _scale_matrix_for_padded_env(
            env,
            width=int(first_weight.shape[1]),
            state_scales=state_scales,
            action_scales=action_scales,
            noise_scales=noise_scales,
            device=first_weight.device,
            dtype=first_weight.dtype,
        )
        first_weight.mul_(scale.unsqueeze(-1))
        if torch.is_tensor(first_weight_packed):
            packed_scale = _scale_matrix_for_packed_env(
                env,
                width=int(first_weight_packed.shape[1]),
                state_scales=state_scales,
                action_scales=action_scales,
                noise_scales=noise_scales,
                device=first_weight_packed.device,
                dtype=first_weight_packed.dtype,
            )
            first_weight_packed.mul_(packed_scale.unsqueeze(-1))
        cache.clear()
    try:
        yield
    finally:
        with torch.no_grad():
            first_weight.copy_(fw_old)
            if torch.is_tensor(first_weight_packed) and fwp_old is not None:
                first_weight_packed.copy_(fwp_old)
            cache.clear()


def _permanently_scale_first_layer(
    storage: dict[str, Any],
    env: dict[str, Any],
    *,
    state_scales: list[float],
    action_scales: list[float],
    noise_scales: list[float],
) -> None:
    first_weight = storage["first_weight"]
    first_weight_packed = storage.get("first_weight_packed")
    cache = storage["runtime_tensor_cache"]
    if not torch.is_tensor(first_weight) or first_weight.ndim != 3:
        raise RuntimeError("reward group balancing requires vectorized reference SCM first_weight")
    if bool(getattr(first_weight, "_exploratory_reward_group_balance_scaled", False)):
        raise RuntimeError("reward group balancing attempted to permanently scale a transition twice")
    with torch.no_grad():
        scale = _scale_matrix_for_padded_env(
            env,
            width=int(first_weight.shape[1]),
            state_scales=state_scales,
            action_scales=action_scales,
            noise_scales=noise_scales,
            device=first_weight.device,
            dtype=first_weight.dtype,
        )
        first_weight.mul_(scale.unsqueeze(-1))
        if torch.is_tensor(first_weight_packed):
            packed_scale = _scale_matrix_for_packed_env(
                env,
                width=int(first_weight_packed.shape[1]),
                state_scales=state_scales,
                action_scales=action_scales,
                noise_scales=noise_scales,
                device=first_weight_packed.device,
                dtype=first_weight_packed.dtype,
            )
            first_weight_packed.mul_(packed_scale.unsqueeze(-1))
        cache.clear()
    setattr(first_weight, "_exploratory_reward_group_balance_scaled", True)


def _permanently_scale_transition_reward_path(
    transition: Callable[..., Any],
    env: dict[str, Any],
    h_list: list[dict[str, Any]],
    *,
    state_scales: list[float],
    action_scales: list[float],
    noise_scales: list[float],
) -> None:
    storage = _reference_core_storage(transition)
    if storage is not None:
        _permanently_scale_first_layer(
            storage,
            env,
            state_scales=state_scales,
            action_scales=action_scales,
            noise_scales=noise_scales,
        )
        return

    partition_sub_index_lists = getattr(
        transition,
        "_reference_scm_partition_sub_index_lists",
        None,
    )
    partition_sub_transition_fns = getattr(
        transition,
        "_reference_scm_partition_sub_transition_fns",
        None,
    )
    if partition_sub_index_lists is None and partition_sub_transition_fns is None:
        raise RuntimeError("reward group balancing requires a reference SCM transition core")
    if partition_sub_index_lists is None or partition_sub_transition_fns is None:
        raise RuntimeError("reward group balancing found an incomplete reference SCM partition")

    batch_size = _infer_env_batch_size(env, h_list)
    for indices_raw, sub_transition in zip(partition_sub_index_lists, partition_sub_transition_fns):
        indices = [int(i) for i in indices_raw]
        if not indices:
            continue
        sub_env = _select_env_subset(env, indices, batch_size)
        sub_env["transition_generator"] = sub_transition
        sub_h_list = [h_list[int(i)] for i in indices]
        _permanently_scale_transition_reward_path(
            sub_transition,
            sub_env,
            sub_h_list,
            state_scales=[state_scales[int(i)] for i in indices],
            action_scales=[action_scales[int(i)] for i in indices],
            noise_scales=[noise_scales[int(i)] for i in indices],
        )


@contextmanager
def _temporarily_scaled_transition_reward_path(
    transition: Callable[..., Any],
    env: dict[str, Any],
    h_list: list[dict[str, Any]],
    *,
    state_scales: list[float],
    action_scales: list[float],
    noise_scales: list[float],
):
    storage = _reference_core_storage(transition)
    if storage is not None:
        with _temporarily_scaled_first_layer(
            storage,
            env,
            state_scales=state_scales,
            action_scales=action_scales,
            noise_scales=noise_scales,
        ):
            yield
        return

    partition_sub_index_lists = getattr(
        transition,
        "_reference_scm_partition_sub_index_lists",
        None,
    )
    partition_sub_transition_fns = getattr(
        transition,
        "_reference_scm_partition_sub_transition_fns",
        None,
    )
    if partition_sub_index_lists is None and partition_sub_transition_fns is None:
        raise RuntimeError("reward group balancing requires a reference SCM transition core")
    if partition_sub_index_lists is None or partition_sub_transition_fns is None:
        raise RuntimeError("reward group balancing found an incomplete reference SCM partition")

    batch_size = _infer_env_batch_size(env, h_list)
    with ExitStack() as stack:
        for indices_raw, sub_transition in zip(partition_sub_index_lists, partition_sub_transition_fns):
            indices = [int(i) for i in indices_raw]
            if not indices:
                continue
            sub_env = _select_env_subset(env, indices, batch_size)
            sub_env["transition_generator"] = sub_transition
            sub_h_list = [h_list[int(i)] for i in indices]
            stack.enter_context(
                _temporarily_scaled_transition_reward_path(
                    sub_transition,
                    sub_env,
                    sub_h_list,
                    state_scales=[state_scales[int(i)] for i in indices],
                    action_scales=[action_scales[int(i)] for i in indices],
                    noise_scales=[noise_scales[int(i)] for i in indices],
                )
            )
        yield


def _combine_reward_only_outputs(original: Any, balanced: Any) -> Any:
    if not isinstance(original, (tuple, list)) or not isinstance(balanced, (tuple, list)):
        return original
    if len(original) < 2 or len(balanced) < 2:
        return original
    out = list(original)
    out[1] = balanced[1]
    return tuple(out)


def _update_reward_group_balance_gain_metadata(
    env: dict[str, Any],
    h_list: list[dict[str, Any]],
    *,
    state_scales: list[float],
    action_scales: list[float],
    noise_scales: list[float],
) -> None:
    n = len(h_list)
    state_gain = env.get("reward_state_input_gain_fraction")
    action_gain = env.get("reward_action_input_gain_fraction")
    noise_gain = env.get("reward_noise_input_gain_fraction")
    if torch.is_tensor(state_gain) and torch.is_tensor(action_gain) and torch.is_tensor(noise_gain):
        device = state_gain.device
        dtype = state_gain.dtype
        state_scale_t = torch.as_tensor(state_scales, device=device, dtype=dtype)
        action_scale_t = torch.as_tensor(action_scales, device=device, dtype=dtype)
        noise_scale_t = torch.as_tensor(noise_scales, device=device, dtype=dtype)
        s = state_gain * state_scale_t
        a = action_gain * action_scale_t
        z = noise_gain * noise_scale_t
        total = (s + a + z).clamp_min(1e-12)
        env["original_reward_state_input_gain_fraction"] = state_gain.detach().clone()
        env["original_reward_action_input_gain_fraction"] = action_gain.detach().clone()
        env["original_reward_noise_input_gain_fraction"] = noise_gain.detach().clone()
        env["original_reward_state_to_action_gain_ratio"] = env.get(
            "reward_state_to_action_gain_ratio",
            torch.full((n,), float("nan"), device=device, dtype=dtype),
        )
        env["reward_state_input_gain_fraction"] = s / total
        env["reward_action_input_gain_fraction"] = a / total
        env["reward_noise_input_gain_fraction"] = z / total
        env["reward_state_to_noise_gain_ratio"] = s / z.clamp_min(1e-12)
        env["reward_state_to_action_gain_ratio"] = s / a.clamp_min(1e-12)


def apply_reward_group_balance_to_env(
    env: dict[str, Any],
    h_list: list[dict[str, Any]],
    *,
    fail_if_unavailable: bool = True,
) -> dict[str, Any]:
    if not any(h_balance_enabled(h) for h in h_list):
        return env
    transition = env.get("transition_generator", None)
    if not callable(transition):
        if fail_if_unavailable:
            raise RuntimeError("reward group balancing requires a transition_generator")
        return env
    if not _transition_balance_available(transition):
        if fail_if_unavailable:
            raise RuntimeError("reward group balancing requires reference SCM core or partition metadata")
        return env

    state_scales, action_scales, noise_scales, profiles = balance_scales_from_h_list(h_list)
    original_transition = transition

    def reward_balanced_transition(*args, **kwargs):
        with _preserve_transition_rng(args, kwargs):
            original_out = original_transition(*args, **kwargs)
        with _temporarily_scaled_transition_reward_path(
            original_transition,
            env,
            h_list,
            state_scales=state_scales,
            action_scales=action_scales,
            noise_scales=noise_scales,
        ):
            balanced_out = original_transition(*args, **kwargs)
        return _combine_reward_only_outputs(original_out, balanced_out)

    for name, value in getattr(original_transition, "__dict__", {}).items():
        setattr(reward_balanced_transition, name, value)
    reward_balanced_transition._exploratory_reward_group_balance_enabled = True
    reward_balanced_transition._exploratory_reward_group_balance_mode = "slow_temporary"
    reward_balanced_transition._exploratory_reward_group_balance_profiles = tuple(profiles)
    env["transition_generator"] = reward_balanced_transition

    _update_reward_group_balance_gain_metadata(
        env,
        h_list,
        state_scales=state_scales,
        action_scales=action_scales,
        noise_scales=noise_scales,
    )
    return env


def apply_prebuilt_reward_group_balance_to_env(
    env: dict[str, Any],
    balanced_env: dict[str, Any],
    h_list: list[dict[str, Any]],
    *,
    fail_if_unavailable: bool = True,
) -> dict[str, Any]:
    """Install a fast reward-path-only wrapper using a pre-scaled twin env.

    The original env is still the authority for next state and terminal output.
    The twin env is built from the same h/seeds and permanently scaled once; it
    contributes only the reward tensor.  This is intentionally equivalent to
    ``apply_reward_group_balance_to_env`` but avoids per-step weight mutation.
    """

    if not any(h_balance_enabled(h) for h in h_list):
        return env
    original_transition = env.get("transition_generator", None)
    balanced_transition = balanced_env.get("transition_generator", None)
    if not callable(original_transition) or not callable(balanced_transition):
        if fail_if_unavailable:
            raise RuntimeError("prebuilt reward group balancing requires two transition_generator callables")
        return env
    if not _transition_balance_available(original_transition) or not _transition_balance_available(balanced_transition):
        if fail_if_unavailable:
            raise RuntimeError("prebuilt reward group balancing requires reference SCM core or partition metadata")
        return env

    state_scales, action_scales, noise_scales, profiles = balance_scales_from_h_list(h_list)
    _permanently_scale_transition_reward_path(
        balanced_transition,
        balanced_env,
        h_list,
        state_scales=state_scales,
        action_scales=action_scales,
        noise_scales=noise_scales,
    )

    def reward_balanced_transition(*args, **kwargs):
        with _preserve_transition_rng(args, kwargs):
            original_out = original_transition(*args, **kwargs)
        balanced_out = balanced_transition(*args, **kwargs)
        return _combine_reward_only_outputs(original_out, balanced_out)

    for name, value in getattr(original_transition, "__dict__", {}).items():
        setattr(reward_balanced_transition, name, value)
    reward_balanced_transition._exploratory_reward_group_balance_enabled = True
    reward_balanced_transition._exploratory_reward_group_balance_mode = "fast_prebuilt"
    reward_balanced_transition._exploratory_reward_group_balance_profiles = tuple(profiles)
    env["transition_generator"] = reward_balanced_transition

    if isinstance(balanced_env, dict):
        balanced_env["policy_generator"] = None
        balanced_env["_exploratory_reward_group_balance_twin_env"] = True

    _update_reward_group_balance_gain_metadata(
        env,
        h_list,
        state_scales=state_scales,
        action_scales=action_scales,
        noise_scales=noise_scales,
    )
    return env


def install_environment_prior_reward_group_balance_patch() -> None:
    """Patch EnvironmentPrior sampling methods for this process only."""

    from ticl.priors.environment_prior import EnvironmentPrior

    if getattr(EnvironmentPrior, "_exploratory_reward_group_balance_patch_installed", False):
        return

    original_family = EnvironmentPrior._sample_environment_family_coarse_batch
    original_batch = EnvironmentPrior._sample_environment_batch

    def patched_family(self, h_list, device, *args, **kwargs):
        env = original_family(self, h_list, device, *args, **kwargs)
        h_items = list(h_list)
        if not any(h_balance_enabled(h) for h in h_items):
            return env
        rng_seeds = _extract_rng_seeds_from_sample_call(args, kwargs)
        if rng_seeds is None:
            return apply_reward_group_balance_to_env(env, h_items, fail_if_unavailable=True)
        twin_kwargs = dict(kwargs)
        twin_kwargs["rng_seeds"] = rng_seeds
        balanced_env = original_family(self, h_items, device, *args[1:], **twin_kwargs)
        return apply_prebuilt_reward_group_balance_to_env(env, balanced_env, h_items, fail_if_unavailable=True)

    def patched_batch(self, h_list, device, *args, **kwargs):
        env = original_batch(self, h_list, device, *args, **kwargs)
        h_items = list(h_list)
        if not any(h_balance_enabled(h) for h in h_items):
            return env
        rng_seeds = _extract_rng_seeds_from_sample_call(args, kwargs)
        if rng_seeds is None:
            return apply_reward_group_balance_to_env(env, h_items, fail_if_unavailable=True)
        twin_kwargs = dict(kwargs)
        twin_kwargs["rng_seeds"] = rng_seeds
        balanced_env = original_batch(self, h_items, device, *args[1:], **twin_kwargs)
        return apply_prebuilt_reward_group_balance_to_env(env, balanced_env, h_items, fail_if_unavailable=True)

    EnvironmentPrior._sample_environment_family_coarse_batch = patched_family
    EnvironmentPrior._sample_environment_batch = patched_batch
    EnvironmentPrior._exploratory_reward_group_balance_patch_installed = True
