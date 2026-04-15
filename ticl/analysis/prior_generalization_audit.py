import argparse
import copy
import hashlib
import json
import math
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from ticl.model_builder import load_model
from ticl.model_configs import get_model_default_config
from ticl.priors.environment_prior import EnvironmentPrior
from ticl.train import _build_policy_step_fn, _freeze_env_h_list_for_replay


CORE_A_ENV_OVERRIDES = {
    "family": {"distribution": "meta_choice", "choice_values": ["scm"]},
    "strict_joint_transition_enabled": True,
    "terminal_reset_enabled": False,
    "terminal_reset_count_target": 0.0,
    "reward_dropout_enabled": False,
    "reward_dropout_randomize": False,
    "reward_dropout_ratio": 0.0,
    "ctrl_reward_weight": 0.0,
    "ctrl_reward_enable_prob": 0.0,
    "survival_reward_weight": 0.0,
    "survival_reward_enable_prob": 0.0,
    "normalized_q_value_weight": 0.0,
    "next_state_flow_matching_weight": 0.0,
    "state_input_scale_enabled": False,
    "state_input_scale": 1.0,
    "state_full_rms_enabled": False,
    "state_highway_enabled": False,
    "action_noise_train_std": 0.0,
    "action_noise_eval_std": 0.0,
    "reinforce_action_transform": "clip",
    "reinforce_action_clip_bound": 5.0,
    "reinforce_reward_transform": "none",
}


def _to_builtin(value):
    if torch.is_tensor(value):
        if value.ndim == 0:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _capture_rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state.get("torch_cuda", None) is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


@contextmanager
def _temporary_seed(seed: Optional[int]):
    if seed is None:
        yield
        return
    state = _capture_rng_state()
    seed_i = int(seed)
    random.seed(seed_i)
    np.random.seed(seed_i)
    torch.manual_seed(seed_i)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_i)
    try:
        yield
    finally:
        _restore_rng_state(state)


def _default_device():
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _safe_std(values: torch.Tensor) -> float:
    if int(values.numel()) <= 1:
        return 0.0
    return float(values.std(unbiased=False).item())


def _finite_stats(values: torch.Tensor, *, prefix: str) -> dict[str, float]:
    flat = values.reshape(-1).detach().cpu().to(dtype=torch.float32)
    finite_mask = torch.isfinite(flat)
    finite = flat[finite_mask]
    nonfinite_count = int((~finite_mask).sum().item())
    stats = {
        f"{prefix}_nonfinite_count": int(nonfinite_count),
        f"{prefix}_nonfinite_share": float(nonfinite_count / max(1, int(flat.numel()))),
    }
    if int(finite.numel()) <= 0:
        stats[f"{prefix}_mean"] = float("nan")
        stats[f"{prefix}_std"] = float("nan")
        return stats
    stats[f"{prefix}_mean"] = float(finite.mean().item())
    stats[f"{prefix}_std"] = _safe_std(finite)
    return stats


def _resolve_default_single_eval_pos(config: dict[str, Any], n_samples: int) -> int:
    eval_positions = config.get("prior", {}).get("eval_positions", None)
    if isinstance(eval_positions, (list, tuple)) and len(eval_positions) > 0:
        try:
            return int(max(1, min(int(n_samples) - 1, int(round(float(eval_positions[0]))))))
        except Exception:
            pass
    return int(max(1, min(int(n_samples) - 1, int(n_samples) // 2)))


def _maybe_load_suite(path: Optional[str]) -> Optional[dict[str, Any]]:
    if path is None:
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def save_prior_suite(path: str, suite: dict[str, Any]) -> str:
    import cloudpickle

    out_path = Path(path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(suite, out_path, pickle_module=cloudpickle)
    return str(out_path)


def _suite_fingerprint(suite: dict[str, Any]) -> str:
    payload = json.dumps(
        {
            "h_list": _to_builtin(suite["h_list"]),
            "env_seeds": _to_builtin(suite["env_seeds"]),
            "rollout_seeds": _to_builtin(suite["rollout_seeds"]),
            "batch_size": int(suite["batch_size"]),
        },
        sort_keys=True,
        ensure_ascii=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _summarize_suite(prior: EnvironmentPrior, suite: dict[str, Any]) -> dict[str, Any]:
    h_list = list(suite["h_list"])
    family_counts = {}
    structure_counts = {}
    obs_dims = []
    action_dims = []
    state_dims = []
    for h in h_list:
        family = str(prior._normalize_family(h.get("family", "scm")))
        family_counts[family] = int(family_counts.get(family, 0) + 1)
        structure_sig = tuple(prior._environment_structure_signature(h))
        structure_counts[repr(structure_sig)] = int(structure_counts.get(repr(structure_sig), 0) + 1)
        state_dim, obs_dim, action_dim, _, _ = prior._sample_dims(h)
        state_dims.append(int(state_dim))
        obs_dims.append(int(obs_dim))
        action_dims.append(int(action_dim))
    structure_top = sorted(structure_counts.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))[:5]
    return {
        "batch_size": int(suite["batch_size"]),
        "suite_seed": None if suite.get("suite_seed", None) is None else int(suite["suite_seed"]),
        "fingerprint": _suite_fingerprint(suite),
        "family_counts": family_counts,
        "unique_structures": int(len(structure_counts)),
        "top_structures": [{"signature": k, "count": int(v)} for k, v in structure_top],
        "state_dim_mean": float(np.mean(state_dims)) if state_dims else 0.0,
        "obs_dim_mean": float(np.mean(obs_dims)) if obs_dims else 0.0,
        "action_dim_mean": float(np.mean(action_dims)) if action_dims else 0.0,
    }


def _group_finite_means(values, *, group_indices: list[int], group_count: int) -> list[float]:
    values_t = torch.as_tensor(values, dtype=torch.float32).reshape(-1)
    out = []
    for group_idx in range(int(group_count)):
        picked = values_t[[i for i, g in enumerate(group_indices) if int(g) == int(group_idx)]]
        finite = picked[torch.isfinite(picked)]
        if int(finite.numel()) <= 0:
            out.append(float("nan"))
        else:
            out.append(float(finite.mean().item()))
    return out


def _finite_mean_std_from_list(values: list[float]) -> tuple[float, float]:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    finite = tensor[torch.isfinite(tensor)]
    if int(finite.numel()) <= 0:
        return float("nan"), float("nan")
    return float(finite.mean().item()), _safe_std(finite)


def _same_env_paired_summary(
    *,
    train_suite: dict[str, Any],
    heldout_suite: dict[str, Any],
    train_metrics: dict[str, Any],
    heldout_metrics: dict[str, Any],
) -> dict[str, Any]:
    train_base_env_indices = [int(v) for v in train_suite.get("base_env_indices", [])]
    heldout_base_env_indices = [int(v) for v in heldout_suite.get("base_env_indices", [])]
    if not train_base_env_indices or not heldout_base_env_indices:
        raise ValueError("same_env paired summary requires base_env_indices on both suites")

    train_base_batch_size = int(train_suite.get("base_batch_size", 0))
    heldout_base_batch_size = int(heldout_suite.get("base_batch_size", 0))
    if train_base_batch_size != heldout_base_batch_size:
        raise ValueError("same_env paired summary requires matching base_batch_size")

    base_batch_size = int(train_base_batch_size)
    train_suffix = _group_finite_means(
        train_metrics["suffix_return_per_env"],
        group_indices=train_base_env_indices,
        group_count=base_batch_size,
    )
    heldout_suffix = _group_finite_means(
        heldout_metrics["suffix_return_per_env"],
        group_indices=heldout_base_env_indices,
        group_count=base_batch_size,
    )
    train_full = _group_finite_means(
        train_metrics["full_return_per_env"],
        group_indices=train_base_env_indices,
        group_count=base_batch_size,
    )
    heldout_full = _group_finite_means(
        heldout_metrics["full_return_per_env"],
        group_indices=heldout_base_env_indices,
        group_count=base_batch_size,
    )
    suffix_delta = [
        float(held - train) if math.isfinite(train) and math.isfinite(held) else float("nan")
        for train, held in zip(train_suffix, heldout_suffix)
    ]
    full_delta = [
        float(held - train) if math.isfinite(train) and math.isfinite(held) else float("nan")
        for train, held in zip(train_full, heldout_full)
    ]
    suffix_delta_mean, suffix_delta_std = _finite_mean_std_from_list(suffix_delta)
    full_delta_mean, full_delta_std = _finite_mean_std_from_list(full_delta)

    return {
        "base_env_batch_size": int(base_batch_size),
        "train_rollouts_per_env": int(train_suite.get("rollouts_per_env", 1)),
        "heldout_rollouts_per_env": int(heldout_suite.get("rollouts_per_env", 1)),
        "train_suffix_return_mean_per_base_env": train_suffix,
        "heldout_suffix_return_mean_per_base_env": heldout_suffix,
        "heldout_minus_train_suffix_return_mean_per_base_env": suffix_delta,
        "heldout_minus_train_suffix_return_mean": float(suffix_delta_mean),
        "heldout_minus_train_suffix_return_std": float(suffix_delta_std),
        "train_full_return_mean_per_base_env": train_full,
        "heldout_full_return_mean_per_base_env": heldout_full,
        "heldout_minus_train_full_return_mean_per_base_env": full_delta,
        "heldout_minus_train_full_return_mean": float(full_delta_mean),
        "heldout_minus_train_full_return_std": float(full_delta_std),
    }


def sample_fixed_prior_suite(
    prior: EnvironmentPrior,
    *,
    batch_size: int,
    suite_seed: int,
    freeze_reward_dropout: bool = True,
) -> dict[str, Any]:
    with _temporary_seed(int(suite_seed)):
        h_list = prior._sample_batch_hypers(int(batch_size))
        if freeze_reward_dropout:
            h_list = _freeze_env_h_list_for_replay(prior, list(h_list))
        env_seeds = prior._sample_seed_list(int(batch_size))
        rollout_seeds = prior._sample_seed_list(int(batch_size))
        # Freeze latent per-environment semantics into the stored h_list so the
        # same suite replays exactly rather than re-sampling hidden split knobs
        # the first time a later audit touches the suite.
        for h, env_seed in zip(h_list, env_seeds):
            prior._sample_environment(h=h, device="cpu", rng_seed=int(env_seed))
    return {
        "h_list": list(h_list),
        "env_seeds": [int(v) for v in env_seeds],
        "rollout_seeds": [int(v) for v in rollout_seeds],
        "batch_size": int(batch_size),
        "suite_seed": int(suite_seed),
    }


def build_rollout_resample_suite(
    prior: EnvironmentPrior,
    *,
    base_suite: dict[str, Any],
    rollout_suite_seed: int,
    rollouts_per_env: int,
) -> dict[str, Any]:
    base_h_list = list(base_suite["h_list"])
    base_env_seeds = [int(v) for v in base_suite["env_seeds"]]
    base_batch_size = int(len(base_h_list))
    repeats = int(max(1, int(rollouts_per_env)))

    with _temporary_seed(int(rollout_suite_seed)):
        rollout_seeds = prior._sample_seed_list(int(base_batch_size * repeats))

    h_list = []
    env_seeds = []
    base_env_indices = []
    for base_idx, (h, env_seed) in enumerate(zip(base_h_list, base_env_seeds)):
        for _ in range(repeats):
            h_list.append(copy.deepcopy(h))
            env_seeds.append(int(env_seed))
            base_env_indices.append(int(base_idx))

    return {
        "h_list": h_list,
        "env_seeds": env_seeds,
        "rollout_seeds": [int(v) for v in rollout_seeds],
        "batch_size": int(base_batch_size * repeats),
        "suite_seed": None if base_suite.get("suite_seed", None) is None else int(base_suite["suite_seed"]),
        "base_batch_size": int(base_batch_size),
        "rollouts_per_env": int(repeats),
        "rollout_suite_seed": int(rollout_suite_seed),
        "base_env_indices": [int(v) for v in base_env_indices],
    }


def _resolve_env_value_vector(
    env_info: dict[str, Any],
    key: str,
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    default: float,
) -> torch.Tensor:
    value = env_info.get(key, None)
    if value is None:
        return torch.full((int(batch_size),), float(default), device=device, dtype=dtype)
    if torch.is_tensor(value):
        out = value.to(device=device, dtype=dtype)
    else:
        out = torch.as_tensor(value, device=device, dtype=dtype)
    if out.ndim == 0:
        return out.reshape(1).expand(int(batch_size))
    if out.ndim == 2 and int(out.shape[-1]) == 1:
        out = out.reshape(-1)
    elif out.ndim > 1:
        out = out.reshape(int(batch_size), -1)[:, 0]
    if int(out.numel()) != int(batch_size):
        raise ValueError(f"{key} must broadcast to batch_size={int(batch_size)}, got shape={tuple(out.shape)}")
    return out.reshape(int(batch_size))


def _build_zero_policy_step_fn():
    def _policy_step_fn(obs_t, action_t, reward_t, reward_mask_t, cache, step_idx, env_info):
        del reward_t, reward_mask_t, cache, step_idx, env_info
        return torch.zeros_like(action_t[:, : int(action_t.shape[-1])], dtype=obs_t.dtype)

    return _policy_step_fn


def _build_constant_policy_step_fn(action_value: float):
    action_value_f = float(action_value)

    def _policy_step_fn(obs_t, action_t, reward_t, reward_mask_t, cache, step_idx, env_info):
        del obs_t, reward_t, reward_mask_t, cache, step_idx, env_info
        return torch.full_like(action_t[:, : int(action_t.shape[-1])], action_value_f, dtype=action_t.dtype)

    return _policy_step_fn


def _build_model_policy_step_fn(*, model, num_features: int, n_samples: int):
    return _build_policy_step_fn(
        model,
        num_features=int(num_features),
        max_cache_len=int(n_samples),
        kv_cache_mode="paged",
        kv_cache_page_size=128,
        allow_grad_mutable_cache=True,
        pg_torch_compile=False,
    )


def _build_ppo_policy_step_fn(*, model, env_cfg: dict[str, Any], device: torch.device, num_features: int):
    from ticl.rl_validation import _resolve_validation_recurrent_ppo_policy

    optimizer_cfg = model.__dict__.get("_audit_optimizer_cfg", {})
    policy = _resolve_validation_recurrent_ppo_policy(
        model=model,
        config={"optimizer": optimizer_cfg},
        env_cfg=env_cfg,
        device=device,
        num_features=int(num_features),
    )
    base_policy_step_fn = policy.make_vectorized_rollout_step_fn()

    def _clear_buffers():
        reset_fn = getattr(policy, "reset_rollout_cache", None)
        if callable(reset_fn):
            batch_size = int(getattr(policy, "_rollout_cache_batch_size", 1) or 1)
            reset_fn(batch_size)
        return None

    def _broadcast_env_value(value, *, batch_size: int, device: torch.device, dtype: torch.dtype):
        if torch.is_tensor(value):
            out = value.to(device=device, dtype=dtype)
        else:
            out = torch.as_tensor(value, device=device, dtype=dtype)
        if out.ndim == 0:
            out = out.reshape(1).expand(int(batch_size))
        else:
            out = out.reshape(-1)
            if int(out.numel()) == 1 and int(batch_size) != 1:
                out = out.expand(int(batch_size))
        if int(out.numel()) != int(batch_size):
            raise ValueError(
                f"Expected env value to broadcast to batch_size={int(batch_size)}, got shape={tuple(out.shape)}"
            )
        return out.reshape(int(batch_size))

    def _policy_step_fn(obs_t, action_t, reward_t, reward_mask_t, cache, step_idx, env_info):
        batch_size = int(obs_t.shape[0])
        env_info_aug = dict(env_info)
        if "action_dim_per_sample" not in env_info_aug:
            env_info_aug["action_dim_per_sample"] = _broadcast_env_value(
                env_info.get("action_dim", int(action_t.shape[-1])),
                batch_size=batch_size,
                device=obs_t.device,
                dtype=torch.long,
            )
        if "phase_t" in env_info_aug and not torch.is_tensor(env_info_aug["phase_t"]):
            env_info_aug["phase_t"] = _broadcast_env_value(
                env_info_aug["phase_t"],
                batch_size=batch_size,
                device=obs_t.device,
                dtype=obs_t.dtype,
            )
        if "terminal_t" in env_info_aug and not torch.is_tensor(env_info_aug["terminal_t"]):
            env_info_aug["terminal_t"] = _broadcast_env_value(
                env_info_aug["terminal_t"],
                batch_size=batch_size,
                device=obs_t.device,
                dtype=obs_t.dtype,
            )
        return base_policy_step_fn(
            obs_t,
            action_t,
            reward_t,
            reward_mask_t,
            cache,
            step_idx,
            env_info_aug,
        )

    _policy_step_fn._clear_buffers = _clear_buffers
    sample_fn = getattr(base_policy_step_fn, "_policy_actor_sample_fn", None)
    if callable(sample_fn):
        _policy_step_fn._policy_actor_sample_fn = sample_fn
    _policy_step_fn._audit_policy = policy
    return _policy_step_fn


def _wrap_policy_step_fn_for_single_env(policy_step_fn):
    def _maybe_batch_tensor(value):
        if not torch.is_tensor(value):
            return value
        if value.ndim == 0:
            return value.reshape(1)
        if value.ndim == 1:
            return value.unsqueeze(0)
        return value

    def _maybe_unbatch_tensor(value):
        if not torch.is_tensor(value):
            return value
        if value.ndim >= 2 and int(value.shape[0]) == 1:
            return value.squeeze(0)
        return value

    def _maybe_unbatch_actor_outputs(value):
        if not isinstance(value, dict):
            return value
        out = {}
        for key, item in value.items():
            out[key] = _maybe_unbatch_tensor(item)
        return out

    def _single_env_policy_step_fn(obs_t, action_t, reward_t, reward_mask_t, cache, step_idx, env_info):
        out = policy_step_fn(
            _maybe_batch_tensor(obs_t),
            _maybe_batch_tensor(action_t),
            _maybe_batch_tensor(reward_t),
            _maybe_batch_tensor(reward_mask_t),
            cache,
            step_idx,
            env_info,
        )
        if isinstance(out, tuple):
            if len(out) == 0:
                return out
            return (_maybe_unbatch_actor_outputs(_maybe_unbatch_tensor(out[0])), *out[1:])
        return _maybe_unbatch_actor_outputs(_maybe_unbatch_tensor(out))

    clear_fn = getattr(policy_step_fn, "_clear_buffers", None)
    if callable(clear_fn):
        _single_env_policy_step_fn._clear_buffers = clear_fn
    return _single_env_policy_step_fn


def _load_model_and_config(checkpoint_path: str, *, device: torch.device):
    model, config = load_model(checkpoint_path, device=device, verbose=False)
    return model, config


def _resolve_audit_env_config(
    base_config: dict[str, Any],
    *,
    core_a: bool,
    reference_semantics_enabled: bool,
) -> dict[str, Any]:
    env_cfg = dict(base_config["prior"]["environment"])
    if core_a:
        env_cfg.update(dict(CORE_A_ENV_OVERRIDES))
    if reference_semantics_enabled:
        env_cfg["reference_semantics_enabled"] = True
    return env_cfg


def _resolve_policy_source(
    *,
    policy_mode: str,
    checkpoint_path: Optional[str],
    device: torch.device,
):
    resolved_mode = str(policy_mode).strip().lower()
    config = get_model_default_config("rlpfn")
    model = None

    if resolved_mode == "zero":
        return {
            "policy_step_fn": _build_zero_policy_step_fn(),
            "resolved_mode": "zero",
            "model": None,
            "config": config,
        }

    if checkpoint_path is None:
        raise ValueError("checkpoint_path is required for policy_mode auto/model/ppo")

    model, config = _load_model_and_config(checkpoint_path, device=device)
    optimizer_cfg = dict(config.get("optimizer", {}))
    has_saved_ppo_state = isinstance(model.__dict__.get("_validation_ppo_policy_state", None), dict)
    uses_ppo = str(optimizer_cfg.get("rl_objective", "")).strip().lower() == "ppo"

    if resolved_mode == "auto":
        resolved_mode = "ppo" if (uses_ppo and has_saved_ppo_state) else "model"

    if resolved_mode not in {"model", "ppo"}:
        raise ValueError(f"Unknown policy_mode={policy_mode!r}")

    return {
        "resolved_mode": resolved_mode,
        "model": model,
        "config": config,
    }


def _maybe_clear_policy_buffers(policy_step_fn) -> None:
    clear_fn = getattr(policy_step_fn, "_clear_buffers", None)
    if callable(clear_fn):
        clear_fn()


def _collect_suite_rewards_serial(
    prior: EnvironmentPrior,
    *,
    suite: dict[str, Any],
    policy_step_fn,
    n_samples: int,
    num_features: int,
    single_eval_pos: int,
    device: torch.device,
    collect_x: bool,
):
    batch_size = int(suite["batch_size"])
    single_env_policy_step_fn = _wrap_policy_step_fn_for_single_env(policy_step_fn)
    reward_steps = []
    x_steps = [] if collect_x else None
    infos = []
    for h, env_seed, rollout_seed in zip(
        list(suite["h_list"]),
        list(suite["env_seeds"]),
        list(suite["rollout_seeds"]),
    ):
        _maybe_clear_policy_buffers(single_env_policy_step_fn)
        env = prior._sample_environment(h, device=device, rng_seed=int(env_seed))
        x_one, y_one, info_one = prior._rollout_single(
            env=env,
            n_samples=int(n_samples),
            num_features=int(num_features),
            single_eval_pos=int(single_eval_pos),
            device=device,
            policy_step_fn=single_env_policy_step_fn,
            collect_x=bool(collect_x),
            collect_runtime_info=True,
            rng_seed=int(rollout_seed),
            store_rewards=True,
            policy_objective_kind="policy_gradient",
        )
        y_cpu = y_one.detach().cpu().to(dtype=torch.float32)
        if y_cpu.ndim == 0:
            y_cpu = y_cpu.reshape(1)
        reward_steps.append(y_cpu)
        if collect_x:
            x_cpu = x_one.detach().cpu().to(dtype=torch.float32)
            x_steps.append(x_cpu)
        infos.append(info_one)

    if batch_size <= 0:
        rewards = torch.empty((int(n_samples), 0), dtype=torch.float32)
        xs = torch.empty((int(n_samples), 0, int(num_features)), dtype=torch.float32) if collect_x else None
    else:
        rewards = torch.stack(reward_steps, dim=1)
        xs = torch.stack(x_steps, dim=1) if collect_x else None
    return rewards, xs, infos


def collect_suite_rewards_serial_profiled(
    prior: EnvironmentPrior,
    *,
    suite: dict[str, Any],
    policy_step_fn,
    n_samples: int,
    num_features: int,
    single_eval_pos: int,
    device: torch.device,
    max_envs: int | None = None,
) -> dict[str, Any]:
    batch_size = int(suite["batch_size"])
    limit = batch_size if max_envs is None else min(int(max_envs), batch_size)
    single_env_policy_step_fn = _wrap_policy_step_fn_for_single_env(policy_step_fn)
    env_profiles = []
    reward_steps = []
    infos = []
    for env_idx, (h, env_seed, rollout_seed) in enumerate(
        zip(
            list(suite["h_list"]),
            list(suite["env_seeds"]),
            list(suite["rollout_seeds"]),
        )
    ):
        if env_idx >= limit:
            break
        _maybe_clear_policy_buffers(single_env_policy_step_fn)
        env_start = time.perf_counter()
        with torch.no_grad():
            env = prior._sample_environment(h, device=device, rng_seed=int(env_seed))
            after_env_build = time.perf_counter()
            _, y_one, info_one = prior._rollout_single(
                env=env,
                n_samples=int(n_samples),
                num_features=int(num_features),
                single_eval_pos=int(single_eval_pos),
                device=device,
                policy_step_fn=single_env_policy_step_fn,
                collect_x=False,
                collect_runtime_info=True,
                rng_seed=int(rollout_seed),
                store_rewards=True,
                policy_objective_kind="policy_gradient",
            )
        env_end = time.perf_counter()
        y_cpu = y_one.detach().cpu().to(dtype=torch.float32)
        if y_cpu.ndim == 0:
            y_cpu = y_cpu.reshape(1)
        reward_steps.append(y_cpu)
        infos.append(info_one)
        reward_np = y_cpu.numpy()
        prefix_steps = min(int(single_eval_pos), int(reward_np.shape[0]))
        suffix = reward_np[prefix_steps:]
        env_profiles.append(
            {
                "env_index": int(env_idx),
                "env_seed": int(env_seed),
                "rollout_seed": int(rollout_seed),
                "env_build_s": float(after_env_build - env_start),
                "rollout_s": float(env_end - after_env_build),
                "total_s": float(env_end - env_start),
                "full_return_sum": float(reward_np.sum()),
                "suffix_return_sum": float(suffix.sum()) if suffix.size > 0 else 0.0,
                "n_reward_steps": int(reward_np.shape[0]),
                "n_suffix_steps": int(suffix.size),
            }
        )

    if limit <= 0:
        rewards = torch.empty((int(n_samples), 0), dtype=torch.float32)
    else:
        rewards = torch.stack(reward_steps, dim=1)
    full_return_per_env = rewards.sum(dim=0) if rewards.numel() > 0 else torch.empty((0,), dtype=torch.float32)
    effective_single_eval_pos = min(int(single_eval_pos), int(rewards.shape[0]))
    suffix_rewards = rewards[effective_single_eval_pos:]
    suffix_return_per_env = (
        suffix_rewards.sum(dim=0) if suffix_rewards.numel() > 0 else torch.zeros((rewards.shape[1],), dtype=torch.float32)
    )
    return {
        "suite_batch_size": int(batch_size),
        "profiled_env_count": int(limit),
        "env_profiles": env_profiles,
        "aggregate": {
            "full_return_mean": float(full_return_per_env.mean().item()) if full_return_per_env.numel() > 0 else 0.0,
            "suffix_return_mean": float(suffix_return_per_env.mean().item()) if suffix_return_per_env.numel() > 0 else 0.0,
            "full_return_std": float(full_return_per_env.std(unbiased=False).item()) if full_return_per_env.numel() > 0 else 0.0,
            "suffix_return_std": float(suffix_return_per_env.std(unbiased=False).item()) if suffix_return_per_env.numel() > 0 else 0.0,
            "total_env_build_s": float(sum(item["env_build_s"] for item in env_profiles)),
            "total_rollout_s": float(sum(item["rollout_s"] for item in env_profiles)),
            "total_wall_s": float(sum(item["total_s"] for item in env_profiles)),
        },
        "infos": infos,
    }


def evaluate_prior_suite(
    prior: EnvironmentPrior,
    *,
    suite: dict[str, Any],
    policy_step_fn,
    n_samples: int,
    num_features: int,
    single_eval_pos: int,
    device: torch.device,
    rollout_backend: str = "family_vectorized",
) -> dict[str, Any]:
    resolved_backend = str(rollout_backend).strip().lower()
    if resolved_backend not in {"family_vectorized", "serial"}:
        raise ValueError(f"Unknown rollout_backend={rollout_backend!r}")

    reward_components = None
    terminal_counts = None
    if resolved_backend == "serial":
        with torch.no_grad():
            rewards, _, _ = _collect_suite_rewards_serial(
                prior,
                suite=suite,
                policy_step_fn=policy_step_fn,
                n_samples=int(n_samples),
                num_features=int(num_features),
                single_eval_pos=int(single_eval_pos),
                device=device,
                collect_x=False,
            )
        effective_single_eval_pos = int(single_eval_pos)
    else:
        _maybe_clear_policy_buffers(policy_step_fn)
        with torch.no_grad():
            rollout = prior.rollout_with_policy(
                policy_step_fn=policy_step_fn,
                batch_size=int(suite["batch_size"]),
                n_samples=int(n_samples),
                num_features=int(num_features),
                device=device,
                single_eval_pos=int(single_eval_pos),
                collect_x=False,
                collect_runtime_info=True,
                store_rewards=True,
                policy_objective_kind="policy_gradient",
                h_list_override=list(suite["h_list"]),
                env_seeds_override=list(suite["env_seeds"]),
                rollout_seeds_override=list(suite["rollout_seeds"]),
            )
        rewards = rollout["rewards"].detach().cpu().to(dtype=torch.float32)
        effective_single_eval_pos = int(rollout.get("single_eval_pos", single_eval_pos))
        reward_components = getattr(prior, "last_rollout_reward_components", None)
        terminal_counts = getattr(prior, "last_rollout_eval_terminal_counts", None)

    suffix_rewards = rewards[effective_single_eval_pos:]
    full_return = rewards.sum(dim=0)
    suffix_return = suffix_rewards.sum(dim=0) if int(suffix_rewards.shape[0]) > 0 else torch.zeros_like(full_return)

    metrics = {
        "rollout_backend": str(resolved_backend),
        "batch_size": int(suite["batch_size"]),
        "n_samples": int(rewards.shape[0]),
        "single_eval_pos": int(effective_single_eval_pos),
        "suffix_horizon": int(max(0, int(rewards.shape[0]) - int(effective_single_eval_pos))),
        "full_return_per_env": _to_builtin(full_return),
        "suffix_return_per_env": _to_builtin(suffix_return),
    }
    metrics.update(_finite_stats(full_return, prefix="full_return"))
    metrics.update(_finite_stats(suffix_return, prefix="suffix_return"))
    if int(rewards.numel()) > 0:
        metrics.update(_finite_stats(rewards, prefix="reward_step"))
    else:
        metrics["reward_step_mean"] = 0.0
        metrics["reward_step_std"] = 0.0
        metrics["reward_step_nonfinite_count"] = 0
        metrics["reward_step_nonfinite_share"] = 0.0
    if int(suffix_rewards.numel()) > 0:
        metrics.update(_finite_stats(suffix_rewards, prefix="suffix_reward_step"))
    else:
        metrics["suffix_reward_step_mean"] = 0.0
        metrics["suffix_reward_step_std"] = 0.0
        metrics["suffix_reward_step_nonfinite_count"] = 0
        metrics["suffix_reward_step_nonfinite_share"] = 0.0

    if isinstance(reward_components, dict):
        for key, value in reward_components.items():
            if not torch.is_tensor(value):
                continue
            value_t = value.detach().cpu().to(dtype=torch.float32)
            if value_t.ndim == 1:
                value_t = value_t.unsqueeze(0)
            comp_name = str(key).strip().lower()
            comp_return = value_t.sum(dim=0)
            metrics.update(_finite_stats(comp_return, prefix=f"{comp_name}_return"))
            metrics.update(_finite_stats(value_t, prefix=comp_name))

    if torch.is_tensor(terminal_counts):
        terminal_counts_cpu = terminal_counts.detach().cpu().to(dtype=torch.float32)
        metrics["suffix_terminal_count_mean"] = float(terminal_counts_cpu.mean().item())
        metrics["suffix_terminal_count_std"] = _safe_std(terminal_counts_cpu)

    return metrics


def evaluate_action_reachability(
    prior: EnvironmentPrior,
    *,
    suite: dict[str, Any],
    n_samples: int,
    num_features: int,
    single_eval_pos: int,
    device: torch.device,
    action_low: float = 0.0,
    action_high: float = 100.0,
) -> dict[str, Any]:
    low_policy = _build_constant_policy_step_fn(action_low)
    high_policy = _build_constant_policy_step_fn(action_high)
    rewards_low, x_low, _ = _collect_suite_rewards_serial(
        prior,
        suite=suite,
        policy_step_fn=low_policy,
        n_samples=int(n_samples),
        num_features=int(num_features),
        single_eval_pos=int(single_eval_pos),
        device=device,
        collect_x=True,
    )
    rewards_high, x_high, _ = _collect_suite_rewards_serial(
        prior,
        suite=suite,
        policy_step_fn=high_policy,
        n_samples=int(n_samples),
        num_features=int(num_features),
        single_eval_pos=int(single_eval_pos),
        device=device,
        collect_x=True,
    )

    effective_single_eval_pos = int(single_eval_pos)
    suffix_low = rewards_low[effective_single_eval_pos:].sum(dim=0) if int(rewards_low.shape[0]) > effective_single_eval_pos else torch.zeros((int(suite["batch_size"]),), dtype=torch.float32)
    suffix_high = rewards_high[effective_single_eval_pos:].sum(dim=0) if int(rewards_high.shape[0]) > effective_single_eval_pos else torch.zeros((int(suite["batch_size"]),), dtype=torch.float32)
    reward_step0_delta_abs = (rewards_low[0] - rewards_high[0]).abs() if int(rewards_low.shape[0]) > 0 else torch.zeros((int(suite["batch_size"]),), dtype=torch.float32)
    suffix_return_delta_abs = (suffix_low - suffix_high).abs()

    obs_max_abs_delta = torch.zeros((int(suite["batch_size"]),), dtype=torch.float32)
    for env_idx, h in enumerate(list(suite["h_list"])):
        obs_dim = int(h.get("obs_dim", prior._sample_dims(h)[1]))
        obs_cap = int(max(0, min(obs_dim, int(x_low.shape[-1]))))
        if obs_cap <= 0 or int(x_low.shape[0]) <= 1:
            continue
        obs_delta = (x_low[1:, env_idx, :obs_cap] - x_high[1:, env_idx, :obs_cap]).abs()
        if int(obs_delta.numel()) > 0:
            obs_max_abs_delta[env_idx] = obs_delta.max()

    def _lt_share(values: torch.Tensor, threshold: float) -> float:
        finite = values[torch.isfinite(values)]
        if int(finite.numel()) <= 0:
            return float("nan")
        return float((finite < float(threshold)).to(dtype=torch.float32).mean().item())

    def _gt_share(values: torch.Tensor, threshold: float) -> float:
        finite = values[torch.isfinite(values)]
        if int(finite.numel()) <= 0:
            return float("nan")
        return float((finite > float(threshold)).to(dtype=torch.float32).mean().item())

    finite_reward = suffix_return_delta_abs[torch.isfinite(suffix_return_delta_abs)]
    finite_obs = obs_max_abs_delta[torch.isfinite(obs_max_abs_delta)]
    state_only_mask = torch.isfinite(obs_max_abs_delta) & torch.isfinite(suffix_return_delta_abs)
    state_only_mask = state_only_mask & (obs_max_abs_delta > 1e-2) & (suffix_return_delta_abs < 1e-3)

    metrics = {
        "probe_backend": "serial",
        "action_low": float(action_low),
        "action_high": float(action_high),
        "single_eval_pos": int(effective_single_eval_pos),
        "reward_step0_delta_abs_per_env": _to_builtin(reward_step0_delta_abs),
        "suffix_return_delta_abs_per_env": _to_builtin(suffix_return_delta_abs),
        "obs_max_abs_delta_per_env": _to_builtin(obs_max_abs_delta),
        "suffix_return_delta_abs_mean": float(finite_reward.mean().item()) if int(finite_reward.numel()) > 0 else float("nan"),
        "suffix_return_delta_abs_std": _safe_std(finite_reward) if int(finite_reward.numel()) > 0 else float("nan"),
        "obs_max_abs_delta_mean": float(finite_obs.mean().item()) if int(finite_obs.numel()) > 0 else float("nan"),
        "obs_max_abs_delta_std": _safe_std(finite_obs) if int(finite_obs.numel()) > 0 else float("nan"),
        "suffix_return_delta_abs_lt_1e_3_share": _lt_share(suffix_return_delta_abs, 1e-3),
        "suffix_return_delta_abs_lt_1e_2_share": _lt_share(suffix_return_delta_abs, 1e-2),
        "suffix_return_delta_abs_lt_1e_1_share": _lt_share(suffix_return_delta_abs, 1e-1),
        "suffix_return_delta_abs_lt_1e0_share": _lt_share(suffix_return_delta_abs, 1.0),
        "obs_max_abs_delta_gt_1e_2_share": _gt_share(obs_max_abs_delta, 1e-2),
        "state_only_sensitive_share_obs_gt_1e_2_reward_lt_1e_3": float(state_only_mask.to(dtype=torch.float32).mean().item()) if int(state_only_mask.numel()) > 0 else float("nan"),
    }
    metrics.update(_finite_stats(reward_step0_delta_abs, prefix="reward_step0_delta_abs"))
    metrics.update(_finite_stats(suffix_return_delta_abs, prefix="suffix_return_delta_abs"))
    metrics.update(_finite_stats(obs_max_abs_delta, prefix="obs_max_abs_delta"))
    return metrics


def run_prior_generalization_audit(
    *,
    checkpoint_path: Optional[str] = None,
    policy_mode: str = "auto",
    device: Optional[str] = None,
    n_samples: Optional[int] = None,
    num_features: Optional[int] = None,
    single_eval_pos: Optional[int] = None,
    train_suite_batch_size: int = 16,
    heldout_suite_batch_size: int = 16,
    train_suite_seed: int = 12345,
    heldout_suite_seed: int = 67890,
    suite_regime: str = "cross_env",
    base_suite_batch_size: Optional[int] = None,
    base_suite_seed: int = 12345,
    train_rollouts_per_env: int = 1,
    heldout_rollouts_per_env: int = 1,
    train_rollout_seed: int = 12345,
    heldout_rollout_seed: int = 67890,
    base_suite_path: Optional[str] = None,
    train_suite_path: Optional[str] = None,
    heldout_suite_path: Optional[str] = None,
    suite_save_dir: Optional[str] = None,
    core_a: bool = False,
    reference_semantics_enabled: bool = False,
    rollout_backend: str = "family_vectorized",
    include_action_reachability_probe: bool = True,
) -> dict[str, Any]:
    device_obj = torch.device(str(device or _default_device()))
    seedless_base_config = get_model_default_config("rlpfn")

    policy_bundle = _resolve_policy_source(
        policy_mode=policy_mode,
        checkpoint_path=checkpoint_path,
        device=device_obj,
    )
    config = policy_bundle["config"]
    env_cfg = _resolve_audit_env_config(
        config,
        core_a=bool(core_a),
        reference_semantics_enabled=bool(reference_semantics_enabled),
    )
    prior = EnvironmentPrior(env_cfg)

    resolved_num_features = int(num_features or config["prior"]["num_features"])
    resolved_n_samples = int(n_samples or config["prior"]["n_samples"])
    resolved_single_eval_pos = int(
        single_eval_pos
        if single_eval_pos is not None
        else _resolve_default_single_eval_pos(config, resolved_n_samples)
    )
    if report_mode := policy_bundle["resolved_mode"]:
        if report_mode == "zero":
            policy_step_fn = policy_bundle["policy_step_fn"]
        elif report_mode == "model":
            policy_step_fn = _build_model_policy_step_fn(
                model=policy_bundle["model"],
                num_features=resolved_num_features,
                n_samples=resolved_n_samples,
            )
        elif report_mode == "ppo":
            optimizer_cfg = dict(config.get("optimizer", {}))
            policy_bundle["model"].__dict__["_audit_optimizer_cfg"] = optimizer_cfg
            policy_step_fn = _build_ppo_policy_step_fn(
                model=policy_bundle["model"],
                env_cfg=env_cfg,
                device=device_obj,
                num_features=resolved_num_features,
            )
        else:
            raise ValueError(f"Unsupported resolved policy mode: {report_mode!r}")
    else:
        raise ValueError("Policy resolution returned an empty mode")

    resolved_suite_regime = str(suite_regime).strip().lower()
    if resolved_suite_regime not in {"cross_env", "same_env"}:
        raise ValueError(f"Unknown suite_regime={suite_regime!r}")

    base_suite = None
    train_suite = _maybe_load_suite(train_suite_path)
    heldout_suite = _maybe_load_suite(heldout_suite_path)

    if resolved_suite_regime == "same_env":
        base_suite = _maybe_load_suite(base_suite_path)
        if base_suite is None:
            effective_base_batch_size = int(base_suite_batch_size or train_suite_batch_size)
            base_suite = sample_fixed_prior_suite(
                prior,
                batch_size=int(effective_base_batch_size),
                suite_seed=int(base_suite_seed),
            )
        if train_suite is None:
            train_suite = build_rollout_resample_suite(
                prior,
                base_suite=base_suite,
                rollout_suite_seed=int(train_rollout_seed),
                rollouts_per_env=int(train_rollouts_per_env),
            )
        if heldout_suite is None:
            heldout_suite = build_rollout_resample_suite(
                prior,
                base_suite=base_suite,
                rollout_suite_seed=int(heldout_rollout_seed),
                rollouts_per_env=int(heldout_rollouts_per_env),
            )
    else:
        if train_suite is None:
            train_suite = sample_fixed_prior_suite(
                prior,
                batch_size=int(train_suite_batch_size),
                suite_seed=int(train_suite_seed),
            )
        if heldout_suite is None:
            heldout_suite = sample_fixed_prior_suite(
                prior,
                batch_size=int(heldout_suite_batch_size),
                suite_seed=int(heldout_suite_seed),
            )

    saved_suite_paths = {}
    if suite_save_dir is not None:
        suite_dir = Path(suite_save_dir).expanduser().resolve()
        suite_dir.mkdir(parents=True, exist_ok=True)
        if base_suite is not None:
            saved_suite_paths["base"] = save_prior_suite(str(suite_dir / "base_suite.pt"), base_suite)
        saved_suite_paths["train"] = save_prior_suite(str(suite_dir / "train_suite.pt"), train_suite)
        saved_suite_paths["heldout"] = save_prior_suite(str(suite_dir / "heldout_suite.pt"), heldout_suite)

    train_metrics = evaluate_prior_suite(
        prior,
        suite=train_suite,
        policy_step_fn=policy_step_fn,
        n_samples=resolved_n_samples,
        num_features=resolved_num_features,
        single_eval_pos=resolved_single_eval_pos,
        device=device_obj,
        rollout_backend=rollout_backend,
    )
    heldout_metrics = evaluate_prior_suite(
        prior,
        suite=heldout_suite,
        policy_step_fn=policy_step_fn,
        n_samples=resolved_n_samples,
        num_features=resolved_num_features,
        single_eval_pos=resolved_single_eval_pos,
        device=device_obj,
        rollout_backend=rollout_backend,
    )

    base_action_reachability = None
    train_action_reachability = None
    heldout_action_reachability = None
    if bool(include_action_reachability_probe):
        if base_suite is not None:
            base_action_reachability = evaluate_action_reachability(
                prior,
                suite=base_suite,
                n_samples=resolved_n_samples,
                num_features=resolved_num_features,
                single_eval_pos=resolved_single_eval_pos,
                device=device_obj,
            )
        train_action_reachability = evaluate_action_reachability(
            prior,
            suite=train_suite,
            n_samples=resolved_n_samples,
            num_features=resolved_num_features,
            single_eval_pos=resolved_single_eval_pos,
            device=device_obj,
        )
        heldout_action_reachability = evaluate_action_reachability(
            prior,
            suite=heldout_suite,
            n_samples=resolved_n_samples,
            num_features=resolved_num_features,
            single_eval_pos=resolved_single_eval_pos,
            device=device_obj,
        )

    comparison = {
        "heldout_minus_train_full_return_mean": float(
            heldout_metrics["full_return_mean"] - train_metrics["full_return_mean"]
        ),
        "heldout_minus_train_suffix_return_mean": float(
            heldout_metrics["suffix_return_mean"] - train_metrics["suffix_return_mean"]
        ),
    }
    for key in sorted(set(train_metrics.keys()) & set(heldout_metrics.keys())):
        if str(key).endswith("_return_mean") and isinstance(train_metrics[key], float) and isinstance(heldout_metrics[key], float):
            comparison[f"heldout_minus_train_{key}"] = float(heldout_metrics[key] - train_metrics[key])

    paired_metrics = None
    if resolved_suite_regime == "same_env":
        paired_metrics = _same_env_paired_summary(
            train_suite=train_suite,
            heldout_suite=heldout_suite,
            train_metrics=train_metrics,
            heldout_metrics=heldout_metrics,
        )

    report = {
        "audit_entry": "prior_generalization_audit",
        "checkpoint_path": None if checkpoint_path is None else str(Path(checkpoint_path).expanduser().resolve()),
        "policy": {
            "requested_mode": str(policy_mode),
            "resolved_mode": str(policy_bundle["resolved_mode"]),
            "device": str(device_obj),
        },
        "config": {
            "n_samples": int(resolved_n_samples),
            "num_features": int(resolved_num_features),
            "single_eval_pos": int(resolved_single_eval_pos),
            "suite_regime": str(resolved_suite_regime),
            "core_a": bool(core_a),
            "reference_semantics_enabled": bool(reference_semantics_enabled),
            "rollout_backend": str(rollout_backend),
            "include_action_reachability_probe": bool(include_action_reachability_probe),
        },
        "suite_paths": saved_suite_paths,
        "base_suite": (
            {
                "summary": _summarize_suite(prior, base_suite),
                "action_reachability": base_action_reachability,
            }
            if base_suite is not None
            else None
        ),
        "train_suite": {
            "summary": _summarize_suite(prior, train_suite),
            "metrics": train_metrics,
            "action_reachability": train_action_reachability,
        },
        "heldout_suite": {
            "summary": _summarize_suite(prior, heldout_suite),
            "metrics": heldout_metrics,
            "action_reachability": heldout_action_reachability,
        },
        "comparison": comparison,
        "paired_metrics": paired_metrics,
    }
    return report


def _default_output_path(*, checkpoint_path: Optional[str], policy_mode: str, suite_regime: str) -> str:
    root = Path(__file__).resolve().parents[3] / "artifacts" / "prior_generalization_audit"
    root.mkdir(parents=True, exist_ok=True)
    stem = "no_checkpoint" if checkpoint_path is None else Path(checkpoint_path).stem
    safe_mode = str(policy_mode).strip().lower()
    safe_regime = str(suite_regime).strip().lower()
    return str((root / f"{stem}_{safe_mode}_{safe_regime}.json").resolve())


def _print_summary(report: dict[str, Any], *, output_path: Optional[str]) -> None:
    train_metrics = report["train_suite"]["metrics"]
    heldout_metrics = report["heldout_suite"]["metrics"]
    print(
        "[prior-audit] "
        f"policy={report['policy']['resolved_mode']} device={report['policy']['device']} "
        f"regime={report['config']['suite_regime']} backend={report['config']['rollout_backend']} "
        f"n_samples={report['config']['n_samples']} single_eval_pos={report['config']['single_eval_pos']}"
    )
    print(
        "[prior-audit] "
        f"train full={train_metrics['full_return_mean']:.6f} suffix={train_metrics['suffix_return_mean']:.6f} "
        f"heldout full={heldout_metrics['full_return_mean']:.6f} suffix={heldout_metrics['suffix_return_mean']:.6f}"
    )
    reward_keys = [
        key
        for key in (
            "reward_env_return_mean",
            "reward_ctrl_return_mean",
            "reward_survival_return_mean",
            "reward_terminal_bonus_return_mean",
        )
        if key in train_metrics and key in heldout_metrics
    ]
    if reward_keys:
        details = " ".join(
            f"{key}=({train_metrics[key]:.6f}->{heldout_metrics[key]:.6f})"
            for key in reward_keys
        )
        print(f"[prior-audit] {details}")
    paired = report.get("paired_metrics", None)
    if isinstance(paired, dict):
        print(
            "[prior-audit] "
            f"same_env paired suffix delta mean={paired['heldout_minus_train_suffix_return_mean']:.6f} "
            f"std={paired['heldout_minus_train_suffix_return_std']:.6f} "
            f"base_envs={paired['base_env_batch_size']}"
        )
    reach = report["train_suite"].get("action_reachability", None)
    if isinstance(reach, dict):
        print(
            "[prior-audit] "
            f"train_action_probe suffix|delta|<1e-3 share={reach['suffix_return_delta_abs_lt_1e_3_share']:.6f} "
            f"obs|delta|>1e-2 share={reach['obs_max_abs_delta_gt_1e_2_share']:.6f} "
            f"state_only_share={reach['state_only_sensitive_share_obs_gt_1e_2_reward_lt_1e_3']:.6f}"
        )
    if output_path is not None:
        print(f"[prior-audit] saved_report={output_path}")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Standalone prior-environment generalization audit")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to a saved .mothernet/.cpkt checkpoint")
    parser.add_argument("--policy-mode", type=str, default="auto", choices=["auto", "model", "ppo", "zero"])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--num-features", type=int, default=None)
    parser.add_argument("--single-eval-pos", type=int, default=None)
    parser.add_argument("--suite-regime", type=str, default="cross_env", choices=["cross_env", "same_env"])
    parser.add_argument("--base-suite-batch-size", type=int, default=None)
    parser.add_argument("--base-suite-seed", type=int, default=12345)
    parser.add_argument("--train-rollouts-per-env", type=int, default=1)
    parser.add_argument("--heldout-rollouts-per-env", type=int, default=1)
    parser.add_argument("--train-rollout-seed", type=int, default=12345)
    parser.add_argument("--heldout-rollout-seed", type=int, default=67890)
    parser.add_argument("--train-suite-batch-size", type=int, default=16)
    parser.add_argument("--heldout-suite-batch-size", type=int, default=16)
    parser.add_argument("--train-suite-seed", type=int, default=12345)
    parser.add_argument("--heldout-suite-seed", type=int, default=67890)
    parser.add_argument("--base-suite-path", type=str, default=None)
    parser.add_argument("--train-suite-path", type=str, default=None)
    parser.add_argument("--heldout-suite-path", type=str, default=None)
    parser.add_argument("--suite-save-dir", type=str, default=None)
    parser.add_argument("--core-a", action="store_true", help="Apply the stripped Core-A audit regime")
    parser.add_argument("--reference-semantics", action="store_true")
    parser.add_argument("--rollout-backend", type=str, default="family_vectorized", choices=["family_vectorized", "serial"])
    parser.add_argument("--no-action-reachability-probe", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--print-json", action="store_true")
    args = parser.parse_args(argv)

    report = run_prior_generalization_audit(
        checkpoint_path=args.checkpoint,
        policy_mode=args.policy_mode,
        device=args.device,
        n_samples=args.n_samples,
        num_features=args.num_features,
        single_eval_pos=args.single_eval_pos,
        suite_regime=args.suite_regime,
        base_suite_batch_size=args.base_suite_batch_size,
        base_suite_seed=args.base_suite_seed,
        train_rollouts_per_env=args.train_rollouts_per_env,
        heldout_rollouts_per_env=args.heldout_rollouts_per_env,
        train_rollout_seed=args.train_rollout_seed,
        heldout_rollout_seed=args.heldout_rollout_seed,
        train_suite_batch_size=args.train_suite_batch_size,
        heldout_suite_batch_size=args.heldout_suite_batch_size,
        train_suite_seed=args.train_suite_seed,
        heldout_suite_seed=args.heldout_suite_seed,
        base_suite_path=args.base_suite_path,
        train_suite_path=args.train_suite_path,
        heldout_suite_path=args.heldout_suite_path,
        suite_save_dir=args.suite_save_dir,
        core_a=bool(args.core_a),
        reference_semantics_enabled=bool(args.reference_semantics),
        rollout_backend=args.rollout_backend,
        include_action_reachability_probe=not bool(args.no_action_reachability_probe),
    )

    output_path = str(Path(args.output).expanduser().resolve()) if args.output else _default_output_path(
        checkpoint_path=args.checkpoint,
        policy_mode=report["policy"]["resolved_mode"],
        suite_regime=report["config"]["suite_regime"],
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    _print_summary(report, output_path=output_path)
    if bool(args.print_json):
        print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
