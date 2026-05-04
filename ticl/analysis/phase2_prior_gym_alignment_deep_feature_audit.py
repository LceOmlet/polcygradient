import argparse
import copy
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ticl.analysis.critic_free_single_env_audit import (
    _build_audit_env_cfg,
)
from ticl.analysis.phase2_alpha_temporal_control_probe import (
    _apply_terminal_reset_actual_state,
    _compute_preterminal_state_next_batch,
    _make_generators_from_seeds,
)
from ticl.analysis.phase2_gym_horizon_state_identity_probe import (
    FixedRandomLinearGaussianPolicy,
    _capture_mujoco_state,
    _make_env,
    _restore_mujoco_state,
)
from ticl.analysis.phase2_suffix_state_identity_probe import (
    _broadcast_like_batch,
    _load_full_frozen_h,
    _resolve_probe_config,
    _to_bool,
    _to_float,
)
from ticl.config_utils import str2bool
from ticl.priors.environment_prior import EnvironmentPrior


DEFAULT_SCREENING_CSV = (
    "/home/chen/RLPFN/artifacts/"
    "phase2_prior_gym_control_feature_audit_gain07_all_openloop_0428/"
    "screening_rows.csv"
)
DEFAULT_OUTPUT_DIR = (
    "/home/chen/RLPFN/artifacts/"
    "phase2_prior_gym_alignment_deep_feature_audit_0428"
)


def _default_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, torch.Tensor):
        return _json_safe(value.detach().cpu().numpy())
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


def _stats(values: list[Any]) -> dict[str, Any]:
    arr = np.asarray([v for v in (_finite_float(v) for v in values) if v is not None], dtype=np.float64)
    if arr.size == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "q05": None,
            "q10": None,
            "q25": None,
            "q50": None,
            "q75": None,
            "q90": None,
            "q95": None,
            "max": None,
        }
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "q05": float(np.quantile(arr, 0.05)),
        "q10": float(np.quantile(arr, 0.10)),
        "q25": float(np.quantile(arr, 0.25)),
        "q50": float(np.quantile(arr, 0.50)),
        "q75": float(np.quantile(arr, 0.75)),
        "q90": float(np.quantile(arr, 0.90)),
        "q95": float(np.quantile(arr, 0.95)),
        "max": float(np.max(arr)),
    }


def _effective_rank_from_singular_values(values: np.ndarray) -> float | None:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals) & (vals > 0.0)]
    if vals.size == 0:
        return None
    total = float(np.sum(vals))
    if total <= 1e-12:
        return None
    probs = vals / total
    entropy = float(-np.sum(probs * np.log(np.maximum(probs, 1e-300))))
    return float(np.exp(entropy))


def _effective_rank_from_matrix(samples: np.ndarray) -> dict[str, Any]:
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 3 or x.shape[1] < 1:
        return {
            "effective_rank": None,
            "effective_rank_fraction": None,
            "top_eigen_share": None,
        }
    x = x - np.mean(x, axis=0, keepdims=True)
    try:
        singular_values = np.linalg.svd(x, compute_uv=False)
    except np.linalg.LinAlgError:
        return {
            "effective_rank": None,
            "effective_rank_fraction": None,
            "top_eigen_share": None,
        }
    power = singular_values * singular_values
    total = float(np.sum(power))
    effective_rank = _effective_rank_from_singular_values(power)
    return {
        "effective_rank": effective_rank,
        "effective_rank_fraction": (
            float(effective_rank / max(int(x.shape[1]), 1)) if effective_rank is not None else None
        ),
        "top_eigen_share": float(np.max(power) / total) if total > 1e-12 else None,
    }


def _dim_distribution(values: np.ndarray, *, dead_abs_threshold: float = 1e-4) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = arr[np.isfinite(arr)]
    base = _stats(finite.tolist())
    if finite.size == 0:
        base.update(
            {
                "dead_dim_fraction_abs": None,
                "dominant_dim_share": None,
                "effective_dim": None,
                "effective_dim_fraction": None,
            }
        )
        return base
    positive = np.maximum(finite, 0.0)
    total = float(np.sum(positive))
    eff = _effective_rank_from_singular_values(positive)
    base.update(
        {
            "dead_dim_fraction_abs": float(np.mean(np.abs(finite) <= float(dead_abs_threshold))),
            "dominant_dim_share": float(np.max(positive) / total) if total > 1e-12 else None,
            "effective_dim": eff,
            "effective_dim_fraction": float(eff / max(int(finite.size), 1)) if eff is not None else None,
        }
    )
    return base


def _read_screening_rows(path: str | Path, *, min_gain: float | None, count: int, seed: int) -> list[dict[str, Any]]:
    rows = []
    with Path(path).expanduser().open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if min_gain is not None:
                gain = _finite_float(row.get("topology_reward_state_input_gain_fraction"))
                if gain is None or gain < float(min_gain):
                    continue
            rows.append(row)
    if int(count) <= 0 or int(count) >= len(rows):
        return rows
    rng = np.random.default_rng(int(seed))
    indices = sorted(rng.choice(len(rows), size=int(count), replace=False).tolist())
    return [rows[int(i)] for i in indices]


def _parse_scalar(text: str) -> Any:
    token = str(text).strip()
    lower = token.lower()
    if lower in {"true", "yes", "y"}:
        return True
    if lower in {"false", "no", "n"}:
        return False
    if lower in {"none", "null"}:
        return None
    try:
        if any(ch in token for ch in (".", "e", "E")):
            return float(token)
        return int(token)
    except ValueError:
        return token


def _parse_case(text: str) -> dict[str, Any]:
    raw = str(text).strip()
    if raw in {"", "baseline", "none"}:
        return {"name": "baseline", "overrides": {}}
    name = None
    body = raw
    if ":" in raw:
        name, body = raw.split(":", 1)
        name = name.strip()
    overrides = {}
    for piece in body.split(","):
        item = piece.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"case item must be key=value, got {item!r} in {text!r}")
        key, value = item.split("=", 1)
        overrides[key.strip()] = _parse_scalar(value)
    if name is None:
        name = "_".join(f"{key}{str(value).replace('.', 'p')}" for key, value in sorted(overrides.items()))
    return {"name": str(name), "overrides": overrides}


def _parse_cases(raw: str) -> list[dict[str, Any]]:
    cases = [_parse_case(piece) for piece in str(raw).split(";") if piece.strip()]
    if not cases:
        cases = [{"name": "baseline", "overrides": {}}]
    names = [case["name"] for case in cases]
    if len(set(names)) != len(names):
        raise ValueError(f"case names must be unique, got {names}")
    return cases


def _parse_gym_env_ids(raw: str, fallback: str) -> list[str]:
    env_ids = [piece.strip() for piece in str(raw or "").split(",") if piece.strip()]
    if not env_ids:
        env_ids = [str(fallback)]
    deduped: list[str] = []
    seen = set()
    for env_id in env_ids:
        if env_id in seen:
            continue
        seen.add(env_id)
        deduped.append(env_id)
    return deduped


def _compose_prior_reward(prior: EnvironmentPrior, env: dict[str, Any], action: torch.Tensor, raw_reward: torch.Tensor) -> torch.Tensor:
    batch_size = int(action.shape[0])
    aux_reward = prior._exact_scm_aux_reward_terms(
        action,
        ctrl_weight=env.get("ctrl_reward_weight", 0.0),
        ctrl_enabled=env.get("ctrl_reward_enabled", False),
        survival_weight=env.get("survival_reward_weight", 0.0),
        survival_enabled=env.get("survival_reward_enabled", False),
    )
    return prior._compose_exact_scm_reward(
        raw_reward.reshape(batch_size),
        aux_reward=aux_reward.reshape(batch_size),
        reward_clip=env.get("reward_clip", float("inf")),
        mode=env.get("reinforce_reward_transform", "none"),
        rms_eps=env.get("reinforce_reward_rms_eps", 1e-6),
        tanh_c=env.get("reinforce_reward_tanh_c", 1.0),
        tanh_bound=env.get("reinforce_reward_tanh_bound", 2.0),
    ).reshape(batch_size)


def _prior_per_action_sensitivity(
    *,
    prior: EnvironmentPrior,
    env: dict[str, Any],
    carry_state_t: torch.Tensor,
    state_t: torch.Tensor,
    action_next: torch.Tensor,
    noise_t: torch.Tensor,
    state_noise_t: torch.Tensor | None,
    device: torch.device,
    dtype: torch.dtype,
    step_idx: int,
    rollout_seed_start: int,
    action_delta: float,
) -> tuple[np.ndarray, np.ndarray]:
    batch_size = int(action_next.shape[0])
    action_dim = int(action_next.shape[1])
    eye = torch.eye(action_dim, device=device, dtype=dtype)
    perturb = eye.unsqueeze(0).expand(batch_size, action_dim, action_dim).reshape(batch_size * action_dim, action_dim)
    action_base = action_next[:, None, :].expand(batch_size, action_dim, action_dim).reshape(batch_size * action_dim, action_dim)
    action_transform_mode = env.get("reinforce_action_transform", "clip")
    action_rms_eps = float(env.get("reinforce_action_rms_eps", 1e-6))
    action_clip_bound = float(env.get("reinforce_action_clip_bound", 5.0))
    action_plus = prior._transform_reinforce_action(
        action_base + float(action_delta) * perturb,
        mode=action_transform_mode,
        rms_eps=action_rms_eps,
        clip_bound=action_clip_bound,
    )
    action_minus = prior._transform_reinforce_action(
        action_base - float(action_delta) * perturb,
        mode=action_transform_mode,
        rms_eps=action_rms_eps,
        clip_bound=action_clip_bound,
    )
    carry_rep = carry_state_t[:, None, :].expand(batch_size, action_dim, carry_state_t.shape[1]).reshape(
        batch_size * action_dim,
        carry_state_t.shape[1],
    )
    state_rep = state_t[:, None, :].expand(batch_size, action_dim, state_t.shape[1]).reshape(
        batch_size * action_dim,
        state_t.shape[1],
    )
    noise_rep = noise_t[:, None, :].expand(batch_size, action_dim, noise_t.shape[1]).reshape(
        batch_size * action_dim,
        noise_t.shape[1],
    )
    if state_noise_t is None:
        state_noise_rep = None
    else:
        state_noise_rep = state_noise_t[:, None, :].expand(batch_size, action_dim, state_noise_t.shape[1]).reshape(
            batch_size * action_dim,
            state_noise_t.shape[1],
        )
    sens_transition_seeds = [
        int(80_000_000_000) + int(rollout_seed_start) + int(step_idx) * int(batch_size * action_dim) + int(i)
        for i in range(batch_size * action_dim)
    ]
    sens_gens_plus = _make_generators_from_seeds(prior, sens_transition_seeds, device)
    sens_gens_minus = _make_generators_from_seeds(prior, sens_transition_seeds, device)
    state_plus, reward_plus_raw, _, _ = _compute_preterminal_state_next_batch(
        prior=prior,
        env=env,
        carry_state_t=carry_rep,
        state_t=state_rep,
        action_next=action_plus,
        noise_t=noise_rep,
        state_noise_t=state_noise_rep,
        rollout_generators=sens_gens_plus,
        device=device,
    )
    state_minus, reward_minus_raw, _, _ = _compute_preterminal_state_next_batch(
        prior=prior,
        env=env,
        carry_state_t=carry_rep,
        state_t=state_rep,
        action_next=action_minus,
        noise_t=noise_rep,
        state_noise_t=state_noise_rep,
        rollout_generators=sens_gens_minus,
        device=device,
    )
    state_sens = torch.linalg.vector_norm(state_plus - state_minus, ord=2, dim=1) / max(2.0 * float(action_delta), 1e-12)
    reward_plus = _compose_prior_reward(prior, env, action_plus, reward_plus_raw)
    reward_minus = _compose_prior_reward(prior, env, action_minus, reward_minus_raw)
    reward_sens = torch.abs(reward_plus - reward_minus) / max(2.0 * float(action_delta), 1e-12)
    state_sens_dim = state_sens.reshape(batch_size, action_dim).mean(dim=0).detach().cpu().numpy()
    reward_sens_dim = reward_sens.reshape(batch_size, action_dim).mean(dim=0).detach().cpu().numpy()
    return state_sens_dim, reward_sens_dim


def measure_prior_deep_features(
    *,
    row: dict[str, Any],
    case_name: str = "baseline",
    frozen_h_overrides: dict[str, Any] | None = None,
    device: str,
    train_env_seed: int,
    rollout_count: int,
    n_steps: int,
    rollout_seed_start: int,
    action_delta: float,
    per_action_stride: int,
) -> dict[str, Any]:
    device_obj = torch.device(str(device))
    config = _resolve_probe_config(
        checkpoint_path=None,
        from_scratch=True,
        from_scratch_model_type="rlpfn",
        device_obj=device_obj,
        build_seed=4040,
        allow_config_only=True,
    )
    env_cfg = _build_audit_env_cfg(config["prior"]["environment"])
    frozen_h, loaded_frozen_h_path = _load_full_frozen_h(str(row["full_frozen_h_json"]))
    if frozen_h_overrides is not None:
        frozen_h.update(copy.deepcopy(frozen_h_overrides))
    prior = EnvironmentPrior(copy.deepcopy(env_cfg))
    env = prior._sample_environment(copy.deepcopy(frozen_h), device=device_obj, rng_seed=int(train_env_seed))
    batch_size = int(rollout_count)
    dtype = torch.float32
    state_dim = int(env["state_dim"])
    obs_dim = int(env["obs_dim"])
    action_dim = int(env["action_dim"])
    noise_dim = int(env["noise_dim"])

    rollout_seeds = [int(rollout_seed_start) + int(i) for i in range(batch_size)]
    rollout_generators = _make_generators_from_seeds(prior, rollout_seeds, device_obj)
    fixed_initial_state_batch = prior._resolve_env_initial_state_batch(
        env,
        batch_size=batch_size,
        state_dim=state_dim,
        device=device_obj,
        dtype=dtype,
    )
    init_action_std = _broadcast_like_batch(env["init_action_std"], batch_size=batch_size, device=device_obj, dtype=dtype)
    carry_state_t, state_t = prior._init_exact_scm_carry_and_visible_state(fixed_initial_state_batch)
    del init_action_std

    reference_semantics_enabled = bool(prior._env_uses_reference_semantics(env))
    action_transform_mode = env.get("reinforce_action_transform", "clip")
    action_rms_eps = float(env.get("reinforce_action_rms_eps", 1e-6))
    action_clip_bound = float(env.get("reinforce_action_clip_bound", 5.0))
    action_noise_train_std = _to_float(env.get("action_noise_train_std", 0.0))
    action_noise_eval_std = _to_float(env.get("action_noise_eval_std", 0.0))
    state_noise_std = _to_float(env.get("state_noise_std", 0.0))
    reward_dropout_enabled = _to_bool(env.get("reward_dropout_enabled", False))
    reward_dropout_ratio = _to_float(env.get("reward_dropout_ratio", 0.0))
    reward_impute_zero = _to_bool(env.get("reward_dropout_impute_zero", True))
    terminal_signal_history = (
        torch.zeros((int(n_steps), batch_size), device=device_obj, dtype=dtype)
        if bool(env.get("terminal_reset_enabled", False))
        else None
    )

    obs_chunks: list[np.ndarray] = []
    reward_values: list[np.ndarray] = []
    raw_reward_values: list[np.ndarray] = []
    step_drift_values: list[np.ndarray] = []
    terminal_events = 0
    first_terminal_step = torch.full((batch_size,), -1, device=device_obj, dtype=torch.int64)
    discounted_effective_returns = torch.zeros((batch_size,), device=device_obj, dtype=dtype)
    discount_t = torch.ones((batch_size,), device=device_obj, dtype=dtype)
    per_dim_state_sens: list[np.ndarray] = []
    per_dim_reward_sens: list[np.ndarray] = []

    for step_idx in range(int(n_steps)):
        obs_chunks.append(state_t[:, :obs_dim].detach().cpu().numpy())
        action_next = prior._stack_randn_with_generators(
            rollout_generators, (batch_size, action_dim), device_obj, dtype
        )
        if (not reference_semantics_enabled) and action_noise_train_std > 0.0 and int(step_idx) < 64:
            eps_t = prior._stack_randn_with_generators(rollout_generators, (batch_size, action_dim), device_obj, dtype)
            action_next = prior._transform_reinforce_action(
                action_next + eps_t * float(action_noise_train_std),
                mode=action_transform_mode,
                rms_eps=action_rms_eps,
                clip_bound=action_clip_bound,
            )
        elif (not reference_semantics_enabled) and action_noise_eval_std > 0.0:
            eps_t = prior._stack_randn_with_generators(rollout_generators, (batch_size, action_dim), device_obj, dtype)
            action_next = prior._transform_reinforce_action(
                action_next + eps_t * float(action_noise_eval_std),
                mode=action_transform_mode,
                rms_eps=action_rms_eps,
                clip_bound=action_clip_bound,
            )
        if noise_dim > 0:
            noise_t = prior._stack_randn_with_generators(rollout_generators, (batch_size, noise_dim), device_obj, dtype)
        else:
            noise_t = torch.zeros((batch_size, 0), device=device_obj, dtype=dtype)
        state_noise_t = None
        if state_noise_std > 0.0:
            state_noise_t = prior._stack_randn_with_generators(
                rollout_generators, (batch_size, state_dim), device_obj, dtype
            )

        if int(per_action_stride) > 0 and int(step_idx) % int(per_action_stride) == 0:
            state_sens_dim, reward_sens_dim = _prior_per_action_sensitivity(
                prior=prior,
                env=env,
                carry_state_t=carry_state_t,
                state_t=state_t,
                action_next=action_next,
                noise_t=noise_t,
                state_noise_t=state_noise_t,
                device=device_obj,
                dtype=dtype,
                step_idx=int(step_idx),
                rollout_seed_start=int(rollout_seed_start),
                action_delta=float(action_delta),
            )
            per_dim_state_sens.append(state_sens_dim)
            per_dim_reward_sens.append(reward_sens_dim)

        state_next_preterminal, reward_next_raw, terminal_signal_next, terminal_bonus_base_next = _compute_preterminal_state_next_batch(
            prior=prior,
            env=env,
            carry_state_t=carry_state_t,
            state_t=state_t,
            action_next=action_next,
            noise_t=noise_t,
            state_noise_t=state_noise_t,
            rollout_generators=rollout_generators,
            device=device_obj,
        )
        reward_next = _compose_prior_reward(prior, env, action_next, reward_next_raw)
        if reward_dropout_enabled and reward_dropout_ratio > 0.0:
            drop_mask = prior._stack_rand_with_generators(rollout_generators, (batch_size,), device_obj, dtype) < float(
                reward_dropout_ratio
            )
            if bool(reward_impute_zero):
                reward_next = torch.where(drop_mask, torch.zeros_like(reward_next), reward_next)
        state_next_actual, terminal_next = _apply_terminal_reset_actual_state(
            prior=prior,
            env=env,
            state_next_preterminal=state_next_preterminal,
            reward_next=reward_next,
            terminal_signal_next=terminal_signal_next,
            terminal_bonus_base_next=terminal_bonus_base_next,
            terminal_signal_history=terminal_signal_history,
            history_index=int(step_idx),
            rollout_generators=rollout_generators,
            fixed_initial_state_batch=fixed_initial_state_batch,
            n_steps=int(n_steps),
        )
        reward_values.append(reward_next.detach().cpu().numpy())
        raw_reward_values.append(reward_next_raw.reshape(batch_size).detach().cpu().numpy())
        discounted_effective_returns += discount_t * reward_next.reshape(batch_size)
        discount_t = discount_t * 0.99
        terminal_bool = terminal_next.bool()
        terminal_events += int(terminal_bool.sum().item())
        first_mask = terminal_bool & (first_terminal_step < 0)
        first_terminal_step = torch.where(first_mask, torch.full_like(first_terminal_step, int(step_idx) + 1), first_terminal_step)
        step_drift = torch.linalg.vector_norm(state_next_actual[:, :obs_dim] - state_t[:, :obs_dim], ord=2, dim=1)
        step_drift_values.append(step_drift.detach().cpu().numpy())
        carry_state_next = prior._mix_exact_scm_carry_state(carry_state_t, state_next_actual, env["alpha"])
        if bool(env.get("terminal_reset_enabled", False)):
            carry_state_next = prior._sync_exact_scm_carry_state_with_visible(
                carry_state_next,
                state_next_actual,
                terminal_next,
            )
        carry_state_t = carry_state_next
        state_t = state_next_actual

    obs_samples = np.concatenate(obs_chunks, axis=0) if obs_chunks else np.zeros((0, obs_dim), dtype=np.float64)
    reward_arr = np.concatenate(reward_values, axis=0) if reward_values else np.asarray([], dtype=np.float64)
    raw_reward_arr = np.concatenate(raw_reward_values, axis=0) if raw_reward_values else np.asarray([], dtype=np.float64)
    step_drift_arr = np.concatenate(step_drift_values, axis=0) if step_drift_values else np.asarray([], dtype=np.float64)
    obs_per_dim_std = np.std(obs_samples, axis=0) if obs_samples.size else np.asarray([], dtype=np.float64)
    state_sens_per_dim = np.mean(np.stack(per_dim_state_sens, axis=0), axis=0) if per_dim_state_sens else np.asarray([])
    reward_sens_per_dim = np.mean(np.stack(per_dim_reward_sens, axis=0), axis=0) if per_dim_reward_sens else np.asarray([])
    first_terminal_np = first_terminal_step.detach().cpu().numpy()
    first_terminal_seen = first_terminal_np[first_terminal_np > 0]

    return {
        "source": "prior",
        "case": str(case_name),
        "overrides": copy.deepcopy(frozen_h_overrides or {}),
        "frozen_h_seed": row.get("frozen_h_seed"),
        "full_frozen_h_json": str(loaded_frozen_h_path),
        "topology_reward_state_input_gain_fraction": _finite_float(row.get("topology_reward_state_input_gain_fraction")),
        "topology_reward_action_input_gain_fraction": _finite_float(row.get("topology_reward_action_input_gain_fraction")),
        "topology_reward_noise_input_gain_fraction": _finite_float(row.get("topology_reward_noise_input_gain_fraction")),
        "horizon_2_bias_corrected_identity_score": _finite_float(row.get("horizon_2_bias_corrected_identity_score")),
        "horizon_5_bias_corrected_identity_score": _finite_float(row.get("horizon_5_bias_corrected_identity_score")),
        "state_dim": int(state_dim),
        "obs_dim": int(obs_dim),
        "action_dim": int(action_dim),
        "noise_dim": int(noise_dim),
        "n_steps": int(n_steps),
        "rollout_count": int(rollout_count),
        "obs_value_std_all": float(np.std(obs_samples)) if obs_samples.size else None,
        "obs_per_dim_std": _dim_distribution(obs_per_dim_std),
        "obs_cov": _effective_rank_from_matrix(obs_samples),
        "reward_effective": _stats(reward_arr.tolist()),
        "reward_raw": _stats(raw_reward_arr.tolist()),
        "discounted_effective_return": _stats(discounted_effective_returns.detach().cpu().numpy().tolist()),
        "step_obs_drift_l2": _stats(step_drift_arr.tolist()),
        "per_action_state_sensitivity": _dim_distribution(state_sens_per_dim),
        "per_action_reward_sensitivity": _dim_distribution(reward_sens_per_dim),
        "per_action_state_sensitivity_values": state_sens_per_dim.tolist(),
        "per_action_reward_sensitivity_values": reward_sens_per_dim.tolist(),
        "terminal_event_rate": float(terminal_events / max(int(n_steps) * int(batch_size), 1)),
        "terminal_first_event_fraction": float(first_terminal_seen.size / max(batch_size, 1)),
        "terminal_first_event_step": _stats(first_terminal_seen.tolist()),
        "sampled_env_snapshot": {
            "reference_semantics_enabled": bool(env.get("reference_semantics_enabled", False)),
            "terminal_reset_enabled": bool(env.get("terminal_reset_enabled", False)),
            "terminal_reset_count_target": float(env.get("terminal_reset_count_target", 0.0)),
            "noise_std": float(frozen_h.get("noise_std", 0.0)),
            "init_std": float(frozen_h.get("init_std", 0.0)),
            "init_state_std": float(frozen_h.get("init_state_std", 0.0)),
            "state_input_scale": float(frozen_h.get("state_input_scale", 1.0)),
            "state_output_scale": float(frozen_h.get("state_output_scale", 1.0)),
            "state_full_rms_enabled": bool(frozen_h.get("state_full_rms_enabled", False)),
            "state_full_rms_target": float(frozen_h.get("state_full_rms_target", 1.0)),
            "effective_reward_scale": float(env.get("reward_scale", 1.0)),
            "requested_reward_scale": float(frozen_h.get("reward_scale", 1.0)),
            "reward_clip": float(env.get("reward_clip", float("inf"))),
            "state_clip": float(env.get("state_clip", float("inf"))),
            "reward_dropout_enabled": bool(env.get("reward_dropout_enabled", False)),
            "reinforce_reward_transform": env.get("reinforce_reward_transform"),
            "prior_mlp_activations": env.get("prior_mlp_activations"),
            "alpha": float(env.get("alpha", 1.0)),
        },
    }


def measure_ant_deep_features(
    *,
    env_id: str,
    reference_state_count: int,
    anchor_step: int,
    rollout_horizon: int,
    policy_seed: int,
    reference_seed_start: int,
    action_delta: float,
) -> dict[str, Any]:
    env = _make_env(str(env_id))
    try:
        obs_dim = int(np.prod(env.observation_space.shape))
        action_dim = int(np.prod(env.action_space.shape))
        policy = FixedRandomLinearGaussianPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            seed=int(policy_seed),
            sigma=0.25,
            policy_type="unit_gaussian",
        )
        action_low = np.asarray(env.action_space.low, dtype=np.float64)
        action_high = np.asarray(env.action_space.high, dtype=np.float64)
        obs_samples = []
        rewards = []
        step_drifts = []
        first_done_steps = []
        done_count = 0
        per_dim_state_sens = []
        per_dim_reward_sens = []
        discounted_returns = []
        for idx in range(int(reference_state_count)):
            seed = int(reference_seed_start) + int(idx)
            obs, _ = env.reset(seed=seed)
            obs = np.asarray(obs, dtype=np.float64)
            rng = np.random.default_rng(seed)
            for _ in range(int(anchor_step)):
                action = policy.action(obs, rng, action_low, action_high)
                obs, reward, terminated, truncated, _ = env.step(action)
                obs = np.asarray(obs, dtype=np.float64)
                if bool(terminated) or bool(truncated):
                    obs, _ = env.reset(seed=seed + 1)
                    obs = np.asarray(obs, dtype=np.float64)
            anchor_state = _capture_mujoco_state(env)
            anchor_obs = np.asarray(obs, dtype=np.float64)
            base_action = policy.action(anchor_obs, rng, action_low, action_high).astype(np.float64)
            state_sens_dims = []
            reward_sens_dims = []
            for action_idx in range(action_dim):
                perturb = np.zeros((action_dim,), dtype=np.float64)
                perturb[int(action_idx)] = float(action_delta)
                action_plus = np.clip(base_action + perturb, action_low, action_high)
                action_minus = np.clip(base_action - perturb, action_low, action_high)
                _restore_mujoco_state(env, anchor_state)
                obs_plus, reward_plus, _, _, _ = env.step(action_plus.astype(np.float32))
                _restore_mujoco_state(env, anchor_state)
                obs_minus, reward_minus, _, _, _ = env.step(action_minus.astype(np.float32))
                state_sens_dims.append(
                    float(np.linalg.norm(np.asarray(obs_plus, dtype=np.float64) - np.asarray(obs_minus, dtype=np.float64)))
                    / max(2.0 * float(action_delta), 1e-12)
                )
                reward_sens_dims.append(float(abs(float(reward_plus) - float(reward_minus))) / max(2.0 * float(action_delta), 1e-12))
            per_dim_state_sens.append(np.asarray(state_sens_dims, dtype=np.float64))
            per_dim_reward_sens.append(np.asarray(reward_sens_dims, dtype=np.float64))

            obs = _restore_mujoco_state(env, anchor_state)
            obs = np.asarray(obs, dtype=np.float64)
            total = 0.0
            discount = 1.0
            first_done = None
            prev_obs = obs
            for step_idx in range(int(rollout_horizon)):
                obs_samples.append(prev_obs.tolist())
                action = policy.action(prev_obs, rng, action_low, action_high)
                obs_next, reward, terminated, truncated, _ = env.step(action.astype(np.float32))
                obs_next = np.asarray(obs_next, dtype=np.float64)
                rewards.append(float(reward))
                total += float(discount) * float(reward)
                discount *= 0.99
                step_drifts.append(float(np.linalg.norm(obs_next - prev_obs)))
                if bool(terminated) or bool(truncated):
                    done_count += 1
                    first_done = int(step_idx) + 1
                    break
                prev_obs = obs_next
            if first_done is not None:
                first_done_steps.append(first_done)
            discounted_returns.append(float(total))
        obs_arr = np.asarray(obs_samples, dtype=np.float64)
        obs_per_dim_std = np.std(obs_arr, axis=0) if obs_arr.size else np.asarray([], dtype=np.float64)
        state_sens_per_dim = np.mean(np.stack(per_dim_state_sens, axis=0), axis=0) if per_dim_state_sens else np.asarray([])
        reward_sens_per_dim = np.mean(np.stack(per_dim_reward_sens, axis=0), axis=0) if per_dim_reward_sens else np.asarray([])
        return {
            "source": "gym",
            "env_id": str(env_id),
            "obs_dim": int(obs_dim),
            "action_dim": int(action_dim),
            "reference_state_count": int(reference_state_count),
            "rollout_horizon": int(rollout_horizon),
            "obs_value_std_all": float(np.std(obs_arr)) if obs_arr.size else None,
            "obs_per_dim_std": _dim_distribution(obs_per_dim_std),
            "obs_cov": _effective_rank_from_matrix(obs_arr),
            "reward_effective": _stats(rewards),
            "discounted_effective_return": _stats(discounted_returns),
            "step_obs_drift_l2": _stats(step_drifts),
            "per_action_state_sensitivity": _dim_distribution(state_sens_per_dim),
            "per_action_reward_sensitivity": _dim_distribution(reward_sens_per_dim),
            "per_action_state_sensitivity_values": state_sens_per_dim.tolist(),
            "per_action_reward_sensitivity_values": reward_sens_per_dim.tolist(),
            "terminal_event_rate": float(done_count / max(int(reference_state_count) * int(rollout_horizon), 1)),
            "terminal_first_event_fraction": float(len(first_done_steps) / max(int(reference_state_count), 1)),
            "terminal_first_event_step": _stats(first_done_steps),
        }
    finally:
        env.close()


def _summarize_prior_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [
        "obs_value_std_all",
        "terminal_event_rate",
        "terminal_first_event_fraction",
    ]
    nested = [
        ("obs_per_dim_std", "q50"),
        ("obs_per_dim_std", "effective_dim_fraction"),
        ("obs_cov", "effective_rank_fraction"),
        ("obs_cov", "top_eigen_share"),
        ("reward_effective", "std"),
        ("discounted_effective_return", "std"),
        ("step_obs_drift_l2", "mean"),
        ("per_action_state_sensitivity", "q50"),
        ("per_action_state_sensitivity", "dead_dim_fraction_abs"),
        ("per_action_state_sensitivity", "dominant_dim_share"),
        ("per_action_state_sensitivity", "effective_dim_fraction"),
        ("per_action_reward_sensitivity", "q50"),
        ("per_action_reward_sensitivity", "dead_dim_fraction_abs"),
        ("per_action_reward_sensitivity", "dominant_dim_share"),
        ("per_action_reward_sensitivity", "effective_dim_fraction"),
    ]
    out = {key: _stats([row.get(key) for row in rows]) for key in keys}
    for outer, inner in nested:
        out[f"{outer}.{inner}"] = _stats([(row.get(outer) or {}).get(inner) for row in rows])
    return out


def _summarize_prior_rows_by_case(rows: list[dict[str, Any]]) -> dict[str, Any]:
    case_names = sorted({str(row.get("case", "baseline")) for row in rows})
    by_case = {}
    for case_name in case_names:
        case_rows = [row for row in rows if str(row.get("case", "baseline")) == case_name]
        by_case[case_name] = {
            "row_count": int(len(case_rows)),
            "summary": _summarize_prior_rows(case_rows),
        }
    baseline = by_case.get("baseline")
    if baseline is not None:
        baseline_summary = baseline["summary"]
        for case_name, payload in by_case.items():
            if case_name == "baseline":
                continue
            deltas = {}
            for key, stat in payload["summary"].items():
                case_mean = _finite_float((stat or {}).get("mean"))
                base_mean = _finite_float((baseline_summary.get(key) or {}).get("mean"))
                deltas[f"{key}_mean_delta_vs_baseline"] = (
                    None if case_mean is None or base_mean is None else float(case_mean - base_mean)
                )
            payload["mean_deltas_vs_baseline"] = deltas
    return by_case


def _ratio(prior_stat: dict[str, Any], ant_value: Any) -> dict[str, Any]:
    ant = _finite_float(ant_value)
    if ant is None or abs(ant) <= 1e-12:
        return {"ant": ant, "prior_q50_over_ant": None, "prior_mean_over_ant": None}
    return {
        "ant": ant,
        "prior_q50_over_ant": (
            float(prior_stat.get("q50")) / ant if _finite_float(prior_stat.get("q50")) is not None else None
        ),
        "prior_mean_over_ant": (
            float(prior_stat.get("mean")) / ant if _finite_float(prior_stat.get("mean")) is not None else None
        ),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Deep prior-vs-Gym feature audit for axes not covered by scalar sensitivity means: "
            "per-action-dim sensitivity, per-obs-dim scale/rank, and terminal dynamics."
        )
    )
    parser.add_argument("--screening-csv", type=str, default=DEFAULT_SCREENING_CSV)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-gain", type=float, default=0.7)
    parser.add_argument("--cases", type=str, default="baseline")
    parser.add_argument("--prior-count", type=int, default=16)
    parser.add_argument("--prior-sample-seed", type=int, default=20260428)
    parser.add_argument("--device", type=str, default=_default_device())
    parser.add_argument("--train-env-seed", type=int, default=2020)
    parser.add_argument("--prior-rollout-count", type=int, default=32)
    parser.add_argument("--prior-n-steps", type=int, default=24)
    parser.add_argument("--prior-per-action-stride", type=int, default=6)
    parser.add_argument("--rollout-seed-start", type=int, default=9100000)
    parser.add_argument("--paired-case-rollout-seeds", type=str2bool, default=True)
    parser.add_argument("--action-delta", type=float, default=0.1)
    parser.add_argument("--measure-ant", type=str2bool, default=True)
    parser.add_argument("--ant-env-id", type=str, default="Ant-v5_clean_obs")
    parser.add_argument(
        "--gym-env-ids",
        type=str,
        default="",
        help=(
            "Comma-separated Gym/MuJoCo reference env ids. Empty keeps backward-compatible "
            "--ant-env-id behavior."
        ),
    )
    parser.add_argument("--ant-reference-state-count", type=int, default=64)
    parser.add_argument("--ant-anchor-step", type=int, default=150)
    parser.add_argument("--ant-rollout-horizon", type=int, default=256)
    parser.add_argument("--ant-policy-seed", type=int, default=20260422)
    parser.add_argument("--ant-reference-seed-start", type=int, default=5000)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_rows = _read_screening_rows(
        args.screening_csv,
        min_gain=args.min_gain,
        count=int(args.prior_count),
        seed=int(args.prior_sample_seed),
    )
    cases = _parse_cases(str(args.cases))
    rows_path = output_dir / "rows.jsonl"
    prior_reports: list[dict[str, Any]] = []
    with rows_path.open("w", encoding="utf-8") as handle:
        for idx, row in enumerate(selected_rows, start=1):
            for case_idx, case in enumerate(cases, start=1):
                print(
                    f"[deep-feature-audit] prior {idx}/{len(selected_rows)} case {case_idx}/{len(cases)} "
                    f"seed={row.get('frozen_h_seed')} gain={row.get('topology_reward_state_input_gain_fraction')} "
                    f"case={case['name']}",
                    flush=True,
                )
                case_rollout_seed_start = int(args.rollout_seed_start) + idx * 100000
                if not bool(args.paired_case_rollout_seeds):
                    case_rollout_seed_start += case_idx * 10000
                report = measure_prior_deep_features(
                    row=row,
                    case_name=str(case["name"]),
                    frozen_h_overrides=dict(case["overrides"]),
                    device=str(args.device),
                    train_env_seed=int(args.train_env_seed),
                    rollout_count=int(args.prior_rollout_count),
                    n_steps=int(args.prior_n_steps),
                    rollout_seed_start=int(case_rollout_seed_start),
                    action_delta=float(args.action_delta),
                    per_action_stride=int(args.prior_per_action_stride),
                )
                prior_reports.append(report)
                handle.write(json.dumps(_json_safe(report), sort_keys=True) + "\n")
                handle.flush()
    gym_env_ids = _parse_gym_env_ids(str(args.gym_env_ids), str(args.ant_env_id))
    gym_references = {}
    ant_report = None
    if bool(args.measure_ant):
        for gym_idx, gym_env_id in enumerate(gym_env_ids, start=1):
            print(
                f"[deep-feature-audit] measuring Gym reference {gym_idx}/{len(gym_env_ids)} env={gym_env_id}",
                flush=True,
            )
            report = measure_ant_deep_features(
                env_id=str(gym_env_id),
                reference_state_count=int(args.ant_reference_state_count),
                anchor_step=int(args.ant_anchor_step),
                rollout_horizon=int(args.ant_rollout_horizon),
                policy_seed=int(args.ant_policy_seed),
                reference_seed_start=int(args.ant_reference_seed_start) + 100000 * (gym_idx - 1),
                action_delta=float(args.action_delta),
            )
            gym_references[str(gym_env_id)] = report
            if ant_report is None:
                ant_report = report
        (output_dir / "gym_references.json").write_text(
            json.dumps(_json_safe(gym_references), sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        if ant_report is not None:
            (output_dir / "ant_reference.json").write_text(
                json.dumps(_json_safe(ant_report), sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
    prior_summary_by_case = _summarize_prior_rows_by_case(prior_reports)
    prior_summary = (
        prior_summary_by_case["baseline"]["summary"]
        if "baseline" in prior_summary_by_case
        else _summarize_prior_rows(prior_reports)
    )
    def _build_gap_table(reference_report: dict[str, Any]) -> dict[str, Any]:
        ant = reference_report or {}
        comparisons = {
            "obs_value_std_all": (prior_summary["obs_value_std_all"], ant.get("obs_value_std_all")),
            "obs_per_dim_std.q50": (
                prior_summary["obs_per_dim_std.q50"],
                (ant.get("obs_per_dim_std") or {}).get("q50"),
            ),
            "obs_cov.effective_rank_fraction": (
                prior_summary["obs_cov.effective_rank_fraction"],
                (ant.get("obs_cov") or {}).get("effective_rank_fraction"),
            ),
            "obs_cov.top_eigen_share": (
                prior_summary["obs_cov.top_eigen_share"],
                (ant.get("obs_cov") or {}).get("top_eigen_share"),
            ),
            "reward_effective.std": (
                prior_summary["reward_effective.std"],
                (ant.get("reward_effective") or {}).get("std"),
            ),
            "discounted_effective_return.std": (
                prior_summary["discounted_effective_return.std"],
                (ant.get("discounted_effective_return") or {}).get("std"),
            ),
            "per_action_state_sensitivity.q50": (
                prior_summary["per_action_state_sensitivity.q50"],
                (ant.get("per_action_state_sensitivity") or {}).get("q50"),
            ),
            "per_action_state_sensitivity.effective_dim_fraction": (
                prior_summary["per_action_state_sensitivity.effective_dim_fraction"],
                (ant.get("per_action_state_sensitivity") or {}).get("effective_dim_fraction"),
            ),
            "per_action_reward_sensitivity.q50": (
                prior_summary["per_action_reward_sensitivity.q50"],
                (ant.get("per_action_reward_sensitivity") or {}).get("q50"),
            ),
            "per_action_reward_sensitivity.effective_dim_fraction": (
                prior_summary["per_action_reward_sensitivity.effective_dim_fraction"],
                (ant.get("per_action_reward_sensitivity") or {}).get("effective_dim_fraction"),
            ),
            "terminal_event_rate": (prior_summary["terminal_event_rate"], ant.get("terminal_event_rate")),
            "terminal_first_event_fraction": (
                prior_summary["terminal_first_event_fraction"],
                ant.get("terminal_first_event_fraction"),
            ),
        }
        return {
            key: {
                **_ratio(prior_stat, ant_value),
                "prior": prior_stat,
            }
            for key, (prior_stat, ant_value) in comparisons.items()
        }
    gap_table_by_gym = {
        str(env_id): _build_gap_table(report) for env_id, report in gym_references.items()
    }
    gap_table = _build_gap_table(ant_report) if ant_report is not None else {}
    summary = {
        "audit_entry": "phase2_prior_gym_alignment_deep_feature_audit",
        "inputs": {
            "screening_csv": str(Path(args.screening_csv).expanduser().resolve()),
            "selected_prior_count": int(len(selected_rows)),
            "min_gain": None if args.min_gain is None else float(args.min_gain),
            "cases": cases,
            "prior_rollout_count": int(args.prior_rollout_count),
            "prior_n_steps": int(args.prior_n_steps),
            "prior_per_action_stride": int(args.prior_per_action_stride),
            "action_delta": float(args.action_delta),
            "device": str(args.device),
            "ant_env_id": str(args.ant_env_id),
            "gym_env_ids": gym_env_ids,
            "paired_case_rollout_seeds": bool(args.paired_case_rollout_seeds),
        },
        "prior_summary": prior_summary,
        "prior_summary_by_case": prior_summary_by_case,
        "ant_reference": ant_report,
        "gym_references": gym_references,
        "gap_table": gap_table,
        "gap_table_by_gym": gap_table_by_gym,
        "interpretation_guardrail": (
            "This audit is diagnostic only. Per-action and per-observation distributions are required before "
            "turning scalar action sensitivity into a rejection rule or generator default."
        ),
        "rows_jsonl": str(rows_path),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(_json_safe(summary), sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(_json_safe(summary), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
