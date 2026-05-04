import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ticl.analysis.critic_free_single_env_audit import (
    _build_audit_env_cfg,
    _build_audit_ppo_policy_step_fn,
    _prepare_audit_fixed_env_contract,
    _seed_all,
)
from ticl.analysis.fixed_env_h import summarize_fixed_env_h
from ticl.analysis.phase2_legacy_bar_longrun_probe import (
    _build_algo,
    _build_fixed_unit_gaussian_policy_step_fn,
    _collect_rollout,
    _resolve_model_and_config,
)
from ticl.analysis.prior_generalization_audit import _maybe_clear_policy_buffers
from ticl.config_utils import str2bool
from ticl.model_configs import get_model_default_config
from ticl.rlpfn_maintained_path import validate_rlpfn_maintained_path_config
from ticl.priors.environment_prior import EnvironmentPrior


def _default_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _to_float(value: Any) -> float:
    if torch.is_tensor(value):
        return float(value.detach().reshape(-1)[0].item())
    return float(value)


def _to_bool(value: Any) -> bool:
    if torch.is_tensor(value):
        return bool(value.detach().reshape(-1)[0].item())
    return bool(value)


def _to_tensor_like(value: Any, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.to(device=device, dtype=dtype)
    return torch.as_tensor(value, device=device, dtype=dtype)


def _broadcast_like_batch(
    value: Any,
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    tensor = _to_tensor_like(value, device=device, dtype=dtype)
    if tensor.ndim == 0:
        return tensor.reshape(1).expand(int(batch_size))
    if tensor.numel() == 1:
        return tensor.reshape(1).expand(int(batch_size))
    return tensor.reshape(int(batch_size))


def _make_generator(*, seed: int, device: torch.device) -> torch.Generator:
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    return gen


def _compute_identity_metrics(return_matrix: np.ndarray, *, eps: float = 1e-8) -> dict[str, float]:
    values = np.asarray(return_matrix, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"return_matrix must have shape (states, repeats), got {tuple(values.shape)}")
    if int(values.shape[0]) <= 0 or int(values.shape[1]) <= 0:
        raise ValueError("return_matrix must be non-empty")
    per_state_means = values.mean(axis=1, dtype=np.float64).astype(np.float32, copy=False)
    per_state_vars = values.var(axis=1, dtype=np.float64).astype(np.float32, copy=False)
    noise = float(per_state_vars.mean(dtype=np.float64))
    signal = float(per_state_means.var(dtype=np.float64))
    total = float(signal + noise)
    identity_score = float(signal / max(total, float(eps)))
    return {
        "noise": noise,
        "signal": signal,
        "identity_score": identity_score,
        "total_variance": total,
        "mean_of_state_means": float(per_state_means.mean(dtype=np.float64)),
        "std_of_state_means": float(per_state_means.std(dtype=np.float64)),
        "mean_within_state_std": float(np.sqrt(np.maximum(per_state_vars, 0.0)).mean(dtype=np.float64)),
    }


def _compute_identity_metrics_from_return_lists(
    state_returns: list[list[float]],
    *,
    eps: float = 1e-8,
) -> dict[str, float]:
    if len(state_returns) <= 0:
        raise ValueError("state_returns must be non-empty")
    per_state_means = []
    per_state_vars = []
    for values in state_returns:
        if len(values) <= 0:
            continue
        arr = np.asarray(values, dtype=np.float32)
        per_state_means.append(float(arr.mean(dtype=np.float64)))
        per_state_vars.append(float(arr.var(dtype=np.float64)))
    if len(per_state_means) <= 0:
        raise ValueError("state_returns must contain at least one non-empty state")
    per_state_means_arr = np.asarray(per_state_means, dtype=np.float32)
    per_state_vars_arr = np.asarray(per_state_vars, dtype=np.float32)
    noise = float(per_state_vars_arr.mean(dtype=np.float64))
    signal = float(per_state_means_arr.var(dtype=np.float64))
    total = float(signal + noise)
    return {
        "noise": noise,
        "signal": signal,
        "identity_score": float(signal / max(total, float(eps))),
        "total_variance": total,
        "mean_of_state_means": float(per_state_means_arr.mean(dtype=np.float64)),
        "std_of_state_means": float(per_state_means_arr.std(dtype=np.float64)),
        "mean_within_state_std": float(np.sqrt(np.maximum(per_state_vars_arr, 0.0)).mean(dtype=np.float64)),
    }


def _append_jsonl(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _load_full_frozen_h(path: str) -> tuple[dict[str, Any], Path]:
    resolved = Path(path).expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(
            f"full frozen_h JSON must decode to a dict, got {type(payload).__name__} from {resolved}"
        )
    return payload, resolved


def _json_safe_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if torch.is_tensor(value):
        tensor = value.detach().cpu()
        if tensor.numel() == 1:
            return tensor.reshape(-1)[0].item()
        return tensor.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(v) for v in value]
    return str(value)


def _select_env_knob_snapshot(frozen_h: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "family",
        "state_dim",
        "obs_dim",
        "action_dim",
        "noise_dim",
        "zero_pad_dim",
        "constrained_dim_sampling_total_budget",
        "alpha",
        "reference_state_inertia_enabled",
        "terminal_reset_enabled",
        "terminal_reset_count_target",
        "prior_mlp_activations",
        "num_layers",
        "prior_mlp_hidden_dim",
        "scm_standard_linear_init_enabled",
        "init_std",
        "noise_std",
        "reward_state_input_gain_fraction_rejection_min",
        "reward_state_input_gain_fraction_rejection_max_tries",
        "prior_mlp_dropout_prob",
        "block_wise_dropout",
        "prior_mlp_scale_weights_sqrt",
        "pre_sample_weights",
        "random_feature_rotation",
        "in_clique",
        "sort_features",
        "y_is_effect",
    ]
    snapshot: dict[str, Any] = {}
    for key in keys:
        if key in frozen_h:
            snapshot[str(key)] = _json_safe_value(frozen_h[key])
    return snapshot


def _select_sampled_env_snapshot(env: dict[str, Any], frozen_h: dict[str, Any]) -> dict[str, Any]:
    snapshot = {
        "family": _json_safe_value(env.get("family")),
        "state_dim": _json_safe_value(env.get("state_dim")),
        "obs_dim": _json_safe_value(env.get("obs_dim")),
        "action_dim": _json_safe_value(env.get("action_dim")),
        "noise_dim": _json_safe_value(env.get("noise_dim")),
        "zero_pad_dim": _json_safe_value(env.get("zero_pad_dim")),
        "alpha": _json_safe_value(env.get("alpha")),
        "reference_state_inertia_enabled": _json_safe_value(env.get("reference_state_inertia_enabled")),
        "terminal_reset_enabled": _json_safe_value(env.get("terminal_reset_enabled")),
        "terminal_reset_count_target": _json_safe_value(env.get("terminal_reset_count_target")),
        "state_highway_enabled": _json_safe_value(env.get("state_highway_enabled")),
        "state_highway_lambda": _json_safe_value(env.get("state_highway_lambda")),
        "state_noise_std": _json_safe_value(env.get("state_noise_std")),
        "reward_scale": _json_safe_value(env.get("reward_scale")),
        "reward_state_input_gain_fraction": _json_safe_value(env.get("reward_state_input_gain_fraction")),
        "reward_action_input_gain_fraction": _json_safe_value(env.get("reward_action_input_gain_fraction")),
        "reward_noise_input_gain_fraction": _json_safe_value(env.get("reward_noise_input_gain_fraction")),
        "reward_state_input_gain_fraction_rejection_min": _json_safe_value(
            env.get("reward_state_input_gain_fraction_rejection_min")
        ),
        "reward_state_input_gain_fraction_rejection_attempt": _json_safe_value(
            env.get("reward_state_input_gain_fraction_rejection_attempt")
        ),
        "reward_state_input_gain_fraction_rejection_max_tries": _json_safe_value(
            env.get("reward_state_input_gain_fraction_rejection_max_tries")
        ),
        "reward_state_input_gain_fraction_rejection_accepted": _json_safe_value(
            env.get("reward_state_input_gain_fraction_rejection_accepted")
        ),
        "ctrl_reward_enabled": _json_safe_value(env.get("ctrl_reward_enabled")),
        "survival_reward_enabled": _json_safe_value(env.get("survival_reward_enabled")),
    }
    for key in (
        "prior_mlp_activations",
        "num_layers",
        "prior_mlp_hidden_dim",
        "scm_standard_linear_init_enabled",
        "init_std",
        "noise_std",
        "prior_mlp_dropout_prob",
        "block_wise_dropout",
        "prior_mlp_scale_weights_sqrt",
        "pre_sample_weights",
        "random_feature_rotation",
        "in_clique",
        "sort_features",
        "y_is_effect",
    ):
        if key in frozen_h:
            snapshot[str(key)] = _json_safe_value(frozen_h[key])
    return snapshot


def _format_progress_line(
    *,
    state_idx: int,
    total_states: int,
    repeat_idx: int,
    total_repeats: int,
    partial_metrics: dict[str, float],
    current_return: float,
) -> str:
    return (
        f"[identity-probe] state {int(state_idx) + 1}/{int(total_states)} "
        f"repeat {int(repeat_idx) + 1}/{int(total_repeats)} "
        f"return={float(current_return):+.4f} "
        f"noise={float(partial_metrics['noise']):.4f} "
        f"signal={float(partial_metrics['signal']):.4f} "
        f"identity={float(partial_metrics['identity_score']):.4f}"
    )


def _build_env_info(
    *,
    env: dict[str, Any],
    action_dim: int,
    obs_dim: int,
    step_idx: int,
    single_eval_pos: int,
    terminal_t: torch.Tensor,
) -> dict[str, Any]:
    terminal_batch = terminal_t.reshape(-1).to(dtype=torch.float32)
    phase_t = torch.ones_like(terminal_batch, dtype=torch.float32)
    if int(step_idx) < int(single_eval_pos):
        phase_t = torch.zeros_like(terminal_batch, dtype=torch.float32)
    return {
        "action_dim": int(action_dim),
        "action_dim_per_sample": torch.full(
            (int(terminal_batch.shape[0]),),
            int(action_dim),
            device=terminal_batch.device,
            dtype=torch.long,
        ),
        "obs_dim": int(obs_dim),
        "obs_slot_dim": int(env["obs_slot_dim"]),
        "action_slot_dim": int(env["action_slot_dim"]),
        "terminal_reset_enabled": bool(env.get("terminal_reset_enabled", False)),
        "phase_t": phase_t,
        "terminal_t": terminal_batch,
    }


def _clone_suffix_bundle(prior: EnvironmentPrior, bundle: dict[str, Any]) -> dict[str, Any]:
    return {
        "step_idx": int(bundle["step_idx"]),
        "carry_state_t": bundle["carry_state_t"].detach().clone(),
        "state_t": bundle["state_t"].detach().clone(),
        "action_t": bundle["action_t"].detach().clone(),
        "reward_t": bundle["reward_t"].detach().clone(),
        "reward_mask_t": bundle["reward_mask_t"].detach().clone(),
        "terminal_t": bundle["terminal_t"].detach().clone(),
        "cache": prior._detach_policy_cache(bundle["cache"], clone_tensors=True),
    }


def _clone_suffix_bundle_batch(prior: EnvironmentPrior, bundle: dict[str, Any]) -> dict[str, Any]:
    cloned = {
        "step_idx": int(bundle["step_idx"]),
        "carry_state_t": bundle["carry_state_t"].detach().clone(),
        "state_t": bundle["state_t"].detach().clone(),
        "action_t": bundle["action_t"].detach().clone(),
        "reward_t": bundle["reward_t"].detach().clone(),
        "reward_mask_t": bundle["reward_mask_t"].detach().clone(),
        "terminal_t": bundle["terminal_t"].detach().clone(),
        "cache": prior._detach_policy_cache(bundle.get("cache", None), clone_tensors=True),
    }
    return cloned


def _slice_suffix_bundle_batch(
    prior: EnvironmentPrior,
    bundle: dict[str, Any],
    start: int,
    end: int,
) -> dict[str, Any]:
    sliced = {
        "step_idx": int(bundle["step_idx"]),
        "carry_state_t": bundle["carry_state_t"][int(start): int(end)].detach().clone(),
        "state_t": bundle["state_t"][int(start): int(end)].detach().clone(),
        "action_t": bundle["action_t"][int(start): int(end)].detach().clone(),
        "reward_t": bundle["reward_t"][int(start): int(end)].detach().clone(),
        "reward_mask_t": bundle["reward_mask_t"][int(start): int(end)].detach().clone(),
        "terminal_t": bundle["terminal_t"][int(start): int(end)].detach().clone(),
    }
    cache = bundle.get("cache", None)
    if cache is None:
        sliced["cache"] = None
    else:
        sliced["cache"] = prior._detach_policy_cache(cache, clone_tensors=True)
    return sliced


def _call_generator_with_noise_batch(fn, x, generators_for_noise):
    try:
        return fn(x, generators_for_noise=generators_for_noise)
    except TypeError:
        try:
            return fn(x, generators_for_noise=generators_for_noise, x_is_dual_packed=False, x_input_is_packed=False)
        except TypeError:
            return fn(x)


def _resolve_fixed_initial_state_batch(
    *,
    prior: EnvironmentPrior,
    env: dict[str, Any],
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    ) -> torch.Tensor:
    return prior._resolve_env_initial_state_batch(
        env,
        batch_size=int(batch_size),
        state_dim=int(env["state_dim"]),
        device=device,
        dtype=dtype,
    )


def _validate_discount_gamma(discount_gamma: float) -> float:
    gamma = float(discount_gamma)
    if (not np.isfinite(gamma)) or gamma < 0.0 or gamma > 1.0:
        raise ValueError(f"discount_gamma must lie in [0, 1], got {discount_gamma!r}")
    return gamma


def _resolve_horizon_diagnostics_steps(raw_steps: Any, *, remaining_horizon: int) -> list[int]:
    if raw_steps is None:
        return []
    horizon = int(remaining_horizon)
    if horizon <= 0:
        raise ValueError(f"remaining_horizon must be positive, got {horizon}")
    if isinstance(raw_steps, str):
        text = raw_steps.strip().lower()
        if text == "" or text in {"none", "false", "off"}:
            return []
        if text in {"powers", "power2", "default"}:
            steps: list[int] = []
            value = 1
            while value <= horizon:
                steps.append(int(value))
                value *= 2
            if steps[-1] != horizon:
                steps.append(int(horizon))
            return steps
        parts = [part.strip() for part in text.split(",") if part.strip()]
        steps = [int(part) for part in parts]
    else:
        steps = [int(v) for v in raw_steps]
    resolved: list[int] = []
    seen: set[int] = set()
    for step in steps:
        if int(step) <= 0 or int(step) > horizon:
            raise ValueError(f"horizon diagnostic step must lie in [1, {horizon}], got {int(step)}")
        if int(step) not in seen:
            resolved.append(int(step))
            seen.add(int(step))
    return resolved


def _compute_horizon_diagnostics(
    discounted_step_rewards: np.ndarray,
    *,
    horizon_steps: list[int],
    continuation_repeats: int,
    eps: float = 1e-8,
) -> list[dict[str, Any]]:
    values = np.asarray(discounted_step_rewards, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError(
            "discounted_step_rewards must have shape (states, repeats, horizon), "
            f"got {tuple(values.shape)}"
        )
    if int(values.shape[0]) <= 0 or int(values.shape[1]) <= 0 or int(values.shape[2]) <= 0:
        raise ValueError("discounted_step_rewards must be non-empty")
    if int(continuation_repeats) != int(values.shape[1]):
        raise ValueError(
            f"continuation_repeats={int(continuation_repeats)} does not match reward tensor repeats={values.shape[1]}"
        )
    cumulative_returns = np.cumsum(values, axis=2, dtype=np.float32)
    diagnostics: list[dict[str, Any]] = []
    for horizon_step in horizon_steps:
        returns_at_horizon = cumulative_returns[:, :, int(horizon_step) - 1]
        metrics = _compute_identity_metrics(returns_at_horizon, eps=float(eps))
        finite_repeat_signal_bias_estimate = float(metrics["noise"] / max(1, int(continuation_repeats)))
        bias_corrected_signal = float(max(0.0, float(metrics["signal"]) - finite_repeat_signal_bias_estimate))
        bias_corrected_identity = float(
            bias_corrected_signal / max(bias_corrected_signal + float(metrics["noise"]), float(eps))
        )
        diagnostics.append(
            {
                "horizon_step": int(horizon_step),
                "g_k_definition": "sum_{i=0}^{k-1} discount_gamma^i * r_{anchor+i+1}",
                **metrics,
                "finite_repeat_signal_bias_estimate": finite_repeat_signal_bias_estimate,
                "bias_corrected_signal": bias_corrected_signal,
                "bias_corrected_identity_score": bias_corrected_identity,
                "signal_to_noise": float(float(metrics["signal"]) / max(float(metrics["noise"]), float(eps))),
                "bias_corrected_signal_to_noise": float(
                    bias_corrected_signal / max(float(metrics["noise"]), float(eps))
                ),
            }
        )
    return diagnostics


def _capture_suffix_bundles_unit_gaussian_batch(
    *,
    prior: EnvironmentPrior,
    env: dict[str, Any],
    policy_step_fn,
    sampled_policy: bool,
    anchor_step: int,
    n_steps: int,
    single_eval_pos: int,
    device: torch.device,
    rng_seeds: list[int],
) -> dict[str, Any]:
    batch_size = int(len(rng_seeds))
    if batch_size <= 0:
        raise ValueError("rng_seeds must be non-empty")
    action_dim = int(env["action_dim"])
    obs_dim = int(env["obs_dim"])
    state_dim = int(env["state_dim"])
    zero_pad_dim = int(env["zero_pad_dim"])
    reference_semantics_enabled = bool(prior._env_uses_reference_semantics(env))
    action_transform_mode = env.get("reinforce_action_transform", "clip")
    action_rms_eps = float(env.get("reinforce_action_rms_eps", 1e-6))
    action_clip_bound = float(env.get("reinforce_action_clip_bound", 5.0))
    state_input_scale = env.get("state_input_scale", 1.0)
    dtype = torch.float32
    rollout_generators = prior._make_generators_from_seeds([int(v) for v in rng_seeds], batch_size, device)

    fixed_initial_state_batch = _resolve_fixed_initial_state_batch(
        prior=prior,
        env=env,
        batch_size=batch_size,
        device=device,
        dtype=dtype,
    )
    init_action_std = _broadcast_like_batch(env["init_action_std"], batch_size=batch_size, device=device, dtype=dtype)
    carry_state_t, state_t = prior._init_exact_scm_carry_and_visible_state(fixed_initial_state_batch)
    action_t = prior._stack_randn_with_generators(
        rollout_generators, (batch_size, action_dim), device, dtype
    ) * init_action_std[:, None]
    reward_t = torch.zeros((batch_size,), device=device, dtype=dtype)
    reward_mask_t = torch.ones((batch_size,), device=device, dtype=dtype)
    terminal_t = torch.zeros((batch_size,), device=device, dtype=dtype)
    cache = None
    zero_pad_t = torch.zeros((batch_size, zero_pad_dim), device=device, dtype=dtype)

    action_noise_train_std = _to_float(env.get("action_noise_train_std", 0.0))
    action_noise_eval_std = _to_float(env.get("action_noise_eval_std", 0.0))
    state_noise_std = _to_float(env.get("state_noise_std", 0.0))
    reward_dropout_enabled = _to_bool(env.get("reward_dropout_enabled", False))
    reward_dropout_ratio = _to_float(env.get("reward_dropout_ratio", 0.0))
    reward_impute_zero = _to_bool(env.get("reward_dropout_impute_zero", True))
    terminal_reset_enabled = _to_bool(env.get("terminal_reset_enabled", False))
    terminal_reset_prob = _to_float(
        prior._terminal_reset_prob_from_count(env.get("terminal_reset_count_target", 0), int(n_steps))
    )
    terminal_reset_prob_t = _broadcast_like_batch(
        float(terminal_reset_prob),
        batch_size=batch_size,
        device=device,
        dtype=dtype,
    )
    terminal_bonus_tanh_c = _to_float(env.get("terminal_bonus_tanh_c", 10.0))
    terminal_bonus_scale_min = _to_float(env.get("terminal_bonus_scale_min", 1.0))
    terminal_bonus_scale_max = _to_float(env.get("terminal_bonus_scale_max", 2.0))
    terminal_signal_history = (
        torch.zeros((int(n_steps), batch_size), device=device, dtype=dtype) if terminal_reset_enabled else None
    )

    _maybe_clear_policy_buffers(policy_step_fn)
    for step_idx in range(int(n_steps)):
        if int(step_idx) == int(anchor_step):
            return {
                "step_idx": int(step_idx),
                "carry_state_t": carry_state_t.detach().clone(),
                "state_t": state_t.detach().clone(),
                "action_t": action_t.detach().clone(),
                "reward_t": reward_t.detach().clone(),
                "reward_mask_t": reward_mask_t.detach().clone(),
                "terminal_t": terminal_t.detach().clone(),
                "cache": prior._detach_policy_cache(cache, clone_tensors=True),
            }

        obs_t = state_t[:, :obs_dim]
        env_info = _build_env_info(
            env=env,
            action_dim=int(action_dim),
            obs_dim=int(obs_dim),
            step_idx=int(step_idx),
            single_eval_pos=int(single_eval_pos),
            terminal_t=terminal_t,
        )
        policy_out = policy_step_fn(
            obs_t,
            action_t,
            reward_t.reshape(batch_size, 1),
            reward_mask_t.reshape(batch_size, 1),
            cache,
            int(step_idx),
            env_info,
        )
        actor_outputs, action_mean, next_cache = prior._unpack_policy_step_output(policy_out)
        action_eps_t = (
            prior._stack_randn_with_generators(rollout_generators, (batch_size, action_dim), device, dtype)
            if bool(sampled_policy)
            else None
        )
        sampled_action = prior._resolve_policy_action_sample(
            policy_step_fn=policy_step_fn,
            actor_outputs=actor_outputs,
            action_mean=action_mean,
            sample_action=bool(sampled_policy),
            action_eps_t=action_eps_t,
            action_transform_mode=action_transform_mode,
            action_rms_eps=action_rms_eps,
            action_clip_bound=action_clip_bound,
            collect_log_probs=False,
            collect_log_prob_score=False,
            legacy_action_std=torch.ones((batch_size,), device=device, dtype=dtype),
        )
        action_next = sampled_action["action_next"]

        if (not reference_semantics_enabled) and action_noise_train_std > 0.0 and int(step_idx) < int(single_eval_pos):
            eps_t = prior._stack_randn_with_generators(rollout_generators, (batch_size, action_dim), device, dtype)
            action_next = prior._transform_reinforce_action(
                action_next + eps_t * float(action_noise_train_std),
                mode=action_transform_mode,
                rms_eps=action_rms_eps,
                clip_bound=action_clip_bound,
            )
        elif (not reference_semantics_enabled) and action_noise_eval_std > 0.0:
            eps_t = prior._stack_randn_with_generators(rollout_generators, (batch_size, action_dim), device, dtype)
            action_next = prior._transform_reinforce_action(
                action_next + eps_t * float(action_noise_eval_std),
                mode=action_transform_mode,
                rms_eps=action_rms_eps,
                clip_bound=action_clip_bound,
            )

        if int(env["noise_dim"]) > 0:
            noise_t = prior._stack_randn_with_generators(
                rollout_generators, (batch_size, int(env["noise_dim"])), device, dtype
            )
        else:
            noise_t = torch.zeros((batch_size, 0), device=device, dtype=dtype)
        env_in = prior._pack_env_input(
            carry_state_t,
            obs_t,
            action_next,
            noise_t,
            zero_pad_t,
            reference_semantics_enabled=reference_semantics_enabled,
            state_input_scale=state_input_scale,
        )
        transition_generator = prior._require_transition_generator(env)
        terminal_signal_next = None
        terminal_bonus_base_next = None
        reward_scale = _broadcast_like_batch(env["reward_scale"], batch_size=batch_size, device=device, dtype=dtype)
        transition_out = _call_generator_with_noise_batch(transition_generator, env_in, rollout_generators)
        x_next, reward_unit, terminal_signal_next, terminal_bonus_base_next = prior._unpack_transition_output(
            transition_out
        )
        reward_next_raw = reward_scale * reward_unit.reshape(batch_size)

        aux_reward_next = prior._exact_scm_aux_reward_terms(
            action_next,
            ctrl_weight=env.get("ctrl_reward_weight", 0.0),
            ctrl_enabled=env.get("ctrl_reward_enabled", False),
            survival_weight=env.get("survival_reward_weight", 0.0),
            survival_enabled=env.get("survival_reward_enabled", False),
        )
        reward_next = prior._compose_exact_scm_reward(
            reward_next_raw.reshape(batch_size),
            aux_reward=aux_reward_next.reshape(batch_size),
            reward_clip=env.get("reward_clip", float("inf")),
            mode=env.get("reinforce_reward_transform", "none"),
            rms_eps=env.get("reinforce_reward_rms_eps", 1e-6),
            tanh_c=env.get("reinforce_reward_tanh_c", 1.0),
            tanh_bound=env.get("reinforce_reward_tanh_bound", 2.0),
        ).reshape(batch_size)
        reward_mask_next = torch.ones((batch_size,), device=device, dtype=dtype)
        if reward_dropout_enabled and reward_dropout_ratio > 0.0:
            drop_mask = prior._stack_rand_with_generators(rollout_generators, (batch_size,), device, dtype) < float(
                reward_dropout_ratio
            )
            reward_mask_next = torch.where(drop_mask, torch.zeros_like(reward_mask_next), reward_mask_next)
            if bool(reward_impute_zero):
                reward_next = torch.where(drop_mask, torch.zeros_like(reward_next), reward_next)

        state_noise_t = None
        if state_noise_std > 0.0:
            state_noise_t = prior._stack_randn_with_generators(
                rollout_generators, (batch_size, state_dim), device, dtype
            )
        state_next = prior._finalize_exact_scm_visible_state_next(
            generated_state_next=x_next,
            visible_state_prev=state_t,
            state_noise_t=state_noise_t,
            state_noise_std=env.get("state_noise_std", 0.0),
            reference_semantics_enabled=reference_semantics_enabled,
            state_clip=env["state_clip"],
            state_highway_enabled=env.get("state_highway_enabled", False),
            state_highway_lambda=env.get("state_highway_lambda", 0.0),
            state_full_rms_enabled=env.get("state_full_rms_enabled", False),
            state_full_rms_target=env.get("state_full_rms_target", 1.0),
        )
        if terminal_reset_enabled:
            terminal_draw = prior._stack_rand_with_generators(rollout_generators, (batch_size,), device, dtype)
            bonus_scale_draw = prior._stack_rand_with_generators(rollout_generators, (batch_size,), device, dtype)
            state_next, reward_next, terminal_next = prior._apply_terminal_reset_step(
                state_next=state_next,
                reward_next=reward_next,
                terminal_draw=terminal_draw,
                reset_prob=terminal_reset_prob_t,
                bonus_scale_draw=bonus_scale_draw,
                bonus_scale_min=float(terminal_bonus_scale_min),
                bonus_scale_max=float(terminal_bonus_scale_max),
                bonus_tanh_c=float(terminal_bonus_tanh_c),
                reset_state=fixed_initial_state_batch,
                enabled=torch.ones((batch_size,), device=device, dtype=torch.bool),
                terminal_signal=terminal_signal_next,
                terminal_bonus_base=terminal_bonus_base_next,
                terminal_signal_history=terminal_signal_history,
                history_index=int(step_idx),
            )
            terminal_next = terminal_next.reshape(batch_size)
        else:
            terminal_next = torch.zeros((batch_size,), device=device, dtype=dtype)

        carry_state_next = prior._mix_exact_scm_carry_state(
            carry_state_t,
            state_next,
            env["alpha"],
        )
        if terminal_reset_enabled:
            carry_state_next = prior._sync_exact_scm_carry_state_with_visible(
                carry_state_next,
                state_next,
                terminal_next,
            )

        carry_state_t = carry_state_next
        state_t = state_next
        action_t = action_next
        reward_t = reward_next
        reward_mask_t = reward_mask_next
        terminal_t = terminal_next
        cache = next_cache

    raise RuntimeError(
        f"Failed to capture suffix bundle for anchor_step={int(anchor_step)} within n_steps={int(n_steps)}."
    )


def _repeat_suffix_bundle_batch(prior: EnvironmentPrior, bundle: dict[str, Any], repeats: int) -> dict[str, Any]:
    repeats = int(repeats)
    if repeats <= 0:
        raise ValueError(f"repeats must be > 0, got {repeats}")
    repeated = {
        "step_idx": int(bundle["step_idx"]),
        "carry_state_t": bundle["carry_state_t"].repeat_interleave(repeats, dim=0),
        "state_t": bundle["state_t"].repeat_interleave(repeats, dim=0),
        "action_t": bundle["action_t"].repeat_interleave(repeats, dim=0),
        "reward_t": bundle["reward_t"].repeat_interleave(repeats, dim=0),
        "reward_mask_t": bundle["reward_mask_t"].repeat_interleave(repeats, dim=0),
        "terminal_t": bundle["terminal_t"].repeat_interleave(repeats, dim=0),
        "cache": prior._detach_policy_cache(bundle.get("cache", None), clone_tensors=True),
    }
    return repeated


def _rollout_suffix_returns_from_bundle_batch_unit_gaussian(
    *,
    prior: EnvironmentPrior,
    env: dict[str, Any],
    policy_step_fn,
    sampled_policy: bool,
    bundle: dict[str, Any],
    n_steps: int,
    single_eval_pos: int,
    discount_gamma: float,
    device: torch.device,
    rng_seeds: list[int],
    return_discounted_step_rewards: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    batch_size = int(len(rng_seeds))
    if batch_size <= 0:
        return np.zeros((0,), dtype=np.float32)
    action_dim = int(env["action_dim"])
    obs_dim = int(env["obs_dim"])
    state_dim = int(env["state_dim"])
    zero_pad_dim = int(env["zero_pad_dim"])
    reference_semantics_enabled = bool(prior._env_uses_reference_semantics(env))
    action_transform_mode = env.get("reinforce_action_transform", "clip")
    action_rms_eps = float(env.get("reinforce_action_rms_eps", 1e-6))
    action_clip_bound = float(env.get("reinforce_action_clip_bound", 5.0))
    state_input_scale = env.get("state_input_scale", 1.0)
    dtype = torch.float32
    rollout_generators = prior._make_generators_from_seeds([int(v) for v in rng_seeds], batch_size, device)

    carry_state_t = bundle["carry_state_t"].detach().clone().to(device=device, dtype=dtype)
    state_t = bundle["state_t"].detach().clone().to(device=device, dtype=dtype)
    action_t = bundle["action_t"].detach().clone().to(device=device, dtype=dtype)
    reward_t = bundle["reward_t"].detach().clone().to(device=device, dtype=dtype)
    reward_mask_t = bundle["reward_mask_t"].detach().clone().to(device=device, dtype=dtype)
    terminal_t = bundle["terminal_t"].detach().clone().to(device=device, dtype=dtype)
    cache = prior._detach_policy_cache(bundle.get("cache", None), clone_tensors=True)
    zero_pad_t = torch.zeros((batch_size, zero_pad_dim), device=device, dtype=dtype)
    total_return = torch.zeros((batch_size,), device=device, dtype=dtype)
    remaining_horizon = int(n_steps) - int(bundle["step_idx"])
    discounted_step_rewards = (
        torch.zeros((batch_size, int(remaining_horizon)), device=device, dtype=dtype)
        if bool(return_discounted_step_rewards)
        else None
    )

    action_noise_train_std = _to_float(env.get("action_noise_train_std", 0.0))
    action_noise_eval_std = _to_float(env.get("action_noise_eval_std", 0.0))
    state_noise_std = _to_float(env.get("state_noise_std", 0.0))
    reward_dropout_enabled = _to_bool(env.get("reward_dropout_enabled", False))
    reward_dropout_ratio = _to_float(env.get("reward_dropout_ratio", 0.0))
    reward_impute_zero = _to_bool(env.get("reward_dropout_impute_zero", True))
    terminal_reset_enabled = _to_bool(env.get("terminal_reset_enabled", False))
    terminal_reset_prob = _to_float(
        prior._terminal_reset_prob_from_count(env.get("terminal_reset_count_target", 0), int(n_steps))
    )
    terminal_reset_prob_t = _broadcast_like_batch(
        float(terminal_reset_prob),
        batch_size=batch_size,
        device=device,
        dtype=dtype,
    )
    terminal_bonus_tanh_c = _to_float(env.get("terminal_bonus_tanh_c", 10.0))
    terminal_bonus_scale_min = _to_float(env.get("terminal_bonus_scale_min", 1.0))
    terminal_bonus_scale_max = _to_float(env.get("terminal_bonus_scale_max", 2.0))
    fixed_initial_state_batch = _resolve_fixed_initial_state_batch(
        prior=prior,
        env=env,
        batch_size=batch_size,
        device=device,
        dtype=dtype,
    )
    terminal_signal_history = (
        torch.zeros((int(n_steps), batch_size), device=device, dtype=dtype) if terminal_reset_enabled else None
    )
    gamma = float(discount_gamma)
    running_discount = torch.ones((batch_size,), device=device, dtype=dtype)

    start_step = int(bundle["step_idx"])
    for step_idx in range(start_step, int(n_steps)):
        obs_t = state_t[:, :obs_dim]
        env_info = _build_env_info(
            env=env,
            action_dim=int(action_dim),
            obs_dim=int(obs_dim),
            step_idx=int(step_idx),
            single_eval_pos=int(single_eval_pos),
            terminal_t=terminal_t,
        )
        policy_out = policy_step_fn(
            obs_t,
            action_t,
            reward_t.reshape(batch_size, 1),
            reward_mask_t.reshape(batch_size, 1),
            cache,
            int(step_idx),
            env_info,
        )
        actor_outputs, action_mean, next_cache = prior._unpack_policy_step_output(policy_out)
        action_eps_t = (
            prior._stack_randn_with_generators(rollout_generators, (batch_size, action_dim), device, dtype)
            if bool(sampled_policy)
            else None
        )
        sampled_action = prior._resolve_policy_action_sample(
            policy_step_fn=policy_step_fn,
            actor_outputs=actor_outputs,
            action_mean=action_mean,
            sample_action=bool(sampled_policy),
            action_eps_t=action_eps_t,
            action_transform_mode=action_transform_mode,
            action_rms_eps=action_rms_eps,
            action_clip_bound=action_clip_bound,
            collect_log_probs=False,
            collect_log_prob_score=False,
            legacy_action_std=torch.ones((batch_size,), device=device, dtype=dtype),
        )
        action_next = sampled_action["action_next"]

        if (not reference_semantics_enabled) and action_noise_train_std > 0.0 and int(step_idx) < int(single_eval_pos):
            eps_t = prior._stack_randn_with_generators(rollout_generators, (batch_size, action_dim), device, dtype)
            action_next = prior._transform_reinforce_action(
                action_next + eps_t * float(action_noise_train_std),
                mode=action_transform_mode,
                rms_eps=action_rms_eps,
                clip_bound=action_clip_bound,
            )
        elif (not reference_semantics_enabled) and action_noise_eval_std > 0.0:
            eps_t = prior._stack_randn_with_generators(rollout_generators, (batch_size, action_dim), device, dtype)
            action_next = prior._transform_reinforce_action(
                action_next + eps_t * float(action_noise_eval_std),
                mode=action_transform_mode,
                rms_eps=action_rms_eps,
                clip_bound=action_clip_bound,
            )

        if int(env["noise_dim"]) > 0:
            noise_t = prior._stack_randn_with_generators(
                rollout_generators, (batch_size, int(env["noise_dim"])), device, dtype
            )
        else:
            noise_t = torch.zeros((batch_size, 0), device=device, dtype=dtype)
        env_in = prior._pack_env_input(
            carry_state_t,
            obs_t,
            action_next,
            noise_t,
            zero_pad_t,
            reference_semantics_enabled=reference_semantics_enabled,
            state_input_scale=state_input_scale,
        )
        transition_generator = prior._require_transition_generator(env)
        terminal_signal_next = None
        terminal_bonus_base_next = None
        reward_scale = _broadcast_like_batch(env["reward_scale"], batch_size=batch_size, device=device, dtype=dtype)
        transition_out = _call_generator_with_noise_batch(transition_generator, env_in, rollout_generators)
        x_next, reward_unit, terminal_signal_next, terminal_bonus_base_next = prior._unpack_transition_output(
            transition_out
        )
        reward_next_raw = reward_scale * reward_unit.reshape(batch_size)

        aux_reward_next = prior._exact_scm_aux_reward_terms(
            action_next,
            ctrl_weight=env.get("ctrl_reward_weight", 0.0),
            ctrl_enabled=env.get("ctrl_reward_enabled", False),
            survival_weight=env.get("survival_reward_weight", 0.0),
            survival_enabled=env.get("survival_reward_enabled", False),
        )
        reward_next = prior._compose_exact_scm_reward(
            reward_next_raw.reshape(batch_size),
            aux_reward=aux_reward_next.reshape(batch_size),
            reward_clip=env.get("reward_clip", float("inf")),
            mode=env.get("reinforce_reward_transform", "none"),
            rms_eps=env.get("reinforce_reward_rms_eps", 1e-6),
            tanh_c=env.get("reinforce_reward_tanh_c", 1.0),
            tanh_bound=env.get("reinforce_reward_tanh_bound", 2.0),
        ).reshape(batch_size)
        reward_mask_next = torch.ones((batch_size,), device=device, dtype=dtype)
        if reward_dropout_enabled and reward_dropout_ratio > 0.0:
            drop_mask = prior._stack_rand_with_generators(rollout_generators, (batch_size,), device, dtype) < float(
                reward_dropout_ratio
            )
            reward_mask_next = torch.where(drop_mask, torch.zeros_like(reward_mask_next), reward_mask_next)
            if bool(reward_impute_zero):
                reward_next = torch.where(drop_mask, torch.zeros_like(reward_next), reward_next)

        state_noise_t = None
        if state_noise_std > 0.0:
            state_noise_t = prior._stack_randn_with_generators(
                rollout_generators, (batch_size, state_dim), device, dtype
            )
        state_next = prior._finalize_exact_scm_visible_state_next(
            generated_state_next=x_next,
            visible_state_prev=state_t,
            state_noise_t=state_noise_t,
            state_noise_std=env.get("state_noise_std", 0.0),
            reference_semantics_enabled=reference_semantics_enabled,
            state_clip=env["state_clip"],
            state_highway_enabled=env.get("state_highway_enabled", False),
            state_highway_lambda=env.get("state_highway_lambda", 0.0),
            state_full_rms_enabled=env.get("state_full_rms_enabled", False),
            state_full_rms_target=env.get("state_full_rms_target", 1.0),
        )
        if terminal_reset_enabled:
            terminal_draw = prior._stack_rand_with_generators(rollout_generators, (batch_size,), device, dtype)
            bonus_scale_draw = prior._stack_rand_with_generators(rollout_generators, (batch_size,), device, dtype)
            state_next, reward_next, terminal_next = prior._apply_terminal_reset_step(
                state_next=state_next,
                reward_next=reward_next,
                terminal_draw=terminal_draw,
                reset_prob=terminal_reset_prob_t,
                bonus_scale_draw=bonus_scale_draw,
                bonus_scale_min=float(terminal_bonus_scale_min),
                bonus_scale_max=float(terminal_bonus_scale_max),
                bonus_tanh_c=float(terminal_bonus_tanh_c),
                reset_state=fixed_initial_state_batch,
                enabled=torch.ones((batch_size,), device=device, dtype=torch.bool),
                terminal_signal=terminal_signal_next,
                terminal_bonus_base=terminal_bonus_base_next,
                terminal_signal_history=terminal_signal_history,
                history_index=int(step_idx),
            )
            terminal_next = terminal_next.reshape(batch_size)
        else:
            terminal_next = torch.zeros((batch_size,), device=device, dtype=dtype)

        discounted_reward_next = running_discount * reward_next
        total_return = total_return + discounted_reward_next
        if discounted_step_rewards is not None:
            discounted_step_rewards[:, int(step_idx) - int(start_step)] = discounted_reward_next
        running_discount = running_discount * gamma
        carry_state_next = prior._mix_exact_scm_carry_state(
            carry_state_t,
            state_next,
            env["alpha"],
        )
        if terminal_reset_enabled:
            carry_state_next = prior._sync_exact_scm_carry_state_with_visible(
                carry_state_next,
                state_next,
                terminal_next,
            )
        carry_state_t = carry_state_next
        state_t = state_next
        action_t = action_next
        reward_t = reward_next
        reward_mask_t = reward_mask_next
        terminal_t = terminal_next
        cache = next_cache

    total_return_np = total_return.detach().cpu().numpy().astype(np.float32, copy=False)
    if discounted_step_rewards is not None:
        return (
            total_return_np,
            discounted_step_rewards.detach().cpu().numpy().astype(np.float32, copy=False),
        )
    return total_return_np


def _capture_suffix_bundle(
    *,
    prior: EnvironmentPrior,
    env: dict[str, Any],
    policy_step_fn,
    sample_action: bool,
    anchor_step: int,
    n_steps: int,
    single_eval_pos: int,
    device: torch.device,
    rng_seed: int,
) -> dict[str, Any]:
    action_dim = int(env["action_dim"])
    obs_dim = int(env["obs_dim"])
    state_dim = int(env["state_dim"])
    zero_pad_dim = int(env["zero_pad_dim"])
    reference_semantics_enabled = bool(prior._env_uses_reference_semantics(env))
    action_transform_mode = env.get("reinforce_action_transform", "clip")
    action_rms_eps = float(env.get("reinforce_action_rms_eps", 1e-6))
    action_clip_bound = float(env.get("reinforce_action_clip_bound", 5.0))
    state_input_scale = env.get("state_input_scale", 1.0)
    gen = _make_generator(seed=int(rng_seed), device=device)
    dtype = torch.float32

    fixed_initial_state = _resolve_fixed_initial_state_batch(
        prior=prior,
        env=env,
        batch_size=1,
        device=device,
        dtype=dtype,
    )
    carry_state_t, state_t = prior._init_exact_scm_carry_and_visible_state(fixed_initial_state)
    action_t = torch.randn((1, action_dim), device=device, dtype=dtype, generator=gen) * _to_float(
        env["init_action_std"]
    )
    reward_t = torch.zeros((1,), device=device, dtype=dtype)
    reward_mask_t = torch.ones((1,), device=device, dtype=dtype)
    terminal_t = torch.zeros((1,), device=device, dtype=dtype)
    cache = None
    zero_pad_t = torch.zeros((1, zero_pad_dim), device=device, dtype=dtype)

    action_noise_train_std = _to_float(env.get("action_noise_train_std", 0.0))
    action_noise_eval_std = _to_float(env.get("action_noise_eval_std", 0.0))
    state_noise_std = _to_float(env.get("state_noise_std", 0.0))
    reward_dropout_enabled = _to_bool(env.get("reward_dropout_enabled", False))
    reward_dropout_ratio = _to_float(env.get("reward_dropout_ratio", 0.0))
    reward_impute_zero = _to_bool(env.get("reward_dropout_impute_zero", True))
    terminal_reset_enabled = _to_bool(env.get("terminal_reset_enabled", False))
    terminal_reset_prob = _to_float(
        prior._terminal_reset_prob_from_count(env.get("terminal_reset_count_target", 0), int(n_steps))
    )
    terminal_bonus_tanh_c = _to_float(env.get("terminal_bonus_tanh_c", 10.0))
    terminal_bonus_scale_min = _to_float(env.get("terminal_bonus_scale_min", 1.0))
    terminal_bonus_scale_max = _to_float(env.get("terminal_bonus_scale_max", 2.0))
    terminal_signal_history = (
        torch.zeros((int(n_steps), 1), device=device, dtype=dtype) if terminal_reset_enabled else None
    )

    _maybe_clear_policy_buffers(policy_step_fn)
    for step_idx in range(int(n_steps)):
        if int(step_idx) == int(anchor_step):
            return {
                "step_idx": int(step_idx),
                "carry_state_t": carry_state_t.detach().clone(),
                "state_t": state_t.detach().clone(),
                "action_t": action_t.detach().clone(),
                "reward_t": reward_t.detach().clone(),
                "reward_mask_t": reward_mask_t.detach().clone(),
                "terminal_t": terminal_t.detach().clone(),
                "cache": prior._detach_policy_cache(cache, clone_tensors=True),
            }

        obs_t = state_t[:, :obs_dim]
        env_info = _build_env_info(
            env=env,
            action_dim=int(action_dim),
            obs_dim=int(obs_dim),
            step_idx=int(step_idx),
            single_eval_pos=int(single_eval_pos),
            terminal_t=terminal_t,
        )
        policy_out = policy_step_fn(
            obs_t,
            action_t,
            reward_t,
            reward_mask_t,
            cache,
            int(step_idx),
            env_info,
        )
        actor_outputs, action_mean, next_cache = prior._unpack_policy_step_output(policy_out)
        sampled_action = prior._resolve_policy_action_sample(
            policy_step_fn=policy_step_fn,
            actor_outputs=actor_outputs,
            action_mean=action_mean,
            sample_action=bool(sample_action),
            action_eps_t=(
                torch.randn((1, action_dim), device=device, dtype=dtype, generator=gen)
                if bool(sample_action)
                else None
            ),
            action_transform_mode=action_transform_mode,
            action_rms_eps=action_rms_eps,
            action_clip_bound=action_clip_bound,
            collect_log_probs=False,
            collect_log_prob_score=False,
            legacy_action_std=torch.ones((1,), device=device, dtype=dtype),
        )
        action_next = sampled_action["action_next"]

        if (not reference_semantics_enabled) and action_noise_train_std > 0.0 and int(step_idx) < int(single_eval_pos):
            eps_t = torch.randn((1, action_dim), device=device, dtype=dtype, generator=gen)
            action_next = prior._transform_reinforce_action(
                action_next + eps_t * float(action_noise_train_std),
                mode=action_transform_mode,
                rms_eps=action_rms_eps,
                clip_bound=action_clip_bound,
            )
        elif (not reference_semantics_enabled) and action_noise_eval_std > 0.0:
            eps_t = torch.randn((1, action_dim), device=device, dtype=dtype, generator=gen)
            action_next = prior._transform_reinforce_action(
                action_next + eps_t * float(action_noise_eval_std),
                mode=action_transform_mode,
                rms_eps=action_rms_eps,
                clip_bound=action_clip_bound,
            )

        noise_t = torch.randn((1, int(env["noise_dim"])), device=device, dtype=dtype, generator=gen)
        env_in = prior._pack_env_input(
            carry_state_t,
            obs_t,
            action_next,
            noise_t,
            zero_pad_t,
            reference_semantics_enabled=reference_semantics_enabled,
            state_input_scale=state_input_scale,
        )
        transition_generator = prior._require_transition_generator(env)
        terminal_signal_next = None
        terminal_bonus_base_next = None
        transition_out = transition_generator(env_in, generator=gen)
        x_next, reward_unit, terminal_signal_next, terminal_bonus_base_next = prior._unpack_transition_output(
            transition_out
        )
        reward_next_raw = _to_tensor_like(env["reward_scale"], device=device, dtype=dtype).reshape(1) * reward_unit.reshape(1)

        aux_reward_next = prior._exact_scm_aux_reward_terms(
            action_next,
            ctrl_weight=env.get("ctrl_reward_weight", 0.0),
            ctrl_enabled=env.get("ctrl_reward_enabled", False),
            survival_weight=env.get("survival_reward_weight", 0.0),
            survival_enabled=env.get("survival_reward_enabled", False),
        )
        reward_next = prior._compose_exact_scm_reward(
            reward_next_raw.reshape(1),
            aux_reward=aux_reward_next.reshape(1),
            reward_clip=env.get("reward_clip", float("inf")),
            mode=env.get("reinforce_reward_transform", "none"),
            rms_eps=env.get("reinforce_reward_rms_eps", 1e-6),
            tanh_c=env.get("reinforce_reward_tanh_c", 1.0),
            tanh_bound=env.get("reinforce_reward_tanh_bound", 2.0),
        ).reshape(1)
        reward_mask_next = torch.ones((1,), device=device, dtype=dtype)
        if reward_dropout_enabled and reward_dropout_ratio > 0.0:
            drop_mask = torch.rand((1,), device=device, dtype=dtype, generator=gen) < float(reward_dropout_ratio)
            reward_mask_next = torch.where(drop_mask, torch.zeros_like(reward_mask_next), reward_mask_next)
            if bool(reward_impute_zero):
                reward_next = torch.where(drop_mask, torch.zeros_like(reward_next), reward_next)

        state_noise_t = None
        if state_noise_std > 0.0:
            state_noise_t = torch.randn((1, state_dim), device=device, dtype=dtype, generator=gen)
        state_next = prior._finalize_exact_scm_visible_state_next(
            generated_state_next=x_next,
            visible_state_prev=state_t,
            state_noise_t=state_noise_t,
            state_noise_std=env.get("state_noise_std", 0.0),
            reference_semantics_enabled=reference_semantics_enabled,
            state_clip=env["state_clip"],
            state_highway_enabled=env.get("state_highway_enabled", False),
            state_highway_lambda=env.get("state_highway_lambda", 0.0),
            state_full_rms_enabled=env.get("state_full_rms_enabled", False),
            state_full_rms_target=env.get("state_full_rms_target", 1.0),
        )
        if terminal_reset_enabled:
            terminal_draw = torch.rand((1,), device=device, dtype=dtype, generator=gen)
            bonus_scale_draw = torch.rand((1,), device=device, dtype=dtype, generator=gen)
            state_next, reward_next, terminal_next = prior._apply_terminal_reset_step(
                state_next=state_next,
                reward_next=reward_next,
                terminal_draw=terminal_draw,
                reset_prob=float(terminal_reset_prob),
                bonus_scale_draw=bonus_scale_draw,
                bonus_scale_min=float(terminal_bonus_scale_min),
                bonus_scale_max=float(terminal_bonus_scale_max),
                bonus_tanh_c=float(terminal_bonus_tanh_c),
                reset_state=fixed_initial_state,
                enabled=True,
                terminal_signal=terminal_signal_next,
                terminal_bonus_base=terminal_bonus_base_next,
                terminal_signal_history=terminal_signal_history,
                history_index=int(step_idx),
            )
            terminal_next = terminal_next.reshape(1)
        else:
            terminal_next = torch.zeros((1,), device=device, dtype=dtype)

        carry_state_next = prior._mix_exact_scm_carry_state(
            carry_state_t,
            state_next,
            env["alpha"],
        )
        if terminal_reset_enabled:
            carry_state_next = prior._sync_exact_scm_carry_state_with_visible(
                carry_state_next,
                state_next,
                terminal_next,
            )

        carry_state_t = carry_state_next
        state_t = state_next
        action_t = action_next
        reward_t = reward_next
        reward_mask_t = reward_mask_next
        terminal_t = terminal_next
        cache = next_cache

    raise RuntimeError(
        f"Failed to capture suffix bundle for anchor_step={int(anchor_step)} within n_steps={int(n_steps)}."
    )


def _rollout_suffix_return_from_bundle(
    *,
    prior: EnvironmentPrior,
    env: dict[str, Any],
    policy_step_fn,
    sample_action: bool,
    bundle: dict[str, Any],
    n_steps: int,
    single_eval_pos: int,
    discount_gamma: float,
    device: torch.device,
    rng_seed: int,
) -> float:
    action_dim = int(env["action_dim"])
    obs_dim = int(env["obs_dim"])
    state_dim = int(env["state_dim"])
    zero_pad_dim = int(env["zero_pad_dim"])
    reference_semantics_enabled = bool(prior._env_uses_reference_semantics(env))
    action_transform_mode = env.get("reinforce_action_transform", "clip")
    action_rms_eps = float(env.get("reinforce_action_rms_eps", 1e-6))
    action_clip_bound = float(env.get("reinforce_action_clip_bound", 5.0))
    state_input_scale = env.get("state_input_scale", 1.0)
    gen = _make_generator(seed=int(rng_seed), device=device)
    dtype = torch.float32

    carry_state_t = bundle["carry_state_t"].detach().clone().to(device=device, dtype=dtype)
    state_t = bundle["state_t"].detach().clone().to(device=device, dtype=dtype)
    action_t = bundle["action_t"].detach().clone().to(device=device, dtype=dtype)
    reward_t = bundle["reward_t"].detach().clone().to(device=device, dtype=dtype)
    reward_mask_t = bundle["reward_mask_t"].detach().clone().to(device=device, dtype=dtype)
    terminal_t = bundle["terminal_t"].detach().clone().to(device=device, dtype=dtype)
    cache = prior._detach_policy_cache(bundle["cache"], clone_tensors=True)
    step_start = int(bundle["step_idx"])
    zero_pad_t = torch.zeros((1, zero_pad_dim), device=device, dtype=dtype)

    action_noise_train_std = _to_float(env.get("action_noise_train_std", 0.0))
    action_noise_eval_std = _to_float(env.get("action_noise_eval_std", 0.0))
    state_noise_std = _to_float(env.get("state_noise_std", 0.0))
    reward_dropout_enabled = _to_bool(env.get("reward_dropout_enabled", False))
    reward_dropout_ratio = _to_float(env.get("reward_dropout_ratio", 0.0))
    reward_impute_zero = _to_bool(env.get("reward_dropout_impute_zero", True))
    terminal_reset_enabled = _to_bool(env.get("terminal_reset_enabled", False))
    terminal_reset_prob = _to_float(
        prior._terminal_reset_prob_from_count(env.get("terminal_reset_count_target", 0), int(n_steps))
    )
    terminal_bonus_tanh_c = _to_float(env.get("terminal_bonus_tanh_c", 10.0))
    terminal_bonus_scale_min = _to_float(env.get("terminal_bonus_scale_min", 1.0))
    terminal_bonus_scale_max = _to_float(env.get("terminal_bonus_scale_max", 2.0))
    fixed_initial_state = _resolve_fixed_initial_state_batch(
        prior=prior,
        env=env,
        batch_size=1,
        device=device,
        dtype=dtype,
    )
    terminal_signal_history = (
        torch.zeros((int(n_steps), 1), device=device, dtype=dtype) if terminal_reset_enabled else None
    )
    gamma = float(discount_gamma)
    running_discount = 1.0

    total_return = 0.0
    _maybe_clear_policy_buffers(policy_step_fn)
    for step_idx in range(int(step_start), int(n_steps)):
        obs_t = state_t[:, :obs_dim]
        env_info = _build_env_info(
            env=env,
            action_dim=int(action_dim),
            obs_dim=int(obs_dim),
            step_idx=int(step_idx),
            single_eval_pos=int(single_eval_pos),
            terminal_t=terminal_t,
        )
        policy_out = policy_step_fn(
            obs_t,
            action_t,
            reward_t,
            reward_mask_t,
            cache,
            int(step_idx),
            env_info,
        )
        actor_outputs, action_mean, next_cache = prior._unpack_policy_step_output(policy_out)
        sampled_action = prior._resolve_policy_action_sample(
            policy_step_fn=policy_step_fn,
            actor_outputs=actor_outputs,
            action_mean=action_mean,
            sample_action=bool(sample_action),
            action_eps_t=(
                torch.randn((1, action_dim), device=device, dtype=dtype, generator=gen)
                if bool(sample_action)
                else None
            ),
            action_transform_mode=action_transform_mode,
            action_rms_eps=action_rms_eps,
            action_clip_bound=action_clip_bound,
            collect_log_probs=False,
            collect_log_prob_score=False,
            legacy_action_std=torch.ones((1,), device=device, dtype=dtype),
        )
        action_next = sampled_action["action_next"]

        if (not reference_semantics_enabled) and action_noise_train_std > 0.0 and int(step_idx) < int(single_eval_pos):
            eps_t = torch.randn((1, action_dim), device=device, dtype=dtype, generator=gen)
            action_next = prior._transform_reinforce_action(
                action_next + eps_t * float(action_noise_train_std),
                mode=action_transform_mode,
                rms_eps=action_rms_eps,
                clip_bound=action_clip_bound,
            )
        elif (not reference_semantics_enabled) and action_noise_eval_std > 0.0:
            eps_t = torch.randn((1, action_dim), device=device, dtype=dtype, generator=gen)
            action_next = prior._transform_reinforce_action(
                action_next + eps_t * float(action_noise_eval_std),
                mode=action_transform_mode,
                rms_eps=action_rms_eps,
                clip_bound=action_clip_bound,
            )

        noise_t = torch.randn((1, int(env["noise_dim"])), device=device, dtype=dtype, generator=gen)
        env_in = prior._pack_env_input(
            carry_state_t,
            obs_t,
            action_next,
            noise_t,
            zero_pad_t,
            reference_semantics_enabled=reference_semantics_enabled,
            state_input_scale=state_input_scale,
        )
        transition_generator = prior._require_transition_generator(env)
        terminal_signal_next = None
        terminal_bonus_base_next = None
        transition_out = transition_generator(env_in, generator=gen)
        x_next, reward_unit, terminal_signal_next, terminal_bonus_base_next = prior._unpack_transition_output(
            transition_out
        )
        reward_next_raw = _to_tensor_like(env["reward_scale"], device=device, dtype=dtype).reshape(1) * reward_unit.reshape(1)

        aux_reward_next = prior._exact_scm_aux_reward_terms(
            action_next,
            ctrl_weight=env.get("ctrl_reward_weight", 0.0),
            ctrl_enabled=env.get("ctrl_reward_enabled", False),
            survival_weight=env.get("survival_reward_weight", 0.0),
            survival_enabled=env.get("survival_reward_enabled", False),
        )
        reward_next = prior._compose_exact_scm_reward(
            reward_next_raw.reshape(1),
            aux_reward=aux_reward_next.reshape(1),
            reward_clip=env.get("reward_clip", float("inf")),
            mode=env.get("reinforce_reward_transform", "none"),
            rms_eps=env.get("reinforce_reward_rms_eps", 1e-6),
            tanh_c=env.get("reinforce_reward_tanh_c", 1.0),
            tanh_bound=env.get("reinforce_reward_tanh_bound", 2.0),
        ).reshape(1)
        reward_mask_next = torch.ones((1,), device=device, dtype=dtype)
        if reward_dropout_enabled and reward_dropout_ratio > 0.0:
            drop_mask = torch.rand((1,), device=device, dtype=dtype, generator=gen) < float(reward_dropout_ratio)
            reward_mask_next = torch.where(drop_mask, torch.zeros_like(reward_mask_next), reward_mask_next)
            if bool(reward_impute_zero):
                reward_next = torch.where(drop_mask, torch.zeros_like(reward_next), reward_next)

        state_noise_t = None
        if state_noise_std > 0.0:
            state_noise_t = torch.randn((1, state_dim), device=device, dtype=dtype, generator=gen)
        state_next = prior._finalize_exact_scm_visible_state_next(
            generated_state_next=x_next,
            visible_state_prev=state_t,
            state_noise_t=state_noise_t,
            state_noise_std=env.get("state_noise_std", 0.0),
            reference_semantics_enabled=reference_semantics_enabled,
            state_clip=env["state_clip"],
            state_highway_enabled=env.get("state_highway_enabled", False),
            state_highway_lambda=env.get("state_highway_lambda", 0.0),
            state_full_rms_enabled=env.get("state_full_rms_enabled", False),
            state_full_rms_target=env.get("state_full_rms_target", 1.0),
        )
        if terminal_reset_enabled:
            terminal_draw = torch.rand((1,), device=device, dtype=dtype, generator=gen)
            bonus_scale_draw = torch.rand((1,), device=device, dtype=dtype, generator=gen)
            state_next, reward_next, terminal_next = prior._apply_terminal_reset_step(
                state_next=state_next,
                reward_next=reward_next,
                terminal_draw=terminal_draw,
                reset_prob=float(terminal_reset_prob),
                bonus_scale_draw=bonus_scale_draw,
                bonus_scale_min=float(terminal_bonus_scale_min),
                bonus_scale_max=float(terminal_bonus_scale_max),
                bonus_tanh_c=float(terminal_bonus_tanh_c),
                reset_state=fixed_initial_state,
                enabled=True,
                terminal_signal=terminal_signal_next,
                terminal_bonus_base=terminal_bonus_base_next,
                terminal_signal_history=terminal_signal_history,
                history_index=int(step_idx),
            )
            terminal_next = terminal_next.reshape(1)
        else:
            terminal_next = torch.zeros((1,), device=device, dtype=dtype)

        total_return += running_discount * float(reward_next.item())
        running_discount *= gamma
        carry_state_next = prior._mix_exact_scm_carry_state(
            carry_state_t,
            state_next,
            env["alpha"],
        )
        if terminal_reset_enabled:
            carry_state_next = prior._sync_exact_scm_carry_state_with_visible(
                carry_state_next,
                state_next,
                terminal_next,
            )
        carry_state_t = carry_state_next
        state_t = state_next
        action_t = action_next
        reward_t = reward_next
        reward_mask_t = reward_mask_next
        terminal_t = terminal_next
        cache = next_cache

    return float(total_return)


def _resolve_probe_config(
    *,
    checkpoint_path: str | None,
    from_scratch: bool,
    from_scratch_model_type: str,
    device_obj: torch.device,
    build_seed: int,
    allow_config_only: bool,
) -> dict[str, Any]:
    if bool(allow_config_only) and bool(from_scratch):
        model_type = str(from_scratch_model_type).strip().lower()
        if model_type != "rlpfn":
            raise ValueError(
                f"phase2_suffix_state_identity_probe only supports from-scratch model_type='rlpfn', got {model_type!r}."
            )
        config = copy.deepcopy(get_model_default_config(model_type))
        validate_rlpfn_maintained_path_config(config)
        return config
    _, config = _resolve_model_and_config(
        checkpoint_path=checkpoint_path,
        from_scratch=bool(from_scratch),
        from_scratch_model_type=str(from_scratch_model_type),
        device_obj=device_obj,
        build_seed=int(build_seed),
    )
    return config


def run_phase2_suffix_state_identity_probe(
    *,
    checkpoint_path: str | None = None,
    device: str | None = None,
    from_scratch: bool = False,
    from_scratch_model_type: str = "rlpfn",
    frozen_h_seed: int = 12345,
    train_env_seed: int = 2020,
    train_rollout_seed: int = 4040,
    single_eval_pos: int = 64,
    anchor_step: int | None = None,
    n_steps: int = 256,
    batch_size: int = 256,
    n_epochs: int = 1,
    learning_rate: float = 2e-4,
    target_kl: float = 0.03,
    outer_epochs: int = 0,
    build_seed: int = 4040,
    ppo_reset_env_state_at_sep: bool = True,
    strict_native_rollout: bool = False,
    value_head_impl: str = "scalar_mlp",
    value_path_adapter_impl: str = "none",
    value_head_mlp_hidden_dim: int = 256,
    space_contract: str = "normalized",
    actor_baseline_mode: str = "learned",
    vf_coef_override: float | None = None,
    policy_loss_coef_override: float | None = None,
    behavior_policy: str = "learned",
    reference_state_count: int = 32,
    continuation_repeats: int = 64,
    reference_rollout_seed_start: int = 5000,
    continuation_rollout_seed_start: int = 10000,
    sampled_policy: bool = False,
    discount_gamma: float = 0.99,
    constrained_dim_sampling_total_budget_override: int | None = None,
    noise_dim_override: int | None = None,
    state_dim_override: int | None = None,
    action_dim_override: int | None = None,
    zero_pad_dim_override: int | None = None,
    alpha_override: float | None = None,
    reference_state_inertia_enabled_override: bool | None = None,
    terminal_reset_enabled_override: bool | None = None,
    terminal_reset_count_target_override: float | None = None,
    frozen_h_overrides: dict[str, Any] | None = None,
    fixed_frozen_h_json: str | None = None,
    save_full_frozen_h_json: str | None = None,
    progress_jsonl: str | None = None,
    print_progress_every_repeat: bool = True,
    horizon_diagnostics_steps: Any = None,
) -> dict[str, Any]:
    # Match CLI semantics for direct Python callers as well: exact-SCM generator
    # construction still has legacy paths that consume torch's global RNG.
    _seed_all(int(build_seed))
    if int(n_steps) <= int(single_eval_pos):
        raise ValueError(
            f"n_steps must exceed single_eval_pos for suffix states to exist, got n_steps={int(n_steps)} "
            f"and single_eval_pos={int(single_eval_pos)}."
        )
    anchor_step_resolved = int(single_eval_pos if anchor_step is None else anchor_step)
    if anchor_step_resolved < int(single_eval_pos) or anchor_step_resolved >= int(n_steps):
        raise ValueError(
            f"anchor_step must lie in suffix range [{int(single_eval_pos)}, {int(n_steps) - 1}], "
            f"got {int(anchor_step_resolved)}."
        )
    behavior_policy_resolved = str(behavior_policy).strip().lower()
    if behavior_policy_resolved not in {"learned", "unit_gaussian"}:
        raise ValueError(
            f"Unsupported behavior_policy={behavior_policy!r}. Expected 'learned' or 'unit_gaussian'."
        )
    if noise_dim_override is not None and int(noise_dim_override) < 0:
        raise ValueError(f"noise_dim_override must be >= 0, got {int(noise_dim_override)}")
    if state_dim_override is not None and int(state_dim_override) <= 0:
        raise ValueError(f"state_dim_override must be > 0, got {int(state_dim_override)}")
    if action_dim_override is not None and int(action_dim_override) <= 0:
        raise ValueError(f"action_dim_override must be > 0, got {int(action_dim_override)}")
    if (
        constrained_dim_sampling_total_budget_override is not None
        and int(constrained_dim_sampling_total_budget_override) <= 0
    ):
        raise ValueError(
            "constrained_dim_sampling_total_budget_override must be > 0, "
            f"got {int(constrained_dim_sampling_total_budget_override)}"
        )
    if zero_pad_dim_override is not None and int(zero_pad_dim_override) < 0:
        raise ValueError(f"zero_pad_dim_override must be >= 0, got {int(zero_pad_dim_override)}")
    if alpha_override is not None and (not np.isfinite(float(alpha_override))):
        raise ValueError(f"alpha_override must be finite, got {alpha_override!r}")
    if (
        terminal_reset_count_target_override is not None
        and float(terminal_reset_count_target_override) < 0.0
        ):
        raise ValueError(
            "terminal_reset_count_target_override must be >= 0, "
            f"got {float(terminal_reset_count_target_override)}"
        )
    gamma_resolved = _validate_discount_gamma(float(discount_gamma))
    horizon_diagnostics_steps_resolved = _resolve_horizon_diagnostics_steps(
        horizon_diagnostics_steps,
        remaining_horizon=int(n_steps) - int(anchor_step_resolved),
    )

    device_obj = torch.device(str(device or _default_device()))
    config = _resolve_probe_config(
        checkpoint_path=checkpoint_path,
        from_scratch=bool(from_scratch),
        from_scratch_model_type=str(from_scratch_model_type),
        device_obj=device_obj,
        build_seed=int(build_seed),
        allow_config_only=behavior_policy_resolved == "unit_gaussian",
    )
    env_cfg = _build_audit_env_cfg(config["prior"]["environment"])
    num_features = int(config["prior"]["num_features"])
    loaded_frozen_h_path = None
    if fixed_frozen_h_json is None:
        fixed_bundle = _prepare_audit_fixed_env_contract(
            env_cfg=env_cfg,
            boundary_contract_mode="normal",
            ppo_reset_env_state_at_sep=bool(ppo_reset_env_state_at_sep),
            frozen_h_seed=int(frozen_h_seed),
            frozen_h_seed_mode="current",
        )
        frozen_h = copy.deepcopy(fixed_bundle["frozen_h"])
    else:
        frozen_h, loaded_frozen_h_path = _load_full_frozen_h(str(fixed_frozen_h_json))
        fixed_bundle = {
            "prior_for_h": None,
            "frozen_h": copy.deepcopy(frozen_h),
            "fixed_env_contract": {
                "frozen_h_seed": int(frozen_h_seed),
                "frozen_h_seed_mode": "external_json",
                "zero_eval_prior_reuses_sampling_state": False,
                "source_full_frozen_h_json": str(loaded_frozen_h_path),
                **summarize_fixed_env_h(frozen_h),
            },
        }
    if frozen_h_overrides is not None:
        if not isinstance(frozen_h_overrides, dict):
            raise TypeError(
                "frozen_h_overrides must be a dict[str, Any] when provided, "
                f"got {type(frozen_h_overrides).__name__}"
            )
        frozen_h.update(copy.deepcopy(frozen_h_overrides))
    if constrained_dim_sampling_total_budget_override is not None:
        frozen_h["constrained_dim_sampling_total_budget"] = int(constrained_dim_sampling_total_budget_override)
    if state_dim_override is not None:
        frozen_h["state_dim"] = int(state_dim_override)
    if action_dim_override is not None:
        frozen_h["action_dim"] = int(action_dim_override)
    if noise_dim_override is not None:
        frozen_h["noise_dim"] = int(noise_dim_override)
    if zero_pad_dim_override is not None:
        frozen_h["zero_pad_dim"] = int(zero_pad_dim_override)
    if alpha_override is not None:
        frozen_h["alpha"] = float(alpha_override)
    if reference_state_inertia_enabled_override is not None:
        frozen_h["reference_state_inertia_enabled"] = bool(reference_state_inertia_enabled_override)
    if terminal_reset_enabled_override is not None:
        frozen_h["terminal_reset_enabled"] = bool(terminal_reset_enabled_override)
    if terminal_reset_count_target_override is not None:
        frozen_h["terminal_reset_count_target"] = float(terminal_reset_count_target_override)
    full_frozen_h_path = None if save_full_frozen_h_json is None else Path(save_full_frozen_h_json).expanduser().resolve()
    if full_frozen_h_path is not None:
        full_frozen_h_path.parent.mkdir(parents=True, exist_ok=True)
        full_frozen_h_path.write_text(
            json.dumps(_json_safe_value(copy.deepcopy(frozen_h)), sort_keys=True) + "\n",
            encoding="utf-8",
        )

    algo = None
    callback = None
    vec_env = None
    progress_path = None if progress_jsonl is None else Path(progress_jsonl).expanduser().resolve()

    try:
        if behavior_policy_resolved == "learned":
            algo, callback, vec_env = _build_algo(
                checkpoint_path=checkpoint_path,
                device_obj=device_obj,
                frozen_h=frozen_h,
                env_cfg=env_cfg,
                num_features=int(num_features),
                train_env_seed=int(train_env_seed),
                train_rollout_seed=int(train_rollout_seed),
                single_eval_pos=int(single_eval_pos),
                n_steps=int(n_steps),
                batch_size=int(batch_size),
                n_epochs=int(n_epochs),
                learning_rate=float(learning_rate),
                target_kl=float(target_kl),
                ppo_reset_env_state_at_sep=bool(ppo_reset_env_state_at_sep),
                actor_baseline_mode=str(actor_baseline_mode),
                build_seed=int(build_seed),
                strict_native_rollout=bool(strict_native_rollout),
                value_head_impl=str(value_head_impl),
                value_path_adapter_impl=str(value_path_adapter_impl),
                value_head_mlp_hidden_dim=int(value_head_mlp_hidden_dim),
                space_contract=str(space_contract),
                from_scratch=bool(from_scratch),
                from_scratch_model_type=str(from_scratch_model_type),
                vf_coef_override=vf_coef_override,
                policy_loss_coef_override=policy_loss_coef_override,
                behavior_policy="learned",
            )
            total_timesteps = max(1, int(n_steps) * max(1, int(outer_epochs)))
            if int(outer_epochs) > 0:
                _collect_rollout(algo, callback, vec_env, progress_timestep=0, total_timesteps=total_timesteps)
                for outer_idx in range(int(outer_epochs)):
                    algo.train()
                    if int(outer_idx + 1) < int(outer_epochs):
                        progress_timestep = int((outer_idx + 1) * int(n_steps))
                        _collect_rollout(
                            algo,
                            callback,
                            vec_env,
                            progress_timestep=progress_timestep,
                            total_timesteps=total_timesteps,
                        )
            algo.policy.set_training_mode(False)
            policy_step_fn = _build_audit_ppo_policy_step_fn(
                algo.policy,
                sampled=bool(sampled_policy),
                single_eval_pos=int(single_eval_pos),
                boundary_contract_mode="normal",
                deterministic_actor_sampling=True,
            )
            prior = algo._rwkv_env_prior
            space_contract_runtime = getattr(algo, "_rwkv_space_contract", None)
        else:
            prior = EnvironmentPrior(copy.deepcopy(env_cfg))
            policy_step_fn = _build_fixed_unit_gaussian_policy_step_fn(sampled=bool(sampled_policy))
            space_contract_runtime = None
        env = prior._sample_environment(copy.deepcopy(frozen_h), device=device_obj, rng_seed=int(train_env_seed))

        anchors: list[dict[str, Any]] = []
        return_matrix = np.zeros((int(reference_state_count), int(continuation_repeats)), dtype=np.float32)
        discounted_step_rewards_flat = (
            np.zeros(
                (
                    int(reference_state_count) * int(continuation_repeats),
                    int(n_steps) - int(anchor_step_resolved),
                ),
                dtype=np.float32,
            )
            if len(horizon_diagnostics_steps_resolved) > 0
            else None
        )
        running_returns: list[list[float]] = [[] for _ in range(int(reference_state_count))]
        reference_seeds = [int(reference_rollout_seed_start) + int(i) for i in range(int(reference_state_count))]
        continuation_seeds = [
            int(continuation_rollout_seed_start) + int(state_idx) * int(continuation_repeats) + int(repeat_idx)
            for state_idx in range(int(reference_state_count))
            for repeat_idx in range(int(continuation_repeats))
        ]

        use_batched_unit_gaussian = bool(
            behavior_policy_resolved == "unit_gaussian"
            and int(reference_state_count) > 0
            and int(continuation_repeats) > 0
        )
        if len(horizon_diagnostics_steps_resolved) > 0 and not bool(use_batched_unit_gaussian):
            raise NotImplementedError("horizon diagnostics are currently implemented for unit_gaussian batched probes")
        if use_batched_unit_gaussian:
            reference_bundle_batch = _capture_suffix_bundles_unit_gaussian_batch(
                prior=prior,
                env=env,
                policy_step_fn=policy_step_fn,
                sampled_policy=bool(sampled_policy),
                anchor_step=int(anchor_step_resolved),
                n_steps=int(n_steps),
                single_eval_pos=int(single_eval_pos),
                device=device_obj,
                rng_seeds=reference_seeds,
            )
            repeated_bundle_batch = _repeat_suffix_bundle_batch(
                prior,
                reference_bundle_batch,
                int(continuation_repeats),
            )
            continuation_returns_flat = np.zeros(
                (int(reference_state_count) * int(continuation_repeats),),
                dtype=np.float32,
            )
            parallel_rollout_batch = max(1, int(batch_size))
            for start_idx in range(0, int(continuation_returns_flat.shape[0]), int(parallel_rollout_batch)):
                end_idx = min(
                    int(continuation_returns_flat.shape[0]),
                    int(start_idx) + int(parallel_rollout_batch),
                )
                chunk_bundle = _slice_suffix_bundle_batch(prior, repeated_bundle_batch, int(start_idx), int(end_idx))
                chunk_result = _rollout_suffix_returns_from_bundle_batch_unit_gaussian(
                    prior=prior,
                    env=env,
                    policy_step_fn=policy_step_fn,
                    sampled_policy=bool(sampled_policy),
                    bundle=chunk_bundle,
                    n_steps=int(n_steps),
                    single_eval_pos=int(single_eval_pos),
                    discount_gamma=float(gamma_resolved),
                    device=device_obj,
                    rng_seeds=continuation_seeds[int(start_idx): int(end_idx)],
                    return_discounted_step_rewards=discounted_step_rewards_flat is not None,
                )
                if discounted_step_rewards_flat is None:
                    chunk_returns = chunk_result
                else:
                    chunk_returns, chunk_discounted_step_rewards = chunk_result
                    discounted_step_rewards_flat[int(start_idx): int(end_idx), :] = chunk_discounted_step_rewards
                continuation_returns_flat[int(start_idx): int(end_idx)] = chunk_returns
                for flat_idx in range(int(start_idx), int(end_idx)):
                    state_idx = int(flat_idx) // int(continuation_repeats)
                    repeat_idx = int(flat_idx) % int(continuation_repeats)
                    reference_seed = int(reference_seeds[int(state_idx)])
                    continuation_seed = int(continuation_seeds[int(flat_idx)])
                    continuation_return = float(continuation_returns_flat[int(flat_idx)])
                    running_returns[int(state_idx)].append(continuation_return)
                    return_matrix[int(state_idx), int(repeat_idx)] = continuation_return
                    partial_metrics = _compute_identity_metrics_from_return_lists(running_returns)
                    _append_jsonl(
                        progress_path,
                        {
                            "event": "repeat_complete",
                            "state_idx": int(state_idx),
                            "state_count_total": int(reference_state_count),
                            "repeat_idx": int(repeat_idx),
                            "repeat_count_total": int(continuation_repeats),
                            "completed_states": int(sum(1 for values in running_returns if len(values) > 0)),
                            "completed_samples": int(sum(len(values) for values in running_returns)),
                            "reference_rollout_seed": int(reference_seed),
                            "continuation_seed": int(continuation_seed),
                            "current_return": float(continuation_return),
                            "partial_metrics": partial_metrics,
                        },
                    )
                    if bool(print_progress_every_repeat):
                        print(
                            _format_progress_line(
                                state_idx=int(state_idx),
                                total_states=int(reference_state_count),
                                repeat_idx=int(repeat_idx),
                                total_repeats=int(continuation_repeats),
                                partial_metrics=partial_metrics,
                                current_return=float(continuation_return),
                            ),
                            flush=True,
                        )
        else:
            for state_idx in range(int(reference_state_count)):
                reference_seed = int(reference_seeds[int(state_idx)])
                bundle = _capture_suffix_bundle(
                    prior=prior,
                    env=env,
                    policy_step_fn=policy_step_fn,
                    sample_action=bool(sampled_policy),
                    anchor_step=int(anchor_step_resolved),
                    n_steps=int(n_steps),
                    single_eval_pos=int(single_eval_pos),
                    device=device_obj,
                    rng_seed=int(reference_seed),
                )
                for repeat_idx in range(int(continuation_repeats)):
                    continuation_seed = int(
                        continuation_seeds[int(state_idx) * int(continuation_repeats) + int(repeat_idx)]
                    )
                    continuation_return = _rollout_suffix_return_from_bundle(
                        prior=prior,
                        env=env,
                        policy_step_fn=policy_step_fn,
                        sample_action=bool(sampled_policy),
                        bundle=_clone_suffix_bundle(prior, bundle),
                        n_steps=int(n_steps),
                        single_eval_pos=int(single_eval_pos),
                        discount_gamma=float(gamma_resolved),
                        device=device_obj,
                        rng_seed=int(continuation_seed),
                    )
                    running_returns[int(state_idx)].append(float(continuation_return))
                    return_matrix[int(state_idx), int(repeat_idx)] = float(continuation_return)
                    partial_metrics = _compute_identity_metrics_from_return_lists(running_returns)
                    _append_jsonl(
                        progress_path,
                        {
                            "event": "repeat_complete",
                            "state_idx": int(state_idx),
                            "state_count_total": int(reference_state_count),
                            "repeat_idx": int(repeat_idx),
                            "repeat_count_total": int(continuation_repeats),
                            "completed_states": int(sum(1 for values in running_returns if len(values) > 0)),
                            "completed_samples": int(sum(len(values) for values in running_returns)),
                            "reference_rollout_seed": int(reference_seed),
                            "continuation_seed": int(continuation_seed),
                            "current_return": float(continuation_return),
                            "partial_metrics": partial_metrics,
                        },
                    )
                    if bool(print_progress_every_repeat):
                        print(
                            _format_progress_line(
                                state_idx=int(state_idx),
                                total_states=int(reference_state_count),
                                repeat_idx=int(repeat_idx),
                                total_repeats=int(continuation_repeats),
                                partial_metrics=partial_metrics,
                                current_return=float(continuation_return),
                            ),
                            flush=True,
                        )

        for state_idx in range(int(reference_state_count)):
            reference_seed = int(reference_seeds[int(state_idx)])
            state_returns = [float(v) for v in return_matrix[int(state_idx)].tolist()]
            completed_return_lists = running_returns[: int(state_idx) + 1]
            anchors.append(
                {
                    "anchor_index": int(state_idx),
                    "reference_rollout_seed": int(reference_seed),
                    "continuation_return_mean": float(np.mean(state_returns, dtype=np.float64)),
                    "continuation_return_std": float(np.std(state_returns, dtype=np.float64)),
                    "continuation_return_min": float(np.min(state_returns)),
                    "continuation_return_max": float(np.max(state_returns)),
                    "continuation_returns": state_returns,
                }
            )
            state_metrics = _compute_identity_metrics_from_return_lists(completed_return_lists)
            _append_jsonl(
                progress_path,
                {
                    "event": "state_complete",
                    "state_idx": int(state_idx),
                    "state_count_total": int(reference_state_count),
                    "completed_states": int(state_idx) + 1,
                    "completed_samples": int(sum(len(values) for values in completed_return_lists)),
                    "reference_rollout_seed": int(reference_seed),
                    "state_return_mean": float(np.mean(state_returns, dtype=np.float64)),
                    "state_return_std": float(np.std(state_returns, dtype=np.float64)),
                    "partial_metrics": state_metrics,
                },
            )
            print(
                (
                    f"[identity-probe] completed state {int(state_idx) + 1}/{int(reference_state_count)} "
                    f"state_mean={float(np.mean(state_returns, dtype=np.float64)):+.4f} "
                    f"state_std={float(np.std(state_returns, dtype=np.float64)):.4f} "
                    f"noise={float(state_metrics['noise']):.4f} "
                    f"signal={float(state_metrics['signal']):.4f} "
                    f"identity={float(state_metrics['identity_score']):.4f}"
                ),
                flush=True,
            )

        aggregate = _compute_identity_metrics(return_matrix)
        horizon_diagnostics = None
        if discounted_step_rewards_flat is not None:
            discounted_step_rewards = discounted_step_rewards_flat.reshape(
                int(reference_state_count),
                int(continuation_repeats),
                int(n_steps) - int(anchor_step_resolved),
            )
            horizon_diagnostics = _compute_horizon_diagnostics(
                discounted_step_rewards,
                horizon_steps=horizon_diagnostics_steps_resolved,
                continuation_repeats=int(continuation_repeats),
            )
        report = {
            "audit_entry": "phase2_suffix_state_identity_probe",
            "init_mode": "from_scratch" if bool(from_scratch) else "checkpoint",
            "from_scratch_model_type": str(from_scratch_model_type),
            "checkpoint_path": (
                None if checkpoint_path is None else str(Path(checkpoint_path).expanduser().resolve())
            ),
            "device": str(device_obj),
            "frozen_h_seed": int(frozen_h_seed),
            "fixed_env_contract": dict(fixed_bundle["fixed_env_contract"]),
            "train_env_seed": int(train_env_seed),
            "train_rollout_seed": int(train_rollout_seed),
            "single_eval_pos": int(single_eval_pos),
            "anchor_step": int(anchor_step_resolved),
            "remaining_horizon": int(n_steps - anchor_step_resolved),
            "n_steps": int(n_steps),
            "batch_size": int(batch_size),
            "n_epochs": int(n_epochs),
            "learning_rate": float(learning_rate),
            "target_kl": float(target_kl),
            "outer_epochs": int(outer_epochs),
            "build_seed": int(build_seed),
            "ppo_reset_env_state_at_sep": bool(ppo_reset_env_state_at_sep),
            "strict_native_rollout": bool(strict_native_rollout),
            "value_head_impl": str(value_head_impl),
            "value_path_adapter_impl": str(value_path_adapter_impl),
            "value_head_mlp_hidden_dim": int(value_head_mlp_hidden_dim),
            "space_contract": str(space_contract),
            "actor_baseline_mode": str(actor_baseline_mode),
            "behavior_policy": str(behavior_policy_resolved),
            "vf_coef_override": None if vf_coef_override is None else float(vf_coef_override),
            "policy_loss_coef_override": (
                None if policy_loss_coef_override is None else float(policy_loss_coef_override)
            ),
            "sampled_policy": bool(sampled_policy),
            "constrained_dim_sampling_total_budget_override": (
                None
                if constrained_dim_sampling_total_budget_override is None
                else int(constrained_dim_sampling_total_budget_override)
            ),
            "state_dim_override": None if state_dim_override is None else int(state_dim_override),
            "action_dim_override": None if action_dim_override is None else int(action_dim_override),
            "noise_dim_override": None if noise_dim_override is None else int(noise_dim_override),
            "zero_pad_dim_override": None if zero_pad_dim_override is None else int(zero_pad_dim_override),
            "alpha_override": None if alpha_override is None else float(alpha_override),
            "reference_state_inertia_enabled_override": (
                None
                if reference_state_inertia_enabled_override is None
                else bool(reference_state_inertia_enabled_override)
            ),
            "terminal_reset_enabled_override": (
                None
                if terminal_reset_enabled_override is None
                else bool(terminal_reset_enabled_override)
            ),
            "terminal_reset_count_target_override": (
                None
                if terminal_reset_count_target_override is None
                else float(terminal_reset_count_target_override)
            ),
            "frozen_h_overrides": (
                None if frozen_h_overrides is None else _json_safe_value(copy.deepcopy(frozen_h_overrides))
            ),
            "fixed_frozen_h_json": None if loaded_frozen_h_path is None else str(loaded_frozen_h_path),
            "save_full_frozen_h_json": None if full_frozen_h_path is None else str(full_frozen_h_path),
            "effective_frozen_h_snapshot": _select_env_knob_snapshot(frozen_h),
            "sampled_env_snapshot": _select_sampled_env_snapshot(env, frozen_h),
            "batched_unit_gaussian_rollout": bool(use_batched_unit_gaussian),
            "parallel_rollout_batch_size": int(batch_size) if bool(use_batched_unit_gaussian) else None,
            "reference_state_count": int(reference_state_count),
            "continuation_repeats": int(continuation_repeats),
            "reference_rollout_seed_start": int(reference_rollout_seed_start),
            "continuation_rollout_seed_start": int(continuation_rollout_seed_start),
            "discount_gamma": float(gamma_resolved),
            "horizon_diagnostics_steps": horizon_diagnostics_steps_resolved,
            "horizon_diagnostics": horizon_diagnostics,
            "progress_jsonl": None if progress_path is None else str(progress_path),
            "aggregate": aggregate,
            "anchors": anchors,
            "space_contract_runtime": space_contract_runtime,
        }
        _append_jsonl(
            progress_path,
            {
                "event": "run_complete",
                "completed_states": int(reference_state_count),
                "completed_samples": int(reference_state_count) * int(continuation_repeats),
                "aggregate": aggregate,
            },
        )
        return report
    finally:
        if vec_env is not None:
            vec_env.close()
        del algo, callback, vec_env
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure conditional suffix-return variance from repeated continuations of matched suffix bundles."
    )
    parser.add_argument("checkpoint_path", nargs="?", default=None, type=str)
    parser.add_argument("--from-scratch", type=str2bool, default=False)
    parser.add_argument("--from-scratch-model-type", type=str, default="rlpfn")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--frozen-h-seed", type=int, default=12345)
    parser.add_argument("--train-env-seed", type=int, default=2020)
    parser.add_argument("--train-rollout-seed", type=int, default=4040)
    parser.add_argument("--single-eval-pos", type=int, default=64)
    parser.add_argument("--anchor-step", type=int, default=None)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument("--outer-epochs", type=int, default=0)
    parser.add_argument("--build-seed", type=int, default=4040)
    parser.add_argument("--ppo-reset-env-state-at-sep", type=str2bool, default=True)
    parser.add_argument("--strict-native-rollout", type=str2bool, default=False)
    parser.add_argument("--value-head-impl", type=str, default="scalar_mlp")
    parser.add_argument("--value-path-adapter-impl", type=str, default="none")
    parser.add_argument("--value-head-mlp-hidden-dim", type=int, default=256)
    parser.add_argument("--space-contract", type=str, default="normalized")
    parser.add_argument("--actor-baseline-mode", type=str, default="learned")
    parser.add_argument("--vf-coef-override", type=float, default=None)
    parser.add_argument("--policy-loss-coef-override", type=float, default=None)
    parser.add_argument("--behavior-policy", type=str, default="learned")
    parser.add_argument("--reference-state-count", type=int, default=32)
    parser.add_argument("--continuation-repeats", type=int, default=64)
    parser.add_argument("--reference-rollout-seed-start", type=int, default=5000)
    parser.add_argument("--continuation-rollout-seed-start", type=int, default=10000)
    parser.add_argument("--sampled-policy", type=str2bool, default=False)
    parser.add_argument("--discount-gamma", type=float, default=0.99)
    parser.add_argument("--constrained-dim-sampling-total-budget-override", type=int, default=None)
    parser.add_argument("--state-dim-override", type=int, default=None)
    parser.add_argument("--action-dim-override", type=int, default=None)
    parser.add_argument("--noise-dim-override", type=int, default=None)
    parser.add_argument("--zero-pad-dim-override", type=int, default=None)
    parser.add_argument("--alpha-override", type=float, default=None)
    parser.add_argument("--reference-state-inertia-enabled-override", type=str2bool, default=None)
    parser.add_argument("--terminal-reset-enabled-override", type=str2bool, default=None)
    parser.add_argument("--terminal-reset-count-target-override", type=float, default=None)
    parser.add_argument("--fixed-frozen-h-json", type=str, default=None)
    parser.add_argument("--save-full-frozen-h-json", type=str, default=None)
    parser.add_argument("--progress-jsonl", type=str, default=None)
    parser.add_argument("--print-progress-every-repeat", type=str2bool, default=True)
    parser.add_argument(
        "--horizon-diagnostics-steps",
        type=str,
        default=None,
        help=(
            "Comma-separated suffix horizons k for G_k diagnostics, or 'powers' for 1,2,4,...,remaining. "
            "G_k = sum_{i=0}^{k-1} discount_gamma^i * r_{anchor+i+1}."
        ),
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="/home/chen/RLPFN/artifacts/phase2_suffix_state_identity_probe.json",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    _seed_all(int(args.build_seed))
    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = run_phase2_suffix_state_identity_probe(
        checkpoint_path=args.checkpoint_path,
        device=args.device,
        from_scratch=bool(args.from_scratch),
        from_scratch_model_type=str(args.from_scratch_model_type),
        frozen_h_seed=int(args.frozen_h_seed),
        train_env_seed=int(args.train_env_seed),
        train_rollout_seed=int(args.train_rollout_seed),
        single_eval_pos=int(args.single_eval_pos),
        anchor_step=args.anchor_step,
        n_steps=int(args.n_steps),
        batch_size=int(args.batch_size),
        n_epochs=int(args.n_epochs),
        learning_rate=float(args.learning_rate),
        target_kl=float(args.target_kl),
        outer_epochs=int(args.outer_epochs),
        build_seed=int(args.build_seed),
        ppo_reset_env_state_at_sep=bool(args.ppo_reset_env_state_at_sep),
        strict_native_rollout=bool(args.strict_native_rollout),
        value_head_impl=str(args.value_head_impl),
        value_path_adapter_impl=str(args.value_path_adapter_impl),
        value_head_mlp_hidden_dim=int(args.value_head_mlp_hidden_dim),
        space_contract=str(args.space_contract),
        actor_baseline_mode=str(args.actor_baseline_mode),
        vf_coef_override=args.vf_coef_override,
        policy_loss_coef_override=args.policy_loss_coef_override,
        behavior_policy=str(args.behavior_policy),
        reference_state_count=int(args.reference_state_count),
        continuation_repeats=int(args.continuation_repeats),
        reference_rollout_seed_start=int(args.reference_rollout_seed_start),
        continuation_rollout_seed_start=int(args.continuation_rollout_seed_start),
        sampled_policy=bool(args.sampled_policy),
        discount_gamma=float(args.discount_gamma),
        constrained_dim_sampling_total_budget_override=args.constrained_dim_sampling_total_budget_override,
        state_dim_override=args.state_dim_override,
        action_dim_override=args.action_dim_override,
        noise_dim_override=args.noise_dim_override,
        zero_pad_dim_override=args.zero_pad_dim_override,
        alpha_override=args.alpha_override,
        reference_state_inertia_enabled_override=args.reference_state_inertia_enabled_override,
        terminal_reset_enabled_override=args.terminal_reset_enabled_override,
        terminal_reset_count_target_override=args.terminal_reset_count_target_override,
        fixed_frozen_h_json=args.fixed_frozen_h_json,
        save_full_frozen_h_json=args.save_full_frozen_h_json,
        progress_jsonl=args.progress_jsonl,
        print_progress_every_repeat=bool(args.print_progress_every_repeat),
        horizon_diagnostics_steps=args.horizon_diagnostics_steps,
    )
    payload = json.dumps(report, sort_keys=True)
    output_path.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
