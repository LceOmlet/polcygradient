import argparse
import copy
import json
from collections import deque
from pathlib import Path
from typing import Any

import torch

from ticl.analysis.critic_free_single_env_audit import (
    _build_audit_env_cfg,
    _prepare_audit_fixed_env_contract,
    _seed_all,
)
from ticl.analysis.phase2_suffix_state_identity_probe import (
    _broadcast_like_batch,
    _call_generator_with_noise_batch,
    _load_full_frozen_h,
    _resolve_probe_config,
    _to_bool,
    _to_float,
)
from ticl.analysis.fixed_env_h import summarize_fixed_env_h
from ticl.config_utils import str2bool
from ticl.priors.environment_prior import EnvironmentPrior


def _default_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _parse_alpha_list(raw: str) -> list[float]:
    values = []
    for piece in str(raw).split(","):
        token = piece.strip()
        if not token:
            continue
        values.append(float(token))
    if not values:
        raise ValueError("alpha_list must contain at least one float.")
    return values


def _format_float(value: float) -> str:
    text = f"{float(value):.6f}"
    text = text.rstrip("0").rstrip(".")
    return text if text else "0"


def _make_generators_from_seeds(prior: EnvironmentPrior, seeds: list[int], device: torch.device):
    return prior._make_generators_from_seeds([int(v) for v in seeds], int(len(seeds)), device)


def _tensor_stats_1d(values: torch.Tensor) -> dict[str, float | int | None]:
    flat = values.detach().reshape(-1).to(dtype=torch.float64, device="cpu")
    finite = flat[torch.isfinite(flat)]
    if int(finite.numel()) == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "q10": None,
            "q25": None,
            "q50": None,
            "q75": None,
            "q90": None,
            "max": None,
        }
    return {
        "count": int(finite.numel()),
        "mean": float(torch.mean(finite).item()),
        "std": float(torch.std(finite, unbiased=False).item()),
        "min": float(torch.min(finite).item()),
        "q10": float(torch.quantile(finite, 0.10).item()),
        "q25": float(torch.quantile(finite, 0.25).item()),
        "q50": float(torch.quantile(finite, 0.50).item()),
        "q75": float(torch.quantile(finite, 0.75).item()),
        "q90": float(torch.quantile(finite, 0.90).item()),
        "max": float(torch.max(finite).item()),
    }


def _compute_preterminal_state_next_batch(
    *,
    prior: EnvironmentPrior,
    env: dict[str, Any],
    carry_state_t: torch.Tensor,
    state_t: torch.Tensor,
    action_next: torch.Tensor,
    noise_t: torch.Tensor,
    state_noise_t: torch.Tensor | None,
    rollout_generators,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    batch_size = int(state_t.shape[0])
    dtype = state_t.dtype
    obs_dim = int(env["obs_dim"])
    zero_pad_dim = int(env["zero_pad_dim"])
    reference_semantics_enabled = bool(prior._env_uses_reference_semantics(env))
    state_input_scale = env.get("state_input_scale", 1.0)
    zero_pad_t = torch.zeros((batch_size, zero_pad_dim), device=device, dtype=dtype)
    obs_t = state_t[:, :obs_dim]
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
    reward_scale = _broadcast_like_batch(
        env["reward_scale"],
        batch_size=batch_size,
        device=device,
        dtype=dtype,
    )
    transition_out = _call_generator_with_noise_batch(transition_generator, env_in, rollout_generators)
    x_next, reward_unit, terminal_signal_next, terminal_bonus_base_next = prior._unpack_transition_output(
        transition_out
    )
    reward_next_raw = reward_scale * reward_unit.reshape(batch_size)

    state_next = prior._finalize_exact_scm_visible_state_next(
        generated_state_next=x_next,
        visible_state_prev=state_t,
        state_noise_t=state_noise_t,
        state_noise_std=env.get("state_noise_std", 0.0),
        reference_semantics_enabled=reference_semantics_enabled,
        state_clip=env["state_clip"],
        state_output_scale=env.get("state_output_scale", 1.0),
        state_highway_enabled=env.get("state_highway_enabled", False),
        state_highway_lambda=env.get("state_highway_lambda", 0.0),
        state_full_rms_enabled=env.get("state_full_rms_enabled", False),
        state_full_rms_target=env.get("state_full_rms_target", 1.0),
    )
    return state_next, reward_next_raw, terminal_signal_next, terminal_bonus_base_next


def _apply_terminal_reset_actual_state(
    *,
    prior: EnvironmentPrior,
    env: dict[str, Any],
    state_next_preterminal: torch.Tensor,
    reward_next: torch.Tensor,
    terminal_signal_next: torch.Tensor | None,
    terminal_bonus_base_next: torch.Tensor | None,
    terminal_signal_history: torch.Tensor | None,
    history_index: int,
    rollout_generators,
    fixed_initial_state_batch: torch.Tensor,
    n_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = int(state_next_preterminal.shape[0])
    device = state_next_preterminal.device
    dtype = state_next_preterminal.dtype
    terminal_reset_enabled = bool(env.get("terminal_reset_enabled", False))
    if not terminal_reset_enabled:
        return state_next_preterminal, torch.zeros((batch_size,), device=device, dtype=dtype)

    terminal_draw = prior._stack_rand_with_generators(rollout_generators, (batch_size,), device, dtype)
    bonus_scale_draw = prior._stack_rand_with_generators(rollout_generators, (batch_size,), device, dtype)
    reset_prob = _broadcast_like_batch(
        float(prior._terminal_reset_prob_from_count(env.get("terminal_reset_count_target", 0), int(n_steps))),
        batch_size=batch_size,
        device=device,
        dtype=dtype,
    )
    reset_state = fixed_initial_state_batch
    state_next, reward_next_out, terminal_next = prior._apply_terminal_reset_step(
        state_next=state_next_preterminal,
        reward_next=reward_next,
        terminal_draw=terminal_draw,
        reset_prob=reset_prob,
        bonus_scale_draw=bonus_scale_draw,
        bonus_scale_min=float(env.get("terminal_bonus_scale_min", 1.0)),
        bonus_scale_max=float(env.get("terminal_bonus_scale_max", 2.0)),
        bonus_tanh_c=float(env.get("terminal_bonus_tanh_c", 10.0)),
        reset_state=reset_state,
        enabled=torch.ones((batch_size,), device=device, dtype=torch.bool),
        terminal_signal=terminal_signal_next,
        terminal_bonus_base=terminal_bonus_base_next,
        terminal_signal_history=terminal_signal_history,
        history_index=int(history_index),
    )
    del reward_next_out
    return state_next, terminal_next.reshape(batch_size).to(dtype=dtype)


def measure_alpha_temporal_control(
    *,
    checkpoint_path: str | None = None,
    device: str | None = None,
    from_scratch: bool = True,
    from_scratch_model_type: str = "rlpfn",
    frozen_h_seed: int = 12345,
    train_env_seed: int = 2020,
    single_eval_pos: int = 64,
    n_steps: int = 256,
    build_seed: int = 4040,
    rollout_count: int = 4096,
    rollout_seed_start: int = 5000,
    sampled_policy: bool = True,
    constrained_dim_sampling_total_budget_override: int | None = None,
    state_dim_override: int | None = None,
    action_dim_override: int | None = None,
    noise_dim_override: int | None = None,
    zero_pad_dim_override: int | None = None,
    fixed_frozen_h_json: str | None = None,
    prior_mlp_activations_override: str | None = None,
    init_std_override: float | None = None,
    noise_std_override: float | None = None,
    scm_standard_linear_init_enabled_override: bool | None = None,
    alpha: float = 1.0,
    reference_state_inertia_enabled_override: bool | None = True,
    terminal_reset_enabled_override: bool | None = None,
    terminal_reset_count_target_override: float | None = None,
    action_delta: float = 0.1,
    behavior_policy: str | None = None,
    discount_gamma: float = 0.99,
    frozen_h_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if int(rollout_count) <= 0:
        raise ValueError(f"rollout_count must be > 0, got {int(rollout_count)}")
    if int(n_steps) <= 4:
        raise ValueError(f"n_steps must be > 4, got {int(n_steps)}")
    if float(action_delta) <= 0.0:
        raise ValueError(f"action_delta must be > 0, got {float(action_delta)}")

    device_obj = torch.device(str(device or _default_device()))
    config = _resolve_probe_config(
        checkpoint_path=checkpoint_path,
        from_scratch=bool(from_scratch),
        from_scratch_model_type=str(from_scratch_model_type),
        device_obj=device_obj,
        build_seed=int(build_seed),
        allow_config_only=True,
    )
    env_cfg = _build_audit_env_cfg(config["prior"]["environment"])
    loaded_frozen_h_path = None
    if fixed_frozen_h_json is not None:
        frozen_h, loaded_frozen_h_path = _load_full_frozen_h(str(fixed_frozen_h_json))
        fixed_bundle = {
            "fixed_env_contract": {
                "source_full_frozen_h_json": str(loaded_frozen_h_path),
                **summarize_fixed_env_h(frozen_h),
            }
        }
    else:
        fixed_bundle = _prepare_audit_fixed_env_contract(
            env_cfg=env_cfg,
            boundary_contract_mode="normal",
            ppo_reset_env_state_at_sep=True,
            frozen_h_seed=int(frozen_h_seed),
            frozen_h_seed_mode="current",
        )
        frozen_h = copy.deepcopy(fixed_bundle["frozen_h"])
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
    if prior_mlp_activations_override is not None:
        frozen_h["prior_mlp_activations"] = str(prior_mlp_activations_override)
    if init_std_override is not None:
        frozen_h["init_std"] = float(init_std_override)
    if noise_std_override is not None:
        frozen_h["noise_std"] = float(noise_std_override)
    if scm_standard_linear_init_enabled_override is not None:
        frozen_h["scm_standard_linear_init_enabled"] = bool(scm_standard_linear_init_enabled_override)
    frozen_h["alpha"] = float(alpha)
    if reference_state_inertia_enabled_override is not None:
        frozen_h["reference_state_inertia_enabled"] = bool(reference_state_inertia_enabled_override)
    if terminal_reset_enabled_override is not None:
        frozen_h["terminal_reset_enabled"] = bool(terminal_reset_enabled_override)
    if terminal_reset_count_target_override is not None:
        frozen_h["terminal_reset_count_target"] = float(terminal_reset_count_target_override)

    prior = EnvironmentPrior(copy.deepcopy(env_cfg))
    env = prior._sample_environment(copy.deepcopy(frozen_h), device=device_obj, rng_seed=int(train_env_seed))
    batch_size = int(rollout_count)
    dtype = torch.float32
    state_dim = int(env["state_dim"])
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
    action_t = prior._stack_randn_with_generators(
        rollout_generators, (batch_size, action_dim), device_obj, dtype
    ) * init_action_std[:, None]
    reward_t = torch.zeros((batch_size,), device=device_obj, dtype=dtype)
    reward_mask_t = torch.ones((batch_size,), device=device_obj, dtype=dtype)
    terminal_t = torch.zeros((batch_size,), device=device_obj, dtype=dtype)
    del action_t, reward_t, reward_mask_t, terminal_t

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
    policy_mode = str(behavior_policy or ("unit_gaussian_iid" if bool(sampled_policy) else "zero"))
    valid_policy_modes = {"unit_gaussian_iid", "zero", "constant_gaussian"}
    if policy_mode not in valid_policy_modes:
        raise ValueError(f"behavior_policy must be one of {sorted(valid_policy_modes)}, got {policy_mode!r}")
    constant_action_next = None
    if policy_mode == "constant_gaussian":
        constant_generators = _make_generators_from_seeds(
            prior,
            [int(30_000_000_000) + int(rollout_seed_start) + int(i) for i in range(batch_size)],
            device_obj,
        )
        constant_action_next = prior._stack_randn_with_generators(
            constant_generators, (batch_size, action_dim), device_obj, dtype
        )

    step1_sum = 0.0
    step1_count = 0
    step4_sum = 0.0
    step4_count = 0
    sensitivity_sum = 0.0
    sensitivity_count = 0
    reward_raw_sensitivity_sum = 0.0
    reward_effective_sensitivity_sum = 0.0
    reward_sensitivity_count = 0
    state_history: deque[torch.Tensor] = deque(maxlen=5)
    terminal_history: deque[torch.Tensor] = deque(maxlen=4)
    discounted_effective_returns = torch.zeros((batch_size,), device=device_obj, dtype=dtype)
    discounted_raw_returns = torch.zeros((batch_size,), device=device_obj, dtype=dtype)
    undiscounted_effective_returns = torch.zeros((batch_size,), device=device_obj, dtype=dtype)
    discount_t = torch.ones((batch_size,), device=device_obj, dtype=dtype)

    for step_idx in range(int(n_steps)):
        state_history.append(state_t.detach().clone())
        if len(state_history) == 5 and len(terminal_history) == 4:
            valid4 = ~torch.stack(list(terminal_history), dim=0).bool().any(dim=0)
            if bool(valid4.any()):
                delta4 = torch.linalg.vector_norm(state_history[-1] - state_history[0], ord=2, dim=1)
                step4_sum += float(delta4[valid4].sum().item())
                step4_count += int(valid4.sum().item())

        if policy_mode == "unit_gaussian_iid":
            action_next = prior._stack_randn_with_generators(
                rollout_generators, (batch_size, action_dim), device_obj, dtype
            )
        elif policy_mode == "zero":
            action_next = torch.zeros((batch_size, action_dim), device=device_obj, dtype=dtype)
        else:
            action_next = constant_action_next
        if (not reference_semantics_enabled) and action_noise_train_std > 0.0 and int(step_idx) < int(single_eval_pos):
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

        sens_direction_seeds = [
            int(10_000_000_000) + int(rollout_seed_start) + int(step_idx) * int(batch_size) + int(i)
            for i in range(batch_size)
        ]
        sens_direction_generators = _make_generators_from_seeds(prior, sens_direction_seeds, device_obj)
        sens_direction = prior._stack_randn_with_generators(
            sens_direction_generators, (batch_size, action_dim), device_obj, dtype
        )
        sens_direction = sens_direction / sens_direction.norm(dim=1, keepdim=True).clamp_min(1e-8)
        action_plus = prior._transform_reinforce_action(
            action_next + float(action_delta) * sens_direction,
            mode=action_transform_mode,
            rms_eps=action_rms_eps,
            clip_bound=action_clip_bound,
        )
        action_minus = prior._transform_reinforce_action(
            action_next - float(action_delta) * sens_direction,
            mode=action_transform_mode,
            rms_eps=action_rms_eps,
            clip_bound=action_clip_bound,
        )
        sens_transition_seeds = [
            int(20_000_000_000) + int(rollout_seed_start) + int(step_idx) * int(batch_size) + int(i)
            for i in range(batch_size)
        ]
        sens_gens_plus = _make_generators_from_seeds(prior, sens_transition_seeds, device_obj)
        sens_gens_minus = _make_generators_from_seeds(prior, sens_transition_seeds, device_obj)
        state_plus, reward_plus_raw, _, _ = _compute_preterminal_state_next_batch(
            prior=prior,
            env=env,
            carry_state_t=carry_state_t,
            state_t=state_t,
            action_next=action_plus,
            noise_t=noise_t,
            state_noise_t=state_noise_t,
            rollout_generators=sens_gens_plus,
            device=device_obj,
        )
        state_minus, reward_minus_raw, _, _ = _compute_preterminal_state_next_batch(
            prior=prior,
            env=env,
            carry_state_t=carry_state_t,
            state_t=state_t,
            action_next=action_minus,
            noise_t=noise_t,
            state_noise_t=state_noise_t,
            rollout_generators=sens_gens_minus,
            device=device_obj,
        )
        sensitivity = torch.linalg.vector_norm(state_plus - state_minus, ord=2, dim=1)
        sensitivity_sum += float(sensitivity.sum().item())
        sensitivity_count += int(sensitivity.numel())
        reward_raw_sensitivity = torch.abs(reward_plus_raw.reshape(batch_size) - reward_minus_raw.reshape(batch_size))
        reward_raw_sensitivity_sum += float(reward_raw_sensitivity.sum().item())

        aux_reward_next = prior._exact_scm_aux_reward_terms(
            action_next,
            ctrl_weight=env.get("ctrl_reward_weight", 0.0),
            ctrl_enabled=env.get("ctrl_reward_enabled", False),
            survival_weight=env.get("survival_reward_weight", 0.0),
            survival_enabled=env.get("survival_reward_enabled", False),
        )
        aux_reward_plus = prior._exact_scm_aux_reward_terms(
            action_plus,
            ctrl_weight=env.get("ctrl_reward_weight", 0.0),
            ctrl_enabled=env.get("ctrl_reward_enabled", False),
            survival_weight=env.get("survival_reward_weight", 0.0),
            survival_enabled=env.get("survival_reward_enabled", False),
        )
        aux_reward_minus = prior._exact_scm_aux_reward_terms(
            action_minus,
            ctrl_weight=env.get("ctrl_reward_weight", 0.0),
            ctrl_enabled=env.get("ctrl_reward_enabled", False),
            survival_weight=env.get("survival_reward_weight", 0.0),
            survival_enabled=env.get("survival_reward_enabled", False),
        )
        reward_plus_effective = prior._compose_exact_scm_reward(
            reward_plus_raw.reshape(batch_size),
            aux_reward=aux_reward_plus.reshape(batch_size),
            reward_clip=env.get("reward_clip", float("inf")),
            mode=env.get("reinforce_reward_transform", "none"),
            rms_eps=env.get("reinforce_reward_rms_eps", 1e-6),
            tanh_c=env.get("reinforce_reward_tanh_c", 1.0),
            tanh_bound=env.get("reinforce_reward_tanh_bound", 2.0),
        ).reshape(batch_size)
        reward_minus_effective = prior._compose_exact_scm_reward(
            reward_minus_raw.reshape(batch_size),
            aux_reward=aux_reward_minus.reshape(batch_size),
            reward_clip=env.get("reward_clip", float("inf")),
            mode=env.get("reinforce_reward_transform", "none"),
            rms_eps=env.get("reinforce_reward_rms_eps", 1e-6),
            tanh_c=env.get("reinforce_reward_tanh_c", 1.0),
            tanh_bound=env.get("reinforce_reward_tanh_bound", 2.0),
        ).reshape(batch_size)
        reward_effective_sensitivity = torch.abs(reward_plus_effective - reward_minus_effective)
        reward_effective_sensitivity_sum += float(reward_effective_sensitivity.sum().item())
        reward_sensitivity_count += int(reward_effective_sensitivity.numel())
        reward_next = prior._compose_exact_scm_reward(
            reward_next_raw.reshape(batch_size),
            aux_reward=aux_reward_next.reshape(batch_size),
            reward_clip=env.get("reward_clip", float("inf")),
            mode=env.get("reinforce_reward_transform", "none"),
            rms_eps=env.get("reinforce_reward_rms_eps", 1e-6),
            tanh_c=env.get("reinforce_reward_tanh_c", 1.0),
            tanh_bound=env.get("reinforce_reward_tanh_bound", 2.0),
        ).reshape(batch_size)
        if reward_dropout_enabled and reward_dropout_ratio > 0.0:
            drop_mask = prior._stack_rand_with_generators(rollout_generators, (batch_size,), device_obj, dtype) < float(
                reward_dropout_ratio
            )
            if bool(reward_impute_zero):
                reward_next = torch.where(drop_mask, torch.zeros_like(reward_next), reward_next)
        discounted_effective_returns += discount_t * reward_next.reshape(batch_size)
        discounted_raw_returns += discount_t * reward_next_raw.reshape(batch_size)
        undiscounted_effective_returns += reward_next.reshape(batch_size)
        discount_t = discount_t * float(discount_gamma)

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
        valid1 = ~terminal_next.bool()
        if bool(valid1.any()):
            delta1 = torch.linalg.vector_norm(state_next_actual - state_t, ord=2, dim=1)
            step1_sum += float(delta1[valid1].sum().item())
            step1_count += int(valid1.sum().item())
        terminal_history.append(terminal_next.detach().clone())
        carry_state_next = prior._mix_exact_scm_carry_state(
            carry_state_t,
            state_next_actual,
            env["alpha"],
        )
        if bool(env.get("terminal_reset_enabled", False)):
            carry_state_next = prior._sync_exact_scm_carry_state_with_visible(
                carry_state_next,
                state_next_actual,
                terminal_next,
            )
        carry_state_t = carry_state_next
        state_t = state_next_actual

    state_history.append(state_t.detach().clone())
    if len(state_history) == 5 and len(terminal_history) == 4:
        valid4 = ~torch.stack(list(terminal_history), dim=0).bool().any(dim=0)
        if bool(valid4.any()):
            delta4 = torch.linalg.vector_norm(state_history[-1] - state_history[0], ord=2, dim=1)
            step4_sum += float(delta4[valid4].sum().item())
            step4_count += int(valid4.sum().item())

    report = {
        "alpha": float(alpha),
        "rollout_count": int(rollout_count),
        "n_steps": int(n_steps),
        "single_eval_pos": int(single_eval_pos),
        "action_delta": float(action_delta),
        "behavior_policy": policy_mode,
        "discount_gamma": float(discount_gamma),
        "sampled_policy": bool(sampled_policy),
        "state_dim": int(env["state_dim"]),
        "obs_dim": int(env["obs_dim"]),
        "action_dim": int(env["action_dim"]),
        "noise_dim": int(env["noise_dim"]),
        "zero_pad_dim": int(env["zero_pad_dim"]),
        "fixed_frozen_h_json": None if loaded_frozen_h_path is None else str(loaded_frozen_h_path),
        "effective_frozen_h_snapshot": {
            "family": frozen_h.get("family"),
            "num_layers": int(frozen_h["num_layers"]) if "num_layers" in frozen_h else None,
            "state_dim": int(frozen_h["state_dim"]) if "state_dim" in frozen_h else None,
            "obs_dim": int(frozen_h["obs_dim"]) if "obs_dim" in frozen_h else None,
            "action_dim": int(frozen_h["action_dim"]) if "action_dim" in frozen_h else None,
            "noise_dim": int(frozen_h["noise_dim"]) if "noise_dim" in frozen_h else None,
            "zero_pad_dim": int(frozen_h["zero_pad_dim"]) if "zero_pad_dim" in frozen_h else None,
            "alpha": float(frozen_h["alpha"]) if "alpha" in frozen_h else None,
            "init_std": float(frozen_h["init_std"]) if "init_std" in frozen_h else None,
            "noise_std": float(frozen_h["noise_std"]) if "noise_std" in frozen_h else None,
            "prior_mlp_activations": frozen_h.get("prior_mlp_activations"),
            "prior_mlp_hidden_dim": int(frozen_h["prior_mlp_hidden_dim"]) if "prior_mlp_hidden_dim" in frozen_h else None,
            "reference_state_inertia_enabled": bool(frozen_h["reference_state_inertia_enabled"]) if "reference_state_inertia_enabled" in frozen_h else None,
            "scm_standard_linear_init_enabled": bool(frozen_h["scm_standard_linear_init_enabled"]) if "scm_standard_linear_init_enabled" in frozen_h else None,
            "terminal_reset_enabled": bool(frozen_h["terminal_reset_enabled"]) if "terminal_reset_enabled" in frozen_h else None,
            "terminal_reset_count_target": float(frozen_h["terminal_reset_count_target"]) if "terminal_reset_count_target" in frozen_h else None,
            "constrained_dim_sampling_total_budget": (
                int(frozen_h["constrained_dim_sampling_total_budget"])
                if "constrained_dim_sampling_total_budget" in frozen_h
                else None
            ),
        },
        "sampled_env_snapshot": {
            "family": env.get("family"),
            "num_layers": int(env["num_layers"]) if "num_layers" in env else None,
            "state_dim": int(env["state_dim"]) if "state_dim" in env else None,
            "obs_dim": int(env["obs_dim"]) if "obs_dim" in env else None,
            "action_dim": int(env["action_dim"]) if "action_dim" in env else None,
            "noise_dim": int(env["noise_dim"]) if "noise_dim" in env else None,
            "zero_pad_dim": int(env["zero_pad_dim"]) if "zero_pad_dim" in env else None,
            "alpha": float(env["alpha"]) if "alpha" in env else None,
            "init_std": float(env["init_std"]) if "init_std" in env else None,
            "noise_std": float(env["noise_std"]) if "noise_std" in env else None,
            "prior_mlp_activations": env.get("prior_mlp_activations"),
            "prior_mlp_hidden_dim": int(env["prior_mlp_hidden_dim"]) if "prior_mlp_hidden_dim" in env else None,
            "reference_state_inertia_enabled": bool(env.get("reference_state_inertia_enabled", False)),
            "scm_standard_linear_init_enabled": bool(env.get("scm_standard_linear_init_enabled", False)),
            "terminal_reset_enabled": bool(env.get("terminal_reset_enabled", False)),
            "terminal_reset_count_target": float(env.get("terminal_reset_count_target", 0.0)),
            "ctrl_reward_enabled": bool(env.get("ctrl_reward_enabled", False)),
            "survival_reward_enabled": bool(env.get("survival_reward_enabled", False)),
        },
        "reference_semantics_enabled": bool(reference_semantics_enabled),
        "reference_state_inertia_enabled": bool(env.get("reference_state_inertia_enabled", False)),
        "terminal_reset_enabled": bool(env.get("terminal_reset_enabled", False)),
        "terminal_reset_count_target": float(env.get("terminal_reset_count_target", 0.0)),
        "mean_l2_s_t1_minus_s_t": (float(step1_sum / max(step1_count, 1)) if step1_count > 0 else None),
        "mean_l2_s_t4_minus_s_t": (float(step4_sum / max(step4_count, 1)) if step4_count > 0 else None),
        "mean_l2_s_t1_action_sensitivity": (
            float(sensitivity_sum / max(sensitivity_count, 1)) if sensitivity_count > 0 else None
        ),
        "mean_l2_s_t1_action_sensitivity_per_action_unit": (
            float((sensitivity_sum / max(sensitivity_count, 1)) / max(2.0 * float(action_delta), 1e-12))
            if sensitivity_count > 0
            else None
        ),
        "mean_abs_reward_raw_t1_action_sensitivity": (
            float(reward_raw_sensitivity_sum / max(reward_sensitivity_count, 1))
            if reward_sensitivity_count > 0
            else None
        ),
        "mean_abs_reward_raw_t1_action_sensitivity_per_action_unit": (
            float((reward_raw_sensitivity_sum / max(reward_sensitivity_count, 1)) / max(2.0 * float(action_delta), 1e-12))
            if reward_sensitivity_count > 0
            else None
        ),
        "mean_abs_reward_effective_t1_action_sensitivity": (
            float(reward_effective_sensitivity_sum / max(reward_sensitivity_count, 1))
            if reward_sensitivity_count > 0
            else None
        ),
        "mean_abs_reward_effective_t1_action_sensitivity_per_action_unit": (
            float((reward_effective_sensitivity_sum / max(reward_sensitivity_count, 1)) / max(2.0 * float(action_delta), 1e-12))
            if reward_sensitivity_count > 0
            else None
        ),
        "state_action_sensitivity_to_step1_drift_ratio": (
            float((sensitivity_sum / max(sensitivity_count, 1)) / max(step1_sum / max(step1_count, 1), 1e-12))
            if sensitivity_count > 0 and step1_count > 0
            else None
        ),
        "discounted_effective_return": _tensor_stats_1d(discounted_effective_returns),
        "discounted_raw_return": _tensor_stats_1d(discounted_raw_returns),
        "undiscounted_effective_return": _tensor_stats_1d(undiscounted_effective_returns),
        "step1_valid_count": int(step1_count),
        "step1_total_count": int(n_steps) * int(batch_size),
        "step1_valid_fraction": float(step1_count / max(int(n_steps) * int(batch_size), 1)),
        "step4_valid_count": int(step4_count),
        "step4_total_count": max(int(n_steps) - 3, 0) * int(batch_size),
        "step4_valid_fraction": float(step4_count / max(max(int(n_steps) - 3, 0) * int(batch_size), 1)),
    }
    return report


def run_alpha_temporal_control_sweep(
    *,
    alpha_list: list[float],
    **kwargs,
) -> dict[str, Any]:
    reports = []
    for alpha in [float(v) for v in alpha_list]:
        report = measure_alpha_temporal_control(alpha=alpha, **kwargs)
        reports.append(report)
        print(
            (
                f"[alpha-temporal-control] alpha={float(alpha):.4f} "
                f"step1={report['mean_l2_s_t1_minus_s_t']:.4f} "
                f"step4={report['mean_l2_s_t4_minus_s_t']:.4f} "
                f"sens={report['mean_l2_s_t1_action_sensitivity']:.4f} "
                f"step1_valid={report['step1_valid_fraction']:.4f} "
                f"step4_valid={report['step4_valid_fraction']:.4f}"
            ),
            flush=True,
        )
    return {"reports": reports}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure temporal drift and one-step action sensitivity under alpha sweeps in the exact prior env."
    )
    parser.add_argument("checkpoint_path", nargs="?", default=None, type=str)
    parser.add_argument("--from-scratch", type=str2bool, default=True)
    parser.add_argument("--from-scratch-model-type", type=str, default="rlpfn")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--frozen-h-seed", type=int, default=12345)
    parser.add_argument("--train-env-seed", type=int, default=2020)
    parser.add_argument("--single-eval-pos", type=int, default=64)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--build-seed", type=int, default=4040)
    parser.add_argument("--rollout-count", type=int, default=4096)
    parser.add_argument("--rollout-seed-start", type=int, default=5000)
    parser.add_argument("--sampled-policy", type=str2bool, default=True)
    parser.add_argument("--constrained-dim-sampling-total-budget-override", type=int, default=None)
    parser.add_argument("--state-dim-override", type=int, default=None)
    parser.add_argument("--action-dim-override", type=int, default=None)
    parser.add_argument("--noise-dim-override", type=int, default=None)
    parser.add_argument("--zero-pad-dim-override", type=int, default=None)
    parser.add_argument("--fixed-frozen-h-json", type=str, default=None)
    parser.add_argument("--prior-mlp-activations-override", type=str, default=None)
    parser.add_argument("--init-std-override", type=float, default=None)
    parser.add_argument("--noise-std-override", type=float, default=None)
    parser.add_argument("--scm-standard-linear-init-enabled-override", type=str2bool, default=None)
    parser.add_argument("--alpha-list", type=str, default="1.0,0.2,0.05")
    parser.add_argument("--reference-state-inertia-enabled-override", type=str2bool, default=True)
    parser.add_argument("--terminal-reset-enabled-override", type=str2bool, default=None)
    parser.add_argument("--terminal-reset-count-target-override", type=float, default=None)
    parser.add_argument("--action-delta", type=float, default=0.1)
    parser.add_argument(
        "--behavior-policy",
        type=str,
        default=None,
        choices=["unit_gaussian_iid", "zero", "constant_gaussian"],
    )
    parser.add_argument("--discount-gamma", type=float, default=0.99)
    parser.add_argument(
        "--output-json",
        type=str,
        default="/home/chen/RLPFN/artifacts/phase2_alpha_temporal_control_probe.json",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    _seed_all(int(args.build_seed))
    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = run_alpha_temporal_control_sweep(
        alpha_list=_parse_alpha_list(args.alpha_list),
        checkpoint_path=args.checkpoint_path,
        device=args.device,
        from_scratch=bool(args.from_scratch),
        from_scratch_model_type=str(args.from_scratch_model_type),
        frozen_h_seed=int(args.frozen_h_seed),
        train_env_seed=int(args.train_env_seed),
        single_eval_pos=int(args.single_eval_pos),
        n_steps=int(args.n_steps),
        build_seed=int(args.build_seed),
        rollout_count=int(args.rollout_count),
        rollout_seed_start=int(args.rollout_seed_start),
        sampled_policy=bool(args.sampled_policy),
        constrained_dim_sampling_total_budget_override=args.constrained_dim_sampling_total_budget_override,
        state_dim_override=args.state_dim_override,
        action_dim_override=args.action_dim_override,
        noise_dim_override=args.noise_dim_override,
        zero_pad_dim_override=args.zero_pad_dim_override,
        fixed_frozen_h_json=args.fixed_frozen_h_json,
        prior_mlp_activations_override=args.prior_mlp_activations_override,
        init_std_override=args.init_std_override,
        noise_std_override=args.noise_std_override,
        scm_standard_linear_init_enabled_override=args.scm_standard_linear_init_enabled_override,
        reference_state_inertia_enabled_override=args.reference_state_inertia_enabled_override,
        terminal_reset_enabled_override=args.terminal_reset_enabled_override,
        terminal_reset_count_target_override=args.terminal_reset_count_target_override,
        action_delta=float(args.action_delta),
        behavior_policy=args.behavior_policy,
        discount_gamma=float(args.discount_gamma),
    )
    payload = json.dumps(report, sort_keys=True)
    output_path.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
