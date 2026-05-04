import argparse
import copy
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from ticl.analysis.fixed_env_h import (
    build_fixed_env_h,
    sample_fixed_env_h_from_prior,
    summarize_fixed_env_h,
)
from ticl.analysis.prior_generalization_audit import (
    _build_zero_policy_step_fn,
    evaluate_prior_suite,
)
from ticl.model_builder import load_model
from ticl.priors.environment_prior import EnvironmentPrior
from ticl.sb3_recurrent_ppo import (
    _explained_variance_with_mask,
    _ppo_autocast_context,
    _recover_raw_from_value_space,
    _restore_saved_recurrent_ppo_policy_head_state,
    build_recurrent_ppo,
)
from stable_baselines3.common.logger import configure as configure_logger


def _phase3_profile_timing_enabled() -> bool:
    return str(os.environ.get("TICL_PHASE3_PROFILE_TIMING", "")).strip().lower() not in {
        "",
        "0",
        "false",
        "no",
        "off",
    }


def _phase3_profile_timing(label: str, started_at: float) -> None:
    if _phase3_profile_timing_enabled():
        print(f"[phase3-profile] {label}={time.perf_counter() - started_at:.3f}s", flush=True)


def _default_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _seed_all(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _clone_h_repeated(h, count: int):
    return [copy.deepcopy(h) for _ in range(int(count))]


def _build_frozen_h_from_mode(
    *,
    prior_cfg: dict,
    frozen_h_seed: int,
    seed_mode: str,
    prior: EnvironmentPrior | None = None,
):
    seed_mode_resolved = str(seed_mode).strip().lower()
    if seed_mode_resolved in {"current", "legacy_numpy_only"}:
        return build_fixed_env_h(
            prior_cfg=prior_cfg,
            frozen_h_seed=int(frozen_h_seed),
            seed_mode=seed_mode_resolved,
        )
    if seed_mode_resolved == "legacy_same_prior_numpy_only":
        if prior is None:
            raise ValueError("legacy_same_prior_numpy_only requires the target prior instance.")
        return sample_fixed_env_h_from_prior(
            prior=prior,
            frozen_h_seed=int(frozen_h_seed),
            seed_mode="legacy_numpy_only",
        )
    if seed_mode_resolved != "legacy_numpy_only":
        raise ValueError(f"Unsupported frozen_h seed mode: {seed_mode}")
    raise ValueError(f"Unsupported frozen_h seed mode: {seed_mode}")


def _make_suite(*, frozen_h, env_seeds: list[int], rollout_seeds: list[int]) -> dict:
    return {
        "h_list": _clone_h_repeated(frozen_h, len(env_seeds)),
        "env_seeds": [int(v) for v in env_seeds],
        "rollout_seeds": [int(v) for v in rollout_seeds],
        "batch_size": len(env_seeds),
        "suite_seed": None,
    }


def _make_deterministic_batch_plan(buffer):
    setter = getattr(buffer, "set_deterministic_batch_plan", None)
    if callable(setter):
        setter(True)
        return
    buffer._deterministic_batch_plan = True


def _run_extra_critic_updates(algo, *, extra_updates: int) -> None:
    return _run_extra_critic_updates_with_scope(algo, extra_updates=extra_updates, scope="shared")


def _run_extra_critic_updates_with_scope(algo, *, extra_updates: int, scope: str = "shared") -> None:
    extra_updates = int(extra_updates)
    if extra_updates <= 0:
        return
    scope = str(scope).strip().lower()
    if scope not in {"shared", "head_only", "critic_path"}:
        raise ValueError(f"Unsupported extra critic update scope: {scope}")
    rollout_buffer = algo.rollout_buffer
    _make_deterministic_batch_plan(rollout_buffer)
    buffer_getter_flat = getattr(rollout_buffer, "get_gpu_flat", None)
    if not callable(buffer_getter_flat):
        raise RuntimeError("extra critic updates require rollout_buffer.get_gpu_flat().")
    original_requires_grad = None
    grad_params = list(algo.policy.parameters())
    if scope in {"head_only", "critic_path"}:
        original_requires_grad = {}
        if scope == "head_only":
            enabled_params = list(algo.policy.get_value_head_trainable_parameters())
        else:
            if not bool(getattr(algo.policy, "separate_value_backbone", False)):
                raise RuntimeError("critic_path extra critic updates require separate value backbone.")
            enabled_params = list(algo.policy.get_value_head_trainable_parameters()) + list(
                algo.policy.value_rlpfn_model.parameters()
            )
        value_param_ids = {id(param) for param in enabled_params}
        grad_params = enabled_params
        for name, param in algo.policy.named_parameters():
            original_requires_grad[name] = bool(param.requires_grad)
            param.requires_grad = id(param) in value_param_ids
    for _ in range(extra_updates):
        try:
            for rollout_data in buffer_getter_flat(algo.batch_size):
                objective_mask = rollout_data.objective_masks > 1e-8
                valid_total = objective_mask.to(
                    device=rollout_data.returns.device,
                    dtype=rollout_data.returns.dtype,
                ).sum().clamp_min(1.0)
                algo.policy.optimizer.zero_grad(set_to_none=True)
                with _ppo_autocast_context(
                    autocast_dtype=getattr(algo, "_rwkv_autocast_dtype", None),
                    device=getattr(algo, "device", rollout_data.observations.device),
                ):
                    eval_outputs = algo.policy.evaluate_actions_with_hidden_flat(
                        rollout_data.observations,
                        rollout_data.actions,
                        seq_lengths=rollout_data.seq_lengths,
                        action_masks=rollout_data.action_masks,
                    )
                    value_logits = eval_outputs["value_logits"]
                    value_target_bucket_idx = getattr(rollout_data, "value_target_bucket_idx", None)
                    value_loss = algo.policy.compute_value_loss_from_logits(
                        value_logits=value_logits,
                        targets=rollout_data.returns.to(device=value_logits.device, dtype=torch.float32),
                        objective_mask=objective_mask,
                        value_target_bucket_idx=value_target_bucket_idx,
                    )
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(grad_params, algo.max_grad_norm)
                algo.policy.optimizer.step()
            _make_deterministic_batch_plan(rollout_buffer)
        finally:
            if original_requires_grad is not None:
                for name, param in algo.policy.named_parameters():
                    param.requires_grad = original_requires_grad[name]


def _masked_corrcoef_np(x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if int(mask.sum()) <= 1:
        return 0.0
    x_m = np.asarray(x, dtype=np.float32).reshape(-1)[mask]
    y_m = np.asarray(y, dtype=np.float32).reshape(-1)[mask]
    x_m = x_m - float(x_m.mean())
    y_m = y_m - float(y_m.mean())
    denom = float(np.linalg.norm(x_m) * np.linalg.norm(y_m))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(x_m, y_m) / denom)


def _measure_rollout_critic_quality(algo) -> dict:
    rollout_buffer = algo.rollout_buffer
    ensure_generator_ready = getattr(rollout_buffer, "_ensure_generator_ready", None)
    if callable(ensure_generator_ready):
        ensure_generator_ready()
    objective_mask = np.asarray(rollout_buffer.objective_masks, dtype=np.float32).reshape(-1) > 1e-8
    normalized_values = np.asarray(rollout_buffer.values, dtype=np.float32).reshape(-1)
    normalized_returns = np.asarray(rollout_buffer.returns, dtype=np.float32).reshape(-1)
    raw_values = rollout_buffer._get_flat_raw_values_tensor(dtype=torch.float32).cpu().numpy()
    raw_returns = rollout_buffer._get_flat_raw_returns_tensor(dtype=torch.float32).cpu().numpy()
    objective_total = int(objective_mask.sum())
    if objective_total > 1:
        normalized_value_std = float(np.asarray(normalized_values[objective_mask], dtype=np.float32).std())
        normalized_return_std = float(np.asarray(normalized_returns[objective_mask], dtype=np.float32).std())
        raw_value_std = float(np.asarray(raw_values[objective_mask], dtype=np.float32).std())
        raw_return_std = float(np.asarray(raw_returns[objective_mask], dtype=np.float32).std())
    else:
        normalized_value_std = 0.0
        normalized_return_std = 0.0
        raw_value_std = 0.0
        raw_return_std = 0.0
    return {
        "objective_total": objective_total,
        "normalized_value_std": normalized_value_std,
        "normalized_return_std": normalized_return_std,
        "raw_value_std": raw_value_std,
        "raw_return_std": raw_return_std,
        "raw_corr": _masked_corrcoef_np(raw_values, raw_returns, objective_mask),
        "explained_variance_normalized": float(
            _explained_variance_with_mask(
                normalized_values,
                normalized_returns,
                mask=objective_mask,
            )
        ),
        "explained_variance_raw": float(
            _explained_variance_with_mask(
                raw_values,
                raw_returns,
                mask=objective_mask,
            )
        ),
    }


@torch.no_grad()
def _measure_current_policy_rollout_critic_quality(algo) -> dict:
    rollout_buffer = algo.rollout_buffer
    buffer_getter_flat = getattr(rollout_buffer, "get_gpu_flat", None)
    if not callable(buffer_getter_flat):
        return _measure_rollout_critic_quality(algo)

    _make_deterministic_batch_plan(rollout_buffer)
    pred_chunks: list[torch.Tensor] = []
    return_chunks: list[torch.Tensor] = []
    normalized_pred_chunks: list[torch.Tensor] = []
    normalized_return_chunks: list[torch.Tensor] = []
    mask_chunks: list[torch.Tensor] = []
    for rollout_data in buffer_getter_flat(algo.batch_size):
        eval_outputs = algo.policy.evaluate_actions_with_hidden_flat(
            rollout_data.observations,
            rollout_data.actions,
            seq_lengths=rollout_data.seq_lengths,
            action_masks=rollout_data.action_masks,
        )
        values = eval_outputs["values"].flatten()
        normalized_pred_chunks.append(values.detach().cpu())
        normalized_return_chunks.append(rollout_data.returns.flatten().detach().cpu())
        raw_pred = _recover_raw_from_value_space(
            values.detach(),
            value_means=rollout_data.rollout_return_means.detach(),
            value_stds=rollout_data.rollout_return_stds.detach(),
        )
        raw_returns = _recover_raw_from_value_space(
            rollout_data.returns.flatten().detach(),
            value_means=rollout_data.rollout_return_means.detach(),
            value_stds=rollout_data.rollout_return_stds.detach(),
        )
        pred_chunks.append(raw_pred.detach().cpu())
        return_chunks.append(raw_returns.detach().cpu())
        mask_chunks.append((rollout_data.objective_masks.flatten() > 1e-8).detach().cpu())
    _make_deterministic_batch_plan(rollout_buffer)

    if not pred_chunks:
        return {
            "objective_total": 0,
            "raw_corr": 0.0,
            "explained_variance_normalized": float("nan"),
            "explained_variance_raw": float("nan"),
        }

    normalized_values = torch.cat(normalized_pred_chunks, dim=0).numpy()
    normalized_returns = torch.cat(normalized_return_chunks, dim=0).numpy()
    raw_values = torch.cat(pred_chunks, dim=0).numpy()
    raw_returns = torch.cat(return_chunks, dim=0).numpy()
    objective_mask = torch.cat(mask_chunks, dim=0).numpy()
    objective_total = int(objective_mask.sum())
    if objective_total > 1:
        normalized_value_std = float(np.asarray(normalized_values[objective_mask], dtype=np.float32).std())
        normalized_return_std = float(np.asarray(normalized_returns[objective_mask], dtype=np.float32).std())
        raw_value_std = float(np.asarray(raw_values[objective_mask], dtype=np.float32).std())
        raw_return_std = float(np.asarray(raw_returns[objective_mask], dtype=np.float32).std())
    else:
        normalized_value_std = 0.0
        normalized_return_std = 0.0
        raw_value_std = 0.0
        raw_return_std = 0.0
    return {
        "objective_total": objective_total,
        "normalized_value_std": normalized_value_std,
        "normalized_return_std": normalized_return_std,
        "raw_value_std": raw_value_std,
        "raw_return_std": raw_return_std,
        "raw_corr": _masked_corrcoef_np(raw_values, raw_returns, objective_mask),
        "explained_variance_normalized": float(
            _explained_variance_with_mask(
                normalized_values,
                normalized_returns,
                mask=objective_mask,
            )
        ),
        "explained_variance_raw": float(
            _explained_variance_with_mask(
                raw_values,
                raw_returns,
                mask=objective_mask,
            )
        ),
    }


def _snapshot_policy_state_dict(policy) -> dict[str, torch.Tensor]:
    state_dict = policy.state_dict()
    return {str(name): tensor.detach().cpu().clone() for name, tensor in state_dict.items()}


def _resolve_vf_coef(config: dict, vf_coef_override: float | None) -> float:
    if vf_coef_override is not None:
        return float(vf_coef_override)
    return float(config["optimizer"].get("ppo_vf_coef", 0.5))


def _resolve_policy_loss_coef(policy_loss_coef_override: float | None) -> float:
    if policy_loss_coef_override is not None:
        return float(policy_loss_coef_override)
    return 1.0


def _measure_suite_critic_quality_with_policy_state(
    *,
    checkpoint_path: str,
    device_obj: torch.device,
    env_cfg: dict,
    num_features: int,
    suite: dict,
    single_eval_pos: int,
    n_steps: int,
    learning_rate: float,
    batch_size: int,
    n_epochs: int,
    target_kl: float,
    ppo_reset_env_state_at_sep: bool,
    ppo_separate_value_backbone: bool,
    deterministic_actor_sampling: bool,
    deterministic_batch_plan: bool,
    strict_native_rollout: bool,
    actor_gae_space: str,
    actor_baseline_mode: str,
    actor_objective_mode: str,
    boundary_contract_mode: str,
    policy_state_dict: dict[str, torch.Tensor],
    build_seed: int | None = None,
    vf_coef_override: float | None = None,
    policy_loss_coef_override: float | None = None,
    value_head_impl: str = "legacy_bar",
    value_path_adapter_impl: str = "none",
) -> dict:
    if build_seed is not None:
        _seed_all(int(build_seed))
    load_model.cache_clear()
    model, config = load_model(checkpoint_path, device=device_obj, verbose=False)
    prior = EnvironmentPrior(copy.deepcopy(env_cfg))
    suite_h_list = [copy.deepcopy(h) for h in list(suite["h_list"])]
    prior._sample_batch_hypers = lambda batch_n, _h_list=suite_h_list: [
        copy.deepcopy(h) for h in _h_list[: int(batch_n)]
    ]
    prior._sample_single_eval_pos = lambda n_samples_arg, rng_seed=None: int(single_eval_pos)
    _apply_boundary_contract_mode_flags(prior, boundary_contract_mode)
    _apply_sep_state_reset_flag(prior, bool(ppo_reset_env_state_at_sep))

    algo, callback, vec_env = build_recurrent_ppo(
        model=model,
        env_prior=prior,
        device=str(device_obj),
        num_features=int(num_features),
        n_envs=int(suite["batch_size"]),
        n_steps=int(n_steps),
        learning_rate=float(learning_rate),
        batch_size=int(batch_size),
        n_epochs=int(n_epochs),
        gamma=float(config["optimizer"].get("ppo_gamma", 0.98)),
        gae_lambda=float(config["optimizer"].get("ppo_gae_lambda", 0.90)),
        clip_range=config["optimizer"].get("ppo_clip_range", 0.2),
        clip_range_vf=None,
        normalize_advantage=False,
        actor_gae_space=str(actor_gae_space),
        actor_baseline_mode=str(actor_baseline_mode),
        actor_objective_mode=str(actor_objective_mode),
        reset_env_state_at_sep=bool(ppo_reset_env_state_at_sep),
        separate_value_backbone=bool(ppo_separate_value_backbone),
        ent_coef=float(config["optimizer"].get("ppo_ent_coef", 0.0)),
        vf_coef=_resolve_vf_coef(config, vf_coef_override),
        max_grad_norm=float(config["optimizer"].get("ppo_max_grad_norm", 0.5)),
        target_kl=float(target_kl),
        strict_fixed_env_mode=True,
        env_rng_seeds=[int(v) for v in list(suite["env_seeds"])],
        rollout_rng_seeds=[int(v) for v in list(suite["rollout_seeds"])],
        deterministic_actor_sampling=bool(deterministic_actor_sampling),
        deterministic_batch_plan=bool(deterministic_batch_plan),
        strict_native_rollout=bool(strict_native_rollout),
        restore_validation_policy_state=False,
        value_head_impl=str(value_head_impl),
        value_path_adapter_impl=str(value_path_adapter_impl),
        verbose=0,
    )
    try:
        algo._rwkv_policy_loss_coef = _resolve_policy_loss_coef(policy_loss_coef_override)
        algo.policy.load_state_dict(policy_state_dict, strict=True)
        vec_env._sample_seed_list = lambda _seeds=list(suite["env_seeds"]): [int(v) for v in list(_seeds)]
        if bool(deterministic_batch_plan):
            _make_deterministic_batch_plan(algo.rollout_buffer)
        algo.ep_info_buffer = []
        algo.ep_success_buffer = []
        algo._last_obs = vec_env.reset()
        algo._last_episode_starts = np.ones((vec_env.num_envs,), dtype=bool)
        algo._last_lstm_states = algo.policy._dummy_states(vec_env.num_envs)
        callback.init_callback(algo)
        algo._update_current_progress_remaining(0, int(algo.n_steps))
        assert algo.collect_rollouts(vec_env, callback, algo.rollout_buffer, n_rollout_steps=algo.n_steps)
        return _measure_rollout_critic_quality(algo)
    finally:
        vec_env.close()
        del algo, callback, vec_env, model, prior
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _prepare_audit_fixed_env_contract(
    *,
    env_cfg: dict,
    boundary_contract_mode: str,
    ppo_reset_env_state_at_sep: bool,
    frozen_h_seed: int,
    frozen_h_seed_mode: str,
):
    prior_for_h = EnvironmentPrior(copy.deepcopy(env_cfg))
    _apply_boundary_contract_mode_flags(prior_for_h, boundary_contract_mode)
    _apply_sep_state_reset_flag(prior_for_h, bool(ppo_reset_env_state_at_sep))

    seed_mode_resolved = str(frozen_h_seed_mode).strip().lower()
    reuse_sampling_prior_state = seed_mode_resolved == "legacy_same_prior_numpy_only"
    frozen_h = _build_frozen_h_from_mode(
        prior_cfg=env_cfg,
        frozen_h_seed=int(frozen_h_seed),
        seed_mode=seed_mode_resolved,
        prior=prior_for_h if reuse_sampling_prior_state else None,
    )
    fixed_env_contract = {
        "frozen_h_seed": int(frozen_h_seed),
        "frozen_h_seed_mode": str(seed_mode_resolved),
        "zero_eval_prior_reuses_sampling_state": bool(reuse_sampling_prior_state),
    }
    fixed_env_contract.update(summarize_fixed_env_h(frozen_h))
    return {
        "prior_for_h": prior_for_h,
        "frozen_h": frozen_h,
        "fixed_env_contract": fixed_env_contract,
    }


def _audit_train_loop(
    algo,
    callback,
    vec_env,
    *,
    outer_epochs: int,
    extra_critic_updates_per_outer_epoch: int = 0,
    extra_critic_update_scope: str = "shared",
    critic_warmup: dict | None = None,
    record_post_train_rollout_critic_quality: bool = False,
) -> list[dict]:
    total_timesteps = int(algo.n_steps) * int(outer_epochs)
    history: list[dict] = []
    if getattr(algo, "_last_obs", None) is None:
        algo.ep_info_buffer = []
        algo.ep_success_buffer = []
        if not hasattr(algo, "_logger"):
            algo.set_logger(configure_logger(folder=None, format_strings=[]))
        algo._last_obs = vec_env.reset()
        algo._last_episode_starts = np.ones((vec_env.num_envs,), dtype=bool)
        algo._last_lstm_states = algo.policy._dummy_states(vec_env.num_envs)
        callback.init_callback(algo)
    warmup_cfg = None
    switched_on = False
    if critic_warmup:
        warmup_cfg = {
            "min_outer_epochs": int(critic_warmup.get("min_outer_epochs", 0)),
            "raw_corr_threshold": float(critic_warmup.get("raw_corr_threshold", 0.0)),
            "switched_on": False,
        }
    for outer_idx in range(int(outer_epochs)):
        algo._update_current_progress_remaining(int(outer_idx * algo.n_steps), total_timesteps)
        baseline_mode_for_epoch = str(getattr(algo.rollout_buffer, "_actor_baseline_mode", "learned")).strip().lower()
        if warmup_cfg is not None:
            baseline_mode_for_epoch = "learned" if switched_on else "zero"
            algo.rollout_buffer._actor_baseline_mode = baseline_mode_for_epoch
        t_collect = time.perf_counter()
        assert algo.collect_rollouts(vec_env, callback, algo.rollout_buffer, n_rollout_steps=algo.n_steps)
        _phase3_profile_timing(f"outer_epoch={outer_idx + 1} collect_rollouts", t_collect)
        t_quality = time.perf_counter()
        critic_quality = _measure_rollout_critic_quality(algo)
        _phase3_profile_timing(f"outer_epoch={outer_idx + 1} critic_quality", t_quality)
        if warmup_cfg is not None and not switched_on:
            if (
                int(outer_idx + 1) >= int(warmup_cfg["min_outer_epochs"])
                and float(critic_quality["raw_corr"]) >= float(warmup_cfg["raw_corr_threshold"])
            ):
                switched_on = True
                warmup_cfg["switched_on"] = True
        t_train = time.perf_counter()
        algo.train()
        _phase3_profile_timing(f"outer_epoch={outer_idx + 1} algo.train", t_train)
        t_extra = time.perf_counter()
        _run_extra_critic_updates_with_scope(
            algo,
            extra_updates=int(extra_critic_updates_per_outer_epoch),
            scope=str(extra_critic_update_scope),
        )
        _phase3_profile_timing(f"outer_epoch={outer_idx + 1} extra_critic_updates", t_extra)
        history_row = {
            "outer_epoch": int(outer_idx + 1),
            "actor_baseline_mode": str(baseline_mode_for_epoch),
            "critic_raw_corr": float(critic_quality["raw_corr"]),
            "critic_explained_variance_normalized": float(critic_quality["explained_variance_normalized"]),
            "critic_explained_variance_raw": float(critic_quality["explained_variance_raw"]),
            "critic_objective_total": int(critic_quality["objective_total"]),
            "critic_normalized_value_std": float(critic_quality["normalized_value_std"]),
            "critic_normalized_return_std": float(critic_quality["normalized_return_std"]),
            "critic_raw_value_std": float(critic_quality["raw_value_std"]),
            "critic_raw_return_std": float(critic_quality["raw_return_std"]),
            "switches_to_learned_next": bool(switched_on),
        }
        if bool(record_post_train_rollout_critic_quality):
            post_train_quality = _measure_current_policy_rollout_critic_quality(algo)
            history_row["critic_raw_corr_post_train_rollout"] = float(post_train_quality["raw_corr"])
            history_row["critic_explained_variance_normalized_post_train_rollout"] = float(
                post_train_quality["explained_variance_normalized"]
            )
            history_row["critic_explained_variance_raw_post_train_rollout"] = float(
                post_train_quality["explained_variance_raw"]
            )
            history_row["critic_objective_total_post_train_rollout"] = int(post_train_quality["objective_total"])
            history_row["critic_normalized_value_std_post_train_rollout"] = float(
                post_train_quality["normalized_value_std"]
            )
            history_row["critic_normalized_return_std_post_train_rollout"] = float(
                post_train_quality["normalized_return_std"]
            )
            history_row["critic_raw_value_std_post_train_rollout"] = float(
                post_train_quality["raw_value_std"]
            )
            history_row["critic_raw_return_std_post_train_rollout"] = float(
                post_train_quality["raw_return_std"]
            )
        history.append(history_row)
    algo._update_current_progress_remaining(total_timesteps, total_timesteps)
    return history


def _apply_boundary_contract_mode_flags(prior, boundary_contract_mode: str) -> None:
    mode = str(boundary_contract_mode).strip().lower()
    setattr(prior, "_audit_force_episode_reset_at_sep", mode == "force_episode_reset_at_sep")
    setattr(
        prior,
        "_audit_reset_env_state_at_sep_keep_actor_history",
        mode in {
            "reset_env_state_at_sep_keep_actor_history",
            "reset_env_state_at_sep_and_clear_first_eval_history",
            "reset_env_state_at_sep_and_reset_hidden_keep_actor_history",
        },
    )


def _apply_sep_state_reset_flag(prior, enabled: bool) -> None:
    setattr(prior, "_rwkv_reset_env_state_at_sep_keep_actor_history", bool(enabled))


def _build_audit_ppo_policy_step_fn(
    policy,
    *,
    sampled: bool,
    single_eval_pos: int | None = None,
    boundary_contract_mode: str = "normal",
    deterministic_actor_sampling: bool = False,
):
    base_policy_step_fn = policy.make_vectorized_rollout_step_fn()

    def _clear_buffers():
        reset_fn = getattr(policy, "reset_rollout_cache", None)
        if callable(reset_fn):
            batch_size = int(getattr(policy, "_rollout_cache_batch_size", 1) or 1)
            reset_fn(batch_size)
        return None

    def _policy_step_fn(obs_t, action_t, reward_t, reward_mask_t, cache, step_idx, env_info):
        batch_size = int(obs_t.shape[0])
        env_info_aug = dict(env_info)
        reward_t_local = reward_t
        reward_mask_t_local = reward_mask_t
        action_t_local = action_t
        cache_local = cache
        boundary_contract_mode_resolved = str(boundary_contract_mode).strip().lower()
        is_first_eval_step = (
            single_eval_pos is not None and int(step_idx) == int(single_eval_pos)
        )
        if is_first_eval_step:
            if boundary_contract_mode_resolved == "phase_lag_one_step":
                if "phase_t" in env_info_aug:
                    phase_t = env_info_aug["phase_t"]
                    if torch.is_tensor(phase_t):
                        env_info_aug["phase_t"] = torch.zeros_like(phase_t)
                    else:
                        env_info_aug["phase_t"] = 0.0
            elif boundary_contract_mode_resolved == "clear_first_eval_history":
                reward_t_local = torch.zeros_like(reward_t)
                reward_mask_t_local = torch.zeros_like(reward_mask_t)
                action_t_local = torch.zeros_like(action_t)
                if "terminal_t" in env_info_aug:
                    terminal_t = env_info_aug["terminal_t"]
                    if torch.is_tensor(terminal_t):
                        env_info_aug["terminal_t"] = torch.zeros_like(terminal_t)
                    else:
                        env_info_aug["terminal_t"] = 0.0
            elif boundary_contract_mode_resolved == "clear_first_eval_reward_terminal_history":
                reward_t_local = torch.zeros_like(reward_t)
                reward_mask_t_local = torch.zeros_like(reward_mask_t)
                if "terminal_t" in env_info_aug:
                    terminal_t = env_info_aug["terminal_t"]
                    if torch.is_tensor(terminal_t):
                        env_info_aug["terminal_t"] = torch.zeros_like(terminal_t)
                    else:
                        env_info_aug["terminal_t"] = 0.0
            elif boundary_contract_mode_resolved == "reset_hidden_at_sep":
                cache_local = None
            elif boundary_contract_mode_resolved == "reset_env_state_at_sep_and_reset_hidden_keep_actor_history":
                cache_local = None
            elif boundary_contract_mode_resolved == "reset_hidden_and_clear_first_eval_history":
                cache_local = None
                reward_t_local = torch.zeros_like(reward_t)
                reward_mask_t_local = torch.zeros_like(reward_mask_t)
                action_t_local = torch.zeros_like(action_t)
                if "terminal_t" in env_info_aug:
                    terminal_t = env_info_aug["terminal_t"]
                    if torch.is_tensor(terminal_t):
                        env_info_aug["terminal_t"] = torch.zeros_like(terminal_t)
                    else:
                        env_info_aug["terminal_t"] = 0.0
            elif boundary_contract_mode_resolved == "force_episode_reset_at_sep":
                pass
            elif boundary_contract_mode_resolved == "reset_env_state_at_sep_keep_actor_history":
                pass
            elif boundary_contract_mode_resolved == "reset_env_state_at_sep_and_clear_first_eval_history":
                reward_t_local = torch.zeros_like(reward_t)
                reward_mask_t_local = torch.zeros_like(reward_mask_t)
                action_t_local = torch.zeros_like(action_t)
                if "terminal_t" in env_info_aug:
                    terminal_t = env_info_aug["terminal_t"]
                    if torch.is_tensor(terminal_t):
                        env_info_aug["terminal_t"] = torch.zeros_like(terminal_t)
                    else:
                        env_info_aug["terminal_t"] = 0.0
            elif boundary_contract_mode_resolved != "normal":
                raise ValueError(f"Unsupported boundary_contract_mode: {boundary_contract_mode}")
        if "action_dim_per_sample" not in env_info_aug:
            action_dim = env_info_aug.get("action_dim", int(action_t_local.shape[-1]))
            if torch.is_tensor(action_dim):
                action_dim_t = action_dim.to(device=obs_t.device, dtype=torch.long).reshape(-1)
                if int(action_dim_t.numel()) == 1 and batch_size != 1:
                    action_dim_t = action_dim_t.expand(batch_size)
            else:
                action_dim_t = torch.full(
                    (batch_size,),
                    int(action_dim),
                    device=obs_t.device,
                    dtype=torch.long,
                )
            env_info_aug["action_dim_per_sample"] = action_dim_t
        return base_policy_step_fn(
            obs_t,
            action_t_local,
            reward_t_local,
            reward_mask_t_local,
            cache_local,
            step_idx,
            env_info_aug,
        )

    if sampled:
        if bool(deterministic_actor_sampling):
            _policy_step_fn._policy_actor_sample_fn = (
                lambda actor_outputs, noise: actor_outputs["action_mean"]
            )
        else:
            sample_fn = getattr(base_policy_step_fn, "_policy_actor_sample_fn", None)
            if callable(sample_fn):
                _policy_step_fn._policy_actor_sample_fn = sample_fn
    _policy_step_fn._clear_buffers = _clear_buffers
    return _policy_step_fn


def _run_eval(
    *,
    prior: EnvironmentPrior,
    suite: dict,
    policy_step_fn,
    n_samples: int,
    num_features: int,
    single_eval_pos: int,
    device: torch.device,
    rollout_backend: str = "serial",
):
    return evaluate_prior_suite(
        prior,
        suite=suite,
        policy_step_fn=policy_step_fn,
        n_samples=int(n_samples),
        num_features=int(num_features),
        single_eval_pos=int(single_eval_pos),
        device=device,
        rollout_backend=str(rollout_backend),
    )


def _build_audit_env_cfg(base_env_cfg: dict) -> dict:
    env_cfg = copy.deepcopy(base_env_cfg)
    env_cfg["batch_parallel_backend"] = "python_thread"
    env_cfg["batch_shared_environment"] = False
    env_cfg["reward_dropout_enabled"] = False
    env_cfg["reward_dropout_randomize"] = False
    env_cfg["reward_dropout_ratio"] = 0.0
    env_cfg["reinforce_reward_transform"] = "none"
    env_cfg["action_noise_train_std"] = 0.0
    env_cfg["action_noise_eval_std"] = 0.0
    env_cfg["normalized_q_value_weight"] = 0.0
    env_cfg["next_state_flow_matching_weight"] = 0.0
    return env_cfg


def run_critic_free_single_env_audit(
    *,
    checkpoint_path: str,
    device: str | None = None,
    frozen_h_seed: int = 12345,
    train_env_seed: int = 2020,
    eval_env_seed_start: int = 3000,
    eval_rollout_seed_start: int = 4000,
    eval_env_count: int = 8,
    single_eval_pos: int = 64,
    randomize_single_eval_pos: bool = False,
    n_steps: int = 256,
    outer_epochs: int = 2,
    learning_rate: float = 2e-4,
    batch_size: int = 256,
    n_epochs: int = 1,
    target_kl: float = 0.03,
    strict_fixed_env_mode: bool = False,
    train_rollout_seed: int | None = None,
    deterministic_actor_sampling: bool = False,
    deterministic_batch_plan: bool = True,
    strict_native_rollout: bool = False,
    restore_validation_policy_state: bool = True,
    restore_validation_policy_head_state: bool = False,
    build_seed: int | None = None,
    frozen_h_seed_mode: str = "current",
    ppo_reset_env_state_at_sep: bool = False,
    ppo_separate_value_backbone: bool = False,
    extra_critic_updates_per_outer_epoch: int = 0,
    extra_critic_update_scope: str = "shared",
    actor_gae_space: str = "normalized",
    actor_objective_mode: str = "tokenwise",
    boundary_contract_mode: str = "normal",
    critic_warmup_min_outer_epochs: int = 0,
    critic_warmup_raw_corr_threshold: float = 0.0,
    record_suite_critic_quality: bool = False,
    record_post_train_rollout_critic_quality: bool = False,
    vf_coef_override: float | None = None,
    policy_loss_coef_override: float | None = None,
    value_head_impl: str = "legacy_bar",
    value_path_adapter_impl: str = "none",
    modes: list[str] | None = None,
) -> dict:
    device_obj = torch.device(str(device or _default_device()))
    resolved_build_seed = (
        int(build_seed)
        if build_seed is not None
        else int(train_rollout_seed if train_rollout_seed is not None else train_env_seed)
    )
    modes = list(modes or ["learned", "zero", "rollout_mean"])
    eval_env_seeds = list(range(int(eval_env_seed_start), int(eval_env_seed_start) + int(eval_env_count)))
    eval_rollout_seeds = list(
        range(int(eval_rollout_seed_start), int(eval_rollout_seed_start) + int(eval_env_count))
    )

    load_model.cache_clear()
    _, config = load_model(checkpoint_path, device=device_obj, verbose=False)
    num_features = int(config["prior"]["num_features"])
    env_cfg = _build_audit_env_cfg(config["prior"]["environment"])

    fixed_env_bundle = _prepare_audit_fixed_env_contract(
        env_cfg=env_cfg,
        boundary_contract_mode=str(boundary_contract_mode),
        ppo_reset_env_state_at_sep=bool(ppo_reset_env_state_at_sep),
        frozen_h_seed=int(frozen_h_seed),
        frozen_h_seed_mode=str(frozen_h_seed_mode),
    )
    prior_for_h = fixed_env_bundle["prior_for_h"]
    frozen_h = fixed_env_bundle["frozen_h"]

    suite = _make_suite(
        frozen_h=frozen_h,
        env_seeds=eval_env_seeds,
        rollout_seeds=eval_rollout_seeds,
    )

    eval_single_eval_pos = int(
        prior_for_h._sample_single_eval_pos(int(n_steps), None)
        if bool(randomize_single_eval_pos)
        else int(single_eval_pos)
    )

    zero_metrics = _run_eval(
        prior=prior_for_h,
        suite=suite,
        policy_step_fn=_build_zero_policy_step_fn(),
        n_samples=n_steps,
        num_features=num_features,
        single_eval_pos=eval_single_eval_pos,
        device=device_obj,
    )
    zero_suffix_mean = float(zero_metrics["suffix_return_mean"])

    report = {
        "audit_entry": "critic_free_single_env_audit",
        "checkpoint_path": str(Path(checkpoint_path).expanduser().resolve()),
        "device": str(device_obj),
        "frozen_h_seed": int(frozen_h_seed),
        "fixed_env_contract": dict(fixed_env_bundle["fixed_env_contract"]),
        "train_env_seed": int(train_env_seed),
        "eval_env_seeds": eval_env_seeds,
        "eval_rollout_seeds": eval_rollout_seeds,
        "single_eval_pos": int(single_eval_pos),
        "randomize_single_eval_pos": bool(randomize_single_eval_pos),
        "eval_single_eval_pos": int(eval_single_eval_pos),
        "n_steps": int(n_steps),
        "outer_epochs": int(outer_epochs),
        "learning_rate": float(learning_rate),
        "batch_size": int(batch_size),
        "n_epochs": int(n_epochs),
        "target_kl": float(target_kl),
        "strict_fixed_env_mode": bool(strict_fixed_env_mode),
        "train_rollout_seed": None if train_rollout_seed is None else int(train_rollout_seed),
        "deterministic_actor_sampling": bool(deterministic_actor_sampling),
        "deterministic_batch_plan": bool(deterministic_batch_plan),
        "strict_native_rollout": bool(strict_native_rollout),
        "restore_validation_policy_state": bool(restore_validation_policy_state),
        "restore_validation_policy_head_state": bool(restore_validation_policy_head_state),
        "build_seed": int(resolved_build_seed),
        "frozen_h_seed_mode": str(frozen_h_seed_mode),
        "ppo_reset_env_state_at_sep": bool(ppo_reset_env_state_at_sep),
        "ppo_separate_value_backbone": bool(ppo_separate_value_backbone),
        "extra_critic_updates_per_outer_epoch": int(extra_critic_updates_per_outer_epoch),
        "extra_critic_update_scope": str(extra_critic_update_scope),
        "actor_gae_space": str(actor_gae_space),
        "actor_objective_mode": str(actor_objective_mode),
        "boundary_contract_mode": str(boundary_contract_mode),
        "critic_warmup_min_outer_epochs": int(critic_warmup_min_outer_epochs),
        "critic_warmup_raw_corr_threshold": float(critic_warmup_raw_corr_threshold),
        "record_suite_critic_quality": bool(record_suite_critic_quality),
        "record_post_train_rollout_critic_quality": bool(record_post_train_rollout_critic_quality),
        "vf_coef_override": None if vf_coef_override is None else float(vf_coef_override),
        "policy_loss_coef_override": (
            None if policy_loss_coef_override is None else float(policy_loss_coef_override)
        ),
        "value_head_impl": str(value_head_impl),
        "value_path_adapter_impl": str(value_path_adapter_impl),
        "zero_suffix_return_mean": float(zero_suffix_mean),
        "modes": {},
    }

    for mode in modes:
        _seed_all(int(resolved_build_seed))
        load_model.cache_clear()
        model, config = load_model(checkpoint_path, device=device_obj, verbose=False)
        prior = EnvironmentPrior(copy.deepcopy(env_cfg))
        prior._sample_batch_hypers = lambda batch_n, _h=frozen_h: _clone_h_repeated(_h, int(batch_n))
        if not bool(randomize_single_eval_pos):
            prior._sample_single_eval_pos = lambda n_samples_arg, rng_seed=None: int(single_eval_pos)
        _apply_boundary_contract_mode_flags(prior, boundary_contract_mode)
        _apply_sep_state_reset_flag(prior, bool(ppo_reset_env_state_at_sep))

        resolved_mode = str(mode).strip().lower()
        build_actor_baseline_mode = "learned" if resolved_mode == "warmup" else resolved_mode

        algo, callback, vec_env = build_recurrent_ppo(
            model=model,
            env_prior=prior,
            device=str(device_obj),
            num_features=num_features,
            n_envs=1,
            n_steps=int(n_steps),
            learning_rate=float(learning_rate),
            batch_size=int(batch_size),
            n_epochs=int(n_epochs),
            gamma=float(config["optimizer"].get("ppo_gamma", 0.98)),
            gae_lambda=float(config["optimizer"].get("ppo_gae_lambda", 0.90)),
            clip_range=config["optimizer"].get("ppo_clip_range", 0.2),
            clip_range_vf=None,
            normalize_advantage=False,
            actor_gae_space=str(actor_gae_space),
            actor_baseline_mode=str(build_actor_baseline_mode),
            actor_objective_mode=str(actor_objective_mode),
            reset_env_state_at_sep=bool(ppo_reset_env_state_at_sep),
            separate_value_backbone=bool(ppo_separate_value_backbone),
            ent_coef=float(config["optimizer"].get("ppo_ent_coef", 0.0)),
            vf_coef=_resolve_vf_coef(config, vf_coef_override),
            max_grad_norm=float(config["optimizer"].get("ppo_max_grad_norm", 0.5)),
            target_kl=float(target_kl),
            strict_fixed_env_mode=bool(strict_fixed_env_mode),
            env_rng_seeds=([int(train_env_seed)] if bool(strict_fixed_env_mode) else None),
            rollout_rng_seeds=(
                [int(train_rollout_seed if train_rollout_seed is not None else train_env_seed)]
                if bool(strict_fixed_env_mode)
                else None
            ),
            deterministic_actor_sampling=bool(deterministic_actor_sampling),
            deterministic_batch_plan=bool(deterministic_batch_plan),
            strict_native_rollout=bool(strict_native_rollout),
            restore_validation_policy_state=bool(restore_validation_policy_state),
            value_head_impl=str(value_head_impl),
            value_path_adapter_impl=str(value_path_adapter_impl),
            verbose=0,
        )
        algo._rwkv_policy_loss_coef = _resolve_policy_loss_coef(policy_loss_coef_override)
        if bool(restore_validation_policy_head_state):
            saved_policy_state = getattr(model, "_validation_ppo_policy_state", None)
            if not isinstance(saved_policy_state, dict):
                raise ValueError(
                    "restore_validation_policy_head_state=True requires model._validation_ppo_policy_state to be present."
                )
            _restore_saved_recurrent_ppo_policy_head_state(algo.policy, saved_policy_state)
        vec_env._sample_seed_list = lambda: [int(train_env_seed)]
        if bool(deterministic_batch_plan):
            _make_deterministic_batch_plan(algo.rollout_buffer)

        det_pre = _run_eval(
            prior=prior,
            suite=suite,
            policy_step_fn=_build_audit_ppo_policy_step_fn(
                algo.policy,
                sampled=False,
                single_eval_pos=eval_single_eval_pos,
                boundary_contract_mode=boundary_contract_mode,
                deterministic_actor_sampling=bool(deterministic_actor_sampling),
            ),
            n_samples=n_steps,
            num_features=num_features,
            single_eval_pos=eval_single_eval_pos,
            device=device_obj,
        )
        smp_pre = _run_eval(
            prior=prior,
            suite=suite,
            policy_step_fn=_build_audit_ppo_policy_step_fn(
                algo.policy,
                sampled=True,
                single_eval_pos=eval_single_eval_pos,
                boundary_contract_mode=boundary_contract_mode,
                deterministic_actor_sampling=bool(deterministic_actor_sampling),
            ),
            n_samples=n_steps,
            num_features=num_features,
            single_eval_pos=eval_single_eval_pos,
            device=device_obj,
        )

        warmup_history = _audit_train_loop(
            algo,
            callback,
            vec_env,
            outer_epochs=int(outer_epochs),
            extra_critic_updates_per_outer_epoch=int(extra_critic_updates_per_outer_epoch),
            extra_critic_update_scope=str(extra_critic_update_scope),
            critic_warmup=(
                {
                    "min_outer_epochs": int(critic_warmup_min_outer_epochs),
                    "raw_corr_threshold": float(critic_warmup_raw_corr_threshold),
                }
                if resolved_mode == "warmup"
                else None
            ),
            record_post_train_rollout_critic_quality=bool(record_post_train_rollout_critic_quality),
        )
        if bool(deterministic_batch_plan):
            _make_deterministic_batch_plan(algo.rollout_buffer)

        det_post = _run_eval(
            prior=prior,
            suite=suite,
            policy_step_fn=_build_audit_ppo_policy_step_fn(
                algo.policy,
                sampled=False,
                single_eval_pos=eval_single_eval_pos,
                boundary_contract_mode=boundary_contract_mode,
                deterministic_actor_sampling=bool(deterministic_actor_sampling),
            ),
            n_samples=n_steps,
            num_features=num_features,
            single_eval_pos=eval_single_eval_pos,
            device=device_obj,
        )
        smp_post = _run_eval(
            prior=prior,
            suite=suite,
            policy_step_fn=_build_audit_ppo_policy_step_fn(
                algo.policy,
                sampled=True,
                single_eval_pos=eval_single_eval_pos,
                boundary_contract_mode=boundary_contract_mode,
                deterministic_actor_sampling=bool(deterministic_actor_sampling),
            ),
            n_samples=n_steps,
            num_features=num_features,
            single_eval_pos=eval_single_eval_pos,
            device=device_obj,
        )

        mode_report = {
            "effective_actor_gae_space": str(getattr(algo, "_rwkv_actor_gae_space", actor_gae_space)),
            "space_contract": getattr(algo, "_rwkv_space_contract", None),
            "pre_det_gap": float(det_pre["suffix_return_mean"] - zero_suffix_mean),
            "post_det_gap": float(det_post["suffix_return_mean"] - zero_suffix_mean),
            "delta_det_gap": float(det_post["suffix_return_mean"] - det_pre["suffix_return_mean"]),
            "pre_smp_gap": float(smp_pre["suffix_return_mean"] - zero_suffix_mean),
            "post_smp_gap": float(smp_post["suffix_return_mean"] - zero_suffix_mean),
            "delta_smp_gap": float(smp_post["suffix_return_mean"] - smp_pre["suffix_return_mean"]),
            "critic_train_history": warmup_history,
        }
        if resolved_mode == "warmup":
            mode_report["critic_warmup_history"] = warmup_history
        if bool(record_suite_critic_quality):
            policy_state_dict = _snapshot_policy_state_dict(algo.policy)
            train_quality_suite = _make_suite(
                frozen_h=frozen_h,
                env_seeds=[int(train_env_seed)],
                rollout_seeds=[
                    int(train_rollout_seed if train_rollout_seed is not None else train_env_seed)
                ],
            )
            mode_report["train_suite_critic_quality_after_train"] = _measure_suite_critic_quality_with_policy_state(
                checkpoint_path=checkpoint_path,
                device_obj=device_obj,
                env_cfg=env_cfg,
                num_features=num_features,
                suite=train_quality_suite,
                single_eval_pos=eval_single_eval_pos,
                n_steps=n_steps,
                learning_rate=learning_rate,
                batch_size=batch_size,
                n_epochs=n_epochs,
                target_kl=target_kl,
                ppo_reset_env_state_at_sep=bool(ppo_reset_env_state_at_sep),
                ppo_separate_value_backbone=bool(ppo_separate_value_backbone),
                deterministic_actor_sampling=bool(deterministic_actor_sampling),
                deterministic_batch_plan=bool(deterministic_batch_plan),
                strict_native_rollout=bool(strict_native_rollout),
                actor_gae_space=str(actor_gae_space),
                actor_baseline_mode=str(build_actor_baseline_mode),
                actor_objective_mode=str(actor_objective_mode),
                boundary_contract_mode=str(boundary_contract_mode),
                policy_state_dict=policy_state_dict,
                    build_seed=int(resolved_build_seed),
                    vf_coef_override=vf_coef_override,
                    policy_loss_coef_override=policy_loss_coef_override,
                    value_head_impl=str(value_head_impl),
                    value_path_adapter_impl=str(value_path_adapter_impl),
                )
            mode_report["eval_suite_critic_quality_after_train"] = _measure_suite_critic_quality_with_policy_state(
                checkpoint_path=checkpoint_path,
                device_obj=device_obj,
                env_cfg=env_cfg,
                num_features=num_features,
                suite=suite,
                single_eval_pos=eval_single_eval_pos,
                n_steps=n_steps,
                learning_rate=learning_rate,
                batch_size=batch_size,
                n_epochs=n_epochs,
                target_kl=target_kl,
                ppo_reset_env_state_at_sep=bool(ppo_reset_env_state_at_sep),
                ppo_separate_value_backbone=bool(ppo_separate_value_backbone),
                deterministic_actor_sampling=bool(deterministic_actor_sampling),
                deterministic_batch_plan=bool(deterministic_batch_plan),
                strict_native_rollout=bool(strict_native_rollout),
                actor_gae_space=str(actor_gae_space),
                actor_baseline_mode=str(build_actor_baseline_mode),
                actor_objective_mode=str(actor_objective_mode),
                boundary_contract_mode=str(boundary_contract_mode),
                policy_state_dict=policy_state_dict,
                build_seed=int(resolved_build_seed),
                vf_coef_override=vf_coef_override,
                policy_loss_coef_override=policy_loss_coef_override,
                value_head_impl=str(value_head_impl),
                value_path_adapter_impl=str(value_path_adapter_impl),
            )
        report["modes"][str(mode)] = mode_report

        vec_env.close()
        del algo, callback, vec_env, model, prior
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return report


def main():
    parser = argparse.ArgumentParser(description="GPU-first critic-free single-environment PPO audit")
    parser.add_argument("checkpoint_path")
    parser.add_argument("--device", default=None)
    parser.add_argument("--frozen-h-seed", type=int, default=12345)
    parser.add_argument("--train-env-seed", type=int, default=2020)
    parser.add_argument("--eval-env-seed-start", type=int, default=3000)
    parser.add_argument("--eval-rollout-seed-start", type=int, default=4000)
    parser.add_argument("--eval-env-count", type=int, default=8)
    parser.add_argument("--single-eval-pos", type=int, default=64)
    parser.add_argument("--randomize-single-eval-pos", action="store_true")
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--outer-epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=1)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument("--strict-fixed-env-mode", action="store_true")
    parser.add_argument("--train-rollout-seed", type=int, default=None)
    parser.add_argument("--frozen-h-seed-mode", default="current")
    parser.add_argument("--deterministic-actor-sampling", action="store_true")
    parser.add_argument("--no-deterministic-batch-plan", action="store_true")
    parser.add_argument("--strict-native-rollout", action="store_true")
    parser.add_argument("--no-restore-validation-policy-state", action="store_true")
    parser.add_argument("--restore-validation-policy-head-state", action="store_true")
    parser.add_argument("--build-seed", type=int, default=None)
    parser.add_argument("--ppo-reset-env-state-at-sep", action="store_true")
    parser.add_argument("--ppo-separate-value-backbone", action="store_true")
    parser.add_argument("--extra-critic-updates-per-outer-epoch", type=int, default=0)
    parser.add_argument("--extra-critic-update-scope", default="shared")
    parser.add_argument("--actor-gae-space", default="normalized")
    parser.add_argument("--actor-objective-mode", default="tokenwise")
    parser.add_argument("--boundary-contract-mode", default="normal")
    parser.add_argument("--critic-warmup-min-outer-epochs", type=int, default=0)
    parser.add_argument("--critic-warmup-raw-corr-threshold", type=float, default=0.0)
    parser.add_argument("--record-suite-critic-quality", action="store_true")
    parser.add_argument("--record-post-train-rollout-critic-quality", action="store_true")
    parser.add_argument("--vf-coef-override", type=float, default=None)
    parser.add_argument("--policy-loss-coef-override", type=float, default=None)
    parser.add_argument("--value-head-impl", default="legacy_bar")
    parser.add_argument("--value-path-adapter-impl", default="none")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["learned", "zero", "rollout_mean"],
    )
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    report = run_critic_free_single_env_audit(
        checkpoint_path=args.checkpoint_path,
        device=args.device,
        frozen_h_seed=args.frozen_h_seed,
        train_env_seed=args.train_env_seed,
        eval_env_seed_start=args.eval_env_seed_start,
        eval_rollout_seed_start=args.eval_rollout_seed_start,
        eval_env_count=args.eval_env_count,
        single_eval_pos=args.single_eval_pos,
        randomize_single_eval_pos=bool(args.randomize_single_eval_pos),
        n_steps=args.n_steps,
        outer_epochs=args.outer_epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        target_kl=args.target_kl,
        strict_fixed_env_mode=bool(args.strict_fixed_env_mode),
        train_rollout_seed=args.train_rollout_seed,
        frozen_h_seed_mode=str(args.frozen_h_seed_mode),
        deterministic_actor_sampling=bool(args.deterministic_actor_sampling),
        deterministic_batch_plan=not bool(args.no_deterministic_batch_plan),
        strict_native_rollout=bool(args.strict_native_rollout),
        restore_validation_policy_state=not bool(args.no_restore_validation_policy_state),
        restore_validation_policy_head_state=bool(args.restore_validation_policy_head_state),
        build_seed=args.build_seed,
        ppo_reset_env_state_at_sep=bool(args.ppo_reset_env_state_at_sep),
        ppo_separate_value_backbone=bool(args.ppo_separate_value_backbone),
        extra_critic_updates_per_outer_epoch=int(args.extra_critic_updates_per_outer_epoch),
        extra_critic_update_scope=str(args.extra_critic_update_scope),
        actor_gae_space=args.actor_gae_space,
        actor_objective_mode=args.actor_objective_mode,
        boundary_contract_mode=args.boundary_contract_mode,
        critic_warmup_min_outer_epochs=int(args.critic_warmup_min_outer_epochs),
        critic_warmup_raw_corr_threshold=float(args.critic_warmup_raw_corr_threshold),
        record_suite_critic_quality=bool(args.record_suite_critic_quality),
        record_post_train_rollout_critic_quality=bool(args.record_post_train_rollout_critic_quality),
        vf_coef_override=args.vf_coef_override,
        policy_loss_coef_override=args.policy_loss_coef_override,
        value_head_impl=str(args.value_head_impl),
        value_path_adapter_impl=str(args.value_path_adapter_impl),
        modes=list(args.modes),
    )
    payload = json.dumps(report, sort_keys=True)
    if args.output_json:
        out_path = Path(args.output_json).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
