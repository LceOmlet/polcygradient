import argparse
import copy
import json
import math
import random
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None  # type: ignore

if cv2 is not None and not hasattr(cv2, "ocl"):
    class _OpenCVOclCompat:
        @staticmethod
        def setUseOpenCL(_enabled):
            return None

    cv2.ocl = _OpenCVOclCompat()  # type: ignore[attr-defined]

from stable_baselines3.common.logger import configure as configure_logger

from ticl.analysis.critic_free_single_env_audit import (
    _apply_boundary_contract_mode_flags,
    _apply_sep_state_reset_flag,
    _build_audit_env_cfg,
    _make_deterministic_batch_plan,
)
from ticl.analysis.phase2_legacy_bar_longrun_probe import (
    _build_algo as _build_legacy_algo,
    _load_full_frozen_h,
    _sanitize_loaded_frozen_h_for_batch_use,
)
from ticl.analysis.phase3_phase2_milestone_contract_gradient_compare import (
    _install_batch1_non_native_compare_shim,
    _max_abs_diff_tensor,
    _param_diff_summary,
    _sync_last_obs,
)
from ticl.config_utils import str2bool
from ticl.model_builder import get_model
from ticl.model_configs import get_model_default_config
from ticl.priors.environment_prior import EnvironmentPrior
from ticl.rlpfn_maintained_path import validate_rlpfn_maintained_path_config
from ticl.sb3_recurrent_ppo import _resolve_actor_advantages, build_recurrent_ppo
from ticl.train import _resolve_recurrent_ppo_training_bridge_kwargs


DEFAULT_FIXED_FROZEN_H_JSON = (
    "/home/chen/RLPFN/artifacts/phase2_single_env_identity_rejection_gain058_seed26026/full_frozen_h.json"
)
DEFAULT_OUTPUT_JSON = (
    "/home/chen/RLPFN/artifacts/phase2_fit_model_migration_gradient_compare/"
    "phase2_fit_model_migration_gradient_compare.json"
)


def _default_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _seed_all(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return _jsonable(value.detach().cpu().item())
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _load_default_config() -> dict[str, Any]:
    config = get_model_default_config("rlpfn")
    validate_rlpfn_maintained_path_config(config)
    return config


def _apply_transition_memory_contract(env_cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    for key in (
        "transition_inner_grouping",
        "transition_inner_min_bucket",
        "reference_scm_partition_max_bytes",
        "reference_scm_memory_guard_fraction",
        "transition_generator_rebuild_each_step",
    ):
        value = getattr(args, key, None)
        if value is not None:
            env_cfg[key] = value
    return env_cfg


def _build_model_from_scratch(*, config: dict[str, Any], device_obj: torch.device, build_seed: int):
    _seed_all(int(build_seed))
    _, model, _, _ = get_model(
        copy.deepcopy(config),
        device=str(device_obj),
        should_train=False,
        verbose=False,
    )
    return model


def _make_fixed_env_prior(
    *,
    base_env_cfg: dict[str, Any],
    fixed_frozen_h_json: str,
    single_eval_pos: int,
    ppo_reset_env_state_at_sep: bool,
) -> EnvironmentPrior:
    env_cfg = copy.deepcopy(base_env_cfg)
    env_cfg["fixed_frozen_h_json"] = str(fixed_frozen_h_json)
    env_cfg["batch_parallel_backend"] = "torch_vectorized"
    env_cfg["batch_vectorized_grouping"] = "family"
    env_cfg["reward_state_input_gain_fraction_rejection_min"] = 0.0
    prior = EnvironmentPrior(env_cfg)
    prior._sample_single_eval_pos = lambda n_samples_arg, rng_seed=None: int(single_eval_pos)
    _apply_boundary_contract_mode_flags(prior, "normal")
    _apply_sep_state_reset_flag(prior, bool(ppo_reset_env_state_at_sep))
    return prior


def _make_nonfixed_env_prior(
    *,
    base_env_cfg: dict[str, Any],
    single_eval_pos: int,
    ppo_reset_env_state_at_sep: bool,
    conditioned_min: float,
) -> EnvironmentPrior:
    env_cfg = copy.deepcopy(base_env_cfg)
    env_cfg.pop("fixed_frozen_h_json", None)
    env_cfg["batch_parallel_backend"] = "torch_vectorized"
    env_cfg["batch_vectorized_grouping"] = "family"
    env_cfg["reward_state_input_gain_fraction_rejection_min"] = 0.0
    env_cfg["reward_state_input_gain_fraction_conditioned_sampling_enabled"] = True
    env_cfg["reward_state_input_gain_fraction_conditioned_min"] = float(conditioned_min)
    prior = EnvironmentPrior(env_cfg)
    prior._sample_single_eval_pos = lambda n_samples_arg, rng_seed=None: int(single_eval_pos)
    _apply_boundary_contract_mode_flags(prior, "normal")
    _apply_sep_state_reset_flag(prior, bool(ppo_reset_env_state_at_sep))
    return prior


def _build_fit_model_bridge_algo(
    *,
    config: dict[str, Any],
    device_obj: torch.device,
    env_prior: EnvironmentPrior,
    build_seed: int,
    train_env_seed: int | None,
    train_rollout_seed: int | None,
    n_envs: int,
    n_steps: int,
    batch_size: int,
    n_epochs: int,
    learning_rate: float,
    target_kl: float,
    ppo_reset_env_state_at_sep: bool,
    actor_baseline_mode: str,
    strict_fixed_env_mode: bool,
    deterministic_actor_sampling: bool,
    deterministic_batch_plan: bool,
    strict_native_rollout: bool,
    value_head_impl: str,
    value_path_adapter_impl: str,
    value_head_mlp_hidden_dim: int,
    space_contract: str,
    normalize_advantage: bool,
    vf_coef: float,
    ppo_separate_value_backbone: bool,
):
    model = _build_model_from_scratch(
        config=config,
        device_obj=device_obj,
        build_seed=int(build_seed),
    )
    optimizer_cfg = dict(config.get("optimizer", {}))
    build_kwargs, bridge_summary = _resolve_recurrent_ppo_training_bridge_kwargs(
        model=model,
        env_prior=env_prior,
        device=str(device_obj),
        num_features=int(config["prior"]["num_features"]),
        n_envs=int(n_envs),
        n_steps=int(n_steps),
        learning_rate=float(learning_rate),
        batch_size=int(batch_size),
        n_epochs=int(n_epochs),
        gamma=float(optimizer_cfg.get("ppo_gamma", 0.98)),
        gae_lambda=float(optimizer_cfg.get("ppo_gae_lambda", 0.90)),
        clip_range=optimizer_cfg.get("ppo_clip_range", 0.2),
        clip_range_vf=None,
        normalize_advantage=bool(normalize_advantage),
        space_contract=str(space_contract),
        value_target_space=None,
        actor_gae_space=None,
        allow_mixed_space_contract=False,
        actor_baseline_mode=str(actor_baseline_mode),
        separate_value_backbone=bool(ppo_separate_value_backbone),
        reset_env_state_at_sep=bool(ppo_reset_env_state_at_sep),
        value_head_impl=str(value_head_impl),
        value_path_adapter_impl=str(value_path_adapter_impl),
        value_head_mlp_hidden_dim=int(value_head_mlp_hidden_dim),
        runtime_normalized_q_value_weight_override=None,
        runtime_next_state_flow_matching_weight_override=None,
        ent_coef=float(optimizer_cfg.get("ppo_ent_coef", 0.0)),
        vf_coef=float(vf_coef),
        max_grad_norm=float(optimizer_cfg.get("ppo_max_grad_norm", 0.5)),
        target_kl=float(target_kl),
        verbose=0,
        ppo_restore_validation_policy_state=False,
        ppo_strict_fixed_env_mode=bool(strict_fixed_env_mode),
        ppo_env_rng_seeds=(
            [int(train_env_seed)] if train_env_seed is not None else None
        ),
        ppo_rollout_rng_seeds=(
            [int(train_rollout_seed)] if train_rollout_seed is not None else None
        ),
        ppo_deterministic_actor_sampling=bool(deterministic_actor_sampling),
        ppo_deterministic_batch_plan=bool(deterministic_batch_plan),
        ppo_strict_native_rollout=bool(strict_native_rollout),
    )
    algo, callback, vec_env = build_recurrent_ppo(**build_kwargs)
    algo._rwkv_policy_loss_coef = 1.0
    if train_env_seed is not None:
        vec_env._sample_seed_list = lambda: [int(train_env_seed)] * int(vec_env.num_envs)
    if bool(deterministic_batch_plan):
        _make_deterministic_batch_plan(algo.rollout_buffer)
    algo.ep_info_buffer = []
    algo.ep_success_buffer = []
    algo.set_logger(configure_logger(folder=None, format_strings=[]))
    return algo, callback, vec_env, bridge_summary, build_kwargs


def _build_legacy_contract_algo(
    *,
    config: dict[str, Any],
    fixed_frozen_h_json: str,
    device_obj: torch.device,
    build_seed: int,
    train_env_seed: int,
    train_rollout_seed: int,
    single_eval_pos: int,
    n_steps: int,
    batch_size: int,
    n_epochs: int,
    learning_rate: float,
    target_kl: float,
    ppo_reset_env_state_at_sep: bool,
    actor_baseline_mode: str,
    strict_native_rollout: bool,
    value_head_impl: str,
    value_path_adapter_impl: str,
    value_head_mlp_hidden_dim: int,
    space_contract: str,
    ppo_separate_value_backbone: bool,
    vf_coef: float,
    transition_inner_grouping: str | None,
    transition_inner_min_bucket: int | None,
    reference_scm_partition_max_bytes: int | None,
    reference_scm_memory_guard_fraction: float | None,
    transition_generator_rebuild_each_step: bool | None,
):
    frozen_h, loaded_path = _load_full_frozen_h(str(fixed_frozen_h_json))
    frozen_h = _sanitize_loaded_frozen_h_for_batch_use(frozen_h)
    base_env_cfg = _build_audit_env_cfg(config["prior"]["environment"])
    for key, value in (
        ("transition_inner_grouping", transition_inner_grouping),
        ("transition_inner_min_bucket", transition_inner_min_bucket),
        ("reference_scm_partition_max_bytes", reference_scm_partition_max_bytes),
        ("reference_scm_memory_guard_fraction", reference_scm_memory_guard_fraction),
        ("transition_generator_rebuild_each_step", transition_generator_rebuild_each_step),
    ):
        if value is not None:
            base_env_cfg[key] = value
    algo, callback, vec_env = _build_legacy_algo(
        checkpoint_path=None,
        device_obj=device_obj,
        frozen_h=frozen_h,
        env_cfg=base_env_cfg,
        num_features=int(config["prior"]["num_features"]),
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
        ppo_separate_value_backbone=bool(ppo_separate_value_backbone),
        from_scratch=True,
        from_scratch_model_type="rlpfn",
        vf_coef_override=float(vf_coef),
        policy_loss_coef_override=None,
        behavior_policy="learned",
    )
    return algo, callback, vec_env, frozen_h, loaded_path


def _path_coverage_summary(vec_env) -> dict[str, Any]:
    env = getattr(vec_env, "_env", None)
    transition = None if env is None else env.get("transition_generator", None)
    gain = None if env is None else env.get("reward_state_input_gain_fraction", None)
    return {
        "vec_env_class": type(vec_env).__name__,
        "num_envs": int(getattr(vec_env, "num_envs", -1)),
        "env_initialized": env is not None,
        "env_family": None if env is None else str(env.get("env_family", None)),
        "transition_class": None if transition is None else type(transition).__name__,
        "transition_reference_scm_vectorized": bool(
            getattr(transition, "_reference_scm_vectorized", False)
        ),
        "reward_state_input_gain_fraction_present": torch.is_tensor(gain),
        "reward_state_input_gain_fraction": (
            None
            if not torch.is_tensor(gain)
            else [float(v) for v in gain.detach().cpu().flatten().tolist()]
        ),
    }


def _tensor_diff_summary(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> dict[str, Any]:
    per_key: dict[str, float] = {}
    max_abs = 0.0
    for key in sorted(set(a.keys()) | set(b.keys())):
        if key not in a or key not in b:
            raise ValueError(f"rollout tensor key mismatch: {key}")
        diff = _max_abs_diff_tensor(a[key], b[key])
        per_key[str(key)] = float(diff)
        max_abs = max(max_abs, diff)
    return {
        "max_abs_rollout_tensor_delta": float(max_abs),
        "per_tensor_max_abs_delta": per_key,
    }


def _gradient_diff_summary(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    if tuple(a.shape) != tuple(b.shape):
        raise ValueError(f"Gradient shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
    a = a.detach().cpu().to(dtype=torch.float64)
    b = b.detach().cpu().to(dtype=torch.float64)
    delta = a - b
    norm_a = torch.linalg.vector_norm(a).item()
    norm_b = torch.linalg.vector_norm(b).item()
    norm_delta = torch.linalg.vector_norm(delta).item()
    denom = max(float(norm_a * norm_b), 1e-30)
    cosine = float(torch.dot(a, b).item() / denom)
    return {
        "num_grad_values": int(a.numel()),
        "legacy_grad_norm": float(norm_a),
        "fit_model_grad_norm": float(norm_b),
        "grad_delta_l2": float(norm_delta),
        "grad_delta_l2_relative_to_legacy": float(norm_delta / max(norm_a, 1e-30)),
        "grad_delta_max_abs": float(delta.abs().max().item()) if delta.numel() else 0.0,
        "grad_cosine_similarity": float(cosine),
    }


def _loss_diff_summary(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "policy_loss",
        "value_loss",
        "entropy_loss",
        "total_loss",
        "approx_kl",
        "clip_fraction",
        "objective_total",
    ]
    out = {}
    for key in keys:
        if key in a and key in b:
            out[key] = {
                "legacy": _jsonable(a[key]),
                "fit_model": _jsonable(b[key]),
                "delta": (
                    float(b[key]) - float(a[key])
                    if isinstance(a[key], (int, float)) and isinstance(b[key], (int, float))
                    else None
                ),
            }
    return out


def _state_dict_snapshot(policy) -> dict[str, torch.Tensor]:
    return {
        str(key): value.detach().cpu().clone()
        for key, value in policy.state_dict().items()
    }


def _state_dict_diff_summary(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> dict[str, Any]:
    if set(a.keys()) != set(b.keys()):
        raise ValueError("state_dict key mismatch")
    max_abs = 0.0
    differing = 0
    for key in sorted(a.keys()):
        diff = _max_abs_diff_tensor(a[key], b[key])
        max_abs = max(max_abs, diff)
        if diff > 0.0:
            differing += 1
    return {
        "policy_param_max_abs_diff": float(max_abs),
        "num_param_tensors_with_diff": int(differing),
    }


def _flatten_policy_grad(policy: torch.nn.Module) -> torch.Tensor:
    flat_parts = []
    for param in policy.parameters():
        if not param.requires_grad:
            continue
        grad = param.grad
        if grad is None:
            flat_parts.append(torch.zeros(param.numel(), device=param.device, dtype=param.dtype))
        else:
            flat_parts.append(grad.detach().reshape(-1))
    if not flat_parts:
        return torch.zeros((0,), dtype=torch.float32)
    return torch.cat(flat_parts).to(dtype=torch.float32)


def _collect_and_measure_main_loss(algo, callback, vec_env, *, train_env_seed: int) -> dict[str, Any]:
    _sync_last_obs(algo, callback, vec_env, train_env_seed=int(train_env_seed))
    assert algo.collect_rollouts(vec_env, callback, algo.rollout_buffer, n_rollout_steps=algo.n_steps)
    if algo.device.type == "cuda":
        torch.cuda.synchronize(device=algo.device)

    rollout_data = next(algo.rollout_buffer.get_gpu_flat(algo.batch_size))
    objective_mask = rollout_data.objective_masks > 1e-8
    valid_total = objective_mask.to(
        device=rollout_data.returns.device,
        dtype=rollout_data.returns.dtype,
    ).sum().clamp_min(1.0)
    advantages = _resolve_actor_advantages(rollout_data)
    eval_outputs = algo.policy.evaluate_actions_with_hidden_flat(
        rollout_data.observations,
        rollout_data.actions,
        seq_lengths=rollout_data.seq_lengths,
        action_masks=rollout_data.action_masks,
    )
    value_logits = eval_outputs["value_logits"]
    log_prob = eval_outputs["log_prob"]
    entropy = eval_outputs["entropy"]
    clip_range = algo.clip_range(algo._current_progress_remaining)
    ratio = torch.exp(log_prob - rollout_data.old_log_prob)
    policy_loss_1 = advantages * ratio
    policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
    clipped_objective = torch.min(policy_loss_1, policy_loss_2)
    policy_num = (
        clipped_objective
        * objective_mask.to(device=clipped_objective.device, dtype=clipped_objective.dtype)
    ).sum()
    policy_loss = -(policy_num / valid_total.to(device=policy_num.device, dtype=policy_num.dtype))

    value_loss = algo.policy.compute_value_loss_from_logits(
        value_logits=value_logits,
        targets=rollout_data.returns.to(device=value_logits.device, dtype=torch.float32),
        objective_mask=objective_mask,
        value_target_bucket_idx=getattr(rollout_data, "value_target_bucket_idx", None),
    )

    entropy_terms = -log_prob if entropy is None else entropy
    entropy_num = (
        entropy_terms
        * objective_mask.to(device=entropy_terms.device, dtype=entropy_terms.dtype)
    ).sum()
    entropy_loss = -(entropy_num / valid_total.to(device=entropy_num.device, dtype=entropy_num.dtype))

    policy_loss_coef = float(getattr(algo, "_rwkv_policy_loss_coef", 1.0))
    total_loss = policy_loss_coef * policy_loss + algo.ent_coef * entropy_loss + algo.vf_coef * value_loss

    algo.policy.optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    grad_vector = _flatten_policy_grad(algo.policy).detach().cpu()

    return {
        "policy_loss": float(policy_loss.detach().cpu().item()),
        "value_loss": float(value_loss.detach().cpu().item()),
        "entropy_loss": float(entropy_loss.detach().cpu().item()),
        "total_loss": float(total_loss.detach().cpu().item()),
        "objective_total": int(objective_mask.sum().detach().cpu().item()),
        "approx_kl": float(
            torch.mean(
                (torch.exp(log_prob - rollout_data.old_log_prob) - 1)
                - (log_prob - rollout_data.old_log_prob)
            )
            .detach()
            .cpu()
            .item()
        ),
        "clip_fraction": float(
            torch.mean((torch.abs(ratio - 1.0) > clip_range).to(dtype=ratio.dtype))
            .detach()
            .cpu()
            .item()
        ),
        "grad_norm": float(torch.linalg.vector_norm(grad_vector).item()),
        "policy_loss_coef": float(policy_loss_coef),
        "rollout_tensors": {
            "actions": rollout_data.actions.detach().cpu(),
            "returns": rollout_data.returns.detach().cpu(),
            "old_values": rollout_data.old_values.detach().cpu(),
            "advantages": advantages.detach().cpu(),
            "old_log_prob": rollout_data.old_log_prob.detach().cpu(),
            "objective_masks": rollout_data.objective_masks.detach().cpu(),
            "episode_starts": rollout_data.episode_starts.detach().cpu(),
            "observations": rollout_data.observations.detach().cpu(),
        },
        "grad_vector": grad_vector,
    }


def _run_one_official_train_update(algo, callback, vec_env, *, train_env_seed: int) -> None:
    _sync_last_obs(algo, callback, vec_env, train_env_seed=int(train_env_seed))
    assert algo.collect_rollouts(vec_env, callback, algo.rollout_buffer, n_rollout_steps=algo.n_steps)
    if algo.device.type == "cuda":
        torch.cuda.synchronize(device=algo.device)
    algo.train()
    if algo.device.type == "cuda":
        torch.cuda.synchronize(device=algo.device)


def _compare_fixed_gradient(args: argparse.Namespace) -> dict[str, Any]:
    device_obj = torch.device(str(args.device))
    config = _load_default_config()
    base_env_cfg = _apply_transition_memory_contract(
        _build_audit_env_cfg(config["prior"]["environment"]),
        args,
    )

    legacy_algo, legacy_callback, legacy_vec_env, frozen_h, loaded_path = _build_legacy_contract_algo(
        config=config,
        fixed_frozen_h_json=str(args.fixed_frozen_h_json),
        device_obj=device_obj,
        build_seed=int(args.build_seed),
        train_env_seed=int(args.train_env_seed),
        train_rollout_seed=int(args.train_rollout_seed),
        single_eval_pos=int(args.single_eval_pos),
        n_steps=int(args.n_steps),
        batch_size=int(args.batch_size),
        n_epochs=int(args.n_epochs),
        learning_rate=float(args.learning_rate),
        target_kl=float(args.target_kl),
        ppo_reset_env_state_at_sep=bool(args.ppo_reset_env_state_at_sep),
        actor_baseline_mode=str(args.actor_baseline_mode),
        strict_native_rollout=bool(args.strict_native_rollout),
        value_head_impl=str(args.value_head_impl),
        value_path_adapter_impl=str(args.value_path_adapter_impl),
        value_head_mlp_hidden_dim=int(args.value_head_mlp_hidden_dim),
        space_contract=str(args.space_contract),
        ppo_separate_value_backbone=bool(args.ppo_separate_value_backbone),
        vf_coef=float(args.vf_coef),
        transition_inner_grouping=args.transition_inner_grouping,
        transition_inner_min_bucket=args.transition_inner_min_bucket,
        reference_scm_partition_max_bytes=args.reference_scm_partition_max_bytes,
        reference_scm_memory_guard_fraction=args.reference_scm_memory_guard_fraction,
        transition_generator_rebuild_each_step=args.transition_generator_rebuild_each_step,
    )
    fixed_prior = _make_fixed_env_prior(
        base_env_cfg=base_env_cfg,
        fixed_frozen_h_json=str(loaded_path),
        single_eval_pos=int(args.single_eval_pos),
        ppo_reset_env_state_at_sep=bool(args.ppo_reset_env_state_at_sep),
    )
    fit_algo, fit_callback, fit_vec_env, bridge_summary, bridge_kwargs = _build_fit_model_bridge_algo(
        config=config,
        device_obj=device_obj,
        env_prior=fixed_prior,
        build_seed=int(args.build_seed),
        train_env_seed=int(args.train_env_seed),
        train_rollout_seed=int(args.train_rollout_seed),
        n_envs=1,
        n_steps=int(args.n_steps),
        batch_size=int(args.batch_size),
        n_epochs=int(args.n_epochs),
        learning_rate=float(args.learning_rate),
        target_kl=float(args.target_kl),
        ppo_reset_env_state_at_sep=bool(args.ppo_reset_env_state_at_sep),
        actor_baseline_mode=str(args.actor_baseline_mode),
        strict_fixed_env_mode=True,
        deterministic_actor_sampling=True,
        deterministic_batch_plan=True,
        strict_native_rollout=bool(args.strict_native_rollout),
        value_head_impl=str(args.value_head_impl),
        value_path_adapter_impl=str(args.value_path_adapter_impl),
        value_head_mlp_hidden_dim=int(args.value_head_mlp_hidden_dim),
        space_contract=str(args.space_contract),
        normalize_advantage=False,
        vf_coef=float(args.vf_coef),
        ppo_separate_value_backbone=bool(args.ppo_separate_value_backbone),
    )

    shims = []
    if bool(args.compare_shim_batch1_non_native) and not bool(args.strict_native_rollout):
        shims.append({"legacy": _install_batch1_non_native_compare_shim(legacy_algo)})
        shims.append({"fit_model": _install_batch1_non_native_compare_shim(fit_algo)})

    try:
        initial_param_diff = _param_diff_summary(legacy_algo.policy, fit_algo.policy)
        legacy_measure = _collect_and_measure_main_loss(
            legacy_algo,
            legacy_callback,
            legacy_vec_env,
            train_env_seed=int(args.train_env_seed),
        )
        fit_measure = _collect_and_measure_main_loss(
            fit_algo,
            fit_callback,
            fit_vec_env,
            train_env_seed=int(args.train_env_seed),
        )
        rollout_diff = _tensor_diff_summary(
            legacy_measure["rollout_tensors"],
            fit_measure["rollout_tensors"],
        )
        grad_diff = _gradient_diff_summary(
            legacy_measure["grad_vector"],
            fit_measure["grad_vector"],
        )
        loss_diff = _loss_diff_summary(legacy_measure, fit_measure)
        fixed_path_coverage = {
            "legacy": _path_coverage_summary(legacy_vec_env),
            "fit_model": _path_coverage_summary(fit_vec_env),
        }
    finally:
        legacy_vec_env.close()
        fit_vec_env.close()

    update_result = None
    if bool(args.run_train_update_compare):
        legacy_update_algo, legacy_update_callback, legacy_update_vec_env, _, _ = _build_legacy_contract_algo(
            config=config,
            fixed_frozen_h_json=str(args.fixed_frozen_h_json),
            device_obj=device_obj,
            build_seed=int(args.build_seed),
            train_env_seed=int(args.train_env_seed),
            train_rollout_seed=int(args.train_rollout_seed),
            single_eval_pos=int(args.single_eval_pos),
            n_steps=int(args.n_steps),
            batch_size=int(args.batch_size),
            n_epochs=int(args.n_epochs),
            learning_rate=float(args.learning_rate),
            target_kl=float(args.target_kl),
            ppo_reset_env_state_at_sep=bool(args.ppo_reset_env_state_at_sep),
            actor_baseline_mode=str(args.actor_baseline_mode),
            strict_native_rollout=bool(args.strict_native_rollout),
            value_head_impl=str(args.value_head_impl),
            value_path_adapter_impl=str(args.value_path_adapter_impl),
            value_head_mlp_hidden_dim=int(args.value_head_mlp_hidden_dim),
            space_contract=str(args.space_contract),
            ppo_separate_value_backbone=bool(args.ppo_separate_value_backbone),
            vf_coef=float(args.vf_coef),
            transition_inner_grouping=args.transition_inner_grouping,
            transition_inner_min_bucket=args.transition_inner_min_bucket,
            reference_scm_partition_max_bytes=args.reference_scm_partition_max_bytes,
            reference_scm_memory_guard_fraction=args.reference_scm_memory_guard_fraction,
            transition_generator_rebuild_each_step=args.transition_generator_rebuild_each_step,
        )
        fixed_prior_update = _make_fixed_env_prior(
            base_env_cfg=base_env_cfg,
            fixed_frozen_h_json=str(loaded_path),
            single_eval_pos=int(args.single_eval_pos),
            ppo_reset_env_state_at_sep=bool(args.ppo_reset_env_state_at_sep),
        )
        fit_update_algo, fit_update_callback, fit_update_vec_env, _, _ = _build_fit_model_bridge_algo(
            config=config,
            device_obj=device_obj,
            env_prior=fixed_prior_update,
            build_seed=int(args.build_seed),
            train_env_seed=int(args.train_env_seed),
            train_rollout_seed=int(args.train_rollout_seed),
            n_envs=1,
            n_steps=int(args.n_steps),
            batch_size=int(args.batch_size),
            n_epochs=int(args.n_epochs),
            learning_rate=float(args.learning_rate),
            target_kl=float(args.target_kl),
            ppo_reset_env_state_at_sep=bool(args.ppo_reset_env_state_at_sep),
            actor_baseline_mode=str(args.actor_baseline_mode),
            strict_fixed_env_mode=True,
            deterministic_actor_sampling=True,
            deterministic_batch_plan=True,
            strict_native_rollout=bool(args.strict_native_rollout),
            value_head_impl=str(args.value_head_impl),
            value_path_adapter_impl=str(args.value_path_adapter_impl),
            value_head_mlp_hidden_dim=int(args.value_head_mlp_hidden_dim),
            space_contract=str(args.space_contract),
            normalize_advantage=False,
            vf_coef=float(args.vf_coef),
            ppo_separate_value_backbone=bool(args.ppo_separate_value_backbone),
        )
        if bool(args.compare_shim_batch1_non_native) and not bool(args.strict_native_rollout):
            _install_batch1_non_native_compare_shim(legacy_update_algo)
            _install_batch1_non_native_compare_shim(fit_update_algo)
        try:
            pre_update_diff = _param_diff_summary(
                legacy_update_algo.policy,
                fit_update_algo.policy,
            )
            _run_one_official_train_update(
                legacy_update_algo,
                legacy_update_callback,
                legacy_update_vec_env,
                train_env_seed=int(args.train_env_seed),
            )
            _run_one_official_train_update(
                fit_update_algo,
                fit_update_callback,
                fit_update_vec_env,
                train_env_seed=int(args.train_env_seed),
            )
            post_update_diff = _param_diff_summary(
                legacy_update_algo.policy,
                fit_update_algo.policy,
            )
            update_result = {
                "pre_update_param_diff": pre_update_diff,
                "post_one_train_update_param_diff": post_update_diff,
            }
        finally:
            legacy_update_vec_env.close()
            fit_update_vec_env.close()

    return {
        "fixed_frozen_h_json": str(loaded_path),
        "frozen_h_key_count": int(len(frozen_h)),
        "legacy_to_fit_model_fixed_gradient_compare": {
            "initial_param_diff": initial_param_diff,
            "loss_diff": loss_diff,
            "rollout_diff": rollout_diff,
            "gradient_diff": grad_diff,
            "fixed_path_coverage": fixed_path_coverage,
            "bridge_summary": bridge_summary,
            "bridge_kwargs_contract": {
                key: _jsonable(value)
                for key, value in bridge_kwargs.items()
                if key
                not in {
                    "model",
                    "env_prior",
                }
            },
            "compare_shims": shims,
            "one_train_update_compare": update_result,
        },
    }


def _run_nonfixed_path_coverage(args: argparse.Namespace) -> dict[str, Any]:
    device_obj = torch.device(str(args.device))
    config = _load_default_config()
    base_env_cfg = _apply_transition_memory_contract(
        _build_audit_env_cfg(config["prior"]["environment"]),
        args,
    )
    nonfixed_prior = _make_nonfixed_env_prior(
        base_env_cfg=base_env_cfg,
        single_eval_pos=int(args.single_eval_pos),
        ppo_reset_env_state_at_sep=bool(args.ppo_reset_env_state_at_sep),
        conditioned_min=float(args.nonfixed_conditioned_min),
    )
    algo, callback, vec_env, bridge_summary, bridge_kwargs = _build_fit_model_bridge_algo(
        config=config,
        device_obj=device_obj,
        env_prior=nonfixed_prior,
        build_seed=int(args.build_seed),
        train_env_seed=None,
        train_rollout_seed=None,
        n_envs=int(args.coverage_n_envs),
        n_steps=int(args.coverage_n_steps),
        batch_size=int(args.coverage_batch_size),
        n_epochs=1,
        learning_rate=float(args.learning_rate),
        target_kl=float(args.target_kl),
        ppo_reset_env_state_at_sep=bool(args.ppo_reset_env_state_at_sep),
        actor_baseline_mode=str(args.actor_baseline_mode),
        strict_fixed_env_mode=False,
        deterministic_actor_sampling=False,
        deterministic_batch_plan=True,
        strict_native_rollout=bool(args.strict_native_rollout),
        value_head_impl=str(args.value_head_impl),
        value_path_adapter_impl=str(args.value_path_adapter_impl),
        value_head_mlp_hidden_dim=int(args.value_head_mlp_hidden_dim),
        space_contract=str(args.space_contract),
        normalize_advantage=False,
        vf_coef=float(args.vf_coef),
        ppo_separate_value_backbone=bool(args.ppo_separate_value_backbone),
    )
    if bool(args.compare_shim_batch1_non_native) and not bool(args.strict_native_rollout):
        _install_batch1_non_native_compare_shim(algo)
    try:
        # Use the actual VecEnv reset/collect path. This proves the non-fixed
        # conditioned sampler does not silently escape the vectorized exact-SCM path.
        _sync_last_obs(algo, callback, vec_env, train_env_seed=int(args.coverage_env_seed))
        collected = bool(
            algo.collect_rollouts(
                vec_env,
                callback,
                algo.rollout_buffer,
                n_rollout_steps=algo.n_steps,
            )
        )
        if algo.device.type == "cuda":
            torch.cuda.synchronize(device=algo.device)
        coverage = _path_coverage_summary(vec_env)
    finally:
        vec_env.close()
    return {
        "nonfixed_conditioned_min": float(args.nonfixed_conditioned_min),
        "coverage_env_seed": int(args.coverage_env_seed),
        "collect_rollouts_completed": bool(collected),
        "path_coverage": coverage,
        "bridge_summary": bridge_summary,
        "bridge_kwargs_contract": {
            key: _jsonable(value)
            for key, value in bridge_kwargs.items()
            if key
            not in {
                "model",
                "env_prior",
            }
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the legacy Phase 2 fixed-env PPO path against the fit_model "
            "training bridge at rollout/loss/gradient level, then smoke-test the "
            "non-fixed vectorized exact-SCM conditioned-training path."
        )
    )
    parser.add_argument("--device", default=_default_device())
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--fixed-frozen-h-json", default=DEFAULT_FIXED_FROZEN_H_JSON)
    parser.add_argument("--build-seed", type=int, default=4040)
    parser.add_argument("--train-env-seed", type=int, default=26024)
    parser.add_argument("--train-rollout-seed", type=int, default=4040)
    parser.add_argument("--single-eval-pos", type=int, default=64)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument("--ppo-reset-env-state-at-sep", type=str2bool, default=True)
    parser.add_argument("--actor-baseline-mode", default="learned")
    parser.add_argument("--strict-native-rollout", type=str2bool, default=False)
    parser.add_argument("--compare-shim-batch1-non-native", type=str2bool, default=False)
    parser.add_argument("--value-head-impl", default="vendor_official")
    parser.add_argument("--value-path-adapter-impl", default="none")
    parser.add_argument("--value-head-mlp-hidden-dim", type=int, default=256)
    parser.add_argument("--space-contract", default="normalized")
    parser.add_argument("--ppo-separate-value-backbone", type=str2bool, default=False)
    parser.add_argument("--vf-coef", type=float, default=0.1)
    parser.add_argument("--transition-inner-grouping", default=None)
    parser.add_argument("--transition-inner-min-bucket", type=int, default=None)
    parser.add_argument("--reference-scm-partition-max-bytes", type=int, default=None)
    parser.add_argument("--reference-scm-memory-guard-fraction", type=float, default=None)
    parser.add_argument("--transition-generator-rebuild-each-step", type=str2bool, default=None)
    parser.add_argument("--run-train-update-compare", type=str2bool, default=True)
    parser.add_argument("--nonfixed-conditioned-min", type=float, default=0.7)
    parser.add_argument("--coverage-n-envs", type=int, default=2)
    parser.add_argument("--coverage-n-steps", type=int, default=8)
    parser.add_argument("--coverage-batch-size", type=int, default=8)
    parser.add_argument("--coverage-env-seed", type=int, default=33000)
    args = parser.parse_args()

    output_path = Path(str(args.output_json)).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if int(args.coverage_batch_size) > int(args.coverage_n_envs) * int(args.coverage_n_steps):
        raise ValueError(
            "--coverage-batch-size must be <= coverage_n_envs * coverage_n_steps "
            "for the smoke PPO collection."
        )

    result = {
        "script": "phase2_fit_model_migration_gradient_compare",
        "args": _jsonable(vars(args)),
    }
    result["fixed_gradient_compare"] = _compare_fixed_gradient(args)
    try:
        result["nonfixed_vectorized_path_coverage"] = _run_nonfixed_path_coverage(args)
    except Exception as exc:
        result["nonfixed_vectorized_path_coverage"] = {
            "status": "failed",
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "traceback": traceback.format_exc(),
            "note": (
                "The fixed legacy-vs-fit_model gradient compare completed before this "
                "failure. The failure happened in the non-fixed conditioned coverage "
                "smoke path and is intentionally preserved instead of silently "
                "lowering reward_state_input_gain_fraction_conditioned_min."
            ),
        }
    output_path.write_text(
        json.dumps(_jsonable(result), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(_jsonable(result), indent=2, sort_keys=True))
    print(f"[phase2-fit-model-migration-gradient-compare] wrote {output_path}")


if __name__ == "__main__":
    main()
