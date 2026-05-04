import argparse
import copy
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

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

from stable_baselines3.common.logger import configure as configure_logger  # noqa: E402

from ticl.analysis.critic_free_single_env_audit import (  # noqa: E402
    _apply_boundary_contract_mode_flags,
    _apply_sep_state_reset_flag,
    _make_deterministic_batch_plan,
)
from ticl.analysis.phase2_gym_prior_pack_isomorphism_sentinel import (  # noqa: E402
    _fill_existing_rollout_buffer_from_pack,
    _json_safe,
    _policy_param_snapshot,
    _summarize_pack,
)
from ticl.analysis.phase2_gym_prior_pack_training_runner import (  # noqa: E402
    _apply_reward_normalization_if_enabled,
    _collect_gym_policy_pack,
    _collect_prior_policy_pack,
    _logger_values,
    _normalize_env_id_list,
    _run_update_from_pack,
    _stable_digest,
    _set_topology_conditioned_sampling,
    _build_algo as _build_pack_algo,
)
from ticl.analysis.phase2_fit_model_migration_gradient_compare import (  # noqa: E402
    _gradient_diff_summary,
    _loss_diff_summary,
    _tensor_diff_summary,
)
from ticl.analysis.phase3_phase2_milestone_contract_gradient_compare import (  # noqa: E402
    _max_abs_diff_tensor,
    _param_diff_summary,
)
from ticl.config_utils import str2bool  # noqa: E402
from ticl.model_builder import get_model  # noqa: E402
from ticl.model_configs import get_model_default_config  # noqa: E402
from ticl.priors.environment_prior import EnvironmentPrior  # noqa: E402
from ticl.rlpfn_maintained_path import (  # noqa: E402
    resolve_rlpfn_token_layout,
    validate_rlpfn_maintained_path_config,
)
from ticl.sb3_recurrent_ppo import _resolve_actor_advantages, build_recurrent_ppo  # noqa: E402
from ticl.train import _resolve_recurrent_ppo_training_bridge_kwargs  # noqa: E402


DEFAULT_OUTPUT_JSON = (
    "/home/chen/RLPFN/artifacts/phase2_fit_model_pack_ppo_migration_parity_0502/"
    "pack_fit_model_ppo_migration_parity_report.json"
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
    if torch.is_tensor(value):
        if value.numel() == 1:
            return _jsonable(value.detach().cpu().item())
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _float_delta(a: Any, b: Any) -> float | None:
    if isinstance(a, (int, float, np.generic)) and isinstance(b, (int, float, np.generic)):
        return float(b) - float(a)
    return None


def _dict_numeric_diff(a: dict[str, Any], b: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in keys:
        if key in a and key in b:
            out[key] = {
                "pack": _jsonable(a[key]),
                "fit_model": _jsonable(b[key]),
                "delta": _float_delta(a[key], b[key]),
            }
    return out


def _enforce_healthy_sampled_topology_contract(
    cfg: dict[str, Any],
    *,
    topology_state_gain_min: float,
    topology_action_gain_min: float,
    topology_state_to_action_ratio_max: float,
    topology_max_attempts: int,
    ppo_batch_size: int,
    ppo_n_epochs: int,
    ppo_vf_coef: float,
    ppo_normalize_advantage: bool,
    ppo_target_kl: float | None,
) -> dict[str, Any]:
    cfg = copy.deepcopy(cfg)
    env_cfg = cfg["prior"]["environment"]
    env_cfg["batch_parallel_backend"] = "torch_vectorized"
    env_cfg["batch_vectorized_grouping"] = "family"
    env_cfg["reinforce_aux_enabled"] = False
    env_cfg["normalized_q_value_weight"] = 0.0
    env_cfg["next_state_flow_matching_weight"] = 0.0
    _set_topology_conditioned_sampling(
        env_cfg,
        state_gain_min=float(topology_state_gain_min),
        action_gain_min=float(topology_action_gain_min),
        state_to_action_ratio_max=float(topology_state_to_action_ratio_max),
        max_attempts=int(topology_max_attempts),
    )

    cfg["prior"]["n_samples"] = 2048
    cfg["dataloader"]["batch_size"] = 2048
    opt = cfg["optimizer"]
    opt["rl_objective"] = "ppo"
    opt["learning_rate"] = 2e-4
    opt["ppo_batch_size"] = int(ppo_batch_size)
    opt["ppo_n_epochs"] = int(ppo_n_epochs)
    opt["ppo_normalize_advantage"] = bool(ppo_normalize_advantage)
    opt["ppo_vf_coef"] = float(ppo_vf_coef)
    opt["ppo_target_kl"] = None if ppo_target_kl is None else float(ppo_target_kl)
    opt["ppo_single_eval_pos"] = None
    opt["ppo_space_contract"] = "raw"
    opt["ppo_value_head_impl"] = "vendor_official"
    opt["ppo_value_path_adapter_impl"] = "none"
    opt["ppo_value_head_mlp_hidden_dim"] = 256
    opt["ppo_runtime_normalized_q_value_weight_override"] = 0.0
    opt["ppo_runtime_next_state_flow_matching_weight_override"] = 0.0
    validate_rlpfn_maintained_path_config(cfg)
    return cfg


def _load_contract_config(args: argparse.Namespace) -> dict[str, Any]:
    cfg = get_model_default_config("rlpfn")
    return _enforce_healthy_sampled_topology_contract(
        cfg,
        topology_state_gain_min=float(args.topology_state_gain_min),
        topology_action_gain_min=float(args.topology_action_gain_min),
        topology_state_to_action_ratio_max=float(args.topology_state_to_action_ratio_max),
        topology_max_attempts=int(args.topology_max_attempts),
        ppo_batch_size=int(args.ppo_batch_size),
        ppo_n_epochs=int(args.ppo_n_epochs),
        ppo_vf_coef=float(args.ppo_vf_coef),
        ppo_normalize_advantage=bool(args.ppo_normalize_advantage),
        ppo_target_kl=args.ppo_target_kl,
    )


def _make_fixed_single_eval_prior(
    env_cfg: dict[str, Any],
    *,
    single_eval_pos: int,
    ppo_reset_env_state_at_sep: bool,
) -> EnvironmentPrior:
    prior = EnvironmentPrior(copy.deepcopy(env_cfg))
    prior._sample_single_eval_pos = lambda n_samples_arg, rng_seed=None: int(single_eval_pos)
    _apply_boundary_contract_mode_flags(prior, "normal")
    _apply_sep_state_reset_flag(prior, bool(ppo_reset_env_state_at_sep))
    return prior


def _build_model_from_scratch(*, cfg: dict[str, Any], device_obj: torch.device, seed: int):
    _seed_all(int(seed))
    _, model, _, _ = get_model(copy.deepcopy(cfg), device=str(device_obj), should_train=False, verbose=False)
    model.to(device_obj)
    model.train()
    return model


def _build_fit_model_algo(
    *,
    cfg: dict[str, Any],
    env_prior: EnvironmentPrior,
    device_obj: torch.device,
    build_seed: int,
    n_envs: int,
    n_steps: int,
    ppo_batch_size: int,
    ppo_n_epochs: int,
    ppo_vf_coef: float,
    ppo_normalize_advantage: bool,
    ppo_target_kl: float | None,
    ppo_reset_env_state_at_sep: bool,
    deterministic_batch_plan: bool,
    sb3_reward_normalization_enabled: bool,
    sb3_observation_normalization_enabled: bool,
    sb3_observation_normalization_clip: float,
    sb3_observation_normalization_epsilon: float,
):
    model = _build_model_from_scratch(cfg=cfg, device_obj=device_obj, seed=int(build_seed))
    opt = cfg["optimizer"]
    build_kwargs, bridge_summary = _resolve_recurrent_ppo_training_bridge_kwargs(
        model=model,
        env_prior=env_prior,
        device=str(device_obj),
        num_features=int(cfg["prior"]["num_features"]),
        n_envs=int(n_envs),
        n_steps=int(n_steps),
        learning_rate=float(opt.get("learning_rate", 2e-4)),
        batch_size=int(ppo_batch_size),
        n_epochs=int(ppo_n_epochs),
        gamma=float(opt.get("ppo_gamma", 0.98)),
        gae_lambda=float(opt.get("ppo_gae_lambda", 0.90)),
        clip_range=opt.get("ppo_clip_range", 0.2),
        clip_range_vf=opt.get("ppo_clip_range_vf", None),
        normalize_advantage=bool(ppo_normalize_advantage),
        space_contract=str(opt.get("ppo_space_contract", "raw")),
        value_target_space=None,
        actor_gae_space=None,
        allow_mixed_space_contract=False,
        actor_baseline_mode=str(opt.get("ppo_actor_baseline_mode", "learned")),
        separate_value_backbone=False,
        reset_env_state_at_sep=bool(ppo_reset_env_state_at_sep),
        value_head_impl="vendor_official",
        value_path_adapter_impl="none",
        value_head_mlp_hidden_dim=256,
        runtime_normalized_q_value_weight_override=0.0,
        runtime_next_state_flow_matching_weight_override=0.0,
        ent_coef=float(opt.get("ppo_ent_coef", 0.0)),
        vf_coef=float(ppo_vf_coef),
        max_grad_norm=float(opt.get("ppo_max_grad_norm", 0.5)),
        target_kl=ppo_target_kl,
        verbose=0,
        ppo_restore_validation_policy_state=False,
        ppo_strict_fixed_env_mode=False,
        ppo_env_rng_seeds=None,
        ppo_rollout_rng_seeds=None,
        ppo_deterministic_actor_sampling=False,
        ppo_deterministic_batch_plan=bool(deterministic_batch_plan),
        ppo_strict_native_rollout=False,
    )
    build_kwargs.update(
        {
            "sb3_reward_normalization_enabled": bool(sb3_reward_normalization_enabled),
            "sb3_reward_normalization_clip": 10.0,
            "sb3_reward_normalization_epsilon": 1e-8,
        }
    )
    algo, callback, vec_env = build_recurrent_ppo(**build_kwargs)
    algo._rwkv_policy_loss_coef = 1.0
    algo._rwkv_aux_q_weight = 0.0
    algo._rwkv_aux_flow_weight = 0.0
    algo._rwkv_sb3_observation_normalization_enabled = bool(sb3_observation_normalization_enabled)
    algo._rwkv_sb3_observation_normalization_clip = float(sb3_observation_normalization_clip)
    algo._rwkv_sb3_observation_normalization_epsilon = float(sb3_observation_normalization_epsilon)
    algo._rwkv_sb3_observation_normalization_mean = np.zeros((int(cfg["prior"]["num_features"]),), dtype=np.float64)
    algo._rwkv_sb3_observation_normalization_var = np.ones((int(cfg["prior"]["num_features"]),), dtype=np.float64)
    algo._rwkv_sb3_observation_normalization_count = np.full((int(cfg["prior"]["num_features"]),), 1e-4, dtype=np.float64)
    if bool(deterministic_batch_plan):
        _make_deterministic_batch_plan(algo.rollout_buffer)
    algo.ep_info_buffer = []
    algo.ep_success_buffer = []
    algo.set_logger(configure_logger(folder=None, format_strings=[]))
    return algo, callback, vec_env, bridge_summary, build_kwargs


def _build_pack_training_algo(
    *,
    cfg: dict[str, Any],
    device_obj: torch.device,
    build_seed: int,
    n_envs: int,
    n_steps: int,
    ppo_batch_size: int,
    ppo_n_epochs: int,
    ppo_vf_coef: float,
    ppo_normalize_advantage: bool,
    ppo_target_kl: float | None,
    single_eval_pos: int,
    sb3_reward_normalization_enabled: bool,
    sb3_observation_normalization_enabled: bool,
    sb3_observation_normalization_clip: float,
    sb3_observation_normalization_epsilon: float,
):
    algo, vec_env = _build_pack_algo(
        cfg=copy.deepcopy(cfg),
        env_cfg=copy.deepcopy(cfg["prior"]["environment"]),
        frozen_h=None,
        frozen_h_list=None,
        frozen_h_list_env_seeds=None,
        prior_mode="sampled_topology",
        device_obj=device_obj,
        num_features=int(cfg["prior"]["num_features"]),
        n_envs=int(n_envs),
        n_steps=int(n_steps),
        build_seed=int(build_seed),
        sb3_reward_normalization_enabled=bool(sb3_reward_normalization_enabled),
        sb3_observation_normalization_enabled=bool(sb3_observation_normalization_enabled),
        sb3_observation_normalization_clip=float(sb3_observation_normalization_clip),
        sb3_observation_normalization_epsilon=float(sb3_observation_normalization_epsilon),
        gain_min=float(0.7),
        topology_state_gain_min=float(cfg["prior"]["environment"]["reward_state_input_gain_fraction_conditioned_min"]),
        topology_action_gain_min=float(cfg["prior"]["environment"]["reward_action_input_gain_fraction_conditioned_min"]),
        topology_state_to_action_ratio_max=float(
            cfg["prior"]["environment"]["reward_state_to_action_gain_ratio_conditioned_max"]
        ),
        topology_max_attempts=int(cfg["prior"]["environment"]["reward_topology_conditioned_sampling_max_attempts"]),
        fixed_env_group_across_updates=False,
        ppo_learning_rate=float(cfg["optimizer"]["learning_rate"]),
        ppo_batch_size=int(ppo_batch_size),
        ppo_n_epochs=int(ppo_n_epochs),
        ppo_vf_coef=float(ppo_vf_coef),
        ppo_normalize_advantage=bool(ppo_normalize_advantage),
        ppo_target_kl=ppo_target_kl,
    )
    algo._rwkv_env_prior._sample_single_eval_pos = lambda n_samples_arg, rng_seed=None: int(single_eval_pos)
    vec_env.env_prior._sample_single_eval_pos = lambda n_samples_arg, rng_seed=None: int(single_eval_pos)
    algo.set_logger(configure_logger(folder=None, format_strings=[]))
    return algo, vec_env


def _copy_normalizer_state(src_algo, dst_algo) -> None:
    for name in (
        "_rwkv_sb3_reward_normalization_enabled",
        "_rwkv_sb3_reward_normalization_clip",
        "_rwkv_sb3_reward_normalization_epsilon",
        "_rwkv_sb3_reward_normalization_returns",
        "_rwkv_last_sb3_reward_normalization_stats",
        "_rwkv_sb3_observation_normalization_enabled",
        "_rwkv_sb3_observation_normalization_clip",
        "_rwkv_sb3_observation_normalization_epsilon",
        "_rwkv_sb3_observation_normalization_mean",
        "_rwkv_sb3_observation_normalization_var",
        "_rwkv_sb3_observation_normalization_count",
    ):
        if hasattr(src_algo, name):
            value = getattr(src_algo, name)
            setattr(dst_algo, name, copy.deepcopy(value))
    if hasattr(src_algo, "_rwkv_sb3_reward_normalization_ret_rms"):
        src_rms = getattr(src_algo, "_rwkv_sb3_reward_normalization_ret_rms")
        dst_rms = getattr(dst_algo, "_rwkv_sb3_reward_normalization_ret_rms", None)
        if dst_rms is not None:
            dst_rms.mean = np.asarray(src_rms.mean).copy()
            dst_rms.var = np.asarray(src_rms.var).copy()
            dst_rms.count = float(src_rms.count)


def _normalizer_summary(algo) -> dict[str, Any]:
    ret_rms = getattr(algo, "_rwkv_sb3_reward_normalization_ret_rms", None)
    return {
        "reward_enabled": bool(getattr(algo, "_rwkv_sb3_reward_normalization_enabled", False)),
        "reward_returns_shape": list(np.asarray(getattr(algo, "_rwkv_sb3_reward_normalization_returns", [])).shape),
        "reward_ret_rms_mean": None if ret_rms is None else float(np.asarray(ret_rms.mean).reshape(())),
        "reward_ret_rms_var": None if ret_rms is None else float(np.asarray(ret_rms.var).reshape(())),
        "reward_ret_rms_count": None if ret_rms is None else float(ret_rms.count),
        "obs_enabled": bool(getattr(algo, "_rwkv_sb3_observation_normalization_enabled", False)),
        "obs_mean_digest": _stable_digest(np.asarray(getattr(algo, "_rwkv_sb3_observation_normalization_mean", []))),
        "obs_var_digest": _stable_digest(np.asarray(getattr(algo, "_rwkv_sb3_observation_normalization_var", []))),
        "obs_count_digest": _stable_digest(np.asarray(getattr(algo, "_rwkv_sb3_observation_normalization_count", []))),
    }


def _flatten_policy_grad(policy: torch.nn.Module) -> torch.Tensor:
    parts = []
    for param in policy.parameters():
        if not bool(param.requires_grad):
            continue
        grad = param.grad
        if grad is None:
            parts.append(torch.zeros(param.numel(), device=param.device, dtype=param.dtype))
        else:
            parts.append(grad.detach().reshape(-1))
    if not parts:
        return torch.zeros((0,), dtype=torch.float32)
    return torch.cat(parts).to(dtype=torch.float32)


def _fill_and_measure_loss(
    algo,
    *,
    pack: dict[str, np.ndarray],
    values: np.ndarray,
    log_probs: np.ndarray,
    num_features: int,
    action_slot_dim: int,
    next_state_target_dim: int,
    single_eval_pos: int,
) -> dict[str, Any]:
    buffer_pack, reward_norm_stats = _apply_reward_normalization_if_enabled(algo, pack)
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
    )
    rollout_data = next(algo.rollout_buffer.get_gpu_flat(algo.batch_size))
    objective_mask = rollout_data.objective_masks > 1e-8
    valid_total = objective_mask.to(device=rollout_data.returns.device, dtype=rollout_data.returns.dtype).sum().clamp_min(1.0)
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
        "buffer_summary": buffer_summary,
        "reward_norm_stats": reward_norm_stats,
        "policy_loss": float(policy_loss.detach().cpu().item()),
        "value_loss": float(value_loss.detach().cpu().item()),
        "entropy_loss": float(entropy_loss.detach().cpu().item()),
        "total_loss": float(total_loss.detach().cpu().item()),
        "objective_total": int(objective_mask.sum().detach().cpu().item()),
        "approx_kl": float(
            torch.mean((torch.exp(log_prob - rollout_data.old_log_prob) - 1) - (log_prob - rollout_data.old_log_prob))
            .detach()
            .cpu()
            .item()
        ),
        "clip_fraction": float(torch.mean((torch.abs(ratio - 1.0) > clip_range).to(dtype=ratio.dtype)).detach().cpu().item()),
        "policy_loss_coef": float(policy_loss_coef),
        "value_loss_weight": float(algo.vf_coef),
        "normalized_q_value_weight": float(getattr(algo, "_rwkv_aux_q_weight", 0.0)),
        "next_state_flow_matching_weight": float(getattr(algo, "_rwkv_aux_flow_weight", 0.0)),
        "rollout_tensors": {
            "actions": rollout_data.actions.detach().cpu(),
            "returns": rollout_data.returns.detach().cpu(),
            "old_values": rollout_data.old_values.detach().cpu(),
            "advantages": advantages.detach().cpu(),
            "old_log_prob": rollout_data.old_log_prob.detach().cpu(),
            "objective_masks": rollout_data.objective_masks.detach().cpu(),
            "episode_starts": rollout_data.episode_starts.detach().cpu(),
            "observations": rollout_data.observations.detach().cpu(),
            "action_masks": rollout_data.action_masks.detach().cpu(),
        },
        "grad_vector": grad_vector,
    }


def _diff_buffer_summaries(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    return {
        "objective_mask_sum": {
            "pack": a.get("objective_mask_sum"),
            "fit_model": b.get("objective_mask_sum"),
            "delta": _float_delta(a.get("objective_mask_sum"), b.get("objective_mask_sum")),
        },
        "returns": _dict_numeric_diff(a.get("returns", {}), b.get("returns", {}), ["mean", "std", "absmax"]),
        "advantages": _dict_numeric_diff(a.get("advantages", {}), b.get("advantages", {}), ["mean", "std", "absmax"]),
        "actor_advantages": _dict_numeric_diff(a.get("actor_advantages", {}), b.get("actor_advantages", {}), ["mean", "std", "absmax"]),
        "digests_equal": {
            key: bool((a.get("digests", {}) or {}).get(key) == (b.get("digests", {}) or {}).get(key))
            for key in sorted(set((a.get("digests", {}) or {}).keys()) | set((b.get("digests", {}) or {}).keys()))
        },
    }


def _compare_sampled_topology_generator(
    *,
    cfg: dict[str, Any],
    device_obj: torch.device,
    build_seed: int,
    env_seed: int,
    n_envs: int,
    n_steps: int,
    ppo_batch_size: int,
    ppo_n_epochs: int,
    ppo_vf_coef: float,
    ppo_normalize_advantage: bool,
    ppo_target_kl: float | None,
    single_eval_pos: int,
    sb3_reward_normalization_enabled: bool,
    sb3_observation_normalization_enabled: bool,
    sb3_observation_normalization_clip: float,
    sb3_observation_normalization_epsilon: float,
) -> dict[str, Any]:
    pack_algo, pack_vec_env = _build_pack_training_algo(
        cfg=cfg,
        device_obj=device_obj,
        build_seed=int(build_seed),
        n_envs=int(n_envs),
        n_steps=int(n_steps),
        ppo_batch_size=int(ppo_batch_size),
        ppo_n_epochs=int(ppo_n_epochs),
        ppo_vf_coef=float(ppo_vf_coef),
        ppo_normalize_advantage=bool(ppo_normalize_advantage),
        ppo_target_kl=ppo_target_kl,
        single_eval_pos=int(single_eval_pos),
        sb3_reward_normalization_enabled=bool(sb3_reward_normalization_enabled),
        sb3_observation_normalization_enabled=bool(sb3_observation_normalization_enabled),
        sb3_observation_normalization_clip=float(sb3_observation_normalization_clip),
        sb3_observation_normalization_epsilon=float(sb3_observation_normalization_epsilon),
    )
    fit_prior = _make_fixed_single_eval_prior(
        cfg["prior"]["environment"],
        single_eval_pos=int(single_eval_pos),
        ppo_reset_env_state_at_sep=True,
    )
    fit_algo, _fit_callback, fit_vec_env, bridge_summary, bridge_kwargs = _build_fit_model_algo(
        cfg=cfg,
        env_prior=fit_prior,
        device_obj=device_obj,
        build_seed=int(build_seed),
        n_envs=int(n_envs),
        n_steps=int(n_steps),
        ppo_batch_size=int(ppo_batch_size),
        ppo_n_epochs=int(ppo_n_epochs),
        ppo_vf_coef=float(ppo_vf_coef),
        ppo_normalize_advantage=bool(ppo_normalize_advantage),
        ppo_target_kl=ppo_target_kl,
        ppo_reset_env_state_at_sep=True,
        deterministic_batch_plan=True,
        sb3_reward_normalization_enabled=bool(sb3_reward_normalization_enabled),
        sb3_observation_normalization_enabled=bool(sb3_observation_normalization_enabled),
        sb3_observation_normalization_clip=float(sb3_observation_normalization_clip),
        sb3_observation_normalization_epsilon=float(sb3_observation_normalization_epsilon),
    )
    seeds = [int(env_seed) + int(i) for i in range(int(n_envs))]
    hyper_rng_seed = int(env_seed) + 424242
    try:
        # The explicit per-env seeds are consumed by environment construction and
        # rollout generators.  The sampled topology h-list itself still follows
        # the global Python/NumPy/Torch RNG streams, so restore those streams on
        # both sides for a true generator contract compare.
        _seed_all(hyper_rng_seed)
        pack_obs = pack_vec_env._full_reset_batch(seeds=seeds)
        _seed_all(hyper_rng_seed)
        fit_obs = fit_vec_env._full_reset_batch(seeds=seeds)
        pack_masks = pack_vec_env.action_masks()
        fit_masks = fit_vec_env.action_masks()
        pack_h_list = getattr(pack_vec_env, "_h_list", None)
        fit_h_list = getattr(fit_vec_env, "_h_list", None)
        pack_env = getattr(pack_vec_env, "_env", None)
        fit_env = getattr(fit_vec_env, "_env", None)
        gain_keys = [
            "reward_state_input_gain_fraction",
            "reward_action_input_gain_fraction",
            "reward_state_to_action_gain_ratio",
            "obs_dim_per_sample",
            "action_dim_per_sample",
            "state_dim_per_sample",
        ]
        env_tensor_diffs = {}
        if isinstance(pack_env, dict) and isinstance(fit_env, dict):
            for key in gain_keys:
                a = pack_env.get(key)
                b = fit_env.get(key)
                if torch.is_tensor(a) and torch.is_tensor(b):
                    env_tensor_diffs[key] = float(_max_abs_diff_tensor(a, b))
        h_digest_equal = _stable_digest(pack_h_list) == _stable_digest(fit_h_list)
        result = {
            "seeds": seeds,
            "hyper_rng_seed": int(hyper_rng_seed),
            "h_digest_equal": bool(h_digest_equal),
            "reset_obs_max_abs_delta": float(np.max(np.abs(np.asarray(pack_obs) - np.asarray(fit_obs)))),
            "action_masks_max_abs_delta": float(np.max(np.abs(np.asarray(pack_masks) - np.asarray(fit_masks)))),
            "env_tensor_max_abs_delta": env_tensor_diffs,
            "pack_single_eval_pos": getattr(pack_vec_env, "_single_eval_pos_np", None),
            "fit_single_eval_pos": getattr(fit_vec_env, "_single_eval_pos_np", None),
            "initial_policy_param_diff": _param_diff_summary(pack_algo.policy, fit_algo.policy),
            "bridge_summary": bridge_summary,
            "bridge_kwargs_contract": {
                key: _jsonable(value)
                for key, value in bridge_kwargs.items()
                if key not in {"model", "env_prior"}
            },
        }
        result["hard_pass"] = bool(
            result["h_digest_equal"]
            and result["reset_obs_max_abs_delta"] == 0.0
            and result["action_masks_max_abs_delta"] == 0.0
            and all(float(v) == 0.0 for v in env_tensor_diffs.values())
            and result["initial_policy_param_diff"]["policy_param_max_abs_diff"] == 0.0
        )
        return result
    finally:
        pack_vec_env.close()
        fit_vec_env.close()
        del pack_algo, fit_algo


def _compare_update_from_pack(
    *,
    label: str,
    cfg: dict[str, Any],
    device_obj: torch.device,
    build_seed: int,
    collect_seed: int,
    n_envs: int,
    n_steps: int,
    ppo_batch_size: int,
    ppo_n_epochs: int,
    ppo_vf_coef: float,
    ppo_normalize_advantage: bool,
    ppo_target_kl: float | None,
    single_eval_pos: int,
    sb3_reward_normalization_enabled: bool,
    sb3_observation_normalization_enabled: bool,
    sb3_observation_normalization_clip: float,
    sb3_observation_normalization_epsilon: float,
    gym_env_ids: list[str] | None,
) -> dict[str, Any]:
    layout = resolve_rlpfn_token_layout(cfg["prior"]["environment"], num_features=int(cfg["prior"]["num_features"]))
    num_features = int(cfg["prior"]["num_features"])
    obs_slot_dim = int(layout["obs_slot_dim"])
    action_slot_dim = int(layout["action_slot_dim"])
    dim_probe = EnvironmentPrior(copy.deepcopy(cfg["prior"]["environment"]))
    next_state_target_dim = int(max(1, dim_probe._resolve_dim_upper_bound(cfg["prior"]["environment"].get("state_dim", None), default=0)))

    collector_algo, collector_vec_env = _build_pack_training_algo(
        cfg=cfg,
        device_obj=device_obj,
        build_seed=int(build_seed),
        n_envs=int(n_envs),
        n_steps=int(n_steps),
        ppo_batch_size=int(ppo_batch_size),
        ppo_n_epochs=int(ppo_n_epochs),
        ppo_vf_coef=float(ppo_vf_coef),
        ppo_normalize_advantage=bool(ppo_normalize_advantage),
        ppo_target_kl=ppo_target_kl,
        single_eval_pos=int(single_eval_pos),
        sb3_reward_normalization_enabled=bool(sb3_reward_normalization_enabled),
        sb3_observation_normalization_enabled=bool(sb3_observation_normalization_enabled),
        sb3_observation_normalization_clip=float(sb3_observation_normalization_clip),
        sb3_observation_normalization_epsilon=float(sb3_observation_normalization_epsilon),
    )
    try:
        if gym_env_ids is None:
            pack, values, log_probs, meta = _collect_prior_policy_pack(
                algo=collector_algo,
                vec_env=collector_vec_env,
                n_steps=int(n_steps),
                seed=int(collect_seed),
                num_features=int(num_features),
                obs_slot_dim=int(obs_slot_dim),
                action_slot_dim=int(action_slot_dim),
                next_state_target_dim=int(next_state_target_dim),
                store_next_state_targets=False,
            )
        else:
            pack, values, log_probs, meta = _collect_gym_policy_pack(
                algo=collector_algo,
                gym_env_ids=gym_env_ids,
                n_steps=int(n_steps),
                seed=int(collect_seed),
                num_features=int(num_features),
                obs_slot_dim=int(obs_slot_dim),
                action_slot_dim=int(action_slot_dim),
                next_state_target_dim=int(next_state_target_dim),
                single_eval_pos=int(single_eval_pos),
                terminal_token_enabled=bool(layout["terminal_token_enabled"]),
                device_obj=device_obj,
                store_next_state_targets=False,
            )
        normalizer_after_collect = _normalizer_summary(collector_algo)
        pack_summary = _summarize_pack(pack, label=str(label))
    finally:
        collector_vec_env.close()
        del collector_algo

    def _new_pair():
        pack_algo, pack_vec_env = _build_pack_training_algo(
            cfg=cfg,
            device_obj=device_obj,
            build_seed=int(build_seed),
            n_envs=int(n_envs),
            n_steps=int(n_steps),
            ppo_batch_size=int(ppo_batch_size),
            ppo_n_epochs=int(ppo_n_epochs),
            ppo_vf_coef=float(ppo_vf_coef),
            ppo_normalize_advantage=bool(ppo_normalize_advantage),
            ppo_target_kl=ppo_target_kl,
            single_eval_pos=int(single_eval_pos),
            sb3_reward_normalization_enabled=bool(sb3_reward_normalization_enabled),
            sb3_observation_normalization_enabled=bool(sb3_observation_normalization_enabled),
            sb3_observation_normalization_clip=float(sb3_observation_normalization_clip),
            sb3_observation_normalization_epsilon=float(sb3_observation_normalization_epsilon),
        )
        fit_prior = _make_fixed_single_eval_prior(
            cfg["prior"]["environment"],
            single_eval_pos=int(single_eval_pos),
            ppo_reset_env_state_at_sep=True,
        )
        fit_algo, _fit_callback, fit_vec_env, bridge_summary, bridge_kwargs = _build_fit_model_algo(
            cfg=cfg,
            env_prior=fit_prior,
            device_obj=device_obj,
            build_seed=int(build_seed),
            n_envs=int(n_envs),
            n_steps=int(n_steps),
            ppo_batch_size=int(ppo_batch_size),
            ppo_n_epochs=int(ppo_n_epochs),
            ppo_vf_coef=float(ppo_vf_coef),
            ppo_normalize_advantage=bool(ppo_normalize_advantage),
            ppo_target_kl=ppo_target_kl,
            ppo_reset_env_state_at_sep=True,
            deterministic_batch_plan=True,
            sb3_reward_normalization_enabled=bool(sb3_reward_normalization_enabled),
            sb3_observation_normalization_enabled=bool(sb3_observation_normalization_enabled),
            sb3_observation_normalization_clip=float(sb3_observation_normalization_clip),
            sb3_observation_normalization_epsilon=float(sb3_observation_normalization_epsilon),
        )
        _copy_normalizer_state(pack_algo, fit_algo)
        return pack_algo, pack_vec_env, fit_algo, fit_vec_env, bridge_summary, bridge_kwargs

    pack_loss_algo, pack_loss_vec_env, fit_loss_algo, fit_loss_vec_env, bridge_summary, bridge_kwargs = _new_pair()
    try:
        pack_loss = _fill_and_measure_loss(
            pack_loss_algo,
            pack=pack,
            values=values,
            log_probs=log_probs,
            num_features=int(num_features),
            action_slot_dim=int(action_slot_dim),
            next_state_target_dim=int(next_state_target_dim),
            single_eval_pos=int(single_eval_pos),
        )
        _copy_normalizer_state(pack_loss_algo, fit_loss_algo)
        fit_loss = _fill_and_measure_loss(
            fit_loss_algo,
            pack=pack,
            values=values,
            log_probs=log_probs,
            num_features=int(num_features),
            action_slot_dim=int(action_slot_dim),
            next_state_target_dim=int(next_state_target_dim),
            single_eval_pos=int(single_eval_pos),
        )
        rollout_diff = _tensor_diff_summary(pack_loss["rollout_tensors"], fit_loss["rollout_tensors"])
        grad_diff = _gradient_diff_summary(pack_loss["grad_vector"], fit_loss["grad_vector"])
        loss_diff = _loss_diff_summary(pack_loss, fit_loss)
        buffer_diff = _diff_buffer_summaries(pack_loss["buffer_summary"], fit_loss["buffer_summary"])
    finally:
        pack_loss_vec_env.close()
        fit_loss_vec_env.close()
        del pack_loss_algo, fit_loss_algo

    pack_update_algo, pack_update_vec_env, fit_update_algo, fit_update_vec_env, _, _ = _new_pair()
    try:
        pre_update_param_diff = _param_diff_summary(pack_update_algo.policy, fit_update_algo.policy)
        _seed_all(int(build_seed) + 9001)
        pack_update, pack_bridge = _run_update_from_pack(
            algo=pack_update_algo,
            pack=pack,
            values=values,
            log_probs=log_probs,
            num_features=int(num_features),
            action_slot_dim=int(action_slot_dim),
            next_state_target_dim=int(next_state_target_dim),
            single_eval_pos=int(single_eval_pos),
        )
        _copy_normalizer_state(pack_update_algo, fit_update_algo)
        _seed_all(int(build_seed) + 9001)
        fit_update, fit_bridge = _run_update_from_pack(
            algo=fit_update_algo,
            pack=pack,
            values=values,
            log_probs=log_probs,
            num_features=int(num_features),
            action_slot_dim=int(action_slot_dim),
            next_state_target_dim=int(next_state_target_dim),
            single_eval_pos=int(single_eval_pos),
        )
        post_update_param_diff = _param_diff_summary(pack_update_algo.policy, fit_update_algo.policy)
        logger_diff = _dict_numeric_diff(
            _logger_values(pack_update_algo),
            _logger_values(fit_update_algo),
            [
                "train/loss",
                "train/policy_gradient_loss",
                "train/value_loss",
                "train/entropy_loss",
                "train/approx_kl",
                "train/clip_fraction",
                "train/explained_variance",
                "train/explained_variance_normalized",
                "train/explained_variance_raw_value_target",
                "train/explained_variance_raw_discounted_return",
                "train/normalized_q_value_weight",
                "train/next_state_flow_matching_weight",
            ],
        )
    finally:
        pack_update_vec_env.close()
        fit_update_vec_env.close()
        del pack_update_algo, fit_update_algo

    hard_pass = bool(
        pre_update_param_diff["policy_param_max_abs_diff"] == 0.0
        and rollout_diff["max_abs_rollout_tensor_delta"] == 0.0
        and grad_diff["grad_delta_max_abs"] <= 1e-7
        and post_update_param_diff["policy_param_max_abs_diff"] <= 1e-7
        and pack_update.get("exception") is None
        and fit_update.get("exception") is None
    )
    return {
        "label": str(label),
        "hard_pass": bool(hard_pass),
        "pack_meta": meta,
        "pack_summary": pack_summary,
        "collector_normalizer_after_collect": normalizer_after_collect,
        "initial_param_diff": pre_update_param_diff,
        "loss_diff": loss_diff,
        "buffer_diff": buffer_diff,
        "rollout_diff": rollout_diff,
        "gradient_diff": grad_diff,
        "one_train_update_compare": {
            "pack_update": pack_update,
            "fit_model_update": fit_update,
            "pack_bridge": pack_bridge,
            "fit_model_bridge": fit_bridge,
            "post_update_param_diff": post_update_param_diff,
            "logger_diff": logger_diff,
        },
        "bridge_summary": bridge_summary,
        "bridge_kwargs_contract": {
            key: _jsonable(value)
            for key, value in bridge_kwargs.items()
            if key not in {"model", "env_prior"}
        },
        "contract": {
            "ppo_n_epochs": int(ppo_n_epochs),
            "ppo_batch_size": int(ppo_batch_size),
            "ppo_vf_coef": float(ppo_vf_coef),
            "ppo_normalize_advantage": bool(ppo_normalize_advantage),
            "ppo_target_kl": None if ppo_target_kl is None else float(ppo_target_kl),
            "normalized_q_value_weight": 0.0,
            "next_state_flow_matching_weight": 0.0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Numerically compare the sampled-topology healthy prior generator and "
            "current gym/prior pack PPO update path against the fit_model PPO bridge."
        )
    )
    parser.add_argument("--device", default=_default_device())
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--build-seed", type=int, default=5052)
    parser.add_argument("--env-seed", type=int, default=6060)
    parser.add_argument("--gym-seed", type=int, default=7070)
    parser.add_argument("--n-envs", type=int, default=2)
    parser.add_argument("--n-steps", type=int, default=128)
    parser.add_argument("--single-eval-pos", type=int, default=64)
    parser.add_argument("--ppo-batch-size", type=int, default=256)
    parser.add_argument("--ppo-n-epochs", type=int, default=4)
    parser.add_argument("--ppo-vf-coef", type=float, default=0.1)
    parser.add_argument("--ppo-normalize-advantage", type=str2bool, default=True)
    parser.add_argument("--ppo-target-kl", type=float, default=None)
    parser.add_argument("--topology-state-gain-min", type=float, default=0.7)
    parser.add_argument("--topology-action-gain-min", type=float, default=0.06)
    parser.add_argument("--topology-state-to-action-ratio-max", type=float, default=10.0)
    parser.add_argument("--topology-max-attempts", type=int, default=4096)
    parser.add_argument("--sb3-reward-normalization-enabled", type=str2bool, default=True)
    parser.add_argument("--sb3-observation-normalization-enabled", type=str2bool, default=True)
    parser.add_argument("--sb3-observation-normalization-clip", type=float, default=10.0)
    parser.add_argument("--sb3-observation-normalization-epsilon", type=float, default=1e-8)
    parser.add_argument("--gym-env-ids", default="BipedalWalker-v3,Pendulum-v1")
    parser.add_argument("--run-gym-parity", type=str2bool, default=True)
    args = parser.parse_args()

    if int(args.ppo_batch_size) > int(args.n_envs) * int(args.n_steps):
        raise ValueError("ppo_batch_size must not exceed n_envs * n_steps.")

    output_path = Path(str(args.output_json)).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    device_obj = torch.device(str(args.device))
    cfg = _load_contract_config(args)
    env_cfg = cfg["prior"]["environment"]
    result: dict[str, Any] = {
        "script": "phase2_fit_model_pack_ppo_migration_parity",
        "args": _jsonable(vars(args)),
        "contract": {
            "generator": "sampled_topology_conditioned",
            "batch_parallel_backend": env_cfg.get("batch_parallel_backend"),
            "batch_vectorized_grouping": env_cfg.get("batch_vectorized_grouping"),
            "reward_topology_conditioned_sampling_enabled": bool(
                env_cfg.get("reward_topology_conditioned_sampling_enabled", False)
            ),
            "reward_state_input_gain_fraction_conditioned_min": float(
                env_cfg["reward_state_input_gain_fraction_conditioned_min"]
            ),
            "reward_action_input_gain_fraction_conditioned_min": float(
                env_cfg["reward_action_input_gain_fraction_conditioned_min"]
            ),
            "reward_state_to_action_gain_ratio_conditioned_max": float(
                env_cfg["reward_state_to_action_gain_ratio_conditioned_max"]
            ),
            "reward_topology_conditioned_sampling_max_attempts": int(
                env_cfg["reward_topology_conditioned_sampling_max_attempts"]
            ),
            "reinforce_aux_enabled": bool(env_cfg.get("reinforce_aux_enabled", True)),
            "normalized_q_value_weight": float(env_cfg.get("normalized_q_value_weight", 0.0)),
            "next_state_flow_matching_weight": float(env_cfg.get("next_state_flow_matching_weight", 0.0)),
            "fit_model_n_samples": int(cfg["prior"]["n_samples"]),
            "fit_model_batch_size": int(cfg["dataloader"]["batch_size"]),
            "fit_model_ppo_single_eval_pos": cfg["optimizer"].get("ppo_single_eval_pos"),
            "fit_model_pg_phase_log_file": cfg["optimizer"].get("pg_phase_log_file"),
            "ppo_batch_size": int(cfg["optimizer"]["ppo_batch_size"]),
            "ppo_n_epochs": int(cfg["optimizer"]["ppo_n_epochs"]),
            "ppo_vf_coef": float(cfg["optimizer"]["ppo_vf_coef"]),
            "ppo_normalize_advantage": bool(cfg["optimizer"]["ppo_normalize_advantage"]),
            "ppo_target_kl": cfg["optimizer"]["ppo_target_kl"],
            "ppo_space_contract": cfg["optimizer"].get("ppo_space_contract"),
            "ppo_value_head_impl": cfg["optimizer"].get("ppo_value_head_impl"),
            "ppo_runtime_normalized_q_value_weight_override": cfg["optimizer"].get(
                "ppo_runtime_normalized_q_value_weight_override"
            ),
            "ppo_runtime_next_state_flow_matching_weight_override": cfg["optimizer"].get(
                "ppo_runtime_next_state_flow_matching_weight_override"
            ),
        },
    }
    result["sampled_topology_generator_parity"] = _compare_sampled_topology_generator(
        cfg=cfg,
        device_obj=device_obj,
        build_seed=int(args.build_seed),
        env_seed=int(args.env_seed),
        n_envs=int(args.n_envs),
        n_steps=int(args.n_steps),
        ppo_batch_size=int(args.ppo_batch_size),
        ppo_n_epochs=int(args.ppo_n_epochs),
        ppo_vf_coef=float(args.ppo_vf_coef),
        ppo_normalize_advantage=bool(args.ppo_normalize_advantage),
        ppo_target_kl=args.ppo_target_kl,
        single_eval_pos=int(args.single_eval_pos),
        sb3_reward_normalization_enabled=bool(args.sb3_reward_normalization_enabled),
        sb3_observation_normalization_enabled=bool(args.sb3_observation_normalization_enabled),
        sb3_observation_normalization_clip=float(args.sb3_observation_normalization_clip),
        sb3_observation_normalization_epsilon=float(args.sb3_observation_normalization_epsilon),
    )
    result["prior_pack_ppo_update_parity"] = _compare_update_from_pack(
        label="prior_sampled_topology_pack",
        cfg=cfg,
        device_obj=device_obj,
        build_seed=int(args.build_seed),
        collect_seed=int(args.env_seed),
        n_envs=int(args.n_envs),
        n_steps=int(args.n_steps),
        ppo_batch_size=int(args.ppo_batch_size),
        ppo_n_epochs=int(args.ppo_n_epochs),
        ppo_vf_coef=float(args.ppo_vf_coef),
        ppo_normalize_advantage=bool(args.ppo_normalize_advantage),
        ppo_target_kl=args.ppo_target_kl,
        single_eval_pos=int(args.single_eval_pos),
        sb3_reward_normalization_enabled=bool(args.sb3_reward_normalization_enabled),
        sb3_observation_normalization_enabled=bool(args.sb3_observation_normalization_enabled),
        sb3_observation_normalization_clip=float(args.sb3_observation_normalization_clip),
        sb3_observation_normalization_epsilon=float(args.sb3_observation_normalization_epsilon),
        gym_env_ids=None,
    )
    if bool(args.run_gym_parity):
        result["gym_pack_ppo_update_parity"] = _compare_update_from_pack(
            label="gym_pack",
            cfg=cfg,
            device_obj=device_obj,
            build_seed=int(args.build_seed),
            collect_seed=int(args.gym_seed),
            n_envs=int(args.n_envs),
            n_steps=int(args.n_steps),
            ppo_batch_size=int(args.ppo_batch_size),
            ppo_n_epochs=int(args.ppo_n_epochs),
            ppo_vf_coef=float(args.ppo_vf_coef),
            ppo_normalize_advantage=bool(args.ppo_normalize_advantage),
            ppo_target_kl=args.ppo_target_kl,
            single_eval_pos=int(args.single_eval_pos),
            sb3_reward_normalization_enabled=bool(args.sb3_reward_normalization_enabled),
            sb3_observation_normalization_enabled=bool(args.sb3_observation_normalization_enabled),
            sb3_observation_normalization_clip=float(args.sb3_observation_normalization_clip),
            sb3_observation_normalization_epsilon=float(args.sb3_observation_normalization_epsilon),
            gym_env_ids=_normalize_env_id_list(str(args.gym_env_ids), int(args.n_envs)),
        )
    result["hard_pass"] = bool(
        result["sampled_topology_generator_parity"]["hard_pass"]
        and result["prior_pack_ppo_update_parity"]["hard_pass"]
        and (
            not bool(args.run_gym_parity)
            or result["gym_pack_ppo_update_parity"]["hard_pass"]
        )
    )
    output_path.write_text(json.dumps(_jsonable(result), indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(_jsonable(result), indent=2, sort_keys=True))
    print(f"[phase2-fit-model-pack-ppo-migration-parity] wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
