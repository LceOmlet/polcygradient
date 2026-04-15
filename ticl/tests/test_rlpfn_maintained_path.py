from copy import deepcopy
import math
import random

import numpy as np
import torch

from ticl.distributions import parse_distributions, sample_distributions
from ticl.model_configs import get_model_default_config
from ticl.models.encoders import Linear
from ticl.models.tabpfn import TabPFN
from ticl.priors.maintained_exact_scm import (
    coerce_bool,
    env_input_layout,
    env_obs_input_dim,
    env_uses_reference_semantics,
    expand_env_value_to_list,
    finalize_env_semantics_summary,
    merge_env_semantics_summary,
    new_env_semantics_accumulator,
    resolve_reference_semantics_enabled,
    resolve_reinforce_action_rms_eps,
    resolve_reinforce_action_transform,
    resolve_reinforce_reward_rms_eps,
    resolve_reinforce_reward_tanh_bound,
    resolve_reinforce_reward_tanh_c,
    resolve_reinforce_reward_transform,
    resolve_scalar,
    resolve_state_full_rms_target,
    resolve_state_full_rms_enabled,
    resolve_state_input_scale,
    resolve_state_input_scale_enabled,
    resolve_strict_joint_transition_enabled,
    resolve_terminal_bonus_scale_max,
    resolve_terminal_bonus_scale_min,
    resolve_terminal_bonus_tanh_c,
    resolve_terminal_reset_count_target,
    resolve_terminal_reset_enabled,
    summarize_env_semantics,
    transition_reference_mode,
)
from ticl.priors.environment_prior import EnvironmentPrior
from ticl.rlpfn_maintained_path import resolve_rlpfn_token_layout, validate_rlpfn_maintained_path_config
from ticl.train import _build_policy_step_fn, _compute_policy_rollout_chunk_loss
from ticl.priors.maintained_policy_rollout import (
    normalize_policy_objective_kind,
    policy_rollout_objective_flags,
    resolve_alpha_grad_local_coordinate_enabled,
    resolve_alpha_grad_one_hop_replay_enabled,
    resolve_alpha_grad_unit_grad_delta,
    resolve_alpha_grad_unit_grad_enabled,
    resolve_alpha_grad_variance_eps,
    resolve_batch_parallel_backend,
    resolve_batch_parallel_workers,
    resolve_batch_shared_environment,
    resolve_batch_vectorized_grouping,
    resolve_batch_vectorized_strict_rng_match,
    resolve_pg_markov_adjacent_replay_enabled,
    resolve_pg_markov_adjacent_replay_sample_prob,
    resolve_pg_one_hop_replay_enabled,
    resolve_pg_replay_window_depth,
)
from ticl.priors.maintained_policy_gradient_loss import (
    attach_common_rollout_diagnostics,
    prepare_policy_gradient_loss_request,
    slice_group_traces_after_eval,
)


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _legacy_coerce_bool(value):
    if torch.is_tensor(value):
        if value.dtype == torch.bool:
            return value
        if value.numel() == 1:
            return bool(value.item())
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off", ""}
    return bool(value)


def _legacy_resolve_strict_joint_transition_enabled(h):
    enabled = h.get("strict_joint_transition_enabled", False)
    return bool(_legacy_coerce_bool(enabled))


def _legacy_resolve_reference_semantics_enabled(h):
    return _legacy_resolve_strict_joint_transition_enabled(h)


def _legacy_env_uses_reference_semantics(env):
    enabled = env.get(
        "reference_semantics_enabled",
        env.get("strict_joint_transition_enabled", False),
    )
    return bool(_legacy_coerce_bool(enabled))


def _legacy_transition_reference_mode(family, reference_semantics_enabled, gp_forward_mode=None):
    family_str = str(family)
    if family_str == "gp" and bool(reference_semantics_enabled):
        mode = str(gp_forward_mode or "exact").strip().lower()
        if mode == "fixed_cost":
            return "gp_fixed_cost"
        return "gp_exact"
    return f"{family_str}_{'exact' if bool(reference_semantics_enabled) else 'legacy'}"


def _legacy_env_obs_input_dim(obs_dim, *, reference_semantics_enabled=False):
    return 0 if bool(reference_semantics_enabled) else int(max(0, int(obs_dim)))


def _legacy_env_input_layout(
    state_dim,
    obs_dim,
    action_dim,
    noise_dim,
    zero_pad_dim,
    *,
    reference_semantics_enabled=False,
):
    state_dim = int(max(0, int(state_dim)))
    obs_dim = int(max(0, int(obs_dim)))
    action_dim = int(max(0, int(action_dim)))
    noise_dim = int(max(0, int(noise_dim)))
    zero_pad_dim = int(max(0, int(zero_pad_dim)))
    obs_input_dim = _legacy_env_obs_input_dim(
        obs_dim,
        reference_semantics_enabled=reference_semantics_enabled,
    )
    action_start = state_dim + obs_input_dim
    noise_start = action_start + action_dim
    zero_start = noise_start + noise_dim
    total_dim = zero_start + zero_pad_dim
    return {
        "total_dim": int(total_dim),
        "include_obs": bool(obs_input_dim > 0),
        "obs_input_dim": int(obs_input_dim),
        "obs_start": int(state_dim) if obs_input_dim > 0 else None,
        "action_start": int(action_start),
        "noise_start": int(noise_start),
        "zero_start": int(zero_start),
    }


def _legacy_resolve_state_input_scale_enabled(h):
    return bool(_legacy_coerce_bool(h.get("state_input_scale_enabled", False)))


def _legacy_resolve_state_full_rms_enabled(h):
    return bool(_legacy_coerce_bool(h.get("state_full_rms_enabled", False)))


def _legacy_resolve_terminal_reset_enabled(h):
    return bool(_legacy_coerce_bool(h.get("terminal_reset_enabled", False)))


def _legacy_resolve_scalar(value):
    return float(sample_distributions({"value": value})["value"])


def _legacy_resolve_state_input_scale(h):
    v = _legacy_resolve_scalar(h.get("state_input_scale", 1.0))
    if not math.isfinite(v):
        return 1.0
    return float(max(1e-6, v))


def _legacy_resolve_state_full_rms_target(h):
    v = _legacy_resolve_scalar(h.get("state_full_rms_target", 1.0))
    if (not math.isfinite(v)) or v <= 0.0:
        return 1.0
    return float(v)


def _legacy_resolve_reinforce_reward_transform(h):
    mode = str(h.get("reinforce_reward_transform", "none")).strip().lower()
    if mode not in {"none", "tanh", "rms", "clip"}:
        mode = "none"
    return mode


def _legacy_resolve_reinforce_reward_rms_eps(h):
    v = _legacy_resolve_scalar(h.get("reinforce_reward_rms_eps", 1e-6))
    if (not math.isfinite(v)) or v <= 0.0:
        return 1e-6
    return float(v)


def _legacy_resolve_reinforce_reward_tanh_c(h):
    v = _legacy_resolve_scalar(h.get("reinforce_reward_tanh_c", 1.0))
    if (not math.isfinite(v)) or v <= 0.0:
        return 1.0
    return float(v)


def _legacy_resolve_reinforce_reward_tanh_bound(h):
    v = _legacy_resolve_scalar(h.get("reinforce_reward_tanh_bound", 2.0))
    if (not math.isfinite(v)) or v <= 0.0:
        return 2.0
    return float(v)


def _legacy_resolve_reinforce_action_transform(h):
    mode = str(h.get("reinforce_action_transform", "none")).strip().lower()
    if mode not in {"tanh", "rms", "none", "clip"}:
        mode = "none"
    return mode


def _legacy_resolve_reinforce_action_rms_eps(h):
    v = _legacy_resolve_scalar(h.get("reinforce_action_rms_eps", 1e-6))
    if (not math.isfinite(v)) or v <= 0.0:
        return 1e-6
    return float(v)


def _legacy_resolve_terminal_reset_count_target(h):
    v = _legacy_resolve_scalar(h.get("terminal_reset_count_target", 0))
    if not math.isfinite(v):
        return 0.0
    return float(max(0.0, float(v)))


def _legacy_resolve_terminal_bonus_tanh_c(h):
    v = _legacy_resolve_scalar(h.get("terminal_bonus_tanh_c", 10.0))
    if (not math.isfinite(v)) or v <= 0.0:
        return 10.0
    return float(v)


def _legacy_resolve_terminal_bonus_scale_min(h):
    v = _legacy_resolve_scalar(h.get("terminal_bonus_scale_min", 1.0))
    if not math.isfinite(v):
        return 1.0
    return float(v)


def _legacy_resolve_terminal_bonus_scale_max(h):
    v = _legacy_resolve_scalar(h.get("terminal_bonus_scale_max", 2.0))
    if not math.isfinite(v):
        return 2.0
    return float(v)


def _legacy_normalize_policy_objective_kind(policy_objective_kind):
    objective_kind = str(policy_objective_kind).strip().lower()
    if objective_kind not in {"policy_gradient", "first_policy_gradient", "reinforce", "alpha_grad"}:
        raise ValueError(f"Unknown policy objective kind: {policy_objective_kind}")
    return objective_kind


def _legacy_policy_rollout_objective_flags(policy_objective_kind):
    objective_kind = _legacy_normalize_policy_objective_kind(policy_objective_kind)
    return {
        "objective_kind": objective_kind,
        "sample_action": objective_kind in {"first_policy_gradient", "reinforce", "alpha_grad"},
        "collect_log_probs": objective_kind in {"reinforce", "alpha_grad"},
        "detach_action_in_env": objective_kind == "reinforce",
        "first_policy_gradient": objective_kind == "first_policy_gradient",
        "reinforce": objective_kind == "reinforce",
        "alpha_grad": objective_kind == "alpha_grad",
    }


def _legacy_resolve_alpha_grad_variance_eps(h):
    v = _legacy_resolve_scalar(h.get("alpha_grad_variance_eps", 1e-6))
    return max(float(v), 0.0)


def _legacy_resolve_alpha_grad_local_coordinate_enabled(h):
    return _legacy_coerce_bool(h.get("alpha_grad_local_coordinate_enabled", True))


def _legacy_resolve_alpha_grad_unit_grad_enabled(h):
    return _legacy_coerce_bool(h.get("alpha_grad_unit_grad_enabled", True))


def _legacy_resolve_alpha_grad_unit_grad_delta(h):
    v = _legacy_resolve_scalar(h.get("alpha_grad_unit_grad_delta", 1e-6))
    if not math.isfinite(v) or v < 0.0:
        return 1e-6
    return float(v)


def _legacy_resolve_pg_one_hop_replay_enabled(h):
    if "pg_one_hop_replay_enabled" in h:
        return _legacy_coerce_bool(h.get("pg_one_hop_replay_enabled", True))
    return _legacy_coerce_bool(h.get("alpha_grad_one_hop_replay_enabled", True))


def _legacy_resolve_pg_replay_window_depth(h):
    del h
    return 1


def _legacy_resolve_pg_markov_adjacent_replay_enabled(h):
    return _legacy_coerce_bool(h.get("pg_markov_adjacent_replay_enabled", False))


def _legacy_resolve_pg_markov_adjacent_replay_sample_prob(h):
    v = _legacy_resolve_scalar(h.get("pg_markov_adjacent_replay_sample_prob", 0.125))
    if not math.isfinite(v):
        return 0.125
    return float(min(1.0, max(0.0, v)))


def _legacy_resolve_alpha_grad_one_hop_replay_enabled(h):
    return _legacy_resolve_pg_one_hop_replay_enabled(h)


def _legacy_resolve_batch_parallel_workers(config, batch_size):
    workers_cfg = config.get("batch_parallel_workers", 1)
    if isinstance(workers_cfg, dict) and "distribution" in workers_cfg:
        workers = int(sample_distributions({"v": workers_cfg})["v"])
    else:
        workers = int(workers_cfg)
    workers = max(1, workers)
    return min(int(batch_size), workers)


def _legacy_resolve_batch_parallel_backend(config):
    backend_cfg = config.get("batch_parallel_backend", "python_thread")
    if isinstance(backend_cfg, dict) and "distribution" in backend_cfg:
        backend = sample_distributions({"v": backend_cfg})["v"]
    else:
        backend = backend_cfg
    backend = str(backend).strip().lower()
    if backend not in {"python_thread", "torch_vectorized"}:
        backend = "python_thread"
    return backend


def _legacy_resolve_batch_shared_environment(config):
    shared_cfg = config.get("batch_shared_environment", False)
    if isinstance(shared_cfg, dict) and "distribution" in shared_cfg:
        shared = sample_distributions({"v": shared_cfg})["v"]
    else:
        shared = shared_cfg
    if isinstance(shared, str):
        token = shared.strip().lower()
        if token in {"1", "true", "yes", "on"}:
            return True
        if token in {"0", "false", "no", "off"}:
            return False
    return bool(shared)


def _legacy_resolve_batch_vectorized_strict_rng_match(config):
    strict_cfg = config.get("batch_vectorized_strict_rng_match", False)
    if isinstance(strict_cfg, dict) and "distribution" in strict_cfg:
        strict = sample_distributions({"v": strict_cfg})["v"]
    else:
        strict = strict_cfg
    if isinstance(strict, str):
        token = strict.strip().lower()
        if token in {"1", "true", "yes", "on"}:
            return True
        if token in {"0", "false", "no", "off"}:
            return False
    return bool(strict)


def _legacy_resolve_batch_vectorized_grouping(config):
    grouping_cfg = config.get("batch_vectorized_grouping", "structure")
    if isinstance(grouping_cfg, dict) and "distribution" in grouping_cfg:
        grouping = sample_distributions({"v": grouping_cfg})["v"]
    else:
        grouping = grouping_cfg
    grouping = str(grouping).strip().lower()
    if grouping not in {"structure", "family"}:
        grouping = "structure"
    return grouping


def _legacy_prepare_policy_gradient_loss_request(
    prior,
    *,
    n_samples,
    tbptt_window,
    policy_objective_kind="policy_gradient",
    normalize=None,
):
    n_samples = int(n_samples)
    objective_kind = _legacy_normalize_policy_objective_kind(policy_objective_kind)
    reinforce_enabled = objective_kind == "reinforce"
    first_pg_enabled = objective_kind == "first_policy_gradient"
    alpha_grad_enabled = objective_kind == "alpha_grad"
    if normalize is None:
        normalize = bool(prior.config.get("policy_gradient_normalize_rewards", False))
    tbptt_window_active = False
    tbptt_window_size = n_samples
    if tbptt_window is not None:
        w = int(tbptt_window)
        if 0 < w < n_samples:
            tbptt_window_active = True
            tbptt_window_size = w
    backend = _legacy_resolve_batch_parallel_backend(prior.config)
    grouping_mode = _legacy_resolve_batch_vectorized_grouping(prior.config)
    strict_rng_match = _legacy_resolve_batch_vectorized_strict_rng_match(prior.config)
    one_hop_replay_enabled = bool(_legacy_resolve_pg_one_hop_replay_enabled(prior.config))
    markov_adjacent_replay_enabled = bool(_legacy_resolve_pg_markov_adjacent_replay_enabled(prior.config))
    markov_adjacent_replay_sample_prob = float(
        _legacy_resolve_pg_markov_adjacent_replay_sample_prob(prior.config)
    )
    one_hop_tbptt_active = bool(
        one_hop_replay_enabled
        and tbptt_window_active
        and (reinforce_enabled or alpha_grad_enabled)
    )
    return {
        "n_samples": n_samples,
        "objective_kind": objective_kind,
        "reinforce_enabled": reinforce_enabled,
        "first_pg_enabled": first_pg_enabled,
        "alpha_grad_enabled": alpha_grad_enabled,
        "normalize": normalize,
        "tbptt_window_active": tbptt_window_active,
        "tbptt_window_size": tbptt_window_size,
        "backend": backend,
        "grouping_mode": grouping_mode,
        "strict_rng_match": strict_rng_match,
        "one_hop_replay_enabled": one_hop_replay_enabled,
        "markov_adjacent_replay_enabled": markov_adjacent_replay_enabled,
        "markov_adjacent_replay_sample_prob": markov_adjacent_replay_sample_prob,
        "one_hop_tbptt_active": one_hop_tbptt_active,
    }


def _legacy_slice_group_traces_after_eval(group_entries, start_idx):
    if not isinstance(group_entries, tuple):
        return group_entries
    sliced_entries = []
    for entry in group_entries:
        if not isinstance(entry, dict):
            continue
        roots = entry.get("action_mean_roots", None)
        mask = entry.get("action_mask", None)
        if not isinstance(roots, tuple):
            continue
        sliced_roots = tuple(roots[int(start_idx):])
        if len(sliced_roots) <= 0:
            continue
        sliced_entry = dict(entry)
        sliced_entry["action_mean_roots"] = sliced_roots
        if torch.is_tensor(mask):
            sliced_entry["action_mask"] = mask[int(start_idx):]
        sliced_entries.append(sliced_entry)
    return tuple(sliced_entries)


def _legacy_attach_common_rollout_diagnostics(rollout, stats):
    stats = deepcopy(stats)
    rewards = rollout["rewards"]
    rollout_profile = rollout.get("rollout_profile", None)
    if isinstance(rollout_profile, dict):
        for key in (
            "policy_cuda_ms",
            "transition_cuda_ms",
            "policy_wall_ms",
            "transition_wall_ms",
            "transition_y_wall_ms",
            "transition_x_wall_ms",
            "transition_group_wall_ms",
            "transition_group_launch_wall_ms",
            "transition_group_sync_wall_ms",
            "transition_env_pack_wall_ms",
            "transition_state_update_wall_ms",
            "transition_noise_wall_ms",
            "transition_fused_wall_ms",
            "transition_fused_launch_wall_ms",
            "transition_gp_first_projection_wall_ms",
            "transition_gp_second_projection_wall_ms",
            "transition_work_actual_est",
            "transition_work_padded_est",
            "transition_work_fill_ratio",
        ):
            stats[f"rollout_{key}"] = float(rollout_profile.get(key, 0.0) or 0.0)
        for key in (
            "transition_gp_projection_call_count",
            "transition_gp_rff_fused_call_count",
            "transition_gp_profile_group_count",
            "transition_gp_profile_sync_group_count",
            "transition_gp_shared_call_count",
            "transition_packed_env_input_group_count",
            "transition_packed_env_input_call_count",
            "transition_only_build_group_count",
            "transition_only_skipped_generator_count",
            "transition_fused_call_count",
            "transition_fused_group_count",
            "transition_fused_enabled",
            "transition_checkpoint_enabled",
            "transition_checkpoint_call_count",
            "transition_group_count",
            "transition_family_group_count",
            "transition_inner_grouping_structure_enabled",
            "transition_inner_min_bucket",
            "transition_bucket_max_batch",
            "transition_async_enabled",
            "transition_async_commit_in_stream",
            "noise_block_size",
            "env_count",
            "strict_joint_transition_count",
            "reference_semantics_count",
            "exact_scm_count",
            "exact_gp_count",
            "legacy_scm_count",
            "legacy_gp_count",
        ):
            stats[f"rollout_{key}"] = int(rollout_profile.get(key, 0) or 0)
        for key in (
            "transition_gp_shared_total_wall_ms",
            "transition_gp_shared_core_wall_ms",
            "transition_gp_shared_noise_wall_ms",
            "transition_gp_shared_checkpoint_wall_ms",
            "transition_gp_shared_post_wall_ms",
            "transition_setup_wall_ms",
            "transition_family_build_wall_ms",
            "transition_generator_build_wall_ms",
            "transition_gp_shared_build_wall_ms",
            "transition_bucket_mean_batch",
            "strict_joint_transition_share",
            "reference_semantics_share",
        ):
            stats[f"rollout_{key}"] = float(rollout_profile.get(key, 0.0) or 0.0)
        stats["rollout_noise_mode"] = rollout_profile.get("noise_mode", None)
        stats["rollout_transition_reference_mode"] = rollout_profile.get(
            "transition_reference_mode",
            None,
        )
    terminal_stats = rollout.get("terminal_stats", None)
    if isinstance(terminal_stats, dict):
        for key in (
            "terminal_count_mean",
            "terminal_count_min",
            "terminal_count_max",
            "terminal_count_target_mean",
            "terminal_count_target_min",
            "terminal_count_target_max",
        ):
            value = terminal_stats.get(key, None)
            if value is None:
                continue
            stats[key] = (
                value.detach()
                if torch.is_tensor(value)
                else torch.as_tensor(value, device=rewards.device, dtype=torch.float32)
            )
    return stats


def _legacy_expand_env_value_to_list(value, batch_size):
    batch_size = int(max(1, int(batch_size)))
    if torch.is_tensor(value):
        if value.ndim == 0:
            return [value.item()] * batch_size
        items = value.detach().cpu().reshape(-1).tolist()
    elif isinstance(value, np.ndarray):
        items = value.reshape(-1).tolist()
    elif isinstance(value, (list, tuple)):
        items = list(value)
    else:
        return [value] * batch_size
    if not items:
        return [None] * batch_size
    if len(items) == batch_size:
        return items
    if len(items) == 1:
        return items * batch_size
    if len(items) < batch_size:
        return items + [items[-1]] * (batch_size - len(items))
    return items[:batch_size]


def _legacy_new_env_semantics_accumulator():
    return {
        "env_count": 0,
        "strict_joint_transition_count": 0,
        "reference_semantics_count": 0,
        "exact_scm_count": 0,
        "exact_gp_count": 0,
        "fixed_gp_count": 0,
        "legacy_scm_count": 0,
        "legacy_gp_count": 0,
    }


def _legacy_finalize_env_semantics_summary(acc):
    if not isinstance(acc, dict):
        return None
    total = int(acc.get("env_count", 0) or 0)
    summary = dict(acc)
    summary["strict_joint_transition_share"] = float(
        float(summary.get("strict_joint_transition_count", 0) or 0) / float(max(1, total))
    )
    summary["reference_semantics_share"] = float(
        float(summary.get("reference_semantics_count", 0) or 0) / float(max(1, total))
    )
    modes = []
    if int(summary.get("exact_scm_count", 0) or 0) > 0:
        modes.append("scm_exact")
    if int(summary.get("exact_gp_count", 0) or 0) > 0:
        modes.append("gp_exact")
    if int(summary.get("fixed_gp_count", 0) or 0) > 0:
        modes.append("gp_fixed_cost")
    if int(summary.get("legacy_scm_count", 0) or 0) > 0:
        modes.append("scm_legacy")
    if int(summary.get("legacy_gp_count", 0) or 0) > 0:
        modes.append("gp_legacy")
    if len(modes) == 1:
        summary["transition_reference_mode"] = modes[0]
    elif len(modes) == 0:
        summary["transition_reference_mode"] = "unknown"
    else:
        summary["transition_reference_mode"] = "mixed"
    return summary


def _legacy_merge_env_semantics_summary(acc, summary):
    if not isinstance(summary, dict):
        return acc
    if acc is None:
        acc = _legacy_new_env_semantics_accumulator()
    for key in (
        "env_count",
        "strict_joint_transition_count",
        "reference_semantics_count",
        "exact_scm_count",
        "exact_gp_count",
        "fixed_gp_count",
        "legacy_scm_count",
        "legacy_gp_count",
    ):
        acc[key] = int(acc.get(key, 0) or 0) + int(summary.get(key, 0) or 0)
    return acc


def _legacy_summarize_env_semantics(env, batch_size):
    batch_size = int(max(1, int(batch_size)))
    families = _legacy_expand_env_value_to_list(env.get("family", "unknown"), batch_size)
    gp_modes = _legacy_expand_env_value_to_list(env.get("reference_gp_forward_mode", None), batch_size)
    references = [
        bool(v)
        for v in _legacy_expand_env_value_to_list(
            env.get("reference_semantics_enabled", env.get("strict_joint_transition_enabled", False)),
            batch_size,
        )
    ]
    stricts = [
        bool(v)
        for v in _legacy_expand_env_value_to_list(env.get("strict_joint_transition_enabled", False), batch_size)
    ]
    exact_scm_count = 0
    exact_gp_count = 0
    fixed_gp_count = 0
    legacy_scm_count = 0
    legacy_gp_count = 0
    for family, reference_enabled, gp_mode in zip(families, references, gp_modes):
        family_str = str(family)
        if family_str == "scm":
            if reference_enabled:
                exact_scm_count += 1
            else:
                legacy_scm_count += 1
        elif family_str == "gp":
            if reference_enabled:
                if str(gp_mode).strip().lower() == "fixed_cost":
                    fixed_gp_count += 1
                else:
                    exact_gp_count += 1
            else:
                legacy_gp_count += 1
    summary = {
        "env_count": int(batch_size),
        "strict_joint_transition_count": int(sum(1 for flag in stricts if flag)),
        "reference_semantics_count": int(sum(1 for flag in references if flag)),
        "exact_scm_count": int(exact_scm_count),
        "exact_gp_count": int(exact_gp_count),
        "fixed_gp_count": int(fixed_gp_count),
        "legacy_scm_count": int(legacy_scm_count),
        "legacy_gp_count": int(legacy_gp_count),
    }
    return _legacy_finalize_env_semantics_summary(summary)


def _small_exact_scm_env_cfg():
    cfg = deepcopy(get_model_default_config("rlpfn"))
    env_cfg = deepcopy(cfg["prior"]["environment"])
    env_cfg.update(
        {
            "state_dim": {"distribution": "uniform_int", "min": 8, "max": 8},
            "obs_dim": {"distribution": "uniform_int", "min": 6, "max": 6},
            "action_dim": {"distribution": "uniform_int", "min": 3, "max": 3},
            "noise_dim": {"distribution": "uniform_int", "min": 2, "max": 2},
            "zero_pad_dim": {"distribution": "uniform_int", "min": 0, "max": 0},
            "reward_dropout_randomize": False,
            "reward_dropout_ratio_min": 0.5,
            "reward_dropout_ratio_max": 0.5,
            "ctrl_reward_weight": 0.0,
            "ctrl_reward_enable_prob": 0.0,
            "survival_reward_weight": 0.0,
            "survival_reward_enable_prob": 0.0,
            "action_noise_train_std": 0.05,
            "action_noise_eval_std": 0.03,
        }
    )
    return cfg, env_cfg


def _capture_exact_scm_get_batch_trace():
    _seed_everything(11)
    cfg, env_cfg = _small_exact_scm_env_cfg()
    prior = EnvironmentPrior(env_cfg)
    num_features = int(cfg["prior"]["num_features"])
    x, y, _ = prior.get_batch(
        batch_size=2,
        n_samples=12,
        num_features=num_features,
        device="cpu",
        single_eval_pos=5,
    )
    info = prior.last_runtime_info
    return {
        "x_shape": tuple(x.shape),
        "y_shape": tuple(y.shape),
        "x00_head": [round(float(v), 6) for v in x[0, 0, :10]],
        "x51_head": [round(float(v), 6) for v in x[5, 1, :10]],
        "x00_meta": [round(float(v), 6) for v in x[0, 0, 400:408]],
        "x51_meta": [round(float(v), 6) for v in x[5, 1, 400:408]],
        "y_head": [round(float(v), 6) for v in y[:6, 0]],
        "y_tail": [round(float(v), 6) for v in y[-3:, 1]],
        "x_sum": round(float(x.sum()), 6),
        "y_sum": round(float(y.sum()), 6),
        "info0": {
            "family": info[0]["family"],
            "transition_reference_mode": info[0]["transition_reference_mode"],
            "strict_joint_transition_enabled": bool(info[0]["strict_joint_transition_enabled"]),
            "reference_semantics_enabled": bool(info[0]["reference_semantics_enabled"]),
            "reward_drop_frac_realized": round(float(info[0]["reward_drop_frac_realized"]), 6),
        },
        "info1": {
            "family": info[1]["family"],
            "transition_reference_mode": info[1]["transition_reference_mode"],
            "strict_joint_transition_enabled": bool(info[1]["strict_joint_transition_enabled"]),
            "reference_semantics_enabled": bool(info[1]["reference_semantics_enabled"]),
            "reward_drop_frac_realized": round(float(info[1]["reward_drop_frac_realized"]), 6),
        },
    }


def _capture_policy_step_trace():
    _seed_everything(20260320)
    model = TabPFN(
        n_out=1,
        n_features=434,
        emsize=32,
        nhead=1,
        nhid_factor=2,
        nlayers=1,
        y_encoder_layer=Linear(1, emsize=32),
        classification_task=False,
        y_encoder="linear",
        x_encoder_type="split_obs_action",
        x_obs_dim=404,
        x_action_dim=30,
        single_eval_causal=True,
    )
    model.eval()
    step_fn = _build_policy_step_fn(
        model,
        num_features=434,
        max_cache_len=16,
        kv_cache_mode="immutable",
        kv_cache_page_size=None,
        allow_grad_mutable_cache=False,
        pg_torch_compile=False,
    )

    obs_t = torch.linspace(-1.0, 1.0, steps=800, dtype=torch.float32).reshape(2, 400)
    action_t = torch.linspace(-0.5, 0.5, steps=60, dtype=torch.float32).reshape(2, 30)
    reward_t = torch.tensor([[0.25], [-0.75]], dtype=torch.float32)
    reward_mask_t = torch.ones((2, 1), dtype=torch.float32)
    phase_t = torch.tensor([[0.0], [1.0]], dtype=torch.float32)
    terminal_t = torch.tensor([[0.0], [1.0]], dtype=torch.float32)
    env_info = {
        "obs_slot_dim": 400,
        "action_slot_dim": 30,
        "action_dim": 30,
        "phase_t": phase_t,
        "terminal_t": terminal_t,
    }

    action_1, cache_1 = step_fn(obs_t, action_t, reward_t, reward_mask_t, None, 0, env_info)
    action_2, cache_2 = step_fn(obs_t.flip(-1), -action_t, reward_t * -0.5, reward_mask_t, cache_1, 1, env_info)

    return {
        "a1_head": [round(float(v), 6) for v in action_1[0, :8].detach()],
        "a1_row1_head": [round(float(v), 6) for v in action_1[1, :8].detach()],
        "a2_head": [round(float(v), 6) for v in action_2[0, :8].detach()],
        "a2_row1_head": [round(float(v), 6) for v in action_2[1, :8].detach()],
        "a1_sum": round(float(action_1.detach().sum()), 6),
        "a2_sum": round(float(action_2.detach().sum()), 6),
        "cache1_type": type(cache_1).__name__,
        "cache2_type": type(cache_2).__name__,
    }


def _capture_alpha_grad_fast_runner_gradient_trace():
    _seed_everything(20260320)
    cfg, env_cfg = _small_exact_scm_env_cfg()
    env_cfg["batch_parallel_backend"] = "torch_vectorized"
    env_cfg["batch_vectorized_grouping"] = "family"
    env_cfg["batch_parallel_workers"] = 1
    prior = EnvironmentPrior(env_cfg)
    model = TabPFN(
        n_out=1,
        n_features=int(cfg["prior"]["num_features"]),
        emsize=32,
        nhead=1,
        nhid_factor=2,
        nlayers=1,
        y_encoder_layer=Linear(1, emsize=32),
        classification_task=False,
        y_encoder="linear",
        x_encoder_type="split_obs_action",
        x_obs_dim=404,
        x_action_dim=30,
        single_eval_causal=True,
    )
    model.train()
    step_fn = _build_policy_step_fn(
        model,
        num_features=int(cfg["prior"]["num_features"]),
        max_cache_len=16,
        kv_cache_mode="immutable",
        kv_cache_page_size=None,
        allow_grad_mutable_cache=False,
        pg_torch_compile=False,
    )

    _seed_everything(515151)
    sampled = prior._sample_batch_hypers(4)
    for h in sampled:
        h["reward_dropout_enabled"] = False
        h["reward_dropout_randomize"] = False
        h["reward_dropout_ratio"] = 0.0
        h["action_noise_train_std"] = 0.05
        h["action_noise_eval_std"] = 0.03
    env_seeds = [17, 29, 43, 59]
    rollout_seeds = [101, 211, 307, 401]

    model.zero_grad(set_to_none=True)
    loss, rollout, stats = prior.rollout_policy_gradient_loss(
        policy_step_fn=step_fn,
        batch_size=4,
        n_samples=12,
        num_features=int(cfg["prior"]["num_features"]),
        device="cpu",
        single_eval_pos=6,
        collect_x=False,
        h_list_override=[dict(h) for h in sampled],
        env_seeds_override=env_seeds,
        rollout_seeds_override=rollout_seeds,
        policy_objective_kind="alpha_grad",
    )
    loss.backward()

    policy_head_grads = [
        p.grad.detach()
        for p in model.policy_action_head.parameters()
        if p.grad is not None
    ]
    policy_head_grad_sum = sum(float(g.abs().sum()) for g in policy_head_grads)
    policy_head_grad_max = max(float(g.abs().max()) for g in policy_head_grads)

    return {
        "loss": round(float(loss.detach()), 6),
        "reward_sum": round(float(rollout["rewards"].detach().sum()), 6),
        "objective": round(float(stats["objective"].detach()), 6),
        "alpha_grad_v0_mean": round(float(stats["alpha_grad_v0_mean"].detach()), 6),
        "alpha_grad_v1_mean": round(float(stats["alpha_grad_v1_mean"].detach()), 6),
        "alpha_grad_mix_abs_max": round(float(stats["alpha_grad_mix_abs_max"].detach()), 6),
        "alpha_grad_valid_share": round(float(stats["alpha_grad_valid_share"].detach()), 6),
        "rollout_transition_group_count": int(stats["rollout_transition_group_count"]),
        "rollout_transition_family_group_count": int(stats["rollout_transition_family_group_count"]),
        "rollout_exact_scm_count": int(stats["rollout_exact_scm_count"]),
        "rollout_transition_reference_mode": stats["rollout_transition_reference_mode"],
        "reinforce_log_prob_mean": round(float(stats["reinforce_log_prob_mean"].detach()), 6),
        "policy_head_grad_sum": round(float(policy_head_grad_sum), 6),
        "policy_head_grad_max": round(float(policy_head_grad_max), 6),
        "attn_in_proj_grad_sum": round(
            float(model.transformer_encoder.layers[0].self_attn.in_proj_weight.grad.detach().abs().sum()),
            6,
        ),
        "attn_in_proj_grad_max": round(
            float(model.transformer_encoder.layers[0].self_attn.in_proj_weight.grad.detach().abs().max()),
            6,
        ),
    }


def _capture_alpha_grad_train_step_trace():
    _seed_everything(20260320)
    cfg, env_cfg = _small_exact_scm_env_cfg()
    env_cfg["batch_parallel_backend"] = "torch_vectorized"
    env_cfg["batch_vectorized_grouping"] = "family"
    env_cfg["batch_parallel_workers"] = 1
    prior = EnvironmentPrior(env_cfg)
    model = TabPFN(
        n_out=1,
        n_features=int(cfg["prior"]["num_features"]),
        emsize=32,
        nhead=1,
        nhid_factor=2,
        nlayers=1,
        y_encoder_layer=Linear(1, emsize=32),
        classification_task=False,
        y_encoder="linear",
        x_encoder_type="split_obs_action",
        x_obs_dim=404,
        x_action_dim=30,
        single_eval_causal=True,
    )
    model.train()
    step_fn = _build_policy_step_fn(
        model,
        num_features=int(cfg["prior"]["num_features"]),
        max_cache_len=16,
        kv_cache_mode="immutable",
        kv_cache_page_size=None,
        allow_grad_mutable_cache=False,
        pg_torch_compile=False,
    )

    _seed_everything(515151)
    sampled = prior._sample_batch_hypers(4)
    for h in sampled:
        h["reward_dropout_enabled"] = False
        h["reward_dropout_randomize"] = False
        h["reward_dropout_ratio"] = 0.0
        h["action_noise_train_std"] = 0.05
        h["action_noise_eval_std"] = 0.03
    env_seeds = [17, 29, 43, 59]
    rollout_seeds = [101, 211, 307, 401]

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.999),
        weight_decay=0.0,
    )
    optimizer.zero_grad(set_to_none=True)
    loss, rollout, stats = _compute_policy_rollout_chunk_loss(
        env_prior=prior,
        policy_step_fn=step_fn,
        batch_size=4,
        n_samples=12,
        num_features=int(cfg["prior"]["num_features"]),
        device="cpu",
        single_eval_pos=6,
        collect_x=False,
        policy_rollout_checkpoint=False,
        policy_rollout_checkpoint_reentrant=True,
        pg_saved_tensors_cpu_offload=False,
        pg_saved_tensors_pin_memory=True,
        pg_tbptt_window=None,
        h_list_override=[dict(h) for h in sampled],
        env_seeds_override=env_seeds,
        rollout_seeds_override=rollout_seeds,
        rl_objective="alpha_grad",
    )
    params_before = {name: p.detach().clone() for name, p in model.named_parameters()}
    loss.backward()

    grad_abs_sum = 0.0
    grad_abs_max = 0.0
    grad_numel = 0
    for param in model.parameters():
        if param.grad is None:
            continue
        grad = param.grad.detach()
        grad_abs_sum += float(grad.abs().sum())
        grad_abs_max = max(grad_abs_max, float(grad.abs().max()))
        grad_numel += grad.numel()

    optimizer.step()

    param_delta_abs_sum = 0.0
    param_delta_abs_max = 0.0
    policy_head_delta_abs_sum = 0.0
    for name, param in model.named_parameters():
        delta = (param.detach() - params_before[name]).abs()
        param_delta_abs_sum += float(delta.sum())
        param_delta_abs_max = max(param_delta_abs_max, float(delta.max()))
        if name.startswith("policy_action_head"):
            policy_head_delta_abs_sum += float(delta.sum())

    return {
        "loss": round(float(loss.detach()), 6),
        "objective": round(float(stats["objective"].detach()), 6),
        "reward_sum": round(float(rollout["rewards"].detach().sum()), 6),
        "alpha_grad_v0_mean": round(float(stats["alpha_grad_v0_mean"].detach()), 6),
        "alpha_grad_v1_mean": round(float(stats["alpha_grad_v1_mean"].detach()), 6),
        "reinforce_log_prob_mean": round(float(stats["reinforce_log_prob_mean"].detach()), 6),
        "rollout_transition_reference_mode": stats["rollout_transition_reference_mode"],
        "rollout_exact_scm_count": int(stats["rollout_exact_scm_count"]),
        "grad_abs_sum": round(float(grad_abs_sum), 6),
        "grad_abs_max": round(float(grad_abs_max), 6),
        "grad_numel": int(grad_numel),
        "param_delta_abs_sum": round(float(param_delta_abs_sum), 6),
        "param_delta_abs_max": round(float(param_delta_abs_max), 6),
        "policy_head_delta_abs_sum": round(float(policy_head_delta_abs_sum), 6),
        "attn_in_proj_delta_abs_sum": round(
            float(
                (
                    model.transformer_encoder.layers[0].self_attn.in_proj_weight.detach()
                    - params_before["transformer_encoder.layers.0.self_attn.in_proj_weight"]
                ).abs().sum()
            ),
            6,
        ),
    }


def test_rlpfn_maintained_path_default_contract():
    cfg = get_model_default_config("rlpfn")
    layout = validate_rlpfn_maintained_path_config(cfg)

    assert cfg["prior"]["prior_type"] == "environment_only"
    assert cfg["prior"]["num_features"] == 434
    assert cfg["prior"]["environment"]["family"] == {
        "distribution": "meta_choice",
        "choice_values": ["scm"],
    }
    assert cfg["prior"]["environment"]["strict_joint_transition_enabled"] is True
    assert cfg["prior"]["environment"]["batch_parallel_backend"] == "torch_vectorized"
    assert cfg["prior"]["environment"]["batch_vectorized_grouping"] == "family"
    assert cfg["prior"]["environment"]["scm_standard_linear_init_enabled"] is True
    assert cfg["prior"]["environment"]["reinforce_sequence_replay_enabled"] is True
    assert cfg["prior"]["environment"]["reinforce_normalize_advantages"] is True
    assert cfg["prior"]["environment"]["reinforce_scale_advantages_by_suffix_episode_count"] is True
    assert cfg["prior"]["environment"]["reinforce_advantage_norm_eps"] == 1e-6
    assert cfg["prior"]["environment"]["reinforce_advantage_norm_clip"] == 10.0
    assert cfg["prior"]["environment"]["policy_gradient_weight"] == 0.4
    assert cfg["prior"]["environment"]["reinforce_aux_enabled"] is True
    assert cfg["prior"]["environment"]["reinforce_aux_backbone_query_pass_enabled"] is False
    assert cfg["prior"]["environment"]["normalized_q_value_weight"] == 0.0
    assert cfg["prior"]["environment"]["next_state_flow_matching_weight"] == 0.2
    assert cfg["prior"]["environment"]["next_state_flow_head_type"] == "rwkv_two_layer"
    assert cfg["prior"]["environment"]["reinforce_sequence_replay_store_legacy_targets"] is False
    assert cfg["prior"]["environment"]["reinforce_sequence_replay_share_train_token_encoding"] is True
    assert cfg["prior"]["environment"]["reinforce_sequence_replay_share_context_forward"] is False
    assert cfg["prior"]["environment"]["ctrl_reward_weight"] == {
        "distribution": "log_uniform",
        "min": 1e-3,
        "max": 3e-1,
    }
    assert cfg["prior"]["environment"]["ctrl_reward_enable_prob"] == 0.7
    assert cfg["prior"]["environment"]["survival_reward_weight"] == 0.0
    assert cfg["prior"]["environment"]["survival_reward_enable_prob"] == 0.0
    assert cfg["prior"]["environment"]["reinforce_action_transform"] == "clip"
    assert cfg["prior"]["environment"]["reinforce_action_rms_eps"] == 1e-6
    assert cfg["prior"]["environment"]["reinforce_action_clip_bound"] == 5.0
    assert cfg["optimizer"]["rl_objective"] == "ppo"
    assert cfg["optimizer"]["pg_tbptt_window"] is None
    assert cfg["optimizer"]["policy_rollout_checkpoint"] is False
    assert cfg["optimizer"]["pg_saved_tensors_cpu_offload"] is False
    assert cfg["optimizer"]["pg_saved_tensors_cpu_offload_scope"] == "policy"
    assert cfg["optimizer"]["pg_saved_tensors_cpu_offload_auto_disable_when_safe"] is False
    assert cfg["transformer"]["backbone"] == "rwkv7"
    assert cfg["transformer"]["rwkv_sequence_replay_checkpoint"] is True
    assert cfg["transformer"]["rwkv_sequence_replay_batch_chunk_size"] == 64
    assert cfg["transformer"]["rwkv_sequence_replay_token_budget"] == 262144
    assert cfg["transformer"]["x_encoder_type"] == "split_obs_action"
    assert cfg["transformer"]["x_obs_dim"] == 404
    assert cfg["transformer"]["x_action_dim"] == 30
    assert cfg["transformer"]["single_eval_causal"] is True
    assert cfg["prior"]["environment"]["action_noise_train_std"] == 0.0
    assert cfg["prior"]["environment"]["action_noise_eval_std"] == 0.0
    assert layout["default_num_features"] == 434
    assert layout["x_obs_dim"] == 404
    assert layout["x_action_dim"] == 30


def test_rlpfn_maintained_token_layout_helper_matches_terminal_toggle():
    _, env_cfg = _small_exact_scm_env_cfg()
    with_terminal = resolve_rlpfn_token_layout(env_cfg)
    assert with_terminal["terminal_token_enabled"] is True
    assert with_terminal["x_obs_dim"] == 404
    assert with_terminal["default_num_features"] == 434

    env_cfg_no_terminal = deepcopy(env_cfg)
    env_cfg_no_terminal["terminal_reset_enabled"] = False
    without_terminal = resolve_rlpfn_token_layout(env_cfg_no_terminal)
    assert without_terminal["terminal_token_enabled"] is False
    assert without_terminal["x_obs_dim"] == 403
    assert without_terminal["default_num_features"] == 433


def test_rlpfn_maintained_exact_scm_helper_contract():
    h = {
        "strict_joint_transition_enabled": "true",
    }
    env = {
        "strict_joint_transition_enabled": False,
        "reference_semantics_enabled": torch.tensor(True),
    }

    assert resolve_strict_joint_transition_enabled(h) is True
    assert resolve_reference_semantics_enabled(h) is True
    assert env_uses_reference_semantics(env) is True
    assert transition_reference_mode("scm", True) == "scm_exact"
    assert transition_reference_mode("scm", False) == "scm_legacy"
    assert transition_reference_mode("gp", True, gp_forward_mode="exact") == "gp_exact"
    assert transition_reference_mode("gp", True, gp_forward_mode="fixed_cost") == "gp_fixed_cost"


def test_rlpfn_maintained_exact_scm_helper_matches_legacy_inline_reference():
    bool_cases = [
        False,
        True,
        "true",
        "false",
        "off",
        "",
        0,
        1,
        torch.tensor(False),
        torch.tensor(True),
        torch.tensor([0, 1, 0], dtype=torch.int64),
        torch.tensor([True, False, True], dtype=torch.bool),
    ]
    for value in bool_cases:
        actual = coerce_bool(value)
        expected = _legacy_coerce_bool(value)
        if torch.is_tensor(expected):
            assert torch.equal(actual, expected)
        else:
            assert actual == expected

    h_cases = [
        {"strict_joint_transition_enabled": False},
        {"strict_joint_transition_enabled": "true"},
        {"strict_joint_transition_enabled": torch.tensor(True)},
        {"strict_joint_transition_enabled": 0},
    ]
    for h in h_cases:
        assert resolve_strict_joint_transition_enabled(h) == _legacy_resolve_strict_joint_transition_enabled(h)
        assert resolve_reference_semantics_enabled(h) == _legacy_resolve_reference_semantics_enabled(h)

    env_cases = [
        {"strict_joint_transition_enabled": False, "reference_semantics_enabled": torch.tensor(True)},
        {"strict_joint_transition_enabled": "true"},
        {"strict_joint_transition_enabled": False, "reference_semantics_enabled": False},
    ]
    for env in env_cases:
        assert env_uses_reference_semantics(env) == _legacy_env_uses_reference_semantics(env)

    mode_cases = [
        ("scm", True, None),
        ("scm", False, None),
        ("gp", True, "exact"),
        ("gp", True, "fixed_cost"),
        ("gp", False, "exact"),
    ]
    for family, enabled, gp_mode in mode_cases:
        assert transition_reference_mode(
            family,
            enabled,
            gp_forward_mode=gp_mode,
        ) == _legacy_transition_reference_mode(family, enabled, gp_forward_mode=gp_mode)


def test_rlpfn_maintained_exact_scm_layout_and_flag_helpers_match_legacy_inline_reference():
    layout_cases = [
        (8, 6, 3, 2, 0, False),
        (8, 6, 3, 2, 0, True),
        (0, 4, 1, 0, 2, False),
        (5, 12, 7, 4, 3, True),
    ]
    for state_dim, obs_dim, action_dim, noise_dim, zero_pad_dim, reference_semantics_enabled in layout_cases:
        assert env_obs_input_dim(
            obs_dim,
            reference_semantics_enabled=reference_semantics_enabled,
        ) == _legacy_env_obs_input_dim(
            obs_dim,
            reference_semantics_enabled=reference_semantics_enabled,
        )
        assert env_input_layout(
            state_dim,
            obs_dim,
            action_dim,
            noise_dim,
            zero_pad_dim,
            reference_semantics_enabled=reference_semantics_enabled,
        ) == _legacy_env_input_layout(
            state_dim,
            obs_dim,
            action_dim,
            noise_dim,
            zero_pad_dim,
            reference_semantics_enabled=reference_semantics_enabled,
        )
        assert EnvironmentPrior._env_input_layout(
            state_dim,
            obs_dim,
            action_dim,
            noise_dim,
            zero_pad_dim,
            reference_semantics_enabled=reference_semantics_enabled,
        ) == _legacy_env_input_layout(
            state_dim,
            obs_dim,
            action_dim,
            noise_dim,
            zero_pad_dim,
            reference_semantics_enabled=reference_semantics_enabled,
        )

    flag_cases = [
        {
            "state_input_scale_enabled": False,
            "state_full_rms_enabled": False,
            "terminal_reset_enabled": False,
        },
        {
            "state_input_scale_enabled": "true",
            "state_full_rms_enabled": "false",
            "terminal_reset_enabled": "1",
        },
        {
            "state_input_scale_enabled": torch.tensor(True),
            "state_full_rms_enabled": torch.tensor(False),
            "terminal_reset_enabled": torch.tensor(True),
        },
    ]
    for h in flag_cases:
        assert resolve_state_input_scale_enabled(h) == _legacy_resolve_state_input_scale_enabled(h)
        assert resolve_state_full_rms_enabled(h) == _legacy_resolve_state_full_rms_enabled(h)
        assert resolve_terminal_reset_enabled(h) == _legacy_resolve_terminal_reset_enabled(h)
        assert EnvironmentPrior._resolve_state_input_scale_enabled(h) == _legacy_resolve_state_input_scale_enabled(h)
        assert EnvironmentPrior._resolve_state_full_rms_enabled(h) == _legacy_resolve_state_full_rms_enabled(h)
        assert EnvironmentPrior._resolve_terminal_reset_enabled(h) == _legacy_resolve_terminal_reset_enabled(h)


def test_rlpfn_maintained_exact_scm_env_semantics_helpers_match_legacy_inline_reference():
    value_cases = [
        ("scm", 3),
        (torch.tensor(True), 2),
        (torch.tensor([1, 0, 1]), 3),
        (np.array(["exact", "fixed_cost"], dtype=object), 3),
        ([], 2),
        ([1], 4),
        ([1, 2], 4),
        ([1, 2, 3, 4], 2),
    ]
    for value, batch_size in value_cases:
        assert expand_env_value_to_list(value, batch_size) == _legacy_expand_env_value_to_list(value, batch_size)

    env = {
        "family": ["scm", "gp", "gp", "scm"],
        "reference_gp_forward_mode": [None, "exact", "fixed_cost", None],
        "reference_semantics_enabled": [True, True, True, False],
        "strict_joint_transition_enabled": [True, True, False, False],
    }
    expected_summary = _legacy_summarize_env_semantics(env, 4)
    assert summarize_env_semantics(env, 4) == expected_summary
    assert EnvironmentPrior._summarize_env_semantics(env, 4) == expected_summary

    acc_expected = _legacy_new_env_semantics_accumulator()
    acc_actual = new_env_semantics_accumulator()
    assert acc_actual == acc_expected
    assert EnvironmentPrior._new_env_semantics_accumulator() == acc_expected

    part_a = _legacy_summarize_env_semantics(
        {"family": ["scm", "gp"], "reference_semantics_enabled": [True, False], "strict_joint_transition_enabled": [True, False]},
        2,
    )
    part_b = _legacy_summarize_env_semantics(
        {"family": ["gp"], "reference_gp_forward_mode": ["fixed_cost"], "reference_semantics_enabled": [True], "strict_joint_transition_enabled": [True]},
        1,
    )
    merged_expected = _legacy_merge_env_semantics_summary(dict(acc_expected), part_a)
    merged_expected = _legacy_merge_env_semantics_summary(merged_expected, part_b)
    merged_actual = merge_env_semantics_summary(dict(acc_actual), part_a)
    merged_actual = merge_env_semantics_summary(merged_actual, part_b)
    assert merged_actual == merged_expected
    assert EnvironmentPrior._merge_env_semantics_summary(dict(acc_actual), part_a) == _legacy_merge_env_semantics_summary(
        dict(acc_expected),
        part_a,
    )
    assert finalize_env_semantics_summary(merged_actual) == _legacy_finalize_env_semantics_summary(merged_expected)
    assert EnvironmentPrior._finalize_env_semantics_summary(merged_actual) == _legacy_finalize_env_semantics_summary(
        merged_expected,
    )


def test_rlpfn_maintained_exact_scm_numeric_resolve_helpers_match_legacy_inline_reference():
    def _assert_seeded_equal(seed, actual_fn, legacy_fn):
        _seed_everything(seed)
        actual = actual_fn()
        _seed_everything(seed)
        expected = legacy_fn()
        if isinstance(actual, float) and isinstance(expected, float):
            if math.isnan(actual) and math.isnan(expected):
                return
        assert actual == expected

    def _parsed_dist(spec):
        return parse_distributions({"value": spec})["value"]

    scalar_cases = [
        1.5,
        float("nan"),
        -3.0,
        _parsed_dist({"distribution": "uniform", "min": 0.25, "max": 0.75}),
    ]
    for value in scalar_cases:
        _assert_seeded_equal(1234, lambda value=value: resolve_scalar(value), lambda value=value: _legacy_resolve_scalar(value))

    h_scalar_cases = [
        {
            "state_input_scale": 0.5,
            "state_full_rms_target": 1.2,
            "reinforce_reward_rms_eps": 1e-5,
            "reinforce_reward_tanh_c": 2.0,
            "reinforce_reward_tanh_bound": 7.0,
            "reinforce_action_rms_eps": 5e-6,
            "terminal_reset_count_target": 3.0,
            "terminal_bonus_tanh_c": 8.0,
            "terminal_bonus_scale_min": 0.5,
            "terminal_bonus_scale_max": 4.0,
        },
        {
            "state_input_scale": float("nan"),
            "state_full_rms_target": -1.0,
            "reinforce_reward_rms_eps": 0.0,
            "reinforce_reward_tanh_c": -2.0,
            "reinforce_reward_tanh_bound": float("nan"),
            "reinforce_action_rms_eps": -1.0,
            "terminal_reset_count_target": float("nan"),
            "terminal_bonus_tanh_c": 0.0,
            "terminal_bonus_scale_min": float("nan"),
            "terminal_bonus_scale_max": float("nan"),
        },
        {
            "state_input_scale": _parsed_dist({"distribution": "uniform", "min": 0.25, "max": 0.75}),
            "state_full_rms_target": _parsed_dist({"distribution": "uniform", "min": 0.8, "max": 1.4}),
            "reinforce_reward_rms_eps": _parsed_dist({"distribution": "uniform", "min": 1e-6, "max": 1e-4}),
            "reinforce_reward_tanh_c": _parsed_dist({"distribution": "uniform", "min": 0.5, "max": 2.5}),
            "reinforce_reward_tanh_bound": _parsed_dist({"distribution": "uniform", "min": 5.0, "max": 15.0}),
            "reinforce_action_rms_eps": _parsed_dist({"distribution": "uniform", "min": 1e-6, "max": 1e-4}),
            "terminal_reset_count_target": _parsed_dist({"distribution": "uniform", "min": 0.0, "max": 5.0}),
            "terminal_bonus_tanh_c": _parsed_dist({"distribution": "uniform", "min": 5.0, "max": 15.0}),
            "terminal_bonus_scale_min": _parsed_dist({"distribution": "uniform", "min": 0.5, "max": 2.0}),
            "terminal_bonus_scale_max": _parsed_dist({"distribution": "uniform", "min": 2.0, "max": 6.0}),
        },
    ]
    for h in h_scalar_cases:
        _assert_seeded_equal(
            2026,
            lambda h=h: resolve_state_input_scale(h),
            lambda h=h: _legacy_resolve_state_input_scale(h),
        )
        _assert_seeded_equal(
            2026,
            lambda h=h: EnvironmentPrior._resolve_state_input_scale(h),
            lambda h=h: _legacy_resolve_state_input_scale(h),
        )
        _assert_seeded_equal(
            2027,
            lambda h=h: resolve_state_full_rms_target(h),
            lambda h=h: _legacy_resolve_state_full_rms_target(h),
        )
        _assert_seeded_equal(
            2027,
            lambda h=h: EnvironmentPrior._resolve_state_full_rms_target(h),
            lambda h=h: _legacy_resolve_state_full_rms_target(h),
        )
        _assert_seeded_equal(
            2028,
            lambda h=h: resolve_reinforce_reward_rms_eps(h),
            lambda h=h: _legacy_resolve_reinforce_reward_rms_eps(h),
        )
        _assert_seeded_equal(
            2028,
            lambda h=h: EnvironmentPrior._resolve_reinforce_reward_rms_eps(h),
            lambda h=h: _legacy_resolve_reinforce_reward_rms_eps(h),
        )
        _assert_seeded_equal(
            2029,
            lambda h=h: resolve_reinforce_reward_tanh_c(h),
            lambda h=h: _legacy_resolve_reinforce_reward_tanh_c(h),
        )
        _assert_seeded_equal(
            2029,
            lambda h=h: EnvironmentPrior._resolve_reinforce_reward_tanh_c(h),
            lambda h=h: _legacy_resolve_reinforce_reward_tanh_c(h),
        )
        _assert_seeded_equal(
            2030,
            lambda h=h: resolve_reinforce_reward_tanh_bound(h),
            lambda h=h: _legacy_resolve_reinforce_reward_tanh_bound(h),
        )
        _assert_seeded_equal(
            2030,
            lambda h=h: EnvironmentPrior._resolve_reinforce_reward_tanh_bound(h),
            lambda h=h: _legacy_resolve_reinforce_reward_tanh_bound(h),
        )
        _assert_seeded_equal(
            2031,
            lambda h=h: resolve_reinforce_action_rms_eps(h),
            lambda h=h: _legacy_resolve_reinforce_action_rms_eps(h),
        )
        _assert_seeded_equal(
            2031,
            lambda h=h: EnvironmentPrior._resolve_reinforce_action_rms_eps(h),
            lambda h=h: _legacy_resolve_reinforce_action_rms_eps(h),
        )
        _assert_seeded_equal(
            2032,
            lambda h=h: resolve_terminal_reset_count_target(h),
            lambda h=h: _legacy_resolve_terminal_reset_count_target(h),
        )
        _assert_seeded_equal(
            2032,
            lambda h=h: EnvironmentPrior._resolve_terminal_reset_count_target(h),
            lambda h=h: _legacy_resolve_terminal_reset_count_target(h),
        )
        _assert_seeded_equal(
            2033,
            lambda h=h: resolve_terminal_bonus_tanh_c(h),
            lambda h=h: _legacy_resolve_terminal_bonus_tanh_c(h),
        )
        _assert_seeded_equal(
            2033,
            lambda h=h: EnvironmentPrior._resolve_terminal_bonus_tanh_c(h),
            lambda h=h: _legacy_resolve_terminal_bonus_tanh_c(h),
        )
        _assert_seeded_equal(
            2034,
            lambda h=h: resolve_terminal_bonus_scale_min(h),
            lambda h=h: _legacy_resolve_terminal_bonus_scale_min(h),
        )
        _assert_seeded_equal(
            2034,
            lambda h=h: EnvironmentPrior._resolve_terminal_bonus_scale_min(h),
            lambda h=h: _legacy_resolve_terminal_bonus_scale_min(h),
        )
        _assert_seeded_equal(
            2035,
            lambda h=h: resolve_terminal_bonus_scale_max(h),
            lambda h=h: _legacy_resolve_terminal_bonus_scale_max(h),
        )
        _assert_seeded_equal(
            2035,
            lambda h=h: EnvironmentPrior._resolve_terminal_bonus_scale_max(h),
            lambda h=h: _legacy_resolve_terminal_bonus_scale_max(h),
        )

    transform_cases = [
        {},
        {"reinforce_reward_transform": "tanh", "reinforce_action_transform": "none"},
        {"reinforce_reward_transform": "bad", "reinforce_action_transform": "bad"},
        {"reinforce_reward_transform": "CLIP", "reinforce_action_transform": "TANH"},
        {"reinforce_reward_transform": "none", "reinforce_action_transform": "clip"},
    ]
    for h in transform_cases:
        assert resolve_reinforce_reward_transform(h) == _legacy_resolve_reinforce_reward_transform(h)
        assert EnvironmentPrior._resolve_reinforce_reward_transform(h) == _legacy_resolve_reinforce_reward_transform(h)
        assert resolve_reinforce_action_transform(h) == _legacy_resolve_reinforce_action_transform(h)
        assert EnvironmentPrior._resolve_reinforce_action_transform(h) == _legacy_resolve_reinforce_action_transform(h)


def test_rlpfn_maintained_policy_rollout_helpers_match_legacy_inline_reference():
    objective_cases = ["policy_gradient", "first_policy_gradient", "reinforce", "alpha_grad", "ALPHA_GRAD"]
    for objective in objective_cases:
        assert normalize_policy_objective_kind(objective) == _legacy_normalize_policy_objective_kind(objective)
        assert EnvironmentPrior._normalize_policy_objective_kind(objective) == _legacy_normalize_policy_objective_kind(
            objective
        )
        assert policy_rollout_objective_flags(objective) == _legacy_policy_rollout_objective_flags(objective)
        assert EnvironmentPrior._policy_rollout_objective_flags(objective) == _legacy_policy_rollout_objective_flags(
            objective
        )

    invalid = "bad-objective"
    try:
        _legacy_normalize_policy_objective_kind(invalid)
    except ValueError as expected_error:
        expected_msg = str(expected_error)
    else:
        raise AssertionError("legacy normalize should reject invalid objective")
    for fn in (normalize_policy_objective_kind, EnvironmentPrior._normalize_policy_objective_kind):
        try:
            fn(invalid)
        except ValueError as actual_error:
            assert str(actual_error) == expected_msg
        else:
            raise AssertionError("normalize should reject invalid objective")

    def _assert_seeded_equal(seed, actual_fn, legacy_fn):
        _seed_everything(seed)
        actual = actual_fn()
        _seed_everything(seed)
        expected = legacy_fn()
        if isinstance(actual, float) and isinstance(expected, float):
            if math.isnan(actual) and math.isnan(expected):
                return
        assert actual == expected

    parsed = lambda spec: parse_distributions({"value": spec})["value"]
    h_cases = [
        {
            "alpha_grad_variance_eps": 1e-5,
            "alpha_grad_local_coordinate_enabled": "true",
            "alpha_grad_unit_grad_enabled": torch.tensor(False),
            "alpha_grad_unit_grad_delta": 5e-6,
            "pg_one_hop_replay_enabled": True,
            "pg_markov_adjacent_replay_enabled": True,
            "pg_markov_adjacent_replay_sample_prob": 0.25,
        },
        {
            "alpha_grad_variance_eps": parsed({"distribution": "uniform", "min": 1e-7, "max": 1e-4}),
            "alpha_grad_local_coordinate_enabled": "false",
            "alpha_grad_unit_grad_enabled": "true",
            "alpha_grad_unit_grad_delta": parsed({"distribution": "uniform", "min": 1e-7, "max": 1e-4}),
            "alpha_grad_one_hop_replay_enabled": True,
            "pg_markov_adjacent_replay_enabled": False,
            "pg_markov_adjacent_replay_sample_prob": parsed({"distribution": "uniform", "min": 0.0, "max": 1.0}),
        },
    ]
    for h in h_cases:
        _assert_seeded_equal(3101, lambda h=h: resolve_alpha_grad_variance_eps(h), lambda h=h: _legacy_resolve_alpha_grad_variance_eps(h))
        _assert_seeded_equal(
            3101,
            lambda h=h: EnvironmentPrior._resolve_alpha_grad_variance_eps(h),
            lambda h=h: _legacy_resolve_alpha_grad_variance_eps(h),
        )
        assert resolve_alpha_grad_local_coordinate_enabled(h) == _legacy_resolve_alpha_grad_local_coordinate_enabled(h)
        assert EnvironmentPrior._resolve_alpha_grad_local_coordinate_enabled(h) == _legacy_resolve_alpha_grad_local_coordinate_enabled(h)
        assert resolve_alpha_grad_unit_grad_enabled(h) == _legacy_resolve_alpha_grad_unit_grad_enabled(h)
        assert EnvironmentPrior._resolve_alpha_grad_unit_grad_enabled(h) == _legacy_resolve_alpha_grad_unit_grad_enabled(h)
        _assert_seeded_equal(
            3102,
            lambda h=h: resolve_alpha_grad_unit_grad_delta(h),
            lambda h=h: _legacy_resolve_alpha_grad_unit_grad_delta(h),
        )
        _assert_seeded_equal(
            3102,
            lambda h=h: EnvironmentPrior._resolve_alpha_grad_unit_grad_delta(h),
            lambda h=h: _legacy_resolve_alpha_grad_unit_grad_delta(h),
        )
        assert resolve_pg_one_hop_replay_enabled(h) == _legacy_resolve_pg_one_hop_replay_enabled(h)
        assert EnvironmentPrior._resolve_pg_one_hop_replay_enabled(h) == _legacy_resolve_pg_one_hop_replay_enabled(h)
        assert resolve_pg_replay_window_depth(h) == _legacy_resolve_pg_replay_window_depth(h)
        assert EnvironmentPrior._resolve_pg_replay_window_depth(h) == _legacy_resolve_pg_replay_window_depth(h)
        assert resolve_pg_markov_adjacent_replay_enabled(h) == _legacy_resolve_pg_markov_adjacent_replay_enabled(h)
        assert EnvironmentPrior._resolve_pg_markov_adjacent_replay_enabled(h) == _legacy_resolve_pg_markov_adjacent_replay_enabled(h)
        _assert_seeded_equal(
            3103,
            lambda h=h: resolve_pg_markov_adjacent_replay_sample_prob(h),
            lambda h=h: _legacy_resolve_pg_markov_adjacent_replay_sample_prob(h),
        )
        _assert_seeded_equal(
            3103,
            lambda h=h: EnvironmentPrior._resolve_pg_markov_adjacent_replay_sample_prob(h),
            lambda h=h: _legacy_resolve_pg_markov_adjacent_replay_sample_prob(h),
        )
        assert resolve_alpha_grad_one_hop_replay_enabled(h) == _legacy_resolve_alpha_grad_one_hop_replay_enabled(h)
        assert EnvironmentPrior._resolve_alpha_grad_one_hop_replay_enabled(h) == _legacy_resolve_alpha_grad_one_hop_replay_enabled(h)


def test_rlpfn_maintained_fast_runner_config_helpers_match_legacy_inline_reference():
    def _assert_seeded_equal(seed, actual_fn, legacy_fn):
        _seed_everything(seed)
        actual = actual_fn()
        _seed_everything(seed)
        expected = legacy_fn()
        assert actual == expected

    config_cases = [
        {
            "batch_parallel_workers": 4,
            "batch_parallel_backend": "torch_vectorized",
            "batch_shared_environment": "true",
            "batch_vectorized_strict_rng_match": "false",
            "batch_vectorized_grouping": "family",
        },
        {
            "batch_parallel_workers": 0,
            "batch_parallel_backend": "bad-backend",
            "batch_shared_environment": "off",
            "batch_vectorized_strict_rng_match": "on",
            "batch_vectorized_grouping": "bad-grouping",
        },
    ]
    for config in config_cases:
        _assert_seeded_equal(
            4101,
            lambda config=config: resolve_batch_parallel_workers(config, 5),
            lambda config=config: _legacy_resolve_batch_parallel_workers(config, 5),
        )
        _assert_seeded_equal(
            4101,
            lambda config=config: EnvironmentPrior(config)._resolve_batch_parallel_workers(5),
            lambda config=config: _legacy_resolve_batch_parallel_workers(config, 5),
        )
        _assert_seeded_equal(
            4102,
            lambda config=config: resolve_batch_parallel_backend(config),
            lambda config=config: _legacy_resolve_batch_parallel_backend(config),
        )
        _assert_seeded_equal(
            4102,
            lambda config=config: EnvironmentPrior(config)._resolve_batch_parallel_backend(),
            lambda config=config: _legacy_resolve_batch_parallel_backend(config),
        )
        _assert_seeded_equal(
            4103,
            lambda config=config: resolve_batch_shared_environment(config),
            lambda config=config: _legacy_resolve_batch_shared_environment(config),
        )
        _assert_seeded_equal(
            4103,
            lambda config=config: EnvironmentPrior(config)._resolve_batch_shared_environment(),
            lambda config=config: _legacy_resolve_batch_shared_environment(config),
        )
        _assert_seeded_equal(
            4104,
            lambda config=config: resolve_batch_vectorized_strict_rng_match(config),
            lambda config=config: _legacy_resolve_batch_vectorized_strict_rng_match(config),
        )
        _assert_seeded_equal(
            4104,
            lambda config=config: EnvironmentPrior(config)._resolve_batch_vectorized_strict_rng_match(),
            lambda config=config: _legacy_resolve_batch_vectorized_strict_rng_match(config),
        )
        _assert_seeded_equal(
            4105,
            lambda config=config: resolve_batch_vectorized_grouping(config),
            lambda config=config: _legacy_resolve_batch_vectorized_grouping(config),
        )
        _assert_seeded_equal(
            4105,
            lambda config=config: EnvironmentPrior(config)._resolve_batch_vectorized_grouping(),
            lambda config=config: _legacy_resolve_batch_vectorized_grouping(config),
        )


def test_rlpfn_maintained_policy_gradient_loss_request_helper_matches_legacy_inline_reference():
    _, env_cfg = _small_exact_scm_env_cfg()
    env_cfg["policy_gradient_normalize_rewards"] = True
    env_cfg["pg_markov_adjacent_replay_enabled"] = True
    env_cfg["pg_markov_adjacent_replay_sample_prob"] = 0.3
    prior = EnvironmentPrior(env_cfg)

    cases = [
        {"n_samples": 12, "tbptt_window": None, "policy_objective_kind": "policy_gradient", "normalize": None},
        {"n_samples": 12, "tbptt_window": 4, "policy_objective_kind": "reinforce", "normalize": False},
        {"n_samples": 12, "tbptt_window": 99, "policy_objective_kind": "alpha_grad", "normalize": True},
    ]
    for case in cases:
        expected = _legacy_prepare_policy_gradient_loss_request(prior, **case)
        actual = prepare_policy_gradient_loss_request(prior, **case)
        wrapped = prior._prepare_policy_gradient_loss_request(**case)
        assert actual == expected
        assert wrapped == expected


def test_rlpfn_maintained_policy_gradient_loss_slice_and_diagnostics_helpers_match_legacy_inline_reference():
    roots = (
        torch.arange(12, dtype=torch.float32).reshape(3, 4),
        torch.arange(12, 24, dtype=torch.float32).reshape(3, 4),
        torch.arange(24, 36, dtype=torch.float32).reshape(3, 4),
    )
    mask = torch.tensor(
        [
            [[True, False, True, False], [False, True, False, True]],
            [[True, True, False, False], [False, False, True, True]],
            [[True, True, True, True], [False, False, False, False]],
        ],
        dtype=torch.bool,
    )
    group_entries = (
        {"indices": (0, 1), "action_mean_roots": roots, "action_mask": mask},
        {"indices": (2,), "action_mean_roots": tuple(root[:1] for root in roots), "action_mask": mask[:1, :1]},
        "skip-me",
    )
    expected_slice = _legacy_slice_group_traces_after_eval(group_entries, 1)
    actual_slice = slice_group_traces_after_eval(group_entries, 1)
    wrapped_slice = EnvironmentPrior._slice_group_traces_after_eval(group_entries, 1)
    assert len(actual_slice) == len(expected_slice)
    assert len(wrapped_slice) == len(expected_slice)
    for actual_entry, wrapped_entry, expected_entry in zip(actual_slice, wrapped_slice, expected_slice):
        assert actual_entry["indices"] == expected_entry["indices"]
        assert wrapped_entry["indices"] == expected_entry["indices"]
        assert torch.equal(actual_entry["action_mask"], expected_entry["action_mask"])
        assert torch.equal(wrapped_entry["action_mask"], expected_entry["action_mask"])
        for actual_root, wrapped_root, expected_root in zip(
            actual_entry["action_mean_roots"],
            wrapped_entry["action_mean_roots"],
            expected_entry["action_mean_roots"],
        ):
            assert torch.equal(actual_root, expected_root)
            assert torch.equal(wrapped_root, expected_root)

    rollout = {
        "rewards": torch.tensor([[1.0, -2.0], [3.0, 4.0]], dtype=torch.float32),
        "rollout_profile": {
            "policy_cuda_ms": 1.5,
            "transition_cuda_ms": 2.5,
            "transition_group_count": 7,
            "transition_family_group_count": 3,
            "transition_bucket_mean_batch": 2.25,
            "transition_async_enabled": 1,
            "noise_mode": "uniform",
            "noise_block_size": 8,
            "env_count": 4,
            "strict_joint_transition_count": 4,
            "strict_joint_transition_share": 1.0,
            "reference_semantics_count": 4,
            "reference_semantics_share": 1.0,
            "exact_scm_count": 4,
            "transition_reference_mode": "scm_exact",
        },
        "terminal_stats": {
            "terminal_count_mean": torch.tensor(2.5),
            "terminal_count_min": 1.0,
            "terminal_count_max": 5.0,
            "terminal_count_target_mean": 3.0,
            "terminal_count_target_min": 2.0,
            "terminal_count_target_max": 6.0,
        },
    }
    base_stats = {"objective": torch.tensor(1.25)}
    expected_stats = _legacy_attach_common_rollout_diagnostics(rollout, base_stats)
    actual_stats = attach_common_rollout_diagnostics(rollout, deepcopy(base_stats))
    wrapped_stats = EnvironmentPrior._attach_common_rollout_diagnostics(rollout, deepcopy(base_stats))

    def _assert_stats_equal(actual, expected):
        assert set(actual.keys()) == set(expected.keys())
        for key in expected:
            actual_value = actual[key]
            expected_value = expected[key]
            if torch.is_tensor(expected_value):
                assert torch.equal(actual_value, expected_value), key
            else:
                assert actual_value == expected_value, key

    _assert_stats_equal(actual_stats, expected_stats)
    _assert_stats_equal(wrapped_stats, expected_stats)


def test_rlpfn_maintained_path_exact_scm_get_batch_trace_matches_golden():
    expected = {
        "x_shape": (12, 2, 434),
        "y_shape": (12, 2),
        "x00_head": [0.007877, -0.016809, -0.007063, -0.010912, -0.019034, 0.024015, 0.0, 0.0, 0.0, 0.0],
        "x51_head": [1.068617, -1.159532, 0.061322, 1.817357, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "x00_meta": [0.0, 1.0, 0.0, 0.0, -0.044135, 0.191324, 0.028874, 0.0],
        "x51_meta": [0.895848, 1.0, 1.0, 0.0, 0.131627, -0.186655, 0.430441, 0.0],
        "y_head": [0.012206, 0.007162, 0.014864, 0.015489, 0.011253, 0.015846],
        "y_tail": [-0.367191, -1.10667, 1.206445],
        "x_sum": 52.351818,
        "y_sum": 4.157647,
        "info0": {
            "family": "scm",
            "transition_reference_mode": "scm_exact",
            "strict_joint_transition_enabled": True,
            "reference_semantics_enabled": True,
            "reward_drop_frac_realized": 0.0,
        },
        "info1": {
            "family": "scm",
            "transition_reference_mode": "scm_exact",
            "strict_joint_transition_enabled": True,
            "reference_semantics_enabled": True,
            "reward_drop_frac_realized": 0.0,
        },
    }

    assert _capture_exact_scm_get_batch_trace() == expected


def test_rlpfn_maintained_path_policy_step_trace_matches_golden():
    expected = {
        "a1_head": [-0.056622, -0.051079, -0.140691, -0.325196, 0.247738, -0.091861, -0.094656, 0.066978],
        "a1_row1_head": [0.016508, -0.106914, -0.203769, 0.047674, 0.240656, 0.216213, -0.165484, 0.05801],
        "a2_head": [0.057632, -0.017828, -0.11649, -0.306731, 0.240711, 0.103525, -0.366433, 0.115897],
        "a2_row1_head": [0.233536, -0.160705, -0.223101, -0.164249, 0.138238, 0.335276, -0.364536, -0.056289],
        "a1_sum": 1.541568,
        "a2_sum": 2.278063,
        "cache1_type": "list",
        "cache2_type": "list",
    }

    assert _capture_policy_step_trace() == expected


def test_rlpfn_maintained_alpha_grad_fast_runner_gradient_trace_matches_golden():
    expected = {
        "loss": -0.226233,
        "reward_sum": 1.433646,
        "objective": 0.226233,
        "alpha_grad_v0_mean": 0.258765,
        "alpha_grad_v1_mean": 0.167517,
        "alpha_grad_mix_abs_max": 0.244621,
        "alpha_grad_valid_share": 1.0,
        "rollout_transition_group_count": 0,
        "rollout_transition_family_group_count": 0,
        "rollout_exact_scm_count": 4,
        "rollout_transition_reference_mode": "scm_exact",
        "reinforce_log_prob_mean": 6.066904,
        "policy_head_grad_sum": 22.163615,
        "policy_head_grad_max": 0.27963,
        "attn_in_proj_grad_sum": 2.410962,
        "attn_in_proj_grad_max": 0.011078,
    }

    assert _capture_alpha_grad_fast_runner_gradient_trace() == expected


def test_rlpfn_maintained_alpha_grad_train_step_trace_matches_golden():
    expected = {
        "loss": -0.226233,
        "objective": 0.226233,
        "reward_sum": 1.433646,
        "alpha_grad_v0_mean": 0.258765,
        "alpha_grad_v1_mean": 0.167517,
        "reinforce_log_prob_mean": 6.066904,
        "rollout_transition_reference_mode": "scm_exact",
        "rollout_exact_scm_count": 4,
        "grad_abs_sum": 40.616607,
        "grad_abs_max": 0.27963,
        "grad_numel": 26622,
        "param_delta_abs_sum": 11.283599,
        "param_delta_abs_max": 0.001,
        "policy_head_delta_abs_sum": 2.306956,
        "attn_in_proj_delta_abs_sum": 3.025408,
    }

    assert _capture_alpha_grad_train_step_trace() == expected
