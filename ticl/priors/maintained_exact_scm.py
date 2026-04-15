import math
import numpy as np
import torch

from ticl.distributions import sample_distributions


def coerce_bool(value):
    if torch.is_tensor(value):
        if value.dtype == torch.bool:
            return value
        if value.numel() == 1:
            return bool(value.item())
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off", ""}
    return bool(value)


def resolve_scalar(value):
    return float(sample_distributions({"value": value})["value"])


def resolve_strict_joint_transition_enabled(h):
    enabled = h.get("strict_joint_transition_enabled", False)
    return bool(coerce_bool(enabled))


def resolve_reference_semantics_enabled(h):
    return resolve_strict_joint_transition_enabled(h)


def env_uses_reference_semantics(env):
    enabled = env.get(
        "reference_semantics_enabled",
        env.get("strict_joint_transition_enabled", False),
    )
    return bool(coerce_bool(enabled))


def transition_reference_mode(family, reference_semantics_enabled, gp_forward_mode=None):
    family_str = str(family)
    if family_str == "gp" and bool(reference_semantics_enabled):
        mode = str(gp_forward_mode or "exact").strip().lower()
        if mode == "fixed_cost":
            return "gp_fixed_cost"
        return "gp_exact"
    return f"{family_str}_{'exact' if bool(reference_semantics_enabled) else 'legacy'}"


def env_obs_input_dim(obs_dim, *, reference_semantics_enabled=False):
    return 0 if bool(reference_semantics_enabled) else int(max(0, int(obs_dim)))


def env_input_layout(
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
    obs_input_dim = env_obs_input_dim(
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


def resolve_state_input_scale_enabled(h):
    return bool(coerce_bool(h.get("state_input_scale_enabled", False)))


def resolve_state_input_scale(h):
    v = resolve_scalar(h.get("state_input_scale", 1.0))
    if not math.isfinite(v):
        return 1.0
    return float(max(1e-6, v))


def resolve_state_full_rms_enabled(h):
    return bool(coerce_bool(h.get("state_full_rms_enabled", False)))


def resolve_state_full_rms_target(h):
    v = resolve_scalar(h.get("state_full_rms_target", 1.0))
    if (not math.isfinite(v)) or v <= 0.0:
        return 1.0
    return float(v)


def resolve_reinforce_reward_transform(h):
    mode = str(h.get("reinforce_reward_transform", "none")).strip().lower()
    if mode not in {"none", "tanh", "rms", "clip"}:
        mode = "none"
    return mode


def resolve_reinforce_reward_rms_eps(h):
    v = resolve_scalar(h.get("reinforce_reward_rms_eps", 1e-6))
    if (not math.isfinite(v)) or v <= 0.0:
        return 1e-6
    return float(v)


def resolve_reinforce_reward_tanh_c(h):
    v = resolve_scalar(h.get("reinforce_reward_tanh_c", 1.0))
    if (not math.isfinite(v)) or v <= 0.0:
        return 1.0
    return float(v)


def resolve_reinforce_reward_tanh_bound(h):
    v = resolve_scalar(h.get("reinforce_reward_tanh_bound", 2.0))
    if (not math.isfinite(v)) or v <= 0.0:
        return 2.0
    return float(v)


def resolve_ctrl_reward_weight(h):
    v = resolve_scalar(h.get("ctrl_reward_weight", 0.0))
    if (not math.isfinite(v)) or v < 0.0:
        return 0.0
    return float(v)


def resolve_ctrl_reward_enable_prob(h):
    v = resolve_scalar(h.get("ctrl_reward_enable_prob", 0.0))
    if not math.isfinite(v):
        return 0.0
    return float(max(0.0, min(1.0, v)))


def resolve_survival_reward_weight(h):
    v = resolve_scalar(h.get("survival_reward_weight", 0.0))
    if (not math.isfinite(v)) or v < 0.0:
        return 0.0
    return float(v)


def resolve_survival_reward_enable_prob(h):
    v = resolve_scalar(h.get("survival_reward_enable_prob", 0.0))
    if not math.isfinite(v):
        return 0.0
    return float(max(0.0, min(1.0, v)))


def resolve_reinforce_action_transform(h):
    mode = str(h.get("reinforce_action_transform", "none")).strip().lower()
    if mode not in {"tanh", "rms", "none", "clip"}:
        mode = "none"
    return mode


def resolve_reinforce_action_rms_eps(h):
    v = resolve_scalar(h.get("reinforce_action_rms_eps", 1e-6))
    if (not math.isfinite(v)) or v <= 0.0:
        return 1e-6
    return float(v)


def resolve_reinforce_action_clip_bound(h):
    v = resolve_scalar(h.get("reinforce_action_clip_bound", 5.0))
    if (not math.isfinite(v)) or v <= 0.0:
        return 5.0
    return float(v)


def resolve_terminal_reset_enabled(h):
    return bool(coerce_bool(h.get("terminal_reset_enabled", False)))


def resolve_terminal_reset_count_target(h):
    v = resolve_scalar(h.get("terminal_reset_count_target", 0))
    if not math.isfinite(v):
        return 0.0
    return float(max(0.0, float(v)))


def resolve_terminal_bonus_tanh_c(h):
    v = resolve_scalar(h.get("terminal_bonus_tanh_c", 10.0))
    if (not math.isfinite(v)) or v <= 0.0:
        return 10.0
    return float(v)


def resolve_terminal_bonus_scale_min(h):
    v = resolve_scalar(h.get("terminal_bonus_scale_min", 1.0))
    if not math.isfinite(v):
        return 1.0
    return float(v)


def resolve_terminal_bonus_scale_max(h):
    v = resolve_scalar(h.get("terminal_bonus_scale_max", 2.0))
    if not math.isfinite(v):
        return 2.0
    return float(v)


def expand_env_value_to_list(value, batch_size):
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


def new_env_semantics_accumulator():
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


def finalize_env_semantics_summary(acc):
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


def merge_env_semantics_summary(acc, summary):
    if not isinstance(summary, dict):
        return acc
    if acc is None:
        acc = new_env_semantics_accumulator()
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


def summarize_env_semantics(env, batch_size):
    batch_size = int(max(1, int(batch_size)))
    families = expand_env_value_to_list(env.get("family", "unknown"), batch_size)
    gp_modes = expand_env_value_to_list(env.get("reference_gp_forward_mode", None), batch_size)
    references = [
        bool(v)
        for v in expand_env_value_to_list(
            env.get("reference_semantics_enabled", env.get("strict_joint_transition_enabled", False)),
            batch_size,
        )
    ]
    stricts = [
        bool(v)
        for v in expand_env_value_to_list(env.get("strict_joint_transition_enabled", False), batch_size)
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
    total = int(batch_size)
    summary = {
        "env_count": total,
        "strict_joint_transition_count": int(sum(1 for flag in stricts if flag)),
        "reference_semantics_count": int(sum(1 for flag in references if flag)),
        "exact_scm_count": int(exact_scm_count),
        "exact_gp_count": int(exact_gp_count),
        "fixed_gp_count": int(fixed_gp_count),
        "legacy_scm_count": int(legacy_scm_count),
        "legacy_gp_count": int(legacy_gp_count),
    }
    return finalize_env_semantics_summary(summary)
