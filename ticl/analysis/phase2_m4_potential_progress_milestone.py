"""M4 potential/progress live prior milestone.

This module is intentionally M4-only.  It keeps the high-performance
family-vectorized runner path and removes the retired reward-group balancing
runtime from the maintained fit_model surface.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import math
from types import MethodType
from typing import Any

import torch


M4_POTENTIAL_PROGRESS_MILESTONE = "m4_potential_progress_live"
SAMPLED_TOPOLOGY_MILESTONE = "sampled_topology"


@dataclass(frozen=True)
class M4TopologyThresholds:
    state_gain_min: float = 0.7
    action_gain_min: float = 0.06
    noise_gain_min: float = 0.0
    state_to_action_ratio_max: float = 10.0
    state_to_noise_ratio_max: float = 0.0

    def validate(self) -> None:
        for name, value in (
            ("state_gain_min", self.state_gain_min),
            ("action_gain_min", self.action_gain_min),
            ("noise_gain_min", self.noise_gain_min),
            ("state_to_action_ratio_max", self.state_to_action_ratio_max),
            ("state_to_noise_ratio_max", self.state_to_noise_ratio_max),
        ):
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")


@dataclass(frozen=True)
class _M4TopologyDecision:
    original_topology_pass: bool
    balanced_topology_pass: bool
    gated_topology_pass: bool
    repair_kind: str
    selected_balance_applied: bool


def normalize_prior_milestone(value: Any) -> str:
    text = str(value or M4_POTENTIAL_PROGRESS_MILESTONE).strip().lower()
    aliases = {
        "m4": M4_POTENTIAL_PROGRESS_MILESTONE,
        "milestone4": M4_POTENTIAL_PROGRESS_MILESTONE,
        "m4_potential_progress": M4_POTENTIAL_PROGRESS_MILESTONE,
        "potential_progress": M4_POTENTIAL_PROGRESS_MILESTONE,
        "potential_progress_live": M4_POTENTIAL_PROGRESS_MILESTONE,
        "sampled": SAMPLED_TOPOLOGY_MILESTONE,
        "sampled_topology_conditioned": SAMPLED_TOPOLOGY_MILESTONE,
    }
    text = aliases.get(text, text)
    if text not in {M4_POTENTIAL_PROGRESS_MILESTONE, SAMPLED_TOPOLOGY_MILESTONE}:
        raise ValueError(
            "Unsupported maintained PPO pack prior milestone "
            f"{value!r}; expected {M4_POTENTIAL_PROGRESS_MILESTONE!r}."
        )
    return text


def m4_potential_progress_contract(
    *,
    state_gain_min: float = 0.7,
    action_gain_min: float = 0.06,
    noise_gain_min: float = 0.0,
    state_to_action_ratio_max: float = 10.0,
    state_to_noise_ratio_max: float = 0.0,
) -> dict[str, Any]:
    thresholds = M4TopologyThresholds(
        state_gain_min=float(state_gain_min),
        action_gain_min=float(action_gain_min),
        noise_gain_min=float(noise_gain_min),
        state_to_action_ratio_max=float(state_to_action_ratio_max),
        state_to_noise_ratio_max=float(state_to_noise_ratio_max),
    )
    thresholds.validate()
    return {
        "milestone": M4_POTENTIAL_PROGRESS_MILESTONE,
        "formal": True,
        "accepted_distribution_property": "m4_potential_progress_and_survival_source_labels",
        "modified_path": (
            "exact_scm_gym_lowtail_potential_progress_source_label + "
            "exact_scm_survival_source_label"
        ),
        "runner_path": "family_vectorized_live",
        "retired_paths": (
            "reward_group_balance_runtime",
            "gated_reward_path_balance",
            "gated_reward_path_balance_terminal_coverage",
            "fixed_frozen_h_list_training",
        ),
        "non_degradation_guards": (
            "state_action_terminal_source_ranges_preserved",
            "zero_pad_inactive_for_family_vectorized_path",
            "reward_density_formula_precheck_before_env_build",
            "survival_component_additive_base_preservation_audited",
            "no_runtime_reward_group_balance_patch",
        ),
        "topology_thresholds": {
            "state_gain_min": float(thresholds.state_gain_min),
            "action_gain_min": float(thresholds.action_gain_min),
            "noise_gain_min": float(thresholds.noise_gain_min),
            "state_to_action_ratio_max": float(thresholds.state_to_action_ratio_max),
            "state_to_noise_ratio_max": float(thresholds.state_to_noise_ratio_max),
        },
    }


def _tensor_list(env: dict[str, Any], key: str, n: int) -> list[float]:
    value = env.get(key)
    if not torch.is_tensor(value):
        raise RuntimeError(f"M4 potential/progress milestone requires env tensor `{key}`")
    out = value.detach().cpu().reshape(-1).tolist()
    if len(out) != int(n):
        raise RuntimeError(f"{key} length mismatch: got {len(out)} expected {int(n)}")
    return [float(v) for v in out]


def _h_float(h: dict[str, Any], key: str, default: float) -> float:
    value = h.get(key, default)
    if torch.is_tensor(value):
        value = value.detach().cpu().reshape(-1)[0].item()
    try:
        out = float(value)
    except Exception:
        out = float(default)
    if not math.isfinite(out):
        out = float(default)
    return float(out)


def _m4_lowtail_gain_summary_from_h_list(self, h_list: list[dict[str, Any]], device) -> dict[str, torch.Tensor]:
    """Formula-level M4 reward topology summary without building SCM tensors."""

    if not h_list:
        raise RuntimeError("M4 source-label topology precheck requires a non-empty h-list")
    if not all(bool(h.get("exact_scm_gym_lowtail_family_selected", False)) for h in h_list):
        raise RuntimeError("M4 source-label sampler produced a non-lowtail h; refusing slow env fallback")
    dims = [self._sample_dims(h) for h in h_list]
    env = {
        "state_dim_per_sample": torch.as_tensor([d[0] for d in dims], device=device, dtype=torch.long),
        "obs_dim_per_sample": torch.as_tensor([d[1] for d in dims], device=device, dtype=torch.long),
        "action_dim_per_sample": torch.as_tensor([d[2] for d in dims], device=device, dtype=torch.long),
        "noise_dim_per_sample": torch.as_tensor([d[3] for d in dims], device=device, dtype=torch.long),
        "exact_scm_gym_lowtail_family_active": torch.ones((len(h_list),), device=device, dtype=torch.bool),
        "exact_scm_gym_lowtail_reward_linear_weight": torch.as_tensor(
            [_h_float(h, "exact_scm_gym_lowtail_reward_linear_weight", 1.25) for h in h_list],
            device=device,
            dtype=torch.float32,
        ),
        "exact_scm_gym_lowtail_reward_potential_delta_weight": torch.as_tensor(
            [_h_float(h, "exact_scm_gym_lowtail_reward_potential_delta_weight", 0.0) for h in h_list],
            device=device,
            dtype=torch.float32,
        ),
        "exact_scm_gym_lowtail_reward_state_cost": torch.as_tensor(
            [_h_float(h, "exact_scm_gym_lowtail_reward_state_cost", 0.05) for h in h_list],
            device=device,
            dtype=torch.float32,
        ),
        "exact_scm_gym_lowtail_reward_ctrl_cost": torch.as_tensor(
            [_h_float(h, "exact_scm_gym_lowtail_reward_ctrl_cost", 0.02) for h in h_list],
            device=device,
            dtype=torch.float32,
        ),
        "exact_scm_gym_lowtail_action_gain": torch.as_tensor(
            [_h_float(h, "exact_scm_gym_lowtail_action_gain", 0.60) for h in h_list],
            device=device,
            dtype=torch.float32,
        ),
        "exact_scm_gym_lowtail_noise_gain": torch.as_tensor(
            [_h_float(h, "exact_scm_gym_lowtail_noise_gain", 0.35) for h in h_list],
            device=device,
            dtype=torch.float32,
        ),
    }
    return self._exact_scm_gym_lowtail_reward_input_gain_summary_batch(env)


def _m4_density_topology_decision(
    row: dict[str, Any],
    *,
    thresholds: M4TopologyThresholds,
) -> _M4TopologyDecision:
    state_density = float(row["reward_state_input_gain_density_ratio"])
    action_density = float(row["reward_action_input_gain_density_ratio"])
    density_ratio = float(row["reward_state_to_action_gain_density_ratio"])
    noise_density = float(row.get("reward_noise_input_gain_density_ratio", 0.0))
    noise_ratio = float(row.get("reward_state_to_noise_gain_density_ratio", 0.0))
    finite_required = (
        math.isfinite(state_density)
        and math.isfinite(action_density)
        and math.isfinite(density_ratio)
        and math.isfinite(noise_density)
        and math.isfinite(noise_ratio)
    )
    density_pass = (
        bool(finite_required)
        and state_density > 0.0
        and action_density > 0.0
        and (
            float(thresholds.state_to_action_ratio_max) <= 0.0
            or density_ratio <= float(thresholds.state_to_action_ratio_max)
        )
        and (
            float(thresholds.state_to_noise_ratio_max) <= 0.0
            or noise_ratio <= float(thresholds.state_to_noise_ratio_max)
        )
    )
    return _M4TopologyDecision(
        original_topology_pass=bool(density_pass),
        balanced_topology_pass=False,
        gated_topology_pass=bool(density_pass),
        repair_kind="m4_density_original_keep" if bool(density_pass) else "m4_density_reject",
        selected_balance_applied=False,
    )


def _bounded_float(value: Any, default: float, *, min_value: float, max_value: float) -> float:
    try:
        out = float(value)
    except Exception:
        out = float(default)
    if not math.isfinite(out):
        out = float(default)
    lo = float(min_value)
    hi = float(max_value)
    if hi < lo:
        hi = lo
    return float(min(max(out, lo), hi))


def _apply_m4_survival_source_label_policy(
    prior,
    h_list: list[dict[str, Any]],
    *,
    batch_indices: list[int] | None = None,
) -> list[dict[str, Any]]:
    if not bool(prior._coerce_bool(prior.config.get("exact_scm_survival_source_label_enabled", False))):
        return h_list
    prob = _bounded_float(
        prior.config.get("exact_scm_survival_source_label_prob", 0.5),
        0.5,
        min_value=0.0,
        max_value=1.0,
    )
    weight_min = _bounded_float(
        prior.config.get("exact_scm_survival_source_label_weight_min", 0.005),
        0.005,
        min_value=0.0,
        max_value=1.0,
    )
    weight_max = _bounded_float(
        prior.config.get("exact_scm_survival_source_label_weight_max", 0.03),
        0.03,
        min_value=weight_min,
        max_value=1.0,
    )
    if weight_min <= 0.0 or weight_max <= 0.0:
        weight_min = 0.0
        weight_max = 0.0
    log_min = math.log(max(weight_min, 1e-12))
    log_max = math.log(max(weight_max, max(weight_min, 1e-12)))
    valid_targets = {"survival_off", "survival_on"}
    for local_idx, h in enumerate(h_list):
        target = str(h.get("survival_component_axis_target", "") or "").strip()
        if target not in valid_targets:
            if batch_indices is not None and local_idx < len(batch_indices):
                target = "survival_on" if int(batch_indices[local_idx]) % 2 == 0 else "survival_off"
            else:
                u = float(prior._latent_uniform_from_h(h, "_m4_survival_source_u"))
                target = "survival_on" if u < prob else "survival_off"
        if target == "survival_on":
            u_weight = float(prior._latent_uniform_from_h(h, "_m4_survival_weight_u"))
            weight = math.exp(log_min + u_weight * (log_max - log_min)) if weight_max > 0.0 else 0.0
            h["survival_reward_weight"] = float(weight)
            h["survival_reward_enable_prob"] = 1.0
            h["_survival_reward_enable_u"] = 0.0
        else:
            h["survival_reward_weight"] = 0.0
            h["survival_reward_enable_prob"] = 0.0
            h["_survival_reward_enable_u"] = 1.0
        h["survival_component_axis_target"] = str(target)
        h["survival_component_axis_rule"] = "m4_small_additive_source_label_v1"
        h["survival_component_preserves_base_reward"] = True
        h["survival_component_audit_schema"] = "phase2_survival_component_reconstruction_audit.v1"
    return h_list


def _attach_m4_metadata(
    env: dict[str, Any],
    *,
    selected_rows: list[dict[str, Any]],
    selected_repair_kinds: list[str],
    device: torch.device,
) -> dict[str, Any]:
    n = int(len(selected_rows))
    env["m4_potential_progress_live_enabled"] = torch.ones((n,), device=device, dtype=torch.bool)
    env["m4_potential_progress_live_repair_kind"] = tuple(str(v) for v in selected_repair_kinds)
    env["m4_potential_progress_live_balance_applied"] = torch.zeros((n,), device=device, dtype=torch.bool)
    for key in (
        "original_topology_pass",
        "balanced_topology_pass",
        "gated_topology_pass",
    ):
        env[f"m4_potential_progress_live_{key}"] = torch.as_tensor(
            [bool(row[key]) for row in selected_rows],
            device=device,
            dtype=torch.bool,
        )
    return env


def install_m4_potential_progress_live_milestone(
    prior,
    *,
    state_gain_min: float = 0.7,
    action_gain_min: float = 0.06,
    noise_gain_min: float = 0.0,
    state_to_action_ratio_max: float = 10.0,
    state_to_noise_ratio_max: float = 0.0,
    max_attempts: int = 4096,
) -> None:
    """Install the M4 live source-label sampler on one EnvironmentPrior instance."""

    thresholds = M4TopologyThresholds(
        state_gain_min=float(state_gain_min),
        action_gain_min=float(action_gain_min),
        noise_gain_min=float(noise_gain_min),
        state_to_action_ratio_max=float(state_to_action_ratio_max),
        state_to_noise_ratio_max=float(state_to_noise_ratio_max),
    )
    thresholds.validate()
    prior.config["reward_topology_conditioned_sampling_enabled"] = True
    prior.config["reward_state_input_gain_fraction_conditioned_sampling_enabled"] = False
    prior.config["reward_state_input_gain_fraction_conditioned_min"] = float(state_gain_min)
    prior.config["reward_action_input_gain_fraction_conditioned_min"] = float(action_gain_min)
    prior.config["reward_noise_input_gain_fraction_conditioned_min"] = float(noise_gain_min)
    prior.config["reward_state_to_action_gain_ratio_conditioned_max"] = float(state_to_action_ratio_max)
    prior.config["reward_state_to_noise_gain_ratio_conditioned_max"] = float(state_to_noise_ratio_max)
    prior.config["reward_topology_conditioned_sampling_max_attempts"] = int(max_attempts)
    prior.config["reward_path_milestone"] = M4_POTENTIAL_PROGRESS_MILESTONE
    prior.config["terminal_count_coverage_enabled"] = False
    prior.config["constrained_dim_sampling_policy"] = "state_first"
    prior.config["constrained_dim_noise_policy"] = "gym_low"
    prior.config["reference_scm_zero_pad_inactive_init_enabled"] = True
    prior.config["exact_scm_gym_lowtail_family_enabled"] = True
    prior.config["exact_scm_gym_lowtail_family_prob"] = 1.0
    prior.config["exact_scm_gym_lowtail_potential_progress_source_label_enabled"] = True
    prior.config["exact_scm_gym_lowtail_potential_progress_source_label_prob"] = 0.5
    prior.config["exact_scm_gym_lowtail_potential_progress_reward_mix"] = 0.35
    prior.config["exact_scm_gym_lowtail_potential_progress_linear_weight"] = 1.25
    prior.config["exact_scm_gym_lowtail_potential_progress_progress_delta_weight"] = 0.0
    prior.config["exact_scm_gym_lowtail_potential_progress_delta_progress_delta_weight"] = 0.65
    prior.config["exact_scm_survival_source_label_enabled"] = True
    prior.config["exact_scm_survival_source_label_prob"] = 0.5
    prior.config["exact_scm_survival_source_label_weight_min"] = 0.005
    prior.config["exact_scm_survival_source_label_weight_max"] = 0.03

    def _sample_m4_conditioned(
        self,
        batch_size,
        device,
        rng_seeds=None,
        *,
        build_policy_generator=True,
        preserve_skipped_generator_rng=True,
        return_final_env=True,
    ):
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        base_seeds = (
            [int(s) for s in rng_seeds]
            if rng_seeds is not None
            else self._sample_seed_list(batch_size)
        )
        if len(base_seeds) != batch_size:
            raise ValueError("rng_seeds must match batch_size")

        accepted_h_list: list[dict[str, Any] | None] = [None] * batch_size
        accepted_seeds: list[int | None] = [None] * batch_size
        accepted_attempts = [0] * batch_size
        accepted_state_gains = [float("nan")] * batch_size
        accepted_action_gains = [float("nan")] * batch_size
        accepted_noise_gains = [float("nan")] * batch_size
        accepted_ratios = [float("nan")] * batch_size
        accepted_noise_ratios = [float("nan")] * batch_size
        selected_rows: list[dict[str, Any] | None] = [None] * batch_size
        selected_repair_kinds = ["m4_density_reject"] * batch_size
        best_state_gains = [-float("inf")] * batch_size
        best_action_gains = [-float("inf")] * batch_size
        best_ratios = [float("inf")] * batch_size
        pending = set(range(batch_size))
        total_draws = 0

        for attempt_idx in range(int(max_attempts)):
            if not pending:
                break
            active = [i for i in range(batch_size) if i in pending]
            candidate_h_list = self._sample_batch_hypers(len(active))
            candidate_h_list = _apply_m4_survival_source_label_policy(
                self,
                candidate_h_list,
                batch_indices=active,
            )
            candidate_seeds = [int(base_seeds[i]) + int(attempt_idx) for i in active]
            summary_source = _m4_lowtail_gain_summary_from_h_list(self, candidate_h_list, device)
            total_draws += len(active)
            state_gains = _tensor_list(summary_source, "reward_state_input_gain_fraction", len(active))
            action_gains = _tensor_list(summary_source, "reward_action_input_gain_fraction", len(active))
            noise_gains = _tensor_list(summary_source, "reward_noise_input_gain_fraction", len(active))
            ratios = _tensor_list(summary_source, "reward_state_to_action_gain_ratio", len(active))
            noise_ratios = _tensor_list(summary_source, "reward_state_to_noise_gain_ratio", len(active))
            state_density_ratios = _tensor_list(
                summary_source,
                "reward_state_input_gain_density_ratio",
                len(active),
            )
            action_density_ratios = _tensor_list(
                summary_source,
                "reward_action_input_gain_density_ratio",
                len(active),
            )
            noise_density_ratios = _tensor_list(
                summary_source,
                "reward_noise_input_gain_density_ratio",
                len(active),
            )
            state_to_action_density_ratios = _tensor_list(
                summary_source,
                "reward_state_to_action_gain_density_ratio",
                len(active),
            )
            state_to_noise_density_ratios = _tensor_list(
                summary_source,
                "reward_state_to_noise_gain_density_ratio",
                len(active),
            )
            for local_idx, global_idx in enumerate(active):
                state_gain = float(state_gains[local_idx])
                action_gain = float(action_gains[local_idx])
                ratio = float(ratios[local_idx])
                if math.isfinite(state_gain):
                    best_state_gains[global_idx] = max(best_state_gains[global_idx], state_gain)
                if math.isfinite(action_gain):
                    best_action_gains[global_idx] = max(best_action_gains[global_idx], action_gain)
                if math.isfinite(ratio):
                    best_ratios[global_idx] = min(best_ratios[global_idx], ratio)
                row = {
                    "reward_state_input_gain_fraction": state_gain,
                    "reward_action_input_gain_fraction": action_gain,
                    "reward_noise_input_gain_fraction": float(noise_gains[local_idx]),
                    "reward_state_to_action_gain_ratio": ratio,
                    "reward_state_to_noise_gain_ratio": float(noise_ratios[local_idx]),
                    "reward_state_input_gain_density_ratio": float(state_density_ratios[local_idx]),
                    "reward_action_input_gain_density_ratio": float(action_density_ratios[local_idx]),
                    "reward_noise_input_gain_density_ratio": float(noise_density_ratios[local_idx]),
                    "reward_state_to_action_gain_density_ratio": float(
                        state_to_action_density_ratios[local_idx]
                    ),
                    "reward_state_to_noise_gain_density_ratio": float(
                        state_to_noise_density_ratios[local_idx]
                    ),
                    "m4_topology_guard_metric": "reward_density_ratio",
                }
                decision = _m4_density_topology_decision(row, thresholds=thresholds)
                row.update(
                    {
                        "original_topology_pass": bool(decision.original_topology_pass),
                        "balanced_topology_pass": bool(decision.balanced_topology_pass),
                        "gated_topology_pass": bool(decision.gated_topology_pass),
                        "gated_repair_kind": str(decision.repair_kind),
                        "selected_balance_applied": bool(decision.selected_balance_applied),
                    }
                )
                if not bool(decision.gated_topology_pass):
                    continue
                accepted_h = copy.deepcopy(candidate_h_list[local_idx])
                accepted_h["reward_state_input_gain_fraction_rejection_min"] = 0.0
                accepted_h["m4_topology_guard_metric"] = "reward_density_ratio"
                accepted_h["m4_runtime_reward_group_balance_required"] = False
                accepted_h_list[global_idx] = accepted_h
                accepted_seeds[global_idx] = int(candidate_seeds[local_idx])
                accepted_attempts[global_idx] = int(attempt_idx + 1)
                accepted_state_gains[global_idx] = float(row["reward_state_input_gain_fraction"])
                accepted_action_gains[global_idx] = float(row["reward_action_input_gain_fraction"])
                accepted_noise_gains[global_idx] = float(row["reward_noise_input_gain_fraction"])
                accepted_ratios[global_idx] = float(row["reward_state_to_action_gain_ratio"])
                accepted_noise_ratios[global_idx] = float(row["reward_state_to_noise_gain_ratio"])
                selected_rows[global_idx] = copy.deepcopy(row)
                selected_repair_kinds[global_idx] = str(decision.repair_kind)
                pending.discard(global_idx)

        if pending:
            pending_str = ", ".join(
                (
                    f"{idx}:best_state={float(best_state_gains[idx]):.6f},"
                    f"best_action={float(best_action_gains[idx]):.6f},"
                    f"best_ratio={float(best_ratios[idx]):.6f}"
                )
                for idx in sorted(pending)
            )
            raise RuntimeError(
                "failed to sample M4 potential/progress exact SCM environments "
                f"within {int(max_attempts)} candidate rounds; {pending_str}"
            )

        final_h_list = [copy.deepcopy(h) for h in accepted_h_list if h is not None]
        final_seeds = [int(s) for s in accepted_seeds if s is not None]
        if len(final_h_list) != batch_size or len(final_seeds) != batch_size:
            raise RuntimeError("internal M4 accepted list length mismatch")
        if not bool(return_final_env):
            return final_h_list, None, final_seeds
        final_env = self._sample_environment_family_coarse_batch(
            h_list=final_h_list,
            device=device,
            rng_seeds=final_seeds,
            build_policy_generator=build_policy_generator,
            preserve_skipped_generator_rng=preserve_skipped_generator_rng,
            _disable_rejection=True,
        )
        final_env = self._attach_vectorized_reward_state_gain_conditioned_metadata(
            final_env,
            h_list=final_h_list,
            accepted_seeds=final_seeds,
            accepted_attempts=accepted_attempts,
            accepted_state_gains=accepted_state_gains,
            accepted_action_gains=accepted_action_gains,
            accepted_noise_gains=accepted_noise_gains,
            accepted_state_to_action_ratios=accepted_ratios,
            accepted_state_to_noise_ratios=accepted_noise_ratios,
            total_candidate_draws=total_draws,
            device=device,
        )
        final_env = _attach_m4_metadata(
            final_env,
            selected_rows=[row if row is not None else {} for row in selected_rows],
            selected_repair_kinds=selected_repair_kinds,
            device=torch.device(device),
        )
        return final_h_list, final_env, final_seeds

    prior._sample_batch_hypers_and_environment_family_coarse_batch_conditioned = MethodType(
        _sample_m4_conditioned,
        prior,
    )
