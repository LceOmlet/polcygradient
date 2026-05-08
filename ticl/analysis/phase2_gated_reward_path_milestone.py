"""Explicit gated reward-path prior milestone integration.

This module installs the already-paritied reward-path balancing contract onto
an EnvironmentPrior instance. It is intentionally opt-in: callers must request
the milestone explicitly, and the exact-SCM/default prior code is not modified.

The formal milestone is deliberately narrow:

* already topology-healthy samples keep their original h unchanged
* width-bias reward-path failures may receive reward-path-only balancing
* balanced samples must pass the same topology guard, otherwise they are
  rejected

This milestone does not narrow dimension samplers and does not modify the
state-transition or terminal/survival paths. The terminal-coverage variant in
this file is kept as an explicitly named exploratory extension, not as part of
the main gated_reward_path_balance milestone contract.
"""

from __future__ import annotations

import copy
import math
import sys
from pathlib import Path
from types import MethodType
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.exploratory.phase2_gated_reward_path_generator_wrapper import (  # noqa: E402
    RewardPathBalanceProfile,
    TopologyThresholds,
    apply_decision_to_row,
    decide_gated_reward_path_repair,
    row_with_balanced_topology,
    select_h_for_decision,
)
from scripts.exploratory.phase2_reward_group_balance_runtime import (  # noqa: E402
    install_environment_prior_reward_group_balance_patch,
)


GATED_REWARD_PATH_MILESTONE = "gated_reward_path_balance"
GATED_REWARD_PATH_TERMINAL_COVERAGE_MILESTONE = "gated_reward_path_balance_terminal_coverage"
SAMPLED_TOPOLOGY_MILESTONE = "sampled_topology"

GATED_REWARD_PATH_MILESTONES = {
    GATED_REWARD_PATH_MILESTONE,
    GATED_REWARD_PATH_TERMINAL_COVERAGE_MILESTONE,
}


def gated_reward_path_milestone_contract(
    *,
    profile: RewardPathBalanceProfile | None = None,
    state_gain_min: float = 0.7,
    action_gain_min: float = 0.06,
    state_to_action_ratio_max: float = 10.0,
) -> dict[str, Any]:
    """Return the non-regression contract for the formal gated milestone."""

    profile = profile or RewardPathBalanceProfile()
    thresholds = TopologyThresholds(
        state_gain_min=float(state_gain_min),
        action_gain_min=float(action_gain_min),
        state_to_action_ratio_max=float(state_to_action_ratio_max),
    )
    profile.validate()
    thresholds.validate()
    return {
        "milestone": GATED_REWARD_PATH_MILESTONE,
        "formal": True,
        "accepted_distribution_property": "gated_reward_path_group_balance",
        "sufficiency_claim": (
            "targets the observed width-biased reward topology failure while "
            "preserving the prior's dimension, state-transition, and terminal "
            "samplers"
        ),
        "decision_order": (
            "original_topology_pass_keep_original_h",
            "width_bias_fail_balance_reward_path_only",
            "balanced_fail_reject",
        ),
        "non_degradation_guards": (
            "already_healthy_samples_are_not_balanced",
            "balanced_samples_must_pass_same_topology_thresholds",
            "state_transition_path_unchanged",
            "terminal_survival_path_unchanged",
            "dimension_samplers_unchanged",
        ),
        "modified_path": "reward_path_only",
        "unchanged_paths": (
            "state_transition",
            "terminal_survival",
            "action_dim_sampler",
            "obs_dim_sampler",
            "state_dim_sampler",
        ),
        "terminal_coverage_in_formal_milestone": False,
        "terminal_coverage_variant": GATED_REWARD_PATH_TERMINAL_COVERAGE_MILESTONE,
        "balance_profile": {
            "name": str(profile.name),
            "state_scale": float(profile.state_scale),
            "action_scale": float(profile.action_scale),
            "noise_scale": float(profile.noise_scale),
        },
        "topology_thresholds": {
            "state_gain_min": float(thresholds.state_gain_min),
            "action_gain_min": float(thresholds.action_gain_min),
            "state_to_action_ratio_max": float(thresholds.state_to_action_ratio_max),
        },
    }


def _stable_unit_interval(*values: int) -> float:
    """Small deterministic hash for milestone-local quota values."""

    x = 0x9E3779B97F4A7C15
    mask = (1 << 64) - 1
    for value in values:
        x ^= int(value) & mask
        x = (x * 0xBF58476D1CE4E5B9) & mask
        x ^= x >> 30
        x = (x * 0x94D049BB133111EB) & mask
        x ^= x >> 31
    return float((x >> 11) & ((1 << 53) - 1)) / float(1 << 53)


def _terminal_coverage_bucket_for_slot(slot_idx: int, batch_size: int) -> str:
    """Return the terminal target coverage bucket for one batch slot.

    Most slots intentionally keep the original maintained-prior terminal target.
    A small deterministic quota adds Gym-validation high-done support without
    changing the exact-SCM terminal equation or narrowing the full distribution.
    """

    n = int(max(1, batch_size))
    if n < 8:
        return "original"
    rank = (int(slot_idx) * 9973) % n
    high_quota = max(1, int(round(0.125 * n)))
    mid_high_quota = max(1, int(round(0.125 * n)))
    if rank < high_quota:
        return "high_50_80"
    if rank < high_quota + mid_high_quota:
        return "mid_20_50"
    return "original"


def _terminal_coverage_target_for_bucket(
    *,
    original_target: float,
    bucket: str,
    seed: int,
    slot_idx: int,
) -> float:
    if str(bucket) == "mid_20_50":
        unit = _stable_unit_interval(int(seed), int(slot_idx), 0x2050)
        return float(20.0 + (50.0 - 20.0) * unit)
    if str(bucket) == "high_50_80":
        unit = _stable_unit_interval(int(seed), int(slot_idx), 0x5080)
        return float(50.0 + (80.0 - 50.0) * unit)
    return float(max(0.0, original_target))


def _h_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off", ""}
    return bool(value)


def _as_tensor(values: list[Any], *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.as_tensor(values, device=device, dtype=dtype)


def _tensor_list(env: dict[str, Any], key: str, n: int) -> list[float]:
    value = env.get(key)
    if not torch.is_tensor(value):
        raise RuntimeError(f"gated reward-path milestone requires env tensor `{key}`")
    out = value.detach().cpu().reshape(-1).tolist()
    if len(out) != int(n):
        raise RuntimeError(f"{key} length mismatch: got {len(out)} expected {int(n)}")
    return [float(v) for v in out]


def _attach_gated_metadata(
    env: dict[str, Any],
    *,
    selected_rows: list[dict[str, Any]],
    selected_repair_kinds: list[str],
    selected_balance_applied: list[bool],
    selected_terminal_coverage_buckets: list[str] | None,
    selected_terminal_coverage_applied: list[bool] | None,
    selected_terminal_original_targets: list[float] | None,
    selected_terminal_targets: list[float] | None,
    device: torch.device,
) -> dict[str, Any]:
    n = int(len(selected_rows))
    env["gated_reward_path_balance_enabled"] = torch.ones((n,), device=device, dtype=torch.bool)
    env["gated_reward_path_balance_applied"] = torch.as_tensor(
        selected_balance_applied,
        device=device,
        dtype=torch.bool,
    )
    env["gated_reward_path_balance_repair_kind"] = tuple(str(v) for v in selected_repair_kinds)
    for key in (
        "original_topology_pass",
        "balanced_topology_pass",
        "gated_topology_pass",
    ):
        env[f"gated_reward_path_{key}"] = torch.as_tensor(
            [bool(row[key]) for row in selected_rows],
            device=device,
            dtype=torch.bool,
        )
    if selected_terminal_coverage_buckets is not None:
        env["gated_terminal_coverage_bucket"] = tuple(
            str(v) for v in selected_terminal_coverage_buckets
        )
        env["gated_terminal_coverage_applied"] = torch.as_tensor(
            selected_terminal_coverage_applied,
            device=device,
            dtype=torch.bool,
        )
        env["gated_terminal_coverage_original_target"] = torch.as_tensor(
            selected_terminal_original_targets,
            device=device,
            dtype=torch.float32,
        )
        env["gated_terminal_coverage_target"] = torch.as_tensor(
            selected_terminal_targets,
            device=device,
            dtype=torch.float32,
        )
    return env


def install_gated_reward_path_balance_milestone(
    prior,
    *,
    state_gain_min: float = 0.7,
    action_gain_min: float = 0.06,
    state_to_action_ratio_max: float = 10.0,
    max_attempts: int = 4096,
    profile: RewardPathBalanceProfile | None = None,
    terminal_count_coverage_enabled: bool = False,
) -> None:
    """Install the gated reward-path milestone on one EnvironmentPrior instance."""

    install_environment_prior_reward_group_balance_patch()
    profile = profile or RewardPathBalanceProfile()
    thresholds = TopologyThresholds(
        state_gain_min=float(state_gain_min),
        action_gain_min=float(action_gain_min),
        state_to_action_ratio_max=float(state_to_action_ratio_max),
    )
    profile.validate()
    thresholds.validate()
    prior.config["reward_topology_conditioned_sampling_enabled"] = True
    prior.config["reward_state_input_gain_fraction_conditioned_sampling_enabled"] = False
    prior.config["reward_state_input_gain_fraction_conditioned_min"] = float(state_gain_min)
    prior.config["reward_action_input_gain_fraction_conditioned_min"] = float(action_gain_min)
    prior.config["reward_state_to_action_gain_ratio_conditioned_max"] = float(state_to_action_ratio_max)
    prior.config["reward_topology_conditioned_sampling_max_attempts"] = int(max_attempts)
    prior.config["reward_path_milestone"] = (
        GATED_REWARD_PATH_TERMINAL_COVERAGE_MILESTONE
        if bool(terminal_count_coverage_enabled)
        else GATED_REWARD_PATH_MILESTONE
    )
    prior.config["reward_path_balance_profile"] = str(profile.name)
    prior.config["terminal_count_coverage_enabled"] = bool(terminal_count_coverage_enabled)

    def _sample_gated_conditioned(
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
        accepted_ratios = [float("nan")] * batch_size
        selected_rows: list[dict[str, Any] | None] = [None] * batch_size
        selected_repair_kinds = ["reject_still_fail"] * batch_size
        selected_balance_applied = [False] * batch_size
        selected_terminal_coverage_buckets = ["original"] * batch_size
        selected_terminal_coverage_applied = [False] * batch_size
        selected_terminal_original_targets = [float("nan")] * batch_size
        selected_terminal_targets = [float("nan")] * batch_size
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
            candidate_seeds = [int(base_seeds[i]) + int(attempt_idx) for i in active]
            candidate_env = self._sample_environment_family_coarse_batch(
                h_list=candidate_h_list,
                device=device,
                rng_seeds=candidate_seeds,
                build_policy_generator=build_policy_generator,
                preserve_skipped_generator_rng=preserve_skipped_generator_rng,
                _disable_rejection=True,
            )
            total_draws += len(active)
            state_gains = _tensor_list(candidate_env, "reward_state_input_gain_fraction", len(active))
            action_gains = _tensor_list(candidate_env, "reward_action_input_gain_fraction", len(active))
            noise_gains = _tensor_list(candidate_env, "reward_noise_input_gain_fraction", len(active))
            ratios = _tensor_list(candidate_env, "reward_state_to_action_gain_ratio", len(active))
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
                }
                row = row_with_balanced_topology(row, profile=profile)
                decision = decide_gated_reward_path_repair(row, thresholds=thresholds)
                row = apply_decision_to_row(row, decision)
                if not bool(decision.gated_topology_pass):
                    continue
                accepted_h = select_h_for_decision(
                    candidate_h_list[local_idx],
                    decision=decision,
                    profile=profile,
                )
                accepted_h["reward_state_input_gain_fraction_rejection_min"] = 0.0
                original_terminal_target = float(accepted_h.get("terminal_reset_count_target", 0.0) or 0.0)
                terminal_bucket = "original"
                terminal_target = float(max(0.0, original_terminal_target))
                terminal_applied = False
                if bool(terminal_count_coverage_enabled):
                    terminal_bucket = _terminal_coverage_bucket_for_slot(global_idx, batch_size)
                    if (
                        terminal_bucket != "original"
                        and _h_bool(accepted_h.get("terminal_reset_enabled", False))
                    ):
                        terminal_target = _terminal_coverage_target_for_bucket(
                            original_target=original_terminal_target,
                            bucket=terminal_bucket,
                            seed=int(candidate_seeds[local_idx]),
                            slot_idx=int(global_idx),
                        )
                        accepted_h["terminal_reset_count_target"] = float(terminal_target)
                        terminal_applied = True
                    else:
                        terminal_bucket = "original"
                accepted_h["gated_terminal_coverage_bucket"] = str(terminal_bucket)
                accepted_h["gated_terminal_coverage_applied"] = bool(terminal_applied)
                accepted_h["gated_terminal_coverage_original_target"] = float(original_terminal_target)
                accepted_h["gated_terminal_coverage_target"] = float(terminal_target)
                accepted_h_list[global_idx] = accepted_h
                accepted_seeds[global_idx] = int(candidate_seeds[local_idx])
                accepted_attempts[global_idx] = int(attempt_idx + 1)
                prefix = "balanced_" if bool(decision.selected_balance_applied) else ""
                accepted_state_gains[global_idx] = float(row[f"{prefix}reward_state_input_gain_fraction"])
                accepted_action_gains[global_idx] = float(row[f"{prefix}reward_action_input_gain_fraction"])
                accepted_ratios[global_idx] = float(row[f"{prefix}reward_state_to_action_gain_ratio"])
                selected_rows[global_idx] = copy.deepcopy(row)
                selected_repair_kinds[global_idx] = str(decision.repair_kind)
                selected_balance_applied[global_idx] = bool(decision.selected_balance_applied)
                selected_terminal_coverage_buckets[global_idx] = str(terminal_bucket)
                selected_terminal_coverage_applied[global_idx] = bool(terminal_applied)
                selected_terminal_original_targets[global_idx] = float(original_terminal_target)
                selected_terminal_targets[global_idx] = float(terminal_target)
                pending.discard(global_idx)
            candidate_env["transition_generator"] = None
            candidate_env["policy_generator"] = None
            candidate_env = None

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
                "failed to sample gated reward-path balanced exact SCM environments "
                f"within {int(max_attempts)} candidate rounds; {pending_str}"
            )

        final_h_list = [copy.deepcopy(h) for h in accepted_h_list if h is not None]
        final_seeds = [int(s) for s in accepted_seeds if s is not None]
        if len(final_h_list) != batch_size or len(final_seeds) != batch_size:
            raise RuntimeError("internal gated milestone accepted list length mismatch")
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
            accepted_state_to_action_ratios=accepted_ratios,
            total_candidate_draws=total_draws,
            device=device,
        )
        final_env = _attach_gated_metadata(
            final_env,
            selected_rows=[row if row is not None else {} for row in selected_rows],
            selected_repair_kinds=selected_repair_kinds,
            selected_balance_applied=selected_balance_applied,
            selected_terminal_coverage_buckets=(
                selected_terminal_coverage_buckets
                if bool(terminal_count_coverage_enabled)
                else None
            ),
            selected_terminal_coverage_applied=(
                selected_terminal_coverage_applied
                if bool(terminal_count_coverage_enabled)
                else None
            ),
            selected_terminal_original_targets=(
                selected_terminal_original_targets
                if bool(terminal_count_coverage_enabled)
                else None
            ),
            selected_terminal_targets=(
                selected_terminal_targets
                if bool(terminal_count_coverage_enabled)
                else None
            ),
            device=torch.device(device),
        )
        return final_h_list, final_env, final_seeds

    prior._sample_batch_hypers_and_environment_family_coarse_batch_conditioned = MethodType(
        _sample_gated_conditioned,
        prior,
    )


def normalize_prior_milestone(value: Any) -> str:
    text = str(value or SAMPLED_TOPOLOGY_MILESTONE).strip().lower()
    aliases = {
        "sampled": SAMPLED_TOPOLOGY_MILESTONE,
        "sampled_topology_conditioned": SAMPLED_TOPOLOGY_MILESTONE,
        "gated": GATED_REWARD_PATH_MILESTONE,
        "gated_reward_path": GATED_REWARD_PATH_MILESTONE,
        "gated_reward_path_balancing": GATED_REWARD_PATH_MILESTONE,
        "gated_terminal": GATED_REWARD_PATH_TERMINAL_COVERAGE_MILESTONE,
        "gated_reward_path_terminal": GATED_REWARD_PATH_TERMINAL_COVERAGE_MILESTONE,
        "gated_reward_path_terminal_coverage": GATED_REWARD_PATH_TERMINAL_COVERAGE_MILESTONE,
        "gated_reward_path_balance_terminal": GATED_REWARD_PATH_TERMINAL_COVERAGE_MILESTONE,
    }
    text = aliases.get(text, text)
    if text not in {SAMPLED_TOPOLOGY_MILESTONE, *GATED_REWARD_PATH_MILESTONES}:
        raise ValueError(
            "Unsupported PPO pack prior milestone "
            f"{value!r}; expected {SAMPLED_TOPOLOGY_MILESTONE!r}, "
            f"{GATED_REWARD_PATH_MILESTONE!r}, or "
            f"{GATED_REWARD_PATH_TERMINAL_COVERAGE_MILESTONE!r}."
        )
    return text
