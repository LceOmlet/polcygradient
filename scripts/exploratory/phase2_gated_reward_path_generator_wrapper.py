#!/usr/bin/env python
"""Minimal exploratory gated reward-path generator wrapper.

This module is intentionally outside the exact-SCM/default generator path.  It
defines the candidate-selection contract that sits above ordinary prior
sampling:

1. already topology-healthy -> keep original h
2. width-bias topology failure fixed by reward-path group balancing -> annotate h
3. still unhealthy -> reject

It does not sample dimensions, does not change action/obs/state dim samplers,
does not change terminal/survival behavior, and does not run PPO.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    from .phase2_reward_group_balance_runtime import (
        annotate_h_with_reward_group_balance,
        balanced_gain_summary_from_raw,
    )
except ImportError:
    from phase2_reward_group_balance_runtime import (
        annotate_h_with_reward_group_balance,
        balanced_gain_summary_from_raw,
    )


@dataclass(frozen=True)
class RewardPathBalanceProfile:
    name: str = "s3_a16_n0125"
    state_scale: float = 3.0
    action_scale: float = 16.0
    noise_scale: float = 0.125

    @classmethod
    def parse(cls, text: str) -> "RewardPathBalanceProfile":
        parts = [part.strip() for part in str(text).split(":")]
        if len(parts) != 4:
            raise ValueError("balance profile must be name:state_scale:action_scale:noise_scale")
        profile = cls(
            name=str(parts[0]),
            state_scale=float(parts[1]),
            action_scale=float(parts[2]),
            noise_scale=float(parts[3]),
        )
        profile.validate()
        return profile

    def validate(self) -> None:
        if not self.name:
            raise ValueError("balance profile name cannot be empty")
        for key in ("state_scale", "action_scale", "noise_scale"):
            value = float(getattr(self, key))
            if (not math.isfinite(value)) or value < 0.0:
                raise ValueError(f"{key} must be finite and nonnegative, got {value}")


@dataclass(frozen=True)
class TopologyThresholds:
    state_gain_min: float = 0.7
    action_gain_min: float = 0.06
    state_to_action_ratio_max: float = 10.0

    def validate(self) -> None:
        if not math.isfinite(float(self.state_gain_min)):
            raise ValueError("state_gain_min must be finite")
        if not math.isfinite(float(self.action_gain_min)):
            raise ValueError("action_gain_min must be finite")
        if not math.isfinite(float(self.state_to_action_ratio_max)):
            raise ValueError("state_to_action_ratio_max must be finite")


@dataclass(frozen=True)
class GatedRewardPathDecision:
    original_topology_pass: bool
    balanced_topology_pass: bool
    gated_topology_pass: bool
    repair_kind: str
    selected_balance_applied: bool
    reject: bool


def topology_pass(row: dict[str, Any], thresholds: TopologyThresholds, *, prefix: str = "") -> bool:
    thresholds.validate()
    state_gain = float(row[f"{prefix}reward_state_input_gain_fraction"])
    action_gain = float(row[f"{prefix}reward_action_input_gain_fraction"])
    ratio = float(row[f"{prefix}reward_state_to_action_gain_ratio"])
    return (
        math.isfinite(state_gain)
        and math.isfinite(action_gain)
        and math.isfinite(ratio)
        and state_gain >= float(thresholds.state_gain_min)
        and action_gain >= float(thresholds.action_gain_min)
        and ratio <= float(thresholds.state_to_action_ratio_max)
    )


def row_with_balanced_topology(
    row: dict[str, Any],
    *,
    profile: RewardPathBalanceProfile,
) -> dict[str, Any]:
    profile.validate()
    out = copy.deepcopy(row)
    balanced = balanced_gain_summary_from_raw(
        state_gain=float(row["reward_state_input_gain_fraction"]),
        action_gain=float(row["reward_action_input_gain_fraction"]),
        noise_gain=float(row["reward_noise_input_gain_fraction"]),
        state_scale=float(profile.state_scale),
        action_scale=float(profile.action_scale),
        noise_scale=float(profile.noise_scale),
    )
    for key, value in balanced.items():
        out[f"balanced_{key}"] = float(value)
    out["balance_profile"] = str(profile.name)
    out["balance_state_scale"] = float(profile.state_scale)
    out["balance_action_scale"] = float(profile.action_scale)
    out["balance_noise_scale"] = float(profile.noise_scale)
    return out


def decide_gated_reward_path_repair(
    row: dict[str, Any],
    *,
    thresholds: TopologyThresholds,
) -> GatedRewardPathDecision:
    original_pass = topology_pass(row, thresholds)
    balanced_pass = topology_pass(row, thresholds, prefix="balanced_")
    gated_pass = bool(original_pass or balanced_pass)
    repair_kind = (
        "original_healthy_keep"
        if original_pass
        else ("width_bias_balance" if balanced_pass else "reject_still_fail")
    )
    selected_balance_applied = bool((not original_pass) and balanced_pass)
    return GatedRewardPathDecision(
        original_topology_pass=bool(original_pass),
        balanced_topology_pass=bool(balanced_pass),
        gated_topology_pass=bool(gated_pass),
        repair_kind=repair_kind,
        selected_balance_applied=selected_balance_applied,
        reject=not gated_pass,
    )


def apply_decision_to_row(row: dict[str, Any], decision: GatedRewardPathDecision) -> dict[str, Any]:
    out = copy.deepcopy(row)
    out["original_topology_pass"] = bool(decision.original_topology_pass)
    out["balanced_topology_pass"] = bool(decision.balanced_topology_pass)
    out["gated_topology_pass"] = bool(decision.gated_topology_pass)
    out["gated_repair_kind"] = str(decision.repair_kind)
    out["selected_repair_kind"] = None if bool(decision.reject) else str(decision.repair_kind)
    out["selected_balance_applied"] = None if bool(decision.reject) else bool(decision.selected_balance_applied)
    return out


def select_h_for_decision(
    h: dict[str, Any],
    *,
    decision: GatedRewardPathDecision,
    profile: RewardPathBalanceProfile,
) -> dict[str, Any]:
    if bool(decision.reject):
        raise ValueError("cannot select h for rejected gated reward-path decision")
    if bool(decision.selected_balance_applied):
        return annotate_h_with_reward_group_balance(
            h,
            state_scale=float(profile.state_scale),
            action_scale=float(profile.action_scale),
            noise_scale=float(profile.noise_scale),
            profile=str(profile.name),
        )
    return copy.deepcopy(h)


def contract_dict(
    *,
    profile: RewardPathBalanceProfile,
    thresholds: TopologyThresholds,
) -> dict[str, Any]:
    return {
        "wrapper": "phase2_gated_reward_path_generator_wrapper",
        "exploratory_only": True,
        "does_not_modify_exact_scm": True,
        "does_not_modify_default_prior_generator": True,
        "does_not_change_dimension_samplers": True,
        "does_not_change_terminal_or_survival": True,
        "decision_order": [
            "already topology healthy -> keep original h",
            "failed original but balanced topology healthy -> annotate h for reward-path-only balancing",
            "still failed -> reject",
        ],
        "balance_profile": asdict(profile),
        "topology_thresholds": asdict(thresholds),
    }


def write_contract(path: str | Path, *, profile: RewardPathBalanceProfile, thresholds: TopologyThresholds) -> None:
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(contract_dict(profile=profile, thresholds=thresholds), indent=2, sort_keys=True) + "\n", encoding="utf-8")
