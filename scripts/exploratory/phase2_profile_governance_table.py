#!/usr/bin/env python
"""Build a unified profile-governance table for milestone-2 successors.

Exploratory/read-only.  This script does not sample environments, train PPO, or
modify exact SCM / fit_model / PPO paths.  It consolidates the current
profile-axis evidence into one maintainable governance artifact so candidate
repairs are judged by profile quality first and trainability only as a guard.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


ART = Path("/home/chen/RLPFN/artifacts")
DEFAULT_OUTPUT_DIR = ART / "phase2_profile_governance_table_0511"

UNIFIED_AUDIT_JSON = ART / "phase2_unified_profile_distribution_audit_0511/unified_profile_distribution_audit.json"
PUSH2_CSV = (
    ART
    / "phase2_joint_profile_guarded_locomotion_push2_candidate_gpu0_0511/"
    "selected_joint_profile_guarded_prior_envs.csv"
)
PUSH2_SMOKE_JSON = (
    ART
    / "phase2_joint_profile_push2_smoke_summary_gpu0_0511/"
    "locomotion_guarded_smoke_summary.json"
)
STRICT_FRONTIER_JSON = (
    ART
    / "phase2_strict_profile_gain_balance_frontier_formal_0510/"
    "strict_profile_gain_balance_frontier_report.json"
)
FEEDBACK_SUBPROFILE_CONTROL_JSON = (
    ART / "phase2_feedback_subprofile_control_audit_0511/feedback_subprofile_control_audit.json"
)
FEEDBACK_BRANCH_JSON = (
    ART / "phase2_feedback_strict_subprofile_branch_v1_n32_0511/feedback_strict_subprofile_branch_report.json"
)
FEEDBACK_BRANCH_SMOKE_JSON = (
    ART
    / "phase2_feedback_strict_subprofile_branch_v1_n32_sidecar_summary_seed9512_0511"
    / "pendulum_like_sidecar_delta_summary_report.json"
)
FEEDBACK_HISTORY_JSON = ART / "phase2_feedback_history_recall_audit_0511/feedback_history_recall_audit.json"
MOUNTAINCAR_DETECTOR_JSON = (
    ART / "phase2_mountaincar_future_basin_detector_v1_0511/mountaincar_future_basin_detector_report.json"
)
MOUNTAINCAR_ANCHOR_JSON = (
    ART / "phase2_gym_mountaincar_future_basin_anchor_0511/gym_mountaincar_future_basin_anchor_report.json"
)
PROFILE_BAND_SELECTOR_V2_JSON = (
    ART / "phase2_profile_band_selector_v2c_centered_protected_feedback_safe_0511/profile_band_selector_report.json"
)
PROFILE_BAND_SELECTOR_V2_SMOKE_JSON = (
    ART
    / "phase2_profile_band_v2c_centered_protected_feedback_safe_delta_smoke_summary_0511"
    / "locomotion_guarded_smoke_summary.json"
)
LONG_HORIZON_JSON = (
    ART / "phase2_long_horizon_control_axis_audit_0509/long_horizon_control_axis_audit_report.json"
)
WITNESS_JSON = ART / "phase2_validation_mode_witness_probe_0508/witness_report.json"

RULES = ("gym_full_obs_action_range", "gym_q90_obs_action_range")
SCOPES = ("full", "q90")
RULE_TO_SCOPE = {
    "gym_full_obs_action_range": "full",
    "gym_q90_obs_action_range": "q90",
}
BROAD_AXES = (
    "locomotion_witness",
    "low_energy_not_ctrl_only",
    "sign_or_phase_sensitive",
    "state_conditioned_energy_injection",
)


AXIS_REGISTRY: list[dict[str, Any]] = [
    {
        "axis_id": "low_energy_weak_action_optimum",
        "profile_key": "low_energy_not_ctrl_only",
        "gym_targets": ["Ant-v5", "HalfCheetah-v5", "Swimmer-v5"],
        "definition": "zero/half or weak action is globally or near-globally strong relative to energetic actions",
        "trust_status": "quota_ready_protected_band",
        "control_role": "protected_band",
        "why": "covered in milestone2 and should not be squeezed by other repairs",
    },
    {
        "axis_id": "sign_phase_sensitive_action",
        "profile_key": "sign_or_phase_sensitive",
        "gym_targets": ["Pendulum-v1", "Reacher-v5", "locomotion phase/direction"],
        "definition": "action sign/phase changes return by a nontrivial margin",
        "trust_status": "quota_ready_protected_band_but_not_sufficient",
        "control_role": "protected_band",
        "why": "necessary broad witness, but too coarse to define feedback quality",
    },
    {
        "axis_id": "state_conditioned_energy_injection",
        "profile_key": "state_conditioned_energy_injection",
        "gym_targets": ["Pendulum-v1", "MountainCarContinuous-v0", "recovery/control cases"],
        "definition": "state-conditioned feedback beats non-state-conditioned baselines",
        "trust_status": "diagnostic_broad_axis",
        "control_role": "protected_band_until_future_basin_detector",
        "why": "useful broad axis but may include immediate-reward shortcuts",
    },
    {
        "axis_id": "context_identifiable_closed_loop_settle_feedback",
        "profile_key": None,
        "gym_targets": ["Pendulum-v1", "stabilization-style tasks"],
        "definition": "feature/sign/gain/profile can be inferred from context and closed-loop feedback improves suffix/settle behavior",
        "trust_status": "pressure_point_subprofile_not_single_quota",
        "control_role": "subprofile_guard_needed",
        "why": "strict pool is dominated by constant/gain=2.0; broad feedback count is not enough",
    },
    {
        "axis_id": "locomotion_sustained_energy_tradeoff",
        "profile_key": "locomotion_witness",
        "gym_targets": ["Ant-v5", "HalfCheetah-v5", "Swimmer-v5", "Walker2d-v5", "Hopper-v5", "Humanoid-v5"],
        "definition": "sustained nontrivial action improves environment reward while terminal is not the primary channel",
        "trust_status": "quota_ready_only_with_joint_profile_guard",
        "control_role": "raised_band_candidate",
        "why": "scarce in milestone2 and push2 shows it can be raised without joint-profile collapse",
    },
    {
        "axis_id": "terminal_survival_action_sensitive",
        "profile_key": None,
        "gym_targets": ["InvertedPendulum-v5", "InvertedDoublePendulum-v5", "some survival locomotion tasks"],
        "definition": "action causally changes first_done_step/survival length and PPO advantages consume survival length",
        "trust_status": "diagnostic_only",
        "control_role": "do_not_quota_yet",
        "why": "Gym done mismatch was outlier-concentrated and reward-shaping candidate was not clean",
    },
    {
        "axis_id": "mountaincar_upfront_investment_future_basin",
        "profile_key": None,
        "gym_targets": ["MountainCarContinuous-v0", "delayed payoff / early investment tasks"],
        "definition": "early costly control changes future state basin and later reward enough to repay control cost",
        "trust_status": "candidate_detector_not_quota_ready",
        "control_role": "log_only_until_gym_anchor_and_health_guard",
        "why": "candidate detector exists but needs Gym-side anchoring and fixed-list health/diversity validation",
    },
]


def _read_json(path: str | Path) -> Any:
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n", "", "none", "nan"}:
        return False
    return bool(value)


def _safe_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    return value


def _pct(value: Any) -> str:
    try:
        out = float(value)
    except Exception:
        return "na"
    return "na" if not math.isfinite(out) else f"{100.0 * out:.1f}%"


def _mean(values: list[float]) -> float:
    clean = [v for v in values if math.isfinite(v)]
    return float(sum(clean) / len(clean)) if clean else math.nan


def _entropy(counter: Counter[tuple[bool, ...]]) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    ent = 0.0
    for count in counter.values():
        p = count / total
        ent -= p * math.log(p + 1e-12)
    return float(ent / math.log(2 ** len(BROAD_AXES)))


def _profile_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    cells: Counter[tuple[bool, ...]] = Counter(
        tuple(_safe_bool(row.get(axis)) for axis in BROAD_AXES) for row in rows
    )
    max_cell = max(cells.values()) if cells else 0
    return {
        "n": n,
        "mode_rates": {
            axis: float(sum(_safe_bool(row.get(axis)) for row in rows) / max(1, n))
            for axis in BROAD_AXES
        },
        "joint_active_cells": int(len(cells)),
        "joint_max_cell_share": float(max_cell / max(1, n)),
        "joint_entropy_norm": _entropy(cells),
        "top_joint_cells": [
            {
                "cell": "".join("1" if bit else "0" for bit in cell),
                "count": int(count),
                "share": float(count / max(1, n)),
            }
            for cell, count in cells.most_common(8)
        ],
    }


def _scope_from_row(row: dict[str, Any]) -> str | None:
    if "scope" in row and str(row["scope"]) in SCOPES:
        return str(row["scope"])
    rule = str(row.get("candidate_rule") or row.get("rule") or "")
    return RULE_TO_SCOPE.get(rule)


def _candidate_profile_from_csv(path: Path) -> dict[str, Any]:
    rows = _read_csv(path)
    out: dict[str, Any] = {}
    for scope in SCOPES:
        scoped = [row for row in rows if _scope_from_row(row) == scope]
        out[scope] = _profile_stats(scoped)
    return out


def _baseline_from_unified(unified: dict[str, Any] | None) -> dict[str, Any]:
    if not unified:
        return {}
    out = {}
    for scope, body in unified.get("baseline", {}).items():
        broad = body.get("broad_profile", {})
        out[scope] = {
            "n": broad.get("n"),
            "mode_rates": broad.get("mode_rates", {}),
            "joint_active_cells": broad.get("joint_profile_active_cells"),
            "joint_max_cell_share": broad.get("joint_profile_dominant_share"),
            "joint_entropy_norm": None,
            "delta_positive_fraction_monitor": broad.get("raw_reward_sum_delta_positive_fraction"),
        }
    return out


def _push2_guard(push2_smoke: dict[str, Any] | None) -> dict[str, Any]:
    if not push2_smoke:
        return {}
    out = {}
    for scope, body in push2_smoke.get("scopes", {}).items():
        cand = body.get("candidate", {})
        raw = cand.get("raw_reward_sum_delta", {})
        env = cand.get("reward_env_sum_delta", {})
        out[scope] = {
            "raw_delta_mean_guard": raw.get("mean"),
            "delta_positive_fraction_monitor": raw.get("positive_fraction"),
            "env_delta_mean_guard": env.get("mean"),
            "diversity_guard_pass": body.get("diversity_guard", {}).get("pass"),
            "delta_smoke_sufficient_guard": body.get("sufficiency_read", {}).get("pass"),
        }
    return out


def _strict_feedback_summary(
    strict: dict[str, Any] | None,
    control: dict[str, Any] | None,
    branch: dict[str, Any] | None,
    branch_smoke: dict[str, Any] | None,
) -> dict[str, Any]:
    if not strict:
        return {"available": False}
    pool = strict.get("pool_summary", {})
    lists = strict.get("generated_lists", [])
    best_n64 = next((item for item in lists if item.get("n") == 64), None)
    return {
        "available": True,
        "pool_profile_counts": pool.get("profile_counts"),
        "pool_gain_counts": pool.get("gain_counts"),
        "pool_profile_dominance": pool.get("profile_dominance"),
        "pool_gain_dominance": pool.get("gain_dominance"),
        "selection_only_sufficient": strict.get("decision", {}).get("selection_only_is_sufficient"),
        "best_effort_n64": best_n64,
        "decision_read": strict.get("decision", {}).get("read"),
        "control_audit": {
            "available": bool(control),
            "simple_existing_scalar_knob_sufficient": (control or {}).get("simple_knob_sufficient"),
            "target_counts": (control or {}).get("target_counts"),
            "strongest_association": (control or {}).get("strongest_association"),
            "read": (control or {}).get("read"),
        },
        "branch_control": {
            "available": bool(branch),
            "selected_summary": (branch or {}).get("selected_summary"),
            "read": (branch or {}).get("read"),
            "short_smoke_overall": (branch_smoke or {}).get("overall"),
        },
    }


def _mountaincar_summary(
    detector: dict[str, Any] | None,
    anchor: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not detector:
        return {"available": False}
    return {
        "available": True,
        "status": "candidate_detector_not_quota_ready",
        "definition": detector.get("definition"),
        "summaries": detector.get("summaries"),
        "read": detector.get("read"),
        "gym_anchor": {
            "available": bool(anchor),
            "read": (anchor or {}).get("read"),
            "per_seed_summary": (anchor or {}).get("per_seed_summary"),
        },
    }


def _selector_profile(selector: dict[str, Any] | None) -> tuple[dict[str, Any], dict[str, Any]]:
    if not selector:
        return {}, {}
    profile: dict[str, Any] = {}
    guards: dict[str, Any] = {}
    for rule, payload in (selector.get("rules") or {}).items():
        scope = RULE_TO_SCOPE.get(rule)
        if not scope:
            continue
        summary = payload.get("selected_summary") or {}
        profile[scope] = {
            "n": summary.get("n"),
            "mode_rates": {
                "locomotion_witness": summary.get("locomotion_witness_rate"),
                "low_energy_not_ctrl_only": summary.get("low_energy_not_ctrl_only_rate"),
                "sign_or_phase_sensitive": summary.get("sign_or_phase_sensitive_rate"),
                "state_conditioned_energy_injection": summary.get("state_conditioned_energy_injection_rate"),
            },
            "joint_active_cells": summary.get("joint_profile_active_cells"),
            "joint_max_cell_share": summary.get("joint_profile_dominant_rate"),
            "joint_entropy_norm": summary.get("joint_profile_entropy_norm"),
            "strict_feedback_count": summary.get("strict_feedback_count"),
            "strict_feedback_profile_counts": summary.get("strict_feedback_profile_counts"),
            "strict_feedback_gain_counts": summary.get("strict_feedback_gain_counts"),
        }
        strict = (payload.get("band_details") or {}).get("strict_feedback_subprofile_branch", {})
        guards[scope] = {
            "selector_guard_pass": payload.get("guard_pass"),
            "profile_guard_pass": payload.get("profile_guard_pass"),
            "strict_feedback_branch_pass": strict.get("passed"),
            "strict_feedback_selected_count": strict.get("selected_strict_count"),
            "strict_feedback_selected_nonconstant_count": strict.get("selected_nonconstant_count"),
            "strict_feedback_selected_non_gain2_count": strict.get("selected_non_gain2_count"),
        }
    return profile, guards


def _selector_smoke_guards(smoke: dict[str, Any] | None) -> dict[str, Any]:
    if not smoke:
        return {}
    out: dict[str, Any] = {}
    for scope, payload in (smoke.get("scopes") or {}).items():
        candidate = payload.get("candidate") or {}
        out[scope] = {
            "delta_positive_fraction_monitor": (candidate.get("raw_reward_sum_delta") or {}).get(
                "positive_fraction"
            ),
            "raw_delta_mean_guard": (candidate.get("raw_reward_sum_delta") or {}).get("mean"),
            "env_delta_mean_guard": (candidate.get("reward_env_sum_delta") or {}).get("mean"),
            "diversity_guard_pass": payload.get("diversity_guard", {}).get("pass"),
            "delta_smoke_sufficiency_pass": payload.get("sufficiency_read", {}).get("pass"),
        }
    return out


def _feedback_summary(history: dict[str, Any] | None, long_horizon: dict[str, Any] | None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "history_read": (history or {}).get("read"),
        "long_horizon_available": bool(long_horizon),
    }
    if long_horizon:
        out["protected_milestone"] = long_horizon.get("milestone_boundary", {}).get("current_protected_milestone")
        out["prior_long_horizon_signature"] = long_horizon.get("prior_long_horizon_signature")
        out["gym_pendulum_anchor"] = long_horizon.get("gym_pendulum_anchor")
    return out


def _witness_summary(witness: dict[str, Any] | None) -> dict[str, Any]:
    if not witness:
        return {"available": False}
    return {
        "available": True,
        "ant_locomotion_energy": witness.get("ant_locomotion_energy"),
        "terminal_survival": witness.get("inverted_terminal_survival"),
        "selected_next_witness": witness.get("selected_next_witness"),
        "sufficient_to_change_milestone": witness.get("sufficient_to_change_milestone"),
    }


def _axis_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    baseline = report["profile_versions"]["milestone2_baseline"]["profile"]
    push2 = report["profile_versions"]["profile_balanced_push2"]["profile"]
    rows: list[dict[str, Any]] = []
    for axis in AXIS_REGISTRY:
        key = axis.get("profile_key")
        row: dict[str, Any] = {
            "axis_id": axis["axis_id"],
            "trust_status": axis["trust_status"],
            "control_role": axis["control_role"],
            "profile_key": key or "",
            "gym_targets": ",".join(axis["gym_targets"]),
            "definition": axis["definition"],
            "why": axis["why"],
        }
        if key:
            for scope in SCOPES:
                row[f"milestone2_{scope}_rate"] = baseline.get(scope, {}).get("mode_rates", {}).get(key)
                row[f"push2_{scope}_rate"] = push2.get(scope, {}).get("mode_rates", {}).get(key)
        rows.append(row)
    return rows


def _profile_version_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for version_id, version in report["profile_versions"].items():
        profile = version.get("profile", {})
        guards = version.get("guards", {})
        for scope in SCOPES:
            body = profile.get(scope, {})
            guard = guards.get(scope, {})
            row = {
                "version_id": version_id,
                "scope": scope,
                "status": version.get("status"),
                "purpose": version.get("purpose"),
                "n": body.get("n"),
                "locomotion": body.get("mode_rates", {}).get("locomotion_witness"),
                "low_energy": body.get("mode_rates", {}).get("low_energy_not_ctrl_only"),
                "sign_phase": body.get("mode_rates", {}).get("sign_or_phase_sensitive"),
                "state_conditioned": body.get("mode_rates", {}).get("state_conditioned_energy_injection"),
                "active_cells": body.get("joint_active_cells"),
                "max_cell_share": body.get("joint_max_cell_share"),
                "entropy_norm": body.get("joint_entropy_norm"),
                "delta_positive_fraction_monitor": guard.get("delta_positive_fraction_monitor"),
                "raw_delta_mean_guard": guard.get("raw_delta_mean_guard"),
                "diversity_guard_pass": guard.get("diversity_guard_pass"),
            }
            rows.append(row)
    return rows


def build_report() -> dict[str, Any]:
    unified = _read_json(UNIFIED_AUDIT_JSON)
    push2_smoke = _read_json(PUSH2_SMOKE_JSON)
    strict = _read_json(STRICT_FRONTIER_JSON)
    feedback_subprofile_control = _read_json(FEEDBACK_SUBPROFILE_CONTROL_JSON)
    feedback_branch = _read_json(FEEDBACK_BRANCH_JSON)
    feedback_branch_smoke = _read_json(FEEDBACK_BRANCH_SMOKE_JSON)
    history = _read_json(FEEDBACK_HISTORY_JSON)
    mountaincar_detector = _read_json(MOUNTAINCAR_DETECTOR_JSON)
    mountaincar_anchor = _read_json(MOUNTAINCAR_ANCHOR_JSON)
    selector_v2 = _read_json(PROFILE_BAND_SELECTOR_V2_JSON)
    selector_v2_smoke = _read_json(PROFILE_BAND_SELECTOR_V2_SMOKE_JSON)
    long_horizon = _read_json(LONG_HORIZON_JSON)
    witness = _read_json(WITNESS_JSON)

    milestone2_profile = _baseline_from_unified(unified)
    push2_profile = _candidate_profile_from_csv(PUSH2_CSV)
    push2_guards = _push2_guard(push2_smoke)
    selector_v2_profile, selector_v2_guards = _selector_profile(selector_v2)
    selector_v2_smoke_guards = _selector_smoke_guards(selector_v2_smoke)
    for scope, guard in selector_v2_smoke_guards.items():
        selector_v2_guards.setdefault(scope, {}).update(guard)
    selector_v2_smoke_pass = bool(
        (selector_v2_smoke or {}).get("overall", {}).get("candidate_sufficient_for_smoke")
    )

    report: dict[str, Any] = {
        "analysis_entry": "phase2_profile_governance_table",
        "contract": {
            "exploratory_only": True,
            "read_only_existing_artifacts": True,
            "does_not_sample_environments": True,
            "does_not_train": True,
            "does_not_modify_exact_scm": True,
            "does_not_modify_fit_model": True,
            "does_not_modify_ppo": True,
            "raw_delta_is_guard_not_objective": True,
        },
        "inputs": {
            "unified_audit_json": str(UNIFIED_AUDIT_JSON),
            "push2_csv": str(PUSH2_CSV),
            "push2_smoke_json": str(PUSH2_SMOKE_JSON),
            "strict_frontier_json": str(STRICT_FRONTIER_JSON),
            "feedback_subprofile_control_json": str(FEEDBACK_SUBPROFILE_CONTROL_JSON),
            "feedback_branch_json": str(FEEDBACK_BRANCH_JSON),
            "feedback_branch_smoke_json": str(FEEDBACK_BRANCH_SMOKE_JSON),
            "feedback_history_json": str(FEEDBACK_HISTORY_JSON),
            "mountaincar_detector_json": str(MOUNTAINCAR_DETECTOR_JSON),
            "mountaincar_anchor_json": str(MOUNTAINCAR_ANCHOR_JSON),
            "profile_band_selector_v2_json": str(PROFILE_BAND_SELECTOR_V2_JSON),
            "profile_band_selector_v2_smoke_json": str(PROFILE_BAND_SELECTOR_V2_SMOKE_JSON),
            "long_horizon_json": str(LONG_HORIZON_JSON),
            "witness_json": str(WITNESS_JSON),
        },
        "axis_registry": AXIS_REGISTRY,
        "profile_versions": {
            "milestone1_sampled_topology_conditioned": {
                "status": "protected_milestone",
                "purpose": "topology health/parity; not reopened by profile work",
                "profile": {},
                "guards": {},
            },
            "milestone2_gated_reward_path_balance": {
                "status": "protected_milestone",
                "purpose": "reward-path width/yield balance with full/q90 health",
                "profile": milestone2_profile,
                "guards": {
                    scope: {
                        "delta_positive_fraction_monitor": milestone2_profile.get(scope, {}).get(
                            "delta_positive_fraction_monitor"
                        )
                    }
                    for scope in SCOPES
                },
            },
            "milestone2_baseline": {
                "status": "reference_profile",
                "purpose": "baseline profile rates from milestone2 fixed-list audit",
                "profile": milestone2_profile,
                "guards": {
                    scope: {
                        "delta_positive_fraction_monitor": milestone2_profile.get(scope, {}).get(
                            "delta_positive_fraction_monitor"
                        )
                    }
                    for scope in SCOPES
                },
            },
            "profile_balanced_push2": {
                "status": "exploratory_candidate_not_milestone",
                "purpose": "raise locomotion inside joint-profile bands; evidence for unified profile-band selector",
                "profile": push2_profile,
                "guards": push2_guards,
            },
            "profile_band_v2c_centered_protected_feedback_safe": {
                "status": (
                    "exploratory_candidate_short_smoke_pass"
                    if selector_v2_smoke_pass
                    else "exploratory_candidate_delta_smoke_failed"
                ),
                "purpose": (
                    "full/q90 unified selector with centered protected bands and minimal feedback pressure; "
                    "judged by post-pre delta mean and diversity; delta-positive fraction is monitoring only"
                ),
                "profile": selector_v2_profile,
                "guards": selector_v2_guards,
            },
            "strict_feedback_profile_gain_frontier": {
                "status": "diagnostic_frontier",
                "purpose": "shows strict feedback/Pendulum-like subprofile collapse and selection-only limits",
                "profile": {},
                "guards": {},
                "subprofile": _strict_feedback_summary(
                    strict,
                    feedback_subprofile_control,
                    feedback_branch,
                    feedback_branch_smoke,
                ),
            },
            "mountaincar_future_basin_detector": {
                "status": "candidate_detector_not_quota_ready",
                "purpose": "upfront investment / delayed basin detector candidate; not quota-controlled yet",
                "profile": {},
                "guards": {},
                "detector": _mountaincar_summary(mountaincar_detector, mountaincar_anchor),
            },
        },
        "feedback_evidence": _feedback_summary(history, long_horizon),
        "witness_evidence": _witness_summary(witness),
        "reductive_control_design": {
            "name": "profile_band_selector",
            "principle": "one auditable selector over detector bits/subprofiles instead of one wrapper per mode",
            "steps": [
                "compute detector bits and subprofile labels",
                "apply marginal protected bands",
                "apply joint-cell cap",
                "apply subprofile caps only for quota-ready subprofiles",
                "apply health guards after candidate materialization",
            ],
            "version_space": {
                "v0_reference": "milestone2_gated_reward_path_balance",
                "v1_profile_balanced": "use push2-style locomotion band under joint profile guard",
                "v2_feedback_subprofile": (
                    "full/q90 selector carries strict feedback branch constraints; v2c remains exploratory "
                    "and must keep post-pre delta and diversity guards healthy"
                ),
                "v3_mountaincar_future_basin": "candidate upfront-investment detector now Gym-anchored; prior fixed-list guard pending",
            },
        },
        "next_read": {
            "closest_to_next_milestone": False,
            "reason": (
                "v2c now gives a unified full/q90 selector with centered protected bands, minimal "
                "feedback pressure, and delta/diversity smoke is only a guard rather than the objective.  "
                "It is still not a milestone because feedback source-pool collapse remains a structural "
                "limit and MountainCar is only Gym-anchored."
            ),
            "highest_value_next": [
                "keep the unified selector framework and continue profile-pressure repair",
                "treat delta-positive fraction as monitoring only; do not optimize profile around it",
                "validate MountainCar upfront-investment/future-basin detector on prior fixed lists before any quota",
            ],
        },
    }
    report["axis_rows"] = _axis_rows(report)
    report["profile_version_rows"] = _profile_version_rows(report)
    return report


def _markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Phase2 Profile Governance Table")
    lines.append("")
    lines.append("Exploratory/read-only.  This is the profile-quality dashboard for possible milestone-2 successors.")
    lines.append("")
    lines.append("## Contract")
    lines.append("")
    for key, value in report["contract"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    lines.append("## Axis Registry")
    lines.append("")
    lines.append("| axis | trust | role | milestone2 full/q90 | push2 full/q90 | read |")
    lines.append("|---|---|---|---:|---:|---|")
    for row in report["axis_rows"]:
        full_base = row.get("milestone2_full_rate")
        q90_base = row.get("milestone2_q90_rate")
        full_push = row.get("push2_full_rate")
        q90_push = row.get("push2_q90_rate")
        lines.append(
            "| {axis} | {trust} | {role} | {base} / {base_q90} | {push} / {push_q90} | {why} |".format(
                axis=row["axis_id"],
                trust=row["trust_status"],
                role=row["control_role"],
                base=_pct(full_base),
                base_q90=_pct(q90_base),
                push=_pct(full_push),
                push_q90=_pct(q90_push),
                why=row["why"],
            )
        )
    lines.append("")
    lines.append("## Profile Versions")
    lines.append("")
    lines.append("| version | scope | status | locomotion | low-energy | sign/phase | state-cond | active cells | max cell | entropy | delta+ monitor | diversity |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for row in report["profile_version_rows"]:
        lines.append(
            "| {version} | {scope} | {status} | {loc} | {low} | {sign} | {state} | {cells} | {max_cell} | {ent} | {raw} | {div} |".format(
                version=row["version_id"],
                scope=row["scope"],
                status=row["status"],
                loc=_pct(row.get("locomotion")),
                low=_pct(row.get("low_energy")),
                sign=_pct(row.get("sign_phase")),
                state=_pct(row.get("state_conditioned")),
                cells=row.get("active_cells", "na"),
                max_cell=_pct(row.get("max_cell_share")),
                ent="na" if row.get("entropy_norm") is None else f"{float(row['entropy_norm']):.3f}",
                raw=_pct(row.get("delta_positive_fraction_monitor")),
                div=row.get("diversity_guard_pass", "na"),
            )
        )
    lines.append("")
    lines.append("## Feedback Subprofile Pressure")
    lines.append("")
    strict = report["profile_versions"]["strict_feedback_profile_gain_frontier"].get("subprofile", {})
    lines.append(f"- available: `{strict.get('available')}`")
    lines.append(f"- pool profile dominance: `{strict.get('pool_profile_dominance')}`")
    lines.append(f"- pool gain dominance: `{strict.get('pool_gain_dominance')}`")
    lines.append(f"- selection-only sufficient: `{strict.get('selection_only_sufficient')}`")
    lines.append(f"- read: {strict.get('decision_read')}")
    control = strict.get("control_audit", {})
    lines.append(f"- control audit available: `{control.get('available')}`")
    lines.append(
        f"- simple existing scalar knob sufficient: `{control.get('simple_existing_scalar_knob_sufficient')}`"
    )
    lines.append(f"- target counts: `{control.get('target_counts')}`")
    lines.append("")
    lines.append("## Reductive Control Design")
    lines.append("")
    design = report["reductive_control_design"]
    lines.append(f"- name: `{design['name']}`")
    lines.append(f"- principle: {design['principle']}")
    for step in design["steps"]:
        lines.append(f"- step: {step}")
    lines.append("")
    lines.append("## Next Read")
    lines.append("")
    nxt = report["next_read"]
    lines.append(f"- closest_to_next_milestone: `{nxt['closest_to_next_milestone']}`")
    lines.append(f"- reason: {nxt['reason']}")
    for item in nxt["highest_value_next"]:
        lines.append(f"- next: {item}")
    lines.append("")
    lines.append("## Sources")
    lines.append("")
    for key, value in report["inputs"].items():
        lines.append(f"- `{key}`: `{value}`")
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report = build_report()
    report_path = out_dir / "profile_governance_table.json"
    md_path = out_dir / "profile_governance_table.md"
    axis_csv = out_dir / "profile_governance_axes.csv"
    versions_csv = out_dir / "profile_governance_versions.csv"
    report["outputs"] = {
        "json": str(report_path),
        "markdown": str(md_path),
        "axis_csv": str(axis_csv),
        "versions_csv": str(versions_csv),
    }
    report_path.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(_markdown(report), encoding="utf-8")
    _write_csv(axis_csv, report["axis_rows"])
    _write_csv(versions_csv, report["profile_version_rows"])
    print(json.dumps(report["outputs"], indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
