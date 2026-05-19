#!/usr/bin/env python
"""Preflight the cheap profile path for a possible milestone3 landing.

This is a read-only integration report.  It consumes the latest cheap
source384/action-state-basin artifacts and answers a narrower question than the
older profile closure plan: is the cheap path ready to become the next
milestone, and if not, what still blocks it?
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


ART = Path("/home/chen/RLPFN/artifacts")
RULES = ("gym_full_obs_action_range", "gym_q90_obs_action_range")

DEFAULT_SOURCE_JSON = (
    ART
    / "phase2_profile_source384_pendulum_action_controls_position_seed9730_0512"
    / "reward_group_balance_probe_report.json"
)
DEFAULT_BASIN_JSON = (
    ART
    / "phase2_pendulum_controllable_basin_guard_source384_action_controls_position_seed9730_0512"
    / "pendulum_controllable_basin_guard.json"
)
DEFAULT_LABEL_JSON = (
    ART
    / "phase2_profile_source384_pendulum_action_controls_position_seed9730_0512"
    / "future_basin_cached_labels"
    / "future_basin_cached_label_adapter.json"
)
DEFAULT_SELECTOR_JSON = (
    ART
    / "phase2_profile_band_selector_source384_action_controls_position_seed9730_cheap_state_guard_0512"
    / "profile_band_selector_report.json"
)
DEFAULT_SMOKE_JSON = (
    ART
    / "phase2_profile_integrated_health_diversity_smoke_source384_action_controls_position_seed9730_cheap_state_guard_mingap001_0512"
    / "profile_integrated_health_diversity_smoke.json"
)
DEFAULT_STABILITY_SMOKE_JSON = (
    ART
    / "phase2_profile_integrated_health_diversity_smoke_source384_action_controls_position_seed9730_cheap_state_guard_mingap001_s128_0513"
    / "profile_integrated_health_diversity_smoke.json"
)
DEFAULT_PARITY_JSON = (
    ART
    / "phase2_profile_source384_action_controls_position_pack_fit_parity_n4_s16_0513"
    / "profile_v2e_pack_fit_model_parity_report.json"
)
DEFAULT_CHEAP_CONTRACT_JSON = (
    ART
    / "phase2_profile_milestone3_cheap_contract_0513"
    / "profile_milestone3_cheap_contract.json"
)
DEFAULT_TOP_LEVEL_CLOSURE_JSON = (
    ART
    / "phase2_profile_closure_progress_plan_cheap_m3_0513"
    / "profile_closure_progress_plan.json"
)
DEFAULT_OUTPUT_DIR = ART / "phase2_profile_milestone3_cheap_closure_preflight_0513"


WEIGHTS = {
    "source_generation": 0.12,
    "action_state_basin_semantics": 0.18,
    "selector_materialization": 0.14,
    "integrated_health_smoke": 0.14,
    "pack_fit_parity": 0.12,
    "cheap_contract_formalization": 0.10,
    "cross_seed_or_longer_stability": 0.10,
    "top_level_gate_integration": 0.06,
    "mountaincar_scope_decision": 0.04,
}


def _read_json(path: str | Path | None) -> dict[str, Any]:
    if path is None or str(path).strip() == "":
        return {}
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        return {}
    return json.loads(resolved.read_text(encoding="utf-8"))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def _component(
    *,
    passed: bool,
    progress: float,
    read: str,
    close_condition: str,
    next_action: str,
    evidence: dict[str, Any] | None = None,
    blocker: str | None = None,
) -> dict[str, Any]:
    clipped = max(0.0, min(1.0, float(progress)))
    return {
        "passed": bool(passed),
        "progress": 1.0 if passed else clipped,
        "read": read,
        "close_condition": close_condition,
        "next_action": next_action,
        "blocker": blocker if not passed else None,
        "evidence": evidence or {},
    }


def _source_component(source: dict[str, Any], *, target_per_rule: int, raw_seen_budget: int) -> dict[str, Any]:
    counts = source.get("selected_count_by_rule", {}) or {}
    sampling = source.get("sampling", {}) or {}
    raw_seen = _safe_int(sampling.get("raw_seen"))
    axis = ((source.get("pendulum_settle_yield_axis") or {}).get("selected_counts_by_rule") or {})
    count_pass = all(_safe_int(counts.get(rule)) >= int(target_per_rule) for rule in RULES)
    speed_pass = raw_seen > 0 and raw_seen <= int(raw_seen_budget)
    axis_pass = True
    axis_evidence: dict[str, Any] = {}
    for rule in RULES:
        rule_axis = axis.get(rule, {}) or {}
        bins = rule_axis.get("bin_counts", {}) or {}
        values = [_safe_int(v) for v in bins.values()]
        axis_evidence[rule] = {"enabled": _safe_int(rule_axis.get("enabled")), "bin_counts": bins}
        if _safe_int(rule_axis.get("enabled")) < 96 or len(set(values)) != 1 or len(values) < 4:
            axis_pass = False
    passed = bool(count_pass and speed_pass and axis_pass)
    partial = sum([count_pass, speed_pass, axis_pass]) / 3.0
    return _component(
        passed=passed,
        progress=partial,
        read=(
            "source384 generation is cheap enough and keeps uniform action-state basin bins"
            if passed
            else "source384 generation or action-state basin bin balance is not ready"
        ),
        close_condition=f"full/q90 selected_count >= {target_per_rule}, raw_seen <= {raw_seen_budget}, and 96/rule basin bins are uniform",
        next_action="rerun source384 with the action-controls-position axis only if counts or bin balance regress",
        blocker="source_generation_not_milestone_ready",
        evidence={"selected_count_by_rule": counts, "raw_seen": raw_seen, "axis": axis_evidence},
    )


def _basin_component(basin: dict[str, Any], *, min_axis_supported_count: int) -> dict[str, Any]:
    decision = basin.get("decision", {}) or {}
    lists = basin.get("lists", []) or []
    min_count = min(
        [
            _safe_int((item.get("coverage", {}) or {}).get("axis_supported_sustained_settle_count"))
            for item in lists
        ]
        or [0]
    )
    passed = bool(decision.get("axis_supported_controllable_basin_ready") and min_count >= int(min_axis_supported_count))
    return _component(
        passed=passed,
        progress=1.0 if passed else min(1.0, min_count / max(1, int(min_axis_supported_count))),
        read=(
            "Pendulum action-state basin semantics are now gated by axis support AND sustained settle"
            if passed
            else "Pendulum basin evidence is still too weak or too broad"
        ),
        close_condition="controllable-basin guard requires action-state axis support and sustained settle together",
        next_action="do not return to transition-only q+v controllability as a quota gate",
        blocker="action_state_basin_semantics_not_closed",
        evidence={"decision": decision, "min_axis_supported_sustained_count": min_count},
    )


def _label_selector_component(label: dict[str, Any], selector: dict[str, Any]) -> dict[str, Any]:
    label_read = label.get("read", {}) or {}
    label_pass = bool(
        label_read.get("generator_cached_label_schema_ready")
        and label_read.get("pendulum_axis_support_populated")
        and label_read.get("pendulum_generator_label_populated")
    )
    selector_rules = selector.get("rules", {}) or {}
    per_rule: dict[str, Any] = {}
    selector_pass = True
    for rule in RULES:
        payload = selector_rules.get(rule, {}) or {}
        selected = payload.get("selected_summary", {}) or {}
        gains = selected.get("strict_feedback_gain_counts", {}) or {}
        repair_effective = _safe_int(selected.get("pendulum_settle_effective_support_count"))
        axis_supported = _safe_int(selected.get("pendulum_axis_supported_sustained_settle_count"))
        strict_feedback = _safe_int(selected.get("strict_feedback_count"))
        ok = (
            bool(payload.get("guard_pass"))
            and bool(payload.get("profile_guard_pass"))
            and strict_feedback >= 20
            and axis_supported >= 8
            and set(str(key) for key in gains) <= {"1.0"}
        )
        selector_pass = selector_pass and ok
        per_rule[rule] = {
            "ok": ok,
            "strict_feedback_count": strict_feedback,
            "strict_feedback_gain_counts": gains,
            "pendulum_axis_supported_sustained_settle_count": axis_supported,
            "pendulum_settle_effective_support_count": repair_effective,
            "guard_pass": bool(payload.get("guard_pass")),
            "profile_guard_pass": bool(payload.get("profile_guard_pass")),
        }
    passed = bool(label_pass and selector_pass)
    return _component(
        passed=passed,
        progress=1.0 if passed else (0.5 if label_pass or selector_pass else 0.0),
        read=(
            "cached labels and selector materialization pass the cheap full/q90 profile gate"
            if passed
            else "cached labels or selector materialization still have an unresolved cheap-path gap"
        ),
        close_condition="cached labels are populated and selector full/q90 guard_pass/profile_guard_pass with gain1 decay feedback",
        next_action="keep Pendulum labels cached; do not recompute full grids inside fit_model",
        blocker="cached_label_or_selector_not_closed",
        evidence={"label_read": label_read, "rules": per_rule},
    )


def _smoke_component(smoke: dict[str, Any], *, min_gap: float) -> dict[str, Any]:
    scopes = smoke.get("scopes", {}) or {}
    per_rule: dict[str, Any] = {}
    rules_pass = True
    for rule in RULES:
        scope = scopes.get(rule, {}) or {}
        gap = (scope.get("settle_gap", {}) or {}).get("gap_active", {}) or {}
        gap_min = _safe_float(gap.get("min"), default=-1.0)
        checks = scope.get("checks", {}) or {}
        ok = bool(scope.get("hard_pass")) and gap_min >= float(min_gap)
        rules_pass = rules_pass and ok
        per_rule[rule] = {
            "ok": ok,
            "gap_min_active": gap_min,
            "hard_pass": bool(scope.get("hard_pass")),
            "checks": checks,
        }
    passed = bool(smoke.get("hard_pass") and rules_pass)
    return _component(
        passed=passed,
        progress=1.0 if passed else 0.5,
        read=(
            "integrated health/diversity smoke passes under the calibrated cheap sustained-gap threshold"
            if passed
            else "integrated smoke still fails or active settle gap is below the cheap threshold"
        ),
        close_condition=f"full/q90 smoke hard_pass and active repair gap min >= {float(min_gap):.4g}",
        next_action="codify this as cheap sustained smoke, not as the old one-step large-gap test",
        blocker="integrated_smoke_not_closed",
        evidence={"rules": per_rule},
    )


def _parity_component(parity: dict[str, Any], *, max_abs_tol: float) -> dict[str, Any]:
    cases = parity.get("cases", []) or []
    per_case = []
    cases_pass = bool(cases)
    for case in cases:
        grad = _safe_float((case.get("update_parity", {}) or {}).get("gradient_diff", {}).get("grad_delta_max_abs"))
        post = _safe_float(
            ((case.get("update_parity", {}) or {}).get("one_train_update_compare", {}) or {})
            .get("post_update_param_diff", {})
            .get("policy_param_max_abs_diff")
        )
        ok = bool(case.get("hard_pass")) and grad <= float(max_abs_tol) and post <= float(max_abs_tol)
        cases_pass = cases_pass and ok
        per_case.append(
            {
                "case": case.get("case"),
                "rule": case.get("rule"),
                "hard_pass": bool(case.get("hard_pass")),
                "repair_count": _safe_int(case.get("selected_pendulum_settle_yield_repair_count")),
                "grad_delta_max_abs": grad,
                "post_policy_param_max_abs_diff": post,
                "ok": ok,
            }
        )
    passed = bool(parity.get("hard_pass") and cases_pass)
    return _component(
        passed=passed,
        progress=1.0 if passed else 0.0,
        read=(
            "latest materialized fixed groups are numerically identical between pack and fit_model"
            if passed
            else "pack-to-fit_model parity is missing or failing for the cheap materialized fixed groups"
        ),
        close_condition="n4/s16 repair-preferred parity passes for full and q90 with zero/near-zero update diff",
        next_action="rerun this parity whenever selector materialization or fit_model bridge changes",
        blocker="pack_fit_parity_missing",
        evidence={"cases": per_case},
    )


def _stability_component(stability: dict[str, Any], *, min_steps: int, min_gap: float) -> dict[str, Any]:
    scopes = stability.get("scopes", {}) or {}
    per_rule: dict[str, Any] = {}
    rules_pass = bool(scopes)
    min_observed_steps: int | None = None
    for rule in RULES:
        scope = scopes.get(rule, {}) or {}
        steps = _safe_int((scope.get("transition_diversity", {}) or {}).get("steps"))
        gap_min = _safe_float(((scope.get("settle_gap", {}) or {}).get("gap_active", {}) or {}).get("min"), -1.0)
        ok = bool(scope.get("hard_pass")) and steps >= int(min_steps) and gap_min >= float(min_gap)
        rules_pass = rules_pass and ok
        min_observed_steps = steps if min_observed_steps is None else min(min_observed_steps, steps)
        per_rule[rule] = {
            "ok": ok,
            "hard_pass": bool(scope.get("hard_pass")),
            "steps": steps,
            "gap_min_active": gap_min,
        }
    passed = bool(stability.get("hard_pass") and rules_pass)
    return _component(
        passed=passed,
        progress=1.0 if passed else 0.0,
        read=(
            "same fixed-group longer cheap smoke passes, so the path is not only a 32-step smoke artifact"
            if passed
            else "latest cheap path still lacks a cross-seed or longer same-fixed-group stability check"
        ),
        close_condition=f"same fixed-group smoke hard_pass with n_steps >= {int(min_steps)} or an equivalent cross-seed source384 pass",
        next_action="a second seed is still useful later, but this longer smoke is enough to close the immediate cheap stability gap",
        blocker="cross_seed_or_longer_stability_missing",
        evidence={"min_observed_steps": min_observed_steps, "rules": per_rule},
    )


def _contract_component(
    contract: dict[str, Any],
    *,
    selector_profile: str,
    min_gap: float,
    min_stability_steps: int,
) -> dict[str, Any]:
    decision = contract.get("decision", {}) or {}
    cheap = contract.get("cheap_contract", {}) or {}
    smoke_contract = cheap.get("sustained_smoke_contract", {}) or {}
    stability_contract = cheap.get("stability_contract", {}) or {}
    contract_profile = str(cheap.get("selector_profile_version_id", ""))
    contract_min_gap = _safe_float(smoke_contract.get("min_gap"), default=-1.0)
    contract_steps = _safe_int(stability_contract.get("same_fixed_group_min_steps"))
    selector_match = bool(contract_profile) and contract_profile == str(selector_profile)
    threshold_match = abs(contract_min_gap - float(min_gap)) <= 1e-12
    stability_match = contract_steps >= int(min_stability_steps)
    passed = bool(
        decision.get("cheap_contract_formalized")
        and selector_match
        and threshold_match
        and stability_match
        and smoke_contract.get("old_one_step_min_gap_0_20_is_not_the_gate") is True
    )
    return _component(
        passed=passed,
        progress=1.0 if passed else (0.55 if "cheap_state_guard" in selector_profile else 0.20),
        read=(
            "cheap_state_guard/min_gap/stability semantics are formalized by the cheap M3 contract"
            if passed
            else "cheap_state_guard works as an experiment, but its threshold/role still needs milestone wording"
        ),
        close_condition="cheap contract artifact formalizes cheap_state_guard, min_gap, cached labels, and stability semantics for the same selector profile",
        next_action="generate/refresh phase2_profile_milestone3_cheap_contract with the current selector and smoke artifacts",
        blocker="cheap_contract_not_formalized",
        evidence={
            "analysis_entry": contract.get("analysis_entry"),
            "cheap_contract_formalized": bool(decision.get("cheap_contract_formalized")),
            "selector_match": selector_match,
            "threshold_match": threshold_match,
            "stability_match": stability_match,
            "selector_profile_version_id": selector_profile,
            "contract_selector_profile_version_id": contract_profile,
            "min_gap": float(min_gap),
            "contract_min_gap": contract_min_gap,
            "min_stability_steps": int(min_stability_steps),
            "contract_stability_steps": contract_steps,
        },
    )


def _top_level_component(top_level: dict[str, Any]) -> dict[str, Any]:
    decision = top_level.get("decision", {}) or {}
    track = ((decision.get("milestone_tracks", {}) or {}).get("cheap_milestone3", {}) or {})
    read = ((top_level.get("diagnostics", {}) or {}).get("cheap_milestone3_preflight_read", {}) or {})
    passed = bool(
        top_level.get("analysis_entry") == "phase2_profile_closure_progress_plan"
        and track.get("consumed_preflight") is True
        and read.get("analysis_entry") == "phase2_profile_milestone3_cheap_closure_preflight"
    )
    return _component(
        passed=passed,
        progress=1.0 if passed else 0.30,
        read=(
            "top-level closure progress plan consumes the cheap M3 preflight as a separate milestone track"
            if passed
            else "evidence is integrated in this preflight but not yet consumed by the top-level closure forecast"
        ),
        close_condition="top-level closure/progress gate consumes this cheap preflight without folding it into broad sufficient coverage",
        next_action="wire this report into phase2_profile_closure_progress_plan as a cheap_milestone3 track",
        blocker="top_level_gate_not_integrated",
        evidence={
            "analysis_entry": top_level.get("analysis_entry"),
            "consumed_preflight": bool(track.get("consumed_preflight")),
            "track_ready": bool(track.get("ready_to_land")),
            "track_hard_blockers": track.get("hard_blockers", []),
            "preflight_read_entry": read.get("analysis_entry"),
        },
    )


def _mountaincar_scope_component(contract: dict[str, Any], label: dict[str, Any]) -> dict[str, Any]:
    decision = contract.get("decision", {}) or {}
    scope = contract.get("mountaincar_scope", {}) or {}
    label_read = label.get("read", {}) or {}
    scope_decision = str(scope.get("decision", ""))
    passed = bool(
        decision.get("mountaincar_scope_decided")
        and scope_decision in {"out_of_scope_for_cheap_m3", "same_pool_labels_and_guards_attached"}
    )
    return _component(
        passed=passed,
        progress=1.0 if passed else 0.50,
        read=(
            "MountainCar scope is explicit for cheap M3"
            if passed
            else "MountainCar future-basin remains coverage telemetry, not a cheap milestone3 quota"
        ),
        close_condition="explicitly mark MountainCar out-of-scope for cheap M3, or attach same-pool cheap labels and guards",
        next_action="refresh the cheap contract with an explicit MountainCar scope decision",
        blocker="mountaincar_scope_not_decided",
        evidence={
            "contract_scope_decision": scope_decision,
            "contract_scope_decided": bool(decision.get("mountaincar_scope_decided")),
            "label_mountaincar_guard_coverage_ready": bool(label_read.get("mountaincar_guard_coverage_ready")),
            "label_mountaincar_quota_ready": bool(label_read.get("mountaincar_quota_ready")),
        },
    )


def build_report(
    *,
    source_json: str | Path,
    basin_json: str | Path,
    label_json: str | Path,
    selector_json: str | Path,
    smoke_json: str | Path,
    stability_smoke_json: str | Path | None,
    parity_json: str | Path,
    cheap_contract_json: str | Path | None = None,
    top_level_closure_json: str | Path | None = None,
    target_per_rule: int = 384,
    raw_seen_budget: int = 16384,
    min_axis_supported_count: int = 96,
    min_gap: float = 0.01,
    min_stability_steps: int = 128,
    parity_abs_tol: float = 1e-9,
    milestone_threshold: float = 0.90,
) -> dict[str, Any]:
    source = _read_json(source_json)
    basin = _read_json(basin_json)
    label = _read_json(label_json)
    selector = _read_json(selector_json)
    smoke = _read_json(smoke_json)
    stability_smoke = _read_json(stability_smoke_json)
    parity = _read_json(parity_json)
    cheap_contract = _read_json(cheap_contract_json)
    top_level_closure = _read_json(top_level_closure_json)

    selector_profile = str((selector.get("config", {}) or {}).get("profile_version_id", ""))
    components = {
        "source_generation": _source_component(
            source,
            target_per_rule=int(target_per_rule),
            raw_seen_budget=int(raw_seen_budget),
        ),
        "action_state_basin_semantics": _basin_component(
            basin,
            min_axis_supported_count=int(min_axis_supported_count),
        ),
        "selector_materialization": _label_selector_component(label, selector),
        "integrated_health_smoke": _smoke_component(smoke, min_gap=float(min_gap)),
        "pack_fit_parity": _parity_component(parity, max_abs_tol=float(parity_abs_tol)),
        "cheap_contract_formalization": _contract_component(
            cheap_contract,
            selector_profile=selector_profile,
            min_gap=float(min_gap),
            min_stability_steps=int(min_stability_steps),
        ),
        "cross_seed_or_longer_stability": _stability_component(
            stability_smoke,
            min_steps=int(min_stability_steps),
            min_gap=float(min_gap),
        ),
        "top_level_gate_integration": _top_level_component(top_level_closure),
        "mountaincar_scope_decision": _mountaincar_scope_component(cheap_contract, label),
    }
    weighted_progress = sum(float(WEIGHTS[key]) * float(components[key]["progress"]) for key in WEIGHTS)
    blockers = [item["blocker"] for item in components.values() if item.get("blocker")]
    hard_blockers = [
        blocker
        for blocker in blockers
        if blocker
        in {
            "cheap_contract_not_formalized",
            "cross_seed_or_longer_stability_missing",
            "top_level_gate_not_integrated",
            "mountaincar_scope_not_decided",
        }
    ]
    ready = bool(weighted_progress >= float(milestone_threshold) and not hard_blockers)
    ordered_gaps = sorted(
        [
            {
                "id": key,
                "weight": float(WEIGHTS[key]),
                "remaining_weighted": float(WEIGHTS[key]) * (1.0 - float(component["progress"])),
                "blocker": component.get("blocker"),
                "next_action": component["next_action"],
            }
            for key, component in components.items()
            if float(component["progress"]) < 1.0
        ],
        key=lambda item: item["remaining_weighted"],
        reverse=True,
    )
    return {
        "analysis_entry": "phase2_profile_milestone3_cheap_closure_preflight",
        "contract": {
            "read_only_existing_artifacts": True,
            "does_not_sample_environments": True,
            "does_not_train": True,
            "does_not_modify_exact_scm": True,
            "does_not_modify_fit_model": True,
            "raw_positive_return_not_a_gate": True,
            "cheap_path_only": True,
        },
        "inputs": {
            "source_json": str(Path(source_json).expanduser().resolve()),
            "basin_json": str(Path(basin_json).expanduser().resolve()),
            "label_json": str(Path(label_json).expanduser().resolve()),
            "selector_json": str(Path(selector_json).expanduser().resolve()),
            "smoke_json": str(Path(smoke_json).expanduser().resolve()),
            "stability_smoke_json": (
                ""
                if stability_smoke_json is None or str(stability_smoke_json).strip() == ""
                else str(Path(stability_smoke_json).expanduser().resolve())
            ),
            "parity_json": str(Path(parity_json).expanduser().resolve()),
            "cheap_contract_json": (
                ""
                if cheap_contract_json is None or str(cheap_contract_json).strip() == ""
                else str(Path(cheap_contract_json).expanduser().resolve())
            ),
            "top_level_closure_json": (
                ""
                if top_level_closure_json is None or str(top_level_closure_json).strip() == ""
                else str(Path(top_level_closure_json).expanduser().resolve())
            ),
        },
        "thresholds": {
            "target_per_rule": int(target_per_rule),
            "raw_seen_budget": int(raw_seen_budget),
            "min_axis_supported_count": int(min_axis_supported_count),
            "min_gap": float(min_gap),
            "min_stability_steps": int(min_stability_steps),
            "parity_abs_tol": float(parity_abs_tol),
            "milestone_threshold": float(milestone_threshold),
        },
        "score": {
            "weighted_progress": weighted_progress,
            "milestone_threshold": float(milestone_threshold),
            "remaining_to_threshold": max(0.0, float(milestone_threshold) - weighted_progress),
            "weights": WEIGHTS,
        },
        "decision": {
            "ready_to_land_milestone3_cheap": ready,
            "blockers": blockers,
            "hard_blockers": hard_blockers,
            "largest_remaining_gaps": ordered_gaps,
            "read": (
                "cheap path can be promoted to milestone3"
                if ready
                else "cheap path is close but still needs formalization/stability/top-level gate integration before milestone3"
            ),
        },
        "components": components,
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Profile Milestone3 Cheap Closure Preflight",
        "",
        f"ready_to_land_milestone3_cheap: `{report['decision']['ready_to_land_milestone3_cheap']}`",
        f"weighted_progress: `{report['score']['weighted_progress']:.3f}`",
        f"remaining_to_threshold: `{report['score']['remaining_to_threshold']:.3f}`",
        "",
        "## Components",
        "",
        "| component | pass | progress | blocker | read |",
        "|---|---:|---:|---|---|",
    ]
    for key, item in report["components"].items():
        blocker = item.get("blocker") or ""
        lines.append(
            f"| `{key}` | `{item['passed']}` | `{item['progress']:.2f}` | `{blocker}` | {item['read']} |"
        )
    lines.extend(["", "## Largest Remaining Gaps", ""])
    for gap in report["decision"]["largest_remaining_gaps"]:
        lines.append(
            f"- `{gap['id']}`: remaining `{gap['remaining_weighted']:.3f}`; next: {gap['next_action']}"
        )
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    report = build_report(
        source_json=args.source_json,
        basin_json=args.basin_json,
        label_json=args.label_json,
        selector_json=args.selector_json,
        smoke_json=args.smoke_json,
        stability_smoke_json=args.stability_smoke_json,
        parity_json=args.parity_json,
        cheap_contract_json=args.cheap_contract_json,
        top_level_closure_json=args.top_level_closure_json,
        target_per_rule=int(args.target_per_rule),
        raw_seen_budget=int(args.raw_seen_budget),
        min_axis_supported_count=int(args.min_axis_supported_count),
        min_gap=float(args.min_gap),
        min_stability_steps=int(args.min_stability_steps),
        parity_abs_tol=float(args.parity_abs_tol),
        milestone_threshold=float(args.milestone_threshold),
    )
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "profile_milestone3_cheap_closure_preflight.json"
    md_path = out_dir / "profile_milestone3_cheap_closure_preflight.md"
    json_path.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({"ready": report["decision"]["ready_to_land_milestone3_cheap"], "json": str(json_path)}, sort_keys=True))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-json", default=str(DEFAULT_SOURCE_JSON))
    parser.add_argument("--basin-json", default=str(DEFAULT_BASIN_JSON))
    parser.add_argument("--label-json", default=str(DEFAULT_LABEL_JSON))
    parser.add_argument("--selector-json", default=str(DEFAULT_SELECTOR_JSON))
    parser.add_argument("--smoke-json", default=str(DEFAULT_SMOKE_JSON))
    parser.add_argument("--stability-smoke-json", default=str(DEFAULT_STABILITY_SMOKE_JSON))
    parser.add_argument("--parity-json", default=str(DEFAULT_PARITY_JSON))
    parser.add_argument("--cheap-contract-json", default=str(DEFAULT_CHEAP_CONTRACT_JSON))
    parser.add_argument("--top-level-closure-json", default=str(DEFAULT_TOP_LEVEL_CLOSURE_JSON))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--target-per-rule", type=int, default=384)
    parser.add_argument("--raw-seen-budget", type=int, default=16384)
    parser.add_argument("--min-axis-supported-count", type=int, default=96)
    parser.add_argument("--min-gap", type=float, default=0.01)
    parser.add_argument("--min-stability-steps", type=int, default=128)
    parser.add_argument("--parity-abs-tol", type=float, default=1e-9)
    parser.add_argument("--milestone-threshold", type=float, default=0.90)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
