#!/usr/bin/env python
"""Formalize the narrow cheap milestone3 contract.

This report is intentionally small and read-only over existing evidence.  It
does not claim broad Gym/profile coverage.  Its job is to make the cheap path
auditable before the top-level progress gate consumes it:

* cheap_state_guard is the selector contract,
* the sustained active settle gap is calibrated at min_gap, not the old one-step
  large-gap smoke,
* Pendulum labels are cached/materialized before fit_model parity,
* MountainCar is explicitly coverage telemetry for this cheap milestone unless
  same-pool labels and guards are attached later.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


ART = Path("/home/chen/RLPFN/artifacts")
RULES = ("gym_full_obs_action_range", "gym_q90_obs_action_range")

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
DEFAULT_OUTPUT_DIR = ART / "phase2_profile_milestone3_cheap_contract_0513"


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


def _smoke_read(smoke: dict[str, Any], *, min_gap: float) -> tuple[bool, dict[str, Any]]:
    scopes = smoke.get("scopes", {}) or {}
    per_rule: dict[str, Any] = {}
    passed = bool(smoke.get("hard_pass")) and bool(scopes)
    for rule in RULES:
        scope = scopes.get(rule, {}) or {}
        gap_min = _safe_float(((scope.get("settle_gap", {}) or {}).get("gap_active", {}) or {}).get("min"), -1.0)
        ok = bool(scope.get("hard_pass")) and gap_min >= float(min_gap)
        passed = passed and ok
        per_rule[rule] = {
            "hard_pass": bool(scope.get("hard_pass")),
            "gap_min_active": gap_min,
            "ok": ok,
        }
    return passed, per_rule


def _stability_read(
    stability_smoke: dict[str, Any],
    *,
    min_gap: float,
    min_stability_steps: int,
) -> tuple[bool, dict[str, Any]]:
    scopes = stability_smoke.get("scopes", {}) or {}
    per_rule: dict[str, Any] = {}
    passed = bool(stability_smoke.get("hard_pass")) and bool(scopes)
    for rule in RULES:
        scope = scopes.get(rule, {}) or {}
        steps = _safe_int((scope.get("transition_diversity", {}) or {}).get("steps"))
        gap_min = _safe_float(((scope.get("settle_gap", {}) or {}).get("gap_active", {}) or {}).get("min"), -1.0)
        ok = bool(scope.get("hard_pass")) and steps >= int(min_stability_steps) and gap_min >= float(min_gap)
        passed = passed and ok
        per_rule[rule] = {
            "hard_pass": bool(scope.get("hard_pass")),
            "steps": steps,
            "gap_min_active": gap_min,
            "ok": ok,
        }
    return passed, per_rule


def build_report(
    *,
    label_json: str | Path,
    selector_json: str | Path,
    smoke_json: str | Path,
    stability_smoke_json: str | Path,
    min_gap: float = 0.01,
    min_stability_steps: int = 128,
    mountaincar_scope: str = "out_of_scope_for_cheap_m3",
) -> dict[str, Any]:
    label = _read_json(label_json)
    selector = _read_json(selector_json)
    smoke = _read_json(smoke_json)
    stability = _read_json(stability_smoke_json)

    label_read = label.get("read", {}) or {}
    selector_profile = str((selector.get("config", {}) or {}).get("profile_version_id", ""))
    selector_contract_ok = "cheap_state_guard" in selector_profile
    cached_label_ok = bool(
        label_read.get("generator_cached_label_schema_ready")
        and label_read.get("pendulum_axis_support_populated")
        and label_read.get("pendulum_generator_label_populated")
    )
    sustained_smoke_ok, sustained_smoke_read = _smoke_read(smoke, min_gap=float(min_gap))
    stability_ok, stability_read = _stability_read(
        stability,
        min_gap=float(min_gap),
        min_stability_steps=int(min_stability_steps),
    )
    mountaincar_scope_decided = str(mountaincar_scope) in {
        "out_of_scope_for_cheap_m3",
        "same_pool_labels_and_guards_attached",
    }
    mountaincar_quota_ready = bool(label_read.get("mountaincar_quota_ready"))
    mountaincar_ok = (
        mountaincar_scope == "same_pool_labels_and_guards_attached" and mountaincar_quota_ready
    ) or mountaincar_scope == "out_of_scope_for_cheap_m3"

    cheap_contract_formalized = bool(
        selector_contract_ok
        and cached_label_ok
        and sustained_smoke_ok
        and stability_ok
        and mountaincar_scope_decided
        and mountaincar_ok
    )
    blockers: list[str] = []
    if not selector_contract_ok:
        blockers.append("cheap_state_guard_selector_contract_missing")
    if not cached_label_ok:
        blockers.append("cached_pendulum_label_policy_not_ready")
    if not sustained_smoke_ok:
        blockers.append("cheap_sustained_gap_calibration_not_supported")
    if not stability_ok:
        blockers.append("cheap_stability_requirement_not_supported")
    if not mountaincar_scope_decided or not mountaincar_ok:
        blockers.append("mountaincar_scope_not_decided")

    return {
        "analysis_entry": "phase2_profile_milestone3_cheap_contract",
        "contract": {
            "read_only_existing_artifacts": True,
            "does_not_sample_environments": True,
            "does_not_train": True,
            "does_not_modify_exact_scm": True,
            "does_not_modify_fit_model": True,
            "does_not_modify_ppo": True,
            "cheap_path_only": True,
            "raw_positive_return_not_a_gate": True,
        },
        "inputs": {
            "label_json": str(Path(label_json).expanduser().resolve()),
            "selector_json": str(Path(selector_json).expanduser().resolve()),
            "smoke_json": str(Path(smoke_json).expanduser().resolve()),
            "stability_smoke_json": str(Path(stability_smoke_json).expanduser().resolve()),
        },
        "cheap_contract": {
            "selector_contract": "cheap_state_guard",
            "selector_profile_version_id": selector_profile,
            "cached_label_policy": "Pendulum axis/support/sustained labels must be materialized before fit_model parity; fit_model must not recompute full rollout grids for this gate.",
            "sustained_smoke_contract": {
                "min_gap": float(min_gap),
                "old_one_step_min_gap_0_20_is_not_the_gate": True,
                "requires_full_and_q90_hard_pass": True,
            },
            "stability_contract": {
                "same_fixed_group_min_steps": int(min_stability_steps),
                "equivalent_cross_seed_source384_smoke_allowed": True,
            },
        },
        "mountaincar_scope": {
            "decision": str(mountaincar_scope),
            "cheap_m3_quota_required": mountaincar_scope == "same_pool_labels_and_guards_attached",
            "current_label_read": {
                "mountaincar_guard_coverage_ready": bool(label_read.get("mountaincar_guard_coverage_ready")),
                "mountaincar_quota_ready": mountaincar_quota_ready,
            },
            "rationale": (
                "For cheap M3, MountainCar remains coverage telemetry because the current cached adapter has no "
                "same-pool quota-ready labels.  It can enter a later milestone only after same-pool cheap labels "
                "and health/diversity guards are attached."
            ),
        },
        "evidence": {
            "selector_contract_ok": selector_contract_ok,
            "cached_label_ok": cached_label_ok,
            "sustained_smoke_ok": sustained_smoke_ok,
            "sustained_smoke_read": sustained_smoke_read,
            "stability_ok": stability_ok,
            "stability_read": stability_read,
        },
        "decision": {
            "cheap_contract_formalized": cheap_contract_formalized,
            "mountaincar_scope_decided": bool(mountaincar_scope_decided and mountaincar_ok),
            "blockers": blockers,
            "read": (
                "cheap milestone3 contract is formalized"
                if cheap_contract_formalized
                else "cheap milestone3 contract still lacks one or more evidence-backed clauses"
            ),
        },
    }


def _markdown(report: dict[str, Any]) -> str:
    decision = report["decision"]
    lines = [
        "# Profile Milestone3 Cheap Contract",
        "",
        f"cheap_contract_formalized: `{decision['cheap_contract_formalized']}`",
        f"mountaincar_scope_decided: `{decision['mountaincar_scope_decided']}`",
        f"blockers: `{decision['blockers']}`",
        "",
        "## Contract",
        "",
        f"- selector: `{report['cheap_contract']['selector_contract']}`",
        f"- min_gap: `{report['cheap_contract']['sustained_smoke_contract']['min_gap']}`",
        f"- stability steps: `{report['cheap_contract']['stability_contract']['same_fixed_group_min_steps']}`",
        f"- MountainCar scope: `{report['mountaincar_scope']['decision']}`",
        "",
        "## Evidence",
        "",
    ]
    for key, value in report["evidence"].items():
        if isinstance(value, bool):
            lines.append(f"- `{key}`: `{value}`")
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    report = build_report(
        label_json=args.label_json,
        selector_json=args.selector_json,
        smoke_json=args.smoke_json,
        stability_smoke_json=args.stability_smoke_json,
        min_gap=float(args.min_gap),
        min_stability_steps=int(args.min_stability_steps),
        mountaincar_scope=args.mountaincar_scope,
    )
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "profile_milestone3_cheap_contract.json"
    md_path = out_dir / "profile_milestone3_cheap_contract.md"
    report["outputs"] = {"json": str(json_path), "markdown": str(md_path)}
    json_path.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({"formalized": report["decision"]["cheap_contract_formalized"], "json": str(json_path)}, sort_keys=True))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-json", default=str(DEFAULT_LABEL_JSON))
    parser.add_argument("--selector-json", default=str(DEFAULT_SELECTOR_JSON))
    parser.add_argument("--smoke-json", default=str(DEFAULT_SMOKE_JSON))
    parser.add_argument("--stability-smoke-json", default=str(DEFAULT_STABILITY_SMOKE_JSON))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--min-gap", type=float, default=0.01)
    parser.add_argument("--min-stability-steps", type=int, default=128)
    parser.add_argument(
        "--mountaincar-scope",
        choices=("out_of_scope_for_cheap_m3", "same_pool_labels_and_guards_attached"),
        default="out_of_scope_for_cheap_m3",
    )
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
