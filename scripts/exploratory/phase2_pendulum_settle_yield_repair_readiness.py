#!/usr/bin/env python
"""Readiness gate for repairing Pendulum settle generator yield.

This script is deliberately read-only.  It consolidates existing artifacts to
answer one narrow question: is the current Exact-SCM milestone2 generator close
enough to yield Pendulum sustained closed-loop settle cases, or do we need a
prior-derived yield repair inside the existing generator family?

The split is:

* current Exact-SCM/state-label preflights: measure real generator yield;
* knob sweeps: falsify cheap global parameter tweaks when they do not repair
  yield;
* quadratic settle specimen: validate the missing structural shape outside
  Exact SCM, as a diagnostic only;
* cheap guard calibration: decide whether default startup can use a cheap or
  cached label rather than the full offline temporal grid;
* profile selector report: confirm cached labels do not break broad profile
  bands.

It does not sample environments, train PPO, or modify fit_model/Exact SCM.  The
standalone specimen is not allowed to enter the prior generator path directly.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import re
from pathlib import Path
from typing import Any


ART = Path("/home/chen/RLPFN/artifacts")
DEFAULT_OUTPUT_DIR = ART / "phase2_pendulum_settle_yield_repair_readiness_0512"
DEFAULT_CURRENT_PREFLIGHT_JSON = (
    ART
    / "phase2_pendulum_settle_state_label_seed9511_gpu0_0512"
    / "pendulum_settle_state_label_preflight.json"
)
DEFAULT_PREFLIGHT_GLOB = str(
    ART
    / "phase2_pendulum_settle_state_label_*_0512"
    / "pendulum_settle_state_label_preflight.json"
)
DEFAULT_BLOCKER_GLOB = str(
    ART
    / "phase2_pendulum_settle_state_blocker*_0512"
    / "pendulum_settle_state_blocker_decomposition.json"
)
DEFAULT_SPECIMEN_JSON = (
    ART
    / "phase2_pendulum_quadratic_settle_specimen_preflight_0512"
    / "pendulum_quadratic_settle_specimen_preflight.json"
)
DEFAULT_CALIBRATION_GLOB = str(
    ART
    / "phase2_pendulum_quadratic_settle_cheap_guard_calibration_0512"
    / "seed_*"
    / "pendulum_quadratic_settle_cheap_guard_calibration.json"
)
DEFAULT_SELECTOR_JSON = (
    ART
    / "phase2_profile_band_selector_seed9511_cached_future_basin_0512"
    / "profile_band_selector_report.json"
)
DEFAULT_CLOSURE_JSON = (
    ART
    / "phase2_profile_closure_progress_plan_cached_future_basin_seed9511_0512"
    / "profile_closure_progress_plan.json"
)


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coverage_summary(preflight: dict[str, Any]) -> dict[str, Any]:
    lists = []
    min_guard_count: int | None = None
    min_guard_ratio: float | None = None
    for item in preflight.get("lists", []) or []:
        coverage = ((item.get("classification") or {}).get("coverage") or {})
        guard_count = _as_int(coverage.get("cheap_settle_guard_count"))
        guard_ratio = _as_float(coverage.get("cheap_settle_guard_ratio"))
        min_guard_count = guard_count if min_guard_count is None else min(min_guard_count, guard_count)
        min_guard_ratio = guard_ratio if min_guard_ratio is None else min(min_guard_ratio, guard_ratio)
        lists.append(
            {
                "label": str(item.get("label", "")),
                "n_envs": _as_int(coverage.get("n_envs")),
                "cheap_settle_guard_count": guard_count,
                "cheap_settle_guard_ratio": guard_ratio,
                "reward_condition_count": _as_int(coverage.get("reward_condition_count")),
                "suffix_reward_condition_count": _as_int(coverage.get("suffix_reward_condition_count")),
                "action_decay_condition_count": _as_int(coverage.get("action_decay_condition_count")),
                "self_state_settle_condition_count": _as_int(
                    coverage.get("self_state_settle_condition_count")
                ),
                "open_state_settle_condition_count": _as_int(
                    coverage.get("open_state_settle_condition_count")
                ),
            }
        )
    decision = preflight.get("decision", {}) or {}
    return {
        "ready": bool(decision.get("cheap_suffix_state_settle_guard_ready")),
        "decision_min_count": _as_int(decision.get("min_count")),
        "decision_min_ratio": _as_float(decision.get("min_ratio")),
        "required_min_count": _as_int(decision.get("required_min_count")),
        "required_min_ratio": _as_float(decision.get("required_min_ratio")),
        "observed_min_guard_count": int(min_guard_count or 0),
        "observed_min_guard_ratio": float(min_guard_ratio or 0.0),
        "lists": lists,
    }


def _trial_name(path: str | Path) -> str:
    return Path(path).expanduser().resolve().parent.name


def _blockers_by_trial(paths: list[str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for path in paths:
        payload = _read_json(path)
        parent = _trial_name(path)
        out[parent] = payload
        # Common paired naming:
        # phase2_pendulum_settle_state_blocker_state_output_scale_085...
        # phase2_pendulum_settle_state_label_state_output_scale_085...
        out[parent.replace("phase2_pendulum_settle_state_blocker", "phase2_pendulum_settle_state_label")] = payload
    return out


def _knob_trial_summaries(preflight_paths: list[str], blocker_paths: list[str]) -> list[dict[str, Any]]:
    blockers = _blockers_by_trial(blocker_paths)
    rows = []
    for path in sorted(preflight_paths):
        payload = _read_json(path)
        coverage = _coverage_summary(payload)
        name = _trial_name(path)
        blocker = blockers.get(name, {})
        blocker_decision = blocker.get("decision", {}) or {}
        rows.append(
            {
                "trial": name,
                "path": str(Path(path).expanduser().resolve()),
                "h_overrides": (payload.get("config", {}) or {}).get("h_overrides", {}),
                "ready": bool(coverage["ready"]),
                "observed_min_guard_count": int(coverage["observed_min_guard_count"]),
                "observed_min_guard_ratio": float(coverage["observed_min_guard_ratio"]),
                "required_min_count": int(coverage["required_min_count"]),
                "required_min_ratio": float(coverage["required_min_ratio"]),
                "dominant_blocker_is_state_basin": bool(
                    blocker_decision.get("dominant_blocker_is_state_basin")
                ),
                "selector_threshold_repair_allowed": bool(
                    blocker_decision.get("selector_threshold_repair_allowed")
                ),
                "strict_contraction_count_safe_under_grid": bool(
                    blocker_decision.get("strict_contraction_count_safe_under_grid")
                ),
            }
        )
    return rows


def _specimen_summary(payload: dict[str, Any]) -> dict[str, Any]:
    aggregate = payload.get("aggregate", {}) or {}
    decision = payload.get("decision", {}) or {}
    ratios = aggregate.get("condition_ratios", {}) or {}
    counts = aggregate.get("condition_counts", {}) or {}
    return {
        "specimen_preflight_pass": bool(decision.get("specimen_preflight_pass")),
        "count_safe": bool(aggregate.get("count_safe")),
        "diversity_pass": bool(aggregate.get("diversity_pass")),
        "n_envs": _as_int(aggregate.get("n_envs")),
        "required_min_guard_count": _as_int(aggregate.get("required_min_guard_count")),
        "required_min_guard_ratio": _as_float(aggregate.get("required_min_guard_ratio")),
        "guard_pass_count": _as_int(counts.get("guard_pass")),
        "guard_pass_ratio": _as_float(ratios.get("guard_pass")),
        "settle_pass_ratio": _as_float(ratios.get("settle_pass")),
        "action_decay_pass_ratio": _as_float(ratios.get("action_decay_pass")),
        "action_sensitive_pass_ratio": _as_float(ratios.get("action_sensitive_pass")),
        "mechanism": payload.get("mechanism", {}),
    }


def _calibration_summary(paths: list[str]) -> dict[str, Any]:
    rows = []
    for path in sorted(paths):
        payload = _read_json(path)
        static = payload.get("static_analytic_best", {}) or {}
        short = payload.get("short_temporal_best", {}) or {}
        decision = payload.get("decision", {}) or {}
        parent_name = Path(path).expanduser().resolve().parent.name
        match = re.search(r"_v(\d+)$", parent_name)
        protocol_version = int(match.group(1)) if match else 1
        rows.append(
            {
                "path": str(Path(path).expanduser().resolve()),
                "protocol_version": protocol_version,
                "cheap_guard_viable": bool(decision.get("cheap_guard_viable")),
                "short_temporal_guard_viable": bool(decision.get("short_temporal_guard_viable")),
                "static_precision": _as_float(static.get("precision")),
                "static_recall": _as_float(static.get("recall")),
                "static_predicted_ratio": _as_float(static.get("predicted_ratio")),
                "short_precision": _as_float(short.get("precision")),
                "short_recall": _as_float(short.get("recall")),
                "short_predicted_ratio": _as_float(short.get("predicted_ratio")),
                "short_truth_ratio": _as_float(short.get("truth_ratio")),
                "short_f1": _as_float(short.get("f1")),
            }
        )
    if not rows:
        return {
            "n_runs": 0,
            "selected_protocol_version": None,
            "static_analytic_guard_viable": False,
            "short_temporal_cached_label_viable": False,
            "runs": [],
        }
    selected_protocol_version = max(int(row["protocol_version"]) for row in rows)
    selected_rows = [row for row in rows if int(row["protocol_version"]) == selected_protocol_version]
    short_precisions = [
        row["short_precision"] for row in selected_rows if row["short_temporal_guard_viable"]
    ]
    short_predicted = [
        row["short_predicted_ratio"] for row in selected_rows if row["short_temporal_guard_viable"]
    ]
    best_short = max(selected_rows, key=lambda row: row["short_f1"])
    return {
        "n_runs": len(rows),
        "selected_protocol_version": int(selected_protocol_version),
        "selected_protocol_run_count": int(len(selected_rows)),
        "static_analytic_guard_viable": any(row["cheap_guard_viable"] for row in selected_rows),
        "short_temporal_cached_label_viable": bool(
            short_precisions
            and min(short_precisions) >= 0.80
            and min(short_predicted or [0.0]) >= 0.20
            and all(row["short_temporal_guard_viable"] for row in selected_rows)
        ),
        "short_min_precision": float(min(short_precisions or [0.0])),
        "short_min_predicted_ratio": float(min(short_predicted or [0.0])),
        "best_short_run": best_short,
        "selected_protocol_runs": selected_rows,
        "runs": rows,
    }


def _selector_summary(payload: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for rule, data in (payload.get("rules", {}) or {}).items():
        selected = data.get("selected_summary", {}) or {}
        out[rule] = {
            "profile_guard_pass": bool(data.get("profile_guard_pass")),
            "n": _as_int(selected.get("n")),
            "joint_profile_active_cells": _as_int(selected.get("joint_profile_active_cells")),
            "joint_profile_dominant_rate": _as_float(selected.get("joint_profile_dominant_rate")),
            "joint_profile_entropy_norm": _as_float(selected.get("joint_profile_entropy_norm")),
            "pendulum_label_available_rate": _as_float(selected.get("pendulum_label_available_rate")),
            "pendulum_short_temporal_settle_guard_count": _as_int(
                selected.get("pendulum_short_temporal_settle_guard_count")
            ),
            "pendulum_short_temporal_settle_guard_rate": _as_float(
                selected.get("pendulum_short_temporal_settle_guard_rate")
            ),
            "mountaincar_short_future_basin_guard_rate": _as_float(
                selected.get("mountaincar_short_future_basin_guard_rate")
            ),
        }
    return out


def build_report(
    *,
    current_preflight_json: str | Path,
    preflight_paths: list[str],
    blocker_paths: list[str],
    specimen_json: str | Path,
    calibration_paths: list[str],
    selector_json: str | Path,
    closure_json: str | Path,
) -> dict[str, Any]:
    current = _coverage_summary(_read_json(current_preflight_json))
    knob_trials = _knob_trial_summaries(preflight_paths, blocker_paths)
    specimen = _specimen_summary(_read_json(specimen_json))
    calibration = _calibration_summary(calibration_paths)
    selector = _selector_summary(_read_json(selector_json))
    closure = _read_json(closure_json)

    best_trial = max(
        knob_trials,
        key=lambda row: (row["observed_min_guard_count"], row["observed_min_guard_ratio"]),
        default=None,
    )
    existing_knobs_falsified = bool(
        knob_trials
        and best_trial is not None
        and best_trial["observed_min_guard_count"] < max(1, best_trial["required_min_count"])
        and not any(row["selector_threshold_repair_allowed"] for row in knob_trials)
    )
    broad_profile_still_guarded = bool(
        selector and all(item["profile_guard_pass"] for item in selector.values())
    )
    quadratic_mechanism_viable = bool(
        specimen["specimen_preflight_pass"] and specimen["count_safe"] and specimen["diversity_pass"]
    )
    ready_for_prior_derived_repair_probe = bool(
        existing_knobs_falsified
        and quadratic_mechanism_viable
        and calibration["short_temporal_cached_label_viable"]
        and broad_profile_still_guarded
    )
    standalone_specimen_integration_allowed = False

    remaining = []
    if not existing_knobs_falsified:
        remaining.append("finish falsifying cheap Exact-SCM knobs before adding a new mechanism")
    if not quadratic_mechanism_viable:
        remaining.append("repair the quadratic settle specimen before any generator-level integration")
    if not calibration["short_temporal_cached_label_viable"]:
        remaining.append("make the short temporal cached label precise enough for startup use")
    if not broad_profile_still_guarded:
        remaining.append("repair profile band selector guard after cached labels are merged")
    remaining.extend(
        [
            "derive an equivalent settle-yield repair from exported prior h or existing Exact-SCM parameters",
            "keep any standalone quadratic specimen as an offline diagnostic only",
            "route fit_model startup through cached cheap labels, not the full feedback grid",
            "run integrated full/q90 health/diversity smoke and pack-to-fit_model parity before milestone promotion",
        ]
    )

    return {
        "analysis_entry": "phase2_pendulum_settle_yield_repair_readiness",
        "schema": "phase2_pendulum_settle_yield_repair_readiness.v1",
        "contract": {
            "read_only_existing_artifacts": True,
            "does_not_sample_environments": True,
            "does_not_train": True,
            "does_not_modify_exact_scm": True,
            "does_not_modify_fit_model": True,
            "does_not_modify_ppo": True,
            "not_a_raw_delta_gate": True,
            "profile_quality_first": True,
            "standalone_specimen_not_allowed_as_generator": True,
        },
        "inputs": {
            "current_preflight_json": str(Path(current_preflight_json).expanduser().resolve()),
            "preflight_count": len(preflight_paths),
            "blocker_count": len(blocker_paths),
            "specimen_json": str(Path(specimen_json).expanduser().resolve()),
            "calibration_count": len(calibration_paths),
            "selector_json": str(Path(selector_json).expanduser().resolve()),
            "closure_json": str(Path(closure_json).expanduser().resolve()),
        },
        "current_exact_scm_yield": current,
        "existing_knob_trials": {
            "best_trial": best_trial,
            "all_trials": knob_trials,
            "existing_knobs_falsified": existing_knobs_falsified,
            "read": (
                "Global alpha/highway/inertia/state-output-scale style knobs did not repair count-safe "
                "Pendulum settle yield; the dominant blocker remains state-basin settle."
                if existing_knobs_falsified
                else "Existing knobs are not yet fully falsified."
            ),
        },
        "quadratic_settle_mechanism": {
            **specimen,
            "structural_mechanism_viable": quadratic_mechanism_viable,
            "standalone_specimen_integration_allowed": standalone_specimen_integration_allowed,
            "read": (
                "A small controllable quadratic settle factor can generate the missing sustained "
                "closed-loop settle family with diversity; because it is independent of the "
                "prior generator, it is diagnostic only and must not be plugged in directly."
                if quadratic_mechanism_viable
                else "The simplified settle factor is not yet structurally viable."
            ),
        },
        "cheap_or_cached_guard": {
            **calibration,
            "read": (
                "Static analytic guard is not reliable enough; short temporal cached label is the viable "
                "startup-compatible guard."
                if calibration["short_temporal_cached_label_viable"]
                else "No startup-compatible guard is ready."
            ),
        },
        "profile_selector_after_cached_labels": {
            "broad_profile_still_guarded": broad_profile_still_guarded,
            "rules": selector,
        },
        "closure_context": {
            "weighted_progress": _as_float(closure.get("progress", {}).get("weighted_progress")),
            "largest_remaining_blocker": (closure.get("decision", {}) or {}).get("largest_remaining_blocker"),
            "second_blocker": (closure.get("decision", {}) or {}).get("second_blocker"),
        },
        "decision": {
            "ready_for_generator_level_prototype": False,
            "ready_for_prior_derived_repair_probe": ready_for_prior_derived_repair_probe,
            "third_milestone_ready": False,
            "why": (
                "The standalone quadratic specimen is diagnostic only. The evidence is sufficient "
                "to search for an equivalent prior-derived repair, but not to plug in an independent "
                "generator or call this a generator-level prototype."
                if ready_for_prior_derived_repair_probe
                else "The evidence is not yet enough even for a prior-derived repair probe."
            ),
            "remaining_before_sufficient_and_uniform": remaining,
        },
    }


def _write_md(report: dict[str, Any], path: Path) -> None:
    decision = report["decision"]
    lines = [
        "# Pendulum Settle Yield Repair Readiness",
        "",
        f"- ready for generator prototype: `{decision['ready_for_generator_level_prototype']}`",
        f"- ready for prior-derived repair probe: `{decision['ready_for_prior_derived_repair_probe']}`",
        f"- third milestone ready: `{decision['third_milestone_ready']}`",
        f"- why: {decision['why']}",
        "",
        "## Current Exact-SCM Yield",
    ]
    current = report["current_exact_scm_yield"]
    lines.append(
        f"- min guard count: `{current['observed_min_guard_count']}` / required `{current['required_min_count']}`"
    )
    for item in current["lists"]:
        lines.append(
            "- {label}: guard `{guard}`, reward/suffix/action/self/open "
            "`{reward}/{suffix}/{action}/{self_state}/{open_state}`".format(
                label=item["label"],
                guard=item["cheap_settle_guard_count"],
                reward=item["reward_condition_count"],
                suffix=item["suffix_reward_condition_count"],
                action=item["action_decay_condition_count"],
                self_state=item["self_state_settle_condition_count"],
                open_state=item["open_state_settle_condition_count"],
            )
        )
    best = report["existing_knob_trials"]["best_trial"]
    if best:
        lines.extend(
            [
                "",
                "## Existing Knobs",
                f"- falsified: `{report['existing_knob_trials']['existing_knobs_falsified']}`",
                f"- best trial: `{best['trial']}` with min guard count `{best['observed_min_guard_count']}`",
                f"- read: {report['existing_knob_trials']['read']}",
            ]
        )
    mechanism = report["quadratic_settle_mechanism"]
    lines.extend(
        [
            "",
            "## Minimal Mechanism",
            f"- structural viable: `{mechanism['structural_mechanism_viable']}`",
            f"- guard pass: `{mechanism['guard_pass_count']}/{mechanism['n_envs']}` "
            f"({mechanism['guard_pass_ratio']:.3f})",
            f"- diversity pass: `{mechanism['diversity_pass']}`",
            f"- standalone specimen integration allowed: `{mechanism['standalone_specimen_integration_allowed']}`",
            f"- read: {mechanism['read']}",
            "",
            "## Guard Strategy",
        ]
    )
    guard = report["cheap_or_cached_guard"]
    lines.append(f"- selected calibration protocol: `v{guard['selected_protocol_version']}`")
    lines.append(f"- static analytic viable: `{guard['static_analytic_guard_viable']}`")
    lines.append(f"- short temporal cached viable: `{guard['short_temporal_cached_label_viable']}`")
    lines.append(f"- short min precision: `{guard['short_min_precision']:.3f}`")
    lines.append(f"- short min predicted ratio: `{guard['short_min_predicted_ratio']:.3f}`")
    lines.append(f"- read: {guard['read']}")
    lines.extend(["", "## Remaining"])
    lines.extend(f"- {item}" for item in decision["remaining_before_sufficient_and_uniform"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current-preflight-json", default=str(DEFAULT_CURRENT_PREFLIGHT_JSON))
    parser.add_argument("--preflight-glob", default=DEFAULT_PREFLIGHT_GLOB)
    parser.add_argument("--blocker-glob", default=DEFAULT_BLOCKER_GLOB)
    parser.add_argument("--specimen-json", default=str(DEFAULT_SPECIMEN_JSON))
    parser.add_argument("--calibration-glob", default=DEFAULT_CALIBRATION_GLOB)
    parser.add_argument("--selector-json", default=str(DEFAULT_SELECTOR_JSON))
    parser.add_argument("--closure-json", default=str(DEFAULT_CLOSURE_JSON))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(
        current_preflight_json=args.current_preflight_json,
        preflight_paths=sorted(glob.glob(args.preflight_glob)),
        blocker_paths=sorted(glob.glob(args.blocker_glob)),
        specimen_json=args.specimen_json,
        calibration_paths=sorted(glob.glob(args.calibration_glob)),
        selector_json=args.selector_json,
        closure_json=args.closure_json,
    )
    json_path = out_dir / "pendulum_settle_yield_repair_readiness.json"
    md_path = out_dir / "pendulum_settle_yield_repair_readiness.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_md(report, md_path)
    print(str(json_path))
    print(str(md_path))


if __name__ == "__main__":
    main()
