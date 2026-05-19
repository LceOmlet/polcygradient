#!/usr/bin/env python
"""Audit remaining pressure in the future-basin profile mainline.

Read-only consolidation.  It predicts what remains before "sufficient coverage"
without optimizing raw scores or changing any generator/PPO path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ART = Path("/home/chen/RLPFN/artifacts")
DEFAULT_OUTPUT_DIR = ART / "phase2_future_basin_profile_pressure_audit_0512"
DEFAULT_SPECTRUM_JSON = (
    ART
    / "phase2_action_conditioned_future_basin_spectrum_0512"
    / "action_conditioned_future_basin_spectrum.json"
)
DEFAULT_SELECTOR_JSON = (
    ART
    / "phase2_future_basin_spectrum_selector_preflight_0512_v3"
    / "future_basin_spectrum_selector_preflight.json"
)
DEFAULT_MOUNTAINCAR_GYM_ANCHOR_JSON = (
    ART
    / "phase2_gym_mountaincar_future_basin_anchor_0511"
    / "gym_mountaincar_future_basin_anchor_report.json"
)


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def _read_json_optional(path: str | Path | None) -> Any:
    if path is None or str(path).strip() == "":
        return None
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        return None
    return json.loads(resolved.read_text(encoding="utf-8"))


def _row_by_id(spectrum: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row.get("spectrum_id")): row for row in spectrum.get("spectrum_rows", []) or []}


def _f(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _scarcity_pressure(rate: float | None, target: float = 0.20) -> float:
    if rate is None:
        return 0.5
    if rate >= target:
        return 0.0
    return min(1.0, (target - rate) / max(1e-9, target))


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def pressure_rows(
    spectrum: dict[str, Any],
    selector: dict[str, Any],
    mountaincar_gym_anchor: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows = _row_by_id(spectrum)
    missing = set(selector.get("missing_generator_labels", []) or [])

    pend = rows.get("nearby_attractive_closed_loop_settle", {})
    pend_ev = pend.get("prior_evidence", {}) or {}
    pend_ready = bool(pend_ev.get("short_temporal_guard_viable"))
    pend_precision = _f(pend_ev.get("short_temporal_precision"), 0.0) or 0.0
    pend_recall = _f(pend_ev.get("short_temporal_recall"), 0.0) or 0.0
    pend_label_missing = "pendulum_short_temporal_settle_guard" in missing

    mountain = rows.get("delayed_upfront_investment_future_basin", {})
    mountain_ev = mountain.get("prior_evidence", {}) or {}
    delayed_rate = _f(mountain_ev.get("delayed_candidate_rate"))
    candidate_rate = _f(mountain_ev.get("future_basin_candidate_rate"))
    mountain_label_missing = "mountaincar_delayed_future_basin_guard" in missing
    mountain_gym_anchor_pass = bool((mountaincar_gym_anchor or {}).get("read", {}).get("gym_anchor_pass"))

    loco = rows.get("sustained_temporal_locomotion", {})
    loco_ev = loco.get("prior_evidence", {}) or {}
    loco_full = _f(loco_ev.get("full_rate"))
    loco_q90 = _f(loco_ev.get("q90_rate"))

    low = rows.get("low_energy_or_weak_action_optimum", {})
    low_ev = low.get("prior_evidence", {}) or {}
    sign = rows.get("sign_phase_sensitive_control", {})
    sign_ev = sign.get("prior_evidence", {}) or {}
    parent = rows.get("state_conditioned_energy_injection_parent", {})
    parent_ev = parent.get("prior_evidence", {}) or {}

    terminal_label_missing = "terminal_action_sensitive_trap_guard" in missing

    out = [
        {
            "pressure_id": "pendulum_settle_generator_label_gap",
            "spectrum_id": "nearby_attractive_closed_loop_settle",
            "semantic_gap": 0.10 if pend_ready else 0.80,
            "execution_gap": 1.0 if pend_label_missing else 0.10,
            "profile_risk": 0.25,
            "actionability": _clamp(0.5 * pend_precision + 0.5 * pend_recall),
            "score": 0.0,
            "evidence": {
                "short_temporal_guard_viable": pend_ready,
                "short_temporal_precision": pend_precision,
                "short_temporal_recall": pend_recall,
                "generator_label_missing": pend_label_missing,
            },
            "next": "put short-temporal settle label on generator candidate rows or cache it; then rerun selector/smoke",
        },
        {
            "pressure_id": "mountaincar_delayed_future_basin_semantic_gap",
            "spectrum_id": "delayed_upfront_investment_future_basin",
            "semantic_gap": max(_scarcity_pressure(delayed_rate, 0.10), _scarcity_pressure(candidate_rate, 0.15)),
            "execution_gap": 1.0 if mountain_label_missing else 0.25,
            "profile_risk": 0.35,
            "actionability": 0.68 if mountain_gym_anchor_pass else 0.55,
            "score": 0.0,
            "evidence": {
                "delayed_candidate_rate": delayed_rate,
                "future_basin_candidate_rate": candidate_rate,
                "generator_label_missing": mountain_label_missing,
                "gym_anchor_pass": mountain_gym_anchor_pass,
                "quota_ready": bool(mountain_ev.get("quota_ready")),
            },
            "next": (
                "derive a short prior-side future-basin guard on generator candidate rows; "
                "do not quota until health/diversity smoke passes"
                if mountain_gym_anchor_pass
                else "do Gym MountainCar anchoring and derive a short future-basin guard before any quota"
            ),
        },
        {
            "pressure_id": "locomotion_sustained_temporal_scarcity",
            "spectrum_id": "sustained_temporal_locomotion",
            "semantic_gap": max(_scarcity_pressure(loco_full, 0.20), _scarcity_pressure(loco_q90, 0.20)),
            "execution_gap": 0.35,
            "profile_risk": 0.55,
            "actionability": 0.75,
            "score": 0.0,
            "evidence": {"full_rate": loco_full, "q90_rate": loco_q90},
            "next": "only repair under low-energy/sign-phase/state-conditioned band preservation",
        },
        {
            "pressure_id": "broad_parent_modes_are_protection_not_repair",
            "spectrum_id": "low_energy_sign_phase_state_conditioned_parent",
            "semantic_gap": 0.15,
            "execution_gap": 0.10,
            "profile_risk": 0.70,
            "actionability": 0.40,
            "score": 0.0,
            "evidence": {
                "low_energy_full": _f(low_ev.get("full_rate")),
                "low_energy_q90": _f(low_ev.get("q90_rate")),
                "sign_phase_full": _f(sign_ev.get("full_rate")),
                "sign_phase_q90": _f(sign_ev.get("q90_rate")),
                "state_conditioned_full": _f(parent_ev.get("full_rate")),
                "state_conditioned_q90": _f(parent_ev.get("q90_rate")),
            },
            "next": "keep as protection bands; do not increase parent rates as a repair",
        },
        {
            "pressure_id": "terminal_trap_detector_missing",
            "spectrum_id": "terminal_trap_avoidance",
            "semantic_gap": 0.65,
            "execution_gap": 1.0 if terminal_label_missing else 0.25,
            "profile_risk": 0.30,
            "actionability": 0.30,
            "score": 0.0,
            "evidence": {"generator_label_missing": terminal_label_missing},
            "next": "defer unless terminal trap becomes next pressure; do not add global survival shaping",
        },
    ]

    for row in out:
        # Higher score means more important.  Actionability matters because a
        # clear, tested guard should be acted on before speculative repairs.
        row["score"] = _clamp(
            0.35 * row["semantic_gap"]
            + 0.30 * row["execution_gap"]
            + 0.20 * row["actionability"]
            + 0.15 * row["profile_risk"]
        )
    return sorted(out, key=lambda row: row["score"], reverse=True)


def progress_forecast(rows: list[dict[str, Any]], selector: dict[str, Any]) -> dict[str, Any]:
    fully_ready = bool((selector.get("read", {}) or {}).get("fully_unified_sampling_ready"))
    selector_ready = bool((selector.get("read", {}) or {}).get("selector_mechanics_ready"))
    max_score = max((row["score"] for row in rows), default=1.0)
    return {
        "profile_vocabulary": 1.0,
        "selector_mechanics": 0.75 if selector_ready else 0.25,
        "generator_level_labels": 0.0 if not fully_ready else 1.0,
        "pendulum_settle_semantics": 0.85,
        "mountaincar_future_basin_semantics": 0.45,
        "locomotion_parent_profile": 0.70,
        "overall_rough_progress": _clamp(0.72 - 0.22 * max_score),
        "largest_pressure_score": max_score,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    spectrum = _read_json(args.spectrum_json)
    selector = _read_json(args.selector_json)
    mountaincar_gym_anchor = _read_json_optional(args.mountaincar_gym_anchor_json)
    rows = pressure_rows(spectrum, selector, mountaincar_gym_anchor)
    forecast = progress_forecast(rows, selector)
    report = {
        "analysis_entry": "phase2_future_basin_profile_pressure_audit",
        "schema": "phase2_future_basin_profile_pressure_audit.v1",
        "contract": {
            "exploratory_only": True,
            "read_only_existing_artifacts": True,
            "does_not_modify_exact_scm": True,
            "does_not_modify_fit_model": True,
            "does_not_modify_ppo": True,
            "not_a_raw_delta_gate": True,
        },
        "pressure_rows": rows,
        "forecast": forecast,
        "read": {
            "largest_pressure": rows[0]["pressure_id"] if rows else None,
            "largest_actionable_pressure": next(
                (row["pressure_id"] for row in rows if row["actionability"] >= 0.7),
                None,
            ),
            "profile_mainline_next": (
                "add generator-level/cached Pendulum short-temporal settle labels first; "
                "in parallel keep MountainCar as semantic scarcity/anchor work, not quota"
            ),
        },
    }
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "future_basin_profile_pressure_audit.json"
    md_path = out_dir / "future_basin_profile_pressure_audit.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(_markdown(report) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output_dir": str(out_dir),
                "largest_pressure": report["read"]["largest_pressure"],
                "largest_actionable_pressure": report["read"]["largest_actionable_pressure"],
                "overall_rough_progress": forecast["overall_rough_progress"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return report


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Future-Basin Profile Pressure Audit",
        "",
        "Read-only pressure audit for the profile mainline. Scores are not raw-return metrics.",
        "",
        "## Forecast",
        "",
    ]
    for key, value in report["forecast"].items():
        lines.append(f"- `{key}`: `{value:.3f}`" if isinstance(value, float) else f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Pressure Rows",
            "",
            "| pressure | score | semantic gap | execution gap | actionability | profile risk | next |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in report["pressure_rows"]:
        lines.append(
            f"| `{row['pressure_id']}` | {row['score']:.3f} | {row['semantic_gap']:.3f} | "
            f"{row['execution_gap']:.3f} | {row['actionability']:.3f} | {row['profile_risk']:.3f} | "
            f"{row['next']} |"
        )
    lines.extend(["", "## Read", "", report["read"]["profile_mainline_next"]])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--spectrum-json", default=str(DEFAULT_SPECTRUM_JSON))
    parser.add_argument("--selector-json", default=str(DEFAULT_SELECTOR_JSON))
    parser.add_argument("--mountaincar-gym-anchor-json", default=str(DEFAULT_MOUNTAINCAR_GYM_ANCHOR_JSON))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
