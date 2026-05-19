#!/usr/bin/env python
"""Build a unified action-conditioned future-basin spectrum report.

This is an exploratory/read-only consolidation.  It does not modify Exact SCM,
fit_model, PPO, or any milestone generator path.

The purpose is to keep Pendulum-like settle, MountainCar-like investment, and
locomotion/sustained-energy modes in one maintainable profile vocabulary:

    action -> future reward-relevant state -> basin -> return

Different Gym environments occupy different regions of that spectrum.  The
report is a profile governance artifact, not a raw-delta optimizer.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ART = Path("/home/chen/RLPFN/artifacts")
DEFAULT_OUTPUT_DIR = ART / "phase2_action_conditioned_future_basin_spectrum_0512"
DEFAULT_UNIFIED_PROFILE_JSON = (
    ART / "phase2_unified_gym_like_mode_profile_gpu0_0510/unified_gym_like_mode_profile_report.json"
)
DEFAULT_MOUNTAINCAR_JSON = (
    ART / "phase2_mountaincar_future_basin_detector_v1_0511/mountaincar_future_basin_detector_report.json"
)
DEFAULT_SETTLE_SPECIMEN_JSON = (
    ART
    / "phase2_pendulum_quadratic_settle_specimen_preflight_0512"
    / "seed_20260516_n2048"
    / "pendulum_quadratic_settle_specimen_preflight.json"
)
DEFAULT_SETTLE_CALIBRATION_JSON = (
    ART
    / "phase2_pendulum_quadratic_settle_cheap_guard_calibration_0512"
    / "seed_20260516_n1024_v3"
    / "pendulum_quadratic_settle_cheap_guard_calibration.json"
)


def _read_json_optional(path: str | Path) -> Any:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        return None
    return json.loads(resolved.read_text(encoding="utf-8"))


def _mode_by_id(unified: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in (unified or {}).get("modes", []) or []:
        out[str(row.get("mode_id"))] = row
    return out


def _float(value: Any, default: float | None = None) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out


def _rate_status(rate: float | None, *, low: float = 0.10, high: float = 0.70) -> str:
    if rate is None:
        return "unknown"
    if rate < low:
        return "scarce"
    if rate > high:
        return "dominant"
    return "covered"


def _pct(value: float | None) -> str:
    return "na" if value is None else f"{100.0 * value:.1f}%"


def _spectrum_key(
    *,
    valence: str,
    horizon: str,
    phase: str,
    feedback: str,
    energy: str,
    reachability: str,
) -> str:
    return "|".join((valence, horizon, phase, feedback, energy, reachability))


def gym_anchors() -> list[dict[str, Any]]:
    """Gym-side anchors, intentionally semantic rather than detector-specific."""

    anchors = [
        {
            "env": "Pendulum-v1",
            "spectrum_key": _spectrum_key(
                valence="attractive",
                horizon="mid",
                phase="settle",
                feedback="phase_sensitive_closed_loop",
                energy="early_then_decay",
                reachability="nearby_basin",
            ),
            "profile": {
                "valence": "attractive",
                "credit_horizon": "mid",
                "control_phase": "settle",
                "feedback": "phase_sensitive_closed_loop",
                "energy": "early_then_decay",
                "reachability": "nearby_basin",
            },
            "why": "official PPO must use state/phase-dependent torque to enter an upright basin, then reduce action.",
        },
        {
            "env": "MountainCarContinuous-v0",
            "spectrum_key": _spectrum_key(
                valence="attractive",
                horizon="delayed",
                phase="invest_then_basin",
                feedback="state_conditioned",
                energy="upfront_investment",
                reachability="distant_basin",
            ),
            "profile": {
                "valence": "attractive",
                "credit_horizon": "delayed",
                "control_phase": "invest_then_basin",
                "feedback": "state_conditioned",
                "energy": "upfront_investment",
                "reachability": "distant_basin",
            },
            "why": "short-term control investment is repaid only after future basin entry.",
        },
        {
            "env": "Ant/HalfCheetah/Swimmer/Walker2d locomotion family",
            "spectrum_key": _spectrum_key(
                valence="attractive",
                horizon="mid",
                phase="sustain",
                feedback="state_conditioned_or_rhythmic",
                energy="sustained_moderate",
                reachability="rolling_basin",
            ),
            "profile": {
                "valence": "attractive",
                "credit_horizon": "mid",
                "control_phase": "sustain",
                "feedback": "state_conditioned_or_rhythmic",
                "energy": "sustained_moderate",
                "reachability": "rolling_basin",
            },
            "why": "forward progress needs temporally sustained, state-conditioned control.",
        },
        {
            "env": "InvertedPendulum/InvertedDoublePendulum survival family",
            "spectrum_key": _spectrum_key(
                valence="trap_avoidance",
                horizon="mid",
                phase="avoid_terminal",
                feedback="closed_loop",
                energy="minimal_corrective",
                reachability="terminal_boundary",
            ),
            "profile": {
                "valence": "trap_avoidance",
                "credit_horizon": "mid",
                "control_phase": "avoid_terminal",
                "feedback": "closed_loop",
                "energy": "minimal_corrective",
                "reachability": "terminal_boundary",
            },
            "why": "the key basin can be a bad terminal basin that action must avoid.",
        },
    ]
    return anchors


def build_report(
    *,
    unified: dict[str, Any] | None,
    mountaincar: dict[str, Any] | None,
    settle_specimen: dict[str, Any] | None,
    settle_calibration: dict[str, Any] | None,
) -> dict[str, Any]:
    modes = _mode_by_id(unified)
    loco = modes.get("locomotion_sustained_temporal_control", {})
    low_energy = modes.get("low_energy_weak_action_optimum", {})
    sign_phase = modes.get("sign_phase_sensitive_action", {})
    state_cond = modes.get("state_conditioned_energy_injection", {})
    feedback = modes.get("context_identifiable_closed_loop_settle_feedback", {})
    terminal = modes.get("terminal_survival_action_sensitive", {})

    mountain_all = (mountaincar or {}).get("all_summary", {}) or {}
    specimen_ratios = ((settle_specimen or {}).get("aggregate", {}) or {}).get("condition_ratios", {}) or {}
    short_best = (settle_calibration or {}).get("short_temporal_best", {}) or {}
    static_best = (settle_calibration or {}).get("static_analytic_best", {}) or {}
    decision = (settle_calibration or {}).get("decision", {}) or {}

    rows = [
        {
            "spectrum_id": "nearby_attractive_closed_loop_settle",
            "gym_anchor": "Pendulum-v1",
            "axis": "attractive | mid | settle | phase-sensitive closed-loop | early-then-decay | nearby basin",
            "prior_evidence": {
                "old_feedback_full_rate": feedback.get("prior_full_rate"),
                "old_feedback_q90_rate": feedback.get("prior_q90_rate"),
                "specimen_guard_ratio": specimen_ratios.get("guard_pass"),
                "short_temporal_precision": short_best.get("precision"),
                "short_temporal_recall": short_best.get("recall"),
                "static_precision": static_best.get("precision"),
                "short_temporal_guard_viable": bool(decision.get("short_temporal_guard_viable")),
            },
            "coverage_read": (
                "mechanism now has a reliable specimen and short-temporal guard; "
                "not yet generator-level or fit_model-startable"
            ),
            "pressure": "high_until_generator_level_selector_exists",
            "next_action": "build generator-level selector branch using short-temporal guard or cached labels; keep full audit offline",
        },
        {
            "spectrum_id": "delayed_upfront_investment_future_basin",
            "gym_anchor": "MountainCarContinuous-v0",
            "axis": "attractive | delayed | invest-then-basin | state-conditioned | upfront investment | distant basin",
            "prior_evidence": {
                "future_basin_candidate_rate": mountain_all.get("mountaincar_future_basin_candidate_rate"),
                "future_basin_strong_rate": mountain_all.get("mountaincar_future_basin_strong_rate"),
                "delayed_candidate_rate": mountain_all.get("mountaincar_delayed_candidate_rate"),
                "upfront_action_decay_rate": mountain_all.get("mountaincar_upfront_action_decay_rate"),
                "quota_ready": bool((mountaincar or {}).get("read", {}).get("quota_ready", False)),
            },
            "coverage_read": "upfront action is common, but delayed future-basin payoff is sparse and not quota-ready",
            "pressure": "high_detector_not_generator_repair_yet",
            "next_action": "anchor with Gym MountainCar and define a cheap future-basin guard before quota",
        },
        {
            "spectrum_id": "sustained_temporal_locomotion",
            "gym_anchor": "Ant/HalfCheetah/Swimmer locomotion family",
            "axis": "attractive | mid | sustain | state-conditioned/rhythmic | sustained moderate | rolling basin",
            "prior_evidence": {
                "full_rate": loco.get("prior_full_rate"),
                "q90_rate": loco.get("prior_q90_rate"),
            },
            "coverage_read": "quantified as scarce in prior, but repair must not collapse low-energy or feedback bins",
            "pressure": "medium_high_after_pendulum_settle_guard",
            "next_action": "use as a profile band only under low-energy/sign-phase/state diversity guards",
        },
        {
            "spectrum_id": "low_energy_or_weak_action_optimum",
            "gym_anchor": "Ant/HalfCheetah/Swimmer low-control cases",
            "axis": "attractive | short/mid | conserve | weak feedback | low energy | nearby/rolling basin",
            "prior_evidence": {
                "full_rate": low_energy.get("prior_full_rate"),
                "q90_rate": low_energy.get("prior_q90_rate"),
            },
            "coverage_read": "not scarce; this is a protection band, not a repair target",
            "pressure": "protect",
            "next_action": "do not increase blindly; ensure future repairs do not lower it",
        },
        {
            "spectrum_id": "sign_phase_sensitive_control",
            "gym_anchor": "Pendulum/Reacher/locomotion phase cases",
            "axis": "mixed | short/mid | corrective | sign/phase sensitive | variable | local basin",
            "prior_evidence": {
                "full_rate": sign_phase.get("prior_full_rate"),
                "q90_rate": sign_phase.get("prior_q90_rate"),
            },
            "coverage_read": "broadly covered but too coarse to explain Pendulum alone",
            "pressure": "protect_and_refine",
            "next_action": "only use as parent tag; rely on settle/invest subprofiles for generator changes",
        },
        {
            "spectrum_id": "state_conditioned_energy_injection_parent",
            "gym_anchor": "Pendulum/MountainCar/locomotion recovery",
            "axis": "mixed | mid/delayed | inject | state-conditioned | variable | future basin",
            "prior_evidence": {
                "full_rate": state_cond.get("prior_full_rate"),
                "q90_rate": state_cond.get("prior_q90_rate"),
            },
            "coverage_read": "parent mode is covered; missing structure is in settle vs investment subtypes",
            "pressure": "parent_tag_only",
            "next_action": "do not repair by increasing parent rate; split into settle/invest/sustain bins",
        },
        {
            "spectrum_id": "terminal_trap_avoidance",
            "gym_anchor": "InvertedPendulum/InvertedDoublePendulum",
            "axis": "trap-avoidance | mid | avoid-terminal | closed-loop | corrective | terminal boundary",
            "prior_evidence": {
                "full_rate": terminal.get("prior_full_rate"),
                "q90_rate": terminal.get("prior_q90_rate"),
            },
            "coverage_read": "same-list detector missing; do not add survival shaping globally",
            "pressure": "missing_detector",
            "next_action": "define action-sensitive terminal guard only if this becomes the next pressure point",
        },
    ]

    remaining = {
        "closed_enough_for_next_candidate_milestone": False,
        "completed": [
            "unified spectrum vocabulary",
            "Pendulum settle specimen",
            "Pendulum short-temporal guard calibration",
            "MountainCar future-basin detector as annotation",
            "locomotion/low-energy/sign-phase parent coverage table",
        ],
        "open": [
            "generator-level selector branch for Pendulum settle using short-temporal guard or cached labels",
            "MountainCar Gym anchor plus cheap future-basin guard before quota",
            "integrated pool64 health/diversity smoke after selector implementation",
            "pack-to-fit_model parity only after generator-level selector is stable",
        ],
        "do_not_do": [
            "do not use pure static analytic settle guard",
            "do not optimize raw positive rate",
            "do not quota MountainCar until anchored and health-checked",
            "do not add global survival reward",
        ],
    }

    return {
        "analysis_entry": "phase2_action_conditioned_future_basin_spectrum",
        "schema": "phase2_action_conditioned_future_basin_spectrum.v1",
        "contract": {
            "exploratory_only": True,
            "read_only_existing_artifacts": True,
            "does_not_modify_exact_scm": True,
            "does_not_modify_fit_model": True,
            "does_not_modify_ppo": True,
            "not_a_raw_delta_gate": True,
        },
        "definition": {
            "core": "action changes future reward-relevant state; future state enters or avoids a basin; return depends on that basin.",
            "axes": [
                "basin_valence: attractive / trap-avoidance / mixed",
                "credit_horizon: immediate / mid / delayed",
                "control_phase: settle / invest / invest-then-basin / sustain / avoid-terminal",
                "feedback_dependence: open-loop / state-conditioned / sign-phase / closed-loop",
                "energy_profile: low / sustained / upfront / early-then-decay / corrective",
                "basin_reachability: nearby / distant / rolling / terminal-boundary",
            ],
        },
        "gym_anchors": gym_anchors(),
        "spectrum_rows": rows,
        "remaining_progress": remaining,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["spectrum_id", "gym_anchor", "axis", "coverage_read", "pressure", "next_action"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Action-Conditioned Future-Basin Spectrum",
        "",
        "Exploratory/read-only profile governance artifact. It is not a milestone change.",
        "",
        "## Core Definition",
        "",
        report["definition"]["core"],
        "",
        "## Axes",
        "",
    ]
    for axis in report["definition"]["axes"]:
        lines.append(f"- {axis}")
    lines.extend(
        [
            "",
            "## Gym Anchors",
            "",
            "| env | spectrum key | why |",
            "|---|---|---|",
        ]
    )
    for anchor in report["gym_anchors"]:
        lines.append(f"| `{anchor['env']}` | `{anchor['spectrum_key']}` | {anchor['why']} |")
    lines.extend(
        [
            "",
            "## Spectrum Coverage Rows",
            "",
            "| spectrum | Gym anchor | pressure | read | next |",
            "|---|---|---|---|---|",
        ]
    )
    for row in report["spectrum_rows"]:
        lines.append(
            f"| `{row['spectrum_id']}` | {row['gym_anchor']} | {row['pressure']} | "
            f"{row['coverage_read']} | {row['next_action']} |"
        )
    lines.extend(["", "## Remaining Progress", ""])
    for key in ("completed", "open", "do_not_do"):
        lines.append(f"### {key}")
        for item in report["remaining_progress"][key]:
            lines.append(f"- {item}")
        lines.append("")
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    report = build_report(
        unified=_read_json_optional(args.unified_profile_json),
        mountaincar=_read_json_optional(args.mountaincar_json),
        settle_specimen=_read_json_optional(args.settle_specimen_json),
        settle_calibration=_read_json_optional(args.settle_calibration_json),
    )
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "action_conditioned_future_basin_spectrum.json"
    md_path = out_dir / "action_conditioned_future_basin_spectrum.md"
    csv_path = out_dir / "action_conditioned_future_basin_spectrum.csv"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(_markdown(report) + "\n", encoding="utf-8")
    _write_csv(csv_path, report["spectrum_rows"])
    print(
        json.dumps(
            {
                "output_dir": str(out_dir),
                "closed_enough": report["remaining_progress"]["closed_enough_for_next_candidate_milestone"],
                "open_count": len(report["remaining_progress"]["open"]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--unified-profile-json", default=str(DEFAULT_UNIFIED_PROFILE_JSON))
    parser.add_argument("--mountaincar-json", default=str(DEFAULT_MOUNTAINCAR_JSON))
    parser.add_argument("--settle-specimen-json", default=str(DEFAULT_SETTLE_SPECIMEN_JSON))
    parser.add_argument("--settle-calibration-json", default=str(DEFAULT_SETTLE_CALIBRATION_JSON))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
