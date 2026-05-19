#!/usr/bin/env python
"""Build the v2e mode coverage table from the frozen mechanism spec.

Read-only table generator.  It keeps the profile main line explicit: broad
axes are protected profile bands; feedback v2e is a definition repair plus
near-best guard; MountainCar remains annotation-only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ART = Path("/home/chen/RLPFN/artifacts")
DEFAULT_SPEC_JSON = ART / "phase2_profile_v2e_mechanism_spec_0511/profile_v2e_mechanism_spec.json"
DEFAULT_OUTPUT_DIR = ART / "phase2_mode_coverage_table_v2e_0511"
BROAD_AXES = {
    "low_energy_weak_action_optimum": {
        "profile_key": "low_energy_not_ctrl_only",
        "gym_targets": "Ant-v5, HalfCheetah-v5, Swimmer-v5",
        "v2e_mechanism": "profile_band_base",
        "status": "protected broad band",
        "decision": "preserve; do not tune as feedback/Pendulum fix",
    },
    "sign_phase_sensitive_action": {
        "profile_key": "sign_or_phase_sensitive",
        "gym_targets": "Pendulum-v1, Reacher-v5, locomotion phase/direction",
        "v2e_mechanism": "profile_band_base",
        "status": "protected broad band, necessary not sufficient",
        "decision": "preserve as witness; strict feedback handles closed-loop subtype",
    },
    "state_conditioned_energy_injection": {
        "profile_key": "state_conditioned_energy_injection",
        "gym_targets": "Pendulum-v1, MountainCarContinuous-v0, recovery/control",
        "v2e_mechanism": "profile_band_base",
        "status": "protected broad band",
        "decision": "preserve; MountainCar future-basin is the narrower delayed subtype",
    },
    "locomotion_sustained_energy_tradeoff": {
        "profile_key": "locomotion_witness",
        "gym_targets": "Ant-v5, HalfCheetah-v5, Swimmer-v5, Walker2d-v5",
        "v2e_mechanism": "profile_band_base",
        "status": "protected broad band",
        "decision": "preserve; no locomotion-only repair in v2e",
    },
}


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def _rate(spec: dict[str, Any], rule: str, key: str) -> float:
    return float(spec["selected_summary"][rule]["broad_axis_rates"][key])


def _strict_read(spec: dict[str, Any], rule: str) -> dict[str, Any]:
    return spec["selected_summary"][rule]["strict_feedback"]


def build_report(spec_json: str | Path) -> dict[str, Any]:
    spec = _read_json(spec_json)
    rows: list[dict[str, Any]] = []
    for mode_id, item in BROAD_AXES.items():
        key = item["profile_key"]
        rows.append(
            {
                "mode_id": mode_id,
                "gym_targets": item["gym_targets"],
                "v2e_mechanism": item["v2e_mechanism"],
                "full_read": f"{_rate(spec, 'gym_full_obs_action_range', key):.3f}",
                "q90_read": f"{_rate(spec, 'gym_q90_obs_action_range', key):.3f}",
                "status": item["status"],
                "decision": item["decision"],
            }
        )
    full_strict = _strict_read(spec, "gym_full_obs_action_range")
    q90_strict = _strict_read(spec, "gym_q90_obs_action_range")
    rows.append(
        {
            "mode_id": "context_identifiable_closed_loop_settle_feedback",
            "gym_targets": "Pendulum-v1, stabilization-style Reacher/Inverted variants",
            "v2e_mechanism": "feedback_source_pool_repair + feedback_near_best_gain_guard",
            "full_read": (
                f"strict={full_strict['strict_count']}; profile={full_strict['profile']}; "
                f"near_gain1={full_strict['near_best_gain1_rate']:.3f}"
            ),
            "q90_read": (
                f"strict={q90_strict['strict_count']}; profile={q90_strict['profile']}; "
                f"near_gain1={q90_strict['near_best_gain1_rate']:.3f}"
            ),
            "status": "definition repair candidate; near-best gain remains guard",
            "decision": "keep v2e definition; do not add hard gain quota",
        }
    )
    pendulum_mechanism = next(
        (m for m in spec.get("mechanisms", []) if m.get("id") == "pendulum_closed_loop_settle_gap"),
        {},
    )
    pendulum_read = pendulum_mechanism.get("current_read", {}).get("reward_relevant_state_audit", {})
    rows.append(
        {
            "mode_id": "pendulum_sustained_closed_loop_settle",
            "gym_targets": "Pendulum-v1",
            "v2e_mechanism": "pendulum_closed_loop_settle_gap",
            "full_read": f"guard_ready={pendulum_read.get('cheap_suffix_state_settle_guard_ready')}",
            "q90_read": f"min_count={pendulum_read.get('min_count')}; min_ratio={pendulum_read.get('min_ratio')}",
            "status": "required but not quota-ready; generator/yield blocker",
            "decision": "do not close by threshold loosening, raw delta, or simple inertia/highway knob",
        }
    )
    rows.append(
        {
            "mode_id": "mountaincar_upfront_investment_future_basin",
            "gym_targets": "MountainCarContinuous-v0, delayed early-investment tasks",
            "v2e_mechanism": "mountaincar_future_basin_detector",
            "full_read": (
                f"{spec['selected_summary']['gym_full_obs_action_range']['mountaincar_candidate_count']}/"
                f"{spec['selected_summary']['gym_full_obs_action_range']['mountaincar_strong_count']}"
            ),
            "q90_read": (
                f"{spec['selected_summary']['gym_q90_obs_action_range']['mountaincar_candidate_count']}/"
                f"{spec['selected_summary']['gym_q90_obs_action_range']['mountaincar_strong_count']}"
            ),
            "status": "coverage annotation only; not quota-ready",
            "decision": "do not quota until prior coverage and Gym anchoring are stable",
        }
    )
    rows.append(
        {
            "mode_id": "terminal_survival_action_sensitive",
            "gym_targets": "InvertedPendulum-v5, InvertedDoublePendulum-v5, survival locomotion",
            "v2e_mechanism": "none in v2e",
            "full_read": "not changed",
            "q90_read": "not changed",
            "status": "separate detector line",
            "decision": "do not fold survival reward into v2e profile",
        }
    )
    return {
        "analysis_entry": "phase2_mode_coverage_table_v2e",
        "contract": {
            "read_only_existing_spec": True,
            "does_not_sample_environments": True,
            "does_not_train": True,
            "does_not_modify_exact_scm": True,
            "does_not_modify_fit_model": True,
            "does_not_modify_ppo": True,
            "profile_quality_first": True,
        },
        "inputs": {"spec_json": str(Path(spec_json).expanduser().resolve())},
        "rows": rows,
        "read": {
            "v2e_main_improvement": (
                "feedback strict source-pool/profile definition is repaired while broad profile bands stay protected"
            ),
            "largest_remaining_unclosed_mode": "MountainCar future-basin remains annotation-only",
            "largest_semantic_blocker": "Pendulum sustained closed-loop settle remains required but not quota-ready",
            "fit_model_status": "not migrated; parity migration should use this table and the v2e mechanism spec as contract",
        },
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Mode Coverage Table v2e",
        "",
        "Read-only table generated from the frozen v2e mechanism spec.",
        "",
        "## Read",
        "",
    ]
    for key, value in report["read"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(
        [
            "",
            "## Coverage",
            "",
            "| mode_id | gym targets | v2e mechanism | full read | q90 read | status | decision |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for row in report["rows"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['mode_id']}`",
                    row["gym_targets"],
                    f"`{row['v2e_mechanism']}`",
                    str(row["full_read"]),
                    str(row["q90_read"]),
                    row["status"],
                    row["decision"],
                ]
            )
            + " |"
        )
    lines.extend(["", "## Files", ""])
    for key, value in report.get("outputs", {}).items():
        lines.append(f"- `{key}`: `{value}`")
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(args.spec_json)
    json_path = out_dir / "mode_coverage_table_v2e.json"
    md_path = out_dir / "mode_coverage_table_v2e.md"
    report["outputs"] = {"json": str(json_path), "markdown": str(md_path)}
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report["outputs"], sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec-json", default=str(DEFAULT_SPEC_JSON))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
