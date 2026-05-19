#!/usr/bin/env python
"""Write the v2e profile-candidate mechanism spec.

Read-only consolidation.  The goal is to make v2e maintainable by separating
mechanism definitions from evidence and by stating which mechanisms are
selection rules, which are annotations, and which are only guards.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


ART = Path("/home/chen/RLPFN/artifacts")
DEFAULT_OUTPUT_DIR = ART / "phase2_profile_v2e_mechanism_spec_0511"
DEFAULT_SELECTED_CSV = (
    ART
    / "phase2_profile_v2e_candidate_annotated_0511"
    / "selected_profile_v2e_candidate_prior_envs.csv"
)
DEFAULT_SOURCE_POOL_JSON = (
    ART / "phase2_feedback_source_pool_broaden_audit_0511/feedback_source_pool_broaden_audit.json"
)
DEFAULT_NEAR_BEST_JSON = (
    ART / "phase2_feedback_near_best_gain_audit_v2d_0511/feedback_near_best_gain_audit.json"
)
DEFAULT_SMOKE_JSON = (
    ART / "phase2_profile_v2e_candidate_smoke_summary_0511/profile_v2e_smoke_summary.json"
)
DEFAULT_DECISION_JSON = (
    ART / "phase2_profile_v2e_decision_summary_0511/profile_v2e_decision_summary.json"
)
DEFAULT_CLOSURE_REGISTER_JSON = (
    ART / "phase2_profile_closure_register_0512/profile_closure_register.json"
)
DEFAULT_REWARD_RELEVANT_STATE_JSON = (
    ART
    / "phase2_pendulum_reward_relevant_state_settle_audit_0512"
    / "pendulum_reward_relevant_state_settle_audit.json"
)
RULES = ("gym_full_obs_action_range", "gym_q90_obs_action_range")
BROAD_AXES = (
    "locomotion_witness",
    "low_energy_not_ctrl_only",
    "sign_or_phase_sensitive",
    "state_conditioned_energy_injection",
)


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).expanduser().resolve().open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    return value


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _rows_by_rule(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    return {rule: [row for row in rows if str(row.get("rule")) == rule] for rule in RULES}


def _strict_summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    strict = [row for row in rows if _bool(row.get("strict_feedback_candidate"))]
    return {
        "strict_count": int(len(strict)),
        "source_pool_definition": dict(
            Counter(str(row.get("strict_feedback_source_pool_definition", "")) for row in strict)
        ),
        "effective_source": dict(Counter(str(row.get("strict_feedback_effective_source", "")) for row in strict)),
        "profile": dict(Counter(str(row.get("strict_feedback_profile", "")) for row in strict)),
        "gain": dict(Counter(str(row.get("strict_feedback_gain", "")) for row in strict)),
        "near_best_gain1_count": int(
            sum(_bool(row.get("v2e_strict_near_best_gain_1")) for row in strict)
        ),
        "near_best_gain05_count": int(
            sum(_bool(row.get("v2e_strict_near_best_gain_0.5")) for row in strict)
        ),
        "near_best_gain1_rate": float(
            sum(_bool(row.get("v2e_strict_near_best_gain_1")) for row in strict)
            / max(1, len(strict))
        ),
        "near_best_gain05_rate": float(
            sum(_bool(row.get("v2e_strict_near_best_gain_0.5")) for row in strict)
            / max(1, len(strict))
        ),
    }


def _selected_summary(selected_rows: list[dict[str, str]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for rule, rows in _rows_by_rule(selected_rows).items():
        cells = Counter(tuple(_bool(row.get(axis)) for axis in BROAD_AXES) for row in rows)
        max_cell = max(cells.values()) if cells else 0
        out[rule] = {
            "n": int(len(rows)),
            "broad_axis_rates": {
                axis: float(sum(_bool(row.get(axis)) for row in rows) / max(1, len(rows)))
                for axis in BROAD_AXES
            },
            "joint_active_cells": int(len(cells)),
            "joint_max_cell_share": float(max_cell / max(1, len(rows))),
            "strict_feedback": _strict_summary(rows),
            "mountaincar_candidate_count": int(
                sum(_bool(row.get("mountaincar_future_basin_candidate")) for row in rows)
            ),
            "mountaincar_strong_count": int(
                sum(_bool(row.get("mountaincar_future_basin_strong")) for row in rows)
            ),
        }
    return out


def build_report(
    *,
    selected_csv: str | Path,
    source_pool_json: str | Path,
    near_best_json: str | Path,
    smoke_json: str | Path,
    decision_json: str | Path,
    closure_register_json: str | Path = DEFAULT_CLOSURE_REGISTER_JSON,
    reward_relevant_state_json: str | Path = DEFAULT_REWARD_RELEVANT_STATE_JSON,
) -> dict[str, Any]:
    selected_rows = _read_csv(selected_csv)
    source_pool = _read_json(source_pool_json)
    near_best = _read_json(near_best_json)
    smoke = _read_json(smoke_json)
    decision = _read_json(decision_json)
    closure_register = _read_json(closure_register_json)
    reward_relevant_state = _read_json(reward_relevant_state_json)
    selected_summary = _selected_summary(selected_rows)
    disallowed_names = {
        str(item.get("name")): item for item in closure_register.get("disallowed_repairs", [])
    }
    allowed_names = {str(item.get("name")): item for item in closure_register.get("allowed_repairs", [])}
    mechanisms = [
        {
            "id": "profile_band_base",
            "name": "unified full/q90 broad-profile band",
            "definition": (
                "Select fixed prior environments with protected broad axes: locomotion witness, "
                "low-energy weak-action optimum, sign/phase sensitivity, and state-conditioned energy injection. "
                "The selected list must keep enough joint-profile cells and avoid one dominant cell."
            ),
            "v2e_role": "inherited_selection_from_v2d",
            "is_quota": True,
            "changes_generator": False,
            "current_read": {
                rule: {
                    "n": item["n"],
                    "joint_active_cells": item["joint_active_cells"],
                    "joint_max_cell_share": item["joint_max_cell_share"],
                    "broad_axis_rates": item["broad_axis_rates"],
                }
                for rule, item in selected_summary.items()
            },
        },
        {
            "id": "feedback_source_pool_repair",
            "name": "strict_or_time_profile_core_no_random",
            "definition": (
                "Feedback strict source pool is the union of original strict best-feedback rows and "
                "time-profile core rows. A time-profile core row means time_profile_core is true and "
                "best_time_profile_mode is not early_random_then_zero. If this time-profile condition holds, "
                "the effective feedback mode/source is taken from best_time_profile_mode; otherwise it is "
                "taken from best_feedback_mode. This repairs feedback profile definition collapse without "
                "raising feedback quantity by itself."
            ),
            "v2e_role": "definition_repair_and_annotation",
            "is_quota": False,
            "changes_generator": False,
            "current_read": {
                "source_pool_audit": source_pool.get("read", {}),
                "selected": {
                    rule: item["strict_feedback"] for rule, item in selected_summary.items()
                },
            },
        },
        {
            "id": "feedback_near_best_gain_guard",
            "name": "near-best gain support",
            "definition": (
                "For each selected env/profile, replay a finite feedback controller family: "
                "profiles constant and decay; observation features 0..3; signs pos/neg; gains 0.5, 1.0, 2.0. "
                "For gain g, compute the best return among modes with that gain. Mark g near-best when "
                "best_return_over_all_gains - best_return_at_gain_g <= max(near_best_abs_margin, "
                "near_best_rel_margin * return_spread). Current defaults are 5.0 and 0.05. "
                "This is a guard/submetric, not a hard gain quota."
            ),
            "v2e_role": "submetric_guard",
            "is_quota": False,
            "changes_generator": False,
            "current_read": near_best.get("overall_read", {}),
        },
        {
            "id": "mountaincar_future_basin_detector",
            "name": "upfront investment plus future basin",
            "definition": (
                "Detect early-control investment patterns: early-only action has nontrivial prefix action "
                "and decays in the suffix; suffix env reward beats zero/open-loop by a dynamic margin; "
                "full episode payoff repays the early action. Candidate uses zero baseline; strong uses "
                "open-loop baseline. This is prior-side coverage only for v2e."
            ),
            "v2e_role": "coverage_annotation_only",
            "is_quota": False,
            "changes_generator": False,
            "current_read": {
                rule: {
                    "candidate_count": item["mountaincar_candidate_count"],
                    "strong_count": item["mountaincar_strong_count"],
                }
                for rule, item in selected_summary.items()
            },
        },
        {
            "id": "pendulum_closed_loop_settle_gap",
            "name": "sustained closed-loop settle",
            "definition": (
                "Pendulum-like settle means a state/phase-conditioned controller first uses enough action to move "
                "toward a high-reward basin, then keeps suffix reward high while action and reward-relevant state "
                "drift settle. It is not the same as one-step feedback witness, time-core support, or weak "
                "non-worsening state drift."
            ),
            "v2e_role": "required_but_not_quota_ready",
            "is_quota": False,
            "changes_generator": False,
            "current_read": {
                "reward_relevant_state_audit": reward_relevant_state.get("decision", {}),
                "disallowed_repairs": {
                    name: disallowed_names.get(name, {}).get("reason")
                    for name in (
                        "time_core_only_pendulum_quota",
                        "existing_signature_formula_quota",
                        "suffix_state_settle_quota_now",
                        "reward_relevant_state_settle_quota_now",
                        "simple_reference_inertia_highway_knob",
                    )
                    if name in disallowed_names
                },
                "allowed_repair": allowed_names.get("cheap_clean_pendulum_settle_guard", {}),
            },
        },
        {
            "id": "generator_level_selector_orchestrator",
            "name": "bounded profile-label selector startup",
            "definition": (
                "fit_model repeatedly generates fresh environment batches, so profile selection must be one bounded "
                "orchestrator: sample candidates, attach cheap or cached labels in chunks, select by profile bands, "
                "and fail explicitly if support is insufficient. Full feedback or reward-state grids stay offline."
            ),
            "v2e_role": "engineering_closure_requirement",
            "is_quota": False,
            "changes_generator": False,
            "current_read": {
                "allowed_repair": allowed_names.get("bounded_generator_level_selector", {}),
                "full_grid_rejected": disallowed_names.get("full_feedback_grid_in_fit_model_startup", {}),
            },
        },
        {
            "id": "integrated_smoke_guard",
            "name": "profile/diversity/delta-mean smoke guard",
            "definition": (
                "Run the fixed full/q90 lists through trusted pack-runner sidecar smoke and check profile guard, "
                "state/reward/done/action diversity guard, and raw/env delta mean guard. Delta sign counts and "
                "raw score levels are not objectives."
            ),
            "v2e_role": "health_guard",
            "is_quota": False,
            "changes_generator": False,
            "current_read": smoke.get("overall", {}),
        },
    ]
    return {
        "analysis_entry": "phase2_profile_v2e_mechanism_spec",
        "contract": {
            "exploratory_only": True,
            "read_only_existing_artifacts": True,
            "does_not_sample_environments": True,
            "does_not_train": True,
            "does_not_modify_exact_scm": True,
            "does_not_modify_fit_model": True,
            "does_not_modify_ppo": True,
            "profile_quality_first": True,
            "delta_sign_fraction_not_a_gate": True,
        },
        "inputs": {
            "selected_csv": str(Path(selected_csv).expanduser().resolve()),
            "source_pool_json": str(Path(source_pool_json).expanduser().resolve()),
            "near_best_json": str(Path(near_best_json).expanduser().resolve()),
            "smoke_json": str(Path(smoke_json).expanduser().resolve()),
            "decision_json": str(Path(decision_json).expanduser().resolve()),
            "closure_register_json": str(Path(closure_register_json).expanduser().resolve()),
            "reward_relevant_state_json": str(Path(reward_relevant_state_json).expanduser().resolve()),
        },
        "v2e_definition": {
            "short_name": "profile_band_v2e_thin_annotated_candidate",
            "core_contract": (
                "v2e is a thin profile-governance candidate. It reuses the v2d fixed environment list exactly, "
                "adds feedback near-best-gain and MountainCar coverage annotations, and relies on smoke guards. "
                "It is not a new generator, not a new PPO path, and not yet a fit_model milestone."
            ),
            "inherits": [
                "milestone2 gated_reward_path_balance healthy prior generator",
                "v2d unified profile-band selected fixed lists",
            ],
            "does_not_include": [
                "hard gain quota",
                "MountainCar quota",
                "Pendulum settle quota before clean count-safe guard",
                "delta-sign-count objective",
                "new exact SCM behavior",
            ],
        },
        "selected_summary": selected_summary,
        "mechanisms": mechanisms,
        "decision_read": decision.get("v2e_candidate_read", {}),
        "next_required": [
            "freeze this mechanism spec as the v2e candidate contract",
            "update the single mode coverage table to reference this spec",
            "resolve Pendulum closed-loop settle as a generator/yield mechanism before quota",
            "do pack-to-fit_model parity migration only against this frozen contract",
        ],
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Profile v2e Mechanism Spec",
        "",
        report["v2e_definition"]["core_contract"],
        "",
        "## Contract",
        "",
        "- read-only/exploratory artifact",
        "- no exact SCM, fit_model, or PPO modification",
        "- v2e reuses v2d fixed lists exactly",
        "- delta sign counts and raw score levels are not objectives",
        "",
        "## Main Mechanisms",
        "",
        "| id | role | quota | generator change | definition |",
        "|---|---|---|---|---|",
    ]
    for mechanism in report["mechanisms"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{mechanism['id']}`",
                    mechanism["v2e_role"],
                    "yes" if mechanism["is_quota"] else "no",
                    "yes" if mechanism["changes_generator"] else "no",
                    mechanism["definition"],
                ]
            )
            + " |"
        )
    lines.extend(["", "## Current Coverage Read", ""])
    lines.append(
        "| rule | n | cells | max cell | strict | strict profile | strict gain | near gain1 | MC cand/strong |"
    )
    lines.append("|---|---:|---:|---:|---:|---|---|---:|---:|")
    for rule, item in report["selected_summary"].items():
        strict = item["strict_feedback"]
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{rule}`",
                    str(item["n"]),
                    str(item["joint_active_cells"]),
                    f"{item['joint_max_cell_share']:.3f}",
                    str(strict["strict_count"]),
                    "`" + json.dumps(strict["profile"], sort_keys=True) + "`",
                    "`" + json.dumps(strict["gain"], sort_keys=True) + "`",
                    f"{strict['near_best_gain1_rate']:.3f}",
                    f"{item['mountaincar_candidate_count']}/{item['mountaincar_strong_count']}",
                ]
            )
            + " |"
        )
    lines.extend(["", "## What Is Fixed", ""])
    lines.append("- Feedback source-pool/profile collapse is fixed at candidate-definition level.")
    lines.append("- Integrated smoke passed profile, diversity, and raw/env delta-mean guards.")
    lines.extend(["", "## What Is Not Claimed", ""])
    lines.append("- Gain=2.0 is not hard-balanced; near-best support is tracked as a guard.")
    lines.append("- MountainCar future-basin is annotated only and is not quota-ready.")
    lines.append("- v2e is not yet a fit_model milestone until parity migration passes.")
    lines.extend(["", "## Next Required", ""])
    for item in report["next_required"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Files", ""])
    for key, value in report.get("outputs", {}).items():
        lines.append(f"- `{key}`: `{value}`")
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(
        selected_csv=args.selected_csv,
        source_pool_json=args.source_pool_json,
        near_best_json=args.near_best_json,
        smoke_json=args.smoke_json,
        decision_json=args.decision_json,
        closure_register_json=args.closure_register_json,
        reward_relevant_state_json=args.reward_relevant_state_json,
    )
    json_path = out_dir / "profile_v2e_mechanism_spec.json"
    md_path = out_dir / "profile_v2e_mechanism_spec.md"
    report["outputs"] = {"json": str(json_path), "markdown": str(md_path)}
    json_path.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report["outputs"], sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--selected-csv", default=str(DEFAULT_SELECTED_CSV))
    parser.add_argument("--source-pool-json", default=str(DEFAULT_SOURCE_POOL_JSON))
    parser.add_argument("--near-best-json", default=str(DEFAULT_NEAR_BEST_JSON))
    parser.add_argument("--smoke-json", default=str(DEFAULT_SMOKE_JSON))
    parser.add_argument("--decision-json", default=str(DEFAULT_DECISION_JSON))
    parser.add_argument("--closure-register-json", default=str(DEFAULT_CLOSURE_REGISTER_JSON))
    parser.add_argument("--reward-relevant-state-json", default=str(DEFAULT_REWARD_RELEVANT_STATE_JSON))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
