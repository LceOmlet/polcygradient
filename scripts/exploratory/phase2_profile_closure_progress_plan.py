#!/usr/bin/env python
"""Quantify remaining progress to profile semantic coverage closure.

This report is intentionally a planning gate, not a selector.  It consumes the
semantic coverage gate plus generator-level preflight artifacts and converts the
remaining blockers into explicit close criteria and minimum next actions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ART = Path("/home/chen/RLPFN/artifacts")
DEFAULT_OUTPUT_DIR = ART / "phase2_profile_closure_progress_plan_0512"
DEFAULT_SEMANTIC_GATE_JSON = (
    ART / "phase2_profile_semantic_coverage_gate_0512/profile_semantic_coverage_gate.json"
)
DEFAULT_ONLINE_PREFLIGHT_JSON = (
    ART / "phase2_profile_online_selector_preflight_0511/profile_online_selector_preflight.json"
)
DEFAULT_FIT_SUFFICIENCY_JSON = (
    ART / "phase2_profile_fit_model_sufficiency_audit_0511/profile_fit_model_sufficiency_audit.json"
)
DEFAULT_PENDULUM_LIGHT_GUARD_JSON = (
    ART / "phase2_pendulum_settle_light_guard_audit_0512/pendulum_settle_light_guard_audit.json"
)
DEFAULT_PENDULUM_H_FEATURE_JSON = (
    ART / "phase2_pendulum_settle_h_feature_relation_audit_0512/pendulum_settle_h_feature_relation_audit.json"
)
DEFAULT_PENDULUM_GUARD_SEARCH_JSON = (
    ART / "phase2_pendulum_settle_guard_search_0512/pendulum_settle_guard_search.json"
)
DEFAULT_PENDULUM_STATE_LABEL_JSON = (
    ART / "phase2_pendulum_settle_state_label_preflight_0512/pendulum_settle_state_label_preflight.json"
)
DEFAULT_PENDULUM_STATE_BLOCKER_JSON = (
    ART / "phase2_pendulum_settle_state_blocker_decomposition_0512/pendulum_settle_state_blocker_decomposition.json"
)
DEFAULT_PENDULUM_REWARD_RELEVANT_STATE_JSON = (
    ART
    / "phase2_pendulum_reward_relevant_state_settle_audit_0512"
    / "pendulum_reward_relevant_state_settle_audit.json"
)
DEFAULT_SELECTOR_PATH_PARITY_JSON = (
    ART / "phase2_profile_v2e_pack_fit_model_parity_0512/profile_v2e_pack_fit_model_parity_report.json"
)
DEFAULT_CHEAP_MILESTONE3_PREFLIGHT_JSON = (
    ART
    / "phase2_profile_milestone3_cheap_closure_preflight_0513"
    / "profile_milestone3_cheap_closure_preflight.json"
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


def _flag(payload: dict[str, Any], *path: str) -> bool:
    cur: Any = payload
    for key in path:
        if not isinstance(cur, dict):
            return False
        cur = cur.get(key)
    return bool(cur)


def _component(status: str, progress: float, close_condition: str, next_action: str) -> dict[str, Any]:
    return {
        "status": status,
        "progress": float(progress),
        "close_condition": close_condition,
        "next_action": next_action,
    }


def build_report(
    *,
    semantic_gate_json: str | Path,
    online_preflight_json: str | Path,
    fit_sufficiency_json: str | Path,
    pendulum_light_guard_json: str | Path,
    pendulum_h_feature_json: str | Path,
    pendulum_guard_search_json: str | Path,
    pendulum_state_label_json: str | Path,
    pendulum_state_blocker_json: str | Path,
    pendulum_reward_relevant_state_json: str | Path,
    selector_path_parity_json: str | Path,
    cheap_milestone3_preflight_json: str | Path | None = None,
) -> dict[str, Any]:
    semantic = _read_json(semantic_gate_json)
    online = _read_json(online_preflight_json)
    fit = _read_json(fit_sufficiency_json)
    light_guard = _read_json(pendulum_light_guard_json)
    h_feature = _read_json(pendulum_h_feature_json)
    guard_search = _read_json(pendulum_guard_search_json)
    state_label = _read_json(pendulum_state_label_json)
    state_blocker = _read_json(pendulum_state_blocker_json)
    reward_relevant_state = _read_json(pendulum_reward_relevant_state_json)
    selector_parity = _read_json(selector_path_parity_json)
    cheap_m3 = _read_json_optional(cheap_milestone3_preflight_json)

    stage_ready = {
        item["name"]: bool(item["ready"])
        for item in online.get("bounded_generator_plan", {}).get("stage_plan", [])
    }
    online_existing = online.get("existing_evidence", {})
    fit_preflight = fit.get("preflight", {})

    engineering = _component(
        "closed",
        1.0,
        "pack PPO, chunked labels, fit_model fixed-list startup, and pack-to-fit_model numeric parity pass",
        "keep this sealed; do not reopen unless code changes touch the trusted path",
    )
    fixed_list = _component(
        "closed",
        1.0,
        "fixed-list profile selector, diversity smoke, and minimal pack-to-fit_model parity pass",
        "do not promote fixed-list success to generator-level success",
    )

    generator_progress_parts = {
        "bounded_contract": True,
        "one_seed_source_pool_ready": bool(online_existing.get("source_pool", {}).get("source_ready_for_candidate_multiplier")),
        "one_seed_selector_pass": bool(online_existing.get("selector", {}).get("hard_pass")),
        "chunked_2048_memory_safe": bool(
            fit_preflight.get("fit_model_2048_scale_projection", {})
            .get("chunked_probe", {})
            .get("memory_safe")
        ),
        "all_at_once_rejected": not bool(
            fit_preflight.get("fit_model_2048_scale_projection", {})
            .get("naive_all_at_once", {})
            .get("memory_safe", True)
        ),
        "selector_path_pack_fit_parity": bool(selector_parity.get("hard_pass")),
        "generator_startup_option": _flag(semantic, "gates", "generator_level_closed"),
    }
    generator_progress = sum(generator_progress_parts.values()) / len(generator_progress_parts)
    generator = _component(
        "blocked",
        generator_progress,
        (
            "fresh batches are selected by a bounded online candidate-pool selector, labels are chunked/cached, "
            "no silent fallback exists, and selector-path pack-to-fit_model parity passes"
        ),
        (
            "turn the existing bounded plan into a generator-level wrapper; execute missing profile-label stages "
            "on GPU0 in chunks, then run selector-path parity"
        ),
    )

    pendulum_read = semantic.get("reads", {}).get("pendulum_settle", {})
    feedback_read = semantic.get("reads", {}).get("feedback", {})
    pendulum_parts = {
        "anchor_defined": bool(pendulum_read.get("anchor_interpretation")),
        "anchor_pass": bool(pendulum_read.get("anchor_pass")),
        "prior_candidates_exist": (pendulum_read.get("prior_max_settle_count") or 0) > 0,
        "prior_quota_ready": bool(pendulum_read.get("prior_quota_ready")),
        "feedback_slack_ready": bool(feedback_read.get("slack", {}).get("ready_to_clear_slack_blocker")),
        "cheap_guard_ready": bool(light_guard.get("decision", {}).get("cheap_guard_ready")),
        "existing_h_knob_ready": bool(
            h_feature.get("decision", {}).get("strict_current_simple_h_knob_ready")
        ),
        "existing_signature_formula_ready": bool(
            guard_search.get("decision", {}).get("cheap_formula_guard_ready")
        ),
        "suffix_state_label_ready": bool(
            state_label.get("decision", {}).get("cheap_suffix_state_settle_guard_ready")
        ),
        "reward_relevant_state_label_ready": bool(
            reward_relevant_state.get("decision", {}).get("cheap_suffix_state_settle_guard_ready")
        ),
        "state_threshold_repair_allowed": bool(
            state_blocker.get("decision", {}).get("selector_threshold_repair_allowed")
        ),
    }
    pendulum_progress = sum(pendulum_parts.values()) / len(pendulum_parts)
    pendulum = _component(
        "blocked",
        pendulum_progress,
        (
            "sustained closed-loop settle has a cheap prior-side guard with quota-safe coverage across full/q90, "
            "and it reproduces the Pendulum anchor without full feedback grid or an unstable scalar h-knob "
            "bias in default startup"
        ),
        (
            "derive a cheap settle guard from late/suffix state-reward improvement or repair the generator mechanism "
            "that creates clean feedback-decay suffix-supported state-basin cases; validate diversity, then re-run "
            "feedback slack and Pendulum prior coverage audits"
        ),
    )

    mountaincar_read = semantic.get("reads", {}).get("mountaincar_future_basin", {})
    mountain_parts = {
        "gym_anchor_pass": bool(mountaincar_read.get("gym_anchor_pass")),
        "prior_detector_present": bool(mountaincar_read.get("prior_scope_summaries")),
        "prior_guard_pass": bool(mountaincar_read.get("prior_guard_pass")),
    }
    mountain_progress = sum(mountain_parts.values()) / len(mountain_parts)
    mountain = _component(
        "annotation_only",
        mountain_progress,
        "future-basin detector has stable prior fixed-list support and diversity guards pass in full/q90",
        "keep as annotation; improve detector coverage only after generator-level and Pendulum settle blockers are unblocked",
    )

    components = {
        "engineering_parity": engineering,
        "fixed_list_profile": fixed_list,
        "generator_level_selector": generator,
        "pendulum_settle_semantics": pendulum,
        "mountaincar_future_basin": mountain,
    }
    weights = {
        "engineering_parity": 0.20,
        "fixed_list_profile": 0.20,
        "generator_level_selector": 0.25,
        "pendulum_settle_semantics": 0.25,
        "mountaincar_future_basin": 0.10,
    }
    weighted_progress = sum(weights[key] * components[key]["progress"] for key in weights)
    close_enough_threshold = 0.90
    cheap_m3_decision = (cheap_m3 or {}).get("decision", {}) if isinstance(cheap_m3, dict) else {}
    cheap_m3_score = (cheap_m3 or {}).get("score", {}) if isinstance(cheap_m3, dict) else {}
    cheap_m3_weighted_progress = float(cheap_m3_score.get("weighted_progress", 0.0))
    cheap_m3_threshold = float(cheap_m3_score.get("milestone_threshold", 0.90))
    cheap_m3_remaining = float(
        cheap_m3_score.get(
            "remaining_to_threshold",
            max(0.0, cheap_m3_threshold - cheap_m3_weighted_progress),
        )
    )
    cheap_m3_track = {
        "consumed_preflight": bool(cheap_m3),
        "ready_to_land": bool(cheap_m3_decision.get("ready_to_land_milestone3_cheap")),
        "weighted_progress": cheap_m3_weighted_progress,
        "milestone_threshold": cheap_m3_threshold,
        "remaining_to_threshold": cheap_m3_remaining,
        "hard_blockers": list(cheap_m3_decision.get("hard_blockers") or []),
        "scope": "narrow_cheap_milestone3_only",
        "read": (
            "cheap M3 is separately landable; broad sufficient coverage remains governed by the existing semantic gate"
            if cheap_m3_decision.get("ready_to_land_milestone3_cheap")
            else "cheap M3 preflight is consumed but still has blockers"
            if cheap_m3
            else "cheap M3 preflight is not yet connected"
        ),
    }

    return {
        "analysis_entry": "phase2_profile_closure_progress_plan",
        "contract": {
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
            "semantic_gate_json": str(Path(semantic_gate_json).expanduser().resolve()),
            "online_preflight_json": str(Path(online_preflight_json).expanduser().resolve()),
            "fit_sufficiency_json": str(Path(fit_sufficiency_json).expanduser().resolve()),
            "pendulum_light_guard_json": str(Path(pendulum_light_guard_json).expanduser().resolve()),
            "pendulum_h_feature_json": str(Path(pendulum_h_feature_json).expanduser().resolve()),
            "pendulum_guard_search_json": str(Path(pendulum_guard_search_json).expanduser().resolve()),
            "pendulum_state_label_json": str(Path(pendulum_state_label_json).expanduser().resolve()),
            "pendulum_state_blocker_json": str(Path(pendulum_state_blocker_json).expanduser().resolve()),
            "pendulum_reward_relevant_state_json": str(
                Path(pendulum_reward_relevant_state_json).expanduser().resolve()
            ),
            "selector_path_parity_json": str(Path(selector_path_parity_json).expanduser().resolve()),
            "cheap_milestone3_preflight_json": (
                ""
                if cheap_milestone3_preflight_json is None
                or str(cheap_milestone3_preflight_json).strip() == ""
                else str(Path(cheap_milestone3_preflight_json).expanduser().resolve())
            ),
        },
        "progress": {
            "weighted_progress": weighted_progress,
            "close_enough_threshold": close_enough_threshold,
            "remaining_to_threshold": max(0.0, close_enough_threshold - weighted_progress),
            "can_claim_sufficient_coverage": bool(
                weighted_progress >= close_enough_threshold
                and semantic.get("gates", {}).get("sufficient_coverage_closed")
            ),
            "weights": weights,
            "components": components,
        },
        "diagnostics": {
            "generator_progress_parts": generator_progress_parts,
            "pendulum_progress_parts": pendulum_parts,
            "pendulum_light_guard_read": light_guard.get("decision", {}),
            "pendulum_h_feature_read": h_feature.get("decision", {}),
            "pendulum_guard_search_read": guard_search.get("decision", {}),
            "pendulum_state_label_read": state_label.get("decision", {}),
            "pendulum_state_blocker_read": state_blocker.get("decision", {}),
            "pendulum_reward_relevant_state_read": reward_relevant_state.get("decision", {}),
            "mountaincar_progress_parts": mountain_parts,
            "selector_path_parity_read": {
                "hard_pass": bool(selector_parity.get("hard_pass")),
                "analysis_entry": selector_parity.get("analysis_entry"),
            },
            "cheap_milestone3_preflight_read": {
                "analysis_entry": (cheap_m3 or {}).get("analysis_entry") if isinstance(cheap_m3, dict) else None,
                "ready_to_land_milestone3_cheap": bool(
                    cheap_m3_decision.get("ready_to_land_milestone3_cheap")
                ),
                "weighted_progress": cheap_m3_track["weighted_progress"],
                "hard_blockers": cheap_m3_track["hard_blockers"],
            },
            "planned_stage_ready": stage_ready,
        },
        "decision": {
            "largest_remaining_blocker": "pendulum_settle_semantics",
            "second_blocker": "generator_level_selector",
            "milestone_tracks": {
                "broad_sufficient_coverage": {
                    "can_claim": bool(
                        weighted_progress >= close_enough_threshold
                        and semantic.get("gates", {}).get("sufficient_coverage_closed")
                    ),
                    "weighted_progress": weighted_progress,
                    "remaining_to_threshold": max(0.0, close_enough_threshold - weighted_progress),
                    "scope": "broad_profile_gym_semantic_coverage",
                },
                "cheap_milestone3": cheap_m3_track,
            },
            "why": (
                "Engineering and fixed-list profile are closed.  The remaining sufficient-coverage gap is semantic: "
                "Pendulum sustained settle is sparse/not reproduced, and generator-level selector semantics are not yet fit_model-startable."
            ),
            "next_minimal_sequence": [
                "Do not add new profile axes.",
                "Do not quota the new suffix-state settle label yet: it is semantically better but not count-safe on full/q90 fixed lists.",
                "Because existing light guards are broad/impure or clean/sparse, existing scalar h knobs do not stably predict clean settle, bounded formulas over existing signature fields do not close the gap, suffix-state settle and reward-relevant-state settle currently have zero count-safe support, and loosening state thresholds is semantically insufficient, improve the clean state-basin settle generator mechanism/yield before adding a quota.",
                "Wire bounded online selector startup only as an explicit no-fallback wrapper after the semantic blocker is resolved; selector-path pack-to-fit_model parity is already passed and should be rerun only if startup code changes.",
                "Only after those pass, revisit MountainCar as a detector/annotation; do not quota it yet.",
            ],
        },
    }


def _markdown(report: dict[str, Any]) -> str:
    progress = report["progress"]
    cheap_track = report["decision"].get("milestone_tracks", {}).get("cheap_milestone3", {})
    broad_track = report["decision"].get("milestone_tracks", {}).get("broad_sufficient_coverage", {})
    lines = [
        "# Profile Closure Progress Plan",
        "",
        "Read-only progress accounting for profile semantic coverage closure.",
        "",
        "## Verdict",
        "",
        f"- weighted progress: `{progress['weighted_progress']:.3f}`",
        f"- close-enough threshold: `{progress['close_enough_threshold']:.3f}`",
        f"- remaining to threshold: `{progress['remaining_to_threshold']:.3f}`",
        f"- can claim sufficient coverage: `{progress['can_claim_sufficient_coverage']}`",
        f"- largest remaining blocker: `{report['decision']['largest_remaining_blocker']}`",
        f"- second blocker: `{report['decision']['second_blocker']}`",
        f"- cheap M3 preflight consumed: `{cheap_track.get('consumed_preflight')}`",
        f"- cheap M3 ready to land: `{cheap_track.get('ready_to_land')}`",
        f"- cheap M3 hard blockers: `{cheap_track.get('hard_blockers')}`",
        "",
        "## Milestone Tracks",
        "",
        f"- broad sufficient coverage: `{broad_track}`",
        f"- cheap milestone3: `{cheap_track}`",
        "",
        "## Components",
        "",
        "| component | weight | progress | status | close condition | next action |",
        "|---|---:|---:|---|---|---|",
    ]
    for key, component in progress["components"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{key}`",
                    f"{progress['weights'][key]:.2f}",
                    f"{component['progress']:.2f}",
                    f"`{component['status']}`",
                    component["close_condition"],
                    component["next_action"],
                ]
            )
            + " |"
        )
    lines.extend(["", "## Next Minimal Sequence", ""])
    for item in report["decision"]["next_minimal_sequence"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Diagnostics", ""])
    for name, payload in report["diagnostics"].items():
        lines.append(f"- `{name}`: `{payload}`")
    lines.extend(["", "## Files", ""])
    for key, value in report.get("outputs", {}).items():
        lines.append(f"- `{key}`: `{value}`")
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(
        semantic_gate_json=args.semantic_gate_json,
        online_preflight_json=args.online_preflight_json,
        fit_sufficiency_json=args.fit_sufficiency_json,
        pendulum_light_guard_json=args.pendulum_light_guard_json,
        pendulum_h_feature_json=args.pendulum_h_feature_json,
        pendulum_guard_search_json=args.pendulum_guard_search_json,
        pendulum_state_label_json=args.pendulum_state_label_json,
        pendulum_state_blocker_json=args.pendulum_state_blocker_json,
        pendulum_reward_relevant_state_json=args.pendulum_reward_relevant_state_json,
        selector_path_parity_json=args.selector_path_parity_json,
        cheap_milestone3_preflight_json=args.cheap_milestone3_preflight_json,
    )
    json_path = out_dir / "profile_closure_progress_plan.json"
    md_path = out_dir / "profile_closure_progress_plan.md"
    report["outputs"] = {"json": str(json_path), "markdown": str(md_path)}
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report["outputs"], sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semantic-gate-json", default=str(DEFAULT_SEMANTIC_GATE_JSON))
    parser.add_argument("--online-preflight-json", default=str(DEFAULT_ONLINE_PREFLIGHT_JSON))
    parser.add_argument("--fit-sufficiency-json", default=str(DEFAULT_FIT_SUFFICIENCY_JSON))
    parser.add_argument("--pendulum-light-guard-json", default=str(DEFAULT_PENDULUM_LIGHT_GUARD_JSON))
    parser.add_argument("--pendulum-h-feature-json", default=str(DEFAULT_PENDULUM_H_FEATURE_JSON))
    parser.add_argument("--pendulum-guard-search-json", default=str(DEFAULT_PENDULUM_GUARD_SEARCH_JSON))
    parser.add_argument("--pendulum-state-label-json", default=str(DEFAULT_PENDULUM_STATE_LABEL_JSON))
    parser.add_argument("--pendulum-state-blocker-json", default=str(DEFAULT_PENDULUM_STATE_BLOCKER_JSON))
    parser.add_argument(
        "--pendulum-reward-relevant-state-json",
        default=str(DEFAULT_PENDULUM_REWARD_RELEVANT_STATE_JSON),
    )
    parser.add_argument("--selector-path-parity-json", default=str(DEFAULT_SELECTOR_PATH_PARITY_JSON))
    parser.add_argument("--cheap-milestone3-preflight-json", default=str(DEFAULT_CHEAP_MILESTONE3_PREFLIGHT_JSON))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
