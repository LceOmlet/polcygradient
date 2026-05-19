#!/usr/bin/env python
"""Forecast v2e profile-candidate readiness and migration gaps.

Read-only consolidation.  This is deliberately not a selector and not a
trainability optimizer: it keeps the profile-mainline question separate from
PPO delta monitoring and fit_model migration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ART = Path("/home/chen/RLPFN/artifacts")
REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ART / "phase2_profile_v2e_readiness_forecast_0511"
DEFAULT_SPEC_JSON = ART / "phase2_profile_v2e_mechanism_spec_0511/profile_v2e_mechanism_spec.json"
DEFAULT_TABLE_JSON = ART / "phase2_mode_coverage_table_v2e_0511/mode_coverage_table_v2e.json"
DEFAULT_SMOKE_JSON = ART / "phase2_profile_v2e_candidate_smoke_summary_0511/profile_v2e_smoke_summary.json"
DEFAULT_DECISION_JSON = (
    ART / "phase2_profile_v2e_decision_summary_0511/profile_v2e_decision_summary.json"
)
DEFAULT_MAINTAINED_PATH = REPO / "ticl/rlpfn_maintained_path.py"
DEFAULT_CLI_PARSING = REPO / "ticl/cli_parsing.py"
DEFAULT_TRAIN = REPO / "ticl/train.py"
DEFAULT_PARITY_SCRIPT = REPO / "ticl/analysis/phase2_fit_model_pack_ppo_migration_parity.py"
DEFAULT_V2E_PARITY_SCRIPT = REPO / "scripts/exploratory/phase2_profile_v2e_pack_fit_model_parity.py"
DEFAULT_V2E_PARITY_JSON = (
    ART
    / "phase2_profile_v2e_pack_fit_model_parity_full_q90_smoke_n2_s16_0511"
    / "profile_v2e_pack_fit_model_parity_report.json"
)
DEFAULT_GENERATOR_GAP_JSON = (
    ART
    / "phase2_profile_v2e_generator_level_gap_audit_0511"
    / "profile_v2e_generator_level_gap_audit.json"
)

V2E_ID = "profile_band_v2e_thin_annotated_candidate"


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def _read_json_optional(path: str | Path | None) -> Any:
    if path is None or str(path).strip() == "":
        return None
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        return None
    return json.loads(resolved.read_text(encoding="utf-8"))


def _read_text_optional(path: str | Path) -> str:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        return ""
    return resolved.read_text(encoding="utf-8")


def _smoke_guard_read(smoke: dict[str, Any]) -> dict[str, Any]:
    overall = smoke.get("overall", {})
    scopes = smoke.get("scopes", {}) or {}
    return {
        "candidate_smoke_guard_pass": bool(overall.get("candidate_smoke_guard_pass")),
        "profile_guard_pass": bool(overall.get("profile_guard_pass")),
        "diversity_guard_pass": bool(overall.get("diversity_guard_pass")),
        "trainability_delta_mean_guard_pass": bool(overall.get("trainability_delta_guard_pass")),
        "scope_delta_means": {
            rule: {
                "raw_delta_mean": (payload.get("smoke", {}).get("raw_reward_sum_delta", {}) or {}).get("mean"),
                "env_delta_mean": (payload.get("smoke", {}).get("reward_env_sum_delta", {}) or {}).get("mean"),
            }
            for rule, payload in scopes.items()
        },
    }


def _profile_read(spec: dict[str, Any], table: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any]:
    selected = spec.get("selected_summary", {})
    rows = table.get("rows", [])
    feedback_row = next(
        (row for row in rows if row.get("mode_id") == "context_identifiable_closed_loop_settle_feedback"),
        {},
    )
    mountaincar_row = next(
        (row for row in rows if row.get("mode_id") == "mountaincar_upfront_investment_future_basin"),
        {},
    )
    return {
        "candidate_id": spec.get("v2e_definition", {}).get("short_name"),
        "candidate_profile_definition_ready": bool(
            (decision.get("v2e_candidate_read", {}) or {}).get("profile_definition_close_enough_for_candidate")
        ),
        "broad_profile_rows": [
            row
            for row in rows
            if row.get("v2e_mechanism") == "profile_band_base"
        ],
        "feedback_row": feedback_row,
        "mountaincar_row": mountaincar_row,
        "joint_profile": {
            rule: {
                "n": payload.get("n"),
                "joint_active_cells": payload.get("joint_active_cells"),
                "joint_max_cell_share": payload.get("joint_max_cell_share"),
            }
            for rule, payload in selected.items()
        },
        "not_claimed": list(spec.get("v2e_definition", {}).get("does_not_include", [])),
    }


def _fit_model_support_read(
    *,
    maintained_path: str | Path,
    cli_parsing: str | Path,
    train_py: str | Path,
    parity_script: str | Path,
    v2e_parity_script: str | Path,
) -> dict[str, Any]:
    files = {
        "maintained_path": str(Path(maintained_path).expanduser().resolve()),
        "cli_parsing": str(Path(cli_parsing).expanduser().resolve()),
        "train": str(Path(train_py).expanduser().resolve()),
        "pack_fit_parity": str(Path(parity_script).expanduser().resolve()),
        "v2e_pack_fit_parity": str(Path(v2e_parity_script).expanduser().resolve()),
    }
    texts = {key: _read_text_optional(path) for key, path in files.items()}
    contains = {key: V2E_ID in text or "profile_band_v2e" in text for key, text in texts.items()}
    startup_supported = bool(contains["maintained_path"] and contains["cli_parsing"] and contains["train"])
    parity_supported = bool(contains["pack_fit_parity"])
    v2e_parity_script_present = bool(Path(v2e_parity_script).expanduser().resolve().exists())
    return {
        "v2e_id": V2E_ID,
        "files": files,
        "contains_v2e_reference": contains,
        "fit_model_startup_option_present": startup_supported,
        "pack_to_fit_model_v2e_parity_present": parity_supported,
        "exploratory_v2e_parity_script_present": v2e_parity_script_present,
        "read": (
            "v2e is not yet a fit_model-startable milestone"
            if not startup_supported
            else "v2e has a fit_model startup path; numeric parity still must be checked"
        ),
    }


def _parity_run_read(v2e_parity_json: str | Path | None) -> dict[str, Any]:
    payload = _read_json_optional(v2e_parity_json)
    if not payload:
        return {
            "available": False,
            "hard_pass": False,
            "case_count": 0,
            "cases": [],
            "read": "no v2e numeric pack-to-fit_model parity report found",
        }
    cases = payload.get("cases", []) or []
    hard_pass = bool(payload.get("hard_pass") and cases and all(bool(case.get("hard_pass")) for case in cases))
    return {
        "available": True,
        "hard_pass": hard_pass,
        "case_count": len(cases),
        "cases": [
            {
                "case": case.get("case"),
                "rule": case.get("rule"),
                "hard_pass": bool(case.get("hard_pass")),
                "n_envs": case.get("n_envs"),
                "n_steps": case.get("n_steps"),
                "grad_delta_max_abs": (
                    case.get("update_parity", {}).get("gradient_diff", {}) or {}
                ).get("grad_delta_max_abs"),
                "post_param_delta": (
                    case.get("update_parity", {})
                    .get("one_train_update_compare", {})
                    .get("post_update_param_diff", {})
                    or {}
                ).get("policy_param_max_abs_diff"),
            }
            for case in cases
        ],
        "read": (
            "minimal full/q90 v2e fixed-list pack-to-fit_model numeric parity passed"
            if hard_pass
            else "v2e numeric parity report exists but did not hard-pass"
        ),
    }


def _generator_gap_read(generator_gap_json: str | Path | None) -> dict[str, Any]:
    payload = _read_json_optional(generator_gap_json)
    if not payload:
        return {
            "available": False,
            "generator_level_selector_ready": False,
            "offline_label_axes": [],
            "online_label_axes": [],
            "read": "no generator-level gap audit found",
        }
    read = payload.get("generator_level_read", {}) or {}
    return {
        "available": True,
        "generator_level_selector_ready": bool(read.get("generator_level_selector_ready")),
        "offline_label_axes": list(read.get("offline_label_axes", [])),
        "online_label_axes": list(read.get("online_label_axes", [])),
        "read": read.get("why_not_ready", ""),
    }


def _score(
    profile: dict[str, Any],
    smoke: dict[str, Any],
    support: dict[str, Any],
    parity_run: dict[str, Any],
    generator_gap: dict[str, Any],
) -> dict[str, Any]:
    components = [
        (
            "profile_contract_frozen",
            0.20,
            1.0 if profile.get("candidate_profile_definition_ready") else 0.0,
            "v2e mechanism spec/table/decision define the candidate without changing generator/PPO",
        ),
        (
            "broad_profile_protected",
            0.20,
            1.0 if len(profile.get("broad_profile_rows", [])) >= 4 else 0.0,
            "low-energy, sign/phase, state-conditioned, and locomotion rows are protected as broad bands",
        ),
        (
            "feedback_definition_repaired",
            0.15,
            1.0 if profile.get("feedback_row") else 0.0,
            "feedback source-pool/profile definition is represented in the unified table",
        ),
        (
            "smoke_health_diversity_passed",
            0.20,
            1.0
            if smoke.get("candidate_smoke_guard_pass")
            and smoke.get("profile_guard_pass")
            and smoke.get("diversity_guard_pass")
            and smoke.get("trainability_delta_mean_guard_pass")
            else 0.0,
            "full/q90 trusted pack-runner smoke passes profile, diversity, and raw/env delta-mean guards",
        ),
        (
            "mountaincar_accounted_but_not_quota_ready",
            0.10,
            0.6 if profile.get("mountaincar_row") else 0.0,
            "MountainCar future-basin is an annotation-only pressure point, not a v2e quota",
        ),
        (
            "minimal_pack_to_fit_model_numeric_parity",
            0.10,
            1.0 if parity_run.get("hard_pass") else 0.0,
            "v2e fixed-list pack and fit_model bridge agree on rollout, loss, gradient, and one update in a minimal full/q90 smoke",
        ),
        (
            "generator_level_selector_readiness",
            0.05,
            1.0 if generator_gap.get("generator_level_selector_ready") else 0.0,
            "fit_model startup must use a generator-level selector, not the paritied fixed 64+64 list",
        ),
    ]
    weighted = sum(weight * value for _, weight, value, _ in components)
    return {
        "candidate_profile_readiness_estimate": weighted,
        "components": [
            {
                "id": item_id,
                "weight": weight,
                "score": value,
                "definition": definition,
            }
            for item_id, weight, value, definition in components
        ],
        "estimate_read": (
            "profile candidate evidence is close, but milestone readiness is gated by fit_model parity migration"
            if weighted >= 0.70
            else "profile candidate still has major missing evidence"
        ),
    }


def build_report(
    *,
    spec_json: str | Path,
    table_json: str | Path,
    smoke_json: str | Path,
    decision_json: str | Path,
    maintained_path: str | Path,
    cli_parsing: str | Path,
    train_py: str | Path,
    parity_script: str | Path,
    v2e_parity_script: str | Path,
    v2e_parity_json: str | Path | None,
    generator_gap_json: str | Path | None,
) -> dict[str, Any]:
    spec = _read_json(spec_json)
    table = _read_json(table_json)
    smoke_payload = _read_json(smoke_json)
    decision = _read_json(decision_json)
    profile = _profile_read(spec, table, decision)
    smoke = _smoke_guard_read(smoke_payload)
    support = _fit_model_support_read(
        maintained_path=maintained_path,
        cli_parsing=cli_parsing,
        train_py=train_py,
        parity_script=parity_script,
        v2e_parity_script=v2e_parity_script,
    )
    parity_run = _parity_run_read(v2e_parity_json)
    generator_gap = _generator_gap_read(generator_gap_json)
    score = _score(profile, smoke, support, parity_run, generator_gap)
    remaining = []
    if not parity_run["hard_pass"]:
        remaining.append("run a v2e fixed-list pack-to-fit_model numeric parity smoke")
    if not generator_gap["generator_level_selector_ready"]:
        remaining.append("port/wrap offline profile labelers into a bounded online candidate-pool selector")
    if generator_gap["generator_level_selector_ready"] and not support["fit_model_startup_option_present"]:
        remaining.append("after generator-level selector parity passes, add a guarded fit_model startup option for v2e")
    remaining.extend(
        [
            "keep MountainCar as detector/coverage annotation unless later evidence justifies a quota",
            "keep delta sign/raw-score counts as monitoring only; profile balance and diversity remain the main gates",
        ]
    )
    return {
        "analysis_entry": "phase2_profile_v2e_readiness_forecast",
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
            "spec_json": str(Path(spec_json).expanduser().resolve()),
            "table_json": str(Path(table_json).expanduser().resolve()),
            "smoke_json": str(Path(smoke_json).expanduser().resolve()),
            "decision_json": str(Path(decision_json).expanduser().resolve()),
        },
        "profile_read": profile,
        "smoke_read": smoke,
        "fit_model_support_read": support,
        "v2e_numeric_parity_read": parity_run,
        "generator_level_gap_read": generator_gap,
        "readiness": score,
        "remaining_before_sufficient_coverage": remaining,
        "next_highest_value_step": (
            "build a bounded online candidate-pool profile selector before adding a fit_model startup option"
            if parity_run.get("hard_pass") and not generator_gap.get("generator_level_selector_ready")
            else (
                "add the guarded fit_model v2e startup option and rerun parity through that option"
                if parity_run.get("hard_pass")
                else "run v2e fixed-list pack-to-fit_model parity preflight; do not expand profile axes until that path is trusted"
            )
        ),
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Profile v2e Readiness Forecast",
        "",
        "Read-only forecast from existing v2e artifacts and fit_model support status.",
        "",
        "## Verdict",
        "",
        f"- readiness estimate: `{report['readiness']['candidate_profile_readiness_estimate']:.3f}`",
        f"- read: {report['readiness']['estimate_read']}",
        f"- next highest-value step: {report['next_highest_value_step']}",
        "",
        "## Components",
        "",
        "| component | weight | score | definition |",
        "|---|---:|---:|---|",
    ]
    for component in report["readiness"]["components"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{component['id']}`",
                    f"{component['weight']:.2f}",
                    f"{component['score']:.2f}",
                    component["definition"],
                ]
            )
            + " |"
        )
    lines.extend(["", "## Fit Model Support", ""])
    support = report["fit_model_support_read"]
    lines.append(f"- startup option present: `{support['fit_model_startup_option_present']}`")
    lines.append(f"- pack-to-fit_model v2e parity present: `{support['pack_to_fit_model_v2e_parity_present']}`")
    lines.append(f"- exploratory v2e parity script present: `{support['exploratory_v2e_parity_script_present']}`")
    lines.append(f"- read: {support['read']}")
    lines.extend(["", "## Numeric Parity", ""])
    parity_run = report["v2e_numeric_parity_read"]
    lines.append(f"- available: `{parity_run['available']}`")
    lines.append(f"- hard pass: `{parity_run['hard_pass']}`")
    lines.append(f"- case count: `{parity_run['case_count']}`")
    lines.append(f"- read: {parity_run['read']}")
    for case in parity_run["cases"]:
        lines.append(
            f"  - `{case['rule']}`: hard=`{case['hard_pass']}`, n={case['n_envs']}, "
            f"steps={case['n_steps']}, grad_delta={case['grad_delta_max_abs']}, "
            f"post_param_delta={case['post_param_delta']}"
        )
    lines.extend(["", "## Generator-Level Gap", ""])
    generator_gap = report["generator_level_gap_read"]
    lines.append(f"- available: `{generator_gap['available']}`")
    lines.append(f"- generator-level selector ready: `{generator_gap['generator_level_selector_ready']}`")
    lines.append(f"- offline label axes: `{generator_gap['offline_label_axes']}`")
    lines.append(f"- online label axes: `{generator_gap['online_label_axes']}`")
    lines.append(f"- read: {generator_gap['read']}")
    lines.extend(["", "## Smoke", ""])
    smoke = report["smoke_read"]
    for key in (
        "candidate_smoke_guard_pass",
        "profile_guard_pass",
        "diversity_guard_pass",
        "trainability_delta_mean_guard_pass",
    ):
        lines.append(f"- `{key}`: `{smoke[key]}`")
    lines.extend(["", "## Remaining", ""])
    for item in report["remaining_before_sufficient_coverage"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Files", ""])
    for key, value in report.get("outputs", {}).items():
        lines.append(f"- `{key}`: `{value}`")
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(
        spec_json=args.spec_json,
        table_json=args.table_json,
        smoke_json=args.smoke_json,
        decision_json=args.decision_json,
        maintained_path=args.maintained_path,
        cli_parsing=args.cli_parsing,
        train_py=args.train_py,
        parity_script=args.parity_script,
        v2e_parity_script=args.v2e_parity_script,
        v2e_parity_json=args.v2e_parity_json,
        generator_gap_json=args.generator_gap_json,
    )
    json_path = out_dir / "profile_v2e_readiness_forecast.json"
    md_path = out_dir / "profile_v2e_readiness_forecast.md"
    report["outputs"] = {"json": str(json_path), "markdown": str(md_path)}
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report["outputs"], sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec-json", default=str(DEFAULT_SPEC_JSON))
    parser.add_argument("--table-json", default=str(DEFAULT_TABLE_JSON))
    parser.add_argument("--smoke-json", default=str(DEFAULT_SMOKE_JSON))
    parser.add_argument("--decision-json", default=str(DEFAULT_DECISION_JSON))
    parser.add_argument("--maintained-path", default=str(DEFAULT_MAINTAINED_PATH))
    parser.add_argument("--cli-parsing", default=str(DEFAULT_CLI_PARSING))
    parser.add_argument("--train-py", default=str(DEFAULT_TRAIN))
    parser.add_argument("--parity-script", default=str(DEFAULT_PARITY_SCRIPT))
    parser.add_argument("--v2e-parity-script", default=str(DEFAULT_V2E_PARITY_SCRIPT))
    parser.add_argument("--v2e-parity-json", default=str(DEFAULT_V2E_PARITY_JSON))
    parser.add_argument("--generator-gap-json", default=str(DEFAULT_GENERATOR_GAP_JSON))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
