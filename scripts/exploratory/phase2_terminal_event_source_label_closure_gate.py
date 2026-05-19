#!/usr/bin/env python
"""Closure gate for the terminal-event reward source-label axis.

This audit does not sample environments, run PPO, or modify the generator.  It
only joins the proof/test artifacts that must all be true before treating the
terminal-event reward axis as a maintained source-label axis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ART = Path("/home/chen/RLPFN/artifacts")
DEFAULT_FORMULA = ART / "phase2_true_runner_reward_formula_path_evidence_audit_0518" / (
    "true_runner_reward_formula_path_evidence_audit.json"
)
DEFAULT_FIXED = ART / "phase2_terminal_event_source_label_fixed_h_list_0518" / (
    "terminal_event_source_label_fixed_h_list_report.json"
)
DEFAULT_CATEGORY = ART / "phase2_terminal_event_source_label_category_span_u2_s128_schema2_0518" / (
    "category_span_audit.json"
)
DEFAULT_ENV_OPT = ART / "phase2_terminal_event_source_label_env_opt_cells_u64_h6_c64_0518" / (
    "category_env_optimizability_cell_audit.json"
)
DEFAULT_PACK_FIT = ART / "phase2_terminal_event_source_label_pack_fit_parity_n2_s32_0518" / (
    "profile_v2e_pack_fit_model_parity_report.json"
)
DEFAULT_OUTPUT_DIR = ART / "phase2_terminal_event_source_label_closure_gate_0518"


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def _axis(category: dict[str, Any], name: str) -> dict[str, Any]:
    for row in category.get("axis_coverage", []):
        if row.get("axis") == name:
            return row
    return {}


def _required_pairs(category: dict[str, Any], name: str) -> list[dict[str, Any]]:
    rows = []
    for row in category.get("pair_coverage", []):
        if row.get("role") == "required_category_pair" and name in {row.get("axis_a"), row.get("axis_b")}:
            rows.append(row)
    return rows


def _markdown(report: dict[str, Any]) -> str:
    read = report["read"]
    lines = [
        "# Terminal Event Source-Label Closure Gate",
        "",
        "Read-only closure gate for the terminal-event reward source-label axis.",
        "",
        "## Read",
        "",
        f"- closure pass: `{read['closure_pass']}`",
        f"- formula path pass: `{read['formula_path_pass']}`",
        f"- fixed source-label pass: `{read['fixed_source_label_pass']}`",
        f"- category span pass: `{read['category_span_pass']}`",
        f"- pack-fit parity pass: `{read['pack_fit_parity_pass']}`",
        f"- per-cell env optimizability pass: `{read['per_cell_env_optimizability_pass']}`",
        f"- aggregate PPO health pass: `{read['aggregate_ppo_health_pass']}`",
        f"- unauthorized reward axes: `{read['unauthorized_reward_axes']}`",
        "",
        "## Terminal Event Axis",
        "",
    ]
    axis = report["terminal_event_axis"]
    lines.append(
        f"- axis status: `{axis.get('status')}`, counts: off={axis.get('count:off')}, on={axis.get('count:on')}"
    )
    lines.append("")
    lines.append("## Required Terminal Event Pairs")
    lines.append("")
    for row in report["terminal_event_required_pairs"]:
        lines.append(
            f"- `{row.get('axis_a')}` × `{row.get('axis_b')}`: `{row.get('status')}`, "
            f"entropy={row.get('normalized_entropy')}, max_cell={row.get('max_cell_fraction')}"
        )
    lines.extend(["", "## Remaining Non-Closure Items", ""])
    for item in report["remaining_items"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Outputs", ""])
    for key, value in report["outputs"].items():
        lines.append(f"- `{key}`: `{value}`")
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    formula = _read_json(args.formula)
    fixed = _read_json(args.fixed)
    category = _read_json(args.category)
    env_opt = _read_json(args.env_opt)
    pack_fit = _read_json(args.pack_fit)

    axis = _axis(category, "terminal_event_reward_source")
    pairs = _required_pairs(category, "terminal_event_reward_source")
    read_formula = formula.get("read", {}) or {}
    read_fixed = fixed.get("read", {}) or {}
    read_category = category.get("read", {}) or {}
    read_env_opt = env_opt.get("read", {}) or {}

    formula_path_pass = bool(
        read_formula.get("terminal_event_true_runner_path_usable")
        and not read_formula.get("terminal_event_pack_fallback_logging_gap")
        and "terminal_event_reward_or_penalty" in set(read_formula.get("source_label_sampler_authorized", []))
    )
    fixed_source_label_pass = bool(
        read_fixed.get("source_label_sampler_materialized")
        and read_fixed.get("high_structure_coupling_count") == 0
        and read_fixed.get("medium_structure_coupling_count") == 0
        and read_fixed.get("label_counts", {}).get("terminal_event_reward_on") == read_fixed.get("label_counts", {}).get(
            "terminal_event_reward_off"
        )
    )
    terminal_axis_pass = bool(axis.get("status") == "pass_uniform")
    terminal_pair_pass = bool(pairs) and all(row.get("status") == "pass_uniform" for row in pairs)
    category_span_pass = bool(
        read_category.get("hard_source_span_pass")
        and read_category.get("high_structure_mode_coupling_count") == 0
        and terminal_axis_pass
        and terminal_pair_pass
    )
    pack_fit_pass = bool(pack_fit.get("hard_pass"))
    per_cell_env_opt_pass = bool(
        read_env_opt.get("aggregate_env_optimizability_pass")
        and read_env_opt.get("per_cell_env_optimizability_pass")
        and read_env_opt.get("nonfinite_candidate_reward_frac") == 0.0
    )
    aggregate_ppo_health_pass = bool(read_category.get("aggregate_ppo_health_pass"))
    closure_pass = bool(
        formula_path_pass
        and fixed_source_label_pass
        and category_span_pass
        and pack_fit_pass
        and per_cell_env_opt_pass
        and aggregate_ppo_health_pass
    )

    unauthorized = []
    if read_formula.get("potential_delta_or_progress_path_usable_but_not_loggable"):
        unauthorized.append("potential_delta_or_progress")

    report = {
        "analysis_entry": "phase2_terminal_event_source_label_closure_gate",
        "contract": {
            "read_only_existing_artifacts": True,
            "does_not_sample_environments": True,
            "does_not_train": True,
            "does_not_modify_generator": True,
            "does_not_modify_ppo": True,
            "proof_and_implementation_separated": True,
        },
        "inputs": {
            "formula": str(Path(args.formula).expanduser().resolve()),
            "fixed": str(Path(args.fixed).expanduser().resolve()),
            "category": str(Path(args.category).expanduser().resolve()),
            "env_opt": str(Path(args.env_opt).expanduser().resolve()),
            "pack_fit": str(Path(args.pack_fit).expanduser().resolve()),
        },
        "read": {
            "closure_pass": closure_pass,
            "formula_path_pass": formula_path_pass,
            "fixed_source_label_pass": fixed_source_label_pass,
            "category_span_pass": category_span_pass,
            "pack_fit_parity_pass": pack_fit_pass,
            "per_cell_env_optimizability_pass": per_cell_env_opt_pass,
            "aggregate_ppo_health_pass": aggregate_ppo_health_pass,
            "unauthorized_reward_axes": unauthorized,
        },
        "terminal_event_axis": axis,
        "terminal_event_required_pairs": pairs,
        "remaining_items": [
            "This closes terminal_event_reward_source as a fixed source-label axis, not the whole training family.",
            "potential_delta/progress remains blocked until true-runner component logging exists.",
            "per-env official PPO post-train delta is still separate from the local M3 env optimizability probe.",
            "category span still lists controllability/projection evidence axes that require separate audits before a full milestone.",
        ],
    }

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": out_dir / "terminal_event_source_label_closure_gate.json",
        "markdown": out_dir / "terminal_event_source_label_closure_gate.md",
    }
    report["outputs"] = {key: str(value) for key, value in paths.items()}
    paths["json"].write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths["markdown"].write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report["read"], sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formula", default=str(DEFAULT_FORMULA))
    parser.add_argument("--fixed", default=str(DEFAULT_FIXED))
    parser.add_argument("--category", default=str(DEFAULT_CATEGORY))
    parser.add_argument("--env-opt", default=str(DEFAULT_ENV_OPT))
    parser.add_argument("--pack-fit", default=str(DEFAULT_PACK_FIT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
