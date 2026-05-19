#!/usr/bin/env python
"""Closure gate for the current category-span evidence chain.

The gate consumes read-only audits and reports whether the currently declared
training-family category is closed.  It deliberately does not decide whether
optional formula axes should be promoted to required; that is a category-design
decision, not an implementation patch.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


ART = Path("/home/chen/RLPFN/artifacts")
DEFAULT_OUTPUT_DIR = ART / "phase2_category_closure_gate_stratified_0518"


def _read(path: str | Path) -> Any:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _status(pass_value: bool, detail: str) -> dict[str, Any]:
    return {"pass": bool(pass_value), "detail": detail}


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_selected = _read(args.source_selected_json)
    span = _read(args.span_json)
    structural = _read(args.structural_json)
    rank = _read(args.rank_json)
    topology = _read(args.topology_json)
    env_opt = _read(args.env_opt_json)
    pack_fit = _read(args.pack_fit_json)
    sidecar_delta = _read(args.sidecar_delta_json)
    realized_nz_builder = _read(args.realized_noise_zero_builder_json) if args.realized_noise_zero_builder_json else None
    realized_nz_naturalness = _read(args.realized_noise_zero_naturalness_json) if args.realized_noise_zero_naturalness_json else None

    reductions = source_selected.get("reduction_rows", [])
    selected_source_pass = all(
        row.get("repair_locus") == "none"
        and row.get("source_status") in {"pass_uniform", "covered_skewed"}
        and row.get("selected_status") in {"pass_uniform", "covered_skewed"}
        for row in reductions
    )
    span_read = span.get("read", {})
    structural_read = structural.get("read", {})
    rank_summary = rank.get("summary", {})
    topology_read = topology.get("read", {})
    env_opt_read = env_opt.get("read", {})
    sidecar_read = sidecar_delta.get("read", {})

    checks = {
        "source_selected_required_axes": _status(
            selected_source_pass,
            "source and selected h-list pass all required source axes; no source-axis repair needed",
        ),
        "hard_source_span": _status(
            bool(span_read.get("hard_source_span_pass")),
            str(span_read.get("hard_source_span_pass_reason")),
        ),
        "aggregate_ppo_health": _status(
            bool(span_read.get("aggregate_ppo_health_pass")),
            "true-runner progress has no PPO health hard failure",
        ),
        "realized_core_span": _status(
            bool(structural_read.get("realized_core_span_pass")),
            "realized state/obs/action/terminal core axes and key pairs span",
        ),
        "obs_projection_rank": _status(
            float(structural_read.get("obs_identity_projection_full_obs_rank_fraction") or 0.0) >= 1.0,
            "realized obs projection has full obs-rank under identity-subset image",
        ),
        "controllability_rank": _status(
            bool(rank_summary.get("controllability_rank_sufficiency_pass")),
            "finite-diff action->state/obs rank full and reward action sensitivity present",
        ),
        "topology_required_axes": _status(
            bool(topology_read.get("hard_topology_pass")),
            "required formula evidence axes pass; optional formula topology collapse is not promoted",
        ),
        "pack_fit_parity": _status(
            bool(pack_fit.get("hard_pass")),
            "collect/update/grad parity hard pass on representative action×terminal cells",
        ),
        "environment_per_cell_optimizability": _status(
            bool(env_opt_read.get("per_cell_env_optimizability_pass"))
            and bool(env_opt_read.get("aggregate_env_optimizability_pass")),
            "all action×terminal cells have local/trajectory improvement and finite rewards",
        ),
        "sidecar_delta_degeneracy_guard": _status(
            bool(sidecar_read.get("aggregate_ppo_health_pass"))
            and int(sidecar_read.get("action_terminal_empty_cell_count") or 0) == 0,
            "sidecar first/last delta proxy has no empty action×terminal cells and aggregate PPO health passes",
        ),
    }
    if realized_nz_builder is not None and realized_nz_naturalness is not None:
        nz_builder_cov = realized_nz_builder.get("selected_realized_noise_zero_coverage", {})
        nz_natural_read = realized_nz_naturalness.get("read", {})
        checks["feasible_realized_noise_zero_span"] = _status(
            bool(nz_builder_cov.get("pass_uniform"))
            and bool(nz_natural_read.get("covered_cell_naturalness_pass")),
            (
                "realized noise×zero is evaluated on feasible materialized-image cells, "
                "not impossible rectangular cells; covered cells are natural enough"
            ),
        )
    hard_failures = [name for name, item in checks.items() if not bool(item["pass"])]

    unresolved_decisions = []
    feasible_nz_closed = bool(checks.get("feasible_realized_noise_zero_span", {}).get("pass", False))
    if not bool(structural_read.get("realized_noise_zero_span_pass")) and not feasible_nz_closed:
        unresolved_decisions.append(
            {
                "axis": "realized_noise_dim × realized_zero_pad_dim",
                "status": "not_uniform_or_partial",
                "decision": (
                    "If realized noise/zero image is a hard training-family category, repair by "
                    "stratifying fixed-list selection over realized image after materialization. "
                    "If it is only padding/noise budget diagnostic, do not change generator."
                ),
            }
        )
    optional_collapsed = topology_read.get("optional_collapsed_formula_axes", [])
    for axis in optional_collapsed:
        unresolved_decisions.append(
            {
                "axis": axis,
                "status": "optional_formula_axis_collapsed",
                "decision": "promote to required only if this topology is part of the formal category; otherwise protect current simple form",
            }
        )
    if not bool(env_opt_read.get("official_ppo_pack_optimizability")):
        unresolved_decisions.append(
            {
                "axis": "per_cell_official_ppo_pack_optimizability",
                "status": "not_run",
                "decision": (
                    "Current evidence has aggregate PPO health plus per-cell env optimizability and sidecar delta guard. "
                    "Run per-cell PPO pack experiments only if per-cell official post-train delta is promoted to hard."
                ),
            }
        )

    report = {
        "analysis_entry": "phase2_category_closure_gate",
        "schema": "phase2_category_closure_gate.v1",
        "contract": {
            "read_only": True,
            "does_not_modify_generator": True,
            "separates_testing_from_development": True,
            "does_not_promote_optional_axes_to_hard_failures": True,
        },
        "current_declared_category_closed": bool(not hard_failures),
        "hard_failures": hard_failures,
        "checks": checks,
        "unresolved_design_decisions": unresolved_decisions,
        "inputs": {
            "source_selected_json": str(Path(args.source_selected_json).expanduser().resolve()),
            "span_json": str(Path(args.span_json).expanduser().resolve()),
            "structural_json": str(Path(args.structural_json).expanduser().resolve()),
            "rank_json": str(Path(args.rank_json).expanduser().resolve()),
            "topology_json": str(Path(args.topology_json).expanduser().resolve()),
            "env_opt_json": str(Path(args.env_opt_json).expanduser().resolve()),
            "pack_fit_json": str(Path(args.pack_fit_json).expanduser().resolve()),
            "sidecar_delta_json": str(Path(args.sidecar_delta_json).expanduser().resolve()),
            "realized_noise_zero_builder_json": (
                None
                if args.realized_noise_zero_builder_json is None
                else str(Path(args.realized_noise_zero_builder_json).expanduser().resolve())
            ),
            "realized_noise_zero_naturalness_json": (
                None
                if args.realized_noise_zero_naturalness_json is None
                else str(Path(args.realized_noise_zero_naturalness_json).expanduser().resolve())
            ),
        },
    }
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "category_closure_gate.json"
    md_path = out_dir / "category_closure_gate.md"
    report["outputs"] = {"json": str(json_path), "markdown": str(md_path)}
    json_path.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Category Closure Gate",
        "",
        f"- current declared category closed: `{report['current_declared_category_closed']}`",
        f"- hard failures: `{hard_failures}`",
        "",
        "## Checks",
        "",
        "| check | pass | detail |",
        "|---|---:|---|",
    ]
    for name, item in checks.items():
        lines.append(f"| `{name}` | `{item['pass']}` | {item['detail']} |")
    lines.extend(["", "## Unresolved Design Decisions", ""])
    if unresolved_decisions:
        for item in unresolved_decisions:
            lines.append(f"- `{item['axis']}`: `{item['status']}`; {item['decision']}")
    else:
        lines.append("- none")
    lines.extend(["", "## Outputs", ""])
    for key, value in report["outputs"].items():
        lines.append(f"- `{key}`: `{value}`")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report["outputs"], sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-selected-json",
        default=str(
            ART
            / "phase2_category_span_source_vs_selected_stratified_0518"
            / "category_span_source_vs_selected_audit.json"
        ),
    )
    parser.add_argument(
        "--span-json",
        default=str(ART / "phase2_category_span_audit_stratified_runner_0518" / "category_span_audit.json"),
    )
    parser.add_argument(
        "--structural-json",
        default=str(
            ART
            / "phase2_category_structural_image_audit_stratified_0518"
            / "category_structural_image_audit.json"
        ),
    )
    parser.add_argument(
        "--rank-json",
        default=str(
            ART
            / "phase2_category_controllability_rank_audit_stratified_0518"
            / "category_controllability_rank_audit.json"
        ),
    )
    parser.add_argument(
        "--topology-json",
        default=str(
            ART
            / "phase2_category_topology_label_audit_stratified_0518"
            / "category_topology_label_audit.json"
        ),
    )
    parser.add_argument(
        "--env-opt-json",
        default=str(
            ART
            / "phase2_category_env_optimizability_stratified_0518"
            / "category_env_optimizability_cell_audit.json"
        ),
    )
    parser.add_argument(
        "--pack-fit-json",
        default=str(
            ART
            / "phase2_stratified_source_fixed_h_list_pack_fit_parity_n9_s16_0518"
            / "gated_pack_fit_model_parity_report.json"
        ),
    )
    parser.add_argument(
        "--sidecar-delta-json",
        default=str(
            ART
            / "phase2_category_span_per_cell_delta_stratified_0518"
            / "category_span_per_cell_delta_audit.json"
        ),
    )
    parser.add_argument("--realized-noise-zero-builder-json", default=None)
    parser.add_argument("--realized-noise-zero-naturalness-json", default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
