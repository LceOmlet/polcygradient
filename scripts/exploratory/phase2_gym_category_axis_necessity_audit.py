#!/usr/bin/env python
"""Read-only reduction of Gym-derived category axes.

This audit deliberately does not implement a new environment family.  It
separates:

* axes that are necessary from the Gym/PPO computation interface,
* axes whose current "collapse" is a desirable health condition,
* optional formula topology axes that need a category decision before repair,
* observed outcome axes that must not be repaired directly.

The output is a development gate: it says which tests must be added or rerun
before any generator change is justified.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


ART = Path("/home/chen/RLPFN/artifacts")
DEFAULT_OUTPUT_DIR = ART / "phase2_gym_category_axis_necessity_audit_realized_noise_zero_0518"


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


def _by_key(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    return {str(row.get(key)): row for row in rows}


def _find_structural(coverage: list[dict[str, Any]], axis: str, image: str) -> dict[str, Any] | None:
    for row in coverage:
        if str(row.get("axis")) == axis and str(row.get("image")) == image:
            return row
    return None


def _source_axis_row(axis: str, source_rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    row = source_rows.get(axis, {})
    source_status = str(row.get("source_status", "missing"))
    selected_status = str(row.get("selected_status", "missing"))
    ok = source_status in {"pass_uniform", "covered_skewed"} and selected_status in {
        "pass_uniform",
        "covered_skewed",
    }
    return {
        "axis": axis,
        "class": "required_source_uniform",
        "current_state": f"source={source_status}; selected={selected_status}",
        "necessity": "Gym-compatible training must not bind this structural source axis to a narrow subrange.",
        "sufficient_pass_form": "source and selected h-list both cover declared bins, with no source-axis repair locus",
        "minimal_next_action": "protect; no generator change",
        "priority": 0 if ok else 1,
        "needs_generator_repair": bool(not ok),
        "needs_test_repair": False,
    }


def _realized_axis_row(axis: str, structural_rows: list[dict[str, Any]]) -> dict[str, Any]:
    row = _find_structural(structural_rows, axis, "realized") or {}
    status = str(row.get("status", "missing"))
    ok = status in {"pass_uniform", "covered_skewed"}
    return {
        "axis": f"realized_{axis}",
        "class": "required_realized_image",
        "current_state": status,
        "necessity": "The source axis must survive materialization into the true runner image.",
        "sufficient_pass_form": "realized image covers feasible bins; impossible rectangular cells are not required",
        "minimal_next_action": "protect; if this fails, repair selection over feasible materialized cells before formulas",
        "priority": 0 if ok else 2,
        "needs_generator_repair": False,
        "needs_test_repair": bool(not ok),
    }


def _topology_row(axis: str, topology_rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    row = topology_rows.get(axis, {})
    return {
        "status": str(row.get("status", "missing")),
        "occupied_expected_labels": int(row.get("occupied_expected_labels") or 0),
        "expected_labels": int(row.get("expected_labels") or 0),
        "entropy": float(row.get("normalized_entropy") or 0.0),
        "hard_pass": bool(row.get("hard_pass", False)),
        "minimal_repair_if_required": str(row.get("minimal_repair_if_required", "")),
    }


def _coupling_status(axis: str, coupling_rows: list[dict[str, Any]]) -> dict[str, Any]:
    highs = [
        row
        for row in coupling_rows
        if str(row.get("topology_axis")) == axis and str(row.get("status")) == "high"
    ]
    return {
        "high_count": len(highs),
        "pairs": [
            {
                "structure_axis": row.get("structure_axis"),
                "cramers_v": row.get("cramers_v"),
            }
            for row in highs
        ],
    }


def _formula_axis_rows(
    topology_rows: dict[str, dict[str, Any]], coupling_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for axis, why in [
        (
            "controllability_rank",
            "Every Gym continuous-control task requires action perturbations to affect state/obs somewhere in the rollout.",
        ),
        (
            "reward_action_dependency",
            "The PPO signal must have some action-sensitive reward path; action-insensitive rewards are a degenerate training family.",
        ),
    ]:
        topo = _topology_row(axis, topology_rows)
        rows.append(
            {
                "axis": axis,
                "class": "required_good_collapse",
                "current_state": topo["status"],
                "necessity": why,
                "sufficient_pass_form": "all environments remain in the good label: full rank / action-sensitive",
                "minimal_next_action": "protect; do not make this uniform",
                "priority": 0 if topo["hard_pass"] else 1,
                "needs_generator_repair": bool(not topo["hard_pass"]),
                "needs_test_repair": False,
            }
        )

    terminal = _topology_row("terminal_formula_topology", topology_rows)
    rows.append(
        {
            "axis": "terminal_formula_topology",
            "class": "candidate_required_formula_topology",
            "current_state": (
                f"{terminal['status']} "
                f"({terminal['occupied_expected_labels']}/{terminal['expected_labels']} labels)"
            ),
            "necessity": (
                "Gym distinguishes time-limit truncation, state-boundary failure, healthy/survival continuation, "
                "and no early terminal tasks.  If the training category claims Gym-and-nearby coverage, terminal "
                "formula type is a source topology, not a done_count statistic."
            ),
            "sufficient_pass_form": (
                "declared terminal formula labels have source coverage, selected coverage, no high coupling to "
                "dims/modes, and pass PPO health; done_count is only an outcome diagnostic"
            ),
            "minimal_next_action": (
                "first add a terminal-formula provenance audit; only then add an explicit terminal-formula source "
                "if the declared labels are missing"
            ),
            "priority": 1,
            "needs_generator_repair": False,
            "needs_test_repair": True,
        }
    )

    component = _topology_row("reward_component_topology", topology_rows)
    rows.append(
        {
            "axis": "reward_component_topology",
            "class": "candidate_required_formula_topology",
            "current_state": (
                f"{component['status']} "
                f"({component['occupied_expected_labels']}/{component['expected_labels']} expected labels; "
                "current label schema is component-string based)"
            ),
            "necessity": (
                "Gym reward formulas commonly mix progress/task reward, control cost, survival/healthy bonus, "
                "terminal penalty, sparse success, and potential-delta structure.  If those are not represented as "
                "formula components, the scalar reward distribution can look broad while the semantic basis collapses."
            ),
            "sufficient_pass_form": (
                "decompose reward into boolean component axes first; each declared component has support and remains "
                "independent from structural source bins; PPO health and per-cell optimizability remain finite"
            ),
            "minimal_next_action": (
                "repair the audit definition first: split component-string labels into component boolean axes; do not "
                "change generator until the boolean audit shows a true missing component"
            ),
            "priority": 1,
            "needs_generator_repair": False,
            "needs_test_repair": True,
        }
    )

    density = _topology_row("reward_density_topology", topology_rows)
    density_coupling = _coupling_status("reward_density_topology", coupling_rows)
    rows.append(
        {
            "axis": "reward_density_topology",
            "class": "candidate_required_independence_axis",
            "current_state": (
                f"{density['status']} "
                f"({density['occupied_expected_labels']}/{density['expected_labels']} labels); "
                f"high_coupling={density_coupling['high_count']}"
            ),
            "necessity": (
                "Reward may depend on state/action/noise-like latent factors, but this axis is not a Gym API field. "
                "It becomes necessary only as an independence guard: reward information must not be secretly encoded "
                "by state_dim/obs_dim/action_dim."
            ),
            "sufficient_pass_form": (
                "if promoted, use explicit reward input-allocation source labels and require no high coupling to "
                "structure; do not infer topology from dim fractions or post-hoc reward magnitudes"
            ),
            "minimal_next_action": (
                "refine the density label to use formula/source provenance; current high coupling makes it a test "
                "priority, not yet a generator patch"
            ),
            "priority": 1 if density_coupling["high_count"] else 3,
            "needs_generator_repair": False,
            "needs_test_repair": True,
            "high_coupling_pairs": density_coupling["pairs"],
        }
    )

    rows.append(
        {
            "axis": "dynamics_temporal_response_topology",
            "class": "missing_required_test_axis",
            "current_state": "not_logged",
            "necessity": (
                "Gym includes immediate control, inertial locomotion, pendulum settling, and MountainCar-style "
                "delayed investment.  Rank-at-one-step is necessary but not sufficient for this temporal topology."
            ),
            "sufficient_pass_form": (
                "impulse-response audit over short horizons labels immediate / inertial / delayed-basin response; "
                "declared labels have coverage and remain independent from source dimensions"
            ),
            "minimal_next_action": (
                "add a read-only impulse-response topology audit before any dynamics generator change"
            ),
            "priority": 1,
            "needs_generator_repair": False,
            "needs_test_repair": True,
        }
    )

    transform = _topology_row("reward_transform_topology", topology_rows)
    rows.append(
        {
            "axis": "reward_transform_topology",
            "class": "protected_simple_or_optional",
            "current_state": (
                f"{transform['status']} "
                f"({transform['occupied_expected_labels']}/{transform['expected_labels']} labels)"
            ),
            "necessity": (
                "Gym exposes scalar reward, not a required diversity of reward transforms.  Transform diversity is "
                "a robustness choice and can easily damage PPO scale semantics."
            ),
            "sufficient_pass_form": "keep simple unless transform invariance is explicitly declared",
            "minimal_next_action": "do not repair now",
            "priority": 4,
            "needs_generator_repair": False,
            "needs_test_repair": False,
        }
    )

    for axis in ["reward_value_shape", "terminal_observed_topology", "done_count_bin"]:
        topo = _topology_row(axis, topology_rows)
        rows.append(
            {
                "axis": axis,
                "class": "diagnostic_observed_image",
                "current_state": (
                    f"{topo['status']} "
                    f"({topo['occupied_expected_labels']}/{topo['expected_labels']} labels)"
                ),
                "necessity": "Observed image/outcome axis; useful for diagnostics, not a direct generator source.",
                "sufficient_pass_form": "use it to falsify source formulas, never as a direct repair target",
                "minimal_next_action": "do not repair directly",
                "priority": 5,
                "needs_generator_repair": False,
                "needs_test_repair": False,
            }
        )
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_selected = _read(args.source_selected_json)
    structural = _read(args.structural_json)
    topology = _read(args.topology_json)
    closure = _read(args.closure_json)

    source_rows = _by_key(source_selected.get("reduction_rows", []), "axis")
    structural_rows = structural.get("axis_coverage", [])
    topology_rows = _by_key(topology.get("axis_summary", []), "axis")
    coupling_rows = topology.get("structure_topology_coupling", [])

    rows: list[dict[str, Any]] = []
    for axis in [
        "state_dim_source",
        "obs_dim_source",
        "action_dim_source",
        "terminal_target_source",
        "noise_dim_source",
        "zero_pad_dim_source",
        "reward_scale_source",
        "init_state_std_source",
        "state_noise_std_source",
        "activation",
    ]:
        rows.append(_source_axis_row(axis, source_rows))

    for axis in ["state_dim", "obs_dim", "action_dim", "terminal_target", "noise_dim", "zero_pad_dim"]:
        rows.append(_realized_axis_row(axis, structural_rows))

    rows.extend(_formula_axis_rows(topology_rows, coupling_rows))

    priority_rows = sorted(
        [row for row in rows if row["priority"] > 0],
        key=lambda row: (int(row["priority"]), str(row["axis"])),
    )
    test_repairs = [row["axis"] for row in priority_rows if row.get("needs_test_repair")]
    generator_repairs = [row["axis"] for row in priority_rows if row.get("needs_generator_repair")]

    report = {
        "analysis_entry": "phase2_gym_category_axis_necessity_audit",
        "schema": "phase2_gym_category_axis_necessity_audit.v1",
        "contract": {
            "read_only": True,
            "does_not_modify_generator": True,
            "does_not_train": True,
            "separates_testing_from_development": True,
            "does_not_promote_observed_outcomes_to_source_axes": True,
        },
        "read": {
            "current_declared_category_closed": bool(closure.get("current_declared_category_closed")),
            "generator_change_justified_now": bool(generator_repairs),
            "generator_change_justified_now_reason": (
                "hard required source/formula axis failed"
                if generator_repairs
                else "no hard required source/formula axis failed; next work is test/definition reduction"
            ),
            "test_repair_axes_before_generator_change": test_repairs,
            "highest_priority": [row["axis"] for row in priority_rows if int(row["priority"]) == 1],
        },
        "axis_reduction_rows": rows,
        "ordered_open_items": priority_rows,
        "inputs": {
            "source_selected_json": str(Path(args.source_selected_json).expanduser().resolve()),
            "structural_json": str(Path(args.structural_json).expanduser().resolve()),
            "topology_json": str(Path(args.topology_json).expanduser().resolve()),
            "closure_json": str(Path(args.closure_json).expanduser().resolve()),
        },
    }

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "gym_category_axis_necessity_audit.json"
    md_path = out_dir / "gym_category_axis_necessity_audit.md"
    report["outputs"] = {"json": str(json_path), "markdown": str(md_path)}
    json_path.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report["outputs"], sort_keys=True))
    return report


def _markdown(report: dict[str, Any]) -> str:
    read = report["read"]
    lines = [
        "# Gym Category Axis Necessity Audit",
        "",
        "Read-only reduction from the Gym/PPO computation interface to category axes.",
        "",
        "## Read",
        "",
        f"- current declared category closed: `{read['current_declared_category_closed']}`",
        f"- generator change justified now: `{read['generator_change_justified_now']}`",
        f"- reason: {read['generator_change_justified_now_reason']}",
        f"- test repair axes before generator change: `{read['test_repair_axes_before_generator_change']}`",
        f"- highest priority open axes: `{read['highest_priority']}`",
        "",
        "## Ordered Open Items",
        "",
        "| priority | axis | class | current state | next action |",
        "|---:|---|---|---|---|",
    ]
    for row in report["ordered_open_items"]:
        lines.append(
            f"| {row['priority']} | `{row['axis']}` | `{row['class']}` | "
            f"{row['current_state']} | {row['minimal_next_action']} |"
        )
    lines.extend(
        [
            "",
            "## Full Axis Reduction",
            "",
            "| axis | class | current state | necessary form | sufficient pass form |",
            "|---|---|---|---|---|",
        ]
    )
    for row in report["axis_reduction_rows"]:
        lines.append(
            f"| `{row['axis']}` | `{row['class']}` | {row['current_state']} | "
            f"{row['necessity']} | {row['sufficient_pass_form']} |"
        )
    lines.extend(
        [
            "",
            "## Development Rule",
            "",
            "- Do not implement a generator repair while `generator_change_justified_now` is false.",
            "- First repair tests/definitions for the listed topology axes.",
            "- Only promote a formula topology to generator work after source-formula coverage, independence, PPO health, and pack-fit criteria are defined.",
            "",
            "## Outputs",
            "",
        ]
    )
    for key, value in report["outputs"].items():
        lines.append(f"- `{key}`: `{value}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-selected-json",
        default=str(
            ART
            / "phase2_category_span_source_vs_selected_realized_noise_zero_0518"
            / "category_span_source_vs_selected_audit.json"
        ),
    )
    parser.add_argument(
        "--structural-json",
        default=str(
            ART
            / "phase2_category_structural_image_audit_realized_noise_zero_0518"
            / "category_structural_image_audit.json"
        ),
    )
    parser.add_argument(
        "--topology-json",
        default=str(
            ART
            / "phase2_category_topology_label_audit_realized_noise_zero_0518"
            / "category_topology_label_audit.json"
        ),
    )
    parser.add_argument(
        "--closure-json",
        default=str(
            ART
            / "phase2_category_closure_gate_realized_noise_zero_0518"
            / "category_closure_gate.json"
        ),
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
