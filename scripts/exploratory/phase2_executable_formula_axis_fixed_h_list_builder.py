#!/usr/bin/env python
"""Build a fixed h-list that repairs executable formula-axis gaps only.

This is a fixed-list materialization step, not a generator rewrite.  It starts
from the current stratified source list and only changes formula fields that are
known to be active in the maintained Exact-SCM runner image:

* terminal formula: terminal_off / hazard_reset / override_hazard_reset /
  min_step_hazard_reset
* reward component: survival_bonus on/off

Lowtail-only fields such as potential_delta, distribution_head, and
shared_energy_terminal are intentionally not used here unless the lowtail
wrapper itself is active.  This prevents dormant h fields from creating a false
audit pass.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any


ART = Path("/home/chen/RLPFN/artifacts")
DEFAULT_INPUT_CSV = (
    ART
    / "phase2_stratified_realized_noise_zero_fixed_h_list_0518"
    / "stratified_realized_noise_zero_fixed_h_list.csv"
)
DEFAULT_OUTPUT_DIR = ART / "phase2_executable_formula_axis_fixed_h_list_0518"
RULE = "executable_terminal_survival_axis_v1"


TERMINAL_LABELS = [
    "terminal_off",
    "hazard_reset",
    "override_hazard_reset",
    "min_step_hazard_reset",
]
SURVIVAL_LABELS = ["survival_off", "survival_on"]
STRUCTURE_BIN_COLUMNS = [
    "action_dim_source_bin",
    "terminal_target_source_bin",
    "realized_noise_dim_bin",
    "realized_zero_pad_dim_bin",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


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


def _read_h(row: dict[str, str]) -> dict[str, Any]:
    raw = str(row.get("full_frozen_h_json", "")).strip()
    path = Path(raw).expanduser()
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return json.loads(raw)


def _cramers_v(xs: list[str], ys: list[str]) -> float:
    n = len(xs)
    if n == 0:
        return 0.0
    x_labels = sorted(set(xs))
    y_labels = sorted(set(ys))
    if len(x_labels) <= 1 or len(y_labels) <= 1:
        return 0.0
    xi = {label: idx for idx, label in enumerate(x_labels)}
    yi = {label: idx for idx, label in enumerate(y_labels)}
    table = [[0.0 for _ in y_labels] for _ in x_labels]
    for x, y in zip(xs, ys):
        table[xi[x]][yi[y]] += 1.0
    row_sum = [sum(row) for row in table]
    col_sum = [sum(table[i][j] for i in range(len(x_labels))) for j in range(len(y_labels))]
    chi2 = 0.0
    for i in range(len(x_labels)):
        for j in range(len(y_labels)):
            expected = row_sum[i] * col_sum[j] / n
            if expected > 0.0:
                chi2 += (table[i][j] - expected) ** 2 / expected
    denom = n * max(1, min(len(x_labels) - 1, len(y_labels) - 1))
    return math.sqrt(max(0.0, chi2 / denom)) if denom > 0 else 0.0


def _assign_labels(n: int, labels: list[str], seed: int) -> list[str]:
    order = list(range(n))
    random.Random(int(seed)).shuffle(order)
    out = [labels[0] for _ in range(n)]
    for rank, idx in enumerate(order):
        out[idx] = labels[rank % len(labels)]
    return out


def _apply_terminal_formula(h: dict[str, Any], label: str, min_step_target: float) -> None:
    h["terminal_formula_axis_target"] = label
    h["exact_scm_gym_lowtail_shared_energy_terminal_enabled"] = False
    if label == "terminal_off":
        h["terminal_reset_enabled"] = False
        h["exact_scm_gym_lowtail_terminal_override_enabled"] = False
        h["terminal_reset_min_step_enabled"] = False
        h["terminal_reset_min_step_target"] = 0
    elif label == "hazard_reset":
        h["terminal_reset_enabled"] = True
        h["exact_scm_gym_lowtail_terminal_override_enabled"] = False
        h["terminal_reset_min_step_enabled"] = False
        h["terminal_reset_min_step_target"] = 0
    elif label == "override_hazard_reset":
        h["terminal_reset_enabled"] = True
        h["exact_scm_gym_lowtail_terminal_override_enabled"] = True
        h["terminal_reset_min_step_enabled"] = False
        h["terminal_reset_min_step_target"] = 0
    elif label == "min_step_hazard_reset":
        h["terminal_reset_enabled"] = True
        h["exact_scm_gym_lowtail_terminal_override_enabled"] = False
        h["terminal_reset_min_step_enabled"] = True
        h["terminal_reset_min_step_target"] = float(min_step_target)
    else:
        raise ValueError(f"unknown terminal label: {label}")


def _apply_survival_component(h: dict[str, Any], label: str, weight: float) -> None:
    h["reward_component_axis_target"] = label
    if label == "survival_on":
        h["survival_reward_weight"] = float(weight)
        h["survival_reward_enable_prob"] = 1.0
        h["_survival_reward_enable_u"] = 0.0
    elif label == "survival_off":
        h["survival_reward_weight"] = 0.0
        h["survival_reward_enable_prob"] = 0.0
        h["_survival_reward_enable_u"] = 1.0
    else:
        raise ValueError(f"unknown survival label: {label}")


def _summarize_coupling(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for formula_axis in ["terminal_formula_axis_target", "reward_component_axis_target"]:
        xs = [str(row[formula_axis]) for row in rows]
        for struct_axis in STRUCTURE_BIN_COLUMNS:
            ys = [str(row.get(struct_axis, "missing")) for row in rows]
            v = _cramers_v(xs, ys)
            if v >= 0.50:
                status = "high"
            elif v >= 0.25:
                status = "medium"
            else:
                status = "low"
            out.append(
                {
                    "formula_axis": formula_axis,
                    "structure_axis": struct_axis,
                    "cramers_v": v,
                    "status": status,
                }
            )
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    input_csv = Path(args.input_csv).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    h_dir = out_dir / "fixed_h"
    h_dir.mkdir(parents=True, exist_ok=True)

    input_rows = _read_csv(input_csv)
    terminal_labels = _assign_labels(len(input_rows), TERMINAL_LABELS, int(args.seed))
    survival_labels = _assign_labels(len(input_rows), SURVIVAL_LABELS, int(args.seed) + 17)

    output_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(input_rows):
        h = _read_h(row)
        h["formula_axis_repair_rule"] = RULE
        h["lowtail_dormant_formula_axes_not_repaired"] = True
        _apply_terminal_formula(h, terminal_labels[idx], float(args.min_step_target))
        _apply_survival_component(h, survival_labels[idx], float(args.survival_weight))

        h_path = h_dir / f"executable_formula_axis_env_{idx:04d}.json"
        h_path.write_text(json.dumps(_json_safe(h), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        out_row: dict[str, Any] = dict(row)
        out_row["env_idx"] = idx
        out_row["rule"] = RULE
        out_row["full_frozen_h_json"] = str(h_path)
        out_row["terminal_formula_axis_target"] = terminal_labels[idx]
        out_row["reward_component_axis_target"] = survival_labels[idx]
        output_rows.append(out_row)

    csv_path = out_dir / "executable_formula_axis_fixed_h_list.csv"
    _write_csv(csv_path, output_rows)

    coupling = _summarize_coupling(output_rows)
    report = {
        "analysis_entry": "phase2_executable_formula_axis_fixed_h_list_builder",
        "schema": "phase2_executable_formula_axis_fixed_h_list_builder.v1",
        "contract": {
            "does_not_modify_generator": True,
            "fixed_h_list_materialization_only": True,
            "uses_only_current_maintained_exact_scm_executable_axes": True,
            "does_not_use_lowtail_dormant_fields_as_audit_pass": True,
        },
        "inputs": {"input_csv": str(input_csv)},
        "outputs": {
            "fixed_csv": str(csv_path),
            "fixed_h_dir": str(h_dir),
            "json": str(out_dir / "executable_formula_axis_fixed_h_list_report.json"),
            "markdown": str(out_dir / "executable_formula_axis_fixed_h_list_report.md"),
            "coupling_csv": str(out_dir / "executable_formula_axis_coupling.csv"),
        },
        "read": {
            "terminal_formula_counts": dict(Counter(terminal_labels)),
            "reward_component_counts": dict(Counter(survival_labels)),
            "high_structure_formula_coupling_count": sum(1 for row in coupling if row["status"] == "high"),
            "lowtail_only_axes_left_unrepaired": [
                "shared_energy_terminal",
                "potential_delta",
                "distribution_head",
            ],
            "lowtail_only_reason": (
                "these fields are dormant in the current maintained Exact-SCM runner image unless "
                "exact_scm_gym_lowtail_family is active"
            ),
        },
    }
    _write_csv(Path(report["outputs"]["coupling_csv"]), coupling)
    Path(report["outputs"]["json"]).write_text(
        json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Executable Formula-Axis Fixed H-List",
        "",
        f"- output fixed csv: `{csv_path}`",
        f"- terminal formula counts: `{report['read']['terminal_formula_counts']}`",
        f"- reward component counts: `{report['read']['reward_component_counts']}`",
        f"- high structure-formula coupling count: `{report['read']['high_structure_formula_coupling_count']}`",
        f"- lowtail-only axes left unrepaired: `{report['read']['lowtail_only_axes_left_unrepaired']}`",
        f"- reason: {report['read']['lowtail_only_reason']}",
    ]
    Path(report["outputs"]["markdown"]).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report["outputs"], sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", default=str(DEFAULT_INPUT_CSV))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--seed", type=int, default=9518)
    parser.add_argument("--survival-weight", type=float, default=0.02)
    parser.add_argument("--min-step-target", type=float, default=8.0)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
