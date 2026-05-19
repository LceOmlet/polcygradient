#!/usr/bin/env python3
"""Materialize a minimal terminal-event reward source-label fixed h-list.

This is not a generator rewrite.  It only toggles the event-reward channel
attached to already-existing terminal events:

* off: terminal events may still occur, but terminal bonus scale is zero.
* on: terminal/reset structure is left unchanged and existing/default mild
  terminal bonus scales are used.

That keeps terminal/reset topology separate from terminal-event reward topology.
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
    / "phase2_executable_formula_axis_fixed_h_list_0518"
    / "executable_formula_axis_fixed_h_list.csv"
)
DEFAULT_OUTPUT_DIR = ART / "phase2_terminal_event_source_label_fixed_h_list_0518"
LABELS = ["terminal_event_reward_off", "terminal_event_reward_on"]
STRUCTURE_BIN_COLUMNS = [
    "action_dim_source_bin",
    "state_dim_source_bin",
    "obs_dim_source_bin",
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


def _assign_labels(n: int, seed: int) -> list[str]:
    order = list(range(n))
    random.Random(int(seed)).shuffle(order)
    out = [LABELS[0] for _ in range(n)]
    for rank, idx in enumerate(order):
        out[idx] = LABELS[rank % len(LABELS)]
    return out


def _cramers_v(xs: list[str], ys: list[str]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if x != "missing" and y != "missing"]
    if not pairs:
        return 0.0
    x_labels = sorted({x for x, _ in pairs})
    y_labels = sorted({y for _, y in pairs})
    if len(x_labels) <= 1 or len(y_labels) <= 1:
        return 0.0
    xi = {label: idx for idx, label in enumerate(x_labels)}
    yi = {label: idx for idx, label in enumerate(y_labels)}
    table = [[0.0 for _ in y_labels] for _ in x_labels]
    for x, y in pairs:
        table[xi[x]][yi[y]] += 1.0
    n = float(len(pairs))
    row_sum = [sum(row) for row in table]
    col_sum = [sum(table[i][j] for i in range(len(x_labels))) for j in range(len(y_labels))]
    chi2 = 0.0
    for i in range(len(x_labels)):
        for j in range(len(y_labels)):
            expected = row_sum[i] * col_sum[j] / n
            if expected > 0.0:
                chi2 += (table[i][j] - expected) ** 2 / expected
    denom = n * max(1, min(len(x_labels) - 1, len(y_labels) - 1))
    return math.sqrt(max(0.0, chi2 / denom)) if denom > 0.0 else 0.0


def _safe_float(value: Any, default: float) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _apply_terminal_event_reward_axis(h: dict[str, Any], label: str) -> None:
    h["terminal_event_reward_axis_target"] = str(label)
    h["terminal_event_reward_axis_rule"] = "terminal_bonus_scale_on_off_v1"
    h["terminal_event_reward_preserves_terminal_structure"] = True
    if label == "terminal_event_reward_off":
        h["terminal_bonus_scale_min_original"] = h.get("terminal_bonus_scale_min", 1.0)
        h["terminal_bonus_scale_max_original"] = h.get("terminal_bonus_scale_max", 2.0)
        h["terminal_bonus_scale_min"] = 0.0
        h["terminal_bonus_scale_max"] = 0.0
        return
    if label == "terminal_event_reward_on":
        lo = max(0.0, _safe_float(h.get("terminal_bonus_scale_min", 1.0), 1.0))
        hi = max(lo, _safe_float(h.get("terminal_bonus_scale_max", 2.0), 2.0))
        if hi <= 0.0:
            lo, hi = 1.0, 2.0
        h["terminal_bonus_scale_min"] = lo
        h["terminal_bonus_scale_max"] = hi
        return
    raise ValueError(f"unknown terminal event reward label: {label}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    input_csv = Path(args.input_csv).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    h_dir = out_dir / "fixed_h"
    h_dir.mkdir(parents=True, exist_ok=True)
    input_rows = _read_csv(input_csv)
    labels = _assign_labels(len(input_rows), int(args.seed))
    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(input_rows):
        h = _read_h(row)
        _apply_terminal_event_reward_axis(h, labels[idx])
        h_path = h_dir / f"terminal_event_source_env_{idx:04d}.json"
        h_path.write_text(json.dumps(_json_safe(h), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        out_row: dict[str, Any] = dict(row)
        out_row["env_idx"] = idx
        out_row["full_frozen_h_json"] = str(h_path)
        out_row["terminal_event_reward_axis_target"] = labels[idx]
        out_row["terminal_bonus_scale_min"] = h["terminal_bonus_scale_min"]
        out_row["terminal_bonus_scale_max"] = h["terminal_bonus_scale_max"]
        rows.append(out_row)

    coupling_rows: list[dict[str, Any]] = []
    xs = [str(row["terminal_event_reward_axis_target"]) for row in rows]
    for axis in STRUCTURE_BIN_COLUMNS:
        ys = [str(row.get(axis, "missing")) for row in rows]
        v = _cramers_v(xs, ys)
        coupling_rows.append(
            {
                "source_axis": "terminal_event_reward_axis_target",
                "structure_axis": axis,
                "cramers_v": v,
                "status": "high" if v >= 0.50 else ("medium" if v >= 0.25 else "low"),
            }
        )

    fixed_csv = out_dir / "terminal_event_source_label_fixed_h_list.csv"
    coupling_csv = out_dir / "terminal_event_source_label_coupling.csv"
    _write_csv(fixed_csv, rows)
    _write_csv(coupling_csv, coupling_rows)
    report = {
        "analysis_entry": "phase2_terminal_event_source_label_fixed_h_list_builder",
        "schema": "phase2_terminal_event_source_label_fixed_h_list_builder.v1",
        "contract": {
            "fixed_h_list_materialization_only": True,
            "does_not_modify_generator": True,
            "does_not_train": True,
            "terminal_structure_preserved": True,
            "source_axis": "terminal_event_reward_on_off",
        },
        "inputs": {"input_csv": str(input_csv)},
        "outputs": {
            "fixed_csv": str(fixed_csv),
            "fixed_h_dir": str(h_dir),
            "json": str(out_dir / "terminal_event_source_label_fixed_h_list_report.json"),
            "markdown": str(out_dir / "terminal_event_source_label_fixed_h_list_report.md"),
            "coupling_csv": str(coupling_csv),
        },
        "read": {
            "label_counts": dict(Counter(labels)),
            "high_structure_coupling_count": sum(1 for row in coupling_rows if row["status"] == "high"),
            "medium_structure_coupling_count": sum(1 for row in coupling_rows if row["status"] == "medium"),
            "source_label_sampler_materialized": True,
            "requires_next": [
                "true-runner sidecar rerun",
                "category span",
                "structure-mode independence",
                "pack-fit parity",
                "per-cell PPO health/optimizability",
            ],
        },
        "coupling": coupling_rows,
    }
    Path(report["outputs"]["json"]).write_text(
        json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Terminal Event Source-Label Fixed H-List",
        "",
        f"- fixed csv: `{fixed_csv}`",
        f"- label counts: `{report['read']['label_counts']}`",
        f"- high structure coupling count: `{report['read']['high_structure_coupling_count']}`",
        f"- medium structure coupling count: `{report['read']['medium_structure_coupling_count']}`",
        "- next: rerun true-runner sidecar and protection gates before milestone use.",
    ]
    Path(report["outputs"]["markdown"]).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report["read"], indent=2, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=9519)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
