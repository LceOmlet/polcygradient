#!/usr/bin/env python3
"""Materialize survival source labels onto an existing fixed h-list.

This is a gate artifact builder, not a training runner. It is only authorized
after the true-runner survival component reconstruction audit proves that the
component is additive and preserves the base environment reward path.
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
    / "phase2_potential_progress_source_label_fixed_h_list_0518"
    / "potential_progress_source_label_fixed_h_list.csv"
)
DEFAULT_AUDIT_JSON = (
    ART
    / "phase2_survival_component_reconstruction_audit_0519"
    / "survival_component_reconstruction_audit.json"
)
DEFAULT_OUTPUT_DIR = ART / "phase2_survival_source_label_fixed_h_list_0519"

LABELS = ("survival_off", "survival_on")
RULE = "m4_small_additive_source_label_v1"
COUPLING_AXES = (
    "action_dim_source_bin",
    "state_dim_source_bin",
    "obs_dim_source_bin",
    "terminal_target_source_bin",
    "realized_noise_dim_bin",
    "realized_zero_pad_dim_bin",
    "terminal_event_reward_axis_target",
    "reward_component_axis_target",
    "terminal_formula_axis_target",
    "potential_progress_reward_axis_target",
    "q90_action_bin",
    "q90_obs_bin",
)


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


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))


def _read_h(row: dict[str, str]) -> dict[str, Any]:
    raw = str(row.get("full_frozen_h_json", "")).strip()
    path = Path(raw).expanduser()
    if path.exists():
        return _read_json(path)
    return json.loads(raw)


def _require_survival_audit(path: Path) -> dict[str, Any]:
    audit = _read_json(path)
    read = audit.get("read", {}) or {}
    required = (
        "hard_pass",
        "component_logging_pass",
        "finite_pass",
        "reconstruction_pass",
        "survival_formula_pass",
        "survival_observed",
        "base_preservation_pass",
        "source_label_sampler_authorized_after_this_audit",
    )
    failed = [key for key in required if not bool(read.get(key))]
    if failed:
        raise RuntimeError(f"survival source-label h-list is not authorized; failed={failed}")
    return audit


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


def _summarize_coupling(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    xs = [str(row.get("survival_component_axis_target", "missing")) for row in rows]
    out: list[dict[str, Any]] = []
    for axis in COUPLING_AXES:
        ys = [str(row.get(axis, "missing") or "missing") for row in rows]
        v = _cramers_v(xs, ys)
        out.append(
            {
                "source_axis": "survival_component_axis_target",
                "other_axis": axis,
                "cramers_v": v,
                "status": "high" if v >= 0.50 else ("medium" if v >= 0.25 else "low"),
            }
        )
    return out


def _best_labels(rows: list[dict[str, str]], seed: int, trials: int) -> list[str]:
    best = _assign_labels(len(rows), seed)
    best_score: tuple[int, int, float, float] | None = None
    rng = random.Random(int(seed))
    for trial in range(max(1, int(trials))):
        labels = _assign_labels(len(rows), int(seed) + 65537 * trial + rng.randrange(65537))
        tmp = [dict(row, survival_component_axis_target=labels[idx]) for idx, row in enumerate(rows)]
        coupling = _summarize_coupling(tmp)
        high = sum(1 for row in coupling if row["status"] == "high")
        medium = sum(1 for row in coupling if row["status"] == "medium")
        vals = [float(row["cramers_v"]) for row in coupling if math.isfinite(float(row["cramers_v"]))]
        score = (high, medium, max(vals or [0.0]), sum(vals))
        if best_score is None or score < best_score:
            best_score = score
            best = labels
            if high == 0 and medium == 0:
                break
    return best


def _survival_weight(label: str, *, rng: random.Random, weight_min: float, weight_max: float) -> float:
    if label == "survival_off" or weight_max <= 0.0:
        return 0.0
    lo = max(float(weight_min), 1e-12)
    hi = max(float(weight_max), lo)
    return math.exp(math.log(lo) + rng.random() * (math.log(hi) - math.log(lo)))


def run(args: argparse.Namespace) -> dict[str, Any]:
    input_csv = Path(args.input_csv).expanduser().resolve()
    audit_path = Path(args.audit_json).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    h_dir = out_dir / "fixed_h"
    h_dir.mkdir(parents=True, exist_ok=True)
    audit = _require_survival_audit(audit_path)
    rows = _read_csv(input_csv)
    labels = _best_labels(rows, int(args.seed), int(args.assignment_trials))
    rng = random.Random(int(args.seed) ^ 0x5A17)
    out_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        h = _read_h(row)
        label = labels[idx]
        weight = _survival_weight(
            label,
            rng=rng,
            weight_min=float(args.weight_min),
            weight_max=float(args.weight_max),
        )
        h["survival_component_axis_target"] = label
        h["survival_component_axis_rule"] = RULE
        h["survival_component_preserves_base_reward"] = True
        h["survival_component_audit_schema"] = "phase2_survival_component_reconstruction_audit.v1"
        h["survival_reward_weight"] = float(weight)
        h["survival_reward_enable_prob"] = 1.0 if label == "survival_on" else 0.0
        h["_survival_reward_enable_u"] = 0.0 if label == "survival_on" else 1.0
        h_path = h_dir / f"survival_source_env_{idx:04d}.json"
        h_path.write_text(json.dumps(_json_safe(h), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        out_row: dict[str, Any] = dict(row)
        out_row["full_frozen_h_json"] = str(h_path)
        out_row["survival_component_axis_target"] = label
        out_row["survival_component_axis_rule"] = RULE
        out_row["survival_reward_weight"] = float(weight)
        out_row["survival_component_preserves_base_reward"] = True
        out_rows.append(out_row)
    coupling = _summarize_coupling(out_rows)
    fixed_csv = out_dir / "survival_source_label_fixed_h_list.csv"
    coupling_csv = out_dir / "survival_source_label_coupling.csv"
    report_path = out_dir / "survival_source_label_fixed_h_list_report.json"
    md_path = out_dir / "survival_source_label_fixed_h_list_report.md"
    _write_csv(fixed_csv, out_rows)
    _write_csv(coupling_csv, coupling)
    report = {
        "analysis_entry": "phase2_survival_source_label_fixed_h_list_builder",
        "schema": "phase2_survival_source_label_fixed_h_list_builder.v1",
        "contract": {
            "fixed_h_list_materialization_only": True,
            "does_not_modify_generator": True,
            "does_not_modify_ppo": True,
            "does_not_train": True,
            "survival_formula_path_pre_authorized": True,
            "base_reward_preserved_by_additive_component": True,
        },
        "inputs": {
            "input_csv": str(input_csv),
            "audit_json": str(audit_path),
            "seed": int(args.seed),
            "weight_min": float(args.weight_min),
            "weight_max": float(args.weight_max),
        },
        "outputs": {
            "fixed_csv": str(fixed_csv),
            "fixed_h_dir": str(h_dir),
            "coupling_csv": str(coupling_csv),
            "json": str(report_path),
            "markdown": str(md_path),
        },
        "read": {
            "audit_hard_pass": bool((audit.get("read", {}) or {}).get("hard_pass")),
            "label_counts": dict(Counter(labels)),
            "high_coupling_count": sum(1 for row in coupling if row["status"] == "high"),
            "medium_coupling_count": sum(1 for row in coupling if row["status"] == "medium"),
            "source_label_sampler_materialized": True,
        },
        "coupling": coupling,
    }
    report_path.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(
        "\n".join(
            [
                "# Survival Source-Label Fixed H-List",
                "",
                f"- fixed csv: `{fixed_csv}`",
                f"- label counts: `{report['read']['label_counts']}`",
                f"- high coupling count: `{report['read']['high_coupling_count']}`",
                f"- medium coupling count: `{report['read']['medium_coupling_count']}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["read"], indent=2, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--audit-json", type=Path, default=DEFAULT_AUDIT_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=51972)
    parser.add_argument("--assignment-trials", type=int, default=4096)
    parser.add_argument("--weight-min", type=float, default=0.005)
    parser.add_argument("--weight-max", type=float, default=0.03)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
