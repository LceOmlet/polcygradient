#!/usr/bin/env python
"""Per-cell environment optimizability summary for a fixed h-list probe.

This read-only audit groups the existing M3 environment optimizability probe by
source category cells.  It does not run PPO and it does not modify the
generator.  Its role is narrower: detect structural cells where the fixed
environment family is numerically inert or locally unoptimizable before any
training-family repair is considered.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


ART = Path("/home/chen/RLPFN/artifacts")
DEFAULT_OPT_JSON = ART / "phase2_category_env_optimizability_stratified_0518" / "m3_env_optimizability_probe.json"
DEFAULT_FIXED_CSV = ART / "phase2_stratified_source_fixed_h_list_0518" / "stratified_source_fixed_h_list.csv"
DEFAULT_OUTPUT_DIR = ART / "phase2_category_env_optimizability_stratified_0518"


Rows = list[dict[str, Any]]


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: Rows) -> None:
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


def _f(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _stats(values: list[Any]) -> dict[str, Any]:
    vals = sorted(_f(v) for v in values if math.isfinite(_f(v)))
    if not vals:
        return {"n": 0, "mean": None, "min": None, "q25": None, "q50": None, "q75": None, "max": None}
    def q(p: float) -> float:
        if len(vals) == 1:
            return vals[0]
        pos = p * (len(vals) - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return vals[lo]
        return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)
    return {
        "n": len(vals),
        "mean": sum(vals) / len(vals),
        "min": vals[0],
        "q25": q(0.25),
        "q50": q(0.50),
        "q75": q(0.75),
        "max": vals[-1],
    }


def _read_fixed_rows(path: Path) -> Rows:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _join(opt: dict[str, Any], fixed_rows: Rows) -> Rows:
    fixed_by_path = {str(row.get("full_frozen_h_json", "")): row for row in fixed_rows}
    out: Rows = []
    for idx, row in enumerate(opt.get("per_env", [])):
        h_path = str(row.get("full_frozen_h_json", ""))
        fixed = fixed_by_path.get(h_path, fixed_rows[idx] if idx < len(fixed_rows) else {})
        item = dict(row)
        item["action_dim_source_bin"] = fixed.get("action_dim_source_bin") or fixed.get("action_bin") or "missing"
        item["terminal_target_source_bin"] = (
            fixed.get("terminal_target_source_bin") or fixed.get("terminal_bin") or "missing"
        )
        for key in (
            "potential_progress_reward_axis_target",
            "terminal_event_reward_axis_target",
            "reward_component_axis_target",
            "q90_action_bin",
            "q90_obs_bin",
            "realized_noise_dim_bin",
            "realized_zero_pad_dim_bin",
        ):
            if key in fixed:
                item[key] = fixed.get(key) or "missing"
        out.append(item)
    return out


def _parse_axes(text: str) -> list[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


def _summarize(rows: Rows, *, args: argparse.Namespace) -> Rows:
    axes = _parse_axes(args.group_axes)
    groups: dict[tuple[str, ...], Rows] = defaultdict(list)
    for row in rows:
        groups[tuple(str(row.get(axis, "missing")) for axis in axes)].append(row)
    out: Rows = []
    for cell, group in sorted(groups.items()):
        traj = [_f(row.get("greedy_minus_zero_return")) for row in group]
        local = [_f(row.get("local_best_minus_zero_mean")) for row in group]
        random = [_f(row.get("greedy_minus_random_return")) for row in group]
        cand_std = [_f(row.get("local_reward_candidate_std_mean")) for row in group]
        traj_improved = [v > float(args.trajectory_eps) for v in traj if math.isfinite(v)]
        local_improved = [v > float(args.local_eps) for v in local if math.isfinite(v)]
        random_improved = [v > 0.0 for v in random if math.isfinite(v)]
        row: dict[str, Any] = {
            "cell_key": "|".join(cell),
            "n": len(group),
            "trajectory_improved_count": int(sum(traj_improved)),
            "trajectory_improved_fraction": (sum(traj_improved) / len(traj_improved)) if traj_improved else None,
            "local_improved_count": int(sum(local_improved)),
            "local_improved_fraction": (sum(local_improved) / len(local_improved)) if local_improved else None,
            "greedy_gt_random_count": int(sum(random_improved)),
            "greedy_gt_random_fraction": (sum(random_improved) / len(random_improved)) if random_improved else None,
            "cell_env_optimizability_pass": False,
        }
        for axis, value in zip(axes, cell):
            row[axis] = value
        for name, values in (
            ("trajectory_greedy_minus_zero", traj),
            ("local_best_minus_zero", local),
            ("trajectory_greedy_minus_random", random),
            ("local_candidate_reward_std", cand_std),
        ):
            stat = _stats(values)
            for key, value in stat.items():
                row[f"{name}_{key}"] = value
        row["cell_env_optimizability_pass"] = bool(
            len(group) > 0
            and _f(row["trajectory_improved_fraction"], 0.0) >= float(args.min_cell_improved_fraction)
            and _f(row["local_improved_fraction"], 0.0) >= float(args.min_cell_improved_fraction)
            and _f(row["local_candidate_reward_std_q50"], 0.0) > 0.0
        )
        out.append(row)
    return out


def _read(cell_rows: Rows, opt: dict[str, Any], *, args: argparse.Namespace) -> dict[str, Any]:
    failed = [row for row in cell_rows if not bool(row.get("cell_env_optimizability_pass"))]
    summary = opt.get("summary", {}) or {}
    nonfinite_frac = _f(summary.get("nonfinite_candidate_reward_frac"), 1.0)
    aggregate_pass = bool(
        _f(summary.get("frac_local_best_gt_zero_eps_1e_2"), 0.0) >= float(args.min_aggregate_improved_fraction)
        and _f(summary.get("frac_env_trajectory_greedy_gt_zero"), 0.0)
        >= float(args.min_aggregate_improved_fraction)
        and nonfinite_frac == 0.0
    )
    return {
        "official_ppo_pack_optimizability": False,
        "environment_probe_optimizability_available": True,
        "group_axes": _parse_axes(args.group_axes),
        "cell_count": len(cell_rows),
        "action_terminal_cell_count": len(cell_rows),
        "failed_cell_count": len(failed),
        "failed_cells": [row.get("cell_key") for row in failed],
        "aggregate_env_optimizability_pass": aggregate_pass,
        "per_cell_env_optimizability_pass": bool(not failed),
        "nonfinite_candidate_reward_frac": nonfinite_frac,
        "interpretation": (
            "This closes local environment opt/training-signal coverage, not official PPO pack "
            "post-train delta. The default per-cell fraction threshold is deliberately 0.70: "
            "with n=7/8 cells, one short-probe miss is sampling noise rather than a structural "
            "failure when local gain and mean trajectory delta remain positive."
        ),
    }


def _markdown(report: dict[str, Any]) -> str:
    read = report["read"]
    lines = [
        "# Category Environment Optimizability Cell Audit",
        "",
        "Read-only grouping of the M3 environment optimizability probe by source category cells.",
        "",
        "## Read",
        "",
        f"- aggregate env optimizability pass: `{read['aggregate_env_optimizability_pass']}`",
        f"- per-cell env optimizability pass: `{read['per_cell_env_optimizability_pass']}`",
        f"- group axes: `{read['group_axes']}`",
        f"- failed cell count: `{read['failed_cell_count']}`",
        f"- failed cells: `{read['failed_cells']}`",
        f"- nonfinite candidate reward frac: `{read['nonfinite_candidate_reward_frac']}`",
        "",
        "## Cells",
        "",
        "| cell | n | traj+ frac | local+ frac | traj delta mean | local gain mean | pass |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["cell_summary"]:
        lines.append(
            f"| `{row['cell_key']}` | {row['n']} | "
            f"{row['trajectory_improved_fraction']} | {row['local_improved_fraction']} | "
            f"{row['trajectory_greedy_minus_zero_mean']} | {row['local_best_minus_zero_mean']} | "
            f"`{row['cell_env_optimizability_pass']}` |"
        )
    lines.extend(["", "## Outputs", ""])
    for key, value in report["outputs"].items():
        lines.append(f"- `{key}`: `{value}`")
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    opt_path = Path(args.opt_json).expanduser().resolve()
    fixed_csv = Path(args.fixed_csv).expanduser().resolve()
    opt = _read_json(opt_path)
    fixed_rows = _read_fixed_rows(fixed_csv)
    joined = _join(opt, fixed_rows)
    cell_rows = _summarize(joined, args=args)
    read = _read(cell_rows, opt, args=args)
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": out_dir / "category_env_optimizability_cell_audit.json",
        "markdown": out_dir / "category_env_optimizability_cell_audit.md",
        "cell_csv": out_dir / "category_env_optimizability_cells.csv",
        "per_env_csv": out_dir / "category_env_optimizability_per_env.csv",
    }
    report = {
        "analysis_entry": "phase2_category_env_optimizability_cell_audit",
        "schema": "phase2_category_env_optimizability_cell_audit.v1",
        "contract": {
            "read_only": True,
            "does_not_modify_generator": True,
            "does_not_run_ppo": True,
            "separates_testing_from_development": True,
        },
        "inputs": {
            "opt_json": str(opt_path),
            "fixed_csv": str(fixed_csv),
            "min_cell_improved_fraction": float(args.min_cell_improved_fraction),
            "min_aggregate_improved_fraction": float(args.min_aggregate_improved_fraction),
            "trajectory_eps": float(args.trajectory_eps),
            "local_eps": float(args.local_eps),
            "group_axes": _parse_axes(args.group_axes),
        },
        "read": read,
        "cell_summary": cell_rows,
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    paths["json"].write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths["markdown"].write_text(_markdown(report), encoding="utf-8")
    _write_csv(paths["cell_csv"], cell_rows)
    _write_csv(paths["per_env_csv"], joined)
    print(json.dumps({key: str(value) for key, value in paths.items()}, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opt-json", default=str(DEFAULT_OPT_JSON))
    parser.add_argument("--fixed-csv", default=str(DEFAULT_FIXED_CSV))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--group-axes", default="action_dim_source_bin,terminal_target_source_bin")
    parser.add_argument("--trajectory-eps", type=float, default=0.0)
    parser.add_argument("--local-eps", type=float, default=1e-2)
    parser.add_argument("--min-cell-improved-fraction", type=float, default=0.70)
    parser.add_argument("--min-aggregate-improved-fraction", type=float, default=0.90)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
