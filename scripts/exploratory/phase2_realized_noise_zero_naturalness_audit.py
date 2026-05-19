#!/usr/bin/env python
"""Naturalness audit for realized noise_dim x zero_pad_dim cells.

This is a read-only diagnostic.  It answers a narrow question before any repair:

If realized noise/zero coverage is not uniform, are the covered cells also
numerically unnatural or unoptimizable?

The guard is intentionally conservative and not Gym-quantile based.  A cell is
"natural enough" here if it is finite, non-collapsed, and locally trainable:

* obs/reward/dynamics have nonzero support,
* action saturation is finite,
* environment optimizability probe has positive local and trajectory deltas.

Missing cells are reported separately.  A missing realized cell is a coverage
issue only if realized noise/zero is promoted to a hard category axis.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ART = Path("/home/chen/RLPFN/artifacts")
DEFAULT_RUN_DIR = ART / "phase2_stratified_source_fixed_h_list_runner_u2_s128_0518"
DEFAULT_ENV_OPT = ART / "phase2_category_env_optimizability_stratified_0518" / "m3_env_optimizability_probe.json"
DEFAULT_OUTPUT_DIR = ART / "phase2_realized_noise_zero_naturalness_stratified_0518"


Rows = list[dict[str, Any]]


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def _read_jsonl(path: str | Path) -> Rows:
    rows: Rows = []
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


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


def _latest_sidecar(run_dir: Path) -> Path:
    candidates = sorted((run_dir / "semantic_sidecars").glob("*_envs.jsonl"))
    if not candidates:
        raise FileNotFoundError(f"no sidecar found under {run_dir / 'semantic_sidecars'}")
    return candidates[-1]


def _bin3(value: Any, lo_hi: tuple[float, float] = (133.0, 266.0)) -> str:
    x = _f(value)
    if not math.isfinite(x):
        return "missing"
    if x <= lo_hi[0]:
        return "low"
    if x <= lo_hi[1]:
        return "mid"
    return "high"


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


def _join(side_rows: Rows, env_opt: dict[str, Any]) -> Rows:
    opt_rows = env_opt.get("per_env", [])
    opt_by_idx = {int(row.get("local_index", idx)): row for idx, row in enumerate(opt_rows)}
    out: Rows = []
    for idx, side in enumerate(side_rows):
        opt = opt_by_idx.get(int(side.get("env_idx", idx)), {})
        row = dict(side)
        row["noise_dim_bin"] = _bin3(side.get("noise_dim"))
        row["zero_pad_dim_bin"] = _bin3(side.get("zero_pad_dim"))
        for key, value in opt.items():
            row[f"opt:{key}"] = value
        out.append(row)
    return out


def _entropy(counts: Counter[str]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    out = 0.0
    for count in counts.values():
        if count <= 0:
            continue
        p = count / total
        out -= p * math.log(p)
    return out


def _coverage(rows: Rows) -> dict[str, Any]:
    bins = ["low", "mid", "high"]
    counts = Counter((row["noise_dim_bin"], row["zero_pad_dim_bin"]) for row in rows)
    expected = len(bins) * len(bins)
    occupied = sum(1 for a in bins for b in bins if counts.get((a, b), 0) > 0)
    pair_counts = Counter({f"{a}|{b}": counts.get((a, b), 0) for a in bins for b in bins})
    ent = _entropy(pair_counts) / math.log(expected)
    total = len(rows)
    return {
        "occupied_cells": occupied,
        "expected_cells": expected,
        "coverage_fraction": occupied / expected,
        "normalized_entropy": ent,
        "max_cell_fraction": max([counts.get((a, b), 0) / max(1, total) for a in bins for b in bins]),
        "missing_cells": [[a, b] for a in bins for b in bins if counts.get((a, b), 0) == 0],
    }


def _cell_rows(rows: Rows, *, args: argparse.Namespace) -> Rows:
    groups: dict[tuple[str, str], Rows] = defaultdict(list)
    for row in rows:
        groups[(row["noise_dim_bin"], row["zero_pad_dim_bin"])].append(row)
    out: Rows = []
    for noise_bin in ["low", "mid", "high"]:
        for zero_bin in ["low", "mid", "high"]:
            group = groups.get((noise_bin, zero_bin), [])
            item: dict[str, Any] = {
                "noise_dim_bin": noise_bin,
                "zero_pad_dim_bin": zero_bin,
                "n": len(group),
            }
            if not group:
                item["cell_naturalness_pass"] = False
                item["failure_reason"] = "missing_cell"
                out.append(item)
                continue

            fields = {
                "obs_per_dim_std_q90": [row.get("obs_per_dim_std_q90") for row in group],
                "pre_norm_obs_per_dim_std_q90": [row.get("pre_norm_obs_per_dim_std_q90") for row in group],
                "next_state_step_l2_delta_to_rms_ratio": [
                    row.get("next_state_step_l2_delta_to_rms_ratio") for row in group
                ],
                "raw_reward_std": [row.get("raw_reward_std") for row in group],
                "training_reward_std": [row.get("training_reward_std") for row in group],
                "action_saturation": [
                    row.get("raw_policy_action_unit_clip_saturation_fraction") for row in group
                ],
                "local_best_minus_zero": [row.get("opt:local_best_minus_zero_mean") for row in group],
                "trajectory_greedy_minus_zero": [row.get("opt:greedy_minus_zero_return") for row in group],
                "candidate_reward_std": [row.get("opt:local_reward_candidate_std_mean") for row in group],
            }
            for name, values in fields.items():
                stat = _stats(values)
                for key, value in stat.items():
                    item[f"{name}_{key}"] = value
            local_ok = sum(_f(v) > float(args.local_eps) for v in fields["local_best_minus_zero"]) / len(group)
            traj_ok = sum(_f(v) > float(args.trajectory_eps) for v in fields["trajectory_greedy_minus_zero"]) / len(group)
            obs_ok = _f(item.get("obs_per_dim_std_q90_q50"), 0.0) > float(args.min_obs_std_q50)
            dyn_ok = _f(item.get("next_state_step_l2_delta_to_rms_ratio_q50"), 0.0) > float(args.min_step_ratio_q50)
            reward_ok = _f(item.get("raw_reward_std_q50"), 0.0) > float(args.min_raw_reward_std_q50)
            candidate_ok = _f(item.get("candidate_reward_std_q50"), 0.0) > 0.0
            finite_ok = all(
                math.isfinite(_f(v))
                for values in fields.values()
                for v in values
            )
            item["local_improved_fraction"] = local_ok
            item["trajectory_improved_fraction"] = traj_ok
            item["obs_noncollapsed_pass"] = obs_ok
            item["dynamics_noncollapsed_pass"] = dyn_ok
            item["reward_noncollapsed_pass"] = reward_ok
            item["candidate_reward_noncollapsed_pass"] = candidate_ok
            item["finite_pass"] = finite_ok
            item["cell_naturalness_pass"] = bool(
                finite_ok
                and obs_ok
                and dyn_ok
                and reward_ok
                and candidate_ok
                and local_ok >= float(args.min_improved_fraction)
                and traj_ok >= float(args.min_improved_fraction)
            )
            reasons = []
            for key in (
                "finite_pass",
                "obs_noncollapsed_pass",
                "dynamics_noncollapsed_pass",
                "reward_noncollapsed_pass",
                "candidate_reward_noncollapsed_pass",
            ):
                if not item[key]:
                    reasons.append(key)
            if local_ok < float(args.min_improved_fraction):
                reasons.append("local_improved_fraction")
            if traj_ok < float(args.min_improved_fraction):
                reasons.append("trajectory_improved_fraction")
            item["failure_reason"] = ",".join(reasons)
            out.append(item)
    return out


def _read(cell_rows: Rows, coverage: dict[str, Any]) -> dict[str, Any]:
    covered = [row for row in cell_rows if int(row.get("n") or 0) > 0]
    covered_fail = [row for row in covered if not bool(row.get("cell_naturalness_pass"))]
    missing = [row for row in cell_rows if int(row.get("n") or 0) == 0]
    return {
        "realized_noise_zero_uniform_pass": bool(
            coverage["occupied_cells"] == coverage["expected_cells"]
            and float(coverage["normalized_entropy"]) >= 0.90
        ),
        "covered_cell_naturalness_pass": bool(not covered_fail),
        "covered_cell_naturalness_fail_count": len(covered_fail),
        "missing_cell_count": len(missing),
        "missing_cells": coverage["missing_cells"],
        "repair_needed_if_axis_hard": bool(missing or coverage["normalized_entropy"] < 0.90),
        "minimal_repair_if_axis_hard": (
            "stratify fixed-list selection by materialized realized noise_dim x zero_pad_dim cells; "
            "do not change Exact SCM formulas unless a covered cell also fails naturalness"
        ),
    }


def _markdown(report: dict[str, Any]) -> str:
    read = report["read"]
    lines = [
        "# Realized Noise/Zero Naturalness Audit",
        "",
        "Read-only diagnostic for realized noise_dim × zero_pad_dim cells.",
        "",
        "## Read",
        "",
        f"- realized noise/zero uniform pass: `{read['realized_noise_zero_uniform_pass']}`",
        f"- covered cell naturalness pass: `{read['covered_cell_naturalness_pass']}`",
        f"- covered cell naturalness fail count: `{read['covered_cell_naturalness_fail_count']}`",
        f"- missing cell count: `{read['missing_cell_count']}`",
        f"- missing cells: `{read['missing_cells']}`",
        f"- repair needed if axis hard: `{read['repair_needed_if_axis_hard']}`",
        f"- minimal repair if axis hard: {read['minimal_repair_if_axis_hard']}",
        "",
        "## Coverage",
        "",
        f"- occupied/expected: `{report['coverage']['occupied_cells']}/{report['coverage']['expected_cells']}`",
        f"- normalized entropy: `{report['coverage']['normalized_entropy']}`",
        f"- max cell fraction: `{report['coverage']['max_cell_fraction']}`",
        "",
        "## Cells",
        "",
        "| noise | zero | n | natural | local+ | traj+ | obs q50 | raw reward std q50 | reason |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in report["cell_summary"]:
        lines.append(
            f"| `{row['noise_dim_bin']}` | `{row['zero_pad_dim_bin']}` | {row['n']} | "
            f"`{row.get('cell_naturalness_pass')}` | {row.get('local_improved_fraction')} | "
            f"{row.get('trajectory_improved_fraction')} | {row.get('obs_per_dim_std_q90_q50')} | "
            f"{row.get('raw_reward_std_q50')} | `{row.get('failure_reason', '')}` |"
        )
    lines.extend(["", "## Outputs", ""])
    for key, value in report["outputs"].items():
        lines.append(f"- `{key}`: `{value}`")
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = Path(args.run_dir).expanduser().resolve()
    sidecar = Path(args.sidecar).expanduser().resolve() if args.sidecar else _latest_sidecar(run_dir)
    env_opt = _read_json(args.env_opt_json)
    side_rows = _read_jsonl(sidecar)
    rows = _join(side_rows, env_opt)
    coverage = _coverage(rows)
    cells = _cell_rows(rows, args=args)
    read = _read(cells, coverage)
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": out_dir / "realized_noise_zero_naturalness_audit.json",
        "markdown": out_dir / "realized_noise_zero_naturalness_audit.md",
        "cell_csv": out_dir / "realized_noise_zero_naturalness_cells.csv",
        "per_env_csv": out_dir / "realized_noise_zero_naturalness_per_env.csv",
    }
    report = {
        "analysis_entry": "phase2_realized_noise_zero_naturalness_audit",
        "schema": "phase2_realized_noise_zero_naturalness_audit.v1",
        "contract": {
            "read_only": True,
            "does_not_modify_generator": True,
            "does_not_run_ppo": True,
            "separates_testing_from_development": True,
        },
        "inputs": {
            "run_dir": str(run_dir),
            "sidecar": str(sidecar),
            "env_opt_json": str(Path(args.env_opt_json).expanduser().resolve()),
        },
        "read": read,
        "coverage": coverage,
        "cell_summary": cells,
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    paths["json"].write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths["markdown"].write_text(_markdown(report), encoding="utf-8")
    _write_csv(paths["cell_csv"], cells)
    _write_csv(paths["per_env_csv"], rows)
    print(json.dumps(report["outputs"], sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--sidecar", default=None)
    parser.add_argument("--env-opt-json", default=str(DEFAULT_ENV_OPT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--min-obs-std-q50", type=float, default=1e-3)
    parser.add_argument("--min-step-ratio-q50", type=float, default=1e-6)
    parser.add_argument("--min-raw-reward-std-q50", type=float, default=1e-6)
    parser.add_argument("--local-eps", type=float, default=1e-2)
    parser.add_argument("--trajectory-eps", type=float, default=0.0)
    parser.add_argument("--min-improved-fraction", type=float, default=0.75)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
