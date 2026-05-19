#!/usr/bin/env python
"""Audit that hard reward-component coverage does not suppress env reward.

The hard component topology is the source-level boolean product:

    base_env_reward always present, ctrl_cost on/off, survival_bonus on/off.

``reward_aux`` is intentionally not an independent source axis; it is the
logged aggregate ``reward_ctrl + reward_survival``.  This audit checks that the
auxiliary components remain auxiliary after enforcing component coverage.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ART = Path("/home/chen/RLPFN/artifacts")
DEFAULT_RUN_DIR = ART / "phase2_executable_formula_axis_runner_u2_s128_0518"
DEFAULT_OUTPUT_DIR = ART / "phase2_reward_component_env_preservation_audit_0518"


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _latest_sidecar(run_dir: Path) -> Path:
    candidates = sorted((run_dir / "semantic_sidecars").glob("*_envs.jsonl"))
    if not candidates:
        raise FileNotFoundError(f"no semantic sidecar under {run_dir / 'semantic_sidecars'}")
    return candidates[-1]


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
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _nonzero(row: dict[str, Any], prefix: str, eps: float = 1e-8) -> bool:
    for suffix in ("std", "sum", "mean", "q90", "q50", "min", "max"):
        if abs(_safe_float(row.get(f"{prefix}_{suffix}"), 0.0)) > eps:
            return True
    return _safe_bool(row.get(f"{prefix}_q90_gt0"))


def _component_label(row: dict[str, Any]) -> str:
    parts = []
    if _nonzero(row, "reward_env") or _nonzero(row, "raw_reward"):
        parts.append("env")
    if _nonzero(row, "reward_ctrl"):
        parts.append("ctrl")
    if _nonzero(row, "reward_survival"):
        parts.append("survival")
    return "+".join(parts) if parts else "zero"


def _stats(values: list[float]) -> dict[str, Any]:
    vals = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=np.float64)
    if vals.size == 0:
        return {"n": 0, "mean": None, "q50": None, "q90": None, "max": None}
    return {
        "n": int(vals.size),
        "mean": float(vals.mean()),
        "q50": float(np.quantile(vals, 0.50)),
        "q90": float(np.quantile(vals, 0.90)),
        "max": float(vals.max()),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = Path(args.run_dir).expanduser().resolve()
    sidecar = Path(args.sidecar).expanduser().resolve() if args.sidecar else _latest_sidecar(run_dir)
    side_rows = _read_jsonl(sidecar)
    per_env: list[dict[str, Any]] = []
    for idx, row in enumerate(side_rows):
        env_std = _safe_float(row.get("reward_env_std"), 0.0)
        aux_std = _safe_float(row.get("reward_aux_std"), 0.0)
        env_scale = abs(_safe_float(row.get("reward_env_mean"), 0.0)) + env_std
        aux_scale = abs(_safe_float(row.get("reward_aux_mean"), 0.0)) + aux_std
        item = {
            "env_idx": int(row.get("env_idx", idx)),
            "reward_component_topology": _component_label(row),
            "base_env_reward_present": _nonzero(row, "reward_env") or _nonzero(row, "raw_reward"),
            "ctrl_present": _nonzero(row, "reward_ctrl"),
            "survival_present": _nonzero(row, "reward_survival"),
            "aux_present": _nonzero(row, "reward_aux"),
            "reward_env_std": env_std,
            "reward_aux_std": aux_std,
            "aux_std_to_env_std_ratio": aux_std / max(env_std, 1e-9),
            "reward_env_absmean_plus_std": env_scale,
            "reward_aux_absmean_plus_std": aux_scale,
            "aux_scale_to_env_scale_ratio": aux_scale / max(env_scale, 1e-9),
        }
        per_env.append(item)

    by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in per_env:
        by_cell[str(row["reward_component_topology"])].append(row)
    expected = ["env", "env+ctrl", "env+survival", "env+ctrl+survival"]
    cell_rows: list[dict[str, Any]] = []
    for cell in expected:
        rows = by_cell.get(cell, [])
        cell_rows.append(
            {
                "cell": cell,
                "n": len(rows),
                "base_env_present_count": sum(1 for row in rows if row["base_env_reward_present"]),
                "aux_std_to_env_std_ratio_q90": _stats([row["aux_std_to_env_std_ratio"] for row in rows])["q90"],
                "aux_std_to_env_std_ratio_max": _stats([row["aux_std_to_env_std_ratio"] for row in rows])["max"],
                "aux_scale_to_env_scale_ratio_q90": _stats([row["aux_scale_to_env_scale_ratio"] for row in rows])["q90"],
                "aux_scale_to_env_scale_ratio_max": _stats([row["aux_scale_to_env_scale_ratio"] for row in rows])["max"],
            }
        )

    counts = Counter(str(row["reward_component_topology"]) for row in per_env)
    base_env_all = all(row["base_env_reward_present"] for row in per_env)
    aux_std_ratios = [float(row["aux_std_to_env_std_ratio"]) for row in per_env]
    aux_scale_ratios = [float(row["aux_scale_to_env_scale_ratio"]) for row in per_env]
    no_aux_suppression = bool(
        base_env_all
        and max(aux_std_ratios or [0.0]) < float(args.max_aux_std_to_env_std)
        and max(aux_scale_ratios or [0.0]) < float(args.max_aux_scale_to_env_scale)
    )
    report = {
        "analysis_entry": "phase2_reward_component_env_preservation_audit",
        "schema": "phase2_reward_component_env_preservation_audit.v1",
        "contract": {
            "read_only": True,
            "does_not_modify_generator": True,
            "does_not_train": True,
            "reward_aux_is_aggregate_not_source_axis": True,
            "hard_component_axis": "base_env_always_present x ctrl_on_off x survival_on_off",
        },
        "read": {
            "component_counts": {cell: int(counts.get(cell, 0)) for cell in expected},
            "component_hard_axis_covered": all(counts.get(cell, 0) > 0 for cell in expected),
            "base_env_reward_present_all": base_env_all,
            "no_auxiliary_suppression_pass": no_aux_suppression,
            "max_aux_std_to_env_std_ratio": max(aux_std_ratios or [0.0]),
            "max_aux_scale_to_env_scale_ratio": max(aux_scale_ratios or [0.0]),
            "thresholds": {
                "max_aux_std_to_env_std": float(args.max_aux_std_to_env_std),
                "max_aux_scale_to_env_scale": float(args.max_aux_scale_to_env_scale),
            },
        },
        "cell_summary": cell_rows,
        "per_env": per_env,
        "inputs": {"run_dir": str(run_dir), "sidecar": str(sidecar)},
    }
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": out_dir / "reward_component_env_preservation_audit.json",
        "markdown": out_dir / "reward_component_env_preservation_audit.md",
        "per_env_csv": out_dir / "reward_component_env_preservation_per_env.csv",
        "cell_csv": out_dir / "reward_component_env_preservation_cell_summary.csv",
    }
    report["outputs"] = {key: str(value) for key, value in paths.items()}
    paths["json"].write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(paths["per_env_csv"], per_env)
    _write_csv(paths["cell_csv"], cell_rows)
    paths["markdown"].write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report["outputs"], sort_keys=True))
    return report


def _markdown(report: dict[str, Any]) -> str:
    read = report["read"]
    lines = [
        "# Reward Component Env Preservation Audit",
        "",
        "Read-only audit that ctrl/survival component coverage does not suppress base env reward.",
        "",
        "## Read",
        "",
        f"- component counts: `{read['component_counts']}`",
        f"- component hard axis covered: `{read['component_hard_axis_covered']}`",
        f"- base env reward present all: `{read['base_env_reward_present_all']}`",
        f"- no auxiliary suppression pass: `{read['no_auxiliary_suppression_pass']}`",
        f"- max aux std/env std ratio: `{read['max_aux_std_to_env_std_ratio']}`",
        f"- max aux scale/env scale ratio: `{read['max_aux_scale_to_env_scale_ratio']}`",
        f"- thresholds: `{read['thresholds']}`",
        "",
        "## Cell Summary",
        "",
        "| cell | n | env present | aux std/env q90 | aux std/env max | aux scale/env q90 | aux scale/env max |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["cell_summary"]:
        lines.append(
            f"| `{row['cell']}` | {row['n']} | {row['base_env_present_count']} | "
            f"{row['aux_std_to_env_std_ratio_q90']} | {row['aux_std_to_env_std_ratio_max']} | "
            f"{row['aux_scale_to_env_scale_ratio_q90']} | {row['aux_scale_to_env_scale_ratio_max']} |"
        )
    lines.extend(["", "## Outputs", ""])
    for key, value in report["outputs"].items():
        lines.append(f"- `{key}`: `{value}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--sidecar", default="")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max-aux-std-to-env-std", type=float, default=1.0)
    parser.add_argument("--max-aux-scale-to-env-scale", type=float, default=1.0)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
