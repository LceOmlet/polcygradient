#!/usr/bin/env python
"""Per-cell sidecar delta audit for a fixed h-list runner.

This is intentionally read-only.  It does not define a new family and does not
change PPO.  It answers one narrow question: after a short fixed-group PPO run,
do source-parameter cells have finite, non-degenerate sidecar deltas?

The result is a proxy for per-cell optimizability, not a replacement for the
official pack optimizability gate.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from phase2_category_span_audit import (
    PAIR_SPECS,
    REQUIRED_PAIR_SET,
    _axis_specs,
    _axis_values,
    _join_rows,
    _json_safe,
    _read_json,
    _read_jsonl,
    _safe_float,
    _stats,
)


ART = Path("/home/chen/RLPFN/artifacts")
DEFAULT_RUN_DIR = ART / "phase2_stratified_source_fixed_h_list_runner_u2_s128_0518"
DEFAULT_OUTPUT_DIR = ART / "phase2_category_span_per_cell_delta_stratified_0518"


Rows = list[dict[str, Any]]


DELTA_FIELDS = [
    "reward_sum",
    "training_reward_sum",
    "raw_reward_sum",
    "reward_env_sum",
    "training_reward_std",
    "raw_reward_std",
    "reward_std",
    "input_reward_token_std",
    "obs_per_dim_std_q90",
    "obs_per_dim_rms_mean",
    "next_state_step_l2_delta_to_rms_ratio",
    "next_state_rms",
    "done_count",
    "truncated_count",
    "action_per_dim_temporal_std_mean",
    "raw_policy_action_unit_clip_saturation_fraction",
]


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


def _sidecars(run_dir: Path) -> list[Path]:
    paths = sorted((run_dir / "semantic_sidecars").glob("*_envs.jsonl"))
    if len(paths) < 2:
        raise FileNotFoundError(f"need at least two sidecar files under {run_dir / 'semantic_sidecars'}")
    return paths


def _latest_progress(run_dir: Path) -> Path | None:
    path = run_dir / "prior_progress.jsonl"
    return path if path.exists() else None


def _progress_health(progress_path: Path | None) -> dict[str, Any]:
    if progress_path is None:
        return {"available": False, "reason": "progress jsonl not found"}
    rows = _read_jsonl(progress_path)
    out: dict[str, list[float]] = defaultdict(list)
    exceptions: list[Any] = []
    for row in rows:
        update = row.get("update", {}) or {}
        logger = update.get("logger", {}) or {}
        train = (logger.get("train", {}) or {})
        for key in ("approx_kl", "clip_fraction", "explained_variance", "value_loss", "loss"):
            value = _safe_float(train.get(key, logger.get(f"train/{key}")))
            if math.isfinite(value):
                out[key].append(value)
        pd = update.get("param_delta", {}) or {}
        for key in ("param_delta_l2", "param_delta_absmax"):
            value = _safe_float(pd.get(key))
            if math.isfinite(value):
                out[key].append(value)
        if update.get("exception"):
            exceptions.append(update.get("exception"))
    summary: dict[str, Any] = {
        "available": True,
        "progress_path": str(progress_path),
        "n_updates": len(rows),
        "exception_count": len(exceptions),
        "exceptions": exceptions,
    }
    for key, values in out.items():
        summary[key] = _stats(values)
    summary["hard_health_pass"] = bool(
        not exceptions
        and (not out.get("approx_kl") or max(out["approx_kl"]) < 0.03)
        and (not out.get("clip_fraction") or max(out["clip_fraction"]) < 0.30)
        and (not out.get("param_delta_l2") or all(v > 0 and math.isfinite(v) for v in out["param_delta_l2"]))
    )
    return summary


def _first_last_joined(run_dir: Path, h_list: Path) -> tuple[Rows, Rows, Path, Path]:
    sidecars = _sidecars(run_dir)
    first_path = sidecars[0]
    last_path = sidecars[-1]
    h_rows = _read_json(h_list)
    if not isinstance(h_rows, list):
        raise TypeError(f"h-list must be a JSON list: {h_list}")
    first = _join_rows(h_rows, _read_jsonl(first_path))
    last = _join_rows(h_rows, _read_jsonl(last_path))
    return first, last, first_path, last_path


def _delta_rows(first: Rows, last: Rows) -> Rows:
    out: Rows = []
    last_by_idx = {int(row.get("env_idx", idx)): row for idx, row in enumerate(last)}
    for idx, before in enumerate(first):
        env_idx = int(before.get("env_idx", idx))
        after = last_by_idx.get(env_idx, {})
        row: dict[str, Any] = {"env_idx": env_idx}
        for key, value in before.items():
            if key.startswith("h:"):
                row[key] = value
        for field in DELTA_FIELDS:
            a = _safe_float(after.get(field))
            b = _safe_float(before.get(field))
            row[f"{field}_first"] = b if math.isfinite(b) else None
            row[f"{field}_last"] = a if math.isfinite(a) else None
            row[f"{field}_delta"] = (a - b) if math.isfinite(a) and math.isfinite(b) else None
        reward_delta = _safe_float(row.get("reward_sum_delta"))
        row["reward_sum_improved"] = bool(math.isfinite(reward_delta) and reward_delta > 0.0)
        out.append(row)
    return out


def _summarize_cell(rows: Rows, idxs: list[int], scope: str, axis_a: str, cell_a: str, axis_b: str | None = None, cell_b: str | None = None) -> dict[str, Any]:
    selected = [rows[i] for i in idxs]
    out: dict[str, Any] = {
        "cell_scope": scope,
        "axis_a": axis_a,
        "cell_a": cell_a,
        "n": len(selected),
        "sidecar_delta_proxy": True,
        "official_pack_optimizability": False,
    }
    if axis_b is not None:
        out["axis_b"] = axis_b
        out["cell_b"] = cell_b
    if not selected:
        out["reward_sum_improved_count"] = 0
        out["reward_sum_improved_fraction"] = None
        return out
    improved = [bool(row.get("reward_sum_improved")) for row in selected]
    out["reward_sum_improved_count"] = int(sum(improved))
    out["reward_sum_improved_fraction"] = sum(improved) / len(improved)
    for field in DELTA_FIELDS:
        stat = _stats([row.get(f"{field}_delta") for row in selected])
        out[f"{field}_delta_mean"] = stat["mean"]
        out[f"{field}_delta_q25"] = stat["q25"]
        out[f"{field}_delta_q50"] = stat["q50"]
        out[f"{field}_delta_q75"] = stat["q75"]
    return out


def _cell_delta_summary(delta_rows: Rows) -> tuple[Rows, dict[str, list[str]]]:
    specs = _axis_specs()
    axis_values = _axis_values(delta_rows, specs)
    rows: Rows = []
    required_axes = [
        name
        for name, spec in specs.items()
        if spec.get("role") == "required_category_axis" and spec.get("kind") in {"structure", "mode"}
    ]
    for axis in required_axes:
        vals = axis_values[axis]
        for cell in specs[axis]["bins"]:
            idxs = [i for i, value in enumerate(vals) if value == cell]
            rows.append(_summarize_cell(delta_rows, idxs, "axis", axis, cell))
    for axis_a, axis_b in PAIR_SPECS:
        if (axis_a, axis_b) not in REQUIRED_PAIR_SET:
            continue
        vals_a = axis_values[axis_a]
        vals_b = axis_values[axis_b]
        for cell_a in specs[axis_a]["bins"]:
            for cell_b in specs[axis_b]["bins"]:
                idxs = [
                    i
                    for i, (value_a, value_b) in enumerate(zip(vals_a, vals_b))
                    if value_a == cell_a and value_b == cell_b
                ]
                rows.append(_summarize_cell(delta_rows, idxs, "pair", axis_a, cell_a, axis_b, cell_b))
    return rows, axis_values


def _read(summary_rows: Rows, progress_health: dict[str, Any]) -> dict[str, Any]:
    target_pair = [
        row
        for row in summary_rows
        if row.get("cell_scope") == "pair"
        and row.get("axis_a") == "action_dim_source"
        and row.get("axis_b") == "terminal_target_source"
    ]
    empty_target = [row for row in target_pair if int(row.get("n") or 0) == 0]
    fractions = [
        _safe_float(row.get("reward_sum_improved_fraction"))
        for row in target_pair
        if math.isfinite(_safe_float(row.get("reward_sum_improved_fraction")))
    ]
    return {
        "per_cell_delta_proxy_available": True,
        "official_pack_optimizability_available": False,
        "action_terminal_empty_cell_count": len(empty_target),
        "action_terminal_min_reward_sum_improved_fraction": min(fractions) if fractions else None,
        "action_terminal_mean_reward_sum_improved_fraction": sum(fractions) / len(fractions) if fractions else None,
        "aggregate_ppo_health_pass": bool(progress_health.get("hard_health_pass")),
        "interpretation": (
            "This is a sidecar first/last delta proxy grouped by source cells. "
            "Use it to detect cell-level degeneracy before running official pack optimizability."
        ),
    }


def _markdown(report: dict[str, Any]) -> str:
    read = report["read"]
    lines = [
        "# Category Span Per-Cell Delta Audit",
        "",
        "Read-only sidecar delta proxy. It groups first/last true-runner sidecar changes by category cells.",
        "",
        "## Inputs",
        "",
        f"- first sidecar: `{report['inputs']['first_sidecar']}`",
        f"- last sidecar: `{report['inputs']['last_sidecar']}`",
        f"- h-list: `{report['inputs']['h_list']}`",
        "",
        "## Read",
        "",
        f"- action×terminal empty cells: `{read['action_terminal_empty_cell_count']}`",
        f"- action×terminal mean reward_sum improved fraction: `{read['action_terminal_mean_reward_sum_improved_fraction']}`",
        f"- action×terminal min reward_sum improved fraction: `{read['action_terminal_min_reward_sum_improved_fraction']}`",
        f"- aggregate PPO health pass: `{read['aggregate_ppo_health_pass']}`",
        "",
        "## Note",
        "",
        "This is not the official pack optimizability gate. It is a per-env sidecar delta proxy used to catch obvious cell-level degeneration.",
        "",
        "## Outputs",
        "",
    ]
    for key, value in report["outputs"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = Path(args.run_dir).expanduser().resolve()
    h_list = (
        Path(args.h_list).expanduser().resolve()
        if args.h_list
        else run_dir / "fixed_env_group" / "prior_fixed_h_list.json"
    )
    first, last, first_sidecar, last_sidecar = _first_last_joined(run_dir, h_list)
    deltas = _delta_rows(first, last)
    cell_summary, _ = _cell_delta_summary(deltas)
    progress = _progress_health(_latest_progress(run_dir))
    read = _read(cell_summary, progress)

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": out_dir / "category_span_per_cell_delta_audit.json",
        "markdown": out_dir / "category_span_per_cell_delta_audit.md",
        "env_delta_csv": out_dir / "category_span_env_delta.csv",
        "cell_delta_csv": out_dir / "category_span_cell_delta.csv",
    }
    report: dict[str, Any] = {
        "analysis_entry": "phase2_category_span_per_cell_delta_audit",
        "schema": "phase2_category_span_per_cell_delta_audit.v1",
        "contract": {
            "read_only": True,
            "does_not_sample_environments": True,
            "does_not_modify_generator": True,
            "does_not_modify_ppo": True,
            "sidecar_delta_proxy_not_official_pack_optimizability": True,
        },
        "inputs": {
            "run_dir": str(run_dir),
            "h_list": str(h_list),
            "first_sidecar": str(first_sidecar),
            "last_sidecar": str(last_sidecar),
            "n_envs": len(deltas),
        },
        "progress_health": progress,
        "cell_delta_summary": cell_summary,
        "read": read,
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    paths["json"].write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths["markdown"].write_text(_markdown(report), encoding="utf-8")
    _write_csv(paths["env_delta_csv"], deltas)
    _write_csv(paths["cell_delta_csv"], cell_summary)
    print(json.dumps({key: str(value) for key, value in paths.items()}, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--h-list", default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
