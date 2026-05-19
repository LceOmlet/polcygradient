#!/usr/bin/env python
"""Audit source-category axes against true-runner realized structural image.

This is a read-only reduction audit.  It deliberately separates:

1. source contract axes stored in the fixed h-list,
2. realized axes logged by the true runner sidecar,
3. formula evidence axes that are not yet logged.

The goal is to prevent source-span coverage from being mistaken for realized
environment coverage, and to make the minimal next repair obvious before any
generator implementation is attempted.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


ART = Path("/home/chen/RLPFN/artifacts")
DEFAULT_RUN_DIR = ART / "phase2_stratified_source_fixed_h_list_runner_u2_s128_0518"
DEFAULT_OUTPUT_DIR = ART / "phase2_category_structural_image_audit_stratified_0518"


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


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _entropy(counts: Counter[str]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    ent = 0.0
    for count in counts.values():
        if count <= 0:
            continue
        p = count / total
        ent -= p * math.log(p)
    return ent


def _normalized_entropy(counts: Counter[str], expected: int) -> float:
    if expected <= 1:
        return 1.0
    return _entropy(counts) / math.log(expected)


def _bin3(value: Any, lo_hi: tuple[float, float]) -> str:
    x = _safe_float(value)
    if not math.isfinite(x):
        return "missing"
    if x <= lo_hi[0]:
        return "low"
    if x <= lo_hi[1]:
        return "mid"
    return "high"


def _status(counts: Counter[str], bins: list[str]) -> str:
    expected = len(bins)
    occupied = sum(1 for b in bins if counts.get(b, 0) > 0)
    ent = _normalized_entropy(Counter({b: counts.get(b, 0) for b in bins}), expected)
    if occupied == expected and ent >= 0.90:
        return "pass_uniform"
    if occupied == expected:
        return "covered_skewed"
    if occupied > 1:
        return "partial"
    return "collapsed"


def _latest_sidecar(run_dir: Path) -> Path:
    candidates = sorted((run_dir / "semantic_sidecars").glob("*_envs.jsonl"))
    if not candidates:
        raise FileNotFoundError(f"no sidecar found under {run_dir / 'semantic_sidecars'}")
    return candidates[-1]


def _join(h_rows: Rows, side_rows: Rows) -> Rows:
    out: Rows = []
    side_by_idx = {int(row.get("env_idx", idx)): row for idx, row in enumerate(side_rows)}
    for idx, h in enumerate(h_rows):
        side = side_by_idx.get(idx, {})
        row: dict[str, Any] = {"env_idx": idx}
        for key, value in h.items():
            row[f"h:{key}"] = value
        for key, value in side.items():
            row[f"side:{key}"] = value
        out.append(row)
    return out


AXES = {
    "state_dim": {
        "bins": ["low", "mid", "high"],
        "source_key": "h:state_dim",
        "realized_key": "side:state_dim",
        "source_edges": (133.0, 266.0),
        "realized_edges": (133.0, 266.0),
        "role": "category_structure",
    },
    "obs_dim": {
        "bins": ["low", "mid", "high"],
        "source_key": "h:obs_dim",
        "realized_key": "side:obs_dim",
        "source_edges": (133.0, 266.0),
        "realized_edges": (133.0, 266.0),
        "role": "category_structure",
    },
    "action_dim": {
        "bins": ["low", "mid", "high"],
        "source_key": "h:action_dim",
        "realized_key": "side:action_dim",
        "source_edges": (10.0, 20.0),
        "realized_edges": (10.0, 20.0),
        "role": "category_structure",
    },
    "noise_dim": {
        "bins": ["low", "mid", "high"],
        "source_key": "h:noise_dim",
        "realized_key": "side:noise_dim",
        "source_edges": (20.0, 40.0),
        "realized_edges": (133.0, 266.0),
        "role": "category_structure",
        "note": "realized noise_dim uses the maintained constrained-dim image, not the original 1..64 source range",
    },
    "zero_pad_dim": {
        "bins": ["low", "mid", "high"],
        "source_key": "h:zero_pad_dim",
        "realized_key": "side:zero_pad_dim",
        "source_edges": (133.0, 266.0),
        "realized_edges": (133.0, 266.0),
        "role": "category_structure",
    },
    "terminal_target": {
        "bins": ["low", "mid", "high"],
        "source_key": "h:terminal_reset_count_target",
        "realized_key": "side:terminal_reset_count_target",
        "source_edges": (26.6667, 53.3333),
        "realized_edges": (26.6667, 53.3333),
        "role": "category_structure",
    },
}


REALIZED_PAIR_SPECS = [
    ("state_dim", "action_dim"),
    ("obs_dim", "action_dim"),
    ("action_dim", "terminal_target"),
    ("state_dim", "terminal_target"),
    ("obs_dim", "terminal_target"),
    ("noise_dim", "zero_pad_dim"),
]


FORMULA_EVIDENCE_AXES = [
    {
        "axis": "controllability_rank",
        "status": "not_logged",
        "minimal_test_form": "finite-diff or autograd rank of action -> next_state/obs/reward at fixed state/noise",
        "minimal_repair_if_failed": "only then repair transition action-coupling rank source; do not alter reward/terminal",
    },
    {
        "axis": "controllability_delay",
        "status": "not_logged",
        "minimal_test_form": "same fixed action impulse over horizon; classify immediate/one-step/multi-step response",
        "minimal_repair_if_failed": "if Exact SCM has no delay family and delay is required, add one explicit natural delay mechanism",
    },
    {
        "axis": "obs_projection_rank",
        "status": "partially_inferable",
        "minimal_test_form": "realized obs_dim/state_dim imply identity-subset rank min(obs_dim,state_dim); covariance rank still needs trajectory evidence",
        "minimal_repair_if_failed": "only repair observation projection source if realized covariance/projection rank collapses",
    },
    {
        "axis": "done_topology_type",
        "status": "not_logged",
        "minimal_test_form": "label terminal mechanism formula: hazard/probabilistic reset vs state-boundary vs time-limit",
        "minimal_repair_if_failed": "define terminal topology source; do not tune done_count as a proxy",
    },
    {
        "axis": "reward_topology_type",
        "status": "partially_logged",
        "minimal_test_form": "classify reward formula path from sidecar components and gain densities: state/action/noise/control/survival/potential/sparse",
        "minimal_repair_if_failed": "repair reward topology source only after formula labels show a missing topology",
    },
    {
        "axis": "per_cell_official_optimizability",
        "status": "not_logged",
        "minimal_test_form": "fixed per-cell pre/post eval using the official pack optimizability path",
        "minimal_repair_if_failed": "repair only the failing structural cell, not the whole generator",
    },
]


def _axis_rows(rows: Rows) -> Rows:
    out: Rows = []
    for axis, spec in AXES.items():
        bins = list(spec["bins"])
        for source in ("source", "realized"):
            key = spec["source_key"] if source == "source" else spec["realized_key"]
            edges = spec["source_edges"] if source == "source" else spec["realized_edges"]
            vals = [_bin3(row.get(key), edges) for row in rows]
            counts = Counter(vals)
            total = len(vals)
            ent = _normalized_entropy(Counter({b: counts.get(b, 0) for b in bins}), len(bins))
            item: dict[str, Any] = {
                "axis": axis,
                "image": source,
                "role": spec["role"],
                "status": _status(counts, bins),
                "n": total,
                "occupied_expected_bins": sum(1 for b in bins if counts.get(b, 0) > 0),
                "expected_bins": len(bins),
                "normalized_entropy": ent,
                "max_bin_fraction": max([counts.get(b, 0) / max(1, total) for b in bins] or [0.0]),
                "note": spec.get("note", ""),
            }
            for b in bins:
                item[f"count:{b}"] = counts.get(b, 0)
            out.append(item)
    return out


def _alignment_rows(rows: Rows) -> Rows:
    out: Rows = []
    for axis, spec in AXES.items():
        mismatches = []
        source_vals = []
        realized_vals = []
        for row in rows:
            s = row.get(spec["source_key"])
            r = row.get(spec["realized_key"])
            source_vals.append(_safe_float(s))
            realized_vals.append(_safe_float(r))
            sv = _safe_float(s)
            rv = _safe_float(r)
            if not (math.isfinite(sv) and math.isfinite(rv) and abs(sv - rv) <= 1e-6):
                mismatches.append((int(row["env_idx"]), s, r))
        out.append(
            {
                "axis": axis,
                "n": len(rows),
                "mismatch_count": len(mismatches),
                "aligned": len(mismatches) == 0,
                "source_min": min([v for v in source_vals if math.isfinite(v)] or [float("nan")]),
                "source_max": max([v for v in source_vals if math.isfinite(v)] or [float("nan")]),
                "realized_min": min([v for v in realized_vals if math.isfinite(v)] or [float("nan")]),
                "realized_max": max([v for v in realized_vals if math.isfinite(v)] or [float("nan")]),
                "sample_mismatches": mismatches[:5],
            }
        )
    return out


def _realized_pair_rows(rows: Rows) -> Rows:
    out: Rows = []
    axis_bins: dict[str, list[str]] = {}
    for axis, spec in AXES.items():
        axis_bins[axis] = [_bin3(row.get(spec["realized_key"]), spec["realized_edges"]) for row in rows]
    for a, b in REALIZED_PAIR_SPECS:
        bins_a = list(AXES[a]["bins"])
        bins_b = list(AXES[b]["bins"])
        pairs = list(zip(axis_bins[a], axis_bins[b]))
        counts = Counter(pairs)
        expected = len(bins_a) * len(bins_b)
        pair_counts = Counter({f"{x}|{y}": counts.get((x, y), 0) for x in bins_a for y in bins_b})
        total = len(rows)
        item: dict[str, Any] = {
            "axis_a": a,
            "axis_b": b,
            "image": "realized",
            "status": _status(pair_counts, list(pair_counts.keys())),
            "occupied_expected_cells": sum(1 for x in bins_a for y in bins_b if counts.get((x, y), 0) > 0),
            "expected_cells": expected,
            "normalized_entropy": _normalized_entropy(pair_counts, expected),
            "max_cell_fraction": max([counts.get((x, y), 0) / max(1, total) for x in bins_a for y in bins_b] or [0.0]),
        }
        for x in bins_a:
            for y in bins_b:
                item[f"count:{x}|{y}"] = counts.get((x, y), 0)
        out.append(item)
    return out


def _obs_projection_rows(rows: Rows) -> Rows:
    out: Rows = []
    for row in rows:
        env_idx = int(row["env_idx"])
        state_dim = int(_safe_float(row.get("side:state_dim")))
        obs_dim = int(_safe_float(row.get("side:obs_dim")))
        rank = max(0, min(state_dim, obs_dim))
        out.append(
            {
                "env_idx": env_idx,
                "realized_state_dim": state_dim,
                "realized_obs_dim": obs_dim,
                "identity_subset_projection_rank": rank,
                "projection_rank_to_obs_dim_ratio": rank / max(1, obs_dim),
                "projection_rank_to_state_dim_ratio": rank / max(1, state_dim),
                "status": "full_obs_rank" if rank >= obs_dim else "partial_obs_rank",
            }
        )
    return out


def _read(axis_rows: Rows, align_rows: Rows, pair_rows: Rows, obs_proj_rows: Rows) -> dict[str, Any]:
    hard_realized_fail = [
        row
        for row in axis_rows
        if row["image"] == "realized"
        and row["axis"] in {"state_dim", "obs_dim", "action_dim", "terminal_target"}
        and row["status"] not in {"pass_uniform", "covered_skewed"}
    ]
    realized_noise_zero_fail = [
        row
        for row in axis_rows
        if row["image"] == "realized"
        and row["axis"] in {"noise_dim", "zero_pad_dim"}
        and row["status"] not in {"pass_uniform", "covered_skewed"}
    ]
    pair_fail = [
        row
        for row in pair_rows
        if row["axis_a"] in {"state_dim", "obs_dim", "action_dim"}
        and row["axis_b"] in {"action_dim", "terminal_target"}
        and row["status"] not in {"pass_uniform", "covered_skewed"}
    ]
    noise_zero_pair = [
        row
        for row in pair_rows
        if row["axis_a"] == "noise_dim" and row["axis_b"] == "zero_pad_dim"
    ]
    obs_partial = [row for row in obs_proj_rows if row["status"] != "full_obs_rank"]
    source_realized_mismatch = [row for row in align_rows if not row["aligned"]]
    return {
        "realized_core_span_pass": bool(not hard_realized_fail and not pair_fail),
        "realized_noise_zero_span_pass": bool(not realized_noise_zero_fail and noise_zero_pair and noise_zero_pair[0]["status"] in {"pass_uniform", "covered_skewed"}),
        "source_realized_all_aligned": bool(not source_realized_mismatch),
        "source_realized_mismatch_axes": [row["axis"] for row in source_realized_mismatch],
        "obs_identity_projection_full_obs_rank_fraction": (
            1.0 - len(obs_partial) / max(1, len(obs_proj_rows))
        ),
        "formula_evidence_closed": False,
        "formula_evidence_closed_reason": "controllability rank/delay, terminal topology, reward topology, and official per-cell optimizability still need dedicated tests",
        "next_minimal_tests": [
            "finite-diff controllability rank/delay audit",
            "reward/done topology formula-label audit",
            "official per-cell optimizability audit",
        ],
    }


def _markdown(report: dict[str, Any]) -> str:
    read = report["read"]
    lines = [
        "# Category Structural Image Audit",
        "",
        "Read-only audit separating fixed h-list source axes from true-runner realized axes.",
        "",
        "## Read",
        "",
        f"- realized core span pass: `{read['realized_core_span_pass']}`",
        f"- realized noise/zero span pass: `{read['realized_noise_zero_span_pass']}`",
        f"- source-realized all aligned: `{read['source_realized_all_aligned']}`",
        f"- source-realized mismatch axes: `{read['source_realized_mismatch_axes']}`",
        f"- obs identity projection full-rank fraction: `{read['obs_identity_projection_full_obs_rank_fraction']}`",
        f"- formula evidence closed: `{read['formula_evidence_closed']}`",
        "",
        "## Axis Coverage",
        "",
        "| axis | image | status | occupied/expected | entropy | max bin |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in report["axis_coverage"]:
        lines.append(
            f"| `{row['axis']}` | `{row['image']}` | `{row['status']}` | "
            f"{row['occupied_expected_bins']}/{row['expected_bins']} | "
            f"{row['normalized_entropy']:.3f} | {row['max_bin_fraction']:.3f} |"
        )
    lines.extend(["", "## Source vs Realized Alignment", ""])
    lines.extend(["| axis | aligned | mismatch | source range | realized range |", "|---|---:|---:|---:|---:|"])
    for row in report["alignment"]:
        lines.append(
            f"| `{row['axis']}` | `{row['aligned']}` | {row['mismatch_count']} | "
            f"{row['source_min']:.3g}..{row['source_max']:.3g} | "
            f"{row['realized_min']:.3g}..{row['realized_max']:.3g} |"
        )
    lines.extend(["", "## Realized Pair Coverage", ""])
    lines.extend(["| axis A | axis B | status | occupied/expected | entropy | max cell |", "|---|---|---|---:|---:|---:|"])
    for row in report["realized_pair_coverage"]:
        lines.append(
            f"| `{row['axis_a']}` | `{row['axis_b']}` | `{row['status']}` | "
            f"{row['occupied_expected_cells']}/{row['expected_cells']} | "
            f"{row['normalized_entropy']:.3f} | {row['max_cell_fraction']:.3f} |"
        )
    lines.extend(["", "## Formula Evidence Axes", ""])
    for row in report["formula_evidence_axes"]:
        lines.append(f"- `{row['axis']}`: `{row['status']}`; test={row['minimal_test_form']}")
    lines.extend(["", "## Outputs", ""])
    for key, value in report["outputs"].items():
        lines.append(f"- `{key}`: `{value}`")
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = Path(args.run_dir).expanduser().resolve()
    h_list_path = Path(args.h_list).expanduser().resolve() if args.h_list else run_dir / "fixed_env_group/prior_fixed_h_list.json"
    sidecar_path = Path(args.sidecar).expanduser().resolve() if args.sidecar else _latest_sidecar(run_dir)
    h_rows = _read_json(h_list_path)
    side_rows = _read_jsonl(sidecar_path)
    if not isinstance(h_rows, list):
        raise TypeError(f"h-list must be a list: {h_list_path}")
    rows = _join(h_rows, side_rows)
    axis_rows = _axis_rows(rows)
    align_rows = _alignment_rows(rows)
    pair_rows = _realized_pair_rows(rows)
    obs_projection = _obs_projection_rows(rows)
    read = _read(axis_rows, align_rows, pair_rows, obs_projection)

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": out_dir / "category_structural_image_audit.json",
        "markdown": out_dir / "category_structural_image_audit.md",
        "axis_csv": out_dir / "category_structural_image_axis_coverage.csv",
        "alignment_csv": out_dir / "category_structural_image_source_realized_alignment.csv",
        "pair_csv": out_dir / "category_structural_image_realized_pair_coverage.csv",
        "obs_projection_csv": out_dir / "category_structural_image_obs_projection.csv",
    }
    report = {
        "analysis_entry": "phase2_category_structural_image_audit",
        "schema": "phase2_category_structural_image_audit.v1",
        "contract": {
            "read_only": True,
            "does_not_modify_generator": True,
            "does_not_modify_ppo": True,
            "separates_testing_from_development": True,
            "separates_source_axes_from_realized_axes": True,
        },
        "inputs": {
            "run_dir": str(run_dir),
            "h_list": str(h_list_path),
            "sidecar": str(sidecar_path),
            "h_count": len(h_rows),
            "sidecar_count": len(side_rows),
        },
        "axis_coverage": axis_rows,
        "alignment": align_rows,
        "realized_pair_coverage": pair_rows,
        "obs_projection": obs_projection,
        "formula_evidence_axes": FORMULA_EVIDENCE_AXES,
        "read": read,
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    paths["json"].write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths["markdown"].write_text(_markdown(report), encoding="utf-8")
    _write_csv(paths["axis_csv"], axis_rows)
    _write_csv(paths["alignment_csv"], align_rows)
    _write_csv(paths["pair_csv"], pair_rows)
    _write_csv(paths["obs_projection_csv"], obs_projection)
    print(json.dumps({key: str(value) for key, value in paths.items()}, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--h-list", default=None)
    parser.add_argument("--sidecar", default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
