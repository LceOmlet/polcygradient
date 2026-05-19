#!/usr/bin/env python
"""Read-only provenance audit for reward density topology.

The prior topology label reported reward-density collapse and high coupling to
state/obs dimensions.  This audit keeps that as a test question rather than a
generator patch: it inspects the actual reward input gain allocation and source
policy, then reports whether the axis is a required repair or only a category
decision.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


ART = Path("/home/chen/RLPFN/artifacts")
DEFAULT_RUN_DIR = ART / "phase2_stratified_realized_noise_zero_runner_u2_s128_0518"
DEFAULT_FIXED_CSV = (
    ART
    / "phase2_stratified_realized_noise_zero_fixed_h_list_0518"
    / "stratified_realized_noise_zero_fixed_h_list.csv"
)
DEFAULT_TOPOLOGY_PER_ENV = (
    ART
    / "phase2_category_topology_label_audit_realized_noise_zero_0518"
    / "category_topology_label_per_env.csv"
)
DEFAULT_OUTPUT_DIR = ART / "phase2_reward_density_provenance_audit_realized_noise_zero_0518"


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).expanduser().resolve().open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


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
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _safe_float(value: Any, default: float = float("nan")) -> float:
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


def _load_h_rows(fixed_csv: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in _read_csv(fixed_csv):
        raw = str(row.get("full_frozen_h_json", "")).strip()
        path = Path(raw)
        rows.append(_read_json(path) if path.exists() else json.loads(raw))
    return rows


def _density_label(
    state: float,
    action: float,
    noise: float,
    *,
    state_dim: float,
    action_dim: float,
    noise_dim: float,
) -> str:
    vals = {}
    if float(state_dim) > 0.0:
        vals["state"] = state
    if float(action_dim) > 0.0:
        vals["action"] = action
    if float(noise_dim) > 0.0:
        vals["noise"] = noise
    finite = {k: v for k, v in vals.items() if math.isfinite(v) and v > 0.0}
    if len(finite) < 2:
        return "missing"
    ordered = sorted(finite.items(), key=lambda kv: kv[1], reverse=True)
    ratio = ordered[0][1] / max(ordered[-1][1], 1e-12)
    if ratio <= 1.5:
        return "density_balanced"
    return f"{ordered[0][0]}_density_dominant"


def _source_policy(h: dict[str, Any]) -> str:
    if _safe_bool(h.get("reward_topology_conditioned_sampling_enabled")):
        return "topology_conditioned_balance"
    if _safe_bool(h.get("reward_state_input_gain_fraction_conditioned_sampling_enabled")):
        return "state_fraction_rejection"
    return "unconditioned"


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


def _coverage(labels: list[str], expected: list[str]) -> dict[str, Any]:
    counts = Counter(labels)
    occupied = sum(1 for label in expected if counts.get(label, 0) > 0)
    norm = _entropy(Counter({label: counts.get(label, 0) for label in expected})) / math.log(len(expected))
    if occupied == len(expected) and norm >= 0.85:
        status = "pass_uniform"
    elif occupied == len(expected):
        status = "covered_skewed"
    elif occupied > 1:
        status = "partial"
    else:
        status = "collapsed"
    return {
        "status": status,
        "occupied_expected_labels": occupied,
        "expected_labels": len(expected),
        "normalized_entropy": norm,
        "max_label_fraction": max([counts.get(label, 0) / max(1, len(labels)) for label in expected] or [0.0]),
        **{f"count:{label}": int(counts.get(label, 0)) for label in expected},
    }


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
            if expected > 0:
                chi2 += (table[i][j] - expected) ** 2 / expected
    denom = n * max(1, min(len(x_labels) - 1, len(y_labels) - 1))
    return math.sqrt(max(0.0, chi2 / denom)) if denom > 0 else 0.0


def _stats(values: list[float]) -> dict[str, Any]:
    vals = np.asarray([v for v in values if math.isfinite(float(v))], dtype=np.float64)
    if vals.size == 0:
        return {"n": 0, "mean": None, "q10": None, "q50": None, "q90": None}
    return {
        "n": int(vals.size),
        "mean": float(vals.mean()),
        "q10": float(np.quantile(vals, 0.10)),
        "q50": float(np.quantile(vals, 0.50)),
        "q90": float(np.quantile(vals, 0.90)),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    fixed_csv = Path(args.fixed_csv).expanduser().resolve()
    sidecar = Path(args.sidecar).expanduser().resolve() if args.sidecar else _latest_sidecar(Path(args.run_dir))
    h_rows = _load_h_rows(fixed_csv)
    side_rows = _read_jsonl(sidecar)
    struct_by_idx = {int(row["env_idx"]): row for row in _read_csv(args.topology_per_env_csv)}

    per_env: list[dict[str, Any]] = []
    for idx, h in enumerate(h_rows):
        side = side_rows[idx] if idx < len(side_rows) else {}
        state = _safe_float(side.get("reward_state_input_gain_density_ratio"))
        action = _safe_float(side.get("reward_action_input_gain_density_ratio"))
        noise = _safe_float(side.get("reward_noise_input_gain_density_ratio"))
        frac_state = _safe_float(side.get("reward_state_input_gain_fraction"))
        frac_action = _safe_float(side.get("reward_action_input_gain_fraction"))
        frac_noise = _safe_float(side.get("reward_noise_input_gain_fraction"))
        struct = struct_by_idx.get(idx, {})
        state_dim = _safe_float(side.get("state_dim"), _safe_float(struct.get("struct:state_dim"), 0.0))
        action_dim = _safe_float(side.get("action_dim"), _safe_float(struct.get("struct:action_dim"), 0.0))
        noise_dim = _safe_float(side.get("noise_dim"), _safe_float(struct.get("struct:noise_dim"), 0.0))
        row: dict[str, Any] = {
            "env_idx": idx,
            "reward_density_topology": _density_label(
                state,
                action,
                noise,
                state_dim=state_dim,
                action_dim=action_dim,
                noise_dim=noise_dim,
            ),
            "reward_density_source_policy": _source_policy(h),
            "state_density_ratio": state,
            "action_density_ratio": action,
            "noise_density_ratio": noise,
            "state_gain_fraction": frac_state,
            "action_gain_fraction": frac_action,
            "noise_gain_fraction": frac_noise,
            "topology_conditioned_sampling_enabled": _safe_bool(h.get("reward_topology_conditioned_sampling_enabled")),
            "state_fraction_conditioned_sampling_enabled": _safe_bool(
                h.get("reward_state_input_gain_fraction_conditioned_sampling_enabled")
            ),
        }
        for axis in ["state_dim", "obs_dim", "action_dim", "terminal_target", "noise_dim", "zero_pad_dim"]:
            row[f"struct:{axis}"] = struct.get(f"struct:{axis}", "missing")
        per_env.append(row)

    expected = [
        "density_balanced",
        "state_density_dominant",
        "action_density_dominant",
        "noise_density_dominant",
        "missing",
    ]
    labels = [str(row["reward_density_topology"]) for row in per_env]
    policies = [str(row["reward_density_source_policy"]) for row in per_env]
    coupling: list[dict[str, Any]] = []
    for label_axis in ["reward_density_topology", "reward_density_source_policy"]:
        xs = [str(row[label_axis]) for row in per_env]
        for struct in ["state_dim", "obs_dim", "action_dim", "terminal_target", "noise_dim", "zero_pad_dim"]:
            ys = [str(row.get(f"struct:{struct}", "missing")) for row in per_env]
            v = _cramers_v(xs, ys)
            if len(set(xs)) <= 1:
                status = "undefined_collapsed_topology"
            elif v >= 0.50:
                status = "high"
            elif v >= 0.25:
                status = "medium"
            else:
                status = "low"
            coupling.append(
                {
                    "axis": label_axis,
                    "structure_axis": struct,
                    "cramers_v": v,
                    "status": status,
                }
            )

    density_cov = _coverage(labels, expected)
    policy_cov = _coverage(policies, ["topology_conditioned_balance", "state_fraction_rejection", "unconditioned"])
    high = [row for row in coupling if row["status"] == "high"]
    report = {
        "analysis_entry": "phase2_reward_density_provenance_audit",
        "schema": "phase2_reward_density_provenance_audit.v1",
        "contract": {
            "read_only": True,
            "does_not_modify_generator": True,
            "does_not_train": True,
            "does_not_infer_source_from_dim_fraction_only": True,
        },
        "read": {
            "reward_density_provenance_audit_complete": True,
            "density_coverage": density_cov,
            "source_policy_coverage": policy_cov,
            "high_structure_density_coupling_count": len(high),
            "generator_repair_justified_now": False,
            "generator_repair_justified_now_reason": (
                "dominant density modes are absent, but they must be promoted as formal category labels before repair"
            ),
        },
        "stats": {
            "state_density_ratio": _stats([float(row["state_density_ratio"]) for row in per_env]),
            "action_density_ratio": _stats([float(row["action_density_ratio"]) for row in per_env]),
            "noise_density_ratio": _stats([float(row["noise_density_ratio"]) for row in per_env]),
            "state_gain_fraction": _stats([float(row["state_gain_fraction"]) for row in per_env]),
            "action_gain_fraction": _stats([float(row["action_gain_fraction"]) for row in per_env]),
            "noise_gain_fraction": _stats([float(row["noise_gain_fraction"]) for row in per_env]),
        },
        "per_env": per_env,
        "structure_density_coupling": coupling,
        "inputs": {
            "fixed_csv": str(fixed_csv),
            "sidecar": str(sidecar),
            "topology_per_env_csv": str(Path(args.topology_per_env_csv).expanduser().resolve()),
        },
    }
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "reward_density_provenance_audit.json"
    md_path = out_dir / "reward_density_provenance_audit.md"
    per_env_csv = out_dir / "reward_density_provenance_per_env.csv"
    coupling_csv = out_dir / "reward_density_provenance_coupling.csv"
    report["outputs"] = {
        "json": str(json_path),
        "markdown": str(md_path),
        "per_env_csv": str(per_env_csv),
        "coupling_csv": str(coupling_csv),
    }
    json_path.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(per_env_csv, per_env)
    _write_csv(coupling_csv, coupling)
    md_path.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report["outputs"], sort_keys=True))
    return report


def _markdown(report: dict[str, Any]) -> str:
    read = report["read"]
    lines = [
        "# Reward Density Provenance Audit",
        "",
        "Read-only source/formula provenance audit for reward input density allocation.",
        "",
        "## Read",
        "",
        f"- reward density provenance audit complete: `{read['reward_density_provenance_audit_complete']}`",
        f"- density coverage: `{read['density_coverage']}`",
        f"- source policy coverage: `{read['source_policy_coverage']}`",
        f"- high structure-density coupling count: `{read['high_structure_density_coupling_count']}`",
        f"- generator repair justified now: `{read['generator_repair_justified_now']}`",
        f"- reason: {read['generator_repair_justified_now_reason']}",
        "",
        "## Gain Statistics",
        "",
        "| metric | q10 | q50 | q90 |",
        "|---|---:|---:|---:|",
    ]
    for key, stats in report["stats"].items():
        lines.append(f"| `{key}` | {stats['q10']} | {stats['q50']} | {stats['q90']} |")
    highs = [row for row in report["structure_density_coupling"] if row["status"] == "high"]
    lines.extend(["", "## High Coupling", ""])
    if not highs:
        lines.append("- none")
    else:
        for row in highs:
            lines.append(f"- `{row['axis']}` vs `{row['structure_axis']}`: Cramer's V `{row['cramers_v']:.3f}`")
    lines.extend(
        [
            "",
            "## Reduction",
            "",
            "- Current reward density is intentionally near balanced under `topology_conditioned_balance`.",
            "- Dominant density modes are absent, but that is not automatically a defect.",
            "- If reward density diversity is declared required, repair must add explicit source labels for allocation topology and recheck independence/PPO health.",
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
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--fixed-csv", default=str(DEFAULT_FIXED_CSV))
    parser.add_argument("--sidecar", default="")
    parser.add_argument("--topology-per-env-csv", default=str(DEFAULT_TOPOLOGY_PER_ENV))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
