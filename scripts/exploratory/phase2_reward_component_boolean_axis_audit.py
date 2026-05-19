#!/usr/bin/env python
"""Read-only reward component axis audit.

The older topology label used a single component string such as
``env+ctrl+aux``.  That is not a sufficient category definition, because Gym-like
reward families vary by independent component mechanisms.  This audit splits
the label into boolean formula/component axes and checks coverage plus hidden
coupling to structural bins.

It intentionally does not edit the generator.  Missing components are reported
as category decisions first; a generator repair is justified only after the
declared component set is accepted.
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
DEFAULT_OUTPUT_DIR = ART / "phase2_reward_component_boolean_axis_audit_realized_noise_zero_0518"


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).expanduser().resolve().open("r", encoding="utf-8", newline="") as handle:
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


def _latest_sidecar(run_dir: Path) -> Path:
    candidates = sorted((run_dir / "semantic_sidecars").glob("*_envs.jsonl"))
    if not candidates:
        raise FileNotFoundError(f"no semantic sidecar under {run_dir / 'semantic_sidecars'}")
    return candidates[-1]


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
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "on"}


def _nonzero_stats(row: dict[str, Any], prefix: str, *, eps: float = 1e-8) -> bool:
    for suffix in ("std", "sum", "mean", "q90", "q50", "min", "max"):
        if abs(_safe_float(row.get(f"{prefix}_{suffix}"), 0.0)) > eps:
            return True
    return _safe_bool(row.get(f"{prefix}_q90_gt0"))


def _load_h_rows(fixed_csv: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in _read_csv(fixed_csv):
        raw = str(row.get("full_frozen_h_json", "")).strip()
        path = Path(raw)
        if path.exists():
            payload = _read_json(path)
        else:
            payload = json.loads(raw)
        rows.append(payload)
    return rows


def _component_labels(h: dict[str, Any], side: dict[str, Any]) -> dict[str, bool]:
    logging = side.get("reward_component_logging")
    if not isinstance(logging, dict):
        logging = {}
    lowtail_active = _safe_bool(side.get("exact_scm_gym_lowtail_family_active")) or _safe_bool(
        h.get("exact_scm_gym_lowtail_family_selected")
    )
    ctrl_enabled = _safe_bool(logging.get("ctrl_enabled")) or _safe_bool(h.get("ctrl_reward_enabled"))
    survival_enabled = _safe_bool(logging.get("survival_enabled")) or _safe_bool(h.get("survival_reward_enabled"))
    ctrl_weight = abs(_safe_float(logging.get("ctrl_weight"), _safe_float(h.get("ctrl_reward_weight"), 0.0)))
    survival_weight = abs(
        _safe_float(logging.get("survival_weight"), _safe_float(h.get("survival_reward_weight"), 0.0))
    )
    return {
        "base_env_reward": _nonzero_stats(side, "reward_env") or _nonzero_stats(side, "raw_reward"),
        "control_cost": (ctrl_enabled and ctrl_weight > 1e-10) or _nonzero_stats(side, "reward_ctrl"),
        "survival_bonus": (survival_enabled and survival_weight > 1e-10)
        or _nonzero_stats(side, "reward_survival"),
        "potential_delta": bool(lowtail_active)
        and abs(_safe_float(h.get("exact_scm_gym_lowtail_reward_potential_delta_weight"), 0.0)) > 1e-10,
        "distribution_head": bool(lowtail_active)
        and (
            _safe_bool(side.get("exact_scm_gym_lowtail_reward_distribution_head_enabled"))
            or _safe_bool(h.get("exact_scm_gym_lowtail_reward_distribution_head_enabled"))
        ),
        "terminal_bonus": _nonzero_stats(side, "reward_terminal_bonus"),
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


def _coverage_row(axis: str, values: list[bool]) -> dict[str, Any]:
    count_true = sum(1 for value in values if value)
    count_false = len(values) - count_true
    if count_true > 0 and count_false > 0:
        status = "covered_boolean"
    elif count_true == len(values):
        status = "always_present"
    else:
        status = "missing"
    return {
        "axis": axis,
        "status": status,
        "n": len(values),
        "count:false": count_false,
        "count:true": count_true,
        "true_fraction": count_true / max(1, len(values)),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    fixed_csv = Path(args.fixed_csv).expanduser().resolve()
    sidecar = Path(args.sidecar).expanduser().resolve() if args.sidecar else _latest_sidecar(Path(args.run_dir))
    h_rows = _load_h_rows(fixed_csv)
    side_rows = _read_jsonl(sidecar)
    topology_rows = _read_csv(args.topology_per_env_csv)
    struct_by_idx = {int(row["env_idx"]): row for row in topology_rows}

    per_env: list[dict[str, Any]] = []
    for idx, h in enumerate(h_rows):
        side = side_rows[idx] if idx < len(side_rows) else {}
        labels = _component_labels(h, side)
        row: dict[str, Any] = {"env_idx": idx}
        row.update(labels)
        struct = struct_by_idx.get(idx, {})
        for axis in ["state_dim", "obs_dim", "action_dim", "terminal_target", "noise_dim", "zero_pad_dim"]:
            row[f"struct:{axis}"] = struct.get(f"struct:{axis}", "missing")
        per_env.append(row)

    component_axes = [
        "base_env_reward",
        "control_cost",
        "survival_bonus",
        "potential_delta",
        "distribution_head",
        "terminal_bonus",
    ]
    executable_current_component_axes = [
        "base_env_reward",
        "control_cost",
        "survival_bonus",
    ]
    coverage = [_coverage_row(axis, [bool(row[axis]) for row in per_env]) for axis in component_axes]

    coupling_rows: list[dict[str, Any]] = []
    for component in component_axes:
        xs = ["present" if row[component] else "absent" for row in per_env]
        for struct in ["state_dim", "obs_dim", "action_dim", "terminal_target", "noise_dim", "zero_pad_dim"]:
            ys = [str(row.get(f"struct:{struct}", "missing")) for row in per_env]
            v = _cramers_v(xs, ys)
            if len(set(xs)) <= 1:
                status = "undefined_collapsed_component"
            elif v >= 0.50:
                status = "high"
            elif v >= 0.25:
                status = "medium"
            else:
                status = "low"
            coupling_rows.append(
                {
                    "component_axis": component,
                    "structure_axis": struct,
                    "cramers_v": v,
                    "status": status,
                }
            )

    missing_semantic_candidates = [
        row["axis"]
        for row in coverage
        if row["axis"] in set(executable_current_component_axes) and row["status"] == "missing"
    ]
    dormant_or_untrusted_candidates = [
        row["axis"]
        for row in coverage
        if row["axis"] in {"potential_delta", "distribution_head", "terminal_bonus"} and row["status"] == "missing"
    ]
    high_coupling = [row for row in coupling_rows if row["status"] == "high"]
    report = {
        "analysis_entry": "phase2_reward_component_boolean_axis_audit",
        "schema": "phase2_reward_component_boolean_axis_audit.v1",
        "contract": {
            "read_only": True,
            "does_not_modify_generator": True,
            "does_not_train": True,
            "separates_component_definition_from_generator_repair": True,
        },
        "read": {
            "component_boolean_audit_complete": True,
            "base_env_reward_present": next(row for row in coverage if row["axis"] == "base_env_reward")[
                "status"
            ]
            != "missing",
            "control_cost_has_support": next(row for row in coverage if row["axis"] == "control_cost")[
                "status"
            ]
            != "missing",
            "executable_current_component_axes": executable_current_component_axes,
            "missing_semantic_candidate_components": missing_semantic_candidates,
            "dormant_or_untrusted_candidate_components": dormant_or_untrusted_candidates,
            "high_structure_component_coupling_count": len(high_coupling),
            "generator_repair_justified_now": False,
            "generator_repair_justified_now_reason": (
                "current maintained Exact-SCM executable components are separated from lowtail-only or unlogged components"
            ),
        },
        "coverage": coverage,
        "structure_component_coupling": coupling_rows,
        "per_env": per_env,
        "inputs": {
            "fixed_csv": str(fixed_csv),
            "sidecar": str(sidecar),
            "topology_per_env_csv": str(Path(args.topology_per_env_csv).expanduser().resolve()),
        },
    }

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "reward_component_boolean_axis_audit.json"
    md_path = out_dir / "reward_component_boolean_axis_audit.md"
    per_env_csv = out_dir / "reward_component_boolean_axis_per_env.csv"
    coupling_csv = out_dir / "reward_component_boolean_axis_coupling.csv"
    coverage_csv = out_dir / "reward_component_boolean_axis_coverage.csv"
    report["outputs"] = {
        "json": str(json_path),
        "markdown": str(md_path),
        "per_env_csv": str(per_env_csv),
        "coupling_csv": str(coupling_csv),
        "coverage_csv": str(coverage_csv),
    }
    json_path.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(per_env_csv, per_env)
    _write_csv(coupling_csv, coupling_rows)
    _write_csv(coverage_csv, coverage)
    md_path.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report["outputs"], sort_keys=True))
    return report


def _markdown(report: dict[str, Any]) -> str:
    read = report["read"]
    lines = [
        "# Reward Component Boolean Axis Audit",
        "",
        "Read-only split of reward topology into boolean component axes.",
        "",
        "## Read",
        "",
        f"- component boolean audit complete: `{read['component_boolean_audit_complete']}`",
        f"- base env reward present: `{read['base_env_reward_present']}`",
        f"- control cost has support: `{read['control_cost_has_support']}`",
        f"- executable current component axes: `{read['executable_current_component_axes']}`",
        f"- missing semantic candidate components: `{read['missing_semantic_candidate_components']}`",
        f"- dormant or untrusted candidate components: `{read['dormant_or_untrusted_candidate_components']}`",
        f"- high structure-component coupling count: `{read['high_structure_component_coupling_count']}`",
        f"- generator repair justified now: `{read['generator_repair_justified_now']}`",
        f"- reason: {read['generator_repair_justified_now_reason']}",
        "",
        "## Coverage",
        "",
        "| component | status | true fraction | true | false |",
        "|---|---|---:|---:|---:|",
    ]
    for row in report["coverage"]:
        lines.append(
            f"| `{row['axis']}` | `{row['status']}` | {row['true_fraction']:.3f} | "
            f"{row['count:true']} | {row['count:false']} |"
        )
    highs = [row for row in report["structure_component_coupling"] if row["status"] == "high"]
    executable = set(read["executable_current_component_axes"])
    executable_missing = [
        row["axis"]
        for row in report["coverage"]
        if row["axis"] in executable and row["status"] == "missing"
    ]
    executable_covered = [
        row["axis"]
        for row in report["coverage"]
        if row["axis"] in executable and row["status"] != "missing"
    ]
    lines.extend(["", "## High Coupling", ""])
    if not highs:
        lines.append("- none")
    else:
        for row in highs:
            lines.append(
                f"- `{row['component_axis']}` vs `{row['structure_axis']}`: Cramer's V `{row['cramers_v']:.3f}`"
            )
    lines.extend(
        [
            "",
            "## Reduction",
            "",
            f"- executable current components covered: `{executable_covered}`",
            f"- executable current components missing: `{executable_missing}`",
            f"- dormant or untrusted candidates: `{read['dormant_or_untrusted_candidate_components']}`",
            "- This does not by itself authorize a generator patch; it authorizes a category decision about which reward components are required.",
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
