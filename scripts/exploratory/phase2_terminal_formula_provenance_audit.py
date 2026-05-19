#!/usr/bin/env python
"""Read-only provenance audit for terminal formula topology.

This audit checks terminal mechanism source labels from the fixed h-list.  It
does not use done_count as a source axis and does not edit the generator.
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
DEFAULT_OUTPUT_DIR = ART / "phase2_terminal_formula_provenance_audit_realized_noise_zero_0518"


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).expanduser().resolve().open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


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


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _load_h_rows(fixed_csv: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in _read_csv(fixed_csv):
        raw = str(row.get("full_frozen_h_json", "")).strip()
        path = Path(raw)
        rows.append(_read_json(path) if path.exists() else json.loads(raw))
    return rows


def _terminal_formula(h: dict[str, Any]) -> str:
    if not _safe_bool(h.get("terminal_reset_enabled")):
        return "terminal_off"
    lowtail_active = _safe_bool(h.get("exact_scm_gym_lowtail_family_selected")) or _safe_bool(
        h.get("exact_scm_gym_lowtail_family_enabled")
    )
    if lowtail_active and _safe_bool(h.get("exact_scm_gym_lowtail_shared_energy_terminal_enabled")):
        return "shared_energy_terminal"
    if _safe_bool(h.get("terminal_reset_min_step_enabled")):
        return "min_step_hazard_reset"
    if _safe_bool(h.get("exact_scm_gym_lowtail_terminal_override_enabled")):
        return "override_hazard_reset"
    return "hazard_reset"


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


def run(args: argparse.Namespace) -> dict[str, Any]:
    fixed_csv = Path(args.fixed_csv).expanduser().resolve()
    h_rows = _load_h_rows(fixed_csv)
    struct_by_idx = {int(row["env_idx"]): row for row in _read_csv(args.topology_per_env_csv)}
    per_env: list[dict[str, Any]] = []
    for idx, h in enumerate(h_rows):
        struct = struct_by_idx.get(idx, {})
        row: dict[str, Any] = {
            "env_idx": idx,
            "terminal_formula_topology": _terminal_formula(h),
            "terminal_reset_enabled": _safe_bool(h.get("terminal_reset_enabled")),
            "terminal_reset_count_target": _safe_float(h.get("terminal_reset_count_target")),
            "terminal_reset_min_step_enabled": _safe_bool(h.get("terminal_reset_min_step_enabled")),
            "terminal_reset_min_step_target": _safe_float(h.get("terminal_reset_min_step_target")),
            "exact_scm_gym_lowtail_terminal_override_enabled": _safe_bool(
                h.get("exact_scm_gym_lowtail_terminal_override_enabled")
            ),
            "exact_scm_gym_lowtail_shared_energy_terminal_enabled": _safe_bool(
                h.get("exact_scm_gym_lowtail_shared_energy_terminal_enabled")
            ),
        }
        for axis in ["state_dim", "obs_dim", "action_dim", "terminal_target", "noise_dim", "zero_pad_dim"]:
            row[f"struct:{axis}"] = struct.get(f"struct:{axis}", "missing")
        per_env.append(row)

    executable_expected = [
        "terminal_off",
        "hazard_reset",
        "override_hazard_reset",
        "min_step_hazard_reset",
    ]
    optional_lowtail_expected = [
        "shared_energy_terminal",
    ]
    expected = executable_expected + optional_lowtail_expected
    labels = [str(row["terminal_formula_topology"]) for row in per_env]
    coupling: list[dict[str, Any]] = []
    for struct in ["state_dim", "obs_dim", "action_dim", "terminal_target", "noise_dim", "zero_pad_dim"]:
        ys = [str(row.get(f"struct:{struct}", "missing")) for row in per_env]
        v = _cramers_v(labels, ys)
        if len(set(labels)) <= 1:
            status = "undefined_collapsed_topology"
        elif v >= 0.50:
            status = "high"
        elif v >= 0.25:
            status = "medium"
        else:
            status = "low"
        coupling.append({"structure_axis": struct, "cramers_v": v, "status": status})

    cov = _coverage(labels, expected)
    executable_cov = _coverage(labels, executable_expected)
    missing = [label for label in executable_expected if executable_cov.get(f"count:{label}", 0) == 0]
    missing_optional_lowtail = [label for label in optional_lowtail_expected if cov.get(f"count:{label}", 0) == 0]
    report = {
        "analysis_entry": "phase2_terminal_formula_provenance_audit",
        "schema": "phase2_terminal_formula_provenance_audit.v1",
        "contract": {
            "read_only": True,
            "does_not_modify_generator": True,
            "does_not_train": True,
            "done_count_is_not_used_as_source_axis": True,
        },
        "read": {
            "terminal_formula_provenance_audit_complete": True,
            "coverage": cov,
            "executable_current_coverage": executable_cov,
            "missing_formula_labels": missing,
            "missing_optional_lowtail_formula_labels": missing_optional_lowtail,
            "high_structure_terminal_formula_coupling_count": sum(1 for row in coupling if row["status"] == "high"),
            "generator_repair_justified_now": False,
            "generator_repair_justified_now_reason": (
                "current maintained Exact-SCM executable terminal labels are separated from lowtail-only labels"
            ),
        },
        "per_env": per_env,
        "structure_terminal_formula_coupling": coupling,
        "inputs": {
            "fixed_csv": str(fixed_csv),
            "topology_per_env_csv": str(Path(args.topology_per_env_csv).expanduser().resolve()),
        },
    }
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "terminal_formula_provenance_audit.json"
    md_path = out_dir / "terminal_formula_provenance_audit.md"
    per_env_csv = out_dir / "terminal_formula_provenance_per_env.csv"
    coupling_csv = out_dir / "terminal_formula_provenance_coupling.csv"
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
    cov = read["coverage"]
    executable_cov = read["executable_current_coverage"]
    lines = [
        "# Terminal Formula Provenance Audit",
        "",
        "Read-only source/formula audit. `done_count` is not treated as a source axis.",
        "",
        "## Read",
        "",
        f"- terminal formula provenance audit complete: `{read['terminal_formula_provenance_audit_complete']}`",
        f"- coverage: `{cov}`",
        f"- executable current coverage: `{executable_cov}`",
        f"- missing formula labels: `{read['missing_formula_labels']}`",
        f"- missing optional lowtail formula labels: `{read['missing_optional_lowtail_formula_labels']}`",
        f"- high structure-terminal formula coupling count: `{read['high_structure_terminal_formula_coupling_count']}`",
        f"- generator repair justified now: `{read['generator_repair_justified_now']}`",
        f"- reason: {read['generator_repair_justified_now_reason']}",
        "",
        "## Reduction",
        "",
        "- Current fixed family collapses to one terminal formula label if coverage is `collapsed`.",
        "- This is a category-design decision, not an automatic failure.",
        "- If Gym-and-nearby category requires boundary/min-step/shared-energy/no-terminal mechanisms, repair must add an explicit terminal formula source independent of terminal target.",
        "",
        "## Outputs",
        "",
    ]
    for key, value in report["outputs"].items():
        lines.append(f"- `{key}`: `{value}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-csv", default=str(DEFAULT_FIXED_CSV))
    parser.add_argument("--topology-per-env-csv", default=str(DEFAULT_TOPOLOGY_PER_ENV))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
