#!/usr/bin/env python
"""Export prior-internal Pendulum settle axis labels as cheap cached labels.

This exporter is deliberately O(n) over an existing source pool or fixed h
list.  It does not sample environments, run PPO, or evaluate a temporal grid.
The label means "the prior-internal settle-yield axis/repair is enabled on this
h", not "the full temporal settle reference guard has passed".  An optional
reference preflight JSON can be supplied to calibrate that cheap label.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


RULES = ("gym_full_obs_action_range", "gym_q90_obs_action_range")
ART = Path("/home/chen/RLPFN/artifacts")
DEFAULT_OUTPUT_DIR = ART / "phase2_pendulum_settle_axis_cheap_label_export_0512"


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


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


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    if not fields:
        fields = ["rule", "source_index"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _infer_rule(path: str | Path, explicit: str | None = None) -> str:
    if explicit:
        if explicit not in RULES:
            raise ValueError(f"unknown rule {explicit!r}")
        return explicit
    text = str(path)
    for rule in RULES:
        if rule in text:
            return rule
    raise ValueError(
        "could not infer rule from fixed h path; pass --fixed-h-list-json as RULE=PATH"
    )


def _axis_enabled_from_mapping(row: dict[str, Any]) -> bool:
    return (
        _as_bool(row.get("pendulum_settle_yield_axis_enabled"))
        or _as_bool(row.get("pendulum_settle_yield_repair_enabled"))
        or _as_bool(row.get("pendulum_settle_yield_repair_planned"))
    )


def _row_from_mapping(rule: str, source_index: int, row: dict[str, Any], *, source: str) -> dict[str, Any]:
    axis_enabled = _axis_enabled_from_mapping(row)
    # This cheap exporter is intentionally file-only.  It can prove that the
    # prior-internal action/state reward-basin mechanism is present, but it does
    # not evaluate a rollout-level short gap or sustained closed-loop settle.
    # Those stronger labels must come from later calibrated probes.
    return {
        "rule": rule,
        "source_index": int(source_index),
        "env_index": int(source_index),
        "pendulum_axis_support": bool(axis_enabled),
        "pendulum_short_gap": False,
        "pendulum_sustained_settle": False,
        "pendulum_profile_quota_ready": False,
        "pendulum_short_gap_label_available": False,
        "pendulum_sustained_settle_label_available": False,
        "pendulum_profile_quota_ready_label_available": False,
        # Legacy compatibility field.  In this split schema it aliases the
        # strict sustained-settle label, not mere axis support.
        "pendulum_short_temporal_settle_guard": False,
        "pendulum_label_available": True,
        "pendulum_settle_guard_precision_class": "axis_support_only_reference_required",
        "pendulum_axis_cheap_label_source": source,
        "pendulum_settle_yield_axis_enabled": _as_bool(row.get("pendulum_settle_yield_axis_enabled")),
        "pendulum_settle_yield_repair_enabled": _as_bool(row.get("pendulum_settle_yield_repair_enabled")),
        "pendulum_settle_yield_repair_planned": _as_bool(row.get("pendulum_settle_yield_repair_planned")),
        "pendulum_settle_yield_axis_bin": row.get("pendulum_settle_yield_axis_bin", ""),
        "pendulum_settle_yield_axis_action_gain": row.get("pendulum_settle_yield_axis_action_gain", ""),
        "pendulum_settle_yield_axis_damping": row.get("pendulum_settle_yield_axis_damping", ""),
        "pendulum_settle_yield_repair_source": row.get("pendulum_settle_yield_repair_source", ""),
        "label_key": f"{rule}:{int(source_index)}",
    }


def rows_from_selected_csv(path: str | Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    ordinal_by_rule = {rule: 0 for rule in RULES}
    for row in _read_csv(path):
        rule = str(row.get("rule", ""))
        if rule not in RULES:
            continue
        idx = ordinal_by_rule[rule]
        ordinal_by_rule[rule] += 1
        source_index = int(row.get("_source_index", idx) or idx)
        out.append(_row_from_mapping(rule, source_index, row, source="selected_csv"))
    return out


def _parse_fixed_h_arg(value: str) -> tuple[str | None, Path]:
    if "=" in value:
        rule, path = value.split("=", 1)
        return rule.strip(), Path(path).expanduser().resolve()
    return None, Path(value).expanduser().resolve()


def rows_from_fixed_h_lists(values: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for value in values:
        explicit_rule, path = _parse_fixed_h_arg(value)
        rule = _infer_rule(path, explicit_rule)
        h_list = _read_json(path)
        for idx, h in enumerate(h_list):
            out.append(_row_from_mapping(rule, idx, h, source="fixed_h_list"))
    return out


def _reference_rows(report: dict[str, Any]) -> dict[tuple[str, int], bool]:
    out: dict[tuple[str, int], bool] = {}
    for item in report.get("lists", []) or []:
        rule = str(item.get("label", ""))
        if rule not in RULES:
            continue
        for row in (item.get("classification", {}) or {}).get("per_env", []) or []:
            idx = int(row.get("source_index", row.get("env_index", 0)))
            out[(rule, idx)] = _as_bool(row.get("cheap_settle_guard"))
    return out


def _summaries(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for rule in RULES:
        rr = [row for row in rows if row.get("rule") == rule]
        if not rr:
            continue
        n = len(rr)
        axis = sum(_as_bool(row.get("pendulum_axis_support")) for row in rr)
        short_gap = sum(_as_bool(row.get("pendulum_short_gap")) for row in rr)
        sustained = sum(_as_bool(row.get("pendulum_sustained_settle")) for row in rr)
        quota = sum(_as_bool(row.get("pendulum_profile_quota_ready")) for row in rr)
        out[rule] = {
            "n": n,
            "pendulum_axis_support_count": int(axis),
            "pendulum_axis_support_rate": float(axis / max(1, n)),
            "pendulum_short_gap_count": int(short_gap),
            "pendulum_short_gap_rate": float(short_gap / max(1, n)),
            "pendulum_sustained_settle_count": int(sustained),
            "pendulum_sustained_settle_rate": float(sustained / max(1, n)),
            "pendulum_profile_quota_ready_count": int(quota),
            "pendulum_profile_quota_ready_rate": float(quota / max(1, n)),
            # Backward-compatible names for old reports.  They now describe
            # axis support, not quota readiness.
            "cheap_guard_count": int(axis),
            "cheap_guard_rate": float(axis / max(1, n)),
            "axis_bin_counts": _counts(row.get("pendulum_settle_yield_axis_bin", "") for row in rr),
            "repair_source_counts": _counts(row.get("pendulum_settle_yield_repair_source", "") for row in rr),
        }
    return out


def _counts(values: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        key = str(value)
        out[key] = out.get(key, 0) + 1
    return out


def _calibration(rows: list[dict[str, Any]], reference_json: str) -> dict[str, Any]:
    if not reference_json:
        return {"enabled": False}
    ref = _reference_rows(_read_json(reference_json))
    by_rule: dict[str, Any] = {}
    for rule in RULES:
        rr = [row for row in rows if row.get("rule") == rule]
        tp = fp = fn = tn = missing = 0
        for row in rr:
            key = (rule, int(row["source_index"]))
            cheap = _as_bool(row.get("pendulum_axis_support"))
            if key not in ref:
                missing += 1
                continue
            reference = bool(ref[key])
            if cheap and reference:
                tp += 1
            elif cheap and not reference:
                fp += 1
            elif (not cheap) and reference:
                fn += 1
            else:
                tn += 1
        precision = None if tp + fp == 0 else tp / (tp + fp)
        recall = None if tp + fn == 0 else tp / (tp + fn)
        by_rule[rule] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "missing_reference": missing,
            "precision": precision,
            "recall": recall,
            "axis_support_positive_count": tp + fp,
            "cheap_positive_count": tp + fp,
            "reference_positive_count": tp + fn,
        }
    return {
        "enabled": True,
        "reference_json": str(Path(reference_json).expanduser().resolve()),
        "by_rule": by_rule,
        "all_rules_reference_positive": all(item["reference_positive_count"] > 0 for item in by_rule.values()),
        "quota_ready": all(
            (item["precision"] is not None and item["precision"] >= 0.8 and item["recall"] is not None and item["recall"] >= 0.8)
            for item in by_rule.values()
        ),
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Pendulum Settle Axis Cheap Label Export",
        "",
        "O(n) cached labels from existing prior-internal settle-yield axis/repair fields.",
        "",
        "## Coverage",
        "",
        "| rule | n | axis support | short gap | sustained settle | quota ready |",
        "|---|---:|---:|---:|",
    ]
    for rule, item in report["summaries"].items():
        lines.append(
            f"| `{rule}` | {item['n']} | "
            f"{item['pendulum_axis_support_count']} ({item['pendulum_axis_support_rate']:.3f}) | "
            f"{item['pendulum_short_gap_count']} ({item['pendulum_short_gap_rate']:.3f}) | "
            f"{item['pendulum_sustained_settle_count']} ({item['pendulum_sustained_settle_rate']:.3f}) | "
            f"{item['pendulum_profile_quota_ready_count']} ({item['pendulum_profile_quota_ready_rate']:.3f}) |"
        )
    if report["calibration"].get("enabled"):
        lines.extend(["", "## Reference Calibration", "", "| rule | cheap+ | ref+ | TP | FP | FN | precision | recall |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
        for rule, item in report["calibration"]["by_rule"].items():
            lines.append(
                f"| `{rule}` | {item['axis_support_positive_count']} | {item['reference_positive_count']} | "
                f"{item['tp']} | {item['fp']} | {item['fn']} | {item['precision']} | {item['recall']} |"
            )
    lines.extend(["", "## Read", "", report["read"]["summary"]])
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    if args.selected_csv:
        rows.extend(rows_from_selected_csv(args.selected_csv))
    if args.fixed_h_list_json:
        rows.extend(rows_from_fixed_h_lists(args.fixed_h_list_json))
    if not rows:
        raise ValueError("provide --selected-csv and/or --fixed-h-list-json")
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "pendulum_short_temporal_settle_per_env.csv"
    json_path = out_dir / "pendulum_settle_axis_cheap_label_export.json"
    md_path = out_dir / "pendulum_settle_axis_cheap_label_export.md"
    _write_csv(csv_path, rows)
    calibration = _calibration(rows, str(args.reference_preflight_json or ""))
    report = {
        "analysis_entry": "phase2_pendulum_settle_axis_cheap_label_export",
        "schema": "phase2_pendulum_settle_axis_cheap_label_export.v1",
        "contract": {
            "exploratory_only": True,
            "does_not_sample_environments": True,
            "does_not_train": True,
            "does_not_modify_exact_scm": True,
            "does_not_modify_fit_model": True,
            "does_not_modify_ppo": True,
            "profile_quality_first": True,
            "raw_delta_not_used": True,
            "cheap_path_is_file_only": True,
            "reference_calibration_optional_offline": True,
        },
        "inputs": {
            "selected_csv": str(Path(args.selected_csv).expanduser().resolve()) if args.selected_csv else None,
            "fixed_h_list_json": list(args.fixed_h_list_json or []),
            "reference_preflight_json": str(Path(args.reference_preflight_json).expanduser().resolve())
            if args.reference_preflight_json
            else None,
        },
        "summaries": _summaries(rows),
        "calibration": calibration,
        "read": {
            "cheap_label_populated": bool(rows),
            "axis_support_label_populated": any(
                item["pendulum_axis_support_count"] > 0 for item in _summaries(rows).values()
            ),
            "short_gap_label_populated": any(
                item["pendulum_short_gap_count"] > 0 for item in _summaries(rows).values()
            ),
            "sustained_settle_label_populated": any(
                item["pendulum_sustained_settle_count"] > 0 for item in _summaries(rows).values()
            ),
            "cheap_label_quota_ready": bool(calibration.get("quota_ready", False)),
            "summary": (
                "Cheap labels split Pendulum support into axis_support, short_gap, "
                "sustained_settle, and profile_quota_ready.  This file-only exporter "
                "only certifies axis_support; stronger labels require calibrated rollout "
                "audits and health/diversity evidence before quota use."
            ),
        },
        "outputs": {"csv": str(csv_path), "json": str(json_path), "markdown": str(md_path)},
    }
    json_path.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report["outputs"], sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-csv", default="")
    parser.add_argument("--fixed-h-list-json", action="append", default=None)
    parser.add_argument("--reference-preflight-json", default="")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
