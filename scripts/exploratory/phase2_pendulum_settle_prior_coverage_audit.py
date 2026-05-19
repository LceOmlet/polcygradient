#!/usr/bin/env python
"""Audit prior coverage for Pendulum sustained settle candidates.

Read-only consolidation.  It checks existing Pendulum-like signature reports
for ``pendulum_like_settle_strict`` support so the closure process can separate
"anchor is defined" from "prior/profile lists contain enough trainable settle
candidates".
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


ART = Path("/home/chen/RLPFN/artifacts")
DEFAULT_SIGNATURE_REPORTS = (
    ART / "phase2_prior_pendulum_like_signature_probe_0509/gated_full_q90_n64_s128_obs4_scales3_timeprofile.json",
    ART
    / "phase2_prior_pendulum_like_signature_probe_seed6261_0509"
    / "gated_full_q90_seed6261_n64_s128_obs4_scales3_timeprofile.json",
    ART
    / "phase2_prior_pendulum_like_signature_probe_seed6262_0510"
    / "gated_full_q90_seed6262_n128_s128_obs4_scales3_timeprofile.json",
)
DEFAULT_OUTPUT_DIR = ART / "phase2_pendulum_settle_prior_coverage_audit_0511"


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def _count(coverage: dict[str, Any], key: str) -> int:
    try:
        return int(float(coverage.get(key, 0) or 0))
    except Exception:
        return 0


def _row(path: str | Path, item: dict[str, Any]) -> dict[str, Any]:
    coverage = ((item.get("classification", {}) or {}).get("coverage", {}) or {})
    n_envs = _count(coverage, "n_envs")
    settle = _count(coverage, "pendulum_like_settle_strict_count")
    strict = _count(coverage, "pendulum_like_strict_count")
    time_core = _count(coverage, "time_profile_core_count")
    suffix = _count(coverage, "suffix_supported_count")
    return {
        "report_path": str(Path(path).expanduser().resolve()),
        "label": str(item.get("label", "")),
        "n_envs": n_envs,
        "pendulum_like_settle_strict_count": settle,
        "pendulum_like_settle_strict_rate": settle / max(1, n_envs),
        "pendulum_like_strict_count": strict,
        "time_profile_core_count": time_core,
        "suffix_supported_count": suffix,
    }


def build_report(
    *,
    signature_reports: list[str | Path],
    min_settle_count: int = 12,
    min_settle_rate: float = 0.15,
) -> dict[str, Any]:
    rows = []
    for path in signature_reports:
        payload = _read_json(path)
        rows.extend(_row(path, item) for item in payload.get("lists", []) or [])
    min_count = min((row["pendulum_like_settle_strict_count"] for row in rows), default=0)
    max_count = max((row["pendulum_like_settle_strict_count"] for row in rows), default=0)
    mean_rate = (
        sum(float(row["pendulum_like_settle_strict_rate"]) for row in rows) / len(rows)
        if rows
        else 0.0
    )
    quota_ready = bool(
        rows
        and all(row["pendulum_like_settle_strict_count"] >= min_settle_count for row in rows)
        and all(row["pendulum_like_settle_strict_rate"] >= min_settle_rate for row in rows)
    )
    return {
        "analysis_entry": "phase2_pendulum_settle_prior_coverage_audit",
        "contract": {
            "read_only_existing_signature_reports": True,
            "does_not_sample_environments": True,
            "does_not_run_gym": True,
            "does_not_run_ppo": True,
            "does_not_train": True,
            "does_not_modify_exact_scm": True,
            "does_not_modify_fit_model": True,
            "does_not_modify_ppo": True,
        },
        "thresholds": {"min_settle_count": int(min_settle_count), "min_settle_rate": float(min_settle_rate)},
        "inputs": {"signature_reports": [str(Path(path).expanduser().resolve()) for path in signature_reports]},
        "rows": rows,
        "read": {
            "settle_candidates_exist": max_count > 0,
            "quota_ready": quota_ready,
            "min_settle_count": min_count,
            "max_settle_count": max_count,
            "mean_settle_rate": mean_rate,
            "blockers": [] if quota_ready else ["settle_strict_prior_coverage_too_sparse"],
            "interpretation": (
                "Prior reports contain sustained-settle candidates, but coverage is too sparse for a safe quota."
                if max_count > 0 and not quota_ready
                else "Prior sustained-settle coverage is quota-ready."
                if quota_ready
                else "No sustained-settle prior candidates were found."
            ),
        },
    }


def _markdown(report: dict[str, Any]) -> str:
    read = report["read"]
    lines = [
        "# Pendulum Settle Prior Coverage Audit",
        "",
        f"- settle candidates exist: `{read['settle_candidates_exist']}`",
        f"- quota ready: `{read['quota_ready']}`",
        f"- min/max settle count: `{read['min_settle_count']}` / `{read['max_settle_count']}`",
        f"- mean settle rate: `{read['mean_settle_rate']:.3f}`",
        f"- read: {read['interpretation']}",
        "",
        "## Rows",
        "",
        "| label | n | settle | rate | strict | time core | suffix |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["rows"]:
        lines.append(
            f"| `{row['label']}` | {row['n_envs']} | {row['pendulum_like_settle_strict_count']} | "
            f"{row['pendulum_like_settle_strict_rate']:.3f} | {row['pendulum_like_strict_count']} | "
            f"{row['time_profile_core_count']} | {row['suffix_supported_count']} |"
        )
    lines.extend(["", "## Files", ""])
    for key, value in report.get("outputs", {}).items():
        lines.append(f"- `{key}`: `{value}`")
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(
        signature_reports=list(args.signature_report),
        min_settle_count=int(args.min_settle_count),
        min_settle_rate=float(args.min_settle_rate),
    )
    json_path = out_dir / "pendulum_settle_prior_coverage_audit.json"
    md_path = out_dir / "pendulum_settle_prior_coverage_audit.md"
    report["outputs"] = {"json": str(json_path), "markdown": str(md_path)}
    json_path.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report["outputs"], sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signature-report", action="append", default=[str(path) for path in DEFAULT_SIGNATURE_REPORTS])
    parser.add_argument("--min-settle-count", type=int, default=12)
    parser.add_argument("--min-settle-rate", type=float, default=0.15)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
