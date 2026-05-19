#!/usr/bin/env python
"""Audit cheap Pendulum sustained-settle guard candidates.

The current strict settle label is too sparse.  This read-only audit asks
whether existing per-env signature fields already contain a cheaper guard that
is both broad enough and clean enough.  It does not run rollouts or add a new
profile axis.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Callable


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
DEFAULT_OUTPUT_DIR = ART / "phase2_pendulum_settle_light_guard_audit_0512"


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def _bool(row: dict[str, Any], key: str) -> bool:
    return bool(row.get(key))


GUARDS: dict[str, tuple[str, Callable[[dict[str, Any]], bool]]] = {
    "strict_current": (
        "current strict label: pendulum_like_core + suffix_supported + feedback_action_decay",
        lambda r: _bool(r, "pendulum_like_settle_strict"),
    ),
    "time_core_suffix": (
        "time-profile action decay plus suffix support; broad but may be impure",
        lambda r: _bool(r, "time_profile_core") and _bool(r, "time_profile_suffix_supported"),
    ),
    "time_core": (
        "time-profile beats open loop with action decay; broadest time-profile witness",
        lambda r: _bool(r, "time_profile_core"),
    ),
    "feedback_decay_suffix": (
        "feedback action decays and suffix reward beats zero",
        lambda r: _bool(r, "feedback_action_decay") and _bool(r, "suffix_supported"),
    ),
    "feedback_core_decay": (
        "feedback beats open loop, is phase-sensitive, and action decays",
        lambda r: _bool(r, "feedback_beats_open_loop_env")
        and _bool(r, "phase_sensitive_feedback")
        and _bool(r, "feedback_action_decay"),
    ),
    "strict_no_suffix": (
        "strict core plus action decay, but without suffix support",
        lambda r: _bool(r, "pendulum_like_core") and _bool(r, "feedback_action_decay"),
    ),
    "early_carryover": (
        "early-only action stops, but suffix/full return still beat zero",
        lambda r: _bool(r, "early_only_causal_carryover_core"),
    ),
    "early_strong_carryover": (
        "early-only action stops, and suffix/full return beat open-loop",
        lambda r: _bool(r, "early_only_strong_carryover_core"),
    ),
}


def _row(report_path: str | Path, label: str, rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    definition, fn = GUARDS[name]
    strict = [bool(GUARDS["strict_current"][1](item)) for item in rows]
    values = [bool(fn(item)) for item in rows]
    count = sum(values)
    strict_count = sum(strict)
    overlap = sum(value and target for value, target in zip(values, strict))
    n = len(rows)
    return {
        "report_path": str(Path(report_path).expanduser().resolve()),
        "label": label,
        "guard": name,
        "definition": definition,
        "n_envs": n,
        "count": count,
        "rate": count / max(1, n),
        "strict_count": strict_count,
        "overlap_with_strict": overlap,
        "precision_vs_strict": overlap / max(1, count),
        "recall_vs_strict": overlap / max(1, strict_count),
    }


def build_report(
    *,
    signature_reports: list[str | Path],
    min_count: int = 12,
    min_rate: float = 0.15,
    min_precision: float = 0.50,
    min_recall: float = 0.80,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in signature_reports:
        payload = _read_json(path)
        for item in payload.get("lists", []) or []:
            per_env = ((item.get("classification") or {}).get("per_env") or [])
            label = str(item.get("label", ""))
            for name in GUARDS:
                rows.append(_row(path, label, per_env, name))

    by_guard: dict[str, dict[str, Any]] = {}
    for name in GUARDS:
        items = [row for row in rows if row["guard"] == name]
        count_ready = bool(
            items
            and all(row["count"] >= min_count for row in items)
            and all(row["rate"] >= min_rate for row in items)
        )
        clean_ready = bool(
            items
            and all(row["precision_vs_strict"] >= min_precision for row in items)
            and all(row["recall_vs_strict"] >= min_recall for row in items)
        )
        by_guard[name] = {
            "definition": GUARDS[name][0],
            "count_ready": count_ready,
            "clean_ready": clean_ready,
            "candidate_ready": bool(count_ready and clean_ready),
            "min_count": min((row["count"] for row in items), default=0),
            "max_count": max((row["count"] for row in items), default=0),
            "min_rate": min((row["rate"] for row in items), default=0.0),
            "min_precision_vs_strict": min((row["precision_vs_strict"] for row in items), default=0.0),
            "min_recall_vs_strict": min((row["recall_vs_strict"] for row in items), default=0.0),
        }

    ready = [name for name, payload in by_guard.items() if payload["candidate_ready"]]
    broad_impure = [
        name for name, payload in by_guard.items() if payload["count_ready"] and not payload["clean_ready"]
    ]
    clean_sparse = [
        name for name, payload in by_guard.items() if payload["clean_ready"] and not payload["count_ready"]
    ]
    return {
        "analysis_entry": "phase2_pendulum_settle_light_guard_audit",
        "contract": {
            "read_only_existing_signature_reports": True,
            "does_not_sample_environments": True,
            "does_not_train": True,
            "does_not_modify_exact_scm": True,
            "does_not_modify_fit_model": True,
            "does_not_modify_ppo": True,
            "does_not_add_profile_axis": True,
        },
        "thresholds": {
            "min_count": int(min_count),
            "min_rate": float(min_rate),
            "min_precision_vs_strict": float(min_precision),
            "min_recall_vs_strict": float(min_recall),
        },
        "inputs": {"signature_reports": [str(Path(path).expanduser().resolve()) for path in signature_reports]},
        "by_guard": by_guard,
        "rows": rows,
        "decision": {
            "cheap_guard_ready": bool(ready),
            "ready_guards": ready,
            "broad_but_impure_guards": broad_impure,
            "clean_but_sparse_guards": clean_sparse,
            "read": (
                "No existing cheap guard is both broad enough and clean enough.  The widest guards meet count "
                "but are too impure; the cleaner guards remain sparse.  This points to a generator/yield problem, "
                "not a threshold-only reporting problem."
                if not ready
                else "At least one existing cheap guard is broad and clean enough for a candidate prior-side settle label."
            ),
            "next_action": (
                "Improve generation/yield for clean feedback-decay settle candidates or design a new cheap suffix-state "
                "settle label; do not promote time_core-only guards to quota because they over-admit non-settle cases."
                if not ready
                else "Validate the ready guard on fresh full/q90 lists and diversity smoke before using it in a selector."
            ),
        },
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Pendulum Settle Light Guard Audit",
        "",
        "Read-only audit of existing per-env signature fields.",
        "",
        "## Verdict",
        "",
        f"- cheap guard ready: `{report['decision']['cheap_guard_ready']}`",
        f"- ready guards: `{report['decision']['ready_guards']}`",
        f"- broad but impure: `{report['decision']['broad_but_impure_guards']}`",
        f"- clean but sparse: `{report['decision']['clean_but_sparse_guards']}`",
        f"- read: {report['decision']['read']}",
        f"- next action: {report['decision']['next_action']}",
        "",
        "## Guard Summary",
        "",
        "| guard | count ready | clean ready | min count | max count | min precision | min recall |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, payload in report["by_guard"].items():
        lines.append(
            f"| `{name}` | `{payload['count_ready']}` | `{payload['clean_ready']}` | "
            f"{payload['min_count']} | {payload['max_count']} | "
            f"{payload['min_precision_vs_strict']:.3f} | {payload['min_recall_vs_strict']:.3f} |"
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
        min_count=args.min_count,
        min_rate=args.min_rate,
        min_precision=args.min_precision,
        min_recall=args.min_recall,
    )
    json_path = out_dir / "pendulum_settle_light_guard_audit.json"
    md_path = out_dir / "pendulum_settle_light_guard_audit.md"
    report["outputs"] = {"json": str(json_path), "markdown": str(md_path)}
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report["outputs"], sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signature-report", action="append", default=[str(path) for path in DEFAULT_SIGNATURE_REPORTS])
    parser.add_argument("--min-count", type=int, default=12)
    parser.add_argument("--min-rate", type=float, default=0.15)
    parser.add_argument("--min-precision", type=float, default=0.50)
    parser.add_argument("--min-recall", type=float, default=0.80)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
