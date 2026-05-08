#!/usr/bin/env python
"""Summarize gated fixed-list guard and short PPO smoke outputs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_GUARD_REPORT = (
    "/home/chen/RLPFN/artifacts/"
    "phase2_exploratory_gated_fixedlist_guard_audit_0506/"
    "gated_fixedlist_guard_audit_report.json"
)
DEFAULT_FULL_PROGRESS = (
    "/home/chen/RLPFN/artifacts/"
    "phase2_exploratory_reward_group_balance_gated_full_fixedlist_n64_s512_u4_e4_advnorm_0506/"
    "prior_progress.jsonl"
)
DEFAULT_Q90_PROGRESS = (
    "/home/chen/RLPFN/artifacts/"
    "phase2_exploratory_reward_group_balance_gated_q90_fixedlist_n64_s512_u4_e4_advnorm_0506/"
    "prior_progress.jsonl"
)
DEFAULT_OUTPUT_DIR = (
    "/home/chen/RLPFN/artifacts/"
    "phase2_exploratory_gated_fixedlist_smoke_summary_0506"
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def _load_progress(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _summarize_progress(path: str | Path) -> dict[str, Any]:
    rows = _load_progress(path)
    out: dict[str, Any] = {
        "path": str(Path(path).expanduser().resolve()),
        "updates": int(len(rows)),
        "hard_pass": len(rows) > 0 and all((row.get("update") or {}).get("exception") is None for row in rows),
    }
    if not rows:
        return out
    series = []
    for idx, row in enumerate(rows):
        stats = (row.get("pack_summary") or {}).get("stats") or {}
        logger = (row.get("update") or {}).get("logger") or {}
        series.append(
            {
                "idx": int(idx),
                "raw_reward_return_mean": stats.get("raw_reward_return_mean"),
                "done_count": stats.get("done_count"),
                "raw_reward_std": stats.get("raw_reward_std"),
                "reward_std": stats.get("reward_std"),
                "action_absmean": stats.get("action_absmean"),
                "approx_kl": logger.get("train/approx_kl"),
                "clip_fraction": logger.get("train/clip_fraction"),
            }
        )
    first = series[0]
    last = series[-1]
    out.update(
        {
            "series": series,
            "raw_reward_return_mean_first": first["raw_reward_return_mean"],
            "raw_reward_return_mean_last": last["raw_reward_return_mean"],
            "raw_reward_return_mean_delta": (
                None
                if first["raw_reward_return_mean"] is None or last["raw_reward_return_mean"] is None
                else float(last["raw_reward_return_mean"] - first["raw_reward_return_mean"])
            ),
            "done_count_min": min(int(item["done_count"]) for item in series if item["done_count"] is not None),
            "done_count_max": max(int(item["done_count"]) for item in series if item["done_count"] is not None),
            "reward_std_min": min(float(item["reward_std"]) for item in series if item["reward_std"] is not None),
            "reward_std_max": max(float(item["reward_std"]) for item in series if item["reward_std"] is not None),
        }
    )
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    guard = _load_json(args.guard_report)
    full = _summarize_progress(args.full_progress)
    q90 = _summarize_progress(args.q90_progress)
    smoke_pass = (
        bool(guard.get("hard_pass"))
        and bool(full.get("hard_pass"))
        and bool(q90.get("hard_pass"))
        and float(full.get("raw_reward_return_mean_delta", -float("inf"))) >= 0.0
        and float(q90.get("raw_reward_return_mean_delta", -float("inf"))) >= 0.0
        and int(full.get("done_count_min", 0)) > 0
        and int(q90.get("done_count_min", 0)) > 0
    )
    report = {
        "analysis_entry": "phase2_gated_fixedlist_smoke_summary",
        "exploratory_only": True,
        "hard_pass": bool(smoke_pass),
        "contract": {
            "does_not_modify_exact_scm": True,
            "does_not_modify_default_prior_generator": True,
            "short_smoke_only_not_longrun_claim": True,
        },
        "guard": {
            "path": str(Path(args.guard_report).expanduser().resolve()),
            "hard_pass": bool(guard.get("hard_pass")),
        },
        "smoke": {
            "full": full,
            "q90": q90,
        },
        "decision": {
            "can_check_other_obvious_anomalies_next": bool(smoke_pass),
            "can_set_milestone_now": False,
            "reason": (
                "Guard and short PPO smoke passed without early health/diversity/optimibility regression. "
                "This is enough to continue anomaly triage, not enough to declare a new milestone."
            ),
        },
    }
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "gated_fixedlist_smoke_summary_report.json"
    report_path.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Gated Fixed-List Smoke Summary",
        "",
        f"hard_pass: {smoke_pass}",
        "",
        f"guard_hard_pass: {guard.get('hard_pass')}",
        "",
        "## Smoke",
    ]
    for name, item in (("full", full), ("q90", q90)):
        lines.append(
            f"- {name}: updates={item.get('updates')}, "
            f"raw_delta={item.get('raw_reward_return_mean_delta')}, "
            f"first={item.get('raw_reward_return_mean_first')}, "
            f"last={item.get('raw_reward_return_mean_last')}, "
            f"done_min={item.get('done_count_min')}, "
            f"done_max={item.get('done_count_max')}"
        )
    lines.extend(
        [
            "",
            "Decision: continue checking only more obvious anomalies. Do not set a new milestone from this short smoke alone.",
        ]
    )
    md_path = out_dir / "gated_fixedlist_smoke_summary.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--guard-report", type=str, default=DEFAULT_GUARD_REPORT)
    parser.add_argument("--full-progress", type=str, default=DEFAULT_FULL_PROGRESS)
    parser.add_argument("--q90-progress", type=str, default=DEFAULT_Q90_PROGRESS)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps(_json_safe({"hard_pass": report["hard_pass"], "output_dir": args.output_dir}), sort_keys=True))


if __name__ == "__main__":
    main()
