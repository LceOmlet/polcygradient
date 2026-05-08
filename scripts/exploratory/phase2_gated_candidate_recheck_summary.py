#!/usr/bin/env python
"""Summarize repeated gated-candidate generation and short health smokes."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_OUTPUT_DIR = "/home/chen/RLPFN/artifacts/phase2_exploratory_gated_candidate_recheck_summary_0506"
DEFAULT_RUNS = {
    "seed6260": {
        "probe_dir": "/home/chen/RLPFN/artifacts/phase2_exploratory_reward_group_balance_gated_n64_0506",
        "guard_report": "/home/chen/RLPFN/artifacts/phase2_exploratory_gated_fixedlist_guard_audit_0506/gated_fixedlist_guard_audit_report.json",
        "full_progress": "/home/chen/RLPFN/artifacts/phase2_exploratory_reward_group_balance_gated_full_fixedlist_n64_s512_u4_e4_advnorm_0506/prior_progress.jsonl",
        "q90_progress": "/home/chen/RLPFN/artifacts/phase2_exploratory_reward_group_balance_gated_q90_fixedlist_n64_s512_u4_e4_advnorm_0506/prior_progress.jsonl",
    },
    "seed6261": {
        "probe_dir": "/home/chen/RLPFN/artifacts/phase2_exploratory_reward_group_balance_gated_n64_seed6261_0506",
        "guard_report": "/home/chen/RLPFN/artifacts/phase2_exploratory_gated_fixedlist_guard_audit_seed6261_0506/gated_fixedlist_guard_audit_report.json",
        "full_progress": "/home/chen/RLPFN/artifacts/phase2_exploratory_reward_group_balance_gated_full_fixedlist_seed6261_n64_s512_u4_e4_advnorm_0506/prior_progress.jsonl",
        "q90_progress": "/home/chen/RLPFN/artifacts/phase2_exploratory_reward_group_balance_gated_q90_fixedlist_seed6261_n64_s512_u4_e4_advnorm_0506/prior_progress.jsonl",
    },
}


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


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _progress_summary(path: str | Path) -> dict[str, Any]:
    rows = _load_jsonl(path)
    out: dict[str, Any] = {
        "path": str(Path(path).expanduser().resolve()),
        "updates": int(len(rows)),
        "hard_pass": bool(rows) and all((row.get("update") or {}).get("exception") is None for row in rows),
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
            "raw_reward_return_mean_delta": float(last["raw_reward_return_mean"] - first["raw_reward_return_mean"]),
            "done_count_min": int(min(item["done_count"] for item in series)),
            "done_count_max": int(max(item["done_count"] for item in series)),
            "reward_std_min": float(min(item["reward_std"] for item in series)),
            "reward_std_max": float(max(item["reward_std"] for item in series)),
        }
    )
    return out


def _run_summary(name: str, paths: dict[str, str]) -> dict[str, Any]:
    probe = _load_json(Path(paths["probe_dir"]) / "reward_group_balance_probe_report.json")
    guard = _load_json(paths["guard_report"])
    selected = _load_csv(Path(paths["probe_dir"]) / "selected_prior_envs.csv")
    repair_counts = dict(Counter(row["selected_repair_kind"] for row in selected))
    by_rule = {}
    for rule in sorted({row["rule"] for row in selected}):
        rows = [row for row in selected if row["rule"] == rule]
        by_rule[rule] = {
            "n": int(len(rows)),
            "state_dim_unique": int(len({row["state_dim"] for row in rows})),
            "obs_dim_unique": int(len({row["obs_dim"] for row in rows})),
            "action_dim_unique": int(len({row["action_dim"] for row in rows})),
            "noise_dim_unique": int(len({row["noise_dim"] for row in rows})),
        }
    full = _progress_summary(paths["full_progress"])
    q90 = _progress_summary(paths["q90_progress"])
    hard_pass = (
        bool(guard.get("hard_pass"))
        and bool(full.get("hard_pass"))
        and bool(q90.get("hard_pass"))
        and float(full.get("raw_reward_return_mean_delta", -float("inf"))) >= 0.0
        and float(q90.get("raw_reward_return_mean_delta", -float("inf"))) >= 0.0
    )
    return {
        "name": name,
        "probe_dir": str(Path(paths["probe_dir"]).expanduser().resolve()),
        "probe_sampling": probe.get("sampling", {}),
        "selected_count_by_rule": probe.get("selected_count_by_rule", {}),
        "repair_counts": repair_counts,
        "support_by_rule": by_rule,
        "guard_hard_pass": bool(guard.get("hard_pass")),
        "smoke_full": full,
        "smoke_q90": q90,
        "hard_pass": bool(hard_pass),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    runs = {name: _run_summary(name, paths) for name, paths in DEFAULT_RUNS.items()}
    all_pass = all(bool(run["hard_pass"]) for run in runs.values())
    report = {
        "analysis_entry": "phase2_gated_candidate_recheck_summary",
        "exploratory_only": True,
        "hard_pass": bool(all_pass),
        "contract": {
            "does_not_modify_exact_scm": True,
            "does_not_modify_default_prior_generator": True,
            "short_smoke_only_not_longrun_claim": True,
            "candidate_reproducibility_check": True,
        },
        "runs": runs,
        "decision": {
            "keep_gated_reward_path_balancing_as_candidate_repair": bool(all_pass),
            "do_not_set_new_milestone_yet": True,
            "next_step": (
                "Stop broad anomaly hunting unless a larger deterministic mismatch appears. "
                "The current largest confirmed mechanism remains reward-path width bias; "
                "prepare a formal exploratory generator-level wrapper only if we decide to "
                "compare this candidate against the existing milestone."
            ),
        },
    }
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "gated_candidate_recheck_summary_report.json"
    report_path.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Gated Candidate Recheck Summary",
        "",
        f"hard_pass: {all_pass}",
        "",
        "## Runs",
    ]
    for name, run in runs.items():
        lines.append(
            f"- {name}: guard={run['guard_hard_pass']}, hard={run['hard_pass']}, "
            f"full_delta={run['smoke_full'].get('raw_reward_return_mean_delta')}, "
            f"q90_delta={run['smoke_q90'].get('raw_reward_return_mean_delta')}, "
            f"repairs={run['repair_counts']}"
        )
    lines.extend(
        [
            "",
            "Decision: keep gated reward-path balancing as candidate repair; no terminal/survival change; no new milestone yet.",
        ]
    )
    md_path = out_dir / "gated_candidate_recheck_summary.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps(_json_safe({"hard_pass": report["hard_pass"], "output_dir": args.output_dir}), sort_keys=True))


if __name__ == "__main__":
    main()
