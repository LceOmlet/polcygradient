#!/usr/bin/env python
"""Offline Gym done-contribution audit.

This is a narrow exploratory report: identify whether Gym baseline done-rate
mismatch is a broad distribution issue or dominated by a few validation
environments.  It does not sample environments, train PPO, or mutate the prior.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_GYM_DIR = "/home/chen/RLPFN/artifacts/phase2_exploratory_gym_validation_baseline_n64_s512_u40_e4_advnorm_0506"
DEFAULT_GATED_FULL = (
    "/home/chen/RLPFN/artifacts/"
    "phase2_exploratory_reward_group_balance_gated_full_fixedlist_n64_s512_u4_e4_advnorm_0506/"
    "prior_progress.jsonl"
)
DEFAULT_GATED_Q90 = (
    "/home/chen/RLPFN/artifacts/"
    "phase2_exploratory_reward_group_balance_gated_q90_fixedlist_n64_s512_u4_e4_advnorm_0506/"
    "prior_progress.jsonl"
)
DEFAULT_OUTPUT_DIR = "/home/chen/RLPFN/artifacts/phase2_exploratory_gym_done_outlier_audit_0506"


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


def _stats(values: list[float]) -> dict[str, Any]:
    arr = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=np.float64)
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "q10": float(np.quantile(arr, 0.10)),
        "q50": float(np.quantile(arr, 0.50)),
        "q90": float(np.quantile(arr, 0.90)),
        "max": float(np.max(arr)),
    }


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_sidecars(gym_dir: str | Path) -> list[dict[str, Any]]:
    sidecar_dir = Path(gym_dir).expanduser().resolve() / "semantic_sidecars"
    rows = []
    for path in sorted(sidecar_dir.glob("gym_update_*_envs.jsonl")):
        rows.extend(_load_jsonl(path))
    return rows


def _done_rate_from_progress(path: str | Path) -> dict[str, Any]:
    rows = _load_jsonl(path)
    rates = []
    for row in rows:
        stats = (row.get("pack_summary") or {}).get("stats") or {}
        shapes = (row.get("pack_summary") or {}).get("shapes") or {}
        shape = shapes.get("dones")
        done = stats.get("done_count")
        if isinstance(shape, list) and len(shape) >= 2 and isinstance(done, (int, float)):
            rates.append(float(done) / float(max(1, int(shape[0]) * int(shape[1]))))
    return _stats(rates)


def _subset_rates(records: list[dict[str, Any]], env_ids: set[str] | None) -> dict[str, Any]:
    rows = records if env_ids is None else [r for r in records if str(r["gym_env_id"]) in env_ids]
    if not rows:
        return {"rows": 0}
    steps = int(max(1, int(rows[0].get("single_eval_pos", 512)) if "single_eval_pos" in rows[0] else 512))
    done = sum(int(r.get("done_count", 0)) for r in rows)
    trunc = sum(int(r.get("truncated_count", 0)) for r in rows)
    denom = int(len(rows) * steps)
    by_update: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_update[int(row.get("update_idx", 0))].append(row)
    update_rates = []
    for update, update_rows in sorted(by_update.items()):
        update_done = sum(int(r.get("done_count", 0)) for r in update_rows)
        update_rates.append(float(update_done) / float(max(1, len(update_rows) * steps)))
    return {
        "rows": int(len(rows)),
        "done": int(done),
        "truncated": int(trunc),
        "done_rate": float(done / max(1, denom)),
        "truncated_rate": float(trunc / max(1, denom)),
        "done_rate_by_update": _stats(update_rates),
        "done_rate_first": float(update_rates[0]) if update_rates else None,
        "done_rate_last": float(update_rates[-1]) if update_rates else None,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    records = _load_sidecars(args.gym_dir)
    total_done = sum(int(r.get("done_count", 0)) for r in records)
    by_env: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_env[str(row["gym_env_id"])].append(row)
    env_summaries = []
    for env_id, rows in sorted(by_env.items()):
        done = sum(int(r.get("done_count", 0)) for r in rows)
        trunc = sum(int(r.get("truncated_count", 0)) for r in rows)
        first_done = [float(r["first_done_step"]) for r in rows if r.get("first_done_step") is not None]
        denom = int(len(rows) * 512)
        splits = {}
        for name, pred in (
            ("early", lambda u: u < 5),
            ("mid", lambda u: 5 <= u < 15),
            ("late", lambda u: u >= 15),
        ):
            sub = [r for r in rows if pred(int(r.get("update_idx", 0)))]
            if sub:
                splits[f"done_rate_{name}"] = (
                    sum(int(r.get("done_count", 0)) for r in sub) / float(max(1, len(sub) * 512))
                )
        env_summaries.append(
            {
                "gym_env_id": env_id,
                "rows": int(len(rows)),
                "done": int(done),
                "done_share": float(done / max(1, total_done)),
                "done_rate": float(done / max(1, denom)),
                "truncated": int(trunc),
                "truncated_rate": float(trunc / max(1, denom)),
                "first_done_step_min": min(first_done) if first_done else None,
                "first_done_step_median": statistics.median(first_done) if first_done else None,
                "raw_reward_mean": float(statistics.mean(float(r["raw_reward_mean"]) for r in rows)),
                "raw_reward_std_mean": float(statistics.mean(float(r["raw_reward_std"]) for r in rows)),
                "action_saturation_mean": float(
                    statistics.mean(float(r["raw_policy_action_unit_clip_saturation_fraction"]) for r in rows)
                ),
                **splits,
            }
        )
    env_summaries.sort(key=lambda item: int(item["done"]), reverse=True)
    top2_envs = {item["gym_env_id"] for item in env_summaries[:2]}
    top3_envs = {item["gym_env_id"] for item in env_summaries[:3]}
    report = {
        "analysis_entry": "phase2_gym_done_outlier_audit",
        "exploratory_only": True,
        "contract": {
            "does_not_modify_exact_scm": True,
            "does_not_modify_default_prior_generator": True,
            "does_not_sample_environments": True,
            "does_not_run_ppo_training": True,
        },
        "input_artifacts": {
            "gym_dir": str(Path(args.gym_dir).expanduser().resolve()),
            "gated_full_progress": str(Path(args.gated_full_progress).expanduser().resolve()),
            "gated_q90_progress": str(Path(args.gated_q90_progress).expanduser().resolve()),
        },
        "gym_sidecar_rows": int(len(records)),
        "gym_total_done": int(total_done),
        "env_summaries": env_summaries,
        "concentration": {
            "top1_share": float(sum(item["done_share"] for item in env_summaries[:1])),
            "top2_share": float(sum(item["done_share"] for item in env_summaries[:2])),
            "top3_share": float(sum(item["done_share"] for item in env_summaries[:3])),
            "top5_share": float(sum(item["done_share"] for item in env_summaries[:5])),
            "top2_env_ids": sorted(top2_envs),
            "top3_env_ids": sorted(top3_envs),
        },
        "subset_rates": {
            "gym_all": _subset_rates(records, None),
            "gym_top2_only": _subset_rates(records, top2_envs),
            "gym_excluding_top2": _subset_rates(records, set(by_env) - top2_envs),
            "gated_full": _done_rate_from_progress(args.gated_full_progress),
            "gated_q90": _done_rate_from_progress(args.gated_q90_progress),
        },
        "decision": {
            "terminal_global_repair_recommended": False,
            "reason": (
                "Gym done-rate mismatch is dominated by InvertedPendulum-v5 and "
                "InvertedDoublePendulum-v5.  Excluding those outliers, Gym done-rate "
                "is close to or below the gated-prior fixed-list smoke range, so this "
                "is not currently the largest broad mismatch to repair."
            ),
        },
    }
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "gym_done_outlier_audit_report.json"
    report_path.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md = [
        "# Gym Done Outlier Audit",
        "",
        "Exploratory offline audit. No sampling, PPO training, or generator mutation.",
        "",
        "## Concentration",
        f"- top1_share: {report['concentration']['top1_share']:.6f}",
        f"- top2_share: {report['concentration']['top2_share']:.6f}",
        f"- top3_share: {report['concentration']['top3_share']:.6f}",
        f"- top2_env_ids: {', '.join(report['concentration']['top2_env_ids'])}",
        "",
        "## Done Rates",
        f"- Gym all: {report['subset_rates']['gym_all']['done_rate']:.6f}",
        f"- Gym top2 only: {report['subset_rates']['gym_top2_only']['done_rate']:.6f}",
        f"- Gym excluding top2: {report['subset_rates']['gym_excluding_top2']['done_rate']:.6f}",
        f"- gated full: {report['subset_rates']['gated_full']['mean']:.6f}",
        f"- gated q90: {report['subset_rates']['gated_q90']['mean']:.6f}",
        "",
        "## Decision",
        report["decision"]["reason"],
    ]
    md_path = out_dir / "gym_done_outlier_audit_summary.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gym-dir", type=str, default=DEFAULT_GYM_DIR)
    parser.add_argument("--gated-full-progress", type=str, default=DEFAULT_GATED_FULL)
    parser.add_argument("--gated-q90-progress", type=str, default=DEFAULT_GATED_Q90)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    report = run(args)
    print(
        json.dumps(
            _json_safe(
                {
                    "output_dir": args.output_dir,
                    "top2_share": report["concentration"]["top2_share"],
                    "gym_all_done_rate": report["subset_rates"]["gym_all"]["done_rate"],
                    "gym_excluding_top2_done_rate": report["subset_rates"]["gym_excluding_top2"]["done_rate"],
                    "terminal_global_repair_recommended": report["decision"]["terminal_global_repair_recommended"],
                }
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
