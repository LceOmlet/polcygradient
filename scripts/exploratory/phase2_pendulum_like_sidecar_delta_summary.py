#!/usr/bin/env python
"""Summarize per-env sidecar deltas for Pendulum-like fixed-list smoke runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_RUN_DIR = "/home/chen/RLPFN/artifacts/phase2_pendulum_like_strict_n64_sidecar_smoke_s512_u8_e4_advnorm_0509"
DEFAULT_OUTPUT_DIR = "/home/chen/RLPFN/artifacts/phase2_pendulum_like_strict_n64_sidecar_summary_0509"


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


def _f(x: Any) -> float | None:
    try:
        y = float(x)
    except (TypeError, ValueError):
        return None
    return y if math.isfinite(y) else None


def _delta(a: Any, b: Any) -> float | None:
    x = _f(a)
    y = _f(b)
    if x is None or y is None:
        return None
    return float(y - x)


def _stats(values: list[Any]) -> dict[str, Any]:
    arr = np.asarray([float(v) for v in values if _f(v) is not None], dtype=np.float64)
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "min": float(np.min(arr)),
        "q10": float(np.quantile(arr, 0.10)),
        "q50": float(np.quantile(arr, 0.50)),
        "q90": float(np.quantile(arr, 0.90)),
        "max": float(np.max(arr)),
        "positive_fraction": float(np.mean(arr > 0.0)),
    }


def _sidecar_path(run_dir: Path, update_idx: int) -> Path:
    return run_dir / "semantic_sidecars" / f"prior_update_{int(update_idx):06d}_envs.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--first-update", type=int, default=0)
    parser.add_argument("--last-update", type=int, default=7)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    summary = _load_json(run_dir / "summary.json")
    contract = summary.get("contract") or {}
    csv_path = Path(contract["fixed_frozen_h_list_csv"]).expanduser().resolve()
    rule = str(contract.get("fixed_frozen_h_list_rule") or "")
    fixed_rows = [row for row in _load_csv(csv_path) if not rule or str(row.get("rule")) == rule]
    first_rows = _load_jsonl(_sidecar_path(run_dir, int(args.first_update)))
    last_rows = _load_jsonl(_sidecar_path(run_dir, int(args.last_update)))
    first_by_idx = {int(row["env_idx"]): row for row in first_rows}
    last_by_idx = {int(row["env_idx"]): row for row in last_rows}
    if len(first_by_idx) != len(last_by_idx):
        raise RuntimeError("first/last sidecar env count mismatch")

    per_env = []
    for env_idx in sorted(first_by_idx):
        a = first_by_idx[env_idx]
        b = last_by_idx[env_idx]
        fixed = fixed_rows[env_idx] if env_idx < len(fixed_rows) else {}
        row = {
            "env_idx": int(env_idx),
            "source_tag": fixed.get("source_tag"),
            "source_env_index": fixed.get("env_index"),
            "env_group_env_seed": a.get("env_group_env_seed"),
            "obs_dim": a.get("obs_dim"),
            "action_dim": a.get("action_dim"),
            "state_dim": a.get("state_dim"),
            "signature_score": _f(fixed.get("signature_score")),
            "best_feedback_mode": fixed.get("best_feedback_mode"),
            "parsed_profile": fixed.get("parsed_profile"),
            "parsed_gain": fixed.get("parsed_gain"),
            "parsed_sign": fixed.get("parsed_sign"),
            "parsed_feature": fixed.get("parsed_feature"),
            "feedback_pair_best_base_mode": fixed.get("feedback_pair_best_base_mode"),
            "feedback_pair_best_sign": fixed.get("feedback_pair_best_sign"),
            "signature_best_feedback_minus_open_env": _f(fixed.get("best_feedback_minus_open_env")),
            "signature_feedback_pair_phase_gap_env": _f(fixed.get("feedback_pair_phase_gap_env")),
            "signature_feedback_suffix_gain_over_zero_env": _f(fixed.get("feedback_suffix_gain_over_zero_env")),
            "raw_reward_sum_first": a.get("raw_reward_sum"),
            "raw_reward_sum_last": b.get("raw_reward_sum"),
            "raw_reward_sum_delta": _delta(a.get("raw_reward_sum"), b.get("raw_reward_sum")),
            "reward_env_sum_first": a.get("reward_env_sum"),
            "reward_env_sum_last": b.get("reward_env_sum"),
            "reward_env_sum_delta": _delta(a.get("reward_env_sum"), b.get("reward_env_sum")),
            "reward_ctrl_sum_first": a.get("reward_ctrl_sum"),
            "reward_ctrl_sum_last": b.get("reward_ctrl_sum"),
            "reward_ctrl_sum_delta": _delta(a.get("reward_ctrl_sum"), b.get("reward_ctrl_sum")),
            "training_reward_sum_delta": _delta(a.get("training_reward_sum"), b.get("training_reward_sum")),
            "done_count_first": a.get("done_count"),
            "done_count_last": b.get("done_count"),
            "done_count_delta": _delta(a.get("done_count"), b.get("done_count")),
            "first_done_step_first": a.get("first_done_step"),
            "first_done_step_last": b.get("first_done_step"),
            "first_done_step_delta": _delta(a.get("first_done_step"), b.get("first_done_step")),
            "action_value_absmean_first": a.get("action_value_absmean"),
            "action_value_absmean_last": b.get("action_value_absmean"),
            "action_value_absmean_delta": _delta(a.get("action_value_absmean"), b.get("action_value_absmean")),
            "action_norm_mean_delta": _delta(a.get("action_norm_mean"), b.get("action_norm_mean")),
            "action_saturation_delta": _delta(
                a.get("raw_policy_action_unit_clip_saturation_fraction"),
                b.get("raw_policy_action_unit_clip_saturation_fraction"),
            ),
            "action_temporal_delta_norm_mean_delta": _delta(
                a.get("action_temporal_delta_norm_mean"),
                b.get("action_temporal_delta_norm_mean"),
            ),
            "obs_value_std_delta": _delta(a.get("obs_value_std"), b.get("obs_value_std")),
            "next_state_step_l2_delta_mean_delta": _delta(
                a.get("next_state_step_l2_delta_mean"),
                b.get("next_state_step_l2_delta_mean"),
            ),
        }
        per_env.append(row)

    by_source = {}
    for row in per_env:
        by_source.setdefault(str(row.get("source_tag")), []).append(row)
    source_summary = {
        source: {
            "n": len(rows),
            "raw_reward_sum_delta": _stats([r.get("raw_reward_sum_delta") for r in rows]),
            "reward_env_sum_delta": _stats([r.get("reward_env_sum_delta") for r in rows]),
            "done_count_delta": _stats([r.get("done_count_delta") for r in rows]),
            "action_value_absmean_delta": _stats([r.get("action_value_absmean_delta") for r in rows]),
        }
        for source, rows in sorted(by_source.items())
    }

    def _bucket_summary(key: str) -> dict[str, Any]:
        buckets: dict[str, list[dict[str, Any]]] = {}
        for row in per_env:
            buckets.setdefault(str(row.get(key)), []).append(row)
        return {
            name: {
                "n": len(rows),
                "raw_reward_sum_delta": _stats([r.get("raw_reward_sum_delta") for r in rows]),
                "reward_env_sum_delta": _stats([r.get("reward_env_sum_delta") for r in rows]),
                "training_reward_sum_delta": _stats([r.get("training_reward_sum_delta") for r in rows]),
                "done_count_delta": _stats([r.get("done_count_delta") for r in rows]),
                "action_value_absmean_delta": _stats([r.get("action_value_absmean_delta") for r in rows]),
                "action_saturation_delta": _stats([r.get("action_saturation_delta") for r in rows]),
            }
            for name, rows in sorted(buckets.items())
        }

    report = {
        "analysis_entry": "phase2_pendulum_like_sidecar_delta_summary",
        "contract": {
            "exploratory_only": True,
            "reads_trusted_pack_runner_sidecars": True,
            "does_not_modify_exact_scm": True,
            "does_not_modify_fit_model": True,
            "does_not_modify_ppo": True,
        },
        "run_dir": str(run_dir),
        "fixed_list_csv": str(csv_path),
        "fixed_list_rule": rule,
        "first_update": int(args.first_update),
        "last_update": int(args.last_update),
        "n_envs": len(per_env),
        "overall": {
            "raw_reward_sum_delta": _stats([r.get("raw_reward_sum_delta") for r in per_env]),
            "reward_env_sum_delta": _stats([r.get("reward_env_sum_delta") for r in per_env]),
            "training_reward_sum_delta": _stats([r.get("training_reward_sum_delta") for r in per_env]),
            "done_count_delta": _stats([r.get("done_count_delta") for r in per_env]),
            "first_done_step_delta": _stats([r.get("first_done_step_delta") for r in per_env]),
            "action_value_absmean_delta": _stats([r.get("action_value_absmean_delta") for r in per_env]),
            "action_saturation_delta": _stats([r.get("action_saturation_delta") for r in per_env]),
        },
        "source_summary": source_summary,
        "bucket_summary": {
            "parsed_profile": _bucket_summary("parsed_profile"),
            "parsed_gain": _bucket_summary("parsed_gain"),
            "parsed_sign": _bucket_summary("parsed_sign"),
            "parsed_feature": _bucket_summary("parsed_feature"),
        },
        "top_raw_improved": sorted(per_env, key=lambda r: float(r.get("raw_reward_sum_delta") or 0.0), reverse=True)[:10],
        "top_raw_regressed": sorted(per_env, key=lambda r: float(r.get("raw_reward_sum_delta") or 0.0))[:10],
        "per_env": per_env,
    }
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "pendulum_like_sidecar_delta_summary_report.json"
    report_path.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Pendulum-Like Sidecar Delta Summary",
        "",
        f"run_dir: {run_dir}",
        f"n_envs: {len(per_env)}",
        "",
        "## Overall",
        "",
        "| metric | mean | positive fraction | q10 | q50 | q90 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, stats in report["overall"].items():
        lines.append(
            f"| {key} | {stats.get('mean')} | {stats.get('positive_fraction')} | "
            f"{stats.get('q10')} | {stats.get('q50')} | {stats.get('q90')} |"
        )
    lines.extend(["", "## Source Summary", "", "| source | n | raw mean | raw positive | env mean | done mean |", "|---|---:|---:|---:|---:|---:|"])
    for source, stats in source_summary.items():
        lines.append(
            f"| {source} | {stats['n']} | {stats['raw_reward_sum_delta'].get('mean')} | "
            f"{stats['raw_reward_sum_delta'].get('positive_fraction')} | "
            f"{stats['reward_env_sum_delta'].get('mean')} | {stats['done_count_delta'].get('mean')} |"
        )

    for bucket_name, bucket in report["bucket_summary"].items():
        lines.extend(
            [
                "",
                f"## Bucket Summary: {bucket_name}",
                "",
                "| bucket | n | raw mean | raw positive | env mean | train mean | done mean | action abs mean |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for name, stats in bucket.items():
            lines.append(
                f"| {name} | {stats['n']} | {stats['raw_reward_sum_delta'].get('mean')} | "
                f"{stats['raw_reward_sum_delta'].get('positive_fraction')} | "
                f"{stats['reward_env_sum_delta'].get('mean')} | "
                f"{stats['training_reward_sum_delta'].get('mean')} | "
                f"{stats['done_count_delta'].get('mean')} | "
                f"{stats['action_value_absmean_delta'].get('mean')} |"
            )

    lines.extend(["", "## Most Regressed Raw Reward", "", "| env | source | raw delta | env delta | done delta | action abs delta | mode | sign |", "|---:|---|---:|---:|---:|---:|---|---|"])
    for row in report["top_raw_regressed"][:8]:
        lines.append(
            f"| {row['env_idx']} | {row.get('source_tag')} | {row.get('raw_reward_sum_delta')} | "
            f"{row.get('reward_env_sum_delta')} | {row.get('done_count_delta')} | "
            f"{row.get('action_value_absmean_delta')} | {row.get('best_feedback_mode')} | "
            f"{row.get('feedback_pair_best_sign')} |"
        )
    lines.extend(["", "## Most Improved Raw Reward", "", "| env | source | raw delta | env delta | done delta | action abs delta | mode | sign |", "|---:|---|---:|---:|---:|---:|---|---|"])
    for row in report["top_raw_improved"][:8]:
        lines.append(
            f"| {row['env_idx']} | {row.get('source_tag')} | {row.get('raw_reward_sum_delta')} | "
            f"{row.get('reward_env_sum_delta')} | {row.get('done_count_delta')} | "
            f"{row.get('action_value_absmean_delta')} | {row.get('best_feedback_mode')} | "
            f"{row.get('feedback_pair_best_sign')} |"
        )
    md_path = out_dir / "pendulum_like_sidecar_delta_summary.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(md_path), "report": str(report_path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
