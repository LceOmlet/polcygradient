#!/usr/bin/env python
"""Factor audit for Pendulum-like fixed-list per-env PPO deltas.

This is an exploratory, read-only analysis. It consumes the trusted pack-runner
sidecar delta summary plus the frozen fixed-list CSV, then reports which
signature/action/done/source factors are associated with short-run PPO
improvement or regression.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_DELTA_REPORT = (
    "/home/chen/RLPFN/artifacts/phase2_pendulum_like_strict_n64_sidecar_summary_0509/"
    "pendulum_like_sidecar_delta_summary_report.json"
)
DEFAULT_OUTPUT_DIR = "/home/chen/RLPFN/artifacts/phase2_pendulum_like_strict_delta_factor_audit_0509"


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


def _f(value: Any) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _stats(values: list[Any]) -> dict[str, Any]:
    arr = np.asarray([float(v) for v in values if _f(v) is not None], dtype=np.float64)
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "q10": float(np.quantile(arr, 0.10)),
        "q25": float(np.quantile(arr, 0.25)),
        "q50": float(np.quantile(arr, 0.50)),
        "q75": float(np.quantile(arr, 0.75)),
        "q90": float(np.quantile(arr, 0.90)),
        "max": float(np.max(arr)),
        "positive_fraction": float(np.mean(arr > 0.0)),
    }


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1)
        i = j
    return ranks


def _corr(rows: list[dict[str, Any]], x_key: str, y_key: str) -> dict[str, Any]:
    pairs = [(float(x), float(y)) for r in rows if (x := _f(r.get(x_key))) is not None and (y := _f(r.get(y_key))) is not None]
    if len(pairs) < 3:
        return {"n": len(pairs)}
    xs = np.asarray([p[0] for p in pairs], dtype=np.float64)
    ys = np.asarray([p[1] for p in pairs], dtype=np.float64)
    if float(np.std(xs)) == 0.0 or float(np.std(ys)) == 0.0:
        return {"n": len(pairs), "pearson": None, "spearman": None}
    pearson = float(np.corrcoef(xs, ys)[0, 1])
    spearman = float(np.corrcoef(_rankdata(xs), _rankdata(ys))[0, 1])
    return {"n": len(pairs), "pearson": pearson, "spearman": spearman}


def _load_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _parse_mode(mode: Any) -> dict[str, Any]:
    text = str(mode or "")
    m = re.search(r"(?:^|_)obs_linear(?:_neg)?_(\d+)x(\d+)", text)
    return {
        "mode_exact": text or None,
        "mode_is_early_then_zero": bool(text.startswith("early_") and text.endswith("_then_zero")),
        "mode_has_neg": "_neg_" in text,
        "mode_obs_feature": int(m.group(1)) if m else None,
        "mode_horizon": int(m.group(2)) if m else None,
    }


def _bucket_quantiles(rows: list[dict[str, Any]], key: str, buckets: int = 4) -> None:
    vals = np.asarray([_f(r.get(key)) for r in rows if _f(r.get(key)) is not None], dtype=np.float64)
    if vals.size == 0:
        return
    cuts = [float(np.quantile(vals, i / buckets)) for i in range(1, buckets)]
    for row in rows:
        x = _f(row.get(key))
        if x is None:
            row[f"{key}_qbucket"] = "missing"
            continue
        bucket = 0
        while bucket < len(cuts) and x > cuts[bucket]:
            bucket += 1
        row[f"{key}_qbucket"] = f"q{bucket + 1}"


def _group_summary(rows: list[dict[str, Any]], key: str, min_n: int = 1) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key))].append(row)
    out = []
    for name, group_rows in sorted(grouped.items()):
        if len(group_rows) < min_n:
            continue
        out.append(
            {
                "group": name,
                "n": len(group_rows),
                "raw_reward_sum_delta": _stats([r.get("raw_reward_sum_delta") for r in group_rows]),
                "reward_env_sum_delta": _stats([r.get("reward_env_sum_delta") for r in group_rows]),
                "training_reward_sum_delta": _stats([r.get("training_reward_sum_delta") for r in group_rows]),
                "done_count_delta": _stats([r.get("done_count_delta") for r in group_rows]),
                "first_done_step_delta": _stats([r.get("first_done_step_delta") for r in group_rows]),
                "action_value_absmean_delta": _stats([r.get("action_value_absmean_delta") for r in group_rows]),
            }
        )
    return sorted(out, key=lambda x: (x["raw_reward_sum_delta"].get("mean") is None, x["raw_reward_sum_delta"].get("mean") or 0.0))


def _subset_stats(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    stats = _stats([r.get("raw_reward_sum_delta") for r in rows])
    if stats.get("n", 0) == 0:
        return None
    return {
        "n": stats["n"],
        "raw_mean": stats["mean"],
        "raw_positive_fraction": stats["positive_fraction"],
        "raw_q25": stats["q25"],
        "raw_q50": stats["q50"],
        "raw_q75": stats["q75"],
    }


def _rule_scan(
    rows: list[dict[str, Any]],
    numeric_keys: list[str],
    categorical_keys: list[str],
    min_n: int = 8,
) -> dict[str, Any]:
    conds = []
    for key in numeric_keys:
        values = np.asarray([_f(r.get(key)) for r in rows if _f(r.get(key)) is not None], dtype=np.float64)
        if values.size < min_n:
            continue
        for q in (0.25, 0.50, 0.75):
            threshold = float(np.quantile(values, q))

            def le(row: dict[str, Any], k: str = key, th: float = threshold) -> bool:
                x = _f(row.get(k))
                return x is not None and x <= th

            def gt(row: dict[str, Any], k: str = key, th: float = threshold) -> bool:
                x = _f(row.get(k))
                return x is not None and x > th

            conds.append((f"{key} <= {threshold:.6g}", le))
            conds.append((f"{key} > {threshold:.6g}", gt))
    for key in categorical_keys:
        for value in sorted({str(r.get(key)) for r in rows}):

            def eq(row: dict[str, Any], k: str = key, v: str = value) -> bool:
                return str(row.get(k)) == v

            conds.append((f"{key} == {value}", eq))

    single_rules = []
    for name, fn in conds:
        subset = [r for r in rows if fn(r)]
        stats = _subset_stats(subset)
        if stats and stats["n"] >= min_n:
            single_rules.append({"rule": name, **stats})

    pair_rules = []
    for i, (name_a, fn_a) in enumerate(conds):
        for name_b, fn_b in conds[i + 1 :]:
            subset = [r for r in rows if fn_a(r) and fn_b(r)]
            stats = _subset_stats(subset)
            if stats and stats["n"] >= min_n:
                pair_rules.append({"rule": f"{name_a} AND {name_b}", **stats})

    return {
        "best_single_rules": sorted(single_rules, key=lambda x: x["raw_mean"], reverse=True)[:20],
        "worst_single_rules": sorted(single_rules, key=lambda x: x["raw_mean"])[:20],
        "best_pair_rules": sorted(pair_rules, key=lambda x: x["raw_mean"], reverse=True)[:20],
        "worst_pair_rules": sorted(pair_rules, key=lambda x: x["raw_mean"])[:20],
    }


def _load_rows(delta_report_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    report = json.loads(delta_report_path.read_text(encoding="utf-8"))
    csv_path = Path(report["fixed_list_csv"]).expanduser().resolve()
    rule = str(report.get("fixed_list_rule") or "")
    fixed_rows = [r for r in _load_csv(csv_path) if not rule or str(r.get("rule")) == rule]
    fixed_by_env_idx = {idx: row for idx, row in enumerate(fixed_rows)}

    rows = []
    for row in report["per_env"]:
        r = dict(row)
        fixed = fixed_by_env_idx.get(int(r["env_idx"]), {})
        for key in (
            "best_feedback_action_abs_prefix",
            "best_feedback_action_abs_suffix",
            "pendulum_like_settle_strict",
            "source_label",
        ):
            if key in fixed:
                r[f"fixed_{key}"] = _f(fixed[key]) if key.startswith("best_feedback_action_abs") else fixed[key]
        source = str(r.get("source_tag") or "")
        parts = source.split("_")
        r["source_seed"] = parts[0] if parts else source
        r["source_range"] = parts[1] if len(parts) > 1 else "unknown"
        r["raw_improved"] = bool((_f(r.get("raw_reward_sum_delta")) or 0.0) > 0.0)
        r["raw_regressed"] = bool((_f(r.get("raw_reward_sum_delta")) or 0.0) < 0.0)
        r.update(_parse_mode(r.get("best_feedback_mode")))
        base = _parse_mode(r.get("feedback_pair_best_base_mode"))
        r["base_mode_obs_feature"] = base["mode_obs_feature"]
        r["base_mode_horizon"] = base["mode_horizon"]
        r["base_mode_has_neg"] = base["mode_has_neg"]
        rows.append(r)
    return report, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delta-report", default=DEFAULT_DELTA_REPORT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    delta_report_path = Path(args.delta_report).expanduser().resolve()
    base_report, rows = _load_rows(delta_report_path)

    for key in (
        "signature_score",
        "signature_best_feedback_minus_open_env",
        "signature_feedback_pair_phase_gap_env",
        "signature_feedback_suffix_gain_over_zero_env",
        "fixed_best_feedback_action_abs_prefix",
        "fixed_best_feedback_action_abs_suffix",
    ):
        _bucket_quantiles(rows, key, buckets=4)

    numeric_predictors = [
        "signature_score",
        "signature_best_feedback_minus_open_env",
        "signature_feedback_pair_phase_gap_env",
        "signature_feedback_suffix_gain_over_zero_env",
        "fixed_best_feedback_action_abs_prefix",
        "fixed_best_feedback_action_abs_suffix",
        "raw_reward_sum_first",
        "reward_env_sum_first",
        "reward_ctrl_sum_first",
        "done_count_first",
        "first_done_step_first",
        "action_value_absmean_first",
        "action_value_absmean_delta",
        "action_norm_mean_delta",
        "action_saturation_delta",
        "action_temporal_delta_norm_mean_delta",
        "done_count_delta",
        "first_done_step_delta",
        "obs_value_std_delta",
        "next_state_step_l2_delta_mean_delta",
    ]
    categorical_predictors = [
        "source_range",
        "source_seed",
        "feedback_pair_best_sign",
        "mode_obs_feature",
        "mode_horizon",
        "mode_has_neg",
        "mode_is_early_then_zero",
        "base_mode_obs_feature",
    ]
    correlations = {
        key: _corr(rows, key, "raw_reward_sum_delta")
        for key in numeric_predictors
    }
    top_correlations = sorted(
        (
            {"field": k, **v}
            for k, v in correlations.items()
            if v.get("spearman") is not None
        ),
        key=lambda x: abs(float(x.get("spearman") or 0.0)),
        reverse=True,
    )

    group_keys = [
        "source_tag",
        "source_range",
        "source_seed",
        "feedback_pair_best_sign",
        "mode_exact",
        "mode_obs_feature",
        "mode_horizon",
        "mode_has_neg",
        "mode_is_early_then_zero",
        "base_mode_obs_feature",
        "base_mode_horizon",
        "signature_score_qbucket",
        "signature_best_feedback_minus_open_env_qbucket",
        "signature_feedback_pair_phase_gap_env_qbucket",
        "signature_feedback_suffix_gain_over_zero_env_qbucket",
        "fixed_best_feedback_action_abs_prefix_qbucket",
        "fixed_best_feedback_action_abs_suffix_qbucket",
    ]
    groups = {key: _group_summary(rows, key, min_n=2 if key in {"mode_exact"} else 1) for key in group_keys}
    rule_scan = _rule_scan(rows, numeric_predictors, categorical_predictors, min_n=8)

    improved = sorted(rows, key=lambda r: float(r.get("raw_reward_sum_delta") or 0.0), reverse=True)[:16]
    regressed = sorted(rows, key=lambda r: float(r.get("raw_reward_sum_delta") or 0.0))[:16]
    contrast_fields = [
        "signature_score",
        "signature_best_feedback_minus_open_env",
        "signature_feedback_pair_phase_gap_env",
        "signature_feedback_suffix_gain_over_zero_env",
        "fixed_best_feedback_action_abs_prefix",
        "fixed_best_feedback_action_abs_suffix",
        "raw_reward_sum_first",
        "done_count_first",
        "first_done_step_first",
        "action_value_absmean_first",
        "action_value_absmean_delta",
        "done_count_delta",
        "first_done_step_delta",
    ]
    top16_contrast = []
    for key in contrast_fields:
        imp = _stats([r.get(key) for r in improved])
        reg = _stats([r.get(key) for r in regressed])
        if imp.get("n", 0) and reg.get("n", 0):
            top16_contrast.append(
                {
                    "field": key,
                    "improved_mean": imp["mean"],
                    "regressed_mean": reg["mean"],
                    "improved_minus_regressed": float(imp["mean"] - reg["mean"]),
                }
            )

    report = {
        "analysis_entry": "phase2_pendulum_like_delta_factor_audit",
        "contract": {
            "exploratory_only": True,
            "reads_existing_delta_report": True,
            "reads_existing_fixed_list_csv": True,
            "does_not_modify_exact_scm": True,
            "does_not_modify_fit_model": True,
            "does_not_modify_ppo": True,
            "does_not_rescore_or_resample_envs": True,
        },
        "delta_report": str(delta_report_path),
        "run_dir": base_report.get("run_dir"),
        "fixed_list_csv": base_report.get("fixed_list_csv"),
        "fixed_list_rule": base_report.get("fixed_list_rule"),
        "n_envs": len(rows),
        "overall": base_report.get("overall"),
        "correlations_with_raw_reward_delta": correlations,
        "top_correlations_by_abs_spearman": top_correlations[:12],
        "groups": groups,
        "rule_scan": rule_scan,
        "top16_improved_vs_regressed_contrast": sorted(
            top16_contrast,
            key=lambda x: abs(float(x["improved_minus_regressed"])),
            reverse=True,
        ),
        "top_raw_improved": improved[:10],
        "top_raw_regressed": regressed[:10],
        "per_env": rows,
    }

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "pendulum_like_delta_factor_audit_report.json"
    json_path.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Pendulum-Like Strict Delta Factor Audit",
        "",
        "Exploratory read-only audit over existing fixed-list and sidecar-delta artifacts.",
        "",
        f"delta_report: {delta_report_path}",
        f"n_envs: {len(rows)}",
        "",
        "## Strongest Numeric Associations",
        "",
        "| field | n | pearson | spearman |",
        "|---|---:|---:|---:|",
    ]
    for item in top_correlations[:12]:
        lines.append(f"| {item['field']} | {item.get('n')} | {item.get('pearson')} | {item.get('spearman')} |")

    def add_group_table(title: str, key: str, max_rows: int = 16) -> None:
        lines.extend(["", f"## {title}", "", "| group | n | raw mean | raw pos | env mean | train mean | done mean | first-done mean | action-abs mean |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
        for item in groups[key][:max_rows]:
            lines.append(
                f"| {item['group']} | {item['n']} | "
                f"{item['raw_reward_sum_delta'].get('mean')} | "
                f"{item['raw_reward_sum_delta'].get('positive_fraction')} | "
                f"{item['reward_env_sum_delta'].get('mean')} | "
                f"{item['training_reward_sum_delta'].get('mean')} | "
                f"{item['done_count_delta'].get('mean')} | "
                f"{item['first_done_step_delta'].get('mean')} | "
                f"{item['action_value_absmean_delta'].get('mean')} |"
            )

    add_group_table("Source Range", "source_range")
    add_group_table("Feedback Sign", "feedback_pair_best_sign")
    add_group_table("Mode Feature", "mode_obs_feature")
    add_group_table("Mode Horizon", "mode_horizon")
    add_group_table("Mode Exact", "mode_exact", max_rows=24)
    add_group_table("Signature Score Quartile", "signature_score_qbucket")
    add_group_table("Suffix Gain Quartile", "signature_feedback_suffix_gain_over_zero_env_qbucket")
    add_group_table("Prefix Action Abs Quartile", "fixed_best_feedback_action_abs_prefix_qbucket")
    add_group_table("Suffix Action Abs Quartile", "fixed_best_feedback_action_abs_suffix_qbucket")

    lines.extend(["", "## Top-16 Improved vs Regressed Contrast", "", "| field | improved mean | regressed mean | difference |", "|---|---:|---:|---:|"])
    for item in report["top16_improved_vs_regressed_contrast"]:
        lines.append(
            f"| {item['field']} | {item['improved_mean']} | {item['regressed_mean']} | {item['improved_minus_regressed']} |"
        )

    def add_rule_table(title: str, items: list[dict[str, Any]]) -> None:
        lines.extend(["", f"## {title}", "", "| rule | n | raw mean | raw pos | q25 | q50 | q75 |", "|---|---:|---:|---:|---:|---:|---:|"])
        for item in items[:12]:
            lines.append(
                f"| {item['rule']} | {item['n']} | {item['raw_mean']} | "
                f"{item['raw_positive_fraction']} | {item['raw_q25']} | {item['raw_q50']} | {item['raw_q75']} |"
            )

    add_rule_table("Best Single Rules", rule_scan["best_single_rules"])
    add_rule_table("Worst Single Rules", rule_scan["worst_single_rules"])
    add_rule_table("Best Pair Rules", rule_scan["best_pair_rules"])
    add_rule_table("Worst Pair Rules", rule_scan["worst_pair_rules"])

    md_path = out_dir / "pendulum_like_delta_factor_audit.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
