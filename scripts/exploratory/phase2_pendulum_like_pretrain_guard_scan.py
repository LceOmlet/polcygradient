#!/usr/bin/env python
"""Scan pretraining-observable guards for Pendulum-like strict candidates.

This is an exploratory, read-only analysis.  It joins:
  - a trusted Pendulum-like fixed-list CSV,
  - a trusted pack-runner per-env post/pre delta report,
  - only fixed-list metadata and update-0/pre-update sidecar fields.

The scan intentionally separates:
  - allowed guard features: measurable before PPO training,
  - labels: trusted post-pre raw/env reward deltas.

It does not mutate exact SCM, fit_model defaults, PPO, or prior milestones.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Callable

import numpy as np


DEFAULT_DELTA_REPORT = (
    "/home/chen/RLPFN/artifacts/"
    "phase2_pendulum_like_expanded_sat_only_n128_sidecar_summary_seed9100_0510/"
    "pendulum_like_sidecar_delta_summary_report.json"
)
DEFAULT_OUTPUT_DIR = (
    "/home/chen/RLPFN/artifacts/"
    "phase2_pendulum_like_expanded_pretrain_guard_scan_seed9100_0510"
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


def _f(value: Any) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _load_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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


def _parse_mode(mode: Any) -> dict[str, Any]:
    text = str(mode or "")
    match = re.search(r"(?:^|_)obs_linear(?:_neg)?_(\d+)x(\d+)", text)
    return {
        "mode_exact": text or None,
        "mode_has_neg": "_neg_" in text,
        "mode_obs_feature": int(match.group(1)) if match else None,
        "mode_horizon": int(match.group(2)) if match else None,
        "mode_is_decay": text.startswith("decay_"),
        "mode_is_early": text.startswith("early_"),
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
    return {
        "n": len(pairs),
        "pearson": float(np.corrcoef(xs, ys)[0, 1]),
        "spearman": float(np.corrcoef(_rankdata(xs), _rankdata(ys))[0, 1]),
    }


def _safe_pair(row: dict[str, Any], a: str, b: str, op: str) -> float | None:
    av = _f(row.get(a))
    bv = _f(row.get(b))
    if av is None or bv is None:
        return None
    if op == "mean":
        return 0.5 * (av + bv)
    if op == "max":
        return max(av, bv)
    if op == "min":
        return min(av, bv)
    if op == "absdiff":
        return abs(av - bv)
    raise ValueError(op)


def _label_stats(rows: list[dict[str, Any]], indices: list[int]) -> dict[str, Any]:
    subset = [rows[i] for i in indices]
    if not subset:
        return {"n": 0}
    return {
        "n": len(subset),
        "raw_delta": _stats([r.get("raw_reward_sum_delta") for r in subset]),
        "env_delta": _stats([r.get("reward_env_sum_delta") for r in subset]),
        "training_delta": _stats([r.get("training_reward_sum_delta") for r in subset]),
        "raw_positive_fraction": float(np.mean([bool(r["raw_improved"]) for r in subset])),
        "env_positive_fraction": float(np.mean([bool(r["env_improved"]) for r in subset])),
        "both_positive_fraction": float(np.mean([bool(r["both_improved"]) for r in subset])),
    }


def _diversity(rows: list[dict[str, Any]], indices: list[int]) -> dict[str, Any]:
    subset = [rows[i] for i in indices]
    return {
        "n": len(subset),
        "source_counts": dict(Counter(str(r.get("source_tag")) for r in subset)),
        "range_counts": dict(Counter(str(r.get("source_range")) for r in subset)),
        "sign_counts": dict(Counter(str(r.get("feedback_pair_best_sign")) for r in subset)),
        "mode_feature_counts": dict(Counter(str(r.get("mode_obs_feature")) for r in subset)),
        "mode_horizon_counts": dict(Counter(str(r.get("mode_horizon")) for r in subset)),
        "mode_exact_counts": dict(Counter(str(r.get("mode_exact")) for r in subset)),
        "action_dim": _stats([r.get("action_dim") for r in subset]),
        "obs_dim": _stats([r.get("obs_dim") for r in subset]),
        "state_dim": _stats([r.get("state_dim") for r in subset]),
        "signature_score": _stats([r.get("signature_score") for r in subset]),
        "pre_done_count_mean": _stats([r.get("pre_done_count_mean") for r in subset]),
        "pre_action_saturation_mean": _stats([r.get("pre_action_saturation_mean") for r in subset]),
        "pre_next_state_l2_mean": _stats([r.get("pre_next_state_l2_mean") for r in subset]),
    }


def _balanced64(rows: list[dict[str, Any]], indices: list[int], limit: int = 64) -> list[int]:
    by_source: dict[str, list[int]] = defaultdict(list)
    for idx in indices:
        by_source[str(rows[idx].get("source_tag"))].append(idx)
    for bucket in by_source.values():
        bucket.sort()
    selected: list[int] = []
    cursor = {source: 0 for source in by_source}
    while len(selected) < int(limit):
        progressed = False
        for source in sorted(by_source):
            bucket = by_source[source]
            pos = cursor[source]
            if pos >= len(bucket):
                continue
            selected.append(bucket[pos])
            cursor[source] = pos + 1
            progressed = True
            if len(selected) >= int(limit):
                break
        if not progressed:
            break
    return selected


def _condition_pool(rows: list[dict[str, Any]], numeric_keys: list[str], categorical_keys: list[str]) -> list[tuple[str, Callable[[dict[str, Any]], bool]]]:
    conditions: list[tuple[str, Callable[[dict[str, Any]], bool]]] = []
    for key in numeric_keys:
        values = np.asarray([_f(row.get(key)) for row in rows if _f(row.get(key)) is not None], dtype=np.float64)
        if values.size < 16 or float(np.std(values)) <= 0.0:
            continue
        for q in (0.10, 0.25, 0.50, 0.75, 0.90):
            threshold = float(np.quantile(values, q))

            def le(row: dict[str, Any], k: str = key, th: float = threshold) -> bool:
                x = _f(row.get(k))
                return x is not None and x <= th

            def gt(row: dict[str, Any], k: str = key, th: float = threshold) -> bool:
                x = _f(row.get(k))
                return x is not None and x > th

            conditions.append((f"{key} <= {threshold:.6g}", le))
            conditions.append((f"{key} > {threshold:.6g}", gt))
    for key in categorical_keys:
        values = sorted({str(row.get(key)) for row in rows})
        if len(values) <= 1:
            continue
        for value in values:

            def eq(row: dict[str, Any], k: str = key, v: str = value) -> bool:
                return str(row.get(k)) == v

            conditions.append((f"{key} == {value}", eq))
    return conditions


def _evaluate_guard(
    rows: list[dict[str, Any]],
    name: str,
    fn: Callable[[dict[str, Any]], bool],
    *,
    min_n: int,
) -> dict[str, Any] | None:
    pass_indices = [idx for idx, row in enumerate(rows) if fn(row)]
    if len(pass_indices) < min_n:
        return None
    selected64 = _balanced64(rows, pass_indices, limit=64)
    return {
        "rule": name,
        "pass": {
            **_label_stats(rows, pass_indices),
            "diversity": _diversity(rows, pass_indices),
            "selected_rows": [{"env_idx": int(rows[i]["env_idx"])} for i in pass_indices],
        },
        "balanced64": {
            **_label_stats(rows, selected64),
            "diversity": _diversity(rows, selected64),
            "selected_rows": [{"env_idx": int(rows[i]["env_idx"])} for i in selected64],
        },
    }


def _build_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    fixed_csv = Path(report["fixed_list_csv"]).expanduser().resolve()
    rule = str(report.get("fixed_list_rule") or "")
    fixed_rows = [r for r in _load_csv(fixed_csv) if not rule or str(r.get("rule")) == rule]
    if len(fixed_rows) != len(report["per_env"]):
        raise RuntimeError(f"fixed CSV rows ({len(fixed_rows)}) and delta rows ({len(report['per_env'])}) differ")
    rows: list[dict[str, Any]] = []
    for idx, delta in enumerate(report["per_env"]):
        fixed = dict(fixed_rows[idx])
        row: dict[str, Any] = {**fixed, **delta}
        row["env_idx"] = int(delta.get("env_idx", idx))
        row["raw_improved"] = bool((_f(delta.get("raw_reward_sum_delta")) or 0.0) > 0.0)
        row["env_improved"] = bool((_f(delta.get("reward_env_sum_delta")) or 0.0) > 0.0)
        row["both_improved"] = bool(row["raw_improved"] and row["env_improved"])
        parts = str(row.get("source_tag") or "").split("_")
        row["source_seed"] = parts[0] if parts else str(row.get("source_tag"))
        row["source_range"] = parts[1] if len(parts) > 1 else "unknown"
        row.update(_parse_mode(row.get("best_feedback_mode")))
        base = _parse_mode(row.get("feedback_pair_best_base_mode"))
        row["base_mode_obs_feature"] = base["mode_obs_feature"]
        row["base_mode_horizon"] = base["mode_horizon"]
        row["base_mode_has_neg"] = base["mode_has_neg"]
        for key in (
            "signature_score",
            "best_feedback_minus_open_env",
            "feedback_pair_phase_gap_env",
            "feedback_suffix_gain_over_zero_env",
            "best_feedback_action_abs_prefix",
            "best_feedback_action_abs_suffix",
            "a_done_count",
            "a_first_done_step",
            "a_raw_reward_sum",
            "a_reward_env_sum",
            "a_training_reward_sum",
            "a_action_saturation",
            "a_action_absmean",
            "a_obs_value_std",
            "a_next_state_step_l2_delta_mean",
            "b_done_count",
            "b_first_done_step",
            "b_raw_reward_sum",
            "b_reward_env_sum",
            "b_training_reward_sum",
            "b_action_saturation",
            "b_action_absmean",
            "b_obs_value_std",
            "b_next_state_step_l2_delta_mean",
            "raw_reward_sum_delta",
            "reward_env_sum_delta",
            "training_reward_sum_delta",
            "action_dim",
            "obs_dim",
            "state_dim",
        ):
            if key in row:
                row[key] = _f(row.get(key))
        for stem, a_key, b_key in (
            ("pre_done_count", "a_done_count", "b_done_count"),
            ("pre_first_done_step", "a_first_done_step", "b_first_done_step"),
            ("pre_raw_reward_sum", "a_raw_reward_sum", "b_raw_reward_sum"),
            ("pre_reward_env_sum", "a_reward_env_sum", "b_reward_env_sum"),
            ("pre_training_reward_sum", "a_training_reward_sum", "b_training_reward_sum"),
            ("pre_action_saturation", "a_action_saturation", "b_action_saturation"),
            ("pre_action_absmean", "a_action_absmean", "b_action_absmean"),
            ("pre_obs_value_std", "a_obs_value_std", "b_obs_value_std"),
            ("pre_next_state_l2", "a_next_state_step_l2_delta_mean", "b_next_state_step_l2_delta_mean"),
        ):
            row[f"{stem}_mean"] = _safe_pair(row, a_key, b_key, "mean")
            row[f"{stem}_max"] = _safe_pair(row, a_key, b_key, "max")
            row[f"{stem}_min"] = _safe_pair(row, a_key, b_key, "min")
            row[f"{stem}_absdiff"] = _safe_pair(row, a_key, b_key, "absdiff")
        rows.append(row)
    return rows


def _write_summary(report: dict[str, Any], out_dir: Path) -> None:
    def fmt(x: Any) -> str:
        return "na" if x is None else f"{float(x):.4f}"

    baseline = report["baseline"]
    lines = [
        "# Pendulum-Like Pretrain Guard Scan",
        "",
        "Exploratory/read-only. Candidate rules use only fixed-list metadata and pre-update sidecar fields.",
        "",
        f"delta_report: {report['delta_report']}",
        f"n_envs: {report['n_envs']}",
        "",
        "## Baseline",
        "",
        f"- raw_positive_fraction: {baseline['raw_positive_fraction']}",
        f"- env_positive_fraction: {baseline['env_positive_fraction']}",
        f"- both_positive_fraction: {baseline['both_positive_fraction']}",
        f"- raw_delta_mean: {baseline['raw_delta']['mean']}",
        f"- env_delta_mean: {baseline['env_delta']['mean']}",
        "",
        "## Strongest Pretrain Numeric Associations",
        "",
        "| field | n | pearson raw | spearman raw |",
        "|---|---:|---:|---:|",
    ]
    for item in report["top_pre_correlations"][:20]:
        c = item["corr"]
        lines.append(f"| {item['field']} | {c.get('n')} | {fmt(c.get('pearson'))} | {fmt(c.get('spearman'))} |")
    lines.extend(
        [
            "",
            "## Best Single Pretrain Guards",
            "",
            "| rule | pass n | pass raw+ | pass both+ | pass raw mean | b64 n | b64 raw+ | b64 both+ | b64 raw mean | b64 sources | b64 signs |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for item in report["best_single_guards"][:20]:
        p = item["pass"]
        b = item["balanced64"]
        lines.append(
            f"| {item['rule']} | {p['n']} | {fmt(p['raw_positive_fraction'])} | {fmt(p['both_positive_fraction'])} | "
            f"{fmt(p['raw_delta']['mean'])} | {b['n']} | {fmt(b['raw_positive_fraction'])} | "
            f"{fmt(b['both_positive_fraction'])} | {fmt(b['raw_delta']['mean'])} | "
            f"{b['diversity']['source_counts']} | {b['diversity']['sign_counts']} |"
        )
    lines.extend(
        [
            "",
            "## Best Pair Pretrain Guards",
            "",
            "| rule | pass n | pass raw+ | pass both+ | pass raw mean | b64 n | b64 raw+ | b64 both+ | b64 raw mean | b64 sources | b64 signs |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for item in report["best_pair_guards"][:20]:
        p = item["pass"]
        b = item["balanced64"]
        lines.append(
            f"| {item['rule']} | {p['n']} | {fmt(p['raw_positive_fraction'])} | {fmt(p['both_positive_fraction'])} | "
            f"{fmt(p['raw_delta']['mean'])} | {b['n']} | {fmt(b['raw_positive_fraction'])} | "
            f"{fmt(b['both_positive_fraction'])} | {fmt(b['raw_delta']['mean'])} | "
            f"{b['diversity']['source_counts']} | {b['diversity']['sign_counts']} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- promote_pretrain_guard: {report['decision']['promote_pretrain_guard']}",
            f"- reason: {report['decision']['reason']}",
            "",
            "Acceptance bar used here: balanced64 exists, raw and both-positive fractions improve by at least "
            f"{report['decision']['min_positive_fraction_gain']}, and source/sign/mode diversity remains non-collapsed.",
        ]
    )
    (out_dir / "pendulum_like_pretrain_guard_scan.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_selected_csv(rows: list[dict[str, Any]], report: dict[str, Any], out_dir: Path) -> str | None:
    viable = report.get("decision", {}).get("best_viable_rules") or []
    if not viable:
        return None
    best = viable[0]
    selected_indices = [int(row.get("env_idx", -1)) for row in best["balanced64"].get("selected_rows", [])]
    if not selected_indices:
        return None
    rows_by_idx = {int(row["env_idx"]): row for row in rows}
    selected_rows = [rows_by_idx[idx] for idx in selected_indices if idx in rows_by_idx]
    if not selected_rows:
        return None
    csv_path = out_dir / "best_pretrain_guard_selected64.csv"
    fieldnames = [
        "rule",
        "guard_rule",
        "source_label",
        "source_tag",
        "source_candidate_index",
        "env_index",
        "env_seed",
        "seed",
        "full_frozen_h_json",
        "signature_score",
        "pendulum_like_strict",
        "best_feedback_mode",
        "feedback_pair_best_base_mode",
        "feedback_pair_best_sign",
        "best_feedback_action_abs_prefix",
        "best_feedback_action_abs_suffix",
        "pre_obs_value_std_mean",
        "pre_action_absmean_mean",
        "pre_action_saturation_mean",
        "pre_next_state_l2_mean",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in selected_rows:
            out = {
                "rule": "pendulum_like_pretrain_guard_selected64",
                "guard_rule": best["rule"],
                "source_label": row.get("source_label"),
                "source_tag": row.get("source_tag"),
                "source_candidate_index": row.get("source_candidate_index"),
                "env_index": row.get("env_index"),
                "env_seed": row.get("seed"),
                "seed": row.get("seed"),
                "full_frozen_h_json": row.get("full_frozen_h_json"),
                "signature_score": row.get("signature_score"),
                "pendulum_like_strict": row.get("pendulum_like_strict"),
                "best_feedback_mode": row.get("best_feedback_mode"),
                "feedback_pair_best_base_mode": row.get("feedback_pair_best_base_mode"),
                "feedback_pair_best_sign": row.get("feedback_pair_best_sign"),
                "best_feedback_action_abs_prefix": row.get("best_feedback_action_abs_prefix"),
                "best_feedback_action_abs_suffix": row.get("best_feedback_action_abs_suffix"),
                "pre_obs_value_std_mean": row.get("pre_obs_value_std_mean"),
                "pre_action_absmean_mean": row.get("pre_action_absmean_mean"),
                "pre_action_saturation_mean": row.get("pre_action_saturation_mean"),
                "pre_next_state_l2_mean": row.get("pre_next_state_l2_mean"),
            }
            writer.writerow(out)
    return str(csv_path)


def _decision(baseline: dict[str, Any], guards: list[dict[str, Any]], min_gain: float) -> dict[str, Any]:
    base_raw = float(baseline["raw_positive_fraction"])
    base_both = float(baseline["both_positive_fraction"])
    viable = []
    for item in guards:
        b = item["balanced64"]
        if int(b["n"]) < 64:
            continue
        div = b["diversity"]
        source_counts = div["source_counts"]
        sign_counts = div["sign_counts"]
        mode_features = div["mode_feature_counts"]
        source_ok = len(source_counts) >= 4 and min(source_counts.values()) >= 4
        sign_ok = len(sign_counts) >= 2 and min(sign_counts.values()) >= 12
        mode_ok = len(mode_features) >= 3
        raw_ok = float(b["raw_positive_fraction"]) >= base_raw + float(min_gain)
        both_ok = float(b["both_positive_fraction"]) >= base_both + float(min_gain)
        if source_ok and sign_ok and mode_ok and raw_ok and both_ok:
            viable.append(item)
    return {
        "promote_pretrain_guard": bool(viable),
        "min_positive_fraction_gain": float(min_gain),
        "viable_rule_count": int(len(viable)),
        "best_viable_rules": viable[:5],
        "reason": (
            "At least one pretraining-only guard clears positive-rate and diversity bars."
            if viable
            else "No pretraining-only guard cleared the positive-rate and diversity bars; keep the axis diagnostic."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delta-report", default=DEFAULT_DELTA_REPORT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-n", type=int, default=64)
    parser.add_argument("--min-positive-fraction-gain", type=float, default=0.10)
    args = parser.parse_args()

    delta_path = Path(args.delta_report).expanduser().resolve()
    delta_report = json.loads(delta_path.read_text(encoding="utf-8"))
    rows = _build_rows(delta_report)
    all_indices = list(range(len(rows)))
    baseline = {**_label_stats(rows, all_indices), "diversity": _diversity(rows, all_indices)}

    numeric_keys = [
        "signature_score",
        "best_feedback_minus_open_env",
        "feedback_pair_phase_gap_env",
        "feedback_suffix_gain_over_zero_env",
        "best_feedback_action_abs_prefix",
        "best_feedback_action_abs_suffix",
        "pre_done_count_mean",
        "pre_done_count_max",
        "pre_done_count_absdiff",
        "pre_first_done_step_mean",
        "pre_first_done_step_min",
        "pre_first_done_step_absdiff",
        "pre_raw_reward_sum_mean",
        "pre_raw_reward_sum_min",
        "pre_raw_reward_sum_absdiff",
        "pre_reward_env_sum_mean",
        "pre_training_reward_sum_mean",
        "pre_action_saturation_mean",
        "pre_action_saturation_max",
        "pre_action_saturation_absdiff",
        "pre_action_absmean_mean",
        "pre_action_absmean_max",
        "pre_action_absmean_absdiff",
        "pre_obs_value_std_mean",
        "pre_obs_value_std_absdiff",
        "pre_next_state_l2_mean",
        "pre_next_state_l2_min",
        "pre_next_state_l2_absdiff",
    ]
    categorical_keys = [
        "feedback_pair_best_sign",
        "mode_exact",
        "mode_obs_feature",
        "mode_horizon",
        "mode_has_neg",
        "mode_is_decay",
        "mode_is_early",
        "base_mode_obs_feature",
        "base_mode_horizon",
        "base_mode_has_neg",
        "pendulum_like_settle_strict",
    ]

    correlations = [
        {"field": key, "corr": _corr(rows, key, "raw_reward_sum_delta")}
        for key in numeric_keys
    ]
    top_corr = sorted(
        correlations,
        key=lambda item: abs(float(item["corr"].get("spearman") or 0.0)),
        reverse=True,
    )

    conditions = _condition_pool(rows, numeric_keys, categorical_keys)
    single = []
    for name, fn in conditions:
        item = _evaluate_guard(rows, name, fn, min_n=int(args.min_n))
        if item is not None:
            single.append(item)

    def guard_rank(item: dict[str, Any]) -> tuple[float, float, float, float]:
        b = item["balanced64"]
        return (
            float(b["both_positive_fraction"]),
            float(b["raw_positive_fraction"]),
            float(b["raw_delta"]["mean"]),
            float(item["pass"]["n"]),
        )

    single = sorted(single, key=guard_rank, reverse=True)
    primitive = single[:40]
    pair = []
    for left, right in combinations(primitive, 2):
        left_name = left["rule"]
        right_name = right["rule"]
        left_fn = next(fn for name, fn in conditions if name == left_name)
        right_fn = next(fn for name, fn in conditions if name == right_name)
        item = _evaluate_guard(
            rows,
            f"{left_name} AND {right_name}",
            lambda row, lf=left_fn, rf=right_fn: lf(row) and rf(row),
            min_n=int(args.min_n),
        )
        if item is not None:
            pair.append(item)
    pair = sorted(pair, key=guard_rank, reverse=True)
    all_ranked = sorted(single + pair, key=guard_rank, reverse=True)
    decision = _decision(baseline, all_ranked, min_gain=float(args.min_positive_fraction_gain))

    report = {
        "analysis_entry": "phase2_pendulum_like_pretrain_guard_scan",
        "contract": {
            "exploratory_only": True,
            "candidate_features_are_pretraining_only": True,
            "labels_are_trusted_pack_runner_post_pre_deltas": True,
            "does_not_modify_exact_scm": True,
            "does_not_modify_fit_model": True,
            "does_not_modify_ppo": True,
            "post_delta_fields_are_labels_only_not_guard_features": True,
        },
        "delta_report": str(delta_path),
        "fixed_list_csv": str(delta_report["fixed_list_csv"]),
        "fixed_list_rule": str(delta_report.get("fixed_list_rule")),
        "n_envs": int(len(rows)),
        "disallowed_guard_features": {
            "dimension_fields": ["action_dim", "obs_dim", "state_dim"],
            "source_identity_fields": ["source_seed", "source_range", "source_tag"],
            "reason": (
                "These are kept in diversity summaries but excluded from candidate guards: "
                "the current line is to preserve dimension/source coverage, not narrow it."
            ),
        },
        "baseline": baseline,
        "top_pre_correlations": top_corr,
        "best_single_guards": single[:50],
        "best_pair_guards": pair[:50],
        "decision": decision,
    }
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    selected_csv = _write_selected_csv(rows, report, out_dir)
    if selected_csv is not None:
        report["best_pretrain_guard_selected64_csv"] = selected_csv
    (out_dir / "pendulum_like_pretrain_guard_scan_report.json").write_text(
        json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_summary(report, out_dir)
    print(out_dir / "pendulum_like_pretrain_guard_scan.md")


if __name__ == "__main__":
    main()
