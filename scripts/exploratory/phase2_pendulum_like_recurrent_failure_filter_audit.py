#!/usr/bin/env python
"""Audit train-before-observable filters for recurrent post-opt failures.

This exploratory script learns *candidate filter rules* from labeled fixed-list
envs. Labels come from trusted pack-runner per-env post/pre deltas across
multiple PPO training seeds. Features are restricted to fixed-list metadata and
pre-update sidecars, so the proposed rules can be used before long training.

It is intentionally read-only: no exact SCM, PPO, or fit_model defaults are
modified.
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


DEFAULT_ALL_CSV = (
    "/home/chen/RLPFN/artifacts/phase2_pendulum_like_strict_cross_seed_all_candidates_0509/"
    "pendulum_like_signature_fixedlist.csv"
)
DEFAULT_SELECTED_CSV = (
    "/home/chen/RLPFN/artifacts/phase2_pendulum_like_strict_terminal_guard_fixedlists_0509/"
    "pendulum_like_terminal_guard_fixedlist.csv"
)
DEFAULT_SIDECAR_A = (
    "/home/chen/RLPFN/artifacts/phase2_pendulum_like_strict_n92_preupdate_sidecar_s512_u1_e1_advnorm_0509/"
    "semantic_sidecars/prior_update_000000_envs.jsonl"
)
DEFAULT_SIDECAR_B = (
    "/home/chen/RLPFN/artifacts/phase2_pendulum_like_strict_n92_preupdate_sidecar_s512_u1_e1_advnorm_seed9292_0509/"
    "semantic_sidecars/prior_update_000000_envs.jsonl"
)
DEFAULT_RECURRENT_REPORT = (
    "/home/chen/RLPFN/artifacts/phase2_pendulum_like_guardA_recurrent_postopt_failure_audit_0509/"
    "guardA_recurrent_postopt_failure_audit_report.json"
)
DEFAULT_OUTPUT_DIR = "/home/chen/RLPFN/artifacts/phase2_pendulum_like_recurrent_failure_filter_audit_0510"


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


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


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
        "q50": float(np.quantile(arr, 0.50)),
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
        "mode_is_time_profile": text.startswith("early_") or text.startswith("decay_"),
    }


def _rank_key_from_row(row: dict[str, Any], prefix: str) -> tuple[float, float, float, float]:
    done = _f(row.get(f"{prefix}_done_count"))
    first_done = _f(row.get(f"{prefix}_first_done_step"))
    sat = _f(row.get(f"{prefix}_action_saturation"))
    action_abs = _f(row.get(f"{prefix}_action_absmean"))
    return (
        float(done if done is not None else 1e9),
        -float(first_done if first_done is not None else -1.0),
        float(sat if sat is not None else 1e9),
        float(action_abs if action_abs is not None else 1e9),
    )


def _balanced_select(rows: list[dict[str, Any]], passing: list[bool], limit: int = 64) -> list[int]:
    by_source: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        if not passing[idx]:
            continue
        by_source[str(row.get("source_tag"))].append(idx)
    for bucket in by_source.values():
        bucket.sort(key=lambda idx: _rank_key_from_row(rows[idx], "a"))
    selected = []
    cursor = {source: 0 for source in by_source}
    while len(selected) < int(limit):
        progressed = False
        for source in sorted(by_source):
            bucket = by_source[source]
            idx = cursor[source]
            if idx >= len(bucket):
                continue
            selected.append(bucket[idx])
            cursor[source] = idx + 1
            progressed = True
            if len(selected) >= int(limit):
                break
        if not progressed:
            break
    return selected


def _diversity(rows: list[dict[str, Any]], indices: list[int]) -> dict[str, Any]:
    subset = [rows[i] for i in indices]
    return {
        "n": len(indices),
        "source_counts": dict(Counter(str(r.get("source_tag")) for r in subset)),
        "sign_counts": dict(Counter(str(r.get("feedback_pair_best_sign")) for r in subset)),
        "mode_feature_counts": dict(Counter(str(r.get("mode_obs_feature")) for r in subset)),
        "mode_horizon_counts": dict(Counter(str(r.get("mode_horizon")) for r in subset)),
        "mode_exact_counts": dict(Counter(str(r.get("mode_exact")) for r in subset)),
        "signature_score": _stats([r.get("signature_score") for r in subset]),
        "a_done_count": _stats([r.get("a_done_count") for r in subset]),
        "a_first_done_step": _stats([r.get("a_first_done_step") for r in subset]),
        "a_raw_reward_sum": _stats([r.get("a_raw_reward_sum") for r in subset]),
        "a_reward_env_sum": _stats([r.get("a_reward_env_sum") for r in subset]),
        "a_action_saturation": _stats([r.get("a_action_saturation") for r in subset]),
    }


def _make_numeric_conditions(rows: list[dict[str, Any]], keys: list[str]) -> list[tuple[str, Callable[[dict[str, Any]], bool]]]:
    out = []
    for key in keys:
        values = np.asarray([_f(row.get(key)) for row in rows if _f(row.get(key)) is not None], dtype=np.float64)
        if values.size < 8:
            continue
        for q in (0.10, 0.25, 0.50, 0.75, 0.90):
            threshold = float(np.quantile(values, q))

            def le(row: dict[str, Any], k: str = key, th: float = threshold) -> bool:
                x = _f(row.get(k))
                return x is not None and x <= th

            def gt(row: dict[str, Any], k: str = key, th: float = threshold) -> bool:
                x = _f(row.get(k))
                return x is not None and x > th

            out.append((f"{key} <= {threshold:.6g}", le))
            out.append((f"{key} > {threshold:.6g}", gt))
    return out


def _make_categorical_conditions(rows: list[dict[str, Any]], keys: list[str]) -> list[tuple[str, Callable[[dict[str, Any]], bool]]]:
    out = []
    for key in keys:
        for value in sorted({str(row.get(key)) for row in rows}):

            def eq(row: dict[str, Any], k: str = key, v: str = value) -> bool:
                return str(row.get(k)) == v

            out.append((f"{key} == {value}", eq))
    return out


def _evaluate_rule(
    rows: list[dict[str, Any]],
    labeled_indices: list[int],
    rule_name: str,
    fn: Callable[[dict[str, Any]], bool],
) -> dict[str, Any] | None:
    hits = [idx for idx in labeled_indices if fn(rows[idx])]
    if not hits:
        return None
    bad_hits = [idx for idx in hits if int(rows[idx].get("raw_fail_count", -1)) >= 2]
    always_hits = [idx for idx in hits if int(rows[idx].get("raw_fail_count", -1)) == 3]
    robust_hits = [idx for idx in hits if int(rows[idx].get("raw_fail_count", -1)) == 0]
    all_hits = [idx for idx, row in enumerate(rows) if fn(row)]
    passing = [not fn(row) for row in rows]
    post_selected = _balanced_select(rows, passing, limit=64)
    return {
        "rule": rule_name,
        "labeled_hit_count": len(hits),
        "bad_hit_count": len(bad_hits),
        "always_bad_hit_count": len(always_hits),
        "robust_hit_count": len(robust_hits),
        "bad_rate_among_hits": float(len(bad_hits) / len(hits)),
        "robust_rate_among_hits": float(len(robust_hits) / len(hits)),
        "all_candidate_hit_count": len(all_hits),
        "remaining_candidate_count": len(rows) - len(all_hits),
        "balanced_select_count_after_filter": len(post_selected),
        "post_filter_diversity": _diversity(rows, post_selected),
    }


def _evaluate_hit_set(
    rows: list[dict[str, Any]],
    labeled_indices: list[int],
    rule_name: str,
    hit_set: set[int],
) -> dict[str, Any] | None:
    hits = [idx for idx in labeled_indices if idx in hit_set]
    if not hits:
        return None
    bad_hits = [idx for idx in hits if int(rows[idx].get("raw_fail_count", -1)) >= 2]
    always_hits = [idx for idx in hits if int(rows[idx].get("raw_fail_count", -1)) == 3]
    robust_hits = [idx for idx in hits if int(rows[idx].get("raw_fail_count", -1)) == 0]
    passing = [idx not in hit_set for idx in range(len(rows))]
    post_selected = _balanced_select(rows, passing, limit=64)
    return {
        "rule": rule_name,
        "labeled_hit_count": len(hits),
        "bad_hit_count": len(bad_hits),
        "always_bad_hit_count": len(always_hits),
        "robust_hit_count": len(robust_hits),
        "bad_rate_among_hits": float(len(bad_hits) / len(hits)),
        "robust_rate_among_hits": float(len(robust_hits) / len(hits)),
        "all_candidate_hit_count": len(hit_set),
        "remaining_candidate_count": len(rows) - len(hit_set),
        "balanced_select_count_after_filter": len(post_selected),
        "post_filter_diversity": _diversity(rows, post_selected),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-csv", default=DEFAULT_ALL_CSV)
    parser.add_argument("--selected-csv", default=DEFAULT_SELECTED_CSV)
    parser.add_argument("--preupdate-sidecar-a", default=DEFAULT_SIDECAR_A)
    parser.add_argument("--preupdate-sidecar-b", default=DEFAULT_SIDECAR_B)
    parser.add_argument("--recurrent-report", default=DEFAULT_RECURRENT_REPORT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    all_rows = _load_csv(args.all_csv)
    side_a = {int(row["env_idx"]): row for row in _load_jsonl(args.preupdate_sidecar_a)}
    side_b = {int(row["env_idx"]): row for row in _load_jsonl(args.preupdate_sidecar_b)}
    labels = json.loads(Path(args.recurrent_report).expanduser().resolve().read_text(encoding="utf-8"))
    label_by_candidate = {}
    for row in labels["recurrent_failures"] + labels["robust_successes"]:
        label_by_candidate[int(row["source_candidate_index"])] = row
    # Add one-fail rows from selected CSV as explicitly labeled non-recurrent.
    selected_rows = _load_csv(args.selected_csv)
    selected_candidates = {int(row["source_candidate_index"]) for row in selected_rows}
    for idx in selected_candidates:
        label_by_candidate.setdefault(idx, {"raw_fail_count": 1})

    rows: list[dict[str, Any]] = []
    for idx, base in enumerate(all_rows):
        row: dict[str, Any] = dict(base)
        parsed = _parse_mode(row.get("best_feedback_mode"))
        row.update(parsed)
        row["candidate_index"] = int(idx)
        label = label_by_candidate.get(idx)
        row["is_labeled"] = label is not None
        if label is not None:
            row["raw_fail_count"] = int(label.get("raw_fail_count", 1))
            row["env_fail_count"] = int(label.get("env_fail_count", row["raw_fail_count"]))
            row["raw_mean_postopt_delta"] = _f(label.get("raw_mean"))
            row["env_mean_postopt_delta"] = _f(label.get("env_mean"))
            row["train_mean_postopt_delta"] = _f(label.get("train_mean"))
        else:
            row["raw_fail_count"] = None
            row["env_fail_count"] = None
        for prefix, side in (("a", side_a[idx]), ("b", side_b[idx])):
            row[f"{prefix}_done_count"] = _f(side.get("done_count"))
            row[f"{prefix}_first_done_step"] = _f(side.get("first_done_step"))
            row[f"{prefix}_raw_reward_sum"] = _f(side.get("raw_reward_sum"))
            row[f"{prefix}_reward_env_sum"] = _f(side.get("reward_env_sum"))
            row[f"{prefix}_reward_ctrl_sum"] = _f(side.get("reward_ctrl_sum"))
            row[f"{prefix}_training_reward_sum"] = _f(side.get("training_reward_sum"))
            row[f"{prefix}_action_saturation"] = _f(side.get("raw_policy_action_unit_clip_saturation_fraction"))
            row[f"{prefix}_action_absmean"] = _f(side.get("action_value_absmean"))
            row[f"{prefix}_obs_value_std"] = _f(side.get("obs_value_std"))
            row[f"{prefix}_next_state_step_l2_delta_mean"] = _f(side.get("next_state_step_l2_delta_mean"))
        for key in (
            "signature_score",
            "best_feedback_minus_open_env",
            "feedback_pair_phase_gap_env",
            "feedback_suffix_gain_over_zero_env",
            "best_feedback_action_abs_prefix",
            "best_feedback_action_abs_suffix",
        ):
            row[key] = _f(row.get(key))
        row["a_b_done_count_max"] = max(row["a_done_count"], row["b_done_count"])
        row["a_b_first_done_step_min"] = min(row["a_first_done_step"], row["b_first_done_step"])
        row["a_b_raw_reward_sum_min"] = min(row["a_raw_reward_sum"], row["b_raw_reward_sum"])
        row["a_b_action_saturation_max"] = max(row["a_action_saturation"], row["b_action_saturation"])
        rows.append(row)

    labeled_indices = [i for i, row in enumerate(rows) if row["is_labeled"]]
    bad_indices = [i for i in labeled_indices if int(rows[i]["raw_fail_count"]) >= 2]
    robust_indices = [i for i in labeled_indices if int(rows[i]["raw_fail_count"]) == 0]

    numeric_keys = [
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
        "a_reward_ctrl_sum",
        "a_training_reward_sum",
        "a_action_saturation",
        "a_action_absmean",
        "a_obs_value_std",
        "a_next_state_step_l2_delta_mean",
        "b_done_count",
        "b_first_done_step",
        "b_raw_reward_sum",
        "b_reward_env_sum",
        "b_reward_ctrl_sum",
        "b_training_reward_sum",
        "b_action_saturation",
        "b_action_absmean",
        "b_obs_value_std",
        "b_next_state_step_l2_delta_mean",
        "a_b_done_count_max",
        "a_b_first_done_step_min",
        "a_b_raw_reward_sum_min",
        "a_b_action_saturation_max",
    ]
    categorical_keys = [
        "source_tag",
        "feedback_pair_best_sign",
        "mode_exact",
        "mode_obs_feature",
        "mode_horizon",
        "mode_has_neg",
        "mode_is_time_profile",
    ]
    conditions = _make_numeric_conditions(rows, numeric_keys) + _make_categorical_conditions(rows, categorical_keys)

    rule_rows = []
    for name, fn in conditions:
        item = _evaluate_rule(rows, labeled_indices, name, fn)
        if item is None:
            continue
        if item["labeled_hit_count"] < 4:
            continue
        rule_rows.append(item)
    best_exclusion_rules = sorted(
        rule_rows,
        key=lambda r: (
            r["bad_rate_among_hits"],
            r["bad_hit_count"],
            -r["robust_hit_count"],
            r["remaining_candidate_count"],
        ),
        reverse=True,
    )[:30]
    conservative_rules = [
        r for r in best_exclusion_rules
        if r["bad_hit_count"] >= 4
        and r["robust_hit_count"] <= 1
        and r["remaining_candidate_count"] >= 64
        and r["balanced_select_count_after_filter"] >= 64
    ]

    primitive_rules = []
    for name, fn in conditions:
        item = _evaluate_rule(rows, labeled_indices, name, fn)
        if item is None:
            continue
        if (
            item["bad_hit_count"] >= 4
            and item["bad_rate_among_hits"] >= 0.50
            and item["remaining_candidate_count"] >= 64
            and item["balanced_select_count_after_filter"] >= 64
        ):
            primitive_rules.append(
                {
                    "name": name,
                    "hit_set": {idx for idx, row in enumerate(rows) if fn(row)},
                    "summary": item,
                }
            )
    primitive_rules.sort(
        key=lambda r: (
            r["summary"]["bad_rate_among_hits"],
            r["summary"]["bad_hit_count"],
            -r["summary"]["robust_hit_count"],
            -r["summary"]["all_candidate_hit_count"],
        ),
        reverse=True,
    )
    primitive_rules = primitive_rules[:24]

    combo_rules = []
    for width in (2, 3):
        for group in combinations(primitive_rules, width):
            names = [g["name"] for g in group]
            hit_set: set[int] = set()
            for g in group:
                hit_set |= set(g["hit_set"])
            item = _evaluate_hit_set(rows, labeled_indices, " OR ".join(names), hit_set)
            if item is None:
                continue
            if (
                item["bad_hit_count"] >= 8
                and item["bad_rate_among_hits"] >= 0.55
                and item["robust_hit_count"] <= 3
                and item["remaining_candidate_count"] >= 64
                and item["balanced_select_count_after_filter"] >= 64
            ):
                combo_rules.append(item)
    combo_rules = sorted(
        combo_rules,
        key=lambda r: (
            r["bad_hit_count"],
            r["always_bad_hit_count"],
            r["bad_rate_among_hits"],
            -r["robust_hit_count"],
            r["remaining_candidate_count"],
        ),
        reverse=True,
    )[:30]

    baseline_indices = _balanced_select(rows, [True] * len(rows), limit=64)
    report = {
        "analysis_entry": "phase2_pendulum_like_recurrent_failure_filter_audit",
        "contract": {
            "exploratory_only": True,
            "labels_are_postopt_per_env_deltas": True,
            "candidate_features_are_pretraining_only": True,
            "does_not_modify_exact_scm": True,
            "does_not_modify_fit_model": True,
            "does_not_modify_ppo": True,
            "does_not_resample_envs": True,
        },
        "all_candidate_count": len(rows),
        "labeled_count": len(labeled_indices),
        "bad_recurrent_count": len(bad_indices),
        "robust_success_count": len(robust_indices),
        "baseline_balanced_selection": _diversity(rows, baseline_indices),
        "bad_recurrent_feature_summary": _diversity(rows, bad_indices),
        "robust_success_feature_summary": _diversity(rows, robust_indices),
        "best_exclusion_rules": best_exclusion_rules,
        "conservative_candidate_rules": conservative_rules,
        "primitive_rules_for_combinations": [
            {k: v for k, v in item["summary"].items() if k != "post_filter_diversity"}
            for item in primitive_rules
        ],
        "combo_candidate_rules": combo_rules,
    }
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "recurrent_failure_filter_audit_report.json"
    json_path.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Recurrent Failure Filter Audit",
        "",
        "Labels: per-env post/pre raw delta failure count across guardA seeds. Features: fixed-list metadata and pre-update sidecars only.",
        "",
        f"all_candidate_count: {len(rows)}",
        f"labeled_count: {len(labeled_indices)}",
        f"bad_recurrent_count: {len(bad_indices)}",
        f"robust_success_count: {len(robust_indices)}",
        "",
        "## Bad vs Robust Feature Summary",
        "",
        f"- bad recurrent source_counts: {report['bad_recurrent_feature_summary']['source_counts']}",
        f"- robust source_counts: {report['robust_success_feature_summary']['source_counts']}",
        f"- bad recurrent mode_exact_counts: {report['bad_recurrent_feature_summary']['mode_exact_counts']}",
        f"- robust mode_exact_counts: {report['robust_success_feature_summary']['mode_exact_counts']}",
        f"- bad recurrent a_done_count: {report['bad_recurrent_feature_summary']['a_done_count']}",
        f"- robust a_done_count: {report['robust_success_feature_summary']['a_done_count']}",
        f"- bad recurrent a_raw_reward_sum: {report['bad_recurrent_feature_summary']['a_raw_reward_sum']}",
        f"- robust a_raw_reward_sum: {report['robust_success_feature_summary']['a_raw_reward_sum']}",
        "",
        "## Best Single Exclusion Rules",
        "",
        "| rule | labeled hits | bad hits | always bad | robust hits | bad rate | all hits | remaining | balanced n |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in best_exclusion_rules[:20]:
        lines.append(
            f"| {r['rule']} | {r['labeled_hit_count']} | {r['bad_hit_count']} | "
            f"{r['always_bad_hit_count']} | {r['robust_hit_count']} | {r['bad_rate_among_hits']} | "
            f"{r['all_candidate_hit_count']} | {r['remaining_candidate_count']} | {r['balanced_select_count_after_filter']} |"
        )
    lines.extend([
        "",
        "## Conservative Candidate Rules",
        "",
        "| rule | labeled hits | bad hits | robust hits | bad rate | all hits | remaining | source counts after balanced select | mode features after balanced select |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
    ])
    for r in conservative_rules[:12]:
        div = r["post_filter_diversity"]
        lines.append(
            f"| {r['rule']} | {r['labeled_hit_count']} | {r['bad_hit_count']} | {r['robust_hit_count']} | "
            f"{r['bad_rate_among_hits']} | {r['all_candidate_hit_count']} | {r['remaining_candidate_count']} | "
            f"{div['source_counts']} | {div['mode_feature_counts']} |"
        )
    lines.extend([
        "",
        "## Candidate Composite Exclusion Rules",
        "",
        "These are OR-combinations of the best single pre-update rules. They are still exploratory; they are useful only if a fresh fixed-list smoke confirms post-opt improvement.",
        "",
        "| rule | labeled hits | bad hits | always bad | robust hits | bad rate | all hits | remaining | source counts after balanced select |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ])
    for r in combo_rules[:12]:
        div = r["post_filter_diversity"]
        lines.append(
            f"| {r['rule']} | {r['labeled_hit_count']} | {r['bad_hit_count']} | {r['always_bad_hit_count']} | "
            f"{r['robust_hit_count']} | {r['bad_rate_among_hits']} | {r['all_candidate_hit_count']} | "
            f"{r['remaining_candidate_count']} | {div['source_counts']} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- This is not yet a final filter. It finds candidate pretraining-observable exclusion rules.",
        "- A viable production/yield control rule must remove recurrent bad labels while keeping at least 64 source-balanced candidates and mode diversity.",
        "- Rules that mostly encode training outcome are disallowed here; all rules above use only fixed metadata and pre-update sidecars.",
    ])
    md_path = out_dir / "recurrent_failure_filter_audit.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
