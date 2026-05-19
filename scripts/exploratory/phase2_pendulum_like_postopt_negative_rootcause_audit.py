#!/usr/bin/env python
"""Audit pretraining-observable causes of post-opt negative Pendulum-like envs.

The health label is the trusted pack-runner per-env post/pre delta:
raw_reward_sum_delta > 0 and reward_env_sum_delta > 0.  Features are restricted
to fixed-list metadata, pre-update sidecar fields, and frozen-h environment
generation parameters.  This script does not mutate exact SCM, fit_model, PPO,
or milestone defaults.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_DELTA_REPORT = (
    "/home/chen/RLPFN/artifacts/phase2_pendulum_like_sat_only_n64_sidecar_summary_seed9100_0510/"
    "pendulum_like_sidecar_delta_summary_report.json"
)
DEFAULT_FIXED_CSV = (
    "/home/chen/RLPFN/artifacts/phase2_pendulum_like_recurrent_filter_fixedlists_0510/"
    "pendulum_like_sat_only_fixedlist.csv"
)
DEFAULT_OUTPUT_DIR = "/home/chen/RLPFN/artifacts/phase2_pendulum_like_postopt_negative_rootcause_audit_0510"

PROTECTED_DIM_FIELDS = {"action_dim", "obs_dim", "state_dim", "noise_dim", "zero_pad_dim"}
DISALLOWED_H_PREFIXES = (
    "reinforce_",
    "pg_",
    "policy_gradient_",
    "normalized_q_",
    "next_state_flow_",
    "batch_",
    "reference_scm_",
    "transition_",
)
DISALLOWED_H_FIELDS = {
    "fixed_frozen_h_json",
    "family",
    "prior_mlp_activations",
    "action_slot_dim",
    "obs_slot_dim",
    "reward_clip",
    "reward_norm_clip",
    "reward_norm_eps",
    "discount",
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return None if not math.isfinite(value) else value
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    return value


def _f(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def _load_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).expanduser().resolve().open("r", encoding="utf-8", newline="") as handle:
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
    }


def _field_group(name: str) -> str:
    if name in PROTECTED_DIM_FIELDS:
        return "protected_dimension"
    if name.startswith("h_"):
        raw = name[2:]
        if raw in {
            "alpha",
            "init_std",
            "init_state_std",
            "init_action_std",
            "lengthscale",
            "outputscale",
            "noise",
            "noise_std",
            "state_noise_std",
            "state_clip",
            "terminal_reset_count_target",
            "reward_dropout_ratio",
            "reward_scale",
            "ctrl_reward_weight",
            "reinforce_reward_tanh_bound",
            "gp_rff_features",
            "prior_mlp_hidden_dim",
            "_constrained_noise_u",
            "_constrained_obs_u",
            "_ctrl_reward_enable_u",
        }:
            return "generator_knob"
        return "h_other"
    if name.startswith("sig_") or name.startswith("fixed_"):
        return "signature"
    if name.startswith("pre_"):
        return "preupdate_sidecar"
    return "other"


def _h_scalar_features(h: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in h.items():
        if key in DISALLOWED_H_FIELDS or key in PROTECTED_DIM_FIELDS:
            continue
        if key.startswith(DISALLOWED_H_PREFIXES):
            continue
        val = _f(value)
        if val is None:
            continue
        out[f"h_{key}"] = val
        if val > 0 and key in {
            "lengthscale",
            "outputscale",
            "noise_std",
            "state_noise_std",
            "init_std",
            "init_state_std",
            "init_action_std",
            "ctrl_reward_weight",
            "reward_dropout_ratio",
            "reward_scale",
            "terminal_reset_count_target",
        }:
            out[f"h_log10_{key}"] = float(math.log10(max(val, 1e-12)))
    return out


def _build_rows(delta_report: dict[str, Any], fixed_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows = []
    per_env = list(delta_report["per_env"])
    if len(per_env) != len(fixed_rows):
        raise RuntimeError(f"delta rows {len(per_env)} != fixed rows {len(fixed_rows)}")
    for idx, (delta, fixed) in enumerate(zip(per_env, fixed_rows)):
        h = _load_json(fixed["full_frozen_h_json"])
        raw_delta = float(delta["raw_reward_sum_delta"])
        env_delta = float(delta["reward_env_sum_delta"])
        row: dict[str, Any] = {
            "env_idx": int(idx),
            "healthy_postopt": bool(raw_delta > 0.0 and env_delta > 0.0),
            "bad_postopt": bool(not (raw_delta > 0.0 and env_delta > 0.0)),
            "raw_reward_sum_delta": raw_delta,
            "reward_env_sum_delta": env_delta,
            "source_tag": fixed.get("source_tag"),
            "best_feedback_mode": fixed.get("best_feedback_mode"),
            "feedback_pair_best_sign": fixed.get("feedback_pair_best_sign"),
            "source_candidate_index": int(fixed.get("source_candidate_index", idx)),
        }
        for key in (
            "signature_score",
            "best_feedback_minus_open_env",
            "feedback_pair_phase_gap_env",
            "feedback_suffix_gain_over_zero_env",
            "best_feedback_action_abs_prefix",
            "best_feedback_action_abs_suffix",
        ):
            row[f"sig_{key}"] = _f(fixed.get(key))
        for key in (
            "raw_reward_sum_first",
            "reward_env_sum_first",
            "reward_ctrl_sum_first",
            "training_reward_sum_first",
            "done_count_first",
            "first_done_step_first",
            "action_value_absmean_first",
        ):
            row[f"pre_{key}"] = _f(delta.get(key))
        row.update(_h_scalar_features(h))
        rows.append(row)
    return rows


def _evaluate_rule(rows: list[dict[str, Any]], name: str, hit: set[int]) -> dict[str, Any] | None:
    if not hit:
        return None
    bad = [idx for idx in hit if rows[idx]["bad_postopt"]]
    good = [idx for idx in hit if rows[idx]["healthy_postopt"]]
    kept = [idx for idx in range(len(rows)) if idx not in hit]
    kept_good = sum(1 for idx in kept if rows[idx]["healthy_postopt"])
    kept_bad = sum(1 for idx in kept if rows[idx]["bad_postopt"])
    total_bad = sum(1 for row in rows if row["bad_postopt"])
    total_good = sum(1 for row in rows if row["healthy_postopt"])
    return {
        "rule": name,
        "hit_count": int(len(hit)),
        "bad_hit_count": int(len(bad)),
        "good_hit_count": int(len(good)),
        "bad_precision": float(len(bad) / len(hit)),
        "bad_recall": float(len(bad) / max(1, total_bad)),
        "good_loss": float(len(good) / max(1, total_good)),
        "kept_count": int(len(kept)),
        "kept_good_count": int(kept_good),
        "kept_bad_count": int(kept_bad),
        "kept_good_fraction": float(kept_good / max(1, len(kept))),
        "kept_source_counts": dict(Counter(str(rows[idx]["source_tag"]) for idx in kept)),
        "hit_source_counts": dict(Counter(str(rows[idx]["source_tag"]) for idx in hit)),
    }


def _scan_numeric_rules(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    rules = []
    for key in keys:
        vals = np.asarray([_f(row.get(key)) for row in rows if _f(row.get(key)) is not None], dtype=np.float64)
        if vals.size < 16 or float(np.std(vals)) <= 1e-12:
            continue
        for q in (0.10, 0.20, 0.25, 0.33, 0.50, 0.67, 0.75, 0.80, 0.90):
            threshold = float(np.quantile(vals, q))
            for op in ("<=", ">"):
                if op == "<=":
                    hit = {idx for idx, row in enumerate(rows) if _f(row.get(key)) is not None and float(row[key]) <= threshold}
                else:
                    hit = {idx for idx, row in enumerate(rows) if _f(row.get(key)) is not None and float(row[key]) > threshold}
                item = _evaluate_rule(rows, f"{key} {op} {threshold:.6g}", hit)
                if item is None:
                    continue
                if item["hit_count"] >= 4 and item["bad_hit_count"] >= 3:
                    item["field"] = key
                    item["group"] = _field_group(key)
                    rules.append(item)
    return rules


def _scan_categorical_rules(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    rules = []
    for key in keys:
        values = sorted({str(row.get(key)) for row in rows})
        for value in values:
            hit = {idx for idx, row in enumerate(rows) if str(row.get(key)) == value}
            item = _evaluate_rule(rows, f"{key} == {value}", hit)
            if item is None:
                continue
            if item["hit_count"] >= 2 and item["bad_hit_count"] >= 2:
                item["field"] = key
                item["group"] = "categorical"
                rules.append(item)
    return rules


def _scan_combo_rules(rows: list[dict[str, Any]], single_rules: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prim = []
    for rule in single_rules:
        if rule["bad_precision"] >= 0.55 and rule["bad_hit_count"] >= 4 and rule["good_hit_count"] <= 8:
            prim.append(rule)
    prim = sorted(
        prim,
        key=lambda r: (r["bad_precision"], r["bad_hit_count"], -r["good_hit_count"]),
        reverse=True,
    )[:20]

    def hit_for_rule(rule: dict[str, Any]) -> set[int]:
        text = str(rule["rule"])
        if " <= " in text:
            key, threshold = text.split(" <= ", 1)
            return {idx for idx, row in enumerate(rows) if _f(row.get(key)) is not None and float(row[key]) <= float(threshold)}
        if " > " in text:
            key, threshold = text.split(" > ", 1)
            return {idx for idx, row in enumerate(rows) if _f(row.get(key)) is not None and float(row[key]) > float(threshold)}
        if " == " in text:
            key, value = text.split(" == ", 1)
            return {idx for idx, row in enumerate(rows) if str(row.get(key)) == value}
        return set()

    out = []
    for width in (2, 3):
        for group in combinations(prim, width):
            hit: set[int] = set()
            names = []
            for rule in group:
                names.append(str(rule["rule"]))
                hit |= hit_for_rule(rule)
            item = _evaluate_rule(rows, " OR ".join(names), hit)
            if item is None:
                continue
            if item["hit_count"] >= 8 and item["bad_precision"] >= 0.55 and item["bad_hit_count"] >= 8:
                out.append(item)
    return sorted(
        out,
        key=lambda r: (r["bad_hit_count"], r["bad_precision"], -r["good_hit_count"], r["kept_good_fraction"]),
        reverse=True,
    )[:24]


def _feature_contrast(rows: list[dict[str, Any]], numeric_keys: list[str]) -> list[dict[str, Any]]:
    good = [row for row in rows if row["healthy_postopt"]]
    bad = [row for row in rows if row["bad_postopt"]]
    out = []
    for key in numeric_keys:
        good_vals = [row.get(key) for row in good if _f(row.get(key)) is not None]
        bad_vals = [row.get(key) for row in bad if _f(row.get(key)) is not None]
        if len(good_vals) < 4 or len(bad_vals) < 4:
            continue
        gs = _stats(good_vals)
        bs = _stats(bad_vals)
        pooled = np.std([float(v) for v in good_vals + bad_vals])
        if float(pooled) <= 1e-12:
            continue
        effect = 0.0 if pooled == 0.0 else (float(bs["mean"]) - float(gs["mean"])) / float(pooled)
        out.append(
            {
                "field": key,
                "group": _field_group(key),
                "good": gs,
                "bad": bs,
                "bad_minus_good_effect": float(effect),
            }
        )
    return sorted(out, key=lambda x: abs(float(x["bad_minus_good_effect"])), reverse=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delta-report", default=DEFAULT_DELTA_REPORT)
    parser.add_argument("--fixed-csv", default=DEFAULT_FIXED_CSV)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    delta_report = _load_json(args.delta_report)
    fixed_rows = _load_csv(args.fixed_csv)
    rows = _build_rows(delta_report, fixed_rows)
    numeric_keys = sorted(
        key for key in {k for row in rows for k in row}
        if all(not isinstance(row.get(key), str) for row in rows if row.get(key) is not None)
        and any(_f(row.get(key)) is not None for row in rows)
        and key not in {
            "env_idx",
            "source_candidate_index",
            "raw_reward_sum_delta",
            "reward_env_sum_delta",
            "healthy_postopt",
            "bad_postopt",
        }
    )
    numeric_keys = [k for k in numeric_keys if _field_group(k) != "protected_dimension"]
    categorical_keys = ["source_tag", "best_feedback_mode", "feedback_pair_best_sign"]

    single_rules = _scan_numeric_rules(rows, numeric_keys) + _scan_categorical_rules(rows, categorical_keys)
    single_rules = sorted(
        single_rules,
        key=lambda r: (r["bad_precision"], r["bad_hit_count"], -r["good_hit_count"], r["bad_recall"]),
        reverse=True,
    )
    conservative_rules = [
        r for r in single_rules
        if r["bad_precision"] >= 0.65
        and r["bad_hit_count"] >= 4
        and r["good_hit_count"] <= 4
        and r["kept_count"] >= 48
    ]
    combo_rules = _scan_combo_rules(rows, single_rules)
    contrast = _feature_contrast(rows, numeric_keys)

    report = {
        "analysis_entry": "phase2_pendulum_like_postopt_negative_rootcause_audit",
        "contract": {
            "exploratory_only": True,
            "health_label": "raw_reward_sum_delta > 0 and reward_env_sum_delta > 0",
            "uses_postopt_only_as_label": True,
            "features_are_pretraining_observable": True,
            "does_not_modify_exact_scm": True,
            "does_not_modify_fit_model": True,
            "does_not_modify_ppo": True,
            "does_not_use_protected_dimension_fields_for_candidate_rules": True,
        },
        "delta_report": str(Path(args.delta_report).expanduser().resolve()),
        "fixed_csv": str(Path(args.fixed_csv).expanduser().resolve()),
        "n": int(len(rows)),
        "healthy_count": int(sum(1 for row in rows if row["healthy_postopt"])),
        "bad_count": int(sum(1 for row in rows if row["bad_postopt"])),
        "bad_source_counts": dict(Counter(str(row["source_tag"]) for row in rows if row["bad_postopt"])),
        "healthy_source_counts": dict(Counter(str(row["source_tag"]) for row in rows if row["healthy_postopt"])),
        "strongest_contrast": contrast[:40],
        "best_single_rules": single_rules[:60],
        "conservative_rules": conservative_rules[:30],
        "combo_rules": combo_rules[:24],
    }

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "postopt_negative_rootcause_audit_report.json"
    json_path.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Pendulum-Like Post-Opt Negative Root Cause Audit",
        "",
        f"delta_report: {report['delta_report']}",
        f"fixed_csv: {report['fixed_csv']}",
        f"n: {report['n']}",
        f"healthy_count: {report['healthy_count']}",
        f"bad_count: {report['bad_count']}",
        "",
        "## Strongest Pretraining Feature Contrasts",
        "",
        "| field | group | bad mean | good mean | bad-good effect | bad q50 | good q50 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for item in contrast[:20]:
        lines.append(
            f"| {item['field']} | {item['group']} | {item['bad']['mean']} | {item['good']['mean']} | "
            f"{item['bad_minus_good_effect']} | {item['bad']['q50']} | {item['good']['q50']} |"
        )
    lines.extend([
        "",
        "## Conservative Single Exclusion Rules",
        "",
        "| rule | group | hit | bad hit | good hit | bad precision | bad recall | kept good/kept |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for item in conservative_rules[:20]:
        lines.append(
            f"| {item['rule']} | {item.get('group')} | {item['hit_count']} | {item['bad_hit_count']} | "
            f"{item['good_hit_count']} | {item['bad_precision']} | {item['bad_recall']} | {item['kept_good_fraction']} |"
        )
    lines.extend([
        "",
        "## Candidate Composite Exclusion Rules",
        "",
        "| rule | hit | bad hit | good hit | bad precision | bad recall | kept good/kept | kept sources |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ])
    for item in combo_rules[:16]:
        lines.append(
            f"| {item['rule']} | {item['hit_count']} | {item['bad_hit_count']} | {item['good_hit_count']} | "
            f"{item['bad_precision']} | {item['bad_recall']} | {item['kept_good_fraction']} | {item['kept_source_counts']} |"
        )
    lines.extend([
        "",
        "## Interpretation Guard",
        "",
        "- Rules above are not health claims. They are candidate pretraining filters.",
        "- A rule becomes useful only if a new fixed-list smoke improves post/pre deltas without reducing diversity.",
    ])
    md_path = out_dir / "postopt_negative_rootcause_audit.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
