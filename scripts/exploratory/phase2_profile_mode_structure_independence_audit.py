#!/usr/bin/env python
"""Audit hidden coupling between profile modes and structural parameters.

Read-only diagnostic for the profile-band mainline.  It reconstructs the
selector input pool with the same label joins used by the selector, then
compares it with the selected fixed list.  The purpose is to catch cases where
terminal reset, dimensions, or reward/input-gain parameters silently encode a
profile mode.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


ART = Path("/home/chen/RLPFN/artifacts")
REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.exploratory import phase2_profile_band_selector as selector  # noqa: E402


DEFAULT_SELECTOR_REPORT = (
    ART
    / "phase2_profile_band_selector_v2n_prior_pack_uniform_terminal_gym_baseline_modes_seed9730_0515"
    / "profile_band_selector_report.json"
)
DEFAULT_SELECTED_CSV = (
    ART
    / "phase2_profile_band_selector_v2n_prior_pack_uniform_terminal_gym_baseline_modes_seed9730_0515"
    / "selected_profile_band_prior_envs.csv"
)
DEFAULT_RELATIVE_AUDIT_JSON = (
    ART / "phase2_profile_v2n_relative_complexity_audit_0515" / "profile_relative_complexity_audit.json"
)
DEFAULT_OUTPUT_DIR = ART / "phase2_profile_v2n_mode_structure_independence_audit_0515"

MODE_KEYS = (
    "locomotion_witness",
    "low_energy_not_ctrl_only",
    "sign_or_phase_sensitive",
    "state_conditioned_energy_injection",
    "strict_feedback_candidate",
    "feedback_near_best_support_set",
    "pendulum_axis_support",
    "pendulum_short_gap",
    "pendulum_medium_settle",
    "pendulum_strict_settle",
    "pendulum_sustained_settle",
    "pendulum_settle_yield_repair_planned",
    "pendulum_settle_effective_support",
)

STRUCTURE_KEYS = (
    "terminal_reset_count_target",
    "action_dim",
    "obs_dim",
    "state_dim",
    "noise_dim",
    "zero_pad_dim",
    "reward_state_input_gain_fraction",
    "reward_action_input_gain_fraction",
    "reward_noise_input_gain_fraction",
    "reward_state_to_action_gain_ratio",
    "balanced_reward_state_input_gain_fraction",
    "balanced_reward_action_input_gain_fraction",
    "balanced_reward_noise_input_gain_fraction",
    "balanced_reward_state_to_action_gain_ratio",
    "balance_state_scale",
    "balance_action_scale",
    "balance_noise_scale",
    "locomotion_env_gain_vs_zero",
    "strict_feedback_gain",
    "feedback_near_best_chosen_gap",
    "pendulum_settle_yield_axis_action_gain",
    "pendulum_settle_yield_axis_damping",
    "pendulum_settle_yield_axis_spring",
    "pendulum_settle_yield_axis_position_coupling",
    "h:state_noise_std",
    "h:alpha",
    "h:state_output_scale",
    "h:state_full_rms_target",
    "h:state_clip",
    "h:state_input_scale",
    "h:init_state_std",
    "h:init_action_std",
    "h:init_std",
    "h:noise_std",
    "h:noise",
    "h:reward_scale",
    "h:reward_clip",
    "h:reward_dropout_ratio",
    "h:reinforce_reward_tanh_bound",
    "h:ctrl_reward_weight",
    "h:survival_reward_weight",
    "h:prior_mlp_hidden_dim",
    "h:num_layers",
    "h:gp_rff_features",
    "h:pendulum_settle_yield_action_gain",
    "h:pendulum_settle_yield_damping",
    "h:pendulum_settle_yield_spring",
    "h:pendulum_settle_yield_position_coupling",
)

PROTECTED_STRUCTURE_KEYS = {
    "terminal_reset_count_target",
    "action_dim",
    "obs_dim",
    "state_dim",
    "noise_dim",
    "zero_pad_dim",
    "reward_state_input_gain_fraction",
    "reward_action_input_gain_fraction",
    "reward_noise_input_gain_fraction",
    "reward_state_to_action_gain_ratio",
    "balanced_reward_state_input_gain_fraction",
    "balanced_reward_action_input_gain_fraction",
    "balanced_reward_noise_input_gain_fraction",
    "balanced_reward_state_to_action_gain_ratio",
    "balance_state_scale",
    "balance_action_scale",
    "balance_noise_scale",
}

H_KEYS = tuple(key[2:] for key in STRUCTURE_KEYS if key.startswith("h:"))


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if isinstance(value, Path):
        return str(value)
    return value


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).expanduser().resolve().open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n", "", "none", "nan"}:
        return False
    return bool(value)


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _stats(values: np.ndarray) -> dict[str, Any]:
    finite = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=np.float64)
    if finite.size == 0:
        return {"n": 0}
    return {
        "n": int(finite.size),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "min": float(np.min(finite)),
        "q25": float(np.quantile(finite, 0.25)),
        "q50": float(np.quantile(finite, 0.50)),
        "q75": float(np.quantile(finite, 0.75)),
        "max": float(np.max(finite)),
    }


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    return _corr(_rankdata(x), _rankdata(y))


def _entropy(probs: np.ndarray) -> float:
    probs = probs[probs > 0.0]
    return float(-np.sum(probs * np.log(probs))) if probs.size else 0.0


def _bin_numeric(values: np.ndarray, *, max_bins: int = 4) -> tuple[np.ndarray, list[str]]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros(len(values), dtype=np.int64), ["missing"]
    unique = np.unique(finite)
    if unique.size <= max_bins:
        labels = [f"{value:g}" for value in unique]
        mapping = {float(value): idx for idx, value in enumerate(unique)}
        bins = np.asarray([mapping.get(float(v), -1) if math.isfinite(float(v)) else -1 for v in values])
        return bins, labels
    quantiles = np.unique(np.quantile(finite, np.linspace(0.0, 1.0, max_bins + 1)))
    if quantiles.size <= 2:
        bins = np.zeros(len(values), dtype=np.int64)
        return bins, [f"{finite.min():g}..{finite.max():g}"]
    inner = quantiles[1:-1]
    bins = np.searchsorted(inner, values, side="right")
    bins = np.asarray([int(v) if math.isfinite(float(raw)) else -1 for v, raw in zip(bins, values)])
    labels = []
    lo = quantiles[0]
    for idx, hi in enumerate(quantiles[1:]):
        labels.append(f"{lo:g}..{hi:g}")
        lo = hi
    return bins, labels


def _contingency(binary: np.ndarray, bins: np.ndarray) -> tuple[np.ndarray, list[int]]:
    kept = (bins >= 0) & np.isfinite(binary)
    used_bins = sorted(int(v) for v in np.unique(bins[kept]))
    if not used_bins:
        return np.zeros((2, 0), dtype=np.float64), []
    index = {value: idx for idx, value in enumerate(used_bins)}
    table = np.zeros((2, len(used_bins)), dtype=np.float64)
    for bit, bin_value in zip(binary[kept], bins[kept]):
        table[int(bit > 0.5), index[int(bin_value)]] += 1.0
    return table, used_bins


def _cramers_v(table: np.ndarray) -> float:
    total = float(np.sum(table))
    if total <= 0.0 or table.shape[1] <= 1:
        return 0.0
    row_sum = table.sum(axis=1, keepdims=True)
    col_sum = table.sum(axis=0, keepdims=True)
    expected = row_sum @ col_sum / total
    mask = expected > 0.0
    chi2 = float(np.sum(((table - expected) ** 2)[mask] / expected[mask]))
    denom = total * max(1, min(table.shape[0] - 1, table.shape[1] - 1))
    return float(math.sqrt(chi2 / denom)) if denom > 0.0 else 0.0


def _normalized_mi(table: np.ndarray) -> float:
    total = float(np.sum(table))
    if total <= 0.0 or table.shape[1] <= 1:
        return 0.0
    pxy = table / total
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    denom = px @ py
    mask = (pxy > 0.0) & (denom > 0.0)
    mi = float(np.sum(pxy[mask] * np.log(pxy[mask] / denom[mask])))
    hx = _entropy(px[:, 0])
    hy = _entropy(py[0])
    return float(mi / math.sqrt(hx * hy)) if hx > 0.0 and hy > 0.0 else 0.0


def _bin_max_deviation(table: np.ndarray) -> float:
    total = float(np.sum(table))
    positives = float(np.sum(table[1]))
    if total <= 0.0 or positives <= 0.0:
        return 0.0
    overall = table.sum(axis=0) / total
    cond = table[1] / positives
    return float(np.max(np.abs(cond - overall))) if table.shape[1] else 0.0


def _frozen_h(row: dict[str, Any]) -> dict[str, Any]:
    path = row.get("full_frozen_h_json")
    if not path:
        return {}
    try:
        payload = _read_json(path)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _enrich_row(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    h = _frozen_h(out)
    for key in ("state_dim", "obs_dim", "action_dim", "noise_dim", "zero_pad_dim", "terminal_reset_count_target"):
        if key not in out or str(out.get(key, "")).strip() == "":
            out[key] = h.get(key)
    for key in H_KEYS:
        out[f"h:{key}"] = h.get(key)
    out["pendulum_settle_effective_support"] = bool(
        _safe_bool(out.get("pendulum_short_temporal_settle_guard"))
        or _safe_bool(out.get("pendulum_axis_support"))
        or _safe_bool(out.get("pendulum_settle_yield_repair_planned"))
    )
    return out


def _selector_source_rows(selector_report_path: Path) -> dict[str, list[dict[str, Any]]]:
    report = _read_json(selector_report_path)
    inputs = report["inputs"]
    by_rule = selector._merge_rows(
        selector._read_csv(inputs["selected_csv"]),
        selector._load_locomotion_labels(inputs["locomotion_csv"]),
        selector._load_action_labels(inputs["action_mode_json"]),
        selector._load_strict_feedback_labels(inputs.get("strict_feedback_csv")),
        selector._load_feedback_near_best_labels(inputs.get("feedback_near_best_csv")),
        selector._load_future_basin_cached_labels(inputs.get("future_basin_cached_labels_csv")),
    )
    return {rule: [_enrich_row(row) for row in rows] for rule, rows in by_rule.items()}


def _selected_rows(path: Path) -> dict[str, list[dict[str, Any]]]:
    by_rule: dict[str, list[dict[str, Any]]] = {rule: [] for rule in selector.RULES}
    for row in _read_csv(path):
        rule = str(row.get("rule", ""))
        if rule in by_rule:
            by_rule[rule].append(_enrich_row(row))
    return by_rule


def _pair_audit(
    *,
    scope: str,
    rule: str,
    rows: list[dict[str, Any]],
    mode_key: str,
    param_key: str,
) -> dict[str, Any] | None:
    mode = np.asarray([1.0 if _safe_bool(row.get(mode_key)) else 0.0 for row in rows], dtype=np.float64)
    values = np.asarray([_safe_float(row.get(param_key)) for row in rows], dtype=np.float64)
    finite = np.isfinite(values)
    mode = mode[finite]
    values = values[finite]
    n = int(values.size)
    mode_count = int(np.sum(mode > 0.5))
    if n < 8 or mode_count < 2 or n - mode_count < 2:
        return None
    in_values = values[mode > 0.5]
    out_values = values[mode <= 0.5]
    pooled_std = float(np.std(values))
    smd = 0.0 if pooled_std <= 1e-12 else float((np.mean(in_values) - np.mean(out_values)) / pooled_std)
    bins, labels = _bin_numeric(values)
    table, used_bins = _contingency(mode, bins)
    cramer_v = _cramers_v(table)
    nmi = _normalized_mi(table)
    bin_dev = _bin_max_deviation(table)
    pearson = _corr(mode, values)
    spearman = _spearman(mode, values)
    abs_pressure = max(abs(spearman), cramer_v, abs(smd) / 1.6, bin_dev)
    pressure = "low"
    if (
        abs(spearman) >= 0.50
        or cramer_v >= 0.50
        or abs(smd) >= 1.00
        or bin_dev >= 0.45
        or nmi >= 0.30
    ):
        pressure = "high"
    elif (
        abs(spearman) >= 0.35
        or cramer_v >= 0.35
        or abs(smd) >= 0.65
        or bin_dev >= 0.30
        or nmi >= 0.18
    ):
        pressure = "medium"
    return {
        "scope": scope,
        "rule": rule,
        "mode": mode_key,
        "parameter": param_key,
        "n": n,
        "mode_count": mode_count,
        "mode_rate": float(mode_count / max(1, n)),
        "overall_mean": float(np.mean(values)),
        "mode_true_mean": float(np.mean(in_values)),
        "mode_false_mean": float(np.mean(out_values)),
        "standardized_mean_diff": smd,
        "pearson": pearson,
        "spearman": spearman,
        "cramers_v": cramer_v,
        "normalized_mi": nmi,
        "bin_max_deviation": bin_dev,
        "pressure_score": abs_pressure,
        "pressure": pressure,
        "bin_labels": [labels[idx] for idx in used_bins if idx < len(labels)],
        "contingency": table.astype(int).tolist(),
    }


def _marginal_summary(scope: str, rule: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"scope": scope, "rule": rule, "n": len(rows), "modes": {}, "parameters": {}}
    for mode_key in MODE_KEYS:
        count = int(sum(_safe_bool(row.get(mode_key)) for row in rows))
        out["modes"][mode_key] = {"count": count, "rate": count / max(1, len(rows))}
    for param_key in STRUCTURE_KEYS:
        values = np.asarray([_safe_float(row.get(param_key)) for row in rows], dtype=np.float64)
        out["parameters"][param_key] = _stats(values)
    return out


def _load_trainability_context(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"available": False}
    report = _read_json(path)
    summaries = report.get("summaries", {}) if isinstance(report, dict) else {}
    keep: dict[str, Any] = {"available": True}
    for key in ("gym", "prior_full", "prior_q90"):
        item = summaries.get(key, {})
        keep[key] = {
            "n": item.get("n"),
            "improved_count": item.get("improved_count"),
            "delta_mean": item.get("delta_mean"),
            "low_mid_high": [
                item.get("low_complex_count"),
                item.get("mid_complex_count"),
                item.get("high_complex_count"),
            ],
            "improved_low_mid_high": [
                item.get("improved_low_complex_count"),
                item.get("improved_mid_complex_count"),
                item.get("improved_high_complex_count"),
            ],
            "ks_compatible_with_gym": item.get("ks_compatible_with_gym"),
            "mean_compatible_with_gym": item.get("mean_compatible_with_gym"),
        }
    return keep


def _audit(args: argparse.Namespace) -> dict[str, Any]:
    selector_report = Path(args.selector_report).expanduser().resolve()
    selected_csv = Path(args.selected_csv).expanduser().resolve()
    source_by_rule = _selector_source_rows(selector_report)
    selected_by_rule = _selected_rows(selected_csv)
    scopes = {"source_pool": source_by_rule, "selected": selected_by_rule}

    pair_rows: list[dict[str, Any]] = []
    marginals: list[dict[str, Any]] = []
    for scope, by_rule in scopes.items():
        for rule, rows in by_rule.items():
            marginals.append(_marginal_summary(scope, rule, rows))
            for mode_key in MODE_KEYS:
                for param_key in STRUCTURE_KEYS:
                    item = _pair_audit(
                        scope=scope,
                        rule=rule,
                        rows=rows,
                        mode_key=mode_key,
                        param_key=param_key,
                    )
                    if item is not None:
                        pair_rows.append(item)

    high = [row for row in pair_rows if row["pressure"] == "high"]
    medium = [row for row in pair_rows if row["pressure"] == "medium"]
    protected_high = [row for row in high if row["parameter"] in PROTECTED_STRUCTURE_KEYS]
    protected_medium = [row for row in medium if row["parameter"] in PROTECTED_STRUCTURE_KEYS]
    high_by_scope_rule: dict[str, int] = Counter(f"{row['scope']}::{row['rule']}" for row in high)
    medium_by_scope_rule: dict[str, int] = Counter(f"{row['scope']}::{row['rule']}" for row in medium)
    protected_high_by_scope_rule: dict[str, int] = Counter(
        f"{row['scope']}::{row['rule']}" for row in protected_high
    )
    protected_medium_by_scope_rule: dict[str, int] = Counter(
        f"{row['scope']}::{row['rule']}" for row in protected_medium
    )

    report = {
        "analysis_entry": "phase2_profile_mode_structure_independence_audit",
        "contract": {
            "read_only": True,
            "does_not_sample_envs": True,
            "does_not_train": True,
            "does_not_modify_selector": True,
            "does_not_modify_fit_model": True,
            "does_not_modify_ppo": True,
            "source_pool_reconstructed_with_selector_label_joins": True,
            "selected_set_checked_after_materialization": True,
            "trainability_context_read_only": True,
        },
        "thresholds": {
            "high": {
                "abs_spearman": 0.50,
                "cramers_v": 0.50,
                "abs_standardized_mean_diff": 1.00,
                "bin_max_deviation": 0.45,
                "normalized_mi": 0.30,
            },
            "medium": {
                "abs_spearman": 0.35,
                "cramers_v": 0.35,
                "abs_standardized_mean_diff": 0.65,
                "bin_max_deviation": 0.30,
                "normalized_mi": 0.18,
            },
        },
        "inputs": {
            "selector_report": str(selector_report),
            "selected_csv": str(selected_csv),
            "relative_audit_json": str(Path(args.relative_audit_json).expanduser().resolve()),
        },
        "marginal_summaries": marginals,
        "pair_audit": pair_rows,
        "summary": {
            "pair_count": len(pair_rows),
            "high_pressure_count": len(high),
            "medium_pressure_count": len(medium),
            "protected_structure_high_pressure_count": len(protected_high),
            "protected_structure_medium_pressure_count": len(protected_medium),
            "high_by_scope_rule": dict(sorted(high_by_scope_rule.items())),
            "medium_by_scope_rule": dict(sorted(medium_by_scope_rule.items())),
            "protected_structure_high_by_scope_rule": dict(
                sorted(protected_high_by_scope_rule.items())
            ),
            "protected_structure_medium_by_scope_rule": dict(
                sorted(protected_medium_by_scope_rule.items())
            ),
            "top_high_pressure": sorted(
                high,
                key=lambda row: (
                    -float(row["pressure_score"]),
                    row["scope"],
                    row["rule"],
                    row["mode"],
                    row["parameter"],
                ),
            )[:40],
            "top_medium_pressure": sorted(
                medium,
                key=lambda row: (
                    -float(row["pressure_score"]),
                    row["scope"],
                    row["rule"],
                    row["mode"],
                    row["parameter"],
                ),
            )[:40],
            "top_protected_structure_high_pressure": sorted(
                protected_high,
                key=lambda row: (
                    -float(row["pressure_score"]),
                    row["scope"],
                    row["rule"],
                    row["mode"],
                    row["parameter"],
                ),
            )[:40],
            "top_protected_structure_medium_pressure": sorted(
                protected_medium,
                key=lambda row: (
                    -float(row["pressure_score"]),
                    row["scope"],
                    row["rule"],
                    row["mode"],
                    row["parameter"],
                ),
            )[:40],
            "hard_pass": len(high) == 0,
            "protected_structure_hard_pass": len(protected_high) == 0,
        },
        "trainability_context": _load_trainability_context(
            Path(args.relative_audit_json).expanduser().resolve()
        ),
    }
    return report


def _markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    train = report.get("trainability_context", {})
    lines = [
        "# Profile Mode/Structure Independence Audit",
        "",
        "Read-only audit for hidden coupling between profile modes and structural parameters.",
        "",
        "## Read",
        "",
        f"- hard pass: `{summary['hard_pass']}`",
        f"- protected-structure hard pass: `{summary['protected_structure_hard_pass']}`",
        f"- high-pressure pairs: `{summary['high_pressure_count']}`",
        f"- medium-pressure pairs: `{summary['medium_pressure_count']}`",
        f"- protected-structure high-pressure pairs: `{summary['protected_structure_high_pressure_count']}`",
        f"- protected-structure medium-pressure pairs: `{summary['protected_structure_medium_pressure_count']}`",
        f"- pair tests: `{summary['pair_count']}`",
        "",
        "A high-pressure pair means a mode is strongly predictable from a structural parameter "
        "or strongly shifts that parameter. Such pairs are not automatically a semantic bug, but "
        "they are not acceptable as hidden bindings for a Gym-compatible profile unless explicitly defined.",
        "",
        "## Trainability Context",
        "",
    ]
    if train.get("available"):
        lines.extend(
            [
                "| run | improved | delta mean | low/mid/high | improved low/mid/high |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for key in ("gym", "prior_full", "prior_q90"):
            item = train.get(key, {})
            lines.append(
                f"| `{key}` | {item.get('improved_count')}/{item.get('n')} | "
                f"{float(item.get('delta_mean') or 0.0):+.2f} | "
                f"{item.get('low_mid_high')} | {item.get('improved_low_mid_high')} |"
            )
    else:
        lines.append("- relative-complexity trainability context not available")

    lines.extend(["", "## Protected Structure High-Pressure Pairs", ""])
    protected_high = summary["top_protected_structure_high_pressure"]
    if not protected_high:
        lines.append("- none")
    else:
        lines.extend(
            [
                "| scope | rule | mode | parameter | score | spearman | Cramer's V | SMD | bin dev |",
                "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in protected_high[:30]:
            lines.append(
                f"| `{row['scope']}` | `{row['rule']}` | `{row['mode']}` | `{row['parameter']}` | "
                f"{float(row['pressure_score']):.3f} | {float(row['spearman']):+.3f} | "
                f"{float(row['cramers_v']):.3f} | {float(row['standardized_mean_diff']):+.3f} | "
                f"{float(row['bin_max_deviation']):.3f} |"
            )

    lines.extend(["", "## High-Pressure Pairs", ""])
    top_high = summary["top_high_pressure"]
    if not top_high:
        lines.append("- none")
    else:
        lines.extend(
            [
                "| scope | rule | mode | parameter | score | spearman | Cramer's V | SMD | bin dev |",
                "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in top_high[:30]:
            lines.append(
                f"| `{row['scope']}` | `{row['rule']}` | `{row['mode']}` | `{row['parameter']}` | "
                f"{float(row['pressure_score']):.3f} | {float(row['spearman']):+.3f} | "
                f"{float(row['cramers_v']):.3f} | {float(row['standardized_mean_diff']):+.3f} | "
                f"{float(row['bin_max_deviation']):.3f} |"
            )

    lines.extend(["", "## Marginal Structure Summary", ""])
    lines.extend(
        [
            "| scope | rule | n | terminal mean | obs dim q50 | action dim q50 | state dim q50 |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in report["marginal_summaries"]:
        params = item["parameters"]
        terminal = params.get("terminal_reset_count_target", {})
        obs = params.get("obs_dim", {})
        action = params.get("action_dim", {})
        state = params.get("state_dim", {})
        lines.append(
            f"| `{item['scope']}` | `{item['rule']}` | {item['n']} | "
            f"{float(terminal.get('mean') or 0.0):.2f} | {float(obs.get('q50') or 0.0):.1f} | "
            f"{float(action.get('q50') or 0.0):.1f} | {float(state.get('q50') or 0.0):.1f} |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- JSON: `{report['outputs']['json']}`",
            f"- pair CSV: `{report['outputs']['pair_csv']}`",
            f"- Markdown: `{report['outputs']['markdown']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report = _audit(args)
    json_path = out_dir / "profile_mode_structure_independence_audit.json"
    pair_csv = out_dir / "mode_structure_pair_audit.csv"
    md_path = out_dir / "profile_mode_structure_independence_audit.md"
    report["outputs"] = {
        "json": str(json_path),
        "pair_csv": str(pair_csv),
        "markdown": str(md_path),
    }
    json_path.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    flat_rows = []
    for row in report["pair_audit"]:
        flat = dict(row)
        flat["bin_labels"] = json.dumps(row["bin_labels"], sort_keys=True)
        flat["contingency"] = json.dumps(row["contingency"], sort_keys=True)
        flat_rows.append(flat)
    _write_csv(pair_csv, flat_rows)
    md_path.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report["outputs"], sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selector-report", default=str(DEFAULT_SELECTOR_REPORT))
    parser.add_argument("--selected-csv", default=str(DEFAULT_SELECTED_CSV))
    parser.add_argument("--relative-audit-json", default=str(DEFAULT_RELATIVE_AUDIT_JSON))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
