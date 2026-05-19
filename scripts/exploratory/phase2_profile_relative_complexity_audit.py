#!/usr/bin/env python
"""Relative Gym/prior complexity audit for profile sidecar runs.

This is a read-only diagnostic.  It scores every environment in a common
behavioral feature space calibrated on Gym, then checks whether prior packs
contain both simpler-than-Gym and more-complex-than-Gym samples without
collapsing the trainable/improved subset into one tail.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from scripts.exploratory.phase2_profile_spectral_rank_audit import (
    DEFAULT_GYM_DIR,
    DEFAULT_PRIOR_FULL_DIR,
    DEFAULT_PRIOR_Q90_DIR,
    DEFAULT_OUTPUT_DIR as DEFAULT_SPECTRAL_OUTPUT_DIR,
    _common_matrix,
    _delta_metric,
    _json_safe,
    _read_jsonl,
    _sidecar_path,
)


DEFAULT_OUTPUT_DIR = DEFAULT_SPECTRAL_OUTPUT_DIR.parent / "phase2_profile_v2n_relative_complexity_audit_0515"

MIN_TAIL_FRACTION = 0.125
MAX_MEAN_PERCENTILE_GAP = 0.12
MAX_KS_DISTANCE = 0.35
MIN_HIGH_COMPLEX_IMPROVED_RETENTION = 0.75

SCORE_FAMILIES = {
    "in_gym_complexity": ["z_norm_energy", "pca95_leverage"],
    "off_gym_novelty": ["nearest_gym_distance", "residual_energy", "residual_ratio"],
    "full_complexity_compatibility": [
        "nearest_gym_distance",
        "pca95_leverage",
        "residual_energy",
        "residual_ratio",
        "z_norm_energy",
    ],
}

FEATURE_SUBSET_DEFINITIONS = {
    "fair_observed_behavior": (
        "Observed obs/action/reward/done sidecar features only. Excludes next_state_targets "
        "because the runner defines Gym next_state as next obs and prior next_state as latent state target."
    ),
    "dimension_complexity": "Action/observation dimension support; a complexity axis, not a behavior mismatch by itself.",
    "latent_next_state_target_drift": (
        "next_state target drift diagnostic. Reported separately because Gym/prior next_state spaces are not directly isomorphic."
    ),
    "transition_ratio_only": "The prior/Gym state-step ratio pressure point isolated from other features.",
}


def _env_indices(rows0: list[dict[str, Any]], rows3: list[dict[str, Any]]) -> list[int]:
    by0 = {int(row["env_idx"]): row for row in rows0}
    by3 = {int(row["env_idx"]): row for row in rows3}
    return sorted(set(by0) & set(by3))


def _load_run(run_dir: Path, *, arm: str) -> dict[str, Any]:
    rows0 = _read_jsonl(_sidecar_path(run_dir, 0))
    rows3 = _read_jsonl(_sidecar_path(run_dir, 3))
    matrix, names = _common_matrix(rows0, rows3)
    deltas = _delta_metric(rows0, rows3, arm=arm)
    return {
        "run_dir": str(run_dir),
        "arm": arm,
        "env_indices": _env_indices(rows0, rows3),
        "matrix": matrix,
        "names": names,
        "deltas": deltas,
        "improved": deltas > 0.0,
    }


def _align_common_features(runs: dict[str, dict[str, Any]]) -> list[str]:
    common = set(runs["gym"]["names"])
    for item in runs.values():
        common &= set(item["names"])
    gym_order = [name for name in runs["gym"]["names"] if name in common]
    if not gym_order:
        raise RuntimeError("no common behavior features are shared across Gym/prior runs")
    return gym_order


def _select_columns(run: dict[str, Any], common_names: list[str]) -> np.ndarray:
    index = {name: idx for idx, name in enumerate(run["names"])}
    return np.asarray(run["matrix"][:, [index[name] for name in common_names]], dtype=np.float64)


def _fit_gym_space(gym_matrix: np.ndarray, names: list[str]) -> dict[str, Any]:
    finite_cols = np.all(np.isfinite(gym_matrix), axis=0)
    x = gym_matrix[:, finite_cols]
    kept_names = [name for name, keep in zip(names, finite_cols) if keep]
    mean = np.mean(x, axis=0, keepdims=True)
    std = np.std(x, axis=0, keepdims=True)
    variable = std[0] > 1e-12
    x = x[:, variable]
    kept_names = [name for name, keep in zip(kept_names, variable) if keep]
    mean = mean[:, variable]
    std = std[:, variable]
    if x.shape[1] == 0:
        raise RuntimeError("Gym common behavior matrix has no variable finite features")
    z = (x - mean) / std
    _, singular_values, vh = np.linalg.svd(z, full_matrices=False)
    energy = singular_values**2
    total = float(np.sum(energy))
    weights = energy / total if total > 0.0 else np.zeros_like(energy)
    cume = np.cumsum(weights)
    rank95 = int(np.searchsorted(cume, 0.95) + 1) if len(cume) else 1
    rank95 = max(1, min(rank95, vh.shape[0]))
    return {
        "kept_names": kept_names,
        "finite_cols": finite_cols,
        "variable_cols_after_finite": variable,
        "mean": mean,
        "std": std,
        "basis95": vh[:rank95],
        "rank95": rank95,
        "gym_z": z,
        "singular_energy_top10": [float(v) for v in weights[:10]],
    }


def _transform(matrix: np.ndarray, space: dict[str, Any]) -> np.ndarray:
    x = matrix[:, space["finite_cols"]]
    x = x[:, space["variable_cols_after_finite"]]
    return (x - space["mean"]) / space["std"]


def _leave_one_out_nearest(z: np.ndarray) -> np.ndarray:
    out = np.empty(z.shape[0], dtype=np.float64)
    for idx in range(z.shape[0]):
        diff = z[idx : idx + 1] - np.delete(z, idx, axis=0)
        out[idx] = float(np.min(np.sqrt(np.mean(diff**2, axis=1)))) if len(diff) else 0.0
    return out


def _nearest_to_gym(z: np.ndarray, gym_z: np.ndarray) -> np.ndarray:
    out = np.empty(z.shape[0], dtype=np.float64)
    for idx in range(z.shape[0]):
        diff = z[idx : idx + 1] - gym_z
        out[idx] = float(np.min(np.sqrt(np.mean(diff**2, axis=1))))
    return out


def _metrics(z: np.ndarray, space: dict[str, Any], *, reference: bool) -> dict[str, np.ndarray]:
    basis = space["basis95"]
    coords = z @ basis.T
    projected = coords @ basis
    residual = z - projected
    norm_energy = np.mean(z**2, axis=1)
    pca_leverage = np.mean(coords**2, axis=1)
    residual_energy = np.mean(residual**2, axis=1)
    residual_ratio = residual_energy / np.maximum(norm_energy, 1e-12)
    nearest = _leave_one_out_nearest(z) if reference else _nearest_to_gym(z, space["gym_z"])
    return {
        "z_norm_energy": norm_energy,
        "pca95_leverage": pca_leverage,
        "residual_energy": residual_energy,
        "residual_ratio": residual_ratio,
        "nearest_gym_distance": nearest,
    }


def _percentile_against(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    ref = np.sort(np.asarray(reference, dtype=np.float64))
    out = np.empty(len(values), dtype=np.float64)
    for idx, value in enumerate(values):
        lo = np.searchsorted(ref, value, side="left")
        hi = np.searchsorted(ref, value, side="right")
        out[idx] = (lo + 0.5 * (hi - lo)) / max(1, len(ref))
    return out


def _score_metrics(
    metrics: dict[str, np.ndarray],
    gym_metrics: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    component_scores = {
        key: _percentile_against(gym_metrics[key], values) for key, values in metrics.items()
    }
    composite = np.mean(np.vstack([component_scores[key] for key in sorted(component_scores)]), axis=0)
    return composite, component_scores


def _value_summary(values: np.ndarray) -> dict[str, Any]:
    finite = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=np.float64)
    if finite.size == 0:
        return {"n": 0}
    return {
        "n": int(finite.size),
        "mean": float(np.mean(finite)),
        "q25": float(np.quantile(finite, 0.25)),
        "q50": float(np.quantile(finite, 0.50)),
        "q75": float(np.quantile(finite, 0.75)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def _off_gym_score_against_train(
    *,
    train_metrics: dict[str, np.ndarray],
    target_metrics: dict[str, np.ndarray],
) -> np.ndarray:
    keys = ("nearest_gym_distance", "residual_energy", "residual_ratio")
    return np.mean(
        np.vstack([_percentile_against(train_metrics[key], target_metrics[key]) for key in keys]),
        axis=0,
    )


def _split_off_gym_generalization_summary(
    *,
    gym_matrix: np.ndarray,
    target_matrices: dict[str, np.ndarray],
    names: list[str],
    n_splits: int = 20,
) -> dict[str, Any]:
    """Separate reference sparsity from true prior PCA-residual excess."""

    n_gym = int(gym_matrix.shape[0])
    if n_gym < 8:
        return {"error": "not enough Gym rows for split diagnostic"}

    rng = np.random.default_rng(9730)
    split_rows: list[dict[str, Any]] = []
    for _ in range(int(n_splits)):
        perm = rng.permutation(n_gym)
        train = gym_matrix[perm[: n_gym // 2]]
        heldout = gym_matrix[perm[n_gym // 2 :]]
        try:
            split_space = _fit_gym_space(train, names)
        except Exception:
            continue
        z_train = _transform(train, split_space)
        train_metrics = _metrics(z_train, split_space, reference=True)
        train_score = _off_gym_score_against_train(
            train_metrics=train_metrics,
            target_metrics=train_metrics,
        )
        train_q75 = float(np.quantile(train_score, 0.75))
        row: dict[str, Any] = {}

        z_heldout = _transform(heldout, split_space)
        heldout_metrics = _metrics(z_heldout, split_space, reference=False)
        heldout_score = _off_gym_score_against_train(
            train_metrics=train_metrics,
            target_metrics=heldout_metrics,
        )
        row["heldout_gym_off_mean"] = float(np.mean(heldout_score))
        row["heldout_gym_high_frac_vs_train_q75"] = float(np.mean(heldout_score >= train_q75))
        for key in ("nearest_gym_distance", "residual_energy", "residual_ratio"):
            row[f"heldout_gym_{key}_median_ratio_vs_train"] = float(
                np.median(heldout_metrics[key]) / max(float(np.median(train_metrics[key])), 1e-12)
            )

        for label, matrix in target_matrices.items():
            z_target = _transform(matrix, split_space)
            target_metrics = _metrics(z_target, split_space, reference=False)
            target_score = _off_gym_score_against_train(
                train_metrics=train_metrics,
                target_metrics=target_metrics,
            )
            row[f"{label}_off_mean"] = float(np.mean(target_score))
            row[f"{label}_high_frac_vs_train_q75"] = float(np.mean(target_score >= train_q75))
            for key in ("nearest_gym_distance", "residual_energy", "residual_ratio"):
                row[f"{label}_{key}_median_ratio_vs_train"] = float(
                    np.median(target_metrics[key])
                    / max(float(np.median(train_metrics[key])), 1e-12)
                )
        split_rows.append(row)

    if not split_rows:
        return {"error": "all Gym split diagnostics failed"}

    def _collect(key: str) -> np.ndarray:
        return np.asarray([float(row[key]) for row in split_rows], dtype=np.float64)

    heldout_high = _collect("heldout_gym_high_frac_vs_train_q75")
    target_reports: dict[str, Any] = {}
    for label in sorted(target_matrices):
        residual_energy_excess = _collect(
            f"{label}_residual_energy_median_ratio_vs_train"
        ) / np.maximum(_collect("heldout_gym_residual_energy_median_ratio_vs_train"), 1e-12)
        residual_ratio_excess = _collect(
            f"{label}_residual_ratio_median_ratio_vs_train"
        ) / np.maximum(_collect("heldout_gym_residual_ratio_median_ratio_vs_train"), 1e-12)
        nearest_excess = _collect(
            f"{label}_nearest_gym_distance_median_ratio_vs_train"
        ) / np.maximum(_collect("heldout_gym_nearest_gym_distance_median_ratio_vs_train"), 1e-12)
        residual_geomean = np.sqrt(
            np.maximum(residual_energy_excess, 1e-12) * np.maximum(residual_ratio_excess, 1e-12)
        )
        target_reports[label] = {
            "nearest_gym_distance_median_excess": _value_summary(nearest_excess),
            "residual_energy_median_excess": _value_summary(residual_energy_excess),
            "residual_ratio_median_excess": _value_summary(residual_ratio_excess),
            "residual_energy_ratio_geomean_excess": _value_summary(residual_geomean),
            "primary_true_prior_excess": "residual_energy_ratio_geomean_excess",
            "primary_true_prior_excess_q50": float(np.quantile(residual_geomean, 0.50)),
            "nearest_distance_excess_q50": float(np.quantile(nearest_excess, 0.50)),
        }

    return {
        "interpretation": (
            "Held-out Gym split diagnostics prevent the raw off-gym percentile gate from "
            "being treated as a sole hard target when the Gym reference itself is sparse. "
            "Values above 1 in target_vs_heldout_excess are true excess beyond held-out Gym."
        ),
        "n_splits": int(len(split_rows)),
        "split_fraction": "half_train_half_heldout",
        "raw_off_gym_gate_is_over_narrow_for_small_gym_reference": bool(
            float(np.mean(heldout_high)) >= 0.50
        ),
        "heldout_gym_high_frac_vs_train_q75": _value_summary(heldout_high),
        "target_vs_heldout_excess": target_reports,
    }


def _ks_distance(reference: np.ndarray, values: np.ndarray) -> float:
    ref = np.sort(np.asarray(reference, dtype=np.float64))
    val = np.sort(np.asarray(values, dtype=np.float64))
    points = np.sort(np.unique(np.concatenate([ref, val])))
    if len(points) == 0:
        return 0.0
    ref_cdf = np.searchsorted(ref, points, side="right") / max(1, len(ref))
    val_cdf = np.searchsorted(val, points, side="right") / max(1, len(val))
    return float(np.max(np.abs(ref_cdf - val_cdf)))


def _wasserstein_1d(reference: np.ndarray, values: np.ndarray) -> float:
    ref = np.sort(np.asarray(reference, dtype=np.float64))
    val = np.sort(np.asarray(values, dtype=np.float64))
    if len(ref) == 0 or len(val) == 0:
        return 0.0
    qs = (np.arange(max(len(ref), len(val))) + 0.5) / max(len(ref), len(val))
    ref_q = np.quantile(ref, qs)
    val_q = np.quantile(val, qs)
    return float(np.mean(np.abs(ref_q - val_q)))


def _feature_category(name: str) -> str:
    prefix = "delta" if name.startswith("d:") else "u0"
    base = name.split(":", 1)[1] if ":" in name else name
    if "reward" in base:
        kind = "reward"
    elif "action" in base or "policy" in base:
        kind = "action"
    elif "obs" in base or "state" in base:
        kind = "state_obs"
    elif "done" in base or "truncated" in base:
        kind = "done"
    else:
        kind = "other"
    return f"{prefix}:{kind}"


def _feature_subset_keep(name: str, subset: str) -> bool:
    base = name.split(":", 1)[1] if ":" in name else name
    if subset == "fair_observed_behavior":
        if base.startswith("next_state"):
            return False
        if base in {"action_dim", "obs_dim", "state_dim"}:
            return False
        if base in {
            "action_norm_mean",
            "action_norm_std",
            "unit_clipped_action_norm_mean",
            "unit_clipped_action_norm_std",
            "action_temporal_delta_norm_mean",
        }:
            return False
        return any(
            token in base
            for token in (
                "obs_",
                "reward",
                "done",
                "truncated",
                "first_done",
                "action_value",
                "action_per_dim",
                "raw_policy_action_unit_clip",
                "unit_clipped_action_value",
            )
        )
    if subset == "dimension_complexity":
        return name in {"u0:action_dim", "u0:obs_dim"}
    if subset == "latent_next_state_target_drift":
        return base.startswith("next_state")
    if subset == "transition_ratio_only":
        return base == "next_state_step_l2_delta_to_rms_ratio"
    raise KeyError(f"unknown feature subset: {subset}")


def _feature_shift_summary(
    kept_names: list[str],
    transformed: dict[str, np.ndarray],
) -> dict[str, Any]:
    gym = transformed["gym"]
    out: dict[str, Any] = {}
    for run in ["prior_full", "prior_q90"]:
        z = transformed[run]
        mean_shift = np.mean(z, axis=0) - np.mean(gym, axis=0)
        median_shift = np.median(z, axis=0) - np.median(gym, axis=0)
        std_ratio = np.std(z, axis=0) / np.maximum(np.std(gym, axis=0), 1e-12)
        rows = []
        category_values: dict[str, list[float]] = {}
        for name, mean, median, ratio in zip(kept_names, mean_shift, median_shift, std_ratio):
            abs_mean = abs(float(mean))
            rows.append(
                {
                    "feature": name,
                    "mean_z_shift": float(mean),
                    "median_z_shift": float(median),
                    "std_ratio": float(ratio),
                    "abs_mean_z_shift": abs_mean,
                }
            )
            category_values.setdefault(_feature_category(name), []).append(abs_mean)
        rows.sort(key=lambda row: row["abs_mean_z_shift"], reverse=True)
        out[run] = {
            "top_abs_mean_z_shift": rows[:20],
            "category_abs_mean_shift": {
                key: {
                    "mean_abs_shift": float(np.mean(values)),
                    "max_abs_shift": float(np.max(values)),
                    "feature_count": int(len(values)),
                }
                for key, values in sorted(category_values.items())
            },
        }
    return out


def _residual_feature_contribution(
    kept_names: list[str],
    transformed: dict[str, np.ndarray],
    space: dict[str, Any],
) -> dict[str, Any]:
    """Rank per-feature pressure outside Gym's PCA95 subspace."""

    basis = space["basis95"]
    gym_z = transformed["gym"]
    gym_residual = gym_z - (gym_z @ basis.T) @ basis
    gym_mean_sq = np.mean(gym_residual**2, axis=0)
    out: dict[str, Any] = {}
    for run in ["prior_full", "prior_q90"]:
        z = transformed[run]
        residual = z - (z @ basis.T) @ basis
        run_mean_sq = np.mean(residual**2, axis=0)
        rows = []
        category_values: dict[str, list[float]] = {}
        for name, run_value, gym_value in zip(kept_names, run_mean_sq, gym_mean_sq):
            excess = float(run_value - gym_value)
            rows.append(
                {
                    "feature": name,
                    "run_residual_mean_sq": float(run_value),
                    "gym_residual_mean_sq": float(gym_value),
                    "excess_residual_mean_sq": excess,
                }
            )
            category_values.setdefault(_feature_category(name), []).append(max(0.0, excess))
        rows.sort(key=lambda row: row["excess_residual_mean_sq"], reverse=True)
        out[run] = {
            "top_excess_residual_mean_sq": rows[:20],
            "category_excess_residual_mean_sq": {
                key: {
                    "mean_positive_excess": float(np.mean(values)),
                    "max_positive_excess": float(np.max(values)),
                    "feature_count": int(len(values)),
                }
                for key, values in sorted(category_values.items())
            },
        }
    return out


def _subset_complexity_summaries(
    *,
    subset: str,
    common_names: list[str],
    runs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    subset_names = [name for name in common_names if _feature_subset_keep(name, subset)]
    if not subset_names:
        return {
            "description": FEATURE_SUBSET_DEFINITIONS[subset],
            "feature_count": 0,
            "error": "no selected features",
        }
    aligned = {name: _select_columns(item, subset_names) for name, item in runs.items()}
    try:
        space = _fit_gym_space(aligned["gym"], subset_names)
    except Exception as exc:
        return {
            "description": FEATURE_SUBSET_DEFINITIONS[subset],
            "feature_count": int(len(subset_names)),
            "error": str(exc),
        }
    transformed = {name: _transform(matrix, space) for name, matrix in aligned.items()}
    metric_map = {
        name: _metrics(z, space, reference=(name == "gym")) for name, z in transformed.items()
    }
    scores: dict[str, np.ndarray] = {}
    component_scores: dict[str, dict[str, np.ndarray]] = {}
    for name, metrics in metric_map.items():
        score, components = _score_metrics(metrics, metric_map["gym"])
        scores[name] = score
        component_scores[name] = components

    gym_scores = scores["gym"]
    gym_improved = runs["gym"]["improved"]
    q25, q75 = [float(v) for v in np.quantile(gym_scores, [0.25, 0.75])]
    gym_improved_high_count = int(np.sum(gym_improved & (gym_scores >= q75)))
    summaries = {
        name: _summary(
            scores=scores[name],
            improved=runs[name]["improved"],
            deltas=runs[name]["deltas"],
            gym_scores=gym_scores,
            gym_improved=gym_improved,
            gym_mean=float(np.mean(gym_scores)),
            gym_std=float(np.std(gym_scores)),
            q25=q25,
            q75=q75,
            gym_improved_high_count=gym_improved_high_count,
        )
        for name in runs
    }
    gate: dict[str, Any] = {}
    for name in ["prior_full", "prior_q90"]:
        item = summaries[name]
        gate[name] = {
            "pass": bool(
                item["has_simple_and_complex_tails"]
                and item["mean_compatible_with_gym"]
                and item["ks_compatible_with_gym"]
                and item["high_complex_improved_retained"]
            ),
            "has_simple_and_complex_tails": item["has_simple_and_complex_tails"],
            "mean_compatible_with_gym": item["mean_compatible_with_gym"],
            "ks_compatible_with_gym": item["ks_compatible_with_gym"],
            "high_complex_improved_retained": item["high_complex_improved_retained"],
        }
    component_means = {
        run_name: {
            key: float(np.mean(values))
            for key, values in sorted(component_scores[run_name].items())
        }
        for run_name in runs
    }
    return {
        "description": FEATURE_SUBSET_DEFINITIONS[subset],
        "feature_count": int(len(subset_names)),
        "kept_feature_count": int(len(space["kept_names"])),
        "rank95": int(space["rank95"]),
        "gym_q25": q25,
        "gym_q75": q75,
        "summaries": summaries,
        "gate": gate,
        "component_means": component_means,
        "feature_shift_summary": _feature_shift_summary(space["kept_names"], transformed),
        "residual_feature_contribution": _residual_feature_contribution(
            space["kept_names"], transformed, space
        ),
    }


def _summary(
    *,
    scores: np.ndarray,
    improved: np.ndarray,
    deltas: np.ndarray,
    gym_scores: np.ndarray,
    gym_improved: np.ndarray,
    gym_mean: float,
    gym_std: float,
    q25: float,
    q75: float,
    gym_improved_high_count: int,
) -> dict[str, Any]:
    low = scores <= q25
    high = scores >= q75
    improved_low = improved & low
    improved_high = improved & high
    low_fraction = float(np.mean(low))
    high_fraction = float(np.mean(high))
    tail_balance = float(min(low_fraction, high_fraction) / max(low_fraction, high_fraction, 1e-12))
    mean_score = float(np.mean(scores))
    mean_gap = mean_score - gym_mean
    mean_z_gap = float(mean_gap / max(gym_std, 1e-12))
    improved_mean = float(np.mean(scores[improved])) if np.any(improved) else 0.0
    gym_improved_mean = float(np.mean(gym_scores[gym_improved])) if np.any(gym_improved) else 0.0
    high_retention_needed = int(math.ceil(MIN_HIGH_COMPLEX_IMPROVED_RETENTION * gym_improved_high_count))
    return {
        "n": int(len(scores)),
        "improved_count": int(np.sum(improved)),
        "delta_mean": float(np.mean(deltas)),
        "complexity_mean": mean_score,
        "complexity_median": float(np.median(scores)),
        "complexity_q10": float(np.quantile(scores, 0.10)),
        "complexity_q25": float(np.quantile(scores, 0.25)),
        "complexity_q75": float(np.quantile(scores, 0.75)),
        "complexity_q90": float(np.quantile(scores, 0.90)),
        "mean_gap_vs_gym": mean_gap,
        "mean_z_gap_vs_gym": mean_z_gap,
        "ks_vs_gym": _ks_distance(gym_scores, scores),
        "wasserstein_vs_gym": _wasserstein_1d(gym_scores, scores),
        "low_complex_count": int(np.sum(low)),
        "mid_complex_count": int(np.sum(~low & ~high)),
        "high_complex_count": int(np.sum(high)),
        "low_complex_fraction": low_fraction,
        "high_complex_fraction": high_fraction,
        "tail_balance": tail_balance,
        "improved_low_complex_count": int(np.sum(improved_low)),
        "improved_mid_complex_count": int(np.sum(improved & ~low & ~high)),
        "improved_high_complex_count": int(np.sum(improved_high)),
        "improved_complexity_mean": improved_mean,
        "improved_mean_gap_vs_gym_improved": improved_mean - gym_improved_mean,
        "has_simple_and_complex_tails": bool(
            low_fraction >= MIN_TAIL_FRACTION and high_fraction >= MIN_TAIL_FRACTION
        ),
        "mean_compatible_with_gym": bool(abs(mean_gap) <= MAX_MEAN_PERCENTILE_GAP),
        "ks_compatible_with_gym": bool(_ks_distance(gym_scores, scores) <= MAX_KS_DISTANCE),
        "high_complex_improved_retained": bool(
            int(np.sum(improved_high)) >= high_retention_needed
        ),
        "high_complex_improved_retention_needed": high_retention_needed,
    }


def _write_per_env_csv(
    path: Path,
    rows: list[dict[str, Any]],
    component_keys: list[str],
    score_keys: list[str],
) -> None:
    fieldnames = [
        "run",
        "env_idx",
        "delta",
        "improved",
        "complexity_score",
        *score_keys,
        "complexity_band",
        *[f"{key}_percentile" for key in component_keys],
        *component_keys,
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = {
        "gym": _load_run(Path(args.gym_dir).expanduser().resolve(), arm="gym"),
        "prior_full": _load_run(Path(args.prior_full_dir).expanduser().resolve(), arm="prior"),
        "prior_q90": _load_run(Path(args.prior_q90_dir).expanduser().resolve(), arm="prior"),
    }
    common_names = _align_common_features(runs)
    aligned = {name: _select_columns(item, common_names) for name, item in runs.items()}

    space = _fit_gym_space(aligned["gym"], common_names)
    transformed = {name: _transform(matrix, space) for name, matrix in aligned.items()}
    feature_shifts = _feature_shift_summary(space["kept_names"], transformed)
    metric_map = {
        name: _metrics(z, space, reference=(name == "gym")) for name, z in transformed.items()
    }
    gym_metrics = metric_map["gym"]
    scores: dict[str, np.ndarray] = {}
    component_scores: dict[str, dict[str, np.ndarray]] = {}
    for name, metrics in metric_map.items():
        score, components = _score_metrics(metrics, gym_metrics)
        scores[name] = score
        component_scores[name] = components

    family_scores = {
        family: {
            name: np.mean(np.vstack([component_scores[name][key] for key in keys]), axis=0)
            for name in runs
        }
        for family, keys in SCORE_FAMILIES.items()
    }

    gym_scores = scores["gym"]
    gym_improved = runs["gym"]["improved"]
    gym_mean = float(np.mean(gym_scores))
    gym_std = float(np.std(gym_scores))
    q25, q75 = [float(v) for v in np.quantile(gym_scores, [0.25, 0.75])]
    gym_improved_high_count = int(np.sum(gym_improved & (gym_scores >= q75)))
    summaries = {
        name: _summary(
            scores=scores[name],
            improved=runs[name]["improved"],
            deltas=runs[name]["deltas"],
            gym_scores=gym_scores,
            gym_improved=gym_improved,
            gym_mean=gym_mean,
            gym_std=gym_std,
            q25=q25,
            q75=q75,
            gym_improved_high_count=gym_improved_high_count,
        )
        for name in runs
    }

    family_refs: dict[str, dict[str, Any]] = {}
    family_summaries: dict[str, dict[str, Any]] = {}
    for family, by_run_scores in family_scores.items():
        ref_scores = by_run_scores["gym"]
        ref_mean = float(np.mean(ref_scores))
        ref_std = float(np.std(ref_scores))
        ref_q25, ref_q75 = [float(v) for v in np.quantile(ref_scores, [0.25, 0.75])]
        ref_improved_high = int(np.sum(gym_improved & (ref_scores >= ref_q75)))
        family_refs[family] = {
            "complexity_q25": ref_q25,
            "complexity_q75": ref_q75,
            "gym_improved_high_complex_count": ref_improved_high,
            "gym_mean": ref_mean,
            "gym_std": ref_std,
        }
        family_summaries[family] = {
            name: _summary(
                scores=by_run_scores[name],
                improved=runs[name]["improved"],
                deltas=runs[name]["deltas"],
                gym_scores=ref_scores,
                gym_improved=gym_improved,
                gym_mean=ref_mean,
                gym_std=ref_std,
                q25=ref_q25,
                q75=ref_q75,
                gym_improved_high_count=ref_improved_high,
            )
            for name in runs
        }

    prior_gate = {}
    family_gate: dict[str, dict[str, dict[str, Any]]] = {}
    for name in ["prior_full", "prior_q90"]:
        item = summaries[name]
        prior_gate[name] = {
            "pass": bool(
                item["has_simple_and_complex_tails"]
                and item["mean_compatible_with_gym"]
                and item["ks_compatible_with_gym"]
                and item["high_complex_improved_retained"]
            ),
            "has_simple_and_complex_tails": item["has_simple_and_complex_tails"],
            "mean_compatible_with_gym": item["mean_compatible_with_gym"],
            "ks_compatible_with_gym": item["ks_compatible_with_gym"],
            "high_complex_improved_retained": item["high_complex_improved_retained"],
        }
    for family, by_run in family_summaries.items():
        family_gate[family] = {}
        for name in ["prior_full", "prior_q90"]:
            item = by_run[name]
            family_gate[family][name] = {
                "pass": bool(
                    item["has_simple_and_complex_tails"]
                    and item["mean_compatible_with_gym"]
                    and item["ks_compatible_with_gym"]
                    and item["high_complex_improved_retained"]
                ),
                "has_simple_and_complex_tails": item["has_simple_and_complex_tails"],
                "mean_compatible_with_gym": item["mean_compatible_with_gym"],
                "ks_compatible_with_gym": item["ks_compatible_with_gym"],
                "high_complex_improved_retained": item["high_complex_improved_retained"],
            }

    subset_audits = {
        subset: _subset_complexity_summaries(subset=subset, common_names=common_names, runs=runs)
        for subset in FEATURE_SUBSET_DEFINITIONS
    }
    fair_subset_names = [
        name for name in common_names if _feature_subset_keep(name, "fair_observed_behavior")
    ]
    if fair_subset_names:
        fair_split_generalization = _split_off_gym_generalization_summary(
            gym_matrix=_select_columns(runs["gym"], fair_subset_names),
            target_matrices={
                "prior_full": _select_columns(runs["prior_full"], fair_subset_names),
                "prior_q90": _select_columns(runs["prior_q90"], fair_subset_names),
            },
            names=fair_subset_names,
        )
    else:
        fair_split_generalization = {"error": "fair observed subset has no common features"}
    fair_gate = subset_audits.get("fair_observed_behavior", {}).get("gate", {})
    fair_observed_behavior_gate_pass = bool(
        fair_gate
        and all(bool(fair_gate.get(name, {}).get("pass", False)) for name in ["prior_full", "prior_q90"])
    )
    target_excess = fair_split_generalization.get("target_vs_heldout_excess", {})
    fair_residual_excess_q50 = {
        name: float(item.get("primary_true_prior_excess_q50", float("nan")))
        for name, item in target_excess.items()
        if isinstance(item, dict)
    }

    component_keys = sorted(gym_metrics)
    per_env_rows: list[dict[str, Any]] = []
    for name in ["gym", "prior_full", "prior_q90"]:
        low = scores[name] <= q25
        high = scores[name] >= q75
        for row_idx, env_idx in enumerate(runs[name]["env_indices"]):
            band = "low" if low[row_idx] else "high" if high[row_idx] else "mid"
            row: dict[str, Any] = {
                "run": name,
                "env_idx": int(env_idx),
                "delta": float(runs[name]["deltas"][row_idx]),
                "improved": bool(runs[name]["improved"][row_idx]),
                "complexity_score": float(scores[name][row_idx]),
                "complexity_band": band,
            }
            for family in SCORE_FAMILIES:
                row[f"{family}_score"] = float(family_scores[family][name][row_idx])
            for key in component_keys:
                row[key] = float(metric_map[name][key][row_idx])
                row[f"{key}_percentile"] = float(component_scores[name][key][row_idx])
            per_env_rows.append(row)

    csv_path = out_dir / "profile_relative_complexity_per_env.csv"
    _write_per_env_csv(
        csv_path,
        per_env_rows,
        component_keys,
        [f"{family}_score" for family in SCORE_FAMILIES],
    )

    hard_read = {
        "relative_complexity_gate_pass": bool(all(item["pass"] for item in prior_gate.values())),
        "fair_observed_behavior_gate_pass": fair_observed_behavior_gate_pass,
        "raw_full_compatibility_gate_pass": bool(all(item["pass"] for item in prior_gate.values())),
        "fair_off_gym_raw_gate_over_narrow": bool(
            fair_split_generalization.get(
                "raw_off_gym_gate_is_over_narrow_for_small_gym_reference",
                False,
            )
        ),
        "fair_residual_excess_q50": fair_residual_excess_q50,
        "interpretation": (
            "fair_observed_behavior_gate_pass is the fair Gym/prior complexity gate. "
            "raw_full_compatibility_gate_pass includes next_state targets and is a conservative "
            "diagnostic because Gym next_state is next obs while prior next_state is latent state target. "
            "fair_off_gym_raw_gate_over_narrow and fair_residual_excess_q50 separate reference "
            "sparsity from true prior PCA-residual excess; they are diagnostics, not selector quotas."
        ),
    }
    report = {
        "analysis_entry": "phase2_profile_relative_complexity_audit",
        "contract": {
            "read_only": True,
            "does_not_train": True,
            "does_not_sample_envs": True,
            "does_not_modify_selector": True,
            "does_not_modify_fit_model": True,
        },
        "inputs": {
            "gym_dir": str(Path(args.gym_dir).expanduser().resolve()),
            "prior_full_dir": str(Path(args.prior_full_dir).expanduser().resolve()),
            "prior_q90_dir": str(Path(args.prior_q90_dir).expanduser().resolve()),
        },
        "gym_reference": {
            "kept_feature_count": int(len(space["kept_names"])),
            "rank95": int(space["rank95"]),
            "complexity_q25": q25,
            "complexity_q75": q75,
            "gym_improved_high_complex_count": gym_improved_high_count,
            "singular_energy_top10": space["singular_energy_top10"],
        },
        "component_keys": component_keys,
        "summaries": summaries,
        "score_family_refs": family_refs,
        "score_family_summaries": family_summaries,
        "feature_subset_audits": subset_audits,
        "fair_observed_split_generalization": fair_split_generalization,
        "feature_shift_summary": feature_shifts,
        "prior_gate": prior_gate,
        "score_family_gate": family_gate,
        "hard_read": hard_read,
        "files": {
            "per_env_csv": str(csv_path),
        },
    }

    json_path = out_dir / "profile_relative_complexity_audit.json"
    md_path = out_dir / "profile_relative_complexity_audit.md"
    json_path.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Profile Relative Complexity Audit",
        "",
        "Read-only per-environment complexity audit calibrated on Gym common behavior features.",
        "",
        f"- relative complexity gate pass: `{hard_read['relative_complexity_gate_pass']}`",
        f"- fair observed-behavior gate pass: `{hard_read['fair_observed_behavior_gate_pass']}`",
        f"- raw full compatibility gate pass: `{hard_read['raw_full_compatibility_gate_pass']}`",
        f"- common kept features: `{len(space['kept_names'])}`",
        f"- Gym PCA rank95: `{space['rank95']}`",
        f"- Gym complexity q25/q75: `{q25:.3f}` / `{q75:.3f}`",
        "",
        "## Distribution Summary",
        "",
        "| run | improved | delta mean | complexity mean | mean gap | KS | low/mid/high | improved low/mid/high | tail balance | gate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in ["gym", "prior_full", "prior_q90"]:
        item = summaries[name]
        gate = "ref" if name == "gym" else str(prior_gate[name]["pass"])
        lines.append(
            f"| `{name}` | {item['improved_count']}/{item['n']} | {item['delta_mean']:+.2f} | "
            f"{item['complexity_mean']:.3f} | {item['mean_gap_vs_gym']:+.3f} | "
            f"{item['ks_vs_gym']:.3f} | "
            f"{item['low_complex_count']}/{item['mid_complex_count']}/{item['high_complex_count']} | "
            f"{item['improved_low_complex_count']}/{item['improved_mid_complex_count']}/{item['improved_high_complex_count']} | "
            f"{item['tail_balance']:.2f} | `{gate}` |"
        )
    lines.extend(
        [
            "",
            "## Feature-Subset Audit",
            "",
            "| subset | run | features | mean | mean gap | KS | low/mid/high | improved low/mid/high | gate |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for subset, audit in subset_audits.items():
        if audit.get("error"):
            lines.append(
                f"| `{subset}` | `error` | {audit.get('feature_count', 0)} |  |  |  |  |  | `{audit['error']}` |"
            )
            continue
        for name in ["gym", "prior_full", "prior_q90"]:
            item = audit["summaries"][name]
            gate = "ref" if name == "gym" else str(audit["gate"][name]["pass"])
            lines.append(
                f"| `{subset}` | `{name}` | {audit['kept_feature_count']} | "
                f"{item['complexity_mean']:.3f} | {item['mean_gap_vs_gym']:+.3f} | "
                f"{item['ks_vs_gym']:.3f} | "
                f"{item['low_complex_count']}/{item['mid_complex_count']}/{item['high_complex_count']} | "
                f"{item['improved_low_complex_count']}/{item['improved_mid_complex_count']}/{item['improved_high_complex_count']} | "
                f"`{gate}` |"
            )
    lines.extend(
        [
            "",
            "## Feature-Subset Definitions",
            "",
        ]
    )
    for subset, description in FEATURE_SUBSET_DEFINITIONS.items():
        lines.append(f"- `{subset}`: {description}")
    lines.extend(
        [
            "",
            "## Largest Feature Shifts",
            "",
            "| run | feature | mean z-shift | median z-shift | std ratio |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for name in ["prior_full", "prior_q90"]:
        for row in feature_shifts[name]["top_abs_mean_z_shift"][:10]:
            lines.append(
                f"| `{name}` | `{row['feature']}` | {row['mean_z_shift']:+.2f} | "
                f"{row['median_z_shift']:+.2f} | {row['std_ratio']:.2f} |"
            )
    lines.extend(
        [
            "",
            "## Shift Categories",
            "",
            "| run | category | mean abs z-shift | max abs z-shift | features |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for name in ["prior_full", "prior_q90"]:
        for category, item in feature_shifts[name]["category_abs_mean_shift"].items():
            lines.append(
                f"| `{name}` | `{category}` | {item['mean_abs_shift']:.2f} | "
                f"{item['max_abs_shift']:.2f} | {item['feature_count']} |"
            )
    lines.extend(
        [
            "",
            "## Score Family Summary",
            "",
            "| family | run | complexity mean | mean gap | KS | low/mid/high | improved low/mid/high | tail balance | gate |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for family in SCORE_FAMILIES:
        for name in ["gym", "prior_full", "prior_q90"]:
            item = family_summaries[family][name]
            gate = "ref" if name == "gym" else str(family_gate[family][name]["pass"])
            lines.append(
                f"| `{family}` | `{name}` | {item['complexity_mean']:.3f} | "
                f"{item['mean_gap_vs_gym']:+.3f} | {item['ks_vs_gym']:.3f} | "
                f"{item['low_complex_count']}/{item['mid_complex_count']}/{item['high_complex_count']} | "
                f"{item['improved_low_complex_count']}/{item['improved_mid_complex_count']}/{item['improved_high_complex_count']} | "
                f"{item['tail_balance']:.2f} | `{gate}` |"
            )
    lines.extend(
        [
            "",
            "## Gate Read",
            "",
            f"- simple and complex tail threshold: `{MIN_TAIL_FRACTION:.2f}` each",
            f"- mean compatibility threshold: `abs(mean gap) <= {MAX_MEAN_PERCENTILE_GAP:.2f}`",
            f"- KS compatibility threshold: `{MAX_KS_DISTANCE:.2f}`",
            f"- high-complex improved retention threshold: `{MIN_HIGH_COMPLEX_IMPROVED_RETENTION:.2f}` of Gym high-complex improved count",
            "",
            "## Component Definition",
            "",
            "- `in_gym_complexity`: average percentile of standardized Gym-space energy and Gym-rank95 leverage.",
            "- `off_gym_novelty`: average percentile of nearest-Gym distance and residual outside Gym rank95.",
            "- `full_complexity_compatibility`: combined score; useful as a conservative compatibility alarm, not as pure complexity.",
            "- `z_norm_energy`: standardized distance from Gym mean.",
            "- `pca95_leverage`: energy inside the Gym rank95 behavior subspace.",
            "- `residual_energy` and `residual_ratio`: off-Gym-rank95 behavior not explained by dominant Gym modes.",
            "- `nearest_gym_distance`: local distance to the nearest Gym environment in common feature space.",
            "- `complexity_score`: average Gym-percentile rank over those components.",
            "",
            "## Files",
            "",
            f"- JSON: `{json_path}`",
            f"- per-env CSV: `{csv_path}`",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    report["files"]["json"] = str(json_path)
    report["files"]["md"] = str(md_path)
    print(json.dumps({"json": str(json_path), "md": str(md_path), **hard_read}, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gym-dir", default=str(DEFAULT_GYM_DIR))
    parser.add_argument("--prior-full-dir", default=str(DEFAULT_PRIOR_FULL_DIR))
    parser.add_argument("--prior-q90-dir", default=str(DEFAULT_PRIOR_Q90_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
