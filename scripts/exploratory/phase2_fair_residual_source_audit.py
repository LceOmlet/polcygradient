#!/usr/bin/env python
"""Source-parameter audit for fair observed Gym/prior residuals.

This is a read-only diagnostic.  It joins the official fair observed
complexity score back to the frozen-h source parameters and sidecar features
so the next intervention can be chosen from the actual computation graph,
instead of from ad-hoc keyword guesses.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from scripts.exploratory.phase2_profile_relative_complexity_audit import (
    SCORE_FAMILIES,
    _align_common_features,
    _feature_subset_keep,
    _fit_gym_space,
    _json_safe,
    _load_run,
    _metrics,
    _score_metrics,
    _select_columns,
    _transform,
)
from scripts.exploratory.phase2_profile_spectral_rank_audit import _read_jsonl, _sidecar_path


DEFAULT_GYM_DIR = Path("/home/chen/RLPFN/artifacts/phase2_profile_v2l_uniform_transfer_gym_sidecar_u16_b512_advnorm_0515")
DEFAULT_PRIOR_DIR = Path("/home/chen/RLPFN/artifacts/phase2_profile_v2p_activation_semantics_fix_q90_opt_u4_b512_tkl003_0516")
DEFAULT_OUTPUT_DIR = Path("/home/chen/RLPFN/artifacts/phase2_fair_residual_source_audit")

NUMERIC_SOURCE_KEYS = (
    "state_dim",
    "obs_dim",
    "action_dim",
    "noise_dim",
    "zero_pad_dim",
    "init_state_std",
    "state_noise_std",
    "alpha",
    "state_full_rms_target",
    "state_output_scale",
    "obs_output_scale",
    "reward_scale",
    "reward_clip",
    "reward_dropout_ratio",
    "ctrl_reward_weight",
    "terminal_reset_count_target",
    "reward_state_input_gain_fraction",
    "reward_action_input_gain_fraction",
    "reward_noise_input_gain_fraction",
    "reward_state_to_action_gain_ratio",
    "reward_state_to_noise_gain_ratio",
    "reward_state_input_gain_density_ratio",
    "reward_action_input_gain_density_ratio",
    "reward_noise_input_gain_density_ratio",
    "state_highway_lambda",
    "action_noise_train_std",
    "action_noise_eval_std",
    "exact_scm_gym_lowtail_family_selected",
    "exact_scm_gym_lowtail_action_gain",
    "exact_scm_gym_lowtail_signed_action_projection_enabled",
    "exact_scm_gym_lowtail_noise_gain",
    "exact_scm_gym_lowtail_init_state_std",
    "exact_scm_gym_lowtail_reward_linear_weight",
    "exact_scm_gym_lowtail_reward_potential_delta_weight",
    "exact_scm_gym_lowtail_reward_state_cost",
    "exact_scm_gym_lowtail_reward_ctrl_cost",
)

REALIZED_SIDECAR_SOURCE_KEYS = {
    "state_dim",
    "obs_dim",
    "action_dim",
    "noise_dim",
    "zero_pad_dim",
    "reward_state_input_gain_fraction",
    "reward_action_input_gain_fraction",
    "reward_noise_input_gain_fraction",
    "reward_state_to_action_gain_ratio",
    "reward_state_to_noise_gain_ratio",
    "reward_state_input_gain_density_ratio",
    "reward_action_input_gain_density_ratio",
    "reward_noise_input_gain_density_ratio",
}

SIDECAR_KEYS = (
    "pre_norm_obs_per_dim_std_q90",
    "pre_norm_obs_value_std",
    "pre_norm_obs_value_absmax",
    "pre_norm_obs_per_dim_rms_mean",
    "obs_norm_state_rms_per_dim_mean",
    "obs_norm_state_rms_per_dim_q90",
    "obs_per_dim_std_q90",
    "obs_value_std",
    "obs_value_absmax",
    "obs_step_l2_delta_mean",
    "obs_step_l2_delta_to_rms_ratio",
    "done_count",
    "first_done_step",
    "raw_reward_std",
    "raw_reward_q10",
    "raw_reward_sum",
    "training_reward_q10",
    "training_reward_q50",
    "training_reward_q90",
    "training_reward_sum",
    "input_reward_token_std",
    "input_reward_token_absmax",
    "action_value_absmax",
    "raw_policy_action_absmax_to_unit_clip",
    "action_per_dim_temporal_std_q10",
    "action_per_dim_temporal_std_q90",
    "next_state_step_l2_delta_to_rms_ratio",
)


def _as_float(value: Any) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float, np.number)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if text == "":
            return float("nan")
        lowered = text.lower()
        if lowered in {"true", "false"}:
            return 1.0 if lowered == "true" else 0.0
        try:
            return float(text)
        except ValueError:
            return float("nan")
    return float("nan")


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return float("nan")
    xx = x[mask]
    yy = y[mask]
    if float(np.std(xx)) <= 1e-12 or float(np.std(yy)) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(xx, yy)[0, 1])


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(values, dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = rank
        start = end
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return float("nan")
    xx = x[mask]
    yy = y[mask]
    if float(np.std(xx)) <= 1e-12 or float(np.std(yy)) <= 1e-12:
        return float("nan")
    return _pearson(_rankdata(xx), _rankdata(yy))


def _summary(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return {"n": 0}
    return {
        "n": int(len(finite)),
        "mean": float(np.mean(finite)),
        "q25": float(np.quantile(finite, 0.25)),
        "q50": float(np.quantile(finite, 0.50)),
        "q75": float(np.quantile(finite, 0.75)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def _bin_value(key: str, value: float) -> str:
    if not math.isfinite(value):
        return "nan"
    if key.endswith("family_selected"):
        return "selected" if value >= 0.5 else "not_selected"
    if key.endswith("action_dim"):
        if value <= 2:
            return "low_1_2"
        if value <= 6:
            return "mid_3_6"
        return "high_7_plus"
    if key.endswith("obs_dim"):
        if value <= 4:
            return "low_1_4"
        if value <= 24:
            return "mid_5_24"
        return "high_25_plus"
    if key.endswith("state_dim"):
        if value <= 8:
            return "low_1_8"
        if value <= 48:
            return "mid_9_48"
        return "high_49_plus"
    if key.endswith("noise_dim"):
        if value <= 0:
            return "zero"
        if value <= 4:
            return "low_1_4"
        if value <= 16:
            return "mid_5_16"
        return "high_17_plus"
    if key.endswith("zero_pad_dim"):
        if value <= 64:
            return "low_0_64"
        if value <= 256:
            return "mid_65_256"
        return "high_257_plus"
    return f"{value:.6g}"


def _source_group_summaries(joined: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in keys:
        buckets: dict[str, list[dict[str, Any]]] = {}
        for row in joined:
            value = _as_float(row.get(key))
            buckets.setdefault(_bin_value(key, value), []).append(row)
        rows: list[dict[str, Any]] = []
        for bucket, items in buckets.items():
            fair = np.asarray([_as_float(item.get("fair_score")) for item in items], dtype=np.float64)
            residual = np.asarray([_as_float(item.get("fair_residual_energy_raw")) for item in items], dtype=np.float64)
            nearest = np.asarray([_as_float(item.get("fair_nearest_gym_distance_raw")) for item in items], dtype=np.float64)
            off = np.asarray([_as_float(item.get("fair_off_gym_score")) for item in items], dtype=np.float64)
            rows.append(
                {
                    "bucket": bucket,
                    "count": int(len(items)),
                    "low_count": int(sum(str(item.get("fair_band")) == "low" for item in items)),
                    "mid_count": int(sum(str(item.get("fair_band")) == "mid" for item in items)),
                    "high_count": int(sum(str(item.get("fair_band")) == "high" for item in items)),
                    "fair_score": _summary(fair),
                    "raw_residual_energy": _summary(residual),
                    "raw_nearest_gym_distance": _summary(nearest),
                    "off_gym_score": _summary(off),
                }
            )
        rows.sort(key=lambda row: (-int(row["count"]), str(row["bucket"])))
        out[key] = rows
    return out


def _gym_lowtail_feature_target_summary(fair: dict[str, Any]) -> dict[str, Any]:
    """Describe the actual Gym low-tail prototype in observed feature space."""

    gym_matrix = np.asarray(fair["gym_matrix"], dtype=np.float64)
    prior_matrix = np.asarray(fair["prior_matrix"], dtype=np.float64)
    names = list(fair["subset_names"])
    gym_score = np.asarray(fair["scores"]["gym"]["all"], dtype=np.float64)
    prior_score = np.asarray(fair["scores"]["prior"]["all"], dtype=np.float64)
    if gym_matrix.size == 0 or prior_matrix.size == 0 or len(names) == 0:
        return {"error": "empty fair observed matrices"}
    low_cut = float(np.quantile(gym_score, 0.25))
    high_cut = float(np.quantile(gym_score, 0.75))
    gym_low = gym_matrix[gym_score <= low_cut]
    gym_mid = gym_matrix[(gym_score > low_cut) & (gym_score < high_cut)]
    gym_high = gym_matrix[gym_score >= high_cut]
    prior_low_like = prior_matrix[prior_score <= low_cut]
    gym_std = np.std(gym_matrix, axis=0)
    gym_std = np.maximum(gym_std, 1e-12)

    rows: list[dict[str, Any]] = []
    for col, name in enumerate(names):
        low_values = gym_low[:, col] if len(gym_low) else np.asarray([], dtype=np.float64)
        mid_values = gym_mid[:, col] if len(gym_mid) else np.asarray([], dtype=np.float64)
        high_values = gym_high[:, col] if len(gym_high) else np.asarray([], dtype=np.float64)
        prior_values = prior_matrix[:, col]
        low_median = float(np.median(low_values)) if len(low_values) else float("nan")
        prior_median = float(np.median(prior_values)) if len(prior_values) else float("nan")
        rows.append(
            {
                "feature": str(name),
                "category": _feature_category(str(name)),
                "gym_low": _summary(low_values),
                "gym_mid": _summary(mid_values),
                "gym_high": _summary(high_values),
                "prior": _summary(prior_values),
                "prior_low_like": _summary(prior_low_like[:, col] if len(prior_low_like) else np.asarray([], dtype=np.float64)),
                "prior_minus_gym_low_median_z": float((prior_median - low_median) / float(gym_std[col])),
                "low_vs_high_median_z": float(
                    (low_median - float(np.median(high_values))) / float(gym_std[col])
                )
                if len(high_values)
                else float("nan"),
            }
        )
    rows.sort(key=lambda row: abs(float(row["prior_minus_gym_low_median_z"])), reverse=True)

    by_category: dict[str, list[float]] = {}
    for row in rows:
        by_category.setdefault(str(row["category"]), []).append(abs(float(row["prior_minus_gym_low_median_z"])))
    category_rows = [
        {
            "category": category,
            "feature_count": int(len(values)),
            "mean_abs_prior_minus_gym_low_median_z": float(np.nanmean(np.asarray(values, dtype=np.float64))),
        }
        for category, values in sorted(by_category.items())
    ]
    category_rows.sort(key=lambda row: float(row["mean_abs_prior_minus_gym_low_median_z"]), reverse=True)

    return {
        "interpretation": (
            "These are the raw fair-observed feature targets for Gym's own low-complexity quartile. "
            "Large prior_minus_gym_low_median_z rows identify concrete generator subtargets before "
            "changing source distributions."
        ),
        "gym_low_score_cut": low_cut,
        "gym_high_score_cut": high_cut,
        "gym_low_count": int(len(gym_low)),
        "gym_mid_count": int(len(gym_mid)),
        "gym_high_count": int(len(gym_high)),
        "prior_low_like_count": int(len(prior_low_like)),
        "category_mismatch": category_rows,
        "top_prior_vs_gym_low_mismatch_features": rows[:40],
    }


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


def _trajectory_block(name: str) -> str:
    base = name.split(":", 1)[1] if ":" in name else name
    if "reward" in base:
        return "reward"
    if "action" in base or "policy" in base:
        return "action"
    if "done" in base or "truncated" in base:
        return "done_truncated"
    if "obs" in base or "state" in base:
        return "obs_state_delta" if name.startswith("d:") else "obs_state_u0"
    return "other"


def _effective_rank(values: np.ndarray) -> float:
    eig = np.asarray(values, dtype=np.float64)
    eig = eig[np.isfinite(eig)]
    eig = eig[eig > 1e-12]
    if eig.size == 0:
        return 0.0
    weights = eig / float(np.sum(eig))
    entropy = -float(np.sum(weights * np.log(np.maximum(weights, 1e-12))))
    return float(np.exp(entropy))


def _safe_cov(matrix: np.ndarray) -> np.ndarray:
    x = np.asarray(matrix, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] <= 1:
        return np.zeros((x.shape[1] if x.ndim == 2 else 0, x.shape[1] if x.ndim == 2 else 0), dtype=np.float64)
    return np.cov(x, rowvar=False, ddof=1)


def _cov_summary(matrix: np.ndarray) -> dict[str, Any]:
    cov = _safe_cov(matrix)
    if cov.size == 0:
        return {"feature_count": 0}
    diag = np.diag(cov)
    eig = np.linalg.eigvalsh(cov)
    offdiag = cov - np.diag(diag)
    return {
        "feature_count": int(cov.shape[0]),
        "trace": float(np.trace(cov)),
        "fro_norm": float(np.linalg.norm(cov, ord="fro")),
        "diag_mean": float(np.mean(diag)) if diag.size else 0.0,
        "diag_rms": float(np.sqrt(np.mean(diag * diag))) if diag.size else 0.0,
        "offdiag_fro_norm": float(np.linalg.norm(offdiag, ord="fro")),
        "top_eigenvalue": float(np.max(eig)) if eig.size else 0.0,
        "effective_rank": _effective_rank(eig),
    }


def _cov_distance_summary(reference: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    ref_cov = _safe_cov(reference)
    target_cov = _safe_cov(target)
    if ref_cov.size == 0 or target_cov.size == 0:
        return {"error": "empty covariance"}
    delta = target_cov - ref_cov
    ref_norm = float(np.linalg.norm(ref_cov, ord="fro"))
    diag_ref = np.diag(ref_cov)
    diag_target = np.diag(target_cov)
    return {
        "cov_fro_delta": float(np.linalg.norm(delta, ord="fro")),
        "cov_fro_delta_over_gym_low": float(np.linalg.norm(delta, ord="fro") / max(ref_norm, 1e-12)),
        "diag_rms_delta": float(np.sqrt(np.mean((diag_target - diag_ref) ** 2))) if diag_ref.size else 0.0,
        "diag_rms_delta_over_gym_low_diag_rms": float(
            np.sqrt(np.mean((diag_target - diag_ref) ** 2)) / max(float(np.sqrt(np.mean(diag_ref * diag_ref))), 1e-12)
        )
        if diag_ref.size
        else 0.0,
    }


def _block_indices(names: list[str]) -> dict[str, list[int]]:
    blocks: dict[str, list[int]] = {
        "obs_state_u0": [],
        "obs_state_delta": [],
        "done_truncated": [],
        "action": [],
        "reward": [],
    }
    for idx, name in enumerate(names):
        block = _trajectory_block(str(name))
        if block in blocks:
            blocks[block].append(idx)
    return blocks


def _residual_covariance_block_audit(
    fair: dict[str, Any],
    joined: list[dict[str, Any]],
    *,
    selected_source_key: str = "h:exact_scm_gym_lowtail_family_selected",
) -> dict[str, Any]:
    """Split the Gym-low PCA residual target into trajectory feature blocks.

    This targets the current failure mode directly: the closest prior rows are
    already near Gym in raw distance and PCA leverage, but remain outside the
    Gym-low PCA residual covariance.  The output names the block-level residual
    covariance gap without changing selector or generator behavior.
    """

    names = list(fair["space"]["kept_names"])
    gym_z = np.asarray(fair["transformed"]["gym"], dtype=np.float64)
    prior_z = np.asarray(fair["transformed"]["prior"], dtype=np.float64)
    gym_score = np.asarray(fair["scores"]["gym"]["all"], dtype=np.float64)
    prior_score = np.asarray(fair["scores"]["prior"]["all"], dtype=np.float64)
    if gym_z.size == 0 or prior_z.size == 0 or gym_score.size == 0 or prior_score.size == 0:
        return {"error": "empty fair observed matrices"}

    residual_gym = _pca_residual_matrix(gym_z, fair["space"])
    residual_prior = _pca_residual_matrix(prior_z, fair["space"])
    low_cut = float(np.quantile(gym_score, 0.25))
    gym_low_idx = np.flatnonzero(gym_score <= low_cut)
    if gym_low_idx.size == 0:
        return {"error": "Gym low residual target is empty"}

    n = min(len(joined), len(prior_score), int(prior_z.shape[0]))
    near_count = min(int(gym_low_idx.size), n)
    prior_near_idx = np.argsort(prior_score[:n])[:near_count]
    selected_idx = np.asarray(
        [idx for idx in range(n) if _as_float(joined[idx].get(selected_source_key)) >= 0.5],
        dtype=np.int64,
    )
    if selected_idx.size == 0:
        selected_idx = prior_near_idx

    row_idx = int(prior_near_idx[0]) if prior_near_idx.size else 0
    blocks = _block_indices(names)
    block_rows: list[dict[str, Any]] = []
    for block, indices in blocks.items():
        if not indices:
            continue
        cols = np.asarray(indices, dtype=np.int64)
        gym_low_block = residual_gym[gym_low_idx][:, cols]
        prior_near_block = residual_prior[prior_near_idx][:, cols]
        selected_block = residual_prior[selected_idx][:, cols]
        closest_block = residual_prior[row_idx, cols]
        gym_low_mean = np.mean(gym_low_block, axis=0)
        near_mean = np.mean(prior_near_block, axis=0)
        selected_mean = np.mean(selected_block, axis=0)
        block_rows.append(
            {
                "block": block,
                "feature_count": int(len(indices)),
                "features": [str(names[idx]) for idx in indices],
                "gym_low_residual_rms": float(np.sqrt(np.mean(gym_low_block * gym_low_block))),
                "prior_near_residual_rms": float(np.sqrt(np.mean(prior_near_block * prior_near_block))),
                "selected_residual_rms": float(np.sqrt(np.mean(selected_block * selected_block))),
                "closest_prior_row_residual_rms": float(np.sqrt(np.mean(closest_block * closest_block))),
                "prior_near_minus_gym_low_mean_residual_rms": float(
                    np.sqrt(np.mean((near_mean - gym_low_mean) ** 2))
                ),
                "selected_minus_gym_low_mean_residual_rms": float(
                    np.sqrt(np.mean((selected_mean - gym_low_mean) ** 2))
                ),
                "gym_low_covariance": _cov_summary(gym_low_block),
                "prior_near_covariance": _cov_summary(prior_near_block),
                "selected_covariance": _cov_summary(selected_block),
                "prior_near_vs_gym_low_covariance": _cov_distance_summary(gym_low_block, prior_near_block),
                "selected_vs_gym_low_covariance": _cov_distance_summary(gym_low_block, selected_block),
            }
        )
    block_rows.sort(
        key=lambda row: float(
            row["prior_near_vs_gym_low_covariance"].get("cov_fro_delta_over_gym_low", 0.0)
        ),
        reverse=True,
    )

    cross_rows: list[dict[str, Any]] = []
    block_items = [(block, idxs) for block, idxs in blocks.items() if idxs]
    for left_pos, (left, left_indices) in enumerate(block_items):
        for right, right_indices in block_items[left_pos + 1 :]:
            left_cols = np.asarray(left_indices, dtype=np.int64)
            right_cols = np.asarray(right_indices, dtype=np.int64)
            gym_left = residual_gym[gym_low_idx][:, left_cols]
            gym_right = residual_gym[gym_low_idx][:, right_cols]
            prior_left = residual_prior[prior_near_idx][:, left_cols]
            prior_right = residual_prior[prior_near_idx][:, right_cols]
            selected_left = residual_prior[selected_idx][:, left_cols]
            selected_right = residual_prior[selected_idx][:, right_cols]

            def _cross_cov(left_matrix: np.ndarray, right_matrix: np.ndarray) -> np.ndarray:
                l = left_matrix - np.mean(left_matrix, axis=0, keepdims=True)
                r = right_matrix - np.mean(right_matrix, axis=0, keepdims=True)
                denom = max(1, int(left_matrix.shape[0]) - 1)
                return (l.T @ r) / float(denom)

            gym_cross = _cross_cov(gym_left, gym_right)
            prior_cross = _cross_cov(prior_left, prior_right)
            selected_cross = _cross_cov(selected_left, selected_right)
            gym_norm = float(np.linalg.norm(gym_cross, ord="fro"))
            cross_rows.append(
                {
                    "block_pair": f"{left} x {right}",
                    "gym_low_cross_cov_fro": gym_norm,
                    "prior_near_cross_cov_fro": float(np.linalg.norm(prior_cross, ord="fro")),
                    "selected_cross_cov_fro": float(np.linalg.norm(selected_cross, ord="fro")),
                    "prior_near_cross_cov_delta_over_gym_low": float(
                        np.linalg.norm(prior_cross - gym_cross, ord="fro") / max(gym_norm, 1e-12)
                    ),
                    "selected_cross_cov_delta_over_gym_low": float(
                        np.linalg.norm(selected_cross - gym_cross, ord="fro") / max(gym_norm, 1e-12)
                    ),
                }
            )
    cross_rows.sort(key=lambda row: float(row["prior_near_cross_cov_delta_over_gym_low"]), reverse=True)

    return {
        "interpretation": (
            "Block covariance target for Gym low-tail in PCA95 residual space. "
            "prior_near uses the same count as Gym low and the lowest-score prior rows; "
            "selected uses exact-SCM selected rows.  Large covariance deltas identify "
            "joint trajectory geometry that a generator must produce, not marginal knobs."
        ),
        "gym_low_score_cut": low_cut,
        "gym_low_count": int(gym_low_idx.size),
        "prior_near_count": int(prior_near_idx.size),
        "selected_count": int(selected_idx.size),
        "closest_prior_row_index": row_idx,
        "closest_prior_env_idx": int(joined[row_idx].get("env_idx", row_idx)) if row_idx < len(joined) else row_idx,
        "closest_prior_score": float(prior_score[row_idx]) if row_idx < len(prior_score) else float("nan"),
        "block_covariance_rows": block_rows,
        "cross_block_covariance_rows": cross_rows,
    }


def _nearest_gym_delta_summary(fair: dict[str, Any]) -> dict[str, Any]:
    gym_z = np.asarray(fair["transformed"]["gym"], dtype=np.float64)
    prior_z = np.asarray(fair["transformed"]["prior"], dtype=np.float64)
    names = list(fair["space"]["kept_names"])
    if gym_z.size == 0 or prior_z.size == 0:
        return {"error": "empty transformed matrices"}

    nearest_dist: list[float] = []
    deltas: list[np.ndarray] = []
    for row in prior_z:
        diff = row.reshape(1, -1) - gym_z
        dist = np.sqrt(np.mean(diff * diff, axis=1))
        nearest_idx = int(np.argmin(dist))
        nearest_dist.append(float(dist[nearest_idx]))
        deltas.append(diff[nearest_idx])
    delta_matrix = np.stack(deltas, axis=0)
    mean_delta = np.mean(delta_matrix, axis=0)
    mean_abs_delta = np.mean(np.abs(delta_matrix), axis=0)
    rms_delta = np.sqrt(np.mean(delta_matrix * delta_matrix, axis=0))
    std_delta = np.std(delta_matrix, axis=0)

    rows = [
        {
            "feature": str(name),
            "category": _feature_category(str(name)),
            "mean_delta_z": float(mean),
            "mean_abs_delta_z": float(abs_mean),
            "rms_delta_z": float(rms),
            "std_delta_z": float(std),
        }
        for name, mean, abs_mean, rms, std in zip(
            names, mean_delta, mean_abs_delta, rms_delta, std_delta
        )
    ]
    rows.sort(key=lambda row: row["rms_delta_z"], reverse=True)
    by_category: dict[str, list[float]] = {}
    for row in rows:
        by_category.setdefault(str(row["category"]), []).append(float(row["rms_delta_z"]))
    category_rows = [
        {
            "category": key,
            "feature_count": int(len(values)),
            "mean_rms_delta_z": float(np.mean(values)),
            "max_rms_delta_z": float(np.max(values)),
        }
        for key, values in sorted(by_category.items())
    ]
    category_rows.sort(key=lambda row: row["mean_rms_delta_z"], reverse=True)
    return {
        "nearest_gym_distance_z": _summary(np.asarray(nearest_dist, dtype=np.float64)),
        "top_nearest_gym_delta_features": rows[:30],
        "category_nearest_gym_delta_rms": category_rows,
    }


def _kept_raw_matrix(fair: dict[str, Any], arm: str) -> np.ndarray:
    matrix = np.asarray(fair[f"{arm}_matrix"], dtype=np.float64)
    space = fair["space"]
    x = matrix[:, space["finite_cols"]]
    return x[:, space["variable_cols_after_finite"]]


def _selected_vs_gym_low_delta_summary(
    fair: dict[str, Any],
    joined: list[dict[str, Any]],
    *,
    source_key: str,
) -> dict[str, Any]:
    """Localize why the closest source subfamily still misses Gym's low tail.

    The comparison is made in the same fair-observed z-space used by the
    official score, but the target set is restricted to Gym's own low quartile.
    That keeps the next intervention tied to the model's measured feature
    geometry rather than to an ad-hoc source-parameter keyword.
    """

    gym_z = np.asarray(fair["transformed"]["gym"], dtype=np.float64)
    prior_z = np.asarray(fair["transformed"]["prior"], dtype=np.float64)
    names = list(fair["space"]["kept_names"])
    gym_score = np.asarray(fair["scores"]["gym"]["all"], dtype=np.float64)
    prior_score = np.asarray(fair["scores"]["prior"]["all"], dtype=np.float64)
    if gym_z.size == 0 or prior_z.size == 0 or not joined:
        return {"error": "empty fair observed matrices or joined rows"}
    n = min(len(joined), len(prior_z), len(prior_score))
    selected_mask = np.asarray(
        [_as_float(row.get(source_key)) >= 0.5 for row in joined[:n]],
        dtype=bool,
    )
    selected_idx = np.flatnonzero(selected_mask)
    if selected_idx.size == 0:
        return {"error": f"no selected rows for {source_key}"}

    low_cut = float(np.quantile(gym_score, 0.25))
    high_cut = float(np.quantile(gym_score, 0.75))
    gym_low_idx = np.flatnonzero(gym_score <= low_cut)
    gym_high_idx = np.flatnonzero(gym_score >= high_cut)
    if gym_low_idx.size == 0:
        return {"error": "Gym low-tail target set is empty"}

    selected_z = prior_z[selected_idx]
    gym_low_z = gym_z[gym_low_idx]
    nearest_low_indices: list[int] = []
    nearest_low_distances: list[float] = []
    nearest_deltas: list[np.ndarray] = []
    for row in selected_z:
        diff = row.reshape(1, -1) - gym_low_z
        dist = np.sqrt(np.mean(diff * diff, axis=1))
        nearest_local = int(np.argmin(dist))
        nearest_low_indices.append(int(gym_low_idx[nearest_local]))
        nearest_low_distances.append(float(dist[nearest_local]))
        nearest_deltas.append(diff[nearest_local])
    delta_matrix = np.stack(nearest_deltas, axis=0)

    selected_raw = _kept_raw_matrix(fair, "prior")[selected_idx]
    gym_low_raw = _kept_raw_matrix(fair, "gym")[gym_low_idx]
    gym_high_raw = _kept_raw_matrix(fair, "gym")[gym_high_idx] if gym_high_idx.size else np.empty((0, selected_raw.shape[1]))
    selected_median_z = np.median(selected_z, axis=0)
    gym_low_median_z = np.median(gym_low_z, axis=0)
    selected_median_raw = np.median(selected_raw, axis=0)
    gym_low_median_raw = np.median(gym_low_raw, axis=0)
    gym_high_median_raw = np.median(gym_high_raw, axis=0) if len(gym_high_raw) else np.full_like(gym_low_median_raw, np.nan)
    gym_raw_std = np.maximum(np.std(_kept_raw_matrix(fair, "gym"), axis=0), 1e-12)

    rows: list[dict[str, Any]] = []
    for col, name in enumerate(names):
        mean_delta = float(np.mean(delta_matrix[:, col]))
        abs_mean = float(np.mean(np.abs(delta_matrix[:, col])))
        rms = float(np.sqrt(np.mean(delta_matrix[:, col] * delta_matrix[:, col])))
        selected_minus_low_raw_z = float((selected_median_raw[col] - gym_low_median_raw[col]) / gym_raw_std[col])
        low_vs_high_raw_z = float((gym_low_median_raw[col] - gym_high_median_raw[col]) / gym_raw_std[col])
        rows.append(
            {
                "feature": str(name),
                "category": _feature_category(str(name)),
                "selected_minus_nearest_low_mean_delta_z": mean_delta,
                "selected_minus_nearest_low_mean_abs_delta_z": abs_mean,
                "selected_minus_nearest_low_rms_delta_z": rms,
                "selected_minus_gym_low_median_z": float(selected_median_z[col] - gym_low_median_z[col]),
                "selected_minus_gym_low_median_raw_z": selected_minus_low_raw_z,
                "gym_low_vs_high_median_raw_z": low_vs_high_raw_z,
                "selected_raw": _summary(selected_raw[:, col]),
                "gym_low_raw": _summary(gym_low_raw[:, col]),
            }
        )
    rows.sort(key=lambda row: float(row["selected_minus_nearest_low_rms_delta_z"]), reverse=True)

    by_category: dict[str, list[float]] = {}
    for row in rows:
        by_category.setdefault(str(row["category"]), []).append(float(row["selected_minus_nearest_low_rms_delta_z"]))
    category_rows = [
        {
            "category": category,
            "feature_count": int(len(values)),
            "mean_rms_delta_z": float(np.nanmean(np.asarray(values, dtype=np.float64))),
            "max_rms_delta_z": float(np.nanmax(np.asarray(values, dtype=np.float64))),
        }
        for category, values in sorted(by_category.items())
    ]
    category_rows.sort(key=lambda row: float(row["mean_rms_delta_z"]), reverse=True)

    selected_scores = prior_score[selected_idx]
    closest_order = np.argsort(selected_scores)[: min(8, len(selected_scores))]
    closest_rows: list[dict[str, Any]] = []
    for local_pos in closest_order:
        idx = int(selected_idx[int(local_pos)])
        closest_rows.append(
            {
                "env_idx": int(joined[idx].get("env_idx", idx)),
                "prior_row_index": idx,
                "fair_score": float(prior_score[idx]),
                "score_gap_above_gym_low_cut": float(prior_score[idx] - low_cut),
                "nearest_gym_low_row_index": int(nearest_low_indices[int(local_pos)]),
                "nearest_gym_low_distance_z": float(nearest_low_distances[int(local_pos)]),
                "raw_residual_energy": _as_float(joined[idx].get("fair_residual_energy_raw")),
                "raw_nearest_gym_distance": _as_float(joined[idx].get("fair_nearest_gym_distance_raw")),
            }
        )

    return {
        "interpretation": (
            "This isolates the remaining gap for the source subfamily closest to Gym low-tail. "
            "Rows are prior selected exact-SCM environments compared only against Gym's own low-score "
            "quartile in the same fair-observed z-space."
        ),
        "source_key": source_key,
        "selected_count": int(selected_idx.size),
        "gym_low_count": int(gym_low_idx.size),
        "gym_low_score_cut": low_cut,
        "selected_score": _summary(selected_scores),
        "selected_score_gap_above_gym_low_cut": _summary(selected_scores - low_cut),
        "selected_to_nearest_gym_low_distance_z": _summary(np.asarray(nearest_low_distances, dtype=np.float64)),
        "category_selected_vs_gym_low_nearest_delta": category_rows,
        "top_selected_vs_gym_low_nearest_delta_features": rows[:40],
        "closest_selected_envs_to_low_cut": closest_rows,
    }


def _pca_residual_feature_rows(fair: dict[str, Any], prior_idx: int, *, limit: int = 16) -> list[dict[str, Any]]:
    prior_z = np.asarray(fair["transformed"]["prior"], dtype=np.float64)
    if prior_z.size == 0 or prior_idx < 0 or prior_idx >= int(prior_z.shape[0]):
        return []
    basis = np.asarray(fair["space"]["basis95"], dtype=np.float64)
    names = list(fair["space"]["kept_names"])
    z = prior_z[int(prior_idx)]
    projected = (z @ basis.T) @ basis
    residual = z - projected
    rows = [
        {
            "feature": str(name),
            "category": _feature_category(str(name)),
            "z": float(z_value),
            "projected_z": float(projected_value),
            "residual_z": float(residual_value),
            "abs_residual_z": abs(float(residual_value)),
        }
        for name, z_value, projected_value, residual_value in zip(names, z, projected, residual)
    ]
    rows.sort(key=lambda row: float(row["abs_residual_z"]), reverse=True)
    return rows[:limit]


def _lowest_prior_component_blockers(
    fair: dict[str, Any],
    joined: list[dict[str, Any]],
    *,
    limit: int = 8,
) -> dict[str, Any]:
    """Explain why the closest prior rows still do not enter Gym low-tail."""

    prior_score = np.asarray(fair["scores"]["prior"]["all"], dtype=np.float64)
    gym_score = np.asarray(fair["scores"]["gym"]["all"], dtype=np.float64)
    if prior_score.size == 0 or gym_score.size == 0:
        return {"error": "empty score arrays"}
    n = min(len(joined), len(prior_score))
    low_cut = float(np.quantile(gym_score, 0.25))
    component_names = sorted(fair["components"]["prior"].keys())
    raw_names = sorted(fair["raw_metrics"]["prior"].keys())
    rows: list[dict[str, Any]] = []
    for idx in np.argsort(prior_score[:n])[: min(limit, n)]:
        idx_int = int(idx)
        component_scores = {
            key: float(np.asarray(fair["components"]["prior"][key], dtype=np.float64)[idx_int])
            for key in component_names
        }
        raw_metrics = {
            key: float(np.asarray(fair["raw_metrics"]["prior"][key], dtype=np.float64)[idx_int])
            for key in raw_names
        }
        rows.append(
            {
                "prior_row_index": idx_int,
                "env_idx": int(joined[idx_int].get("env_idx", idx_int)),
                "fair_score": float(prior_score[idx_int]),
                "score_gap_above_gym_low_cut": float(prior_score[idx_int] - low_cut),
                "fair_band": joined[idx_int].get("fair_band"),
                "exact_scm_selected": _as_float(
                    joined[idx_int].get("h:exact_scm_gym_lowtail_family_selected")
                ),
                "terminal_reset_count_target": _as_float(
                    joined[idx_int].get("h:terminal_reset_count_target")
                ),
                "init_state_std": _as_float(joined[idx_int].get("h:init_state_std")),
                "component_scores": component_scores,
                "raw_metrics": raw_metrics,
                "top_pca_residual_features": _pca_residual_feature_rows(fair, idx_int),
            }
        )
    return {
        "interpretation": (
            "Lowest-score prior rows are decomposed into score components. If nearest/in-gym "
            "components are low but residual_energy/residual_ratio stay high, the next target is "
            "Gym PCA residual covariance, not marginal terminal/action/obs tuning."
        ),
        "gym_low_score_cut": low_cut,
        "lowest_prior_rows": rows,
    }


def _nearest_self_distances(matrix: np.ndarray) -> np.ndarray:
    out = np.empty(int(matrix.shape[0]), dtype=np.float64)
    for idx, row in enumerate(matrix):
        distances = np.sqrt(np.mean((matrix - row.reshape(1, -1)) ** 2, axis=1))
        distances[idx] = np.inf
        out[idx] = float(np.min(distances))
    return out


def _nearest_cross_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    out = np.empty(int(left.shape[0]), dtype=np.float64)
    for idx, row in enumerate(left):
        distances = np.sqrt(np.mean((right - row.reshape(1, -1)) ** 2, axis=1))
        out[idx] = float(np.min(distances))
    return out


def _density_overlap_summary(fair: dict[str, Any]) -> dict[str, Any]:
    """Check whether prior/Gym occupy the same sparse z-space density support.

    This complements percentile gates.  If the Gym reference is itself sparse,
    percentile tails can look harsh; nearest-neighbor coverage and a 1NN domain
    classifier expose whether prior/Gym are nevertheless interleaved in the
    same z-space support.
    """

    gym_z = np.asarray(fair["transformed"]["gym"], dtype=np.float64)
    prior_z = np.asarray(fair["transformed"]["prior"], dtype=np.float64)
    if gym_z.size == 0 or prior_z.size == 0:
        return {"error": "empty transformed matrices"}

    gym_to_gym = _nearest_self_distances(gym_z)
    prior_to_prior = _nearest_self_distances(prior_z)
    prior_to_gym = _nearest_cross_distances(prior_z, gym_z)
    gym_to_prior = _nearest_cross_distances(gym_z, prior_z)
    gym_q75 = float(np.quantile(gym_to_gym, 0.75))
    gym_q90 = float(np.quantile(gym_to_gym, 0.90))
    gym_q95 = float(np.quantile(gym_to_gym, 0.95))
    gym_q99 = float(np.quantile(gym_to_gym, 0.99))
    gym_median = float(np.median(gym_to_gym))

    all_z = np.vstack([gym_z, prior_z])
    labels = np.asarray([0] * len(gym_z) + [1] * len(prior_z), dtype=np.int64)
    correct = 0
    other_minus_same: list[float] = []
    for idx, row in enumerate(all_z):
        distances = np.sqrt(np.mean((all_z - row.reshape(1, -1)) ** 2, axis=1))
        distances[idx] = np.inf
        nearest_idx = int(np.argmin(distances))
        correct += int(labels[nearest_idx] == labels[idx])
        same = distances[labels == labels[idx]]
        other = distances[labels != labels[idx]]
        other_minus_same.append(float(np.min(other) - np.min(same)))

    cross = np.sqrt(np.mean((gym_z[:, None, :] - prior_z[None, :, :]) ** 2, axis=2))
    gym_pair = np.sqrt(np.mean((gym_z[:, None, :] - gym_z[None, :, :]) ** 2, axis=2))
    prior_pair = np.sqrt(np.mean((prior_z[:, None, :] - prior_z[None, :, :]) ** 2, axis=2))
    energy_distance_like = float(2.0 * np.mean(cross) - np.mean(gym_pair) - np.mean(prior_pair))

    metric_map = {
        "gym": _metrics(gym_z, fair["space"], reference=True),
        "prior": _metrics(prior_z, fair["space"], reference=False),
    }
    component_ratios: dict[str, Any] = {}
    for key in ("nearest_gym_distance", "residual_energy", "residual_ratio", "z_norm_energy", "pca95_leverage"):
        gym_values = np.asarray(metric_map["gym"][key], dtype=np.float64)
        prior_values = np.asarray(metric_map["prior"][key], dtype=np.float64)
        component_ratios[key] = {
            "gym": _summary(gym_values),
            "prior": _summary(prior_values),
            "prior_to_gym_median_ratio": float(
                np.median(prior_values) / max(float(np.median(gym_values)), 1e-12)
            ),
        }

    return {
        "interpretation": (
            "If prior and Gym cover the same sparse z-space support, prior_to_gym nearest-neighbor "
            "distances should be close to Gym leave-one-out distances, bidirectional coverage should "
            "be high, and 1NN domain accuracy should be near chance."
        ),
        "gym_to_gym_nearest_loo": _summary(gym_to_gym),
        "prior_to_prior_nearest_loo": _summary(prior_to_prior),
        "prior_to_gym_nearest": _summary(prior_to_gym),
        "gym_to_prior_nearest": _summary(gym_to_prior),
        "prior_to_gym_median_over_gym_loo_median": float(np.median(prior_to_gym) / max(gym_median, 1e-12)),
        "gym_to_prior_median_over_gym_loo_median": float(np.median(gym_to_prior) / max(gym_median, 1e-12)),
        "prior_covered_by_gym_loo_q75_fraction": float(np.mean(prior_to_gym <= gym_q75)),
        "prior_covered_by_gym_loo_q90_fraction": float(np.mean(prior_to_gym <= gym_q90)),
        "prior_covered_by_gym_loo_q95_fraction": float(np.mean(prior_to_gym <= gym_q95)),
        "prior_covered_by_gym_loo_q99_fraction": float(np.mean(prior_to_gym <= gym_q99)),
        "gym_covered_by_prior_under_gym_loo_q75_fraction": float(np.mean(gym_to_prior <= gym_q75)),
        "gym_covered_by_prior_under_gym_loo_q90_fraction": float(np.mean(gym_to_prior <= gym_q90)),
        "gym_covered_by_prior_under_gym_loo_q95_fraction": float(np.mean(gym_to_prior <= gym_q95)),
        "gym_covered_by_prior_under_gym_loo_q99_fraction": float(np.mean(gym_to_prior <= gym_q99)),
        "one_nn_domain_accuracy": float(correct / max(1, int(len(all_z)))),
        "one_nn_other_minus_same_distance": _summary(np.asarray(other_minus_same, dtype=np.float64)),
        "energy_distance_like": energy_distance_like,
        "mean_cross_distance": float(np.mean(cross)),
        "mean_gym_pair_distance": float(np.mean(gym_pair)),
        "mean_prior_pair_distance": float(np.mean(prior_pair)),
        "component_metric_overlap": component_ratios,
    }


def _percentile_against(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    ref = np.sort(np.asarray(reference, dtype=np.float64))
    out = np.empty(int(len(values)), dtype=np.float64)
    for idx, value in enumerate(np.asarray(values, dtype=np.float64)):
        lo = np.searchsorted(ref, value, side="left")
        hi = np.searchsorted(ref, value, side="right")
        out[idx] = (lo + 0.5 * (hi - lo)) / max(1, len(ref))
    return out


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


def _pca_residual_matrix(z: np.ndarray, space: dict[str, Any]) -> np.ndarray:
    basis = np.asarray(space["basis95"], dtype=np.float64)
    coords = z @ basis.T
    projected = coords @ basis
    return z - projected


def _gym_split_residual_feature_contribution_summary(
    fair: dict[str, Any],
    *,
    n_splits: int = 20,
) -> dict[str, Any]:
    """Localize true prior residual excess to feature groups under Gym split fits."""

    gym_matrix = np.asarray(fair["gym_matrix"], dtype=np.float64)
    prior_matrix = np.asarray(fair["prior_matrix"], dtype=np.float64)
    names = list(fair["subset_names"])
    n_gym = int(gym_matrix.shape[0])
    if n_gym < 8:
        return {"error": "not enough Gym rows for split residual diagnostic"}

    rng = np.random.default_rng(9731)
    by_feature: dict[str, dict[str, list[float]]] = {}
    for _ in range(int(n_splits)):
        perm = rng.permutation(n_gym)
        train = gym_matrix[perm[: n_gym // 2]]
        heldout = gym_matrix[perm[n_gym // 2 :]]
        try:
            split_space = _fit_gym_space(train, names)
        except Exception:
            continue
        z_heldout = _transform(heldout, split_space)
        z_prior = _transform(prior_matrix, split_space)
        heldout_residual = _pca_residual_matrix(z_heldout, split_space)
        prior_residual = _pca_residual_matrix(z_prior, split_space)
        heldout_rms = np.sqrt(np.mean(heldout_residual * heldout_residual, axis=0))
        prior_rms = np.sqrt(np.mean(prior_residual * prior_residual, axis=0))
        for name, heldout_value, prior_value in zip(
            split_space["kept_names"],
            heldout_rms,
            prior_rms,
        ):
            bucket = by_feature.setdefault(
                str(name),
                {"heldout_rms": [], "prior_rms": [], "excess": []},
            )
            bucket["heldout_rms"].append(float(heldout_value))
            bucket["prior_rms"].append(float(prior_value))
            bucket["excess"].append(float(prior_value / max(float(heldout_value), 1e-12)))

    if not by_feature:
        return {"error": "all Gym split residual diagnostics failed"}

    rows: list[dict[str, Any]] = []
    for name, values in by_feature.items():
        heldout_arr = np.asarray(values["heldout_rms"], dtype=np.float64)
        prior_arr = np.asarray(values["prior_rms"], dtype=np.float64)
        excess_arr = np.asarray(values["excess"], dtype=np.float64)
        rows.append(
            {
                "feature": name,
                "category": _feature_category(name),
                "split_count": int(len(excess_arr)),
                "heldout_residual_rms": _summary(heldout_arr),
                "prior_residual_rms": _summary(prior_arr),
                "prior_over_heldout_residual_rms": _summary(excess_arr),
                "priority_score": float(
                    np.quantile(prior_arr, 0.50) * np.quantile(excess_arr, 0.50)
                ),
            }
        )
    rows.sort(key=lambda row: float(row["priority_score"]), reverse=True)

    category_values: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        cat = str(row["category"])
        bucket = category_values.setdefault(cat, {"priority": [], "excess_q50": [], "prior_q50": []})
        bucket["priority"].append(float(row["priority_score"]))
        bucket["excess_q50"].append(float(row["prior_over_heldout_residual_rms"].get("q50", float("nan"))))
        bucket["prior_q50"].append(float(row["prior_residual_rms"].get("q50", float("nan"))))
    category_rows = [
        {
            "category": key,
            "feature_count": int(len(values["priority"])),
            "mean_priority_score": float(np.nanmean(np.asarray(values["priority"], dtype=np.float64))),
            "mean_excess_q50": float(np.nanmean(np.asarray(values["excess_q50"], dtype=np.float64))),
            "mean_prior_residual_rms_q50": float(np.nanmean(np.asarray(values["prior_q50"], dtype=np.float64))),
        }
        for key, values in sorted(category_values.items())
    ]
    category_rows.sort(key=lambda row: float(row["mean_priority_score"]), reverse=True)
    return {
        "interpretation": (
            "This is the feature-level residual target after correcting for Gym reference sparsity. "
            "A high priority score means the prior has large PCA95 residual on that observed feature "
            "and that residual is larger than held-out Gym under the same split fit."
        ),
        "n_splits": int(n_splits),
        "top_residual_excess_features": rows[:30],
        "category_residual_excess": category_rows,
    }


def _gym_split_generalization_summary(fair: dict[str, Any], *, n_splits: int = 20) -> dict[str, Any]:
    """Estimate whether the empirical Gym PCA95 off-gym gate is over-narrow."""

    gym_matrix = np.asarray(fair["gym_matrix"], dtype=np.float64)
    prior_matrix = np.asarray(fair["prior_matrix"], dtype=np.float64)
    names = list(fair["subset_names"])
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
        z_heldout = _transform(heldout, split_space)
        z_prior = _transform(prior_matrix, split_space)
        train_metrics = _metrics(z_train, split_space, reference=True)
        train_score = _off_gym_score_against_train(
            train_metrics=train_metrics,
            target_metrics=train_metrics,
        )
        train_q75 = float(np.quantile(train_score, 0.75))
        row: dict[str, Any] = {}
        for label, target_z in (("heldout_gym", z_heldout), ("prior", z_prior)):
            target_metrics = _metrics(target_z, split_space, reference=False)
            score = _off_gym_score_against_train(
                train_metrics=train_metrics,
                target_metrics=target_metrics,
            )
            row[f"{label}_off_mean"] = float(np.mean(score))
            row[f"{label}_off_q50"] = float(np.quantile(score, 0.50))
            row[f"{label}_off_q75"] = float(np.quantile(score, 0.75))
            row[f"{label}_high_frac_vs_train_q75"] = float(np.mean(score >= train_q75))
            for key in ("nearest_gym_distance", "residual_energy", "residual_ratio"):
                row[f"{label}_{key}_median_ratio_vs_train"] = float(
                    np.median(target_metrics[key]) / max(float(np.median(train_metrics[key])), 1e-12)
                )
        split_rows.append(row)
    if not split_rows:
        return {"error": "all Gym split diagnostics failed"}

    def _collect(key: str) -> np.ndarray:
        return np.asarray([float(row[key]) for row in split_rows], dtype=np.float64)

    def _paired_excess(prior_key: str, heldout_key: str) -> dict[str, float]:
        prior = _collect(prior_key)
        heldout = _collect(heldout_key)
        return _summary(prior / np.maximum(heldout, 1e-12))

    residual_energy_excess = _collect("prior_residual_energy_median_ratio_vs_train") / np.maximum(
        _collect("heldout_gym_residual_energy_median_ratio_vs_train"),
        1e-12,
    )
    residual_ratio_excess = _collect("prior_residual_ratio_median_ratio_vs_train") / np.maximum(
        _collect("heldout_gym_residual_ratio_median_ratio_vs_train"),
        1e-12,
    )
    residual_excess_geomean = np.sqrt(
        np.maximum(residual_energy_excess, 1e-12) * np.maximum(residual_ratio_excess, 1e-12)
    )
    nearest_excess = _collect("prior_nearest_gym_distance_median_ratio_vs_train") / np.maximum(
        _collect("heldout_gym_nearest_gym_distance_median_ratio_vs_train"),
        1e-12,
    )

    return {
        "interpretation": (
            "If held-out Gym is also high off-gym under split-fit PCA95, the empirical "
            "off-gym residual gate is partly over-narrow and should not be used as the "
            "only hard pass criterion."
        ),
        "n_splits": int(len(split_rows)),
        "split_fraction": "half_train_half_heldout",
        "metrics": {key: _summary(_collect(key)) for key in sorted(split_rows[0])},
        "prior_vs_heldout_excess": {
            "interpretation": (
                "Paired split ratios separate reference sparsity from a real prior residual problem. "
                "Values near 1 mean prior is no worse than held-out Gym under the same train PCA95 fit; "
                "values well above 1 are true residual excess and should drive generator changes."
            ),
            "nearest_gym_distance_median_excess": _summary(nearest_excess),
            "residual_energy_median_excess": _summary(residual_energy_excess),
            "residual_ratio_median_excess": _summary(residual_ratio_excess),
            "residual_energy_ratio_geomean_excess": _summary(residual_excess_geomean),
            "off_mean_excess": _paired_excess("prior_off_mean", "heldout_gym_off_mean"),
            "high_frac_excess": _paired_excess(
                "prior_high_frac_vs_train_q75",
                "heldout_gym_high_frac_vs_train_q75",
            ),
        },
        "mechanism_read": {
            "raw_off_gym_gate_is_over_narrow_for_small_gym_reference": bool(
                np.mean(_collect("heldout_gym_high_frac_vs_train_q75")) >= 0.50
            ),
            "primary_true_prior_excess": "residual_energy_ratio_geomean_excess",
            "primary_true_prior_excess_q50": float(np.quantile(residual_excess_geomean, 0.50)),
            "nearest_distance_excess_q50": float(np.quantile(nearest_excess, 0.50)),
            "interpretation": (
                "If nearest-distance excess is modest but residual excess is high, prior is near Gym "
                "samples in observed-feature z-space but lies outside Gym's PCA95 residual geometry. "
                "The next generator target is residual covariance/observable dynamics, not selector "
                "quota or terminal-only tuning."
            ),
        },
    }


def _load_fixed_h_list(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "fixed_env_group" / "prior_fixed_h_list.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"expected list in {path}")
    return [item if isinstance(item, dict) else {} for item in payload]


def _common_fair_scores(gym_dir: Path, prior_dir: Path) -> dict[str, Any]:
    runs = {
        "gym": _load_run(gym_dir, arm="gym"),
        "prior": _load_run(prior_dir, arm="prior"),
    }
    common_names = _align_common_features({"gym": runs["gym"], "prior_full": runs["prior"], "prior_q90": runs["prior"]})
    subset_names = [name for name in common_names if _feature_subset_keep(name, "fair_observed_behavior")]
    if not subset_names:
        raise RuntimeError("fair observed subset has no common features")
    matrices = {name: _select_columns(run, subset_names) for name, run in runs.items()}
    space = _fit_gym_space(matrices["gym"], subset_names)
    transformed = {name: _transform(matrix, space) for name, matrix in matrices.items()}
    metric_by_run = {
        "gym": _metrics(transformed["gym"], space, reference=True),
        "prior": _metrics(transformed["prior"], space, reference=False),
    }
    scores: dict[str, dict[str, np.ndarray]] = {}
    components: dict[str, dict[str, np.ndarray]] = {}
    for name in ("gym", "prior"):
        composite, comps = _score_metrics(metric_by_run[name], metric_by_run["gym"])
        scores[name] = {"all": composite}
        components[name] = comps
        for family, keys in SCORE_FAMILIES.items():
            scores[name][family] = np.mean(np.vstack([comps[key] for key in keys]), axis=0)
    return {
        "subset_names": subset_names,
        "gym_matrix": matrices["gym"],
        "prior_matrix": matrices["prior"],
        "space": space,
        "transformed": transformed,
        "raw_metrics": metric_by_run,
        "scores": scores,
        "components": components,
    }


def run_audit(*, gym_dir: Path, prior_dir: Path, output_dir: Path) -> dict[str, Any]:
    fair = _common_fair_scores(gym_dir, prior_dir)
    prior_rows0 = _read_jsonl(_sidecar_path(prior_dir, 0))
    prior_rows3 = _read_jsonl(_sidecar_path(prior_dir, 3))
    h_list = _load_fixed_h_list(prior_dir)
    n = min(len(prior_rows0), len(prior_rows3), len(fair["scores"]["prior"]["all"]))
    if h_list:
        n = min(n, len(h_list))
    score = np.asarray(fair["scores"]["prior"]["all"][:n], dtype=np.float64)
    off_gym = np.asarray(fair["scores"]["prior"]["off_gym_novelty"][:n], dtype=np.float64)
    residual = np.asarray(fair["components"]["prior"]["residual_energy"][:n], dtype=np.float64)
    raw_nearest = np.asarray(fair["raw_metrics"]["prior"]["nearest_gym_distance"][:n], dtype=np.float64)
    raw_residual_energy = np.asarray(fair["raw_metrics"]["prior"]["residual_energy"][:n], dtype=np.float64)
    raw_residual_ratio = np.asarray(fair["raw_metrics"]["prior"]["residual_ratio"][:n], dtype=np.float64)
    gym_score = np.asarray(fair["scores"]["gym"]["all"], dtype=np.float64)
    gym_q25 = float(np.quantile(gym_score, 0.25))
    gym_q75 = float(np.quantile(gym_score, 0.75))
    low_mask = score <= gym_q25
    high_mask = score >= gym_q75

    joined: list[dict[str, Any]] = []
    source_values: dict[str, list[float]] = {key: [] for key in NUMERIC_SOURCE_KEYS}
    side_values: dict[str, list[float]] = {f"u0:{key}": [] for key in SIDECAR_KEYS}
    side_values.update({f"d:{key}": [] for key in SIDECAR_KEYS})
    delta_training = []
    for idx in range(n):
        h = h_list[idx] if idx < len(h_list) else {}
        row0 = prior_rows0[idx]
        row3 = prior_rows3[idx]
        entry: dict[str, Any] = {
            "env_idx": int(row0.get("env_idx", idx)),
            "fair_score": float(score[idx]),
            "fair_off_gym_score": float(off_gym[idx]),
            "fair_residual_energy_percentile": float(residual[idx]),
            "fair_nearest_gym_distance_raw": float(raw_nearest[idx]),
            "fair_residual_energy_raw": float(raw_residual_energy[idx]),
            "fair_residual_ratio_raw": float(raw_residual_ratio[idx]),
            "fair_band": "low" if low_mask[idx] else ("high" if high_mask[idx] else "mid"),
        }
        for key in NUMERIC_SOURCE_KEYS:
            # These fields are realized by env construction after source h
            # sampling.  For constrained dim policies and reward-path summaries,
            # the frozen h may contain raw sampler latents, while the sidecar row
            # contains the actual tensor layout/gain used by rollout.
            if key in REALIZED_SIDECAR_SOURCE_KEYS:
                value = _as_float(row0.get(key, h.get(key)))
            else:
                value = _as_float(h.get(key, row0.get(key)))
            entry[f"h:{key}"] = value
            source_values[key].append(value)
        for key in SIDECAR_KEYS:
            u0 = _as_float(row0.get(key))
            u3 = _as_float(row3.get(key))
            entry[f"u0:{key}"] = u0
            entry[f"d:{key}"] = u3 - u0 if math.isfinite(u0) and math.isfinite(u3) else float("nan")
            side_values[f"u0:{key}"].append(entry[f"u0:{key}"])
            side_values[f"d:{key}"].append(entry[f"d:{key}"])
        delta_training.append(entry.get("d:training_reward_sum", float("nan")))
        joined.append(entry)

    def _correlation_table(values: dict[str, list[float]], target: np.ndarray) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for key, vals in values.items():
            arr = np.asarray(vals[:n], dtype=np.float64)
            rows.append(
                {
                    "key": key,
                    "pearson": _pearson(arr, target),
                    "spearman": _spearman(arr, target),
                    "abs_spearman": abs(_spearman(arr, target)) if math.isfinite(_spearman(arr, target)) else float("nan"),
                    "summary": _summary(arr),
                    "high_minus_mid_mean": (
                        float(np.nanmean(arr[high_mask]) - np.nanmean(arr[~high_mask]))
                        if np.any(high_mask) and np.any(~high_mask)
                        else float("nan")
                    ),
                }
            )
        rows.sort(key=lambda r: (-1.0 if not math.isfinite(r["abs_spearman"]) else -float(r["abs_spearman"]), r["key"]))
        return rows

    report = {
        "audit_entry": "phase2_fair_residual_source_audit",
        "inputs": {
            "gym_dir": str(gym_dir),
            "prior_dir": str(prior_dir),
        },
        "fair_score_reference": {
            "gym_q25": gym_q25,
            "gym_q75": gym_q75,
            "prior_low_count": int(np.sum(low_mask)),
            "prior_mid_count": int(np.sum((~low_mask) & (~high_mask))),
            "prior_high_count": int(np.sum(high_mask)),
            "prior_score": _summary(score),
            "prior_off_gym_score": _summary(off_gym),
            "prior_residual_energy_percentile": _summary(residual),
            "prior_nearest_gym_distance_raw": _summary(raw_nearest),
            "prior_residual_energy_raw": _summary(raw_residual_energy),
            "prior_residual_ratio_raw": _summary(raw_residual_ratio),
            "training_reward_delta": _summary(np.asarray(delta_training, dtype=np.float64)),
        },
        "nearest_gym_delta_summary": _nearest_gym_delta_summary(fair),
        "density_overlap_summary": _density_overlap_summary(fair),
        "gym_lowtail_feature_target_summary": _gym_lowtail_feature_target_summary(fair),
        "exact_scm_selected_vs_gym_low_delta_summary": _selected_vs_gym_low_delta_summary(
            fair,
            joined,
            source_key="h:exact_scm_gym_lowtail_family_selected",
        ),
        "lowest_prior_component_blockers": _lowest_prior_component_blockers(fair, joined),
        "residual_covariance_block_audit": _residual_covariance_block_audit(fair, joined),
        "gym_split_generalization_summary": _gym_split_generalization_summary(fair),
        "gym_split_residual_feature_contribution_summary": (
            _gym_split_residual_feature_contribution_summary(fair)
        ),
        "source_correlations_with_fair_score": _correlation_table(source_values, score)[:30],
        "source_correlations_with_off_gym_score": _correlation_table(source_values, off_gym)[:30],
        "source_correlations_with_residual_energy": _correlation_table(source_values, residual)[:30],
        "source_correlations_with_raw_residual_energy": _correlation_table(source_values, raw_residual_energy)[:30],
        "source_correlations_with_raw_residual_ratio": _correlation_table(source_values, raw_residual_ratio)[:30],
        "source_correlations_with_raw_nearest_gym_distance": _correlation_table(source_values, raw_nearest)[:30],
        "sidecar_correlations_with_raw_residual_energy": _correlation_table(side_values, raw_residual_energy)[:40],
        "sidecar_correlations_with_raw_residual_ratio": _correlation_table(side_values, raw_residual_ratio)[:40],
        "sidecar_correlations_with_fair_score": _correlation_table(side_values, score)[:40],
        "source_group_summaries": _source_group_summaries(
            joined,
            (
                "h:exact_scm_gym_lowtail_family_selected",
                "h:action_dim",
                "h:obs_dim",
                "h:state_dim",
                "h:noise_dim",
                "h:zero_pad_dim",
            ),
        ),
        "top_high_fair_envs": sorted(joined, key=lambda r: -float(r["fair_score"]))[:12],
        "top_lowest_fair_envs": sorted(joined, key=lambda r: float(r["fair_score"]))[:12],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "fair_residual_source_audit.json"
    json_path.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gym-dir", type=Path, default=DEFAULT_GYM_DIR)
    parser.add_argument("--prior-dir", type=Path, default=DEFAULT_PRIOR_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    report = run_audit(gym_dir=args.gym_dir, prior_dir=args.prior_dir, output_dir=args.output_dir)
    print(
        json.dumps(
            {
                "json": str(args.output_dir / "fair_residual_source_audit.json"),
                "prior_low_count": report["fair_score_reference"]["prior_low_count"],
                "prior_high_count": report["fair_score_reference"]["prior_high_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
