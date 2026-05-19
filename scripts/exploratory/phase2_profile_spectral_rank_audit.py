#!/usr/bin/env python
"""Spectral rank audit for Gym/prior profile sidecar runs.

This is a read-only diagnostic.  It compares effective rank on common
per-environment behavioral sidecar features and separately audits prior-only
profile/hyperparameter labels.  It does not train, sample environments, or
change selector behavior.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ART = Path("/home/chen/RLPFN/artifacts")
DEFAULT_GYM_DIR = ART / "phase2_profile_v2l_uniform_transfer_gym_sidecar_u16_b512_advnorm_0515"
DEFAULT_PRIOR_FULL_DIR = (
    ART / "phase2_profile_v2n_uniform_terminal_gym_baseline_full_sidecar_u16_b512_advnorm_0515"
)
DEFAULT_PRIOR_Q90_DIR = (
    ART / "phase2_profile_v2n_uniform_terminal_gym_baseline_q90_sidecar_u16_b512_advnorm_0515"
)
DEFAULT_SELECTOR_CSV = (
    ART
    / "phase2_profile_band_selector_v2n_prior_pack_uniform_terminal_gym_baseline_modes_seed9730_0515"
    / "selected_profile_band_prior_envs.csv"
)
DEFAULT_OUTPUT_DIR = ART / "phase2_profile_v2n_spectral_rank_audit_0515"


COMMON_FEATURE_CANDIDATES = [
    "action_dim",
    "obs_dim",
    "state_dim",
    "done_count",
    "first_done_step",
    "truncated_count",
    "raw_reward_sum",
    "raw_reward_mean",
    "raw_reward_std",
    "raw_reward_q10",
    "raw_reward_q50",
    "raw_reward_q90",
    "training_reward_sum",
    "training_reward_mean",
    "training_reward_std",
    "training_reward_q10",
    "training_reward_q50",
    "training_reward_q90",
    "reward_sum",
    "reward_mean",
    "reward_std",
    "input_reward_token_absmax",
    "input_reward_token_mean",
    "input_reward_token_std",
    "action_norm_mean",
    "action_norm_std",
    "action_value_absmax",
    "action_value_absmean",
    "action_value_mean",
    "action_value_std",
    "action_per_dim_temporal_std_mean",
    "action_per_dim_temporal_std_q10",
    "action_per_dim_temporal_std_q50",
    "action_per_dim_temporal_std_q90",
    "action_temporal_delta_norm_mean",
    "unit_clipped_action_norm_mean",
    "unit_clipped_action_norm_std",
    "unit_clipped_action_value_absmean",
    "raw_policy_action_absmax_to_unit_clip",
    "raw_policy_action_unit_clip_excess_absmean",
    "raw_policy_action_unit_clip_saturation_fraction",
    "next_state_rms",
    "next_state_step_l2_delta_mean",
    "next_state_step_l2_delta_q90",
    "next_state_step_l2_delta_to_rms_ratio",
    "obs_per_dim_rms_mean",
    "obs_per_dim_std_mean",
    "obs_per_dim_std_q90",
    "obs_value_absmax",
    "obs_value_mean",
    "obs_value_std",
]

PRIOR_PROFILE_FEATURES = [
    "locomotion_witness",
    "low_energy_not_ctrl_only",
    "sign_or_phase_sensitive",
    "state_conditioned_energy_injection",
    "strict_feedback_candidate",
    "pendulum_short_temporal_settle_guard",
    "pendulum_axis_support",
    "pendulum_settle_yield_repair_planned",
    "terminal_reset_count_target",
    "balanced_reward_action_input_gain_fraction",
    "state_dim",
    "obs_dim",
    "action_dim",
]


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
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _sidecar_path(run_dir: Path, update: int) -> Path:
    files = sorted((run_dir / "semantic_sidecars").glob(f"*_update_{update:06d}_envs.jsonl"))
    if not files:
        raise FileNotFoundError(f"missing sidecar update {update} under {run_dir}")
    return files[0]


def _float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _bool_float(value: Any) -> float:
    text = str(value).strip().lower()
    return 1.0 if text in {"1", "true", "yes", "y"} else 0.0


def _rank_stats(matrix: np.ndarray, feature_names: list[str]) -> dict[str, Any]:
    x = np.asarray(matrix, dtype=np.float64)
    finite_cols = np.all(np.isfinite(x), axis=0)
    x = x[:, finite_cols]
    names = [name for name, keep in zip(feature_names, finite_cols) if keep]
    if x.size == 0 or x.shape[0] < 2:
        return {
            "n": int(matrix.shape[0]),
            "feature_count": 0,
            "kept_feature_count": 0,
            "raw_rank": 0,
            "effective_rank_entropy": 0.0,
            "participation_rank": 0.0,
            "stable_rank": 0.0,
            "rank90": 0,
            "rank95": 0,
            "rank99": 0,
            "top_energy_fraction": [],
            "kept_features": names,
        }
    x = x - np.mean(x, axis=0, keepdims=True)
    std = np.std(x, axis=0, keepdims=True)
    keep = (std[0] > 1e-12)
    x = x[:, keep] / std[:, keep]
    names = [name for name, yes in zip(names, keep) if yes]
    if x.size == 0:
        return {
            "n": int(matrix.shape[0]),
            "feature_count": int(matrix.shape[1]),
            "kept_feature_count": 0,
            "raw_rank": 0,
            "effective_rank_entropy": 0.0,
            "participation_rank": 0.0,
            "stable_rank": 0.0,
            "rank90": 0,
            "rank95": 0,
            "rank99": 0,
            "top_energy_fraction": [],
            "kept_features": names,
        }
    _, singular_values, _ = np.linalg.svd(x, full_matrices=False)
    eig = singular_values**2
    total = float(np.sum(eig))
    if total <= 0.0:
        weights = np.zeros_like(eig)
    else:
        weights = eig / total
    positive = weights[weights > 0]
    entropy_rank = float(np.exp(-np.sum(positive * np.log(positive)))) if len(positive) else 0.0
    participation = float((np.sum(eig) ** 2) / np.sum(eig**2)) if np.sum(eig**2) > 0 else 0.0
    stable = float(np.sum(eig) / np.max(eig)) if len(eig) and np.max(eig) > 0 else 0.0
    cume = np.cumsum(weights)
    return {
        "n": int(matrix.shape[0]),
        "feature_count": int(matrix.shape[1]),
        "kept_feature_count": int(x.shape[1]),
        "raw_rank": int(np.sum(singular_values > 1e-10)),
        "effective_rank_entropy": entropy_rank,
        "participation_rank": participation,
        "stable_rank": stable,
        "rank90": int(np.searchsorted(cume, 0.90) + 1) if len(cume) else 0,
        "rank95": int(np.searchsorted(cume, 0.95) + 1) if len(cume) else 0,
        "rank99": int(np.searchsorted(cume, 0.99) + 1) if len(cume) else 0,
        "top_energy_fraction": [float(v) for v in weights[:10]],
        "kept_features": names,
    }


def _common_matrix(rows0: list[dict[str, Any]], rows3: list[dict[str, Any]]) -> tuple[np.ndarray, list[str]]:
    by0 = {int(row["env_idx"]): row for row in rows0}
    by3 = {int(row["env_idx"]): row for row in rows3}
    env_indices = sorted(set(by0) & set(by3))
    names: list[str] = []
    cols: list[list[float]] = []
    for key in COMMON_FEATURE_CANDIDATES:
        state_values = [_float(by0[idx].get(key)) for idx in env_indices]
        delta_values = [_float(by3[idx].get(key)) - _float(by0[idx].get(key)) for idx in env_indices]
        if all(math.isfinite(v) for v in state_values):
            names.append(f"u0:{key}")
            cols.append(state_values)
        if all(math.isfinite(v) for v in delta_values):
            names.append(f"d:{key}")
            cols.append(delta_values)
    if not cols:
        return np.empty((len(env_indices), 0)), []
    return np.asarray(cols, dtype=np.float64).T, names


def _delta_metric(rows0: list[dict[str, Any]], rows3: list[dict[str, Any]], *, arm: str) -> np.ndarray:
    by0 = {int(row["env_idx"]): row for row in rows0}
    by3 = {int(row["env_idx"]): row for row in rows3}
    values = []
    for idx in sorted(set(by0) & set(by3)):
        key = "raw_reward_sum" if arm == "gym" else "reward_env_sum"
        if key not in by0[idx] or key not in by3[idx]:
            key = "raw_reward_sum"
        values.append(_float(by3[idx].get(key)) - _float(by0[idx].get(key)))
    return np.asarray(values, dtype=np.float64)


def _prior_profile_matrix(selector_csv: Path, rule: str) -> tuple[np.ndarray, list[str]]:
    rows = [row for row in _read_csv(selector_csv) if str(row.get("rule")) == rule]
    rows.sort(key=lambda row: int(float(row.get("profile_band_rank", row.get("_source_index", 0)))))
    cols: list[list[float]] = []
    names: list[str] = []
    for key in PRIOR_PROFILE_FEATURES:
        if key in {
            "locomotion_witness",
            "low_energy_not_ctrl_only",
            "sign_or_phase_sensitive",
            "state_conditioned_energy_injection",
            "strict_feedback_candidate",
            "pendulum_short_temporal_settle_guard",
            "pendulum_axis_support",
            "pendulum_settle_yield_repair_planned",
        }:
            values = [_bool_float(row.get(key)) for row in rows]
        else:
            values = [_float(row.get(key)) for row in rows]
        if all(math.isfinite(v) for v in values):
            cols.append(values)
            names.append(key)
    terminal_values = [_float(row.get("terminal_reset_count_target")) for row in rows]
    if all(math.isfinite(v) for v in terminal_values):
        for idx, (lo, hi) in enumerate([(0, 20), (20, 40), (40, 60), (60, 80)]):
            if idx == 0:
                values = [1.0 if lo <= value <= hi else 0.0 for value in terminal_values]
            else:
                values = [1.0 if lo < value <= hi else 0.0 for value in terminal_values]
            cols.append(values)
            names.append(f"terminal_bin_{lo}_{hi}")
    if not cols:
        return np.empty((len(rows), 0)), []
    return np.asarray(cols, dtype=np.float64).T, names


def _run_audit(name: str, run_dir: Path, *, arm: str) -> dict[str, Any]:
    rows0 = _read_jsonl(_sidecar_path(run_dir, 0))
    rows3 = _read_jsonl(_sidecar_path(run_dir, 3))
    matrix, names = _common_matrix(rows0, rows3)
    deltas = _delta_metric(rows0, rows3, arm=arm)
    improved = deltas > 0
    return {
        "run_dir": str(run_dir),
        "arm": arm,
        "delta_mean": float(np.mean(deltas)),
        "delta_median": float(np.median(deltas)),
        "improved_count": int(np.sum(improved)),
        "n": int(len(deltas)),
        "common_behavior_all": _rank_stats(matrix, names),
        "common_behavior_improved": _rank_stats(matrix[improved], names) if np.any(improved) else None,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = {
        "gym": _run_audit("gym", Path(args.gym_dir).expanduser().resolve(), arm="gym"),
        "prior_full": _run_audit(
            "prior_full", Path(args.prior_full_dir).expanduser().resolve(), arm="prior"
        ),
        "prior_q90": _run_audit(
            "prior_q90", Path(args.prior_q90_dir).expanduser().resolve(), arm="prior"
        ),
    }
    selector_csv = Path(args.selector_csv).expanduser().resolve()
    for run_name, rule in [
        ("prior_full", "gym_full_obs_action_range"),
        ("prior_q90", "gym_q90_obs_action_range"),
    ]:
        profile_matrix, profile_names = _prior_profile_matrix(selector_csv, rule)
        improved_count = runs[run_name]["improved_count"]
        run_deltas = None
        # Re-read deltas so the profile subset uses the same env order.
        run_dir = Path(runs[run_name]["run_dir"])
        rows0 = _read_jsonl(_sidecar_path(run_dir, 0))
        rows3 = _read_jsonl(_sidecar_path(run_dir, 3))
        run_deltas = _delta_metric(rows0, rows3, arm="prior")
        improved = run_deltas > 0
        runs[run_name]["prior_profile_all"] = _rank_stats(profile_matrix, profile_names)
        runs[run_name]["prior_profile_improved"] = (
            _rank_stats(profile_matrix[improved], profile_names) if np.any(improved) else None
        )
        runs[run_name]["profile_rows"] = int(profile_matrix.shape[0])
        runs[run_name]["profile_improved_count"] = int(improved_count)

    gym_rank = runs["gym"]["common_behavior_improved"]["effective_rank_entropy"]
    gym_participation = runs["gym"]["common_behavior_improved"]["participation_rank"]
    comparisons = {}
    for key in ["prior_full", "prior_q90"]:
        item = runs[key]
        er = item["common_behavior_improved"]["effective_rank_entropy"]
        pr = item["common_behavior_improved"]["participation_rank"]
        comparisons[key] = {
            "improved_count_vs_gym": float(item["improved_count"] / max(1, runs["gym"]["improved_count"])),
            "improved_effective_rank_vs_gym": float(er / gym_rank) if gym_rank > 0 else None,
            "improved_participation_rank_vs_gym": float(pr / gym_participation)
            if gym_participation > 0
            else None,
            "common_behavior_rank_not_below_gym_90pct": bool(er >= 0.90 * gym_rank),
            "improved_count_not_below_gym_minus_one": bool(
                item["improved_count"] >= runs["gym"]["improved_count"] - 1
            ),
        }
    hard_read = {
        "spectral_dimension_gate_pass": bool(
            all(
                comparisons[key]["common_behavior_rank_not_below_gym_90pct"]
                and comparisons[key]["improved_count_not_below_gym_minus_one"]
                for key in comparisons
            )
        ),
        "interpretation": (
            "pass means prior improved subsets are not materially lower-dimensional than Gym "
            "under common behavioral sidecar features; profile-label rank is reported as a "
            "prior-internal collapse guard, not a direct Gym comparator"
        ),
    }
    report = {
        "analysis_entry": "phase2_profile_spectral_rank_audit",
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
            "selector_csv": str(selector_csv),
        },
        "runs": runs,
        "comparisons": comparisons,
        "hard_read": hard_read,
    }
    json_path = out_dir / "profile_spectral_rank_audit.json"
    md_path = out_dir / "profile_spectral_rank_audit.md"
    json_path.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Profile Spectral Rank Audit",
        "",
        "Read-only spectral audit over common Gym/prior sidecar behavior features. "
        "Effective rank is entropy rank over standardized singular-value energy.",
        "",
        f"- spectral dimension gate pass: `{hard_read['spectral_dimension_gate_pass']}`",
        "",
        "## Common Behavior Rank",
        "",
        "| run | improved | delta mean | all e-rank | improved e-rank | improved/gym | all participation | improved participation | rank95 improved |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key in ["gym", "prior_full", "prior_q90"]:
        item = runs[key]
        all_rank = item["common_behavior_all"]
        imp_rank = item["common_behavior_improved"]
        ratio = 1.0 if key == "gym" else comparisons[key]["improved_effective_rank_vs_gym"]
        lines.append(
            f"| `{key}` | {item['improved_count']}/{item['n']} | {item['delta_mean']:+.2f} | "
            f"{all_rank['effective_rank_entropy']:.2f} | {imp_rank['effective_rank_entropy']:.2f} | "
            f"{ratio:.2f} | {all_rank['participation_rank']:.2f} | "
            f"{imp_rank['participation_rank']:.2f} | {imp_rank['rank95']} |"
        )
    lines.extend(
        [
            "",
            "## Prior Profile Label Rank",
            "",
            "| run | all e-rank | improved e-rank | all participation | improved participation | rank95 improved |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for key in ["prior_full", "prior_q90"]:
        item = runs[key]
        all_rank = item["prior_profile_all"]
        imp_rank = item["prior_profile_improved"]
        lines.append(
            f"| `{key}` | {all_rank['effective_rank_entropy']:.2f} | "
            f"{imp_rank['effective_rank_entropy']:.2f} | {all_rank['participation_rank']:.2f} | "
            f"{imp_rank['participation_rank']:.2f} | {imp_rank['rank95']} |"
        )
    lines.extend(
        [
            "",
            "## Read",
            "",
            "- The common-behavior spectrum is the fair Gym/prior comparator.",
            "- Prior profile-label spectrum is a collapse guard for selector/hyperparameter diversity.",
            "- This audit should gate long fit_model runs together with terminal uniformity, mode counts, and pack-to-fit_model parity.",
            "",
            "## Files",
            "",
            f"- JSON: `{json_path}`",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"json": str(json_path), "md": str(md_path), **hard_read}, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gym-dir", default=str(DEFAULT_GYM_DIR))
    parser.add_argument("--prior-full-dir", default=str(DEFAULT_PRIOR_FULL_DIR))
    parser.add_argument("--prior-q90-dir", default=str(DEFAULT_PRIOR_Q90_DIR))
    parser.add_argument("--selector-csv", default=str(DEFAULT_SELECTOR_CSV))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
