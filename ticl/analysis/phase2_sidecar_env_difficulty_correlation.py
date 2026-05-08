import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_PRIOR_DIR = (
    "/home/chen/RLPFN/artifacts/"
    "phase2_prior_pack_semprobe_sampledgain_fixedgroup_n1024_u200_0428"
)
DISABLED_REASON = (
    "phase2_sidecar_env_difficulty_correlation is disabled: it is built around "
    "hand-written correlation metrics that are not official SB3/PPO diagnostics "
    "and have not been accepted as trusted parity evidence."
)


BASELINE_FEATURES = [
    "reward_state_input_gain_fraction",
    "reward_action_input_gain_fraction",
    "reward_noise_input_gain_fraction",
    "reward_state_to_action_gain_ratio",
    "reward_state_to_noise_gain_ratio",
    "next_state_step_l2_delta_to_rms_ratio",
    "obs_per_dim_std_mean",
    "obs_per_dim_rms_mean",
    "obs_value_absmax",
    "input_reward_token_std",
    "reward_std",
    "action_per_dim_temporal_std_mean",
    "action_temporal_delta_norm_mean",
    "corr_action_norm_reward",
    "corr_action_delta_norm_next_reward",
    "action_dim",
    "state_dim",
    "obs_dim",
]


OUTCOME_KEYS = [
    "reward_sum_delta_last_minus_first",
    "reward_sum_late_mean_minus_early_mean",
    "reward_sum_slope_per_update",
    "reward_std_delta_last_minus_first",
    "next_state_drift_ratio_delta_last_minus_first",
    "action_temporal_std_delta_last_minus_first",
    "late_reward_sum_mean",
    "early_reward_sum_mean",
    "last_reward_sum",
    "first_reward_sum",
]


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _stats(values: list[Any]) -> dict[str, Any]:
    arr = np.asarray([v for v in (_finite_float(v) for v in values) if v is not None], dtype=np.float64)
    if arr.size == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "q10": None,
            "q25": None,
            "q50": None,
            "q75": None,
            "q90": None,
            "max": None,
        }
    return {
        "count": int(arr.size),
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


def _pearson(xs: list[Any], ys: list[Any]) -> float | None:
    pairs = [(_finite_float(x), _finite_float(y)) for x, y in zip(xs, ys)]
    pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
    if len(pairs) < 8:
        return None
    x_arr = np.asarray([x for x, _ in pairs], dtype=np.float64)
    y_arr = np.asarray([y for _, y in pairs], dtype=np.float64)
    x0 = x_arr - float(np.mean(x_arr))
    y0 = y_arr - float(np.mean(y_arr))
    denom = float(np.linalg.norm(x0) * np.linalg.norm(y0))
    if denom <= 1e-12:
        return None
    return float(np.dot(x0, y0) / denom)


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(values, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        rank = 0.5 * (start + end - 1)
        ranks[order[start:end]] = rank
        start = end
    return ranks


def _spearman(xs: list[Any], ys: list[Any]) -> float | None:
    pairs = [(_finite_float(x), _finite_float(y)) for x, y in zip(xs, ys)]
    pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
    if len(pairs) < 8:
        return None
    x_arr = np.asarray([x for x, _ in pairs], dtype=np.float64)
    y_arr = np.asarray([y for _, y in pairs], dtype=np.float64)
    return _pearson(_rankdata(x_arr).tolist(), _rankdata(y_arr).tolist())


def _linear_slope(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    x_arr = np.asarray(xs, dtype=np.float64)
    y_arr = np.asarray(ys, dtype=np.float64)
    x0 = x_arr - float(np.mean(x_arr))
    denom = float(np.dot(x0, x0))
    if denom <= 1e-12:
        return None
    return float(np.dot(x0, y_arr - float(np.mean(y_arr))) / denom)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _load_sidecars(experiment_dir: Path, arm: str) -> dict[int, list[dict[str, Any]]]:
    sidecar_dir = experiment_dir / "semantic_sidecars"
    out: dict[int, list[dict[str, Any]]] = {}
    for path in sorted(sidecar_dir.glob(f"{arm}_update_*_envs.jsonl")):
        stem = path.stem
        update_token = stem.split("_update_", 1)[1].split("_", 1)[0]
        out[int(update_token)] = _read_jsonl(path)
    return out


def _build_per_env_rows(sidecars: dict[int, list[dict[str, Any]]], *, early_count: int, late_count: int) -> list[dict[str, Any]]:
    if not sidecars:
        return []
    updates = sorted(sidecars)
    first_update = updates[0]
    last_update = updates[-1]
    by_env: dict[int, dict[int, dict[str, Any]]] = {}
    for update_idx, rows in sidecars.items():
        for row in rows:
            by_env.setdefault(int(row["env_idx"]), {})[int(update_idx)] = row

    out: list[dict[str, Any]] = []
    early_updates = updates[: max(1, min(int(early_count), len(updates)))]
    late_updates = updates[-max(1, min(int(late_count), len(updates))) :]
    for env_idx, row_by_update in sorted(by_env.items()):
        if first_update not in row_by_update or last_update not in row_by_update:
            continue
        first = row_by_update[first_update]
        last = row_by_update[last_update]
        env_updates = [u for u in updates if u in row_by_update]
        reward_by_update = [
            _finite_float(row_by_update[u].get("reward_sum"))
            for u in env_updates
        ]
        valid_curve = [(float(u), float(v)) for u, v in zip(env_updates, reward_by_update) if v is not None]
        early_vals = [
            _finite_float(row_by_update[u].get("reward_sum"))
            for u in early_updates
            if u in row_by_update
        ]
        late_vals = [
            _finite_float(row_by_update[u].get("reward_sum"))
            for u in late_updates
            if u in row_by_update
        ]
        early_vals = [v for v in early_vals if v is not None]
        late_vals = [v for v in late_vals if v is not None]
        row: dict[str, Any] = {
            "env_idx": int(env_idx),
            "arm": first.get("arm"),
            "gym_env_id": first.get("gym_env_id"),
            "available_update_count": int(len(env_updates)),
            "first_update_idx": int(first_update),
            "last_update_idx": int(last_update),
            "first_reward_sum": _finite_float(first.get("reward_sum")),
            "last_reward_sum": _finite_float(last.get("reward_sum")),
            "early_reward_sum_mean": float(np.mean(early_vals)) if early_vals else None,
            "late_reward_sum_mean": float(np.mean(late_vals)) if late_vals else None,
            "reward_sum_slope_per_update": _linear_slope(
                [x for x, _ in valid_curve],
                [y for _, y in valid_curve],
            ),
            "reward_std_delta_last_minus_first": (
                float(last["reward_std"]) - float(first["reward_std"])
                if _finite_float(first.get("reward_std")) is not None and _finite_float(last.get("reward_std")) is not None
                else None
            ),
            "next_state_drift_ratio_delta_last_minus_first": (
                float(last["next_state_step_l2_delta_to_rms_ratio"]) - float(first["next_state_step_l2_delta_to_rms_ratio"])
                if _finite_float(first.get("next_state_step_l2_delta_to_rms_ratio")) is not None
                and _finite_float(last.get("next_state_step_l2_delta_to_rms_ratio")) is not None
                else None
            ),
            "action_temporal_std_delta_last_minus_first": (
                float(last["action_per_dim_temporal_std_mean"]) - float(first["action_per_dim_temporal_std_mean"])
                if _finite_float(first.get("action_per_dim_temporal_std_mean")) is not None
                and _finite_float(last.get("action_per_dim_temporal_std_mean")) is not None
                else None
            ),
        }
        first_reward = row["first_reward_sum"]
        last_reward = row["last_reward_sum"]
        early_mean = row["early_reward_sum_mean"]
        late_mean = row["late_reward_sum_mean"]
        row["reward_sum_delta_last_minus_first"] = (
            float(last_reward) - float(first_reward)
            if first_reward is not None and last_reward is not None
            else None
        )
        row["reward_sum_late_mean_minus_early_mean"] = (
            float(late_mean) - float(early_mean)
            if early_mean is not None and late_mean is not None
            else None
        )
        for key in BASELINE_FEATURES:
            row[f"baseline_{key}"] = first.get(key)
        out.append(row)
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(_json_safe(row))


def _correlations(per_env_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    features = [f"baseline_{key}" for key in BASELINE_FEATURES]
    for feature in features:
        xs = [row.get(feature) for row in per_env_rows]
        for outcome in OUTCOME_KEYS:
            ys = [row.get(outcome) for row in per_env_rows]
            rows.append(
                {
                    "feature": feature,
                    "outcome": outcome,
                    "pearson": _pearson(xs, ys),
                    "spearman": _spearman(xs, ys),
                    "x_stats": _stats(xs),
                    "y_stats": _stats(ys),
                }
            )
    rows.sort(key=lambda row: abs(row["spearman"]) if row.get("spearman") is not None else -1.0, reverse=True)
    return rows


def _quantile_effects(per_env_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for feature_key in [f"baseline_{key}" for key in BASELINE_FEATURES]:
        values = np.asarray([
            _finite_float(row.get(feature_key))
            for row in per_env_rows
        ], dtype=object)
        finite = np.asarray([v is not None for v in values], dtype=bool)
        if int(finite.sum()) < 20:
            continue
        finite_values = np.asarray([float(v) for v in values[finite]], dtype=np.float64)
        lo = float(np.quantile(finite_values, 0.20))
        hi = float(np.quantile(finite_values, 0.80))
        low_rows = [row for row in per_env_rows if (_finite_float(row.get(feature_key)) is not None and float(row[feature_key]) <= lo)]
        high_rows = [row for row in per_env_rows if (_finite_float(row.get(feature_key)) is not None and float(row[feature_key]) >= hi)]
        for outcome in OUTCOME_KEYS:
            low_vals = [row.get(outcome) for row in low_rows]
            high_vals = [row.get(outcome) for row in high_rows]
            low_stats = _stats(low_vals)
            high_stats = _stats(high_vals)
            low_mean = low_stats.get("mean")
            high_mean = high_stats.get("mean")
            rows.append(
                {
                    "feature": feature_key,
                    "outcome": outcome,
                    "low_threshold_q20": lo,
                    "high_threshold_q80": hi,
                    "low_count": len(low_rows),
                    "high_count": len(high_rows),
                    "low_mean": low_mean,
                    "high_mean": high_mean,
                    "high_minus_low_mean": (
                        float(high_mean) - float(low_mean)
                        if low_mean is not None and high_mean is not None
                        else None
                    ),
                    "low_stats": low_stats,
                    "high_stats": high_stats,
                }
            )
    rows.sort(
        key=lambda row: abs(row["high_minus_low_mean"])
        if row.get("high_minus_low_mean") is not None and row.get("outcome") == "reward_sum_late_mean_minus_early_mean"
        else -1.0,
        reverse=True,
    )
    return rows


def _quantile_trajectories(
    sidecars: dict[int, list[dict[str, Any]]],
    per_env_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    env_features = {int(row["env_idx"]): row for row in per_env_rows}
    for feature_key in [f"baseline_{key}" for key in BASELINE_FEATURES]:
        finite_pairs = [
            (env_idx, value)
            for env_idx, row in env_features.items()
            if (value := _finite_float(row.get(feature_key))) is not None
        ]
        if len(finite_pairs) < 20:
            continue
        values = np.asarray([value for _, value in finite_pairs], dtype=np.float64)
        lo = float(np.quantile(values, 0.20))
        hi = float(np.quantile(values, 0.80))
        low_envs = {env_idx for env_idx, value in finite_pairs if value <= lo}
        high_envs = {env_idx for env_idx, value in finite_pairs if value >= hi}
        for update_idx in sorted(sidecars):
            update_rows = sidecars[update_idx]
            for group_name, env_set in (("low20", low_envs), ("high20", high_envs)):
                group_rows = [row for row in update_rows if int(row["env_idx"]) in env_set]
                rows.append(
                    {
                        "feature": feature_key,
                        "group": group_name,
                        "update_idx": int(update_idx),
                        "feature_threshold_q20": lo,
                        "feature_threshold_q80": hi,
                        "env_count": len(group_rows),
                        "reward_sum_stats": _stats([row.get("reward_sum") for row in group_rows]),
                        "reward_std_stats": _stats([row.get("reward_std") for row in group_rows]),
                        "next_state_drift_ratio_stats": _stats([
                            row.get("next_state_step_l2_delta_to_rms_ratio")
                            for row in group_rows
                        ]),
                    }
                )
    return rows


def run_analysis(experiment_dir: str, arm: str, output_dir: str | None, early_count: int, late_count: int) -> dict[str, Any]:
    raise RuntimeError(DISABLED_REASON)
    exp_dir = Path(experiment_dir).expanduser().resolve()
    out_dir = Path(output_dir).expanduser().resolve() if output_dir else exp_dir / "analysis"
    sidecars = _load_sidecars(exp_dir, arm)
    per_env_rows = _build_per_env_rows(sidecars, early_count=int(early_count), late_count=int(late_count))
    corr_rows = _correlations(per_env_rows)
    quant_rows = _quantile_effects(per_env_rows)
    trajectory_rows = _quantile_trajectories(sidecars, per_env_rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / f"{arm}_per_env_trajectory_features.csv", per_env_rows)
    _write_csv(out_dir / f"{arm}_feature_outcome_correlations.csv", corr_rows)
    _write_csv(out_dir / f"{arm}_feature_quantile_effects.csv", quant_rows)
    _write_csv(out_dir / f"{arm}_feature_quantile_trajectories.csv", trajectory_rows)
    report = {
        "audit_entry": "phase2_sidecar_env_difficulty_correlation",
        "experiment_dir": str(exp_dir),
        "arm": str(arm),
        "output_dir": str(out_dir),
        "update_count": int(len(sidecars)),
        "updates": sorted(int(v) for v in sidecars.keys()),
        "env_count": int(len(per_env_rows)),
        "early_count": int(early_count),
        "late_count": int(late_count),
        "outcomes": {key: _stats([row.get(key) for row in per_env_rows]) for key in OUTCOME_KEYS},
        "baseline_features": {
            f"baseline_{key}": _stats([row.get(f"baseline_{key}") for row in per_env_rows])
            for key in BASELINE_FEATURES
        },
        "top_abs_spearman": corr_rows[:30],
        "top_late_minus_early_quantile_effects": [
            row for row in quant_rows if row.get("outcome") == "reward_sum_late_mean_minus_early_mean"
        ][:30],
        "files": {
            "per_env_trajectory_features_csv": str(out_dir / f"{arm}_per_env_trajectory_features.csv"),
            "feature_outcome_correlations_csv": str(out_dir / f"{arm}_feature_outcome_correlations.csv"),
            "feature_quantile_effects_csv": str(out_dir / f"{arm}_feature_quantile_effects.csv"),
            "feature_quantile_trajectories_csv": str(out_dir / f"{arm}_feature_quantile_trajectories.csv"),
            "summary_json": str(out_dir / f"{arm}_difficulty_correlation_summary.json"),
        },
    }
    (out_dir / f"{arm}_difficulty_correlation_summary.json").write_text(
        json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Correlate fixed-env sidecar features with per-env training outcomes.")
    parser.add_argument("--experiment-dir", type=str, default=DEFAULT_PRIOR_DIR)
    parser.add_argument("--arm", type=str, default="prior")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--early-count", type=int, default=3)
    parser.add_argument("--late-count", type=int, default=3)
    args = parser.parse_args()
    report = run_analysis(
        experiment_dir=args.experiment_dir,
        arm=args.arm,
        output_dir=args.output_dir or None,
        early_count=int(args.early_count),
        late_count=int(args.late_count),
    )
    print(json.dumps(_json_safe({
        "env_count": report["env_count"],
        "update_count": report["update_count"],
        "summary_json": report["files"]["summary_json"],
    }), sort_keys=True))


if __name__ == "__main__":
    main()
