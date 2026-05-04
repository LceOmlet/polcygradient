import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_OUTPUT_DIR = (
    "/home/chen/RLPFN/artifacts/"
    "phase2_pretrain_reward_basin_gate_validation_0429"
)

DEFAULT_PRIOR_RUNS = [
    (
        "balanced_midnoise_candidate",
        "/home/chen/RLPFN/artifacts/"
        "phase2_trainability_prior_balanced_midnoise_fixedlist_n8_u200_obsnorm_0429",
    ),
    (
        "population_random_control",
        "/home/chen/RLPFN/artifacts/"
        "phase2_trainability_prior_random_control_fixedlist_n8_u200_obsnorm_0429",
    ),
]


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.generic):
        return _json_safe(value.item())
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
        return {"count": 0, "mean": None, "q10": None, "q50": None, "q90": None, "min": None, "max": None}
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "q10": float(np.quantile(arr, 0.10)),
        "q50": float(np.quantile(arr, 0.50)),
        "q90": float(np.quantile(arr, 0.90)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _pearson(xs: list[Any], ys: list[Any]) -> float | None:
    pairs = [(_finite_float(x), _finite_float(y)) for x, y in zip(xs, ys)]
    pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
    if len(pairs) < 3:
        return None
    x_arr = np.asarray([x for x, _ in pairs], dtype=np.float64)
    y_arr = np.asarray([y for _, y in pairs], dtype=np.float64)
    if float(np.std(x_arr)) <= 1e-12 or float(np.std(y_arr)) <= 1e-12:
        return None
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def _linear_slope(values: list[Any]) -> float | None:
    ys = [_finite_float(v) for v in values]
    pairs = [(float(idx), y) for idx, y in enumerate(ys) if y is not None]
    if len(pairs) < 2:
        return None
    x_arr = np.asarray([x for x, _ in pairs], dtype=np.float64)
    y_arr = np.asarray([y for _, y in pairs], dtype=np.float64)
    x_centered = x_arr - float(np.mean(x_arr))
    denom = float(np.sum(x_centered * x_centered))
    if denom <= 1e-12:
        return None
    return float(np.sum(x_centered * (y_arr - float(np.mean(y_arr)))) / denom)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _load_sidecars(run_dir: Path, arm: str = "prior") -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = {}
    sidecar_dir = run_dir / "semantic_sidecars"
    for path in sorted(sidecar_dir.glob(f"{arm}_update_*_envs.jsonl")):
        token = path.stem.split("_update_", 1)[1].split("_", 1)[0]
        out[int(token)] = _read_jsonl(path)
    return out


def _load_h_list(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "fixed_env_group" / "prior_fixed_h_list.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _estimate_ctrl_sum(row: dict[str, Any], h: dict[str, Any], *, n_steps: int = 256) -> float:
    ctrl_enabled = bool(
        float(h.get("_ctrl_reward_enable_u", 1.0)) < float(h.get("ctrl_reward_enable_prob", 0.0))
    )
    if not ctrl_enabled:
        return 0.0
    action_mean = float(row.get("action_value_mean") or 0.0)
    action_std = float(row.get("action_value_std") or 0.0)
    mean_action_sq = action_std * action_std + action_mean * action_mean
    return -float(h.get("ctrl_reward_weight") or 0.0) * float(n_steps) * mean_action_sq


def _augment_row(row: dict[str, Any], h: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    ctrl_sum = (
        float(row["reward_ctrl_sum"])
        if _finite_float(row.get("reward_ctrl_sum")) is not None
        else _estimate_ctrl_sum(row, h)
    )
    reward_sum = float(row.get("reward_sum") or 0.0)
    out["ctrl_sum_est"] = float(ctrl_sum)
    out["env_sum_est"] = (
        float(row["reward_env_sum"])
        if _finite_float(row.get("reward_env_sum")) is not None
        else float(reward_sum - ctrl_sum)
    )
    out["has_exact_reward_components"] = bool(_finite_float(row.get("reward_env_sum")) is not None)
    out["ctrl_enabled"] = bool(float(h.get("_ctrl_reward_enable_u", 1.0)) < float(h.get("ctrl_reward_enable_prob", 0.0)))
    out["ctrl_weight"] = float(h.get("ctrl_reward_weight") or 0.0)
    out["h_noise_std"] = _finite_float(h.get("noise_std"))
    out["h_init_std"] = _finite_float(h.get("init_std"))
    out["h_init_action_std"] = _finite_float(h.get("init_action_std"))
    out["h_constrained_obs_u"] = _finite_float(h.get("_constrained_obs_u"))
    out["h_constrained_noise_u"] = _finite_float(h.get("_constrained_noise_u"))
    out["h_action_dim"] = _finite_float(h.get("action_dim"))
    out["h_state_dim"] = _finite_float(h.get("state_dim"))
    out["h_noise_dim"] = _finite_float(h.get("noise_dim"))
    return out


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [_finite_float(row.get(key)) for row in rows]
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


def _share(numer: Any, denom_parts: list[Any]) -> float | None:
    num = _finite_float(numer)
    vals = [_finite_float(v) for v in denom_parts]
    vals = [abs(v) for v in vals if v is not None]
    denom = float(sum(vals))
    if num is None or denom <= 1e-12:
        return None
    return float(abs(num) / denom)


def _group_worst_abs_share(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [_finite_float(row.get(key)) for row in rows]
    vals = [v for v in vals if v is not None]
    denom = float(sum(abs(v) for v in vals))
    if denom <= 1e-12:
        return None
    return float(max(abs(v) for v in vals) / denom)


def _run_rows(label: str, run_dir: Path, early_count: int, late_count: int) -> list[dict[str, Any]]:
    sidecars = _load_sidecars(run_dir, "prior")
    h_list = _load_h_list(run_dir)
    updates = sorted(sidecars)
    if not updates:
        raise ValueError(f"No sidecars found in {run_dir}")
    first_update = updates[0]
    early_updates = updates[: max(1, min(int(early_count), len(updates)))]
    late_updates = updates[-max(1, min(int(late_count), len(updates))) :]
    by_env: dict[int, dict[int, dict[str, Any]]] = {}
    for update_idx, rows in sidecars.items():
        for row in rows:
            env_idx = int(row["env_idx"])
            by_env.setdefault(env_idx, {})[int(update_idx)] = _augment_row(row, h_list[env_idx])

    out = []
    for env_idx, rows_by_update in sorted(by_env.items()):
        first = rows_by_update[first_update]
        early = [rows_by_update[u] for u in early_updates if u in rows_by_update]
        late = [rows_by_update[u] for u in late_updates if u in rows_by_update]
        if not late:
            continue
        row = {
            "run_label": str(label),
            "run_dir": str(run_dir),
            "env_idx": int(env_idx),
            "env_seed": first.get("env_group_env_seed"),
            "available_update_count": int(len(rows_by_update)),
            "first_update": int(first_update),
            "last_update": int(max(rows_by_update)),
            "gate_total_reward_sum": first.get("reward_sum"),
            "gate_total_reward_mean": first.get("reward_mean"),
            "gate_total_reward_q10": first.get("reward_q10"),
            "gate_total_reward_q50": first.get("reward_q50"),
            "gate_total_reward_q90": first.get("reward_q90"),
            "gate_total_reward_std": first.get("reward_std"),
            "gate_env_reward_sum": first.get("reward_env_sum"),
            "gate_env_reward_mean": first.get("reward_env_mean"),
            "gate_env_reward_q10": first.get("reward_env_q10"),
            "gate_env_reward_q50": first.get("reward_env_q50"),
            "gate_env_reward_q90": first.get("reward_env_q90"),
            "gate_env_reward_positive_fraction": first.get("reward_env_positive_fraction"),
            "gate_ctrl_reward_sum": first.get("reward_ctrl_sum"),
            "gate_ctrl_reward_mean": first.get("reward_ctrl_mean"),
            "gate_ctrl_reward_std": first.get("reward_ctrl_std"),
            "gate_ctrl_reward_q10": first.get("reward_ctrl_q10"),
            "gate_ctrl_reward_q50": first.get("reward_ctrl_q50"),
            "gate_ctrl_reward_q90": first.get("reward_ctrl_q90"),
            "gate_action_norm_mean": first.get("action_norm_mean"),
            "gate_action_value_std": first.get("action_value_std"),
            "sidecar_action_dim": first.get("action_dim"),
            "sidecar_obs_dim": first.get("obs_dim"),
            "sidecar_state_dim": first.get("state_dim"),
            "reward_state_input_gain_fraction": first.get("reward_state_input_gain_fraction"),
            "reward_action_input_gain_fraction": first.get("reward_action_input_gain_fraction"),
            "reward_noise_input_gain_fraction": first.get("reward_noise_input_gain_fraction"),
            "reward_state_to_action_gain_ratio": first.get("reward_state_to_action_gain_ratio"),
            "reward_state_to_noise_gain_ratio": first.get("reward_state_to_noise_gain_ratio"),
            "gate_env_q90_or_total_proxy": (
                first.get("reward_env_q90")
                if _finite_float(first.get("reward_env_q90")) is not None
                else first.get("reward_q90")
            ),
            "gate_env_positive_fraction_or_q90_proxy": (
                first.get("reward_env_positive_fraction")
                if _finite_float(first.get("reward_env_positive_fraction")) is not None
                else (1.0 if float(first.get("reward_q90") or 0.0) > 0.0 else 0.0)
            ),
            "gate_total_reward_positive_tail_proxy_q90_gt0": bool(float(first.get("reward_q90") or 0.0) > 0.0),
            "gate_reject_total_q90_lt0": bool(float(first.get("reward_q90") or 0.0) < 0.0),
            "gate_reject_env_q90_or_total_proxy_lt0": bool(
                float(
                    first.get("reward_env_q90")
                    if _finite_float(first.get("reward_env_q90")) is not None
                    else first.get("reward_q90")
                )
                < 0.0
            ),
            "has_exact_reward_components": bool(first.get("has_exact_reward_components", False)),
            "gate_ctrl_sum_est": first.get("ctrl_sum_est"),
            "gate_env_sum_est": first.get("env_sum_est"),
            "early_total_sum_mean": _mean(early, "reward_sum"),
            "early_env_sum_est_mean": _mean(early, "env_sum_est"),
            "early_ctrl_sum_est_mean": _mean(early, "ctrl_sum_est"),
            "early_total_sum_slope": _linear_slope([row.get("reward_sum") for row in early]),
            "early_env_sum_est_slope": _linear_slope([row.get("env_sum_est") for row in early]),
            "early_ctrl_sum_est_slope": _linear_slope([row.get("ctrl_sum_est") for row in early]),
            "late_total_sum_mean": _mean(late, "reward_sum"),
            "late_env_sum_est_mean": _mean(late, "env_sum_est"),
            "late_ctrl_sum_est_mean": _mean(late, "ctrl_sum_est"),
            "late_total_sum_slope": _linear_slope([row.get("reward_sum") for row in late]),
            "late_env_sum_est_slope": _linear_slope([row.get("env_sum_est") for row in late]),
            "late_ctrl_sum_est_slope": _linear_slope([row.get("ctrl_sum_est") for row in late]),
            "late_reward_q90_mean": _mean(late, "reward_q90"),
            "late_reward_q50_mean": _mean(late, "reward_q50"),
            "late_env_reward_q90_mean": _mean(late, "reward_env_q90"),
            "late_env_reward_positive_fraction_mean": _mean(late, "reward_env_positive_fraction"),
            "late_ctrl_reward_q50_mean": _mean(late, "reward_ctrl_q50"),
            "late_action_norm_mean": _mean(late, "action_norm_mean"),
            "late_corr_action_norm_reward_mean": _mean(late, "corr_action_norm_reward"),
            "late_done_count_mean": _mean(late, "done_count"),
            "delta_late_minus_early_total_sum": None,
            "delta_late_minus_early_env_sum_est": None,
            "delta_late_minus_early_ctrl_sum_est": None,
            "single_env_late_negative_share_abs": None,
        }
        for key in [
            "ctrl_enabled",
            "ctrl_weight",
            "h_noise_std",
            "h_init_std",
            "h_init_action_std",
            "h_constrained_obs_u",
            "h_constrained_noise_u",
            "h_action_dim",
            "h_state_dim",
            "h_noise_dim",
        ]:
            row[key] = first.get(key)
        row["gate_ctrl_abs_share_total_components"] = _share(
            row.get("gate_ctrl_sum_est"),
            [row.get("gate_ctrl_sum_est"), row.get("gate_env_sum_est")],
        )
        row["gate_ctrl_abs_over_env_abs"] = _share(
            row.get("gate_ctrl_sum_est"),
            [row.get("gate_env_sum_est")],
        )
        row["late_ctrl_abs_share_total_components"] = _share(
            row.get("late_ctrl_sum_est_mean"),
            [row.get("late_ctrl_sum_est_mean"), row.get("late_env_sum_est_mean")],
        )
        row["late_ctrl_abs_over_env_abs"] = _share(
            row.get("late_ctrl_sum_est_mean"),
            [row.get("late_env_sum_est_mean")],
        )
        gate_pos = _finite_float(row.get("gate_env_reward_positive_fraction"))
        gate_q90 = _finite_float(row.get("gate_env_q90_or_total_proxy"))
        gate_reward_std = _finite_float(row.get("gate_total_reward_std"))
        row["gate_reject_env_no_tail"] = bool(
            (gate_q90 is not None and gate_q90 < 0.0)
            or (gate_pos is not None and gate_pos < 0.20)
        )
        row["gate_reject_ctrl_dominance_proxy"] = bool(
            (_finite_float(row.get("gate_ctrl_abs_share_total_components")) or 0.0) > 0.50
            or (_finite_float(row.get("gate_ctrl_abs_over_env_abs")) or 0.0) > 1.0
        )
        row["gate_reject_no_tail_high_spread"] = bool(
            row["gate_reject_env_no_tail"]
            and gate_reward_std is not None
            and gate_reward_std > 0.30
        )
        if row["late_total_sum_mean"] is not None and row["early_total_sum_mean"] is not None:
            row["delta_late_minus_early_total_sum"] = float(row["late_total_sum_mean"]) - float(row["early_total_sum_mean"])
        if row["late_env_sum_est_mean"] is not None and row["early_env_sum_est_mean"] is not None:
            row["delta_late_minus_early_env_sum_est"] = (
                float(row["late_env_sum_est_mean"]) - float(row["early_env_sum_est_mean"])
            )
        if row["late_ctrl_sum_est_mean"] is not None and row["early_ctrl_sum_est_mean"] is not None:
            row["delta_late_minus_early_ctrl_sum_est"] = (
                float(row["late_ctrl_sum_est_mean"]) - float(row["early_ctrl_sum_est_mean"])
            )
        out.append(row)
    total_negative_abs = sum(
        abs(float(row["late_total_sum_mean"]))
        for row in out
        if _finite_float(row.get("late_total_sum_mean")) is not None and float(row["late_total_sum_mean"]) < 0.0
    )
    for row in out:
        late_total = _finite_float(row.get("late_total_sum_mean"))
        row["single_env_late_negative_share_abs"] = (
            abs(late_total) / max(total_negative_abs, 1e-12)
            if late_total is not None and late_total < 0.0
            else 0.0
        )
        row["gate_reject_large_negative_share_gt0p5"] = bool(row["single_env_late_negative_share_abs"] > 0.5)
        late_total = _finite_float(row.get("late_total_sum_mean"))
        late_env = _finite_float(row.get("late_env_sum_est_mean"))
        late_ctrl_share = _finite_float(row.get("late_ctrl_abs_share_total_components")) or 0.0
        row["late_ctrl_drag_env_positive_total_negative"] = bool(
            late_total is not None
            and late_env is not None
            and late_total < 0.0
            and late_env > 0.0
            and late_ctrl_share > 0.50
        )
        row["late_reject_early_env_slope_nonpositive_high_contribution"] = bool(
            row["gate_reject_large_negative_share_gt0p5"]
            and (_finite_float(row.get("early_env_sum_est_slope")) or 0.0) <= 0.0
        )
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(_json_safe(row))


def _gate_effect(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    rejected = [row for row in rows if bool(row.get(key))]
    kept = [row for row in rows if not bool(row.get(key))]
    return {
        "gate_key": str(key),
        "env_count": int(len(rows)),
        "rejected_count": int(len(rejected)),
        "kept_count": int(len(kept)),
        "rejected_late_total_sum": _stats([row.get("late_total_sum_mean") for row in rejected]),
        "kept_late_total_sum": _stats([row.get("late_total_sum_mean") for row in kept]),
        "rejected_late_env_sum_est": _stats([row.get("late_env_sum_est_mean") for row in rejected]),
        "kept_late_env_sum_est": _stats([row.get("late_env_sum_est_mean") for row in kept]),
        "rejected_gate_q90": _stats([row.get("gate_total_reward_q90") for row in rejected]),
        "kept_gate_q90": _stats([row.get("gate_total_reward_q90") for row in kept]),
        "rejected_gate_env_q90_or_proxy": _stats([row.get("gate_env_q90_or_total_proxy") for row in rejected]),
        "kept_gate_env_q90_or_proxy": _stats([row.get("gate_env_q90_or_total_proxy") for row in kept]),
        "rejected_late_ctrl_share": _stats([row.get("late_ctrl_abs_share_total_components") for row in rejected]),
        "kept_late_ctrl_share": _stats([row.get("late_ctrl_abs_share_total_components") for row in kept]),
    }


def _group_summary(run_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "gate_group_total_sum": _stats([row.get("gate_total_reward_sum") for row in run_rows]),
        "gate_group_env_sum_est": _stats([row.get("gate_env_sum_est") for row in run_rows]),
        "gate_group_ctrl_sum_est": _stats([row.get("gate_ctrl_sum_est") for row in run_rows]),
        "late_group_total_sum": _stats([row.get("late_total_sum_mean") for row in run_rows]),
        "late_group_env_sum_est": _stats([row.get("late_env_sum_est_mean") for row in run_rows]),
        "late_group_ctrl_sum_est": _stats([row.get("late_ctrl_sum_est_mean") for row in run_rows]),
        "gate_worst_abs_share_total": _group_worst_abs_share(run_rows, "gate_total_reward_sum"),
        "gate_worst_abs_share_env": _group_worst_abs_share(run_rows, "gate_env_sum_est"),
        "gate_worst_abs_share_ctrl": _group_worst_abs_share(run_rows, "gate_ctrl_sum_est"),
        "late_worst_abs_share_total": _group_worst_abs_share(run_rows, "late_total_sum_mean"),
        "late_worst_abs_share_env": _group_worst_abs_share(run_rows, "late_env_sum_est_mean"),
        "late_worst_abs_share_ctrl": _group_worst_abs_share(run_rows, "late_ctrl_sum_est_mean"),
        "late_ctrl_drag_env_positive_total_negative_count": int(
            sum(bool(row.get("late_ctrl_drag_env_positive_total_negative")) for row in run_rows)
        ),
        "late_large_negative_share_count": int(
            sum(bool(row.get("gate_reject_large_negative_share_gt0p5")) for row in run_rows)
        ),
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_run: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_run.setdefault(str(row["run_label"]), []).append(row)
    runs = {}
    for label, run_rows in sorted(by_run.items()):
        q90 = [row.get("gate_total_reward_q90") for row in run_rows]
        late_total = [row.get("late_total_sum_mean") for row in run_rows]
        late_env = [row.get("late_env_sum_est_mean") for row in run_rows]
        runs[label] = {
            "env_count": int(len(run_rows)),
            "gate_q90_vs_late_total_pearson": _pearson(q90, late_total),
            "gate_q90_vs_late_env_est_pearson": _pearson(q90, late_env),
            "late_total_sum": _stats(late_total),
            "late_env_sum_est": _stats(late_env),
            "late_ctrl_sum_est": _stats([row.get("late_ctrl_sum_est_mean") for row in run_rows]),
            "late_ctrl_abs_share_total_components": _stats(
                [row.get("late_ctrl_abs_share_total_components") for row in run_rows]
            ),
            "early_env_sum_est_slope": _stats([row.get("early_env_sum_est_slope") for row in run_rows]),
            "early_total_sum_slope": _stats([row.get("early_total_sum_slope") for row in run_rows]),
            "gate_total_reward_q90": _stats(q90),
            "gate_env_q90_or_total_proxy": _stats([row.get("gate_env_q90_or_total_proxy") for row in run_rows]),
            "group_summary": _group_summary(run_rows),
            "has_exact_reward_components": bool(any(row.get("has_exact_reward_components") for row in run_rows)),
            "gate_total_reward_q50": _stats([row.get("gate_total_reward_q50") for row in run_rows]),
            "gate_effects": [
                _gate_effect(run_rows, "gate_reject_total_q90_lt0"),
                _gate_effect(run_rows, "gate_reject_env_q90_or_total_proxy_lt0"),
                _gate_effect(run_rows, "gate_reject_env_no_tail"),
                _gate_effect(run_rows, "gate_reject_no_tail_high_spread"),
                _gate_effect(run_rows, "gate_reject_ctrl_dominance_proxy"),
                _gate_effect(run_rows, "late_ctrl_drag_env_positive_total_negative"),
                _gate_effect(run_rows, "gate_reject_large_negative_share_gt0p5"),
                _gate_effect(run_rows, "late_reject_early_env_slope_nonpositive_high_contribution"),
            ],
            "worst_late_total_envs": sorted(
                run_rows,
                key=lambda row: float(row.get("late_total_sum_mean") or 0.0),
            )[:5],
        }
    all_q90 = [row.get("gate_total_reward_q90") for row in rows]
    all_late = [row.get("late_total_sum_mean") for row in rows]
    return {
        "audit_entry": "phase2_pretrain_reward_basin_gate_validation",
        "contract": (
            "Read-only validation from existing fixed-group sidecars. It estimates ctrl/env sums "
            "from action moments and frozen_h ctrl settings; exact reward_env quantiles require "
            "future component logging in the canonical runner."
        ),
        "limitations": [
            "Current sidecars store total reward quantiles, not per-step reward_env quantiles.",
            "ctrl_sum/env_sum decomposition is exact only for the current configs with reward_transform=none and survival=0; those conditions are checked manually by frozen_h fields.",
            "positive-tail fraction is represented here by the q90>0 proxy because per-step sign counts were not stored.",
            "n=8 per run, so correlations are evidence for a gate hypothesis, not a population-level proof.",
        ],
        "combined": {
            "env_count": int(len(rows)),
            "gate_q90_vs_late_total_pearson": _pearson(all_q90, all_late),
            "gate_total_reward_q90": _stats(all_q90),
            "late_total_sum": _stats(all_late),
        },
        "runs": runs,
    }


def run_validation(runs: list[tuple[str, str]], output_dir: str, early_count: int, late_count: int) -> dict[str, Any]:
    out_dir = Path(output_dir).expanduser().resolve()
    all_rows: list[dict[str, Any]] = []
    for label, run_dir_str in runs:
        all_rows.extend(
            _run_rows(
                label,
                Path(run_dir_str).expanduser().resolve(),
                early_count=int(early_count),
                late_count=int(late_count),
            )
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "per_env_gate_rows.csv", all_rows)
    summary = _summarize(all_rows)
    summary["files"] = {
        "per_env_gate_rows_csv": str(out_dir / "per_env_gate_rows.csv"),
        "summary_json": str(out_dir / "summary.json"),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _parse_run_arg(values: list[str]) -> list[tuple[str, str]]:
    if not values:
        return DEFAULT_PRIOR_RUNS
    runs = []
    for value in values:
        if "=" not in value:
            raise ValueError("--run must be label=/path/to/run")
        label, path = value.split("=", 1)
        runs.append((label.strip(), path.strip()))
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate pretrain reward-basin gate against fixed-group outcomes.")
    parser.add_argument("--run", action="append", default=[], help="label=/path/to/prior-run; repeatable")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--early-count", type=int, default=10)
    parser.add_argument("--late-count", type=int, default=10)
    args = parser.parse_args()
    summary = run_validation(
        _parse_run_arg(args.run),
        output_dir=str(args.output_dir),
        early_count=int(args.early_count),
        late_count=int(args.late_count),
    )
    print(
        json.dumps(
            _json_safe(
                {
                    "summary_json": summary["files"]["summary_json"],
                    "per_env_gate_rows_csv": summary["files"]["per_env_gate_rows_csv"],
                    "combined_gate_q90_vs_late_total_pearson": summary["combined"]["gate_q90_vs_late_total_pearson"],
                    "runs": {
                        label: {
                            "env_count": run["env_count"],
                            "gate_q90_vs_late_total_pearson": run["gate_q90_vs_late_total_pearson"],
                        }
                        for label, run in summary["runs"].items()
                    },
                }
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
