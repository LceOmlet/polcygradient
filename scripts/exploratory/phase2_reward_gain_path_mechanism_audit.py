#!/usr/bin/env python
"""Read-only reward input-gain path mechanism audit.

This audit follows the actual computation chain instead of proposing a new
counterfactual.  It joins the reward input-group gain fractions recorded by the
training collector with rollout sidecar features and relative-complexity scores.
The goal is to decide whether reward input allocation is a stable formula node
behind the observed low-tail/off-gym failure before changing the generator.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.exploratory.phase2_profile_spectral_rank_audit import _sidecar_path  # noqa: E402


ART = Path("/home/chen/RLPFN/artifacts")
DEFAULT_SELECTED_CSV = (
    ART
    / "phase2_profile_v2p_dimcoupled_profile_label_join_independence_audit_0516"
    / "profile_labeled_selected_prior_envs.csv"
)
DEFAULT_RELATIVE_CSV = (
    ART
    / "phase2_profile_v2p_activation_semantics_fix_relative_complexity_audit_0516"
    / "profile_relative_complexity_per_env.csv"
)
DEFAULT_GYM_DIR = ART / "phase2_profile_v2l_uniform_transfer_gym_sidecar_u16_b512_advnorm_0515"
DEFAULT_PRIOR_FULL_DIR = ART / "phase2_profile_v2p_activation_semantics_fix_full_opt_u4_b512_tkl003_0516"
DEFAULT_PRIOR_Q90_DIR = ART / "phase2_profile_v2p_activation_semantics_fix_q90_opt_u4_b512_tkl003_0516"
DEFAULT_OUTPUT_DIR = ART / "phase2_profile_v2p_reward_gain_path_mechanism_audit_0516"

RUN_RULES = {
    "prior_full": "gym_full_obs_action_range",
    "prior_q90": "gym_q90_obs_action_range",
}

GAIN_KEYS = (
    "reward_state_input_gain_fraction",
    "reward_action_input_gain_fraction",
    "reward_noise_input_gain_fraction",
    "reward_state_to_action_gain_ratio",
    "reward_state_to_noise_gain_ratio",
)

SELECTED_GAIN_KEYS = (
    "reward_state_input_gain_fraction",
    "reward_action_input_gain_fraction",
    "reward_noise_input_gain_fraction",
    "reward_state_to_action_gain_ratio",
    "balanced_reward_state_input_gain_fraction",
    "balanced_reward_action_input_gain_fraction",
    "balanced_reward_noise_input_gain_fraction",
    "balanced_reward_state_to_action_gain_ratio",
)

STRUCTURE_KEYS = (
    "state_dims",
    "obs_dims",
    "action_dims",
    "terminal_reset_count_target",
    "ctrl_reward_weight",
)

SIDECAR_KEYS = (
    "obs_per_dim_std_q90",
    "obs_per_dim_std_mean",
    "obs_value_std",
    "raw_reward_std",
    "input_reward_token_std",
    "reward_env_std",
    "next_state_step_l2_delta_to_rms_ratio",
    "done_count",
    "action_norm_mean",
    "action_temporal_delta_norm_mean",
)

RELATIVE_KEYS = (
    "full_complexity_compatibility_score",
    "in_gym_complexity_score",
    "off_gym_novelty_score",
    "residual_energy_percentile",
    "residual_ratio_percentile",
    "nearest_gym_distance_percentile",
    "delta",
)

TARGETS = (
    "u0:obs_per_dim_std_q90",
    "u0:raw_reward_std",
    "u0:input_reward_token_std",
    "u0:next_state_step_l2_delta_to_rms_ratio",
    "u0:done_count",
    "off_gym_novelty_score",
    "full_complexity_compatibility_score",
    "delta",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _quantiles(values: list[float], qs: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9)) -> dict[str, Any]:
    x = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    if x.size == 0:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    out: dict[str, Any] = {
        "n": int(x.size),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }
    for q in qs:
        out[f"q{int(q * 100):02d}"] = float(np.quantile(x, q))
    return out


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    idx = 0
    while idx < len(values):
        end = idx + 1
        while end < len(values) and values[order[end]] == values[order[idx]]:
            end += 1
        ranks[order[idx:end]] = 0.5 * (idx + end - 1) + 1.0
        idx = end
    return ranks


def _corr(x: list[float], y: list[float], *, spearman: bool) -> float | None:
    xx = np.asarray(x, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    keep = np.isfinite(xx) & np.isfinite(yy)
    xx = xx[keep]
    yy = yy[keep]
    if xx.size < 4:
        return None
    if spearman:
        xx = _rankdata(xx)
        yy = _rankdata(yy)
    xx = xx - float(np.mean(xx))
    yy = yy - float(np.mean(yy))
    denom = float(np.sqrt(np.sum(xx**2) * np.sum(yy**2)))
    if denom <= 1e-12:
        return None
    return float(np.sum(xx * yy) / denom)


def _load_progress_update(run_dir: Path, update: int) -> dict[str, Any]:
    rows = _read_jsonl(run_dir / "prior_progress.jsonl")
    for row in rows:
        update_value = row.get("update_idx", None)
        if update_value is None or isinstance(update_value, dict):
            update_value = row.get("update", -1)
        if isinstance(update_value, dict):
            update_value = update_value.get("update_idx", update_value.get("idx", -1))
        if int(update_value) == int(update):
            return row
    raise RuntimeError(f"missing update {update} in {run_dir / 'prior_progress.jsonl'}")


def _collector_value(collector: dict[str, Any], key: str, env_idx: int) -> float:
    value = collector.get(key)
    if isinstance(value, list):
        if env_idx >= len(value):
            return float("nan")
        return _float(value[env_idx])
    return _float(value)


def _load_relative(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for row in _read_csv(path):
        run = row.get("run", "")
        env_idx = int(float(row.get("env_idx", "nan")))
        out[(run, env_idx)] = {key: _float(row.get(key)) for key in RELATIVE_KEYS}
        out[(run, env_idx)]["improved"] = _bool(row.get("improved"))
        out[(run, env_idx)]["complexity_band"] = row.get("complexity_band", "")
    return out


def _selected_by_rule(path: Path, rule: str) -> list[dict[str, str]]:
    rows = [row for row in _read_csv(path) if row.get("rule") == rule]
    if rows and "profile_env_idx" in rows[0] and rows[0].get("profile_env_idx", "") != "":
        return sorted(rows, key=lambda row: int(float(row.get("profile_env_idx", "nan"))))
    if rows and "profile_band_rank" in rows[0] and rows[0].get("profile_band_rank", "") != "":
        return sorted(rows, key=lambda row: int(float(row.get("profile_band_rank", "nan"))))
    return rows


def _sidecar_by_update(run_dir: Path, update: int) -> dict[int, dict[str, Any]]:
    return {int(row["env_idx"]): row for row in _read_jsonl(_sidecar_path(run_dir, update))}


def _add_sidecar(row: dict[str, Any], sidecar0: dict[int, dict[str, Any]], sidecar3: dict[int, dict[str, Any]]) -> None:
    env_idx = int(row["env_idx"])
    r0 = sidecar0[env_idx]
    r3 = sidecar3[env_idx]
    for key in SIDECAR_KEYS:
        v0 = _float(r0.get(key))
        v3 = _float(r3.get(key))
        if math.isfinite(v0):
            row[f"u0:{key}"] = v0
        if math.isfinite(v3):
            row[f"u3:{key}"] = v3
        if math.isfinite(v0) and math.isfinite(v3):
            row[f"d:{key}"] = v3 - v0


def _build_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    relative = _load_relative(Path(args.relative_csv))
    run_dirs = {
        "prior_full": Path(args.prior_full_dir),
        "prior_q90": Path(args.prior_q90_dir),
    }
    selected = {run: _selected_by_rule(Path(args.selected_csv), rule) for run, rule in RUN_RULES.items()}
    rows: list[dict[str, Any]] = []
    selected_mismatch: dict[str, Any] = {}
    for run, run_dir in run_dirs.items():
        progress0 = _load_progress_update(run_dir, 0)
        collector = progress0["collector"]
        sidecar0 = _sidecar_by_update(run_dir, 0)
        sidecar3 = _sidecar_by_update(run_dir, 3)
        max_delta_by_key: dict[str, float] = {}
        for env_idx in sorted(sidecar0):
            row: dict[str, Any] = {
                "run": run,
                "rule": RUN_RULES[run],
                "env_idx": env_idx,
            }
            for key in GAIN_KEYS:
                row[f"collector:{key}"] = _collector_value(collector, key, env_idx)
            for key in STRUCTURE_KEYS:
                row[f"collector:{key}"] = _collector_value(collector, key, env_idx)
            row.update(relative[(run, env_idx)])
            _add_sidecar(row, sidecar0, sidecar3)
            if env_idx < len(selected[run]):
                selected_row = selected[run][env_idx]
                selected_balance_applied = _bool(selected_row.get("selected_balance_applied"))
                for key in SELECTED_GAIN_KEYS:
                    value = _float(selected_row.get(key))
                    if math.isfinite(value):
                        row[f"selected:{key}"] = value
                selected_actual_pairs = {
                    "reward_state_input_gain_fraction": (
                        "balanced_reward_state_input_gain_fraction"
                        if selected_balance_applied
                        else "reward_state_input_gain_fraction"
                    ),
                    "reward_action_input_gain_fraction": (
                        "balanced_reward_action_input_gain_fraction"
                        if selected_balance_applied
                        else "reward_action_input_gain_fraction"
                    ),
                    "reward_noise_input_gain_fraction": (
                        "balanced_reward_noise_input_gain_fraction"
                        if selected_balance_applied
                        else "reward_noise_input_gain_fraction"
                    ),
                    "reward_state_to_action_gain_ratio": (
                        "balanced_reward_state_to_action_gain_ratio"
                        if selected_balance_applied
                        else "reward_state_to_action_gain_ratio"
                    ),
                }
                for collector_key, selected_key in selected_actual_pairs.items():
                    value = _float(selected_row.get(selected_key))
                    if math.isfinite(value):
                        row[f"selected_actual:{collector_key}"] = value
                        cvalue = row.get(f"collector:{collector_key}")
                        if isinstance(cvalue, float) and math.isfinite(cvalue):
                            delta = abs(cvalue - value)
                            max_delta_by_key[collector_key] = max(max_delta_by_key.get(collector_key, 0.0), float(delta))
                row["selected:selected_balance_applied"] = 1.0 if selected_balance_applied else 0.0
                for key in ("state_dim", "obs_dim", "action_dim", "noise_dim", "zero_pad_dim"):
                    value = _float(selected_row.get(key))
                    if math.isfinite(value):
                        row[f"selected:{key}"] = value
            rows.append(row)
        selected_mismatch[run] = {
            "selected_count": len(selected[run]),
            "collector_count": len(sidecar0),
            "max_abs_delta_vs_selected_balanced": max_delta_by_key,
            "hard_pass_1e_5": all(delta <= 1e-5 for delta in max_delta_by_key.values()),
        }
    gym0 = _sidecar_by_update(Path(args.gym_dir), 0)
    gym3 = _sidecar_by_update(Path(args.gym_dir), 3)
    gym_reference: dict[str, Any] = {}
    for key in SIDECAR_KEYS:
        gym_reference[f"u0:{key}"] = _quantiles([_float(row.get(key)) for row in gym0.values()])
        gym_reference[f"u3:{key}"] = _quantiles([_float(row.get(key)) for row in gym3.values()])
    gym_reference["selected_mismatch"] = selected_mismatch
    return rows, gym_reference


def _top_correlations(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    predictors = [f"collector:{key}" for key in GAIN_KEYS + STRUCTURE_KEYS]
    predictors += [
        "selected:state_dim",
        "selected:obs_dim",
        "selected:action_dim",
        "selected:noise_dim",
        "selected:zero_pad_dim",
        "selected:selected_balance_applied",
    ]
    out: dict[str, list[dict[str, Any]]] = {}
    for target in TARGETS:
        y = [_float(row.get(target)) for row in rows]
        items: list[dict[str, Any]] = []
        for predictor in predictors:
            x = [_float(row.get(predictor)) for row in rows]
            pearson = _corr(x, y, spearman=False)
            spearman = _corr(x, y, spearman=True)
            if pearson is None and spearman is None:
                continue
            score = max(abs(pearson or 0.0), abs(spearman or 0.0))
            if score < 0.18:
                continue
            items.append(
                {
                    "predictor": predictor,
                    "pearson": pearson,
                    "spearman": spearman,
                    "score": score,
                }
            )
        out[target] = sorted(items, key=lambda item: item["score"], reverse=True)[:12]
    return out


def _split_summary(rows: list[dict[str, Any]], gym_reference: dict[str, Any]) -> dict[str, Any]:
    obs_q25 = gym_reference["u0:obs_per_dim_std_q90"].get("q25")
    out: dict[str, Any] = {}
    if isinstance(obs_q25, float):
        below = [row for row in rows if _float(row.get("u0:obs_per_dim_std_q90")) < obs_q25]
        compatible = [row for row in rows if _float(row.get("u0:obs_per_dim_std_q90")) >= obs_q25]
        keys = [f"collector:{key}" for key in GAIN_KEYS + STRUCTURE_KEYS] + [
            "u0:raw_reward_std",
            "u0:next_state_step_l2_delta_to_rms_ratio",
            "off_gym_novelty_score",
            "full_complexity_compatibility_score",
            "delta",
        ]
        out["obs_below_gym_q25_vs_compatible"] = {
            "gym_obs_q25": obs_q25,
            "below_count": len(below),
            "compatible_count": len(compatible),
            "below_means": {key: _quantiles([_float(row.get(key)) for row in below]).get("mean") for key in keys},
            "compatible_means": {
                key: _quantiles([_float(row.get(key)) for row in compatible]).get("mean") for key in keys
            },
        }
    return out


def _quartile_splits(rows: list[dict[str, Any]]) -> dict[str, Any]:
    predictors = tuple(f"collector:{key}" for key in GAIN_KEYS[:3])
    out: dict[str, Any] = {}
    for predictor in predictors:
        vals = np.asarray([_float(row.get(predictor)) for row in rows], dtype=np.float64)
        keep = np.isfinite(vals)
        if int(np.sum(keep)) < 8:
            continue
        q25 = float(np.quantile(vals[keep], 0.25))
        q75 = float(np.quantile(vals[keep], 0.75))
        low_rows = [row for row, keep_i, value in zip(rows, keep, vals) if keep_i and value <= q25]
        high_rows = [row for row, keep_i, value in zip(rows, keep, vals) if keep_i and value >= q75]
        item: dict[str, Any] = {"q25": q25, "q75": q75, "low_n": len(low_rows), "high_n": len(high_rows)}
        for target in TARGETS:
            low_mean = _quantiles([_float(row.get(target)) for row in low_rows]).get("mean")
            high_mean = _quantiles([_float(row.get(target)) for row in high_rows]).get("mean")
            item[target] = {
                "low_q_mean": low_mean,
                "high_q_mean": high_mean,
                "high_minus_low": (
                    float(high_mean) - float(low_mean)
                    if isinstance(low_mean, float) and isinstance(high_mean, float)
                    else None
                ),
            }
        out[predictor] = item
    return out


def _run_summary(rows: list[dict[str, Any]], gym_reference: dict[str, Any]) -> dict[str, Any]:
    summary = {
        "n": len(rows),
        "improved_count": int(sum(bool(row.get("improved")) for row in rows)),
        "delta": _quantiles([_float(row.get("delta")) for row in rows]),
        "gain_stats": {key: _quantiles([_float(row.get(f"collector:{key}")) for row in rows]) for key in GAIN_KEYS},
        "target_stats": {target: _quantiles([_float(row.get(target)) for row in rows]) for target in TARGETS},
        "top_correlations": _top_correlations(rows),
        "quartile_splits": _quartile_splits(rows),
        "split_summary": _split_summary(rows, gym_reference),
    }
    obs_q25 = gym_reference["u0:obs_per_dim_std_q90"].get("q25")
    if isinstance(obs_q25, float):
        summary["obs_below_gym_q25_count"] = int(
            sum(_float(row.get("u0:obs_per_dim_std_q90")) < obs_q25 for row in rows)
        )
    return summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows, gym_reference = _build_rows(args)
    by_run = {run: [row for row in rows if row["run"] == run] for run in RUN_RULES}
    report = {
        "analysis_entry": "phase2_reward_gain_path_mechanism_audit",
        "contract": {
            "read_only": True,
            "does_not_sample_envs": True,
            "does_not_train": True,
            "does_not_modify_generator": True,
            "mechanism_chain": [
                "collector reward input-gain metadata",
                "semantic rollout sidecar",
                "relative complexity residual",
            ],
        },
        "inputs": {
            "selected_csv": str(Path(args.selected_csv).resolve()),
            "relative_csv": str(Path(args.relative_csv).resolve()),
            "gym_dir": str(Path(args.gym_dir).resolve()),
            "prior_full_dir": str(Path(args.prior_full_dir).resolve()),
            "prior_q90_dir": str(Path(args.prior_q90_dir).resolve()),
        },
        "collector_selected_parity": gym_reference.pop("selected_mismatch"),
        "gym_reference": gym_reference,
        "run_summaries": {run: _run_summary(run_rows, gym_reference) for run, run_rows in by_run.items()},
        "mechanism_interpretation": {
            "candidate_formula_node": "reward input-group allocation before reward_unit output",
            "not_a_final_fix": "post-h reward_group_balance_runtime is only a diagnostic/intervention path; final fix should be a source/generator allocation change if this audit remains stable.",
        },
    }
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    _write_csv(out_dir / "reward_gain_path_mechanism_rows.csv", rows, fieldnames)
    (out_dir / "reward_gain_path_mechanism_audit.json").write_text(
        json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-csv", default=str(DEFAULT_SELECTED_CSV))
    parser.add_argument("--relative-csv", default=str(DEFAULT_RELATIVE_CSV))
    parser.add_argument("--gym-dir", default=str(DEFAULT_GYM_DIR))
    parser.add_argument("--prior-full-dir", default=str(DEFAULT_PRIOR_FULL_DIR))
    parser.add_argument("--prior-q90-dir", default=str(DEFAULT_PRIOR_Q90_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = run(args)
    print(json.dumps(_json_safe(report), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
