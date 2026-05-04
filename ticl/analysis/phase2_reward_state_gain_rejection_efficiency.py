import argparse
import copy
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from ticl.analysis.critic_free_single_env_audit import (
    _build_audit_env_cfg,
    _prepare_audit_fixed_env_contract,
    _seed_all,
)
from ticl.analysis.fixed_env_h import summarize_fixed_env_h
from ticl.model_configs import get_model_default_config
from ticl.priors.environment_prior import EnvironmentPrior
from ticl.rlpfn_maintained_path import validate_rlpfn_maintained_path_config


def _json_safe(value: Any) -> Any:
    if torch.is_tensor(value):
        if value.ndim == 0:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, type):
        return value.__name__
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _activation_name(value: Any) -> str:
    if isinstance(value, type):
        return str(value.__name__).lower()
    text = str(value)
    for name in ("relu", "tanh", "sin", "identity"):
        if name in text.lower():
            return name
    return text


def _finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _extract_row(*, frozen_h_seed: int, attempt_seed: int, frozen_h: dict[str, Any], env: dict[str, Any]) -> dict[str, Any]:
    state_dim = int(env.get("state_dim", frozen_h.get("state_dim", 0)) or 0)
    action_dim = int(env.get("action_dim", frozen_h.get("action_dim", 0)) or 0)
    noise_dim = int(env.get("noise_dim", frozen_h.get("noise_dim", 0)) or 0)
    pad_dim = int(env.get("zero_pad_dim", frozen_h.get("zero_pad_dim", 0)) or 0)
    total_input = max(1, state_dim + action_dim + noise_dim + pad_dim)
    obs_dim = int(env.get("obs_dim", frozen_h.get("obs_dim", 0)) or 0)
    gain = _finite_float(env.get("reward_state_input_gain_fraction"))
    return {
        "frozen_h_seed": int(frozen_h_seed),
        "attempt_seed": int(attempt_seed),
        "reward_state_input_gain_fraction": gain,
        "reward_action_input_gain_fraction": _finite_float(env.get("reward_action_input_gain_fraction")),
        "reward_noise_input_gain_fraction": _finite_float(env.get("reward_noise_input_gain_fraction")),
        "reward_state_to_noise_gain_ratio": _finite_float(env.get("reward_state_to_noise_gain_ratio")),
        "reward_state_to_action_gain_ratio": _finite_float(env.get("reward_state_to_action_gain_ratio")),
        "legal": bool(math.isfinite(gain)),
        "state_dim": state_dim,
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "noise_dim": noise_dim,
        "zero_pad_dim": pad_dim,
        "total_input_dim": int(total_input),
        "state_dim_fraction": float(state_dim / total_input),
        "obs_dim_fraction_of_state": float(obs_dim / max(1, state_dim)),
        "action_dim_fraction": float(action_dim / total_input),
        "noise_dim_fraction": float(noise_dim / total_input),
        "zero_pad_dim_fraction": float(pad_dim / total_input),
        "activation": _activation_name(frozen_h.get("prior_mlp_activations")),
        "num_layers": int(frozen_h.get("num_layers", 0) or 0),
        "hidden_dim": int(frozen_h.get("prior_mlp_hidden_dim", 0) or 0),
        "init_std": _finite_float(frozen_h.get("init_std")),
        "noise_std": _finite_float(frozen_h.get("noise_std")),
        "reward_scale": _finite_float(frozen_h.get("reward_scale")),
        "ctrl_reward_enabled": bool(env.get("ctrl_reward_enabled", False)),
        "ctrl_reward_weight": _finite_float(env.get("ctrl_reward_weight", frozen_h.get("ctrl_reward_weight", 0.0))),
        "terminal_reset_enabled": bool(env.get("terminal_reset_enabled", False)),
        "terminal_reset_count_target": _finite_float(
            env.get("terminal_reset_count_target", frozen_h.get("terminal_reset_count_target", 0.0))
        ),
        "terminal_bonus_tanh_c": _finite_float(frozen_h.get("terminal_bonus_tanh_c")),
        "terminal_bonus_scale_min": _finite_float(frozen_h.get("terminal_bonus_scale_min")),
        "terminal_bonus_scale_max": _finite_float(frozen_h.get("terminal_bonus_scale_max")),
        "frozen_h_fingerprint": summarize_fixed_env_h(frozen_h)["frozen_h_fingerprint"],
    }


def _stats(values: list[float]) -> dict[str, float | int | None]:
    arr = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=np.float64)
    if arr.size == 0:
        return {"count": 0}
    qs = np.quantile(arr, [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0])
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(qs[0]),
        "q10": float(qs[1]),
        "q25": float(qs[2]),
        "q50": float(qs[3]),
        "q75": float(qs[4]),
        "q90": float(qs[5]),
        "q95": float(qs[6]),
        "q99": float(qs[7]),
        "max": float(qs[8]),
    }


def _fraction(numer: int, denom: int) -> float:
    return float(numer / denom) if int(denom) > 0 else float("nan")


def _cat_dist(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    if not rows:
        return {}
    counts = Counter(str(row.get(key)) for row in rows)
    total = float(sum(counts.values()))
    return {k: float(v / total) for k, v in sorted(counts.items())}


def _tv_distance(a: dict[str, float], b: dict[str, float]) -> float:
    keys = set(a) | set(b)
    return float(0.5 * sum(abs(float(a.get(k, 0.0)) - float(b.get(k, 0.0))) for k in keys))


def _mean_abs_smd(rows: list[dict[str, Any]], ref_rows: list[dict[str, Any]], keys: list[str]) -> dict[str, Any]:
    if not rows or not ref_rows:
        return {"mean_abs_smd": None, "max_abs_smd": None, "per_feature": {}}
    out = {}
    values = []
    for key in keys:
        x = np.asarray([_finite_float(row.get(key)) for row in rows], dtype=np.float64)
        r = np.asarray([_finite_float(row.get(key)) for row in ref_rows], dtype=np.float64)
        x = x[np.isfinite(x)]
        r = r[np.isfinite(r)]
        if x.size == 0 or r.size == 0:
            continue
        scale = float(r.std())
        if scale <= 1e-12:
            scale = 1.0
        smd = float((x.mean() - r.mean()) / scale)
        out[key] = smd
        values.append(abs(smd))
    return {
        "mean_abs_smd": float(np.mean(values)) if values else None,
        "max_abs_smd": float(np.max(values)) if values else None,
        "per_feature": out,
    }


def _spearman(x: list[float], y: list[float]) -> float:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    if x_arr.size < 3:
        return float("nan")
    xr = np.argsort(np.argsort(x_arr)).astype(np.float64)
    yr = np.argsort(np.argsort(y_arr)).astype(np.float64)
    if xr.std() <= 1e-12 or yr.std() <= 1e-12:
        return float("nan")
    return float(np.corrcoef(xr, yr)[0, 1])


def _candidate_filters() -> dict[str, Callable[[dict[str, Any]], bool]]:
    return {
        "state_fraction_ge_0.50": lambda r: float(r["state_dim_fraction"]) >= 0.50,
        "state_fraction_ge_0.60": lambda r: float(r["state_dim_fraction"]) >= 0.60,
        "state_dim_ge_250": lambda r: int(r["state_dim"]) >= 250,
        "state_dim_ge_300": lambda r: int(r["state_dim"]) >= 300,
        "noise_dim_le_32": lambda r: int(r["noise_dim"]) <= 32,
        "noise_fraction_le_0.10": lambda r: float(r["noise_dim_fraction"]) <= 0.10,
        "state_dim_ge_250_noise_dim_le_32": lambda r: int(r["state_dim"]) >= 250 and int(r["noise_dim"]) <= 32,
        "state_dim_ge_300_noise_dim_le_32": lambda r: int(r["state_dim"]) >= 300 and int(r["noise_dim"]) <= 32,
        "state_fraction_ge_0.50_noise_fraction_le_0.10": lambda r: (
            float(r["state_dim_fraction"]) >= 0.50 and float(r["noise_dim_fraction"]) <= 0.10
        ),
        "state_dim_ge_300_noise_dim_le_32_pad_fraction_le_0.40": lambda r: (
            int(r["state_dim"]) >= 300
            and int(r["noise_dim"]) <= 32
            and float(r["zero_pad_dim_fraction"]) <= 0.40
        ),
        "terminal_reset_enabled": lambda r: bool(r["terminal_reset_enabled"]),
        "terminal_reset_disabled": lambda r: not bool(r["terminal_reset_enabled"]),
        "ctrl_reward_enabled": lambda r: bool(r["ctrl_reward_enabled"]),
        "ctrl_reward_disabled": lambda r: not bool(r["ctrl_reward_enabled"]),
        "activation_relu": lambda r: str(r["activation"]) == "relu",
        "activation_tanh": lambda r: str(r["activation"]) == "tanh",
        "activation_sin": lambda r: str(r["activation"]) == "sin",
    }


def run_analysis(
    *,
    seed_start: int,
    count: int,
    train_env_seed: int,
    rejection_min: float,
    max_tries: int,
    build_seed: int,
    device: str,
    output_dir: str,
) -> dict[str, Any]:
    _seed_all(int(build_seed))
    device_obj = torch.device(str(device))
    config = get_model_default_config("rlpfn")
    validate_rlpfn_maintained_path_config(config)
    env_cfg = _build_audit_env_cfg(config["prior"]["environment"])
    prior = EnvironmentPrior(copy.deepcopy(env_cfg))

    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    raw_rows_path = output_path / "single_attempt_rows.jsonl"
    rejection_rows_path = output_path / "rejection_rows.jsonl"

    single_rows: list[dict[str, Any]] = []
    rejection_rows: list[dict[str, Any]] = []
    with raw_rows_path.open("w", encoding="utf-8") as raw_f, rejection_rows_path.open("w", encoding="utf-8") as rej_f:
        for idx in range(int(count)):
            frozen_h_seed = int(seed_start) + int(idx)
            fixed = _prepare_audit_fixed_env_contract(
                env_cfg=env_cfg,
                boundary_contract_mode="normal",
                ppo_reset_env_state_at_sep=True,
                frozen_h_seed=int(frozen_h_seed),
                frozen_h_seed_mode="current",
            )
            frozen_h = copy.deepcopy(fixed["frozen_h"])
            first_env = prior._sample_environment_once(
                copy.deepcopy(frozen_h),
                device=device_obj,
                rng_seed=int(train_env_seed),
            )
            first_row = _extract_row(
                frozen_h_seed=frozen_h_seed,
                attempt_seed=int(train_env_seed),
                frozen_h=frozen_h,
                env=first_env,
            )
            first_row["legal"] = bool(float(first_row["reward_state_input_gain_fraction"]) >= float(rejection_min))
            single_rows.append(first_row)
            raw_f.write(json.dumps(_json_safe(first_row), sort_keys=True) + "\n")

            accepted = False
            best_gain = -float("inf")
            accepted_row = None
            for attempt_idx in range(int(max_tries)):
                attempt_seed = int(train_env_seed) + int(attempt_idx)
                env = (
                    first_env
                    if attempt_idx == 0
                    else prior._sample_environment_once(copy.deepcopy(frozen_h), device=device_obj, rng_seed=attempt_seed)
                )
                gain = _finite_float(env.get("reward_state_input_gain_fraction"), -float("inf"))
                best_gain = max(best_gain, gain)
                if math.isfinite(gain) and gain >= float(rejection_min):
                    accepted_row = _extract_row(
                        frozen_h_seed=frozen_h_seed,
                        attempt_seed=attempt_seed,
                        frozen_h=frozen_h,
                        env=env,
                    )
                    accepted_row["rejection_accepted"] = True
                    accepted_row["rejection_attempt"] = int(attempt_idx + 1)
                    accepted_row["rejection_max_tries"] = int(max_tries)
                    accepted_row["rejection_best_gain"] = float(best_gain)
                    accepted = True
                    break
            if not accepted:
                accepted_row = copy.deepcopy(first_row)
                accepted_row["rejection_accepted"] = False
                accepted_row["rejection_attempt"] = None
                accepted_row["rejection_max_tries"] = int(max_tries)
                accepted_row["rejection_best_gain"] = float(best_gain)
            rejection_rows.append(accepted_row)
            rej_f.write(json.dumps(_json_safe(accepted_row), sort_keys=True) + "\n")

    legal_rows = [r for r in single_rows if bool(r["legal"])]
    accepted_rows = [r for r in rejection_rows if bool(r.get("rejection_accepted"))]
    continuous_keys = [
        "state_dim",
        "action_dim",
        "noise_dim",
        "zero_pad_dim",
        "state_dim_fraction",
        "action_dim_fraction",
        "noise_dim_fraction",
        "zero_pad_dim_fraction",
        "obs_dim_fraction_of_state",
        "num_layers",
        "hidden_dim",
        "init_std",
        "noise_std",
        "reward_scale",
        "terminal_reset_count_target",
        "terminal_bonus_scale_min",
        "terminal_bonus_scale_max",
    ]
    categorical_keys = ["activation", "terminal_reset_enabled", "ctrl_reward_enabled"]

    filter_summaries = {}
    for name, predicate in _candidate_filters().items():
        kept = [r for r in single_rows if predicate(r)]
        kept_legal = [r for r in kept if bool(r["legal"])]
        smd_all = _mean_abs_smd(kept, legal_rows, continuous_keys)
        smd_legal = _mean_abs_smd(kept_legal, legal_rows, continuous_keys)
        cat_all = {
            key: _tv_distance(_cat_dist(kept, key), _cat_dist(legal_rows, key)) for key in categorical_keys
        }
        cat_legal = {
            key: _tv_distance(_cat_dist(kept_legal, key), _cat_dist(legal_rows, key))
            for key in categorical_keys
        }
        filter_summaries[name] = {
            "kept_count": int(len(kept)),
            "kept_fraction": _fraction(len(kept), len(single_rows)),
            "legal_count_in_kept": int(len(kept_legal)),
            "legal_rate_in_kept": _fraction(len(kept_legal), len(kept)),
            "overall_legal_yield": _fraction(len(kept_legal), len(single_rows)),
            "all_kept_vs_default_legal_continuous_smd": smd_all,
            "legal_kept_vs_default_legal_continuous_smd": smd_legal,
            "all_kept_vs_default_legal_categorical_tv": cat_all,
            "legal_kept_vs_default_legal_categorical_tv": cat_legal,
        }

    feature_correlations = {
        key: _spearman([_finite_float(r.get(key)) for r in single_rows], [float(r["reward_state_input_gain_fraction"]) for r in single_rows])
        for key in continuous_keys
    }
    accepted_attempts = [
        int(r["rejection_attempt"]) for r in rejection_rows if bool(r.get("rejection_accepted")) and r.get("rejection_attempt")
    ]
    summary = {
        "audit_entry": "phase2_reward_state_gain_rejection_efficiency",
        "seed_start": int(seed_start),
        "count": int(count),
        "train_env_seed": int(train_env_seed),
        "rejection_min": float(rejection_min),
        "rejection_max_tries": int(max_tries),
        "device": str(device_obj),
        "single_attempt": {
            "legal_count": int(len(legal_rows)),
            "legal_rate": _fraction(len(legal_rows), len(single_rows)),
            "gain_stats": _stats([r["reward_state_input_gain_fraction"] for r in single_rows]),
            "legal_feature_means": {
                key: float(np.nanmean([_finite_float(r.get(key)) for r in legal_rows])) if legal_rows else None
                for key in continuous_keys
            },
            "all_feature_means": {
                key: float(np.nanmean([_finite_float(r.get(key)) for r in single_rows])) if single_rows else None
                for key in continuous_keys
            },
            "legal_categorical_distributions": {key: _cat_dist(legal_rows, key) for key in categorical_keys},
            "all_categorical_distributions": {key: _cat_dist(single_rows, key) for key in categorical_keys},
        },
        "current_rejection_sampling": {
            "accepted_count": int(len(accepted_rows)),
            "accepted_rate": _fraction(len(accepted_rows), len(rejection_rows)),
            "failed_count": int(len(rejection_rows) - len(accepted_rows)),
            "attempt_stats_accepted_only": _stats(accepted_attempts),
            "mean_env_builds_per_accepted": float(sum(accepted_attempts) / len(accepted_attempts))
            if accepted_attempts
            else None,
            "best_gain_stats_for_failures": _stats(
                [r["rejection_best_gain"] for r in rejection_rows if not bool(r.get("rejection_accepted"))]
            ),
        },
        "feature_spearman_with_gain": feature_correlations,
        "candidate_prefilters": filter_summaries,
        "paths": {
            "single_attempt_rows_jsonl": str(raw_rows_path),
            "rejection_rows_jsonl": str(rejection_rows_path),
            "summary_json": str(output_path / "summary.json"),
        },
    }
    (output_path / "summary.json").write_text(json.dumps(_json_safe(summary), indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure reward_state_input_gain_fraction rejection-sampling efficiency."
    )
    parser.add_argument("--seed-start", type=int, default=30000)
    parser.add_argument("--count", type=int, default=2048)
    parser.add_argument("--train-env-seed", type=int, default=2020)
    parser.add_argument("--rejection-min", type=float, default=0.580)
    parser.add_argument("--max-tries", type=int, default=128)
    parser.add_argument("--build-seed", type=int, default=4040)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/chen/RLPFN/artifacts/phase2_reward_state_gain_rejection_efficiency_seed30000_count2048_min058",
    )
    args = parser.parse_args()
    summary = run_analysis(
        seed_start=int(args.seed_start),
        count=int(args.count),
        train_env_seed=int(args.train_env_seed),
        rejection_min=float(args.rejection_min),
        max_tries=int(args.max_tries),
        build_seed=int(args.build_seed),
        device=str(args.device),
        output_dir=str(args.output_dir),
    )
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
