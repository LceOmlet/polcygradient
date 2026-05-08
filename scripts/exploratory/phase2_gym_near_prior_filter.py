#!/usr/bin/env python
"""Exploratory Gym-near prior environment selector.

This script intentionally lives outside the exact-SCM implementation path.  It
does not change the prior generator.  It samples ordinary prior hyper-parameters,
keeps concrete environments whose dimensions fall in Gym-validation feature
ranges, checks the existing reward-topology health criteria on those concrete
instances, and writes a fixed-frozen-h CSV consumable by the trusted pack runner.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ticl.analysis.fixed_env_h import _to_builtin, summarize_fixed_env_h  # noqa: E402
from ticl.model_configs import get_model_default_config  # noqa: E402
from ticl.priors.environment_prior import EnvironmentPrior  # noqa: E402
from ticl.rlpfn_maintained_path import validate_rlpfn_maintained_path_config  # noqa: E402
from ticl.train import _freeze_env_h_list_for_replay  # noqa: E402


DEFAULT_GYM_ENV_IDS = (
    "Ant-v5,BipedalWalker-v3,HalfCheetah-v5,Hopper-v5,Humanoid-v5,"
    "InvertedDoublePendulum-v5,InvertedPendulum-v5,LunarLanderContinuous-v3,"
    "MountainCarContinuous-v0,Pendulum-v1,Reacher-v5,Swimmer-v5,Walker2d-v5"
)


def _json_safe(value: Any) -> Any:
    if torch.is_tensor(value):
        if value.ndim == 0:
            return _json_safe(value.detach().cpu().item())
        return [_json_safe(v) for v in value.detach().cpu().tolist()]
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return repr(value)


def _stats(values: list[float] | np.ndarray) -> dict[str, float | int | None]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "n": 0,
            "min": None,
            "q10": None,
            "q25": None,
            "mean": None,
            "median": None,
            "q75": None,
            "q90": None,
            "max": None,
        }
    return {
        "n": int(arr.size),
        "min": float(arr.min()),
        "q10": float(np.quantile(arr, 0.10)),
        "q25": float(np.quantile(arr, 0.25)),
        "mean": float(arr.mean()),
        "median": float(np.quantile(arr, 0.50)),
        "q75": float(np.quantile(arr, 0.75)),
        "q90": float(np.quantile(arr, 0.90)),
        "max": float(arr.max()),
    }


def _parse_csv_list(text: str) -> list[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


def _gym_validation_features(env_ids: list[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import gymnasium as gym

    rows: list[dict[str, Any]] = []
    for env_id in env_ids:
        env = gym.make(env_id)
        try:
            obs_dim = int(np.prod(env.observation_space.shape))
            action_dim = int(np.prod(env.action_space.shape))
            low = np.asarray(env.action_space.low, dtype=np.float64).reshape(-1)
            high = np.asarray(env.action_space.high, dtype=np.float64).reshape(-1)
            rows.append(
                {
                    "env_id": str(env_id),
                    "obs_dim": int(obs_dim),
                    "action_dim": int(action_dim),
                    "action_abs_bound_min": float(np.minimum(np.abs(low), np.abs(high)).min()),
                    "action_abs_bound_max": float(np.maximum(np.abs(low), np.abs(high)).max()),
                    "max_episode_steps": getattr(env.spec, "max_episode_steps", None),
                }
            )
        finally:
            env.close()

    obs_dims = np.asarray([int(row["obs_dim"]) for row in rows], dtype=np.int64)
    action_dims = np.asarray([int(row["action_dim"]) for row in rows], dtype=np.int64)
    feature_range = {
        "obs_min": int(obs_dims.min()),
        "obs_max": int(obs_dims.max()),
        "obs_q75": float(np.quantile(obs_dims, 0.75)),
        "obs_q90": float(np.quantile(obs_dims, 0.90)),
        "obs_values": sorted(set(int(v) for v in obs_dims.tolist())),
        "action_min": int(action_dims.min()),
        "action_max": int(action_dims.max()),
        "action_q75": float(np.quantile(action_dims, 0.75)),
        "action_q90": float(np.quantile(action_dims, 0.90)),
        "action_values": sorted(set(int(v) for v in action_dims.tolist())),
    }
    return rows, feature_range


def _make_rules(feature_range: dict[str, Any]) -> dict[str, Callable[[dict[str, Any]], bool]]:
    obs_min = int(feature_range["obs_min"])
    obs_max = int(feature_range["obs_max"])
    obs_q90 = int(math.ceil(float(feature_range["obs_q90"])))
    action_min = int(feature_range["action_min"])
    action_max = int(feature_range["action_max"])
    action_q90 = int(math.ceil(float(feature_range["action_q90"])))

    def full(row: dict[str, Any]) -> bool:
        return (
            obs_min <= int(row["obs_dim"]) <= obs_max
            and action_min <= int(row["action_dim"]) <= action_max
        )

    def q90(row: dict[str, Any]) -> bool:
        return (
            obs_min <= int(row["obs_dim"]) <= obs_q90
            and action_min <= int(row["action_dim"]) <= action_q90
        )

    return {
        "gym_full_obs_action_range": full,
        "gym_q90_obs_action_range": q90,
    }


def _tensor_vector(env: dict[str, Any], name: str, n: int) -> list[float]:
    value = env.get(name)
    if torch.is_tensor(value):
        data = value.detach().cpu().reshape(-1).tolist()
        if len(data) != n:
            raise RuntimeError(f"{name} length mismatch: got {len(data)} expected {n}")
        return [float(v) for v in data]
    return [float("nan")] * n


def _dim_row(prior: EnvironmentPrior, h: dict[str, Any], candidate_idx: int, env_seed: int) -> dict[str, Any]:
    state_dim, obs_dim, action_dim, noise_dim, zero_pad_dim = prior._sample_dims(h)
    return {
        "candidate_idx": int(candidate_idx),
        "seed": int(env_seed),
        "env_seed": int(env_seed),
        "state_dim": int(state_dim),
        "obs_dim": int(obs_dim),
        "action_dim": int(action_dim),
        "noise_dim": int(noise_dim),
        "zero_pad_dim": int(zero_pad_dim),
    }


def _topology_pass(row: dict[str, Any], thresholds: dict[str, float]) -> bool:
    state_gain = float(row["reward_state_input_gain_fraction"])
    action_gain = float(row["reward_action_input_gain_fraction"])
    ratio = float(row["reward_state_to_action_gain_ratio"])
    return (
        math.isfinite(state_gain)
        and math.isfinite(action_gain)
        and math.isfinite(ratio)
        and state_gain >= float(thresholds["state_gain_min"])
        and action_gain >= float(thresholds["action_gain_min"])
        and ratio <= float(thresholds["state_to_action_ratio_max"])
    )


def _write_selected_h(
    *,
    prior: EnvironmentPrior,
    h: dict[str, Any],
    path: Path,
) -> dict[str, Any]:
    frozen_h = _freeze_env_h_list_for_replay(prior, [copy.deepcopy(h)])[0]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_builtin(frozen_h), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summarize_fixed_env_h(frozen_h)


def run(args: argparse.Namespace) -> dict[str, Any]:
    rng_seed = int(args.seed)
    random.seed(rng_seed)
    np.random.seed(rng_seed)
    torch.manual_seed(rng_seed)

    out_dir = Path(args.output_dir).expanduser().resolve()
    frozen_dir = out_dir / "frozen_h"
    out_dir.mkdir(parents=True, exist_ok=True)
    frozen_dir.mkdir(parents=True, exist_ok=True)

    gym_env_ids = _parse_csv_list(args.gym_env_ids)
    gym_rows, gym_range = _gym_validation_features(gym_env_ids)
    all_rules = _make_rules(gym_range)
    selected_rule_names = _parse_csv_list(args.rules)
    unknown_rules = sorted(set(selected_rule_names) - set(all_rules))
    if unknown_rules:
        raise ValueError(f"Unknown rule(s): {unknown_rules}; available={sorted(all_rules)}")

    cfg = copy.deepcopy(get_model_default_config("rlpfn"))
    validate_rlpfn_maintained_path_config(cfg)
    env_cfg = copy.deepcopy(cfg["prior"]["environment"])
    prior = EnvironmentPrior(copy.deepcopy(env_cfg))

    thresholds = {
        "state_gain_min": float(args.topology_state_gain_min),
        "action_gain_min": float(args.topology_action_gain_min),
        "state_to_action_ratio_max": float(args.topology_state_to_action_ratio_max),
    }

    device = torch.device(str(args.device))
    target_per_rule = int(args.target_per_rule)
    max_raw_samples = int(args.max_raw_samples)
    raw_batch_size = int(args.raw_batch_size)
    env_eval_batch_size = int(args.env_eval_batch_size)
    selected_rows_by_rule: dict[str, list[dict[str, Any]]] = {rule: [] for rule in selected_rule_names}
    selected_fingerprints_by_rule: dict[str, set[str]] = {rule: set() for rule in selected_rule_names}

    raw_seen = 0
    dim_matched = 0
    env_evaluated = 0
    topology_pass_count = 0
    pending: list[tuple[dict[str, Any], dict[str, Any], list[str]]] = []
    rule_dim_match_counts = {rule: 0 for rule in selected_rule_names}
    rule_topology_match_counts = {rule: 0 for rule in selected_rule_names}

    def rules_for_dim_row(row: dict[str, Any]) -> list[str]:
        return [rule for rule in selected_rule_names if all_rules[rule](row)]

    def rule_done(rule: str) -> bool:
        return len(selected_rows_by_rule[rule]) >= target_per_rule

    def all_done() -> bool:
        return all(rule_done(rule) for rule in selected_rule_names)

    def evaluate_pending(force: bool = False) -> None:
        nonlocal env_evaluated, topology_pass_count
        while len(pending) >= env_eval_batch_size or (force and pending):
            chunk = pending[:env_eval_batch_size]
            del pending[:env_eval_batch_size]
            h_list = [copy.deepcopy(item[0]) for item in chunk]
            dim_rows = [copy.deepcopy(item[1]) for item in chunk]
            rule_lists = [list(item[2]) for item in chunk]
            seeds = [int(row["env_seed"]) for row in dim_rows]
            env = prior._sample_environment_family_coarse_batch(
                h_list=h_list,
                device=device,
                rng_seeds=seeds,
                build_policy_generator=False,
                preserve_skipped_generator_rng=True,
                _disable_rejection=True,
            )
            n = len(chunk)
            state_gains = _tensor_vector(env, "reward_state_input_gain_fraction", n)
            action_gains = _tensor_vector(env, "reward_action_input_gain_fraction", n)
            ratios = _tensor_vector(env, "reward_state_to_action_gain_ratio", n)
            noise_gains = _tensor_vector(env, "reward_noise_input_gain_fraction", n)
            env_evaluated += n
            for local_idx, dim_row in enumerate(dim_rows):
                health_row = copy.deepcopy(dim_row)
                health_row.update(
                    {
                        "reward_state_input_gain_fraction": float(state_gains[local_idx]),
                        "reward_action_input_gain_fraction": float(action_gains[local_idx]),
                        "reward_state_to_action_gain_ratio": float(ratios[local_idx]),
                        "reward_noise_input_gain_fraction": float(noise_gains[local_idx]),
                    }
                )
                if not _topology_pass(health_row, thresholds):
                    continue
                topology_pass_count += 1
                for rule in rule_lists[local_idx]:
                    rule_topology_match_counts[rule] += 1
                    if rule_done(rule):
                        continue
                    h_path = frozen_dir / rule / f"{rule}_{len(selected_rows_by_rule[rule]):04d}_seed{int(health_row['env_seed'])}.json"
                    fingerprint = _write_selected_h(prior=prior, h=h_list[local_idx], path=h_path)[
                        "frozen_h_fingerprint"
                    ]
                    if fingerprint in selected_fingerprints_by_rule[rule]:
                        continue
                    selected_fingerprints_by_rule[rule].add(str(fingerprint))
                    selected_row = copy.deepcopy(health_row)
                    selected_row.update(
                        {
                            "rule": str(rule),
                            "full_frozen_h_json": str(h_path),
                            "frozen_h_fingerprint": str(fingerprint),
                        }
                    )
                    selected_rows_by_rule[rule].append(selected_row)
            env["transition_generator"] = None
            env["policy_generator"] = None
            del env
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if all_done():
                pending.clear()
                break

    while raw_seen < max_raw_samples and not all_done():
        batch_n = min(raw_batch_size, max_raw_samples - raw_seen)
        h_batch = list(prior._sample_batch_hypers(batch_n))
        for local_idx, h in enumerate(h_batch):
            candidate_idx = raw_seen + local_idx
            env_seed = int(args.env_seed_start) + int(candidate_idx)
            dim_row = _dim_row(prior, h, candidate_idx, env_seed)
            matched_rules = rules_for_dim_row(dim_row)
            if not matched_rules:
                continue
            dim_matched += 1
            for rule in matched_rules:
                rule_dim_match_counts[rule] += 1
            if any(not rule_done(rule) for rule in matched_rules):
                pending.append((copy.deepcopy(h), dim_row, matched_rules))
                evaluate_pending(force=False)
            if all_done():
                break
        raw_seen += batch_n
        if raw_seen % int(args.progress_every_raw) == 0 or all_done():
            counts = {rule: len(rows) for rule, rows in selected_rows_by_rule.items()}
            print(
                json.dumps(
                    {
                        "raw_seen": int(raw_seen),
                        "dim_matched": int(dim_matched),
                        "env_evaluated": int(env_evaluated),
                        "topology_pass": int(topology_pass_count),
                        "selected": counts,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    evaluate_pending(force=True)

    csv_path = out_dir / "selected_prior_envs.csv"
    fieldnames = [
        "rule",
        "seed",
        "env_seed",
        "candidate_idx",
        "state_dim",
        "obs_dim",
        "action_dim",
        "noise_dim",
        "zero_pad_dim",
        "reward_state_input_gain_fraction",
        "reward_action_input_gain_fraction",
        "reward_state_to_action_gain_ratio",
        "reward_noise_input_gain_fraction",
        "frozen_h_fingerprint",
        "full_frozen_h_json",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rule in selected_rule_names:
            for row in selected_rows_by_rule[rule]:
                writer.writerow({name: row.get(name) for name in fieldnames})

    def selected_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "n": int(len(rows)),
            "obs_dim": _stats([float(row["obs_dim"]) for row in rows]),
            "action_dim": _stats([float(row["action_dim"]) for row in rows]),
            "state_dim": _stats([float(row["state_dim"]) for row in rows]),
            "reward_state_input_gain_fraction": _stats(
                [float(row["reward_state_input_gain_fraction"]) for row in rows]
            ),
            "reward_action_input_gain_fraction": _stats(
                [float(row["reward_action_input_gain_fraction"]) for row in rows]
            ),
            "reward_state_to_action_gain_ratio": _stats(
                [float(row["reward_state_to_action_gain_ratio"]) for row in rows]
            ),
            "reward_noise_input_gain_fraction": _stats(
                [float(row["reward_noise_input_gain_fraction"]) for row in rows]
            ),
        }

    report = {
        "audit_entry": "phase2_gym_near_prior_filter",
        "exploratory_only": True,
        "contract": {
            "does_not_modify_exact_scm": True,
            "does_not_modify_default_prior_generator": True,
            "output_is_fixed_frozen_list_for_pack_runner": True,
            "dimension_features_are_filter_coordinates_not_generator_knobs": True,
            "health_criteria": thresholds,
        },
        "gym_env_ids": gym_env_ids,
        "gym_validation_rows": gym_rows,
        "gym_feature_range": gym_range,
        "rules": selected_rule_names,
        "sampling": {
            "seed": rng_seed,
            "env_seed_start": int(args.env_seed_start),
            "raw_seen": int(raw_seen),
            "max_raw_samples": int(max_raw_samples),
            "dim_matched": int(dim_matched),
            "env_evaluated": int(env_evaluated),
            "topology_pass": int(topology_pass_count),
            "rule_dim_match_counts": rule_dim_match_counts,
            "rule_topology_match_counts": rule_topology_match_counts,
            "target_per_rule": int(target_per_rule),
            "device": str(device),
        },
        "outputs": {
            "csv": str(csv_path),
            "frozen_h_dir": str(frozen_dir),
        },
        "selected_summary_by_rule": {
            rule: selected_summary(selected_rows_by_rule[rule]) for rule in selected_rule_names
        },
        "selected_count_by_rule": {rule: len(rows) for rule, rows in selected_rows_by_rule.items()},
    }
    report_path = out_dir / "selection_report.json"
    report_path.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(_json_safe({"csv": str(csv_path), "report": str(report_path), "selected_count_by_rule": report["selected_count_by_rule"]}), sort_keys=True))
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--gym-env-ids", type=str, default=DEFAULT_GYM_ENV_IDS)
    parser.add_argument(
        "--rules",
        type=str,
        default="gym_full_obs_action_range",
        help="Comma-separated rules. Available: gym_full_obs_action_range,gym_q90_obs_action_range",
    )
    parser.add_argument("--target-per-rule", type=int, default=64)
    parser.add_argument("--seed", type=int, default=6060)
    parser.add_argument("--env-seed-start", type=int, default=7060000)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--raw-batch-size", type=int, default=2048)
    parser.add_argument("--env-eval-batch-size", type=int, default=64)
    parser.add_argument("--max-raw-samples", type=int, default=200000)
    parser.add_argument("--progress-every-raw", type=int, default=8192)
    parser.add_argument("--topology-state-gain-min", type=float, default=0.7)
    parser.add_argument("--topology-action-gain-min", type=float, default=0.06)
    parser.add_argument("--topology-state-to-action-ratio-max", type=float, default=10.0)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
