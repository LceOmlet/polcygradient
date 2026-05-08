#!/usr/bin/env python
"""Exploratory audit for topology failures inside Gym-near prior bins.

This is a read-only, environment-side audit.  It samples prior candidates,
selects those whose structural dimensions fall in Gym-validation feature
ranges, evaluates the existing reward-topology health metrics, and reports
which topology clause rejects each candidate.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from phase2_gym_near_prior_filter import (  # noqa: E402
    DEFAULT_GYM_ENV_IDS,
    _dim_row,
    _gym_validation_features,
    _json_safe,
    _make_rules,
    _parse_csv_list,
    _stats,
    _tensor_vector,
)
from ticl.model_configs import get_model_default_config  # noqa: E402
from ticl.priors.environment_prior import EnvironmentPrior  # noqa: E402
from ticl.rlpfn_maintained_path import validate_rlpfn_maintained_path_config  # noqa: E402


def _failure_reasons(row: dict[str, Any], thresholds: dict[str, float]) -> list[str]:
    reasons: list[str] = []
    state_gain = float(row["reward_state_input_gain_fraction"])
    action_gain = float(row["reward_action_input_gain_fraction"])
    ratio = float(row["reward_state_to_action_gain_ratio"])
    if (not math.isfinite(state_gain)) or state_gain < float(thresholds["state_gain_min"]):
        reasons.append("state_gain_low")
    if (not math.isfinite(action_gain)) or action_gain < float(thresholds["action_gain_min"]):
        reasons.append("action_gain_low")
    if (not math.isfinite(ratio)) or ratio > float(thresholds["state_to_action_ratio_max"]):
        reasons.append("state_action_ratio_high")
    return reasons


def _row_with_health(
    *,
    dim_row: dict[str, Any],
    state_gain: float,
    action_gain: float,
    ratio: float,
    noise_gain: float,
    thresholds: dict[str, float],
) -> dict[str, Any]:
    row = copy.deepcopy(dim_row)
    row.update(
        {
            "reward_state_input_gain_fraction": float(state_gain),
            "reward_action_input_gain_fraction": float(action_gain),
            "reward_state_to_action_gain_ratio": float(ratio),
            "reward_noise_input_gain_fraction": float(noise_gain),
            "state_gain_margin": float(state_gain) - float(thresholds["state_gain_min"]),
            "action_gain_margin": float(action_gain) - float(thresholds["action_gain_min"]),
            "state_to_action_ratio_margin": float(thresholds["state_to_action_ratio_max"]) - float(ratio),
        }
    )
    reasons = _failure_reasons(row, thresholds)
    row["topology_pass"] = len(reasons) == 0
    row["failure_reasons"] = ";".join(reasons) if reasons else "pass"
    row["n_failure_reasons"] = int(len(reasons))
    return row


def _count_reasons(rows: list[dict[str, Any]]) -> dict[str, Any]:
    combo_counts = Counter(str(row["failure_reasons"]) for row in rows)
    reason_counts: Counter[str] = Counter()
    single_blocker_counts: Counter[str] = Counter()
    for row in rows:
        reasons = [part for part in str(row["failure_reasons"]).split(";") if part and part != "pass"]
        reason_counts.update(reasons)
        if len(reasons) == 1:
            single_blocker_counts.update(reasons)
    return {
        "combination_counts": dict(sorted(combo_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "single_blocker_counts": dict(sorted(single_blocker_counts.items())),
    }


def _metric_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pass_rows = [row for row in rows if bool(row["topology_pass"])]
    fail_rows = [row for row in rows if not bool(row["topology_pass"])]

    def pack(subset: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "n": int(len(subset)),
            "obs_dim": _stats([float(row["obs_dim"]) for row in subset]),
            "action_dim": _stats([float(row["action_dim"]) for row in subset]),
            "state_dim": _stats([float(row["state_dim"]) for row in subset]),
            "reward_state_input_gain_fraction": _stats(
                [float(row["reward_state_input_gain_fraction"]) for row in subset]
            ),
            "reward_action_input_gain_fraction": _stats(
                [float(row["reward_action_input_gain_fraction"]) for row in subset]
            ),
            "reward_state_to_action_gain_ratio": _stats(
                [float(row["reward_state_to_action_gain_ratio"]) for row in subset]
            ),
            "reward_noise_input_gain_fraction": _stats(
                [float(row["reward_noise_input_gain_fraction"]) for row in subset]
            ),
            "state_gain_margin": _stats([float(row["state_gain_margin"]) for row in subset]),
            "action_gain_margin": _stats([float(row["action_gain_margin"]) for row in subset]),
            "state_to_action_ratio_margin": _stats(
                [float(row["state_to_action_ratio_margin"]) for row in subset]
            ),
        }

    return {
        "all": pack(rows),
        "pass": pack(pass_rows),
        "fail": pack(fail_rows),
        "pass_rate": float(len(pass_rows) / len(rows)) if rows else None,
        "failure_counts": _count_reasons(rows),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    seed = int(args.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    gym_env_ids = _parse_csv_list(args.gym_env_ids)
    gym_rows, gym_range = _gym_validation_features(gym_env_ids)
    all_rules = _make_rules(gym_range)
    rule_names = _parse_csv_list(args.rules)
    unknown = sorted(set(rule_names) - set(all_rules))
    if unknown:
        raise ValueError(f"Unknown rule(s): {unknown}; available={sorted(all_rules)}")

    thresholds = {
        "state_gain_min": float(args.topology_state_gain_min),
        "action_gain_min": float(args.topology_action_gain_min),
        "state_to_action_ratio_max": float(args.topology_state_to_action_ratio_max),
    }

    cfg = copy.deepcopy(get_model_default_config("rlpfn"))
    validate_rlpfn_maintained_path_config(cfg)
    prior = EnvironmentPrior(copy.deepcopy(cfg["prior"]["environment"]))
    device = torch.device(str(args.device))

    target = int(args.target_dim_matched_per_rule)
    raw_batch_size = int(args.raw_batch_size)
    eval_batch_size = int(args.env_eval_batch_size)
    max_raw = int(args.max_raw_samples)

    rows_by_rule: dict[str, list[dict[str, Any]]] = {rule: [] for rule in rule_names}
    raw_seen = 0
    pending: list[tuple[dict[str, Any], dict[str, Any], list[str]]] = []
    env_evaluated = 0

    def done(rule: str) -> bool:
        return len(rows_by_rule[rule]) >= target

    def all_done() -> bool:
        return all(done(rule) for rule in rule_names)

    def active_rules_for(row: dict[str, Any]) -> list[str]:
        return [rule for rule in rule_names if (not done(rule)) and all_rules[rule](row)]

    def evaluate_pending(force: bool = False) -> None:
        nonlocal env_evaluated
        while len(pending) >= eval_batch_size or (force and pending):
            chunk = pending[:eval_batch_size]
            del pending[:eval_batch_size]
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
                health_row = _row_with_health(
                    dim_row=dim_row,
                    state_gain=float(state_gains[local_idx]),
                    action_gain=float(action_gains[local_idx]),
                    ratio=float(ratios[local_idx]),
                    noise_gain=float(noise_gains[local_idx]),
                    thresholds=thresholds,
                )
                for rule in rule_lists[local_idx]:
                    if done(rule):
                        continue
                    out_row = copy.deepcopy(health_row)
                    out_row["rule"] = str(rule)
                    rows_by_rule[rule].append(out_row)
            env["transition_generator"] = None
            env["policy_generator"] = None
            del env
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if all_done():
                pending.clear()
                break

    while raw_seen < max_raw and not all_done():
        batch_n = min(raw_batch_size, max_raw - raw_seen)
        h_batch = list(prior._sample_batch_hypers(batch_n))
        for local_idx, h in enumerate(h_batch):
            candidate_idx = raw_seen + local_idx
            env_seed = int(args.env_seed_start) + int(candidate_idx)
            dim_row = _dim_row(prior, h, candidate_idx, env_seed)
            rules = active_rules_for(dim_row)
            if not rules:
                continue
            pending.append((copy.deepcopy(h), dim_row, rules))
            evaluate_pending(force=False)
            if all_done():
                break
        raw_seen += batch_n
        if raw_seen % int(args.progress_every_raw) == 0 or all_done():
            print(
                json.dumps(
                    {
                        "raw_seen": int(raw_seen),
                        "env_evaluated": int(env_evaluated),
                        "rows_by_rule": {rule: len(rows) for rule, rows in rows_by_rule.items()},
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    evaluate_pending(force=True)

    csv_path = out_dir / "gym_near_topology_failure_audit.csv"
    fieldnames = [
        "rule",
        "topology_pass",
        "failure_reasons",
        "n_failure_reasons",
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
        "state_gain_margin",
        "action_gain_margin",
        "state_to_action_ratio_margin",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rule in rule_names:
            for row in rows_by_rule[rule]:
                writer.writerow({key: row.get(key) for key in fieldnames})

    report = {
        "audit_entry": "phase2_gym_near_topology_failure_audit",
        "exploratory_only": True,
        "contract": {
            "does_not_modify_exact_scm": True,
            "does_not_modify_default_prior_generator": True,
            "environment_side_only": True,
            "ppo_training_not_run": True,
            "dimension_features_are_filter_coordinates_not_generator_knobs": True,
            "health_criteria": thresholds,
        },
        "gym_env_ids": gym_env_ids,
        "gym_validation_rows": gym_rows,
        "gym_feature_range": gym_range,
        "rules": rule_names,
        "sampling": {
            "seed": seed,
            "env_seed_start": int(args.env_seed_start),
            "raw_seen": int(raw_seen),
            "max_raw_samples": int(max_raw),
            "env_evaluated": int(env_evaluated),
            "target_dim_matched_per_rule": int(target),
            "rows_by_rule": {rule: len(rows) for rule, rows in rows_by_rule.items()},
            "device": str(device),
        },
        "outputs": {"csv": str(csv_path)},
        "summary_by_rule": {rule: _metric_summary(rows_by_rule[rule]) for rule in rule_names},
    }
    report_path = out_dir / "failure_audit_report.json"
    report_path.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path), "csv": str(csv_path)}, sort_keys=True))
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--gym-env-ids", type=str, default=DEFAULT_GYM_ENV_IDS)
    parser.add_argument(
        "--rules",
        type=str,
        default="gym_full_obs_action_range,gym_q90_obs_action_range",
    )
    parser.add_argument("--target-dim-matched-per-rule", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=6160)
    parser.add_argument("--env-seed-start", type=int, default=8060000)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--raw-batch-size", type=int, default=4096)
    parser.add_argument("--env-eval-batch-size", type=int, default=128)
    parser.add_argument("--max-raw-samples", type=int, default=160000)
    parser.add_argument("--progress-every-raw", type=int, default=8192)
    parser.add_argument("--topology-state-gain-min", type=float, default=0.7)
    parser.add_argument("--topology-action-gain-min", type=float, default=0.06)
    parser.add_argument("--topology-state-to-action-ratio-max", type=float, default=10.0)
    return parser


def main(argv: list[str] | None = None) -> None:
    run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()
