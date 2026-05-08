#!/usr/bin/env python
"""Exploratory reward-path-only group-balance probe for Gym-near priors.

This script does not modify the exact-SCM generator or default prior config.
It samples ordinary prior candidates, filters by Gym-validation dimension
ranges, recomputes reward-topology health under a reward-path-only group
balancing profile, writes a fixed-frozen-h list annotated for the companion
exploratory runner, and records light environment-side diversity probes.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import sys
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
from phase2_reward_group_balance_runtime import (  # noqa: E402
    annotate_h_with_reward_group_balance,
    apply_reward_group_balance_to_env,
)
from phase2_gated_reward_path_generator_wrapper import (  # noqa: E402
    RewardPathBalanceProfile,
    TopologyThresholds,
    apply_decision_to_row,
    contract_dict,
    decide_gated_reward_path_repair,
    row_with_balanced_topology,
    select_h_for_decision,
)
from ticl.analysis.fixed_env_h import _to_builtin, summarize_fixed_env_h  # noqa: E402
from ticl.model_configs import get_model_default_config  # noqa: E402
from ticl.priors.environment_prior import EnvironmentPrior  # noqa: E402
from ticl.rlpfn_maintained_path import validate_rlpfn_maintained_path_config  # noqa: E402
from ticl.train import _freeze_env_h_list_for_replay  # noqa: E402


def _topology_pass(row: dict[str, Any], thresholds: dict[str, float], *, prefix: str = "") -> bool:
    # Backward-compatible helper for older local callers; the probe itself uses
    # phase2_gated_reward_path_generator_wrapper for the authoritative decision.
    from phase2_gated_reward_path_generator_wrapper import topology_pass  # noqa: WPS433

    return topology_pass(
        row,
        TopologyThresholds(
            state_gain_min=float(thresholds["state_gain_min"]),
            action_gain_min=float(thresholds["action_gain_min"]),
            state_to_action_ratio_max=float(thresholds["state_to_action_ratio_max"]),
        ),
        prefix=prefix,
    )


def _write_selected_h(
    *,
    prior: EnvironmentPrior,
    h: dict[str, Any],
    path: Path,
    apply_balance: bool,
    state_scale: float,
    action_scale: float,
    noise_scale: float,
    profile: str,
) -> dict[str, Any]:
    frozen_h = _freeze_env_h_list_for_replay(prior, [copy.deepcopy(h)])[0]
    if apply_balance:
        frozen_h = annotate_h_with_reward_group_balance(
            frozen_h,
            state_scale=float(state_scale),
            action_scale=float(action_scale),
            noise_scale=float(noise_scale),
            profile=str(profile),
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_builtin(frozen_h), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summarize_fixed_env_h(frozen_h)


def _parse_profile(text: str) -> tuple[str, float, float, float]:
    profile = RewardPathBalanceProfile.parse(text)
    return profile.name, profile.state_scale, profile.action_scale, profile.noise_scale


def _summarize_rows(rows: list[dict[str, Any]], *, prefix: str = "") -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    return {
        "n": int(len(rows)),
        "obs_dim": _stats([float(row["obs_dim"]) for row in rows]),
        "action_dim": _stats([float(row["action_dim"]) for row in rows]),
        "state_dim": _stats([float(row["state_dim"]) for row in rows]),
        "noise_dim": _stats([float(row["noise_dim"]) for row in rows]),
        "reward_state_input_gain_fraction": _stats(
            [float(row[f"{prefix}reward_state_input_gain_fraction"]) for row in rows]
        ),
        "reward_action_input_gain_fraction": _stats(
            [float(row[f"{prefix}reward_action_input_gain_fraction"]) for row in rows]
        ),
        "reward_noise_input_gain_fraction": _stats(
            [float(row[f"{prefix}reward_noise_input_gain_fraction"]) for row in rows]
        ),
        "reward_state_to_action_gain_ratio": _stats(
            [float(row[f"{prefix}reward_state_to_action_gain_ratio"]) for row in rows]
        ),
    }


def _transition_diversity_probe(
    *,
    prior: EnvironmentPrior,
    selected_rows: list[dict[str, Any]],
    selected_h: list[dict[str, Any]],
    steps: int,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    if not selected_rows or steps <= 0 or batch_size <= 0:
        return {"n_envs": 0, "steps": int(max(0, steps))}
    h_list = [copy.deepcopy(h) for h in selected_h[:batch_size]]
    rows = selected_rows[:batch_size]
    seeds = [int(row["env_seed"]) for row in rows]
    env = prior._sample_environment_family_coarse_batch(
        h_list=h_list,
        device=device,
        rng_seeds=seeds,
        build_policy_generator=False,
        preserve_skipped_generator_rng=True,
        _disable_rejection=True,
    )
    apply_reward_group_balance_to_env(env, h_list)
    transition = env["transition_generator"]
    packed_width = int(getattr(transition, "_packed_input_cap", env.get("env_input_dim", 1)))
    state_cap = int(env.get("state_dim", 1))
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    x = torch.randn((len(h_list), packed_width), device=device, dtype=torch.float32, generator=generator)
    state_deltas = []
    rewards = []
    terminal_signals = []
    for _ in range(int(steps)):
        action_noise = torch.randn(x.shape, device=device, dtype=torch.float32, generator=generator)
        x = 0.5 * x + 0.5 * action_noise
        out = transition(x, x_input_is_packed=True)
        state_next, reward_next, terminal_signal, _terminal_bonus = EnvironmentPrior._unpack_transition_output(out)
        prev_state = x[:, :state_cap]
        common = min(int(prev_state.shape[1]), int(state_next.shape[1]))
        if common > 0:
            state_deltas.append((state_next[:, :common] - prev_state[:, :common]).detach().cpu().reshape(-1).numpy())
        rewards.append(reward_next.detach().cpu().reshape(-1).numpy())
        if terminal_signal is not None:
            terminal_signals.append(terminal_signal.detach().cpu().reshape(-1).numpy())
        x[:, :state_next.shape[1]] = state_next[:, : min(state_next.shape[1], x.shape[1])]
    state_delta = np.concatenate(state_deltas) if state_deltas else np.asarray([], dtype=np.float64)
    reward = np.concatenate(rewards) if rewards else np.asarray([], dtype=np.float64)
    terminal = np.concatenate(terminal_signals) if terminal_signals else np.asarray([], dtype=np.float64)
    env["transition_generator"] = None
    env["policy_generator"] = None
    del env
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "n_envs": int(len(h_list)),
        "steps": int(steps),
        "state_delta_abs": _stats(np.abs(state_delta).astype(np.float64).tolist()),
        "state_delta_rms": float(np.sqrt(np.mean(np.square(state_delta)))) if state_delta.size else None,
        "reward_raw": _stats(reward.astype(np.float64).tolist()),
        "reward_raw_std": float(np.std(reward.astype(np.float64))) if reward.size else None,
        "terminal_signal": _stats(terminal.astype(np.float64).tolist()) if terminal.size else {"n": 0},
        "terminal_signal_std": float(np.std(terminal.astype(np.float64))) if terminal.size else None,
        "finite_reward_fraction": float(np.mean(np.isfinite(reward))) if reward.size else None,
        "finite_state_delta_fraction": float(np.mean(np.isfinite(state_delta))) if state_delta.size else None,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    rng_seed = int(args.seed)
    random.seed(rng_seed)
    np.random.seed(rng_seed)
    torch.manual_seed(rng_seed)

    profile = RewardPathBalanceProfile.parse(args.balance_profile)
    profile_name = str(profile.name)
    state_scale = float(profile.state_scale)
    action_scale = float(profile.action_scale)
    noise_scale = float(profile.noise_scale)
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

    thresholds = {
        "state_gain_min": float(args.topology_state_gain_min),
        "action_gain_min": float(args.topology_action_gain_min),
        "state_to_action_ratio_max": float(args.topology_state_to_action_ratio_max),
    }
    threshold_obj = TopologyThresholds(
        state_gain_min=float(args.topology_state_gain_min),
        action_gain_min=float(args.topology_action_gain_min),
        state_to_action_ratio_max=float(args.topology_state_to_action_ratio_max),
    )

    cfg = copy.deepcopy(get_model_default_config("rlpfn"))
    validate_rlpfn_maintained_path_config(cfg)
    prior = EnvironmentPrior(copy.deepcopy(cfg["prior"]["environment"]))
    device = torch.device(str(args.device))

    raw_seen = 0
    dim_matched = 0
    env_evaluated = 0
    original_topology_pass = 0
    balanced_topology_pass = 0
    gated_topology_pass = 0
    selected_rows_by_rule: dict[str, list[dict[str, Any]]] = {rule: [] for rule in selected_rule_names}
    selected_h_by_rule: dict[str, list[dict[str, Any]]] = {rule: [] for rule in selected_rule_names}
    selected_fingerprints_by_rule: dict[str, set[str]] = {rule: set() for rule in selected_rule_names}
    rule_dim_match_counts = {rule: 0 for rule in selected_rule_names}
    rule_original_pass_counts = {rule: 0 for rule in selected_rule_names}
    rule_balanced_pass_counts = {rule: 0 for rule in selected_rule_names}
    rule_gated_pass_counts = {rule: 0 for rule in selected_rule_names}
    rule_original_keep_counts = {rule: 0 for rule in selected_rule_names}
    rule_balanced_repair_counts = {rule: 0 for rule in selected_rule_names}
    rule_still_fail_counts = {rule: 0 for rule in selected_rule_names}
    audit_rows: list[dict[str, Any]] = []
    pending: list[tuple[dict[str, Any], dict[str, Any], list[str]]] = []

    def rule_done(rule: str) -> bool:
        return len(selected_rows_by_rule[rule]) >= int(args.target_per_rule)

    def all_done() -> bool:
        return all(rule_done(rule) for rule in selected_rule_names)

    def rules_for_dim(row: dict[str, Any]) -> list[str]:
        return [rule for rule in selected_rule_names if all_rules[rule](row)]

    def evaluate_pending(force: bool = False) -> None:
        nonlocal env_evaluated, original_topology_pass, balanced_topology_pass, gated_topology_pass
        while len(pending) >= int(args.env_eval_batch_size) or (force and pending):
            chunk = pending[: int(args.env_eval_batch_size)]
            del pending[: int(args.env_eval_batch_size)]
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
            noise_gains = _tensor_vector(env, "reward_noise_input_gain_fraction", n)
            ratios = _tensor_vector(env, "reward_state_to_action_gain_ratio", n)
            env_evaluated += n
            for local_idx, dim_row in enumerate(dim_rows):
                row = copy.deepcopy(dim_row)
                row.update(
                    {
                        "reward_state_input_gain_fraction": float(state_gains[local_idx]),
                        "reward_action_input_gain_fraction": float(action_gains[local_idx]),
                        "reward_noise_input_gain_fraction": float(noise_gains[local_idx]),
                        "reward_state_to_action_gain_ratio": float(ratios[local_idx]),
                    }
                )
                row = row_with_balanced_topology(row, profile=profile)
                decision = decide_gated_reward_path_repair(row, thresholds=threshold_obj)
                row = apply_decision_to_row(row, decision)
                audit_rows.append(row)
                if row["original_topology_pass"]:
                    original_topology_pass += 1
                if row["balanced_topology_pass"]:
                    balanced_topology_pass += 1
                if row["gated_topology_pass"]:
                    gated_topology_pass += 1
                for rule in rule_lists[local_idx]:
                    if row["original_topology_pass"]:
                        rule_original_pass_counts[rule] += 1
                    if row["balanced_topology_pass"]:
                        rule_balanced_pass_counts[rule] += 1
                    if row["gated_topology_pass"]:
                        rule_gated_pass_counts[rule] += 1
                    else:
                        rule_still_fail_counts[rule] += 1
                    if (not row["gated_topology_pass"]) or rule_done(rule):
                        continue
                    apply_balance = bool(decision.selected_balance_applied)
                    h_selected = select_h_for_decision(h_list[local_idx], decision=decision, profile=profile)
                    h_path = frozen_dir / rule / f"{rule}_{len(selected_rows_by_rule[rule]):04d}_seed{int(row['env_seed'])}.json"
                    fingerprint = _write_selected_h(
                        prior=prior,
                        h=h_list[local_idx],
                        path=h_path,
                        apply_balance=apply_balance,
                        state_scale=state_scale,
                        action_scale=action_scale,
                        noise_scale=noise_scale,
                        profile=profile_name,
                    )["frozen_h_fingerprint"]
                    if fingerprint in selected_fingerprints_by_rule[rule]:
                        continue
                    selected_fingerprints_by_rule[rule].add(str(fingerprint))
                    selected_row = copy.deepcopy(row)
                    selected_row.update(
                        {
                            "rule": str(rule),
                            "full_frozen_h_json": str(h_path),
                            "frozen_h_fingerprint": str(fingerprint),
                            "selected_repair_kind": str(decision.repair_kind),
                            "selected_balance_applied": bool(apply_balance),
                        }
                    )
                    selected_rows_by_rule[rule].append(selected_row)
                    selected_h_by_rule[rule].append(h_selected)
                    if apply_balance:
                        rule_balanced_repair_counts[rule] += 1
                    else:
                        rule_original_keep_counts[rule] += 1
            env["transition_generator"] = None
            env["policy_generator"] = None
            del env
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if all_done():
                pending.clear()
                break

    while raw_seen < int(args.max_raw_samples) and not all_done():
        batch_n = min(int(args.raw_batch_size), int(args.max_raw_samples) - raw_seen)
        h_batch = list(prior._sample_batch_hypers(batch_n))
        for local_idx, h in enumerate(h_batch):
            candidate_idx = raw_seen + local_idx
            env_seed = int(args.env_seed_start) + int(candidate_idx)
            dim_row = _dim_row(prior, h, candidate_idx, env_seed)
            rules = rules_for_dim(dim_row)
            if not rules:
                continue
            dim_matched += 1
            for rule in rules:
                rule_dim_match_counts[rule] += 1
            if any(not rule_done(rule) for rule in rules):
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
                        "dim_matched": int(dim_matched),
                        "env_evaluated": int(env_evaluated),
                        "original_topology_pass": int(original_topology_pass),
                        "balanced_topology_pass": int(balanced_topology_pass),
                        "gated_topology_pass": int(gated_topology_pass),
                        "selected": {rule: len(rows) for rule, rows in selected_rows_by_rule.items()},
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    evaluate_pending(force=True)

    audit_csv = out_dir / "reward_group_balance_candidates.csv"
    selected_csv = out_dir / "selected_prior_envs.csv"
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
        "reward_noise_input_gain_fraction",
        "reward_state_to_action_gain_ratio",
        "balanced_reward_state_input_gain_fraction",
        "balanced_reward_action_input_gain_fraction",
        "balanced_reward_noise_input_gain_fraction",
        "balanced_reward_state_to_action_gain_ratio",
        "original_topology_pass",
        "balanced_topology_pass",
        "gated_topology_pass",
        "gated_repair_kind",
        "balance_profile",
        "balance_state_scale",
        "balance_action_scale",
        "balance_noise_scale",
        "selected_repair_kind",
        "selected_balance_applied",
        "frozen_h_fingerprint",
        "full_frozen_h_json",
    ]
    with audit_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[name for name in fieldnames if name not in {"rule", "frozen_h_fingerprint", "full_frozen_h_json"}])
        writer.writeheader()
        for row in audit_rows:
            writer.writerow({name: row.get(name) for name in writer.fieldnames})
    with selected_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rule in selected_rule_names:
            for row in selected_rows_by_rule[rule]:
                writer.writerow({name: row.get(name) for name in fieldnames})

    diversity = {
        rule: _transition_diversity_probe(
            prior=prior,
            selected_rows=selected_rows_by_rule[rule],
            selected_h=selected_h_by_rule[rule],
            steps=int(args.diversity_steps),
            batch_size=int(args.diversity_n_envs),
            device=device,
            seed=int(args.seed) + 9000 + rule_idx,
        )
        for rule_idx, rule in enumerate(selected_rule_names)
    }
    report = {
        "audit_entry": "phase2_reward_group_balance_probe",
        "exploratory_only": True,
        "contract": {
            "does_not_modify_exact_scm": True,
            "does_not_modify_default_prior_generator": True,
            "environment_side_only_until_companion_runner": True,
            "dimension_features_are_filter_coordinates_not_generator_knobs": True,
            "reward_path_only_contract": (
                "state/terminal transition path remains original; only reward is recomputed "
                "with state/action/noise input group scales inside the exploratory runtime wrapper"
            ),
            "health_criteria": thresholds,
            "gated_reward_path_wrapper": contract_dict(profile=profile, thresholds=threshold_obj),
        },
        "balance_profile": {
            "name": str(profile_name),
            "state_scale": float(state_scale),
            "action_scale": float(action_scale),
            "noise_scale": float(noise_scale),
        },
        "gym_env_ids": gym_env_ids,
        "gym_validation_rows": gym_rows,
        "gym_feature_range": gym_range,
        "rules": selected_rule_names,
        "sampling": {
            "seed": int(args.seed),
            "env_seed_start": int(args.env_seed_start),
            "raw_seen": int(raw_seen),
            "max_raw_samples": int(args.max_raw_samples),
            "dim_matched": int(dim_matched),
            "env_evaluated": int(env_evaluated),
            "original_topology_pass": int(original_topology_pass),
            "balanced_topology_pass": int(balanced_topology_pass),
            "gated_topology_pass": int(gated_topology_pass),
            "rule_dim_match_counts": rule_dim_match_counts,
            "rule_original_pass_counts": rule_original_pass_counts,
            "rule_balanced_pass_counts": rule_balanced_pass_counts,
            "rule_gated_pass_counts": rule_gated_pass_counts,
            "rule_original_keep_counts": rule_original_keep_counts,
            "rule_balanced_repair_counts": rule_balanced_repair_counts,
            "rule_still_fail_counts": rule_still_fail_counts,
            "target_per_rule": int(args.target_per_rule),
            "device": str(device),
        },
        "selected_count_by_rule": {rule: int(len(rows)) for rule, rows in selected_rows_by_rule.items()},
        "selected_summary_by_rule_original": {
            rule: _summarize_rows(selected_rows_by_rule[rule], prefix="") for rule in selected_rule_names
        },
        "selected_summary_by_rule_balanced": {
            rule: _summarize_rows(selected_rows_by_rule[rule], prefix="balanced_") for rule in selected_rule_names
        },
        "transition_diversity_probe": diversity,
        "outputs": {
            "candidate_csv": str(audit_csv),
            "selected_csv": str(selected_csv),
            "frozen_h_dir": str(frozen_dir),
        },
    }
    report_path = out_dir / "reward_group_balance_probe_report.json"
    report_path.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(_json_safe({"report": str(report_path), "selected_csv": str(selected_csv), "selected_count_by_rule": report["selected_count_by_rule"]}), sort_keys=True))
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--gym-env-ids", type=str, default=DEFAULT_GYM_ENV_IDS)
    parser.add_argument("--rules", type=str, default="gym_full_obs_action_range,gym_q90_obs_action_range")
    parser.add_argument("--balance-profile", type=str, default="s3_a16_n0125:3:16:0.125")
    parser.add_argument("--target-per-rule", type=int, default=64)
    parser.add_argument("--seed", type=int, default=6260)
    parser.add_argument("--env-seed-start", type=int, default=8160000)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--raw-batch-size", type=int, default=4096)
    parser.add_argument("--env-eval-batch-size", type=int, default=64)
    parser.add_argument("--max-raw-samples", type=int, default=65536)
    parser.add_argument("--progress-every-raw", type=int, default=8192)
    parser.add_argument("--topology-state-gain-min", type=float, default=0.7)
    parser.add_argument("--topology-action-gain-min", type=float, default=0.06)
    parser.add_argument("--topology-state-to-action-ratio-max", type=float, default=10.0)
    parser.add_argument("--diversity-n-envs", type=int, default=64)
    parser.add_argument("--diversity-steps", type=int, default=32)
    return parser


def main() -> None:
    run(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
