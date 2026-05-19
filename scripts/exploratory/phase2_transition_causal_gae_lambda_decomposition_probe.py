#!/usr/bin/env python
"""GAE/lambda decomposition for q90 transition-causal credit loss.

Exploratory/read-only.  Reuses the canonical pack-runner official PPO
collect/fill path; performs no training and does not mutate milestone code.

The previous credit probe found:
    q90 transition_and_primary has positive raw/VecNorm MC return alignment,
    while official actor advantages do not reinforce the same direction.

This probe asks whether the loss happens because:
    - the useful signal is only visible at long horizon,
    - the learned value baseline subtracts/reverses it,
    - GAE(lambda) itself shortens the horizon too much,
    - or the signal was already unstable under zero-value returns.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.exploratory.phase2_transition_causal_context_gate import _candidate_labels  # noqa: E402
from scripts.exploratory.phase2_transition_causal_initial_ppo_signal_probe import (  # noqa: E402
    DEFAULT_TRANSITION_REPORT,
    _alignment_scores,
    _build_pack_buffer,
    _direction_tensor_for_best_pairs,
    _json_safe,
    _label_stats,
    _load_json,
    _normalize_advantages_like_official,
    _policy_action_mean_std,
)


DEFAULT_OUTPUT_DIR = (
    "/home/chen/RLPFN/artifacts/"
    "phase2_transition_causal_gae_lambda_decomposition_probe_0509"
)


def _discounted_returns(
    rewards: np.ndarray,
    *,
    episode_starts: np.ndarray,
    dones: np.ndarray,
    gamma: float,
) -> np.ndarray:
    rewards = np.asarray(rewards, dtype=np.float32)
    episode_starts = np.asarray(episode_starts, dtype=np.float32)
    dones = np.asarray(dones, dtype=bool).reshape((-1,))
    out = np.zeros_like(rewards, dtype=np.float32)
    next_return = np.zeros((int(rewards.shape[1]),), dtype=np.float32)
    for step in reversed(range(int(rewards.shape[0]))):
        if step == int(rewards.shape[0]) - 1:
            next_non_terminal = 1.0 - dones.astype(np.float32, copy=False)
        else:
            next_non_terminal = 1.0 - episode_starts[step + 1].astype(np.float32, copy=False)
        next_return = rewards[step] + float(gamma) * next_non_terminal * next_return
        out[step] = next_return
    return out


def _gae(
    rewards: np.ndarray,
    values: np.ndarray,
    last_values: np.ndarray,
    *,
    episode_starts: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> np.ndarray:
    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    episode_starts = np.asarray(episode_starts, dtype=np.float32)
    last_values = np.asarray(last_values, dtype=np.float32).reshape((int(rewards.shape[1]),))
    dones = np.asarray(dones, dtype=bool).reshape((-1,))
    out = np.zeros_like(rewards, dtype=np.float32)
    last_gae_lam = np.zeros((int(rewards.shape[1]),), dtype=np.float32)
    for step in reversed(range(int(rewards.shape[0]))):
        if step == int(rewards.shape[0]) - 1:
            next_non_terminal = 1.0 - dones.astype(np.float32, copy=False)
            next_values = last_values
        else:
            next_non_terminal = 1.0 - episode_starts[step + 1].astype(np.float32, copy=False)
            next_values = values[step + 1]
        delta = rewards[step] + float(gamma) * next_non_terminal * next_values - values[step]
        last_gae_lam = delta + float(gamma) * float(gae_lambda) * next_non_terminal * last_gae_lam
        out[step] = last_gae_lam
    return out


def _center_by_objective_per_env(values: np.ndarray, objective_masks: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    mask = np.asarray(objective_masks, dtype=np.float32) > 1e-8
    out = np.array(values, copy=True, dtype=np.float32)
    for env_idx in range(int(values.shape[1])):
        m = mask[:, env_idx]
        if not bool(m.any()):
            out[:, env_idx] = 0.0
            continue
        out[:, env_idx] = values[:, env_idx] - float(np.mean(values[m, env_idx], dtype=np.float64))
    return out


def _source_row(item: dict[str, Any], source: str) -> dict[str, Any]:
    payload = item["alignment_by_credit_source"][source]
    d = payload["by_label"]
    return {
        "source": source,
        "global_score": payload["global_score"],
        "candidate_minus_not": d.get("transition_causal_candidate_minus_not"),
        "candidate_auc": d.get("transition_causal_candidate_auc"),
        "transition_primary_minus_not": d.get("transition_and_primary_minus_not"),
        "transition_primary_auc": d.get("transition_and_primary_auc"),
        "primary_minus_not": d.get("primary_long_horizon_target_minus_not"),
        "primary_auc": d.get("primary_long_horizon_target_auc"),
    }


def _weight_alignment_table(
    *,
    actions: np.ndarray,
    action_mean: np.ndarray,
    action_std: np.ndarray,
    action_masks: np.ndarray,
    objective_masks: np.ndarray,
    direction: np.ndarray,
    weights: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, weight in weights.items():
        score = _alignment_scores(
            actions=actions,
            action_mean=action_mean,
            action_std=action_std,
            action_masks=action_masks,
            objective_masks=objective_masks,
            advantages=np.asarray(weight, dtype=np.float32),
            direction=direction,
        )
        out[name] = {
            "global_score": score["global_score"],
            "by_label": _label_stats(score["per_env"], labels),
            "per_env": score["per_env"],
            "per_env_denominator": score["per_env_denominator"],
        }
    return out


def _scan_one(transition_item: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    label = str(transition_item["label"])
    print(f"[gae-lambda-decomp] collect official PPO pack: {label}", flush=True)
    algo, pack, meta = _build_pack_buffer(transition_item=transition_item, args=args)
    try:
        tokens = np.asarray(pack["tokens"], dtype=np.float32)
        actions = np.asarray(pack["actions"], dtype=np.float32)
        action_masks = np.asarray(pack["action_masks"], dtype=np.float32)
        objective_masks = np.asarray(algo.rollout_buffer.objective_masks, dtype=np.float32)
        rewards = np.asarray(algo.rollout_buffer.rewards, dtype=np.float32)
        values = np.asarray(algo.rollout_buffer.values, dtype=np.float32)
        last_values = np.asarray(pack.get("last_values"), dtype=np.float32)
        episode_starts = np.asarray(pack["episode_starts"], dtype=np.float32)
        dones = np.asarray(pack["dones"][-1], dtype=bool)
        gamma = float(algo.gamma)

        action_mean, action_std = _policy_action_mean_std(
            algo=algo,
            tokens=tokens,
            action_masks=action_masks,
        )
        direction = _direction_tensor_for_best_pairs(
            tokens=tokens,
            actions=actions,
            per_env=list(transition_item["per_env"]),
            seed=int(args.seed),
            random_action_scale=float(args.random_action_scale),
            early_frac=float(args.time_profile_early_frac),
            late_multiplier=float(args.time_profile_late_multiplier),
        )
        label_payload = _candidate_labels(list(transition_item["per_env"]))
        labels = {
            name: np.asarray(mask, dtype=bool)
            for name, mask in label_payload.items()
            if isinstance(mask, np.ndarray)
        }
        labels["selector_hard_proxy"] = np.asarray(
            [bool(row.get("selector_hard_proxy", False)) for row in transition_item["per_env"]],
            dtype=bool,
        )

        zeros = np.zeros_like(values, dtype=np.float32)
        last_zeros = np.zeros_like(last_values, dtype=np.float32)
        weights: dict[str, np.ndarray] = {
            "reward_step_centered": _center_by_objective_per_env(rewards, objective_masks),
            "zero_value_discounted_return": _center_by_objective_per_env(
                _discounted_returns(
                    rewards,
                    episode_starts=episode_starts,
                    dones=dones,
                    gamma=gamma,
                ),
                objective_masks,
            ),
            "official_values": values,
            "negative_official_values": -values,
        }
        for lam in (0.0, 0.3, 0.6, 0.9, 1.0):
            zero_gae = _gae(
                rewards,
                zeros,
                last_zeros,
                episode_starts=episode_starts,
                dones=dones,
                gamma=gamma,
                gae_lambda=float(lam),
            )
            key = f"zero_value_gae_lambda_{lam:.1f}"
            weights[f"{key}_uncentered"] = zero_gae
            weights[key] = _center_by_objective_per_env(
                zero_gae,
                objective_masks,
            )
            weights[f"{key}_train_normalized"] = _normalize_advantages_like_official(
                zero_gae,
                objective_masks,
            )
            official_gae = _gae(
                rewards,
                values,
                last_values,
                episode_starts=episode_starts,
                dones=dones,
                gamma=gamma,
                gae_lambda=float(lam),
            )
            key = f"official_value_gae_lambda_{lam:.1f}"
            weights[f"{key}_uncentered"] = official_gae
            weights[key] = _center_by_objective_per_env(
                official_gae,
                objective_masks,
            )
            weights[f"{key}_train_normalized"] = _normalize_advantages_like_official(
                official_gae,
                objective_masks,
            )
        official_actor_adv = np.asarray(algo.rollout_buffer.actor_advantages, dtype=np.float32)
        weights["official_actor_advantage_train_normalized"] = _normalize_advantages_like_official(
            official_actor_adv,
            objective_masks,
        )
        table = _weight_alignment_table(
            actions=actions,
            action_mean=action_mean,
            action_std=action_std,
            action_masks=action_masks,
            objective_masks=objective_masks,
            direction=direction,
            weights=weights,
            labels=labels,
        )
        return {
            "label": label,
            "fixed_h_list_json": str(transition_item["fixed_h_list_json"]),
            "n_envs": int(len(transition_item["per_env"])),
            "collector_meta": meta,
            "label_counts": {name: int(np.asarray(mask, dtype=bool).sum()) for name, mask in labels.items()},
            "alignment_by_credit_source": table,
            "contract": {
                "official_ppo_collect_fill_path": True,
                "no_train_step": True,
                "gae_formula_matches_sb3_style_one_step_td_recursion": True,
                "diagnostic_centering": "per-env objective-suffix mean subtraction for horizon/lambda comparisons",
            },
        }
    finally:
        del algo
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _decision(lists: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for item in lists:
        selected_sources = [
            "reward_step_centered",
            "zero_value_gae_lambda_0.0",
            "zero_value_gae_lambda_0.9",
            "zero_value_gae_lambda_0.9_uncentered",
            "zero_value_gae_lambda_0.9_train_normalized",
            "zero_value_gae_lambda_1.0",
            "official_value_gae_lambda_0.0",
            "official_value_gae_lambda_0.9",
            "official_value_gae_lambda_0.9_uncentered",
            "official_value_gae_lambda_0.9_train_normalized",
            "official_value_gae_lambda_1.0",
            "negative_official_values",
            "official_actor_advantage_train_normalized",
        ]
        rows.append(
            {
                "label": item["label"],
                "sources": [_source_row(item, source) for source in selected_sources],
            }
        )
    return {
        "rows": rows,
        "interpretation": (
            "If zero_value long-horizon sources are positive for transition_and_primary but "
            "official_value sources are not, the nearest issue is value-baseline/TD credit, "
            "not lack of raw environment signal."
        ),
    }


def _fmt(value: Any) -> str:
    return "na" if value is None else f"{float(value):+.4f}"


def _write_summary(report: dict[str, Any], output_dir: Path) -> None:
    lines = [
        "# Transition-Causal GAE/Lambda Decomposition Probe",
        "",
        "Exploratory/read-only. Same official PPO pack collect/fill path; no training.",
        "",
    ]
    for item in report["lists"]:
        lines.extend(
            [
                f"## {item['label']}",
                "",
                "| credit source | global | candidate - non | candidate AUC | trans+primary - non | trans+primary AUC |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for source in (
            "reward_step_centered",
            "zero_value_discounted_return",
            "zero_value_gae_lambda_0.0",
            "zero_value_gae_lambda_0.3",
            "zero_value_gae_lambda_0.6",
            "zero_value_gae_lambda_0.9",
            "zero_value_gae_lambda_0.9_uncentered",
            "zero_value_gae_lambda_0.9_train_normalized",
            "zero_value_gae_lambda_1.0",
            "official_values",
            "negative_official_values",
            "official_value_gae_lambda_0.0",
            "official_value_gae_lambda_0.3",
            "official_value_gae_lambda_0.6",
            "official_value_gae_lambda_0.9",
            "official_value_gae_lambda_0.9_uncentered",
            "official_value_gae_lambda_0.9_train_normalized",
            "official_value_gae_lambda_1.0",
            "official_actor_advantage_train_normalized",
        ):
            row = _source_row(item, source)
            lines.append(
                "| "
                f"{source} | {_fmt(row['global_score'])} | "
                f"{_fmt(row['candidate_minus_not'])} | {_fmt(row['candidate_auc'])} | "
                f"{_fmt(row['transition_primary_minus_not'])} | {_fmt(row['transition_primary_auc'])} |"
            )
        lines.append("")
    lines.extend(
        [
            "Interpretation:",
            "",
            "- Positive zero-value long-horizon alignment means the sampled rollout contains a usable reward signal before value subtraction.",
            "- Negative official-value GAE alignment with positive zero-value alignment means the value baseline/TD recursion is the nearest failure point.",
            "- Negative zero-value alignment means the candidate label itself is not sufficient on this sampled PPO rollout.",
        ]
    )
    (output_dir / "transition_causal_gae_lambda_decomposition_probe_summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transition-report", type=str, default=DEFAULT_TRANSITION_REPORT)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=9090)
    parser.add_argument("--n-steps", type=int, default=128)
    parser.add_argument("--single-eval-pos", type=int, default=64)
    parser.add_argument("--random-action-scale", type=float, default=1.0)
    parser.add_argument("--time-profile-early-frac", type=float, default=0.35)
    parser.add_argument("--time-profile-late-multiplier", type=float, default=0.25)
    parser.add_argument("--sb3-reward-normalization-enabled", type=lambda x: str(x).lower() in {"1", "true", "yes"}, default=True)
    parser.add_argument("--sb3-observation-normalization-enabled", type=lambda x: str(x).lower() in {"1", "true", "yes"}, default=True)
    parser.add_argument("--sb3-observation-normalization-clip", type=float, default=10.0)
    parser.add_argument("--sb3-observation-normalization-epsilon", type=float, default=1e-8)
    parser.add_argument("--gain-min", type=float, default=0.7)
    parser.add_argument("--topology-state-gain-min", type=float, default=0.7)
    parser.add_argument("--topology-action-gain-min", type=float, default=0.06)
    parser.add_argument("--topology-state-to-action-ratio-max", type=float, default=15.0)
    parser.add_argument("--topology-max-attempts", type=int, default=512)
    parser.add_argument("--ppo-learning-rate", type=float, default=2e-4)
    parser.add_argument("--ppo-n-epochs", type=int, default=4)
    parser.add_argument("--ppo-vf-coef", type=float, default=0.1)
    parser.add_argument("--ppo-normalize-advantage", type=lambda x: str(x).lower() in {"1", "true", "yes"}, default=True)
    args = parser.parse_args(argv)

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    transition = _load_json(args.transition_report)
    report = {
        "analysis_entry": "phase2_transition_causal_gae_lambda_decomposition_probe",
        "config": vars(args),
        "contract": {
            "exploratory_only": True,
            "no_train_step": True,
            "does_not_modify_prior_or_ppo": True,
        },
        "lists": [_scan_one(item, args) for item in transition["lists"]],
    }
    report["decision"] = _decision(report["lists"])
    (out_dir / "transition_causal_gae_lambda_decomposition_probe_report.json").write_text(
        json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_summary(report, out_dir)
    print(out_dir / "transition_causal_gae_lambda_decomposition_probe_summary.md")


if __name__ == "__main__":
    main()
