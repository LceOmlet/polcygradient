#!/usr/bin/env python
"""Credit decomposition for the transition-causal initial PPO signal.

Exploratory/read-only.  This script reuses the canonical pack-runner official
PPO collect/fill path and does not train or mutate milestone code.

It separates three hypotheses:

1. Environment return itself does not support the intervention-beneficial
   direction in the sampled PPO rollout.
2. Raw return supports it, but VecNorm/value/GAE credit assignment reverses or
   erases it.
3. All credit signals support it, so failure lies later in policy update/readout.
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
    DEFAULT_OUTPUT_DIR as _INITIAL_DEFAULT_OUTPUT_DIR,
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
from ticl.sb3_recurrent_ppo import _discounted_returns_from_rewards  # noqa: E402


DEFAULT_OUTPUT_DIR = (
    "/home/chen/RLPFN/artifacts/"
    "phase2_transition_causal_credit_decomposition_probe_0509"
)


def _center_by_objective_per_env(values: np.ndarray, objective_masks: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    mask = np.asarray(objective_masks, dtype=np.float32) > 1e-8
    out = np.array(values, copy=True, dtype=np.float32)
    for env_idx in range(int(values.shape[1])):
        m = mask[:, env_idx]
        if not bool(m.any()):
            out[:, env_idx] = 0.0
            continue
        mean = float(np.mean(values[m, env_idx], dtype=np.float64))
        out[:, env_idx] = values[:, env_idx] - mean
    return out


def _raw_value_target_like(buffer) -> np.ndarray:
    means = np.asarray(buffer.rollout_return_means, dtype=np.float32)
    stds = np.asarray(buffer.rollout_return_stds, dtype=np.float32)
    returns = np.asarray(buffer.returns, dtype=np.float32)
    return means + (stds * returns)


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
    rows: dict[str, Any] = {}
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
        rows[name] = {
            "global_score": score["global_score"],
            "by_label": _label_stats(score["per_env"], labels),
            "per_env": score["per_env"],
            "per_env_denominator": score["per_env_denominator"],
        }
    return rows


def _scan_one(transition_item: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    label = str(transition_item["label"])
    print(f"[credit-decomp] collect official PPO pack: {label}", flush=True)
    algo, pack, meta = _build_pack_buffer(transition_item=transition_item, args=args)
    try:
        tokens = np.asarray(pack["tokens"], dtype=np.float32)
        actions = np.asarray(pack["actions"], dtype=np.float32)
        action_masks = np.asarray(pack["action_masks"], dtype=np.float32)
        objective_masks = np.asarray(algo.rollout_buffer.objective_masks, dtype=np.float32)
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
        final_dones = np.asarray(pack["dones"][-1], dtype=bool)
        episode_starts = np.asarray(pack["episode_starts"], dtype=np.float32)
        raw_mc = _discounted_returns_from_rewards(
            np.asarray(pack["raw_rewards"], dtype=np.float32),
            episode_starts=episode_starts,
            dones=final_dones,
            gamma=float(algo.gamma),
        )
        buffer_mc = _discounted_returns_from_rewards(
            np.asarray(algo.rollout_buffer.rewards, dtype=np.float32),
            episode_starts=episode_starts,
            dones=final_dones,
            gamma=float(algo.gamma),
        )
        official_adv = np.asarray(algo.rollout_buffer.actor_advantages, dtype=np.float32)
        official_adv_norm = _normalize_advantages_like_official(official_adv, objective_masks)
        value_residual = np.asarray(algo.rollout_buffer.returns - algo.rollout_buffer.values, dtype=np.float32)
        raw_value_target = _raw_value_target_like(algo.rollout_buffer)
        raw_value_residual = raw_value_target - (
            np.asarray(algo.rollout_buffer.rollout_return_means, dtype=np.float32)
            + np.asarray(algo.rollout_buffer.rollout_return_stds, dtype=np.float32)
            * np.asarray(algo.rollout_buffer.values, dtype=np.float32)
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
        weights = {
            "env_raw_mc_return": raw_mc,
            "env_raw_mc_return_centered": _center_by_objective_per_env(raw_mc, objective_masks),
            "vecnorm_buffer_mc_return": buffer_mc,
            "vecnorm_buffer_mc_return_centered": _center_by_objective_per_env(buffer_mc, objective_masks),
            "official_value_residual": value_residual,
            "official_value_residual_raw_space": raw_value_residual,
            "official_actor_advantage": official_adv,
            "official_actor_advantage_train_normalized": official_adv_norm,
        }
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
        per_env = []
        for env_idx, row in enumerate(transition_item["per_env"]):
            per_env.append(
                {
                    "env_index": int(row.get("env_index", env_idx)),
                    "best_pair_name": row.get("best_pair_name"),
                    "best_pair_sign": row.get("best_pair_sign"),
                    "transition_causal_candidate": bool(labels["transition_causal_candidate"][env_idx]),
                    "primary_long_horizon_target": bool(labels["primary_long_horizon_target"][env_idx]),
                    "transition_and_primary": bool(labels["transition_and_primary"][env_idx]),
                    **{
                        f"{name}_alignment": float(payload["per_env"][env_idx])
                        for name, payload in table.items()
                    },
                }
            )
        return {
            "label": label,
            "fixed_h_list_json": str(transition_item["fixed_h_list_json"]),
            "n_envs": int(len(per_env)),
            "collector_meta": meta,
            "label_counts": {name: int(np.asarray(mask, dtype=bool).sum()) for name, mask in labels.items()},
            "alignment_by_credit_source": table,
            "per_env": per_env,
            "contract": {
                "official_ppo_collect_fill_path": True,
                "no_train_step": True,
                "environment_raw_reward_used_only_for_diagnostic_return": True,
                "ppo_credit_sources_use_rollout_buffer_values": True,
            },
        }
    finally:
        del algo
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


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


def _decision(lists: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for item in lists:
        source_rows = [_source_row(item, source) for source in item["alignment_by_credit_source"]]
        rows.append({"label": item["label"], "sources": source_rows})

    raw_support = all(
        float(_source_row(item, "env_raw_mc_return_centered")["candidate_minus_not"] or 0.0) > 0.02
        for item in lists
    )
    vecnorm_support = all(
        float(_source_row(item, "vecnorm_buffer_mc_return_centered")["candidate_minus_not"] or 0.0) > 0.02
        for item in lists
    )
    gae_support = all(
        float(_source_row(item, "official_actor_advantage_train_normalized")["candidate_minus_not"] or 0.0) > 0.02
        for item in lists
    )
    return {
        "raw_return_supports_candidate": bool(raw_support),
        "vecnorm_return_supports_candidate": bool(vecnorm_support),
        "official_actor_advantage_supports_candidate": bool(gae_support),
        "credit_assignment_flip": bool(raw_support and not gae_support),
        "rows": rows,
        "interpretation": (
            "raw_return_supports_candidate=false means this transition label is not a sufficient "
            "environment repair target on the sampled PPO rollouts.  raw=true and actor=false "
            "would instead implicate PPO credit assignment/value/VecNorm/objective masking."
        ),
    }


def _fmt(value: Any) -> str:
    return "na" if value is None else f"{float(value):+.4f}"


def _write_summary(report: dict[str, Any], output_dir: Path) -> None:
    lines = [
        "# Transition-Causal Credit Decomposition Probe",
        "",
        "Exploratory/read-only. Same official PPO pack collect/fill path; no training.",
        "",
        f"raw_return_supports_candidate: `{report['decision']['raw_return_supports_candidate']}`",
        f"vecnorm_return_supports_candidate: `{report['decision']['vecnorm_return_supports_candidate']}`",
        f"official_actor_advantage_supports_candidate: `{report['decision']['official_actor_advantage_supports_candidate']}`",
        f"credit_assignment_flip: `{report['decision']['credit_assignment_flip']}`",
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
            "env_raw_mc_return",
            "env_raw_mc_return_centered",
            "vecnorm_buffer_mc_return",
            "vecnorm_buffer_mc_return_centered",
            "official_value_residual",
            "official_value_residual_raw_space",
            "official_actor_advantage",
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
            "- If raw centered return is not positive for candidates, the current transition-causal label is still a surface label rather than a sufficient missing-mode repair.",
            "- If raw centered return is positive but official actor advantage is not, the nearest bug/limitation is in PPO credit assignment or value/VecNorm/objective semantics.",
        ]
    )
    (output_dir / "transition_causal_credit_decomposition_probe_summary.md").write_text(
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
        "analysis_entry": "phase2_transition_causal_credit_decomposition_probe",
        "config": vars(args),
        "contract": {
            "exploratory_only": True,
            "no_train_step": True,
            "does_not_modify_prior_or_ppo": True,
            "initial_probe_output_dir": _INITIAL_DEFAULT_OUTPUT_DIR,
        },
        "lists": [_scan_one(item, args) for item in transition["lists"]],
    }
    report["decision"] = _decision(report["lists"])
    (out_dir / "transition_causal_credit_decomposition_probe_report.json").write_text(
        json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_summary(report, out_dir)
    print(out_dir / "transition_causal_credit_decomposition_probe_summary.md")


if __name__ == "__main__":
    main()
