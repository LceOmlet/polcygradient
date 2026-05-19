#!/usr/bin/env python
"""Initial-PPO-signal probe for transition-causal prior modes.

This is an exploratory/read-only probe.  It does not mutate the exact SCM,
fit_model defaults, PPO implementation, or prior milestone definitions.

Question:
    When the transition-causal intervention probe says that one signed
    feedback direction is beneficial, does the official PPO rollout/update
    path produce an initial actor signal in that same direction?

Why this is the highest-value split:
    If the official PPO advantage-weighted score-function signal is not
    aligned with the oracle intervention direction, then changing the prior
    generator around that label is unlikely to fix Pendulum-like failure by
    itself.  The failure is closer to objective/readout/exploration.

    If it is aligned, but short closed-loop training still does not improve,
    the label may be useful but the current closed-loop optimizer does not
    convert the signal reliably.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.exploratory.phase2_prior_action_mode_coverage_probe import (  # noqa: E402
    _load_fixed_h_list,
    _summary,
)
from scripts.exploratory.phase2_prior_pendulum_like_signature_probe import (  # noqa: E402
    PendulumSignatureProbePolicy,
)
from scripts.exploratory.phase2_transition_causal_context_gate import (  # noqa: E402
    _candidate_labels,
)
from scripts.exploratory.phase2_transition_causal_paired_intervention_gate import (  # noqa: E402
    _binary_auc,
)
from ticl.analysis import phase2_gym_prior_pack_training_runner as runner  # noqa: E402


DEFAULT_TRANSITION_REPORT = (
    "/home/chen/RLPFN/artifacts/phase2_signed_transition_causal_path_probe_0509/"
    "signed_transition_causal_path_probe_report.json"
)
DEFAULT_OUTPUT_DIR = (
    "/home/chen/RLPFN/artifacts/"
    "phase2_transition_causal_initial_ppo_signal_probe_0509"
)


_PAIR_RE = re.compile(r"^(?P<family>constant|decay)_f(?P<feature>\d+)_x(?P<scale>\d+)$")


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _parse_pair_modes(pair_name: str) -> tuple[str, str] | None:
    match = _PAIR_RE.match(str(pair_name))
    if match is None:
        return None
    prefix = "" if match.group("family") == "constant" else "decay_"
    feature = int(match.group("feature"))
    scale = int(match.group("scale"))
    return (
        f"{prefix}obs_linear_{feature}x{scale}",
        f"{prefix}obs_linear_neg_{feature}x{scale}",
    )


def _normalize_advantages_like_official(advantages: np.ndarray, objective_masks: np.ndarray) -> np.ndarray:
    adv = np.asarray(advantages, dtype=np.float32)
    mask = np.asarray(objective_masks, dtype=np.float32) > 1e-8
    if not bool(mask.any()):
        return np.zeros_like(adv, dtype=np.float32)
    valid = adv[mask].astype(np.float64, copy=False)
    if int(valid.size) <= 1:
        return np.zeros_like(adv, dtype=np.float32)
    # torch.std() in the official path is unbiased by default.
    std = float(np.std(valid, ddof=1))
    if not math.isfinite(std) or std <= 0.0:
        return np.zeros_like(adv, dtype=np.float32)
    mean = float(np.mean(valid))
    return ((adv.astype(np.float64, copy=False) - mean) / (std + 1e-8)).astype(np.float32, copy=False)


def _policy_action_mean_std(
    *,
    algo,
    tokens: np.ndarray,
    action_masks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n_steps, n_envs, num_features = tokens.shape
    obs_flat = np.asarray(tokens, dtype=np.float32).swapaxes(0, 1).reshape((int(n_steps) * int(n_envs), int(num_features)))
    seq_lengths = np.full((int(n_envs),), int(n_steps), dtype=np.int64)
    obs_t = torch.as_tensor(obs_flat, device=algo.device, dtype=torch.float32)
    masks_flat = (
        np.asarray(action_masks, dtype=np.float32)
        .swapaxes(0, 1)
        .reshape((int(n_steps) * int(n_envs), int(action_masks.shape[-1])))
    )
    masks_t = torch.as_tensor(masks_flat, device=algo.device, dtype=torch.float32)
    with torch.no_grad():
        hidden = algo.policy._official_sequence_hidden_flat(obs_t, seq_lengths=seq_lengths)
        latent_pi = algo.policy.mlp_extractor.forward_actor(hidden)
        distribution = algo.policy._get_action_dist_from_latent(latent_pi)
        mean_t = distribution.distribution.mean
        std_t = distribution.distribution.stddev
        action_mask_t = algo.policy._coerce_action_masks(masks_t, int(obs_t.shape[0]), device=obs_t.device)
        valid = action_mask_t.to(device=obs_t.device, dtype=torch.bool)
        mean_t = mean_t.masked_fill(~valid, 0.0)
        std_t = std_t.masked_fill(~valid, 1.0)
    mean = mean_t.detach().cpu().numpy().astype(np.float32, copy=False)
    std = std_t.detach().cpu().numpy().astype(np.float32, copy=False)
    action_dim = int(mean.shape[-1])
    mean = mean.reshape((int(n_envs), int(n_steps), action_dim)).swapaxes(0, 1)
    std = std.reshape((int(n_envs), int(n_steps), action_dim)).swapaxes(0, 1)
    return mean, std


def _direction_tensor_for_best_pairs(
    *,
    tokens: np.ndarray,
    actions: np.ndarray,
    per_env: list[dict[str, Any]],
    seed: int,
    random_action_scale: float,
    early_frac: float,
    late_multiplier: float,
) -> np.ndarray:
    n_steps, n_envs, action_dim = actions.shape
    direction = np.zeros((int(n_steps), int(n_envs), int(action_dim)), dtype=np.float32)
    tokens_t = torch.as_tensor(tokens, dtype=torch.float32)
    zeros_t = torch.zeros((int(n_envs), int(action_dim)), dtype=torch.float32)
    cache: dict[str, tuple[PendulumSignatureProbePolicy, PendulumSignatureProbePolicy]] = {}
    for env_idx, row in enumerate(per_env):
        pair_name = str(row.get("best_pair_name", "none"))
        pair_modes = _parse_pair_modes(pair_name)
        if pair_modes is None:
            continue
        if pair_name not in cache:
            pos_mode, neg_mode = pair_modes
            cache[pair_name] = (
                PendulumSignatureProbePolicy(
                    pos_mode,
                    seed=int(seed),
                    random_scale=float(random_action_scale),
                    n_steps=int(n_steps),
                    early_frac=float(early_frac),
                    late_multiplier=float(late_multiplier),
                ),
                PendulumSignatureProbePolicy(
                    neg_mode,
                    seed=int(seed),
                    random_scale=float(random_action_scale),
                    n_steps=int(n_steps),
                    early_frac=float(early_frac),
                    late_multiplier=float(late_multiplier),
                ),
            )
        pos_policy, neg_policy = cache[pair_name]
        sign = 1.0 if str(row.get("best_pair_sign")) == "pos" else -1.0
        for step_idx in range(int(n_steps)):
            pos = pos_policy._action_mean(tokens_t[step_idx], zeros_t, int(step_idx))
            neg = neg_policy._action_mean(tokens_t[step_idx], zeros_t, int(step_idx))
            direction[step_idx, env_idx] = (
                float(sign)
                * (pos[env_idx] - neg[env_idx]).detach().cpu().numpy().astype(np.float32, copy=False)
            )
    return direction


def _safe_mean(values: np.ndarray) -> float | None:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return float(np.mean(arr))


def _label_stats(scores: np.ndarray, labels: dict[str, np.ndarray]) -> dict[str, Any]:
    out: dict[str, Any] = {"all": _summary(scores)}
    for name, mask_raw in labels.items():
        mask = np.asarray(mask_raw, dtype=bool)
        out[name] = _summary(scores[mask])
        out[f"not_{name}"] = _summary(scores[~mask])
        out[f"{name}_minus_not"] = None
        if int(mask.sum()) > 0 and int((~mask).sum()) > 0:
            out[f"{name}_minus_not"] = float(np.mean(scores[mask]) - np.mean(scores[~mask]))
        out[f"{name}_auc"] = _binary_auc(mask, np.asarray(scores, dtype=np.float64))
    return out


def _alignment_scores(
    *,
    actions: np.ndarray,
    action_mean: np.ndarray,
    action_std: np.ndarray,
    action_masks: np.ndarray,
    objective_masks: np.ndarray,
    advantages: np.ndarray,
    direction: np.ndarray,
) -> dict[str, Any]:
    mask = (np.asarray(action_masks, dtype=np.float32) > 1e-8).astype(np.float32, copy=False)
    obj = (np.asarray(objective_masks, dtype=np.float32) > 1e-8).astype(np.float32, copy=False)
    adv = np.asarray(advantages, dtype=np.float32)
    score_fn = (np.asarray(actions, dtype=np.float32) - np.asarray(action_mean, dtype=np.float32)) / np.square(
        np.maximum(np.asarray(action_std, dtype=np.float32), 1e-6)
    )
    direction = np.asarray(direction, dtype=np.float32)
    signed = adv[:, :, None] * score_fn * direction * mask * obj[:, :, None]
    denom = np.abs(adv[:, :, None] * score_fn * direction) * mask * obj[:, :, None]
    per_env_num = signed.sum(axis=(0, 2)).astype(np.float64, copy=False)
    per_env_den = denom.sum(axis=(0, 2)).astype(np.float64, copy=False)
    per_env = np.divide(
        per_env_num,
        np.maximum(per_env_den, 1e-12),
        out=np.zeros_like(per_env_num, dtype=np.float64),
        where=per_env_den > 1e-12,
    )
    global_score = float(np.sum(per_env_num) / max(float(np.sum(per_env_den)), 1e-12))

    direction_abs = np.abs(direction) * mask * obj[:, :, None]
    action_projection_num = (np.asarray(actions, dtype=np.float32) * direction * mask * obj[:, :, None]).sum(axis=(0, 2))
    action_projection_den = np.maximum(direction_abs.sum(axis=(0, 2)), 1e-12)
    action_projection = (action_projection_num / action_projection_den).astype(np.float64, copy=False)
    return {
        "global_score": global_score,
        "per_env": per_env.astype(np.float32, copy=False),
        "per_env_numerator": per_env_num.astype(np.float32, copy=False),
        "per_env_denominator": per_env_den.astype(np.float32, copy=False),
        "action_projection_per_env": action_projection.astype(np.float32, copy=False),
    }


def _build_pack_buffer(
    *,
    transition_item: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[Any, dict[str, np.ndarray], dict[str, Any]]:
    cfg = copy.deepcopy(runner.get_model_default_config("rlpfn"))
    runner.validate_rlpfn_maintained_path_config(cfg)
    env_cfg = copy.deepcopy(cfg["prior"]["environment"])
    num_features = int(cfg["prior"]["num_features"])
    layout = runner.resolve_rlpfn_token_layout(env_cfg, num_features=int(num_features))
    obs_slot_dim = int(layout["obs_slot_dim"])
    action_slot_dim = int(layout["action_slot_dim"])
    h_list, seed_list = _load_fixed_h_list(transition_item["fixed_h_list_json"], limit=int(transition_item["n_envs"]))
    n_envs = int(len(h_list))
    env_seeds = [int(v) for v in seed_list] if seed_list is not None else [int(args.seed) + i for i in range(n_envs)]
    algo, vec_env = runner._build_algo(
        cfg=cfg,
        env_cfg=env_cfg,
        model_override=None,
        frozen_h=None,
        frozen_h_list=h_list,
        frozen_h_list_env_seeds=env_seeds,
        prior_mode="fixed_frozen_list",
        device_obj=torch.device(str(args.device)),
        num_features=int(num_features),
        n_envs=int(n_envs),
        n_steps=int(args.n_steps),
        build_seed=int(args.seed),
        fixed_single_eval_pos=int(args.single_eval_pos),
        sb3_reward_normalization_enabled=bool(args.sb3_reward_normalization_enabled),
        sb3_observation_normalization_enabled=bool(args.sb3_observation_normalization_enabled),
        sb3_observation_normalization_clip=float(args.sb3_observation_normalization_clip),
        sb3_observation_normalization_epsilon=float(args.sb3_observation_normalization_epsilon),
        gain_min=float(args.gain_min),
        topology_state_gain_min=float(args.topology_state_gain_min),
        topology_action_gain_min=float(args.topology_action_gain_min),
        topology_state_to_action_ratio_max=float(args.topology_state_to_action_ratio_max),
        topology_max_attempts=int(args.topology_max_attempts),
        fixed_env_group_across_updates=True,
        ppo_learning_rate=float(args.ppo_learning_rate),
        ppo_batch_size=None,
        ppo_n_epochs=int(args.ppo_n_epochs),
        ppo_vf_coef=float(args.ppo_vf_coef),
        ppo_normalize_advantage=bool(args.ppo_normalize_advantage),
        ppo_target_kl=None,
    )
    try:
        pack, values, log_probs, meta = runner._collect_prior_policy_pack(
            algo=algo,
            vec_env=vec_env,
            n_steps=int(args.n_steps),
            seed=int(args.seed),
            num_features=int(num_features),
            obs_slot_dim=int(obs_slot_dim),
            action_slot_dim=int(action_slot_dim),
            next_state_target_dim=0,
            store_next_state_targets=False,
        )
        buffer_pack, reward_norm_stats = runner._apply_reward_normalization_if_enabled(algo, pack)
        buffer_summary = runner._fill_existing_rollout_buffer_from_pack(
            algo.rollout_buffer,
            buffer_pack,
            num_features=int(num_features),
            action_slot_dim=int(action_slot_dim),
            next_state_target_dim=0,
            single_eval_pos=int(args.single_eval_pos),
            values_by_step_env=values,
            log_probs_by_step_env=log_probs,
            last_values_by_env=pack.get("last_values"),
            compute_returns=True,
        )
        meta = {
            **dict(meta),
            "buffer_summary": buffer_summary,
            "reward_norm": reward_norm_stats,
            "num_features": int(num_features),
            "obs_slot_dim": int(obs_slot_dim),
            "action_slot_dim": int(action_slot_dim),
        }
        return algo, pack, meta
    finally:
        vec_env.close()


def _scan_one(transition_item: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    label = str(transition_item["label"])
    print(f"[initial-ppo-signal] collect official PPO pack: {label}", flush=True)
    algo, pack, meta = _build_pack_buffer(transition_item=transition_item, args=args)
    try:
        tokens = np.asarray(pack["tokens"], dtype=np.float32)
        actions = np.asarray(pack["actions"], dtype=np.float32)
        action_masks = np.asarray(pack["action_masks"], dtype=np.float32)
        objective_masks = np.asarray(algo.rollout_buffer.objective_masks, dtype=np.float32)
        raw_adv = np.asarray(algo.rollout_buffer.actor_advantages, dtype=np.float32)
        norm_adv = _normalize_advantages_like_official(raw_adv, objective_masks)
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
        raw_scores = _alignment_scores(
            actions=actions,
            action_mean=action_mean,
            action_std=action_std,
            action_masks=action_masks,
            objective_masks=objective_masks,
            advantages=raw_adv,
            direction=direction,
        )
        norm_scores = _alignment_scores(
            actions=actions,
            action_mean=action_mean,
            action_std=action_std,
            action_masks=action_masks,
            objective_masks=objective_masks,
            advantages=norm_adv,
            direction=direction,
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
                    "raw_adv_alignment": float(raw_scores["per_env"][env_idx]),
                    "normalized_adv_alignment": float(norm_scores["per_env"][env_idx]),
                    "action_projection_on_beneficial_direction": float(
                        norm_scores["action_projection_per_env"][env_idx]
                    ),
                    "alignment_denominator": float(norm_scores["per_env_denominator"][env_idx]),
                    "raw_adv_mean_suffix": _safe_mean(raw_adv[:, env_idx][objective_masks[:, env_idx] > 1e-8]),
                    "norm_adv_mean_suffix": _safe_mean(norm_adv[:, env_idx][objective_masks[:, env_idx] > 1e-8]),
                }
            )
        return {
            "label": label,
            "fixed_h_list_json": str(transition_item["fixed_h_list_json"]),
            "n_envs": int(len(per_env)),
            "collector_meta": meta,
            "contract": {
                "ppo_path": "pack runner official PPO collect -> reward norm -> fill official rollout buffer",
                "no_train_step": True,
                "score_function_signal": "advantage * (sampled_action - policy_mean) / policy_std^2",
                "beneficial_direction": (
                    "per-env best signed obs-linear pair from transition-causal path probe, "
                    "evaluated on the PPO-visible token sequence"
                ),
                "environment_side_not_modified": True,
            },
            "label_counts": {name: int(np.asarray(mask, dtype=bool).sum()) for name, mask in labels.items()},
            "raw_advantage_alignment": {
                "global_score": raw_scores["global_score"],
                "by_label": _label_stats(raw_scores["per_env"], labels),
            },
            "normalized_advantage_alignment": {
                "global_score": norm_scores["global_score"],
                "by_label": _label_stats(norm_scores["per_env"], labels),
            },
            "action_projection_by_label": _label_stats(norm_scores["action_projection_per_env"], labels),
            "per_env": per_env,
        }
    finally:
        del algo
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _decision(lists: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for item in lists:
        by_label = item["normalized_advantage_alignment"]["by_label"]
        rows.append(
            {
                "label": item["label"],
                "transition_candidate_minus_not": by_label.get("transition_causal_candidate_minus_not"),
                "transition_and_primary_minus_not": by_label.get("transition_and_primary_minus_not"),
                "primary_minus_not": by_label.get("primary_long_horizon_target_minus_not"),
                "transition_candidate_auc": by_label.get("transition_causal_candidate_auc"),
                "transition_and_primary_auc": by_label.get("transition_and_primary_auc"),
                "global_score": item["normalized_advantage_alignment"].get("global_score"),
            }
        )
    clean_candidate_advantage = all(
        row["transition_candidate_minus_not"] is not None and float(row["transition_candidate_minus_not"]) > 0.05
        for row in rows
    )
    clean_candidate_auc = all(
        row["transition_candidate_auc"] is not None and float(row["transition_candidate_auc"]) >= 0.65
        for row in rows
    )
    return {
        "candidate_signal_cleanly_present": bool(clean_candidate_advantage and clean_candidate_auc),
        "rows": rows,
        "interpretation": (
            "If false, the current transition-causal label is not yet a sufficient generator repair axis: "
            "official PPO initial score-function signal does not consistently privilege that label."
        ),
    }


def _write_summary(report: dict[str, Any], output_dir: Path) -> None:
    lines = [
        "# Transition-Causal Initial PPO Signal Probe",
        "",
        "Exploratory/read-only probe. It uses the pack runner official PPO collect/fill path, "
        "then measures whether initial actor advantages point along the per-env intervention-beneficial direction.",
        "",
        f"Decision candidate_signal_cleanly_present: `{report['decision']['candidate_signal_cleanly_present']}`",
        "",
        "| list | n | global normalized alignment | candidate - non | candidate AUC | trans+primary - non | trans+primary AUC | action projection cand-non |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report["lists"]:
        d = item["normalized_advantage_alignment"]["by_label"]
        ap = item["action_projection_by_label"]
        def fmt(x: Any) -> str:
            return "na" if x is None else f"{float(x):+.4f}"
        lines.append(
            "| "
            f"{item['label']} | {item['n_envs']} | "
            f"{fmt(item['normalized_advantage_alignment']['global_score'])} | "
            f"{fmt(d.get('transition_causal_candidate_minus_not'))} | "
            f"{fmt(d.get('transition_causal_candidate_auc'))} | "
            f"{fmt(d.get('transition_and_primary_minus_not'))} | "
            f"{fmt(d.get('transition_and_primary_auc'))} | "
            f"{fmt(ap.get('transition_causal_candidate_minus_not'))} |"
        )
    lines.extend(
        [
            "",
            "Interpretation rule:",
            "",
            "- Clean positive candidate alignment would support turning this label into a repair candidate.",
            "- Weak, negative, or inconsistent alignment falsifies the idea that the current label alone is the next clean milestone axis.",
            "- Positive action projection with negative advantage alignment means sampled actions may point slightly the right way, but the official PPO initial objective does not reinforce that direction.",
        ]
    )
    (output_dir / "transition_causal_initial_ppo_signal_probe_summary.md").write_text(
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
        "analysis_entry": "phase2_transition_causal_initial_ppo_signal_probe",
        "config": vars(args),
        "contract": {
            "exploratory_only": True,
            "ppo_and_environment_separated": True,
            "ppo_consumes_pack_tokens_actions_rewards": True,
            "environment_side_only_used_for_precomputed_intervention_direction": True,
        },
        "lists": [_scan_one(item, args) for item in transition["lists"]],
    }
    report["decision"] = _decision(report["lists"])
    (out_dir / "transition_causal_initial_ppo_signal_probe_report.json").write_text(
        json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_summary(report, out_dir)
    print(out_dir / "transition_causal_initial_ppo_signal_probe_summary.md")


if __name__ == "__main__":
    main()
