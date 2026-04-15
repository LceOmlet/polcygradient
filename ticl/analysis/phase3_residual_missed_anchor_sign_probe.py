import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ticl.analysis.phase2_guardrail import DEFAULT_PHASE2_SUMMARY, assert_phase2_green
from ticl.analysis.critic_free_single_env_audit import _make_deterministic_batch_plan
from ticl.analysis.phase3_heldout_reset_semantics_probe import (
    _objective_episode_segments,
    _objective_terminal_reset_count_from_segments,
)
from ticl.analysis.phase3_multi_env_optimization_audit import (
    _bind_vec_env_to_fixed_suite,
    _default_device,
    _load_required_suite,
    _load_reused_suite_metrics,
    _resolve_train_profile,
)
from ticl.analysis.phase3_dual_channel_anchor_coverage_probe import (
    build_dual_channel_anchor_coverage_probe,
)
from ticl.analysis.phase3_positive_regime_anchor_structure_probe import (
    build_positive_regime_anchor_structure_probe,
)
from ticl.analysis.phase3_train_rollout_quality_probe import _collect_one_rollout
from ticl.analysis.prior_generalization_audit import _resolve_audit_env_config, _summarize_suite
from ticl.model_builder import load_model
from ticl.priors.environment_prior import EnvironmentPrior
from ticl.sb3_recurrent_ppo import _normalize_advantages_with_mask, build_recurrent_ppo


STAGE_PREFIXES = (
    "raw_return",
    "raw_residual",
    "raw_actor_adv",
    "normalized_actor_adv",
)
SIGN_STATE_EPS = 0.10

DEFAULT_SUITE_SPECS = [
    {
        "suite_name": "pair1",
        "zero_json": "/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/zero_cross_env_control.json",
        "ppo_json": "/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/ppo_cross_env_baseline.json",
        "distance_probe_json": "/home/chen/RLPFN/artifacts/phase3_post_reset_tail_distance_probe_pair1_suite_matched.json",
        "tail_mask_probe_json": "/home/chen/RLPFN/artifacts/phase3_terminal_tail_mask_probe_pair1_suite_matched.json",
        "train_suite_path": "/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/train_suite.pt",
        "heldout_suite_path": "/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/heldout_suite.pt",
    },
    {
        "suite_name": "pair2",
        "zero_json": "/home/chen/RLPFN/artifacts/phase3_cross_env_baseline_pair2/zero_cross_env_control.json",
        "ppo_json": "/home/chen/RLPFN/artifacts/phase3_cross_env_baseline_pair2/ppo_cross_env_baseline.json",
        "distance_probe_json": "/home/chen/RLPFN/artifacts/phase3_post_reset_tail_distance_probe_pair2.json",
        "tail_mask_probe_json": "/home/chen/RLPFN/artifacts/phase3_terminal_tail_mask_probe_pair2.json",
        "train_suite_path": "/home/chen/RLPFN/artifacts/phase3_cross_env_baseline_pair2/suites/train_suite.pt",
        "heldout_suite_path": "/home/chen/RLPFN/artifacts/phase3_cross_env_baseline_pair2/suites/heldout_suite.pt",
    },
    {
        "suite_name": "seed13579",
        "zero_json": "/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_13579/zero_cross_env_control.json",
        "ppo_json": "/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_13579/ppo_cross_env_baseline.json",
        "distance_probe_json": "/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_13579/phase3_post_reset_tail_distance_probe.json",
        "tail_mask_probe_json": "/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_13579/phase3_terminal_tail_mask_probe.json",
        "train_suite_path": "/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_13579/train_suite.pt",
        "heldout_suite_path": "/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_13579/heldout_suite.pt",
    },
]


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(float(v) for v in values) / len(values))


def _safe_ratio(num: float, den: float) -> float:
    den_f = float(den)
    if den_f <= 0.0:
        return 0.0
    return float(float(num) / den_f)


def _rank_desc(rows: list[dict[str, Any]], key: str) -> dict[int, int]:
    ordered = sorted(rows, key=lambda row: float(row[key]), reverse=True)
    return {int(row["env_index"]): int(rank) for rank, row in enumerate(ordered, start=1)}


def _positive_negative_mass(values: np.ndarray) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    pos = float(np.clip(arr, a_min=0.0, a_max=None).sum())
    neg = float(np.clip(-arr, a_min=0.0, a_max=None).sum())
    abs_mass = float(np.abs(arr).sum())
    return pos, neg, abs_mass


def _stage_stats(values: np.ndarray, *, prefix: str) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    pos, neg, abs_mass = _positive_negative_mass(arr)
    balance = float((pos - neg) / abs_mass) if abs_mass > 0.0 else 0.0
    out = {
        f"{prefix}_mean": float(arr.mean()) if int(arr.size) > 0 else 0.0,
        f"{prefix}_std": float(arr.std()) if int(arr.size) > 0 else 0.0,
        f"{prefix}_positive_mass": pos,
        f"{prefix}_negative_mass": neg,
        f"{prefix}_abs_mass": abs_mass,
        f"{prefix}_positive_count": int(np.count_nonzero(arr > 0.0)),
        f"{prefix}_negative_count": int(np.count_nonzero(arr < 0.0)),
        f"{prefix}_balance": balance,
    }
    return out


def _sign_state(positive_mass: float, negative_mass: float, *, eps: float = SIGN_STATE_EPS) -> str:
    total = float(positive_mass) + float(negative_mass)
    if total <= 0.0:
        return "zero"
    balance = float((float(positive_mass) - float(negative_mass)) / total)
    if balance >= float(eps):
        return "positive"
    if balance <= -float(eps):
        return "wrong_sign"
    return "near_zero_or_balanced"


def _dominant_label(named_values: dict[str, float], *, tol: float = 1e-8) -> str:
    if not named_values:
        return "none"
    label, value = max(named_values.items(), key=lambda item: float(item[1]))
    if float(value) <= float(tol):
        return "none"
    return str(label)


def _missed_anchor_keys(
    *,
    suite_specs: list[dict[str, Any]],
    anchor_source: str,
    anchor_min_pre_gap: float,
    captured_min_mass_share_within_suite: float,
) -> tuple[dict[str, dict[int, dict[str, Any]]], dict[str, Any]]:
    source = str(anchor_source).strip().lower()
    if source == "dual_channel_residual_missed":
        anchor_payload = build_dual_channel_anchor_coverage_probe(
            suite_specs,
            strong_gain_min_pre_gap=float(anchor_min_pre_gap),
            coverage_share_threshold=float(captured_min_mass_share_within_suite),
        )
        anchor_rows = list(anchor_payload["residual_missed_summary"]["env_items"])
    elif source == "positive_regime_missed":
        anchor_payload = build_positive_regime_anchor_structure_probe(
            suite_specs,
            anchor_min_pre_gap=float(anchor_min_pre_gap),
            captured_min_mass_share_within_suite=float(captured_min_mass_share_within_suite),
        )
        anchor_rows = list(anchor_payload["missed_summary"]["env_items"])
    else:
        raise ValueError(f"Unsupported anchor_source={anchor_source!r}")
    missed = {}
    for row in anchor_rows:
        suite_name = str(row["suite_name"])
        env_index = int(row["env_index"])
        missed.setdefault(suite_name, {})[env_index] = dict(row)
    return missed, anchor_payload


def _collect_suite_rows(
    *,
    checkpoint_path: str,
    suite_spec: dict[str, Any],
    device_obj: torch.device,
    n_samples: int,
    single_eval_pos: int,
    train_profile: str,
    batch_size: int | None,
    n_epochs: int | None,
    learning_rate: float | None,
    target_kl: float | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    train_suite = _load_required_suite(str(suite_spec["train_suite_path"]))
    heldout_suite = _load_required_suite(str(suite_spec["heldout_suite_path"]))

    load_model.cache_clear()
    model, config = load_model(checkpoint_path, device=device_obj, verbose=False)
    env_cfg = _resolve_audit_env_config(config, core_a=False, reference_semantics_enabled=False)
    eval_prior = EnvironmentPrior(copy.deepcopy(env_cfg))
    eval_prior._sample_single_eval_pos = lambda n_samples_arg, rng_seed=None: int(single_eval_pos)
    train_suite_summary = _summarize_suite(eval_prior, train_suite)
    heldout_suite_summary = _summarize_suite(eval_prior, heldout_suite)

    reused_zero = _load_reused_suite_metrics(
        report_json_path=str(suite_spec["zero_json"]),
        expected_policy_mode="zero",
        train_suite_summary=train_suite_summary,
        heldout_suite_summary=heldout_suite_summary,
        n_samples=int(n_samples),
        single_eval_pos=int(single_eval_pos),
        rollout_backend="serial",
    )
    reused_pre = _load_reused_suite_metrics(
        report_json_path=str(suite_spec["ppo_json"]),
        expected_policy_mode="ppo",
        train_suite_summary=train_suite_summary,
        heldout_suite_summary=heldout_suite_summary,
        n_samples=int(n_samples),
        single_eval_pos=int(single_eval_pos),
        rollout_backend="serial",
    )

    optimizer_cfg = dict(config.get("optimizer", {}))
    profile_cfg = _resolve_train_profile(
        optimizer_cfg=optimizer_cfg,
        train_profile=str(train_profile),
        batch_size=batch_size,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        target_kl=target_kl,
    )

    prior = EnvironmentPrior(copy.deepcopy(env_cfg))
    prior._sample_batch_hypers = lambda batch_n: [copy.deepcopy(h) for h in list(heldout_suite["h_list"])]
    prior._sample_single_eval_pos = lambda n_samples_arg, rng_seed=None: int(single_eval_pos)

    algo, callback, vec_env = build_recurrent_ppo(
        model=model,
        env_prior=prior,
        device=str(device_obj),
        num_features=int(config["prior"]["num_features"]),
        n_envs=int(heldout_suite["batch_size"]),
        n_steps=int(n_samples),
        learning_rate=float(profile_cfg["learning_rate"]),
        batch_size=int(profile_cfg["batch_size"]),
        n_epochs=int(profile_cfg["n_epochs"]),
        gamma=float(optimizer_cfg.get("ppo_gamma", 1.0)),
        gae_lambda=float(optimizer_cfg.get("ppo_gae_lambda", 0.95)),
        clip_range=optimizer_cfg.get("ppo_clip_range", 0.2),
        clip_range_vf=None,
        normalize_advantage=bool(profile_cfg["normalize_advantage"]),
        actor_gae_space=str(profile_cfg["actor_gae_space"]),
        actor_baseline_mode=str(profile_cfg["actor_baseline_mode"]),
        actor_objective_mode=str(profile_cfg["actor_objective_mode"]),
        reset_env_state_at_sep=bool(profile_cfg["reset_env_state_at_sep"]),
        separate_value_backbone=bool(profile_cfg["separate_value_backbone"]),
        restore_validation_policy_state=True,
        runtime_normalized_q_value_weight_override=profile_cfg["runtime_normalized_q_value_weight_override"],
        runtime_next_state_flow_matching_weight_override=profile_cfg["runtime_next_state_flow_matching_weight_override"],
        ent_coef=float(optimizer_cfg.get("ppo_ent_coef", 0.0)),
        vf_coef=float(profile_cfg["vf_coef"]),
        max_grad_norm=float(optimizer_cfg.get("ppo_max_grad_norm", 0.5)),
        target_kl=profile_cfg["target_kl"],
        verbose=0,
    )
    _bind_vec_env_to_fixed_suite(vec_env, suite=heldout_suite, single_eval_pos=int(single_eval_pos))
    _make_deterministic_batch_plan(algo.rollout_buffer)
    suite_t0 = time.perf_counter()
    _collect_one_rollout(algo, callback, vec_env)
    suite_runtime_wall_s = float(time.perf_counter() - suite_t0)

    rollout_buffer = algo.rollout_buffer
    objective_masks = np.asarray(rollout_buffer.objective_masks, dtype=np.float32)
    episode_starts = np.asarray(rollout_buffer.episode_starts, dtype=np.float32)
    actor_advantages = np.asarray(rollout_buffer.actor_advantages, dtype=np.float32)
    raw_values = (
        rollout_buffer._get_flat_raw_values_tensor(dtype=torch.float32).detach().cpu().numpy().reshape(objective_masks.shape)
    )
    raw_returns = (
        rollout_buffer._get_flat_raw_returns_tensor(dtype=torch.float32).detach().cpu().numpy().reshape(objective_masks.shape)
    )

    objective_flat = torch.as_tensor(objective_masks.reshape(-1) > 1e-8, dtype=torch.bool)
    actor_adv_flat = torch.as_tensor(actor_advantages.reshape(-1), dtype=torch.float32)
    normalized_adv_flat = (
        _normalize_advantages_with_mask(actor_adv_flat, objective_flat, eps=1e-8)
        if bool(getattr(algo, "normalize_advantage", False))
        else actor_adv_flat
    )
    normalized_actor_advantages = normalized_adv_flat.reshape(objective_masks.shape).detach().cpu().numpy()
    vec_env.close()

    zero_full = reused_zero["heldout"]["full_return_per_env"]
    zero_suffix = reused_zero["heldout"]["suffix_return_per_env"]
    pre_full = reused_pre["heldout"]["full_return_per_env"]
    pre_suffix = reused_pre["heldout"]["suffix_return_per_env"]
    env_rows: list[dict[str, Any]] = []
    for env_idx in range(int(objective_masks.shape[1])):
        obj = objective_masks[:, env_idx] > 1e-8
        if int(obj.sum()) <= 0:
            continue
        obj_returns = raw_returns[:, env_idx][obj]
        obj_values = raw_values[:, env_idx][obj]
        residual = obj_returns - obj_values
        obj_raw_adv = actor_advantages[:, env_idx][obj]
        obj_norm_adv = normalized_actor_advantages[:, env_idx][obj]
        segments = _objective_episode_segments(obj, episode_starts[:, env_idx] > 1e-8)

        row = {
            "suite_name": str(suite_spec["suite_name"]),
            "env_index": int(env_idx),
            "env_seed": int(list(heldout_suite["env_seeds"])[env_idx]),
            "rollout_seed": int(list(heldout_suite["rollout_seeds"])[env_idx]),
            "objective_token_count": int(obj.sum()),
            "objective_terminal_reset_count": int(_objective_terminal_reset_count_from_segments(segments)),
            "zero_full_return": float(zero_full[env_idx]),
            "zero_suffix_return": float(zero_suffix[env_idx]),
            "pre_full_return": float(pre_full[env_idx]),
            "pre_suffix_return": float(pre_suffix[env_idx]),
            "pre_vs_zero_full_gap": float(float(pre_full[env_idx]) - float(zero_full[env_idx])),
            "pre_vs_zero_suffix_gap": float(float(pre_suffix[env_idx]) - float(zero_suffix[env_idx])),
            "raw_value_return_corr": (
                float(np.corrcoef(obj_values, obj_returns)[0, 1])
                if int(obj_values.size) > 1 and float(np.std(obj_values)) > 0.0 and float(np.std(obj_returns)) > 0.0
                else 0.0
            ),
            "raw_value_mean": float(obj_values.mean()) if int(obj_values.size) > 0 else 0.0,
            "raw_value_std": float(obj_values.std()) if int(obj_values.size) > 0 else 0.0,
            "return_to_residual_flipped_positive_count": int(np.count_nonzero((obj_returns > 0.0) & (residual <= 0.0))),
            "return_to_residual_flipped_positive_mass": float(
                np.clip(obj_returns[(obj_returns > 0.0) & (residual <= 0.0)], a_min=0.0, a_max=None).sum()
            ),
            "residual_to_actor_flipped_positive_count": int(np.count_nonzero((residual > 0.0) & (obj_raw_adv <= 0.0))),
            "residual_to_actor_flipped_positive_mass": float(
                np.clip(residual[(residual > 0.0) & (obj_raw_adv <= 0.0)], a_min=0.0, a_max=None).sum()
            ),
            "actor_to_norm_flipped_positive_count": int(np.count_nonzero((obj_raw_adv > 0.0) & (obj_norm_adv <= 0.0))),
            "actor_to_norm_flipped_positive_mass": float(
                np.clip(obj_raw_adv[(obj_raw_adv > 0.0) & (obj_norm_adv <= 0.0)], a_min=0.0, a_max=None).sum()
            ),
        }
        row.update(_stage_stats(obj_returns, prefix="raw_return"))
        row.update(_stage_stats(residual, prefix="raw_residual"))
        row.update(_stage_stats(obj_raw_adv, prefix="raw_actor_adv"))
        row.update(_stage_stats(obj_norm_adv, prefix="normalized_actor_adv"))
        env_rows.append(row)

    stage_mass_keys = [f"{prefix}_positive_mass" for prefix in STAGE_PREFIXES]
    stage_share_keys = [f"{prefix}_positive_mass_share_within_suite" for prefix in STAGE_PREFIXES]
    totals = {key: float(sum(float(row[key]) for row in env_rows)) for key in stage_mass_keys}
    pre_rank = _rank_desc(env_rows, "pre_vs_zero_suffix_gap")
    ranks_by_stage = {key: _rank_desc(env_rows, key) for key in stage_mass_keys}
    for row in env_rows:
        env_index = int(row["env_index"])
        row["pre_gap_rank_within_suite"] = int(pre_rank[env_index])
        for mass_key in stage_mass_keys:
            prefix = mass_key[: -len("_positive_mass")]
            share_key = f"{prefix}_positive_mass_share_within_suite"
            rank_key = f"{prefix}_positive_mass_rank_within_suite"
            row[share_key] = _safe_ratio(float(row[mass_key]), totals[mass_key])
            row[rank_key] = int(ranks_by_stage[mass_key][env_index])
        row["return_to_residual_positive_mass_retention"] = _safe_ratio(
            float(row["raw_residual_positive_mass"]),
            float(row["raw_return_positive_mass"]),
        )
        row["residual_to_actor_positive_mass_retention"] = _safe_ratio(
            float(row["raw_actor_adv_positive_mass"]),
            float(row["raw_residual_positive_mass"]),
        )
        row["actor_to_norm_positive_mass_retention"] = _safe_ratio(
            float(row["normalized_actor_adv_positive_mass"]),
            float(row["raw_actor_adv_positive_mass"]),
        )
        row["normalized_vs_return_positive_mass_retention"] = _safe_ratio(
            float(row["normalized_actor_adv_positive_mass"]),
            float(row["raw_return_positive_mass"]),
        )
        row["return_to_residual_flipped_positive_mass_share"] = _safe_ratio(
            float(row["return_to_residual_flipped_positive_mass"]),
            float(row["raw_return_positive_mass"]),
        )
        row["residual_to_actor_flipped_positive_mass_share"] = _safe_ratio(
            float(row["residual_to_actor_flipped_positive_mass"]),
            float(row["raw_residual_positive_mass"]),
        )
        row["actor_to_norm_flipped_positive_mass_share"] = _safe_ratio(
            float(row["actor_to_norm_flipped_positive_mass"]),
            float(row["raw_actor_adv_positive_mass"]),
        )
        row["share_drop_return_to_residual"] = float(
            row["raw_return_positive_mass_share_within_suite"] - row["raw_residual_positive_mass_share_within_suite"]
        )
        row["share_drop_residual_to_actor"] = float(
            row["raw_residual_positive_mass_share_within_suite"] - row["raw_actor_adv_positive_mass_share_within_suite"]
        )
        row["share_drop_actor_to_norm"] = float(
            row["raw_actor_adv_positive_mass_share_within_suite"]
            - row["normalized_actor_adv_positive_mass_share_within_suite"]
        )
        row["dominant_share_drop_stage"] = _dominant_label(
            {
                "baseline": max(float(row["share_drop_return_to_residual"]), 0.0),
                "actor_adv": max(float(row["share_drop_residual_to_actor"]), 0.0),
                "normalization": max(float(row["share_drop_actor_to_norm"]), 0.0),
            }
        )
        row["dominant_flip_stage"] = _dominant_label(
            {
                "baseline": float(row["return_to_residual_flipped_positive_mass_share"]),
                "actor_adv": float(row["residual_to_actor_flipped_positive_mass_share"]),
                "normalization": float(row["actor_to_norm_flipped_positive_mass_share"]),
            }
        )
        row["raw_return_sign_state"] = _sign_state(
            float(row["raw_return_positive_mass"]),
            float(row["raw_return_negative_mass"]),
        )
        row["raw_residual_sign_state"] = _sign_state(
            float(row["raw_residual_positive_mass"]),
            float(row["raw_residual_negative_mass"]),
        )
        row["raw_actor_adv_sign_state"] = _sign_state(
            float(row["raw_actor_adv_positive_mass"]),
            float(row["raw_actor_adv_negative_mass"]),
        )
        row["normalized_actor_adv_sign_state"] = _sign_state(
            float(row["normalized_actor_adv_positive_mass"]),
            float(row["normalized_actor_adv_negative_mass"]),
        )
        if str(row["raw_return_sign_state"]) != "positive":
            diagnosis = "raw_return_shape"
        elif str(row["raw_residual_sign_state"]) != "positive":
            diagnosis = "baseline_cancellation"
        elif str(row["raw_actor_adv_sign_state"]) != "positive":
            diagnosis = "actor_adv_sign_formation"
        elif str(row["normalized_actor_adv_sign_state"]) != "positive":
            diagnosis = "adv_normalization"
        elif str(row["dominant_share_drop_stage"]) != "none":
            diagnosis = f"relative_underweight_after_{row['dominant_share_drop_stage']}"
        else:
            diagnosis = "underweighted_without_sign_flip"
        row["diagnosis"] = str(diagnosis)
        row["rank_gap_pre_to_norm_adv"] = int(
            int(row["normalized_actor_adv_positive_mass_rank_within_suite"]) - int(row["pre_gap_rank_within_suite"])
        )

    suite_summary = {
        "suite_name": str(suite_spec["suite_name"]),
        "train_suite_path": str(Path(suite_spec["train_suite_path"]).expanduser().resolve()),
        "heldout_suite_path": str(Path(suite_spec["heldout_suite_path"]).expanduser().resolve()),
        "zero_json": str(Path(suite_spec["zero_json"]).expanduser().resolve()),
        "ppo_json": str(Path(suite_spec["ppo_json"]).expanduser().resolve()),
        "env_count": int(len(env_rows)),
        "objective_token_total": int(sum(int(row["objective_token_count"]) for row in env_rows)),
        "suite_runtime_wall_s": float(suite_runtime_wall_s),
        "suite_pre_minus_zero_suffix": float(reused_pre["heldout"]["suffix_return_mean"] - reused_zero["heldout"]["suffix_return_mean"]),
    }
    return env_rows, suite_summary


def _build_report(
    *,
    anchor_source: str,
    anchor_payload: dict[str, Any],
    suite_rows: dict[str, list[dict[str, Any]]],
    suite_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    missed_by_suite = {}
    source = str(anchor_source).strip().lower()
    if source == "dual_channel_residual_missed":
        anchor_rows = list(anchor_payload["residual_missed_summary"]["env_items"])
        anchor_count = int(anchor_payload["residual_missed_summary"]["count"])
        anchor_suite_counts = dict(anchor_payload["residual_missed_summary"]["suite_counts"])
        anchor_excerpt = {
            "strong_gain_anchor_count": int(anchor_payload["strong_gain_anchor_count"]),
            "baseline_reset_tail_count": int(anchor_payload["baseline_reset_tail_summary"]["count"]),
            "dual_channel_count": int(anchor_payload["dual_channel_summary"]["count"]),
            "recovered_count": int(anchor_payload["recovered_anchor_summary"]["count"]),
            "residual_missed_count": int(anchor_payload["residual_missed_summary"]["count"]),
            "residual_missed_suite_counts": dict(anchor_payload["residual_missed_summary"]["suite_counts"]),
        }
    elif source == "positive_regime_missed":
        anchor_rows = list(anchor_payload["missed_summary"]["env_items"])
        anchor_count = int(anchor_payload["missed_summary"]["count"])
        anchor_suite_counts = dict(anchor_payload["missed_summary"]["suite_counts"])
        anchor_excerpt = {
            "strong_gain_anchor_count": int(anchor_payload["strong_gain_anchor_count"]),
            "captured_count": int(anchor_payload["captured_summary"]["count"]),
            "missed_count": int(anchor_payload["missed_summary"]["count"]),
            "missed_suite_counts": dict(anchor_payload["missed_summary"]["suite_counts"]),
        }
    else:
        raise ValueError(f"Unsupported anchor_source={anchor_source!r}")
    for row in anchor_rows:
        missed_by_suite.setdefault(str(row["suite_name"]), {})[int(row["env_index"])] = dict(row)

    missed_rows: list[dict[str, Any]] = []
    for suite_name, env_rows in suite_rows.items():
        missed_keys = missed_by_suite.get(str(suite_name), {})
        for row in env_rows:
            env_index = int(row["env_index"])
            if env_index not in missed_keys:
                continue
            anchor_row = missed_keys[env_index]
            merged = dict(row)
            for key, value in anchor_row.items():
                merged[f"anchor_{key}"] = value
            missed_rows.append(merged)

    missed_rows.sort(
        key=lambda row: (
            float(row["pre_vs_zero_suffix_gap"]),
            -float(row["normalized_actor_adv_positive_mass_share_within_suite"]),
        ),
        reverse=True,
    )

    diagnosis_counts = {}
    final_state_counts = {}
    dominant_share_drop_counts = {}
    dominant_flip_counts = {}
    for row in missed_rows:
        diagnosis_counts[str(row["diagnosis"])] = int(diagnosis_counts.get(str(row["diagnosis"]), 0) + 1)
        final_state = str(row["normalized_actor_adv_sign_state"])
        final_state_counts[final_state] = int(final_state_counts.get(final_state, 0) + 1)
        dominant_share_drop_counts[str(row["dominant_share_drop_stage"])] = int(
            dominant_share_drop_counts.get(str(row["dominant_share_drop_stage"]), 0) + 1
        )
        dominant_flip_counts[str(row["dominant_flip_stage"])] = int(
            dominant_flip_counts.get(str(row["dominant_flip_stage"]), 0) + 1
        )

    aggregate = {
        "count": int(len(missed_rows)),
        "suite_counts": {
            suite_name: int(sum(1 for row in missed_rows if str(row["suite_name"]) == suite_name))
            for suite_name in sorted({str(row["suite_name"]) for row in missed_rows})
        },
        "mean_pre_gap": _mean([float(row["pre_vs_zero_suffix_gap"]) for row in missed_rows]),
        "mean_full_gap": _mean([float(row["pre_vs_zero_full_gap"]) for row in missed_rows]),
        "mean_raw_return_positive_mass_share_within_suite": _mean(
            [float(row["raw_return_positive_mass_share_within_suite"]) for row in missed_rows]
        ),
        "mean_raw_residual_positive_mass_share_within_suite": _mean(
            [float(row["raw_residual_positive_mass_share_within_suite"]) for row in missed_rows]
        ),
        "mean_raw_actor_adv_positive_mass_share_within_suite": _mean(
            [float(row["raw_actor_adv_positive_mass_share_within_suite"]) for row in missed_rows]
        ),
        "mean_normalized_actor_adv_positive_mass_share_within_suite": _mean(
            [float(row["normalized_actor_adv_positive_mass_share_within_suite"]) for row in missed_rows]
        ),
        "mean_return_to_residual_flipped_positive_mass_share": _mean(
            [float(row["return_to_residual_flipped_positive_mass_share"]) for row in missed_rows]
        ),
        "mean_residual_to_actor_flipped_positive_mass_share": _mean(
            [float(row["residual_to_actor_flipped_positive_mass_share"]) for row in missed_rows]
        ),
        "mean_actor_to_norm_flipped_positive_mass_share": _mean(
            [float(row["actor_to_norm_flipped_positive_mass_share"]) for row in missed_rows]
        ),
        "mean_rank_gap_pre_to_norm_adv": _mean([float(row["rank_gap_pre_to_norm_adv"]) for row in missed_rows]),
        "diagnosis_counts": diagnosis_counts,
        "final_normalized_actor_adv_sign_state_counts": final_state_counts,
        "dominant_share_drop_stage_counts": dominant_share_drop_counts,
        "dominant_flip_stage_counts": dominant_flip_counts,
    }

    conclusions = {
        "missed_anchor_count_matches_anchor_probe": bool(int(aggregate["count"]) == int(anchor_count)),
        "baseline_stage_often_dominates_share_drop": bool(
            int(dominant_share_drop_counts.get("baseline", 0))
            > max(
                int(dominant_share_drop_counts.get("actor_adv", 0)),
                int(dominant_share_drop_counts.get("normalization", 0)),
            )
        ),
        "normalization_is_not_primary_for_missed_anchors": bool(
            float(aggregate["mean_actor_to_norm_flipped_positive_mass_share"])
            < max(
                float(aggregate["mean_return_to_residual_flipped_positive_mass_share"]),
                float(aggregate["mean_residual_to_actor_flipped_positive_mass_share"]),
            )
        ),
        "wrong_sign_exists_among_missed_anchors": bool(int(final_state_counts.get("wrong_sign", 0)) > 0),
        "near_zero_or_balanced_exists_among_missed_anchors": bool(
            int(final_state_counts.get("near_zero_or_balanced", 0)) > 0
        ),
        "missed_anchors_remain_underweighted_in_final_actor_mass": bool(
            float(aggregate["mean_normalized_actor_adv_positive_mass_share_within_suite"])
            < float(aggregate["mean_raw_return_positive_mass_share_within_suite"])
        ),
    }

    keep_fields = [
        "suite_name",
        "env_index",
        "env_seed",
        "rollout_seed",
        "pre_vs_zero_suffix_gap",
        "pre_vs_zero_full_gap",
        "pre_gap_rank_within_suite",
        "normalized_actor_adv_positive_mass_rank_within_suite",
        "rank_gap_pre_to_norm_adv",
        "objective_token_count",
        "objective_terminal_reset_count",
        "raw_value_return_corr",
        "raw_return_sign_state",
        "raw_residual_sign_state",
        "raw_actor_adv_sign_state",
        "normalized_actor_adv_sign_state",
        "raw_return_positive_mass_share_within_suite",
        "raw_residual_positive_mass_share_within_suite",
        "raw_actor_adv_positive_mass_share_within_suite",
        "normalized_actor_adv_positive_mass_share_within_suite",
        "return_to_residual_flipped_positive_mass_share",
        "residual_to_actor_flipped_positive_mass_share",
        "actor_to_norm_flipped_positive_mass_share",
        "dominant_share_drop_stage",
        "dominant_flip_stage",
        "diagnosis",
        "anchor_dist_mass_share_within_suite",
        "anchor_dist_mass_rank_within_suite",
    ]

    return {
        "anchor_probe_excerpt": anchor_excerpt,
        "suite_summaries": suite_summaries,
        "aggregate": aggregate,
        "missed_anchor_rows": [{field: row.get(field) for field in keep_fields} for row in missed_rows],
        "conclusions": conclusions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Probe why residual missed gain anchors collapse to underweighted or wrong-sign actor objective mass."
    )
    parser.add_argument("checkpoint_path", type=str)
    parser.add_argument("--suite-specs-json", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--n-samples", type=int, default=2048)
    parser.add_argument("--single-eval-pos", type=int, default=1946)
    parser.add_argument(
        "--train-profile",
        type=str,
        default="trusted_sep_reset_mainline",
        choices=["trusted_sep_reset_mainline", "legacy_actor_only_probe"],
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--n-epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--target-kl", type=float, default=None)
    parser.add_argument(
        "--anchor-source",
        type=str,
        default="dual_channel_residual_missed",
        choices=["dual_channel_residual_missed", "positive_regime_missed"],
    )
    parser.add_argument("--anchor-min-pre-gap", type=float, default=1.0)
    parser.add_argument("--captured-min-mass-share-within-suite", type=float, default=0.05)
    parser.add_argument("--phase2-summary-path", type=str, default=DEFAULT_PHASE2_SUMMARY)
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output_json).expanduser().resolve()
    if output_path.exists() and not bool(args.overwrite):
        print(output_path.read_text())
        return 0

    t0 = time.perf_counter()
    phase2_summary = assert_phase2_green(args.phase2_summary_path)
    del phase2_summary

    suite_specs = DEFAULT_SUITE_SPECS
    if args.suite_specs_json is not None:
        suite_specs = json.loads(Path(args.suite_specs_json).expanduser().resolve().read_text())

    missed_anchor_keys, anchor_payload = _missed_anchor_keys(
        suite_specs=suite_specs,
        anchor_source=str(args.anchor_source),
        anchor_min_pre_gap=float(args.anchor_min_pre_gap),
        captured_min_mass_share_within_suite=float(args.captured_min_mass_share_within_suite),
    )
    target_suite_specs = [spec for spec in suite_specs if missed_anchor_keys.get(str(spec["suite_name"]))]

    checkpoint_path = str(Path(args.checkpoint_path).expanduser().resolve())
    device_obj = torch.device(str(args.device or _default_device()))
    suite_rows = {}
    suite_summaries = []
    for spec in target_suite_specs:
        rows, suite_summary = _collect_suite_rows(
            checkpoint_path=checkpoint_path,
            suite_spec=spec,
            device_obj=device_obj,
            n_samples=int(args.n_samples),
            single_eval_pos=int(args.single_eval_pos),
            train_profile=str(args.train_profile),
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            learning_rate=args.learning_rate,
            target_kl=args.target_kl,
        )
        suite_rows[str(spec["suite_name"])] = rows
        suite_summaries.append(suite_summary)

    report = _build_report(
        anchor_source=str(args.anchor_source),
        anchor_payload=anchor_payload,
        suite_rows=suite_rows,
        suite_summaries=suite_summaries,
    )
    result = {
        "audit_entry": "phase3_residual_missed_anchor_sign_probe",
        "config": {
            "checkpoint_path": checkpoint_path,
            "n_samples": int(args.n_samples),
            "single_eval_pos": int(args.single_eval_pos),
            "train_profile": str(args.train_profile),
            "batch_size": None if args.batch_size is None else int(args.batch_size),
            "n_epochs": None if args.n_epochs is None else int(args.n_epochs),
            "learning_rate": None if args.learning_rate is None else float(args.learning_rate),
            "target_kl": None if args.target_kl is None else float(args.target_kl),
            "anchor_source": str(args.anchor_source),
            "anchor_min_pre_gap": float(args.anchor_min_pre_gap),
            "captured_min_mass_share_within_suite": float(args.captured_min_mass_share_within_suite),
            "suite_names": [str(spec["suite_name"]) for spec in target_suite_specs],
            "phase2_summary_path": str(Path(args.phase2_summary_path).expanduser().resolve()),
        },
        "runtime_wall_s": float(time.perf_counter() - t0),
    }
    result.update(report)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(output_path.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
