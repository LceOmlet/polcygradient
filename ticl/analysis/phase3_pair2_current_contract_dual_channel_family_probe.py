import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_CHECKPOINT_PATH = "/home/chen/RLPFN/rwkv/models_diff/rlpfn_03_30_2026_22_37_53_epoch_13.cpkt"
DEFAULT_TRAIN_SUITE_PATH = "/home/chen/RLPFN/artifacts/phase3_cross_env_baseline_pair2/suites/train_suite.pt"
DEFAULT_ANTITRANSFER_JSON = (
    "/home/chen/RLPFN/artifacts/phase3_pair2_high_mass_family_antitransfer_probe.json"
)
DEFAULT_OUTPUT_JSON = (
    "/home/chen/RLPFN/artifacts/phase3_pair2_current_contract_dual_channel_family_probe.json"
)
DEFAULT_PHASE2_SUMMARY = (
    "/home/chen/RLPFN/artifacts/phase2_regression_pack/phase2_regression_suite_summary.json"
)
DEFAULT_TRAIN_PROFILE = "phase2_shared_backbone_contract"
DEFAULT_N_SAMPLES = 256
DEFAULT_SINGLE_EVAL_POS = 64
TAIL_LEN = 8


def _load_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def _collect_one_rollout(algo, callback, vec_env) -> None:
    from stable_baselines3.common.logger import configure as configure_logger

    if getattr(algo, "_last_obs", None) is None:
        algo.ep_info_buffer = []
        algo.ep_success_buffer = []
        if not hasattr(algo, "_logger"):
            algo.set_logger(configure_logger(folder=None, format_strings=[]))
        algo._last_obs = vec_env.reset()
        algo._last_episode_starts = np.ones((vec_env.num_envs,), dtype=bool)
        algo._last_lstm_states = algo.policy._dummy_states(vec_env.num_envs)
        callback.init_callback(algo)
    algo._update_current_progress_remaining(0, int(algo.n_steps))
    assert algo.collect_rollouts(vec_env, callback, algo.rollout_buffer, n_rollout_steps=algo.n_steps)


def _share(numerator: float, denominator: float) -> float:
    denominator = float(denominator)
    if abs(denominator) <= 1e-12:
        return 0.0
    return float(float(numerator) / denominator)


def _candidate_metrics(
    *,
    name: str,
    description: str,
    mask: np.ndarray,
    objective_mask: np.ndarray,
    env_indices: np.ndarray,
    objective_episode_indices: np.ndarray,
    raw_actor_advantages: np.ndarray,
) -> dict[str, Any]:
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    objective_mask = np.asarray(objective_mask, dtype=bool).reshape(-1)
    env_indices = np.asarray(env_indices, dtype=np.int64).reshape(-1)
    objective_episode_indices = np.asarray(objective_episode_indices, dtype=np.int64).reshape(-1)
    raw_actor_advantages = np.asarray(raw_actor_advantages, dtype=np.float64).reshape(-1)
    if (
        mask.shape != objective_mask.shape
        or mask.shape != raw_actor_advantages.shape
        or mask.shape != env_indices.shape
        or mask.shape != objective_episode_indices.shape
    ):
        raise ValueError("Candidate metrics expect mask/objective/adv arrays with identical shapes.")
    token_count = int(mask.sum())
    raw_negative_mass = (
        float(np.abs(raw_actor_advantages[mask & (raw_actor_advantages < 0.0)]).sum()) if token_count > 0 else 0.0
    )
    raw_positive_mass = (
        float(np.clip(raw_actor_advantages[mask], a_min=0.0, a_max=None).sum()) if token_count > 0 else 0.0
    )
    raw_abs_mass = float(np.abs(raw_actor_advantages[mask]).sum()) if token_count > 0 else 0.0
    batch_negative_mass = float(np.abs(raw_actor_advantages[objective_mask & (raw_actor_advantages < 0.0)]).sum())
    batch_positive_mass = float(np.clip(raw_actor_advantages[objective_mask], a_min=0.0, a_max=None).sum())
    return {
        "name": str(name),
        "description": str(description),
        "token_count": int(token_count),
        "env_indices": sorted({int(v) for v in np.unique(env_indices[mask]).tolist()}),
        "episode_indices": sorted({int(v) for v in np.unique(objective_episode_indices[mask]).tolist()}),
        "raw_negative_mass": float(raw_negative_mass),
        "raw_positive_mass": float(raw_positive_mass),
        "raw_abs_mass": float(raw_abs_mass),
        "raw_negative_share_of_batch": float(_share(raw_negative_mass, batch_negative_mass)),
        "raw_positive_share_of_batch": float(_share(raw_positive_mass, batch_positive_mass)),
    }


def _summarize_recovered_anchor_first_objective_non_tail_outer_batches(
    *,
    objective_mask: np.ndarray,
    env_indices: np.ndarray,
    objective_episode_indices: np.ndarray,
    objective_episode_positions: np.ndarray,
    raw_actor_advantages: np.ndarray,
    target_env_position_end_by_env: dict[int, int],
    batch_size: int,
) -> dict[str, Any]:
    objective_mask = np.asarray(objective_mask, dtype=bool).reshape(-1)
    env_indices = np.asarray(env_indices, dtype=np.int64).reshape(-1)
    objective_episode_indices = np.asarray(objective_episode_indices, dtype=np.int64).reshape(-1)
    objective_episode_positions = np.asarray(objective_episode_positions, dtype=np.int64).reshape(-1)
    raw_actor_advantages = np.asarray(raw_actor_advantages, dtype=np.float64).reshape(-1)
    if not (
        objective_mask.shape
        == env_indices.shape
        == objective_episode_indices.shape
        == objective_episode_positions.shape
        == raw_actor_advantages.shape
    ):
        raise ValueError("Outer-batch family summary expects equal-length flat arrays.")
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    family_mask = np.zeros_like(objective_mask, dtype=bool)
    for env_id, position_end in sorted(target_env_position_end_by_env.items()):
        family_mask |= (
            objective_mask
            & (env_indices == int(env_id))
            & (objective_episode_indices == 0)
            & (objective_episode_positions <= int(position_end))
        )

    total_token_count = int(family_mask.sum())
    total_negative_mass = float(np.abs(raw_actor_advantages[family_mask & (raw_actor_advantages < 0.0)]).sum())
    rows: list[dict[str, Any]] = []
    for batch_idx, start in enumerate(range(0, int(objective_mask.shape[0]), batch_size), start=1):
        end = int(min(start + batch_size, int(objective_mask.shape[0])))
        batch_mask = family_mask[start:end]
        if not bool(np.any(batch_mask)):
            continue
        batch_env = env_indices[start:end]
        batch_pos = objective_episode_positions[start:end]
        batch_adv = raw_actor_advantages[start:end]
        env_rows = []
        for env_id in sorted({int(v) for v in np.unique(batch_env[batch_mask]).tolist()}):
            local_mask = batch_mask & (batch_env == int(env_id))
            env_rows.append(
                {
                    "env_index": int(env_id),
                    "token_count": int(local_mask.sum()),
                    "objective_position_min": int(batch_pos[local_mask].min()),
                    "objective_position_max": int(batch_pos[local_mask].max()),
                }
            )
        negative_mass = float(np.abs(batch_adv[batch_mask & (batch_adv < 0.0)]).sum())
        rows.append(
            {
                "outer_batch_idx": int(batch_idx),
                "token_count": int(batch_mask.sum()),
                "negative_mass": float(negative_mass),
                "abs_mass": float(np.abs(batch_adv[batch_mask]).sum()),
                "env_rows": env_rows,
                "token_share_of_family": float(_share(int(batch_mask.sum()), total_token_count)),
                "negative_share_of_family": float(_share(negative_mass, total_negative_mass)),
            }
        )

    if not rows:
        return {
            "target_env_position_end_by_env": {
                str(int(k)): int(v) for k, v in sorted(target_env_position_end_by_env.items())
            },
            "rows": [],
            "dominant_row_by_negative_mass": None,
            "dominant_row_by_token_count": None,
            "recommended_snapshot_selector": None,
            "single_outer_batch_covers_full_family": False,
            "family_spans_multiple_outer_batches": False,
            "family_is_split_across_distinct_env_local_outer_batches": False,
        }

    dominant_row_by_negative_mass = max(rows, key=lambda row: (float(row["negative_mass"]), int(row["token_count"])))
    dominant_row_by_token_count = max(rows, key=lambda row: (int(row["token_count"]), float(row["negative_mass"])))
    dominant_env_rows = list(dominant_row_by_negative_mass["env_rows"])
    dominant_env = dict(dominant_env_rows[0]) if len(dominant_env_rows) == 1 else None
    recommended_snapshot_selector = (
        None
        if dominant_env is None
        else {
            "target_outer_batch_idx": int(dominant_row_by_negative_mass["outer_batch_idx"]),
            "target_env_index": int(dominant_env["env_index"]),
            "target_objective_episode_index": 0,
            "target_objective_position_start": int(dominant_env["objective_position_min"]),
            "target_objective_position_end": int(dominant_env["objective_position_max"]),
        }
    )
    distinct_env_sets = {
        tuple(int(env_row["env_index"]) for env_row in row["env_rows"])
        for row in rows
    }
    return {
        "target_env_position_end_by_env": {
            str(int(k)): int(v) for k, v in sorted(target_env_position_end_by_env.items())
        },
        "rows": rows,
        "dominant_row_by_negative_mass": dict(dominant_row_by_negative_mass),
        "dominant_row_by_token_count": dict(dominant_row_by_token_count),
        "recommended_snapshot_selector": recommended_snapshot_selector,
        "single_outer_batch_covers_full_family": bool(
            any(int(row["token_count"]) == int(total_token_count) for row in rows)
        ),
        "family_spans_multiple_outer_batches": bool(len(rows) > 1),
        "family_is_split_across_distinct_env_local_outer_batches": bool(
            len(rows) > 1 and len(distinct_env_sets) == len(rows)
        ),
    }


def _build_env_episode_rows(
    *,
    env_indices: np.ndarray,
    objective_episode_indices: np.ndarray,
    objective_episode_positions: np.ndarray,
    objective_mask: np.ndarray,
    raw_actor_advantages: np.ndarray,
    env_ids: list[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for env_id in env_ids:
        env_mask = objective_mask & (env_indices == int(env_id))
        present_episodes = sorted(int(v) for v in np.unique(objective_episode_indices[env_mask]).tolist())
        first_episode_index = None if not present_episodes else int(present_episodes[0])
        first_episode_non_tail_token_count = 0
        first_episode_all_tail = False
        spans: list[dict[str, Any]] = []
        for episode_idx in present_episodes:
            episode_mask = env_mask & (objective_episode_indices == int(episode_idx))
            positions = objective_episode_positions[episode_mask]
            max_position = int(positions.max()) if positions.size > 0 else -1
            non_tail_mask = positions <= int(max_position - TAIL_LEN)
            non_tail_token_count = int(non_tail_mask.sum()) if positions.size > 0 else 0
            raw_neg = float(np.abs(raw_actor_advantages[episode_mask & (raw_actor_advantages < 0.0)]).sum())
            raw_pos = float(np.clip(raw_actor_advantages[episode_mask], a_min=0.0, a_max=None).sum())
            spans.append(
                {
                    "objective_episode_index": int(episode_idx),
                    "token_count": int(episode_mask.sum()),
                    "objective_position_min": int(positions.min()) if positions.size > 0 else -1,
                    "objective_position_max": max_position,
                    "non_tail_token_count_under_tail_len_8": int(non_tail_token_count),
                    "raw_negative_mass": float(raw_neg),
                    "raw_positive_mass": float(raw_pos),
                }
            )
            if episode_idx == first_episode_index:
                first_episode_non_tail_token_count = int(non_tail_token_count)
                first_episode_all_tail = bool(int(episode_mask.sum()) > 0 and non_tail_token_count == 0)
        rows.append(
            {
                "env_index": int(env_id),
                "objective_token_count": int(env_mask.sum()),
                "objective_episode_count": int(len(present_episodes)),
                "first_objective_episode_index": first_episode_index,
                "first_objective_episode_non_tail_token_count_under_tail_len_8": int(
                    first_episode_non_tail_token_count
                ),
                "first_objective_episode_all_tail_under_tail_len_8": bool(first_episode_all_tail),
                "objective_episode_spans": spans,
            }
        )
    return rows


def build_pair2_current_contract_dual_channel_family_probe(
    *,
    checkpoint_path: str,
    train_suite_path: str,
    antitransfer_json: str,
    phase2_summary_path: str,
    device: str | None,
    n_samples: int,
    single_eval_pos: int,
    train_profile: str,
) -> dict[str, Any]:
    import torch

    from ticl.analysis.critic_free_single_env_audit import _make_deterministic_batch_plan
    from ticl.analysis.phase2_guardrail import assert_phase2_green
    from ticl.analysis.phase3_multi_env_optimization_audit import (
        _bind_vec_env_to_fixed_suite,
        _default_device,
        _load_required_suite,
        _resolve_train_profile,
    )
    from ticl.analysis.phase3_train_rollout_quality_probe import _train_rollout_quality_collect_contract_kwargs
    from ticl.analysis.prior_generalization_audit import _resolve_audit_env_config
    from ticl.model_builder import load_model
    from ticl.priors.environment_prior import EnvironmentPrior
    from ticl.sb3_recurrent_ppo import build_recurrent_ppo

    t0 = time.perf_counter()
    phase2_summary = assert_phase2_green(phase2_summary_path)
    anti = _load_json(antitransfer_json)
    if str(anti.get("probe_entry", "")) != "phase3_pair2_high_mass_family_antitransfer_probe":
        raise ValueError("Expected canonical pair2 anti-transfer probe artifact.")

    recovered_envs = [int(v) for v in anti["pair2_anchor_structure"]["recovered_dual_channel_env_indices"]]
    residual_envs = [int(v) for v in anti["pair2_anchor_structure"]["residual_missed_env_indices"]]
    high_mass_episodes = [
        int(v) for v in anti["current_env12_family_class"]["current_high_mass_candidate"]["locked_objective_episode_indices"]
    ]
    locked_outer_batch_local_high_mass_token_count = int(
        anti["current_env12_family_class"]["current_high_mass_candidate"]["token_count"]
    )

    checkpoint_path = str(Path(checkpoint_path).expanduser().resolve())
    device_obj = torch.device(str(device or _default_device()))
    train_suite = _load_required_suite(train_suite_path)

    load_model.cache_clear()
    model, config = load_model(checkpoint_path, device=device_obj, verbose=False)
    env_cfg = _resolve_audit_env_config(config, core_a=False, reference_semantics_enabled=False)
    optimizer_cfg = dict(config.get("optimizer", {}))
    profile_cfg = _resolve_train_profile(
        optimizer_cfg=optimizer_cfg,
        train_profile=str(train_profile),
        batch_size=None,
        n_epochs=None,
        learning_rate=None,
        target_kl=None,
    )

    prior = EnvironmentPrior(copy.deepcopy(env_cfg))
    prior._sample_batch_hypers = lambda batch_n: [copy.deepcopy(h) for h in list(train_suite["h_list"])]
    prior._sample_single_eval_pos = lambda n_samples_arg, rng_seed=None: int(single_eval_pos)

    algo, callback, vec_env = build_recurrent_ppo(
        model=model,
        env_prior=prior,
        device=str(device_obj),
        num_features=int(config["prior"]["num_features"]),
        n_envs=int(train_suite["batch_size"]),
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
        **_train_rollout_quality_collect_contract_kwargs(train_suite),
        actor_objective_runtime_current_suite_name=None,
        reset_env_state_at_sep=bool(profile_cfg["reset_env_state_at_sep"]),
        separate_value_backbone=bool(profile_cfg["separate_value_backbone"]),
        restore_validation_policy_head_state=True,
        runtime_normalized_q_value_weight_override=profile_cfg["runtime_normalized_q_value_weight_override"],
        runtime_next_state_flow_matching_weight_override=profile_cfg["runtime_next_state_flow_matching_weight_override"],
        ent_coef=float(optimizer_cfg.get("ppo_ent_coef", 0.0)),
        vf_coef=float(profile_cfg["vf_coef"]),
        max_grad_norm=float(optimizer_cfg.get("ppo_max_grad_norm", 0.5)),
        target_kl=profile_cfg["target_kl"],
        verbose=0,
    )
    _bind_vec_env_to_fixed_suite(vec_env, suite=train_suite, single_eval_pos=int(single_eval_pos))
    _make_deterministic_batch_plan(algo.rollout_buffer)
    _collect_one_rollout(algo, callback, vec_env)

    rollout_buffer = algo.rollout_buffer
    rollout_buffer._ensure_generator_ready()

    objective_mask = np.asarray(rollout_buffer.objective_masks, dtype=np.float32).reshape(-1) > 1e-8
    env_indices = np.asarray(rollout_buffer._flat_env_indices, dtype=np.int64).reshape(-1)
    objective_episode_indices = np.asarray(
        rollout_buffer._flat_objective_episode_indices,
        dtype=np.int64,
    ).reshape(-1)
    objective_episode_positions = np.asarray(
        rollout_buffer._flat_objective_episode_positions,
        dtype=np.int64,
    ).reshape(-1)
    raw_actor_advantages = np.asarray(rollout_buffer.actor_advantages, dtype=np.float32).reshape(-1)

    if not bool(np.any(objective_mask)):
        raise ValueError("Expected objective-valid tokens in the current guarded contract rollout.")

    env_rows = _build_env_episode_rows(
        env_indices=env_indices,
        objective_episode_indices=objective_episode_indices,
        objective_episode_positions=objective_episode_positions,
        objective_mask=objective_mask,
        raw_actor_advantages=raw_actor_advantages,
        env_ids=sorted({12, *recovered_envs, *residual_envs}),
    )
    env_row_by_id = {int(row["env_index"]): row for row in env_rows}

    recovered_non_tail_mask = np.zeros_like(objective_mask, dtype=bool)
    recovered_complete_mask = np.zeros_like(objective_mask, dtype=bool)
    recovered_non_tail_position_end_by_env: dict[int, int] = {}
    for env_id in recovered_envs:
        row = env_row_by_id.get(int(env_id))
        if row is None or row["first_objective_episode_index"] is None:
            continue
        first_epi = int(row["first_objective_episode_index"])
        local_mask = objective_mask & (env_indices == int(env_id)) & (objective_episode_indices == first_epi)
        recovered_complete_mask |= local_mask
        positions = objective_episode_positions[local_mask]
        if positions.size <= 0:
            continue
        max_pos = int(positions.max())
        recovered_non_tail_position_end_by_env[int(env_id)] = int(max_pos - TAIL_LEN)
        recovered_non_tail_mask |= local_mask & (objective_episode_positions <= int(max_pos - TAIL_LEN))

    env12_high_mass_mask = objective_mask & (env_indices == 12) & np.isin(
        objective_episode_indices,
        np.asarray(high_mass_episodes, dtype=np.int64),
    )
    fused_non_tail_mask = env12_high_mass_mask | recovered_non_tail_mask
    fused_complete_mask = env12_high_mass_mask | recovered_complete_mask

    candidates = {
        "env12_high_mass_family": _candidate_metrics(
            name="env12_high_mass_family",
            description="Current train-positive / heldout-negative env12-local reset-channel family.",
            mask=env12_high_mass_mask,
            objective_mask=objective_mask,
            env_indices=env_indices,
            objective_episode_indices=objective_episode_indices,
            raw_actor_advantages=raw_actor_advantages,
        ),
        "recovered_anchor_first_objective_non_tail_family": _candidate_metrics(
            name="recovered_anchor_first_objective_non_tail_family",
            description="Recovered pair2 non-reset anchors, restricted to current-contract first-objective non-tail tokens.",
            mask=recovered_non_tail_mask,
            objective_mask=objective_mask,
            env_indices=env_indices,
            objective_episode_indices=objective_episode_indices,
            raw_actor_advantages=raw_actor_advantages,
        ),
        "recovered_anchor_first_objective_complete_family": _candidate_metrics(
            name="recovered_anchor_first_objective_complete_family",
            description="Recovered pair2 non-reset anchors, using the complete current-contract first objective episode.",
            mask=recovered_complete_mask,
            objective_mask=objective_mask,
            env_indices=env_indices,
            objective_episode_indices=objective_episode_indices,
            raw_actor_advantages=raw_actor_advantages,
        ),
        "dual_channel_fused_non_tail_family": _candidate_metrics(
            name="dual_channel_fused_non_tail_family",
            description="Env12 high-mass reset channel plus recovered-anchor first-objective non-tail coverage.",
            mask=fused_non_tail_mask,
            objective_mask=objective_mask,
            env_indices=env_indices,
            objective_episode_indices=objective_episode_indices,
            raw_actor_advantages=raw_actor_advantages,
        ),
        "dual_channel_fused_complete_family": _candidate_metrics(
            name="dual_channel_fused_complete_family",
            description="Env12 high-mass reset channel plus recovered-anchor complete first objective episode coverage.",
            mask=fused_complete_mask,
            objective_mask=objective_mask,
            env_indices=env_indices,
            objective_episode_indices=objective_episode_indices,
            raw_actor_advantages=raw_actor_advantages,
        ),
    }

    env12_high_mass_outer_batch_local_not_portable = bool(
        candidates["env12_high_mass_family"]["token_count"] <= 1
        and locked_outer_batch_local_high_mass_token_count >= 16
    )
    recommended_name = (
        "recovered_anchor_first_objective_non_tail_family"
        if env12_high_mass_outer_batch_local_not_portable
        else (
            "dual_channel_fused_non_tail_family"
            if candidates["recovered_anchor_first_objective_non_tail_family"]["token_count"] > 0
            else "dual_channel_fused_complete_family"
        )
    )
    recommended = dict(candidates[recommended_name])
    recovered_non_tail_outer_batch_distribution = _summarize_recovered_anchor_first_objective_non_tail_outer_batches(
        objective_mask=objective_mask,
        env_indices=env_indices,
        objective_episode_indices=objective_episode_indices,
        objective_episode_positions=objective_episode_positions,
        raw_actor_advantages=raw_actor_advantages,
        target_env_position_end_by_env={
            int(env_id): int(position_end)
            for env_id, position_end in recovered_non_tail_position_end_by_env.items()
            if int(position_end) >= 0
        },
        batch_size=int(profile_cfg["batch_size"]),
    )

    return {
        "probe_entry": "phase3_pair2_current_contract_dual_channel_family_probe",
        "phase2_preflight": {
            "required": True,
            "summary_path": str(Path(phase2_summary_path).expanduser().resolve()),
            "phase2_all_checks_pass": bool(phase2_summary["validation"]["all_checks_pass"]),
            "phase2_pass_flags": dict(phase2_summary["pass_flags"]),
        },
        "inputs": {
            "checkpoint_path": checkpoint_path,
            "train_suite_path": str(Path(train_suite_path).expanduser().resolve()),
            "antitransfer_json": str(Path(antitransfer_json).expanduser().resolve()),
            "n_samples": int(n_samples),
            "single_eval_pos": int(single_eval_pos),
            "train_profile": str(profile_cfg["train_profile"]),
            "tail_len": int(TAIL_LEN),
            "train_env_rng_seeds": [int(v) for v in train_suite["env_seeds"]],
            "train_rollout_rng_seeds": [int(v) for v in train_suite["rollout_seeds"]],
        },
        "pair2_anchor_sets": {
            "recovered_dual_channel_env_indices": recovered_envs,
            "residual_missed_env_indices": residual_envs,
            "current_env12_high_mass_episode_indices": high_mass_episodes,
            "locked_outer_batch_local_high_mass_token_count": int(locked_outer_batch_local_high_mass_token_count),
        },
        "current_contract_env_rows": env_rows,
        "candidate_families": candidates,
        "recovered_anchor_first_objective_non_tail_outer_batch_distribution": recovered_non_tail_outer_batch_distribution,
        "comparisons": {
            "recovered_non_tail_minus_high_mass_raw_negative_share_of_batch": float(
                candidates["recovered_anchor_first_objective_non_tail_family"]["raw_negative_share_of_batch"]
                - candidates["env12_high_mass_family"]["raw_negative_share_of_batch"]
            ),
            "fused_non_tail_minus_high_mass_raw_negative_share_of_batch": float(
                candidates["dual_channel_fused_non_tail_family"]["raw_negative_share_of_batch"]
                - candidates["env12_high_mass_family"]["raw_negative_share_of_batch"]
            ),
            "fused_complete_minus_high_mass_raw_negative_share_of_batch": float(
                candidates["dual_channel_fused_complete_family"]["raw_negative_share_of_batch"]
                - candidates["env12_high_mass_family"]["raw_negative_share_of_batch"]
            ),
        },
        "conclusions": {
            "snapshot_only_family_search_was_structurally_env12_local": True,
            "current_contract_recovered_anchor_non_tail_support_exists": bool(
                candidates["recovered_anchor_first_objective_non_tail_family"]["token_count"] > 0
            ),
            "current_contract_recovered_anchor_complete_support_exists": bool(
                candidates["recovered_anchor_first_objective_complete_family"]["token_count"] > 0
            ),
            "env12_high_mass_episode_ids_are_not_env_wide_portable_in_current_contract": bool(
                env12_high_mass_outer_batch_local_not_portable
            ),
            "hybrid_dual_channel_family_requires_runtime_local_plus_env_wide_selector": bool(
                env12_high_mass_outer_batch_local_not_portable
                and candidates["recovered_anchor_first_objective_non_tail_family"]["token_count"] > 0
            ),
            "different_branch_specific_family_is_available_without_shared_core_edit": bool(
                recommended["token_count"] > candidates["env12_high_mass_family"]["token_count"]
                and len(recommended["env_indices"]) > len(candidates["env12_high_mass_family"]["env_indices"])
            ),
            "recommended_family_leaves_env12_local_reset_channel_class": bool(12 not in set(recovered_envs)),
            "recovered_anchor_non_tail_family_spans_multiple_outer_batches": bool(
                recovered_non_tail_outer_batch_distribution["family_spans_multiple_outer_batches"]
            ),
            "single_outer_batch_snapshot_is_insufficient_for_full_recovered_anchor_family": bool(
                not recovered_non_tail_outer_batch_distribution["single_outer_batch_covers_full_family"]
            ),
        },
        "recommendation": {
            "recommended_next_family": recommended_name,
            "candidate": recommended,
            "recommended_snapshot_selector": recovered_non_tail_outer_batch_distribution["recommended_snapshot_selector"],
            "reason": (
                "The current train-positive family is env12-local. Under the same strict 256/64 contract, "
                "the next admissible family should add recovered non-reset anchor coverage rather than retune "
                "episode indices inside env12 only. When the locked env12 high-mass episode ids do not port into "
                "env-wide current-contract coordinates, prefer the recovered-anchor non-tail family as the clean "
                "next branch-specific family and treat any fused dual-channel family as a later hybrid runtime task."
            ),
        },
        "runtime_wall_s": float(time.perf_counter() - t0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Current-contract pair2 probe for a dual-channel branch-specific family candidate."
    )
    parser.add_argument("--checkpoint-path", type=str, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--train-suite-path", type=str, default=DEFAULT_TRAIN_SUITE_PATH)
    parser.add_argument("--antitransfer-json", type=str, default=DEFAULT_ANTITRANSFER_JSON)
    parser.add_argument("--phase2-summary-path", type=str, default=DEFAULT_PHASE2_SUMMARY)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES)
    parser.add_argument("--single-eval-pos", type=int, default=DEFAULT_SINGLE_EVAL_POS)
    parser.add_argument("--train-profile", type=str, default=DEFAULT_TRAIN_PROFILE)
    parser.add_argument("--output-json", type=str, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output_json).expanduser().resolve()
    if output_path.exists() and not bool(args.overwrite):
        print(output_path.read_text(encoding="utf-8"))
        return 0

    report = build_pair2_current_contract_dual_channel_family_probe(
        checkpoint_path=args.checkpoint_path,
        train_suite_path=args.train_suite_path,
        antitransfer_json=args.antitransfer_json,
        phase2_summary_path=args.phase2_summary_path,
        device=args.device,
        n_samples=int(args.n_samples),
        single_eval_pos=int(args.single_eval_pos),
        train_profile=str(args.train_profile),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
