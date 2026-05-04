import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ticl.analysis.phase2_guardrail import DEFAULT_PHASE2_SUMMARY, assert_phase2_green
from ticl.analysis.phase3_multi_env_optimization_audit import (
    _bind_vec_env_to_fixed_suite,
    _default_device,
    _load_required_suite,
    _resolve_train_profile,
)
from ticl.analysis.phase3_pair2_current_contract_dual_channel_family_probe import (
    DEFAULT_ANTITRANSFER_JSON,
    _build_env_episode_rows,
    _candidate_metrics,
    _collect_one_rollout,
    _load_json,
)
from ticl.analysis.phase3_pair2_recovered_anchor_first_objective_non_tail_family_heldout_transfer_compare_pack import (
    PAIR2_RECOVERED_ANCHOR_NON_TAIL_LOCKED_TRAIN_COMPARE_JSON,
)
from ticl.analysis.phase3_pair2_recovered_anchor_first_objective_non_tail_family_train_update_ab_compare_pack import (
    PAIR2_RECOVERED_ANCHOR_NON_TAIL_BASELINE_POST_POLICY_BUNDLE_PATH,
    PAIR2_RECOVERED_ANCHOR_NON_TAIL_BRANCH_SPECIFIC_MODE,
    PAIR2_RECOVERED_ANCHOR_NON_TAIL_OVERRIDE_POST_POLICY_BUNDLE_PATH,
)
from ticl.analysis.phase3_pair2_train_update_ab_compare_pack import (
    PAIR2_CHECKPOINT_PATH,
    PAIR2_HELDOUT_SUITE_PATH,
    PAIR2_TRAIN_SUITE_PATH,
    PAIR2_TRAIN_UPDATE_NSAMPLES,
    PAIR2_TRAIN_UPDATE_SINGLE_EVAL_POS,
)
from ticl.analysis.phase3_train_rollout_quality_probe import _train_rollout_quality_collect_contract_kwargs
from ticl.analysis.prior_generalization_audit import _resolve_audit_env_config
from ticl.model_builder import load_model
from ticl.priors.environment_prior import EnvironmentPrior
from ticl.sb3_recurrent_ppo import (
    PAIR2_RECOVERED_ANCHOR_FIRST_OBJECTIVE_NON_TAIL_FAMILY_OBJECTIVE_EPISODE_INDEX,
    PAIR2_RECOVERED_ANCHOR_FIRST_OBJECTIVE_NON_TAIL_FAMILY_TARGET_ENV_INDICES,
    PAIR2_RECOVERED_ANCHOR_FIRST_OBJECTIVE_NON_TAIL_FAMILY_TAIL_LEN,
    build_recurrent_ppo,
)


DEFAULT_LOCKED_HELDOUT_TRANSFER_COMPARE_JSON = (
    "/home/chen/RLPFN/artifacts/phase3_pair2_recovered_anchor_first_objective_non_tail_family_heldout_transfer_compare_pack.json"
)
DEFAULT_LOCKED_TRAIN_CURRENT_CONTRACT_PROBE_JSON = (
    "/home/chen/RLPFN/artifacts/phase3_pair2_current_contract_dual_channel_family_probe.json"
)
DEFAULT_OUTPUT_JSON = (
    "/home/chen/RLPFN/artifacts/phase3_pair2_recovered_anchor_first_objective_non_tail_antitransfer_decomposition_probe.json"
)
POSITION_BUCKET_NAMES = ("q1", "q2", "q3", "q4")


def _persist_json(path: str, payload: dict[str, Any]) -> str:
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return str(output_path)


def _runtime_target_env_indices() -> list[int]:
    return [int(v) for v in PAIR2_RECOVERED_ANCHOR_FIRST_OBJECTIVE_NON_TAIL_FAMILY_TARGET_ENV_INDICES]


def _load_locked_heldout_transfer_compare(path: str) -> dict[str, Any]:
    payload = _load_json(path)
    if (
        str(payload.get("audit_entry", ""))
        != "phase3_pair2_recovered_anchor_first_objective_non_tail_family_heldout_transfer_compare_pack"
    ):
        raise RuntimeError(f"Expected locked recovered-anchor non-tail heldout-transfer compare artifact at {path}")
    conclusions = dict(payload.get("conclusions", {}))
    if not bool(conclusions.get("train_side_reproduction_matches_locked_compare", False)):
        raise RuntimeError(f"Locked heldout-transfer compare failed train-side reproduction guard: {path}")
    if not bool(conclusions.get("heldout_transfer_material", False)):
        raise RuntimeError(f"Locked heldout-transfer compare is not material: {path}")
    if bool(conclusions.get("heldout_transfer_is_only_numeric_noise", False)):
        raise RuntimeError(f"Locked heldout-transfer compare collapsed to numeric noise: {path}")
    if bool(conclusions.get("heldout_full_same_sign_as_train_full", True)):
        raise RuntimeError(f"Locked heldout-transfer compare did not preserve full-return sign flip: {path}")
    if bool(conclusions.get("heldout_suffix_same_sign_as_train_suffix", True)):
        raise RuntimeError(f"Locked heldout-transfer compare did not preserve suffix sign flip: {path}")
    return payload


def _load_locked_train_current_contract_probe(path: str) -> dict[str, Any]:
    payload = _load_json(path)
    if str(payload.get("probe_entry", "")) != "phase3_pair2_current_contract_dual_channel_family_probe":
        raise RuntimeError(f"Expected current-contract dual-channel family probe artifact at {path}")
    if str(payload.get("recommendation", {}).get("recommended_next_family", "")) != (
        "recovered_anchor_first_objective_non_tail_family"
    ):
        raise RuntimeError(
            "Locked current-contract probe no longer recommends recovered_anchor_first_objective_non_tail_family: "
            f"{path}"
        )
    return payload


def _family_mask_env_rows(
    *,
    mask: np.ndarray,
    env_indices: np.ndarray,
    objective_episode_indices: np.ndarray,
    objective_episode_positions: np.ndarray,
    raw_actor_advantages: np.ndarray,
) -> list[dict[str, Any]]:
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    env_indices = np.asarray(env_indices, dtype=np.int64).reshape(-1)
    objective_episode_indices = np.asarray(objective_episode_indices, dtype=np.int64).reshape(-1)
    objective_episode_positions = np.asarray(objective_episode_positions, dtype=np.int64).reshape(-1)
    raw_actor_advantages = np.asarray(raw_actor_advantages, dtype=np.float64).reshape(-1)
    rows: list[dict[str, Any]] = []
    for env_idx in sorted(int(v) for v in np.unique(env_indices[mask]).tolist()):
        local_mask = mask & (env_indices == int(env_idx))
        if not bool(np.any(local_mask)):
            continue
        local_positions = objective_episode_positions[local_mask]
        local_adv = raw_actor_advantages[local_mask]
        rows.append(
            {
                "env_index": int(env_idx),
                "objective_episode_indices": sorted(
                    int(v) for v in np.unique(objective_episode_indices[local_mask]).tolist()
                ),
                "token_count": int(local_mask.sum()),
                "objective_position_min": int(local_positions.min()),
                "objective_position_max": int(local_positions.max()),
                "raw_negative_mass": float(np.abs(local_adv[local_adv < 0.0]).sum()),
                "raw_positive_mass": float(np.clip(local_adv, a_min=0.0, a_max=None).sum()),
            }
        )
    return rows


def _mask_position_bucket_summary(
    *,
    mask: np.ndarray,
    env_indices: np.ndarray,
    objective_episode_positions: np.ndarray,
    raw_actor_advantages: np.ndarray,
) -> dict[str, dict[str, float]]:
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    env_indices = np.asarray(env_indices, dtype=np.int64).reshape(-1)
    objective_episode_positions = np.asarray(objective_episode_positions, dtype=np.int64).reshape(-1)
    raw_actor_advantages = np.asarray(raw_actor_advantages, dtype=np.float64).reshape(-1)
    bucket_rows = {
        name: {"token_count": 0, "raw_negative_mass": 0.0, "raw_positive_mass": 0.0}
        for name in POSITION_BUCKET_NAMES
    }
    total_negative_mass = float(np.abs(raw_actor_advantages[mask & (raw_actor_advantages < 0.0)]).sum())
    total_positive_mass = float(np.clip(raw_actor_advantages[mask], a_min=0.0, a_max=None).sum())
    total_tokens = int(mask.sum())
    for env_idx in sorted(int(v) for v in np.unique(env_indices[mask]).tolist()):
        local_mask = mask & (env_indices == int(env_idx))
        local_positions = objective_episode_positions[local_mask]
        local_adv = raw_actor_advantages[local_mask]
        if int(local_positions.size) <= 0:
            continue
        local_max = int(local_positions.max())
        denom = max(1, local_max + 1)
        bucket_ids = np.clip((local_positions * len(POSITION_BUCKET_NAMES) // denom).astype(np.int64), 0, 3)
        for bucket_idx, bucket_name in enumerate(POSITION_BUCKET_NAMES):
            bucket_mask = bucket_ids == int(bucket_idx)
            if not bool(np.any(bucket_mask)):
                continue
            bucket_adv = local_adv[bucket_mask]
            bucket_rows[bucket_name]["token_count"] += int(bucket_mask.sum())
            bucket_rows[bucket_name]["raw_negative_mass"] += float(np.abs(bucket_adv[bucket_adv < 0.0]).sum())
            bucket_rows[bucket_name]["raw_positive_mass"] += float(
                np.clip(bucket_adv, a_min=0.0, a_max=None).sum()
            )
    for bucket_name in POSITION_BUCKET_NAMES:
        bucket_rows[bucket_name]["token_share_of_family"] = (
            float(bucket_rows[bucket_name]["token_count"] / total_tokens) if total_tokens > 0 else 0.0
        )
        bucket_rows[bucket_name]["raw_negative_share_of_family"] = (
            float(bucket_rows[bucket_name]["raw_negative_mass"] / total_negative_mass)
            if total_negative_mass > 0.0
            else 0.0
        )
        bucket_rows[bucket_name]["raw_positive_share_of_family"] = (
            float(bucket_rows[bucket_name]["raw_positive_mass"] / total_positive_mass)
            if total_positive_mass > 0.0
            else 0.0
        )
    return bucket_rows


def _mask_overlap_summary(
    *,
    semantic_mask: np.ndarray,
    runtime_mask: np.ndarray,
    objective_mask: np.ndarray,
    env_indices: np.ndarray,
    objective_episode_indices: np.ndarray,
    raw_actor_advantages: np.ndarray,
) -> dict[str, Any]:
    semantic_mask = np.asarray(semantic_mask, dtype=bool).reshape(-1)
    runtime_mask = np.asarray(runtime_mask, dtype=bool).reshape(-1)
    overlap_mask = semantic_mask & runtime_mask
    semantic_metrics = _candidate_metrics(
        name="semantic",
        description="semantic family",
        mask=semantic_mask,
        objective_mask=objective_mask,
        env_indices=env_indices,
        objective_episode_indices=objective_episode_indices,
        raw_actor_advantages=raw_actor_advantages,
    )
    runtime_metrics = _candidate_metrics(
        name="runtime",
        description="runtime selector family",
        mask=runtime_mask,
        objective_mask=objective_mask,
        env_indices=env_indices,
        objective_episode_indices=objective_episode_indices,
        raw_actor_advantages=raw_actor_advantages,
    )
    overlap_metrics = _candidate_metrics(
        name="overlap",
        description="semantic/runtime overlap",
        mask=overlap_mask,
        objective_mask=objective_mask,
        env_indices=env_indices,
        objective_episode_indices=objective_episode_indices,
        raw_actor_advantages=raw_actor_advantages,
    )
    semantic_negative_mass = float(semantic_metrics["raw_negative_mass"])
    semantic_token_count = int(semantic_metrics["token_count"])
    return {
        "runtime_selector_matches_semantic_family_exact": bool(np.array_equal(runtime_mask, semantic_mask)),
        "semantic_token_count": int(semantic_metrics["token_count"]),
        "runtime_token_count": int(runtime_metrics["token_count"]),
        "overlap_token_count": int(overlap_metrics["token_count"]),
        "semantic_negative_mass": float(semantic_negative_mass),
        "runtime_negative_mass": float(runtime_metrics["raw_negative_mass"]),
        "overlap_negative_mass": float(overlap_metrics["raw_negative_mass"]),
        "runtime_token_share_of_semantic": (
            float(int(runtime_metrics["token_count"]) / semantic_token_count) if semantic_token_count > 0 else 0.0
        ),
        "overlap_negative_share_of_semantic": (
            float(float(overlap_metrics["raw_negative_mass"]) / semantic_negative_mass)
            if semantic_negative_mass > 0.0
            else 0.0
        ),
        "runtime_env_indices": list(runtime_metrics["env_indices"]),
        "semantic_env_indices": list(semantic_metrics["env_indices"]),
    }


def _summarize_delta_group(rows: list[dict[str, float]], *, env_indices: list[int]) -> dict[str, Any]:
    env_set = {int(v) for v in env_indices}
    subset = [dict(row) for row in rows if int(row["env_index"]) in env_set]
    full_sum = float(sum(float(row["full_gap_delta"]) for row in subset))
    suffix_sum = float(sum(float(row["suffix_gap_delta"]) for row in subset))
    return {
        "env_indices": sorted(int(v) for v in env_set),
        "count": int(len(subset)),
        "full_gap_delta_sum": float(full_sum),
        "suffix_gap_delta_sum": float(suffix_sum),
        "full_gap_delta_mean": float(full_sum / len(subset)) if subset else 0.0,
        "suffix_gap_delta_mean": float(suffix_sum / len(subset)) if subset else 0.0,
    }


def _build_post_gap_delta_rows(
    *,
    baseline_report: dict[str, Any],
    override_report: dict[str, Any],
    split: str,
) -> list[dict[str, float]]:
    zero_full = list(dict(baseline_report["zero_control"])[split]["full_return_per_env"])
    zero_suffix = list(dict(baseline_report["zero_control"])[split]["suffix_return_per_env"])
    baseline_post = list(dict(baseline_report["post"])[split]["full_return_per_env"])
    baseline_post_suffix = list(dict(baseline_report["post"])[split]["suffix_return_per_env"])
    override_post = list(dict(override_report["post"])[split]["full_return_per_env"])
    override_post_suffix = list(dict(override_report["post"])[split]["suffix_return_per_env"])
    rows = []
    for env_index, (zf, zs, bf, bs, of, os) in enumerate(
        zip(zero_full, zero_suffix, baseline_post, baseline_post_suffix, override_post, override_post_suffix)
    ):
        rows.append(
            {
                "env_index": int(env_index),
                "baseline_post_vs_zero_full": float(bf - zf),
                "override_post_vs_zero_full": float(of - zf),
                "full_gap_delta": float(of - bf),
                "baseline_post_vs_zero_suffix": float(bs - zs),
                "override_post_vs_zero_suffix": float(os - zs),
                "suffix_gap_delta": float(os - bs),
            }
        )
    return rows


def _top_rows(rows: list[dict[str, float]], *, key: str, descending: bool, limit: int = 4) -> list[dict[str, float]]:
    return sorted(rows, key=lambda row: float(row[key]), reverse=bool(descending))[: int(limit)]


def _suite_delta_breakdown(
    *,
    rows: list[dict[str, float]],
    runtime_target_envs: list[int],
    semantic_envs: list[int],
) -> dict[str, Any]:
    total_full = float(sum(float(row["full_gap_delta"]) for row in rows))
    total_suffix = float(sum(float(row["suffix_gap_delta"]) for row in rows))
    runtime_group = _summarize_delta_group(rows, env_indices=runtime_target_envs)
    semantic_group = _summarize_delta_group(rows, env_indices=semantic_envs)
    complement_runtime = _summarize_delta_group(
        rows,
        env_indices=sorted(int(row["env_index"]) for row in rows if int(row["env_index"]) not in set(runtime_target_envs)),
    )
    complement_semantic = _summarize_delta_group(
        rows,
        env_indices=sorted(int(row["env_index"]) for row in rows if int(row["env_index"]) not in set(semantic_envs)),
    )
    return {
        "per_env_rows": rows,
        "full_gap_delta_sum": float(total_full),
        "suffix_gap_delta_sum": float(total_suffix),
        "runtime_target_group": runtime_group,
        "runtime_target_complement_group": complement_runtime,
        "semantic_recovered_group": semantic_group,
        "semantic_recovered_complement_group": complement_semantic,
        "top_positive_suffix_envs": _top_rows(rows, key="suffix_gap_delta", descending=True),
        "top_negative_suffix_envs": _top_rows(rows, key="suffix_gap_delta", descending=False),
        "top_positive_full_envs": _top_rows(rows, key="full_gap_delta", descending=True),
        "top_negative_full_envs": _top_rows(rows, key="full_gap_delta", descending=False),
    }


def _assert_close_float(name: str, actual: float, expected: float, *, atol: float = 1e-6) -> None:
    if abs(float(actual) - float(expected)) > float(atol):
        raise RuntimeError(
            f"{name} drifted beyond tolerance: actual={float(actual)!r}, expected={float(expected)!r}"
        )


def _assert_train_semantic_matches_locked_probe(
    *,
    train_snapshot: dict[str, Any],
    locked_probe: dict[str, Any],
) -> None:
    locked_candidate = dict(locked_probe["candidate_families"]["recovered_anchor_first_objective_non_tail_family"])
    observed = dict(train_snapshot["semantic_family"])
    if list(observed["env_indices"]) != list(locked_candidate["env_indices"]):
        raise RuntimeError(
            "Train recovered semantic family env indices drifted from locked current-contract probe: "
            f"{observed['env_indices']!r} vs {locked_candidate['env_indices']!r}"
        )
    if int(observed["token_count"]) != int(locked_candidate["token_count"]):
        raise RuntimeError(
            "Train recovered semantic family token_count drifted from locked current-contract probe: "
            f"{observed['token_count']!r} vs {locked_candidate['token_count']!r}"
        )
    _assert_close_float(
        "train.semantic_family.raw_negative_share_of_batch",
        float(observed["raw_negative_share_of_batch"]),
        float(locked_candidate["raw_negative_share_of_batch"]),
    )
    _assert_close_float(
        "train.semantic_family.raw_negative_mass",
        float(observed["raw_negative_mass"]),
        float(locked_candidate["raw_negative_mass"]),
    )


def _collect_suite_family_snapshot(
    *,
    model,
    config: dict[str, Any],
    suite_path: str,
    antitransfer_json: str,
    device_obj: torch.device,
    n_samples: int,
    single_eval_pos: int,
    train_profile: str,
) -> dict[str, Any]:
    optimizer_cfg = dict(config.get("optimizer", {}))
    profile_cfg = _resolve_train_profile(
        optimizer_cfg=optimizer_cfg,
        train_profile=str(train_profile),
        batch_size=None,
        n_epochs=None,
        learning_rate=None,
        target_kl=None,
    )
    suite = _load_required_suite(suite_path)
    anti = _load_json(antitransfer_json)
    recovered_envs = [int(v) for v in anti["pair2_anchor_structure"]["recovered_dual_channel_env_indices"]]
    residual_envs = [int(v) for v in anti["pair2_anchor_structure"]["residual_missed_env_indices"]]

    env_cfg = _resolve_audit_env_config(config, core_a=False, reference_semantics_enabled=False)
    prior = EnvironmentPrior(copy.deepcopy(env_cfg))
    prior._sample_batch_hypers = lambda batch_n: [copy.deepcopy(h) for h in list(suite["h_list"])]
    prior._sample_single_eval_pos = lambda n_samples_arg, rng_seed=None: int(single_eval_pos)

    algo, callback, vec_env = build_recurrent_ppo(
        model=model,
        env_prior=prior,
        device=str(device_obj),
        num_features=int(config["prior"]["num_features"]),
        n_envs=int(suite["batch_size"]),
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
        actor_objective_mode="tokenwise",
        **_train_rollout_quality_collect_contract_kwargs(suite),
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
    _bind_vec_env_to_fixed_suite(vec_env, suite=suite, single_eval_pos=int(single_eval_pos))
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
    vec_env.close()

    if not bool(np.any(objective_mask)):
        raise RuntimeError(f"Expected objective-valid tokens in suite snapshot {suite_path}")

    env_rows = _build_env_episode_rows(
        env_indices=env_indices,
        objective_episode_indices=objective_episode_indices,
        objective_episode_positions=objective_episode_positions,
        objective_mask=objective_mask,
        raw_actor_advantages=raw_actor_advantages,
        env_ids=sorted({*_runtime_target_env_indices(), *recovered_envs, *residual_envs}),
    )
    env_row_by_id = {int(row["env_index"]): row for row in env_rows}

    semantic_mask = np.zeros_like(objective_mask, dtype=bool)
    semantic_episode_index_by_env: dict[int, int] = {}
    semantic_position_end_by_env: dict[int, int] = {}
    for env_id in recovered_envs:
        row = env_row_by_id.get(int(env_id))
        if row is None or row["first_objective_episode_index"] is None:
            continue
        first_episode_index = int(row["first_objective_episode_index"])
        local_episode_mask = (
            objective_mask
            & (env_indices == int(env_id))
            & (objective_episode_indices == int(first_episode_index))
        )
        positions = objective_episode_positions[local_episode_mask]
        if int(positions.size) <= 0:
            continue
        position_end = int(positions.max()) - int(PAIR2_RECOVERED_ANCHOR_FIRST_OBJECTIVE_NON_TAIL_FAMILY_TAIL_LEN)
        if int(position_end) >= 0:
            semantic_episode_index_by_env[int(env_id)] = int(first_episode_index)
            semantic_position_end_by_env[int(env_id)] = int(position_end)
            semantic_mask |= local_episode_mask & (objective_episode_positions <= int(position_end))

    runtime_mask = np.zeros_like(objective_mask, dtype=bool)
    runtime_episode_index_by_env: dict[int, int] = {}
    runtime_position_end_by_env: dict[int, int] = {}
    for env_id in _runtime_target_env_indices():
        local_episode_mask = (
            objective_mask
            & (env_indices == int(env_id))
            & (objective_episode_indices == int(PAIR2_RECOVERED_ANCHOR_FIRST_OBJECTIVE_NON_TAIL_FAMILY_OBJECTIVE_EPISODE_INDEX))
        )
        positions = objective_episode_positions[local_episode_mask]
        if int(positions.size) <= 0:
            continue
        position_end = int(positions.max()) - int(PAIR2_RECOVERED_ANCHOR_FIRST_OBJECTIVE_NON_TAIL_FAMILY_TAIL_LEN)
        runtime_episode_index_by_env[int(env_id)] = int(
            PAIR2_RECOVERED_ANCHOR_FIRST_OBJECTIVE_NON_TAIL_FAMILY_OBJECTIVE_EPISODE_INDEX
        )
        runtime_position_end_by_env[int(env_id)] = int(position_end)
        if int(position_end) >= 0:
            runtime_mask |= local_episode_mask & (objective_episode_positions <= int(position_end))

    semantic_metrics = _candidate_metrics(
        name="semantic_recovered_anchor_first_objective_non_tail_family",
        description="Suite-local semantic recovered-anchor first-objective non-tail family.",
        mask=semantic_mask,
        objective_mask=objective_mask,
        env_indices=env_indices,
        objective_episode_indices=objective_episode_indices,
        raw_actor_advantages=raw_actor_advantages,
    )
    runtime_metrics = _candidate_metrics(
        name="runtime_selector_family",
        description="Exact runtime selector coordinates used by the pair2 recovered-anchor non-tail mode.",
        mask=runtime_mask,
        objective_mask=objective_mask,
        env_indices=env_indices,
        objective_episode_indices=objective_episode_indices,
        raw_actor_advantages=raw_actor_advantages,
    )
    overlap_summary = _mask_overlap_summary(
        semantic_mask=semantic_mask,
        runtime_mask=runtime_mask,
        objective_mask=objective_mask,
        env_indices=env_indices,
        objective_episode_indices=objective_episode_indices,
        raw_actor_advantages=raw_actor_advantages,
    )

    return {
        "suite_path": str(Path(suite_path).expanduser().resolve()),
        "suite_batch_size": int(suite["batch_size"]),
        "recovered_envs_from_antitransfer_probe": list(recovered_envs),
        "runtime_target_env_indices": _runtime_target_env_indices(),
        "runtime_target_objective_episode_index": int(
            PAIR2_RECOVERED_ANCHOR_FIRST_OBJECTIVE_NON_TAIL_FAMILY_OBJECTIVE_EPISODE_INDEX
        ),
        "runtime_tail_len": int(PAIR2_RECOVERED_ANCHOR_FIRST_OBJECTIVE_NON_TAIL_FAMILY_TAIL_LEN),
        "current_contract_env_rows": env_rows,
        "semantic_family": semantic_metrics,
        "runtime_selector_family": runtime_metrics,
        "semantic_family_env_rows": _family_mask_env_rows(
            mask=semantic_mask,
            env_indices=env_indices,
            objective_episode_indices=objective_episode_indices,
            objective_episode_positions=objective_episode_positions,
            raw_actor_advantages=raw_actor_advantages,
        ),
        "runtime_selector_env_rows": _family_mask_env_rows(
            mask=runtime_mask,
            env_indices=env_indices,
            objective_episode_indices=objective_episode_indices,
            objective_episode_positions=objective_episode_positions,
            raw_actor_advantages=raw_actor_advantages,
        ),
        "semantic_family_episode_index_by_env": {str(int(k)): int(v) for k, v in sorted(semantic_episode_index_by_env.items())},
        "semantic_family_position_end_by_env": {str(int(k)): int(v) for k, v in sorted(semantic_position_end_by_env.items())},
        "runtime_selector_episode_index_by_env": {str(int(k)): int(v) for k, v in sorted(runtime_episode_index_by_env.items())},
        "runtime_selector_position_end_by_env": {str(int(k)): int(v) for k, v in sorted(runtime_position_end_by_env.items())},
        "semantic_family_position_buckets": _mask_position_bucket_summary(
            mask=semantic_mask,
            env_indices=env_indices,
            objective_episode_positions=objective_episode_positions,
            raw_actor_advantages=raw_actor_advantages,
        ),
        "runtime_selector_position_buckets": _mask_position_bucket_summary(
            mask=runtime_mask,
            env_indices=env_indices,
            objective_episode_positions=objective_episode_positions,
            raw_actor_advantages=raw_actor_advantages,
        ),
        "semantic_vs_runtime_overlap": overlap_summary,
    }


def build_pair2_recovered_anchor_first_objective_non_tail_antitransfer_decomposition_probe(
    *,
    checkpoint_path: str,
    train_suite_path: str,
    heldout_suite_path: str,
    antitransfer_json: str,
    locked_heldout_transfer_compare_json: str,
    locked_train_current_contract_probe_json: str,
    phase2_summary_path: str,
    device: str | None,
    n_samples: int,
    single_eval_pos: int,
    train_profile: str,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    phase2_summary = assert_phase2_green(phase2_summary_path)
    locked_compare = _load_locked_heldout_transfer_compare(locked_heldout_transfer_compare_json)
    locked_train_probe = _load_locked_train_current_contract_probe(locked_train_current_contract_probe_json)
    baseline_report = _load_json(locked_compare["baseline_eval_report"]["path"])
    override_report = _load_json(locked_compare["override_eval_report"]["path"])

    load_model.cache_clear()
    device_obj = torch.device(str(device or _default_device()))
    model, config = load_model(str(Path(checkpoint_path).expanduser().resolve()), device=device_obj, verbose=False)

    train_snapshot = _collect_suite_family_snapshot(
        model=model,
        config=config,
        suite_path=train_suite_path,
        antitransfer_json=antitransfer_json,
        device_obj=device_obj,
        n_samples=int(n_samples),
        single_eval_pos=int(single_eval_pos),
        train_profile=str(train_profile),
    )
    heldout_snapshot = _collect_suite_family_snapshot(
        model=model,
        config=config,
        suite_path=heldout_suite_path,
        antitransfer_json=antitransfer_json,
        device_obj=device_obj,
        n_samples=int(n_samples),
        single_eval_pos=int(single_eval_pos),
        train_profile=str(train_profile),
    )
    _assert_train_semantic_matches_locked_probe(
        train_snapshot=train_snapshot,
        locked_probe=locked_train_probe,
    )

    train_rows = _build_post_gap_delta_rows(
        baseline_report=baseline_report,
        override_report=override_report,
        split="train",
    )
    heldout_rows = _build_post_gap_delta_rows(
        baseline_report=baseline_report,
        override_report=override_report,
        split="heldout",
    )
    train_delta = _suite_delta_breakdown(
        rows=train_rows,
        runtime_target_envs=_runtime_target_env_indices(),
        semantic_envs=[int(v) for v in train_snapshot["semantic_family"]["env_indices"]],
    )
    heldout_delta = _suite_delta_breakdown(
        rows=heldout_rows,
        runtime_target_envs=_runtime_target_env_indices(),
        semantic_envs=[int(v) for v in heldout_snapshot["semantic_family"]["env_indices"]],
    )

    train_suffix_sum = float(train_delta["suffix_gap_delta_sum"])
    heldout_suffix_sum = float(heldout_delta["suffix_gap_delta_sum"])
    train_runtime_target_suffix = float(train_delta["runtime_target_group"]["suffix_gap_delta_sum"])
    heldout_runtime_target_suffix = float(heldout_delta["runtime_target_group"]["suffix_gap_delta_sum"])
    heldout_semantic_suffix = float(heldout_delta["semantic_recovered_group"]["suffix_gap_delta_sum"])

    return {
        "probe_entry": "phase3_pair2_recovered_anchor_first_objective_non_tail_antitransfer_decomposition_probe",
        "phase2_preflight": {
            "required": True,
            "summary_path": str(Path(phase2_summary_path).expanduser().resolve()),
            "phase2_all_checks_pass": bool(phase2_summary["validation"]["all_checks_pass"]),
            "phase2_pass_flags": dict(phase2_summary["pass_flags"]),
        },
        "inputs": {
            "checkpoint_path": str(Path(checkpoint_path).expanduser().resolve()),
            "train_suite_path": str(Path(train_suite_path).expanduser().resolve()),
            "heldout_suite_path": str(Path(heldout_suite_path).expanduser().resolve()),
            "antitransfer_json": str(Path(antitransfer_json).expanduser().resolve()),
            "locked_heldout_transfer_compare_json": str(Path(locked_heldout_transfer_compare_json).expanduser().resolve()),
            "locked_train_current_contract_probe_json": str(
                Path(locked_train_current_contract_probe_json).expanduser().resolve()
            ),
            "n_samples": int(n_samples),
            "single_eval_pos": int(single_eval_pos),
            "train_profile": str(train_profile),
        },
        "locked_compare": {
            "path": str(Path(locked_heldout_transfer_compare_json).expanduser().resolve()),
            "train_full_return_delta_change": float(locked_compare["comparison"]["train_full_return_delta_change"]),
            "train_suffix_return_delta_change": float(locked_compare["comparison"]["train_suffix_return_delta_change"]),
            "heldout_full_return_delta_change": float(locked_compare["comparison"]["heldout_full_return_delta_change"]),
            "heldout_suffix_return_delta_change": float(locked_compare["comparison"]["heldout_suffix_return_delta_change"]),
        },
        "train_suite_snapshot": train_snapshot,
        "heldout_suite_snapshot": heldout_snapshot,
        "outcome_decomposition": {
            "train": train_delta,
            "heldout": heldout_delta,
        },
        "cross_suite_comparisons": {
            "train_runtime_selector_matches_semantic_family_exact": bool(
                train_snapshot["semantic_vs_runtime_overlap"]["runtime_selector_matches_semantic_family_exact"]
            ),
            "heldout_runtime_selector_matches_semantic_family_exact": bool(
                heldout_snapshot["semantic_vs_runtime_overlap"]["runtime_selector_matches_semantic_family_exact"]
            ),
            "train_semantic_raw_negative_share_of_batch": float(
                train_snapshot["semantic_family"]["raw_negative_share_of_batch"]
            ),
            "heldout_semantic_raw_negative_share_of_batch": float(
                heldout_snapshot["semantic_family"]["raw_negative_share_of_batch"]
            ),
            "heldout_over_train_semantic_negative_share_ratio": (
                float(
                    float(heldout_snapshot["semantic_family"]["raw_negative_share_of_batch"])
                    / float(train_snapshot["semantic_family"]["raw_negative_share_of_batch"])
                )
                if float(train_snapshot["semantic_family"]["raw_negative_share_of_batch"]) > 0.0
                else 0.0
            ),
            "heldout_runtime_overlap_negative_share_of_semantic": float(
                heldout_snapshot["semantic_vs_runtime_overlap"]["overlap_negative_share_of_semantic"]
            ),
            "train_runtime_overlap_negative_share_of_semantic": float(
                train_snapshot["semantic_vs_runtime_overlap"]["overlap_negative_share_of_semantic"]
            ),
        },
        "conclusions": {
            "train_gain_is_not_runtime_target_local": bool(
                train_suffix_sum > 0.0 and train_runtime_target_suffix <= 0.0
            ),
            "train_gain_is_spillover_dominant": bool(
                train_suffix_sum > 0.0
                and float(train_delta["runtime_target_complement_group"]["suffix_gap_delta_sum"])
                > abs(train_runtime_target_suffix)
            ),
            "heldout_loss_is_not_runtime_target_local": bool(
                heldout_suffix_sum < 0.0
                and abs(heldout_runtime_target_suffix) < abs(heldout_suffix_sum)
            ),
            "heldout_loss_is_not_semantic_family_local": bool(
                heldout_suffix_sum < 0.0
                and abs(heldout_semantic_suffix) < abs(heldout_suffix_sum)
            ),
            "heldout_semantic_family_adds_env13": bool(
                13 in set(int(v) for v in heldout_snapshot["semantic_family"]["env_indices"])
                and 13 not in set(int(v) for v in train_snapshot["semantic_family"]["env_indices"])
            ),
            "heldout_env15_first_objective_episode_shifted_away_from_zero": bool(
                int(heldout_snapshot["semantic_family_episode_index_by_env"].get("15", -1)) != 0
            ),
            "runtime_selector_is_not_suite_invariant_semantic_family": bool(
                train_snapshot["semantic_vs_runtime_overlap"]["runtime_selector_matches_semantic_family_exact"]
                and not heldout_snapshot["semantic_vs_runtime_overlap"]["runtime_selector_matches_semantic_family_exact"]
            ),
            "heldout_semantic_negative_mass_exceeds_train_semantic_negative_mass": bool(
                float(heldout_snapshot["semantic_family"]["raw_negative_share_of_batch"])
                > float(train_snapshot["semantic_family"]["raw_negative_share_of_batch"])
            ),
        },
        "runtime_s": float(time.perf_counter() - t0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 3 decomposition of the recovered-anchor first-objective non-tail family anti-transfer line."
        )
    )
    parser.add_argument("--checkpoint-path", type=str, default=PAIR2_CHECKPOINT_PATH)
    parser.add_argument("--train-suite-path", type=str, default=PAIR2_TRAIN_SUITE_PATH)
    parser.add_argument("--heldout-suite-path", type=str, default=PAIR2_HELDOUT_SUITE_PATH)
    parser.add_argument("--antitransfer-json", type=str, default=DEFAULT_ANTITRANSFER_JSON)
    parser.add_argument(
        "--locked-heldout-transfer-compare-json",
        type=str,
        default=DEFAULT_LOCKED_HELDOUT_TRANSFER_COMPARE_JSON,
    )
    parser.add_argument(
        "--locked-train-current-contract-probe-json",
        type=str,
        default=DEFAULT_LOCKED_TRAIN_CURRENT_CONTRACT_PROBE_JSON,
    )
    parser.add_argument("--phase2-summary-path", type=str, default=DEFAULT_PHASE2_SUMMARY)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--n-samples", type=int, default=PAIR2_TRAIN_UPDATE_NSAMPLES)
    parser.add_argument("--single-eval-pos", type=int, default=PAIR2_TRAIN_UPDATE_SINGLE_EVAL_POS)
    parser.add_argument(
        "--train-profile",
        type=str,
        default="phase2_shared_backbone_contract",
        choices=[
            "phase2_shared_backbone_contract",
            "trusted_sep_reset_mainline",
            "legacy_actor_only_probe",
        ],
    )
    parser.add_argument("--output-json", type=str, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output_json).expanduser().resolve()
    if output_path.exists() and not bool(args.overwrite):
        print(output_path.read_text(encoding="utf-8"))
        return 0

    report = build_pair2_recovered_anchor_first_objective_non_tail_antitransfer_decomposition_probe(
        checkpoint_path=args.checkpoint_path,
        train_suite_path=args.train_suite_path,
        heldout_suite_path=args.heldout_suite_path,
        antitransfer_json=args.antitransfer_json,
        locked_heldout_transfer_compare_json=args.locked_heldout_transfer_compare_json,
        locked_train_current_contract_probe_json=args.locked_train_current_contract_probe_json,
        phase2_summary_path=args.phase2_summary_path,
        device=args.device,
        n_samples=args.n_samples,
        single_eval_pos=args.single_eval_pos,
        train_profile=args.train_profile,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
