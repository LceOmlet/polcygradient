import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

import torch

from ticl.analysis.critic_free_single_env_audit import (
    _audit_train_loop,
    _build_audit_env_cfg,
    _make_deterministic_batch_plan,
)
from ticl.analysis.phase2_guardrail import DEFAULT_PHASE2_SUMMARY, assert_phase2_green
from ticl.analysis.phase3_multi_env_optimization_audit import (
    _bind_vec_env_to_fixed_suite,
    _clone_suite_h_list,
    _default_device,
    _load_required_suite,
    _resolve_train_profile,
)
from ticl.analysis.phase3_pair2_train_update_ab_compare_pack import (
    PAIR2_BRANCH_SPECIFIC_MODE,
    PAIR2_CHECKPOINT_PATH,
    PAIR2_SNAPSHOT_TARGET_OUTER_BATCH_IDX,
    PAIR2_SNAPSHOT_TARGET_ENV_INDEX,
    PAIR2_SNAPSHOT_TARGET_OBJECTIVE_EPISODE_INDEX,
    PAIR2_SNAPSHOT_TARGET_OBJECTIVE_POSITION_END,
    PAIR2_SNAPSHOT_TARGET_OBJECTIVE_POSITION_START,
    PAIR2_TRAIN_SUITE_PATH,
    PAIR2_TRAIN_UPDATE_NSAMPLES,
    PAIR2_TRAIN_UPDATE_SINGLE_EVAL_POS,
    _load_pair2_preflight,
)
from ticl.model_builder import load_model
from ticl.priors.environment_prior import EnvironmentPrior
from ticl.sb3_recurrent_ppo import build_recurrent_ppo


DEFAULT_OUTPUT_JSON = (
    "/home/chen/RLPFN/artifacts/phase3_pair2_guarded_target_present_selector_trace.json"
)
DEFAULT_BASELINE_SELECTOR_TRACE_JSON = (
    "/home/chen/RLPFN/artifacts/phase3_pair2_guarded_target_present_baseline_selector_trace.json"
)
DEFAULT_BASELINE_SNAPSHOT_JSON = (
    "/home/chen/RLPFN/artifacts/phase3_pair2_guarded_target_present_baseline_snapshot.json"
)
DEFAULT_OVERRIDE_SELECTOR_TRACE_JSON = (
    "/home/chen/RLPFN/artifacts/phase3_pair2_guarded_target_present_override_selector_trace.json"
)
DEFAULT_OVERRIDE_SNAPSHOT_JSON = (
    "/home/chen/RLPFN/artifacts/phase3_pair2_guarded_target_present_override_snapshot.json"
)


def _configure_capture(
    algo,
    *,
    snapshot_json: str,
    selector_trace_json: str,
    target_outer_batch_idx: int | None,
    target_env_index: int,
    target_objective_episode_index: int,
    target_objective_position_start: int,
    target_objective_position_end: int,
) -> None:
    algo._rwkv_train_outer_batch_snapshot_path = str(Path(snapshot_json).expanduser().resolve())
    algo._rwkv_train_outer_batch_snapshot_written = False
    algo._rwkv_last_train_outer_batch_snapshot_path = None
    algo._rwkv_train_outer_batch_selector_trace_path = str(
        Path(selector_trace_json).expanduser().resolve()
    )
    algo._rwkv_train_outer_batch_selector_trace_rows = []
    algo._rwkv_train_outer_batch_selector_trace_completed = False
    algo._rwkv_last_train_outer_batch_selector_trace_path = None
    algo._rwkv_train_outer_batch_snapshot_target_outer_batch_idx = (
        None if target_outer_batch_idx is None else int(target_outer_batch_idx)
    )
    algo._rwkv_train_outer_batch_snapshot_target_env_index = int(target_env_index)
    algo._rwkv_train_outer_batch_snapshot_target_objective_episode_index = int(
        target_objective_episode_index
    )
    algo._rwkv_train_outer_batch_snapshot_target_objective_position_start = int(
        target_objective_position_start
    )
    algo._rwkv_train_outer_batch_snapshot_target_objective_position_end = int(
        target_objective_position_end
    )


def _load_optional_json(path: str | None) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        return None
    return json.loads(resolved.read_text())


def _build_capture_summary(
    *,
    selector_trace: dict[str, Any] | None,
    snapshot: dict[str, Any] | None,
) -> dict[str, Any]:
    trace_summary = None if selector_trace is None else dict(selector_trace.get("summary", {}))
    snapshot_summary = None if snapshot is None else dict(snapshot.get("summary", {}))
    target_selector = None if snapshot is None else dict(snapshot.get("snapshot_target_selector", {}))
    first_match = None if not trace_summary else trace_summary.get("first_match")
    snapshot_outer_batch_idx = (
        None if snapshot_summary is None else int(snapshot_summary["outer_batch_idx"])
    )
    return {
        "selector_trace_written": bool(selector_trace is not None),
        "snapshot_written": bool(snapshot is not None),
        "target_appeared": bool(trace_summary and trace_summary.get("target_appeared", False)),
        "first_match": first_match,
        "snapshot_target_match_count": None
        if snapshot is None
        else int(snapshot.get("snapshot_target_match_count", 0)),
        "snapshot_outer_batch_idx": snapshot_outer_batch_idx,
        "snapshot_objective_total": None
        if snapshot_summary is None
        else int(snapshot_summary["objective_total"]),
        "snapshot_target_selector": target_selector,
        "same_hit_relationship_preserved": bool(
            trace_summary
            and trace_summary.get("target_appeared", False)
            and snapshot is not None
            and first_match is not None
            and snapshot_outer_batch_idx == int(first_match["outer_batch_idx"])
        ),
    }


def _sorted_positions(row: dict[str, Any]) -> list[int]:
    return sorted(int(v) for v in list(row.get("target_episode_positions_present", [])))


def _contiguous_span(positions: list[int]) -> dict[str, int] | None:
    if not positions:
        return None
    return {
        "position_start": int(min(positions)),
        "position_end": int(max(positions)),
        "token_count": int(len(positions)),
    }


def _candidate_rows(trace_payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = list(trace_payload.get("rows", []))
    candidates: list[dict[str, Any]] = []
    for row in rows:
        if int(row.get("target_env_objective_token_count", 0)) <= 0:
            continue
        if int(row.get("target_episode_token_count", 0)) <= 0:
            continue
        positions = _sorted_positions(row)
        span = _contiguous_span(positions)
        if span is None:
            continue
        candidates.append(
            {
                "epoch_idx": int(row["epoch_idx"]),
                "outer_batch_idx": int(row["outer_batch_idx"]),
                "target_status": str(row["target_status"]),
                "target_env_index": int(row["target_env_index"]),
                "target_objective_episode_index": int(row["target_objective_episode_index"]),
                "target_episode_positions_present": positions,
                "candidate_position_start": int(span["position_start"]),
                "candidate_position_end": int(span["position_end"]),
                "candidate_token_count": int(span["token_count"]),
                "target_env_objective_episode_spans": [
                    {
                        "objective_episode_index": int(seg["objective_episode_index"]),
                        "token_count": int(seg["token_count"]),
                        "objective_position_min": int(seg["objective_position_min"]),
                        "objective_position_max": int(seg["objective_position_max"]),
                    }
                    for seg in list(row.get("target_env_objective_episode_spans", []))
                ],
            }
        )
    return candidates


def _pick_canonical_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not candidates:
        return None
    ordered = sorted(
        candidates,
        key=lambda row: (
            int(row["outer_batch_idx"]),
            int(row["epoch_idx"]),
            -int(row["candidate_token_count"]),
            int(row["candidate_position_start"]),
        ),
    )
    return dict(ordered[0])


def _row_covers_requested_range(
    row: dict[str, Any],
    *,
    target_objective_position_start: int,
    target_objective_position_end: int,
) -> bool:
    positions = set(_sorted_positions(row))
    if not positions:
        return False
    for pos in range(
        int(target_objective_position_start),
        int(target_objective_position_end) + 1,
    ):
        if int(pos) not in positions:
            return False
    return True


def _condense_env_present_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "epoch_idx": int(row["epoch_idx"]),
        "outer_batch_idx": int(row["outer_batch_idx"]),
        "target_status": str(row["target_status"]),
        "objective_env_indices_present": [int(v) for v in list(row["objective_env_indices_present"])],
        "target_env_objective_token_count": int(row["target_env_objective_token_count"]),
        "target_env_objective_episode_indices_present": [
            int(v) for v in list(row.get("target_env_objective_episode_indices_present", []))
        ],
        "target_env_objective_episode_spans": [
            {
                "objective_episode_index": int(seg["objective_episode_index"]),
                "token_count": int(seg["token_count"]),
                "objective_position_min": int(seg["objective_position_min"]),
                "objective_position_max": int(seg["objective_position_max"]),
            }
            for seg in list(row.get("target_env_objective_episode_spans", []))
        ],
        "target_episode_token_count": int(row["target_episode_token_count"]),
        "target_episode_positions_present": _sorted_positions(row),
        "target_match_count": int(row["target_match_count"]),
        "target_match_positions": [int(v) for v in list(row.get("target_match_positions", []))],
    }


def _build_arm_summary(
    *,
    mode_label: str,
    actor_objective_mode_override: str | None,
    actor_objective_runtime_current_suite_name: str | None,
    selector_trace: dict[str, Any] | None,
    snapshot: dict[str, Any] | None,
    elapsed_wall_time_sec: float,
    last_update_wall_time_sec: float,
    total_env_steps: int,
    target_selector: dict[str, int | None],
) -> dict[str, Any]:
    rows = [] if selector_trace is None else list(selector_trace.get("rows", []))
    env_present_rows = [
        dict(row) for row in rows if int(row.get("target_env_objective_token_count", 0)) > 0
    ]
    episode_present_rows = [
        dict(row) for row in env_present_rows if int(row.get("target_episode_token_count", 0)) > 0
    ]
    requested_range_present_rows = [
        dict(row)
        for row in episode_present_rows
        if _row_covers_requested_range(
            row,
            target_objective_position_start=int(target_selector["target_objective_position_start"]),
            target_objective_position_end=int(target_selector["target_objective_position_end"]),
        )
    ]
    matched_rows = [
        dict(row) for row in rows if int(row.get("target_match_count", 0)) > 0
    ]
    candidates = _candidate_rows(selector_trace or {})
    canonical_candidate = _pick_canonical_candidate(candidates)
    selector_match_bug_suspected = bool(
        not matched_rows and requested_range_present_rows
    )
    if matched_rows:
        absence_reason = "target_present"
    elif not env_present_rows:
        absence_reason = "target_env_absent"
    elif not episode_present_rows:
        absence_reason = "target_episode_absent"
    elif not requested_range_present_rows:
        absence_reason = "target_position_range_absent"
    else:
        absence_reason = "selector_match_bug_suspected"
    return {
        "mode_label": str(mode_label),
        "actor_objective_mode_override": None
        if actor_objective_mode_override is None
        else str(actor_objective_mode_override),
        "actor_objective_runtime_current_suite_name": None
        if actor_objective_runtime_current_suite_name is None
        else str(actor_objective_runtime_current_suite_name),
        "runtime": {
            "elapsed_wall_time_sec": float(elapsed_wall_time_sec),
            "last_update_wall_time_sec": float(last_update_wall_time_sec),
            "elapsed_sec_per_env_step": float(elapsed_wall_time_sec / float(max(1, total_env_steps))),
            "update_sec_per_env_step": float(last_update_wall_time_sec / float(max(1, total_env_steps))),
        },
        "capture": _build_capture_summary(selector_trace=selector_trace, snapshot=snapshot),
        "selector_trace": {
            "path": None if selector_trace is None else selector_trace.get("source_path", None),
            "completed": bool(selector_trace and selector_trace.get("completed", False)),
            "summary": None if selector_trace is None else dict(selector_trace.get("summary", {})),
            "target_absence_reason": str(absence_reason),
            "selector_match_bug_suspected": bool(selector_match_bug_suspected),
            "env_present_outer_batches": [
                int(row["outer_batch_idx"]) for row in env_present_rows
            ],
            "episode_present_outer_batches": [
                int(row["outer_batch_idx"]) for row in episode_present_rows
            ],
            "requested_position_range_present_outer_batches": [
                int(row["outer_batch_idx"]) for row in requested_range_present_rows
            ],
            "matched_outer_batches": [
                int(row["outer_batch_idx"]) for row in matched_rows
            ],
            "env_present_rows": [
                _condense_env_present_row(row) for row in env_present_rows
            ],
            "candidate_rows": candidates,
            "canonical_present_selector": canonical_candidate,
            "canonical_selector_preserves_env_and_episode_identity": bool(
                canonical_candidate is not None
                and int(canonical_candidate["target_env_index"]) == int(target_selector["target_env_index"])
                and int(canonical_candidate["target_objective_episode_index"])
                == int(target_selector["target_objective_episode_index"])
            ),
            "canonical_selector_requires_position_change": bool(
                canonical_candidate is not None
                and (
                    int(canonical_candidate["candidate_position_start"])
                    != int(target_selector["target_objective_position_start"])
                    or int(canonical_candidate["candidate_position_end"])
                    != int(target_selector["target_objective_position_end"])
                )
            ),
            "row_count": int(len(rows)),
            "rows": rows,
        },
        "snapshot": {
            "path": None if snapshot is None else snapshot.get("source_path", None),
            "summary": None if snapshot is None else dict(snapshot.get("summary", {})),
        },
    }


def _run_single_arm_capture(
    *,
    checkpoint_path: str,
    train_suite_path: str,
    phase2_summary_path: str,
    device: str | None,
    selector_trace_json: str,
    snapshot_json: str,
    actor_objective_mode_override: str | None,
    actor_objective_runtime_current_suite_name: str | None,
) -> dict[str, Any]:
    phase2_summary = assert_phase2_green(phase2_summary_path)
    checkpoint_path = str(Path(checkpoint_path).expanduser().resolve())
    train_suite_path = str(Path(train_suite_path).expanduser().resolve())
    device_obj = torch.device(str(device or _default_device()))
    suite = _load_required_suite(train_suite_path)

    started_at = time.perf_counter()
    load_model.cache_clear()
    model, config = load_model(checkpoint_path, device=device_obj, verbose=False)
    env_cfg = _build_audit_env_cfg(config["prior"]["environment"])
    prior = EnvironmentPrior(copy.deepcopy(env_cfg))
    prior._sample_batch_hypers = lambda batch_n: _clone_suite_h_list(suite)
    prior._sample_single_eval_pos = lambda n_samples_arg, rng_seed=None: int(
        PAIR2_TRAIN_UPDATE_SINGLE_EVAL_POS
    )

    optimizer_cfg = dict(config.get("optimizer", {}))
    profile_cfg = _resolve_train_profile(
        optimizer_cfg=optimizer_cfg,
        train_profile="phase2_shared_backbone_contract",
        batch_size=256,
        n_epochs=1,
        learning_rate=2e-4,
        target_kl=0.03,
    )
    if actor_objective_mode_override is not None:
        profile_cfg["actor_objective_mode"] = str(actor_objective_mode_override).strip().lower()

    algo, callback, vec_env = build_recurrent_ppo(
        model=model,
        env_prior=prior,
        device=str(device_obj),
        num_features=int(config["prior"]["num_features"]),
        n_envs=int(suite["batch_size"]),
        n_steps=int(PAIR2_TRAIN_UPDATE_NSAMPLES),
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
        strict_fixed_env_mode=True,
        env_rng_seeds=[int(v) for v in suite["env_seeds"]],
        rollout_rng_seeds=[int(v) for v in suite["rollout_seeds"]],
        deterministic_actor_sampling=True,
        deterministic_batch_plan=True,
        strict_native_rollout=True,
        actor_objective_runtime_current_suite_name=actor_objective_runtime_current_suite_name,
        reset_env_state_at_sep=bool(profile_cfg["reset_env_state_at_sep"]),
        separate_value_backbone=bool(profile_cfg["separate_value_backbone"]),
        restore_validation_policy_state=bool(profile_cfg["restore_validation_policy_state"]),
        restore_validation_policy_head_state=True,
        runtime_normalized_q_value_weight_override=profile_cfg[
            "runtime_normalized_q_value_weight_override"
        ],
        runtime_next_state_flow_matching_weight_override=profile_cfg[
            "runtime_next_state_flow_matching_weight_override"
        ],
        ent_coef=float(optimizer_cfg.get("ppo_ent_coef", 0.0)),
        vf_coef=float(profile_cfg["vf_coef"]),
        max_grad_norm=float(optimizer_cfg.get("ppo_max_grad_norm", 0.5)),
        target_kl=profile_cfg["target_kl"],
        verbose=0,
    )
    _bind_vec_env_to_fixed_suite(
        vec_env,
        suite=suite,
        single_eval_pos=int(PAIR2_TRAIN_UPDATE_SINGLE_EVAL_POS),
    )
    _make_deterministic_batch_plan(algo.rollout_buffer)
    _configure_capture(
        algo,
        snapshot_json=snapshot_json,
        selector_trace_json=selector_trace_json,
        target_outer_batch_idx=int(PAIR2_SNAPSHOT_TARGET_OUTER_BATCH_IDX),
        target_env_index=int(PAIR2_SNAPSHOT_TARGET_ENV_INDEX),
        target_objective_episode_index=int(PAIR2_SNAPSHOT_TARGET_OBJECTIVE_EPISODE_INDEX),
        target_objective_position_start=int(PAIR2_SNAPSHOT_TARGET_OBJECTIVE_POSITION_START),
        target_objective_position_end=int(PAIR2_SNAPSHOT_TARGET_OBJECTIVE_POSITION_END),
    )

    try:
        train_history = _audit_train_loop(
            algo,
            callback,
            vec_env,
            outer_epochs=1,
        )
    finally:
        vec_env.close()

    selector_trace = _load_optional_json(
        getattr(algo, "_rwkv_last_train_outer_batch_selector_trace_path", None)
    )
    if selector_trace is not None:
        selector_trace["source_path"] = getattr(
            algo,
            "_rwkv_last_train_outer_batch_selector_trace_path",
            None,
        )
    snapshot = _load_optional_json(
        getattr(algo, "_rwkv_last_train_outer_batch_snapshot_path", None)
    )
    if snapshot is not None:
        snapshot["source_path"] = getattr(
            algo,
            "_rwkv_last_train_outer_batch_snapshot_path",
            None,
        )

    return {
        "config": {
            "checkpoint_path": checkpoint_path,
            "train_suite_path": train_suite_path,
            "device": str(device_obj),
            "n_steps": int(PAIR2_TRAIN_UPDATE_NSAMPLES),
            "batch_size": int(profile_cfg["batch_size"]),
            "outer_epochs": 1,
            "single_eval_pos": int(PAIR2_TRAIN_UPDATE_SINGLE_EVAL_POS),
            "train_profile": str(profile_cfg["train_profile"]),
            "ppo_actor_baseline_mode": str(profile_cfg["actor_baseline_mode"]),
            "ppo_normalize_advantage": bool(profile_cfg["normalize_advantage"]),
            "ppo_actor_gae_space": str(profile_cfg["actor_gae_space"]),
            "ppo_vf_coef": float(profile_cfg["vf_coef"]),
            "ppo_reset_env_state_at_sep": bool(profile_cfg["reset_env_state_at_sep"]),
            "ppo_separate_value_backbone": bool(profile_cfg["separate_value_backbone"]),
            "ppo_restore_validation_policy_state": bool(
                profile_cfg["restore_validation_policy_state"]
            ),
            "ppo_restore_validation_policy_head_state": True,
            "strict_fixed_env_mode": True,
            "deterministic_actor_sampling": True,
            "deterministic_batch_plan": True,
            "strict_native_rollout": True,
            "actor_objective_mode_override": None
            if actor_objective_mode_override is None
            else str(actor_objective_mode_override),
            "actor_objective_runtime_current_suite_name": None
            if actor_objective_runtime_current_suite_name is None
            else str(actor_objective_runtime_current_suite_name),
            "selector_trace_json": str(Path(selector_trace_json).expanduser().resolve()),
            "snapshot_json": str(Path(snapshot_json).expanduser().resolve()),
        },
        "phase2_reference_contract": dict(phase2_summary["trusted_contract"]),
        "train_suite_fingerprint": str(suite.get("fingerprint", "")),
        "train_history": train_history,
        "runtime": {
            "elapsed_wall_time_sec": float(time.perf_counter() - started_at),
            "last_update_wall_time_sec": float(
                getattr(algo, "_rwkv_last_update_wall_time_sec", 0.0)
            ),
            "total_env_steps": int(PAIR2_TRAIN_UPDATE_NSAMPLES) * int(suite["batch_size"]),
        },
        "selector_trace": selector_trace,
        "snapshot": snapshot,
    }


def build_pair2_guarded_target_present_selector_trace_report(
    *,
    checkpoint_path: str = PAIR2_CHECKPOINT_PATH,
    train_suite_path: str = PAIR2_TRAIN_SUITE_PATH,
    device: str | None = None,
    phase2_summary_path: str = DEFAULT_PHASE2_SUMMARY,
    baseline_selector_trace_json: str = DEFAULT_BASELINE_SELECTOR_TRACE_JSON,
    baseline_snapshot_json: str = DEFAULT_BASELINE_SNAPSHOT_JSON,
    override_selector_trace_json: str = DEFAULT_OVERRIDE_SELECTOR_TRACE_JSON,
    override_snapshot_json: str = DEFAULT_OVERRIDE_SNAPSHOT_JSON,
) -> dict[str, Any]:
    preflight = _load_pair2_preflight()
    phase2_summary = assert_phase2_green(phase2_summary_path)
    target_selector = {
        "target_outer_batch_idx": int(PAIR2_SNAPSHOT_TARGET_OUTER_BATCH_IDX),
        "target_env_index": int(PAIR2_SNAPSHOT_TARGET_ENV_INDEX),
        "target_objective_episode_index": int(PAIR2_SNAPSHOT_TARGET_OBJECTIVE_EPISODE_INDEX),
        "target_objective_position_start": int(PAIR2_SNAPSHOT_TARGET_OBJECTIVE_POSITION_START),
        "target_objective_position_end": int(PAIR2_SNAPSHOT_TARGET_OBJECTIVE_POSITION_END),
    }
    baseline = _run_single_arm_capture(
        checkpoint_path=checkpoint_path,
        train_suite_path=train_suite_path,
        phase2_summary_path=phase2_summary_path,
        device=device,
        selector_trace_json=baseline_selector_trace_json,
        snapshot_json=baseline_snapshot_json,
        actor_objective_mode_override=None,
        actor_objective_runtime_current_suite_name=None,
    )
    override = _run_single_arm_capture(
        checkpoint_path=checkpoint_path,
        train_suite_path=train_suite_path,
        phase2_summary_path=phase2_summary_path,
        device=device,
        selector_trace_json=override_selector_trace_json,
        snapshot_json=override_snapshot_json,
        actor_objective_mode_override=PAIR2_BRANCH_SPECIFIC_MODE,
        actor_objective_runtime_current_suite_name="pair2",
    )

    baseline_arm = _build_arm_summary(
        mode_label="baseline",
        actor_objective_mode_override=None,
        actor_objective_runtime_current_suite_name=None,
        selector_trace=baseline["selector_trace"],
        snapshot=baseline["snapshot"],
        elapsed_wall_time_sec=float(baseline["runtime"]["elapsed_wall_time_sec"]),
        last_update_wall_time_sec=float(baseline["runtime"]["last_update_wall_time_sec"]),
        total_env_steps=int(baseline["runtime"]["total_env_steps"]),
        target_selector=target_selector,
    )
    override_arm = _build_arm_summary(
        mode_label="override",
        actor_objective_mode_override=PAIR2_BRANCH_SPECIFIC_MODE,
        actor_objective_runtime_current_suite_name="pair2",
        selector_trace=override["selector_trace"],
        snapshot=override["snapshot"],
        elapsed_wall_time_sec=float(override["runtime"]["elapsed_wall_time_sec"]),
        last_update_wall_time_sec=float(override["runtime"]["last_update_wall_time_sec"]),
        total_env_steps=int(override["runtime"]["total_env_steps"]),
        target_selector=target_selector,
    )

    baseline_rows = list(baseline_arm["selector_trace"]["rows"])
    override_rows = list(override_arm["selector_trace"]["rows"])
    baseline_capture = dict(baseline_arm["capture"])
    override_capture = dict(override_arm["capture"])

    return {
        "probe_entry": "phase3_pair2_guarded_target_present_selector_trace",
        "preflight": {
            "source_path": str(
                Path(
                    "/home/chen/RLPFN/artifacts/phase3_preflight_contract_check_pair2.json"
                ).expanduser().resolve()
            ),
            "preflight_passed": bool(preflight.get("preflight_passed", False)),
            "contract_args": dict(preflight.get("contract_args", {})),
            "allowed_diff": dict(dict(preflight.get("contract_diff", {})).get("allowed_diff", {})),
            "unexpected_diff": dict(dict(preflight.get("contract_diff", {})).get("unexpected_diff", {})),
        },
        "phase2_reference_contract": dict(phase2_summary["trusted_contract"]),
        "target_selector": dict(target_selector),
        "shared_contract_context": {
            "checkpoint_path": str(Path(checkpoint_path).expanduser().resolve()),
            "train_suite_path": str(Path(train_suite_path).expanduser().resolve()),
            "n_steps": int(PAIR2_TRAIN_UPDATE_NSAMPLES),
            "single_eval_pos": int(PAIR2_TRAIN_UPDATE_SINGLE_EVAL_POS),
            "strict_fixed_env_mode": True,
            "deterministic_actor_sampling": True,
            "deterministic_batch_plan": True,
            "strict_native_rollout": True,
            "ppo_restore_validation_policy_head_state": True,
            "ppo_reset_env_state_at_sep": True,
            "ppo_separate_value_backbone": False,
            "ppo_actor_baseline_mode": "learned",
            "ppo_normalize_advantage": False,
            "ppo_actor_gae_space": "normalized",
        },
        "baseline": baseline_arm,
        "override": override_arm,
        "conclusions": {
            "selector_trace_completed_in_both_arms": bool(
                baseline_arm["selector_trace"]["completed"]
                and override_arm["selector_trace"]["completed"]
            ),
            "selector_row_layout_identical_across_arms": bool(baseline_rows == override_rows),
            "snapshot_and_trace_consistent_in_both_arms": bool(
                baseline_capture["same_hit_relationship_preserved"]
                == baseline_capture["target_appeared"]
                and override_capture["same_hit_relationship_preserved"]
                == override_capture["target_appeared"]
            ),
            "target_present_in_baseline": bool(baseline_capture["target_appeared"]),
            "target_present_in_override": bool(override_capture["target_appeared"]),
            "target_present_in_neither_arm": bool(
                not baseline_capture["target_appeared"] and not override_capture["target_appeared"]
            ),
            "env12_enters_trusted_outer_batch": bool(
                baseline_arm["selector_trace"]["env_present_outer_batches"]
                or override_arm["selector_trace"]["env_present_outer_batches"]
            ),
            "objective_episode_present_for_env12": bool(
                baseline_arm["selector_trace"]["episode_present_outer_batches"]
                or override_arm["selector_trace"]["episode_present_outer_batches"]
            ),
            "requested_selector_episode_absent_despite_env12_presence": bool(
                (
                    baseline_arm["selector_trace"]["env_present_outer_batches"]
                    or override_arm["selector_trace"]["env_present_outer_batches"]
                )
                and not (
                    baseline_arm["selector_trace"]["episode_present_outer_batches"]
                    or override_arm["selector_trace"]["episode_present_outer_batches"]
                )
            ),
            "requested_selector_range_present_for_env12": bool(
                baseline_arm["selector_trace"]["requested_position_range_present_outer_batches"]
                or override_arm["selector_trace"]["requested_position_range_present_outer_batches"]
            ),
            "requested_selector_position_range_looks_stale": bool(
                not baseline_capture["target_appeared"]
                and not override_capture["target_appeared"]
                and (
                    baseline_arm["selector_trace"]["canonical_selector_requires_position_change"]
                    or override_arm["selector_trace"]["canonical_selector_requires_position_change"]
                )
            ),
            "selector_match_bug_suspected": bool(
                baseline_arm["selector_trace"]["selector_match_bug_suspected"]
                or override_arm["selector_trace"]["selector_match_bug_suspected"]
            ),
            "current_selector_definition_cannot_fire_under_trusted_contract": bool(
                not baseline_capture["target_appeared"]
                and not override_capture["target_appeared"]
                and (
                    (
                        baseline_arm["selector_trace"]["env_present_outer_batches"]
                        or override_arm["selector_trace"]["env_present_outer_batches"]
                    )
                    and not (
                        baseline_arm["selector_trace"]["episode_present_outer_batches"]
                        or override_arm["selector_trace"]["episode_present_outer_batches"]
                    )
                )
            ),
            "runtime_mode_changes_selector_membership": bool(
                baseline_capture["target_appeared"] != override_capture["target_appeared"]
                or baseline_rows != override_rows
            ),
        },
        "recommendation": {
            "next_focus": (
                "run_pair2_train_update_ab_compare_pack"
                if (
                    baseline_capture["target_appeared"]
                    and override_capture["target_appeared"]
                )
                else (
                "repair_selector_definition"
                if (
                    not baseline_capture["target_appeared"]
                    and not override_capture["target_appeared"]
                    and (
                        (
                            baseline_arm["selector_trace"]["env_present_outer_batches"]
                            or override_arm["selector_trace"]["env_present_outer_batches"]
                        )
                        and not (
                            baseline_arm["selector_trace"]["episode_present_outer_batches"]
                            or override_arm["selector_trace"]["episode_present_outer_batches"]
                        )
                    )
                )
                else (
                "repair_selector_definition"
                if (
                    not baseline_capture["target_appeared"]
                    and not override_capture["target_appeared"]
                    and (
                        baseline_arm["selector_trace"]["canonical_selector_requires_position_change"]
                        or override_arm["selector_trace"]["canonical_selector_requires_position_change"]
                    )
                )
                else (
                    "inspect_selector_match_bug"
                    if (
                        baseline_arm["selector_trace"]["selector_match_bug_suspected"]
                        or override_arm["selector_trace"]["selector_match_bug_suspected"]
                    )
                    else "treat_current_small_control_as_absent_from_trusted_update_scope"
                ))
                )
            ),
            "reason": (
                "This guarded trace separates three cases cleanly: exact target present, target absent because "
                "the current selector range is stale, or target absent because env/episode never enters the trusted "
                "train-side outer-batch scope."
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a guarded pair2 train-side target-present selector trace under the trusted update contract."
    )
    parser.add_argument("--checkpoint-path", type=str, default=PAIR2_CHECKPOINT_PATH)
    parser.add_argument("--train-suite-path", type=str, default=PAIR2_TRAIN_SUITE_PATH)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--phase2-summary-path", type=str, default=DEFAULT_PHASE2_SUMMARY)
    parser.add_argument(
        "--baseline-selector-trace-json",
        type=str,
        default=DEFAULT_BASELINE_SELECTOR_TRACE_JSON,
    )
    parser.add_argument("--baseline-snapshot-json", type=str, default=DEFAULT_BASELINE_SNAPSHOT_JSON)
    parser.add_argument(
        "--override-selector-trace-json",
        type=str,
        default=DEFAULT_OVERRIDE_SELECTOR_TRACE_JSON,
    )
    parser.add_argument("--override-snapshot-json", type=str, default=DEFAULT_OVERRIDE_SNAPSHOT_JSON)
    parser.add_argument("--output-json", type=str, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output_json).expanduser().resolve()
    if output_path.exists() and not bool(args.overwrite):
        print(output_path.read_text())
        return 0

    report = build_pair2_guarded_target_present_selector_trace_report(
        checkpoint_path=args.checkpoint_path,
        train_suite_path=args.train_suite_path,
        device=args.device,
        phase2_summary_path=args.phase2_summary_path,
        baseline_selector_trace_json=args.baseline_selector_trace_json,
        baseline_snapshot_json=args.baseline_snapshot_json,
        override_selector_trace_json=args.override_selector_trace_json,
        override_snapshot_json=args.override_snapshot_json,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
