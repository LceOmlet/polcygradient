import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ticl.analysis.critic_free_single_env_audit import (
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
    PAIR2_CHECKPOINT_PATH,
    PAIR2_TRAIN_SUITE_PATH,
)
from ticl.model_builder import load_model
from ticl.priors.environment_prior import EnvironmentPrior
from ticl.sb3_recurrent_ppo import build_recurrent_ppo


DEFAULT_OUTPUT_JSON = (
    "/home/chen/RLPFN/artifacts/phase3_pair2_phase2_scale_target_present_capture.json"
)
DEFAULT_SELECTOR_TRACE_JSON = (
    "/home/chen/RLPFN/artifacts/phase3_pair2_phase2_scale_target_present_selector_trace.json"
)
DEFAULT_SNAPSHOT_JSON = (
    "/home/chen/RLPFN/artifacts/phase3_pair2_phase2_scale_target_present_snapshot.json"
)

PHASE2_EQ_N_STEPS = 256
PHASE2_EQ_BATCH_SIZE = 256
PHASE2_EQ_OUTER_EPOCHS = 2
PHASE2_EQ_SINGLE_EVAL_POS = 64

TARGET_OUTER_BATCH_IDX = None
TARGET_ENV_INDEX = 12
TARGET_OBJECTIVE_EPISODE_INDEX = 10
TARGET_OBJECTIVE_POSITION_START = 10
TARGET_OBJECTIVE_POSITION_END = 13


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
    algo._rwkv_train_outer_batch_selector_trace_path = str(Path(selector_trace_json).expanduser().resolve())
    algo._rwkv_train_outer_batch_selector_trace_rows = []
    algo._rwkv_train_outer_batch_selector_trace_completed = False
    algo._rwkv_last_train_outer_batch_selector_trace_path = None
    algo._rwkv_train_outer_batch_snapshot_target_outer_batch_idx = (
        None if target_outer_batch_idx is None else int(target_outer_batch_idx)
    )
    algo._rwkv_train_outer_batch_snapshot_target_env_index = int(target_env_index)
    algo._rwkv_train_outer_batch_snapshot_target_objective_episode_index = int(target_objective_episode_index)
    algo._rwkv_train_outer_batch_snapshot_target_objective_position_start = int(target_objective_position_start)
    algo._rwkv_train_outer_batch_snapshot_target_objective_position_end = int(target_objective_position_end)


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
    return {
        "selector_trace_written": bool(selector_trace is not None),
        "snapshot_written": bool(snapshot is not None),
        "target_appeared": bool(trace_summary and trace_summary.get("target_appeared", False)),
        "first_match": None if not trace_summary else trace_summary.get("first_match"),
        "snapshot_target_match_count": None if snapshot is None else int(snapshot.get("snapshot_target_match_count", 0)),
        "snapshot_outer_batch_idx": None if snapshot_summary is None else int(snapshot_summary["outer_batch_idx"]),
        "snapshot_objective_total": None if snapshot_summary is None else int(snapshot_summary["objective_total"]),
        "snapshot_target_selector": target_selector,
        "same_hit_relationship_preserved": bool(
            trace_summary
            and trace_summary.get("target_appeared", False)
            and snapshot is not None
            and int(snapshot.get("snapshot_target_match_count", 0)) == 4
        ),
    }


def build_pair2_phase2_scale_target_present_capture(
    *,
    checkpoint_path: str = PAIR2_CHECKPOINT_PATH,
    train_suite_path: str = PAIR2_TRAIN_SUITE_PATH,
    device: str | None = None,
    phase2_summary_path: str = DEFAULT_PHASE2_SUMMARY,
    output_json: str = DEFAULT_OUTPUT_JSON,
    selector_trace_json: str = DEFAULT_SELECTOR_TRACE_JSON,
    snapshot_json: str = DEFAULT_SNAPSHOT_JSON,
    target_outer_batch_idx: int | None = TARGET_OUTER_BATCH_IDX,
    target_env_index: int = TARGET_ENV_INDEX,
    target_objective_episode_index: int = TARGET_OBJECTIVE_EPISODE_INDEX,
    target_objective_position_start: int = TARGET_OBJECTIVE_POSITION_START,
    target_objective_position_end: int = TARGET_OBJECTIVE_POSITION_END,
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
    prior._sample_single_eval_pos = lambda n_samples_arg, rng_seed=None: int(PHASE2_EQ_SINGLE_EVAL_POS)

    optimizer_cfg = dict(config.get("optimizer", {}))
    profile_cfg = _resolve_train_profile(
        optimizer_cfg=optimizer_cfg,
        train_profile="phase2_shared_backbone_contract",
        batch_size=int(PHASE2_EQ_BATCH_SIZE),
        n_epochs=1,
        learning_rate=0.0002,
        target_kl=0.03,
    )
    algo, callback, vec_env = build_recurrent_ppo(
        model=model,
        env_prior=prior,
        device=str(device_obj),
        num_features=int(config["prior"]["num_features"]),
        n_envs=int(suite["batch_size"]),
        n_steps=int(PHASE2_EQ_N_STEPS),
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
        actor_objective_runtime_current_suite_name=None,
        reset_env_state_at_sep=bool(profile_cfg["reset_env_state_at_sep"]),
        separate_value_backbone=bool(profile_cfg["separate_value_backbone"]),
        restore_validation_policy_state=bool(profile_cfg["restore_validation_policy_state"]),
        runtime_normalized_q_value_weight_override=profile_cfg["runtime_normalized_q_value_weight_override"],
        runtime_next_state_flow_matching_weight_override=profile_cfg["runtime_next_state_flow_matching_weight_override"],
        ent_coef=float(optimizer_cfg.get("ppo_ent_coef", 0.0)),
        vf_coef=float(profile_cfg["vf_coef"]),
        max_grad_norm=float(optimizer_cfg.get("ppo_max_grad_norm", 0.5)),
        target_kl=profile_cfg["target_kl"],
        verbose=0,
    )
    _bind_vec_env_to_fixed_suite(vec_env, suite=suite, single_eval_pos=int(PHASE2_EQ_SINGLE_EVAL_POS))
    _make_deterministic_batch_plan(algo.rollout_buffer)
    _configure_capture(
        algo,
        snapshot_json=snapshot_json,
        selector_trace_json=selector_trace_json,
        target_outer_batch_idx=target_outer_batch_idx,
        target_env_index=target_env_index,
        target_objective_episode_index=target_objective_episode_index,
        target_objective_position_start=target_objective_position_start,
        target_objective_position_end=target_objective_position_end,
    )

    try:
        algo.learn(
            total_timesteps=int(PHASE2_EQ_N_STEPS) * int(PHASE2_EQ_OUTER_EPOCHS),
            callback=callback,
        )
    finally:
        vec_env.close()

    selector_trace = _load_optional_json(getattr(algo, "_rwkv_last_train_outer_batch_selector_trace_path", None))
    snapshot = _load_optional_json(getattr(algo, "_rwkv_last_train_outer_batch_snapshot_path", None))
    capture_summary = _build_capture_summary(selector_trace=selector_trace, snapshot=snapshot)

    report = {
        "probe_entry": "phase3_pair2_phase2_scale_target_present_capture",
        "config": {
            "checkpoint_path": checkpoint_path,
            "train_suite_path": train_suite_path,
            "device": str(device_obj),
            "n_steps": int(PHASE2_EQ_N_STEPS),
            "batch_size": int(PHASE2_EQ_BATCH_SIZE),
            "outer_epochs": int(PHASE2_EQ_OUTER_EPOCHS),
            "single_eval_pos": int(PHASE2_EQ_SINGLE_EVAL_POS),
            "train_profile": str(profile_cfg["train_profile"]),
            "ppo_actor_baseline_mode": str(profile_cfg["actor_baseline_mode"]),
            "ppo_normalize_advantage": bool(profile_cfg["normalize_advantage"]),
            "ppo_actor_gae_space": str(profile_cfg["actor_gae_space"]),
            "ppo_vf_coef": float(profile_cfg["vf_coef"]),
            "ppo_reset_env_state_at_sep": bool(profile_cfg["reset_env_state_at_sep"]),
            "ppo_separate_value_backbone": bool(profile_cfg["separate_value_backbone"]),
            "ppo_restore_validation_policy_state": bool(profile_cfg["restore_validation_policy_state"]),
            "runtime_normalized_q_value_weight_override": (
                None
                if profile_cfg["runtime_normalized_q_value_weight_override"] is None
                else float(profile_cfg["runtime_normalized_q_value_weight_override"])
            ),
            "runtime_next_state_flow_matching_weight_override": (
                None
                if profile_cfg["runtime_next_state_flow_matching_weight_override"] is None
                else float(profile_cfg["runtime_next_state_flow_matching_weight_override"])
            ),
            "strict_fixed_env_mode": True,
            "deterministic_actor_sampling": True,
            "deterministic_batch_plan": True,
            "target_outer_batch_idx": None if target_outer_batch_idx is None else int(target_outer_batch_idx),
            "target_env_index": int(target_env_index),
            "target_objective_episode_index": int(target_objective_episode_index),
            "target_objective_position_start": int(target_objective_position_start),
            "target_objective_position_end": int(target_objective_position_end),
            "selector_trace_json": str(Path(selector_trace_json).expanduser().resolve()),
            "snapshot_json": str(Path(snapshot_json).expanduser().resolve()),
            "output_json": str(Path(output_json).expanduser().resolve()),
        },
        "phase2_reference_contract": dict(phase2_summary["trusted_contract"]),
        "runtime": {
            "elapsed_wall_time_sec": float(time.perf_counter() - started_at),
            "last_update_wall_time_sec": float(getattr(algo, "_rwkv_last_update_wall_time_sec", 0.0)),
        },
        "capture": capture_summary,
        "selector_trace_path": getattr(algo, "_rwkv_last_train_outer_batch_selector_trace_path", None),
        "snapshot_path": getattr(algo, "_rwkv_last_train_outer_batch_snapshot_path", None),
    }
    Path(output_json).expanduser().resolve().write_text(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a Phase-2-scale strict pair2 target-present train-side capture."
    )
    parser.add_argument("--checkpoint-path", type=str, default=PAIR2_CHECKPOINT_PATH)
    parser.add_argument("--train-suite-path", type=str, default=PAIR2_TRAIN_SUITE_PATH)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--phase2-summary-path", type=str, default=DEFAULT_PHASE2_SUMMARY)
    parser.add_argument("--selector-trace-json", type=str, default=DEFAULT_SELECTOR_TRACE_JSON)
    parser.add_argument("--snapshot-json", type=str, default=DEFAULT_SNAPSHOT_JSON)
    parser.add_argument("--output-json", type=str, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--target-outer-batch-idx", type=int, default=TARGET_OUTER_BATCH_IDX)
    parser.add_argument("--target-env-index", type=int, default=TARGET_ENV_INDEX)
    parser.add_argument("--target-objective-episode-index", type=int, default=TARGET_OBJECTIVE_EPISODE_INDEX)
    parser.add_argument("--target-objective-position-start", type=int, default=TARGET_OBJECTIVE_POSITION_START)
    parser.add_argument("--target-objective-position-end", type=int, default=TARGET_OBJECTIVE_POSITION_END)
    args = parser.parse_args()

    report = build_pair2_phase2_scale_target_present_capture(
        checkpoint_path=args.checkpoint_path,
        train_suite_path=args.train_suite_path,
        device=args.device,
        phase2_summary_path=args.phase2_summary_path,
        output_json=args.output_json,
        selector_trace_json=args.selector_trace_json,
        snapshot_json=args.snapshot_json,
        target_outer_batch_idx=args.target_outer_batch_idx,
        target_env_index=args.target_env_index,
        target_objective_episode_index=args.target_objective_episode_index,
        target_objective_position_start=args.target_objective_position_start,
        target_objective_position_end=args.target_objective_position_end,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
