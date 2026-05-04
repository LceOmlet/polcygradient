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
from ticl.analysis.phase3_pair1_train_update_ab_compare_pack import (
    PAIR1_CHECKPOINT_PATH,
    PAIR1_PREFLIGHT_JSON,
    PAIR1_TRAIN_SUITE_PATH,
    _load_pair1_preflight,
)
from ticl.model_builder import load_model
from ticl.priors.environment_prior import EnvironmentPrior
from ticl.sb3_recurrent_ppo import build_recurrent_ppo


DEFAULT_OUTPUT_JSON = "/home/chen/RLPFN/artifacts/phase3_pair1_same_contract_train_drift_probe.json"
PRIMARY_COLLECT_SNAPSHOT_JSON = (
    "/home/chen/RLPFN/artifacts/phase3_pair1_same_contract_train_drift_primary_collect_snapshot.json"
)
PRIMARY_TRAIN_SNAPSHOT_JSON = (
    "/home/chen/RLPFN/artifacts/phase3_pair1_same_contract_train_drift_primary_train_snapshot.json"
)
REPEAT_COLLECT_SNAPSHOT_JSON = (
    "/home/chen/RLPFN/artifacts/phase3_pair1_same_contract_train_drift_repeat_collect_snapshot.json"
)
REPEAT_TRAIN_SNAPSHOT_JSON = (
    "/home/chen/RLPFN/artifacts/phase3_pair1_same_contract_train_drift_repeat_train_snapshot.json"
)

PHASE2_EQ_N_STEPS = 256
PHASE2_EQ_BATCH_SIZE = 256
PHASE2_EQ_OUTER_EPOCHS = 1
PHASE2_EQ_SINGLE_EVAL_POS = 64
FIRST_TRAIN_OUTER_BATCH_IDX = 1

_COLLECT_DIGEST_KEYS = (
    "observations",
    "actions",
    "rewards",
    "values",
    "log_probs",
    "returns",
    "advantages",
    "actor_advantages",
    "objective_masks",
    "episode_starts",
    "rollout_return_means",
    "rollout_return_stds",
    "position_env_indices",
    "position_step_indices",
    "position_objective_episode_indices",
    "position_objective_episode_positions",
    "position_objective_global_positions",
)
_TRAIN_EXACT_KEYS = (
    "flat_batch_indices",
    "flat_env_indices",
    "flat_step_indices",
    "flat_objective_episode_indices",
    "flat_objective_episode_positions",
    "flat_objective_global_positions",
    "objective_mask",
    "episode_starts",
    "seq_start_indices",
    "seq_lengths",
)
_TRAIN_FLOAT_KEYS = (
    "actions",
    "old_values",
    "old_log_prob",
    "returns",
    "pre_normalization_actor_advantages",
    "post_normalization_advantages",
)


def _configure_probe_capture(
    algo,
    *,
    collect_snapshot_json: str,
    train_snapshot_json: str,
) -> None:
    algo._rwkv_rollout_collect_snapshot_path = str(Path(collect_snapshot_json).expanduser().resolve())
    algo._rwkv_rollout_collect_snapshot_written = False
    algo._rwkv_last_rollout_collect_snapshot_path = None
    algo._rwkv_train_outer_batch_snapshot_path = str(Path(train_snapshot_json).expanduser().resolve())
    algo._rwkv_train_outer_batch_snapshot_written = False
    algo._rwkv_last_train_outer_batch_snapshot_path = None
    algo._rwkv_train_outer_batch_snapshot_target_outer_batch_idx = int(FIRST_TRAIN_OUTER_BATCH_IDX)
    algo._rwkv_train_outer_batch_snapshot_target_env_index = None
    algo._rwkv_train_outer_batch_snapshot_target_objective_episode_index = None
    algo._rwkv_train_outer_batch_snapshot_target_objective_position_start = None
    algo._rwkv_train_outer_batch_snapshot_target_objective_position_end = None


def _load_json(path: str | None) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        return None
    return json.loads(resolved.read_text())


def _compare_collect_snapshots(
    left: dict[str, Any],
    right: dict[str, Any],
) -> dict[str, Any]:
    left_digests = dict(left.get("array_digests", {}))
    right_digests = dict(right.get("array_digests", {}))
    digest_rows = []
    differing_keys: list[str] = []
    for key in _COLLECT_DIGEST_KEYS:
        left_entry = dict(left_digests.get(key, {}))
        right_entry = dict(right_digests.get(key, {}))
        same = left_entry.get("sha256") == right_entry.get("sha256")
        if not same:
            differing_keys.append(str(key))
        digest_rows.append(
            {
                "key": str(key),
                "same_digest": bool(same),
                "left_sha256": left_entry.get("sha256"),
                "right_sha256": right_entry.get("sha256"),
            }
        )
    return {
        "digest_rows": digest_rows,
        "differing_keys": differing_keys,
        "collect_identical": bool(not differing_keys),
        "left_summary": dict(left.get("summary", {})),
        "right_summary": dict(right.get("summary", {})),
    }


def _compare_train_snapshots(
    left: dict[str, Any],
    right: dict[str, Any],
) -> dict[str, Any]:
    exact_rows = []
    differing_exact_keys: list[str] = []
    for key in _TRAIN_EXACT_KEYS:
        left_arr = np.asarray(left.get(key, []))
        right_arr = np.asarray(right.get(key, []))
        same = left_arr.shape == right_arr.shape and np.array_equal(left_arr, right_arr)
        if not same:
            differing_exact_keys.append(str(key))
        exact_rows.append(
            {
                "key": str(key),
                "same": bool(same),
                "left_shape": [int(v) for v in left_arr.shape],
                "right_shape": [int(v) for v in right_arr.shape],
            }
        )

    float_rows = []
    differing_float_keys: list[str] = []
    max_abs_delta = 0.0
    for key in _TRAIN_FLOAT_KEYS:
        left_arr = np.asarray(left.get(key, []), dtype=np.float64)
        right_arr = np.asarray(right.get(key, []), dtype=np.float64)
        if left_arr.shape != right_arr.shape:
            same = False
            field_max_abs_delta = float("inf")
        else:
            diff = np.abs(left_arr - right_arr)
            field_max_abs_delta = 0.0 if diff.size <= 0 else float(np.max(diff))
            same = bool(field_max_abs_delta <= 1e-12)
        if not same:
            differing_float_keys.append(str(key))
        if np.isfinite(field_max_abs_delta):
            max_abs_delta = max(max_abs_delta, field_max_abs_delta)
        float_rows.append(
            {
                "key": str(key),
                "same": bool(same),
                "left_shape": [int(v) for v in left_arr.shape],
                "right_shape": [int(v) for v in right_arr.shape],
                "max_abs_delta": field_max_abs_delta,
            }
        )

    return {
        "exact_rows": exact_rows,
        "float_rows": float_rows,
        "differing_exact_keys": differing_exact_keys,
        "differing_float_keys": differing_float_keys,
        "train_input_identical": bool(not differing_exact_keys and not differing_float_keys),
        "max_abs_float_delta": float(max_abs_delta),
        "left_summary": dict(left.get("summary", {})),
        "right_summary": dict(right.get("summary", {})),
    }


def _run_single_capture(
    *,
    checkpoint_path: str,
    train_suite_path: str,
    phase2_summary_path: str,
    device: str | None,
    collect_snapshot_json: str,
    train_snapshot_json: str,
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
        strict_native_rollout=True,
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
    _configure_probe_capture(
        algo,
        collect_snapshot_json=collect_snapshot_json,
        train_snapshot_json=train_snapshot_json,
    )

    try:
        algo.learn(
            total_timesteps=int(PHASE2_EQ_N_STEPS) * int(PHASE2_EQ_OUTER_EPOCHS),
            callback=callback,
        )
    finally:
        vec_env.close()

    collect_snapshot = _load_json(getattr(algo, "_rwkv_last_rollout_collect_snapshot_path", None))
    train_snapshot = _load_json(getattr(algo, "_rwkv_last_train_outer_batch_snapshot_path", None))
    if collect_snapshot is None:
        raise RuntimeError("Expected rollout collect snapshot to be written.")
    if train_snapshot is None:
        raise RuntimeError("Expected first train outer-batch snapshot to be written.")
    return {
        "phase2_reference_contract": dict(phase2_summary["trusted_contract"]),
        "config": {
            "checkpoint_path": checkpoint_path,
            "train_suite_path": train_suite_path,
            "device": str(device_obj),
            "n_steps": int(PHASE2_EQ_N_STEPS),
            "batch_size": int(PHASE2_EQ_BATCH_SIZE),
            "outer_epochs": int(PHASE2_EQ_OUTER_EPOCHS),
            "single_eval_pos": int(PHASE2_EQ_SINGLE_EVAL_POS),
            "train_profile": str(profile_cfg["train_profile"]),
            "strict_fixed_env_mode": True,
            "deterministic_actor_sampling": True,
            "deterministic_batch_plan": True,
            "strict_native_rollout": True,
            "actor_objective_mode": str(profile_cfg["actor_objective_mode"]),
            "actor_baseline_mode": str(profile_cfg["actor_baseline_mode"]),
            "actor_gae_space": str(profile_cfg["actor_gae_space"]),
            "normalize_advantage": bool(profile_cfg["normalize_advantage"]),
        },
        "runtime": {
            "elapsed_wall_time_sec": float(time.perf_counter() - started_at),
            "rollout_wall_time_sec": float(getattr(algo, "_rwkv_last_rollout_wall_time_sec", 0.0)),
            "update_wall_time_sec": float(getattr(algo, "_rwkv_last_update_wall_time_sec", 0.0)),
        },
        "collect_snapshot_path": getattr(algo, "_rwkv_last_rollout_collect_snapshot_path", None),
        "train_snapshot_path": getattr(algo, "_rwkv_last_train_outer_batch_snapshot_path", None),
        "collect_snapshot": collect_snapshot,
        "train_snapshot": train_snapshot,
    }


def build_pair1_same_contract_train_drift_probe(
    *,
    checkpoint_path: str = PAIR1_CHECKPOINT_PATH,
    train_suite_path: str = PAIR1_TRAIN_SUITE_PATH,
    phase2_summary_path: str = DEFAULT_PHASE2_SUMMARY,
    device: str | None = None,
) -> dict[str, Any]:
    preflight = _load_pair1_preflight()
    primary = _run_single_capture(
        checkpoint_path=checkpoint_path,
        train_suite_path=train_suite_path,
        phase2_summary_path=phase2_summary_path,
        device=device,
        collect_snapshot_json=PRIMARY_COLLECT_SNAPSHOT_JSON,
        train_snapshot_json=PRIMARY_TRAIN_SNAPSHOT_JSON,
    )
    repeat = _run_single_capture(
        checkpoint_path=checkpoint_path,
        train_suite_path=train_suite_path,
        phase2_summary_path=phase2_summary_path,
        device=device,
        collect_snapshot_json=REPEAT_COLLECT_SNAPSHOT_JSON,
        train_snapshot_json=REPEAT_TRAIN_SNAPSHOT_JSON,
    )

    collect_compare = _compare_collect_snapshots(
        primary["collect_snapshot"],
        repeat["collect_snapshot"],
    )
    train_compare = _compare_train_snapshots(
        primary["train_snapshot"],
        repeat["train_snapshot"],
    )
    if not collect_compare["collect_identical"]:
        drift_origin = "collect"
    elif not train_compare["train_input_identical"]:
        drift_origin = "train_input_transport"
    else:
        drift_origin = "update_or_later_batches"

    return {
        "probe_entry": "phase3_pair1_same_contract_train_drift_probe",
        "preflight": {
            "source_path": PAIR1_PREFLIGHT_JSON,
            "preflight_passed": bool(preflight.get("preflight_passed", False)),
            "contract_args": dict(preflight.get("contract_args", {})),
            "allowed_diff": dict(dict(preflight.get("contract_diff", {})).get("allowed_diff", {})),
            "unexpected_diff": dict(dict(preflight.get("contract_diff", {})).get("unexpected_diff", {})),
        },
        "config": dict(primary["config"]),
        "phase2_reference_contract": dict(primary["phase2_reference_contract"]),
        "primary": {
            "runtime": dict(primary["runtime"]),
            "collect_snapshot_path": primary["collect_snapshot_path"],
            "train_snapshot_path": primary["train_snapshot_path"],
            "collect_summary": dict(primary["collect_snapshot"].get("summary", {})),
            "train_summary": dict(primary["train_snapshot"].get("summary", {})),
        },
        "repeat": {
            "runtime": dict(repeat["runtime"]),
            "collect_snapshot_path": repeat["collect_snapshot_path"],
            "train_snapshot_path": repeat["train_snapshot_path"],
            "collect_summary": dict(repeat["collect_snapshot"].get("summary", {})),
            "train_summary": dict(repeat["train_snapshot"].get("summary", {})),
        },
        "comparison": {
            "collect": collect_compare,
            "train": train_compare,
        },
        "conclusions": {
            "collect_identical": bool(collect_compare["collect_identical"]),
            "first_train_batch_input_identical": bool(train_compare["train_input_identical"]),
            "drift_origin": str(drift_origin),
            "nondeterminism_starts_in_collect": bool(drift_origin == "collect"),
            "nondeterminism_starts_after_collect": bool(drift_origin != "collect"),
        },
        "recommendation": {
            "reason": (
                "Repeat the same pair1 shared-backbone many-env train contract twice and compare the first "
                "rollout collect snapshot plus the first train outer-batch input to localize whether drift "
                "begins in collect or only after training starts."
            ),
            "if_collect_identical_then_focus_next_on": (
                "optimizer/update kernel path or later outer-batch aggregation"
            ),
            "if_collect_diverges_then_focus_next_on": (
                "same-contract rollout collect path under strict native mode"
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only pair1 same-contract train drift probe that compares first rollout "
            "collect output and first train outer-batch input across two baseline runs."
        )
    )
    parser.add_argument("--checkpoint-path", type=str, default=PAIR1_CHECKPOINT_PATH)
    parser.add_argument("--train-suite-path", type=str, default=PAIR1_TRAIN_SUITE_PATH)
    parser.add_argument("--phase2-summary-path", type=str, default=DEFAULT_PHASE2_SUMMARY)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-json", type=str, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output_json).expanduser().resolve()
    if output_path.exists() and not bool(args.overwrite):
        print(output_path.read_text())
        return 0

    report = build_pair1_same_contract_train_drift_probe(
        checkpoint_path=args.checkpoint_path,
        train_suite_path=args.train_suite_path,
        phase2_summary_path=args.phase2_summary_path,
        device=args.device,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
