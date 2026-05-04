import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ticl.analysis.critic_free_single_env_audit import _build_audit_env_cfg, _make_deterministic_batch_plan
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
from ticl.sb3_recurrent_ppo import _build_objective_position_metadata, build_recurrent_ppo


DEFAULT_OUTPUT_JSON = (
    "/home/chen/RLPFN/artifacts/phase3_pair1_strict_collect_boundary_drift_source_probe.json"
)

PHASE2_EQ_N_STEPS = 256
PHASE2_EQ_BATCH_SIZE = 256
PHASE2_EQ_SINGLE_EVAL_POS = 64


def _first_diff_step_float(
    left: np.ndarray,
    right: np.ndarray,
    *,
    atol: float = 1e-8,
) -> dict[str, Any]:
    left_arr = np.asarray(left, dtype=np.float64)
    right_arr = np.asarray(right, dtype=np.float64)
    if left_arr.shape != right_arr.shape:
        raise ValueError(f"Shape mismatch: {left_arr.shape!r} vs {right_arr.shape!r}")
    if left_arr.ndim == 0:
        max_abs_delta = float(abs(left_arr.item() - right_arr.item()))
        return {
            "identical": bool(max_abs_delta <= atol),
            "first_diff_step": None if max_abs_delta <= atol else 0,
            "max_abs_delta": float(max_abs_delta),
        }
    trailing_shape = left_arr.shape[1:]
    left_flat = left_arr.reshape((left_arr.shape[0], -1))
    right_flat = right_arr.reshape((right_arr.shape[0], -1))
    step_max_abs = np.max(np.abs(left_flat - right_flat), axis=1)
    diff_steps = np.flatnonzero(step_max_abs > float(atol))
    first_step = None if diff_steps.size <= 0 else int(diff_steps[0])
    first_env_index = None
    first_slot_index = None
    if first_step is not None:
        diff_mask = np.abs(left_flat[first_step] - right_flat[first_step]) > float(atol)
        first_linear_index = int(np.flatnonzero(diff_mask)[0])
        if len(trailing_shape) >= 1:
            unraveled = np.unravel_index(first_linear_index, trailing_shape)
            first_env_index = int(unraveled[0])
            first_slot_index = None if len(unraveled) <= 1 else int(unraveled[1])
    return {
        "identical": bool(first_step is None),
        "first_diff_step": first_step,
        "first_diff_env_index": first_env_index,
        "first_diff_slot_index": first_slot_index,
        "max_abs_delta": float(step_max_abs.max(initial=0.0)),
    }


def _first_diff_step_exact(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    left_arr = np.asarray(left)
    right_arr = np.asarray(right)
    if left_arr.shape != right_arr.shape:
        raise ValueError(f"Shape mismatch: {left_arr.shape!r} vs {right_arr.shape!r}")
    if left_arr.ndim == 0:
        same = bool(left_arr.item() == right_arr.item())
        return {
            "identical": same,
            "first_diff_step": None if same else 0,
        }
    left_flat = left_arr.reshape((left_arr.shape[0], -1))
    right_flat = right_arr.reshape((right_arr.shape[0], -1))
    diff_steps = np.flatnonzero(np.any(left_flat != right_flat, axis=1))
    first_step = None if diff_steps.size <= 0 else int(diff_steps[0])
    return {
        "identical": bool(first_step is None),
        "first_diff_step": first_step,
    }


def _capture_one_collect(
    *,
    checkpoint_path: str,
    train_suite_path: str,
    phase2_summary_path: str,
    device: str | None,
) -> dict[str, Any]:
    assert_phase2_green(phase2_summary_path)
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
    algo.ep_info_buffer = []
    algo.ep_success_buffer = []
    algo.policy.reset_rollout_cache(vec_env.num_envs)
    algo._last_obs = vec_env.reset()
    algo._last_episode_starts = np.ones((vec_env.num_envs,), dtype=bool)
    algo._last_lstm_states = algo.policy._dummy_states(vec_env.num_envs)
    callback.init_callback(algo)
    algo._update_current_progress_remaining(0, int(PHASE2_EQ_N_STEPS))

    captured_steps: list[dict[str, np.ndarray]] = []
    original_rollout = algo._rwkv_env_prior._rollout_family_group_vectorized_with_policy

    def _wrapped_rollout(*args, **kwargs):
        sink = kwargs.get("_policy_ppo_step_sink", None)
        if sink is None:
            raise RuntimeError("Expected _policy_ppo_step_sink in PPO rollout.")

        def _capture_sink(payload):
            captured_steps.append(
                {
                    "obs": payload["obs"].detach().cpu().numpy().astype(np.float32, copy=False),
                    "action": payload["action"].detach().cpu().numpy().astype(np.float32, copy=False),
                    "reward": payload["reward"].detach().cpu().numpy().astype(np.float32, copy=False),
                    "values": payload["values"].detach().cpu().numpy().astype(np.float32, copy=False),
                    "log_probs": payload["log_probs"].detach().cpu().numpy().astype(np.float32, copy=False),
                    "episode_starts": payload["episode_starts"].detach().cpu().numpy().astype(np.float32, copy=False),
                    "dones": payload["dones"].detach().cpu().numpy().astype(np.bool_, copy=False),
                }
            )
            return sink(payload)

        kwargs["_policy_ppo_step_sink"] = _capture_sink
        return original_rollout(*args, **kwargs)

    algo._rwkv_env_prior._rollout_family_group_vectorized_with_policy = _wrapped_rollout
    try:
        ok = algo.collect_rollouts(vec_env, callback, algo.rollout_buffer, n_rollout_steps=algo.n_steps)
        if not ok:
            raise RuntimeError("collect_rollouts returned False")
    finally:
        algo._rwkv_env_prior._rollout_family_group_vectorized_with_policy = original_rollout
        vec_env.close()

    step_obs = np.stack([row["obs"] for row in captured_steps], axis=0)
    step_actions = np.stack([row["action"] for row in captured_steps], axis=0)
    step_rewards = np.stack([row["reward"] for row in captured_steps], axis=0)
    step_values = np.stack([row["values"] for row in captured_steps], axis=0)
    step_log_probs = np.stack([row["log_probs"] for row in captured_steps], axis=0)
    step_episode_starts = np.stack([row["episode_starts"] for row in captured_steps], axis=0)
    step_dones = np.stack([row["dones"] for row in captured_steps], axis=0)
    objective_masks = np.zeros_like(step_episode_starts, dtype=np.float32)
    objective_masks[int(PHASE2_EQ_SINGLE_EVAL_POS) :, :] = 1.0
    position_metadata = _build_objective_position_metadata(step_episode_starts, objective_masks)

    return {
        "config": {
            "checkpoint_path": checkpoint_path,
            "train_suite_path": train_suite_path,
            "device": str(device_obj),
            "n_steps": int(PHASE2_EQ_N_STEPS),
            "batch_size": int(PHASE2_EQ_BATCH_SIZE),
            "single_eval_pos": int(PHASE2_EQ_SINGLE_EVAL_POS),
            "strict_fixed_env_mode": True,
            "deterministic_actor_sampling": True,
            "deterministic_batch_plan": True,
            "strict_native_rollout": True,
            "actor_objective_mode": str(profile_cfg["actor_objective_mode"]),
        },
        "runtime": {
            "elapsed_wall_time_sec": float(time.perf_counter() - started_at),
            "rollout_wall_time_sec": float(getattr(algo, "_rwkv_last_rollout_wall_time_sec", 0.0)),
        },
        "step_arrays": {
            "obs": step_obs,
            "action": step_actions,
            "reward": step_rewards,
            "values": step_values,
            "log_probs": step_log_probs,
            "episode_starts": step_episode_starts,
            "dones": step_dones.astype(np.int64, copy=False),
            "objective_masks": objective_masks.astype(np.int64, copy=False),
            "objective_episode_indices": np.asarray(
                position_metadata["objective_episode_indices"],
                dtype=np.int64,
            ),
            "objective_episode_positions": np.asarray(
                position_metadata["objective_episode_positions"],
                dtype=np.int64,
            ),
        },
        "within_run_contract": {
            "episode_starts_follow_previous_dones": bool(
                np.array_equal(step_episode_starts[1:] > 1e-8, step_dones[:-1])
            ),
            "step0_episode_starts_all_ones": bool(np.all(step_episode_starts[0] > 1e-8)),
        },
    }


def build_pair1_strict_collect_boundary_drift_source_probe(
    *,
    checkpoint_path: str = PAIR1_CHECKPOINT_PATH,
    train_suite_path: str = PAIR1_TRAIN_SUITE_PATH,
    phase2_summary_path: str = DEFAULT_PHASE2_SUMMARY,
    device: str | None = None,
) -> dict[str, Any]:
    preflight = _load_pair1_preflight()
    primary = _capture_one_collect(
        checkpoint_path=checkpoint_path,
        train_suite_path=train_suite_path,
        phase2_summary_path=phase2_summary_path,
        device=device,
    )
    repeat = _capture_one_collect(
        checkpoint_path=checkpoint_path,
        train_suite_path=train_suite_path,
        phase2_summary_path=phase2_summary_path,
        device=device,
    )

    primary_arrays = primary["step_arrays"]
    repeat_arrays = repeat["step_arrays"]
    field_compare = {
        "obs": _first_diff_step_float(primary_arrays["obs"], repeat_arrays["obs"]),
        "action": _first_diff_step_float(primary_arrays["action"], repeat_arrays["action"]),
        "reward": _first_diff_step_float(primary_arrays["reward"], repeat_arrays["reward"]),
        "values": _first_diff_step_float(primary_arrays["values"], repeat_arrays["values"]),
        "log_probs": _first_diff_step_float(primary_arrays["log_probs"], repeat_arrays["log_probs"]),
        "dones": _first_diff_step_exact(primary_arrays["dones"], repeat_arrays["dones"]),
        "episode_starts": _first_diff_step_exact(
            np.asarray(primary_arrays["episode_starts"], dtype=np.float32) > 1e-8,
            np.asarray(repeat_arrays["episode_starts"], dtype=np.float32) > 1e-8,
        ),
        "objective_masks": _first_diff_step_exact(
            primary_arrays["objective_masks"],
            repeat_arrays["objective_masks"],
        ),
        "objective_episode_indices": _first_diff_step_exact(
            primary_arrays["objective_episode_indices"],
            repeat_arrays["objective_episode_indices"],
        ),
        "objective_episode_positions": _first_diff_step_exact(
            primary_arrays["objective_episode_positions"],
            repeat_arrays["objective_episode_positions"],
        ),
    }

    action_first = field_compare["action"]["first_diff_step"]
    dones_first = field_compare["dones"]["first_diff_step"]
    episode_first = field_compare["episode_starts"]["first_diff_step"]
    obj_ep_first = field_compare["objective_episode_indices"]["first_diff_step"]
    boundary_downstream_of_action = bool(
        action_first is not None
        and dones_first is not None
        and episode_first is not None
        and int(action_first) <= int(dones_first)
        and int(episode_first) == int(dones_first) + 1
        and (obj_ep_first is None or int(obj_ep_first) >= int(episode_first))
    )

    return {
        "probe_entry": "phase3_pair1_strict_collect_boundary_drift_source_probe",
        "preflight": {
            "source_path": PAIR1_PREFLIGHT_JSON,
            "preflight_passed": bool(preflight.get("preflight_passed", False)),
            "contract_args": dict(preflight.get("contract_args", {})),
            "allowed_diff": dict(dict(preflight.get("contract_diff", {})).get("allowed_diff", {})),
            "unexpected_diff": dict(dict(preflight.get("contract_diff", {})).get("unexpected_diff", {})),
        },
        "config": dict(primary["config"]),
        "primary": {
            "runtime": dict(primary["runtime"]),
            "within_run_contract": dict(primary["within_run_contract"]),
        },
        "repeat": {
            "runtime": dict(repeat["runtime"]),
            "within_run_contract": dict(repeat["within_run_contract"]),
        },
        "field_compare": field_compare,
        "conclusions": {
            "obs_locked": bool(field_compare["obs"]["identical"]),
            "objective_mask_locked": bool(field_compare["objective_masks"]["identical"]),
            "action_drifts": bool(not field_compare["action"]["identical"]),
            "dones_drift": bool(not field_compare["dones"]["identical"]),
            "episode_starts_drift": bool(not field_compare["episode_starts"]["identical"]),
            "objective_episode_indices_drift": bool(
                not field_compare["objective_episode_indices"]["identical"]
            ),
            "boundary_drift_is_downstream_of_action_drift": bool(boundary_downstream_of_action),
            "boundary_source_is_not_layout_or_mask": bool(
                field_compare["obs"]["identical"] and field_compare["objective_masks"]["identical"]
            ),
        },
        "recommendation": {
            "reason": (
                "Compare the earliest differing collect step across two pair1 strict-native same-contract runs "
                "to test whether boundary drift starts in episode boundary logic itself or is induced by earlier "
                "policy action/value drift."
            ),
            "next_if_boundary_downstream_of_action": (
                "probe the earliest divergent strict-native policy step hidden/cache/action-mean path"
            ),
            "next_if_boundary_not_downstream_of_action": (
                "probe terminal/reset bookkeeping inside the vectorized environment transition path"
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only pair1 strict collect boundary drift source probe. "
            "Compares the earliest step where action/done/episode-start drift appears "
            "across two same-contract strict-native many-env collects."
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

    report = build_pair1_strict_collect_boundary_drift_source_probe(
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
