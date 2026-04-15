import argparse
import copy
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ticl.analysis.phase2_guardrail import (
    DEFAULT_PHASE2_SUMMARY,
    assert_phase2_green,
)
from ticl.analysis.critic_free_single_env_audit import _make_deterministic_batch_plan
from ticl.analysis.phase3_multi_env_optimization_audit import (
    _bind_vec_env_to_fixed_suite,
    _load_required_suite,
    _resolve_train_profile,
)
from ticl.analysis.prior_generalization_audit import _resolve_audit_env_config
from ticl.model_builder import load_model
from ticl.priors.environment_prior import EnvironmentPrior
from ticl.sb3_recurrent_ppo import build_recurrent_ppo


DEFAULT_PAIR1_TRAIN_SUITE = (
    "/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/train_suite.pt"
)


def _default_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _hash_ndarray(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(array.view(np.uint8).tobytes()).hexdigest()


def _to_numpy(value: torch.Tensor | np.ndarray | Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _hash_tensor(value: torch.Tensor | np.ndarray | Any) -> str:
    return _hash_ndarray(_to_numpy(value))


def _mask_action_parts(
    actions: np.ndarray,
    action_masks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    actions = np.asarray(actions, dtype=np.float32)
    action_masks = np.asarray(action_masks, dtype=np.float32)
    valid = actions * (action_masks > 1e-8).astype(np.float32, copy=False)
    tail = actions * (action_masks <= 1e-8).astype(np.float32, copy=False)
    return valid, tail


def _build_many_env_algo(
    *,
    checkpoint_path: str,
    train_suite: dict[str, Any],
    device_obj: torch.device,
    n_samples: int,
    single_eval_pos: int,
    train_profile: str,
) -> tuple[Any, Any, Any]:
    load_model.cache_clear()
    model, config = load_model(checkpoint_path, device=device_obj, verbose=False)
    env_cfg = _resolve_audit_env_config(
        config,
        core_a=False,
        reference_semantics_enabled=False,
    )
    prior = EnvironmentPrior(copy.deepcopy(env_cfg))
    prior._sample_batch_hypers = lambda batch_n: [copy.deepcopy(h) for h in list(train_suite["h_list"])]
    prior._sample_single_eval_pos = lambda n_samples_arg, rng_seed=None: int(single_eval_pos)

    optimizer_cfg = dict(config.get("optimizer", {}))
    profile_cfg = _resolve_train_profile(
        optimizer_cfg=optimizer_cfg,
        train_profile=str(train_profile),
        batch_size=None,
        n_epochs=None,
        learning_rate=None,
        target_kl=None,
    )
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
        reset_env_state_at_sep=bool(profile_cfg["reset_env_state_at_sep"]),
        separate_value_backbone=bool(profile_cfg["separate_value_backbone"]),
        strict_fixed_env_mode=True,
        env_rng_seeds=[int(v) for v in train_suite["env_seeds"]],
        rollout_rng_seeds=[int(v) for v in train_suite["rollout_seeds"]],
        deterministic_actor_sampling=True,
        deterministic_batch_plan=True,
        restore_validation_policy_state=True,
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
    algo.ep_info_buffer = []
    algo.ep_success_buffer = []
    algo.policy.reset_rollout_cache(vec_env.num_envs)
    algo._last_obs = vec_env.reset()
    algo._last_episode_starts = np.ones((vec_env.num_envs,), dtype=bool)
    algo._last_lstm_states = algo.policy._dummy_states(vec_env.num_envs)
    callback.init_callback(algo)
    algo._update_current_progress_remaining(0, int(n_samples))
    return algo, callback, vec_env


def _capture_one_collect(
    *,
    checkpoint_path: str,
    train_suite: dict[str, Any],
    device_obj: torch.device,
    n_samples: int,
    single_eval_pos: int,
    train_profile: str,
    run_seed: int,
) -> dict[str, Any]:
    random.seed(int(run_seed))
    np.random.seed(int(run_seed))
    torch.manual_seed(int(run_seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(run_seed))

    algo, callback, vec_env = _build_many_env_algo(
        checkpoint_path=checkpoint_path,
        train_suite=train_suite,
        device_obj=device_obj,
        n_samples=int(n_samples),
        single_eval_pos=int(single_eval_pos),
        train_profile=str(train_profile),
    )
    captured_steps: list[dict[str, np.ndarray]] = []
    original_rollout = algo._rwkv_env_prior._rollout_family_group_vectorized_with_policy

    def _wrapped_rollout(*args, **kwargs):
        sink = kwargs.get("_policy_ppo_step_sink", None)
        if sink is None:
            raise RuntimeError("Expected _policy_ppo_step_sink in PPO rollout.")

        def _capture_sink(payload):
            captured_steps.append(
                {
                    "episode_starts": payload["episode_starts"].detach().cpu().numpy().astype(np.float32, copy=False),
                    "dones": payload["dones"].detach().cpu().numpy().astype(np.bool_, copy=False),
                    "action": payload["action"].detach().cpu().numpy().astype(np.float32, copy=False),
                    "action_mask": payload["action_mask"].detach().cpu().numpy().astype(np.float32, copy=False),
                    "reward": payload["reward"].detach().cpu().numpy().astype(np.float32, copy=False),
                }
            )
            return sink(payload)

        kwargs["_policy_ppo_step_sink"] = _capture_sink
        return original_rollout(*args, **kwargs)

    algo._rwkv_env_prior._rollout_family_group_vectorized_with_policy = _wrapped_rollout
    try:
        wall_t0 = torch.cuda.Event(enable_timing=True) if algo.device.type == "cuda" else None
        wall_t1 = torch.cuda.Event(enable_timing=True) if algo.device.type == "cuda" else None
        import time

        t0 = time.perf_counter()
        if wall_t0 is not None:
            wall_t0.record()
        ok = algo.collect_rollouts(vec_env, callback, algo.rollout_buffer, n_rollout_steps=algo.n_steps)
        if wall_t1 is not None:
            wall_t1.record()
            torch.cuda.synchronize(device=algo.device)
        collect_wall_s = float(time.perf_counter() - t0)
        if not ok:
            raise RuntimeError("collect_rollouts returned False")

        batch_inds, env_change = next(algo.rollout_buffer._iter_batch_plan(algo.batch_size))
        flat_batch = next(algo.rollout_buffer.get_gpu_flat(algo.batch_size))
        step_episode_starts = np.stack([step["episode_starts"] for step in captured_steps], axis=0)
        step_dones = np.stack([step["dones"] for step in captured_steps], axis=0)
        step_actions = np.stack([step["action"] for step in captured_steps], axis=0)
        step_action_masks = np.stack([step["action_mask"] for step in captured_steps], axis=0)
        step_rewards = np.stack([step["reward"] for step in captured_steps], axis=0)
        step_actions_valid, step_actions_tail = _mask_action_parts(step_actions, step_action_masks)
        buffer_actions_valid, buffer_actions_tail = _mask_action_parts(
            algo.rollout_buffer.actions,
            algo.rollout_buffer.action_masks,
        )
        step_terminal_start_alignment = bool(
            np.array_equal(
                step_episode_starts[1:] > 1e-8,
                step_dones[:-1],
            )
        )
        seq_layout_matches_batch = bool(
            np.array_equal(np.asarray(flat_batch.seq_start_indices, dtype=np.int64), np.asarray(algo.rollout_buffer.seq_start_indices, dtype=np.int64))
        ) if getattr(algo.rollout_buffer, "seq_start_indices", None) is not None else True
        return {
            "collect_wall_s": float(collect_wall_s),
            "step_hashes": {
                "episode_starts": _hash_ndarray(step_episode_starts),
                "dones": _hash_ndarray(step_dones.astype(np.uint8, copy=False)),
                "rewards": _hash_ndarray(step_rewards),
                "actions_full": _hash_ndarray(step_actions),
                "actions_valid": _hash_ndarray(step_actions_valid),
                "actions_tail": _hash_ndarray(step_actions_tail),
                "action_masks": _hash_ndarray(step_action_masks),
            },
            "step_diagnostics": {
                "action_tail_abs_max": float(np.abs(step_actions_tail).max(initial=0.0)),
                "action_tail_nonzero_count": int(np.count_nonzero(np.abs(step_actions_tail) > 1e-8)),
                "episode_starts_follow_previous_dones": bool(step_terminal_start_alignment),
            },
            "buffer_hashes": {
                "episode_starts": _hash_ndarray(algo.rollout_buffer.episode_starts),
                "objective_masks": _hash_ndarray(algo.rollout_buffer.objective_masks),
                "returns": _hash_ndarray(algo.rollout_buffer.returns),
                "advantages": _hash_ndarray(algo.rollout_buffer.advantages),
                "actor_advantages": _hash_ndarray(algo.rollout_buffer.actor_advantages),
                "actions_full": _hash_ndarray(algo.rollout_buffer.actions),
                "actions_valid": _hash_ndarray(buffer_actions_valid),
                "actions_tail": _hash_ndarray(buffer_actions_tail),
            },
            "batch_plan_hashes": {
                "batch_inds": _hash_ndarray(np.asarray(batch_inds, dtype=np.int64)),
                "env_change": _hash_ndarray(np.asarray(env_change, dtype=np.uint8)),
            },
            "flat_batch_hashes": {
                "seq_start_indices": _hash_ndarray(np.asarray(flat_batch.seq_start_indices, dtype=np.int64)),
                "seq_lengths": _hash_ndarray(np.asarray(flat_batch.seq_lengths, dtype=np.int64)),
                "episode_starts": _hash_tensor(flat_batch.episode_starts),
                "returns": _hash_tensor(flat_batch.returns),
                "advantages": _hash_tensor(flat_batch.advantages),
                "actor_advantages": _hash_tensor(flat_batch.actor_advantages),
                "actions_full": _hash_tensor(flat_batch.actions),
            },
            "flat_batch_diagnostics": {
                "seq_layout_matches_buffer_batch": bool(seq_layout_matches_batch),
            },
        }
    finally:
        algo._rwkv_env_prior._rollout_family_group_vectorized_with_policy = original_rollout
        vec_env.close()
        del algo, callback, vec_env
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _compare_runs(run_a: dict[str, Any], run_b: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for section in ("step_hashes", "buffer_hashes", "batch_plan_hashes", "flat_batch_hashes"):
        for key, value_a in run_a[section].items():
            out[f"{section}.{key}"] = bool(value_a == run_b[section][key])
    return out


def _classify_instability(comparisons: dict[str, bool]) -> dict[str, Any]:
    step_episode_starts_equal = bool(comparisons["step_hashes.episode_starts"])
    step_dones_equal = bool(comparisons["step_hashes.dones"])
    step_rewards_equal = bool(comparisons["step_hashes.rewards"])
    step_actions_valid_equal = bool(comparisons["step_hashes.actions_valid"])
    step_actions_tail_equal = bool(comparisons["step_hashes.actions_tail"])
    batch_plan_equal = bool(comparisons["batch_plan_hashes.batch_inds"]) and bool(
        comparisons["batch_plan_hashes.env_change"]
    )
    seq_layout_equal = bool(comparisons["flat_batch_hashes.seq_start_indices"]) and bool(
        comparisons["flat_batch_hashes.seq_lengths"]
    )
    buffer_returns_equal = bool(comparisons["buffer_hashes.returns"])
    buffer_episode_starts_equal = bool(comparisons["buffer_hashes.episode_starts"])
    terminal_reset_episode_starts_instability = not (
        step_episode_starts_equal and step_dones_equal and buffer_episode_starts_equal
    )
    seq_packing_flatten_instability = bool(
        step_episode_starts_equal
        and step_dones_equal
        and batch_plan_equal
        and not seq_layout_equal
    )
    masked_action_tail_cosmetic_only = bool(
        step_actions_valid_equal
        and not step_actions_tail_equal
        and step_rewards_equal
        and step_episode_starts_equal
        and step_dones_equal
    )
    if not buffer_returns_equal:
        if terminal_reset_episode_starts_instability or not step_rewards_equal:
            return_path_driver = "upstream_collect_drift"
        elif seq_packing_flatten_instability:
            return_path_driver = "sequence_packing_or_flatten_order"
        else:
            return_path_driver = "unresolved_non_tail_drift"
    else:
        return_path_driver = "stable"
    return {
        "terminal_reset_or_episode_starts_instability": bool(terminal_reset_episode_starts_instability),
        "seq_packing_or_flatten_order_instability": bool(seq_packing_flatten_instability),
        "masked_action_tail_is_cosmetic_only": bool(masked_action_tail_cosmetic_only),
        "return_path_driver": str(return_path_driver),
    }


def run_phase3_shortrun_sequence_layout_source_probe(
    *,
    checkpoint_path: str,
    train_suite_path: str = DEFAULT_PAIR1_TRAIN_SUITE,
    device: str | None = None,
    n_samples: int = 2048,
    single_eval_pos: int = 1946,
    train_profile: str = "legacy_actor_only_probe",
    phase2_summary_path: str = DEFAULT_PHASE2_SUMMARY,
) -> dict[str, Any]:
    phase2_summary = assert_phase2_green(phase2_summary_path)
    train_suite = _load_required_suite(str(train_suite_path))
    checkpoint_path = str(Path(checkpoint_path).expanduser().resolve())
    device_obj = torch.device(str(device or _default_device()))
    run1 = _capture_one_collect(
        checkpoint_path=checkpoint_path,
        train_suite=train_suite,
        device_obj=device_obj,
        n_samples=int(n_samples),
        single_eval_pos=int(single_eval_pos),
        train_profile=str(train_profile),
        run_seed=2020,
    )
    run2 = _capture_one_collect(
        checkpoint_path=checkpoint_path,
        train_suite=train_suite,
        device_obj=device_obj,
        n_samples=int(n_samples),
        single_eval_pos=int(single_eval_pos),
        train_profile=str(train_profile),
        run_seed=2020,
    )
    comparisons = _compare_runs(run1, run2)
    conclusion = _classify_instability(comparisons)
    return {
        "audit_entry": "phase3_shortrun_sequence_layout_source_probe",
        "checkpoint_path": checkpoint_path,
        "contract": {
            "phase2_summary_path": str(Path(phase2_summary_path).expanduser().resolve()),
            "phase2_all_checks_pass": bool(phase2_summary["validation"]["all_checks_pass"]),
            "train_suite_path": str(Path(train_suite_path).expanduser().resolve()),
            "train_suite_batch_size": int(train_suite["batch_size"]),
            "device": str(device_obj),
            "n_samples": int(n_samples),
            "single_eval_pos": int(single_eval_pos),
            "train_profile": str(train_profile),
            "strict_fixed_env_mode": True,
            "deterministic_actor_sampling": True,
            "deterministic_batch_plan": True,
            "restore_validation_policy_state": True,
        },
        "runs": [run1, run2],
        "comparisons": comparisons,
        "conclusion": conclusion,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 3 many-env shortrun repeated-collect probe for sequence-layout instability sources."
    )
    parser.add_argument("checkpoint_path", type=str)
    parser.add_argument("--train-suite-path", type=str, default=DEFAULT_PAIR1_TRAIN_SUITE)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--n-samples", type=int, default=2048)
    parser.add_argument("--single-eval-pos", type=int, default=1946)
    parser.add_argument(
        "--train-profile",
        type=str,
        default="legacy_actor_only_probe",
        choices=["legacy_actor_only_probe", "trusted_sep_reset_mainline"],
    )
    parser.add_argument("--phase2-summary-path", type=str, default=DEFAULT_PHASE2_SUMMARY)
    parser.add_argument(
        "--output-json",
        type=str,
        default="/home/chen/RLPFN/artifacts/phase3_shortrun_sequence_layout_source_probe.json",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output_json).expanduser().resolve()
    if output_path.exists() and not bool(args.overwrite):
        print(output_path.read_text())
        return 0

    report = run_phase3_shortrun_sequence_layout_source_probe(
        checkpoint_path=args.checkpoint_path,
        train_suite_path=args.train_suite_path,
        device=args.device,
        n_samples=int(args.n_samples),
        single_eval_pos=int(args.single_eval_pos),
        train_profile=str(args.train_profile),
        phase2_summary_path=args.phase2_summary_path,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
