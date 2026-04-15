import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ticl.analysis.phase2_guardrail import DEFAULT_PHASE2_SUMMARY, assert_phase2_green
from ticl.analysis.critic_free_single_env_audit import _make_deterministic_batch_plan
from ticl.analysis.phase3_heldout_reset_semantics_probe import _objective_episode_segments
from ticl.analysis.phase3_multi_env_optimization_audit import (
    _bind_vec_env_to_fixed_suite,
    _default_device,
    _load_required_suite,
    _resolve_train_profile,
)
from ticl.analysis.phase3_residual_missed_anchor_sign_probe import DEFAULT_SUITE_SPECS
from ticl.analysis.phase3_train_rollout_quality_probe import _collect_one_rollout
from ticl.analysis.prior_generalization_audit import _resolve_audit_env_config
from ticl.model_builder import load_model
from ticl.priors.environment_prior import EnvironmentPrior
from ticl.sb3_recurrent_ppo import build_recurrent_ppo


BOUNDARY_CONTRACT_TOL = 1e-6


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(float(v) for v in values) / len(values))


def _max_abs(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(max(abs(float(v)) for v in values))


def _load_suite_specs(path: str | None) -> list[dict[str, Any]]:
    if path is None:
        return [dict(spec) for spec in list(DEFAULT_SUITE_SPECS)]
    payload = json.loads(Path(path).expanduser().resolve().read_text())
    return [dict(spec) for spec in list(payload)]


def _select_suite(spec: dict[str, Any], *, suite_split: str) -> tuple[dict[str, Any], str]:
    split = str(suite_split).strip().lower()
    if split == "train":
        return _load_required_suite(str(spec["train_suite_path"])), str(Path(spec["train_suite_path"]).expanduser().resolve())
    if split == "heldout":
        return _load_required_suite(str(spec["heldout_suite_path"])), str(
            Path(spec["heldout_suite_path"]).expanduser().resolve()
        )
    raise ValueError(f"Unsupported suite_split={suite_split!r}")


def _collect_suite_payload(
    *,
    checkpoint_path: str,
    suite_spec: dict[str, Any],
    suite_split: str,
    device_obj: torch.device,
    n_samples: int,
    single_eval_pos: int,
    train_profile: str,
    batch_size: int | None,
    n_epochs: int | None,
    learning_rate: float | None,
    target_kl: float | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    suite, suite_path = _select_suite(suite_spec, suite_split=str(suite_split))

    load_model.cache_clear()
    model, config = load_model(checkpoint_path, device=device_obj, verbose=False)
    env_cfg = _resolve_audit_env_config(config, core_a=False, reference_semantics_enabled=False)

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
        actor_objective_mode=str(profile_cfg["actor_objective_mode"]),
        reset_env_state_at_sep=bool(profile_cfg["reset_env_state_at_sep"]),
        separate_value_backbone=bool(profile_cfg["separate_value_backbone"]),
        strict_fixed_env_mode=True,
        env_rng_seeds=[int(v) for v in suite["env_seeds"]],
        rollout_rng_seeds=[int(v) for v in suite["rollout_seeds"]],
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
    _bind_vec_env_to_fixed_suite(vec_env, suite=suite, single_eval_pos=int(single_eval_pos))
    _make_deterministic_batch_plan(algo.rollout_buffer)
    t0 = time.perf_counter()
    _collect_one_rollout(algo, callback, vec_env)
    runtime_wall_s = float(time.perf_counter() - t0)

    rollout_buffer = algo.rollout_buffer
    payload = {
        "suite_name": str(suite_spec["suite_name"]),
        "suite_split": str(suite_split),
        "suite_path": str(suite_path),
        "batch_size": int(suite["batch_size"]),
        "env_seeds": [int(v) for v in suite["env_seeds"]],
        "rollout_seeds": [int(v) for v in suite["rollout_seeds"]],
        "rewards": np.asarray(rollout_buffer.rewards, dtype=np.float32),
        "values_norm": np.asarray(rollout_buffer.values, dtype=np.float32),
        "advantages_norm": np.asarray(rollout_buffer.advantages, dtype=np.float32),
        "objective_masks": np.asarray(rollout_buffer.objective_masks, dtype=np.float32),
        "episode_starts": np.asarray(rollout_buffer.episode_starts, dtype=np.float32),
        "rollout_return_means": np.asarray(rollout_buffer.rollout_return_means, dtype=np.float32),
        "rollout_return_stds": np.asarray(rollout_buffer.rollout_return_stds, dtype=np.float32),
        "last_dones": np.asarray(algo._last_episode_starts, dtype=bool),
        "runtime_wall_s": float(runtime_wall_s),
        "collect_contract_debug": {
            "strict_fixed_env_mode": bool(getattr(algo, "_rwkv_strict_fixed_env_mode", False)),
            "env_rng_seed_spec": (
                None
                if getattr(algo, "_rwkv_env_rng_seeds", None) is None
                else [int(v) for v in getattr(algo, "_rwkv_env_rng_seeds", None)]
            ),
            "rollout_rng_seed_spec": (
                None
                if getattr(algo, "_rwkv_rollout_rng_seeds", None) is None
                else [int(v) for v in getattr(algo, "_rwkv_rollout_rng_seeds", None)]
            ),
            "deterministic_actor_sampling": bool(getattr(algo, "_rwkv_deterministic_actor_sampling", False)),
            "deterministic_batch_plan": bool(getattr(algo, "_rwkv_deterministic_batch_plan", False)),
            "last_collect_rollout_env_rng_seeds": (
                None
                if getattr(algo, "_rwkv_last_collect_rollout_env_rng_seeds", None) is None
                else [int(v) for v in getattr(algo, "_rwkv_last_collect_rollout_env_rng_seeds", None)]
            ),
            "last_collect_rollout_rollout_rng_seeds": (
                None
                if getattr(algo, "_rwkv_last_collect_rollout_rollout_rng_seeds", None) is None
                else [int(v) for v in getattr(algo, "_rwkv_last_collect_rollout_rollout_rng_seeds", None)]
            ),
        },
    }
    vec_env.close()
    return payload, profile_cfg


def _analyze_suite_payload(payload: dict[str, Any], *, tol: float) -> dict[str, Any]:
    rewards = np.asarray(payload["rewards"], dtype=np.float32)
    values_norm = np.asarray(payload["values_norm"], dtype=np.float32)
    advantages_norm = np.asarray(payload["advantages_norm"], dtype=np.float32)
    objective_masks = np.asarray(payload["objective_masks"], dtype=np.float32) > 1e-8
    episode_starts = np.asarray(payload["episode_starts"], dtype=np.float32) > 1e-8
    means = np.asarray(payload["rollout_return_means"], dtype=np.float32)
    stds = np.asarray(payload["rollout_return_stds"], dtype=np.float32)
    last_dones = np.asarray(payload["last_dones"], dtype=bool).reshape((-1,))

    terminal_rows: list[dict[str, Any]] = []
    env_rows: list[dict[str, Any]] = []
    time_steps, n_envs = rewards.shape
    for env_idx in range(int(n_envs)):
        env_objective_mask = objective_masks[:, env_idx]
        objective_steps = np.flatnonzero(env_objective_mask)
        local_by_step = {int(step): int(local_idx) for local_idx, step in enumerate(objective_steps.tolist())}
        segments = _objective_episode_segments(env_objective_mask, episode_starts[:, env_idx])
        objective_episode_index_by_local: dict[int, int] = {}
        for episode_idx, (start_local, end_local) in enumerate(segments):
            for local_idx in range(int(start_local), int(end_local)):
                objective_episode_index_by_local[int(local_idx)] = int(episode_idx)
        env_terminal_rows: list[dict[str, Any]] = []
        for step in objective_steps.tolist():
            step = int(step)
            if step == int(time_steps) - 1:
                next_non_terminal = 0.0 if bool(last_dones[env_idx]) else 1.0
            else:
                next_non_terminal = 0.0 if bool(episode_starts[step + 1, env_idx]) else 1.0
            if float(next_non_terminal) > 0.0:
                continue
            std = float(max(float(stds[step, env_idx]), 1e-8))
            mean = float(means[step, env_idx])
            reward_raw = float(rewards[step, env_idx])
            recovered_reward_norm = float(advantages_norm[step, env_idx] + values_norm[step, env_idx])
            reward_over_std = float(reward_raw / std)
            actual_boundary_shift_norm = float(recovered_reward_norm - reward_over_std)
            expected_boundary_shift_norm = float(-mean / std)
            contract_error = float(actual_boundary_shift_norm - expected_boundary_shift_norm)
            row = {
                "suite_name": str(payload["suite_name"]),
                "suite_split": str(payload["suite_split"]),
                "env_index": int(env_idx),
                "env_seed": int(payload["env_seeds"][env_idx]),
                "rollout_seed": int(payload["rollout_seeds"][env_idx]),
                "global_step": int(step),
                "objective_local_index": int(local_by_step[int(step)]),
                "objective_episode_index": int(objective_episode_index_by_local.get(int(local_by_step[int(step)]), 0)),
                "reward_raw": float(reward_raw),
                "value_norm": float(values_norm[step, env_idx]),
                "advantage_norm": float(advantages_norm[step, env_idx]),
                "recovered_reward_norm": float(recovered_reward_norm),
                "reward_over_std": float(reward_over_std),
                "actual_boundary_shift_norm": float(actual_boundary_shift_norm),
                "expected_boundary_shift_norm": float(expected_boundary_shift_norm),
                "boundary_shift_contract_error": float(contract_error),
                "abs_boundary_shift_contract_error": float(abs(contract_error)),
                "terminal_due_to_last_done": bool(step == int(time_steps) - 1 and bool(last_dones[env_idx])),
                "contract_matches_full_rollout_centering": bool(abs(float(contract_error)) <= float(tol)),
            }
            env_terminal_rows.append(row)
            terminal_rows.append(row)

        env_errors = [float(row["abs_boundary_shift_contract_error"]) for row in env_terminal_rows]
        env_rows.append(
            {
                "suite_name": str(payload["suite_name"]),
                "suite_split": str(payload["suite_split"]),
                "env_index": int(env_idx),
                "env_seed": int(payload["env_seeds"][env_idx]),
                "rollout_seed": int(payload["rollout_seeds"][env_idx]),
                "objective_terminal_token_count": int(len(env_terminal_rows)),
                "objective_terminal_local_indices": [int(row["objective_local_index"]) for row in env_terminal_rows],
                "max_abs_boundary_shift_contract_error": _max_abs(env_errors),
                "mean_abs_boundary_shift_contract_error": _mean(env_errors),
                "boundary_shift_mismatch_count": int(
                    sum(1 for row in env_terminal_rows if not bool(row["contract_matches_full_rollout_centering"]))
                ),
            }
        )

    suite_errors = [float(row["abs_boundary_shift_contract_error"]) for row in terminal_rows]
    suite_summary = {
        "suite_name": str(payload["suite_name"]),
        "suite_split": str(payload["suite_split"]),
        "suite_path": str(payload["suite_path"]),
        "batch_size": int(payload["batch_size"]),
        "runtime_wall_s": float(payload["runtime_wall_s"]),
        "collect_contract_debug": dict(payload["collect_contract_debug"]),
        "objective_terminal_token_count": int(len(terminal_rows)),
        "envs_with_objective_terminal_tokens": int(sum(1 for row in env_rows if int(row["objective_terminal_token_count"]) > 0)),
        "max_abs_boundary_shift_contract_error": _max_abs(suite_errors),
        "mean_abs_boundary_shift_contract_error": _mean(suite_errors),
        "boundary_shift_mismatch_count": int(
            sum(1 for row in terminal_rows if not bool(row["contract_matches_full_rollout_centering"]))
        ),
        "mismatch_env_indices": [
            int(row["env_index"]) for row in env_rows if int(row["boundary_shift_mismatch_count"]) > 0
        ],
    }
    return {
        "suite_summary": suite_summary,
        "env_rows": env_rows,
        "terminal_rows": terminal_rows,
    }


def _build_scan(
    *,
    checkpoint_path: str,
    suite_specs: list[dict[str, Any]],
    suite_split: str,
    device_obj: torch.device,
    n_samples: int,
    single_eval_pos: int,
    train_profile: str,
    batch_size: int | None,
    n_epochs: int | None,
    learning_rate: float | None,
    target_kl: float | None,
    tol: float,
) -> dict[str, Any]:
    suite_reports = []
    for suite_spec in suite_specs:
        payload, profile_cfg = _collect_suite_payload(
            checkpoint_path=checkpoint_path,
            suite_spec=suite_spec,
            suite_split=str(suite_split),
            device_obj=device_obj,
            n_samples=int(n_samples),
            single_eval_pos=int(single_eval_pos),
            train_profile=str(train_profile),
            batch_size=batch_size,
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            target_kl=target_kl,
        )
        analysis = _analyze_suite_payload(payload, tol=float(tol))
        suite_reports.append(
            {
                "suite_name": str(suite_spec["suite_name"]),
                "profile_cfg": {
                    "batch_size": int(profile_cfg["batch_size"]),
                    "n_epochs": int(profile_cfg["n_epochs"]),
                    "learning_rate": float(profile_cfg["learning_rate"]),
                    "target_kl": (
                        None if profile_cfg["target_kl"] is None else float(profile_cfg["target_kl"])
                    ),
                    "reset_env_state_at_sep": bool(profile_cfg["reset_env_state_at_sep"]),
                    "actor_baseline_mode": str(profile_cfg["actor_baseline_mode"]),
                    "actor_gae_space": str(profile_cfg["actor_gae_space"]),
                    "actor_objective_mode": str(profile_cfg["actor_objective_mode"]),
                    "separate_value_backbone": bool(profile_cfg["separate_value_backbone"]),
                },
                **analysis,
            }
        )

    suite_rows = [dict(report["suite_summary"]) for report in suite_reports]
    env_rows = []
    terminal_rows = []
    for report in suite_reports:
        env_rows.extend(list(report["env_rows"]))
        terminal_rows.extend(list(report["terminal_rows"]))

    terminal_errors = [float(row["abs_boundary_shift_contract_error"]) for row in terminal_rows]
    aggregate = {
        "suite_count": int(len(suite_reports)),
        "suite_names": [str(report["suite_name"]) for report in suite_reports],
        "suite_split": str(suite_split),
        "objective_terminal_token_count": int(len(terminal_rows)),
        "envs_with_objective_terminal_tokens": int(sum(1 for row in env_rows if int(row["objective_terminal_token_count"]) > 0)),
        "boundary_shift_mismatch_count": int(
            sum(1 for row in terminal_rows if not bool(row["contract_matches_full_rollout_centering"]))
        ),
        "max_abs_boundary_shift_contract_error": _max_abs(terminal_errors),
        "mean_abs_boundary_shift_contract_error": _mean(terminal_errors),
        "suite_mismatch_names": [
            str(row["suite_name"]) for row in suite_rows if int(row["boundary_shift_mismatch_count"]) > 0
        ],
    }

    return {
        "audit_entry": "phase3_runtime_terminal_boundary_drift_scan",
        "config": {
            "checkpoint_path": str(Path(checkpoint_path).expanduser().resolve()),
            "suite_split": str(suite_split),
            "n_samples": int(n_samples),
            "single_eval_pos": int(single_eval_pos),
            "train_profile": str(train_profile),
            "batch_size": None if batch_size is None else int(batch_size),
            "n_epochs": None if n_epochs is None else int(n_epochs),
            "learning_rate": None if learning_rate is None else float(learning_rate),
            "target_kl": None if target_kl is None else float(target_kl),
            "boundary_contract_tol": float(tol),
            "suite_specs": [
                {
                    "suite_name": str(spec["suite_name"]),
                    "train_suite_path": str(Path(spec["train_suite_path"]).expanduser().resolve()),
                    "heldout_suite_path": str(Path(spec["heldout_suite_path"]).expanduser().resolve()),
                }
                for spec in suite_specs
            ],
        },
        "aggregate": aggregate,
        "suite_rows": suite_rows,
        "env_rows": sorted(
            env_rows,
            key=lambda row: (
                -float(row["max_abs_boundary_shift_contract_error"]),
                -int(row["objective_terminal_token_count"]),
                str(row["suite_name"]),
                int(row["env_index"]),
            ),
        ),
        "terminal_rows": sorted(
            terminal_rows,
            key=lambda row: (
                -float(row["abs_boundary_shift_contract_error"]),
                str(row["suite_name"]),
                int(row["env_index"]),
                int(row["global_step"]),
            ),
        ),
        "conclusions": {
            "no_runtime_terminal_boundary_drift_detected": bool(int(aggregate["boundary_shift_mismatch_count"]) == 0),
            "all_runtime_terminal_tokens_match_full_rollout_centering_contract": bool(
                int(aggregate["boundary_shift_mismatch_count"]) == 0
            ),
            "current_phase3_train_collect_path_already_respects_boundary_guardrail": bool(
                int(aggregate["boundary_shift_mismatch_count"]) == 0
            ),
            "small_runtime_fix_for_terminal_boundary_scope_not_currently_localized_in_train_collect_path": bool(
                int(aggregate["boundary_shift_mismatch_count"]) == 0
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scan current Phase 3 runtime collect path for terminal-token boundary drift away from full-rollout centering."
    )
    parser.add_argument("checkpoint_path", type=str)
    parser.add_argument("--suite-specs-json", type=str, default=None)
    parser.add_argument("--suite-split", type=str, default="train", choices=["train", "heldout"])
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
    parser.add_argument("--boundary-contract-tol", type=float, default=BOUNDARY_CONTRACT_TOL)
    parser.add_argument("--phase2-summary-path", type=str, default=DEFAULT_PHASE2_SUMMARY)
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output_json).expanduser().resolve()
    if output_path.exists() and not bool(args.overwrite):
        print(output_path.read_text())
        return 0

    t0 = time.perf_counter()
    assert_phase2_green(args.phase2_summary_path)
    suite_specs = _load_suite_specs(args.suite_specs_json)
    result = _build_scan(
        checkpoint_path=str(Path(args.checkpoint_path).expanduser().resolve()),
        suite_specs=suite_specs,
        suite_split=str(args.suite_split),
        device_obj=torch.device(str(args.device or _default_device())),
        n_samples=int(args.n_samples),
        single_eval_pos=int(args.single_eval_pos),
        train_profile=str(args.train_profile),
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        learning_rate=args.learning_rate,
        target_kl=args.target_kl,
        tol=float(args.boundary_contract_tol),
    )
    result["runtime_wall_s"] = float(time.perf_counter() - t0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(output_path.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
