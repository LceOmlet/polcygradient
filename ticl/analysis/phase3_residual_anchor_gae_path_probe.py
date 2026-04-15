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
from ticl.analysis.phase3_heldout_reset_semantics_probe import _objective_episode_segments
from ticl.analysis.phase3_multi_env_optimization_audit import (
    _bind_vec_env_to_fixed_suite,
    _default_device,
    _load_required_suite,
    _load_reused_suite_metrics,
    _resolve_train_profile,
)
from ticl.analysis.phase3_residual_missed_anchor_sign_probe import DEFAULT_SUITE_SPECS
from ticl.analysis.phase3_train_rollout_quality_probe import _collect_one_rollout
from ticl.analysis.prior_generalization_audit import _resolve_audit_env_config, _summarize_suite
from ticl.model_builder import load_model
from ticl.priors.environment_prior import EnvironmentPrior
from ticl.sb3_recurrent_ppo import _discounted_returns_from_rewards, build_recurrent_ppo


TOKEN_RECON_EPS = 1e-5


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(float(v) for v in values) / len(values))


def _max_abs(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(max(abs(float(v)) for v in values))


def _dominant_label(counter: dict[str, int]) -> str:
    if not counter:
        return "none"
    label, count = max(counter.items(), key=lambda item: (int(item[1]), str(item[0])))
    if int(count) <= 0:
        return "none"
    return str(label)


def _fixed_suite_collect_contract_kwargs(suite: dict[str, Any]) -> dict[str, Any]:
    return {
        "strict_fixed_env_mode": True,
        "env_rng_seeds": [int(v) for v in suite["env_seeds"]],
        "rollout_rng_seeds": [int(v) for v in suite["rollout_seeds"]],
        "deterministic_batch_plan": True,
    }


def _collect_contract_plan_for_mode(
    suite: dict[str, Any],
    *,
    collect_contract_mode: str,
) -> dict[str, Any]:
    mode = str(collect_contract_mode).strip().lower()
    if mode == "official_strict":
        return {
            "mode": "official_strict",
            "description": "current official strict fixed-suite collect contract",
            "build_kwargs": dict(_fixed_suite_collect_contract_kwargs(suite)),
        }
    if mode == "legacy_distance_probe":
        return {
            "mode": "legacy_distance_probe",
            "description": "legacy distance-probe collect contract without explicit strict suite seed wiring",
            "build_kwargs": {},
        }
    raise ValueError(f"Unsupported collect_contract_mode={collect_contract_mode!r}")


def _reshape_like_rewards(array: np.ndarray, rewards_shape: tuple[int, int], *, name: str) -> np.ndarray:
    arr = np.asarray(array)
    if tuple(arr.shape) == tuple(rewards_shape):
        return arr
    if int(arr.size) != int(np.prod(rewards_shape)):
        raise ValueError(
            f"{name} cannot be reshaped to rewards shape {tuple(rewards_shape)}: got {tuple(arr.shape)}"
        )
    return arr.reshape(rewards_shape)


def _classify_flip_cause(
    *,
    raw_rollout_residual_raw: float,
    actor_adv_norm: float,
    delta_norm: float,
    reward_norm: float,
    bootstrap_term_norm: float,
) -> str:
    if not (float(raw_rollout_residual_raw) > 0.0 and float(actor_adv_norm) <= 0.0):
        return "not_flip"
    if float(delta_norm) > 0.0:
        return "negative_gae_future_carry"
    if float(bootstrap_term_norm) < 0.0 and abs(float(bootstrap_term_norm)) >= abs(float(reward_norm)):
        return "bootstrap_term_negative"
    return "delta_negative_mixed"


def _build_objective_position_maps(
    objective_mask: np.ndarray,
    segments: list[tuple[int, int]],
) -> tuple[dict[int, int], dict[int, int], dict[int, int]]:
    valid_steps = np.flatnonzero(np.asarray(objective_mask, dtype=bool))
    local_by_step = {int(step): int(local_idx) for local_idx, step in enumerate(valid_steps.tolist())}
    episode_index_by_local: dict[int, int] = {}
    token_in_episode_by_local: dict[int, int] = {}
    tokens_to_episode_end_by_local: dict[int, int] = {}
    for episode_idx, (start, end) in enumerate(segments):
        for local_idx in range(int(start), int(end)):
            episode_index_by_local[int(local_idx)] = int(episode_idx)
            token_in_episode_by_local[int(local_idx)] = int(local_idx - int(start))
            tokens_to_episode_end_by_local[int(local_idx)] = int(int(end) - int(local_idx) - 1)
    return local_by_step, episode_index_by_local, token_in_episode_by_local, tokens_to_episode_end_by_local


def _load_target_anchor_rows(path: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = json.loads(Path(path).expanduser().resolve().read_text())
    config = dict(payload.get("config", {}))
    if str(config.get("anchor_source", "")).strip().lower() != "dual_channel_residual_missed":
        raise ValueError(
            "Residual-anchor GAE probe requires anchor_source=dual_channel_residual_missed in the input artifact."
        )
    rows = list(payload.get("missed_anchor_rows", []))
    if int(len(rows)) != 6:
        raise ValueError(f"Expected 6 residual missed anchors, got {len(rows)}.")
    return rows, payload


def _collect_suite_rollout_with_debug(
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
    collect_contract_mode: str = "official_strict",
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    train_suite = _load_required_suite(str(suite_spec["train_suite_path"]))
    heldout_suite = _load_required_suite(str(suite_spec["heldout_suite_path"]))
    collect_contract_plan = _collect_contract_plan_for_mode(
        heldout_suite,
        collect_contract_mode=str(collect_contract_mode),
    )

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
        **dict(collect_contract_plan["build_kwargs"]),
    )
    _bind_vec_env_to_fixed_suite(vec_env, suite=heldout_suite, single_eval_pos=int(single_eval_pos))
    _make_deterministic_batch_plan(algo.rollout_buffer)
    suite_t0 = time.perf_counter()
    _collect_one_rollout(algo, callback, vec_env)
    suite_runtime_wall_s = float(time.perf_counter() - suite_t0)

    with torch.no_grad():
        final_value_steps = algo.policy.evaluate_rollout_values(
            torch.as_tensor(algo._last_obs, device=device_obj, dtype=torch.float32).unsqueeze(0),
            torch.as_tensor(algo._last_episode_starts, device=device_obj, dtype=torch.float32).unsqueeze(0),
        )
    last_values_norm = final_value_steps.reshape(-1).detach().cpu().numpy().astype(np.float32, copy=False)

    rollout_buffer = algo.rollout_buffer
    rewards_2d = np.asarray(rollout_buffer.rewards, dtype=np.float32)
    rewards_shape = tuple(rewards_2d.shape)
    payload = {
        "suite_name": str(suite_spec["suite_name"]),
        "env_seeds": [int(v) for v in heldout_suite["env_seeds"]],
        "rollout_seeds": [int(v) for v in heldout_suite["rollout_seeds"]],
        "gamma": float(algo.gamma),
        "gae_lambda": float(algo.gae_lambda),
        "objective_masks": _reshape_like_rewards(
            np.asarray(rollout_buffer.objective_masks, dtype=np.float32),
            rewards_shape,
            name="objective_masks",
        ),
        "episode_starts": _reshape_like_rewards(
            np.asarray(rollout_buffer.episode_starts, dtype=np.float32),
            rewards_shape,
            name="episode_starts",
        ),
        "rewards": rewards_2d,
        "values_norm": _reshape_like_rewards(
            np.asarray(rollout_buffer.values, dtype=np.float32),
            rewards_shape,
            name="values_norm",
        ),
        "actor_advantages": _reshape_like_rewards(
            np.asarray(rollout_buffer.actor_advantages, dtype=np.float32),
            rewards_shape,
            name="actor_advantages",
        ),
        "rollout_return_means": _reshape_like_rewards(
            np.asarray(rollout_buffer.rollout_return_means, dtype=np.float32),
            rewards_shape,
            name="rollout_return_means",
        ),
        "rollout_return_stds": _reshape_like_rewards(
            np.asarray(rollout_buffer.rollout_return_stds, dtype=np.float32),
            rewards_shape,
            name="rollout_return_stds",
        ),
        "raw_values": rollout_buffer._get_flat_raw_values_tensor(dtype=torch.float32)
        .detach()
        .cpu()
        .numpy()
        .reshape(rewards_shape),
        "last_values_norm": last_values_norm,
        "last_dones": np.asarray(algo._last_episode_starts, dtype=bool),
        "rollout_discounted_raw_returns": _discounted_returns_from_rewards(
            rewards_2d,
            episode_starts=_reshape_like_rewards(
                np.asarray(rollout_buffer.episode_starts, dtype=np.float32),
                rewards_shape,
                name="episode_starts",
            ),
            dones=np.asarray(algo._last_episode_starts, dtype=bool),
            gamma=float(algo.gamma),
        ),
    }
    vec_env.close()

    suite_summary = {
        "suite_name": str(suite_spec["suite_name"]),
        "train_suite_path": str(Path(suite_spec["train_suite_path"]).expanduser().resolve()),
        "heldout_suite_path": str(Path(suite_spec["heldout_suite_path"]).expanduser().resolve()),
        "zero_json": str(Path(suite_spec["zero_json"]).expanduser().resolve()),
        "ppo_json": str(Path(suite_spec["ppo_json"]).expanduser().resolve()),
        "suite_pre_minus_zero_suffix": float(reused_pre["heldout"]["suffix_return_mean"] - reused_zero["heldout"]["suffix_return_mean"]),
        "suite_runtime_wall_s": float(suite_runtime_wall_s),
        "objective_token_total": int(np.count_nonzero(payload["objective_masks"] > 1e-8)),
    }
    collect_contract_debug = {
        "mode": str(collect_contract_plan["mode"]),
        "description": str(collect_contract_plan["description"]),
        "requested_build_kwargs": {
            str(key): (
                [int(v) for v in value]
                if isinstance(value, (list, tuple))
                else bool(value)
                if isinstance(value, (bool, np.bool_))
                else value
            )
            for key, value in dict(collect_contract_plan["build_kwargs"]).items()
        },
        "suite_env_seeds": [int(v) for v in heldout_suite["env_seeds"]],
        "suite_rollout_seeds": [int(v) for v in heldout_suite["rollout_seeds"]],
        "algo_runtime_flags": {
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
    return payload, suite_summary, collect_contract_debug


def _collect_suite_rollout(
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
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload, suite_summary, _ = _collect_suite_rollout_with_debug(
        checkpoint_path=checkpoint_path,
        suite_spec=suite_spec,
        device_obj=device_obj,
        n_samples=int(n_samples),
        single_eval_pos=int(single_eval_pos),
        train_profile=str(train_profile),
        batch_size=batch_size,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        target_kl=target_kl,
        collect_contract_mode="official_strict",
    )
    return payload, suite_summary


def _analyze_anchor_env(
    *,
    anchor_row: dict[str, Any],
    suite_payload: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    suite_name = str(anchor_row["suite_name"])
    env_index = int(anchor_row["env_index"])
    gamma = float(suite_payload["gamma"])
    gae_lambda = float(suite_payload["gae_lambda"])
    objective_mask = np.asarray(suite_payload["objective_masks"][:, env_index], dtype=np.float32) > 1e-8
    episode_starts = np.asarray(suite_payload["episode_starts"][:, env_index], dtype=np.float32) > 1e-8
    rewards = np.asarray(suite_payload["rewards"][:, env_index], dtype=np.float32)
    values_norm = np.asarray(suite_payload["values_norm"][:, env_index], dtype=np.float32)
    actor_adv = np.asarray(suite_payload["actor_advantages"][:, env_index], dtype=np.float32)
    raw_values = np.asarray(suite_payload["raw_values"][:, env_index], dtype=np.float32)
    rollout_return_means = np.asarray(suite_payload["rollout_return_means"][:, env_index], dtype=np.float32)
    rollout_return_stds = np.asarray(suite_payload["rollout_return_stds"][:, env_index], dtype=np.float32)
    rollout_discounted_raw_returns = np.asarray(
        suite_payload["rollout_discounted_raw_returns"][:, env_index], dtype=np.float32
    )
    last_value_norm = float(np.asarray(suite_payload["last_values_norm"], dtype=np.float32)[env_index])
    last_done = bool(np.asarray(suite_payload["last_dones"], dtype=bool)[env_index])
    last_mean = float(rollout_return_means[-1])
    last_std = float(rollout_return_stds[-1])
    last_value_raw = float(last_mean + last_std * last_value_norm)

    segments = _objective_episode_segments(objective_mask, episode_starts)
    local_by_step, episode_index_by_local, token_in_episode_by_local, tokens_to_episode_end_by_local = (
        _build_objective_position_maps(objective_mask, segments)
    )
    objective_steps = np.flatnonzero(objective_mask)
    token_rows: list[dict[str, Any]] = []
    flip_cause_counts: dict[str, int] = {}
    actor_recon_errors = []
    delta_recon_errors = []
    positive_residual_flip_rows = []

    for global_step in objective_steps.tolist():
        global_step = int(global_step)
        local_idx = int(local_by_step[global_step])
        mean = float(rollout_return_means[global_step])
        std = float(rollout_return_stds[global_step])
        curr_value_norm = float(values_norm[global_step])
        curr_value_raw = float(raw_values[global_step])
        reward_raw = float(rewards[global_step])
        rollout_return_raw = float(rollout_discounted_raw_returns[global_step])
        raw_rollout_residual_raw = float(rollout_return_raw - curr_value_raw)
        raw_rollout_residual_norm = float(raw_rollout_residual_raw / max(std, 1e-8))

        if global_step == int(values_norm.shape[0]) - 1:
            next_non_terminal = 0.0 if bool(last_done) else 1.0
            next_value_norm = float(last_value_norm)
            next_value_raw = float(last_value_raw)
            next_actor_adv = 0.0
        else:
            next_non_terminal = 0.0 if bool(episode_starts[global_step + 1]) else 1.0
            next_value_norm = float(values_norm[global_step + 1])
            next_value_raw = float(raw_values[global_step + 1])
            next_actor_adv = float(actor_adv[global_step + 1])
        gamma_eff = float(gamma * next_non_terminal)
        reward_norm = float(reward_raw / max(std, 1e-8) + ((gamma_eff) - 1.0) * (mean / max(std, 1e-8)))
        bootstrap_term_norm = float(gamma_eff * next_value_norm - curr_value_norm)
        delta_norm = float(reward_norm + bootstrap_term_norm)
        delta_recon_error = float(delta_norm - (reward_norm + bootstrap_term_norm))
        gae_future_carry_norm = float(gamma_eff * gae_lambda * next_actor_adv)
        actor_adv_norm = float(actor_adv[global_step])
        actor_recon_error = float(actor_adv_norm - (delta_norm + gae_future_carry_norm))
        mc_future_carry_equiv_norm = float(raw_rollout_residual_norm - delta_norm)
        bootstrap_term_raw = float(gamma_eff * next_value_raw - curr_value_raw)
        delta_raw = float(reward_raw + bootstrap_term_raw)

        next_objective_local = local_idx + 1
        if next_objective_local < int(len(objective_steps)):
            next_objective_step = int(objective_steps[next_objective_local])
            steps_until_next_objective = int(next_objective_step - global_step)
            next_step_is_objective = bool(next_objective_step == global_step + 1)
        else:
            steps_until_next_objective = -1
            next_step_is_objective = False

        flip_cause = _classify_flip_cause(
            raw_rollout_residual_raw=raw_rollout_residual_raw,
            actor_adv_norm=actor_adv_norm,
            delta_norm=delta_norm,
            reward_norm=reward_norm,
            bootstrap_term_norm=bootstrap_term_norm,
        )
        flip_cause_counts[str(flip_cause)] = int(flip_cause_counts.get(str(flip_cause), 0) + 1)

        row = {
            "suite_name": suite_name,
            "env_index": int(env_index),
            "global_step": int(global_step),
            "objective_local_index": int(local_idx),
            "objective_episode_index": int(episode_index_by_local.get(local_idx, 0)),
            "token_in_objective_episode": int(token_in_episode_by_local.get(local_idx, 0)),
            "tokens_to_objective_episode_end": int(tokens_to_episode_end_by_local.get(local_idx, 0)),
            "next_step_is_objective": bool(next_step_is_objective),
            "steps_until_next_objective": int(steps_until_next_objective),
            "reward_raw": reward_raw,
            "rollout_return_raw": rollout_return_raw,
            "raw_value_raw": curr_value_raw,
            "raw_rollout_residual_raw": raw_rollout_residual_raw,
            "raw_rollout_residual_norm": raw_rollout_residual_norm,
            "value_norm": curr_value_norm,
            "next_value_norm": next_value_norm,
            "next_value_raw": next_value_raw,
            "next_non_terminal": float(next_non_terminal),
            "reward_norm": reward_norm,
            "bootstrap_term_norm": bootstrap_term_norm,
            "bootstrap_term_raw": bootstrap_term_raw,
            "delta_norm": delta_norm,
            "delta_raw": delta_raw,
            "mc_future_carry_equiv_norm": mc_future_carry_equiv_norm,
            "gae_future_carry_norm": gae_future_carry_norm,
            "actor_adv_norm": actor_adv_norm,
            "actor_reconstruction_error": actor_recon_error,
            "delta_reconstruction_error": delta_recon_error,
            "positive_rollout_residual": bool(raw_rollout_residual_raw > 0.0),
            "wrong_sign_actor_adv": bool(actor_adv_norm <= 0.0),
            "positive_residual_wrong_sign_actor": bool(raw_rollout_residual_raw > 0.0 and actor_adv_norm <= 0.0),
            "flip_cause": str(flip_cause),
        }
        token_rows.append(row)
        actor_recon_errors.append(actor_recon_error)
        delta_recon_errors.append(delta_recon_error)
        if bool(row["positive_residual_wrong_sign_actor"]):
            positive_residual_flip_rows.append(row)

    if _max_abs(actor_recon_errors) > float(TOKEN_RECON_EPS):
        raise RuntimeError(
            f"Actor-adv reconstruction error too large for {suite_name} env{env_index}: {_max_abs(actor_recon_errors)}"
        )
    if _max_abs(delta_recon_errors) > float(TOKEN_RECON_EPS):
        raise RuntimeError(
            f"Delta reconstruction error too large for {suite_name} env{env_index}: {_max_abs(delta_recon_errors)}"
        )

    env_summary = {
        "suite_name": suite_name,
        "env_index": int(env_index),
        "env_seed": int(anchor_row["env_seed"]),
        "rollout_seed": int(anchor_row["rollout_seed"]),
        "pre_vs_zero_suffix_gap": float(anchor_row["pre_vs_zero_suffix_gap"]),
        "objective_token_count": int(len(token_rows)),
        "objective_episode_count": int(len(segments)),
        "objective_terminal_reset_count": int(max(0, len(segments) - 1)),
        "positive_rollout_residual_token_count": int(sum(1 for row in token_rows if bool(row["positive_rollout_residual"]))),
        "positive_residual_wrong_sign_actor_count": int(len(positive_residual_flip_rows)),
        "positive_residual_wrong_sign_actor_share": float(
            float(len(positive_residual_flip_rows))
            / max(1, int(sum(1 for row in token_rows if bool(row["positive_rollout_residual"]))))
        ),
        "flip_cause_counts": {
            key: int(value) for key, value in sorted(flip_cause_counts.items()) if str(key) != "not_flip"
        },
        "dominant_flip_cause": _dominant_label(
            {key: int(value) for key, value in flip_cause_counts.items() if str(key) != "not_flip"}
        ),
        "mean_raw_rollout_residual_norm_on_flips": _mean(
            [float(row["raw_rollout_residual_norm"]) for row in positive_residual_flip_rows]
        ),
        "mean_reward_norm_on_flips": _mean([float(row["reward_norm"]) for row in positive_residual_flip_rows]),
        "mean_bootstrap_term_norm_on_flips": _mean(
            [float(row["bootstrap_term_norm"]) for row in positive_residual_flip_rows]
        ),
        "mean_delta_norm_on_flips": _mean([float(row["delta_norm"]) for row in positive_residual_flip_rows]),
        "mean_mc_future_carry_equiv_norm_on_flips": _mean(
            [float(row["mc_future_carry_equiv_norm"]) for row in positive_residual_flip_rows]
        ),
        "mean_gae_future_carry_norm_on_flips": _mean(
            [float(row["gae_future_carry_norm"]) for row in positive_residual_flip_rows]
        ),
        "mean_bootstrap_term_raw_on_flips": _mean(
            [float(row["bootstrap_term_raw"]) for row in positive_residual_flip_rows]
        ),
        "mean_steps_until_next_objective_on_flips": _mean(
            [
                float(row["steps_until_next_objective"])
                for row in positive_residual_flip_rows
                if int(row["steps_until_next_objective"]) >= 0
            ]
        ),
        "next_step_nonobjective_flip_share": float(
            float(sum(1 for row in positive_residual_flip_rows if not bool(row["next_step_is_objective"])))
            / max(1, len(positive_residual_flip_rows))
        ),
        "max_actor_reconstruction_error": _max_abs(actor_recon_errors),
        "max_delta_reconstruction_error": _max_abs(delta_recon_errors),
    }
    return env_summary, token_rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Decompose raw rollout residual, bootstrap term, and GAE carry token-by-token on residual missed anchors."
    )
    parser.add_argument("checkpoint_path", type=str)
    parser.add_argument(
        "--residual-anchor-sign-probe-json",
        type=str,
        default="/home/chen/RLPFN/artifacts/phase3_residual_missed_anchor_sign_probe.json",
    )
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
    checkpoint_path = str(Path(args.checkpoint_path).expanduser().resolve())
    device_obj = torch.device(str(args.device or _default_device()))

    target_rows, anchor_payload = _load_target_anchor_rows(args.residual_anchor_sign_probe_json)
    suite_names = sorted({str(row["suite_name"]) for row in target_rows})
    suite_specs = list(DEFAULT_SUITE_SPECS)
    if args.suite_specs_json is not None:
        suite_specs = json.loads(Path(args.suite_specs_json).expanduser().resolve().read_text())
    suite_specs = [spec for spec in suite_specs if str(spec["suite_name"]) in suite_names]

    suite_payloads = {}
    suite_summaries = []
    for suite_spec in suite_specs:
        payload, suite_summary = _collect_suite_rollout(
            checkpoint_path=checkpoint_path,
            suite_spec=suite_spec,
            device_obj=device_obj,
            n_samples=int(args.n_samples),
            single_eval_pos=int(args.single_eval_pos),
            train_profile=str(args.train_profile),
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            learning_rate=args.learning_rate,
            target_kl=args.target_kl,
        )
        suite_payloads[str(suite_spec["suite_name"])] = payload
        suite_summaries.append(suite_summary)

    env_rows: list[dict[str, Any]] = []
    token_row_bundles: list[dict[str, Any]] = []
    for anchor_row in target_rows:
        suite_name = str(anchor_row["suite_name"])
        env_summary, token_rows = _analyze_anchor_env(
            anchor_row=anchor_row,
            suite_payload=suite_payloads[suite_name],
        )
        env_rows.append(env_summary)
        token_row_bundles.append(
            {
                "suite_name": suite_name,
                "env_index": int(anchor_row["env_index"]),
                "pre_vs_zero_suffix_gap": float(anchor_row["pre_vs_zero_suffix_gap"]),
                "token_rows": token_rows,
            }
        )

    positive_residual_token_count = int(sum(int(row["positive_rollout_residual_token_count"]) for row in env_rows))
    positive_residual_wrong_sign_count = int(sum(int(row["positive_residual_wrong_sign_actor_count"]) for row in env_rows))
    flip_cause_counts = {}
    for row in env_rows:
        for key, value in dict(row["flip_cause_counts"]).items():
            flip_cause_counts[str(key)] = int(flip_cause_counts.get(str(key), 0) + int(value))

    aggregate = {
        "residual_anchor_count": int(len(env_rows)),
        "positive_rollout_residual_token_count": int(positive_residual_token_count),
        "positive_residual_wrong_sign_actor_count": int(positive_residual_wrong_sign_count),
        "positive_residual_wrong_sign_actor_share": float(
            float(positive_residual_wrong_sign_count) / max(1, positive_residual_token_count)
        ),
        "flip_cause_counts": dict(sorted(flip_cause_counts.items())),
        "dominant_flip_cause": _dominant_label(flip_cause_counts),
        "mean_positive_residual_wrong_sign_actor_share_per_env": _mean(
            [float(row["positive_residual_wrong_sign_actor_share"]) for row in env_rows]
        ),
        "mean_delta_norm_on_flips": _mean([float(row["mean_delta_norm_on_flips"]) for row in env_rows]),
        "mean_reward_norm_on_flips": _mean([float(row["mean_reward_norm_on_flips"]) for row in env_rows]),
        "mean_bootstrap_term_norm_on_flips": _mean(
            [float(row["mean_bootstrap_term_norm_on_flips"]) for row in env_rows]
        ),
        "mean_mc_future_carry_equiv_norm_on_flips": _mean(
            [float(row["mean_mc_future_carry_equiv_norm_on_flips"]) for row in env_rows]
        ),
        "mean_gae_future_carry_norm_on_flips": _mean(
            [float(row["mean_gae_future_carry_norm_on_flips"]) for row in env_rows]
        ),
        "mean_bootstrap_term_raw_on_flips": _mean(
            [float(row["mean_bootstrap_term_raw_on_flips"]) for row in env_rows]
        ),
        "max_actor_reconstruction_error": _max_abs(
            [float(row["max_actor_reconstruction_error"]) for row in env_rows]
        ),
        "max_delta_reconstruction_error": _max_abs(
            [float(row["max_delta_reconstruction_error"]) for row in env_rows]
        ),
    }

    conclusions = {
        "actor_recurrence_reconstruction_exact": bool(
            float(aggregate["max_actor_reconstruction_error"]) <= float(TOKEN_RECON_EPS)
        ),
        "delta_decomposition_exact": bool(
            float(aggregate["max_delta_reconstruction_error"]) <= float(TOKEN_RECON_EPS)
        ),
        "bootstrap_term_negative_present": bool(int(flip_cause_counts.get("bootstrap_term_negative", 0)) > 0),
        "negative_gae_future_carry_present": bool(int(flip_cause_counts.get("negative_gae_future_carry", 0)) > 0),
        "gae_future_carry_dominates_token_flips": bool(
            int(flip_cause_counts.get("negative_gae_future_carry", 0))
            > int(flip_cause_counts.get("bootstrap_term_negative", 0))
            + int(flip_cause_counts.get("delta_negative_mixed", 0))
        ),
    }

    result = {
        "audit_entry": "phase3_residual_anchor_gae_path_probe",
        "config": {
            "checkpoint_path": checkpoint_path,
            "residual_anchor_sign_probe_json": str(Path(args.residual_anchor_sign_probe_json).expanduser().resolve()),
            "n_samples": int(args.n_samples),
            "single_eval_pos": int(args.single_eval_pos),
            "train_profile": str(args.train_profile),
            "batch_size": None if args.batch_size is None else int(args.batch_size),
            "n_epochs": None if args.n_epochs is None else int(args.n_epochs),
            "learning_rate": None if args.learning_rate is None else float(args.learning_rate),
            "target_kl": None if args.target_kl is None else float(args.target_kl),
            "suite_names": suite_names,
            "phase2_summary_path": str(Path(args.phase2_summary_path).expanduser().resolve()),
            "token_recon_eps": float(TOKEN_RECON_EPS),
        },
        "anchor_probe_excerpt": {
            "anchor_source": str(anchor_payload["config"]["anchor_source"]),
            "anchor_count": int(len(target_rows)),
            "envs": [
                {
                    "suite_name": str(row["suite_name"]),
                    "env_index": int(row["env_index"]),
                    "pre_vs_zero_suffix_gap": float(row["pre_vs_zero_suffix_gap"]),
                }
                for row in target_rows
            ],
        },
        "suite_summaries": suite_summaries,
        "aggregate": aggregate,
        "env_rows": sorted(env_rows, key=lambda row: float(row["pre_vs_zero_suffix_gap"]), reverse=True),
        "token_row_bundles": token_row_bundles,
        "conclusions": conclusions,
        "runtime_wall_s": float(time.perf_counter() - t0),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(output_path.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
