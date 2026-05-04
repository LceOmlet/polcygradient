import argparse
import copy
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ticl.analysis.phase2_action_shift_local_reward_direction_probe import (  # noqa: E402
    _policy_action_from_obs,
)
from ticl.analysis.phase2_gym_prior_pack_training_runner import (  # noqa: E402
    _add_prior_reward_component_meta_from_h_list,
    _apply_observation_normalization_if_enabled,
    _apply_reward_normalization_if_enabled,
    _build_algo,
    _collect_prior_policy_pack,
    _fill_existing_rollout_buffer_from_pack,
    _json_safe,
    _load_fixed_frozen_h_list_from_csv,
    _reward_component_arrays_from_pack,
)
from ticl.analysis.phase2_ppo_update_horizon_bad_action_probe import (  # noqa: E402
    _clone_flat_rollout_data,
    _eval_log_prob,
    _flat_update_rows_from_rollout,
    _parse_capture_updates,
    _parse_ints,
    _selected_time_indices,
)
from ticl.analysis.phase2_worst_delta_action_state_counterfactual_probe import (  # noqa: E402
    _clone_cpu_state_dict,
    _parse_horizons,
    _state_at_step_from_pack,
    _stats,
    _write_csv,
    _write_json,
)
from ticl.model_configs import get_model_default_config  # noqa: E402
from ticl.priors.environment_prior import EnvironmentPrior  # noqa: E402
from ticl.rlpfn_maintained_path import (  # noqa: E402
    resolve_rlpfn_token_layout,
    validate_rlpfn_maintained_path_config,
)


DEFAULT_SELECTED_CSV = (
    "/home/chen/RLPFN/artifacts/"
    "phase2_reserve_probe_selected_ratio10_q90p1_posp15_n64_seed9090_0430/selected_envs.csv"
)
DEFAULT_OUTPUT_DIR = (
    "/home/chen/RLPFN/artifacts/"
    "phase2_advantage_vs_empirical_q_probe_env34_9_36_0501"
)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_json_safe(row), sort_keys=True) + "\n")


def _index_update_rows(rows: list[dict[str, Any]]) -> dict[tuple[int, int], dict[str, Any]]:
    out: dict[tuple[int, int], dict[str, Any]] = {}
    for row in rows:
        if not bool(row.get("objective", False)):
            continue
        out[(int(row["env_idx"]), int(row["step_idx"]))] = row
    return out


def _make_eval_items(
    *,
    pack: dict[str, np.ndarray],
    h_list: list[dict[str, Any]],
    env_seeds: list[int],
    env_indices: list[int],
    time_indices: list[int],
    update_idx: int,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for env_idx in env_indices:
        h = h_list[int(env_idx)]
        action_dim = int(h["action_dim"])
        for step_idx in time_indices:
            state = _state_at_step_from_pack(
                pack=pack,
                h=h,
                env_idx=int(env_idx),
                step_idx=int(step_idx),
            )
            if state is None:
                continue
            h_eval = copy.deepcopy(h)
            h_eval["initial_state"] = np.asarray(state, dtype=np.float32).reshape(-1).tolist()
            rollout_action_active = np.asarray(
                pack["actions"][int(step_idx), int(env_idx), :action_dim],
                dtype=np.float32,
            ).reshape(-1)
            items.append(
                {
                    "h": h_eval,
                    "source_h": h,
                    "env_idx": int(env_idx),
                    "env_seed": int(env_seeds[int(env_idx)]),
                    "eval_env_seed": 730000
                    + int(update_idx) * 100000
                    + int(env_idx) * 1000
                    + int(step_idx),
                    "step_idx": int(step_idx),
                    "update_idx": int(update_idx),
                    "source_pack": f"pre_update_{int(update_idx)}",
                    "action_dim": int(action_dim),
                    "rollout_action_active": rollout_action_active,
                }
            )
    return items


def _action_modes(
    *,
    rollout_action: np.ndarray,
    pre_action: np.ndarray,
    post_action: np.ndarray,
) -> dict[str, np.ndarray]:
    rollout = np.asarray(rollout_action, dtype=np.float32).reshape(-1)
    pre = np.asarray(pre_action, dtype=np.float32).reshape(-1)
    post = np.asarray(post_action, dtype=np.float32).reshape(-1)
    delta = post - pre
    return {
        "rollout_first": rollout,
        "pre_first": pre,
        "post_first": post,
        "mirror_post_first": (pre - delta).astype(np.float32, copy=False),
        "half_post_first": (pre + 0.5 * delta).astype(np.float32, copy=False),
        "small_plus_post_dir": (pre + 0.25 * delta).astype(np.float32, copy=False),
        "small_minus_post_dir": (pre - 0.25 * delta).astype(np.float32, copy=False),
    }


def _roll_first_action_then_pre_policy(
    *,
    cfg: dict[str, Any],
    env_cfg: dict[str, Any],
    items: list[dict[str, Any]],
    pre_policy_state: dict[str, torch.Tensor],
    post_policy_state: dict[str, torch.Tensor],
    device_obj: torch.device,
    num_features: int,
    obs_slot_dim: int,
    action_slot_dim: int,
    max_horizon: int,
    horizons: list[int],
    chunk_size: int,
    deterministic_actions: bool,
    build_seed: int,
    objective_reward_scale: float,
    objective_reward_clip: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    horizon_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    horizon_set = {int(v) for v in horizons}
    if not items:
        return horizon_rows, step_rows, action_rows

    for chunk_start in range(0, len(items), int(chunk_size)):
        base_chunk = items[chunk_start : chunk_start + int(chunk_size)]
        base_h_list = [copy.deepcopy(item["h"]) for item in base_chunk]
        base_env_seeds = [int(item["eval_env_seed"]) for item in base_chunk]
        algo_eval = None
        vec_eval = None
        try:
            algo_eval, vec_eval = _build_algo(
                cfg=cfg,
                env_cfg=env_cfg,
                frozen_h=None,
                frozen_h_list=base_h_list,
                frozen_h_list_env_seeds=base_env_seeds,
                prior_mode="fixed_frozen_list",
                device_obj=device_obj,
                num_features=int(num_features),
                n_envs=int(len(base_chunk)),
                n_steps=int(max_horizon),
                build_seed=int(build_seed) + int(chunk_start),
                sb3_reward_normalization_enabled=False,
                sb3_observation_normalization_enabled=True,
                sb3_observation_normalization_clip=10.0,
                sb3_observation_normalization_epsilon=1e-8,
                gain_min=0.0,
                topology_state_gain_min=0.0,
                topology_action_gain_min=0.0,
                topology_state_to_action_ratio_max=0.0,
                topology_max_attempts=4096,
                fixed_env_group_across_updates=True,
            )
            obs0 = vec_eval._full_reset_batch(seeds=base_env_seeds)
            obs_dims = [int(h.get("obs_dim", obs_slot_dim)) for h in base_h_list]
            masks0 = vec_eval.action_masks().astype(np.float32, copy=False)
            obs_for_policy0, _obs_stats = _apply_observation_normalization_if_enabled(
                algo_eval,
                np.asarray(obs0, dtype=np.float32),
                obs_dims=obs_dims,
                obs_slot_dim=int(obs_slot_dim),
                stats_acc=None,
            )
            episode_starts0 = np.ones((int(len(base_chunk)),), dtype=np.float32)
            pre_actions0 = _policy_action_from_obs(
                algo=algo_eval,
                policy_state_dict=pre_policy_state,
                obs_for_policy=obs_for_policy0,
                episode_starts=episode_starts0,
                masks=masks0,
                deterministic=bool(deterministic_actions),
            )
            post_actions0 = _policy_action_from_obs(
                algo=algo_eval,
                policy_state_dict=post_policy_state,
                obs_for_policy=obs_for_policy0,
                episode_starts=episode_starts0,
                masks=masks0,
                deterministic=bool(deterministic_actions),
            )

            mode_items: list[dict[str, Any]] = []
            for local_idx, item in enumerate(base_chunk):
                action_dim = int(item["action_dim"])
                modes = _action_modes(
                    rollout_action=np.asarray(item["rollout_action_active"], dtype=np.float32)[:action_dim],
                    pre_action=pre_actions0[local_idx, :action_dim],
                    post_action=post_actions0[local_idx, :action_dim],
                )
                rollout = np.asarray(item["rollout_action_active"], dtype=np.float64)[:action_dim]
                pre = pre_actions0[local_idx, :action_dim].astype(np.float64)
                post = post_actions0[local_idx, :action_dim].astype(np.float64)
                delta = post - pre
                rollout_delta = rollout - pre
                delta_norm = float(np.linalg.norm(delta))
                rollout_delta_norm = float(np.linalg.norm(rollout_delta))
                denom = max(delta_norm * rollout_delta_norm, 1e-12)
                action_rows.append(
                    {
                        "update_idx": int(item["update_idx"]),
                        "env_idx": int(item["env_idx"]),
                        "step_idx": int(item["step_idx"]),
                        "rollout_action_norm": float(np.linalg.norm(rollout)),
                        "pre_first_action_norm": float(np.linalg.norm(pre)),
                        "post_first_action_norm": float(np.linalg.norm(post)),
                        "post_minus_pre_action_norm": float(delta_norm),
                        "rollout_minus_pre_action_norm": float(rollout_delta_norm),
                        "rollout_cosine_to_post_shift": float(np.dot(rollout_delta, delta) / denom),
                        "rollout_projection_on_post_shift": float(
                            np.dot(rollout_delta, delta) / max(delta_norm, 1e-12)
                        ),
                    }
                )
                for mode, action_active in modes.items():
                    mode_items.append(
                        {
                            **item,
                            "first_action_mode": str(mode),
                            "first_action_active": np.asarray(action_active, dtype=np.float32).reshape(-1),
                            "post_minus_pre_action_norm": float(delta_norm),
                            "rollout_minus_pre_action_norm": float(rollout_delta_norm),
                        }
                    )
        finally:
            if vec_eval is not None:
                vec_eval.close()
            del algo_eval
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        for mode_start in range(0, len(mode_items), int(chunk_size)):
            chunk = mode_items[mode_start : mode_start + int(chunk_size)]
            mode_h_list = [copy.deepcopy(item["h"]) for item in chunk]
            mode_env_seeds = [int(item["eval_env_seed"]) for item in chunk]
            algo_mode = None
            vec_mode = None
            try:
                algo_mode, vec_mode = _build_algo(
                    cfg=cfg,
                    env_cfg=env_cfg,
                    frozen_h=None,
                    frozen_h_list=mode_h_list,
                    frozen_h_list_env_seeds=mode_env_seeds,
                    prior_mode="fixed_frozen_list",
                    device_obj=device_obj,
                    num_features=int(num_features),
                    n_envs=int(len(chunk)),
                    n_steps=int(max_horizon),
                    build_seed=int(build_seed) + int(chunk_start) + 10000 + int(mode_start),
                    sb3_reward_normalization_enabled=False,
                    sb3_observation_normalization_enabled=True,
                    sb3_observation_normalization_clip=10.0,
                    sb3_observation_normalization_epsilon=1e-8,
                    gain_min=0.0,
                    topology_state_gain_min=0.0,
                    topology_action_gain_min=0.0,
                    topology_state_to_action_ratio_max=0.0,
                    topology_max_attempts=4096,
                    fixed_env_group_across_updates=True,
                )
                algo_mode.policy.load_state_dict(pre_policy_state)
                algo_mode.policy.reset_rollout_cache(int(len(chunk)))
                algo_mode.policy.set_training_mode(False)
                obs = vec_mode._full_reset_batch(seeds=mode_env_seeds)
                mode_obs_dims = [int(item["h"].get("obs_dim", obs_slot_dim)) for item in chunk]
                episode_starts = np.ones((int(len(chunk)),), dtype=np.float32)
                reward_env_by_env: list[list[float]] = [[] for _ in chunk]
                reward_total_by_env: list[list[float]] = [[] for _ in chunk]
                reward_objective_by_env: list[list[float]] = [[] for _ in chunk]
                done_by_env: list[list[bool]] = [[] for _ in chunk]
                meta: dict[str, Any] = {}
                _add_prior_reward_component_meta_from_h_list(meta, mode_h_list, int(len(chunk)))
                for local_step in range(int(max_horizon)):
                    masks = vec_mode.action_masks().astype(np.float32, copy=False)
                    if local_step == 0:
                        action_arr = np.zeros((int(len(chunk)), int(action_slot_dim)), dtype=np.float32)
                        for local_idx, item in enumerate(chunk):
                            action_dim = int(item["action_dim"])
                            action_arr[local_idx, :action_dim] = np.asarray(
                                item["first_action_active"],
                                dtype=np.float32,
                            )[:action_dim]
                        action_arr = (action_arr * masks).astype(np.float32, copy=False)
                    else:
                        obs_for_policy, _obs_stats = _apply_observation_normalization_if_enabled(
                            algo_mode,
                            np.asarray(obs, dtype=np.float32),
                            obs_dims=mode_obs_dims,
                            obs_slot_dim=int(obs_slot_dim),
                            stats_acc=None,
                        )
                        action_arr = _policy_action_from_obs(
                            algo=algo_mode,
                            policy_state_dict=pre_policy_state,
                            obs_for_policy=obs_for_policy,
                            episode_starts=episode_starts,
                            masks=masks,
                            deterministic=bool(deterministic_actions),
                        )
                    vec_mode.step_async(action_arr)
                    obs, reward, done, _infos = vec_mode.step_wait()
                    reward_np = np.asarray(reward, dtype=np.float64)
                    done_np = np.asarray(done, dtype=bool)
                    for local_idx, item in enumerate(chunk):
                        action_dim = int(item["action_dim"])
                        active_action = action_arr[local_idx : local_idx + 1, :action_dim]
                        active_mask = masks[local_idx : local_idx + 1, :action_dim]
                        components, _component_meta = _reward_component_arrays_from_pack(
                            arm="prior",
                            reward=np.asarray([float(reward_np[local_idx])], dtype=np.float64),
                            active_actions=active_action.astype(np.float64, copy=False),
                            active_action_masks=active_mask.astype(np.float64, copy=False),
                            meta=meta,
                            env_idx=int(local_idx),
                        )
                        reward_env = float(
                            np.asarray(
                                components.get(
                                    "reward_env",
                                    np.asarray([float(reward_np[local_idx])], dtype=np.float64),
                                )
                            ).reshape(-1)[0]
                        )
                        reward_env_by_env[local_idx].append(float(reward_env))
                        reward_total_by_env[local_idx].append(float(reward_np[local_idx]))
                        reward_objective = float(
                            np.clip(
                                float(reward_np[local_idx]) / max(float(objective_reward_scale), 1e-12),
                                -float(objective_reward_clip),
                                float(objective_reward_clip),
                            )
                        )
                        reward_objective_by_env[local_idx].append(float(reward_objective))
                        done_by_env[local_idx].append(bool(done_np[local_idx]))
                        step_rows.append(
                            {
                                "update_idx": int(item["update_idx"]),
                                "env_idx": int(item["env_idx"]),
                                "step_idx": int(item["step_idx"]),
                                "first_action_mode": str(item["first_action_mode"]),
                                "rollout_step": int(local_step + 1),
                                "reward_env": float(reward_env),
                                "reward_total": float(reward_np[local_idx]),
                                "reward_objective": float(reward_objective),
                                "done": bool(done_np[local_idx]),
                            }
                        )
                    episode_starts = done_np.astype(np.float32, copy=False)
                    prefix_len = int(local_step + 1)
                    if prefix_len in horizon_set:
                        for local_idx, item in enumerate(chunk):
                            env_rewards = np.asarray(
                                reward_env_by_env[local_idx][:prefix_len],
                                dtype=np.float64,
                            )
                            total_rewards = np.asarray(
                                reward_total_by_env[local_idx][:prefix_len],
                                dtype=np.float64,
                            )
                            objective_rewards = np.asarray(
                                reward_objective_by_env[local_idx][:prefix_len],
                                dtype=np.float64,
                            )
                            done_values = np.asarray(done_by_env[local_idx][:prefix_len], dtype=bool)
                            horizon_rows.append(
                                {
                                    "update_idx": int(item["update_idx"]),
                                    "env_idx": int(item["env_idx"]),
                                    "step_idx": int(item["step_idx"]),
                                    "first_action_mode": str(item["first_action_mode"]),
                                    "horizon": int(prefix_len),
                                    "reward_env_sum": float(np.sum(env_rewards)),
                                    "reward_env_q10": float(np.quantile(env_rewards, 0.10)),
                                    "reward_env_q50": float(np.quantile(env_rewards, 0.50)),
                                    "reward_env_q90": float(np.quantile(env_rewards, 0.90)),
                                    "reward_env_positive_fraction": float(np.mean(env_rewards > 0.0)),
                                    "reward_total_sum": float(np.sum(total_rewards)),
                                    "reward_objective_sum": float(np.sum(objective_rewards)),
                                    "reward_objective_q10": float(np.quantile(objective_rewards, 0.10)),
                                    "reward_objective_q50": float(np.quantile(objective_rewards, 0.50)),
                                    "reward_objective_q90": float(np.quantile(objective_rewards, 0.90)),
                                    "reward_objective_positive_fraction": float(
                                        np.mean(objective_rewards > 0.0)
                                    ),
                                    "objective_reward_scale": float(objective_reward_scale),
                                    "objective_reward_clip": float(objective_reward_clip),
                                    "done_count": int(np.sum(done_values)),
                                    "post_minus_pre_action_norm": float(
                                        item["post_minus_pre_action_norm"]
                                    ),
                                    "rollout_minus_pre_action_norm": float(
                                        item["rollout_minus_pre_action_norm"]
                                    ),
                                }
                            )
            finally:
                if vec_mode is not None:
                    vec_mode.close()
                del algo_mode
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    return horizon_rows, step_rows, action_rows


def _pair_q_rows(
    *,
    horizon_rows: list[dict[str, Any]],
    update_rows: list[dict[str, Any]],
    action_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    update_index = _index_update_rows(update_rows)
    action_index = {
        (int(row["update_idx"]), int(row["env_idx"]), int(row["step_idx"])): row
        for row in action_rows
    }
    grouped: dict[tuple[int, int, int, int], dict[str, dict[str, Any]]] = {}
    for row in horizon_rows:
        key = (
            int(row["update_idx"]),
            int(row["env_idx"]),
            int(row["step_idx"]),
            int(row["horizon"]),
        )
        grouped.setdefault(key, {})[str(row["first_action_mode"])] = row

    out: list[dict[str, Any]] = []
    for (update_idx, env_idx, step_idx, horizon), values in sorted(grouped.items()):
        rollout = values.get("rollout_first")
        pre = values.get("pre_first")
        if rollout is None or pre is None:
            continue
        urow = update_index.get((int(env_idx), int(step_idx)))
        if urow is None:
            continue
        q_by_mode = {
            mode: float(row["reward_env_sum"])
            for mode, row in values.items()
            if math.isfinite(float(row["reward_env_sum"]))
        }
        q_objective_by_mode = {
            mode: float(row["reward_objective_sum"])
            for mode, row in values.items()
            if "reward_objective_sum" in row and math.isfinite(float(row["reward_objective_sum"]))
        }
        sorted_modes = sorted(q_by_mode.items(), key=lambda item: item[1], reverse=True)
        sorted_objective_modes = sorted(q_objective_by_mode.items(), key=lambda item: item[1], reverse=True)
        mode_rank = {mode: rank + 1 for rank, (mode, _value) in enumerate(sorted_modes)}
        objective_mode_rank = {
            mode: rank + 1 for rank, (mode, _value) in enumerate(sorted_objective_modes)
        }
        q_values = np.asarray([value for _mode, value in sorted_modes], dtype=np.float64)
        q_objective_values = np.asarray(
            [value for _mode, value in sorted_objective_modes],
            dtype=np.float64,
        )
        rollout_q = float(q_by_mode["rollout_first"])
        pre_q = float(q_by_mode["pre_first"])
        post_q = q_by_mode.get("post_first")
        mirror_q = q_by_mode.get("mirror_post_first")
        median_q = float(np.median(q_values)) if q_values.size else float("nan")
        best_mode = str(sorted_modes[0][0]) if sorted_modes else ""
        best_q = float(sorted_modes[0][1]) if sorted_modes else float("nan")
        worst_mode = str(sorted_modes[-1][0]) if sorted_modes else ""
        worst_q = float(sorted_modes[-1][1]) if sorted_modes else float("nan")
        rollout_objective_q = float(q_objective_by_mode.get("rollout_first", float("nan")))
        pre_objective_q = float(q_objective_by_mode.get("pre_first", float("nan")))
        post_objective_q = q_objective_by_mode.get("post_first")
        mirror_objective_q = q_objective_by_mode.get("mirror_post_first")
        median_objective_q = (
            float(np.median(q_objective_values)) if q_objective_values.size else float("nan")
        )
        best_objective_mode = str(sorted_objective_modes[0][0]) if sorted_objective_modes else ""
        best_objective_q = (
            float(sorted_objective_modes[0][1]) if sorted_objective_modes else float("nan")
        )
        worst_objective_mode = str(sorted_objective_modes[-1][0]) if sorted_objective_modes else ""
        worst_objective_q = (
            float(sorted_objective_modes[-1][1]) if sorted_objective_modes else float("nan")
        )
        raw_adv = float(urow["actor_advantage_raw"])
        std_adv = float(urow["actor_advantage_standardized"])
        action_row = action_index.get((int(update_idx), int(env_idx), int(step_idx)), {})
        record: dict[str, Any] = {
            "update_idx": int(update_idx),
            "env_idx": int(env_idx),
            "step_idx": int(step_idx),
            "horizon": int(horizon),
            "candidate_count": int(len(q_by_mode)),
            "best_mode": best_mode,
            "best_reward_env_sum": best_q,
            "worst_mode": worst_mode,
            "worst_reward_env_sum": worst_q,
            "median_reward_env_sum": median_q,
            "best_mode_objective": best_objective_mode,
            "best_reward_objective_sum": best_objective_q,
            "worst_mode_objective": worst_objective_mode,
            "worst_reward_objective_sum": worst_objective_q,
            "median_reward_objective_sum": median_objective_q,
            "rollout_reward_env_sum": rollout_q,
            "pre_reward_env_sum": pre_q,
            "post_reward_env_sum": None if post_q is None else float(post_q),
            "mirror_reward_env_sum": None if mirror_q is None else float(mirror_q),
            "half_post_reward_env_sum": q_by_mode.get("half_post_first"),
            "small_plus_reward_env_sum": q_by_mode.get("small_plus_post_dir"),
            "small_minus_reward_env_sum": q_by_mode.get("small_minus_post_dir"),
            "rollout_minus_pre_reward_env_sum": float(rollout_q - pre_q),
            "rollout_minus_best_reward_env_sum": float(rollout_q - best_q),
            "rollout_minus_median_reward_env_sum": float(rollout_q - median_q),
            "rollout_minus_post_reward_env_sum": None
            if post_q is None
            else float(rollout_q - float(post_q)),
            "rollout_minus_mirror_reward_env_sum": None
            if mirror_q is None
            else float(rollout_q - float(mirror_q)),
            "rollout_rank_desc": int(mode_rank["rollout_first"]),
            "rollout_reward_objective_sum": rollout_objective_q,
            "pre_reward_objective_sum": pre_objective_q,
            "post_reward_objective_sum": None
            if post_objective_q is None
            else float(post_objective_q),
            "mirror_reward_objective_sum": None
            if mirror_objective_q is None
            else float(mirror_objective_q),
            "half_post_reward_objective_sum": q_objective_by_mode.get("half_post_first"),
            "small_plus_reward_objective_sum": q_objective_by_mode.get("small_plus_post_dir"),
            "small_minus_reward_objective_sum": q_objective_by_mode.get("small_minus_post_dir"),
            "rollout_minus_pre_reward_objective_sum": float(rollout_objective_q - pre_objective_q),
            "rollout_minus_best_reward_objective_sum": float(
                rollout_objective_q - best_objective_q
            ),
            "rollout_minus_median_reward_objective_sum": float(
                rollout_objective_q - median_objective_q
            ),
            "rollout_rank_desc_objective": int(objective_mode_rank.get("rollout_first", -1)),
            "rollout_is_best_objective": bool(best_objective_mode == "rollout_first"),
            "rollout_below_pre_objective": bool(rollout_objective_q < pre_objective_q),
            "rollout_below_post_objective": bool(
                post_objective_q is not None and rollout_objective_q < float(post_objective_q)
            ),
            "rollout_below_mirror_objective": bool(
                mirror_objective_q is not None and rollout_objective_q < float(mirror_objective_q)
            ),
            "rollout_below_median_objective": bool(rollout_objective_q < median_objective_q),
            "rollout_not_best_objective": bool(best_objective_mode != "rollout_first"),
            "rollout_is_best": bool(best_mode == "rollout_first"),
            "rollout_below_pre": bool(rollout_q < pre_q),
            "rollout_below_post": bool(post_q is not None and rollout_q < float(post_q)),
            "rollout_below_mirror": bool(mirror_q is not None and rollout_q < float(mirror_q)),
            "rollout_below_median": bool(rollout_q < median_q),
            "rollout_not_best": bool(best_mode != "rollout_first"),
            "rollout_done_count": int(rollout.get("done_count", 0)),
            "rollout_positive_fraction": float(rollout.get("reward_env_positive_fraction", 0.0)),
            "pre_positive_fraction": float(pre.get("reward_env_positive_fraction", 0.0)),
            "rollout_q90": float(rollout.get("reward_env_q90", 0.0)),
            "pre_q90": float(pre.get("reward_env_q90", 0.0)),
            "actor_advantage_raw": raw_adv,
            "actor_advantage_standardized": std_adv,
            "old_log_prob": float(urow["old_log_prob"]),
            "post_log_prob": float(urow["post_log_prob"]),
            "logprob_delta": float(urow["logprob_delta"]),
            "ratio": float(urow["ratio"]),
            "return_target": float(urow["return_target"]),
            "old_value": float(urow["old_value"]),
            "td_residual_proxy": float(urow["td_residual_proxy"]),
            "advantage_positive": bool(raw_adv > 0.0),
            "advantage_negative": bool(raw_adv < 0.0),
            "ratio_gt_1": bool(float(urow["ratio"]) > 1.0),
            "adv_pos_rollout_below_pre": bool(raw_adv > 0.0 and rollout_q < pre_q),
            "adv_pos_rollout_below_median": bool(raw_adv > 0.0 and rollout_q < median_q),
            "adv_pos_rollout_not_best": bool(raw_adv > 0.0 and best_mode != "rollout_first"),
            "adv_neg_rollout_best": bool(raw_adv < 0.0 and best_mode == "rollout_first"),
            "adv_pos_rollout_below_pre_objective": bool(
                raw_adv > 0.0 and rollout_objective_q < pre_objective_q
            ),
            "adv_pos_rollout_below_median_objective": bool(
                raw_adv > 0.0 and rollout_objective_q < median_objective_q
            ),
            "adv_pos_rollout_not_best_objective": bool(
                raw_adv > 0.0 and best_objective_mode != "rollout_first"
            ),
            "adv_neg_rollout_best_objective": bool(
                raw_adv < 0.0 and best_objective_mode == "rollout_first"
            ),
            "ppo_boosted_rollout_below_pre": bool(
                raw_adv > 0.0 and float(urow["ratio"]) > 1.0 and rollout_q < pre_q
            ),
            "ppo_boosted_rollout_below_median": bool(
                raw_adv > 0.0 and float(urow["ratio"]) > 1.0 and rollout_q < median_q
            ),
            "ppo_boosted_rollout_below_pre_objective": bool(
                raw_adv > 0.0
                and float(urow["ratio"]) > 1.0
                and rollout_objective_q < pre_objective_q
            ),
            "ppo_boosted_rollout_below_median_objective": bool(
                raw_adv > 0.0
                and float(urow["ratio"]) > 1.0
                and rollout_objective_q < median_objective_q
            ),
        }
        for key, value in action_row.items():
            if key not in {"update_idx", "env_idx", "step_idx"}:
                record[f"action_{key}"] = value
        out.append(record)
    return out


def _pearson(x_values: list[Any], y_values: list[Any]) -> float | None:
    xs: list[float] = []
    ys: list[float] = []
    for x, y in zip(x_values, y_values):
        try:
            xf = float(x)
            yf = float(y)
        except (TypeError, ValueError):
            continue
        if math.isfinite(xf) and math.isfinite(yf):
            xs.append(xf)
            ys.append(yf)
    if len(xs) < 3:
        return None
    x_arr = np.asarray(xs, dtype=np.float64)
    y_arr = np.asarray(ys, dtype=np.float64)
    if float(np.std(x_arr)) <= 1e-12 or float(np.std(y_arr)) <= 1e-12:
        return None
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def _summary_by(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row.get(key) for key in keys), []).append(row)
    out: list[dict[str, Any]] = []
    for group_key, group_rows in sorted(grouped.items(), key=lambda item: tuple(str(v) for v in item[0])):
        record = {key: value for key, value in zip(keys, group_key)}
        count = max(1, len(group_rows))
        adv_pos = [row for row in group_rows if bool(row["advantage_positive"])]
        boosted = [row for row in group_rows if bool(row["advantage_positive"]) and bool(row["ratio_gt_1"])]
        record.update(
            {
                "count": int(len(group_rows)),
                "advantage_positive_fraction": float(len(adv_pos) / count),
                "ratio_gt_1_fraction": float(
                    len([row for row in group_rows if bool(row["ratio_gt_1"])]) / count
                ),
                "rollout_minus_pre_reward_env_sum": _stats(
                    [row["rollout_minus_pre_reward_env_sum"] for row in group_rows]
                ),
                "rollout_minus_best_reward_env_sum": _stats(
                    [row["rollout_minus_best_reward_env_sum"] for row in group_rows]
                ),
                "rollout_minus_median_reward_objective_sum": _stats(
                    [row["rollout_minus_median_reward_objective_sum"] for row in group_rows]
                ),
                "rollout_minus_best_reward_objective_sum": _stats(
                    [row["rollout_minus_best_reward_objective_sum"] for row in group_rows]
                ),
                "rollout_rank_desc": _stats([row["rollout_rank_desc"] for row in group_rows]),
                "rollout_rank_desc_objective": _stats(
                    [row["rollout_rank_desc_objective"] for row in group_rows]
                ),
                "rollout_below_pre_fraction": float(
                    len([row for row in group_rows if bool(row["rollout_below_pre"])]) / count
                ),
                "rollout_below_median_fraction": float(
                    len([row for row in group_rows if bool(row["rollout_below_median"])]) / count
                ),
                "rollout_not_best_fraction": float(
                    len([row for row in group_rows if bool(row["rollout_not_best"])]) / count
                ),
                "rollout_below_median_objective_fraction": float(
                    len([row for row in group_rows if bool(row["rollout_below_median_objective"])])
                    / count
                ),
                "rollout_not_best_objective_fraction": float(
                    len([row for row in group_rows if bool(row["rollout_not_best_objective"])])
                    / count
                ),
                "adv_pos_rollout_below_pre_fraction_all": float(
                    len([row for row in group_rows if bool(row["adv_pos_rollout_below_pre"])]) / count
                ),
                "adv_pos_rollout_below_median_fraction_all": float(
                    len([row for row in group_rows if bool(row["adv_pos_rollout_below_median"])]) / count
                ),
                "adv_pos_rollout_not_best_fraction_all": float(
                    len([row for row in group_rows if bool(row["adv_pos_rollout_not_best"])]) / count
                ),
                "adv_pos_rollout_below_median_objective_fraction_all": float(
                    len(
                        [
                            row
                            for row in group_rows
                            if bool(row["adv_pos_rollout_below_median_objective"])
                        ]
                    )
                    / count
                ),
                "adv_neg_rollout_best_objective_fraction_all": float(
                    len([row for row in group_rows if bool(row["adv_neg_rollout_best_objective"])])
                    / count
                ),
                "ppo_boosted_rollout_below_pre_fraction_all": float(
                    len([row for row in group_rows if bool(row["ppo_boosted_rollout_below_pre"])])
                    / count
                ),
                "ppo_boosted_rollout_below_median_fraction_all": float(
                    len([row for row in group_rows if bool(row["ppo_boosted_rollout_below_median"])])
                    / count
                ),
                "ppo_boosted_rollout_below_median_objective_fraction_all": float(
                    len(
                        [
                            row
                            for row in group_rows
                            if bool(row["ppo_boosted_rollout_below_median_objective"])
                        ]
                    )
                    / count
                ),
                "adv_pos_rollout_below_pre_fraction_among_adv_pos": float(
                    len([row for row in adv_pos if bool(row["rollout_below_pre"])])
                    / max(1, len(adv_pos))
                ),
                "adv_pos_rollout_below_median_fraction_among_adv_pos": float(
                    len([row for row in adv_pos if bool(row["rollout_below_median"])])
                    / max(1, len(adv_pos))
                ),
                "boosted_rollout_below_pre_fraction_among_boosted": float(
                    len([row for row in boosted if bool(row["rollout_below_pre"])])
                    / max(1, len(boosted))
                ),
                "boosted_rollout_below_median_fraction_among_boosted": float(
                    len([row for row in boosted if bool(row["rollout_below_median"])])
                    / max(1, len(boosted))
                ),
                "corr_advantage_raw_vs_rollout_minus_pre": _pearson(
                    [row["actor_advantage_raw"] for row in group_rows],
                    [row["rollout_minus_pre_reward_env_sum"] for row in group_rows],
                ),
                "corr_advantage_raw_vs_rollout_minus_pre_objective": _pearson(
                    [row["actor_advantage_raw"] for row in group_rows],
                    [row["rollout_minus_pre_reward_objective_sum"] for row in group_rows],
                ),
                "corr_advantage_raw_vs_rollout_minus_median_objective": _pearson(
                    [row["actor_advantage_raw"] for row in group_rows],
                    [row["rollout_minus_median_reward_objective_sum"] for row in group_rows],
                ),
                "corr_advantage_raw_vs_rollout_minus_best": _pearson(
                    [row["actor_advantage_raw"] for row in group_rows],
                    [row["rollout_minus_best_reward_env_sum"] for row in group_rows],
                ),
                "corr_ratio_vs_rollout_minus_pre": _pearson(
                    [row["ratio"] for row in group_rows],
                    [row["rollout_minus_pre_reward_env_sum"] for row in group_rows],
                ),
            }
        )
        out.append(record)
    return out


def _top_damning_rows(rows: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    candidates = [
        row
        for row in rows
        if bool(row["adv_pos_rollout_below_median"]) or bool(row["ppo_boosted_rollout_below_median"])
    ]
    candidates.sort(
        key=lambda row: (
            bool(row["ppo_boosted_rollout_below_median"]),
            float(row["actor_advantage_raw"]),
            -float(row["rollout_minus_median_reward_env_sum"]),
        ),
        reverse=True,
    )
    return candidates[: int(limit)]


def _write_conclusion(path: Path, summary: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# Advantage vs empirical Q conclusion")
    lines.append("")
    lines.append("检查对象：env34/9/36，固定同一 captured state，对 rollout/pre/post/mirror/perturb first-action 分别闭环滚 5/10/20 步，其余步使用 pre-policy。")
    lines.append("")
    for row in summary.get("summary_by_env_horizon", []):
        env_idx = row.get("env_idx")
        horizon = row.get("horizon")
        lines.append(
            "- env{env} h{h}: count={count}, adv+ rollout<pre all={pre_all:.3f}, "
            "adv+ rollout<median all={med_all:.3f}, boosted rollout<median all={boost_med:.3f}, "
            "corr(adv, rollout-pre)={corr}".format(
                env=env_idx,
                h=horizon,
                count=row.get("count"),
                pre_all=float(row.get("adv_pos_rollout_below_pre_fraction_all") or 0.0),
                med_all=float(row.get("adv_pos_rollout_below_median_fraction_all") or 0.0),
                boost_med=float(row.get("ppo_boosted_rollout_below_median_fraction_all") or 0.0),
                corr=row.get("corr_advantage_raw_vs_rollout_minus_pre"),
            )
        )
    lines.append("")
    lines.append("解释：如果某 env 的 adv+ 且 rollout empirical Q 排名低/低于 median 的比例高，说明 critic/GAE 对 rollout action 的局部信用和 5-20 步真实 reward basin 排序错配；如果比例低，则不能把该 env 硬解释为 advantage 错配，需要回到 state/terminal/closed-loop basin。")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_probe(
    *,
    output_dir: str,
    device: str,
    selected_csv: str,
    selected_rule: str,
    target_env_indices: str,
    n_envs: int,
    n_steps: int,
    updates: int,
    capture_updates: str,
    seed: int,
    time_start: int,
    time_stride: int,
    max_time_points: int,
    horizons: str,
    eval_chunk_size: int,
    deterministic_actions: bool,
    actor_baseline_mode: str,
    actor_gae_lambda_override: float | None,
) -> dict[str, Any]:
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    device_obj = torch.device(device)
    target_envs = _parse_ints(target_env_indices)
    horizon_values = _parse_horizons(horizons)
    capture_set = _parse_capture_updates(capture_updates, updates=int(updates))
    time_indices = _selected_time_indices(
        int(n_steps),
        start=int(time_start),
        stride=int(time_stride),
        max_count=int(max_time_points),
    )

    cfg = copy.deepcopy(get_model_default_config("rlpfn"))
    validate_rlpfn_maintained_path_config(cfg)
    env_cfg = copy.deepcopy(cfg["prior"]["environment"])
    num_features = int(cfg["prior"]["num_features"])
    layout = resolve_rlpfn_token_layout(env_cfg, num_features=num_features)
    action_slot_dim = int(layout["action_slot_dim"])
    obs_slot_dim = int(layout["obs_slot_dim"])
    h_list, h_paths, env_seeds = _load_fixed_frozen_h_list_from_csv(
        selected_csv,
        rule=str(selected_rule),
        limit=int(n_envs),
    )
    if len(h_list) != int(n_envs):
        raise ValueError(f"Expected {int(n_envs)} envs, got {len(h_list)}")
    if max(target_envs) >= int(n_envs):
        raise ValueError(f"target_env_indices out of range for n_envs={n_envs}: {target_envs}")

    dim_probe_prior = EnvironmentPrior(copy.deepcopy(env_cfg))
    next_state_target_dim = int(
        max(
            *[int((h or {}).get("state_dim", 1)) for h in h_list],
            int(dim_probe_prior._resolve_dim_upper_bound(env_cfg.get("state_dim", None), default=0)),
            1,
        )
    )

    algo, vec_env = _build_algo(
        cfg=cfg,
        env_cfg=env_cfg,
        frozen_h=None,
        frozen_h_list=h_list,
        frozen_h_list_env_seeds=env_seeds,
        prior_mode="fixed_frozen_list",
        device_obj=device_obj,
        num_features=int(num_features),
        n_envs=int(n_envs),
        n_steps=int(n_steps),
        build_seed=int(seed),
        sb3_reward_normalization_enabled=True,
        sb3_observation_normalization_enabled=True,
        sb3_observation_normalization_clip=10.0,
        sb3_observation_normalization_epsilon=1e-8,
        gain_min=0.0,
        topology_state_gain_min=0.0,
        topology_action_gain_min=0.0,
        topology_state_to_action_ratio_max=0.0,
        topology_max_attempts=4096,
        fixed_env_group_across_updates=True,
        actor_baseline_mode=str(actor_baseline_mode),
        actor_gae_lambda_override=actor_gae_lambda_override,
    )

    all_update_rows: list[dict[str, Any]] = []
    all_horizon_rows: list[dict[str, Any]] = []
    all_step_rows: list[dict[str, Any]] = []
    all_action_rows: list[dict[str, Any]] = []
    all_q_pair_rows: list[dict[str, Any]] = []
    progress_rows: list[dict[str, Any]] = []
    try:
        for update_idx in range(int(updates)):
            pre_policy_state = _clone_cpu_state_dict(algo.policy)
            pack, values, log_probs, _meta = _collect_prior_policy_pack(
                algo=algo,
                vec_env=vec_env,
                n_steps=int(n_steps),
                seed=int(seed),
                num_features=int(num_features),
                obs_slot_dim=int(obs_slot_dim),
                action_slot_dim=int(action_slot_dim),
                next_state_target_dim=int(next_state_target_dim),
            )
            buffer_pack, reward_norm_stats = _apply_reward_normalization_if_enabled(algo, pack)
            reward_objective_scale = 1.0
            reward_objective_clip = float(getattr(algo, "_rwkv_sb3_reward_normalization_clip", 10.0))
            if bool(reward_norm_stats.get("enabled", False)):
                reward_objective_scale = math.sqrt(
                    max(
                        float(reward_norm_stats.get("ret_rms_var", 1.0)),
                        0.0,
                    )
                    + float(getattr(algo, "_rwkv_sb3_reward_normalization_epsilon", 1e-8))
                )
            _fill_existing_rollout_buffer_from_pack(
                algo.rollout_buffer,
                buffer_pack,
                num_features=int(num_features),
                action_slot_dim=int(action_slot_dim),
                next_state_target_dim=int(next_state_target_dim),
                single_eval_pos=64,
                values_by_step_env=values,
                log_probs_by_step_env=log_probs,
                compute_returns=True,
            )
            rollout_data = _clone_flat_rollout_data(algo)
            pre_eval_logprob = _eval_log_prob(algo, rollout_data)
            algo.train()
            post_policy_state = _clone_cpu_state_dict(algo.policy)
            post_logprob = _eval_log_prob(algo, rollout_data)
            flat_rows = _flat_update_rows_from_rollout(algo, rollout_data, post_logprob)
            for row in flat_rows:
                row["update_idx"] = int(update_idx)
                row["pre_eval_minus_old_logprob"] = float(
                    pre_eval_logprob[int(row["flat_idx"])] - row["old_log_prob"]
                )
            progress_rows.append(
                {
                    "update_idx": int(update_idx),
                    "captured": bool(update_idx in capture_set),
                    "reward_mean": float(np.mean(pack["rewards"])),
                    "target_env_reward_sum_mean": float(
                        np.mean([np.sum(pack["rewards"][:, int(env_idx)]) for env_idx in target_envs])
                    ),
                    "reward_norm": reward_norm_stats,
                }
            )
            if update_idx not in capture_set:
                continue

            items = _make_eval_items(
                pack=pack,
                h_list=h_list,
                env_seeds=env_seeds,
                env_indices=target_envs,
                time_indices=time_indices,
                update_idx=int(update_idx),
            )
            horizon_rows, step_rows, action_rows = _roll_first_action_then_pre_policy(
                cfg=cfg,
                env_cfg=env_cfg,
                items=items,
                pre_policy_state=pre_policy_state,
                post_policy_state=post_policy_state,
                device_obj=device_obj,
                num_features=int(num_features),
                obs_slot_dim=int(obs_slot_dim),
                action_slot_dim=int(action_slot_dim),
                max_horizon=int(max(horizon_values)),
                horizons=horizon_values,
                chunk_size=int(eval_chunk_size),
                deterministic_actions=bool(deterministic_actions),
                build_seed=int(seed) + 30000 + int(update_idx) * 1000,
                objective_reward_scale=float(reward_objective_scale),
                objective_reward_clip=float(reward_objective_clip),
            )
            selected_update_rows = [
                row
                for row in flat_rows
                if int(row["env_idx"]) in set(target_envs) and int(row["step_idx"]) in set(time_indices)
            ]
            q_pair_rows = _pair_q_rows(
                horizon_rows=horizon_rows,
                update_rows=selected_update_rows,
                action_rows=action_rows,
            )
            all_update_rows.extend(selected_update_rows)
            all_horizon_rows.extend(horizon_rows)
            all_step_rows.extend(step_rows)
            all_action_rows.extend(action_rows)
            all_q_pair_rows.extend(q_pair_rows)
            algo.policy.load_state_dict(post_policy_state)
    finally:
        vec_env.close()
        del algo
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = {
        "artifact": "phase2_advantage_vs_empirical_q_probe",
        "interpretation_contract": (
            "For env34/9/36 captured PPO updates, fix the same state and compare empirical "
            "reward_env Q_k for rollout/pre/post/mirror/perturb first actions, then join PPO "
            "actor advantage on the rollout action. High advantage-positive rollout-below-median "
            "means critic/GAE assigns positive credit to an action that ranks poorly under "
            "same-state 5/10/20-step empirical Q."
        ),
        "selected_csv": str(Path(selected_csv).expanduser().resolve()),
        "selected_rule": str(selected_rule),
        "target_envs": [int(v) for v in target_envs],
        "target_env_seeds": [int(env_seeds[int(v)]) for v in target_envs],
        "target_frozen_h_paths": [str(h_paths[int(v)]) for v in target_envs],
        "n_envs": int(n_envs),
        "n_steps": int(n_steps),
        "updates": int(updates),
        "capture_updates": sorted(int(v) for v in capture_set),
        "time_indices": [int(v) for v in time_indices],
        "horizons": [int(v) for v in horizon_values],
        "deterministic_actions": bool(deterministic_actions),
        "actor_baseline_mode": str(actor_baseline_mode),
        "actor_gae_lambda_override": None
        if actor_gae_lambda_override is None
        else float(actor_gae_lambda_override),
        "q_pair_rows_count": int(len(all_q_pair_rows)),
        "summary_by_env_horizon": _summary_by(all_q_pair_rows, ["env_idx", "horizon"]),
        "summary_by_update_env_horizon": _summary_by(
            all_q_pair_rows,
            ["update_idx", "env_idx", "horizon"],
        ),
        "summary_all_by_horizon": _summary_by(all_q_pair_rows, ["horizon"]),
        "top_damning_rows": _top_damning_rows(all_q_pair_rows, limit=20),
        "files": {
            "summary_json": str(out_dir / "summary.json"),
            "conclusion_md": str(out_dir / "conclusion.md"),
            "progress_jsonl": str(out_dir / "progress.jsonl"),
            "update_rows_csv": str(out_dir / "update_rows.csv"),
            "action_rows_csv": str(out_dir / "action_rows.csv"),
            "horizon_rows_csv": str(out_dir / "horizon_rows.csv"),
            "horizon_step_rows_csv": str(out_dir / "horizon_step_rows.csv"),
            "q_pair_rows_csv": str(out_dir / "q_pair_rows.csv"),
        },
    }
    _write_json(out_dir / "summary.json", summary)
    _write_conclusion(out_dir / "conclusion.md", summary)
    _write_jsonl(out_dir / "progress.jsonl", progress_rows)
    _write_csv(out_dir / "update_rows.csv", all_update_rows)
    _write_csv(out_dir / "action_rows.csv", all_action_rows)
    _write_csv(out_dir / "horizon_rows.csv", all_horizon_rows)
    _write_csv(out_dir / "horizon_step_rows.csv", all_step_rows)
    _write_csv(out_dir / "q_pair_rows.csv", all_q_pair_rows)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare PPO advantage with same-state empirical multi-step Q ranking.")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--selected-csv", type=str, default=DEFAULT_SELECTED_CSV)
    parser.add_argument("--selected-rule", type=str, default="ratio10_q90p1_posp15")
    parser.add_argument("--target-env-indices", type=str, default="34,9,36")
    parser.add_argument("--n-envs", type=int, default=64)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--updates", type=int, default=12)
    parser.add_argument("--capture-updates", type=str, default="0,5,11")
    parser.add_argument("--seed", type=int, default=4040)
    parser.add_argument("--time-start", type=int, default=64)
    parser.add_argument("--time-stride", type=int, default=16)
    parser.add_argument("--max-time-points", type=int, default=12)
    parser.add_argument("--horizons", type=str, default="5,10,20")
    parser.add_argument("--eval-chunk-size", type=int, default=96)
    parser.add_argument(
        "--deterministic-actions",
        type=lambda x: str(x).strip().lower() not in {"0", "false", "no"},
        default=True,
    )
    parser.add_argument(
        "--actor-baseline-mode",
        choices=["learned", "zero", "rollout_mean"],
        default="learned",
    )
    parser.add_argument("--actor-gae-lambda-override", type=float, default=None)
    args = parser.parse_args()
    summary = run_probe(
        output_dir=args.output_dir,
        device=args.device,
        selected_csv=args.selected_csv,
        selected_rule=args.selected_rule,
        target_env_indices=args.target_env_indices,
        n_envs=int(args.n_envs),
        n_steps=int(args.n_steps),
        updates=int(args.updates),
        capture_updates=str(args.capture_updates),
        seed=int(args.seed),
        time_start=int(args.time_start),
        time_stride=int(args.time_stride),
        max_time_points=int(args.max_time_points),
        horizons=str(args.horizons),
        eval_chunk_size=int(args.eval_chunk_size),
        deterministic_actions=bool(args.deterministic_actions),
        actor_baseline_mode=str(args.actor_baseline_mode),
        actor_gae_lambda_override=args.actor_gae_lambda_override,
    )
    print(
        json.dumps(
            _json_safe(
                {
                    "summary_json": summary["files"]["summary_json"],
                    "conclusion_md": summary["files"]["conclusion_md"],
                    "target_envs": summary["target_envs"],
                    "q_pair_rows_count": summary["q_pair_rows_count"],
                    "summary_all_by_horizon": summary["summary_all_by_horizon"],
                }
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
