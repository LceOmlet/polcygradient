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

from ticl.analysis.phase2_action_shift_local_reward_direction_probe import _policy_action_from_obs  # noqa: E402
from ticl.analysis.phase2_advantage_vs_empirical_q_probe import DEFAULT_SELECTED_CSV  # noqa: E402
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
    _parse_capture_updates,
    _parse_horizons,
    _parse_ints,
    _selected_time_indices,
)
from ticl.analysis.phase2_worst_delta_action_state_counterfactual_probe import (  # noqa: E402
    _clone_cpu_state_dict,
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


DEFAULT_Q_PAIR_CSV = (
    "/home/chen/RLPFN/artifacts/"
    "phase2_objective_space_empirical_q_probe_env34_9_36_0501/q_pair_rows.csv"
)
DEFAULT_DELTA_CSV = (
    "/home/chen/RLPFN/artifacts/"
    "phase2_delta_worst_env_analysis_0430/topology_ratio10_reserve_n64_delta_rank_with_seed.csv"
)
DEFAULT_OUTPUT_DIR = (
    "/home/chen/RLPFN/artifacts/"
    "phase2_long_empirical_return_vs_gae_probe_env34_9_36_0501"
)


def _read_csv_rows(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).expanduser().open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _to_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _as_step_env_array(values: np.ndarray, *, n_steps: int, n_envs: int, name: str) -> np.ndarray:
    arr = np.asarray(values)
    if arr.shape == (int(n_steps), int(n_envs)):
        return arr
    if arr.shape == (int(n_steps) * int(n_envs),):
        return arr.reshape((int(n_steps), int(n_envs)))
    if arr.shape == (int(n_steps) * int(n_envs), 1):
        return arr.reshape((int(n_steps), int(n_envs)))
    if arr.ndim == 3 and arr.shape[0] == int(n_steps) and arr.shape[1] == 1 and arr.shape[2] == int(n_envs):
        return arr[:, 0, :]
    if arr.ndim == 3 and arr.shape[0] == int(n_steps) and arr.shape[1] == int(n_envs) and arr.shape[2] == 1:
        return arr[:, :, 0]
    raise ValueError(f"{name} cannot be normalized to [steps, envs], got shape={tuple(arr.shape)}")


def _policy_action_value_from_obs(
    *,
    algo,
    policy_state_dict: dict[str, torch.Tensor],
    obs_for_policy: np.ndarray,
    episode_starts: np.ndarray,
    masks: np.ndarray,
    deterministic: bool,
) -> tuple[np.ndarray, np.ndarray]:
    algo.policy.load_state_dict(policy_state_dict)
    algo.policy.reset_rollout_cache(int(obs_for_policy.shape[0]))
    algo.policy.set_training_mode(False)
    obs_t = torch.as_tensor(obs_for_policy, device=algo.device, dtype=torch.float32)
    starts_t = torch.as_tensor(episode_starts, device=algo.device, dtype=torch.float32)
    masks_t = torch.as_tensor(masks, device=algo.device, dtype=torch.float32)
    dummy_states = algo.policy._dummy_states(int(obs_t.shape[0]))
    with torch.no_grad():
        actions_t, values_t, _log_probs_t, _ = algo.policy(
            obs_t,
            dummy_states,
            starts_t,
            deterministic=bool(deterministic),
            action_masks=masks_t,
        )
    return (
        actions_t.detach().cpu().numpy().astype(np.float32, copy=False),
        values_t.detach().cpu().numpy().reshape(-1).astype(np.float32, copy=False),
    )


def _select_mismatch_keys(
    *,
    q_pair_csv: str,
    target_envs: list[int],
    horizons: list[int],
) -> tuple[set[tuple[int, int, int]], dict[tuple[int, int, int, int], dict[str, Any]]]:
    rows = _read_csv_rows(q_pair_csv)
    q_index: dict[tuple[int, int, int, int], dict[str, Any]] = {}
    keys: set[tuple[int, int, int]] = set()
    horizon_set = {int(v) for v in horizons}
    target_set = {int(v) for v in target_envs}
    for row in rows:
        env_idx = int(float(row["env_idx"]))
        horizon = int(float(row["horizon"]))
        if env_idx not in target_set or horizon not in horizon_set:
            continue
        key4 = (
            int(float(row["update_idx"])),
            env_idx,
            int(float(row["step_idx"])),
            horizon,
        )
        q_index[key4] = row
        if _to_bool(row.get("adv_pos_rollout_below_median_objective")) or _to_bool(
            row.get("adv_neg_rollout_best_objective")
        ):
            keys.add(key4[:3])
    return keys, q_index


def _discounted_prefix(values: list[float], gamma: float, *, limit: int) -> float:
    out = 0.0
    weight = 1.0
    for value in values[: int(limit)]:
        out += weight * float(value)
        weight *= float(gamma)
    return float(out)


def _summarize(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row.get(key) for key in keys), []).append(row)
    out: list[dict[str, Any]] = []
    for key_values, group in sorted(grouped.items(), key=lambda item: tuple(str(v) for v in item[0])):
        record = {key: value for key, value in zip(keys, key_values)}
        count = max(1, len(group))
        record.update(
            {
                "count": int(len(group)),
                "delta_env_sum": _stats([row.get("delta_env_sum") for row in group]),
                "gae_advantage": _stats([row.get("gae_advantage") for row in group]),
                "empirical_advantage_bootstrap": _stats(
                    [row.get("empirical_advantage_bootstrap") for row in group]
                ),
                "gae_minus_empirical_advantage": _stats(
                    [row.get("gae_minus_empirical_advantage") for row in group]
                ),
                "same_sign_fraction": float(
                    np.mean(
                        [
                            (float(row["gae_advantage"]) >= 0.0)
                            == (float(row["empirical_advantage_bootstrap"]) >= 0.0)
                            for row in group
                        ]
                    )
                ),
                "gae_positive_empirical_negative_fraction": float(
                    np.mean(
                        [
                            float(row["gae_advantage"]) > 0.0
                            and float(row["empirical_advantage_bootstrap"]) < 0.0
                            for row in group
                        ]
                    )
                ),
                "gae_negative_empirical_positive_fraction": float(
                    np.mean(
                        [
                            float(row["gae_advantage"]) < 0.0
                            and float(row["empirical_advantage_bootstrap"]) > 0.0
                            for row in group
                        ]
                    )
                ),
                "done_before_horizon_fraction": float(
                    np.mean([bool(row["done_before_or_at_horizon"]) for row in group])
                ),
                "effective_steps": _stats([row.get("effective_steps") for row in group]),
                "bootstrap_contribution": _stats([row.get("bootstrap_contribution") for row in group]),
                "empirical_target_minus_v0": _stats([row.get("empirical_target_minus_v0") for row in group]),
            }
        )
        out.append(record)
    return out


def run_probe(
    *,
    output_dir: str,
    device: str,
    selected_csv: str,
    selected_rule: str,
    q_pair_csv: str,
    delta_csv: str,
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
    target_envs = _parse_ints(target_env_indices)
    horizon_values = _parse_horizons(horizons)
    max_horizon = int(max(horizon_values))
    capture_set = _parse_capture_updates(capture_updates, updates=int(updates))
    time_indices = _selected_time_indices(
        int(n_steps),
        start=int(time_start),
        stride=int(time_stride),
        max_count=int(max_time_points),
    )
    mismatch_keys, q_index = _select_mismatch_keys(
        q_pair_csv=q_pair_csv,
        target_envs=target_envs,
        horizons=[5, 10, 20],
    )
    delta_index = {int(float(row["env_idx"])): row for row in _read_csv_rows(delta_csv)}

    cfg = copy.deepcopy(get_model_default_config("rlpfn"))
    validate_rlpfn_maintained_path_config(cfg)
    env_cfg = copy.deepcopy(cfg["prior"]["environment"])
    num_features = int(cfg["prior"]["num_features"])
    layout = resolve_rlpfn_token_layout(env_cfg, num_features=num_features)
    action_slot_dim = int(layout["action_slot_dim"])
    obs_slot_dim = int(layout["obs_slot_dim"])
    h_list, _h_paths, env_seeds = _load_fixed_frozen_h_list_from_csv(
        selected_csv,
        rule=str(selected_rule),
        limit=int(n_envs),
    )
    dim_probe_prior = EnvironmentPrior(copy.deepcopy(env_cfg))
    next_state_target_dim = int(
        max(
            *[int((h or {}).get("state_dim", 1)) for h in h_list],
            int(dim_probe_prior._resolve_dim_upper_bound(env_cfg.get("state_dim", None), default=0)),
            1,
        )
    )
    device_obj = torch.device(device)
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

    result_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
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
            if bool(reward_norm_stats.get("enabled", False)):
                reward_objective_scale = math.sqrt(
                    max(float(reward_norm_stats.get("ret_rms_var", 1.0)), 0.0)
                    + float(getattr(algo, "_rwkv_sb3_reward_normalization_epsilon", 1e-8))
                )
            reward_objective_clip = float(getattr(algo, "_rwkv_sb3_reward_normalization_clip", 10.0))
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
            buffer_values_2d = _as_step_env_array(
                np.asarray(algo.rollout_buffer.values, dtype=np.float32).copy(),
                n_steps=int(n_steps),
                n_envs=int(n_envs),
                name="buffer_values",
            )
            buffer_actor_adv_2d = _as_step_env_array(
                np.asarray(algo.rollout_buffer.actor_advantages, dtype=np.float32).copy(),
                n_steps=int(n_steps),
                n_envs=int(n_envs),
                name="buffer_actor_advantages",
            )
            progress_rows.append(
                {
                    "update_idx": int(update_idx),
                    "captured": bool(update_idx in capture_set),
                    "reward_norm": reward_norm_stats,
                }
            )
            if update_idx in capture_set:
                items: list[dict[str, Any]] = []
                for env_idx in target_envs:
                    h = h_list[int(env_idx)]
                    action_dim = int(h["action_dim"])
                    for step_idx in time_indices:
                        key = (int(update_idx), int(env_idx), int(step_idx))
                        if key not in mismatch_keys:
                            continue
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
                        rollout_action = np.asarray(
                            pack["actions"][int(step_idx), int(env_idx), :action_dim],
                            dtype=np.float32,
                        ).reshape(-1)
                        items.append(
                            {
                                "h": h_eval,
                                "source_h": h,
                                "env_idx": int(env_idx),
                                "fixed_env_seed": int(env_seeds[int(env_idx)]),
                                "eval_env_seed": 810000
                                + int(update_idx) * 100000
                                + int(env_idx) * 1000
                                + int(step_idx),
                                "step_idx": int(step_idx),
                                "update_idx": int(update_idx),
                                "action_dim": int(action_dim),
                                "first_action": rollout_action,
                                "v0": float(buffer_values_2d[int(step_idx), int(env_idx)]),
                                "gae_advantage": float(buffer_actor_adv_2d[int(step_idx), int(env_idx)]),
                            }
                        )
                for chunk_start in range(0, len(items), int(eval_chunk_size)):
                    chunk = items[chunk_start : chunk_start + int(eval_chunk_size)]
                    if not chunk:
                        continue
                    h_eval_list = [copy.deepcopy(item["h"]) for item in chunk]
                    eval_seeds = [int(item["eval_env_seed"]) for item in chunk]
                    algo_eval = None
                    vec_eval = None
                    try:
                        algo_eval, vec_eval = _build_algo(
                            cfg=cfg,
                            env_cfg=env_cfg,
                            frozen_h=None,
                            frozen_h_list=h_eval_list,
                            frozen_h_list_env_seeds=eval_seeds,
                            prior_mode="fixed_frozen_list",
                            device_obj=device_obj,
                            num_features=int(num_features),
                            n_envs=int(len(chunk)),
                            n_steps=int(max_horizon),
                            build_seed=int(seed) + 50000 + int(update_idx) * 1000 + int(chunk_start),
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
                        algo_eval.policy.load_state_dict(pre_policy_state)
                        algo_eval.policy.reset_rollout_cache(int(len(chunk)))
                        algo_eval.policy.set_training_mode(False)
                        obs = vec_eval._full_reset_batch(seeds=eval_seeds)
                        obs_dims = [int(item["h"].get("obs_dim", obs_slot_dim)) for item in chunk]
                        episode_starts = np.ones((int(len(chunk)),), dtype=np.float32)
                        meta: dict[str, Any] = {}
                        _add_prior_reward_component_meta_from_h_list(meta, h_eval_list, int(len(chunk)))
                        rewards_objective: list[list[float]] = [[] for _ in chunk]
                        rewards_env: list[list[float]] = [[] for _ in chunk]
                        dones_by_env: list[list[bool]] = [[] for _ in chunk]
                        value_at_horizon: dict[tuple[int, int], float] = {}
                        done_seen = [False for _ in chunk]
                        done_first_step = [None for _ in chunk]
                        for local_step in range(int(max_horizon)):
                            masks = vec_eval.action_masks().astype(np.float32, copy=False)
                            if local_step == 0:
                                action_arr = np.zeros((int(len(chunk)), int(action_slot_dim)), dtype=np.float32)
                                for local_idx, item in enumerate(chunk):
                                    action_dim = int(item["action_dim"])
                                    action_arr[local_idx, :action_dim] = np.asarray(
                                        item["first_action"],
                                        dtype=np.float32,
                                    )[:action_dim]
                                action_arr = (action_arr * masks).astype(np.float32, copy=False)
                            else:
                                obs_for_policy, _obs_stats = _apply_observation_normalization_if_enabled(
                                    algo_eval,
                                    np.asarray(obs, dtype=np.float32),
                                    obs_dims=obs_dims,
                                    obs_slot_dim=int(obs_slot_dim),
                                    stats_acc=None,
                                )
                                action_arr = _policy_action_from_obs(
                                    algo=algo_eval,
                                    policy_state_dict=pre_policy_state,
                                    obs_for_policy=obs_for_policy,
                                    episode_starts=episode_starts,
                                    masks=masks,
                                    deterministic=bool(deterministic_actions),
                                )
                            vec_eval.step_async(action_arr)
                            obs, reward, done, _infos = vec_eval.step_wait()
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
                                objective_reward = float(
                                    np.clip(
                                        float(reward_np[local_idx]) / max(float(reward_objective_scale), 1e-12),
                                        -float(reward_objective_clip),
                                        float(reward_objective_clip),
                                    )
                                )
                                if not done_seen[local_idx]:
                                    rewards_objective[local_idx].append(objective_reward)
                                    rewards_env[local_idx].append(reward_env)
                                    dones_by_env[local_idx].append(bool(done_np[local_idx]))
                                    if bool(done_np[local_idx]):
                                        done_seen[local_idx] = True
                                        done_first_step[local_idx] = int(local_step + 1)
                                step_rows.append(
                                    {
                                        "update_idx": int(item["update_idx"]),
                                        "env_idx": int(item["env_idx"]),
                                        "step_idx": int(item["step_idx"]),
                                        "rollout_step": int(local_step + 1),
                                        "reward_objective": objective_reward,
                                        "reward_env": reward_env,
                                        "done": bool(done_np[local_idx]),
                                    }
                                )
                            episode_starts = done_np.astype(np.float32, copy=False)
                            prefix_len = int(local_step + 1)
                            if prefix_len in set(horizon_values):
                                masks_h = vec_eval.action_masks().astype(np.float32, copy=False)
                                obs_for_value, _obs_stats = _apply_observation_normalization_if_enabled(
                                    algo_eval,
                                    np.asarray(obs, dtype=np.float32),
                                    obs_dims=obs_dims,
                                    obs_slot_dim=int(obs_slot_dim),
                                    stats_acc=None,
                                )
                                _actions_h, values_h = _policy_action_value_from_obs(
                                    algo=algo_eval,
                                    policy_state_dict=pre_policy_state,
                                    obs_for_policy=obs_for_value,
                                    episode_starts=episode_starts,
                                    masks=masks_h,
                                    deterministic=bool(deterministic_actions),
                                )
                                for local_idx, value_h in enumerate(values_h.tolist()):
                                    value_at_horizon[(local_idx, prefix_len)] = float(value_h)
                        for local_idx, item in enumerate(chunk):
                            for horizon in horizon_values:
                                effective_steps = int(min(int(horizon), len(rewards_objective[local_idx])))
                                obj_discounted = _discounted_prefix(
                                    rewards_objective[local_idx],
                                    float(algo.gamma),
                                    limit=effective_steps,
                                )
                                env_discounted = _discounted_prefix(
                                    rewards_env[local_idx],
                                    float(algo.gamma),
                                    limit=effective_steps,
                                )
                                done_before_or_at_h = (
                                    done_first_step[local_idx] is not None
                                    and int(done_first_step[local_idx]) <= int(horizon)
                                )
                                end_value = 0.0 if done_before_or_at_h else float(
                                    value_at_horizon.get((local_idx, int(horizon)), 0.0)
                                )
                                bootstrap = (
                                    0.0
                                    if done_before_or_at_h
                                    else float(float(algo.gamma) ** int(horizon) * end_value)
                                )
                                empirical_target = float(obj_discounted + bootstrap)
                                v0 = float(item["v0"])
                                empirical_adv = float(empirical_target - v0)
                                gae = float(item["gae_advantage"])
                                qrow = q_index.get(
                                    (
                                        int(item["update_idx"]),
                                        int(item["env_idx"]),
                                        int(item["step_idx"]),
                                        int(horizon)
                                        if int(horizon) in {5, 10, 20}
                                        else 20,
                                    )
                                )
                                result_rows.append(
                                    {
                                        "update_idx": int(item["update_idx"]),
                                        "env_idx": int(item["env_idx"]),
                                        "step_idx": int(item["step_idx"]),
                                        "horizon": int(horizon),
                                        "fixed_env_seed": int(item["fixed_env_seed"]),
                                        "delta_env_sum": _to_float(
                                            delta_index[int(item["env_idx"])]["delta_env_sum"]
                                        ),
                                        "mismatch_source_horizon": None
                                        if qrow is None
                                        else int(qrow["horizon"]),
                                        "source_adv_pos_rollout_below_median_objective": None
                                        if qrow is None
                                        else _to_bool(qrow["adv_pos_rollout_below_median_objective"]),
                                        "source_adv_neg_rollout_best_objective": None
                                        if qrow is None
                                        else _to_bool(qrow["adv_neg_rollout_best_objective"]),
                                        "gae_advantage": gae,
                                        "v0": v0,
                                        "gae_target": float(gae + v0),
                                        "empirical_discounted_objective_return": float(obj_discounted),
                                        "empirical_discounted_env_return": float(env_discounted),
                                        "end_value": float(end_value),
                                        "bootstrap_contribution": float(bootstrap),
                                        "empirical_target_bootstrap": float(empirical_target),
                                        "empirical_advantage_bootstrap": float(empirical_adv),
                                        "gae_minus_empirical_advantage": float(gae - empirical_adv),
                                        "gae_target_minus_empirical_target": float((gae + v0) - empirical_target),
                                        "empirical_target_minus_v0": float(empirical_target - v0),
                                        "effective_steps": int(effective_steps),
                                        "done_before_or_at_horizon": bool(done_before_or_at_h),
                                        "done_first_step": done_first_step[local_idx],
                                        "reward_objective_scale": float(reward_objective_scale),
                                    }
                                )
                    finally:
                        if vec_eval is not None:
                            vec_eval.close()
                        del algo_eval
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
            algo.train()
    finally:
        vec_env.close()
        del algo
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = {
        "artifact": "phase2_long_empirical_return_vs_gae_probe",
        "contract": (
            "For objective-space mismatch rows only, replay the same captured state and rollout first action, "
            "then close-loop under pre-policy for horizons 20/50/100/200. Compare empirical discounted "
            "objective return targets with the buffer GAE advantage used by PPO."
        ),
        "q_pair_csv": str(Path(q_pair_csv).expanduser().resolve()),
        "selected_csv": str(Path(selected_csv).expanduser().resolve()),
        "target_envs": [int(v) for v in target_envs],
        "target_env_seeds": [int(env_seeds[int(v)]) for v in target_envs],
        "capture_updates": sorted(int(v) for v in capture_set),
        "time_indices": [int(v) for v in time_indices],
        "horizons": [int(v) for v in horizon_values],
        "actor_baseline_mode": str(actor_baseline_mode),
        "actor_gae_lambda_override": None
        if actor_gae_lambda_override is None
        else float(actor_gae_lambda_override),
        "mismatch_keys_count": int(len(mismatch_keys)),
        "rows_count": int(len(result_rows)),
        "summary_by_env_horizon": _summarize(result_rows, ["env_idx", "horizon"]),
        "summary_by_env_source_horizon": _summarize(
            result_rows,
            ["env_idx", "mismatch_source_horizon", "horizon"],
        ),
        "files": {
            "summary_json": str(out_dir / "summary.json"),
            "result_rows_csv": str(out_dir / "long_return_vs_gae_rows.csv"),
            "step_rows_csv": str(out_dir / "long_return_step_rows.csv"),
            "progress_jsonl": str(out_dir / "progress.jsonl"),
        },
    }
    _write_json(out_dir / "summary.json", summary)
    _write_csv(out_dir / "long_return_vs_gae_rows.csv", result_rows)
    _write_csv(out_dir / "long_return_step_rows.csv", step_rows)
    with (out_dir / "progress.jsonl").open("w", encoding="utf-8") as handle:
        for row in progress_rows:
            handle.write(json.dumps(_json_safe(row), sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare long empirical return targets against GAE targets.")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--selected-csv", type=str, default=DEFAULT_SELECTED_CSV)
    parser.add_argument("--selected-rule", type=str, default="ratio10_q90p1_posp15")
    parser.add_argument("--q-pair-csv", type=str, default=DEFAULT_Q_PAIR_CSV)
    parser.add_argument("--delta-csv", type=str, default=DEFAULT_DELTA_CSV)
    parser.add_argument("--target-env-indices", type=str, default="34,9,36")
    parser.add_argument("--n-envs", type=int, default=64)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--updates", type=int, default=12)
    parser.add_argument("--capture-updates", type=str, default="0,5,11")
    parser.add_argument("--seed", type=int, default=4040)
    parser.add_argument("--time-start", type=int, default=64)
    parser.add_argument("--time-stride", type=int, default=16)
    parser.add_argument("--max-time-points", type=int, default=12)
    parser.add_argument("--horizons", type=str, default="20,50,100,200")
    parser.add_argument("--eval-chunk-size", type=int, default=64)
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
        q_pair_csv=args.q_pair_csv,
        delta_csv=args.delta_csv,
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
                    "result_rows_csv": summary["files"]["result_rows_csv"],
                    "rows_count": summary["rows_count"],
                    "mismatch_keys_count": summary["mismatch_keys_count"],
                }
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
