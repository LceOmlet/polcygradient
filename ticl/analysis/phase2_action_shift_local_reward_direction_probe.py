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
    "phase2_action_shift_local_reward_direction_probe_env9_36_0501"
)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_json_safe(row), sort_keys=True) + "\n")


def _policy_action_from_obs(
    *,
    algo,
    policy_state_dict: dict[str, torch.Tensor],
    obs_for_policy: np.ndarray,
    episode_starts: np.ndarray,
    masks: np.ndarray,
    deterministic: bool,
) -> np.ndarray:
    algo.policy.load_state_dict(policy_state_dict)
    algo.policy.reset_rollout_cache(int(obs_for_policy.shape[0]))
    algo.policy.set_training_mode(False)
    obs_t = torch.as_tensor(obs_for_policy, device=algo.device, dtype=torch.float32)
    starts_t = torch.as_tensor(episode_starts, device=algo.device, dtype=torch.float32)
    masks_t = torch.as_tensor(masks, device=algo.device, dtype=torch.float32)
    dummy_states = algo.policy._dummy_states(int(obs_t.shape[0]))
    with torch.no_grad():
        actions_t, _values_t, _log_probs_t, _ = algo.policy(
            obs_t,
            dummy_states,
            starts_t,
            deterministic=bool(deterministic),
            action_masks=masks_t,
        )
    return actions_t.detach().cpu().numpy().astype(np.float32, copy=False)


def _make_eval_items(
    *,
    pack: dict[str, np.ndarray],
    h_list: list[dict[str, Any]],
    env_indices: list[int],
    time_indices: list[int],
    update_idx: int,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for env_idx in env_indices:
        h = h_list[int(env_idx)]
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
            items.append(
                {
                    "h": h_eval,
                    "env_idx": int(env_idx),
                    "step_idx": int(step_idx),
                    "update_idx": int(update_idx),
                    "source_pack": f"pre_update_{int(update_idx)}",
                }
            )
    return items


def _first_action_modes(pre_action: np.ndarray, post_action: np.ndarray) -> dict[str, np.ndarray]:
    pre = np.asarray(pre_action, dtype=np.float32)
    post = np.asarray(post_action, dtype=np.float32)
    delta = post - pre
    return {
        "pre_first": pre,
        "post_shift_first": post,
        "mirror_shift_first": (pre - delta).astype(np.float32, copy=False),
        "half_post_shift_first": (pre + 0.5 * delta).astype(np.float32, copy=False),
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
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    horizon_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    horizon_set = {int(v) for v in horizons}
    if not items:
        return horizon_rows, step_rows, action_rows
    expanded: list[dict[str, Any]] = []
    for item in items:
        # Modes are materialized after computing pre/post actions on the reset state.
        expanded.append(item)
    for chunk_start in range(0, len(expanded), int(chunk_size)):
        base_chunk = expanded[chunk_start : chunk_start + int(chunk_size)]
        h_list = [copy.deepcopy(item["h"]) for item in base_chunk]
        env_seeds = [
            880000 + int(item["update_idx"]) * 100000 + int(item["env_idx"]) * 1000 + int(item["step_idx"])
            for item in base_chunk
        ]
        algo_eval = None
        vec_env = None
        try:
            algo_eval, vec_env = _build_algo(
                cfg=cfg,
                env_cfg=env_cfg,
                frozen_h=None,
                frozen_h_list=h_list,
                frozen_h_list_env_seeds=env_seeds,
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
            obs0 = vec_env._full_reset_batch(seeds=env_seeds)
            obs_dims = [int(h.get("obs_dim", obs_slot_dim)) for h in h_list]
            masks0 = vec_env.action_masks().astype(np.float32, copy=False)
            obs_for_policy0, _stats0 = _apply_observation_normalization_if_enabled(
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
                action_dim = int(h_list[local_idx]["action_dim"])
                modes = _first_action_modes(
                    pre_actions0[local_idx, :action_dim],
                    post_actions0[local_idx, :action_dim],
                )
                delta = post_actions0[local_idx, :action_dim] - pre_actions0[local_idx, :action_dim]
                delta_norm = float(np.linalg.norm(delta.astype(np.float64)))
                for mode, action_active in modes.items():
                    h_mode = copy.deepcopy(h_list[local_idx])
                    mode_items.append(
                        {
                            **item,
                            "h": h_mode,
                            "env_seed": int(env_seeds[local_idx]),
                            "first_action_mode": str(mode),
                            "first_action_active": np.asarray(action_active, dtype=np.float32).reshape(-1),
                            "action_dim": int(action_dim),
                            "pre_first_action_norm": float(
                                np.linalg.norm(pre_actions0[local_idx, :action_dim].astype(np.float64))
                            ),
                            "post_first_action_norm": float(
                                np.linalg.norm(post_actions0[local_idx, :action_dim].astype(np.float64))
                            ),
                            "post_minus_pre_first_action_norm": float(delta_norm),
                        }
                    )
                action_rows.append(
                    {
                        "update_idx": int(item["update_idx"]),
                        "env_idx": int(item["env_idx"]),
                        "step_idx": int(item["step_idx"]),
                        "pre_first_action_norm": float(
                            np.linalg.norm(pre_actions0[local_idx, :action_dim].astype(np.float64))
                        ),
                        "post_first_action_norm": float(
                            np.linalg.norm(post_actions0[local_idx, :action_dim].astype(np.float64))
                        ),
                        "post_minus_pre_first_action_norm": float(delta_norm),
                    }
                )

            for mode_start in range(0, len(mode_items), int(chunk_size)):
                chunk = mode_items[mode_start : mode_start + int(chunk_size)]
                mode_h_list = [copy.deepcopy(item["h"]) for item in chunk]
                mode_env_seeds = [int(item["env_seed"]) for item in chunk]
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
                                    item["first_action_active"], dtype=np.float32
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
                                    "done": bool(done_np[local_idx]),
                                    "first_action_delta_norm": float(item["post_minus_pre_first_action_norm"]),
                                }
                            )
                        episode_starts = done_np.astype(np.float32, copy=False)
                        prefix_len = int(local_step + 1)
                        if prefix_len in horizon_set:
                            for local_idx, item in enumerate(chunk):
                                env_rewards = np.asarray(
                                    reward_env_by_env[local_idx][:prefix_len], dtype=np.float64
                                )
                                total_rewards = np.asarray(
                                    reward_total_by_env[local_idx][:prefix_len], dtype=np.float64
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
                                        "reward_env_q90": float(np.quantile(env_rewards, 0.90)),
                                        "reward_env_positive_fraction": float(np.mean(env_rewards > 0.0)),
                                        "reward_total_sum": float(np.sum(total_rewards)),
                                        "done_count": int(np.sum(done_values)),
                                        "first_action_delta_norm": float(item["post_minus_pre_first_action_norm"]),
                                    }
                                )
                finally:
                    if vec_mode is not None:
                        vec_mode.close()
                    del algo_mode
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        finally:
            if vec_env is not None:
                vec_env.close()
            del algo_eval
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return horizon_rows, step_rows, action_rows


def _pair_direction_rows(horizon_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
        base = values.get("pre_first")
        post = values.get("post_shift_first")
        mirror = values.get("mirror_shift_first")
        half = values.get("half_post_shift_first")
        if base is None or post is None or mirror is None:
            continue
        post_minus_base = float(post["reward_env_sum"] - base["reward_env_sum"])
        mirror_minus_base = float(mirror["reward_env_sum"] - base["reward_env_sum"])
        half_minus_base = None if half is None else float(half["reward_env_sum"] - base["reward_env_sum"])
        out.append(
            {
                "update_idx": int(update_idx),
                "env_idx": int(env_idx),
                "step_idx": int(step_idx),
                "horizon": int(horizon),
                "base_reward_env_sum": float(base["reward_env_sum"]),
                "post_shift_reward_env_sum": float(post["reward_env_sum"]),
                "mirror_shift_reward_env_sum": float(mirror["reward_env_sum"]),
                "half_post_shift_reward_env_sum": None if half is None else float(half["reward_env_sum"]),
                "post_shift_minus_base": float(post_minus_base),
                "mirror_shift_minus_base": float(mirror_minus_base),
                "half_post_shift_minus_base": half_minus_base,
                "directional_asymmetry_post_minus_mirror": float(
                    post["reward_env_sum"] - mirror["reward_env_sum"]
                ),
                "post_shift_local_downhill": bool(post_minus_base < 0.0 and post_minus_base < mirror_minus_base),
                "mirror_shift_better_than_post": bool(mirror_minus_base > post_minus_base),
                "post_shift_worse_than_base": bool(post_minus_base < 0.0),
                "mirror_shift_better_than_base": bool(mirror_minus_base > 0.0),
                "base_done_count": int(base["done_count"]),
                "post_shift_done_count": int(post["done_count"]),
                "mirror_shift_done_count": int(mirror["done_count"]),
                "post_shift_minus_base_done_count": int(post["done_count"] - base["done_count"]),
                "first_action_delta_norm": float(base["first_action_delta_norm"]),
            }
        )
    return out


def _summary_by(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row.get(key) for key in keys), []).append(row)
    out: list[dict[str, Any]] = []
    for group_key, group_rows in sorted(grouped.items(), key=lambda item: tuple(str(v) for v in item[0])):
        record = {key: value for key, value in zip(keys, group_key)}
        downhill = [row for row in group_rows if bool(row["post_shift_local_downhill"])]
        post_worse = [row for row in group_rows if bool(row["post_shift_worse_than_base"])]
        mirror_better = [row for row in group_rows if bool(row["mirror_shift_better_than_base"])]
        record.update(
            {
                "count": int(len(group_rows)),
                "post_shift_minus_base": _stats([row["post_shift_minus_base"] for row in group_rows]),
                "mirror_shift_minus_base": _stats([row["mirror_shift_minus_base"] for row in group_rows]),
                "directional_asymmetry_post_minus_mirror": _stats(
                    [row["directional_asymmetry_post_minus_mirror"] for row in group_rows]
                ),
                "post_shift_local_downhill_fraction": float(len(downhill) / max(1, len(group_rows))),
                "post_shift_worse_than_base_fraction": float(len(post_worse) / max(1, len(group_rows))),
                "mirror_shift_better_than_base_fraction": float(len(mirror_better) / max(1, len(group_rows))),
                "first_action_delta_norm": _stats([row["first_action_delta_norm"] for row in group_rows]),
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
    )
    all_horizon_rows: list[dict[str, Any]] = []
    all_step_rows: list[dict[str, Any]] = []
    all_action_rows: list[dict[str, Any]] = []
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
            algo.train()
            post_policy_state = _clone_cpu_state_dict(algo.policy)
            progress_rows.append(
                {
                    "update_idx": int(update_idx),
                    "captured": bool(update_idx in capture_set),
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
                build_seed=int(seed) + int(update_idx) * 10000,
            )
            all_horizon_rows.extend(horizon_rows)
            all_step_rows.extend(step_rows)
            all_action_rows.extend(action_rows)
    finally:
        vec_env.close()
        del algo
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    pair_rows = _pair_direction_rows(all_horizon_rows)
    summary = {
        "artifact": "phase2_action_shift_local_reward_direction_probe",
        "interpretation_contract": (
            "At the same captured state, compute deterministic pre and post policy first actions. "
            "Evaluate only the first-action shift direction: baseline uses pre action, post_shift uses "
            "post action, mirror_shift uses pre - (post-pre), then all remaining steps use pre-policy. "
            "If post_shift is worse than baseline and worse than mirror_shift, PPO's action shift points "
            "locally downhill for the multi-step reward basin."
        ),
        "target_envs": [int(v) for v in target_envs],
        "target_env_seeds": [int(env_seeds[int(v)]) for v in target_envs],
        "target_frozen_h_paths": [str(h_paths[int(v)]) for v in target_envs],
        "capture_updates": sorted(int(v) for v in capture_set),
        "time_indices": [int(v) for v in time_indices],
        "horizons": [int(v) for v in horizon_values],
        "pair_rows_count": int(len(pair_rows)),
        "summary_by_env_horizon": _summary_by(pair_rows, ["env_idx", "horizon"]),
        "summary_by_update_env_horizon": _summary_by(pair_rows, ["update_idx", "env_idx", "horizon"]),
        "summary_all_by_horizon": _summary_by(pair_rows, ["horizon"]),
        "files": {
            "summary_json": str(out_dir / "summary.json"),
            "progress_jsonl": str(out_dir / "progress.jsonl"),
            "first_action_rows_csv": str(out_dir / "first_action_rows.csv"),
            "horizon_rows_csv": str(out_dir / "horizon_rows.csv"),
            "horizon_direction_pairs_csv": str(out_dir / "horizon_direction_pairs.csv"),
            "step_rows_csv": str(out_dir / "step_rows.csv"),
        },
    }
    _write_json(out_dir / "summary.json", summary)
    _write_jsonl(out_dir / "progress.jsonl", progress_rows)
    _write_csv(out_dir / "first_action_rows.csv", all_action_rows)
    _write_csv(out_dir / "horizon_rows.csv", all_horizon_rows)
    _write_csv(out_dir / "horizon_direction_pairs.csv", pair_rows)
    _write_csv(out_dir / "step_rows.csv", all_step_rows)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe whether PPO post-pre action shift is locally reward-downhill.")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--selected-csv", type=str, default=DEFAULT_SELECTED_CSV)
    parser.add_argument("--selected-rule", type=str, default="ratio10_q90p1_posp15")
    parser.add_argument("--target-env-indices", type=str, default="9,36")
    parser.add_argument("--n-envs", type=int, default=64)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--updates", type=int, default=12)
    parser.add_argument("--capture-updates", type=str, default="0,5,11")
    parser.add_argument("--seed", type=int, default=4040)
    parser.add_argument("--time-start", type=int, default=64)
    parser.add_argument("--time-stride", type=int, default=16)
    parser.add_argument("--max-time-points", type=int, default=12)
    parser.add_argument("--horizons", type=str, default="5,10,20")
    parser.add_argument("--eval-chunk-size", type=int, default=64)
    parser.add_argument(
        "--deterministic-actions",
        type=lambda x: str(x).strip().lower() not in {"0", "false", "no"},
        default=True,
    )
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
    )
    print(
        json.dumps(
            _json_safe(
                {
                    "summary_json": summary["files"]["summary_json"],
                    "target_envs": summary["target_envs"],
                    "pair_rows_count": summary["pair_rows_count"],
                    "summary_all_by_horizon": summary["summary_all_by_horizon"],
                }
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
