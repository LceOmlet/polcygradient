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

from ticl.analysis.phase2_advantage_vs_empirical_q_probe import DEFAULT_SELECTED_CSV  # noqa: E402
from ticl.analysis.phase2_gym_prior_pack_training_runner import (  # noqa: E402
    _apply_reward_normalization_if_enabled,
    _build_algo,
    _collect_prior_policy_pack,
    _fill_existing_rollout_buffer_from_pack,
    _json_safe,
    _load_fixed_frozen_h_list_from_csv,
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
from ticl.sb3_recurrent_ppo import _resolve_actor_advantages  # noqa: E402


DEFAULT_Q_PAIR_CSV = (
    "/home/chen/RLPFN/artifacts/"
    "phase2_advantage_vs_empirical_q_probe_env34_9_36_0501/q_pair_rows.csv"
)
DEFAULT_DELTA_CSV = (
    "/home/chen/RLPFN/artifacts/"
    "phase2_delta_worst_env_analysis_0430/topology_ratio10_reserve_n64_delta_rank_with_seed.csv"
)
DEFAULT_OUTPUT_DIR = (
    "/home/chen/RLPFN/artifacts/"
    "phase2_gae_source_decomposition_probe_env34_9_36_0501"
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


def _index_q_rows(rows: list[dict[str, Any]]) -> dict[tuple[int, int, int, int], dict[str, Any]]:
    out: dict[tuple[int, int, int, int], dict[str, Any]] = {}
    for row in rows:
        key = (
            int(float(row["update_idx"])),
            int(float(row["env_idx"])),
            int(float(row["step_idx"])),
            int(float(row["horizon"])),
        )
        out[key] = row
    return out


def _index_delta_rows(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for row in rows:
        out[int(float(row["env_idx"]))] = row
    return out


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_json_safe(row), sort_keys=True) + "\n")


def _component_decomposition(
    *,
    rewards: np.ndarray,
    raw_rewards: np.ndarray,
    values: np.ndarray,
    episode_starts: np.ndarray,
    dones_last: np.ndarray,
    gamma: float,
    gae_lambda: float,
    step_idx: int,
    env_idx: int,
) -> dict[str, Any]:
    n_steps = int(rewards.shape[0])
    env = int(env_idx)
    step = int(step_idx)
    last_gae = 0.0
    direct_reward = 0.0
    raw_reward_same_weights = 0.0
    next_value = 0.0
    curr_value = 0.0
    weight = 1.0
    terms: list[dict[str, Any]] = []
    cut_by_done = False
    for t in range(step, n_steps):
        if t == n_steps - 1:
            nonterminal = 1.0 - float(bool(dones_last[env]))
            v_next = 0.0
        else:
            nonterminal = 1.0 - float(episode_starts[t + 1, env] > 0.5)
            v_next = float(values[t + 1, env])
        v_curr = float(values[t, env])
        reward_t = float(rewards[t, env])
        raw_reward_t = float(raw_rewards[t, env])
        gamma_eff = float(gamma) * nonterminal
        delta_t = reward_t + gamma_eff * v_next - v_curr
        direct_reward += weight * reward_t
        raw_reward_same_weights += weight * raw_reward_t
        next_value += weight * gamma_eff * v_next
        curr_value += weight * (-v_curr)
        terms.append(
            {
                "relative_step": int(t - step),
                "abs_step": int(t),
                "weight": float(weight),
                "reward_buffer": float(reward_t),
                "reward_raw_total": float(raw_reward_t),
                "value_curr": float(v_curr),
                "value_next": float(v_next),
                "next_nonterminal": float(nonterminal),
                "gamma_eff": float(gamma_eff),
                "delta": float(delta_t),
                "weighted_delta": float(weight * delta_t),
            }
        )
        last_gae += weight * delta_t
        if nonterminal <= 0.0:
            cut_by_done = True
            break
        weight *= float(gamma) * float(gae_lambda)
        if weight < 1e-8:
            break
    first_next_nonterminal = (
        1.0 - float(bool(dones_last[env]))
        if step == n_steps - 1
        else 1.0 - float(episode_starts[step + 1, env] > 0.5)
    )
    raw0 = float(raw_rewards[step, env])
    buffer0 = float(rewards[step, env])
    return {
        "gae_recomputed": float(last_gae),
        "gae_reward_buffer_contrib": float(direct_reward),
        "gae_reward_raw_total_same_weights_contrib": float(raw_reward_same_weights),
        "gae_reward_norm_delta_vs_raw_same_weights": float(direct_reward - raw_reward_same_weights),
        "gae_next_value_contrib": float(next_value),
        "gae_current_value_contrib": float(curr_value),
        "gae_value_net_contrib": float(next_value + curr_value),
        "gae_weighted_delta_sum": float(direct_reward + next_value + curr_value),
        "gae_effective_terms": int(len(terms)),
        "gae_cut_by_done_or_reset": bool(cut_by_done),
        "first_step_next_nonterminal": float(first_next_nonterminal),
        "first_step_done_or_reset_after": bool(first_next_nonterminal <= 0.0),
        "first_step_raw_reward_total": float(raw0),
        "first_step_buffer_reward": float(buffer0),
        "first_step_reward_norm_ratio": None if abs(raw0) < 1e-12 else float(buffer0 / raw0),
        "first_step_old_value": float(values[step, env]),
        "first_step_next_value": float(0.0 if step == n_steps - 1 else values[step + 1, env]),
        "terms_json": json.dumps(_json_safe(terms), sort_keys=True),
    }


def _flat_component_decomposition(
    *,
    rewards_flat: np.ndarray,
    raw_rewards_flat: np.ndarray,
    values_flat: np.ndarray,
    episode_starts_flat: np.ndarray,
    gamma: float,
    gae_lambda: float,
    flat_idx: int,
) -> dict[str, Any]:
    n = int(rewards_flat.shape[0])
    idx = int(flat_idx)
    direct_reward = 0.0
    raw_reward_same_weights = 0.0
    next_value = 0.0
    curr_value = 0.0
    last_gae = 0.0
    weight = 1.0
    terms: list[dict[str, Any]] = []
    cut_by_done = False
    for pos in range(idx, n):
        if pos == n - 1:
            nonterminal = 0.0
            v_next = 0.0
        else:
            nonterminal = 1.0 - float(episode_starts_flat[pos + 1] > 0.5)
            v_next = float(values_flat[pos + 1])
        v_curr = float(values_flat[pos])
        reward_t = float(rewards_flat[pos])
        raw_reward_t = float(raw_rewards_flat[pos])
        gamma_eff = float(gamma) * nonterminal
        delta_t = reward_t + gamma_eff * v_next - v_curr
        direct_reward += weight * reward_t
        raw_reward_same_weights += weight * raw_reward_t
        next_value += weight * gamma_eff * v_next
        curr_value += weight * (-v_curr)
        terms.append(
            {
                "relative_flat_pos": int(pos - idx),
                "flat_pos": int(pos),
                "weight": float(weight),
                "reward_buffer": float(reward_t),
                "reward_raw_total": float(raw_reward_t),
                "value_curr": float(v_curr),
                "value_next": float(v_next),
                "next_nonterminal": float(nonterminal),
                "gamma_eff": float(gamma_eff),
                "delta": float(delta_t),
                "weighted_delta": float(weight * delta_t),
            }
        )
        last_gae += weight * delta_t
        if nonterminal <= 0.0:
            cut_by_done = True
            break
        weight *= float(gamma) * float(gae_lambda)
        if weight < 1e-8:
            break
    first_next_nonterminal = 0.0 if idx == n - 1 else 1.0 - float(episode_starts_flat[idx + 1] > 0.5)
    raw0 = float(raw_rewards_flat[idx])
    buffer0 = float(rewards_flat[idx])
    return {
        "gae_recomputed": float(last_gae),
        "gae_reward_buffer_contrib": float(direct_reward),
        "gae_reward_raw_total_same_weights_contrib": float(raw_reward_same_weights),
        "gae_reward_norm_delta_vs_raw_same_weights": float(direct_reward - raw_reward_same_weights),
        "gae_next_value_contrib": float(next_value),
        "gae_current_value_contrib": float(curr_value),
        "gae_value_net_contrib": float(next_value + curr_value),
        "gae_weighted_delta_sum": float(direct_reward + next_value + curr_value),
        "gae_effective_terms": int(len(terms)),
        "gae_cut_by_done_or_reset": bool(cut_by_done),
        "first_step_next_nonterminal": float(first_next_nonterminal),
        "first_step_done_or_reset_after": bool(first_next_nonterminal <= 0.0),
        "first_step_raw_reward_total": float(raw0),
        "first_step_buffer_reward": float(buffer0),
        "first_step_reward_norm_ratio": None if abs(raw0) < 1e-12 else float(buffer0 / raw0),
        "first_step_old_value": float(values_flat[idx]),
        "first_step_next_value": float(0.0 if idx == n - 1 else values_flat[idx + 1]),
        "terms_json": json.dumps(_json_safe(terms), sort_keys=True),
    }


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


def _flat_rollout_tensor_to_step_env(
    values: Any,
    *,
    rollout_data: Any,
    n_steps: int,
    n_envs: int,
    name: str,
) -> np.ndarray:
    arr = values.detach().to(dtype=torch.float32).cpu().numpy().reshape(-1)
    env_indices = np.asarray(rollout_data.flat_env_indices, dtype=np.int64).reshape(-1)
    step_indices = np.asarray(rollout_data.flat_step_indices, dtype=np.int64).reshape(-1)
    if int(arr.shape[0]) != int(env_indices.shape[0]) or int(arr.shape[0]) != int(step_indices.shape[0]):
        raise ValueError(
            f"{name} flat length mismatch: values={arr.shape[0]} "
            f"env={env_indices.shape[0]} step={step_indices.shape[0]}"
        )
    out = np.full((int(n_steps), int(n_envs)), np.nan, dtype=np.float32)
    out[step_indices, env_indices] = arr.astype(np.float32, copy=False)
    if not np.isfinite(out).all():
        missing = int(np.size(out) - np.isfinite(out).sum())
        raise ValueError(f"{name} did not cover all [step, env] entries; missing={missing}")
    return out


def _summarize_joined(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row.get(key) for key in keys), []).append(row)
    out: list[dict[str, Any]] = []
    for group_key, group_rows in sorted(grouped.items(), key=lambda item: tuple(str(v) for v in item[0])):
        record = {key: value for key, value in zip(keys, group_key)}
        mismatch = [
            row
            for row in group_rows
            if bool(row.get("adv_pos_rollout_below_median", False))
            or bool(row.get("adv_neg_rollout_best", False))
        ]
        degraded = [row for row in group_rows if float(row.get("delta_env_sum", 0.0)) < 0.0]
        record.update(
            {
                "count": int(len(group_rows)),
                "mismatch_count": int(len(mismatch)),
                "degraded_count": int(len(degraded)),
                "delta_env_sum": _stats([row.get("delta_env_sum") for row in group_rows]),
                "actor_advantage_raw": _stats([row.get("actor_advantage_raw") for row in group_rows]),
                "q_rollout_minus_median_reward_env_sum": _stats(
                    [row.get("q_rollout_minus_median_reward_env_sum") for row in group_rows]
                ),
                "gae_reward_buffer_contrib": _stats(
                    [row.get("gae_reward_buffer_contrib") for row in group_rows]
                ),
                "gae_next_value_contrib": _stats(
                    [row.get("gae_next_value_contrib") for row in group_rows]
                ),
                "gae_current_value_contrib": _stats(
                    [row.get("gae_current_value_contrib") for row in group_rows]
                ),
                "gae_value_net_contrib": _stats(
                    [row.get("gae_value_net_contrib") for row in group_rows]
                ),
                "gae_effective_terms": _stats([row.get("gae_effective_terms") for row in group_rows]),
                "cut_fraction": float(
                    sum(bool(row.get("gae_cut_by_done_or_reset", False)) for row in group_rows)
                    / max(1, len(group_rows))
                ),
                "first_step_cut_fraction": float(
                    sum(bool(row.get("first_step_done_or_reset_after", False)) for row in group_rows)
                    / max(1, len(group_rows))
                ),
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
) -> dict[str, Any]:
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    q_index = _index_q_rows(_read_csv_rows(q_pair_csv))
    delta_index = _index_delta_rows(_read_csv_rows(delta_csv))
    device_obj = torch.device(device)
    target_envs = _parse_ints(target_env_indices)
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

    joined_rows: list[dict[str, Any]] = []
    progress_rows: list[dict[str, Any]] = []
    try:
        for update_idx in range(int(updates)):
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
            raw_pack_rewards = np.asarray(pack["rewards"], dtype=np.float32).copy()
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
            # Snapshot the logical step/env buffer before get_gpu_flat() mutates
            # several arrays into env-major flat form. Rewards intentionally stay
            # step/env inside the buffer, so mixing post-flatten arrays silently
            # breaks GAE decomposition.
            buffer_rewards_2d = _as_step_env_array(
                np.asarray(algo.rollout_buffer.rewards, dtype=np.float32).copy(),
                n_steps=int(n_steps),
                n_envs=int(n_envs),
                name="buffer_rewards",
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
            buffer_episode_starts_2d = _as_step_env_array(
                np.asarray(algo.rollout_buffer.episode_starts, dtype=np.float32).copy(),
                n_steps=int(n_steps),
                n_envs=int(n_envs),
                name="buffer_episode_starts",
            )
            rollout_data = _clone_flat_rollout_data(algo)
            pre_eval_logprob = _eval_log_prob(algo, rollout_data)
            algo.train()
            post_logprob = _eval_log_prob(algo, rollout_data)
            flat_rows = _flat_update_rows_from_rollout(algo, rollout_data, post_logprob)
            flat_batch_indices = getattr(rollout_data, "flat_batch_indices", None)
            flat_batch_indices_np = (
                np.asarray(flat_batch_indices, dtype=np.int64).reshape(-1)
                if flat_batch_indices is not None
                else np.arange(len(flat_rows), dtype=np.int64)
            )
            flat_index = {}
            for row in flat_rows:
                row["update_idx"] = int(update_idx)
                row["buffer_flat_idx"] = int(flat_batch_indices_np[int(row["flat_idx"])])
                row["pre_eval_minus_old_logprob"] = float(
                    pre_eval_logprob[int(row["flat_idx"])] - row["old_log_prob"]
                )
                if bool(row.get("objective", False)):
                    flat_index[(int(row["env_idx"]), int(row["step_idx"]))] = row
            progress_rows.append(
                {
                    "update_idx": int(update_idx),
                    "captured": bool(update_idx in capture_set),
                    "reward_norm": reward_norm_stats,
                }
            )
            if update_idx not in capture_set:
                continue
            raw_rewards = _as_step_env_array(
                np.asarray(raw_pack_rewards, dtype=np.float32),
                n_steps=int(n_steps),
                n_envs=int(n_envs),
                name="raw_pack_rewards",
            )
            dones_last = np.asarray(buffer_pack["dones"][-1], dtype=bool)
            for env_idx in target_envs:
                delta_row = delta_index[int(env_idx)]
                for step_idx in time_indices:
                    flat_row = flat_index.get((int(env_idx), int(step_idx)))
                    if flat_row is None:
                        continue
                    flat_idx = int(flat_row["buffer_flat_idx"])
                    decomp = _component_decomposition(
                        rewards=buffer_rewards_2d,
                        raw_rewards=raw_rewards,
                        values=buffer_values_2d,
                        episode_starts=buffer_episode_starts_2d,
                        dones_last=dones_last,
                        gamma=float(algo.gamma),
                        gae_lambda=float(algo.gae_lambda),
                        step_idx=int(step_idx),
                        env_idx=int(env_idx),
                    )
                    for horizon in (5, 10, 20):
                        qrow = q_index.get((int(update_idx), int(env_idx), int(step_idx), int(horizon)))
                        if qrow is None:
                            continue
                        row: dict[str, Any] = {
                            "update_idx": int(update_idx),
                            "env_idx": int(env_idx),
                            "step_idx": int(step_idx),
                            "horizon": int(horizon),
                            "fixed_env_seed": int(float(delta_row["fixed_env_seed"])),
                            "delta_env_sum": _to_float(delta_row["delta_env_sum"]),
                            "early_env_sum": _to_float(delta_row["early_env_sum"]),
                            "late_env_sum": _to_float(delta_row["late_env_sum"]),
                            "delta_q90": _to_float(delta_row["delta_q90"]),
                            "delta_pos_frac": _to_float(delta_row["delta_pos_frac"]),
                            "delta_done": _to_float(delta_row["delta_done"]),
                            "degraded_delta_lt_0": bool(_to_float(delta_row["delta_env_sum"]) < 0.0),
                            "degraded_delta_lt_minus10": bool(_to_float(delta_row["delta_env_sum"]) < -10.0),
                            "q_rollout_reward_env_sum": _to_float(qrow["rollout_reward_env_sum"]),
                            "q_pre_reward_env_sum": _to_float(qrow["pre_reward_env_sum"]),
                            "q_median_reward_env_sum": _to_float(qrow["median_reward_env_sum"]),
                            "q_best_reward_env_sum": _to_float(qrow["best_reward_env_sum"]),
                            "q_rollout_minus_median_reward_env_sum": _to_float(
                                qrow["rollout_minus_median_reward_env_sum"]
                            ),
                            "q_rollout_minus_best_reward_env_sum": _to_float(
                                qrow["rollout_minus_best_reward_env_sum"]
                            ),
                            "q_rollout_rank_desc": int(float(qrow["rollout_rank_desc"])),
                            "q_best_mode": str(qrow["best_mode"]),
                            "adv_pos_rollout_below_median": _to_bool(qrow["adv_pos_rollout_below_median"]),
                            "adv_neg_rollout_best": _to_bool(qrow["adv_neg_rollout_best"]),
                            "actor_advantage_raw": float(flat_row["actor_advantage_raw"]),
                            "rollout_data_flat_idx": int(flat_row["flat_idx"]),
                            "buffer_flat_idx": int(flat_idx),
                            "actor_advantage_from_buffer": float(buffer_actor_adv_2d[int(step_idx), int(env_idx)]),
                            "actor_advantage_recompute_abs_err": float(
                                abs(
                                    float(decomp["gae_recomputed"])
                                    - float(buffer_actor_adv_2d[int(step_idx), int(env_idx)])
                                )
                            ),
                            "old_value": float(flat_row["old_value"]),
                            "return_target": float(flat_row["return_target"]),
                            "td_residual_proxy": float(flat_row["td_residual_proxy"]),
                            "ratio": float(flat_row["ratio"]),
                            "logprob_delta": float(flat_row["logprob_delta"]),
                            "pre_eval_minus_old_logprob": float(flat_row["pre_eval_minus_old_logprob"]),
                            "reward_norm_enabled": bool(reward_norm_stats.get("enabled", False)),
                            "reward_norm_ret_rms_var": _to_float(reward_norm_stats.get("ret_rms_var")),
                            "reward_norm_buffer_reward_std": _to_float(
                                (reward_norm_stats.get("buffer_reward") or {}).get("std")
                            ),
                            "reward_norm_raw_reward_std": _to_float(
                                (reward_norm_stats.get("raw_reward") or {}).get("std")
                            ),
                        }
                        row.update(decomp)
                        joined_rows.append(row)
    finally:
        vec_env.close()
        del algo
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = {
        "artifact": "phase2_gae_source_decomposition_probe",
        "outcome_contract": (
            "Failure is linked to post-train delta_env_sum = late_env_sum - early_env_sum, "
            "not absolute return. Rows join mismatch evidence with delta_env_sum and decompose "
            "actor GAE into reward-normalized buffer reward, value/bootstrap, old value, and done/reset cuts."
        ),
        "selected_csv": str(Path(selected_csv).expanduser().resolve()),
        "q_pair_csv": str(Path(q_pair_csv).expanduser().resolve()),
        "delta_csv": str(Path(delta_csv).expanduser().resolve()),
        "target_envs": [int(v) for v in target_envs],
        "target_env_seeds": [int(env_seeds[int(v)]) for v in target_envs],
        "n_steps": int(n_steps),
        "updates": int(updates),
        "capture_updates": sorted(int(v) for v in capture_set),
        "time_indices": [int(v) for v in time_indices],
        "joined_rows_count": int(len(joined_rows)),
        "summary_by_env_horizon": _summarize_joined(joined_rows, ["env_idx", "horizon"]),
        "summary_by_delta_degraded_horizon": _summarize_joined(
            joined_rows,
            ["degraded_delta_lt_0", "horizon"],
        ),
        "files": {
            "summary_json": str(out_dir / "summary.json"),
            "progress_jsonl": str(out_dir / "progress.jsonl"),
            "joined_rows_csv": str(out_dir / "joined_gae_q_delta_rows.csv"),
        },
    }
    _write_json(out_dir / "summary.json", summary)
    _write_jsonl(out_dir / "progress.jsonl", progress_rows)
    _write_csv(out_dir / "joined_gae_q_delta_rows.csv", joined_rows)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Decompose actor GAE source terms and join to post-train delta.")
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
    )
    print(
        json.dumps(
            _json_safe(
                {
                    "summary_json": summary["files"]["summary_json"],
                    "joined_rows_csv": summary["files"]["joined_rows_csv"],
                    "joined_rows_count": summary["joined_rows_count"],
                    "summary_by_env_horizon": summary["summary_by_env_horizon"],
                }
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
