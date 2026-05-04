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
    _apply_reward_normalization_if_enabled,
    _build_algo,
    _collect_prior_policy_pack,
    _fill_existing_rollout_buffer_from_pack,
    _json_safe,
    _load_fixed_frozen_h_list_from_csv,
)
from ticl.analysis.phase2_worst_delta_action_state_counterfactual_probe import (  # noqa: E402
    _clone_cpu_state_dict,
    _make_horizon_start_item,
    _parse_horizons,
    _policy_actions_on_pack,
    _roll_horizon_policy_items,
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
from ticl.sb3_recurrent_ppo import (  # noqa: E402
    _normalize_advantages_with_mask,
    _resolve_actor_advantages,
)


DEFAULT_SELECTED_CSV = (
    "/home/chen/RLPFN/artifacts/"
    "phase2_reserve_probe_selected_ratio10_q90p1_posp15_n64_seed9090_0430/selected_envs.csv"
)
DEFAULT_OUTPUT_DIR = (
    "/home/chen/RLPFN/artifacts/"
    "phase2_ppo_update_horizon_bad_action_probe_env34_9_36_0501"
)


def _parse_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in str(value).split(",") if part.strip()]


def _parse_capture_updates(value: str, *, updates: int) -> set[int]:
    captures = {int(part.strip()) for part in str(value).split(",") if part.strip()}
    if not captures:
        captures = {0, max(0, int(updates) - 1)}
    return {idx for idx in captures if 0 <= int(idx) < int(updates)}


def _finite(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_json_safe(row), sort_keys=True) + "\n")


def _selected_time_indices(n_steps: int, *, start: int, stride: int, max_count: int) -> list[int]:
    out: list[int] = []
    for step_idx in range(max(1, int(start)), int(n_steps), max(1, int(stride))):
        out.append(int(step_idx))
        if len(out) >= int(max_count):
            break
    return out


def _clone_flat_rollout_data(algo) -> Any:
    batches = list(algo.rollout_buffer.get_gpu_flat(algo.batch_size))
    if len(batches) != 1:
        raise RuntimeError(f"Expected a single flat batch, got {len(batches)}")
    return batches[0]


def _eval_log_prob(algo, rollout_data) -> np.ndarray:
    with torch.no_grad():
        eval_outputs = algo.policy.evaluate_actions_with_hidden_flat(
            rollout_data.observations,
            rollout_data.actions,
            seq_lengths=rollout_data.seq_lengths,
            action_masks=rollout_data.action_masks,
        )
    return eval_outputs["log_prob"].detach().to(dtype=torch.float32).cpu().numpy().reshape(-1)


def _flat_update_rows_from_rollout(algo, rollout_data, post_log_prob: np.ndarray) -> list[dict[str, Any]]:
    objective_mask_t = rollout_data.objective_masks > 1e-8
    raw_adv_t = _resolve_actor_advantages(rollout_data).detach()
    normalized_adv_t = _normalize_advantages_with_mask(raw_adv_t, objective_mask_t, eps=1e-8).detach()
    old_log_prob_t = rollout_data.old_log_prob.detach()
    old_log_prob = old_log_prob_t.to(dtype=torch.float32).cpu().numpy().reshape(-1)
    raw_adv = raw_adv_t.to(dtype=torch.float32).cpu().numpy().reshape(-1)
    normalized_adv = normalized_adv_t.to(dtype=torch.float32).cpu().numpy().reshape(-1)
    objective_mask = objective_mask_t.detach().cpu().numpy().reshape(-1).astype(bool)
    flat_env_indices = np.asarray(rollout_data.flat_env_indices, dtype=np.int64).reshape(-1)
    flat_step_indices = np.asarray(rollout_data.flat_step_indices, dtype=np.int64).reshape(-1)
    returns = rollout_data.returns.detach().to(dtype=torch.float32).cpu().numpy().reshape(-1)
    old_values = rollout_data.old_values.detach().to(dtype=torch.float32).cpu().numpy().reshape(-1)
    post_log_prob = np.asarray(post_log_prob, dtype=np.float64).reshape(-1)
    logprob_delta = post_log_prob - old_log_prob.astype(np.float64)
    ratio = np.exp(np.clip(logprob_delta, -50.0, 50.0))
    rows: list[dict[str, Any]] = []
    for idx in range(int(flat_env_indices.shape[0])):
        rows.append(
            {
                "flat_idx": int(idx),
                "env_idx": int(flat_env_indices[idx]),
                "step_idx": int(flat_step_indices[idx]),
                "objective": bool(objective_mask[idx]),
                "actor_advantage_raw": float(raw_adv[idx]),
                "actor_advantage_standardized": float(normalized_adv[idx]),
                "old_log_prob": float(old_log_prob[idx]),
                "post_log_prob": float(post_log_prob[idx]),
                "logprob_delta": float(logprob_delta[idx]),
                "ratio": float(ratio[idx]),
                "return_target": float(returns[idx]),
                "old_value": float(old_values[idx]),
                "td_residual_proxy": float(returns[idx] - old_values[idx]),
                "ppo_would_boost_sampled_action": bool(objective_mask[idx] and raw_adv[idx] > 0.0 and ratio[idx] > 1.0),
            }
        )
    return rows


def _index_rows(rows: list[dict[str, Any]]) -> dict[tuple[int, int], dict[str, Any]]:
    out: dict[tuple[int, int], dict[str, Any]] = {}
    for row in rows:
        if not bool(row.get("objective", False)):
            continue
        out[(int(row["env_idx"]), int(row["step_idx"]))] = row
    return out


def _action_projection_rows(
    *,
    pack: dict[str, np.ndarray],
    h_list: list[dict[str, Any]],
    pre_actions: np.ndarray,
    post_actions: np.ndarray,
    env_indices: list[int],
    time_indices: list[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for env_idx in env_indices:
        action_dim = int(h_list[int(env_idx)]["action_dim"])
        for step_idx in time_indices:
            sampled = np.asarray(pack["actions"][int(step_idx), int(env_idx), :action_dim], dtype=np.float64)
            pre = np.asarray(pre_actions[int(step_idx), int(env_idx), :action_dim], dtype=np.float64)
            post = np.asarray(post_actions[int(step_idx), int(env_idx), :action_dim], dtype=np.float64)
            policy_delta = post - pre
            sampled_delta = sampled - pre
            policy_norm = float(np.linalg.norm(policy_delta))
            sampled_delta_norm = float(np.linalg.norm(sampled_delta))
            denom = max(policy_norm * sampled_delta_norm, 1e-12)
            rows.append(
                {
                    "env_idx": int(env_idx),
                    "step_idx": int(step_idx),
                    "sampled_action_norm": float(np.linalg.norm(sampled)),
                    "pre_policy_action_norm": float(np.linalg.norm(pre)),
                    "post_policy_action_norm": float(np.linalg.norm(post)),
                    "post_minus_pre_action_norm": policy_norm,
                    "sampled_minus_pre_action_norm": sampled_delta_norm,
                    "projection_on_sampled_delta": float(np.dot(policy_delta, sampled_delta) / max(sampled_delta_norm, 1e-12)),
                    "cosine_to_sampled_delta": float(np.dot(policy_delta, sampled_delta) / denom),
                }
            )
    return rows


def _make_horizon_items(
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
        for step_idx in time_indices:
            state = _state_at_step_from_pack(
                pack=pack,
                h=h,
                env_idx=int(env_idx),
                step_idx=int(step_idx),
            )
            if state is None:
                continue
            item = _make_horizon_start_item(
                h=h,
                env_seed=990000 + int(update_idx) * 100000 + int(env_idx) * 1000 + int(step_idx),
                initial_state=state,
                env_idx=int(env_idx),
                source_pack=f"pre_update_{int(update_idx)}",
                step_idx=int(step_idx),
            )
            item["source_env_seed"] = int(env_seeds[int(env_idx)])
            items.append(item)
    return items


def _pair_horizon_rows(horizon_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str, int, int], dict[str, dict[str, Any]]] = {}
    for row in horizon_rows:
        key = (
            int(row["env_idx"]),
            str(row["source_pack"]),
            int(row["start_step_idx"]),
            int(row["horizon"]),
        )
        grouped.setdefault(key, {})[str(row["policy_label"])] = row
    pair_rows: list[dict[str, Any]] = []
    for (env_idx, source_pack, step_idx, horizon), values in sorted(grouped.items()):
        pre = values.get("pre_update_policy")
        post = values.get("post_update_policy")
        if pre is None or post is None:
            continue
        pair_rows.append(
            {
                "env_idx": int(env_idx),
                "source_pack": str(source_pack),
                "step_idx": int(step_idx),
                "horizon": int(horizon),
                "pre_reward_env_sum": float(pre["reward_env_sum"]),
                "post_reward_env_sum": float(post["reward_env_sum"]),
                "post_minus_pre_reward_env_sum": float(post["reward_env_sum"] - pre["reward_env_sum"]),
                "pre_reward_env_q90": float(pre["reward_env_q90"]),
                "post_reward_env_q90": float(post["reward_env_q90"]),
                "post_minus_pre_reward_env_q90": float(post["reward_env_q90"] - pre["reward_env_q90"]),
                "pre_reward_env_positive_fraction": float(pre["reward_env_positive_fraction"]),
                "post_reward_env_positive_fraction": float(post["reward_env_positive_fraction"]),
                "post_minus_pre_reward_env_positive_fraction": float(
                    post["reward_env_positive_fraction"] - pre["reward_env_positive_fraction"]
                ),
                "pre_done_count": int(pre["done_count"]),
                "post_done_count": int(post["done_count"]),
                "post_minus_pre_done_count": int(post["done_count"] - pre["done_count"]),
            }
        )
    return pair_rows


def _pair_horizon_step_rows(step_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str, int, int], dict[str, dict[str, Any]]] = {}
    for row in step_rows:
        key = (
            int(row["env_idx"]),
            str(row["source_pack"]),
            int(row["start_step_idx"]),
            int(row["rollout_step"]),
        )
        grouped.setdefault(key, {})[str(row["policy_label"])] = row
    pair_rows: list[dict[str, Any]] = []
    for (env_idx, source_pack, step_idx, rollout_step), values in sorted(grouped.items()):
        pre = values.get("pre_update_policy")
        post = values.get("post_update_policy")
        if pre is None or post is None:
            continue
        pair_rows.append(
            {
                "env_idx": int(env_idx),
                "source_pack": str(source_pack),
                "step_idx": int(step_idx),
                "rollout_step": int(rollout_step),
                "pre_reward_env": float(pre["reward_env"]),
                "post_reward_env": float(post["reward_env"]),
                "post_minus_pre_reward_env": float(post["reward_env"] - pre["reward_env"]),
                "pre_reward_total": float(pre["reward_total"]),
                "post_reward_total": float(post["reward_total"]),
                "post_minus_pre_reward_total": float(post["reward_total"] - pre["reward_total"]),
                "pre_done": bool(pre["done"]),
                "post_done": bool(post["done"]),
                "post_minus_pre_done": int(bool(post["done"])) - int(bool(pre["done"])),
                "pre_action_norm": float(pre["action_norm"]),
                "post_action_norm": float(post["action_norm"]),
                "post_minus_pre_action_norm": float(post["action_norm"] - pre["action_norm"]),
                "pre_action_std": float(pre["action_std"]),
                "post_action_std": float(post["action_std"]),
                "post_minus_pre_action_std": float(post["action_std"] - pre["action_std"]),
                "pre_obs_before_norm": float(pre["obs_before_norm"]),
                "post_obs_before_norm": float(post["obs_before_norm"]),
                "post_minus_pre_obs_before_norm": float(post["obs_before_norm"] - pre["obs_before_norm"]),
                "pre_obs_delta_norm": float(pre["obs_delta_norm"]),
                "post_obs_delta_norm": float(post["obs_delta_norm"]),
                "post_minus_pre_obs_delta_norm": float(post["obs_delta_norm"] - pre["obs_delta_norm"]),
            }
        )
    return pair_rows


def _joined_rows(
    *,
    update_idx: int,
    env_indices: list[int],
    horizon_pairs: list[dict[str, Any]],
    flat_rows: list[dict[str, Any]],
    action_projection_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    flat_index = _index_rows(flat_rows)
    action_index = {(int(row["env_idx"]), int(row["step_idx"])): row for row in action_projection_rows}
    out: list[dict[str, Any]] = []
    env_set = {int(v) for v in env_indices}
    for hrow in horizon_pairs:
        key = (int(hrow["env_idx"]), int(hrow["step_idx"]))
        if int(hrow["env_idx"]) not in env_set:
            continue
        urow = flat_index.get(key)
        if urow is None:
            continue
        arow = action_index.get(key, {})
        post_minus_pre = float(hrow["post_minus_pre_reward_env_sum"])
        raw_adv = float(urow["actor_advantage_raw"])
        ratio = float(urow["ratio"])
        row = {
            "update_idx": int(update_idx),
            **hrow,
            **{f"update_{key}": value for key, value in urow.items() if key not in {"env_idx", "step_idx"}},
            **{f"action_{key}": value for key, value in arow.items() if key not in {"env_idx", "step_idx"}},
            "horizon_worse": bool(post_minus_pre < 0.0),
            "advantage_positive": bool(raw_adv > 0.0),
            "ratio_gt_1": bool(ratio > 1.0),
            "ppo_boosted_horizon_worse": bool(post_minus_pre < 0.0 and raw_adv > 0.0 and ratio > 1.0),
            "ppo_suppressed_horizon_worse": bool(post_minus_pre < 0.0 and raw_adv < 0.0 and ratio < 1.0),
        }
        out.append(row)
    return out


def _summary_by(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row.get(key) for key in keys), []).append(row)
    out: list[dict[str, Any]] = []
    for group_key, group_rows in sorted(grouped.items(), key=lambda item: tuple(str(v) for v in item[0])):
        horizon_delta = [row["post_minus_pre_reward_env_sum"] for row in group_rows]
        bad_rows = [row for row in group_rows if bool(row["horizon_worse"])]
        boosted_bad = [row for row in group_rows if bool(row["ppo_boosted_horizon_worse"])]
        ppo_objective = [
            float(row["update_actor_advantage_raw"]) * math.log(max(float(row["update_ratio"]), 1e-12))
            for row in group_rows
        ]
        record = {key: value for key, value in zip(keys, group_key)}
        record.update(
            {
                "count": int(len(group_rows)),
                "horizon_delta": _stats(horizon_delta),
                "horizon_worse_fraction": float(len(bad_rows) / max(1, len(group_rows))),
                "ppo_boosted_horizon_worse_fraction_all": float(len(boosted_bad) / max(1, len(group_rows))),
                "ppo_boosted_fraction_among_horizon_worse": float(len(boosted_bad) / max(1, len(bad_rows))),
                "advantage_raw_on_horizon_worse": _stats(
                    [row["update_actor_advantage_raw"] for row in bad_rows]
                ),
                "ratio_on_horizon_worse": _stats([row["update_ratio"] for row in bad_rows]),
                "policy_objective_increment_proxy": _stats(ppo_objective),
            }
        )
        if len(group_rows) >= 3:
            x = np.asarray([float(row["update_actor_advantage_raw"]) for row in group_rows], dtype=np.float64)
            y = np.asarray(horizon_delta, dtype=np.float64)
            if float(np.std(x)) > 1e-12 and float(np.std(y)) > 1e-12:
                record["corr_advantage_raw_vs_horizon_delta"] = float(np.corrcoef(x, y)[0, 1])
            z = np.asarray([float(row["update_logprob_delta"]) for row in group_rows], dtype=np.float64)
            if float(np.std(z)) > 1e-12 and float(np.std(y)) > 1e-12:
                record["corr_logprob_delta_vs_horizon_delta"] = float(np.corrcoef(z, y)[0, 1])
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
    )

    all_update_rows: list[dict[str, Any]] = []
    all_action_projection_rows: list[dict[str, Any]] = []
    all_horizon_rows: list[dict[str, Any]] = []
    all_horizon_pair_rows: list[dict[str, Any]] = []
    all_horizon_step_rows: list[dict[str, Any]] = []
    all_horizon_step_pair_rows: list[dict[str, Any]] = []
    all_joined_rows: list[dict[str, Any]] = []
    progress_rows: list[dict[str, Any]] = []
    try:
        for update_idx in range(int(updates)):
            pre_policy_state = _clone_cpu_state_dict(algo.policy)
            pack, values, log_probs, meta = _collect_prior_policy_pack(
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
            rollout_data = _clone_flat_rollout_data(algo)
            pre_eval_logprob = _eval_log_prob(algo, rollout_data)
            before_train_state = _clone_cpu_state_dict(algo.policy)
            algo.train()
            post_policy_state = _clone_cpu_state_dict(algo.policy)
            post_logprob = _eval_log_prob(algo, rollout_data)
            flat_rows = _flat_update_rows_from_rollout(algo, rollout_data, post_logprob)
            for row in flat_rows:
                row["update_idx"] = int(update_idx)
                row["pre_eval_minus_old_logprob"] = float(pre_eval_logprob[int(row["flat_idx"])] - row["old_log_prob"])
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

            pre_actions = _policy_actions_on_pack(
                algo=algo,
                policy_state_dict=before_train_state,
                pack=pack,
                deterministic=bool(deterministic_actions),
                torch_seed=int(seed) + 1000 + int(update_idx),
            )
            post_actions = _policy_actions_on_pack(
                algo=algo,
                policy_state_dict=post_policy_state,
                pack=pack,
                deterministic=bool(deterministic_actions),
                torch_seed=int(seed) + 1000 + int(update_idx),
            )
            projection_rows = _action_projection_rows(
                pack=pack,
                h_list=h_list,
                pre_actions=pre_actions,
                post_actions=post_actions,
                env_indices=target_envs,
                time_indices=time_indices,
            )
            for row in projection_rows:
                row["update_idx"] = int(update_idx)
            horizon_items = _make_horizon_items(
                pack=pack,
                h_list=h_list,
                env_seeds=env_seeds,
                env_indices=target_envs,
                time_indices=time_indices,
                update_idx=int(update_idx),
            )
            horizon_rows: list[dict[str, Any]] = []
            horizon_step_rows: list[dict[str, Any]] = []
            for policy_label, policy_state, seed_offset in (
                ("pre_update_policy", pre_policy_state, 101),
                ("post_update_policy", post_policy_state, 202),
            ):
                horizon_rows.extend(
                    _roll_horizon_policy_items(
                        cfg=cfg,
                        env_cfg=env_cfg,
                        items=horizon_items,
                        policy_state_dict=policy_state,
                        policy_label=str(policy_label),
                        device_obj=device_obj,
                        num_features=int(num_features),
                        obs_slot_dim=int(obs_slot_dim),
                        action_slot_dim=int(action_slot_dim),
                        max_horizon=int(max(horizon_values)),
                        horizons=horizon_values,
                        chunk_size=int(eval_chunk_size),
                        deterministic_actions=bool(deterministic_actions),
                        torch_seed=int(seed) + int(seed_offset) + int(update_idx) * 10000,
                        step_rows_out=horizon_step_rows,
                    )
                )
            for row in horizon_rows:
                row["update_idx"] = int(update_idx)
            for row in horizon_step_rows:
                row["update_idx"] = int(update_idx)
            horizon_pairs = _pair_horizon_rows(horizon_rows)
            for row in horizon_pairs:
                row["update_idx"] = int(update_idx)
            horizon_step_pairs = _pair_horizon_step_rows(horizon_step_rows)
            for row in horizon_step_pairs:
                row["update_idx"] = int(update_idx)
            joined = _joined_rows(
                update_idx=int(update_idx),
                env_indices=target_envs,
                horizon_pairs=horizon_pairs,
                flat_rows=flat_rows,
                action_projection_rows=projection_rows,
            )
            all_update_rows.extend(
                [
                    row
                    for row in flat_rows
                    if int(row["env_idx"]) in set(target_envs) and int(row["step_idx"]) in set(time_indices)
                ]
            )
            all_action_projection_rows.extend(projection_rows)
            all_horizon_rows.extend(horizon_rows)
            all_horizon_pair_rows.extend(horizon_pairs)
            all_horizon_step_rows.extend(horizon_step_rows)
            all_horizon_step_pair_rows.extend(horizon_step_pairs)
            all_joined_rows.extend(joined)
            algo.policy.load_state_dict(post_policy_state)
    finally:
        vec_env.close()
        del algo
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = {
        "artifact": "phase2_ppo_update_horizon_bad_action_probe",
        "interpretation_contract": (
            "For captured PPO updates, this probe joins actor advantage and post/pre log-prob ratio "
            "on the old rollout action with a same-state closed-loop post-policy minus pre-policy "
            "reward_env counterfactual over horizons 5/10/20. A row with advantage>0, ratio>1, and "
            "post-pre horizon delta<0 is direct evidence that the PPO update increased probability "
            "of an action/trajectory that is worse in the short multi-step counterfactual."
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
        "normalize_advantage_used_by_ppo": False,
        "joined_rows_count": int(len(all_joined_rows)),
        "summary_by_env_horizon": _summary_by(all_joined_rows, ["env_idx", "horizon"]),
        "summary_by_update_env_horizon": _summary_by(all_joined_rows, ["update_idx", "env_idx", "horizon"]),
        "summary_all_by_horizon": _summary_by(all_joined_rows, ["horizon"]),
        "files": {
            "summary_json": str(out_dir / "summary.json"),
            "progress_jsonl": str(out_dir / "progress.jsonl"),
            "update_rows_csv": str(out_dir / "update_rows.csv"),
            "action_projection_rows_csv": str(out_dir / "action_projection_rows.csv"),
            "horizon_rows_csv": str(out_dir / "horizon_rows.csv"),
            "horizon_pairs_csv": str(out_dir / "horizon_pairs.csv"),
            "horizon_step_rows_csv": str(out_dir / "horizon_step_rows.csv"),
            "horizon_step_pairs_csv": str(out_dir / "horizon_step_pairs.csv"),
            "joined_rows_csv": str(out_dir / "joined_rows.csv"),
        },
    }
    _write_json(out_dir / "summary.json", summary)
    _write_jsonl(out_dir / "progress.jsonl", progress_rows)
    _write_csv(out_dir / "update_rows.csv", all_update_rows)
    _write_csv(out_dir / "action_projection_rows.csv", all_action_projection_rows)
    _write_csv(out_dir / "horizon_rows.csv", all_horizon_rows)
    _write_csv(out_dir / "horizon_pairs.csv", all_horizon_pair_rows)
    _write_csv(out_dir / "horizon_step_rows.csv", all_horizon_step_rows)
    _write_csv(out_dir / "horizon_step_pairs.csv", all_horizon_step_pair_rows)
    _write_csv(out_dir / "joined_rows.csv", all_joined_rows)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Join PPO advantage/ratio with horizon counterfactual bad actions.")
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
                    "joined_rows_count": summary["joined_rows_count"],
                    "summary_all_by_horizon": summary["summary_all_by_horizon"],
                }
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
