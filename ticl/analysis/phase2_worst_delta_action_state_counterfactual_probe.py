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
    _build_algo,
    _collect_prior_policy_pack,
    _json_safe,
    _load_fixed_frozen_h_list_from_csv,
    _reward_component_arrays_from_pack,
    _run_update_from_pack,
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
DEFAULT_DELTA_CSV = (
    "/home/chen/RLPFN/artifacts/"
    "phase2_delta_worst_env_analysis_0430/topology_ratio10_reserve_n64_delta_rank_with_seed.csv"
)
DEFAULT_OUTPUT_DIR = (
    "/home/chen/RLPFN/artifacts/"
    "phase2_worst_delta_action_state_counterfactual_probe_0430"
)


def _finite(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _stats(values: list[Any]) -> dict[str, Any]:
    arr = np.asarray([v for v in (_finite(v) for v in values) if v is not None], dtype=np.float64)
    if arr.size == 0:
        return {"count": 0, "mean": None, "q10": None, "q50": None, "q90": None, "min": None, "max": None}
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "q10": float(np.quantile(arr, 0.10)),
        "q50": float(np.quantile(arr, 0.50)),
        "q90": float(np.quantile(arr, 0.90)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_json_safe(row), sort_keys=True) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(_json_safe(row))


def _load_worst_env_indices(delta_csv: str | Path, limit: int) -> list[int]:
    rows: list[tuple[float, int]] = []
    with Path(delta_csv).expanduser().open("r", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            env_idx = int(float(raw["env_idx"]))
            delta = _finite(raw.get("delta_env_sum"))
            if delta is None:
                continue
            rows.append((float(delta), env_idx))
    rows.sort(key=lambda pair: pair[0])
    return [env_idx for _delta, env_idx in rows[: int(limit)]]


def _parse_env_indices(value: str, *, delta_csv: str | Path, limit: int) -> list[int]:
    if str(value).strip():
        return [int(part.strip()) for part in str(value).split(",") if part.strip()]
    return _load_worst_env_indices(delta_csv, int(limit))


def _parse_horizons(value: str) -> list[int]:
    horizons = sorted({int(part.strip()) for part in str(value).split(",") if part.strip()})
    if not horizons or min(horizons) <= 0:
        raise ValueError(f"Expected positive comma-separated horizons, got {value!r}")
    return horizons


def _clone_cpu_state_dict(module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def _policy_actions_on_pack(
    *,
    algo,
    policy_state_dict: dict[str, torch.Tensor],
    pack: dict[str, np.ndarray],
    deterministic: bool,
    torch_seed: int,
) -> np.ndarray:
    algo.policy.load_state_dict(policy_state_dict)
    algo.policy.reset_rollout_cache(int(pack["tokens"].shape[1]))
    algo.policy.set_training_mode(False)
    torch.manual_seed(int(torch_seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(torch_seed))
    out = np.zeros_like(pack["actions"], dtype=np.float32)
    for step_idx in range(int(pack["tokens"].shape[0])):
        obs_t = torch.as_tensor(pack["tokens"][step_idx], device=algo.device, dtype=torch.float32)
        starts_t = torch.as_tensor(pack["episode_starts"][step_idx], device=algo.device, dtype=torch.float32)
        masks_t = torch.as_tensor(pack["action_masks"][step_idx], device=algo.device, dtype=torch.float32)
        dummy_states = algo.policy._dummy_states(int(obs_t.shape[0]))
        with torch.no_grad():
            actions_t, _values_t, _log_probs_t, _ = algo.policy(
                obs_t,
                dummy_states,
                starts_t,
                deterministic=bool(deterministic),
                action_masks=masks_t,
            )
        out[step_idx] = actions_t.detach().cpu().numpy().astype(np.float32, copy=False)
    return out


def _state_at_step_from_pack(
    *,
    pack: dict[str, np.ndarray],
    h: dict[str, Any],
    env_idx: int,
    step_idx: int,
) -> np.ndarray | None:
    state_dim = int(h["state_dim"])
    if int(step_idx) <= 0:
        # The training run sampled the reset state inside the vectorized env. It is not
        # persisted in the frozen_h, so exact same-state replay starts from t>=1.
        return None
    state = np.asarray(pack["next_state_targets"][int(step_idx) - 1, int(env_idx), :state_dim], dtype=np.float32)
    mask = np.asarray(pack["next_state_masks"][int(step_idx) - 1, int(env_idx), :state_dim], dtype=np.float32)
    if state.shape[0] != state_dim or float(np.sum(mask > 0.5)) < max(1, state_dim // 2):
        return None
    return state.copy()


def _make_eval_item(
    *,
    h: dict[str, Any],
    env_seed: int,
    initial_state: np.ndarray,
    action: np.ndarray,
    action_dim: int,
    action_slot_dim: int,
    env_idx: int,
    source_pack: str,
    step_idx: int,
    action_kind: str,
) -> dict[str, Any]:
    h_eval = copy.deepcopy(h)
    h_eval["initial_state"] = np.asarray(initial_state, dtype=np.float32).reshape(-1).tolist()
    action_full = np.zeros((int(action_slot_dim),), dtype=np.float32)
    copy_dim = int(min(int(action_dim), int(action.shape[0]), int(action_slot_dim)))
    action_full[:copy_dim] = np.asarray(action[:copy_dim], dtype=np.float32)
    return {
        "h": h_eval,
        "env_seed": int(env_seed),
        "action": action_full,
        "env_idx": int(env_idx),
        "source_pack": str(source_pack),
        "step_idx": int(step_idx),
        "action_kind": str(action_kind),
    }


def _eval_one_step_items(
    *,
    cfg: dict[str, Any],
    env_cfg: dict[str, Any],
    items: list[dict[str, Any]],
    device_obj: torch.device,
    num_features: int,
    action_slot_dim: int,
    chunk_size: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not items:
        return rows
    for chunk_start in range(0, len(items), int(chunk_size)):
        chunk = items[chunk_start : chunk_start + int(chunk_size)]
        h_list = [copy.deepcopy(item["h"]) for item in chunk]
        env_seeds = [int(item["env_seed"]) for item in chunk]
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
                n_envs=int(len(chunk)),
                n_steps=1,
                build_seed=777001 + int(chunk_start),
                sb3_reward_normalization_enabled=False,
                sb3_observation_normalization_enabled=False,
                sb3_observation_normalization_clip=10.0,
                sb3_observation_normalization_epsilon=1e-8,
                gain_min=0.0,
                topology_state_gain_min=0.0,
                topology_action_gain_min=0.0,
                topology_state_to_action_ratio_max=0.0,
                topology_max_attempts=4096,
                fixed_env_group_across_updates=True,
            )
            _obs = vec_env._full_reset_batch(seeds=env_seeds)
            masks = vec_env.action_masks().astype(np.float32, copy=False)
            action_arr = np.stack([item["action"] for item in chunk], axis=0).astype(np.float32, copy=False)
            action_arr = action_arr * masks
            vec_env.step_async(action_arr)
            _next_obs, reward, done, infos = vec_env.step_wait()
            meta: dict[str, Any] = {}
            _add_prior_reward_component_meta_from_h_list(meta, h_list, int(len(chunk)))
            for local_idx, item in enumerate(chunk):
                action_dim = int(h_list[local_idx]["action_dim"])
                active_action = action_arr[local_idx : local_idx + 1, :action_dim]
                active_mask = masks[local_idx : local_idx + 1, :action_dim]
                components, component_meta = _reward_component_arrays_from_pack(
                    arm="prior",
                    reward=np.asarray([float(reward[local_idx])], dtype=np.float64),
                    active_actions=active_action.astype(np.float64, copy=False),
                    active_action_masks=active_mask.astype(np.float64, copy=False),
                    meta=meta,
                    env_idx=int(local_idx),
                )
                reward_env = components.get("reward_env", np.asarray([float(reward[local_idx])], dtype=np.float64))
                rows.append(
                    {
                        "env_idx": int(item["env_idx"]),
                        "source_pack": str(item["source_pack"]),
                        "step_idx": int(item["step_idx"]),
                        "action_kind": str(item["action_kind"]),
                        "env_seed": int(item["env_seed"]),
                        "reward_total": float(reward[local_idx]),
                        "reward_env": float(np.asarray(reward_env).reshape(-1)[0]),
                        "done": bool(done[local_idx]),
                        "truncated": bool((infos[local_idx] or {}).get("TimeLimit.truncated", False)),
                        "component_contract_exact": bool(
                            component_meta.get("contract_exact_for_current_none_transform_runs", False)
                        ),
                    }
                )
        finally:
            if vec_env is not None:
                vec_env.close()
            del algo_eval
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return rows


def _make_horizon_start_item(
    *,
    h: dict[str, Any],
    env_seed: int,
    initial_state: np.ndarray,
    env_idx: int,
    source_pack: str,
    step_idx: int,
) -> dict[str, Any]:
    h_eval = copy.deepcopy(h)
    h_eval["initial_state"] = np.asarray(initial_state, dtype=np.float32).reshape(-1).tolist()
    return {
        "h": h_eval,
        "env_seed": int(env_seed),
        "env_idx": int(env_idx),
        "source_pack": str(source_pack),
        "step_idx": int(step_idx),
    }


def _roll_horizon_policy_items(
    *,
    cfg: dict[str, Any],
    env_cfg: dict[str, Any],
    items: list[dict[str, Any]],
    policy_state_dict: dict[str, torch.Tensor],
    policy_label: str,
    device_obj: torch.device,
    num_features: int,
    obs_slot_dim: int,
    action_slot_dim: int,
    max_horizon: int,
    horizons: list[int],
    chunk_size: int,
    deterministic_actions: bool,
    torch_seed: int,
    step_rows_out: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not items:
        return rows
    horizon_set = {int(v) for v in horizons}
    for chunk_start in range(0, len(items), int(chunk_size)):
        chunk = items[chunk_start : chunk_start + int(chunk_size)]
        h_list = [copy.deepcopy(item["h"]) for item in chunk]
        env_seeds = [int(item["env_seed"]) for item in chunk]
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
                n_envs=int(len(chunk)),
                n_steps=int(max_horizon),
                build_seed=int(torch_seed) + int(chunk_start),
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
            algo_eval.policy.load_state_dict(policy_state_dict)
            algo_eval.policy.reset_rollout_cache(int(len(chunk)))
            algo_eval.policy.set_training_mode(False)
            torch.manual_seed(int(torch_seed) + int(chunk_start))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(torch_seed) + int(chunk_start))
            obs = vec_env._full_reset_batch(seeds=env_seeds)
            obs_dims = [int(h.get("obs_dim", obs_slot_dim)) for h in h_list]
            episode_starts = np.ones((int(len(chunk)),), dtype=np.float32)
            reward_env_by_env: list[list[float]] = [[] for _ in chunk]
            reward_total_by_env: list[list[float]] = [[] for _ in chunk]
            done_by_env: list[list[bool]] = [[] for _ in chunk]
            meta: dict[str, Any] = {}
            _add_prior_reward_component_meta_from_h_list(meta, h_list, int(len(chunk)))
            for local_step in range(int(max_horizon)):
                masks = vec_env.action_masks().astype(np.float32, copy=False)
                obs_before = np.asarray(obs, dtype=np.float32)
                obs_for_policy, _obs_stats = _apply_observation_normalization_if_enabled(
                    algo_eval,
                    obs_before,
                    obs_dims=obs_dims,
                    obs_slot_dim=int(obs_slot_dim),
                    stats_acc=None,
                )
                obs_t = torch.as_tensor(obs_for_policy, device=algo_eval.device, dtype=torch.float32)
                starts_t = torch.as_tensor(episode_starts, device=algo_eval.device, dtype=torch.float32)
                masks_t = torch.as_tensor(masks, device=algo_eval.device, dtype=torch.float32)
                dummy_states = algo_eval.policy._dummy_states(int(obs_t.shape[0]))
                with torch.no_grad():
                    actions_t, _values_t, _log_probs_t, _ = algo_eval.policy(
                        obs_t,
                        dummy_states,
                        starts_t,
                        deterministic=bool(deterministic_actions),
                        action_masks=masks_t,
                    )
                action_arr = actions_t.detach().cpu().numpy().astype(np.float32, copy=False)
                vec_env.step_async(action_arr)
                obs, reward, done, infos = vec_env.step_wait()
                obs_after = np.asarray(obs, dtype=np.float32)
                reward_np = np.asarray(reward, dtype=np.float64)
                done_np = np.asarray(done, dtype=bool)
                for local_idx, item in enumerate(chunk):
                    action_dim = int(h_list[local_idx]["action_dim"])
                    obs_dim = int(h_list[local_idx].get("obs_dim", obs_slot_dim))
                    obs_dim = int(max(0, min(obs_dim, int(obs_slot_dim), int(obs_before.shape[1]))))
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
                    reward_env = components.get(
                        "reward_env",
                        np.asarray([float(reward_np[local_idx])], dtype=np.float64),
                    )
                    reward_env_by_env[local_idx].append(float(np.asarray(reward_env).reshape(-1)[0]))
                    reward_total_by_env[local_idx].append(float(reward_np[local_idx]))
                    done_by_env[local_idx].append(bool(done_np[local_idx]))
                    if step_rows_out is not None:
                        active_action_flat = np.asarray(active_action, dtype=np.float64).reshape(-1)
                        obs_before_active = (
                            np.asarray(obs_before[local_idx, :obs_dim], dtype=np.float64).reshape(-1)
                            if obs_dim > 0
                            else np.asarray([], dtype=np.float64)
                        )
                        obs_after_active = (
                            np.asarray(obs_after[local_idx, :obs_dim], dtype=np.float64).reshape(-1)
                            if obs_dim > 0
                            else np.asarray([], dtype=np.float64)
                        )
                        step_rows_out.append(
                            {
                                "env_idx": int(item["env_idx"]),
                                "env_seed": int(item["env_seed"]),
                                "source_pack": str(item["source_pack"]),
                                "start_step_idx": int(item["step_idx"]),
                                "policy_label": str(policy_label),
                                "rollout_step": int(local_step + 1),
                                "reward_env": float(np.asarray(reward_env).reshape(-1)[0]),
                                "reward_total": float(reward_np[local_idx]),
                                "done": bool(done_np[local_idx]),
                                "action_norm": float(np.linalg.norm(active_action_flat))
                                if active_action_flat.size
                                else 0.0,
                                "action_mean": float(np.mean(active_action_flat))
                                if active_action_flat.size
                                else 0.0,
                                "action_std": float(np.std(active_action_flat))
                                if active_action_flat.size
                                else 0.0,
                                "action_linf": float(np.max(np.abs(active_action_flat)))
                                if active_action_flat.size
                                else 0.0,
                                "obs_before_norm": float(np.linalg.norm(obs_before_active))
                                if obs_before_active.size
                                else 0.0,
                                "obs_after_norm": float(np.linalg.norm(obs_after_active))
                                if obs_after_active.size
                                else 0.0,
                                "obs_delta_norm": float(np.linalg.norm(obs_after_active - obs_before_active))
                                if obs_after_active.size and obs_before_active.size
                                else 0.0,
                            }
                        )
                episode_starts = done_np.astype(np.float32, copy=False)
                prefix_len = int(local_step + 1)
                if prefix_len in horizon_set:
                    for local_idx, item in enumerate(chunk):
                        env_rewards = reward_env_by_env[local_idx][:prefix_len]
                        total_rewards = reward_total_by_env[local_idx][:prefix_len]
                        done_values = done_by_env[local_idx][:prefix_len]
                        rows.append(
                            {
                                "env_idx": int(item["env_idx"]),
                                "env_seed": int(item["env_seed"]),
                                "source_pack": str(item["source_pack"]),
                                "start_step_idx": int(item["step_idx"]),
                                "policy_label": str(policy_label),
                                "horizon": int(prefix_len),
                                "reward_env_sum": float(np.sum(env_rewards)),
                                "reward_env_mean": float(np.mean(env_rewards)),
                                "reward_env_q10": float(np.quantile(env_rewards, 0.10)),
                                "reward_env_q50": float(np.quantile(env_rewards, 0.50)),
                                "reward_env_q90": float(np.quantile(env_rewards, 0.90)),
                                "reward_env_positive_fraction": float(np.mean(np.asarray(env_rewards) > 0.0)),
                                "reward_total_sum": float(np.sum(total_rewards)),
                                "done_count": int(np.sum(done_values)),
                            }
                        )
        finally:
            if vec_env is not None:
                vec_env.close()
            del algo_eval
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return rows


def run_probe(
    *,
    output_dir: str,
    device: str,
    selected_csv: str,
    selected_rule: str,
    delta_csv: str,
    target_env_indices: str,
    target_limit: int,
    n_envs: int,
    n_steps: int,
    updates: int,
    seed: int,
    time_stride: int,
    max_time_points: int,
    horizons: str,
    eval_chunk_size: int,
    deterministic_actions: bool,
) -> dict[str, Any]:
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    device_obj = torch.device(device)
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
    target_envs = _parse_env_indices(target_env_indices, delta_csv=delta_csv, limit=int(target_limit))
    horizon_values = _parse_horizons(horizons)
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
    early_policy_state = _clone_cpu_state_dict(algo.policy)
    early_pack = None
    late_policy_state = None
    late_pack = None
    progress_rows: list[dict[str, Any]] = []
    try:
        for update_idx in range(int(updates)):
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
            if update_idx == 0:
                early_pack = {key: np.asarray(value).copy() for key, value in pack.items()}
            if update_idx == int(updates) - 1:
                late_policy_state = _clone_cpu_state_dict(algo.policy)
                late_pack = {key: np.asarray(value).copy() for key, value in pack.items()}
            update, bridge = _run_update_from_pack(
                algo=algo,
                pack=pack,
                values=values,
                log_probs=log_probs,
                num_features=int(num_features),
                action_slot_dim=int(action_slot_dim),
                next_state_target_dim=int(next_state_target_dim),
                single_eval_pos=64,
            )
            progress_rows.append(
                {
                    "update_idx": int(update_idx),
                    "reward_mean": float(np.mean(pack["rewards"])),
                    "target_env_reward_sum_mean": float(
                        np.mean([np.sum(pack["rewards"][:, int(env_idx)]) for env_idx in target_envs])
                    ),
                    "update_exception": update.get("exception"),
                    "bridge_reward_norm": bridge.get("reward_norm"),
                }
            )
        if early_pack is None or late_pack is None or late_policy_state is None:
            raise RuntimeError("Failed to capture early/late packs and policy states.")

        early_actions_on_early = _policy_actions_on_pack(
            algo=algo,
            policy_state_dict=early_policy_state,
            pack=early_pack,
            deterministic=bool(deterministic_actions),
            torch_seed=int(seed) + 17,
        )
        late_actions_on_early = _policy_actions_on_pack(
            algo=algo,
            policy_state_dict=late_policy_state,
            pack=early_pack,
            deterministic=bool(deterministic_actions),
            torch_seed=int(seed) + 17,
        )
        early_actions_on_late = _policy_actions_on_pack(
            algo=algo,
            policy_state_dict=early_policy_state,
            pack=late_pack,
            deterministic=bool(deterministic_actions),
            torch_seed=int(seed) + 23,
        )
        late_actions_on_late = _policy_actions_on_pack(
            algo=algo,
            policy_state_dict=late_policy_state,
            pack=late_pack,
            deterministic=bool(deterministic_actions),
            torch_seed=int(seed) + 23,
        )
    finally:
        vec_env.close()
        del algo
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    np.savez_compressed(
        out_dir / "captured_target_pack_slices.npz",
        target_envs=np.asarray(target_envs, dtype=np.int64),
        early_rewards=early_pack["rewards"][:, target_envs],
        late_rewards=late_pack["rewards"][:, target_envs],
        early_actions=early_pack["actions"][:, target_envs, :],
        late_actions=late_pack["actions"][:, target_envs, :],
        early_policy_actions_on_early=early_actions_on_early[:, target_envs, :],
        late_policy_actions_on_early=late_actions_on_early[:, target_envs, :],
        early_policy_actions_on_late=early_actions_on_late[:, target_envs, :],
        late_policy_actions_on_late=late_actions_on_late[:, target_envs, :],
    )

    time_indices = list(range(1, int(n_steps), max(1, int(time_stride))))[: int(max_time_points)]
    eval_items: list[dict[str, Any]] = []
    horizon_items: list[dict[str, Any]] = []
    for source_name, source_pack, old_actions, new_actions in (
        ("early_states", early_pack, early_actions_on_early, late_actions_on_early),
        ("late_states", late_pack, early_actions_on_late, late_actions_on_late),
    ):
        for env_idx in target_envs:
            h = h_list[int(env_idx)]
            action_dim = int(h["action_dim"])
            for step_idx in time_indices:
                state = _state_at_step_from_pack(
                    pack=source_pack,
                    h=h,
                    env_idx=int(env_idx),
                    step_idx=int(step_idx),
                )
                if state is None:
                    continue
                horizon_items.append(
                    _make_horizon_start_item(
                        h=h,
                        env_seed=990000 + int(env_idx) * 1000 + int(step_idx),
                        initial_state=state,
                        env_idx=int(env_idx),
                        source_pack=source_name,
                        step_idx=int(step_idx),
                    )
                )
                old_action = old_actions[int(step_idx), int(env_idx), :action_dim]
                new_action = new_actions[int(step_idx), int(env_idx), :action_dim]
                mid_action = old_action + 0.5 * (new_action - old_action)
                base_seed = 880000 + int(env_idx) * 1000 + int(step_idx)
                for action_kind, action in (
                    ("old_policy", old_action),
                    ("halfway_old_to_new", mid_action),
                    ("late_policy", new_action),
                ):
                    eval_items.append(
                        _make_eval_item(
                            h=h,
                            env_seed=base_seed,
                            initial_state=state,
                            action=action,
                            action_dim=action_dim,
                            action_slot_dim=action_slot_dim,
                            env_idx=int(env_idx),
                            source_pack=source_name,
                            step_idx=int(step_idx),
                            action_kind=action_kind,
                        )
                    )

    eval_rows = _eval_one_step_items(
        cfg=cfg,
        env_cfg=env_cfg,
        items=eval_items,
        device_obj=device_obj,
        num_features=int(num_features),
        action_slot_dim=int(action_slot_dim),
        chunk_size=int(eval_chunk_size),
    )
    horizon_rows: list[dict[str, Any]] = []
    horizon_step_rows: list[dict[str, Any]] = []
    for policy_label, policy_state, seed_offset in (
        ("old_policy", early_policy_state, 101),
        ("late_policy", late_policy_state, 202),
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
                torch_seed=int(seed) + int(seed_offset),
                step_rows_out=horizon_step_rows,
            )
        )

    grouped: dict[tuple[int, str, int], dict[str, float]] = {}
    for row in eval_rows:
        grouped[(int(row["env_idx"]), str(row["source_pack"]), int(row["step_idx"]))] = {
            **grouped.get((int(row["env_idx"]), str(row["source_pack"]), int(row["step_idx"])), {}),
            str(row["action_kind"]): float(row["reward_env"]),
        }
    pair_rows: list[dict[str, Any]] = []
    for (env_idx, source_pack, step_idx), values in sorted(grouped.items()):
        if "old_policy" not in values or "late_policy" not in values:
            continue
        pair_rows.append(
            {
                "env_idx": int(env_idx),
                "source_pack": str(source_pack),
                "step_idx": int(step_idx),
                "old_reward_env": float(values["old_policy"]),
                "halfway_reward_env": values.get("halfway_old_to_new"),
                "late_reward_env": float(values["late_policy"]),
                "late_minus_old_reward_env": float(values["late_policy"] - values["old_policy"]),
                "halfway_minus_old_reward_env": (
                    None
                    if values.get("halfway_old_to_new") is None
                    else float(values["halfway_old_to_new"] - values["old_policy"])
                ),
            }
        )

    def _rows_for(source: str, env_idx: int | None = None) -> list[dict[str, Any]]:
        out = [row for row in pair_rows if row["source_pack"] == source]
        if env_idx is not None:
            out = [row for row in out if int(row["env_idx"]) == int(env_idx)]
        return out

    env_summary: list[dict[str, Any]] = []
    for env_idx in target_envs:
        early_rows = _rows_for("early_states", int(env_idx))
        late_rows = _rows_for("late_states", int(env_idx))
        env_summary.append(
            {
                "env_idx": int(env_idx),
                "env_seed": int(env_seeds[int(env_idx)]),
                "early_state_late_minus_old": _stats([r["late_minus_old_reward_env"] for r in early_rows]),
                "late_state_late_minus_old": _stats([r["late_minus_old_reward_env"] for r in late_rows]),
                "state_basin_old_action_late_state_minus_early_state_mean": (
                    None
                    if _stats([r["old_reward_env"] for r in late_rows])["mean"] is None
                    or _stats([r["old_reward_env"] for r in early_rows])["mean"] is None
                    else float(
                        _stats([r["old_reward_env"] for r in late_rows])["mean"]
                        - _stats([r["old_reward_env"] for r in early_rows])["mean"]
                    )
                ),
                "state_basin_late_action_late_state_minus_early_state_mean": (
                    None
                    if _stats([r["late_reward_env"] for r in late_rows])["mean"] is None
                    or _stats([r["late_reward_env"] for r in early_rows])["mean"] is None
                    else float(
                        _stats([r["late_reward_env"] for r in late_rows])["mean"]
                        - _stats([r["late_reward_env"] for r in early_rows])["mean"]
                    )
                ),
            }
        )

    horizon_grouped: dict[tuple[int, str, int, int], dict[str, dict[str, Any]]] = {}
    for row in horizon_rows:
        key = (
            int(row["env_idx"]),
            str(row["source_pack"]),
            int(row["start_step_idx"]),
            int(row["horizon"]),
        )
        horizon_grouped.setdefault(key, {})[str(row["policy_label"])] = row
    horizon_pair_rows: list[dict[str, Any]] = []
    for (env_idx, source_pack, start_step_idx, horizon), values in sorted(horizon_grouped.items()):
        old = values.get("old_policy")
        late = values.get("late_policy")
        if old is None or late is None:
            continue
        horizon_pair_rows.append(
            {
                "env_idx": int(env_idx),
                "source_pack": str(source_pack),
                "start_step_idx": int(start_step_idx),
                "horizon": int(horizon),
                "old_reward_env_sum": float(old["reward_env_sum"]),
                "late_reward_env_sum": float(late["reward_env_sum"]),
                "late_minus_old_reward_env_sum": float(late["reward_env_sum"] - old["reward_env_sum"]),
                "old_reward_env_q90": float(old["reward_env_q90"]),
                "late_reward_env_q90": float(late["reward_env_q90"]),
                "late_minus_old_reward_env_q90": float(late["reward_env_q90"] - old["reward_env_q90"]),
                "old_reward_env_positive_fraction": float(old["reward_env_positive_fraction"]),
                "late_reward_env_positive_fraction": float(late["reward_env_positive_fraction"]),
                "late_minus_old_reward_env_positive_fraction": float(
                    late["reward_env_positive_fraction"] - old["reward_env_positive_fraction"]
                ),
                "old_done_count": int(old["done_count"]),
                "late_done_count": int(late["done_count"]),
                "late_minus_old_done_count": int(late["done_count"] - old["done_count"]),
            }
        )

    horizon_step_grouped: dict[tuple[int, str, int, int], dict[str, dict[str, Any]]] = {}
    for row in horizon_step_rows:
        key = (
            int(row["env_idx"]),
            str(row["source_pack"]),
            int(row["start_step_idx"]),
            int(row["rollout_step"]),
        )
        horizon_step_grouped.setdefault(key, {})[str(row["policy_label"])] = row
    horizon_step_pair_rows: list[dict[str, Any]] = []
    for (env_idx, source_pack, start_step_idx, rollout_step), values in sorted(horizon_step_grouped.items()):
        old = values.get("old_policy")
        late = values.get("late_policy")
        if old is None or late is None:
            continue
        horizon_step_pair_rows.append(
            {
                "env_idx": int(env_idx),
                "source_pack": str(source_pack),
                "start_step_idx": int(start_step_idx),
                "rollout_step": int(rollout_step),
                "old_reward_env": float(old["reward_env"]),
                "late_reward_env": float(late["reward_env"]),
                "late_minus_old_reward_env": float(late["reward_env"] - old["reward_env"]),
                "old_done": bool(old["done"]),
                "late_done": bool(late["done"]),
                "late_minus_old_done": int(bool(late["done"])) - int(bool(old["done"])),
                "old_action_norm": float(old["action_norm"]),
                "late_action_norm": float(late["action_norm"]),
                "late_minus_old_action_norm": float(late["action_norm"] - old["action_norm"]),
                "old_action_std": float(old["action_std"]),
                "late_action_std": float(late["action_std"]),
                "late_minus_old_action_std": float(late["action_std"] - old["action_std"]),
                "old_action_linf": float(old["action_linf"]),
                "late_action_linf": float(late["action_linf"]),
                "late_minus_old_action_linf": float(late["action_linf"] - old["action_linf"]),
                "old_obs_before_norm": float(old["obs_before_norm"]),
                "late_obs_before_norm": float(late["obs_before_norm"]),
                "late_minus_old_obs_before_norm": float(late["obs_before_norm"] - old["obs_before_norm"]),
                "old_obs_delta_norm": float(old["obs_delta_norm"]),
                "late_obs_delta_norm": float(late["obs_delta_norm"]),
                "late_minus_old_obs_delta_norm": float(late["obs_delta_norm"] - old["obs_delta_norm"]),
            }
        )

    def _step_stats(source_pack: str, rollout_step: int, key: str) -> dict[str, Any]:
        return _stats(
            [
                row[key]
                for row in horizon_step_pair_rows
                if str(row["source_pack"]) == str(source_pack)
                and int(row["rollout_step"]) == int(rollout_step)
            ]
        )

    step_summary = {
        str(step): {
            source_pack: {
                "late_minus_old_reward_env": _step_stats(
                    source_pack, int(step), "late_minus_old_reward_env"
                ),
                "late_minus_old_action_norm": _step_stats(
                    source_pack, int(step), "late_minus_old_action_norm"
                ),
                "late_minus_old_obs_before_norm": _step_stats(
                    source_pack, int(step), "late_minus_old_obs_before_norm"
                ),
                "late_minus_old_done": _step_stats(source_pack, int(step), "late_minus_old_done"),
            }
            for source_pack in ("early_states", "late_states")
        }
        for step in range(1, int(max(horizon_values)) + 1)
    }

    def _horizon_stats(source_pack: str, horizon: int, key: str) -> dict[str, Any]:
        return _stats(
            [
                row[key]
                for row in horizon_pair_rows
                if str(row["source_pack"]) == str(source_pack) and int(row["horizon"]) == int(horizon)
            ]
        )

    horizon_summary = {
        str(horizon): {
            source_pack: {
                "late_minus_old_reward_env_sum": _horizon_stats(
                    source_pack, int(horizon), "late_minus_old_reward_env_sum"
                ),
                "late_minus_old_reward_env_q90": _horizon_stats(
                    source_pack, int(horizon), "late_minus_old_reward_env_q90"
                ),
                "late_minus_old_reward_env_positive_fraction": _horizon_stats(
                    source_pack, int(horizon), "late_minus_old_reward_env_positive_fraction"
                ),
                "late_minus_old_done_count": _horizon_stats(
                    source_pack, int(horizon), "late_minus_old_done_count"
                ),
            }
            for source_pack in ("early_states", "late_states")
        }
        for horizon in horizon_values
    }

    summary = {
        "artifact": "phase2_worst_delta_action_state_counterfactual_probe",
        "contract": (
            "Reruns the fixed n64 group with the canonical pack/update path to capture early and late policy states. "
            "For worst-delta envs only, it replays old/new deterministic policy actions on the same captured early/late "
            "history tokens, then evaluates one-step exact-SCM rewards from the same captured raw state. "
            "This is a diagnostic; it does not change generator defaults or gates."
        ),
        "output_dir": str(out_dir),
        "selected_csv": str(selected_csv),
        "delta_csv": str(delta_csv),
        "target_envs": [int(v) for v in target_envs],
        "updates": int(updates),
        "n_envs": int(n_envs),
        "n_steps": int(n_steps),
        "time_indices": [int(v) for v in time_indices],
        "horizons": [int(v) for v in horizon_values],
        "deterministic_actions": bool(deterministic_actions),
        "progress_jsonl": str(out_dir / "progress.jsonl"),
        "counterfactual_rows_csv": str(out_dir / "counterfactual_rows.csv"),
        "counterfactual_pairs_csv": str(out_dir / "counterfactual_pairs.csv"),
        "horizon_rows_csv": str(out_dir / "horizon_rows.csv"),
        "horizon_pairs_csv": str(out_dir / "horizon_pairs.csv"),
        "horizon_step_rows_csv": str(out_dir / "horizon_step_rows.csv"),
        "horizon_step_pairs_csv": str(out_dir / "horizon_step_pairs.csv"),
        "env_summary_csv": str(out_dir / "env_summary.csv"),
        "captured_npz": str(out_dir / "captured_target_pack_slices.npz"),
        "progress_summary": {
            "first_target_env_reward_sum_mean": progress_rows[0]["target_env_reward_sum_mean"] if progress_rows else None,
            "last_target_env_reward_sum_mean": progress_rows[-1]["target_env_reward_sum_mean"] if progress_rows else None,
        },
        "all_early_state_late_minus_old": _stats(
            [r["late_minus_old_reward_env"] for r in pair_rows if r["source_pack"] == "early_states"]
        ),
        "all_late_state_late_minus_old": _stats(
            [r["late_minus_old_reward_env"] for r in pair_rows if r["source_pack"] == "late_states"]
        ),
        "all_state_basin_old_action_late_minus_early": _stats(
            [
                row["state_basin_old_action_late_state_minus_early_state_mean"]
                for row in env_summary
                if row["state_basin_old_action_late_state_minus_early_state_mean"] is not None
            ]
        ),
        "horizon_summary": horizon_summary,
        "step_summary": step_summary,
        "env_summary": env_summary,
    }

    _write_jsonl(out_dir / "progress.jsonl", progress_rows)
    _write_csv(out_dir / "counterfactual_rows.csv", eval_rows)
    _write_csv(out_dir / "counterfactual_pairs.csv", pair_rows)
    _write_csv(out_dir / "horizon_rows.csv", horizon_rows)
    _write_csv(out_dir / "horizon_pairs.csv", horizon_pair_rows)
    _write_csv(out_dir / "horizon_step_rows.csv", horizon_step_rows)
    _write_csv(out_dir / "horizon_step_pairs.csv", horizon_step_pair_rows)
    _write_csv(out_dir / "env_summary.csv", env_summary)
    _write_json(out_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Worst-delta action/state counterfactual probe.")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--selected-csv", type=str, default=DEFAULT_SELECTED_CSV)
    parser.add_argument("--selected-rule", type=str, default="ratio10_q90p1_posp15")
    parser.add_argument("--delta-csv", type=str, default=DEFAULT_DELTA_CSV)
    parser.add_argument("--target-env-indices", type=str, default="")
    parser.add_argument("--target-limit", type=int, default=8)
    parser.add_argument("--n-envs", type=int, default=64)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--updates", type=int, default=50)
    parser.add_argument("--seed", type=int, default=4040)
    parser.add_argument("--time-stride", type=int, default=16)
    parser.add_argument("--max-time-points", type=int, default=12)
    parser.add_argument("--horizons", type=str, default="5,10,20")
    parser.add_argument("--eval-chunk-size", type=int, default=96)
    parser.add_argument("--deterministic-actions", type=lambda x: str(x).lower() not in {"0", "false", "no"}, default=True)
    args = parser.parse_args()
    summary = run_probe(
        output_dir=args.output_dir,
        device=args.device,
        selected_csv=args.selected_csv,
        selected_rule=args.selected_rule,
        delta_csv=args.delta_csv,
        target_env_indices=args.target_env_indices,
        target_limit=int(args.target_limit),
        n_envs=int(args.n_envs),
        n_steps=int(args.n_steps),
        updates=int(args.updates),
        seed=int(args.seed),
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
                    "summary_json": str(Path(args.output_dir).expanduser().resolve() / "summary.json"),
                    "target_envs": summary["target_envs"],
                    "all_early_state_late_minus_old": summary["all_early_state_late_minus_old"],
                    "all_late_state_late_minus_old": summary["all_late_state_late_minus_old"],
                    "all_state_basin_old_action_late_minus_early": summary[
                        "all_state_basin_old_action_late_minus_early"
                    ],
                    "horizon_summary": summary["horizon_summary"],
                }
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
