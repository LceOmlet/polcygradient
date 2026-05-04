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
    _parse_capture_updates,
    _parse_ints,
    _selected_time_indices,
)
from ticl.analysis.phase2_worst_delta_action_state_counterfactual_probe import (  # noqa: E402
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
    "phase2_gae_lambda_mask_bootstrap_variant_probe_env34_9_36_0501"
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


def _index_delta_rows(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(float(row["env_idx"])): row for row in rows}


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


def _compute_variant_advantage(
    *,
    rewards: np.ndarray,
    values: np.ndarray,
    episode_starts: np.ndarray,
    terminated: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    gae_lambda: float,
    step_idx: int,
    env_idx: int,
    mask_mode: str,
    bootstrap_mode: str,
) -> dict[str, Any]:
    n_steps = int(rewards.shape[0])
    step = int(step_idx)
    env = int(env_idx)
    lam = float(gae_lambda)
    gamma_f = float(gamma)
    mask_mode = str(mask_mode)
    bootstrap_mode = str(bootstrap_mode)
    total = 0.0
    reward_contrib = 0.0
    next_value_contrib = 0.0
    curr_value_contrib = 0.0
    weight = 1.0
    first_next_nonterminal = 1.0
    cut = False
    terms = 0
    for t in range(step, n_steps):
        if t == n_steps - 1:
            if mask_mode == "no_cut":
                nt = 1.0
            elif mask_mode == "terminated_only":
                nt = 1.0 - float(bool(terminated[t, env]))
            else:
                nt = 1.0 - float(bool(dones[t, env]))
            next_v = 0.0
        else:
            if mask_mode == "no_cut":
                nt = 1.0
            elif mask_mode == "terminated_only":
                nt = 1.0 - float(bool(terminated[t, env]))
            else:
                nt = 1.0 - float(episode_starts[t + 1, env] > 0.5)
            next_v = float(values[t + 1, env])
        if t == step:
            first_next_nonterminal = float(nt)
        curr_v = float(values[t, env])
        reward = float(rewards[t, env])
        gamma_eff = gamma_f * float(nt)
        if bootstrap_mode == "full":
            delta = reward + gamma_eff * next_v - curr_v
            next_term = gamma_eff * next_v
            curr_term = -curr_v
        elif bootstrap_mode == "no_next_value":
            delta = reward - curr_v
            next_term = 0.0
            curr_term = -curr_v
        elif bootstrap_mode == "reward_only":
            delta = reward
            next_term = 0.0
            curr_term = 0.0
        else:
            raise ValueError(f"unknown bootstrap_mode={bootstrap_mode!r}")
        total += weight * delta
        reward_contrib += weight * reward
        next_value_contrib += weight * next_term
        curr_value_contrib += weight * curr_term
        terms += 1
        if nt <= 0.0 and mask_mode != "no_cut":
            cut = True
            break
        weight *= gamma_f * lam * float(nt)
        if weight < 1e-10:
            break
    return {
        "variant_advantage": float(total),
        "variant_reward_contrib": float(reward_contrib),
        "variant_next_value_contrib": float(next_value_contrib),
        "variant_current_value_contrib": float(curr_value_contrib),
        "variant_value_net_contrib": float(next_value_contrib + curr_value_contrib),
        "variant_effective_terms": int(terms),
        "variant_cut_by_mask": bool(cut),
        "variant_first_next_nonterminal": float(first_next_nonterminal),
    }


def _variant_specs() -> list[dict[str, Any]]:
    return [
        {"variant": "lambda0p90_episode_full", "gae_lambda": 0.90, "mask_mode": "episode", "bootstrap_mode": "full"},
        {"variant": "lambda0p50_episode_full", "gae_lambda": 0.50, "mask_mode": "episode", "bootstrap_mode": "full"},
        {"variant": "lambda0p00_episode_full", "gae_lambda": 0.00, "mask_mode": "episode", "bootstrap_mode": "full"},
        {"variant": "lambda0p95_episode_full", "gae_lambda": 0.95, "mask_mode": "episode", "bootstrap_mode": "full"},
        {"variant": "lambda1p00_episode_full", "gae_lambda": 1.00, "mask_mode": "episode", "bootstrap_mode": "full"},
        {"variant": "lambda0p90_terminated_only_full", "gae_lambda": 0.90, "mask_mode": "terminated_only", "bootstrap_mode": "full"},
        {"variant": "lambda0p90_no_cut_full", "gae_lambda": 0.90, "mask_mode": "no_cut", "bootstrap_mode": "full"},
        {"variant": "lambda0p90_episode_no_next_value", "gae_lambda": 0.90, "mask_mode": "episode", "bootstrap_mode": "no_next_value"},
        {"variant": "lambda0p90_episode_reward_only", "gae_lambda": 0.90, "mask_mode": "episode", "bootstrap_mode": "reward_only"},
    ]


def _summarize(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row.get(key) for key in keys), []).append(row)
    out: list[dict[str, Any]] = []
    for key_values, group in sorted(grouped.items(), key=lambda item: tuple(str(v) for v in item[0])):
        count = max(1, len(group))
        record = {key: value for key, value in zip(keys, key_values)}
        mismatch = [
            row
            for row in group
            if bool(row["variant_adv_pos_rollout_below_median_objective"])
            or bool(row["variant_adv_neg_rollout_best_objective"])
        ]
        record.update(
            {
                "count": int(len(group)),
                "objective_mismatch_count": int(len(mismatch)),
                "objective_mismatch_fraction": float(len(mismatch) / count),
                "adv_pos_rollout_below_median_objective_count": int(
                    sum(bool(row["variant_adv_pos_rollout_below_median_objective"]) for row in group)
                ),
                "adv_neg_rollout_best_objective_count": int(
                    sum(bool(row["variant_adv_neg_rollout_best_objective"]) for row in group)
                ),
                "variant_advantage": _stats([row["variant_advantage"] for row in group]),
                "variant_effective_terms": _stats([row["variant_effective_terms"] for row in group]),
                "variant_first_next_nonterminal_mean": float(
                    np.mean([float(row["variant_first_next_nonterminal"]) for row in group])
                ),
                "variant_cut_fraction": float(np.mean([bool(row["variant_cut_by_mask"]) for row in group])),
                "delta_env_sum": _stats([row["delta_env_sum"] for row in group]),
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
    q_rows_all = _read_csv_rows(q_pair_csv)
    q_rows = [
        row
        for row in q_rows_all
        if int(float(row["env_idx"])) in set(_parse_ints(target_env_indices))
    ]
    q_keys = {
        (
            int(float(row["update_idx"])),
            int(float(row["env_idx"])),
            int(float(row["step_idx"])),
        )
        for row in q_rows
    }
    delta_index = _index_delta_rows(_read_csv_rows(delta_csv))
    target_envs = _parse_ints(target_env_indices)
    capture_set = _parse_capture_updates(capture_updates, updates=int(updates))
    time_indices = _selected_time_indices(
        int(n_steps),
        start=int(time_start),
        stride=int(time_stride),
        max_count=int(max_time_points),
    )
    device_obj = torch.device(device)

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

    base_rows: list[dict[str, Any]] = []
    variant_rows: list[dict[str, Any]] = []
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
            rewards_2d = _as_step_env_array(
                np.asarray(algo.rollout_buffer.rewards, dtype=np.float32).copy(),
                n_steps=int(n_steps),
                n_envs=int(n_envs),
                name="buffer_rewards",
            )
            values_2d = _as_step_env_array(
                np.asarray(algo.rollout_buffer.values, dtype=np.float32).copy(),
                n_steps=int(n_steps),
                n_envs=int(n_envs),
                name="buffer_values",
            )
            actor_adv_2d = _as_step_env_array(
                np.asarray(algo.rollout_buffer.actor_advantages, dtype=np.float32).copy(),
                n_steps=int(n_steps),
                n_envs=int(n_envs),
                name="actor_advantages",
            )
            episode_starts_2d = _as_step_env_array(
                np.asarray(algo.rollout_buffer.episode_starts, dtype=np.float32).copy(),
                n_steps=int(n_steps),
                n_envs=int(n_envs),
                name="episode_starts",
            )
            terminated_2d = _as_step_env_array(
                np.asarray(buffer_pack["terminated"], dtype=bool),
                n_steps=int(n_steps),
                n_envs=int(n_envs),
                name="terminated",
            )
            dones_2d = _as_step_env_array(
                np.asarray(buffer_pack["dones"], dtype=bool),
                n_steps=int(n_steps),
                n_envs=int(n_envs),
                name="dones",
            )
            progress_rows.append(
                {
                    "update_idx": int(update_idx),
                    "captured": bool(update_idx in capture_set),
                    "reward_norm": reward_norm_stats,
                }
            )
            if update_idx in capture_set:
                for env_idx in target_envs:
                    for step_idx in time_indices:
                        if (int(update_idx), int(env_idx), int(step_idx)) not in q_keys:
                            continue
                        current_adv = float(actor_adv_2d[int(step_idx), int(env_idx)])
                        base_rows.append(
                            {
                                "update_idx": int(update_idx),
                                "env_idx": int(env_idx),
                                "step_idx": int(step_idx),
                                "actor_advantage_current": current_adv,
                                "delta_env_sum": _to_float(delta_index[int(env_idx)]["delta_env_sum"]),
                            }
                        )
                        for spec in _variant_specs():
                            decomp = _compute_variant_advantage(
                                rewards=rewards_2d,
                                values=values_2d,
                                episode_starts=episode_starts_2d,
                                terminated=terminated_2d,
                                dones=dones_2d,
                                gamma=float(algo.gamma),
                                gae_lambda=float(spec["gae_lambda"]),
                                step_idx=int(step_idx),
                                env_idx=int(env_idx),
                                mask_mode=str(spec["mask_mode"]),
                                bootstrap_mode=str(spec["bootstrap_mode"]),
                            )
                            variant_rows.append(
                                {
                                    "update_idx": int(update_idx),
                                    "env_idx": int(env_idx),
                                    "step_idx": int(step_idx),
                                    "variant": str(spec["variant"]),
                                    "gae_lambda": float(spec["gae_lambda"]),
                                    "mask_mode": str(spec["mask_mode"]),
                                    "bootstrap_mode": str(spec["bootstrap_mode"]),
                                    "actor_advantage_current": current_adv,
                                    "current_adv_recompute_abs_err": abs(
                                        current_adv - float(decomp["variant_advantage"])
                                    )
                                    if spec["variant"] == "lambda0p90_episode_full"
                                    else None,
                                    "delta_env_sum": _to_float(delta_index[int(env_idx)]["delta_env_sum"]),
                                    **decomp,
                                }
                            )
            algo.train()
    finally:
        vec_env.close()
        del algo
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    q_index = {
        (
            int(float(row["update_idx"])),
            int(float(row["env_idx"])),
            int(float(row["step_idx"])),
            int(float(row["horizon"])),
        ): row
        for row in q_rows
    }
    joined_rows: list[dict[str, Any]] = []
    for vrow in variant_rows:
        for horizon in (5, 10, 20):
            qrow = q_index.get(
                (
                    int(vrow["update_idx"]),
                    int(vrow["env_idx"]),
                    int(vrow["step_idx"]),
                    int(horizon),
                )
            )
            if qrow is None:
                continue
            adv = float(vrow["variant_advantage"])
            rollout_below_median_objective = _to_bool(qrow["rollout_below_median_objective"])
            rollout_is_best_objective = _to_bool(qrow["rollout_is_best_objective"])
            joined_rows.append(
                {
                    **vrow,
                    "horizon": int(horizon),
                    "rollout_minus_median_reward_objective_sum": _to_float(
                        qrow["rollout_minus_median_reward_objective_sum"]
                    ),
                    "rollout_minus_best_reward_objective_sum": _to_float(
                        qrow["rollout_minus_best_reward_objective_sum"]
                    ),
                    "rollout_rank_desc_objective": int(float(qrow["rollout_rank_desc_objective"])),
                    "rollout_below_median_objective": bool(rollout_below_median_objective),
                    "rollout_is_best_objective": bool(rollout_is_best_objective),
                    "variant_adv_pos_rollout_below_median_objective": bool(
                        adv > 0.0 and rollout_below_median_objective
                    ),
                    "variant_adv_neg_rollout_best_objective": bool(
                        adv < 0.0 and rollout_is_best_objective
                    ),
                }
            )

    summary = {
        "artifact": "phase2_gae_lambda_mask_bootstrap_variant_probe",
        "contract": (
            "Only objective-space mismatch rows are evaluated. Variants change GAE lambda, "
            "done/reset mask, or value bootstrap offline; no generator/gate changes are made."
        ),
        "q_pair_csv": str(Path(q_pair_csv).expanduser().resolve()),
        "target_envs": [int(v) for v in target_envs],
        "target_env_seeds": [int(env_seeds[int(v)]) for v in target_envs],
        "n_steps": int(n_steps),
        "updates": int(updates),
        "capture_updates": sorted(int(v) for v in capture_set),
        "time_indices": [int(v) for v in time_indices],
        "variant_specs": _variant_specs(),
        "joined_rows_count": int(len(joined_rows)),
        "current_recompute_max_abs_err": max(
            [
                _to_float(row["current_adv_recompute_abs_err"])
                for row in variant_rows
                if row["current_adv_recompute_abs_err"] is not None
            ],
            default=float("nan"),
        ),
        "summary_by_variant_horizon": _summarize(joined_rows, ["variant", "horizon"]),
        "summary_by_env_variant_horizon": _summarize(joined_rows, ["env_idx", "variant", "horizon"]),
        "files": {
            "summary_json": str(out_dir / "summary.json"),
            "progress_jsonl": str(out_dir / "progress.jsonl"),
            "variant_rows_csv": str(out_dir / "variant_rows.csv"),
            "joined_variant_q_rows_csv": str(out_dir / "joined_variant_q_rows.csv"),
        },
    }
    _write_json(out_dir / "summary.json", summary)
    with (out_dir / "progress.jsonl").open("w", encoding="utf-8") as handle:
        for row in progress_rows:
            handle.write(json.dumps(_json_safe(row), sort_keys=True) + "\n")
    _write_csv(out_dir / "variant_rows.csv", variant_rows)
    _write_csv(out_dir / "joined_variant_q_rows.csv", joined_rows)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline GAE lambda/mask/bootstrap variant probe.")
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
                    "joined_rows_csv": summary["files"]["joined_variant_q_rows_csv"],
                    "joined_rows_count": summary["joined_rows_count"],
                    "current_recompute_max_abs_err": summary["current_recompute_max_abs_err"],
                }
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
