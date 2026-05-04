#!/usr/bin/env python3
"""Replay saved Phase-2 runner checkpoints into per-env semantic sidecars.

This is intentionally a thin wrapper around
``phase2_gym_prior_pack_training_runner`` collectors.  It does not resume
training and does not run an optimizer step; it restores the saved policy plus
SB3-style obs/reward normalizer state, collects one fresh pack per checkpoint,
then writes the same per-env semantic rows that the runner sidecar would write.
"""

from __future__ import annotations

import argparse
import csv
import copy
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ticl.analysis.critic_free_single_env_audit import _seed_all  # noqa: E402
from ticl.analysis.phase2_gym_prior_pack_training_runner import (  # noqa: E402
    RunningMeanStd,
    _build_algo,
    _collect_gym_policy_pack,
    _collect_prior_policy_pack,
    _json_safe,
    _load_fixed_frozen_h_list_from_csv,
    _normalize_env_id_list,
    _per_env_semantic_probe_rows_from_pack,
    _semantic_probe_from_pack,
    _vector_stats,
    _write_jsonl,
)
from ticl.model_configs import get_model_default_config  # noqa: E402
from ticl.priors.environment_prior import EnvironmentPrior  # noqa: E402
from ticl.rlpfn_maintained_path import (  # noqa: E402
    resolve_rlpfn_token_layout,
    validate_rlpfn_maintained_path_config,
)


DEFAULT_GYM_CHECKPOINTS = (
    "/home/chen/RLPFN/artifacts/"
    "phase2_clean_official_gym_pack_multigym_n64_u100_obsrewardnorm_gpu0_0501/checkpoints/"
    "gym_update_000000.pt,"
    "/home/chen/RLPFN/artifacts/"
    "phase2_clean_official_gym_pack_multigym_n64_u100_obsrewardnorm_gpu0_0501/checkpoints/"
    "gym_update_000075.pt"
)
DEFAULT_PRIOR_CHECKPOINTS = (
    "/home/chen/RLPFN/artifacts/"
    "phase2_clean_official_prior_ratio10_q90p1_posp15_n64_u100_obsrewardnorm_gpu0_0501/checkpoints/"
    "prior_update_000000.pt,"
    "/home/chen/RLPFN/artifacts/"
    "phase2_clean_official_prior_ratio10_q90p1_posp15_n64_u100_obsrewardnorm_gpu0_0501/checkpoints/"
    "prior_update_000075.pt"
)
DEFAULT_SELECTED_CSV = (
    "/home/chen/RLPFN/artifacts/"
    "phase2_reserve_probe_selected_ratio10_q90p1_posp15_n64_seed9090_0430/selected_envs.csv"
)

COMMON_DELTA_METRICS = [
    "reward_sum",
    "training_reward_sum",
    "raw_reward_sum",
    "reward_env_sum",
    "reward_q90",
    "training_reward_q90",
    "raw_reward_q90",
    "done_count",
    "first_done_step",
    "action_value_mean",
    "action_value_std",
    "action_value_absmean",
    "action_value_absmax",
    "action_norm_mean",
    "action_norm_std",
    "unit_clipped_action_value_absmean",
    "unit_clipped_action_value_absmax",
    "unit_clipped_action_norm_mean",
    "unit_clipped_action_norm_std",
    "gym_normalized_executed_action_l2_upper_bound",
    "gym_native_action_l2_upper_bound",
    "raw_policy_action_norm_mean_to_gym_bound",
    "unit_clipped_action_norm_mean_to_gym_bound",
    "raw_policy_action_absmax_to_unit_clip",
    "raw_policy_action_unit_clip_saturation_fraction",
    "raw_policy_action_unit_clip_excess_absmean",
    "action_temporal_delta_norm_mean",
    "action_per_dim_temporal_std_mean",
    "action_per_dim_temporal_std_q10",
    "action_per_dim_temporal_std_q50",
    "action_per_dim_temporal_std_q90",
    "action_constant_dim_fraction_std_lt_0p05",
    "action_constant_dim_fraction_std_lt_0p10",
    "obs_per_dim_rms_mean",
    "next_state_step_l2_delta_to_rms_ratio",
]


def _split_paths(value: str) -> list[Path]:
    return [Path(part.strip()).expanduser().resolve() for part in str(value).split(",") if part.strip()]


def _restore_checkpoint_state(algo, checkpoint_path: Path, *, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    algo.policy.load_state_dict(checkpoint["policy_state_dict"], strict=True)
    algo.policy.to(device)
    algo.policy.set_training_mode(False)
    algo._n_updates = int(checkpoint.get("n_updates", 0))

    obs_norm = checkpoint.get("obs_norm", {}) or {}
    algo._rwkv_sb3_observation_normalization_enabled = bool(obs_norm.get("enabled", False))
    if bool(obs_norm.get("enabled", False)):
        algo._rwkv_sb3_observation_normalization_mean = np.asarray(obs_norm["mean"], dtype=np.float64)
        algo._rwkv_sb3_observation_normalization_var = np.asarray(obs_norm["var"], dtype=np.float64)
        algo._rwkv_sb3_observation_normalization_count = np.asarray(obs_norm["count"], dtype=np.float64)
        algo._rwkv_sb3_observation_normalization_clip = float(obs_norm.get("clip", 10.0))
        algo._rwkv_sb3_observation_normalization_epsilon = float(obs_norm.get("epsilon", 1e-8))

    reward_norm = checkpoint.get("reward_norm", {}) or {}
    algo._rwkv_sb3_reward_normalization_enabled = bool(reward_norm.get("enabled", False))
    algo._rwkv_sb3_reward_normalization_returns = np.asarray(
        reward_norm.get("returns", np.asarray([], dtype=np.float32)),
        dtype=np.float32,
    )
    if bool(reward_norm.get("enabled", False)):
        ret_rms = RunningMeanStd(shape=())
        ret_rms.mean = np.asarray(reward_norm["ret_rms_mean"], dtype=np.float64)
        ret_rms.var = np.asarray(reward_norm["ret_rms_var"], dtype=np.float64)
        ret_rms.count = float(reward_norm["ret_rms_count"])
        algo._rwkv_sb3_reward_normalization_ret_rms = ret_rms

    return checkpoint


def _build_replay_context(
    *,
    device: torch.device,
    n_envs: int,
    n_steps: int,
    seed: int,
    selected_csv: str,
    selected_rule: str,
    selected_limit: int,
    sb3_reward_normalization_enabled: bool,
    sb3_observation_normalization_enabled: bool,
) -> tuple[Any, Any, dict[str, Any]]:
    cfg = copy.deepcopy(get_model_default_config("rlpfn"))
    validate_rlpfn_maintained_path_config(cfg)
    env_cfg = copy.deepcopy(cfg["prior"]["environment"])
    num_features = int(cfg["prior"]["num_features"])
    layout = resolve_rlpfn_token_layout(env_cfg, num_features=num_features)
    obs_slot_dim = int(layout["obs_slot_dim"])
    action_slot_dim = int(layout["action_slot_dim"])
    frozen_h_list, _h_paths, frozen_h_list_env_seeds = _load_fixed_frozen_h_list_from_csv(
        selected_csv,
        rule=selected_rule,
        limit=selected_limit,
    )
    dim_probe_prior = EnvironmentPrior(copy.deepcopy(env_cfg))
    next_state_target_dim = int(
        max(
            *[int((h or {}).get("state_dim", 1)) for h in frozen_h_list],
            int(dim_probe_prior._resolve_dim_upper_bound(env_cfg.get("state_dim", None), default=0)),
            1,
        )
    )
    algo, vec_env = _build_algo(
        cfg=cfg,
        env_cfg=env_cfg,
        frozen_h=None,
        frozen_h_list=frozen_h_list,
        frozen_h_list_env_seeds=frozen_h_list_env_seeds,
        prior_mode="fixed_frozen_list",
        device_obj=device,
        num_features=num_features,
        n_envs=n_envs,
        n_steps=n_steps,
        build_seed=seed,
        sb3_reward_normalization_enabled=sb3_reward_normalization_enabled,
        sb3_observation_normalization_enabled=sb3_observation_normalization_enabled,
        sb3_observation_normalization_clip=10.0,
        sb3_observation_normalization_epsilon=1e-8,
        gain_min=0.7,
        topology_state_gain_min=0.7,
        topology_action_gain_min=0.06,
        topology_state_to_action_ratio_max=15.0,
        topology_max_attempts=512,
        fixed_env_group_across_updates=True,
    )
    meta = {
        "num_features": num_features,
        "obs_slot_dim": obs_slot_dim,
        "action_slot_dim": action_slot_dim,
        "next_state_target_dim": next_state_target_dim,
    }
    return algo, vec_env, meta


def _as_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _write_delta_csv(
    *,
    arm: str,
    sidecar_rows_by_update: dict[int, list[dict[str, Any]]],
    output_path: Path,
) -> dict[str, Any]:
    updates = sorted(sidecar_rows_by_update)
    if len(updates) < 2:
        return {"arm": arm, "written": False, "reason": "need at least two checkpoints"}
    early_update = int(updates[0])
    late_update = int(updates[-1])
    early_by_env = {int(row["env_idx"]): row for row in sidecar_rows_by_update[early_update]}
    late_by_env = {int(row["env_idx"]): row for row in sidecar_rows_by_update[late_update]}
    env_indices = sorted(set(early_by_env) & set(late_by_env))
    fieldnames = [
        "arm",
        "env_idx",
        "gym_env_id",
        "env_group_env_seed",
        "early_update_idx",
        "late_update_idx",
    ]
    for metric in COMMON_DELTA_METRICS:
        fieldnames.extend([f"early_{metric}", f"late_{metric}", f"delta_{metric}"])
    rows: list[dict[str, Any]] = []
    for env_idx in env_indices:
        early = early_by_env[env_idx]
        late = late_by_env[env_idx]
        row: dict[str, Any] = {
            "arm": arm,
            "env_idx": int(env_idx),
            "gym_env_id": late.get("gym_env_id") or early.get("gym_env_id"),
            "env_group_env_seed": late.get("env_group_env_seed") or early.get("env_group_env_seed"),
            "early_update_idx": early_update,
            "late_update_idx": late_update,
        }
        for metric in COMMON_DELTA_METRICS:
            a = _as_float(early.get(metric))
            b = _as_float(late.get(metric))
            row[f"early_{metric}"] = a
            row[f"late_{metric}"] = b
            row[f"delta_{metric}"] = (b - a) if a is not None and b is not None else None
        rows.append(row)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(_json_safe(rows))
    delta_summary = {
        f"delta_{metric}": _vector_stats([row.get(f"delta_{metric}") for row in rows])
        for metric in COMMON_DELTA_METRICS
    }
    return {
        "arm": arm,
        "written": True,
        "path": str(output_path),
        "early_update_idx": early_update,
        "late_update_idx": late_update,
        "n_envs": len(rows),
        "delta_summary": delta_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arms", default="gym,prior")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-envs", type=int, default=64)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--gym-seed", type=int, default=4040)
    parser.add_argument("--prior-seed", type=int, default=9090)
    parser.add_argument("--gym-env-ids", default="Ant-v5,HalfCheetah-v5,Hopper-v5,Walker2d-v5,Humanoid-v5")
    parser.add_argument("--selected-csv", default=DEFAULT_SELECTED_CSV)
    parser.add_argument("--selected-rule", default="ratio10_q90p1_posp15")
    parser.add_argument("--selected-limit", type=int, default=64)
    parser.add_argument("--gym-checkpoints", default=DEFAULT_GYM_CHECKPOINTS)
    parser.add_argument("--prior-checkpoints", default=DEFAULT_PRIOR_CHECKPOINTS)
    parser.add_argument("--replay-seed", type=int, default=20260501)
    parser.add_argument("--sb3-reward-normalization-enabled", type=bool, default=True)
    parser.add_argument("--sb3-observation-normalization-enabled", type=bool, default=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = args.output_dir.expanduser().resolve()
    sidecar_dir = out_dir / "semantic_sidecars"
    out_dir.mkdir(parents=True, exist_ok=True)
    arms = [part.strip().lower() for part in str(args.arms).split(",") if part.strip()]
    progress_rows: list[dict[str, Any]] = []
    sidecar_rows_by_arm_update: dict[str, dict[int, list[dict[str, Any]]]] = {}

    for arm in arms:
        seed = int(args.gym_seed if arm == "gym" else args.prior_seed)
        checkpoints = _split_paths(args.gym_checkpoints if arm == "gym" else args.prior_checkpoints)
        algo, vec_env, replay_meta = _build_replay_context(
            device=device,
            n_envs=int(args.n_envs),
            n_steps=int(args.n_steps),
            seed=seed,
            selected_csv=str(args.selected_csv),
            selected_rule=str(args.selected_rule),
            selected_limit=int(args.selected_limit),
            sb3_reward_normalization_enabled=bool(args.sb3_reward_normalization_enabled),
            sb3_observation_normalization_enabled=bool(args.sb3_observation_normalization_enabled),
        )
        try:
            sidecar_rows_by_arm_update[arm] = {}
            for ckpt_idx, checkpoint_path in enumerate(checkpoints):
                checkpoint = _restore_checkpoint_state(algo, checkpoint_path, device=device)
                update_idx = int(checkpoint.get("update_idx", ckpt_idx))
                _seed_all(int(args.replay_seed) + int(update_idx) + (0 if arm == "gym" else 100000))
                if arm == "gym":
                    pack, _values, _log_probs, meta = _collect_gym_policy_pack(
                        algo=algo,
                        gym_env_ids=_normalize_env_id_list(args.gym_env_ids, int(args.n_envs)),
                        n_steps=int(args.n_steps),
                        seed=seed,
                        num_features=int(replay_meta["num_features"]),
                        obs_slot_dim=int(replay_meta["obs_slot_dim"]),
                        action_slot_dim=int(replay_meta["action_slot_dim"]),
                        next_state_target_dim=int(replay_meta["next_state_target_dim"]),
                        single_eval_pos=64,
                        terminal_token_enabled=True,
                        device_obj=device,
                    )
                elif arm == "prior":
                    pack, _values, _log_probs, meta = _collect_prior_policy_pack(
                        algo=algo,
                        vec_env=vec_env,
                        n_steps=int(args.n_steps),
                        seed=seed,
                        num_features=int(replay_meta["num_features"]),
                        obs_slot_dim=int(replay_meta["obs_slot_dim"]),
                        action_slot_dim=int(replay_meta["action_slot_dim"]),
                        next_state_target_dim=int(replay_meta["next_state_target_dim"]),
                    )
                else:
                    raise ValueError(f"Unsupported arm={arm!r}")
                aggregate = _semantic_probe_from_pack(
                    arm=arm,
                    pack=pack,
                    meta=meta,
                    obs_slot_dim=int(replay_meta["obs_slot_dim"]),
                    action_slot_dim=int(replay_meta["action_slot_dim"]),
                )
                sidecar_rows = _per_env_semantic_probe_rows_from_pack(
                    arm=arm,
                    update_idx=update_idx,
                    pack=pack,
                    meta=meta,
                    obs_slot_dim=int(replay_meta["obs_slot_dim"]),
                    action_slot_dim=int(replay_meta["action_slot_dim"]),
                )
                sidecar_path = sidecar_dir / f"{arm}_checkpoint_update_{update_idx:06d}_envs.jsonl"
                _write_jsonl(sidecar_path, sidecar_rows)
                sidecar_rows_by_arm_update[arm][update_idx] = sidecar_rows
                progress_rows.append(
                    {
                        "arm": arm,
                        "checkpoint_path": str(checkpoint_path),
                        "checkpoint_update_idx": update_idx,
                        "checkpoint_n_updates": int(checkpoint.get("n_updates", 0)),
                        "env_seed": seed,
                        "collector": meta,
                        "semantic_probe": aggregate,
                        "semantic_probe_sidecar_path": str(sidecar_path),
                        "replay_contract": {
                            "source": "checkpoint policy/normalizer replay",
                            "optimizer_step": False,
                            "fresh_pack_collection": True,
                            "replay_seed": int(args.replay_seed),
                        },
                    }
                )
        finally:
            close_fn = getattr(vec_env, "close", None)
            if callable(close_fn):
                close_fn()

    _write_jsonl(out_dir / "checkpoint_replay_progress.jsonl", progress_rows)
    delta_reports = {}
    for arm, rows_by_update in sidecar_rows_by_arm_update.items():
        delta_reports[arm] = _write_delta_csv(
            arm=arm,
            sidecar_rows_by_update=rows_by_update,
            output_path=out_dir / f"{arm}_checkpoint_delta.csv",
        )
    summary = {
        "audit_entry": "phase2_checkpoint_replay_sidecar",
        "output_dir": str(out_dir),
        "progress_jsonl": str(out_dir / "checkpoint_replay_progress.jsonl"),
        "delta_reports": delta_reports,
        "arms": arms,
        "device": str(device),
        "n_envs": int(args.n_envs),
        "n_steps": int(args.n_steps),
        "selected_csv": str(args.selected_csv),
        "selected_rule": str(args.selected_rule),
        "contract": (
            "Loads saved policy and SB3-style obs/reward normalization from checkpoint; "
            "collects a fresh stochastic pack with fixed replay seed; writes runner-defined per-env semantic sidecars; "
            "does not resume optimizer or claim to reconstruct the original hidden RNG stream."
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(_json_safe(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(_json_safe({"summary": str(out_dir / "summary.json"), "arms": arms}), sort_keys=True))


if __name__ == "__main__":
    main()
