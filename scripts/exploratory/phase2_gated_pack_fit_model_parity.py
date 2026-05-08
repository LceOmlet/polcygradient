#!/usr/bin/env python
"""Pack-to-fit_model parity for gated reward-path fixed lists.

This script validates the exploratory migration boundary without changing the
default exact-SCM or fit_model paths.  It installs the reward-path balancing
runtime patch only in this process, loads annotated fixed-list environments, and
checks that the trusted pack runner and the fit_model PPO bridge agree on:

* fixed-list rollout pack collection
* rollout-buffer tensors, returns, advantages, and VecNorm state
* PPO losses, gradients, and one train update
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None  # type: ignore

if cv2 is not None and not hasattr(cv2, "ocl"):
    class _OpenCVOclCompat:
        @staticmethod
        def setUseOpenCL(_enabled):
            return None

    cv2.ocl = _OpenCVOclCompat()  # type: ignore[attr-defined]

from phase2_reward_group_balance_runtime import (  # noqa: E402
    install_environment_prior_reward_group_balance_patch,
)
from stable_baselines3.common.logger import configure as configure_logger  # noqa: E402
from ticl.analysis.phase2_fit_model_migration_gradient_compare import (  # noqa: E402
    _gradient_diff_summary,
    _loss_diff_summary,
    _tensor_diff_summary,
)
from ticl.analysis.phase2_fit_model_pack_ppo_migration_parity import (  # noqa: E402
    _build_fit_model_algo,
    _copy_normalizer_state,
    _diff_buffer_summaries,
    _fill_and_measure_loss,
    _load_contract_config,
    _make_fixed_single_eval_prior,
    _normalizer_summary,
)
from ticl.analysis.phase2_gym_prior_pack_isomorphism_sentinel import (  # noqa: E402
    _json_safe,
    _summarize_pack,
)
from ticl.analysis.phase3_phase2_milestone_contract_gradient_compare import (  # noqa: E402
    _param_diff_summary,
)
from ticl.analysis.phase2_gym_prior_pack_training_runner import (  # noqa: E402
    _build_algo as _build_pack_algo,
    _collect_prior_policy_pack,
    _logger_values,
    _run_update_from_pack,
    _stable_digest,
)
from ticl.analysis.phase2_legacy_bar_longrun_probe import (  # noqa: E402
    _load_full_frozen_h,
    _sanitize_loaded_frozen_h_for_batch_use,
)
from ticl.config_utils import str2bool  # noqa: E402
from ticl.priors.environment_prior import EnvironmentPrior  # noqa: E402
from ticl.rlpfn_maintained_path import resolve_rlpfn_token_layout  # noqa: E402


DEFAULT_PROBE_DIRS = (
    "/home/chen/RLPFN/artifacts/phase2_exploratory_reward_group_balance_gated_n64_0506,"
    "/home/chen/RLPFN/artifacts/phase2_exploratory_reward_group_balance_gated_n64_seed6261_0506"
)


def _default_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _seed_all(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _parse_csv_list(text: str) -> list[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if torch.is_tensor(value):
        if value.numel() == 1:
            return _jsonable(value.detach().cpu().item())
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _max_abs_np(left: Any, right: Any) -> float:
    a = np.asarray(left)
    b = np.asarray(right)
    if a.shape != b.shape:
        return float("inf")
    if a.size == 0:
        return 0.0
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def _dict_numeric_diff(a: dict[str, Any], b: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in keys:
        if key in a and key in b:
            delta = None
            if isinstance(a[key], (int, float, np.generic)) and isinstance(b[key], (int, float, np.generic)):
                delta = float(b[key]) - float(a[key])
            out[key] = {"pack": _jsonable(a[key]), "fit_model": _jsonable(b[key]), "delta": delta}
    return out


def _load_selected_fixed_list(
    *,
    probe_dir: Path,
    rule: str,
    n_envs: int,
    prefer_keep: bool,
) -> tuple[list[dict[str, Any]], list[str], list[int], list[dict[str, Any]]]:
    csv_path = probe_dir / "selected_prior_envs.csv"
    with csv_path.open("r", encoding="utf-8") as handle:
        rows = [dict(row) for row in csv.DictReader(handle) if str(row.get("rule", "")) == str(rule)]
    if not rows:
        raise ValueError(f"No selected rows for rule={rule!r} in {csv_path}.")
    selected: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    if bool(prefer_keep):
        for row in rows:
            if str(row.get("selected_repair_kind", "")) != "original_healthy_keep":
                continue
            path = str(row.get("full_frozen_h_json", ""))
            if path and path not in seen_paths:
                selected.append(row)
                seen_paths.add(path)
                break
    for row in rows:
        if len(selected) >= int(n_envs):
            break
        path = str(row.get("full_frozen_h_json", ""))
        if path and path not in seen_paths:
            selected.append(row)
            seen_paths.add(path)
    if len(selected) != int(n_envs):
        raise ValueError(f"Need n_envs={int(n_envs)} rows for {rule}, got {len(selected)} from {csv_path}.")
    h_list: list[dict[str, Any]] = []
    h_paths: list[str] = []
    env_seeds: list[int] = []
    for row in selected:
        h, resolved = _load_full_frozen_h(row["full_frozen_h_json"])
        h_list.append(_sanitize_loaded_frozen_h_for_batch_use(h))
        h_paths.append(str(resolved))
        env_seeds.append(int(float(row.get("env_seed", row.get("seed", 0)))))
    return h_list, h_paths, env_seeds, selected


def _build_fixed_pack_algo(
    *,
    cfg: dict[str, Any],
    fixed_h_list: list[dict[str, Any]],
    fixed_env_seeds: list[int],
    device_obj: torch.device,
    build_seed: int,
    n_steps: int,
    ppo_batch_size: int,
    ppo_n_epochs: int,
    ppo_vf_coef: float,
    ppo_normalize_advantage: bool,
    ppo_target_kl: float | None,
    single_eval_pos: int,
    sb3_reward_normalization_enabled: bool,
    sb3_observation_normalization_enabled: bool,
    sb3_observation_normalization_clip: float,
    sb3_observation_normalization_epsilon: float,
):
    algo, vec_env = _build_pack_algo(
        cfg=copy.deepcopy(cfg),
        env_cfg=copy.deepcopy(cfg["prior"]["environment"]),
        frozen_h=None,
        frozen_h_list=copy.deepcopy(fixed_h_list),
        frozen_h_list_env_seeds=[int(v) for v in fixed_env_seeds],
        prior_mode="fixed_frozen_list",
        device_obj=device_obj,
        num_features=int(cfg["prior"]["num_features"]),
        n_envs=int(len(fixed_h_list)),
        n_steps=int(n_steps),
        build_seed=int(build_seed),
        fixed_single_eval_pos=int(single_eval_pos),
        sb3_reward_normalization_enabled=bool(sb3_reward_normalization_enabled),
        sb3_observation_normalization_enabled=bool(sb3_observation_normalization_enabled),
        sb3_observation_normalization_clip=float(sb3_observation_normalization_clip),
        sb3_observation_normalization_epsilon=float(sb3_observation_normalization_epsilon),
        gain_min=0.7,
        topology_state_gain_min=float(cfg["prior"]["environment"].get("reward_state_input_gain_fraction_conditioned_min", 0.7)),
        topology_action_gain_min=float(cfg["prior"]["environment"].get("reward_action_input_gain_fraction_conditioned_min", 0.06)),
        topology_state_to_action_ratio_max=float(
            cfg["prior"]["environment"].get("reward_state_to_action_gain_ratio_conditioned_max", 10.0)
        ),
        topology_max_attempts=int(cfg["prior"]["environment"].get("reward_topology_conditioned_sampling_max_attempts", 4096)),
        fixed_env_group_across_updates=False,
        ppo_learning_rate=float(cfg["optimizer"]["learning_rate"]),
        ppo_batch_size=int(ppo_batch_size),
        ppo_n_epochs=int(ppo_n_epochs),
        ppo_vf_coef=float(ppo_vf_coef),
        ppo_normalize_advantage=bool(ppo_normalize_advantage),
        ppo_target_kl=ppo_target_kl,
    )
    algo.set_logger(configure_logger(folder=None, format_strings=[]))
    return algo, vec_env


def _build_fixed_fit_algo(
    *,
    cfg: dict[str, Any],
    fixed_h_list: list[dict[str, Any]],
    fixed_env_seeds: list[int],
    device_obj: torch.device,
    build_seed: int,
    n_steps: int,
    ppo_batch_size: int,
    ppo_n_epochs: int,
    ppo_vf_coef: float,
    ppo_normalize_advantage: bool,
    ppo_target_kl: float | None,
    single_eval_pos: int,
    sb3_reward_normalization_enabled: bool,
    sb3_observation_normalization_enabled: bool,
    sb3_observation_normalization_clip: float,
    sb3_observation_normalization_epsilon: float,
):
    prior = _make_fixed_single_eval_prior(
        cfg["prior"]["environment"],
        single_eval_pos=int(single_eval_pos),
        ppo_reset_env_state_at_sep=True,
    )
    fixed_copy = [copy.deepcopy(h) for h in fixed_h_list]
    prior.config["reward_state_input_gain_fraction_conditioned_sampling_enabled"] = False
    prior.config["reward_topology_conditioned_sampling_enabled"] = False
    prior._phase2_fixed_env_group_h_list = fixed_copy
    prior._phase2_fixed_env_group_env_seeds = [int(v) for v in fixed_env_seeds]

    def _sample_fixed(batch_n, _fixed_h_list=fixed_copy):
        if int(batch_n) != len(_fixed_h_list):
            raise ValueError(
                "fixed fit_model parity environment group requires the same n_envs; "
                f"requested={int(batch_n)}, fixed={len(_fixed_h_list)}"
            )
        return [copy.deepcopy(h) for h in _fixed_h_list]

    prior._sample_batch_hypers = _sample_fixed
    return _build_fit_model_algo(
        cfg=cfg,
        env_prior=prior,
        device_obj=device_obj,
        build_seed=int(build_seed),
        n_envs=int(len(fixed_h_list)),
        n_steps=int(n_steps),
        ppo_batch_size=int(ppo_batch_size),
        ppo_n_epochs=int(ppo_n_epochs),
        ppo_vf_coef=float(ppo_vf_coef),
        ppo_normalize_advantage=bool(ppo_normalize_advantage),
        ppo_target_kl=ppo_target_kl,
        ppo_reset_env_state_at_sep=True,
        deterministic_batch_plan=True,
        sb3_reward_normalization_enabled=bool(sb3_reward_normalization_enabled),
        sb3_observation_normalization_enabled=bool(sb3_observation_normalization_enabled),
        sb3_observation_normalization_clip=float(sb3_observation_normalization_clip),
        sb3_observation_normalization_epsilon=float(sb3_observation_normalization_epsilon),
    )


def _collect_with_algo_pair(
    *,
    cfg: dict[str, Any],
    fixed_h_list: list[dict[str, Any]],
    fixed_env_seeds: list[int],
    device_obj: torch.device,
    build_seed: int,
    collect_seed: int,
    n_steps: int,
    ppo_batch_size: int,
    ppo_n_epochs: int,
    ppo_vf_coef: float,
    ppo_normalize_advantage: bool,
    ppo_target_kl: float | None,
    single_eval_pos: int,
    sb3_reward_normalization_enabled: bool,
    sb3_observation_normalization_enabled: bool,
    sb3_observation_normalization_clip: float,
    sb3_observation_normalization_epsilon: float,
    num_features: int,
    obs_slot_dim: int,
    action_slot_dim: int,
    next_state_target_dim: int,
) -> dict[str, Any]:
    pack_algo, pack_vec_env = _build_fixed_pack_algo(
        cfg=cfg,
        fixed_h_list=fixed_h_list,
        fixed_env_seeds=fixed_env_seeds,
        device_obj=device_obj,
        build_seed=build_seed,
        n_steps=n_steps,
        ppo_batch_size=ppo_batch_size,
        ppo_n_epochs=ppo_n_epochs,
        ppo_vf_coef=ppo_vf_coef,
        ppo_normalize_advantage=ppo_normalize_advantage,
        ppo_target_kl=ppo_target_kl,
        single_eval_pos=single_eval_pos,
        sb3_reward_normalization_enabled=sb3_reward_normalization_enabled,
        sb3_observation_normalization_enabled=sb3_observation_normalization_enabled,
        sb3_observation_normalization_clip=sb3_observation_normalization_clip,
        sb3_observation_normalization_epsilon=sb3_observation_normalization_epsilon,
    )
    fit_algo, _fit_callback, fit_vec_env, fit_bridge_summary, fit_bridge_kwargs = _build_fixed_fit_algo(
        cfg=cfg,
        fixed_h_list=fixed_h_list,
        fixed_env_seeds=fixed_env_seeds,
        device_obj=device_obj,
        build_seed=build_seed,
        n_steps=n_steps,
        ppo_batch_size=ppo_batch_size,
        ppo_n_epochs=ppo_n_epochs,
        ppo_vf_coef=ppo_vf_coef,
        ppo_normalize_advantage=ppo_normalize_advantage,
        ppo_target_kl=ppo_target_kl,
        single_eval_pos=single_eval_pos,
        sb3_reward_normalization_enabled=sb3_reward_normalization_enabled,
        sb3_observation_normalization_enabled=sb3_observation_normalization_enabled,
        sb3_observation_normalization_clip=sb3_observation_normalization_clip,
        sb3_observation_normalization_epsilon=sb3_observation_normalization_epsilon,
    )
    try:
        initial_param_diff = _param_diff_summary(pack_algo.policy, fit_algo.policy)
        _seed_all(int(collect_seed))
        pack, values, log_probs, meta = _collect_prior_policy_pack(
            algo=pack_algo,
            vec_env=pack_vec_env,
            n_steps=int(n_steps),
            seed=int(collect_seed),
            num_features=int(num_features),
            obs_slot_dim=int(obs_slot_dim),
            action_slot_dim=int(action_slot_dim),
            next_state_target_dim=int(next_state_target_dim),
            store_next_state_targets=False,
        )
        pack_normalizer = _normalizer_summary(pack_algo)
        _seed_all(int(collect_seed))
        fit_pack, fit_values, fit_log_probs, fit_meta = _collect_prior_policy_pack(
            algo=fit_algo,
            vec_env=fit_vec_env,
            n_steps=int(n_steps),
            seed=int(collect_seed),
            num_features=int(num_features),
            obs_slot_dim=int(obs_slot_dim),
            action_slot_dim=int(action_slot_dim),
            next_state_target_dim=int(next_state_target_dim),
            store_next_state_targets=False,
        )
        fit_normalizer = _normalizer_summary(fit_algo)
        pack_summary = _summarize_pack(pack, label="pack_collect")
        fit_summary = _summarize_pack(fit_pack, label="fit_model_collect")
        digest_keys = sorted(set(pack_summary.get("digests", {})) | set(fit_summary.get("digests", {})))
        digest_equal = {
            key: bool((pack_summary.get("digests", {}) or {}).get(key) == (fit_summary.get("digests", {}) or {}).get(key))
            for key in digest_keys
        }
        values_max_abs_delta = _max_abs_np(values, fit_values)
        log_probs_max_abs_delta = _max_abs_np(log_probs, fit_log_probs)
        meta_digest_equal = bool(_stable_digest(meta) == _stable_digest(fit_meta))
        normalizer_equal = bool(_stable_digest(pack_normalizer) == _stable_digest(fit_normalizer))
        hard_pass = bool(
            initial_param_diff["policy_param_max_abs_diff"] == 0.0
            and all(digest_equal.values())
            and values_max_abs_delta == 0.0
            and log_probs_max_abs_delta == 0.0
            and meta_digest_equal
            and normalizer_equal
        )
        return {
            "hard_pass": hard_pass,
            "pack": pack,
            "values": values,
            "log_probs": log_probs,
            "meta": meta,
            "pack_summary": pack_summary,
            "fit_summary": fit_summary,
            "initial_param_diff": initial_param_diff,
            "digest_equal": digest_equal,
            "values_max_abs_delta": values_max_abs_delta,
            "log_probs_max_abs_delta": log_probs_max_abs_delta,
            "meta_digest_equal": meta_digest_equal,
            "normalizer_equal": normalizer_equal,
            "pack_normalizer_after_collect": pack_normalizer,
            "fit_normalizer_after_collect": fit_normalizer,
            "fit_bridge_summary": fit_bridge_summary,
            "fit_bridge_kwargs_contract": {
                key: _jsonable(value)
                for key, value in fit_bridge_kwargs.items()
                if key not in {"model", "env_prior"}
            },
        }
    finally:
        pack_vec_env.close()
        fit_vec_env.close()
        del pack_algo, fit_algo


def _compare_update_from_fixed_pack(
    *,
    cfg: dict[str, Any],
    fixed_h_list: list[dict[str, Any]],
    fixed_env_seeds: list[int],
    pack: dict[str, np.ndarray],
    values: np.ndarray,
    log_probs: np.ndarray,
    device_obj: torch.device,
    build_seed: int,
    n_steps: int,
    ppo_batch_size: int,
    ppo_n_epochs: int,
    ppo_vf_coef: float,
    ppo_normalize_advantage: bool,
    ppo_target_kl: float | None,
    single_eval_pos: int,
    sb3_reward_normalization_enabled: bool,
    sb3_observation_normalization_enabled: bool,
    sb3_observation_normalization_clip: float,
    sb3_observation_normalization_epsilon: float,
    num_features: int,
    action_slot_dim: int,
    next_state_target_dim: int,
) -> dict[str, Any]:
    def _new_pair():
        pack_algo, pack_vec_env = _build_fixed_pack_algo(
            cfg=cfg,
            fixed_h_list=fixed_h_list,
            fixed_env_seeds=fixed_env_seeds,
            device_obj=device_obj,
            build_seed=build_seed,
            n_steps=n_steps,
            ppo_batch_size=ppo_batch_size,
            ppo_n_epochs=ppo_n_epochs,
            ppo_vf_coef=ppo_vf_coef,
            ppo_normalize_advantage=ppo_normalize_advantage,
            ppo_target_kl=ppo_target_kl,
            single_eval_pos=single_eval_pos,
            sb3_reward_normalization_enabled=sb3_reward_normalization_enabled,
            sb3_observation_normalization_enabled=sb3_observation_normalization_enabled,
            sb3_observation_normalization_clip=sb3_observation_normalization_clip,
            sb3_observation_normalization_epsilon=sb3_observation_normalization_epsilon,
        )
        fit_algo, _fit_callback, fit_vec_env, fit_bridge_summary, fit_bridge_kwargs = _build_fixed_fit_algo(
            cfg=cfg,
            fixed_h_list=fixed_h_list,
            fixed_env_seeds=fixed_env_seeds,
            device_obj=device_obj,
            build_seed=build_seed,
            n_steps=n_steps,
            ppo_batch_size=ppo_batch_size,
            ppo_n_epochs=ppo_n_epochs,
            ppo_vf_coef=ppo_vf_coef,
            ppo_normalize_advantage=ppo_normalize_advantage,
            ppo_target_kl=ppo_target_kl,
            single_eval_pos=single_eval_pos,
            sb3_reward_normalization_enabled=sb3_reward_normalization_enabled,
            sb3_observation_normalization_enabled=sb3_observation_normalization_enabled,
            sb3_observation_normalization_clip=sb3_observation_normalization_clip,
            sb3_observation_normalization_epsilon=sb3_observation_normalization_epsilon,
        )
        _copy_normalizer_state(pack_algo, fit_algo)
        return pack_algo, pack_vec_env, fit_algo, fit_vec_env, fit_bridge_summary, fit_bridge_kwargs

    pack_loss_algo, pack_loss_vec_env, fit_loss_algo, fit_loss_vec_env, bridge_summary, bridge_kwargs = _new_pair()
    try:
        pack_loss = _fill_and_measure_loss(
            pack_loss_algo,
            pack=pack,
            values=values,
            log_probs=log_probs,
            num_features=num_features,
            action_slot_dim=action_slot_dim,
            next_state_target_dim=next_state_target_dim,
            single_eval_pos=single_eval_pos,
        )
        _copy_normalizer_state(pack_loss_algo, fit_loss_algo)
        fit_loss = _fill_and_measure_loss(
            fit_loss_algo,
            pack=pack,
            values=values,
            log_probs=log_probs,
            num_features=num_features,
            action_slot_dim=action_slot_dim,
            next_state_target_dim=next_state_target_dim,
            single_eval_pos=single_eval_pos,
        )
        rollout_diff = _tensor_diff_summary(pack_loss["rollout_tensors"], fit_loss["rollout_tensors"])
        grad_diff = _gradient_diff_summary(pack_loss["grad_vector"], fit_loss["grad_vector"])
        loss_diff = _loss_diff_summary(pack_loss, fit_loss)
        buffer_diff = _diff_buffer_summaries(pack_loss["buffer_summary"], fit_loss["buffer_summary"])
    finally:
        pack_loss_vec_env.close()
        fit_loss_vec_env.close()
        del pack_loss_algo, fit_loss_algo

    pack_update_algo, pack_update_vec_env, fit_update_algo, fit_update_vec_env, _, _ = _new_pair()
    try:
        pre_update_param_diff = _param_diff_summary(pack_update_algo.policy, fit_update_algo.policy)
        _seed_all(int(build_seed) + 9001)
        pack_update, pack_bridge = _run_update_from_pack(
            algo=pack_update_algo,
            pack=pack,
            values=values,
            log_probs=log_probs,
            num_features=num_features,
            action_slot_dim=action_slot_dim,
            next_state_target_dim=next_state_target_dim,
            single_eval_pos=single_eval_pos,
        )
        _copy_normalizer_state(pack_update_algo, fit_update_algo)
        _seed_all(int(build_seed) + 9001)
        fit_update, fit_bridge = _run_update_from_pack(
            algo=fit_update_algo,
            pack=pack,
            values=values,
            log_probs=log_probs,
            num_features=num_features,
            action_slot_dim=action_slot_dim,
            next_state_target_dim=next_state_target_dim,
            single_eval_pos=single_eval_pos,
        )
        post_update_param_diff = _param_diff_summary(pack_update_algo.policy, fit_update_algo.policy)
        logger_diff = _dict_numeric_diff(
            _logger_values(pack_update_algo),
            _logger_values(fit_update_algo),
            [
                "train/loss",
                "train/policy_gradient_loss",
                "train/value_loss",
                "train/entropy_loss",
                "train/approx_kl",
                "train/clip_fraction",
                "train/explained_variance",
                "train/normalized_q_value_weight",
                "train/next_state_flow_matching_weight",
            ],
        )
    finally:
        pack_update_vec_env.close()
        fit_update_vec_env.close()
        del pack_update_algo, fit_update_algo

    hard_pass = bool(
        pre_update_param_diff["policy_param_max_abs_diff"] == 0.0
        and rollout_diff["max_abs_rollout_tensor_delta"] == 0.0
        and grad_diff["grad_delta_max_abs"] <= 1e-7
        and post_update_param_diff["policy_param_max_abs_diff"] <= 1e-7
        and pack_update.get("exception") is None
        and fit_update.get("exception") is None
    )
    return {
        "hard_pass": hard_pass,
        "initial_param_diff": pre_update_param_diff,
        "loss_diff": loss_diff,
        "buffer_diff": buffer_diff,
        "rollout_diff": rollout_diff,
        "gradient_diff": grad_diff,
        "one_train_update_compare": {
            "pack_update": pack_update,
            "fit_model_update": fit_update,
            "pack_bridge": pack_bridge,
            "fit_model_bridge": fit_bridge,
            "post_update_param_diff": post_update_param_diff,
            "logger_diff": logger_diff,
        },
        "bridge_summary": bridge_summary,
        "bridge_kwargs_contract": {
            key: _jsonable(value)
            for key, value in bridge_kwargs.items()
            if key not in {"model", "env_prior"}
        },
    }


def _audit_case(
    *,
    case_name: str,
    probe_dir: Path,
    rule: str,
    args: argparse.Namespace,
    cfg: dict[str, Any],
    device_obj: torch.device,
) -> dict[str, Any]:
    fixed_h_list, h_paths, env_seeds, selected_rows = _load_selected_fixed_list(
        probe_dir=probe_dir,
        rule=rule,
        n_envs=int(args.n_envs),
        prefer_keep=bool(args.prefer_keep),
    )
    layout = resolve_rlpfn_token_layout(cfg["prior"]["environment"], num_features=int(cfg["prior"]["num_features"]))
    num_features = int(cfg["prior"]["num_features"])
    obs_slot_dim = int(layout["obs_slot_dim"])
    action_slot_dim = int(layout["action_slot_dim"])
    dim_probe = EnvironmentPrior(copy.deepcopy(cfg["prior"]["environment"]))
    next_state_target_dim = int(max(1, dim_probe._resolve_dim_upper_bound(cfg["prior"]["environment"].get("state_dim", None), default=0)))
    collect = _collect_with_algo_pair(
        cfg=cfg,
        fixed_h_list=fixed_h_list,
        fixed_env_seeds=env_seeds,
        device_obj=device_obj,
        build_seed=int(args.build_seed),
        collect_seed=int(args.collect_seed),
        n_steps=int(args.n_steps),
        ppo_batch_size=int(args.ppo_batch_size),
        ppo_n_epochs=int(args.ppo_n_epochs),
        ppo_vf_coef=float(args.ppo_vf_coef),
        ppo_normalize_advantage=bool(args.ppo_normalize_advantage),
        ppo_target_kl=args.ppo_target_kl,
        single_eval_pos=int(args.single_eval_pos),
        sb3_reward_normalization_enabled=bool(args.sb3_reward_normalization_enabled),
        sb3_observation_normalization_enabled=bool(args.sb3_observation_normalization_enabled),
        sb3_observation_normalization_clip=float(args.sb3_observation_normalization_clip),
        sb3_observation_normalization_epsilon=float(args.sb3_observation_normalization_epsilon),
        num_features=num_features,
        obs_slot_dim=obs_slot_dim,
        action_slot_dim=action_slot_dim,
        next_state_target_dim=next_state_target_dim,
    )
    update = _compare_update_from_fixed_pack(
        cfg=cfg,
        fixed_h_list=fixed_h_list,
        fixed_env_seeds=env_seeds,
        pack=collect["pack"],
        values=collect["values"],
        log_probs=collect["log_probs"],
        device_obj=device_obj,
        build_seed=int(args.build_seed),
        n_steps=int(args.n_steps),
        ppo_batch_size=int(args.ppo_batch_size),
        ppo_n_epochs=int(args.ppo_n_epochs),
        ppo_vf_coef=float(args.ppo_vf_coef),
        ppo_normalize_advantage=bool(args.ppo_normalize_advantage),
        ppo_target_kl=args.ppo_target_kl,
        single_eval_pos=int(args.single_eval_pos),
        sb3_reward_normalization_enabled=bool(args.sb3_reward_normalization_enabled),
        sb3_observation_normalization_enabled=bool(args.sb3_observation_normalization_enabled),
        sb3_observation_normalization_clip=float(args.sb3_observation_normalization_clip),
        sb3_observation_normalization_epsilon=float(args.sb3_observation_normalization_epsilon),
        num_features=num_features,
        action_slot_dim=action_slot_dim,
        next_state_target_dim=next_state_target_dim,
    )
    row_kinds = [str(row.get("selected_repair_kind")) for row in selected_rows]
    hard_pass = bool(collect["hard_pass"] and update["hard_pass"])
    public_collect = {k: v for k, v in collect.items() if k not in {"pack", "values", "log_probs", "meta"}}
    return {
        "case": str(case_name),
        "hard_pass": hard_pass,
        "probe_dir": str(probe_dir),
        "rule": str(rule),
        "n_envs": int(args.n_envs),
        "n_steps": int(args.n_steps),
        "selected_h_paths": h_paths,
        "selected_env_seeds": env_seeds,
        "selected_repair_kind_counts": dict(Counter(row_kinds)),
        "collection_parity": public_collect,
        "update_parity": update,
        "contract": {
            "fixed_list_annotations_interpreted_by_same_runtime_patch": True,
            "store_next_state_targets": False,
            "ppo_n_epochs": int(args.ppo_n_epochs),
            "ppo_batch_size": int(args.ppo_batch_size),
            "ppo_vf_coef": float(args.ppo_vf_coef),
            "ppo_normalize_advantage": bool(args.ppo_normalize_advantage),
            "normalized_q_value_weight": 0.0,
            "next_state_flow_matching_weight": 0.0,
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    install_environment_prior_reward_group_balance_patch()
    if int(args.ppo_batch_size) > int(args.n_envs) * int(args.n_steps):
        raise ValueError("ppo_batch_size must not exceed n_envs * n_steps.")
    device_obj = torch.device(str(args.device))
    cfg = _load_contract_config(args)
    probe_dirs = [Path(p).expanduser().resolve() for p in _parse_csv_list(args.probe_dirs)]
    rules = _parse_csv_list(args.rules)
    cases: list[dict[str, Any]] = []
    for probe_idx, probe_dir in enumerate(probe_dirs):
        for rule in rules:
            case_name = f"{probe_dir.name}:{rule}"
            cases.append(
                _audit_case(
                    case_name=case_name,
                    probe_dir=probe_dir,
                    rule=rule,
                    args=args,
                    cfg=cfg,
                    device_obj=device_obj,
                )
            )
            if int(args.max_cases) > 0 and len(cases) >= int(args.max_cases):
                break
        if int(args.max_cases) > 0 and len(cases) >= int(args.max_cases):
            break
    hard_pass = bool(cases and all(bool(case["hard_pass"]) for case in cases))
    report = {
        "analysis_entry": "phase2_gated_pack_fit_model_parity",
        "exploratory_only": True,
        "hard_pass": hard_pass,
        "contract": {
            "does_not_modify_exact_scm": True,
            "does_not_modify_default_prior_generator": True,
            "does_not_train_fit_model_longrun": True,
            "installs_reward_group_balance_patch_only_in_this_process": True,
            "migration_scope": "annotated fixed-list trusted pack runner to fit_model PPO bridge",
        },
        "args": _jsonable(vars(args)),
        "cases": cases,
    }
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "gated_pack_fit_model_parity_report.json"
    report_path.write_text(json.dumps(_jsonable(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_lines = [
        "# Gated Pack-to-fit_model Parity",
        "",
        f"hard_pass: {hard_pass}",
        "",
        "## Cases",
    ]
    for case in cases:
        update = case["update_parity"]
        collect = case["collection_parity"]
        md_lines.append(
            f"- {case['case']}: hard={case['hard_pass']}, "
            f"collect={collect.get('hard_pass')}, update={update.get('hard_pass')}, "
            f"grad_delta={update.get('gradient_diff', {}).get('grad_delta_max_abs')}, "
            f"post_param_delta={update.get('one_train_update_compare', {}).get('post_update_param_diff', {}).get('policy_param_max_abs_diff')}"
        )
    md_path = out_dir / "gated_pack_fit_model_parity_summary.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(json.dumps({"hard_pass": hard_pass, "report": str(report_path)}, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-dirs", type=str, default=DEFAULT_PROBE_DIRS)
    parser.add_argument("--rules", type=str, default="gym_full_obs_action_range,gym_q90_obs_action_range")
    parser.add_argument("--max-cases", type=int, default=2)
    parser.add_argument("--prefer-keep", type=str2bool, default=True)
    parser.add_argument("--output-dir", type=str, default="/home/chen/RLPFN/artifacts/phase2_exploratory_gated_pack_fit_model_parity_0506")
    parser.add_argument("--device", type=str, default=_default_device())
    parser.add_argument("--build-seed", type=int, default=5052)
    parser.add_argument("--collect-seed", type=int, default=6060)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--n-steps", type=int, default=64)
    parser.add_argument("--single-eval-pos", type=int, default=32)
    parser.add_argument("--ppo-batch-size", type=int, default=128)
    parser.add_argument("--ppo-n-epochs", type=int, default=4)
    parser.add_argument("--ppo-vf-coef", type=float, default=0.1)
    parser.add_argument("--ppo-normalize-advantage", type=str2bool, default=True)
    parser.add_argument("--ppo-target-kl", type=float, default=None)
    parser.add_argument("--topology-state-gain-min", type=float, default=0.7)
    parser.add_argument("--topology-action-gain-min", type=float, default=0.06)
    parser.add_argument("--topology-state-to-action-ratio-max", type=float, default=10.0)
    parser.add_argument("--topology-max-attempts", type=int, default=4096)
    parser.add_argument("--sb3-reward-normalization-enabled", type=str2bool, default=True)
    parser.add_argument("--sb3-observation-normalization-enabled", type=str2bool, default=True)
    parser.add_argument("--sb3-observation-normalization-clip", type=float, default=10.0)
    parser.add_argument("--sb3-observation-normalization-epsilon", type=float, default=1e-8)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
