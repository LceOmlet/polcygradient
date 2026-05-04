#!/usr/bin/env python3
"""Same-config gradient parity for the BipedalWalker gym-pack PPO update.

This is an analysis-only probe.  It reuses the trusted phase2 gym-pack runner
to collect one BipedalWalker pack, then replays the same rollout buffer through
the vendored official RecurrentPPO.train loop and the local MaskedRecurrentPPO
train loop under controlled ablations.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _patch_cv2_for_vendored_sb3() -> None:
    try:
        import cv2  # type: ignore
    except Exception:
        return
    if hasattr(cv2, "ocl"):
        return

    class _OpenCLShim:
        @staticmethod
        def setUseOpenCL(flag: bool) -> None:
            del flag

    cv2.ocl = _OpenCLShim()  # type: ignore[attr-defined]


_patch_cv2_for_vendored_sb3()

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stable_baselines3.common.running_mean_std import RunningMeanStd  # noqa: E402
from sb3_contrib.ppo_recurrent.ppo_recurrent import RecurrentPPO  # noqa: E402

from ticl.analysis.critic_free_single_env_audit import _seed_all  # noqa: E402
from ticl.analysis.critic_value_fit_probe import _clone_h_repeated  # noqa: E402
from ticl.analysis.phase2_gym_prior_pack_isomorphism_sentinel import (  # noqa: E402
    DEFAULT_FROZEN_H_JSON,
    _fill_existing_rollout_buffer_from_pack,
    _json_safe,
    _summarize_pack,
)
from ticl.analysis.phase2_gym_prior_pack_training_runner import (  # noqa: E402
    _build_algo,
    _collect_gym_policy_pack,
    _load_full_frozen_h,
    _sanitize_loaded_frozen_h_for_batch_use,
)
from ticl.model_configs import get_model_default_config  # noqa: E402
from ticl.priors.environment_prior import EnvironmentPrior  # noqa: E402
from ticl.rlpfn_maintained_path import resolve_rlpfn_token_layout, validate_rlpfn_maintained_path_config  # noqa: E402


DEFAULT_CHECKPOINT = (
    "/home/chen/RLPFN/artifacts/"
    "phase2_clean_official_sidecar_rawreward_gym_bipedalwalker_only_n64_u100_obsrewardnorm_gpu0_0502/"
    "checkpoints/gym_update_000000.pt"
)
DEFAULT_OUTPUT_JSON = (
    "/home/chen/RLPFN/artifacts/"
    "phase2_bipedalwalker_gympack_gradient_parity_0502/"
    "gradient_parity_report.json"
)


class _CaptureLogger:
    def __init__(self) -> None:
        self.rows: dict[str, Any] = {}

    def record(self, key: str, value: Any, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        if torch.is_tensor(value):
            value = value.detach().cpu().item()
        if isinstance(value, np.generic):
            value = value.item()
        self.rows[str(key)] = value


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _tensor_stats(tensor: torch.Tensor) -> dict[str, Any]:
    arr = tensor.detach().to(dtype=torch.float64).cpu().numpy().reshape(-1)
    return {
        "numel": int(arr.size),
        "finite": bool(np.isfinite(arr).all()),
        "mean": float(arr.mean()) if arr.size else 0.0,
        "std": float(arr.std()) if arr.size else 0.0,
        "l2": float(np.linalg.norm(arr)) if arr.size else 0.0,
        "absmax": float(np.max(np.abs(arr))) if arr.size else 0.0,
    }


def _named_vector(policy, *, kind: str, before: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    for name, param in policy.named_parameters():
        if not bool(param.requires_grad):
            continue
        if kind == "param":
            value = param.detach()
        elif kind == "grad":
            value = torch.zeros_like(param.detach()) if param.grad is None else param.grad.detach()
        elif kind == "delta":
            if before is None or name not in before:
                raise KeyError(f"Missing before snapshot for {name}")
            value = param.detach() - before[name].to(device=param.device, dtype=param.dtype)
        else:
            raise ValueError(kind)
        chunks.append(value.reshape(-1).to(device="cpu", dtype=torch.float64))
    if not chunks:
        return torch.zeros((0,), dtype=torch.float64)
    return torch.cat(chunks, dim=0)


def _param_snapshot(policy) -> dict[str, torch.Tensor]:
    return {
        name: param.detach().cpu().clone()
        for name, param in policy.named_parameters()
        if bool(param.requires_grad)
    }


def _diff_stats(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    if int(a.numel()) != int(b.numel()):
        return {"same_numel": False, "a_numel": int(a.numel()), "b_numel": int(b.numel())}
    diff = a.to(dtype=torch.float64) - b.to(dtype=torch.float64)
    return {
        "same_numel": True,
        "numel": int(diff.numel()),
        "l2": float(torch.linalg.vector_norm(diff).cpu()) if diff.numel() else 0.0,
        "absmax": float(diff.abs().max().cpu()) if diff.numel() else 0.0,
        "mean_abs": float(diff.abs().mean().cpu()) if diff.numel() else 0.0,
    }


def _logger_diff(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "train/policy_gradient_loss",
        "train/value_loss",
        "train/entropy_loss",
        "train/approx_kl",
        "train/clip_fraction",
        "train/loss",
        "train/std",
        "train/subbatches",
        "train/outer_batches",
    ]
    out: dict[str, Any] = {}
    for key in keys:
        av = _finite_float(a.get(key))
        bv = _finite_float(b.get(key))
        if av is None or bv is None:
            out[key] = {"a": av, "b": bv, "abs_diff": None}
        else:
            out[key] = {"a": av, "b": bv, "abs_diff": abs(av - bv)}
    return out


def _load_checkpoint_into_algo(algo, checkpoint_path: str | Path | None) -> dict[str, Any]:
    if checkpoint_path is None or str(checkpoint_path).strip() == "":
        return {"loaded": False}
    path = Path(checkpoint_path).expanduser().resolve()
    payload = torch.load(path, map_location=algo.device, weights_only=False)
    algo.policy.load_state_dict(payload["policy_state_dict"], strict=True)
    reward_norm = payload.get("reward_norm") or {}
    algo._rwkv_sb3_reward_normalization_enabled = bool(reward_norm.get("enabled", False))
    algo._rwkv_sb3_reward_normalization_returns = np.asarray(
        reward_norm.get("returns", np.zeros((int(algo.n_envs),), dtype=np.float32)),
        dtype=np.float32,
    )
    ret_rms = RunningMeanStd(shape=())
    if "ret_rms_mean" in reward_norm:
        ret_rms.mean = np.asarray(reward_norm["ret_rms_mean"])
        ret_rms.var = np.asarray(reward_norm["ret_rms_var"])
        ret_rms.count = float(reward_norm["ret_rms_count"])
    algo._rwkv_sb3_reward_normalization_ret_rms = ret_rms
    obs_norm = payload.get("obs_norm") or {}
    algo._rwkv_sb3_observation_normalization_enabled = bool(obs_norm.get("enabled", False))
    algo._rwkv_sb3_observation_normalization_mean = np.asarray(obs_norm.get("mean", []), dtype=np.float64)
    algo._rwkv_sb3_observation_normalization_var = np.asarray(obs_norm.get("var", []), dtype=np.float64)
    algo._rwkv_sb3_observation_normalization_count = np.asarray(obs_norm.get("count", []), dtype=np.float64)
    algo._rwkv_sb3_observation_normalization_clip = float(obs_norm.get("clip", 10.0))
    algo._rwkv_sb3_observation_normalization_epsilon = float(obs_norm.get("epsilon", 1e-8))
    return {
        "loaded": True,
        "path": str(path),
        "checkpoint_format": str(payload.get("checkpoint_format")),
        "update_idx": int(payload.get("update_idx", -1)),
        "n_updates": int(payload.get("n_updates", -1)),
    }


def _prepare_config(*, frozen_h_json: str | Path) -> dict[str, Any]:
    cfg = copy.deepcopy(get_model_default_config("rlpfn"))
    validate_rlpfn_maintained_path_config(cfg)
    env_cfg = copy.deepcopy(cfg["prior"]["environment"])
    num_features = int(cfg["prior"]["num_features"])
    layout = resolve_rlpfn_token_layout(env_cfg, num_features=num_features)
    frozen_h, frozen_h_path = _load_full_frozen_h(frozen_h_json)
    frozen_h = _sanitize_loaded_frozen_h_for_batch_use(frozen_h)
    dim_probe_prior = EnvironmentPrior(copy.deepcopy(env_cfg))
    next_state_target_dim = int(
        max(
            int((frozen_h or {}).get("state_dim", 1)),
            int(dim_probe_prior._resolve_dim_upper_bound(env_cfg.get("state_dim", None), default=0)),
            1,
        )
    )
    return {
        "cfg": cfg,
        "env_cfg": env_cfg,
        "frozen_h": frozen_h,
        "frozen_h_path": str(frozen_h_path),
        "num_features": int(num_features),
        "obs_slot_dim": int(layout["obs_slot_dim"]),
        "action_slot_dim": int(layout["action_slot_dim"]),
        "next_state_target_dim": int(next_state_target_dim),
    }


def _make_algo(
    *,
    prepared: dict[str, Any],
    device_obj: torch.device,
    n_envs: int,
    n_steps: int,
    seed: int,
    checkpoint_path: str | Path | None,
):
    algo, vec_env = _build_algo(
        cfg=copy.deepcopy(prepared["cfg"]),
        env_cfg=copy.deepcopy(prepared["env_cfg"]),
        frozen_h=copy.deepcopy(prepared["frozen_h"]),
        frozen_h_list=None,
        frozen_h_list_env_seeds=None,
        prior_mode="fixed_frozen",
        device_obj=device_obj,
        num_features=int(prepared["num_features"]),
        n_envs=int(n_envs),
        n_steps=int(n_steps),
        build_seed=int(seed),
        sb3_reward_normalization_enabled=True,
        sb3_observation_normalization_enabled=True,
        sb3_observation_normalization_clip=10.0,
        sb3_observation_normalization_epsilon=1e-8,
        gain_min=0.7,
        topology_state_gain_min=0.7,
        topology_action_gain_min=0.06,
        topology_state_to_action_ratio_max=15.0,
        topology_max_attempts=512,
        fixed_env_group_across_updates=True,
    )
    checkpoint = _load_checkpoint_into_algo(algo, checkpoint_path)
    return algo, vec_env, checkpoint


def _make_pack_variant(pack: dict[str, np.ndarray], *, action_mask_mode: str) -> dict[str, np.ndarray]:
    out = dict(pack)
    if action_mask_mode == "actual":
        out["action_masks"] = np.asarray(pack["action_masks"], dtype=np.float32).copy()
    elif action_mask_mode == "all_ones":
        out["action_masks"] = np.ones_like(np.asarray(pack["action_masks"], dtype=np.float32))
    else:
        raise ValueError(f"Unsupported action_mask_mode={action_mask_mode!r}")
    return out


def _fill_buffer(
    *,
    algo,
    pack: dict[str, np.ndarray],
    values: np.ndarray,
    log_probs: np.ndarray,
    prepared: dict[str, Any],
    single_eval_pos: int,
    compute_returns: bool = True,
) -> dict[str, Any]:
    return _fill_existing_rollout_buffer_from_pack(
        algo.rollout_buffer,
        pack,
        num_features=int(prepared["num_features"]),
        action_slot_dim=int(prepared["action_slot_dim"]),
        next_state_target_dim=int(prepared["next_state_target_dim"]),
        single_eval_pos=int(single_eval_pos),
        values_by_step_env=values,
        log_probs_by_step_env=log_probs,
        last_values_by_env=pack.get("last_values"),
        compute_returns=bool(compute_returns),
    )


def _replay_values_log_probs(
    *,
    algo,
    pack: dict[str, np.ndarray],
    prepared: dict[str, Any],
    single_eval_pos: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    n_steps, n_envs = np.asarray(pack["rewards"]).shape
    zeros_v = np.zeros((int(n_steps), int(n_envs)), dtype=np.float32)
    zeros_lp = np.zeros_like(zeros_v)
    _fill_buffer(
        algo=algo,
        pack=pack,
        values=zeros_v,
        log_probs=zeros_lp,
        prepared=prepared,
        single_eval_pos=int(single_eval_pos),
        compute_returns=False,
    )
    sample = next(algo.rollout_buffer.get_gpu_flat(algo.batch_size))
    with torch.no_grad():
        outputs = algo.policy.evaluate_actions_with_hidden_flat(
            sample.observations,
            sample.actions,
            seq_lengths=sample.seq_lengths,
            action_masks=sample.action_masks,
        )
    values_flat = outputs["values"].detach().reshape(-1).cpu().numpy().astype(np.float32, copy=False)
    logp_flat = outputs["log_prob"].detach().reshape(-1).cpu().numpy().astype(np.float32, copy=False)
    values = np.zeros((int(n_steps), int(n_envs)), dtype=np.float32)
    log_probs = np.zeros_like(values)
    for idx, (step_idx, env_idx) in enumerate(zip(sample.flat_step_indices, sample.flat_env_indices)):
        values[int(step_idx), int(env_idx)] = values_flat[int(idx)]
        log_probs[int(step_idx), int(env_idx)] = logp_flat[int(idx)]
    replay = {
        "values": {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "absmax": float(np.max(np.abs(values))),
        },
        "log_probs": {
            "mean": float(log_probs.mean()),
            "std": float(log_probs.std()),
            "absmax": float(np.max(np.abs(log_probs))),
        },
    }
    return values, log_probs, replay


def _patch_policy_value_loss_to_official_mse(policy) -> None:
    def _official_mse_value_loss(
        self,
        *,
        value_logits,
        targets,
        objective_mask,
        value_target_bucket_idx=None,
    ):
        del value_target_bucket_idx
        mask = objective_mask.reshape(-1).to(device=value_logits.device, dtype=torch.bool)
        values = self._value_from_logits(value_logits).reshape(-1)
        targets_flat = targets.reshape(-1).to(device=values.device, dtype=torch.float32)
        if not bool(mask.any().item()):
            return values.new_zeros(())
        return torch.mean(((targets_flat - values) ** 2)[mask])

    policy.compute_value_loss_from_logits = types.MethodType(_official_mse_value_loss, policy)


def _run_train_case(
    *,
    case_name: str,
    train_impl: str,
    prepared: dict[str, Any],
    pack: dict[str, np.ndarray],
    values: np.ndarray,
    log_probs: np.ndarray,
    device_obj: torch.device,
    n_envs: int,
    n_steps: int,
    seed: int,
    checkpoint_path: str | Path | None,
    single_eval_pos: int,
    force_one_sequence_subbatch: bool,
    patch_value_loss_to_mse: bool,
) -> dict[str, Any]:
    _seed_all(int(seed) + 17)
    algo, vec_env, checkpoint = _make_algo(
        prepared=prepared,
        device_obj=device_obj,
        n_envs=int(n_envs),
        n_steps=int(n_steps),
        seed=int(seed),
        checkpoint_path=checkpoint_path,
    )
    try:
        logger = _CaptureLogger()
        algo.set_logger(logger)
        if bool(patch_value_loss_to_mse):
            _patch_policy_value_loss_to_official_mse(algo.policy)
        if bool(force_one_sequence_subbatch):
            algo._resolve_sequence_subbatch_size = lambda rollout_data: int(
                getattr(rollout_data, "n_seq", rollout_data.lstm_states.pi[0].shape[1])
            )
        buffer_summary = _fill_buffer(
            algo=algo,
            pack=pack,
            values=values,
            log_probs=log_probs,
            prepared=prepared,
            single_eval_pos=int(single_eval_pos),
            compute_returns=True,
        )
        before = _param_snapshot(algo.policy)
        exception = None
        try:
            if train_impl == "official":
                RecurrentPPO.train(algo)  # type: ignore[arg-type]
            elif train_impl == "local":
                algo.train()
            else:
                raise ValueError(f"Unsupported train_impl={train_impl!r}")
        except Exception as exc:  # pragma: no cover - report, do not hide
            exception = repr(exc)
        grad_vec = _named_vector(algo.policy, kind="grad")
        delta_vec = _named_vector(algo.policy, kind="delta", before=before)
        return {
            "case_name": str(case_name),
            "train_impl": str(train_impl),
            "exception": exception,
            "checkpoint": checkpoint,
            "single_eval_pos": int(single_eval_pos),
            "force_one_sequence_subbatch": bool(force_one_sequence_subbatch),
            "patch_value_loss_to_mse": bool(patch_value_loss_to_mse),
            "buffer_summary": buffer_summary,
            "logger": dict(logger.rows),
            "grad": _tensor_stats(grad_vec),
            "param_delta": _tensor_stats(delta_vec),
            "_grad_vec": grad_vec,
            "_delta_vec": delta_vec,
        }
    finally:
        vec_env.close()
        del algo
        if torch.cuda.is_available() and device_obj.type == "cuda":
            torch.cuda.empty_cache()


def _public_case(case: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in case.items() if not str(k).startswith("_")}


def _compare_cases(name: str, a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": str(name),
        "a": str(a["case_name"]),
        "b": str(b["case_name"]),
        "logger_abs_diffs": _logger_diff(a.get("logger", {}), b.get("logger", {})),
        "grad_diff": _diff_stats(a["_grad_vec"], b["_grad_vec"]),
        "param_delta_diff": _diff_stats(a["_delta_vec"], b["_delta_vec"]),
    }


def run_probe(
    *,
    output_json: str | Path,
    checkpoint_path: str | Path | None,
    device: str,
    n_envs: int,
    n_steps: int,
    seed: int,
    gym_env_id: str,
    single_eval_pos: int,
    frozen_h_json: str | Path,
) -> dict[str, Any]:
    device_obj = torch.device(device)
    prepared = _prepare_config(frozen_h_json=frozen_h_json)

    source_algo, source_vec_env, checkpoint = _make_algo(
        prepared=prepared,
        device_obj=device_obj,
        n_envs=int(n_envs),
        n_steps=int(n_steps),
        seed=int(seed),
        checkpoint_path=checkpoint_path,
    )
    try:
        gym_env_ids = ",".join([str(gym_env_id)] * int(n_envs))
        pack, rollout_values, rollout_log_probs, meta = _collect_gym_policy_pack(
            algo=source_algo,
            gym_env_ids=[str(gym_env_id)] * int(n_envs),
            n_steps=int(n_steps),
            seed=int(seed),
            num_features=int(prepared["num_features"]),
            obs_slot_dim=int(prepared["obs_slot_dim"]),
            action_slot_dim=int(prepared["action_slot_dim"]),
            next_state_target_dim=int(prepared["next_state_target_dim"]),
            single_eval_pos=int(single_eval_pos),
            terminal_token_enabled=True,
            device_obj=device_obj,
        )
        del gym_env_ids
        pack_summary = _summarize_pack(pack, label="bipedalwalker_gym_pack_gradient_parity")

        actual_pack = _make_pack_variant(pack, action_mask_mode="actual")
        fullslot_pack = _make_pack_variant(pack, action_mask_mode="all_ones")
        full_values, full_log_probs, full_replay = _replay_values_log_probs(
            algo=source_algo,
            pack=fullslot_pack,
            prepared=prepared,
            single_eval_pos=0,
        )
        active_values, active_log_probs, active_replay = _replay_values_log_probs(
            algo=source_algo,
            pack=actual_pack,
            prepared=prepared,
            single_eval_pos=int(single_eval_pos),
        )
        old_log_prob_replay_diff = {
            "actual_collector_vs_active_replay_absmax": float(
                np.max(np.abs(np.asarray(rollout_log_probs) - np.asarray(active_log_probs)))
            ),
            "actual_collector_vs_active_replay_mean_abs": float(
                np.mean(np.abs(np.asarray(rollout_log_probs) - np.asarray(active_log_probs)))
            ),
            "actual_mask_logprob_mean": float(np.mean(active_log_probs)),
            "fullslot_logprob_mean": float(np.mean(full_log_probs)),
            "fullslot_minus_actual_logprob_mean": float(np.mean(full_log_probs - active_log_probs)),
        }
    finally:
        source_vec_env.close()
        del source_algo
        if torch.cuda.is_available() and device_obj.type == "cuda":
            torch.cuda.empty_cache()

    common = {
        "prepared": prepared,
        "device_obj": device_obj,
        "n_envs": int(n_envs),
        "n_steps": int(n_steps),
        "seed": int(seed),
        "checkpoint_path": checkpoint_path,
    }
    official = _run_train_case(
        case_name="official_fullslot_fullobjective_mse",
        train_impl="official",
        pack=fullslot_pack,
        values=full_values,
        log_probs=full_log_probs,
        single_eval_pos=0,
        force_one_sequence_subbatch=False,
        patch_value_loss_to_mse=False,
        **common,
    )
    local_mse_one = _run_train_case(
        case_name="local_fullslot_fullobjective_mse_one_subbatch",
        train_impl="local",
        pack=fullslot_pack,
        values=full_values,
        log_probs=full_log_probs,
        single_eval_pos=0,
        force_one_sequence_subbatch=True,
        patch_value_loss_to_mse=True,
        **common,
    )
    local_mse_actual_subbatch = _run_train_case(
        case_name="local_fullslot_fullobjective_mse_actual_subbatch",
        train_impl="local",
        pack=fullslot_pack,
        values=full_values,
        log_probs=full_log_probs,
        single_eval_pos=0,
        force_one_sequence_subbatch=False,
        patch_value_loss_to_mse=True,
        **common,
    )
    local_vendor_full = _run_train_case(
        case_name="local_fullslot_fullobjective_vendor_value",
        train_impl="local",
        pack=fullslot_pack,
        values=full_values,
        log_probs=full_log_probs,
        single_eval_pos=0,
        force_one_sequence_subbatch=False,
        patch_value_loss_to_mse=False,
        **common,
    )
    local_runner_actual = _run_train_case(
        case_name="local_runner_actual_active4_suffix_vendor_value",
        train_impl="local",
        pack=actual_pack,
        values=rollout_values,
        log_probs=rollout_log_probs,
        single_eval_pos=int(single_eval_pos),
        force_one_sequence_subbatch=False,
        patch_value_loss_to_mse=False,
        **common,
    )

    comparisons = [
        _compare_cases("official_vs_local_mse_one_subbatch_control", official, local_mse_one),
        _compare_cases("official_vs_local_mse_actual_subbatch", official, local_mse_actual_subbatch),
        _compare_cases("official_vs_local_vendor_fullslot_fullobjective", official, local_vendor_full),
        _compare_cases("official_vs_local_runner_actual", official, local_runner_actual),
        _compare_cases("local_mse_one_subbatch_vs_actual_subbatch", local_mse_one, local_mse_actual_subbatch),
        _compare_cases("local_vendor_fullslot_fullobjective_vs_runner_actual", local_vendor_full, local_runner_actual),
    ]
    control = comparisons[0]
    hard_pass = (
        official["exception"] is None
        and local_mse_one["exception"] is None
        and bool(control["grad_diff"].get("same_numel", False))
        and float(control["grad_diff"].get("absmax", float("inf"))) <= 5e-5
        and float(control["param_delta_diff"].get("absmax", float("inf"))) <= 5e-5
    )
    report = {
        "audit_entry": "phase2_bipedalwalker_gympack_gradient_parity",
        "hard_pass": bool(hard_pass),
        "scope": {
            "purpose": (
                "Numerically compare the same BipedalWalker gym-pack rollout/update config against the "
                "vendored official RecurrentPPO train loop, while separating PPO math from runner-only "
                "action/objective masks and vendor value-head loss."
            ),
            "gym_env_id": str(gym_env_id),
            "n_envs": int(n_envs),
            "n_steps": int(n_steps),
            "seed": int(seed),
            "single_eval_pos": int(single_eval_pos),
            "device": str(device_obj),
        },
        "checkpoint": checkpoint,
        "pack_meta": meta,
        "pack_summary": pack_summary,
        "old_log_prob_replay_diff": old_log_prob_replay_diff,
        "replay_summaries": {
            "fullslot": full_replay,
            "active4": active_replay,
        },
        "cases": [
            _public_case(official),
            _public_case(local_mse_one),
            _public_case(local_mse_actual_subbatch),
            _public_case(local_vendor_full),
            _public_case(local_runner_actual),
        ],
        "comparisons": comparisons,
        "interpretation": {
            "official_vs_local_mse_one_subbatch_control": (
                "Should be near-zero if the local PPO formula and RWKV sequence replay match official "
                "when action/objective masks and vendor value loss are removed."
            ),
            "official_vs_local_mse_actual_subbatch": (
                "Isolates whether local sequence subbatching is numerically equivalent to official full-batch PPO."
            ),
            "official_vs_local_vendor_fullslot_fullobjective": (
                "Adds the same-config vendor value-head loss back while keeping official action/full objective semantics."
            ),
            "official_vs_local_runner_actual": (
                "The direct same-config runner update: active 4-dim action masks, suffix objective mask, "
                "vendor value-head loss, and actual sequence subbatching."
            ),
        },
    }
    out_path = Path(output_json).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=str, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--checkpoint-path", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-envs", type=int, default=64)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--seed", type=int, default=4040)
    parser.add_argument("--gym-env-id", type=str, default="BipedalWalker-v3")
    parser.add_argument("--single-eval-pos", type=int, default=64)
    parser.add_argument("--frozen-h-json", type=str, default=DEFAULT_FROZEN_H_JSON)
    args = parser.parse_args(argv)
    report = run_probe(
        output_json=args.output_json,
        checkpoint_path=args.checkpoint_path,
        device=args.device,
        n_envs=int(args.n_envs),
        n_steps=int(args.n_steps),
        seed=int(args.seed),
        gym_env_id=args.gym_env_id,
        single_eval_pos=int(args.single_eval_pos),
        frozen_h_json=args.frozen_h_json,
    )
    print(json.dumps({"hard_pass": bool(report["hard_pass"]), "output_json": str(args.output_json)}, sort_keys=True))


if __name__ == "__main__":
    main()
