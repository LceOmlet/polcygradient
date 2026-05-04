import argparse
import copy
import json
import random
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ticl.analysis.critic_free_single_env_audit import (
    _apply_boundary_contract_mode_flags,
    _apply_sep_state_reset_flag,
    _build_audit_env_cfg,
    _make_deterministic_batch_plan,
)
from ticl.analysis.fixed_env_h import build_fixed_env_h
from ticl.analysis.phase2_guardrail import (
    TRUSTED_PHASE2_LAUNCH_SUMMARY,
    assert_phase2_launch_chain_green,
)
from ticl.analysis.phase3_multi_env_optimization_audit import (
    _bind_vec_env_to_fixed_suite,
    _clone_suite_h_list,
    _resolve_train_profile,
)
from ticl.model_builder import load_model
from ticl.priors.environment_prior import EnvironmentPrior
from ticl.sb3_recurrent_ppo import _resolve_actor_advantages, build_recurrent_ppo


DEFAULT_OUTPUT_JSON = (
    "/home/chen/RLPFN/artifacts/phase3_phase2_milestone_contract_gradient_compare.json"
)
DEFAULT_CHECKPOINT_PATH = "/home/chen/RLPFN/rwkv/models_diff/rlpfn_03_30_2026_22_37_53_epoch_13.cpkt"


def _default_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _clone_h_repeated(h, count: int) -> list[dict[str, Any]]:
    return [copy.deepcopy(h) for _ in range(int(count))]


def _seed_all(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _build_fixed_env_h(*, prior_cfg: dict[str, Any], frozen_h_seed: int):
    return build_fixed_env_h(prior_cfg=prior_cfg, frozen_h_seed=int(frozen_h_seed))


def _sync_last_obs(algo, callback, vec_env, *, train_env_seed: int) -> None:
    vec_env._sample_seed_list = lambda: [int(train_env_seed)]
    algo.ep_info_buffer = []
    algo.ep_success_buffer = []
    algo._last_obs = vec_env.reset()
    algo._last_episode_starts = np.ones((vec_env.num_envs,), dtype=bool)
    algo._last_lstm_states = algo.policy._dummy_states(vec_env.num_envs)
    callback.init_callback(algo)
    algo._update_current_progress_remaining(0, int(algo.n_steps))


def _flatten_policy_grad(policy: torch.nn.Module) -> torch.Tensor:
    flat_parts = []
    for param in policy.parameters():
        if not param.requires_grad:
            continue
        grad = param.grad
        if grad is None:
            flat_parts.append(torch.zeros(param.numel(), device=param.device, dtype=param.dtype))
        else:
            flat_parts.append(grad.detach().reshape(-1))
    if not flat_parts:
        return torch.zeros((0,), dtype=torch.float32)
    return torch.cat(flat_parts).to(dtype=torch.float32)


def _max_abs_diff_tensor(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().cpu().to(dtype=torch.float32)
    b = b.detach().cpu().to(dtype=torch.float32)
    if tuple(a.shape) != tuple(b.shape):
        raise ValueError(f"Tensor shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
    if a.numel() == 0:
        return 0.0
    return float((a - b).abs().max().item())


def _param_diff_summary(policy_a, policy_b) -> dict[str, Any]:
    max_diff = 0.0
    differing = []
    state_a = policy_a.state_dict()
    state_b = policy_b.state_dict()
    if set(state_a.keys()) != set(state_b.keys()):
        raise ValueError("Policy state dict keys differ between phase2 and phase3 contract builds")
    for key in sorted(state_a.keys()):
        diff = _max_abs_diff_tensor(state_a[key], state_b[key])
        max_diff = max(max_diff, diff)
        if diff > 0.0:
            differing.append({"name": str(key), "max_abs_diff": float(diff)})
    return {
        "policy_param_max_abs_diff": float(max_diff),
        "num_param_tensors_with_diff": int(len(differing)),
        "differing_tensors": differing,
    }


def _signature_diff(a: dict[str, Any], b: dict[str, Any]) -> dict[str, dict[str, Any]]:
    diff: dict[str, dict[str, Any]] = {}
    all_keys = sorted(set(a.keys()) | set(b.keys()))
    for key in all_keys:
        if a.get(key, None) != b.get(key, None):
            diff[str(key)] = {
                "phase2": a.get(key, None),
                "phase3": b.get(key, None),
            }
    return diff


def _install_batch1_non_native_compare_shim(algo) -> dict[str, Any]:
    """Normalize batch=1 official-eval rollout jitter inside this compare harness only.

    Under the current trusted Phase 2 contract we intentionally keep
    `strict_native_rollout=false`. In that mode, batch-1 no-grad rollout steps route
    through the vendor official-eval ScriptModule path. We observed a reproducible
    cross-build drift there even when:
    - checkpoint weights are exact,
    - env/token inputs are exact,
    - fixed-env contract is exact.

    The compare harness cares about semantic parity between the Phase 2-side and
    Phase 3-side contracts, not about instance-local vendor kernel jitter. For this
    one narrow case we therefore replace the batch=1 official-eval branch with the
    same deterministic native-step fallback on both sides. This does not touch the
    training/runtime codepath outside this audit script.
    """

    policy = algo.policy
    core = policy.rlpfn_model.rwkv_core
    original_forward_step = core.forward_step

    def _wrapped_forward_step(self, token: torch.Tensor, state=None):
        if (
            token.ndim == 2
            and int(token.shape[0]) == 1
            and token.is_cuda
            and (not self.training)
            and (not torch.is_grad_enabled())
            and (not bool(getattr(self, "force_native_eval_forward_step", False)))
        ):
            prev = bool(getattr(self, "force_native_eval_forward_step", False))
            self.force_native_eval_forward_step = True
            try:
                return original_forward_step(token, state)
            finally:
                self.force_native_eval_forward_step = prev
        return original_forward_step(token, state)

    core.forward_step = types.MethodType(_wrapped_forward_step, core)
    return {
        "batch1_non_native_compare_shim_applied": True,
        "shim_scope": "phase3_phase2_exact_compare_only",
        "shim_reason": (
            "Current trusted Phase 2 contract keeps strict_native_rollout=false, but "
            "vendor batch=1 official-eval ScriptModule shows cross-build jitter. "
            "Compare harness replaces only that branch with a deterministic shared "
            "native-step fallback on both sides."
        ),
    }


def _collect_and_measure_main_loss(algo, callback, vec_env, *, train_env_seed: int) -> dict[str, Any]:
    _sync_last_obs(algo, callback, vec_env, train_env_seed=int(train_env_seed))
    assert algo.collect_rollouts(vec_env, callback, algo.rollout_buffer, n_rollout_steps=algo.n_steps)
    if algo.device.type == "cuda":
        torch.cuda.synchronize(device=algo.device)

    rollout_data = next(algo.rollout_buffer.get_gpu_flat(algo.batch_size))
    objective_mask = rollout_data.objective_masks > 1e-8
    valid_total = objective_mask.to(
        device=rollout_data.returns.device,
        dtype=rollout_data.returns.dtype,
    ).sum().clamp_min(1.0)
    advantages = _resolve_actor_advantages(rollout_data)
    eval_outputs = algo.policy.evaluate_actions_with_hidden_flat(
        rollout_data.observations,
        rollout_data.actions,
        seq_lengths=rollout_data.seq_lengths,
        action_masks=rollout_data.action_masks,
    )
    value_bardist = algo.policy.get_value_bardist()
    values = eval_outputs["values"].flatten()
    value_logits = eval_outputs["value_logits"]
    log_prob = eval_outputs["log_prob"]
    entropy = eval_outputs["entropy"]
    clip_range = algo.clip_range(algo._current_progress_remaining)
    ratio = torch.exp(log_prob - rollout_data.old_log_prob)
    policy_loss_1 = advantages * ratio
    policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
    clipped_objective = torch.min(policy_loss_1, policy_loss_2)
    policy_num = (
        clipped_objective
        * objective_mask.to(device=clipped_objective.device, dtype=clipped_objective.dtype)
    ).sum()
    policy_loss = -(policy_num / valid_total.to(device=policy_num.device, dtype=policy_num.dtype))

    value_target_bucket_idx = getattr(rollout_data, "value_target_bucket_idx", None)
    if value_target_bucket_idx is not None:
        value_errors = value_bardist.nll_from_bucket_idx(
            value_logits.to(dtype=torch.float32),
            value_target_bucket_idx,
        )
    else:
        value_errors = value_bardist(
            value_logits.to(dtype=torch.float32),
            rollout_data.returns.to(device=value_logits.device, dtype=torch.float32),
        )
    value_num = (
        value_errors
        * objective_mask.to(device=value_errors.device, dtype=value_errors.dtype)
    ).sum()
    value_loss = value_num / valid_total.to(device=value_num.device, dtype=value_num.dtype)

    entropy_terms = -log_prob if entropy is None else entropy
    entropy_num = (
        entropy_terms
        * objective_mask.to(device=entropy_terms.device, dtype=entropy_terms.dtype)
    ).sum()
    entropy_loss = -(entropy_num / valid_total.to(device=entropy_num.device, dtype=entropy_num.dtype))

    total_loss = policy_loss + algo.ent_coef * entropy_loss + algo.vf_coef * value_loss

    algo.policy.optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    grad_vector = _flatten_policy_grad(algo.policy).detach().cpu()

    return {
        "policy_loss": float(policy_loss.detach().cpu().item()),
        "value_loss": float(value_loss.detach().cpu().item()),
        "entropy_loss": float(entropy_loss.detach().cpu().item()),
        "total_loss": float(total_loss.detach().cpu().item()),
        "objective_total": int(objective_mask.sum().detach().cpu().item()),
        "approx_kl": float(
            torch.mean((torch.exp(log_prob - rollout_data.old_log_prob) - 1) - (log_prob - rollout_data.old_log_prob))
            .detach()
            .cpu()
            .item()
        ),
        "clip_fraction": float(
            torch.mean((torch.abs(ratio - 1.0) > clip_range).to(dtype=ratio.dtype)).detach().cpu().item()
        ),
        "grad_norm": float(torch.linalg.vector_norm(grad_vector).item()),
        "aux_q_weight": float(getattr(algo, "_rwkv_aux_q_weight", 0.0)),
        "aux_flow_weight": float(getattr(algo, "_rwkv_aux_flow_weight", 0.0)),
        "rollout_tensors": {
            "actions": rollout_data.actions.detach().cpu(),
            "returns": rollout_data.returns.detach().cpu(),
            "old_values": rollout_data.old_values.detach().cpu(),
            "advantages": advantages.detach().cpu(),
            "old_log_prob": rollout_data.old_log_prob.detach().cpu(),
            "objective_masks": rollout_data.objective_masks.detach().cpu(),
            "episode_starts": rollout_data.episode_starts.detach().cpu(),
            "observations": rollout_data.observations.detach().cpu(),
        },
        "grad_vector": grad_vector,
    }


def _build_phase2_milestone_signature(*, config: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    optimizer_cfg = dict(config.get("optimizer", {}))
    return {
        "n_envs": 1,
        "n_steps": int(args["n_steps"]),
        "learning_rate": float(args["learning_rate"]),
        "batch_size": int(args["batch_size"]),
        "n_epochs": int(args["n_epochs"]),
        "gamma": float(optimizer_cfg.get("ppo_gamma", 1.0)),
        "gae_lambda": float(optimizer_cfg.get("ppo_gae_lambda", 0.95)),
        "clip_range": optimizer_cfg.get("ppo_clip_range", 0.2),
        "normalize_advantage": False,
        "actor_gae_space": "normalized",
        "actor_baseline_mode": "learned",
        "actor_objective_mode": "tokenwise",
        "reset_env_state_at_sep": True,
        "separate_value_backbone": False,
        "strict_fixed_env_mode": True,
        "env_rng_seeds": [int(args["train_env_seed"])],
        "rollout_rng_seeds": [int(args["train_rollout_seed"])],
        "deterministic_actor_sampling": True,
        "deterministic_batch_plan": True,
        "strict_native_rollout": bool(args.get("strict_native_rollout", False)),
        "restore_validation_policy_state": False,
        "actor_objective_runtime_current_suite_name": None,
        "runtime_normalized_q_value_weight_override": None,
        "runtime_next_state_flow_matching_weight_override": None,
        "ent_coef": float(optimizer_cfg.get("ppo_ent_coef", 0.0)),
        "vf_coef": float(optimizer_cfg.get("ppo_vf_coef", 0.5)),
        "max_grad_norm": float(optimizer_cfg.get("ppo_max_grad_norm", 0.5)),
        "target_kl": float(args["target_kl"]),
    }


def _build_phase3_contract_signature(*, config: dict[str, Any], profile_cfg: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    optimizer_cfg = dict(config.get("optimizer", {}))
    return {
        "n_envs": 1,
        "n_steps": int(args["n_steps"]),
        "learning_rate": float(profile_cfg["learning_rate"]),
        "batch_size": int(profile_cfg["batch_size"]),
        "n_epochs": int(profile_cfg["n_epochs"]),
        "gamma": float(optimizer_cfg.get("ppo_gamma", 1.0)),
        "gae_lambda": float(optimizer_cfg.get("ppo_gae_lambda", 0.95)),
        "clip_range": optimizer_cfg.get("ppo_clip_range", 0.2),
        "normalize_advantage": bool(profile_cfg["normalize_advantage"]),
        "actor_gae_space": str(profile_cfg["actor_gae_space"]),
        "actor_baseline_mode": str(profile_cfg["actor_baseline_mode"]),
        "actor_objective_mode": str(profile_cfg["actor_objective_mode"]),
        "reset_env_state_at_sep": bool(profile_cfg["reset_env_state_at_sep"]),
        "separate_value_backbone": bool(profile_cfg["separate_value_backbone"]),
        "strict_fixed_env_mode": True,
        "env_rng_seeds": [int(args["train_env_seed"])],
        "rollout_rng_seeds": [int(args["train_rollout_seed"])],
        "deterministic_actor_sampling": True,
        "deterministic_batch_plan": True,
        "strict_native_rollout": bool(args.get("strict_native_rollout", False)),
        "restore_validation_policy_state": bool(profile_cfg["restore_validation_policy_state"]),
        "actor_objective_runtime_current_suite_name": None,
        "runtime_normalized_q_value_weight_override": profile_cfg["runtime_normalized_q_value_weight_override"],
        "runtime_next_state_flow_matching_weight_override": profile_cfg["runtime_next_state_flow_matching_weight_override"],
        "ent_coef": float(optimizer_cfg.get("ppo_ent_coef", 0.0)),
        "vf_coef": float(profile_cfg["vf_coef"]),
        "max_grad_norm": float(optimizer_cfg.get("ppo_max_grad_norm", 0.5)),
        "target_kl": float(profile_cfg["target_kl"]),
    }


def _build_phase2_algo(
    *,
    checkpoint_path: str,
    device_obj: torch.device,
    frozen_h,
    args: dict[str, Any],
):
    _seed_all(int(args["build_seed"]))
    load_model.cache_clear()
    model, config = load_model(checkpoint_path, device=device_obj, verbose=False)
    env_cfg = _build_audit_env_cfg(config["prior"]["environment"])
    prior = EnvironmentPrior(copy.deepcopy(env_cfg))
    prior._sample_batch_hypers = lambda batch_n, _h=frozen_h: _clone_h_repeated(_h, int(batch_n))
    prior._sample_single_eval_pos = lambda n_samples_arg, rng_seed=None: int(args["single_eval_pos"])
    _apply_boundary_contract_mode_flags(prior, "normal")
    _apply_sep_state_reset_flag(prior, True)
    signature = _build_phase2_milestone_signature(config=config, args=args)
    algo, callback, vec_env = build_recurrent_ppo(
        model=model,
        env_prior=prior,
        device=str(device_obj),
        num_features=int(config["prior"]["num_features"]),
        n_envs=int(signature["n_envs"]),
        n_steps=int(signature["n_steps"]),
        learning_rate=float(signature["learning_rate"]),
        batch_size=int(signature["batch_size"]),
        n_epochs=int(signature["n_epochs"]),
        gamma=float(signature["gamma"]),
        gae_lambda=float(signature["gae_lambda"]),
        clip_range=signature["clip_range"],
        clip_range_vf=None,
        normalize_advantage=bool(signature["normalize_advantage"]),
        ent_coef=float(signature["ent_coef"]),
        vf_coef=float(signature["vf_coef"]),
        max_grad_norm=float(signature["max_grad_norm"]),
        target_kl=float(signature["target_kl"]),
        reset_env_state_at_sep=bool(signature["reset_env_state_at_sep"]),
        separate_value_backbone=bool(signature["separate_value_backbone"]),
        strict_fixed_env_mode=bool(signature["strict_fixed_env_mode"]),
        env_rng_seeds=list(signature["env_rng_seeds"]),
        rollout_rng_seeds=list(signature["rollout_rng_seeds"]),
        deterministic_actor_sampling=bool(signature["deterministic_actor_sampling"]),
        deterministic_batch_plan=bool(signature["deterministic_batch_plan"]),
        strict_native_rollout=bool(signature["strict_native_rollout"]),
        restore_validation_policy_state=bool(signature["restore_validation_policy_state"]),
        actor_gae_space=str(signature["actor_gae_space"]),
        actor_baseline_mode=str(signature["actor_baseline_mode"]),
        actor_objective_mode=str(signature["actor_objective_mode"]),
        actor_objective_runtime_current_suite_name=signature["actor_objective_runtime_current_suite_name"],
        runtime_normalized_q_value_weight_override=signature["runtime_normalized_q_value_weight_override"],
        runtime_next_state_flow_matching_weight_override=signature["runtime_next_state_flow_matching_weight_override"],
        verbose=0,
    )
    vec_env._sample_seed_list = lambda: [int(args["train_env_seed"])]
    _make_deterministic_batch_plan(algo.rollout_buffer)
    return {
        "algo": algo,
        "callback": callback,
        "vec_env": vec_env,
        "signature": signature,
        "config": config,
    }


def _build_phase3_contract_algo(
    *,
    checkpoint_path: str,
    device_obj: torch.device,
    frozen_h,
    args: dict[str, Any],
):
    _seed_all(int(args["build_seed"]))
    load_model.cache_clear()
    model, config = load_model(checkpoint_path, device=device_obj, verbose=False)
    env_cfg = _build_audit_env_cfg(config["prior"]["environment"])
    prior = EnvironmentPrior(copy.deepcopy(env_cfg))
    prior._sample_batch_hypers = lambda batch_n, _h=frozen_h: _clone_h_repeated(_h, int(batch_n))
    prior._sample_single_eval_pos = lambda n_samples_arg, rng_seed=None: int(args["single_eval_pos"])
    profile_cfg = _resolve_train_profile(
        optimizer_cfg=dict(config.get("optimizer", {})),
        train_profile="phase2_shared_backbone_contract",
        batch_size=int(args["batch_size"]),
        n_epochs=int(args["n_epochs"]),
        learning_rate=float(args["learning_rate"]),
        target_kl=float(args["target_kl"]),
    )
    signature = _build_phase3_contract_signature(config=config, profile_cfg=profile_cfg, args=args)
    algo, callback, vec_env = build_recurrent_ppo(
        model=model,
        env_prior=prior,
        device=str(device_obj),
        num_features=int(config["prior"]["num_features"]),
        n_envs=int(signature["n_envs"]),
        n_steps=int(signature["n_steps"]),
        learning_rate=float(signature["learning_rate"]),
        batch_size=int(signature["batch_size"]),
        n_epochs=int(signature["n_epochs"]),
        gamma=float(signature["gamma"]),
        gae_lambda=float(signature["gae_lambda"]),
        clip_range=signature["clip_range"],
        clip_range_vf=None,
        normalize_advantage=bool(signature["normalize_advantage"]),
        ent_coef=float(signature["ent_coef"]),
        vf_coef=float(signature["vf_coef"]),
        max_grad_norm=float(signature["max_grad_norm"]),
        target_kl=float(signature["target_kl"]),
        reset_env_state_at_sep=bool(signature["reset_env_state_at_sep"]),
        separate_value_backbone=bool(signature["separate_value_backbone"]),
        strict_fixed_env_mode=bool(signature["strict_fixed_env_mode"]),
        env_rng_seeds=list(signature["env_rng_seeds"]),
        rollout_rng_seeds=list(signature["rollout_rng_seeds"]),
        deterministic_actor_sampling=bool(signature["deterministic_actor_sampling"]),
        deterministic_batch_plan=bool(signature["deterministic_batch_plan"]),
        strict_native_rollout=bool(signature["strict_native_rollout"]),
        restore_validation_policy_state=bool(signature["restore_validation_policy_state"]),
        actor_gae_space=str(signature["actor_gae_space"]),
        actor_baseline_mode=str(signature["actor_baseline_mode"]),
        actor_objective_mode=str(signature["actor_objective_mode"]),
        actor_objective_runtime_current_suite_name=signature["actor_objective_runtime_current_suite_name"],
        runtime_normalized_q_value_weight_override=signature["runtime_normalized_q_value_weight_override"],
        runtime_next_state_flow_matching_weight_override=signature["runtime_next_state_flow_matching_weight_override"],
        verbose=0,
    )
    suite = {
        "batch_size": 1,
        "h_list": [copy.deepcopy(frozen_h)],
        "env_seeds": [int(args["train_env_seed"])],
        "rollout_seeds": [int(args["train_rollout_seed"])],
    }
    _bind_vec_env_to_fixed_suite(vec_env, suite=suite, single_eval_pos=int(args["single_eval_pos"]))
    _make_deterministic_batch_plan(algo.rollout_buffer)
    return {
        "algo": algo,
        "callback": callback,
        "vec_env": vec_env,
        "signature": signature,
        "config": config,
        "profile_cfg": profile_cfg,
    }


def run_phase3_phase2_milestone_contract_gradient_compare(
    *,
    checkpoint_path: str,
    device: str | None = None,
    frozen_h_seed: int = 12345,
    train_env_seed: int = 2020,
    train_rollout_seed: int = 4040,
    single_eval_pos: int = 64,
    n_steps: int = 256,
    batch_size: int = 256,
    n_epochs: int = 1,
    learning_rate: float = 2e-4,
    target_kl: float = 0.03,
    strict_native_rollout: bool = False,
    phase2_summary_path: str = TRUSTED_PHASE2_LAUNCH_SUMMARY,
) -> dict[str, Any]:
    phase2_summary = assert_phase2_launch_chain_green(phase2_summary_path)
    checkpoint_path = str(Path(checkpoint_path).expanduser().resolve())
    device_obj = torch.device(str(device or _default_device()))
    args = {
        "build_seed": 4040,
        "frozen_h_seed": int(frozen_h_seed),
        "train_env_seed": int(train_env_seed),
        "train_rollout_seed": int(train_rollout_seed),
        "single_eval_pos": int(single_eval_pos),
        "n_steps": int(n_steps),
        "batch_size": int(batch_size),
        "n_epochs": int(n_epochs),
        "learning_rate": float(learning_rate),
        "target_kl": float(target_kl),
        "strict_native_rollout": bool(strict_native_rollout),
    }

    load_model.cache_clear()
    _, config = load_model(checkpoint_path, device=device_obj, verbose=False)
    env_cfg = _build_audit_env_cfg(config["prior"]["environment"])
    frozen_h = _build_fixed_env_h(prior_cfg=env_cfg, frozen_h_seed=int(frozen_h_seed))

    phase2_bundle = _build_phase2_algo(
        checkpoint_path=checkpoint_path,
        device_obj=device_obj,
        frozen_h=frozen_h,
        args=args,
    )
    phase3_bundle = _build_phase3_contract_algo(
        checkpoint_path=checkpoint_path,
        device_obj=device_obj,
        frozen_h=frozen_h,
        args=args,
    )
    compare_runtime_shims: dict[str, Any] = {
        "batch1_non_native_compare_shim_applied": False,
    }
    if not bool(strict_native_rollout):
        compare_runtime_shims = _install_batch1_non_native_compare_shim(phase2_bundle["algo"])
        _install_batch1_non_native_compare_shim(phase3_bundle["algo"])

    try:
        initial_param_diff = _param_diff_summary(
            phase2_bundle["algo"].policy,
            phase3_bundle["algo"].policy,
        )
        phase2_measure = _collect_and_measure_main_loss(
            phase2_bundle["algo"],
            phase2_bundle["callback"],
            phase2_bundle["vec_env"],
            train_env_seed=int(train_env_seed),
        )
        phase3_measure = _collect_and_measure_main_loss(
            phase3_bundle["algo"],
            phase3_bundle["callback"],
            phase3_bundle["vec_env"],
            train_env_seed=int(train_env_seed),
        )
    finally:
        phase2_bundle["vec_env"].close()
        phase3_bundle["vec_env"].close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    grad_a = phase2_measure.pop("grad_vector")
    grad_b = phase3_measure.pop("grad_vector")
    rollout_a = phase2_measure.pop("rollout_tensors")
    rollout_b = phase3_measure.pop("rollout_tensors")

    rollout_diff = {
        key: _max_abs_diff_tensor(rollout_a[key], rollout_b[key])
        for key in sorted(rollout_a.keys())
    }
    scalar_diff = {
        key: float(abs(float(phase2_measure[key]) - float(phase3_measure[key])))
        for key in sorted(phase2_measure.keys())
    }
    grad_compare = {
        "cosine_similarity": float(
            torch.nn.functional.cosine_similarity(
                grad_a.unsqueeze(0),
                grad_b.unsqueeze(0),
                dim=1,
                eps=1e-8,
            ).item()
        ),
        "l2_delta_norm": float(torch.linalg.vector_norm(grad_b - grad_a).item()),
        "phase2_grad_norm": float(torch.linalg.vector_norm(grad_a).item()),
        "phase3_grad_norm": float(torch.linalg.vector_norm(grad_b).item()),
        "max_abs_grad_delta": _max_abs_diff_tensor(grad_a, grad_b),
    }
    signature_diff = _signature_diff(phase2_bundle["signature"], phase3_bundle["signature"])

    report = {
        "audit_entry": "phase3_phase2_milestone_contract_gradient_compare",
        "checkpoint_path": checkpoint_path,
        "device": str(device_obj),
        "phase2_summary_path": str(Path(phase2_summary_path).expanduser().resolve()),
        "phase2_guardrail_passed": bool(phase2_summary["validation"]["all_checks_pass"]),
        "contract_args": dict(args),
        "phase2_signature": phase2_bundle["signature"],
        "phase3_signature": phase3_bundle["signature"],
        "compare_runtime_shims": compare_runtime_shims,
        "signature_diff": signature_diff,
        "initial_param_diff": initial_param_diff,
        "phase2_measure": phase2_measure,
        "phase3_measure": phase3_measure,
        "rollout_tensor_max_abs_diff": rollout_diff,
        "scalar_measure_abs_diff": scalar_diff,
        "grad_compare": grad_compare,
        "conclusions": {
            "signature_exact_match": len(signature_diff) == 0,
            "initial_policy_state_exact_match": float(initial_param_diff["policy_param_max_abs_diff"]) == 0.0,
            "rollout_tensors_exact_match": max(rollout_diff.values(), default=0.0) <= 0.0,
            "scalar_measures_exact_match": max(scalar_diff.values(), default=0.0) <= 0.0,
            "gradient_exact_match": float(grad_compare["max_abs_grad_delta"]) <= 0.0,
        },
        "recommendation": {
            "next_focus": "many_env_target_side_ab_only_after_contract_match",
            "reason": (
                "Only after this single-env contract compare is exact can the many-env baseline vs narrow-adjustment A/B "
                "be interpreted as isolating the two intended differences."
            ),
        },
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Gradient-level compare between the Phase 2 milestone single-env path and the Phase 3 phase2_shared_backbone_contract path."
    )
    parser.add_argument("checkpoint_path", type=str, nargs="?", default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--frozen-h-seed", type=int, default=12345)
    parser.add_argument("--train-env-seed", type=int, default=2020)
    parser.add_argument("--train-rollout-seed", type=int, default=4040)
    parser.add_argument("--single-eval-pos", type=int, default=64)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument(
        "--strict-native-rollout",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Toggle strict-native rollout in the trusted milestone compare. "
            "The current trusted Phase 2 launch anchor keeps this disabled."
        ),
    )
    parser.add_argument("--phase2-summary-path", type=str, default=TRUSTED_PHASE2_LAUNCH_SUMMARY)
    parser.add_argument("--output-json", type=str, default=DEFAULT_OUTPUT_JSON)
    args = parser.parse_args()

    report = run_phase3_phase2_milestone_contract_gradient_compare(
        checkpoint_path=args.checkpoint_path,
        device=args.device,
        frozen_h_seed=args.frozen_h_seed,
        train_env_seed=args.train_env_seed,
        train_rollout_seed=args.train_rollout_seed,
        single_eval_pos=args.single_eval_pos,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        learning_rate=args.learning_rate,
        target_kl=args.target_kl,
        strict_native_rollout=bool(args.strict_native_rollout),
        phase2_summary_path=args.phase2_summary_path,
    )
    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
