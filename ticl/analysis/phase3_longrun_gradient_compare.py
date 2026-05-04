import argparse
import copy
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ticl.analysis.critic_free_single_env_audit import _build_audit_env_cfg
from ticl.analysis.phase3_longrun_driver import _phase3_isolated_extra_config
from ticl.analysis.sep_state_reset_gradient_compare import _build_fixed_env_h, _clone_h_repeated, _flatten_policy_grad
from ticl.config_utils import update_config
from ticl.fit_model import _apply_continue_run_resume_safe_defaults, _merge_missing_keys
from ticl.model_builder import load_model
from ticl.model_configs import get_model_default_config
from ticl.priors.environment_prior import EnvironmentPrior
from ticl.sb3_recurrent_ppo import (
    _recover_raw_from_value_space,
    _resolve_actor_advantages,
    build_recurrent_ppo,
)


def _default_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _load_checkpoint_config(checkpoint_path: str) -> dict[str, Any]:
    states = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(states, (list, tuple)) or len(states) < 4:
        raise ValueError(f"Unexpected checkpoint payload shape in {checkpoint_path}")
    config = states[3]
    if not isinstance(config, dict):
        raise TypeError(f"Checkpoint config is not a dict: {type(config)!r}")
    return config


def _resolve_continue_run_config(*, checkpoint_path: str, model_type: str = "rlpfn") -> dict[str, Any]:
    config = copy.deepcopy(_load_checkpoint_config(checkpoint_path))
    defaults = get_model_default_config(model_type, None)
    _merge_missing_keys(config, defaults)
    _apply_continue_run_resume_safe_defaults(config, defaults, model_type)
    return config


def _relevant_training_signature(config: dict[str, Any]) -> dict[str, Any]:
    optimizer = config.get("optimizer", {})
    prior_env = config.get("prior", {}).get("environment", {})
    q_runtime_override = optimizer.get("ppo_runtime_normalized_q_value_weight_override")
    flow_runtime_override = optimizer.get("ppo_runtime_next_state_flow_matching_weight_override")
    effective_q_weight = (
        q_runtime_override
        if q_runtime_override is not None
        else prior_env.get("normalized_q_value_weight")
    )
    effective_flow_weight = (
        flow_runtime_override
        if flow_runtime_override is not None
        else prior_env.get("next_state_flow_matching_weight")
    )
    return {
        "optimizer.rl_objective": optimizer.get("rl_objective"),
        "optimizer.ppo_n_epochs": optimizer.get("ppo_n_epochs"),
        "optimizer.ppo_gamma": optimizer.get("ppo_gamma"),
        "optimizer.ppo_gae_lambda": optimizer.get("ppo_gae_lambda"),
        "optimizer.ppo_clip_range": optimizer.get("ppo_clip_range"),
        "optimizer.ppo_normalize_advantage": optimizer.get("ppo_normalize_advantage"),
        "optimizer.ppo_actor_gae_space": optimizer.get("ppo_actor_gae_space"),
        "optimizer.ppo_actor_baseline_mode": optimizer.get("ppo_actor_baseline_mode"),
        "optimizer.ppo_separate_value_backbone": optimizer.get("ppo_separate_value_backbone"),
        "optimizer.ppo_reset_env_state_at_sep": optimizer.get("ppo_reset_env_state_at_sep"),
        "optimizer.ppo_runtime_normalized_q_value_weight_override": q_runtime_override,
        "optimizer.ppo_runtime_next_state_flow_matching_weight_override": flow_runtime_override,
        "optimizer.ppo_vf_coef": optimizer.get("ppo_vf_coef"),
        "optimizer.ppo_target_kl": optimizer.get("ppo_target_kl"),
        "prior.environment.normalized_q_value_weight": prior_env.get("normalized_q_value_weight"),
        "prior.environment.next_state_flow_matching_weight": prior_env.get("next_state_flow_matching_weight"),
        "prior.environment.next_state_flow_head_type": prior_env.get("next_state_flow_head_type"),
        "effective.normalized_q_value_weight": effective_q_weight,
        "effective.next_state_flow_matching_weight": effective_flow_weight,
    }


def _signature_diff(left: dict[str, Any], right: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for key in sorted(set(left.keys()) | set(right.keys())):
        if left.get(key) != right.get(key):
            out[key] = {"left": left.get(key), "right": right.get(key)}
    return out


def _to_builtin(value: Any) -> Any:
    if torch.is_tensor(value):
        if value.ndim == 0:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _leaf_count(value: Any) -> int:
    if isinstance(value, dict):
        if not value:
            return 1
        return int(sum(_leaf_count(v) for v in value.values()))
    if isinstance(value, list):
        if not value:
            return 1
        return int(sum(_leaf_count(v) for v in value))
    return 1


def _fixed_env_contract_summary(frozen_h: Any) -> dict[str, Any]:
    frozen_h_builtin = _to_builtin(frozen_h)
    payload = json.dumps(frozen_h_builtin, sort_keys=True, ensure_ascii=True)
    top_level_keys = (
        sorted(str(k) for k in frozen_h_builtin.keys()) if isinstance(frozen_h_builtin, dict) else []
    )
    return {
        "frozen_h_fingerprint": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "frozen_h_leaf_count": int(_leaf_count(frozen_h_builtin)),
        "top_level_key_count": int(len(top_level_keys)),
        "top_level_keys_sample": top_level_keys[:16],
        "latent_uniform_keys_sample": [k for k in top_level_keys if k.endswith("_u")][:16],
    }


def _build_algo_for_config(
    *,
    checkpoint_path: str,
    resolved_config: dict[str, Any],
    device_obj: torch.device,
    frozen_h,
    train_env_seed: int,
    rollout_seed: int,
    single_eval_pos: int,
    n_steps: int,
    batch_size: int,
    learning_rate: float,
):
    load_model.cache_clear()
    model, _ = load_model(checkpoint_path, device=device_obj, verbose=False)
    num_features = int(resolved_config["prior"]["num_features"])
    env_cfg = _build_audit_env_cfg(resolved_config["prior"]["environment"])
    prior = EnvironmentPrior(copy.deepcopy(env_cfg))
    prior._sample_batch_hypers = lambda batch_n, _h=frozen_h: _clone_h_repeated(_h, int(batch_n))
    prior._sample_single_eval_pos = lambda n_samples_arg, rng_seed=None: int(single_eval_pos)

    optimizer_cfg = resolved_config["optimizer"]
    algo, callback, vec_env = build_recurrent_ppo(
        model=model,
        env_prior=prior,
        device=str(device_obj),
        num_features=num_features,
        n_envs=1,
        n_steps=int(n_steps),
        learning_rate=float(learning_rate),
        batch_size=int(batch_size),
        n_epochs=int(optimizer_cfg.get("ppo_n_epochs", 4)),
        gamma=float(optimizer_cfg.get("ppo_gamma", 1.0)),
        gae_lambda=float(optimizer_cfg.get("ppo_gae_lambda", 0.95)),
        clip_range=optimizer_cfg.get("ppo_clip_range", 0.2),
        clip_range_vf=optimizer_cfg.get("ppo_clip_range_vf", None),
        normalize_advantage=bool(optimizer_cfg.get("ppo_normalize_advantage", True)),
        actor_gae_space=str(optimizer_cfg.get("ppo_actor_gae_space", "normalized")),
        actor_baseline_mode=str(optimizer_cfg.get("ppo_actor_baseline_mode", "learned")),
        actor_objective_mode="tokenwise",
        separate_value_backbone=bool(optimizer_cfg.get("ppo_separate_value_backbone", False)),
        reset_env_state_at_sep=bool(optimizer_cfg.get("ppo_reset_env_state_at_sep", True)),
        strict_fixed_env_mode=True,
        env_rng_seeds=int(train_env_seed),
        rollout_rng_seeds=int(rollout_seed),
        deterministic_actor_sampling=True,
        deterministic_batch_plan=True,
        restore_validation_policy_state=True,
        ent_coef=float(optimizer_cfg.get("ppo_ent_coef", 0.0)),
        vf_coef=float(optimizer_cfg.get("ppo_vf_coef", 0.5)),
        max_grad_norm=float(optimizer_cfg.get("ppo_max_grad_norm", 0.5)),
        target_kl=optimizer_cfg.get("ppo_target_kl", None),
        verbose=0,
    )
    algo.ep_info_buffer = []
    algo.ep_success_buffer = []
    algo._last_obs = vec_env.reset()
    algo._last_episode_starts = np.ones((vec_env.num_envs,), dtype=bool)
    algo._last_lstm_states = algo.policy._dummy_states(vec_env.num_envs)
    callback.init_callback(algo)
    algo._update_current_progress_remaining(0, int(n_steps))
    return algo, callback, vec_env


def _collect_and_measure_total_grad(algo, callback, vec_env) -> dict[str, Any]:
    t_collect0 = time.perf_counter()
    assert algo.collect_rollouts(vec_env, callback, algo.rollout_buffer, n_rollout_steps=algo.n_steps)
    if algo.device.type == "cuda":
        torch.cuda.synchronize(device=algo.device)
    collect_wall_s = float(time.perf_counter() - t_collect0)
    rollout_data = next(algo.rollout_buffer.get_gpu_flat(algo.batch_size))
    objective_mask = rollout_data.objective_masks > 1e-8
    valid_total = objective_mask.to(
        device=rollout_data.returns.device,
        dtype=rollout_data.returns.dtype,
    ).sum().clamp_min(1.0)
    advantages = _resolve_actor_advantages(rollout_data)

    q_targets_full = None
    flow_xt_full = flow_t_full = flow_dx_full = None
    flow_denom_total = None
    q_weight = float(getattr(algo, "_rwkv_aux_q_weight", 0.0))
    flow_weight = float(getattr(algo, "_rwkv_aux_flow_weight", 0.0))
    if q_weight > 0.0:
        q_targets_full = rollout_data.normalized_q_targets
        if q_targets_full is None:
            raw_returns_full = _recover_raw_from_value_space(
                rollout_data.returns.detach(),
                value_means=rollout_data.rollout_return_means.detach(),
                value_stds=rollout_data.rollout_return_stds.detach(),
            )
            q_targets_full = algo._normalize_flat_sequence_returns_like_reinforce(
                raw_returns_full,
                seq_start_indices=rollout_data.seq_start_indices,
                seq_lengths=rollout_data.seq_lengths,
                valid_mask_flat=objective_mask.detach(),
                eps=1e-6,
            )
    if flow_weight > 0.0:
        env_prior = getattr(algo, "_rwkv_env_prior", None)
        if env_prior is None:
            raise RuntimeError("Flow aux expected bound EnvironmentPrior")
        flow_xt_full, flow_t_full, flow_dx_full = env_prior._sample_condot_flow_matching_path(
            rollout_data.next_states.detach()
        )
        flow_denom_total = rollout_data.next_state_masks.sum().clamp_min(1.0)

    eval_outputs = algo.policy.evaluate_actions_with_hidden_flat(
        rollout_data.observations,
        rollout_data.actions,
        seq_lengths=rollout_data.seq_lengths,
        action_masks=rollout_data.action_masks,
        include_normalized_q_logits=(q_targets_full is not None),
        flow_matching_xt=flow_xt_full,
        flow_matching_t=flow_t_full,
    )
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

    value_errors = algo.policy.value_bardist(
        value_logits.to(dtype=torch.float32),
        rollout_data.returns.to(device=value_logits.device, dtype=torch.float32),
    )
    value_num = (
        value_errors
        * objective_mask.to(device=value_errors.device, dtype=value_errors.dtype)
    ).sum()
    value_loss = value_num / valid_total.to(device=value_num.device, dtype=value_num.dtype)

    if entropy is None:
        entropy_terms = -log_prob
    else:
        entropy_terms = entropy
    entropy_num = (
        entropy_terms
        * objective_mask.to(device=entropy_terms.device, dtype=entropy_terms.dtype)
    ).sum()
    entropy_loss = -(entropy_num / valid_total.to(device=entropy_num.device, dtype=entropy_num.dtype))

    aux_loss = torch.zeros((), device=policy_loss.device, dtype=policy_loss.dtype)
    aux_stats = None
    aux_loss_info = algo._compute_aux_losses_flat(
        hidden=eval_outputs["hidden"],
        actions=eval_outputs["actions"],
        seq_lengths=rollout_data.seq_lengths,
        normalized_q_logits=eval_outputs.get("normalized_q_logits", None),
        next_state_flow=eval_outputs.get("next_state_flow", None),
        normalized_q_targets=q_targets_full,
        objective_mask=objective_mask,
        q_target_denom=valid_total,
        flow_matching_xt=flow_xt_full,
        flow_matching_t=flow_t_full,
        flow_matching_dx=flow_dx_full,
        next_states=rollout_data.next_states,
        next_state_masks=rollout_data.next_state_masks,
        flow_denom=flow_denom_total,
    )
    if aux_loss_info is not None:
        aux_loss, aux_stats = aux_loss_info

    total_loss = policy_loss + algo.ent_coef * entropy_loss + algo.vf_coef * value_loss + aux_loss
    algo.policy.optimizer.zero_grad(set_to_none=True)
    t_backward0 = time.perf_counter()
    total_loss.backward()
    if algo.device.type == "cuda":
        torch.cuda.synchronize(device=algo.device)
    backward_wall_s = float(time.perf_counter() - t_backward0)
    grad_flat = _flatten_policy_grad(algo.policy)

    return {
        "policy_loss": float(policy_loss.detach().cpu().item()),
        "value_loss": float(value_loss.detach().cpu().item()),
        "entropy_loss": float(entropy_loss.detach().cpu().item()),
        "aux_loss": float(aux_loss.detach().cpu().item()),
        "total_loss": float(total_loss.detach().cpu().item()),
        "objective_total": int(objective_mask.sum().detach().cpu().item()),
        "returns_std": float(rollout_data.returns.detach().std().item() if rollout_data.returns.numel() > 1 else 0.0),
        "advantages_std": float(advantages.detach().std().item() if advantages.numel() > 1 else 0.0),
        "flow_weight": float(flow_weight),
        "q_weight": float(q_weight),
        "sep_state_reset_count": int(getattr(algo, "_rwkv_last_sep_state_reset_count", 0) or 0),
        "grad_norm": float(torch.linalg.vector_norm(grad_flat).detach().cpu().item()),
        "collect_wall_s": collect_wall_s,
        "backward_wall_s": backward_wall_s,
        "flow_matching_loss": float(aux_stats["next_state_flow_matching_loss"].detach().cpu().item()) if aux_stats is not None and "next_state_flow_matching_loss" in aux_stats else None,
        "normalized_q_value_loss": float(aux_stats["normalized_q_value_loss"].detach().cpu().item()) if aux_stats is not None and "normalized_q_value_loss" in aux_stats else None,
        "grad_vector": grad_flat.detach().cpu(),
    }


def run_phase3_longrun_gradient_compare(
    *,
    checkpoint_path: str,
    device: str | None = None,
    frozen_h_seed: int = 12345,
    train_env_seed: int = 2020,
    rollout_seed: int = 2020,
    single_eval_pos: int = 1946,
    n_steps: int = 2048,
    batch_size: int = 2048,
    learning_rate: float = 2e-4,
) -> dict[str, Any]:
    checkpoint_path = str(Path(checkpoint_path).expanduser().resolve())
    device_obj = torch.device(str(device or _default_device()))
    resume_config = _resolve_continue_run_config(checkpoint_path=checkpoint_path)
    isolated_config = copy.deepcopy(resume_config)
    update_config(isolated_config, _phase3_isolated_extra_config())

    resume_sig = _relevant_training_signature(resume_config)
    isolated_sig = _relevant_training_signature(isolated_config)
    sig_diff = _signature_diff(resume_sig, isolated_sig)

    env_cfg = _build_audit_env_cfg(resume_config["prior"]["environment"])
    frozen_h = _build_fixed_env_h(prior_cfg=env_cfg, frozen_h_seed=int(frozen_h_seed))

    reports = {}
    grad_vectors = {}
    for mode_name, resolved_config in (
        ("resume_checkpoint_semantics", resume_config),
        ("phase3_isolated_semantics", isolated_config),
    ):
        random.seed(int(rollout_seed))
        np.random.seed(int(rollout_seed))
        torch.manual_seed(int(rollout_seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(rollout_seed))
        algo, callback, vec_env = _build_algo_for_config(
            checkpoint_path=checkpoint_path,
            resolved_config=resolved_config,
            device_obj=device_obj,
            frozen_h=frozen_h,
            train_env_seed=int(train_env_seed),
            rollout_seed=int(rollout_seed),
            single_eval_pos=int(single_eval_pos),
            n_steps=int(n_steps),
            batch_size=int(batch_size),
            learning_rate=float(learning_rate),
        )
        try:
            report = _collect_and_measure_total_grad(algo, callback, vec_env)
            grad_vectors[mode_name] = report.pop("grad_vector")
            reports[mode_name] = report
        finally:
            vec_env.close()
            del algo, callback, vec_env
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    grad_a = grad_vectors["resume_checkpoint_semantics"]
    grad_b = grad_vectors["phase3_isolated_semantics"]
    cosine = torch.nn.functional.cosine_similarity(
        grad_a.unsqueeze(0),
        grad_b.unsqueeze(0),
        dim=1,
        eps=1e-8,
    ).item()
    grad_delta = grad_b - grad_a
    return {
        "audit_entry": "phase3_longrun_gradient_compare",
        "checkpoint_path": checkpoint_path,
        "device": str(device_obj),
        "frozen_h_seed": int(frozen_h_seed),
        "train_env_seed": int(train_env_seed),
        "rollout_seed": int(rollout_seed),
        "single_eval_pos": int(single_eval_pos),
        "n_steps": int(n_steps),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "fixed_env_contract": _fixed_env_contract_summary(frozen_h),
        "training_signature": {
            "resume_checkpoint_semantics": resume_sig,
            "phase3_isolated_semantics": isolated_sig,
            "diff": sig_diff,
            "exact_match": len(sig_diff) == 0,
        },
        "modes": reports,
        "grad_compare": {
            "cosine_similarity": float(cosine),
            "l2_delta_norm": float(torch.linalg.vector_norm(grad_delta).item()),
            "resume_grad_norm": float(torch.linalg.vector_norm(grad_a).item()),
            "phase3_isolated_grad_norm": float(torch.linalg.vector_norm(grad_b).item()),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Gradient-level compare between continue-run checkpoint semantics and phase3-isolated longrun semantics.")
    parser.add_argument("checkpoint_path", type=str)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--frozen-h-seed", type=int, default=12345)
    parser.add_argument("--train-env-seed", type=int, default=2020)
    parser.add_argument("--rollout-seed", type=int, default=2020)
    parser.add_argument("--single-eval-pos", type=int, default=1946)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    report = run_phase3_longrun_gradient_compare(
        checkpoint_path=args.checkpoint_path,
        device=args.device,
        frozen_h_seed=args.frozen_h_seed,
        train_env_seed=args.train_env_seed,
        rollout_seed=args.rollout_seed,
        single_eval_pos=args.single_eval_pos,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    output_path = args.output_json
    if output_path is None:
        output_path = "/home/chen/RLPFN/artifacts/phase3_longrun_gradient_compare.json"
    out = Path(output_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
