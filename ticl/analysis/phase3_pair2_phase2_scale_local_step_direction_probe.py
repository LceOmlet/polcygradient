import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ticl.analysis.critic_free_single_env_audit import (
    _build_audit_env_cfg,
    _make_deterministic_batch_plan,
)
from ticl.analysis.phase2_guardrail import DEFAULT_PHASE2_SUMMARY, assert_phase2_green
from ticl.analysis.phase3_multi_env_optimization_audit import (
    _bind_vec_env_to_fixed_suite,
    _clone_suite_h_list,
    _default_device,
    _load_required_suite,
    _resolve_train_profile,
)
from ticl.analysis.phase3_pair2_train_update_ab_compare_pack import (
    PAIR2_CHECKPOINT_PATH,
    PAIR2_TRAIN_SUITE_PATH,
)
from ticl.model_builder import load_model
from ticl.priors.environment_prior import EnvironmentPrior
from ticl.sb3_recurrent_ppo import (
    _normalize_advantages_with_mask,
    _resolve_actor_advantages,
    MaskedRecurrentFlatBatchSamples,
    build_recurrent_ppo,
)


DEFAULT_OUTPUT_JSON = (
    "/home/chen/RLPFN/artifacts/phase3_pair2_phase2_scale_local_step_direction_probe.json"
)

PHASE2_EQ_N_STEPS = 256
PHASE2_EQ_BATCH_SIZE = 256
PHASE2_EQ_OUTER_EPOCHS = 2
PHASE2_EQ_SINGLE_EVAL_POS = 64
TARGET_OUTER_BATCH_IDX = 13


def _init_algo_runtime(algo, callback, vec_env, *, total_timesteps: int) -> None:
    algo.ep_info_buffer = []
    algo.ep_success_buffer = []
    algo._last_obs = vec_env.reset()
    algo._last_episode_starts = np.ones((vec_env.num_envs,), dtype=bool)
    algo._last_lstm_states = algo.policy._dummy_states(vec_env.num_envs)
    callback.init_callback(algo)
    algo._update_current_progress_remaining(0, int(total_timesteps))


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


def _flatten_trainable_params(policy: torch.nn.Module) -> torch.Tensor:
    flat_parts = []
    for param in policy.parameters():
        if not param.requires_grad:
            continue
        flat_parts.append(param.detach().reshape(-1))
    if not flat_parts:
        return torch.zeros((0,), dtype=torch.float32)
    return torch.cat(flat_parts).to(dtype=torch.float32)


def _vector_alignment(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    ref = reference.detach().to(dtype=torch.float32).cpu()
    cand = candidate.detach().to(dtype=torch.float32).cpu()
    ref_norm = float(torch.linalg.vector_norm(ref).item())
    cand_norm = float(torch.linalg.vector_norm(cand).item())
    dot = float(torch.dot(ref, cand).item())
    cosine = 0.0
    if ref_norm > 0.0 and cand_norm > 0.0:
        cosine = float(dot / (ref_norm * cand_norm))
    projection_on_reference = 0.0
    if ref_norm > 0.0:
        projection_on_reference = float(dot / (ref_norm * ref_norm))
    return {
        "dot": dot,
        "cosine": cosine,
        "reference_norm": ref_norm,
        "candidate_norm": cand_norm,
        "projection_on_reference": projection_on_reference,
    }


def _objective_env_indices_present(rollout_data: MaskedRecurrentFlatBatchSamples) -> list[int]:
    objective_mask = np.asarray(
        rollout_data.objective_masks.detach().cpu().numpy() > 1e-8,
        dtype=bool,
    ).reshape(-1)
    flat_env_indices = np.asarray(rollout_data.flat_env_indices, dtype=np.int64).reshape(-1)
    if not bool(np.any(objective_mask)):
        return []
    return sorted(int(v) for v in np.unique(flat_env_indices[objective_mask]).tolist())


def _slice_flat_tensor(tensor: torch.Tensor, *, flat_start: int, flat_end: int) -> torch.Tensor:
    return tensor[flat_start:flat_end]


def _run_flat_outer_batch_step(
    algo,
    rollout_data: MaskedRecurrentFlatBatchSamples,
    *,
    loss_mode: str,
) -> dict[str, Any]:
    if loss_mode not in {"actor_only", "full_main_loss"}:
        raise ValueError(f"unsupported loss_mode: {loss_mode}")

    objective_mask = rollout_data.objective_masks > 1e-8
    objective_total = int(objective_mask.sum().detach().cpu().item())
    if objective_total <= 0:
        raise ValueError("outer batch has no objective tokens")

    advantages = _resolve_actor_advantages(rollout_data)
    if bool(algo.normalize_advantage):
        advantages = _normalize_advantages_with_mask(advantages, objective_mask, eps=1e-8)
    valid_total = objective_mask.to(
        device=rollout_data.returns.device,
        dtype=rollout_data.returns.dtype,
    ).sum().clamp_min(1.0)
    n_seq_total = int(rollout_data.n_seq)
    seq_subbatch_size = int(algo._resolve_sequence_subbatch_size(rollout_data))
    clip_range = float(algo.clip_range(algo._current_progress_remaining))
    value_bardist = algo.policy.get_value_bardist()

    pre_params = _flatten_trainable_params(algo.policy).cpu()
    algo.policy.optimizer.zero_grad(set_to_none=True)

    policy_num_total = rollout_data.returns.new_zeros(())
    value_num_total = rollout_data.returns.new_zeros(())
    entropy_num_total = rollout_data.returns.new_zeros(())
    loss_total = rollout_data.returns.new_zeros(())

    for start_seq in range(0, n_seq_total, seq_subbatch_size):
        end_seq = min(n_seq_total, start_seq + seq_subbatch_size)
        flat_start = int(rollout_data.seq_start_indices[int(start_seq)])
        flat_end = (
            int(rollout_data.seq_start_indices[int(end_seq)])
            if int(end_seq) < n_seq_total
            else int(rollout_data.observations.shape[0])
        )
        sub_seq_lengths = np.asarray(
            rollout_data.seq_lengths[int(start_seq) : int(end_seq)],
            dtype=np.int64,
        )
        sub_advantages = advantages[flat_start:flat_end]
        sub_actions = rollout_data.actions[flat_start:flat_end]
        sub_old_log_prob = rollout_data.old_log_prob[flat_start:flat_end]
        sub_returns = rollout_data.returns[flat_start:flat_end]
        sub_value_target_bucket_idx = getattr(rollout_data, "value_target_bucket_idx", None)
        if sub_value_target_bucket_idx is not None:
            sub_value_target_bucket_idx = sub_value_target_bucket_idx[flat_start:flat_end]
        sub_objective_mask = rollout_data.objective_masks[flat_start:flat_end] > 1e-8
        sub_objective_f = sub_objective_mask.to(
            device=rollout_data.returns.device,
            dtype=rollout_data.returns.dtype,
        )
        sub_action_masks = rollout_data.action_masks[flat_start:flat_end]

        eval_outputs = algo.policy.evaluate_actions_with_hidden_flat(
            rollout_data.observations[flat_start:flat_end],
            sub_actions,
            seq_lengths=sub_seq_lengths,
            action_masks=sub_action_masks,
        )
        value_logits = eval_outputs["value_logits"]
        log_prob = eval_outputs["log_prob"]
        entropy = eval_outputs["entropy"]
        ratio = torch.exp(log_prob - sub_old_log_prob)
        policy_loss_1 = sub_advantages * ratio
        policy_loss_2 = sub_advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
        clipped_objective = torch.min(policy_loss_1, policy_loss_2)
        policy_num = (
            clipped_objective
            * sub_objective_f.to(device=clipped_objective.device, dtype=clipped_objective.dtype)
        ).sum()
        policy_loss = -(policy_num / valid_total.to(device=policy_num.device, dtype=policy_num.dtype))

        if sub_value_target_bucket_idx is not None:
            value_errors = value_bardist.nll_from_bucket_idx(
                value_logits.to(dtype=torch.float32),
                sub_value_target_bucket_idx,
            )
        else:
            value_errors = value_bardist(
                value_logits.to(dtype=torch.float32),
                sub_returns.to(device=value_logits.device, dtype=torch.float32),
            )
        value_num = (
            value_errors
            * sub_objective_f.to(device=value_errors.device, dtype=value_errors.dtype)
        ).sum()
        value_loss = value_num / valid_total.to(device=value_num.device, dtype=value_num.dtype)

        if entropy is None:
            entropy_terms = -log_prob
        else:
            entropy_terms = entropy
        entropy_num = (
            entropy_terms
            * sub_objective_f.to(device=entropy_terms.device, dtype=entropy_terms.dtype)
        ).sum()
        entropy_loss = -(entropy_num / valid_total.to(device=entropy_num.device, dtype=entropy_num.dtype))

        if loss_mode == "actor_only":
            subbatch_loss = policy_loss
        else:
            subbatch_loss = policy_loss + algo.ent_coef * entropy_loss + algo.vf_coef * value_loss

        subbatch_loss.backward()
        policy_num_total = policy_num_total + policy_num.detach()
        value_num_total = value_num_total + value_num.detach()
        entropy_num_total = entropy_num_total + entropy_num.detach()
        loss_total = loss_total + subbatch_loss.detach()

    grad_flat = _flatten_policy_grad(algo.policy).cpu()
    grad_norm_pre_clip = float(torch.linalg.vector_norm(grad_flat).item())
    total_grad_norm_pre_clip = float(torch.nn.utils.clip_grad_norm_(algo.policy.parameters(), algo.max_grad_norm).item())
    algo.policy.optimizer.step()
    post_params = _flatten_trainable_params(algo.policy).cpu()
    delta = (post_params - pre_params).to(dtype=torch.float32)

    return {
        "loss_mode": str(loss_mode),
        "env_index": int(_objective_env_indices_present(rollout_data)[0]),
        "objective_total": int(objective_total),
        "policy_loss": float((-(policy_num_total / valid_total.to(device=policy_num_total.device, dtype=policy_num_total.dtype))).detach().cpu().item()),
        "value_loss": float((value_num_total / valid_total.to(device=value_num_total.device, dtype=value_num_total.dtype)).detach().cpu().item()),
        "entropy_loss": float((-(entropy_num_total / valid_total.to(device=entropy_num_total.device, dtype=entropy_num_total.dtype))).detach().cpu().item()),
        "loss_total_scalar": float(loss_total.detach().cpu().item()),
        "ent_coef": float(algo.ent_coef),
        "vf_coef": float(algo.vf_coef),
        "aux_q_weight": float(getattr(algo, "_rwkv_aux_q_weight", 0.0)),
        "aux_flow_weight": float(getattr(algo, "_rwkv_aux_flow_weight", 0.0)),
        "grad_norm_pre_clip_flat": float(grad_norm_pre_clip),
        "clip_grad_norm_total_pre_clip": float(total_grad_norm_pre_clip),
        "delta_norm": float(torch.linalg.vector_norm(delta).item()),
        "grad_vector": grad_flat,
        "delta_vector": delta,
    }


def _deepcopy_optimizer_state_dict(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    return copy.deepcopy(optimizer.state_dict())


def build_pair2_phase2_scale_local_step_direction_probe(
    *,
    checkpoint_path: str = PAIR2_CHECKPOINT_PATH,
    train_suite_path: str = PAIR2_TRAIN_SUITE_PATH,
    device: str | None = None,
    phase2_summary_path: str = DEFAULT_PHASE2_SUMMARY,
    output_json: str = DEFAULT_OUTPUT_JSON,
    target_outer_batch_idx: int = TARGET_OUTER_BATCH_IDX,
) -> dict[str, Any]:
    phase2_summary = assert_phase2_green(phase2_summary_path)
    checkpoint_path = str(Path(checkpoint_path).expanduser().resolve())
    train_suite_path = str(Path(train_suite_path).expanduser().resolve())
    device_obj = torch.device(str(device or _default_device()))
    suite = _load_required_suite(train_suite_path)

    started_at = time.perf_counter()
    load_model.cache_clear()
    model, config = load_model(checkpoint_path, device=device_obj, verbose=False)
    env_cfg = _build_audit_env_cfg(config["prior"]["environment"])
    prior = EnvironmentPrior(copy.deepcopy(env_cfg))
    prior._sample_batch_hypers = lambda batch_n: _clone_suite_h_list(suite)
    prior._sample_single_eval_pos = lambda n_samples_arg, rng_seed=None: int(PHASE2_EQ_SINGLE_EVAL_POS)

    optimizer_cfg = dict(config.get("optimizer", {}))
    profile_cfg = _resolve_train_profile(
        optimizer_cfg=optimizer_cfg,
        train_profile="phase2_shared_backbone_contract",
        batch_size=int(PHASE2_EQ_BATCH_SIZE),
        n_epochs=1,
        learning_rate=0.0002,
        target_kl=0.03,
    )
    algo, callback, vec_env = build_recurrent_ppo(
        model=model,
        env_prior=prior,
        device=str(device_obj),
        num_features=int(config["prior"]["num_features"]),
        n_envs=int(suite["batch_size"]),
        n_steps=int(PHASE2_EQ_N_STEPS),
        learning_rate=float(profile_cfg["learning_rate"]),
        batch_size=int(profile_cfg["batch_size"]),
        n_epochs=int(profile_cfg["n_epochs"]),
        gamma=float(optimizer_cfg.get("ppo_gamma", 1.0)),
        gae_lambda=float(optimizer_cfg.get("ppo_gae_lambda", 0.95)),
        clip_range=optimizer_cfg.get("ppo_clip_range", 0.2),
        clip_range_vf=None,
        normalize_advantage=bool(profile_cfg["normalize_advantage"]),
        actor_gae_space=str(profile_cfg["actor_gae_space"]),
        actor_baseline_mode=str(profile_cfg["actor_baseline_mode"]),
        actor_objective_mode=str(profile_cfg["actor_objective_mode"]),
        strict_fixed_env_mode=True,
        env_rng_seeds=[int(v) for v in suite["env_seeds"]],
        rollout_rng_seeds=[int(v) for v in suite["rollout_seeds"]],
        deterministic_actor_sampling=True,
        deterministic_batch_plan=True,
        actor_objective_runtime_current_suite_name=None,
        reset_env_state_at_sep=bool(profile_cfg["reset_env_state_at_sep"]),
        separate_value_backbone=bool(profile_cfg["separate_value_backbone"]),
        restore_validation_policy_state=bool(profile_cfg["restore_validation_policy_state"]),
        runtime_normalized_q_value_weight_override=profile_cfg["runtime_normalized_q_value_weight_override"],
        runtime_next_state_flow_matching_weight_override=profile_cfg["runtime_next_state_flow_matching_weight_override"],
        ent_coef=float(optimizer_cfg.get("ppo_ent_coef", 0.0)),
        vf_coef=float(profile_cfg["vf_coef"]),
        max_grad_norm=float(optimizer_cfg.get("ppo_max_grad_norm", 0.5)),
        target_kl=profile_cfg["target_kl"],
        verbose=0,
    )
    _bind_vec_env_to_fixed_suite(vec_env, suite=suite, single_eval_pos=int(PHASE2_EQ_SINGLE_EVAL_POS))
    _make_deterministic_batch_plan(algo.rollout_buffer)
    _init_algo_runtime(
        algo,
        callback,
        vec_env,
        total_timesteps=int(PHASE2_EQ_N_STEPS) * int(PHASE2_EQ_OUTER_EPOCHS),
    )

    try:
        assert algo.collect_rollouts(vec_env, callback, algo.rollout_buffer, n_rollout_steps=algo.n_steps)
        _make_deterministic_batch_plan(algo.rollout_buffer)
        flat_batches = list(algo.rollout_buffer.get_gpu_flat(algo.batch_size))
        if int(target_outer_batch_idx) > int(len(flat_batches)):
            raise ValueError(
                f"trusted rollout only exposed {len(flat_batches)} flat outer batches, "
                f"cannot inspect batch {target_outer_batch_idx}"
            )

        for batch_idx in range(1, int(target_outer_batch_idx)):
            _run_flat_outer_batch_step(algo, flat_batches[int(batch_idx - 1)], loss_mode="full_main_loss")

        pre_branch_policy_state = copy.deepcopy(algo.policy.state_dict())
        pre_branch_optimizer_state = _deepcopy_optimizer_state_dict(algo.policy.optimizer)

        target_batch = flat_batches[int(target_outer_batch_idx - 1)]
        target_batch_envs = _objective_env_indices_present(target_batch)
        if len(target_batch_envs) != 1 or int(target_batch_envs[0]) != 12:
            raise ValueError(
                f"expected target batch {target_outer_batch_idx} to be env12 singleton, got {target_batch_envs}"
            )

        actor_report = _run_flat_outer_batch_step(algo, target_batch, loss_mode="actor_only")

        algo.policy.load_state_dict(copy.deepcopy(pre_branch_policy_state))
        algo.policy.optimizer.load_state_dict(copy.deepcopy(pre_branch_optimizer_state))
        full_report = _run_flat_outer_batch_step(algo, target_batch, loss_mode="full_main_loss")
    finally:
        vec_env.close()

    actor_grad = actor_report.pop("grad_vector")
    actor_delta = actor_report.pop("delta_vector")
    full_grad = full_report.pop("grad_vector")
    full_delta = full_report.pop("delta_vector")
    residual_grad = (full_grad - actor_grad).to(dtype=torch.float32)
    residual_delta = (full_delta - actor_delta).to(dtype=torch.float32)
    grad_actor_vs_full = _vector_alignment(actor_grad, full_grad)
    delta_actor_vs_full = _vector_alignment(actor_delta, full_delta)
    grad_residual_vs_actor = _vector_alignment(actor_grad, residual_grad)
    delta_residual_vs_actor = _vector_alignment(actor_delta, residual_delta)
    aux_weights_nonzero = bool(
        float(actor_report["aux_q_weight"]) > 0.0
        or float(actor_report["aux_flow_weight"]) > 0.0
        or float(full_report["aux_q_weight"]) > 0.0
        or float(full_report["aux_flow_weight"]) > 0.0
    )

    report = {
        "probe_entry": "phase3_pair2_phase2_scale_local_step_direction_probe",
        "config": {
            "checkpoint_path": checkpoint_path,
            "train_suite_path": train_suite_path,
            "device": str(device_obj),
            "n_steps": int(PHASE2_EQ_N_STEPS),
            "batch_size": int(PHASE2_EQ_BATCH_SIZE),
            "outer_epochs_reference": int(PHASE2_EQ_OUTER_EPOCHS),
            "single_eval_pos": int(PHASE2_EQ_SINGLE_EVAL_POS),
            "strict_fixed_env_mode": True,
            "deterministic_actor_sampling": True,
            "deterministic_batch_plan": True,
            "target_outer_batch_idx": int(target_outer_batch_idx),
            "pre_branch_prefix_steps": int(target_outer_batch_idx - 1),
            "read_only_branch_compare": True,
            "compare_scope": "actor_only vs policy+entropy+value main_loss on the same pre-batch13 state",
        },
        "phase2_reference_contract": dict(phase2_summary["trusted_contract"]),
        "runtime": {
            "elapsed_wall_time_sec": float(time.perf_counter() - started_at),
        },
        "target_batch": {
            "outer_batch_idx": int(target_outer_batch_idx),
            "env_index": int(target_batch_envs[0]),
            "objective_total": int((target_batch.objective_masks > 1e-8).sum().detach().cpu().item()),
        },
        "actor_only": actor_report,
        "full_main_loss": full_report,
        "alignment": {
            "grad_actor_vs_full": grad_actor_vs_full,
            "delta_actor_vs_full": delta_actor_vs_full,
            "grad_residual_vs_actor": grad_residual_vs_actor,
            "delta_residual_vs_actor": delta_residual_vs_actor,
        },
        "conclusions": {
            "full_main_loss_delta_is_not_opposed_to_actor_only_step": bool(
                delta_actor_vs_full["projection_on_reference"] > 0.0
            ),
            "full_main_loss_gradient_is_dominated_by_non_actor_component": bool(
                grad_actor_vs_full["candidate_norm"] > 10.0 * max(1e-12, grad_actor_vs_full["reference_norm"])
                and abs(grad_actor_vs_full["cosine"]) < 0.2
            ),
            "full_main_loss_residual_is_roughly_orthogonal_not_opposing": bool(
                abs(delta_residual_vs_actor["projection_on_reference"]) < 0.1
                and abs(delta_residual_vs_actor["cosine"]) < 0.2
            ),
            "auxiliary_shared_core_path_is_still_unchecked_in_this_compare": bool(aux_weights_nonzero),
            "shared_core_value_path_simple_opposition_is_not_supported": bool(
                delta_residual_vs_actor["projection_on_reference"] >= 0.0
            ),
        },
        "recommendation": {
            "next_focus": "local_step_component_split",
            "reason": (
                "This compare does not support a simple opposite-direction value interference story. "
                "The next narrow question is whether the non-actor local-step component is mainly value-path or "
                "auxiliary shared-core traffic."
            ),
        },
    }
    Path(output_json).expanduser().resolve().write_text(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only local-step direction compare between actor-only and full PPO main-loss on trusted batch 13."
    )
    parser.add_argument("--checkpoint-path", type=str, default=PAIR2_CHECKPOINT_PATH)
    parser.add_argument("--train-suite-path", type=str, default=PAIR2_TRAIN_SUITE_PATH)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--phase2-summary-path", type=str, default=DEFAULT_PHASE2_SUMMARY)
    parser.add_argument("--output-json", type=str, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--target-outer-batch-idx", type=int, default=TARGET_OUTER_BATCH_IDX)
    args = parser.parse_args()

    report = build_pair2_phase2_scale_local_step_direction_probe(
        checkpoint_path=args.checkpoint_path,
        train_suite_path=args.train_suite_path,
        device=args.device,
        phase2_summary_path=args.phase2_summary_path,
        output_json=args.output_json,
        target_outer_batch_idx=int(args.target_outer_batch_idx),
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
