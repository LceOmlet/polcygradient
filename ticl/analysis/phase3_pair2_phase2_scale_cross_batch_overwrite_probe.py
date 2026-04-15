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
    build_recurrent_ppo,
)


DEFAULT_OUTPUT_JSON = (
    "/home/chen/RLPFN/artifacts/phase3_pair2_phase2_scale_cross_batch_overwrite_probe.json"
)

PHASE2_EQ_N_STEPS = 256
PHASE2_EQ_BATCH_SIZE = 256
PHASE2_EQ_OUTER_EPOCHS = 2
PHASE2_EQ_SINGLE_EVAL_POS = 64

TARGET_OUTER_BATCH_IDX = 13
FOLLOWING_OUTER_BATCH_IDXS = (14, 15, 16)


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


def _objective_env_indices_present(rollout_data) -> list[int]:
    objective_mask = np.asarray(
        rollout_data.objective_masks.detach().cpu().numpy() > 1e-8,
        dtype=bool,
    ).reshape(-1)
    flat_env_indices = np.asarray(rollout_data.flat_env_indices, dtype=np.int64).reshape(-1)
    if not bool(np.any(objective_mask)):
        return []
    return sorted(int(v) for v in np.unique(flat_env_indices[objective_mask]).tolist())


def _simulate_policy_only_cross_batch_overwrite(
    algo,
    *,
    target_outer_batch_idx: int,
    following_outer_batch_idxs: tuple[int, ...],
) -> dict[str, Any]:
    target_outer_batch_idx = int(target_outer_batch_idx)
    following_outer_batch_idxs = tuple(int(v) for v in following_outer_batch_idxs)
    selected_outer_batch_idxs = {target_outer_batch_idx, *following_outer_batch_idxs}

    iterator = algo.rollout_buffer.get_gpu_flat(algo.batch_size)
    selected_reports: dict[int, dict[str, Any]] = {}
    total_outer_batches = 0

    for rollout_data in iterator:
        total_outer_batches += 1
        objective_mask = rollout_data.objective_masks > 1e-8
        objective_total = int(objective_mask.sum().detach().cpu().item())
        if objective_total <= 0:
            continue

        advantages = _resolve_actor_advantages(rollout_data)
        if bool(algo.normalize_advantage):
            advantages = _normalize_advantages_with_mask(advantages, objective_mask, eps=1e-8)
        valid_total = objective_mask.to(
            device=rollout_data.returns.device,
            dtype=rollout_data.returns.dtype,
        ).sum().clamp_min(1.0)

        capture_this_batch = int(total_outer_batches) in selected_outer_batch_idxs
        pre_params = None
        if capture_this_batch:
            pre_params = _flatten_trainable_params(algo.policy)

        eval_outputs = algo.policy.evaluate_actions_with_hidden_flat(
            rollout_data.observations,
            rollout_data.actions,
            seq_lengths=rollout_data.seq_lengths,
            action_masks=rollout_data.action_masks,
        )
        clip_range = algo.clip_range(algo._current_progress_remaining)
        log_prob = eval_outputs["log_prob"]
        ratio = torch.exp(log_prob - rollout_data.old_log_prob)
        policy_loss_1 = advantages * ratio
        policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
        clipped_objective = torch.min(policy_loss_1, policy_loss_2)
        policy_num = (
            clipped_objective
            * objective_mask.to(device=clipped_objective.device, dtype=clipped_objective.dtype)
        ).sum()
        policy_loss = -(policy_num / valid_total.to(device=policy_num.device, dtype=policy_num.dtype))

        algo.policy.optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        grad_flat = _flatten_policy_grad(algo.policy).cpu() if capture_this_batch else None
        grad_norm = (
            float(torch.linalg.vector_norm(grad_flat).item())
            if grad_flat is not None
            else None
        )
        total_grad_norm = float(torch.nn.utils.clip_grad_norm_(algo.policy.parameters(), algo.max_grad_norm).item())
        algo.policy.optimizer.step()

        if capture_this_batch:
            assert pre_params is not None
            post_params = _flatten_trainable_params(algo.policy).cpu()
            delta = (post_params - pre_params.cpu()).to(dtype=torch.float32)
            selected_reports[int(total_outer_batches)] = {
                "outer_batch_idx": int(total_outer_batches),
                "objective_env_indices_present": _objective_env_indices_present(rollout_data),
                "objective_total": int(objective_total),
                "policy_loss": float(policy_loss.detach().cpu().item()),
                "policy_num": float(policy_num.detach().cpu().item()),
                "clip_range": float(clip_range),
                "grad_norm_pre_clip_flat": None if grad_norm is None else float(grad_norm),
                "clip_grad_norm_total_pre_clip": float(total_grad_norm),
                "delta_norm": float(torch.linalg.vector_norm(delta).item()),
                "delta_vector": delta,
            }

    if int(total_outer_batches) < int(max(selected_outer_batch_idxs)):
        raise ValueError(
            f"trusted rollout only exposed {total_outer_batches} outer batches, "
            f"cannot inspect up to batch {max(selected_outer_batch_idxs)}"
        )
    if int(target_outer_batch_idx) not in selected_reports:
        raise ValueError(f"target outer batch {target_outer_batch_idx} was not captured")
    for idx in following_outer_batch_idxs:
        if int(idx) not in selected_reports:
            raise ValueError(f"following outer batch {idx} was not captured")

    target_delta = selected_reports[int(target_outer_batch_idx)]["delta_vector"]
    pairwise = {}
    downstream_delta = torch.zeros_like(target_delta)
    for idx in following_outer_batch_idxs:
        candidate_delta = selected_reports[int(idx)]["delta_vector"]
        pairwise[str(int(idx))] = {
            "outer_batch_idx": int(idx),
            "env_index": int(selected_reports[int(idx)]["objective_env_indices_present"][0]),
            "delta_alignment_vs_target": _vector_alignment(target_delta, candidate_delta),
        }
        downstream_delta = downstream_delta + candidate_delta

    downstream_alignment = _vector_alignment(target_delta, downstream_delta)
    net_after_target_alignment = _vector_alignment(target_delta, target_delta + downstream_delta)

    cleaned_selected_reports = {}
    for idx, report in selected_reports.items():
        cleaned = dict(report)
        cleaned["env_index"] = int(cleaned["objective_env_indices_present"][0]) if cleaned["objective_env_indices_present"] else None
        cleaned.pop("delta_vector")
        cleaned_selected_reports[str(int(idx))] = cleaned

    return {
        "total_outer_batches": int(total_outer_batches),
        "selected_outer_batches": cleaned_selected_reports,
        "pairwise_downstream_alignment": pairwise,
        "downstream_cumulative_alignment_vs_target": downstream_alignment,
        "net_alignment_after_target_plus_downstream": net_after_target_alignment,
    }


def build_pair2_phase2_scale_cross_batch_overwrite_probe(
    *,
    checkpoint_path: str = PAIR2_CHECKPOINT_PATH,
    train_suite_path: str = PAIR2_TRAIN_SUITE_PATH,
    device: str | None = None,
    phase2_summary_path: str = DEFAULT_PHASE2_SUMMARY,
    output_json: str = DEFAULT_OUTPUT_JSON,
    target_outer_batch_idx: int = TARGET_OUTER_BATCH_IDX,
    following_outer_batch_idxs: tuple[int, ...] = FOLLOWING_OUTER_BATCH_IDXS,
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
        overwrite_report = _simulate_policy_only_cross_batch_overwrite(
            algo,
            target_outer_batch_idx=int(target_outer_batch_idx),
            following_outer_batch_idxs=tuple(int(v) for v in following_outer_batch_idxs),
        )
    finally:
        vec_env.close()

    downstream_projection = float(
        overwrite_report["downstream_cumulative_alignment_vs_target"]["projection_on_reference"]
    )
    net_retention = float(
        overwrite_report["net_alignment_after_target_plus_downstream"]["projection_on_reference"]
    )

    report = {
        "probe_entry": "phase3_pair2_phase2_scale_cross_batch_overwrite_probe",
        "config": {
            "checkpoint_path": checkpoint_path,
            "train_suite_path": train_suite_path,
            "device": str(device_obj),
            "n_steps": int(PHASE2_EQ_N_STEPS),
            "batch_size": int(PHASE2_EQ_BATCH_SIZE),
            "outer_epochs_reference": int(PHASE2_EQ_OUTER_EPOCHS),
            "single_eval_pos": int(PHASE2_EQ_SINGLE_EVAL_POS),
            "train_profile": str(profile_cfg["train_profile"]),
            "ppo_actor_baseline_mode": str(profile_cfg["actor_baseline_mode"]),
            "ppo_normalize_advantage": bool(profile_cfg["normalize_advantage"]),
            "ppo_actor_gae_space": str(profile_cfg["actor_gae_space"]),
            "ppo_vf_coef": float(profile_cfg["vf_coef"]),
            "ppo_reset_env_state_at_sep": bool(profile_cfg["reset_env_state_at_sep"]),
            "ppo_separate_value_backbone": bool(profile_cfg["separate_value_backbone"]),
            "ppo_restore_validation_policy_state": bool(profile_cfg["restore_validation_policy_state"]),
            "runtime_normalized_q_value_weight_override": (
                None
                if profile_cfg["runtime_normalized_q_value_weight_override"] is None
                else float(profile_cfg["runtime_normalized_q_value_weight_override"])
            ),
            "runtime_next_state_flow_matching_weight_override": (
                None
                if profile_cfg["runtime_next_state_flow_matching_weight_override"] is None
                else float(profile_cfg["runtime_next_state_flow_matching_weight_override"])
            ),
            "strict_fixed_env_mode": True,
            "deterministic_actor_sampling": True,
            "deterministic_batch_plan": True,
            "target_outer_batch_idx": int(target_outer_batch_idx),
            "following_outer_batch_idxs": [int(v) for v in following_outer_batch_idxs],
            "policy_only_read_only_simulation": True,
        },
        "phase2_reference_contract": dict(phase2_summary["trusted_contract"]),
        "runtime": {
            "elapsed_wall_time_sec": float(time.perf_counter() - started_at),
        },
        "overwrite_report": overwrite_report,
        "conclusions": {
            "downstream_steps_have_net_negative_projection_on_target_direction": bool(downstream_projection < 0.0),
            "downstream_steps_reduce_target_step_retention": bool(net_retention < 1.0),
            "downstream_steps_overwrite_more_than_half_of_target_direction": bool(net_retention < 0.5),
            "downstream_steps_fully_reverse_target_direction": bool(net_retention < 0.0),
        },
        "recommendation": {
            "next_focus": "cross_batch_overwrite_localization",
            "reason": (
                "If the downstream steps have a negative projection on the target step direction, the next narrow "
                "question is which later env-step is responsible for the overwrite and whether that overwrite is "
                "shared-core or branch-specific."
            ),
        },
    }
    Path(output_json).expanduser().resolve().write_text(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only actor-only cross-batch overwrite probe on the trusted Phase-2-scale pair2 runtime."
    )
    parser.add_argument("--checkpoint-path", type=str, default=PAIR2_CHECKPOINT_PATH)
    parser.add_argument("--train-suite-path", type=str, default=PAIR2_TRAIN_SUITE_PATH)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--phase2-summary-path", type=str, default=DEFAULT_PHASE2_SUMMARY)
    parser.add_argument("--output-json", type=str, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--target-outer-batch-idx", type=int, default=TARGET_OUTER_BATCH_IDX)
    parser.add_argument(
        "--following-outer-batch-idxs",
        type=int,
        nargs="+",
        default=list(FOLLOWING_OUTER_BATCH_IDXS),
    )
    args = parser.parse_args()

    report = build_pair2_phase2_scale_cross_batch_overwrite_probe(
        checkpoint_path=args.checkpoint_path,
        train_suite_path=args.train_suite_path,
        device=args.device,
        phase2_summary_path=args.phase2_summary_path,
        output_json=args.output_json,
        target_outer_batch_idx=int(args.target_outer_batch_idx),
        following_outer_batch_idxs=tuple(int(v) for v in args.following_outer_batch_idxs),
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
