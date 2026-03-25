import torch


def prepare_policy_gradient_loss_request(
    prior,
    *,
    n_samples,
    tbptt_window,
    policy_objective_kind="policy_gradient",
    normalize=None,
):
    n_samples = int(n_samples)
    objective_kind = prior._normalize_policy_objective_kind(policy_objective_kind)
    reinforce_enabled = objective_kind == "reinforce"
    first_pg_enabled = objective_kind == "first_policy_gradient"
    alpha_grad_enabled = objective_kind == "alpha_grad"
    if normalize is None:
        normalize = bool(prior.config.get("policy_gradient_normalize_rewards", False))
    tbptt_window_active = False
    tbptt_window_size = n_samples
    if tbptt_window is not None:
        w = int(tbptt_window)
        if 0 < w < n_samples:
            tbptt_window_active = True
            tbptt_window_size = w
    backend = prior._resolve_batch_parallel_backend()
    grouping_mode = prior._resolve_batch_vectorized_grouping()
    strict_rng_match = prior._resolve_batch_vectorized_strict_rng_match()
    one_hop_replay_enabled = bool(prior._resolve_pg_one_hop_replay_enabled(prior.config))
    markov_adjacent_replay_enabled = bool(prior._resolve_pg_markov_adjacent_replay_enabled(prior.config))
    markov_adjacent_replay_sample_prob = float(
        prior._resolve_pg_markov_adjacent_replay_sample_prob(prior.config)
    )
    one_hop_tbptt_active = bool(
        one_hop_replay_enabled
        and tbptt_window_active
        and (reinforce_enabled or alpha_grad_enabled)
    )
    return {
        "n_samples": n_samples,
        "objective_kind": objective_kind,
        "reinforce_enabled": reinforce_enabled,
        "first_pg_enabled": first_pg_enabled,
        "alpha_grad_enabled": alpha_grad_enabled,
        "normalize": normalize,
        "tbptt_window_active": tbptt_window_active,
        "tbptt_window_size": tbptt_window_size,
        "backend": backend,
        "grouping_mode": grouping_mode,
        "strict_rng_match": strict_rng_match,
        "one_hop_replay_enabled": one_hop_replay_enabled,
        "markov_adjacent_replay_enabled": markov_adjacent_replay_enabled,
        "markov_adjacent_replay_sample_prob": markov_adjacent_replay_sample_prob,
        "one_hop_tbptt_active": one_hop_tbptt_active,
    }


def slice_group_traces_after_eval(group_entries, start_idx):
    if not isinstance(group_entries, tuple):
        return group_entries
    sliced_entries = []
    for entry in group_entries:
        if not isinstance(entry, dict):
            continue
        roots = entry.get("action_mean_roots", None)
        mask = entry.get("action_mask", None)
        if not isinstance(roots, tuple):
            continue
        sliced_roots = tuple(roots[int(start_idx):])
        if len(sliced_roots) <= 0:
            continue
        sliced_entry = dict(entry)
        sliced_entry["action_mean_roots"] = sliced_roots
        if torch.is_tensor(mask):
            sliced_entry["action_mask"] = mask[int(start_idx):]
        sliced_entries.append(sliced_entry)
    return tuple(sliced_entries)


def attach_common_rollout_diagnostics(rollout, stats):
    if not isinstance(stats, dict):
        raise TypeError("stats must be a dict")
    rewards = rollout.get("rewards", None) if isinstance(rollout, dict) else None
    if not torch.is_tensor(rewards):
        raise TypeError("rollout['rewards'] must be a tensor")
    device = rewards.device
    dtype = torch.float32

    rollout_profile = rollout.get("rollout_profile", None)
    if isinstance(rollout_profile, dict):
        stats["rollout_policy_cuda_ms"] = float(rollout_profile.get("policy_cuda_ms", 0.0))
        stats["rollout_transition_cuda_ms"] = float(rollout_profile.get("transition_cuda_ms", 0.0))
        stats["rollout_policy_wall_ms"] = float(rollout_profile.get("policy_wall_ms", 0.0))
        stats["rollout_transition_wall_ms"] = float(rollout_profile.get("transition_wall_ms", 0.0))
        stats["rollout_transition_y_wall_ms"] = float(rollout_profile.get("transition_y_wall_ms", 0.0))
        stats["rollout_transition_x_wall_ms"] = float(rollout_profile.get("transition_x_wall_ms", 0.0))
        stats["rollout_transition_group_wall_ms"] = float(
            rollout_profile.get("transition_group_wall_ms", 0.0)
        )
        stats["rollout_transition_group_launch_wall_ms"] = float(
            rollout_profile.get("transition_group_launch_wall_ms", 0.0)
        )
        stats["rollout_transition_group_sync_wall_ms"] = float(
            rollout_profile.get("transition_group_sync_wall_ms", 0.0)
        )
        stats["rollout_transition_env_pack_wall_ms"] = float(
            rollout_profile.get("transition_env_pack_wall_ms", 0.0)
        )
        stats["rollout_transition_state_update_wall_ms"] = float(
            rollout_profile.get("transition_state_update_wall_ms", 0.0)
        )
        stats["rollout_transition_noise_wall_ms"] = float(
            rollout_profile.get("transition_noise_wall_ms", 0.0)
        )
        stats["rollout_transition_fused_wall_ms"] = float(
            rollout_profile.get("transition_fused_wall_ms", 0.0)
        )
        stats["rollout_transition_fused_launch_wall_ms"] = float(
            rollout_profile.get("transition_fused_launch_wall_ms", 0.0)
        )
        stats["rollout_transition_gp_first_projection_wall_ms"] = float(
            rollout_profile.get("transition_gp_first_projection_wall_ms", 0.0)
        )
        stats["rollout_transition_gp_second_projection_wall_ms"] = float(
            rollout_profile.get("transition_gp_second_projection_wall_ms", 0.0)
        )
        stats["rollout_transition_gp_projection_call_count"] = int(
            rollout_profile.get("transition_gp_projection_call_count", 0) or 0
        )
        stats["rollout_transition_gp_rff_fused_call_count"] = int(
            rollout_profile.get("transition_gp_rff_fused_call_count", 0) or 0
        )
        stats["rollout_transition_gp_profile_group_count"] = int(
            rollout_profile.get("transition_gp_profile_group_count", 0) or 0
        )
        stats["rollout_transition_gp_profile_sync_group_count"] = int(
            rollout_profile.get("transition_gp_profile_sync_group_count", 0) or 0
        )
        stats["rollout_transition_gp_shared_total_wall_ms"] = float(
            rollout_profile.get("transition_gp_shared_total_wall_ms", 0.0) or 0.0
        )
        stats["rollout_transition_gp_shared_core_wall_ms"] = float(
            rollout_profile.get("transition_gp_shared_core_wall_ms", 0.0) or 0.0
        )
        stats["rollout_transition_gp_shared_noise_wall_ms"] = float(
            rollout_profile.get("transition_gp_shared_noise_wall_ms", 0.0) or 0.0
        )
        stats["rollout_transition_gp_shared_checkpoint_wall_ms"] = float(
            rollout_profile.get("transition_gp_shared_checkpoint_wall_ms", 0.0) or 0.0
        )
        stats["rollout_transition_gp_shared_post_wall_ms"] = float(
            rollout_profile.get("transition_gp_shared_post_wall_ms", 0.0) or 0.0
        )
        stats["rollout_transition_gp_shared_call_count"] = int(
            rollout_profile.get("transition_gp_shared_call_count", 0) or 0
        )
        stats["rollout_transition_packed_env_input_group_count"] = int(
            rollout_profile.get("transition_packed_env_input_group_count", 0) or 0
        )
        stats["rollout_transition_packed_env_input_call_count"] = int(
            rollout_profile.get("transition_packed_env_input_call_count", 0) or 0
        )
        stats["rollout_transition_only_build_group_count"] = int(
            rollout_profile.get("transition_only_build_group_count", 0) or 0
        )
        stats["rollout_transition_only_skipped_generator_count"] = int(
            rollout_profile.get("transition_only_skipped_generator_count", 0) or 0
        )
        stats["rollout_transition_setup_wall_ms"] = float(
            rollout_profile.get("transition_setup_wall_ms", 0.0) or 0.0
        )
        stats["rollout_transition_family_build_wall_ms"] = float(
            rollout_profile.get("transition_family_build_wall_ms", 0.0) or 0.0
        )
        stats["rollout_transition_generator_build_wall_ms"] = float(
            rollout_profile.get("transition_generator_build_wall_ms", 0.0) or 0.0
        )
        stats["rollout_transition_gp_shared_build_wall_ms"] = float(
            rollout_profile.get("transition_gp_shared_build_wall_ms", 0.0) or 0.0
        )
        stats["rollout_transition_fused_call_count"] = int(
            rollout_profile.get("transition_fused_call_count", 0) or 0
        )
        stats["rollout_transition_fused_group_count"] = int(
            rollout_profile.get("transition_fused_group_count", 0) or 0
        )
        stats["rollout_transition_fused_enabled"] = int(
            rollout_profile.get("transition_fused_enabled", 0) or 0
        )
        stats["rollout_transition_checkpoint_enabled"] = int(
            rollout_profile.get("transition_checkpoint_enabled", 0) or 0
        )
        stats["rollout_transition_checkpoint_call_count"] = int(
            rollout_profile.get("transition_checkpoint_call_count", 0) or 0
        )
        stats["rollout_transition_group_count"] = int(rollout_profile.get("transition_group_count", 0))
        stats["rollout_transition_family_group_count"] = int(
            rollout_profile.get("transition_family_group_count", 0) or 0
        )
        stats["rollout_transition_inner_grouping_structure_enabled"] = int(
            rollout_profile.get("transition_inner_grouping_structure_enabled", 0) or 0
        )
        stats["rollout_transition_inner_min_bucket"] = int(
            rollout_profile.get("transition_inner_min_bucket", 0) or 0
        )
        stats["rollout_transition_bucket_max_batch"] = int(
            rollout_profile.get("transition_bucket_max_batch", 0) or 0
        )
        stats["rollout_transition_bucket_mean_batch"] = float(
            rollout_profile.get("transition_bucket_mean_batch", 0.0) or 0.0
        )
        stats["rollout_transition_work_actual_est"] = float(
            rollout_profile.get("transition_work_actual_est", 0.0) or 0.0
        )
        stats["rollout_transition_work_padded_est"] = float(
            rollout_profile.get("transition_work_padded_est", 0.0) or 0.0
        )
        stats["rollout_transition_work_fill_ratio"] = float(
            rollout_profile.get("transition_work_fill_ratio", 0.0) or 0.0
        )
        stats["rollout_transition_async_enabled"] = int(
            rollout_profile.get("transition_async_enabled", 0) or 0
        )
        stats["rollout_transition_async_commit_in_stream"] = int(
            rollout_profile.get("transition_async_commit_in_stream", 0) or 0
        )
        for key in (
            "reward_env_mean",
            "reward_env_std",
            "reward_ctrl_mean",
            "reward_ctrl_std",
            "reward_survival_mean",
            "reward_survival_std",
        ):
            value = rollout_profile.get(key, None)
            if value is not None:
                stats[key] = value
        stats["rollout_noise_mode"] = rollout_profile.get("noise_mode", None)
        stats["rollout_noise_block_size"] = int(rollout_profile.get("noise_block_size", 0) or 0)
        stats["rollout_env_count"] = int(rollout_profile.get("env_count", 0) or 0)
        stats["rollout_strict_joint_transition_count"] = int(
            rollout_profile.get("strict_joint_transition_count", 0) or 0
        )
        stats["rollout_strict_joint_transition_share"] = float(
            rollout_profile.get("strict_joint_transition_share", 0.0) or 0.0
        )
        stats["rollout_reference_semantics_count"] = int(
            rollout_profile.get("reference_semantics_count", 0) or 0
        )
        stats["rollout_reference_semantics_share"] = float(
            rollout_profile.get("reference_semantics_share", 0.0) or 0.0
        )
        stats["rollout_exact_scm_count"] = int(rollout_profile.get("exact_scm_count", 0) or 0)
        stats["rollout_exact_gp_count"] = int(rollout_profile.get("exact_gp_count", 0) or 0)
        stats["rollout_legacy_scm_count"] = int(rollout_profile.get("legacy_scm_count", 0) or 0)
        stats["rollout_legacy_gp_count"] = int(rollout_profile.get("legacy_gp_count", 0) or 0)
        stats["rollout_transition_reference_mode"] = rollout_profile.get(
            "transition_reference_mode",
            None,
        )

    terminal_stats = rollout.get("terminal_stats", None)
    if isinstance(terminal_stats, dict):
        for key in (
            "terminal_count_mean",
            "terminal_count_min",
            "terminal_count_max",
            "terminal_count_target_mean",
            "terminal_count_target_min",
            "terminal_count_target_max",
        ):
            value = terminal_stats.get(key, None)
            if value is None:
                continue
            stats[key] = (
                value.detach()
                if torch.is_tensor(value)
                else torch.as_tensor(value, device=device, dtype=dtype)
            )
    return stats
