import torch


def prepare_policy_rollout_request(
    prior,
    batch_size,
    *,
    h_list_override=None,
    env_seeds_override=None,
    rollout_seeds_override=None,
):
    batch_size = int(batch_size)
    backend = prior._resolve_batch_parallel_backend()
    strict_rng_match = prior._resolve_batch_vectorized_strict_rng_match()
    grouping_mode = prior._resolve_batch_vectorized_grouping()
    if h_list_override is not None:
        if len(h_list_override) != batch_size:
            raise ValueError("h_list_override length must match batch_size")
        h_list = list(h_list_override)
    else:
        h_list = prior._sample_batch_hypers(batch_size)
    if env_seeds_override is not None:
        if len(env_seeds_override) != batch_size:
            raise ValueError("env_seeds_override length must match batch_size")
        env_seeds = [int(s) for s in env_seeds_override]
    else:
        env_seeds = prior._sample_seed_list(batch_size) if strict_rng_match else None
    if rollout_seeds_override is not None:
        if len(rollout_seeds_override) != batch_size:
            raise ValueError("rollout_seeds_override length must match batch_size")
        rollout_seeds = [int(s) for s in rollout_seeds_override]
    else:
        rollout_seeds = prior._sample_seed_list(batch_size) if strict_rng_match else None
    return {
        "backend": backend,
        "strict_rng_match": bool(strict_rng_match),
        "grouping_mode": grouping_mode,
        "h_list": h_list,
        "env_seeds": env_seeds,
        "rollout_seeds": rollout_seeds,
    }


def _new_rollout_profile_accumulator(*, n_samples, batch_size):
    return {
        "policy_cuda_ms": 0.0,
        "transition_cuda_ms": 0.0,
        "policy_wall_ms": 0.0,
        "transition_wall_ms": 0.0,
        "transition_y_wall_ms": 0.0,
        "transition_x_wall_ms": 0.0,
        "transition_group_wall_ms": 0.0,
        "transition_group_launch_wall_ms": 0.0,
        "transition_group_sync_wall_ms": 0.0,
        "transition_env_pack_wall_ms": 0.0,
        "transition_state_update_wall_ms": 0.0,
        "transition_noise_wall_ms": 0.0,
        "transition_fused_wall_ms": 0.0,
        "transition_fused_launch_wall_ms": 0.0,
        "transition_gp_first_projection_wall_ms": 0.0,
        "transition_gp_second_projection_wall_ms": 0.0,
        "transition_gp_projection_call_count": 0,
        "transition_gp_rff_fused_call_count": 0,
        "transition_gp_profile_group_count": 0,
        "transition_gp_profile_sync_group_count": 0,
        "transition_gp_shared_total_wall_ms": 0.0,
        "transition_gp_shared_core_wall_ms": 0.0,
        "transition_gp_shared_noise_wall_ms": 0.0,
        "transition_gp_shared_checkpoint_wall_ms": 0.0,
        "transition_gp_shared_post_wall_ms": 0.0,
        "transition_gp_shared_call_count": 0,
        "transition_packed_env_input_group_count": 0,
        "transition_packed_env_input_call_count": 0,
        "transition_only_build_group_count": 0,
        "transition_only_skipped_generator_count": 0,
        "transition_setup_wall_ms": 0.0,
        "transition_family_build_wall_ms": 0.0,
        "transition_generator_build_wall_ms": 0.0,
        "transition_gp_shared_build_wall_ms": 0.0,
        "transition_fused_call_count": 0,
        "transition_fused_group_count": 0,
        "transition_fused_enabled": 0,
        "transition_checkpoint_enabled": 0,
        "transition_checkpoint_call_count": 0,
        "transition_group_count": 0,
        "transition_family_group_count": 0,
        "transition_inner_grouping_structure_enabled": 0,
        "transition_inner_min_bucket": 0,
        "transition_bucket_max_batch": 0,
        "transition_work_actual_est": 0.0,
        "transition_work_padded_est": 0.0,
        "transition_async_enabled": 0,
        "noise_mode": None,
        "noise_block_size": 0,
        "steps": int(n_samples),
        "batch_size": int(batch_size),
        "env_count": 0,
        "strict_joint_transition_count": 0,
        "reference_semantics_count": 0,
        "exact_scm_count": 0,
        "exact_gp_count": 0,
        "fixed_gp_count": 0,
        "legacy_scm_count": 0,
        "legacy_gp_count": 0,
    }


def _project_rollout_env_semantics(profile):
    return (
        {
            key: profile[key]
            for key in (
                "env_count",
                "strict_joint_transition_count",
                "strict_joint_transition_share",
                "reference_semantics_count",
                "reference_semantics_share",
                "exact_scm_count",
                "exact_gp_count",
                "legacy_scm_count",
                "legacy_gp_count",
                "transition_reference_mode",
            )
        }
        if isinstance(profile, dict)
        else None
    )


def dispatch_policy_rollout(prior, ctx):
    policy_step_fn = ctx["policy_step_fn"]
    batch_size = int(ctx["batch_size"])
    n_samples = int(ctx["n_samples"])
    num_features = int(ctx["num_features"])
    single_eval_pos = int(ctx["single_eval_pos"])
    device = ctx["device"]
    collect_x = bool(ctx["collect_x"])
    collect_runtime_info = bool(ctx["collect_runtime_info"])
    tbptt_window = ctx["tbptt_window"]
    tbptt_reward_sink_supports_aux = bool(ctx["tbptt_reward_sink_supports_aux"])
    store_rewards = bool(ctx["store_rewards"])
    policy_objective_kind = ctx["policy_objective_kind"]
    policy_collect_log_probs = bool(ctx["_policy_collect_log_probs"])
    policy_collect_action_trace = bool(ctx["_policy_collect_action_trace"])
    policy_collect_reinforce_replay = bool(ctx.get("_policy_collect_reinforce_replay", False))
    policy_detach_action_in_env = ctx["_policy_detach_action_in_env"]
    policy_disable_log_probs = bool(ctx.get("_policy_disable_log_probs", False))
    policy_force_no_grad = bool(ctx.get("_policy_force_no_grad", False))
    policy_defer_reinforce_replay = bool(ctx.get("_policy_defer_reinforce_replay", False))
    alpha_grad_trace_roots_only = bool(ctx["alpha_grad_trace_roots_only"])
    backend = ctx["backend"]
    strict_rng_match = bool(ctx["strict_rng_match"])
    grouping_mode = ctx["grouping_mode"]
    h_list = ctx["h_list"]
    env_seeds = ctx["env_seeds"]
    rollout_seeds = ctx["rollout_seeds"]
    make_alpha_grad_tbptt_group_sink = ctx["make_alpha_grad_tbptt_group_sink"]
    alpha_grad_outer_merge_state = ctx["alpha_grad_outer_merge_state"]

    x = ctx["x"]
    rewards = ctx["rewards"]
    reinforce_log_probs = ctx["reinforce_log_probs"]
    reinforce_log_prob_scores = None
    policy_action_mean = None
    policy_action_mask = None
    policy_action_mean_roots = None
    policy_action_mean_root_steps = None
    policy_action_group_traces = []
    infos = ctx["infos"]

    collect_action_trace = bool(policy_collect_action_trace)
    reinforce_sequence_replay_fn = getattr(policy_step_fn, "_reinforce_sequence_replay_fn", None)
    fit_action_dim_fn = getattr(policy_step_fn, "_fit_action_dim_fn", None)
    reinforce_sequence_replay_applied = False
    reinforce_sequence_replay_eval_start = None
    reinforce_sequence_replay_action_steps = None
    reinforce_replay_payload = None
    reinforce_decomp_accum = {
        "reinforce_action_dim_mean": 0.0,
        "reinforce_action_std_mean": 0.0,
        "reinforce_logprob_log_std_mean": 0.0,
        "reinforce_logprob_z2_mean": 0.0,
    }
    reinforce_decomp_count = 0.0
    reinforce_decomp_min = {
        "reinforce_action_dim_min": float("inf"),
        "reinforce_action_std_min": float("inf"),
        "reinforce_logprob_log_std_min": float("inf"),
    }
    reinforce_decomp_max = {
        "reinforce_action_dim_max": float("-inf"),
        "reinforce_action_std_max": float("-inf"),
        "reinforce_logprob_log_std_max": float("-inf"),
        "reinforce_logprob_z2_max": float("-inf"),
    }

    def _maybe_replay_reinforce_log_probs(x_tokens, group_reinforce, group_replay):
        nonlocal reinforce_sequence_replay_applied
        nonlocal reinforce_sequence_replay_eval_start
        nonlocal reinforce_sequence_replay_action_steps
        nonlocal reinforce_decomp_count
        if not policy_collect_reinforce_replay:
            return group_reinforce.get("log_probs", None) if isinstance(group_reinforce, dict) else None
        if not callable(reinforce_sequence_replay_fn):
            raise RuntimeError("reinforce sequence replay was requested but policy_step_fn has no replay helper")
        if x_tokens is None:
            raise RuntimeError("reinforce sequence replay requires collected rollout x tokens")
        if not isinstance(group_replay, dict):
            raise RuntimeError("reinforce sequence replay requested but rollout did not record replay tensors")
        reward_in = group_replay.get("reward_in", None)
        sampled_action = group_replay.get("sampled_action", None)
        action_std = group_replay.get("action_std", None)
        action_mask = group_replay.get("action_mask", None)
        eval_start = int(group_replay.get("eval_start", 0) or 0)
        full_length = int(group_replay.get("full_length", int(x_tokens.shape[0])) or int(x_tokens.shape[0]))
        if not all(torch.is_tensor(t) for t in (reward_in, sampled_action, action_std, action_mask)):
            raise RuntimeError("reinforce sequence replay tensors are incomplete")
        action_mean_replay = reinforce_sequence_replay_fn(x_tokens, reward_in)
        if int(sampled_action.shape[0]) != int(action_mean_replay.shape[0]):
            action_mean_replay = action_mean_replay[eval_start : eval_start + int(sampled_action.shape[0])]
        replay_action_dim = int(sampled_action.shape[-1])
        if callable(fit_action_dim_fn):
            action_mean_replay = fit_action_dim_fn(action_mean_replay.reshape(-1, int(action_mean_replay.shape[-1])), replay_action_dim)
            action_mean_replay = action_mean_replay.reshape(
                int(sampled_action.shape[0]),
                int(sampled_action.shape[1]),
                replay_action_dim,
            )
        elif int(action_mean_replay.shape[-1]) != replay_action_dim:
            action_mean_replay = action_mean_replay[..., :replay_action_dim]
        action_mean_replay = action_mean_replay.to(dtype=sampled_action.dtype)
        reinforce_sequence_replay_applied = True
        reinforce_sequence_replay_eval_start = int(eval_start)
        reinforce_sequence_replay_action_steps = int(sampled_action.shape[0])
        suffix_log_probs = prior._gaussian_log_prob(
            sampled_action,
            action_mean_replay,
            action_std,
            mask=action_mask,
        ).to(dtype=torch.float32)
        decomp_stats = prior._gaussian_log_prob_decomposition_stats(
            sampled_action,
            action_mean_replay,
            action_std,
            mask=action_mask,
        )
        group_weight = float(max(1, int(sampled_action.shape[1])))
        reinforce_decomp_count += group_weight
        for key in reinforce_decomp_accum:
            reinforce_decomp_accum[key] += float(torch.as_tensor(decomp_stats[key]).detach().cpu()) * group_weight
        for key in reinforce_decomp_min:
            reinforce_decomp_min[key] = min(
                reinforce_decomp_min[key],
                float(torch.as_tensor(decomp_stats[key]).detach().cpu()),
            )
        for key in reinforce_decomp_max:
            reinforce_decomp_max[key] = max(
                reinforce_decomp_max[key],
                float(torch.as_tensor(decomp_stats[key]).detach().cpu()),
            )
        if int(sampled_action.shape[0]) == int(full_length) and eval_start == 0:
            return suffix_log_probs
        full_log_probs = torch.zeros(
            (full_length, int(suffix_log_probs.shape[1])),
            device=suffix_log_probs.device,
            dtype=suffix_log_probs.dtype,
        )
        full_log_probs[eval_start : eval_start + int(suffix_log_probs.shape[0])] = suffix_log_probs
        return full_log_probs

    if policy_collect_reinforce_replay and backend != "torch_vectorized":
        raise RuntimeError("reinforce sequence replay currently requires torch_vectorized rollout backend")

    if backend == "torch_vectorized":
        effective_grouping_mode = str(grouping_mode)
        if strict_rng_match:
            effective_grouping_mode = "structure"
        if policy_collect_reinforce_replay and effective_grouping_mode != "family":
            raise RuntimeError(
                "reinforce sequence replay currently requires torch_vectorized family grouping without strict RNG remapping"
            )
        grouped = {}
        for b, h in enumerate(h_list):
            sig = prior._environment_group_signature(h, effective_grouping_mode)
            grouped.setdefault(sig, []).append((b, h))
        if alpha_grad_outer_merge_state["enabled"]:
            alpha_grad_outer_merge_state["expected_group_count"] = int(len(grouped))
        rollout_profile_acc = None
        rollout_terminal_acc = None

        for group in grouped.values():
            group_indices = [idx for idx, _ in group]
            group_h_list = [h for _, h in group]
            group_env_seeds = [env_seeds[idx] for idx in group_indices] if env_seeds is not None else None
            group_rollout_seeds = [rollout_seeds[idx] for idx in group_indices] if rollout_seeds is not None else None
            group_tbptt_reward_sink = make_alpha_grad_tbptt_group_sink(group_indices)
            if effective_grouping_mode == "family":
                x_group, y_group, infos_group = prior._rollout_family_group_vectorized_with_policy(
                    h_list=group_h_list,
                    policy_step_fn=policy_step_fn,
                    n_samples=n_samples,
                    num_features=num_features,
                    single_eval_pos=single_eval_pos,
                    device=device,
                    collect_x=collect_x,
                    collect_runtime_info=collect_runtime_info,
                    env_rng_seeds=group_env_seeds,
                    rollout_rng_seeds=group_rollout_seeds,
                    tbptt_window=tbptt_window,
                    tbptt_reward_sink=group_tbptt_reward_sink,
                    tbptt_reward_sink_supports_aux=tbptt_reward_sink_supports_aux,
                    store_rewards=store_rewards,
                    policy_objective_kind=policy_objective_kind,
                    _policy_collect_log_probs=policy_collect_log_probs,
                    _policy_collect_action_trace=policy_collect_action_trace,
                    _policy_collect_reinforce_replay=policy_collect_reinforce_replay,
                    _policy_detach_action_in_env=policy_detach_action_in_env,
                    _policy_disable_log_probs=policy_disable_log_probs,
                    _policy_force_no_grad=policy_force_no_grad,
                )
            else:
                env_batch = prior._sample_environment_batch(
                    h_list=group_h_list,
                    device=device,
                    rng_seeds=group_env_seeds,
                )
                x_group, y_group, infos_group = prior._rollout_distinct_envs_vectorized_with_policy(
                    env=env_batch,
                    policy_step_fn=policy_step_fn,
                    batch_size=len(group_indices),
                    n_samples=n_samples,
                    num_features=num_features,
                    single_eval_pos=single_eval_pos,
                    device=device,
                    collect_x=collect_x,
                    collect_runtime_info=collect_runtime_info,
                    rng_seeds=group_rollout_seeds,
                    tbptt_window=tbptt_window,
                    tbptt_reward_sink=group_tbptt_reward_sink,
                    tbptt_reward_sink_supports_aux=tbptt_reward_sink_supports_aux,
                    store_rewards=store_rewards,
                    policy_objective_kind=policy_objective_kind,
                    _policy_collect_log_probs=policy_collect_log_probs,
                    _policy_collect_action_trace=policy_collect_action_trace,
                    _policy_detach_action_in_env=policy_detach_action_in_env,
                    _policy_disable_log_probs=policy_disable_log_probs,
                    _policy_force_no_grad=policy_force_no_grad,
                )
            if collect_x:
                x[:, group_indices] = x_group
            if store_rewards:
                rewards[:, group_indices] = y_group
            group_reinforce = prior.last_rollout_reinforce
            group_replay = getattr(prior, "last_rollout_reinforce_replay", None)
            if policy_collect_reinforce_replay and isinstance(group_replay, dict):
                reward_in_group = group_replay.get("reward_in", None)
                sampled_action_group = group_replay.get("sampled_action", None)
                action_std_group = group_replay.get("action_std", None)
                action_mask_group = group_replay.get("action_mask", None)
                eval_start_group = int(group_replay.get("eval_start", 0) or 0)
                full_length_group = int(group_replay.get("full_length", int(n_samples)) or int(n_samples))
                if all(
                    torch.is_tensor(t)
                    for t in (
                        reward_in_group,
                        sampled_action_group,
                        action_std_group,
                        action_mask_group,
                    )
                ):
                    replay_steps = int(sampled_action_group.shape[0])
                    action_dim_group = int(sampled_action_group.shape[-1])
                    if reinforce_replay_payload is None:
                        reinforce_replay_payload = {
                            "reward_in": torch.empty(
                                (full_length_group, batch_size),
                                device=reward_in_group.device,
                                dtype=reward_in_group.dtype,
                            ),
                            "sampled_action": torch.empty(
                                (replay_steps, batch_size, action_dim_group),
                                device=sampled_action_group.device,
                                dtype=sampled_action_group.dtype,
                            ),
                            "action_std": torch.empty(
                                (replay_steps, batch_size),
                                device=action_std_group.device,
                                dtype=action_std_group.dtype,
                            ),
                            "action_mask": torch.empty(
                                (batch_size, int(action_mask_group.shape[-1])),
                                device=action_mask_group.device,
                                dtype=action_mask_group.dtype,
                            ),
                            "eval_start": int(eval_start_group),
                            "full_length": int(full_length_group),
                        }
                    else:
                        if (
                            int(reinforce_replay_payload["eval_start"]) != int(eval_start_group)
                            or int(reinforce_replay_payload["full_length"]) != int(full_length_group)
                            or tuple(reinforce_replay_payload["sampled_action"].shape[:1] + reinforce_replay_payload["sampled_action"].shape[2:])
                            != (replay_steps, action_dim_group)
                        ):
                            raise RuntimeError("reinforce replay payload shapes were inconsistent across rollout groups")
                    reinforce_replay_payload["reward_in"][:, group_indices] = reward_in_group
                    reinforce_replay_payload["sampled_action"][:, group_indices] = sampled_action_group
                    reinforce_replay_payload["action_std"][:, group_indices] = action_std_group
                    reinforce_replay_payload["action_mask"][group_indices] = action_mask_group
            group_log_probs = None
            if policy_collect_reinforce_replay and (not policy_defer_reinforce_replay):
                group_log_probs = _maybe_replay_reinforce_log_probs(
                    x_group,
                    group_reinforce,
                    group_replay,
                )
            elif isinstance(group_reinforce, dict) and torch.is_tensor(group_reinforce.get("log_probs", None)):
                group_log_probs = group_reinforce["log_probs"]
            if torch.is_tensor(group_log_probs):
                if reinforce_log_probs is None:
                    reinforce_log_probs = torch.empty(
                        (int(group_log_probs.shape[0]), batch_size),
                        device=group_log_probs.device,
                        dtype=group_log_probs.dtype,
                    )
                reinforce_log_probs[:, group_indices] = group_log_probs
            if isinstance(group_reinforce, dict):
                group_log_prob_score = group_reinforce.get("log_prob_score", None)
                if torch.is_tensor(group_log_prob_score):
                    if reinforce_log_prob_scores is None:
                        reinforce_log_prob_scores = torch.empty(
                            (int(group_log_prob_score.shape[0]), batch_size, int(group_log_prob_score.shape[-1])),
                            device=group_log_prob_score.device,
                            dtype=group_log_prob_score.dtype,
                        )
                    reinforce_log_prob_scores[:, group_indices] = group_log_prob_score
            if collect_action_trace:
                group_policy_trace = prior.last_rollout_policy_trace
                if isinstance(group_policy_trace, dict) and torch.is_tensor(group_policy_trace.get("action_mask", None)):
                    group_action_mean = group_policy_trace.get("action_mean", None)
                    group_action_mask = group_policy_trace["action_mask"]
                    if policy_action_mask is None:
                        action_shape = tuple(group_action_mask.shape)
                        policy_action_mask = torch.empty(
                            action_shape[:1] + (batch_size, action_shape[2]),
                            device=device,
                            dtype=torch.bool,
                        )
                    policy_action_mask[:, group_indices] = group_action_mask
                    if torch.is_tensor(group_action_mean) and (not alpha_grad_trace_roots_only):
                        if policy_action_mean is None:
                            action_shape = tuple(group_action_mean.shape)
                            policy_action_mean = torch.empty(
                                action_shape[:1] + (batch_size, action_shape[2]),
                                device=device,
                                dtype=group_action_mean.dtype,
                            )
                        policy_action_mean[:, group_indices] = group_action_mean
                    group_action_roots = group_policy_trace.get("action_mean_roots", None)
                    if isinstance(group_action_roots, tuple):
                        policy_action_group_traces.append(
                            {
                                "indices": tuple(int(i) for i in group_indices),
                                "action_mean_roots": group_action_roots,
                                "action_mask": group_action_mask,
                            }
                        )
                        if not alpha_grad_trace_roots_only:
                            if policy_action_mean_root_steps is None:
                                policy_action_mean_root_steps = [
                                    torch.zeros(
                                        (batch_size, int(root.shape[-1])),
                                        device=root.device,
                                        dtype=root.dtype,
                                    )
                                    for root in group_action_roots
                                ]
                            elif len(policy_action_mean_root_steps) != len(group_action_roots):
                                raise RuntimeError("alpha_grad action roots time dimension mismatch across rollout groups")
                            for t_idx, root in enumerate(group_action_roots):
                                full_root = policy_action_mean_root_steps[t_idx]
                                if tuple(full_root.shape) != (batch_size, int(root.shape[-1])):
                                    raise RuntimeError("alpha_grad action roots width mismatch across rollout groups")
                                full_root[group_indices] = root
            for local_idx, global_idx in enumerate(group_indices):
                infos[global_idx] = infos_group[local_idx]
            rollout_terminal_acc = prior._merge_terminal_count_summary(
                rollout_terminal_acc,
                prior.last_rollout_terminal_stats,
                batch_weight=(float(len(group_indices)) / float(max(1, batch_size))),
                device=device,
                dtype=torch.float32,
            )
            group_profile = prior.last_rollout_profile
            if isinstance(group_profile, dict):
                if rollout_profile_acc is None:
                    rollout_profile_acc = _new_rollout_profile_accumulator(n_samples=n_samples, batch_size=batch_size)
                rollout_profile_acc["policy_cuda_ms"] += float(group_profile.get("policy_cuda_ms", 0.0))
                rollout_profile_acc["transition_cuda_ms"] += float(group_profile.get("transition_cuda_ms", 0.0))
                rollout_profile_acc["policy_wall_ms"] += float(group_profile.get("policy_wall_ms", 0.0))
                rollout_profile_acc["transition_wall_ms"] += float(group_profile.get("transition_wall_ms", 0.0))
                rollout_profile_acc["transition_y_wall_ms"] += float(group_profile.get("transition_y_wall_ms", 0.0))
                rollout_profile_acc["transition_x_wall_ms"] += float(group_profile.get("transition_x_wall_ms", 0.0))
                rollout_profile_acc["transition_group_wall_ms"] += float(group_profile.get("transition_group_wall_ms", 0.0))
                rollout_profile_acc["transition_group_launch_wall_ms"] += float(
                    group_profile.get("transition_group_launch_wall_ms", 0.0)
                )
                rollout_profile_acc["transition_group_sync_wall_ms"] += float(
                    group_profile.get("transition_group_sync_wall_ms", 0.0)
                )
                rollout_profile_acc["transition_env_pack_wall_ms"] += float(
                    group_profile.get("transition_env_pack_wall_ms", 0.0)
                )
                rollout_profile_acc["transition_state_update_wall_ms"] += float(
                    group_profile.get("transition_state_update_wall_ms", 0.0)
                )
                rollout_profile_acc["transition_noise_wall_ms"] += float(
                    group_profile.get("transition_noise_wall_ms", 0.0)
                )
                rollout_profile_acc["transition_fused_wall_ms"] += float(
                    group_profile.get("transition_fused_wall_ms", 0.0)
                )
                rollout_profile_acc["transition_fused_launch_wall_ms"] += float(
                    group_profile.get("transition_fused_launch_wall_ms", 0.0)
                )
                rollout_profile_acc["transition_gp_first_projection_wall_ms"] += float(
                    group_profile.get("transition_gp_first_projection_wall_ms", 0.0)
                )
                rollout_profile_acc["transition_gp_second_projection_wall_ms"] += float(
                    group_profile.get("transition_gp_second_projection_wall_ms", 0.0)
                )
                rollout_profile_acc["transition_gp_projection_call_count"] += int(
                    group_profile.get("transition_gp_projection_call_count", 0) or 0
                )
                rollout_profile_acc["transition_gp_rff_fused_call_count"] += int(
                    group_profile.get("transition_gp_rff_fused_call_count", 0) or 0
                )
                rollout_profile_acc["transition_gp_profile_group_count"] += int(
                    group_profile.get("transition_gp_profile_group_count", 0) or 0
                )
                rollout_profile_acc["transition_gp_profile_sync_group_count"] += int(
                    group_profile.get("transition_gp_profile_sync_group_count", 0) or 0
                )
                rollout_profile_acc["transition_gp_shared_total_wall_ms"] += float(
                    group_profile.get("transition_gp_shared_total_wall_ms", 0.0) or 0.0
                )
                rollout_profile_acc["transition_gp_shared_core_wall_ms"] += float(
                    group_profile.get("transition_gp_shared_core_wall_ms", 0.0) or 0.0
                )
                rollout_profile_acc["transition_gp_shared_noise_wall_ms"] += float(
                    group_profile.get("transition_gp_shared_noise_wall_ms", 0.0) or 0.0
                )
                rollout_profile_acc["transition_gp_shared_checkpoint_wall_ms"] += float(
                    group_profile.get("transition_gp_shared_checkpoint_wall_ms", 0.0) or 0.0
                )
                rollout_profile_acc["transition_gp_shared_post_wall_ms"] += float(
                    group_profile.get("transition_gp_shared_post_wall_ms", 0.0) or 0.0
                )
                rollout_profile_acc["transition_gp_shared_call_count"] += int(
                    group_profile.get("transition_gp_shared_call_count", 0) or 0
                )
                rollout_profile_acc["transition_packed_env_input_group_count"] += int(
                    group_profile.get("transition_packed_env_input_group_count", 0) or 0
                )
                rollout_profile_acc["transition_packed_env_input_call_count"] += int(
                    group_profile.get("transition_packed_env_input_call_count", 0) or 0
                )
                rollout_profile_acc["transition_only_build_group_count"] += int(
                    group_profile.get("transition_only_build_group_count", 0) or 0
                )
                rollout_profile_acc["transition_only_skipped_generator_count"] += int(
                    group_profile.get("transition_only_skipped_generator_count", 0) or 0
                )
                rollout_profile_acc["transition_setup_wall_ms"] += float(
                    group_profile.get("transition_setup_wall_ms", 0.0) or 0.0
                )
                rollout_profile_acc["transition_family_build_wall_ms"] += float(
                    group_profile.get("transition_family_build_wall_ms", 0.0) or 0.0
                )
                rollout_profile_acc["transition_generator_build_wall_ms"] += float(
                    group_profile.get("transition_generator_build_wall_ms", 0.0) or 0.0
                )
                rollout_profile_acc["transition_gp_shared_build_wall_ms"] += float(
                    group_profile.get("transition_gp_shared_build_wall_ms", 0.0) or 0.0
                )
                rollout_profile_acc["transition_fused_call_count"] += int(group_profile.get("transition_fused_call_count", 0))
                rollout_profile_acc["transition_fused_group_count"] += int(group_profile.get("transition_fused_group_count", 0))
                rollout_profile_acc["transition_fused_enabled"] = int(
                    max(
                        int(rollout_profile_acc.get("transition_fused_enabled", 0) or 0),
                        int(group_profile.get("transition_fused_enabled", 0) or 0),
                    )
                )
                rollout_profile_acc["transition_checkpoint_enabled"] = int(
                    max(
                        int(rollout_profile_acc.get("transition_checkpoint_enabled", 0) or 0),
                        int(group_profile.get("transition_checkpoint_enabled", 0) or 0),
                    )
                )
                rollout_profile_acc["transition_checkpoint_call_count"] += int(
                    group_profile.get("transition_checkpoint_call_count", 0) or 0
                )
                rollout_profile_acc["transition_group_count"] += int(group_profile.get("transition_group_count", 0))
                rollout_profile_acc["transition_family_group_count"] += int(
                    group_profile.get("transition_family_group_count", 0) or 0
                )
                rollout_profile_acc["transition_inner_grouping_structure_enabled"] = int(
                    max(
                        int(rollout_profile_acc.get("transition_inner_grouping_structure_enabled", 0) or 0),
                        int(group_profile.get("transition_inner_grouping_structure_enabled", 0) or 0),
                    )
                )
                rollout_profile_acc["transition_inner_min_bucket"] = int(
                    max(
                        int(rollout_profile_acc.get("transition_inner_min_bucket", 0) or 0),
                        int(group_profile.get("transition_inner_min_bucket", 0) or 0),
                    )
                )
                rollout_profile_acc["transition_bucket_max_batch"] = int(
                    max(
                        int(rollout_profile_acc.get("transition_bucket_max_batch", 0) or 0),
                        int(group_profile.get("transition_bucket_max_batch", 0) or 0),
                    )
                )
                rollout_profile_acc["transition_work_actual_est"] += float(
                    group_profile.get("transition_work_actual_est", 0.0) or 0.0
                )
                rollout_profile_acc["transition_work_padded_est"] += float(
                    group_profile.get("transition_work_padded_est", 0.0) or 0.0
                )
                rollout_profile_acc["transition_async_enabled"] = int(
                    max(
                        int(rollout_profile_acc.get("transition_async_enabled", 0) or 0),
                        int(group_profile.get("transition_async_enabled", 0) or 0),
                    )
                )
                group_noise_mode = group_profile.get("noise_mode", None)
                if group_noise_mode is not None:
                    group_noise_mode = str(group_noise_mode)
                    current_noise_mode = rollout_profile_acc.get("noise_mode", None)
                    if current_noise_mode is None:
                        rollout_profile_acc["noise_mode"] = group_noise_mode
                    elif str(current_noise_mode) != group_noise_mode:
                        rollout_profile_acc["noise_mode"] = "mixed"
                try:
                    group_noise_block_size = int(group_profile.get("noise_block_size", 0) or 0)
                except Exception:
                    group_noise_block_size = 0
                if group_noise_block_size > int(rollout_profile_acc.get("noise_block_size", 0) or 0):
                    rollout_profile_acc["noise_block_size"] = group_noise_block_size
                for key in (
                    "env_count",
                    "strict_joint_transition_count",
                    "reference_semantics_count",
                    "exact_scm_count",
                    "exact_gp_count",
                    "fixed_gp_count",
                    "legacy_scm_count",
                    "legacy_gp_count",
                ):
                    rollout_profile_acc[key] += int(group_profile.get(key, 0) or 0)
        if rollout_profile_acc is not None:
            rollout_profile_acc["transition_bucket_mean_batch"] = float(
                batch_size / max(1, int(rollout_profile_acc.get("transition_group_count", 0) or 0))
            )
            rollout_profile_acc["transition_work_fill_ratio"] = float(
                float(rollout_profile_acc.get("transition_work_actual_est", 0.0) or 0.0)
                / max(1e-9, float(rollout_profile_acc.get("transition_work_padded_est", 0.0) or 0.0))
            )
            rollout_profile_acc = prior._finalize_env_semantics_summary(rollout_profile_acc)
        prior.last_rollout_profile = rollout_profile_acc
        prior.last_rollout_env_semantics = _project_rollout_env_semantics(rollout_profile_acc)
        prior.last_rollout_reinforce_replay = reinforce_replay_payload
        reinforce_meta = {}
        if reinforce_decomp_count > 0.0:
            reinforce_meta = {
                "reinforce_action_dim_mean": torch.as_tensor(
                    reinforce_decomp_accum["reinforce_action_dim_mean"] / reinforce_decomp_count,
                    device=device,
                    dtype=torch.float32,
                ),
                "reinforce_action_std_mean": torch.as_tensor(
                    reinforce_decomp_accum["reinforce_action_std_mean"] / reinforce_decomp_count,
                    device=device,
                    dtype=torch.float32,
                ),
                "reinforce_logprob_log_std_mean": torch.as_tensor(
                    reinforce_decomp_accum["reinforce_logprob_log_std_mean"] / reinforce_decomp_count,
                    device=device,
                    dtype=torch.float32,
                ),
                "reinforce_logprob_z2_mean": torch.as_tensor(
                    reinforce_decomp_accum["reinforce_logprob_z2_mean"] / reinforce_decomp_count,
                    device=device,
                    dtype=torch.float32,
                ),
            }
            for key, value in reinforce_decomp_min.items():
                reinforce_meta[key] = torch.as_tensor(value, device=device, dtype=torch.float32)
            for key, value in reinforce_decomp_max.items():
                reinforce_meta[key] = torch.as_tensor(value, device=device, dtype=torch.float32)
        prior.last_rollout_reinforce = (
            {
                "log_probs": reinforce_log_probs,
                "log_prob_score": reinforce_log_prob_scores,
                "sequence_replay_applied": bool(reinforce_sequence_replay_applied),
                "sequence_replay_eval_start": (
                    None
                    if reinforce_sequence_replay_eval_start is None
                    else int(reinforce_sequence_replay_eval_start)
                ),
                "sequence_replay_action_steps": (
                    None
                    if reinforce_sequence_replay_action_steps is None
                    else int(reinforce_sequence_replay_action_steps)
                ),
                **reinforce_meta,
            }
            if reinforce_log_probs is not None
            else None
        )
        if policy_action_mean_root_steps is not None:
            policy_action_mean_roots = tuple(policy_action_mean_root_steps)
        prior.last_rollout_policy_trace = (
            {
                "action_mean": policy_action_mean,
                "action_mean_roots": policy_action_mean_roots,
                "action_mask": policy_action_mask,
                "group_traces": tuple(policy_action_group_traces) if policy_action_group_traces else None,
            }
            if (
                collect_action_trace
                and policy_action_mask is not None
                and (
                    (policy_action_mean is not None)
                    or (policy_action_mean_roots is not None)
                    or bool(policy_action_group_traces)
                )
            )
            else None
        )
        prior.last_rollout_terminal_stats = prior._finalize_terminal_count_summary(
            rollout_terminal_acc,
            device=device,
            dtype=torch.float32,
        )
        if alpha_grad_outer_merge_state["enabled"] and alpha_grad_outer_merge_state["window_buckets"]:
            raise RuntimeError("alpha_grad TBPTT outer merge finished with incomplete window buckets")
        prior.last_runtime_info = infos if collect_runtime_info else [None] * batch_size
        return {
            "x": x,
            "rewards": rewards,
            "info": prior.last_runtime_info,
            "single_eval_pos": single_eval_pos,
            "rollout_profile": prior.last_rollout_profile,
            "reinforce": prior.last_rollout_reinforce,
            "policy_trace": prior.last_rollout_policy_trace,
            "terminal_stats": prior.last_rollout_terminal_stats,
        }

    rollout_env_semantics_acc = None
    rollout_terminal_acc = None
    if alpha_grad_outer_merge_state["enabled"]:
        alpha_grad_outer_merge_state["expected_group_count"] = int(batch_size)
    for b, h in enumerate(h_list):
        env = prior._sample_environment(
            h=h,
            device=device,
            rng_seed=(env_seeds[b] if env_seeds is not None else None),
        )
        x_one, y_one, info = prior._rollout_single(
            env=env,
            n_samples=n_samples,
            num_features=num_features,
            single_eval_pos=single_eval_pos,
            device=device,
            policy_step_fn=policy_step_fn,
            collect_x=collect_x,
            collect_runtime_info=collect_runtime_info,
            rng_seed=(rollout_seeds[b] if rollout_seeds is not None else None),
            tbptt_window=tbptt_window,
            tbptt_reward_sink=make_alpha_grad_tbptt_group_sink((b,)),
            tbptt_reward_sink_supports_aux=tbptt_reward_sink_supports_aux,
            store_rewards=store_rewards,
            policy_objective_kind=policy_objective_kind,
            _policy_collect_log_probs=policy_collect_log_probs,
            _policy_collect_action_trace=policy_collect_action_trace,
            _policy_detach_action_in_env=policy_detach_action_in_env,
        )
        if collect_x:
            x[:, b] = x_one
        if store_rewards:
            rewards[:, b] = y_one
        if reinforce_log_probs is not None:
            rollout_reinforce_one = prior.last_rollout_reinforce
            if isinstance(rollout_reinforce_one, dict) and torch.is_tensor(rollout_reinforce_one.get("log_probs", None)):
                reinforce_log_probs[:, b] = rollout_reinforce_one["log_probs"].reshape(n_samples)
                rollout_log_prob_score_one = rollout_reinforce_one.get("log_prob_score", None)
                if torch.is_tensor(rollout_log_prob_score_one):
                    if reinforce_log_prob_scores is None:
                        reinforce_log_prob_scores = torch.empty(
                            (
                                int(rollout_log_prob_score_one.shape[0]),
                                batch_size,
                                int(rollout_log_prob_score_one.shape[-1]),
                            ),
                            device=rollout_log_prob_score_one.device,
                            dtype=rollout_log_prob_score_one.dtype,
                        )
                    reinforce_log_prob_scores[:, b:b + 1] = rollout_log_prob_score_one
        if collect_action_trace:
            rollout_policy_one = prior.last_rollout_policy_trace
            if (
                isinstance(rollout_policy_one, dict)
                and torch.is_tensor(rollout_policy_one.get("action_mean", None))
                and torch.is_tensor(rollout_policy_one.get("action_mask", None))
            ):
                if policy_action_mean is None:
                    action_shape = tuple(rollout_policy_one["action_mean"].shape)
                    policy_action_mean = torch.empty(
                        action_shape[:1] + (batch_size, action_shape[2]),
                        device=device,
                        dtype=rollout_policy_one["action_mean"].dtype,
                    )
                    policy_action_mask = torch.empty(
                        action_shape[:1] + (batch_size, action_shape[2]),
                        device=device,
                        dtype=torch.bool,
                    )
                policy_action_mean[:, b:b + 1] = rollout_policy_one["action_mean"]
                policy_action_mask[:, b:b + 1] = rollout_policy_one["action_mask"]
                rollout_action_roots = rollout_policy_one.get("action_mean_roots", None)
                if isinstance(rollout_action_roots, tuple):
                    policy_action_group_traces.append(
                        {
                            "indices": (int(b),),
                            "action_mean_roots": rollout_action_roots,
                            "action_mask": rollout_policy_one["action_mask"],
                        }
                    )
                    if policy_action_mean_root_steps is None:
                        policy_action_mean_root_steps = [
                            torch.zeros(
                                (batch_size, int(root.shape[-1])),
                                device=root.device,
                                dtype=root.dtype,
                            )
                            for root in rollout_action_roots
                        ]
                    elif len(policy_action_mean_root_steps) != len(rollout_action_roots):
                        raise RuntimeError("alpha_grad action roots time dimension mismatch across serial rollouts")
                    for t_idx, root in enumerate(rollout_action_roots):
                        full_root = policy_action_mean_root_steps[t_idx]
                        if tuple(full_root.shape) != (batch_size, int(root.shape[-1])):
                            raise RuntimeError("alpha_grad action roots width mismatch across serial rollouts")
                        full_root[b] = root
        infos[b] = info
        rollout_terminal_acc = prior._merge_terminal_count_summary(
            rollout_terminal_acc,
            prior.last_rollout_terminal_stats,
            batch_weight=(1.0 / float(max(1, batch_size))),
            device=device,
            dtype=torch.float32,
        )
        rollout_env_semantics_acc = prior._merge_env_semantics_summary(
            rollout_env_semantics_acc,
            prior._summarize_env_semantics(env, 1),
        )
    prior.last_runtime_info = infos if collect_runtime_info else [None] * batch_size
    prior.last_rollout_profile = prior._finalize_env_semantics_summary(rollout_env_semantics_acc)
    if isinstance(prior.last_rollout_profile, dict):
        prior.last_rollout_profile["steps"] = int(n_samples)
        prior.last_rollout_profile["batch_size"] = int(batch_size)
    prior.last_rollout_env_semantics = _project_rollout_env_semantics(prior.last_rollout_profile)
    prior.last_rollout_reinforce_replay = reinforce_replay_payload
    reinforce_meta = {}
    if reinforce_decomp_count > 0.0:
        reinforce_meta = {
            "reinforce_action_dim_mean": torch.as_tensor(
                reinforce_decomp_accum["reinforce_action_dim_mean"] / reinforce_decomp_count,
                device=device,
                dtype=torch.float32,
            ),
            "reinforce_action_std_mean": torch.as_tensor(
                reinforce_decomp_accum["reinforce_action_std_mean"] / reinforce_decomp_count,
                device=device,
                dtype=torch.float32,
            ),
            "reinforce_logprob_log_std_mean": torch.as_tensor(
                reinforce_decomp_accum["reinforce_logprob_log_std_mean"] / reinforce_decomp_count,
                device=device,
                dtype=torch.float32,
            ),
            "reinforce_logprob_z2_mean": torch.as_tensor(
                reinforce_decomp_accum["reinforce_logprob_z2_mean"] / reinforce_decomp_count,
                device=device,
                dtype=torch.float32,
            ),
        }
        for key, value in reinforce_decomp_min.items():
            reinforce_meta[key] = torch.as_tensor(value, device=device, dtype=torch.float32)
        for key, value in reinforce_decomp_max.items():
            reinforce_meta[key] = torch.as_tensor(value, device=device, dtype=torch.float32)
    prior.last_rollout_reinforce = (
        {
            "log_probs": reinforce_log_probs,
            "log_prob_score": reinforce_log_prob_scores,
            "sequence_replay_applied": bool(reinforce_sequence_replay_applied),
            "sequence_replay_eval_start": (
                None
                if reinforce_sequence_replay_eval_start is None
                else int(reinforce_sequence_replay_eval_start)
            ),
            "sequence_replay_action_steps": (
                None
                if reinforce_sequence_replay_action_steps is None
                else int(reinforce_sequence_replay_action_steps)
            ),
            **reinforce_meta,
        }
        if reinforce_log_probs is not None
        else None
    )
    if policy_action_mean_root_steps is not None:
        policy_action_mean_roots = tuple(policy_action_mean_root_steps)
    prior.last_rollout_policy_trace = (
        {
            "action_mean": policy_action_mean,
            "action_mean_roots": policy_action_mean_roots,
            "action_mask": policy_action_mask,
            "group_traces": tuple(policy_action_group_traces) if policy_action_group_traces else None,
        }
        if (
            collect_action_trace
            and policy_action_mask is not None
            and (
                (policy_action_mean is not None)
                or (policy_action_mean_roots is not None)
                or bool(policy_action_group_traces)
            )
        )
        else None
    )
    prior.last_rollout_terminal_stats = prior._finalize_terminal_count_summary(
        rollout_terminal_acc,
        device=device,
        dtype=torch.float32,
    )
    if alpha_grad_outer_merge_state["enabled"] and alpha_grad_outer_merge_state["window_buckets"]:
        raise RuntimeError("alpha_grad TBPTT outer merge finished with incomplete serial window buckets")
    return {
        "x": x,
        "rewards": rewards,
        "info": prior.last_runtime_info,
        "single_eval_pos": single_eval_pos,
        "rollout_profile": prior.last_rollout_profile,
        "reinforce": prior.last_rollout_reinforce,
        "policy_trace": prior.last_rollout_policy_trace,
        "terminal_stats": prior.last_rollout_terminal_stats,
    }
