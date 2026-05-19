import os
from copy import deepcopy

from ticl.rlpfn_constants import TERMINAL_RESET_COUNT_TARGET_MAX


RLPFN_MAINTAINED_ENV_DEFAULTS = {
    "family": {"distribution": "meta_choice", "choice_values": ["scm"]},
    "obs_slot_dim": 400,
    "action_slot_dim": 30,
    "constrained_dim_sampling_enabled": True,
    "constrained_dim_sampling_policy": "gym_state_obs_coupled",
    "constrained_dim_sampling_total_budget": 400,
    "strict_joint_transition_enabled": True,
    "reference_scm_zero_pad_inactive_init_enabled": True,
    "state_input_scale_enabled": False,
    "state_input_scale": 1.0,
    "scm_standard_linear_init_enabled": True,
    "state_full_rms_enabled": True,
    "state_full_rms_target": 1.0,
    "ctrl_reward_weight": {"distribution": "log_uniform", "min": 1e-3, "max": 3e-1},
    "ctrl_reward_enable_prob": 0.7,
    "survival_reward_weight": 0.0,
    "survival_reward_enable_prob": 0.0,
    "reinforce_reward_transform": "none",
    "reinforce_reward_rms_eps": 1e-6,
    "reinforce_reward_tanh_c": 10.0,
    "reinforce_reward_tanh_bound": {"distribution": "uniform", "min": 0.0, "max": 2.0},
    "reinforce_action_transform": "none",
    "reinforce_action_rms_eps": 1e-6,
    "reinforce_action_clip_bound": 5.0,
    "reinforce_normalize_advantages": True,
    "reinforce_scale_advantages_by_suffix_episode_count": True,
    "reinforce_advantage_norm_eps": 1e-6,
    "reinforce_advantage_norm_clip": 10.0,
    "policy_gradient_weight": 0.4,
    "reinforce_aux_enabled": False,
    "reinforce_aux_backbone_query_pass_enabled": False,
    "normalized_q_value_weight": 0.0,
    "next_state_flow_matching_weight": 0.0,
    "next_state_flow_head_type": "rwkv_two_layer",
    "reinforce_sequence_replay_enabled": True,
    "reinforce_sequence_replay_share_context_forward": False,
    "first_policy_gradient_state_grad_clip_norm": 4.0,
    "first_policy_gradient_action_grad_clip_value": 0.0,
    "first_policy_gradient_action_grad_clip_norm": 1.0,
    "alpha_grad_local_coordinate_enabled": True,
    "alpha_grad_unit_grad_enabled": True,
    "alpha_grad_unit_grad_delta": 1e-6,
    "pg_one_hop_replay_enabled": True,
    "alpha_grad_one_hop_replay_enabled": True,
    "pg_markov_adjacent_replay_enabled": True,
    "pg_markov_adjacent_replay_sample_prob": 0.5,
    "pg_replay_window_depth": 1,
    "action_noise_train_std": 0.0,
    "action_noise_eval_std": 0.0,
    "reward_dropout_enabled": True,
    "reward_dropout_randomize": True,
    "reward_dropout_ratio_min": 0.1,
    "reward_dropout_ratio_max": 1.0,
    "reward_dropout_impute_zero": True,
    "terminal_reset_enabled": True,
    "terminal_reset_count_target": {"distribution": "uniform", "min": 0.0, "max": TERMINAL_RESET_COUNT_TARGET_MAX},
    "terminal_bonus_tanh_c": 10.0,
    "terminal_bonus_scale_min": 1.0,
    "terminal_bonus_scale_max": 2.0,
    "batch_parallel_backend": "torch_vectorized",
    "batch_shared_environment": False,
    "batch_vectorized_grouping": "family",
    "reward_topology_conditioned_sampling_enabled": True,
    "reward_state_input_gain_fraction_conditioned_min": 0.7,
    "reward_action_input_gain_fraction_conditioned_min": 0.06,
    "reward_state_to_action_gain_ratio_conditioned_max": 10.0,
    "reward_topology_conditioned_sampling_max_attempts": 4096,
}


def resolve_rlpfn_token_layout(env_cfg, *, num_features=None):
    obs_slot_dim = int(env_cfg.get("obs_slot_dim", 400))
    action_slot_dim = int(env_cfg.get("action_slot_dim", 30))
    phase_token_enabled = True
    terminal_token_enabled = bool(env_cfg.get("terminal_reset_enabled", False))
    x_obs_dim = int(obs_slot_dim) + 2 + int(phase_token_enabled) + int(terminal_token_enabled)
    x_action_dim = int(action_slot_dim)
    default_num_features = int(x_obs_dim) + int(x_action_dim)
    resolved_num_features = int(default_num_features if num_features is None else num_features)
    return {
        "obs_slot_dim": int(obs_slot_dim),
        "action_slot_dim": int(action_slot_dim),
        "phase_token_enabled": bool(phase_token_enabled),
        "terminal_token_enabled": bool(terminal_token_enabled),
        "x_obs_dim": int(x_obs_dim),
        "x_action_dim": int(x_action_dim),
        "default_num_features": int(default_num_features),
        "num_features": int(resolved_num_features),
    }


def apply_rlpfn_maintained_path_defaults(config):
    env_cfg = config["prior"]["environment"]
    env_cfg.update(deepcopy(RLPFN_MAINTAINED_ENV_DEFAULTS))
    # The pack-aligned PPO contract trains only actor/critic PPO objectives.
    # Keep aux disabled in the environment contract and again at the runtime
    # bridge so there is no hidden loss-path fork.
    env_cfg["reinforce_aux_enabled"] = False
    env_cfg["normalized_q_value_weight"] = 0.0
    env_cfg["next_state_flow_matching_weight"] = 0.0
    layout = resolve_rlpfn_token_layout(env_cfg)

    config["prior"]["classification"]["num_features_sampler"] = "fixed"
    config["prior"]["classification"]["pad_zeros"] = False
    config["prior"]["classification"]["max_num_classes"] = 0
    config["transformer"]["classification_task"] = False
    config["transformer"]["y_encoder"] = "linear"
    config["transformer"]["x_encoder_type"] = "split_obs_action"
    config["transformer"]["x_obs_dim"] = int(layout["x_obs_dim"])
    config["transformer"]["x_action_dim"] = int(layout["x_action_dim"])
    config["prior"]["num_features"] = int(layout["default_num_features"])
    config["prior"]["n_samples"] = 2048
    config["dataloader"]["batch_size"] = 2048
    config["transformer"]["single_eval_causal"] = True
    config["transformer"]["backbone"] = "rwkv7"
    config["transformer"]["rwkv_sequence_replay_checkpoint"] = True
    config["transformer"]["rwkv_sequence_replay_batch_chunk_size"] = int(
        os.environ.get("TICL_RWKV_SEQUENCE_REPLAY_BATCH_CHUNK_SIZE", "1024")
    )
    config["transformer"]["rwkv_sequence_replay_token_budget"] = int(
        os.environ.get("TICL_RWKV_SEQUENCE_REPLAY_TOKEN_BUDGET", "262144")
    )
    config["optimizer"]["rl_objective"] = "ppo"
    # Keep fit_model rlpfn on the pack-trainer PPO contract by default.
    # These values are intentionally set here rather than requiring every
    # training command to repeat the same semantic alignment flags.
    config["optimizer"]["learning_rate"] = 2e-4
    # Let the official PPO bridge choose the largest RWKV-safe sequence batch
    # from n_steps * replay_chunk_capacity.  A fixed 256-transition batch makes
    # the default 2048x2048 rollout split into 16384 tiny outer batches.
    config["optimizer"]["ppo_batch_size"] = None
    config["optimizer"]["ppo_n_epochs"] = 4
    config["optimizer"]["ppo_gamma"] = 0.98
    config["optimizer"]["ppo_gae_lambda"] = 0.90
    config["optimizer"]["ppo_clip_range"] = 0.2
    config["optimizer"]["ppo_clip_range_vf"] = None
    config["optimizer"]["ppo_normalize_advantage"] = True
    config["optimizer"]["ppo_space_contract"] = "raw"
    config["optimizer"]["ppo_target_kl"] = None
    config["optimizer"]["ppo_deterministic_batch_plan"] = True
    config["optimizer"]["ppo_strict_native_rollout"] = False
    config["optimizer"]["ppo_reset_env_state_at_sep"] = True
    config["optimizer"]["ppo_value_head_impl"] = "vendor_official"
    config["optimizer"]["ppo_value_path_adapter_impl"] = "none"
    config["optimizer"]["ppo_value_head_mlp_hidden_dim"] = 256
    config["optimizer"]["ppo_actor_baseline_mode"] = "learned"
    config["optimizer"]["ppo_vf_coef"] = 0.1
    config["optimizer"]["ppo_trusted_pack_runner_required"] = True
    config["optimizer"]["ppo_pack_prior_milestone"] = "m4_potential_progress_live"
    config["optimizer"]["ppo_pack_prior_mode"] = "sampled_topology"
    config["optimizer"]["ppo_pack_fixed_frozen_h_list_csv"] = None
    config["optimizer"]["ppo_pack_fixed_frozen_h_list_rule"] = None
    config["optimizer"]["ppo_pack_fixed_frozen_h_list_limit"] = None
    config["optimizer"]["ppo_pack_fixed_env_group_across_updates"] = False
    config["optimizer"]["ppo_pack_sb3_reward_normalization_enabled"] = True
    config["optimizer"]["ppo_pack_sb3_observation_normalization_enabled"] = True
    config["optimizer"]["ppo_pack_sb3_observation_normalization_clip"] = 10.0
    config["optimizer"]["ppo_pack_sb3_observation_normalization_epsilon"] = 1e-8
    # The semantic health probe is an explicit diagnostic/sidecar tool.  Keeping
    # it off in default large-scale pretraining avoids materializing extra
    # rollout-wide statistics on top of the canonical PPO pack.
    config["optimizer"]["ppo_pack_semantic_probes_enabled"] = False
    config["optimizer"]["ppo_pack_semantic_probe_sidecar_enabled"] = False
    config["optimizer"]["ppo_pack_semantic_probe_sidecar_every"] = 1
    config["optimizer"]["ppo_pack_checkpoint_every"] = 0
    config["optimizer"]["ppo_pack_checkpoint_include_optimizer"] = False
    config["optimizer"]["ppo_pack_topology_state_gain_min"] = 0.7
    config["optimizer"]["ppo_pack_topology_action_gain_min"] = 0.06
    config["optimizer"]["ppo_pack_topology_state_to_action_ratio_max"] = 10.0
    config["optimizer"]["ppo_pack_topology_max_attempts"] = 4096
    # Pack PPO trains only the actor/critic objective.  The fit_model PPO
    # bridge must not silently add an auxiliary loss.
    config["optimizer"]["ppo_runtime_normalized_q_value_weight_override"] = 0.0
    config["optimizer"]["ppo_runtime_next_state_flow_matching_weight_override"] = 0.0
    return layout


def validate_rlpfn_maintained_path_config(config):
    env_cfg = config["prior"]["environment"]
    layout = resolve_rlpfn_token_layout(env_cfg, num_features=config["prior"].get("num_features", None))
    if config["prior"]["prior_type"] != "environment_only":
        raise ValueError("Maintained RLPFN path requires prior_type=environment_only.")
    if str(config["optimizer"]["rl_objective"]).strip().lower() != "ppo":
        raise ValueError("Maintained RLPFN path requires optimizer.rl_objective=ppo.")
    if str(config["transformer"].get("backbone", "transformer")).strip().lower() != "rwkv7":
        raise ValueError("Maintained RLPFN path requires transformer.backbone=rwkv7.")
    if str(config["transformer"]["x_encoder_type"]).strip().lower() != "split_obs_action":
        raise ValueError("Maintained RLPFN path requires transformer.x_encoder_type=split_obs_action.")
    if not bool(config["transformer"]["single_eval_causal"]):
        raise ValueError("Maintained RLPFN path requires transformer.single_eval_causal=True.")
    if int(config["transformer"]["x_obs_dim"]) != int(layout["x_obs_dim"]):
        raise ValueError("Maintained RLPFN path requires transformer.x_obs_dim to match token layout.")
    if int(config["transformer"]["x_action_dim"]) != int(layout["x_action_dim"]):
        raise ValueError("Maintained RLPFN path requires transformer.x_action_dim to match token layout.")
    if int(config["prior"]["num_features"]) != int(layout["default_num_features"]):
        raise ValueError("Maintained RLPFN path requires prior.num_features to match token layout.")
    if not bool(env_cfg.get("strict_joint_transition_enabled", False)):
        raise ValueError("Maintained RLPFN path requires exact-SCM strict_joint_transition_enabled=True.")
    if not bool(env_cfg.get("reinforce_sequence_replay_enabled", False)):
        raise ValueError("Maintained RLPFN path requires reinforce_sequence_replay_enabled=True.")
    terminal_target = env_cfg.get("terminal_reset_count_target", 0.0)
    terminal_max = None
    if isinstance(terminal_target, dict):
        terminal_max = terminal_target.get("max", None)
    else:
        terminal_max = terminal_target
    try:
        terminal_max_value = float(terminal_max)
    except (TypeError, ValueError):
        terminal_max_value = 0.0
    if terminal_max_value > TERMINAL_RESET_COUNT_TARGET_MAX:
        raise ValueError(
            "Maintained RLPFN path requires terminal_reset_count_target max "
            f"<= {TERMINAL_RESET_COUNT_TARGET_MAX:g}."
        )
    optimizer_cfg = config["optimizer"]
    if not bool(optimizer_cfg.get("ppo_trusted_pack_runner_required", False)):
        raise ValueError("Maintained RLPFN path requires ppo_trusted_pack_runner_required=True.")
    if str(optimizer_cfg.get("ppo_space_contract", "")).strip().lower() != "raw":
        raise ValueError("Maintained RLPFN path requires ppo_space_contract='raw'.")
    if str(optimizer_cfg.get("ppo_value_head_impl", "")).strip().lower() != "vendor_official":
        raise ValueError("Maintained RLPFN path requires ppo_value_head_impl='vendor_official'.")
    if str(optimizer_cfg.get("ppo_value_path_adapter_impl", "")).strip().lower() != "none":
        raise ValueError("Maintained RLPFN path requires ppo_value_path_adapter_impl='none'.")
    if not bool(optimizer_cfg.get("ppo_deterministic_batch_plan", False)):
        raise ValueError("Maintained RLPFN path requires ppo_deterministic_batch_plan=True.")
    prior_mode = str(optimizer_cfg.get("ppo_pack_prior_mode", "")).strip().lower()
    if prior_mode != "sampled_topology":
        raise ValueError(
            "Maintained RLPFN fit_model training requires live ppo_pack_prior_mode='sampled_topology'. "
            "Fixed frozen h-list runners are audit-only and must not be used for M2-M4 training."
        )
    if str(optimizer_cfg.get("ppo_pack_fixed_frozen_h_list_csv") or "").strip():
        raise ValueError("Maintained RLPFN fit_model training must not set ppo_pack_fixed_frozen_h_list_csv.")
    prior_milestone = str(optimizer_cfg.get("ppo_pack_prior_milestone", "sampled_topology")).strip().lower()
    prior_milestone = {
        "m4": "m4_potential_progress_live",
        "milestone4": "m4_potential_progress_live",
        "m4_potential_progress": "m4_potential_progress_live",
        "potential_progress": "m4_potential_progress_live",
        "potential_progress_live": "m4_potential_progress_live",
    }.get(prior_milestone, prior_milestone)
    if prior_milestone != "m4_potential_progress_live":
        raise ValueError(
            "Maintained RLPFN fit_model training requires ppo_pack_prior_milestone="
            "'m4_potential_progress_live'. Retired balanced/gated runner milestones are not supported."
        )
    if bool(optimizer_cfg.get("ppo_pack_fixed_env_group_across_updates", True)):
        raise ValueError("Maintained RLPFN path requires a fresh sampled topology batch each PPO update.")
    if abs(float(optimizer_cfg.get("ppo_vf_coef", 0.0)) - 0.1) > 1e-12:
        raise ValueError("Maintained RLPFN path requires ppo_vf_coef=0.1.")
    return layout
