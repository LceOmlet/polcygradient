from copy import deepcopy


RLPFN_MAINTAINED_ENV_DEFAULTS = {
    "family": {"distribution": "meta_choice", "choice_values": ["scm"]},
    "obs_slot_dim": 400,
    "action_slot_dim": 30,
    "constrained_dim_sampling_enabled": True,
    "constrained_dim_sampling_total_budget": 400,
    "strict_joint_transition_enabled": True,
    "state_input_scale_enabled": False,
    "state_input_scale": 1.0,
    "state_full_rms_enabled": True,
    "state_full_rms_target": 1.0,
    "ctrl_reward_weight": {"distribution": "log_uniform", "min": 1e-3, "max": 3e-1},
    "ctrl_reward_enable_prob": 0.7,
    "survival_reward_weight": 0.0,
    "survival_reward_enable_prob": 0.0,
    "reinforce_reward_transform": "tanh",
    "reinforce_reward_rms_eps": 1e-6,
    "reinforce_reward_tanh_c": 10.0,
    "reinforce_reward_tanh_bound": {"distribution": "uniform", "min": 0.0, "max": 2.0},
    "reinforce_action_transform": "none",
    "reinforce_action_rms_eps": 1e-6,
    "reinforce_normalize_advantages": True,
    "reinforce_scale_advantages_by_suffix_episode_count": True,
    "reinforce_advantage_norm_eps": 1e-6,
    "reinforce_advantage_norm_clip": 10.0,
    "policy_gradient_weight": 0.4,
    "normalized_q_value_weight": 1.0,
    "next_state_flow_matching_weight": 1.0,
    "next_state_flow_head_type": "rwkv_two_layer",
    "reinforce_sequence_replay_enabled": True,
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
    "action_noise_train_std": {"distribution": "log_uniform", "min": 1e-2, "max": 0.2},
    "action_noise_eval_std": {"distribution": "log_uniform", "min": 1e-2, "max": 0.1},
    "reward_dropout_enabled": True,
    "reward_dropout_randomize": True,
    "reward_dropout_ratio_min": 0.1,
    "reward_dropout_ratio_max": 1.0,
    "reward_dropout_impute_zero": True,
    "terminal_reset_enabled": True,
    "terminal_reset_count_target": {"distribution": "uniform", "min": 0.0, "max": 20.0},
    "terminal_bonus_tanh_c": 10.0,
    "terminal_bonus_scale_min": 1.0,
    "terminal_bonus_scale_max": 2.0,
    "batch_parallel_backend": "torch_vectorized",
    "batch_shared_environment": False,
    "batch_vectorized_grouping": "family",
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
    config["transformer"]["single_eval_causal"] = True
    config["transformer"]["backbone"] = "rwkv7"
    config["transformer"]["rwkv_sequence_replay_checkpoint"] = True
    config["transformer"]["rwkv_sequence_replay_batch_chunk_size"] = 64
    config["transformer"]["rwkv_sequence_replay_token_budget"] = 262144
    config["optimizer"]["rl_objective"] = "reinforce"
    return layout


def validate_rlpfn_maintained_path_config(config):
    env_cfg = config["prior"]["environment"]
    layout = resolve_rlpfn_token_layout(env_cfg, num_features=config["prior"].get("num_features", None))
    if config["prior"]["prior_type"] != "environment_only":
        raise ValueError("Maintained RLPFN path requires prior_type=environment_only.")
    if str(config["optimizer"]["rl_objective"]).strip().lower() != "reinforce":
        raise ValueError("Maintained RLPFN path requires optimizer.rl_objective=reinforce.")
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
    return layout
