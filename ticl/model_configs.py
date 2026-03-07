import torch

from ticl.config_utils import merge_dicts


def get_optimizer_config():
    optimizer = {
        "aggregate_k_gradients": 1,
        "rl_objective": "supervised",
        "policy_rollout_chunk_size": None,
        "policy_rollout_checkpoint": False,
        "policy_rollout_checkpoint_reentrant": False,
        "policy_rollout_chunk_autotune": True,
        "policy_rollout_chunk_grow_every": 8,
        "policy_rollout_chunk_grow_factor": 2.0,
        "pg_grad_mutable_kv_cache": False,
        "pg_saved_tensors_cpu_offload": False,
        "pg_saved_tensors_pin_memory": True,
        "pg_oom_debug_raise": False,
        "pg_oom_fail_fast": False,
        "pg_kv_cache_mode": "auto",
        "pg_kv_cache_page_size": None,
        "pg_tbptt_window": 64,
        "pg_env_replay_steps": 1,
        "pg_oom_reduce_tbptt_first": False,
        "pg_torch_compile": False,
        "pg_torch_compile_backend": "inductor",
        "pg_torch_compile_mode": "reduce-overhead",
        "pg_torch_compile_fullgraph": False,
        "pg_torch_compile_dynamic": False,
        "pg_compile_observe_recompiles": False,
        "pg_compile_observe_log_every_batches": 1,
        "pg_compile_observe_output_path": None,
        "pg_compile_observe_reset_after_warmup": True,
        "pg_phase_log_every_batches": 1,
        "pg_phase_log_file": None,
        "adamw_fused": True,
        "train_profiler_enabled": False,
        "train_profiler_output_path": None,
        "train_profiler_wandb": True,
        "train_profiler_ema_alpha": 0.2,
        "train_profiler_warmup_epochs": 0,
        "train_profiler_warmup_batches": 0,
        "train_profiler_log_every_batches": 0,
        "train_gpu_observer_enabled": False,
        "train_gpu_observer_interval_sec": 1.0,
        "train_gpu_observer_output_path": None,
        "train_gpu_stage_output_path": None,
        "train_kernel_profiler_enabled": False,
        "train_kernel_profiler_output_dir": None,
        "train_kernel_profiler_wait_steps": 1,
        "train_kernel_profiler_warmup_steps": 1,
        "train_kernel_profiler_active_steps": 3,
        "train_kernel_profiler_repeat_steps": 1,
        "train_kernel_profiler_record_shapes": True,
        "train_kernel_profiler_profile_memory": True,
        "train_kernel_profiler_with_stack": False,
        "train_kernel_profiler_with_flops": False,
        "train_kernel_profiler_log_every_batches": 0,
        "train_kernel_profiler_export_trace": True,
        "train_kernel_profiler_summary_top_k": 20,
        "learning_rate": 0.00003,
        "epochs": 4000,
        "train_mixed_precision": True,
        'stop_after_epochs': None,
        'reduce_lr_on_spike': False,
        'warmup_epochs': 20,
        'learning_rate_schedule': 'cosine',
        'min_lr': 1e-8,
        'adam_beta1': 0.9,
        'spike_tolerance': 4,
        'weight_decay': 0.0,
        'lr_decay': 0.99,
        'adaptive_batch_size': True
    }
    return {'optimizer': optimizer}


def get_transformer_config():
    transformer = {
        "emsize": 512,
        "nlayers": 12,
        "dropout": 0.0,
        "nhid_factor": 2,
        'nhead': 512 // 128,
        'init_method': None,
        'recompute_attn': True,
        'pre_norm': False,
        'y_encoder': "one_hot",
        'classification_task': True,
        'efficient_eval_masking': True,
        # single-directional attention for single-eval masking path
        'single_eval_causal': False,
        'input_normalization': False,
        'tabpfn_zero_weights': True,
        # x encoder layout:
        # - "single": one linear encoder over full num_features
        # - "split_obs_action": two heads (obs/reward/mask + action)
        'x_encoder_type': 'single',
        'x_obs_dim': None,
        'x_action_dim': None,
    #    'model': 'standard_attention',
    }
    return {'transformer': transformer}

def get_linear_attention_config():
    linear_attention = {
        "emsize": 512,
        "nlayers": 12,
        "dropout": 0.0,
        "nhid_factor": 2,
        'nhead': 512 // 128,
        'linear_attention_cfg': {
            'd_state': 16,
            'expand': 1,
        },
        'init_method': None,
        'recompute_attn': True,
        'pre_norm': False,
        'y_encoder': "one_hot",
        'classification_task': True,
        'efficient_eval_masking': True,
        'input_normalization': False,
        'tabpfn_zero_weights': True,
        'all_layers_same_init': False,
        'model': 'linear_attention',
        'feature_map': 'elu',
        'norm_output': False,
    }
    return {'linear_attention': linear_attention}


def get_prior_config(max_features=100, n_samples=1024+128):
    """"
    Returns the configuration parameters for the tabular multiclass wrapper.
    """

    prior = {
        "num_features": max_features,
        "n_samples": n_samples,
        "eval_positions": [n_samples * 0.95],
        'heterogeneous_batches': False,
        'multiclass_loss_type': 'nono',  # 'compatible'
        'prior_type': 'prior_bag',
        'prior_bag': {'prior_bag_exp_weights_1': {'distribution': 'uniform', 'min': 2.0, 'max': 10.0}}}

    mlp_prior_config = {"pre_sample_causes": True,
                        "sampling": 'normal',  # hp.choice('sampling', ['mixed', 'normal']), # uniform
                        'prior_mlp_scale_weights_sqrt': True,
                        'random_feature_rotation': True,
                        "num_layers": {'distribution': 'meta_gamma', 'max_alpha': 2, 'max_scale': 3, 'round': True, 'lower_bound': 2},
                        "prior_mlp_hidden_dim": {'distribution': 'meta_gamma', 'max_alpha': 3, 'max_scale': 100, 'round': True, 'lower_bound': 4},
                        "prior_mlp_dropout_prob": {'distribution': 'meta_beta', 'scale': 0.6, 'min': 0.1, 'max': 5.0},
                        # This mustn't be too high since activations get too large otherwise
                        "init_std": {'distribution': 'log_uniform', 'min': 1e-2, 'max': 12},
                        "noise_std": {'distribution': 'log_uniform', 'min': 1e-4, 'max': .5},
                        "num_causes": {'distribution': 'meta_gamma', 'max_alpha': 3, 'max_scale': 7, 'round': True,
                                       'lower_bound': 2},
                        "is_causal": {'distribution': 'meta_choice', 'choice_values': [True, False]},
                        "pre_sample_weights": {'distribution': 'meta_choice', 'choice_values': [True, False]},
                        "y_is_effect": {'distribution': 'meta_choice', 'choice_values': [True, False]},
                        # "sampling": {'distribution': 'meta_choice', 'choice_values': ['normal', 'mixed']},
                        "prior_mlp_activations": {'distribution': 'meta_choice', 'choice_values': [
                            torch.nn.Tanh, torch.nn.Identity, torch.nn.ReLU
                        ]},
                        "block_wise_dropout": {'distribution': 'meta_choice', 'choice_values': [True, False]},
                        "sort_features": {'distribution': 'meta_choice', 'choice_values': [True, False]},
                        "in_clique": {'distribution': 'meta_choice', 'choice_values': [True, False]},
                        'add_uninformative_features': False}

    prior['mlp'] = mlp_prior_config

    gp_prior_config = {
        'outputscale': {'distribution': 'log_uniform', 'min': 1e-5, 'max': 8},
        'lengthscale': {'distribution': 'log_uniform', 'min': 1e-5, 'max': 8},
        'noise': {'distribution': 'meta_choice', 'choice_values': [0.00001, 0.0001, 0.01]},
        "sampling": 'normal',  # hp.choice('sampling', ['mixed', 'normal']), # uniform
    }

    prior['gp'] = gp_prior_config

    environment_prior_config = {
        # Family sampling: SCM-style MLP or GP-style random features.
        "family": {"distribution": "meta_choice", "choice_values": ["scm", "gp"]},
        # Required randomization ranges.
        "action_dim": {"distribution": "uniform_int", "min": 1, "max": 30},
        "state_dim": {"distribution": "uniform_int", "min": 1, "max": 400},
        "obs_dim": {"distribution": "uniform_int", "min": 1, "max": 400},
        "noise_dim": {"distribution": "uniform_int", "min": 1, "max": 64},
        "zero_pad_dim": {"distribution": "uniform_int", "min": 0, "max": 400},
        # Fixed token slots for PFN input:
        # head-1 uses [s_t(obs slot), r_t, r_mask_t] => 402 dims by default.
        # head-2 uses [a_t] => 30 dims by default.
        "obs_slot_dim": 400,
        "action_slot_dim": 30,
        # Reward dropout for sparse-reward simulation.
        "reward_dropout_enabled": True,
        "reward_dropout_randomize": True,
        "reward_dropout_ratio": 0.0,
        "reward_dropout_ratio_min": 0.1,
        "reward_dropout_ratio_max": 1.0,
        "reward_dropout_impute_zero": True,
        # Parallel generation across independent batch columns in get_batch().
        "batch_parallel_workers": 4,
        # Backend for per-column batch generation:
        # - "python_thread": legacy ThreadPoolExecutor path
        # - "torch_vectorized": torch tensorized rollout path (no python thread pool)
        "batch_parallel_backend": "python_thread",
        # Whether all columns in a batch share one sampled environment function family.
        # This is required for strict tensorized vectorization.
        "batch_shared_environment": False,
        # When True, torch_vectorized path matches serial per-column RNG streams exactly.
        # This is primarily for semantic A/B tests and is slower than the default fast path.
        "batch_vectorized_strict_rng_match": False,
        # Vectorized grouping for policy rollout:
        # - "structure": strict homogeneous grouping (legacy behavior)
        # - "family": coarser grouping for larger batched policy steps
        "batch_vectorized_grouping": "structure",
        # Rollout/randomization knobs.
        "alpha": {"distribution": "uniform", "min": 0.05, "max": 0.35},
        "init_state_std": {"distribution": "log_uniform", "min": 1e-3, "max": 1.0},
        "init_action_std": {"distribution": "log_uniform", "min": 1e-3, "max": 1.0},
        "state_noise_std": {"distribution": "log_uniform", "min": 1e-4, "max": 0.2},
        "action_noise_train_std": {"distribution": "log_uniform", "min": 1e-4, "max": 0.2},
        "action_noise_eval_std": {"distribution": "log_uniform", "min": 1e-4, "max": 0.1},
        "reward_scale": {"distribution": "uniform", "min": 0.1, "max": 10.0},
        "reward_clip": 10.0,
        "state_clip": 8.0,
        # Optional state residual highway (default off, no behavior change).
        "state_highway_enabled": False,
        "state_highway_lambda": 0.0,
        # anti-explosion&vanishing-v2:
        # two-sided corridor regularization on log gain of consecutive
        # latent-state increments.
        "anti_explosion_vanishing_v2_enabled": False,
        "anti_explosion_vanishing_v2_lambda": 0.05,
        "anti_explosion_vanishing_v2_gain_lo": 0.85,
        "anti_explosion_vanishing_v2_gain_hi": 1.15,
        "anti_explosion_vanishing_v2_huber_delta": 0.05,
        "anti_explosion_vanishing_v2_eps": 1e-6,
        "anti_explosion_vanishing_v2_detach_reference": True,
        # anti-explosion&vanishing-v3:
        # decoupled drift+tail regularization on per-step log gain of
        # latent-state increments.
        "anti_explosion_vanishing_v3_enabled": False,
        "anti_explosion_vanishing_v3_lambda_drift": 0.02,
        "anti_explosion_vanishing_v3_lambda_tail": 0.05,
        "anti_explosion_vanishing_v3_gain_lo": 0.85,
        "anti_explosion_vanishing_v3_gain_hi": 1.15,
        "anti_explosion_vanishing_v3_tail_tau": 0.02,
        "anti_explosion_vanishing_v3_eps": 1e-6,
        "anti_explosion_vanishing_v3_detach_reference": True,
        # anti-explosion&vanishing-v4:
        # controlled highway-subspace update + drift/tail regularization
        # on per-step update gain (TBPTT-friendly).
        "anti_explosion_vanishing_v4_enabled": False,
        "anti_explosion_vanishing_v4_lambda_drift": 0.08,
        "anti_explosion_vanishing_v4_lambda_tail": 0.25,
        "anti_explosion_vanishing_v4_gain_lo": 0.97,
        "anti_explosion_vanishing_v4_gain_hi": 1.03,
        "anti_explosion_vanishing_v4_tail_tau": 0.010,
        "anti_explosion_vanishing_v4_eps": 1e-6,
        "anti_explosion_vanishing_v4_detach_reference": True,
        "anti_explosion_vanishing_v4_highway_ratio": 0.25,
        "anti_explosion_vanishing_v4_update_scale": 0.08,
        "anti_explosion_vanishing_v4_update_clip": 0.0,
        # anti-explosion&vanishing-v5:
        # detached reward-signal thermostat that rescales policy-gradient
        # loss magnitude without changing ascent direction on sum of rewards.
        "anti_explosion_vanishing_v5_enabled": False,
        "anti_explosion_vanishing_v5_target_std": 0.25,
        "anti_explosion_vanishing_v5_scale_lo": 0.5,
        "anti_explosion_vanishing_v5_scale_hi": 4.0,
        "anti_explosion_vanishing_v5_eps": 1e-6,
        "anti_explosion_vanishing_v5_detach_reference": True,
        # Policy-gradient stability knobs for differentiable rollout.
        # Train objective default: maximize raw discounted reward mean directly.
        "policy_gradient_normalize_rewards": False,
        "reward_norm_eps": 1e-6,
        "reward_norm_clip": 10.0,
        # Keep Bellman-style undiscounted default unless overridden.
        "discount": 1.0,
        # Lipschitz safeguards: project sampled generator matrices by
        # Frobenius norm and cap GP outputscale for bounded transition Jacobians.
        "lipschitz_enforce": False,
        "lipschitz_weight_fro_norm_max": 1.0,
        "lipschitz_gp_outputscale_max": 1.0,
        # SCM (aligned with priors/mlp.py names).
        "num_layers": {"distribution": "meta_gamma", "max_alpha": 2, "max_scale": 3, "round": True, "lower_bound": 2},
        "prior_mlp_hidden_dim": {"distribution": "meta_gamma", "max_alpha": 3, "max_scale": 128, "round": True, "lower_bound": 8},
        "prior_mlp_activations": {"distribution": "meta_choice", "choice_values": [torch.nn.Tanh, torch.nn.ReLU, torch.nn.Identity]},
        "init_std": {"distribution": "log_uniform", "min": 1e-3, "max": 1.0},
        "noise_std": {"distribution": "log_uniform", "min": 1e-4, "max": 0.2},
        # GP (aligned with priors/fast_gp.py names).
        "lengthscale": {"distribution": "log_uniform", "min": 1e-5, "max": 8},
        "outputscale": {"distribution": "log_uniform", "min": 1e-5, "max": 8},
        "noise": {"distribution": "meta_choice", "choice_values": [0.00001, 0.0001, 0.01]},
        "gp_rff_features": {"distribution": "uniform_int", "min": 32, "max": 256},
    }
    prior['environment'] = environment_prior_config

    prior['step_function'] = {
        'max_steps': 1,
        'sampling': 'uniform',
    }

    max_num_classes = 10
    classsification_prior = {
        "max_num_classes": max_num_classes,
        "num_classes": {'distribution': 'uniform_int', 'min': 2, 'max': max_num_classes},
        # "noise_type": "Gaussian",  # NN unused?!
        "balanced": False,
        'output_multiclass_ordered_p': 0.,
        'multiclass_max_steps': 10,
        "multiclass_type": 'rank',
        'categorical_feature_p': .2,  # diff: .0
        'nan_prob_no_reason': 0.0,
        'nan_prob_a_reason': 0.0,
        'set_value_to_nan': .9,
        'num_features_sampler': 'uniform',
        'pad_zeros': True,
        'feature_curriculum': False,
    }
    prior['classification'] = classsification_prior

    dataloader = {
        "batch_size": 8 * 16,
        "num_steps": 8 ,
        'min_eval_pos': 2,
        'random_n_samples': 0,
        'n_test_samples': 0,
    }
    
    openmlloader = {
        'valid_data': 'old',
        'max_samples': float('inf'),
        'pca': False,
    }

    prior['boolean'] = {
        'max_fraction_uninformative': 0.5,
        'p_uninformative': 0.5
    }

    return {'prior': prior, 'dataloader': dataloader, 'openmlloader': openmlloader}


def get_mothernet_config():
    return {'mothernet': {
        'weight_embedding_rank': 32,
        'low_rank_weights': True,
        'predicted_hidden_layer_size': 512,
        'predicted_activation': 'relu',
        'decoder_type': "class_average",
        'decoder_embed_dim': 1024,
        'predicted_hidden_layers': 2,
        'decoder_hidden_layers': 1,
        'decoder_hidden_size': 2048,
        'decoder_activation': 'gelu'}
    }


def get_additive_config():
    return {'additive': {
        'input_bin_embedding': 'none',
        'factorized_output': False,
        'output_rank': 16,
        'bin_embedding_rank': 16,
        'input_layer_norm': False,
        'shape_attention': False,
        'shape_attention_heads': 1,
        'n_shape_functions': 32,
        'shape_init': 'constant',
        'n_bins': 64,
        'nan_bin': False,
        'sklearn_binning': False,
        'fourier_features': 0,
        'marginal_residual': "none",
        'categorical_embedding': False

    }}


def get_biattention_config():
    return {'biattention': {
        'input_embedding': 'linear',
    }}


def get_shared_defaults(encoder_type = 'transformer'):
    config = get_prior_config()
    config.update(get_optimizer_config())
    if encoder_type == 'transformer':
        config.update(get_transformer_config())
    elif encoder_type == 'linear_attention':
        config.update(get_linear_attention_config())
    else:
        raise ValueError(f"Unknown encoder type {encoder_type}")
    return config


def get_mothernet_default_config():
    config = get_shared_defaults()
    config.update(get_mothernet_config())
    return config


def get_additive_default_config():
    config = get_shared_defaults()
    config.update(get_mothernet_config())
    config.update(get_additive_config())
    config['mothernet']['decoder_type'] = 'class_average'
    config['mothernet']['decoder_hidden_size'] = 512
    return config


def get_baam_default_config():
    config = get_shared_defaults()
    config.update(get_mothernet_config())
    config.update(get_additive_config())
    config.update(get_biattention_config())
    config['prior']['classification']['pad_zeros'] = False
    config['mothernet']['decoder_type'] = 'class_average'
    config['mothernet']['decoder_hidden_size'] = 512
    return config


def get_perceiver_default_config():
    config = get_shared_defaults()
    config['perceiver'] = {'num_latents': 512}
    config.update(get_mothernet_config())
    config['mothernet']['decoder_type'] = 'output_attention'
    return config


def get_tabpfn_default_config():
    config = get_shared_defaults()
    return config

def get_rlpfn_default_config():
    # RLPFN uses EnvironmentPrior directly with a split x-encoder:
    # - obs/reward/mask head: 402 dims (400 + 1 + 1)
    # - action head: 30 dims
    config = get_shared_defaults()
    config['prior']['prior_type'] = 'environment_only'
    config['prior']['num_features'] = 432
    # RLPFN default rollout horizon.
    config['prior']['n_samples'] = 1024

    env_cfg = config['prior']['environment']
    env_cfg.update({
        "obs_slot_dim": 400,
        "action_slot_dim": 30,
        "reward_dropout_enabled": True,
        "reward_dropout_randomize": True,
        "reward_dropout_ratio_min": 0.1,
        "reward_dropout_ratio_max": 1.0,
        "reward_dropout_impute_zero": True,
        "batch_parallel_backend": "torch_vectorized",
        "batch_shared_environment": False,
        "batch_vectorized_grouping": "family",
    })

    # Keep a strict fixed feature width for split heads.
    config['prior']['classification']['num_features_sampler'] = 'fixed'
    config['prior']['classification']['pad_zeros'] = False
    config['prior']['classification']['max_num_classes'] = 0
    config['transformer']['classification_task'] = False
    config['transformer']['y_encoder'] = 'linear'
    config['transformer']['x_encoder_type'] = 'split_obs_action'
    config['transformer']['x_obs_dim'] = int(env_cfg["obs_slot_dim"]) + 2
    config['transformer']['x_action_dim'] = int(env_cfg["action_slot_dim"])
    config['transformer']['single_eval_causal'] = True
    config['optimizer']['rl_objective'] = 'policy_gradient'
    # Policy-gradient rollout chunking over batch columns.
    # None means full-batch rollout chunk (max parallel width).
    config['optimizer']['policy_rollout_chunk_size'] = None
    # Recompute rollout forward during backward to reduce peak training memory.
    # Semantics are preserved (same sampled environments via RNG state restore),
    # with extra compute overhead.
    config['optimizer']['policy_rollout_checkpoint'] = True
    # Use reentrant checkpoint by default for rollout: it avoids building/storing
    # the full inner autograd graph during forward.
    config['optimizer']['policy_rollout_checkpoint_reentrant'] = True
    # Enable grad-mutable KV cache on paged mode (stable default).
    config['optimizer']['pg_grad_mutable_kv_cache'] = True
    config['optimizer']['pg_kv_cache_mode'] = "paged"
    config['optimizer']['pg_kv_cache_page_size'] = 128
    # Keep rollout chunk fixed by default.
    config['optimizer']['policy_rollout_chunk_autotune'] = False
    config['optimizer']['policy_rollout_chunk_grow_every'] = 8
    config['optimizer']['policy_rollout_chunk_grow_factor'] = 2.0
    # Keep PG path on eager by default for stable startup latency; compile can be
    # enabled explicitly when throughput profiling confirms a net gain.
    config['optimizer']['pg_torch_compile'] = False
    # Keep profiler opt-in by default so first-run training remains feasible.
    config['optimizer']['train_profiler_enabled'] = False
    config['optimizer']['train_profiler_output_path'] = None
    config['optimizer']['train_profiler_wandb'] = True
    config['optimizer']['train_profiler_ema_alpha'] = 0.2
    config['optimizer']['train_profiler_warmup_epochs'] = 0
    config['optimizer']['train_profiler_warmup_batches'] = 0
    config['optimizer']['train_profiler_log_every_batches'] = 0
    config['optimizer']['train_gpu_observer_enabled'] = False
    config['optimizer']['train_gpu_observer_interval_sec'] = 1.0
    config['optimizer']['train_gpu_observer_output_path'] = None
    config['optimizer']['train_gpu_stage_output_path'] = None
    config['optimizer']['train_kernel_profiler_enabled'] = False
    config['optimizer']['train_kernel_profiler_output_dir'] = None
    config['optimizer']['train_kernel_profiler_wait_steps'] = 1
    config['optimizer']['train_kernel_profiler_warmup_steps'] = 1
    config['optimizer']['train_kernel_profiler_active_steps'] = 3
    config['optimizer']['train_kernel_profiler_repeat_steps'] = 1
    config['optimizer']['train_kernel_profiler_record_shapes'] = True
    config['optimizer']['train_kernel_profiler_profile_memory'] = True
    config['optimizer']['train_kernel_profiler_with_stack'] = False
    config['optimizer']['train_kernel_profiler_with_flops'] = False
    config['optimizer']['train_kernel_profiler_log_every_batches'] = 0
    config['optimizer']['train_kernel_profiler_export_trace'] = True
    config['optimizer']['train_kernel_profiler_summary_top_k'] = 20
    # With reentrant rollout checkpoint defaulted on, saved-tensor CPU offload is
    # not needed by default and can otherwise shift pressure to host RAM.
    config['optimizer']['pg_saved_tensors_cpu_offload'] = False
    # Enable TBPTT by default for memory/throughput tradeoff.
    config['optimizer']['pg_tbptt_window'] = 64
    # Keep one rollout->update cycle per batch by default for throughput-first
    # benchmarking and simpler PG phase attribution.
    config['optimizer']['pg_env_replay_steps'] = 1
    # Keep rollout batch parallel width as large as possible under OOM:
    # shrink TBPTT window first before shrinking rollout chunk.
    config['optimizer']['pg_oom_reduce_tbptt_first'] = True
    # Fail fast on PG OOM during skyline/perf runs: avoid fallback retries
    # (TBPTT/chunk degradation) polluting per-batch wall-time measurements.
    config['optimizer']['pg_oom_debug_raise'] = False
    config['optimizer']['pg_oom_fail_fast'] = True
    # Physical batch has moved to 256 on the maintained mainline; scale the
    # default step size up with it instead of keeping the legacy tiny batch LR.
    config['optimizer']['learning_rate'] = 3e-4
    # Documented throughput mainline now uses physical batch 256.
    config['dataloader']['batch_size'] = 256
    return config


def get_batabpfn_default_config():
    config = get_shared_defaults()
    config.update(get_biattention_config())
    config['biattention']['input_embedding'] = 'fourier'
    config['prior']['classification']['pad_zeros'] = False
    return config

def get_tabflex_default_config():
    config = get_shared_defaults(encoder_type='linear_attention')
    return config

def get_la_mothernet_default_config():
    config = get_shared_defaults(encoder_type='linear_attention')
    config.update(get_mothernet_config())
    return config

def get_model_default_config(model_type, model = None):
    if model_type == 'mothernet':
        config = get_mothernet_default_config()
    elif model_type == 'batabpfn':
        config = get_batabpfn_default_config()
    elif model_type == 'tabpfn':
        config = get_tabpfn_default_config()
    elif model_type == 'rlpfn':
        config = get_rlpfn_default_config()
    elif model_type == 'additive':
        config = get_additive_default_config()
    elif model_type == 'baam':
        config = get_baam_default_config()
    elif model_type == 'perceiver':
        config = get_perceiver_default_config()
    elif model_type in ['tabflex', 'ssm_tabpfn']:
        config = get_tabflex_default_config()
    elif model_type == 'la_mothernet':
        config = get_la_mothernet_default_config()
    else:
        raise ValueError(f"Unknown model type {model_type}")
    config['model_type'] = model_type
    return config
