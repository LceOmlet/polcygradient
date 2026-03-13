import torch

import ticl.model_configs as model_configs
from ticl.model_configs import get_model_default_config
from ticl.models.encoders import Linear
from ticl.models.tabpfn import TabPFN


def test_rlpfn_default_config_uses_split_encoder():
    cfg = get_model_default_config("rlpfn")
    assert cfg["prior"]["prior_type"] == "environment_only"
    assert cfg["prior"]["num_features"] == 432
    assert cfg["prior"]["environment"]["batch_parallel_backend"] == "torch_vectorized"
    assert cfg["prior"]["environment"]["batch_shared_environment"] is False
    assert cfg["prior"]["environment"]["batch_vectorized_grouping"] == "family"
    assert cfg["transformer"]["x_encoder_type"] == "split_obs_action"
    assert cfg["transformer"]["x_obs_dim"] == 402
    assert cfg["transformer"]["x_action_dim"] == 30
    assert cfg["transformer"]["single_eval_causal"] is True
    assert cfg["prior"]["classification"]["num_features_sampler"] == "fixed"
    assert cfg["optimizer"]["rl_objective"] == "alpha_grad"
    assert cfg["prior"]["environment"]["family"] == {
        "distribution": "meta_choice",
        "choice_values": ["scm"],
    }
    assert cfg["prior"]["environment"]["constrained_dim_sampling_enabled"] is True
    assert cfg["prior"]["environment"]["constrained_dim_sampling_total_budget"] == 400
    assert cfg["prior"]["environment"]["strict_joint_transition_enabled"] is True
    assert cfg["prior"]["environment"]["state_input_scale_enabled"] is False
    assert cfg["prior"]["environment"]["state_input_scale"] == 1.0
    assert cfg["prior"]["environment"]["state_full_rms_enabled"] is True
    assert cfg["prior"]["environment"]["state_full_rms_target"] == 1.0
    assert cfg["prior"]["environment"]["reinforce_reward_transform"] == "tanh"
    assert cfg["prior"]["environment"]["reinforce_reward_rms_eps"] == 1e-6
    assert cfg["prior"]["environment"]["reinforce_reward_tanh_c"] == 10.0
    assert cfg["prior"]["environment"]["reinforce_reward_tanh_bound"] == {
        "distribution": "uniform",
        "min": 1.0,
        "max": 10.0,
    }
    assert cfg["prior"]["environment"]["reinforce_action_transform"] == "rms"
    assert cfg["prior"]["environment"]["reinforce_action_rms_eps"] == 1e-6
    assert cfg["prior"]["environment"]["first_policy_gradient_state_grad_clip_norm"] == 4.0
    assert cfg["prior"]["environment"]["first_policy_gradient_action_grad_clip_value"] == 0.0
    assert cfg["prior"]["environment"]["first_policy_gradient_action_grad_clip_norm"] == 1.0
    assert cfg["prior"]["environment"]["alpha_grad_local_coordinate_enabled"] is True
    assert cfg["prior"]["environment"]["alpha_grad_unit_grad_enabled"] is True
    assert cfg["prior"]["environment"]["alpha_grad_unit_grad_delta"] == 1e-6
    assert cfg["prior"]["environment"]["action_noise_train_std"] == {
        "distribution": "log_uniform",
        "min": 1e-2,
        "max": 0.2,
    }
    assert cfg["prior"]["environment"]["action_noise_eval_std"] == {
        "distribution": "log_uniform",
        "min": 1e-2,
        "max": 0.1,
    }
    assert cfg["optimizer"]["policy_rollout_chunk_size"] is None
    assert cfg["optimizer"]["policy_rollout_checkpoint_reentrant"] is True
    assert cfg["optimizer"]["pg_grad_mutable_kv_cache"] is True
    assert cfg["optimizer"]["pg_kv_cache_mode"] == "paged"
    assert cfg["optimizer"]["pg_kv_cache_page_size"] == 128
    assert cfg["optimizer"]["policy_rollout_chunk_autotune"] is False
    assert cfg["optimizer"]["policy_rollout_chunk_grow_every"] == 8
    assert cfg["optimizer"]["policy_rollout_chunk_grow_factor"] == 2.0
    assert cfg["optimizer"]["learning_rate"] == 4e-4
    assert cfg["dataloader"]["batch_size"] == 1024
    assert cfg["optimizer"]["pg_torch_compile"] is False
    assert cfg["optimizer"]["adamw_fused"] is True
    assert cfg["optimizer"]["train_profiler_enabled"] is False
    assert cfg["optimizer"]["train_profiler_output_path"] is None
    assert cfg["optimizer"]["train_profiler_wandb"] is True
    assert cfg["optimizer"]["train_profiler_ema_alpha"] == 0.2
    assert cfg["optimizer"]["train_profiler_warmup_epochs"] == 0
    assert cfg["optimizer"]["train_profiler_warmup_batches"] == 0
    assert cfg["optimizer"]["train_profiler_log_every_batches"] == 0
    assert cfg["optimizer"]["train_gpu_observer_enabled"] is False
    assert cfg["optimizer"]["train_gpu_observer_interval_sec"] == 1.0
    assert cfg["optimizer"]["train_gpu_observer_output_path"] is None
    assert cfg["optimizer"]["train_gpu_stage_output_path"] is None
    assert cfg["optimizer"]["train_kernel_profiler_enabled"] is False
    assert cfg["optimizer"]["train_kernel_profiler_output_dir"] is None
    assert cfg["optimizer"]["train_kernel_profiler_wait_steps"] == 1
    assert cfg["optimizer"]["train_kernel_profiler_warmup_steps"] == 1
    assert cfg["optimizer"]["train_kernel_profiler_active_steps"] == 3
    assert cfg["optimizer"]["train_kernel_profiler_repeat_steps"] == 1
    assert cfg["optimizer"]["train_kernel_profiler_record_shapes"] is True
    assert cfg["optimizer"]["train_kernel_profiler_profile_memory"] is True
    assert cfg["optimizer"]["train_kernel_profiler_with_stack"] is False
    assert cfg["optimizer"]["train_kernel_profiler_with_flops"] is False
    assert cfg["optimizer"]["train_kernel_profiler_log_every_batches"] == 0
    assert cfg["optimizer"]["pg_tbptt_window"] == 32
    assert cfg["optimizer"]["pg_env_replay_steps"] == 1
    assert cfg["optimizer"]["pg_oom_debug_raise"] is False
    assert cfg["optimizer"]["pg_oom_fail_fast"] is True
    assert cfg["optimizer"]["pg_saved_tensors_cpu_offload"] is True
    assert cfg["optimizer"]["pg_saved_tensors_cpu_offload_scope"] == "policy"
    assert cfg["optimizer"]["pg_saved_tensors_pin_memory"] is False
    assert cfg["optimizer"]["pg_saved_tensors_cpu_offload_auto_disable_when_safe"] is True
    assert cfg["optimizer"]["pg_saved_tensors_cpu_offload_auto_min_free_gb"] == 8.0
    assert cfg["optimizer"]["pg_saved_tensors_cpu_offload_auto_max_batch_size"] == 64
    assert cfg["optimizer"]["pg_saved_tensors_cpu_offload_auto_max_n_samples"] == 1024
    assert cfg["prior"]["environment"]["anti_explosion_vanishing_v5_enabled"] is False
    assert cfg["prior"]["environment"]["anti_explosion_vanishing_v5_next_enabled"] is False
    assert cfg["prior"]["environment"]["lipschitz_enforce"] is False
    assert cfg["prior"]["environment"]["anti_explosion_vanishing_v5_next_state_gain_lo"] == 0.985
    assert cfg["prior"]["environment"]["anti_explosion_vanishing_v5_next_state_gain_hi"] == 1.035
    assert cfg["prior"]["environment"]["anti_explosion_vanishing_v5_next_state_rms_lo"] == 4e-3
    assert cfg["prior"]["environment"]["anti_explosion_vanishing_v5_next_state_rms_hi"] == 9e-2
    assert cfg["prior"]["environment"]["anti_explosion_vanishing_v5_next_state_low_boost_cap"] == 1.5
    assert cfg["prior"]["environment"]["anti_explosion_vanishing_v5_next_loss_scale_hi"] == 4.0
    assert cfg["prior"]["environment"]["anti_explosion_vanishing_v5_next_step_grad_rms_lo"] == 1e-4
    assert cfg["prior"]["environment"]["anti_explosion_vanishing_v5_next_step_grad_rms_hi"] == 3e-2
    assert cfg["prior"]["environment"]["anti_explosion_vanishing_v5_next_step_low_boost_cap"] == 4.0
    assert cfg["prior"]["environment"]["lipschitz_weight_fro_norm_max"] == 1.0
    assert cfg["prior"]["environment"]["lipschitz_gp_outputscale_max"] == 1.0


def test_rlpfn_terminal_reset_enabled_expands_split_obs_slots(monkeypatch):
    orig_get_shared_defaults = model_configs.get_shared_defaults

    def _patched_shared_defaults():
        cfg = orig_get_shared_defaults()
        cfg["prior"]["environment"]["terminal_reset_enabled"] = True
        return cfg

    monkeypatch.setattr(model_configs, "get_shared_defaults", _patched_shared_defaults)
    cfg = model_configs.get_rlpfn_default_config()
    assert cfg["transformer"]["x_obs_dim"] == 403


def test_tabpfn_split_obs_action_encoder_forward():
    emsize = 32
    model = TabPFN(
        n_out=1,
        n_features=12,
        emsize=emsize,
        nhead=1,
        nhid_factor=2,
        nlayers=1,
        y_encoder_layer=Linear(1, emsize=emsize),
        classification_task=False,
        y_encoder="linear",
        x_encoder_type="split_obs_action",
        x_obs_dim=8,
        x_action_dim=4,
    )
    x = torch.randn(7, 3, 12)
    y = torch.randn(7, 3)
    out = model((x, y), single_eval_pos=5)
    assert out.shape == (2, 3, 1)
    assert torch.isfinite(out).all()


def _assert_nested_tensor_close(a, b, atol=1e-6, rtol=1e-5):
    if torch.is_tensor(a):
        assert torch.is_tensor(b)
        assert torch.allclose(a, b, atol=atol, rtol=rtol)
        return
    if isinstance(a, (list, tuple)):
        assert isinstance(b, type(a))
        assert len(a) == len(b)
        for ai, bi in zip(a, b):
            _assert_nested_tensor_close(ai, bi, atol=atol, rtol=rtol)
        return
    if isinstance(a, dict):
        assert isinstance(b, dict)
        assert set(a.keys()) == set(b.keys())
        for k in a.keys():
            _assert_nested_tensor_close(a[k], b[k], atol=atol, rtol=rtol)
        return
    assert a == b


def test_tabpfn_forward_policy_step_split_matches_materialized_token():
    torch.manual_seed(20260302)
    emsize = 16
    model = TabPFN(
        n_out=2,
        n_features=12,
        emsize=emsize,
        nhead=1,
        nhid_factor=2,
        nlayers=2,
        dropout=0.0,
        y_encoder_layer=Linear(1, emsize=emsize),
        classification_task=False,
        y_encoder="linear",
        x_encoder_type="split_obs_action",
        x_obs_dim=8,
        x_action_dim=4,
        single_eval_causal=True,
    )

    batch_size = 5
    obs_t = torch.randn(batch_size, 6)
    action_t = torch.randn(batch_size, 4)
    reward_t = torch.randn(batch_size, 1)
    reward_mask_t = torch.rand(batch_size, 1)

    x_token = torch.zeros(1, batch_size, 12, dtype=obs_t.dtype)
    x_token[0, :, :6] = obs_t
    x_token[0, :, 6] = reward_t.reshape(-1)
    x_token[0, :, 7] = reward_mask_t.reshape(-1)
    x_token[0, :, 8:12] = action_t
    y_token = reward_t.reshape(1, batch_size)

    out_ref, cache_ref = model.forward_policy_step(
        x_token,
        y_token,
        kv_cache=None,
        max_cache_len=32,
        kv_cache_mode="immutable",
    )
    out_split, cache_split = model.forward_policy_step_split(
        obs_t,
        action_t,
        reward_t,
        reward_mask_t,
        kv_cache=None,
        max_cache_len=32,
        kv_cache_mode="immutable",
    )

    assert torch.allclose(out_ref, out_split, atol=1e-6, rtol=1e-5)
    _assert_nested_tensor_close(cache_ref, cache_split, atol=1e-6, rtol=1e-5)


def test_tabpfn_forward_policy_step_split_matches_materialized_token_with_terminal():
    torch.manual_seed(20260313)
    emsize = 16
    model = TabPFN(
        n_out=2,
        n_features=13,
        emsize=emsize,
        nhead=1,
        nhid_factor=2,
        nlayers=2,
        dropout=0.0,
        y_encoder_layer=Linear(1, emsize=emsize),
        classification_task=False,
        y_encoder="linear",
        x_encoder_type="split_obs_action",
        x_obs_dim=9,
        x_action_dim=4,
        single_eval_causal=True,
    )

    batch_size = 5
    obs_t = torch.randn(batch_size, 6)
    action_t = torch.randn(batch_size, 4)
    reward_t = torch.randn(batch_size, 1)
    reward_mask_t = torch.rand(batch_size, 1)
    terminal_t = torch.rand(batch_size, 1)

    x_token = torch.zeros(1, batch_size, 13, dtype=obs_t.dtype)
    x_token[0, :, :6] = obs_t
    x_token[0, :, 6] = reward_t.reshape(-1)
    x_token[0, :, 7] = reward_mask_t.reshape(-1)
    x_token[0, :, 8] = terminal_t.reshape(-1)
    x_token[0, :, 9:13] = action_t
    y_token = reward_t.reshape(1, batch_size)

    out_ref, cache_ref = model.forward_policy_step(
        x_token,
        y_token,
        kv_cache=None,
        max_cache_len=32,
        kv_cache_mode="immutable",
    )
    out_split, cache_split = model.forward_policy_step_split(
        obs_t,
        action_t,
        reward_t,
        reward_mask_t,
        terminal_t=terminal_t,
        kv_cache=None,
        max_cache_len=32,
        kv_cache_mode="immutable",
    )

    assert torch.allclose(out_ref, out_split, atol=1e-6, rtol=1e-5)
    _assert_nested_tensor_close(cache_ref, cache_split, atol=1e-6, rtol=1e-5)
