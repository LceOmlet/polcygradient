import torch

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
    assert cfg["optimizer"]["rl_objective"] == "policy_gradient"
    assert cfg["optimizer"]["policy_rollout_chunk_size"] is None
    assert cfg["optimizer"]["policy_rollout_checkpoint_reentrant"] is True
    assert cfg["optimizer"]["pg_grad_mutable_kv_cache"] is True
    assert cfg["optimizer"]["pg_kv_cache_mode"] == "paged"
    assert cfg["optimizer"]["pg_kv_cache_page_size"] == 128
    assert cfg["optimizer"]["policy_rollout_chunk_autotune"] is False
    assert cfg["optimizer"]["policy_rollout_chunk_grow_every"] == 8
    assert cfg["optimizer"]["policy_rollout_chunk_grow_factor"] == 2.0
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
    assert cfg["optimizer"]["pg_tbptt_window"] == 128
    assert cfg["optimizer"]["pg_oom_debug_raise"] is False
    assert cfg["optimizer"]["pg_saved_tensors_cpu_offload"] is False


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
