import pytest

from ticl.fit_model import main
from ticl.cli_parsing import make_model_level_argparser
from ticl.rl_validation import RLPFN_DEFAULT_OOP_ENVS
from ticl.model_configs import get_model_default_config
from argparse import Namespace
from ticl.fit_model import _cli_flag_is_set, _apply_continue_run_cli_overrides


def test_fit_model_help():
    try:
        main(['--help'])
    except SystemExit as e:
        assert e.code == 0
    else:
        assert False, "Expected SystemExit"


def test_rlpfn_parser_exposes_new_environment_and_causal_flags():
    parser = make_model_level_argparser()
    args = parser.parse_args(
        [
            "rlpfn",
            "--single-eval-causal", "false",
            "--family", "gp",
            "--state-dim", "12",
            "--reward-norm-eps", "1e-5",
            "--gp-rff-features", "64",
            "--batch-parallel-backend", "torch_vectorized",
            "--batch-shared-environment", "true",
            "--batch-vectorized-strict-rng-match", "true",
            "--rl-objective", "policy_gradient",
        ]
    )

    assert args.transformer.single_eval_causal is False
    assert args.prior.environment.family == "gp"
    assert args.prior.environment.state_dim == 12
    assert args.prior.environment.reward_norm_eps == 1e-5
    assert args.prior.environment.gp_rff_features == 64
    assert args.prior.environment.batch_parallel_backend == "torch_vectorized"
    assert args.prior.environment.batch_shared_environment is True
    assert args.prior.environment.batch_vectorized_strict_rng_match is True
    assert args.prior.environment.batch_vectorized_grouping == "family"
    assert args.optimizer.rl_objective == "policy_gradient"
    assert args.optimizer.policy_rollout_chunk_size is None
    assert args.optimizer.policy_rollout_chunk_autotune is False
    assert args.optimizer.policy_rollout_chunk_grow_every == 8
    assert args.optimizer.policy_rollout_chunk_grow_factor == 2.0
    assert args.optimizer.policy_rollout_checkpoint_reentrant is True
    assert args.optimizer.pg_grad_mutable_kv_cache is True
    assert args.optimizer.pg_kv_cache_mode == "paged"
    assert args.optimizer.pg_kv_cache_page_size == 128
    assert args.optimizer.pg_torch_compile is False
    assert args.optimizer.adamw_fused is True
    assert args.optimizer.train_profiler_enabled is False
    assert args.optimizer.train_profiler_output_path is None
    assert args.optimizer.train_profiler_wandb is True
    assert args.optimizer.train_profiler_ema_alpha == 0.2
    assert args.optimizer.train_profiler_warmup_epochs == 0
    assert args.optimizer.train_profiler_warmup_batches == 0
    assert args.optimizer.train_profiler_log_every_batches == 0
    assert args.optimizer.train_gpu_observer_enabled is False
    assert args.optimizer.train_gpu_observer_interval_sec == 1.0
    assert args.optimizer.train_gpu_observer_output_path is None
    assert args.optimizer.train_gpu_stage_output_path is None
    assert args.optimizer.train_kernel_profiler_enabled is False
    assert args.optimizer.train_kernel_profiler_output_dir is None
    assert args.optimizer.train_kernel_profiler_wait_steps == 1
    assert args.optimizer.train_kernel_profiler_warmup_steps == 1
    assert args.optimizer.train_kernel_profiler_active_steps == 3
    assert args.optimizer.train_kernel_profiler_repeat_steps == 1
    assert args.optimizer.train_kernel_profiler_record_shapes is True
    assert args.optimizer.train_kernel_profiler_profile_memory is True
    assert args.optimizer.train_kernel_profiler_with_stack is False
    assert args.optimizer.train_kernel_profiler_with_flops is False
    assert args.optimizer.train_kernel_profiler_log_every_batches == 0
    assert args.optimizer.pg_tbptt_window == 32
    assert args.optimizer.pg_env_replay_steps == 1
    assert args.optimizer.pg_oom_reduce_tbptt_first is True
    assert args.optimizer.pg_oom_debug_raise is False
    assert args.optimizer.pg_saved_tensors_cpu_offload is True
    assert args.optimizer.pg_saved_tensors_cpu_offload_scope == "policy"
    assert args.optimizer.pg_saved_tensors_pin_memory is False
    assert args.optimizer.pg_saved_tensors_cpu_offload_auto_disable_when_safe is False
    assert args.optimizer.pg_saved_tensors_cpu_offload_auto_min_free_gb == 8.0
    assert args.optimizer.pg_saved_tensors_cpu_offload_auto_max_batch_size == 64
    assert args.optimizer.pg_saved_tensors_cpu_offload_auto_max_n_samples == 1024
    assert args.orchestration.rl_validate_enabled is True
    assert args.orchestration.rl_validate_envs.split(",") == RLPFN_DEFAULT_OOP_ENVS
    assert args.orchestration.rl_validate_context_lower_bound == 2048


def test_rlpfn_parser_accepts_rl_validate_context_lower_bound():
    parser = make_model_level_argparser()
    args = parser.parse_args(
        [
            "rlpfn",
            "--rl-validate-context-lower-bound", "4096",
        ]
    )
    assert args.orchestration.rl_validate_context_lower_bound == 4096


def test_rlpfn_parser_accepts_aev5_next_flags():
    parser = make_model_level_argparser()
    args = parser.parse_args(
        [
            "rlpfn",
            "--anti-explosion-vanishing-v5-next-enabled", "true",
            "--anti-explosion-vanishing-v5-next-state-gain-lo", "0.97",
            "--anti-explosion-vanishing-v5-next-state-gain-hi", "1.02",
            "--anti-explosion-vanishing-v5-next-step-grad-rms-hi", "0.05",
        ]
    )
    assert args.prior.environment.anti_explosion_vanishing_v5_next_enabled is True
    assert args.prior.environment.anti_explosion_vanishing_v5_next_state_gain_lo == 0.97
    assert args.prior.environment.anti_explosion_vanishing_v5_next_state_gain_hi == 1.02
    assert args.prior.environment.anti_explosion_vanishing_v5_next_step_grad_rms_hi == 0.05


def test_rlpfn_parser_accepts_saved_tensors_offload_flags():
    parser = make_model_level_argparser()
    args = parser.parse_args(
        [
            "rlpfn",
            "--pg-saved-tensors-cpu-offload", "true",
            "--pg-saved-tensors-cpu-offload-scope", "policy",
            "--pg-saved-tensors-pin-memory", "false",
            "--pg-saved-tensors-cpu-offload-auto-disable-when-safe", "true",
            "--pg-saved-tensors-cpu-offload-auto-min-free-gb", "10",
            "--pg-saved-tensors-cpu-offload-auto-max-batch-size", "32",
            "--pg-saved-tensors-cpu-offload-auto-max-n-samples", "512",
        ]
    )
    assert args.optimizer.pg_saved_tensors_cpu_offload is True
    assert args.optimizer.pg_saved_tensors_cpu_offload_scope == "policy"
    assert args.optimizer.pg_saved_tensors_pin_memory is False
    assert args.optimizer.pg_saved_tensors_cpu_offload_auto_disable_when_safe is True
    assert args.optimizer.pg_saved_tensors_cpu_offload_auto_min_free_gb == 10.0
    assert args.optimizer.pg_saved_tensors_cpu_offload_auto_max_batch_size == 32
    assert args.optimizer.pg_saved_tensors_cpu_offload_auto_max_n_samples == 512


def test_rlpfn_parser_accepts_reinforce_objective():
    parser = make_model_level_argparser()
    args = parser.parse_args(
        [
            "rlpfn",
            "--rl-objective", "reinforce",
        ]
    )
    assert args.optimizer.rl_objective == "reinforce"


def test_rlpfn_parser_accepts_first_policy_gradient_objective():
    parser = make_model_level_argparser()
    args = parser.parse_args(
        [
            "rlpfn",
            "--rl-objective", "first_policy_gradient",
        ]
    )
    assert args.optimizer.rl_objective == "first_policy_gradient"


def test_rlpfn_parser_accepts_alpha_grad_objective():
    parser = make_model_level_argparser()
    args = parser.parse_args(
        [
            "rlpfn",
            "--rl-objective", "alpha_grad",
        ]
    )
    assert args.optimizer.rl_objective == "alpha_grad"


def test_rlpfn_parser_accepts_first_pg_action_grad_clip_options():
    parser = make_model_level_argparser()
    args = parser.parse_args(
        [
            "rlpfn",
            "--first-policy-gradient-action-grad-clip-value", "2.5",
            "--first-policy-gradient-action-grad-clip-norm", "1.5",
        ]
    )
    assert args.prior.environment.first_policy_gradient_action_grad_clip_value == 2.5
    assert args.prior.environment.first_policy_gradient_action_grad_clip_norm == 1.5


def test_rlpfn_parser_accepts_alpha_grad_coordinate_and_unit_options():
    parser = make_model_level_argparser()
    args = parser.parse_args(
        [
            "rlpfn",
            "--alpha-grad-local-coordinate-enabled", "false",
            "--alpha-grad-unit-grad-enabled", "false",
            "--alpha-grad-unit-grad-delta", "1e-4",
        ]
    )
    assert args.prior.environment.alpha_grad_local_coordinate_enabled is False
    assert args.prior.environment.alpha_grad_unit_grad_enabled is False
    assert args.prior.environment.alpha_grad_unit_grad_delta == pytest.approx(1e-4)


def test_rlpfn_parser_defaults_enable_joint_env_and_budgeted_dims():
    cfg = get_model_default_config("rlpfn")

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
        "min": 0.0,
        "max": 10.0,
    }
    assert cfg["prior"]["environment"]["reinforce_action_transform"] == "rms"
    assert cfg["prior"]["environment"]["reinforce_action_rms_eps"] == 1e-6
    assert cfg["prior"]["environment"]["first_policy_gradient_state_grad_clip_norm"] == 4.0
    assert cfg["prior"]["environment"]["first_policy_gradient_action_grad_clip_value"] == 4.0
    assert cfg["prior"]["environment"]["first_policy_gradient_action_grad_clip_norm"] == 0.0
    assert cfg["prior"]["environment"]["alpha_grad_local_coordinate_enabled"] is False
    assert cfg["prior"]["environment"]["alpha_grad_unit_grad_enabled"] is False
    assert cfg["prior"]["environment"]["alpha_grad_unit_grad_delta"] == 1e-6
    assert cfg["prior"]["environment"]["terminal_reset_enabled"] is False
    assert cfg["prior"]["environment"]["reference_scm_partition_max_bytes"] == 2 * 1024 * 1024 * 1024
    assert cfg["optimizer"]["pg_saved_tensors_cpu_offload"] is True
    assert cfg["optimizer"]["pg_saved_tensors_cpu_offload_scope"] == "policy"
    assert cfg["optimizer"]["pg_saved_tensors_pin_memory"] is False
    assert cfg["optimizer"]["pg_saved_tensors_cpu_offload_auto_disable_when_safe"] is False
    assert cfg["optimizer"]["pg_saved_tensors_cpu_offload_auto_min_free_gb"] == 8.0
    assert cfg["optimizer"]["pg_saved_tensors_cpu_offload_auto_max_batch_size"] == 64
    assert cfg["optimizer"]["pg_saved_tensors_cpu_offload_auto_max_n_samples"] == 1024
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


def test_rlpfn_parser_accepts_pg_grad_mutable_kv_cache_flag():
    parser = make_model_level_argparser()
    args = parser.parse_args(
        [
            "rlpfn",
            "--pg-grad-mutable-kv-cache", "false",
        ]
    )
    assert args.optimizer.pg_grad_mutable_kv_cache is False


def test_rlpfn_parser_accepts_pg_kv_cache_flags():
    parser = make_model_level_argparser()
    args = parser.parse_args(
        [
            "rlpfn",
            "--pg-kv-cache-mode", "paged",
            "--pg-kv-cache-page-size", "1",
        ]
    )
    assert args.optimizer.pg_kv_cache_mode == "paged"
    assert args.optimizer.pg_kv_cache_page_size == 1


def test_rlpfn_parser_accepts_pg_tbptt_window_flag():
    parser = make_model_level_argparser()
    args = parser.parse_args(
        [
            "rlpfn",
            "--pg-tbptt-window", "64",
        ]
    )
    assert args.optimizer.pg_tbptt_window == 64


def test_rlpfn_parser_accepts_pg_env_replay_steps_flag():
    parser = make_model_level_argparser()
    args = parser.parse_args(
        [
            "rlpfn",
            "--pg-env-replay-steps", "4",
        ]
    )
    assert args.optimizer.pg_env_replay_steps == 4


def test_rlpfn_parser_accepts_pg_oom_reduce_tbptt_first_flag():
    parser = make_model_level_argparser()
    args = parser.parse_args(
        [
            "rlpfn",
            "--pg-oom-reduce-tbptt-first", "false",
        ]
    )
    assert args.optimizer.pg_oom_reduce_tbptt_first is False


def test_rlpfn_parser_accepts_batch_vectorized_grouping_flag():
    parser = make_model_level_argparser()
    args = parser.parse_args(
        [
            "rlpfn",
            "--batch-vectorized-grouping", "structure",
        ]
    )
    assert args.prior.environment.batch_vectorized_grouping == "structure"


def test_rlpfn_parser_accepts_pg_oom_debug_raise_flag():
    parser = make_model_level_argparser()
    args = parser.parse_args(
        [
            "rlpfn",
            "--pg-oom-debug-raise", "true",
        ]
    )
    assert args.optimizer.pg_oom_debug_raise is True


def test_rlpfn_parser_accepts_policy_rollout_checkpoint_reentrant_flag():
    parser = make_model_level_argparser()
    args = parser.parse_args(
        [
            "rlpfn",
            "--policy-rollout-checkpoint-reentrant", "false",
        ]
    )
    assert args.optimizer.policy_rollout_checkpoint_reentrant is False


def test_rlpfn_parser_accepts_pg_torch_compile_flags():
    parser = make_model_level_argparser()
    args = parser.parse_args(
        [
            "rlpfn",
            "--pg-torch-compile", "true",
            "--pg-torch-compile-backend", "eager",
            "--pg-torch-compile-mode", "reduce-overhead",
            "--pg-torch-compile-fullgraph", "false",
            "--pg-torch-compile-dynamic", "false",
        ]
    )
    assert args.optimizer.pg_torch_compile is True
    assert args.optimizer.pg_torch_compile_backend == "eager"
    assert args.optimizer.pg_torch_compile_mode == "reduce-overhead"
    assert args.optimizer.pg_torch_compile_fullgraph is False
    assert args.optimizer.pg_torch_compile_dynamic is False


def test_rlpfn_parser_accepts_chunk_autotune_and_adamw_fused_flags():
    parser = make_model_level_argparser()
    args = parser.parse_args(
        [
            "rlpfn",
            "--policy-rollout-chunk-autotune", "false",
            "--policy-rollout-chunk-grow-every", "3",
            "--policy-rollout-chunk-grow-factor", "1.5",
            "--adamw-fused", "false",
        ]
    )
    assert args.optimizer.policy_rollout_chunk_autotune is False
    assert args.optimizer.policy_rollout_chunk_grow_every == 3
    assert args.optimizer.policy_rollout_chunk_grow_factor == 1.5
    assert args.optimizer.adamw_fused is False


def test_rlpfn_parser_accepts_train_profiler_flags():
    parser = make_model_level_argparser()
    args = parser.parse_args(
        [
            "rlpfn",
            "--train-profiler-enabled", "true",
            "--train-profiler-output-path", "/tmp/rlpfn_profile.jsonl",
            "--train-profiler-wandb", "false",
            "--train-profiler-ema-alpha", "0.3",
            "--train-profiler-warmup-epochs", "5",
            "--train-profiler-warmup-batches", "7",
            "--train-profiler-log-every-batches", "2",
        ]
    )
    assert args.optimizer.train_profiler_enabled is True
    assert args.optimizer.train_profiler_output_path == "/tmp/rlpfn_profile.jsonl"
    assert args.optimizer.train_profiler_wandb is False
    assert args.optimizer.train_profiler_ema_alpha == 0.3
    assert args.optimizer.train_profiler_warmup_epochs == 5
    assert args.optimizer.train_profiler_warmup_batches == 7
    assert args.optimizer.train_profiler_log_every_batches == 2


def test_rlpfn_parser_accepts_gpu_observer_flags():
    parser = make_model_level_argparser()
    args = parser.parse_args(
        [
            "rlpfn",
            "--train-gpu-observer-enabled", "true",
            "--train-gpu-observer-interval-sec", "0.5",
            "--train-gpu-observer-output-path", "/tmp/rlpfn_gpu_samples.jsonl",
            "--train-gpu-stage-output-path", "/tmp/rlpfn_gpu_stages.jsonl",
        ]
    )
    assert args.optimizer.train_gpu_observer_enabled is True
    assert args.optimizer.train_gpu_observer_interval_sec == 0.5
    assert args.optimizer.train_gpu_observer_output_path == "/tmp/rlpfn_gpu_samples.jsonl"
    assert args.optimizer.train_gpu_stage_output_path == "/tmp/rlpfn_gpu_stages.jsonl"


def test_rlpfn_parser_accepts_kernel_profiler_flags():
    parser = make_model_level_argparser()
    args = parser.parse_args(
        [
            "rlpfn",
            "--train-kernel-profiler-enabled", "true",
            "--train-kernel-profiler-output-dir", "/tmp/rlpfn_kernel_profile",
            "--train-kernel-profiler-wait-steps", "2",
            "--train-kernel-profiler-warmup-steps", "3",
            "--train-kernel-profiler-active-steps", "4",
            "--train-kernel-profiler-repeat-steps", "2",
            "--train-kernel-profiler-record-shapes", "false",
            "--train-kernel-profiler-profile-memory", "false",
            "--train-kernel-profiler-with-stack", "true",
            "--train-kernel-profiler-with-flops", "true",
            "--train-kernel-profiler-log-every-batches", "5",
        ]
    )
    assert args.optimizer.train_kernel_profiler_enabled is True
    assert args.optimizer.train_kernel_profiler_output_dir == "/tmp/rlpfn_kernel_profile"
    assert args.optimizer.train_kernel_profiler_wait_steps == 2
    assert args.optimizer.train_kernel_profiler_warmup_steps == 3
    assert args.optimizer.train_kernel_profiler_active_steps == 4
    assert args.optimizer.train_kernel_profiler_repeat_steps == 2
    assert args.optimizer.train_kernel_profiler_record_shapes is False
    assert args.optimizer.train_kernel_profiler_profile_memory is False
    assert args.optimizer.train_kernel_profiler_with_stack is True
    assert args.optimizer.train_kernel_profiler_with_flops is True
    assert args.optimizer.train_kernel_profiler_log_every_batches == 5


def test_cli_flag_is_set_detects_split_and_equals_forms():
    assert _cli_flag_is_set(["rlpfn", "--policy-rollout-chunk-size", "1"], "--policy-rollout-chunk-size")
    assert _cli_flag_is_set(["rlpfn", "--policy-rollout-chunk-size=1"], "--policy-rollout-chunk-size")
    assert not _cli_flag_is_set(["rlpfn", "--batch-size", "8"], "--policy-rollout-chunk-size")


def test_continue_run_cli_override_applies_policy_rollout_chunk_size():
    config = {"optimizer": {"policy_rollout_chunk_size": 64}}
    args = Namespace(optimizer=Namespace(policy_rollout_chunk_size=1))
    argv = ["rlpfn", "--policy-rollout-chunk-size", "1"]
    out = _apply_continue_run_cli_overrides(config, args, argv)
    assert out["optimizer"]["policy_rollout_chunk_size"] == 1


def test_continue_run_cli_override_keeps_old_chunk_size_when_flag_not_set():
    config = {"optimizer": {"policy_rollout_chunk_size": 64}}
    args = Namespace(optimizer=Namespace(policy_rollout_chunk_size=None))
    argv = ["rlpfn", "--batch-size", "8"]
    out = _apply_continue_run_cli_overrides(config, args, argv)
    assert out["optimizer"]["policy_rollout_chunk_size"] == 64


def test_continue_run_cli_override_applies_pg_tbptt_window():
    config = {"optimizer": {"pg_tbptt_window": None}}
    args = Namespace(optimizer=Namespace(pg_tbptt_window=128))
    argv = ["rlpfn", "--pg-tbptt-window", "128"]
    out = _apply_continue_run_cli_overrides(config, args, argv)
    assert out["optimizer"]["pg_tbptt_window"] == 128


def test_continue_run_cli_override_applies_pg_env_replay_steps():
    config = {"optimizer": {"pg_env_replay_steps": 8}}
    args = Namespace(optimizer=Namespace(pg_env_replay_steps=3))
    argv = ["rlpfn", "--pg-env-replay-steps", "3"]
    out = _apply_continue_run_cli_overrides(config, args, argv)
    assert out["optimizer"]["pg_env_replay_steps"] == 3


def test_continue_run_cli_override_applies_train_profiler_flags():
    config = {
        "optimizer": {
            "train_profiler_enabled": False,
            "train_profiler_output_path": None,
            "train_profiler_wandb": True,
            "train_profiler_ema_alpha": 0.2,
            "train_profiler_warmup_epochs": 0,
            "train_profiler_warmup_batches": 0,
            "train_profiler_log_every_batches": 0,
        }
    }
    args = Namespace(
        optimizer=Namespace(
            train_profiler_enabled=True,
            train_profiler_output_path="/tmp/profile.jsonl",
            train_profiler_wandb=False,
            train_profiler_ema_alpha=0.4,
            train_profiler_warmup_epochs=3,
            train_profiler_warmup_batches=11,
            train_profiler_log_every_batches=2,
        )
    )
    argv = [
        "rlpfn",
        "--train-profiler-enabled", "true",
        "--train-profiler-output-path", "/tmp/profile.jsonl",
        "--train-profiler-wandb", "false",
        "--train-profiler-ema-alpha", "0.4",
        "--train-profiler-warmup-epochs", "3",
        "--train-profiler-warmup-batches", "11",
        "--train-profiler-log-every-batches", "2",
    ]
    out = _apply_continue_run_cli_overrides(config, args, argv)
    assert out["optimizer"]["train_profiler_enabled"] is True
    assert out["optimizer"]["train_profiler_output_path"] == "/tmp/profile.jsonl"
    assert out["optimizer"]["train_profiler_wandb"] is False
    assert out["optimizer"]["train_profiler_ema_alpha"] == 0.4
    assert out["optimizer"]["train_profiler_warmup_epochs"] == 3
    assert out["optimizer"]["train_profiler_warmup_batches"] == 11
    assert out["optimizer"]["train_profiler_log_every_batches"] == 2


def test_continue_run_cli_override_applies_gpu_observer_flags():
    config = {
        "optimizer": {
            "train_gpu_observer_enabled": False,
            "train_gpu_observer_interval_sec": 1.0,
            "train_gpu_observer_output_path": None,
            "train_gpu_stage_output_path": None,
        }
    }
    args = Namespace(
        optimizer=Namespace(
            train_gpu_observer_enabled=True,
            train_gpu_observer_interval_sec=0.5,
            train_gpu_observer_output_path="/tmp/gpu_samples.jsonl",
            train_gpu_stage_output_path="/tmp/gpu_stages.jsonl",
        )
    )
    argv = [
        "rlpfn",
        "--train-gpu-observer-enabled", "true",
        "--train-gpu-observer-interval-sec", "0.5",
        "--train-gpu-observer-output-path", "/tmp/gpu_samples.jsonl",
        "--train-gpu-stage-output-path", "/tmp/gpu_stages.jsonl",
    ]
    out = _apply_continue_run_cli_overrides(config, args, argv)
    assert out["optimizer"]["train_gpu_observer_enabled"] is True
    assert out["optimizer"]["train_gpu_observer_interval_sec"] == 0.5
    assert out["optimizer"]["train_gpu_observer_output_path"] == "/tmp/gpu_samples.jsonl"
    assert out["optimizer"]["train_gpu_stage_output_path"] == "/tmp/gpu_stages.jsonl"


def test_continue_run_cli_override_applies_kernel_profiler_flags():
    config = {
        "optimizer": {
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
        }
    }
    args = Namespace(
        optimizer=Namespace(
            train_kernel_profiler_enabled=True,
            train_kernel_profiler_output_dir="/tmp/kernel_profile",
            train_kernel_profiler_wait_steps=2,
            train_kernel_profiler_warmup_steps=3,
            train_kernel_profiler_active_steps=4,
            train_kernel_profiler_repeat_steps=5,
            train_kernel_profiler_record_shapes=False,
            train_kernel_profiler_profile_memory=False,
            train_kernel_profiler_with_stack=True,
            train_kernel_profiler_with_flops=True,
            train_kernel_profiler_log_every_batches=7,
        )
    )
    argv = [
        "rlpfn",
        "--train-kernel-profiler-enabled", "true",
        "--train-kernel-profiler-output-dir", "/tmp/kernel_profile",
        "--train-kernel-profiler-wait-steps", "2",
        "--train-kernel-profiler-warmup-steps", "3",
        "--train-kernel-profiler-active-steps", "4",
        "--train-kernel-profiler-repeat-steps", "5",
        "--train-kernel-profiler-record-shapes", "false",
        "--train-kernel-profiler-profile-memory", "false",
        "--train-kernel-profiler-with-stack", "true",
        "--train-kernel-profiler-with-flops", "true",
        "--train-kernel-profiler-log-every-batches", "7",
    ]
    out = _apply_continue_run_cli_overrides(config, args, argv)
    assert out["optimizer"]["train_kernel_profiler_enabled"] is True
    assert out["optimizer"]["train_kernel_profiler_output_dir"] == "/tmp/kernel_profile"
    assert out["optimizer"]["train_kernel_profiler_wait_steps"] == 2
    assert out["optimizer"]["train_kernel_profiler_warmup_steps"] == 3
    assert out["optimizer"]["train_kernel_profiler_active_steps"] == 4
    assert out["optimizer"]["train_kernel_profiler_repeat_steps"] == 5
    assert out["optimizer"]["train_kernel_profiler_record_shapes"] is False
    assert out["optimizer"]["train_kernel_profiler_profile_memory"] is False
    assert out["optimizer"]["train_kernel_profiler_with_stack"] is True
    assert out["optimizer"]["train_kernel_profiler_with_flops"] is True
    assert out["optimizer"]["train_kernel_profiler_log_every_batches"] == 7
