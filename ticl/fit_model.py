import socket
import sys
import time
import random
import os
import threading
import atexit

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

_FIT_MODEL_HEARTBEAT_STAGE = "module_import"
_FIT_MODEL_HEARTBEAT_STOP = threading.Event()
_FIT_MODEL_HEARTBEAT_STARTED = False


def _fit_model_env_truthy(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _set_fit_model_heartbeat_stage(stage: str) -> None:
    global _FIT_MODEL_HEARTBEAT_STAGE
    _FIT_MODEL_HEARTBEAT_STAGE = str(stage)


def _fit_model_heartbeat_interval_sec() -> float:
    raw_value = os.environ.get(
        "TICL_FIT_MODEL_HEARTBEAT_INTERVAL_SEC",
        os.environ.get("TICL_PPO_PHASE_LOG_MIN_INTERVAL_SEC", "30"),
    )
    try:
        return max(1.0, float(raw_value))
    except Exception:
        return 30.0


def _emit_fit_model_heartbeat(start_time: float) -> None:
    print(
        "[fit-model-heartbeat] "
        f"date={time.strftime('%Y-%m-%dT%H:%M:%S%z')} "
        f"pid={os.getpid()} "
        f"elapsed_s={float(time.perf_counter() - start_time):.1f} "
        f"stage={_FIT_MODEL_HEARTBEAT_STAGE}",
        flush=True,
    )


def _fit_model_heartbeat_loop(start_time: float, interval_sec: float) -> None:
    while not _FIT_MODEL_HEARTBEAT_STOP.wait(float(interval_sec)):
        _emit_fit_model_heartbeat(start_time)


def _start_fit_model_heartbeat_if_enabled() -> None:
    global _FIT_MODEL_HEARTBEAT_STARTED
    if _FIT_MODEL_HEARTBEAT_STARTED:
        return
    if not _fit_model_env_truthy("TICL_FIT_MODEL_HEARTBEAT_ENABLED", default=False):
        return
    _FIT_MODEL_HEARTBEAT_STARTED = True
    interval_sec = _fit_model_heartbeat_interval_sec()
    start_time = time.perf_counter()
    _emit_fit_model_heartbeat(start_time)
    thread = threading.Thread(
        target=_fit_model_heartbeat_loop,
        args=(start_time, interval_sec),
        name="fit-model-heartbeat",
        daemon=True,
    )
    thread.start()


_start_fit_model_heartbeat_if_enabled()
atexit.register(_FIT_MODEL_HEARTBEAT_STOP.set)

import mlflow

import torch
import numpy as np

from git import Repo

from ticl.rlpfn_skyline_env import apply_rlpfn_skyline_env_defaults


apply_rlpfn_skyline_env_defaults()

from ticl.model_builder import get_model
from ticl.utils import (
    init_device,
    get_model_string,
    synetune_handle_checkpoint,
    make_training_callback,
    enforce_path_filename_limit,
)
from ticl.config_utils import compare_dicts, flatten_dict, update_config
from ticl.cli_parsing import make_model_level_argparser
from ticl.model_configs import get_model_default_config
from ticl.host_memory_guard import install_host_rss_limit_guard
from ticl.rlpfn_anchor_contract import apply_rlpfn_anchor_compare_contract
from argparse import Namespace


def _seed_python_numpy_torch(seed: int) -> None:
    seed_int = int(seed)
    random.seed(seed_int)
    np.random.seed(seed_int)
    torch.manual_seed(seed_int)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_int)


def _merge_missing_keys(dst, src):
    for k, v in src.items():
        if k not in dst:
            dst[k] = v
        elif isinstance(dst.get(k), dict) and isinstance(v, dict):
            _merge_missing_keys(dst[k], v)
    return dst


def _cli_flag_is_set(argv, flag):
    if not argv:
        return False
    for tok in argv:
        if tok == flag or str(tok).startswith(f"{flag}="):
            return True
    return False


def _get_nested_attr(obj, parts):
    cur = obj
    for part in parts:
        if cur is None or not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
    return cur


def _set_nested_config_value(config, path_parts, value):
    if not path_parts:
        return config
    cur = config
    for part in path_parts[:-1]:
        if part not in cur or not isinstance(cur.get(part), dict):
            cur[part] = {}
        cur = cur[part]
    cur[path_parts[-1]] = value
    return config


def _selected_subparser(parser, model_type):
    for action in getattr(parser, "_actions", []):
        choices = getattr(action, "choices", None)
        if isinstance(choices, dict) and model_type in choices:
            return choices[model_type]
    return None


def _apply_explicit_continue_run_cli_overrides_from_parser(config, args, argv, parser):
    model_type = getattr(args, "model_type", None)
    if parser is None or model_type is None:
        return config
    subparser = _selected_subparser(parser, model_type)
    if subparser is None:
        return config
    for group in getattr(subparser, "_action_groups", []):
        title = str(getattr(group, "title", "") or "").strip()
        if not title or title in {"positional arguments", "options"}:
            continue
        group_parts = title.split(".")
        ns = _get_nested_attr(args, group_parts)
        if ns is None:
            continue
        if title == "general":
            config_path_prefix = []
        else:
            config_path_prefix = group_parts
        for action in getattr(group, "_group_actions", []):
            dest = getattr(action, "dest", None)
            if not dest or dest == "help":
                continue
            option_strings = tuple(getattr(action, "option_strings", ()) or ())
            if not option_strings:
                continue
            if not any(_cli_flag_is_set(argv, opt) for opt in option_strings):
                continue
            if not hasattr(ns, dest):
                continue
            value = getattr(ns, dest)
            _set_nested_config_value(config, [*config_path_prefix, dest], value)
    return config


_RLPFN_CONTINUE_RUN_RESUME_SAFE_DEFAULT_KEYS = (
    ("prior", "environment", "reinforce_normalize_advantages"),
    ("prior", "environment", "reinforce_scale_advantages_by_suffix_episode_count"),
    ("prior", "environment", "reinforce_advantage_norm_eps"),
    ("prior", "environment", "reinforce_advantage_norm_clip"),
    ("prior", "environment", "policy_gradient_weight"),
    ("prior", "environment", "reinforce_reward_tanh_c"),
    ("prior", "environment", "reinforce_reward_tanh_bound"),
    ("prior", "environment", "terminal_bonus_tanh_c"),
    ("prior", "environment", "terminal_bonus_scale_min"),
    ("prior", "environment", "terminal_bonus_scale_max"),
    ("prior", "environment", "terminal_reset_count_target"),
    ("prior", "environment", "ctrl_reward_weight"),
    ("prior", "environment", "ctrl_reward_enable_prob"),
    ("prior", "environment", "survival_reward_weight"),
    ("prior", "environment", "survival_reward_enable_prob"),
)

_CONTINUE_RUN_EPHEMERAL_OPTIMIZER_FLAGS = (
    ("--train-profiler-output-path", "train_profiler_output_path"),
    ("--train-gpu-observer-output-path", "train_gpu_observer_output_path"),
    ("--train-gpu-stage-output-path", "train_gpu_stage_output_path"),
    ("--train-kernel-profiler-output-dir", "train_kernel_profiler_output_dir"),
    ("--pg-compile-observe-output-path", "pg_compile_observe_output_path"),
    ("--pg-phase-log-file", "pg_phase_log_file"),
    ("--ppo-pack-output-dir", "ppo_pack_output_dir"),
)


def _get_nested_config_value(config, path_parts):
    cur = config
    for part in path_parts:
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _apply_continue_run_resume_safe_defaults(config, new_defaults, model_type):
    if str(model_type).strip().lower() != "rlpfn":
        return config
    for path_parts in _RLPFN_CONTINUE_RUN_RESUME_SAFE_DEFAULT_KEYS:
        value = _get_nested_config_value(new_defaults, path_parts)
        if value is None:
            continue
        _set_nested_config_value(config, list(path_parts), value)
    return config


def _clear_continue_run_stale_output_paths(config, argv):
    if "optimizer" not in config:
        config["optimizer"] = {}
    for flag, key in _CONTINUE_RUN_EPHEMERAL_OPTIMIZER_FLAGS:
        if _cli_flag_is_set(argv, flag):
            continue
        config["optimizer"][key] = None
    return config


def _apply_rlpfn_anchor_compare_contract_if_requested(config, args):
    requested = bool(getattr(args.orchestration, "rlpfn_anchor_compare_contract", False))
    if not requested:
        return config
    if str(getattr(args, "model_type", "")).strip().lower() != "rlpfn":
        raise ValueError(
            "rlpfn_anchor_compare_contract is only supported for model_type=rlpfn."
        )
    warm_start_from = getattr(args.orchestration, "warm_start_from", None)
    if warm_start_from is None:
        raise ValueError(
            "rlpfn_anchor_compare_contract requires --warm-start-from or an equivalent continue-run checkpoint "
            "because it forces restore_validation_policy_state=True for trusted anchor compare."
        )
    return apply_rlpfn_anchor_compare_contract(config)


def _apply_extra_fast_rwkv_safe_overrides(config, attention_type):
    if str(config[attention_type].get('backbone', 'transformer')).strip().lower() != 'rwkv7':
        return config
    rwkv_head_size = int(config[attention_type].get('rwkv_head_size', 64) or 64)
    emsize = int(config[attention_type].get('emsize', rwkv_head_size) or rwkv_head_size)
    if emsize < rwkv_head_size or (emsize % rwkv_head_size) != 0:
        emsize = int(max(rwkv_head_size, ((emsize + rwkv_head_size - 1) // rwkv_head_size) * rwkv_head_size))
    config[attention_type]['emsize'] = emsize
    config[attention_type]['nhead'] = 1
    return config


def _apply_continue_run_cli_overrides(config, args, argv, parser=None):
    config = _apply_explicit_continue_run_cli_overrides_from_parser(config, args, argv, parser)
    # Continue-run keeps historical config semantics, but explicitly provided
    # CLI safety knobs must override old checkpoint values.
    if _cli_flag_is_set(argv, "--policy-rollout-chunk-size"):
        if "optimizer" not in config:
            config["optimizer"] = {}
        config["optimizer"]["policy_rollout_chunk_size"] = args.optimizer.policy_rollout_chunk_size
    if _cli_flag_is_set(argv, "--pg-tbptt-window"):
        if "optimizer" not in config:
            config["optimizer"] = {}
        config["optimizer"]["pg_tbptt_window"] = args.optimizer.pg_tbptt_window
    if _cli_flag_is_set(argv, "--pg-env-replay-steps"):
        if "optimizer" not in config:
            config["optimizer"] = {}
        config["optimizer"]["pg_env_replay_steps"] = args.optimizer.pg_env_replay_steps
    env_override_flags = (
        ("--pg-one-hop-replay-enabled", "pg_one_hop_replay_enabled"),
        ("--alpha-grad-one-hop-replay-enabled", "alpha_grad_one_hop_replay_enabled"),
        ("--pg-markov-adjacent-replay-enabled", "pg_markov_adjacent_replay_enabled"),
        ("--pg-markov-adjacent-replay-sample-prob", "pg_markov_adjacent_replay_sample_prob"),
        ("--pg-replay-window-depth", "pg_replay_window_depth"),
    )
    for flag, key in env_override_flags:
        if _cli_flag_is_set(argv, flag):
            if "prior" not in config:
                config["prior"] = {}
            if "environment" not in config["prior"]:
                config["prior"]["environment"] = {}
            config["prior"]["environment"][key] = getattr(args.prior.environment, key)
    profiler_flags = (
        ("--train-profiler-enabled", "train_profiler_enabled"),
        ("--train-profiler-output-path", "train_profiler_output_path"),
        ("--train-profiler-wandb", "train_profiler_wandb"),
        ("--train-profiler-ema-alpha", "train_profiler_ema_alpha"),
        ("--train-profiler-warmup-epochs", "train_profiler_warmup_epochs"),
        ("--train-profiler-warmup-batches", "train_profiler_warmup_batches"),
        ("--train-profiler-log-every-batches", "train_profiler_log_every_batches"),
        ("--train-gpu-observer-enabled", "train_gpu_observer_enabled"),
        ("--train-gpu-observer-interval-sec", "train_gpu_observer_interval_sec"),
        ("--train-gpu-observer-output-path", "train_gpu_observer_output_path"),
        ("--train-gpu-stage-output-path", "train_gpu_stage_output_path"),
        ("--train-host-rss-limit-gib", "train_host_rss_limit_gib"),
        ("--train-host-rss-limit-poll-interval-sec", "train_host_rss_limit_poll_interval_sec"),
        ("--train-host-rss-limit-try-rlimit-as", "train_host_rss_limit_try_rlimit_as"),
        ("--train-kernel-profiler-enabled", "train_kernel_profiler_enabled"),
        ("--train-kernel-profiler-output-dir", "train_kernel_profiler_output_dir"),
        ("--train-kernel-profiler-wait-steps", "train_kernel_profiler_wait_steps"),
        ("--train-kernel-profiler-warmup-steps", "train_kernel_profiler_warmup_steps"),
        ("--train-kernel-profiler-active-steps", "train_kernel_profiler_active_steps"),
        ("--train-kernel-profiler-repeat-steps", "train_kernel_profiler_repeat_steps"),
        ("--train-kernel-profiler-record-shapes", "train_kernel_profiler_record_shapes"),
        ("--train-kernel-profiler-profile-memory", "train_kernel_profiler_profile_memory"),
        ("--train-kernel-profiler-with-stack", "train_kernel_profiler_with_stack"),
        ("--train-kernel-profiler-with-flops", "train_kernel_profiler_with_flops"),
        ("--train-kernel-profiler-log-every-batches", "train_kernel_profiler_log_every_batches"),
        ("--train-kernel-profiler-export-trace", "train_kernel_profiler_export_trace"),
        ("--train-kernel-profiler-summary-top-k", "train_kernel_profiler_summary_top_k"),
        ("--pg-oom-fail-fast", "pg_oom_fail_fast"),
        ("--pg-compile-observe-recompiles", "pg_compile_observe_recompiles"),
        ("--pg-compile-observe-log-every-batches", "pg_compile_observe_log_every_batches"),
        ("--pg-compile-observe-output-path", "pg_compile_observe_output_path"),
        ("--pg-compile-observe-reset-after-warmup", "pg_compile_observe_reset_after_warmup"),
        ("--pg-phase-log-every-batches", "pg_phase_log_every_batches"),
        ("--pg-phase-log-file", "pg_phase_log_file"),
    )
    for flag, key in profiler_flags:
        if _cli_flag_is_set(argv, flag):
            if "optimizer" not in config:
                config["optimizer"] = {}
            config["optimizer"][key] = getattr(args.optimizer, key)
    return config


def main(argv, extra_config=None):
    # extra config is used for testing purposes only
    # this is the generic entry point for training any model, so it has A LOT of options
    _set_fit_model_heartbeat_stage("main_parse_args")
    parser = make_model_level_argparser()
    args = parser.parse_args(args=argv or ['--help'])
    model = None
    if hasattr(args, "linear_attention") and hasattr(args.linear_attention, "model"):
        model = args.linear_attention.model
    _set_fit_model_heartbeat_stage("load_default_config")
    config = get_model_default_config(args.model_type, model)

    _set_fit_model_heartbeat_stage("init_device")
    device, rank, num_gpus = init_device(args.general.gpu_id, args.general.use_cpu)
    # handle syne-tune restarts
    _set_fit_model_heartbeat_stage("checkpoint_orchestration")
    orchestration = args.orchestration
    orchestration.base_path, orchestration.continue_run, orchestration.warm_start_from, report = synetune_handle_checkpoint(orchestration)

    if orchestration.create_new_run and not orchestration.continue_run:
        raise ValueError("Specifying create-new-run makes no sense when not continuing run")
    base_path = orchestration.base_path
    torch.set_num_threads(24)
    for group_name in vars(args):
        _set_fit_model_heartbeat_stage("merge_cli_config")
        if group_name == "model_type":
            # the only non-group argument from the top level parser
            config['model_type'] = args.model_type
            continue
        if group_name not in config:
            config[group_name] = {}
        for k, v in vars(getattr(args, group_name)).items():
            if isinstance(v, Namespace):
                if k not in config[group_name]:
                    config[group_name][k] = {}
                # FIXME we only allow one level of nesting, we should do recursion here really.
                config[group_name][k].update(vars(v))
            else:
                config[group_name][k] = v
        config[group_name].update()
    if args.orchestration.seed_everything_value is not None:
        _set_fit_model_heartbeat_stage("seed_everything")
        _seed_python_numpy_torch(int(args.orchestration.seed_everything_value))
    elif args.orchestration.seed_everything:
        _set_fit_model_heartbeat_stage("seed_everything")
        import lightning as L
        L.seed_everything(42)

    if 'transformer' in config:
        attention_type = 'transformer'
    elif 'linear_attention' in config:
        attention_type = 'linear_attention'
    else:
        raise ValueError(f"Unknown attention type")

    # promote general group to top level
    config.update(config.pop('general'))
    config['num_gpus'] = 1
    config['device'] = device

    if not config[attention_type]['classification_task']:
        print('Setting regression parameters')
        config['prior']['classification']['max_num_classes'] = 0
        config[attention_type]['y_encoder'] = 'linear'
        if 'mothernet' in config:
            config['mothernet']['decoder_type'] = 'average'

    warm_start_weights = orchestration.warm_start_from
    config[attention_type]['nhead'] = config[attention_type]['emsize'] // 128

    config['dataloader']['num_steps'] = config['dataloader']['num_steps'] or 1024 * \
        64 // config['dataloader']['batch_size'] // config['optimizer']['aggregate_k_gradients']

    if args.orchestration.extra_fast_test:
        config['prior']['n_samples'] = 2 * 16
        config[attention_type]['nhead'] = 1
        _apply_extra_fast_rwkv_safe_overrides(config, attention_type)

    if extra_config is not None:
        _set_fit_model_heartbeat_stage("apply_extra_config")
        update_config(config, extra_config)
        if args.orchestration.extra_fast_test:
            _apply_extra_fast_rwkv_safe_overrides(config, attention_type)

    host_rss_guard = None
    host_rss_guard_status = None
    _set_fit_model_heartbeat_stage("host_rss_guard_setup")
    try:
        host_rss_guard, host_rss_guard_status = install_host_rss_limit_guard(
            limit_gib=config["optimizer"].get("train_host_rss_limit_gib", None),
            poll_interval_sec=config["optimizer"].get("train_host_rss_limit_poll_interval_sec", 0.02),
            try_rlimit_as=config["optimizer"].get("train_host_rss_limit_try_rlimit_as", False),
        )
    except Exception as exc:
        host_rss_guard = None
        host_rss_guard_status = {
            "enabled": False,
            "limit_gib": None,
            "poll_interval_sec": None,
            "try_rlimit_as": bool(config["optimizer"].get("train_host_rss_limit_try_rlimit_as", False)),
            "rlimit_as": {"requested": False, "applied": False, "reason": f"guard_init_failed: {exc}"},
        }
    if isinstance(host_rss_guard_status, dict) and bool(host_rss_guard_status.get("enabled", False)):
        rlimit_status = host_rss_guard_status.get("rlimit_as", {})
        rlimit_msg = "disabled"
        if isinstance(rlimit_status, dict) and bool(host_rss_guard_status.get("try_rlimit_as", False)):
            if bool(rlimit_status.get("applied", False)):
                rlimit_msg = "applied"
            else:
                rlimit_msg = f"skipped({rlimit_status.get('reason', 'unknown')})"
        print(
            "Host RSS guard:",
            f"limit_gib={float(host_rss_guard_status['limit_gib']):.2f}",
            f"poll_interval_sec={float(host_rss_guard_status['poll_interval_sec']):.3f}",
            f"try_rlimit_as={bool(host_rss_guard_status.get('try_rlimit_as', False))}",
            f"rlimit_as={rlimit_msg}",
        )
    elif (
        isinstance(host_rss_guard_status, dict)
        and config["optimizer"].get("train_host_rss_limit_gib", None) is not None
        and not bool(host_rss_guard_status.get("sensor_available", True))
    ):
        print(
            "[host-rss-limit] disabled because current RSS sensor is unavailable on this platform/runtime."
        )

    save_every = orchestration.save_every

    model_state, optimizer_state, scheduler = None, None, None
    if warm_start_weights is not None:
        _set_fit_model_heartbeat_stage("warm_start_load_checkpoint")
        # PyTorch 2.6 changed torch.load() default to weights_only=True.
        # Our training checkpoints store config / optimizer / scheduler state,
        # so warm-start loading must opt back into full checkpoint loading.
        loaded_states = torch.load(warm_start_weights, map_location='cpu', weights_only=False)
        if isinstance(loaded_states, (list, tuple)) and len(loaded_states) >= 5:
            model_state, old_optimizer_state, old_scheduler, old_config = loaded_states[:4]
        else:
            model_state, old_optimizer_state, old_scheduler, old_config = loaded_states
        module_prefix = 'module.'
        model_state = {k.replace(module_prefix, ''): v for k, v in model_state.items()}
        if args.orchestration.continue_run:
            config = old_config
            # Forward compatibility: keep old run semantics, but fill newly
            # introduced defaults so safety knobs (e.g. rollout chunking) exist.
            new_defaults = get_model_default_config(args.model_type, model)
            _merge_missing_keys(config, new_defaults)
            _apply_continue_run_resume_safe_defaults(config, new_defaults, args.model_type)
            # we want to overwrite specific parts of the old config with current values
            config['device'] = device
            config['orchestration']['warm_start_from'] = warm_start_weights
            config['orchestration']['continue_run'] = True
            optimizer_state = old_optimizer_state
            config['orchestration']['stop_after_epochs'] = args.orchestration.stop_after_epochs
            if not args.orchestration.restart_scheduler:
                scheduler = old_scheduler
            _apply_continue_run_cli_overrides(config, args, argv, parser=parser)
            _clear_continue_run_stale_output_paths(config, argv)
            if _cli_flag_is_set(argv, "--policy-rollout-chunk-size"):
                print(
                    "[continue-run-override] policy_rollout_chunk_size set from CLI to",
                    config["optimizer"]["policy_rollout_chunk_size"],
                )
            if _cli_flag_is_set(argv, "--pg-tbptt-window"):
                print(
                    "[continue-run-override] pg_tbptt_window set from CLI to",
                    config["optimizer"]["pg_tbptt_window"],
                )
            if _cli_flag_is_set(argv, "--pg-env-replay-steps"):
                print(
                    "[continue-run-override] pg_env_replay_steps set from CLI to",
                    config["optimizer"]["pg_env_replay_steps"],
                )
        else:
            print("WARNING warm starting with new settings")
            compare_dicts(config, old_config)

    # Continue-run restores the checkpoint config above, so apply extra_config
    # again here to make caller-supplied overrides authoritative for resumed
    # training as well.
    if extra_config is not None:
        update_config(config, extra_config)
        if args.orchestration.extra_fast_test:
            _apply_extra_fast_rwkv_safe_overrides(config, attention_type)

    rwkv_replay_chunk_override = os.environ.get("TICL_RWKV_SEQUENCE_REPLAY_BATCH_CHUNK_SIZE", "").strip()
    if rwkv_replay_chunk_override:
        config.setdefault(attention_type, {})["rwkv_sequence_replay_batch_chunk_size"] = int(
            rwkv_replay_chunk_override
        )
        print(
            "[env-override] rwkv_sequence_replay_batch_chunk_size set to",
            config[attention_type]["rwkv_sequence_replay_batch_chunk_size"],
        )
    rwkv_replay_token_budget_override = os.environ.get("TICL_RWKV_SEQUENCE_REPLAY_TOKEN_BUDGET", "").strip()
    if rwkv_replay_token_budget_override:
        config.setdefault(attention_type, {})["rwkv_sequence_replay_token_budget"] = int(
            rwkv_replay_token_budget_override
        )
        print(
            "[env-override] rwkv_sequence_replay_token_budget set to",
            config[attention_type]["rwkv_sequence_replay_token_budget"],
        )

    config = _apply_rlpfn_anchor_compare_contract_if_requested(config, args)

    if config['orchestration']['detect_anomaly']:
        print("ENABLING GRADIENT DEBUGGING (detect-anomaly)! Don't use for training.")
        torch.autograd.set_detect_anomaly(True)

    _set_fit_model_heartbeat_stage("resolve_model_string")
    model_string = get_model_string(config, num_gpus, device, parser)
    pg_phase_log_file_default = os.path.join(base_path, "log", f"{model_string}.log")
    if "optimizer" not in config:
        config["optimizer"] = {}
    pg_phase_log_file_cfg = config["optimizer"].get("pg_phase_log_file", None)
    if pg_phase_log_file_cfg is None:
        # PPO phase heartbeat is diagnostic progress output, not a training
        # semantic artifact.  Keep it on stdout/tmux by default so it does not
        # pollute the persistent metric log file.  Users can still opt in with
        # --pg-phase-log-file when they explicitly want a heartbeat file.
        config["optimizer"]["pg_phase_log_file"] = ""
    pg_phase_log_file_effective = config["optimizer"].get("pg_phase_log_file", None)
    if pg_phase_log_file_effective is not None and str(pg_phase_log_file_effective).strip() != "":
        pg_phase_log_file_safe = enforce_path_filename_limit(pg_phase_log_file_effective)
        if str(pg_phase_log_file_safe) != str(pg_phase_log_file_effective):
            print(
                "[filename-limit] pg_phase_log_file basename was truncated to fit filesystem limits:"
                f" {pg_phase_log_file_safe}"
            )
        config["optimizer"]["pg_phase_log_file"] = pg_phase_log_file_safe
    if config["optimizer"].get("pg_phase_log_every_batches", None) is None:
        config["optimizer"]["pg_phase_log_every_batches"] = 1
    if (
        str(config.get("model_type", "")).strip().lower() == "rlpfn"
        and str(config["optimizer"].get("rl_objective", "")).strip().lower() == "ppo"
        and bool(config["optimizer"].get("ppo_trusted_pack_runner_required", False))
        and (
            config["optimizer"].get("ppo_pack_output_dir", None) is None
            or str(config["optimizer"].get("ppo_pack_output_dir", "")).strip() == ""
        )
    ):
        pack_output_dir = os.path.join(base_path, "ppo_pack_runner", model_string)
        pack_output_dir_safe = enforce_path_filename_limit(pack_output_dir)
        if str(pack_output_dir_safe) != str(pack_output_dir):
            print(
                "[filename-limit] ppo_pack_output_dir basename was truncated to fit filesystem limits:"
                f" {pack_output_dir_safe}"
            )
        config["optimizer"]["ppo_pack_output_dir"] = pack_output_dir_safe
    _set_fit_model_heartbeat_stage("training_callback_setup")
    save_callback = make_training_callback(
        save_every, 
        model_string, 
        base_path, 
        report, 
        config, 
        orchestration.use_mlflow,
        orchestration.st_checkpoint_dir, 
        classification=config[attention_type]['classification_task'], 
        validate=orchestration.validate
    )

    mlflow_hostname = os.environ.get("MLFLOW_HOSTNAME", None)
    if orchestration.use_wandb:
        import wandb
        from ticl.environment import WANDB_INFO
        wandb_data, flatten_key_dict = flatten_dict(config, track_keys=True)
        wandb_config = {k: v for k, v in wandb_data.items() if k not in ['wallclock_times', 'losses', 'learning_rates']}
        wandb.init(
            dir=WANDB_INFO['dir'],
            project=WANDB_INFO['project'],
            entity=WANDB_INFO['entity'],
            id=model_string,
            config=wandb_config,
        )

    try:
        if (not orchestration.use_mlflow) or mlflow_hostname is None:
            print("Not logging run with mlflow, set MLFLOW_HOSTNAME environment to variable enable mlflow.")
            _set_fit_model_heartbeat_stage("get_model_train_start")
            total_loss, model, dl, epoch = get_model(
                config, 
                device, 
                should_train=True,
                verbose=1, 
                epoch_callback=save_callback, 
                model_state=model_state,
                optimizer_state=optimizer_state, 
                scheduler=scheduler,
                load_model_strict=orchestration.continue_run or orchestration.load_strict
            )
            _set_fit_model_heartbeat_stage("get_model_train_done")
        else:
            print(f"Logging run with mlflow at host {mlflow_hostname}")
            _set_fit_model_heartbeat_stage("mlflow_setup")
            mlflow.set_tracking_uri(f"http://{mlflow_hostname}:5000")

            tries = 0
            while tries < 5:
                try:
                    mlflow.set_experiment(orchestration.experiment)
                    break
                except:
                    tries += 1
                    print(f"Failed to set experiment, retrying {tries}/5")
                    time.sleep(5)

            if orchestration.continue_run and not orchestration.create_new_run:
                # find run id via mlflow
                run_ids = mlflow.search_runs(filter_string=f"attribute.run_name='{model_string}'")['run_id']
                if len(run_ids) > 1:
                    raise ValueError(f"Found more than one run with name {model_string}")
                if len(run_ids) < 1:
                    raise ValueError(f"Found no run with name {model_string}")
                run_id = run_ids.iloc[0]
                run_args = {'run_id': run_id}

            else:
                run_args = {'run_name': model_string}

            path = os.path.dirname(os.path.abspath(__file__))
            run_args['tags'] = {'mlflow.source.git.commit': Repo(path, search_parent_directories=True).head.object.hexsha}

            with mlflow.start_run(**run_args):
                _set_fit_model_heartbeat_stage("get_model_train_start")
                mlflow.log_param('hostname', socket.gethostname())
                mlflow.log_params({k: v for k, v in flatten_dict(config).items() if k not in ['wallclock_times', 'losses', 'learning_rates']})
                total_loss, model, dl, epoch = get_model(
                    config, 
                    device, 
                    should_train=True, 
                    verbose=1, 
                    epoch_callback=save_callback, 
                    model_state=model_state,
                    optimizer_state=optimizer_state, 
                    scheduler=scheduler,
                    load_model_strict=orchestration.continue_run or orchestration.load_strict
                )
                _set_fit_model_heartbeat_stage("get_model_train_done")
    finally:
        _set_fit_model_heartbeat_stage("shutdown")
        if host_rss_guard is not None:
            host_rss_guard.stop()

    if rank == 0:
        save_callback(model, None, None, "on_exit")
    return {'loss': total_loss, 'model': model, 'dataloader': dl,
            'config': config, 'base_path': base_path,
            'model_string': model_string, 'epoch': epoch}


if __name__ == "__main__":
    main(sys.argv[1:])
