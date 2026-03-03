import math
import os
import time, wandb
import random
import gc
from contextlib import nullcontext

import torch
import numpy as np
from torch import nn
from tqdm import tqdm
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.checkpoint import checkpoint

import ticl.utils as utils
from ticl.utils import ExponentialLR, ReduceLROnSpike, init_dist
from ticl.profiling import TrainProfiler, TrainProfilerConfig
from ticl.gpu_observer import GPUProcessObserver
from ticl.kernel_profiling import TrainKernelProfiler, TrainKernelProfilerConfig

import pdb


def _has_nonfinite_gradients(model, device):
    has_nonfinite = torch.zeros((), dtype=torch.bool, device=device)
    for p in model.parameters():
        g = p.grad
        if g is not None:
            has_nonfinite = has_nonfinite | (~torch.isfinite(g).all())
    return bool(has_nonfinite.item())


def _extract_optimizer_step(optimizer):
    # Return a representative optimizer step counter from the first populated
    # parameter state. Useful for observability in low-loss PG runs.
    try:
        for group in optimizer.param_groups:
            for p in group.get("params", []):
                state = optimizer.state.get(p, None)
                if not state:
                    continue
                step = state.get("step", None)
                if step is None:
                    continue
                if torch.is_tensor(step):
                    return float(step.detach().item())
                return float(step)
    except Exception:
        return None
    return None


def _resolve_environment_prior(prior):
    seen = set()
    cur = prior
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if hasattr(cur, "rollout_policy_gradient_loss"):
            return cur
        cur = getattr(cur, "base_prior", None)
    return None


def _saved_tensors_cpu_offload_context(enabled: bool, pin_memory: bool):
    if not bool(enabled):
        return nullcontext()
    graph_mod = getattr(torch.autograd, "graph", None)
    if graph_mod is None or not hasattr(graph_mod, "save_on_cpu"):
        return nullcontext()
    return graph_mod.save_on_cpu(pin_memory=bool(pin_memory))


def _is_oom_exception(exc: BaseException) -> bool:
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    accel_err_cls = getattr(torch, "AcceleratorError", None)
    if accel_err_cls is not None and isinstance(exc, accel_err_cls):
        return True
    msg = str(exc).strip().lower()
    if not msg:
        return False
    oom_markers = (
        "out of memory",
        "cudaerrormemoryallocation",
        "cudnn_status_alloc_failed",
        "cublas_status_alloc_failed",
        "hip out of memory",
        "xpu out of memory",
        "mps backend out of memory",
    )
    return any(marker in msg for marker in oom_markers)


def _resolve_policy_autocast_dtype(device):
    device_obj = device if isinstance(device, torch.device) else torch.device(str(device))
    if device_obj.type != "cuda":
        return None
    dtype_env = str(os.environ.get("TICL_POLICY_AUTOCAST_DTYPE", "auto")).strip().lower()
    if dtype_env in {"fp16", "float16", "half", "16"}:
        return torch.float16
    if dtype_env in {"bf16", "bfloat16"}:
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    # Stability-first default for policy rollout: prefer bf16 on supported GPUs
    # (wider exponent than fp16), fallback to fp16 otherwise.
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def _amp_autocast_context(scaler, *, device, dtype):
    if scaler is None:
        return nullcontext()
    try:
        device_type = torch.device(str(device)).type
    except Exception:
        device_type = "cuda"
    if dtype is None:
        return autocast(device_type=device_type)
    return autocast(device_type=device_type, dtype=dtype)


def _set_policy_inner_recompute_attn(model, enabled: bool):
    """
    Toggle per-layer forward_step attention recomputation used by policy rollout.
    Returns (changed_layers, total_layers_with_flag).
    """
    model_ref = model.module if hasattr(model, "module") else model
    encoder = getattr(model_ref, "transformer_encoder", None)
    layers = getattr(encoder, "layers", None)
    if layers is None:
        return 0, 0
    target = bool(enabled)
    changed = 0
    total = 0
    for layer in layers:
        if not hasattr(layer, "recompute_attn"):
            continue
        total += 1
        if bool(layer.recompute_attn) != target:
            layer.recompute_attn = target
            changed += 1
    return changed, total


def _snapshot_policy_inner_recompute_attn(model):
    model_ref = model.module if hasattr(model, "module") else model
    encoder = getattr(model_ref, "transformer_encoder", None)
    layers = getattr(encoder, "layers", None)
    if layers is None:
        return []
    snapshot = []
    for layer in layers:
        if hasattr(layer, "recompute_attn"):
            snapshot.append((layer, bool(layer.recompute_attn)))
    return snapshot


def _restore_policy_inner_recompute_attn(snapshot):
    for layer, state in snapshot:
        layer.recompute_attn = bool(state)


def _fit_last_dim(x, target_dim):
    d = int(x.shape[-1])
    t = int(target_dim)
    if d == t:
        return x
    if d > t:
        return x[..., :t]
    pad_shape = list(x.shape[:-1]) + [t - d]
    pad = torch.zeros(pad_shape, device=x.device, dtype=x.dtype)
    return torch.cat([x, pad], dim=-1)


def _fit_action_dim(a, action_dim):
    d = int(a.shape[-1])
    t = int(action_dim)
    if d == t:
        return a
    if d > t:
        return a[..., :t]
    repeats = (t + d - 1) // max(1, d)
    tiled = a.repeat(1, repeats)
    return tiled[..., :t]


def _infer_batch_shape_for_profile(data, targets):
    # Returns (batch_size, n_samples) as logical training workload counters.
    batch_size = None
    n_samples = None
    try:
        if isinstance(data, tuple):
            for item in data:
                if torch.is_tensor(item) and item.ndim >= 2:
                    n_samples = int(item.shape[0])
                    batch_size = int(item.shape[1])
                    break
        elif torch.is_tensor(data) and data.ndim >= 2:
            n_samples = int(data.shape[0])
            batch_size = int(data.shape[1])
    except Exception:
        batch_size = None
        n_samples = None

    if batch_size is None or n_samples is None:
        try:
            if torch.is_tensor(targets) and targets.ndim >= 2:
                n_samples = int(targets.shape[0])
                batch_size = int(targets.shape[1])
            elif torch.is_tensor(targets) and targets.ndim == 1:
                n_samples = int(targets.shape[0])
                batch_size = 1
        except Exception:
            pass

    return batch_size, n_samples


def _maybe_emit_profile_interval(
    epoch_profiler,
    *,
    epoch_start_time,
    epoch_idx,
    batch_idx,
    train_profiler_log_every_batches,
    verbose=False,
):
    if epoch_profiler is None or not epoch_profiler.enabled():
        return None
    every = int(train_profiler_log_every_batches)
    if every <= 0:
        return None
    if (int(batch_idx) + 1) % every != 0:
        return None
    if epoch_start_time is None:
        return None

    record = epoch_profiler.emit_interval(
        wall_time_sec=float(max(0.0, time.time() - float(epoch_start_time))),
        gpu_time_sec=0.0,
        peak_alloc_gib=None,
        peak_reserved_gib=None,
        batch_index_end=int(batch_idx) + 1,
    )
    if (
        record is not None
        and wandb.run
        and bool(epoch_profiler.config.log_to_wandb)
    ):
        wandb.log({f"profile_interval/{k}": v for k, v in record.items()})
    if verbose and (record is not None):
        print(
            f" [profile-interval] epoch={int(epoch_idx) if epoch_idx is not None else -1} "
            f"batch={int(batch_idx) + 1} "
            f"steps/s={record['batches_per_sec']:.2f} "
            f"samples/s={record['samples_per_sec']:.2f} "
            f"tokens/s={record['tokens_per_sec']:.2f}"
        )
    return record


def _format_gpu_stage_summary(prefix: str, rec: dict) -> str:
    if not rec:
        return ""
    samples = int(rec.get("samples", 0) or 0)
    util_avg = rec.get("gpu_util_avg", None)
    util_max = rec.get("gpu_util_max", None)
    proc_mem_avg = rec.get("process_mem_avg_mib", None)
    proc_mem_max = rec.get("process_mem_max_mib", None)
    mem_share_avg = rec.get("process_mem_share_avg", None)
    proc_sm_avg = rec.get("process_sm_util_avg", None)

    def _fmt(v, digits=2):
        return "na" if v is None else f"{float(v):.{digits}f}"

    return (
        f"{prefix} samples={samples} "
        f"gpu_util_avg={_fmt(util_avg)} gpu_util_max={_fmt(util_max)} "
        f"proc_mem_avg_mib={_fmt(proc_mem_avg, 1)} proc_mem_max_mib={_fmt(proc_mem_max, 1)} "
        f"proc_mem_share_avg={_fmt(mem_share_avg)} "
        f"proc_sm_util_avg={_fmt(proc_sm_avg)}"
    )


def _maybe_step_kernel_profiler(kernel_profiler, *, epoch_idx, batch_idx, verbose=False):
    if kernel_profiler is None or not kernel_profiler.enabled():
        return None
    record = kernel_profiler.step(epoch=int(epoch_idx), batch=int(batch_idx))
    if record is None:
        return None
    top_cuda_op = record.get("top_cuda_op")
    top_cuda_time = record.get("top_cuda_self_time_us")
    top_cpu_op = record.get("top_cpu_op")
    top_cpu_time = record.get("top_cpu_self_time_us")
    sync_cpu_share = record.get("sync_self_cpu_share", 0.0)
    phase_ops = record.get("phase_ops", []) or []
    top_phase = None
    if isinstance(phase_ops, list) and phase_ops:
        top_phase = max(
            phase_ops,
            key=lambda p: float(p.get("cuda_time_us", 0.0) or 0.0),
        )
    if verbose:
        msg = (
            f"[kernel-profile] epoch={int(epoch_idx)} batch={int(batch_idx)} "
            f"top_cuda_op={top_cuda_op} top_cuda_self_us={float(top_cuda_time):.1f} "
            f"top_cpu_op={top_cpu_op} top_cpu_self_us={float(top_cpu_time):.1f} "
            f"sync_cpu_share={float(sync_cpu_share):.3f}"
        )
        if isinstance(top_phase, dict) and top_phase.get("name") is not None:
            msg += (
                f" top_phase={top_phase.get('name')} "
                f"top_phase_cuda_us={float(top_phase.get('cuda_time_us', 0.0) or 0.0):.1f}"
            )
        print(msg)
    if wandb.run is not None:
        wandb.log(
            {
                "kernel_profile/epoch": int(epoch_idx),
                "kernel_profile/batch": int(batch_idx),
                "kernel_profile/top_cuda_self_time_us": float(top_cuda_time),
                "kernel_profile/top_cpu_self_time_us": float(top_cpu_time),
            }
        )
    return record


def _compile_policy_forward_step(
    policy_forward_step,
    *,
    enabled: bool,
    backend: str,
    mode: str,
    fullgraph: bool,
    dynamic: bool,
):
    if not bool(enabled):
        return policy_forward_step, False
    if not hasattr(torch, "compile"):
        return policy_forward_step, False
    try:
        compiled = torch.compile(
            policy_forward_step,
            backend=str(backend),
            mode=str(mode),
            fullgraph=bool(fullgraph),
            dynamic=bool(dynamic),
        )
        return compiled, True
    except Exception as e:
        print(f"[pg-compile-warn] failed to compile forward_policy_step, fallback to eager: {e}")
        return policy_forward_step, False


def _build_policy_step_fn(
    model,
    num_features,
    max_cache_len=None,
    kv_cache_mode: str = "immutable",
    kv_cache_page_size=None,
    allow_grad_mutable_cache=False,
    allow_grad_inplace_paged_cache=False,
    pg_torch_compile=False,
    pg_torch_compile_backend="inductor",
    pg_torch_compile_mode="reduce-overhead",
    pg_torch_compile_fullgraph=False,
    pg_torch_compile_dynamic=False,
):
    model_ref = model.module if hasattr(model, "module") else model
    if not hasattr(model_ref, "forward_policy_step"):
        raise ValueError("policy_gradient objective requires model.forward_policy_step")

    num_features = int(num_features)
    policy_forward_step = model_ref.forward_policy_step
    policy_forward_step, compile_active = _compile_policy_forward_step(
        policy_forward_step,
        enabled=bool(pg_torch_compile),
        backend=str(pg_torch_compile_backend),
        mode=str(pg_torch_compile_mode),
        fullgraph=bool(pg_torch_compile_fullgraph),
        dynamic=bool(pg_torch_compile_dynamic),
    )
    if compile_active:
        print(
            "[pg-compile] enabled forward_policy_step compile "
            f"(backend={pg_torch_compile_backend}, mode={pg_torch_compile_mode}, "
                f"fullgraph={bool(pg_torch_compile_fullgraph)}, dynamic={bool(pg_torch_compile_dynamic)})"
        )

    split_policy_forward_step = None
    split_compile_active = False
    split_obs_slot_dim = None
    split_action_slot_dim = None
    model_encoder = getattr(model_ref, "encoder", None)
    model_x_encoder_type = str(getattr(model_ref, "x_encoder_type", "")).strip().lower()
    split_fastpath_available = (
        model_x_encoder_type == "split_obs_action"
        and hasattr(model_ref, "forward_policy_step_split")
        and model_encoder is not None
        and hasattr(model_encoder, "obs_dim")
        and hasattr(model_encoder, "action_dim")
    )
    if split_fastpath_available:
        split_policy_forward_step = model_ref.forward_policy_step_split
        split_policy_forward_step, split_compile_active = _compile_policy_forward_step(
            split_policy_forward_step,
            enabled=bool(pg_torch_compile),
            backend=str(pg_torch_compile_backend),
            mode=str(pg_torch_compile_mode),
            fullgraph=bool(pg_torch_compile_fullgraph),
            dynamic=bool(pg_torch_compile_dynamic),
        )
        if split_compile_active:
            print(
                "[pg-compile] enabled forward_policy_step_split compile "
                f"(backend={pg_torch_compile_backend}, mode={pg_torch_compile_mode}, "
                f"fullgraph={bool(pg_torch_compile_fullgraph)}, dynamic={bool(pg_torch_compile_dynamic)})"
            )
        split_obs_slot_dim = int(max(0, int(model_encoder.obs_dim) - 2))
        split_action_slot_dim = int(model_encoder.action_dim)

    # Reuse token buffers across rollout steps to avoid per-step cat/pad
    # allocations in the hot loop.
    token_buf = {
        "x": None,
        "y": None,
        "batch_size": None,
        "dtype": None,
        "device": None,
    }

    def policy_step_fn(obs_t, action_t, reward_t, reward_mask_t, cache, step_idx, env_info):
        nonlocal policy_forward_step, compile_active
        nonlocal split_policy_forward_step, split_compile_active
        del step_idx
        obs_slot_dim = int(env_info["obs_slot_dim"])
        action_slot_dim = int(env_info["action_slot_dim"])
        action_dim = int(env_info["action_dim"])
        batch_size = int(obs_t.shape[0])

        # Keep policy-gradient semantics while preventing reward-token history
        # from being retained through PFN inputs.
        reward_scalar = reward_t.reshape(batch_size).detach().to(dtype=obs_t.dtype)
        reward_mask = reward_mask_t.reshape(batch_size).detach().to(dtype=obs_t.dtype)

        use_split_fastpath = (
            bool(split_fastpath_available)
            and split_policy_forward_step is not None
            and split_obs_slot_dim is not None
            and split_action_slot_dim is not None
            and obs_slot_dim == int(split_obs_slot_dim)
            and action_slot_dim == int(split_action_slot_dim)
        )

        def _call_split_forward_step():
            return split_policy_forward_step(
                obs_t,
                action_t,
                reward_scalar.reshape(batch_size, 1),
                reward_mask.reshape(batch_size, 1),
                kv_cache=cache,
                max_cache_len=max_cache_len,
                kv_cache_mode=kv_cache_mode,
                kv_cache_page_size=kv_cache_page_size,
                allow_grad_mutable_cache=allow_grad_mutable_cache,
                allow_grad_inplace_paged_cache=allow_grad_inplace_paged_cache,
            )

        if use_split_fastpath:
            if split_compile_active:
                try:
                    compiler_mod = getattr(torch, "compiler", None)
                    if compiler_mod is not None and hasattr(compiler_mod, "cudagraph_mark_step_begin"):
                        compiler_mod.cudagraph_mark_step_begin()
                    out, kv_cache = _call_split_forward_step()
                except Exception as e:
                    print(f"[pg-compile-warn] runtime compile failure on split policy step, fallback to eager: {e}")
                    split_compile_active = False
                    split_policy_forward_step = model_ref.forward_policy_step_split
                    out, kv_cache = _call_split_forward_step()
            else:
                out, kv_cache = _call_split_forward_step()
            action_raw = out.squeeze(0)
            action_next = _fit_action_dim(action_raw, action_dim)
            return action_next, kv_cache

        reuse_token_buf = not torch.is_grad_enabled()
        if reuse_token_buf:
            needs_realloc = (
                token_buf["x"] is None
                or token_buf["y"] is None
                or token_buf["batch_size"] != batch_size
                or token_buf["dtype"] != obs_t.dtype
                or token_buf["device"] != obs_t.device
            )
            if needs_realloc:
                token_buf["x"] = torch.zeros(
                    (1, batch_size, num_features),
                    device=obs_t.device,
                    dtype=obs_t.dtype,
                )
                token_buf["y"] = torch.zeros(
                    (1, batch_size),
                    device=obs_t.device,
                    dtype=obs_t.dtype,
                )
                token_buf["batch_size"] = batch_size
                token_buf["dtype"] = obs_t.dtype
                token_buf["device"] = obs_t.device
            x_token = token_buf["x"]
            y_token = token_buf["y"]
        else:
            x_token = torch.zeros(
                (1, batch_size, num_features),
                device=obs_t.device,
                dtype=obs_t.dtype,
            )
            y_token = reward_scalar.reshape(1, batch_size).clone()
        x_row = x_token[0]
        x_row.zero_()

        obs_copy = int(min(obs_t.shape[-1], obs_slot_dim, num_features))
        if obs_copy > 0:
            x_row[:, :obs_copy] = obs_t[:, :obs_copy]
        if obs_slot_dim < num_features:
            x_row[:, obs_slot_dim] = reward_scalar
        if (obs_slot_dim + 1) < num_features:
            x_row[:, obs_slot_dim + 1] = reward_mask
        action_write_start = obs_slot_dim + 2
        if action_write_start < num_features:
            action_copy = int(min(action_t.shape[-1], action_slot_dim, num_features - action_write_start))
            if action_copy > 0:
                x_row[:, action_write_start: action_write_start + action_copy] = action_t[:, :action_copy]
        if reuse_token_buf:
            y_token[0].copy_(reward_scalar)

        def _call_forward_step():
            return policy_forward_step(
                x_token,
                y_token,
                kv_cache=cache,
                max_cache_len=max_cache_len,
                kv_cache_mode=kv_cache_mode,
                kv_cache_page_size=kv_cache_page_size,
                allow_grad_mutable_cache=allow_grad_mutable_cache,
                allow_grad_inplace_paged_cache=allow_grad_inplace_paged_cache,
            )

        if compile_active:
            try:
                compiler_mod = getattr(torch, "compiler", None)
                if compiler_mod is not None and hasattr(compiler_mod, "cudagraph_mark_step_begin"):
                    compiler_mod.cudagraph_mark_step_begin()
                out, kv_cache = _call_forward_step()
            except Exception as e:
                print(f"[pg-compile-warn] runtime compile failure, fallback to eager: {e}")
                compile_active = False
                policy_forward_step = model_ref.forward_policy_step
                out, kv_cache = _call_forward_step()
        else:
            out, kv_cache = _call_forward_step()
        action_raw = out.squeeze(0)
        action_next = _fit_action_dim(action_raw, action_dim)
        return action_next, kv_cache

    return policy_step_fn


def _capture_rng_state(device):
    device_obj = device if isinstance(device, torch.device) else torch.device(str(device))
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if device_obj.type == "cuda" and torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state(device=device_obj)
        state["cuda_device"] = device_obj
    return state


def _restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state:
        torch.cuda.set_rng_state(state["torch_cuda"], device=state["cuda_device"])


def _compute_policy_rollout_chunk_loss(
    env_prior,
    policy_step_fn,
    *,
    batch_size,
    n_samples,
    num_features,
    device,
    single_eval_pos,
    collect_x,
    policy_rollout_checkpoint=False,
    policy_rollout_checkpoint_reentrant=True,
    pg_saved_tensors_cpu_offload=False,
    pg_saved_tensors_pin_memory=True,
    pg_tbptt_window=None,
    tbptt_loss_sink=None,
):
    if tbptt_loss_sink is not None:
        # True TBPTT mode: window loss is backpropagated as soon as the window ends.
        # This keeps memory bounded by one TBPTT window and is incompatible with
        # outer rollout checkpointing.
        with _saved_tensors_cpu_offload_context(
            enabled=pg_saved_tensors_cpu_offload,
            pin_memory=pg_saved_tensors_pin_memory,
        ):
            return env_prior.rollout_policy_gradient_loss(
                policy_step_fn=policy_step_fn,
                batch_size=batch_size,
                n_samples=n_samples,
                num_features=num_features,
                device=device,
                single_eval_pos=single_eval_pos,
                collect_x=collect_x,
                tbptt_window=pg_tbptt_window,
                tbptt_loss_sink=tbptt_loss_sink,
            )

    if not policy_rollout_checkpoint:
        with _saved_tensors_cpu_offload_context(
            enabled=pg_saved_tensors_cpu_offload,
            pin_memory=pg_saved_tensors_pin_memory,
        ):
            return env_prior.rollout_policy_gradient_loss(
                policy_step_fn=policy_step_fn,
                batch_size=batch_size,
                n_samples=n_samples,
                num_features=num_features,
                device=device,
                single_eval_pos=single_eval_pos,
                collect_x=collect_x,
                tbptt_window=pg_tbptt_window,
                tbptt_loss_sink=None,
            )

    rng_state = _capture_rng_state(device)
    dummy = torch.zeros((), device=device, dtype=torch.float32, requires_grad=True)

    def _rollout_loss_only(_dummy):
        del _dummy
        _restore_rng_state(rng_state)
        with _saved_tensors_cpu_offload_context(
            enabled=pg_saved_tensors_cpu_offload,
            pin_memory=pg_saved_tensors_pin_memory,
        ):
            pg_loss_inner, _, pg_stats_inner = env_prior.rollout_policy_gradient_loss(
                policy_step_fn=policy_step_fn,
                batch_size=batch_size,
                n_samples=n_samples,
                num_features=num_features,
                device=device,
                single_eval_pos=single_eval_pos,
                collect_x=collect_x,
                tbptt_window=pg_tbptt_window,
                tbptt_loss_sink=None,
            )
        return (
            pg_loss_inner,
            pg_stats_inner["objective"].detach(),
            pg_stats_inner["reward_mean"].detach(),
            pg_stats_inner["reward_std"].detach(),
            pg_stats_inner.get("reward_min", torch.zeros((), device=device, dtype=torch.float32)).detach(),
            pg_stats_inner.get("reward_max", torch.zeros((), device=device, dtype=torch.float32)).detach(),
            pg_stats_inner.get("reward_abs_max", torch.zeros((), device=device, dtype=torch.float32)).detach(),
            pg_stats_inner.get("reward_clip_hit_share", torch.zeros((), device=device, dtype=torch.float32)).detach(),
            pg_stats_inner.get(
                "reward_norm_clip_hit_share",
                torch.zeros((), device=device, dtype=torch.float32),
            ).detach(),
        )

    (
        pg_loss_chunk,
        objective,
        reward_mean,
        reward_std,
        reward_min,
        reward_max,
        reward_abs_max,
        reward_clip_hit_share,
        reward_norm_clip_hit_share,
    ) = checkpoint(
        _rollout_loss_only,
        dummy,
        use_reentrant=bool(policy_rollout_checkpoint_reentrant),
        preserve_rng_state=False,
    )
    pg_stats_chunk = {
        "objective": objective,
        "reward_mean": reward_mean,
        "reward_std": reward_std,
        "reward_min": reward_min,
        "reward_max": reward_max,
        "reward_abs_max": reward_abs_max,
        "reward_clip_hit_share": reward_clip_hit_share,
        "reward_norm_clip_hit_share": reward_norm_clip_hit_share,
    }
    return pg_loss_chunk, None, pg_stats_chunk


def eval_criterion(criterion, targets, output, device, n_out):
    if isinstance(criterion, nn.GaussianNLLLoss):
        assert output.shape[-1] == 2, \
            'need to write a little bit of code to handle multiple regression targets at once'

        mean_pred = output[..., 0]
        var_pred = output[..., 1].abs()
        losses = criterion(mean_pred.flatten(), targets.flatten(), var=var_pred.flatten())
    elif isinstance(criterion, (nn.MSELoss, nn.BCEWithLogitsLoss)):
        losses = criterion(output.flatten(), targets.flatten())
    elif isinstance(criterion, nn.CrossEntropyLoss):
        losses = criterion(output.reshape(-1, n_out)[:, :int(targets.max()) + 1], targets.long().flatten())
    else:
        losses = criterion(output, targets)
    losses = losses.view(*output.shape[0:2])
    return utils.torch_nanmean(losses.mean(0), return_nanshare=True)


def train_epoch(
    model, 
    aggregate_k_gradients, 
    using_dist, 
    scaler, 
    dl, 
    device, 
    optimizer, 
    criterion, 
    n_out, 
    epoch_idx=None,
    progress_bar=False,
    epoch_profiler=None,
    epoch_start_time=None,
    train_profiler_log_every_batches=0,
    verbose=False,
    kernel_profiler=None,
):
    model.train()  # Turn on the train mode
    total_loss = torch.tensor(0., device = device)
    nan_steps = torch.tensor(0., device = device)
    ignore_steps = torch.tensor(0., device = device)
    valid_steps = torch.tensor(0., device=device)
    skipped_steps = 0
    grad_accum_steps = 0
    requires_midaccum_grad_finite_check = aggregate_k_gradients > 1
    track_ignore_share = isinstance(criterion, nn.CrossEntropyLoss)
    steps_per_epoch = len(dl)
    assert len(dl) % aggregate_k_gradients == 0, 'Please set the number of steps per epoch s.t. `aggregate_k_gradients` divides it.'
    optimizer.zero_grad(set_to_none=True)
    if progress_bar:
        desc = f"Epoch {epoch_idx}" if epoch_idx is not None else "Epoch"
        dl = tqdm(dl, total=steps_per_epoch, desc=desc, unit="step")
    for batch, (data, targets, single_eval_pos) in enumerate(dl):
        batch_epoch_for_profile = int(epoch_idx if epoch_idx is not None else getattr(dl, "epoch_count", -1))
        # change the description of the progress bar
        if progress_bar and hasattr(dl, "set_postfix"):
            dl.set_postfix(
                step=f"{batch + 1}/{steps_per_epoch}",
                train_ctx=int(single_eval_pos),
                test_ctx=int(data[1].shape[0] - single_eval_pos),
            )
        if wandb.run is not None:
            wandb.log({'train_train_sample_number': single_eval_pos, 'train_test_sample_number': data[1].shape[0] - single_eval_pos})

        if using_dist and not ((grad_accum_steps + 1) == aggregate_k_gradients):
            cm = model.no_sync()
        else:
            cm = nullcontext()
        with cm:
            with (
                kernel_profiler.phase("train.forward")
                if (kernel_profiler is not None and kernel_profiler.enabled())
                else nullcontext()
            ):
                with _amp_autocast_context(
                    scaler,
                    device=device,
                    dtype=(torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16),
                ):
                    # for mothernet, la_mothernet, model is MLPModelPredictor from ticl.py
                    output = model(
                        tuple(e.to(device) if torch.is_tensor(e) else e for e in data)
                        if isinstance(data, tuple) else data.to(device),
                        single_eval_pos=single_eval_pos
                    )

                    targets = targets.to(device)
                    if single_eval_pos is not None:
                        targets = targets[single_eval_pos:]
                    loss, nan_share = eval_criterion(
                        criterion,
                        targets,
                        output,
                        device=device,
                        n_out=n_out
                    )
                    loss = loss / aggregate_k_gradients

            if not torch.isfinite(loss).all():
                skipped_steps += 1
                optimizer.zero_grad(set_to_none=True)
                grad_accum_steps = 0
                loss_val = float(loss.detach().float().mean().cpu())
                print(f"[train-skip] epoch={batch_epoch_for_profile} batch={batch} reason=loss_nonfinite loss={loss_val}")
                if epoch_profiler is not None:
                    bs_prof, ns_prof = _infer_batch_shape_for_profile(data, targets)
                    epoch_profiler.record_batch(valid=False, batch_size=bs_prof, n_samples=ns_prof)
                    _maybe_emit_profile_interval(
                        epoch_profiler,
                        epoch_start_time=epoch_start_time,
                        epoch_idx=epoch_idx,
                        batch_idx=batch,
                        train_profiler_log_every_batches=train_profiler_log_every_batches,
                        verbose=verbose,
                    )
                if wandb.run is not None:
                    wandb.log({'train_skip': 1, 'train_skip_reason': 'loss_nonfinite', 'train_skip_loss': loss_val})
                _maybe_step_kernel_profiler(
                    kernel_profiler,
                    epoch_idx=batch_epoch_for_profile,
                    batch_idx=batch,
                    verbose=verbose,
                )
                continue

            if wandb.run:
                wandb.log({'batch_loss': float(loss.detach().mean().cpu()) * aggregate_k_gradients})
            with (
                kernel_profiler.phase("train.backward")
                if (kernel_profiler is not None and kernel_profiler.enabled())
                else nullcontext()
            ):
                loss.backward()

            if requires_midaccum_grad_finite_check and _has_nonfinite_gradients(model, device=device):
                skipped_steps += 1
                optimizer.zero_grad(set_to_none=True)
                grad_accum_steps = 0
                print(f"[train-skip] epoch={batch_epoch_for_profile} batch={batch} reason=grad_nonfinite")
                if epoch_profiler is not None:
                    bs_prof, ns_prof = _infer_batch_shape_for_profile(data, targets)
                    epoch_profiler.record_batch(valid=False, batch_size=bs_prof, n_samples=ns_prof)
                    _maybe_emit_profile_interval(
                        epoch_profiler,
                        epoch_start_time=epoch_start_time,
                        epoch_idx=epoch_idx,
                        batch_idx=batch,
                        train_profiler_log_every_batches=train_profiler_log_every_batches,
                        verbose=verbose,
                    )
                if wandb.run is not None:
                    wandb.log({'train_skip': 1, 'train_skip_reason': 'grad_nonfinite'})
                _maybe_step_kernel_profiler(
                    kernel_profiler,
                    epoch_idx=batch_epoch_for_profile,
                    batch_idx=batch,
                    verbose=verbose,
                )
                continue

            grad_accum_steps += 1

            if grad_accum_steps == aggregate_k_gradients:
                with (
                    kernel_profiler.phase("train.step")
                    if (kernel_profiler is not None and kernel_profiler.enabled())
                    else nullcontext()
                ):
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., foreach=True)
                if not torch.isfinite(grad_norm):
                    skipped_steps += 1
                    optimizer.zero_grad(set_to_none=True)
                    grad_accum_steps = 0
                    print(f"[train-skip] epoch={batch_epoch_for_profile} batch={batch} reason=grad_norm_nonfinite")
                    if epoch_profiler is not None:
                        bs_prof, ns_prof = _infer_batch_shape_for_profile(data, targets)
                        epoch_profiler.record_batch(valid=False, batch_size=bs_prof, n_samples=ns_prof)
                        _maybe_emit_profile_interval(
                            epoch_profiler,
                            epoch_start_time=epoch_start_time,
                            epoch_idx=epoch_idx,
                            batch_idx=batch,
                            train_profiler_log_every_batches=train_profiler_log_every_batches,
                            verbose=verbose,
                        )
                    if wandb.run is not None:
                        wandb.log({'train_skip': 1, 'train_skip_reason': 'grad_norm_nonfinite'})
                    _maybe_step_kernel_profiler(
                        kernel_profiler,
                        epoch_idx=batch_epoch_for_profile,
                        batch_idx=batch,
                        verbose=verbose,
                    )
                    continue
                with (
                    kernel_profiler.phase("train.step")
                    if (kernel_profiler is not None and kernel_profiler.enabled())
                    else nullcontext()
                ):
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                grad_accum_steps = 0

            total_loss += loss.detach().mean()
            valid_steps += 1
            nan_steps += nan_share
            if track_ignore_share:
                ignore_steps += (targets == -100).float().mean()
            if epoch_profiler is not None:
                bs_prof, ns_prof = _infer_batch_shape_for_profile(data, targets)
                epoch_profiler.record_batch(valid=True, batch_size=bs_prof, n_samples=ns_prof)
                _maybe_emit_profile_interval(
                    epoch_profiler,
                    epoch_start_time=epoch_start_time,
                    epoch_idx=epoch_idx,
                    batch_idx=batch,
                    train_profiler_log_every_batches=train_profiler_log_every_batches,
                    verbose=verbose,
                )
            _maybe_step_kernel_profiler(
                kernel_profiler,
                epoch_idx=batch_epoch_for_profile,
                batch_idx=batch,
                verbose=verbose,
            )

    if grad_accum_steps > 0:
        optimizer.zero_grad(set_to_none=True)

    if valid_steps.item() == 0:
        print("[train-warn] all batches were skipped due to non-finite loss/grad.")
        return float('inf'), 1.0, 0.0

    denom = valid_steps
    if skipped_steps > 0:
        print(f"[train-skip-summary] skipped={skipped_steps} valid={int(valid_steps.item())} total={steps_per_epoch}")
    mean_loss = float((total_loss / denom * aggregate_k_gradients).detach().cpu())
    return (mean_loss,
            nan_steps.cpu().item() / denom.cpu().item(),
            ignore_steps.cpu().item() / denom.cpu().item())


def train_epoch_policy_gradient(
    model,
    aggregate_k_gradients,
    using_dist,
    scaler,
    dl,
    device,
    optimizer,
    env_prior,
    policy_rollout_chunk_size=None,
    policy_rollout_chunk_autotune=True,
    policy_rollout_chunk_grow_every=8,
    policy_rollout_chunk_grow_factor=2.0,
    policy_rollout_checkpoint=False,
    policy_rollout_checkpoint_reentrant=True,
    pg_grad_mutable_kv_cache=False,
    pg_saved_tensors_cpu_offload=False,
    pg_saved_tensors_pin_memory=True,
    pg_oom_debug_raise=False,
    pg_kv_cache_mode="auto",
    pg_kv_cache_page_size=None,
    pg_tbptt_window=None,
    pg_oom_reduce_tbptt_first=False,
    pg_torch_compile=False,
    pg_torch_compile_backend="inductor",
    pg_torch_compile_mode="reduce-overhead",
    pg_torch_compile_fullgraph=False,
    pg_torch_compile_dynamic=False,
    epoch_idx=None,
    progress_bar=False,
    epoch_profiler=None,
    epoch_start_time=None,
    train_profiler_log_every_batches=0,
    verbose=False,
    gpu_observer=None,
    kernel_profiler=None,
):
    model.train()
    policy_autocast_dtype = _resolve_policy_autocast_dtype(device)
    device_obj = device if isinstance(device, torch.device) else torch.device(str(device))
    total_loss = torch.tensor(0.0, device=device)
    valid_steps = torch.tensor(0.0, device=device)
    skipped_steps = 0
    grad_accum_steps = 0
    requires_midaccum_grad_finite_check = aggregate_k_gradients > 1

    steps_per_epoch = len(dl)
    assert steps_per_epoch % aggregate_k_gradients == 0, 'Please set the number of steps per epoch s.t. `aggregate_k_gradients` divides it.'
    optimizer.zero_grad(set_to_none=True)
    batch_size_hint = int(dl.batch_size)

    tbptt_streaming_default = False
    if pg_tbptt_window is not None:
        try:
            tbptt_streaming_default = int(pg_tbptt_window) > 0
        except Exception:
            tbptt_streaming_default = False

    allow_grad_mutable_kv = bool(pg_grad_mutable_kv_cache) and bool(policy_rollout_checkpoint)
    requested_kv_mode = "auto" if pg_kv_cache_mode is None else str(pg_kv_cache_mode).strip().lower()
    if requested_kv_mode not in {"auto", "immutable", "static", "paged"}:
        requested_kv_mode = "auto"
    if not allow_grad_mutable_kv:
        kv_cache_mode = "immutable"
    else:
        kv_cache_mode = requested_kv_mode if requested_kv_mode != "auto" else "paged"
        if kv_cache_mode == "immutable":
            allow_grad_mutable_kv = False
    if kv_cache_mode == "paged":
        kv_cache_page_size = 128 if pg_kv_cache_page_size is None else int(max(1, pg_kv_cache_page_size))
    else:
        kv_cache_page_size = None
    inplace_paged_kv_env = str(os.environ.get("TICL_POLICY_INPLACE_PAGED_KV", "auto")).strip().lower()
    if inplace_paged_kv_env in {"1", "true", "yes", "on"}:
        allow_grad_inplace_paged_kv = True
    elif inplace_paged_kv_env in {"0", "false", "no", "off"}:
        allow_grad_inplace_paged_kv = False
    else:
        # Auto mode stays conservative; in-place paged append is opt-in via env
        # because it can raise peak memory under long-horizon PG rollout.
        allow_grad_inplace_paged_kv = False
    if (not allow_grad_mutable_kv) or (kv_cache_mode != "paged"):
        allow_grad_inplace_paged_kv = False

    # Reentrant checkpoint is unsafe with in-place-updated static KV cache.
    # Paged cache uses copy-on-write appends and is safe for reentrant.
    mutable_reentrant_safe = (
        (not allow_grad_mutable_kv)
        or (kv_cache_mode == "paged")
    )
    checkpoint_reentrant_active = bool(policy_rollout_checkpoint_reentrant) and bool(mutable_reentrant_safe)
    # In TBPTT streaming mode the outer rollout checkpoint path is not used,
    # so disabling inner recompute_attn would only increase peak memory.
    disable_inner_recompute = bool(policy_rollout_checkpoint) and (not tbptt_streaming_default)
    recompute_snapshot = []
    if disable_inner_recompute:
        # Avoid nested checkpointing with policy rollout checkpoint, and avoid
        # checkpoint/in-place version conflicts for mutable static KV cache.
        recompute_snapshot = _snapshot_policy_inner_recompute_attn(model)
        _set_policy_inner_recompute_attn(model, enabled=False)

    policy_step_fn = _build_policy_step_fn(
        model,
        num_features=dl.num_features,
        max_cache_len=int(dl.n_samples),
        kv_cache_mode=kv_cache_mode,
        kv_cache_page_size=kv_cache_page_size,
        allow_grad_mutable_cache=allow_grad_mutable_kv,
        allow_grad_inplace_paged_cache=allow_grad_inplace_paged_kv,
        pg_torch_compile=bool(pg_torch_compile),
        pg_torch_compile_backend=str(pg_torch_compile_backend),
        pg_torch_compile_mode=str(pg_torch_compile_mode),
        pg_torch_compile_fullgraph=bool(pg_torch_compile_fullgraph),
        pg_torch_compile_dynamic=bool(pg_torch_compile_dynamic),
    )
    model_ref = model.module if hasattr(model, "module") else model
    policy_step_profile_consumer = getattr(model_ref, "consume_policy_step_profile", None)
    batch_size = int(dl.batch_size)
    n_samples = int(dl.n_samples)
    num_features = int(dl.num_features)
    if pg_tbptt_window is None:
        current_tbptt_window = None
    else:
        tbptt_value = int(pg_tbptt_window)
        current_tbptt_window = tbptt_value if tbptt_value > 0 else None
    if policy_rollout_chunk_size is None:
        # Default: run rollout chunks at full batch width.
        rollout_chunk_cap = batch_size
    else:
        chunk_value = int(policy_rollout_chunk_size)
        if chunk_value <= 0:
            rollout_chunk_cap = batch_size
        else:
            rollout_chunk_cap = int(max(1, min(batch_size, chunk_value)))
    current_rollout_chunk_size = int(rollout_chunk_cap)
    rollout_chunk_autotune_active = bool(policy_rollout_chunk_autotune) and (rollout_chunk_cap > 1)
    rollout_chunk_grow_every = int(max(1, policy_rollout_chunk_grow_every))
    rollout_chunk_grow_factor = float(policy_rollout_chunk_grow_factor)
    if (not math.isfinite(rollout_chunk_grow_factor)) or rollout_chunk_grow_factor <= 1.0:
        rollout_chunk_grow_factor = 2.0
    stable_batches_for_chunk_growth = 0
    try:
        pg_phase_log_every = int(os.environ.get("TICL_PG_PHASE_LOG_EVERY_BATCHES", "50"))
    except Exception:
        pg_phase_log_every = 50
    pg_phase_log_every = max(1, pg_phase_log_every)
    try:
        pg_same_cfg_oom_retries = int(os.environ.get("TICL_PG_OOM_SAME_CFG_RETRIES", "0"))
    except Exception:
        pg_same_cfg_oom_retries = 0
    pg_same_cfg_oom_retries = max(0, pg_same_cfg_oom_retries)
    mem_guard_flag = str(os.environ.get("TICL_PG_MEM_GUARD", "1")).strip().lower()
    pg_mem_guard_enabled = mem_guard_flag not in {"0", "false", "no", "off"}
    try:
        pg_mem_guard_reserved_frac = float(os.environ.get("TICL_PG_MEM_GUARD_RESERVED_FRAC", "0.90"))
    except Exception:
        pg_mem_guard_reserved_frac = 0.90
    if (not math.isfinite(pg_mem_guard_reserved_frac)) or pg_mem_guard_reserved_frac <= 0.0:
        pg_mem_guard_reserved_frac = 0.90
    pg_mem_guard_reserved_frac = float(min(0.995, max(0.50, pg_mem_guard_reserved_frac)))
    try:
        pg_mem_guard_slack_gb = float(os.environ.get("TICL_PG_MEM_GUARD_SLACK_GB", "3.0"))
    except Exception:
        pg_mem_guard_slack_gb = 3.0
    if (not math.isfinite(pg_mem_guard_slack_gb)) or pg_mem_guard_slack_gb < 0.0:
        pg_mem_guard_slack_gb = 3.0
    pg_mem_guard_slack_bytes = int(pg_mem_guard_slack_gb * (1024.0 ** 3))

    iterator = range(steps_per_epoch)
    if progress_bar:
        desc = f"Epoch {epoch_idx}" if epoch_idx is not None else "Epoch"
        iterator = tqdm(iterator, total=steps_per_epoch, desc=desc, unit="step")

    for batch in iterator:
        if pg_mem_guard_enabled and device_obj.type == "cuda" and torch.cuda.is_available():
            try:
                mem_alloc = int(torch.cuda.memory_allocated(device_obj))
                mem_reserved = int(torch.cuda.memory_reserved(device_obj))
                mem_total = int(torch.cuda.get_device_properties(device_obj).total_memory)
            except Exception:
                mem_alloc = 0
                mem_reserved = 0
                mem_total = 0
            slack_bytes = int(max(0, mem_reserved - mem_alloc))
            need_cleanup = False
            if mem_total > 0 and mem_reserved >= int(float(mem_total) * pg_mem_guard_reserved_frac):
                need_cleanup = True
            if slack_bytes >= pg_mem_guard_slack_bytes:
                need_cleanup = True
            if need_cleanup:
                gc.collect()
                torch.cuda.empty_cache()
        batch_epoch_for_profile = int(epoch_idx if epoch_idx is not None else getattr(dl, "epoch_count", -1))
        should_log_phase = bool(
            verbose and (
                (batch == 0)
                or ((batch + 1) == steps_per_epoch)
                or (((batch + 1) % pg_phase_log_every) == 0)
            )
        )
        if using_dist and not ((grad_accum_steps + 1) == aggregate_k_gradients):
            cm = model.no_sync()
        else:
            cm = nullcontext()

        with cm:
            single_eval_pos = env_prior._sample_single_eval_pos(n_samples, None)
            batch_rng_state = _capture_rng_state(device)
            max_batch_oom_retries = 8
            batch_attempt = 0
            same_cfg_oom_retries = 0
            max_same_cfg_oom_retries = int(pg_same_cfg_oom_retries)
            enable_same_cfg_oom_retry = (
                max_same_cfg_oom_retries > 0
                and device_obj.type == "cuda"
                and torch.cuda.is_available()
            )
            gpu_observer_active = (
                gpu_observer is not None
                and hasattr(gpu_observer, "enabled")
                and bool(gpu_observer.enabled())
            )
            while True:
                _restore_rng_state(batch_rng_state)
                batch_attempt += 1

                batch_loss = torch.tensor(0.0, device=device)
                batch_objective = torch.tensor(0.0, device=device)
                batch_reward_mean = torch.tensor(0.0, device=device)
                batch_reward_std = torch.tensor(0.0, device=device)
                batch_reward_min_value = float("inf")
                batch_reward_max_value = float("-inf")
                batch_reward_absmax_value = 0.0
                batch_reward_clip_hit_share = 0.0
                batch_reward_norm_clip_hit_share = 0.0
                batch_nonfinite = False
                batch_rollout_wall = 0.0
                batch_backward_wall = 0.0
                batch_backward_calls = 0
                batch_step_wall = 0.0
                batch_rollout_policy_cuda_ms = 0.0
                batch_rollout_transition_cuda_ms = 0.0
                batch_rollout_policy_wall_ms = 0.0
                batch_rollout_transition_wall_ms = 0.0
                batch_rollout_transition_y_wall_ms = 0.0
                batch_rollout_transition_x_wall_ms = 0.0
                batch_rollout_transition_group_wall_ms = 0.0
                batch_rollout_transition_env_pack_wall_ms = 0.0
                batch_rollout_transition_state_update_wall_ms = 0.0
                batch_rollout_transition_group_count = 0
                batch_policy_step_calls = 0
                batch_policy_step_encode_ms = 0.0
                batch_policy_step_transformer_ms = 0.0
                batch_policy_step_decoder_ms = 0.0
                batch_policy_step_total_ms = 0.0
                batch_policy_step_transformer_layer_calls = 0
                batch_policy_step_transformer_layer_proj_ms = 0.0
                batch_policy_step_transformer_layer_cache_ms = 0.0
                batch_policy_step_transformer_layer_attnff_ms = 0.0
                batch_policy_step_transformer_layer_attn_core_ms = 0.0
                batch_policy_step_transformer_layer_finalize_ms = 0.0
                batch_policy_step_transformer_layer_finalize_attn_outproj_ms = 0.0
                batch_policy_step_transformer_layer_finalize_ffn_ms = 0.0
                batch_policy_step_transformer_layer_total_ms = 0.0
                batch_policy_step_layer_paged_single_page_calls = 0
                batch_policy_step_layer_paged_flash_prefix_calls = 0
                batch_policy_step_layer_paged_flash_merge_calls = 0
                batch_policy_step_layer_paged_dense_calls = 0
                batch_pg_loss_signature = None
                try:
                    sig_fn = getattr(env_prior, "policy_gradient_loss_signature", None)
                    cfg = getattr(env_prior, "config", {})
                    if callable(sig_fn):
                        _resolve_scalar = getattr(env_prior, "_resolve_scalar", None)
                        discount_cfg = cfg.get("discount", 1.0) if isinstance(cfg, dict) else 1.0
                        eps_cfg = cfg.get("reward_norm_eps", 1e-6) if isinstance(cfg, dict) else 1e-6
                        clip_cfg = cfg.get("reward_norm_clip", 10.0) if isinstance(cfg, dict) else 10.0
                        if callable(_resolve_scalar):
                            discount_v = _resolve_scalar(discount_cfg)
                            eps_v = _resolve_scalar(eps_cfg)
                            clip_v = _resolve_scalar(clip_cfg)
                        else:
                            discount_v = discount_cfg
                            eps_v = eps_cfg
                            clip_v = clip_cfg
                        batch_pg_loss_signature = sig_fn(
                            normalize=bool(cfg.get("policy_gradient_normalize_rewards", False)) if isinstance(cfg, dict) else False,
                            discount=float(max(0.0, min(1.0, discount_v))),
                            detach_stats=True,
                            eps=eps_v,
                            clip=clip_v,
                        )
                    else:
                        batch_pg_loss_signature = "na"
                except Exception:
                    batch_pg_loss_signature = None
                batch_grad_norm_value = None
                batch_optimizer_step_value = None
                batch_optimizer_stepped = False
                rollout_t_start_unix = None
                rollout_t_end_unix = None
                backward_t_start_unix = None
                backward_t_end_unix = None
                step_t_start_unix = None
                step_t_end_unix = None
                rollout_cuda_pairs = []
                backward_cuda_pairs = []
                step_cuda_pairs = []

                batch_oom = False
                for chunk_start in range(0, batch_size, current_rollout_chunk_size):
                    chunk_bs = min(current_rollout_chunk_size, batch_size - chunk_start)
                    chunk_weight = float(chunk_bs) / float(batch_size)
                    tbptt_stream_backward_active = (
                        current_tbptt_window is not None and int(current_tbptt_window) > 0
                    )
                    tbptt_window_backward_called = False

                    if tbptt_stream_backward_active:
                        backward_scale = float(chunk_weight) / float(aggregate_k_gradients)

                        def _tbptt_chunk_loss_sink(weighted_window_loss):
                            nonlocal tbptt_window_backward_called, batch_backward_wall, batch_backward_calls
                            nonlocal backward_t_start_unix, backward_t_end_unix
                            window_loss_scaled = weighted_window_loss * backward_scale
                            backward_t0_unix = time.time()
                            backward_t0 = time.perf_counter()
                            backward_cuda_start = None
                            if gpu_observer_active and ("cuda" in str(device)) and torch.cuda.is_available():
                                backward_cuda_start = torch.cuda.Event(enable_timing=True)
                                backward_cuda_start.record()
                            with (
                                kernel_profiler.phase("pg.backward")
                                if (kernel_profiler is not None and kernel_profiler.enabled())
                                else nullcontext()
                            ):
                                if scaler is None:
                                    window_loss_scaled.backward()
                                else:
                                    scaler.scale(window_loss_scaled).backward()
                            if backward_cuda_start is not None:
                                backward_cuda_end = torch.cuda.Event(enable_timing=True)
                                backward_cuda_end.record()
                                backward_cuda_pairs.append((backward_cuda_start, backward_cuda_end))
                            batch_backward_wall += (time.perf_counter() - backward_t0)
                            batch_backward_calls += 1
                            backward_t1_unix = time.time()
                            if backward_t_start_unix is None:
                                backward_t_start_unix = backward_t0_unix
                            backward_t_end_unix = backward_t1_unix
                            tbptt_window_backward_called = True
                    else:
                        _tbptt_chunk_loss_sink = None

                    try:
                        rollout_t0_unix = time.time()
                        rollout_t0 = time.perf_counter()
                        rollout_cuda_start = None
                        if gpu_observer_active and ("cuda" in str(device)) and torch.cuda.is_available():
                            rollout_cuda_start = torch.cuda.Event(enable_timing=True)
                            rollout_cuda_start.record()
                        with (
                            kernel_profiler.phase("pg.rollout")
                            if (kernel_profiler is not None and kernel_profiler.enabled())
                            else nullcontext()
                        ):
                            with _amp_autocast_context(
                                scaler,
                                device=device,
                                dtype=policy_autocast_dtype,
                            ):
                                pg_loss_chunk, rollout_chunk, pg_stats_chunk = _compute_policy_rollout_chunk_loss(
                                    env_prior=env_prior,
                                    policy_step_fn=policy_step_fn,
                                    batch_size=chunk_bs,
                                    n_samples=n_samples,
                                    num_features=num_features,
                                    device=device,
                                    single_eval_pos=single_eval_pos,
                                    collect_x=False,
                                    policy_rollout_checkpoint=bool(policy_rollout_checkpoint) and (not tbptt_stream_backward_active),
                                    policy_rollout_checkpoint_reentrant=checkpoint_reentrant_active,
                                    pg_saved_tensors_cpu_offload=pg_saved_tensors_cpu_offload,
                                    pg_saved_tensors_pin_memory=pg_saved_tensors_pin_memory,
                                    pg_tbptt_window=current_tbptt_window,
                                    tbptt_loss_sink=_tbptt_chunk_loss_sink,
                                )
                                weighted_pg_loss = pg_loss_chunk * chunk_weight
                                loss = weighted_pg_loss / aggregate_k_gradients
                                if tbptt_stream_backward_active and tbptt_window_backward_called:
                                    loss = loss.detach()
                                metric_loss = loss.detach().mean()
                                if tbptt_stream_backward_active and tbptt_window_backward_called:
                                    # In streaming-TBPTT mode gradients are consumed inside
                                    # tbptt_loss_sink; expose a meaningful monitor loss.
                                    metric_loss = (
                                        -pg_stats_chunk["objective"].detach() * float(chunk_weight)
                                    ) / float(aggregate_k_gradients)
                        if rollout_cuda_start is not None:
                            rollout_cuda_end = torch.cuda.Event(enable_timing=True)
                            rollout_cuda_end.record()
                            rollout_cuda_pairs.append((rollout_cuda_start, rollout_cuda_end))
                        batch_rollout_wall += (time.perf_counter() - rollout_t0)
                        rollout_t1_unix = time.time()
                        if rollout_t_start_unix is None:
                            rollout_t_start_unix = rollout_t0_unix
                        rollout_t_end_unix = rollout_t1_unix
                        if not torch.isfinite(loss).all():
                            batch_nonfinite = True
                            loss_val = float(loss.detach().float().mean().cpu())
                            break

                        if (not tbptt_stream_backward_active) and loss.requires_grad:
                            backward_t0_unix = time.time()
                            backward_t0 = time.perf_counter()
                            backward_cuda_start = None
                            if gpu_observer_active and ("cuda" in str(device)) and torch.cuda.is_available():
                                backward_cuda_start = torch.cuda.Event(enable_timing=True)
                                backward_cuda_start.record()
                            with (
                                kernel_profiler.phase("pg.backward")
                                if (kernel_profiler is not None and kernel_profiler.enabled())
                                else nullcontext()
                            ):
                                if scaler is None:
                                    loss.backward()
                                else:
                                    scaler.scale(loss).backward()
                            if backward_cuda_start is not None:
                                backward_cuda_end = torch.cuda.Event(enable_timing=True)
                                backward_cuda_end.record()
                                backward_cuda_pairs.append((backward_cuda_start, backward_cuda_end))
                            batch_backward_wall += (time.perf_counter() - backward_t0)
                            batch_backward_calls += 1
                            backward_t1_unix = time.time()
                            if backward_t_start_unix is None:
                                backward_t_start_unix = backward_t0_unix
                            backward_t_end_unix = backward_t1_unix
                        if tbptt_stream_backward_active and (not tbptt_window_backward_called):
                            if loss.requires_grad:
                                # Compatibility fallback for mocked/test rollout helpers that
                                # do not invoke tbptt_loss_sink.
                                backward_t0_unix = time.time()
                                backward_t0 = time.perf_counter()
                                backward_cuda_start = None
                                if gpu_observer_active and ("cuda" in str(device)) and torch.cuda.is_available():
                                    backward_cuda_start = torch.cuda.Event(enable_timing=True)
                                    backward_cuda_start.record()
                                with (
                                    kernel_profiler.phase("pg.backward")
                                    if (kernel_profiler is not None and kernel_profiler.enabled())
                                    else nullcontext()
                                ):
                                    if scaler is None:
                                        loss.backward()
                                    else:
                                        scaler.scale(loss).backward()
                                if backward_cuda_start is not None:
                                    backward_cuda_end = torch.cuda.Event(enable_timing=True)
                                    backward_cuda_end.record()
                                    backward_cuda_pairs.append((backward_cuda_start, backward_cuda_end))
                                batch_backward_wall += (time.perf_counter() - backward_t0)
                                batch_backward_calls += 1
                                backward_t1_unix = time.time()
                                if backward_t_start_unix is None:
                                    backward_t_start_unix = backward_t0_unix
                                backward_t_end_unix = backward_t1_unix
                            else:
                                raise RuntimeError("TBPTT streaming backward did not receive any window losses.")
                    except Exception as e:
                        if not _is_oom_exception(e):
                            raise
                        if bool(pg_oom_debug_raise):
                            print(
                                "[pg-oom-debug] re-raising OOM for full traceback "
                                "(set --pg-oom-debug-raise false to enable auto chunk fallback)."
                            )
                            raise
                        # Some batches hit transient allocator pressure; retry once
                        # with unchanged TBPTT/chunk before degrading configuration.
                        if enable_same_cfg_oom_retry and same_cfg_oom_retries < max_same_cfg_oom_retries:
                            same_cfg_oom_retries += 1
                            batch_oom = True
                            if "cuda" in str(device):
                                torch.cuda.empty_cache()
                            gc.collect()
                            print(
                                f"[pg-oom] epoch={batch_epoch_for_profile} batch={batch} "
                                f"retrying same config once "
                                f"(chunk={current_rollout_chunk_size}, tbptt={current_tbptt_window})"
                            )
                            break
                        if bool(pg_oom_reduce_tbptt_first):
                            if current_tbptt_window is None and n_samples > 1:
                                current_tbptt_window = max(1, n_samples // 2)
                                batch_oom = True
                                if "cuda" in str(device):
                                    torch.cuda.empty_cache()
                                print(
                                    f"[pg-oom] epoch={batch_epoch_for_profile} batch={batch} "
                                    f"reducing TBPTT window to {current_tbptt_window} "
                                    "(keeping policy rollout chunk size unchanged)"
                                )
                                break
                            if current_tbptt_window is not None and current_tbptt_window > 1:
                                new_tbptt = max(1, int(current_tbptt_window // 2))
                                if new_tbptt < current_tbptt_window:
                                    current_tbptt_window = new_tbptt
                                    batch_oom = True
                                    if "cuda" in str(device):
                                        torch.cuda.empty_cache()
                                    print(
                                        f"[pg-oom] epoch={batch_epoch_for_profile} batch={batch} "
                                        f"reducing TBPTT window to {current_tbptt_window} "
                                        "(keeping policy rollout chunk size unchanged)"
                                    )
                                    break
                        batch_oom = True
                        if "cuda" in str(device):
                            torch.cuda.empty_cache()
                        new_chunk = max(1, int(current_rollout_chunk_size // 2))
                        if new_chunk < current_rollout_chunk_size:
                            current_rollout_chunk_size = new_chunk
                            print(
                                f"[pg-oom] epoch={batch_epoch_for_profile} batch={batch} "
                                f"reducing policy rollout chunk size to {current_rollout_chunk_size}"
                            )
                        else:
                            print(
                                f"[pg-oom] epoch={batch_epoch_for_profile} batch={batch} "
                                "OOM at minimum policy rollout chunk size=1. "
                                "Memory is sequence-length dominated; batch chunking cannot reduce further."
                            )
                        break
                    batch_loss += metric_loss
                    batch_objective += pg_stats_chunk["objective"].detach() * chunk_weight
                    batch_reward_mean += pg_stats_chunk["reward_mean"].detach() * chunk_weight
                    batch_reward_std += pg_stats_chunk["reward_std"].detach() * chunk_weight
                    chunk_reward_min = pg_stats_chunk.get("reward_min", None)
                    if chunk_reward_min is not None:
                        try:
                            batch_reward_min_value = min(batch_reward_min_value, float(chunk_reward_min.detach().cpu()))
                        except Exception:
                            pass
                    chunk_reward_max = pg_stats_chunk.get("reward_max", None)
                    if chunk_reward_max is not None:
                        try:
                            batch_reward_max_value = max(batch_reward_max_value, float(chunk_reward_max.detach().cpu()))
                        except Exception:
                            pass
                    chunk_reward_absmax = pg_stats_chunk.get("reward_abs_max", None)
                    if chunk_reward_absmax is not None:
                        try:
                            batch_reward_absmax_value = max(
                                batch_reward_absmax_value,
                                float(chunk_reward_absmax.detach().cpu()),
                            )
                        except Exception:
                            pass
                    chunk_reward_clip_hit = pg_stats_chunk.get("reward_clip_hit_share", None)
                    if chunk_reward_clip_hit is not None:
                        try:
                            batch_reward_clip_hit_share += float(chunk_reward_clip_hit.detach().cpu()) * chunk_weight
                        except Exception:
                            pass
                    chunk_reward_norm_clip_hit = pg_stats_chunk.get("reward_norm_clip_hit_share", None)
                    if chunk_reward_norm_clip_hit is not None:
                        try:
                            batch_reward_norm_clip_hit_share += (
                                float(chunk_reward_norm_clip_hit.detach().cpu()) * chunk_weight
                            )
                        except Exception:
                            pass
                    policy_cuda_ms = pg_stats_chunk.get("rollout_policy_cuda_ms", None)
                    if policy_cuda_ms is not None:
                        try:
                            batch_rollout_policy_cuda_ms += float(policy_cuda_ms)
                        except Exception:
                            pass
                    transition_cuda_ms = pg_stats_chunk.get("rollout_transition_cuda_ms", None)
                    if transition_cuda_ms is not None:
                        try:
                            batch_rollout_transition_cuda_ms += float(transition_cuda_ms)
                        except Exception:
                            pass
                    policy_wall_ms = pg_stats_chunk.get("rollout_policy_wall_ms", None)
                    if policy_wall_ms is not None:
                        try:
                            batch_rollout_policy_wall_ms += float(policy_wall_ms)
                        except Exception:
                            pass
                    transition_wall_ms = pg_stats_chunk.get("rollout_transition_wall_ms", None)
                    if transition_wall_ms is not None:
                        try:
                            batch_rollout_transition_wall_ms += float(transition_wall_ms)
                        except Exception:
                            pass
                    transition_y_wall_ms = pg_stats_chunk.get("rollout_transition_y_wall_ms", None)
                    if transition_y_wall_ms is not None:
                        try:
                            batch_rollout_transition_y_wall_ms += float(transition_y_wall_ms)
                        except Exception:
                            pass
                    transition_x_wall_ms = pg_stats_chunk.get("rollout_transition_x_wall_ms", None)
                    if transition_x_wall_ms is not None:
                        try:
                            batch_rollout_transition_x_wall_ms += float(transition_x_wall_ms)
                        except Exception:
                            pass
                    transition_group_wall_ms = pg_stats_chunk.get("rollout_transition_group_wall_ms", None)
                    if transition_group_wall_ms is not None:
                        try:
                            batch_rollout_transition_group_wall_ms += float(transition_group_wall_ms)
                        except Exception:
                            pass
                    transition_env_pack_wall_ms = pg_stats_chunk.get("rollout_transition_env_pack_wall_ms", None)
                    if transition_env_pack_wall_ms is not None:
                        try:
                            batch_rollout_transition_env_pack_wall_ms += float(transition_env_pack_wall_ms)
                        except Exception:
                            pass
                    transition_state_update_wall_ms = pg_stats_chunk.get("rollout_transition_state_update_wall_ms", None)
                    if transition_state_update_wall_ms is not None:
                        try:
                            batch_rollout_transition_state_update_wall_ms += float(transition_state_update_wall_ms)
                        except Exception:
                            pass
                    transition_group_count = pg_stats_chunk.get("rollout_transition_group_count", None)
                    if transition_group_count is not None:
                        try:
                            batch_rollout_transition_group_count += int(transition_group_count)
                        except Exception:
                            pass
                    if callable(policy_step_profile_consumer):
                        try:
                            step_profile = policy_step_profile_consumer()
                        except Exception:
                            step_profile = None
                        if isinstance(step_profile, dict):
                            batch_policy_step_calls += int(step_profile.get("calls", 0) or 0)
                            batch_policy_step_encode_ms += float(step_profile.get("encode_wall_s", 0.0) or 0.0) * 1000.0
                            batch_policy_step_transformer_ms += float(
                                step_profile.get("transformer_wall_s", 0.0) or 0.0
                            ) * 1000.0
                            batch_policy_step_decoder_ms += float(step_profile.get("decoder_wall_s", 0.0) or 0.0) * 1000.0
                            batch_policy_step_total_ms += float(step_profile.get("total_wall_s", 0.0) or 0.0) * 1000.0
                            batch_policy_step_transformer_layer_calls += int(
                                step_profile.get("transformer_layer_calls", 0) or 0
                            )
                            batch_policy_step_transformer_layer_proj_ms += float(
                                step_profile.get("transformer_layer_proj_wall_s", 0.0) or 0.0
                            ) * 1000.0
                            batch_policy_step_transformer_layer_cache_ms += float(
                                step_profile.get("transformer_layer_cache_wall_s", 0.0) or 0.0
                            ) * 1000.0
                            batch_policy_step_transformer_layer_attnff_ms += float(
                                step_profile.get("transformer_layer_attnff_wall_s", 0.0) or 0.0
                            ) * 1000.0
                            batch_policy_step_transformer_layer_attn_core_ms += float(
                                step_profile.get("transformer_layer_attn_core_wall_s", 0.0) or 0.0
                            ) * 1000.0
                            batch_policy_step_transformer_layer_finalize_ms += float(
                                step_profile.get("transformer_layer_finalize_wall_s", 0.0) or 0.0
                            ) * 1000.0
                            batch_policy_step_transformer_layer_finalize_attn_outproj_ms += float(
                                step_profile.get("transformer_layer_finalize_attn_outproj_wall_s", 0.0) or 0.0
                            ) * 1000.0
                            batch_policy_step_transformer_layer_finalize_ffn_ms += float(
                                step_profile.get("transformer_layer_finalize_ffn_wall_s", 0.0) or 0.0
                            ) * 1000.0
                            batch_policy_step_transformer_layer_total_ms += float(
                                step_profile.get("transformer_layer_total_wall_s", 0.0) or 0.0
                            ) * 1000.0
                            batch_policy_step_layer_paged_single_page_calls += int(
                                step_profile.get("transformer_layer_paged_path_single_page", 0) or 0
                            )
                            batch_policy_step_layer_paged_flash_prefix_calls += int(
                                step_profile.get("transformer_layer_paged_path_flash_prefix", 0) or 0
                            )
                            batch_policy_step_layer_paged_flash_merge_calls += int(
                                step_profile.get("transformer_layer_paged_path_flash_merge", 0) or 0
                            )
                            batch_policy_step_layer_paged_dense_calls += int(
                                step_profile.get("transformer_layer_paged_path_dense", 0) or 0
                            )

                if batch_oom and batch_attempt < max_batch_oom_retries:
                    optimizer.zero_grad(set_to_none=True)
                    grad_accum_steps = 0
                    if verbose:
                        print(
                            f"[pg-oom-retry] epoch={batch_epoch_for_profile} batch={batch} "
                            f"attempt={batch_attempt + 1}/{max_batch_oom_retries} "
                            f"chunk={current_rollout_chunk_size} tbptt={current_tbptt_window}"
                        )
                    continue
                break

            if callable(policy_step_profile_consumer):
                try:
                    step_profile_tail = policy_step_profile_consumer()
                except Exception:
                    step_profile_tail = None
                if isinstance(step_profile_tail, dict):
                    batch_policy_step_calls += int(step_profile_tail.get("calls", 0) or 0)
                    batch_policy_step_encode_ms += float(step_profile_tail.get("encode_wall_s", 0.0) or 0.0) * 1000.0
                    batch_policy_step_transformer_ms += float(
                        step_profile_tail.get("transformer_wall_s", 0.0) or 0.0
                    ) * 1000.0
                    batch_policy_step_decoder_ms += float(step_profile_tail.get("decoder_wall_s", 0.0) or 0.0) * 1000.0
                    batch_policy_step_total_ms += float(step_profile_tail.get("total_wall_s", 0.0) or 0.0) * 1000.0
                    batch_policy_step_transformer_layer_calls += int(
                        step_profile_tail.get("transformer_layer_calls", 0) or 0
                    )
                    batch_policy_step_transformer_layer_proj_ms += float(
                        step_profile_tail.get("transformer_layer_proj_wall_s", 0.0) or 0.0
                    ) * 1000.0
                    batch_policy_step_transformer_layer_cache_ms += float(
                        step_profile_tail.get("transformer_layer_cache_wall_s", 0.0) or 0.0
                    ) * 1000.0
                    batch_policy_step_transformer_layer_attnff_ms += float(
                        step_profile_tail.get("transformer_layer_attnff_wall_s", 0.0) or 0.0
                    ) * 1000.0
                    batch_policy_step_transformer_layer_attn_core_ms += float(
                        step_profile_tail.get("transformer_layer_attn_core_wall_s", 0.0) or 0.0
                    ) * 1000.0
                    batch_policy_step_transformer_layer_finalize_ms += float(
                        step_profile_tail.get("transformer_layer_finalize_wall_s", 0.0) or 0.0
                    ) * 1000.0
                    batch_policy_step_transformer_layer_finalize_attn_outproj_ms += float(
                        step_profile_tail.get("transformer_layer_finalize_attn_outproj_wall_s", 0.0) or 0.0
                    ) * 1000.0
                    batch_policy_step_transformer_layer_finalize_ffn_ms += float(
                        step_profile_tail.get("transformer_layer_finalize_ffn_wall_s", 0.0) or 0.0
                    ) * 1000.0
                    batch_policy_step_transformer_layer_total_ms += float(
                        step_profile_tail.get("transformer_layer_total_wall_s", 0.0) or 0.0
                    ) * 1000.0
                    batch_policy_step_layer_paged_single_page_calls += int(
                        step_profile_tail.get("transformer_layer_paged_path_single_page", 0) or 0
                    )
                    batch_policy_step_layer_paged_flash_prefix_calls += int(
                        step_profile_tail.get("transformer_layer_paged_path_flash_prefix", 0) or 0
                    )
                    batch_policy_step_layer_paged_flash_merge_calls += int(
                        step_profile_tail.get("transformer_layer_paged_path_flash_merge", 0) or 0
                    )
                    batch_policy_step_layer_paged_dense_calls += int(
                        step_profile_tail.get("transformer_layer_paged_path_dense", 0) or 0
                    )

            stage_cuda_ms = {"rollout": None, "backward": None}
            if gpu_observer_active and ("cuda" in str(device)) and torch.cuda.is_available():
                all_pairs = rollout_cuda_pairs + backward_cuda_pairs
                if all_pairs:
                    torch.cuda.synchronize(device)
                if rollout_cuda_pairs:
                    stage_cuda_ms["rollout"] = float(sum(s.elapsed_time(e) for s, e in rollout_cuda_pairs))
                if backward_cuda_pairs:
                    stage_cuda_ms["backward"] = float(sum(s.elapsed_time(e) for s, e in backward_cuda_pairs))
            rollout_cuda_busy_ratio = None
            backward_cuda_busy_ratio = None
            if stage_cuda_ms["rollout"] is not None and batch_rollout_wall > 0.0:
                rollout_cuda_busy_ratio = float((stage_cuda_ms["rollout"] / 1000.0) / max(1e-9, batch_rollout_wall))
            if stage_cuda_ms["backward"] is not None and batch_backward_wall > 0.0:
                backward_cuda_busy_ratio = float((stage_cuda_ms["backward"] / 1000.0) / max(1e-9, batch_backward_wall))

            rollout_breakdown_suffix = ""
            rollout_breakdown_total_ms = batch_rollout_policy_cuda_ms + batch_rollout_transition_cuda_ms
            if rollout_breakdown_total_ms > 0.0:
                rollout_policy_share = float(batch_rollout_policy_cuda_ms / max(1e-9, rollout_breakdown_total_ms))
                rollout_breakdown_suffix = (
                    f" rollout_policy_cuda_ms={batch_rollout_policy_cuda_ms:.2f}"
                    f" rollout_transition_cuda_ms={batch_rollout_transition_cuda_ms:.2f}"
                    f" rollout_policy_share={rollout_policy_share:.3f}"
                )
            rollout_wall_total_ms = batch_rollout_policy_wall_ms + batch_rollout_transition_wall_ms
            if rollout_wall_total_ms > 0.0:
                rollout_policy_wall_share = float(batch_rollout_policy_wall_ms / max(1e-9, rollout_wall_total_ms))
                transition_y_share = float(
                    batch_rollout_transition_y_wall_ms / max(1e-9, batch_rollout_transition_wall_ms)
                )
                transition_x_share = float(
                    batch_rollout_transition_x_wall_ms / max(1e-9, batch_rollout_transition_wall_ms)
                )
                transition_group_share = float(
                    batch_rollout_transition_group_wall_ms / max(1e-9, batch_rollout_transition_wall_ms)
                )
                transition_env_pack_share = float(
                    batch_rollout_transition_env_pack_wall_ms / max(1e-9, batch_rollout_transition_wall_ms)
                )
                transition_state_update_share = float(
                    batch_rollout_transition_state_update_wall_ms / max(1e-9, batch_rollout_transition_wall_ms)
                )
                rollout_breakdown_suffix += (
                    f" rollout_policy_wall_ms={batch_rollout_policy_wall_ms:.2f}"
                    f" rollout_transition_wall_ms={batch_rollout_transition_wall_ms:.2f}"
                    f" rollout_policy_wall_share={rollout_policy_wall_share:.3f}"
                    f" rollout_transition_y_share={transition_y_share:.3f}"
                    f" rollout_transition_x_share={transition_x_share:.3f}"
                    f" rollout_transition_group_share={transition_group_share:.3f}"
                    f" rollout_transition_env_pack_share={transition_env_pack_share:.3f}"
                    f" rollout_transition_state_update_share={transition_state_update_share:.3f}"
                    f" rollout_transition_group_count={int(batch_rollout_transition_group_count)}"
                )
            if batch_policy_step_total_ms > 0.0:
                policy_step_encode_share = float(batch_policy_step_encode_ms / max(1e-9, batch_policy_step_total_ms))
                policy_step_transformer_share = float(
                    batch_policy_step_transformer_ms / max(1e-9, batch_policy_step_total_ms)
                )
                policy_step_decoder_share = float(batch_policy_step_decoder_ms / max(1e-9, batch_policy_step_total_ms))
                rollout_breakdown_suffix += (
                    f" policy_step_calls={int(batch_policy_step_calls)}"
                    f" policy_step_total_ms={batch_policy_step_total_ms:.2f}"
                    f" policy_step_encode_share={policy_step_encode_share:.3f}"
                    f" policy_step_transformer_share={policy_step_transformer_share:.3f}"
                    f" policy_step_decoder_share={policy_step_decoder_share:.3f}"
                )
            if batch_policy_step_transformer_layer_total_ms > 0.0:
                policy_step_layer_proj_share = float(
                    batch_policy_step_transformer_layer_proj_ms / max(1e-9, batch_policy_step_transformer_layer_total_ms)
                )
                policy_step_layer_cache_share = float(
                    batch_policy_step_transformer_layer_cache_ms / max(1e-9, batch_policy_step_transformer_layer_total_ms)
                )
                policy_step_layer_attnff_share = float(
                    batch_policy_step_transformer_layer_attnff_ms / max(1e-9, batch_policy_step_transformer_layer_total_ms)
                )
                policy_step_layer_attn_core_share = float(
                    batch_policy_step_transformer_layer_attn_core_ms
                    / max(1e-9, batch_policy_step_transformer_layer_total_ms)
                )
                policy_step_layer_finalize_share = float(
                    batch_policy_step_transformer_layer_finalize_ms
                    / max(1e-9, batch_policy_step_transformer_layer_total_ms)
                )
                policy_step_layer_finalize_attn_outproj_share = float(
                    batch_policy_step_transformer_layer_finalize_attn_outproj_ms
                    / max(1e-9, batch_policy_step_transformer_layer_total_ms)
                )
                policy_step_layer_finalize_ffn_share = float(
                    batch_policy_step_transformer_layer_finalize_ffn_ms
                    / max(1e-9, batch_policy_step_transformer_layer_total_ms)
                )
                rollout_breakdown_suffix += (
                    f" policy_step_tf_layer_calls={int(batch_policy_step_transformer_layer_calls)}"
                    f" policy_step_tf_layer_total_ms={batch_policy_step_transformer_layer_total_ms:.2f}"
                    f" policy_step_tf_layer_proj_share={policy_step_layer_proj_share:.3f}"
                    f" policy_step_tf_layer_cache_share={policy_step_layer_cache_share:.3f}"
                    f" policy_step_tf_layer_attnff_share={policy_step_layer_attnff_share:.3f}"
                    f" policy_step_tf_layer_attn_core_share={policy_step_layer_attn_core_share:.3f}"
                    f" policy_step_tf_layer_finalize_share={policy_step_layer_finalize_share:.3f}"
                    f" policy_step_tf_layer_finalize_attn_outproj_share={policy_step_layer_finalize_attn_outproj_share:.3f}"
                    f" policy_step_tf_layer_finalize_ffn_share={policy_step_layer_finalize_ffn_share:.3f}"
                )
                paged_dispatch_total = int(
                    batch_policy_step_layer_paged_single_page_calls
                    + batch_policy_step_layer_paged_flash_prefix_calls
                    + batch_policy_step_layer_paged_flash_merge_calls
                    + batch_policy_step_layer_paged_dense_calls
                )
                if paged_dispatch_total > 0:
                    rollout_breakdown_suffix += (
                        f" policy_step_paged_dispatch(single/flashp/flashm/dense)="
                        f"{batch_policy_step_layer_paged_single_page_calls}/"
                        f"{batch_policy_step_layer_paged_flash_prefix_calls}/"
                        f"{batch_policy_step_layer_paged_flash_merge_calls}/"
                        f"{batch_policy_step_layer_paged_dense_calls}"
                    )
            if rollout_cuda_busy_ratio is not None:
                rollout_breakdown_suffix += f" rollout_cuda_busy_ratio={rollout_cuda_busy_ratio:.3f}"
            if backward_cuda_busy_ratio is not None:
                rollout_breakdown_suffix += f" backward_cuda_busy_ratio={backward_cuda_busy_ratio:.3f}"
            if batch_pg_loss_signature is not None:
                rollout_breakdown_suffix += f" pg_loss_sig={batch_pg_loss_signature}"

            if gpu_observer_active:
                epoch_obs = int(epoch_idx if epoch_idx is not None else getattr(dl, "epoch_count", -1))
                stage_windows = {
                    "rollout": (rollout_t_start_unix, rollout_t_end_unix),
                    "backward": (backward_t_start_unix, backward_t_end_unix),
                }
                batch_status = "oom" if batch_oom else ("loss_nonfinite" if batch_nonfinite else "ok")
                for stage_name, (stage_t0, stage_t1) in stage_windows.items():
                    if stage_t0 is None or stage_t1 is None:
                        continue
                    duration_sec = float(max(1e-9, stage_t1 - stage_t0))
                    cuda_ms = stage_cuda_ms.get(stage_name, None)
                    cuda_busy_ratio = None
                    if cuda_ms is not None:
                        cuda_busy_ratio = float((cuda_ms / 1000.0) / duration_sec)
                    stage_extra = {
                        "status": batch_status,
                        "chunk_size": int(current_rollout_chunk_size),
                        "tbptt_window": (None if current_tbptt_window is None else int(current_tbptt_window)),
                        "cuda_elapsed_ms": cuda_ms,
                        "cuda_busy_ratio": cuda_busy_ratio,
                    }
                    if stage_name == "rollout" and rollout_breakdown_total_ms > 0.0:
                        stage_extra["rollout_policy_cuda_ms"] = float(batch_rollout_policy_cuda_ms)
                        stage_extra["rollout_transition_cuda_ms"] = float(batch_rollout_transition_cuda_ms)
                        stage_extra["rollout_policy_share"] = float(
                            batch_rollout_policy_cuda_ms / max(1e-9, rollout_breakdown_total_ms)
                        )
                    if stage_name == "rollout" and rollout_wall_total_ms > 0.0:
                        stage_extra["rollout_policy_wall_ms"] = float(batch_rollout_policy_wall_ms)
                        stage_extra["rollout_transition_wall_ms"] = float(batch_rollout_transition_wall_ms)
                        stage_extra["rollout_policy_wall_share"] = float(
                            batch_rollout_policy_wall_ms / max(1e-9, rollout_wall_total_ms)
                        )
                        stage_extra["rollout_transition_y_share"] = float(
                            batch_rollout_transition_y_wall_ms / max(1e-9, batch_rollout_transition_wall_ms)
                        )
                        stage_extra["rollout_transition_x_share"] = float(
                            batch_rollout_transition_x_wall_ms / max(1e-9, batch_rollout_transition_wall_ms)
                        )
                        stage_extra["rollout_transition_group_share"] = float(
                            batch_rollout_transition_group_wall_ms / max(1e-9, batch_rollout_transition_wall_ms)
                        )
                        stage_extra["rollout_transition_env_pack_share"] = float(
                            batch_rollout_transition_env_pack_wall_ms / max(1e-9, batch_rollout_transition_wall_ms)
                        )
                        stage_extra["rollout_transition_state_update_share"] = float(
                            batch_rollout_transition_state_update_wall_ms / max(1e-9, batch_rollout_transition_wall_ms)
                        )
                        stage_extra["rollout_transition_group_count"] = int(batch_rollout_transition_group_count)
                    if stage_name == "rollout" and batch_policy_step_total_ms > 0.0:
                        stage_extra["policy_step_calls"] = int(batch_policy_step_calls)
                        stage_extra["policy_step_total_ms"] = float(batch_policy_step_total_ms)
                        stage_extra["policy_step_encode_share"] = float(
                            batch_policy_step_encode_ms / max(1e-9, batch_policy_step_total_ms)
                        )
                        stage_extra["policy_step_transformer_share"] = float(
                            batch_policy_step_transformer_ms / max(1e-9, batch_policy_step_total_ms)
                        )
                        stage_extra["policy_step_decoder_share"] = float(
                            batch_policy_step_decoder_ms / max(1e-9, batch_policy_step_total_ms)
                        )
                    if stage_name == "rollout" and batch_policy_step_transformer_layer_total_ms > 0.0:
                        stage_extra["policy_step_tf_layer_calls"] = int(batch_policy_step_transformer_layer_calls)
                        stage_extra["policy_step_tf_layer_total_ms"] = float(
                            batch_policy_step_transformer_layer_total_ms
                        )
                        stage_extra["policy_step_tf_layer_proj_share"] = float(
                            batch_policy_step_transformer_layer_proj_ms
                            / max(1e-9, batch_policy_step_transformer_layer_total_ms)
                        )
                        stage_extra["policy_step_tf_layer_cache_share"] = float(
                            batch_policy_step_transformer_layer_cache_ms
                            / max(1e-9, batch_policy_step_transformer_layer_total_ms)
                        )
                        stage_extra["policy_step_tf_layer_attnff_share"] = float(
                            batch_policy_step_transformer_layer_attnff_ms
                            / max(1e-9, batch_policy_step_transformer_layer_total_ms)
                        )
                        stage_extra["policy_step_tf_layer_attn_core_share"] = float(
                            batch_policy_step_transformer_layer_attn_core_ms
                            / max(1e-9, batch_policy_step_transformer_layer_total_ms)
                        )
                        stage_extra["policy_step_tf_layer_finalize_share"] = float(
                            batch_policy_step_transformer_layer_finalize_ms
                            / max(1e-9, batch_policy_step_transformer_layer_total_ms)
                        )
                        stage_extra["policy_step_tf_layer_finalize_attn_outproj_share"] = float(
                            batch_policy_step_transformer_layer_finalize_attn_outproj_ms
                            / max(1e-9, batch_policy_step_transformer_layer_total_ms)
                        )
                        stage_extra["policy_step_tf_layer_finalize_ffn_share"] = float(
                            batch_policy_step_transformer_layer_finalize_ffn_ms
                            / max(1e-9, batch_policy_step_transformer_layer_total_ms)
                        )
                    if stage_name == "rollout" and batch_pg_loss_signature is not None:
                        stage_extra["pg_loss_sig"] = str(batch_pg_loss_signature)
                    stage_rec = gpu_observer.record_stage(
                        epoch=epoch_obs,
                        batch=int(batch),
                        stage=stage_name,
                        start_time_unix=float(stage_t0),
                        end_time_unix=float(stage_t1),
                        extra=stage_extra,
                    )
                    if should_log_phase:
                        breakdown_info = ""
                        if stage_name == "rollout":
                            breakdown_info = rollout_breakdown_suffix
                        print(
                            f"[pg-gpu-stage] epoch={epoch_obs} batch={batch} stage={stage_name} "
                            f"{_format_gpu_stage_summary('', stage_rec)} "
                            f"cuda_ms={'na' if cuda_ms is None else f'{cuda_ms:.2f}'} "
                            f"cuda_busy_ratio={'na' if cuda_busy_ratio is None else f'{cuda_busy_ratio:.3f}'}"
                            f"{breakdown_info}"
                        )
                    if wandb.run is not None:
                        wandb_payload = {
                            f"pg_gpu/{stage_name}_samples": int(stage_rec.get("samples", 0) or 0),
                            f"pg_gpu/{stage_name}_gpu_util_avg": stage_rec.get("gpu_util_avg"),
                            f"pg_gpu/{stage_name}_gpu_util_max": stage_rec.get("gpu_util_max"),
                            f"pg_gpu/{stage_name}_proc_mem_avg_mib": stage_rec.get("process_mem_avg_mib"),
                            f"pg_gpu/{stage_name}_proc_mem_max_mib": stage_rec.get("process_mem_max_mib"),
                            f"pg_gpu/{stage_name}_proc_mem_share_avg": stage_rec.get("process_mem_share_avg"),
                            f"pg_gpu/{stage_name}_proc_sm_util_avg": stage_rec.get("process_sm_util_avg"),
                            f"pg_gpu/{stage_name}_proc_mem_util_avg": stage_rec.get("process_mem_util_avg"),
                        }
                        if cuda_ms is not None:
                            wandb_payload[f"pg_gpu/{stage_name}_cuda_elapsed_ms"] = float(cuda_ms)
                        if cuda_busy_ratio is not None:
                            wandb_payload[f"pg_gpu/{stage_name}_cuda_busy_ratio"] = float(cuda_busy_ratio)
                        if stage_name == "rollout" and rollout_breakdown_total_ms > 0.0:
                            wandb_payload["pg_gpu/rollout_policy_cuda_ms"] = float(batch_rollout_policy_cuda_ms)
                            wandb_payload["pg_gpu/rollout_transition_cuda_ms"] = float(batch_rollout_transition_cuda_ms)
                            wandb_payload["pg_gpu/rollout_policy_share"] = float(
                                batch_rollout_policy_cuda_ms / max(1e-9, rollout_breakdown_total_ms)
                            )
                        if stage_name == "rollout" and rollout_wall_total_ms > 0.0:
                            wandb_payload["pg_gpu/rollout_policy_wall_ms"] = float(batch_rollout_policy_wall_ms)
                            wandb_payload["pg_gpu/rollout_transition_wall_ms"] = float(batch_rollout_transition_wall_ms)
                            wandb_payload["pg_gpu/rollout_policy_wall_share"] = float(
                                batch_rollout_policy_wall_ms / max(1e-9, rollout_wall_total_ms)
                            )
                            wandb_payload["pg_gpu/rollout_transition_y_share"] = float(
                                batch_rollout_transition_y_wall_ms / max(1e-9, batch_rollout_transition_wall_ms)
                            )
                            wandb_payload["pg_gpu/rollout_transition_x_share"] = float(
                                batch_rollout_transition_x_wall_ms / max(1e-9, batch_rollout_transition_wall_ms)
                            )
                            wandb_payload["pg_gpu/rollout_transition_group_share"] = float(
                                batch_rollout_transition_group_wall_ms / max(1e-9, batch_rollout_transition_wall_ms)
                            )
                            wandb_payload["pg_gpu/rollout_transition_env_pack_share"] = float(
                                batch_rollout_transition_env_pack_wall_ms / max(1e-9, batch_rollout_transition_wall_ms)
                            )
                            wandb_payload["pg_gpu/rollout_transition_state_update_share"] = float(
                                batch_rollout_transition_state_update_wall_ms / max(1e-9, batch_rollout_transition_wall_ms)
                            )
                            wandb_payload["pg_gpu/rollout_transition_group_count"] = int(
                                batch_rollout_transition_group_count
                            )
                        if stage_name == "rollout" and batch_policy_step_total_ms > 0.0:
                            wandb_payload["pg_gpu/policy_step_calls"] = int(batch_policy_step_calls)
                            wandb_payload["pg_gpu/policy_step_total_ms"] = float(batch_policy_step_total_ms)
                            wandb_payload["pg_gpu/policy_step_encode_share"] = float(
                                batch_policy_step_encode_ms / max(1e-9, batch_policy_step_total_ms)
                            )
                            wandb_payload["pg_gpu/policy_step_transformer_share"] = float(
                                batch_policy_step_transformer_ms / max(1e-9, batch_policy_step_total_ms)
                            )
                            wandb_payload["pg_gpu/policy_step_decoder_share"] = float(
                                batch_policy_step_decoder_ms / max(1e-9, batch_policy_step_total_ms)
                            )
                        if stage_name == "rollout" and batch_policy_step_transformer_layer_total_ms > 0.0:
                            wandb_payload["pg_gpu/policy_step_tf_layer_calls"] = int(
                                batch_policy_step_transformer_layer_calls
                            )
                            wandb_payload["pg_gpu/policy_step_tf_layer_total_ms"] = float(
                                batch_policy_step_transformer_layer_total_ms
                            )
                            wandb_payload["pg_gpu/policy_step_tf_layer_proj_share"] = float(
                                batch_policy_step_transformer_layer_proj_ms
                                / max(1e-9, batch_policy_step_transformer_layer_total_ms)
                            )
                            wandb_payload["pg_gpu/policy_step_tf_layer_cache_share"] = float(
                                batch_policy_step_transformer_layer_cache_ms
                                / max(1e-9, batch_policy_step_transformer_layer_total_ms)
                            )
                            wandb_payload["pg_gpu/policy_step_tf_layer_attnff_share"] = float(
                                batch_policy_step_transformer_layer_attnff_ms
                                / max(1e-9, batch_policy_step_transformer_layer_total_ms)
                            )
                            wandb_payload["pg_gpu/policy_step_tf_layer_attn_core_share"] = float(
                                batch_policy_step_transformer_layer_attn_core_ms
                                / max(1e-9, batch_policy_step_transformer_layer_total_ms)
                            )
                            wandb_payload["pg_gpu/policy_step_tf_layer_finalize_share"] = float(
                                batch_policy_step_transformer_layer_finalize_ms
                                / max(1e-9, batch_policy_step_transformer_layer_total_ms)
                            )
                            wandb_payload["pg_gpu/policy_step_tf_layer_finalize_attn_outproj_share"] = float(
                                batch_policy_step_transformer_layer_finalize_attn_outproj_ms
                                / max(1e-9, batch_policy_step_transformer_layer_total_ms)
                            )
                            wandb_payload["pg_gpu/policy_step_tf_layer_finalize_ffn_share"] = float(
                                batch_policy_step_transformer_layer_finalize_ffn_ms
                                / max(1e-9, batch_policy_step_transformer_layer_total_ms)
                            )
                        if stage_name == "rollout" and batch_pg_loss_signature is not None:
                            wandb_payload["pg_gpu/pg_loss_sig"] = str(batch_pg_loss_signature)
                        wandb.log(wandb_payload)

            if batch_oom:
                skipped_steps += 1
                stable_batches_for_chunk_growth = 0
                optimizer.zero_grad(set_to_none=True)
                grad_accum_steps = 0
                if should_log_phase:
                    print(
                        f"[pg-phase] epoch={batch_epoch_for_profile} batch={batch} "
                        f"rollout_s={batch_rollout_wall:.3f} backward_s={batch_backward_wall:.3f} "
                        f"step_s={batch_step_wall:.3f} backward_calls={batch_backward_calls} "
                        f"chunk={current_rollout_chunk_size} tbptt={current_tbptt_window} status=oom"
                        f"{rollout_breakdown_suffix}"
                    )
                if epoch_profiler is not None:
                    epoch_profiler.record_batch(valid=False, batch_size=batch_size, n_samples=n_samples)
                    _maybe_emit_profile_interval(
                        epoch_profiler,
                        epoch_start_time=epoch_start_time,
                        epoch_idx=epoch_idx,
                        batch_idx=batch,
                        train_profiler_log_every_batches=train_profiler_log_every_batches,
                        verbose=verbose,
                    )
                if wandb.run is not None:
                    wandb.log(
                        {
                            'train_skip': 1,
                            'train_skip_reason': 'policy_rollout_oom',
                            'policy_rollout_chunk_size': current_rollout_chunk_size,
                        }
                    )
                _maybe_step_kernel_profiler(
                    kernel_profiler,
                    epoch_idx=batch_epoch_for_profile,
                    batch_idx=batch,
                    verbose=verbose,
                )
                continue

            if progress_bar and hasattr(iterator, "set_postfix"):
                step_eval_pos = int(single_eval_pos)
                iterator.set_postfix(
                    step=f"{batch + 1}/{steps_per_epoch}",
                    train_ctx=step_eval_pos,
                    test_ctx=int(n_samples - step_eval_pos),
                )

            if batch_nonfinite:
                skipped_steps += 1
                stable_batches_for_chunk_growth = 0
                optimizer.zero_grad(set_to_none=True)
                grad_accum_steps = 0
                if should_log_phase:
                    print(
                        f"[pg-phase] epoch={batch_epoch_for_profile} batch={batch} "
                        f"rollout_s={batch_rollout_wall:.3f} backward_s={batch_backward_wall:.3f} "
                        f"step_s={batch_step_wall:.3f} backward_calls={batch_backward_calls} "
                        f"chunk={current_rollout_chunk_size} tbptt={current_tbptt_window} status=loss_nonfinite"
                        f"{rollout_breakdown_suffix}"
                    )
                print(f"[train-skip] epoch={batch_epoch_for_profile} batch={batch} reason=loss_nonfinite loss={loss_val}")
                if epoch_profiler is not None:
                    epoch_profiler.record_batch(valid=False, batch_size=batch_size, n_samples=n_samples)
                    _maybe_emit_profile_interval(
                        epoch_profiler,
                        epoch_start_time=epoch_start_time,
                        epoch_idx=epoch_idx,
                        batch_idx=batch,
                        train_profiler_log_every_batches=train_profiler_log_every_batches,
                        verbose=verbose,
                    )
                if wandb.run is not None:
                    wandb.log({'train_skip': 1, 'train_skip_reason': 'loss_nonfinite', 'train_skip_loss': loss_val})
                _maybe_step_kernel_profiler(
                    kernel_profiler,
                    epoch_idx=batch_epoch_for_profile,
                    batch_idx=batch,
                    verbose=verbose,
                )
                continue

            if wandb.run is not None:
                wandb_payload = {
                    'batch_loss': float(batch_loss.detach().mean().cpu()) * aggregate_k_gradients,
                    'policy_objective': float(batch_objective.detach().cpu()),
                    'policy_reward_mean': float(batch_reward_mean.detach().cpu()),
                    'policy_reward_std': float(batch_reward_std.detach().cpu()),
                    'policy_reward_clip_hit_share': float(batch_reward_clip_hit_share),
                    'policy_reward_norm_clip_hit_share': float(batch_reward_norm_clip_hit_share),
                }
                if math.isfinite(batch_reward_min_value):
                    wandb_payload['policy_reward_min'] = float(batch_reward_min_value)
                if math.isfinite(batch_reward_max_value):
                    wandb_payload['policy_reward_max'] = float(batch_reward_max_value)
                if math.isfinite(batch_reward_absmax_value):
                    wandb_payload['policy_reward_absmax'] = float(batch_reward_absmax_value)
                wandb.log(wandb_payload)

            if (scaler is None) and requires_midaccum_grad_finite_check and _has_nonfinite_gradients(model, device=device):
                skipped_steps += 1
                stable_batches_for_chunk_growth = 0
                optimizer.zero_grad(set_to_none=True)
                grad_accum_steps = 0
                print(f"[train-skip] epoch={batch_epoch_for_profile} batch={batch} reason=grad_nonfinite")
                if epoch_profiler is not None:
                    epoch_profiler.record_batch(valid=False, batch_size=batch_size, n_samples=n_samples)
                    _maybe_emit_profile_interval(
                        epoch_profiler,
                        epoch_start_time=epoch_start_time,
                        epoch_idx=epoch_idx,
                        batch_idx=batch,
                        train_profiler_log_every_batches=train_profiler_log_every_batches,
                        verbose=verbose,
                    )
                if wandb.run is not None:
                    wandb.log({'train_skip': 1, 'train_skip_reason': 'grad_nonfinite'})
                _maybe_step_kernel_profiler(
                    kernel_profiler,
                    epoch_idx=batch_epoch_for_profile,
                    batch_idx=batch,
                    verbose=verbose,
                )
                continue

            grad_accum_steps += 1
            if grad_accum_steps == aggregate_k_gradients:
                step_t0_unix = time.time()
                step_t0 = time.perf_counter()
                step_cuda_start = None
                if gpu_observer_active and ("cuda" in str(device)) and torch.cuda.is_available():
                    step_cuda_start = torch.cuda.Event(enable_timing=True)
                    step_cuda_start.record()
                with (
                    kernel_profiler.phase("pg.step")
                    if (kernel_profiler is not None and kernel_profiler.enabled())
                    else nullcontext()
                ):
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., foreach=True)
                try:
                    batch_grad_norm_value = float(grad_norm.detach().float().cpu())
                except Exception:
                    try:
                        batch_grad_norm_value = float(grad_norm)
                    except Exception:
                        batch_grad_norm_value = None
                if not torch.isfinite(grad_norm):
                    batch_step_wall += (time.perf_counter() - step_t0)
                    if step_cuda_start is not None:
                        step_cuda_end = torch.cuda.Event(enable_timing=True)
                        step_cuda_end.record()
                        step_cuda_pairs.append((step_cuda_start, step_cuda_end))
                    step_t1_unix = time.time()
                    if step_t_start_unix is None:
                        step_t_start_unix = step_t0_unix
                    step_t_end_unix = step_t1_unix
                    if gpu_observer_active and step_t_start_unix is not None and step_t_end_unix is not None:
                        step_cuda_ms = None
                        if step_cuda_pairs and ("cuda" in str(device)) and torch.cuda.is_available():
                            torch.cuda.synchronize(device)
                            step_cuda_ms = float(sum(s.elapsed_time(e) for s, e in step_cuda_pairs))
                        step_dur = float(max(1e-9, step_t_end_unix - step_t_start_unix))
                        step_busy_ratio = None if step_cuda_ms is None else float((step_cuda_ms / 1000.0) / step_dur)
                        epoch_obs = int(epoch_idx if epoch_idx is not None else getattr(dl, "epoch_count", -1))
                        stage_rec = gpu_observer.record_stage(
                            epoch=epoch_obs,
                            batch=int(batch),
                            stage="step",
                            start_time_unix=float(step_t_start_unix),
                            end_time_unix=float(step_t_end_unix),
                            extra={
                                "status": "grad_norm_nonfinite",
                                "chunk_size": int(current_rollout_chunk_size),
                                "tbptt_window": (None if current_tbptt_window is None else int(current_tbptt_window)),
                                "cuda_elapsed_ms": step_cuda_ms,
                                "cuda_busy_ratio": step_busy_ratio,
                            },
                        )
                        if should_log_phase:
                            print(
                                f"[pg-gpu-stage] epoch={epoch_obs} batch={batch} stage=step "
                                f"{_format_gpu_stage_summary('', stage_rec)} "
                                f"cuda_ms={'na' if step_cuda_ms is None else f'{step_cuda_ms:.2f}'} "
                                f"cuda_busy_ratio={'na' if step_busy_ratio is None else f'{step_busy_ratio:.3f}'}"
                            )
                        if wandb.run is not None:
                            wandb_payload = {
                                "pg_gpu/step_samples": int(stage_rec.get("samples", 0) or 0),
                                "pg_gpu/step_gpu_util_avg": stage_rec.get("gpu_util_avg"),
                                "pg_gpu/step_gpu_util_max": stage_rec.get("gpu_util_max"),
                                "pg_gpu/step_proc_mem_avg_mib": stage_rec.get("process_mem_avg_mib"),
                                "pg_gpu/step_proc_mem_max_mib": stage_rec.get("process_mem_max_mib"),
                                "pg_gpu/step_proc_mem_share_avg": stage_rec.get("process_mem_share_avg"),
                                "pg_gpu/step_proc_sm_util_avg": stage_rec.get("process_sm_util_avg"),
                                "pg_gpu/step_proc_mem_util_avg": stage_rec.get("process_mem_util_avg"),
                            }
                            if step_cuda_ms is not None:
                                wandb_payload["pg_gpu/step_cuda_elapsed_ms"] = float(step_cuda_ms)
                            if step_busy_ratio is not None:
                                wandb_payload["pg_gpu/step_cuda_busy_ratio"] = float(step_busy_ratio)
                            wandb.log(wandb_payload)
                    skipped_steps += 1
                    stable_batches_for_chunk_growth = 0
                    optimizer.zero_grad(set_to_none=True)
                    grad_accum_steps = 0
                    if scaler is not None:
                        scaler.update()
                    if should_log_phase:
                        print(
                            f"[pg-phase] epoch={batch_epoch_for_profile} batch={batch} "
                            f"rollout_s={batch_rollout_wall:.3f} backward_s={batch_backward_wall:.3f} "
                            f"step_s={batch_step_wall:.3f} backward_calls={batch_backward_calls} "
                            f"chunk={current_rollout_chunk_size} tbptt={current_tbptt_window} status=grad_norm_nonfinite"
                            f"{rollout_breakdown_suffix}"
                        )
                    print(f"[train-skip] epoch={batch_epoch_for_profile} batch={batch} reason=grad_norm_nonfinite")
                    if epoch_profiler is not None:
                        epoch_profiler.record_batch(valid=False, batch_size=batch_size, n_samples=n_samples)
                        _maybe_emit_profile_interval(
                            epoch_profiler,
                            epoch_start_time=epoch_start_time,
                            epoch_idx=epoch_idx,
                            batch_idx=batch,
                            train_profiler_log_every_batches=train_profiler_log_every_batches,
                            verbose=verbose,
                        )
                    if wandb.run is not None:
                        wandb.log({'train_skip': 1, 'train_skip_reason': 'grad_norm_nonfinite'})
                    _maybe_step_kernel_profiler(
                        kernel_profiler,
                        epoch_idx=batch_epoch_for_profile,
                        batch_idx=batch,
                        verbose=verbose,
                    )
                    continue
                with (
                    kernel_profiler.phase("pg.step")
                    if (kernel_profiler is not None and kernel_profiler.enabled())
                    else nullcontext()
                ):
                    if scaler is None:
                        optimizer.step()
                    else:
                        scaler.step(optimizer)
                        scaler.update()
                batch_optimizer_stepped = True
                batch_optimizer_step_value = _extract_optimizer_step(optimizer)
                optimizer.zero_grad(set_to_none=True)
                grad_accum_steps = 0
                batch_step_wall += (time.perf_counter() - step_t0)
                if step_cuda_start is not None:
                    step_cuda_end = torch.cuda.Event(enable_timing=True)
                    step_cuda_end.record()
                    step_cuda_pairs.append((step_cuda_start, step_cuda_end))
                step_t1_unix = time.time()
                if step_t_start_unix is None:
                    step_t_start_unix = step_t0_unix
                step_t_end_unix = step_t1_unix
                if gpu_observer_active and step_t_start_unix is not None and step_t_end_unix is not None:
                    step_cuda_ms = None
                    if step_cuda_pairs and ("cuda" in str(device)) and torch.cuda.is_available():
                        torch.cuda.synchronize(device)
                        step_cuda_ms = float(sum(s.elapsed_time(e) for s, e in step_cuda_pairs))
                    step_dur = float(max(1e-9, step_t_end_unix - step_t_start_unix))
                    step_busy_ratio = None if step_cuda_ms is None else float((step_cuda_ms / 1000.0) / step_dur)
                    epoch_obs = int(epoch_idx if epoch_idx is not None else getattr(dl, "epoch_count", -1))
                    stage_rec = gpu_observer.record_stage(
                        epoch=epoch_obs,
                        batch=int(batch),
                        stage="step",
                        start_time_unix=float(step_t_start_unix),
                        end_time_unix=float(step_t_end_unix),
                        extra={
                            "status": "ok",
                            "chunk_size": int(current_rollout_chunk_size),
                            "tbptt_window": (None if current_tbptt_window is None else int(current_tbptt_window)),
                            "cuda_elapsed_ms": step_cuda_ms,
                            "cuda_busy_ratio": step_busy_ratio,
                        },
                    )
                    if should_log_phase:
                        print(
                            f"[pg-gpu-stage] epoch={epoch_obs} batch={batch} stage=step "
                            f"{_format_gpu_stage_summary('', stage_rec)} "
                            f"cuda_ms={'na' if step_cuda_ms is None else f'{step_cuda_ms:.2f}'} "
                            f"cuda_busy_ratio={'na' if step_busy_ratio is None else f'{step_busy_ratio:.3f}'}"
                        )
                    if wandb.run is not None:
                        wandb_payload = {
                            "pg_gpu/step_samples": int(stage_rec.get("samples", 0) or 0),
                            "pg_gpu/step_gpu_util_avg": stage_rec.get("gpu_util_avg"),
                            "pg_gpu/step_gpu_util_max": stage_rec.get("gpu_util_max"),
                            "pg_gpu/step_proc_mem_avg_mib": stage_rec.get("process_mem_avg_mib"),
                            "pg_gpu/step_proc_mem_max_mib": stage_rec.get("process_mem_max_mib"),
                            "pg_gpu/step_proc_mem_share_avg": stage_rec.get("process_mem_share_avg"),
                            "pg_gpu/step_proc_sm_util_avg": stage_rec.get("process_sm_util_avg"),
                            "pg_gpu/step_proc_mem_util_avg": stage_rec.get("process_mem_util_avg"),
                        }
                        if step_cuda_ms is not None:
                            wandb_payload["pg_gpu/step_cuda_elapsed_ms"] = float(step_cuda_ms)
                        if step_busy_ratio is not None:
                            wandb_payload["pg_gpu/step_cuda_busy_ratio"] = float(step_busy_ratio)
                        wandb.log(wandb_payload)

            total_loss += batch_loss.detach().mean()
            valid_steps += 1
            if should_log_phase:
                grad_norm_info = "na" if batch_grad_norm_value is None else f"{float(batch_grad_norm_value):.3e}"
                opt_step_info = "na" if batch_optimizer_step_value is None else f"{float(batch_optimizer_step_value):.0f}"
                reward_min_info = "na" if not math.isfinite(batch_reward_min_value) else f"{batch_reward_min_value:+.3e}"
                reward_max_info = "na" if not math.isfinite(batch_reward_max_value) else f"{batch_reward_max_value:+.3e}"
                reward_absmax_info = (
                    "na" if not math.isfinite(batch_reward_absmax_value) else f"{batch_reward_absmax_value:.3e}"
                )
                print(
                    f"[pg-phase] epoch={batch_epoch_for_profile} batch={batch} "
                    f"rollout_s={batch_rollout_wall:.3f} backward_s={batch_backward_wall:.3f} "
                    f"step_s={batch_step_wall:.3f} backward_calls={batch_backward_calls} "
                    f"chunk={current_rollout_chunk_size} tbptt={current_tbptt_window} status=ok "
                    f"objective={float(batch_objective.detach().cpu()):+.3e} "
                    f"reward_mean={float(batch_reward_mean.detach().cpu()):+.3e} "
                    f"reward_std={float(batch_reward_std.detach().cpu()):.3e} "
                    f"reward_min={reward_min_info} reward_max={reward_max_info} "
                    f"reward_absmax={reward_absmax_info} "
                    f"clip_hit={float(batch_reward_clip_hit_share):.3f} "
                    f"norm_clip_hit={float(batch_reward_norm_clip_hit_share):.3f} "
                    f"grad_norm={grad_norm_info} "
                    f"stepped={int(bool(batch_optimizer_stepped))} "
                    f"opt_step={opt_step_info}"
                    f"{rollout_breakdown_suffix}"
                )
            if epoch_profiler is not None:
                epoch_profiler.record_batch(valid=True, batch_size=batch_size, n_samples=n_samples)
                _maybe_emit_profile_interval(
                    epoch_profiler,
                    epoch_start_time=epoch_start_time,
                    epoch_idx=epoch_idx,
                    batch_idx=batch,
                    train_profiler_log_every_batches=train_profiler_log_every_batches,
                    verbose=verbose,
                )
            if rollout_chunk_autotune_active and current_rollout_chunk_size < rollout_chunk_cap:
                stable_batches_for_chunk_growth += 1
                if stable_batches_for_chunk_growth >= rollout_chunk_grow_every:
                    grown = int(
                        min(
                            rollout_chunk_cap,
                            max(
                                current_rollout_chunk_size + 1,
                                int(math.ceil(current_rollout_chunk_size * rollout_chunk_grow_factor)),
                            ),
                        )
                    )
                    if grown > current_rollout_chunk_size:
                        current_rollout_chunk_size = grown
                        print(
                            f"[pg-autotune] epoch={batch_epoch_for_profile} batch={batch} "
                            f"increasing policy rollout chunk size to {current_rollout_chunk_size}"
                        )
                    stable_batches_for_chunk_growth = 0
            _maybe_step_kernel_profiler(
                kernel_profiler,
                epoch_idx=batch_epoch_for_profile,
                batch_idx=batch,
                verbose=verbose,
            )

    if grad_accum_steps > 0:
        optimizer.zero_grad(set_to_none=True)

    if valid_steps.item() == 0:
        print("[train-warn] all batches were skipped due to non-finite loss/grad.")
        _restore_policy_inner_recompute_attn(recompute_snapshot)
        return float('inf'), 0.0, 0.0

    if skipped_steps > 0:
        print(f"[train-skip-summary] skipped={skipped_steps} valid={int(valid_steps.item())} total={steps_per_epoch}")
    mean_loss = float((total_loss / valid_steps * aggregate_k_gradients).detach().cpu())
    _restore_policy_inner_recompute_attn(recompute_snapshot)
    return mean_loss, 0.0, 0.0


def train(dl, model, criterion, optimizer_state=None, scheduler=None,
          epochs=10, stop_after_epochs=None, learning_rate=None, min_lr=None, weight_decay=0.0, warmup_epochs=10,
          device='cuda:0',
          aggregate_k_gradients=1, verbose=True, epoch_callback=None, train_mixed_precision=False, adaptive_batch_size=False,
          learning_rate_schedule='cosine', lr_decay=0.99, adam_beta1=0.9, reduce_lr_on_spike=False,
          adamw_fused=True,
          train_profiler_enabled=False,
          train_profiler_output_path=None,
          train_profiler_wandb=True,
          train_profiler_ema_alpha=0.2,
          train_profiler_warmup_epochs=0,
          train_profiler_warmup_batches=0,
          train_profiler_log_every_batches=0,
          train_gpu_observer_enabled=False,
          train_gpu_observer_interval_sec=1.0,
          train_gpu_observer_output_path=None,
          train_gpu_stage_output_path=None,
          train_kernel_profiler_enabled=False,
          train_kernel_profiler_output_dir=None,
          train_kernel_profiler_wait_steps=1,
          train_kernel_profiler_warmup_steps=1,
          train_kernel_profiler_active_steps=3,
          train_kernel_profiler_repeat_steps=1,
          train_kernel_profiler_record_shapes=True,
          train_kernel_profiler_profile_memory=True,
          train_kernel_profiler_with_stack=False,
          train_kernel_profiler_with_flops=False,
          train_kernel_profiler_log_every_batches=0,
          train_kernel_profiler_export_trace=True,
          train_kernel_profiler_summary_top_k=20,
          spike_tolerance=4, progress_bar=False, rl_objective='supervised',
          policy_rollout_chunk_size=None,
          policy_rollout_chunk_autotune=True,
          policy_rollout_chunk_grow_every=8,
          policy_rollout_chunk_grow_factor=2.0,
          policy_rollout_checkpoint=False,
          policy_rollout_checkpoint_reentrant=True,
          pg_grad_mutable_kv_cache=False,
          pg_saved_tensors_cpu_offload=False,
          pg_saved_tensors_pin_memory=True,
          pg_oom_debug_raise=False,
          pg_kv_cache_mode="auto",
          pg_kv_cache_page_size=None,
          pg_tbptt_window=None,
          pg_oom_reduce_tbptt_first=False,
          pg_torch_compile=False,
          pg_torch_compile_backend="inductor",
          pg_torch_compile_mode="reduce-overhead",
          pg_torch_compile_fullgraph=False,
          pg_torch_compile_dynamic=False,
          ):
    using_dist, rank, device = init_dist(device)
    rl_objective = str(rl_objective).strip().lower()
    if rl_objective not in {'supervised', 'policy_gradient'}:
        raise ValueError(f"Unknown rl_objective: {rl_objective}")
    if rank == 0 and verbose:
        print(f'Using {device} device')

    model.to(device)
    if criterion is not None:
        criterion.to(device)

    policy_tf32_prev = None
    if (
        rl_objective == 'policy_gradient'
        and ("cuda" in str(device))
        and torch.cuda.is_available()
    ):
        tf32_flag = str(os.environ.get("TICL_POLICY_TF32", "1")).strip().lower()
        policy_tf32_enabled = tf32_flag not in {"0", "false", "no", "off"}
        if policy_tf32_enabled:
            policy_tf32_prev = bool(torch.backends.cuda.matmul.allow_tf32)
            torch.backends.cuda.matmul.allow_tf32 = True
            if rank == 0 and verbose:
                print(
                    "Policy TF32 matmul:",
                    bool(torch.backends.cuda.matmul.allow_tf32),
                    f"(prev={policy_tf32_prev})",
                )

    env_prior = None
    train_profiler = None
    gpu_observer = None
    kernel_profiler = None
    kernel_profiler_export_trace = (
        True if train_kernel_profiler_export_trace is None else bool(train_kernel_profiler_export_trace)
    )
    kernel_profiler_summary_top_k = (
        20
        if train_kernel_profiler_summary_top_k is None
        else int(max(1, train_kernel_profiler_summary_top_k))
    )
    if rank == 0:
        profiler_cfg = TrainProfilerConfig(
            enabled=bool(train_profiler_enabled),
            output_path=train_profiler_output_path,
            log_to_wandb=bool(train_profiler_wandb),
            ema_alpha=float(train_profiler_ema_alpha),
            warmup_epochs=int(train_profiler_warmup_epochs),
            warmup_batches=int(train_profiler_warmup_batches),
        )
        train_profiler = TrainProfiler(profiler_cfg, rl_objective=rl_objective)
        if train_profiler.enabled() and verbose:
            print(
                "Train profiler: enabled "
                f"(output_path={profiler_cfg.output_path}, ema_alpha={profiler_cfg.ema_alpha:.2f}, "
                f"warmup_epochs={profiler_cfg.warmup_epochs}, warmup_batches={profiler_cfg.warmup_batches}, "
                f"log_every_batches={int(max(0, train_profiler_log_every_batches))}, "
                f"wandb={profiler_cfg.log_to_wandb})"
            )
        gpu_observer = GPUProcessObserver(
            enabled=bool(train_gpu_observer_enabled),
            device=device,
            sample_interval_sec=float(train_gpu_observer_interval_sec),
            output_path=train_gpu_observer_output_path,
            stage_output_path=train_gpu_stage_output_path,
        )
        gpu_observer.start()
        if gpu_observer.enabled() and verbose:
            print(
                "GPU observer: enabled "
                f"(device_index={gpu_observer.device_index}, pid={gpu_observer.pid}, "
                f"interval_sec={gpu_observer.sample_interval_sec:.2f}, "
                f"sample_output={train_gpu_observer_output_path}, "
                f"stage_output={train_gpu_stage_output_path})"
            )
        kernel_profiler_cfg = TrainKernelProfilerConfig(
            enabled=bool(train_kernel_profiler_enabled),
            output_dir=train_kernel_profiler_output_dir,
            wait_steps=int(train_kernel_profiler_wait_steps),
            warmup_steps=int(train_kernel_profiler_warmup_steps),
            active_steps=int(train_kernel_profiler_active_steps),
            repeat_steps=int(train_kernel_profiler_repeat_steps),
            record_shapes=bool(train_kernel_profiler_record_shapes),
            profile_memory=bool(train_kernel_profiler_profile_memory),
            with_stack=bool(train_kernel_profiler_with_stack),
            with_flops=bool(train_kernel_profiler_with_flops),
            log_every_batches=int(train_kernel_profiler_log_every_batches),
            export_trace=kernel_profiler_export_trace,
            summary_top_k=kernel_profiler_summary_top_k,
        )
        kernel_profiler = TrainKernelProfiler(
            kernel_profiler_cfg,
            device=device,
            worker_name=f"rank{rank}_pid{os.getpid()}",
        )
        kernel_profiler.start()
        if kernel_profiler.enabled() and verbose:
            print(
                "Kernel profiler: enabled "
                f"(output_dir={kernel_profiler_cfg.output_dir}, "
                f"schedule={kernel_profiler_cfg.wait_steps}/"
                f"{kernel_profiler_cfg.warmup_steps}/"
                f"{kernel_profiler_cfg.active_steps}/"
                f"{kernel_profiler_cfg.repeat_steps}, "
                f"log_every_batches={kernel_profiler_cfg.log_every_batches}, "
                f"record_shapes={kernel_profiler_cfg.record_shapes}, "
                f"profile_memory={kernel_profiler_cfg.profile_memory}, "
                f"with_stack={kernel_profiler_cfg.with_stack}, "
                f"with_flops={kernel_profiler_cfg.with_flops}, "
                f"export_trace={kernel_profiler_cfg.export_trace}, "
                f"summary_top_k={kernel_profiler_cfg.summary_top_k})"
            )
    if rl_objective == 'policy_gradient':
        if using_dist:
            raise ValueError("policy_gradient objective does not support distributed training yet.")
        env_prior = _resolve_environment_prior(getattr(dl, "prior", None))
        if env_prior is None:
            raise ValueError("policy_gradient objective requires EnvironmentPrior in dataloader prior chain.")
        if rank == 0 and verbose:
            print("Using policy_gradient objective with differentiable EnvironmentPrior rollout.")
            env_backend = str(getattr(env_prior, "config", {}).get("batch_parallel_backend", "python_thread"))
            env_grouping = str(getattr(env_prior, "config", {}).get("batch_vectorized_grouping", "structure"))
            env_strict = bool(getattr(env_prior, "config", {}).get("batch_vectorized_strict_rng_match", False))
            print(
                "Environment rollout backend:",
                env_backend,
                f"(grouping={env_grouping}, strict_rng_match={env_strict})",
            )
            pg_normalize_rewards = bool(getattr(env_prior, "config", {}).get("policy_gradient_normalize_rewards", False))
            pg_discount = float(getattr(env_prior, "config", {}).get("discount", 1.0))
            print(
                "Policy objective:",
                ("normalized_reward_mean" if pg_normalize_rewards else "raw_discounted_reward_mean"),
                f"(discount={pg_discount:.6g})",
            )
            if policy_rollout_chunk_size is None:
                print("Policy rollout chunk size: auto(batch_size)")
            else:
                chunk_print = int(policy_rollout_chunk_size)
                if chunk_print <= 0:
                    chunk_print = int(dl.batch_size)
                print(
                    "Policy rollout chunk size:",
                    int(max(1, min(int(dl.batch_size), chunk_print))),
                )
            print(
                "Policy rollout chunk autotune:",
                bool(policy_rollout_chunk_autotune),
                f"(grow_every={int(max(1, policy_rollout_chunk_grow_every))}, "
                f"grow_factor={float(policy_rollout_chunk_grow_factor):.2f})",
            )
            print("Policy rollout checkpoint:", bool(policy_rollout_checkpoint))
            tbptt_streaming_default = False
            if pg_tbptt_window is not None:
                try:
                    tbptt_streaming_default = int(pg_tbptt_window) > 0
                except Exception:
                    tbptt_streaming_default = False
            allow_grad_mutable_kv = bool(pg_grad_mutable_kv_cache) and bool(policy_rollout_checkpoint)
            requested_kv_mode = "auto" if pg_kv_cache_mode is None else str(pg_kv_cache_mode).strip().lower()
            if requested_kv_mode not in {"auto", "immutable", "static", "paged"}:
                requested_kv_mode = "auto"
            if not allow_grad_mutable_kv:
                kv_mode_print = "immutable"
            else:
                kv_mode_print = requested_kv_mode if requested_kv_mode != "auto" else "paged"
                if kv_mode_print == "immutable":
                    allow_grad_mutable_kv = False
            if kv_mode_print == "paged":
                page_size_print = 128 if pg_kv_cache_page_size is None else int(max(1, pg_kv_cache_page_size))
            else:
                page_size_print = None
            inplace_paged_kv_env = str(os.environ.get("TICL_POLICY_INPLACE_PAGED_KV", "auto")).strip().lower()
            if inplace_paged_kv_env in {"1", "true", "yes", "on"}:
                inplace_kv_print = True
            elif inplace_paged_kv_env in {"0", "false", "no", "off"}:
                inplace_kv_print = False
            else:
                inplace_kv_print = False
            if (not allow_grad_mutable_kv) or (kv_mode_print != "paged"):
                inplace_kv_print = False
            mutable_reentrant_safe = (
                (not allow_grad_mutable_kv)
                or (kv_mode_print == "paged")
            )
            checkpoint_reentrant_active = bool(policy_rollout_checkpoint_reentrant) and bool(mutable_reentrant_safe)
            disable_inner_recompute = bool(policy_rollout_checkpoint) and (not tbptt_streaming_default)
            print("Policy rollout checkpoint reentrant:", checkpoint_reentrant_active)
            if disable_inner_recompute:
                print(
                    "Policy inner forward_step recompute_attn: auto-disabled "
                    "(lossless memory route with rollout checkpoint)."
                )
            elif bool(policy_rollout_checkpoint) and bool(tbptt_streaming_default):
                print(
                    "Policy inner forward_step recompute_attn: kept enabled "
                    "(TBPTT streaming path does not use outer rollout checkpoint)."
                )
            print(
                "Policy grad-mutable KV cache:",
                bool(allow_grad_mutable_kv),
            )
            print("Policy KV cache mode:", kv_mode_print)
            if kv_mode_print == "paged":
                print("Policy KV cache page size:", int(page_size_print))
                paged_train_mode = str(os.environ.get("TICL_POLICY_PAGED_ATTN_TRAIN_MODE", "auto")).strip().lower()
                if paged_train_mode not in {"auto", "dense", "flash_merge", "flash_prefix"}:
                    paged_train_mode = "auto"
                print("Policy paged-attn train mode:", paged_train_mode)
                try:
                    dense_page_cap = int(os.environ.get("TICL_POLICY_PAGED_ATTN_DENSE_PAGE_SIZE", "128"))
                except Exception:
                    dense_page_cap = 128
                print("Policy paged-attn dense page cap:", int(max(1, dense_page_cap)))
                print("Policy KV cache in-place append:", bool(inplace_kv_print))
            if bool(policy_rollout_checkpoint_reentrant) and not checkpoint_reentrant_active:
                print(
                    "[pg-checkpoint-note] reentrant checkpoint is disabled because the selected mutable KV mode "
                    "is not reentrant-safe; using non-reentrant checkpoint for autograd safety."
                )
            print("Policy saved-tensors CPU offload:", bool(pg_saved_tensors_cpu_offload))
            print("Policy OOM debug re-raise:", bool(pg_oom_debug_raise))
            if pg_tbptt_window is None:
                print("Policy TBPTT window: disabled(full-horizon)")
            else:
                print("Policy TBPTT window:", int(pg_tbptt_window))
            print("Policy OOM reduce TBPTT first:", bool(pg_oom_reduce_tbptt_first))
            if bool(policy_rollout_checkpoint) and (not checkpoint_reentrant_active):
                if bool(pg_saved_tensors_cpu_offload):
                    print(
                        "[pg-memory-warn] non-reentrant rollout checkpoint + saved-tensors CPU offload "
                        "can shift very large PG activation memory to host RAM."
                    )
                if bool(allow_grad_mutable_kv) and not (
                    kv_mode_print == "paged"
                ):
                    print(
                        "[pg-memory-warn] mutable KV mode with non-reentrant checkpoint may keep "
                        "long-horizon rollout memory high."
                    )
            print("Policy torch.compile:", bool(pg_torch_compile))
            try:
                from torch.nn.attention import sdpa_kernel as _sdpa_kernel_probe  # noqa: F401
                sdpa_api_available = True
            except Exception:
                sdpa_api_available = False
            print("Policy runtime torch:", str(torch.__version__), f"(sdpa_kernel_api={sdpa_api_available})")
            finalize_fastpath_env = str(os.environ.get("TICL_POLICY_FINALIZE_2D_FASTPATH", "1")).strip().lower()
            finalize_fastpath_on = finalize_fastpath_env not in {"0", "false", "no", "off"}
            print("Policy finalize 2D fastpath:", bool(finalize_fastpath_on))
            policy_autocast_dtype = _resolve_policy_autocast_dtype(device)
            if policy_autocast_dtype is not None:
                if policy_autocast_dtype == torch.float16:
                    print("Policy autocast dtype:", "fp16")
                elif policy_autocast_dtype == torch.bfloat16:
                    print("Policy autocast dtype:", "bf16")
                else:
                    print("Policy autocast dtype:", str(policy_autocast_dtype))
            print("GPU observer enabled:", bool(train_gpu_observer_enabled))
            if bool(train_gpu_observer_enabled):
                print(
                    "GPU observer sampling:",
                    f"interval_sec={float(max(0.1, train_gpu_observer_interval_sec)):.2f}",
                    f"samples_jsonl={train_gpu_observer_output_path}",
                    f"stages_jsonl={train_gpu_stage_output_path}",
                )
            print("Kernel profiler enabled:", bool(train_kernel_profiler_enabled))
            if bool(train_kernel_profiler_enabled):
                print(
                    "Kernel profiler schedule:",
                    f"wait={int(max(0, train_kernel_profiler_wait_steps))}",
                    f"warmup={int(max(0, train_kernel_profiler_warmup_steps))}",
                    f"active={int(max(1, train_kernel_profiler_active_steps))}",
                    f"repeat={int(max(1, train_kernel_profiler_repeat_steps))}",
                    f"output_dir={train_kernel_profiler_output_dir}",
                    f"log_every_batches={int(max(0, train_kernel_profiler_log_every_batches))}",
                    f"export_trace={kernel_profiler_export_trace}",
                    f"summary_top_k={kernel_profiler_summary_top_k}",
                )

    n_out = model.n_out
    if using_dist:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank, broadcast_buffers=False)
        if rank == 0:
            print("Distributed training")
    elif "cuda" in device:
        print(f"Single GPU training on {torch.cuda.get_device_name()}")
    elif "cpu" in device:
        pass
    else:
        raise ValueError(f"Invalid device: {device}")

    if rank == 0:
        model.learning_rates = getattr(model, 'learning_rates', [])
        model.losses = getattr(model, 'losses', [])
        model.wallclock_times = getattr(model, 'wallclock_times', [])
        model.start_time = time.time()
        if len(model.wallclock_times):
            model.start_time -= model.wallclock_times[-1]
        if epoch_callback is not None:
            epoch_callback(model, None, None, "start")

    dl.model = model
    adamw_kwargs = dict(lr=learning_rate, weight_decay=weight_decay, betas=(adam_beta1, 0.999))
    optimizer = None
    if bool(adamw_fused) and ("cuda" in str(device)):
        try:
            optimizer = torch.optim.AdamW(model.parameters(), fused=True, **adamw_kwargs)
            if rank == 0 and verbose:
                print("AdamW fused: enabled")
        except Exception as e:
            if rank == 0 and verbose:
                print(f"[optim-warn] fused AdamW unavailable, fallback to standard AdamW: {e}")
            optimizer = torch.optim.AdamW(model.parameters(), **adamw_kwargs)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), **adamw_kwargs)
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    spike_scheduler = None
    if scheduler is None:
        if learning_rate_schedule == 'cosine':
            base_scheduler = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=min_lr)
        elif learning_rate_schedule == 'exponential':
            base_scheduler = ExponentialLR(optimizer, gamma=lr_decay, min_lr=min_lr)
        elif learning_rate_schedule == 'constant':
            base_scheduler = ExponentialLR(optimizer, gamma=1, min_lr=min_lr)
        else:
            raise ValueError(f"Invalid learning rate schedule: {learning_rate_schedule}")
        # add linear warmup to scheduler
        scheduler = SequentialLR(optimizer, [LinearLR(optimizer, start_factor=1e-10, end_factor=1, total_iters=warmup_epochs),
                                             base_scheduler], milestones=[warmup_epochs])

        start_epoch = 1
    else:
        start_epoch = scheduler.last_epoch + 1

    if reduce_lr_on_spike:
        # In this case we're not properly restarting the scheduler when we load a checkpoint, sad
        spike_scheduler = ReduceLROnSpike(optimizer, smoothing=10, factor=0.5, min_lr=min_lr, tolerance=spike_tolerance, verbose=True)
    try:
        scaler_device_type = torch.device(str(device)).type
    except Exception:
        scaler_device_type = "cuda" if "cuda" in str(device) else "cpu"
    scaler = GradScaler("cuda") if train_mixed_precision and scaler_device_type == "cuda" else None

    # check that everything uses up-to-date APIs
    utils.check_compatibility(dl)

    total_loss = float('inf')
    increased_batch_size = 0
    epoch = start_epoch
    if stop_after_epochs is not None:
        epochs = min(epochs, stop_after_epochs)
    if "cuda" in device:
        gpu_start_time = torch.cuda.Event(enable_timing=True)
        gpu_end_time = torch.cuda.Event(enable_timing=True)

    try:
        train_time, inference_time, train_gpu_time = [], [], []
        pg_warmup_note_emitted = False
        for epoch in range(start_epoch, epochs + 1):
            if verbose:
                print(f"start of epoch {epoch}")

            prev_total_loss = total_loss
            epoch_start_time = time.time()
            if train_profiler is not None and train_profiler.enabled():
                train_profiler.start_epoch(epoch)
            if "cuda" in device:
                torch.cuda.reset_peak_memory_stats()
                gpu_start_time.record()
            
            if rl_objective == 'policy_gradient':
                new_loss, nan_share, ignore_share = train_epoch_policy_gradient(
                    model=model,
                    aggregate_k_gradients=aggregate_k_gradients,
                    using_dist=using_dist,
                    scaler=scaler,
                    dl=dl,
                    device=device,
                    optimizer=optimizer,
                    env_prior=env_prior,
                    policy_rollout_chunk_size=policy_rollout_chunk_size,
                    policy_rollout_chunk_autotune=policy_rollout_chunk_autotune,
                    policy_rollout_chunk_grow_every=policy_rollout_chunk_grow_every,
                    policy_rollout_chunk_grow_factor=policy_rollout_chunk_grow_factor,
                    policy_rollout_checkpoint=policy_rollout_checkpoint,
                    policy_rollout_checkpoint_reentrant=policy_rollout_checkpoint_reentrant,
                    pg_grad_mutable_kv_cache=pg_grad_mutable_kv_cache,
                    pg_saved_tensors_cpu_offload=pg_saved_tensors_cpu_offload,
                    pg_saved_tensors_pin_memory=pg_saved_tensors_pin_memory,
                    pg_oom_debug_raise=pg_oom_debug_raise,
                    pg_kv_cache_mode=pg_kv_cache_mode,
                    pg_kv_cache_page_size=pg_kv_cache_page_size,
                    pg_tbptt_window=pg_tbptt_window,
                    pg_oom_reduce_tbptt_first=pg_oom_reduce_tbptt_first,
                    pg_torch_compile=pg_torch_compile,
                    pg_torch_compile_backend=pg_torch_compile_backend,
                    pg_torch_compile_mode=pg_torch_compile_mode,
                    pg_torch_compile_fullgraph=pg_torch_compile_fullgraph,
                    pg_torch_compile_dynamic=pg_torch_compile_dynamic,
                    epoch_idx=epoch,
                    progress_bar=progress_bar,
                    epoch_profiler=train_profiler,
                    epoch_start_time=epoch_start_time,
                    train_profiler_log_every_batches=train_profiler_log_every_batches,
                    verbose=verbose,
                    gpu_observer=gpu_observer,
                    kernel_profiler=kernel_profiler,
                )
            else:
                new_loss, nan_share, ignore_share = train_epoch(
                    model, 
                    aggregate_k_gradients, 
                    using_dist, 
                    scaler, 
                    dl, 
                    device, 
                    optimizer, 
                    criterion, 
                    n_out,
                    epoch_idx=epoch,
                    progress_bar=progress_bar,
                    epoch_profiler=train_profiler,
                    epoch_start_time=epoch_start_time,
                    train_profiler_log_every_batches=train_profiler_log_every_batches,
                    verbose=verbose,
                    kernel_profiler=kernel_profiler,
                )

            total_loss = new_loss
            if spike_scheduler is not None:
                last_lr = spike_scheduler.get_last_lr()[0]
            else:
                last_lr = scheduler.get_last_lr()[0]
            if (
                rl_objective == 'policy_gradient'
                and (not pg_warmup_note_emitted)
                and warmup_epochs > 0
                and epoch <= warmup_epochs
                and learning_rate is not None
                and float(last_lr) <= float(learning_rate) * 1e-3
            ):
                print(
                    "[pg-note] learning rate is still in warmup phase "
                    f"(epoch {epoch}/{warmup_epochs}, lr={last_lr:.3e}, base_lr={float(learning_rate):.3e}); "
                    "early objective fluctuations are expected."
                )
                pg_warmup_note_emitted = True

            train_time.append(time.time() - epoch_start_time)
            if "cuda" in device:
                gpu_end_time.record()
                torch.cuda.synchronize()
                train_gpu_time.append(gpu_start_time.elapsed_time(gpu_end_time)/1000)
                peak_alloc_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)
                peak_reserved_gib = torch.cuda.max_memory_reserved() / (1024 ** 3)
            else:
                train_gpu_time.append(0)
                peak_alloc_gib = None
                peak_reserved_gib = None

            profile_record = None
            if train_profiler is not None and train_profiler.enabled():
                profile_record = train_profiler.end_epoch(
                    wall_time_sec=float(train_time[-1]),
                    gpu_time_sec=float(train_gpu_time[-1]),
                    peak_alloc_gib=peak_alloc_gib,
                    peak_reserved_gib=peak_reserved_gib,
                )
                if (
                    profile_record is not None
                    and wandb.run
                    and bool(train_profiler.config.log_to_wandb)
                ):
                    wandb.log({f"profile/{k}": v for k, v in profile_record.items()})

            gpu_epoch_record = None
            if gpu_observer is not None and gpu_observer.enabled():
                gpu_epoch_record = gpu_observer.window_stats(
                    start_time_unix=float(epoch_start_time),
                    end_time_unix=float(time.time()),
                )
                if wandb.run is not None:
                    wandb.log(
                        {
                            "gpu_epoch/samples": int(gpu_epoch_record.get("samples", 0) or 0),
                            "gpu_epoch/gpu_util_avg": gpu_epoch_record.get("gpu_util_avg"),
                            "gpu_epoch/gpu_util_max": gpu_epoch_record.get("gpu_util_max"),
                            "gpu_epoch/gpu_mem_util_avg": gpu_epoch_record.get("gpu_mem_util_avg"),
                            "gpu_epoch/gpu_power_avg_w": gpu_epoch_record.get("gpu_power_avg_w"),
                            "gpu_epoch/process_mem_avg_mib": gpu_epoch_record.get("process_mem_avg_mib"),
                            "gpu_epoch/process_mem_max_mib": gpu_epoch_record.get("process_mem_max_mib"),
                            "gpu_epoch/process_mem_share_avg": gpu_epoch_record.get("process_mem_share_avg"),
                            "gpu_epoch/process_sm_util_avg": gpu_epoch_record.get("process_sm_util_avg"),
                            "gpu_epoch/process_sm_util_max": gpu_epoch_record.get("process_sm_util_max"),
                            "gpu_epoch/process_mem_util_avg": gpu_epoch_record.get("process_mem_util_avg"),
                            "gpu_epoch/process_mem_util_max": gpu_epoch_record.get("process_mem_util_max"),
                        }
                    )

            if verbose:
                print('-' * 89)
                if "cuda" in device:
                    print(
                        f' peak gpu mem alloc/reserved {peak_alloc_gib:5.2f}GiB/{peak_reserved_gib:5.2f}GiB |',
                    )
                if rl_objective == 'policy_gradient':
                    mean_loss_str = f"{float(total_loss):+.6e}"
                else:
                    mean_loss_str = f"{float(total_loss):5.4f}"
                print(
                    f'| end of epoch {epoch:3d} | Wallclock time: {train_time[-1]:5.2f}s | GPU time: {train_gpu_time[-1]:5.2f}s | mean loss {mean_loss_str} | ')
                if profile_record is not None:
                    print(
                        " profile "
                        f"steps/s {profile_record['batches_per_sec']:.2f} "
                        f"(ema {profile_record['ema_batches_per_sec']:.2f}) | "
                        f"samples/s {profile_record['samples_per_sec']:.2f} | "
                        f"tokens/s {profile_record['tokens_per_sec']:.2f} | "
                        f"valid/skipped {profile_record['steps_valid']}/{profile_record['steps_skipped']}"
                    )
                if gpu_epoch_record is not None:
                    print(
                        " gpu-observer "
                        f"samples {int(gpu_epoch_record.get('samples', 0) or 0)} | "
                        f"gpu_util_avg {gpu_epoch_record.get('gpu_util_avg')} | "
                        f"gpu_util_max {gpu_epoch_record.get('gpu_util_max')} | "
                        f"proc_mem_avg_mib {gpu_epoch_record.get('process_mem_avg_mib')} | "
                        f"proc_mem_max_mib {gpu_epoch_record.get('process_mem_max_mib')} | "
                        f"proc_sm_util_avg {gpu_epoch_record.get('process_sm_util_avg')} | "
                        f"proc_sm_util_max {gpu_epoch_record.get('process_sm_util_max')}"
                    )

                if wandb.run: 
                    wandb.log({"avg_train_time": sum(train_time)/len(train_time), "train_time": train_time[-1]})
                    wandb.log({"avg_train_gpu_time": sum(train_gpu_time)/len(train_gpu_time), "train_gpu_time": train_gpu_time[-1]})

                print(
                    f' lr {last_lr}'
                    f' nan share {nan_share:5.2f} ignore share (for classification tasks) {ignore_share:5.4f}')
                print('-' * 89)
                
            if (
                rl_objective != 'policy_gradient'
                and math.isfinite(prev_total_loss)
                and new_loss > 1.5 * prev_total_loss
            ):
                print("LOSS DIVERGED")
                return total_loss, model.to('cpu'), dl, epoch
            
            if adaptive_batch_size:
                if increased_batch_size == 0 and epoch >= 20:
                    aggregate_k_gradients *= 2
                    increased_batch_size = 1
                    print("increased aggregate_k_gradients size to", aggregate_k_gradients)
                elif increased_batch_size == 1 and epoch >= 50:
                    aggregate_k_gradients *= 2
                    increased_batch_size = 2
                    print("increased aggregate_k_gradients size to", aggregate_k_gradients)
                elif increased_batch_size == 2 and epoch >= 200 and aggregate_k_gradients < 8:
                    aggregate_k_gradients *= 2
                    increased_batch_size = 3
                    print("increased aggregate_k_gradients size to", aggregate_k_gradients)
                if aggregate_k_gradients > 8:
                    aggregate_k_gradients = 8
                    
            scheduler.step()
            if spike_scheduler is not None:
                spike_scheduler.step(metrics=total_loss)
            # stepping with wallclock time based scheduler
            if epoch_callback is not None and rank == 0:
                model.learning_rates.append(last_lr)
                model.losses.append(total_loss)
                model.wallclock_times.append(time.time() - model.start_time)
                output = epoch_callback(model, optimizer, scheduler, epoch)
                if output: 
                    if isinstance(output, dict):
                        inf_t = output.get("inference_time", None)
                        if inf_t is not None:
                            inference_time.append(inf_t)
                            if wandb.run:
                                wandb.log({"avg_inference_time": sum(inference_time)/len(inference_time), "inference_time": inference_time[-1]})
                    else:
                        inference_time.append(output)
                        if wandb.run:
                            wandb.log({"avg_inference_time": sum(inference_time)/len(inference_time), "inference_time": inference_time[-1]})


    except KeyboardInterrupt:
        pass
    finally:
        if kernel_profiler is not None:
            kernel_profiler.stop()
        if gpu_observer is not None:
            gpu_observer.stop()
        if policy_tf32_prev is not None:
            torch.backends.cuda.matmul.allow_tf32 = bool(policy_tf32_prev)

    if rank == 0:  # trivially true for non-parallel training
        return total_loss, model.to('cpu'), dl, epoch
