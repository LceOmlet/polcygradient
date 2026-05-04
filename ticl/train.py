import inspect
import math
import os
import json
import time, wandb
import random
import gc
import traceback
from collections import OrderedDict
from contextlib import nullcontext
from functools import wraps

import torch
import numpy as np
from torch import nn
from tqdm import tqdm
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.checkpoint import checkpoint
try:
    import psutil
except Exception:  # pragma: no cover - optional runtime dependency
    psutil = None

import ticl.utils as utils
from ticl.utils import ExponentialLR, ReduceLROnSpike, init_dist
from ticl.profiling import TrainProfiler, TrainProfilerConfig
from ticl.gpu_observer import GPUProcessObserver
from ticl.kernel_profiling import TrainKernelProfiler, TrainKernelProfilerConfig
from ticl.priors.environment_prior import EnvironmentPrior
from ticl.priors.maintained_policy_gradient_loss import attach_common_rollout_diagnostics

def _has_nonfinite_gradients(model, device):
    has_nonfinite = torch.zeros((), dtype=torch.bool, device=device)
    for p in model.parameters():
        g = p.grad
        if g is not None:
            has_nonfinite = has_nonfinite | (~torch.isfinite(g).all())
    return bool(has_nonfinite.item())


def _env_flag_enabled(name, default="0"):
    return str(os.environ.get(name, default)).strip().lower() not in {"0", "false", "no", "off"}


def _looks_like_maintained_rlpfn_env_prior(env_prior) -> bool:
    cfg = getattr(env_prior, "config", {}) or {}
    return bool(
        cfg.get("strict_joint_transition_enabled", False)
        and cfg.get("reinforce_sequence_replay_enabled", False)
    )


def _get_host_memory_snapshot():
    if psutil is None:
        return None
    try:
        proc = psutil.Process(os.getpid())
        rss_gib = float(proc.memory_info().rss) / (1024.0 ** 3)
        avail_gib = float(psutil.virtual_memory().available) / (1024.0 ** 3)
        return {"rss_gib": rss_gib, "avail_gib": avail_gib}
    except Exception:
        return None


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


_PPO_TRAIN_DIAGNOSTIC_KEYS = {
    "objective_total",
    "explained_variance_value_target",
    "explained_variance_value_target_objective",
    "explained_variance_raw_value_target",
    "explained_variance_raw_value_target_objective",
    "explained_variance_raw_discounted_return",
    "explained_variance_raw_discounted_return_objective",
    "raw_value_target_corr",
    "raw_discounted_return_corr",
}
_PPO_TRAIN_DIAGNOSTIC_PREFIXES = (
    "value_space_",
    "raw_value_",
    "raw_value_target_",
    "raw_discounted_return_",
    "value_scale_",
    "input_",
)


def _copy_ppo_train_diagnostic_logger_values(target: dict, logger_values: dict) -> None:
    for key, value in logger_values.items():
        key_str = str(key)
        metric_name = None
        if key_str.startswith("train/legacy_pre_rollout_"):
            metric_name = key_str.removeprefix("train/legacy_pre_rollout_")
        elif key_str.startswith("train/"):
            candidate = key_str.removeprefix("train/")
            if candidate in _PPO_TRAIN_DIAGNOSTIC_KEYS or candidate.startswith(_PPO_TRAIN_DIAGNOSTIC_PREFIXES):
                metric_name = candidate
        if metric_name is not None:
            target[str(metric_name)] = value


def _iter_validation_policy_state_carriers(model):
    queue = [model]
    seen = set()
    while queue:
        candidate = queue.pop(0)
        if candidate is None:
            continue
        ident = id(candidate)
        if ident in seen:
            continue
        seen.add(ident)
        yield candidate
        for nested_attr in ("module", "model", "_orig_mod"):
            nested = getattr(candidate, nested_attr, None)
            if nested is not None and id(nested) not in seen:
                queue.append(nested)


def _resolve_saved_validation_ppo_policy_state(model):
    for candidate in _iter_validation_policy_state_carriers(model):
        candidate_dict = getattr(candidate, "__dict__", {})
        saved_state = candidate_dict.get("_validation_ppo_policy_state", None)
        if isinstance(saved_state, dict):
            return saved_state
    return None


def _parse_ppo_seed_spec(seed_spec, *, name):
    if seed_spec is None:
        return None
    if torch.is_tensor(seed_spec):
        seed_spec = seed_spec.detach().cpu().tolist()
    if isinstance(seed_spec, np.ndarray):
        seed_spec = seed_spec.tolist()
    if isinstance(seed_spec, (list, tuple)):
        return [int(v) for v in seed_spec]
    if isinstance(seed_spec, (int, np.integer)):
        return int(seed_spec)
    seed_text = str(seed_spec).strip()
    if seed_text == "":
        return None
    try:
        parsed = json.loads(seed_text)
    except Exception:
        parsed = seed_text
    if parsed is None:
        return None
    if isinstance(parsed, (list, tuple)):
        return [int(v) for v in parsed]
    if isinstance(parsed, (int, np.integer)):
        return int(parsed)
    if isinstance(parsed, str):
        parsed_text = parsed.strip()
        if parsed_text == "":
            return None
        if "," in parsed_text:
            return [int(part.strip()) for part in parsed_text.split(",") if part.strip() != ""]
        return int(parsed_text)
    raise ValueError(f"Unsupported {name} seed spec: {seed_spec!r}")


def _resolve_recurrent_ppo_training_bridge_kwargs(
    *,
    model,
    env_prior,
    device,
    num_features,
    n_envs,
    n_steps,
    learning_rate,
    batch_size,
    n_epochs,
    gamma,
    gae_lambda,
    clip_range,
    clip_range_vf,
    normalize_advantage,
    space_contract="normalized",
    value_target_space=None,
    actor_gae_space=None,
    allow_mixed_space_contract=False,
    actor_baseline_mode,
    separate_value_backbone,
    reset_env_state_at_sep,
    value_head_impl="legacy_bar",
    value_path_adapter_impl="none",
    value_head_mlp_hidden_dim=256,
    runtime_normalized_q_value_weight_override,
    runtime_next_state_flow_matching_weight_override,
    ent_coef,
    vf_coef,
    max_grad_norm,
    target_kl,
    verbose,
    ppo_restore_validation_policy_state,
    ppo_strict_fixed_env_mode,
    ppo_env_rng_seeds,
    ppo_rollout_rng_seeds,
    ppo_deterministic_actor_sampling,
    ppo_deterministic_batch_plan,
    ppo_strict_native_rollout,
):
    saved_validation_policy_state = _resolve_saved_validation_ppo_policy_state(model)
    has_saved_validation_policy_state = isinstance(saved_validation_policy_state, dict)
    restore_validation_policy_state = (
        bool(has_saved_validation_policy_state)
        if ppo_restore_validation_policy_state is None
        else bool(ppo_restore_validation_policy_state)
    )
    env_rng_seeds = _parse_ppo_seed_spec(ppo_env_rng_seeds, name="ppo_env_rng_seeds")
    rollout_rng_seeds = _parse_ppo_seed_spec(ppo_rollout_rng_seeds, name="ppo_rollout_rng_seeds")
    build_kwargs = {
        "model": model,
        "env_prior": env_prior,
        "device": device,
        "num_features": num_features,
        "n_envs": int(n_envs),
        "n_steps": int(n_steps),
        "learning_rate": learning_rate,
        "batch_size": int(batch_size),
        "n_epochs": int(n_epochs),
        "gamma": float(gamma),
        "gae_lambda": float(gae_lambda),
        "clip_range": clip_range,
        "clip_range_vf": clip_range_vf,
        "normalize_advantage": bool(normalize_advantage),
        "space_contract": str(space_contract),
        "value_target_space": value_target_space,
        "actor_gae_space": actor_gae_space,
        "allow_mixed_space_contract": bool(allow_mixed_space_contract),
        "actor_baseline_mode": str(actor_baseline_mode),
        "separate_value_backbone": bool(separate_value_backbone),
        "reset_env_state_at_sep": bool(reset_env_state_at_sep),
        "value_head_impl": str(value_head_impl),
        "value_path_adapter_impl": str(value_path_adapter_impl),
        "value_head_mlp_hidden_dim": int(value_head_mlp_hidden_dim),
        "strict_fixed_env_mode": bool(ppo_strict_fixed_env_mode),
        "env_rng_seeds": env_rng_seeds,
        "rollout_rng_seeds": rollout_rng_seeds,
        "deterministic_actor_sampling": bool(ppo_deterministic_actor_sampling),
        "deterministic_batch_plan": bool(ppo_deterministic_batch_plan),
        "strict_native_rollout": bool(ppo_strict_native_rollout),
        "restore_validation_policy_state": bool(restore_validation_policy_state),
        "runtime_normalized_q_value_weight_override": runtime_normalized_q_value_weight_override,
        "runtime_next_state_flow_matching_weight_override": runtime_next_state_flow_matching_weight_override,
        "ent_coef": float(ent_coef),
        "vf_coef": float(vf_coef),
        "max_grad_norm": float(max_grad_norm),
        "target_kl": target_kl,
        "verbose": int(verbose),
    }
    runtime_summary = {
        "checkpoint_saved_validation_policy_state_present": bool(has_saved_validation_policy_state),
        "restore_validation_policy_state": bool(restore_validation_policy_state),
        "strict_fixed_env_mode": bool(build_kwargs["strict_fixed_env_mode"]),
        "env_rng_seeds": env_rng_seeds,
        "rollout_rng_seeds": rollout_rng_seeds,
        "deterministic_actor_sampling": bool(build_kwargs["deterministic_actor_sampling"]),
        "deterministic_batch_plan": bool(build_kwargs["deterministic_batch_plan"]),
        "strict_native_rollout": bool(build_kwargs["strict_native_rollout"]),
        "value_head_impl": str(build_kwargs["value_head_impl"]),
        "value_path_adapter_impl": str(build_kwargs["value_path_adapter_impl"]),
        "value_head_mlp_hidden_dim": int(build_kwargs["value_head_mlp_hidden_dim"]),
    }
    return build_kwargs, runtime_summary


def _clear_policy_rollout_artifacts(env_prior=None, policy_step_fn=None):
    if env_prior is not None:
        clear_fn = getattr(env_prior, "clear_rollout_artifacts", None)
        if callable(clear_fn):
            clear_fn()
    if policy_step_fn is not None:
        clear_fn = getattr(policy_step_fn, "_clear_buffers", None)
        if callable(clear_fn):
            clear_fn()


def _append_text_log_line(path, line):
    if path is None:
        return
    path_str = str(path).strip()
    if not path_str:
        return
    try:
        log_dir = os.path.dirname(os.path.abspath(path_str))
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(path_str, "a", encoding="utf-8") as f:
            f.write(str(line).rstrip("\n") + "\n")
    except Exception:
        # Logging must never break training.
        pass


def _compute_gradient_observability(model, device, zero_tol=1e-12):
    """
    Compute robust gradient distribution stats from current parameter grads.
    Non-finite values are tracked explicitly and excluded from moment stats.
    """
    device_obj = device if isinstance(device, torch.device) else torch.device(str(device))
    with torch.no_grad():
        grad_numel = torch.zeros((), device=device_obj, dtype=torch.int64)
        grad_nonfinite = torch.zeros((), device=device_obj, dtype=torch.int64)
        grad_zero = torch.zeros((), device=device_obj, dtype=torch.int64)
        grad_abs_sum = torch.zeros((), device=device_obj, dtype=torch.float64)
        grad_sq_sum = torch.zeros((), device=device_obj, dtype=torch.float64)
        grad_abs_max = torch.zeros((), device=device_obj, dtype=torch.float32)
        has_any_grad = False

        for p in model.parameters():
            g = p.grad
            if g is None:
                continue
            has_any_grad = True
            g_det = g.detach()
            if g_det.is_sparse:
                g_det = g_det.coalesce().values()
            g_f = g_det.to(dtype=torch.float32)
            finite_mask = torch.isfinite(g_f)
            grad_numel = grad_numel + torch.as_tensor(g_f.numel(), device=device_obj, dtype=torch.int64)
            grad_nonfinite = grad_nonfinite + (~finite_mask).to(torch.int64).sum()
            g_clean = torch.where(finite_mask, g_f, torch.zeros_like(g_f))
            abs_clean = g_clean.abs()
            grad_abs_sum = grad_abs_sum + abs_clean.to(dtype=torch.float64).sum()
            grad_sq_sum = grad_sq_sum + (g_clean.to(dtype=torch.float64) * g_clean.to(dtype=torch.float64)).sum()
            grad_abs_max = torch.maximum(grad_abs_max, abs_clean.max())
            if zero_tol > 0.0:
                grad_zero = grad_zero + ((abs_clean <= float(zero_tol)) & finite_mask).to(torch.int64).sum()
            else:
                grad_zero = grad_zero + ((abs_clean == 0.0) & finite_mask).to(torch.int64).sum()

        if not has_any_grad:
            return None

        finite_count = torch.clamp(grad_numel - grad_nonfinite, min=0)
        finite_count_f = torch.clamp(finite_count.to(dtype=torch.float64), min=1.0)
        grad_numel_f = torch.clamp(grad_numel.to(dtype=torch.float64), min=1.0)

        grad_abs_mean = grad_abs_sum / finite_count_f
        grad_rms = torch.sqrt(torch.clamp(grad_sq_sum / finite_count_f, min=0.0))
        grad_l2 = torch.sqrt(torch.clamp(grad_sq_sum, min=0.0))
        grad_nonfinite_share = grad_nonfinite.to(dtype=torch.float64) / grad_numel_f
        grad_zero_share = grad_zero.to(dtype=torch.float64) / finite_count_f

        return {
            "grad_numel": int(grad_numel.detach().item()),
            "grad_nonfinite": int(grad_nonfinite.detach().item()),
            "grad_nonfinite_share": float(grad_nonfinite_share.detach().item()),
            "grad_zero_share": float(grad_zero_share.detach().item()),
            "grad_abs_mean": float(grad_abs_mean.detach().item()),
            "grad_abs_max": float(grad_abs_max.detach().item()),
            "grad_rms": float(grad_rms.detach().item()),
            "grad_l2": float(grad_l2.detach().item()),
        }



def _summarize_nonfinite_gradient_tensors(model, topk=8):
    summaries = []
    prefix_agg = {}
    with torch.no_grad():
        for name, p in model.named_parameters():
            g = p.grad
            if g is None:
                continue
            g_det = g.detach()
            if g_det.is_sparse:
                g_det = g_det.coalesce().values()
            g_f = g_det.to(dtype=torch.float32)
            finite_mask = torch.isfinite(g_f)
            nonfinite_count = int((~finite_mask).to(torch.int64).sum().item())
            if nonfinite_count <= 0:
                continue
            numel = int(g_f.numel())
            finite_vals = torch.where(finite_mask, g_f.abs(), torch.zeros_like(g_f))
            finite_abs_max = float(finite_vals.max().item()) if numel > 0 else 0.0
            finite_abs_mean = float(finite_vals.sum().item() / max(int(finite_mask.to(torch.int64).sum().item()), 1))
            share = float(nonfinite_count / max(numel, 1))
            item = {
                'name': name,
                'shape': tuple(p.shape),
                'nonfinite_count': nonfinite_count,
                'numel': numel,
                'share': share,
                'finite_abs_max': finite_abs_max,
                'finite_abs_mean': finite_abs_mean,
            }
            summaries.append(item)
            prefix = name.rsplit('.', 1)[0] if '.' in name else name
            agg = prefix_agg.setdefault(prefix, {'nonfinite_count': 0, 'numel': 0})
            agg['nonfinite_count'] += nonfinite_count
            agg['numel'] += numel
    summaries.sort(key=lambda x: (x['nonfinite_count'], x['share'], x['finite_abs_max']), reverse=True)
    prefix_summaries = []
    for prefix, agg in prefix_agg.items():
        numel = int(agg['numel'])
        nonfinite_count = int(agg['nonfinite_count'])
        prefix_summaries.append({
            'prefix': prefix,
            'nonfinite_count': nonfinite_count,
            'numel': numel,
            'share': float(nonfinite_count / max(numel, 1)),
        })
    prefix_summaries.sort(key=lambda x: (x['nonfinite_count'], x['share']), reverse=True)
    return {'params': summaries[:max(int(topk), 0)], 'prefixes': prefix_summaries[:max(int(topk), 0)]}


def _emit_nonfinite_gradient_diagnostics(model, *, epoch, batch, topk=8):
    diag = _summarize_nonfinite_gradient_tensors(model, topk=topk)
    for rank, item in enumerate(diag.get('params', []), start=1):
        shape = 'x'.join(str(int(v)) for v in item.get('shape', ()))
        print(
            f"[grad-nonfinite-top] epoch={epoch} batch={batch} rank={rank} "
            f"name={item['name']} shape={shape} nonfinite={item['nonfinite_count']} "
            f"numel={item['numel']} share={item['share']:.3e} "
            f"finite_abs_max={item['finite_abs_max']:.3e} "
            f"finite_abs_mean={item['finite_abs_mean']:.3e}"
        )
    for rank, item in enumerate(diag.get('prefixes', []), start=1):
        print(
            f"[grad-nonfinite-prefix] epoch={epoch} batch={batch} rank={rank} "
            f"prefix={item['prefix']} nonfinite={item['nonfinite_count']} "
            f"numel={item['numel']} share={item['share']:.3e}"
        )

def _scale_model_grads_(model, scale):
    scale_f = float(scale)
    if (not math.isfinite(scale_f)) or abs(scale_f - 1.0) <= 1e-9:
        return
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is None:
                continue
            p.grad.mul_(scale_f)


def _stats_tensor_or_default(pg_stats, key, *, device, default):
    value = pg_stats.get(key, None)
    if value is None:
        return torch.as_tensor(default, device=device, dtype=torch.float32)
    if torch.is_tensor(value):
        return value.detach()
    return torch.as_tensor(value, device=device, dtype=torch.float32)

def _resolve_environment_prior(prior):
    seen = set()
    cur = prior
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, EnvironmentPrior) or hasattr(cur, "rollout_policy_gradient_loss"):
            return cur
        cur = getattr(cur, "base_prior", None)
    return None


def _resolve_env_prior_pg_one_hop_replay_enabled(prior) -> bool:
    env_prior = _resolve_environment_prior(prior)
    if env_prior is None:
        return False
    cfg = getattr(env_prior, "config", None)
    resolve_fn = getattr(env_prior, "_resolve_pg_one_hop_replay_enabled", None)
    if callable(resolve_fn):
        try:
            return bool(resolve_fn(cfg if isinstance(cfg, dict) else {}))
        except Exception:
            pass
    if not isinstance(cfg, dict):
        return False
    for key in ("pg_one_hop_replay_enabled", "alpha_grad_one_hop_replay_enabled"):
        if key in cfg:
            try:
                return bool(cfg.get(key))
            except Exception:
                return False
    return False


def _saved_tensors_cpu_offload_context(enabled: bool, pin_memory: bool):
    if not bool(enabled):
        return nullcontext()
    graph_mod = getattr(torch.autograd, "graph", None)
    if graph_mod is None or not hasattr(graph_mod, "save_on_cpu"):
        return nullcontext()
    return graph_mod.save_on_cpu(pin_memory=bool(pin_memory))


def _wrap_policy_step_fn_saved_tensors_offload(policy_step_fn, *, enabled: bool, pin_memory: bool):
    if (not bool(enabled)) or policy_step_fn is None:
        return policy_step_fn
    offload_state = {"enabled": bool(enabled)}

    @wraps(policy_step_fn)
    def _wrapped(*args, **kwargs):
        if not bool(offload_state["enabled"]):
            return policy_step_fn(*args, **kwargs)
        with _saved_tensors_cpu_offload_context(enabled=True, pin_memory=bool(pin_memory)):
            return policy_step_fn(*args, **kwargs)

    try:
        _wrapped.__dict__.update(getattr(policy_step_fn, "__dict__", {}))
    except Exception:
        pass
    _wrapped._ticl_saved_tensors_cpu_offload_default_enabled = bool(enabled)

    def _set_saved_tensors_cpu_offload_enabled(flag):
        offload_state["enabled"] = bool(flag)

    def _get_saved_tensors_cpu_offload_enabled():
        return bool(offload_state["enabled"])

    _wrapped._ticl_set_saved_tensors_cpu_offload_enabled = _set_saved_tensors_cpu_offload_enabled
    _wrapped._ticl_get_saved_tensors_cpu_offload_enabled = _get_saved_tensors_cpu_offload_enabled
    return _wrapped


def _resolve_effective_policy_saved_tensors_offload(
    *,
    enabled,
    scope,
    device,
    batch_size,
    n_samples,
    one_hop_replay_enabled=False,
    auto_disable_when_safe=False,
    auto_min_free_gb=8.0,
    auto_max_batch_size=64,
    auto_max_n_samples=1024,
):
    enabled = bool(enabled)
    scope = str(scope or "all").strip().lower()
    if (not enabled) or scope != "policy":
        return enabled
    if not bool(auto_disable_when_safe):
        return enabled
    if bool(one_hop_replay_enabled):
        return enabled
    try:
        batch_size = int(batch_size)
        n_samples = int(n_samples)
        auto_max_batch_size = int(auto_max_batch_size)
        auto_max_n_samples = int(auto_max_n_samples)
    except Exception:
        return enabled
    if batch_size > max(0, auto_max_batch_size) or n_samples > max(0, auto_max_n_samples):
        return enabled
    device_obj = device if isinstance(device, torch.device) else torch.device(str(device))
    if device_obj.type != "cuda" or (not torch.cuda.is_available()):
        return enabled
    try:
        free_bytes, _total_bytes = torch.cuda.mem_get_info(device_obj)
    except Exception:
        return enabled
    try:
        min_free_bytes = int(max(0.0, float(auto_min_free_gb)) * (1024 ** 3))
    except Exception:
        min_free_bytes = 0
    if int(free_bytes) >= int(min_free_bytes):
        return False
    return enabled


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


def _fit_actor_output_dim(actor_outputs, action_dim):
    if not isinstance(actor_outputs, dict):
        return actor_outputs
    fitted = dict(actor_outputs)
    for key in (
        "action_mean",
        "action_std",
        "action_log_std",
        "raw_action_mean",
        "raw_action_std",
    ):
        value = fitted.get(key, None)
        if not torch.is_tensor(value):
            continue
        if value.ndim >= 2 and int(value.shape[0]) == 1:
            value = value.squeeze(0)
        original_shape = tuple(value.shape)
        if not original_shape:
            continue
        fitted[key] = _fit_action_dim(
            value.reshape(-1, int(original_shape[-1])),
            action_dim,
        ).reshape(*original_shape[:-1], int(action_dim))
    return fitted


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


def _snapshot_dynamo_counters():
    try:
        from torch._dynamo.utils import counters as dynamo_counters
    except Exception:
        return None
    snapshot = {}
    try:
        groups_iter = dynamo_counters.items()
    except Exception:
        return None
    for group_name, group_counter in groups_iter:
        try:
            items_iter = group_counter.items()
        except Exception:
            continue
        group_snap = {}
        for key, value in items_iter:
            try:
                group_snap[str(key)] = int(value)
            except Exception:
                continue
        snapshot[str(group_name)] = group_snap
    return snapshot


def _diff_dynamo_counters(current_snapshot, baseline_snapshot):
    if not isinstance(current_snapshot, dict):
        return {}
    baseline_snapshot = baseline_snapshot if isinstance(baseline_snapshot, dict) else {}
    delta = {}
    for group_name in set(current_snapshot.keys()) | set(baseline_snapshot.keys()):
        cur_group = current_snapshot.get(group_name, {})
        base_group = baseline_snapshot.get(group_name, {})
        if not isinstance(cur_group, dict):
            cur_group = {}
        if not isinstance(base_group, dict):
            base_group = {}
        for key_name in set(cur_group.keys()) | set(base_group.keys()):
            cur_v = int(cur_group.get(key_name, 0) or 0)
            base_v = int(base_group.get(key_name, 0) or 0)
            dv = int(cur_v - base_v)
            if dv != 0:
                delta[f"{group_name}.{key_name}"] = dv
    return delta


def _format_compile_counter_delta(counter_delta, max_items=8):
    if not isinstance(counter_delta, dict) or len(counter_delta) == 0:
        return "none"
    items = sorted(
        counter_delta.items(),
        key=lambda kv: (-abs(int(kv[1])), str(kv[0])),
    )
    top_items = items[: int(max(1, max_items))]
    return ",".join(f"{k}:{int(v):+d}" for k, v in top_items)


def _resolve_policy_model_ref(model):
    queue = [model]
    seen = set()
    while queue:
        candidate = queue.pop(0)
        if candidate is None:
            continue
        ident = id(candidate)
        if ident in seen:
            continue
        seen.add(ident)
        if hasattr(candidate, "forward_policy_step"):
            return candidate
        for attr in ("module", "model", "_orig_mod"):
            nested = getattr(candidate, attr, None)
            if nested is not None and id(nested) not in seen:
                queue.append(nested)
    return model


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
    model_ref = _resolve_policy_model_ref(model)
    if not hasattr(model_ref, "forward_policy_step"):
        raise ValueError("RL policy objectives require model.forward_policy_step")
    require_policy_action_head = getattr(model_ref, "require_policy_action_head", None)
    if callable(require_policy_action_head):
        require_policy_action_head()
    policy_action_head_required_fn = getattr(model_ref, "policy_action_head_required", None)
    has_policy_action_head_fn = getattr(model_ref, "has_policy_action_head", None)
    attach_policy_action_head_helpers = True
    if callable(policy_action_head_required_fn):
        attach_policy_action_head_helpers = bool(policy_action_head_required_fn())
    elif callable(has_policy_action_head_fn):
        attach_policy_action_head_helpers = bool(has_policy_action_head_fn())

    num_features = int(num_features)
    model_encoder = getattr(model_ref, "encoder", None)
    model_x_encoder_type = str(getattr(model_ref, "x_encoder_type", "")).strip().lower()
    split_fastpath_available = (
        model_x_encoder_type == "split_obs_action"
        and hasattr(model_ref, "forward_policy_step_split")
        and model_encoder is not None
        and hasattr(model_encoder, "obs_dim")
        and hasattr(model_encoder, "action_dim")
    )
    split_compile_pref = str(os.environ.get("TICL_POLICY_SPLIT_STEP_COMPILE", "auto")).strip().lower()
    if split_compile_pref not in {"auto", "on", "off"}:
        split_compile_pref = "auto"
    split_compile_requested = bool(pg_torch_compile)
    if split_compile_pref == "off":
        split_compile_requested = False
    elif split_compile_pref == "auto":
        # Mutable paged KV cache mutates page lengths during rollout; compiling
        # split step on this path causes heavy recompile storms.
        if bool(allow_grad_mutable_cache) and str(kv_cache_mode).strip().lower() == "paged":
            split_compile_requested = False
    main_compile_requested = bool(pg_torch_compile)
    if split_fastpath_available and (not split_compile_requested) and split_compile_pref == "auto":
        # When split fastpath is active and split compile is auto-disabled, the
        # main policy_step compile is not used in this workload; skip it to
        # avoid compile warmup pollution.
        main_compile_requested = False

    if bool(pg_torch_compile) and split_fastpath_available and (not split_compile_requested):
        reason = (
            "auto-disabled for mutable paged KV cache"
            if split_compile_pref == "auto"
            else "disabled by TICL_POLICY_SPLIT_STEP_COMPILE=off"
        )
        print(f"[pg-compile-note] split forward step compile {reason}.")
        if not main_compile_requested:
            print("[pg-compile-note] main forward step compile skipped (split fastpath active).")

    policy_forward_step = getattr(model_ref, "forward_policy_step_actor", None)
    if not callable(policy_forward_step):
        policy_forward_step = model_ref.forward_policy_step
    policy_forward_step, compile_active = _compile_policy_forward_step(
        policy_forward_step,
        enabled=bool(main_compile_requested),
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
    split_obs_total_dim = None
    split_action_slot_dim = None
    if split_fastpath_available:
        split_policy_forward_step = getattr(model_ref, "forward_policy_step_split_actor", None)
        if not callable(split_policy_forward_step):
            split_policy_forward_step = model_ref.forward_policy_step_split
        split_policy_forward_step, split_compile_active = _compile_policy_forward_step(
            split_policy_forward_step,
            enabled=bool(split_compile_requested),
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
        split_obs_total_dim = int(model_encoder.obs_dim)
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
    token_alloc_opt_flag = str(os.environ.get("TICL_POLICY_STEP_TOKEN_ALLOC_OPT", "1")).strip().lower()
    token_alloc_opt = token_alloc_opt_flag not in {"0", "false", "no", "off"}

    def policy_step_fn(
        obs_t,
        action_t,
        reward_t,
        reward_mask_t,
        cache,
        step_idx,
        env_info,
        *,
        _policy_action_head_params_override=None,
    ):
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
        phase_scalar = None
        if bool(split_fastpath_available) and isinstance(env_info, dict) and ("phase_t" in env_info):
            phase_scalar = env_info["phase_t"].reshape(batch_size).detach().to(dtype=obs_t.dtype)
        terminal_scalar = None
        if isinstance(env_info, dict) and ("terminal_t" in env_info):
            terminal_scalar = env_info["terminal_t"].reshape(batch_size).detach().to(dtype=obs_t.dtype)
        extra_scalar_slots_expected = (
            int(max(0, int(split_obs_total_dim) - int(obs_slot_dim) - 2))
            if split_obs_total_dim is not None
            else 0
        )
        phase_token_enabled = bool(phase_scalar is not None)
        terminal_token_enabled = bool(terminal_scalar is not None)
        remaining_scalar_slots = int(
            max(0, extra_scalar_slots_expected - int(phase_token_enabled) - int(terminal_token_enabled))
        )
        if remaining_scalar_slots > 0 and phase_scalar is None:
            phase_scalar = torch.zeros((batch_size,), device=obs_t.device, dtype=obs_t.dtype)
            phase_token_enabled = True
            remaining_scalar_slots -= 1
        if remaining_scalar_slots > 0 and terminal_scalar is None:
            terminal_scalar = torch.zeros((batch_size,), device=obs_t.device, dtype=obs_t.dtype)
            terminal_token_enabled = True
            remaining_scalar_slots -= 1
        split_obs_slot_dim = (
            int(
                max(
                    0,
                    int(split_obs_total_dim)
                    - (2 + int(phase_token_enabled) + int(terminal_token_enabled)),
                )
            )
            if split_obs_total_dim is not None
            else None
        )

        use_split_fastpath = (
            bool(split_fastpath_available)
            and split_policy_forward_step is not None
            and split_obs_slot_dim is not None
            and split_action_slot_dim is not None
            and obs_slot_dim == int(split_obs_slot_dim)
            and action_slot_dim == int(split_action_slot_dim)
        )

        step_kv_cache_mode = kv_cache_mode
        step_allow_grad_mutable_cache = allow_grad_mutable_cache
        step_allow_grad_inplace_paged_cache = allow_grad_inplace_paged_cache
        if (
            (not torch.is_grad_enabled())
            and bool(allow_grad_mutable_cache)
            and str(kv_cache_mode).strip().lower() == "paged"
        ):
            # Reentrant rollout checkpoint executes the first forward pass under
            # no-grad. The paged+mutable cache path is numerically unstable in
            # that mode after a prior paged backward, while immutable cache is
            # semantically equivalent for the no-grad replay pass.
            step_kv_cache_mode = "immutable"
            step_allow_grad_mutable_cache = False
            step_allow_grad_inplace_paged_cache = False

        def _call_split_forward_step():
            split_extra_kwargs = {}
            if _policy_action_head_params_override is not None:
                split_extra_kwargs["policy_action_head_params_override"] = _policy_action_head_params_override
            return split_policy_forward_step(
                obs_t,
                action_t,
                reward_scalar.reshape(batch_size, 1),
                reward_mask.reshape(batch_size, 1),
                phase_t=(phase_scalar.reshape(batch_size, 1) if phase_token_enabled else None),
                terminal_t=(terminal_scalar.reshape(batch_size, 1) if terminal_token_enabled else None),
                kv_cache=cache,
                max_cache_len=max_cache_len,
                kv_cache_mode=step_kv_cache_mode,
                kv_cache_page_size=kv_cache_page_size,
                allow_grad_mutable_cache=step_allow_grad_mutable_cache,
                allow_grad_inplace_paged_cache=step_allow_grad_inplace_paged_cache,
                **split_extra_kwargs,
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
                    split_policy_forward_step = (
                        getattr(model_ref, "forward_policy_step_split_actor", None)
                        or model_ref.forward_policy_step_split
                    )
                    out, kv_cache = _call_split_forward_step()
            else:
                out, kv_cache = _call_split_forward_step()
            if isinstance(out, dict):
                return _fit_actor_output_dim(out, action_dim), kv_cache
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
            if token_alloc_opt:
                # Grad-enabled path: allocate once per step and clear exactly
                # once below; avoid zeros+zero_ double initialization.
                x_token = torch.empty(
                    (1, batch_size, num_features),
                    device=obs_t.device,
                    dtype=obs_t.dtype,
                )
                # reward_scalar is detached; view is sufficient and avoids clone.
                y_token = reward_scalar.reshape(1, batch_size)
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
        phase_idx = obs_slot_dim + 2
        if phase_token_enabled and phase_idx < num_features:
            x_row[:, phase_idx] = phase_scalar
        terminal_idx = obs_slot_dim + 2 + int(phase_token_enabled)
        if terminal_token_enabled and terminal_idx < num_features:
            x_row[:, terminal_idx] = terminal_scalar
        action_write_start = obs_slot_dim + 2 + int(phase_token_enabled) + int(terminal_token_enabled)
        if action_write_start < num_features:
            action_copy = int(min(action_t.shape[-1], action_slot_dim, num_features - action_write_start))
            if action_copy > 0:
                x_row[:, action_write_start: action_write_start + action_copy] = action_t[:, :action_copy]
        if reuse_token_buf:
            y_token[0].copy_(reward_scalar)

        def _call_forward_step():
            forward_extra_kwargs = {}
            if _policy_action_head_params_override is not None:
                forward_extra_kwargs["policy_action_head_params_override"] = _policy_action_head_params_override
            return policy_forward_step(
                x_token,
                y_token,
                kv_cache=cache,
                max_cache_len=max_cache_len,
                kv_cache_mode=step_kv_cache_mode,
                kv_cache_page_size=kv_cache_page_size,
                allow_grad_mutable_cache=step_allow_grad_mutable_cache,
                allow_grad_inplace_paged_cache=step_allow_grad_inplace_paged_cache,
                **forward_extra_kwargs,
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
                policy_forward_step = (
                    getattr(model_ref, "forward_policy_step_actor", None)
                    or model_ref.forward_policy_step
                )
                out, kv_cache = _call_forward_step()
        else:
            out, kv_cache = _call_forward_step()
        if isinstance(out, dict):
            return _fit_actor_output_dim(out, action_dim), kv_cache
        action_raw = out.squeeze(0)
        action_next = _fit_action_dim(action_raw, action_dim)
        return action_next, kv_cache

    def _clear_policy_step_buffers():
        token_buf["x"] = None
        token_buf["y"] = None
        token_buf["batch_size"] = None
        token_buf["dtype"] = None
        token_buf["device"] = None

    policy_step_fn._compile_active = bool(compile_active or split_compile_active)
    policy_step_fn._clear_buffers = _clear_policy_step_buffers
    policy_step_fn._model_ref = model_ref
    reinforce_sequence_replay_fn = getattr(model_ref, "replay_policy_sequence_tokens", None)
    policy_step_fn._reinforce_sequence_replay_fn = (
        reinforce_sequence_replay_fn if callable(reinforce_sequence_replay_fn) else None
    )
    reinforce_sequence_replay_outputs_fn = getattr(model_ref, "replay_policy_actor_outputs", None)
    if not callable(reinforce_sequence_replay_outputs_fn):
        reinforce_sequence_replay_outputs_fn = getattr(model_ref, "replay_policy_sequence_outputs", None)
    policy_step_fn._reinforce_sequence_replay_outputs_fn = (
        reinforce_sequence_replay_outputs_fn if callable(reinforce_sequence_replay_outputs_fn) else None
    )
    policy_step_fn._fit_action_dim_fn = _fit_action_dim
    sample_action_fn = getattr(model_ref, "sample_policy_action_from_outputs", None)
    log_prob_action_fn = getattr(model_ref, "log_prob_policy_action_from_outputs", None)
    score_action_fn = getattr(model_ref, "log_prob_score_wrt_mean_from_outputs", None)
    decomp_action_fn = getattr(model_ref, "policy_log_prob_decomposition_stats_from_outputs", None)
    policy_step_fn._policy_actor_sample_fn = (
        sample_action_fn if attach_policy_action_head_helpers and callable(sample_action_fn) else None
    )
    policy_step_fn._policy_actor_log_prob_fn = (
        log_prob_action_fn if attach_policy_action_head_helpers and callable(log_prob_action_fn) else None
    )
    policy_step_fn._policy_actor_score_fn = (
        score_action_fn if attach_policy_action_head_helpers and callable(score_action_fn) else None
    )
    policy_step_fn._policy_actor_decomp_stats_fn = (
        decomp_action_fn if attach_policy_action_head_helpers and callable(decomp_action_fn) else None
    )
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


def _freeze_env_h_list_for_replay(env_prior, h_list):
    if h_list is None:
        return None
    ratio_sampler = getattr(env_prior, "_sample_reward_dropout_ratio", None)
    constrained_dim_enabled = getattr(env_prior, "_resolve_constrained_dim_sampling_enabled", None)
    latent_uniform_sampler = getattr(env_prior, "_latent_uniform_from_h", None)
    exact_reward_term_sampler = getattr(env_prior, "_sample_exact_scm_reward_term_enabled", None)
    frozen = []
    for h in h_list:
        if isinstance(h, dict):
            h_frozen = dict(h)
        else:
            h_frozen = h
        if isinstance(h_frozen, dict):
            reward_dropout_enabled = bool(h_frozen.get("reward_dropout_enabled", True))
            reward_dropout_randomize = bool(h_frozen.get("reward_dropout_randomize", True))
            if reward_dropout_enabled and reward_dropout_randomize:
                if callable(ratio_sampler):
                    ratio = float(ratio_sampler(h_frozen))
                else:
                    ratio = float(h_frozen.get("reward_dropout_ratio", 0.0))
                h_frozen["reward_dropout_randomize"] = False
                h_frozen["reward_dropout_ratio"] = float(max(0.0, min(1.0, ratio)))
            if callable(constrained_dim_enabled) and bool(constrained_dim_enabled(h_frozen)):
                if callable(latent_uniform_sampler):
                    latent_uniform_sampler(h_frozen, "_constrained_obs_u")
                    state_dim = int(max(1, min(400, int(h_frozen.get("state_dim", 1)))))
                    total_budget = int(max(1, min(400, int(h_frozen.get("constrained_dim_sampling_total_budget", 400)))))
                    if total_budget > state_dim:
                        latent_uniform_sampler(h_frozen, "_constrained_noise_u")
            if callable(exact_reward_term_sampler):
                exact_reward_term_sampler(
                    h_frozen,
                    prob_key="ctrl_reward_enable_prob",
                    latent_key="_ctrl_reward_enable_u",
                )
                exact_reward_term_sampler(
                    h_frozen,
                    prob_key="survival_reward_enable_prob",
                    latent_key="_survival_reward_enable_u",
                )
        frozen.append(h_frozen)
    return frozen


def _derive_secondary_rollout_seed(seed: int) -> int:
    seed_i = int(seed) & 0x7FFFFFFF
    mixed = (seed_i * 1103515245 + 12345) & 0x7FFFFFFF
    if mixed == seed_i:
        mixed = (mixed + 1) & 0x7FFFFFFF
    return int(mixed)


def _resolve_anil_policy_action_head(policy_step_fn):
    model_ref = getattr(policy_step_fn, "_model_ref", None)
    head = getattr(model_ref, "policy_action_head", None)
    require_head = getattr(model_ref, "require_policy_action_head", None)
    if callable(require_head):
        require_head()
    if not isinstance(head, nn.Module):
        raise RuntimeError("ANIL requires a model with a concrete policy_action_head module.")
    named_params = tuple(head.named_parameters())
    if len(named_params) <= 0:
        raise RuntimeError("ANIL requires policy_action_head to expose trainable parameters.")
    return model_ref, head, OrderedDict((name, param) for name, param in named_params)


def _wrap_policy_step_fn_with_head_override(policy_step_fn, head_params_override):
    if head_params_override is None:
        return policy_step_fn
    model_ref = getattr(policy_step_fn, "_model_ref", None)

    def _wrapped(*args, **kwargs):
        kwargs["_policy_action_head_params_override"] = head_params_override
        return policy_step_fn(*args, **kwargs)

    _wrapped.__dict__.update(getattr(policy_step_fn, "__dict__", {}))
    _wrapped._model_ref = model_ref
    replay_fn = getattr(model_ref, "replay_policy_sequence_tokens", None)
    if callable(replay_fn):
        def _replay_with_override(x_tokens, y_tokens, *, eval_start=0):
            return replay_fn(
                x_tokens,
                y_tokens,
                eval_start=eval_start,
                policy_action_head_params_override=head_params_override,
            )

        _wrapped._reinforce_sequence_replay_fn = _replay_with_override
    replay_outputs_fn = getattr(model_ref, "replay_policy_actor_outputs", None)
    if not callable(replay_outputs_fn):
        replay_outputs_fn = getattr(model_ref, "replay_policy_sequence_outputs", None)
    if callable(replay_outputs_fn):
        def _replay_outputs_with_override(
            x_tokens,
            y_tokens,
            *,
            eval_start=0,
            action_query=None,
            flow_matching_xt=None,
            flow_matching_t=None,
        ):
            kwargs = {
                "eval_start": eval_start,
                "policy_action_head_params_override": head_params_override,
            }
            sig = inspect.signature(replay_outputs_fn)
            if "action_query" in sig.parameters:
                kwargs["action_query"] = action_query
            if "flow_matching_xt" in sig.parameters:
                kwargs["flow_matching_xt"] = flow_matching_xt
            if "flow_matching_t" in sig.parameters:
                kwargs["flow_matching_t"] = flow_matching_t
            return replay_outputs_fn(
                x_tokens,
                y_tokens,
                **kwargs,
            )

        _wrapped._reinforce_sequence_replay_outputs_fn = _replay_outputs_with_override
    encode_train_tokens_fn = getattr(model_ref, "encode_train_sequence_tokens", None)
    if callable(encode_train_tokens_fn):
        _wrapped._reinforce_sequence_encode_train_tokens_fn = encode_train_tokens_fn
    replay_outputs_from_train_tokens_fn = getattr(
        model_ref,
        "replay_policy_actor_outputs_from_train_tokens",
        None,
    )
    if not callable(replay_outputs_from_train_tokens_fn):
        replay_outputs_from_train_tokens_fn = getattr(
            model_ref,
            "replay_policy_sequence_outputs_from_train_tokens",
            None,
        )
    if callable(replay_outputs_from_train_tokens_fn):
        def _replay_outputs_from_train_tokens_with_override(
            train_tokens,
            *,
            eval_start=0,
            action_query=None,
            flow_matching_xt=None,
            flow_matching_t=None,
        ):
            kwargs = {
                "eval_start": eval_start,
                "policy_action_head_params_override": head_params_override,
            }
            sig = inspect.signature(replay_outputs_from_train_tokens_fn)
            if "action_query" in sig.parameters:
                kwargs["action_query"] = action_query
            if "flow_matching_xt" in sig.parameters:
                kwargs["flow_matching_xt"] = flow_matching_xt
            if "flow_matching_t" in sig.parameters:
                kwargs["flow_matching_t"] = flow_matching_t
            return replay_outputs_from_train_tokens_fn(
                train_tokens,
                **kwargs,
            )

        _wrapped._reinforce_sequence_replay_outputs_from_train_tokens_fn = (
            _replay_outputs_from_train_tokens_with_override
        )
    init_kv_cache_fn = getattr(model_ref, "init_kv_cache", None)
    if callable(init_kv_cache_fn):
        _wrapped._reinforce_sequence_init_kv_cache_fn = init_kv_cache_fn
    append_train_token_to_kv_fn = getattr(model_ref, "append_train_token_to_kv", None)
    if callable(append_train_token_to_kv_fn):
        _wrapped._reinforce_sequence_append_train_token_to_kv_fn = append_train_token_to_kv_fn
    predict_query_replay_outputs_with_kv_fn = getattr(model_ref, "predict_query_replay_outputs_with_kv", None)
    if callable(predict_query_replay_outputs_with_kv_fn):
        def _predict_query_replay_outputs_with_override(
            x_query,
            kv_cache,
            *,
            action_query=None,
            flow_matching_xt=None,
            flow_matching_t=None,
        ):
            return predict_query_replay_outputs_with_kv_fn(
                x_query,
                kv_cache,
                action_query=action_query,
                flow_matching_xt=flow_matching_xt,
                flow_matching_t=flow_matching_t,
                policy_action_head_params_override=head_params_override,
            )

        _wrapped._reinforce_sequence_predict_query_outputs_with_kv_fn = (
            _predict_query_replay_outputs_with_override
        )
    stream_replay_aux_outputs_with_kv_fn = getattr(model_ref, "stream_replay_aux_outputs_with_kv", None)
    if callable(stream_replay_aux_outputs_with_kv_fn):
        def _stream_replay_aux_outputs_with_kv_with_override(
            x_tokens,
            y_tokens,
            *,
            full_length,
            q_query_tokens=None,
            q_query_start=0,
            q_query_stop=None,
            q_action_query=None,
            flow_query_tokens=None,
            flow_query_start=0,
            flow_query_stop=None,
            flow_action_query=None,
            flow_matching_xt=None,
            flow_matching_t=None,
        ):
            return stream_replay_aux_outputs_with_kv_fn(
                x_tokens,
                y_tokens,
                full_length=full_length,
                q_query_tokens=q_query_tokens,
                q_query_start=q_query_start,
                q_query_stop=q_query_stop,
                q_action_query=q_action_query,
                flow_query_tokens=flow_query_tokens,
                flow_query_start=flow_query_start,
                flow_query_stop=flow_query_stop,
                flow_action_query=flow_action_query,
                flow_matching_xt=flow_matching_xt,
                flow_matching_t=flow_matching_t,
                policy_action_head_params_override=head_params_override,
            )

        _wrapped._reinforce_sequence_stream_replay_aux_outputs_with_kv_fn = (
            _stream_replay_aux_outputs_with_kv_with_override
        )
    replay_aux_strict_sequence_outputs_fn = getattr(model_ref, "replay_aux_strict_sequence_outputs", None)
    if callable(replay_aux_strict_sequence_outputs_fn):
        def _replay_aux_strict_sequence_outputs_with_override(
            x_tokens,
            y_tokens,
            *,
            full_length,
            q_query_tokens=None,
            q_query_start=0,
            q_query_stop=None,
            q_action_query=None,
            flow_query_tokens=None,
            flow_query_start=0,
            flow_query_stop=None,
            flow_action_query=None,
            flow_matching_xt=None,
            flow_matching_t=None,
            output_chunk_sink=None,
        ):
            return replay_aux_strict_sequence_outputs_fn(
                x_tokens,
                y_tokens,
                full_length=full_length,
                q_query_tokens=q_query_tokens,
                q_query_start=q_query_start,
                q_query_stop=q_query_stop,
                q_action_query=q_action_query,
                flow_query_tokens=flow_query_tokens,
                flow_query_start=flow_query_start,
                flow_query_stop=flow_query_stop,
                flow_action_query=flow_action_query,
                flow_matching_xt=flow_matching_xt,
                flow_matching_t=flow_matching_t,
                output_chunk_sink=output_chunk_sink,
            )

        _wrapped._reinforce_sequence_replay_aux_strict_sequence_outputs_fn = (
            _replay_aux_strict_sequence_outputs_with_override
        )
    replay_aux_sequence_outputs_fn = getattr(model_ref, "replay_aux_sequence_outputs", None)
    if callable(replay_aux_sequence_outputs_fn):
        def _replay_aux_sequence_outputs_with_override(
            x_tokens,
            y_tokens,
            *,
            full_length,
            q_query_tokens=None,
            q_query_start=0,
            q_query_stop=None,
            q_action_query=None,
            flow_query_tokens=None,
            flow_query_start=0,
            flow_query_stop=None,
            flow_action_query=None,
            flow_matching_xt=None,
            flow_matching_t=None,
        ):
            return replay_aux_sequence_outputs_fn(
                x_tokens,
                y_tokens,
                full_length=full_length,
                q_query_tokens=q_query_tokens,
                q_query_start=q_query_start,
                q_query_stop=q_query_stop,
                q_action_query=q_action_query,
                flow_query_tokens=flow_query_tokens,
                flow_query_start=flow_query_start,
                flow_query_stop=flow_query_stop,
                flow_action_query=flow_action_query,
                flow_matching_xt=flow_matching_xt,
                flow_matching_t=flow_matching_t,
            )

        _wrapped._reinforce_sequence_replay_aux_sequence_outputs_fn = (
            _replay_aux_sequence_outputs_with_override
        )
    replay_aux_strict_sequence_outputs_from_train_tokens_fn = getattr(
        model_ref,
        "replay_aux_strict_sequence_outputs_from_train_tokens",
        None,
    )
    if callable(replay_aux_strict_sequence_outputs_from_train_tokens_fn):
        def _replay_aux_strict_sequence_outputs_from_train_tokens_with_override(
            train_tokens,
            *,
            full_length,
            q_query_tokens=None,
            q_query_start=0,
            q_query_stop=None,
            q_action_query=None,
            flow_query_tokens=None,
            flow_query_start=0,
            flow_query_stop=None,
            flow_action_query=None,
            flow_matching_xt=None,
            flow_matching_t=None,
            output_chunk_sink=None,
        ):
            return replay_aux_strict_sequence_outputs_from_train_tokens_fn(
                train_tokens,
                full_length=full_length,
                q_query_tokens=q_query_tokens,
                q_query_start=q_query_start,
                q_query_stop=q_query_stop,
                q_action_query=q_action_query,
                flow_query_tokens=flow_query_tokens,
                flow_query_start=flow_query_start,
                flow_query_stop=flow_query_stop,
                flow_action_query=flow_action_query,
                flow_matching_xt=flow_matching_xt,
                flow_matching_t=flow_matching_t,
                output_chunk_sink=output_chunk_sink,
            )

        _wrapped._reinforce_sequence_replay_aux_strict_sequence_outputs_from_train_tokens_fn = (
            _replay_aux_strict_sequence_outputs_from_train_tokens_with_override
        )
    replay_aux_sequence_outputs_from_train_tokens_fn = getattr(
        model_ref,
        "replay_aux_sequence_outputs_from_train_tokens",
        None,
    )
    if callable(replay_aux_sequence_outputs_from_train_tokens_fn):
        def _replay_aux_sequence_outputs_from_train_tokens_with_override(
            train_tokens,
            *,
            full_length,
            q_query_tokens=None,
            q_query_start=0,
            q_query_stop=None,
            q_action_query=None,
            flow_query_tokens=None,
            flow_query_start=0,
            flow_query_stop=None,
            flow_action_query=None,
            flow_matching_xt=None,
            flow_matching_t=None,
        ):
            return replay_aux_sequence_outputs_from_train_tokens_fn(
                train_tokens,
                full_length=full_length,
                q_query_tokens=q_query_tokens,
                q_query_start=q_query_start,
                q_query_stop=q_query_stop,
                q_action_query=q_action_query,
                flow_query_tokens=flow_query_tokens,
                flow_query_start=flow_query_start,
                flow_query_stop=flow_query_stop,
                flow_action_query=flow_action_query,
                flow_matching_xt=flow_matching_xt,
                flow_matching_t=flow_matching_t,
            )

        _wrapped._reinforce_sequence_replay_aux_sequence_outputs_from_train_tokens_fn = (
            _replay_aux_sequence_outputs_from_train_tokens_with_override
        )
    replay_policy_and_aux_sequence_outputs_fn = getattr(
        model_ref,
        "replay_policy_and_aux_sequence_outputs",
        None,
    )
    if callable(replay_policy_and_aux_sequence_outputs_fn):
        def _replay_policy_and_aux_sequence_outputs_with_override(
            x_tokens,
            y_tokens,
            *,
            full_length,
            eval_start=0,
            q_query_tokens=None,
            q_query_start=0,
            q_query_stop=None,
            q_action_query=None,
            flow_query_tokens=None,
            flow_query_start=0,
            flow_query_stop=None,
            flow_action_query=None,
            flow_matching_xt=None,
            flow_matching_t=None,
        ):
            return replay_policy_and_aux_sequence_outputs_fn(
                x_tokens,
                y_tokens,
                full_length=full_length,
                eval_start=eval_start,
                q_query_tokens=q_query_tokens,
                q_query_start=q_query_start,
                q_query_stop=q_query_stop,
                q_action_query=q_action_query,
                flow_query_tokens=flow_query_tokens,
                flow_query_start=flow_query_start,
                flow_query_stop=flow_query_stop,
                flow_action_query=flow_action_query,
                flow_matching_xt=flow_matching_xt,
                flow_matching_t=flow_matching_t,
                policy_action_head_params_override=head_params_override,
            )

        _wrapped._reinforce_sequence_replay_policy_and_aux_sequence_outputs_fn = (
            _replay_policy_and_aux_sequence_outputs_with_override
        )
    replay_policy_and_aux_sequence_outputs_from_train_tokens_fn = getattr(
        model_ref,
        "replay_policy_and_aux_sequence_outputs_from_train_tokens",
        None,
    )
    if callable(replay_policy_and_aux_sequence_outputs_from_train_tokens_fn):
        def _replay_policy_and_aux_sequence_outputs_from_train_tokens_with_override(
            train_tokens,
            *,
            full_length,
            eval_start=0,
            q_query_tokens=None,
            q_query_start=0,
            q_query_stop=None,
            q_action_query=None,
            flow_query_tokens=None,
            flow_query_start=0,
            flow_query_stop=None,
            flow_action_query=None,
            flow_matching_xt=None,
            flow_matching_t=None,
        ):
            return replay_policy_and_aux_sequence_outputs_from_train_tokens_fn(
                train_tokens,
                full_length=full_length,
                eval_start=eval_start,
                q_query_tokens=q_query_tokens,
                q_query_start=q_query_start,
                q_query_stop=q_query_stop,
                q_action_query=q_action_query,
                flow_query_tokens=flow_query_tokens,
                flow_query_start=flow_query_start,
                flow_query_stop=flow_query_stop,
                flow_action_query=flow_action_query,
                flow_matching_xt=flow_matching_xt,
                flow_matching_t=flow_matching_t,
                policy_action_head_params_override=head_params_override,
            )

        _wrapped._reinforce_sequence_replay_policy_and_aux_sequence_outputs_from_train_tokens_fn = (
            _replay_policy_and_aux_sequence_outputs_from_train_tokens_with_override
        )
    stream_replay_policy_and_aux_outputs_with_kv_fn = getattr(
        model_ref,
        "stream_replay_policy_and_aux_outputs_with_kv",
        None,
    )
    if callable(stream_replay_policy_and_aux_outputs_with_kv_fn):
        def _stream_replay_policy_and_aux_outputs_with_kv_with_override(
            x_tokens,
            y_tokens,
            *,
            full_length,
            eval_start=0,
            q_query_tokens=None,
            q_query_start=0,
            q_query_stop=None,
            q_action_query=None,
            flow_query_tokens=None,
            flow_query_start=0,
            flow_query_stop=None,
            flow_action_query=None,
            flow_matching_xt=None,
            flow_matching_t=None,
        ):
            return stream_replay_policy_and_aux_outputs_with_kv_fn(
                x_tokens,
                y_tokens,
                full_length=full_length,
                eval_start=eval_start,
                q_query_tokens=q_query_tokens,
                q_query_start=q_query_start,
                q_query_stop=q_query_stop,
                q_action_query=q_action_query,
                flow_query_tokens=flow_query_tokens,
                flow_query_start=flow_query_start,
                flow_query_stop=flow_query_stop,
                flow_action_query=flow_action_query,
                flow_matching_xt=flow_matching_xt,
                flow_matching_t=flow_matching_t,
                policy_action_head_params_override=head_params_override,
            )

        _wrapped._reinforce_sequence_stream_policy_and_aux_outputs_with_kv_fn = (
            _stream_replay_policy_and_aux_outputs_with_kv_with_override
        )
    return _wrapped


def _collect_anil_replay_rollout(
    env_prior,
    policy_step_fn,
    *,
    h,
    env_seed,
    rollout_seed,
    n_samples,
    num_features,
    device,
    single_eval_pos,
):
    rollout = env_prior.rollout_with_policy(
        policy_step_fn=policy_step_fn,
        batch_size=1,
        n_samples=int(n_samples),
        num_features=int(num_features),
        device=device,
        single_eval_pos=int(single_eval_pos),
        collect_x=True,
        collect_runtime_info=False,
        store_rewards=True,
        policy_objective_kind="reinforce",
        h_list_override=[h],
        env_seeds_override=[int(env_seed)],
        rollout_seeds_override=[int(rollout_seed)],
        _policy_collect_reinforce_replay=True,
        _policy_disable_log_probs=True,
        _policy_force_no_grad=True,
        _policy_defer_reinforce_replay=True,
    )
    replay_payload = getattr(env_prior, "last_rollout_reinforce_replay", None)
    if not isinstance(replay_payload, dict):
        raise RuntimeError("ANIL rollout did not record reinforce replay payload.")
    return rollout, dict(replay_payload)


def _adapt_anil_policy_action_head(
    env_prior,
    policy_step_fn,
    *,
    support_rollout,
    support_replay_payload,
    anil_inner_steps,
    anil_inner_learning_rate,
):
    _, _, head_params = _resolve_anil_policy_action_head(policy_step_fn)
    fit_action_dim_fn = getattr(policy_step_fn, "_fit_action_dim_fn", None)
    support_single_eval_pos = int(support_rollout.get("single_eval_pos", 0))
    adapted_head_params = head_params
    support_stats_last = None
    for _ in range(int(anil_inner_steps)):
        support_step_fn = _wrap_policy_step_fn_with_head_override(policy_step_fn, adapted_head_params)
        support_loss, support_stats_last, _ = env_prior.reinforce_sequence_replay_loss_from_rollout(
            policy_step_fn=support_step_fn,
            x_tokens=support_rollout["x"],
            rewards=support_rollout["rewards"],
            terminal_counts=support_rollout.get("terminal_eval_counts", None),
            replay_payload=support_replay_payload,
            single_eval_pos=support_single_eval_pos,
            baseline_mode="leave_one_out",
            fit_action_dim_fn=fit_action_dim_fn,
        )
        support_grads = torch.autograd.grad(
            support_loss,
            tuple(adapted_head_params.values()),
            create_graph=True,
            allow_unused=False,
        )
        adapted_head_params = OrderedDict(
            (name, param - (float(anil_inner_learning_rate) * grad))
            for (name, param), grad in zip(adapted_head_params.items(), support_grads)
        )
    if support_stats_last is None:
        raise RuntimeError("ANIL inner adaptation did not produce support statistics.")
    return adapted_head_params, support_stats_last


def _merge_anil_rollout_profile(profile_acc, rollout_profile):
    if not isinstance(rollout_profile, dict):
        return profile_acc
    if profile_acc is None:
        profile_acc = {}
    for key, value in rollout_profile.items():
        if torch.is_tensor(value):
            if value.ndim != 0:
                continue
            value = float(value.detach().cpu().item())
        if isinstance(value, bool):
            value = int(value)
        if isinstance(value, (int, float)):
            profile_acc[key] = profile_acc.get(key, 0.0) + float(value)
        elif isinstance(value, str):
            prev = profile_acc.get(key, None)
            if prev is None or prev == value:
                profile_acc[key] = value
            else:
                profile_acc[key] = "mixed"
    return profile_acc


def _accumulate_anil_scalar_stats(stats_accum, stats, weight):
    if not isinstance(stats, dict):
        return
    for key, value in stats.items():
        if torch.is_tensor(value):
            if value.ndim != 0:
                continue
            value_f = float(value.detach().cpu().item())
        elif isinstance(value, bool):
            value_f = float(int(value))
        elif isinstance(value, (int, float)):
            value_f = float(value)
        else:
            continue
        if key.endswith("_min"):
            prev = stats_accum.get(key, None)
            stats_accum[key] = value_f if prev is None else min(prev, value_f)
        elif key.endswith("_max") or key.endswith("_abs_max"):
            prev = stats_accum.get(key, None)
            stats_accum[key] = value_f if prev is None else max(prev, value_f)
        else:
            stats_accum[key] = stats_accum.get(key, 0.0) + (value_f * float(weight))


def _accumulate_anil_terminal_stats(terminal_accum, terminal_stats, weight):
    if not isinstance(terminal_stats, dict):
        return
    for key, value in terminal_stats.items():
        if torch.is_tensor(value):
            if value.ndim != 0:
                continue
            value_f = float(value.detach().cpu().item())
        elif isinstance(value, (int, float, bool)):
            value_f = float(value)
        else:
            continue
        if key.endswith("_min"):
            prev = terminal_accum.get(key, None)
            terminal_accum[key] = value_f if prev is None else min(prev, value_f)
        elif key.endswith("_max"):
            prev = terminal_accum.get(key, None)
            terminal_accum[key] = value_f if prev is None else max(prev, value_f)
        else:
            terminal_accum[key] = terminal_accum.get(key, 0.0) + (value_f * float(weight))


def _finalize_anil_scalar_stats(stats_accum, *, device):
    stats_out = {}
    int_like_suffixes = (
        "_count",
        "_steps",
        "_chunk_count",
        "_batch_chunk",
        "_applied",
        "_enabled",
        "_task_count",
        "_inner_steps",
    )
    for key, value in stats_accum.items():
        if any(key.endswith(suffix) for suffix in int_like_suffixes):
            stats_out[key] = int(round(float(value)))
        else:
            stats_out[key] = torch.as_tensor(float(value), device=device, dtype=torch.float32)
    return stats_out


def _compute_anil_rollout_chunk_loss(
    env_prior,
    policy_step_fn,
    *,
    batch_size,
    n_samples,
    num_features,
    device,
    single_eval_pos,
    h_list_override=None,
    env_seeds_override=None,
    rollout_seeds_override=None,
    anil_inner_steps=1,
    anil_inner_learning_rate=0.1,
    anil_query_loss_sink=None,
):
    if int(batch_size) <= 0:
        raise ValueError("ANIL batch_size must be positive.")
    model_ref, _, _ = _resolve_anil_policy_action_head(policy_step_fn)
    replay_fn = getattr(model_ref, "replay_policy_sequence_tokens", None)
    if not callable(replay_fn):
        raise RuntimeError("ANIL currently requires a policy model with replay_policy_sequence_tokens support.")
    device_obj = device if isinstance(device, torch.device) else torch.device(str(device))
    if device_obj.type != "cuda":
        raise RuntimeError("ANIL currently requires CUDA because it uses the official RWKV sequence replay path.")
    if int(anil_inner_steps) <= 0:
        raise ValueError("ANIL requires anil_inner_steps >= 1.")
    frozen_h_list = _freeze_env_h_list_for_replay(
        env_prior,
        list(h_list_override) if h_list_override is not None else env_prior._sample_batch_hypers(int(batch_size)),
    )
    env_seeds = (
        [int(s) for s in env_seeds_override]
        if env_seeds_override is not None
        else env_prior._sample_seed_list(int(batch_size))
    )
    support_rollout_seeds = (
        [int(s) for s in rollout_seeds_override]
        if rollout_seeds_override is not None
        else env_prior._sample_seed_list(int(batch_size))
    )
    query_rollout_seeds = [_derive_secondary_rollout_seed(seed) for seed in support_rollout_seeds]

    loss_total = torch.zeros((), device=device_obj, dtype=torch.float32)
    loss_monitor = torch.zeros((), device=device_obj, dtype=torch.float32)
    query_stats_accum = {}
    support_objective_sum = 0.0
    rollout_profile_acc = None
    query_terminal_stats_accum = {}
    task_weight = 1.0 / float(int(batch_size))
    for task_idx in range(int(batch_size)):
        task_h = frozen_h_list[task_idx]
        task_env_seed = int(env_seeds[task_idx])
        support_rollout, support_replay_payload = _collect_anil_replay_rollout(
            env_prior,
            policy_step_fn,
            h=task_h,
            env_seed=task_env_seed,
            rollout_seed=int(support_rollout_seeds[task_idx]),
            n_samples=n_samples,
            num_features=num_features,
            device=device,
            single_eval_pos=single_eval_pos,
        )
        rollout_profile_acc = _merge_anil_rollout_profile(
            rollout_profile_acc,
            support_rollout.get("rollout_profile", None),
        )
        adapted_head_params, support_stats = _adapt_anil_policy_action_head(
            env_prior,
            policy_step_fn,
            support_rollout=support_rollout,
            support_replay_payload=support_replay_payload,
            anil_inner_steps=anil_inner_steps,
            anil_inner_learning_rate=anil_inner_learning_rate,
        )
        support_objective_sum += (
            float(torch.as_tensor(support_stats["objective"]).detach().cpu().item()) * float(task_weight)
        )

        query_step_fn = _wrap_policy_step_fn_with_head_override(policy_step_fn, adapted_head_params)
        query_rollout, query_replay_payload = _collect_anil_replay_rollout(
            env_prior,
            query_step_fn,
            h=task_h,
            env_seed=task_env_seed,
            rollout_seed=int(query_rollout_seeds[task_idx]),
            n_samples=n_samples,
            num_features=num_features,
            device=device,
            single_eval_pos=single_eval_pos,
        )
        rollout_profile_acc = _merge_anil_rollout_profile(
            rollout_profile_acc,
            query_rollout.get("rollout_profile", None),
        )
        query_single_eval_pos = int(query_rollout.get("single_eval_pos", 0))
        query_loss, query_stats, _ = env_prior.reinforce_sequence_replay_loss_from_rollout(
            policy_step_fn=query_step_fn,
            x_tokens=query_rollout["x"],
            rewards=query_rollout["rewards"],
            terminal_counts=query_rollout.get("terminal_eval_counts", None),
            replay_payload=query_replay_payload,
            single_eval_pos=query_single_eval_pos,
            baseline_mode="leave_one_out",
            fit_action_dim_fn=getattr(policy_step_fn, "_fit_action_dim_fn", None),
        )
        _accumulate_anil_scalar_stats(query_stats_accum, query_stats, task_weight)
        _accumulate_anil_terminal_stats(
            query_terminal_stats_accum,
            query_rollout.get("terminal_stats", None),
            task_weight,
        )
        scaled_query_loss = query_loss * float(task_weight)
        loss_monitor = loss_monitor + scaled_query_loss.detach()
        if callable(anil_query_loss_sink):
            anil_query_loss_sink(scaled_query_loss)
        else:
            loss_total = loss_total + scaled_query_loss

    stats = _finalize_anil_scalar_stats(query_stats_accum, device=device_obj)
    stats["anil_support_objective"] = torch.as_tensor(
        float(support_objective_sum),
        device=device_obj,
        dtype=torch.float32,
    )
    stats["anil_inner_steps"] = int(anil_inner_steps)
    stats["anil_inner_learning_rate"] = float(anil_inner_learning_rate)
    stats["anil_task_count"] = int(batch_size)
    if query_terminal_stats_accum:
        stats.update(_finalize_anil_scalar_stats(query_terminal_stats_accum, device=device_obj))
    if isinstance(rollout_profile_acc, dict):
        fake_rollout = {
            "rewards": torch.zeros((1, int(batch_size)), device=device_obj, dtype=torch.float32),
            "rollout_profile": rollout_profile_acc,
        }
        attach_common_rollout_diagnostics(fake_rollout, stats)
    return (loss_monitor.detach() if callable(anil_query_loss_sink) else loss_total), None, stats


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
    pg_saved_tensors_cpu_offload_scope="all",
    pg_saved_tensors_pin_memory=True,
    pg_saved_tensors_cpu_offload_auto_disable_when_safe=False,
    pg_saved_tensors_cpu_offload_auto_min_free_gb=8.0,
    pg_saved_tensors_cpu_offload_auto_max_batch_size=64,
    pg_saved_tensors_cpu_offload_auto_max_n_samples=1024,
    pg_tbptt_window=None,
    tbptt_loss_sink=None,
    reinforce_replay_loss_sink=None,
    anil_inner_steps=1,
    anil_inner_learning_rate=0.1,
    anil_query_loss_sink=None,
    h_list_override=None,
    env_seeds_override=None,
    rollout_seeds_override=None,
    rl_objective="policy_gradient",
):
    offload_scope = str(pg_saved_tensors_cpu_offload_scope or "all").strip().lower()
    if offload_scope not in {"all", "policy"}:
        offload_scope = "all"
    one_hop_replay_enabled = _resolve_env_prior_pg_one_hop_replay_enabled(env_prior)
    effective_policy_saved_tensors_cpu_offload = _resolve_effective_policy_saved_tensors_offload(
        enabled=pg_saved_tensors_cpu_offload,
        scope=offload_scope,
        device=device,
        batch_size=batch_size,
        n_samples=n_samples,
        one_hop_replay_enabled=one_hop_replay_enabled,
        auto_disable_when_safe=pg_saved_tensors_cpu_offload_auto_disable_when_safe,
        auto_min_free_gb=pg_saved_tensors_cpu_offload_auto_min_free_gb,
        auto_max_batch_size=pg_saved_tensors_cpu_offload_auto_max_batch_size,
        auto_max_n_samples=pg_saved_tensors_cpu_offload_auto_max_n_samples,
    )
    full_offload_enabled = bool(effective_policy_saved_tensors_cpu_offload) and (offload_scope == "all")
    policy_step_fn = _wrap_policy_step_fn_saved_tensors_offload(
        policy_step_fn,
        enabled=bool(effective_policy_saved_tensors_cpu_offload) and (offload_scope == "policy"),
        pin_memory=pg_saved_tensors_pin_memory,
    )
    rollout_kwargs = {}
    if h_list_override is not None:
        rollout_kwargs["h_list_override"] = h_list_override
    if env_seeds_override is not None:
        rollout_kwargs["env_seeds_override"] = env_seeds_override
    if rollout_seeds_override is not None:
        rollout_kwargs["rollout_seeds_override"] = rollout_seeds_override

    if str(rl_objective).strip().lower() == "anil":
        if pg_tbptt_window is not None or tbptt_loss_sink is not None:
            raise RuntimeError("ANIL currently requires pg_tbptt_window=None and does not support TBPTT.")
        if bool(policy_rollout_checkpoint):
            raise RuntimeError("ANIL currently does not support policy_rollout_checkpoint.")
        return _compute_anil_rollout_chunk_loss(
            env_prior,
            policy_step_fn,
            batch_size=batch_size,
            n_samples=n_samples,
            num_features=num_features,
            device=device,
            single_eval_pos=single_eval_pos,
            h_list_override=rollout_kwargs.get("h_list_override", None),
            env_seeds_override=rollout_kwargs.get("env_seeds_override", None),
            rollout_seeds_override=rollout_kwargs.get("rollout_seeds_override", None),
            anil_inner_steps=anil_inner_steps,
            anil_inner_learning_rate=anil_inner_learning_rate,
            anil_query_loss_sink=anil_query_loss_sink,
        )

    if tbptt_loss_sink is not None:
        # True TBPTT mode: window loss is backpropagated as soon as the window ends.
        # This keeps memory bounded by one TBPTT window and is incompatible with
        # outer rollout checkpointing.
        with _saved_tensors_cpu_offload_context(
            enabled=full_offload_enabled,
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
                reinforce_replay_loss_sink=reinforce_replay_loss_sink,
                policy_objective_kind=rl_objective,
                **rollout_kwargs,
            )

    if not policy_rollout_checkpoint:
        with _saved_tensors_cpu_offload_context(
            enabled=full_offload_enabled,
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
                reinforce_replay_loss_sink=reinforce_replay_loss_sink,
                policy_objective_kind=rl_objective,
                **rollout_kwargs,
            )

    rng_state = _capture_rng_state(device)
    dummy = torch.zeros((), device=device, dtype=torch.float32, requires_grad=True)

    def _rollout_loss_only(_dummy):
        del _dummy
        _restore_rng_state(rng_state)
        with _saved_tensors_cpu_offload_context(
            enabled=full_offload_enabled,
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
                reinforce_replay_loss_sink=reinforce_replay_loss_sink,
                policy_objective_kind=rl_objective,
                **rollout_kwargs,
            )
        return (
            pg_loss_inner,
            pg_stats_inner["objective"].detach(),
            pg_stats_inner["reward_mean"].detach(),
            pg_stats_inner["reward_std"].detach(),
            pg_stats_inner.get("policy_total_loss", pg_loss_inner).detach(),
            pg_stats_inner.get("reinforce_loss", pg_loss_inner).detach(),
            pg_stats_inner.get(
                "normalized_q_value_loss",
                torch.zeros((), device=device, dtype=torch.float32),
            ).detach(),
            pg_stats_inner.get(
                "next_state_flow_matching_loss",
                torch.zeros((), device=device, dtype=torch.float32),
            ).detach(),
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
        policy_total_loss,
        reinforce_loss,
        normalized_q_value_loss,
        next_state_flow_matching_loss,
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
        "policy_total_loss": policy_total_loss,
        "reinforce_loss": reinforce_loss,
        "normalized_q_value_loss": normalized_q_value_loss,
        "next_state_flow_matching_loss": next_state_flow_matching_loss,
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
    _clear_policy_rollout_artifacts(env_prior=env_prior, policy_step_fn=policy_step_fn)

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
    pg_saved_tensors_cpu_offload_scope="all",
    pg_saved_tensors_pin_memory=True,
    pg_saved_tensors_cpu_offload_auto_disable_when_safe=False,
    pg_saved_tensors_cpu_offload_auto_min_free_gb=8.0,
    pg_saved_tensors_cpu_offload_auto_max_batch_size=64,
    pg_saved_tensors_cpu_offload_auto_max_n_samples=1024,
    pg_oom_debug_raise=False,
    pg_oom_fail_fast=False,
    pg_kv_cache_mode="auto",
    pg_kv_cache_page_size=None,
    pg_tbptt_window=None,
    pg_env_replay_steps=1,
    pg_oom_reduce_tbptt_first=False,
    pg_torch_compile=False,
    pg_torch_compile_backend="inductor",
    pg_torch_compile_mode="reduce-overhead",
    pg_torch_compile_fullgraph=False,
    pg_torch_compile_dynamic=False,
    pg_compile_observe_recompiles=False,
    pg_compile_observe_log_every_batches=1,
    pg_compile_observe_output_path=None,
    pg_compile_observe_reset_after_warmup=True,
    pg_phase_log_every_batches=1,
    pg_phase_log_file=None,
    anil_inner_steps=1,
    anil_inner_learning_rate=0.1,
    epoch_idx=None,
    progress_bar=False,
    epoch_profiler=None,
    epoch_start_time=None,
    train_profiler_log_every_batches=0,
    verbose=False,
    gpu_observer=None,
    kernel_profiler=None,
    rl_objective="policy_gradient",
):
    model.train()
    rl_objective = str(rl_objective).strip().lower()
    policy_autocast_dtype = _resolve_policy_autocast_dtype(device)
    device_obj = device if isinstance(device, torch.device) else torch.device(str(device))
    total_loss = torch.tensor(0.0, device=device)
    valid_steps = torch.tensor(0.0, device=device)
    skipped_steps = 0
    grad_accum_steps = 0
    try:
        pg_env_replay_steps = int(pg_env_replay_steps)
    except Exception:
        pg_env_replay_steps = 1
    pg_env_replay_steps = max(1, pg_env_replay_steps)
    if pg_env_replay_steps > 1 and int(aggregate_k_gradients) != 1:
        raise ValueError(
            "pg_env_replay_steps > 1 requires aggregate_k_gradients == 1 "
            "so each replay rollout is followed by exactly one optimizer step."
        )
    if rl_objective == "anil" and pg_env_replay_steps != 1:
        raise ValueError("ANIL currently requires pg_env_replay_steps == 1.")
    requires_midaccum_grad_finite_check = aggregate_k_gradients > 1
    pg_epoch_objective_values = []
    pg_epoch_reward_mean_values = []
    pg_epoch_reward_std_values = []
    pg_epoch_reward_env_mean_values = []
    pg_epoch_reward_env_std_values = []
    pg_epoch_reward_env_return_values = []
    pg_epoch_reward_ctrl_mean_values = []
    pg_epoch_reward_ctrl_std_values = []
    pg_epoch_reward_ctrl_return_values = []
    pg_epoch_reward_absmax_values = []
    pg_epoch_clip_hit_values = []
    pg_epoch_norm_clip_hit_values = []
    pg_epoch_grad_norm_values = []
    pg_epoch_grad_abs_mean_values = []
    pg_epoch_grad_abs_max_values = []
    pg_epoch_grad_rms_values = []
    pg_epoch_grad_nonfinite_share_values = []
    pg_epoch_grad_zero_share_values = []
    pg_epoch_lr_step_proxy_values = []
    pg_epoch_rollout_wall_values = []
    pg_epoch_backward_wall_values = []
    pg_epoch_step_wall_values = []
    pg_epoch_policy_total_loss_values = []
    pg_epoch_reinforce_loss_values = []
    pg_epoch_normalized_q_value_loss_values = []
    pg_epoch_next_state_flow_matching_loss_values = []
    pg_epoch_loss_signatures = set()

    base_steps_per_epoch = int(len(dl))
    steps_per_epoch = int(base_steps_per_epoch)
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
    compile_mode_effective = str(pg_torch_compile_mode)
    if bool(pg_torch_compile):
        compile_no_cudagraph_env = str(
            os.environ.get("TICL_POLICY_COMPILE_NO_CUDAGRAPHS", "auto")
        ).strip().lower()
        if compile_no_cudagraph_env in {"1", "true", "yes", "on"}:
            force_no_cudagraphs = True
        elif compile_no_cudagraph_env in {"0", "false", "no", "off"}:
            force_no_cudagraphs = False
        else:
            # Mutable paged KV updates are incompatible with cudagraph replay
            # under dynamic compilation in this rollout workload.
            force_no_cudagraphs = bool(allow_grad_mutable_kv) and (str(kv_cache_mode) == "paged")
        if force_no_cudagraphs and str(compile_mode_effective).strip().lower() == "reduce-overhead":
            # Keep compile startup bounded: use default mode without cudagraph
            # assumptions instead of max-autotune no-cudagraph mode.
            compile_mode_effective = "default"

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
        pg_torch_compile_mode=str(compile_mode_effective),
        pg_torch_compile_fullgraph=bool(pg_torch_compile_fullgraph),
        pg_torch_compile_dynamic=bool(pg_torch_compile_dynamic),
    )
    policy_step_compile_active = bool(getattr(policy_step_fn, "_compile_active", False))
    if bool(pg_torch_compile) and str(compile_mode_effective) != str(pg_torch_compile_mode):
        print(
            "[pg-compile-note] adjusted compile mode for mutable paged KV cache: "
            f"{pg_torch_compile_mode} -> {compile_mode_effective}"
        )
    model_ref = model.module if hasattr(model, "module") else model
    policy_step_profile_consumer = getattr(model_ref, "consume_policy_step_profile", None)
    policy_fastpath_compile_active_fn = getattr(model_ref, "policy_fastpath_compile_active", None)
    policy_fastpath_compile_active = bool(
        policy_fastpath_compile_active_fn()
    ) if callable(policy_fastpath_compile_active_fn) else False
    policy_fastpath_compile_cfg_fn = getattr(model_ref, "get_policy_fastpath_compile_config", None)
    if callable(policy_fastpath_compile_cfg_fn):
        policy_fastpath_compile_cfg = policy_fastpath_compile_cfg_fn()
    else:
        policy_fastpath_compile_cfg = {"finalize_torch_compile": False}
    policy_fastpath_warmup_fn = getattr(model_ref, "warmup_policy_fastpaths", None)
    batch_size = int(dl.batch_size)
    n_samples = int(dl.n_samples)
    num_features = int(dl.num_features)
    if pg_tbptt_window is None:
        current_tbptt_window = None
    else:
        tbptt_value = int(pg_tbptt_window)
        current_tbptt_window = tbptt_value if tbptt_value > 0 else None
    if current_tbptt_window is None:
        current_tbptt_stream_merge_windows = 1
    else:
        try:
            current_tbptt_stream_merge_windows = int(
                os.environ.get("TICL_POLICY_TBPTT_STREAM_MERGE_WINDOWS", "1")
            )
        except Exception:
            current_tbptt_stream_merge_windows = 1
        current_tbptt_stream_merge_windows = int(max(1, current_tbptt_stream_merge_windows))
    tbptt_merge_auto_flag = str(
        os.environ.get("TICL_POLICY_TBPTT_STREAM_MERGE_AUTO", "0")
    ).strip().lower()
    tbptt_merge_auto_enabled = (
        tbptt_merge_auto_flag in {"1", "true", "yes", "on"}
        and current_tbptt_window is not None
    )
    try:
        tbptt_merge_auto_max_windows = int(
            os.environ.get("TICL_POLICY_TBPTT_STREAM_MERGE_AUTO_MAX_WINDOWS", "2")
        )
    except Exception:
        tbptt_merge_auto_max_windows = 2
    tbptt_merge_auto_max_windows = int(max(1, tbptt_merge_auto_max_windows))
    if tbptt_merge_auto_enabled:
        current_tbptt_stream_merge_windows = int(
            max(int(current_tbptt_stream_merge_windows), int(tbptt_merge_auto_max_windows))
        )
    try:
        tbptt_merge_auto_min_free_gb = float(
            os.environ.get("TICL_POLICY_TBPTT_STREAM_MERGE_AUTO_MIN_FREE_GB", "18.0")
        )
    except Exception:
        tbptt_merge_auto_min_free_gb = 18.0
    if (not math.isfinite(tbptt_merge_auto_min_free_gb)) or tbptt_merge_auto_min_free_gb < 0.0:
        tbptt_merge_auto_min_free_gb = 18.0
    tbptt_merge_auto_min_free_bytes = int(tbptt_merge_auto_min_free_gb * (1024.0 ** 3))
    try:
        tbptt_merge_auto_reserved_frac = float(
            os.environ.get("TICL_POLICY_TBPTT_STREAM_MERGE_AUTO_RESERVED_FRAC", "0.72")
        )
    except Exception:
        tbptt_merge_auto_reserved_frac = 0.72
    if (not math.isfinite(tbptt_merge_auto_reserved_frac)) or tbptt_merge_auto_reserved_frac <= 0.0:
        tbptt_merge_auto_reserved_frac = 0.72
    tbptt_merge_auto_reserved_frac = float(min(0.98, max(0.50, tbptt_merge_auto_reserved_frac)))
    tbptt_merge_guard_flag = str(
        os.environ.get("TICL_POLICY_TBPTT_STREAM_MERGE_GUARD", "1")
    ).strip().lower()
    tbptt_merge_guard_enabled = (
        tbptt_merge_guard_flag not in {"0", "false", "no", "off"}
        and current_tbptt_window is not None
    )
    try:
        tbptt_merge_guard_min_free_gb = float(
            os.environ.get("TICL_POLICY_TBPTT_STREAM_MERGE_GUARD_MIN_FREE_GB", "12.0")
        )
    except Exception:
        tbptt_merge_guard_min_free_gb = 12.0
    if (not math.isfinite(tbptt_merge_guard_min_free_gb)) or tbptt_merge_guard_min_free_gb < 0.0:
        tbptt_merge_guard_min_free_gb = 12.0
    tbptt_merge_guard_min_free_bytes = int(tbptt_merge_guard_min_free_gb * (1024.0 ** 3))
    try:
        tbptt_merge_guard_reserved_frac = float(
            os.environ.get("TICL_POLICY_TBPTT_STREAM_MERGE_GUARD_RESERVED_FRAC", "0.82")
        )
    except Exception:
        tbptt_merge_guard_reserved_frac = 0.82
    if (not math.isfinite(tbptt_merge_guard_reserved_frac)) or tbptt_merge_guard_reserved_frac <= 0.0:
        tbptt_merge_guard_reserved_frac = 0.82
    tbptt_merge_guard_reserved_frac = float(min(0.98, max(0.50, tbptt_merge_guard_reserved_frac)))
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
        pg_phase_log_every_cfg = int(pg_phase_log_every_batches)
    except Exception:
        pg_phase_log_every_cfg = 1
    pg_phase_log_every_cfg = max(1, pg_phase_log_every_cfg)
    try:
        pg_phase_log_every = int(
            os.environ.get("TICL_PG_PHASE_LOG_EVERY_BATCHES", str(pg_phase_log_every_cfg))
        )
    except Exception:
        pg_phase_log_every = pg_phase_log_every_cfg
    pg_phase_log_every = max(1, pg_phase_log_every)
    try:
        pg_detail_log_every = int(os.environ.get("TICL_PG_DETAIL_LOG_EVERY_BATCHES", "50"))
    except Exception:
        pg_detail_log_every = 50
    pg_detail_log_every = max(1, pg_detail_log_every)
    try:
        grad_zero_tol = float(os.environ.get("TICL_PG_GRAD_ZERO_TOL", "1e-12"))
    except Exception:
        grad_zero_tol = 1e-12
    if (not math.isfinite(grad_zero_tol)) or grad_zero_tol < 0.0:
        grad_zero_tol = 1e-12
    try:
        pg_same_cfg_oom_retries = int(os.environ.get("TICL_PG_OOM_SAME_CFG_RETRIES", "0"))
    except Exception:
        pg_same_cfg_oom_retries = 0
    pg_same_cfg_oom_retries = max(0, pg_same_cfg_oom_retries)
    oom_fail_fast_flag = str(os.environ.get("TICL_POLICY_OOM_FAIL_FAST", "0")).strip().lower()
    oom_fail_fast_env = oom_fail_fast_flag in {"1", "true", "yes", "on"}
    pg_oom_fail_fast = bool(pg_oom_fail_fast) or bool(oom_fail_fast_env)
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
    compile_warmup_flag = str(os.environ.get("TICL_POLICY_COMPILE_WARMUP", "1")).strip().lower()
    compile_warmup_enabled = bool(policy_step_compile_active or policy_fastpath_compile_active) and (
        compile_warmup_flag not in {"0", "false", "no", "off"}
    )
    try:
        compile_warmup_steps = int(os.environ.get("TICL_POLICY_COMPILE_WARMUP_STEPS", "1"))
    except Exception:
        compile_warmup_steps = 1
    compile_warmup_steps = int(max(1, compile_warmup_steps))
    try:
        compile_warmup_n_samples_cap = int(
            os.environ.get("TICL_POLICY_COMPILE_WARMUP_SAMPLES", "2")
        )
    except Exception:
        compile_warmup_n_samples_cap = 2
    compile_warmup_n_samples_cap = int(max(2, compile_warmup_n_samples_cap))
    try:
        compile_warmup_chunk_cap = int(
            os.environ.get("TICL_POLICY_COMPILE_WARMUP_CHUNK", str(batch_size))
        )
    except Exception:
        compile_warmup_chunk_cap = int(batch_size)
    compile_warmup_chunk_cap = int(max(1, min(int(batch_size), compile_warmup_chunk_cap)))
    compile_warmup_done = not bool(compile_warmup_enabled)
    compile_observe_enabled = bool(policy_step_compile_active or policy_fastpath_compile_active) and bool(
        pg_compile_observe_recompiles
    )
    try:
        compile_observe_log_every_batches = int(pg_compile_observe_log_every_batches)
    except Exception:
        compile_observe_log_every_batches = 1
    compile_observe_log_every_batches = int(max(1, compile_observe_log_every_batches))
    compile_observe_output_path = (
        str(pg_compile_observe_output_path).strip()
        if pg_compile_observe_output_path is not None
        else None
    )
    if compile_observe_output_path == "":
        compile_observe_output_path = None
    compile_observe_reset_after_warmup = bool(pg_compile_observe_reset_after_warmup)
    compile_counter_prev_snapshot = _snapshot_dynamo_counters() if compile_observe_enabled else None
    if compile_observe_output_path is not None:
        try:
            out_dir = os.path.dirname(os.path.abspath(compile_observe_output_path))
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
        except Exception:
            compile_observe_output_path = None

    def _emit_pg_phase_log_line(msg):
        stamped = f"[{time.strftime('%m/%d/%Y %H:%M:%S')}] {msg}"
        print(stamped)
        _append_text_log_line(pg_phase_log_file, stamped)

    iterator = range(steps_per_epoch)
    if progress_bar:
        desc = f"Epoch {epoch_idx}" if epoch_idx is not None else "Epoch"
        iterator = tqdm(iterator, total=steps_per_epoch, desc=desc, unit="step")

    for batch in iterator:
        _clear_policy_rollout_artifacts(env_prior=env_prior, policy_step_fn=policy_step_fn)
        pg_loss_chunk = None
        rollout_chunk = None
        pg_stats_chunk = None
        weighted_pg_loss = None
        loss = None
        metric_loss = None
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
                or (((batch + 1) % pg_detail_log_every) == 0)
            )
        )
        should_log_pg_phase = bool(
            (batch == 0)
            or ((batch + 1) == steps_per_epoch)
            or (((batch + 1) % pg_phase_log_every) == 0)
        )
        if using_dist and not ((grad_accum_steps + 1) == aggregate_k_gradients):
            cm = model.no_sync()
        else:
            cm = nullcontext()

        with cm:
            if pg_env_replay_steps > 1:
                fixed_single_eval_pos = int(env_prior._sample_single_eval_pos(n_samples, None))
                fixed_batch_h_list_override = _freeze_env_h_list_for_replay(
                    env_prior,
                    env_prior._sample_batch_hypers(batch_size),
                )
                fixed_batch_env_seeds_override = env_prior._sample_seed_list(batch_size)
            else:
                fixed_single_eval_pos = None
                fixed_batch_h_list_override = None
                fixed_batch_env_seeds_override = None

            def _format_pg_phase_start_alpha_suffix(h_list_preview):
                if h_list_preview is None:
                    return ""
                alpha_values = []
                for h in h_list_preview:
                    if not isinstance(h, dict) or ("alpha" not in h):
                        continue
                    try:
                        alpha_values.append(float(max(1e-4, min(1.0, float(h["alpha"])))))
                    except Exception:
                        continue
                if len(alpha_values) <= 0:
                    return ""
                alpha_tensor = torch.as_tensor(alpha_values, dtype=torch.float32)
                return (
                    f" env_alpha_mean={float(alpha_tensor.mean().item()):.3e}"
                    f" env_alpha_min={float(alpha_tensor.min().item()):.3e}"
                    f" env_alpha_max={float(alpha_tensor.max().item()):.3e}"
                )

            def _format_pg_phase_start_terminal_suffix(h_list_preview):
                if h_list_preview is None:
                    return ""
                target_values = []
                for h in h_list_preview:
                    if not isinstance(h, dict):
                        continue
                    try:
                        if not bool(env_prior._resolve_terminal_reset_enabled(h)):
                            continue
                        target_values.append(float(env_prior._resolve_terminal_reset_count_target(h)))
                    except Exception:
                        continue
                if len(target_values) <= 0:
                    return ""
                target_tensor = torch.as_tensor(target_values, dtype=torch.float32)
                return (
                    f" envXmean={float(target_tensor.mean().item()):.3f}"
                    f" envXmin={float(target_tensor.min().item()):.0f}"
                    f" envXmax={float(target_tensor.max().item()):.0f}"
                )

            def _format_pg_phase_terminal_suffix(
                target_mean,
                target_min,
                target_max,
                realized_mean,
                realized_min,
                realized_max,
                bonus_mean,
                bonus_min,
                bonus_max,
                bonus_event_mean,
            ):
                parts = []
                if target_mean is not None:
                    parts.append(
                        f" Xmean={float(target_mean):.3f}"
                        f" Xmin={float(target_min):.0f}"
                        f" Xmax={float(target_max):.0f}"
                    )
                if realized_mean is not None:
                    parts.append(
                        f" hatXmean={float(realized_mean):.3f}"
                        f" hatXmin={float(realized_min):.0f}"
                        f" hatXmax={float(realized_max):.0f}"
                    )
                if bonus_mean is not None:
                    parts.append(
                        f" bonus_mean={float(bonus_mean):+.3e}"
                        f" bonus_min={float(bonus_min):+.3e}"
                        f" bonus_max={float(bonus_max):+.3e}"
                    )
                if bonus_event_mean is not None:
                    parts.append(f" bonus_event_mean={float(bonus_event_mean):+.3e}")
                return "".join(parts)

            for replay_step_idx in range(pg_env_replay_steps):
                replay_phase_suffix = ""
                if pg_env_replay_steps > 1:
                    replay_phase_suffix = (
                        f" base_batch={int(batch)} "
                        f"replay_step={replay_step_idx + 1}/{pg_env_replay_steps}"
                    )
                if fixed_single_eval_pos is not None:
                    single_eval_pos = fixed_single_eval_pos
                    batch_h_list_override = fixed_batch_h_list_override
                    batch_env_seeds_override = fixed_batch_env_seeds_override
                else:
                    single_eval_pos = env_prior._sample_single_eval_pos(n_samples, None)
                    batch_h_list_override = None
                    batch_env_seeds_override = None
                pg_phase_start_alpha_suffix = ""
                pg_phase_start_terminal_suffix = ""
                if should_log_pg_phase:
                    if batch_h_list_override is not None:
                        pg_phase_start_alpha_suffix = _format_pg_phase_start_alpha_suffix(batch_h_list_override)
                        pg_phase_start_terminal_suffix = _format_pg_phase_start_terminal_suffix(batch_h_list_override)
                    elif callable(getattr(env_prior, "_sample_batch_hypers", None)):
                        preview_rng_state = _capture_rng_state(device)
                        try:
                            preview_h_list = env_prior._sample_batch_hypers(batch_size)
                        finally:
                            _restore_rng_state(preview_rng_state)
                        pg_phase_start_alpha_suffix = _format_pg_phase_start_alpha_suffix(preview_h_list)
                        pg_phase_start_terminal_suffix = _format_pg_phase_start_terminal_suffix(preview_h_list)
                if should_log_pg_phase:
                    _emit_pg_phase_log_line(
                        f"[pg-phase-start] epoch={batch_epoch_for_profile} batch={batch}{replay_phase_suffix} "
                        f"chunk={current_rollout_chunk_size} tbptt={current_tbptt_window} "
                        f"status=start rl_objective={rl_objective} "
                        f"batch_size={int(batch_size)} n_samples={int(n_samples)} "
                        f"single_eval_pos={int(single_eval_pos)}"
                        f"{pg_phase_start_alpha_suffix}"
                        f"{pg_phase_start_terminal_suffix}"
                    )
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
                    _clear_policy_rollout_artifacts(env_prior=env_prior, policy_step_fn=policy_step_fn)
                    pg_loss_chunk = None
                    rollout_chunk = None
                    pg_stats_chunk = None
                    weighted_pg_loss = None
                    loss = None
                    metric_loss = None

                    batch_loss = torch.tensor(0.0, device=device)
                    batch_objective = torch.tensor(0.0, device=device)
                    batch_reward_mean = torch.tensor(0.0, device=device)
                    batch_reward_std = torch.tensor(0.0, device=device)
                    batch_reward_env_mean_value = None
                    batch_reward_env_std_value = None
                    batch_reward_env_return_value = None
                    batch_reward_ctrl_mean_value = None
                    batch_reward_ctrl_std_value = None
                    batch_reward_ctrl_return_value = None
                    batch_policy_total_loss_value = None
                    batch_reinforce_loss_value = None
                    batch_normalized_q_value_loss_value = None
                    batch_next_state_flow_matching_loss_value = None
                    batch_reward_min_value = float("inf")
                    batch_reward_max_value = float("-inf")
                    batch_reward_absmax_value = 0.0
                    batch_reward_nonfinite_share = 0.0
                    batch_reward_nan_share = 0.0
                    batch_reward_inf_share = 0.0
                    batch_reward_clip_hit_share = 0.0
                    batch_reward_norm_clip_hit_share = 0.0
                    batch_terminal_count_mean_value = None
                    batch_terminal_count_min_value = None
                    batch_terminal_count_max_value = None
                    batch_terminal_target_mean_value = None
                    batch_terminal_target_min_value = None
                    batch_terminal_target_max_value = None
                    batch_terminal_bonus_mean_value = None
                    batch_terminal_bonus_min_value = None
                    batch_terminal_bonus_max_value = None
                    batch_terminal_bonus_event_mean_value = None
                    batch_pg_bridge_replay_count_value = 0
                    batch_reinforce_return_nonfinite_share = 0.0
                    batch_reinforce_log_prob_nonfinite_share = 0.0
                    batch_reinforce_adv_nonfinite_share = 0.0
                    batch_reinforce_log_prob_mean_value = None
                    batch_reinforce_log_prob_std_value = None
                    batch_reinforce_action_dim_mean_value = None
                    batch_reinforce_action_dim_min_value = float("inf")
                    batch_reinforce_action_dim_max_value = float("-inf")
                    batch_reinforce_action_std_mean_value = None
                    batch_reinforce_action_std_min_value = float("inf")
                    batch_reinforce_action_std_max_value = float("-inf")
                    batch_reinforce_logprob_log_std_mean_value = None
                    batch_reinforce_logprob_log_std_min_value = float("inf")
                    batch_reinforce_logprob_log_std_max_value = float("-inf")
                    batch_reinforce_logprob_z2_mean_value = None
                    batch_reinforce_logprob_z2_max_value = float("-inf")
                    batch_state_abs_max_value = 0.0
                    batch_action_std_min_value = float("inf")
                    batch_action_std_max_value = float("-inf")
                    batch_nonfinite = False
                    batch_rollout_wall = 0.0
                    batch_backward_wall = 0.0
                    batch_backward_calls = 0
                    batch_tbptt_stream_backward_active = False
                    batch_tbptt_stream_merge_windows = 1
                    batch_tbptt_stream_backward_launches = 0
                    batch_tbptt_stream_backward_roots = 0
                    batch_tbptt_stream_merge_guard_prefallbacks = 0
                    batch_tbptt_stream_merge_guard_flushes = 0
                    batch_step_wall = 0.0
                    batch_rollout_policy_cuda_ms = 0.0
                    batch_rollout_transition_cuda_ms = 0.0
                    batch_rollout_policy_wall_ms = 0.0
                    batch_rollout_transition_wall_ms = 0.0
                    batch_rollout_transition_y_wall_ms = 0.0
                    batch_rollout_transition_x_wall_ms = 0.0
                    batch_rollout_transition_gp_first_projection_wall_ms = 0.0
                    batch_rollout_transition_gp_second_projection_wall_ms = 0.0
                    batch_rollout_transition_gp_projection_call_count = 0
                    batch_rollout_transition_gp_rff_fused_call_count = 0
                    batch_rollout_transition_gp_profile_group_count = 0
                    batch_rollout_transition_gp_profile_sync_group_count = 0
                    batch_rollout_transition_packed_env_input_group_count = 0
                    batch_rollout_transition_packed_env_input_call_count = 0
                    batch_rollout_transition_only_build_group_count = 0
                    batch_rollout_transition_only_skipped_generator_count = 0
                    batch_rollout_transition_group_wall_ms = 0.0
                    batch_rollout_transition_group_launch_wall_ms = 0.0
                    batch_rollout_transition_group_sync_wall_ms = 0.0
                    batch_rollout_transition_env_pack_wall_ms = 0.0
                    batch_rollout_transition_state_update_wall_ms = 0.0
                    batch_rollout_transition_noise_wall_ms = 0.0
                    batch_rollout_transition_fused_wall_ms = 0.0
                    batch_rollout_transition_fused_launch_wall_ms = 0.0
                    batch_rollout_transition_fused_call_count = 0
                    batch_rollout_transition_fused_group_count = 0
                    batch_rollout_transition_fused_enabled = 0
                    batch_rollout_transition_checkpoint_enabled = 0
                    batch_rollout_transition_checkpoint_call_count = 0
                    batch_rollout_transition_group_count = 0
                    batch_rollout_transition_family_group_count = 0
                    batch_rollout_transition_inner_grouping_structure_enabled = 0
                    batch_rollout_transition_inner_min_bucket = 0
                    batch_rollout_transition_bucket_max_batch = 0
                    batch_rollout_transition_work_actual_est = 0.0
                    batch_rollout_transition_work_padded_est = 0.0
                    batch_rollout_transition_async_enabled = 0
                    batch_rollout_noise_mode = None
                    batch_rollout_noise_block_size = None
                    batch_rollout_env_count = 0
                    batch_rollout_strict_joint_transition_count = 0
                    batch_rollout_reference_semantics_count = 0
                    batch_rollout_strict_joint_transition_share = 0.0
                    batch_rollout_reference_semantics_share = 0.0
                    batch_rollout_exact_scm_count = 0
                    batch_rollout_exact_gp_count = 0
                    batch_rollout_legacy_scm_count = 0
                    batch_rollout_legacy_gp_count = 0
                    batch_rollout_transition_reference_mode = None
                    batch_tbptt_backward_mem_guard_hits = 0
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
                    batch_policy_step_transformer_layer_finalize_attn_outproj_linear_ms = 0.0
                    batch_policy_step_transformer_layer_finalize_attn_outproj_norm_ms = 0.0
                    batch_policy_step_transformer_layer_finalize_ffn_ms = 0.0
                    batch_policy_step_transformer_layer_finalize_ffn_linear1_act_ms = 0.0
                    batch_policy_step_transformer_layer_finalize_ffn_linear2_residual_norm_ms = 0.0
                    batch_policy_step_transformer_layer_finalize_ffn_linear2_ms = 0.0
                    batch_policy_step_transformer_layer_finalize_ffn_residual_norm_ms = 0.0
                    batch_policy_step_transformer_layer_finalize_compiled_ms = 0.0
                    batch_policy_step_transformer_layer_total_ms = 0.0
                    batch_policy_step_layer_paged_single_page_calls = 0
                    batch_policy_step_layer_paged_flash_prefix_calls = 0
                    batch_policy_step_layer_paged_flash_prefix_zero_fastpath_calls = 0
                    batch_policy_step_layer_paged_flash_merge_calls = 0
                    batch_policy_step_layer_paged_dense_calls = 0
                    batch_policy_step_layer_paged_page_count_sum = 0
                    batch_policy_step_layer_paged_valid_len_sum = 0
                    batch_policy_step_layer_paged_last_page_tokens_sum = 0
                    batch_policy_step_layer_paged_prefix_len_sum = 0
                    batch_policy_step_layer_flash_prefix_valid_tokens_sum = 0
                    batch_policy_step_layer_flash_prefix_prefix_tokens_sum = 0
                    batch_policy_step_layer_flash_prefix_tail_tokens_sum = 0
                    batch_policy_step_layer_dense_valid_tokens_sum = 0
                    batch_policy_step_layer_dense_prefix_tokens_sum = 0
                    batch_policy_step_layer_dense_tail_tokens_sum = 0
                    batch_compile_warmup_wall = 0.0
                    batch_compile_warmup_ok = None
                    batch_compile_counter_delta_nonzero = 0
                    batch_compile_counter_delta_total = 0
                    batch_compile_counter_delta_graph_breaks = 0
                    batch_compile_counter_delta_unique_graphs = 0
                    batch_compile_counter_delta_recompiles = 0
                    batch_compile_counter_delta_summary = "none"
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
                                objective_kind=rl_objective,
                            )
                        else:
                            batch_pg_loss_signature = "na"
                    except Exception:
                        batch_pg_loss_signature = None
                    batch_grad_norm_value = None
                    batch_grad_norm_postclip_value = None
                    batch_grad_numel_value = None
                    batch_grad_nonfinite_count = None
                    batch_grad_nonfinite_share = None
                    batch_grad_zero_share = None
                    batch_grad_abs_mean_value = None
                    batch_grad_abs_max_value = None
                    batch_grad_rms_value = None
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

                    if (not compile_warmup_done) and bool(compile_warmup_enabled):
                        warmup_chunk_bs = int(
                            max(1, min(int(current_rollout_chunk_size), int(compile_warmup_chunk_cap)))
                        )
                        warmup_n_samples = int(max(2, min(int(n_samples), int(compile_warmup_n_samples_cap))))
                        if warmup_n_samples < int(n_samples):
                            warmup_single_eval_pos = int(max(1, min(int(single_eval_pos), warmup_n_samples - 1)))
                        else:
                            warmup_single_eval_pos = int(single_eval_pos)
                        warmup_h_list_override = None
                        warmup_env_seeds_override = None
                        if batch_h_list_override is not None:
                            warmup_h_list_override = batch_h_list_override[:warmup_chunk_bs]
                        if batch_env_seeds_override is not None:
                            warmup_env_seeds_override = batch_env_seeds_override[:warmup_chunk_bs]
                        warmup_rng_state = _capture_rng_state(device)
                        warmup_t0 = time.perf_counter()
                        warmup_ok = True
                        try:
                            if bool(policy_step_compile_active):
                                for _ in range(int(compile_warmup_steps)):
                                    with _amp_autocast_context(
                                        scaler,
                                        device=device,
                                        dtype=policy_autocast_dtype,
                                    ):
                                        _warmup_loss, _, _ = _compute_policy_rollout_chunk_loss(
                                            env_prior=env_prior,
                                            policy_step_fn=policy_step_fn,
                                            batch_size=warmup_chunk_bs,
                                            n_samples=warmup_n_samples,
                                            num_features=num_features,
                                            device=device,
                                            single_eval_pos=warmup_single_eval_pos,
                                            collect_x=False,
                                            policy_rollout_checkpoint=False,
                                            policy_rollout_checkpoint_reentrant=checkpoint_reentrant_active,
                                            pg_saved_tensors_cpu_offload=pg_saved_tensors_cpu_offload,
                                            pg_saved_tensors_cpu_offload_scope=pg_saved_tensors_cpu_offload_scope,
                                            pg_saved_tensors_pin_memory=pg_saved_tensors_pin_memory,
                                            pg_saved_tensors_cpu_offload_auto_disable_when_safe=pg_saved_tensors_cpu_offload_auto_disable_when_safe,
                                            pg_saved_tensors_cpu_offload_auto_min_free_gb=pg_saved_tensors_cpu_offload_auto_min_free_gb,
                                            pg_saved_tensors_cpu_offload_auto_max_batch_size=pg_saved_tensors_cpu_offload_auto_max_batch_size,
                                            pg_saved_tensors_cpu_offload_auto_max_n_samples=pg_saved_tensors_cpu_offload_auto_max_n_samples,
                                        pg_tbptt_window=current_tbptt_window,
                                        tbptt_loss_sink=None,
                                        anil_inner_steps=anil_inner_steps,
                                        anil_inner_learning_rate=anil_inner_learning_rate,
                                        h_list_override=warmup_h_list_override,
                                        env_seeds_override=warmup_env_seeds_override,
                                        rl_objective=rl_objective,
                                        )
                                        del _warmup_loss
                            elif bool(policy_fastpath_compile_active) and callable(policy_fastpath_warmup_fn):
                                for _ in range(int(compile_warmup_steps)):
                                    with _amp_autocast_context(
                                        scaler,
                                        device=device,
                                        dtype=policy_autocast_dtype,
                                    ):
                                        warmup_ok = bool(
                                            policy_fastpath_warmup_fn(batch_size=warmup_chunk_bs)
                                        ) and bool(warmup_ok)
                            else:
                                warmup_ok = False
                            if device_obj.type == "cuda" and torch.cuda.is_available():
                                torch.cuda.synchronize(device_obj)
                        except Exception as warmup_err:
                            warmup_ok = False
                            print(
                                "[pg-compile-warn] compile warmup skipped due to runtime error: "
                                f"{warmup_err}"
                            )
                        finally:
                            _restore_rng_state(warmup_rng_state)
                            optimizer.zero_grad(set_to_none=True)
                            _clear_policy_rollout_artifacts(env_prior=env_prior, policy_step_fn=policy_step_fn)
                            if callable(policy_step_profile_consumer):
                                try:
                                    policy_step_profile_consumer()
                                except Exception:
                                    pass
                        compile_warmup_done = True
                        warmup_dt = float(time.perf_counter() - warmup_t0)
                        batch_compile_warmup_wall = float(warmup_dt)
                        batch_compile_warmup_ok = bool(warmup_ok)
                        if compile_observe_enabled and compile_observe_reset_after_warmup:
                            compile_counter_prev_snapshot = _snapshot_dynamo_counters()
                        print(
                            "[pg-compile-warmup] "
                            f"ok={int(bool(warmup_ok))} steps={int(compile_warmup_steps)} "
                            f"chunk={int(warmup_chunk_bs)} n_samples={int(warmup_n_samples)} "
                            f"wall_s={warmup_dt:.3f}"
                        )
                        if compile_observe_enabled and compile_observe_reset_after_warmup:
                            print("[pg-compile-observe] baseline reset after compile warmup.")
    
                    batch_oom = False
                    for chunk_start in range(0, batch_size, current_rollout_chunk_size):
                        chunk_bs = min(current_rollout_chunk_size, batch_size - chunk_start)
                        chunk_weight = float(chunk_bs) / float(batch_size)
                        chunk_h_list_override = None
                        chunk_env_seeds_override = None
                        if batch_h_list_override is not None:
                            chunk_h_list_override = batch_h_list_override[chunk_start: chunk_start + chunk_bs]
                        if batch_env_seeds_override is not None:
                            chunk_env_seeds_override = batch_env_seeds_override[chunk_start: chunk_start + chunk_bs]
                        tbptt_stream_backward_active = (
                            current_tbptt_window is not None and int(current_tbptt_window) > 0
                        )
                        tbptt_stream_merge_windows = 1
                        if tbptt_stream_backward_active:
                            batch_tbptt_stream_backward_active = True
                            tbptt_stream_merge_windows = int(max(1, current_tbptt_stream_merge_windows))
                            if (
                                tbptt_merge_auto_enabled
                                and device_obj.type == "cuda"
                                and torch.cuda.is_available()
                            ):
                                auto_target = int(min(
                                    max(1, tbptt_stream_merge_windows),
                                    max(1, tbptt_merge_auto_max_windows),
                                ))
                                if auto_target > 1:
                                    try:
                                        mem_reserved_now = int(torch.cuda.memory_reserved(device_obj))
                                        mem_total_now = int(torch.cuda.get_device_properties(device_obj).total_memory)
                                    except Exception:
                                        mem_reserved_now = 0
                                        mem_total_now = 0
                                    free_now = int(max(0, mem_total_now - mem_reserved_now))
                                    allow_auto_merge = bool(
                                        (mem_total_now > 0)
                                        and (free_now >= int(tbptt_merge_auto_min_free_bytes))
                                        and (
                                            mem_reserved_now
                                            <= int(float(mem_total_now) * float(tbptt_merge_auto_reserved_frac))
                                        )
                                    )
                                    tbptt_stream_merge_windows = int(auto_target if allow_auto_merge else 1)
                            if (
                                int(tbptt_stream_merge_windows) > 1
                                and tbptt_merge_guard_enabled
                                and device_obj.type == "cuda"
                                and torch.cuda.is_available()
                            ):
                                try:
                                    mem_reserved_now = int(torch.cuda.memory_reserved(device_obj))
                                    mem_total_now = int(torch.cuda.get_device_properties(device_obj).total_memory)
                                except Exception:
                                    mem_reserved_now = 0
                                    mem_total_now = 0
                                free_now = int(max(0, mem_total_now - mem_reserved_now))
                                allow_merge_prefight = bool(
                                    (mem_total_now > 0)
                                    and (free_now >= int(tbptt_merge_guard_min_free_bytes))
                                    and (
                                        mem_reserved_now
                                        <= int(float(mem_total_now) * float(tbptt_merge_guard_reserved_frac))
                                    )
                                )
                                if not allow_merge_prefight:
                                    tbptt_stream_merge_windows = 1
                                    batch_tbptt_stream_merge_guard_prefallbacks += 1
                            batch_tbptt_stream_merge_windows = max(
                                int(batch_tbptt_stream_merge_windows),
                                int(tbptt_stream_merge_windows),
                            )
                        tbptt_window_backward_called = False
                        tbptt_pending_window_losses = (
                            []
                            if tbptt_stream_backward_active and int(tbptt_stream_merge_windows) > 1
                            else None
                        )
    
                        backward_scale = float(chunk_weight) / float(aggregate_k_gradients)
                        reinforce_replay_backward_called = False
                        anil_query_backward_called = False

                        if tbptt_stream_backward_active:
                            tbptt_merge_guard_active = bool(
                                (tbptt_pending_window_losses is not None)
                                and tbptt_merge_guard_enabled
                                and device_obj.type == "cuda"
                                and torch.cuda.is_available()
                            )

                            def _tbptt_merge_guard_should_flush_pending():
                                if (
                                    (not tbptt_merge_guard_active)
                                    or tbptt_pending_window_losses is None
                                    or (len(tbptt_pending_window_losses) <= 0)
                                ):
                                    return False
                                try:
                                    mem_reserved_now = int(torch.cuda.memory_reserved(device_obj))
                                    mem_total_now = int(torch.cuda.get_device_properties(device_obj).total_memory)
                                except Exception:
                                    return False
                                if mem_total_now <= 0:
                                    return False
                                free_now = int(max(0, mem_total_now - mem_reserved_now))
                                return bool(
                                    (free_now < int(tbptt_merge_guard_min_free_bytes))
                                    or (
                                        mem_reserved_now
                                        > int(float(mem_total_now) * float(tbptt_merge_guard_reserved_frac))
                                    )
                                )

                            def _tbptt_backward_from_losses(window_losses_scaled, *, retain_graph=False):
                                nonlocal tbptt_window_backward_called, batch_backward_wall, batch_backward_calls
                                nonlocal backward_t_start_unix, backward_t_end_unix
                                nonlocal batch_tbptt_stream_backward_launches, batch_tbptt_stream_backward_roots
                                nonlocal batch_tbptt_backward_mem_guard_hits
                                if window_losses_scaled is None:
                                    return
                                num_roots = int(len(window_losses_scaled))
                                if num_roots <= 0:
                                    return
                                if (
                                    pg_mem_guard_enabled
                                    and ("cuda" in str(device))
                                    and torch.cuda.is_available()
                                ):
                                    try:
                                        mem_alloc_now = int(torch.cuda.memory_allocated(device_obj))
                                        mem_reserved_now = int(torch.cuda.memory_reserved(device_obj))
                                        mem_total_now = int(torch.cuda.get_device_properties(device_obj).total_memory)
                                    except Exception:
                                        mem_alloc_now = 0
                                        mem_reserved_now = 0
                                        mem_total_now = 0
                                    slack_bytes_now = int(max(0, mem_reserved_now - mem_alloc_now))
                                    need_cleanup_now = False
                                    if (
                                        mem_total_now > 0
                                        and mem_reserved_now
                                        >= int(float(mem_total_now) * pg_mem_guard_reserved_frac)
                                    ):
                                        need_cleanup_now = True
                                    if slack_bytes_now >= pg_mem_guard_slack_bytes:
                                        need_cleanup_now = True
                                    if need_cleanup_now:
                                        gc.collect()
                                        torch.cuda.empty_cache()
                                        batch_tbptt_backward_mem_guard_hits += 1
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
                                    applied_scale = 1.0
                                    if scaler is None:
                                        if num_roots == 1:
                                            window_losses_scaled[0].backward(retain_graph=bool(retain_graph))
                                        else:
                                            torch.autograd.backward(
                                                window_losses_scaled,
                                                retain_graph=bool(retain_graph),
                                            )
                                    else:
                                        applied_scale = float(scaler.get_scale())
                                        scaled_losses = [scaler.scale(loss_root) for loss_root in window_losses_scaled]
                                        if num_roots == 1:
                                            scaled_losses[0].backward(retain_graph=bool(retain_graph))
                                        else:
                                            torch.autograd.backward(
                                                scaled_losses,
                                                retain_graph=bool(retain_graph),
                                            )
                                if backward_cuda_start is not None:
                                    backward_cuda_end = torch.cuda.Event(enable_timing=True)
                                    backward_cuda_end.record()
                                    backward_cuda_pairs.append((backward_cuda_start, backward_cuda_end))
                                batch_backward_wall += (time.perf_counter() - backward_t0)
                                batch_backward_calls += 1
                                batch_tbptt_stream_backward_launches += 1
                                batch_tbptt_stream_backward_roots += int(num_roots)
                                backward_t1_unix = time.time()
                                if backward_t_start_unix is None:
                                    backward_t_start_unix = backward_t0_unix
                                backward_t_end_unix = backward_t1_unix
                                tbptt_window_backward_called = True
                                return float(applied_scale)

                            def _tbptt_flush_pending_windows():
                                if tbptt_pending_window_losses is None or (len(tbptt_pending_window_losses) == 0):
                                    return
                                merged_losses = list(tbptt_pending_window_losses)
                                tbptt_pending_window_losses.clear()
                                _tbptt_backward_from_losses(merged_losses)

                            def _tbptt_chunk_loss_sink(weighted_window_loss):
                                nonlocal batch_tbptt_stream_merge_guard_flushes
                                payload = None
                                loss_root = weighted_window_loss
                                retain_graph = False
                                immediate = False
                                if isinstance(weighted_window_loss, dict):
                                    payload = weighted_window_loss
                                    loss_root = weighted_window_loss.get("loss_root", None)
                                    retain_graph = bool(weighted_window_loss.get("retain_graph", False))
                                    immediate = bool(weighted_window_loss.get("immediate", False))
                                if (not torch.is_tensor(loss_root)) or (not bool(loss_root.requires_grad)):
                                    return None
                                window_loss_scaled = loss_root * backward_scale
                                if immediate:
                                    _tbptt_flush_pending_windows()
                                    applied_scale = _tbptt_backward_from_losses(
                                        [window_loss_scaled],
                                        retain_graph=bool(retain_graph),
                                    )
                                    if isinstance(payload, dict):
                                        return {"applied_scale": float(applied_scale)}
                                    return None
                                if tbptt_pending_window_losses is not None:
                                    if _tbptt_merge_guard_should_flush_pending():
                                        batch_tbptt_stream_merge_guard_flushes += 1
                                        _tbptt_flush_pending_windows()
                                    tbptt_pending_window_losses.append(window_loss_scaled)
                                    if len(tbptt_pending_window_losses) < int(tbptt_stream_merge_windows):
                                        return
                                    _tbptt_flush_pending_windows()
                                    return None
                                _tbptt_backward_from_losses([window_loss_scaled])
                                return None
                            _tbptt_chunk_loss_sink._ticl_accepts_tbptt_payload = True
                        else:
                            _tbptt_chunk_loss_sink = None
                            _tbptt_flush_pending_windows = None

                        if (not tbptt_stream_backward_active) and (current_tbptt_window is None):
                            def _reinforce_replay_loss_sink(loss_root):
                                nonlocal reinforce_replay_backward_called
                                nonlocal batch_backward_wall, batch_backward_calls
                                nonlocal backward_t_start_unix, backward_t_end_unix
                                retain_graph = False
                                if isinstance(loss_root, dict):
                                    retain_graph = bool(loss_root.get("retain_graph", False))
                                    loss_root = loss_root.get("loss", None)
                                if (not torch.is_tensor(loss_root)) or (not bool(loss_root.requires_grad)):
                                    return None
                                reinforce_replay_backward_called = True
                                scaled_loss = loss_root * backward_scale
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
                                        scaled_loss.backward(retain_graph=bool(retain_graph))
                                    else:
                                        scaler.scale(scaled_loss).backward(retain_graph=bool(retain_graph))
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
                                return None
                            _reinforce_replay_loss_sink._ticl_accepts_replay_payload = True
                        else:
                            _reinforce_replay_loss_sink = None

                        if (not tbptt_stream_backward_active) and (current_tbptt_window is None) and rl_objective == "anil":
                            def _anil_query_loss_sink(loss_root):
                                nonlocal anil_query_backward_called
                                nonlocal batch_backward_wall, batch_backward_calls
                                nonlocal backward_t_start_unix, backward_t_end_unix
                                if (not torch.is_tensor(loss_root)) or (not bool(loss_root.requires_grad)):
                                    return None
                                anil_query_backward_called = True
                                scaled_loss = loss_root * backward_scale
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
                                        scaled_loss.backward()
                                    else:
                                        scaler.scale(scaled_loss).backward()
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
                                return None
                        else:
                            _anil_query_loss_sink = None
    
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
                                    rollout_override_kwargs = {}
                                    if chunk_h_list_override is not None:
                                        rollout_override_kwargs["h_list_override"] = chunk_h_list_override
                                    if chunk_env_seeds_override is not None:
                                        rollout_override_kwargs["env_seeds_override"] = chunk_env_seeds_override
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
                                        pg_saved_tensors_cpu_offload_scope=pg_saved_tensors_cpu_offload_scope,
                                        pg_saved_tensors_pin_memory=pg_saved_tensors_pin_memory,
                                        pg_saved_tensors_cpu_offload_auto_disable_when_safe=pg_saved_tensors_cpu_offload_auto_disable_when_safe,
                                        pg_saved_tensors_cpu_offload_auto_min_free_gb=pg_saved_tensors_cpu_offload_auto_min_free_gb,
                                        pg_saved_tensors_cpu_offload_auto_max_batch_size=pg_saved_tensors_cpu_offload_auto_max_batch_size,
                                        pg_saved_tensors_cpu_offload_auto_max_n_samples=pg_saved_tensors_cpu_offload_auto_max_n_samples,
                                        pg_tbptt_window=current_tbptt_window,
                                        tbptt_loss_sink=_tbptt_chunk_loss_sink,
                                        reinforce_replay_loss_sink=_reinforce_replay_loss_sink,
                                        anil_inner_steps=anil_inner_steps,
                                        anil_inner_learning_rate=anil_inner_learning_rate,
                                        anil_query_loss_sink=_anil_query_loss_sink,
                                        rl_objective=rl_objective,
                                        **rollout_override_kwargs,
                                    )
                                    chunk_terminal_stats = getattr(env_prior, "last_rollout_terminal_stats", None)
                                    weighted_pg_loss = pg_loss_chunk * chunk_weight
                                    loss = weighted_pg_loss / aggregate_k_gradients
                                    if tbptt_stream_backward_active and tbptt_window_backward_called:
                                        loss = loss.detach()
                                    if reinforce_replay_backward_called:
                                        loss = loss.detach()
                                    if anil_query_backward_called:
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
                            if tbptt_stream_backward_active and callable(_tbptt_flush_pending_windows):
                                _tbptt_flush_pending_windows()
    
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
                        except KeyboardInterrupt:
                            traceback.print_exc()
                            raise
                        except Exception as e:
                            if not _is_oom_exception(e):
                                raise
                            if bool(pg_oom_fail_fast):
                                print(
                                    f"[pg-oom-fail-fast] epoch={batch_epoch_for_profile} batch={batch} "
                                    f"chunk={current_rollout_chunk_size} tbptt={current_tbptt_window} "
                                    "raising immediately (fallback disabled)."
                                )
                                raise
                            if bool(pg_oom_debug_raise):
                                print(
                                    "[pg-oom-debug] re-raising OOM for full traceback "
                                    "(set --pg-oom-debug-raise false to enable auto chunk fallback)."
                                )
                                raise
                            if tbptt_stream_backward_active and int(current_tbptt_stream_merge_windows) > 1:
                                current_tbptt_stream_merge_windows = int(
                                    max(1, int(current_tbptt_stream_merge_windows) // 2)
                                )
                                batch_oom = True
                                if "cuda" in str(device):
                                    torch.cuda.empty_cache()
                                gc.collect()
                                print(
                                    f"[pg-oom] epoch={batch_epoch_for_profile} batch={batch} "
                                    f"reducing TBPTT stream merge windows to {int(current_tbptt_stream_merge_windows)} "
                                    f"(chunk={current_rollout_chunk_size}, tbptt={current_tbptt_window})"
                                )
                                break
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
                        chunk_reward_env_mean = pg_stats_chunk.get("reward_env_mean", None)
                        if chunk_reward_env_mean is not None:
                            try:
                                contrib = float(torch.as_tensor(chunk_reward_env_mean).detach().cpu()) * chunk_weight
                                if batch_reward_env_mean_value is None:
                                    batch_reward_env_mean_value = contrib
                                else:
                                    batch_reward_env_mean_value += contrib
                            except Exception:
                                pass
                        chunk_reward_env_std = pg_stats_chunk.get("reward_env_std", None)
                        if chunk_reward_env_std is not None:
                            try:
                                contrib = float(torch.as_tensor(chunk_reward_env_std).detach().cpu()) * chunk_weight
                                if batch_reward_env_std_value is None:
                                    batch_reward_env_std_value = contrib
                                else:
                                    batch_reward_env_std_value += contrib
                            except Exception:
                                pass
                        chunk_reward_env_return = pg_stats_chunk.get("reward_env_return_mean", None)
                        if chunk_reward_env_return is not None:
                            try:
                                contrib = float(torch.as_tensor(chunk_reward_env_return).detach().cpu()) * chunk_weight
                                if batch_reward_env_return_value is None:
                                    batch_reward_env_return_value = contrib
                                else:
                                    batch_reward_env_return_value += contrib
                            except Exception:
                                pass
                        chunk_reward_ctrl_mean = pg_stats_chunk.get("reward_ctrl_mean", None)
                        if chunk_reward_ctrl_mean is not None:
                            try:
                                contrib = float(torch.as_tensor(chunk_reward_ctrl_mean).detach().cpu()) * chunk_weight
                                if batch_reward_ctrl_mean_value is None:
                                    batch_reward_ctrl_mean_value = contrib
                                else:
                                    batch_reward_ctrl_mean_value += contrib
                            except Exception:
                                pass
                        chunk_reward_ctrl_std = pg_stats_chunk.get("reward_ctrl_std", None)
                        if chunk_reward_ctrl_std is not None:
                            try:
                                contrib = float(torch.as_tensor(chunk_reward_ctrl_std).detach().cpu()) * chunk_weight
                                if batch_reward_ctrl_std_value is None:
                                    batch_reward_ctrl_std_value = contrib
                                else:
                                    batch_reward_ctrl_std_value += contrib
                            except Exception:
                                pass
                        chunk_reward_ctrl_return = pg_stats_chunk.get("reward_ctrl_return_mean", None)
                        if chunk_reward_ctrl_return is not None:
                            try:
                                contrib = float(torch.as_tensor(chunk_reward_ctrl_return).detach().cpu()) * chunk_weight
                                if batch_reward_ctrl_return_value is None:
                                    batch_reward_ctrl_return_value = contrib
                                else:
                                    batch_reward_ctrl_return_value += contrib
                            except Exception:
                                pass
                        chunk_policy_total_loss = pg_stats_chunk.get("policy_total_loss", None)
                        if chunk_policy_total_loss is not None:
                            try:
                                contrib = (
                                    float(torch.as_tensor(chunk_policy_total_loss).detach().cpu())
                                    * chunk_weight
                                    / float(aggregate_k_gradients)
                                )
                                if batch_policy_total_loss_value is None:
                                    batch_policy_total_loss_value = contrib
                                else:
                                    batch_policy_total_loss_value += contrib
                            except Exception:
                                pass
                        chunk_reinforce_loss = pg_stats_chunk.get("reinforce_loss", None)
                        if chunk_reinforce_loss is not None:
                            try:
                                contrib = (
                                    float(torch.as_tensor(chunk_reinforce_loss).detach().cpu())
                                    * chunk_weight
                                    / float(aggregate_k_gradients)
                                )
                                if batch_reinforce_loss_value is None:
                                    batch_reinforce_loss_value = contrib
                                else:
                                    batch_reinforce_loss_value += contrib
                            except Exception:
                                pass
                        chunk_normalized_q_value_loss = pg_stats_chunk.get("normalized_q_value_loss", None)
                        if chunk_normalized_q_value_loss is not None:
                            try:
                                contrib = (
                                    float(torch.as_tensor(chunk_normalized_q_value_loss).detach().cpu())
                                    * chunk_weight
                                    / float(aggregate_k_gradients)
                                )
                                if batch_normalized_q_value_loss_value is None:
                                    batch_normalized_q_value_loss_value = contrib
                                else:
                                    batch_normalized_q_value_loss_value += contrib
                            except Exception:
                                pass
                        chunk_next_state_flow_matching_loss = pg_stats_chunk.get("next_state_flow_matching_loss", None)
                        if chunk_next_state_flow_matching_loss is not None:
                            try:
                                contrib = (
                                    float(torch.as_tensor(chunk_next_state_flow_matching_loss).detach().cpu())
                                    * chunk_weight
                                    / float(aggregate_k_gradients)
                                )
                                if batch_next_state_flow_matching_loss_value is None:
                                    batch_next_state_flow_matching_loss_value = contrib
                                else:
                                    batch_next_state_flow_matching_loss_value += contrib
                            except Exception:
                                pass
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
                        chunk_reward_nonfinite_share = pg_stats_chunk.get("reward_nonfinite_share", None)
                        if chunk_reward_nonfinite_share is not None:
                            try:
                                batch_reward_nonfinite_share += (
                                    float(chunk_reward_nonfinite_share.detach().cpu()) * chunk_weight
                                )
                            except Exception:
                                pass
                        chunk_reward_nan_share = pg_stats_chunk.get("reward_nan_share", None)
                        if chunk_reward_nan_share is not None:
                            try:
                                batch_reward_nan_share += float(chunk_reward_nan_share.detach().cpu()) * chunk_weight
                            except Exception:
                                pass
                        chunk_reward_inf_share = pg_stats_chunk.get("reward_inf_share", None)
                        if chunk_reward_inf_share is not None:
                            try:
                                batch_reward_inf_share += float(chunk_reward_inf_share.detach().cpu()) * chunk_weight
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
                        if chunk_terminal_stats is not None:
                            try:
                                chunk_terminal_mean = chunk_terminal_stats.get("terminal_count_mean", None)
                                if chunk_terminal_mean is not None:
                                    contrib = float(torch.as_tensor(chunk_terminal_mean).detach().cpu()) * chunk_weight
                                    if batch_terminal_count_mean_value is None:
                                        batch_terminal_count_mean_value = contrib
                                    else:
                                        batch_terminal_count_mean_value += contrib
                                chunk_terminal_min = chunk_terminal_stats.get("terminal_count_min", None)
                                if chunk_terminal_min is not None:
                                    value = float(torch.as_tensor(chunk_terminal_min).detach().cpu())
                                    batch_terminal_count_min_value = (
                                        value
                                        if batch_terminal_count_min_value is None
                                        else min(batch_terminal_count_min_value, value)
                                    )
                                chunk_terminal_max = chunk_terminal_stats.get("terminal_count_max", None)
                                if chunk_terminal_max is not None:
                                    value = float(torch.as_tensor(chunk_terminal_max).detach().cpu())
                                    batch_terminal_count_max_value = (
                                        value
                                        if batch_terminal_count_max_value is None
                                        else max(batch_terminal_count_max_value, value)
                                    )
                                chunk_terminal_target_mean = chunk_terminal_stats.get("terminal_count_target_mean", None)
                                if chunk_terminal_target_mean is not None:
                                    contrib = float(torch.as_tensor(chunk_terminal_target_mean).detach().cpu()) * chunk_weight
                                    if batch_terminal_target_mean_value is None:
                                        batch_terminal_target_mean_value = contrib
                                    else:
                                        batch_terminal_target_mean_value += contrib
                                chunk_terminal_target_min = chunk_terminal_stats.get("terminal_count_target_min", None)
                                if chunk_terminal_target_min is not None:
                                    value = float(torch.as_tensor(chunk_terminal_target_min).detach().cpu())
                                    batch_terminal_target_min_value = (
                                        value
                                        if batch_terminal_target_min_value is None
                                        else min(batch_terminal_target_min_value, value)
                                    )
                                chunk_terminal_target_max = chunk_terminal_stats.get("terminal_count_target_max", None)
                                if chunk_terminal_target_max is not None:
                                    value = float(torch.as_tensor(chunk_terminal_target_max).detach().cpu())
                                    batch_terminal_target_max_value = (
                                        value
                                        if batch_terminal_target_max_value is None
                                        else max(batch_terminal_target_max_value, value)
                                    )
                            except Exception:
                                pass
                        chunk_terminal_bonus_mean = pg_stats_chunk.get("terminal_bonus_mean", None)
                        if chunk_terminal_bonus_mean is not None:
                            try:
                                contrib = float(torch.as_tensor(chunk_terminal_bonus_mean).detach().cpu()) * chunk_weight
                                if batch_terminal_bonus_mean_value is None:
                                    batch_terminal_bonus_mean_value = contrib
                                else:
                                    batch_terminal_bonus_mean_value += contrib
                            except Exception:
                                pass
                        chunk_terminal_bonus_min = pg_stats_chunk.get("terminal_bonus_min", None)
                        if chunk_terminal_bonus_min is not None:
                            try:
                                value = float(torch.as_tensor(chunk_terminal_bonus_min).detach().cpu())
                                batch_terminal_bonus_min_value = (
                                    value
                                    if batch_terminal_bonus_min_value is None
                                    else min(batch_terminal_bonus_min_value, value)
                                )
                            except Exception:
                                pass
                        chunk_terminal_bonus_max = pg_stats_chunk.get("terminal_bonus_max", None)
                        if chunk_terminal_bonus_max is not None:
                            try:
                                value = float(torch.as_tensor(chunk_terminal_bonus_max).detach().cpu())
                                batch_terminal_bonus_max_value = (
                                    value
                                    if batch_terminal_bonus_max_value is None
                                    else max(batch_terminal_bonus_max_value, value)
                                )
                            except Exception:
                                pass
                        chunk_terminal_bonus_event_mean = pg_stats_chunk.get("terminal_bonus_event_mean", None)
                        if chunk_terminal_bonus_event_mean is not None:
                            try:
                                contrib = float(torch.as_tensor(chunk_terminal_bonus_event_mean).detach().cpu()) * chunk_weight
                                if batch_terminal_bonus_event_mean_value is None:
                                    batch_terminal_bonus_event_mean_value = contrib
                                else:
                                    batch_terminal_bonus_event_mean_value += contrib
                            except Exception:
                                pass
                        chunk_bridge_replay_count = pg_stats_chunk.get("pg_bridge_replay_count", None)
                        if chunk_bridge_replay_count is not None:
                            try:
                                batch_pg_bridge_replay_count_value += int(
                                    torch.as_tensor(chunk_bridge_replay_count).detach().cpu().item()
                                )
                            except Exception:
                                pass
                        chunk_reinforce_return_nonfinite = pg_stats_chunk.get("reinforce_return_nonfinite_share", None)
                        if chunk_reinforce_return_nonfinite is not None:
                            try:
                                batch_reinforce_return_nonfinite_share += (
                                    float(chunk_reinforce_return_nonfinite.detach().cpu()) * chunk_weight
                                )
                            except Exception:
                                pass
                        chunk_reinforce_log_prob_nonfinite = pg_stats_chunk.get(
                            "reinforce_log_prob_nonfinite_share", None
                        )
                        if chunk_reinforce_log_prob_nonfinite is not None:
                            try:
                                batch_reinforce_log_prob_nonfinite_share += (
                                    float(chunk_reinforce_log_prob_nonfinite.detach().cpu()) * chunk_weight
                                )
                            except Exception:
                                pass
                        chunk_reinforce_adv_nonfinite = pg_stats_chunk.get("reinforce_adv_nonfinite_share", None)
                        if chunk_reinforce_adv_nonfinite is not None:
                            try:
                                batch_reinforce_adv_nonfinite_share += (
                                    float(chunk_reinforce_adv_nonfinite.detach().cpu()) * chunk_weight
                                )
                            except Exception:
                                pass
                        chunk_reinforce_log_prob_mean = pg_stats_chunk.get("reinforce_log_prob_mean", None)
                        if chunk_reinforce_log_prob_mean is not None:
                            try:
                                contrib = float(chunk_reinforce_log_prob_mean.detach().cpu()) * chunk_weight
                                if batch_reinforce_log_prob_mean_value is None:
                                    batch_reinforce_log_prob_mean_value = contrib
                                else:
                                    batch_reinforce_log_prob_mean_value += contrib
                            except Exception:
                                pass
                        chunk_reinforce_log_prob_std = pg_stats_chunk.get("reinforce_log_prob_std", None)
                        if chunk_reinforce_log_prob_std is not None:
                            try:
                                contrib = float(chunk_reinforce_log_prob_std.detach().cpu()) * chunk_weight
                                if batch_reinforce_log_prob_std_value is None:
                                    batch_reinforce_log_prob_std_value = contrib
                                else:
                                    batch_reinforce_log_prob_std_value += contrib
                            except Exception:
                                pass
                        chunk_reinforce_action_dim_mean = pg_stats_chunk.get("reinforce_action_dim_mean", None)
                        if chunk_reinforce_action_dim_mean is not None:
                            try:
                                contrib = float(torch.as_tensor(chunk_reinforce_action_dim_mean).detach().cpu()) * chunk_weight
                                if batch_reinforce_action_dim_mean_value is None:
                                    batch_reinforce_action_dim_mean_value = contrib
                                else:
                                    batch_reinforce_action_dim_mean_value += contrib
                            except Exception:
                                pass
                        chunk_reinforce_action_dim_min = pg_stats_chunk.get("reinforce_action_dim_min", None)
                        if chunk_reinforce_action_dim_min is not None:
                            try:
                                batch_reinforce_action_dim_min_value = min(
                                    batch_reinforce_action_dim_min_value,
                                    float(torch.as_tensor(chunk_reinforce_action_dim_min).detach().cpu()),
                                )
                            except Exception:
                                pass
                        chunk_reinforce_action_dim_max = pg_stats_chunk.get("reinforce_action_dim_max", None)
                        if chunk_reinforce_action_dim_max is not None:
                            try:
                                batch_reinforce_action_dim_max_value = max(
                                    batch_reinforce_action_dim_max_value,
                                    float(torch.as_tensor(chunk_reinforce_action_dim_max).detach().cpu()),
                                )
                            except Exception:
                                pass
                        chunk_reinforce_action_std_mean = pg_stats_chunk.get("reinforce_action_std_mean", None)
                        if chunk_reinforce_action_std_mean is not None:
                            try:
                                contrib = float(torch.as_tensor(chunk_reinforce_action_std_mean).detach().cpu()) * chunk_weight
                                if batch_reinforce_action_std_mean_value is None:
                                    batch_reinforce_action_std_mean_value = contrib
                                else:
                                    batch_reinforce_action_std_mean_value += contrib
                            except Exception:
                                pass
                        chunk_reinforce_action_std_min = pg_stats_chunk.get("reinforce_action_std_min", None)
                        if chunk_reinforce_action_std_min is not None:
                            try:
                                batch_reinforce_action_std_min_value = min(
                                    batch_reinforce_action_std_min_value,
                                    float(torch.as_tensor(chunk_reinforce_action_std_min).detach().cpu()),
                                )
                            except Exception:
                                pass
                        chunk_reinforce_action_std_max = pg_stats_chunk.get("reinforce_action_std_max", None)
                        if chunk_reinforce_action_std_max is not None:
                            try:
                                batch_reinforce_action_std_max_value = max(
                                    batch_reinforce_action_std_max_value,
                                    float(torch.as_tensor(chunk_reinforce_action_std_max).detach().cpu()),
                                )
                            except Exception:
                                pass
                        chunk_reinforce_logprob_log_std_mean = pg_stats_chunk.get("reinforce_logprob_log_std_mean", None)
                        if chunk_reinforce_logprob_log_std_mean is not None:
                            try:
                                contrib = float(torch.as_tensor(chunk_reinforce_logprob_log_std_mean).detach().cpu()) * chunk_weight
                                if batch_reinforce_logprob_log_std_mean_value is None:
                                    batch_reinforce_logprob_log_std_mean_value = contrib
                                else:
                                    batch_reinforce_logprob_log_std_mean_value += contrib
                            except Exception:
                                pass
                        chunk_reinforce_logprob_log_std_min = pg_stats_chunk.get("reinforce_logprob_log_std_min", None)
                        if chunk_reinforce_logprob_log_std_min is not None:
                            try:
                                batch_reinforce_logprob_log_std_min_value = min(
                                    batch_reinforce_logprob_log_std_min_value,
                                    float(torch.as_tensor(chunk_reinforce_logprob_log_std_min).detach().cpu()),
                                )
                            except Exception:
                                pass
                        chunk_reinforce_logprob_log_std_max = pg_stats_chunk.get("reinforce_logprob_log_std_max", None)
                        if chunk_reinforce_logprob_log_std_max is not None:
                            try:
                                batch_reinforce_logprob_log_std_max_value = max(
                                    batch_reinforce_logprob_log_std_max_value,
                                    float(torch.as_tensor(chunk_reinforce_logprob_log_std_max).detach().cpu()),
                                )
                            except Exception:
                                pass
                        chunk_reinforce_logprob_z2_mean = pg_stats_chunk.get("reinforce_logprob_z2_mean", None)
                        if chunk_reinforce_logprob_z2_mean is not None:
                            try:
                                contrib = float(torch.as_tensor(chunk_reinforce_logprob_z2_mean).detach().cpu()) * chunk_weight
                                if batch_reinforce_logprob_z2_mean_value is None:
                                    batch_reinforce_logprob_z2_mean_value = contrib
                                else:
                                    batch_reinforce_logprob_z2_mean_value += contrib
                            except Exception:
                                pass
                        chunk_reinforce_logprob_z2_max = pg_stats_chunk.get("reinforce_logprob_z2_max", None)
                        if chunk_reinforce_logprob_z2_max is not None:
                            try:
                                batch_reinforce_logprob_z2_max_value = max(
                                    batch_reinforce_logprob_z2_max_value,
                                    float(torch.as_tensor(chunk_reinforce_logprob_z2_max).detach().cpu()),
                                )
                            except Exception:
                                pass
                        chunk_state_abs_max = pg_stats_chunk.get("state_abs_max", None)
                        if chunk_state_abs_max is not None:
                            try:
                                batch_state_abs_max_value = max(
                                    batch_state_abs_max_value,
                                    float(chunk_state_abs_max.detach().cpu()),
                                )
                            except Exception:
                                pass
                        chunk_action_std_min = pg_stats_chunk.get("action_std_min", None)
                        if chunk_action_std_min is not None:
                            try:
                                batch_action_std_min_value = min(
                                    batch_action_std_min_value,
                                    float(chunk_action_std_min.detach().cpu()),
                                )
                            except Exception:
                                pass
                        chunk_action_std_max = pg_stats_chunk.get("action_std_max", None)
                        if chunk_action_std_max is not None:
                            try:
                                batch_action_std_max_value = max(
                                    batch_action_std_max_value,
                                    float(chunk_action_std_max.detach().cpu()),
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
                        gp_first_projection_wall_ms = pg_stats_chunk.get(
                            "rollout_transition_gp_first_projection_wall_ms", None
                        )
                        if gp_first_projection_wall_ms is not None:
                            try:
                                batch_rollout_transition_gp_first_projection_wall_ms += float(
                                    gp_first_projection_wall_ms
                                )
                            except Exception:
                                pass
                        gp_second_projection_wall_ms = pg_stats_chunk.get(
                            "rollout_transition_gp_second_projection_wall_ms", None
                        )
                        if gp_second_projection_wall_ms is not None:
                            try:
                                batch_rollout_transition_gp_second_projection_wall_ms += float(
                                    gp_second_projection_wall_ms
                                )
                            except Exception:
                                pass
                        gp_projection_call_count = pg_stats_chunk.get(
                            "rollout_transition_gp_projection_call_count", None
                        )
                        if gp_projection_call_count is not None:
                            try:
                                batch_rollout_transition_gp_projection_call_count += int(gp_projection_call_count)
                            except Exception:
                                pass
                        gp_rff_fused_call_count = pg_stats_chunk.get(
                            "rollout_transition_gp_rff_fused_call_count", None
                        )
                        if gp_rff_fused_call_count is not None:
                            try:
                                batch_rollout_transition_gp_rff_fused_call_count += int(gp_rff_fused_call_count)
                            except Exception:
                                pass
                        gp_profile_group_count = pg_stats_chunk.get(
                            "rollout_transition_gp_profile_group_count", None
                        )
                        if gp_profile_group_count is not None:
                            try:
                                batch_rollout_transition_gp_profile_group_count += int(gp_profile_group_count)
                            except Exception:
                                pass
                        gp_profile_sync_group_count = pg_stats_chunk.get(
                            "rollout_transition_gp_profile_sync_group_count", None
                        )
                        if gp_profile_sync_group_count is not None:
                            try:
                                batch_rollout_transition_gp_profile_sync_group_count += int(
                                    gp_profile_sync_group_count
                                )
                            except Exception:
                                pass
                        packed_env_input_group_count = pg_stats_chunk.get(
                            "rollout_transition_packed_env_input_group_count", None
                        )
                        if packed_env_input_group_count is not None:
                            try:
                                batch_rollout_transition_packed_env_input_group_count += int(
                                    packed_env_input_group_count
                                )
                            except Exception:
                                pass
                        packed_env_input_call_count = pg_stats_chunk.get(
                            "rollout_transition_packed_env_input_call_count", None
                        )
                        if packed_env_input_call_count is not None:
                            try:
                                batch_rollout_transition_packed_env_input_call_count += int(
                                    packed_env_input_call_count
                                )
                            except Exception:
                                pass
                        transition_only_build_group_count = pg_stats_chunk.get(
                            "rollout_transition_only_build_group_count", None
                        )
                        if transition_only_build_group_count is not None:
                            try:
                                batch_rollout_transition_only_build_group_count += int(
                                    transition_only_build_group_count
                                )
                            except Exception:
                                pass
                        transition_only_skipped_generator_count = pg_stats_chunk.get(
                            "rollout_transition_only_skipped_generator_count", None
                        )
                        if transition_only_skipped_generator_count is not None:
                            try:
                                batch_rollout_transition_only_skipped_generator_count += int(
                                    transition_only_skipped_generator_count
                                )
                            except Exception:
                                pass
                        transition_group_wall_ms = pg_stats_chunk.get("rollout_transition_group_wall_ms", None)
                        if transition_group_wall_ms is not None:
                            try:
                                batch_rollout_transition_group_wall_ms += float(transition_group_wall_ms)
                            except Exception:
                                pass
                        transition_group_launch_wall_ms = pg_stats_chunk.get(
                            "rollout_transition_group_launch_wall_ms", None
                        )
                        if transition_group_launch_wall_ms is not None:
                            try:
                                batch_rollout_transition_group_launch_wall_ms += float(transition_group_launch_wall_ms)
                            except Exception:
                                pass
                        transition_group_sync_wall_ms = pg_stats_chunk.get(
                            "rollout_transition_group_sync_wall_ms", None
                        )
                        if transition_group_sync_wall_ms is not None:
                            try:
                                batch_rollout_transition_group_sync_wall_ms += float(transition_group_sync_wall_ms)
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
                        transition_noise_wall_ms = pg_stats_chunk.get("rollout_transition_noise_wall_ms", None)
                        if transition_noise_wall_ms is not None:
                            try:
                                batch_rollout_transition_noise_wall_ms += float(transition_noise_wall_ms)
                            except Exception:
                                pass
                        transition_fused_wall_ms = pg_stats_chunk.get("rollout_transition_fused_wall_ms", None)
                        if transition_fused_wall_ms is not None:
                            try:
                                batch_rollout_transition_fused_wall_ms += float(transition_fused_wall_ms)
                            except Exception:
                                pass
                        transition_fused_launch_wall_ms = pg_stats_chunk.get(
                            "rollout_transition_fused_launch_wall_ms", None
                        )
                        if transition_fused_launch_wall_ms is not None:
                            try:
                                batch_rollout_transition_fused_launch_wall_ms += float(transition_fused_launch_wall_ms)
                            except Exception:
                                pass
                        transition_fused_call_count = pg_stats_chunk.get("rollout_transition_fused_call_count", None)
                        if transition_fused_call_count is not None:
                            try:
                                batch_rollout_transition_fused_call_count += int(transition_fused_call_count)
                            except Exception:
                                pass
                        transition_fused_group_count = pg_stats_chunk.get("rollout_transition_fused_group_count", None)
                        if transition_fused_group_count is not None:
                            try:
                                batch_rollout_transition_fused_group_count += int(transition_fused_group_count)
                            except Exception:
                                pass
                        transition_fused_enabled = pg_stats_chunk.get("rollout_transition_fused_enabled", None)
                        if transition_fused_enabled is not None:
                            try:
                                batch_rollout_transition_fused_enabled = int(
                                    max(
                                        int(batch_rollout_transition_fused_enabled),
                                        int(transition_fused_enabled),
                                    )
                                )
                            except Exception:
                                pass
                        transition_checkpoint_enabled = pg_stats_chunk.get(
                            "rollout_transition_checkpoint_enabled", None
                        )
                        if transition_checkpoint_enabled is not None:
                            try:
                                batch_rollout_transition_checkpoint_enabled = int(
                                    max(
                                        int(batch_rollout_transition_checkpoint_enabled),
                                        int(transition_checkpoint_enabled),
                                    )
                                )
                            except Exception:
                                pass
                        transition_checkpoint_call_count = pg_stats_chunk.get(
                            "rollout_transition_checkpoint_call_count", None
                        )
                        if transition_checkpoint_call_count is not None:
                            try:
                                batch_rollout_transition_checkpoint_call_count += int(
                                    transition_checkpoint_call_count
                                )
                            except Exception:
                                pass
                        transition_group_count = pg_stats_chunk.get("rollout_transition_group_count", None)
                        if transition_group_count is not None:
                            try:
                                batch_rollout_transition_group_count += int(transition_group_count)
                            except Exception:
                                pass
                        transition_family_group_count = pg_stats_chunk.get(
                            "rollout_transition_family_group_count", None
                        )
                        if transition_family_group_count is not None:
                            try:
                                batch_rollout_transition_family_group_count += int(
                                    transition_family_group_count
                                )
                            except Exception:
                                pass
                        transition_inner_grouping_structure_enabled = pg_stats_chunk.get(
                            "rollout_transition_inner_grouping_structure_enabled", None
                        )
                        if transition_inner_grouping_structure_enabled is not None:
                            try:
                                batch_rollout_transition_inner_grouping_structure_enabled = int(
                                    max(
                                        int(batch_rollout_transition_inner_grouping_structure_enabled),
                                        int(transition_inner_grouping_structure_enabled),
                                    )
                                )
                            except Exception:
                                pass
                        transition_inner_min_bucket = pg_stats_chunk.get(
                            "rollout_transition_inner_min_bucket", None
                        )
                        if transition_inner_min_bucket is not None:
                            try:
                                batch_rollout_transition_inner_min_bucket = int(
                                    max(
                                        int(batch_rollout_transition_inner_min_bucket),
                                        int(transition_inner_min_bucket),
                                    )
                                )
                            except Exception:
                                pass
                        transition_bucket_max_batch = pg_stats_chunk.get(
                            "rollout_transition_bucket_max_batch", None
                        )
                        if transition_bucket_max_batch is not None:
                            try:
                                batch_rollout_transition_bucket_max_batch = int(
                                    max(
                                        int(batch_rollout_transition_bucket_max_batch),
                                        int(transition_bucket_max_batch),
                                    )
                                )
                            except Exception:
                                pass
                        transition_work_actual_est = pg_stats_chunk.get(
                            "rollout_transition_work_actual_est", None
                        )
                        if transition_work_actual_est is not None:
                            try:
                                batch_rollout_transition_work_actual_est += float(
                                    transition_work_actual_est
                                )
                            except Exception:
                                pass
                        transition_work_padded_est = pg_stats_chunk.get(
                            "rollout_transition_work_padded_est", None
                        )
                        if transition_work_padded_est is not None:
                            try:
                                batch_rollout_transition_work_padded_est += float(
                                    transition_work_padded_est
                                )
                            except Exception:
                                pass
                        transition_async_enabled = pg_stats_chunk.get("rollout_transition_async_enabled", None)
                        if transition_async_enabled is not None:
                            try:
                                batch_rollout_transition_async_enabled = int(
                                    max(
                                        int(batch_rollout_transition_async_enabled),
                                        int(transition_async_enabled),
                                    )
                                )
                            except Exception:
                                pass
                        rollout_noise_mode = pg_stats_chunk.get("rollout_noise_mode", None)
                        if rollout_noise_mode is not None:
                            batch_rollout_noise_mode = str(rollout_noise_mode)
                        rollout_noise_block_size = pg_stats_chunk.get("rollout_noise_block_size", None)
                        if rollout_noise_block_size is not None:
                            try:
                                batch_rollout_noise_block_size = int(rollout_noise_block_size)
                            except Exception:
                                pass
                        rollout_env_count = pg_stats_chunk.get("rollout_env_count", None)
                        if rollout_env_count is not None:
                            try:
                                batch_rollout_env_count = int(
                                    max(int(batch_rollout_env_count), int(rollout_env_count))
                                )
                            except Exception:
                                pass
                        rollout_strict_joint_transition_count = pg_stats_chunk.get(
                            "rollout_strict_joint_transition_count", None
                        )
                        if rollout_strict_joint_transition_count is not None:
                            try:
                                batch_rollout_strict_joint_transition_count = int(
                                    max(
                                        int(batch_rollout_strict_joint_transition_count),
                                        int(rollout_strict_joint_transition_count),
                                    )
                                )
                            except Exception:
                                pass
                        rollout_reference_semantics_count = pg_stats_chunk.get(
                            "rollout_reference_semantics_count", None
                        )
                        if rollout_reference_semantics_count is not None:
                            try:
                                batch_rollout_reference_semantics_count = int(
                                    max(
                                        int(batch_rollout_reference_semantics_count),
                                        int(rollout_reference_semantics_count),
                                    )
                                )
                            except Exception:
                                pass
                        rollout_strict_joint_transition_share = pg_stats_chunk.get(
                            "rollout_strict_joint_transition_share", None
                        )
                        if rollout_strict_joint_transition_share is not None:
                            try:
                                batch_rollout_strict_joint_transition_share = float(
                                    max(
                                        float(batch_rollout_strict_joint_transition_share),
                                        float(rollout_strict_joint_transition_share),
                                    )
                                )
                            except Exception:
                                pass
                        rollout_reference_semantics_share = pg_stats_chunk.get(
                            "rollout_reference_semantics_share", None
                        )
                        if rollout_reference_semantics_share is not None:
                            try:
                                batch_rollout_reference_semantics_share = float(
                                    max(
                                        float(batch_rollout_reference_semantics_share),
                                        float(rollout_reference_semantics_share),
                                    )
                                )
                            except Exception:
                                pass
                        rollout_exact_scm_count = pg_stats_chunk.get("rollout_exact_scm_count", None)
                        if rollout_exact_scm_count is not None:
                            try:
                                batch_rollout_exact_scm_count = int(
                                    max(int(batch_rollout_exact_scm_count), int(rollout_exact_scm_count))
                                )
                            except Exception:
                                pass
                        rollout_exact_gp_count = pg_stats_chunk.get("rollout_exact_gp_count", None)
                        if rollout_exact_gp_count is not None:
                            try:
                                batch_rollout_exact_gp_count = int(
                                    max(int(batch_rollout_exact_gp_count), int(rollout_exact_gp_count))
                                )
                            except Exception:
                                pass
                        rollout_legacy_scm_count = pg_stats_chunk.get("rollout_legacy_scm_count", None)
                        if rollout_legacy_scm_count is not None:
                            try:
                                batch_rollout_legacy_scm_count = int(
                                    max(int(batch_rollout_legacy_scm_count), int(rollout_legacy_scm_count))
                                )
                            except Exception:
                                pass
                        rollout_legacy_gp_count = pg_stats_chunk.get("rollout_legacy_gp_count", None)
                        if rollout_legacy_gp_count is not None:
                            try:
                                batch_rollout_legacy_gp_count = int(
                                    max(int(batch_rollout_legacy_gp_count), int(rollout_legacy_gp_count))
                                )
                            except Exception:
                                pass
                        rollout_transition_reference_mode = pg_stats_chunk.get(
                            "rollout_transition_reference_mode",
                            None,
                        )
                        if rollout_transition_reference_mode is not None:
                            rollout_transition_reference_mode = str(rollout_transition_reference_mode)
                            if batch_rollout_transition_reference_mode is None:
                                batch_rollout_transition_reference_mode = rollout_transition_reference_mode
                            elif batch_rollout_transition_reference_mode != rollout_transition_reference_mode:
                                batch_rollout_transition_reference_mode = "mixed"
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
                                batch_policy_step_transformer_layer_finalize_attn_outproj_linear_ms += float(
                                    step_profile.get(
                                        "transformer_layer_finalize_attn_outproj_linear_wall_s", 0.0
                                    ) or 0.0
                                ) * 1000.0
                                batch_policy_step_transformer_layer_finalize_attn_outproj_norm_ms += float(
                                    step_profile.get(
                                        "transformer_layer_finalize_attn_outproj_norm_wall_s", 0.0
                                    ) or 0.0
                                ) * 1000.0
                                batch_policy_step_transformer_layer_finalize_ffn_ms += float(
                                    step_profile.get("transformer_layer_finalize_ffn_wall_s", 0.0) or 0.0
                                ) * 1000.0
                                batch_policy_step_transformer_layer_finalize_ffn_linear1_act_ms += float(
                                    step_profile.get(
                                        "transformer_layer_finalize_ffn_linear1_act_wall_s", 0.0
                                    ) or 0.0
                                ) * 1000.0
                                batch_policy_step_transformer_layer_finalize_ffn_linear2_residual_norm_ms += float(
                                    step_profile.get(
                                        "transformer_layer_finalize_ffn_linear2_residual_norm_wall_s", 0.0
                                    ) or 0.0
                                ) * 1000.0
                                batch_policy_step_transformer_layer_finalize_ffn_linear2_ms += float(
                                    step_profile.get("transformer_layer_finalize_ffn_linear2_wall_s", 0.0) or 0.0
                                ) * 1000.0
                                batch_policy_step_transformer_layer_finalize_ffn_residual_norm_ms += float(
                                    step_profile.get("transformer_layer_finalize_ffn_residual_norm_wall_s", 0.0) or 0.0
                                ) * 1000.0
                                batch_policy_step_transformer_layer_finalize_compiled_ms += float(
                                    step_profile.get("transformer_layer_finalize_compiled_wall_s", 0.0) or 0.0
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
                                batch_policy_step_layer_paged_flash_prefix_zero_fastpath_calls += int(
                                    step_profile.get("transformer_layer_paged_path_flash_prefix_zero_fastpath", 0)
                                    or 0
                                )
                                batch_policy_step_layer_paged_flash_merge_calls += int(
                                    step_profile.get("transformer_layer_paged_path_flash_merge", 0) or 0
                                )
                                batch_policy_step_layer_paged_dense_calls += int(
                                    step_profile.get("transformer_layer_paged_path_dense", 0) or 0
                                )
                                batch_policy_step_layer_paged_page_count_sum += int(
                                    step_profile.get("transformer_layer_paged_page_count_sum", 0) or 0
                                )
                                batch_policy_step_layer_paged_valid_len_sum += int(
                                    step_profile.get("transformer_layer_paged_valid_len_sum", 0) or 0
                                )
                                batch_policy_step_layer_paged_last_page_tokens_sum += int(
                                    step_profile.get("transformer_layer_paged_last_page_tokens_sum", 0) or 0
                                )
                                batch_policy_step_layer_paged_prefix_len_sum += int(
                                    step_profile.get("transformer_layer_paged_prefix_len_sum", 0) or 0
                                )
                                batch_policy_step_layer_flash_prefix_valid_tokens_sum += int(
                                    step_profile.get("transformer_layer_flash_prefix_valid_tokens_sum", 0) or 0
                                )
                                batch_policy_step_layer_flash_prefix_prefix_tokens_sum += int(
                                    step_profile.get("transformer_layer_flash_prefix_prefix_tokens_sum", 0) or 0
                                )
                                batch_policy_step_layer_flash_prefix_tail_tokens_sum += int(
                                    step_profile.get("transformer_layer_flash_prefix_tail_tokens_sum", 0) or 0
                                )
                                batch_policy_step_layer_dense_valid_tokens_sum += int(
                                    step_profile.get("transformer_layer_dense_valid_tokens_sum", 0) or 0
                                )
                                batch_policy_step_layer_dense_prefix_tokens_sum += int(
                                    step_profile.get("transformer_layer_dense_prefix_tokens_sum", 0) or 0
                                )
                                batch_policy_step_layer_dense_tail_tokens_sum += int(
                                    step_profile.get("transformer_layer_dense_tail_tokens_sum", 0) or 0
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
                        batch_policy_step_transformer_layer_finalize_attn_outproj_linear_ms += float(
                            step_profile_tail.get("transformer_layer_finalize_attn_outproj_linear_wall_s", 0.0) or 0.0
                        ) * 1000.0
                        batch_policy_step_transformer_layer_finalize_attn_outproj_norm_ms += float(
                            step_profile_tail.get("transformer_layer_finalize_attn_outproj_norm_wall_s", 0.0) or 0.0
                        ) * 1000.0
                        batch_policy_step_transformer_layer_finalize_ffn_ms += float(
                            step_profile_tail.get("transformer_layer_finalize_ffn_wall_s", 0.0) or 0.0
                        ) * 1000.0
                        batch_policy_step_transformer_layer_finalize_ffn_linear1_act_ms += float(
                            step_profile_tail.get("transformer_layer_finalize_ffn_linear1_act_wall_s", 0.0) or 0.0
                        ) * 1000.0
                        batch_policy_step_transformer_layer_finalize_ffn_linear2_residual_norm_ms += float(
                            step_profile_tail.get("transformer_layer_finalize_ffn_linear2_residual_norm_wall_s", 0.0)
                            or 0.0
                        ) * 1000.0
                        batch_policy_step_transformer_layer_finalize_ffn_linear2_ms += float(
                            step_profile_tail.get("transformer_layer_finalize_ffn_linear2_wall_s", 0.0) or 0.0
                        ) * 1000.0
                        batch_policy_step_transformer_layer_finalize_ffn_residual_norm_ms += float(
                            step_profile_tail.get("transformer_layer_finalize_ffn_residual_norm_wall_s", 0.0) or 0.0
                        ) * 1000.0
                        batch_policy_step_transformer_layer_finalize_compiled_ms += float(
                            step_profile_tail.get("transformer_layer_finalize_compiled_wall_s", 0.0) or 0.0
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
                        batch_policy_step_layer_paged_flash_prefix_zero_fastpath_calls += int(
                            step_profile_tail.get("transformer_layer_paged_path_flash_prefix_zero_fastpath", 0)
                            or 0
                        )
                        batch_policy_step_layer_paged_flash_merge_calls += int(
                            step_profile_tail.get("transformer_layer_paged_path_flash_merge", 0) or 0
                        )
                        batch_policy_step_layer_paged_dense_calls += int(
                            step_profile_tail.get("transformer_layer_paged_path_dense", 0) or 0
                        )
                        batch_policy_step_layer_paged_page_count_sum += int(
                            step_profile_tail.get("transformer_layer_paged_page_count_sum", 0) or 0
                        )
                        batch_policy_step_layer_paged_valid_len_sum += int(
                            step_profile_tail.get("transformer_layer_paged_valid_len_sum", 0) or 0
                        )
                        batch_policy_step_layer_paged_last_page_tokens_sum += int(
                            step_profile_tail.get("transformer_layer_paged_last_page_tokens_sum", 0) or 0
                        )
                        batch_policy_step_layer_paged_prefix_len_sum += int(
                            step_profile_tail.get("transformer_layer_paged_prefix_len_sum", 0) or 0
                        )
                        batch_policy_step_layer_flash_prefix_valid_tokens_sum += int(
                            step_profile_tail.get("transformer_layer_flash_prefix_valid_tokens_sum", 0) or 0
                        )
                        batch_policy_step_layer_flash_prefix_prefix_tokens_sum += int(
                            step_profile_tail.get("transformer_layer_flash_prefix_prefix_tokens_sum", 0) or 0
                        )
                        batch_policy_step_layer_flash_prefix_tail_tokens_sum += int(
                            step_profile_tail.get("transformer_layer_flash_prefix_tail_tokens_sum", 0) or 0
                        )
                        batch_policy_step_layer_dense_valid_tokens_sum += int(
                            step_profile_tail.get("transformer_layer_dense_valid_tokens_sum", 0) or 0
                        )
                        batch_policy_step_layer_dense_prefix_tokens_sum += int(
                            step_profile_tail.get("transformer_layer_dense_prefix_tokens_sum", 0) or 0
                        )
                        batch_policy_step_layer_dense_tail_tokens_sum += int(
                            step_profile_tail.get("transformer_layer_dense_tail_tokens_sum", 0) or 0
                        )

                if compile_observe_enabled:
                    compile_counter_curr_snapshot = _snapshot_dynamo_counters()
                    compile_counter_delta = _diff_dynamo_counters(
                        compile_counter_curr_snapshot,
                        compile_counter_prev_snapshot,
                    )
                    compile_counter_prev_snapshot = compile_counter_curr_snapshot
                    batch_compile_counter_delta_nonzero = int(len(compile_counter_delta))
                    batch_compile_counter_delta_total = int(
                        sum(abs(int(v)) for v in compile_counter_delta.values())
                    )
                    batch_compile_counter_delta_graph_breaks = int(
                        sum(
                            int(v)
                            for k, v in compile_counter_delta.items()
                            if "graph_break" in str(k).lower()
                        )
                    )
                    batch_compile_counter_delta_unique_graphs = int(
                        compile_counter_delta.get("stats.unique_graphs", 0)
                        + compile_counter_delta.get("stats.unique_graph", 0)
                    )
                    batch_compile_counter_delta_recompiles = int(
                        sum(
                            int(v)
                            for k, v in compile_counter_delta.items()
                            if "recompile" in str(k).lower()
                        )
                    )
                    batch_compile_counter_delta_summary = _format_compile_counter_delta(
                        compile_counter_delta,
                        max_items=10,
                    )
                    if compile_observe_output_path is not None:
                        try:
                            with open(compile_observe_output_path, "a", encoding="utf-8") as f:
                                f.write(
                                    json.dumps(
                                        {
                                            "epoch": int(batch_epoch_for_profile),
                                            "batch": int(batch),
                                            "batch_status": (
                                                "oom"
                                                if batch_oom
                                                else ("loss_nonfinite" if batch_nonfinite else "ok")
                                            ),
                                            "delta_nonzero": int(batch_compile_counter_delta_nonzero),
                                            "delta_total_abs": int(batch_compile_counter_delta_total),
                                            "delta_graph_breaks": int(batch_compile_counter_delta_graph_breaks),
                                            "delta_unique_graphs": int(batch_compile_counter_delta_unique_graphs),
                                            "delta_recompiles": int(batch_compile_counter_delta_recompiles),
                                            "delta": {
                                                str(k): int(v)
                                                for k, v in compile_counter_delta.items()
                                            },
                                        },
                                        ensure_ascii=True,
                                        sort_keys=True,
                                    )
                                    + "\n"
                                )
                        except Exception:
                            pass
                    if should_log_phase or (((batch + 1) % compile_observe_log_every_batches) == 0):
                        print(
                            f"[pg-compile-observe] epoch={batch_epoch_for_profile} batch={batch} "
                            f"delta_nonzero={int(batch_compile_counter_delta_nonzero)} "
                            f"delta_total_abs={int(batch_compile_counter_delta_total)} "
                            f"recompiles={int(batch_compile_counter_delta_recompiles)} "
                            f"graph_breaks={int(batch_compile_counter_delta_graph_breaks)} "
                            f"unique_graphs={int(batch_compile_counter_delta_unique_graphs)} "
                            f"delta_top={batch_compile_counter_delta_summary}"
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
                batch_rollout_backward_overlap_wall = 0.0
                batch_rollout_forward_est_wall = float(batch_rollout_wall)
                if batch_tbptt_stream_backward_active and batch_backward_wall > 0.0:
                    batch_rollout_backward_overlap_wall = float(min(batch_rollout_wall, batch_backward_wall))
                    batch_rollout_forward_est_wall = float(
                        max(0.0, batch_rollout_wall - batch_rollout_backward_overlap_wall)
                    )

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
                    transition_async_mode = bool(int(batch_rollout_transition_async_enabled) > 0)
                    transition_group_share = float(
                        batch_rollout_transition_group_wall_ms / max(1e-9, batch_rollout_transition_wall_ms)
                    )
                    transition_env_pack_share = float(
                        batch_rollout_transition_env_pack_wall_ms / max(1e-9, batch_rollout_transition_wall_ms)
                    )
                    transition_state_update_share = float(
                        batch_rollout_transition_state_update_wall_ms / max(1e-9, batch_rollout_transition_wall_ms)
                    )
                    transition_noise_share = float(
                        batch_rollout_transition_noise_wall_ms / max(1e-9, batch_rollout_transition_wall_ms)
                    )
                    transition_fused_share = float(
                        batch_rollout_transition_fused_wall_ms / max(1e-9, batch_rollout_transition_wall_ms)
                    )
                    rollout_breakdown_suffix += (
                        f" rollout_policy_wall_ms={batch_rollout_policy_wall_ms:.2f}"
                        f" rollout_transition_wall_ms={batch_rollout_transition_wall_ms:.2f}"
                        f" rollout_policy_wall_share={rollout_policy_wall_share:.3f}"
                        f" rollout_transition_group_share={transition_group_share:.3f}"
                        f" rollout_transition_env_pack_share={transition_env_pack_share:.3f}"
                        f" rollout_transition_state_update_share={transition_state_update_share:.3f}"
                        f" rollout_transition_noise_share={transition_noise_share:.3f}"
                        f" rollout_transition_fused_share={transition_fused_share:.3f}"
                        f" rollout_transition_fused_calls={int(batch_rollout_transition_fused_call_count)}"
                        f" rollout_transition_fused_groups={int(batch_rollout_transition_fused_group_count)}"
                        f" rollout_transition_group_count={int(batch_rollout_transition_group_count)}"
                    )
                    if (
                        int(batch_rollout_transition_fused_enabled) > 0
                        or batch_rollout_transition_fused_launch_wall_ms > 0.0
                    ):
                        transition_fused_launch_share = float(
                            batch_rollout_transition_fused_launch_wall_ms
                            / max(1e-9, batch_rollout_transition_wall_ms)
                        )
                        rollout_breakdown_suffix += (
                            f" rollout_transition_fused_enabled={int(batch_rollout_transition_fused_enabled)}"
                            f" rollout_transition_fused_launch_share={transition_fused_launch_share:.3f}"
                        )
                    if (
                        int(batch_rollout_transition_checkpoint_enabled) > 0
                        or int(batch_rollout_transition_checkpoint_call_count) > 0
                    ):
                        rollout_breakdown_suffix += (
                            f" rollout_transition_checkpoint_enabled="
                            f"{int(batch_rollout_transition_checkpoint_enabled)}"
                            f" rollout_transition_checkpoint_calls="
                            f"{int(batch_rollout_transition_checkpoint_call_count)}"
                        )
                    if not transition_async_mode:
                        transition_y_share = float(
                            batch_rollout_transition_y_wall_ms / max(1e-9, batch_rollout_transition_wall_ms)
                        )
                        transition_x_share = float(
                            batch_rollout_transition_x_wall_ms / max(1e-9, batch_rollout_transition_wall_ms)
                        )
                        rollout_breakdown_suffix += (
                            f" rollout_transition_y_share={transition_y_share:.3f}"
                            f" rollout_transition_x_share={transition_x_share:.3f}"
                        )
                    if (
                        batch_rollout_transition_gp_first_projection_wall_ms > 0.0
                        or batch_rollout_transition_gp_second_projection_wall_ms > 0.0
                        or int(batch_rollout_transition_gp_projection_call_count) > 0
                        or int(batch_rollout_transition_gp_profile_group_count) > 0
                    ):
                        rollout_breakdown_suffix += (
                            f" rollout_transition_gp_first_proj_ms="
                            f"{batch_rollout_transition_gp_first_projection_wall_ms:.2f}"
                            f" rollout_transition_gp_second_proj_ms="
                            f"{batch_rollout_transition_gp_second_projection_wall_ms:.2f}"
                            f" rollout_transition_gp_proj_calls="
                            f"{int(batch_rollout_transition_gp_projection_call_count)}"
                            f" rollout_transition_gp_rff_fused_calls="
                            f"{int(batch_rollout_transition_gp_rff_fused_call_count)}"
                            f" rollout_transition_gp_profile_groups="
                            f"{int(batch_rollout_transition_gp_profile_group_count)}"
                            f" rollout_transition_gp_profile_sync_groups="
                            f"{int(batch_rollout_transition_gp_profile_sync_group_count)}"
                        )
                    if (
                        int(batch_rollout_transition_packed_env_input_group_count) > 0
                        or int(batch_rollout_transition_packed_env_input_call_count) > 0
                    ):
                        rollout_breakdown_suffix += (
                            f" rollout_transition_packed_env_input_groups="
                            f"{int(batch_rollout_transition_packed_env_input_group_count)}"
                            f" rollout_transition_packed_env_input_calls="
                            f"{int(batch_rollout_transition_packed_env_input_call_count)}"
                        )
                    if (
                        int(batch_rollout_transition_only_build_group_count) > 0
                        or int(batch_rollout_transition_only_skipped_generator_count) > 0
                    ):
                        rollout_breakdown_suffix += (
                            f" rollout_transition_only_build_groups="
                            f"{int(batch_rollout_transition_only_build_group_count)}"
                            f" rollout_transition_only_skipped_generators="
                            f"{int(batch_rollout_transition_only_skipped_generator_count)}"
                        )
                    if (
                        int(batch_rollout_transition_async_enabled) > 0
                        or batch_rollout_transition_group_launch_wall_ms > 0.0
                        or batch_rollout_transition_group_sync_wall_ms > 0.0
                    ):
                        transition_group_launch_share = float(
                            batch_rollout_transition_group_launch_wall_ms
                            / max(1e-9, batch_rollout_transition_wall_ms)
                        )
                        transition_group_sync_share = float(
                            batch_rollout_transition_group_sync_wall_ms
                            / max(1e-9, batch_rollout_transition_wall_ms)
                        )
                        rollout_breakdown_suffix += (
                            f" rollout_transition_async_enabled={int(batch_rollout_transition_async_enabled)}"
                            f" rollout_transition_group_launch_share={transition_group_launch_share:.3f}"
                            f" rollout_transition_group_sync_share={transition_group_sync_share:.3f}"
                        )
                    if int(batch_rollout_transition_group_count) > 0:
                        transition_bucket_mean_batch = float(
                            float(max(1, int(batch_size))) / float(max(1, int(batch_rollout_transition_group_count)))
                        )
                        rollout_breakdown_suffix += (
                            f" rollout_transition_family_group_count="
                            f"{int(batch_rollout_transition_family_group_count)}"
                            f" rollout_transition_inner_grouping_structure_enabled="
                            f"{int(batch_rollout_transition_inner_grouping_structure_enabled)}"
                            f" rollout_transition_inner_min_bucket="
                            f"{int(batch_rollout_transition_inner_min_bucket)}"
                            f" rollout_transition_bucket_max_batch="
                            f"{int(batch_rollout_transition_bucket_max_batch)}"
                            f" rollout_transition_bucket_mean_batch={transition_bucket_mean_batch:.2f}"
                        )
                    if batch_rollout_transition_work_padded_est > 0.0:
                        transition_work_fill_ratio = float(
                            batch_rollout_transition_work_actual_est
                            / max(1e-9, batch_rollout_transition_work_padded_est)
                        )
                        rollout_breakdown_suffix += (
                            f" rollout_transition_work_fill_ratio={transition_work_fill_ratio:.3f}"
                        )
                if batch_rollout_noise_mode is not None:
                    rollout_breakdown_suffix += f" rollout_noise_mode={batch_rollout_noise_mode}"
                if batch_rollout_noise_block_size is not None:
                    rollout_breakdown_suffix += (
                        f" rollout_noise_block_size={int(batch_rollout_noise_block_size)}"
                    )
                if batch_rollout_transition_reference_mode is not None:
                    rollout_breakdown_suffix += (
                        f" env_ref_mode={batch_rollout_transition_reference_mode}"
                        f" env_ref_share={float(batch_rollout_reference_semantics_share):.3f}"
                        f" env_strict_share={float(batch_rollout_strict_joint_transition_share):.3f}"
                        f" env_exact_scm={int(batch_rollout_exact_scm_count)}"
                        f" env_exact_gp={int(batch_rollout_exact_gp_count)}"
                        f" env_legacy_scm={int(batch_rollout_legacy_scm_count)}"
                        f" env_legacy_gp={int(batch_rollout_legacy_gp_count)}"
                    )
                batch_size_metric = int(max(1, int(batch_size)))
                batch_wall_excl_compile = float(batch_rollout_wall + batch_backward_wall + batch_step_wall)
                batch_wall_incl_compile = float(batch_wall_excl_compile + batch_compile_warmup_wall)
                batch_wall_excl_compile_per_batch_item = float(
                    batch_wall_excl_compile / float(batch_size_metric)
                )
                batch_wall_incl_compile_per_batch_item = float(
                    batch_wall_incl_compile / float(batch_size_metric)
                )
                rollout_breakdown_suffix += (
                    f" batch_size={batch_size_metric}"
                    f" batch_wall_excl_compile_s={batch_wall_excl_compile:.3f}"
                    f" batch_wall_incl_compile_s={batch_wall_incl_compile:.3f}"
                    f" batch_wall_excl_compile_per_batch_item_s="
                    f"{batch_wall_excl_compile_per_batch_item:.6f}"
                    f" batch_wall_incl_compile_per_batch_item_s="
                    f"{batch_wall_incl_compile_per_batch_item:.6f}"
                )
                if batch_compile_warmup_wall > 0.0:
                    rollout_breakdown_suffix += f" compile_warmup_s={batch_compile_warmup_wall:.3f}"
                    if batch_compile_warmup_ok is not None:
                        rollout_breakdown_suffix += f" compile_warmup_ok={int(bool(batch_compile_warmup_ok))}"
                if compile_observe_enabled:
                    rollout_breakdown_suffix += (
                        f" compile_counter_delta_nonzero={int(batch_compile_counter_delta_nonzero)}"
                        f" compile_counter_delta_total_abs={int(batch_compile_counter_delta_total)}"
                        f" compile_counter_recompiles={int(batch_compile_counter_delta_recompiles)}"
                        f" compile_counter_graph_breaks={int(batch_compile_counter_delta_graph_breaks)}"
                        f" compile_counter_unique_graphs={int(batch_compile_counter_delta_unique_graphs)}"
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
                    policy_step_layer_finalize_attn_outproj_linear_share = float(
                        batch_policy_step_transformer_layer_finalize_attn_outproj_linear_ms
                        / max(1e-9, batch_policy_step_transformer_layer_total_ms)
                    )
                    policy_step_layer_finalize_attn_outproj_norm_share = float(
                        batch_policy_step_transformer_layer_finalize_attn_outproj_norm_ms
                        / max(1e-9, batch_policy_step_transformer_layer_total_ms)
                    )
                    policy_step_layer_finalize_ffn_share = float(
                        batch_policy_step_transformer_layer_finalize_ffn_ms
                        / max(1e-9, batch_policy_step_transformer_layer_total_ms)
                    )
                    policy_step_layer_finalize_ffn_linear1_act_share = float(
                        batch_policy_step_transformer_layer_finalize_ffn_linear1_act_ms
                        / max(1e-9, batch_policy_step_transformer_layer_total_ms)
                    )
                    policy_step_layer_finalize_ffn_linear2_residual_norm_share = float(
                        batch_policy_step_transformer_layer_finalize_ffn_linear2_residual_norm_ms
                        / max(1e-9, batch_policy_step_transformer_layer_total_ms)
                    )
                    policy_step_layer_finalize_ffn_linear2_share = float(
                        batch_policy_step_transformer_layer_finalize_ffn_linear2_ms
                        / max(1e-9, batch_policy_step_transformer_layer_total_ms)
                    )
                    policy_step_layer_finalize_ffn_residual_norm_share = float(
                        batch_policy_step_transformer_layer_finalize_ffn_residual_norm_ms
                        / max(1e-9, batch_policy_step_transformer_layer_total_ms)
                    )
                    policy_step_layer_finalize_compiled_share = float(
                        batch_policy_step_transformer_layer_finalize_compiled_ms
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
                        f" policy_step_tf_layer_finalize_attn_outproj_linear_share="
                        f"{policy_step_layer_finalize_attn_outproj_linear_share:.3f}"
                        f" policy_step_tf_layer_finalize_attn_outproj_norm_share="
                        f"{policy_step_layer_finalize_attn_outproj_norm_share:.3f}"
                        f" policy_step_tf_layer_finalize_ffn_share={policy_step_layer_finalize_ffn_share:.3f}"
                        f" policy_step_tf_layer_finalize_ffn_linear1_act_share="
                        f"{policy_step_layer_finalize_ffn_linear1_act_share:.3f}"
                        f" policy_step_tf_layer_finalize_ffn_linear2_residual_norm_share="
                        f"{policy_step_layer_finalize_ffn_linear2_residual_norm_share:.3f}"
                        f" policy_step_tf_layer_finalize_ffn_linear2_share="
                        f"{policy_step_layer_finalize_ffn_linear2_share:.3f}"
                        f" policy_step_tf_layer_finalize_ffn_residual_norm_share="
                        f"{policy_step_layer_finalize_ffn_residual_norm_share:.3f}"
                    )
                    if batch_policy_step_transformer_layer_finalize_compiled_ms > 0.0:
                        rollout_breakdown_suffix += (
                            f" policy_step_tf_layer_finalize_compiled_share="
                            f"{policy_step_layer_finalize_compiled_share:.3f}"
                        )
                    paged_dispatch_total = int(
                        batch_policy_step_layer_paged_single_page_calls
                        + batch_policy_step_layer_paged_flash_prefix_calls
                        + batch_policy_step_layer_paged_flash_merge_calls
                        + batch_policy_step_layer_paged_dense_calls
                    )
                    if paged_dispatch_total > 0:
                        rollout_breakdown_suffix += (
                            f" policy_step_paged_dispatch(single/flashp/flashp0/flashm/dense)="
                            f"{batch_policy_step_layer_paged_single_page_calls}/"
                            f"{batch_policy_step_layer_paged_flash_prefix_calls}/"
                            f"{batch_policy_step_layer_paged_flash_prefix_zero_fastpath_calls}/"
                            f"{batch_policy_step_layer_paged_flash_merge_calls}/"
                            f"{batch_policy_step_layer_paged_dense_calls}"
                        )
                        if batch_policy_step_layer_paged_flash_prefix_calls > 0:
                            flash_prefix_calls = float(batch_policy_step_layer_paged_flash_prefix_calls)
                            rollout_breakdown_suffix += (
                                f" policy_step_flashp_avg_tokens(valid/prefix/tail)="
                                f"{(batch_policy_step_layer_flash_prefix_valid_tokens_sum / flash_prefix_calls):.1f}/"
                                f"{(batch_policy_step_layer_flash_prefix_prefix_tokens_sum / flash_prefix_calls):.1f}/"
                                f"{(batch_policy_step_layer_flash_prefix_tail_tokens_sum / flash_prefix_calls):.1f}"
                            )
                        if batch_policy_step_layer_paged_dense_calls > 0:
                            dense_calls = float(batch_policy_step_layer_paged_dense_calls)
                            rollout_breakdown_suffix += (
                                f" policy_step_dense_avg_tokens(valid/prefix/tail)="
                                f"{(batch_policy_step_layer_dense_valid_tokens_sum / dense_calls):.1f}/"
                                f"{(batch_policy_step_layer_dense_prefix_tokens_sum / dense_calls):.1f}/"
                                f"{(batch_policy_step_layer_dense_tail_tokens_sum / dense_calls):.1f}"
                            )
                        rollout_breakdown_suffix += (
                            f" policy_step_paged_avg(page_count/valid/last_page/prefix)="
                            f"{(batch_policy_step_layer_paged_page_count_sum / float(paged_dispatch_total)):.1f}/"
                            f"{(batch_policy_step_layer_paged_valid_len_sum / float(paged_dispatch_total)):.1f}/"
                            f"{(batch_policy_step_layer_paged_last_page_tokens_sum / float(paged_dispatch_total)):.1f}/"
                            f"{(batch_policy_step_layer_paged_prefix_len_sum / float(paged_dispatch_total)):.1f}"
                        )
                if rollout_cuda_busy_ratio is not None:
                    rollout_breakdown_suffix += f" rollout_cuda_busy_ratio={rollout_cuda_busy_ratio:.3f}"
                if backward_cuda_busy_ratio is not None:
                    rollout_breakdown_suffix += f" backward_cuda_busy_ratio={backward_cuda_busy_ratio:.3f}"
                if batch_tbptt_stream_backward_active:
                    rollout_instage_backward_share = float(
                        batch_rollout_backward_overlap_wall / max(1e-9, batch_rollout_wall)
                    )
                    tbptt_stream_merge_eff = float(
                        batch_tbptt_stream_backward_roots / max(1, batch_tbptt_stream_backward_launches)
                    )
                    rollout_breakdown_suffix += (
                        f" tbptt_stream_merge_windows={int(batch_tbptt_stream_merge_windows)}"
                        f" tbptt_stream_backward_calls={int(batch_tbptt_stream_backward_launches)}"
                        f" tbptt_stream_backward_roots={int(batch_tbptt_stream_backward_roots)}"
                        f" tbptt_stream_merge_effective={tbptt_stream_merge_eff:.2f}"
                        f" rollout_forward_est_s={batch_rollout_forward_est_wall:.3f}"
                        f" rollout_instage_backward_s={batch_rollout_backward_overlap_wall:.3f}"
                        f" rollout_instage_backward_share={rollout_instage_backward_share:.3f}"
                    )
                    if int(batch_tbptt_stream_merge_guard_flushes) > 0:
                        rollout_breakdown_suffix += (
                            f" tbptt_stream_merge_guard_flushes={int(batch_tbptt_stream_merge_guard_flushes)}"
                        )
                    if int(batch_tbptt_stream_merge_guard_prefallbacks) > 0:
                        rollout_breakdown_suffix += (
                            f" tbptt_stream_merge_guard_prefallbacks={int(batch_tbptt_stream_merge_guard_prefallbacks)}"
                        )
                if batch_pg_loss_signature is not None:
                    rollout_breakdown_suffix += f" pg_loss_sig={batch_pg_loss_signature}"
                rollout_breakdown_suffix += (
                    f" reward_nonfinite={float(batch_reward_nonfinite_share):.3f}"
                    f" reward_nan={float(batch_reward_nan_share):.3f}"
                    f" reward_inf={float(batch_reward_inf_share):.3f}"
                    f" return_nonfinite={float(batch_reinforce_return_nonfinite_share):.3f}"
                    f" logprob_nonfinite={float(batch_reinforce_log_prob_nonfinite_share):.3f}"
                    f" adv_nonfinite={float(batch_reinforce_adv_nonfinite_share):.3f}"
                )
                if math.isfinite(float(batch_state_abs_max_value)):
                    rollout_breakdown_suffix += f" state_absmax={float(batch_state_abs_max_value):.3e}"
                if batch_reinforce_log_prob_mean_value is not None and math.isfinite(float(batch_reinforce_log_prob_mean_value)):
                    rollout_breakdown_suffix += f" logprob_mean={float(batch_reinforce_log_prob_mean_value):+.3e}"
                if batch_reinforce_log_prob_std_value is not None and math.isfinite(float(batch_reinforce_log_prob_std_value)):
                    rollout_breakdown_suffix += f" logprob_std={float(batch_reinforce_log_prob_std_value):.3e}"
                if batch_reinforce_action_dim_mean_value is not None and math.isfinite(float(batch_reinforce_action_dim_mean_value)):
                    rollout_breakdown_suffix += f" actdim_mean={float(batch_reinforce_action_dim_mean_value):.3f}"
                if math.isfinite(float(batch_reinforce_action_dim_min_value)):
                    rollout_breakdown_suffix += f" actdim_min={float(batch_reinforce_action_dim_min_value):.0f}"
                if math.isfinite(float(batch_reinforce_action_dim_max_value)):
                    rollout_breakdown_suffix += f" actdim_max={float(batch_reinforce_action_dim_max_value):.0f}"
                if batch_reinforce_action_std_mean_value is not None and math.isfinite(float(batch_reinforce_action_std_mean_value)):
                    rollout_breakdown_suffix += f" policy_std_mean={float(batch_reinforce_action_std_mean_value):.3e}"
                if math.isfinite(float(batch_reinforce_action_std_min_value)):
                    rollout_breakdown_suffix += f" policy_std_min={float(batch_reinforce_action_std_min_value):.3e}"
                if math.isfinite(float(batch_reinforce_action_std_max_value)):
                    rollout_breakdown_suffix += f" policy_std_max={float(batch_reinforce_action_std_max_value):.3e}"
                if batch_reinforce_logprob_log_std_mean_value is not None and math.isfinite(float(batch_reinforce_logprob_log_std_mean_value)):
                    rollout_breakdown_suffix += f" logstd_mean={float(batch_reinforce_logprob_log_std_mean_value):+.3e}"
                if math.isfinite(float(batch_reinforce_logprob_log_std_min_value)):
                    rollout_breakdown_suffix += f" logstd_min={float(batch_reinforce_logprob_log_std_min_value):+.3e}"
                if math.isfinite(float(batch_reinforce_logprob_log_std_max_value)):
                    rollout_breakdown_suffix += f" logstd_max={float(batch_reinforce_logprob_log_std_max_value):+.3e}"
                if batch_reinforce_logprob_z2_mean_value is not None and math.isfinite(float(batch_reinforce_logprob_z2_mean_value)):
                    rollout_breakdown_suffix += f" z2_mean={float(batch_reinforce_logprob_z2_mean_value):.3e}"
                if math.isfinite(float(batch_reinforce_logprob_z2_max_value)):
                    rollout_breakdown_suffix += f" z2_max={float(batch_reinforce_logprob_z2_max_value):.3e}"
                if math.isfinite(float(batch_action_std_min_value)):
                    rollout_breakdown_suffix += f" action_std_min={float(batch_action_std_min_value):.3e}"
                if math.isfinite(float(batch_action_std_max_value)):
                    rollout_breakdown_suffix += f" action_std_max={float(batch_action_std_max_value):.3e}"
                host_mem_snapshot = _get_host_memory_snapshot()
                if isinstance(host_mem_snapshot, dict):
                    rollout_breakdown_suffix += (
                        f" host_rss_gib={host_mem_snapshot['rss_gib']:.2f}"
                        f" host_avail_gib={host_mem_snapshot['avail_gib']:.2f}"
                    )
    
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
                            "tbptt_stream_backward": bool(batch_tbptt_stream_backward_active),
                            "tbptt_stream_merge_windows": int(batch_tbptt_stream_merge_windows),
                            "cuda_elapsed_ms": cuda_ms,
                            "cuda_busy_ratio": cuda_busy_ratio,
                        }
                        if stage_name == "rollout" and rollout_breakdown_total_ms > 0.0:
                            stage_extra["rollout_policy_cuda_ms"] = float(batch_rollout_policy_cuda_ms)
                            stage_extra["rollout_transition_cuda_ms"] = float(batch_rollout_transition_cuda_ms)
                            stage_extra["rollout_policy_share"] = float(
                                batch_rollout_policy_cuda_ms / max(1e-9, rollout_breakdown_total_ms)
                            )
                        if stage_name == "rollout":
                            batch_size_metric = int(max(1, int(batch_size)))
                            batch_wall_excl_compile = float(batch_rollout_wall + batch_backward_wall + batch_step_wall)
                            batch_wall_incl_compile = float(
                                batch_wall_excl_compile + batch_compile_warmup_wall
                            )
                            stage_extra["batch_wall_excl_compile_s"] = float(batch_wall_excl_compile)
                            stage_extra["batch_wall_incl_compile_s"] = float(batch_wall_incl_compile)
                            stage_extra["batch_size"] = int(batch_size_metric)
                            stage_extra["batch_wall_excl_compile_per_batch_item_s"] = float(
                                batch_wall_excl_compile / float(batch_size_metric)
                            )
                            stage_extra["batch_wall_incl_compile_per_batch_item_s"] = float(
                                batch_wall_incl_compile / float(batch_size_metric)
                            )
                            if batch_compile_warmup_wall > 0.0:
                                stage_extra["compile_warmup_s"] = float(batch_compile_warmup_wall)
                                if batch_compile_warmup_ok is not None:
                                    stage_extra["compile_warmup_ok"] = int(bool(batch_compile_warmup_ok))
                            if compile_observe_enabled:
                                stage_extra["compile_counter_delta_nonzero"] = int(
                                    batch_compile_counter_delta_nonzero
                                )
                                stage_extra["compile_counter_delta_total_abs"] = int(
                                    batch_compile_counter_delta_total
                                )
                                stage_extra["compile_counter_recompiles"] = int(
                                    batch_compile_counter_delta_recompiles
                                )
                                stage_extra["compile_counter_graph_breaks"] = int(
                                    batch_compile_counter_delta_graph_breaks
                                )
                                stage_extra["compile_counter_unique_graphs"] = int(
                                    batch_compile_counter_delta_unique_graphs
                                )
                        if stage_name == "rollout" and rollout_wall_total_ms > 0.0:
                            stage_extra["rollout_policy_wall_ms"] = float(batch_rollout_policy_wall_ms)
                            stage_extra["rollout_transition_wall_ms"] = float(batch_rollout_transition_wall_ms)
                            stage_extra["rollout_policy_wall_share"] = float(
                                batch_rollout_policy_wall_ms / max(1e-9, rollout_wall_total_ms)
                            )
                            if int(batch_rollout_transition_async_enabled) <= 0:
                                stage_extra["rollout_transition_y_share"] = float(
                                    batch_rollout_transition_y_wall_ms / max(1e-9, batch_rollout_transition_wall_ms)
                                )
                                stage_extra["rollout_transition_x_share"] = float(
                                    batch_rollout_transition_x_wall_ms / max(1e-9, batch_rollout_transition_wall_ms)
                                )
                            stage_extra["rollout_transition_group_share"] = float(
                                batch_rollout_transition_group_wall_ms / max(1e-9, batch_rollout_transition_wall_ms)
                            )
                            if (
                                int(batch_rollout_transition_async_enabled) > 0
                                or batch_rollout_transition_group_launch_wall_ms > 0.0
                                or batch_rollout_transition_group_sync_wall_ms > 0.0
                            ):
                                stage_extra["rollout_transition_async_enabled"] = int(
                                    batch_rollout_transition_async_enabled
                                )
                                stage_extra["rollout_transition_group_launch_share"] = float(
                                    batch_rollout_transition_group_launch_wall_ms
                                    / max(1e-9, batch_rollout_transition_wall_ms)
                                )
                                stage_extra["rollout_transition_group_sync_share"] = float(
                                    batch_rollout_transition_group_sync_wall_ms
                                    / max(1e-9, batch_rollout_transition_wall_ms)
                                )
                            stage_extra["rollout_transition_env_pack_share"] = float(
                                batch_rollout_transition_env_pack_wall_ms / max(1e-9, batch_rollout_transition_wall_ms)
                            )
                            stage_extra["rollout_transition_state_update_share"] = float(
                                batch_rollout_transition_state_update_wall_ms / max(1e-9, batch_rollout_transition_wall_ms)
                            )
                            stage_extra["rollout_transition_noise_share"] = float(
                                batch_rollout_transition_noise_wall_ms / max(1e-9, batch_rollout_transition_wall_ms)
                            )
                            stage_extra["rollout_transition_fused_share"] = float(
                                batch_rollout_transition_fused_wall_ms / max(1e-9, batch_rollout_transition_wall_ms)
                            )
                            stage_extra["rollout_transition_fused_calls"] = int(
                                batch_rollout_transition_fused_call_count
                            )
                            stage_extra["rollout_transition_fused_groups"] = int(
                                batch_rollout_transition_fused_group_count
                            )
                            if (
                                int(batch_rollout_transition_fused_enabled) > 0
                                or batch_rollout_transition_fused_launch_wall_ms > 0.0
                            ):
                                stage_extra["rollout_transition_fused_enabled"] = int(
                                    batch_rollout_transition_fused_enabled
                                )
                                stage_extra["rollout_transition_fused_launch_share"] = float(
                                    batch_rollout_transition_fused_launch_wall_ms
                                    / max(1e-9, batch_rollout_transition_wall_ms)
                                )
                            if (
                                int(batch_rollout_transition_checkpoint_enabled) > 0
                                or int(batch_rollout_transition_checkpoint_call_count) > 0
                            ):
                                stage_extra["rollout_transition_checkpoint_enabled"] = int(
                                    batch_rollout_transition_checkpoint_enabled
                                )
                                stage_extra["rollout_transition_checkpoint_calls"] = int(
                                    batch_rollout_transition_checkpoint_call_count
                                )
                            stage_extra["rollout_transition_group_count"] = int(batch_rollout_transition_group_count)
                            stage_extra["rollout_transition_family_group_count"] = int(
                                batch_rollout_transition_family_group_count
                            )
                            stage_extra["rollout_transition_inner_grouping_structure_enabled"] = int(
                                batch_rollout_transition_inner_grouping_structure_enabled
                            )
                            stage_extra["rollout_transition_inner_min_bucket"] = int(
                                batch_rollout_transition_inner_min_bucket
                            )
                            stage_extra["rollout_transition_bucket_max_batch"] = int(
                                batch_rollout_transition_bucket_max_batch
                            )
                            stage_extra["rollout_transition_bucket_mean_batch"] = float(
                                float(max(1, int(batch_size)))
                                / float(max(1, int(batch_rollout_transition_group_count)))
                            )
                            if batch_rollout_transition_work_padded_est > 0.0:
                                stage_extra["rollout_transition_work_fill_ratio"] = float(
                                    batch_rollout_transition_work_actual_est
                                    / max(1e-9, batch_rollout_transition_work_padded_est)
                                )
                            if batch_rollout_noise_mode is not None:
                                stage_extra["rollout_noise_mode"] = str(batch_rollout_noise_mode)
                            if batch_rollout_noise_block_size is not None:
                                stage_extra["rollout_noise_block_size"] = int(batch_rollout_noise_block_size)
                            if batch_tbptt_stream_backward_active:
                                stage_extra["tbptt_stream_backward_calls"] = int(
                                    batch_tbptt_stream_backward_launches
                                )
                                stage_extra["tbptt_stream_backward_roots"] = int(
                                    batch_tbptt_stream_backward_roots
                                )
                                stage_extra["tbptt_stream_merge_effective"] = float(
                                    batch_tbptt_stream_backward_roots
                                    / max(1, batch_tbptt_stream_backward_launches)
                                )
                                stage_extra["rollout_forward_est_s"] = float(batch_rollout_forward_est_wall)
                                stage_extra["rollout_instage_backward_s"] = float(batch_rollout_backward_overlap_wall)
                                stage_extra["rollout_instage_backward_share"] = float(
                                    batch_rollout_backward_overlap_wall / max(1e-9, batch_rollout_wall)
                                )
                                stage_extra["tbptt_stream_merge_guard_flushes"] = int(
                                    batch_tbptt_stream_merge_guard_flushes
                                )
                                stage_extra["tbptt_stream_merge_guard_prefallbacks"] = int(
                                    batch_tbptt_stream_merge_guard_prefallbacks
                                )
                                stage_extra["tbptt_backward_mem_guard_hits"] = int(
                                    batch_tbptt_backward_mem_guard_hits
                                )
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
                            if batch_policy_step_transformer_layer_finalize_compiled_ms > 0.0:
                                stage_extra["policy_step_tf_layer_finalize_compiled_share"] = float(
                                    batch_policy_step_transformer_layer_finalize_compiled_ms
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
                            if stage_name == "rollout":
                                batch_size_metric = int(max(1, int(batch_size)))
                                batch_wall_excl_compile = float(batch_rollout_wall + batch_backward_wall + batch_step_wall)
                                batch_wall_incl_compile = float(
                                    batch_wall_excl_compile + batch_compile_warmup_wall
                                )
                                wandb_payload["pg_gpu/batch_size"] = int(batch_size_metric)
                                wandb_payload["pg_gpu/batch_wall_excl_compile_s"] = float(batch_wall_excl_compile)
                                wandb_payload["pg_gpu/batch_wall_incl_compile_s"] = float(batch_wall_incl_compile)
                                wandb_payload["pg_gpu/batch_wall_excl_compile_per_batch_item_s"] = float(
                                    batch_wall_excl_compile / float(batch_size_metric)
                                )
                                wandb_payload["pg_gpu/batch_wall_incl_compile_per_batch_item_s"] = float(
                                    batch_wall_incl_compile / float(batch_size_metric)
                                )
                                if batch_compile_warmup_wall > 0.0:
                                    wandb_payload["pg_gpu/compile_warmup_s"] = float(batch_compile_warmup_wall)
                                    if batch_compile_warmup_ok is not None:
                                        wandb_payload["pg_gpu/compile_warmup_ok"] = int(bool(batch_compile_warmup_ok))
                                if compile_observe_enabled:
                                    wandb_payload["pg_gpu/compile_counter_delta_nonzero"] = int(
                                        batch_compile_counter_delta_nonzero
                                    )
                                    wandb_payload["pg_gpu/compile_counter_delta_total_abs"] = int(
                                        batch_compile_counter_delta_total
                                    )
                                    wandb_payload["pg_gpu/compile_counter_recompiles"] = int(
                                        batch_compile_counter_delta_recompiles
                                    )
                                    wandb_payload["pg_gpu/compile_counter_graph_breaks"] = int(
                                        batch_compile_counter_delta_graph_breaks
                                    )
                                    wandb_payload["pg_gpu/compile_counter_unique_graphs"] = int(
                                        batch_compile_counter_delta_unique_graphs
                                    )
                            if stage_name == "rollout" and rollout_wall_total_ms > 0.0:
                                wandb_payload["pg_gpu/rollout_policy_wall_ms"] = float(batch_rollout_policy_wall_ms)
                                wandb_payload["pg_gpu/rollout_transition_wall_ms"] = float(batch_rollout_transition_wall_ms)
                                wandb_payload["pg_gpu/rollout_policy_wall_share"] = float(
                                    batch_rollout_policy_wall_ms / max(1e-9, rollout_wall_total_ms)
                                )
                                if int(batch_rollout_transition_async_enabled) <= 0:
                                    wandb_payload["pg_gpu/rollout_transition_y_share"] = float(
                                        batch_rollout_transition_y_wall_ms / max(1e-9, batch_rollout_transition_wall_ms)
                                    )
                                    wandb_payload["pg_gpu/rollout_transition_x_share"] = float(
                                        batch_rollout_transition_x_wall_ms / max(1e-9, batch_rollout_transition_wall_ms)
                                    )
                                wandb_payload["pg_gpu/rollout_transition_group_share"] = float(
                                    batch_rollout_transition_group_wall_ms / max(1e-9, batch_rollout_transition_wall_ms)
                                )
                                if (
                                    int(batch_rollout_transition_async_enabled) > 0
                                    or batch_rollout_transition_group_launch_wall_ms > 0.0
                                    or batch_rollout_transition_group_sync_wall_ms > 0.0
                                ):
                                    wandb_payload["pg_gpu/rollout_transition_async_enabled"] = int(
                                        batch_rollout_transition_async_enabled
                                    )
                                    wandb_payload["pg_gpu/rollout_transition_group_launch_share"] = float(
                                        batch_rollout_transition_group_launch_wall_ms
                                        / max(1e-9, batch_rollout_transition_wall_ms)
                                    )
                                    wandb_payload["pg_gpu/rollout_transition_group_sync_share"] = float(
                                        batch_rollout_transition_group_sync_wall_ms
                                        / max(1e-9, batch_rollout_transition_wall_ms)
                                    )
                                wandb_payload["pg_gpu/rollout_transition_env_pack_share"] = float(
                                    batch_rollout_transition_env_pack_wall_ms / max(1e-9, batch_rollout_transition_wall_ms)
                                )
                                wandb_payload["pg_gpu/rollout_transition_state_update_share"] = float(
                                    batch_rollout_transition_state_update_wall_ms / max(1e-9, batch_rollout_transition_wall_ms)
                                )
                                wandb_payload["pg_gpu/rollout_transition_noise_share"] = float(
                                    batch_rollout_transition_noise_wall_ms / max(1e-9, batch_rollout_transition_wall_ms)
                                )
                                wandb_payload["pg_gpu/rollout_transition_fused_share"] = float(
                                    batch_rollout_transition_fused_wall_ms / max(1e-9, batch_rollout_transition_wall_ms)
                                )
                                wandb_payload["pg_gpu/rollout_transition_fused_calls"] = int(
                                    batch_rollout_transition_fused_call_count
                                )
                                wandb_payload["pg_gpu/rollout_transition_fused_groups"] = int(
                                    batch_rollout_transition_fused_group_count
                                )
                                if (
                                    int(batch_rollout_transition_fused_enabled) > 0
                                    or batch_rollout_transition_fused_launch_wall_ms > 0.0
                                ):
                                    wandb_payload["pg_gpu/rollout_transition_fused_enabled"] = int(
                                        batch_rollout_transition_fused_enabled
                                    )
                                    wandb_payload["pg_gpu/rollout_transition_fused_launch_share"] = float(
                                        batch_rollout_transition_fused_launch_wall_ms
                                        / max(1e-9, batch_rollout_transition_wall_ms)
                                    )
                                if (
                                    int(batch_rollout_transition_checkpoint_enabled) > 0
                                    or int(batch_rollout_transition_checkpoint_call_count) > 0
                                ):
                                    wandb_payload["pg_gpu/rollout_transition_checkpoint_enabled"] = int(
                                        batch_rollout_transition_checkpoint_enabled
                                    )
                                    wandb_payload["pg_gpu/rollout_transition_checkpoint_calls"] = int(
                                        batch_rollout_transition_checkpoint_call_count
                                    )
                                wandb_payload["pg_gpu/rollout_transition_group_count"] = int(
                                    batch_rollout_transition_group_count
                                )
                                wandb_payload["pg_gpu/rollout_transition_family_group_count"] = int(
                                    batch_rollout_transition_family_group_count
                                )
                                wandb_payload["pg_gpu/rollout_transition_inner_grouping_structure_enabled"] = int(
                                    batch_rollout_transition_inner_grouping_structure_enabled
                                )
                                wandb_payload["pg_gpu/rollout_transition_inner_min_bucket"] = int(
                                    batch_rollout_transition_inner_min_bucket
                                )
                                wandb_payload["pg_gpu/rollout_transition_bucket_max_batch"] = int(
                                    batch_rollout_transition_bucket_max_batch
                                )
                                wandb_payload["pg_gpu/rollout_transition_bucket_mean_batch"] = float(
                                    float(max(1, int(batch_size)))
                                    / float(max(1, int(batch_rollout_transition_group_count)))
                                )
                                if batch_rollout_transition_work_padded_est > 0.0:
                                    wandb_payload["pg_gpu/rollout_transition_work_fill_ratio"] = float(
                                        batch_rollout_transition_work_actual_est
                                        / max(1e-9, batch_rollout_transition_work_padded_est)
                                    )
                                if batch_rollout_noise_mode is not None:
                                    wandb_payload["pg_gpu/rollout_noise_mode"] = str(batch_rollout_noise_mode)
                                if batch_rollout_noise_block_size is not None:
                                    wandb_payload["pg_gpu/rollout_noise_block_size"] = int(batch_rollout_noise_block_size)
                                if batch_tbptt_stream_backward_active:
                                    wandb_payload["pg_gpu/tbptt_stream_merge_windows"] = int(
                                        batch_tbptt_stream_merge_windows
                                    )
                                    wandb_payload["pg_gpu/tbptt_stream_backward_calls"] = int(
                                        batch_tbptt_stream_backward_launches
                                    )
                                    wandb_payload["pg_gpu/tbptt_stream_backward_roots"] = int(
                                        batch_tbptt_stream_backward_roots
                                    )
                                    wandb_payload["pg_gpu/tbptt_stream_merge_effective"] = float(
                                        batch_tbptt_stream_backward_roots
                                        / max(1, batch_tbptt_stream_backward_launches)
                                    )
                                    wandb_payload["pg_gpu/rollout_forward_est_s"] = float(batch_rollout_forward_est_wall)
                                    wandb_payload["pg_gpu/rollout_instage_backward_s"] = float(
                                        batch_rollout_backward_overlap_wall
                                    )
                                    wandb_payload["pg_gpu/rollout_instage_backward_share"] = float(
                                        batch_rollout_backward_overlap_wall / max(1e-9, batch_rollout_wall)
                                    )
                                    wandb_payload["pg_gpu/tbptt_stream_merge_guard_flushes"] = int(
                                        batch_tbptt_stream_merge_guard_flushes
                                    )
                                    wandb_payload["pg_gpu/tbptt_stream_merge_guard_prefallbacks"] = int(
                                        batch_tbptt_stream_merge_guard_prefallbacks
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
                                if batch_policy_step_transformer_layer_finalize_compiled_ms > 0.0:
                                    wandb_payload["pg_gpu/policy_step_tf_layer_finalize_compiled_share"] = float(
                                        batch_policy_step_transformer_layer_finalize_compiled_ms
                                        / max(1e-9, batch_policy_step_transformer_layer_total_ms)
                                    )
                            if stage_name == "rollout" and batch_pg_loss_signature is not None:
                                wandb_payload["pg_gpu/pg_loss_sig"] = str(batch_pg_loss_signature)
                            wandb.log(wandb_payload)
    
                terminal_phase_suffix = _format_pg_phase_terminal_suffix(
                    batch_terminal_target_mean_value,
                    batch_terminal_target_min_value,
                    batch_terminal_target_max_value,
                    batch_terminal_count_mean_value,
                    batch_terminal_count_min_value,
                    batch_terminal_count_max_value,
                    batch_terminal_bonus_mean_value,
                    batch_terminal_bonus_min_value,
                    batch_terminal_bonus_max_value,
                    batch_terminal_bonus_event_mean_value,
                )

                if batch_oom:
                    skipped_steps += 1
                    stable_batches_for_chunk_growth = 0
                    optimizer.zero_grad(set_to_none=True)
                    grad_accum_steps = 0
                    if should_log_pg_phase:
                        _emit_pg_phase_log_line(
                            f"[pg-phase] epoch={batch_epoch_for_profile} batch={batch}{replay_phase_suffix} "
                            f"rollout_s={batch_rollout_wall:.3f} backward_s={batch_backward_wall:.3f} "
                            f"step_s={batch_step_wall:.3f} backward_calls={batch_backward_calls} "
                            f"chunk={current_rollout_chunk_size} tbptt={current_tbptt_window} status=oom"
                            f"{terminal_phase_suffix}"
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
                    if should_log_pg_phase:
                        _emit_pg_phase_log_line(
                            f"[pg-phase] epoch={batch_epoch_for_profile} batch={batch}{replay_phase_suffix} "
                            f"rollout_s={batch_rollout_wall:.3f} backward_s={batch_backward_wall:.3f} "
                            f"step_s={batch_step_wall:.3f} backward_calls={batch_backward_calls} "
                            f"chunk={current_rollout_chunk_size} tbptt={current_tbptt_window} status=loss_nonfinite"
                            f"{terminal_phase_suffix}"
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
                    if should_log_pg_phase:
                        _emit_pg_phase_log_line(
                            f"[pg-phase] epoch={batch_epoch_for_profile} batch={batch}{replay_phase_suffix} "
                            f"rollout_s={batch_rollout_wall:.3f} backward_s={batch_backward_wall:.3f} "
                            f"step_s={batch_step_wall:.3f} backward_calls={batch_backward_calls} "
                            f"chunk={current_rollout_chunk_size} tbptt={current_tbptt_window} status=grad_nonfinite"
                            f"{terminal_phase_suffix}"
                            f"{rollout_breakdown_suffix}"
                        )
                    _emit_nonfinite_gradient_diagnostics(model, epoch=batch_epoch_for_profile, batch=batch)
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
                        grad_stats = _compute_gradient_observability(
                            model,
                            device=device,
                            zero_tol=grad_zero_tol,
                        )
                        nonfinite_grad_diag = None
                        if isinstance(grad_stats, dict) and int(grad_stats.get("grad_nonfinite", 0)) > 0:
                            nonfinite_grad_diag = _summarize_nonfinite_gradient_tensors(model, topk=8)
                        if isinstance(grad_stats, dict):
                            batch_grad_numel_value = int(grad_stats.get("grad_numel", 0))
                            batch_grad_nonfinite_count = int(grad_stats.get("grad_nonfinite", 0))
                            batch_grad_nonfinite_share = float(grad_stats.get("grad_nonfinite_share", 0.0))
                            batch_grad_zero_share = float(grad_stats.get("grad_zero_share", 0.0))
                            batch_grad_abs_mean_value = float(grad_stats.get("grad_abs_mean", 0.0))
                            batch_grad_abs_max_value = float(grad_stats.get("grad_abs_max", 0.0))
                            batch_grad_rms_value = float(grad_stats.get("grad_rms", 0.0))
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., foreach=True)
                    try:
                        batch_grad_norm_value = float(grad_norm.detach().float().cpu())
                    except Exception:
                        try:
                            batch_grad_norm_value = float(grad_norm)
                        except Exception:
                            batch_grad_norm_value = None
                    if batch_grad_norm_value is not None and math.isfinite(float(batch_grad_norm_value)):
                        batch_grad_norm_postclip_value = float(min(float(batch_grad_norm_value), 1.0))
                    if not torch.isfinite(grad_norm):
                        grad_abs_mean_info = (
                            "na"
                            if batch_grad_abs_mean_value is None
                            else f"{float(batch_grad_abs_mean_value):.3e}"
                        )
                        grad_abs_max_info = (
                            "na"
                            if batch_grad_abs_max_value is None
                            else f"{float(batch_grad_abs_max_value):.3e}"
                        )
                        grad_rms_info = (
                            "na"
                            if batch_grad_rms_value is None
                            else f"{float(batch_grad_rms_value):.3e}"
                        )
                        grad_nonfinite_count_info = (
                            "na"
                            if batch_grad_nonfinite_count is None
                            else f"{int(batch_grad_nonfinite_count)}"
                        )
                        grad_nonfinite_share_info = (
                            "na"
                            if batch_grad_nonfinite_share is None
                            else f"{float(batch_grad_nonfinite_share):.3e}"
                        )
                        grad_numel_info = (
                            "na"
                            if batch_grad_numel_value is None
                            else f"{int(batch_grad_numel_value)}"
                        )
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
                        if should_log_pg_phase:
                            _emit_pg_phase_log_line(
                                f"[pg-phase] epoch={batch_epoch_for_profile} batch={batch}{replay_phase_suffix} "
                                f"rollout_s={batch_rollout_wall:.3f} backward_s={batch_backward_wall:.3f} "
                                f"step_s={batch_step_wall:.3f} backward_calls={batch_backward_calls} "
                                f"chunk={current_rollout_chunk_size} tbptt={current_tbptt_window} status=grad_norm_nonfinite"
                                f"{terminal_phase_suffix}"
                                f" grad_abs_mean={grad_abs_mean_info}"
                                f" grad_abs_max={grad_abs_max_info}"
                                f" grad_rms={grad_rms_info}"
                                f" grad_nonfinite={grad_nonfinite_count_info}"
                                f" grad_nonfinite_share={grad_nonfinite_share_info}"
                                f" grad_numel={grad_numel_info}"
                                f"{rollout_breakdown_suffix}"
                            )
                        if nonfinite_grad_diag is not None:
                            for rank, item in enumerate(nonfinite_grad_diag.get("params", []), start=1):
                                shape = 'x'.join(str(int(v)) for v in item.get('shape', ()))
                                print(
                                    f"[grad-nonfinite-top] epoch={batch_epoch_for_profile} batch={batch} rank={rank} "
                                    f"name={item['name']} shape={shape} nonfinite={item['nonfinite_count']} "
                                    f"numel={item['numel']} share={item['share']:.3e} "
                                    f"finite_abs_max={item['finite_abs_max']:.3e} "
                                    f"finite_abs_mean={item['finite_abs_mean']:.3e}"
                                )
                            for rank, item in enumerate(nonfinite_grad_diag.get("prefixes", []), start=1):
                                print(
                                    f"[grad-nonfinite-prefix] epoch={batch_epoch_for_profile} batch={batch} rank={rank} "
                                    f"prefix={item['prefix']} nonfinite={item['nonfinite_count']} "
                                    f"numel={item['numel']} share={item['share']:.3e}"
                                )
                        else:
                            _emit_nonfinite_gradient_diagnostics(model, epoch=batch_epoch_for_profile, batch=batch)
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
                try:
                    batch_objective_value = float(batch_objective.detach().cpu())
                    if math.isfinite(batch_objective_value):
                        pg_epoch_objective_values.append(batch_objective_value)
                except Exception:
                    pass
                try:
                    batch_reward_mean_value = float(batch_reward_mean.detach().cpu())
                    if math.isfinite(batch_reward_mean_value):
                        pg_epoch_reward_mean_values.append(batch_reward_mean_value)
                except Exception:
                    pass
                try:
                    batch_reward_std_value = float(batch_reward_std.detach().cpu())
                    if math.isfinite(batch_reward_std_value):
                        pg_epoch_reward_std_values.append(batch_reward_std_value)
                except Exception:
                    pass
                if batch_reward_env_mean_value is not None and math.isfinite(float(batch_reward_env_mean_value)):
                    pg_epoch_reward_env_mean_values.append(float(batch_reward_env_mean_value))
                if batch_reward_env_std_value is not None and math.isfinite(float(batch_reward_env_std_value)):
                    pg_epoch_reward_env_std_values.append(float(batch_reward_env_std_value))
                if batch_reward_env_return_value is not None and math.isfinite(float(batch_reward_env_return_value)):
                    pg_epoch_reward_env_return_values.append(float(batch_reward_env_return_value))
                if batch_reward_ctrl_mean_value is not None and math.isfinite(float(batch_reward_ctrl_mean_value)):
                    pg_epoch_reward_ctrl_mean_values.append(float(batch_reward_ctrl_mean_value))
                if batch_reward_ctrl_std_value is not None and math.isfinite(float(batch_reward_ctrl_std_value)):
                    pg_epoch_reward_ctrl_std_values.append(float(batch_reward_ctrl_std_value))
                if batch_reward_ctrl_return_value is not None and math.isfinite(float(batch_reward_ctrl_return_value)):
                    pg_epoch_reward_ctrl_return_values.append(float(batch_reward_ctrl_return_value))
                if math.isfinite(batch_reward_absmax_value):
                    pg_epoch_reward_absmax_values.append(float(batch_reward_absmax_value))
                if math.isfinite(batch_reward_clip_hit_share):
                    pg_epoch_clip_hit_values.append(float(batch_reward_clip_hit_share))
                if math.isfinite(batch_reward_norm_clip_hit_share):
                    pg_epoch_norm_clip_hit_values.append(float(batch_reward_norm_clip_hit_share))
                if batch_grad_norm_value is not None and math.isfinite(float(batch_grad_norm_value)):
                    grad_norm_float = float(batch_grad_norm_value)
                    pg_epoch_grad_norm_values.append(grad_norm_float)
                    try:
                        lr_now = float(optimizer.param_groups[0].get("lr", float("nan")))
                    except Exception:
                        lr_now = float("nan")
                    if math.isfinite(lr_now):
                        pg_epoch_lr_step_proxy_values.append(abs(lr_now * grad_norm_float))
                if batch_grad_abs_mean_value is not None and math.isfinite(float(batch_grad_abs_mean_value)):
                    pg_epoch_grad_abs_mean_values.append(float(batch_grad_abs_mean_value))
                if batch_grad_abs_max_value is not None and math.isfinite(float(batch_grad_abs_max_value)):
                    pg_epoch_grad_abs_max_values.append(float(batch_grad_abs_max_value))
                if batch_grad_rms_value is not None and math.isfinite(float(batch_grad_rms_value)):
                    pg_epoch_grad_rms_values.append(float(batch_grad_rms_value))
                if batch_grad_nonfinite_share is not None and math.isfinite(float(batch_grad_nonfinite_share)):
                    pg_epoch_grad_nonfinite_share_values.append(float(batch_grad_nonfinite_share))
                if batch_grad_zero_share is not None and math.isfinite(float(batch_grad_zero_share)):
                    pg_epoch_grad_zero_share_values.append(float(batch_grad_zero_share))
                if math.isfinite(batch_rollout_wall):
                    pg_epoch_rollout_wall_values.append(float(batch_rollout_wall))
                if math.isfinite(batch_backward_wall):
                    pg_epoch_backward_wall_values.append(float(batch_backward_wall))
                if math.isfinite(batch_step_wall):
                    pg_epoch_step_wall_values.append(float(batch_step_wall))
                if (
                    batch_policy_total_loss_value is not None
                    and math.isfinite(float(batch_policy_total_loss_value))
                ):
                    pg_epoch_policy_total_loss_values.append(float(batch_policy_total_loss_value))
                if batch_reinforce_loss_value is not None and math.isfinite(float(batch_reinforce_loss_value)):
                    pg_epoch_reinforce_loss_values.append(float(batch_reinforce_loss_value))
                if (
                    batch_normalized_q_value_loss_value is not None
                    and math.isfinite(float(batch_normalized_q_value_loss_value))
                ):
                    pg_epoch_normalized_q_value_loss_values.append(float(batch_normalized_q_value_loss_value))
                if (
                    batch_next_state_flow_matching_loss_value is not None
                    and math.isfinite(float(batch_next_state_flow_matching_loss_value))
                ):
                    pg_epoch_next_state_flow_matching_loss_values.append(
                        float(batch_next_state_flow_matching_loss_value)
                    )
                if batch_pg_loss_signature is not None:
                    pg_epoch_loss_signatures.add(str(batch_pg_loss_signature))
                if should_log_pg_phase:
                    loss_total_info = (
                        "na"
                        if batch_policy_total_loss_value is None
                        else f"{float(batch_policy_total_loss_value):+.3e}"
                    )
                    loss_reinforce_info = (
                        "na"
                        if batch_reinforce_loss_value is None
                        else f"{float(batch_reinforce_loss_value):+.3e}"
                    )
                    loss_qaux_info = (
                        "na"
                        if batch_normalized_q_value_loss_value is None
                        else f"{float(batch_normalized_q_value_loss_value):+.3e}"
                    )
                    loss_fmaux_info = (
                        "na"
                        if batch_next_state_flow_matching_loss_value is None
                        else f"{float(batch_next_state_flow_matching_loss_value):+.3e}"
                    )
                    reward_env_info = (
                        "na"
                        if batch_reward_env_mean_value is None
                        else f"{float(batch_reward_env_mean_value):+.3e}"
                    )
                    reward_env_std_info = (
                        "na"
                        if batch_reward_env_std_value is None
                        else f"{float(batch_reward_env_std_value):.3e}"
                    )
                    reward_env_return_info = (
                        "na"
                        if batch_reward_env_return_value is None
                        else f"{float(batch_reward_env_return_value):+.3e}"
                    )
                    reward_ctrl_info = (
                        "na"
                        if batch_reward_ctrl_mean_value is None
                        else f"{float(batch_reward_ctrl_mean_value):+.3e}"
                    )
                    reward_ctrl_std_info = (
                        "na"
                        if batch_reward_ctrl_std_value is None
                        else f"{float(batch_reward_ctrl_std_value):.3e}"
                    )
                    reward_ctrl_return_info = (
                        "na"
                        if batch_reward_ctrl_return_value is None
                        else f"{float(batch_reward_ctrl_return_value):+.3e}"
                    )
                    grad_norm_info = "na" if batch_grad_norm_value is None else f"{float(batch_grad_norm_value):.3e}"
                    grad_norm_post_info = (
                        "na"
                        if batch_grad_norm_postclip_value is None
                        else f"{float(batch_grad_norm_postclip_value):.3e}"
                    )
                    grad_abs_mean_info = (
                        "na"
                        if batch_grad_abs_mean_value is None
                        else f"{float(batch_grad_abs_mean_value):.3e}"
                    )
                    grad_abs_max_info = (
                        "na"
                        if batch_grad_abs_max_value is None
                        else f"{float(batch_grad_abs_max_value):.3e}"
                    )
                    grad_rms_info = (
                        "na"
                        if batch_grad_rms_value is None
                        else f"{float(batch_grad_rms_value):.3e}"
                    )
                    grad_nonfinite_count_info = (
                        "na"
                        if batch_grad_nonfinite_count is None
                        else f"{int(batch_grad_nonfinite_count)}"
                    )
                    grad_nonfinite_share_info = (
                        "na"
                        if batch_grad_nonfinite_share is None
                        else f"{float(batch_grad_nonfinite_share):.3e}"
                    )
                    grad_zero_share_info = (
                        "na"
                        if batch_grad_zero_share is None
                        else f"{float(batch_grad_zero_share):.3e}"
                    )
                    grad_numel_info = (
                        "na"
                        if batch_grad_numel_value is None
                        else f"{int(batch_grad_numel_value)}"
                    )
                    opt_step_info = "na" if batch_optimizer_step_value is None else f"{float(batch_optimizer_step_value):.0f}"
                    reward_min_info = "na" if not math.isfinite(batch_reward_min_value) else f"{batch_reward_min_value:+.3e}"
                    reward_max_info = "na" if not math.isfinite(batch_reward_max_value) else f"{batch_reward_max_value:+.3e}"
                    reward_absmax_info = (
                        "na" if not math.isfinite(batch_reward_absmax_value) else f"{batch_reward_absmax_value:.3e}"
                    )
                    _emit_pg_phase_log_line(
                        f"[pg-phase] epoch={batch_epoch_for_profile} batch={batch}{replay_phase_suffix} "
                        f"rollout_s={batch_rollout_wall:.3f} backward_s={batch_backward_wall:.3f} "
                        f"step_s={batch_step_wall:.3f} backward_calls={batch_backward_calls} "
                        f"bridge_count={int(batch_pg_bridge_replay_count_value)} "
                        f"chunk={current_rollout_chunk_size} tbptt={current_tbptt_window} status=ok "
                        f"loss_total={loss_total_info} "
                        f"loss_reinforce={loss_reinforce_info} "
                        f"loss_qaux={loss_qaux_info} "
                        f"loss_fmaux={loss_fmaux_info} "
                        f"objective={float(batch_objective.detach().cpu()):+.3e} "
                        f"reward_mean={float(batch_reward_mean.detach().cpu()):+.3e} "
                        f"reward_env_mean={reward_env_info} "
                        f"reward_env_std={reward_env_std_info} "
                        f"reward_env_return={reward_env_return_info} "
                        f"reward_ctrl_mean={reward_ctrl_info} "
                        f"reward_ctrl_std={reward_ctrl_std_info} "
                        f"reward_ctrl_return={reward_ctrl_return_info} "
                        f"reward_std={float(batch_reward_std.detach().cpu()):.3e} "
                        f"reward_min={reward_min_info} reward_max={reward_max_info} "
                        f"reward_absmax={reward_absmax_info} "
                        f"clip_hit={float(batch_reward_clip_hit_share):.3f} "
                        f"norm_clip_hit={float(batch_reward_norm_clip_hit_share):.3f} "
                        f"grad_norm={grad_norm_info} "
                        f"grad_norm_post={grad_norm_post_info} "
                        f"grad_abs_mean={grad_abs_mean_info} "
                        f"grad_abs_max={grad_abs_max_info} "
                        f"grad_rms={grad_rms_info} "
                        f"grad_nonfinite={grad_nonfinite_count_info} "
                        f"grad_nonfinite_share={grad_nonfinite_share_info} "
                        f"grad_zero_share={grad_zero_share_info} "
                        f"grad_numel={grad_numel_info} "
                        f"stepped={int(bool(batch_optimizer_stepped))} "
                        f"opt_step={opt_step_info}"
                        f"{terminal_phase_suffix}"
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

    def _mean_std(values):
        if not values:
            return None, None
        arr = np.asarray(values, dtype=np.float64)
        return float(arr.mean()), float(arr.std(ddof=0))

    objective_mean, objective_std = _mean_std(pg_epoch_objective_values)
    reward_mean_mean, reward_mean_std = _mean_std(pg_epoch_reward_mean_values)
    reward_std_mean, reward_std_std = _mean_std(pg_epoch_reward_std_values)
    reward_env_mean_mean, reward_env_mean_std = _mean_std(pg_epoch_reward_env_mean_values)
    reward_env_std_mean, reward_env_std_std = _mean_std(pg_epoch_reward_env_std_values)
    reward_env_return_mean, reward_env_return_std = _mean_std(pg_epoch_reward_env_return_values)
    reward_ctrl_mean_mean, reward_ctrl_mean_std = _mean_std(pg_epoch_reward_ctrl_mean_values)
    reward_ctrl_std_mean, reward_ctrl_std_std = _mean_std(pg_epoch_reward_ctrl_std_values)
    reward_ctrl_return_mean, reward_ctrl_return_std = _mean_std(pg_epoch_reward_ctrl_return_values)
    reward_absmax_mean, reward_absmax_std = _mean_std(pg_epoch_reward_absmax_values)
    grad_norm_mean, grad_norm_std = _mean_std(pg_epoch_grad_norm_values)
    grad_abs_mean_mean, grad_abs_mean_std = _mean_std(pg_epoch_grad_abs_mean_values)
    grad_abs_max_mean, grad_abs_max_std = _mean_std(pg_epoch_grad_abs_max_values)
    grad_rms_mean, grad_rms_std = _mean_std(pg_epoch_grad_rms_values)
    grad_nonfinite_share_mean, grad_nonfinite_share_std = _mean_std(pg_epoch_grad_nonfinite_share_values)
    grad_zero_share_mean, grad_zero_share_std = _mean_std(pg_epoch_grad_zero_share_values)
    step_proxy_mean, step_proxy_std = _mean_std(pg_epoch_lr_step_proxy_values)
    rollout_wall_mean, _ = _mean_std(pg_epoch_rollout_wall_values)
    backward_wall_mean, _ = _mean_std(pg_epoch_backward_wall_values)
    step_wall_mean, _ = _mean_std(pg_epoch_step_wall_values)
    policy_total_loss_mean, policy_total_loss_std = _mean_std(pg_epoch_policy_total_loss_values)
    reinforce_loss_mean, reinforce_loss_std = _mean_std(pg_epoch_reinforce_loss_values)
    normalized_q_value_loss_mean, normalized_q_value_loss_std = _mean_std(
        pg_epoch_normalized_q_value_loss_values
    )
    next_state_flow_matching_loss_mean, next_state_flow_matching_loss_std = _mean_std(
        pg_epoch_next_state_flow_matching_loss_values
    )
    clip_hit_mean, _ = _mean_std(pg_epoch_clip_hit_values)
    norm_clip_hit_mean, _ = _mean_std(pg_epoch_norm_clip_hit_values)
    objective_sign_flips = 0
    if len(pg_epoch_objective_values) >= 2:
        objective_sign_flips = sum(
            1
            for prev, cur in zip(pg_epoch_objective_values[:-1], pg_epoch_objective_values[1:])
            if (prev > 0.0 and cur < 0.0) or (prev < 0.0 and cur > 0.0)
        )
    objective_flip_rate = (
        float(objective_sign_flips) / float(max(1, len(pg_epoch_objective_values) - 1))
        if pg_epoch_objective_values
        else None
    )
    objective_jitter_mean_abs_delta = None
    if len(pg_epoch_objective_values) >= 2:
        objective_jitter_mean_abs_delta = float(
            np.mean(np.abs(np.diff(np.asarray(pg_epoch_objective_values, dtype=np.float64))))
        )
    eps_snr = 1e-12
    objective_snr = None
    if objective_mean is not None and objective_std is not None:
        objective_snr = float(abs(objective_mean) / max(eps_snr, objective_std))
    grad_norm_snr = None
    if grad_norm_mean is not None and grad_norm_std is not None:
        grad_norm_snr = float(abs(grad_norm_mean) / max(eps_snr, grad_norm_std))
    if hasattr(model, "module"):
        target_model = model.module
    else:
        target_model = model
    target_model.last_pg_epoch_metrics = {
        "valid_steps": int(valid_steps.item()),
        "skipped_steps": int(skipped_steps),
        "env_replay_steps": int(pg_env_replay_steps),
        "tbptt_window": (None if current_tbptt_window is None else int(current_tbptt_window)),
        "objective_mean": objective_mean,
        "objective_std": objective_std,
        "objective_snr": objective_snr,
        "objective_sign_flip_rate": objective_flip_rate,
        "objective_jitter_mean_abs_delta": objective_jitter_mean_abs_delta,
        "reward_mean_mean": reward_mean_mean,
        "reward_mean_std": reward_mean_std,
        "reward_std_mean": reward_std_mean,
        "reward_std_std": reward_std_std,
        "reward_env_mean_mean": reward_env_mean_mean,
        "reward_env_mean_std": reward_env_mean_std,
        "reward_env_std_mean": reward_env_std_mean,
        "reward_env_std_std": reward_env_std_std,
        "reward_env_return_mean": reward_env_return_mean,
        "reward_env_return_std": reward_env_return_std,
        "reward_ctrl_mean_mean": reward_ctrl_mean_mean,
        "reward_ctrl_mean_std": reward_ctrl_mean_std,
        "reward_ctrl_std_mean": reward_ctrl_std_mean,
        "reward_ctrl_std_std": reward_ctrl_std_std,
        "reward_ctrl_return_mean": reward_ctrl_return_mean,
        "reward_ctrl_return_std": reward_ctrl_return_std,
        "reward_absmax_mean": reward_absmax_mean,
        "reward_absmax_std": reward_absmax_std,
        "clip_hit_mean": clip_hit_mean,
        "norm_clip_hit_mean": norm_clip_hit_mean,
        "grad_norm_mean": grad_norm_mean,
        "grad_norm_std": grad_norm_std,
        "grad_norm_snr": grad_norm_snr,
        "grad_abs_mean_mean": grad_abs_mean_mean,
        "grad_abs_mean_std": grad_abs_mean_std,
        "grad_abs_max_mean": grad_abs_max_mean,
        "grad_abs_max_std": grad_abs_max_std,
        "grad_rms_mean": grad_rms_mean,
        "grad_rms_std": grad_rms_std,
        "grad_nonfinite_share_mean": grad_nonfinite_share_mean,
        "grad_nonfinite_share_std": grad_nonfinite_share_std,
        "grad_zero_share_mean": grad_zero_share_mean,
        "grad_zero_share_std": grad_zero_share_std,
        "lr_grad_step_proxy_mean": step_proxy_mean,
        "lr_grad_step_proxy_std": step_proxy_std,
        "rollout_wall_mean": rollout_wall_mean,
        "backward_wall_mean": backward_wall_mean,
        "step_wall_mean": step_wall_mean,
        "policy_total_loss_mean": policy_total_loss_mean,
        "policy_total_loss_std": policy_total_loss_std,
        "reinforce_loss_mean": reinforce_loss_mean,
        "reinforce_loss_std": reinforce_loss_std,
        "normalized_q_value_loss_mean": normalized_q_value_loss_mean,
        "normalized_q_value_loss_std": normalized_q_value_loss_std,
        "next_state_flow_matching_loss_mean": next_state_flow_matching_loss_mean,
        "next_state_flow_matching_loss_std": next_state_flow_matching_loss_std,
        "pg_loss_signature": ",".join(sorted(pg_epoch_loss_signatures)) if pg_epoch_loss_signatures else None,
    }
    if verbose and target_model.last_pg_epoch_metrics.get("objective_snr") is not None:
        print(
            "[pg-epoch] "
            f"obj_mean={target_model.last_pg_epoch_metrics['objective_mean']:+.3e} "
            f"obj_std={target_model.last_pg_epoch_metrics['objective_std']:.3e} "
            f"obj_snr={target_model.last_pg_epoch_metrics['objective_snr']:.3e} "
            f"reward_env={target_model.last_pg_epoch_metrics['reward_env_mean_mean'] if target_model.last_pg_epoch_metrics['reward_env_mean_mean'] is not None else 'na'} "
            f"reward_env_std={target_model.last_pg_epoch_metrics['reward_env_std_mean'] if target_model.last_pg_epoch_metrics['reward_env_std_mean'] is not None else 'na'} "
            f"reward_env_ret={target_model.last_pg_epoch_metrics['reward_env_return_mean'] if target_model.last_pg_epoch_metrics['reward_env_return_mean'] is not None else 'na'} "
            f"reward_ctrl={target_model.last_pg_epoch_metrics['reward_ctrl_mean_mean'] if target_model.last_pg_epoch_metrics['reward_ctrl_mean_mean'] is not None else 'na'} "
            f"reward_ctrl_std={target_model.last_pg_epoch_metrics['reward_ctrl_std_mean'] if target_model.last_pg_epoch_metrics['reward_ctrl_std_mean'] is not None else 'na'} "
            f"reward_ctrl_ret={target_model.last_pg_epoch_metrics['reward_ctrl_return_mean'] if target_model.last_pg_epoch_metrics['reward_ctrl_return_mean'] is not None else 'na'} "
            f"loss_total={target_model.last_pg_epoch_metrics['policy_total_loss_mean'] if target_model.last_pg_epoch_metrics['policy_total_loss_mean'] is not None else 'na'} "
            f"loss_reinf={target_model.last_pg_epoch_metrics['reinforce_loss_mean'] if target_model.last_pg_epoch_metrics['reinforce_loss_mean'] is not None else 'na'} "
            f"loss_qaux={target_model.last_pg_epoch_metrics['normalized_q_value_loss_mean'] if target_model.last_pg_epoch_metrics['normalized_q_value_loss_mean'] is not None else 'na'} "
            f"loss_fmaux={target_model.last_pg_epoch_metrics['next_state_flow_matching_loss_mean'] if target_model.last_pg_epoch_metrics['next_state_flow_matching_loss_mean'] is not None else 'na'} "
            f"grad_snr={target_model.last_pg_epoch_metrics['grad_norm_snr'] if target_model.last_pg_epoch_metrics['grad_norm_snr'] is not None else 'na'} "
            f"flip_rate={target_model.last_pg_epoch_metrics['objective_sign_flip_rate'] if target_model.last_pg_epoch_metrics['objective_sign_flip_rate'] is not None else 'na'}"
        )

    mean_loss = float((total_loss / valid_steps * aggregate_k_gradients).detach().cpu())
    _restore_policy_inner_recompute_attn(recompute_snapshot)
    return mean_loss, 0.0, 0.0


def train_epoch_official_recurrent_ppo(
    *,
    model,
    ppo_algo,
    ppo_callback,
    ppo_total_timesteps_target: int,
    epoch_idx=None,
):
    rollout_wall_t0 = time.perf_counter()
    continue_training = ppo_algo.collect_rollouts(
        ppo_algo.env,
        ppo_callback,
        ppo_algo.rollout_buffer,
        n_rollout_steps=ppo_algo.n_steps,
    )
    rollout_wall_s = float(time.perf_counter() - rollout_wall_t0)
    if not continue_training:
        raise RuntimeError("Official RecurrentPPO rollout callback requested early stop.")
    ppo_algo._update_current_progress_remaining(ppo_algo.num_timesteps, ppo_total_timesteps_target)
    if epoch_idx is not None:
        ppo_algo.dump_logs(int(epoch_idx))
    train_wall_t0 = time.perf_counter()
    ppo_algo.train()
    train_wall_s = float(time.perf_counter() - train_wall_t0)

    logger_values = dict(getattr(ppo_algo.logger, "name_to_value", {}))

    full_ep_rew_mean = None
    full_ep_len_mean = None
    ep_infos = list(getattr(ppo_algo, "ep_info_buffer", []) or [])
    if len(ep_infos) > 0:
        try:
            full_ep_rew_mean = float(np.mean([float(ep["r"]) for ep in ep_infos if "r" in ep]))
        except Exception:
            full_ep_rew_mean = None
        try:
            full_ep_len_mean = float(np.mean([float(ep["l"]) for ep in ep_infos if "l" in ep]))
        except Exception:
            full_ep_len_mean = None

    target_model = model.module if hasattr(model, "module") else model
    target_model.last_pg_epoch_metrics = None
    reward_component_stats = dict(getattr(ppo_algo, "_rwkv_last_reward_component_stats", {}) or {})
    # Older/fake algos may only expose SB3's full-episode buffer. Keep that as
    # an explicit fallback under full_* names without overriding objective-
    # aligned rollout stats emitted by the strict PPO collector.
    if full_ep_rew_mean is not None:
        reward_component_stats.setdefault("full_ep_rew_mean", float(full_ep_rew_mean))
    if full_ep_len_mean is not None:
        reward_component_stats.setdefault("full_ep_len_mean", float(full_ep_len_mean))
    if reward_component_stats.get("ep_rew_mean", None) is None and full_ep_rew_mean is not None:
        reward_component_stats["ep_rew_mean"] = float(full_ep_rew_mean)
    if reward_component_stats.get("ep_len_mean", None) is None and full_ep_len_mean is not None:
        reward_component_stats["ep_len_mean"] = float(full_ep_len_mean)
    ppo_logger_metrics = dict(logger_values)
    ppo_logger_metrics.setdefault("time/total_timesteps", int(getattr(ppo_algo, "num_timesteps", 0)))
    for key, value in reward_component_stats.items():
        if value is not None:
            ppo_logger_metrics.setdefault(f"rollout/{key}", float(value))
    target_model.last_ppo_logger_metrics = ppo_logger_metrics
    episode_reward_mean = reward_component_stats.get("ep_rew_mean", None)
    episode_length_mean = reward_component_stats.get("ep_len_mean", None)
    target_model.last_ppo_epoch_metrics = {
        "policy_total_loss_mean": logger_values.get("train/loss", None),
        "policy_gradient_loss_mean": logger_values.get("train/policy_gradient_loss", None),
        "value_loss_mean": logger_values.get("train/value_loss", None),
        "entropy_loss_mean": logger_values.get("train/entropy_loss", None),
        "normalized_q_value_loss_mean": logger_values.get("train/normalized_q_value_loss", None),
        "next_state_flow_matching_loss_mean": logger_values.get("train/next_state_flow_matching_loss", None),
        "approx_kl_mean": logger_values.get("train/approx_kl", None),
        "clip_fraction_mean": logger_values.get("train/clip_fraction", None),
        "explained_variance": logger_values.get("train/explained_variance", None),
        "explained_variance_normalized": logger_values.get("train/explained_variance_normalized", None),
        "clip_range": logger_values.get("train/clip_range", None),
        "clip_range_vf": logger_values.get("train/clip_range_vf", None),
        "policy_loss_weight": 1.0,
        "entropy_loss_weight": float(getattr(ppo_algo, "ent_coef", 0.0)),
        "value_loss_weight": float(getattr(ppo_algo, "vf_coef", 0.0)),
        "normalized_q_value_weight": float(getattr(ppo_algo, "_rwkv_aux_q_weight", 0.0)),
        "next_state_flow_matching_weight": float(getattr(ppo_algo, "_rwkv_aux_flow_weight", 0.0)),
        "n_updates": logger_values.get("train/n_updates", None),
        "episode_reward_mean": episode_reward_mean,
        "episode_length_mean": episode_length_mean,
        "full_episode_reward_mean": reward_component_stats.get("full_ep_rew_mean", None),
        "full_episode_length_mean": reward_component_stats.get("full_ep_len_mean", None),
        "num_timesteps": int(getattr(ppo_algo, "num_timesteps", 0)),
        "rollout_wall_time_sec": rollout_wall_s,
        "update_wall_time_sec": train_wall_s,
        "logged_update_wall_time_sec": logger_values.get("train/update_wall_time_sec", None),
        "outer_batches": logger_values.get("train/outer_batches", None),
        "subbatches": logger_values.get("train/subbatches", None),
        "sep_state_reset_enabled": logger_values.get("train/sep_state_reset_enabled", None),
        "sep_state_reset_count": logger_values.get("train/sep_state_reset_count", None),
    }
    for key, value in logger_values.items():
        if str(key).startswith("train/legacy_pre_rollout_"):
            target_model.last_ppo_epoch_metrics[str(key).removeprefix("train/")] = value
    _copy_ppo_train_diagnostic_logger_values(target_model.last_ppo_epoch_metrics, logger_values)
    for key, value in reward_component_stats.items():
        target_model.last_ppo_epoch_metrics[key] = value
    for reward_key in (
        "reward_mean",
        "reward_std",
        "reward_return_mean",
        "reward_return_std",
        "reward_env_mean",
        "reward_env_std",
        "reward_env_return_mean",
        "reward_env_return_std",
        "reward_ctrl_mean",
        "reward_ctrl_std",
        "reward_ctrl_return_mean",
        "reward_ctrl_return_std",
        "reward_survival_mean",
        "reward_survival_std",
        "reward_survival_return_mean",
        "reward_survival_return_std",
        "reward_terminal_bonus_mean",
        "reward_terminal_bonus_std",
        "reward_terminal_bonus_return_mean",
        "reward_terminal_bonus_return_std",
    ):
        target_model.last_ppo_epoch_metrics.setdefault(reward_key, None)
    mean_loss = logger_values.get("train/loss", 0.0)
    return float(mean_loss), 0.0, 0.0


def train_epoch_trusted_pack_recurrent_ppo(
    *,
    model,
    ppo_algo,
    ppo_vec_env,
    ppo_pack_runner_state: dict,
    epoch_idx: int,
    total_epochs: int,
    seed: int,
    fixed_env_group_across_updates: bool,
    semantic_probes_enabled: bool,
    semantic_probe_sidecar_enabled: bool,
    semantic_probe_sidecar_every: int,
    checkpoint_every: int,
    checkpoint_include_optimizer: bool,
):
    from ticl.analysis.phase2_gym_prior_pack_training_runner import _run_pack_update_once

    update_idx = int(epoch_idx) - 1
    env_seed = int(seed) if bool(fixed_env_group_across_updates) else int(seed) + 1000 * int(update_idx)
    wall_t0 = time.perf_counter()
    row, _digest = _run_pack_update_once(
        arm="prior",
        algo=ppo_algo,
        vec_env=ppo_vec_env,
        gym_env_ids_arg="",
        output_dir=ppo_pack_runner_state["output_dir"],
        update_idx=int(update_idx),
        env_seed=int(env_seed),
        fixed_env_group_across_updates=bool(fixed_env_group_across_updates),
        fixed_group_dump=ppo_pack_runner_state.get("fixed_group_dump"),
        device_obj=torch.device(str(ppo_algo.device)),
        num_features=int(ppo_pack_runner_state["num_features"]),
        obs_slot_dim=int(ppo_pack_runner_state["obs_slot_dim"]),
        action_slot_dim=int(ppo_pack_runner_state["action_slot_dim"]),
        next_state_target_dim=int(ppo_pack_runner_state["next_state_target_dim"]),
        n_envs=int(ppo_pack_runner_state["n_envs"]),
        n_steps=int(ppo_pack_runner_state["n_steps"]),
        single_eval_pos=ppo_pack_runner_state.get("single_eval_pos", None),
        terminal_token_enabled=bool(ppo_pack_runner_state["terminal_token_enabled"]),
        semantic_probes_enabled=bool(semantic_probes_enabled),
        semantic_probe_sidecar_enabled=bool(semantic_probe_sidecar_enabled),
        semantic_probe_sidecar_every=int(semantic_probe_sidecar_every),
        checkpoint_every=int(checkpoint_every),
        checkpoint_include_optimizer=bool(checkpoint_include_optimizer),
        force_checkpoint=(int(epoch_idx) == int(total_epochs)),
        progress_path=ppo_pack_runner_state["progress_path"],
    )
    wall_s = float(time.perf_counter() - wall_t0)
    update = dict(row.get("update", {}) or {})
    if update.get("exception") is not None:
        raise RuntimeError(f"Trusted PPO pack update failed: {update.get('exception')}")
    ppo_algo.num_timesteps = int(getattr(ppo_algo, "num_timesteps", 0)) + (
        int(ppo_pack_runner_state["n_envs"]) * int(ppo_pack_runner_state["n_steps"])
    )
    logger_values = dict(update.get("logger", {}) or {})
    target_model = model.module if hasattr(model, "module") else model
    target_model.last_pg_epoch_metrics = None
    ppo_logger_metrics = dict(logger_values)
    ppo_logger_metrics.setdefault("time/total_timesteps", int(getattr(ppo_algo, "num_timesteps", 0)))
    reward_component_stats = dict(row.get("reward_component_summary", {}) or {})
    for key, value in reward_component_stats.items():
        if value is not None:
            try:
                value_f = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value_f):
                ppo_logger_metrics.setdefault(f"rollout/{key}", value_f)
    target_model.last_ppo_logger_metrics = ppo_logger_metrics
    pack_summary = dict(row.get("pack_summary", {}) or {})
    pack_stats = dict(pack_summary.get("stats", {}) or {})
    bridge = dict(row.get("bridge", {}) or {})
    reward_norm = dict(bridge.get("reward_norm", {}) or {})
    raw_reward_norm_stats = dict(reward_norm.get("raw_reward", {}) or {})
    buffer_reward_norm_stats = dict(reward_norm.get("buffer_reward", {}) or {})
    logged_update_wall_time_sec = logger_values.get("train/update_wall_time_sec", None)
    try:
        logged_update_wall_s = float(logged_update_wall_time_sec)
    except (TypeError, ValueError):
        logged_update_wall_s = float("nan")
    rollout_wall_s = None
    if math.isfinite(logged_update_wall_s):
        rollout_wall_s = max(0.0, float(wall_s) - logged_update_wall_s)
    update_wall_s = logged_update_wall_s if math.isfinite(logged_update_wall_s) else float(wall_s)
    target_model.last_ppo_epoch_metrics = {
        "policy_total_loss_mean": logger_values.get("train/loss", None),
        "policy_gradient_loss_mean": logger_values.get("train/policy_gradient_loss", None),
        "value_loss_mean": logger_values.get("train/value_loss", None),
        "entropy_loss_mean": logger_values.get("train/entropy_loss", None),
        "normalized_q_value_loss_mean": logger_values.get("train/normalized_q_value_loss", None),
        "next_state_flow_matching_loss_mean": logger_values.get("train/next_state_flow_matching_loss", None),
        "approx_kl_mean": logger_values.get("train/approx_kl", None),
        "clip_fraction_mean": logger_values.get("train/clip_fraction", None),
        "explained_variance": logger_values.get("train/explained_variance", None),
        "explained_variance_normalized": logger_values.get("train/explained_variance_normalized", None),
        "clip_range": logger_values.get("train/clip_range", None),
        "clip_range_vf": logger_values.get("train/clip_range_vf", None),
        "policy_loss_weight": 1.0,
        "entropy_loss_weight": float(getattr(ppo_algo, "ent_coef", 0.0)),
        "value_loss_weight": float(getattr(ppo_algo, "vf_coef", 0.0)),
        "normalized_q_value_weight": float(getattr(ppo_algo, "_rwkv_aux_q_weight", 0.0)),
        "next_state_flow_matching_weight": float(getattr(ppo_algo, "_rwkv_aux_flow_weight", 0.0)),
        "n_updates": logger_values.get("train/n_updates", update.get("n_updates", None)),
        "num_timesteps": int(getattr(ppo_algo, "num_timesteps", 0)),
        "rollout_wall_time_sec": rollout_wall_s,
        "update_wall_time_sec": update_wall_s,
        "logged_update_wall_time_sec": logged_update_wall_time_sec,
        "pack_wall_time_sec": wall_s,
        "outer_batches": logger_values.get("train/outer_batches", None),
        "subbatches": logger_values.get("train/subbatches", None),
        "pack_raw_reward_mean": raw_reward_norm_stats.get(
            "mean",
            reward_component_stats.get("reward_mean", pack_stats.get("raw_reward_mean")),
        ),
        "pack_buffer_reward_mean": buffer_reward_norm_stats.get("mean", pack_stats.get("reward_mean")),
        "pack_single_eval_pos": row.get("single_eval_pos"),
        "pack_progress_path": str(ppo_pack_runner_state["progress_path"]),
        "reward_norm_enabled": reward_norm.get("enabled", False),
        "episode_reward_mean": reward_component_stats.get("ep_rew_mean", None),
        "episode_length_mean": reward_component_stats.get("ep_len_mean", None),
        "full_episode_reward_mean": reward_component_stats.get("full_ep_rew_mean", None),
        "full_episode_length_mean": reward_component_stats.get("full_ep_len_mean", None),
        "sep_state_reset_enabled": logger_values.get("train/sep_state_reset_enabled", None),
        "sep_state_reset_count": logger_values.get("train/sep_state_reset_count", None),
    }
    for key, value in logger_values.items():
        if str(key).startswith("train/legacy_pre_rollout_"):
            target_model.last_ppo_epoch_metrics[str(key).removeprefix("train/")] = value
    _copy_ppo_train_diagnostic_logger_values(target_model.last_ppo_epoch_metrics, logger_values)
    for key, value in reward_component_stats.items():
        if value is not None:
            target_model.last_ppo_epoch_metrics[key] = value
    target_model.last_ppo_epoch_metrics.setdefault(
        "reward_mean",
        raw_reward_norm_stats.get("mean", pack_stats.get("raw_reward_mean")),
    )
    target_model.last_ppo_epoch_metrics.setdefault(
        "reward_std",
        raw_reward_norm_stats.get("std", pack_stats.get("raw_reward_std")),
    )
    target_model.last_ppo_epoch_metrics.setdefault(
        "reward_return_mean",
        pack_stats.get("raw_reward_return_mean", pack_stats.get("reward_return_mean")),
    )
    target_model.last_ppo_epoch_metrics.setdefault(
        "reward_return_std",
        pack_stats.get("raw_reward_return_std", pack_stats.get("reward_return_std")),
    )
    for reward_key in (
        "reward_mean",
        "reward_std",
        "reward_return_mean",
        "reward_return_std",
        "reward_env_mean",
        "reward_env_std",
        "reward_env_return_mean",
        "reward_env_return_std",
        "reward_ctrl_mean",
        "reward_ctrl_std",
        "reward_ctrl_return_mean",
        "reward_ctrl_return_std",
        "reward_survival_mean",
        "reward_survival_std",
        "reward_survival_return_mean",
        "reward_survival_return_std",
        "reward_terminal_bonus_mean",
        "reward_terminal_bonus_std",
        "reward_terminal_bonus_return_mean",
        "reward_terminal_bonus_return_std",
    ):
        target_model.last_ppo_epoch_metrics.setdefault(reward_key, None)
    mean_loss = logger_values.get("train/loss", 0.0)
    return float(mean_loss), 0.0, 0.0


def _resolve_official_ppo_rollout_shape(
    *,
    batch_size: int,
    n_samples: int,
    configured_n_envs=None,
    configured_n_steps=None,
):
    derived_n_envs = int(max(1, int(batch_size)))
    derived_n_steps = int(max(1, int(n_samples)))
    if configured_n_envs is not None and int(configured_n_envs) != derived_n_envs:
        raise ValueError(
            "Official RecurrentPPO rollout n_envs must match the existing EnvironmentPrior batch_size. "
            f"Expected {derived_n_envs}, got {int(configured_n_envs)}."
        )
    if configured_n_steps is not None and int(configured_n_steps) != derived_n_steps:
        raise ValueError(
            "Official RecurrentPPO rollout n_steps must match the existing EnvironmentPrior n_samples. "
            f"Expected {derived_n_steps}, got {int(configured_n_steps)}."
        )
    return derived_n_envs, derived_n_steps


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
          train_host_rss_limit_gib=None,
          train_host_rss_limit_poll_interval_sec=0.02,
          train_host_rss_limit_try_rlimit_as=False,
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
          pg_saved_tensors_cpu_offload_scope="all",
          pg_saved_tensors_pin_memory=True,
          pg_saved_tensors_cpu_offload_auto_disable_when_safe=False,
          pg_saved_tensors_cpu_offload_auto_min_free_gb=8.0,
          pg_saved_tensors_cpu_offload_auto_max_batch_size=64,
          pg_saved_tensors_cpu_offload_auto_max_n_samples=1024,
          pg_oom_debug_raise=False,
          pg_oom_fail_fast=False,
          pg_kv_cache_mode="auto",
          pg_kv_cache_page_size=None,
          pg_tbptt_window=None,
          pg_env_replay_steps=1,
          pg_oom_reduce_tbptt_first=False,
          pg_torch_compile=False,
          pg_torch_compile_backend="inductor",
          pg_torch_compile_mode="reduce-overhead",
          pg_torch_compile_fullgraph=False,
          pg_torch_compile_dynamic=False,
          pg_compile_observe_recompiles=False,
          pg_compile_observe_log_every_batches=1,
          pg_compile_observe_output_path=None,
          pg_compile_observe_reset_after_warmup=True,
          pg_phase_log_every_batches=1,
          pg_phase_log_file=None,
          anil_inner_steps=1,
          anil_inner_learning_rate=0.1,
          ppo_n_envs=None,
          ppo_n_steps=None,
          ppo_batch_size=None,
          ppo_n_epochs=4,
          ppo_gamma=0.98,
          ppo_gae_lambda=0.90,
          ppo_clip_range=0.2,
          ppo_clip_range_vf=None,
          ppo_normalize_advantage=True,
          ppo_space_contract="raw",
          ppo_value_target_space=None,
          ppo_actor_gae_space=None,
          ppo_allow_mixed_space_contract=False,
          ppo_actor_baseline_mode="learned",
          ppo_separate_value_backbone=False,
          ppo_reset_env_state_at_sep=True,
          ppo_value_head_impl="vendor_official",
          ppo_value_path_adapter_impl="none",
          ppo_value_head_mlp_hidden_dim=256,
          ppo_restore_validation_policy_state=None,
          ppo_strict_fixed_env_mode=False,
          ppo_env_rng_seeds=None,
          ppo_rollout_rng_seeds=None,
          ppo_single_eval_pos=None,
          ppo_deterministic_actor_sampling=False,
          ppo_deterministic_batch_plan=True,
          ppo_strict_native_rollout=False,
          ppo_runtime_normalized_q_value_weight_override=None,
          ppo_runtime_next_state_flow_matching_weight_override=None,
          ppo_ent_coef=0.0,
          ppo_vf_coef=0.1,
          ppo_max_grad_norm=0.5,
          ppo_target_kl=None,
          ppo_trusted_pack_runner_required=False,
          ppo_pack_output_dir=None,
          ppo_pack_seed=4040,
          ppo_pack_prior_mode="sampled_topology",
          ppo_pack_fixed_env_group_across_updates=False,
          ppo_pack_sb3_reward_normalization_enabled=True,
          ppo_pack_sb3_observation_normalization_enabled=True,
          ppo_pack_sb3_observation_normalization_clip=10.0,
          ppo_pack_sb3_observation_normalization_epsilon=1e-8,
          ppo_pack_semantic_probes_enabled=True,
          ppo_pack_semantic_probe_sidecar_enabled=False,
          ppo_pack_semantic_probe_sidecar_every=1,
          ppo_pack_checkpoint_every=0,
          ppo_pack_checkpoint_include_optimizer=False,
          ppo_pack_topology_state_gain_min=0.7,
          ppo_pack_topology_action_gain_min=0.06,
          ppo_pack_topology_state_to_action_ratio_max=10.0,
          ppo_pack_topology_max_attempts=4096,
          ):
    del train_host_rss_limit_gib, train_host_rss_limit_poll_interval_sec, train_host_rss_limit_try_rlimit_as
    using_dist, rank, device = init_dist(device)
    rl_objective = str(rl_objective).strip().lower()
    if rl_objective not in {'supervised', 'policy_gradient', 'first_policy_gradient', 'reinforce', 'alpha_grad', 'anil', 'ppo'}:
        raise ValueError(f"Unknown rl_objective: {rl_objective}")
    if (
        rl_objective in {'policy_gradient', 'first_policy_gradient', 'reinforce', 'alpha_grad', 'anil'}
        and _looks_like_maintained_rlpfn_env_prior(env_prior)
    ):
        raise RuntimeError(
            "Maintained RLPFN policy-gradient/ANIL paths are disabled because they are not "
            "numerically tied to the trusted pack PPO runner and healthy sampled-topology prior. "
            "Use rl_objective='ppo' with ppo_trusted_pack_runner_required=True."
        )
    if rank == 0 and verbose:
        print(f'Using {device} device')

    model.to(device)
    if criterion is not None:
        criterion.to(device)

    policy_tf32_prev = None
    if (
        rl_objective in {'policy_gradient', 'first_policy_gradient', 'reinforce', 'alpha_grad', 'anil'}
        and ("cuda" in str(device))
        and torch.cuda.is_available()
    ):
        policy_tf32_prev = bool(torch.backends.cuda.matmul.allow_tf32)
        policy_tf32_enabled = _env_flag_enabled("TICL_POLICY_TF32", "1")
        if policy_tf32_enabled:
            torch.backends.cuda.matmul.allow_tf32 = True
        if rank == 0 and verbose:
            print(
                "Policy TF32 matmul:",
                bool(policy_tf32_enabled),
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
    if rl_objective in {'policy_gradient', 'first_policy_gradient', 'reinforce', 'alpha_grad', 'anil', 'ppo'}:
        if using_dist:
            raise ValueError(f"{rl_objective} objective does not support distributed training yet.")
        env_prior = _resolve_environment_prior(getattr(dl, "prior", None))
        if env_prior is None:
            raise ValueError(f"{rl_objective} objective requires EnvironmentPrior in dataloader prior chain.")
        if rank == 0 and verbose and rl_objective in {'policy_gradient', 'first_policy_gradient', 'reinforce', 'alpha_grad', 'anil'}:
            print(f"Using {rl_objective} objective with differentiable EnvironmentPrior rollout.")
            env_backend = str(getattr(env_prior, "config", {}).get("batch_parallel_backend", "python_thread"))
            env_grouping = str(getattr(env_prior, "config", {}).get("batch_vectorized_grouping", "structure"))
            env_strict = bool(getattr(env_prior, "config", {}).get("batch_vectorized_strict_rng_match", False))
            print(
                "Environment rollout backend:",
                env_backend,
                f"(grouping={env_grouping}, strict_rng_match={env_strict})",
            )
            transition_fused_env = str(os.environ.get("TICL_POLICY_FUSED_TRANSITION_GENERATOR", "1")).strip().lower()
            transition_fused_on = transition_fused_env not in {"0", "false", "no", "off"}
            transition_stream_env = str(os.environ.get("TICL_POLICY_TRANSITION_STREAM_FUSION", "1")).strip().lower()
            transition_stream_on = transition_stream_env in {"1", "true", "yes", "on"}
            transition_async_commit_env = str(
                os.environ.get("TICL_POLICY_ASYNC_GROUP_COMMIT_IN_STREAM", "auto")
            ).strip().lower()
            if transition_async_commit_env in {"1", "true", "yes", "on"}:
                transition_async_commit_on = True
            elif transition_async_commit_env in {"0", "false", "no", "off"}:
                transition_async_commit_on = False
            else:
                transition_async_commit_on = bool(transition_stream_on)
            print("Policy transition fused generator:", bool(transition_fused_on))
            print("Policy transition stream fusion:", bool(transition_stream_on))
            try:
                transition_stream_max_groups_print = int(
                    max(1, int(os.environ.get("TICL_POLICY_TRANSITION_STREAM_FUSION_MAX_GROUPS", "2")))
                )
            except Exception:
                transition_stream_max_groups_print = 2
            print("Policy transition stream fusion max groups:", int(transition_stream_max_groups_print))
            print(
                "Policy transition async commit in-stream:",
                bool(transition_async_commit_on),
                f"(mode={transition_async_commit_env or 'auto'})",
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
                flash_prefix_async_env = str(os.environ.get("TICL_POLICY_FLASH_PREFIX_ASYNC", "1")).strip().lower()
                flash_prefix_async_on = flash_prefix_async_env in {"1", "true", "yes", "on"}
                print("Policy flash-prefix async:", bool(flash_prefix_async_on))
                try:
                    flashprefix_dense_tokens = int(
                        os.environ.get("TICL_POLICY_PAGED_ATTN_FLASHPREFIX_DENSE_MAX_TOKENS", "64")
                    )
                except Exception:
                    flashprefix_dense_tokens = 64
                print(
                    "Policy flash-prefix dense max tokens:",
                    int(max(0, flashprefix_dense_tokens)),
                )
                try:
                    flashprefix_tail_dense_tokens = int(
                        os.environ.get("TICL_POLICY_FLASH_PREFIX_TAIL_DENSE_MAX_TOKENS", "0")
                    )
                except Exception:
                    flashprefix_tail_dense_tokens = 0
                print(
                    "Policy flash-prefix tail dense max tokens:",
                    int(max(0, flashprefix_tail_dense_tokens)),
                )
                try:
                    dense_page_cap = int(os.environ.get("TICL_POLICY_PAGED_ATTN_DENSE_PAGE_SIZE", "128"))
                except Exception:
                    dense_page_cap = 128
                print("Policy paged-attn dense page cap:", int(max(1, dense_page_cap)))
                tail_freeze_env = str(os.environ.get("TICL_POLICY_TAIL_FREEZE", "1")).strip().lower()
                tail_freeze_on = tail_freeze_env not in {"0", "false", "no", "off"}
                print("Policy paged tail freeze:", bool(tail_freeze_on))
                prefix_compact_env = str(
                    os.environ.get("TICL_POLICY_PREFIX_COMPACT_ON_TBPTT_DETACH", "1")
                ).strip().lower()
                prefix_compact_on = prefix_compact_env not in {"0", "false", "no", "off"}
                print("Policy TBPTT prefix compaction:", bool(prefix_compact_on))
                print("Policy KV cache in-place append:", bool(inplace_kv_print))
            if bool(policy_rollout_checkpoint_reentrant) and not checkpoint_reentrant_active:
                print(
                    "[pg-checkpoint-note] reentrant checkpoint is disabled because the selected mutable KV mode "
                    "is not reentrant-safe; using non-reentrant checkpoint for autograd safety."
                )
            print("Policy saved-tensors CPU offload:", bool(pg_saved_tensors_cpu_offload))
            print("Policy saved-tensors CPU offload scope:", str(pg_saved_tensors_cpu_offload_scope))
            print(
                "Policy saved-tensors CPU offload auto-disable when safe:",
                bool(pg_saved_tensors_cpu_offload_auto_disable_when_safe),
            )
            if bool(pg_saved_tensors_cpu_offload_auto_disable_when_safe):
                print(
                    "Policy saved-tensors CPU offload auto guard:",
                    f"min_free_gb={float(pg_saved_tensors_cpu_offload_auto_min_free_gb):.2f} "
                    f"max_batch={int(pg_saved_tensors_cpu_offload_auto_max_batch_size)} "
                    f"max_n_samples={int(pg_saved_tensors_cpu_offload_auto_max_n_samples)}",
                )
            print("Policy OOM debug re-raise:", bool(pg_oom_debug_raise))
            if pg_tbptt_window is None:
                print("Policy TBPTT window: disabled(full-horizon)")
            else:
                print("Policy TBPTT window:", int(pg_tbptt_window))
                tbptt_store_rewards_env = str(
                    os.environ.get("TICL_POLICY_TBPTT_STORE_REWARDS", "0")
                ).strip().lower()
                tbptt_store_rewards_on = tbptt_store_rewards_env in {"1", "true", "yes", "on"}
                print("Policy TBPTT store rewards tensor:", bool(tbptt_store_rewards_on))
                try:
                    tbptt_merge_windows_print = int(
                        max(1, int(os.environ.get("TICL_POLICY_TBPTT_STREAM_MERGE_WINDOWS", "1")))
                    )
                except Exception:
                    tbptt_merge_windows_print = 1
                print("Policy TBPTT stream merge windows:", int(tbptt_merge_windows_print))
                tbptt_merge_auto_env = str(
                    os.environ.get("TICL_POLICY_TBPTT_STREAM_MERGE_AUTO", "0")
                ).strip().lower()
                tbptt_merge_auto_on_print = tbptt_merge_auto_env in {"1", "true", "yes", "on"}
                print("Policy TBPTT stream merge auto:", bool(tbptt_merge_auto_on_print))
                if bool(tbptt_merge_auto_on_print):
                    try:
                        tbptt_merge_auto_max_print = int(
                            max(1, int(os.environ.get("TICL_POLICY_TBPTT_STREAM_MERGE_AUTO_MAX_WINDOWS", "2")))
                        )
                    except Exception:
                        tbptt_merge_auto_max_print = 2
                    try:
                        tbptt_merge_auto_min_free_print = float(
                            os.environ.get("TICL_POLICY_TBPTT_STREAM_MERGE_AUTO_MIN_FREE_GB", "18.0")
                        )
                    except Exception:
                        tbptt_merge_auto_min_free_print = 18.0
                    try:
                        tbptt_merge_auto_reserved_print = float(
                            os.environ.get("TICL_POLICY_TBPTT_STREAM_MERGE_AUTO_RESERVED_FRAC", "0.72")
                        )
                    except Exception:
                        tbptt_merge_auto_reserved_print = 0.72
                    print(
                        "Policy TBPTT stream merge auto config:",
                        f"max_windows={int(max(1, tbptt_merge_auto_max_print))}",
                        f"min_free_gb={float(max(0.0, tbptt_merge_auto_min_free_print)):.1f}",
                        f"reserved_frac={float(min(0.98, max(0.50, tbptt_merge_auto_reserved_print))):.2f}",
                    )
                tbptt_merge_guard_env = str(
                    os.environ.get("TICL_POLICY_TBPTT_STREAM_MERGE_GUARD", "1")
                ).strip().lower()
                tbptt_merge_guard_on_print = tbptt_merge_guard_env not in {"0", "false", "no", "off"}
                print("Policy TBPTT stream merge guard:", bool(tbptt_merge_guard_on_print))
                if bool(tbptt_merge_guard_on_print):
                    try:
                        tbptt_merge_guard_min_free_print = float(
                            os.environ.get("TICL_POLICY_TBPTT_STREAM_MERGE_GUARD_MIN_FREE_GB", "12.0")
                        )
                    except Exception:
                        tbptt_merge_guard_min_free_print = 12.0
                    try:
                        tbptt_merge_guard_reserved_print = float(
                            os.environ.get("TICL_POLICY_TBPTT_STREAM_MERGE_GUARD_RESERVED_FRAC", "0.82")
                        )
                    except Exception:
                        tbptt_merge_guard_reserved_print = 0.82
                    print(
                        "Policy TBPTT stream merge guard config:",
                        f"min_free_gb={float(max(0.0, tbptt_merge_guard_min_free_print)):.1f}",
                        f"reserved_frac={float(min(0.98, max(0.50, tbptt_merge_guard_reserved_print))):.2f}",
                    )
            try:
                pg_env_replay_steps_print = int(max(1, int(pg_env_replay_steps)))
            except Exception:
                pg_env_replay_steps_print = 1
            print("Policy env replay steps:", pg_env_replay_steps_print)
            envgen_checkpoint_env = str(os.environ.get("TICL_POLICY_ENVGEN_CHECKPOINT", "1")).strip().lower()
            envgen_checkpoint_on = envgen_checkpoint_env not in {"0", "false", "no", "off"}
            envgen_checkpoint_reentrant_env = str(
                os.environ.get("TICL_POLICY_ENVGEN_CHECKPOINT_REENTRANT", "0")
            ).strip().lower()
            print("Policy envgen checkpoint:", bool(envgen_checkpoint_on))
            print(
                "Policy envgen checkpoint reentrant:",
                bool(envgen_checkpoint_reentrant_env in {"1", "true", "yes", "on"}),
            )
            envgen_ragged_affine_env = str(
                os.environ.get("TICL_POLICY_ENVGEN_RAGGED_AFFINE", "0")
            ).strip().lower()
            print(
                "Policy envgen ragged affine:",
                bool(envgen_ragged_affine_env not in {"0", "false", "no", "off"}),
            )
            fused_transition_scm_hidden_fused_env = str(
                os.environ.get("TICL_POLICY_FUSED_TRANSITION_SCM_HIDDEN_FUSED", "0")
            ).strip().lower()
            print(
                "Policy fused transition SCM hidden-fused:",
                bool(fused_transition_scm_hidden_fused_env not in {"0", "false", "no", "off"}),
            )
            fused_transition_gp_input_fused_env = str(
                os.environ.get("TICL_POLICY_FUSED_TRANSITION_GP_INPUT_FUSED", "0")
            ).strip().lower()
            print(
                "Policy fused transition GP input/RFF fused:",
                bool(fused_transition_gp_input_fused_env not in {"0", "false", "no", "off"}),
            )
            fused_transition_gp_output_fused_env = str(
                os.environ.get("TICL_POLICY_FUSED_TRANSITION_GP_OUTPUT_FUSED", "0")
            ).strip().lower()
            print(
                "Policy fused transition GP output projection fused:",
                bool(fused_transition_gp_output_fused_env not in {"0", "false", "no", "off"}),
            )
            fused_transition_gp_output_subgraph_env = str(
                os.environ.get("TICL_POLICY_FUSED_TRANSITION_GP_OUTPUT_SUBGRAPH", "0")
            ).strip().lower()
            print(
                "Policy fused transition GP output subgraph fused:",
                bool(fused_transition_gp_output_subgraph_env not in {"0", "false", "no", "off"}),
            )
            fused_transition_gp_rff_fused_env = str(
                os.environ.get("TICL_POLICY_FUSED_TRANSITION_GP_RFF_FUSED", "0")
            ).strip().lower()
            print(
                "Policy fused transition GP RFF fused:",
                bool(fused_transition_gp_rff_fused_env not in {"0", "false", "no", "off"}),
            )
            fused_transition_gp_packed_env_input_env = str(
                os.environ.get("TICL_POLICY_FUSED_TRANSITION_GP_PACKED_ENV_INPUT", "0")
            ).strip().lower()
            print(
                "Policy fused transition GP packed env input:",
                bool(fused_transition_gp_packed_env_input_env not in {"0", "false", "no", "off"}),
            )
            fused_transition_gp_shared_first_proj_env = str(
                os.environ.get("TICL_POLICY_FUSED_TRANSITION_GP_SHARED_FIRST_PROJ", "0")
            ).strip().lower()
            print(
                "Policy fused transition GP shared first proj:",
                bool(fused_transition_gp_shared_first_proj_env not in {"0", "false", "no", "off"}),
            )
            print(
                "Policy GP RFF tile config:",
                f"block_o={os.environ.get('TICL_POLICY_GP_RFF_BLOCK_O', '32')}"
                f" block_k={os.environ.get('TICL_POLICY_GP_RFF_BLOCK_K', '32')}"
                f" num_warps={os.environ.get('TICL_POLICY_GP_RFF_NUM_WARPS', '4')}",
            )
            profile_gp_projection_timing_env = str(
                os.environ.get("TICL_PROFILE_GP_PROJECTION_TIMING", "0")
            ).strip().lower()
            print(
                "Policy GP projection timing profile:",
                bool(profile_gp_projection_timing_env in {"1", "true", "yes", "on"}),
            )
            print(
                "Policy transition inner grouping:",
                str(getattr(env_prior, "transition_inner_grouping", "family")),
            )
            print(
                "Policy transition inner min bucket:",
                int(getattr(env_prior, "transition_inner_min_bucket", 0) or 0),
            )
            print(
                "Policy reference SCM partition max bytes:",
                int(getattr(env_prior, "reference_scm_partition_max_bytes", 0) or 0),
            )
            print(
                "Policy reference SCM memory guard fraction:",
                float(getattr(env_prior, "reference_scm_memory_guard_fraction", 0.0) or 0.0),
            )
            print(
                "Policy transition generator rebuild each step:",
                bool(getattr(env_prior, "transition_generator_rebuild_each_step", False)),
            )
            try:
                pg_phase_log_every_print = int(max(1, int(pg_phase_log_every_batches)))
            except Exception:
                pg_phase_log_every_print = 1
            print("Policy phase log every batches:", pg_phase_log_every_print)
            if pg_phase_log_file is None or str(pg_phase_log_file).strip() == "":
                print("Policy phase log file: disabled")
            else:
                print("Policy phase log file:", str(pg_phase_log_file))
            print("Policy OOM reduce TBPTT first:", bool(pg_oom_reduce_tbptt_first))
            oom_fail_fast_env = str(os.environ.get("TICL_POLICY_OOM_FAIL_FAST", "0")).strip().lower()
            print(
                "Policy OOM fail-fast:",
                bool(pg_oom_fail_fast) or bool(oom_fail_fast_env in {"1", "true", "yes", "on"}),
            )
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
            model_ref = model.module if hasattr(model, "module") else model
            policy_fastpath_compile_cfg_fn = getattr(model_ref, "get_policy_fastpath_compile_config", None)
            if callable(policy_fastpath_compile_cfg_fn):
                policy_fastpath_compile_cfg = policy_fastpath_compile_cfg_fn()
            else:
                policy_fastpath_compile_cfg = {"finalize_torch_compile": False}
            finalize_torch_compile_on = bool(
                policy_fastpath_compile_cfg.get("finalize_torch_compile", False)
            )
            print("Policy torch.compile:", bool(pg_torch_compile))
            print("Policy finalize torch.compile:", bool(finalize_torch_compile_on))
            if bool(finalize_torch_compile_on):
                print(
                    "Policy finalize torch.compile config:",
                    f"backend={policy_fastpath_compile_cfg.get('backend', 'inductor')}",
                    f"mode={policy_fastpath_compile_cfg.get('mode', 'reduce-overhead')}",
                    f"fullgraph={bool(policy_fastpath_compile_cfg.get('fullgraph', False))}",
                    f"dynamic={bool(policy_fastpath_compile_cfg.get('dynamic', False))}",
                )
            if bool(pg_torch_compile) or bool(finalize_torch_compile_on):
                compile_warmup_env = str(os.environ.get("TICL_POLICY_COMPILE_WARMUP", "1")).strip().lower()
                compile_warmup_on = compile_warmup_env not in {"0", "false", "no", "off"}
                try:
                    compile_warmup_steps_print = int(
                        max(1, int(os.environ.get("TICL_POLICY_COMPILE_WARMUP_STEPS", "1")))
                    )
                except Exception:
                    compile_warmup_steps_print = 1
                try:
                    compile_warmup_samples_print = int(
                        max(2, int(os.environ.get("TICL_POLICY_COMPILE_WARMUP_SAMPLES", "2")))
                    )
                except Exception:
                    compile_warmup_samples_print = 2
                try:
                    compile_warmup_chunk_print = int(
                        max(1, int(os.environ.get("TICL_POLICY_COMPILE_WARMUP_CHUNK", str(int(dl.batch_size)))))
                    )
                except Exception:
                    compile_warmup_chunk_print = int(dl.batch_size)
                print(
                    "Policy compile warmup:",
                    bool(compile_warmup_on),
                    f"(steps={int(compile_warmup_steps_print)}, "
                    f"samples={int(compile_warmup_samples_print)}, "
                    f"chunk={int(max(1, min(int(dl.batch_size), compile_warmup_chunk_print)))})",
                )
                print(
                    "Policy compile recompile observability:",
                    bool(pg_compile_observe_recompiles),
                    f"(log_every_batches={int(max(1, pg_compile_observe_log_every_batches))}, "
                    f"reset_after_warmup={bool(pg_compile_observe_reset_after_warmup)}, "
                    f"output_path={pg_compile_observe_output_path})",
                )
            try:
                from torch.nn.attention import sdpa_kernel as _sdpa_kernel_probe  # noqa: F401
                sdpa_api_available = True
            except Exception:
                sdpa_api_available = False
            print("Policy runtime torch:", str(torch.__version__), f"(sdpa_kernel_api={sdpa_api_available})")
            finalize_fastpath_on = _env_flag_enabled("TICL_POLICY_FINALIZE_2D_FASTPATH", "1")
            print("Policy finalize 2D fastpath:", bool(finalize_fastpath_on))
            finalize_default_fastpath_on = _env_flag_enabled(
                "TICL_POLICY_FINALIZE_2D_ZERO_DROPOUT_POSTNORM_GELU_FASTPATH",
                "0",
            )
            print(
                "Policy finalize 2D zero-dropout/post-norm GELU fastpath:",
                bool(finalize_default_fastpath_on),
            )
            step_proj_2d_on = _env_flag_enabled("TICL_POLICY_STEP_PROJ_2D", "1")
            print("Policy step projection 2D fastpath:", bool(step_proj_2d_on))
            step_layer_2d_on = _env_flag_enabled("TICL_POLICY_STEP_LAYER_2D_LOOP", "1")
            print("Policy step layer 2D loop:", bool(step_layer_2d_on))
            try:
                inplace_paged_page_size_print = int(max(0, int(os.environ.get("TICL_POLICY_INPLACE_PAGED_PAGE_SIZE", "0"))))
            except Exception:
                inplace_paged_page_size_print = 0
            print("Policy inplace paged page size override:", int(inplace_paged_page_size_print))
            token_alloc_opt_on = _env_flag_enabled("TICL_POLICY_STEP_TOKEN_ALLOC_OPT", "1")
            print("Policy step token alloc fastpath:", bool(token_alloc_opt_on))
            token_layout_prepack_on = _env_flag_enabled("TICL_POLICY_TOKEN_LAYOUT_PREPACK", "1")
            print("Policy token layout prepack:", bool(token_layout_prepack_on))
            cache_container_reuse_on = _env_flag_enabled("TICL_POLICY_CACHE_CONTAINER_REUSE", "1")
            print("Policy cache container reuse:", bool(cache_container_reuse_on))
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
        if rank == 0 and verbose and rl_objective == "ppo":
            env_backend = str(getattr(env_prior, "config", {}).get("batch_parallel_backend", "python_thread"))
            reward_transform_mode = str(getattr(env_prior, "config", {}).get("reinforce_reward_transform", "none")).strip().lower()
            action_transform_mode = str(getattr(env_prior, "config", {}).get("reinforce_action_transform", "none")).strip().lower()
            if bool(ppo_trusted_pack_runner_required):
                print("Using trusted canonical PPO pack runner with RWKV official train plumbing.")
            else:
                print("Using direct strict official RecurrentPPO path with RWKV official rollout/train plumbing.")
            print(
                "PPO environment backend:",
                env_backend,
                f"(reward_transform={reward_transform_mode}, action_transform={action_transform_mode})",
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
    optimizer = None
    spike_scheduler = None
    ppo_algo = None
    ppo_callback = None
    ppo_vec_env = None
    ppo_callback_started = False
    ppo_total_timesteps_target = None
    ppo_pack_runner_active = False
    ppo_pack_runner_state = None
    if rl_objective == "ppo":
        from ticl.sb3_recurrent_ppo import (
            build_recurrent_ppo,
            extract_validation_recurrent_ppo_policy_state,
            resolve_official_ppo_update_batch_size,
        )

        if optimizer_state is not None or scheduler is not None:
            raise ValueError(
                "rl_objective='ppo' delegates optimization and scheduling to official sb3-contrib RecurrentPPO; "
                "external optimizer_state/scheduler resume is not supported in this path."
            )
        ppo_batch_envs = int(dl.batch_size)
        ppo_rollout_steps = int(dl.n_samples)
        ppo_num_features = int(dl.num_features)
        if ppo_single_eval_pos is not None:
            fixed_ppo_single_eval_pos = int(max(1, min(int(ppo_rollout_steps) - 1, int(ppo_single_eval_pos))))
            original_sample_single_eval_pos = env_prior._sample_single_eval_pos

            def _sample_fixed_ppo_single_eval_pos(n_samples_arg, single_eval_pos_arg=None):
                del single_eval_pos_arg
                return original_sample_single_eval_pos(int(n_samples_arg), fixed_ppo_single_eval_pos)

            env_prior._sample_single_eval_pos = _sample_fixed_ppo_single_eval_pos
            if rank == 0 and verbose:
                print(
                    "PPO strict compare fixed single_eval_pos:",
                    int(fixed_ppo_single_eval_pos),
                )
        ppo_n_envs, ppo_n_steps = _resolve_official_ppo_rollout_shape(
            batch_size=ppo_batch_envs,
            n_samples=ppo_rollout_steps,
            configured_n_envs=ppo_n_envs,
            configured_n_steps=ppo_n_steps,
        )
        ppo_batch_size = resolve_official_ppo_update_batch_size(
            model=model,
            n_envs=ppo_n_envs,
            n_steps=ppo_n_steps,
            configured_batch_size=ppo_batch_size,
        )
        ppo_pack_runner_active = bool(ppo_trusted_pack_runner_required)
        if rank == 0 and verbose:
            print(
                (
                    "Using trusted canonical PPO pack runner with official RWKV train path "
                    if bool(ppo_pack_runner_active)
                    else "Using strict official RecurrentPPO with official RWKV rollout/train paths "
                )
                + f"(n_envs={int(ppo_n_envs)}, n_steps={int(ppo_n_steps)}, "
                f"batch_size={int(ppo_batch_size)})."
            )
            try:
                ppo_progress_min_interval_print = float(
                    os.environ.get("TICL_PPO_PHASE_LOG_MIN_INTERVAL_SEC", "10.0")
                )
            except Exception:
                ppo_progress_min_interval_print = 10.0
            print(
                "PPO phase progress logging:",
                f"every_batches={int(max(1, int(pg_phase_log_every_batches)))}",
                f"min_interval_sec={float(max(0.0, ppo_progress_min_interval_print)):.1f}",
                f"log_file={pg_phase_log_file if (pg_phase_log_file is not None and str(pg_phase_log_file).strip() != '') else 'disabled'}",
            )
        if bool(ppo_pack_runner_active):
            if ppo_pack_output_dir is None or str(ppo_pack_output_dir).strip() == "":
                raise RuntimeError(
                    "Trusted RLPFN PPO requires ppo_pack_output_dir so pack progress, sidecars, and "
                    "checkpoints cannot silently disappear. fit_model fills this automatically."
                )
            if str(ppo_space_contract).strip().lower() != "raw":
                raise RuntimeError("Trusted RLPFN PPO pack runner requires ppo_space_contract='raw'.")
            if ppo_clip_range_vf is not None:
                raise RuntimeError("Trusted RLPFN PPO pack runner requires ppo_clip_range_vf=None.")
            if ppo_value_target_space is not None and str(ppo_value_target_space).strip().lower() != "raw":
                raise RuntimeError("Trusted RLPFN PPO pack runner requires value_target_space None/raw.")
            if ppo_actor_gae_space is not None and str(ppo_actor_gae_space).strip().lower() != "raw":
                raise RuntimeError("Trusted RLPFN PPO pack runner requires actor_gae_space None/raw.")
            if str(ppo_actor_baseline_mode).strip().lower() != "learned":
                raise RuntimeError("Trusted RLPFN PPO pack runner requires actor_baseline_mode='learned'.")
            if bool(ppo_allow_mixed_space_contract):
                raise RuntimeError("Trusted RLPFN PPO pack runner forbids mixed PPO space contracts.")
            if str(ppo_value_head_impl).strip().lower() != "vendor_official":
                raise RuntimeError("Trusted RLPFN PPO pack runner requires vendor_official value head.")
            if str(ppo_value_path_adapter_impl).strip().lower() != "none":
                raise RuntimeError("Trusted RLPFN PPO pack runner requires value_path_adapter_impl='none'.")
            if bool(ppo_separate_value_backbone):
                raise RuntimeError("Trusted RLPFN PPO pack runner forbids separate_value_backbone.")
            if not bool(ppo_reset_env_state_at_sep):
                raise RuntimeError("Trusted RLPFN PPO pack runner requires reset_env_state_at_sep=True.")
            if bool(ppo_strict_native_rollout):
                raise RuntimeError("Trusted RLPFN PPO pack runner forbids the direct strict_native_rollout path.")
            if bool(ppo_deterministic_actor_sampling):
                raise RuntimeError("Trusted RLPFN PPO pack runner trains with stochastic actor sampling.")
            if not bool(ppo_deterministic_batch_plan):
                raise RuntimeError("Trusted RLPFN PPO pack runner requires deterministic PPO minibatch ordering.")
            if (
                ppo_runtime_normalized_q_value_weight_override not in (None, 0, 0.0)
                or ppo_runtime_next_state_flow_matching_weight_override not in (None, 0, 0.0)
            ):
                raise RuntimeError("Trusted RLPFN PPO pack runner forbids auxiliary PPO losses.")
            if learning_rate is None:
                raise RuntimeError("Trusted RLPFN PPO pack runner requires an explicit learning_rate.")
            if ppo_env_rng_seeds is not None or ppo_rollout_rng_seeds is not None:
                raise RuntimeError("Trusted RLPFN PPO pack runner uses pack env_seed plumbing, not direct PPO seed specs.")
            import copy
            from pathlib import Path

            from ticl.analysis.phase2_gym_prior_pack_training_runner import (
                _build_algo as _build_trusted_pack_algo,
                _dump_fixed_prior_group_if_available,
            )
            from ticl.rlpfn_maintained_path import resolve_rlpfn_token_layout

            env_cfg_for_pack = copy.deepcopy(getattr(env_prior, "config", {}) or {})
            layout = resolve_rlpfn_token_layout(env_cfg_for_pack, num_features=int(ppo_num_features))
            dim_probe_prior = EnvironmentPrior(copy.deepcopy(env_cfg_for_pack))
            next_state_target_dim = int(
                max(
                    int(dim_probe_prior._resolve_dim_upper_bound(env_cfg_for_pack.get("state_dim", None), default=0)),
                    1,
                )
            )
            pack_cfg = {
                "optimizer": {
                    "ppo_gamma": float(ppo_gamma),
                    "ppo_gae_lambda": float(ppo_gae_lambda),
                    "ppo_clip_range": ppo_clip_range,
                    "ppo_ent_coef": float(ppo_ent_coef),
                    "ppo_max_grad_norm": float(ppo_max_grad_norm),
                }
            }
            ppo_algo, ppo_vec_env = _build_trusted_pack_algo(
                cfg=pack_cfg,
                env_cfg=env_cfg_for_pack,
                model_override=model,
                frozen_h=None,
                frozen_h_list=None,
                frozen_h_list_env_seeds=None,
                prior_mode=str(ppo_pack_prior_mode),
                device_obj=torch.device(str(device)),
                num_features=int(ppo_num_features),
                n_envs=int(ppo_n_envs),
                n_steps=int(ppo_n_steps),
                build_seed=int(ppo_pack_seed),
                fixed_single_eval_pos=None,
                sb3_reward_normalization_enabled=bool(ppo_pack_sb3_reward_normalization_enabled),
                sb3_observation_normalization_enabled=bool(ppo_pack_sb3_observation_normalization_enabled),
                sb3_observation_normalization_clip=float(ppo_pack_sb3_observation_normalization_clip),
                sb3_observation_normalization_epsilon=float(ppo_pack_sb3_observation_normalization_epsilon),
                gain_min=float(ppo_pack_topology_state_gain_min),
                topology_state_gain_min=float(ppo_pack_topology_state_gain_min),
                topology_action_gain_min=float(ppo_pack_topology_action_gain_min),
                topology_state_to_action_ratio_max=float(ppo_pack_topology_state_to_action_ratio_max),
                topology_max_attempts=int(ppo_pack_topology_max_attempts),
                fixed_env_group_across_updates=bool(ppo_pack_fixed_env_group_across_updates),
                ppo_learning_rate=float(learning_rate),
                ppo_batch_size=int(ppo_batch_size),
                ppo_n_epochs=int(ppo_n_epochs),
                ppo_vf_coef=float(ppo_vf_coef),
                ppo_normalize_advantage=bool(ppo_normalize_advantage),
                ppo_target_kl=ppo_target_kl,
            )
            pack_output_dir = Path(str(ppo_pack_output_dir)).expanduser().resolve()
            pack_output_dir.mkdir(parents=True, exist_ok=True)
            ppo_pack_runner_state = {
                "output_dir": pack_output_dir,
                "progress_path": pack_output_dir / "prior_progress.jsonl",
                "fixed_group_dump": _dump_fixed_prior_group_if_available(
                    vec_env=ppo_vec_env,
                    output_dir=pack_output_dir,
                    arm="prior",
                ),
                "num_features": int(ppo_num_features),
                "obs_slot_dim": int(layout["obs_slot_dim"]),
                "action_slot_dim": int(layout["action_slot_dim"]),
                "next_state_target_dim": int(next_state_target_dim),
                "n_envs": int(ppo_n_envs),
                "n_steps": int(ppo_n_steps),
                "single_eval_pos": None if ppo_single_eval_pos is None else int(ppo_single_eval_pos),
                "terminal_token_enabled": bool(layout["terminal_token_enabled"]),
            }
            validation_ppo_policy_state = extract_validation_recurrent_ppo_policy_state(ppo_algo.policy)
            model.__dict__["_validation_sb3_policy_live"] = ppo_algo.policy
            model.__dict__["_validation_ppo_policy_state"] = validation_ppo_policy_state
            module_target = getattr(model, "module", None)
            if module_target is not None:
                module_target.__dict__["_validation_sb3_policy_live"] = ppo_algo.policy
                module_target.__dict__["_validation_ppo_policy_state"] = validation_ppo_policy_state
            start_epoch = 1
        else:
            if bool(ppo_trusted_pack_runner_required):
                raise RuntimeError("Internal error: trusted PPO pack runner was required but not activated.")
            looks_like_maintained_rlpfn = _looks_like_maintained_rlpfn_env_prior(env_prior)
            if bool(looks_like_maintained_rlpfn):
                raise RuntimeError(
                    "Direct RLPFN PPO collect_rollouts is forbidden by default because it bypasses the "
                    "pack trainer obs/reward VecNorm, sidecar, and checkpoint semantics. Use "
                    "ppo_trusted_pack_runner_required=True."
                )
            if rank == 0 and verbose:
                print(
                    "[ppo-direct-warning] direct RecurrentPPO collect_rollouts is not the maintained RLPFN pack path. "
                    "Keep ppo_trusted_pack_runner_required=True for trusted RLPFN training."
                )
            ppo_build_kwargs, ppo_runtime_summary = _resolve_recurrent_ppo_training_bridge_kwargs(
                model=model,
                env_prior=env_prior,
                device=device,
                num_features=ppo_num_features,
                n_envs=int(ppo_n_envs),
                n_steps=int(ppo_n_steps),
                learning_rate=learning_rate,
                batch_size=int(ppo_batch_size),
                n_epochs=int(ppo_n_epochs),
                gamma=float(ppo_gamma),
                gae_lambda=float(ppo_gae_lambda),
                clip_range=ppo_clip_range,
                clip_range_vf=ppo_clip_range_vf,
                normalize_advantage=bool(ppo_normalize_advantage),
                space_contract=str(ppo_space_contract),
                value_target_space=ppo_value_target_space,
                actor_gae_space=ppo_actor_gae_space,
                allow_mixed_space_contract=bool(ppo_allow_mixed_space_contract),
                actor_baseline_mode=str(ppo_actor_baseline_mode),
                separate_value_backbone=bool(ppo_separate_value_backbone),
                reset_env_state_at_sep=bool(ppo_reset_env_state_at_sep),
                value_head_impl=str(ppo_value_head_impl),
                value_path_adapter_impl=str(ppo_value_path_adapter_impl),
                value_head_mlp_hidden_dim=int(ppo_value_head_mlp_hidden_dim),
                runtime_normalized_q_value_weight_override=ppo_runtime_normalized_q_value_weight_override,
                runtime_next_state_flow_matching_weight_override=ppo_runtime_next_state_flow_matching_weight_override,
                ent_coef=float(ppo_ent_coef),
                vf_coef=float(ppo_vf_coef),
                max_grad_norm=float(ppo_max_grad_norm),
                target_kl=ppo_target_kl,
                verbose=int(verbose),
                ppo_restore_validation_policy_state=ppo_restore_validation_policy_state,
                ppo_strict_fixed_env_mode=ppo_strict_fixed_env_mode,
                ppo_env_rng_seeds=ppo_env_rng_seeds,
                ppo_rollout_rng_seeds=ppo_rollout_rng_seeds,
                ppo_deterministic_actor_sampling=ppo_deterministic_actor_sampling,
                ppo_deterministic_batch_plan=ppo_deterministic_batch_plan,
                ppo_strict_native_rollout=ppo_strict_native_rollout,
            )
            if rank == 0 and verbose:
                print(
                    "PPO runtime bridge:",
                    f"restore_validation_policy_state={bool(ppo_runtime_summary['restore_validation_policy_state'])}",
                    f"strict_native_rollout={bool(ppo_runtime_summary['strict_native_rollout'])}",
                    f"strict_fixed_env_mode={bool(ppo_runtime_summary['strict_fixed_env_mode'])}",
                    f"deterministic_actor_sampling={bool(ppo_runtime_summary['deterministic_actor_sampling'])}",
                    f"deterministic_batch_plan={bool(ppo_runtime_summary['deterministic_batch_plan'])}",
                )
            ppo_algo, ppo_callback, ppo_vec_env = build_recurrent_ppo(**ppo_build_kwargs)

            def _attach_validation_ppo_policy_handles(target, *, live_policy, saved_policy_state):
                if target is None:
                    return
                target.__dict__["_validation_sb3_policy_live"] = live_policy
                target.__dict__["_validation_ppo_policy_state"] = saved_policy_state

            validation_ppo_policy_state = extract_validation_recurrent_ppo_policy_state(ppo_algo.policy)
            _attach_validation_ppo_policy_handles(model, live_policy=ppo_algo.policy, saved_policy_state=validation_ppo_policy_state)
            module_target = getattr(model, "module", None)
            if module_target is not None:
                _attach_validation_ppo_policy_handles(
                    module_target,
                    live_policy=ppo_algo.policy,
                    saved_policy_state=validation_ppo_policy_state,
                )
            start_epoch = 1
    else:
        adamw_kwargs = dict(lr=learning_rate, weight_decay=weight_decay, betas=(adam_beta1, 0.999))
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
        if scheduler is None:
            if learning_rate_schedule == 'cosine':
                base_scheduler = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=min_lr)
            elif learning_rate_schedule == 'exponential':
                base_scheduler = ExponentialLR(optimizer, gamma=lr_decay, min_lr=min_lr)
            elif learning_rate_schedule == 'constant':
                base_scheduler = ExponentialLR(optimizer, gamma=1, min_lr=min_lr)
            else:
                raise ValueError(f"Invalid learning rate schedule: {learning_rate_schedule}")
            scheduler = SequentialLR(
                optimizer,
                [LinearLR(optimizer, start_factor=1e-10, end_factor=1, total_iters=warmup_epochs), base_scheduler],
                milestones=[warmup_epochs],
            )
            start_epoch = 1
        else:
            start_epoch = scheduler.last_epoch + 1

        if reduce_lr_on_spike:
            spike_scheduler = ReduceLROnSpike(
                optimizer,
                smoothing=10,
                factor=0.5,
                min_lr=min_lr,
                tolerance=spike_tolerance,
                verbose=True,
            )
    try:
        scaler_device_type = torch.device(str(device)).type
    except Exception:
        scaler_device_type = "cuda" if "cuda" in str(device) else "cpu"
    scaler_autocast_dtype = None
    if train_mixed_precision and scaler_device_type == "cuda":
        scaler_autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = (
        GradScaler("cuda")
        if train_mixed_precision and scaler_device_type == "cuda" and scaler_autocast_dtype == torch.float16
        else None
    )
    if rl_objective == "ppo" and ppo_algo is not None:
        setattr(ppo_algo, "_rwkv_grad_scaler", scaler)
        setattr(ppo_algo, "_rwkv_autocast_dtype", scaler_autocast_dtype)

    # check that everything uses up-to-date APIs
    utils.check_compatibility(dl)

    total_loss = float('inf')
    epoch = start_epoch
    pg_phase_log_file_rank0 = pg_phase_log_file if int(rank) == 0 else None
    if rl_objective == "ppo" and ppo_algo is not None:
        setattr(ppo_algo, "_rwkv_progress_log_enabled", bool(int(rank) == 0 and verbose))
        setattr(ppo_algo, "_rwkv_progress_log_every", int(max(1, int(pg_phase_log_every_batches))))
        try:
            ppo_progress_min_interval_sec = float(
                os.environ.get("TICL_PPO_PHASE_LOG_MIN_INTERVAL_SEC", "10.0")
            )
        except Exception:
            ppo_progress_min_interval_sec = 10.0
        setattr(ppo_algo, "_rwkv_progress_log_min_interval_sec", float(max(0.0, ppo_progress_min_interval_sec)))
        setattr(ppo_algo, "_rwkv_progress_log_file", pg_phase_log_file_rank0)
    if stop_after_epochs is not None:
        epochs = min(epochs, stop_after_epochs)
    if rl_objective == "ppo" and ppo_algo is not None and not bool(ppo_pack_runner_active):
        ppo_total_timesteps_target = int(max(1, epochs)) * int(ppo_algo.n_envs) * int(ppo_algo.n_steps)
        ppo_vec_env = getattr(ppo_algo, "env", None)
        ppo_setup_reset_stub_enabled = False
        if hasattr(ppo_vec_env, "enable_direct_collect_reset_stub"):
            # The RWKV PPO collect path below bypasses VecEnv.step_wait and
            # samples the real exact-SCM rollout batch itself. SB3 still
            # performs a mandatory reset inside _setup_learn(); building the
            # full transition generator there is dead work and can consume
            # tens of GiB at 2048x2048. Limit the stub to this reset only so
            # normal VecEnv reset/action-mask semantics stay intact elsewhere.
            ppo_vec_env.enable_direct_collect_reset_stub(True)
            ppo_setup_reset_stub_enabled = True
        try:
            ppo_total_timesteps_target, ppo_callback = ppo_algo._setup_learn(
                int(ppo_total_timesteps_target),
                callback=ppo_callback,
                reset_num_timesteps=True,
                tb_log_name="ppo",
                progress_bar=False,
            )
        finally:
            if ppo_setup_reset_stub_enabled:
                ppo_vec_env.enable_direct_collect_reset_stub(False)
        ppo_callback.on_training_start(locals(), globals())
        ppo_callback_started = True
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
            
            if rl_objective == 'ppo':
                if bool(ppo_pack_runner_active):
                    if ppo_pack_runner_state is None:
                        raise RuntimeError("Trusted PPO pack runner is active but its runner state was not initialized.")
                    new_loss, nan_share, ignore_share = train_epoch_trusted_pack_recurrent_ppo(
                        model=model,
                        ppo_algo=ppo_algo,
                        ppo_vec_env=ppo_vec_env,
                        ppo_pack_runner_state=ppo_pack_runner_state,
                        epoch_idx=int(epoch),
                        total_epochs=int(epochs),
                        seed=int(ppo_pack_seed),
                        fixed_env_group_across_updates=bool(ppo_pack_fixed_env_group_across_updates),
                        semantic_probes_enabled=bool(ppo_pack_semantic_probes_enabled),
                        semantic_probe_sidecar_enabled=bool(ppo_pack_semantic_probe_sidecar_enabled),
                        semantic_probe_sidecar_every=int(ppo_pack_semantic_probe_sidecar_every),
                        checkpoint_every=int(ppo_pack_checkpoint_every),
                        checkpoint_include_optimizer=bool(ppo_pack_checkpoint_include_optimizer),
                    )
                else:
                    new_loss, nan_share, ignore_share = train_epoch_official_recurrent_ppo(
                        model=model,
                        ppo_algo=ppo_algo,
                        ppo_callback=ppo_callback,
                        ppo_total_timesteps_target=int(ppo_total_timesteps_target),
                        epoch_idx=epoch,
                    )
            elif rl_objective in {'policy_gradient', 'first_policy_gradient', 'reinforce', 'alpha_grad', 'anil'}:
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
                    pg_saved_tensors_cpu_offload_scope=pg_saved_tensors_cpu_offload_scope,
                    pg_saved_tensors_pin_memory=pg_saved_tensors_pin_memory,
                    pg_saved_tensors_cpu_offload_auto_disable_when_safe=pg_saved_tensors_cpu_offload_auto_disable_when_safe,
                    pg_saved_tensors_cpu_offload_auto_min_free_gb=pg_saved_tensors_cpu_offload_auto_min_free_gb,
                    pg_saved_tensors_cpu_offload_auto_max_batch_size=pg_saved_tensors_cpu_offload_auto_max_batch_size,
                    pg_saved_tensors_cpu_offload_auto_max_n_samples=pg_saved_tensors_cpu_offload_auto_max_n_samples,
                    pg_oom_debug_raise=pg_oom_debug_raise,
                    pg_oom_fail_fast=pg_oom_fail_fast,
                    pg_kv_cache_mode=pg_kv_cache_mode,
                    pg_kv_cache_page_size=pg_kv_cache_page_size,
                    pg_tbptt_window=pg_tbptt_window,
                    pg_env_replay_steps=pg_env_replay_steps,
                    pg_oom_reduce_tbptt_first=pg_oom_reduce_tbptt_first,
                    pg_torch_compile=pg_torch_compile,
                    pg_torch_compile_backend=pg_torch_compile_backend,
                    pg_torch_compile_mode=pg_torch_compile_mode,
                    pg_torch_compile_fullgraph=pg_torch_compile_fullgraph,
                    pg_torch_compile_dynamic=pg_torch_compile_dynamic,
                    pg_compile_observe_recompiles=pg_compile_observe_recompiles,
                    pg_compile_observe_log_every_batches=pg_compile_observe_log_every_batches,
                    pg_compile_observe_output_path=pg_compile_observe_output_path,
                    pg_compile_observe_reset_after_warmup=pg_compile_observe_reset_after_warmup,
                    pg_phase_log_every_batches=pg_phase_log_every_batches,
                    pg_phase_log_file=pg_phase_log_file_rank0,
                    anil_inner_steps=anil_inner_steps,
                    anil_inner_learning_rate=anil_inner_learning_rate,
                    epoch_idx=epoch,
                    progress_bar=progress_bar,
                    epoch_profiler=train_profiler,
                    epoch_start_time=epoch_start_time,
                    train_profiler_log_every_batches=train_profiler_log_every_batches,
                    verbose=verbose,
                    gpu_observer=gpu_observer,
                    kernel_profiler=kernel_profiler,
                    rl_objective=rl_objective,
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
            if rl_objective == 'ppo':
                validation_ppo_policy_state = extract_validation_recurrent_ppo_policy_state(ppo_algo.policy)
                model.__dict__["_validation_sb3_policy_live"] = ppo_algo.policy
                model.__dict__["_validation_ppo_policy_state"] = validation_ppo_policy_state
                module_target = getattr(model, "module", None)
                if module_target is not None:
                    module_target.__dict__["_validation_sb3_policy_live"] = ppo_algo.policy
                    module_target.__dict__["_validation_ppo_policy_state"] = validation_ppo_policy_state
                ppo_policy_optimizer = getattr(getattr(ppo_algo, "policy", None), "optimizer", None)
                if ppo_policy_optimizer is None:
                    raise RuntimeError(
                        "PPO path expected an internal SB3 policy optimizer, but none was available."
                    )
                last_lr = float(ppo_policy_optimizer.param_groups[0]["lr"])
            elif spike_scheduler is not None:
                last_lr = spike_scheduler.get_last_lr()[0]
            else:
                last_lr = scheduler.get_last_lr()[0]
            if (
                rl_objective in {'policy_gradient', 'first_policy_gradient', 'reinforce', 'alpha_grad', 'anil'}
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
                if rl_objective in {'policy_gradient', 'first_policy_gradient', 'reinforce', 'alpha_grad', 'anil'}:
                    mean_loss_str = f"{float(total_loss):+.6e}"
                elif rl_objective == 'ppo':
                    mean_loss_str = f"{float(total_loss):+.6e}"
                else:
                    mean_loss_str = f"{float(total_loss):5.4f}"
                print(
                    f'| end of epoch {epoch:3d} | Wallclock time: {train_time[-1]:5.2f}s | GPU time: {train_gpu_time[-1]:5.2f}s | mean loss {mean_loss_str} | ')
                if rl_objective in {'policy_gradient', 'first_policy_gradient', 'reinforce', 'alpha_grad', 'anil'}:
                    pg_diag = getattr(model, "last_pg_epoch_metrics", None)
                    if hasattr(model, "module"):
                        pg_diag = getattr(model.module, "last_pg_epoch_metrics", pg_diag)
                    if isinstance(pg_diag, dict):
                        def _fmt_pg_loss_value(key):
                            value = pg_diag.get(key, None)
                            if value is None:
                                return "na"
                            try:
                                return f"{float(value):+.3e}"
                            except Exception:
                                return "na"

                        print(
                            " pg-losses "
                            f"total { _fmt_pg_loss_value('policy_total_loss_mean') } | "
                            f"reinforce { _fmt_pg_loss_value('reinforce_loss_mean') } | "
                            f"qaux { _fmt_pg_loss_value('normalized_q_value_loss_mean') } | "
                            f"fmaux { _fmt_pg_loss_value('next_state_flow_matching_loss_mean') }"
                        )
                elif rl_objective == 'ppo':
                    ppo_diag = getattr(model, "last_ppo_epoch_metrics", None)
                    if hasattr(model, "module"):
                        ppo_diag = getattr(model.module, "last_ppo_epoch_metrics", ppo_diag)
                    if isinstance(ppo_diag, dict):
                        def _fmt_ppo_value(key):
                            value = ppo_diag.get(key, None)
                            if value is None:
                                return "na"
                            try:
                                return f"{float(value):+.3e}"
                            except Exception:
                                return "na"

                        print(
                            " ppo-losses "
                            f"total { _fmt_ppo_value('policy_total_loss_mean') } | "
                            f"pg { _fmt_ppo_value('policy_gradient_loss_mean') } | "
                            f"value { _fmt_ppo_value('value_loss_mean') } | "
                            f"kl { _fmt_ppo_value('approx_kl_mean') } | "
                            f"clip_frac { _fmt_ppo_value('clip_fraction_mean') }"
                        )
                        print(
                            " ppo-aux "
                            f"qaux { _fmt_ppo_value('normalized_q_value_loss_mean') } | "
                            f"fmaux { _fmt_ppo_value('next_state_flow_matching_loss_mean') }"
                        )
                        print(
                            " ppo-reward "
                            f"reward { _fmt_ppo_value('reward_mean') } | "
                            f"reward_std { _fmt_ppo_value('reward_std') } | "
                            f"reward_ret { _fmt_ppo_value('reward_return_mean') } | "
                            f"ep_rew { _fmt_ppo_value('episode_reward_mean') } | "
                            f"full_ep_rew { _fmt_ppo_value('full_episode_reward_mean') }"
                        )
                        print(
                            " ppo-components "
                            f"env { _fmt_ppo_value('reward_env_mean') } | "
                            f"env_ret { _fmt_ppo_value('reward_env_return_mean') } | "
                            f"ctrl { _fmt_ppo_value('reward_ctrl_mean') } | "
                            f"ctrl_ret { _fmt_ppo_value('reward_ctrl_return_mean') } | "
                            f"surv { _fmt_ppo_value('reward_survival_mean') } | "
                            f"surv_ret { _fmt_ppo_value('reward_survival_return_mean') } | "
                            f"term_bonus { _fmt_ppo_value('reward_terminal_bonus_mean') }"
                        )
                        print(
                            " ppo-weights "
                            f"pg { _fmt_ppo_value('policy_loss_weight') } | "
                            f"ent { _fmt_ppo_value('entropy_loss_weight') } | "
                            f"value { _fmt_ppo_value('value_loss_weight') } | "
                            f"qaux { _fmt_ppo_value('normalized_q_value_weight') } | "
                            f"fmaux { _fmt_ppo_value('next_state_flow_matching_weight') }"
                        )
                        print(
                            " ppo-phases "
                            f"rollout_s { _fmt_ppo_value('rollout_wall_time_sec') } | "
                            f"update_s { _fmt_ppo_value('update_wall_time_sec') } | "
                            f"outer_batches { _fmt_ppo_value('outer_batches') } | "
                            f"subbatches { _fmt_ppo_value('subbatches') } | "
                            f"sep_state_reset { _fmt_ppo_value('sep_state_reset_enabled') } | "
                            f"sep_resets { _fmt_ppo_value('sep_state_reset_count') }"
                        )
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
                rl_objective not in {'policy_gradient', 'first_policy_gradient', 'reinforce', 'alpha_grad', 'anil', 'ppo'}
                and math.isfinite(prev_total_loss)
                and new_loss > 1.5 * prev_total_loss
            ):
                print("LOSS DIVERGED")
                return total_loss, model.to('cpu'), dl, epoch
            
            if rl_objective != 'ppo':
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
        traceback.print_exc()
        raise
    finally:
        if ppo_callback_started and (ppo_callback is not None):
            try:
                ppo_callback.on_training_end()
            except Exception:
                pass
        if ppo_vec_env is not None:
            try:
                ppo_vec_env.close()
            except Exception:
                pass
        if kernel_profiler is not None:
            kernel_profiler.stop()
        if gpu_observer is not None:
            gpu_observer.stop()
        if policy_tf32_prev is not None:
            torch.backends.cuda.matmul.allow_tf32 = bool(policy_tf32_prev)

    if rank == 0:  # trivially true for non-parallel training
        return total_loss, model.to('cpu'), dl, epoch
