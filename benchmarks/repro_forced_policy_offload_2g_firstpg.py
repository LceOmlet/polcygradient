import argparse
import json
import os
import subprocess
import threading
import time
from pathlib import Path

import psutil


def _bool_flag(v: str) -> bool:
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "on"}:
        return True
    if s in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid bool: {v}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-repo", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--conda-env", default="rlpfn")
    parser.add_argument(
        "--objective",
        default="first_policy_gradient",
        choices=["first_policy_gradient", "alpha_grad"],
    )
    parser.add_argument("--checkpoint", type=_bool_flag, default=True)
    parser.add_argument("--reentrant", type=_bool_flag, default=True)
    parser.add_argument("--mutable-kv", type=_bool_flag, default=True)
    parser.add_argument("--tbptt-stream", type=_bool_flag, default=True)
    parser.add_argument("--tbptt-window", type=int, default=32)
    parser.add_argument(
        "--execution-mode",
        choices=["fit_model", "direct"],
        default="fit_model",
        help="fit_model matches train-loop defaults (skyline env, autocast, scaler, optimizer step); direct keeps payload-only chunk loss.",
    )
    # Safety defaults are intentionally conservative because prior attempts
    # have destabilized the host. Override only deliberately.
    parser.add_argument("--gpu-guard-mib", type=int, default=5000)
    parser.add_argument("--gpu-total-guard-mib", type=int, default=6000)
    parser.add_argument("--rss-guard-gb", type=float, default=6.0)
    parser.add_argument("--memory-max-gb", type=float, default=8.0)
    parser.add_argument("--min-available-gb", type=float, default=24.0)
    parser.add_argument("--timeout-sec", type=int, default=180)
    parser.add_argument("--poll-interval-sec", type=float, default=0.25)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-known-risky", action="store_true")
    args = parser.parse_args()

    target_repo = Path(args.target_repo).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    git_rev = "unknown"
    try:
        git_rev = (
            subprocess.check_output(
                ["git", "-C", str(target_repo), "rev-parse", "--short", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            .strip()
            or "unknown"
        )
    except Exception:
        git_rev = "unknown"

    tbptt_window = int(max(1, args.tbptt_window))
    objective = str(args.objective)
    label = (
        f"{git_rev}_{objective}_{args.execution_mode}_ckpt{int(args.checkpoint)}_reent{int(args.reentrant)}_"
        f"mutable{int(args.mutable_kv)}_stream{int(args.tbptt_stream)}"
    )
    log_path = out_dir / f"{label}.log"
    summary_path = out_dir / f"{label}.json"

    if not target_repo.exists():
        summary = {
            "label": label,
            "status": "invalid_target_repo",
            "target_repo": str(target_repo),
        }
        summary_path.write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
        return 2

    # Known-risky combinations are blocked by default.
    # Evidence so far:
    # - outer-checkpoint ckpt=1,reentrant=1,mutable=1 hit >11 GiB GPU process
    #   memory in the direct harness.
    # - ckpt=0 in related fit_model reproductions previously drove memory blow-ups.
    risky_reasons = []
    if args.checkpoint and (not args.tbptt_stream) and args.reentrant and args.mutable_kv:
        risky_reasons.append("known_direct_harness_gpu_blowup")
    if not args.checkpoint:
        risky_reasons.append("checkpoint_disabled_known_high_risk")
    if risky_reasons and not args.allow_known_risky:
        summary = {
            "label": label,
            "status": "blocked_known_risky_combo",
            "git_rev": git_rev,
            "objective": objective,
            "checkpoint": args.checkpoint,
            "reentrant": args.reentrant,
            "mutable_kv": args.mutable_kv,
            "tbptt_stream": args.tbptt_stream,
            "tbptt_window": tbptt_window,
            "reasons": risky_reasons,
            "gpu_guard_mib": args.gpu_guard_mib,
            "gpu_total_guard_mib": args.gpu_total_guard_mib,
            "rss_guard_gb": args.rss_guard_gb,
            "timeout_sec": args.timeout_sec,
            "poll_interval_sec": args.poll_interval_sec,
        }
        summary_path.write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
        return 3

    available_gb = psutil.virtual_memory().available / (1024 ** 3)
    if available_gb < float(args.min_available_gb):
        summary = {
            "label": label,
            "status": "blocked_low_available_memory",
            "git_rev": git_rev,
            "objective": objective,
            "available_gb": available_gb,
            "min_available_gb": float(args.min_available_gb),
            "tbptt_stream": args.tbptt_stream,
            "tbptt_window": tbptt_window,
            "poll_interval_sec": args.poll_interval_sec,
        }
        summary_path.write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
        return 4

    systemd_run = subprocess.run(
        ["bash", "-lc", "command -v systemd-run >/dev/null 2>&1"],
        capture_output=True,
        text=True,
    )
    if systemd_run.returncode != 0:
        summary = {
            "label": label,
            "status": "blocked_missing_systemd_run",
            "git_rev": git_rev,
            "objective": objective,
            "tbptt_stream": args.tbptt_stream,
            "tbptt_window": tbptt_window,
            "poll_interval_sec": args.poll_interval_sec,
        }
        summary_path.write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
        return 5

    inner = f"""
import json, os, sys, time, torch
sys.path.insert(0, {repr(str(target_repo))})
from ticl.fit_model import _apply_rlpfn_skyline_env_defaults
_apply_rlpfn_skyline_env_defaults(['rlpfn'])
from ticl.model_configs import get_model_default_config
from ticl.model_builder import get_model
from ticl.priors.environment_prior import EnvironmentPrior
from ticl.train import (
    _amp_autocast_context,
    _build_policy_step_fn,
    _compute_policy_rollout_chunk_loss,
    _resolve_policy_autocast_dtype,
)

config = get_model_default_config('rlpfn')
config['optimizer']['rl_objective'] = {objective!r}
config['optimizer']['pg_saved_tensors_cpu_offload'] = True
config['optimizer']['pg_saved_tensors_cpu_offload_scope'] = 'policy'
config['optimizer']['pg_saved_tensors_pin_memory'] = False
config['optimizer']['pg_saved_tensors_cpu_offload_auto_disable_when_safe'] = False
config['optimizer']['policy_rollout_checkpoint'] = {bool(args.checkpoint)}
config['optimizer']['policy_rollout_checkpoint_reentrant'] = {bool(args.reentrant)}
config['optimizer']['pg_kv_cache_mode'] = 'paged'
config['optimizer']['pg_kv_cache_page_size'] = 128
config['optimizer']['pg_grad_mutable_kv_cache'] = {bool(args.mutable_kv)}
config['optimizer']['pg_tbptt_window'] = {tbptt_window}
config['optimizer']['pg_oom_fail_fast'] = True
config['dataloader']['batch_size'] = 64
env = config['prior']['environment']
env['reference_scm_partition_max_bytes'] = 2 * 1024 * 1024 * 1024
env['batch_parallel_backend'] = 'torch_vectorized'
env['batch_vectorized_grouping'] = 'family'
env['terminal_reset_enabled'] = False
env['alpha_grad_local_coordinate_enabled'] = False
env['alpha_grad_unit_grad_enabled'] = False
env['first_policy_gradient_action_grad_clip_value'] = 4.0
env['first_policy_gradient_action_grad_clip_norm'] = 0.0
config['transformer']['x_obs_dim'] = (
    int(env['obs_slot_dim'])
    + 2
    + 1  # phase token
    + int(bool(env.get('terminal_reset_enabled', False)))
)

_, model, _, _ = get_model(config, device='cuda', should_train=False, verbose=False)
model = model.to('cuda')
model.train()
prior = EnvironmentPrior(dict(env))
step_fn = _build_policy_step_fn(
    model,
    num_features=int(config['prior']['num_features']),
    max_cache_len=1024,
    kv_cache_mode=config['optimizer']['pg_kv_cache_mode'],
    kv_cache_page_size=config['optimizer']['pg_kv_cache_page_size'],
    allow_grad_mutable_cache=bool(config['optimizer']['pg_grad_mutable_kv_cache']),
    pg_torch_compile=False,
    pg_torch_compile_backend='eager',
    pg_torch_compile_mode='reduce-overhead',
    pg_torch_compile_fullgraph=False,
    pg_torch_compile_dynamic=False,
)
policy_autocast_dtype = _resolve_policy_autocast_dtype('cuda')
torch.cuda.reset_peak_memory_stats()
start = time.time()
streamed_roots = []
optimizer = None
scaler = None

if {args.execution_mode == "fit_model"}:
    adamw_kwargs = dict(
        lr=float(config['optimizer']['learning_rate']),
        weight_decay=float(config['optimizer']['weight_decay']),
        betas=(float(config['optimizer']['adam_beta1']), 0.999),
    )
    try:
        optimizer = torch.optim.AdamW(model.parameters(), fused=True, **adamw_kwargs)
    except Exception:
        optimizer = torch.optim.AdamW(model.parameters(), **adamw_kwargs)
    optimizer.zero_grad(set_to_none=True)
    scaler = torch.amp.GradScaler('cuda')
else:
    model.zero_grad(set_to_none=True)

def _tbptt_sink(loss_root):
    streamed_roots.append(float(loss_root.detach().cpu()))
    if scaler is None:
        loss_root.backward()
    else:
        scaler.scale(loss_root).backward()

with _amp_autocast_context(scaler, device='cuda', dtype=policy_autocast_dtype):
    loss, rollout, stats = _compute_policy_rollout_chunk_loss(
        env_prior=prior,
        policy_step_fn=step_fn,
        batch_size=64,
        n_samples=1024,
        num_features=int(config['prior']['num_features']),
        device='cuda',
        single_eval_pos=697,
        collect_x=False,
        policy_rollout_checkpoint={bool(args.checkpoint)} and (not {bool(args.tbptt_stream)}),
        policy_rollout_checkpoint_reentrant={bool(args.reentrant)},
        pg_saved_tensors_cpu_offload=True,
        pg_saved_tensors_cpu_offload_scope='policy',
        pg_saved_tensors_pin_memory=False,
        pg_saved_tensors_cpu_offload_auto_disable_when_safe=False,
        pg_saved_tensors_cpu_offload_auto_min_free_gb=8.0,
        pg_saved_tensors_cpu_offload_auto_max_batch_size=64,
        pg_saved_tensors_cpu_offload_auto_max_n_samples=1024,
        pg_tbptt_window={tbptt_window},
        tbptt_loss_sink=(_tbptt_sink if {bool(args.tbptt_stream)} else None),
        rl_objective={objective!r},
    )
mid = time.time()
if len(streamed_roots) == 0:
    if scaler is None:
        loss.backward()
    else:
        scaler.scale(loss).backward()
after_backward = time.time()
peak_alloc_after_backward_mib = torch.cuda.max_memory_allocated() / (1024**2)
peak_reserved_after_backward_mib = torch.cuda.max_memory_reserved() / (1024**2)
if optimizer is not None:
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
end = time.time()
print(json.dumps({{
    'payload_status': 'ok',
    'execution_mode': {args.execution_mode!r},
    'rollout_plus_loss_s': mid - start,
    'backward_s': after_backward - mid,
    'step_s': end - after_backward,
    'wall_s': end - start,
    'tbptt_stream': {bool(args.tbptt_stream)},
    'tbptt_window': {tbptt_window},
    'streamed_root_count': int(len(streamed_roots)),
    'peak_alloc_mib': torch.cuda.max_memory_allocated() / (1024**2),
    'peak_reserved_mib': torch.cuda.max_memory_reserved() / (1024**2),
    'peak_alloc_after_backward_mib': float(peak_alloc_after_backward_mib),
    'peak_reserved_after_backward_mib': float(peak_reserved_after_backward_mib),
    'objective_value': float(stats['objective'].detach().cpu()),
    'rl_objective': {objective!r},
    'reward_mean': float(stats['reward_mean'].detach().cpu()),
    'reward_std': float(stats['reward_std'].detach().cpu()),
}}))
"""

    env = os.environ.copy()
    cmd = [
        "systemd-run",
        "--user",
        "--scope",
        "--quiet",
        "-p",
        f"MemoryMax={args.memory_max_gb}G",
        "-p",
        "MemorySwapMax=0",
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        args.conda_env,
        "python",
        "-c",
        inner,
    ]
    if args.dry_run:
        summary = {
            "label": label,
            "status": "dry_run",
            "git_rev": git_rev,
            "objective": objective,
            "command": cmd,
            "gpu_guard_mib": args.gpu_guard_mib,
            "gpu_total_guard_mib": args.gpu_total_guard_mib,
            "rss_guard_gb": args.rss_guard_gb,
            "memory_max_gb": args.memory_max_gb,
            "min_available_gb": args.min_available_gb,
            "timeout_sec": args.timeout_sec,
            "tbptt_stream": args.tbptt_stream,
            "tbptt_window": tbptt_window,
            "poll_interval_sec": args.poll_interval_sec,
        }
        summary_path.write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
        return 0

    proc = subprocess.Popen(
        cmd,
        cwd=str(target_repo),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    root = psutil.Process(proc.pid)
    peak_rss_sum = 0
    peak_gpu_process = 0
    peak_gpu_total = 0
    status = "running"
    start_time = time.time()

    with log_path.open("w") as lf:
        log_lock = threading.Lock()

        def _emit(msg: str) -> None:
            print(msg, flush=True)
            with log_lock:
                lf.write(msg + "\n")
                lf.flush()

        def poll():
            nonlocal peak_rss_sum, peak_gpu_process, peak_gpu_total, status
            rss_guard_bytes = int(args.rss_guard_gb * (1024 ** 3))
            while proc.poll() is None:
                if (time.time() - start_time) > int(args.timeout_sec):
                    status = "timeout_kill"
                    _emit(
                        f"[monitor] label={label} status={status} "
                        f"rss_gib={peak_rss_sum / (1024 ** 3):.3f} "
                        f"gpu_process_mib={peak_gpu_process} gpu_total_mib={peak_gpu_total}"
                    )
                    proc.kill()
                    break
                rss_sum = 0
                process_mib = peak_gpu_process
                total_mib = peak_gpu_total
                try:
                    pids = {proc.pid} | {c.pid for c in root.children(recursive=True)}
                except Exception:
                    pids = {proc.pid}
                try:
                    for pid in pids:
                        try:
                            rss_sum += psutil.Process(pid).memory_info().rss
                        except Exception:
                            pass
                    peak_rss_sum = max(peak_rss_sum, rss_sum)
                    if rss_sum > rss_guard_bytes:
                        status = "rss_guard_kill"
                        _emit(
                            f"[monitor] label={label} status={status} "
                            f"rss_gib={rss_sum / (1024 ** 3):.3f} "
                            f"peak_rss_gib={peak_rss_sum / (1024 ** 3):.3f} "
                            f"gpu_process_mib={peak_gpu_process} peak_gpu_process_mib={peak_gpu_process} "
                            f"gpu_total_mib={peak_gpu_total} peak_gpu_total_mib={peak_gpu_total}"
                        )
                        proc.kill()
                        break
                except Exception:
                    pass
                try:
                    out = subprocess.check_output(
                        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                        text=True,
                        stderr=subprocess.DEVNULL,
                    )
                    totals = subprocess.check_output(
                        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                        text=True,
                        stderr=subprocess.DEVNULL,
                    )
                    total_vals = [int(x.strip().split()[0]) for x in totals.strip().splitlines() if x.strip()]
                    if total_vals:
                        total_mib = max(total_vals)
                        peak_gpu_total = max(peak_gpu_total, max(total_vals))
                        if peak_gpu_total > int(args.gpu_total_guard_mib):
                            status = "gpu_total_guard_kill"
                            _emit(
                                f"[monitor] label={label} status={status} "
                                f"rss_gib={rss_sum / (1024 ** 3):.3f} "
                                f"peak_rss_gib={peak_rss_sum / (1024 ** 3):.3f} "
                                f"gpu_process_mib={peak_gpu_process} peak_gpu_process_mib={peak_gpu_process} "
                                f"gpu_total_mib={total_mib} peak_gpu_total_mib={peak_gpu_total}"
                            )
                            proc.kill()
                            break
                    for line in out.strip().splitlines():
                        if not line.strip():
                            continue
                        pid_s, mem_s = [x.strip() for x in line.split(",")[:2]]
                        if int(pid_s) in pids:
                            process_mib = max(process_mib, int(float(mem_s)))
                            peak_gpu_process = max(peak_gpu_process, int(float(mem_s)))
                    if peak_gpu_process > int(args.gpu_guard_mib):
                        status = "gpu_guard_kill"
                        _emit(
                            f"[monitor] label={label} status={status} "
                            f"rss_gib={rss_sum / (1024 ** 3):.3f} "
                            f"peak_rss_gib={peak_rss_sum / (1024 ** 3):.3f} "
                            f"gpu_process_mib={process_mib} peak_gpu_process_mib={peak_gpu_process} "
                            f"gpu_total_mib={total_mib} peak_gpu_total_mib={peak_gpu_total}"
                        )
                        proc.kill()
                        break
                except Exception:
                    pass
                _emit(
                    f"[monitor] label={label} status=running "
                    f"rss_gib={rss_sum / (1024 ** 3):.3f} "
                    f"peak_rss_gib={peak_rss_sum / (1024 ** 3):.3f} "
                    f"gpu_process_mib={process_mib} peak_gpu_process_mib={peak_gpu_process} "
                    f"gpu_total_mib={total_mib} peak_gpu_total_mib={peak_gpu_total}"
                )
                time.sleep(max(0.05, float(args.poll_interval_sec)))

        t = threading.Thread(target=poll, daemon=True)
        t.start()
        for line in proc.stdout:
            print(line, end="", flush=True)
            with log_lock:
                lf.write(line)
                lf.flush()
        rc = proc.wait()
        t.join(timeout=2)

    payload = {}
    if log_path.exists():
        for line in reversed(log_path.read_text(errors="replace").splitlines()):
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                payload = json.loads(line)
                break

    if status == "running":
        if rc == 0 and payload:
            status = "ok"
        elif payload and payload.get("payload_status") == "ok":
            status = f"payload_ok_exit_{rc}"
        else:
            status = f"exit_{rc}"

    summary = {
        "label": label,
        "git_rev": git_rev,
        "target_repo": str(target_repo),
        "requested_objective": objective,
        "status": status,
        "returncode": rc,
        "peak_gpu_process_mib": peak_gpu_process,
        "peak_gpu_total_mib": peak_gpu_total,
        "peak_rss_sum_gib": peak_rss_sum / (1024 ** 3),
        "log_path": str(log_path),
        "poll_interval_sec": float(args.poll_interval_sec),
        "tbptt_stream": bool(args.tbptt_stream),
        "tbptt_window": tbptt_window,
        "effective_outer_checkpoint": bool(args.checkpoint) and (not bool(args.tbptt_stream)),
    }
    summary.update(payload)
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
