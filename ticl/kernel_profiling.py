import json
import os
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch


@dataclass
class TrainKernelProfilerConfig:
    enabled: bool = False
    output_dir: Optional[str] = None
    wait_steps: int = 1
    warmup_steps: int = 1
    active_steps: int = 3
    repeat_steps: int = 1
    record_shapes: bool = True
    profile_memory: bool = True
    with_stack: bool = False
    with_flops: bool = False
    log_every_batches: int = 0
    export_trace: bool = True
    summary_top_k: int = 20

    def normalized(self) -> "TrainKernelProfilerConfig":
        return TrainKernelProfilerConfig(
            enabled=bool(self.enabled),
            output_dir=self.output_dir,
            wait_steps=int(max(0, self.wait_steps)),
            warmup_steps=int(max(0, self.warmup_steps)),
            active_steps=int(max(1, self.active_steps)),
            repeat_steps=int(max(1, self.repeat_steps)),
            record_shapes=bool(self.record_shapes),
            profile_memory=bool(self.profile_memory),
            with_stack=bool(self.with_stack),
            with_flops=bool(self.with_flops),
            log_every_batches=int(max(0, self.log_every_batches)),
            export_trace=bool(self.export_trace),
            summary_top_k=int(max(1, self.summary_top_k)),
        )


class TrainKernelProfiler:
    """
    Optional torch.profiler wrapper for stage-aware kernel observability.

    - optionally exports traces into `output_dir` via tensorboard_trace_handler;
    - emits lightweight top-k op summaries as JSONL;
    - provides `phase()` context manager to mark rollout/backward/step ranges.
    """

    def __init__(self, config: TrainKernelProfilerConfig, device, worker_name: Optional[str] = None):
        self.config = config.normalized()
        self.device = device
        self.worker_name = worker_name or f"pid{os.getpid()}"
        self._prof = None
        self._active = False
        self._step_count = 0
        self._summary_path = None

    def enabled(self) -> bool:
        if not bool(self.config.enabled):
            return False
        return bool(hasattr(torch, "profiler") and hasattr(torch.profiler, "profile"))

    def start(self) -> None:
        if not self.enabled() or self._active:
            return
        out_dir = self.config.output_dir
        if out_dir is None:
            out_dir = os.path.join("log", "kernel_profile")
        out_dir = str(out_dir)
        os.makedirs(out_dir, exist_ok=True)
        self._summary_path = os.path.join(out_dir, "kernel_profile_summary.jsonl")

        activities = [torch.profiler.ProfilerActivity.CPU]
        if ("cuda" in str(self.device)) and torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        schedule = torch.profiler.schedule(
            wait=int(self.config.wait_steps),
            warmup=int(self.config.warmup_steps),
            active=int(self.config.active_steps),
            repeat=int(self.config.repeat_steps),
        )
        on_trace_ready = None
        if bool(self.config.export_trace):
            on_trace_ready = torch.profiler.tensorboard_trace_handler(
                out_dir,
                worker_name=str(self.worker_name),
            )

        profile_kwargs = {
            "activities": activities,
            "schedule": schedule,
            "record_shapes": bool(self.config.record_shapes),
            "profile_memory": bool(self.config.profile_memory),
            "with_stack": bool(self.config.with_stack),
            "with_flops": bool(self.config.with_flops),
        }
        if on_trace_ready is not None:
            profile_kwargs["on_trace_ready"] = on_trace_ready
        try:
            self._prof = torch.profiler.profile(**profile_kwargs)
        except TypeError:
            # Older torch versions may not accept `with_flops`.
            profile_kwargs.pop("with_flops", None)
            self._prof = torch.profiler.profile(**profile_kwargs)

        self._prof.__enter__()
        self._active = True

    def stop(self) -> None:
        if not self._active:
            return
        try:
            if self._prof is not None:
                self._prof.__exit__(None, None, None)
        finally:
            self._prof = None
            self._active = False

    @contextmanager
    def phase(self, name: str):
        if (not self._active) or self._prof is None:
            with nullcontext():
                yield
            return
        label = str(name)
        use_nvtx = ("cuda" in str(self.device)) and torch.cuda.is_available()
        with torch.profiler.record_function(label):
            if use_nvtx:
                try:
                    torch.cuda.nvtx.range_push(label)
                except Exception:
                    use_nvtx = False
            try:
                yield
            finally:
                if use_nvtx:
                    try:
                        torch.cuda.nvtx.range_pop()
                    except Exception:
                        pass

    @staticmethod
    def _event_metric(event, name: str) -> float:
        v = getattr(event, name, 0.0)
        if v is None:
            return 0.0
        try:
            return float(v)
        except Exception:
            return 0.0

    def _emit_summary(self, *, epoch: int, batch: int) -> Optional[Dict[str, Any]]:
        if (not self._active) or self._prof is None:
            return None
        events = self._prof.key_averages()
        if not events:
            return None
        total_self_cuda_us = sum(self._event_metric(e, "self_cuda_time_total") for e in events)
        total_self_cpu_us = sum(self._event_metric(e, "self_cpu_time_total") for e in events)

        by_cuda = sorted(
            events,
            key=lambda e: self._event_metric(e, "self_cuda_time_total"),
            reverse=True,
        )
        by_cpu = sorted(
            events,
            key=lambda e: self._event_metric(e, "self_cpu_time_total"),
            reverse=True,
        )

        top_k = int(max(1, self.config.summary_top_k))
        top_ops: List[Dict[str, Any]] = []
        for e in by_cuda[:top_k]:
            self_cuda_us = self._event_metric(e, "self_cuda_time_total")
            self_cpu_us = self._event_metric(e, "self_cpu_time_total")
            top_ops.append(
                {
                    "name": str(getattr(e, "key", "unknown")),
                    "count": int(getattr(e, "count", 0)),
                    "self_cuda_time_us": self_cuda_us,
                    "cuda_time_us": self._event_metric(e, "cuda_time_total"),
                    "self_cpu_time_us": self_cpu_us,
                    "cpu_time_us": self._event_metric(e, "cpu_time_total"),
                    "self_cuda_share": (
                        0.0 if total_self_cuda_us <= 0.0 else float(self_cuda_us / total_self_cuda_us)
                    ),
                    "self_cpu_share": (
                        0.0 if total_self_cpu_us <= 0.0 else float(self_cpu_us / total_self_cpu_us)
                    ),
                }
            )

        phase_order = (
            "pg.rollout",
            "pg.backward",
            "pg.step",
            "train.forward",
            "train.backward",
            "train.step",
        )
        phase_events = {str(getattr(e, "key", "")): e for e in events}
        phase_ops: List[Dict[str, Any]] = []
        for name in phase_order:
            event = phase_events.get(name)
            if event is None:
                continue
            phase_ops.append(
                {
                    "name": str(name),
                    "count": int(getattr(event, "count", 0)),
                    "self_cuda_time_us": self._event_metric(event, "self_cuda_time_total"),
                    "cuda_time_us": self._event_metric(event, "cuda_time_total"),
                    "self_cpu_time_us": self._event_metric(event, "self_cpu_time_total"),
                    "cpu_time_us": self._event_metric(event, "cpu_time_total"),
                }
            )

        sync_terms = (
            "synchronize",
            "cudadevicesynchronize",
            "cudastreamsynchronize",
            "cudaeventsynchronize",
            "cudalaunchkernel",
            "cudamemcpy",
        )
        sync_events = []
        for e in events:
            key = str(getattr(e, "key", ""))
            key_lower = key.lower()
            if any(term in key_lower for term in sync_terms):
                sync_events.append(e)
        sync_events = sorted(
            sync_events,
            key=lambda e: self._event_metric(e, "self_cpu_time_total"),
            reverse=True,
        )
        sync_self_cpu_us = 0.0
        sync_count = 0
        sync_top_cpu = []
        for e in sync_events[:10]:
            self_cpu_us = self._event_metric(e, "self_cpu_time_total")
            sync_self_cpu_us += self_cpu_us
            sync_count += int(getattr(e, "count", 0))
            sync_top_cpu.append(
                {
                    "name": str(getattr(e, "key", "unknown")),
                    "count": int(getattr(e, "count", 0)),
                    "self_cpu_time_us": self_cpu_us,
                    "cpu_time_us": self._event_metric(e, "cpu_time_total"),
                }
            )
        top_cuda = by_cuda[0] if by_cuda else None
        top_cpu = by_cpu[0] if by_cpu else None
        rec = {
            "timestamp_unix": float(time.time()),
            "epoch": int(epoch),
            "batch": int(batch),
            "step": int(self._step_count),
            "top_cuda_op": (None if top_cuda is None else str(getattr(top_cuda, "key", "unknown"))),
            "top_cuda_self_time_us": (0.0 if top_cuda is None else self._event_metric(top_cuda, "self_cuda_time_total")),
            "top_cpu_op": (None if top_cpu is None else str(getattr(top_cpu, "key", "unknown"))),
            "top_cpu_self_time_us": (0.0 if top_cpu is None else self._event_metric(top_cpu, "self_cpu_time_total")),
            "total_self_cuda_time_us": float(total_self_cuda_us),
            "total_self_cpu_time_us": float(total_self_cpu_us),
            "phase_ops": phase_ops,
            "sync_top_cpu": sync_top_cpu,
            "sync_self_cpu_time_us": float(sync_self_cpu_us),
            "sync_self_cpu_share": (
                0.0 if total_self_cpu_us <= 0.0 else float(sync_self_cpu_us / total_self_cpu_us)
            ),
            "sync_count": int(sync_count),
            "top_ops": top_ops,
        }
        if self._summary_path is not None:
            with open(self._summary_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=True))
                f.write("\n")
        return rec

    def step(self, *, epoch: int, batch: int) -> Optional[Dict[str, Any]]:
        if (not self._active) or self._prof is None:
            return None
        self._step_count += 1
        self._prof.step()
        every = int(self.config.log_every_batches)
        if every > 0 and (self._step_count % every == 0):
            return self._emit_summary(epoch=int(epoch), batch=int(batch))
        return None
