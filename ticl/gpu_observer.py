import json
import os
import subprocess
import threading
import time
from typing import Dict, List, Optional

import torch

try:
    import pynvml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pynvml = None


def _parse_num(token: str) -> Optional[float]:
    t = str(token).strip()
    if not t or t in {"N/A", "[N/A]", "[Not Supported]"}:
        return None
    try:
        return float(t)
    except Exception:
        return None


def _resolve_cuda_device_index(device) -> Optional[int]:
    try:
        dev = device if isinstance(device, torch.device) else torch.device(str(device))
    except Exception:
        return None
    if dev.type != "cuda":
        return None
    if dev.index is not None:
        return int(dev.index)
    if torch.cuda.is_available():
        return int(torch.cuda.current_device())
    return None


class GPUProcessObserver:
    """
    Best-effort GPU observer for training observability.

    Notes:
    - `gpu_util_percent` is device-level utilization from nvidia-smi.
    - process-level SM/MEM util is sampled from NVML when available, and falls
      back to `nvidia-smi pmon` parsing otherwise.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        device,
        sample_interval_sec: float = 1.0,
        output_path: Optional[str] = None,
        stage_output_path: Optional[str] = None,
        pid: Optional[int] = None,
    ):
        self._enabled = bool(enabled)
        self.device_index = _resolve_cuda_device_index(device)
        if self.device_index is None:
            self._enabled = False
        self.sample_interval_sec = float(max(0.1, sample_interval_sec))
        self.output_path = output_path
        self.stage_output_path = stage_output_path
        self.pid = int(os.getpid() if pid is None else pid)

        self._samples: List[Dict[str, object]] = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = None
        self._warned_sampling_failure = False
        self._nvml_handle = None
        self._nvml_last_ts_ms = 0
        self._process_util_backend = "none"
        self._init_nvml()

    def enabled(self) -> bool:
        return bool(self._enabled)

    def start(self) -> None:
        if not self.enabled():
            return
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._sample_loop,
            name="gpu-process-observer",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=5.0)
        self._thread = None
        if self._nvml_handle is not None and pynvml is not None:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
            self._nvml_handle = None

    def sample_count(self) -> int:
        with self._lock:
            return len(self._samples)

    def _init_nvml(self) -> None:
        if not self.enabled() or pynvml is None:
            return
        try:
            pynvml.nvmlInit()
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(int(self.device_index))
            self._process_util_backend = "nvml"
        except Exception:
            self._nvml_handle = None
            self._process_util_backend = "pmon"

    def _sample_process_util_nvml(self):
        if self._nvml_handle is None or pynvml is None:
            return False, None, None
        try:
            samples = pynvml.nvmlDeviceGetProcessUtilization(
                self._nvml_handle,
                int(self._nvml_last_ts_ms),
            )
        except Exception:
            return False, None, None

        proc_samples = []
        for s in samples:
            try:
                if int(getattr(s, "pid", -1)) == int(self.pid):
                    proc_samples.append(s)
            except Exception:
                continue
        if samples:
            try:
                self._nvml_last_ts_ms = int(max(getattr(s, "timeStamp", 0) for s in samples))
            except Exception:
                pass
        if not proc_samples:
            return True, None, None

        sm_vals = []
        mem_vals = []
        for s in proc_samples:
            sm_v = _parse_num(getattr(s, "smUtil", None))
            mem_v = _parse_num(getattr(s, "memUtil", None))
            if sm_v is not None:
                sm_vals.append(float(sm_v))
            if mem_v is not None:
                mem_vals.append(float(mem_v))
        sm_util = (sum(sm_vals) / len(sm_vals)) if sm_vals else None
        mem_util = (sum(mem_vals) / len(mem_vals)) if mem_vals else None
        return True, sm_util, mem_util

    def _sample_process_util_pmon(self):
        cmd = [
            "nvidia-smi",
            "pmon",
            "-i",
            str(int(self.device_index)),
            "-s",
            "um",
            "-c",
            "1",
        ]
        try:
            out = subprocess.check_output(
                cmd,
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=2.0,
            )
        except Exception:
            return False, None, None

        found = False
        sm_util = None
        mem_util = None
        for line in out.splitlines():
            line = line.strip()
            if (not line) or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            row_gpu = _parse_num(parts[0])
            row_pid = _parse_num(parts[1])
            if row_gpu is None or row_pid is None:
                continue
            if int(row_gpu) != int(self.device_index):
                continue
            if int(row_pid) != int(self.pid):
                continue
            found = True
            sm_util = _parse_num(parts[3])
            mem_util = _parse_num(parts[4])
            break
        return True, (None if not found else sm_util), (None if not found else mem_util)

    def _sample_process_util(self):
        backend = str(self._process_util_backend)
        if backend == "nvml":
            ok, sm, mem = self._sample_process_util_nvml()
            if ok:
                return True, sm, mem, "nvml"
            # NVML transient failure: try pmon before giving up.
            ok2, sm2, mem2 = self._sample_process_util_pmon()
            if ok2:
                self._process_util_backend = "pmon"
                return True, sm2, mem2, "pmon"
            return False, None, None, "none"
        ok, sm, mem = self._sample_process_util_pmon()
        if ok:
            return True, sm, mem, "pmon"
        return False, None, None, "none"

    def _sample_loop(self) -> None:
        while not self._stop_event.is_set():
            sample = self._sample_once()
            if sample is not None:
                with self._lock:
                    self._samples.append(sample)
                self._write_jsonl(self.output_path, sample)
            self._stop_event.wait(self.sample_interval_sec)

    def _sample_once(self) -> Optional[Dict[str, object]]:
        gpu_cmd = [
            "nvidia-smi",
            "--query-gpu=index,uuid,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw",
            "--format=csv,noheader,nounits",
        ]
        proc_cmd = [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ]
        try:
            gpu_out = subprocess.check_output(
                gpu_cmd,
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=2.0,
            )
            proc_out = subprocess.check_output(
                proc_cmd,
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=2.0,
            )
        except Exception:
            if not self._warned_sampling_failure:
                self._warned_sampling_failure = True
                print("[gpu-observer-warn] nvidia-smi sampling failed; observer disabled for this run.")
            self._enabled = False
            return None

        gpu_rows = {}
        for line in gpu_out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 7:
                continue
            idx = _parse_num(parts[0])
            uuid = parts[1]
            if idx is None:
                continue
            gpu_rows[int(idx)] = {
                "uuid": uuid,
                "gpu_util_percent": _parse_num(parts[2]),
                "gpu_mem_util_percent": _parse_num(parts[3]),
                "gpu_mem_used_mib": _parse_num(parts[4]),
                "gpu_mem_total_mib": _parse_num(parts[5]),
                "gpu_power_w": _parse_num(parts[6]),
            }
        gpu_row = gpu_rows.get(int(self.device_index))
        if gpu_row is None:
            return None

        target_uuid = str(gpu_row["uuid"])
        process_mem_mib = 0.0
        process_present = False
        for line in proc_out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            proc_uuid = parts[0]
            proc_pid = _parse_num(parts[1])
            if proc_pid is None:
                continue
            if int(proc_pid) != int(self.pid):
                continue
            if proc_uuid != target_uuid:
                continue
            used = _parse_num(parts[2])
            if used is not None:
                process_mem_mib += float(used)
                process_present = True

        total_mib = gpu_row.get("gpu_mem_total_mib", None)
        if total_mib is not None and total_mib > 0.0:
            process_mem_share_percent = (process_mem_mib / total_mib) * 100.0
        else:
            process_mem_share_percent = None

        proc_util_available, proc_sm_util, proc_mem_util, proc_util_backend = self._sample_process_util()

        sample = {
            "timestamp_unix": float(time.time()),
            "pid": int(self.pid),
            "device_index": int(self.device_index),
            "gpu_uuid": target_uuid,
            "gpu_util_percent": gpu_row.get("gpu_util_percent"),
            "gpu_mem_util_percent": gpu_row.get("gpu_mem_util_percent"),
            "gpu_mem_used_mib": gpu_row.get("gpu_mem_used_mib"),
            "gpu_mem_total_mib": gpu_row.get("gpu_mem_total_mib"),
            "gpu_power_w": gpu_row.get("gpu_power_w"),
            "process_present": bool(process_present),
            "process_mem_mib": float(process_mem_mib),
            "process_mem_share_percent": process_mem_share_percent,
            "process_util_available": bool(proc_util_available),
            "process_util_backend": str(proc_util_backend),
            "process_sm_util_percent": (None if proc_sm_util is None else float(proc_sm_util)),
            "process_mem_util_percent": (None if proc_mem_util is None else float(proc_mem_util)),
        }
        return sample

    def window_stats(self, start_time_unix: float, end_time_unix: float) -> Dict[str, object]:
        t0 = float(min(start_time_unix, end_time_unix))
        t1 = float(max(start_time_unix, end_time_unix))
        with self._lock:
            window = [
                s for s in self._samples
                if t0 <= float(s["timestamp_unix"]) <= t1
            ]
        if not window:
            return {
                "samples": 0,
                "gpu_util_avg": None,
                "gpu_util_max": None,
                "gpu_mem_util_avg": None,
                "gpu_power_avg_w": None,
                "process_mem_avg_mib": None,
                "process_mem_max_mib": None,
                "process_mem_share_avg": None,
                "process_sm_util_avg": None,
                "process_sm_util_max": None,
                "process_mem_util_avg": None,
                "process_mem_util_max": None,
                "process_util_available": None,
            }

        def _nums(key):
            vals = [s.get(key) for s in window]
            return [float(v) for v in vals if v is not None]

        gpu_util = _nums("gpu_util_percent")
        gpu_mem_util = _nums("gpu_mem_util_percent")
        gpu_power = _nums("gpu_power_w")
        process_mem = _nums("process_mem_mib")
        process_mem_share = _nums("process_mem_share_percent")
        process_sm_util = _nums("process_sm_util_percent")
        process_mem_util = _nums("process_mem_util_percent")
        process_util_available = [bool(s.get("process_util_available", False)) for s in window]

        return {
            "samples": int(len(window)),
            "gpu_util_avg": (sum(gpu_util) / len(gpu_util)) if gpu_util else None,
            "gpu_util_max": max(gpu_util) if gpu_util else None,
            "gpu_mem_util_avg": (sum(gpu_mem_util) / len(gpu_mem_util)) if gpu_mem_util else None,
            "gpu_power_avg_w": (sum(gpu_power) / len(gpu_power)) if gpu_power else None,
            "process_mem_avg_mib": (sum(process_mem) / len(process_mem)) if process_mem else None,
            "process_mem_max_mib": max(process_mem) if process_mem else None,
            "process_mem_share_avg": (sum(process_mem_share) / len(process_mem_share)) if process_mem_share else None,
            "process_sm_util_avg": (sum(process_sm_util) / len(process_sm_util)) if process_sm_util else None,
            "process_sm_util_max": max(process_sm_util) if process_sm_util else None,
            "process_mem_util_avg": (sum(process_mem_util) / len(process_mem_util)) if process_mem_util else None,
            "process_mem_util_max": max(process_mem_util) if process_mem_util else None,
            "process_util_available": bool(any(process_util_available)),
        }

    def record_stage(
        self,
        *,
        epoch: int,
        batch: int,
        stage: str,
        start_time_unix: float,
        end_time_unix: float,
        extra: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        stats = self.window_stats(start_time_unix, end_time_unix)
        rec = {
            "epoch": int(epoch),
            "batch": int(batch),
            "stage": str(stage),
            "pid": int(self.pid),
            "device_index": int(self.device_index),
            "t_start_unix": float(start_time_unix),
            "t_end_unix": float(end_time_unix),
            "duration_sec": float(max(0.0, end_time_unix - start_time_unix)),
            "process_util_available": bool(stats.get("process_util_available", False)),
            **stats,
        }
        if extra:
            rec.update(dict(extra))
        self._write_jsonl(self.stage_output_path, rec)
        return rec

    @staticmethod
    def _write_jsonl(path: Optional[str], payload: Dict[str, object]) -> None:
        if path is None:
            return
        out_path = str(path)
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True))
            f.write("\n")
