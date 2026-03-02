import subprocess
import time

from ticl.gpu_observer import GPUProcessObserver


def test_gpu_observer_parses_gpu_and_process_memory(monkeypatch):
    pid = 4242

    def _fake_check_output(cmd, text, stderr, timeout):
        joined = " ".join(cmd)
        if "--query-gpu=index,uuid,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw" in joined:
            return "0, GPU-AAA, 37, 12, 12000, 49140, 103.5\n1, GPU-BBB, 5, 1, 100, 49140, 40.0\n"
        if "--query-compute-apps=gpu_uuid,pid,used_gpu_memory" in joined:
            return "GPU-AAA, 4242, 8000\nGPU-AAA, 8888, 1000\n"
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(subprocess, "check_output", _fake_check_output)

    obs = GPUProcessObserver(
        enabled=True,
        device="cuda:0",
        sample_interval_sec=1.0,
        pid=pid,
    )
    sample = obs._sample_once()
    assert sample is not None
    assert sample["device_index"] == 0
    assert sample["gpu_uuid"] == "GPU-AAA"
    assert sample["gpu_util_percent"] == 37.0
    assert sample["gpu_mem_total_mib"] == 49140.0
    assert sample["process_present"] is True
    assert sample["process_mem_mib"] == 8000.0
    assert sample["process_util_available"] is False


def test_gpu_observer_window_stats_aggregates_correctly():
    obs = GPUProcessObserver(enabled=True, device="cuda:0", sample_interval_sec=1.0, pid=1)
    t0 = time.time()
    obs._samples = [
        {
            "timestamp_unix": t0 + 0.1,
            "gpu_util_percent": 10.0,
            "gpu_mem_util_percent": 2.0,
            "gpu_power_w": 80.0,
            "process_mem_mib": 1000.0,
            "process_mem_share_percent": 2.0,
        },
        {
            "timestamp_unix": t0 + 0.2,
            "gpu_util_percent": 30.0,
            "gpu_mem_util_percent": 6.0,
            "gpu_power_w": 100.0,
            "process_mem_mib": 3000.0,
            "process_mem_share_percent": 6.0,
        },
    ]
    stats = obs.window_stats(t0, t0 + 1.0)
    assert stats["samples"] == 2
    assert stats["gpu_util_avg"] == 20.0
    assert stats["gpu_util_max"] == 30.0
    assert stats["gpu_mem_util_avg"] == 4.0
    assert stats["gpu_power_avg_w"] == 90.0
    assert stats["process_mem_avg_mib"] == 2000.0
    assert stats["process_mem_max_mib"] == 3000.0
    assert stats["process_mem_share_avg"] == 4.0
