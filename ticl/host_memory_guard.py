import math
import os
import threading

try:
    import psutil
except Exception:  # pragma: no cover - optional runtime dependency
    psutil = None

try:
    import resource
except Exception:  # pragma: no cover - platform dependent
    resource = None


def _current_rss_bytes():
    if psutil is not None:
        try:
            return int(psutil.Process(os.getpid()).memory_info().rss)
        except Exception:
            pass
    try:
        with open("/proc/self/statm", "r", encoding="utf-8") as f:
            fields = f.read().strip().split()
        if len(fields) >= 2:
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            return int(fields[1]) * page_size
    except Exception:
        pass
    return None


def _normalize_limit_bytes(limit_gib):
    try:
        limit_gib = float(limit_gib)
    except Exception:
        return None
    if (not math.isfinite(limit_gib)) or limit_gib <= 0.0:
        return None
    return int(limit_gib * (1024.0 ** 3))


def _try_set_rlimit_as(limit_bytes):
    if resource is None or limit_bytes is None:
        return {"requested": False, "applied": False, "reason": "unsupported"}
    rlimit_as = getattr(resource, "RLIMIT_AS", None)
    if rlimit_as is None:
        return {"requested": False, "applied": False, "reason": "missing_rlimit_as"}
    try:
        soft, hard = resource.getrlimit(rlimit_as)
        new_soft = int(limit_bytes)
        new_hard = hard
        if hard not in (-1, resource.RLIM_INFINITY):
            new_soft = min(new_soft, int(hard))
            new_hard = int(hard)
        elif hard in (-1, resource.RLIM_INFINITY):
            new_hard = int(limit_bytes)
        resource.setrlimit(rlimit_as, (int(new_soft), int(new_hard)))
        return {
            "requested": True,
            "applied": True,
            "soft": int(new_soft),
            "hard": int(new_hard),
        }
    except Exception as exc:
        return {
            "requested": True,
            "applied": False,
            "reason": str(exc),
        }


class HostRSSLimitGuard:
    def __init__(self, *, limit_bytes, poll_interval_sec=0.02, exit_code=99):
        self.limit_bytes = int(limit_bytes)
        self.poll_interval_sec = float(max(0.01, poll_interval_sec))
        self.exit_code = int(exit_code)
        self._stop_event = threading.Event()
        self._thread = None

    def _trip(self, rss_bytes):
        rss_gib = float(rss_bytes) / (1024.0 ** 3)
        limit_gib = float(self.limit_bytes) / (1024.0 ** 3)
        msg = (
            f"[host-rss-limit] rss_gib={rss_gib:.2f} "
            f"limit_gib={limit_gib:.2f} "
            f"poll_interval_sec={self.poll_interval_sec:.3f} "
            "-> exiting immediately\n"
        )
        try:
            os.write(2, msg.encode("utf-8", "replace"))
        except Exception:
            pass
        os._exit(self.exit_code)

    def _run(self):
        while not self._stop_event.is_set():
            rss_bytes = _current_rss_bytes()
            if rss_bytes is not None and rss_bytes > self.limit_bytes:
                self._trip(rss_bytes)
            self._stop_event.wait(self.poll_interval_sec)

    def start(self):
        if self._thread is not None:
            return
        rss_bytes = _current_rss_bytes()
        if rss_bytes is not None and rss_bytes > self.limit_bytes:
            self._trip(rss_bytes)
        self._thread = threading.Thread(
            target=self._run,
            name="ticl-host-rss-limit-guard",
            daemon=True,
        )
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=max(0.1, self.poll_interval_sec * 4.0))


def install_host_rss_limit_guard(
    *,
    limit_gib,
    poll_interval_sec=0.02,
    try_rlimit_as=False,
    exit_code=99,
):
    limit_bytes = _normalize_limit_bytes(limit_gib)
    status = {
        "enabled": bool(limit_bytes is not None),
        "limit_gib": None if limit_bytes is None else float(limit_bytes) / (1024.0 ** 3),
        "poll_interval_sec": float(max(0.01, poll_interval_sec)),
        "try_rlimit_as": bool(try_rlimit_as),
        "rlimit_as": {"requested": False, "applied": False, "reason": "disabled"},
        "sensor_available": True,
    }
    if limit_bytes is None:
        return None, status
    if _current_rss_bytes() is None:
        status["enabled"] = False
        status["sensor_available"] = False
        return None, status
    if bool(try_rlimit_as):
        status["rlimit_as"] = _try_set_rlimit_as(limit_bytes)
    guard = HostRSSLimitGuard(
        limit_bytes=limit_bytes,
        poll_interval_sec=poll_interval_sec,
        exit_code=exit_code,
    )
    guard.start()
    return guard, status
