import json
import os
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any


PROFILE_SCHEMA_VERSION = 2


@dataclass
class TrainProfilerConfig:
    enabled: bool = False
    output_path: Optional[str] = None
    log_to_wandb: bool = True
    ema_alpha: float = 0.2
    warmup_epochs: int = 0
    warmup_batches: int = 0

    def normalized(self) -> "TrainProfilerConfig":
        alpha = float(self.ema_alpha)
        if alpha <= 0.0 or alpha > 1.0:
            alpha = 0.2
        warmup = int(max(0, self.warmup_epochs))
        warmup_batches = int(max(0, self.warmup_batches))
        return TrainProfilerConfig(
            enabled=bool(self.enabled),
            output_path=self.output_path,
            log_to_wandb=bool(self.log_to_wandb),
            ema_alpha=alpha,
            warmup_epochs=warmup,
            warmup_batches=warmup_batches,
        )


@dataclass
class EpochProfileRecord:
    schema_version: int
    scope: str
    complete_epoch: bool
    batch_index_end: int
    epoch: int
    rl_objective: str
    wall_time_sec: float
    gpu_time_sec: float
    steps_total: int
    steps_valid: int
    steps_skipped: int
    samples_total: int
    tokens_total: int
    batches_per_sec: float
    valid_batches_per_sec: float
    samples_per_sec: float
    tokens_per_sec: float
    ema_batches_per_sec: float
    ema_valid_batches_per_sec: float
    ema_samples_per_sec: float
    ema_tokens_per_sec: float
    peak_alloc_gib: Optional[float]
    peak_reserved_gib: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TrainProfiler:
    """
    Lightweight, schema-defined training profiler for long-running experiments.

    Design principles:
    - low-overhead counters in hot loops;
    - explicit metric definitions with stable schema;
    - optional JSONL persistence for post-hoc analysis;
    - optional EMA smoothing for dashboard stability.
    """

    def __init__(self, config: TrainProfilerConfig, rl_objective: str):
        self.config = config.normalized()
        self.rl_objective = str(rl_objective)
        self.records = []
        self._ema: Dict[str, Optional[float]] = {
            "batches_per_sec": None,
            "valid_batches_per_sec": None,
            "samples_per_sec": None,
            "tokens_per_sec": None,
        }
        self._epoch = 0
        self._steps_total = 0
        self._steps_valid = 0
        self._steps_skipped = 0
        self._samples_total = 0
        self._tokens_total = 0

    def enabled(self) -> bool:
        return bool(self.config.enabled)

    def start_epoch(self, epoch: int) -> None:
        if not self.enabled():
            return
        self._epoch = int(epoch)
        self._steps_total = 0
        self._steps_valid = 0
        self._steps_skipped = 0
        self._samples_total = 0
        self._tokens_total = 0

    def _in_warmup(self) -> bool:
        if self._epoch <= int(self.config.warmup_epochs):
            return True
        if self._steps_total <= int(self.config.warmup_batches):
            return True
        return False

    def record_batch(self, *, valid: bool, batch_size: Optional[int], n_samples: Optional[int]) -> None:
        if not self.enabled():
            return
        self._steps_total += 1
        if bool(valid):
            self._steps_valid += 1
        else:
            self._steps_skipped += 1
        if batch_size is not None:
            bs = int(max(0, batch_size))
            self._samples_total += bs
            if n_samples is not None:
                ns = int(max(0, n_samples))
                self._tokens_total += (bs * ns)

    def _update_ema(self, key: str, value: float) -> float:
        prev = self._ema.get(key, None)
        if prev is None:
            cur = float(value)
        else:
            a = float(self.config.ema_alpha)
            cur = (a * float(value)) + ((1.0 - a) * float(prev))
        self._ema[key] = cur
        return cur

    @staticmethod
    def _safe_rate(num: float, denom: float) -> float:
        if denom <= 0.0:
            return 0.0
        return float(num) / float(denom)

    def _emit(
        self,
        *,
        scope: str,
        complete_epoch: bool,
        batch_index_end: int,
        wall_time_sec: float,
        gpu_time_sec: float,
        peak_alloc_gib: Optional[float],
        peak_reserved_gib: Optional[float],
    ) -> Optional[Dict[str, Any]]:
        if not self.enabled():
            return None

        wall = max(1e-12, float(wall_time_sec))
        batches_per_sec = self._safe_rate(self._steps_total, wall)
        valid_batches_per_sec = self._safe_rate(self._steps_valid, wall)
        samples_per_sec = self._safe_rate(self._samples_total, wall)
        tokens_per_sec = self._safe_rate(self._tokens_total, wall)

        if self._in_warmup():
            ema_batches = batches_per_sec
            ema_valid_batches = valid_batches_per_sec
            ema_samples = samples_per_sec
            ema_tokens = tokens_per_sec
            self._ema["batches_per_sec"] = batches_per_sec
            self._ema["valid_batches_per_sec"] = valid_batches_per_sec
            self._ema["samples_per_sec"] = samples_per_sec
            self._ema["tokens_per_sec"] = tokens_per_sec
        else:
            ema_batches = self._update_ema("batches_per_sec", batches_per_sec)
            ema_valid_batches = self._update_ema("valid_batches_per_sec", valid_batches_per_sec)
            ema_samples = self._update_ema("samples_per_sec", samples_per_sec)
            ema_tokens = self._update_ema("tokens_per_sec", tokens_per_sec)

        rec = EpochProfileRecord(
            schema_version=PROFILE_SCHEMA_VERSION,
            scope=str(scope),
            complete_epoch=bool(complete_epoch),
            batch_index_end=int(batch_index_end),
            epoch=int(self._epoch),
            rl_objective=self.rl_objective,
            wall_time_sec=float(wall_time_sec),
            gpu_time_sec=float(gpu_time_sec),
            steps_total=int(self._steps_total),
            steps_valid=int(self._steps_valid),
            steps_skipped=int(self._steps_skipped),
            samples_total=int(self._samples_total),
            tokens_total=int(self._tokens_total),
            batches_per_sec=float(batches_per_sec),
            valid_batches_per_sec=float(valid_batches_per_sec),
            samples_per_sec=float(samples_per_sec),
            tokens_per_sec=float(tokens_per_sec),
            ema_batches_per_sec=float(ema_batches),
            ema_valid_batches_per_sec=float(ema_valid_batches),
            ema_samples_per_sec=float(ema_samples),
            ema_tokens_per_sec=float(ema_tokens),
            peak_alloc_gib=(None if peak_alloc_gib is None else float(peak_alloc_gib)),
            peak_reserved_gib=(None if peak_reserved_gib is None else float(peak_reserved_gib)),
        )
        data = rec.to_dict()
        self.records.append(data)
        self._write_record(data)
        return data

    def emit_interval(
        self,
        *,
        wall_time_sec: float,
        gpu_time_sec: float = 0.0,
        peak_alloc_gib: Optional[float] = None,
        peak_reserved_gib: Optional[float] = None,
        batch_index_end: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        if batch_index_end is None:
            batch_index_end = int(self._steps_total)
        return self._emit(
            scope="interval",
            complete_epoch=False,
            batch_index_end=int(batch_index_end),
            wall_time_sec=wall_time_sec,
            gpu_time_sec=gpu_time_sec,
            peak_alloc_gib=peak_alloc_gib,
            peak_reserved_gib=peak_reserved_gib,
        )

    def end_epoch(
        self,
        *,
        wall_time_sec: float,
        gpu_time_sec: float,
        peak_alloc_gib: Optional[float],
        peak_reserved_gib: Optional[float],
    ) -> Optional[Dict[str, Any]]:
        return self._emit(
            scope="epoch",
            complete_epoch=True,
            batch_index_end=int(self._steps_total),
            wall_time_sec=wall_time_sec,
            gpu_time_sec=gpu_time_sec,
            peak_alloc_gib=peak_alloc_gib,
            peak_reserved_gib=peak_reserved_gib,
        )

    def _write_record(self, data: Dict[str, Any]) -> None:
        path = self.config.output_path
        if path is None:
            return
        out_path = str(path)
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=True))
            f.write("\n")
