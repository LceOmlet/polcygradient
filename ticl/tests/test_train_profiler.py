import json

from ticl.profiling import TrainProfiler, TrainProfilerConfig, PROFILE_SCHEMA_VERSION


def test_train_profiler_config_normalization():
    cfg = TrainProfilerConfig(
        enabled=True,
        output_path=None,
        log_to_wandb=True,
        ema_alpha=3.0,
        warmup_epochs=-5,
        warmup_batches=-2,
    ).normalized()
    assert cfg.enabled is True
    assert cfg.ema_alpha == 0.2
    assert cfg.warmup_epochs == 0
    assert cfg.warmup_batches == 0


def test_train_profiler_epoch_record_and_ema():
    profiler = TrainProfiler(
        TrainProfilerConfig(enabled=True, output_path=None, ema_alpha=0.5, warmup_epochs=0),
        rl_objective="policy_gradient",
    )

    profiler.start_epoch(1)
    profiler.record_batch(valid=True, batch_size=4, n_samples=8)
    profiler.record_batch(valid=False, batch_size=4, n_samples=8)
    rec1 = profiler.end_epoch(
        wall_time_sec=2.0,
        gpu_time_sec=1.0,
        peak_alloc_gib=3.0,
        peak_reserved_gib=4.0,
    )
    assert rec1 is not None
    assert rec1["schema_version"] == PROFILE_SCHEMA_VERSION
    assert rec1["scope"] == "epoch"
    assert rec1["complete_epoch"] is True
    assert rec1["batch_index_end"] == 2
    assert rec1["epoch"] == 1
    assert rec1["steps_total"] == 2
    assert rec1["steps_valid"] == 1
    assert rec1["steps_skipped"] == 1
    assert rec1["samples_total"] == 8
    assert rec1["tokens_total"] == 64
    assert rec1["batches_per_sec"] == 1.0
    assert rec1["ema_batches_per_sec"] == 1.0

    profiler.start_epoch(2)
    profiler.record_batch(valid=True, batch_size=4, n_samples=8)
    rec2 = profiler.end_epoch(
        wall_time_sec=1.0,
        gpu_time_sec=0.5,
        peak_alloc_gib=2.5,
        peak_reserved_gib=3.5,
    )
    assert rec2 is not None
    # epoch2 batches_per_sec=1, so ema stays at 1
    assert rec2["ema_batches_per_sec"] == 1.0


def test_train_profiler_interval_records_available_before_epoch_end():
    profiler = TrainProfiler(
        TrainProfilerConfig(enabled=True, output_path=None, ema_alpha=0.5, warmup_epochs=0, warmup_batches=0),
        rl_objective="policy_gradient",
    )
    profiler.start_epoch(1)
    profiler.record_batch(valid=True, batch_size=2, n_samples=4)
    rec = profiler.emit_interval(wall_time_sec=1.0, batch_index_end=1)
    assert rec is not None
    assert rec["scope"] == "interval"
    assert rec["complete_epoch"] is False
    assert rec["batch_index_end"] == 1
    assert rec["steps_total"] == 1


def test_train_profiler_writes_jsonl(tmp_path):
    out = tmp_path / "profile.jsonl"
    profiler = TrainProfiler(
        TrainProfilerConfig(enabled=True, output_path=str(out), ema_alpha=0.2, warmup_epochs=0),
        rl_objective="supervised",
    )
    profiler.start_epoch(3)
    profiler.record_batch(valid=True, batch_size=2, n_samples=10)
    rec = profiler.end_epoch(
        wall_time_sec=1.0,
        gpu_time_sec=0.0,
        peak_alloc_gib=None,
        peak_reserved_gib=None,
    )
    assert rec is not None
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["schema_version"] == PROFILE_SCHEMA_VERSION
    assert payload["epoch"] == 3
    assert payload["rl_objective"] == "supervised"
