# Rollout Kernel Skyline (2026-03-02)

## Goal
Improve single-batch wallclock under fixed task scale (`n_samples=1024`, `batch_size=8`, no semantic downscaling).

## Command (same workload)

```bash
CONDA_NO_PLUGINS=true conda run --no-capture-output -n rlpfn \
  python -u -m ticl.fit_model rlpfn \
  --epochs 1 --num-steps 1 \
  --validate False --rl-validate-enabled False --progress-bar False \
  --train-profiler-enabled True --train-profiler-wandb False --train-profiler-log-every-batches 1 \
  --train-gpu-observer-enabled True --train-gpu-observer-interval-sec 0.2
```

## A/B Runs

- Baseline: `benchmarks/skyline_20260302_rollout_contig/current_worktree_noretain_retry/run.log`
- Split policy-step only: `benchmarks/skyline_20260302_rollout_kernel/20260302_104246_splitstep.log`
- New skyline: `benchmarks/skyline_20260302_rollout_kernel/20260302_104751_layerpagedsdpa.log`

## Metrics (baseline -> new skyline)

- Wallclock: `89.13s -> 70.63s` (`-20.8%`)
- Rollout stage: `89.041s -> 70.577s` (`-20.7%`)
- Backward stage: `53.544s -> 42.283s` (`-21.0%`)
- GPU util avg: `16.84 -> 20.20` (`+19.9%`)
- Process SM util avg: `16.94 -> 19.49` (`+15.0%`)
- Peak alloc: `26.62 GiB -> 26.35 GiB` (`-1.0%`)

## Kernel-level change that produced the gain

1. In paged KV + grad path, `forward_step` now prefers fused SDPA on dense paged views instead of multi-page manual attention reduction loop.
2. `mutable_paged_grad` page-size cap was relaxed from hard upper bound `32` to configured `kv_cache_page_size` (default `128`) to reduce page fragmentation and per-step page-loop overhead.

These two changes reduced tiny-kernel launch overhead in policy rollout without changing task scale.
