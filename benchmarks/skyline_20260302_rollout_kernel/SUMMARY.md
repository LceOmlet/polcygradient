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

## Follow-up Probes (same workload, kept for regression tracking)

- Kernel-profiler probe (`20260302_105332_kernelprof`):
  - Used only for observability; profiling overhead inflated runtime (`rollout 137.378s`, `backward 89.367s`).
  - Not considered skyline candidate.
- QKV-fused projection probe (`20260302_105845_qkvfused`):
  - `wallclock 70.96s` vs skyline `70.63s` (no gain).
  - Reverted (did not keep code change).
- Environment `einsum->bmm` probe (`20260302_110339_envbmm`):
  - `wallclock 72.32s` vs skyline `70.63s` (regression on this workload).
  - Reverted (did not keep code change).
- Family subgroup CUDA-stream probe (`20260302_111025_groupstream`):
  - `wallclock 71.20s` vs skyline `70.63s` (no gain, higher peak reserve).
  - Reverted (did not keep code change).
- No-split policy-step probe (`20260302_112049_nosplitfastpath`):
  - `wallclock 72.90s` vs skyline `70.63s` (regression).
  - Reverted (kept split fast path enabled).
- Static mutable KV probe (`20260302_112238_statickv`):
  - Triggered autograd version conflict in backward (in-place mutation on cache tensor).
  - Not adopted; kept paged mutable KV path.
- Rollout breakdown observability probe (`20260302_113300_breakdownlog`):
  - Added per-batch rollout split metrics:
    - `rollout_policy_cuda_ms=19595.67`
    - `rollout_transition_cuda_ms=10753.38`
    - `rollout_policy_share=0.646`
  - Used to identify policy-step kernels as dominant rollout-forward component.
- Inner recompute-disable probe in TBPTT path (`20260302_113943_norecompute_tbptt`):
  - `wallclock 72.37s` vs skyline `70.63s` (no gain on no-seed workload).
  - Reverted.
- `torch.compile` reduce-overhead probe (`20260302_114150_compile_probe`):
  - Runtime failure on mutable KV path (CUDAGraph overwritten output).
  - Not adopted.
- `torch.compile` default probe (`20260302_114300_compile_default_probe`):
  - Hit Dynamo `recompile_limit` due changing KV stride; `wallclock 303.01s`.
  - Not adopted.
- Non-reentrant forward_step checkpoint probe:
  - No-seed run (`20260302_114945_nonreentrant_stepckpt`) showed an apparent win (`68.46s`) but was not stable.
  - Seeded A/B showed regression:
    - non-reentrant (`20260302_115347_nonreentrant_seeded`): `76.99s`
    - reentrant (`20260302_115531_reentrant_seeded`): `73.16s`
  - Reverted (kept `use_reentrant=True`).
- TF32 matmul probe on policy-gradient CUDA path:
  - Added default policy-only switch in code (`TICL_POLICY_TF32=1`, opt-out via `TICL_POLICY_TF32=0`).
  - Seeded comparisons:
    - TF32 on (`20260302_115950_reentrant_seeded_tf32default`): `71.34s`
    - TF32 off (`20260302_120346_reentrant_seeded_tf32off`, `20260302_120519_reentrant_seeded_tf32off_rep2`): `71.96s`, `73.47s`
  - Mean seeded wallclock improvement from TF32 enable is ~2%.
  - No-seed run (`20260302_120130_tf32default_noseed`) remained noisy (`73.74s`), so no-seed skyline is unchanged.

Current retained skyline remains `20260302_104751_layerpagedsdpa`.
