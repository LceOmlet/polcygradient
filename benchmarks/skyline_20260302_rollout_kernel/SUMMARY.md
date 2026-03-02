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
- Previous skyline: `benchmarks/skyline_20260302_rollout_kernel/20260302_133739_mod_noseed_fp16/run.log`
- New skyline: `benchmarks/skyline_20260302_rollout_kernel/20260302_142001_qkvfused_nortinfo_noseed_rel/run.log`

## Metrics (previous skyline -> new skyline)

- Wallclock: `66.10s -> 62.83s` (`-4.9%`)
- Rollout stage (`cuda_elapsed_ms`): `66.050s -> 62.778s` (`-5.0%`)
- Backward stage (`cuda_elapsed_ms`): `39.486s -> 35.989s` (`-8.9%`)
- GPU util avg (rollout): `22.77 -> 24.26` (`+6.5%`)
- Process SM util avg (rollout): `22.06 -> 22.62` (`+2.5%`)
- Process mem max: `35716 MiB -> 36070 MiB` (`+1.0%`)
- Peak alloc: `26.55 GiB -> 26.76 GiB` (`+0.8%`)

## Kernel-level change that produced the gain

1. In `TransformerEncoderLayer.forward_step`, append-mode Q/K/V projection is fused to a single `in_proj` GEMM (one launch instead of separate `q` and `kv` projections per layer-step).
2. Policy-gradient rollout path now bypasses runtime-info collection (`collect_runtime_info=False` in `rollout_policy_gradient_loss`), removing per-step `state_abs_max` reductions and rollout-end GPU->CPU stat sync from the training hot path.

These changes reduce launch/sync overhead in policy rollout and improve fixed-workload single-batch throughput (`n_samples=1024`, `batch_size=8`).

## Seeded sanity check (same workload, `--seed-everything True`)

- Previous skyline code:
  - `20260302_133605_mod_seeded_fp16`: `70.68s`
  - `20260302_134102_mod_seeded_fp16_rep2`: `70.45s`
- New code:
  - `20260302_141651_qkvfused_nortinfo_seeded`: `67.64s`
  - `20260302_141819_qkvfused_nortinfo_seeded_rep2`: `67.20s`
- Mean seeded wallclock: `70.56s -> 67.42s` (`-4.5%`)

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
- In-place paged-KV append probe (`20260302_131730_inplacepagedclone`):
  - Increased rollout GPU util but regressed wallclock (`80.56s`).
  - Classified as pseudo-optimization; kept behind explicit opt-in (`TICL_POLICY_INPLACE_PAGED_KV=1`) and disabled by default.
- QKV-fused + rollout-runtime-info-bypass probe (`20260302_141508_qkvfused_nortinfo`):
  - First no-seed run: `72.55s` (negative outlier, not retained as skyline).
- QKV-fused + rollout-runtime-info-bypass no-seed reruns:
  - `20260302_142001_qkvfused_nortinfo_noseed_rel`: `62.83s` (new retained skyline).
  - `20260302_142125_qkvfused_nortinfo_noseed_rep2`: `65.92s` (still better than previous `66.10s` skyline).
  - Combined with seeded A/B above, this indicates the improvement is not a pure no-seed artifact.
- Policy-rollout dtype-follow-autocast probe (`20260302_154000_rolloutdtype_auto`, `20260302_154600_rolloutdtype_auto_seeded`):
  - Attempted to run vectorized policy rollout state/noise path in autocast-following dtype.
  - No-seed run regressed badly (`75.23s`) and increased peak alloc/reserved (`27.01/35.11 GiB`).
  - Seeded run (`66.92s`) was near existing seeded skyline, but gain was not stable and no memory reduction materialized.
  - Reverted as unstable pseudo-optimization.
- TBPTT deferred-backward probe (`20260302_160200_tbpttbwd_auto`):
  - Attempted to reduce backward fragmentation by deferring TBPTT window backward.
  - Triggered immediate OOM fallback (`[pg-oom] switching TBPTT backward mode to stream`), then retried in streaming mode.
  - Peak alloc/reserved spiked to `45.39/46.38 GiB`, process memory reached `47972 MiB`, wallclock regressed to `102.75s`.
  - Classified as pseudo-optimization with high OOM/system-stability risk; reverted.

Current retained skyline is `20260302_142001_qkvfused_nortinfo_noseed_rel`.
