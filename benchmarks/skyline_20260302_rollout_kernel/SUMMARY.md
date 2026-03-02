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
- Training paged-attention online reduction probe (`20260302_161500_pagedattn_online`):
  - Replaced training multi-page dense SDPA (with full `cat` views) by page-chunk online softmax reduction.
  - Strongly reduced memory (`peak alloc/reserved 6.10/6.12 GiB`) but severely increased runtime (`162.89s`).
  - Classified as pseudo-optimization for fixed-workload speed target; reverted.
- Training paged-attention prefix+tail reduction probe (`20260302_162500_pagedattn_prefix`):
  - Reduced chunk count versus pure online mode by merging full-prefix and tail pages in two-way reduction.
  - Memory stayed low (`peak alloc/reserved 6.68/6.69 GiB`) but wallclock still regressed (`92.23s`).
  - Not a skyline candidate for same-workload speed; reverted.
- Training paged-attention Flash-LSE merge probe (`20260302_163800_pagedattn_flashmerge`):
  - Implemented exact chunk merge via FlashAttention LSE reduction.
  - Wallclock regressed to `71.18s` (`rollout/backward 71.131/40.631s`, peak `27.38/34.63 GiB`).
  - Not retained for fixed-workload skyline.
- Flash-LSE merge with large chunk cap (`20260302_164300_flashmerge_chunk4096`):
  - Reduced merge chunk count to near single-chunk behavior.
  - Wallclock improved vs previous flash-merge probe (`68.32s`) but still slower than skyline `62.83s`.
  - Not retained for fixed-workload skyline.
- Flash prefix+tail exact merge probe (`20260302_165000_flashprefix`):
  - Enabled exact two-way FlashAttention merge (`prefix + tail`) without dense KV concat.
  - Strong memory reduction (`peak alloc/reserved 7.47/7.55 GiB`) but fixed-batch wallclock regressed (`79.51s`).
- Flash prefix mode with widened batch (`20260302_165800_flashprefix_bs32`):
  - `wallclock 76.23s`, peak `23.63/25.32 GiB`.
  - Normalized metric improved: `76.23 / 32 = 2.382s` per batch-unit.
- Flash prefix mode with widened batch (`20260302_171500_flashprefix_bs48`):
  - `wallclock 87.12s`, peak `32.88/36.59 GiB`.
  - Normalized metric improved further: `87.12 / 48 = 1.815s` per batch-unit.
- Flash prefix mode with widened batch (`20260302_172200_flashprefix_bs64`):
  - `wallclock 83.13s`, `rollout/backward 83.073/48.345s`, peak `40.03/45.79 GiB`.
  - Normalized metric: `83.13 / 64 = 1.299s` per batch-unit (best observed).
  - GPU memory stayed below OOM threshold while host memory remained stable (`~5.8 GiB` used).
- Dense attention OOM control (`20260302_173000_dense_bs64_oomprobe`):
  - Same `bs64` under dense paged-attention path fails with CUDA OOM during KV page COW append.
  - Confirms widened-batch throughput path is enabled by flash-prefix memory reduction rather than metric artifact.
- Flash-prefix COW page-size knob implementation (code-level):
  - Added `TICL_POLICY_PAGED_ATTN_FLASHPREFIX_PAGE_SIZE` for training paged-KV flash-prefix mode.
  - Purpose: control mutable tail-page copy growth in COW append hot path without changing rollout chunk/TBPTT semantics.
  - Default kept at `128` (no behavior change unless explicitly enabled).
- Flash-prefix page-size sweep on throughput path (`batch_size=64`):
  - `page32` (`20260302_174200_flashprefix_bs64_page32`): stable but slower (`85.81s`, `1.341s/batch-unit`).
  - `page48` (`20260302_180500_flashprefix_bs64_page48`): faster (`82.34s`, `1.287s/batch-unit`), higher SM util.
  - `page64/128` (`20260302_175800_flashprefix_bs64_page64`, `20260302_174900_flashprefix_bs64_page128`): triggered OOM fallback/instability, rejected.
- Batch-width sweep under `flash_prefix + page48`:
  - `bs68`:
    - `20260302_181900_flashprefix_bs68_page48`: `79.94s` (`1.176s/batch-unit`)
    - `20260302_183600_flashprefix_bs68_page48_rep2`: `87.38s` (`1.285s/batch-unit`)
    - `20260302_190200_flashprefix_bs68_page48_rep3`: `82.89s` (`1.219s/batch-unit`)
    - Aggregate: mean `83.40s` (`1.226s/batch-unit`), median `82.89s` (`1.219s/batch-unit`)
  - `bs70`:
    - `20260302_182600_flashprefix_bs70_page48_oomprobe`: one successful run `84.69s` (`1.210s/batch-unit`)
    - `20260302_191100_flashprefix_bs70_page48_rep2`: CUDA OOM in backward, rejected for stability.
  - `bs72` (`20260302_181200_flashprefix_bs72_page48_oomprobe`): CUDA OOM, rejected.
- Stability controls / negative probes (kept for anti-regression evidence):
  - `bs64 + page48 + seeded` (`20260302_184600_flashprefix_bs64_page48_seeded`): CUDA OOM (environment path pressure), indicates reduced safety margin at seeded heavy draws.
  - `bs68 + page32` (`20260302_185200_flashprefix_bs68_page32`): CUDA OOM, rejected.
  - In-place flash-prefix tail-clone probe (`20260302_193500_flashprefix_inplace_bs64_page48`, `20260302_194000_flashprefix_inplace_bs64_page32`):
    - Experimental branch attempted to combine in-place paged-KV append with flash-prefix page-size control and reduced prefix cloning.
    - Both runs failed with CUDA OOM at `bs64` (`page48`: rollout path OOM, `page32`: backward OOM).
    - Process memory still climbed to ~`46.83 GiB`; rejected and not merged.
  - Paged-attn recompute-checkpoint probe (`20260302_184300_flashprefix_bs68_page48_pagedckpt`):
    - Added a temporary inner checkpoint for paged flash attention (COW, non-inplace) to reduce activation pressure.
    - No OOM, but wallclock regressed badly to `127.68s` (`1.878s/batch-unit`) with `rollout/backward=127.619/77.131s`.
    - Peak alloc/reserved remained high (`42.81/46.36 GiB`), process memory max `47888 MiB`; reverted as pseudo-optimization.
  - Step-FFN recompute probe (`20260302_185100_flashprefix_bs68_page48_ffnckpt`):
    - Added temporary per-step FFN checkpointing (`TICL_POLICY_RECOMPUTE_STEP_FFN=1`) to reduce activation memory.
    - Memory improved (`peak alloc 36.82 GiB`) but throughput regressed heavily (`100.48s`, `1.478s/batch-unit`).
    - Reverted as pseudo-optimization for throughput skyline.
  - Transition-group prebind + single-group fastpath probe (`20260302_193000_flashprefix_bs68_page48_groupfast`, `20260302_194100_flashprefix_bs70_page48_groupfast`, `20260302_194900_flashprefix_bs70_page48_groupfast_rep2`, `20260302_200500_flashprefix_bs70_page48_groupfast_rep3`, `20260302_195700_flashprefix_bs72_page48_groupfast`, `20260302_201200_flashprefix_bs68_page48_groupfast_rep2`):
    - Targeted hot-loop Python/index overhead in `_rollout_family_group_vectorized_with_policy` by prebinding group constants and adding single-group transition fastpath.
    - `bs70` became 3/3 successful (`84.73s`, `86.59s`, `87.11s`; no OOM), but normalized throughput stayed around `1.210~1.244s/batch-unit` and did not beat retained `bs68` robust median (`1.219`) by a meaningful margin.
    - `bs72` still OOM (`20260302_195700...`), and `bs68` showed high variance including a slow outlier (`99.11s`), so the change was not kept.

## Skyline status

- Fixed-workload wallclock skyline (`n_samples=1024`, `batch_size=8` unchanged): `20260302_142001_qkvfused_nortinfo_noseed_rel` (`62.83s`).
- Throughput-normalized skyline (`wallclock / batch_size`, reduced-memory widened-batch path):
  - Previous retained: `20260302_172200_flashprefix_bs64` (`83.13s`, `1.299s/batch-unit`).
  - New retained (multi-run robust): `flash_prefix + page48 + bs68` with median `82.89s` (`1.219s/batch-unit`) and mean `83.40s` (`1.226s/batch-unit`).
  - Best observed single run in this cohort: `20260302_181900_flashprefix_bs68_page48` (`79.94s`, `1.176s/batch-unit`).
  - Reproduce setting: `TICL_POLICY_PAGED_ATTN_TRAIN_MODE=flash_prefix TICL_POLICY_PAGED_ATTN_FLASHPREFIX_PAGE_SIZE=48` with `--batch-size 68`.
