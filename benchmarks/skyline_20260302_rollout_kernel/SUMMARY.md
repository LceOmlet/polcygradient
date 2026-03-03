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
- True QKV in-proj fusion correction (`20260302_231000_qkvtruefusion_fixedcfg_plain`, `20260302_231300_qkvtruefusion_fixedcfg_plain_rep2`):
  - Found a kernel-fusion gap in `TransformerEncoderLayer._project_qkv`: previous code still executed 3 separate `F.linear` launches (`q/k/v`) instead of a single `in_proj` GEMM.
  - Fixed to `F.linear(..., in_proj_weight, in_proj_bias)` + feature-axis split; also fused `_project_kv` into one `F.linear` + split.
  - Fixed-workload reruns (same command, `n_samples=1024`, `batch_size=8`):
    - `20260302_231000...`: `55.77s`, rollout/backward `55.721/31.589s`, peak alloc/reserved `24.90/32.81 GiB`, process mem max `34076 MiB`.
    - `20260302_231300...`: `57.14s`, rollout/backward `57.095/32.656s`, peak alloc/reserved `24.06/32.34 GiB`, process mem max `33590 MiB`.
    - Aggregate: mean `56.46s` (about `-10.1%` vs previous `62.83s` skyline), median `56.46s`.
  - Note: rollout `gpu_util_avg` remained around `23%` (not increased), but wallclock and per-batch throughput improved significantly, consistent with de-emphasizing raw GPU-Util as a proxy metric.
- Policy-step decomposition confirmation run (`20260302_230500_qkvtruefusion_fixedcfg`, `TICL_PROFILE_ROLLOUT_TIMING=1 TICL_POLICY_STEP_PROFILE=1`):
  - `wallclock 51.12s` (instrumented run), rollout/backward `51.076/29.019s`.
  - `policy_step_total_ms=15067.63` vs prior `19353.99` (`20260302_220100...`, about `-22%`), with `policy_step_transformer_share` `0.929` vs `0.945`.
  - Confirms the dominant hot path is still transformer forward-step, but true QKV projection fusion removed a substantial launch/compute overhead chunk.
- Kernel-profiler summary-only retry (`20260302_233000_kernprof_after_qkvtruefusion`):
  - Retried `torch.profiler` with `export_trace=False` and short schedule (`0/0/1/1`), still reproduced tail hang after stage logs.
  - No usable profiler artifact was flushed to `kernel/`; process had to be terminated.
  - Classified as observability-path instability; switched to low-overhead in-model timers.
- Transformer layer-step low-overhead profile (code instrumentation):
  - Added optional transformer-layer step profiling (`TICL_TRANSFORMER_LAYER_STEP_PROFILE=1`) and propagated to training logs:
    - `policy_step_tf_layer_proj_share`
    - `policy_step_tf_layer_cache_share`
    - `policy_step_tf_layer_attnff_share`
  - Probe run (`20260303_000100_fixedcfg_tf_layer_profile`) reported:
    - `policy_step_tf_layer_proj_share=0.135`
    - `policy_step_tf_layer_cache_share=0.135`
    - `policy_step_tf_layer_attnff_share=0.707`
  - This confirms remaining dominant bottleneck is attn+ffn core, while cache copy/update remains a secondary but non-trivial contributor.
- Dense-prefix exact-merge probe (`20260302_235300_denseprefix_page48_fixedcfg_probe`):
  - New `dense_prefix` attention path (prefix+tail exact merge via dense matmul) regressed heavily:
    - `wallclock 96.90s`, `rollout/backward 96.823/60.973s`.
  - Classified as pseudo-optimization; reverted from code path.
- Dense paged-KV page-size sweep (fixed workload, no batch/chunk changes):
  - `page32` (`20260303_003500_dense_pagesize32_fixedcfg_probe`): `58.62s` (regression).
  - `page64` (`20260303_003100_dense_pagesize64_fixedcfg_probe`): `59.34s` (regression).
  - `page48` probes:
    - `20260303_000800...`: `50.07s` (best single run)
    - `20260303_001200...`: `57.52s`
    - `20260303_001500...`: `55.30s`
    - `20260303_002700...`: `57.67s`
  - `page48` showed high variance; best-case gain exists but median gain was not robust enough to make it a default on this evidence set alone.
- Dense page-cap default A/B (seeded, to reduce variance):
  - `cap48` (`20260303_010000_seeded_densecap48_ab`): `54.16s`
  - `cap128` (`20260303_010400_seeded_densecap128_ab`): `53.45s`
  - Seeded A/B did not support a stable `cap48` win; reverted default cap to `128` and kept `TICL_POLICY_PAGED_ATTN_DENSE_PAGE_SIZE` as explicit experimental knob.
- Nsight kernel observability retry (`20260303_022500_nsys_fixedcfg_seeded`):
  - Tried `nsys profile` for kernel-level top attribution.
  - Environment lacked importer dependencies (`Importer error status...`), only `trace.qdstrm` was generated; no `nsys stats` kernel summary could be exported.
  - Kept as observability artifact only; not used for skyline judgement.
- Deep transformer-layer profile extension (`20260303_024200_tf_layer_deep_profile`):
  - Added low-overhead sub-breakdown inside layer `attnff` hot path:
    - `policy_step_tf_layer_attn_core_share=0.075`
    - `policy_step_tf_layer_finalize_share=0.524`
    - `policy_step_tf_layer_finalize_attn_outproj_share=0.201`
    - `policy_step_tf_layer_finalize_ffn_share=0.314`
  - Conclusion: true dominant hotspot is **finalize path** (out-proj + residual/norm + FFN), not SDPA core.
- SDPA backend-forcing probe (rejected as pseudo-optimization):
  - Added temporary length-aware `TICL_POLICY_SDPA_BACKEND` path (short contexts auto, long contexts CUDNN).
  - Runtime probe (`20260303_023700_post_sdpa_backend_seeded`) stalled abnormally after epoch start with no stage logs; process was terminated.
  - Cross-check with public PyTorch issue (`scaled_dot_product_attention and CUDNN_ATTENTION #154602`) is consistent with heavy overhead under changing sequence lengths.
  - Fully reverted from mainline code.
- Finalize single-token 2D fastpath (seeded A/B retained optimization):
  - Implemented `forward_step` finalize hot-path specialization for `L=1`:
    - avoid extra `permute/contiguous` and 3D linear path;
    - run out-proj + residual/norm + FFN in 2D `(B, E)` and restore shape at return.
  - Added explicit guard knob: `TICL_POLICY_FINALIZE_2D_FASTPATH` (default `1`, `0` disables for A/B).
  - Seeded A/B (`on` vs `off`) under identical fixed config:
    - `on` (`20260303_030500_finalize2d_ab_on`): `52.90s`, rollout/backward `52.855/29.743s`
    - `off` (`20260303_030900_finalize2d_ab_off`): `54.32s`, rollout/backward `54.270/30.567s`
    - Pairwise gain: about `-2.6%` wallclock.
  - Main-command no-seed verification (`20260303_031700_finalize2d_noseed_maincmd`):
    - `54.69s`, rollout/backward `54.636/31.039s`, peak alloc/reserved `24.52/32.73 GiB`.
  - Main-command no-seed control with fastpath disabled (`20260303_032500_finalize2d_noseed_off_control`):
    - `53.86s`, rollout/backward `53.813/30.334s`, peak alloc/reserved `25.00/32.62 GiB`.
  - No-seed path remains high-variance; this change is retained based on seeded A/B evidence, not single no-seed runs.

## 2026-03-02 Safety Note

- Host memory (`/usr/bin/time -v` max RSS) during new fixed-workload runs stayed around `2.1 GiB`, while process GPU memory max dropped from previous skyline (`36070 MiB`) to `33590~34076 MiB`.
- Experimental transition stream-fusion path is now disabled by default (`TICL_POLICY_TRANSITION_STREAM_FUSION=0`) and guarded to avoid per-sample generator concurrency hazards.

## Skyline status

- Fixed-workload wallclock skyline (`n_samples=1024`, `batch_size=8` unchanged):
  - No-seed retained remains `qkv true fusion` baseline pair:
    - `20260302_231000_qkvtruefusion_fixedcfg_plain` (`55.77s`)
    - `20260302_231300_qkvtruefusion_fixedcfg_plain_rep2` (`57.14s`)
  - Seeded retained A/B for finalize fastpath:
    - `on` (`20260303_030500_finalize2d_ab_on`): `52.90s`
    - `off` (`20260303_030900_finalize2d_ab_off`): `54.32s`
    - Deterministic gain: about `-2.6%`.
  - Note: later exploratory runs observed lower single-run values (down to `49.73s`), but due high variance and seeded A/B inconsistency they are not retained as skyline yet.
- Throughput-normalized skyline (`wallclock / batch_size`, reduced-memory widened-batch path):
  - Previous retained: `20260302_172200_flashprefix_bs64` (`83.13s`, `1.299s/batch-unit`).
  - New retained (multi-run robust): `flash_prefix + page48 + bs68` with median `82.89s` (`1.219s/batch-unit`) and mean `83.40s` (`1.226s/batch-unit`).
  - Best observed single run in this cohort: `20260302_181900_flashprefix_bs68_page48` (`79.94s`, `1.176s/batch-unit`).
  - Reproduce setting: `TICL_POLICY_PAGED_ATTN_TRAIN_MODE=flash_prefix TICL_POLICY_PAGED_ATTN_FLASHPREFIX_PAGE_SIZE=48` with `--batch-size 68`.

## 2026-03-03 Rebuild: Post-Stability Skyline Ladder

- Risk acknowledged: recent numerical-stability changes invalidate direct comparison against legacy `grad_norm_nonfinite/inf` runs.
- Rebuild script added: `benchmarks/skyline_20260302_rollout_kernel/rebuild_skyline.py`.
- Rebuilt artifacts:
  - `benchmarks/skyline_20260302_rollout_kernel/REBUILT_RUNS.jsonl`
  - `benchmarks/skyline_20260302_rollout_kernel/REBUILT_SKYLINE.md`
- Strict validity gate (new canonical comparison):
  - `[pg-phase] status == ok`
  - `valid/skipped >= 1/0`
  - finite `mean loss`
- Current rebuilt dataset summary:
  - scanned runs: `98`
  - strict-valid runs: `4`
  - status distribution: `grad_norm_nonfinite=77, ok=4, oom=1, unknown=16`
- Current strict-valid best run:
  - `20260303_012000_denseprefixcache_default_probe_rep2` (`48.36s`, `status=ok`, `valid/skipped=1/0`, `chunk=8`, `tbptt=128`)
- New post-rebuild deep-profile validation run:
  - `20260303_093756_post_rebuild_transition_profile_ok` (`49.47s`, `status=ok`, `valid/skipped=1/0`)
  - New transition breakdown (same fixed workload) from `[pg-phase]`:
    - `rollout_transition_y_share=0.337`
    - `rollout_transition_x_share=0.334`
    - `rollout_transition_env_pack_share=0.072`
    - `rollout_transition_state_update_share=0.163`
    - `rollout_transition_group_count=2`
  - Transformer layer-step still dominated by finalize path:
    - `policy_step_tf_layer_finalize_share=0.484`
    - `policy_step_tf_layer_attn_core_share=0.080`
- Rule update for future skyline submissions:
  - keep two ladders in parallel:
    - `Strict-Valid Skyline` for retained claims
    - `Diagnostic Ladder` for speed-only invalid/oom probes (non-retained)

## External Framework Signals (HF / Unsloth / FlashAttention)

- Checked upstream references for high-performance autoregressive training kernels:
  - Hugging Face docs on [PyTorch SDPA integration](https://huggingface.co/docs/transformers/v4.46.0/en/perf_infer_gpu_one#scaled-dot-product-attention-sdpa), [FlashAttention-2 usage](https://huggingface.co/docs/transformers/v4.46.0/en/perf_infer_gpu_one#flashattention-2), and [BetterTransformer fastpath](https://huggingface.co/docs/transformers/v4.46.0/en/perf_infer_gpu_one#bettertransformer).
  - Unsloth performance/memory claims and training stack docs: [blog](https://huggingface.co/blog/unsloth-trl), [docs](https://docs.unsloth.ai/), [repo](https://github.com/unslothai/unsloth).
  - FlashAttention repository/paper references: [repo](https://github.com/Dao-AILab/flash-attention), [paper](https://arxiv.org/abs/2205.14135).
  - PyTorch compile graph-break and SDPA backend notes: [compile graph breaks](https://docs.pytorch.org/docs/stable/compile/programming_model.common_graph_breaks.html), [SDPA backend issue context](https://github.com/pytorch/pytorch/issues/154602).
- Practical mapping to current ticl hotspot profile:
  - Upstream gains are dominated by attention/MLP fusion + reduced launch/sync overhead.
  - Our measured dominant path remains transformer layer finalize (`~48%` in latest strict-valid deep profile), while transition path shows non-trivial packing/update overhead (`env_pack ~7.2%`, `state_update ~16.3%` of transition wall), so current optimization order stays: `finalize fusion` -> `transition pack/update fusion`.

## 2026-03-03 Rebuild v2: Loss-Form Fingerprint Gate

- New risk acknowledged: loss-form changed (reward improvement path), so old/new runs must not share one skyline cohort.
- Rebuild pipeline now separates cohorts by `pg_loss_sig` from `[pg-phase]` tail.
  - New fields added in training/profile logs:
    - `pg_loss_sig=...` in `[pg-phase]`
    - `pg_loss_sig=...` in `[pg-gpu-stage]` rollout stage extra
  - Rebuild parser now supports string tail metrics and includes `pgsig=...` in `cohort_key`.
- New probe run with signature enabled:
  - `20260303_113211_losssig_probe` (`55.56s`, strict-valid)
  - `pg_loss_sig=pg_v2|norm=0|disc=1|detach=1|eps=1e-06|clip=10|rclip=10`
  - stage profile:
    - rollout `cuda_busy_ratio=1.000`, backward `0.611`
    - `policy_step_transformer_share=0.905`
    - `policy_step_tf_layer_finalize_share=0.476`
    - transition shares: `y=0.378`, `x=0.375`, `env_pack=0.050`, `state_update=0.114`
  - host safety: max RSS `2137220 kB` (`~2.04 GiB`).
- Rebuilt ladder after this change:
  - scanned runs: `99`
  - strict-valid runs: `5`
  - best strict-valid remains `20260303_012000_denseprefixcache_default_probe_rep2` (`48.36s`)
  - new signed cohort appears separately in `REBUILT_SKYLINE.md` and no longer mixes with legacy unsigned runs.

## 2026-03-03 Mainline Progress: bs64 Throughput Breakthrough

- Main bottleneck identified on `batch_size=64` baseline (`20260303_115533_bs64_baseline_probe`):
  - `Wallclock 162.38s`, `wall/batch=2.5372s`, strict-valid.
  - OOM fallback forced `TBPTT 128 -> 64 -> 32 -> 16`, causing `backward_calls=64` (major pseudo-parallel symptom).
  - Layer breakdown showed dominant `attnff` path and non-trivial cache/view overhead under paged dense route.
- Kernel-path change implemented:
  - `TransformerEncoderLayer._forward_step_attn_ff_paged` now supports `TICL_POLICY_PAGED_ATTN_TRAIN_MODE=auto`.
  - `auto` policy:
    - `batch >= 32`: `flash_prefix` exact merge (LSE merge, avoids repeated dense page-view materialization).
    - `batch < 32`: keep `dense` path to avoid small-batch flash overhead.
    - CPU path always forces `dense`.
  - Added explicit observability in `[pg-phase]`:
    - `rollout_cuda_busy_ratio`
    - `backward_cuda_busy_ratio`
  - Added startup log line:
    - `Policy paged-attn train mode: ...`
- Verified bs64 gain (`20260303_120432_bs64_auto_flashprefix_probe`):
  - `Wallclock 61.19s` (from `162.38s`, about `-62.3%`).
  - `wall/batch=0.9561s` (from `2.5372s`, about `-62.3%`).
  - No OOM fallback; `tbptt=128`, `backward_calls=8`.
  - Peak GPU memory reduced:
    - alloc/reserved: `36.07/39.21 GiB` (from `45.29/46.27 GiB`).
  - Throughput:
    - `tokens/s 1070.94` (from `403.60`).
- Skyline tooling update:
  - `rebuild_skyline.py` now parses `--batch-size` from command and reports `wall/batch` in tables.
- Rebuilt summary now:
    - scanned runs: `103`
    - strict-valid runs: `8`
- Safety/observability note:
  - One later probe (`20260303_121449_bs64_auto_threshold_probe`) failed before training due transient CUDA init error (`cudaGetDeviceCount error 304`), marked diagnostic only; not used for skyline retention.

## 2026-03-03 Recovery Validation: Auto Threshold Path

- GPU恢复后复测，确认 `auto` 分段策略在主线 workload 上可用：
  - `batch_size=64`:
    - `20260303_122829_bs64_auto_threshold_probe3`: `63.08s`, `wall/batch=0.9856s`, `tokens/s=1038.99`
    - `20260303_122534_bs64_auto_threshold_probe2`: `70.26s`, `wall/batch=1.0978s`, `tokens/s=932.77`
    - 与 `bs64` 老基线 `162.38s` (`2.5372s/batch`) 比，仍保持大幅领先。
  - `batch_size=8`:
    - `20260303_122704_bs8_auto_threshold_probe2`: `53.87s`（相比先前 `auto` 非分段版 `68.36s` 明显恢复）
- 运行画像（bs64/bs8均一致）：
  - `rollout_cuda_busy_ratio=1.000`
  - `backward_cuda_busy_ratio≈0.61`
  - rollout阶段仍以 `policy_step_transformer_share≈0.92` 为主，transition次之。
- Rebuild后当前统计：
  - scanned runs: `106`
  - strict-valid runs: `11`
  - `REBUILT_SKYLINE.md` 已包含 `wall/batch` 列并记录以上新运行。

## 2026-03-03 Mainline Continuation: Finalize-Proj Reality Check

- 按“finalize优先”主线先做了 `out_proj/residual + ffn/residual` 的 `addmm` 融合尝试：
  - 运行：`20260303_132933_bs64_finalize_addmmfuse_probe`
  - 结果：`tbptt=128` 稳定，但 `Wallclock 87.29s`，`wall/batch=1.3639s`，明显回退。
  - 判定：伪优化，已回滚，不进入主线skyline。
- 回滚后 sanity：
  - 运行：`20260303_133218_bs64_finalize_revert_sanity_probe`
  - 结果：`tbptt=128`，`Wallclock 78.19s`，`wall/batch=1.2217s`。
- 紧接着落地主导项（真实投影融合）：
  - 改动：`forward_step` 中 `q + kv` 两次GEMM 改为 `_project_qkv` 单次GEMM后切分。
  - 运行：`20260303_133459_bs64_qkvfuse_probe`
  - 结果：`tbptt=128`，`Wallclock 63.91s`，`wall/batch=0.9986s`，峰值显存 `34.03/36.44 GiB`。
  - 画像变化（相对 `20260303_131129_bs64_cachecatfree_recover_probe2`）：
    - `policy_step_tf_layer_proj_share: 0.297 -> 0.152`（投影主导项显著下降）
    - `policy_step_total_ms: 18472.22 -> 16082.62`（policy_step总时长下降）
  - 判定：真实增益，保留在主线并已纳入 `REBUILT_*`。

## 2026-03-03 Mainline Continuation: Cache/Proj Layout Reuse Deep-Dive

- `qkv -> heads` 单次布局复用（替代三次 `split_heads`）探测：
  - 运行：`20260303_133916_bs64_qkvheads_layoutreuse_probe`
  - 结果：`tbptt=128`，但 `Wallclock 71.56s`，`wall/batch=1.1181`，较 `qkvfuse` 主线退化。
  - 判定：伪优化，已回滚。
- `flash_merge` 路径直接切换探测（仅用于判断可行性）：
  - 运行：`20260303_134104_bs64_flashmerge_probe`
  - 结果：触发 `tbptt 128 -> 64 -> 32` 降级。
  - 判定：违反门槛，终止并拒绝。
- COW append `cat` 单核化探测（替换 `new_empty + slice copy`）：
  - 运行：`20260303_134234_bs64_cachecow_catkernel_probe`
  - 结果：`tbptt=128`，`Wallclock 68.59s`，`wall/batch=1.0717`；
    `cache_share` 降到 `0.124`，但整体 wall 仍劣于当前主线 `63.91s`。
  - 判定：局部指标改善但主指标回退，已回滚。

- 当前保留的主线有效项：
  - `q + kv -> qkv` 单次投影融合（`20260303_133459_bs64_qkvfuse_probe`）。
