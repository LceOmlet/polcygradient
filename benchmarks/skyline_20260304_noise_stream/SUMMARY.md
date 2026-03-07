# Rollout Noise-Stream + TBPTT Observability Skyline (2026-03-04)

## Goal
- Keep fixed workload (`python -m ticl.fit_model rlpfn --epochs 1 --num-steps 1`) and improve:
  - per-batch wall time,
  - memory peak stability,
  - observability quality (remove pseudo metrics).

## Code changes in this skyline
- `pg_env_replay_steps` default set to `1` (throughput-first benchmarking baseline).
- Added rollout noise streaming path in policy rollout:
  - `TICL_POLICY_ROLLOUT_NOISE_STREAM` (default on),
  - `TICL_POLICY_ROLLOUT_NOISE_BLOCK_SIZE` (default 64),
  - avoids full-horizon noise tensor preallocation in non-strict-seed path.
- Added rollout noise profile fields:
  - `rollout_noise_mode`, `rollout_noise_block_size`, `rollout_transition_noise_wall_ms`.
- Fixed rollout-profile aggregation for grouped rollout path to preserve new noise fields.
- Fixed misleading noise timing scope:
  - old probe accidentally included non-noise transition work (`noise_share > 1` pseudo observation),
  - now counts only noise-related ops (sampling/injection/dropout/state-noise).
- Added TBPTT streaming overlap observability:
  - `rollout_forward_est_s`,
  - `rollout_instage_backward_s`,
  - `rollout_instage_backward_share`.
- Refactored family-group transition hot loop toward true batch-parallel math:
  - moved per-group reward clip/dropout/state-noise/clip-tanh post-processing
    to full-batch tensor operations (group loop keeps only generator calls and
    base state update writes).
  - reduced per-step Python-side subgroup post-processing overhead.
- Added host-memory observability in `pg-phase`:
  - `host_rss_gib`,
  - `host_avail_gib`.
- Added TBPTT stream backward merge capability (`TICL_POLICY_TBPTT_STREAM_MERGE_WINDOWS`)
  with OOM auto-fallback; default kept at `1` for skyline stability.

## Command

```bash
TICL_PROFILE_ROLLOUT_TIMING=1 TICL_PROFILE_ROLLOUT_BREAKDOWN=1 TICL_POLICY_STEP_PROFILE=1 \
CONDA_NO_PLUGINS=true conda run --no-capture-output -n rlpfn \
  python -u -m ticl.fit_model rlpfn \
  --seed-everything True \
  --epochs 1 --num-steps 1 \
  --validate False --rl-validate-enabled False --progress-bar False \
  --train-profiler-enabled True --train-profiler-wandb False --train-profiler-log-every-batches 1 \
  --train-gpu-observer-enabled True --train-gpu-observer-interval-sec 0.2
```

## Seeded A/B (same command, only noise mode differs)

- `stream-off`: `20260304_081253_streamoff_seeded.log`
- `stream-on`: `20260304_081429_streamon_seeded.log`

### Metrics (`stream-off -> stream-on`)

- `rollout_s`: `75.160 -> 73.348` (`-2.4%`)
- `backward_s`: `42.750 -> 41.993` (`-1.8%`)
- epoch wallclock: `75.21s -> 73.40s` (`-2.4%`)
- peak alloc/reserved: `30.25/35.17 GiB -> 30.13/35.05 GiB`
- rollout `gpu_util_avg`: `25.66 -> 27.53`
- rollout noise mode: `full_prealloc -> block_stream`

Interpretation:
- `block_stream` is a real (small but consistent) improvement under fixed seed/workload.
- Memory peak also improves slightly; no OOM fallback observed.

## Mainline after group-loop vectorization + TBPTT merge capability

### Command

Same as above, plus:

```bash
TICL_POLICY_ROLLOUT_NOISE_STREAM=1
```

### Seeded runs

- Previous retained seeded baseline:
  - `20260304_081429_streamon_seeded.log`
- New code (replicate runs):
  - `20260304_083427_mainline_seeded_groupvec_defaultmerge1.log`
  - `20260304_083754_mainline_seeded_groupvec_defaultmerge1_rep2.log`

### Metrics (seeded baseline -> new mean)

- `rollout_s`: `73.348 -> 71.818` (mean of two runs, about `-2.1%`)
- `backward_s`: `41.993 -> 41.014` (mean, about `-2.3%`)
- rollout `gpu_util_avg`: `27.53 -> 26.59` (similar magnitude; not the optimization target)
- peak alloc/reserved: `30.13/35.05 GiB -> 30.13/35.05 GiB` (stable)
- new overlap split (mean):
  - `rollout_forward_est_s ≈ 30.804`
  - `rollout_instage_backward_share ≈ 0.572`

Interpretation:
- Mainline speedup is real under fixed seed and fixed workload.
- TBPTT overlap remains the dominant pseudo-parallel component inside rollout stage.
- Transition-loop refactor reduced subgroup post-processing cost (`transition_state_update_share`
  from older ~`0.058` to ~`0.026` in seeded runs), but dominant time is still
  policy transformer + in-stage backward.

## New hard-kernel pass (2026-03-04): transition async-group launch + flash-prefix async scheduling

### Code changes

- `EnvironmentPrior` transition loop:
  - upgraded from "group-local y/x async + immediate wait" to true
    **cross-group async launch + unified sync/commit**.
  - added observability fields:
    - `rollout_transition_async_enabled`
    - `rollout_transition_group_launch_share`
    - `rollout_transition_group_sync_share`
  - `TICL_POLICY_TRANSITION_STREAM_FUSION` default switched to on (`1`) for
    non-strict-seed vectorized family-group rollout (`group_count > 1`).
- Transformer step scheduling:
  - added async dual-stream dispatch for paged `flash_prefix` path:
    - prefix flash attention and tail flash attention launch concurrently,
    - merge after one stream wait.
  - env flag: `TICL_POLICY_FLASH_PREFIX_ASYNC` (default on).
  - startup observability line: `Policy flash-prefix async: ...`.

### A/B logs (same fixed command with layer profile)

- baseline (old behavior): `20260304_0900_baseline_layerprofile_noseed.log`
- new-on (both async paths enabled): `20260304_0920_asyncgroup_flashprefixasync_noseed.log`
- new-off ablation (same code, both async paths forced off):
  `20260304_0930_ablation_off_noseed.log`

### Key metrics

- `baseline -> new-on`
  - `rollout_s`: `72.772 -> 71.547` (`-1.68%`)
  - `backward_s`: `41.112 -> 40.486` (`-1.52%`)
  - `policy_step_total_ms`: `15498.59 -> 15433.92` (`-0.42%`)
  - `policy_step_tf_layer_total_ms`: `14210.40 -> 14148.95` (`-0.43%`)
  - `policy_step_tf_layer_cache_share`: `0.203 -> 0.196`
  - `policy_step_tf_layer_attn_core_share`: `0.149 -> 0.132`
- `new-off -> new-on` (same code ablation)
  - `rollout_s`: `74.746 -> 71.547` (`-4.28%`)
  - `backward_s`: `43.298 -> 40.486` (`-6.49%`)
  - `policy_step_total_ms`: `16078.86 -> 15433.92` (`-4.01%`)
  - `policy_step_tf_layer_total_ms`: `14732.34 -> 14148.95` (`-3.96%`)

### Notes on observability

- when transition async-group mode is active, previous `transition_y_share` and
  `transition_x_share` are no longer meaningful wall decomposition (launch-only
  path). New launch/sync shares are now emitted to avoid pseudo interpretation.

### Non-seeded check

- `20260304_083603_mainline_noseed_groupvec_defaultmerge1.log`
  - `rollout_s=68.407`
  - `backward_s=39.146`
  - peak `23.01/25.79 GiB`

## TBPTT merge-window probe (negative / guarded)

- Probe log: `20260304_083256_mainline_seeded_after_groupvec_tbpttmerge.log`
- With `TICL_POLICY_TBPTT_STREAM_MERGE_WINDOWS=2`, run hit OOM fallback:
  - reduced merge windows to `1`,
  - then reduced TBPTT window `64 -> 32`.

Interpretation:
- Coarser streaming-backward merge can increase activation pressure enough to
  trigger OOM on this workload.
- Feature is retained as explicit opt-in + auto-fallback, but default remains
  `1` to protect skyline and safety.

## Observability correction evidence

- Pre-fix probe (`20260304_080631_streamoff.log`) reported:
  - `rollout_transition_noise_share=1.017` (invalid, pseudo observation).
- Post-fix probe (`20260304_081000_streamoff_fixednoise.log`) reported:
  - `rollout_transition_noise_share=0.064` (reasonable).

## TBPTT overlap evidence (new metrics)

- Probe: `20260304_082004_streamon_seeded_tbpttobs.log`
- Key fields:
  - `rollout_s=76.288`
  - `backward_s=43.756`
  - `rollout_forward_est_s=32.531`
  - `rollout_instage_backward_s=43.756`
  - `rollout_instage_backward_share=0.574`

Interpretation:
- In streaming-TBPTT mode, a large fraction of the measured rollout stage is actually in-stage backward work.
- This separates true rollout-forward bottlenecks from pseudo “rollout too slow” observations.

## Continuation pass (2026-03-04): group commit de-serialization + paged-tail freeze

### Code changes

- Transition hot loop (`EnvironmentPrior`):
  - removed per-step subgroup `cat/pad` aggregation in transition commit path,
  - switched to direct slice writes into preallocated `reward_next_raw/state_next`.
- TBPTT window boundary cache scheduling:
  - paged cache detach now marks `tail_frozen=True`,
  - COW append path starts a fresh tail page when `tail_frozen`, avoiding
    repeated copy-growth of long detached history tail.

### Logs

- reference (before this pass): `20260304_1000_mainline_after_groupmeta_tbpttmultiroot_noseed.log`
- new run #1: `20260304_1025_tailfreeze_groupassign_noseed.log`
- new run #2 (replicate): `20260304_1033_tailfreeze_groupassign_noseed_rep2.log`

### Key observations

- vs reference `1000`, replicate `1033`:
  - `rollout_s`: `61.625 -> 64.060` (`+3.95%`)
  - `backward_s`: `35.413 -> 36.724` (`+3.70%`)
  - `policy_step_tf_layer_cache_share`: `0.204 -> 0.190` (`-6.9%`)
  - paged dispatch mix changed:
    - `single/flashp = 1536/10752 -> 768/11520`
- run `1025` is a high-variance non-seeded outlier (`rollout_s=80.336`,
  transition dominated by launch share `0.912`).

Interpretation:
- this pass does reduce transformer cache-path share (targeted pseudo-serial
  copy component), but non-seeded rollout wall-time variance remains high.
- For strict skyline decisions, next A/B should be fixed-seed on this branch.

## Fixed-seed A/B for `tail_frozen` (2026-03-04)

### Logs

- `tail_frozen=on`:
  - `20260304_1105_tailfreeze_on_seeded.log`
- `tail_frozen=off`:
  - `20260304_1110_tailfreeze_off_seeded.log`

### Key metrics (`off -> on`)

- `rollout_s`: `73.629 -> 74.118` (`+0.66%`, near-noise)
- `backward_s`: `42.255 -> 41.755` (`-1.18%`)
- `rollout_instage_backward_share`: `0.574 -> 0.563` (improved)
- `policy_step_tf_layer_cache_share`: `0.201 -> 0.184` (improved)
- peak alloc/reserved: `30.19/34.32 GiB -> 24.22/25.87 GiB` (large memory win)

Decision:
- Keep `tail_frozen` path enabled by default (`TICL_POLICY_TAIL_FREEZE=1`) for
  major VRAM reduction with no material wall-time regression.

### Post-fix repeat (authoritative, same branch after compile-path hardening)

- `tail_frozen=on`: `20260304_1220_tailfreeze_on_seeded_postcompilefix.log`
- `tail_frozen=off`: `20260304_1225_tailfreeze_off_seeded_postcompilefix.log`

Metrics (`off -> on`):

- `rollout_s`: `75.694 -> 71.218` (`-5.91%`)
- `backward_s`: `43.043 -> 40.581` (`-5.72%`)
- `policy_step_total_ms`: `16179.56 -> 15102.49` (`-6.66%`)
- `rollout_forward_est_s`: `32.651 -> 30.636` (`-6.17%`)
- `rollout_instage_backward_share`: `0.569 -> 0.570` (essentially flat)
- peak alloc/reserved: `30.19/34.32 GiB -> 24.22/25.87 GiB`

Final decision:

- `tail_frozen=on` is retained as the current fixed-seed skyline choice.

## TBPTT merge-window recheck under tail freeze (2026-03-04)

- Probe: `20260304_1205_tbpttmerge2_tailfreeze_on_seeded.log`
- Config: `TICL_POLICY_TBPTT_STREAM_MERGE_WINDOWS=2`
- Result:
  - immediate OOM fallback to `merge_windows=1`
  - `peak alloc/reserved` still surged to `43.66/46.30 GiB`
  - final run effectively stayed at merge=1 (`tbptt_stream_merge_windows=1`)

Decision:
- keep default `merge_windows=1`; merge>1 remains non-viable for current
  horizon/model under strict no-OOM skyline.

## `torch.compile` step-fusion probe (negative for this skyline)

### Logs

- first probe (recompile/cudagraph conflict): `20260304_1118_compile_warmup_seeded_tailfreeze_on.log`
  - warmup failed: `ok=0`, `wall_s=134.156`
- second probe after fixes (compile-mode + profiling contamination fixes):
  - `20260304_1143_compile_warmup_seeded_tailfreeze_on_fix2.log`
  - warmup succeeded but still expensive: `ok=1`, `wall_s=117.213`
  - run terminated before final pg-phase (compile startup still dominates).

### Code hardening added (retained)

- compile path now auto-adjusts mode for mutable paged KV:
  - `reduce-overhead -> default` when `TICL_POLICY_COMPILE_NO_CUDAGRAPHS=auto`
- compile-time profile contamination removed:
  - policy/layer `perf_counter` profiling auto-disabled inside compiling graph
  - warmup now clears accumulated step-profile stats to avoid pseudo observations

Decision:
- Do not promote `pg_torch_compile` into current throughput skyline for single-batch
  fixed workload; compile startup/recompile overhead still dominates.

## Next hard pass (non-compile, WIP)

### Code changes

- `TransformerEncoderLayer` paged-KV cache hot path:
  - added packed-page COW fast append (`_append_to_kv_pages_cow_packed`) to
    avoid per-step page-capacity scans in the dominant mutable-paged training path.
  - added `paged_packed` cache metadata and one-time packed detection fallback
    for dense->paged compatibility path.
  - paged cache no longer materializes/stores transient public `k/v` views for
    forward_step (keeps `k_pages/v_pages` as source of truth), reducing cache-path
    overhead and redundant tensor metadata churn.
- `train.py` observability:
  - added compile-warmup exclusion metrics to rollout logs/stages/wandb:
    - `compile_warmup_s`
    - `batch_wall_excl_compile_s`
    - `batch_wall_incl_compile_s`
    - `compile_warmup_ok`

### Validation status

- Unit regression:
  - `pytest -q ticl/tests/priors/test_environment_prior.py ticl/tests/test_train_policy_rollout_checkpoint.py`
  - result: `51 passed`
- Fixed-seed GPU benchmark re-run attempt:
  - `20260304_1330_tailfreeze_on_seeded_pagedpacked.log`
  - blocked by runtime CUDA init failure on host (`cudaGetDeviceCount error 304`,
    `nvidia-smi: Failed to initialize NVML`), so no new authoritative skyline
    numbers were produced in this pass.

## Continuation pass (2026-03-04, CUDA restored): non-compile hard-kernel optimization

### Scope

- target: keep `replay_step=1`, no compile dependency, reduce pseudo-serial
  launch/copy overhead in:
  - `policy_step_transformer_share` path (`forward_step` cache container churn),
  - `transition_group_share` path (env generator kernels).

### Code changes

- `TransformerEncoderLayer` / `TransformerEncoderSimple`:
  - added cache container reuse toggle: `TICL_POLICY_CACHE_CONTAINER_REUSE`
    (default `1`), reusing per-layer cache dict and encoder cache list in
    mutable rollout path to reduce per-step Python object churn.
- `EnvironmentPrior` env generators:
  - introduced `_batch_affine` and switched SCM/GP batch generator hot paths
    from `einsum("bi,bij->bj")` to CUDA `bmm/baddbmm` kernels.
  - strict CPU semantics retained (CPU path still uses original einsum) so
    strict-RNG semantic tests remain unchanged.
  - toggle: `TICL_POLICY_ENVGEN_BMM` (default `1`).

### Fixed-seed measurements

- pre-pass reference (same branch before envgen-bmm):
  - `20260304_1047_tailfreeze_on_seeded_layerprof_afterpush.log`
  - `rollout_s=71.777`, `backward_s=40.681`
  - `rollout_transition_wall_ms=13170.34`
- envgen-bmm on:
  - `20260304_1110_tailfreeze_on_seeded_layerprof_bmmgen.log`
    - `rollout_s=69.980`, `backward_s=39.919`
    - `rollout_transition_wall_ms=12290.55`
  - `20260304_1116_tailfreeze_on_seeded_layerprof_bmmgen_rep2.log`
    - `rollout_s=72.355`, `backward_s=41.626`
    - `rollout_transition_wall_ms=12509.27`
- envgen-bmm off (A/B):
  - `20260304_1123_tailfreeze_on_seeded_layerprof_bmmgen_off.log`
  - `rollout_s=73.050`, `backward_s=42.408`
  - `rollout_transition_wall_ms=13091.10`

Interpretation:

- `TICL_POLICY_ENVGEN_BMM=1` consistently lowers transition wall share in A/B
  and improves end-to-end wall in fixed-seed probes.
- `TICL_POLICY_CACHE_CONTAINER_REUSE` shows small/variable gains; retained as
  default-on toggle but not treated as the primary win in this pass.

### Mainline skyline probe (no layer-profile overhead)

- `20260304_1130_tailfreeze_on_seeded_mainline_bmmgen.log`
  - `rollout_s=68.847`
  - `backward_s=39.623`
  - `policy_step_total_ms=14490.08`
  - `rollout_transition_wall_ms=12145.68`
  - peak alloc/reserved: `24.22/25.86 GiB`

Compared with prior authoritative on-run (`20260304_1220_tailfreeze_on_seeded_postcompilefix.log`):

- `rollout_s`: `71.218 -> 68.847` (`-3.33%`)
- `backward_s`: `40.581 -> 39.623` (`-2.36%`)
- `policy_step_total_ms`: `15102.49 -> 14490.08` (`-4.06%`)

### TBPTT merge-window recheck on new branch

- `20260304_1140_tailfreeze_on_seeded_bmmgen_tbpttmerge2.log`
- with `TICL_POLICY_TBPTT_STREAM_MERGE_WINDOWS=2`, run again hit OOM fallback:
  - immediately reduced to merge=`1`
  - peak alloc/reserved rose to `43.65/46.28 GiB`

Decision:

- keep `merge_windows=1` as skyline default (merge>1 still non-viable under
  current horizon/model).

## Continuation pass (2026-03-04, code-only in current sandbox): transition fused generator + non-compile step launch reduction

### Code changes

- `EnvironmentPrior` transition hot loop:
  - enabled fused transition generator path (`x_next + reward_next` in one batched call)
    via `TICL_POLICY_FUSED_TRANSITION_GENERATOR` (default `1`).
  - integrated fused path into transition-group async scheduler:
    - fused groups use one stream (`stream_transition`) and unified sync,
    - non-fused groups keep dual `stream_y/stream_x`.
  - added observability fields:
    - `rollout_transition_fused_share`
    - `rollout_transition_fused_launch_share`
    - `rollout_transition_fused_calls`
    - `rollout_transition_fused_groups`
    - `rollout_transition_fused_enabled`
  - fields propagated through rollout aggregation and `stats` export.

- `TransformerEncoderLayer` (`forward_step`, non-compile path):
  - added flash-prefix low-length dense fallback threshold:
    - `TICL_POLICY_PAGED_ATTN_FLASHPREFIX_DENSE_MAX_TOKENS` (default `192`).
  - when `valid_len <= threshold` in paged `flash_prefix` train mode, route to
    single dense SDPA (one attention launch) instead of dual prefix/tail flash
    launches + merge.

- `train.py` observability:
  - rollout phase now logs/records/wandb-exports fused-transition shares/counts.
  - startup config print now includes:
    - `Policy transition fused generator`
    - `Policy transition stream fusion`
    - `Policy flash-prefix dense max tokens`

### Validation status in this sandbox

- syntax check:
  - `python -m py_compile ticl/priors/environment_prior.py ticl/models/layer.py ticl/train.py`
- unit checks (family-group policy rollout):
  - passed:
    - `test_environment_prior_rollout_with_policy_family_grouping_matches_serial_in_deterministic_setup`
    - `test_environment_prior_rollout_with_policy_family_grouping_batches_policy_calls`
    - `test_environment_prior_rollout_with_policy_family_grouping_uses_coarse_subgroups`

### Benchmark status

- attempted authoritative command (`python -m ticl.fit_model rlpfn --epochs 1 --num-steps 1`, with rollout profile flags),
  but current execution sandbox cannot initialize CUDA:
  - `torch.cuda.is_available() == False`
  - `cudaGetDeviceCount error 304`
  - `nvidia-smi: Failed to initialize NVML`
- therefore this pass is code-complete but **not yet GPU-measured** in current sandbox.

## Continuation pass (2026-03-04, CUDA restored): fixed-seed A/B + threshold-range validation

### Baseline command

```bash
TICL_PROFILE_ROLLOUT_TIMING=1 TICL_PROFILE_ROLLOUT_BREAKDOWN=1 TICL_POLICY_STEP_PROFILE=1 \
CONDA_NO_PLUGINS=true conda run --no-capture-output -n rlpfn \
  python -u -m ticl.fit_model rlpfn \
  --seed-everything True \
  --epochs 1 --num-steps 1 \
  --validate False --rl-validate-enabled False --progress-bar False \
  --train-profiler-enabled True --train-profiler-wandb False --train-profiler-log-every-batches 1 \
  --train-gpu-observer-enabled True --train-gpu-observer-interval-sec 0.2
```

Note:
- this pass runs with `policy torch.compile=False`, so no compile warmup is executed.
- skyline wall metric therefore equals warmup-excluded wall (for compile-enabled runs, use `batch_wall_excl_compile_s`).

### Hard A/B (same fixed seed, only new kernel-path switches differ)

- `fused_on`:
  - `20260304_113228_seeded_mainline_fused_on.log`
  - `rollout_s=58.168`, `backward_s=33.532`
  - `rollout_transition_wall_ms=7214.64`
  - `policy_step_total_ms=14536.16`
- `fused_off + dense0`:
  - `20260304_113349_seeded_mainline_fused_off_dense0.log`
  - `rollout_s=69.922`, `backward_s=39.634`
  - `rollout_transition_wall_ms=12411.39`
  - `policy_step_total_ms=15174.61`

Result (`off -> on`):
- `rollout_s`: `69.922 -> 58.168` (`-16.81%`)
- `backward_s`: `39.634 -> 33.532` (`-15.40%`)
- `rollout_transition_wall_ms`: `12411.39 -> 7214.64` (`-41.87%`)
- `policy_step_total_ms`: `15174.61 -> 14536.16` (`-4.20%`)

Interpretation:
- dominant win comes from transition hot-loop launch reduction (fused path), not parameter micro-tuning.
- this directly addresses transition-group pseudo-parallel bottleneck.

### Observability fix (avoid pseudo signal)

- `20260304_113844_seeded_mainline_fused_on_metricfix.log`
- previous fused async path could show `rollout_transition_fused_share=0.000` (misleading).
- after metric fix:
  - `rollout_transition_fused_share=0.815`
  - `rollout_transition_fused_launch_share=0.815`
- now fused share reflects actual async launch cost and avoids pseudo observation.

### Flash-prefix dense fallback parameter sweep (range validation)

All runs use `TICL_POLICY_FUSED_TRANSITION_GENERATOR=1`.

- `dense_max_tokens=0`:
  - `20260304_113523_seeded_ablation_fused_on_dense0.log`
  - `rollout_s=58.507`, `backward_s=33.719`, `policy_step_total_ms=14611.49`
- `dense_max_tokens=64`:
  - `20260304_114020_seeded_ablation_fused_on_dense64.log`
  - `rollout_s=57.575`, `backward_s=32.830`, `policy_step_total_ms=14573.14`
- `dense_max_tokens=512`:
  - `20260304_114140_seeded_ablation_fused_on_dense512.log`
  - `rollout_s=59.208`, `backward_s=33.602`, `policy_step_total_ms=15148.44`

Range conclusion:
- too-large threshold (`512`) regresses wall time (dense path over-expands).
- small threshold (`64`) is best among tested values on fixed workload.
- default updated to:
  - `TICL_POLICY_PAGED_ATTN_FLASHPREFIX_DENSE_MAX_TOKENS=64`
- this is a bounded non-compile optimization range, reducing risk of parameter-induced instability.

### Mainline check after default update

- `20260304_114315_seeded_mainline_default64.log`
  - confirms startup default print: `Policy flash-prefix dense max tokens: 64`
  - `rollout_s=59.595`, `backward_s=34.533`
  - `rollout_transition_fused_share=0.820`

Note:
- short fixed-seed single-batch runs still have normal runtime variance; skyline decisions use A/B direction + multi-run evidence, not one outlier.

## Continuation pass (2026-03-04): warmup-excluded skyline metric hardening + TBPTT merge safety

### Metric hardening

- `train.py` rollout logs now always emit:
  - `batch_wall_excl_compile_s`
  - `batch_wall_incl_compile_s`
- regardless of whether compile warmup is active.
- when compile warmup exists, `compile_warmup_s` remains explicitly emitted.

Purpose:
- skyline parser can use a single stable metric (`batch_wall_excl_compile_s`) as
  the primary throughput KPI and avoid warmup contamination.

### TBPTT merge auto (safety-first)

- added optional auto-merge scheduler:
  - `TICL_POLICY_TBPTT_STREAM_MERGE_AUTO` (default `0`)
  - `TICL_POLICY_TBPTT_STREAM_MERGE_AUTO_MAX_WINDOWS` (default `2`)
  - `TICL_POLICY_TBPTT_STREAM_MERGE_AUTO_MIN_FREE_GB` (default `18`)
  - `TICL_POLICY_TBPTT_STREAM_MERGE_AUTO_RESERVED_FRAC` (default `0.72`)
- rationale:
  - attempt to reduce backward launch count only when memory headroom is safe.
  - avoid silent OOM/instability by keeping default-off until per-workload
    range is validated.

### Auto-merge probe and decision

- probe log: `20260304_115246_seeded_mainline_tbpttmergeauto.log`
  - auto path attempted merge>1, hit OOM, then auto-fell back to merge=1.
  - peak alloc/reserved reached `45.56/46.28 GiB`.
- decision:
  - keep `TICL_POLICY_TBPTT_STREAM_MERGE_AUTO=0` as default for skyline stability.
  - retain feature for explicit, workload-specific exploration.

### Post-hardening mainline check

- `20260304_115430_seeded_mainline_default_after_mergeautooff.log`
  - `rollout_s=58.061`
  - `backward_s=32.982`
  - `batch_wall_excl_compile_s=91.043`
  - `batch_wall_incl_compile_s=91.043`
  - `rollout_instage_backward_share=0.568`

## Small-scope Triton/compile probe retry (2026-03-04, fixed seed, single threshold)

### Scope (as requested)

- single probe configuration:
  - `pg_torch_compile=True`
  - backend `inductor`, mode `reduce-overhead` (runtime auto-adjusted to `default`)
  - `TICL_POLICY_PAGED_ATTN_FLASHPREFIX_DENSE_MAX_TOKENS=64`
  - compile warmup isolated:
    - `TICL_POLICY_COMPILE_WARMUP=1`
    - `TICL_POLICY_COMPILE_WARMUP_STEPS=1`
    - `TICL_POLICY_COMPILE_WARMUP_SAMPLES=2`
    - `TICL_POLICY_COMPILE_WARMUP_CHUNK=8`
- comparison baseline (same fixed seed/workload, eager):
  - `20260304_121550_seeded_eager_baseline_dense64.log`
- probe run:
  - `20260304_121732_seeded_triton_probe_inductor_dense64_retry.log`

### A/B result (eager -> compile)

- `batch_wall_excl_compile_s`: `91.057 -> 111.666` (`+22.6%`, worse)
- `batch_wall_incl_compile_s`: `91.057 -> 133.340`
- `compile_warmup_s`: `0 -> 21.674` (`compile_warmup_ok=1`)
- `rollout_s`: `57.914 -> 76.600` (`+32.3%`, worse)
- `backward_s`: `33.143 -> 35.066` (`+5.8%`, worse)
- `policy_step_total_ms`: `14637.78 -> 14782.40` (`+1.0%`, no gain)
- peak alloc/reserved: `30.97/31.72 GiB -> 30.99/31.91 GiB` (similar)

### Probe diagnosis

- compile run emitted `torch._dynamo hit config.recompile_limit (8)` with
  dynamic paged-KV shape mismatch (`k_pages` length change), i.e. compile storm
  risk is still present in this path.

Decision:
- do **not** mainline Triton/compile for current skyline.
- keep skyline KPI on `batch_wall_excl_compile_s` and continue non-compile
  kernel-path optimization as the primary track.

## Continuation pass (2026-03-04): transition async inline-commit probe + TBPTT merge guard

### Scope

- continue attacking dominant pseudo-serial sections:
  - transition-group async scheduling in `EnvironmentPrior` rollout hot loop,
  - TBPTT streaming backward launch granularity with strict OOM safety.

### Code changes retained

- `train.py`:
  - added TBPTT stream-merge runtime guard (non-compile):
    - `TICL_POLICY_TBPTT_STREAM_MERGE_GUARD` (default `1`)
    - `TICL_POLICY_TBPTT_STREAM_MERGE_GUARD_MIN_FREE_GB` (default `12.0`)
    - `TICL_POLICY_TBPTT_STREAM_MERGE_GUARD_RESERVED_FRAC` (default `0.82`)
  - guard checks CUDA memory before buffering additional TBPTT window roots and
    can force early flush to avoid merge-induced memory spikes.
  - added per-chunk preflight guard downgrade (`merge_windows -> 1`) when
    initial memory headroom is already below guard thresholds.
  - observability fields added:
    - `tbptt_stream_merge_guard_flushes` (log/stage/wandb).
    - `tbptt_stream_merge_guard_prefallbacks` (log/stage/wandb).
  - startup now prints TBPTT merge-guard config.

### Code changes probed then reverted (not mainlined)

- `EnvironmentPrior` transition path async inline-commit:
  - tried writing `state_next/reward_next_raw` directly inside per-group streams
    and removing default-stream post-sync state-update loop.
- fixed-seed probes showed decomposition shifts (`transition_state_update_share`
  near `0`), but end-to-end wall did not improve robustly; variant was reverted
  to avoid pseudo optimization drift.

### Logs

- baseline reference (pre-pass eager):
  - `20260304_121550_seeded_eager_baseline_dense64.log`
- async-inline probe:
  - `20260304_123036_seeded_mainline_asyncinline_default.log`
  - `20260304_123359_seeded_mainline_asyncinline_default_rep2.log`
- merge-window probe with guard:
  - `20260304_123206_seeded_mainline_asyncinline_merge2_guard.log`
- merge-window probe after preflight-guard extension:
  - `20260304_123909_seeded_mainline_merge2_guard_prefight.log`
- post-revert + retained merge-guard mainline check:
  - `20260304_123614_seeded_mainline_after_revert_with_mergeguard.log`

### Key observations

- async-inline transition probe:
  - reduced measured transition post-sync state-update share to ~`0`,
  - but fixed-seed wall metrics did not show stable gain across repeats.
- TBPTT merge=2 with guard still hit first-attempt OOM and auto fallback:
  - `[pg-oom] ... reducing TBPTT stream merge windows to 1`
  - peak alloc/reserved reached `45.56/46.28 GiB`
  - final effective stats returned to merge=`1` (`tbptt_stream_backward_calls=16`).
- preflight guard extension did not eliminate this case:
  - `tbptt_stream_merge_guard_prefallbacks` stayed `0` in the probe log,
  - OOM still occurred before any effective merge reduction, then runtime
    fallback forced merge windows to `1`.

Decision:
- keep TBPTT merge-guard instrumentation/safety path.
- do not mainline async-inline transition commit path (reverted).
- keep skyline on non-compile eager path with `batch_wall_excl_compile_s` as KPI.

## Continuation pass (2026-03-04): policy-step cache launch compression + compile recompile observability

### Scope

- target dominant forward hotspot `policy_step_transformer_share` on non-compile path.
- avoid pseudo tuning; reduce real kernel launch/copy count in `forward_step` cache path.
- add CLI-level compile/recompile observability so warmup pollution and recompile storms are separable from KPI.

### Code changes

- `ticl/models/layer.py`
  - `TransformerEncoderLayer._concat_dim2` switched to single `torch.cat` kernel path.
  - COW tail-page growth paths (`_append_to_kv_pages_cow*`) now reuse `_concat_dim2`
    instead of manual `new_empty + two slice-copy` sequence.
  - added ablation switch:
    - `TICL_POLICY_CAT_FUSION` (default `1`, `0` restores old manual copy path).
- `ticl/train.py`, `ticl/cli_parsing.py`, `ticl/model_configs.py`, `ticl/fit_model.py`
  - added compile observability CLI knobs:
    - `--pg-compile-observe-recompiles`
    - `--pg-compile-observe-log-every-batches`
    - `--pg-compile-observe-output-path`
    - `--pg-compile-observe-reset-after-warmup`
  - new per-batch metrics (log/stage/wandb/jsonl):
    - `compile_counter_delta_nonzero`
    - `compile_counter_delta_total_abs`
    - `compile_counter_recompiles`
    - `compile_counter_graph_breaks`
    - `compile_counter_unique_graphs`
  - warmup completion can reset compile-counter baseline to remove warmup contamination.

### Fixed-seed hard A/B (`TICL_POLICY_CAT_FUSION`)

Command base (same workload/seed, only cat-fusion toggle differs):

```bash
TICL_PROFILE_ROLLOUT_TIMING=1 TICL_PROFILE_ROLLOUT_BREAKDOWN=1 TICL_POLICY_STEP_PROFILE=1 \
CONDA_NO_PLUGINS=true conda run --no-capture-output -n rlpfn \
  python -u -m ticl.fit_model rlpfn \
  --seed-everything True \
  --epochs 1 --num-steps 1 \
  --validate False --rl-validate-enabled False --progress-bar False \
  --train-profiler-enabled True --train-profiler-wandb False --train-profiler-log-every-batches 1 \
  --train-gpu-observer-enabled True --train-gpu-observer-interval-sec 0.2
```

Logs:

- `cat_fusion=on`: `20260304_133800_seeded_catfusion_on.log`
- `cat_fusion=off`: `20260304_133950_seeded_catfusion_off.log`

Metrics (`off -> on`):

- `rollout_s`: `61.405 -> 56.086` (`-8.7%`)
- `backward_s`: `35.856 -> 32.027` (`-10.7%`)
- `batch_wall_excl_compile_s`: `97.261 -> 88.113` (`-9.4%`)
- `policy_step_total_ms`: `15104.23 -> 13397.00` (`-11.3%`)
- `policy_step_transformer_share`: `0.929 -> 0.918`

### Layer-profile evidence (same branch)

Logs:

- `cat_fusion=on`: `20260304_132100_seeded_mainline_catfuse_layerprofile.log`
- `cat_fusion=off`: `20260304_134430_seeded_catfusion_off_layerprofile.log`

Key decomposition (`off -> on`):

- `policy_step_tf_layer_total_ms`: `13895.13 -> 12235.98` (`-11.9%`)
- `policy_step_tf_layer_cache_share`: `0.189 -> 0.077` (major cache-path reduction)
- paged dispatch remained identical route family (`single/flashp`, no dense fallback promotion).

Conclusion:

- dominant pseudo-serial cache copy/concat path is a real bottleneck;
  replacing multi-copy sequence with cat-fused path provides robust end-to-end gain.
- this is retained as mainline default (`TICL_POLICY_CAT_FUSION=1`).

### Compile observability probe (CLI path verification)

Probe log:

- `20260304_134900_seeded_compileobserve_probe_v2.log`
- counters jsonl:
  `20260304_134900_seeded_compileobserve_probe_v2.counters.jsonl`

Probe config highlights:

- `--pg-torch-compile True`
- `--pg-compile-observe-recompiles True`
- `--pg-compile-observe-log-every-batches 1`
- `--pg-compile-observe-reset-after-warmup True`
- warmup isolated by env:
  `TICL_POLICY_COMPILE_WARMUP=1`,
  `TICL_POLICY_COMPILE_WARMUP_STEPS=1`,
  `TICL_POLICY_COMPILE_WARMUP_SAMPLES=2`,
  `TICL_POLICY_COMPILE_WARMUP_CHUNK=8`.

Observed:

- compile log now emits explicit per-batch counter deltas.
- `compile_counter_recompiles=1` (capturing `unimplemented.recompile_limit reached` delta).
- KPI remains warmup-excluded:
  - `batch_wall_excl_compile_s=102.777`
  - `batch_wall_incl_compile_s=123.572`
  - `compile_warmup_s=20.795`

Decision:

- keep compile path off mainline skyline.
- keep new compile observability knobs enabled for targeted probe runs to isolate recompile pollution.

## Continuation pass (2026-03-06): zero-prefix flash fastpath hard A/B + hotspot re-profile

### Fixed-seed A/B (`TICL_POLICY_FLASH_PREFIX_ZERO_FASTPATH`)

Command base (same workload/seed, only zero-fastpath toggle differs):

```bash
TICL_PROFILE_ROLLOUT_TIMING=1 TICL_PROFILE_ROLLOUT_BREAKDOWN=1 TICL_POLICY_STEP_PROFILE=1 \
TICL_POLICY_COMPILE_WARMUP=0 \
CONDA_NO_PLUGINS=true conda run --no-capture-output -n rlpfn \
  python -u -m ticl.fit_model rlpfn \
  --seed-everything True \
  --epochs 1 --num-steps 1 \
  --validate False --rl-validate-enabled False --progress-bar False \
  --train-profiler-enabled True --train-profiler-wandb False --train-profiler-log-every-batches 1 \
  --train-gpu-observer-enabled True --train-gpu-observer-interval-sec 0.2 \
  --pg-env-replay-steps 1 --pg-tbptt-window 64
```

Logs:

- `zero_fastpath=off`: `20260306_seeded_zeroprefix_fastpath_off_rerun.log`
- `zero_fastpath=on`: `20260306_seeded_zeroprefix_fastpath_on_rerun.log`

Metrics (`off -> on`):

- `batch_wall_excl_compile_s`: `89.891 -> 83.276` (`-7.4%`)
- `rollout_s`: `57.092 -> 53.227` (`-6.8%`)
- `backward_s`: `32.800 -> 30.049` (`-8.4%`)
- `policy_step_total_ms`: `14049.68 -> 12888.45` (`-8.3%`)
- `policy_step_transformer_share`: `0.917 -> 0.910`
- peak alloc/reserved: `30.97/31.72 GiB -> 30.93/31.68 GiB` (stable)

Decision:

- retain `TICL_POLICY_FLASH_PREFIX_ZERO_FASTPATH=1` as mainline default.

### Layer-profile diagnostic (same branch, fixed seed)

Log:

- `20260306_seeded_zeroprefix_fastpath_on_layerprofile.log`

Key decomposition:

- `policy_step_tf_layer_total_ms=11484.23`
- `policy_step_tf_layer_attnff_share=0.668`
- `policy_step_tf_layer_attn_core_share=0.081`
- `policy_step_tf_layer_finalize_share=0.535`
- `policy_step_tf_layer_finalize_ffn_share=0.298`
- `policy_step_paged_dispatch(single/flashp/flashm/dense)=768/11520/0/0`
- `policy_step_flashp_avg_tokens(valid/prefix/tail)=544.5/0.0/544.5`

Interpretation:

- current non-compile hotspot is no longer flash attention core; dominant cost is
  finalize/FFN path inside per-step transformer layers.
- zero-prefix flash path coverage is effectively full under current auto mode
  (`prefix≈0` on flash-prefix calls), so this fastpath addresses a true dominant branch.

### Observability enhancement (to avoid pseudo interpretation)

Code now emits explicit zero-fastpath coverage in paged dispatch tuple:

- `policy_step_paged_dispatch(single/flashp/flashp0/flashm/dense)=...`

Validation log:

- `20260306_seeded_zeroprefix_fastpath_on_layerprofile_v2.log`
- observed: `policy_step_paged_dispatch(...)=768/11520/11520/0/0`
  (all flash-prefix calls used zero-fastpath in this workload).

### Negative probes (not mainlined)

- `TICL_POLICY_INPLACE_PAGED_KV=1` probe:
  - `20260306_seeded_probe_inplace_paged_kv1.log`
  - immediate OOM fallback at `tbptt=64` (`TBPTT window reduced to 32`), aborted.
- `TICL_POLICY_TBPTT_STREAM_MERGE_AUTO=1` probe:
  - `20260306_seeded_probe_tbptt_merge_auto1.log`
  - immediate OOM fallback (`merge_windows -> 1`), aborted.

Decision:

- keep skyline defaults conservative on these two knobs; continue mainline on
  zero-fastpath + cat-fusion path with fixed-seed KPI `batch_wall_excl_compile_s`.

## Continuation pass (2026-03-06): step-level 2D kernel-dispatch compression (non-compile)

### Code changes

- `TransformerEncoderLayer.forward_step`:
  - added 2D projection fastpath for single-token step:
    - `TICL_POLICY_STEP_PROJ_2D` (default `1`)
  - avoids per-layer `permute/unsqueeze` projection route on `(B,1,E)` and
    dispatches `in_proj` directly on `(B,E)`.
- `TransformerEncoderSimple.forward_step`:
  - added 2D inter-layer loop for single-token step:
    - `TICL_POLICY_STEP_LAYER_2D_LOOP` (default `1`)
  - keeps layer loop in `(B,E)` and restores `(1,B,E)` only at encoder output.
- startup observability:
  - `Policy step projection 2D fastpath: ...`
  - `Policy step layer 2D loop: ...`

### Fixed-seed hard A/B (no layer-profile overhead in KPI run)

Command base (same seed/workload; only 2D fastpath toggles differ):

```bash
TICL_PROFILE_ROLLOUT_TIMING=1 TICL_PROFILE_ROLLOUT_BREAKDOWN=1 TICL_POLICY_STEP_PROFILE=1 \
TICL_POLICY_FLASH_PREFIX_ZERO_FASTPATH=1 TICL_POLICY_COMPILE_WARMUP=0 \
CONDA_NO_PLUGINS=true conda run --no-capture-output -n rlpfn \
  python -u -m ticl.fit_model rlpfn \
  --seed-everything True \
  --epochs 1 --num-steps 1 \
  --validate False --rl-validate-enabled False --progress-bar False \
  --train-profiler-enabled True --train-profiler-wandb False --train-profiler-log-every-batches 1 \
  --train-gpu-observer-enabled True --train-gpu-observer-interval-sec 0.2 \
  --pg-env-replay-steps 1 --pg-tbptt-window 64
```

Logs:

- `step2d=off` (`TICL_POLICY_STEP_PROJ_2D=0 TICL_POLICY_STEP_LAYER_2D_LOOP=0`):
  - `20260306_seeded_step2d_off_nolayer.log`
- `step2d=on` (`TICL_POLICY_STEP_PROJ_2D=1 TICL_POLICY_STEP_LAYER_2D_LOOP=1`):
  - `20260306_seeded_step2d_on_nolayer.log`

Metrics (`off -> on`):

- `batch_wall_excl_compile_s`: `89.321 -> 83.701` (`-6.3%`)
- `rollout_s`: `56.735 -> 53.168` (`-6.3%`)
- `backward_s`: `32.587 -> 30.533` (`-6.3%`)
- `policy_step_total_ms`: `13448.21 -> 12422.19` (`-7.6%`)
- peak alloc/reserved: `30.93/31.68 GiB -> 30.93/31.68 GiB` (stable)

### Layer-profile cross-check

Layer-profile runs (`TICL_TRANSFORMER_LAYER_STEP_PROFILE=1`) used for hotspot decomposition:

- `20260306_seeded_step2d_off.log`
- `20260306_seeded_step2d_on.log`

Observed:

- projection share decreased (`0.196 -> 0.173`) and layer total ms decreased
  (`11664.03 -> 11356.64`) on `step2d=on`,
- but layer-profile instrumentation itself adds variance to end-to-end KPI.

Decision:

- keep both 2D step fastpaths enabled by default (`=1`) and use
  non-layer-profile fixed-seed KPI (`batch_wall_excl_compile_s`) for skyline judgment.

## Continuation pass (2026-03-06): policy-step token allocation fastpath (non-compile)

### Code changes

- `train._build_policy_step_fn`:
  - added grad-path token allocation fastpath gate:
    - `TICL_POLICY_STEP_TOKEN_ALLOC_OPT` (default `1`)
  - `on` path removes redundant per-step work:
    - avoids `zeros + zero_` double init for `x_token`
    - avoids `clone()` for detached scalar `y_token`
- startup observability:
  - `Policy step token alloc fastpath: ...`

### Fixed-seed hard A/B (`TICL_POLICY_STEP_TOKEN_ALLOC_OPT`)

Command base (same seed/workload; only token-alloc toggle differs):

```bash
TICL_PROFILE_ROLLOUT_TIMING=1 TICL_PROFILE_ROLLOUT_BREAKDOWN=1 TICL_POLICY_STEP_PROFILE=1 \
CONDA_NO_PLUGINS=true conda run --no-capture-output -n rlpfn \
  python -u -m ticl.fit_model rlpfn \
  --seed-everything True \
  --epochs 1 --num-steps 1 \
  --validate False --rl-validate-enabled False --progress-bar False \
  --train-profiler-enabled True --train-profiler-wandb False --train-profiler-log-every-batches 1 \
  --train-gpu-observer-enabled True --train-gpu-observer-interval-sec 0.2 \
  --pg-env-replay-steps 1 --pg-tbptt-window 64
```

Logs:

- `token_alloc_opt=off` (`TICL_POLICY_STEP_TOKEN_ALLOC_OPT=0`):
  - `20260306_seeded_tokenallocopt_off.log`
- `token_alloc_opt=on` (`TICL_POLICY_STEP_TOKEN_ALLOC_OPT=1`):
  - `20260306_seeded_tokenallocopt_on_ab.log`

Metrics (`off -> on`):

- `batch_wall_excl_compile_s`: `85.342 -> 83.703` (`-1.9%`)
- `rollout_s`: `54.269 -> 53.284` (`-1.8%`)
- `backward_s`: `31.074 -> 30.419` (`-2.1%`)
- `policy_step_total_ms`: `12793.95 -> 12607.49` (`-1.5%`)
- peak alloc/reserved: unchanged (`30.93/31.68 GiB`)

Decision:

- keep `TICL_POLICY_STEP_TOKEN_ALLOC_OPT=1` as default mainline optimization.

## Continuation pass (2026-03-06): TBPTT OOM fail-fast + token-layout probe

### Code changes

- Added explicit policy OOM fail-fast knob (disable all TBPTT/chunk fallback and raise immediately):
  - CLI: `--pg-oom-fail-fast`
  - config: `optimizer.pg_oom_fail_fast` (default `False`)
  - effective behavior = `--pg-oom-fail-fast` OR env `TICL_POLICY_OOM_FAIL_FAST=1`
- Added rollout token-layout prepack probe switch in family-group rollout hot loop:
  - `TICL_POLICY_TOKEN_LAYOUT_PREPACK` (default `0`)
  - `on`: precompute token write indices/masks outside `for t in range(n_samples)` to remove repeated
    `arange/nonzero` construction per step.

### Fixed-seed A/B (`TICL_POLICY_TOKEN_LAYOUT_PREPACK`, fail-fast enabled)

Command base (same seed/workload; only token-layout toggle differs):

```bash
TICL_PROFILE_ROLLOUT_TIMING=1 TICL_PROFILE_ROLLOUT_BREAKDOWN=1 TICL_POLICY_STEP_PROFILE=1 \
CONDA_NO_PLUGINS=true conda run --no-capture-output -n rlpfn \
  python -u -m ticl.fit_model rlpfn \
  --seed-everything True \
  --epochs 1 --num-steps 1 \
  --validate False --rl-validate-enabled False --progress-bar False \
  --pg-env-replay-steps 1 --pg-tbptt-window 64 \
  --pg-oom-fail-fast True --pg-oom-reduce-tbptt-first False \
  --policy-rollout-chunk-autotune False \
  --train-profiler-enabled True --train-profiler-wandb False --train-profiler-log-every-batches 1 \
  --train-gpu-observer-enabled True --train-gpu-observer-interval-sec 0.2
```

Logs:

- `token_layout_prepack=off` (`TICL_POLICY_TOKEN_LAYOUT_PREPACK=0`):
  - `20260306_seeded_tokenlayout_off_failfast.log`
- `token_layout_prepack=on` (`TICL_POLICY_TOKEN_LAYOUT_PREPACK=1`):
  - `20260306_seeded_tokenlayout_on_failfast.log`

Metrics (`off -> on`):

- `batch_wall_excl_compile_s`: `83.250 -> 83.068` (`-0.22%`)
- `rollout_s`: `52.725 -> 52.804` (`+0.15%`)
- `backward_s`: `30.524 -> 30.263` (`-0.86%`)
- `policy_step_total_ms`: `12256.68 -> 12382.32` (`+1.03%`)

Decision:

- Improvement is marginal/mixed; keep this path as optional probe knob, not mainline skyline driver.

### TBPTT stream-merge=2 fail-fast check

Probe:

- `TICL_POLICY_TBPTT_STREAM_MERGE_WINDOWS=2` with `--pg-oom-fail-fast True`
- log: `20260306_seeded_tbpttmerge2_failfast.log`

Observed:

- immediate OOM with explicit fail-fast marker:
  - `[pg-oom-fail-fast] ... raising immediately (fallback disabled).`
- OOM site in transition generator path (`torch.baddbmm`) under this memory envelope.

Decision:

- keep `tbptt_stream_merge_windows=1` on current skyline unless memory is reduced first.

## Continuation pass (2026-03-06): flash-prefix tail-dense launch-reduction probe

### Code changes

- `TransformerEncoderLayer._forward_step_attn_ff_paged`:
  - added optional small-tail dense route under flash-prefix path:
    - `TICL_POLICY_FLASH_PREFIX_TAIL_DENSE_MAX_TOKENS` (default `0`)
  - when `prefix` exists and `tail_take <= threshold`, route to one dense SDPA
    (`prefix+tail` cat) to reduce dual flash-prefix dispatch+merge launches.
- startup observability:
  - `Policy flash-prefix tail dense max tokens: ...`

### Fixed-seed fail-fast A/B

Command base (same seed/workload; only tail-dense threshold differs):

```bash
TICL_PROFILE_ROLLOUT_TIMING=1 TICL_PROFILE_ROLLOUT_BREAKDOWN=1 TICL_POLICY_STEP_PROFILE=1 \
CONDA_NO_PLUGINS=true conda run --no-capture-output -n rlpfn \
  python -u -m ticl.fit_model rlpfn \
  --seed-everything True \
  --epochs 1 --num-steps 1 \
  --validate False --rl-validate-enabled False --progress-bar False \
  --train-profiler-enabled True --train-profiler-wandb False --train-profiler-log-every-batches 1 \
  --train-gpu-observer-enabled True --train-gpu-observer-interval-sec 0.2 \
  --pg-env-replay-steps 1 --pg-tbptt-window 64 \
  --pg-oom-fail-fast True --pg-oom-reduce-tbptt-first False \
  --policy-rollout-chunk-autotune False
```

Logs:

- `tail_dense=0`:
  - `20260306_seeded_taildense_off_failfast.log`
  - `20260306_seeded_taildense_off_failfast_r2.log`
- `tail_dense=16`:
  - `20260306_seeded_taildense16_on_failfast.log`
  - `20260306_seeded_taildense16_on_failfast_r2.log`
- `tail_dense=32`:
  - `20260306_seeded_taildense32_on_failfast.log`

Metrics:

- pair-1 (`off -> on16`):
  - `batch_wall_excl_compile_s`: `81.059 -> 80.953` (`-0.13%`)
  - `policy_step_total_ms`: `12414.21 -> 11930.76` (`-3.89%`)
- pair-2 (`off -> on16`):
  - `batch_wall_excl_compile_s`: `83.222 -> 82.295` (`-1.11%`)
  - `policy_step_total_ms`: `12572.99 -> 12454.02` (`-0.95%`)
- `tail_dense=32` regressed:
  - `batch_wall_excl_compile_s`: `81.059 -> 83.621` (`+3.16%`)

Order-sensitivity check (`on16` first, `off` second) showed large drift
(`85.306 -> 81.085`), indicating runtime variance is high and this probe is not
stable enough for skyline mainline.

Decision:

- keep `TICL_POLICY_FLASH_PREFIX_TAIL_DENSE_MAX_TOKENS` as probe knob with default `0`.
- do **not** mainline this optimization into skyline defaults yet.

## Continuation pass (2026-03-06): TBPTT OOM fallback fail-fast default for rlpfn

- Updated `get_rlpfn_default_config()`:
  - `optimizer.pg_oom_fail_fast = True`
- Goal: skyline/perf runs fail immediately on OOM (no TBPTT/chunk fallback
  retries), so `batch_wall_excl_compile_s` is not polluted by fallback paths.

## Continuation pass (2026-03-06): transition async in-stream commit probe

### Code changes

- `EnvironmentPrior._rollout_family_group_vectorized_with_policy`:
  - added optional async commit path for transition-group stream fusion:
    - `TICL_POLICY_ASYNC_GROUP_COMMIT_IN_STREAM`
  - when enabled, async branches write `reward_next_raw/state_next` directly in
    their CUDA streams, then do one unified `wait_stream` (removes the default-
    stream deferred state-update loop).
- startup observability:
  - `Policy transition async commit in-stream: ...`
- default:
  - keep this probe **disabled by default** (`TICL_POLICY_ASYNC_GROUP_COMMIT_IN_STREAM=0`)
    after A/B showed wall-time regression.

### Fixed-seed fail-fast A/B

Command base (same seed/workload; only async-commit toggle differs):

```bash
TICL_POLICY_OOM_FAIL_FAST=1 TICL_PROFILE_ROLLOUT_TIMING=1 TICL_PROFILE_ROLLOUT_BREAKDOWN=1 TICL_POLICY_STEP_PROFILE=1 \
CONDA_NO_PLUGINS=true conda run --no-capture-output -n rlpfn \
  python -u -m ticl.fit_model rlpfn \
  --seed-everything True \
  --epochs 1 --num-steps 1 \
  --validate False --rl-validate-enabled False --progress-bar False \
  --train-profiler-enabled True --train-profiler-wandb False --train-profiler-log-every-batches 1 \
  --train-gpu-observer-enabled True --train-gpu-observer-interval-sec 0.2
```

Logs:

- `async_commit=off`:
  - `20260306_seeded_asynccommit_off.log`
  - `20260306_seeded_asynccommit_off_r2.log`
- `async_commit=on`:
  - `20260306_seeded_asynccommit_on.log`

Metrics (`off -> on`):

- `batch_wall_excl_compile_s`: `82.311 -> 83.498` (`+1.44%`, regression)
- `rollout_s`: `52.499 -> 53.107` (`+1.16%`)
- `backward_s`: `29.812 -> 30.392` (`+1.95%`)

Transition sub-metrics (expected direction, but not enough to improve total wall):

- `rollout_transition_state_update_share`: `0.058 -> 0.000`
- `rollout_transition_group_sync_share`: `0.067 -> 0.008`
- `rollout_transition_fused_share`: `0.805 -> 0.863`

Reverse check (`off_r2`):

- `batch_wall_excl_compile_s=82.562` (still clearly below `on=83.498`),
  confirming regression is not just one-shot noise.

Decision:

- Keep this path as an explicit probe only.
- Mainline skyline keeps `TICL_POLICY_ASYNC_GROUP_COMMIT_IN_STREAM=0`.

## Continuation pass (2026-03-06): token-layout prepack mainlined (fixed-seed A/B)

### Hotspot diagnosis (layer-step profile)

From fixed-seed baseline with `TICL_TRANSFORMER_LAYER_STEP_PROFILE=1`:

- `policy_step_transformer_share=0.891` (policy forward dominant)
- inside transformer layer:
  - `policy_step_tf_layer_attnff_share=0.712`
  - `policy_step_tf_layer_finalize_share=0.570`
  - `policy_step_tf_layer_proj_share=0.174`
  - `policy_step_tf_layer_cache_share=0.088`
- dispatch mix:
  - `policy_step_paged_dispatch(single/flashp/flashp0/flashm/dense)=768/11520/11520/0/0`

This confirms the rollout hot loop should keep reducing non-kernel host-side
per-step work, instead of parameter-only tuning.

### Code changes

- `ticl/priors/environment_prior.py`
  - changed `TICL_POLICY_TOKEN_LAYOUT_PREPACK` default from `0` to `1`.
  - rationale: precompute token scatter layout once and reuse in rollout loop.
    Semantics are unchanged.
- `ticl/train.py`
  - added startup observability line:
    - `Policy token layout prepack: True/False`

### Fixed-seed hard A/B (`default on` vs forced off)

Command base (same workload/seed, only prepack toggle differs):

```bash
TICL_POLICY_OOM_FAIL_FAST=1 TICL_PROFILE_ROLLOUT_TIMING=1 TICL_PROFILE_ROLLOUT_BREAKDOWN=1 \
TICL_POLICY_STEP_PROFILE=1 TICL_TRANSFORMER_LAYER_STEP_PROFILE=1 \
CONDA_NO_PLUGINS=true conda run --no-capture-output -n rlpfn \
  python -u -m ticl.fit_model rlpfn \
  --seed-everything True \
  --epochs 1 --num-steps 1 \
  --validate False --rl-validate-enabled False --progress-bar False \
  --train-profiler-enabled True --train-profiler-wandb False --train-profiler-log-every-batches 1 \
  --train-gpu-observer-enabled True --train-gpu-observer-interval-sec 0.2 \
  --pg-env-replay-steps 1
```

Logs:

- `prepack=on(default)`: `20260306_seeded_tokenprepack_default_on_ab.log`
- `prepack=off`: `20260306_seeded_tokenprepack_forced_off_ab.log`

Metrics (`off -> on`):

- `rollout_s`: `54.884 -> 53.209` (`-3.05%`)
- `backward_s`: `31.680 -> 30.338` (`-4.24%`)
- `batch_wall_excl_compile_s`: `86.563 -> 83.547` (`-3.48%`)
- `policy_step_total_ms`: `13090.06 -> 12829.31` (`-1.99%`)
- `policy_step_tf_layer_total_ms`: `11190.76 -> 10953.95` (`-2.12%`)

Decision:

- This is a stable positive optimization and is now mainlined by default
  (`TICL_POLICY_TOKEN_LAYOUT_PREPACK=1`, still overrideable).

### Rejected side probes (not mainlined)

- `TICL_POLICY_INPLACE_PAGED_KV=1`:
  - regressed wall and raised memory peak significantly (up to ~44 GiB reserved).
- `TICL_POLICY_PAGED_ATTN_TRAIN_MODE=dense` / `flash_merge`:
  - immediate OOM under fail-fast; excluded from skyline mainline.

## Continuation pass (2026-03-06): negative micro-probe rollback + default guards

Goal for this pass:

- keep fixed-seed KPI on `batch_wall_excl_compile_s`
- test low-level rollout micro-optimizations under strict A/B
- keep only net-positive path on mainline defaults

### Fixed-seed hard A/B: `TICL_POLICY_STATE_POSTPROCESS_INPLACE`

Logs:

- `off`: `20260306_seeded_statepost_inplace_off_ab.log`
- `on`: `20260306_seeded_statepost_inplace_on_ab.log`

Metrics (`off -> on`):

- `batch_wall_excl_compile_s`: `81.574 -> 83.508` (`+2.37%`, worse)
- `rollout_s`: `52.047 -> 53.315` (`+2.44%`, worse)
- `backward_s`: `29.527 -> 30.193` (`+2.26%`, worse)

Decision:

- keep `TICL_POLICY_STATE_POSTPROCESS_INPLACE` as probe knob only
- default set to `0` (off)

### Fixed-seed hard A/B: `TICL_POLICY_REWARD_MASK_BUFFER_REUSE`

Logs:

- `off`: `20260306_seeded_rewardmaskbuf_off_ab.log`
- `on`: `20260306_seeded_rewardmaskbuf_on_ab.log`

Metrics (`off -> on`):

- `batch_wall_excl_compile_s`: `82.932 -> 83.417` (`+0.58%`, worse)
- `rollout_s`: `52.891 -> 53.145` (`+0.48%`, worse)
- `backward_s`: `30.041 -> 30.272` (`+0.77%`, worse)

Decision:

- keep `TICL_POLICY_REWARD_MASK_BUFFER_REUSE` as probe knob only
- default set to `0` (off)

### Fixed-seed hard A/B: `TICL_POLICY_ASYNC_GROUP_COMMIT_IN_STREAM`

Logs:

- `off`: `20260306_seeded_asynccommit_off_ab.log`
- `on`: `20260306_seeded_asynccommit_on_ab.log`

Metrics (`off -> on`):

- `batch_wall_excl_compile_s`: `80.745 -> 85.162` (`+5.47%`, worse)
- `rollout_s`: `51.701 -> 53.989` (`+4.42%`, worse)
- `backward_s`: `29.043 -> 31.172` (`+7.33%`, worse)
- `rollout_transition_group_sync_share`: `0.066 -> 0.008` (sync cost drops)
- `rollout_transition_fused_launch_share`: `0.809 -> 0.864` (launch share rises)

Interpretation:

- in-stream commit reduces explicit sync/update share, but raises launch-side pressure and hurts end-to-end wall time in this workload.
- keep default `TICL_POLICY_ASYNC_GROUP_COMMIT_IN_STREAM=0`.

### Observability updates retained

- startup prints now include:
  - `Policy state postprocess inplace`
  - `Policy reward-mask buffer reuse`

This keeps probe reproducibility explicit in logs while mainline defaults remain on the non-regressing path.

## Continuation pass (2026-03-06): flash-prefix dense-threshold hard retune (non-compile)

Goal:

- target `policy_step_transformer_share` dominant path without compile dependency
- reduce step-level kernel launch/merge overhead in paged `flash_prefix` route
- keep fixed-seed KPI on `batch_wall_excl_compile_s`

### Why this pass

- baseline fixed-seed reprofile still showed transformer-dominant rollout:
  - `policy_step_transformer_share=0.903`
  - `rollout_instage_backward_share=0.573`
  - `batch_wall_excl_compile_s=84.376`
- previous probe `TICL_POLICY_INPLACE_PAGED_KV=1` was rejected:
  - wall regressed and peak reserved rose to ~`43.88 GiB`.

### A/B probe: `TICL_POLICY_PAGED_ATTN_FLASHPREFIX_DENSE_MAX_TOKENS`

Same fixed command/seed; only this threshold changed.

Logs:

- `dense64(control)`: `20260306_seeded_flashprefix_dense64_control.log`
- `dense256`: `20260306_seeded_flashprefix_dense256_ab.log`
- `dense256(rep2)`: `20260306_seeded_flashprefix_dense256_rep2.log`
- `dense512`: `20260306_seeded_flashprefix_dense512_probe.log`
- `default-after-switch-check`: `20260306_seeded_default_after_dense256_mainline.log`

Metrics (`64 -> 256`):

- `rollout_s`: `53.626 -> 51.778` (`-3.45%`)
- `backward_s`: `30.751 -> 29.477` (`-4.14%`)
- `batch_wall_excl_compile_s`: `84.376 -> 81.256` (`-3.70%`)
- `policy_step_total_ms`: `12672.27 -> 12356.85` (`-2.49%`)
- `policy_step_transformer_share`: `0.903 -> 0.883`
- peak alloc/reserved: `30.93/31.68 GiB -> 30.93/31.68 GiB` (unchanged)

Replicate stability check (`64 -> 256(rep2)`):

- `batch_wall_excl_compile_s`: `84.376 -> 84.117` (`-0.31%`, near-noise)
- `policy_step_transformer_share`: `0.903 -> 0.903` (flat)

Default-after-switch check (`expected 256 behavior`):

- startup print confirms `Policy flash-prefix dense max tokens: 256`
- observed `batch_wall_excl_compile_s=85.953` (worse than control), indicating high run-to-run variance in this probe.

Metrics (`64 -> 512`):

- `batch_wall_excl_compile_s`: `84.376 -> 82.322` (`-2.43%`)
- worse than `256` on the same workload/seed.

Decision:

- probe kept as tunable env only; **not** promoted to mainline default due unstable A/B.
- mainline default remains `TICL_POLICY_PAGED_ATTN_FLASHPREFIX_DENSE_MAX_TOKENS=64`.

### Hard-path probes rejected in same pass

Logs:

- `flash_merge`: `20260306_seeded_flashmerge_probe.log`
- `tbptt_merge_windows=2`: `20260306_seeded_tbptt_merge2_probe.log`

Result:

- both OOM under fail-fast (no fallback), so excluded from skyline mainline.

## Continuation pass (2026-03-06): compile recompile pollution isolation (hard reject + fix)

### Symptom (confirmed pseudo-optimization)

Compile probe with mutable paged KV + split fastpath:

- log: `20260306_184343_seeded_compile_probe.log`
- key line:
  - `batch_wall_excl_compile_s=304.586`
  - `compile_warmup_s=36.024`
  - `compile_counter_recompiles=1`
  - runtime warning: `forward_policy_step_split` hit recompile limit due `kv_cache[0]['k_pages'][0]` token-length mismatch.

Interpretation:

- this compile route is polluted by dynamic page-shape recompiles; KPI is invalid for mainline decision.

### Code fix

- `ticl/train.py` (`_build_policy_step_fn`):
  - added split-step compile mode gate `TICL_POLICY_SPLIT_STEP_COMPILE={auto,on,off}`.
  - `auto` now disables split compile on mutable paged KV path (default case here) to avoid recompile storms.
  - when split fastpath is active and split compile auto-disabled, main forward-step compile is also skipped.
  - exposed runtime flag on callable: `policy_step_fn._compile_active`.
- `ticl/train.py` (PG epoch loop):
  - compile warmup/compile-observe are now enabled only when `_compile_active=True`.
  - this prevents fake compile warmup accounting when compile was requested but not actually active.

### Fixed-seed A/B after fix

Logs:

- compile-on (after fix): `20260306_184343_seeded_compile_probe_afterfix.log`
- eager control (same code): `20260306_184343_seeded_eager_postfix.log`

Metrics (`eager -> compile-on-after-fix`):

- `batch_wall_excl_compile_s`: `84.475 -> 83.997` (`-0.57%`, near-noise)
- no `compile_warmup_s` / compile-counter fields emitted in batch line (compile path correctly inactive).
- startup confirms:
  - `[pg-compile-note] split forward step compile auto-disabled for mutable paged KV cache.`
  - `[pg-compile-note] main forward step compile skipped (split fastpath active).`

Decision:

- compile pseudo-optimization path is isolated and no longer pollutes KPI.
- keep eager path as mainline for this workload; compile remains opt-in for future static-shape attempts.

## Continuation pass (2026-03-06): transition async-commit auto mainline (non-compile hard optimization)

Goal:

- continue attacking rollout hot serial section (`transition_group_share`) with kernel-path changes (not parameter tuning)
- evaluate with fixed seed and KPI only on `batch_wall_excl_compile_s`
- keep `--pg-env-replay-steps 1` and `--pg-oom-fail-fast True` for fast failure and clean A/B

### Code changes

- `ticl/priors/environment_prior.py`
  - `TICL_POLICY_ASYNC_GROUP_COMMIT_IN_STREAM` default changed from `0` to `auto`.
  - `auto` now enables in-stream group commit when transition stream-fusion is active.
  - streamlined async path to avoid per-step deferred-op tuple bookkeeping when commit is already in-stream.
  - observability added: `rollout_transition_async_commit_in_stream`.

### Fixed-seed hard A/B (after code change)

Same command base; only async-commit policy differs:

- `off`: `TICL_POLICY_ASYNC_GROUP_COMMIT_IN_STREAM=0`
- `auto(default)`: env not set

Logs:

- `off`: `20260306_193500_seeded_asynccommit_off_aftercode.log`
- `auto`: `20260306_193500_seeded_asynccommit_auto_aftercode.log`

Metrics (`off -> auto`):

- `rollout_s`: `53.713 -> 52.107` (`-2.99%`)
- `backward_s`: `30.552 -> 29.518` (`-3.38%`)
- `batch_wall_excl_compile_s`: `84.265 -> 81.626` (`-3.13%`)
- `policy_step_total_ms`: `12729.10 -> 12504.03` (`-1.77%`)
- `rollout_transition_group_sync_share`: `0.065 -> 0.008`
- `rollout_transition_state_update_share`: `0.057 -> 0.000`

Interpretation:

- this is a real non-compile speedup from reducing transition-group sync/update serial sections.
- improvement is monotonic with no extra VRAM pressure (`peak reserved` stayed ~`31.68 GiB`).

Default-path confirmation (no async env override):

- log: `20260306_193500_seeded_default_mainline_after_asyncauto.log`
- `batch_wall_excl_compile_s=80.923` (better than `off` by `-3.97%` vs `84.265`)

### Related probes in same pass

- `TICL_POLICY_SPLIT_ENCODE_FUSION`:
  - `batch_wall_excl_compile_s: 84.038 -> 83.032` (`-1.20%`, positive but secondary).
- `TICL_POLICY_INPLACE_PAGED_KV=1`:
  - rejected (`batch_wall_excl_compile_s: 83.032 -> 86.849`, peak reserved `31.68 -> 43.88 GiB`).

## Continuation pass (2026-03-06): TBPTT prefix compaction + envgen checkpoint (memory win, throughput reject)

Goal:

- attack the first-window memory wall before TBPTT detach, which was blocking
  clean fixed-seed benchmarking with fail-fast enabled
- reduce prefix-history duplication after TBPTT detach
- preserve observability so low-VRAM and throughput modes can be separated

### Code changes

- `ticl/models/layer.py`
  - paged KV cache now carries `prefix_base_len` so detached dense prefix and
    mutable tail pages can coexist without semantic drift.
  - query path now materializes `k_prefix/v_prefix + tail pages` correctly.
- `ticl/priors/environment_prior.py`
  - TBPTT detach now optionally compacts full detached paged-prefix pages into
    dense `k_prefix/v_prefix` (`TICL_POLICY_PREFIX_COMPACT_ON_TBPTT_DETACH`, default on).
  - added opt-in env-transition activation checkpoint:
    - `TICL_POLICY_ENVGEN_CHECKPOINT`
    - `TICL_POLICY_ENVGEN_CHECKPOINT_REENTRANT`
  - envgen checkpoint externalizes sampled noise from the recomputed core and
    clones the reused rollout input buffer to keep backward recompute safe.
  - added rollout observability:
    - `rollout_transition_checkpoint_enabled`
    - `rollout_transition_checkpoint_calls`
- `ticl/train.py`
  - startup prints for prefix compaction + envgen checkpoint mode.
- tests:
  - paged-TBPTT detach compaction semantic regression test
  - envgen checkpoint semantic equivalence test (loss / rollout / grad)

### Fixed-seed observations

Early fail-fast probe on the raw default line:

- previous code path: immediate first-window OOM before any authoritative
  `batch_wall_excl_compile_s`
- with envgen checkpoint on:
  - log: `20260306_seeded_envgen_checkpoint_on_merge1.log`
  - first window becomes measurable instead of OOM
  - `batch_wall_excl_compile_s=106.026`
  - peak reserved `23.94 GiB`

This confirms the dominant first-window memory term was in env transition
activations, not post-detach prefix history.

### Fixed-seed A/B on stable training line (`aev5 + lipschitz`, same seed / same command, only envgen checkpoint differs)

Logs:

- `off`: `20260306_seeded_aev5_lipschitz_envgen_checkpoint_off.log`
- `on`: `20260306_seeded_aev5_lipschitz_envgen_checkpoint_on.log`

Metrics (`off -> on`):

- `batch_wall_excl_compile_s`: `85.865 -> 108.696` (`+26.6%`, worse)
- `rollout_s`: `55.287 -> 68.311` (`+23.6%`, worse)
- `backward_s`: `30.579 -> 40.385` (`+32.1%`, worse)
- peak reserved: `46.20 -> 23.94 GiB` (large VRAM win)
- rollout `gpu_util_avg`: `29.31 -> 30.50` (slightly higher, but not useful)
- `rollout_transition_checkpoint_calls=2048`

Interpretation:

- envgen checkpoint is a real low-VRAM mode, not a throughput optimization.
- recompute cost dominates any benefit from lower reserved memory at fixed
  workload.
- do **not** mainline envgen checkpoint into the throughput skyline.

### Follow-up probe: spend saved VRAM on larger TBPTT window

Log:

- `20260306_seeded_aev5_lipschitz_envgen_checkpoint_on_tbptt128.log`

Metrics (`tbptt=64 checkpoint-on -> tbptt=128 checkpoint-on`):

- `tbptt_stream_backward_calls`: `16 -> 8`
- `batch_wall_excl_compile_s`: `108.696 -> 108.284` (`-0.38%`, near-noise)
- peak reserved: `23.94 -> 38.40 GiB`

Interpretation:

- larger TBPTT window consumes the recovered VRAM, but does not recover the
  recompute tax into a positive throughput gain.

Decision:

- keep prefix compaction on by default (semantic-safe memory reduction for
  detached paged KV history).
- under the old `batch=64` / raw-batch-wall regime, envgen checkpoint stayed
  off by default because throughput regressed.
- that conclusion is superseded below for the later `batch=128` /
  per-batch-item skyline regime.

## Continuation pass (2026-03-06): batch=128 mainline + per-batch-item skyline KPI

Goal:

- keep `TBPTT=64` fixed
- raise physical `batch_size` to `128`
- stop judging skyline on raw batch wall once batch size changes; use
  `batch_wall_excl_compile_s / batch_size` directly from training logs

### Code changes

- `ticl/model_configs.py`
  - `rlpfn` default physical `batch_size` changed from `64` to `128`.
- `ticl/train.py`
  - `[pg-phase]`, GPU-stage JSON, and wandb payload now emit:
    - `batch_size`
    - `batch_wall_excl_compile_per_batch_item_s`
    - `batch_wall_incl_compile_per_batch_item_s`
- `ticl/fit_model.py`
  - mainline `python -m ticl.fit_model rlpfn` now defaults
    `TICL_POLICY_ENVGEN_CHECKPOINT=1`.
  - reason: with `batch=128` and fail-fast enabled, the non-checkpoint path
    OOMs before the first authoritative batch KPI.

### Fixed-seed observations (`TBPTT=64`, `replay_step=1`, fail-fast, merge=1)

Logs:

- `batch=64`, envgen checkpoint off:
  - `20260306_seeded_batch64_kpi_per_item.log`
- `batch=128`, envgen checkpoint off:
  - `20260306_seeded_batch128_kpi_per_item.log`
- `batch=128`, envgen checkpoint on:
  - `20260306_seeded_batch128_kpi_per_item_envgenckpt.log`

Metrics:

- `batch=64`, envgen checkpoint off:
  - `batch_wall_excl_compile_s=86.463`
  - `batch_wall_excl_compile_per_batch_item_s=1.350990`
  - peak reserved `46.20 GiB`
  - GPU observer `gpu_util_avg=29.61`
- `batch=128`, envgen checkpoint off:
  - immediate OOM under fail-fast at env transition `_batch_affine`
  - no authoritative batch KPI; excluded from skyline
- `batch=128`, envgen checkpoint on:
  - `batch_wall_excl_compile_s=128.773`
  - `batch_wall_excl_compile_per_batch_item_s=1.006041`
  - peak reserved `46.17 GiB`
  - GPU observer `gpu_util_avg=37.08`

Outcome:

- raw batch wall worsens as expected when doubling physical batch:
  - `86.463 -> 128.773` (`+48.9%`)
- normalized skyline KPI improves materially:
  - `1.350990 -> 1.006041` (`-25.5%`, better)
- GPU utilization also moves in the right direction:
  - `29.61 -> 37.08` (`+7.47` absolute points, `+25.2%` relative)

Interpretation:

- the blocking memory wall for `batch=128` is still env-transition activation
  residency, not policy-step compute.
- envgen checkpoint is throughput-negative at `batch=64`, but becomes
  throughput-positive under the new skyline regime because it unlocks full
  `chunk=128` execution and the per-item wall drops substantially.
- for the `batch=128` regime, skyline should now be maintained on
  `batch_wall_excl_compile_per_batch_item_s`, not raw
  `batch_wall_excl_compile_s`.

## Continuation pass (2026-03-07): fused-transition stable-input slots A/B

Goal:

- keep the new `batch=128 / TBPTT=64 / replay_step=1 / fail-fast` mainline
- attack the env-transition checkpoint path directly
- remove redundant transition-input copies from the fused dual-generator path
  before escalating to heavier kernel rewrites

### Code changes

- `ticl/priors/environment_prior.py`
  - fused transition builders now mark their temporary dual-packed input as
    checkpoint-stable, so the inner envgen checkpoint no longer clones
    `x_dual` redundantly.
  - added optional `TICL_POLICY_FUSED_TRANSITION_STABLE_INPUT_SLOTS` path:
    pre-allocates per-window dual-input slots for fused transition so rollout
    can skip `torch.cat([x, x])` allocation churn.
  - added rollout profile fields:
    - `transition_stable_dual_input_enabled`
    - `transition_stable_dual_input_call_count`
- `ticl/train.py`
  - startup logs, GPU-stage JSON, and wandb payload now surface the new
    stable-dual-input observability.
- `ticl/tests/priors/test_environment_prior.py`
  - added direct semantic regression that compares fused transition on
    dual-packed input vs the legacy `cat([x, x])` path under checkpoint.

### Important implementation note

- the first stable-slot implementation used one stacked 3D tensor and sliced
  views per timestep.
- that is invalid under checkpoint: writing a different slot still increments
  the shared base tensor version and backward aborts with an in-place version
  mismatch.
- the surviving implementation uses a list of independent slot tensors instead.

### Fixed-seed A/B (`batch_wall_excl_compile_s`)

Logs:

- stable slots off:
  - `20260306_seeded_batch128_stabledualslots_off.log`
- stable slots on:
  - `20260307_seeded_batch128_stabledualslots_on_r2.log`

Metrics:

- stable slots off:
  - `batch_wall_excl_compile_s=142.066`
  - `batch_wall_excl_compile_per_batch_item_s=1.109893`
  - `rollout_transition_wall_ms=25554.24`
  - `gpu_util_avg=36.43`
- stable slots on:
  - `batch_wall_excl_compile_s=146.016`
  - `batch_wall_excl_compile_per_batch_item_s=1.140750`
  - `rollout_transition_wall_ms=26609.02`
  - `gpu_util_avg=35.55`

Outcome:

- raw batch wall regressed:
  - `142.066 -> 146.016` (`+2.78%`)
- per-item wall also regressed:
  - `1.109893 -> 1.140750` (`+2.78%`)
- transition wall itself regressed:
  - `25554.24 ms -> 26609.02 ms` (`+4.13%`)

Decision:

- keep the redundant inner checkpoint-clone removal in the fused transition
  path.
- keep stable-input slots code and observability for future experimentation,
  but do **not** enable it by default.
- mainline default is therefore:
  - `TICL_POLICY_FUSED_TRANSITION_STABLE_INPUT_SLOTS=0`

## Continuation pass (2026-03-07): transition inner grouping A/B

Goal:

- keep the `batch=128 / TBPTT=64 / replay_step=1 / fail-fast` mainline
- test whether the family-group transition kernel is dominated by padding waste
- split only the inner transition batch, while keeping the outer policy-step
  batch wide

### Code changes

- `ticl/priors/environment_prior.py`
  - added `TICL_POLICY_TRANSITION_INNER_GROUPING` with modes:
    - `family` (current mainline)
    - `structure` (exact structure buckets)
    - `pow2`
    - `pow2_no_depth`
  - added transition-bucket work proxy and rollout observability:
    - `transition_family_group_count`
    - `transition_bucket_max_batch`
    - `transition_bucket_mean_batch`
    - `transition_work_fill_ratio`
  - exact/fused family rollout now allocates CUDA streams after final inner
    bucket construction, so stream fusion keys off real transition-group count.
- `ticl/train.py`
  - startup log, phase log, GPU-stage JSON, and wandb now surface the new
    transition-bucket observability.
- `ticl/tests/priors/test_environment_prior.py`
  - added regressions for:
    - coarse family mode still using `[3, 1]` inner family buckets
    - exact structure mode splitting to singleton buckets in the toy test
    - family vs structure semantic equivalence in deterministic setup

### Fixed-seed batch benchmark

Logs:

- family baseline:
  - `20260307_seeded_batch128_transition_inner_family.log`
- exact structure probe:
  - `20260307_seeded_batch128_transition_inner_structure.log`
- coarse pow2-no-depth probe:
  - `20260307_seeded_batch128_transition_inner_pow2nodepth.log`
- offline grouping diagnostic:
  - `20260307_transition_inner_grouping_diagnostic.txt`

Metrics:

- family baseline:
  - `batch_wall_excl_compile_s=133.713`
  - `batch_wall_excl_compile_per_batch_item_s=1.044631`
  - `rollout_transition_wall_ms=15932.39`
- exact structure:
  - did not finish before manual termination
  - lower bound at termination: `>252s` wall (`>1.88x` slower than family)
- pow2_no_depth:
  - did not finish before manual termination
  - lower bound at termination: `>202s` wall (`>2.38x` slower than the
    84.54s family epoch wall)

Offline sampled-hypers diagnostic (`batch=128`, seed `42`):

- family:
  - `group_count=2`
  - `max_batch=66`
  - `fill_ratio=0.0945`
- exact structure:
  - `group_count=128`
  - `max_batch=1`
  - `singletons=128`
  - `fill_ratio=0.8986`
- pow2_no_depth:
  - `group_count=53`
  - `max_batch=16`
  - `singletons=33`
  - `fill_ratio=0.4724`
- pow2:
  - `group_count=74`
  - `max_batch=16`
  - `singletons=55`
  - `fill_ratio=0.7461`

Outcome:

- the diagnosis is real: family grouping wastes most transition compute on
  padding (`fill_ratio ~= 9.45%`).
- but naive inner bucketing is still a net loss, because launch/sync overhead
  dominates once the family group fractures into dozens of tiny buckets.
- the profitable next step is therefore **not** finer grouping; it is to
  rewrite the fused transition kernel so it stops padding the state/reward dual
  path internally.

Decision:

- keep the observability and bucket-mode code for future experiments.
- keep mainline default on:
  - `TICL_POLICY_TRANSITION_INNER_GROUPING=family`
- do **not** commit/push, because this pass did not produce a new positive
  skyline.

### Extra hybrid probe (2026-03-07)

- added `TICL_POLICY_TRANSITION_INNER_MIN_BUCKET` so only sufficiently large
  inner buckets are kept and the rest merge back into the family group.
- tested `pow2_no_depth + min_bucket=2`.
- result:
  - `20260307_seeded_batch128_transition_inner_pow2nodepth_min2.log`
  - timed out at `240s` without reaching a phase line
  - offline proxy for the same seed/config:
    - `group_count=22`
    - `max_batch=19`
    - `fill_ratio=0.3093`
- conclusion:
  - even after merging the tiny buckets back, two-digit inner group counts are
    still too expensive for the current implementation.
  - the transition-grouping branch is exhausted enough to stop here.

## Continuation pass (2026-03-07): Triton ragged batched affine prototype for envgen

Goal:

- stay on the same `batch=128 / TBPTT=64 / replay_step=1 / fail-fast` mainline
- replace the envgen hetero-batch padded `bmm` path with a ragged batched
  affine prototype
- check the result with fixed-seed A/B on `batch_wall_excl_compile_s`
- explicitly rule out Triton first-compile pollution before judging the KPI

### Code changes

- `ticl/priors/environment_prior.py`
  - added CUDA-only Triton ragged batched affine kernels plus custom autograd
    for `grad_x`
  - hetero SCM builder now precomputes per-layer active input indices and can
    dispatch each affine through the ragged path
  - hetero GP builder now does the same for both the input-to-feature and
    feature-to-output affine stages
  - the path is opt-in behind:
    - `TICL_POLICY_ENVGEN_RAGGED_AFFINE=1`
- `ticl/train.py`
  - startup log now prints `Policy envgen ragged affine`
- `ticl/tests/priors/test_environment_prior.py`
  - added CUDA semantic regression comparing dense vs ragged hetero envgen
    forward/backward for both `scm` and `gp`

### Validation

- `conda run -n rlpfn python -m py_compile ...` passed
- targeted pytest passed:
  - `ragged_affine_matches_dense_hetero_batch_semantics`
  - `fused_transition_dual_packed_matches_cat_semantics`
  - `transition_min_bucket_merges_small_buckets_back_to_family`

### Fixed-seed A/B (`batch_wall_excl_compile_s`)

Command notes:

- explicit control of the previous memory confounder:
  - `TICL_POLICY_TBPTT_STREAM_MERGE_AUTO=0`
  - `TICL_POLICY_TBPTT_STREAM_MERGE_WINDOWS=1`
- other runtime knobs held constant between runs:
  - `TICL_POLICY_ENVGEN_CHECKPOINT=1`
  - `TICL_POLICY_ENVGEN_CHECKPOINT_REENTRANT=0`
  - `TICL_POLICY_TRANSITION_INNER_GROUPING=family`
  - `TICL_POLICY_TRANSITION_INNER_MIN_BUCKET=0`
  - `--seed-everything True -E 1 -n 1 -b 128`
  - `--pg-tbptt-window 64 --pg-env-replay-steps 1`

Logs:

- control (`ragged=off`):
  - `20260307_seeded_batch128_ragged_affine_off.log`
- prototype first run (`ragged=on`):
  - `20260307_seeded_batch128_ragged_affine_on.log`
- prototype warm-cache rerun (`ragged=on`):
  - `20260307_seeded_batch128_ragged_affine_on_warm.log`

Metrics:

- control (`ragged=off`):
  - `batch_wall_excl_compile_s=130.739`
  - `batch_wall_excl_compile_per_batch_item_s=1.021402`
  - `gpu_util_avg=38.31`
  - peak alloc/reserved: `36.18 / 46.17 GiB`
- prototype first run (`ragged=on`):
  - `batch_wall_excl_compile_s=136.732`
  - `batch_wall_excl_compile_per_batch_item_s=1.068215`
  - `gpu_util_avg=28.55`
  - peak alloc/reserved: `35.30 / 46.15 GiB`
- prototype warm-cache rerun (`ragged=on`):
  - `batch_wall_excl_compile_s=137.105`
  - `batch_wall_excl_compile_per_batch_item_s=1.071130`
  - `gpu_util_avg=29.12`
  - peak alloc/reserved: `35.30 / 46.15 GiB`

Outcome:

- first run vs control:
  - `130.739 -> 136.732` (`+4.59%`, worse)
- warm-cache rerun vs control:
  - `130.739 -> 137.105` (`+4.87%`, worse)
- per-item wall also regressed:
  - `1.021402 -> 1.071130` (`+4.87%`, worse)
- GPU utilization regressed materially:
  - `38.31 -> 29.12` (`-9.19` absolute points)
- peak allocated memory dropped slightly:
  - `36.18 -> 35.30 GiB`
- reserved memory stayed flat:
  - `46.17 -> 46.15 GiB`

Interpretation:

- this prototype did not produce a positive skyline.
- the warm-cache rerun stayed slightly worse than the first run, so the
  regression is **not** explained by Triton first-compile pollution.
- the prototype saves a small amount of live allocation, but loses too much on
  kernel efficiency / occupancy to recover it as wall-time gain.

Decision:

- keep the ragged affine prototype code and semantic test for future kernel
  work, but keep it disabled by default:
  - `TICL_POLICY_ENVGEN_RAGGED_AFFINE=0`
- do **not** commit/push, because this pass did not produce a new skyline.

## Continuation pass (2026-03-07): family-specialized paired transition generator

Goal:

- stay on the transition mainline
- stop using `cat([x, x])` / doubled batch inside fused transition
- replace the legacy dual-batch transition builder with a family-specialized
  paired path that keeps state/reward branches separate while preserving
  legacy fixed-seed semantics

### Code changes

- `ticl/priors/environment_prior.py`
  - added `TICL_POLICY_FUSED_TRANSITION_PAIRED`
  - added SCM paired transition branch builder with joint per-layer sampling so
    state/reward weights are sampled in the exact same order as the legacy
    dual-batch builder
  - added GP paired transition branch builder with the same sampling-order
    preservation
  - added paired transition wrapper that:
    - preserves legacy dual noise-generator advancement
    - supports both regular input and `x_is_dual_packed=True`
    - uses a shared checkpoint snapshot when the rollout input buffer is reused
  - stable dual-input slots now auto-disable when the transition generator does
    not prefer dual-packed inputs
- `ticl/train.py`
  - startup log now prints `Policy fused transition paired-specialized`
- `ticl/tests/priors/test_environment_prior.py`
  - added strict-seed regression comparing legacy dual vs paired-specialized
    transition for both `scm` and `gp`, including dual-packed input

### Validation

- `conda run -n rlpfn python -m py_compile ...` passed
- targeted pytest passed:
  - `paired_transition_matches_legacy_dual_semantics`
  - `fused_transition_dual_packed_matches_cat_semantics`
  - `ragged_affine_matches_dense_hetero_batch_semantics`

### Fixed-seed A/B (`batch_wall_excl_compile_s`)

Command notes:

- held constant:
  - `TICL_POLICY_ENVGEN_CHECKPOINT=1`
  - `TICL_POLICY_ENVGEN_CHECKPOINT_REENTRANT=0`
  - `TICL_POLICY_ENVGEN_RAGGED_AFFINE=0`
  - `TICL_POLICY_FUSED_TRANSITION_STABLE_INPUT_SLOTS=0`
  - `TICL_POLICY_TRANSITION_INNER_GROUPING=family`
  - `TICL_POLICY_TRANSITION_INNER_MIN_BUCKET=0`
  - `TICL_POLICY_TBPTT_STREAM_MERGE_AUTO=0`
  - `TICL_POLICY_TBPTT_STREAM_MERGE_WINDOWS=1`
  - `--seed-everything True -E 1 -n 1 -b 128`
  - `--pg-tbptt-window 64 --pg-env-replay-steps 1`
- only changed:
  - `TICL_POLICY_FUSED_TRANSITION_PAIRED=0/1`

Logs:

- legacy dual baseline:
  - `20260307_seeded_batch128_paired_transition_off.log`
- paired-specialized:
  - `20260307_seeded_batch128_paired_transition_on.log`

Metrics:

- legacy dual baseline:
  - `batch_wall_excl_compile_s=133.025`
  - `batch_wall_excl_compile_per_batch_item_s=1.039260`
  - `gpu_util_avg=36.96`
  - peak alloc/reserved: `36.18 / 46.17 GiB`
- paired-specialized:
  - `batch_wall_excl_compile_s=189.702`
  - `batch_wall_excl_compile_per_batch_item_s=1.482049`
  - `gpu_util_avg=29.27`
  - peak alloc/reserved: `34.99 / 46.09 GiB`

Outcome:

- raw batch wall regressed severely:
  - `133.025 -> 189.702` (`+42.61%`, worse)
- per-item wall also regressed:
  - `1.039260 -> 1.482049` (`+42.61%`, worse)
- GPU utilization dropped:
  - `36.96 -> 29.27` (`-7.69` absolute points)
- peak allocated memory improved modestly:
  - `36.18 -> 34.99 GiB`
- reserved memory remained effectively flat:
  - `46.17 -> 46.09 GiB`

Interpretation:

- removing doubled-batch padding alone is not enough.
- the paired-specialized path cut some activation footprint, but replacing one
  fused transition call with two checkpointed branch executions destroyed
  occupancy and increased rollout/backward time sharply.
- this is strong evidence that the current dominant bottleneck is not just
  padded math volume; it is also the launch/recompute structure of the fused
  transition path.

Offline sampled-hypers diagnostic (`batch=128`, seed `42`):

- `gp` family (`66` samples):
  - padded work shares:
    - input/RFF stage: `72.87%`
    - final/output stage: `27.13%`
  - fill ratios:
    - input/RFF stage: `32.95%`
    - final/output stage: `14.44%`
  - reward branch contribution inside actual final-stage work is tiny:
    - `0.53%`
- `scm` family (`62` samples):
  - padded work shares:
    - input stage: `11.66%`
    - hidden stack: `83.92%`
    - final stage: `4.42%`
  - fill ratios:
    - input stage: `23.23%`
    - hidden stack: `1.77%`
    - final stage: `17.26%`
  - reward branch contribution inside actual final-stage work is tiny:
    - `0.15%`

Implication:

- for `scm`, the next profitable kernel target is the hidden stack, not the
  final reward head.
- for `gp`, the next target is the input/RFF projection path first, then the
  final projection.

Decision:

- keep the paired-specialized code and tests for further kernel work, but keep
  it disabled by default:
  - `TICL_POLICY_FUSED_TRANSITION_PAIRED=0`
- do **not** commit/push, because this pass did not produce a new skyline.

## 2026-03-07 07:49: SCM hidden-stack branch-fused prototype

Goal:

- stay on the transition mainline and replace the dominant `scm` hidden-stack
  padded path with a single-graph branch-fused kernel, instead of expanding the
  generic ragged path or revisiting grouping.

Code:

- `ticl/priors/environment_prior.py`
  - added `TICL_POLICY_FUSED_TRANSITION_SCM_HIDDEN_FUSED`
  - added `_build_scm_hetero_hidden_fused_transition_fn(...)`
  - wired `scm` hetero transition batching to prefer the hidden-fused path when
    enabled, ahead of the legacy dual path
- `ticl/train.py`
  - startup log now prints `Policy fused transition SCM hidden-fused`
- `ticl/tests/priors/test_environment_prior.py`
  - added strict-seed regression:
    - `scm_hidden_fused_transition_matches_legacy_dual_semantics`

Validation:

- `conda run -n rlpfn python -m py_compile ...` passed
- targeted pytest passed:
  - `scm_hidden_fused_transition_matches_legacy_dual_semantics`
  - `paired_transition_matches_legacy_dual_semantics`
  - `fused_transition_dual_packed_matches_cat_semantics`

Fixed-seed A/B (`batch_wall_excl_compile_s`):

Command notes:

- held constant:
  - `TICL_POLICY_ENVGEN_CHECKPOINT=1`
  - `TICL_POLICY_ENVGEN_CHECKPOINT_REENTRANT=0`
  - `TICL_POLICY_ENVGEN_RAGGED_AFFINE=0`
  - `TICL_POLICY_FUSED_TRANSITION_STABLE_INPUT_SLOTS=0`
  - `TICL_POLICY_FUSED_TRANSITION_PAIRED=0`
  - `TICL_POLICY_TRANSITION_INNER_GROUPING=family`
  - `TICL_POLICY_TRANSITION_INNER_MIN_BUCKET=0`
  - `TICL_POLICY_TBPTT_STREAM_MERGE_AUTO=0`
  - `TICL_POLICY_TBPTT_STREAM_MERGE_WINDOWS=1`
  - `--seed-everything True -E 1 -n 1 -b 128`
  - `--pg-tbptt-window 64 --pg-env-replay-steps 1`
- only changed:
  - `TICL_POLICY_FUSED_TRANSITION_SCM_HIDDEN_FUSED=0/1`

Logs:

- hidden-fused disabled:
  - `20260307_seeded_batch128_scmhiddenfused_off.log`
- hidden-fused enabled:
  - `20260307_seeded_batch128_scmhiddenfused_on.log`

Metrics:

- hidden-fused disabled:
  - `batch_wall_excl_compile_s=136.994`
  - `batch_wall_excl_compile_per_batch_item_s=1.070267`
  - `gpu_util_avg=36.86`
  - `rollout_s=86.245`
  - `backward_s=50.749`
- hidden-fused enabled:
  - `batch_wall_excl_compile_s=155.141`
  - `batch_wall_excl_compile_per_batch_item_s=1.212040`
  - `gpu_util_avg=45.84`
  - `rollout_s=97.773`
  - `backward_s=57.368`

Outcome:

- raw batch wall regressed:
  - `136.994 -> 155.141` (`+13.25%`, worse)
- per-item wall also regressed:
  - `1.070267 -> 1.212040` (`+13.25%`, worse)
- GPU utilization increased:
  - `36.86 -> 45.84` (`+8.98` absolute points)
- both rollout and backward got slower:
  - rollout: `86.245 -> 97.773`
  - backward: `50.749 -> 57.368`

Interpretation:

- this is still not a skyline, so do **not** enable it by default.
- however, it is materially better than the earlier paired-specialized split
  branch path:
  - paired-specialized: `189.702`
  - hidden-fused: `155.141`
- this confirms the mainline diagnosis:
  - splitting state/reward into separate checkpointed branches is a major
    throughput mistake
  - moving more work back into one fused graph recovers a large fraction of the
    loss
- the remaining regression means the hidden-stack bottleneck is not solved by a
  simple block-diagonal fusion of the existing dense path:
  - occupancy went up, but padded math / memory traffic still dominates enough
    to lose on wall time
  - first-layer split plus dense block-diagonal hidden layers is still too much
    redundant work

Decision:

- keep the SCM hidden-fused code and tests, but keep it disabled by default:
  - `TICL_POLICY_FUSED_TRANSITION_SCM_HIDDEN_FUSED=0`
- do **not** commit/push, because this pass did not produce a new skyline.

Next bottlenecks:

- highest priority:
  - replace the `scm` hidden stack with a true branch-fused kernel that avoids
    dense block-diagonal padded matmul
- second priority:
  - target the `gp` input/RFF projection path, which is still the dominant `gp`
    padded-work share
- de-prioritized:
  - outer/inner grouping tweaks
  - generic ragged affine expansion
  - split-branch paired transition paths

## 2026-03-07 08:08: SCM hidden-stack packed no-padding fused kernel

Goal:

- continue the same `scm` hidden-stack mainline by replacing the dense
  block-diagonal hidden stack with a packed `2B` branch batch and a specialized
  prefix-ragged affine kernel.

Code:

- `ticl/priors/environment_prior.py`
  - added Triton prefix-tiled affine kernels and autograd wrapper for
    prefix-only hidden layers
  - added `_build_active_tile_map(...)`
  - added `_batch_affine_prefix_tiled(...)`
  - rewrote `_build_scm_hetero_hidden_fused_transition_fn(...)` so that:
    - the first state/reward projection stays separate to preserve strict-seed
      sampling order
    - hidden/final/post layers are packed as a single `2B` branch batch
    - each layer uses its own `(max_in, max_out)` cap instead of a global
      `state_hidden_cap + reward_hidden_cap` work tensor
    - hidden stack execution uses the new prefix-ragged path instead of dense
      block-diagonal `_batch_affine`

Validation:

- `conda run -n rlpfn python -m py_compile ...` passed
- targeted pytest passed:
  - `scm_hidden_fused_transition_matches_legacy_dual_semantics`
  - `paired_transition_matches_legacy_dual_semantics`
  - `fused_transition_dual_packed_matches_cat_semantics`

Fixed-seed A/B (`batch_wall_excl_compile_s`):

Command notes:

- held constant:
  - `TICL_POLICY_ENVGEN_CHECKPOINT=1`
  - `TICL_POLICY_ENVGEN_CHECKPOINT_REENTRANT=0`
  - `TICL_POLICY_ENVGEN_RAGGED_AFFINE=0`
  - `TICL_POLICY_FUSED_TRANSITION_STABLE_INPUT_SLOTS=0`
  - `TICL_POLICY_FUSED_TRANSITION_PAIRED=0`
  - `TICL_POLICY_TRANSITION_INNER_GROUPING=family`
  - `TICL_POLICY_TRANSITION_INNER_MIN_BUCKET=0`
  - `TICL_POLICY_TBPTT_STREAM_MERGE_AUTO=0`
  - `TICL_POLICY_TBPTT_STREAM_MERGE_WINDOWS=1`
  - `--seed-everything True -E 1 -n 1 -b 128`
  - `--pg-tbptt-window 64 --pg-env-replay-steps 1`
- only changed:
  - `TICL_POLICY_FUSED_TRANSITION_SCM_HIDDEN_FUSED=0/1`

Logs:

- hidden-fused disabled:
  - `20260307_seeded_batch128_scmhiddenfused_packed_off.log`
- hidden-fused enabled, cold:
  - `20260307_seeded_batch128_scmhiddenfused_packed_on.log`
- hidden-fused enabled, warm rerun:
  - `20260307_seeded_batch128_scmhiddenfused_packed_on_warm.log`

Metrics:

- hidden-fused disabled:
  - `batch_wall_excl_compile_s=132.729`
  - `batch_wall_excl_compile_per_batch_item_s=1.036947`
  - `gpu_util_avg=37.07`
  - `rollout_s=83.440`
  - `backward_s=49.290`
- hidden-fused enabled, cold:
  - `batch_wall_excl_compile_s=136.789`
  - `batch_wall_excl_compile_per_batch_item_s=1.068666`
  - `gpu_util_avg=30.41`
  - `rollout_s=86.960`
  - `backward_s=49.829`
- hidden-fused enabled, warm:
  - `batch_wall_excl_compile_s=133.683`
  - `batch_wall_excl_compile_per_batch_item_s=1.044399`
  - `gpu_util_avg=29.17`
  - `rollout_s=85.194`
  - `backward_s=48.489`

Outcome:

- cold run still regressed:
  - `132.729 -> 136.789` (`+3.06%`, worse)
- warm rerun removed most of the first-run penalty:
  - `136.789 -> 133.683` (`-2.27%` vs cold)
- after warm rerun it is still not a skyline:
  - `132.729 -> 133.683` (`+0.72%`, still worse)

Interpretation:

- the packed `2B` hidden-stack rewrite is a large step forward relative to the
  earlier dense block-diagonal hidden-fused path:
  - old hidden-fused: `155.141`
  - packed hidden-fused warm: `133.683`
- this validates the mainline direction:
  - layer-local caps and packed branch execution remove most of the previous
    hidden-stack waste
  - the remaining delta is small enough that Triton first-run cost mattered,
    so warm rerun was necessary
- the remaining regression is concentrated in rollout forward time:
  - rollout: `83.440 -> 85.194` (`+1.754s`)
  - backward: `49.290 -> 48.489` (`-0.801s`)
- this means the current packed prefix kernel already helps backward graph
  pressure slightly, but its forward path still loses to the legacy dense
  cuBLAS/bmm path on the realized shapes.

Decision:

- keep the packed SCM hidden-fused path in tree, but keep it disabled by
  default:
  - `TICL_POLICY_FUSED_TRANSITION_SCM_HIDDEN_FUSED=0`
- do **not** commit/push, because this pass still did not produce a new
  skyline.

Updated bottlenecks:

- highest priority:
  - reduce forward launch/dispatch overhead inside the packed `scm` hidden
    stack, because backward is now roughly neutral-to-better while rollout
    forward remains slower
- second priority:
  - revisit the `scm` first projection only after the hidden forward kernel is
    cheaper; it is now the most likely remaining serial overhead inside this
    path
- third priority:
  - `gp` input/RFF projection remains the next family-level hotspot after `scm`

## 2026-03-07 08:32: SCM hidden-stack packed forward fusion skyline

Goal:

- continue only on the `scm` hidden-stack forward path and reduce packed-path
  forward launch/dispatch overhead without returning to split branches or
  grouping experiments.

Code:

- `ticl/priors/environment_prior.py`
  - tightened Triton ragged loops to sample-local `in_size/out_size`
  - fused packed hidden-layer `affine + activation` into the prefix-tiled
    kernel path
  - extended prefix-tiled autograd to apply row-wise activation codes and
    handle their backward
- `ticl/fit_model.py`
  - default-enabled `TICL_POLICY_FUSED_TRANSITION_SCM_HIDDEN_FUSED=1` for
    `python -m ticl.fit_model rlpfn`

Validation:

- `conda run -n rlpfn python -m py_compile ...` passed
- targeted pytest passed:
  - `scm_hidden_fused_transition_matches_legacy_dual_semantics`

Fixed-seed A/B (`batch_wall_excl_compile_s`):

Command notes:

- held constant:
  - `TICL_POLICY_ENVGEN_CHECKPOINT=1`
  - `TICL_POLICY_ENVGEN_CHECKPOINT_REENTRANT=0`
  - `TICL_POLICY_ENVGEN_RAGGED_AFFINE=0`
  - `TICL_POLICY_FUSED_TRANSITION_STABLE_INPUT_SLOTS=0`
  - `TICL_POLICY_FUSED_TRANSITION_PAIRED=0`
  - `TICL_POLICY_TRANSITION_INNER_GROUPING=family`
  - `TICL_POLICY_TRANSITION_INNER_MIN_BUCKET=0`
  - `TICL_POLICY_TBPTT_STREAM_MERGE_AUTO=0`
  - `TICL_POLICY_TBPTT_STREAM_MERGE_WINDOWS=1`
  - `--seed-everything True -E 1 -n 1 -b 128`
  - `--pg-tbptt-window 64 --pg-env-replay-steps 1`
- only changed:
  - `TICL_POLICY_FUSED_TRANSITION_SCM_HIDDEN_FUSED=0/1`

Logs:

- hidden-fused disabled, sequential confirm:
  - `20260307_seeded_batch128_scmhiddenfused_packed_fuseact_off_seq.log`
- hidden-fused enabled, cold:
  - `20260307_seeded_batch128_scmhiddenfused_packed_fuseact_on.log`
- hidden-fused enabled, sequential confirm:
  - `20260307_seeded_batch128_scmhiddenfused_packed_fuseact_on_seq.log`

Metrics:

- hidden-fused disabled, sequential confirm:
  - `batch_wall_excl_compile_s=134.733`
  - `batch_wall_excl_compile_per_batch_item_s=1.052605`
  - `rollout_s=84.830`
  - `backward_s=49.903`
  - `gpu_util_avg=36.12`
- hidden-fused enabled, cold:
  - `batch_wall_excl_compile_s=120.653`
  - `batch_wall_excl_compile_per_batch_item_s=0.942600`
  - `rollout_s=76.247`
  - `backward_s=44.406`
  - `gpu_util_avg=31.49`
- hidden-fused enabled, sequential confirm:
  - `batch_wall_excl_compile_s=121.483`
  - `batch_wall_excl_compile_per_batch_item_s=0.949090`
  - `rollout_s=76.453`
  - `backward_s=45.031`
  - `gpu_util_avg=31.50`

Outcome:

- new skyline confirmed on sequential A/B:
  - `134.733 -> 121.483` (`-9.83%`, better)
- per-item KPI improved equally:
  - `1.052605 -> 0.949090` (`-9.83%`, better)
- rollout and backward both improved materially:
  - rollout: `84.830 -> 76.453` (`-9.88%`)
  - backward: `49.903 -> 45.031` (`-9.76%`)

Interpretation:

- the earlier packed path was still losing mainly on forward launch overhead.
- two changes together crossed the line:
  - stop scanning masked `k/o` tiles beyond each sample's true size
  - fuse packed hidden-layer activation into the affine kernel path
- this is strong evidence that the remaining `scm` hidden-stack bottleneck was
  not the math graph structure anymore; it was the residual forward launch and
  masked-tile overhead inside the packed kernel.
- note:
  - a parallel confirmation attempt caused mutual OOM between two concurrent
    benchmark processes on the same GPU; those logs are diagnostic only and
    should not be used for skyline judgment

Decision:

- keep `TICL_POLICY_FUSED_TRANSITION_SCM_HIDDEN_FUSED=1` as the default for
  `python -m ticl.fit_model rlpfn`
- commit/push this skyline

Next bottlenecks:

- highest priority after this skyline:
  - keep focus on `scm`, but only if a new forward-only hotspot emerges inside
    the first projection or adjacent policy path
- next family-level target:
  - `gp` input/RFF projection
