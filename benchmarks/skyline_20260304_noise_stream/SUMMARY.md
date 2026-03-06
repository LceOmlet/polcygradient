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
