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

## 2026-03-07 08:58: GP input/RFF projection active-dim packing still negative

Goal:

- stay strictly on the `gp` input/RFF projection path after the `scm`
  hidden-stack skyline
- reduce `gp` rollout forward overhead without returning to grouping,
  split-branch, or generic ragged side paths

Code:

- `ticl/priors/environment_prior.py`
  - added `TICL_POLICY_FUSED_TRANSITION_GP_INPUT_FUSED`
  - added an exact active-dim packing path for the `gp` input/RFF projection
  - kept all variants behind the same opt-in flag; default remains off
- `ticl/train.py`
  - startup log now prints `Policy fused transition GP input/RFF fused`
- `ticl/tests/priors/test_environment_prior.py`
  - added `gp_input_rff_fused_transition_matches_legacy_dual_semantics`

Validation:

- `conda run -n rlpfn python -m py_compile ...` passed
- targeted pytest passed:
  - `gp_input_rff_fused_transition_matches_legacy_dual_semantics`
  - `scm_hidden_fused_transition_matches_legacy_dual_semantics`
  - `paired_transition_matches_legacy_dual_semantics`

Fixed-seed A/B (`batch_wall_excl_compile_s`):

Command notes:

- held constant:
  - `TICL_POLICY_ENVGEN_CHECKPOINT=1`
  - `TICL_POLICY_ENVGEN_CHECKPOINT_REENTRANT=0`
  - `TICL_POLICY_ENVGEN_RAGGED_AFFINE=0`
  - `TICL_POLICY_FUSED_TRANSITION_STABLE_INPUT_SLOTS=0`
  - `TICL_POLICY_FUSED_TRANSITION_PAIRED=0`
  - `TICL_POLICY_FUSED_TRANSITION_SCM_HIDDEN_FUSED=1`
  - `TICL_POLICY_TRANSITION_INNER_GROUPING=family`
  - `TICL_POLICY_TRANSITION_INNER_MIN_BUCKET=0`
  - `TICL_POLICY_TBPTT_STREAM_MERGE_AUTO=0`
  - `TICL_POLICY_TBPTT_STREAM_MERGE_WINDOWS=1`
  - `--seed-everything True -E 1 -n 1 -b 128`
  - `--pg-tbptt-window 64 --pg-env-replay-steps 1`
- only changed:
  - `TICL_POLICY_FUSED_TRANSITION_GP_INPUT_FUSED=0/1`

Logs:

- baseline, GP input/RFF fused disabled:
  - `20260307_seeded_batch128_gpinputfused_off.log`
- exact packed-input + layout-tiled Triton:
  - `20260307_seeded_batch128_gpinputfused_gather_on.log`
- exact packed-input + prefix-tiled Triton:
  - `20260307_seeded_batch128_gpinputfused_prefix_on.log`
- exact packed-input + dense `batch_affine`:
  - `20260307_seeded_batch128_gpinputfused_packdense_on.log`

Metrics:

- fused disabled:
  - `batch_wall_excl_compile_s=118.801`
  - `batch_wall_excl_compile_per_batch_item_s=0.928134`
  - `rollout_s=75.089`
  - `backward_s=43.712`
  - `rollout_forward_est_s=31.376`
- packed-input + layout-tiled Triton:
  - `batch_wall_excl_compile_s=126.011`
  - `batch_wall_excl_compile_per_batch_item_s=0.984462`
  - `rollout_s=78.853`
  - `backward_s=47.158`
  - `rollout_forward_est_s=31.696`
- packed-input + prefix-tiled Triton:
  - `batch_wall_excl_compile_s=128.211`
  - `batch_wall_excl_compile_per_batch_item_s=1.001648`
  - `rollout_s=80.179`
  - `backward_s=48.032`
  - `rollout_forward_est_s=32.148`
- packed-input + dense `batch_affine`:
  - `batch_wall_excl_compile_s=127.897`
  - `batch_wall_excl_compile_per_batch_item_s=0.999199`
  - `rollout_s=79.979`
  - `backward_s=47.918`
  - `rollout_forward_est_s=32.061`

Outcome:

- no new skyline
- all three exact active-dim packing variants regressed versus baseline:
  - layout-tiled Triton: `+6.07%`
  - prefix-tiled Triton: `+7.92%`
  - packed-input + dense `batch_affine`: `+7.66%`

Interpretation:

- the `gp` input/RFF hotspot is not dominated by inactive input-width padding or
  input-index indirection alone
- even when the packed path is exact and uses dense batched GEMM, forward does
  not improve:
  - `31.376 -> 32.061` for the best exact packed-input variant
- this means the next `gp` mainline target should move deeper than pure input
  packing:
  - the remaining cost is more likely in the RFF feature width / downstream
    projection structure and the dual-packed state/reward execution path

Decision:

- keep `TICL_POLICY_FUSED_TRANSITION_GP_INPUT_FUSED=0` as the default
- do not commit/push; this is a recorded negative result

Next GP directions:

- first:
  - profile and target the `gp` post-RFF/output projection path (`phi -> a`)
    because input packing alone did not move forward wall time
- second:
  - if `gp` still needs a fused path, fuse a larger exact subgraph rather than
    just repacking the input dimension

## 2026-03-07 09:12: GP output projection and larger fused subgraph still negative

Goal:

- stay strictly on the `gp` mainline after the negative input-packing results
- first target the `phi -> a` output projection directly
- if that still failed, try a larger exact transition-only fused subgraph
  without returning to input-packing variants

Code:

- `ticl/priors/environment_prior.py`
  - added `TICL_POLICY_FUSED_TRANSITION_GP_OUTPUT_FUSED`
  - added exact prefix-tiled `gp` output projection fastpath for `phi -> a`
  - added `TICL_POLICY_FUSED_TRANSITION_GP_OUTPUT_SUBGRAPH`
  - added a larger exact `gp` transition-only fused subgraph path:
    - state branch keeps dense `phi -> a`
    - reward branch uses exact scalar head dot-product
    - both branches are executed inside one transition core with preserved
      dual-noise semantics
- `ticl/train.py`
  - startup log now prints:
    - `Policy fused transition GP output projection fused`
    - `Policy fused transition GP output subgraph fused`
- `ticl/tests/priors/test_environment_prior.py`
  - added:
    - `gp_output_projection_fused_transition_matches_legacy_dual_semantics`
    - `gp_output_subgraph_fused_transition_matches_legacy_dual_semantics`

Validation:

- `conda run -n rlpfn python -m py_compile ...` passed
- targeted pytest passed:
  - `gp_output_projection_fused_transition_matches_legacy_dual_semantics`
  - `gp_output_subgraph_fused_transition_matches_legacy_dual_semantics`
  - `gp_input_rff_fused_transition_matches_legacy_dual_semantics`
  - `scm_hidden_fused_transition_matches_legacy_dual_semantics`
  - `paired_transition_matches_legacy_dual_semantics`

Fixed-seed A/B (`batch_wall_excl_compile_s`):

Command notes:

- held constant:
  - `TICL_POLICY_ENVGEN_CHECKPOINT=1`
  - `TICL_POLICY_ENVGEN_CHECKPOINT_REENTRANT=0`
  - `TICL_POLICY_ENVGEN_RAGGED_AFFINE=0`
  - `TICL_POLICY_FUSED_TRANSITION_STABLE_INPUT_SLOTS=0`
  - `TICL_POLICY_FUSED_TRANSITION_PAIRED=0`
  - `TICL_POLICY_FUSED_TRANSITION_SCM_HIDDEN_FUSED=1`
  - `TICL_POLICY_TRANSITION_INNER_GROUPING=family`
  - `TICL_POLICY_TRANSITION_INNER_MIN_BUCKET=0`
  - `TICL_POLICY_TBPTT_STREAM_MERGE_AUTO=0`
  - `TICL_POLICY_TBPTT_STREAM_MERGE_WINDOWS=1`
  - `--seed-everything True -E 1 -n 1 -b 128`
  - `--pg-tbptt-window 64 --pg-env-replay-steps 1`
- baseline:
  - `TICL_POLICY_FUSED_TRANSITION_GP_INPUT_FUSED=0`
  - `TICL_POLICY_FUSED_TRANSITION_GP_OUTPUT_FUSED=0`
  - `TICL_POLICY_FUSED_TRANSITION_GP_OUTPUT_SUBGRAPH=0`

Logs:

- baseline:
  - `20260307_seeded_batch128_gpinputfused_off.log`
- output projection fused:
  - `20260307_seeded_batch128_gpoutputfused_on.log`
- larger exact GP output subgraph:
  - `20260307_seeded_batch128_gpoutputsubgraph_on.log`

Metrics:

- baseline:
  - `batch_wall_excl_compile_s=118.801`
  - `rollout_s=75.089`
  - `backward_s=43.712`
  - `rollout_forward_est_s=31.376`
- output projection fused:
  - `batch_wall_excl_compile_s=124.700`
  - `rollout_s=78.247`
  - `backward_s=46.453`
  - `rollout_forward_est_s=31.794`
- larger exact GP output subgraph:
  - `batch_wall_excl_compile_s=129.859`
  - `rollout_s=81.100`
  - `backward_s=48.760`
  - `rollout_forward_est_s=32.340`

Outcome:

- no new skyline
- output projection fused regressed:
  - `118.801 -> 124.700` (`+4.97%`)
- larger exact GP output subgraph regressed even more:
  - `118.801 -> 129.859` (`+9.31%`)

Interpretation:

- this round materially narrows the `gp` search space:
  - pure input packing already failed
  - direct `phi -> a` optimization also failed
  - a larger exact transition-only fused subgraph also failed
- all three results worsened `rollout_forward_est_s`, so the remaining `gp`
  bottleneck is not just output padding or graph fragmentation in the output
  head
- the next `gp` mainline target should therefore move earlier:
  - first projection / RFF generation (`x -> W,b -> cos`)
  - or better observability that measures the first projection and second
    projection separately under fixed seed

Decision:

- keep all new `gp` output-side flags default-off
- do not commit/push; this is a recorded negative result

## 2026-03-07 09:44: GP first projection / RFF timing fixed, exact fused path still not a skyline

Goal:

- stay strictly on the `gp` mainline after the negative input/output-side trials
- target the earlier `x -> W,b -> cos` first projection / RFF generation path
- improve observability first where needed, but do not return to input-packing or output-projection side paths

Code:

- `ticl/priors/environment_prior.py`
  - added `TICL_POLICY_FUSED_TRANSITION_GP_RFF_FUSED`
  - added exact GP first-projection/RFF fused path inside `_build_gp_hetero_batch_fn`
  - added GP projection stage timing counters (`first_projection_wall_s`, `second_projection_wall_s`)
  - fixed mixed-family profiling by forcing GP transition groups onto the synchronous path only when
    `TICL_PROFILE_GP_PROJECTION_TIMING=1`; default hot path is unchanged
  - added observability counters:
    - `transition_gp_profile_group_count`
    - `transition_gp_profile_sync_group_count`
- `ticl/train.py`
  - startup log prints `Policy fused transition GP RFF fused`
  - startup log prints `Policy GP projection timing profile`
  - `pg-phase` now prints:
    - `rollout_transition_gp_first_proj_ms`
    - `rollout_transition_gp_second_proj_ms`
    - `rollout_transition_gp_proj_calls`
    - `rollout_transition_gp_rff_fused_calls`
    - `rollout_transition_gp_profile_groups`
    - `rollout_transition_gp_profile_sync_groups`
- `ticl/tests/priors/test_environment_prior.py`
  - added `gp_rff_fused_transition_matches_legacy_dual_semantics`
  - added `mixed_family_gp_projection_profile_survives_async_rollout`

Validation:

- `conda run -n rlpfn python -m py_compile ...` passed
- targeted pytest passed:
  - `gp_rff_fused_transition_matches_legacy_dual_semantics`
  - `mixed_family_gp_projection_profile_survives_async_rollout`

Mixed-family diagnostic (`TICL_PROFILE_GP_PROJECTION_TIMING=1`):

Log:

- `20260307_seeded_batch128_gpprojtiming_diag_sync.log`

Metrics:

- `batch_wall_excl_compile_s=125.236`
- `rollout_transition_gp_first_proj_ms=726.82`
- `rollout_transition_gp_second_proj_ms=377.38`
- `rollout_transition_gp_proj_calls=1984`
- `rollout_transition_gp_profile_groups=1`
- `rollout_transition_gp_profile_sync_groups=1`

Interpretation:

- the earlier missing GP stage timing in mixed-family training batches was an observability problem caused by the
  async transition-launch path, not absence of GP work
- under fixed seed, one of the two family groups is GP and only that group is forced synchronous in timing mode
- on this batch, the GP first projection is the larger share of measured GP projection time:
  - first projection: `65.8%`
  - second projection: `34.2%`

Fixed-seed A/B (`batch_wall_excl_compile_s`):

Command notes:

- held constant:
  - `TICL_POLICY_ENVGEN_CHECKPOINT=1`
  - `TICL_POLICY_ENVGEN_CHECKPOINT_REENTRANT=0`
  - `TICL_POLICY_ENVGEN_RAGGED_AFFINE=0`
  - `TICL_POLICY_FUSED_TRANSITION_STABLE_INPUT_SLOTS=0`
  - `TICL_POLICY_FUSED_TRANSITION_PAIRED=0`
  - `TICL_POLICY_FUSED_TRANSITION_SCM_HIDDEN_FUSED=1`
  - `TICL_POLICY_FUSED_TRANSITION_GP_INPUT_FUSED=0`
  - `TICL_POLICY_FUSED_TRANSITION_GP_OUTPUT_FUSED=0`
  - `TICL_POLICY_FUSED_TRANSITION_GP_OUTPUT_SUBGRAPH=0`
  - `TICL_POLICY_TRANSITION_INNER_GROUPING=family`
  - `TICL_POLICY_TRANSITION_INNER_MIN_BUCKET=0`
  - `TICL_POLICY_TBPTT_STREAM_MERGE_AUTO=0`
  - `TICL_POLICY_TBPTT_STREAM_MERGE_WINDOWS=1`
  - `--seed-everything True -E 1 -n 1 -b 128`
  - `--pg-tbptt-window 64 --pg-env-replay-steps 1`
- only changed:
  - `TICL_POLICY_FUSED_TRANSITION_GP_RFF_FUSED=0/1`

Logs:

- baseline, RFF fused disabled:
  - `20260307_seeded_batch128_gprfffused_off.log`
- RFF fused enabled, first run:
  - `20260307_seeded_batch128_gprfffused_on.log`
- RFF fused enabled, sequential confirm:
  - `20260307_seeded_batch128_gprfffused_on_seq.log`

Metrics:

- baseline:
  - `batch_wall_excl_compile_s=127.711`
  - `rollout_s=80.235`
  - `backward_s=47.476`
  - `rollout_forward_est_s=32.759`
- RFF fused enabled, first run:
  - `batch_wall_excl_compile_s=124.877`
  - `rollout_s=78.402`
  - `backward_s=46.475`
  - `rollout_forward_est_s=31.927`
- RFF fused enabled, sequential confirm:
  - `batch_wall_excl_compile_s=129.967`
  - `rollout_s=81.318`
  - `backward_s=48.650`
  - `rollout_forward_est_s=32.668`

Outcome:

- no new skyline
- the first `on` run looked positive (`-2.22%`), but the sequential confirm regressed (`+1.77%`)
- mean of the two `on` runs is `127.422`, only `-0.23%` versus baseline, which is too small and unstable to count

Interpretation:

- the exact GP first-projection/RFF fused path is not yet a robust throughput win on the mixed-family training skyline
- the new timing proves the `gp` mainline target was chosen correctly: the first projection is indeed heavier than the
  second projection on the real batch
- however, the contribution is still only one GP family group inside the mixed batch, so modest improvements are easy to
  drown in normal end-to-end variance

Decision:

- keep `TICL_POLICY_FUSED_TRANSITION_GP_RFF_FUSED=0` as the default
- keep the new GP projection timing observability in tree
- do not commit/push; no skyline was confirmed

Next GP directions (ordered by importance):

- highest priority:
  - optimize inside the first projection itself (`x @ W + b` and cosine application), not broader GP-side graph reshaping
- second:
  - if a future exact kernel looks promising, validate it first with the GP timing profile enabled so the GP family share
    is directly observable before relying on full mixed-batch wall time
- third:
  - if mixed-batch variance keeps masking small GP gains, use a fixed GP-only diagnostic to qualify the kernel before
    returning to the full `rlpfn` skyline

## 2026-03-07 10:06: GP first-projection cos-specific kernel attempt rejected by GP-only steady-state

Goal:

- stay strictly on the `gp` first-projection mainline (`x @ W + b + cos`)
- do not expand to larger GP subgraphs
- because mixed-family wall time can hide small GP effects, qualify the candidate with a fixed GP-only diagnostic first

What was tried:

- implemented a cos-specific first-projection fastpath for `TICL_POLICY_FUSED_TRANSITION_GP_RFF_FUSED`
- the idea was to specialize the first projection kernel and reduce generic activation-path overhead
- after GP-only steady-state measurement, this kernel was rejected and reverted; only the GP timing observability work remains in tree

Validation:

- after reverting the negative kernel, `py_compile` passed
- targeted pytest passed:
  - `gp_rff_fused_transition_matches_legacy_dual_semantics`
  - `mixed_family_gp_projection_profile_survives_async_rollout`

GP-only diagnostic method:

- fixed `family=gp`
- `batch_size=128`
- `n_samples=64`
- `num_features=432`
- `TICL_PROFILE_GP_PROJECTION_TIMING=1`
- same process warmup twice; only the second rollout is reported to avoid Triton first-compile pollution

Logs:

- baseline steady-state:
  - `20260307_gp_only_rffdiag_off_warm2.txt`
- candidate steady-state:
  - `20260307_gp_only_rffdiag_on_warm2.txt`

Metrics:

- baseline steady-state:
  - `wall_s=0.606313`
  - `transition_wall_ms=108.141`
  - `gp_first_proj_ms=9.110`
  - `gp_second_proj_ms=5.969`
  - `gp_proj_calls=64`
- candidate steady-state:
  - `wall_s=0.816237`
  - `transition_wall_ms=315.255`
  - `gp_first_proj_ms=237.004`
  - `gp_second_proj_ms=6.706`
  - `gp_proj_calls=64`

Outcome:

- rejected before full `rlpfn` A/B
- GP-only steady-state already regressed badly:
  - rollout wall: `0.606313 -> 0.816237` (`+34.62%`)
  - transition wall: `108.141 -> 315.255` (`+191.53%`)
  - first projection wall: `9.110 -> 237.004` (`+2501.47%`)
- second projection stayed near baseline, confirming the regression is inside the attempted first-projection kernel itself

Interpretation:

- this was not a mixed-family masking problem; the candidate kernel is intrinsically worse even in GP-only steady-state
- compile pollution was checked explicitly by warming inside the same process; the regression persisted
- therefore there was no reason to spend another full fixed-seed `python -m ticl.fit_model rlpfn` run on this kernel

Decision:

- revert the cos-specific GP first-projection kernel attempt
- keep `TICL_POLICY_FUSED_TRANSITION_GP_RFF_FUSED=0` as the default
- keep the GP projection timing observability changes, since they are now required to screen future GP kernels correctly

Next GP directions:

- first:
  - keep using `TICL_PROFILE_GP_PROJECTION_TIMING=1` as the gate before any full `rlpfn` benchmark
- second:
  - target cheaper first-projection changes than a custom Triton rewrite, for example launch-shape tuning or better reuse inside the existing generic tiled kernel path
- third:
  - only return to full mixed-family skyline once GP-only steady-state shows a clear first-projection win

## 2026-03-07 10:42: GP first-projection generic tiled launch-shape tuning rejected by GP-only steady-state

Goal:

- stay strictly on the `gp` first-projection mainline
- do not add a new kernel family
- only tune the existing generic tiled path used by `TICL_POLICY_FUSED_TRANSITION_GP_RFF_FUSED`
- keep using `TICL_PROFILE_GP_PROJECTION_TIMING=1` as the gate before any full `rlpfn` run

Code changes kept in tree:

- the generic tiled autograd wrappers now accept runtime `block_o`, `block_k`, and `num_warps`
- the `gp` RFF fused path now reads:
  - `TICL_POLICY_GP_RFF_BLOCK_O`
  - `TICL_POLICY_GP_RFF_BLOCK_K`
  - `TICL_POLICY_GP_RFF_NUM_WARPS`
- these only tune the existing generic tiled kernels; no new Triton algorithm was added
- training startup now prints the active GP RFF tile config
- the GPU-only GP RFF semantics test is now explicitly skipped on CPU-only machines

Validation:

- `py_compile` passed
- targeted pytest passed in the current sandbox:
  - `gp_rff_fused_transition_matches_legacy_dual_semantics`
  - `mixed_family_gp_projection_profile_survives_async_rollout`
  - both skipped cleanly without CUDA instead of failing spuriously

Important observability correction:

- a first GP-only sweep was invalid as a gate:
  - `20260307_gp_only_rff_tile_sweep.jsonl`
  - `20260307_gp_only_rff_tile_sweep_family.jsonl`
- reason:
  - passing `env_seeds_override` / `rollout_seeds_override` into `rollout_with_policy` causes
    `EnvironmentPrior._sample_environment_family_coarse_batch(...)` to build per-sample generators
  - that disables the fused `transition_generator` path entirely (`generators is not None`)
  - symptom in the invalid logs:
    - `gp_rff_fused_calls=0`
    - `gp_proj_calls=0` or `gp_first_proj_ms=0`
- decision:
  - do not use strict seed overrides for this GP-only gate
  - instead, fix the sampled `h_list` and reseed the process before each config so the fused path remains active

Valid GP-only steady-state method:

- `family=gp` only
- `batch_size=128`
- `n_samples=64`
- `num_features=432`
- fixed sampled `h_list` with 4 repeated GP structures
- same process warmup twice; only the second rollout is reported
- `TICL_PROFILE_ROLLOUT_TIMING=1`
- `TICL_PROFILE_GP_PROJECTION_TIMING=1`
- valid log:
  - `20260307_gp_only_rff_tile_sweep_family_fused.jsonl`

Representative sampled structures:

- sample 1:
  - `state_dim=311`
  - `obs_dim=230`
  - `action_dim=24`
  - `noise_dim=24`
  - `zero_pad_dim=353`
  - `gp_rff_features=154`
- sample 2:
  - `state_dim=50`
  - `obs_dim=209`
  - `action_dim=11`
  - `noise_dim=24`
  - `zero_pad_dim=376`
  - `gp_rff_features=79`
- sample 3:
  - `state_dim=31`
  - `obs_dim=29`
  - `action_dim=29`
  - `noise_dim=39`
  - `zero_pad_dim=95`
  - `gp_rff_features=213`
- sample 4:
  - `state_dim=369`
  - `obs_dim=69`
  - `action_dim=22`
  - `noise_dim=32`
  - `zero_pad_dim=191`
  - `gp_rff_features=186`

Results:

- baseline (`off`):
  - `wall_s=0.501950`
  - `transition_wall_ms=90.880`
  - `gp_first_proj_ms=9.257`
  - `gp_second_proj_ms=5.686`
  - `gp_proj_calls=64`
  - `gp_rff_fused_calls=0`
  - `gp_profile_groups=1`
- `block_o=32 block_k=32 num_warps=4`:
  - `wall_s=0.533223`
  - `transition_wall_ms=93.207`
  - `gp_first_proj_ms=19.847`
  - `gp_second_proj_ms=6.083`
  - `gp_rff_fused_calls=64`
- `block_o=32 block_k=32 num_warps=2`:
  - `wall_s=0.533826`
  - `transition_wall_ms=95.120`
  - `gp_first_proj_ms=20.195`
  - `gp_second_proj_ms=6.351`
  - `gp_rff_fused_calls=64`
- `block_o=64 block_k=32 num_warps=4`:
  - `wall_s=0.532688`
  - `transition_wall_ms=94.702`
  - `gp_first_proj_ms=20.006`
  - `gp_second_proj_ms=6.357`
  - `gp_rff_fused_calls=64`

Outcome:

- no new skyline
- all tested launch/tile settings regressed in the GP-only gate
- best tuned candidate was still clearly negative:
  - wall: `0.501950 -> 0.532688` (`+6.12%`)
  - transition wall: `90.880 -> 94.702` (`+4.21%`)
  - first projection wall: `9.257 -> 20.006` (`+116.13%`)
- therefore there was no reason to spend a full fixed-seed `python -m ticl.fit_model rlpfn` run on these settings

Interpretation:

- the current generic tiled RFF path is not launch-shape limited in the way this sweep assumed
- changing `BLOCK_O/BLOCK_K/num_warps` on the existing kernel made the first projection substantially slower even when the
  rest of the GP transition path was held fixed
- the gate itself is now trustworthy because:
  - `gp_profile_groups=1`
  - `gp_proj_calls=64`
  - `gp_rff_fused_calls=64` on the candidate runs

Decision:

- keep `TICL_POLICY_FUSED_TRANSITION_GP_RFF_FUSED=0` as the default
- keep the GP RFF tile tuning hooks in tree for future controlled experiments
- do not commit/push; no skyline was found

Next directions:

- first:
  - do not continue blind tile/warp sweeps on the current generic tiled path
- second:
  - if GP stays on the mainline, the next worthwhile target is not wider tuning but a cheaper first-projection dataflow
    inside the existing path, with the GP-only gate kept in front
- third:
  - always avoid `env_seeds_override` / `rollout_seeds_override` when using this GP-only fused-transition gate, because
    they silently disable the fused transition builder

## 2026-03-07 11:34: GP packed env-input first-projection dataflow rejected by GP-only steady-state

Goal:

- stay on the `gp` mainline
- stop sweeping generic tiled launch shape
- hit the first-projection dataflow itself by removing rollout-side `zero_pad` traffic before `x @ W + b`
- gate everything with `TICL_PROFILE_GP_PROJECTION_TIMING=1` before any full `rlpfn` run

Code kept in tree:

- added `TICL_POLICY_FUSED_TRANSITION_GP_PACKED_ENV_INPUT`
- GP transition builder now keeps two exact paths:
  - natural env-input path
  - packed rollout env-input path
- packed-path observability added:
  - `transition_packed_env_input_group_count`
  - `transition_packed_env_input_call_count`
- fixed the GP fused-transition semantics test so candidate flags no longer perturb `x/grad` sampling order:
  - use a dedicated value generator after building the transition function

Validation:

- `py_compile` passed
- targeted pytest passed:
  - `gp_rff_fused_transition_matches_legacy_dual_semantics`
  - `mixed_family_gp_projection_profile_survives_async_rollout`

Important observability correction:

- the first GP-only packed-input gate attempt was invalid even though it used `family=gp`
- reason:
  - it still left `batch_vectorized_grouping=structure`
  - that kept the rollout off the family-coarse fused path
- symptom in the invalid log:
  - `transition_fused_groups=0`
  - `gp_proj_calls=0`
  - `packed_env_input_calls=0`
- after fixing `batch_vectorized_grouping=family`, the gate became valid and hit the packed path

Valid GP-only steady-state method:

- `family=gp`
- `batch_size=128`
- `n_samples=64`
- `num_features=432`
- `batch_vectorized_grouping=family`
- `TICL_PROFILE_ROLLOUT_TIMING=1`
- `TICL_PROFILE_GP_PROJECTION_TIMING=1`
- `TICL_POLICY_FUSED_TRANSITION_GP_RFF_FUSED=0`
- same process warmup twice; only warm step 2 is compared
- fixed repeated `h_list` with the same 4 heterogeneous GP structures used in the earlier valid GP gate

Logs:

- valid forward-order gate (`off -> on`):
  - `20260307_gp_only_packed_env_input_family_fused.jsonl`
- valid reverse-order gate (`on -> off`):
  - `20260307_gp_only_packed_env_input_family_fused_rev.txt`

Metrics:

- forward-order steady-state baseline (`off`, warm 2):
  - `wall_s=0.484033`
  - `transition_wall_ms=67.366`
  - `transition_env_pack_wall_ms=6.975`
  - `gp_first_proj_ms=9.008`
  - `gp_second_proj_ms=5.491`
  - `gp_proj_calls=64`
  - `transition_fused_groups=1`
- forward-order steady-state candidate (`on`, warm 2):
  - `wall_s=0.699474`
  - `transition_wall_ms=74.263`
  - `transition_env_pack_wall_ms=16.068`
  - `gp_first_proj_ms=9.361`
  - `gp_second_proj_ms=5.734`
  - `gp_proj_calls=64`
  - `transition_fused_groups=1`
  - `packed_env_input_groups=1`
  - `packed_env_input_calls=64`
- reverse-order steady-state candidate (`on`, warm 2):
  - `wall_s=0.638676`
  - `transition_wall_ms=72.897`
  - `transition_env_pack_wall_ms=16.282`
  - `gp_first_proj_ms=9.313`
  - `gp_second_proj_ms=5.655`
  - `gp_proj_calls=64`
  - `transition_fused_groups=1`
  - `packed_env_input_groups=1`
  - `packed_env_input_calls=64`
- reverse-order steady-state baseline (`off`, warm 2):
  - `wall_s=0.508286`
  - `transition_wall_ms=69.725`
  - `transition_env_pack_wall_ms=7.449`
  - `gp_first_proj_ms=9.124`
  - `gp_second_proj_ms=5.513`
  - `gp_proj_calls=64`
  - `transition_fused_groups=1`

Outcome:

- no new skyline
- the packed env-input path definitely executed, but it did not reduce the first projection
- forward-order comparison:
  - wall: `0.484033 -> 0.699474` (`+44.51%`)
  - transition wall: `67.366 -> 74.263` (`+10.24%`)
  - first projection wall: `9.008 -> 9.361` (`+3.92%`)
  - env-pack wall: `6.975 -> 16.068` (`+130.38%`)
- reverse-order comparison confirms the same direction:
  - wall: `0.508286 -> 0.638676` (`+25.65%`)
  - transition wall: `69.725 -> 72.897` (`+4.55%`)
  - first projection wall: `9.124 -> 9.313` (`+2.07%`)
  - env-pack wall: `7.449 -> 16.282` (`+118.58%`)

Interpretation:

- this is not a launch-shape problem anymore; the packed path is spending more time in rollout-side packing than it saves inside the first projection
- `zero_pad` traffic was not the dominating first-projection bottleneck in this GP path under the current family-coarse fused rollout
- the observability added here is still useful because it cleanly separates:
  - invalid gates where the fused path was never active
  - valid gates where the packed path did run and still lost

Decision:

- do not run a full fixed-seed `python -m ticl.fit_model rlpfn` A/B for this candidate
- do not commit/push; no skyline was found
- keep `TICL_POLICY_FUSED_TRANSITION_GP_PACKED_ENV_INPUT=0` as the effective default path

Next directions:

- first:
  - if GP stays on the mainline, stop attacking `zero_pad` packing on the rollout side
- second:
  - the next GP target should be inside the first projection compute/data reuse itself, not another input-layout rewrite
- third:
  - keep using `TICL_PROFILE_GP_PROJECTION_TIMING=1` and the new packed-path counters before any future full `rlpfn` benchmark

## 2026-03-07 12:26: GP shared first-projection compute reuse lowers `gp_first_proj_ms` but still loses on GP-only wall

Goal:

- stay on the `gp` mainline
- stop changing rollout-side input layout
- target first-projection compute/data reuse directly inside the fused transition path
- keep `TICL_PROFILE_GP_PROJECTION_TIMING=1` as the gate before any full `rlpfn` run

Code kept in tree:

- added `TICL_POLICY_FUSED_TRANSITION_GP_SHARED_FIRST_PROJ`
- new shared-input GP transition path:
  - keeps legacy semantics for normal rollout input
  - preserves legacy RNG consumption by rebuilding the shared path from the pre-legacy RNG state and then restoring the post-legacy RNG state
  - falls back to legacy dual-path semantics for explicit `x_is_dual_packed=True`
- second projection inside the shared path was also collapsed back to a single dual-output affine after the first version showed a clear second-projection regression
- startup logging now prints the new flag
- semantics coverage was extended inside `gp_rff_fused_transition_matches_legacy_dual_semantics`

Validation:

- `py_compile` passed
- targeted pytest passed:
  - `gp_rff_fused_transition_matches_legacy_dual_semantics`
  - `mixed_family_gp_projection_profile_survives_async_rollout`

Valid GP-only gate method:

- same gate as the packed env-input experiment:
  - `family=gp`
  - `batch_size=128`
  - `n_samples=64`
  - `num_features=432`
  - `batch_vectorized_grouping=family`
  - `TICL_PROFILE_ROLLOUT_TIMING=1`
  - `TICL_PROFILE_GP_PROJECTION_TIMING=1`
- held fixed:
  - `TICL_POLICY_ENVGEN_CHECKPOINT=1`
  - `TICL_POLICY_ENVGEN_CHECKPOINT_REENTRANT=0`
  - `TICL_POLICY_ENVGEN_RAGGED_AFFINE=0`
  - `TICL_POLICY_FUSED_TRANSITION_GP_INPUT_FUSED=0`
  - `TICL_POLICY_FUSED_TRANSITION_GP_OUTPUT_FUSED=0`
  - `TICL_POLICY_FUSED_TRANSITION_GP_OUTPUT_SUBGRAPH=0`
  - `TICL_POLICY_FUSED_TRANSITION_GP_RFF_FUSED=0`
  - `TICL_POLICY_FUSED_TRANSITION_GP_PACKED_ENV_INPUT=0`
- only changed:
  - `TICL_POLICY_FUSED_TRANSITION_GP_SHARED_FIRST_PROJ=0/1`

Logs:

- forward-order gate (`off -> on`):
  - `20260307_gp_only_shared_first_proj_family_fused.jsonl`
- reverse-order gate (`on -> off`):
  - `20260307_gp_only_shared_first_proj_family_fused_rev.txt`

Metrics:

- forward-order steady-state baseline (`off`, warm 2):
  - `wall_s=0.471135`
  - `transition_wall_ms=66.430`
  - `gp_first_proj_ms=9.230`
  - `gp_second_proj_ms=5.272`
- forward-order steady-state candidate (`on`, warm 2):
  - `wall_s=0.599431`
  - `transition_wall_ms=66.536`
  - `gp_first_proj_ms=7.894`
  - `gp_second_proj_ms=5.316`
- reverse-order steady-state candidate (`on`, warm 2):
  - `wall_s=0.596048`
  - `transition_wall_ms=66.614`
  - `gp_first_proj_ms=7.480`
  - `gp_second_proj_ms=5.211`
- reverse-order steady-state baseline (`off`, warm 2):
  - `wall_s=0.470394`
  - `transition_wall_ms=66.572`
  - `gp_first_proj_ms=8.924`
  - `gp_second_proj_ms=5.342`

Outcome:

- no new skyline
- the shared path consistently improves the measured first projection:
  - forward order: `9.230 -> 7.894` (`-14.47%`)
  - reverse order: `8.924 -> 7.480` (`-16.18%`)
- after collapsing second projection back to a single affine, `gp_second_proj_ms` is effectively back at baseline
- but the GP-only steady-state wall still regresses badly:
  - forward order: `0.471135 -> 0.599431` (`+27.23%`)
  - reverse order: `0.470394 -> 0.596048` (`+26.71%`)
- `transition_wall_ms` is nearly flat in both directions, so the remaining regression is not explained by the current transition profile slices

Interpretation:

- this is the first GP mainline attempt that truly reduced `gp_first_proj_ms` under a valid GP-only gate
- however, the candidate still loses on the real KPI because there is a larger unobserved overhead outside the currently exposed `transition_wall_ms / gp_first_proj_ms / gp_second_proj_ms` slices
- the missing cost is now the dominant issue, not the first projection itself

Decision:

- do not run a full fixed-seed `python -m ticl.fit_model rlpfn` A/B for this candidate
- do not commit/push; no skyline was found
- keep `TICL_POLICY_FUSED_TRANSITION_GP_SHARED_FIRST_PROJ=0` as the effective default path

Next directions:

- first:
  - before any more GP kernel work, add finer observability around the shared transition path so the missing `wall_s` regression can be assigned to a real bucket
- second:
  - if that hidden cost turns out to be host-side launch/sync overhead, the next GP step should fuse more of the shared path into one launch instead of only lowering `gp_first_proj_ms`
- third:
  - if the hidden cost is outside transition entirely, stop on GP and return to the larger non-GP mainline bottleneck

## 2026-03-07 13:01: shared-GP hidden wall regression is in setup/build, not in the transition loop

Goal:

- stay on the `gp` mainline only long enough to assign the remaining shared-path wall regression to a real bucket
- do not change the kernel again
- use the new shared-path observability to decide whether GP should continue

Code kept in tree:

- extended shared GP projection profiling with:
  - `shared_total_wall_s`
  - `shared_core_wall_s`
  - `shared_noise_wall_s`
  - `shared_checkpoint_wall_s`
  - `shared_post_wall_s`
  - `shared_call_count`
- added rollout setup/build buckets:
  - `transition_setup_wall_ms`
  - `transition_family_build_wall_ms`
  - `transition_generator_build_wall_ms`
  - `transition_gp_shared_build_wall_ms`
- fixed `_consume_gp_projection_profile()` propagation so the shared-path counters reach the top-level rollout profile

Validation:

- `py_compile` passed
- targeted pytest passed:
  - `gp_rff_fused_transition_matches_legacy_dual_semantics`
  - `mixed_family_gp_projection_profile_survives_async_rollout`

Diagnostic log:

- rerun after the shared-profile propagation fix:
  - `20260307_gp_shared_first_proj_observe_rerun.jsonl`

Steady-state rows used for diagnosis:

- forward-order baseline (`off`, warm 2):
  - `wall_s=0.491454`
  - `transition_wall_ms=69.771`
  - `transition_setup_wall_ms=400.010`
  - `transition_generator_build_wall_ms=148.791`
  - `gp_first_proj_ms=9.382`
- forward-order shared candidate (`on`, warm 2):
  - `wall_s=0.613760`
  - `transition_wall_ms=69.025`
  - `transition_setup_wall_ms=523.039`
  - `transition_generator_build_wall_ms=273.897`
  - `gp_first_proj_ms=8.048`
  - `gp_shared_total_wall_ms=35.762`
- reverse-order baseline (`off_rev`, warm 2):
  - `wall_s=0.467157`
  - `transition_wall_ms=65.999`
  - `transition_setup_wall_ms=380.532`
  - `transition_generator_build_wall_ms=141.097`
  - `gp_first_proj_ms=8.904`
- reverse-order shared candidate (`on_rev`, warm 2):
  - `wall_s=0.584543`
  - `transition_wall_ms=66.387`
  - `transition_setup_wall_ms=498.316`
  - `transition_generator_build_wall_ms=257.802`
  - `gp_first_proj_ms=7.432`
  - `gp_shared_total_wall_ms=35.045`

Outcome:

- no new skyline
- the shared path still lowers the real first-projection slice:
  - forward: `9.382 -> 8.048` (`-14.21%`)
  - reverse: `8.904 -> 7.432` (`-16.53%`)
- but the GP-only steady-state wall still regresses:
  - forward: `0.491454 -> 0.613760` (`+24.89%`)
  - reverse: `0.467157 -> 0.584543` (`+25.13%`)
- the regression is not in the transition loop:
  - forward: `transition_wall_ms 69.771 -> 69.025` (`-0.746 ms`)
  - reverse: `transition_wall_ms 65.999 -> 66.387` (`+0.387 ms`)
- the regression is almost entirely in setup/build before the transition loop:
  - forward: `transition_setup_wall_ms 400.010 -> 523.039` (`+123.029 ms`)
  - reverse: `transition_setup_wall_ms 380.532 -> 498.316` (`+117.785 ms`)
- that setup/build regression is itself dominated by transition-generator construction:
  - forward: `transition_generator_build_wall_ms 148.791 -> 273.897` (`+125.105 ms`)
  - reverse: `transition_generator_build_wall_ms 141.097 -> 257.802` (`+116.705 ms`)
- the new shared-path transition counters are now live, but they are not the main regression source:
  - `gp_shared_total_wall_ms` is only about `35 ms`
  - `unprofiled_wall_ms` stays almost flat

Interpretation:

- this closes the GP observability gap
- the candidate does improve the compute path that was targeted
- however, the real KPI is lost earlier, in host-side/setup-side transition-generator build work
- this is not a case where another GP transition kernel tweak should continue blindly

Decision:

- stop the current GP mainline here
- do not run a full fixed-seed `python -m ticl.fit_model rlpfn` A/B for this candidate
- do not commit/push; no skyline was found
- return to the larger non-GP mainline bottleneck

Next directions:

- first:
  - leave the GP shared-path observability in tree as a gate
- second:
  - move back to the larger non-GP bottleneck instead of continuing GP kernel work
- third:
  - only reopen GP if there is a concrete plan to remove transition-generator build cost itself rather than shaving another few milliseconds off `gp_first_proj_ms`

## 2026-03-07 11:50: restored `TBPTT stream merge auto=0` default; finalize compile still rejected on the stable base

Goal:

- stop the `gp` mainline
- return to the larger non-GP bottleneck
- first restore the documented batch128 skyline base before evaluating any new policy-side candidate

Observed regression:

- the code default in `fit_model.py` had drifted to:
  - `TICL_POLICY_TBPTT_STREAM_MERGE_AUTO=1`
- but the maintained skyline notes already required:
  - `TICL_POLICY_TBPTT_STREAM_MERGE_AUTO=0`
- with current code and fixed-seed batch128 fail-fast, the drifted default now OOMs before a usable phase line:
  - log: `20260307_seeded_batch128_finalizecompile_off_v2.log`

Stability probe:

- forcing `TICL_POLICY_TBPTT_STREAM_MERGE_AUTO=0` restores the full batch128 line:
  - log: `20260307_seeded_batch128_mergeauto0_probe.log`
  - `batch_wall_excl_compile_s=125.318`
  - `rollout_s=78.633`
  - `backward_s=46.685`
  - peak alloc/reserved `35.37 / 45.99 GiB`

Decision on the default:

- restore `TICL_POLICY_TBPTT_STREAM_MERGE_AUTO=0` in `fit_model.py`
- this is a skyline stability fix, not a new skyline

Non-GP candidate tested on the restored stable base:

- candidate:
  - policy finalize-only compile
  - `backend=inductor`
  - `mode=max-autotune-no-cudagraphs`
  - `fullgraph=True`
  - `dynamic=False`
- fixed base:
  - `TICL_POLICY_TBPTT_STREAM_MERGE_AUTO=0`
  - all GP fused flags forced back to `0`

Logs:

- stable eager base:
  - `20260307_seeded_batch128_mergeauto0_probe.log`
- finalize-compile first run:
  - `20260307_seeded_batch128_finalizecompile_on_mergeauto0.log`
- finalize-compile warm rerun:
  - `20260307_seeded_batch128_finalizecompile_on_mergeauto0_warm.log`

Metrics:

- eager stable base:
  - `batch_wall_excl_compile_s=125.318`
- finalize compile, first run:
  - `batch_wall_excl_compile_s=150.582`
  - `batch_wall_incl_compile_s=180.283`
  - `compile_warmup_s=29.701`
  - `compile_counter_recompiles=0`
  - `compile_counter_graph_breaks=0`
  - `compile_counter_unique_graphs=2`
- finalize compile, warm rerun:
  - `batch_wall_excl_compile_s=136.543`
  - `batch_wall_incl_compile_s=142.735`
  - `compile_warmup_s=6.192`
  - `compile_counter_recompiles=0`
  - `compile_counter_graph_breaks=0`
  - `compile_counter_unique_graphs=2`

Outcome:

- no new skyline
- compile still loses badly even after cache warmup:
  - first run: `125.318 -> 150.582` (`+20.16%`)
  - warm rerun: `125.318 -> 136.543` (`+8.96%`)
- this was not a recompile problem:
  - `recompiles=0`
  - `graph_breaks=0`
- but it still carried heavy in-batch inductor autotune/benchmark activity:
  - first run `delta_total_abs=1386`
  - warm rerun `delta_total_abs=906`

Interpretation:

- the useful fix from this round is restoring the stable `merge_auto=0` base
- the finalize-only compile path is not the right next step for this training workload
- it is not enough to place compile on a small local function if the resulting inductor autotune cost still lands inside the measured batch

Decision:

- keep `TICL_POLICY_TBPTT_STREAM_MERGE_AUTO=0` as the default again
- do not enable finalize compile by default
- do not commit/push; no new skyline was found

Next directions:

- first:
  - continue on the non-GP policy mainline from the restored stable base
- second:
  - avoid further compile-side exploration unless the autotune work itself can be fully pre-eliminated before batch timing
- third:
  - return to eager hot-path work inside policy transformer finalize / FFN / out-proj

## 2026-03-07 eager finalize 2D zero-dropout/post-norm GELU fastpath

Scope:

- stayed on the non-GP policy mainline
- targeted eager `policy transformer finalize / FFN / out-proj`
- implemented a strict-equivalence 2D finalize specialization for the dominant default setting:
  - single-token finalize
  - zero dropout
  - post-norm
  - GELU

Code:

- added the specialized eager path in `ticl/models/layer.py`
- added an env gate:
  - `TICL_POLICY_FINALIZE_2D_ZERO_DROPOUT_POSTNORM_GELU_FASTPATH`
- added a forward-policy-step output/grad equivalence test in:
  - `ticl/tests/models/test_tabpfn_kv_cache.py`
- added startup logging in `ticl/train.py`

Important safety outcome:

- the specialization is semantically correct under the dedicated test
- but it is not safe to leave enabled by default on the current batch-128 skyline
- default was therefore restored to `0` in:
  - `ticl/models/layer.py`
  - `ticl/fit_model.py`
  - `ticl/train.py` startup print default

Validation:

- `py_compile` passed
- `pytest -q ticl/tests/models/test_tabpfn_kv_cache.py -k "finalize_compile_preserves_semantics or finalize_default_eager_fastpath_preserves_semantics"` passed

Logs:

- `off`:
  - `20260307_seeded_batch128_finalize2d_default_off.log`
- `on`:
  - `20260307_seeded_batch128_finalize2d_default_on.log`

Observed results on the default `python -m ticl.fit_model rlpfn` path with fixed seed and `TICL_POLICY_TBPTT_STREAM_MERGE_AUTO=0`:

- `off` produced three phase lines:
  - batch 0: `batch_wall_excl_compile_s=104.716`
  - batch 1: `batch_wall_excl_compile_s=95.273`
  - batch 2: `batch_wall_excl_compile_s=98.748`
  - mean over the observed three batches: `99.579`
- `on` failed before the first phase line:
  - fail-fast OOM in envgen `_batch_affine`
  - attempted allocation: `80.00 MiB`
  - free GPU memory at failure: `83.44 MiB`

Interpretation:

- this candidate is not a skyline
- more importantly, it reduces memory headroom enough to break the current batch-128 fail-fast skyline before batch 0 completes
- the eager finalize hot path can still be studied, but any further candidate must be screened for memory-headroom regression immediately, not just batch wall

Decision:

- keep the code and observability
- keep the specialization default disabled
- no commit/push; no new skyline was found

Next directions:

- return to the larger non-GP eager hotspot, but avoid candidates that increase activation/live-buffer pressure around the existing envgen memory limit
- if finalize is revisited again, first measure memory headroom on batch 0 before spending time on wall-only A/B

## 2026-03-07 finalize 2D profile split fix and attn->finalize 2D candidate

Scope:

- stayed on the non-GP mainline
- first fixed a profile blind spot in the single-token 2D finalize path
- then tested an `attn -> finalize` 2D fastpath candidate
- removed the candidate after fixed-seed A/B proved it was a false positive

Code kept:

- 2D finalize profile split is now observable in `ticl/models/layer.py`
  for single-token policy-step path:
  - `finalize_attn_outproj_wall_s`
  - `finalize_ffn_wall_s`
- startup defaults remain unchanged for optimization flags:
  - no new fastpath is enabled by default

Validation:

- `py_compile` passed
- `pytest -q ticl/tests/models/test_tabpfn_kv_cache.py -k "finalize_compile_preserves_semantics or finalize_default_eager_fastpath_preserves_semantics"` passed

New observability result:

- after the profile split fix, the dominant eager policy-step transformer breakdown on the stable base is:
  - `policy_step_transformer_share=0.926`
  - `policy_step_tf_layer_attnff_share=0.736`
  - `policy_step_tf_layer_finalize_share=0.510`
  - `policy_step_tf_layer_finalize_attn_outproj_share=0.166`
  - `policy_step_tf_layer_finalize_ffn_share=0.281`
- so within the current single-token eager transformer path:
  - FFN is heavier than finalize out-proj
  - but out-proj is still a meaningful fraction of layer time

Logs:

- stable base profile gate:
  - `20260307_seeded_batch128_attnff2d_profile_off.log`
- candidate profile gate:
  - `20260307_seeded_batch128_attnff2d_profile_on.log`
- stable base fixed-seed batch-0 A/B:
  - `20260307_seeded_batch128_attnff2d_off.log`
- candidate fixed-seed batch-0 A/B:
  - `20260307_seeded_batch128_attnff2d_on.log`

Candidate tested:

- single-token `attn -> finalize` 2D fastpath
- goal:
  - reduce per-layer python/dispatch overhead without changing math
  - avoid the live-buffer increase that broke the previous finalize specialization

Observed profile gate:

- looked positive under profile instrumentation:
  - `batch_wall_excl_compile_s: 110.471 -> 98.560` (`-10.8%`)

Observed real fixed-seed A/B (`batch 0`, no profile):

- stable base:
  - `batch_wall_excl_compile_s=99.652`
- candidate:
  - `batch_wall_excl_compile_s=110.342`
- real result:
  - `99.652 -> 110.342` (`+10.7%`, worse)

Interpretation:

- this candidate was a profile-induced false positive
- it improved the instrumented path, not the real training path
- the useful outcome from this round is the repaired finalize substage observability, not the candidate itself

Decision:

- keep the finalize 2D observability fix
- remove the `attn->finalize 2D` candidate path
- no commit/push; no new skyline was found

Next directions:

- continue on the non-GP eager mainline with the new finalize substage observability available
- do not trust profile-gate wins unless they survive a no-profile fixed-seed A/B
- prioritize candidates that target the real FFN/out-proj dominant work, not just python-side instrumented overhead

## 2026-03-07 finalize substage split and work-path gating

Scope:

- stayed on the non-GP mainline
- continued to target the real eager `finalize_ffn_share` / `finalize_attn_outproj_share` hotspot
- did not keep any profile-only fastpath candidate
- added finer-grained 2D finalize observability before committing to a new kernel path

Code kept:

- `ticl/models/layer.py`
  now records these single-token finalize substages:
  - `finalize_attn_outproj_linear_wall_s`
  - `finalize_attn_outproj_norm_wall_s`
  - `finalize_ffn_linear1_act_wall_s`
  - `finalize_ffn_linear2_residual_norm_wall_s`
  - `finalize_ffn_linear2_wall_s`
  - `finalize_ffn_residual_norm_wall_s`
- `ticl/models/tabpfn.py`
  propagates the new per-layer finalize substages through policy-step profiling
- `ticl/train.py`
  aggregates and prints the new `policy_step_tf_layer_finalize_*_share` fields on the phase line

Validation:

- `py_compile` passed
- `pytest -q ticl/tests/models/test_tabpfn_kv_cache.py -k "finalize_compile_preserves_semantics or finalize_default_eager_fastpath_preserves_semantics"` passed

Stable-base profile logs:

- `20260307_seeded_batch128_finalize_subsplit_profile.log`
- `20260307_seeded_batch128_finalize_subsplit_v2_profile.log`

Observed stable-base batch-0 breakdown (`20260307_seeded_batch128_finalize_subsplit_v2_profile.log`):

- `batch_wall_excl_compile_s=95.930`
- `policy_step_tf_layer_finalize_share=0.506`
- `policy_step_tf_layer_finalize_attn_outproj_share=0.182`
- `policy_step_tf_layer_finalize_attn_outproj_linear_share=0.122`
- `policy_step_tf_layer_finalize_attn_outproj_norm_share=0.060`
- `policy_step_tf_layer_finalize_ffn_share=0.276`
- `policy_step_tf_layer_finalize_ffn_linear1_act_share=0.124`
- `policy_step_tf_layer_finalize_ffn_linear2_residual_norm_share=0.152`
- `policy_step_tf_layer_finalize_ffn_linear2_share=0.068`
- `policy_step_tf_layer_finalize_ffn_residual_norm_share=0.084`

Interpretation:

- inside the current eager single-token finalize path, the dominant substage is now clearly:
  - `ffn linear2 + residual_norm` (`0.152` of transformer-layer total)
- after splitting that block once more, the larger part is:
  - `ffn residual_norm` (`0.084`)
  over
  - `ffn linear2` (`0.068`)
- on the out-proj side, the larger part is:
  - `attn out-proj linear` (`0.122`)
  over
  - `attn out-proj norm` (`0.060`)

Work-path candidates explicitly gated out this round:

- `addmm` residual fusion microbench:
  - log: `20260307_finalize_addmm_microbench.txt`
  - `base=0.0017034296`
  - `addmm_residual=0.0023329711`
  - result: slower, not worth integrating
- Triton fused `add + layer_norm` prototype with torch backward:
  - log: `20260307_finalize_fused_add_layernorm_microbench.txt`
  - `base=0.0005358312`
  - `fused=0.0015727561`
  - result: much slower overall at `E=512`, not worth integrating in this form

Decision:

- keep the finer-grained finalize observability
- do not introduce a new finalize kernel candidate from this round
- no commit/push; no new skyline was found

Next directions:

- if finalize is continued on the mainline, the next real candidate should target:
  - the `ffn residual_norm` block first
  or
  - a stricter fused MLP path that removes more than one kernel boundary at once
- do not revisit `addmm` residual fusion or forward-only Triton add-norm prototypes unless the backward path is also fused and re-gated

## 2026-03-07 high-level rollout/transition execution experiments: no new skyline

Kept changes:
- Added opt-in `TICL_POLICY_TRANSITION_ONLY_ENV_BUILD` to let family-coarse env build skip `x/y/policy` generators when fused transition is actually usable.
- Added rollout observability for transition-only build hits:
  - `rollout_transition_only_build_group_count`
  - `rollout_transition_only_skipped_generator_count`
- Added experimental `balancedK` transition inner grouping (`balanced2`, etc.): per-family sort by estimated transition work, then split into a small number of contiguous buckets.
- Added `TICL_POLICY_TRANSITION_STREAM_FUSION_MAX_GROUPS` (default `2`) so multi-group bucketing does not explode into high-concurrency fake parallelism by default.
- Restored safe default semantics for policy rollout: skipping unused `policy_generator` build is now behind explicit opt-in `TICL_POLICY_SKIP_UNUSED_POLICY_GENERATOR_BUILD=1` instead of always-on, because it changes RNG-consumption order.

A/B 1: transition-only env build
- fixed-seed (`--seed-everything True`), `batch=128`, `tbptt=64`, `merge_auto=0`
- off:
  - `20260307_seeded_batch128_transitiononlybuild_off.log`
  - `batch_wall_excl_compile_s=121.706`
- on:
  - `20260307_seeded_batch128_transitiononlybuild_on.log`
  - `batch_wall_excl_compile_s=124.317`
- result:
  - `+2.15%` slower on true KPI
- profile gate also agreed:
  - off: `20260307_seeded_batch128_transitiononlybuild_profile_off.log`
    - `rollout_transition_wall_ms=8777.56`
    - `rollout_transition_only_build_groups=0`
    - `rollout_transition_only_skipped_generators=2` (only explicit policy-generator skip in that experiment)
  - on: `20260307_seeded_batch128_transitiononlybuild_profile_on.log`
    - `rollout_transition_wall_ms=10551.79`
    - `rollout_transition_only_build_groups=2`
    - `rollout_transition_only_skipped_generators=6`
- conclusion:
  - removing those generator builds does not help this skyline; setup/build is not the dominant limiter in this form.

A/B 2: balanced family transition bucketing
- goal: raise `transition_work_fill_ratio` without exploding to dozens of groups like `structure`
- candidate: `TICL_POLICY_TRANSITION_INNER_GROUPING=balanced2`
- result:
  - `20260307_seeded_batch128_transition_balanced2_on.log`
  - `20260307_seeded_batch128_transition_balanced2_on_v2.log`
  - `20260307_seeded_batch128_transition_balanced2_on_v3.log`
  - all fail-fast OOM on batch 0 under the current `batch=128`, `tbptt=64` skyline
- the OOM moved between:
  - transition envgen (`_batch_affine` / SCM hidden path)
  - policy paged-attn flash-prefix path (`tail_v.clone()`)
- conclusion:
  - the main issue is no longer just padding.
  - changing transition grouping also perturbs the whole-batch execution mode and memory pressure seen by policy forward.
  - this is a strong signal that batch order / transition order coupling is itself a high-level bottleneck and risk source.

Current takeaways:
- No new skyline in this round, so no commit/push.
- `transition-only env build` stays opt-in and off by default.
- `balanced2` stays experimental and off by default.
- `transition stream fusion max groups=2` is kept as a safe guardrail for multi-group experiments; it does not affect the current default `family` path (which uses 2 groups).
- The next serious execution-mode target should not be another micro-kernel. It should be one of:
  - decoupling transition grouping order from policy batch order,
  - or improving TBPTT multi-backward merge logic using the correct decision point and memory signal.

## 2026-03-07 balanced2 clean rerun after killing stale fit_model process: new skyline

Root cause correction:
- A previously leaked `python -m ticl.fit_model rlpfn --seed-everything True` process (`pid=2551335`) was still holding about `12-26 GiB` of GPU memory during later experiments.
- After killing that stale process and its parent shell, the earlier `balanced2` fail-fast OOM no longer reproduced.

Clean fixed-seed A/B:
- settings:
  - `batch=128`
  - `tbptt=64`
  - `merge_auto=0`
  - no profile
- family clean reruns:
  - `20260307_seeded_batch128_transition_family_clean.log`
    - `batch_wall_excl_compile_s=124.671`
  - `20260307_seeded_batch128_transition_family_clean_r2.log`
    - `batch_wall_excl_compile_s=122.485`
- balanced2 clean reruns:
  - `20260307_seeded_batch128_transition_balanced2_clean.log`
    - `batch_wall_excl_compile_s=118.934`
  - `20260307_seeded_batch128_transition_balanced2_clean_r2.log`
    - `batch_wall_excl_compile_s=121.498`

Measured effect:
- family mean:
  - `(124.671 + 122.485) / 2 = 123.578`
- balanced2 mean:
  - `(118.934 + 121.498) / 2 = 120.216`
- result:
  - `-2.72%` batch wall on the true KPI after removing the leaked-process contamination

Interpretation:
- The earlier `balanced2` OOMs were not a real property of the candidate under the intended skyline conditions.
- With clean GPU state, `balanced2` keeps the batch runnable and reduces rollout-side wall enough to beat the restored `family` baseline.
- This is not a huge gain, but it is a real positive skyline on the current execution-mode mainline, so `TICL_POLICY_TRANSITION_INNER_GROUPING=balanced2` is now promoted to the default for `python -m ticl.fit_model rlpfn`.

## 2026-03-07 documented-only default cleanup: removed hidden non-summary defaults

Goal:

- stop relying on performance-affecting defaults that were never promoted into
  the maintained skyline notes
- keep `python -m ticl.fit_model rlpfn` aligned with documented mainline knobs
  only, even if an undocumented path happened to benchmark slightly better in
  an ad hoc local probe

Code:

- `ticl/fit_model.py`
  - removed undocumented default pinning for:
    - `TICL_POLICY_INPLACE_FLASH_PREFIX`
    - `TICL_POLICY_FORCE_FLASH_SINGLE_PAGE`
- `ticl/models/layer.py`
  - restored `TICL_POLICY_INPLACE_FLASH_PREFIX` implicit default from `1` to
    `0`
- `ticl/train.py`
  - restored startup print default for `Policy inplace flash-prefix` to `False`
- `ticl/models/tabpfn.py`
  - restored `TICL_POLICY_ASSUME_FINITE_INPUTS` implicit default from `1` to
    `0`

Rationale:

- these knobs had no maintained skyline entry in `SUMMARY.md`
- leaving them enabled by implicit code default would make later A/B results
  depend on non-documented behavior, which is incompatible with maintaining a
  real skyline

Validation:

- `python -m py_compile ticl/fit_model.py ticl/models/layer.py ticl/models/tabpfn.py ticl/train.py`
  passed
- targeted semantics regression checks passed:
  - `pytest -q ticl/tests/models/test_tabpfn_kv_cache.py -k "tbptt_detach_prefix_compaction_preserves_paged_cache_semantics or forward_policy_step_finalize_default_eager_fastpath_preserves_semantics"`
- fixed-seed batch-128 baseline with documented defaults only:
  - log: `20260307_seeded_batch128_doc_only_defaults.log`
  - startup confirms:
    - `Policy inplace flash-prefix: False`
    - `Policy transition inner grouping: balanced2`
    - `Policy transition async commit in-stream: True (mode=auto)`
    - `Policy KV cache in-place append: False`
  - phase line:
    - `batch_wall_excl_compile_s=119.660`
    - `batch_wall_excl_compile_per_batch_item_s=0.934841`
    - `rollout_s=74.581`
    - `backward_s=45.079`

Notes:

- this is a skyline hygiene fix, not a new throughput skyline
- run status was still `grad_norm_nonfinite`, but this pass intentionally
  judged only throughput under identical fixed-seed single-batch conditions
- future throughput work should compare against this documented-only base, not
  against hidden defaults that were never accepted into the maintained notes

## 2026-03-07 batch128 follow-up: TF32 and cache-container-reuse remain non-pinned on the documented base

Goal:

- continue on the documented `batch=128`, `tbptt=64` base
- check whether any already-implemented low-level positives should be promoted
  from implicit code defaults into explicit skyline defaults
- avoid pinning small/noisy candidates that would only add more default drift

Validation runs (same fixed-seed command, no profile):

- documented base rerun:
  - `20260307_seeded_batch128_doc_only_defaults_r2.log`
  - `batch_wall_excl_compile_s=123.057`
  - `batch_wall_excl_compile_per_batch_item_s=0.961380`
- `TICL_POLICY_TF32=0`:
  - `20260307_seeded_batch128_tf32_off_docbase.log`
  - `batch_wall_excl_compile_s=122.704`
  - `batch_wall_excl_compile_per_batch_item_s=0.958624`
- `TICL_POLICY_CACHE_CONTAINER_REUSE=0`:
  - `20260307_seeded_batch128_cachereuse_off_docbase.log`
  - `batch_wall_excl_compile_s=120.704`
  - `batch_wall_excl_compile_per_batch_item_s=0.943003`
- all three runs kept the same observed peak alloc/reserved:
  - `12.27 / 13.49 GiB`

Interpretation:

- current single-batch variance on this machine is large enough that these two
  candidates are not strong enough to promote as new explicit skyline pins on
  the batch-128 base
- `TF32` remains acceptable as a documented code-default optimization, but this
  pass did not produce a batch-128-specific hard A/B strong enough to add a new
  `fit_model.py` default pin
- `cache_container_reuse` remains even less conclusive on the current base, so
  it also stays unpinned

Code-quality/observability follow-up:

- `ticl/train.py`
  - startup now always prints `Policy TF32 matmul: ...`
  - startup now prints `Policy cache container reuse: ...`
- this prevents future A/B logs from silently depending on hidden code defaults
  for these two knobs

Decision:

- no new throughput skyline in this round
- keep the documented batch-128 base unchanged
- use the new startup observability to decide later whether `TF32` or
  `cache_container_reuse` should be explicitly pinned or explicitly removed

## 2026-03-07 documented positive-default closure: explicitly pin TF32 and cache-container-reuse for `rlpfn`

Goal:

- finish the default audit in descending documented-gain order
- remove the last two documented positive paths that were still relying on
  implicit code defaults instead of explicit `rlpfn` skyline pinning

Context:

- after the larger documented gains had already been pinned into
  `ticl/fit_model.py`, the remaining implemented/documented positives not yet
  explicitly pinned were:
  - `TICL_POLICY_TF32=1`
  - `TICL_POLICY_CACHE_CONTAINER_REUSE=1`

Fixed-seed checks on the documented batch-128 base:

- `TF32=0`:
  - `20260307_seeded_batch128_tf32_off_docbase.log`
  - `batch_wall_excl_compile_s=122.704`
- `CACHE_CONTAINER_REUSE=0`:
  - `20260307_seeded_batch128_cachereuse_off_docbase.log`
  - `batch_wall_excl_compile_s=120.704`
- `TF32=0 + CACHE_CONTAINER_REUSE=0`:
  - `20260307_seeded_batch128_tf32off_cachereuseoff_docbase.log`
  - `batch_wall_excl_compile_s=123.114`
  - startup confirms:
    - `Policy TF32 matmul: False`
    - `Policy cache container reuse: False`

Interpretation:

- disabling both together does not improve the documented batch-128 base
- neither flag produced strong enough batch-128 evidence to claim a new skyline
  by itself, but there is also no fixed-seed evidence that they should now be
  removed from the maintained path
- the remaining issue was therefore code-quality / reproducibility, not raw
  kernel throughput

Code:

- `ticl/fit_model.py`
  - explicitly pins for `python -m ticl.fit_model rlpfn`:
    - `TICL_POLICY_TF32=1`
    - `TICL_POLICY_CACHE_CONTAINER_REUSE=1`

Decision:

- this is a default-closure / anti-drift fix, not a new throughput skyline
- the documented batch-128 mainline now no longer depends on hidden code
  defaults for these two retained positive paths

## 2026-03-07 code-quality cleanup: remove dead debug/comment code and dedupe env-flag parsing

Goal:

- improve code structure without changing semantics, throughput, or memory
- remove obviously dead code that had accumulated around the skyline work
- reduce duplicated env-flag parsing in startup observability so future default
  changes are less likely to drift

Code:

- `ticl/fit_model.py`
  - removed stale commented-out allocator/debug lines
  - removed unused `root_dir`, `pandas`, and `pdb`
  - removed a long dead commented-out wandb run-dedup block
- `ticl/train.py`
  - removed unused `pdb`
  - added `_env_flag_enabled(...)` and used it for TF32 / startup policy-flag
    observability instead of repeated ad hoc string parsing

Validation:

- `python -m py_compile ticl/fit_model.py ticl/train.py` passed
- targeted semantics regression checks passed:
  - `pytest -q ticl/tests/models/test_tabpfn_kv_cache.py -k "tbptt_detach_prefix_compaction_preserves_paged_cache_semantics or forward_policy_step_finalize_default_eager_fastpath_preserves_semantics"`
- fixed-seed batch-128 cleanup baseline:
  - `20260307_seeded_batch128_cleanup_baseline.log`
  - startup confirms:
    - `Policy TF32 matmul: True`
    - `Policy cache container reuse: True`
  - phase line:
    - `batch_wall_excl_compile_s=119.698`
    - `batch_wall_excl_compile_per_batch_item_s=0.935144`
  - peak alloc/reserved:
    - `12.27 / 13.49 GiB`

Interpretation:

- the cleanup pass does not introduce a visible throughput or memory regression
  relative to the documented batch-128 baseline
- this pass is purely code-quality / anti-drift work, not a new skyline

## 2026-03-07 batch-256 throughput scaling check on the documented mainline

Goal:

- test whether the current documented mainline remains runnable at
  `batch_size=256` with `TBPTT=64`
- record normalized throughput (`wall / batch_size`) and memory usage on the
  same fixed-seed single-batch benchmark

Command notes:

- fixed seed
- `batch_size=256`
- `pg_tbptt_window=64`
- `pg_env_replay_steps=1`
- fail-fast enabled
- no profiler / GPU observer

Result:

- log:
  - `20260307_seeded_batch256_mainline.log`
- startup confirms:
  - `Policy TF32 matmul: True`
  - `Policy cache container reuse: True`
- phase line:
  - `batch_wall_excl_compile_s=126.545`
  - `batch_wall_excl_compile_per_batch_item_s=0.494318`
  - `rollout_s=79.453`
  - `backward_s=47.092`
- epoch wallclock:
  - `79.60s`
  - normalized `79.60 / 256 = 0.310938 s`
- peak alloc/reserved:
  - `24.32 / 29.41 GiB`

Interpretation:

- the current documented mainline is stable at `batch=256` under this
  throughput-only benchmark (no fail-fast OOM)
- normalized throughput improves materially versus the documented `batch=128`
  mainline regime
- this is a scaling observation, not a replacement of the maintained batch-128
  skyline

## 2026-03-07 batch-256 default adoption + falsified micro-probe cleanup

Goal:

- make the maintained `rlpfn` mainline use the proven `batch=256` regime by
  default
- remove already-falsified micro-probe branches that were still increasing code
  complexity around the rollout hot path
- keep semantics, throughput, and memory behavior aligned with the retained
  mainline path

Code:

- `ticl/model_configs.py`
  - `get_rlpfn_default_config()` now sets `dataloader.batch_size = 256`
- removed falsified / non-maintained branches:
  - `TICL_POLICY_STATE_POSTPROCESS_INPLACE`
  - `TICL_POLICY_REWARD_MASK_BUFFER_REUSE`
  - `TICL_POLICY_INPLACE_FLASH_PREFIX`
  - `TICL_POLICY_ASSUME_FINITE_INPUTS`
- corresponding startup/fit-model guard entries were removed so these old probe
  knobs no longer pollute later A/B work
- added config regression coverage:
  - `ticl/tests/test_rlpfn_split_encoder.py` now asserts `batch_size == 256`

Rationale:

- these removed paths were already documented as rejected or outside the
  maintained skyline
- keeping them in tree only added inactive branching and extra env-surface
  area, making later diagnosis noisier
- the kept code path after cleanup is exactly the previous default-off/mainline
  behavior

Validation:

- `python -m py_compile ticl/model_configs.py ticl/fit_model.py ticl/train.py ticl/models/layer.py ticl/models/tabpfn.py ticl/priors/environment_prior.py`
  passed
- targeted semantics tests passed:
  - `pytest -q ticl/tests/priors/test_environment_prior.py -k "state_highway_postprocess_toggle_semantics or state_highway_enabled_rollout_smoke"`
  - `pytest -q ticl/tests/models/test_tabpfn_kv_cache.py -k "tbptt_detach_prefix_compaction_preserves_paged_cache_semantics or forward_policy_step_finalize_default_eager_fastpath_preserves_semantics"`
- default batch-256 fixed-seed mainline check (no `-b` CLI override):
  - log: `20260307_seeded_batch256_default_after_cleanup.log`
  - phase line:
    - `batch_size=256`
    - `batch_wall_excl_compile_s=130.202`
    - `batch_wall_excl_compile_per_batch_item_s=0.508601`
    - `rollout_s=82.208`
    - `backward_s=47.994`
  - epoch wallclock:
    - `82.35s`
    - normalized `82.35 / 256 = 0.321680 s`
  - peak alloc/reserved:
    - `24.32 / 29.41 GiB`

Interpretation:

- the default batch-size switch to 256 is effective and remains runnable on the
  documented mainline
- after removing the falsified micro-probes, throughput and memory stay in the
  same regime as the earlier batch-256 scaling check
- this pass improves code quality and reduces future profiling noise without
  introducing a new negative memory or throughput symptom

## 2026-03-07 batch-256 mainline hygiene: learning-rate x10 + falsified transition-runtime cleanup

Goal:

- keep the maintained `batch=256 / TBPTT=64 / replay_step=1 / fail-fast`
  mainline
- reflect the much larger maintained physical batch in the default optimizer
  step size instead of inheriting the tiny legacy LR
- remove already-falsified transition runtime branches that no longer belong on
  the maintained path
- confirm that throughput and VRAM stay on the same skyline regime after the
  cleanup

Audit result before code changes:

- no additional documented positive environment/kernel knobs remained to be
  enabled for `python -m ticl.fit_model rlpfn`; the retained positive set was
  already fully pinned in `ticl/fit_model.py`
- the remaining cleanup opportunity was therefore code-quality/runtime-branch
  reduction, not another new optimization flag

Code:

- `ticl/model_configs.py`
  - `get_rlpfn_default_config()` now sets:
    - `optimizer.learning_rate = 3e-4` (10x over the previous `3e-5`)
- `ticl/priors/environment_prior.py`
  - removed the runtime flag plumbing for:
    - `TICL_POLICY_FUSED_TRANSITION_STABLE_INPUT_SLOTS`
    - `TICL_POLICY_FUSED_TRANSITION_PAIRED`
  - removed the rollout hot-path stable-dual-slot branch and its dead stats
    propagation
  - removed the live transition-builder selection branches for paired
    transition specialization
  - removed now-unused transition metadata fields that existed only for those
    branches
- `ticl/train.py`
  - removed startup logging and batch/stage/wandb aggregation for the deleted
    stable-dual-slot counters
- `ticl/fit_model.py`
  - removed the no-longer-valid skyline guard env pins for the deleted flags
- `ticl/tests/priors/test_environment_prior.py`
  - removed the paired-transition regression that only exercised the deleted
    runtime branch
- `ticl/tests/test_rlpfn_split_encoder.py`
  - now asserts the maintained `rlpfn` default learning rate is `3e-4`

Validation:

- `python -m py_compile ticl/model_configs.py ticl/fit_model.py ticl/train.py ticl/priors/environment_prior.py ticl/tests/priors/test_environment_prior.py ticl/tests/test_rlpfn_split_encoder.py`
  passed
- targeted semantics tests passed:
  - `conda run -n rlpfn python -m pytest -q ticl/tests/priors/test_environment_prior.py -k "fused_transition_dual_packed_matches_cat_semantics or scm_hidden_fused_transition_matches_legacy_dual_semantics or envgen_checkpoint_preserves_rollout_semantics or ragged_affine_matches_dense_hetero_batch_semantics or mixed_family_gp_projection_profile_survives_async_rollout"`
  - `conda run -n rlpfn python -m pytest -q ticl/tests/test_rlpfn_split_encoder.py`
- fixed-seed batch-256 mainline check:
  - log: `20260307_seeded_batch256_lr10x_cleanup.log`
  - startup confirms:
    - `Policy TBPTT window: 64`
    - `Policy env replay steps: 1`
    - `Policy transition inner grouping: balanced2`
    - `Policy fused transition SCM hidden-fused: True`
    - `Policy cache container reuse: True`
  - warmup note confirms new default LR:
    - `base_lr=3.000e-04`
  - phase line:
    - `batch_wall_excl_compile_s=125.739`
    - `batch_wall_excl_compile_per_batch_item_s=0.491167`
    - `rollout_s=79.403`
    - `backward_s=46.336`
  - peak alloc/reserved:
    - `24.32 / 29.41 GiB`

Interpretation:

- this pass does not introduce a new kernel skyline knob; the documented
  positive-default audit for batch-256 is effectively closed
- raising the maintained default LR to `3e-4` does not perturb the throughput or
  memory skyline under the fixed-seed single-batch benchmark
- removing the falsified `paired/stable-slots` runtime branches reduces later
  profiling noise while keeping the batch-256 mainline on the same performance
  and VRAM regime
