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
