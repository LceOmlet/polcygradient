# Per-Feature V2.5 Memory Investigation

Last updated: 2026-03-19

## Goal

Localize the true resident memory plateau in the default `per_feature_v25` training path without changing training semantics.

Primary workload under investigation:

```bash
cd /home/chen/RLPFN/ticl-milestone-alpha-grad
conda run --no-capture-output -n rlpfn \
  python -m ticl.fit_model rlpfn \
  --backbone-variant per_feature_v25 \
  -n 1 -E 1 --stop-after-epochs 1 \
  --validate false --progress-bar false --seed-everything true
```

Effective default batch at the time of writing:

- `batch_size=256`
- `n_samples=1024`
- `tbptt=32`
- `alpha_grad`
- `paged KV`

## Attribution Rules

These rules exist to avoid repeating earlier mistakes.

1. The final OOM stack identifies the trigger, not the dominant resident live set.
2. A phase is only treated as the main cause if its allocator plateau is corroborated by full-batch phase-local measurements.
3. Small-workload profiling may narrow the search space, but it is not sufficient for full-batch attribution.
4. No semantic optimization should be promoted based only on the trigger moving later in the step.

## Confirmed Findings

### 1. TBPTT sink is not the dominant plateau

Full-batch phase snapshots showed that `tbptt_sink_pre/post` and `tbptt_detach_pre/post` sit below the late-step plateau.

Representative default-path observation:

- `tbptt_boundary_post`: about `32.12 GiB`
- `tbptt_detach_pre/post`: about `43.21 GiB`
- `tbptt_sink_pre/post`: about `43.21 GiB`

This means TBPTT boundary/detach/sink are not the main source of the `45+ GiB` live set.

### 2. The default-path plateau is late in rollout

On the default path:

- `policy_post / transition_pre / transition_input_post`: about `45.55 GiB`
- `transition_generator_post / transition_post`: about `45.66 GiB`
- next-step `policy_pre`: about `45.66 GiB`

Interpretation:

- The plateau exists before the next policy step begins.
- The last increase happens between transition input preparation and post-generator/post-commit state.
- The next policy step inherits that elevated live set.

### 3. The generator-to-next-policy handoff is now split

With synchronized phase snapshots on the default path:

- `policy_post / transition_pre / transition_input_post`: about `45.02 GiB`
- `transition_generator_post`: about `45.12 GiB`
- `transition_commit_post`: about `45.12 GiB`
- `transition_post`: about `45.12 GiB`
- `step_rollover_post`: about `45.12 GiB`
- `step_tail_post`: about `45.12 GiB`
- next-step `policy_pre`: about `45.12 GiB`

Interpretation:

- the remaining visible step-local increase is small, about `0.10 GiB`
- that increase happens before or at `transition_generator_post`
- after that point, commit/finalize/rollover/tail do not materially raise the plateau further
- therefore the next cut should move inside the transition generator output materialization path, not later bookkeeping

### 4. Generator-local profiling isolates one dominant subsegment

With generator-local synchronized profiling on the default path, the dominant subsegment was:

- `generator_fused_call_sync`

Representative maxima:

- local `peak_delta`: about `115 MiB`
- `retained_to_transition_post`: about `114 MiB`
- `retained_to_step_tail_post`: about `114 MiB`
- `retained_to_next_policy_pre`: about `114.4 MiB`
- `saved-for-backward total`: about `827.4 MiB`
- `saved-for-backward max single tensor`: about `61.5 MiB`

By contrast:

- `generator_unpack_sync`: near zero local peak/retained
- `generator_reward_scale_sync`: near zero local peak/retained
- `generator_commit_sync`: about `1.2 MiB` local peak and sub-`1 MiB` retained

Interpretation:

- the real retained growth is inside the fused transition generator call itself
- not in unpacking, scaling, or commit bookkeeping
- the saved-for-backward footprint inside that fused call is much larger than the visible retained plateau increment

This strongly suggests that the next attribution cut should move inside the fused transition generator implementation.

### 5. Throughput and utilization observations

Two observations are already strong enough to act on:

- On a completing `b=64, n=64` timing run, `rollout_policy_wall_share` was about `0.581`, while transition wall time was smaller in aggregate.
- Inside transition, `rollout_transition_fused_share` was about `0.843`, meaning most transition time is already concentrated in the fused path.

Full-batch external GPU sampling during the default-path run showed low compute utilization:

- active-window `utilization.gpu` average: about `23.16%`
- active-window median: about `24%`
- active-window 90th percentile: about `28%`

Interpretation:

- the current default path is not close to saturating the GPU
- a large part of runtime is likely spent in synchronization, graph retention overhead, or many small/serialized kernels instead of sustained dense compute

### 6. The in-place/flash-merge route did not reduce the plateau

On the `in-place paged KV + flash_merge(32)` route:

- `transition_post / next-step policy_pre`: about `45.35 GiB`
- `policy_post / transition_pre / transition_input_post`: about `45.77 GiB`
- `env_batch_affine_pre`: about `45.86 GiB`

Interpretation:

- The plateau was shifted earlier into the policy side.
- It was not materially reduced.
- Therefore this route should not be treated as a better default based on current evidence.

### 7. Trigger migration is still not attribution

In the synchronized full-batch run, the final failing call moved again, this time to
policy-side flash-prefix merge. The resident plateau evidence above still points to the
same handoff region. This is another reminder that trigger migration alone should not
drive optimization decisions.

## Wrong Turns To Avoid Repeating

### Misattribution of `baddbmm`

Earlier reasoning incorrectly treated `torch.baddbmm` as the bottleneck because it was the final failing call in one route.

That was wrong because:

- the failing allocation was small relative to total live memory
- phase-local measurements later showed that the plateau was already present before `baddbmm`

Correct interpretation:

- `baddbmm` can be the trigger
- it is not automatically the dominant resident allocation source

### Treating trigger migration as success

Moving the OOM from one callsite to another is not enough.

A change is only useful if it lowers the dominant full-batch plateau, not just the final stack frame.

## Current Focus

The next split is the handoff from:

- `transition_generator_post`
- internals of the transition generator output path
- immediate generator output materialization
- next-step `policy_pre`

The purpose is to determine whether the inherited plateau is dominated by:

- transition-generator subgraph intermediates
- generator outputs that stay alive too long
- async stream materialization that becomes visible only when synchronized

## 8. Exact SCM hidden-loop checkpoint: correct semantics, limited plateau gain

An exact non-reentrant checkpoint was implemented for the strict reference SCM hidden loop,
with generator-state replay so that per-row CUDA generators advance identically with and
without checkpointing.

Semantics check:

- output tensors matched
- input gradients matched
- final row-generator states matched exactly

Full-batch comparison on the same codebase, same default command, only toggling
`TICL_POLICY_REFERENCE_SCM_HIDDEN_LOOP_CHECKPOINT=0/1`:

- checkpoint `off`
  - `policy_post / transition_input_post`: about `45.37 GiB`
  - `transition_generator_post / next policy_pre`: about `45.48 GiB`
  - generator-local `saved-for-backward`: about `827.4 MiB`
  - generator-local `ret_next`: about `114.4 MiB`
  - generator-local wall: about `16.88 s`

- checkpoint `on`
  - `policy_post / transition_input_post`: about `45.30 GiB`
  - `transition_generator_post / next policy_pre`: about `45.41 GiB`
  - generator-local `saved-for-backward`: about `107.1 MiB`
  - generator-local `ret_next`: about `113.7 MiB`
  - generator-local wall: about `18.92 s`

Interpretation:

- this checkpoint does exactly what it should for saved activations
- it meaningfully cuts generator-local saved-for-backward footprint
- it only lowers the full-batch plateau by about `70 MiB`
- it does not recover `batch_size=256` completion by itself
- it also makes the fused generator segment slower

Decision:

- keep the implementation available behind `TICL_POLICY_REFERENCE_SCM_HIDDEN_LOOP_CHECKPOINT=1`
- do not treat it as the solved default path, because the main retained plateau is still elsewhere

## 9. Explicit cache is not the 45 GiB plateau

After fixing object-memory profiling for `cuda` vs `cuda:0` device matching, the default
full-batch `tbptt=32` run produced:

- `policy_cache_post`: about `1272 MiB`
- `policy_cache_pre`: about `1272 MiB`
- `policy_cache_detach_post`: about `1152 MiB`
- `tbptt_boundary_cache_replay_leaves`: about `384 MiB`
- `tbptt_boundary_state_payload`: about `0.4 MiB`

At the same time, the allocator plateau remained:

- `policy_post / transition_input_post`: about `45.37 GiB`
- `transition_generator_post / transition_post / next policy_pre`: about `45.48 GiB`

Interpretation:

- explicit paged KV cache is large enough to matter
- but it is still about one to three GiB, not forty-plus GiB
- TBPTT boundary carry objects are also far smaller than the plateau
- therefore the dominant resident set is not the explicit cache object itself

## 10. The dominant resident set is policy-side autograd state that scales with context length

Two full-batch runs now constrain the main cause:

### `tbptt=32`

- plateau: about `45.48 GiB`
- `policy_step` saved non-parameter unique storage: about `1387 MiB`
- explicit `policy_cache_post`: about `1272 MiB`
- after the last completed sink (`step 95`), `tbptt_sink_post` is about `41.51 GiB`
- by the failing step (`105/106`), the plateau has risen by about `3.97 GiB`
- over the same span, explicit cache grows only from about `1152 MiB` to about `1272 MiB`, i.e. about `0.12 GiB`

### `tbptt=16`

- plateau still rises to about `45.15 GiB`
- OOM happens later, around step `271` instead of step `106`
- `policy_step` saved non-parameter unique storage rises to about `3379 MiB`
- after the last completed sink (`step 255`), `tbptt_sink_post` is about `37.07 GiB`
- by the failing step (`271`), the plateau has risen by about `8.08 GiB`

Interpretation:

- reducing TBPTT from `32` to `16` does **not** halve the plateau
- instead, it lets the rollout progress much farther before the same memory budget is exhausted
- during that longer rollout, policy-side saved non-parameter storage grows from about `1.39 GiB` to about `3.38 GiB`
- that growth factor (`~2.44x`) closely tracks the increase in step / effective context length (`271 / 105 ~= 2.58x`)
- the explicit cache growth visible in the `tbptt=32` run is only about `12 MiB / step`
- the allocator plateau growth between `tbptt_sink_post` and OOM is about `397 MiB / step` for `tbptt=32`, and about `505 MiB / step` for `tbptt=16`

This is the strongest evidence so far that the dominant resident set is:

- policy-side autograd saved activations / saved tensors
- whose size grows with per-step context length / cache length
- not the explicit paged KV cache object
- not TBPTT boundary carry objects
- not the transition generator

## 11. Practical attribution status

What is now strongly supported:

- the `45 GiB` plateau is mainly policy-side
- the dominant class is implicit autograd-saved tensors, not explicit cache containers
- the dominant growth driver is effective context length at each policy step
- within the policy path, the only component whose activation footprint naturally scales with current cache/context length is `item_block.forward_step` inside `per_feature_v25`

What is still not fully decomposed:

- which exact policy submodule contributes the largest share of those saved tensors
- whether the largest share inside `item_block.forward_step` is attention-core state or post-attention / FFN state

Therefore the next optimization cut should target:

- policy-side saved activation volume inside `per_feature_v25`
- especially computation whose saved tensors scale with current cache/context length

and should **not** target:

- generator unpack/commit
- TBPTT boundary payload structure
- explicit paged KV object layout alone

## 12. Step-local split: feature block vs item block

I added a policy-step-local split inside `PerFeatureTabPFN` so the OOM report now exposes:

- `pf_feature_ms`
- `pf_item_ms`
- `pf_saved_np_u_last(feature/item)`
- `pf_saved_np_r_last(feature/item)`

On the default full-batch OOM path (`tbptt=32`), the last completed policy step reported:

- `feature` saved non-parameter unique storage: about `12 MiB`
- `item` saved non-parameter unique storage: about `1373.5 MiB`
- `feature` wall time: about `1899 ms`
- `item` wall time: about `7306 ms`

Interpretation:

- `feature_block` is **not** the dominant saved-activation source
- `item_block.forward_step` is the dominant saved-activation source by about two orders of magnitude
- `item_block.forward_step` is also the dominant policy-side wall-time sink

This is the strongest attribution so far, because it separates the two `per_feature_v25` policy branches directly inside the model, on the real full-batch OOM path.

## 13. Why `item_block.forward_step` is the right place structurally

The code path in `TransformerEncoderLayer.forward_step` showed another important detail:

- for dense / immutable KV path, `recompute_attn=True` already wraps `_forward_step_attn_ff(...)` in checkpoint
- for paged KV path, the hot branch went directly through `_forward_step_attn_ff_paged(...)`
- therefore, before any change, the paged training path did **not** honor `recompute_attn`

That makes `item_block.forward_step` the only policy branch that both:

- scales naturally with current cache/context length
- dominates saved activations on the real OOM path
- and was still missing a recompute cut on the paged training branch

## 14. Paged recompute experiment outcome

I implemented an exact, non-reentrant checkpoint around the paged branch and verified its semantics on a focused regression test:

- `perfeature + paged + mutable cache + recompute` matched the no-recompute reference on outputs and parameter gradients

However, the full-batch result was not good enough to keep as default.

### Before paged recompute

- OOM around step `105/106`
- plateau about `45.48 GiB`
- `tbptt_sink_post` at step `95`: about `41.51 GiB`
- `policy_cache_post`: about `1272 MiB`
- `pf_saved_np_u_last(item)`: about `1373.5 MiB`

### With paged recompute enabled

- OOM later, around step `121/122`
- but plateau still reached about `45.65 GiB`
- `tbptt_sink_post` at step `95` dropped to about `36.25 GiB`
- `policy_cache_post` grew to about `1464 MiB`
- `pf_saved_np_u_last(item)`: about `1501.7 MiB`
- wall time increased substantially

Interpretation:

- the paged recompute cut was **directionally helpful** for the same early window (`tbptt_sink_post` dropped by about `5.26 GiB`)
- but it did **not** reduce the final end-of-run plateau enough to complete the batch
- instead it mainly let the rollout advance farther, which in turn allowed the explicit cache/context length to grow more before OOM
- because it was slower and did not solve the actual batch-completion problem, it should **not** be the default path

Practical decision:

- keep paged recompute as an opt-in experiment only
- do **not** rely on it as the main fix
- continue investigating the remaining policy-side resident set on the default path

## 15. Finer default-path profile inside `item_block.forward_step`

I then kept the default path unchanged and only added a finer profile for the real default rollout path.

The OOM summary now exposes:

- `tf_item_ms(cache/attn_core/finalize/ffn_l1)`
- `tf_saved_np_u_last(attnff/finalize)`
- `tf_saved_np_r_last(attnff/finalize)`
- `tf_paged_avg(page_count/valid/prefix)`

On the default full-batch OOM path, the last completed policy step reported:

- `tf_item_ms(cache/attn_core/finalize/ffn_l1)=643.16 / 2185.82 / 3297.08 / 820.38`
- `tf_saved_np_u_last(attnff/finalize)=325.7 MiB / 17.9 MiB`
- `tf_saved_np_r_last(attnff/finalize)=324.2 MiB / 17.9 MiB`
- `tf_paged_avg(page_count/valid/prefix)=1.0 / 53.5 / 38.0`

Interpretation:

- inside the default `item_block` path, `finalize/ffn` is the largest **time** sink
- but `finalize` is **not** the dominant saved-activation source
- the dominant saved tensors within `item_block` live earlier in the broader `attnff` path
- because `attnff_saved` is much larger than `finalize_saved`, the remaining memory suspect is the pre-finalize paged-attention branch, not the FFN tail

This is an important narrowing:

- `feature_block` is not the memory problem
- `finalize/ffn` is mainly a speed problem
- the memory problem within `item_block` is earlier than `finalize`

So the next observation cut, if needed, should separate:

- paged cache append / prefix handling
- pre-finalize paged attention / merge path

and **not** spend more cycles on:

- generator
- TBPTT boundary payload
- finalize FFN alone

## 16. Default-path paged attention split inside `item_block`

I kept the default path intact and split the default paged-attention route further into:

- cache append
- prefix maintenance
- dense/view build
- dispatch by path
- `flash_prefix` internals: prepare, prefix core, tail core, merge

I also kept the full-batch phase/object profiles on and reran the same default command.

### Full-batch OOM plateau remains unchanged

- `policy_post / transition_input_post`: about `45.37 GiB`
- `transition_generator_post / transition_post / next policy_pre`: about `45.48 GiB`
- `tbptt_sink_post` at step `95`: about `41.51 GiB`

So this cut did **not** discover a hidden post-generator spike; the large resident plateau is still already present before the final env step.

### Explicit cache is still not the dominant resident set

The object profile still shows:

- `policy_cache_post`: about `1272 MiB`
- `policy_cache_detach_post`: about `1152 MiB`
- `tbptt_boundary_cache_replay_leaves`: about `384 MiB`

These numbers are much smaller than the `45+ GiB` plateau, so explicit cache tensors are not sufficient to explain the resident set by themselves.

### Smallest useful default-path units now exposed

On the last completed policy step before OOM, the default path reported:

- `tf_item_ms(cache/attn_core/finalize/ffn_l1)=672.14 / 2672.23 / 3347.14 / 844.34`
- `tf_cache_ms(append/prefix_maint/view/clone)=598.56 / 36.06 / 6.90 / 0.00`
- `tf_dispatch_ms(single/flash_prefix/flash_merge/dense)=1165.54 / 4859.98 / 0.00 / 0.00`
- `tf_dispatch_count(single/flash_prefix/flash_merge/dense)=1024 / 2368 / 0 / 0`
- `tf_flash_prefix_ms(prepare/prefix/tail/merge)=38.43 / 554.05 / 706.47 / 1217.20`
- `tf_flash_prefix_saved_np_u_last(prefix/tail/merge)=294.0 / 36.0 / 0.1 MiB`
- `tf_paged_avg(page_count/valid/prefix)=1.0 / 53.5 / 38.0`

### Interpretation

This is the strongest default-path evidence so far:

- the path is overwhelmingly dominated by `single_page + flash_prefix`; `flash_merge` and `dense` are not active here
- inside `flash_prefix`, the **saved-activation** load is concentrated in the **prefix attention core**
- the **merge** step is a large **time** sink but contributes almost no saved non-parameter storage
- `tail` attention contributes some saved storage, but far less than the prefix core
- cache append is non-trivial in time (`~599 ms`) and can still be the final OOM trigger, but it does not explain the resident plateau by itself

So the smallest useful units, in order of evidence, are now:

1. `flash_prefix prefix-core SDPA`  
   It carries the largest saved non-parameter storage (`~294 MiB` on the last completed step).

2. `flash_prefix merge`  
   It is a major time sink (`~1217 ms`) but not a major memory sink.

3. `flash_prefix tail-core SDPA`  
   It contributes some saved storage (`~36 MiB`) and some time (`~706 ms`), but it is clearly secondary to the prefix core.

4. `paged tail append (COW packed)`  
   It still matters for runtime and can be the final allocator trigger, but the evidence so far does not support it as the dominant resident-memory source.

This means the next optimization candidate, if we move from profiling to implementation later, should be chosen from:

- reducing or recomputing the `flash_prefix prefix-core` saved activations
- reducing the wall time of `flash_prefix merge`

and **not** from:

- generator-side changes
- TBPTT boundary payload changes
- FFN-tail changes alone

## 17. `flash_prefix` alias breakdown and merge sub-op breakdown

I continued along the two most promising lines only:

1. classify what `flash_prefix prefix-core` actually saves for backward
2. split `flash_prefix merge` into sub-ops

### 17.1 Prefix-core saved-for-backward is almost entirely `K/V`, not new intermediates

On the last completed default-path policy step before OOM, the profile now reports:

- `tf_flash_prefix_saved_np_u_last(prefix:q/k/v/other)=4.5 / 144.0 / 144.0 / 1.5 MiB`
- `tf_flash_prefix_saved_np_u_last(tail:q/k/v/other)=4.5 / 15.0 / 15.0 / 1.5 MiB`

Interpretation:

- the large `~294 MiB` prefix-core saved set is **not** dominated by newly created hidden intermediates
- it is almost entirely aliases to:
  - prefix `K`: about `144 MiB`
  - prefix `V`: about `144 MiB`
- `Q` is small (`~4.5 MiB`)
- true `other/new` saved storage is tiny (`~1.5 MiB`)

This is the most important memory result so far:

- **recomputing prefix-core alone is not the right memory fix**
- the prefix-core saved-tensor profile mostly reflects references to the already-resident prefix cache
- therefore, a "checkpoint prefix-core to drop saved activations" change would have limited impact on the `45 GiB` resident plateau

So the precise memory modification target is now clear:

- if we want to save memory here, we must reduce the resident **prefix K/V cache itself**
- not the tiny `other` intermediates inside prefix-core backward

In practical terms, the only meaningful modifications on this branch are:

- approximate:
  - immutable-prefix head sharing
  - immutable-prefix KV quantization
- exact:
  - none identified inside prefix-core itself; this branch does not expose a meaningful exact memory lever beyond changing how the prefix cache is stored globally

### 17.2 Merge is a speed problem, and the hot sub-op is the blend

The same full-batch step reports:

- `tf_flash_prefix_merge_ms(wait/pfx_cast/tail_cast/logaddexp/scale/blend)=`
  `92.85 / 176.74 / 163.94 / 73.55 / 173.04 / 508.50`

Interpretation:

- `wait_stream` is present but **not** the dominant cost
- `logaddexp` is relatively small
- the largest single merge sub-op is the final **blend**
- the two `to(float32)+normalize` paths and the scale computation are also substantial

So the precise speed modification target is now also clear:

- do **not** optimize `wait_stream` first
- do **not** optimize `logaddexp` first
- optimize the elementwise merge pipeline itself

### 17.3 Clear modification methods

At this point the modification directions are specific enough to stop profiling this branch.

#### Memory-focused method

Do **not** spend more time on prefix-core checkpoint/recompute.

Instead, if an approximate method is acceptable, target the resident prefix cache directly:

- apply immutable-prefix head sharing earlier / more broadly
- or quantize immutable-prefix `K/V`

Reason:

- the prefix-core backward footprint is overwhelmingly `K/V` alias storage, not fresh temporary activations

#### Speed-focused method

Implement a fused `flash_prefix merge` kernel (CUDA/Triton/custom op) that performs:

- prefix/tail cast to FP32
- LSE combine
- scale computation
- output blend

in one fused step.

Reason:

- the current merge pipeline spends most of its time in:
  - blend: `~508 ms`
  - cast/normalize: `~341 ms`
  - scale: `~173 ms`
- these are exactly the operations a fused merge kernel would collapse

This is now the most evidence-backed speed optimization on the default path.

## 18. Immutable-prefix head sharing: root cause and validated memory cut

### 18.1 Why the first full-batch A/B looked ineffective

The first `off` vs `TICL_POLICY_IMMUTABLE_PREFIX_HEAD_SHARING=mean` full-batch runs were misleadingly close.
The missing piece was not the idea itself, but **where the mode was applied**.

At TBPTT detach time, the cache can still have:

- no `k_prefix/v_prefix` yet
- only a mutable paged tail

In that case the previous implementation dropped the requested sharing mode and reset
`prefix_head_sharing` back to `"off"`.

That meant:

- the detached cache remembered no future sharing intent
- later full pages that became immutable prefix in the next window were still stored as full multi-head prefix

So the earlier A/B was not testing "real default-path head sharing"; it was mostly testing a no-op.

### 18.2 Fix

The fix is small and targeted:

- in `EnvironmentPrior._apply_immutable_prefix_head_sharing(...)`
- if the requested mode is `mean/first` but no `k_prefix/v_prefix` has been materialized yet
- persist `prefix_head_sharing=mode` anyway

This lets later prefix maintenance in `TransformerEncoderLayer.forward_step(...)` apply `_reduce_prefix_heads(...)`
when new full pages are moved from the paged tail into immutable prefix.

I also added a regression test that covers exactly this delayed-materialization path:

- `test_tbptt_detach_head_sharing_mode_persists_before_prefix_materializes`

### 18.3 Full-batch A/B after the fix

Using the same default full-batch profiling command:

- `off`
  - `policy_cache_post ~= 1272 MiB`
  - `policy_cache_detach_post ~= 1152 MiB`
  - `tf_flash_prefix_saved_np_u_last(prefix:q/k/v/other)=4.5/144.0/144.0/1.5 MiB`
  - `tf_flash_prefix_saved_np_u_last(prefix/tail/merge)=294.0/36.0/0.1 MiB`
  - plateau about `45.48 GiB`
  - reached `step=105/106`

- `mean`
  - `policy_cache_post ~= 640 MiB`
  - `policy_cache_detach_post ~= 640 MiB`
  - `tf_flash_prefix_saved_np_u_last(prefix:q/k/v/other)=4.5/48.0/48.0/1.5 MiB`
  - `tf_flash_prefix_saved_np_u_last(prefix/tail/merge)=102.0/45.0/0.1 MiB`
  - plateau about `45.43 GiB`
  - reached `step=108/109`

Interpretation:

- this is a **real** memory cut on the default training path
- the main win is resident prefix cache size and the prefix-core `K/V` references saved for backward
- the plateau does not collapse, so this is not the full solution
- but it is a meaningful and validated reduction in the dominant prefix branch

### 18.4 Default-path integration

After validating the fix, I wired `per_feature_v25` to opt into:

- `TICL_POLICY_IMMUTABLE_PREFIX_HEAD_SHARING=mean`

by default in the `rlpfn` training entrypoint, while still letting an explicit shell env override win.

This keeps the approximation narrowly scoped:

- enabled by default for `per_feature_v25`
- not forced onto the standard backbone

The default full-batch run now prints:

- `Policy immutable-prefix head sharing: mean`

and reproduces the same improved memory numbers above.

## 19. Allocator snapshot method: resident-set attribution by active block stacks

To avoid repeating the earlier mistake of treating the final OOM trigger as the main cause,
I added a targeted CUDA allocator snapshot profiler on the default path.

Method:

- enable `torch.cuda.memory._record_memory_history(enabled="state", context="state", stacks="python")`
- capture `_snapshot()` only on selected `phase@step`
- group `active_allocated` blocks by Python allocation stack
- compare snapshots by stack-delta, not just by final stack trace

For the main default full-batch run I captured:

- `policy_pre@95`
- `policy_post@95`
- `tbptt_sink_post@95`
- `policy_pre@108`
- `policy_post@108`

This was enough to separate:

- cross-step resident growth
- same-step policy growth
- final OOM trigger

## 20. The true dominant resident growers are now localized

### 20.1 Full-signature dominant groups at `policy_pre@95`

The first precise resident-set snapshot on the default path was:

- `policy_pre@95: alloc ~= 40.43 GiB`

Top active groups:

1. exact transition affine output path
   - `ticl/priors/environment_prior.py:13045:_batch_affine`
   - caller chain:
     - `ticl/priors/environment_prior.py:3842:transition_fn`
     - `ticl/priors/environment_prior.py:3442:transition_fn`
   - resident size: about `10.15 GiB`

2. item-cache grow, `V` branch
   - `ticl/models/layer.py:1053:_concat_dim2`
   - caller chain:
     - `ticl/models/layer.py:2233:_append_to_kv_pages_cow_packed`
     - `ticl/models/layer.py:2526:forward_step`
   - resident size: about `9.37 GiB`

3. item-cache grow, `K` branch
   - `ticl/models/layer.py:1053:_concat_dim2`
   - caller chain:
     - `ticl/models/layer.py:2232:_append_to_kv_pages_cow_packed`
     - `ticl/models/layer.py:2526:forward_step`
   - resident size: about `9.35 GiB`

This resolves the earlier ambiguity around the duplicated `_concat_dim2` leaf:

- one group is `grown_v`
- the other is `grown_k`

### 20.2 Same-step growth at `95`

`policy_post@95 - policy_pre@95`:

- `grown_v` `_concat_dim2`: about `+192 MiB`
- `grown_k` `_concat_dim2`: about `+192 MiB`
- `qkv` projection in policy step:
  - `ticl/models/layer.py:2289:forward_step`
  - caller chain:
    - `ticl/models/perfeature_tabpfn.py:619:_forward_item_step_batched`
    - `ticl/models/perfeature_tabpfn.py:715:forward_step`
  - about `+24 MiB`

Interpretation:

- the dominant same-step policy growth is not FFN tail or finalize
- it is the paged-tail grow path in `_append_to_kv_pages_cow_packed`
- the `qkv` projection is visible but clearly smaller

### 20.3 Cross-step growth from `tbptt_sink_post@95` to `policy_pre@108`

This is the most important delta, because it explains how the plateau climbs before the next OOM:

- exact transition affine path:
  - `ticl/priors/environment_prior.py:13045:_batch_affine`
  - about `+1296 MiB`
  - `+408` active blocks

- `grown_v` `_concat_dim2`:
  - about `+478 MiB`
  - `+352` active blocks

- `grown_k` `_concat_dim2`:
  - about `+477 MiB`
  - `+352` active blocks

Interpretation:

- the largest cross-step resident grower is now clearly `_batch_affine`
- the second and third largest resident growers are the `V/K` page-grow cats in the item cache
- explicit cache containers are still far too small to explain the plateau on their own

### 20.4 Same-step growth at `108`

`policy_post@108 - policy_pre@108`:

- `grown_v` `_concat_dim2`: about `+78.5 MiB`
- `grown_k` `_concat_dim2`: about `+78.0 MiB`
- `qkv` projection: about `+24 MiB`

Interpretation:

- by `108`, the same-step policy growth is still mostly the page-grow cats
- but the larger plateau increase between `95` and `108` is dominated by the transition affine path, not by one more large policy spike

### 20.5 Final OOM trigger vs dominant resident sources

The final OOM still triggers in:

- `ticl/models/layer.py:2233`
- `grown_v = TransformerEncoderLayer._concat_dim2((prev_v, v_new_bhld))`

This is only the trigger.

The dominant resident sources identified by active-block attribution are:

1. `EnvironmentPrior._batch_affine(...)` exact-SCM transition affine outputs
2. `TransformerEncoderLayer._append_to_kv_pages_cow_packed(...)` page-grow cats for:
   - `grown_v`
   - `grown_k`

That attribution is now based on:

- phase-local plateau measurements
- explicit object-memory measurements
- active-block stack snapshots
- same-step and cross-step snapshot deltas

This is the strongest evidence chain so far, and it is specific enough to guide future optimization work without falling back to OOM-trigger reasoning.

## 21. Active-path affine-focus profiling rules out a common false cause

### 21.1 Initial profiler miss was caused by patching the wrong rollout copy

`environment_prior.py` contains more than one rollout/profile setup block.

The first affine-focus patch was attached to the earlier copy, while the default
training path actually runs through the later `_rollout_family_group_vectorized_with_policy(...)`
block around `16795+`.

After moving the profile initialization to the active path, the OOM summary
started printing:

- `updates=3740`
- `samples=3740`

This confirms the active exact-SCM path was profiled, not just the inactive copy.

### 21.2 What affine-focus now proves

Default full-batch run with only:

- `TICL_POLICY_AFFINE_FOCUS_PROFILE=1`
- `TICL_POLICY_PHASE_MEMORY_PROFILE=1`
- `TICL_POLICY_POLICY_LOCAL_PROFILE=1`
- `TICL_POLICY_STEP_PROFILE=1`

OOM summary:

- `first_affine_tail`
  - `updates=3740`
  - `reqgrad_samples=3740`
  - `saved_u(max) ~= 119.8 MiB`
  - `tracked_affine_u(max) ~= 0.0 MiB`
  - `tracked_affine_r(max) ~= 0.0 MiB`

Interpretation:

- the first exact-SCM affine output participates in autograd (`requires_grad=1`)
- but the later transition tail does **not** save the first-affine tensor itself for backward
- therefore the large `_batch_affine` resident group seen in allocator snapshots is **not**
  explained by “first-affine output got directly stashed by saved-for-backward hooks”

This rules out an important false cause.

### 21.3 What remains consistent with all evidence

Allocator snapshots still show:

- `_batch_affine` cross-step resident growth:
  - `tbptt_sink_post@95 -> policy_pre@108`
  - about `+1296 MiB`

At the same time, affine-focus shows the first-affine tensor itself is not the
thing being saved by the tail.

So the most consistent explanation is now:

- `_batch_affine` appears as a dominant resident group because its outputs stay
  alive as part of the still-live exact transition graph across unrolled steps
- not because the tail explicitly saves the first-affine output tensor in a
  saved-for-backward slot

In other words, this is a **live-graph retention** problem, not a simple
“tail saved the wrong tensor” problem.

## 22. Policy-side resident cause remains the paged grow cats

Nothing in the new evidence weakens the earlier policy-side attribution.

Still true on the default path:

- same-step policy growth at `95`:
  - `grown_v`: about `+192 MiB`
  - `grown_k`: about `+192 MiB`
- cross-step growth from `tbptt_sink_post@95 -> policy_pre@108`:
  - `grown_v`: about `+478 MiB`
  - `grown_k`: about `+477 MiB`

The end-to-end `step_profile` summary still does not surface the new
`paged_grow_*` counters, so that profiling path is not yet trusted as a source
of final numbers. However, the allocator snapshot attribution is already strong:

- `grown_k/grown_v` are still the second and third largest resident growers
- the trigger line may vary, but the resident attribution does not

This means the policy-side resident problem is still:

- mutable packed-page growth via `torch.cat(prev_page, new_token)`
- not explicit cache-container size
- not FFN/finalize
- not flash-prefix merge as the primary memory source

## 23. Transition-side root graph had a missing marker, then converged on the reward branch

### 23.1 Root-graph marker was initially too narrow

The original root-graph affine marker list only matched:

- `BmmBackward`
- `BaddbmmBackward`
- `RaggedBatchAffineFnBackward`
- `TiledRaggedBatchAffineFnBackward`

This missed the actual exact-SCM training-time custom backward:

- `_PrefixSampleInputActivatedAffineUpdateGradInputFnBackward`

After adding:

- `PrefixSampleInputActivatedAffineUpdateGradInputFnBackward`
- `PrefixTiledBatchAffineFnBackward`

the root graph became able to see the exact-SCM hidden-loop custom affine path.

### 23.2 Looking only at `state_t/reward_t` was too late

Even after fixing the marker list, the late explicit roots still do **not** expose
the affine path:

- `state_t`
- `state_delta`
- `reward_t`
- `tbptt_reward_*`
- `aev*_prev_delta`

These roots remain small `Where/Mul/Sub` graphs and do not identify the dominant
transition resident source.

This means the useful root has to be taken earlier inside the transition path.

### 23.3 Earlier transition roots narrow the culprit to the reward branch

New roots recorded on the active family-group path:

- `transition_out_g`
- `transition_x_next_g`
- `transition_reward_unit_g`
- `transition_reward_raw_g`
- `transition_state_next_g`

Results on stable exact-SCM runs:

- `transition_reward_unit_g`
  - exposes `_PrefixSampleInputActivatedAffineUpdateGradInputFnBackward`
  - small run (`b=2, n=32`): `affine_nodes=3`
  - mid run (`b=16, n=128`): `affine_nodes=6`
- `transition_x_next_g`
  - `affine_nodes=0`
- `transition_state_next_g`
  - `affine_nodes=0`
- `transition_reward_raw_g`
  - `affine_nodes=0`
- `transition_out_g`
  - `requires_grad=0`
  - `has_grad_fn=0`

Interpretation:

- the exact-SCM custom affine/update backward is visible on the **reward-unit**
  branch
- it is **not** visible on the `x/state` branch roots we currently carry forward
- the packed fused output `transition_out_g` itself is not a useful autograd root
  for this diagnosis

This is the strongest direct graph-level evidence so far for where the transition
retained path actually lives.

## 24. Transition affine-focus segments converge on `hidden_grad_fused_update`

With segment profiling enabled on the active exact-SCM hidden loop, the transition
subsegments now separate cleanly.

### 24.1 Small stable run (`b=2, n=32`)

Key result:

- `hidden_grad_fused_update`
  - `saved_u(max) ~= 6.9 MiB`
  - `wall ~= 1.3-2.1 s`

All other hidden-loop segments stay near zero saved storage:

- `hidden_mask_apply`
- `hidden_output_state_gather`
- `hidden_output_reward_gather`
- `first_affine_tail`

### 24.2 Mid stable run (`b=16, n=128`)

Key result:

- `hidden_grad_fused_update`
  - `saved_u(max) ~= 170.9 MiB`
  - `samples = 768`
  - `wall ~= 2.00 s`

Other segments remain much smaller:

- `hidden_output_state_gather`
  - `saved_u(max) ~= 0.3 MiB`
- `hidden_output_reward_gather`
  - `saved_u(max) ~= 0.3 MiB`
- `hidden_mask_apply`
  - `saved_u(max) ~= 0.0 MiB`
- `first_affine_tail`
  - `saved_u(max) ~= 0.0 MiB`

### 24.3 Full-batch OOM run (`b=256, n=1024`)

The OOM summary is still consistent with the same subsegment:

- `hidden_grad_fused_update`
  - `saved_u(max) ~= 119.5 MiB`
  - `samples = 16720`
  - `wall ~= 8.28 s`

No other segment is in the same range.

Interpretation:

- inside the exact-SCM transition path, the minimal subsegment with the strongest
  evidence is now `hidden_grad_fused_update`
- this is the only transition subsegment that is simultaneously:
  - on the exact training-time path
  - large in `saved-for-backward`
  - nontrivial in wall time
  - connected to the earlier reward-side custom affine backward root

## 25. Policy-side minimal unit remains `grown_k/grown_v`

The policy-side attribution did not move in this round.

What is now trusted end-to-end:

- `tf_paged_grow_*` counters are now surfaced through `per_feature_v25` step
  profile aggregation
- allocator snapshot deltas still show the same dominant policy residents:
  - `grown_v = _concat_dim2(prev_v, v_new_bhld)`
  - `grown_k = _concat_dim2(prev_k, k_new_bhld)`

So the policy-side minimal unit is still:

- `_append_to_kv_pages_cow_packed(...)`
  - specifically the page-grow `torch.cat` for:
    - `grown_v`
    - `grown_k`

This remains distinct from:

- explicit cache container size
- flash-prefix merge
- FFN/finalize tail

## 26. Current strongest attribution state

At this point the evidence is strongest for these two minimal units:

1. Transition side:
   - exact-SCM hidden-loop `hidden_grad_fused_update`
   - seen directly via segment profiling
   - graph-visible on `transition_reward_unit_g`

2. Policy side:
   - paged mutable-tail grow cats
   - `grown_k`
   - `grown_v`
   - seen directly via allocator snapshots and step-profile `paged_grow_*`

This is the first point in the investigation where both sides have been reduced
from “large plateau somewhere in policy/transition” to concrete minimal
calculation units.
