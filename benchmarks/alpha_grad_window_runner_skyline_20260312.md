## Alpha Family TBPTT Window Runner

### Scope

- Worktree: `/tmp/rlpfn_alpha_grad_bridge_20260312`
- Base commit: `623952c`
- Change: dedicated `alpha_grad + torch_vectorized + family + active TBPTT` window runner
- Goal: eliminate non-final-window second-backward failure without changing `g0/g1/SCM` math

### CPU Semantic Checks

- `test_environment_prior_alpha_grad_rollout_gradients_are_finite`
- `test_environment_prior_alpha_grad_tbptt_multi_group_rollout_gradients_are_finite`
- `test_environment_prior_alpha_grad_tbptt_nonfinal_family_window_gradients_are_finite`
- `test_alpha_grad_tbptt_window_equal_horizon_matches_full_horizon_semantics`
- `test_alpha_grad_tbptt_family_vectorized_handles_nonfinal_window_case`
- `test_mlp_prior.py`
- strict SCM/reference subset

All passed in the experimental worktree.

### Fixed-Override Risky Load

Setup:

- objective: `alpha_grad`
- backend: `torch_vectorized`
- grouping: `family`
- `batch_size=64`
- `n_samples=1024`
- `single_eval_pos=697`
- `pg_tbptt_window=32`
- model: default `rlpfn` transformer (`25.23M` backbone params)
- guard: fail if peak allocated or reserved GPU memory exceeds `10 GiB`
- overrides fixed across runs: same `h_list`, `env_seeds`, `rollout_seeds`

Results:

| variant | status | batch wall (s) | peak alloc (MiB) | peak reserved (MiB) | objective | reward mean | reward std |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| accepted `623952c` | ok | 273.471 | 3229.740 | 3330.0 | -2.867032 | -0.089595 | 2.360374 |
| window runner | ok | 270.688 | 3229.130 | 3328.0 | -2.867032 | -0.089595 | 2.360374 |

Interpretation:

- The dedicated family window runner fixes the non-final-window graph-lifetime issue.
- Under fixed overrides, it matches the accepted baseline on objective and reward stats.
- Risky-load GPU memory is effectively unchanged.
- Runtime is also effectively unchanged.
- This change is therefore correctness-oriented, not a measurable performance optimization.

### Cross-Objective Risky-Load Comparison

Setup:

- worktree: `/tmp/rlpfn_alpha_grad_bridge_20260312`
- backend: `torch_vectorized`
- grouping: `family`
- `batch_size=64`
- `n_samples=1024`
- `single_eval_pos=697`
- `pg_tbptt_window=32`
- `kv_cache_mode=paged`
- guard: fail if peak allocated or reserved GPU memory exceeds `10 GiB`
- fixed overrides across runs: same `h_list`, `env_seeds`, `rollout_seeds`

Results:

| objective | status | batch wall (s) | peak alloc (MiB) | peak reserved (MiB) | RSS (GiB) | objective | reward mean | reward std |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `reinforce` | ok | 121.753 | 3177.008 | 3226.0 | 1.612 | -2.818624 | -0.088082 | 2.362865 |
| `first_policy_gradient` | ok | 190.881 | 3227.543 | 3292.0 | 1.768 | -0.089595 | -0.089595 | 2.360374 |
| `alpha_grad` | ok | 247.793 | 3229.130 | 3328.0 | 1.758 | -2.867032 | -0.089595 | 2.360374 |

Derived deltas:

- `first_policy_gradient - reinforce`: `+69.128s`, `+50.535 MiB`
- `alpha_grad - first_policy_gradient`: `+56.912s`, `+1.586 MiB`
- `alpha_grad - reinforce`: `+126.040s`, `+52.122 MiB`

Interpretation:

- The largest remaining optimization headroom is on the `g1` / `first_policy_gradient` branch, not on `g0`.
- GPU memory is nearly identical across all three objectives on this fixed risky load, so the dominant problem is time, not an alpha-specific memory spike.
- The extra alpha cost over `first_policy_gradient` is smaller than the extra `first_policy_gradient` cost over `reinforce`, which matches earlier diagnostics that `g0` is already relatively cheap and the main expensive path is `g1`.
- Coarse `nvidia-smi` utilization sampling is not used as a decision signal here; these conclusions are based on fixed-override batch wall and guarded peak memory only.

### Backward Time Split

Using the same fixed risky load, the guarded harness also split wall time into:

- `sink_backward_s`: time spent inside the TBPTT sink `loss.backward()`
- `non_sink_s`: everything else in `_compute_policy_rollout_chunk_loss(...)`

Results:

| objective | batch wall (s) | sink backward (s) | non-sink (s) | sink share |
| --- | ---: | ---: | ---: | ---: |
| `reinforce` | 114.171 | 15.796 | 98.375 | 13.8% |
| `first_policy_gradient` | 194.036 | 86.704 | 107.332 | 44.7% |
| `alpha_grad` | 258.235 | 76.274 | 181.961 | 29.5% |

Interpretation:

- `first_policy_gradient` lifts the direct sink backward cost sharply over `reinforce`; this is the clearest evidence that the `g1` shared backward path is the main remaining time head.
- `alpha_grad` does not further increase sink backward over `first_policy_gradient`; its extra time mostly lands in `non_sink`, consistent with the earlier accepted diagnosis that alpha-specific overhead sits in the internal `g1`-related computation rather than the final `loss.backward()`.
- The next meaningful optimization should therefore target the `g1` path itself, not `g0`, and not generic rollout sharing.

### Risky-Load Rejected Candidates

All candidates below were tested on the same fixed risky load:

- backend `torch_vectorized`
- grouping `family`
- `batch_size=64`
- `n_samples=1024`
- `single_eval_pos=697`
- `pg_tbptt_window=32`
- `kv_cache_mode=paged`
- guard `10 GiB`

#### 1. Transition Inner Grouping = `structure`

Hypothesis:

- The 64 strict SCM samples are heterogeneous in activation and hidden depth, so a finer transition grouping might reduce masked `where` work in `g1`.

Evidence:

- Sampled risky-load batch distribution:
  - activations: `relu=28`, `identity=19`, `tanh=17`
  - hidden blocks: `1:20`, `2:24`, `3:8`, `4:5`, `5:3`, `6:2`, `8:2`
  - strict/reference semantics: `64 / 64`
- Baseline `balanced2`: `batch_wall_s=194.036`, `sink_backward_s=86.704`
- `structure`: `batch_wall_s=192.984`, `sink_backward_s=84.047`

Interpretation:

- The gain is about `1.1s` on a `~194s` batch, well below the bar for a meaningful optimization.
- This candidate is rejected for now.

#### 2. Fused SCM Activation-Backward Rowwise Update

Hypothesis:

- Replace the fused affine custom-backward activation derivative path from repeated full-tensor `torch.where(...)` to rowwise in-place masked updates.

Evidence:

- SCM/reference tests still passed after the patch.
- Risky-load `first_policy_gradient` improved only from `194.036s` to `190.731s`.
- Peak memory and reward/objective statistics were unchanged.
- Single-window profiler still showed essentially unchanged `aten::where` / `WhereBackward0` dominance, so the patch did not hit the real source.

Interpretation:

- This was a real but weak improvement, about `1.7%` wall.
- It does not meet the “significant optimization” bar and was reverted.

#### 3. Strict SCM Layer Compile

Hypothesis:

- `TICL_POLICY_REFERENCE_SCM_LAYER_COMPILE=1` might fuse the strict SCM layer-step `where`-heavy path.

Evidence:

- Two-batch steady-state check was used to avoid counting compile warmup as improvement.
- Compile-off:
  - batch 1: `183.647s`
  - batch 2: `184.099s`
- Compile-on:
  - batch 1: `254.405s`
  - batch 2: `205.393s`
- Compile-on also hit `torch._dynamo` recompile-limit warnings on the real risky load:
  - reason: `tensor 'b' stride mismatch`

Interpretation:

- Even the second batch is clearly slower than baseline.
- Current strict SCM layer compile is rejected as an optimization path for this workload.

### Current Decision

- The largest remaining headroom is still `g1 / first_policy_gradient`.
- The strongest already-tested low-risk candidates did not deliver significant gains.
- No new code change is retained from this round; only benchmark evidence is kept.

### Accepted: No-Copy Strict SCM Grad-Fused Hidden Update

Hypothesis:

- The earlier strict SCM grad-fused candidate was faster, but its memory regression came from saving step-local contiguous copies of per-layer hidden `w/b` tensors.
- If the grad-fused path keeps the original layer views and only computes `dL/dz`, it should preserve the time win without the multi-GiB GPU regression.

Implementation:

- Added a CUDA-only grad-fused sample-update path for strict SCM hidden layers.
- Forward still uses the existing sample-fused update kernel.
- Backward only returns `dL/dz`.
- Crucially, it saves the original hidden-layer `w/b` views instead of step-local `.contiguous()` copies.
- Enabled by default through `TICL_POLICY_REFERENCE_SCM_HIDDEN_UPDATE_GRAD_FUSED=1` semantics on this experimental branch.

Semantic checks:

- strict SCM CUDA forward/gradient equivalence vs unfused path passed
- `alpha_grad` family/TBPTT finite-gradient checks passed
- `test_mlp_prior.py` passed

Risky-load evidence, fixed overrides `B=64, ns=1024, sep=697, tbptt=32, paged, family, policy offload`:

| objective | baseline wall (s) | new wall (s) | delta | baseline peak alloc (MiB) | new peak alloc (MiB) | delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `first_policy_gradient` | 284.605 | 271.181 | -13.424 | 1112.434 | 1113.152 | +0.718 |
| `alpha_grad` | 400.780 | 369.806 | -30.974 | 1118.526 | 1119.268 | +0.742 |

Derived split changes:

- `first_policy_gradient sink_backward`: `132.973s -> 114.028s`
- `alpha_grad sink_backward`: `139.667s -> 108.692s`

Interpretation:

- This is the first `g1` optimization on this branch that clears the acceptance bar:
  - meaningful risky-load wall reduction
  - essentially flat GPU peak memory
  - semantic checks still pass
- The evidence supports the prior diagnosis: the old regression source was step-local hidden-weight copies, not `WhereBackward` by itself.

### Accepted Memory Control: Policy-Only Saved-Tensors CPU Offload

Hypothesis:

- The biggest GPU memory head is the shared transformer/policy saved-tensor path, not `g1`-specific SCM state.
- If CPU offload is limited to `policy_step_fn` only, it should recover most of the GPU-memory benefit of full saved-tensor offload while avoiding most of the host-memory and wall-time penalty.

Implementation:

- Added `pg_saved_tensors_cpu_offload_scope` with choices:
  - `all`: existing behavior, offload all saved tensors during policy-gradient rollout
  - `policy`: only offload tensors saved during `policy_step_fn`
- Default remains `all` when offload is enabled, so existing CLI/config behavior is preserved.

Risky-load evidence, fixed overrides `B=64, ns=1024, sep=697, tbptt=32, paged, family`:

#### `first_policy_gradient`

| scope | batch wall (s) | peak alloc (MiB) | peak reserved (MiB) | RSS (GiB) | objective | reward mean | reward std |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `none` | 179.344 | 3227.543 | 3292.0 | 1.738 | -0.089595 | -0.089595 | 2.360374 |
| `policy` | 284.605 | 1112.434 | 1168.0 | 7.339 | -0.089595 | -0.089595 | 2.360374 |
| `all` | 439.609 | 1088.646 | 1142.0 | 15.874 | -0.089595 | -0.089595 | 2.360374 |

Interpretation:

- `policy` recovers almost all of the GPU reduction of full offload:
  - `3227.543 -> 1112.434 MiB` vs full `1088.646 MiB`
- but is much cheaper than full offload:
  - wall `284.605s` vs `439.609s`
  - RSS `7.339 GiB` vs `15.874 GiB`
- objective and reward stats are unchanged.

#### `alpha_grad`

| scope | batch wall (s) | peak alloc (MiB) | peak reserved (MiB) | RSS (GiB) | objective | reward mean | reward std |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `none` | 270.688 | 3229.130 | 3328.0 | 1.758 | -2.867032 | -0.089595 | 2.360374 |
| `policy` | 400.780 | 1118.526 | 1176.0 | 7.422 | -2.867032 | -0.089595 | 2.360374 |

Interpretation:

- The same shared-memory control works for `alpha_grad`.
- GPU peak drops by about `2.11 GiB` while preserving alpha objective/reward statistics.
- The time cost is material, but still much smaller than expected from full-offload behavior.

Decision:

- This is the first memory-control change in this line that clears the “significant and evidence-backed” bar.
- It was first accepted as an opt-in control and later promoted to the experimental default after additional risky-load validation.
- For larger effective batch via lower GPU memory, prefer:
  - `--pg-saved-tensors-cpu-offload true`
  - `--pg-saved-tensors-cpu-offload-scope policy`
  - `--pg-saved-tensors-pin-memory false`

### Promoted Default: Policy-Only Offload

The experimental branch now promotes the following `rlpfn` defaults:

- `pg_saved_tensors_cpu_offload = true`
- `pg_saved_tensors_cpu_offload_scope = policy`
- `pg_saved_tensors_pin_memory = false`

Reason:

- On the maintained risky load this is the best demonstrated memory control so far.
- It lowers GPU memory by about `2.1 GiB` relative to the no-offload baseline while avoiding the much larger host-memory and wall-time penalty of full-rollout offload.

Default-risky-load evidence after promotion, same fixed overrides `B=64, ns=1024, sep=697, tbptt=32, paged, family`:

| objective | batch wall (s) | sink backward (s) | non-sink (s) | peak alloc (MiB) | peak reserved (MiB) | RSS (GiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `reinforce` | 201.570 | 62.046 | 139.524 | 1089.750 | 1144.0 | 7.253 |
| `first_policy_gradient` | 284.605 | 132.973 | 151.633 | 1112.434 | 1168.0 | 7.339 |
| `alpha_grad` | 400.780 | 125.626 | 275.154 | 1118.526 | 1176.0 | 7.422 |

Interpretation:

- GPU memory is now tightly controlled across all three objectives on the risky load.
- The direct `g1` shared-backward gap remains large:
  - `first_policy_gradient sink_backward - reinforce sink_backward = +70.927s`
- `alpha_grad` still carries a larger extra `non-sink` cost than `first_policy_gradient`, but that overhead is still `g1`-related alpha logic, not a fresh GPU-memory problem.
- Under the new default, the next optimization target remains time, not memory.

### Accepted: Planned Rowwise Noise for Strict SCM

Hypothesis:

- After the accepted no-copy hidden grad-fused update, the largest remaining shared forward head inside strict SCM `transition_generator` was rowwise hidden-noise sampling.
- The current implementation recomputed `nonzero(scale > 0)` per sample on every layer and every step.
- If those active indices are precomputed once per layer and reused, while preserving the exact per-sample/per-layer `torch.randn(..., generator=g)` call order, the optimization should be semantics-safe and should reduce risky-load wall time without increasing memory.

Evidence before code:

- Risky-load shared forward split (`B=64, ns=1024, sep=697, tbptt=32, paged, family, policy offload`) showed:
  - `transition_generator_s ≈ 97.610s`
  - `rowwise_noise_s ≈ 45.346s`
  - `hidden_update_grad_fused_s ≈ 22.536s`
  - `batch_affine_s ≈ 2.332s`
- A microbenchmark on the real risky-batch hidden-noise scales, preserving exact sub-batch generator mapping, showed:
  - current helper: `45.223 ms`
  - indices-only planned helper: `21.053 ms`
  - `same_output = true`
  - `same_tail = true`
- A previous “batched big draw then slice” candidate was rejected because it changed CUDA generator consumption order; this accepted variant does not.

Implementation:

- Added `_build_rowwise_scaled_noise_plan(...)`
- Added `_sample_rowwise_scaled_noise_with_plan(...)`
- Precompute per-layer active-index plans once in strict SCM reference builder
- Use the planned helper only on the `noise_generators is not None` path, keeping the same generator call order as the original rowwise implementation

Focused checks:

- rowwise-noise plan helper matches unplanned helper exactly on CUDA
- strict SCM grad-fused update equivalence still passes
- `alpha_grad` finite-gradient focused test passes
- `test_mlp_prior.py` passes

Risky-load evidence, same fixed overrides `B=64, ns=1024, sep=697, tbptt=32, paged, family, policy offload`:

| objective | before wall (s) | after wall (s) | delta | before peak alloc (MiB) | after peak alloc (MiB) | delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `first_policy_gradient` | 271.181 | 255.523 | -15.658 | 1113.152 | 1113.661 | +0.509 |
| `alpha_grad` | 369.806 | 344.818 | -24.988 | 1119.268 | 1119.777 | +0.509 |

In-situ risky-load split for `first_policy_gradient` after the change:

- `transition_generator_s: 97.610 -> 81.268s`
- `policy_step_s: 54.843 -> 54.288s`
- `hidden_update_grad_fused_s: 22.536 -> 23.252s`

Interpretation:

- The measured win lands exactly where expected: strict SCM shared forward, not policy forward and not `g1` sink backward.
- GPU memory is unchanged within noise.
- This is a semantics-safe shared optimization that benefits both `first_policy_gradient` and `alpha_grad`.

## Accepted: cache strict-SCM runtime tensors inside the joint transition closure

Hypothesis:

- After the accepted rowwise-noise planning change, `transition_other` still contained a measurable shared-forward head from repeated `.to(...)` materialization inside the strict SCM joint transition closure.
- If those runtime tensors are cached once per `(device, dtype)` at builder scope, the shared forward path should improve without changing rollout, `g0`, `g1`, or SCM semantics.

Evidence before touching code:

- Risky-load `first_policy_gradient` split after the rowwise-noise change still showed:
  - `transition_generator_s ≈ 89.17s`
  - `rowwise_noise_plan_s ≈ 30.60s`
  - `hidden_update_grad_fused_s ≈ 25.05s`
  - `tensor_to_s ≈ 9.00s` over `1,156,096` calls
  - `transition_residual_s ≈ 16.96s`
- This singled out repeated runtime tensor/materialization churn as the next safest shared head to attack.

Implementation:

- Add a closure-level cache in `_build_reference_scm_joint_transition_padded_batch_fn(...)`, keyed by `(device, dtype)`
- Precompute and reuse:
  - first affine weights/biases
  - hidden weights/biases/noise-scale stacks
  - hidden masks and activation masks
  - prefix/tile metadata tensors used by the strict SCM layer-step helpers
- Leave all stochastic paths, `g0/g1`, and SCM math unchanged

Focused checks:

- `strict_reference_scm_hidden_update_grad_fused_matches_unfused_on_cuda`
- `rowwise_scaled_noise_with_plan_matches_unplanned_on_cuda`
- `alpha_grad_rollout_gradients_are_finite`
- `alpha_grad` TBPTT checkpoint subset
- `test_mlp_prior.py`

Risky-load evidence, same fixed overrides `B=64, ns=1024, sep=697, tbptt=32, paged, family, policy offload`:

| objective | before wall (s) | after wall (s) | delta | before peak alloc (MiB) | after peak alloc (MiB) | delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `first_policy_gradient` | 255.523 | 250.071 | -5.452 | 1113.661 | 1113.661 | +0.000 |
| `alpha_grad` | 344.818 | 339.817 | -5.001 | 1119.777 | 1119.777 | +0.000 |

In-situ risky-load split for `first_policy_gradient` after the change:

- `transition_generator_s: 89.170 -> 81.944s`
- `tensor_to_s: 9.00 -> 7.40s`
- `tensor_to_calls: 1,156,096 -> 538,204`
- `policy_step_s: 52.30s` unchanged within noise

Interpretation:

- The win is shared-forward only and lands where the profiler predicted.
- GPU memory is flat within noise; no new host-memory pressure was introduced.
- This is a semantics-safe runtime-cache cleanup, not a mathematical change.
