## Alpha Family TBPTT Window Runner

### Accepted Experimental Skyline: 2GB Strict-SCM Partition

Scope:

- worktree: `/tmp/rlpfn_alpha_grad_bridge_20260312`
- branch: `experiment/alpha-family-window-runner-20260312`
- change: raise default strict reference SCM partition budget from `256 MiB` to `2 GiB`
- intent: reduce tiny strict-SCM hidden backward launches under the same full-offload training setup
- acceptance basis: user-directed relaxed criterion that allows this acceleration to become the new skyline despite deterministic execution drift

Guarded risky-load benchmark setup:

- backend: `torch_vectorized`
- grouping: `family`
- `batch_size=64`
- `n_samples=1024`
- `single_eval_pos=697`
- `pg_tbptt_window=32`
- `kv_cache_mode=paged`
- saved-tensors offload:
  - `pg_saved_tensors_cpu_offload=true`
  - `pg_saved_tensors_cpu_offload_scope=policy`
  - `pg_saved_tensors_pin_memory=false`
  - `pg_saved_tensors_cpu_offload_auto_disable_when_safe=false`
- guard:
  - fail if peak allocated or reserved GPU memory exceeds `10 GiB`
  - fail if host RSS exceeds `20 GiB`
- fixed overrides across runs: same `h_list`, `env_seeds`, `rollout_seeds`

Results:

| objective | status | batch wall (s) | sink backward (s) | non-sink (s) | peak alloc (MiB) | peak reserved (MiB) | RSS (GiB) | objective | reward mean | reward std |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `first_policy_gradient` | ok | 171.641 | 87.630 | 84.011 | 2702.345 | 2746.0 | 7.623 | 0.025617 | 0.025617 | 2.808138 |
| `alpha_grad` | ok | 244.836 | 84.157 | 160.679 | 2707.985 | 2786.0 | 7.308 | 0.819741 | 0.025617 | 2.808138 |

Focused regression checks after changing the default:

- strict SCM/reference subset: `13 passed`
- `alpha` TBPTT subset: `2 passed`
- `test_mlp_prior.py`: `9 passed`
- tiny CLI smoke:
  - `first_policy_gradient`: passed
  - `alpha_grad`: passed

Interpretation:

- This change materially improves risky-load throughput under forced full offload while staying inside the active GPU/host guards.
- It remains an experimental skyline because the partition change also changes strict-SCM execution grouping and therefore is not strictly forward-equivalent to the older baseline.
- Under the user’s current acceptance criterion, this acceleration is now the active skyline to optimize from.

### Enabled Robustness Default: `action` Adjoint Norm Clip = `1.0`

Scope:

- branch baseline: experimental skyline above (`2 GiB` strict-SCM partition, full policy offload forced on)
- default:
  - `first_policy_gradient_state_grad_clip_norm = 4.0` (unchanged)
  - `first_policy_gradient_action_grad_clip_value = 0.0` (disabled by default)
  - `first_policy_gradient_action_grad_clip_norm = 1.0`

Motivation:

- `g1` is highly sensitive to small forward execution perturbations.
- Under the accepted `2 GiB` partition skyline, the goal is to improve gradient robustness without materially changing risky-load throughput or memory.

Small deterministic robustness evidence, fixed `B=64, ns=128, sep=103`, full offload forced on:

#### `first_policy_gradient`

| setting | cosine(`256MB`,`2GB`) | grad norm default | grad norm `2GB` |
| --- | ---: | ---: | ---: |
| current default (`state=4, action_value=4, action_norm=0`) | `-0.3416` | `16994.23` | `17919.81` |
| candidate (`state=4, action_value=4, action_norm=1`) | `0.4573` | `4111.90` | `3938.35` |

#### `alpha_grad`

| setting | cosine(`256MB`,`2GB`) | grad norm default | grad norm `2GB` |
| --- | ---: | ---: | ---: |
| current default (`state=4, action_value=4, action_norm=0`) | `0.8974` | `39211.61` | `73233.91` |
| candidate (`state=4, action_value=4, action_norm=1`) | `0.8928` | `7326.14` | `11739.42` |

Risky-load impact, fixed `B=64, ns=1024, sep=697`, full offload forced on:

| objective | current wall (s) | candidate wall (s) | current peak alloc (MiB) | candidate peak alloc (MiB) |
| --- | ---: | ---: | ---: | ---: |
| `first_policy_gradient` | `171.641` | `173.836` | `2702.345` | `2702.361` |
| `alpha_grad` | `244.836` | `242.501` | `2707.985` | `2708.000` |

Interpretation:

- The candidate materially improves `first_pg` robustness to the accepted partition perturbation.
- `alpha` remains high-cosine and its risky-load wall/peak memory are effectively unchanged.
- It is kept as benchmark evidence, but not enabled by default because the combined speed/memory tradeoff improvement is too small.

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

## Accepted: reuse strict-SCM runtime views and fixed index tensors

Hypothesis:

- After caching the coarse runtime tensors, `transition_other` still retained a measurable shared-forward residue from:
  - rebuilding hidden weight/bias/noise view lists on every transition call
  - repeating fixed `.to(...)` conversions for `activation_codes`, `input_mask`, `state_mask`, and output gather indices
- These are semantics-invariant runtime preparations, so reusing them should be safe and should land entirely in shared forward.

Evidence before touching code:

- Risky-load `first_policy_gradient` profile after the previous accepted cache change:
  - `transition_generator_s ≈ 81.94s`
  - `transition_residual_s ≈ 15.38s`
  - `tensor_to_s ≈ 7.40s` over `538,204` calls
  - `rowwise_noise_plan_s ≈ 28.29s`
  - `hidden_update_grad_fused_s ≈ 23.81s`
- This made the remaining runtime-preparation churn the safest next shared-forward target.

Implementation:

- Extend the strict SCM `(device, dtype)` runtime cache to also reuse:
  - `hidden_weights_runtime`, `hidden_biases_runtime`, `hidden_noise_scales_runtime`
  - `input_mask_runtime`
  - `state_mask_runtime`
  - `state_select_idx_runtime`
  - `reward_select_idx_runtime`
- Replace repeated `activation_codes_t.to(...)` calls with the already-cached `activation_codes_runtime`
- Leave all stochastic paths, rollout order, `g0/g1`, and SCM math unchanged

Focused checks:

- `rowwise_scaled_noise_with_plan_matches_unplanned_on_cuda`
- `strict_reference_scm_hidden_update_grad_fused_matches_unfused_on_cuda`
- `alpha_grad_rollout_gradients_are_finite`
- `alpha_grad` TBPTT checkpoint subset
- `test_mlp_prior.py`

Risky-load evidence, same fixed overrides `B=64, ns=1024, sep=697, tbptt=32, paged, family, policy offload`:

| objective | before wall (s) | after wall (s) | delta | before peak alloc (MiB) | after peak alloc (MiB) | delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `first_policy_gradient` | 250.071 | 245.492 | -4.579 | 1113.661 | 1113.430 | -0.231 |
| `alpha_grad` | 339.817 | 332.320 | -7.497 | 1119.777 | 1119.777 | +0.000 |

In-situ risky-load split for `first_policy_gradient` after the change:

- `transition_generator_s: 81.944 -> 78.271s`
- `transition_residual_s: 15.376 -> 13.382s`
- `tensor_to_s: 7.40 -> 6.47s`
- `tensor_to_calls: 538,204 -> 360,120`

Interpretation:

- The measured win again lands exactly in shared forward, not in `g1` sink backward.
- GPU memory stays flat within noise.
- This is another semantics-safe strict-SCM runtime-preparation cleanup that benefits both `first_policy_gradient` and `alpha_grad`.

## Accepted: feed hidden-update grad-fused path prepacked int32 metadata

Hypothesis:

- After the previous accepted shared-forward cleanups, `hidden_update_grad_fused` still consumed a large block of `transition_generator` time.
- A focused risky-load probe showed a real internal prep head:
  - `hidden_update_grad_fused_s ≈ 17.56s`
  - inside it, `tensor_to_s ≈ 4.75s`
  - `clone_s ≈ 1.35s`
  - `contiguous_s ≈ 0.26s`
- If the strict SCM runtime cache also stores the `int32` metadata that this path needs, and the helper fast-path accepts already-`int32` contiguous inputs, the gain should land entirely in shared forward without touching SCM math.

Implementation:

- Extend the strict SCM `(device, dtype)` runtime cache with:
  - `activation_codes_runtime_i32`
  - `hidden_active_masks_runtime_i32`
  - `hidden_prefix_sizes_runtime_i32`
- Route the grad-fused hidden update call through those cached tensors
- Add a no-op fast-path in `_PrefixSampleInputActivatedAffineUpdateGradInputFn.forward(...)` when `active_mask`, `in_sizes`, `out_sizes`, and `activation_codes` already arrive as contiguous `int32`
- Leave kernels, rollout order, stochasticity, `g0/g1`, and SCM semantics unchanged

Focused checks:

- `rowwise_scaled_noise_with_plan_matches_unplanned_on_cuda`
- `strict_reference_scm_hidden_update_grad_fused_matches_unfused_on_cuda`
- `alpha_grad_rollout_gradients_are_finite`
- `alpha_grad` TBPTT checkpoint subset
- `test_mlp_prior.py`

Focused internal evidence on risky-load `first_policy_gradient`:

- before:
  - `hidden_update_grad_fused_s = 17.56s`
  - `hudf_tensor_to_s = 4.75s`
  - `hudf_contiguous_s = 0.26s`
  - `hudf_clone_s = 1.35s`
- after:
  - `hidden_update_grad_fused_s = 13.10s`
  - `hudf_tensor_to_s = 0.00s`
  - `hudf_contiguous_s = 0.00s`
  - `hudf_clone_s = 2.07s`

Risky-load evidence, same fixed overrides `B=64, ns=1024, sep=697, tbptt=32, paged, family, policy offload`:

| objective | before wall (s) | after wall (s) | delta | before peak alloc (MiB) | after peak alloc (MiB) | delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `first_policy_gradient` | 245.492 | 228.414 | -17.078 | 1113.430 | 1108.717 | -4.713 |
| `alpha_grad` | 332.320 | 328.151 | -4.169 | 1119.777 | 1114.904 | -4.873 |

In-situ risky-load split for `first_policy_gradient` after the change:

- `transition_generator_s: 78.271 -> 68.502s`
- `hidden_update_grad_fused_s: 23.254 -> 16.266s`
- `rowwise_noise_plan_s: 28.170 -> 26.469s`
- `policy_step_s: 52.645 -> 50.893s`

Interpretation:

- This is the largest accepted shared-forward win so far after the policy-offload baseline.
- The gain lands primarily in `hidden_update_grad_fused`, exactly where the probe predicted.
- GPU memory improves slightly; there is no host-memory regression.

## Accepted: prepack rowwise-noise runtime plan with active-scale slices

Hypothesis:

- After the accepted hidden-update metadata packing change, shared forward was dominated again by `rowwise_noise_plan`.
- Focused risky-load evidence showed that the biggest costs inside the helper were not the RNG itself:
  - `rowwise_total_s ≈ 25.87s`
  - `rowwise_mul_assign_s ≈ 13.58s`
  - `rowwise_any_s ≈ 4.68s`
  - `rowwise_randn_s ≈ 4.53s`
- So the safest next shared-forward optimization was to cache the deterministic parts of the rowwise plan:
  - whether the layer has any active noise at all
  - per-row active scale slices
- This preserves generator consumption order because the code still issues the exact same per-row `torch.randn(..., generator=g)` calls with the same shapes and in the same order.

Implementation:

- Add `_build_rowwise_scaled_noise_runtime_plan(...)`
- Extend the strict SCM `(device, dtype)` runtime cache with per-layer rowwise runtime plans containing:
  - `active`
  - `scale_slices`
  - `has_any`
  - `batch_size`
  - `width`
- Update `_sample_rowwise_scaled_noise_with_plan(...)` to consume that runtime plan directly
- Keep the original fallback path for the legacy tuple-form plan

Focused checks:

- `rowwise_scaled_noise_with_plan_matches_unplanned_on_cuda`
- `strict_reference_scm_hidden_update_grad_fused_matches_unfused_on_cuda`
- `alpha_grad_rollout_gradients_are_finite`
- `alpha_grad` TBPTT checkpoint subset
- `test_mlp_prior.py`

Risky-load evidence, same fixed overrides `B=64, ns=1024, sep=697, tbptt=32, paged, family, policy offload`:

| objective | before wall (s) | after wall (s) | delta | before peak alloc (MiB) | after peak alloc (MiB) | delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `first_policy_gradient` | 228.414 | 219.663 | -8.751 | 1108.717 | 1109.218 | +0.501 |
| `alpha_grad` | 328.151 | 324.985 | -3.166 | 1114.904 | 1115.174 | +0.270 |

In-situ risky-load split for `first_policy_gradient` after the change:

- `transition_generator_s: 68.502 -> 59.696s`
- `rowwise_noise_plan_s: 26.469 -> 15.853s`
- `hidden_update_grad_fused_s: 16.266s` unchanged within noise
- `policy_step_s: 53.203s` unchanged within noise

Interpretation:

- This is a semantics-safe shared-forward win that lands almost entirely in the expected `rowwise_noise_plan` hotspot.
- GPU memory remains flat within noise.
- The remaining top heads are now `g1 autograd.grad` on the alpha side and the still-large strict-SCM shared forward remainder, not rowwise plan bookkeeping itself.

## Accepted: auto-bypass policy saved-tensors offload when the risky load is already memory-safe

Hypothesis:

- After the accepted shared-forward optimizations, the strongest pushed baseline still paid a large runtime tax for `policy` saved-tensors CPU offload.
- Focused `g1 autograd.grad` profiling showed that, on the risky load we actually care about, the biggest remaining practical head was no longer strict SCM math:
  - with `policy` offload enabled, `cudaMemcpyAsync / cudaEventRecord / cudaStreamSynchronize` dominated self CPU time inside `g1`
  - the same risky load was already safe without policy offload, with peak GPU allocation only around `3.2 GiB`
- So the next highest-confidence change was not another math/kernel patch. It was to keep the existing `policy` offload mechanism, but automatically bypass it when the current CUDA free memory and load size are inside a conservative validated safe envelope.

Implementation:

- Add `_resolve_effective_policy_saved_tensors_offload(...)` in `train.py`
- Thread four new knobs through parser/config/train plumbing:
  - `pg_saved_tensors_cpu_offload_auto_disable_when_safe`
  - `pg_saved_tensors_cpu_offload_auto_min_free_gb`
  - `pg_saved_tensors_cpu_offload_auto_max_batch_size`
  - `pg_saved_tensors_cpu_offload_auto_max_n_samples`
- Keep the rollout math unchanged:
  - when the guard says "safe", policy-scope offload is bypassed for that batch
  - otherwise behavior falls back to the original offload path
- Set `rlpfn` defaults to:
  - `pg_saved_tensors_cpu_offload = true`
  - `pg_saved_tensors_cpu_offload_scope = policy`
  - `pg_saved_tensors_pin_memory = false`
  - `pg_saved_tensors_cpu_offload_auto_disable_when_safe = true`
  - `pg_saved_tensors_cpu_offload_auto_min_free_gb = 8.0`
  - `pg_saved_tensors_cpu_offload_auto_max_batch_size = 64`
  - `pg_saved_tensors_cpu_offload_auto_max_n_samples = 1024`

Focused checks:

- parser/default config subsets
- `_resolve_effective_policy_saved_tensors_offload(...)` helper tests
- strict SCM focused subset
- `alpha_grad` TBPTT subset
- `test_mlp_prior.py`

Risky-load evidence, same fixed overrides `B=64, ns=1024, sep=697, tbptt=32, paged, family`:

| objective | previous pushed baseline wall (s) | auto-bypass wall (s) | delta | previous pushed baseline peak alloc (MiB) | auto-bypass peak alloc (MiB) | delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `first_policy_gradient` | 219.663 | 118.865 | -100.798 | 1109.218 | 3259.454 | +2150.236 |
| `alpha_grad` | 324.985 | 154.503 | -170.482 | 1115.174 | 3263.593 | +2148.419 |

Additional safety evidence on the same runs:

- `first_policy_gradient`
  - `status = ok`
  - `peak_reserved = 3326 MiB`
  - `rss = 1.704 GiB`
- `alpha_grad`
  - `status = ok`
  - `peak_reserved = 3364 MiB`
  - `rss = 1.712 GiB`

Interpretation:

- This is the first change in this sequence that directly attacks the currently dominant runtime/offload scheduling tax instead of another strict-SCM micro-hotspot.
- The speedup is very large and lands on both `first_policy_gradient` and `alpha_grad`.
- GPU memory does rise relative to the pushed policy-offload baseline, but it remains far below the risky-load guard and far below card capacity, so the change stays within the validated safe envelope.
- Because the resolver only bypasses policy offload when the current load is inside that envelope, this remains a runtime policy change, not a training-semantics change.

## Accepted: terminal-on-latest-skyline preserves old auto-bypass semantics under true `256 MiB` partition

Scope:

- current branch head: `af774ab`
- old comparison point: `d974845`
- fixed risky-load harness:
  - same saved payload (`h_list`, `env_seeds`, `rollout_seeds`)
  - `B=64`
  - `ns=1024`
  - `sep=697`
  - `tbptt=32`
  - `paged`
  - `family`
  - `pg_saved_tensors_cpu_offload=true`
  - `pg_saved_tensors_cpu_offload_scope=policy`
  - `pg_saved_tensors_cpu_offload_auto_disable_when_safe=true`
- semantic alignment overrides:
  - `reference_scm_partition_max_bytes=256 MiB`
  - `terminal_reset_enabled=false`
  - `alpha_grad_local_coordinate_enabled=false`
  - `alpha_grad_unit_grad_enabled=false`
  - `first_policy_gradient_action_grad_clip_value=4`
  - `first_policy_gradient_action_grad_clip_norm=0`

Root cause and fix:

- The terminal branch initially looked much weaker than the old auto-bypass skyline, but the comparison was corrupted by a real override bug:
  - `EnvironmentPrior.__init__` ignored `cfg["reference_scm_partition_max_bytes"]` and only read the environment variable fallback
  - so "forced `256 MiB`" comparisons were silently still running at the `2 GiB` latest-skyline partition
- A second semantics bug was present in generic family rollout:
  - when terminal reset was disabled, `env_info["terminal_t"]` was still passed into `policy_step_fn`
  - this injected an all-zero terminal token into `first_policy_gradient` even though terminal was off
- After fixing both bugs, the old auto-bypass skyline and the current terminal branch align again on the same true `256 MiB` line.

Focused checks:

- `partition_budget_respects_config_when_env_unset`
- `partition_budget_env_override_wins`
- `family_rollout_omits_terminal_token_when_terminal_disabled`
- terminal-focused subset
- `py_compile`

Same-harness semantic comparison after the fix:

| objective | commit | wall (s) | peak alloc (MiB) | peak reserved (MiB) | objective | reward mean | reward std |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `first_policy_gradient` | `d974845` | 96.331 | 3207.932 | 3272.0 | 0.09917917 | 0.09917917 | 2.39829898 |
| `first_policy_gradient` | `2ce7b15` | 99.993 | 3207.939 | 3272.0 | 0.09917917 | 0.09917917 | 2.39829898 |
| `alpha_grad` | `d974845` | 129.577 | 3209.354 | 3308.0 | 3.17373276 | 0.09917917 | 2.39829898 |
| `alpha_grad` | `2ce7b15` | 128.492 | 3209.361 | 3308.0 | 3.17373276 | 0.09917917 | 2.39829898 |

Interpretation:

- Under the true old auto-bypass skyline settings, current terminal-on-latest-skyline is now semantically aligned with the old skyline for both `first_policy_gradient` and `alpha_grad`.
- Raw `af774ab` is **not** the true `256 MiB` compatibility point: before `2ce7b15`, the branch still ignored `reference_scm_partition_max_bytes`, so it actually stayed on the `2 GiB` memory line.
- The earlier apparent memory regression was not a new runtime cost that needed further compression; it was the partition-override bug.
- Once the line is forced back to true `256 MiB`, the extra memory disappears, so there is no additional semantics-safe memory reduction to apply on this old-skyline compatibility path.
- The current default `2 GiB` skyline remains a separate accepted speed-oriented line with intentionally different execution grouping and a different memory/time tradeoff.

## Alpha Grad Semantic Coverage Matrix

Scope:

- current branch head: `2ce7b15`
- default path under audit:
  - `rl_objective=alpha_grad`
  - `terminal_reset_enabled=true`
  - `alpha_grad_local_coordinate_enabled=true`
  - `alpha_grad_unit_grad_enabled=true`
  - `reference_scm_partition_max_bytes=2 GiB`
- compatibility line under audit:
  - terminal disabled
  - `local_coordinate=false`
  - `unit_grad=false`
  - `reference_scm_partition_max_bytes=256 MiB`

Feature inventory and certificates:

| feature | semantic contract | certificates / regression tests | latest scan |
| --- | --- | --- | --- |
| SCM / reference transition semantics | strict shared-vectorized rollout preserves SCM/reference builder semantics for the default environment contract | `test_environment_prior_strict_reference_semantics_shared_vectorized_rollout` | pass |
| Partition-budget control | config-level `reference_scm_partition_max_bytes` is respected, and env var override still wins when explicitly set | `test_environment_prior_partition_budget_respects_config_when_env_unset`, `test_environment_prior_partition_budget_env_override_wins` | pass |
| `g0` score path | reinforce log-prob score matches autograd gradient, and analytic `g0` matches the pre-change autograd path used by `alpha_grad` | `test_environment_prior_reinforce_log_prob_score_matches_autograd_gradient`, `test_environment_prior_alpha_grad_log_prob_score_matches_autograd_g0_path` | pass |
| Core `alpha_grad` mix math | `alpha_grad_loss_from_rollout_tensors(...)` matches the manual dense formula and returns stable finite gradients | `test_environment_prior_alpha_grad_matches_manual_action_space_mixing`, `test_environment_prior_alpha_grad_rollout_gradients_are_finite` | pass |
| Local-coordinate alpha | grouped traces mix in per-group local coordinates when enabled, and preserve legacy coarse semantics when disabled | `test_environment_prior_alpha_grad_group_traces_match_dense_manual_gradient`, `test_environment_prior_alpha_grad_group_traces_local_coordinate_matches_manual_group_local_gradient` | pass |
| Unit-grad alpha | alpha is computed from unit-normalized `g0/g1` blocks while raw gradients are still mixed back into the surrogate | `test_environment_prior_alpha_grad_matches_manual_action_space_mixing`, `test_environment_prior_alpha_grad_group_traces_local_coordinate_matches_manual_group_local_gradient` | pass |
| TBPTT semantic equivalence | alpha TBPTT full-window semantics match full horizon, and non-final family windows stay finite | `test_alpha_grad_tbptt_window_equal_horizon_matches_full_horizon_semantics`, `test_environment_prior_alpha_grad_tbptt_nonfinal_family_window_gradients_are_finite`, `test_alpha_grad_tbptt_family_vectorized_handles_nonfinal_window_case` | pass |
| Split encoder / policy split tokenization | default alpha config, action/state clip defaults, and terminal token expansion match the split encoder contract | `test_rlpfn_default_config_uses_split_encoder`, `test_rlpfn_default_terminal_reset_expands_split_obs_slots`, `test_tabpfn_forward_policy_step_split_matches_materialized_token_with_terminal` | pass |
| Parser / config plumbing | CLI and model defaults expose alpha/local/unit/terminal/partition options consistently | `test_rlpfn_parser_accepts_alpha_grad_objective`, `test_rlpfn_parser_accepts_alpha_grad_coordinate_and_unit_options`, `test_rlpfn_parser_defaults_enable_joint_env_and_budgeted_dims` | pass |
| Terminal tail rule | `D_t` uses two-sided tail selection, allows `X=0`, uses per-sample history after warmup, and requires a non-extreme first history hit before relaxing | `test_environment_prior_terminal_tail_event_from_signal_selects_two_sided_tails_from_history`, `test_environment_prior_terminal_tail_event_from_signal_allows_zero_count`, `test_environment_prior_terminal_tail_event_from_signal_uses_per_sample_history_not_batch`, `test_environment_prior_terminal_tail_event_from_signal_uses_batch_only_during_warmup`, `test_environment_prior_terminal_tail_event_from_signal_requires_non_extreme_history_hit_before_relaxing` | pass |
| Terminal rollout integration | terminal bonus/reset/token/stats propagate through single, structure, family, and alpha family-TBPTT paths | `test_environment_prior_rollout_policy_gradient_loss_reports_terminal_count_stats`, `test_environment_prior_rollout_with_policy_structure_grouping_matches_serial_with_terminal_reset`, `test_environment_prior_rollout_with_policy_family_grouping_matches_serial_with_terminal_reset`, `test_environment_prior_alpha_grad_family_tbptt_reports_terminal_count_stats` | pass |
| Terminal-disabled compatibility | when terminal is disabled, family rollout must not inject a zero terminal token into policy inputs | `test_environment_prior_family_rollout_omits_terminal_token_when_terminal_disabled` | pass |
| Old auto-bypass skyline compatibility | current terminal branch matches the old `d974845` auto-bypass skyline under true `256 MiB`, terminal-off, non-local, non-unit overrides | guarded same-payload benchmark table above | pass |

Consolidated semantic scan run on `2ce7b15`:

- `py_compile`:
  - `environment_prior.py`
  - `test_environment_prior.py`
  - `test_fit_model_parsing.py`
  - `test_rlpfn_split_encoder.py`
  - `test_train_policy_rollout_checkpoint.py`
- parser subset:
  - `pytest -q ticl/tests/test_fit_model_parsing.py -k 'alpha_grad or terminal or partition'`
  - result: `2 passed`
- split/default subset:
  - `pytest -q ticl/tests/test_rlpfn_split_encoder.py -k 'default or terminal or split'`
  - result: `5 passed`
- environment/terminal/alpha subset:
  - `pytest -q ticl/tests/priors/test_environment_prior.py -k 'alpha_grad or terminal or partition_budget or strict_reference_semantics_shared_vectorized_rollout or reinforce_log_prob_score_matches_autograd_gradient'`
  - result: `28 passed`
- alpha TBPTT checkpoint subset:
  - `pytest -q ticl/tests/test_train_policy_rollout_checkpoint.py -k 'alpha_grad_tbptt_window_equal_horizon_matches_full_horizon_semantics or alpha_grad_tbptt_family_vectorized_handles_nonfinal_window_case'`
  - result: `2 passed`
- MLP semantic baseline:
  - `pytest -q ticl/tests/priors/test_mlp_prior.py`
  - result: `9 passed`

Legacy certificate closure:

- `test_first_policy_gradient_tbptt_family_vectorized_matches_structure_backend_on_real_env_cfg` has been repaired by forcing the family-side rollout onto the explicit semantic A/B path with `batch_vectorized_strict_rng_match=true`.
- Focused regression after the repair:
  - `pytest -q ticl/tests/test_train_policy_rollout_checkpoint.py -k 'test_first_policy_gradient_tbptt_family_vectorized_matches_structure_backend_on_real_env_cfg or alpha_grad_tbptt_window_equal_horizon_matches_full_horizon_semantics or alpha_grad_tbptt_family_vectorized_handles_nonfinal_window_case or test_policy_rollout_nonreentrant_checkpoint_disables_policy_saved_tensors_offload or test_policy_rollout_reentrant_checkpoint_keeps_policy_saved_tensors_offload_enabled'`
  - result: `5 passed`
- This closes the previously documented non-default legacy `first_pg` checkpoint certificate gap.

## Retained Baseline: Direct TBPTT-Streaming Reproduction of the Forced Policy-Offload 2GiB Low-Memory Line

This is a later recovered/reproduced baseline, not the current top-of-file active skyline. It is kept here at the end of the document because it was established after the main skyline sequence and is primarily a retained reference line.

Scope:

- target commit: `070ff68`
- harness: [repro_forced_policy_offload_2g_firstpg.py](/home/chen/RLPFN/ticl/benchmarks/repro_forced_policy_offload_2g_firstpg.py)
- setup:
  - `pg_saved_tensors_cpu_offload=true`
  - `pg_saved_tensors_cpu_offload_scope=policy`
  - `pg_saved_tensors_cpu_offload_auto_disable_when_safe=false`
  - `reference_scm_partition_max_bytes=2 GiB`
  - `batch_size=64`
  - `n_samples=1024`
  - `single_eval_pos=697`
  - `kv_cache_mode=paged`
  - `family`
  - `terminal_reset_enabled=false`
  - `alpha_grad_local_coordinate_enabled=false`
  - `alpha_grad_unit_grad_enabled=false`
  - `first_policy_gradient_action_grad_clip_value=4`
  - `first_policy_gradient_action_grad_clip_norm=0`

Key correction:

- Earlier direct reproductions were accidentally running the workload as a full-rollout graph retention path.
- The real low-memory line uses the same TBPTT streaming semantics as training:
  - `pg_tbptt_window=32`
  - `tbptt_loss_sink` active
  - effective outer rollout checkpoint = `false`
- This is the smallest code-path difference that explains why earlier reproductions split into:
  - host-RSS blow-up on non-reentrant outer-checkpoint paths
  - higher GPU memory on reentrant outer-checkpoint paths

Guarded reproduction evidence on the corrected streaming harness:

| objective | payload status | teardown status | wall (s) | peak alloc (MiB) | peak reserved (MiB) | `nvidia-smi` process peak (MiB) | `nvidia-smi` total peak (MiB) | peak RSS sum (GiB) | streamed roots |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `first_policy_gradient` | `ok` | `rss_guard_kill` | 129.533 | 2115.753 | 2156.0 | 2162 | 2644 | 6.316 | 32 |
| `alpha_grad` | `ok` | `rss_guard_kill` | 189.745 | 2447.570 | 2520.0 | 2464 | 2946 | 6.040 | 32 |

Artifacts:

- [070ff68 first_pg summary](/home/chen/RLPFN/ticl/benchmarks/repro_forced_policy_offload_2g_firstpg_runs/070ff68_first_policy_gradient_ckpt1_reent1_mutable1_stream1.json)
- [070ff68 alpha summary](/home/chen/RLPFN/ticl/benchmarks/repro_forced_policy_offload_2g_firstpg_runs/070ff68_alpha_grad_ckpt1_reent1_mutable1_stream1.json)

Interpretation:

- The previously missing low-memory behavior was not hidden in terminal or newer `alpha` defaults.
- The missing piece was reproducing the correct execution semantics: TBPTT streaming with per-window backward, not an outer-checkpoint whole-rollout graph.
- Once that path is matched, both `first_pg` and `alpha_grad` fall back into the expected `~2-2.5 GiB alloc / ~2.6-3.0 GiB nvidia-smi` memory class.
- Under the current hard guard, both runs complete their JSON payload successfully and are then torn down by the RSS guard once the process-tree sum crosses `~6 GiB`; this keeps the machine safe while preserving the low-memory reproduction evidence.
