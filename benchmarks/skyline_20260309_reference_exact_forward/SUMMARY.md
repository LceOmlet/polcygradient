# Reference-Exact Forward Skyline (2026-03-09)

## Goal
- Recover forward throughput lost after aligning environment semantics to [`mlp.py`](/home/chen/RLPFN/ticl/ticl/priors/mlp.py) and [`fast_gp.py`](/home/chen/RLPFN/ticl/ticl/priors/fast_gp.py).
- Keep `rlpfn` defaults semantically exact for:
  - `SCM`: shared-activation `state/reward` sampling,
  - `GP`: exact finite-query first-call semantics.
- Treat semantic A/B as a hard gate for every optimization step.

## Optimization Order (largest expected gain first)
1. `SCM` strict/reference family-coarse transition:
   replace per-sample forward calls with one batched exact forward while preserving
   the same sampled weights, sampled feature indices, and sampled hidden noise.
   Status: implemented.
2. `GP` strict/reference family-coarse transition:
   batch exact query/covariance work without changing the exact finite-query semantics.
   Status: implemented.
3. Reference-aware subgrouping:
   split family groups into structure-compatible buckets only when that improves
   exact-transition throughput and still preserves rollout RNG order.
   Status: experimental, not default.
4. Small transition-side packing/launch reductions:
   remove remaining reference-path Python-side packing overhead after steps 1-3.
   Status: pending.

## Step 1

### Change
- Added a padded batched exact `SCM` reference builder:
  - [`environment_prior.py:1609`](/home/chen/RLPFN/ticl/ticl/priors/environment_prior.py:1609)
- The homogeneous strict/reference batch entry now uses that builder:
  - [`environment_prior.py:2054`](/home/chen/RLPFN/ticl/ticl/priors/environment_prior.py:2054)
- The default family-coarse hetero strict/reference `SCM` path now wraps the same
  batched exact core instead of looping over per-sample reference builders:
  - [`environment_prior.py:6871`](/home/chen/RLPFN/ticl/ticl/priors/environment_prior.py:6871)

### Semantic Gate
The step is accepted only if all of the following stay green:

- direct `SCM` exact-reference match against the `mlp.py`-style reference evaluator:
  - [`test_environment_prior.py:3071`](/home/chen/RLPFN/ticl/ticl/tests/priors/test_environment_prior.py:3071)
- family-coarse strict/reference `SCM` batch match against the same reference evaluator:
  - [`test_environment_prior.py:3106`](/home/chen/RLPFN/ticl/ticl/tests/priors/test_environment_prior.py:3106)
- direct `GP` exact-reference match against the `fast_gp.py`-style first-call evaluator:
  - [`test_environment_prior.py:3180`](/home/chen/RLPFN/ticl/ticl/tests/priors/test_environment_prior.py:3180)
- family-coarse strict/reference `GP` batch match against the same exact first-call evaluator:
  - [`test_environment_prior.py:3207`](/home/chen/RLPFN/ticl/ticl/tests/priors/test_environment_prior.py:3207)
- vectorized family-group rollout still matches serial rollout under strict/reference setup:
  - [`test_environment_prior.py:4069`](/home/chen/RLPFN/ticl/ticl/tests/priors/test_environment_prior.py:4069)

### Validation Command
```bash
CONDA_NO_PLUGINS=true conda run --no-capture-output -n rlpfn \
  python -m pytest -q ticl/ticl/tests/priors/test_environment_prior.py -k \
  'strict_reference_scm_family_coarse_batch_matches_reference_builder or \
    strict_reference_gp_family_coarse_batch_matches_reference_builder or \
    strict_reference_scm_joint_transition_matches_reference_builder or \
    strict_reference_gp_joint_transition_matches_exact_first_call or \
    family_grouping_matches_serial_in_deterministic_setup_with_strict_joint_transition'
```

Observed result on this step:
- `5 passed`

## Step 1 Microbenchmark

### Scope
- Device: `cuda`
- Batch: `16`
- Path: strict/reference `SCM` family-coarse transition only
- Compare:
  - old behavior: per-sample loop over `_build_reference_scm_joint_transition_fn`
  - new behavior: batched exact transition from the new padded builder

### Result
- `fast_ms_per_iter`: `5.408`
- `slow_ms_per_iter`: `6.631`
- speedup: `1.226x`
- semantic diff:
  - `max_state_diff = 2.98e-08`
  - `max_reward_diff = 1.49e-08`

Interpretation:
- Step 1 is a real forward optimization on the default exact-reference `SCM` path.
- The speedup is moderate but clean: it comes from removing Python-side per-sample
  transition calls, not from weakening semantics.

## Notes
- `SCM` and `GP` exact family-coarse forward paths are now both batched.
- Exact-reference acceptance is test-gated, not benchmark-gated.
- Remaining large opportunities are now:
  - reference-aware subgrouping,
  - smaller transition packing/launch reductions after subgrouping.

## Step 2

### Change
- Added a padded batched exact `GP` reference builder with:
  - per-sample cache state,
  - per-sample stable Cholesky jitter,
  - per-sample RNG consumption that only advances on active output dimensions.
- New entry points:
  - independent Cholesky stabilizer: [`environment_prior.py:2119`](/home/chen/RLPFN/ticl/ticl/priors/environment_prior.py:2119)
  - batched exact `GP` builder: [`environment_prior.py:2237`](/home/chen/RLPFN/ticl/ticl/priors/environment_prior.py:2237)
  - homogeneous strict/reference batch entry: [`environment_prior.py:2451`](/home/chen/RLPFN/ticl/ticl/priors/environment_prior.py:2451)
  - family-coarse hetero strict/reference `GP` path: [`environment_prior.py:7165`](/home/chen/RLPFN/ticl/ticl/priors/environment_prior.py:7165)

### Semantic Gate
- direct `GP` exact first-call match:
  - [`test_environment_prior.py:3180`](/home/chen/RLPFN/ticl/ticl/tests/priors/test_environment_prior.py:3180)
- family-coarse strict/reference `GP` first-call batch match:
  - [`test_environment_prior.py:3207`](/home/chen/RLPFN/ticl/ticl/tests/priors/test_environment_prior.py:3207)
- family-coarse strict/reference `GP` repeated identical-query cache semantics:
  - [`test_environment_prior.py:3265`](/home/chen/RLPFN/ticl/ticl/tests/priors/test_environment_prior.py:3265)
- family-coarse strict/reference `GP` multi-step cache semantics against the old
  per-sample exact builder:
  - [`test_environment_prior.py:3315`](/home/chen/RLPFN/ticl/ticl/tests/priors/test_environment_prior.py:3315)
- strict/reference vectorized family-group rollout still matches serial rollout:
  - [`test_environment_prior.py:4191`](/home/chen/RLPFN/ticl/ticl/tests/priors/test_environment_prior.py:4191)

### Validation Command
```bash
CONDA_NO_PLUGINS=true conda run --no-capture-output -n rlpfn \
  python -m pytest -q ticl/ticl/tests/priors/test_environment_prior.py -k \
  'strict_reference_gp_joint_transition_matches_exact_first_call or \
    strict_reference_gp_family_coarse_batch_matches_reference_builder or \
    strict_reference_gp_family_coarse_batch_repeats_identical_query_without_noise or \
    strict_reference_gp_family_coarse_batch_multistep_matches_slow_builder or \
    strict_reference_gp_joint_transition_repeats_identical_query_without_noise or \
    family_grouping_matches_serial_in_deterministic_setup_with_strict_joint_transition'
```

Observed result on this step:
- `6 passed`

### Step 2 Microbenchmark

#### Scope
- Device: `cuda`
- Batch: `12`
- Sequence length: `12`
- Path: strict/reference `GP` family-coarse transition only
- Compare:
  - old behavior: per-sample loop over `_build_reference_gp_joint_transition_fn`
  - new behavior: batched exact cache/query transition from the new padded builder

#### Result
- `fast_ms_per_seq`: `104.147`
- `slow_ms_per_seq`: `303.787`
- speedup: `2.917x`
- semantic diff over the benchmarked sequence:
  - `max_state_diff = 2.22e-4`
  - `max_reward_diff = 2.06e-4`

Interpretation:
- The speedup is substantial and comes from batching exact query/covariance work.
- The benchmark diff is not bitwise-zero on GPU because batched linear algebra changes
  floating-point reduction order.
- The formal semantic gate for this step is the reference test suite above, especially
  the multi-step cache parity test on CPU and the strict serial-vs-vectorized rollout test.

## Step 3

### Change
- Probed reference-aware `structure` bucketing as an exact-reference execution-mode
  candidate:
  - [`environment_prior.py:1060`](/home/chen/RLPFN/ticl/ticl/priors/environment_prior.py:1060)
  - [`environment_prior.py:10206`](/home/chen/RLPFN/ticl/ticl/priors/environment_prior.py:10206)
- Added a small-bucket fallback threshold so tiny structure buckets can merge back
  to family-coarse execution instead of over-fragmenting the rollout hot loop:
  - [`environment_prior.py:1070`](/home/chen/RLPFN/ticl/ticl/priors/environment_prior.py:1070)
  - [`environment_prior.py:10209`](/home/chen/RLPFN/ticl/ticl/priors/environment_prior.py:10209)

This step changes only how heterogeneous transition groups are partitioned before
exact execution. It does not change sampled `SCM` / `GP` parameters, rollout RNG,
or the `mlp.py` / `fast_gp.py` reference semantics themselves.

### Semantic Gate
- default grouping/min-bucket regression:
  - [`test_environment_prior.py:4721`](/home/chen/RLPFN/ticl/ticl/tests/priors/test_environment_prior.py:4721)
- explicit structure bucketing still splits as expected when `min_bucket=0`:
  - [`test_environment_prior.py:4635`](/home/chen/RLPFN/ticl/ticl/tests/priors/test_environment_prior.py:4635)
- structure bucketing matches family bucketing on rollout, loss, and gradients:
  - [`test_environment_prior.py:4719`](/home/chen/RLPFN/ticl/ticl/tests/priors/test_environment_prior.py:4719)
- small-bucket fallback preserves family semantics:
  - [`test_environment_prior.py:4810`](/home/chen/RLPFN/ticl/ticl/tests/priors/test_environment_prior.py:4810)
- strict-reference `SCM` still matches the `mlp.py`-style evaluator:
  - [`test_environment_prior.py:3071`](/home/chen/RLPFN/ticl/ticl/tests/priors/test_environment_prior.py:3071)
- strict-reference `GP` still matches the `fast_gp.py`-style exact first-call evaluator:
  - [`test_environment_prior.py:3180`](/home/chen/RLPFN/ticl/ticl/tests/priors/test_environment_prior.py:3180)
- strict-reference vectorized family-group rollout still matches serial rollout:
  - [`test_environment_prior.py:4191`](/home/chen/RLPFN/ticl/ticl/tests/priors/test_environment_prior.py:4191)

### Validation Command
```bash
CONDA_NO_PLUGINS=true conda run --no-capture-output -n rlpfn \
  python -m pytest -q ticl/ticl/tests/priors/test_environment_prior.py -k \
  'transition_inner_grouping_defaults_to_structure_with_min_bucket_two or \
    rollout_with_policy_family_grouping_uses_transition_structure_buckets or \
    transition_structure_bucketing_matches_family_bucket_semantics or \
    transition_min_bucket_merges_small_buckets_back_to_family or \
    strict_reference_scm_joint_transition_matches_reference_builder or \
    strict_reference_gp_joint_transition_matches_exact_first_call or \
    family_grouping_matches_serial_in_deterministic_setup_with_strict_joint_transition'
```

Observed result on this step:
- `7 passed`

### Notes
- Semantic gate stayed green, but this probe is not kept as the maintained default.
- The maintained default remains the trusted coarse path until a subgrouping mode
  is both semantically clean and performance-confirmed on the exact-reference line.
- A fresh GPU microbenchmark was not added in this turn because the current
  sandboxed validation environment does not expose stable CUDA benchmarking.
