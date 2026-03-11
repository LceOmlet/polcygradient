# Alpha Grad Skyline 2026-03-12

## Setup

- Worktree: `/tmp/rlpfn_alpha_grad_20260311`
- Branch: `experiment/alpha-grad-20260311`
- Hardware: single `NVIDIA GeForce RTX 4090`
- Guard rails:
  - kill if host available memory `< 10 GiB`
  - kill if process RSS `> 24 GiB`
  - kill if process GPU memory `> 18 GiB`
- Fixed risky load:
  - `batch_size = 64`
  - `n_samples = 1024`
  - `single_eval_pos = 697`
  - `tbptt_window = 32`
  - `validate = false`
  - `rl_validate_enabled = false`
  - `seed_everything = true`

## Baseline

### Reinforce

- Status: `ok`
- Log: `/tmp/alpha_guard_reinforce_b64_ns1024_sep697_20260312.log`
- Train profiler: `/tmp/rlpfn_alpha_grad_20260311/alpha_guard_reinforce_b64_ns1024_sep697_20260312.train_profiler.jsonl`
- Batch metrics:
  - `rollout_s = 51.022`
  - `backward_s = 19.395`
  - `batch_wall_excl_compile_s = 70.417`
  - `peak_alloc_gib = 1.859`
  - `peak_reserved_gib = 1.887`

### First Policy Gradient

- Status: `ok`
- Summary: `/tmp/alpha_guard_firstpg_b64_ns1024_sep697_20260312.summary.json`
- Log: `/tmp/alpha_guard_firstpg_b64_ns1024_sep697_20260312.log`
- Train profiler: `/tmp/rlpfn_alpha_grad_20260311/alpha_guard_firstpg_b64_ns1024_sep697_20260312.train_profiler.jsonl`
- Batch metrics:
  - `rollout_s = 194.850`
  - `backward_s = 104.868`
  - `batch_wall_excl_compile_s = 299.718`
  - `peak_alloc_gib = 4.712`
  - `peak_reserved_gib = 4.775`
  - `gpu_peak_mib = 5364`
  - `gpu_util_avg = 91.09`
  - `gpu_util_max = 99`

### Alpha Grad Current Accepted Implementation

- Status: `ok`
- Summary: `/tmp/alpha_guard_alphagrad_b64_ns1024_sep697_20260312.summary.json`
- Log: `/tmp/alpha_guard_alphagrad_b64_ns1024_sep697_20260312.log`
- Train profiler: `/tmp/rlpfn_alpha_grad_20260311/alpha_guard_alphagrad_b64_ns1024_sep697_20260312.train_profiler.jsonl`
- Batch metrics:
  - `rollout_s = 356.183`
  - `backward_s = 91.876`
  - `batch_wall_excl_compile_s = 448.059`
  - `peak_alloc_gib = 4.807`
  - `peak_reserved_gib = 4.863`
  - `gpu_peak_mib = 5454`
  - `gpu_util_avg = 91.84`
  - `gpu_util_max = 99`

## Diagnosis

- Current dominant alpha-specific slowdown is in `alpha_grad_loss_from_rollout_tensors(...)`, not in shared rollout/SCM forward.
- Risky-load guarded diag showed:
  - `action_trace_kind = group_traces`
  - `group_trace_count = 1`
  - `group_root_tensor_count = 32` per TBPTT window
  - `autograd.grad_calls = 2` per alpha window
  - almost all alpha-specific wall time was inside those `autograd.grad` calls
- This ruled out grouped merge as the main runtime hotspot at the risky load.

## Optimization Steps

### Action Trace Memory Trim

- Commit: `6b88791`
- Change:
  - removed redundant dense `action_mean` materialization for grouped alpha traces
  - stopped stacking tuple roots inside alpha loss when not needed
  - stopped materializing redundant full-batch `policy_action_mean_root_steps` when alpha already keeps grouped roots only
- Effect:
  - reduced alpha internal trace memory
  - reduced micro-benchmark CUDA peak from `11.31 MiB` to `10.37 MiB`

### Rejected: Batched VJP For `g0/g1`

- Status: rejected
- Log: `/tmp/alpha_guard_alphagrad_b64_ns1024_sep697_after_batchedvjp_20260312.log`
- Train profiler: `/tmp/rlpfn_alpha_grad_20260311/alpha_guard_alphagrad_b64_ns1024_sep697_after_batchedvjp_20260312.train_profiler.jsonl`
- Result:
  - `rollout_s = 365.055`
  - `backward_s = 94.938`
  - `batch_wall_excl_compile_s = 459.993`
  - `peak_alloc_gib = 4.849`
- Reason:
  - mathematically correct
  - slower than the accepted implementation
  - not worth keeping

### Rejected: Single-Group Dense Action Fastpath

- Status: rejected
- Log: `/tmp/alpha_guard_alphagrad_b64_ns1024_sep697_after_densefastpath_20260312.log`
- Train profiler: `/tmp/rlpfn_alpha_grad_20260311/alpha_guard_alphagrad_b64_ns1024_sep697_after_densefastpath_20260312.train_profiler.jsonl`
- Result:
  - `rollout_s = 187.325`
  - `backward_s = 99.166`
  - `batch_wall_excl_compile_s = 286.491`
  - `peak_alloc_gib = 4.827`
  - but `grad_norm = 0` and `grad_zero_share = 1.000`
- Reason:
  - the forwarded dense action trace was not an upstream differentiable root for the loss
  - it silently zeroed alpha gradients
  - this is a semantic regression and must not be used

## Semantic Checks

- Alpha env tests: pass
- Alpha TBPTT checkpoint test: pass
- Strict SCM builder baseline: pass
- `mlp.py` prior tests: pass

## Notes

- This document intentionally tracks only guarded runs that keep host memory and GPU memory within explicit thresholds.
- Risky-load comparisons use one batch to expose subtle regressions without pushing the machine into swap or editor-lock territory.
