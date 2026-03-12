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
