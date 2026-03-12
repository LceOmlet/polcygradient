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
