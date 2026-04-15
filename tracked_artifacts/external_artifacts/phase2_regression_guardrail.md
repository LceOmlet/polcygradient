# Phase 2 Regression Guardrail

## Status

Phase 2 is frozen for exploratory development.

Allowed work:

- maintain regression usability
- protect the trusted fixed-env contract
- reject regressions before Phase 3 changes are accepted

Disallowed work unless a regression fails first:

- new actor/critic semantic probes
- new critic rescue attempts
- architecture changes justified only by Phase 2 quick metrics

## Trusted Contract

The maintained Phase 2 numeric contract is:

- shared actor-critic backbone
- `ppo_reset_env_state_at_sep = True`
- `strict_fixed_env_mode = True`
- `deterministic_actor_sampling = True`
- `deterministic_batch_plan = True`
- `frozen_h_seed = 12345`
- `train_env_seed = 2020`
- `train_rollout_seed = 4040`
- `single_eval_pos = 64`
- `n_steps = 256`
- `batch_size = 256`
- `n_epochs = 1`
- `outer_epochs = 2`
- `learning_rate = 2e-4`
- `target_kl = 0.03`

This contract explicitly locks:

- env latent
- env seed
- rollout seed
- actor sampling
- minibatch order

## Important Legacy

These are the Phase 2 assets that must remain usable.

1. Actor-side SEP-reset gradient snapshot
   Path: `/home/chen/RLPFN/artifacts/legacy_frozen_h_seed12345_sep_state_reset_gradient_snapshot.json`
   Purpose: fixed legacy gradient-level comparison for the actor bug fix.

2. Official-vs-monkey PPO regression
   Path: `/home/chen/RLPFN/artifacts/critic_official_vs_monkey_regression_seed12345.json`
   Purpose: prove the maintained official strict path matches the legacy monkey implementation under the trusted contract.

3. Shared-backbone positive-gap positive-return milestone
   Path: `/home/chen/RLPFN/artifacts/critic_help_vs_zero_state_reset_shared_seed12345_strict_fixed_env_deterministic_quick.json`
   Purpose: maintained Phase 2 milestone for learned critic usefulness.

## Maintained Milestone

Milestone artifact:

- `/home/chen/RLPFN/artifacts/critic_help_vs_zero_state_reset_shared_seed12345_strict_fixed_env_deterministic_quick.json`

Reference values:

- `zero_suffix_return_mean = -0.5634000301`
- learned:
  - `delta_smp_gap = +0.7247717381`
  - `post_smp_gap = +0.6964260340`
  - `post_return = +0.1330260038`
- zero:
  - `delta_smp_gap = +0.1517906189`
  - `post_smp_gap = +0.1843308210`
  - `post_return = -0.3790692091`

## Regression Pack

Single entrypoint:

- `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase2_regression_suite.py`
- manifest:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase2_regression_manifest.json`

Default command:

```bash
cd /home/chen/RLPFN/reinforce-terminal-explore
python ticl/analysis/phase2_regression_suite.py
```

Fast assembly command:

```bash
cd /home/chen/RLPFN/reinforce-terminal-explore
python ticl/analysis/phase2_regression_suite.py \
  --reuse-canonical-official-monkey \
  --reuse-canonical-shared-milestone
```

Default output directory:

- `/home/chen/RLPFN/artifacts/phase2_regression_pack`

Generated outputs:

- `critic_official_vs_monkey_regression.json`
- `shared_backbone_milestone_quick.json`
- `phase2_regression_suite_summary.json`

Current assembled summary:

- `/home/chen/RLPFN/artifacts/phase2_regression_pack/phase2_regression_suite_summary.json`

Usage note:

- full rerun mode is the real regression path
- fast assembly mode exists to keep the guardrail pack usable without waiting for the shared milestone quick audit every time
- both modes still run manifest validation; fast mode is not exempt from numeric/timing/gradient checks
- the regression suite now defaults to reusing an existing valid summary instead of rerunning identical work
- the output directory is protected by a lock file to prevent concurrent overwrite
- validation failure is a hard failure by default; the suite exits non-zero unless explicitly overridden

## Accept Criteria

Phase 2 is considered intact only if all of the following hold:

1. Official-vs-monkey numeric regression still matches:
   - `policy_loss_abs_diff = 0.0`
   - `grad_l2_delta = 0.0`
   - rollout tensors remain exact-match

2. Shared milestone still holds:
   - learned `delta_smp_gap > zero delta_smp_gap`
   - learned `post_return > 0`

3. No new regression path replaces this contract with sampled or unfrozen comparisons.
4. Manifest validation remains green:
   - `validation.all_checks_pass = true`
   - no timing drift outside the allowed ratio band
   - no gradient or rollout-tensor drift outside the manifest tolerances

## Handoff Rule

Phase 3 work should proceed only after Phase 2 guardrails remain green.

If a Phase 3 change changes any Phase 2 result under this contract, treat that as a regression until proven otherwise.
