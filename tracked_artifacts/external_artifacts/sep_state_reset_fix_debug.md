# SEP State Reset Fix Debug Note

## 1. Bug summary

The original training semantics had a boundary mismatch at `single_eval_pos`:

- PPO switched the optimization phase from `explore` to `exploit`.
- The environment state did **not** switch to a new aligned boundary.
- As a result, the first exploit segment inherited pre-SEP environment state consequences.

This produced a credit-assignment mismatch:

- actor objective was computed on post-SEP tokens as if they were clean exploit tokens
- but those tokens still sat on top of state carried over from the explore phase

The most credible root cause is therefore:

- **pre-SEP environment state continuity crossing SEP**

This note records only the effective landed changes for that fix candidate.

## 2. Fix principle

Do the minimum thing required to align the optimization boundary with the environment boundary:

- at `step == single_eval_pos`
- reset **environment state only**
- do **not** reset actor hidden state
- do **not** clear action history
- do **not** clear reward/terminal history
- do **not** rewrite GAE or actor objective

This keeps the fix narrow and avoids introducing new shaping behavior.

## 3. Effective code changes

### 3.1 Config surface

The candidate fix is exposed as:

- `optimizer.ppo_reset_env_state_at_sep`

Current definition:

- [model_configs.py](/home/chen/RLPFN/reinforce-terminal-explore/ticl/model_configs.py)

Current default is `True` on the main training path. This is a temporary promotion based on the current strongest evidence, while broader long-run cross-seed confirmation continues.

### 3.2 Main training wiring

The training entrypoint now forwards the flag into PPO construction:

- [train.py](/home/chen/RLPFN/reinforce-terminal-explore/ticl/train.py)

Effective behavior:

- main training can now run with `reset_env_state_at_sep=True`
- logging includes:
  - `sep_state_reset`
  - `sep_resets`

### 3.3 PPO / env integration

The PPO builder and environment adapters now honor the flag:

- [sb3_recurrent_ppo.py](/home/chen/RLPFN/reinforce-terminal-explore/ticl/sb3_recurrent_ppo.py)
- [environment_prior.py](/home/chen/RLPFN/reinforce-terminal-explore/ticl/priors/environment_prior.py)

Effective behavior:

- at SEP, environment state is reset on the main training path
- actor history / hidden / action history remain unchanged
- both serial prior rollout and PPO VecEnv path support the same state-boundary semantics
- training logs expose whether a SEP reset happened

### 3.4 Main-path observability

The following observability was added:

- PPO training logger:
  - `train/sep_state_reset_enabled`
  - `train/sep_state_reset_count`
- env info dictionaries expose `sep_state_reset`
- console progress line includes:
  - `sep_state_reset`
  - `sep_resets`

This is required so long training runs can prove the fix is actually active.

### 3.5 Regression tests

Main-path regression tests were added here:

- [test_sb3_recurrent_ppo.py](/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_sb3_recurrent_ppo.py)

Coverage:

- PPO builder enables SEP state reset on the real training path
- VecEnv step path applies SEP state reset and reports it in `info`

## 4. Fixed legacy gradient regression snapshot

The canonical fixed-environment gradient snapshot for later numeric comparisons is:

- [legacy_frozen_h_seed12345_sep_state_reset_gradient_snapshot.json](/home/chen/RLPFN/artifacts/legacy_frozen_h_seed12345_sep_state_reset_gradient_snapshot.json)

It is produced by:

- [sep_state_reset_gradient_compare.py](/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/sep_state_reset_gradient_compare.py)

Frozen regression contract:

- `frozen_h_seed = 12345`
- `train_env_seed = 2020`
- `single_eval_pos = 64`
- `n_steps = 256`
- `batch_size = 256`
- `n_epochs = 1`
- `learning_rate = 2e-4`
- `target_kl = 0.03`

Reference values:

- normal `policy_loss = -0.26173797249794006`
- state-reset `policy_loss = -0.027393745258450508`
- gradient cosine similarity `= 0.065281443297863`
- gradient L2 delta norm `= 5.522181510925293`
- state-reset `sep_state_reset_count = 1`

Purpose:

- future formal training regressions should compare against this snapshot before changing SEP-boundary semantics

## 5. Long-run evidence currently supporting the candidate

One fixed long-run environment already showed a strong positive effect:

- `frozen_h_seed = 12345`
- baseline `delta_smp_gap = +1.4638671875`
- state-reset `delta_smp_gap = +45.134765625`

This is enough to keep the fix as the strongest mainline candidate.

It is **not yet** enough to flip the config default to `True`, because long-run cross-seed confirmation is still incomplete.

## 6. What this fix does not change

This fix does **not** do any of the following:

- no hidden-state reset at SEP
- no action-history clearing
- no reward/terminal-history clearing
- no actor-objective shaping
- no SEP-synced GAE rewrite
- no extra hyperparameters

This is deliberate. Those broader interventions were tested as probes and were not selected as the minimal mainline candidate.

## 7. Current status

Current status of `ppo_reset_env_state_at_sep`:

- wired into main training
- observable in logs and env info
- backed by fixed legacy gradient snapshot
- backed by one strong fixed-environment long-run win
- temporarily promoted to default on the main training path

Promotion criterion:

- keep or revert this default based on additional long-run `frozen_h_seed` evidence
