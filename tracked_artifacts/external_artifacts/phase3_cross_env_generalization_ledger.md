# Phase 3 Cross-Environment Generalization Ledger

## Scope

This ledger tracks only Phase 3:

- cross-environment generalization
- fixed-suite baselines
- held-out prior-environment performance

It intentionally excludes:

- further critic-rescue experiments
- additional boundary-shaping probes
- actor/critic architecture debugging

Those lines remain recorded in:

- [semantic_audit_plan.md](/home/chen/RLPFN/artifacts/semantic_audit_plan.md)
- [critic_fit_debug.md](/home/chen/RLPFN/artifacts/critic_fit_debug.md)
- [sep_state_reset_fix_debug.md](/home/chen/RLPFN/artifacts/sep_state_reset_fix_debug.md)

## Current Preconditions

Accepted mainline state before Phase 3:

- `SEP state reset` is the current mainline repair candidate
- critic rescue is paused
- cross-environment generalization should be judged from fixed suites and matched controls, not from ad hoc quick probes

## Reusable Driver

Phase 3 long-run evaluation is now packaged into a single driver:

- `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_longrun_driver.py`

The driver performs three steps under one command:

1. continue training from a warm-start checkpoint
2. select the latest produced checkpoint
3. evaluate that checkpoint on Phase 3 pair1 and pair2 fixed suites, then compare against the saved zero controls

Outputs are written under the requested `base_path`:

- `phase3_pair1_ppo.json`
- `phase3_pair2_ppo.json`
- `phase3_longrun_summary.json`

Important boundary:

- this driver currently uses `fit_model.py` continue-run semantics
- so training config is inherited from the warm-start checkpoint, then merged with current missing defaults
- it is therefore **not** a fully isolated “Phase 3 only” training benchmark
- in particular, if the checkpoint carries auxiliary PPO-era training terms, those remain active unless explicitly overridden

## Baseline Definition

The baseline for Phase 3 is:

- checkpoint:
  - `/home/chen/RLPFN/rwkv/models_diff/rlpfn_03_30_2026_22_37_53_epoch_13.cpkt`
- audit entry:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/prior_generalization_audit.py`
- regime:
  - `cross_env`
- backend:
  - `serial`
- fixed suites:
  - train suite seed `12345`
  - heldout suite seed `67890`
  - train batch size `16`
  - heldout batch size `16`

Required paired reports:

1. `ppo` policy on fixed suites
2. `zero` policy on the exact same fixed suites

Interpretation rule:

- no claim of usable cross-environment generalization unless `ppo` clearly beats `zero` on held-out suites under the same saved suites

## Active Baseline Run

PPO baseline artifact target:

- `/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/ppo_cross_env_baseline.json`

Saved suite directory:

- `/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites`

Completed:

- matched `zero` control on the saved suites
- first baseline summary

## Process Notes

- Fixed suites have already been materialized and validated:
  - `/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/train_suite.pt`
  - `/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/heldout_suite.pt`
- Suite structure is correct for the intended fixed-suite compare:
  - `train`: `batch_size=16`, `suite_seed=12345`
  - `heldout`: `batch_size=16`, `suite_seed=67890`
- The first canonical PPO rerun initially failed for an environment reason, not an audit-logic reason:
  - RWKV official CUDA extension required `ninja` on `PATH`
  - this was corrected by running with:
    - `PATH=/home/liangchen/miniconda3/envs/rlpfn/bin:$PATH`
- After fixing `PATH`, canonical runs were restarted on the already-saved suites:
  - `ppo` on fixed suites
  - matched `zero` control on the exact same suites

Current status:

- suite creation: complete
- suite validation: complete
- canonical `ppo` result: complete
- canonical `zero` result: complete

Run hygiene note:

- an accidental duplicate `ppo` baseline process was detected during rerun
- the duplicate suite-path `ppo` process was terminated
- the original canonical `ppo` run and the matched `zero` control were kept alive

## First Baseline Summary

Artifacts:

- PPO:
  - `/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/ppo_cross_env_baseline.json`
- zero control:
  - `/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/zero_cross_env_control.json`

Fixed-suite config actually used by both reports:

- `suite_regime = cross_env`
- `rollout_backend = serial`
- `n_samples = 2048`
- `single_eval_pos = 1946`
- suffix horizon `= 102`

PPO means:

- train full return mean: `-13.5057`
- train suffix return mean: `-0.5712`
- heldout full return mean: `+9.3046`
- heldout suffix return mean: `+0.2338`

zero means:

- train full return mean: `-1.1161`
- train suffix return mean: `-0.0542`
- heldout full return mean: `-49.7059`
- heldout suffix return mean: `-2.5785`

Direct `ppo - zero` deltas:

- train full return mean delta: `-12.3896`
- train suffix return mean delta: `-0.5170`
- heldout full return mean delta: `+59.0105`
- heldout suffix return mean delta: `+2.8123`

Interpretation:

- this first fixed-suite baseline does show positive cross-environment evidence
- PPO is materially better than zero on the heldout suite
- PPO is worse than zero on the train suite for this same baseline
- so the current evidence is:
  - heldout generalization signal: positive
  - train-side dominance over zero: not established
  - promotion beyond a first baseline: not yet justified

## Promotion Gate

Do not promote any cross-environment claim until all of the following hold:

1. fixed-suite `ppo` and `zero` runs are both present
2. held-out suffix return for `ppo` is materially better than `zero`
3. train-heldout gap is not explained by suite artifacts alone

## Next Actions

1. Add one more paired fixed-suite baseline under the same protocol to reduce suite-pair luck further.
2. Or hold suite seeds fixed and compare a second checkpoint under the exact same audit protocol.
3. Keep the same decision rule: `ppo` must beat `zero` on heldout suites from identical saved suites.
4. Only after one additional confirmation axis should Phase 3 be promoted beyond “repeatable positive baseline evidence”.

## Second Fixed-Suite Pair

Second suite pair in progress:

- train suite seed `24680`
- heldout suite seed `13579`
- suite dir:
  - `/home/chen/RLPFN/artifacts/phase3_cross_env_baseline_pair2/suites`

Current artifacts:

- zero control complete:
  - `/home/chen/RLPFN/artifacts/phase3_cross_env_baseline_pair2/zero_cross_env_control.json`
- PPO baseline:
  - `/home/chen/RLPFN/artifacts/phase3_cross_env_baseline_pair2/ppo_cross_env_baseline.json`
  - complete

Second-pair zero summary:

- train full return mean: `-26.2392`
- train suffix return mean: `-0.9983`
- heldout full return mean: `-60.2048`
- heldout suffix return mean: `-3.4085`

Second-pair PPO summary:

- train full return mean: `+40.5695`
- train suffix return mean: `+2.2064`
- heldout full return mean: `-34.0368`
- heldout suffix return mean: `-1.8627`

Second-pair direct `ppo - zero` deltas:

- train full return mean delta: `+66.8087`
- train suffix return mean delta: `+3.2047`
- heldout full return mean delta: `+26.1680`
- heldout suffix return mean delta: `+1.5458`

Repeatability update:

- heldout win does repeat on the second fixed-suite pair
- first pair heldout suffix delta: `+2.8123`
- second pair heldout suffix delta: `+1.5458`
- this is now repeated positive evidence for heldout cross-environment advantage over zero
- the train-side story is mixed across pairs:
  - pair 1: PPO worse than zero on train
  - pair 2: PPO better than zero on train

Current Phase 3 conclusion:

- the heldout advantage is now repeatable across at least two fixed-suite pairs
- Phase 3 has cleared the minimal “single-pair artifact” concern
- broader promotion still requires at least one more axis of confirmation, such as:
  - another suite-seed pair
  - or another checkpoint under the same audit protocol

## Checkpoint-Dependence Follow-up

Current checkpoint-dependence probe:

- reference checkpoint:
  - `/home/chen/RLPFN/rwkv/models_diff/rlpfn_03_30_2026_22_37_53_epoch_13.cpkt`
- comparison checkpoint:
  - `/home/chen/RLPFN/rwkv/models_diff/rlpfn_03_30_2026_22_37_53_epoch_15.cpkt`

Rationale:

- same training lineage
- nearest available neighboring checkpoint
- best choice for isolating whether the heldout generalization signal depends on the specific checkpoint

Current run state:

- epoch 15 on fixed-suite pair 1:
  - output target:
    - `/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/ppo_epoch15_on_pair1.json`
  - status:
    - complete

epoch-15 on pair-1 summary:

- train full return mean: `-13.0434`
- train suffix return mean: `-0.5489`
- heldout full return mean: `+9.7629`
- heldout suffix return mean: `+0.2466`

epoch-15 vs pair-1 zero control:

- train full return mean delta: `-11.9273`
- train suffix return mean delta: `-0.4947`
- heldout full return mean delta: `+59.4688`
- heldout suffix return mean delta: `+2.8251`

Checkpoint-dependence update:

- the heldout win is preserved when moving from `epoch_13` to `epoch_15` on the same fixed suite pair
- epoch 13 heldout suffix delta on pair 1: `+2.8123`
- epoch 15 heldout suffix delta on pair 1: `+2.8251`
- this makes the current heldout signal look less checkpoint-specific than before
- train-side behavior remains similar across the two checkpoints on pair 1:
  - both are worse than zero on train for this suite pair

epoch-15 on pair-2 summary:

- artifact:
  - `/home/chen/RLPFN/artifacts/phase3_cross_env_baseline_pair2/ppo_epoch15_on_pair2.json`
- train full return mean: `+41.0650`
- train suffix return mean: `+2.2341`
- heldout full return mean: `-33.2365`
- heldout suffix return mean: `-1.8257`

epoch-15 vs pair-2 zero control:

- train full return mean delta: `+67.3042`
- train suffix return mean delta: `+3.2324`
- heldout full return mean delta: `+26.9682`
- heldout suffix return mean delta: `+1.5828`

Checkpoint x suite combined update:

- heldout win survives on:
  - pair 1 with `epoch_13`
  - pair 1 with `epoch_15`
  - pair 2 with `epoch_13`
  - pair 2 with `epoch_15`
- this is now repeated positive evidence across:
  - two fixed suite pairs
  - two neighboring checkpoints from the same training lineage

Upgraded Phase 3 conclusion:

- the current heldout cross-environment advantage is no longer explained well by:
  - a single suite-pair artifact
  - or a single-checkpoint artifact
- Phase 3 now has repeated positive signal across suite and checkpoint axes
- broader promotion beyond this still benefits from one more independent axis, but the minimal repeatability bar has been cleared

Decision rule remains unchanged:

- do not conclude checkpoint robustness until the epoch-15 run is complete and compared against the already-saved zero control on the exact same suites

## Longrun alignment check

Question checked:

- whether the current `phase3_longrun_driver.py --phase3-isolated` training command is still semantically aligned with the existing Phase 3 short-run benchmark lineage

Important distinction:

- the short-run Phase 3 baseline is an evaluation-only benchmark on fixed suites
- the longrun driver is a continue-training benchmark followed by the same fixed-suite evaluation
- so only the evaluation protocol can be expected to align exactly; training semantics must be checked separately

Evaluation-protocol alignment:

- aligned items:
  - fixed suite regime: `cross_env`
  - rollout backend: `serial`
  - `n_samples = 2048`
  - `single_eval_pos = 1946`
  - pair-1 / pair-2 fixed suite paths

Training-semantics alignment check:

- artifact:
  - `/home/chen/RLPFN/artifacts/phase3_longrun_gradient_compare.json`
- checkpoint:
  - `/home/chen/RLPFN/rwkv/models_diff/rlpfn_03_30_2026_22_37_53_epoch_13.cpkt`
- fixed env:
  - `frozen_h_seed = 12345`
  - `train_env_seed = 2020`
  - `rollout_seed = 2020`
  - `single_eval_pos = 1946`
  - `n_steps = 2048`

Gradient-compare result:

- relevant training signature diff:
  - empty
- `exact_match = true`
- total-loss gradient cosine similarity:
  - `1.00015`
- gradient delta L2 norm:
  - `0.00415`

Interpretation:

- for this checkpoint lineage, `--phase3-isolated` is effectively a no-op on the resolved PPO training semantics
- the previous mismatch concern was not that `phase3-isolated` changed PPO behavior; it was that the longrun command still remains a continue-training benchmark rather than a from-scratch Phase 3 benchmark
- `train/value_loss` remains expected
- `next_state_flow_matching_weight = 0.2` also remains active in both modes because it is structural for this checkpoint lineage

## Short-run-aligned training surrogate gradient check

Question checked:

- if Phase 3 short-run `ppo` evaluation is treated as the reference behavior, what training surrogate is maximally aligned to it, and how far is that surrogate from the current continue-run training semantics?

Short-run-aligned surrogate definition:

- actor-only PPO surrogate
- `ppo_actor_baseline_mode = zero`
- `ppo_normalize_advantage = false`
- `ppo_vf_coef = 0.0`
- `ppo_reset_env_state_at_sep = false`
- `prior.environment.next_state_flow_matching_weight = 0.0`
- `prior.environment.normalized_q_value_weight = 0.0`

Rationale:

- Phase 3 short-run `ppo` benchmark is evaluation-only
- it uses the PPO actor policy to choose actions on fixed suites
- it does not execute critic training or aux-loss training
- therefore a strict training-side surrogate must remove critic and aux contributions rather than pretending the short-run benchmark had training gradients

Artifact:

- `/home/chen/RLPFN/artifacts/phase3_shortrun_aligned_gradient_compare.json`

Fixed compare setting:

- checkpoint:
  - `/home/chen/RLPFN/rwkv/models_diff/rlpfn_03_30_2026_22_37_53_epoch_13.cpkt`
- `frozen_h_seed = 12345`
- `train_env_seed = 2020`
- `rollout_seed = 2020`
- `single_eval_pos = 1946`
- `n_steps = 2048`

Result:

- training signature diff is not empty
- changed items:
  - `optimizer.ppo_actor_baseline_mode`: `learned -> zero`
  - `optimizer.ppo_normalize_advantage`: `true -> false`
  - `optimizer.ppo_reset_env_state_at_sep`: `true -> false`
  - `optimizer.ppo_vf_coef`: `0.5 -> 0.0`
  - `prior.environment.next_state_flow_matching_weight`: `0.2 -> 0.0`
- total-loss gradient cosine similarity:
  - `0.09327`
- gradient delta L2 norm:
  - `880.618`

Interpretation:

- the current continue-run longrun training semantics are not close to the Phase 3 short-run-aligned actor-only surrogate
- so it is incorrect to describe the current longrun training command as “fully aligned” to the short-run Phase 3 benchmark
- what *is* aligned is only the fixed-suite evaluation protocol

Implementation update:

- `phase3_longrun_driver.py` now supports:
  - `--phase3-train-profile shortrun_aligned`
- this training profile applies:
  - `ppo_actor_baseline_mode = zero`
  - `ppo_normalize_advantage = false`
  - `ppo_vf_coef = 0.0`
  - `ppo_reset_env_state_at_sep = false`
  - runtime aux overrides:
    - `ppo_runtime_normalized_q_value_weight_override = 0.0`
    - `ppo_runtime_next_state_flow_matching_weight_override = 0.0`

Clarification:

- the fixed-env gradient compare helper uses `_build_audit_env_cfg`, which already zeros:
  - `normalized_q_value_weight`
  - `next_state_flow_matching_weight`
- so the fixed-env gradient delta above isolates mainly:
  - actor baseline mode
  - advantage normalization
  - value-loss usage
  - SEP state-reset usage
- the new longrun driver profile is still needed to enforce the same no-aux semantics during actual continuation training

SEP state reset clarification:

- the optimistic Phase 3 short-run result is evaluation-only, so it does not come from a training-time `ppo_reset_env_state_at_sep` setting
- `ppo_reset_env_state_at_sep = true` remains the correct default for main PPO training because it fixes the separate actor/environment boundary bug
- but it is not part of the short-run Phase 3 evaluation semantics

## High-precision short-run-aligned profile regression

Question checked:

- does the new `--phase3-train-profile shortrun_aligned` driver profile numerically match an independently constructed manual reference of the same semantics?

Artifact:

- `/home/chen/RLPFN/artifacts/phase3_shortrun_profile_gradient_regression_v2.json`

Fixed compare setting:

- checkpoint:
  - `/home/chen/RLPFN/rwkv/models_diff/rlpfn_03_30_2026_22_37_53_epoch_13.cpkt`
- `frozen_h_seed = 12345`
- `train_env_seed = 2020`
- `rollout_seed = 2020`
- `single_eval_pos = 1946`
- `n_steps = 2048`
- `batch_size = 2048`

Result:

- training signature diff:
  - empty
- `exact_match = true`
- policy-gradient cosine similarity:
  - `1.0000033`
- gradient delta L2 norm:
  - `0.19665`
- effective runtime aux weights:
  - `effective.normalized_q_value_weight = 0.0`
  - `effective.next_state_flow_matching_weight = 0.0`

Interpretation:

- the previously low-cosine compare was comparing different semantics
- the `shortrun_aligned` driver profile itself now matches its manual reference to numerical precision
- this is the correct training-side profile to use for a strict Phase 3 short-run-aligned longrun benchmark
