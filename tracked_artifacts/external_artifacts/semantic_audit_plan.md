# Semantic Audit Plan

Date: 2026-03-31
Workspace: `/home/chen/RLPFN`
Scope: environment prior semantics, value semantics, PPO integration semantics, and Gym validation semantics.

## Goal

Audit whether the current training and validation stack is semantically clean enough to support context-based generalization, rather than merely producing plausible logs.

This audit should answer:

1. Is each environment prior's state transition mathematically correct?
2. Is the current state usage pipeline, including any RMS/postprocess before feeding the same SCM again, semantically correct?
3. Do transition + state usage together still define a learnable and generalizable context-learning task?
4. Is the current value definition correct?
5. Can value behave like a TabPFN-style regressor target, and is the fitted behavior actually meaningful?
6. Is PPO integrated correctly onto this objective?
7. Are terminal / explore / exploit phase semantics correct?
8. Is the model connected to the real validation environments correctly?

## Working Rules

- Prioritize semantics that can change learning behavior or generalization.
- Do not prioritize cosmetic refactors or weak optimizer-side concerns unless they directly corrupt the task definition.
- For every suspected issue, distinguish between:
  - Confirmed active-path bug
  - Confirmed design mismatch
  - Naming / maintenance debt only
  - Intentional behavior, not a bug

## Phase 3 Addendum (2026-04-10)

This audit now has an explicit Phase 3 rule:

- stop using distance-threshold search alone as the main evidence line
- require:
  - cross-suite reproduction on suite-matched baselines
  - semantic contrast on the same rollout contract
- do not reuse pair1 heldout-delta artifacts on later suites

Current locked Phase 3 audit findings:

- cross-suite compare-contract bug was active:
  - non-pair1 suites were still being pointed at pair1 heldout-delta artifacts
  - fix path:
    - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_subset_delta_fallback.py`
- reset-count semantics bug was active:
  - raw reset counting and compressed objective segmentation were inconsistent
  - fix path:
    - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_heldout_reset_semantics_probe.py`
- current cross-suite result boundary:
  - stable:
    - near-terminal post-reset tail bias reproduces on pair1 and pair2
    - `dist<=4 & late-step` ablation produces nearly identical bias-reduction deltas on pair1 and pair2
  - not yet stable:
    - broader `late-step` aggregate sign patterns

Current Phase 3 probe plan:

1. Generate suite-matched zero/PPO baselines for each new heldout suite.
2. Re-run post-reset tail probes without `reuse_heldout_subset_delta_json`.
3. Add semantic contrasts that change reset/tail semantics while preserving suite identity.
4. Only after cross-suite direction stays stable, consider repair proposals.

Current update after `seed_24680`:

- suite-matched baselines were completed
- but the suite-level heldout `PPO-vs-zero` sign flipped negative
- consequence:
  - `pre_vs_zero_suffix_gap` is not yet a regime-invariant target across suites
- semantic contrasts also showed:
  - removing `post_reset_terminal_tail`
  - or removing `post_reset_only`
  - reduces reset-bias
  - but worsens pre-gap alignment on this suite
- implication:
  - Phase 3 root-cause work must now separate:
    - reset-density tracking
    - suite-level policy-better-than-zero regime
  - do not collapse them into one “tail bias hurts generalization” sentence

Current update after `seed_13579` and four-group aggregation:

- four-group regime summary now exists:
  - `/home/chen/RLPFN/artifacts/phase3_cross_suite_regime_summary.json`
- strongest current Phase 3 root-cause statement:
  - the objective fits reset-heavy near-terminal mass consistently
  - but whether that mass corresponds to real heldout gain depends on suite-level `PPO-vs-zero` regime
- therefore the bottleneck is better described as:
  - mixed-regime optimization / fit conflict
  - not a single universally harmful tail-mask bug
- practical audit consequence:
  - any future many-env optimization diagnosis must stratify by suite-level `ppo_minus_zero_suffix` sign before aggregating token-quality correlations

Current update after the regime-aware mixed-update probe:

- canonical artifact:
  - `/home/chen/RLPFN/artifacts/phase3_regime_mixed_update_probe.json`
- strongest current fitting/optimization read:
  - the cross-suite conflict is visible at raw token-quality correlation level:
    - positive regime:
      - `corr_dist1_to_4_mass_vs_pre_gap = -0.0479`
    - nonpositive regime:
      - `corr_dist1_to_4_mass_vs_pre_gap = +0.0495`
  - the canonical artifact booleans now read:
    - `regime_split_visible_in_raw_correlation = true`
    - `pooled_sign_hides_regime_split = true`
    - `positive_regime_dominates_pooled_update = true`
  - but pooled many-env statistics are dominated by the positive regime:
    - positive regime `mass_share = 0.6799`
    - positive regime `pooled_covariance_contribution_share = 0.9317`
- audit consequence:
  - pooled token-quality correlation is not a reliable root-cause summary
  - future Phase 3 bottleneck diagnosis must preserve both:
    - regime split
    - regime weighting
  - otherwise:
    - the mixed-regime conflict can be hidden by mass imbalance rather than resolved

Current update after the regime-internal suite/env concentration probe:

- canonical artifact:
  - `/home/chen/RLPFN/artifacts/phase3_regime_env_concentration_probe.json`
- strongest current bottleneck read:
  - nonpositive regime does contain a real local positive signal:
    - `local_corr_mass_vs_pre_gap = +0.0495`
    - `regime_local_covariance_numerator = +0.1511`
  - but global centering converts it into negative pooled contribution:
    - `pooled_covariance_numerator_contribution = -0.7622`
  - positive regime then dominates the pooled path by both:
    - larger mass share:
      - `0.6799 vs 0.3201`
    - much larger absolute pooled covariance:
      - roughly `13.6x`
- important anomaly:
  - update-mass concentration and pooled-covariance concentration are not the same object
  - current artifact shows:
    - mass is dominated by pair1 heavy-mass envs
    - absolute pooled covariance is dominated by a tiny-mass pair2 outlier env
- audit consequence:
  - future Phase 3 optimization diagnosis must separate:
    - mass dominance
    - covariance dominance
  - and cannot use pooled covariance as a shorthand for “where the update mass went”

## Status Legend

- `todo`: not yet audited
- `in_progress`: actively auditing
- `confirmed_issue`: high-confidence semantic issue
- `confirmed_intent`: behavior looks deliberate and should not be treated as a bug
- `needs_experiment`: plausible issue, but requires targeted A/B or smoke validation
- `cleared_for_now`: no active-path bug found so far

## Checklist

### 0. Core Minimal-Prior Audit Track

Status: `in_progress`

Latest applied step:
- 2026-03-31: set `scm_standard_linear_init_enabled=True` as the default for the active exact-SCM prior path and the maintained RLPFN path, as the first stabilization step before adding any stronger prior-level controllability filter.
- 2026-03-31: ran a focused serial constant-action probe on 32 fixed Core-A prior envs comparing four regimes:
  - `old_like_no_std_no_rms`
  - `std_init_only`
  - `state_rms_only`
  - `std_init_plus_state_rms`
  Result:
  - standard init removed nonfinite reward deltas in this probe, but by itself it did not solve explosion and often increased reward sensitivity magnitude.
  - state RMS also removed nonfinite reward deltas in this probe and reduced magnitude more consistently than std-init-only.
  - `std_init_plus_state_rms` was the best of the four on this probe, eliminating nonfinite reward deltas and >1e6 tails, but it still left a large heavy tail (`reward_abs_gt_1e2_share ≈ 0.47`), so explosion is reduced rather than solved.
  - reward-delta “vanishing” was not the dominant issue in this short-horizon probe; the dominant issue remained a heavy-tailed reward channel.
- 2026-03-31: traced one fixed exact-SCM env step-by-step to verify which mechanisms actually fire.
  Result:
  - standard init is genuinely active: for a ReLU env with first layer shape `(412, 733)`, first-layer weight RMS moved from `0.00772` to `0.06962`, matching `sqrt(2 / fan_in)`, and the first hidden layer moved from `0.00773` to `0.05220`, matching `sqrt(2 / 733)`.
  - `state_full_rms` is genuinely active when the state is large, but it is a one-sided clip-only mechanism. On the traced env it changed `state_next_pre_rms` from `27.97` to `1.0` under `std_init_plus_state_rms`, while it did nothing in the small-scale regime where `state_next_pre_rms` was only `0.31`.
  - the dominant scale growth was observed before RMS, inside the transition trunk and reward branch:
    - step-0 first-linear RMS grew from `2.62` to `23.59`
    - step-0 outputs-flat RMS grew from `0.295` to `26.94`
    - step-0 reward raw grew from `0.073` to `23.08`
  - therefore the main problem is not “RMS failing to apply”, but “reward/state trunk scale becoming too large before the post-state RMS stage”.

Purpose:
- Establish the simplest meaningful environment-prior regime first.
- Verify value correctness and policy generalization before trusting any result from terminal / aux / reward-shaping variants.

Core question:
- If we remove terminal-reset style complications and reduce the prior to basic state-action transition learning, does the project still define a value target that is correct and a policy task that can generalize by context?

Minimal baseline candidate:
- `family="scm"`
- `strict_joint_transition_enabled=True`
- `terminal_reset_enabled=False`
- `reward_dropout_enabled=False`
- `ctrl_reward_weight=0` and disable ctrl reward
- `survival_reward_weight=0` and disable survival reward
- `normalized_q_value_weight=0`
- `next_state_flow_matching_weight=0`
- `reinforce_action_transform="none"`
- `reinforce_reward_transform="none"`

Two sub-regimes to audit separately:
- `Core-A`: no terminal, no aux reward, no flow/q aux, no state RMS, no state highway
- `Core-B`: same as Core-A, but turn on exactly one state transformation at a time
  - `state_full_rms_enabled=True`
  - then `state_input_scale_enabled=True`
  - then any other postprocess feature one by one

Important reference candidate:
- `reference_semantics_enabled=True` is useful as a "cleaned exact SCM" regime because it currently forces:
  - `alpha=1`
  - `state_noise_std=0`
  - `reward_scale=1`
  - `reward_clip=inf`
  - `state_highway_enabled=False`
  - `reward_dropout_enabled=False`
- But it does not by itself prove policy/value correctness, so it should be treated as a controlled baseline, not as proof.

Current code evidence for the minimal-track setup:
- Clean no-terminal/no-dropout exact-SCM rollout tests already exist:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/priors/test_environment_prior.py#L3566`
- Reference semantics really do override several messy env knobs:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/priors/test_environment_prior.py#L5528`
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/priors/environment_prior.py#L5843`
- Token layout already distinguishes terminal vs no-terminal mode:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_rlpfn_maintained_path.py#L1095`
- Dedicated prior-only audit entry now exists and runs against real checkpoints:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/prior_generalization_audit.py`

Minimal-track acceptance criteria:
- Value target remains well-defined and interpretable without terminal mechanics.
- Critic fit quality is meaningful in the minimal regime.
- Policy improves or at least adapts in a way consistent with context, not just reward shaping or terminal heuristics.
- Any failure here blocks trust in more complex regimes.

Current strongest finding in this track:
- Core-A prior environments currently do not define a reliably controllable policy-learning task.
- Empirically, fixed-environment same-rollout controls show that PPO held-out gains are indistinguishable from zero policy.
- Additional action-sensitivity probes show that, within sampled Core-A exact-SCM environments, a substantial fraction are nearly action-insensitive while others become numerically unstable under large actions.
- This means Step 1 is currently blocked by the prior task definition itself, not only by PPO fitting.
- Reachability clarification now confirmed:
  - `action -> reward` exists in the exact-SCM transition path, but it is not guaranteed to be strong for every sampled environment.
  - On fixed Core-A environments, one sampled env showed strong immediate and suffix sensitivity to constant-action scale changes, while another remained numerically identical from action `0` to `100`.
  - Therefore the current issue is not "action missing from the graph"; it is "sampled reward/state outputs are not constrained to be functionally controllable by action."
- `state_full_rms(state) -> reward` clarification now confirmed:
  - there is no same-step path from `state_full_rms` to reward in the active vectorized rollout path
  - reward is composed first, and only then is `state_next` postprocessed by `state_full_rms`
  - fixed-env A/B with the same `h / env_seed / rollout_seed` confirmed this empirically:
    - step-0 reward stayed identical when only `state_full_rms_enabled` was toggled in `h`
    - later rewards changed, consistent with a future-step path through `state_next`
- Additional same-env controllability clarification:
  - some sampled environments show weak but nonzero `action -> obs/state` while still having effectively zero `action -> reward`
  - on fixed env `env_seed=1942138939`, changing constant action from `0` to `100` left rewards numerically unchanged up to floating-point noise, while observable token differences stayed small but nonzero (`obs_max_abs_diff ~= 0.016`)
  - so the current failure mode is not only "no action effect at all"; it can also be "action perturbs state weakly, but reward remains functionally insensitive"
- New confirmed active-path issue:
  - the serial exact-SCM rollout path and the main family-vectorized rollout path are not semantically identical on the same sampled environment
  - fixed-env checks with identical `h / env_seed / rollout_seed / policy` show:
    - `env_seed=1942138939` matches almost exactly between `_rollout_single` and `rollout_with_policy`
    - `env_seed=376376244` diverges materially (`reward_max_abs_diff ~= 5.18e-2`, `x_max_abs_diff ~= 1.60e-1`)
  - `_rollout_single` itself is stable under repeated replay with the same seed, so this is not a single-path nondeterminism artifact
- New distribution-level risk:
  - a fresh 12-env scan of Core-A exact-SCM showed that single-vs-family rollout mismatch is not an isolated env
  - observed `max_abs_diff` ranged from near machine precision to extremely large values (up to `7.996e9`)
  - mismatch severity appears especially bad for `Identity`, and nontrivial for some `Tanh` / `sin`, while sampled `ReLU` cases were usually much closer
  - this strongly suggests a vectorized exact-SCM semantics bug or contract mismatch, not just an unlucky prior sample
- Secondary confirmed bug:
  - `_rollout_distinct_envs_vectorized_with_policy` currently throws `NameError: ppo_obs_steps is not defined`
  - this is not the main family-rollout issue above, but it blocks one of the intended parity-debug paths
- New serial-only controllability conclusion:
  - a dedicated serial action-reachability probe now exists in the audit entry
  - on a fresh 32-env Core-A suite with constant-action `0` vs `100`:
    - all finite envs showed observable-state change (`obs_max_abs_delta_gt_1e_2_share = 1.0`)
    - only `10%` had suffix-return delta below `1e-2`
    - `40%` still had suffix-return delta below `1.0`
    - but the reward-delta distribution was extremely heavy-tailed, with finite median around `2.12` and upper tail exploding (`p90 ~= 6.59e15`)
  - this sharpens the core diagnosis:
    - the main failure is not "action cannot reach state"
    - the main failure is "reward sensitivity is badly conditioned: often weak relative to state change, but sometimes explosively large"
    - this makes stable policy learning difficult across many sampled environments even before considering OOD transfer
- New serial-PPO integration issue:
  - a serial same-env audit of the real PPO checkpoint does not yet run cleanly
  - current blocker is a batch/cache contract bug when reusing the official PPO validation policy in `_rollout_single`
  - this is now a confirmed audit-path integration bug, separate from the prior-task definition issues above

Execution steps for the minimal track:

1. Build `Core-A` config:
   - no terminal
   - no reward dropout
   - no ctrl/survival reward
   - no q aux / no flow aux
   - no state RMS / no state highway
2. Verify transition semantics first:
   - serial vs vectorized parity
   - exact/reference vs maintained helper parity where applicable
3. Verify critic target semantics next:
   - check whether value target is still coherent in this stripped regime
   - check whether bardistribution fit and EV metrics tell a consistent story
4. Verify policy generalization next:
   - context-poor vs context-rich evaluation split
   - does additional context improve exploit policy in the real validation loop?
5. Re-enable one complexity at a time:
   - state RMS
   - state input scaling
   - terminal reset
   - aux rewards
   - flow aux
6. Attribute regressions:
   - if generalization breaks only after a feature is re-enabled, that feature becomes the primary suspect

Fast test ladder for the minimal track:

- `L0: deterministic semantic/unit checks`
  - transition parity: serial vs vectorized, same config/seed
  - token contract: no-terminal vs terminal-disabled layout
  - value target reconstruction: rewards -> returns -> bootstrap -> raw/normalized recovery
  - validation contract: exploit return summary uses unseen rollout, not training rollout reuse

- `L1: context-benefit sentinel on held-out environments`
  - Freeze policy after a short train chunk.
  - Sample fresh `h_list` / fresh environments not used for the training update.
  - Evaluate three conditions:
    - `ctx`: normal explore then exploit
    - `no_ctx`: exploit immediately with zero/minimal context
    - `shuf_ctx`: exploit with context copied from a different environment
  - Primary metrics:
    - `context_gain = R(ctx) - R(no_ctx)`
    - `causal_context_gain = R(ctx) - R(shuf_ctx)`
  - Interpretation:
    - `context_gain > 0` is necessary
    - `causal_context_gain > 0` is stronger evidence that the model uses environment-specific context, not just warmup history

- `L2: null and anti-causal controls`
  - `null-env`: reward independent of action/context
    - expected result: context benefit should stay near zero
  - `distractor-prefix env`: prefix observations/history do not predict suffix reward dynamics
    - expected result: shuffling or deleting prefix should not hurt much
  - If context still appears to help here, suspect leakage or metric illusion

- `L3: oracle-simple environments`
  - Build one or two very small exact-SCM tasks where the optimal exploit action or exact return is analytically available
  - Examples:
    - hidden linear coefficient inferred from prefix, exploit reward depends on matching that coefficient
    - deterministic quadratic reward with closed-form optimal action
  - Use these to separate:
    - representation failure
    - critic target failure
    - policy optimization failure

- `L4: head-only sanity checks`
  - Freeze rollout data or hidden states and fit only the critic head on one fixed batch
  - If head-only fitting cannot overfit a tiny fixed batch, suspect target/support/math bug
  - If critic head overfits easily but PPO critic does not improve online, suspect optimization/shared-backbone conflict

- `L5: representation probes`
  - After prefix context, freeze the model and train a tiny probe to predict:
    - hidden env parameter
    - next-step reward
    - oracle exploit action
  - If probe succeeds but policy does not, the representation exists and PPO/actor integration is the likely issue
  - If probe fails, the prior or token/state semantics are the likely issue

- `L6: short multi-seed smoke matrix`
  - use 2-3 seeds only
  - report train reward and held-out context metrics together
  - preferred summary:
    - `train_reward`
    - `heldout_R(ctx)`
    - `heldout_R(no_ctx)`
    - `heldout_R(shuf_ctx)`
    - `context_gain`
    - `causal_context_gain`
  - This is the fastest way to tell "real generalization" from luck, collapse, or reward shaping

Prior-environment generalization observables:

- `P0: fixed held-out prior suite`
  - Sample and persist a small set of held-out `h_list` / rollout seeds from `EnvironmentPrior`
  - Never use them for training updates
  - Re-evaluate the same suite after each short train chunk
  - This is the primary low-noise signal for "did learning on prior environments transfer to other prior environments?"

- `P1: fresh held-out prior resampling`
  - In addition to the fixed suite, resample a fresh held-out prior batch periodically
  - This guards against overfitting to the fixed held-out suite itself

- `P2: initial-policy / random-policy baselines`
  - Always compare the current policy against:
    - the untrained initial policy
    - a random-action or zero-action baseline
  - If the trained policy does not beat these on held-out prior environments, there is no usable prior-environment generalization

- `P3: train-vs-heldout gap`
  - Report both:
    - train prior objective / returns
    - held-out prior objective / returns
  - If only train improves, suspect memorization, leakage, or overly narrow adaptation

- `P4: oracle-regret when available`
  - For simple exact-SCM tasks with known optimal action or exact return:
    - report regret on held-out prior environments
  - This is better than raw reward whenever an oracle is available

- `P5: fixed-seed trend first, resampled trend second`
  - Use fixed held-out prior seeds to detect optimization progress quickly
  - Use fresh resampled prior environments to verify the progress is not just seed memorization

Dedicated audit entry:

- `A0: standalone prior-only audit runner`
  - Implemented at:
    - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/prior_generalization_audit.py`
  - Purpose:
    - hold Gym validation completely out of the loop
    - evaluate one frozen policy directly on fixed `train prior suite` and `held-out prior suite`
    - support checkpoint-backed actor loading and zero-policy smoke baselines
  - Current outputs:
    - fixed-suite summaries
    - full return / suffix return on train and held-out prior suites
    - suffix reward-component return means when available
    - heldout-minus-train gaps
  - Current status:
    - implemented
    - targeted tests added
    - next step is to use this entry as the main instrument for the minimal Core-A audit track

## Progress Log

- 2026-03-31:
  - Confirmed that the next audit step must focus on prior-environment transfer, not Gym/OOD validation.
  - Added a standalone prior-only audit entry:
    - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/prior_generalization_audit.py`
  - The new entry can:
    - generate fixed train / held-out prior suites
    - save or reload those suites
    - load checkpoint-backed policy state or run a zero-policy baseline
    - report direct train-vs-heldout prior returns without mixing in real-env validation semantics

### 1. Environment Prior Transition Correctness

Status: `todo`

Checks:
- Exact SCM transition input packing is correct.
- State, action, noise, and reward heads are composed correctly.
- Terminal reset and reward bonus logic do not silently alter transition semantics.
- Vectorized and serial paths are semantically equivalent under matched config/seed.

Acceptance criteria:
- Same config/seed yields matching transition semantics across maintained/reference paths.
- No active-path branch uses stale legacy math unintentionally.

### 2. State Reuse / RMS / Same-SCM Re-entry

Status: `todo`

Checks:
- State postprocess, state RMS, and re-entry into the next transition are ordered intentionally.
- RMS and clipping do not destroy causal information needed for generalization.
- Prefix/suffix semantics do not accidentally see inconsistent state spaces.

Acceptance criteria:
- One clear state space is used for transition, policy input, and next-state targets, or the mappings are explicitly justified.

### 3. Learnability / Generalization of the Prior Task

Status: `todo`

Checks:
- The resulting prior still defines a nontrivial but learnable context-learning problem.
- Support/query or explore/exploit splits match the intended generalization target.
- Training-time split semantics match validation-time adaptation semantics closely enough.

Acceptance criteria:
- No major train/validation task-definition mismatch remains.

### 4. Value Definition

Status: `in_progress`

Checks:
- Value target semantics are clear and stable.
- Raw-space vs normalized-space value meanings are consistent.
- Bootstrap, GAE, and reported diagnostics all refer to the same target definition.

Acceptance criteria:
- Value target can be stated in one sentence without caveats.

### 5. Value as TabPFN-Style Regressor

Status: `todo`

Checks:
- Bardistribution support and target normalization actually match.
- Mean prediction quality and distributional fit are both interpretable.
- Low/negative NLL does not hide a meaningless mean predictor.

Acceptance criteria:
- Value loss, normalized EV, and raw EV can be jointly explained without contradiction.

### 6. PPO Integration Correctness

Status: `in_progress`

Checks:
- PPO old/new log-prob semantics are correct.
- Objective masking is applied where intended.
- Buffer fields, returns, advantages, and aux targets use the intended spaces.
- PPO is not accidentally borrowing REINFORCE-only math on the active path.

Acceptance criteria:
- Active PPO path is mathematically self-consistent and does not silently activate unrelated training logic.

### 7. Terminal / Explore / Exploit Phase Semantics

Status: `confirmed_issue`

Current finding:
- PPO training switches to eval/objective at a random step index `single_eval_pos`, independent of episode boundaries.
- Gym validation switches to eval only after a whole explore rollout finishes, then resets into a fresh rollout before exploit/eval begins.

Why this matters:
- Training optimizes "random suffix of a continuous rollout".
- Validation measures "new rollout after accumulated explore context".
- This is a task-definition mismatch and is currently the strongest learning-relevant concern.

Primary evidence:
- Training samples a scalar `single_eval_pos` and builds `objective_mask` from `step_idx >= single_eval_pos`:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/sb3_recurrent_ppo.py#L1713`
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/sb3_recurrent_ppo.py#L1750`
- Environment rollout flips `phase_t` only by timestep threshold:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/priors/environment_prior.py#L14840`
- Validation flips phase only after completing a rollout, then resets:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/rl_validation.py#L895`
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/rl_validation.py#L936`
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/rl_validation.py#L965`

Next actions:
- Decide whether training should become boundary-aligned with validation.
- If yes, add an episode-aligned eval suffix mode and compare against current behavior.

### 8. Validation Environment Wiring

Status: `in_progress`

Checks:
- Validation action selection path is explicit and has no silent fallback.
- Policy input token semantics in validation match training as closely as possible.
- Reward/action transforms are intentional and documented.

Acceptance criteria:
- Validation path is deterministic to audit and uses only declared backends.

## Confirmed Non-Issues / Rejected Suspicions

- `flow aux` using prefix data is intentional and should not be treated as a bug.
  - Prefix flow actions/states are explicitly stored for the flow task.
  - Evidence:
    - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/priors/environment_prior.py#L14957`
    - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/priors/environment_prior.py#L15589`

- `policy_objective_kind="reinforce"` on the PPO rollout path is currently treated as naming / flag debt, not a confirmed active-path math bug.
  - It enables sampling/log-prob flag behavior, but PPO path does not also enable REINFORCE replay by default.
  - Evidence:
    - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/priors/maintained_policy_rollout.py#L14`
    - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/priors/environment_prior.py#L13518`

## Progress Log

- 2026-03-31: Created audit plan and seeded it with the strongest currently confirmed semantic issue: training/validation phase-boundary mismatch.
- 2026-03-31: Marked `flow aux` prefix training as intentional, not a bug.
- 2026-03-31: Marked PPO's `policy_objective_kind="reinforce"` usage as unresolved naming debt, but not yet a confirmed active-path math corruption.
- 2026-03-31: Promoted the minimal no-terminal SCM regime to the top-priority audit track, because value correctness and policy generalization must be established there before trusting more complex environments.
- 2026-03-31: Confirmed that `return_*` validation metrics are collected from separate Gym validation rollouts after training updates and are not used to update the policy; these are the primary current observables for cross-environment generalization.
- 2026-03-31: Marked PPO `value_loss` and `explained_variance_normalized` as weak observables for generalization in the current setup; they may diagnose critic-fit semantics, but they currently do not reliably indicate context generalization.
- 2026-03-31: Built a dedicated prior-only audit runner at `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/prior_generalization_audit.py` and verified it on a real checkpoint: `/home/chen/RLPFN/rwkv/models_diff/rlpfn_03_30_2026_18_20_50_epoch_2.cpkt`.
- 2026-03-31: First real Core-A smoke on fixed prior suites showed weak held-out prior performance and a negative heldout-vs-train suffix gap (`-0.9216`) in `/home/chen/RLPFN/artifacts/prior_generalization_audit/rlpfn_epoch2_smoke.json`.
- 2026-03-31: Running the same fixed Core-A suites with `zero` policy reproduced the same train-suite nonfinite return pattern (`full_return_nonfinite_count=2`, `suffix_return_nonfinite_count=1`) in `/home/chen/RLPFN/artifacts/prior_generalization_audit/zero_on_same_suites_corea.json`, making the environment-prior regime itself the leading suspect rather than PPO policy loading.
- 2026-03-31: Extended the prior-only audit runner with a dedicated `same_env` regime that fixes base prior environments and resamples only rollout seeds, plus paired per-environment heldout-vs-train summaries.
- 2026-03-31: Real `same_env` Core-A audit on `/home/chen/RLPFN/rwkv/models_diff/rlpfn_03_30_2026_18_20_50_epoch_2.cpkt` produced a positive paired suffix delta (`+0.1286`) in `/home/chen/RLPFN/artifacts/prior_generalization_audit/rlpfn_epoch2_same_env.json`, but the matched `zero`-policy control on the exact same suites produced virtually the same paired suffix delta (`+0.1286`) in `/home/chen/RLPFN/artifacts/prior_generalization_audit/zero_same_env_control.json`.
- 2026-03-31: Current strongest conclusion for Step 1: the apparent same-environment heldout improvement is not yet evidence of PPO learning or context generalization, because it is indistinguishable from the zero-policy control on the same fixed prior environments and rollout splits.
- 2026-03-31: Same-environment action probes on the fixed Core-A suites showed PPO mean actions are tiny (`action_abs_mean≈0.0216`) while random actions with much larger magnitude change returns very little for most base environments; in the 4-env probe, only 1 base env was strongly action-sensitive, 1 was weakly sensitive, and 2 were essentially action-insensitive.
- 2026-03-31: On a fresh 32-env Core-A prior sample, comparing `const_0` vs `const_100` actions produced 3/32 nonfinite environments; among the remaining 29 finite environments, about 20.7% had `|delta| < 1e-3`, 31.0% had `|delta| < 1e-2`, and 41.4% had `|delta| < 1e-1`. This is now treated as a confirmed high-priority issue in the minimal prior regime: controllability is weak or absent for a substantial slice of sampled environments, while another slice is numerically unstable.
- 2026-03-31: Verified that exact-SCM `scm_standard_linear_init_enabled=True` and `state_full_rms` both truly take effect, but the dominant scale growth happens in the shared transition trunk before `state_full_rms` runs; current-step reward is read before RMS and therefore can still inherit large trunk scale.
- 2026-03-31: Implemented `reinforce_action_transform=\"clip\"` with `reinforce_action_clip_bound=5.0` across the exact-SCM rollout paths, made it the maintained default, and added targeted unit coverage for both the pure transform and rollout token path.
- 2026-03-31: Re-ran the fixed 32-env Core-A serial action-reachability probe with `std_init × {clip,no_clip} × {state_rms,on/off}` and saved the comparison to `/home/chen/RLPFN/artifacts/prior_generalization_audit/scm_stability_compare_action_clip_state_rms.json`.
- 2026-03-31: Current strongest stabilization conclusion: action clipping is the first change that clearly collapses the heavy reward tail without destroying state reachability. On the fixed suite, `std_init_no_clip_no_rms` had `|suffix delta|` median `41.96`, p90 `204.61`, and `>1e2` share `0.3125`, while `std_init_clip_no_rms` dropped to median `1.81`, p90 `10.23`, and `>1e2` share `0.0`. Adding `state_rms` on top still helps somewhat (`std_init_clip_state_rms` p90 `5.39`), but clipping appears to be the primary control knob for the remaining explosion issue in the minimal prior regime.
- 2026-03-31: Long-horizon serial probe (`n_samples=128`, fixed 32-env Core-A suite, constant action `100`) shows that `clip` alone is not sufficient for reward stability over long rollouts. Results in `/home/chen/RLPFN/artifacts/prior_generalization_audit/long_horizon_action_clip_stability.json`: `std_init_no_clip_no_rms` has all-step abs p90 `1.31e5` and step-127 p90 `6.61e11`; `std_init_clip_no_rms` improves this but is still heavy-tailed with all-step abs p90 `6.48e3` and step-127 p90 `9.23e10`. By contrast, `state_rms` is what removes the long-horizon recurrence blow-up (`std_init_no_clip_state_rms` step-127 p90 `36.0`), and `clip + state_rms` is the first combination that is stably small over long horizons (`all-step abs p90 1.99`, step-127 p90 1.79, max 4.66`).

## Next Audit Order

1. Core minimal-prior audit track
2. Terminal / explore / exploit phase contract
3. Validation environment wiring and action/reward token semantics
4. PPO integration math on the active path
5. Value definition and critic-fit meaning
6. State transition and state-reuse semantics in the environment prior

## Mainline Audit Ledger

Purpose:
- Keep the mainline investigation focused on the smallest unresolved blockers.
- Prevent repeating already-credible checks unless a new codepath or configuration invalidates them.
- Force every next experiment to say which prior uncertainty it is reducing.

### Mainline 1: Environment Scientific / Learnable

Current status: `partially_cleared`

Credible findings:
- Non-vectorized, no-terminal prior environments are stable enough to study.
  Evidence:
  - reward/state do not frequently vanish or explode in the serial minimal-prior probes
  - simple constant-action policies can separate returns on fixed environments
- The environment prior is not uniformly uncontrollable.
  Evidence:
  - action reachability probes show substantial state/reward response on a nontrivial fraction of fixed environments
- This means Mainline 1 is no longer blocked by “the environment always returns garbage.”

Still open:
- Some sampled environments remain weakly controllable or numerically awkward.
- The prior family is therefore not fully “clean,” but it is good enough to continue Mainline 2 on fixed-environment controlled probes.

Do not repeat by default:
- Re-running generic “is reward/state stable?” checks on the same minimal-prior serial path.
- Re-running zero-vs-constant-action reachability sweeps unless the prior family or reward transform changes.

Repeat only if:
- environment family changes
- reward transform changes
- state transition semantics change

### Mainline 2: PPO Single-Environment Optimizable

Current status: `not_yet_cleared`

Credible findings:
- First-update overstep is real.
  Evidence:
  - high `approx_kl` / `clip_fraction` under early PPO settings
  - improved when using `target_kl`, larger `batch_size`, or smaller `ppo_n_epochs`
- Learned baseline is not the primary culprit in the quick GPU critic-free audit.
  Evidence:
  - `learned` outperformed `zero` and `rollout_mean` in the quick single-env audit
  - artifact: `/home/chen/RLPFN/artifacts/critic_free_single_env_audit_quick.json`
- Mainline-like settings are more stable than quick settings across frozen prior seeds, but still mildly negative on average.
  Evidence:
  - artifact: `/home/chen/RLPFN/artifacts/frozen_h_seed_quick_vs_mainline.json`
  - `mainline_like delta_smp_gap_mean = -0.2117`
  - `quick delta_smp_gap_mean = -1.1586`
- Remaining failure is not mainly “exploit suffix too short.”
  Evidence:
  - in fixed-`mainline_like` sweeps, changing `single_eval_pos / n_steps` does not show a clean monotone “longer suffix fixes learning” pattern
  - same frozen prior seed can flip sign non-monotonically across `single_eval_pos`
- Under `mainline_like`, the current token-wise actor signal barely predicts real exploit improvement.
  Evidence:
  - artifact: `/home/chen/RLPFN/artifacts/mainline_like_actor_objective_alignment_compare.json`
  - token-wise actor signal vs `delta_suffix_return`: Pearson `0.0208`, Spearman `-0.0521`
  - top actor-signal quartile still contains `5/8` non-improving trajectories
- A simple trajectory-level raw-suffix signal is modestly better than the current token-wise actor signal for improvement ranking, but still weak on delta.
  Evidence:
  - same artifact as above
  - trajectory raw suffix vs `delta_suffix_return`: Pearson `0.1222`, Spearman `0.0700`
  - trajectory raw suffix vs `post_suffix_return`: Pearson `0.9854`, Spearman `0.9769`
- A pure trajectory-level actor objective swap is not a good fix under `mainline_like`.
  Evidence:
  - artifacts:
    - `/home/chen/RLPFN/artifacts/mainline_like_objective_alignment_compare_tokenwise.json`
    - `/home/chen/RLPFN/artifacts/mainline_like_objective_alignment_compare_traj_suffix.json`
    - `/home/chen/RLPFN/artifacts/critic_free_single_env_audit_tokenwise_learned.json`
    - `/home/chen/RLPFN/artifacts/critic_free_single_env_audit_traj_suffix_learned.json`
  - alignment compare:
    - `tokenwise corr_mean_vs_delta = 0.2912`
    - `trajectory_suffix_return corr_mean_vs_delta = 0.1080`
  - mainline-like single-env audit (`learned` baseline):
    - `tokenwise delta_smp_gap = +1.0728`
    - `trajectory_suffix_return delta_smp_gap = -0.6321`
- A bounded trajectory-level correction layered onto token-wise PPO is mixed so far.
  Evidence:
  - artifacts:
    - `/home/chen/RLPFN/artifacts/mainline_like_objective_alignment_compare_traj_correction.json`
    - `/home/chen/RLPFN/artifacts/critic_free_single_env_audit_traj_correction_learned.json`
  - alignment compare:
    - `tokenwise corr_mean_vs_delta = 0.2912`
    - `tokenwise_suffix_return_correction corr_mean_vs_delta = 0.1653`
  - mainline-like single-env audit (`learned` baseline, frozen_h_seed=12345):
    - `tokenwise delta_smp_gap = +1.0728`
    - `tokenwise_suffix_return_correction delta_smp_gap = +2.7867`
- Cross-`frozen_h_seed` single-env audit says the bounded correction is not enough to restore robust positive learning.
  Evidence:
  - artifact:
    - `/home/chen/RLPFN/artifacts/critic_free_cross_frozen_h_tokenwise_vs_correction.json`
  - `tokenwise`:
    - `delta_smp_gap_mean = -0.2125`
    - `positive_count = 2/4`
  - `tokenwise_suffix_return_correction`:
    - `delta_smp_gap_mean = -0.1087`
    - `positive_count = 2/4`
  - interpretation:
    - correction recovers some bad seeds but flips some good seeds; not enough to call the effective computation model restored
- Under `mainline_like`, actor-objective / raw-improvement mismatch is strongest in the early exploit suffix.
  Evidence:
  - artifact:
    - `/home/chen/RLPFN/artifacts/mainline_like_suffix_position_probe.json`
  - overall:
    - `corr_adv_vs_parent_delta = -0.0568`
    - `high_adv_nonpositive_parent_count = 853 / 1536`
  - by suffix quartile:
    - `q1_early corr_adv_vs_parent_delta = -0.1665`
    - `q2_mid_early = -0.0896`
    - `q3_mid_late = +0.0114`
    - `q4_late = +0.0131`
  - interpretation:
    - the worst misalignment is concentrated near the explore->exploit boundary, not uniformly across the whole suffix
- A boundary-local ablation that drops the earliest exploit bucket is a materially stronger fix than prior global reweighting attempts.
  Evidence:
  - artifact:
    - `/home/chen/RLPFN/artifacts/critic_free_cross_frozen_h_tokenwise_vs_q1_drop.json`
  - `tokenwise`:
    - `delta_smp_gap_mean = -1.0807`
    - `positive_count = 1/4`
  - `tokenwise_drop_q1_early`:
    - `delta_smp_gap_mean = +1.3063`
    - `positive_count = 3/4`
  - interpretation:
    - removing actor update pressure on the earliest exploit quartile improves cross-`frozen_h_seed` single-env audit substantially
    - this is a stronger sign of boundary-local phase/contract mismatch than the earlier global trajectory-level correction experiments
- In the minimal q1 bifurcation, actor-only q1 drop outperforms actor+value/return contract shift.
  Evidence:
  - artifacts:
    - `/home/chen/RLPFN/artifacts/critic_free_cross_frozen_h_tokenwise_vs_q1_drop.json`
    - `/home/chen/RLPFN/artifacts/critic_free_cross_frozen_h_q1_contract_shift.json`
  - `actor-only q1 drop`:
    - `delta_smp_gap_mean = +1.3063`
    - `positive_count = 3/4`
  - `actor+value/return contract shift q1`:
    - `delta_smp_gap_mean = -0.1081`
    - `positive_count = 2/4`
  - interpretation:
    - the strongest win comes from suppressing actor pressure on the earliest exploit bucket
    - shifting the broader value/return training contract at q1 does not help in the same way
    - this points more toward boundary-local actor credit mismatch than toward a broader q1 value/return contract bug
- Using “the first terminal after `single_eval_pos`” is not currently a native PPO optimization boundary in this stack.
  Evidence:
  - PPO envs emit `done/truncated` at rollout length (`n_steps`), while terminal-reset events remain internal transition semantics.
  - interpretation:
    - do not treat “first terminal after `single_eval_pos`” as the default next boundary probe
    - if terminal-reset matters, inspect it first as a stratification variable, not as the primary training boundary
- Gym/VecEnv adapter `done/truncated` semantics were incorrect and have now been fixed, but this does not by itself rewrite Mainline 2.
  Evidence:
  - adapter fix now emits:
    - `terminated` on internal terminal-reset events
    - `truncated` on `n_steps`
    - `dones = terminated | truncated`
  - targeted adapter tests pass
  - the strict PPO training rollout path already consumed prior terminal events as `dones` inside the custom sink before this adapter fix
  - interpretation:
    - treat this as an audit-harness / adapter-correctness repair
    - do not assume prior Mainline 2 findings are invalidated unless a dedicated post-fix re-audit contradicts them
- Critic fit is weak.
  Evidence:
  - critic ranking and explained variance stay near zero
  - bar critic is not obviously failing because of support mismatch or too-small gradients
  - fixed-buffer scalar probes do not substantially outperform bar NLL

Current best interpretation:
- Mainline 2 is now blocked more by actor-objective / exploit-objective mismatch and weak critic usefulness than by raw environment instability.
- Overstep was one blocker, but after controlling it, the remaining gap still does not close.
- The current token-wise PPO actor target appears to lose trajectory-level exploit ordering information that is still visible in raw suffix return.
- But replacing it with a pure trajectory-level suffix score makes things worse; the remaining fix likely needs a correction, not a full replacement.
- A simple bounded correction may help on at least one fixed prior environment, but it does not yet improve cross-trajectory alignment under `mainline_like`.
- Even with cross-seed reweighting correction, learning is still not robustly positive; the remaining blocker likely sits deeper than simple actor-weight reshaping.
- The remaining blocker now looks more boundary-local than global:
  - early exploit tokens carry the strongest high-advantage / non-improvement mismatch
  - later suffix positions are much closer to neutral alignment
- An audit-only actor-side `single_eval_pos`-synced GAE is **not** currently a stronger fix than plain tokenwise PPO.
  Evidence:
  - artifact:
    - `/home/chen/RLPFN/artifacts/critic_free_single_env_audit_sep_synced_learned.json`
    - `/home/chen/RLPFN/artifacts/critic_free_cross_frozen_h_tokenwise_vs_sep_synced_quick.json`
  - on one fixed prior environment (`frozen_h_seed=12345`, `eval_env_count=8`), `normalized_sep_synced` can still improve:
    - `delta_smp_gap = +0.6888`
  - but in a cross-`frozen_h_seed` quick compare (`eval_env_count=4`), it is worse than the current `normalized` actor GAE:
    - `normalized delta_smp_gap_mean = +1.0641`, `positive_count = 3/4`
    - `normalized_sep_synced delta_smp_gap_mean = -0.3359`, `positive_count = 2/4`
  - interpretation:
    - a pure SEP-synced actor-side GAE/normalization rewrite does not look like the general root fix
    - the bug signal is still more local than “all actor credit should reset at SEP”
- A boundary-local intervention now outperforms both:
  - plain tokenwise PPO
  - broad trajectory-level correction
- Among boundary-local interventions tried so far:
  - actor-only q1 suppression is the best
  - q1 contract shift is not
- This shifts the leading hypothesis from “global actor-target mismatch” toward “bad credit assignment / contract mismatch concentrated at the exploit boundary.”
- A finer boundary-credit quick spot check suggests the worst mismatch may sit in the very first few exploit steps, not necessarily the whole q1 bucket.
  Evidence:
  - artifact:
    - `/home/chen/RLPFN/artifacts/critic_free_boundary_credit_refinement_single_seed_quick.json`
  - under quick single-seed settings (`frozen_h_seed=12345`, `n_steps=128`, `single_eval_pos=64`, `outer_epochs=1`):
    - `tokenwise delta_smp_gap = +0.7257`
    - `tokenwise_drop_first_8_suffix_steps delta_smp_gap = +2.1168`
    - `tokenwise_linear_ramp_first_8_suffix_steps delta_smp_gap = -0.4200`
    - `tokenwise_drop_q1_early delta_smp_gap = +0.2249`
  - interpretation:
    - a hard drop on the first 8 suffix steps may outperform dropping the whole q1 bucket
    - a smooth linear ramp over the first 8 steps can be worse than baseline
    - current best guess is shifting from “whole q1 is bad” toward “a small earliest boundary prefix is especially bad”
  - caution:
    - this is only a quick single-seed result; do not treat it as the new default conclusion until cross-`frozen_h_seed` confirmation exists
- A pre-terminal-tail probe does **not** support shifting the boundary to “the first terminal after `single_eval_pos`”.
  Evidence:
  - artifacts:
    - `/home/chen/RLPFN/artifacts/mainline_like_preterminal_tail_probe.json`
    - `/home/chen/RLPFN/artifacts/mainline_like_preterminal_tail_probe_quick.json`
  - under both a mainline-like probe and a quicker single-seed probe:
    - overall suffix `corr(actor_adv, delta_suffix_return)` is near `0`
    - but the last few objective-valid tokens **before the first post-SEP terminal** have strongly positive correlation
    - mainline-like: `preterminal_tail_corr_adv_vs_delta = +0.8911`
    - quick: `preterminal_tail_corr_adv_vs_delta = +0.8776`
  - interpretation:
    - those pre-terminal tail tokens look more aligned with real improvement than the average suffix token
    - so “drop a few tokens before the first post-SEP terminal” is currently the wrong next boundary shaping hypothesis
    - keep the main boundary focus on the earliest exploit prefix, not on pre-terminal tails
- A compact bad-credit distribution summary confirms that the mismatch is front-loaded relative to `single_eval_pos`, then decays toward neutral deeper into suffix.
  Evidence:
  - artifact:
    - `/home/chen/RLPFN/artifacts/mainline_like_bad_credit_distribution.json`
  - with `single_eval_pos = 64`, `n_steps = 256`, suffix length is `192`, and:
    - overall bad positive-credit mass share is `0.551`
    - first `8` suffix steps: bad positive-credit mass share within-bin `0.617`
    - steps `16..31`: `0.671`
    - steps `32..63`: `0.712`
    - steps `64+`: `0.495`
  - interpretation:
    - the worst bad-credit concentration is not spread uniformly across suffix
    - it is elevated in the early-to-mid early suffix (`0..63` steps after `single_eval_pos`)
    - deeper suffix is much closer to neutral and should not be the primary suppression target
- A pure actor-side `single_eval_pos` contract reset does not look like the general fix.
  Evidence:
  - artifacts:
    - `/home/chen/RLPFN/artifacts/critic_free_cross_frozen_h_tokenwise_vs_sep_synced_quick.json`
    - `/home/chen/RLPFN/artifacts/critic_free_cross_frozen_h_hidden_reset_probe_quick_partial.json`
  - `normalized_sep_synced` is worse than plain `normalized` in cross-`frozen_h_seed` quick compare
  - `reset_hidden_at_sep` alone is also not enough:
    - partial quick mean `delta_smp_gap = -0.1841`
    - `positive_count = 1/2`
  - interpretation:
    - the bug is not “all actor credit should reset at SEP”
    - the bug is also not “just reset recurrent hidden state at SEP”
- A stronger bug-style signal appears only when the first exploit token contract is changed as a pair:
  - reset recurrent hidden/cache at `single_eval_pos`
  - clear the first exploit token's incoming history inputs (`reward_t`, `reward_mask_t`, `action_t`, `terminal_t`)
  Evidence:
  - artifact:
    - `/home/chen/RLPFN/artifacts/critic_free_cross_frozen_h_hidden_reset_probe_quick_partial.json`
  - quick partial compare over `frozen_h_seed in {12345, 23456}`:
    - `normal`: mean `delta_smp_gap = -0.3905`, `positive_count = 0/2`
    - `reset_hidden_and_clear_first_eval_history`: mean `delta_smp_gap = +1.0277`, `positive_count = 2/2`
  - interpretation:
    - the leading root-cause candidate has narrowed from a broad boundary-credit issue to a first-exploit-token contract mismatch
    - the likely bug is the combined carry-over of recurrent state and step-history features across the SEP transition, not either one alone
- That first-token contract hypothesis does **not** yet survive a stricter `mainline_like` partial re-check.
  Evidence:
  - artifact:
    - `/home/chen/RLPFN/artifacts/critic_free_cross_frozen_h_first_token_contract_mainline_like_partial.json`
  - first two `frozen_h_seed` results under `mainline_like` (`n_steps=256`, `outer_epochs=2`, `eval_env_count=8`):
    - `normal`: mean `delta_smp_gap = +0.6439`, `positive_count = 1/2`
    - `reset_hidden_and_clear_first_eval_history`: mean `delta_smp_gap = -0.3556`, `positive_count = 1/2`
  - interpretation:
    - the exact “first exploit token contract bug” is not yet a robust general explanation
    - quick wins from first-token surgery likely exposed a nearby issue, but not the full mainline-like root cause
    - the leading structural candidate shifts upward from “first exploit token” to “first exploit episode remains semantically mixed with pre-SEP episode history”
- An audit-only `force_episode_reset_at_sep` is structurally meaningful, but not yet a general fix.
  Evidence:
  - artifacts:
    - `/home/chen/RLPFN/artifacts/sep_episode_boundary_probe_force_only_seed12345.json`
    - `/home/chen/RLPFN/artifacts/sep_episode_boundary_probe_force_only_seed23456.json`
    - `/home/chen/RLPFN/artifacts/sep_episode_boundary_probe_force_only_seed34567.json`
    - `/home/chen/RLPFN/artifacts/sep_episode_boundary_probe_force_only_seed45678.json`
    - `/home/chen/RLPFN/artifacts/mainline_like_bad_credit_distribution_force_episode_reset_at_sep.json`
  - cross-`frozen_h_seed` compare against existing `mainline_like` `normalized` baseline:
    - baseline mean `delta_smp_gap = +0.5948`, `positive_count = 2/4`
    - `force_episode_reset_at_sep` mean `delta_smp_gap = +0.4584`, `positive_count = 2/4`
  - bad-credit distribution improves sharply under the force-reset probe:
    - overall bad positive-credit mass share: `0.5510 -> 0.4291`
    - `0..7`: `0.6170 -> 0.3846`
    - `8..15`: `0.6033 -> 0.3880`
    - `16..31`: `0.6712 -> 0.4812`
    - `32..63`: `0.7120 -> 0.5143`
  - interpretation:
    - missing episode-boundary semantics at SEP is likely part of the bug
    - but forcing SEP to become a full episode reset does not by itself produce a robust cross-seed performance win
    - treat this as evidence for a structural mismatch, not as a validated general repair
- A more local first-exploit-episode compare points to a hybrid first exploit episode as the main remaining structural problem.
  Evidence:
  - artifacts:
    - `/home/chen/RLPFN/artifacts/mainline_like_first_exploit_episode_probe_normal.json`
    - `/home/chen/RLPFN/artifacts/mainline_like_first_exploit_episode_probe_force_episode_reset_at_sep.json`
    - `/home/chen/RLPFN/artifacts/mainline_like_first_exploit_episode_compare.json`
  - under normal semantics:
    - `first_ep_0_15`: bad positive-credit mass share `0.3996`
    - `first_ep_16_63`: `0.5638`
    - `preterminal_tail`: `0.4650`
    - `after_first_terminal`: `0.5608`
  - under `force_episode_reset_at_sep`:
    - `first_ep_0_15`: `0.3837`
    - `first_ep_16_63`: `0.3621`
    - `preterminal_tail`: `0.3499`
    - `after_first_terminal`: `0.2982`
  - interpretation:
    - after forcing SEP to become a real episode reset, the `16..63` segment is no longer uniquely bad
    - the strong early/mid first-exploit-episode pathology largely flattens
    - this shifts the main structural explanation away from a “special bad token window” and toward a hybrid first exploit episode whose credit semantics are polluted by pre-SEP trajectory continuity
- Dropping the entire first exploit episode from the actor objective is **not** a fix.
  Evidence:
  - artifacts:
    - `/home/chen/RLPFN/artifacts/mainline_like_objective_alignment_compare_tokenwise_vs_post_first_terminal_only.json`
    - `/home/chen/RLPFN/artifacts/critic_free_cross_frozen_h_tokenwise_vs_post_first_terminal_only.json`
  - actor-signal alignment gets worse under `tokenwise_post_first_terminal_only`:
    - tokenwise `corr_mean_vs_delta = +0.2912`
    - post-first-terminal-only `corr_mean_vs_delta = +0.0660`
    - tokenwise trajectory-level `corr_raw_suffix_vs_delta = +0.1787`
    - post-first-terminal-only trajectory-level `corr_raw_suffix_vs_delta = -0.2695`
  - cross-`frozen_h_seed` single-env audit also gets worse:
    - baseline tokenwise mean `delta_smp_gap = +0.5948`, `positive_count = 2/4`
    - post-first-terminal-only mean `delta_smp_gap = -0.3159`, `positive_count = 1/4`
  - interpretation:
    - the whole first exploit episode is not “invalid credit”
    - it still contains useful exploit signal
    - the remaining bug is therefore narrower than “drop everything before the first post-SEP terminal”
    - the stronger structural reading is:
      - pre-SEP continuity pollutes part of the first exploit episode
      - but removing the whole episode throws away too much valid signal
- Isolating only **environment state continuity** at SEP explains a large part of the first-exploit-episode credit pathology, but is not a complete repair by itself.
  Evidence:
  - artifacts:
    - `/home/chen/RLPFN/artifacts/mainline_like_first_exploit_episode_probe_normal_vs_state_only_vs_force_reset.json`
    - `/home/chen/RLPFN/artifacts/critic_free_cross_frozen_h_normal_vs_state_only_reset_compare.json`
  - under `reset_env_state_at_sep_keep_actor_history`:
    - `first_ep_0_15` bad positive-credit mass share: `0.3996 -> 0.3100`
    - `first_ep_16_63`: `0.5638 -> 0.2783`
    - `preterminal_tail`: `0.4650 -> 0.4534`
    - `after_first_terminal`: `0.5608 -> 0.4065`
  - cross-`frozen_h_seed` single-env audit:
    - baseline tokenwise mean `delta_smp_gap = +0.5948`, `positive_count = 2/4`
    - state-reset-only mean `delta_smp_gap = +0.5311`, `positive_count = 3/4`
    - seed behavior flips rather than cleanly dominates:
      - `12345`: `+4.7874 -> -0.3600`
      - `23456`: `-1.6493 -> +1.5133`
      - `34567`: `-1.2400 -> +0.2958`
      - `45678`: `+0.4814 -> +0.6752`
  - interpretation:
    - pre-SEP environment state continuity is a major contaminant of first-exploit-episode bad credit
    - but state reset alone is not a full fix
    - once state is reset while actor history/cache continuity stays intact, performance becomes mixed rather than consistently improved
    - this shifts the remaining root-cause focus to the **interaction between state continuity and actor-side history continuity**, rather than to either one in isolation
- Adding `clear_first_eval_history` on top of SEP state reset over-corrects.
  Evidence:
  - artifacts:
    - `/home/chen/RLPFN/artifacts/mainline_like_first_exploit_episode_probe_normal_vs_state_vs_state_plus_history_vs_force.json`
    - `/home/chen/RLPFN/artifacts/critic_free_cross_frozen_h_normal_vs_state_vs_state_plus_history_compare.json`
  - first-exploit-episode bad-credit distribution improves further under `reset_env_state_at_sep_and_clear_first_eval_history`:
    - `first_ep_0_15`: `0.3996 -> 0.1015`
    - `first_ep_16_63`: `0.5638 -> 0.1957`
  - but cross-`frozen_h_seed` single-env audit gets worse:
    - state-reset-only mean `delta_smp_gap = +0.5311`, `positive_count = 3/4`
    - state-reset-plus-history-clear mean `delta_smp_gap = -0.0972`, `positive_count = 1/4`
  - interpretation:
    - the incoming reward/action/terminal history at the first exploit token is not simply “bad signal” that should be zeroed
    - clearing it removes some of the same harmful continuity as state reset, but also removes useful exploit signal
    - therefore the remaining bug is not a generic “first exploit history should be blank” bug
    - the higher-confidence root-cause focus stays on a narrower continuity-contract mismatch, not on globally clearing first-eval history
- `reset_hidden_at_sep` alone does not repair the boundary bug.
  Evidence:
  - artifacts:
    - `/home/chen/RLPFN/artifacts/mainline_like_first_exploit_episode_probe_reset_hidden_at_sep.json`
    - `/home/chen/RLPFN/artifacts/critic_free_cross_frozen_h_reset_hidden_at_sep_compare.json`
  - first-exploit bad-credit mass does not improve:
    - `first_ep_0_15`: `0.3996 -> 0.6418`
    - `first_ep_16_63`: `0.5638 -> 0.5737`
  - cross-`frozen_h_seed` audit is worse than baseline:
    - baseline mean `delta_smp_gap = +0.5948`
    - hidden-only mean `delta_smp_gap = -0.6463`
  - interpretation:
    - recurrent hidden continuity is not the primary standalone contaminant
    - cutting hidden continuity in isolation is more consistent with a misrepair than with a root fix
- `clear_first_eval_reward_terminal_history` alone also does not repair the boundary bug.
  Evidence:
  - artifacts:
    - `/home/chen/RLPFN/artifacts/mainline_like_first_exploit_episode_probe_clear_first_eval_reward_terminal_history.json`
    - `/home/chen/RLPFN/artifacts/critic_free_cross_frozen_h_clear_first_eval_reward_terminal_history_compare.json`
  - first-exploit bad-credit mass remains high:
    - `first_ep_0_15`: `0.3996 -> 0.5861`
    - `first_ep_16_63`: `0.5638 -> 0.5985`
  - cross-`frozen_h_seed` audit is worse than baseline:
    - reward/terminal-only mean `delta_smp_gap = -0.8552`
  - interpretation:
    - reward/terminal history continuity is not the primary standalone contaminant either
    - the main structural signal still points to environment state continuity as the only channel whose isolated severing materially cleans up first-exploit bad credit without immediately collapsing cross-seed behavior
- Adding `reset_hidden_at_sep` on top of SEP state reset is not a cleaner root fix than state reset alone.
  Evidence:
  - artifacts:
    - `/home/chen/RLPFN/artifacts/mainline_like_first_exploit_episode_probe_reset_env_state_at_sep_and_reset_hidden_keep_actor_history.json`
    - `/home/chen/RLPFN/artifacts/critic_free_cross_frozen_h_reset_env_state_at_sep_and_reset_hidden_keep_actor_history_compare.json`
  - first-exploit bad-credit mass improves only weakly relative to normal and is clearly worse than state-reset-only on the core problematic segment:
    - `first_ep_0_15`: `0.3996 -> 0.3753` vs state-reset-only `0.3100`
    - `first_ep_16_63`: `0.5638 -> 0.5167` vs state-reset-only `0.2783`
  - cross-`frozen_h_seed` audit:
    - baseline mean `delta_smp_gap = +0.5948`, `positive_count = 2/4`
    - state-reset-only mean `delta_smp_gap = +0.5311`, `positive_count = 3/4`
    - state-reset-plus-hidden-reset mean `delta_smp_gap = +0.6399`, `positive_count = 2/4`
    - seed values:
      - `12345`: `+1.6280`
      - `23456`: `-0.4013`
      - `34567`: `-0.3864`
      - `45678`: `+1.7193`
  - interpretation:
    - hidden reset on top of state reset is not a stable additive repair
    - it partially gives back the structural bad-credit gains of state-reset-only while only marginally improving mean audit score
    - hidden continuity therefore does not currently look like the dominant remaining bug channel
- Current inductive reading:
  - the leading bug component is still `pre-SEP environment state continuity`
  - neither hidden continuity alone nor reward/terminal continuity alone explains the reversal
  - adding hidden reset on top of state reset does not currently look like the missing interaction-term fix either
  - the dominant structural-fix signal remains `reset_env_state_at_sep_keep_actor_history`
  - next best pure bug probe:
    - stop branching into more continuity cuts for now
    - instead test whether the state-reset-only semantics restore actor-objective alignment itself:
      - bad-credit distribution is already better
      - next verify whether actor-signal vs raw exploit improvement alignment improves enough to justify a root fix
    - do not reopen broad history clearing, hidden-only cuts, or broad actor shaping

Do not repeat by default:
- Re-running generic `ppo_n_epochs / lr / batch_size / target_kl` sweeps already shown to control overstep.
- Re-opening “is learned baseline the main culprit?” unless the value path changes.
- Re-opening “is the exploit suffix simply too short?” unless `n_steps` or phase semantics change.
- Re-opening token-wise-vs-trajectory-level ranking comparison unless the actor objective changes.
- Re-opening a pure `trajectory_suffix_return` actor-objective swap on the same `mainline_like` semantics.
- Re-opening the same bounded correction vs tokenwise comparison without widening scope or changing semantics.
- Re-opening the same bounded correction compare on the same single `frozen_h_seed` without changing the audit scope.
- Re-opening the same q1-drop cross-seed audit unless the boundary semantics change.
- Re-opening the same q1 contract-shift audit unless the boundary semantics change.
- Re-opening the same `force_episode_reset_at_sep` audit unless the episode-boundary semantics change.
- Re-opening the same first-exploit-episode normal-vs-force compare unless episode-boundary semantics change.
- Re-opening the same post-first-terminal-only actor-objective audit unless first-exploit-episode semantics change.
- Re-opening the same state-reset-only continuity probe on unchanged `mainline_like` semantics.
- Re-opening the same state-reset-plus-history-clear probe on unchanged `mainline_like` semantics.
- Re-opening hidden-only continuity cuts on unchanged `mainline_like` semantics.
- Re-opening reward/terminal-only continuity cuts on unchanged `mainline_like` semantics.
- Re-opening state-reset-plus-hidden-reset continuity cuts on unchanged `mainline_like` semantics.

Repeat only if:
- actor objective changes
- critic/value semantics change
- single-eval phase contract changes
- audit harness semantics change

### Invalid / Low-Value Repeats

- Do not repeat “generic PPO hyperparameter sweeps” unless one of:
  - actor objective changed
  - critic semantics changed
  - phase contract changed
- Do not repeat “learned vs zero baseline” checks on the same value path.
- Do not repeat “is exploit suffix too short?” checks on the same `mainline_like` phase contract.
- Do not repeat token-wise-vs-trajectory-level ranking comparison unless the actor objective changes.
- Do not repeat pure trajectory-level actor-objective swaps on the same `mainline_like` semantics.
- Do not repeat the same bounded correction on the same single `frozen_h_seed` without widening the audit scope.
- Do not repeat the same bounded correction vs tokenwise cross-seed audit unless the correction semantics change.
- Do not repeat the same q1-drop boundary audit unless the boundary semantics change.
- Do not repeat the same q1 contract-shift boundary audit unless the boundary semantics change.
- Do not repeat the same `force_episode_reset_at_sep` structural probe unless SEP episode-boundary semantics change.
- Do not repeat the same first-exploit-episode segment compare unless episode-boundary semantics change.
- Do not repeat the same post-first-terminal-only actor-objective swap unless first-exploit-episode semantics change.
- Do not repeat broad environment-stability sweeps on the same minimal-prior serial path.
- Prefer the next experiment only when it rules out exactly one unresolved cause.

## Mainline Preflight Gate

Before every new experiment, answer these four items in one short note:

1. Which mainline does this target?
- `Mainline 1`
- `Mainline 2`

2. Which currently unresolved uncertainty does it reduce?
- name exactly one primary uncertainty

3. Which already-credible conclusions does it rely on?
- list only the relevant ones from the ledger above

4. What result would actually change the next action?
- if no result would change the next action, do not run the experiment

## Current Mainline 2 Next Question

Primary next question:
- Under `mainline_like`, why does improvement in PPO actor objective still fail to stably convert into improvement in raw exploit suffix return?

Highest-value current direction:
- Continue direct actor-objective vs raw-exploit alignment checks under `mainline_like`
- Prefer experiments that discriminate:
  - actor-target mismatch
  - weak critic usefulness
  - remaining phase-contract mismatch
- Current best intervention direction:
- stop iterating whole-objective rewrites and broad actor-weight reshapes
- do not currently prioritize a full SEP-synced actor-GAE rewrite
- continue treating the remaining failure as boundary-local near the explore/exploit boundary
- prefer analyses that localize which suffix positions or return terms are driving negative updates
- most immediate next probe:
  - compare finer boundary-local actor warmup windows before any broader phase-contract rewrite
- after the SEP-synced ablation, next best step is to distinguish:
  - whether the remaining failure lives inside the first exploit episode after SEP, rather than at SEP itself
  - use structural probes before new shaping or hyperparameter changes
- after `force_episode_reset_at_sep`, the next best step is to localize which part of the first exploit episode still carries bad credit under otherwise cleaner episode semantics
  - current best structural reading:
    - the main bug is not a single special `16..63` window by itself
    - it is not “drop the whole first exploit episode” either
    - it is more likely a narrower contract mismatch inside the first exploit episode caused by pre-SEP continuity
  - next best pure bug probe:
    - isolate which pre-SEP carried variable is the critical contaminant:
      - environment state continuity
      - reward/action/terminal history continuity
      - recurrent hidden continuity
    - prefer bug probes that sever one continuity channel at a time without introducing a new shaping hyperparameter
  - current narrowed reading after the continuity probes:
    - environment state continuity is the only isolated channel with a strong structural-fix signal
    - hidden-only, reward/terminal-only, and state-plus-history-reset do not explain the reversal
    - state-plus-hidden-reset is not a convincingly better root-fix candidate than state-reset-only
  - next best pure bug probe:
    - keep `reset_env_state_at_sep_keep_actor_history` fixed
    - directly compare actor-objective alignment under:
      - normal semantics
      - state-reset-only semantics
    - if alignment meaningfully improves under state-reset-only, promote state continuity crossing SEP to the leading root-cause candidate
  - implementation status:
    - `mainline_like_objective_alignment_compare.py` now accepts `boundary_contract_mode`
    - the `normal` vs `reset_env_state_at_sep_keep_actor_history` alignment compare has landed
  - alignment evidence:
    - artifacts:
      - `/home/chen/RLPFN/artifacts/mainline_like_objective_alignment_compare_normal_vs_state_reset.json`
      - `/home/chen/RLPFN/artifacts/mainline_like_objective_alignment_compare_normal_repeat.json`
      - `/home/chen/RLPFN/artifacts/mainline_like_objective_alignment_compare_state_reset.json`
    - under `reset_env_state_at_sep_keep_actor_history`, actor-signal alignment improves materially relative to normal semantics:
      - `corr_mean_vs_delta`: `-0.1203 -> -0.0083`
      - `corr_pos_mass_vs_delta`: `-0.1706 -> -0.0217`
      - `high_nonpositive_delta_count`: `5/8 -> 3/8`
      - trajectory-level `corr_raw_suffix_vs_delta`: `-0.0766 -> +0.2523`
  - interpretation:
    - state-reset-only does not make tokenwise actor signal strongly correct, but it does move it from clearly anti-aligned toward near-neutral / partially aligned
    - this is enough to promote `pre-SEP environment state continuity crossing SEP` to the leading current root-cause candidate
    - the remaining mismatch now looks secondary rather than primary
  - main-training candidate status:
    - `ppo_reset_env_state_at_sep` is now wired into the main PPO build/train path rather than only audit-only boundary flags
    - touched code paths:
      - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/model_configs.py`
      - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/train.py`
      - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/sb3_recurrent_ppo.py`
      - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/priors/environment_prior.py`
    - observability added:
      - `train/sep_state_reset_enabled`
      - `train/sep_state_reset_count`
      - console `ppo-phases` line now prints `sep_state_reset` and `sep_resets`
    - direct verification:
      - `py_compile` passed on the touched files
      - targeted tests passed:
        - `build_recurrent_ppo_enables_sep_state_reset_on_main_training_path`
        - `step_wait_resets_state_at_sep_when_enabled`
  - fixed-env gradient-level compare:
    - artifact:
      - `/home/chen/RLPFN/artifacts/sep_state_reset_gradient_compare.json`
      - canonical legacy snapshot:
        - `/home/chen/RLPFN/artifacts/legacy_frozen_h_seed12345_sep_state_reset_gradient_snapshot.json`
      - debug note:
        - `/home/chen/RLPFN/artifacts/sep_state_reset_fix_debug.md`
    - setup:
      - same checkpoint
      - same frozen `h`
      - same train env seed
      - compare `normal` vs `ppo_reset_env_state_at_sep=True`
      - measure actor policy-loss gradient on the collected rollout buffer
    - result:
      - gradient cosine similarity is only `0.0653`
      - policy loss changes from `-0.2617` to `-0.0274`
      - `sep_state_reset_count` flips from `0` to `1`
    - legacy regression contract:
      - fixed legacy environment: `frozen_h_seed = 12345`
      - use this snapshot for future gradient-level numeric comparisons before changing main training semantics
      - target parameters:
        - `train_env_seed = 2020`
        - `single_eval_pos = 64`
        - `n_steps = 256`
        - `batch_size = 256`
        - `n_epochs = 1`
        - `learning_rate = 2e-4`
        - `target_kl = 0.03`
    - interpretation:
      - once the fix is on the main path, actor update direction on the same fixed environment changes materially rather than cosmetically
      - this is strong evidence that the candidate fix is active in real training semantics, not only in offline analysis
  - audit hygiene:
    - `critic_free_single_env_audit.py` now applies `ppo_reset_env_state_at_sep` to both the training prior and the fixed-environment evaluation prior
    - this avoids a semantic mismatch where training used the candidate environment fix but evaluation still measured the old environment contract
  - next step:
    - rerun the fixed-environment and cross-`frozen_h_seed` mainline-like audits through the real main-training flag
    - if they also improve, promote `state-reset-at-SEP` from candidate to mainline fix
    - if they do not, keep the fix candidate isolated and continue debugging the remaining actor-objective mismatch
  - long-run audit note:
    - a `n_steps=2048`, `outer_epochs=16`, `randomize_single_eval_pos=True`, `ppo_reset_env_state_at_sep=True` run completed successfully through the main training path
    - artifact payload showed:
      - `ppo_reset_env_state_at_sep = true`
      - `randomize_single_eval_pos = true`
      - `eval_single_eval_pos = 1975`
      - `delta_smp_gap = -0.8679`
  - critic-fit audit note:
    - debug note:
      - `/home/chen/RLPFN/artifacts/critic_fit_debug.md`
    - fixed legacy environment policy-help check:
      - artifact:
        - `/home/chen/RLPFN/artifacts/critic_help_vs_zero_state_reset_seed12345_quick.json`
      - result:
        - with `ppo_reset_env_state_at_sep = true`, `learned` baseline underperforms `zero` baseline on the fixed legacy environment:
          - learned `delta_smp_gap = +1.4883`
          - zero `delta_smp_gap = +3.2682`
    - fixed-buffer critic-only fit check:
      - artifact:
        - `/home/chen/RLPFN/artifacts/critic_value_fit_probe_state_reset_seed12345.json`
      - result:
        - critic quality starts near random:
          - step `0`: `raw_corr = -0.0311`, `explained_variance_raw = -0.000108`
        - but improves strongly under dedicated value-only fitting:
          - step `200`: `raw_corr = 0.4675`, `explained_variance_raw = 0.2184`
          - step `1000`: `raw_corr = 0.8006`, `explained_variance_raw = 0.6404`
      - reading:
        - critic is trainable on this fixed environment
        - current failure is more consistent with under-training / budget allocation than with an intrinsically impossible target
    - extra critic-updates audit:
      - artifacts:
        - `/home/chen/RLPFN/artifacts/critic_help_vs_zero_state_reset_seed12345_quick.json`
        - `/home/chen/RLPFN/artifacts/critic_help_vs_zero_state_reset_plus_extra_critic_seed12345_quick.json`
        - `/home/chen/RLPFN/artifacts/critic_help_vs_zero_state_reset_plus_head_only_extra_critic_seed12345_quick.json`
      - result:
        - current PPO:
          - learned `delta_smp_gap = +1.4883`
          - zero `delta_smp_gap = +3.2682`
        - with `extra_critic_updates_per_outer_epoch = 200`:
          - learned `delta_smp_gap = -1.9939`
          - zero `delta_smp_gap = +1.1190`
        - with `extra_critic_updates_per_outer_epoch = 200`, `head_only`:
          - learned `delta_smp_gap = -2.8659`
          - zero `delta_smp_gap = -2.6440`
      - reading:
        - naive extra critic-only updates on the current shared representation do not make learned critic superior to zero baseline
        - they worsen learned-baseline policy learning on this probe
        - head-only extra critic updates also fail, so simply narrowing update scope within the current architecture is not enough
        - critic fit is trainable in isolation, but “just add critic steps” is now a low-value direction on the current actor/critic stack
    - ranking-gated critic warmup audit:
      - artifact:
        - `/home/chen/RLPFN/artifacts/critic_warmup_ranking_gate_probe_state_reset_seed12345.json`
      - setup:
        - fixed legacy environment `frozen_h_seed = 12345`
        - repaired `SEP state reset` semantics on
        - actor kept on `zero` baseline until:
          - at least `2` outer epochs elapsed
          - and rollout critic `raw_corr >= 0.2`
      - result:
        - the handoff condition never fired in the first `4` outer epochs
        - critic rollout quality stayed poor throughout:
          - epoch 1: `raw_corr = 0.0044`, `EV = -0.4174`
          - epoch 2: `raw_corr = 0.0361`, `EV = -0.4396`
          - epoch 3: `raw_corr = -0.0431`, `EV = -0.5024`
          - epoch 4: `raw_corr = -0.2502`, `EV = -0.9920`
      - reading:
        - this is stronger than “warmup was weak”
        - under a reasonable ranking gate, the critic never becomes ready enough to hand actor control back to `learned`
        - so ranking-gated delayed critic usage does not currently rescue the shared structure
    - separate value-backbone smoke:
      - artifacts:
        - `/home/chen/RLPFN/artifacts/critic_help_vs_zero_state_reset_separate_value_backbone_seed12345_smoke.json`
        - `/home/chen/RLPFN/artifacts/critic_help_vs_zero_state_reset_shared_seed12345_smoke.json`
      - setup:
        - fixed legacy environment `frozen_h_seed = 12345`
        - repaired `SEP state reset` semantics on
        - matched smoke audit:
          - `eval_env_count = 2`
          - `n_steps = 256`
          - `outer_epochs = 1`
      - result:
        - shared backbone:
          - learned `delta_smp_gap = -1.6352`
          - zero `delta_smp_gap = -0.6843`
        - separate value backbone:
          - learned `delta_smp_gap = -0.2497`
          - zero `delta_smp_gap = -2.3034`
      - reading:
        - this is only a smoke, not a promotion test
        - but it is the first critic probe where decoupling changes the learned-vs-zero ordering in the expected direction
        - under matched settings, shared path still prefers `zero`, while separate value path prefers `learned`
      - decision:
        - if critic work continues, critic-path decoupling is now the leading direction
    - separate value-backbone positivity check:
      - current answer:
        - no
      - artifacts:
        - `/home/chen/RLPFN/artifacts/critic_help_vs_zero_state_reset_separate_value_backbone_seed12345_smoke.json`
        - `/home/chen/RLPFN/artifacts/critic_help_vs_zero_state_reset_separate_value_backbone_seed12345_quick.json`
      - result:
        - smoke:
          - learned `delta_smp_gap = -0.2497`
          - zero `delta_smp_gap = -2.3034`
        - completed quick:
          - learned `delta_smp_gap = -1.0881`
          - zero `delta_smp_gap = +0.5929`
      - reading:
        - the completed quick overrides the smoke-level optimism
        - under `SEP state reset` plus separate value backbone, learned critic is still negative and still worse than `zero`
        - therefore critic separation alone is not enough to pull the audit positive
    - separate critic underfit check:
      - artifact:
        - `/home/chen/RLPFN/artifacts/critic_value_fit_probe_state_reset_separate_value_backbone_seed12345.json`
      - result:
        - starting point remains near-random:
          - step `0`: `raw_corr = -0.0583`, `EV = -0.0010`
        - `50` critic-only updates are still not enough:
          - `raw_corr = 0.0605`, `EV = 0.0037`
        - `200` critic-only updates fit well:
          - `raw_corr = 0.7450`, `EV = 0.5545`
      - reading:
        - under separate value backbone, critic underfit is now a major remaining cause
        - the critic path can become useful, but it is still being consumed by actor before it is fitted enough
    - delayed critic handoff on separate path:
      - artifacts:
        - `/home/chen/RLPFN/artifacts/critic_warmup_ranking_gate_probe_state_reset_separate_value_backbone_seed12345_smoke.json`
        - `/home/chen/RLPFN/artifacts/critic_help_vs_zero_state_reset_separate_value_backbone_seed12345_outer4_smoke.json`
      - setup:
        - fixed legacy environment `12345`
        - `SEP state reset` on
        - separate value backbone on
        - `outer_epochs = 4`
        - warmup gate:
          - minimum `2` epochs
          - rollout critic `raw_corr >= 0.2`
      - result:
        - gate never fired
        - critic rollout quality stayed poor:
          - epoch 1: `raw_corr = 0.0495`
          - epoch 2: `raw_corr = 0.0675`
          - epoch 3: `raw_corr = 0.0142`
          - epoch 4: `raw_corr = -0.0891`
        - warmup `delta_smp_gap = +2.5499`
        - matched separate-baseline smoke:
          - learned `delta_smp_gap = -9.0706`
          - zero `delta_smp_gap = +7.5272`
      - reading:
        - delayed handoff does not make learned critic ready enough for actor use
        - the positive result comes from staying on `zero`, not from a successful handoff to `learned`
      - next direction:
        - if critic work continues, prefit the separate critic more aggressively before actor consumes it
    - separate critic prefit / warm-start:
      - artifact:
        - `/home/chen/RLPFN/artifacts/critic_prefit_warmup_critic_path_state_reset_separate_value_backbone_seed12345_smoke.json`
      - setup:
        - keep `SEP state reset`
        - keep separate value backbone
        - actor starts on `zero`
        - add `extra_critic_updates_per_outer_epoch = 200`
        - add `extra_critic_update_scope = critic_path`
        - keep handoff gate:
          - minimum `2` outer epochs
          - rollout `raw_corr >= 0.2`
      - result:
        - gate never fired
        - rollout critic quality remained poor:
          - epoch 1: `raw_corr = -0.0901`
          - epoch 2: `raw_corr = +0.0220`
          - epoch 3: `raw_corr = -0.0890`
          - epoch 4: `raw_corr = +0.0529`
        - actor stayed on `zero`
        - `delta_smp_gap = +0.3862`
      - reading:
        - stronger critic-path-only prefit still does not make the critic ready enough for live handoff
        - this variant is weaker than the simpler zero-only warmup and should not be promoted to long-run
    - interpretation:
      - the fix was active
      - but this run is not yet a clean decision point, because the audit also randomized the evaluation SEP and happened to land on a very late suffix boundary (`1975 / 2048`), leaving only a `73`-step suffix horizon
      - that makes the measured exploit-return gap much higher variance and less representative of the mainline objective than a fixed or controlled eval SEP
    - next audit hygiene requirement:
      - decouple training SEP randomization from evaluation SEP
      - keep training-side SEP random if desired, but evaluate under a fixed SEP (or a controlled SEP sweep) so long-run comparisons remain interpretable
  - fixed-eval long-run check:
    - with `single_eval_pos=64`, `n_steps=2048`, `outer_epochs=16`, and `ppo_reset_env_state_at_sep=true`, one completed long-run audit on `frozen_h_seed=12345` showed a strong improvement over the matched baseline:
      - baseline `delta_smp_gap = +1.4639`
      - state-reset `delta_smp_gap = +45.1348`
    - this is strong single-seed evidence that the candidate fix helps in the actual long-run training regime when evaluation SEP is controlled
    - however, cross-`frozen_h_seed` long-run confirmation is still incomplete, so the temporary default should still be treated as provisional until at least one additional long-run seed pair confirms the direction
    - attempted follow-up:
      - started a second long-run pair on `frozen_h_seed=23456` under the same `2048 / 16 / fixed SEP` regime
      - did not complete within the current interactive audit window, so it cannot be used as promotion evidence yet
    - decision constraint:
      - do not treat the new default as settled until the second long-run seed pair lands
  - critic phase-2 root-cause narrowing:
    - stop broad critic-rescue variants for now and focus on evidence that distinguishes:
      - plain optimization underfit
      - rollout-target instability
    - new current-code fixed-buffer rerun:
      - artifact:
        - `/home/chen/RLPFN/artifacts/critic_value_fit_probe_state_reset_separate_value_backbone_seed12345_rerun.json`
      - result:
        - step `0`: `raw_corr = 0.0631`, `EV = 0.0005`
        - step `200`: `raw_corr = 0.5667`, `EV = 0.3133`
      - reading:
        - under the current code, separate critic still can fit one fixed rollout buffer meaningfully
    - rollout-to-rollout transfer probe:
      - artifact:
        - `/home/chen/RLPFN/artifacts/critic_rollout_transfer_probe_state_reset_separate_value_backbone_seed12345.json`
      - setup:
        - keep `SEP state reset`
        - keep separate value backbone
        - fixed legacy environment `12345`
        - fit critic on one anchor rollout for `200` critic-only steps
        - keep actor frozen implicitly and verify param drift
        - then collect fresh rollouts with the same actor
      - result:
        - anchor rollout prefit:
          - `raw_corr = 0.0406`
          - `EV = 0.0008`
        - anchor rollout postfit:
          - `raw_corr = 0.4186`
          - `EV = 0.1710`
        - parameter deltas:
          - actor `L2 delta = 0.0`
          - value `L2 delta = 8.3352`
        - fresh-rollout target correlations vs anchor:
          - rollout 1: `raw_target_corr = -0.1721`
          - rollout 2: `raw_target_corr = +0.0696`
          - rollout 3: `raw_target_corr = -0.2195`
        - fresh-rollout raw-return means shift heavily:
          - anchor `-68.20`
          - rollout 1 `+21.24`
          - rollout 2 `+96.70`
          - rollout 3 `-276.07`
      - reading:
        - actor did not move, so the instability is not caused by policy drift
        - normalized-target correlation is similarly poor, so this is not mainly a raw-space de-normalization bug
        - the dominant remaining critic bottleneck now looks like:
          - **rollout-to-rollout Monte Carlo target instability under the current stochastic policy**
          - one sampled rollout is not a stable enough critic target for the next rollout handoff
      - implication:
        - do not claim the critic path is fundamentally wrong yet
        - do not keep tuning warmup gates in the dark
        - if critic debugging continues, the next high-value probes should target target-stability directly:
          - increase on-policy support per critic update
          - measure usefulness under reduced policy-sampling noise
          - or evaluate multi-rollout target averaging before actor consumption
    - critic rollout-contract root cause now locked:
      - exact source:
        - `_freeze_env_h_list_for_replay()` was not freezing env-construction latent uniforms used later by `_rollout_family_group_vectorized_with_policy()`:
          - `_constrained_obs_u`
          - `_constrained_noise_u`
          - `_ctrl_reward_enable_u`
          - `_survival_reward_enable_u`
      - consequence:
        - repeated “fixed-env” PPO rollouts could still change:
          - `obs_dim`
          - `noise_dim`
          - `zero_pad_dim`
          - exact reward-term enable flags
      - code locations:
        - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/train.py`
        - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/priors/environment_prior.py`
      - regression:
        - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_train_policy_rollout_checkpoint.py`
        - `test_freeze_env_h_list_for_replay_locks_env_construction_latents`
        - result: `1 passed`
    - after-fix forced-seed contract probe:
      - artifact:
        - `/home/chen/RLPFN/artifacts/critic_rollout_forced_seed_contract_probe_after_freeze_fix.json`
      - result:
        - run 1:
          - `obs_dim = 217`
          - `noise_dim = 117`
          - `zero_pad_dim = 12`
        - run 2:
          - `obs_dim = 217`
          - `noise_dim = 117`
          - `zero_pad_dim = 12`
        - `obs_shape_equal = true`
      - reading:
        - the previously drifting environment-construction contract is now fixed under forced seeds
    - after-fix seed-stability probe:
      - artifact:
        - `/home/chen/RLPFN/artifacts/critic_rollout_seed_stability_probe_state_reset_separate_value_backbone_seed12345_after_freeze_fix.json`
      - setup:
        - deterministic actor sampling
        - compare repeated rollouts under:
          - current `collect_rollouts()`
          - forced fixed `env_rng_seeds + rollout_rng_seeds`
      - result:
        - current repeated-rollout target corr vs anchor:
          - rollout 1: `0.4411`
          - rollout 2: `-0.2984`
        - forced-seed repeated-rollout target corr vs anchor:
          - rollout 1: `1.0000`
          - rollout 2: `1.0000`
      - reading:
        - rollout target stability is now achieved under the forced-seed contract
        - therefore “critic underfit” can be reconsidered, but only under this stabilized contract
    - stable-contract minimal recheck:
      - artifact:
        - `/home/chen/RLPFN/artifacts/critic_help_vs_zero_state_reset_separate_value_backbone_seed12345_forced_seed_minismoke.json`
      - setup:
        - fixed legacy `12345`
        - `SEP state reset`
        - separate value backbone
        - forced training seeds:
          - `env_rng_seed = 2020`
          - `rollout_rng_seed = 4040`
        - minimal scope:
          - `eval_env_count = 1`
          - `outer_epochs = 1`
      - result:
        - learned `delta_smp_gap = +4.6850`
        - zero `delta_smp_gap = +4.2308`
      - reading:
        - once rollout target drift is removed, learned critic is no longer obviously worse than zero in the fixed-env minimal recheck
        - this is not enough to promote critic as solved
        - but it is enough to reject the earlier stronger claim that critic still necessarily loses to zero after separation
    - stable-contract canonical quick:
      - artifact:
        - `/home/chen/RLPFN/artifacts/critic_help_vs_zero_state_reset_separate_value_backbone_seed12345_forced_seed_quick.json`
      - setup:
        - same fixed legacy `12345`
        - same `SEP state reset`
        - same separate value backbone
        - same forced training seeds:
          - `env_rng_seed = 2020`
          - `rollout_rng_seed = 4040`
        - canonical quick scope:
          - `eval_env_count = 8`
          - `outer_epochs = 2`
      - result:
        - learned `delta_smp_gap = +0.1315`
        - zero `delta_smp_gap = -1.9235`
      - reading:
        - under the stabilized rollout contract, learned critic stays competitive when scaled up to canonical quick
        - in this fixed-env canonical quick, learned clearly beats zero
        - therefore the earlier stronger “critic still loses to zero after separation” claim was confounded by rollout-contract drift
    - shared-backbone stable-contract canonical quick:
      - artifact:
        - `/home/chen/RLPFN/artifacts/critic_help_vs_zero_state_reset_shared_seed12345_forced_seed_quick.json`
      - setup:
        - same fixed legacy `12345`
        - same `SEP state reset`
        - shared actor/critic backbone
        - same forced training seeds:
          - `env_rng_seed = 2020`
          - `rollout_rng_seed = 4040`
        - canonical quick scope:
          - `eval_env_count = 8`
          - `outer_epochs = 2`
      - result:
        - learned `delta_smp_gap = +1.0642`
        - zero `delta_smp_gap = -2.2664`
      - reading:
        - under the same stabilized rollout contract, shared backbone also beats zero
        - therefore current evidence does not support “shared representation is the main critic bottleneck”
        - earlier shared-backbone negative readings were also confounded by rollout-contract drift
    - current phase-2 critic diagnosis update:
      - before the replay-contract fix, critic conclusions were confounded by rollout target drift
      - after the fix, the fixed-env stabilized compare now shows:
        - learned critic can beat zero in both minimal and canonical-quick rechecks
        - shared backbone can also beat zero in canonical quick under the same stabilized contract
      - so the remaining question is narrower:
        - why the unmodified PPO training path still departs from this stabilized contract
        - and which non-forced stochastic source collapses critic usefulness outside the controlled regime
    - seed-source ablation on default PPO training path:
      - artifact:
        - `/home/chen/RLPFN/artifacts/critic_rollout_seed_source_ablation_shared_seed12345.json`
      - script:
        - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/critic_rollout_seed_source_ablation.py`
      - setup:
        - fixed legacy `12345`
        - shared backbone
        - `SEP state reset`
        - deterministic actor sampling
        - compare four conditions:
          - none
          - `env_rng_seed` only
          - `rollout_rng_seed` only
          - both
      - result:
        - none:
          - rollout 1 corr `-0.1707`
          - rollout 2 corr `+0.0079`
        - env-only:
          - rollout 1 corr `+0.0524`
          - rollout 2 corr `+0.1774`
        - rollout-only:
          - rollout 1 corr `-0.2223`
          - rollout 2 corr `-0.5449`
        - both:
          - rollout 1 corr `+1.0000`
          - rollout 2 corr `+1.0000`
      - reading:
        - the stabilized contract requires both seed channels
        - neither `env_rng_seed` nor `rollout_rng_seed` alone is sufficient
        - current `collect_rollouts()` still deviates because it threads neither channel into `_rollout_family_group_vectorized_with_policy(...)`
      - decision:
        - do not globally fix these seeds in default stochastic training
        - but do formally thread both channels into PPO under an explicit strict/fixed-env mode for:
          - fixed-env regression
          - critic-fit debugging
          - repeated-rollout comparability audits
    - official strict/fixed-env mode landed in PPO path:
      - code:
        - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/sb3_recurrent_ppo.py`
      - formal parameters:
        - `strict_fixed_env_mode`
        - `env_rng_seeds`
        - `rollout_rng_seeds`
      - regression tests:
        - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_sb3_recurrent_ppo.py`
        - `3 passed`
      - official canonical quick reruns:
        - shared:
          - `/home/chen/RLPFN/artifacts/critic_help_vs_zero_state_reset_shared_seed12345_strict_fixed_env_quick.json`
          - learned `delta_smp_gap = -1.0160`
          - zero `delta_smp_gap = -2.2350`
        - separate value backbone:
          - `/home/chen/RLPFN/artifacts/critic_help_vs_zero_state_reset_separate_value_backbone_seed12345_strict_fixed_env_quick.json`
          - learned `delta_smp_gap = -0.7333`
          - zero `delta_smp_gap = -2.2814`
      - summary:
        - `/home/chen/RLPFN/artifacts/critic_strict_fixed_env_official_compare_summary.json`
      - reading:
        - official mode reproduces the important ranking-level conclusion from the earlier monkey-patched compares:
          - shared: learned beats zero
          - separate: learned beats zero
        - exact absolute `delta_smp_gap` parity is not preserved, so the trusted invariant is ranking-level critic usefulness under the stabilized contract, not exact numeric equality
      - decision:
        - retire monkey-patch compare results as primary evidence
        - use official strict/fixed-env mode as the only trusted fixed-env critic path going forward
    - numeric regression path for fixed-env critic compare:
      - issue:
        - official strict sampled mode preserved the learned-vs-zero ranking
        - but absolute `delta_smp_gap` could still flip sign and become unsuitable for long-term numeric regression
      - code update:
        - added official `deterministic_actor_sampling` flag to:
          - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/sb3_recurrent_ppo.py`
          - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/critic_free_single_env_audit.py`
      - regression tests:
        - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_sb3_recurrent_ppo.py`
        - deterministic actor sampling propagation and override checks added
        - `4 passed`
      - deterministic strict canonical quick reruns:
        - shared:
          - `/home/chen/RLPFN/artifacts/critic_help_vs_zero_state_reset_shared_seed12345_strict_fixed_env_deterministic_quick.json`
          - learned `delta_smp_gap = +0.7248`
          - zero `delta_smp_gap = +0.1518`
        - separate value backbone:
          - `/home/chen/RLPFN/artifacts/critic_help_vs_zero_state_reset_separate_value_backbone_seed12345_strict_fixed_env_deterministic_quick.json`
          - learned `delta_smp_gap = +0.1450`
          - zero `delta_smp_gap = -0.2084`
      - summary:
        - `/home/chen/RLPFN/artifacts/critic_strict_fixed_env_deterministic_regression_summary.json`
      - reading:
        - policy action sampling RNG was already controlled by `rollout_rng_seeds`; that was not the missing randomness channel
        - the sampled strict-mode sign flip is a variance problem, not a replay-contract failure
        - deterministic actor sampling restores positive numeric fixed-env deltas under the official stabilized contract
      - decision:
        - use official strict + deterministic actor sampling for fixed-env numeric regression
        - use official strict + sampled actor for ranking-level critic usefulness checks
    - formal official-vs-monkey regression locked down:
      - script:
        - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/critic_official_vs_monkey_regression.py`
      - artifact:
        - `/home/chen/RLPFN/artifacts/critic_official_vs_monkey_regression_seed12345.json`
      - compare protocol:
        - fixed legacy checkpoint
        - official strict fixed-env mode
        - deterministic actor sampling
        - same-instance official-vs-monkey toggle compare
      - important finding:
        - earlier large official-vs-monkey drift was mostly a compare-harness bug
        - two separately built algos do not start from identical PPO heads
        - unsynced differing tensors:
          - `action_net.weight`
          - `value_net.weight`
      - final regression result:
        - `policy_loss_abs_diff = 0.0`
        - `grad_cosine = 1.0000032186508179`
        - `grad_l2_delta = 0.0`
        - `actions_max_abs_diff = 0.0`
        - `returns_max_abs_diff = 0.0`
        - `values_max_abs_diff = 0.0`
        - `advantages_max_abs_diff = 0.0`
        - `log_probs_max_abs_diff = 0.0`
      - timing:
        - `official_mean_s = 5.2245651858`
        - `monkey_mean_s = 5.1640754031`
        - `official_over_monkey_ratio = 1.0117135746`
      - regression protection:
        - keep unit protection on:
          - strict fixed-env seed plumbing
          - deterministic actor sampling override
        - use the official regression script + artifact as the authoritative phase-2 numeric compare
      - decision:
        - the maintained official strict path is now the only trusted branch for phase 2 fixed-env critic work
        - retire separate-build official-vs-monkey comparisons as primary evidence
    - shared actor-critic milestone promoted:
      - artifact:
        - `/home/chen/RLPFN/artifacts/critic_help_vs_zero_state_reset_shared_seed12345_strict_fixed_env_deterministic_quick.json`
      - contract:
        - shared actor-critic backbone
        - `ppo_reset_env_state_at_sep = True`
        - `strict_fixed_env_mode = True`
        - `deterministic_actor_sampling = True`
        - `frozen_h_seed = 12345`
        - `train_env_seed = 2020`
        - `train_rollout_seed = 4040`
        - `single_eval_pos = 64`
        - `n_steps = 256`
        - `batch_size = 256`
        - `n_epochs = 1`
        - `outer_epochs = 2`
        - `target_kl = 0.03`
      - reference values:
        - `zero_suffix_return_mean = -0.5634000301`
        - learned:
          - `delta_smp_gap = +0.7247717381`
          - `post_smp_gap = +0.6964260340`
          - `post_return = +0.1330260038`
        - zero:
          - `delta_smp_gap = +0.1517906189`
          - `post_smp_gap = +0.1843308210`
          - `post_return = -0.3790692091`
      - reading:
        - this is the current phase-2 shared-backbone milestone
        - it is the maintained official branch's first positive-gap and positive-return fixed-env learned-critic result
      - decision:
        - use this artifact and contract as the baseline for later phase-2 maintenance and numeric comparisons
    - deterministic minibatch order formalized:
      - code:
        - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/sb3_recurrent_ppo.py`
      - formal parameter:
        - `deterministic_batch_plan`
      - audit/regression wiring:
        - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/critic_free_single_env_audit.py`
        - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/critic_official_vs_monkey_regression.py`
      - protection:
        - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_sb3_recurrent_ppo.py`
        - `4 passed`
      - updated official-vs-monkey artifact:
        - `/home/chen/RLPFN/artifacts/critic_official_vs_monkey_regression_seed12345.json`
      - result:
        - `policy_loss_abs_diff = 0.0`
        - `grad_cosine = 1.000003695487976`
        - `grad_l2_delta = 0.0`
        - rollout tensor diffs remain `0.0`
        - `official_over_monkey_ratio = 1.0160`
      - reading:
        - minibatch order is no longer controlled by audit-side monkey override
        - fixed-env numeric regression now explicitly locks all critical random sources
        - default stochastic training semantics and efficiency remain unchanged
      - decision:
        - phase-2 fixed-env numeric compare contract is now:
          - frozen env latent
          - fixed env seed
          - fixed rollout seed
          - deterministic actor sampling
          - deterministic minibatch order
    - phase 2 frozen for exploration; regression-only maintenance:
      - guardrail doc:
        - `/home/chen/RLPFN/artifacts/phase2_regression_guardrail.md`
      - maintained rerun entrypoint:
        - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase2_regression_suite.py`
      - maintained purpose:
        - rerun official-vs-monkey gradient regression
        - rerun shared-backbone fixed-env deterministic milestone compare
        - summarize pass/fail in one artifact
      - generated summary target:
        - `/home/chen/RLPFN/artifacts/phase2_regression_pack/phase2_regression_suite_summary.json`
      - current pack status:
        - fast assembly completed successfully
        - summary artifact present and usable
        - pass flags:
          - official-vs-monkey numeric match
          - shared learned beats zero
          - shared learned post return positive
        - manifest validation:
          - path:
            - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase2_regression_manifest.json`
          - `all_checks_pass = true`
        - protection behavior:
          - valid summary reused by default
          - output-dir lock prevents concurrent overwrite
          - validation failure is hard-fail by default
      - decision:
        - stop further phase-2 exploratory work
        - keep phase 2 only as a trusted regression gate for later changes
    - phase 3 guardrail pack added:
      - guardrail doc:
        - `/home/chen/RLPFN/artifacts/phase3_regression_guardrail.md`
      - regression entrypoint:
        - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_regression_suite.py`
      - manifest:
        - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_regression_manifest.json`
      - assembled summary:
        - `/home/chen/RLPFN/artifacts/phase3_regression_pack/phase3_regression_suite_summary.json`
      - validation:
        - `all_checks_pass = true`
      - protected conclusions:
        - trusted phase-3 longrun profile gradient alignment
        - fixed-suite checkpoint cross-env generalization numbers
        - trusted pair1 optimization baseline
    - phase 3 guardrail tightened; shortrun gradient regression quarantined:
      - longrun canonical updated to strict fixed-env / deterministic actor / deterministic batch contract
      - source artifact fingerprints added to phase-3 pack reuse logic
      - stable longrun canonical:
        - `/home/chen/RLPFN/artifacts/phase3_longrun_gradient_compare.json`
        - `resume_grad_norm = 8.315512657165527`
        - `phase3_isolated_grad_norm = 8.315511703491211`
      - quarantined diagnostic:
        - `/home/chen/RLPFN/artifacts/phase3_shortrun_profile_gradient_regression_v2.json`
        - instability summary:
          - `/home/chen/RLPFN/artifacts/phase3_shortrun_regression_instability_summary.json`
        - sequence-layout probe:
          - `/home/chen/RLPFN/artifacts/phase3_shortrun_sequence_layout_instability_probe.json`
      - reason:
        - repeated strict shortrun reruns still produced materially different absolute gradient magnitudes
        - deterministic algorithms alone did not remove the split
        - therefore shortrun remains informative but is not a trusted pass/fail regression target
    - phase 3 many-env optimization compare-contract bug identified and contained:
      - root cause:
        - `build_recurrent_ppo()` in the phase-3 optimization path was previously using fresh PPO heads instead of restoring the checkpoint validation PPO policy state
      - evidence:
        - restore probe:
          - `/home/chen/RLPFN/artifacts/phase3_restore_validation_policy_state_probe.json`
          - no-restore max abs diff vs saved PPO validation state `= 13.5944`
          - restore enabled max abs diff `= 0.0`
        - mismatch summary:
          - `/home/chen/RLPFN/artifacts/phase3_policy_restore_root_cause_summary.json`
          - old pair1 `pre_heldout_vs_zero.suffix_return_gap = 0.000276`
          - canonical pair1 `heldout_vs_zero.suffix_return_gap = 2.812291`
      - code:
        - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/sb3_recurrent_ppo.py`
        - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_multi_env_optimization_audit.py`
        - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_longrun_gradient_compare.py`
      - protection:
        - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_sb3_recurrent_ppo.py`
          - added restore-state regression
      - decision:
        - historical `phase3_multi_env_trusted_pair1.json` is quarantined
        - do not interpret its pre/post deltas as a trusted many-env optimization bottleneck
      - scope:
        - no impact on the trusted Phase 2 regression pack
        - no impact on checkpoint-level fixed-suite cross-environment baseline numbers
        - impact is limited to historical Phase 3 optimization paths that evaluated `pre` with a fresh-built PPO policy
      - runtime note:
        - restored-policy full pair1 rerun (`n_samples=2048`, `outer_epochs=1`, serial backend) did not finish within the interactive budget and was terminated
        - so the compare-contract bug is fixed in code, but a replacement trusted Phase-3 optimization artifact is still pending
## 2026-04-10 Phase 3 restored-policy rebaseline follow-up

- Scope check:
  - This does **not** affect Phase 2 trusted conclusions.
  - It affects only Phase 3 many-env optimization baselines that evaluate `pre` with PPO policy objects.
- compare-contract fix:
  - `build_recurrent_ppo(... restore_validation_policy_state=True)` now restores the saved validation PPO state in Phase 3 optimization entrypoints.
- follow-up runtime finding:
  - serial restored-policy pair1 rerun remains too slow to finish in the current interactive budget
  - artifact:
    - `/home/chen/RLPFN/artifacts/phase3_pair1_restored_policy_rerun/runtime_blocker_summary.json`
- backend propagation fix:
  - Phase 3 `pre/post` PPO evaluation now follows the explicit `rollout_backend` parameter instead of silently forcing serial.
  - guarded by:
    - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_phase3_multi_env_optimization_audit.py`
- backend divergence finding:
  - `family_vectorized` does not match `serial` closely enough for trusted restored-policy PPO pre/post evaluation
  - canonical probe:
    - `/home/chen/RLPFN/artifacts/phase3_backend_equivalence_probe_2048.json`
  - summary:
    - `/home/chen/RLPFN/artifacts/phase3_backend_divergence_summary.json`
  - key numbers:
    - full return mean abs diff: `6.2094`
    - suffix return mean abs diff: `0.0821`
    - runtime ratio family/serial: `1.0697`
- consequence:
  - `family_vectorized` cannot be used to establish a new trusted Phase 3 optimization baseline
  - Phase 3 remains diagnostic-only

## 2026-04-10 monitored serial rerun

- Added monitored Phase 3 rerun entrypoint:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_monitored_optimization_run.py`
- This does not change Phase 3 semantics. It only adds:
  - phase timing
  - periodic Python stack dumps
  - explicit runtime blocker evidence
- Monitored rerun result:
  - `/home/chen/RLPFN/artifacts/phase3_serial_monitored_runtime_summary.json`
- Observed:
  - `zero_train` finished in `134.29s`
  - run was still inside `zero_heldout` when interrupted at total wall `190.04s`
  - stack dumps stayed inside:
    - `_collect_suite_rewards_serial`
    - `_rollout_single`
    - `_pack_env_input`
    - `_apply_state_full_rms`
- New conclusion:
  - current restored-policy serial rerun is not blocked first by PPO optimization logic
  - it is already runtime-bound by serial exact-SCM zero-control suite evaluation

## 2026-04-10 Phase 3 trusted reuse progress

- Added strict reuse of canonical pair1 artifacts into Phase 3 optimization audit:
  - zero-control reuse
  - pre-policy checkpoint baseline reuse
- Both reuse paths validate:
  - suite fingerprint
  - `n_samples`
  - `single_eval_pos`
  - `rollout_backend`
  - `core_a=False`
  - `reference_semantics_enabled=False`
- Guard coverage:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_phase3_multi_env_optimization_audit.py`
  - current relevant test status: `10 passed`

- Pre-path equivalence evidence:
  - `/home/chen/RLPFN/artifacts/phase3_pre_path_equivalence_probe.json`
  - current Phase 3 `pre_train` path vs canonical checkpoint PPO baseline path:
    - full return mean abs diff: `9.16e-05`
    - suffix return mean abs diff: `3.64e-06`

- Monitored rerun after reusing zero + pre:
  - `/home/chen/RLPFN/artifacts/phase3_reuse_zero_pre_runtime_summary.json`
  - result:
    - zero recompute removed
    - pre recompute removed
    - `train_loop` completed in `103.09s`
    - next blocker is post-train policy evaluation

- Updated bottleneck order:
  1. zero-control recompute: solved
  2. pre-policy recompute: solved
  3. train loop: completes
  4. post-train serial policy eval: now dominant

## 2026-04-10 Phase 3 post-policy bundle split

- Added audit-only post-policy bundle save/resume support:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_multi_env_optimization_audit.py`
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_monitored_optimization_run.py`
- New arguments:
  - `--save-post-policy-bundle-path`
  - `--resume-post-policy-bundle-path`
- This does not change PPO semantics; it only removes duplicate recomputation when rebuilding Phase 3 diagnostics.

- Bundle-save monitored run:
  - `/home/chen/RLPFN/artifacts/phase3_pair1_post_policy_bundle/monitor_summary.json`
  - saved bundle:
    - `/home/chen/RLPFN/artifacts/phase3_pair1_post_policy_bundle/phase3_pair1_post_policy_bundle.pt`
  - result:
    - `train_loop` completed in `105.21s`
    - interruption only after entering `post_train`

- Post-only resumed run:
  - `/home/chen/RLPFN/artifacts/phase3_pair1_post_policy_only_pipe/monitor_summary.json`
  - `/home/chen/RLPFN/artifacts/phase3_pair1_post_policy_only_pipe/stack_dump.log`
  - result:
    - resumed directly into `post_train`
    - remained in `post_train` for `235.67s` before interruption
    - stack stayed inside serial exact-SCM rollout/policy-eval code

- Summary:
  - `/home/chen/RLPFN/artifacts/phase3_post_policy_bundle_runtime_summary.json`

- New conclusion:
  - the remaining Phase 3 rebuild blocker is no longer duplicated zero/pre/train work
  - the blocker is the cost of a single post-train serial exact-SCM policy evaluation

## 2026-04-10 Phase 3 post-eval profiler

- Added:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_post_eval_profile.py`
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/prior_generalization_audit.py`
    - `collect_suite_rewards_serial_profiled(...)`
- Guard status:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_phase3_multi_env_optimization_audit.py`
  - current relevant total: `13 passed`

- Real bug fixed:
  - initial profiler omitted `torch.no_grad()`
  - this produced false profiling OOM by accumulating autograd graphs
  - fixed and guarded by test

- Runtime constraint fixed in workflow:
  - concurrent GPU post-eval profilers are invalid
  - they caused false OOM
  - post-train and post-heldout profiling must be serialized

- Train profile:
  - `/home/chen/RLPFN/artifacts/phase3_post_train_profile_max4.json`
  - first 4 env rollout times:
    - `36.94s`
    - `35.87s`
    - `36.50s`
    - `34.84s`

- Heldout profile:
  - `/home/chen/RLPFN/artifacts/phase3_post_heldout_profile_max4.json`
  - first 4 env rollout times:
    - `38.30s`
    - `36.46s`
    - `35.78s`
    - `36.03s`

- Summary:
  - `/home/chen/RLPFN/artifacts/phase3_post_eval_profile_summary.json`

- New conclusion:
  - both post-train and post-heldout serial exact-SCM evals are slow at similar scale
  - first-4-env evidence does not indicate runtime concentration in a small number of pathological envs
  - current runtime blocker is the uniformly expensive serial rollout path itself

## 2026-04-10 Phase 3 post-eval hotspot profile

- Added:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_post_eval_hotspot_profile.py`
- Artifacts:
  - `/home/chen/RLPFN/artifacts/phase3_post_train_hotspot_env0.json`
  - `/home/chen/RLPFN/artifacts/phase3_post_heldout_hotspot_env0.json`
  - `/home/chen/RLPFN/artifacts/phase3_post_eval_hotspot_env0_summary.json`

- Script-level duplicate-run protection:
  - `phase3_post_eval_profile.py` now reuses an existing `output_json` unless `--overwrite`
  - `phase3_post_eval_hotspot_profile.py` now reuses an existing `output_json` unless `--overwrite`

- Locked hotspot numbers:
  - train env0 wall: `37.12s`
  - heldout env0 wall: `39.32s`
  - train env0 hotspot shares:
    - `_build_obs_from_rollout_step_inputs`: `12.70%`
    - `_apply_terminal_reset_step`: `11.04%`
    - `_terminal_tail_event_from_signal`: `8.40%`
    - `_apply_state_full_rms`: `1.71%`
  - heldout env0 hotspot shares:
    - `_apply_terminal_reset_step`: `13.72%`
    - `_build_obs_from_rollout_step_inputs`: `12.04%`
    - `_terminal_tail_event_from_signal`: `11.28%`
    - `_apply_state_full_rms`: `1.72%`

- New conclusion:
  - named hotspot mass is concentrated in observation build plus terminal-reset bookkeeping
  - `state_full_rms` is not a leading bottleneck under the current Phase 3 contract
  - train vs heldout single-env runtime remains same-order (`ratio = 1.059`)
  - no new trusted Phase 3 baseline is established yet

## 2026-04-10 Phase 3 post-eval subpath profile

- Added:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_post_eval_subpath_profile.py`
- Guard:
  - script reuses existing `output_json` unless `--overwrite`
- Artifacts:
  - `/home/chen/RLPFN/artifacts/phase3_post_train_subpath_env0.json`
  - `/home/chen/RLPFN/artifacts/phase3_post_heldout_subpath_env0.json`
  - `/home/chen/RLPFN/artifacts/phase3_post_eval_subpath_env0_summary.json`
- Test status:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_phase3_multi_env_optimization_audit.py`
  - current relevant total: `16 passed`

- Locked subpath numbers:
  - build-obs total:
    - train `11.86%`
    - heldout `11.79%`
  - build-obs dominant subpath:
    - `reward_mask_phase_terminal_write`
      - train `7.03%`
      - heldout `7.10%`
  - build-obs secondary subpath:
    - `action_tail_write`
      - train `2.60%`
      - heldout `2.58%`
  - terminal-reset total:
    - train `10.97%`
    - heldout `13.56%`
  - terminal-reset dominant subpath:
    - `terminal_event_call`
      - train `8.44%`
      - heldout `11.13%`
  - terminal-tail dominant inner subpaths are boundary computations:
    - train:
      - lower `1.54%`
      - upper `1.15%`
    - heldout:
      - lower `3.16%`
      - upper `2.38%`

- New conclusion:
  - the current serial exact-SCM blocker is not raw state RMS normalization
  - within `_build_obs_from_rollout_step_inputs`, the main cost is scalar slot write logic, not obs prefix copy
  - within terminal-reset, the main cost is `terminal_event_call`, specifically tail boundary math
  - heldout env0 is heavier than train env0 because tail-boundary work is triggered more often
  - no new trusted Phase 3 baseline is established yet

## 2026-04-10 Phase 3 runtime ratio vs Phase 2

- Summary:
  - `/home/chen/RLPFN/artifacts/phase3_vs_phase2_runtime_ratio_summary.json`

- Locked numbers:
  - Phase 2 trusted collect:
    - `0.02105s/step`
  - Phase 3 many-env collect:
    - isolated `0.02349s/step` (`1.116x`)
    - resumed `0.02494s/step` (`1.185x`)
  - Phase 3 single-env post-eval:
    - train env0 `0.01743s/step`
    - heldout env0 `0.01781s/step`
  - Phase 3 projected train+heldout full post-eval:
    - `1154.60s`

- New conclusion:
  - Phase 3 is not blocked by catastrophic per-step slowdown relative to the trusted Phase 2 branch
  - collect-path slowdown is moderate and tolerable for lightweight diagnostics
  - the blocking cost is full-suite serial post-eval scaling
  - therefore Phase 3 cross-environment bottleneck and bug testing should continue only in reuse-based lightweight form
  - no new trusted Phase 3 baseline is established yet

## 2026-04-10 Phase 3 lightweight many-env concentration probe

- Added:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_env_delta_concentration_probe.py`
- Guard:
  - script reuses existing `output_json` unless `--overwrite`
- Test status:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_phase3_multi_env_optimization_audit.py`
  - current relevant total: `17 passed`
- Artifacts:
  - `/home/chen/RLPFN/artifacts/phase3_env_delta_concentration_probe_max4.json`
  - `/home/chen/RLPFN/artifacts/phase3_env_delta_concentration_summary_max4.json`

- Locked numbers:
  - runtime:
    - train subset rollout `142.40s`
    - heldout subset rollout `141.71s`
    - combined `284.12s`
  - train subset `[0,1,2,3]`:
    - suffix gap delta mean `+0.0419`
    - positive count `3/4`
    - top-1 abs-mass share `0.538`
    - top-2 abs-mass share `0.930`
  - heldout subset `[0,1,2,3]`:
    - suffix gap delta mean `-0.1189`
    - positive count `1/4`
    - nonpositive count `3/4`
    - top-1 abs-mass share `0.698`
    - top-2 abs-mass share `0.859`

- New conclusion:
  - under the current lightweight trusted contract, train-side update benefit exists but is narrow and concentrated
  - heldout-side response in the tested subset is weaker and mostly nonpositive
  - many-env update bottleneck currently looks more like concentrated train improvement with poor heldout transfer than absent checkpoint semantics
  - this is still subset-only diagnostic evidence, not a new trusted Phase 3 baseline

## 2026-04-10 Phase 3 lightweight train-rollout quality probe

- Added:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_train_rollout_quality_probe.py`
- Guard:
  - script reuses existing `output_json` unless `--overwrite`
  - it performs only one restored-policy many-env `collect_rollouts()` on the fixed train suite
  - it reuses canonical zero/pre artifacts and does not rebuild post eval
- Test status:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_phase3_multi_env_optimization_audit.py`
  - current relevant total: `18 passed`
- Artifacts:
  - `/home/chen/RLPFN/artifacts/phase3_train_rollout_quality_probe.json`
  - `/home/chen/RLPFN/artifacts/phase3_train_rollout_quality_summary.json`

- Locked numbers:
  - objective token count:
    - `102 / env` for all 16 train envs
  - positive normalized actor-advantage mass:
    - top-1 env share `0.2102`
    - top-2 env share `0.3429`
    - dominant env index `10`
  - full-train-suite correlations:
    - `corr(pre_suffix_gap, positive_norm_mass) = -0.2885`
    - `corr(terminal_resets, positive_norm_mass) = +0.5392`
    - `corr(value_corr, positive_norm_mass) = +0.3591`
  - subset `[0,1,2,3]` vs already measured post deltas:
    - `corr(delta, pre_suffix_gap) = +0.8768`
    - `corr(delta, positive_norm_mass) = -0.9263`
    - `corr(delta, value_corr) = +0.8531`
    - `corr(delta, terminal_resets) = -0.0085`
    - `corr(delta, last16_positive_share) = -0.0813`

- New conclusion:
  - Phase 3 many-env update concentration is not explained by objective token count; that term is fixed
  - raw positive-mass concentration is also not sufficient to explain realized positive delta
  - within the currently measured subset, positive improvement aligns much better with:
    - pre-existing train suffix quality
    - critic raw value-return alignment
  - and aligns poorly with:
    - terminal-reset count
    - objective-tail last-16 positive-mass share
  - this shifts the current bottleneck reading from “mass concentration only” to “rollout/objective quality mismatch”
  - still diagnostic-only; no new trusted Phase 3 baseline

## 2026-04-10 Phase 3 train-env quality contrast

- Added:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_train_env_quality_contrast_probe.py`
- Guard:
  - script reuses existing `output_json` unless `--overwrite`
  - script only reads the canonical rollout-quality artifact
- Test status:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_phase3_multi_env_optimization_audit.py`
  - current full-file total: `15 passed`
- Artifact:
  - `/home/chen/RLPFN/artifacts/phase3_train_env_quality_contrast_probe.json`
  - `/home/chen/RLPFN/artifacts/phase3_train_env_quality_contrast_regime_pair1_positive.json`

- Locked grouping:
  - subset mass median `25.2618`
  - high-mass nonpositive envs: `[1]`
  - low-mass positive envs: `[0, 3]`
  - high-mass positive envs: `[2]`

- Locked contrasts:
  - low-mass positive minus high-mass nonpositive:
    - `pre_suffix_gap = +5.8694`
    - `raw value-return corr = +0.0589`
    - `first16_positive_share = +0.1782`
    - `last16_positive_share = -0.1410`
    - `episode_count = 0.0`
    - `terminal_reset_count = 0.0`

- New conclusion:
  - in the measured train subset, “high positive mass but no gain” is distinguishable from “lower mass but gain” mainly by:
    - larger existing `pre_suffix_gap`
    - better `raw value-return corr`
    - token-quality placement
  - it is **not** primarily distinguished by:
    - objective episode count
    - terminal-reset count
  - current Phase 3 bottleneck reading therefore tightens to:
    - train-update effectiveness depends more on token-quality placement and critic alignment than on raw positive-mass size
  - the current train-side artifact is pair1-positive regime labeled and fingerprint-verified, so this is not a mixed-suite proxy
  - the current train-side artifact is pair1-positive regime labeled and fingerprint-verified, so this is not a mixed-suite proxy

## 2026-04-10 Phase 3 objective token-bucket probe

- Added:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_train_token_bucket_probe.py`
- Guard:
  - script reuses existing `output_json` unless `--overwrite`
  - same lightweight contract as the train-rollout quality probe
- Test status:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_phase3_multi_env_optimization_audit.py`
  - current relevant total: `20 passed`
- Artifacts:
  - `/home/chen/RLPFN/artifacts/phase3_train_token_bucket_probe.json`
  - `/home/chen/RLPFN/artifacts/phase3_train_token_bucket_summary.json`

- Locked numbers:
  - runtime wall `101.04s`
  - subset bucket alignment vs `suffix_gap_delta`:
    - `Q1 positive_mass_share = -0.6810`
    - `Q2 positive_mass_share = -0.6538`
    - `Q3 positive_mass_share = +0.8468`
    - `Q4 positive_mass_share = +0.6420`
  - low-mass positive minus high-mass nonpositive:
    - `Q1 positive_mass_share = -0.2990`
    - `Q2 positive_mass_share = -0.1573`
    - `Q3 positive_mass_share = +0.0615`
    - `Q4 positive_mass_share = +0.3948`
    - `Q1 value_corr = +0.5608`

- New conclusion:
  - finer token buckets overturn the earlier coarse “early mass helps” reading
  - in the measured subset:
    - early-bucket positive mass (`Q1/Q2`) aligns with worse realized delta
    - later-bucket positive mass (`Q3/Q4`) aligns with better realized delta
    - `Q3` is the strongest positive-alignment bucket
  - successful low-mass envs differ from the failed high-mass env by:
    - less `Q1/Q2` positive mass
    - more `Q4` positive mass
    - better `Q1` value-return correlation
  - current Phase 3 bottleneck reading sharpens to:
    - many-env update quality depends on **bucket placement of objective-token mass** plus early-bucket critic alignment, not simply on total positive mass

## 2026-04-10 Phase 3 shortrun many-env sequence-layout source probe

- Added:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_shortrun_sequence_layout_source_probe.py`
- Guard:
  - script reuses existing `output_json` unless `--overwrite`
  - same strict numeric contract as Phase 2, with only many-env fixed-suite batching added
- Test status:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_phase3_shortrun_sequence_layout_source_probe.py`
  - current relevant total: `22 passed`
- Artifact:
  - `/home/chen/RLPFN/artifacts/phase3_shortrun_sequence_layout_source_probe.json`

- Locked findings:
  - repeated many-env shortrun `collect_rollouts()` is stable under the current strict contract:
    - step payload hashes match for `episode_starts`, `dones`, `rewards`, `actions`, `action_masks`
    - rollout-buffer hashes match for `episode_starts`, `returns`, `advantages`, `actor_advantages`
    - flat-batch hashes match for `seq_start_indices`, `seq_lengths`, `episode_starts`, `returns`
  - `episode_starts_follow_previous_dones = true`
  - `action_tail_nonzero_count = 0`
  - `seq_layout_matches_buffer_batch = true`
  - runtime stayed stable across repeated runs:
    - `114.82s`
    - `114.16s`

- New conclusion:
  - the previously quarantined shortrun sequence-layout instability does not reproduce on the current correct many-env fixed-suite collect path
  - terminal-reset / `episode_starts`, seq packing / flatten order, and masked action tail are therefore removed as active root-cause candidates for the current many-env collect path
  - this narrows the remaining Phase 3 bottleneck search space, but does not change the trust boundary:
    - Phase 2 remains the only trusted branch
    - Phase 3 remains diagnostic-only

## 2026-04-10 Phase 3 heldout token-bucket transfer probe

- Added:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_heldout_token_bucket_probe.py`
- Guard:
  - script reuses existing `output_json` unless `--overwrite`
  - same lightweight contract as the train token-bucket probe
  - reuses the canonical zero/pre baselines and heldout subset deltas instead of rebuilding them
- Test status:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_phase3_heldout_token_bucket_probe.py`
  - current relevant total: `23 passed`
- Artifact:
  - `/home/chen/RLPFN/artifacts/phase3_heldout_token_bucket_probe.json`

- Locked numbers:
  - runtime wall `102.00s`
  - heldout subset bucket alignment vs `suffix_gap_delta`:
    - `Q1 positive_mass_share = +0.4197`
    - `Q2 positive_mass_share = +0.8897`
    - `Q3 positive_mass_share = -0.1213`
    - `Q4 positive_mass_share = -0.8492`
  - heldout low-mass positive minus high-mass nonpositive:
    - `Q1 positive_mass_share = -0.5840`
    - `Q2 positive_mass_share = -0.2893`
    - `Q3 positive_mass_share = -0.1027`
    - `Q4 positive_mass_share = -0.0239`
    - `Q1 value_corr = +0.0110`

- New conclusion:
  - the train-side `Q3/Q4` success signature does **not** transfer to the measured heldout subset
  - in the heldout subset, the only positive-delta case is dominated by `Q1/Q2` positive mass rather than `Q3/Q4`
  - therefore the current Phase 3 bottleneck reading tightens again:
    - train-side no-gain cases are better explained by early-bucket low-quality mass and weaker early-bucket critic alignment
    - heldout-side non-transfer is explained by the absence of the train-side late-bucket success pattern
  - this remains subset-only diagnostic evidence, not a new trusted Phase 3 baseline

## 2026-04-10 Phase 3 transfer bottleneck synthesis

- Added:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_transfer_bottleneck_probe.py`
- Guard:
  - script reuses existing `output_json` unless `--overwrite`
  - pure synthesis over the canonical train/heldout token-bucket artifacts
  - no rollout collection and no rebuild work
- Test status:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_phase3_transfer_bottleneck_probe.py`
  - current relevant total: `24 passed`
- Artifact:
  - `/home/chen/RLPFN/artifacts/phase3_transfer_bottleneck_probe.json`

- Locked numbers:
  - runtime wall `0.0017s`
  - sign-flip buckets:
    - `["q1", "q2", "q3", "q4"]`
  - train positive `Q12 - Q34 = -0.3584`
  - heldout positive `Q12 - Q34 = +0.9219`
  - train positive pre-gap minus nonpositive = `+3.9575`
  - heldout positive pre-gap minus nonpositive = `-0.4330`

- New conclusion:
  - the current train-vs-heldout discrepancy is **mode inversion**, not simple weakening
  - train-side positive cases are later-bucket (`Q3/Q4`) dominated and sit on higher pre-gap environments
  - heldout-side positive cases are earlier-bucket (`Q1/Q2`) dominated and sit on lower pre-gap environments
  - therefore the current many-env update is not merely failing to preserve the train success magnitude on heldout:
    - it is selecting a qualitatively different update mode on heldout
  - Phase 3 bottleneck reading now tightens to:
    - train side: early-bucket low-quality mass and weak early-bucket alignment
    - heldout side: transfer mode inversion relative to the train-side late-bucket success signature
  - diagnostic-only; no new trusted Phase 3 baseline

## 2026-04-10 Phase 3 heldout rollout-quality probe

- Added:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_heldout_rollout_quality_probe.py`
- Guard:
  - script reuses existing `output_json` unless `--overwrite`
  - one restored-policy heldout-suite collect only
  - reuses canonical zero/pre baselines and heldout subset deltas
- Test status:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_phase3_heldout_rollout_quality_probe.py`
  - current relevant total: `25 passed`
- Artifact:
  - `/home/chen/RLPFN/artifacts/phase3_heldout_rollout_quality_probe.json`

- Locked numbers:
  - runtime wall `104.70s`
  - heldout full-rollout coupling:
    - `corr_pre_suffix_gap_vs_positive_norm_mass = -0.1206`
    - `corr_value_corr_vs_positive_norm_mass = +0.2778`
    - `corr_terminal_resets_vs_positive_norm_mass = +0.9237`
    - `corr_last16_positive_share_vs_positive_norm_mass = -0.1284`
  - heldout subset alignment:
    - `corr_suffix_gap_delta_vs_pre_suffix_gap = -0.7658`
    - `corr_suffix_gap_delta_vs_positive_norm_mass = +0.1908`
    - `corr_suffix_gap_delta_vs_value_corr = -0.0615`
  - top heldout pre-gap envs:
    - env `12`: `pre_gap = 27.6898`, `positive_mass = 0.0`
    - env `11`: `pre_gap = 11.0164`, `positive_mass = 0.3290`

- New conclusion:
  - heldout signal is present; the heldout suite is not well-described as “too hard to learn at all”
  - positive update mass on heldout is much more reset-driven than pre-gap-driven
  - the cleanest highest-pre-gap heldout envs are nearly starved of positive mass
  - this points to a heldout-side selection problem in the current many-env update

## 2026-04-10 Phase 3 root-cause synthesis

- Added:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_root_cause_probe.py`
- Guard:
  - script reuses existing `output_json` unless `--overwrite`
  - pure synthesis over canonical diagnostics
  - no rollout collection
- Test status:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_phase3_root_cause_probe.py`
  - current relevant total: `26 passed`
- Artifact:
  - `/home/chen/RLPFN/artifacts/phase3_root_cause_probe.json`

- Locked numbers:
  - runtime wall `0.0027s`
  - sequence-layout:
    - `collect_path_stable = true`
  - transfer:
    - all bucket alignment signs flip between train and heldout
    - transfer pattern is mode inversion, not simple weakening
  - heldout top-2 pre-gap env capture:
    - envs `[12, 11]`
    - mean pre-gap `19.3531`
    - positive-mass share `0.000626`

- New conclusion:
  - current evidence supports an **optimization selection bias**
  - current evidence does **not** support:
    - an active collect/layout bug
    - a pure “task is unlearnable” explanation
    - a pure “advantage normalization caused the failure” explanation
  - the heldout reset-heavy bias is already visible in raw positive actor-advantage mass:
    - `corr_raw_positive_mass_vs_terminal_resets = +0.9203`
    - `corr_raw_positive_mass_vs_pre_suffix_gap = -0.1274`
  - therefore normalization may preserve this bias, but is not its primary source
  - the deepest current Phase 3 reading is:
    - train side: early-bucket low-quality mass and weak early-bucket alignment
    - heldout side: update mass is overly reset-driven and starves the cleanest high-pre-gap environments
    - transfer fails through mode inversion relative to the train-side late-bucket success signature
  - diagnostic-only; no new trusted Phase 3 baseline

## 2026-04-10 Heldout reset-semantics refinement

- Added probe:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_heldout_reset_semantics_probe.py`
- Added test:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_phase3_heldout_reset_semantics_probe.py`
  - focused run: `2 passed`
- Artifact:
  - `/home/chen/RLPFN/artifacts/phase3_heldout_reset_semantics_probe.json`
- Runtime:
  - wall `100.97s`

- New conclusion:
  - heldout reset-heavy env positive actor-advantage mass is **not** mainly piled into:
    - `post_first_reset`
    - `terminal_tail`
    - `post_reset_terminal_tail`
  - aggregate reset-heavy shares:
    - `first_episode.actor_adv_positive_mass_share = 0.5327`
    - `post_first_reset.actor_adv_positive_mass_share = 0.4673`
    - `terminal_tail.actor_adv_positive_mass_share = 0.4323`
    - `non_tail.actor_adv_positive_mass_share = 0.5677`
    - `post_reset_non_tail.actor_adv_positive_mass_share = 0.3037`
    - `post_reset_terminal_tail.actor_adv_positive_mass_share = 0.1636`
  - raw residual positive mass is even more concentrated in non-tail:
    - `non_tail.raw_residual_positive_mass_share = 0.8454`
    - `terminal_tail.raw_residual_positive_mass_share = 0.1546`
    - `post_reset_non_tail.raw_residual_positive_mass_share = 0.5312`
    - `post_reset_terminal_tail.raw_residual_positive_mass_share = 0.0095`
  - deepest current Phase 3 reading is sharpened to:
    - not a pure terminal-tail semantics bug
    - not a pure post-first-reset dominance bug
    - more likely reset-conditioned non-tail objective weighting / baseline interaction
  - diagnostic-only; no new trusted Phase 3 baseline

## 2026-04-10 Heldout layered mass decomposition

- Added probe:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_heldout_layered_mass_probe.py`
- Added test:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_phase3_heldout_layered_mass_probe.py`
  - focused run: `2 passed`
- Artifact:
  - `/home/chen/RLPFN/artifacts/phase3_heldout_layered_mass_probe.json`
- Runtime:
  - wall `101.91s`

- New conclusion:
  - earliest reset-heavy bias appears at **raw actor-advantage mass**
    - `earliest_reset_bias_layer = raw_actor_adv_positive_mass`
  - flags:
    - `raw_actor_objective_bias_supported = true`
    - `return_target_shape_bias_supported = false`
    - `critic_baseline_bias_supported = false`
    - `normalization_primary_driver_supported = false`
  - interpretation:
    - bias is created after return/residual, before normalization
    - primary culprit is objective weighting / mask semantics inside raw actor-advantage construction
  - diagnostic-only; no new trusted Phase 3 baseline

## 2026-04-10 Heldout segment ablation

- Added probe:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_heldout_segment_ablation_probe.py`
- Added test:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_phase3_heldout_segment_ablation_probe.py`
  - focused run: `1 passed`
- Artifact:
  - `/home/chen/RLPFN/artifacts/phase3_heldout_segment_ablation_probe.json`

- New conclusion (raw actor-adv positive mass):
  - base:
    - `corr_vs_terminal_resets = +0.7272`
    - `corr_vs_pre_suffix_gap = -0.2241`
  - `terminal_tail` ablation drops reset-correlation the most:
    - `corr_vs_terminal_resets = +0.3358`
  - `post_reset_terminal_tail` also drops reset-correlation:
    - `corr_vs_terminal_resets = +0.5026`
  - `post_reset_non_tail` is smaller:
    - `corr_vs_terminal_resets = +0.6373`
  - removing `non_tail` or `first_episode_non_tail` makes reset-correlation worse
  - interpretation:
    - the dominant bias driver in heldout is terminal-tail semantics,
      even though terminal-tail is not the majority mass holder
    - this is a bias source, not a direct fix (pre-gap capture does not improve)
  - diagnostic-only; no new trusted Phase 3 baseline

## 2026-04-10 Terminal-tail mask decomposition

- Added probe:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_terminal_tail_mask_probe.py`
- Added test:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_phase3_terminal_tail_mask_probe.py`
  - focused run: `1 passed`
- Artifact:
  - `/home/chen/RLPFN/artifacts/phase3_terminal_tail_mask_probe.json`
- Runtime:
  - wall `101.92s`

- New conclusion:
  - removing terminal-tail tokens collapses reset-bias as tail length grows
  - `tail_len=8` removal yields:
    - `terminal_tail corr_reset = +0.0402`
    - `corr_pre = +0.0430`
  - first-terminal tail is more biased than post-reset tail
  - primary bias source is terminal-tail mask semantics
  - diagnostic-only; no new trusted Phase 3 baseline

## 2026-04-10 Tail-bias propagation (raw return vs GAE-adv)

- Added probe:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_tail_bias_propagation_probe.py`
- Added test:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_phase3_tail_bias_propagation_probe.py`
  - focused run: `1 passed`
- Artifact:
  - `/home/chen/RLPFN/artifacts/phase3_tail_bias_propagation_probe.json`
- Runtime:
  - wall `103.69s`

- New conclusion:
  - raw return tail bias is already present:
    - `raw corr_reset = +0.5282` (all tail_len variants)
  - GAE reduces but does **not** remove tail bias:
    - `tail_len=8` terminal_tail `+0.2578`
    - `first_terminal_tail` `+0.4061`
    - `post_reset_terminal_tail` `+0.2122`
  - therefore bias is not created by GAE propagation; it is amplified/reshaped
  - root remains terminal-tail objective mask semantics
  - diagnostic-only; no new trusted Phase 3 baseline

## 2026-04-10 Terminal-tail rule decomposition (mask interactions)

- Updated probe:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_terminal_tail_mask_probe.py`
- Updated test:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_phase3_terminal_tail_mask_probe.py`
  - focused run: `1 passed`
- Artifact:
  - `/home/chen/RLPFN/artifacts/phase3_terminal_tail_mask_probe.json`
- Runtime:
  - wall `105.75s`

- New conclusion:
  - `first_episode_only` mask is the strongest bias driver:
    - `corr_reset = +0.8456`
    - `corr_pre = -0.2792`
    - `top2_pre_gap_capture = 0.000084`
  - `post_reset_only` mask is comparatively benign:
    - `corr_reset = +0.2237`
    - `corr_pre = +0.2354`
    - `top2_pre_gap_capture = 0.1750`
  - terminal-tail still contributes, but is second-order to first-episode-only
  - root cause now sharpened to:
    - first-episode objective weighting
    - with terminal-tail interactions inside first episode
  - diagnostic-only; no new trusted Phase 3 baseline

## 2026-04-10 First-episode-only ablation verify

- Added probe:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_first_episode_ablation_verify.py`
- Added test:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_phase3_first_episode_ablation_verify.py`
  - focused run: `1 passed`
- Artifact:
  - `/home/chen/RLPFN/artifacts/phase3_first_episode_ablation_verify.json`

- New conclusion:
  - removing `first_episode_only` does **not** reduce reset-bias
  - base:
    - `corr_reset = +0.7225`
    - `corr_pre = -0.0421`
  - after ablation:
    - `corr_reset = +0.8456`
    - `corr_pre = -0.2792`
  - therefore `first_episode_only` is not a unique necessary condition
  - diagnostic-only; no new trusted Phase 3 baseline

## 2026-04-10 Post-reset-only ablation verify

- Added probe:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_post_reset_ablation_verify.py`
- Added test:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_phase3_post_reset_ablation_verify.py`
  - focused run: `1 passed` (conda env `rlpfn`)
- Artifact:
  - `/home/chen/RLPFN/artifacts/phase3_post_reset_ablation_verify.json`

- New conclusion:
  - removing `post_reset_only` **reduces** reset-bias and **improves** pre-gap alignment
  - base:
    - `corr_reset = +0.7225`
    - `corr_pre = -0.0421`
  - after ablation:
    - `corr_reset = +0.2237`
    - `corr_pre = +0.2354`
  - therefore `post_reset_only` is **not** a necessary condition for the bias
  - diagnostic-only; no new trusted Phase 3 baseline

## 2026-04-10 Tail interaction probe (objective_mask ∧ tail)

- Added probe:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_terminal_tail_interaction_probe.py`
- Added test:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_phase3_terminal_tail_interaction_probe.py`
  - focused run: `1 passed`
- Artifact:
  - `/home/chen/RLPFN/artifacts/phase3_terminal_tail_interaction_probe.json`

- New conclusion:
  - dominant reset-correlation bucket is `post_reset_tail_mass`:
    - `corr_reset = +0.9952`
    - `corr_pre = -0.2229`
    - `mass_share = 0.1765`
  - `terminal_tail_mass` overall remains strongly reset-correlated:
    - `corr_reset = +0.9317`
  - interaction term to prioritize:
    - `post_reset_only ∧ terminal_tail`
  - diagnostic-only; no new trusted Phase 3 baseline

## 2026-04-10 Post-reset terminal-tail ablation verify

- Added probe:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_post_reset_tail_ablation_verify.py`
- Added test:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_phase3_post_reset_tail_ablation_verify.py`
  - focused run: `1 passed` (conda env `rlpfn`)
- Artifact:
  - `/home/chen/RLPFN/artifacts/phase3_post_reset_tail_ablation_verify.json`

- New conclusion:
  - removing `post_reset_terminal_tail` reduces reset-bias and improves pre-gap alignment
  - base:
    - `corr_reset = +0.7225`
    - `corr_pre = -0.0421`
  - after ablation:
    - `corr_reset = +0.4034`
    - `corr_pre = +0.0713`
  - interaction term is a strong contributor but not the only bias source
  - diagnostic-only; no new trusted Phase 3 baseline

## 2026-04-10 First-episode tail vs non-tail ablation verify

- Added probe:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_first_episode_tail_ablation_verify.py`
- Added test:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_phase3_first_episode_tail_ablation_verify.py`
  - focused run: `1 passed` (conda env `rlpfn`)
- Artifact:
  - `/home/chen/RLPFN/artifacts/phase3_first_episode_tail_ablation_verify.json`

- New conclusion:
  - removing `first_episode_tail` slightly reduces reset-bias but worsens pre-gap alignment
  - removing `first_episode_non_tail` worsens both reset-bias and pre-gap alignment
  - therefore neither is a minimal necessary bias source
  - dominant remaining lever remains post-reset terminal-tail
  - diagnostic-only; no new trusted Phase 3 baseline

## 2026-04-10 Post-reset terminal-tail band probe

- Added probe:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_post_reset_tail_band_probe.py`
- Added test:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_phase3_post_reset_tail_band_probe.py`
  - focused run: `1 passed` (conda env `rlpfn`)
- Artifact:
  - `/home/chen/RLPFN/artifacts/phase3_post_reset_tail_band_probe.json`

- New conclusion:
  - all post-reset tail bands are strongly reset-correlated
  - last-token band is marginally the most correlated but not unique
  - no single tail_len cutoff isolates all bias
  - diagnostic-only; no new trusted Phase 3 baseline

## 2026-04-10 Post-reset tail episode/position split

- Added probe:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_post_reset_tail_episode_probe.py`
- Added test:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_phase3_post_reset_tail_episode_probe.py`
  - focused run: `1 passed` (conda env `rlpfn`)
- Artifact:
  - `/home/chen/RLPFN/artifacts/phase3_post_reset_tail_episode_probe.json`
- Runtime:
  - wall `103.94s`

- New conclusion:
  - episode-2 and episode-3+ tails are both strongly reset-biased
  - early vs late reset positions do not separate the bias
  - this split does not identify a narrower necessary condition
  - diagnostic-only; no new trusted Phase 3 baseline

## 2026-04-10 Post-reset tail distance/step probe

- Added probe:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_post_reset_tail_distance_probe.py`
- Added test:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_phase3_post_reset_tail_distance_probe.py`
  - focused run: `1 passed` (conda env `rlpfn`)
- Artifact:
  - `/home/chen/RLPFN/artifacts/phase3_post_reset_tail_distance_probe.json`
- Runtime:
  - wall `102.74s`

- New conclusion:
  - distance 1-4 tail tokens are the strongest reset-biased subset
  - late post-reset steps dominate both mass and bias
  - early post-reset steps are weaker and sparse
  - candidate narrow condition: post-reset tail distance ≤ 4 within late-step region
  - diagnostic-only; no new trusted Phase 3 baseline

## 2026-04-10 Post-reset tail dist≤4 & late-step ablation verify

- Added probe:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_post_reset_tail_dist4_late_ablation_verify.py`
- Added test:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_phase3_post_reset_tail_dist4_late_ablation_verify.py`
  - focused run: `1 passed` (conda env `rlpfn`)
- Artifact:
  - `/home/chen/RLPFN/artifacts/phase3_post_reset_tail_dist4_late_ablation_verify.json`
- Runtime:
  - wall `105.24s`

- New conclusion:
  - removing post-reset tail dist≤4 & late-step reduces reset-bias and improves pre-gap alignment
  - effect is material but not a full removal
  - diagnostic-only; no new trusted Phase 3 baseline

## 2026-04-10 Post-reset tail dist≤2 & late-step ablation verify

- Added probe:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_post_reset_tail_dist2_late_ablation_verify.py`
- Added test:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_phase3_post_reset_tail_dist2_late_ablation_verify.py`
  - focused run: `1 passed` (conda env `rlpfn`)
- Artifact:
  - `/home/chen/RLPFN/artifacts/phase3_post_reset_tail_dist2_late_ablation_verify.json`
- Runtime:
  - wall `103.70s`

- New conclusion:
  - dist≤2 late-step ablation yields negligible change
  - minimal necessary subset is larger than dist≤2
  - diagnostic-only; no new trusted Phase 3 baseline

## 2026-04-10 Next Probe Plan: Cross-Suite Repro + Semantic Contrast

Purpose:
- stop distance-threshold search
- establish **cross-suite reproducibility** and **semantic-contrast** evidence

Plan A: Cross-suite reproducibility
- run post-reset tail probes on at least two additional heldout suites (different suite seeds)
- compare bias direction/strength and mass concentration

Plan B: Semantic contrast
- run controlled contrast with reset semantics altered
  - disable terminal reset OR
  - null terminal-tail mask
- confirm whether bias collapses or flips

Plan C: Mechanism summary
- regress bias magnitude on reset count, episode length, tail occupancy
- produce a suite-invariant mechanism statement

Constraint:
- Phase 2 pack remains the hard gate
- no change to training semantics during probes

## 2026-04-11 Pair1 suite-matched repair and anchor constraints

- Locked repair:
  - use `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_suite_matched_gap_repair.py`
  - pair1 repaired artifacts are the only valid inputs for cross-suite/regime analysis:
    - `/home/chen/RLPFN/artifacts/phase3_post_reset_tail_distance_probe_pair1_suite_matched.json`
    - `/home/chen/RLPFN/artifacts/phase3_terminal_tail_mask_probe_pair1_suite_matched.json`
    - `/home/chen/RLPFN/artifacts/phase3_post_reset_tail_ablation_verify_pair1_suite_matched.json`
    - `/home/chen/RLPFN/artifacts/phase3_post_reset_ablation_verify_pair1_suite_matched.json`
- Quarantine:
  - do not use the old pair1 non-repaired artifacts to infer cross-suite or regime behavior
- Locked anchor read:
  - pair1 heavy mass is sparse reset-active structure, not a broad suite-wide effect
  - pair1 and pair2 both contain real positive-gain anchors outside the dominant `dist1_to_4_mass` proxy
  - pair2 env12 is a real baseline gain signal but a tiny-mass proxy outlier
- Immediate implication:
  - future optimization/fitting probes must separate:
    - proxy-captured reset-active anchors
    - proxy-missed or proxy-underweighted gain anchors

## 2026-04-11 Positive-regime structure split

- Locked probe:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_positive_regime_anchor_structure_probe.py`
- Locked artifact:
  - `/home/chen/RLPFN/artifacts/phase3_positive_regime_anchor_structure_probe.json`
- Locked reading:
  - proxy-captured strong gain anchors are reset-heavy and post-reset-tail weighted
  - proxy-missed strong gain anchors remain genuinely positive on suffix/full return
  - proxy-missed strong gain anchors are predominantly first-episode / non-reset weighted
- Audit implication:
  - future many-env objective probes must stop assuming reset-tail positive mass is sufficient coverage for gain
  - the next narrow design hypothesis is a two-channel objective/weighting:
    - retain reset-tail gain channel
    - add first-episode / non-reset gain coverage channel

## 2026-04-11 Dual-channel coverage follow-up

- Locked probe:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_dual_channel_anchor_coverage_probe.py`
- Locked artifact:
  - `/home/chen/RLPFN/artifacts/phase3_dual_channel_anchor_coverage_probe.json`
- Locked reading:
  - adding a first-episode / non-reset coverage channel materially improves strong-gain anchor coverage
  - but it does not eliminate misses
  - residual missed anchors are characterized by near-zero positive actor-adv mass, not just wrong channel weighting
- Audit implication:
  - next many-env optimization probe should not stop at channel design
  - it should trace why residual real-gain anchors produce low/zero positive actor-adv mass under the current objective
## 2026-04-12 nonpositive regime complement

- Completed the complementary regime-labeled train-side contrast for `seed24680`.
- Locked artifact:
  - `/home/chen/RLPFN/artifacts/phase3_train_env_quality_contrast_regime_seed24680_nonpositive.json`
- Key read:
  - `high_mass_nonpositive = [1, 2]`
  - `low_mass_positive = [0, 3]`
  - `high positive mass alone is not sufficient for positive delta`
- This closes the missing regime side of the train-side diagnostic without changing any core semantics.

## 2026-04-12 regime comparison summary pack

- Completed the positive vs nonpositive regime comparison pack.
- Locked artifact:
  - `/home/chen/RLPFN/artifacts/phase3_train_env_quality_regime_compare_pack.json`
- Key read:
  - shared signals survive both regimes
  - the only regime-specific separator is positive-only early-mass strength
  - pooled weighting repair is not yet justified
- This is the current decision gate before any weighting tweak.

## 2026-04-12 residual flip-mode compare pack

- Completed an active-search compare pack on the residual sign-formation layer.
- Locked artifact:
  - `/home/chen/RLPFN/artifacts/phase3_residual_flip_mode_compare_pack.json`
- Key read:
  - future-carry-dominant residual flips carry the stronger heldout-gain anchors and stronger raw residuals
  - bootstrap-dominant residual flips are more locally negative, but reach nearly the same final wrong-sign actor-adv magnitude
  - if a small repair is attempted next, the future-carry source chain is the narrower active target
- This is still diagnostic-only, not a new baseline.

## 2026-04-12 future-carry source-chain compare pack

- Completed the env12 vs env1 future-carry source-chain comparison.
- Locked artifact:
  - `/home/chen/RLPFN/artifacts/phase3_future_carry_source_chain_compare_pack.json`
- Key read:
  - env12 is terminal-reset-tail dominant
  - env1 is true-negative-future-residual dominant
  - the only shared negative source bucket is baseline_bootstrap_semantic_mismatch
  - this is the best shared small-fix candidate inside the future-carry chain, but branch-specific follow-up still remains necessary
- This remains diagnostic-only.

## 2026-04-12 baseline-bootstrap semantic mismatch compare pack

- Completed the follow-up compare pack that splits the shared baseline-bootstrap candidate into tail core vs mid-episode extension.
- Locked artifact:
  - `/home/chen/RLPFN/artifacts/phase3_baseline_bootstrap_semantic_mismatch_compare_pack.json`
- Key read:
  - both env12 and env1 share a near-terminal baseline-bootstrap core
  - env1 is tail-only for this bucket
  - env12 has an additional mid-episode extension
  - the narrow repair scope is `near_terminal_tail_core`
- This remains diagnostic-only.

## 2026-04-12 baseline-bootstrap terminal-adjacent core compare pack

- Completed the final shrink of the shared baseline-bootstrap axis to the terminal-adjacent core.
- Locked artifact:
  - `/home/chen/RLPFN/artifacts/phase3_baseline_bootstrap_terminal_adjacent_core_compare_pack.json`
- Key read:
  - both env12 and env1 share the terminal-adjacent baseline-bootstrap core
  - env12 still has a mid-episode extension
  - the narrow repair scope is `terminal_adjacent_core`
- Operational read:
  - treat `terminal_adjacent_core` as a guardrail
  - the remaining branch-specific direction is env12 mid-episode extension handling
- This remains diagnostic-only.

## 2026-04-12 env12 mid-episode extension compare pack

- Completed the env12-only branch-specific compare pack for the mid-episode extension block.
- Locked artifact:
  - `/home/chen/RLPFN/artifacts/phase3_env12_mid_episode_extension_compare_pack.json`
- Key read:
  - the env12 mid-episode extension is a contiguous four-token block
  - it is uniformly dominated by baseline_bootstrap_semantic_mismatch
  - no finer split inside the block is supported by current evidence
- Operational read:
  - keep `env12_mid_episode_extension_block` as the remaining branch-specific narrow control direction
  - keep `terminal_adjacent_core` as the shared guardrail
- This remains diagnostic-only.

## 2026-04-12 env12 mid-episode extension counterfactual pack

- Completed a read-only counterfactual pack on the env12 mid-episode extension block.
- Locked artifact:
  - `/home/chen/RLPFN/artifacts/phase3_env12_mid_episode_extension_counterfactual_pack.json`
- Key read:
  - replacing the block with `terminal_adjacent_core` materially shrinks GAE future-carry and raw residual
  - the shared guardrail remains unchanged
- Operational read:
  - keep `env12_mid_episode_extension_block` as a narrow branch-specific control candidate
  - keep `terminal_adjacent_core` as the shared guardrail
- This remains diagnostic-only.

## 2026-04-12 env12 mid-episode extension runtime control probe

- Completed a runtime semantic control probe for the env12 mid-episode extension block.
- Locked artifact:
  - `/home/chen/RLPFN/artifacts/phase3_env12_mid_episode_extension_runtime_control_probe.json`
- Key read:
  - the block can be realized as a contiguous four-token branch-specific gate
  - no shared-core edit is required
  - the shared guardrail remains unchanged
- Operational read:
  - keep `env12_mid_episode_extension_block` as a verified narrow control block
  - keep `terminal_adjacent_core` as the shared guardrail
- This remains diagnostic-only.

## 2026-04-12 env12 mid-episode extension runtime gate feasibility probe

- Completed a runtime feasibility probe for the env12 mid-episode extension block.
- Locked artifact:
  - `/home/chen/RLPFN/artifacts/phase3_env12_mid_episode_extension_runtime_gate_feasibility_probe.json`
- Key read:
  - the current runtime mode set does not exactly encode this internal four-token block
  - the smallest runtime edit surface is `MaskedRecurrentPPO._apply_actor_objective_postprocess`
  - no shared-core edit is required
- Operational read:
  - keep `env12_mid_episode_extension_block` as a runtime-feasible branch-specific control candidate
  - keep `terminal_adjacent_core` as the shared guardrail
- This remains diagnostic-only.

## 2026-04-12 env12 mid-episode extension runtime mode regression

- Completed a minimal env12 runtime-mode regression for the branch-specific mid-episode extension block.
- Locked artifact:
  - `/home/chen/RLPFN/artifacts/phase3_env12_mid_episode_extension_runtime_mode_regression.json`
- Key read:
  - runtime mode `tokenwise_scale_env12_mid_episode_extension_block` reproduces the same env12 negative-carry shrink
  - only objective positions `18..21` in objective episode `0` are scaled
  - the shared core remains unchanged
- Operational read:
  - keep `env12_mid_episode_extension_block` as a runtime-feasible branch-specific control candidate
  - keep `terminal_adjacent_core` as the shared guardrail
- This is the first runtime-level confirmation that the narrow branch-specific control can be realized without a shared-core edit.

## 2026-04-12 seed24680 strict train-side A/B for the branch-specific runtime mode

- Reran the train-side rollout-quality probe under the strict collect contract:
  - `strict_fixed_env_mode = true`
  - `deterministic_actor_sampling = true`
  - `deterministic_batch_plan = true`
  - train suite env/rollout seeds were wired directly from the suite artifact
- Locked result:
  - baseline and override artifacts match exactly on the quality metrics used by the side-effect compare pack
  - `max_abs_metric_delta = 0.0`
  - `runtime_mode_no_other_suite_side_effects = true`
- Operational read:
  - the branch-specific runtime mode remains local under the strict A/B contract
  - keep `env12_mid_episode_extension_block` as the runtime-feasible branch-specific control candidate

## 2026-04-13 seed13579 strict train-side A/B for the branch-specific runtime mode

- Repeated the same strict collect contract on a second non-target suite:
  - `strict_fixed_env_mode = true`
  - `deterministic_actor_sampling = true`
  - `deterministic_batch_plan = true`
  - train suite env/rollout seeds were wired directly from the suite artifact
- Locked result:
  - baseline and override artifacts match exactly on the quality metrics used by the side-effect compare pack
  - `max_abs_metric_delta = 0.0`
  - `runtime_mode_no_other_suite_side_effects = true`
- Operational read:
  - the runtime-feasible branch-specific control stays local on a second non-target suite
  - freeze `env12_mid_episode_extension_block` as a verified small control block

## 2026-04-13 pair2 strict train-side A/B for the branch-specific runtime mode

- Repeated the same strict collect contract on the pair2 suite path:
  - `strict_fixed_env_mode = true`
  - `deterministic_actor_sampling = true`
  - `deterministic_batch_plan = true`
  - train suite env/rollout seeds were wired directly from the suite artifact
- Locked result:
  - baseline and override artifacts match exactly on the quality metrics used by the side-effect compare pack
  - `max_abs_metric_delta = 0.0`
  - `runtime_mode_no_other_suite_side_effects = true`
- Operational read:
  - the runtime-feasible branch-specific control stays local on a third non-target suite path
  - freeze `env12_mid_episode_extension_block` as a verified small control block

## 2026-04-14 pair2 proxy-to-train transfer audit

- Built a read-only audit to explain why the `env12_mid_episode_extension_block` local proxy gain did not turn into train-side gain.
- Locked artifact:
  - `/home/chen/RLPFN/artifacts/phase3_pair2_proxy_to_train_transfer_audit.json`
- Key read:
  - the local carry proxy shrink is real, but the affected block is too small relative to the pair2 batch:
    - `block_token_fraction_of_batch = 0.2451%`
    - `block_abs_shrink_share_of_batch_abs_mass = 0.0079%`
    - `block_abs_shrink_share_of_batch_negative_mass = 0.0158%`
    - `block_abs_shrink_share_of_env12_negative_mass = 0.0862%`
  - target-side train A/B is therefore not surprising:
    - `train_full_return_delta_change = -0.018476486206054688`
    - `train_suffix_return_delta_change = 0.0`
- Operational read:
  - keep `env12_mid_episode_extension_block` as a runtime-feasible branch-specific control candidate only
  - do not expect local proxy shrink to imply train-side gain under the current many-env aggregation contract
  - next root-cause direction is `proxy_to_train_update_transfer_semantics`, not more block tuning
