# Phase 3 Regression Guardrail

## Status

Phase 3 is the active optimization frontier.

Phase 2 remains frozen and is only a preflight gate.

Current trust boundary:

- Phase 3 is diagnostic only.
- Phase 3 pack integrity may be green while Phase 3 claims remain untrusted.
- The only currently trusted branch is the maintained Phase 2 shared actor-critic backbone contract.
- Phase 2 pack green is now a code-level preflight requirement for Phase 3 tooling.
- Do not treat `rlpfn_maintained_path.py` defaults as the current trusted semantic source; use the milestone checkpoint config and the Phase 2 pack instead.

## Correct Contract

Phase 3 must be interpreted as:

- batch many-environment PPO
- one fixed `n_samples` rollout per environment
- each rollout may contain multiple internal episodes
- fixed `train_suite`
- fixed `heldout_suite`
- train on `train_suite`
- evaluate on both `train_suite` and `heldout_suite`

This is not a one-episode-per-environment regime.

Checked semantic facts for the current diagnostic lineage:

- exact-SCM in project terms:
  - `family = "scm"`
  - `strict_joint_transition_enabled = true`
- still using the maintained reward/state-control semantics seen in the fixed suites:
  - `state_full_rms_enabled = true`
  - `ctrl_reward_enable_prob = 0.7`
  - `reinforce_reward_transform = "tanh"`
  - `terminal_reset_enabled = true`
- train and heldout are separate fixed SCM batches, not one shared batch:
  - train `suite_seed = 12345`
  - heldout `suite_seed = 67890`
- evaluation is reproducible fixed-suite reinitialization, not fresh random re-sampling each call

## Cross-Suite Guardrail

- Cross-suite probes must not reuse pair1-specific heldout-delta artifacts on other suites.
- The forbidden stale input is:
  - `/home/chen/RLPFN/artifacts/phase3_env_delta_concentration_probe_max4.json`
- For non-pair1 suites, heldout per-env `pre_vs_zero_suffix_gap` must come from suite-matched:
  - zero baseline JSON
  - PPO baseline JSON
- Official fallback path:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_subset_delta_fallback.py`
- Covered by:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_phase3_subset_delta_fallback.py`
- Cross-suite baseline generation should also disable irrelevant action-reachability work:
  - add `--no-action-reachability-probe`
  - reason:
    - the reused baseline contract only needs suite metrics
    - action-reachability adds extra serial runtime but does not change the consumed baseline compare fields

Reset-count semantics are also locked:

- `objective_terminal_reset_count` must be derived from compressed objective-token segments, not from raw `episode_starts & objective_mask`.
- Official helper:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_heldout_reset_semantics_probe.py`
  - `_objective_terminal_reset_count_from_segments(...)`
- Covered by:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_phase3_heldout_reset_semantics_probe.py`
- Guardrail meaning:
  - no accepted probe artifact may contain rows with:
    - `objective_terminal_reset_count = 0`
    - and nonzero `post_reset_*` mass

## Current Diagnostic Record

1. The current Phase 3 longrun artifact exists and is preserved for comparison, but it is not trusted as a semantic baseline.
   Artifact:
   - `/home/chen/RLPFN/artifacts/phase3_longrun_gradient_compare.json`

2. The legacy `shortrun_aligned` profile mapping is now quarantined from pass/fail.
   Artifacts:
   - `/home/chen/RLPFN/artifacts/phase3_shortrun_profile_gradient_regression_v2.json`
   - `/home/chen/RLPFN/artifacts/phase3_shortrun_regression_instability_summary.json`
   - `/home/chen/RLPFN/artifacts/phase3_shortrun_sequence_layout_instability_probe.json`
   Reading:
   - exact-match still holds within a single run
   - cross-run gradient magnitude is not yet stable enough to use as a trusted numeric regression gate
   - the hidden drift is downstream of build/init:
     - PPO head hashes are stable
     - `obs` and `old_log_prob` can match while `actions`, `returns`, and sequence layout still drift

3. Cross-environment generalization has been observed at checkpoint level on fixed suites, but this remains a Phase 3 diagnostic observation rather than a trusted branch-level conclusion.
   Artifacts:
   - `/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/ppo_cross_env_baseline.json`
   - `/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/zero_cross_env_control.json`
   - `/home/chen/RLPFN/artifacts/phase3_cross_env_baseline_pair2/ppo_cross_env_baseline.json`
   - `/home/chen/RLPFN/artifacts/phase3_cross_env_baseline_pair2/zero_cross_env_control.json`

4. Current many-env PPO optimization remains weak on pair1 under the current baseline, but this is likewise diagnostic only.
   Artifact:
   - `/home/chen/RLPFN/artifacts/phase3_multi_env_trusted_pair1.json`

5. Cross-suite reproduction is now partially locked on pair2.
   Artifacts:
   - `/home/chen/RLPFN/artifacts/phase3_post_reset_tail_distance_probe_pair2.json`
   - `/home/chen/RLPFN/artifacts/phase3_post_reset_tail_dist4_late_ablation_pair2.json`
   Locked reading:
   - stable across pair1 and pair2:
     - near-terminal `post_reset_tail_dist_1..4` remains positively correlated with reset density and negatively correlated with heldout `pre_vs_zero_suffix_gap`
     - `dist<=4 & late-step` ablation reduces reset-bias and improves pre-gap alignment with nearly identical deltas on pair1 and pair2
   - not stable enough across pair1 and pair2:
     - the broad `post_reset_late_k4` bucket as a standalone root-cause summary
   Therefore:
   - protect the narrower “near-terminal post-reset tail” finding
   - do not promote the broader late-step aggregate as the trusted cross-suite mechanism yet

6. New suite `seed_24680` breaks the stronger pair1/pair2 interpretation.
   Artifacts:
   - `/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_24680/zero_cross_env_control.json`
   - `/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_24680/ppo_cross_env_baseline.json`
   - `/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_24680/phase3_post_reset_tail_distance_probe.json`
   - `/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_24680/phase3_terminal_tail_mask_probe.json`
   - `/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_24680/phase3_post_reset_tail_ablation_verify.json`
   - `/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_24680/phase3_post_reset_ablation_verify.json`
   Locked reading:
   - suite-level checkpoint heldout `PPO-vs-zero` flips negative:
     - suffix `ppo_minus_zero = -0.0708`
     - full `ppo_minus_zero = -1.2121`
   - `post_reset_tail_dist_1..4` still track reset density
   - but they no longer carry the pair1/pair2 negative `corr_vs_pre_suffix_gap`; they flip positive on this suite
   - semantic ablations on `post_reset_terminal_tail` and `post_reset_only` reduce `corr_reset` but make `corr_pre` more negative
   Therefore:
   - do not treat “post-reset tail bias directly harms pre-gap” as a suite-invariant law
   - before comparing `corr_vs_pre_suffix_gap` across suites, first record whether the suite-level heldout `PPO-vs-zero` sign is positive or negative
   - current stable claim is limited to:
     - post-reset-related mass tracks reset density
   - current unstable claim is:
     - that this same mass must always hurt heldout policy advantage

7. Four-group aggregation locks a stricter root-cause boundary.
   Artifact:
   - `/home/chen/RLPFN/artifacts/phase3_cross_suite_regime_summary.json`
   Entry point:
   - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_cross_suite_regime_summary.py`
   Covered suites:
   - `pair1`
   - `pair2`
   - `seed24680`
   - `seed13579`
   Locked reading:
   - stable:
     - near-terminal `post_reset_tail_dist_1..4` tracks reset density across all four groups
   - unstable:
     - the sign of its relation to `pre_vs_zero_suffix_gap`
     - the size and sign of semantic-ablation improvement
   Therefore:
   - do not frame the bottleneck as a single token-mask bug
   - frame it as a regime-conditioned optimization/fit bottleneck:
     - reset-heavy mass is fitted consistently
     - gain-sign semantics are not consistent across random groups
   Operational rule:
   - before using `pre_vs_zero_suffix_gap` as a token-quality target across suites, first partition suites by suite-level `ppo_minus_zero_suffix` sign

8. Regime-aware mixed-update analysis is now the canonical readout for pooled many-env token-quality claims.
   Artifact:
   - `/home/chen/RLPFN/artifacts/phase3_regime_mixed_update_probe.json`
   Entry point:
   - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_regime_mixed_update_probe.py`
   Covered by:
   - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_phase3_regime_mixed_update_probe.py`
   Locked reading:
   - pooled all-env `corr_dist1_to_4_mass_vs_pre_gap` can look weakly negative while still hiding an opposite-sign split across regimes
   - canonical booleans expected from the artifact:
     - `regime_split_visible_in_raw_correlation = true`
     - `pooled_sign_hides_regime_split = true`
     - `positive_regime_dominates_pooled_update = true`
   - current four-suite artifact shows:
     - positive regime:
       - `corr_dist1_to_4_mass_vs_pre_gap = -0.0479`
       - `mass_share = 0.6799`
       - `pooled_covariance_contribution_share = 0.9317`
     - nonpositive regime:
       - `corr_dist1_to_4_mass_vs_pre_gap = +0.0495`
       - `mass_share = 0.3201`
       - `pooled_covariance_contribution_share = 0.0683`
   Therefore:
   - do not interpret pooled token-quality correlation without regime stratification
   - do not assume the pooled sign is the mechanism
   - when analyzing many-env update conflict, always report:
     - suite regime split
     - per-regime correlation
     - per-regime mass share
     - per-regime pooled covariance contribution

9. Regime-internal suite/env concentration analysis is now required before any claim about “which environments dominate update”.
   Artifact:
   - `/home/chen/RLPFN/artifacts/phase3_regime_env_concentration_probe.json`
   Entry point:
   - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_regime_env_concentration_probe.py`
   Covered by:
   - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_phase3_regime_env_concentration_probe.py`
   Current rerun:
   - `2026-04-12` current code path reran cleanly and preserved the regime split / concentration reading.
   Locked reading:
   - nonpositive regime has a real local positive signal:
     - `local_corr_mass_vs_pre_gap = +0.0495`
     - `regime_local_covariance_numerator = +0.1511`
   - but it flips negative after pooled centering:
     - `pooled_covariance_numerator_contribution = -0.7622`
   - positive regime dominates both mass and pooled covariance:
     - `mass_share_global = 0.6799`
     - `pooled_covariance_numerator_contribution = -10.4017`
   - positive regime mass is itself concentrated:
     - positive regime `top3_share = 0.8432`
     - largest positive suite share within regime:
       - `pair1 = 0.8814`
   Operational rule:
   - do not collapse “update dominance” into one number
   - always separate:
     - top envs by update-mass proxy
     - top envs by pooled covariance contribution
   - because the current artifact shows they can differ sharply:
     - pair1 heavy-mass envs dominate update mass
     - pair2 env12 dominates absolute pooled covariance with tiny mass share
   Runtime note:
   - this probe is artifact-only aggregation and is seconds-scale
   - it is safe to keep in the Phase 3 guardrail path

10. Runtime guardrail on fresh cross-suite baselines is now explicit.
   - zero baseline with `--no-action-reachability-probe` is still a minutes-scale serial job
   - PPO baseline with `--no-action-reachability-probe` is a `10-20` minute serial job on fresh suites
   - current examples:
     - `seed24680` PPO baseline:
       - `ELAPSED_SEC = 1176.22`
     - `seed13579` PPO baseline:
       - `ELAPSED_SEC = 1122.79`
   Meaning:
   - this runtime profile is expected and should not be mistaken for a deadlock by itself
   - treat lack of CPU/GPU progress, not wall-clock alone, as the anomaly signal

## Regression Pack

Entry point:

- `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_regression_suite.py`

Manifest:

- `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_regression_manifest.json`

Default command:

```bash
cd /home/chen/RLPFN/reinforce-terminal-explore
python ticl/analysis/phase3_regression_suite.py
```

Preflight requirement:

- `/home/chen/RLPFN/artifacts/phase2_regression_pack/phase2_regression_suite_summary.json`
- must have:
  - `validation.all_checks_pass = true`
  - `pass_flags.official_vs_monkey_numeric_match = true`
  - `pass_flags.shared_learned_beats_zero = true`
  - `pass_flags.shared_learned_post_return_positive = true`
- otherwise Phase 3 pack assembly refuses to run

Output directory:

- `/home/chen/RLPFN/artifacts/phase3_regression_pack`

Generated summary:

- `/home/chen/RLPFN/artifacts/phase3_regression_pack/phase3_regression_suite_summary.json`

## Guardrail Behavior

- valid summary is reused by default to avoid repeated tests
- summary reuse is rejected automatically if any canonical source artifact hash changes
- output directory is protected by a lock file
- validation failure is a hard failure by default

## Pack Role

The Phase 3 pack now serves two limited purposes:

- preserve canonical Phase 3 artifacts and their fingerprints
- prevent silent overwrite or accidental reuse of stale summaries

It does not certify Phase 3 semantics as trusted.

It also does not permit stale cross-suite compare inputs. Entry commands and future probes must preserve suite identity all the way through zero/PPO baselines and heldout-delta derivation.

## Accept Criteria

1. Source artifact fingerprints remain stable unless explicitly re-baselined.
2. Canonical Phase 3 artifacts remain loadable and copied into the pack without silent drift.
3. Phase 2 guardrails remain the only hard semantic gate.

## Current Set

- diagnostic only:
  - `/home/chen/RLPFN/artifacts/phase3_longrun_gradient_compare.json`
  - fixed-suite checkpoint baselines
- quarantined legacy optimization artifact:
  - `/home/chen/RLPFN/artifacts/phase3_multi_env_trusted_pair1.json`
  - reason:
    - historical `pre` evaluation was taken from a fresh-built PPO policy rather than the checkpoint validation PPO policy state
    - this made the pre/post optimization deltas a compare-contract artifact, not a trustworthy many-env optimization baseline
  - restoration probe:
    - `/home/chen/RLPFN/artifacts/phase3_restore_validation_policy_state_probe.json`
    - `restore_validation_policy_state=False`:
      - max abs diff vs saved PPO validation state `= 13.5944`
    - `restore_validation_policy_state=True`:
      - max abs diff vs saved PPO validation state `= 0.0`
- quarantined:
  - `/home/chen/RLPFN/artifacts/phase3_shortrun_profile_gradient_regression_v2.json`
  - `/home/chen/RLPFN/artifacts/phase3_shortrun_regression_instability_summary.json`
  - `/home/chen/RLPFN/artifacts/phase3_shortrun_sequence_layout_instability_probe.json`
  - reason:
    - recurrent-sequence layout and resulting gradient magnitude are not numerically stable across repeated strict runs yet
    - deterministic algorithms alone are not sufficient to lock the shortrun numeric contract

## Handoff Rule

Phase 3 development should proceed only if:

- the Phase 2 regression pack is green
- the Phase 3 regression pack shows intact fingerprints and no silent artifact drift

Even when both conditions hold, Phase 3 findings remain exploratory until a future Phase 3 trust baseline is explicitly re-established.

## Current Root Cause

The most important current Phase 3 compare-contract bug is:

- `build_recurrent_ppo()` in the Phase 3 audit path was previously starting from fresh PPO heads instead of restoring the checkpoint validation PPO policy state

This matters because:

- checkpoint-level fixed-suite generalization is measured with the saved validation PPO policy
- historical Phase 3 optimization `pre` was measured with a different PPO policy object
- therefore old `pre/post` optimization deltas cannot be used as a trusted many-env bottleneck conclusion

Impact boundary:

- this does not change the trusted Phase 2 shared actor-critic milestone
- this does not change checkpoint-level cross-environment generalization baselines that already evaluate through the saved validation PPO policy
- it changes only the interpretation of historical Phase 3 optimization artifacts that started from a fresh-built PPO policy
# 2026-04-10 addendum: restored-policy Phase 3 baseline is still blocked

- The Phase 2 pack remains the only trusted hard gate.
- The Phase 3 compare-contract bug around fresh-built PPO policy vs saved validation PPO state is fixed.
- That fix did **not** yield a new trusted Phase 3 baseline yet, because:
  - serial restored-policy pre/post evaluation is too slow to complete within the current budget
  - `family_vectorized` is not numerically equivalent enough to replace `serial`

Reference artifacts:
- `/home/chen/RLPFN/artifacts/phase3_restore_validation_policy_state_probe.json`
- `/home/chen/RLPFN/artifacts/phase3_policy_restore_root_cause_summary.json`
- `/home/chen/RLPFN/artifacts/phase3_backend_equivalence_probe_2048.json`
- `/home/chen/RLPFN/artifacts/phase3_backend_divergence_summary.json`
- `/home/chen/RLPFN/artifacts/phase3_pair1_restored_policy_family_vectorized/runtime.json`

Operational rule:
- do not promote any Phase 3 many-env optimization result to trusted status unless it comes from a serial-correct restored-policy compare path
- do not use `family_vectorized` PPO pre/post evaluation as a trusted substitute

Runtime-control rule:
- if a restored-policy Phase 3 rerun exceeds the expected wall budget, use:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_monitored_optimization_run.py`
- this monitored runner is the canonical way to determine:
  - which phase is consuming time
  - whether the process is making forward progress
  - which Python stack segment dominates the overrun
- current monitored evidence shows the serial rerun is already runtime-bound in:
  - `zero_train`
  - then `zero_heldout`
  before any PPO `pre_train` / `post_train` comparison finishes

Trusted reuse rule:
- canonical pair1 artifacts may be reused to remove duplicate recomputation, but only through the audited reuse paths in:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_multi_env_optimization_audit.py`
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_monitored_optimization_run.py`
- allowed reuse categories:
  - zero-control
  - pre-policy checkpoint baseline
- required exact-match checks:
  - suite fingerprint
  - `n_samples`
  - `single_eval_pos`
  - `rollout_backend`
  - `core_a=False`
  - `reference_semantics_enabled=False`
- current evidence supporting pre-policy reuse:
  - `/home/chen/RLPFN/artifacts/phase3_pre_path_equivalence_probe.json`
  - batch=1 exact-SCM probe shows only `9.16e-05` full-return mean diff and `3.64e-06` suffix-return mean diff

Current Phase 3 runtime order after trusted reuse:
- `train_loop` completes
- next blocker is post-train serial PPO evaluation
- therefore future runtime work should target post-train / post-heldout policy evaluation, not zero-control or pre-policy recompute

Post-policy bundle rule:
- trusted runtime-control now includes a third reusable artifact class:
  - post-policy bundle saved immediately after `train_loop`
- canonical paths:
  - `/home/chen/RLPFN/artifacts/phase3_pair1_post_policy_bundle/phase3_pair1_post_policy_bundle.pt`
  - `/home/chen/RLPFN/artifacts/phase3_post_policy_bundle_runtime_summary.json`
- allowed usage:
  - resume directly into `post_train` / `post_heldout` evaluation without rerunning zero-control, pre-policy, or train-loop
- required exact-match checks are enforced on load:
  - checkpoint path
  - train/heldout suite fingerprints
  - `n_samples`
  - `single_eval_pos`
  - `rollout_backend`
  - training profile fields inherited from the trusted Phase 3 audit contract

Current interpretation:
- saving and resuming the post-policy bundle is trusted for runtime decomposition
- it has already established that:
  - duplicated earlier phases are no longer the blocker
  - `post_train` serial exact-SCM policy evaluation alone can still consume >`235s`
- this is sufficient to rule out “rerun overhead” as the main remaining blocker

Post-eval profiling rule:
- the canonical profiler for this runtime question is:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_post_eval_profile.py`
- it is diagnostic-only and does not change PPO semantics
- it relies on the trusted saved post-policy bundle and Phase 2 preflight
- profiling must be **serialized**
  - do not run train/heldout profiler jobs concurrently on the same GPU
  - concurrent profiler runs already produced false OOM and are now treated as invalid runtime evidence

Current profile-backed interpretation:
- artifacts:
  - `/home/chen/RLPFN/artifacts/phase3_post_train_profile_max4.json`
  - `/home/chen/RLPFN/artifacts/phase3_post_heldout_profile_max4.json`
  - `/home/chen/RLPFN/artifacts/phase3_post_eval_profile_summary.json`
- locked conclusion:
  - both `post_train` and `post_heldout` serial eval are slow at similar scale
  - first-4-env evidence does not support a single pathological-environment bottleneck
  - current runtime blocker is the uniformly expensive serial exact-SCM rollout path itself

Hotspot follow-up rule:
- the canonical single-env hotspot profiler is:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_post_eval_hotspot_profile.py`
- the canonical subpath profiler is:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_post_eval_subpath_profile.py`
- canonical artifacts:
  - `/home/chen/RLPFN/artifacts/phase3_post_train_hotspot_env0.json`
  - `/home/chen/RLPFN/artifacts/phase3_post_heldout_hotspot_env0.json`
  - `/home/chen/RLPFN/artifacts/phase3_post_eval_hotspot_env0_summary.json`
  - `/home/chen/RLPFN/artifacts/phase3_post_train_subpath_env0.json`
  - `/home/chen/RLPFN/artifacts/phase3_post_heldout_subpath_env0.json`
  - `/home/chen/RLPFN/artifacts/phase3_post_eval_subpath_env0_summary.json`
- duplicate-run protection:
  - both post-eval profiler scripts reuse an existing `output_json` by default
  - use `--overwrite` only when the compare contract changed
- locked hotspot interpretation:
  - `_build_obs_from_rollout_step_inputs` and terminal-reset bookkeeping dominate the named hotspots
  - `_apply_state_full_rms` is not a primary hotspot under the current trusted contract
  - single-env train vs heldout runtime remains in the same order of magnitude
  - deeper subpath interpretation:
    - `build_obs` cost is dominated by scalar slot writes plus masked action-tail writes, not raw obs prefix copy
    - terminal-reset cost is dominated by `terminal_event_call`
    - inside `terminal_tail_event_from_signal`, lower/upper tail boundary math is the dominant subpath
    - heldout env0 is heavier than train env0 mainly because terminal-tail boundary work is triggered more often

Runtime budget interpretation:
- canonical runtime summary:
  - `/home/chen/RLPFN/artifacts/phase3_vs_phase2_runtime_ratio_summary.json`
- locked interpretation:
  - Phase 3 many-env collect is only moderately slower than the Phase 2 trusted collect reference (`~1.12x` to `~1.18x` per step)
  - single-env post-eval per-step cost is not slower than the Phase 2 trusted per-step collect reference
  - but full train+heldout serial post-eval is still projected at `> 1150s`
- operational rule:
  - Phase 3 bottleneck/bug testing may continue only if it reuses:
    - zero-control baseline
    - pre-policy checkpoint baseline
    - saved post-policy bundle
  - do not launch full pair1 rebuild as an inner-loop diagnostic

Lightweight many-env bottleneck rule:
- canonical lightweight concentration probe:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_env_delta_concentration_probe.py`
- current canonical artifact:
  - `/home/chen/RLPFN/artifacts/phase3_env_delta_concentration_probe_max4.json`
  - summary:
    - `/home/chen/RLPFN/artifacts/phase3_env_delta_concentration_summary_max4.json`
- duplicate-run protection:
  - output reuse is default
  - use `--overwrite` only if contract inputs changed
- locked interpretation:
  - first-4-env train update is mildly positive but concentrated
  - first-4-env heldout update is mostly nonpositive and more concentrated
  - use this probe to test many-env bottlenecks before considering any heavier rebuild
  - do not promote its result to trusted status; it is subset-only diagnostic evidence

Lightweight train-rollout quality rule:
- canonical train-rollout quality probe:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_train_rollout_quality_probe.py`
- canonical artifacts:
  - `/home/chen/RLPFN/artifacts/phase3_train_rollout_quality_probe.json`
  - `/home/chen/RLPFN/artifacts/phase3_train_rollout_quality_summary.json`
- duplicate-run protection:
  - output reuse is default
  - use `--overwrite` only if contract inputs changed
- locked interpretation:
  - this probe is allowed because it only performs one full-train-suite `collect_rollouts()` and reuses canonical `zero/pre` artifacts
  - it is the canonical way to answer which rollout/objective-token quality is driving train-side update concentration
  - current locked reading:
    - objective token count is uniform and therefore not the driver
    - realized positive delta in the measured subset tracks pre-existing suffix gap and critic value quality more than raw positive-mass concentration
    - terminal-reset count and last-16 tail share are not the primary driver in the measured subset
  - do not escalate from this probe to full pair1 rebuild unless the Phase 2 pack remains green and the new contract target is explicitly justified

Train-env contrast rule:
- canonical contrast probe:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_train_env_quality_contrast_probe.py`
- canonical artifact:
  - `/home/chen/RLPFN/artifacts/phase3_train_env_quality_contrast_probe.json`
- regime-labeled artifact:
  - `/home/chen/RLPFN/artifacts/phase3_train_env_quality_contrast_regime_pair1_positive.json`
- duplicate-run protection:
  - output reuse is default
  - use `--overwrite` only if the source rollout-quality artifact changed
- locked interpretation:
  - use this probe to answer one narrow question only:
    - why some train envs show high positive advantage mass but no realized gain, while lower-mass envs do gain
  - the currently trusted train-side artifact is pair1-positive regime labeled and fingerprint-verified
  - current locked reading:
    - `pre_suffix_gap` and `raw value-return corr` separate these cases better than positive-mass magnitude
    - coarse early-vs-tail shares are not sufficient; use the token-bucket probe below for the canonical conclusion
    - objective episode count / terminal-reset count are not the primary separator in the measured subset

Train token-bucket rule:
- canonical token-bucket probe:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_train_token_bucket_probe.py`
- canonical artifacts:
  - `/home/chen/RLPFN/artifacts/phase3_train_token_bucket_probe.json`
  - `/home/chen/RLPFN/artifacts/phase3_train_token_bucket_summary.json`
- duplicate-run protection:
  - output reuse is default
  - use `--overwrite` only if the source contract changed
- locked interpretation:
  - the canonical structured split is `Q1/Q2/Q3/Q4` over objective-token relative position
  - current locked reading:
    - `Q1/Q2` positive mass is anti-correlated with positive realized delta in the measured subset
    - `Q3/Q4` positive mass is positively correlated with positive realized delta
    - successful low-mass envs shift positive mass away from `Q1/Q2` toward `Q4`
    - successful low-mass envs also have better `Q1` value-return correlation than the failed high-mass case
  - this probe supersedes the earlier coarse first16/last16 interpretation

Heldout token-bucket transfer rule:
- canonical transfer probe:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_heldout_token_bucket_probe.py`
- canonical artifact:
  - `/home/chen/RLPFN/artifacts/phase3_heldout_token_bucket_probe.json`
- duplicate-run protection:
  - output reuse is default
  - use `--overwrite` only if the source contract changed
- locked interpretation:
  - this probe exists only to answer one narrow transfer question:
    - whether the train-side `Q3/Q4` success signature appears on the heldout subset under the same lightweight contract
  - current locked reading:
    - the heldout subset does **not** reproduce the train-side `Q3/Q4` success pattern
    - `Q2` positive mass aligns positively with heldout delta in the measured subset, while `Q4` aligns negatively
    - therefore do not infer heldout improvement potential from the train token-bucket result alone
    - the current transfer failure reading is:
      - train-side late-bucket success exists
      - heldout-side transfer of that late-bucket success mode does not
  - this remains subset-only diagnostic evidence and does **not** promote Phase 3 to trusted status

Train-vs-heldout transfer bottleneck rule:
- canonical synthesis probe:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_transfer_bottleneck_probe.py`
- canonical artifact:
  - `/home/chen/RLPFN/artifacts/phase3_transfer_bottleneck_probe.json`
- duplicate-run protection:
  - output reuse is default
  - use `--overwrite` only if the source canonical artifacts changed
- locked interpretation:
  - this probe is the canonical summary of the current many-env transfer failure mode
  - current locked reading:
    - the train-side and heldout-side bucket patterns do not merely differ in strength
    - they flip sign on all four bucket alignment channels (`Q1/Q2/Q3/Q4`)
    - train-side positive cases are `Q3/Q4`-dominated and higher-pre-gap
    - heldout-side positive cases are `Q1/Q2`-dominated and lower-pre-gap
  - therefore the current Phase 3 bottleneck should be read as:
    - transfer mode inversion, not simple attenuation of the train success signature
  - this remains diagnostic-only and does **not** promote Phase 3 to trusted status

Heldout rollout-quality rule:
- canonical probe:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_heldout_rollout_quality_probe.py`
- canonical artifact:
  - `/home/chen/RLPFN/artifacts/phase3_heldout_rollout_quality_probe.json`
- duplicate-run protection:
  - output reuse is default
  - use `--overwrite` only if the source contract changed
- locked interpretation:
  - this probe exists to answer whether the heldout suite lacks usable signal or whether the current update is selecting the wrong heldout environments/tokens
  - current locked reading:
    - usable heldout signal exists; the strongest heldout pre-gap envs are real
    - positive mass on heldout is coupled much more strongly to terminal resets than to pre-gap or critic alignment
    - the cleanest highest-pre-gap heldout envs are nearly starved of positive mass
  - therefore the heldout bottleneck should not be described as “the task is just hard”; it is a selection problem in the current many-env update

Phase 3 root-cause rule:
- canonical synthesis probe:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_root_cause_probe.py`
- canonical artifact:
  - `/home/chen/RLPFN/artifacts/phase3_root_cause_probe.json`
- duplicate-run protection:
  - output reuse is default
  - use `--overwrite` only if one of the source canonical artifacts changed
- locked interpretation:
  - current evidence supports:
    - optimization selection bias
    - transfer mode inversion
  - current evidence does **not** support:
    - an active collect/layout bug
    - a pure task-unlearnable explanation
    - a “normalization alone caused it” explanation
  - current locked reading:
    - the heldout reset-heavy bias is already present in raw positive actor-advantage mass
    - therefore advantage normalization may preserve the bias, but is not the primary source of it
  - this is the current deepest diagnostic root-cause summary and remains diagnostic-only

Heldout reset-semantics rule:
- canonical probe:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_heldout_reset_semantics_probe.py`
- canonical artifact:
  - `/home/chen/RLPFN/artifacts/phase3_heldout_reset_semantics_probe.json`
- duplicate-run protection:
  - output reuse is default
  - use `--overwrite` only if the heldout lightweight source artifacts changed
- locked interpretation:
  - the heldout reset-heavy bias is **not** primarily concentrated in:
    - `post_first_reset`
    - `terminal_tail`
    - `post_reset_terminal_tail`
  - the strongest current candidate sink is:
    - reset-conditioned `post_reset_non_tail`
  - raw residual positive mass is even more non-tail-heavy than actor-adv positive mass
  - therefore the best current reading is:
    - not a pure terminal-tail semantics bug
    - not a pure post-first-reset dominance bug
    - instead: reset-conditioned non-tail objective weighting / baseline interaction
  - diagnostic-only; this does **not** establish a new trusted Phase 3 baseline

Heldout layered mass decomposition rule:
- canonical probe:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_heldout_layered_mass_probe.py`
- canonical artifact:
  - `/home/chen/RLPFN/artifacts/phase3_heldout_layered_mass_probe.json`
- duplicate-run protection:
  - output reuse is default
  - use `--overwrite` only if the heldout lightweight source artifacts changed
- locked interpretation:
  - earliest supported reset-bias layer:
    - `raw_actor_adv_positive_mass`
  - current evidence does **not** support:
    - raw return target shape as the primary source
    - raw residual / critic-baseline subtraction as the primary source
    - normalization as the primary source
  - current best reading is:
    - reset-conditioned non-tail objective weighting / actor-advantage construction bias
  - diagnostic-only; this does **not** establish a new trusted Phase 3 baseline

Heldout layered mass decomposition rule:
- canonical probe:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_heldout_layered_mass_probe.py`
- canonical artifact:
  - `/home/chen/RLPFN/artifacts/phase3_heldout_layered_mass_probe.json`
- duplicate-run protection:
  - output reuse is default
  - use `--overwrite` only if the heldout lightweight source artifacts changed
- locked interpretation:
  - `earliest_reset_bias_layer = raw_actor_adv_positive_mass`
  - therefore:
    - return target shape bias is **not** the primary driver
    - critic baseline bias is **not** the primary driver
    - normalization is **not** the primary driver
  - current best root-cause classification:
    - objective weighting / mask semantics bias inside raw actor-advantage construction
  - diagnostic-only; this does **not** establish a new trusted Phase 3 baseline

Heldout segment ablation rule:
- canonical probe:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_heldout_segment_ablation_probe.py`
- canonical artifact:
  - `/home/chen/RLPFN/artifacts/phase3_heldout_segment_ablation_probe.json`
- duplicate-run protection:
  - output reuse is default
  - use `--overwrite` only if the heldout lightweight source artifacts changed
- locked interpretation:
  - base reset-correlation is high for raw actor-adv positive mass
  - removing `terminal_tail` produces the largest drop in reset-correlation
  - removing `post_reset_terminal_tail` produces the second-largest drop
  - removing `post_reset_non_tail` is smaller
  - removing `non_tail` or `first_episode_non_tail` worsens reset-correlation
  - conclusion:
    - terminal-tail mask semantics are the dominant bias driver
    - this is a bias source, not a direct fix (pre-gap capture does not improve)
  - diagnostic-only; this does **not** establish a new trusted Phase 3 baseline

Terminal-tail mask decomposition rule:
- canonical probe:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_terminal_tail_mask_probe.py`
- canonical artifact:
  - `/home/chen/RLPFN/artifacts/phase3_terminal_tail_mask_probe.json`
- duplicate-run protection:
  - output reuse is default
  - use `--overwrite` only if the heldout lightweight source artifacts changed
- locked interpretation:
  - removing terminal-tail tokens collapses reset-correlation as tail length increases
  - terminal-tail semantics are the dominant bias driver
  - first-terminal tail is slightly more biased than post-reset tail
  - diagnostic-only; this does **not** establish a new trusted Phase 3 baseline

Tail-bias propagation rule:
- canonical probe:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_tail_bias_propagation_probe.py`
- canonical artifact:
  - `/home/chen/RLPFN/artifacts/phase3_tail_bias_propagation_probe.json`
- duplicate-run protection:
  - output reuse is default
  - use `--overwrite` only if the heldout lightweight source artifacts changed
- locked interpretation:
  - raw return tail bias exists before GAE
  - GAE reduces but does not remove tail bias
  - therefore bias is not created by GAE propagation
  - root remains terminal-tail objective mask semantics
  - diagnostic-only; this does **not** establish a new trusted Phase 3 baseline

Terminal-tail rule decomposition rule:
- canonical probe:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_terminal_tail_mask_probe.py`
- canonical artifact:
  - `/home/chen/RLPFN/artifacts/phase3_terminal_tail_mask_probe.json`
- duplicate-run protection:
  - output reuse is default
  - use `--overwrite` only if the heldout lightweight source artifacts changed
- locked interpretation:
  - `first_episode_only` is the strongest bias driver:
    - highest `corr_vs_terminal_resets`
    - most negative `corr_vs_pre_suffix_gap`
    - near-zero `top2_pre_gap_capture`
  - `post_reset_only` is comparatively benign and correlates positively with pre-gap
  - terminal-tail is still a bias driver, but second-order to first-episode-only
  - diagnostic-only; this does **not** establish a new trusted Phase 3 baseline

First-episode-only ablation verification rule:
- canonical probe:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_first_episode_ablation_verify.py`
- canonical artifact:
  - `/home/chen/RLPFN/artifacts/phase3_first_episode_ablation_verify.json`
- duplicate-run protection:
  - output reuse is default
  - no new rollout; reads terminal-tail probe output
- locked interpretation:
  - removing `first_episode_only` does **not** reduce reset-bias
  - bias persists (and increases) in the remaining post-reset-only mass
  - therefore `first_episode_only` is not a unique necessary condition
  - diagnostic-only; this does **not** establish a new trusted Phase 3 baseline

Post-reset-only ablation verification rule:
- canonical probe:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_post_reset_ablation_verify.py`
- canonical artifact:
  - `/home/chen/RLPFN/artifacts/phase3_post_reset_ablation_verify.json`
- duplicate-run protection:
  - output reuse is default
  - no new rollout; reads terminal-tail probe output
- locked interpretation:
  - removing `post_reset_only` **reduces** reset-bias
  - removing `post_reset_only` **improves** pre-gap alignment
  - therefore `post_reset_only` is **not** a necessary condition for the bias
  - diagnostic-only; this does **not** establish a new trusted Phase 3 baseline

Terminal-tail interaction probe rule:
- canonical probe:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_terminal_tail_interaction_probe.py`
- canonical artifact:
  - `/home/chen/RLPFN/artifacts/phase3_terminal_tail_interaction_probe.json`
- duplicate-run protection:
  - output reuse is default
  - no new rollout; reads terminal-tail probe output
- locked interpretation:
  - strongest reset correlation sits in `post_reset_tail_mass`
  - `terminal_tail_mass` remains strongly reset-correlated overall
  - the interaction term `post_reset_only ∧ terminal_tail` is the sharpest bias driver
  - diagnostic-only; this does **not** establish a new trusted Phase 3 baseline

Post-reset terminal-tail ablation verification rule:
- canonical probe:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_post_reset_tail_ablation_verify.py`
- canonical artifact:
  - `/home/chen/RLPFN/artifacts/phase3_post_reset_tail_ablation_verify.json`
- duplicate-run protection:
  - output reuse is default
  - no new rollout; reads terminal-tail probe output
- locked interpretation:
  - removing `post_reset_terminal_tail` reduces reset-bias
  - removing `post_reset_terminal_tail` improves pre-gap alignment
  - interaction term is a strong contributor but not the only bias source
  - diagnostic-only; this does **not** establish a new trusted Phase 3 baseline

First-episode tail vs non-tail ablation verification rule:
- canonical probe:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_first_episode_tail_ablation_verify.py`
- canonical artifact:
  - `/home/chen/RLPFN/artifacts/phase3_first_episode_tail_ablation_verify.json`
- duplicate-run protection:
  - output reuse is default
  - no new rollout; reads terminal-tail probe output
- locked interpretation:
  - removing `first_episode_tail` reduces reset-bias but worsens pre-gap alignment
  - removing `first_episode_non_tail` worsens both reset-bias and pre-gap alignment
  - neither is a minimal necessary bias source
  - diagnostic-only; this does **not** establish a new trusted Phase 3 baseline

Post-reset terminal-tail band probe rule:
- canonical probe:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_post_reset_tail_band_probe.py`
- canonical artifact:
  - `/home/chen/RLPFN/artifacts/phase3_post_reset_tail_band_probe.json`
- duplicate-run protection:
  - output reuse is default
  - no new rollout; reads terminal-tail probe output
- locked interpretation:
  - all tail bands remain strongly reset-correlated
  - last-token band is marginally strongest but not uniquely
  - no single tail_len cutoff isolates all bias
  - diagnostic-only; this does **not** establish a new trusted Phase 3 baseline

Post-reset tail episode/position probe rule:
- canonical probe:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_post_reset_tail_episode_probe.py`
- canonical artifact:
  - `/home/chen/RLPFN/artifacts/phase3_post_reset_tail_episode_probe.json`
- note:
  - this probe collects a fresh rollout on the heldout suite
  - tail_len fixed at 8
- locked interpretation:
  - episode-2 and episode-3+ tails are both strongly reset-biased
  - early vs late reset positions do not separate the bias
  - no narrower necessary condition found in this split
  - diagnostic-only; this does **not** establish a new trusted Phase 3 baseline

Post-reset tail distance/step probe rule:
- canonical probe:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_post_reset_tail_distance_probe.py`
- canonical artifact:
  - `/home/chen/RLPFN/artifacts/phase3_post_reset_tail_distance_probe.json`
- note:
  - this probe collects a fresh rollout on the heldout suite
  - tail_len fixed at 8
- locked interpretation:
  - distance 1-4 tail tokens carry the highest reset-bias
  - bias is dominated by post-reset late steps (beyond first k)
  - early post-reset steps carry less mass and weaker bias
  - diagnostic-only; this does **not** establish a new trusted Phase 3 baseline

Post-reset tail dist≤4 & late-step ablation rule:
- canonical probe:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_post_reset_tail_dist4_late_ablation_verify.py`
- canonical artifact:
  - `/home/chen/RLPFN/artifacts/phase3_post_reset_tail_dist4_late_ablation_verify.json`
- note:
  - this probe collects a fresh rollout on the heldout suite
  - tail_len fixed at 8
- locked interpretation:
  - ablation reduces reset-bias and improves pre-gap alignment
  - effect is material but not full removal
  - diagnostic-only; this does **not** establish a new trusted Phase 3 baseline

Post-reset tail dist≤2 & late-step ablation rule:
- canonical probe:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_post_reset_tail_dist2_late_ablation_verify.py`
- canonical artifact:
  - `/home/chen/RLPFN/artifacts/phase3_post_reset_tail_dist2_late_ablation_verify.json`
- note:
  - this probe collects a fresh rollout on the heldout suite
  - tail_len fixed at 8
- locked interpretation:
  - dist≤2 late-step removal produces negligible change
  - minimal necessary subset is larger than dist≤2
  - diagnostic-only; this does **not** establish a new trusted Phase 3 baseline

Shortrun many-env sequence-layout source rule:
- canonical source probe:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_shortrun_sequence_layout_source_probe.py`
- canonical artifact:
  - `/home/chen/RLPFN/artifacts/phase3_shortrun_sequence_layout_source_probe.json`
- duplicate-run protection:
  - output reuse is default
  - use `--overwrite` only if the source contract changed
- locked interpretation:
  - this probe inherits the current strict numeric contract:
    - fixed pair1 train suite
    - `strict_fixed_env_mode = true`
    - `deterministic_actor_sampling = true`
    - `deterministic_batch_plan = true`
    - `restore_validation_policy_state = true`
  - current locked reading:
    - repeated many-env shortrun `collect_rollouts()` is stable at all checked layers:
      - step payload
      - rollout buffer
      - flat-batch sequence layout
    - `episode_starts` correctly follows previous-step `dones`
    - masked action tail is identically zero in the measured contract
    - therefore the old quarantined `phase3_shortrun_sequence_layout_instability_probe.json` must not be used to describe the current many-env fixed-suite collect path
  - this removes one suspected source of ambiguity, but it does **not** promote Phase 3 to trusted status

Pair1 suite-matched pre-gap repair rule:
- quarantine these historical-only artifacts for cross-suite/regime reasoning:
  - `/home/chen/RLPFN/artifacts/phase3_post_reset_tail_distance_probe.json`
  - `/home/chen/RLPFN/artifacts/phase3_terminal_tail_mask_probe.json`
  - `/home/chen/RLPFN/artifacts/phase3_post_reset_tail_ablation_verify.json`
  - `/home/chen/RLPFN/artifacts/phase3_post_reset_ablation_verify.json`
- cause:
  - pair1 `pre_vs_zero_suffix_gap` was partially reused from `/home/chen/RLPFN/artifacts/phase3_env_delta_concentration_probe_max4.json`
  - envs outside that partial subset were silently written as `0.0`
- canonical repaired pair1 artifacts:
  - `/home/chen/RLPFN/artifacts/phase3_post_reset_tail_distance_probe_pair1_suite_matched.json`
  - `/home/chen/RLPFN/artifacts/phase3_terminal_tail_mask_probe_pair1_suite_matched.json`
  - `/home/chen/RLPFN/artifacts/phase3_post_reset_tail_ablation_verify_pair1_suite_matched.json`
  - `/home/chen/RLPFN/artifacts/phase3_post_reset_ablation_verify_pair1_suite_matched.json`
- locked interpretation after repair:
  - pair1 `dist1_to_4_mean_corr_pre = +0.4612`
  - pair1 removing `post_reset_terminal_tail` reduces reset-bias but worsens pre-gap alignment:
    - `corr_pre_delta = -0.2590`
  - pair1 removing `post_reset_only` reduces reset-bias but worsens pre-gap alignment:
    - `corr_pre_delta = -0.2647`
  - therefore old pair1 claims that these ablations improve pre-gap are invalid
- code guard:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_subset_delta_fallback.py`
  - partial `reuse_heldout_subset_delta_json` coverage now raises instead of silently zero-filling missing envs

Regime anchor probe rule:
- canonical probe:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_regime_anchor_env_probe.py`
- canonical artifact:
  - `/home/chen/RLPFN/artifacts/phase3_regime_anchor_env_probe.json`
- duplicate-run protection:
  - output reuse is default
  - reads existing suite-matched baseline and distance artifacts only
- locked interpretation:
  - pair1 heavy mass is concentrated in a sparse reset-active subset:
    - top3 mass share within pair1 = `0.9567`
    - nonzero-mass env count = `6/16`
  - pair1 top-mass envs are not all uniformly good:
    - env10 / env12 are real high-gain anchors
    - env5 is high-mass but negative-gap
  - pair1 gain is not exhaustively captured by `dist1_to_4_mass`:
    - env11 is zero-mass but large positive-gap
  - pair2 env12 is unlikely to be a compare-contract or nonfinite metric glitch:
    - zero/PPO contract fields match
    - nonfinite counts are all zero
    - suffix/full gaps are both strongly positive
  - pair2 env12 is a proxy outlier:
    - pre-gap rank `1`
    - mass rank `3`
    - mass share within pair2 `0.0117`

Positive-regime captured-vs-missed anchor structure rule:
- canonical probe:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_positive_regime_anchor_structure_probe.py`
- canonical artifact:
  - `/home/chen/RLPFN/artifacts/phase3_positive_regime_anchor_structure_probe.json`
- duplicate-run protection:
  - output reuse is default
  - reads canonical positive-suite baseline, distance, and tail-mask artifacts only
- locked contract:
  - strong gain anchor threshold:
    - `pre_vs_zero_suffix_gap > 1.0`
  - captured anchor threshold:
    - `dist_mass_share_within_suite >= 0.05`
- locked interpretation:
  - captured strong gain anchors are sparse and currently only appear in pair1
  - captured anchors are more reset-heavy and more post-reset-tail weighted:
    - mean reset count `1.5`
    - mean `post_reset_terminal_tail_removed_share = 0.4638`
  - missed strong gain anchors are still real positive-gain envs:
    - mean suffix gap `4.8114`
    - mean full gap `85.1207`
  - missed anchors are strongly first-episode skewed and weakly represented by the reset-tail proxy:
    - mean `dist_mass_share_within_suite = 0.0015`
    - mean `first_episode_only_removed_share = 0.7468`
    - mean `post_reset_terminal_tail_removed_share = 0.0281`
  - locked design implication:
    - current reset-tail proxy underweights real gain anchors
    - any future objective/weighting fix must preserve reset-tail gain anchors while adding coverage for first-episode / non-reset positive-gain anchors

Dual-channel anchor coverage rule:
- canonical probe:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_dual_channel_anchor_coverage_probe.py`
- canonical artifact:
  - `/home/chen/RLPFN/artifacts/phase3_dual_channel_anchor_coverage_probe.json`
- duplicate-run protection:
  - output reuse is default
  - reads canonical positive-suite distance/tail-mask artifacts only
- locked contract:
  - reset-tail channel:
    - `dist1_to_4_mass`
  - added coverage channel:
    - `first_episode_non_tail_mass = first_episode_only_mass - first_terminal_tail_mass`
  - strong gain anchor:
    - `pre_vs_zero_suffix_gap > 1.0`
  - capture threshold:
    - per-suite channel share `>= 0.05`
- locked interpretation:
  - dual channel improves strong-gain anchor coverage from `2/12` to `6/12`
  - recovered anchors are mainly non-reset / first-episode dominated anchors
  - residual missed anchors remain `6/12`
  - residual missed anchors have much weaker signal than recovered anchors:
    - mean base positive mass `0.3949` vs recovered `14.8114`
    - mean first-episode-non-tail share `0.0080` vs recovered `0.1546`
  - locked design implication:
    - adding a first-episode / non-reset coverage channel is directionally correct but not sufficient
    - the next bottleneck after coverage is positive actor-adv mass formation itself on some real-gain heldout anchors

Residual missed-anchor sign-formation rule:
- canonical probe:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_residual_missed_anchor_sign_probe.py`
- canonical artifact:
  - `/home/chen/RLPFN/artifacts/phase3_residual_missed_anchor_sign_probe.json`
- duplicate-run protection:
  - output reuse is default
  - default anchor source is explicitly:
    - `dual_channel_residual_missed`
  - do not reinterpret the default output as the broader positive-regime `proxy-missed` set
- locked contract:
  - residual missed set is inherited from:
    - `/home/chen/RLPFN/artifacts/phase3_dual_channel_anchor_coverage_probe.json`
  - fixed-size residual set:
    - `6` anchors
  - rollout contract:
    - Phase 2 green
    - `trusted_sep_reset_mainline`
    - `n_samples = 2048`
    - `single_eval_pos = 1946`
- locked interpretation:
  - residual missed anchors are mostly wrong-sign at final normalized actor-adv:
    - `5 / 6`
  - final within-suite positive mass is strongly underweighted:
    - mean raw return share `0.06275`
    - mean raw actor-adv share `0.01778`
    - mean normalized actor-adv share `0.01762`
  - positive mass is not mainly lost at return -> residual:
    - mean flip share `0.0030`
  - positive mass is mainly lost at residual -> actor:
    - mean flip share `0.7663`
    - dominant share-drop stage is usually `actor_adv`
    - dominant flip stage is usually `actor_adv`
  - normalization is not the primary explanation overall:
    - mean actor -> normalization flip share `0.1783`
    - only one residual missed anchor is a clear normalization outlier
  - locked design implication:
    - do not spend the next cycle re-tuning semantic coverage alone
    - do not spend the next cycle blaming normalization alone
    - the next root-cause probe should target the residual -> actor transition directly

Residual-anchor GAE path rule:
- canonical probe:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_residual_anchor_gae_path_probe.py`
- canonical artifact:
  - `/home/chen/RLPFN/artifacts/phase3_residual_anchor_gae_path_probe.json`
- duplicate-run protection:
  - output reuse is default
  - default target set is inherited from:
    - `/home/chen/RLPFN/artifacts/phase3_residual_missed_anchor_sign_probe.json`
  - expected target contract:
    - `anchor_source = dual_channel_residual_missed`
    - `anchor_count = 6`
- locked math guard:
  - actor-adv recurrence reconstruction must remain exact:
    - `max_actor_reconstruction_error <= 1e-5`
  - delta decomposition must remain exact:
    - `max_delta_reconstruction_error <= 1e-5`
  - if either guard fails, the artifact is not trustworthy
- locked interpretation:
  - the residual-missed set is not homogeneous
  - pair2 anchors are true sign-formation failures inside bootstrap / GAE:
    - positive raw residual tokens exist
    - many of them flip to wrong-sign actor-adv
  - pair1 env11 / env14 are upstream failures:
    - positive raw rollout residual token count is `0`
    - they must not be explained as “positive residual got flipped by GAE”
  - across all flip tokens:
    - `bootstrap_term_negative = 57`
    - `delta_negative_mixed = 12`
    - `negative_gae_future_carry = 52`
  - aggregate counts are slightly bootstrap-heavy, but env-level reading matters:
    - pair2 env12 / env1 lean future-carry
    - pair2 env7 / env4 lean bootstrap-negative
  - non-objective gap is not the dominant hidden bug in this contract:
    - mean `next_step_nonobjective_flip_share = 0.0273`
- locked design implication:
  - future probes must split pair1-like upstream failures from pair2-like GAE sign-flip failures
  - the next narrow probe should stay inside pair2 positive-residual flip tokens and compare:
    - bootstrap-dominant flips
    - future-carry-dominant flips

Pair2 flip-mode split rule:
- canonical probe:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_pair2_flip_mode_split_probe.py`
- canonical artifact:
  - `/home/chen/RLPFN/artifacts/phase3_pair2_flip_mode_split_probe.json`
- duplicate-run protection:
  - output reuse is default
  - reads only:
    - `/home/chen/RLPFN/artifacts/phase3_residual_anchor_gae_path_probe.json`
- locked contract:
  - target scope is fixed:
    - `suite_name = pair2`
    - `positive_residual_wrong_sign_actor = true`
  - split buckets are fixed:
    - `future_carry_dominant`:
      - `flip_cause = negative_gae_future_carry`
    - `bootstrap_dominant`:
      - `flip_cause in {bootstrap_term_negative, delta_negative_mixed}`
- locked interpretation:
  - both flip modes must remain present:
    - bootstrap `69`
    - future-carry `52`
  - bootstrap leads slightly by count, but future-carry carries stronger anchors:
    - future mean `pre_vs_zero_suffix_gap = 10.7195`
    - bootstrap mean `8.0105`
  - future-carry tokens should preserve positive local delta on average:
    - mean `delta_norm = +0.03037`
  - bootstrap tokens should remain locally negative on average:
    - mean `delta_norm = -0.05774`
  - future-carry tokens should carry the stronger recursive negative signal:
    - mean `gae_future_carry_norm = -0.21387`
    - bootstrap mean `-0.12538`
  - final wrong-sign actor-adv magnitude should stay similar across both modes:
    - future mean `actor_adv_norm = -0.18349`
    - bootstrap mean `-0.18312`
  - temporal position alone is not a trustworthy separator:
    - do not reduce this result to a simple early-token vs tail-token story
- locked design implication:
  - future work on pair2 bottlenecks must keep bootstrap and future-carry as separate regimes
  - the most valuable next probe is the future-carry source chain on `env12`-like anchors, not another pooled count summary

Future-carry source-chain rule:
- canonical probe:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_future_carry_source_chain_probe.py`
- canonical artifact:
  - `/home/chen/RLPFN/artifacts/phase3_future_carry_source_chain_probe_pair2_env12.json`
- duplicate-run protection:
  - output reuse is default
  - reads only:
    - `/home/chen/RLPFN/artifacts/phase3_residual_anchor_gae_path_probe.json`
    - `/home/chen/RLPFN/artifacts/phase3_pair2_flip_mode_split_probe.json`
- locked contract:
  - target scope is fixed:
    - `suite_name = pair2`
    - `env_index = 12`
    - `flip_mode = future_carry_dominant`
  - negative source split is fixed:
    - `true_negative_future_residual`
    - `terminal_reset_tail_semantic_mismatch`
    - `baseline_bootstrap_semantic_mismatch`
  - classification guard:
    - do not use `tokens_to_objective_episode_end == 0` alone to label terminal tail
    - terminal-tail source requires:
      - `next_non_terminal = 0`
    - nonterminal rollout-horizon endpoints stay in:
      - `baseline_bootstrap_semantic_mismatch`
- locked guardrails:
  - objective episode chains must remain contiguous
  - inferred recursion factor must stay stable near `0.95`
  - weighted downstream `delta` sum must reconstruct `gae_future_carry_norm`
    - max reconstruction error `<= 1e-6`
- locked interpretation:
  - `pair2 env12` future-carry starts count:
    - `23`
  - true negative future residual mass must remain zero:
    - `0.0`
  - negative future carry is fully explained by semantic/bootstrap mismatch:
    - terminal-reset tail share `0.7417`
    - baseline/bootstrap share `0.2583`
  - dominant source by start token:
    - terminal-reset tail `18`
    - baseline/bootstrap `5`
  - dominant source token must remain:
    - episode0 local `56`
    - aggregated negative contribution `11.6735`
    - raw residual still positive `6.3613`
  - episode1 endpoint must not be mislabeled:
    - local `101` is a nonterminal rollout-horizon bootstrap source, not terminal-reset tail
- locked design implication:
  - do not explain the env12 future-carry failure as genuine bad downstream future reward
  - the next narrow probe should target why the downstream local deltas themselves become negative under the current bootstrap / tail contract

Env12 negative-delta rule:
- canonical probe:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_env12_negative_delta_probe.py`
- canonical artifact:
  - `/home/chen/RLPFN/artifacts/phase3_env12_negative_delta_probe.json`
- duplicate-run protection:
  - output reuse is default
  - reads only:
    - `/home/chen/RLPFN/artifacts/phase3_residual_anchor_gae_path_probe.json`
    - `/home/chen/RLPFN/artifacts/phase3_future_carry_source_chain_probe_pair2_env12.json`
- locked contract:
  - target scope is fixed:
    - `suite_name = pair2`
    - `env_index = 12`
    - locals `56, 29, 38, 51`
  - normalized local TD split is fixed:
    - `reward_norm`
    - `boundary_shift_norm`
    - `bootstrap_term_norm`
  - raw-space fields are auxiliary only:
    - do not base the main interpretation on `raw_value_raw`, `bootstrap_term_raw`, or `delta_raw`
- locked interpretation:
  - local `56` must remain:
    - `terminal_reset_boundary_shift_dominant`
    - with boundary shift `-1.1600` larger than bootstrap magnitude `-0.15949`
  - locals `29 / 38 / 51` must all remain bootstrap-dominant
  - locals `38 / 51` must remain positive-reward but negative-delta tokens
  - cluster means must keep the same sign pattern:
    - mean reward term positive
    - mean bootstrap term negative and larger in magnitude
    - mean delta negative
  - current artifact also contains a raw-value contract anomaly on these focus tokens:
    - max raw-value contract error `20.3240`
    - do not let later analyses silently reinterpret this as trustworthy raw-space evidence
- locked design implication:
  - split the next debugging step by token regime:
    - local `56`:
      - terminal-reset boundary normalization
    - locals `29 / 38 / 51`:
      - nonterminal bootstrap stepdown / value-drop

Env12 boundary-stepdown root rule:
- canonical probe:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_env12_boundary_stepdown_root_probe.py`
- canonical artifact:
  - `/home/chen/RLPFN/artifacts/phase3_env12_boundary_stepdown_root_probe.json`
- duplicate-run protection:
  - output reuse is default
  - reads only:
    - `/home/chen/RLPFN/artifacts/phase3_env12_negative_delta_probe.json`
    - `/home/chen/RLPFN/artifacts/phase3_residual_anchor_gae_path_probe.json`
- locked contract:
  - target scope is fixed:
    - `suite_name = pair2`
    - `env_index = 12`
    - local `56` plus local `101` nonterminal contrast
    - locals `29 / 38 / 51` cluster
  - classification is fixed:
    - local `56` is a boundary-normalization question
    - locals `29 / 38 / 51` are local value-peak stepdown questions
  - do not rerun rollout collection to answer this question
    - this probe is artifact-only by design
- locked interpretation:
  - local `56` must remain terminal-boundary dominated:
    - boundary vs immediate reward-over-std ratio `41.6161x`
    - boundary share of `reward_norm` magnitude `0.9765`
    - boundary share of `delta_norm` magnitude `0.8609`
  - local `101` must remain the nonterminal contrast with:
    - `boundary_shift_norm = 0.0`
  - locals `29 / 38 / 51` must remain:
    - all local value peaks
    - with mean next-step drop larger than mean reward term
  - pattern split must remain heterogeneous:
    - oscillatory peaks `2`
    - late-preterminal peaks `1`
  - locals `38 / 51` must remain positive-reward but negative-delta tokens
- locked design implication:
  - do not collapse local `56` into a generic tail-token story
  - do not collapse locals `29 / 38 / 51` into one generic bootstrap bucket
  - subsequent fixes or probes should preserve the distinction between:
    - terminal boundary-shift semantics
    - oscillatory local over-peak stepdown
    - late-preterminal peak stepdown

Env12 local56 boundary-contract rule:
- canonical probe:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_env12_local56_boundary_contract_probe.py`
- canonical artifact:
  - `/home/chen/RLPFN/artifacts/phase3_env12_local56_boundary_contract_probe.json`
- duplicate-run protection:
  - output reuse is default
  - reads only:
    - `/home/chen/RLPFN/artifacts/phase3_env12_negative_delta_probe.json`
    - `/home/chen/RLPFN/artifacts/phase3_residual_anchor_gae_path_probe.json`
- locked contract:
  - target scope is fixed:
    - `suite_name = pair2`
    - `env_index = 12`
    - local `56`
    - nonterminal contrasts `55 / 57 / 101`
  - use full-rollout boundary stats only:
    - `mean_raw = std_raw * mean_over_std`
    - boundary shift at terminal is evaluated against full-rollout `mean/std`
  - do not estimate the terminal boundary constant from objective-only token rows
- locked interpretation:
  - full boundary contract must remain:
    - `mean/std = 1.1600`
    - `mean_raw = 16.3398`
    - `std_raw = 14.0860`
    - nonterminal rows imply `gamma = 1.0`
  - local `56` boundary shift must remain:
    - `-1.1600`
  - removing only the boundary term must recover most of the local negative delta:
    - delta from `-1.34736` to `-0.18736`
  - objective-only stats must remain different from the true boundary constant:
    - objective-only `mean/std = 1.6895`
    - do not treat that as the PPO boundary contract
- locked design implication:
  - do not explain `local56` as a generic bad reward token
  - do not use objective-subset return stats to debug terminal boundary normalization

Env12 value peak-drop contract rule:
- canonical probe:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_env12_value_peak_drop_contract_probe.py`
- canonical artifact:
  - `/home/chen/RLPFN/artifacts/phase3_env12_value_peak_drop_contract_probe.json`
- duplicate-run protection:
  - output reuse is default
  - reads only:
    - `/home/chen/RLPFN/artifacts/phase3_env12_negative_delta_probe.json`
    - `/home/chen/RLPFN/artifacts/phase3_residual_anchor_gae_path_probe.json`
- locked contract:
  - target scope is fixed:
    - `suite_name = pair2`
    - `env_index = 12`
    - locals `29 / 38 / 51`
  - work only in normalized value space:
    - target step-change from normalized rollout returns
    - value stepdown from `value_norm - next_value_norm`
  - the following identities must hold:
    - reward equals target step-change under current `gamma = 1` nonterminal contract
    - delta equals reward minus value-stepdown
    - stepdown excess equals error-drop
- locked interpretation:
  - `local29` must remain:
    - `hallucinated_peak_against_target_trend`
  - `locals38 / 51` must remain:
    - `amplified_target_peak`
  - all three must keep positive next-step raw future return
  - failure is not “future gain absent”
  - failure is:
    - `value_stepdown > target_step_change`
- locked design implication:
  - do not collapse `29 / 38 / 51` into a generic future-carry or no-future-gain explanation
  - future debugging on this branch should target why value-error peaks at the current token and relaxes at the next token

Env12 local56 mean-source rule:
- canonical probe:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_env12_local56_mean_source_probe.py`
- canonical artifact:
  - `/home/chen/RLPFN/artifacts/phase3_env12_local56_mean_source_probe.json`
- duplicate-run protection:
  - output reuse is default
  - reads only:
    - `/home/chen/RLPFN/artifacts/phase3_env12_local56_boundary_contract_probe.json`
    - `/home/chen/RLPFN/artifacts/phase3_residual_anchor_gae_path_probe.json`
- locked contract:
  - target scope is fixed:
    - `suite_name = pair2`
    - `env_index = 12`
  - this is a weighted-average decomposition probe:
    - no rollout replay
    - no direct terminal-bonus trace
  - use the full-rollout boundary contract as the source of truth for:
    - `mean_raw`
    - `std_raw`
    - `mean/std`
  - use the residual-anchor token rows only for:
    - objective-tail timing
    - objective episode means
    - local56 position readout
- locked interpretation:
  - the objective window is only the last `102 / 2048` tokens:
    - `4.98%` of the rollout
  - the omitted prefix must remain the dominant mean source:
    - implied prefix mean raw `16.8468`
    - objective-only mean raw `6.6663`
  - post-reset objective episode must remain a mean-lowering segment:
    - episode1 mean `2.8182`
    - episode0 mean `9.7042`
  - local `56` itself must remain below both means:
    - return raw `5.6887`
    - reward raw negative
  - therefore the large terminal boundary constant is a full-rollout centering-scope effect, not a local positive-source effect
- locked anomaly handling:
  - do not use `/home/chen/RLPFN/artifacts/phase3_terminal_tail_mask_probe_pair2.json` in the local56 source chain until reconciled
  - reason:
    - env12 terminal-reset count there disagrees with the trusted local56 terminal evidence

Env12 local56 boundary-scope counterfactual rule:
- canonical probe:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_env12_local56_boundary_scope_counterfactual_probe.py`
- canonical artifact:
  - `/home/chen/RLPFN/artifacts/phase3_env12_local56_boundary_scope_counterfactual_probe.json`
- duplicate-run protection:
  - output reuse is default
  - reads only:
    - `/home/chen/RLPFN/artifacts/phase3_env12_local56_boundary_contract_probe.json`
    - `/home/chen/RLPFN/artifacts/phase3_residual_anchor_gae_path_probe.json`
- locked contract:
  - compare only read-only scope swaps for local56:
    - current full rollout
    - objective-all
    - objective-episode0-only
    - objective-episode1-only
  - this is a reward-space boundary counterfactual only
    - not a retrained value-head comparison
- locked interpretation:
  - current boundary shift must remain:
    - `-1.1600`
  - objective-all must worsen it:
    - `-1.68946`
    - `1.456x` current magnitude
  - objective-episode0-only must worsen it most:
    - `-4.55267`
    - `3.925x` current magnitude
  - objective-episode1-only must also worsen it:
    - `-1.61690`
    - `1.394x` current magnitude
  - therefore no tested objective-aligned centering scope shrinks the local56 terminal boundary penalty
- locked design implication:
  - do not promote “switch boundary normalization to objective-aligned scope” as a fix candidate on the current trusted branch
  - if we later want to change this contract, it must be justified by a different normalization design, not by this naive scope swap

Env12 local56 boundary mean-std-interaction rule:
- canonical probe:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_env12_local56_boundary_mean_std_interaction_probe.py`
- canonical artifact:
  - `/home/chen/RLPFN/artifacts/phase3_env12_local56_boundary_mean_std_interaction_probe.json`
- duplicate-run protection:
  - output reuse is default
  - reads only:
    - `/home/chen/RLPFN/artifacts/phase3_env12_local56_boundary_scope_counterfactual_probe.json`
- locked contract:
  - compare current full-rollout scope against:
    - objective-all
    - objective-episode0-only
    - objective-episode1-only
  - use exact additive decomposition:
    - `mean_only`
    - `std_only`
    - `interaction`
  - reconstruction error must stay numerically zero up to floating-point noise
- locked interpretation:
  - for every tested objective-aligned scope:
    - `mean_only < 0`
    - `std_only > 0`
    - `interaction < 0`
  - dominant worsening driver must remain:
    - `std_too_small`
  - do not describe the local56 scope-worsening as “mean too large”
  - do not describe interaction as the main culprit
- locked design implication:
  - if we ever redesign the boundary contract, the first quantity to reason about is scale collapse, not centering mean inflation

Env12 local56 scale-collapse rule:
- canonical probe:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_env12_local56_scale_collapse_probe.py`
- canonical artifact:
  - `/home/chen/RLPFN/artifacts/phase3_env12_local56_scale_collapse_probe.json`
- duplicate-run protection:
  - output reuse is default
  - reads only:
    - `/home/chen/RLPFN/artifacts/phase3_env12_local56_mean_source_probe.json`
    - `/home/chen/RLPFN/artifacts/phase3_env12_local56_boundary_scope_counterfactual_probe.json`
- locked contract:
  - target scope is fixed:
    - `suite_name = pair2`
    - `env_index = 12`
    - local `56`
  - decompose scale in two stages only:
    - full rollout:
      - omitted prefix vs objective tail
    - objective tail:
      - episode0 vs episode1
  - do not reintroduce non-trusted mask artifacts into this chain
- locked interpretation:
  - the omitted prefix must remain the dominant full-rollout variance source:
    - within-prefix share `97.1372%`
  - the objective tail must remain the scale-collapsed slice:
    - objective `std = 3.9458`
    - full `std = 14.0860`
    - ratio `0.2801x`
  - the omitted prefix must not be described as compressed:
    - prefix `std = 14.2421`
    - it is slightly larger than full-rollout `std`
  - within the objective tail, most remaining variance must remain:
    - between episode means
    - share `75.0837%`
  - both objective episodes must remain individually narrower than objective-all:
    - episode0 `std = 2.1315`
    - episode1 `std = 1.7430`
- locked design implication:
  - do not frame the local56 scale-collapse story as “objective-aligned mean is wrong”
  - do not frame it as “prefix returns are low-scale”
  - the current trusted read is:
    - scale collapses because objective-aligned scope discards the high-variance prefix and keeps a short low-variance tail
  - if boundary normalization is ever redesigned, reason about:
    - truncation-induced variance loss
    - episode segmentation inside the objective tail

Cross-suite terminal-local scale-chain replication rule:
- canonical probe:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_terminal_local_scale_chain_probe.py`
- covered by:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_phase3_terminal_local_scale_chain_probe.py`
- canonical replication artifact:
  - `/home/chen/RLPFN/artifacts/phase3_pair1_env12_terminal_scale_chain_probe.json`
- duplicate-run protection:
  - output reuse is default
  - rerun only with `--overwrite`
- locked contract:
  - current trusted replication point is fixed:
    - `suite_name = pair1`
    - `env_index = 12`
    - `n_samples = 2048`
    - `single_eval_pos = 1946`
  - infer full-rollout boundary contract from token rows only:
    - nonterminal `reward_raw / reward_norm` -> full `std`
    - terminal boundary shift -> full `mean/std`
  - do not replace that contract with direct env-level discounted-return moments
  - multi-terminal objective rows are allowed:
    - do not force a `selected_target_local_index` when the artifact reports multiple terminal locals
  - do not equate:
    - `objective_terminal_reset_count`
    - objective terminal token-row count
- locked interpretation:
  - historical pair1 env12 artifact is preserved, but it is no longer a trusted current replication point
  - reason:
    - `/home/chen/RLPFN/artifacts/phase3_pair1_env12_objective_reset_contract_probe.json`
    - fresh official collect now reports:
      - `objective_terminal_reset_count_from_segments = 0`
      - `terminal_row_count_within_objective = 0`
    - while the stored distance artifact still says:
      - `objective_terminal_reset_count = 1`
  - operational rule:
    - do not use the historical pair1 env12 scale-chain artifact in the current official cross-suite stable set
    - repaired boolean guardrail still applies in general:
      - do not require `prefix std > full std` to call the variance mechanism “prefix-dominated”
      - full variance can be slightly above prefix variance because of between-group mean structure

Seed13579 env12 objective-reset contract rule:
- canonical probe:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_objective_reset_contract_probe.py`
- covered by:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_phase3_objective_reset_contract_probe.py`
- canonical formal artifact:
  - `/home/chen/RLPFN/artifacts/phase3_seed13579_env12_objective_reset_contract_probe.json`
- repeated formal rerun artifact:
  - `/home/chen/RLPFN/artifacts/phase3_seed13579_env12_objective_reset_contract_probe_repeatB.json`
- preserved first conflicting fresh collect:
  - `/home/chen/RLPFN/artifacts/phase3_seed13579_env12_objective_reset_contract_debug_runA.json`
- locked reading:
  - pre-fix helper behavior was unstable on this suite/env:
    - `debug_runA` produced no objective reset
  - stored distance row:
    - `objective_terminal_reset_count = 1`
  - after wiring fresh collect to strict suite seeds, the formal probe produced:
    - `objective_terminal_reset_count_from_segments = 1`
    - `terminal_row_count_within_objective = 1`
  - repeated formal rerun produced the same structural fields exactly:
    - compressed segments `[[0, 91], [91, 102]]`
    - `episode_starts_on_objective_positions = [91]`
    - `terminal_rows_within_objective_global_steps = [2036]`
  - the post-fix reset-contract read is now replay-stable across two reruns
- operational rule:
  - keep `debug_runA` only as pre-fix bug evidence
  - use the post-fix formal artifacts as the current reset-contract baseline for `seed13579 env12`
  - do not yet use `seed13579 env12` to promote the cross-suite variance-collapse claim:
    - the scale-chain probe itself still needs to be rerun under the repaired collect contract

Seed13579 env12 repaired scale-chain rule:
- canonical artifact:
  - `/home/chen/RLPFN/artifacts/phase3_seed13579_env12_terminal_scale_chain_probe.json`
- repeated formal rerun artifact:
  - `/home/chen/RLPFN/artifacts/phase3_seed13579_env12_terminal_scale_chain_probe_repeatB.json`
- probe:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_terminal_local_scale_chain_probe.py`
- covered by:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_phase3_terminal_local_scale_chain_probe.py`
- locked reading:
  - this suite/env now supports the same high-level variance mechanism under the repaired collect contract:
    - `objective_scope_std_collapse_is_explained_by_truncating_away_high_variance_prefix = true`
    - `objective_window_is_low_variance_slice_not_high_variance_source = true`
    - within-prefix share `94.9054%`
    - objective/full std ratio `0.1040x`
    - objective/full variance ratio `0.01081x`
  - the repaired contract also fixes the boundary read:
    - `single_terminal_local = true`
    - `terminal_local_indices = [90]`
    - `all_terminal_candidates_match_full_rollout_centering_contract = true`
  - repeated rerun must agree exactly on the key semantic fields:
    - `selected_target_local_index = 90`
    - terminal global step `2036`
    - same boundary-contract core
    - same mean-source summary
    - same scale-collapse readout
    - same objective-variance decomposition
  - do not overgeneralize the objective-internal decomposition:
    - `dominant_objective_variance_source = within_episode0`
    - not `between_episode_means`
    - therefore the internal tail-variance driver is not a cross-suite invariant law
- protected interpretation:
  - cross-suite-stable:
    - truncation-induced variance loss
    - objective tail as the low-variance slice
  - cross-suite-unstable:
    - which objective sub-bucket dominates the residual variance after truncation
  - current official fresh-collect replication set:
    - `/home/chen/RLPFN/artifacts/phase3_seed13579_env12_terminal_scale_chain_probe.json`
    - `/home/chen/RLPFN/artifacts/phase3_seed13579_env12_terminal_scale_chain_probe_repeatB.json`

Pair1 env12 reset-contract mismatch rule:
- canonical artifact:
  - `/home/chen/RLPFN/artifacts/phase3_pair1_env12_objective_reset_contract_probe.json`
- replay-diff artifact:
  - `/home/chen/RLPFN/artifacts/phase3_pair1_env12_reset_contract_replay_diff_probe.json`
- locked reading:
  - stored pair1 distance row claims one objective reset
  - fresh official collect under the repaired suite-seed contract claims zero objective resets and zero in-objective terminal rows
  - replay-diff probe now explains the mismatch source:
    - `legacy_distance_probe_runA` reproduces stored reset count `1`
    - `legacy_distance_probe_runB` drifts to reset count `0`
    - `official_strict` deterministically replays suite env/rollout seeds and yields reset count `0`
  - therefore pair1 env12 is a legacy-vs-official collect-contract sentinel, not an unexplained fresh official bug
- operational rule:
  - quarantine `/home/chen/RLPFN/artifacts/phase3_pair1_env12_terminal_scale_chain_probe.json` from the official cross-suite stable set
  - do not use pair1 legacy distance artifacts together with current official strict fresh-collect conclusions:
    - legacy distance artifacts belong to an unseeded collect contract
    - current official fresh collect belongs to a strict suite-seeded contract
  - do not treat pair1 env12 as a current official fresh-collect replication point

Pair2 env12 reset-contract mismatch rule:
- canonical artifact:
  - `/home/chen/RLPFN/artifacts/phase3_pair2_env12_objective_reset_contract_probe.json`
- replay-diff artifact:
  - `/home/chen/RLPFN/artifacts/phase3_pair2_env12_reset_contract_replay_diff_probe.json`
- locked reading:
  - stored pair2 distance row claims one objective reset
  - fresh official collect under the repaired suite-seed contract claims:
    - `objective_terminal_reset_count_from_segments = 0`
    - `terminal_row_count_within_objective = 0`
  - the generic fresh-collect scale-chain rerun therefore fails before contract recovery:
    - `Could not infer full-rollout mean/std from terminal token rows.`
  - replay-diff probe now explains the mismatch source:
    - `legacy_distance_probe_runA` reproduces stored reset count `1`
    - `legacy_distance_probe_runB` drifts to reset count `2`
    - `official_strict` deterministically replays suite env/rollout seeds and yields reset count `0`
  - therefore pair2 env12 is a legacy-vs-official collect-contract sentinel, not an unexplained fresh official bug
- operational rule:
  - do not treat pair2 env12 as a current official fresh-collect replication point for the terminal-local scale chain
  - do not use the pair2 env12 local56 chain as proof of current official replay-stable replication:
    - it is now historical artifact-scoped evidence only
  - do not mix pair2 legacy distance artifacts with current official strict fresh-collect conclusions:
    - legacy distance artifacts belong to an unseeded collect contract
    - current official fresh collect belongs to a strict suite-seeded contract
  - pair2 env12 itself remains excluded from the current official saved/replay-stable set

Official-strict reset census rule:
- canonical probe:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_official_strict_reset_census.py`
- covered by:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_phase3_official_strict_reset_census.py`
- canonical artifact:
  - `/home/chen/RLPFN/artifacts/phase3_official_strict_reset_census_pair1_pair2.json`
- locked reading:
  - under current official strict collect, pair1/pair2 do contain scale-chain entry candidates:
    - `pair1 env10`
    - `pair1 env13`
    - `pair2 env2`
    - `pair2 env9`
    - `pair2 env13`
  - suite-level counts:
    - pair1 official reset-positive envs `10`, scale-chain-entry envs `2`
    - pair2 official reset-positive envs `4`, scale-chain-entry envs `3`
  - known positive-distance but non-entry failures remain:
    - pair1 env `5`, `12`
    - pair2 env `1`, `5`, `7`, `12`
    - failure reason:
      - `Could not infer full-rollout mean/std from terminal token rows.`
- operational rule:
  - do not overpromote census candidates into replay-stable official replication points by themselves
  - current saved/replay-stable official scale-chain points are:
    - `seed13579 env12`
    - `pair1 env13`
    - `pair2 env13`
  - remaining census candidates for future formal rerun + repeatB are:
    - `pair1 env10`
    - `pair2 env2`
    - `pair2 env9`

Pair2 env13 official saved scale-chain rule:
- canonical artifact:
  - `/home/chen/RLPFN/artifacts/phase3_pair2_env13_terminal_scale_chain_probe.json`
- repeated formal rerun artifact:
  - `/home/chen/RLPFN/artifacts/phase3_pair2_env13_terminal_scale_chain_probe_repeatB.json`
- probe:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_terminal_local_scale_chain_probe.py`
- locked reading:
  - this suite/env is now replay-stable across saved run + repeatB:
    - `selected_target_local_index = 37`
    - terminal global step `1983`
    - same boundary-contract core
    - same mean-source summary
    - same scale-collapse readout
    - same conclusions
  - it supports the same high-level mechanism as the existing official point:
    - `objective_scope_std_collapse_is_explained_by_truncating_away_high_variance_prefix = true`
    - `objective_window_is_low_variance_slice_not_high_variance_source = true`
    - `all_terminal_candidates_worsen_under_objective_scope = true`
    - within-prefix share `93.6230%`
    - objective/full std ratio `0.07307x`
  - do not overgeneralize the objective-internal dominant bucket:
    - `dominant_objective_variance_source = within_episode1`
    - this is not a cross-suite law
- operational rule:
  - pair2 env13 is approved as the second official saved/replay-stable scale-chain replication point

Pair1 env13 official saved scale-chain rule:
- canonical artifact:
  - `/home/chen/RLPFN/artifacts/phase3_pair1_env13_terminal_scale_chain_probe.json`
- repeated formal rerun artifact:
  - `/home/chen/RLPFN/artifacts/phase3_pair1_env13_terminal_scale_chain_probe_repeatB.json`
- probe:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_terminal_local_scale_chain_probe.py`
- locked reading:
  - this suite/env is replay-stable across saved run + repeatB:
    - `selected_target_local_index = 22`
    - terminal global step `1968`
    - same boundary-contract core
    - same mean-source summary
    - same scale-collapse readout
    - same conclusions
  - it supports the same high-level mechanism as the other official points:
    - `objective_scope_std_collapse_is_explained_by_truncating_away_high_variance_prefix = true`
    - `objective_window_is_low_variance_slice_not_high_variance_source = true`
    - `all_terminal_candidates_worsen_under_objective_scope = true`
    - within-prefix share `98.2202%`
    - objective/full std ratio `0.23858x`
  - do not overgeneralize the objective-internal dominant bucket:
    - `dominant_objective_variance_source = within_episode1`
    - this is not a cross-suite law
- operational rule:
  - pair1 env13 is approved as the third official saved/replay-stable scale-chain replication point

Official boundary-scope counterfactual pack rule:
- canonical artifact:
  - `/home/chen/RLPFN/artifacts/phase3_official_boundary_scope_counterfactual_pack.json`
- probe:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_official_boundary_scope_counterfactual_pack.py`
- locked reading:
  - official saved/replay-stable point set:
    - `seed13579 env12`
    - `pair2 env13`
    - `pair1 env13`
  - all three points already satisfy the better contract:
    - current point uses `current_full_rollout`
    - current point matches the full-rollout centering contract
  - all three objective-scope counterfactuals are worse:
    - boundary abs shrink if using full over objective is strictly positive at every point
    - reward-norm abs shrink if using full over objective is strictly positive at every point
  - aggregate shrink range:
    - boundary abs shrink min/mean/max:
      - `0.4360 / 0.6068 / 0.9142`
    - reward-norm abs shrink min/mean/max:
      - `0.2850 / 1.4054 / 2.1711`
  - aggregate mechanism range:
    - objective/full std ratio:
      - `0.07307x .. 0.23858x`
    - within-prefix share:
      - `93.6230% .. 98.2202%`
- operational rule:
  - treat `full-rollout terminal boundary centering` as a locked Phase 3 guardrail, not as an optional heuristic
  - do not promote `objective_all_tokens` terminal centering as a fix candidate
  - if a future runtime path appears to use objective-scoped terminal centering, treat that as a contract violation to localize
  - do not use this pack to justify a broad many-env objective rewrite by itself

Runtime train-collect boundary drift scan rule:
- canonical artifact:
  - `/home/chen/RLPFN/artifacts/phase3_runtime_terminal_boundary_drift_scan_train.json`
- probe:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_runtime_terminal_boundary_drift_scan.py`
- locked reading:
  - current scan scope:
    - `suite_split = train`
    - suites:
      - `pair1`
      - `pair2`
      - `seed13579`
  - scanned live runtime terminal rows:
    - objective terminal token count `17`
    - envs with objective terminal tokens `14`
  - drift result:
    - boundary shift mismatch count `0`
    - max abs boundary-shift contract error `8.6939e-08`
    - mean abs boundary-shift contract error `3.3567e-08`
  - suite-level max abs errors:
    - `pair1 train = 8.6939e-08`
    - `pair2 train = 2.5343e-08`
    - `seed13579 train = 6.1974e-08`
- operational rule:
  - do not localize the next small runtime fix to train-time terminal boundary centering unless a new drift scan actually finds a mismatch
  - treat current Phase 3 train collect as already compliant with the `full-rollout centering` guardrail
  - if future work touches rollout return normalization or terminal reward handling, rerun this scan before trusting the change

Runtime buffer transport contract rule:
- canonical artifact:
  - `/home/chen/RLPFN/artifacts/phase3_shortrun_buffer_transport_contract_scan.json`
- probe:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_shortrun_buffer_transport_contract_scan.py`
- locked reading:
  - current official Phase 3 train path preserves exactly:
    - `objective_masks`
    - `episode_starts`
    - `rollout_return_means`
    - `rollout_return_stds`
  - contract checked end-to-end:
    - raw rollout buffer
    - `get_gpu_flat()`
    - `get_gpu()`
    - train-side subbatch slice
  - aggregate mismatch counts:
    - source -> flat `0`
    - source -> padded valid `0`
    - flat -> padded valid `0`
    - train-side subbatch flat -> padded `0`
    - seq layout mismatch `0`
- operational rule:
  - treat this as the required regression gate for any future work that touches minibatch packing or transport
  - do not localize a fix to the four transport-critical fields unless this scan first reports a mismatch
Regime-labeled train-side contrast rule:
- canonical nonpositive complement:
  - `/home/chen/RLPFN/artifacts/phase3_train_env_quality_contrast_regime_seed24680_nonpositive.json`
- locked reading:
  - the nonpositive regime contrast is now available in the same labeled wrapper shape as the positive regime contrast
  - high positive mass alone does not imply positive delta in the nonpositive regime
  - future train-side concentration probes must be split by suite regime before any pooling
  - do not treat the nonpositive regime as missing just because it required a bundle/bootstrap step to recover

Regime comparison pack rule:
- canonical artifact:
  - `/home/chen/RLPFN/artifacts/phase3_train_env_quality_regime_compare_pack.json`
 - locked reading:
   - shared signals across regimes are stable:
     - high positive mass alone is not sufficient
     - episode segmentation is not the primary separator
     - low-mass positive envs still beat high-mass nonpositive envs on pre-suffix gap and value corr
   - the only regime-specific separator is early-mass strength, which is positive-only
   - current recommendation remains guardrail-first, not pooled weighting repair

Phase 3 root-cause closure:
- collect / buffer transport are stable
- regime split stays the default guardrail
- train/heldout transfer is mode inversion, not simple weakening
- residual sign-formation failures remain the narrow failure class
- pooled weighting repair is still not justified without a new regime-labeled contrast

Phase 3 residual flip-mode compare pack:
- canonical artifact:
  - `/home/chen/RLPFN/artifacts/phase3_residual_flip_mode_compare_pack.json`
- locked reading:
  - future-carry-dominant residual flips carry stronger anchors and stronger raw residuals
  - bootstrap-dominant residual flips are locally more negative but weaker on anchor strength
  - the two modes reach nearly the same final actor-adv magnitude
- operational rule:
  - if any small repair is attempted next, target the future-carry source chain first
  - keep regime split as the default guardrail until a new labeled contrast is found

Phase 3 future-carry source-chain compare pack:
- canonical artifact:
  - `/home/chen/RLPFN/artifacts/phase3_future_carry_source_chain_compare_pack.json`
- locked reading:
  - env12 is terminal-reset-tail dominant
  - env1 is true-negative-future-residual dominant
  - the only shared negative source bucket is baseline_bootstrap_semantic_mismatch
  - shared candidate coverage is cross-anchor but still branch-aware
- operational rule:
  - if a small fix is attempted next, baseline_bootstrap_semantic_mismatch is the shared candidate target
  - branch-specific follow-up still remains necessary

Phase 3 baseline-bootstrap semantic mismatch compare pack:
- canonical artifact:
  - `/home/chen/RLPFN/artifacts/phase3_baseline_bootstrap_semantic_mismatch_compare_pack.json`
- locked reading:
  - both env12 and env1 share a near-terminal baseline-bootstrap core
  - env1 is tail-only for the bucket
  - env12 also has a mid-episode extension
  - the narrow repair scope is near_terminal_tail_core, not the full pooled bootstrap bucket
- operational rule:
  - if a small fix is attempted next, keep the scope to near-terminal baseline-bootstrap tail core
  - branch-specific follow-up still remains necessary

Phase 3 baseline-bootstrap terminal-adjacent core compare pack:
- canonical artifact:
  - `/home/chen/RLPFN/artifacts/phase3_baseline_bootstrap_terminal_adjacent_core_compare_pack.json`
- locked reading:
  - both env12 and env1 share the terminal-adjacent baseline-bootstrap core
  - env12 still has a mid-episode extension
  - the smallest shared repair scope is terminal_adjacent_core
- operational rule:
  - keep `terminal_adjacent_core` as a guardrail, not a pooled fix target
  - if any narrow follow-up is attempted next, keep the scope to terminal_adjacent_core and treat env12 mid-episode extension as branch-specific
  - branch-specific follow-up still remains necessary

Phase 3 env12 mid-episode extension compare pack:
- canonical artifact:
  - `/home/chen/RLPFN/artifacts/phase3_env12_mid_episode_extension_compare_pack.json`
- locked reading:
  - the env12 mid-episode extension is a contiguous four-token block
  - it is uniformly dominated by baseline_bootstrap_semantic_mismatch
  - no finer split inside the block is supported by current evidence
- operational rule:
  - keep `env12_mid_episode_extension_block` as the remaining branch-specific narrow control direction
  - keep `terminal_adjacent_core` as the shared guardrail

Phase 3 env12 mid-episode extension counterfactual pack:
- canonical artifact:
  - `/home/chen/RLPFN/artifacts/phase3_env12_mid_episode_extension_counterfactual_pack.json`
- locked reading:
  - counterfactual replacement with `terminal_adjacent_core` materially shrinks env12 negative carry and raw residual
  - the shared guardrail still does not change
- operational rule:
  - keep `env12_mid_episode_extension_block` as a narrow branch-specific control candidate
  - keep `terminal_adjacent_core` as the shared guardrail

Phase 3 env12 mid-episode extension runtime control probe:
- canonical artifact:
  - `/home/chen/RLPFN/artifacts/phase3_env12_mid_episode_extension_runtime_control_probe.json`
- locked reading:
  - the env12 mid-episode extension can be realized as a contiguous four-token branch-specific gate
  - no shared-core edit is required
  - the shared guardrail remains `terminal_adjacent_core`
- operational rule:
  - keep `env12_mid_episode_extension_block` as a verified narrow control block
  - keep `terminal_adjacent_core` as the shared guardrail

Phase 3 env12 mid-episode extension runtime gate feasibility probe:
- canonical artifact:
  - `/home/chen/RLPFN/artifacts/phase3_env12_mid_episode_extension_runtime_gate_feasibility_probe.json`
- locked reading:
  - the current runtime mode set does not exactly encode this internal four-token block
  - the smallest runtime edit surface is `MaskedRecurrentPPO._apply_actor_objective_postprocess`
  - no shared-core edit is required
- operational rule:
  - keep `env12_mid_episode_extension_block` as a runtime-feasible branch-specific control candidate
  - keep `terminal_adjacent_core` as the shared guardrail

Phase 3 strict train-side A/B on `seed24680`:
- canonical artifact:
  - `/home/chen/RLPFN/artifacts/phase3_train_rollout_quality_mode_side_effect_compare_pack_seed24680.json`
- locked reading:
  - baseline and override were rerun under the same strict collect contract
  - `max_abs_metric_delta = 0.0`
  - `runtime_mode_no_other_suite_side_effects = true`
  - the runtime branch-specific mode does not leak into the non-target suite under the strict contract
- operational rule:
  - keep the current runtime gate as the narrow branch-specific control
  - do not reintroduce pooled weighting

Phase 3 strict train-side A/B on `seed13579`:
- canonical artifact:
  - `/home/chen/RLPFN/artifacts/phase3_train_rollout_quality_mode_side_effect_compare_pack_seed13579.json`
- locked reading:
  - baseline and override were rerun under the same strict collect contract
  - `max_abs_metric_delta = 0.0`
  - `runtime_mode_no_other_suite_side_effects = true`
  - the runtime branch-specific mode is local on a second non-target suite as well
- operational rule:
  - freeze `tokenwise_scale_env12_mid_episode_extension_block` as a runtime-feasible branch-specific control candidate
  - keep `terminal_adjacent_core` as the shared guardrail

Phase 3 env12 mid-episode extension runtime mode regression:
- canonical artifact:
  - `/home/chen/RLPFN/artifacts/phase3_env12_mid_episode_extension_runtime_mode_regression.json`
- locked reading:
  - the branch-specific control is realizable by runtime mode `tokenwise_scale_env12_mid_episode_extension_block`
  - only objective positions `18..21` in objective episode `0` are scaled
  - the regression reproduces the env12 negative-carry shrink while leaving the shared core unchanged
- operational rule:
  - keep `env12_mid_episode_extension_block` as a runtime-feasible branch-specific control candidate
  - keep `terminal_adjacent_core` as the shared guardrail

Phase 3 strict train-side A/B on `phase3_cross_env_baseline_pair2`:
- canonical artifact:
  - `/home/chen/RLPFN/artifacts/phase3_train_rollout_quality_mode_side_effect_compare_pack_pair2.json`
- locked reading:
  - baseline and override were rerun under the same strict collect contract on the pair2 suite path
  - `max_abs_metric_delta = 0.0`
  - `runtime_mode_no_other_suite_side_effects = true`
  - the runtime branch-specific mode is local on a third non-target suite as well
- operational rule:
  - freeze `tokenwise_scale_env12_mid_episode_extension_block` as a runtime-feasible branch-specific control candidate
  - keep `terminal_adjacent_core` as the shared guardrail

Phase 3 pair2 proxy-to-train transfer audit:
- canonical artifact:
  - `/home/chen/RLPFN/artifacts/phase3_pair2_proxy_to_train_transfer_audit.json`
- locked reading:
  - the env12 local proxy shrink is real, but it is too small relative to the pair2 batch-weighted actor-advantage mass to imply train-side improvement
  - the four-token block accounts for only:
    - `0.2451%` of batch objective tokens
    - `0.0079%` of total normalized abs mass
    - `0.0158%` of total normalized negative mass
    - `0.0862%` of env12 negative mass
  - target-side A/B therefore should not be read as surprising:
    - `train_full_return_delta_change = -0.018476486206054688`
    - `train_suffix_return_delta_change = 0.0`
- operational rule:
  - keep `env12_mid_episode_extension_block` as a runtime-feasible branch-specific control candidate, not a default repair
  - do not interpret local proxy shrink as expected train-side gain under the current many-env aggregation contract
  - next diagnostic scope is update aggregation semantics, not block retuning

Phase 3 train outer-batch snapshot hook:
- canonical runtime snapshot targets:
  - `/home/chen/RLPFN/artifacts/phase3_pair2_train_update_ab_baseline_outer_batch_snapshot.json`
  - `/home/chen/RLPFN/artifacts/phase3_pair2_train_update_ab_override_outer_batch_snapshot.json`
- locked reading:
  - exact update-aggregation audit now has an official strict path
  - the snapshot captures:
    - pre-normalization `actor_advantages`
    - post-normalization `advantages`
    - `ratio`
    - `clipped_objective`
    - `objective_mask`
    - flat env/step/objective-local position metadata
  - the hook is read-only and scoped to the strict pair2 A/B contract
- operational rule:
  - when claiming exact train-side aggregation behavior, use the strict pair2 outer-batch snapshot path
  - do not substitute batch-mass approximations for exact update-aggregation conclusions once the snapshot is available

Phase 3 pair2 exact update aggregation audit:
- canonical artifact:
  - `/home/chen/RLPFN/artifacts/phase3_pair2_exact_update_aggregation_audit.json`
- locked reading:
  - the current one-shot snapshot hook captures the first outer batch that reaches `train()`
  - on the strict pair2 rerun, that captured batch contained only:
    - `env_index = 0`
    - `objective_episode_index = 2`
    - `objective_position = 0..101`
  - the requested env12 branch-specific block was absent:
    - `env12 objective token count in snapshot = 0`
    - `block_token_count = 0`
  - therefore the current snapshot contract is not yet sufficient for exact env12-block transfer analysis
- operational rule:
  - do not claim exact env12 block-level update aggregation from the current first-batch snapshot contract
  - the next narrow runtime change, if this line continues, must be snapshot selection/triggering, not pooled weighting or shared-core repair

Phase 3 selectable snapshot hook:
- locked reading:
  - the outer-batch snapshot hook now supports selector-gated capture by:
    - `target_outer_batch_idx`
    - `target_env_index`
    - `target_objective_episode_index`
    - `target_objective_position_start`
    - `target_objective_position_end`
  - this is still read-only instrumentation only
- operational rule:
  - use selector-gated capture when the root-cause question is about a specific env-segment block
  - if strict runtime validation does not finish within the expected audit envelope, treat that as a runtime-phase anomaly first
  - do not convert selector-gated instrumentation work into shared-core or pooled-weighting edits

Phase 3 pair2 outer-batch selector trace:
- canonical artifact:
  - `/home/chen/RLPFN/artifacts/phase3_pair2_train_side_selector_trace.json`
- locked reading:
  - the strict pair2 train-side outer-batch stream was traced batch-by-batch for selector:
    - `env_index = 12`
    - `objective_episode_index = 0`
    - `objective_position = 18..21`
  - result:
    - `row_count = 128`
    - `target_appeared = false`
    - only one batch contained any `env12` objective tokens:
      - `outer_batch_idx = 104`
      - `objective_episode_indices_present = [9, 10, 11]`
    - `objective_episode_index = 0` never appeared for `env12` on the traced train-side path
  - therefore the env12 block-level exact update-aggregation audit is still unavailable on the current strict pair2 runtime path:
    - not because selector capture is missing
    - but because the requested block does not enter the train-side batch stream under that identity
- operational rule:
  - do not claim exact env12 block-level train aggregation from strict pair2 until this segment-identity mismatch is resolved or explicitly reindexed
  - the next narrow diagnostic, if this line continues, is train-side segment identity tracing, not pooled weighting, shared-core repair, or further block retuning

Phase 3 pair2 train segment identity probe:
- canonical artifact:
  - `/home/chen/RLPFN/artifacts/phase3_pair2_train_segment_identity_probe.json`
- locked reading:
  - source env12 block identity from the saved artifact is:
    - `objective_episode_index = 0`
    - `objective_local_index = 18..21`
    - `global_step = 1964..1967`
  - current strict pair2 train-side rollout at those same global steps is:
    - raw rollout identity:
      - `raw_objective_episode_index = 10`
      - `raw_objective_episode_position = 10..13`
      - `raw_objective_global_position = 18..21`
    - compressed objective identity:
      - `compressed_objective_episode_index = 1`
      - `compressed_objective_episode_position = 10..13`
  - flatten stage preserves the rollout raw identity exactly
  - outer batch `104` preserves the flat raw identity exactly
  - therefore the mismatch is already present before `get_gpu_flat` and before outer-batch transport
- operational rule:
  - do not treat `get_gpu_flat` or outer batch `104` as the current renumbering root cause
  - the remaining narrow root-cause line is source artifact segment identity versus current strict rollout segment identity
  - if this line continues, only do read-only reset/segment contrast on the same env12 objective window

Phase 3 pair2 env12 window reset/segment compare:
- canonical artifact:
  - `/home/chen/RLPFN/artifacts/phase3_pair2_env12_window_reset_segment_compare.json`
- guardrail reading:
  - source env12 window `1946..2002` is one long objective segment:
    - `episode 0`
    - `local 0..56`
  - current strict env12 window `1946..2002` is split inside the same window at:
    - `1954`
    - `1984`
  - current strict segment spans are:
    - `1946..1953`
    - `1954..1983`
    - `1984..2002`
  - target block `1964..1967` therefore maps as:
    - source local `18..21`
    - current strict episode-local `10..13`
    - current strict objective-global `18..21`
  - the `+8` local-index shift equals the full width of the first strict split block `1946..1953`
- operational rule:
  - do not interpret source env12 local indices in this window as stable current strict episode-local coordinates
  - exact block-level compare is only valid after reindexing to current strict reset/segment contract or when expressed in objective-global positions

Phase 3 pair2 env12 source-vs-strict suite parity correction:
- canonical artifact:
  - `/home/chen/RLPFN/artifacts/phase3_pair2_env12_window_reset_segment_compare_heldout_suite.json`
- guardrail reading:
  - the source artifact generation chain uses the `pair2 heldout suite`
  - the earlier `1954/1984` split compare used the `pair2 train suite`
  - these are not the same env12:
    - env/rollout seeds differ
    - sampled environment hypers differ
  - under the corrected same-suite compare on `pair2 heldout env12`, current official strict collect matches the source window contract:
    - one continuous segment across `1946..2002`
    - no internal split at `1954`
    - no internal split at `1984`
    - target block `1964..1967` remains local `18..21`
- operational rule:
  - do not treat the earlier `1954/1984` split as a same-environment reset-contract bug
  - treat it as a suite-selection mismatch unless it reappears under the same heldout suite contract

Phase 3 analysis suite-selection contract lock:
- locked code:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_pair2_env12_window_reset_segment_compare.py`
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_phase3_pair2_env12_window_reset_segment_compare.py`
- guardrail reading:
  - source-side bundle selection is `suite_name + env_index`, not `env_index` alone
  - source `phase3_residual_anchor_gae_path_probe` implies compare-side suite role `heldout`
  - default canonical compare now auto-resolves to the heldout suite path
  - explicit `train_suite.pt` compare is rejected unless mismatch is intentionally enabled
- operational rule:
  - do not allow same-env conclusions to be drawn from a source/current compare unless suite role matches the source artifact contract
  - treat any mixed-suite run as a counterfactual mismatch probe, not as a canonical artifact

Phase 3 analysis suite-contract audit:
- canonical audit note:
  - `/home/chen/RLPFN/artifacts/phase3_analysis_suite_contract_audit.md`
- guardrail reading:
  - audited `token_row_bundles` source readers are safe if they already key by `suite_name + env_index`
  - the only confirmed vulnerability in this bug class was the env12 window reset compare probe, which is now fixed
  - `phase3_pair2_train_segment_identity_probe.py` is an intentional train-side identity probe, not a same-suite reset-contract compare
- operational rule:
  - do not spend more root-cause cycles on suite-selection mismatch for audited probes unless a new multi-suite source reader without `suite_name` filtering appears

Phase 3 root-cause probe trust matrix:
- canonical note:
  - `/home/chen/RLPFN/artifacts/phase3_root_cause_probe_trust_matrix.md`
- guardrail reading:
  - trusted for bottleneck reasoning:
    - `phase3_pair2_proxy_to_train_transfer_audit.json`
    - `phase3_pair2_train_segment_identity_probe.json`
  - conditionally trusted:
    - `phase3_pair2_exact_update_aggregation_audit.json`
    - use only for batch-level dilution stages, not exact env12 block attribution
  - guardrail-only:
    - `phase3_pair2_env12_window_reset_segment_compare.json`
- operational rule:
  - continue bottleneck work only on the still-trusted `proxy_to_train_update_transfer_semantics` axis
  - do not promote absent-block exact aggregation claims into root-cause conclusions

Phase 3 pair2 captured-scope update-transfer probe:
- canonical artifact:
  - `/home/chen/RLPFN/artifacts/phase3_pair2_captured_scope_update_transfer_probe.json`
- locked reading:
  - the actual strict captured outer-batch scope is:
    - `env_index = 0`
    - `objective_episode_index = 2`
    - `objective_position = 0..101`
    - `token_count = 102`
  - this single captured segment already equals the whole-batch objective
  - baseline and override are numerically identical on that captured scope across:
    - pre-normalization actor advantages
    - post-normalization advantages
    - clipped policy objective
    - ratio mean
    - clip-active fraction
  - therefore the current captured batch is not the transfer site for the env12 branch-specific gain
  - and this artifact cannot support an in-batch many-env dilution claim, because the captured objective scope is not a many-env mixture
- operational rule:
  - do not use the current exact update aggregation artifact as proof that env12 signal was diluted inside the captured strict batch
  - use it only to lock the narrower fact that:
    - the captured batch is a non-target env0 batch
    - the branch-specific override leaves that batch unchanged

Phase 3 pair2 target-present strict train-side outer batch capture:
- canonical artifacts:
  - `/home/chen/RLPFN/artifacts/phase3_pair2_target_present_outer_batch_selector_trace.json`
  - `/home/chen/RLPFN/artifacts/phase3_pair2_target_present_outer_batch_snapshot.json`
- locked reading:
  - current strict train-side target-present identity is:
    - `outer_batch_idx = 104`
    - `env_index = 12`
    - `raw_objective_episode_index = 10`
    - `raw_objective_episode_position = 10..13`
    - `raw_objective_global_position = 18..21`
  - selector trace confirms:
    - `target_appeared = true`
    - first match at `epoch 1 / outer batch 104`
    - `target_match_count = 4`
  - snapshot confirms the exact target block is present in that batch with:
    - `snapshot_target_match_count = 4`
    - `objective_total = 102`
    - `actor_objective_mode = tokenwise`
    - `runtime_current_suite_name = pair2`
  - contract caveat:
    - this capture was produced under the pair2 many-env audit scale:
      - `n_samples = 2048`
      - `single_eval_pos = 1946`
      - `outer_epochs = 1`
    - it is not Phase 2 runtime-comparable
- trusted runtime reference remains Phase 2:
  - `/home/chen/RLPFN/artifacts/phase2_regression_pack/phase2_regression_suite_summary.json`
  - trusted contract:
    - `n_steps = 256`
    - `batch_size = 256`
    - `outer_epochs = 2`
    - `single_eval_pos = 64`
    - `strict_fixed_env_mode = true`
    - `deterministic_actor_sampling = true`
    - `deterministic_batch_plan = true`
- operational rule:
  - do not use the stale source-local selector `env12 / episode0 / pos18..21` for strict train-side capture
  - use the locked current strict selector:
    - `outer_batch 104 / env12 / raw episode10 / pos10..13`
  - do not use this target-present capture to reason about runtime parity with Phase 2

Phase 3 pair2 Phase-2-scale target-present capture:
- canonical artifacts:
  - `/home/chen/RLPFN/artifacts/phase3_pair2_phase2_scale_target_present_capture.json`
  - `/home/chen/RLPFN/artifacts/phase3_pair2_phase2_scale_target_present_selector_trace.json`
- locked reading:
  - under trusted Phase 2-equivalent scale:
    - `n_steps = 256`
    - `batch_size = 256`
    - `outer_epochs = 2`
    - `single_eval_pos = 64`
  - the previously identified current-strict selector does not hit:
    - `env12 / objective_episode_index 10 / objective_position 10..13`
  - selector trace shows:
    - `target_appeared = false`
    - only one outer batch contains env12 objective tokens:
      - `outer_batch_idx = 13`
    - in that batch, target episode `10` exists only at positions `0..7`
    - failure mode:
      - `target_status = target_position_range_absent`
  - no target-present snapshot is written at trusted Phase 2 scale
- operational rule:
  - do not carry the `outer_batch 104 / env12 / raw episode10 / pos10..13` selector from the larger pair2 many-env contract into the trusted Phase 2-scale contract

Phase 3 pair2 Phase-2-scale selector reidentification:
- canonical artifact:
  - `/home/chen/RLPFN/artifacts/phase3_pair2_phase2_scale_selector_reidentified.json`
- locked reading:
  - trusted Phase-2-scale stale selector:
    - `env12 / objective_episode_index 10 / position 10..13`
  - trusted Phase-2-scale reidentified selector:
    - `outer_batch_idx = 13`
    - `env_index = 12`
    - `objective_episode_index = 10`
    - `objective_position = 0..7`
    - `token_count = 8`
  - preserved:
    - env identity
    - episode identity
  - changed:
    - position range only
- operational rule:
  - trusted Phase-2-scale recapture must use:
    - `outer_batch13 / env12 / episode10 / pos0..7`

Phase 3 pair2 Phase-2-scale trusted selector recapture:
- canonical artifacts:
  - `/home/chen/RLPFN/artifacts/phase3_pair2_phase2_scale_target_present_recapture.json`
  - `/home/chen/RLPFN/artifacts/phase3_pair2_phase2_scale_target_present_recapture_selector_trace.json`
  - `/home/chen/RLPFN/artifacts/phase3_pair2_phase2_scale_target_present_recapture_snapshot.json`
- locked reading:
  - trusted selector hit:
    - `target_appeared = true`
    - `epoch_idx = 1`
    - `outer_batch_idx = 13`
    - `target_match_count = 8`
  - trusted snapshot fixed point:
    - `env12 / objective_episode_index 10 / episode_pos 0..7`
    - `objective_global_pos 22..29`
    - `snapshot_target_match_count = 8`
    - all target rows are `objective_mask = 1`
  - trusted snapshot sign:
    - pre-normalization actor advantages are all negative
    - post-normalization advantages are all negative
    - clipped objectives are all negative
- operational rule:
  - for trusted Phase-2-scale train-side analysis, use:
    - `outer_batch13 / env12 / episode10 / pos0..7 / global22..29`
  - do not fall back to the stale larger-contract selector:
    - `env12 / episode10 / pos10..13`

Phase 3 pair2 Phase-2-scale exact update aggregation audit:
- canonical artifact:
  - `/home/chen/RLPFN/artifacts/phase3_pair2_phase2_scale_exact_update_aggregation_audit.json`
- locked reading:
  - trusted target scope:
    - `outer_batch13 / env12 / episode10 / pos0..7 / global22..29`
    - `token_count = 8`
  - exact target negative mass:
    - pre:
      - `7.032025098800659`
    - post:
      - `14.139300107955933`
    - policy-num:
      - `13.329894781112671`
  - exact target retention:
    - `post_over_pre_retention = 2.0107010298310013`
    - `policy_over_post_retention = 0.9427549227569034`
  - exact target share of whole-batch negative mass after clipping:
    - `0.17802543705133486`
  - whole-batch and env12 metrics are identical in this captured trusted-scale batch
- operational rule:
  - do not use this artifact as evidence for cross-env dilution
  - do use it as evidence that, inside this captured trusted-scale batch:
    - normalization is the dominant amplification stage for the target block
    - ratio/clipping only modestly attenuates that amplified target mass

Phase 3 pair2 Phase-2-scale batch-to-epoch aggregation probe:
- canonical artifact:
  - `/home/chen/RLPFN/artifacts/phase3_pair2_phase2_scale_batch_to_epoch_aggregation_probe.json`
- locked reading:
  - trusted epoch layout:
    - `16` outer batches
    - one env per outer batch
    - `192` objective tokens per outer batch
    - env order:
      - `env0 .. env15`
  - trusted target batch:
    - `outer_batch13 = env12`
    - target scope appears in exactly one outer batch per epoch
  - trusted train-loop contract:
    - optimizer steps once per outer batch
    - outer-batch loss is normalized by that batch's own objective total
    - outer-batch updates are sequential within the epoch
  - trusted implication:
    - this line is about cross-batch overwrite, not intra-batch cross-env mixing
- operational rule:
  - do not go back to pooled-weighting or same-batch cross-env dilution explanations for this trusted-scale target block
  - if this line continues, next read-only work should inspect whether later outer-batch steps counteract the env12 local step

Phase 3 pair2 Phase-2-scale cross-batch overwrite probe:
- canonical artifact:
  - `/home/chen/RLPFN/artifacts/phase3_pair2_phase2_scale_cross_batch_overwrite_probe.json`
- locked reading:
  - trusted actor-only downstream steps `outer_batch14..16` are aligned with the target `outer_batch13` step
  - pairwise cosine vs target:
    - `outer_batch14`: `0.7779969016740984`
    - `outer_batch15`: `0.85268642774419`
    - `outer_batch16`: `0.8921939052438096`
  - downstream cumulative projection on target direction:
    - `1.955839623062271`
  - net retention after target plus downstream:
    - `2.9583765580495096`
- operational rule:
  - do not explain the trusted target-block failure as immediate actor-only overwrite from `outer_batch14..16`
  - if this line continues, the next narrow explanation axis is:
    - full-loss/shared-core interference
    - not immediate downstream actor-only opposition

Phase 3 pair2 Phase-2-scale local step direction probe:
- canonical artifact:
  - `/home/chen/RLPFN/artifacts/phase3_pair2_phase2_scale_local_step_direction_probe.json`
- locked reading:
  - trusted compare scope:
    - actor-only vs `policy + value (+ entropy if enabled)` on the same pre-`batch13` state
    - aux losses excluded from this compare
  - target batch:
    - `outer_batch13 = env12`
  - actor vs full gradient:
    - `cosine = 0.017400420572603232`
    - full gradient norm is about `59x` actor-only
  - actor vs full parameter delta:
    - `cosine = 0.8892663378371908`
    - projection on actor direction:
      - `1.0185639457613322`
  - residual (`full - actor`) vs actor delta:
    - `cosine = 0.031602329955936204`
    - projection on actor direction:
      - `0.016680717798082487`
- operational rule:
  - do not explain this trusted local step as simple opposite-direction value interference
  - do treat it as evidence that the shared value path injects a large mostly orthogonal local-step component
  - keep in mind:
    - actual trusted runtime still has `aux_flow_weight = 0.2`
    - so the auxiliary shared-core line is not closed by this artifact

Phase 3 contract guardrail after Phase 2 baseline correction:
- locked code fact:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_multi_env_optimization_audit.py`
    - default train profile is now `phase2_shared_backbone_contract`
- locked trusted baseline:
  - the only accepted Phase 3 baseline is:
    - Phase 2 shared actor-critic backbone contract
    - plus many-env
    - plus the narrow adjustment under test
    - and no other runtime/profile drift
- locked forbidden drift:
  - do not call `trusted_sep_reset_mainline` a trusted baseline for Phase 3 root-cause or A/B work
  - do not rely on:
    - `normalize_advantage = True`
    - nonzero or unresolved runtime q/flow weights
    - `_resolve_audit_env_config(...)` in place of `_build_audit_env_cfg(...)`
  - do not mix actor-only / aux-flow / pooled-weighting exploratory lines into the baseline compare
- operational rule:
  - treat all pre-correction `phase3_pair2_phase2_scale_*` artifacts as exploratory only
  - only artifacts re-run under `phase2_shared_backbone_contract` can be used as trusted evidence for whether a narrow adjustment improves many-env optimization

Phase 2 milestone vs current Phase 3 baseline gradient compare:
- canonical artifact:
  - `/home/chen/RLPFN/artifacts/phase3_phase2_milestone_contract_gradient_compare.json`
- locked reading:
  - current single-env contract signatures now match exactly:
    - `signature_diff = {}`
  - current initial PPO policy state also matches exactly after synchronized build seeding:
    - `policy_param_max_abs_diff = 0.0`
  - but current runtime is still not numerically exact:
    - `old_values max diff = 0.0019040`
    - `advantages max diff = 0.0019382`
    - `max_abs_grad_delta = 5.4240e-06`
- operational rule:
  - do not claim the current Phase 3 baseline is fully milestone-identical yet
  - do claim the remaining mismatch is narrow:
    - not config drift
    - not parameter init drift
    - likely a residual one-env fixed-suite collect/reset path drift
  - therefore:
    - do not start trusted many-env narrow-adjustment A/B until this residual compare is explained or eliminated

Phase 2 one-env fixed-suite collect drift probe:
- canonical artifact:
  - `/home/chen/RLPFN/artifacts/phase3_phase2_one_env_fixed_suite_collect_drift_probe.json`
- locked reading:
  - reset-layer numeric drift exists:
    - reset obs max diff: `0.2776488`
    - reset `_state_t` max diff: `0.2776488`
    - reset `_action_t` max diff: `0.1761909`
  - but strict collect does not first diverge there:
    - first policy-step visible inputs are exact:
      - `obs_t = 0.0`
      - `action_t = 0.0`
      - `reward_t = 0.0`
      - `reward_mask_t = 0.0`
    - first policy-step cache input diff: `0.0`
    - all captured `env_info` tensor leaves are exact across `29` leaves:
      - `env_info_all_tensor_max_abs_diff = 0.0`
    - first `obs_full` built by `_build_obs_from_rollout_step_inputs` is exact:
      - `obs_full max diff = 0.0`
  - earliest collect-relevant drift is already inside the first model forward path:
    - `drift_localization.stage = model_forward_path`
    - first actor-output diffs:
      - `action_mean = 3.3885e-05`
      - `value_logits = 0.00362635`
      - `values = 1.3649e-04`
    - first stored rollout-buffer step then matches that same scale:
      - `actions = 3.3885e-05`
      - `values = 1.3649e-04`
      - `rewards = 1.0595e-05`
      - `next_states = 1.4305e-05`
- operational rule:
  - do not keep explaining the residual trusted drift as reset transport mismatch
  - do not trust many-env A/B yet
  - from this point, the next trusted narrowing step is:
    - first-step forward-subpath compare
    - not more suite/profile/reset speculation

Phase 2 one-env first-step forward-subpath probe:
- canonical artifact:
  - `/home/chen/RLPFN/artifacts/phase3_phase2_one_env_first_step_forward_subpath_probe.json`
- locked reading:
  - this probe traces the real first call inside the policy step; it is not a second-pass replay
  - first-call forward subpath results:
    - `encoded_token = 0.0`
    - `rwkv_input_token = 0.0`
    - `rwkv_forward_hidden = 0.015625`
    - `latent_pi = 0.015625`
    - `latent_vf = 0.015625`
    - `action_distribution_mean = 3.3885e-05`
    - `value_logits = 0.00362635`
  - returned first-step outputs match the downstream part of that same split:
    - `returned_action_mean = 3.3885e-05`
    - `returned_value_logits = 0.00362635`
    - `returned_values = 1.3649e-04`
  - localization:
    - `forward_subpath_localization.stage = rwkv_forward_step_path`
- operational rule:
  - do not keep treating the remaining trusted drift as reset-path or tokenization drift
  - do not rely on any earlier “second-pass recompute says forward path is exact” reading
  - the current defended root cause frontier is:
    - exact token into the core
    - non-exact hidden out of `rwkv_core.forward_step`
  - therefore:
    - no trusted many-env A/B yet
    - next narrowing step must target RWKV core cache/dtype/kernel behavior on the first step

Phase 2 one-env RWKV core-step cache/dtype/kernel probe:
- canonical artifact:
  - `/home/chen/RLPFN/artifacts/phase3_phase2_one_env_rwkv_core_step_probe.json`
- locked reading:
  - runtime was normal for this probe:
    - `elapsed_wall_time_sec = 17.2676`
    - this is below the Phase 2 standard-test runtime scale of about `103s`
  - trusted contract is still exact at the setup level:
    - `signature_diff = {}`
    - `policy_param_max_abs_diff = 0.0`
  - real first RWKV core path flags are exact:
    - `path_flags_exact_match = true`
    - both sides use `uses_official_eval_path = true`
  - upstream/core-materialization checks are exact:
    - `token_arg_max_abs_diff = 0.0`
    - `init_state_max_abs_diff = 0.0`
    - `state_arg_max_abs_diff = 0.0`
    - `official_flat_state_input_max_abs_diff = 0.0`
    - `official_weight_snapshot_max_abs_diff = 0.0`
  - native PyTorch core replay is exact:
    - `native_replay_hidden_max_abs_diff = 0.0`
  - actual official eval hidden is not exact:
    - `actual_hidden_max_abs_diff = 0.015625`
  - localization:
    - `core_step_localization.stage = official_eval_kernel_path`
- operational rule:
  - do not explain the residual Phase 2-vs-current drift as:
    - reset transport
    - token encoding
    - cache construction
    - official eval weight snapshot
    - native RWKV replay
  - do not start trusted many-env narrow-adjustment A/B until this official-eval shortcut mismatch is either eliminated or explicitly guarded
  - the next allowed narrowing step is exactly:
    - read-only official-eval-vs-native milestone contract probe
    - same one-env trusted contract
    - measure whether bypassing/simulating the native core path closes first-step, rollout, and gradient drift

Phase 2 official-eval vs native milestone contract probe:
- canonical artifact:
  - `/home/chen/RLPFN/artifacts/phase3_phase2_official_eval_vs_native_contract_probe.json`
- locked reading:
  - this probe tests the already-localized narrow runtime control:
    - force native RWKV core path instead of the batch=1/no-grad/eval official eval shortcut
    - no production training code is changed by the probe
  - runtime was normal:
    - `elapsed_wall_time_sec = 23.7160`
    - this is below the Phase 2 standard-test runtime scale of about `103s`
  - trusted setup remains exact:
    - `signature_diff = {}`
    - `policy_param_max_abs_diff = 0.0`
  - both first rollout paths would have used official eval without the override:
    - `phase2_would_use_official = true`
    - `phase3_would_use_official = true`
  - forced-native calls were used for the rollout:
    - `phase2_native_calls = 256`
    - `phase3_native_calls = 256`
  - default reference drift before the override:
    - `default_first_hidden_max_abs_diff = 0.015625`
    - `default_rollout_max_abs_diff = 0.0019381940`
    - `default_old_values_max_abs_diff = 0.0019039959`
    - `default_returns_max_abs_diff = 0.0002077818`
    - `default_advantages_max_abs_diff = 0.0019381940`
    - `default_grad_max_abs_delta = 5.4240227e-06`
  - forced native closes the contract exactly:
    - `native_first_hidden_max_abs_diff = 0.0`
    - `native_rollout_max_abs_diff = 0.0`
    - `native_grad_max_abs_delta = 0.0`
    - all scalar loss/stat diffs are `0.0`
- operational rule:
  - the remaining Phase 2-vs-current baseline drift is now attributed to the official eval shortcut
  - do not route trusted regression comparisons through unguarded official eval fastpath
  - do not convert this result into an objective/weighting change
  - before trusted many-env A/B:
    - formalize a minimal strict-regression/native-rollout guard or bypass
    - rerun the original milestone gradient compare without analysis monkeypatch
    - require exact first-step / rollout / scalar / gradient match

Formal strict-native rollout guard:
- locked code facts:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/models/rwkv7_pfn.py`
    - `RWKV7Core.force_native_eval_forward_step` defaults to `False`
    - official eval shortcut is bypassed only when this flag is true
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/sb3_recurrent_ppo.py`
    - `build_recurrent_ppo(..., strict_native_rollout=False)` defaults to normal production behavior
    - explicit `strict_native_rollout=True` sets the actor core and optional value core to native forward-step mode
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_phase2_milestone_contract_gradient_compare.py`
    - trusted milestone compare defaults to `strict_native_rollout=True`
    - the field is part of both Phase 2 and Phase 3 contract signatures
- canonical green artifact:
  - `/home/chen/RLPFN/artifacts/phase3_phase2_milestone_contract_gradient_compare.json`
- locked green result:
  - observed command wall time was about `28.6s`
  - `strict_native_rollout = true`
  - `signature_diff = {}`
  - `policy_param_max_abs_diff = 0.0`
  - all rollout tensor max diffs are `0.0`
  - all scalar loss/stat diffs are `0.0`
  - gradient compare:
    - `max_abs_grad_delta = 0.0`
    - `l2_delta_norm = 0.0`
    - `phase2_grad_norm = 5.755584239959717`
    - `phase3_grad_norm = 5.755584239959717`
- verification:
  - helper tests:
    - `5 passed`
  - default official eval fastpath preservation test:
    - `1 passed`
- operational rule:
  - this is now the only trusted Phase 2 milestone comparison contract for Phase 3 work
  - normal production/default PPO behavior must not silently enable this bypass
  - trusted numeric/gradient compares must explicitly include `strict_native_rollout=True`
  - Phase 3 A/B work can proceed only if all non-target settings match this guarded Phase 2 shared-backbone contract
  - any future artifact without this field is not comparable to the current trusted baseline

Phase 3 pair2 preflight contract:
- canonical artifact:
  - `/home/chen/RLPFN/artifacts/phase3_preflight_contract_check_pair2.json`
- locked code facts:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_multi_env_optimization_audit.py`
    - accepts and reports `strict_native_rollout`
    - post-policy bundle contracts include `strict_native_rollout`
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_preflight_contract_check.py`
    - compares many-env baseline signature against the guarded Phase 2 milestone signature
- locked green result:
  - `elapsed_wall_time_sec = 1.6969`
  - `preflight_passed = true`
  - Phase 3 pair2 baseline uses:
    - `train_profile = phase2_shared_backbone_contract`
    - `n_envs = 16`
    - `n_steps = 256`
    - `batch_size = 256`
    - `n_epochs = 1`
    - `learning_rate = 0.0002`
    - `target_kl = 0.03`
    - `actor_baseline_mode = learned`
    - `actor_gae_space = normalized`
    - `actor_objective_mode = tokenwise`
    - `normalize_advantage = false`
    - `vf_coef = 0.5`
    - `reset_env_state_at_sep = true`
    - `separate_value_backbone = false`
    - `strict_fixed_env_mode = true`
    - `deterministic_actor_sampling = true`
    - `deterministic_batch_plan = true`
    - `strict_native_rollout = true`
  - runtime guard confirmed:
    - `algo_strict_native_rollout = true`
    - `actor_core_force_native_eval_forward_step = true`
  - only allowed Phase 2-vs-Phase 3 baseline diffs:
    - `n_envs`
    - `env_rng_seeds`
    - `rollout_rng_seeds`
  - `unexpected_diff = {}`
- operational rule:
  - pair2 many-env baseline is now contract-green
  - no many-env target-side A/B is trusted unless it first matches this preflight contract
  - the override arm may differ only by the previously justified narrow actor-objective mode
  - forbid adding:
    - actor-only baseline
    - aux-flow/q runtime override
    - pooled weighting change
    - `normalize_advantage=True`
    - separate value backbone
    - unguarded official eval shortcut

Phase 3 pair2 target-side A/B guardrail:
- canonical artifact:
  - `/home/chen/RLPFN/artifacts/phase3_pair2_train_update_ab_compare_pack.json`
- status:
  - the initial target-side A/B numbers were superseded by the deterministic-eval rerun on 2026-04-15
  - do not use any older result where policy eval failed to forward `deterministic_actor_sampling = true`
- locked code facts:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_pair2_train_update_ab_compare_pack.py`
    - enforces the pair2 preflight artifact before running A/B
    - uses Phase-2-scale `n_samples = 256` and `single_eval_pos = 64`
    - requires `strict_native_rollout = true`
    - rejects any baseline runtime-scoped objective mode
    - permits the override arm to differ only by:
      - `actor_objective_mode_override = tokenwise_scale_env12_mid_episode_extension_block`
      - `actor_objective_runtime_current_suite_name = pair2`
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_multi_env_optimization_audit.py`
    - `_run_policy_eval()` forwards `deterministic_actor_sampling`
    - post-policy bundle contracts include `actor_objective_mode`
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_pair2_post_policy_side_effect_eval.py`
    - performs eval-only heldout/non-target side-effect checks from saved post-policy bundles
- verification:
  - `python -m py_compile reinforce-terminal-explore/ticl/analysis/phase3_multi_env_optimization_audit.py reinforce-terminal-explore/ticl/analysis/phase3_pair2_train_update_ab_compare_pack.py reinforce-terminal-explore/ticl/analysis/phase3_pair2_post_policy_side_effect_eval.py`
  - `conda run -n rlpfn python -m pytest reinforce-terminal-explore/ticl/tests/test_phase3_multi_env_optimization_audit.py::test_post_policy_bundle_roundtrip reinforce-terminal-explore/ticl/tests/test_phase3_multi_env_optimization_audit.py::test_run_phase3_audit_resumes_post_policy_bundle_without_train_loop reinforce-terminal-explore/ticl/tests/test_phase3_pair2_train_update_ab_compare_pack.py -q`
  - result:
    - `4 passed`
- locked result:
  - `preflight_passed = true`
  - `unexpected_diff = {}`
  - shared baseline/override contract keeps:
    - `ppo_actor_baseline_mode = learned`
    - `ppo_actor_gae_space = normalized`
    - `ppo_normalize_advantage = false`
    - `ppo_vf_coef = 0.5`
    - `ppo_reset_env_state_at_sep = true`
    - `ppo_separate_value_backbone = false`
    - `runtime_normalized_q_value_weight_override = null`
    - `runtime_next_state_flow_matching_weight_override = null`
    - `strict_fixed_env_mode = true`
    - `deterministic_actor_sampling = true`
    - `deterministic_batch_plan = true`
    - `strict_native_rollout = true`
  - pre metrics unchanged:
    - `pre_train_vs_zero_gap_delta.full_return_gap = 0.0`
    - `pre_train_vs_zero_gap_delta.suffix_return_gap = 0.0`
  - baseline post-train vs zero:
    - `full_return_gap = -0.0539064407`
    - `suffix_return_gap = -0.0732917786`
  - override post-train vs zero:
    - `full_return_gap = -0.0136370659`
    - `suffix_return_gap = -0.0299243927`
  - override improvement:
    - `post_train_vs_zero_gap_delta.full_return_gap = +0.0402693748`
    - `post_train_vs_zero_gap_delta.suffix_return_gap = +0.0433673859`
  - post-policy bundles saved:
    - `/home/chen/RLPFN/artifacts/phase3_pair2_train_update_ab_baseline_post_policy_bundle.pt`
    - `/home/chen/RLPFN/artifacts/phase3_pair2_train_update_ab_override_post_policy_bundle.pt`
  - heldout eval skipped:
    - this artifact is target-side only
    - do not cite it as heldout generalization evidence
  - train outer-batch snapshot selector did not hit:
    - `baseline.written = false`
    - `override.written = false`
    - do not cite it as exact env12 block aggregation evidence
- runtime guard:
  - complete command wall time:
    - `real = 456.12s`
  - training is not the slow path:
    - baseline train loop `19.981s`
    - override train loop `18.448s`
  - serial strict-native train-suite policy eval dominates runtime:
    - baseline `pre_train_eval = 122.541s`
    - baseline `post_train_eval = 123.717s`
    - override `post_train_eval = 124.880s`
  - any future rerun should avoid repeating full strict serial eval unless the heldout/non-target result is explicitly needed
- operational rule:
  - the branch-specific adjustment is a runtime-feasible small-control candidate with target-side benefit on pair2
  - it is not a default fix because the target-side train gap remains negative and heldout was not evaluated
  - future Phase 3 A/B must continue to inherit this preflight contract and may not add actor-only, aux-flow, pooled weighting, `normalize_advantage=True`, separate value backbone, or unguarded official eval shortcut

Phase 3 pair2 eval-only heldout side-effect guardrail:
- canonical artifact:
  - `/home/chen/RLPFN/artifacts/phase3_pair2_post_policy_side_effect_eval.json`
- canonical inputs:
  - `/home/chen/RLPFN/artifacts/phase3_pair2_train_update_ab_baseline_post_policy_bundle.pt`
  - `/home/chen/RLPFN/artifacts/phase3_pair2_train_update_ab_override_post_policy_bundle.pt`
- locked result:
  - `eval_only = true`
  - `no_train_loop = true`
  - heldout average side-effect:
    - `override_minus_baseline_full_return_mean = +0.0055379868`
    - `override_minus_baseline_suffix_return_mean = +0.0107862949`
  - per-env side-effect is mixed:
    - full-return positive/negative env count:
      - `9 / 7`
    - suffix-return positive/negative env count:
      - `9 / 7`
    - largest full-return loss:
      - `-0.9088620096`
    - largest full-return gain:
      - `+0.7089591026`
- operational rule:
  - this branch-specific control currently has small average heldout impact, not a clean all-env improvement
  - do not promote it to default without repeat determinism and an additional target-side suite A/B
