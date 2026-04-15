# Phase 3 Multi-Env Optimization Ledger

## Scope

This ledger supersedes the earlier Phase 3 checkpoint-only cross-environment evaluation ledger **for optimization questions**.

The old Phase 3 ledger remains useful for:

- checkpoint-level heldout generalization observation

It is **not** the right baseline for:

- locating optimization bottlenecks in the Phase 3 training semantics
- measuring whether continued PPO training improves heldout cross-environment return

## Correct Phase 3 Contract

Phase 3 optimization should be defined as:

- multi-environment training
- multi-environment testing
- one fixed `n_samples` rollout per environment
- each environment rollout may contain multiple internal episodes via terminal reset
- fixed `train_suite`
- fixed `heldout_suite`
- training on `train_suite`
- evaluation on both `train_suite` and `heldout_suite`

This matches the actual question:

- does the training procedure improve cross-environment heldout performance under the intended many-env regime?

## Baseline Semantics

Current baseline training profile for this ledger:

- `train_profile = trusted_sep_reset_mainline`
- `ppo_actor_baseline_mode = learned`
- `ppo_normalize_advantage = true`
- `ppo_actor_gae_space = normalized`
- `ppo_vf_coef = 0.5`
- `ppo_reset_env_state_at_sep = true`
- `ppo_separate_value_backbone = false`
- runtime aux overrides = `None`
- trusted optimizer settings:
  - `batch_size = 256`
  - `n_epochs = 1`
  - `learning_rate = 2e-4`
  - `target_kl = 0.03`

Rationale:

- this matches the only currently trusted training semantics recorded in:
  - `/home/chen/RLPFN/artifacts/sep_state_reset_fix_debug.md`
- the only intended change for Phase 3 is:
  - switch from single-environment training to fixed many-environment suite training
- all other training semantics should stay aligned to the trusted mainline path unless explicitly being tested

## Driver

Training/evaluation audit entry:

- `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_multi_env_optimization_audit.py`

Core behavior:

- binds PPO training env batch to a fixed `train_suite`
- sets `n_envs = train_suite.batch_size`
- sets `n_steps = n_samples`
- trains PPO on that fixed many-env batch for `outer_epochs`
- evaluates pre/post on both `train_suite` and `heldout_suite`
- keeps one fixed-length rollout per environment via the suite definition
- does **not** imply one episode per environment; a rollout can cross multiple terminal-reset episodes

## Status

- 2026-04-12: Regime-internal suite/env concentration probe reran under the current code path and remained stable; it is still the canonical readout for “who is actually dominating update mass” versus “who is dominating pooled covariance”.
  - artifact:
    - `/home/chen/RLPFN/artifacts/phase3_regime_env_concentration_probe.json`
  - entrypoint:
    - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_regime_env_concentration_probe.py`
  - tests:
    - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_phase3_regime_env_concentration_probe.py`
  - locked reading:
    - positive regime update-mass concentration is sharp:
      - positive regime `top3_share = 0.8432`
      - largest positive suite share within regime:
        - `pair1 mass_share_within_regime = 0.8814`
    - nonpositive regime keeps a local positive signal:
      - `local_corr_mass_vs_pre_gap = +0.0495`
      - `regime_local_covariance_numerator = +0.1511`
    - but the same regime becomes negative after pooled centering:
      - `pooled_covariance_numerator_contribution = -0.7622`
    - positive regime dominates the pooled update path:
      - `mass_share_global = 0.6799`
    - current rerun remained identical at the regime level:
      - `positive_regime_mass_share_exceeds_nonpositive = true`
      - `positive_regime_few_envs_dominate_mass = true`
      - `nonpositive_signal_flips_under_global_centering = true`
      - `pooled_covariance_numerator_contribution = -10.4017`
      - absolute pooled-cov dominance ratio vs nonpositive is roughly `13.6x`
  - important anomaly now locked:
    - top update-mass envs and top pooled-cov envs are not the same objects
    - positive regime top mass envs are the pair1 heavy-mass block:
      - `pair1 env10 = 41.36%` of positive mass
      - `pair1 env12 = 22.82%`
      - `pair1 env5 = 20.14%`
    - but the largest absolute pooled-cov contributor is:
      - `pair2 env12`
      - `dist1_to_4_mass_share = 0.13%` of positive regime mass
      - `pre_vs_zero_suffix_gap = 21.4524`
      - `pooled_covariance_numerator = -8.3785`
    - implication:
      - pooled covariance is not equivalent to mass concentration
      - future many-env bottleneck probes must report both:
        - top envs by update-mass proxy
        - top envs by pooled covariance contribution

- 2026-04-11: Regime-aware mixed-update probe now exists and is the canonical way to read many-env token-quality conflict.
  - artifact:
    - `/home/chen/RLPFN/artifacts/phase3_regime_mixed_update_probe.json`
  - entrypoint:
    - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_regime_mixed_update_probe.py`
  - tests:
    - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_phase3_regime_mixed_update_probe.py`
  - locked reading:
    - pooled all-env correlation is not a safe summary by itself
    - canonical booleans now expected in the artifact:
      - `regime_split_visible_in_raw_correlation = true`
      - `pooled_sign_hides_regime_split = true`
      - `positive_regime_dominates_pooled_update = true`
    - on the current four-suite set:
      - all-env pooled:
        - `corr_dist1_to_4_mass_vs_pre_gap = -0.0445`
      - positive regime:
        - `corr_dist1_to_4_mass_vs_pre_gap = -0.0479`
        - `mass_share = 0.6799`
        - `pooled_covariance_contribution_share = 0.9317`
      - nonpositive regime:
        - `corr_dist1_to_4_mass_vs_pre_gap = +0.0495`
        - `mass_share = 0.3201`
        - `pooled_covariance_contribution_share = 0.0683`
    - implication:
      - the regime split is real at raw-correlation level
      - but pooled update statistics are dominated by the positive regime because it carries most of the near-terminal mass and most of the covariance contribution
      - therefore:
        - mixing regimes can hide the conflict instead of exposing it
        - future optimization/fitting probes must report:
          - per-regime correlation
          - per-regime mass share
          - per-regime covariance contribution
        - and must not stop at one pooled token-quality number

- 2026-04-11: Four-group cross-suite regime summary now exists and changes the root-cause framing.
  - summary artifact:
    - `/home/chen/RLPFN/artifacts/phase3_cross_suite_regime_summary.json`
  - summary entrypoint:
    - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_cross_suite_regime_summary.py`
  - tests:
    - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_phase3_cross_suite_regime_summary.py`
  - groups covered:
    - `pair1`
    - `pair2`
    - `seed24680`
    - `seed13579`
  - locked cross-group findings:
    - `dist1_to_4_reset_tracking_consistent = true`
      - near-terminal `post_reset_tail_dist_1..4` tracks reset density in all four groups
    - `dist1_to_4_pre_alignment_consistent = false`
      - the same near-terminal mass does **not** have a stable relation to `pre_vs_zero_suffix_gap`
    - `post_reset_tail_ablation_pre_improvement_consistent = false`
      - semantic ablation effects on `pre-gap` are not stable across groups
  - regime split:
    - positive regime:
      - suites where suite-level `ppo_minus_zero_suffix > 0`
      - current members:
        - `pair1`
        - `pair2`
        - `seed13579`
      - group mean:
        - `dist1_to_4_mean_corr_reset = +0.6281`
        - `dist1_to_4_mean_corr_pre = -0.1402`
    - nonpositive regime:
      - suites where suite-level `ppo_minus_zero_suffix <= 0`
      - current member:
        - `seed24680`
      - group mean:
        - `dist1_to_4_mean_corr_reset = +0.5817`
        - `dist1_to_4_mean_corr_pre = +0.0493`
  - root-cause implication:
    - the stable bottleneck is **not** “post-reset tail always hurts generalization”
    - the stable bottleneck is:
      - the objective reliably learns/reset-tracks reset-heavy near-terminal mass
      - but the mapping from that mass to actual heldout `PPO-vs-zero` gain is regime-dependent
    - therefore the current many-env fitting/optimization bottleneck is a **mixed-regime supervision problem**
      - reset-density features are fitted consistently
      - gain-sign/quality semantics are not consistent across random groups
    - this explains why:
      - token-bias probes can look strong
      - while cross-suite optimization conclusions keep flipping

- 2026-04-11: New suite `seed_13579` completed and behaves closer to pair2 than to `seed24680`.
  - new suite-matched baselines:
    - zero:
      - `/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_13579/zero_cross_env_control.json`
    - PPO:
      - `/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_13579/ppo_cross_env_baseline.json`
  - runtime:
    - zero baseline:
      - `ELAPSED_SEC = 282.22`
    - PPO baseline:
      - `ELAPSED_SEC = 1122.79`
    - note:
      - this matches the `seed24680` runtime scale
      - serial PPO baseline on fresh suites should be budgeted as a `10-20` minute task
  - baseline comparison:
    - heldout suffix:
      - zero `= -1.8247`
      - PPO `= -1.8100`
      - `ppo_minus_zero_suffix = +0.0147`
    - heldout full:
      - zero `= -34.2788`
      - PPO `= -33.9857`
      - `ppo_minus_zero_full = +0.2931`
  - distance probe:
    - `/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_13579/phase3_post_reset_tail_distance_probe.json`
    - key buckets:
      - dist1:
        - `corr_reset = +0.7062`
        - `corr_pre = -0.1986`
      - dist2:
        - `corr_reset = +0.3830`
        - `corr_pre = -0.0479`
      - dist4:
        - `corr_reset = +0.3830`
        - `corr_pre = -0.2826`
  - semantic contrast:
    - tail mask:
      - `/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_13579/phase3_terminal_tail_mask_probe.json`
    - remove `post_reset_terminal_tail`:
      - `/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_13579/phase3_post_reset_tail_ablation_verify.json`
      - zero effect:
        - `corr_reset_delta = 0.0`
        - `corr_pre_delta = 0.0`
    - remove `post_reset_only`:
      - `/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_13579/phase3_post_reset_ablation_verify.json`
      - tiny effect:
        - `corr_reset_delta = -0.0058`
        - `corr_pre_delta = +0.0012`
  - interpretation:
    - this suite keeps the pair1/pair2 negative `corr_pre` direction
    - but the semantic ablation leverage is much weaker than pair1
    - therefore:
      - even inside the positive regime, the same token family is not equally actionable across groups

- 2026-04-11: New heldout suite `seed_24680` completed and it weakens the current mechanism claim.
  - new suite-matched baselines:
    - zero:
      - `/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_24680/zero_cross_env_control.json`
    - PPO:
      - `/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_24680/ppo_cross_env_baseline.json`
  - runtime note:
    - cross-suite baseline commands must include:
      - `--no-action-reachability-probe`
    - reason:
      - action-reachability is irrelevant to the baseline reuse contract here
      - leaving it on adds extra serial rollout cost without changing the reused heldout/train return metrics
    - observed runtime after removing it:
      - zero baseline still took roughly `5` minutes
      - PPO baseline took `ELAPSED_SEC = 1176.22`
    - interpretation:
      - this is a runtime-control issue, not a compare-contract bug
      - serial PPO exact-SCM evaluation remains expensive on fresh suites
  - baseline comparison result on `seed_24680`:
    - heldout suffix return:
      - zero `= 2.2873`
      - PPO `= 2.2165`
      - `ppo_minus_zero_suffix = -0.0708`
    - heldout full return:
      - zero `= 41.1848`
      - PPO `= 39.9727`
      - `ppo_minus_zero_full = -1.2121`
  - implication:
    - unlike pair1 and pair2, this suite does not preserve a positive global heldout `PPO-vs-zero` checkpoint delta
    - therefore `pre_vs_zero_suffix_gap` is not a uniform “goodness” label across all suites unless the suite-level sign is checked first

- 2026-04-11: Cross-suite reproduction on `seed_24680` does not preserve the pair1/pair2 pre-gap direction.
  - artifact:
    - `/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_24680/phase3_post_reset_tail_distance_probe.json`
  - stable part that remains:
    - `post_reset_tail_dist_1..4` are still positively correlated with reset count:
      - dist1 `corr_reset = +0.5818`
      - dist2 `corr_reset = +0.5702`
      - dist3 `corr_reset = +0.6031`
      - dist4 `corr_reset = +0.5715`
  - unstable part:
    - the same buckets now have **positive**, not negative, correlation with `pre_vs_zero_suffix_gap`:
      - dist1 `corr_pre = +0.0538`
      - dist2 `corr_pre = +0.0466`
      - dist3 `corr_pre = +0.0507`
      - dist4 `corr_pre = +0.0460`
  - consequence:
    - the pair1/pair2 statement:
      - “near-terminal post-reset tail is anti-aligned with heldout pre-gap”
      - is not suite-invariant
    - the currently stable cross-suite claim is narrower:
      - post-reset tail mass tends to track reset density
    - but whether it hurts or helps `pre_vs_zero_suffix_gap` is currently suite-dependent

- 2026-04-11: Semantic contrast on `seed_24680` shows reset-bias and pre-gap alignment are decoupled.
  - tail-mask artifact:
    - `/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_24680/phase3_terminal_tail_mask_probe.json`
  - no-rerun semantic contrasts:
    - remove `post_reset_terminal_tail`:
      - `/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_24680/phase3_post_reset_tail_ablation_verify.json`
    - remove `post_reset_only`:
      - `/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_24680/phase3_post_reset_ablation_verify.json`
  - results:
    - base:
      - `corr_reset = +0.7264`
      - `corr_pre = -0.2657`
    - remove `post_reset_terminal_tail`:
      - `corr_reset_delta = -0.0635`
      - `corr_pre_delta = -0.0374`
    - remove `post_reset_only`:
      - `corr_reset_delta = -0.0904`
      - `corr_pre_delta = -0.0460`
  - interpretation:
    - removing post-reset mass still reduces reset-bias
    - but on this suite it makes `pre-gap` alignment **worse**, not better
    - therefore:
      - reset-bias and pre-gap alignment are no longer the same axis
      - the current root-cause statement cannot remain:
        - “post-reset tail bias is the direct cause of poor heldout pre-gap”
      - the mechanism must now be split into at least:
        - reset-density tracking
        - suite-level `PPO-vs-zero` regime/sign

- 2026-04-10: Cross-suite probe contract corrected before further generalization claims.
  - confirmed compare-contract bug:
    - `/home/chen/RLPFN/artifacts/phase3_cross_suite_probe_entrypoints.md` was still telling later suites to reuse:
      - `/home/chen/RLPFN/artifacts/phase3_env_delta_concentration_probe_max4.json`
    - that artifact is pair1-specific and must not be reused for other heldout suites
  - fix landed in code:
    - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_subset_delta_fallback.py`
    - probes now derive heldout per-env `pre_vs_zero_suffix_gap` directly from suite-matched:
      - zero baseline JSON
      - PPO baseline JSON
    - this path is used when `reuse_heldout_subset_delta_json` is omitted
  - tests:
    - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_phase3_subset_delta_fallback.py`
  - operational rule:
    - for any suite other than canonical pair1, do not reuse pair1 heldout-delta artifacts
    - only reuse zero/PPO baselines that were generated on the same fixed suite

- 2026-04-10: Objective terminal-reset count semantics were inconsistent and are now fixed.
  - confirmed diagnostic bug:
    - some heldout rows could report:
      - `objective_terminal_reset_count = 0`
      - while still carrying nonzero `post_reset_*` mass
  - root cause:
    - `objective_terminal_reset_count` had been counted from raw `episode_starts & objective_mask`
    - but `post_reset_*` segmentation was computed from compressed objective-token segments
    - those are not the same contract once objective masking removes tokens
  - fix landed in:
    - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_heldout_reset_semantics_probe.py`
    - new helper:
      - `_objective_terminal_reset_count_from_segments(...)`
  - probes updated to use the same segment definition:
    - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_terminal_tail_mask_probe.py`
    - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_post_reset_tail_distance_probe.py`
    - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_post_reset_tail_episode_probe.py`
    - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_post_reset_tail_dist4_late_ablation_verify.py`
    - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_post_reset_tail_dist2_late_ablation_verify.py`
    - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_heldout_layered_mass_probe.py`
    - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_tail_bias_propagation_probe.py`
    - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_heldout_segment_ablation_probe.py`
  - tests:
    - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_phase3_heldout_reset_semantics_probe.py`
  - post-fix verification on pair2:
    - `/home/chen/RLPFN/artifacts/phase3_post_reset_tail_distance_probe_pair2.json`
    - no rows remain with:
      - `objective_terminal_reset_count = 0`
      - and nonzero `post_reset_*` mass
  - boundary:
    - this is a Phase 3 diagnostic-metric fix
    - it does not change the trusted Phase 2 milestone contract

- 2026-04-10: Cross-suite reproduction now has one stable result and one unstable result.
  - verified reference run:
    - distance probe:
      - `/home/chen/RLPFN/artifacts/phase3_post_reset_tail_distance_probe_pair2.json`
    - targeted ablation:
      - `/home/chen/RLPFN/artifacts/phase3_post_reset_tail_dist4_late_ablation_pair2.json`
  - pair2 reproduces the near-terminal post-reset tail bias:
    - `post_reset_tail_dist_1`:
      - `corr_vs_terminal_resets = +0.6259`
      - `corr_vs_pre_suffix_gap = -0.0869`
    - `post_reset_tail_dist_2`:
      - `corr_vs_terminal_resets = +0.5606`
      - `corr_vs_pre_suffix_gap = -0.0706`
    - `post_reset_tail_dist_3`:
      - `corr_vs_terminal_resets = +0.6240`
      - `corr_vs_pre_suffix_gap = -0.0592`
    - `post_reset_tail_dist_4`:
      - `corr_vs_terminal_resets = +0.6121`
      - `corr_vs_pre_suffix_gap = -0.0875`
  - pair2 does not reproduce every broader aggregate sign pattern:
    - `post_reset_late_k4` on pair1:
      - `corr_vs_terminal_resets = +0.8463`
      - `corr_vs_pre_suffix_gap = -0.2638`
    - `post_reset_late_k4` on pair2:
      - `corr_vs_terminal_resets = +0.5733`
      - `corr_vs_pre_suffix_gap = +0.5028`
  - stable compare signal:
    - the narrower `post-reset tail dist 1..4` slice remains reset-biased across pair1 and pair2
    - the broader `late-step` aggregation is not suite-invariant enough to promote as the root mechanism by itself
  - strongest reproduced counterfactual so far:
    - pair1 `dist<=4 & late-step` ablation:
      - `corr_reset_delta = -0.0633`
      - `corr_pre_delta = +0.0309`
    - pair2 `dist<=4 & late-step` ablation:
      - `corr_reset_delta = -0.0608`
      - `corr_pre_delta = +0.0289`
  - current interpretation:
    - the reproducible mechanism is not “all late-step mass”
    - the reproducible mechanism is “near-terminal post-reset tail mass”
    - next work should stay on:
      - cross-suite reproduction with suite-matched baselines
      - semantic contrast of the reset/tail rules
    - do not return to distance-threshold search as the main line of evidence

- 2026-04-10: Trusted semantic source narrowed further.
  - do not infer the current trusted lineage from:
    - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/rlpfn_maintained_path.py`
  - reason:
    - the current milestone checkpoint does not exactly match those maintained defaults
    - verified mismatch:
      - maintained defaults say `reinforce_action_transform = "clip"`
      - current trusted checkpoint config has `reinforce_action_transform = "none"`
  - implication:
    - for current work, the only trusted semantic source is:
      - the maintained Phase 2 pack
      - the milestone checkpoint config actually loaded from disk
      - the fixed-suite artifacts actually used by Phase 3

- 2026-04-10: Phase 3 semantic contract checked against code and fixed suites.
  - verified from checkpoint config and suite `h_list`:
    - `family = "scm"`
    - `strict_joint_transition_enabled = true`
    - `state_full_rms_enabled = true`
    - `ctrl_reward_enable_prob = 0.7`
    - `reinforce_reward_transform = "tanh"`
    - `next_state_flow_matching_weight = 0.2`
    - `normalized_q_value_weight = 0.0`
    - `terminal_reset_enabled = true`
    - `reward_dropout_enabled = true`
    - `x_encoder_type = "split_obs_action"`
    - `single_eval_causal = true`
    - `backbone = "rwkv7"`
  - verified from suite contents:
    - train and heldout suites are both fixed batches of `16` environments
    - both suites are exact-SCM in the maintained-project sense:
      - `family = "scm"`
      - `strict_joint_transition_enabled = true`
    - train and heldout suites are different fixed batches:
      - train `suite_seed = 12345`
      - heldout `suite_seed = 67890`
  - verified from audit code:
    - Phase 3 optimization audit uses:
      - `core_a = false`
      - `reference_semantics_enabled = false`
    - so the current Phase 3 diagnostic path is not secretly using the simplified Core-A environment overrides
  - important nuance:
    - evaluation uses a new batch relative to training, but not a freshly re-sampled batch on every call
    - it re-initializes the same fixed heldout SCM suite for reproducibility
  - interpretation:
    - the current Phase 3 path is still many-env exact-SCM and not simplified along the checked reward/state-control axes
    - but trusted conclusions remain Phase-2-only until a new Phase 3 trust baseline is established

- 2026-04-10: Trust boundary narrowed further.
  - current Phase 3 longrun artifacts are no longer treated as trusted semantic baselines
  - current Phase 3 checkpoint and optimization readouts are retained only as diagnostics
  - the only currently trusted branch is:
    - `/home/chen/RLPFN/artifacts/critic_help_vs_zero_state_reset_shared_seed12345_strict_fixed_env_deterministic_quick.json`
  - implication:
    - Phase 3 work may continue, but all Phase 3 conclusions are exploratory until a new Phase 3 trust baseline is explicitly re-established
    - Phase 2 remains the only hard semantic gate
  - protection assets updated:
    - `/home/chen/RLPFN/artifacts/phase3_regression_guardrail.md`
    - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_regression_manifest.json`
    - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_regression_suite.py`

- 2026-04-10: Phase 2 green preflight was promoted from documentation to code.
  - Phase 3 entrypoints now require a green Phase 2 pack before they will run:
    - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_regression_suite.py`
    - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_multi_env_optimization_audit.py`
  - green means:
    - `validation.all_checks_pass = true`
    - `pass_flags.official_vs_monkey_numeric_match = true`
    - `pass_flags.shared_learned_beats_zero = true`
    - `pass_flags.shared_learned_post_return_positive = true`
  - implication:
    - no new Phase 3 result is acceptable unless it is explicitly built on the maintained Phase 2 shared actor-critic milestone branch

- 2026-04-10: Phase 2 was frozen for exploration and converted into a maintained regression gate.
  - guardrail doc:
    - `/home/chen/RLPFN/artifacts/phase2_regression_guardrail.md`
  - rerun entrypoint:
    - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase2_regression_suite.py`
  - assembled pack summary:
    - `/home/chen/RLPFN/artifacts/phase2_regression_pack/phase2_regression_suite_summary.json`
  - implication for Phase 3:
    - do not reopen Phase 2 semantic work during Phase 3 unless the maintained regression pack fails first
    - any Phase 3 training-path change must preserve the trusted Phase 2 contract before new optimization conclusions are accepted

- 2026-04-10: Phase 3 contract wording corrected against code.
  - the real training regime is:
    - batch of many fixed environments
    - each environment contributes one fixed `n_samples` rollout
    - each rollout may contain multiple internal episodes
  - this was confirmed from:
    - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_multi_env_optimization_audit.py`
    - `n_envs = train_suite["batch_size"]`
    - `n_steps = n_samples`
  - consequence:
    - the optimization bottleneck should be interpreted as low exploit-density and cross-env heterogeneity inside fixed-length rollouts, not as a one-episode-per-env regime

- 2026-04-10: Current cross-environment generalization check re-read under the corrected contract.
  - fixed-suite checkpoint baselines still show heldout generalization over zero:
    - pair 1:
      - train full gap `= -12.3896`
      - train suffix gap `= -0.5170`
      - heldout full gap `= +59.0105`
      - heldout suffix gap `= +2.8123`
    - pair 2:
      - train full gap `= +66.8087`
      - train suffix gap `= +3.2047`
      - heldout full gap `= +26.1680`
      - heldout suffix gap `= +1.5458`
  - trusted pair1 continued-training baseline remains weak on optimization:
    - `/home/chen/RLPFN/artifacts/phase3_multi_env_trusted_pair1.json`
    - one PPO outer epoch changes:
      - heldout suffix delta `= +0.0191`
      - heldout full-return delta `= -0.2745`
      - train suffix delta `= -0.0794`
      - train full-return delta `= -1.0401`
  - reading:
    - cross-environment semantics are present at checkpoint level
    - current many-env PPO optimization still does not give a clean heldout-improving update under the trusted baseline

- 2026-04-10: Phase 3 regression guardrail pack added.
  - guardrail doc:
    - `/home/chen/RLPFN/artifacts/phase3_regression_guardrail.md`
  - manifest:
    - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_regression_manifest.json`
  - regression entrypoint:
    - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_regression_suite.py`
  - assembled summary:
    - `/home/chen/RLPFN/artifacts/phase3_regression_pack/phase3_regression_suite_summary.json`
  - validation:
    - `all_checks_pass = true`
    - but this now means pack integrity only, not semantic trust
  - protected records:
    - canonical Phase 3 artifacts are fingerprinted and preserved
    - fixed-suite checkpoint cross-env generalization numbers are preserved as diagnostics
    - pair1 optimization baseline is preserved as a diagnostic snapshot
  - operational rule:
    - Phase 2 regression pack is the only hard trust gate
    - Phase 3 regression pack only prevents silent artifact drift and duplicate stale reuse

- 2026-04-09: New Phase 3 optimization ledger created after rejecting the checkpoint-only Phase 3 baseline as insufficient for optimization debugging.
- 2026-04-09: New audit driver created at `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_multi_env_optimization_audit.py`.
- 2026-04-09: Driver import path fixed by adding the missing `evaluate_prior_suite` import.
- 2026-04-09: Static validation passed:
  - `py_compile` passed
  - module import passed
- 2026-04-09: Driver default profile corrected:
  - previous baseline semantics were wrong for the current objective
  - the driver now defaults to `trusted_sep_reset_mainline`
  - old `zero / no-critic / no-aux / no-SEP-reset` behavior is retained only as `legacy_actor_only_probe`
- 2026-04-09: Historical probe re-interpretation:
  - `/home/chen/RLPFN/artifacts/phase3_multi_env_optimization_pair1_smoke.json`
  - `/home/chen/RLPFN/artifacts/phase3_multi_env_optimization_pair1_sep_reset.json`
  - these two artifacts are now treated only as historical `legacy_actor_only_probe` references
  - they are no longer the primary Phase 3 baseline
- 2026-04-09: Trusted baseline launched on pair1:
  - checkpoint: `/home/chen/RLPFN/rwkv/models_diff/rlpfn_03_30_2026_22_37_53_epoch_13.cpkt`
  - train profile: `trusted_sep_reset_mainline`
  - train suite: `/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/train_suite.pt`
  - heldout suite: `/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/heldout_suite.pt`
  - `n_samples = 2048`
  - `single_eval_pos = 1946`
  - `outer_epochs = 1`
  - `rollout_backend = serial`
  - output target: `/home/chen/RLPFN/artifacts/phase3_multi_env_trusted_pair1.json`
  - status at ledger update: running
- 2026-04-09: First fixed-contract smoke launched on pair1:
  - checkpoint: `/home/chen/RLPFN/rwkv/models_diff/rlpfn_03_30_2026_22_37_53_epoch_13.cpkt`
  - train suite: `/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/train_suite.pt`
  - heldout suite: `/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/heldout_suite.pt`
  - `n_samples = 2048`
  - `single_eval_pos = 1946`
  - `outer_epochs = 1`
  - `n_epochs = 4`
  - `batch_size = 2048`
  - `rollout_backend = serial`
  - output target: `/home/chen/RLPFN/artifacts/phase3_multi_env_optimization_pair1_smoke.json`
  - status at ledger update: running
- 2026-04-09: Runtime check for the first pair1 smoke:
  - process remained alive in `R` state for > 15 minutes
  - CPU remained active at ~100%
  - result file had not been emitted yet
  - no crash signature observed
  - current interpretation: the first fixed-contract smoke is valid but more expensive than expected, so future first-pass comparisons may need a reduced-cost smoke before the full `2048 x pair eval x pre/post` run.
- 2026-04-09: First pair1 smoke completed:
  - artifact: `/home/chen/RLPFN/artifacts/phase3_multi_env_optimization_pair1_smoke.json`
  - `pre_train_vs_zero`:
    - `full_return_gap = -0.15463924407958984`
    - `suffix_return_gap = -0.00708240270614624`
  - `post_train_vs_zero`:
    - `full_return_gap = -1.8246030807495117`
    - `suffix_return_gap = -0.08399713039398193`
  - `pre_heldout_vs_zero`:
    - `full_return_gap = -0.19002532958984375`
    - `suffix_return_gap = -0.007953405380249023`
  - `post_heldout_vs_zero`:
    - `full_return_gap = -1.6575956344604492`
    - `suffix_return_gap = -0.07213851809501648`
  - pre/post delta:
    - `train_full_return_delta = -1.6699638366699219`
    - `train_suffix_return_delta = -0.0769147276878357`
    - `heldout_full_return_delta = -1.4675703048706055`
    - `heldout_suffix_return_delta = -0.06418511271476746`
  - train-history summary:
    - `outer_epoch = 1`
    - `critic_raw_corr = 0.013844179920852184`
    - `critic_explained_variance_raw = -1.1447815895080566`
  - interpretation:
    - current Phase 3 baseline training semantics do not improve heldout performance
    - under the correct many-env training/testing contract, a single PPO training step makes both train and heldout gaps more negative
  - status after driver correction:
    - this result should be treated as a historical probe under the now-rejected `legacy_actor_only_probe` semantics
    - it should not be used as the primary Phase 3 baseline any more

## Immediate Decision

- Stop treating the old actor-only/no-aux probe as the main Phase 3 baseline.
- Re-base Phase 3 on the trusted mainline training profile and only then re-evaluate bottlenecks.

## Next Optimization Axis

The next comparisons should stay under the same fixed-suite Phase 3 contract and change **one** training semantic at a time, but only **after** re-basing to the trusted mainline profile.

Current active work:

1. rebuild the pair1 baseline under `trusted_sep_reset_mainline`
2. compare pre/post `train` and `heldout` gaps against zero under that trusted baseline
3. only after that, decide whether a new bottleneck remains

## Root-Cause-First Bottleneck Readout

- 2026-04-09: Re-read the Phase 3 ledgers, fixed-suite artifacts, and the long-run legacy log before proposing any further training change.

What is already exposed by evidence:

- The dominant Phase 3 blocker is **not** "the model cannot represent cross-environment semantics at all".
  - Fixed-suite checkpoint evaluation already shows repeatable heldout advantage over `zero`.
  - Pair 1, epoch-13 checkpoint:
    - heldout suffix delta vs zero: `+2.8123`
  - Pair 2, epoch-13 checkpoint:
    - heldout suffix delta vs zero: `+1.5458`
  - Neighbor checkpoint epoch-15 preserves the same direction:
    - pair 1 heldout suffix mean: `0.2338 -> 0.2466`
    - pair 2 heldout suffix mean: `-1.8627 -> -1.8257`
  - Therefore the current model lineage already contains usable heldout semantics.

- The dominant blocker is also **not** "action cannot causally affect reward in these suites".
  - Both pair 1 and pair 2 fixed suites have strong action reachability on the exploit suffix.
  - Pair 1:
    - `suffix_return_delta_abs_mean ~= 36129.97`
  - Pair 2:
    - `suffix_return_delta_abs_mean ~= 33774.97`
  - So Phase 3 is not currently blocked by missing action-to-reward connectivity.

- The strongest current bottleneck candidate is the **optimization contract itself** under the fixed many-env regime:
  - `n_samples = 2048`
  - `single_eval_pos = 1946`
  - suffix horizon `= 102`
  - `train_batch_size = 16`
  - `unique_structures = 16`
  - one rollout per environment
  - therefore one outer epoch exposes only `16 * 102 = 1632` exploit tokens total, while every env in the batch is structurally distinct.
  - This is a very low semantic-density update regime relative to the observed return/action heteroskedasticity.

- The train-side metrics are already consistent with a heteroskedastic multi-env optimization bottleneck rather than a pure generalization bottleneck.
  - Pair 1 checkpoint baseline:
    - heldout beats `zero`
    - train loses to `zero`
  - Pair 2 checkpoint baseline:
    - heldout beats `zero`
    - train also beats `zero`
  - This mixed train-side story, together with repeated heldout wins, is more consistent with unstable suite-level optimization density / weighting than with "no semantics learned".

- Historical optimization probes under the short-run aligned profile also point to objective-side instability rather than missing semantics.
  - `/home/chen/RLPFN/artifacts/phase3_multi_env_optimization_pair1_sep_reset.json`
  - Under one PPO training epoch on the fixed pair-1 train suite:
    - `critic_raw_corr ~= -0.00076`
    - `critic_explained_variance_raw ~= -0.5226`
    - train suffix delta `= -0.2851`
    - heldout suffix delta `= -0.02135`
  - This indicates that continued optimization can immediately degrade both train and heldout under a low-density update contract.

- The legacy long-run actor-only log remains useful as a symptom trace, not as a baseline.
  - `/home/chen/RLPFN/artifacts/phase3_longrun_epoch13_shortrun_aligned/log/rlpfn_ppo_actor_baseline_modezero_ppo_normalize_advantageFalse_ppo_reset_env_state_at_sepFalse_ppo_runtime_next_state_flow_matching_weight_override0_ppo_runtime_normalized_q_value_weight_override0_ppo_vf_coef0_reinforce_ac_h7cda225936.log`
  - It shows that PPO can temporarily fall into a degenerate short-episode attractor (`ep_len_mean ~= 3.94` at epoch 4) without improving full-return semantics.
  - This is not the trusted Phase 3 baseline, but it does expose the class of failure: optimization can find shortcut behavior long before semantic exploitation is stably improved.

Current root-cause-first interpretation:

- The big head bottleneck is currently most likely:
  - **too little exploit supervision per structurally unique environment**
  - under **very high per-env return / sensitivity heterogeneity**
  - inside a **many-env PPO update that mixes all envs together immediately**

- In other words, Phase 3 does not currently look bottlenecked by absent representation.
- It looks bottlenecked by a poor semantic signal-to-noise ratio at the update interface.

What should be probed next before any further modification:

1. Per-environment update concentration on the fixed suites.
   - Measure how much actor positive-mass / advantage mass / gradient mass comes from each of the 16 train-suite envs.
   - Goal:
     - test whether a few outlier envs dominate the whole PPO step.

2. Fixed-suite semantic-use probe, not a training tweak.
   - On the same pair-1 and pair-2 suites, run:
     - `ctx`
     - `no_ctx`
     - `shuf_ctx`
   - Goal:
     - distinguish "semantic context is present but optimization wastes it"
     - from "heldout win is only a warmed-history artifact".

3. Suffix-density sensitivity under the same suites.
   - Keep suites fixed.
   - Change only the optimization contract that determines exploit-token density.
   - Goal:
     - test whether the current failure tracks available exploit supervision per env more strongly than it tracks checkpoint identity.

Until these three probes are done, no new Phase 3 training fix should be promoted.

Deferred until the trusted baseline is re-established:

- critic removal / restoration probes
- aux removal / restoration probes
- `SEP reset` as a one-variable compare, because it is already part of the trusted baseline
- 2026-04-09: Extended runtime observation for the `SEP reset` compare:
  - process remained alive in `R` state for > 40 minutes
  - CPU remained active at ~100%
  - result file had still not been emitted
  - baseline pair1 smoke completed in materially less time
  - initial observation only:
    - the full `2048` run had a much larger wall time
    - the reward-side conclusion was still pending because the run had not completed

- 2026-04-09: Slowdown diagnosis for `ppo_reset_env_state_at_sep=True`:
  - code-path inspection:
    - in `EnvironmentPriorPPOBatchVecEnv.step_wait()`, `SEP reset` only does a one-step state replacement at the boundary
    - it does **not** mark `done`
    - it does **not** trigger `_full_reset_batch()`
    - it does **not** create repeated episode restarts
  - reduced training-only benchmark (`n_steps = 512`, one outer epoch, same fixed train suite):
    - baseline `train_sec = 37.2434`
    - `SEP reset` `train_sec = 34.9187`
  - reduced post-eval-only benchmark (`n_samples = 32`, same fixed train/heldout suites after one training epoch):
    - baseline `post_eval_sec = 42.9482`
    - `SEP reset` `post_eval_sec = 43.0484`
  - hard conclusion:
    - the observed large wall-time increase in the full `2048` run is **not** explained by the direct `SEP reset` code path
    - `SEP reset` does not introduce a meaningful per-step slowdown in reduced train-only or post-eval-only benchmarks
    - therefore runtime alone is not a valid rejection criterion for this axis

- 2026-04-10: Phase-3 regression contract hardened; `shortrun` quarantined.
  - code:
    - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_longrun_gradient_compare.py`
    - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_regression_suite.py`
  - changes:
    - longrun gradient regression now uses official strict fixed-env seed channels
    - deterministic actor sampling is enabled for the trusted longrun regression contract
    - deterministic batch plan remains enabled
    - regression-pack summary reuse now fingerprints source artifacts, so stale summaries are rejected automatically
  - stable trusted artifact:
    - `/home/chen/RLPFN/artifacts/phase3_longrun_gradient_compare.json`
    - repeated strict reruns matched on:
      - `resume_grad_norm = 8.315512657165527`
      - `phase3_isolated_grad_norm = 8.315511703491211`
      - `l2_delta_norm = 3.819410994765349e-05`
      - runtime remained in the same band (`collect_wall_s ~= 48-53s`, `backward_wall_s ~= 0.20-0.27s`)
  - unresolved anomaly:
    - `/home/chen/RLPFN/artifacts/phase3_shortrun_regression_instability_summary.json`
    - `/home/chen/RLPFN/artifacts/phase3_shortrun_sequence_layout_instability_probe.json`
    - even under the same strict seed contract, repeated `shortrun_profile_gradient_regression` runs produced materially different absolute gradients
    - observed driver grad norms included:
      - `677.8103`
      - `6.3286`
      - `17.2207`
    - enabling `torch.use_deterministic_algorithms(True)` did not remove the split:
      - deterministic runs still landed in both the `677.8103` and `6.3286` regimes
    - within each run, driver/reference still matched closely
  - narrowing evidence for the shortrun anomaly:
    - PPO head initialization hashes were stable across repeated builds
    - env seed and rollout seed channels were both present and fixed
    - objective-valid `obs` and `old_log_prob` could remain identical while `actions/returns/advantages` still drifted
    - this points away from environment-instance drift and toward unresolved recurrent rollout / sequence-layout instability in the shortrun regression path
  - decision:
    - keep longrun regression in the trusted Phase-3 pack
    - quarantine shortrun profile regression from pack pass/fail
    - do not use shortrun gradient magnitudes as a baseline for future comparisons until the sequence-layout instability is resolved

- 2026-04-10: Phase-3 many-env optimization compare-contract bug identified.
  - root cause:
    - `phase3_multi_env_optimization_audit.py` was building PPO via `build_recurrent_ppo()` from the checkpoint backbone but **not** restoring the saved checkpoint validation PPO policy state
    - as a result, historical `pre` evaluation used a fresh-built PPO policy while checkpoint baselines used the saved validation PPO policy
  - evidence:
    - historical artifact:
      - `/home/chen/RLPFN/artifacts/phase3_multi_env_trusted_pair1.json`
    - canonical checkpoint baseline:
      - `/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/ppo_cross_env_baseline.json`
      - `/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/zero_cross_env_control.json`
    - restore probe:
      - `/home/chen/RLPFN/artifacts/phase3_restore_validation_policy_state_probe.json`
      - `restore_validation_policy_state=False`:
        - max abs diff vs saved PPO validation state `= 13.5944`
      - `restore_validation_policy_state=True`:
        - max abs diff vs saved PPO validation state `= 0.0`
    - compare summary:
      - `/home/chen/RLPFN/artifacts/phase3_policy_restore_root_cause_summary.json`
      - old pair1 `pre_heldout_vs_zero.suffix_return_gap = 0.000276`
      - canonical pair1 `heldout_vs_zero.suffix_return_gap = 2.812291`
  - code changes:
    - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/sb3_recurrent_ppo.py`
      - new explicit `restore_validation_policy_state` support in `build_recurrent_ppo()`
    - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_multi_env_optimization_audit.py`
      - now enables `restore_validation_policy_state=True`
    - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_longrun_gradient_compare.py`
      - now enables `restore_validation_policy_state=True`
  - protection:
    - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_sb3_recurrent_ppo.py`
      - restore-state regression added
  - decision:
    - historical `phase3_multi_env_trusted_pair1.json` is now legacy/quarantined
    - do not use old pair1 many-env optimization deltas to claim a bottleneck
    - before any new Phase 3 bottleneck analysis, rerun pair1 from the restored-policy path or keep conclusions at the compare-contract level only
  - scope:
    - does **not** invalidate the Phase 2 trusted branch
    - does **not** invalidate checkpoint-level fixed-suite cross-environment baselines from `prior_generalization_audit(policy_mode='ppo')`
    - because those baselines use the saved validation PPO policy state directly
    - it **does** invalidate historical Phase 3 optimization artifacts that built a fresh PPO policy and then used that fresh policy for `pre` evaluation
  - runtime note:
    - a full rerun attempt with:
      - `n_samples = 2048`
      - `outer_epochs = 1`
      - `rollout_backend = serial`
      - restored-policy path enabled
    - remained active for multiple minutes without emitting an output artifact in the current interactive budget and was terminated
    - therefore:
      - the restored-policy fix is code-complete and numerically protected at the policy-state level
      - but a new trusted Phase-3 many-env optimization baseline has **not** yet been re-established
      - runtime itself remains an open Phase-3 risk
# 2026-04-10: restored-policy pair1 baseline rebaseline attempt still blocked

- Phase 2 trust remains unaffected. The new issue is isolated to Phase 3 many-env optimization compare contract and runtime.
- The compare-contract bug around fresh-built PPO policy vs saved validation PPO state is fixed in code:
  - `build_recurrent_ppo(... restore_validation_policy_state=True)` is now wired into Phase 3 optimization entrypoints.
- I attempted to rebuild a corrected pair1 many-env optimization baseline.
  - `serial` pre/post policy evaluation remains a runtime blocker.
  - full rerun attempt:
    - artifact dir: `/home/chen/RLPFN/artifacts/phase3_pair1_restored_policy_rerun`
    - runtime: `377.64s`, interrupted during pre-train serial evaluation
    - no output artifact emitted
- I then enabled backend propagation so Phase 3 `pre/post` PPO evaluation follows the same `rollout_backend` argument used by zero-control.
  - code:
    - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/critic_free_single_env_audit.py`
    - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_multi_env_optimization_audit.py`
  - guard test:
    - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_phase3_multi_env_optimization_audit.py`
- I tested whether `family_vectorized` can replace `serial` for trusted baseline rebuilding.
  - small probe:
    - `/home/chen/RLPFN/artifacts/phase3_backend_equivalence_probe.json`
  - canonical probe:
    - `/home/chen/RLPFN/artifacts/phase3_backend_equivalence_probe_2048.json`
  - summary:
    - `/home/chen/RLPFN/artifacts/phase3_backend_divergence_summary.json`
- Result:
  - `family_vectorized` is not semantically equivalent enough to `serial` for restored-policy PPO pre/post evaluation.
  - canonical batch=1, `n_samples=2048`, `single_eval_pos=1946`:
    - serial full return mean: `-22.5120`
    - family_vectorized full return mean: `-28.7215`
    - abs diff: `6.2094`
    - serial suffix return mean: `-0.5082`
    - family_vectorized suffix return mean: `-0.5903`
    - abs diff: `0.0821`
    - runtime ratio `family/serial = 1.0697`
  - So `family_vectorized` is neither close enough nor faster enough to establish a new trusted Phase 3 optimization baseline.
- I also launched a full `family_vectorized` pair1 rerun and then terminated it after confirming it still did not emit an artifact in time:
  - `/home/chen/RLPFN/artifacts/phase3_pair1_restored_policy_family_vectorized/runtime.json`
  - `107.25s`, terminated, no output JSON produced

## Current Phase 3 trust state

- checkpoint-level fixed-suite cross-environment generalization remains valid
- historical `phase3_multi_env_trusted_pair1.json` remains quarantined
- a new trusted many-env optimization baseline still does **not** exist
- do **not** use `family_vectorized` pre/post policy evaluation to rebaseline Phase 3
- Phase 3 remains diagnostic-only until a serial-correct restored-policy optimization artifact is rebuilt or the backend divergence is root-caused

## 2026-04-10 monitored serial rerun: runtime blocker phase is now pinned down

- Added monitored rerun entrypoint:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_monitored_optimization_run.py`
- Purpose:
  - rerun restored-policy Phase 3 optimization without changing semantics
  - emit phase timing JSONL
  - emit periodic Python stack dumps
  - identify which phase dominates runtime before assuming a hang
- Monitored rerun artifact dir:
  - `/home/chen/RLPFN/artifacts/phase3_pair1_restored_policy_serial_monitored`
- Summary:
  - `/home/chen/RLPFN/artifacts/phase3_serial_monitored_runtime_summary.json`
- Observed runtime:
  - total interrupted wall time: `190.04s`
  - `zero_train` completed in `134.29s`
  - execution was still inside `zero_heldout` when interrupted
- Stack dumps:
  - `/home/chen/RLPFN/artifacts/phase3_pair1_restored_policy_serial_monitored/stack_dump.log`
  - all captured stacks remained inside serial suite reward collection:
    - `prior_generalization_audit._collect_suite_rewards_serial`
    - `environment_prior._rollout_single`
    - hotspots observed in:
      - `_pack_env_input`
      - `_apply_state_full_rms`
- Important implication:
  - the current serial restored-policy rerun is not first blocked by PPO `pre_train` evaluation
  - it is already dominated by **recomputing zero-control train/heldout suite evaluations**
  - therefore Phase 3 runtime blocker is now phase-pinned, not just “long overall”
- Trust consequence:
  - still no new trusted Phase 3 optimization baseline
  - but we now have a concrete handle on the runtime bottleneck and do not need to guess where the time goes

## 2026-04-10 zero-control reuse and pre-path equivalence

- I added strict contract-checked reuse of canonical pair1 artifacts into:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_multi_env_optimization_audit.py`
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_monitored_optimization_run.py`
- New supported inputs:
  - `--reuse-zero-control-json`
  - `--reuse-pre-policy-json`
- Reuse is only allowed when all of the following match:
  - suite fingerprint
  - `n_samples`
  - `single_eval_pos`
  - `rollout_backend`
  - `core_a=False`
  - `reference_semantics_enabled=False`
- Guard tests:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_phase3_multi_env_optimization_audit.py`
  - current relevant suite: `10 passed`

- I also measured whether current `pre_train` path is numerically equivalent to canonical checkpoint PPO baseline path:
  - `/home/chen/RLPFN/artifacts/phase3_pre_path_equivalence_probe.json`
- Result on batch=1, `n_samples=2048`, `single_eval_pos=1946`:
  - current pre path full return mean: `-22.5120316`
  - canonical PPO baseline path full return mean: `-22.5121231`
  - abs diff: `9.16e-05`
  - current pre path suffix return mean: `-0.5081826`
  - canonical PPO baseline path suffix return mean: `-0.5081862`
  - abs diff: `3.64e-06`
- This is close enough to justify canonical `pre` reuse under the same contract.

## 2026-04-10 reuse-zero + reuse-pre monitored rerun

- Monitored rerun artifact dir:
  - `/home/chen/RLPFN/artifacts/phase3_pair1_restored_policy_serial_reuse_zero_pre`
- Summary:
  - `/home/chen/RLPFN/artifacts/phase3_reuse_zero_pre_runtime_summary.json`
- Result:
  - zero-control recompute removed
  - pre-policy recompute removed
  - `train_loop` completed in `103.09s`
  - the next runtime blocker is post-train policy evaluation
- Note:
  - raw phase log for this run labels the first post-train policy eval as `pre_train`
  - stack location shows it is actually the post-train `_run_policy_eval` call in `phase3_multi_env_optimization_audit.py`
  - so the interpretation in `phase3_reuse_zero_pre_runtime_summary.json` is the authoritative one

## Updated current bottleneck order

1. zero-control serial recompute: solved by trusted reuse
2. pre-policy serial recompute: solved by trusted reuse
3. train_loop: runs and completes
4. post-train serial policy evaluation: current dominant blocker

## 2026-04-10 post-policy bundle split

- Added audit-only post-policy bundle support to:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_multi_env_optimization_audit.py`
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_monitored_optimization_run.py`
- New capabilities:
  - save train-loop-complete PPO policy state:
    - `--save-post-policy-bundle-path`
  - resume directly into post-train/post-heldout evaluation:
    - `--resume-post-policy-bundle-path`
- This does not change training semantics. It only removes duplicate recomputation when runtime control is the bottleneck.
- Guard coverage:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_phase3_multi_env_optimization_audit.py`
  - current relevant status: `12 passed`

- New saved bundle:
  - `/home/chen/RLPFN/artifacts/phase3_pair1_post_policy_bundle/phase3_pair1_post_policy_bundle.pt`
- Bundle monitor evidence:
  - `/home/chen/RLPFN/artifacts/phase3_pair1_post_policy_bundle/monitor_summary.json`
  - `train_loop` completed in `105.21s`
  - interruption occurred only after entering `post_train`

- Post-only resume evidence:
  - `/home/chen/RLPFN/artifacts/phase3_pair1_post_policy_only_pipe/monitor_summary.json`
  - `/home/chen/RLPFN/artifacts/phase3_pair1_post_policy_only_pipe/stack_dump.log`
  - resumed run entered `post_train` immediately and remained there for `235.67s` before interruption
  - stack dumps stayed inside serial exact-SCM rollout/policy-eval code:
    - `prior_generalization_audit._collect_suite_rewards_serial`
    - `environment_prior._rollout_single`
    - `sb3_recurrent_ppo._build_obs_from_rollout_step_inputs`
    - `environment_prior._apply_state_full_rms`

- Summary artifact:
  - `/home/chen/RLPFN/artifacts/phase3_post_policy_bundle_runtime_summary.json`

- New conclusion locked:
  - Phase 3 progress is now split cleanly into:
    - reusable zero-control baseline
    - reusable pre-policy checkpoint baseline
    - reusable train-loop-complete post-policy bundle
  - even after removing all duplicated earlier phases, a **single** `post_train` serial PPO evaluation remains runtime-dominant
  - therefore the next blocker is not repeated training; it is the cost of post-train serial exact-SCM evaluation itself

## 2026-04-10 post-train / post-heldout serial eval profiling

- Added audit-only post-eval profiler:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_post_eval_profile.py`
- Added profiled serial collector:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/prior_generalization_audit.py`
  - `collect_suite_rewards_serial_profiled(...)`
- Guard coverage:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_phase3_multi_env_optimization_audit.py`
  - current relevant status: `13 passed`

- Important bug fixed in the profiler path:
  - the first version of `collect_suite_rewards_serial_profiled(...)` accidentally omitted `torch.no_grad()`
  - this caused false OOM during profiling by retaining autograd graphs
  - the guard test now explicitly checks that serial profiled rollout runs with grad disabled

- Important runtime constraint locked:
  - parallel GPU profiler runs are not allowed
  - running train/heldout post-eval profilers concurrently caused OOM
  - profiling must be serialized to avoid contaminating runtime conclusions

- Train post-eval profile (`max_envs=4`):
  - `/home/chen/RLPFN/artifacts/phase3_post_train_profile_max4.json`
  - aggregate:
    - wall: `144.18s`
    - total rollout: `144.16s`
  - per-env rollout times:
    - `36.94s`
    - `35.87s`
    - `36.50s`
    - `34.84s`

- Heldout post-eval profile (`max_envs=4`):
  - `/home/chen/RLPFN/artifacts/phase3_post_heldout_profile_max4.json`
  - aggregate:
    - wall: `146.59s`
    - total rollout: `146.57s`
  - per-env rollout times:
    - `38.30s`
    - `36.46s`
    - `35.78s`
    - `36.03s`

- Summary:
  - `/home/chen/RLPFN/artifacts/phase3_post_eval_profile_summary.json`

- New conclusion locked:
  - both `post_train` and `post_heldout` serial eval are slow at the same order of magnitude
  - in the first 4 profiled envs, there is **no evidence** that runtime is dominated by one pathological environment
  - current runtime is better explained as a uniformly expensive serial exact-SCM rollout cost, not a small-env concentration bug

## 2026-04-10 single-env hotspot profile on post-eval path

- Added audit-only hotspot profiler:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_post_eval_hotspot_profile.py`
- Artifacts:
  - `/home/chen/RLPFN/artifacts/phase3_post_train_hotspot_env0.json`
  - `/home/chen/RLPFN/artifacts/phase3_post_heldout_hotspot_env0.json`
  - `/home/chen/RLPFN/artifacts/phase3_post_eval_hotspot_env0_summary.json`
- Duplicate-run guard:
  - both post-eval profiler scripts now reuse an existing `output_json` by default
  - rerun is allowed only with explicit `--overwrite`
  - do not rerun the same hotspot/profile artifact unless the contract actually changed

- Locked conclusions:
  - train env0 wall: `37.12s`
  - heldout env0 wall: `39.32s`
  - wall ratio `heldout/train = 1.059`
  - `_build_obs_from_rollout_step_inputs` is the largest named hotspot on train env0:
    - share of wall `12.70%`
  - `_apply_terminal_reset_step` is the largest named hotspot on heldout env0:
    - share of wall `13.72%`
  - terminal-reset path is material on both sides:
    - train env0: `_apply_terminal_reset_step + _terminal_tail_event_from_signal = 19.44%`
    - heldout env0: `25.00%`
  - `_apply_state_full_rms` is not a primary hotspot:
    - train env0 `1.71%`
    - heldout env0 `1.72%`
  - therefore the current serial exact-SCM blocker is better explained by:
    - observation-build cost
    - terminal-reset / terminal-tail bookkeeping
    - not by `state_full_rms`
    - and not by a single pathological env in the currently profiled sample

## 2026-04-10 single-env subpath profile inside build-obs and terminal-reset

- Added audit-only subpath profiler:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_post_eval_subpath_profile.py`
- Artifacts:
  - `/home/chen/RLPFN/artifacts/phase3_post_train_subpath_env0.json`
  - `/home/chen/RLPFN/artifacts/phase3_post_heldout_subpath_env0.json`
  - `/home/chen/RLPFN/artifacts/phase3_post_eval_subpath_env0_summary.json`
- Duplicate-run guard:
  - subpath profiler reuses an existing `output_json` by default
  - rerun is allowed only with explicit `--overwrite`

- Locked conclusions:
  - the profiler inherits the same trusted evaluation contract as current Phase 3 diagnostics:
    - shared actor-critic backbone
    - `ppo_reset_env_state_at_sep = True`
    - `single_eval_pos = 1946`
    - `rollout_backend = serial`
    - post-policy bundle restored from the trusted Phase 3 audit path
  - `build_obs.total` is stable across train/heldout env0:
    - train: `11.86%` of wall
    - heldout: `11.79%`
  - inside `build_obs`, the dominant subpath is not raw obs prefix copy:
    - `reward_mask_phase_terminal_write`
      - train: `7.03%`
      - heldout: `7.10%`
    - `action_tail_write`
      - train: `2.60%`
      - heldout: `2.58%`
    - `obs_prefix_copy`
      - train: `0.78%`
      - heldout: `0.75%`
  - `terminal_reset.total` is materially heavier on heldout env0:
    - train: `10.97%`
    - heldout: `13.56%`
  - inside terminal-reset, the dominant cost is the event path:
    - `terminal_event_call`
      - train: `8.44%`
      - heldout: `11.13%`
  - inside `terminal_tail_event_from_signal`, the expensive subpaths are boundary math, not final scatter:
    - train:
      - `lower_tail_boundary = 1.54%`
      - `upper_tail_boundary = 1.15%`
      - calls `850 / 850`
    - heldout:
      - `lower_tail_boundary = 3.16%`
      - `upper_tail_boundary = 2.38%`
      - calls `1790 / 1790`
  - therefore the current serial exact-SCM blocker is more precisely explained by:
    - `build_obs`: scalar slot writes plus masked action-tail writes
    - terminal path: tail boundary computations and event bookkeeping
    - not by `state_full_rms`
    - and not by duplicated zero/pre/train recomputation

## 2026-04-10 Phase 3 vs Phase 2 runtime ratio checkpoint

- Summary artifact:
  - `/home/chen/RLPFN/artifacts/phase3_vs_phase2_runtime_ratio_summary.json`

- Locked comparison basis:
  - Phase 2 trusted reference:
    - `/home/chen/RLPFN/artifacts/critic_official_vs_monkey_regression_seed12345.json`
    - strict fixed-env deterministic collect
  - Phase 3 collect reference:
    - `/home/chen/RLPFN/artifacts/phase3_longrun_gradient_compare.json`
  - Phase 3 post-eval reference:
    - `/home/chen/RLPFN/artifacts/phase3_post_train_subpath_env0.json`
    - `/home/chen/RLPFN/artifacts/phase3_post_heldout_subpath_env0.json`

- Locked runtime numbers:
  - Phase 2 trusted collect:
    - `5.3885s / 256 = 0.02105s per step`
  - Phase 3 many-env collect path:
    - isolated semantics: `48.1049s / 2048 = 0.02349s per step`
    - ratio vs Phase 2: `1.116x`
    - resumed semantics: `51.0717s / 2048 = 0.02494s per step`
    - ratio vs Phase 2: `1.185x`
  - Phase 3 single-env post-eval:
    - train env0: `35.6960s / 2048 = 0.01743s per step`
    - heldout env0: `36.4668s / 2048 = 0.01781s per step`
    - both are below the Phase 2 trusted per-step collect reference
  - Phase 3 full pair1 rebuild:
    - projected single post-train suite wall from env0 sample: `571.14s`
    - projected single post-heldout suite wall from env0 sample: `583.47s`
    - projected train+heldout post-eval wall: `1154.60s`

- Locked conclusion:
  - Phase 3 is **not** globally slow because each rollout step is catastrophically slower than Phase 2
  - current collect-path slowdown vs Phase 2 is moderate (`~1.12x` to `~1.18x`) and tolerable for targeted diagnostics
  - the real blocker is full-suite serial exact-SCM post-eval scaling, not per-step collapse
  - therefore Phase 3 cross-environment bottleneck tests may continue **only** in lightweight reuse-based form
  - do **not** use full pair1 rebuild as an inner-loop diagnostic until post-eval scaling is reduced or amortized

## 2026-04-10 lightweight many-env delta concentration probe

- Added diagnostic entrypoint:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_env_delta_concentration_probe.py`
- Duplicate-run guard:
  - reuse existing `output_json` by default
  - rerun only with explicit `--overwrite`
- Contract:
  - must pass Phase 2 preflight first
  - reuses canonical pair1 artifacts instead of rebuilding them:
    - zero-control baseline
    - checkpoint pre-policy baseline
    - saved post-policy bundle
  - current diagnostic run:
    - `max_envs = 4`
    - train env indices `[0, 1, 2, 3]`
    - heldout env indices `[0, 1, 2, 3]`

- Artifacts:
  - `/home/chen/RLPFN/artifacts/phase3_env_delta_concentration_probe_max4.json`
  - `/home/chen/RLPFN/artifacts/phase3_env_delta_concentration_summary_max4.json`

- Runtime:
  - train subset rollout: `142.40s`
  - heldout subset rollout: `141.71s`
  - combined: `284.12s`

- Locked conclusions:
  - this probe is light enough to run under the current diagnostic contract, but it is still too slow to use as a default inner loop
  - on the first 4 train envs:
    - suffix gap delta mean is positive: `+0.0419`
    - `3 / 4` envs are positive
    - update mass is concentrated:
      - top-1 abs-mass share `0.538`
      - top-2 abs-mass share `0.930`
  - on the first 4 heldout envs:
    - suffix gap delta mean is negative: `-0.1189`
    - `3 / 4` envs are nonpositive
    - concentration is even stronger:
      - top-1 abs-mass share `0.698`
      - top-2 abs-mass share `0.859`
  - current Phase 3 bottleneck reading under this lightweight contract is:
    - update benefit on the train side is narrow and concentrated
    - heldout-side response is weaker and mostly nonpositive in the tested subset
    - therefore checkpoint-level cross-environment semantics still exist, but the current many-env update is not stably amplifying heldout performance
  - this remains diagnostic-only and does **not** establish a new trusted Phase 3 baseline

## 2026-04-10 train-env token-quality contrast

- Added diagnostic entrypoint:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_train_env_quality_contrast_probe.py`
- Duplicate-run guard:
  - reuse existing `output_json` by default
  - rerun only with explicit `--overwrite`
- Contract:
  - reads only the existing canonical train-rollout quality artifact
  - does **not** recollect rollouts
  - does **not** rebuild zero/pre/post

- Artifacts:
  - `/home/chen/RLPFN/artifacts/phase3_train_env_quality_contrast_probe.json`
  - `/home/chen/RLPFN/artifacts/phase3_train_env_quality_contrast_regime_pair1_positive.json`

- Locked grouping:
  - subset mass median: `25.2618`
  - high-mass nonpositive envs: `[1]`
  - low-mass positive envs: `[0, 3]`
  - high-mass positive envs: `[2]`

- Locked regime context:
  - current train-quality fingerprint matches the trusted `pair1` train fingerprint
  - `suite_regime = positive`
  - `suite_regime_summary_membership_verified = true`
  - this is the positive-regime train-side contrast, not an unlabeled mixed-suite proxy

- Locked contrasts:
  - low-mass positive minus high-mass nonpositive:
    - `pre_suffix_gap = +5.8694`
    - `raw_value_return_corr = +0.0589`
    - `first16_positive_share = +0.1782`
    - `last16_positive_share = -0.1410`
    - `episode_count = 0.0`
    - `terminal_reset_count = 0.0`

- Locked conclusions:
  - high positive mass alone is not sufficient for positive train delta
  - compared with the high-mass/nonpositive case, the low-mass/positive cases are distinguished by:
    - substantially higher `pre_suffix_gap`
    - substantially higher `raw value-return corr`
    - more early-segment positive mass and less tail-segment positive mass
  - per-env objective episode segmentation is not the primary separator in this measured subset
  - current Phase 3 lightweight bottleneck reading is therefore:
    - the train-update difference is better explained by token-quality placement and critic alignment than by raw mass magnitude or episode count
  - in the current pair1-positive regime, the same separation remains visible on the train-side env items

## 2026-04-10 objective token-bucket probe

- Added diagnostic entrypoint:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_train_token_bucket_probe.py`
- Duplicate-run guard:
  - reuse existing `output_json` by default
  - rerun only with explicit `--overwrite`
- Contract:
  - same lightweight contract as the train-rollout quality probe
  - one restored-policy full-train-suite `collect_rollouts()`
  - no zero/pre/post rebuild

- Artifacts:
  - `/home/chen/RLPFN/artifacts/phase3_train_token_bucket_probe.json`
  - `/home/chen/RLPFN/artifacts/phase3_train_token_bucket_summary.json`

- Locked numbers:
  - runtime wall: `101.04s`
  - subset bucket alignment vs `suffix_gap_delta`:
    - `Q1 positive_mass_share corr = -0.6810`
    - `Q2 positive_mass_share corr = -0.6538`
    - `Q3 positive_mass_share corr = +0.8468`
    - `Q4 positive_mass_share corr = +0.6420`
  - low-mass positive minus high-mass nonpositive:
    - `Q1 positive_mass_share = -0.2990`
    - `Q2 positive_mass_share = -0.1573`
    - `Q3 positive_mass_share = +0.0615`
    - `Q4 positive_mass_share = +0.3948`
    - `Q1 value_corr = +0.5608`

- Locked conclusions:
  - finer token buckets overturn the coarse “early mass helps” reading
  - in the measured subset:
    - early-bucket positive mass (`Q1/Q2`) is **anti-correlated** with positive realized delta
    - later-bucket positive mass (`Q3/Q4`) is **positively correlated** with positive realized delta
    - `Q3` aligns even more strongly than `Q4`
  - successful low-mass envs differ from failed high-mass envs by:
    - lower `Q1/Q2` positive-mass share
    - higher `Q4` positive-mass share
    - better `Q1` value-return correlation
  - current Phase 3 bottleneck reading tightens further:
    - the problem is not just “how much positive mass exists”
    - it is **where inside the objective segment the positive mass lands**, together with early-bucket critic alignment

## 2026-04-10 lightweight train-rollout quality probe

- Added diagnostic entrypoint:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_train_rollout_quality_probe.py`
- Duplicate-run guard:
  - reuse existing `output_json` by default
  - rerun only with explicit `--overwrite`
- Contract:
  - must pass Phase 2 preflight first
  - reuses canonical pair1 artifacts:
    - zero-control baseline
    - checkpoint pre-policy baseline
    - optional subset post-delta diagnostic
  - does **not** rebuild `zero/pre/post`
  - does **not** run `train_loop`; it only runs one full-train-suite `collect_rollouts()` under the restored validation-policy contract

- Artifacts:
  - `/home/chen/RLPFN/artifacts/phase3_train_rollout_quality_probe.json`
  - `/home/chen/RLPFN/artifacts/phase3_train_rollout_quality_summary.json`

- Locked numbers:
  - objective token count is uniform across all 16 train envs:
    - `102 / env`
  - positive normalized actor-advantage mass concentration:
    - top-1 env share `0.2102`
    - top-2 env share `0.3429`
    - dominant env index `10`
  - full-train-suite correlations:
    - `corr(pre_suffix_gap, positive_norm_mass) = -0.2885`
    - `corr(terminal_resets, positive_norm_mass) = +0.5392`
    - `corr(value_corr, positive_norm_mass) = +0.3591`
  - subset `[0,1,2,3]` alignment against already measured post deltas:
    - `corr(suffix_gap_delta, pre_suffix_gap) = +0.8768`
    - `corr(suffix_gap_delta, positive_norm_mass) = -0.9263`
    - `corr(suffix_gap_delta, value_corr) = +0.8531`
    - `corr(suffix_gap_delta, terminal_resets) = -0.0085`
    - `corr(suffix_gap_delta, last16_positive_share) = -0.0813`

- Locked conclusions:
  - update concentration is **not** explained by objective token count; that quantity is constant across envs
  - raw positive mass concentration also does **not** explain realized improvement
    - the largest positive-mass envs are not the envs with the largest positive post delta in the measured subset
  - in the measured subset, realized positive delta tracks:
    - pre-existing train suffix gap
    - critic raw value-return correlation
  - in the measured subset, realized positive delta does **not** track:
    - terminal-reset count
    - last-16 objective-tail positive mass share
  - current Phase 3 bottleneck reading is therefore:
    - many-env update is being driven by uneven **rollout/objective quality**, not by objective token count
    - the current train-update signal does not cleanly transfer to heldout under the present contract
  - this remains diagnostic-only and does **not** establish a new trusted Phase 3 baseline

## 2026-04-10 heldout token-bucket transfer probe

- Added diagnostic entrypoint:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_heldout_token_bucket_probe.py`
- Duplicate-run guard:
  - reuse existing `output_json` by default
  - rerun only with explicit `--overwrite`
- Contract:
  - same lightweight contract as the train token-bucket probe
  - reuses canonical zero-control baseline, pre-policy checkpoint baseline, and heldout subset deltas
  - one restored-policy heldout-suite `collect_rollouts()`
  - no zero/pre/post rebuild

- Artifact:
  - `/home/chen/RLPFN/artifacts/phase3_heldout_token_bucket_probe.json`

- Locked numbers:
  - runtime wall: `102.00s`
  - heldout subset bucket alignment vs `suffix_gap_delta`:
    - `Q1 positive_mass_share corr = +0.4197`
    - `Q2 positive_mass_share corr = +0.8897`
    - `Q3 positive_mass_share corr = -0.1213`
    - `Q4 positive_mass_share corr = -0.8492`
  - heldout low-mass positive minus high-mass nonpositive:
    - `Q1 positive_mass_share = -0.5840`
    - `Q2 positive_mass_share = -0.2893`
    - `Q3 positive_mass_share = -0.1027`
    - `Q4 positive_mass_share = -0.0239`
    - `Q1 value_corr = +0.0110`

- Locked conclusions:
  - the train-side `Q3/Q4` success pattern does **not** transfer to the measured heldout subset
  - in the heldout subset, realized positive delta is not associated with stronger `Q3/Q4` positive mass
  - instead, the only positive heldout case in the measured subset is dominated by `Q1/Q2` positive mass
  - therefore the current many-env update is not stably transporting the train-side late-bucket success mode into heldout environments
  - combined with the train token-bucket probe, the current bottleneck reading tightens to:
    - train-side nonpositive cases are explained by low-quality early-bucket mass and weaker early-bucket critic alignment
    - heldout-side failure is explained by transfer mismatch: the train-side `Q3/Q4` success signature is not preserved on the heldout subset
  - this remains subset-only diagnostic evidence and does **not** establish a new trusted Phase 3 baseline

## 2026-04-10 train-vs-heldout transfer bottleneck summary

- Added diagnostic entrypoint:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_transfer_bottleneck_probe.py`
- Duplicate-run guard:
  - reuse existing `output_json` by default
  - rerun only with explicit `--overwrite`
- Contract:
  - pure synthesis over the canonical train and heldout token-bucket artifacts
  - no rollout collection
  - no zero/pre/post rebuild

- Artifact:
  - `/home/chen/RLPFN/artifacts/phase3_transfer_bottleneck_probe.json`

- Locked numbers:
  - runtime wall: `0.0017s`
  - train positive bucket alignment signs:
    - `Q1/Q2` negative
    - `Q3/Q4` positive
  - heldout positive bucket alignment signs:
    - `Q1/Q2` positive
    - `Q3/Q4` negative
  - sign-flip buckets:
    - `["q1", "q2", "q3", "q4"]`
  - train positive `Q12 - Q34 = -0.3584`
  - heldout positive `Q12 - Q34 = +0.9219`
  - train positive pre-gap minus train nonpositive = `+3.9575`
  - heldout positive pre-gap minus heldout nonpositive = `-0.4330`

- Locked conclusions:
  - the current train-vs-heldout discrepancy is a **mode inversion**, not simple weakening
  - train-side positive updates are `Q3/Q4`-dominated and occur on higher-pre-gap environments
  - heldout-side positive response, when it occurs, is `Q1/Q2`-dominated and occurs on lower-pre-gap environments
  - therefore the current many-env update is not merely failing to preserve the magnitude of the train success pattern:
    - it is selecting a qualitatively different bucket mode on the heldout subset
  - this further tightens the current Phase 3 bottleneck reading:
    - train-side bottleneck: early-bucket low-quality mass and weak early-bucket alignment
    - heldout-side bottleneck: transfer **mode inversion** relative to the train-side late-bucket success signature
  - this remains diagnostic-only and does **not** establish a new trusted Phase 3 baseline

## 2026-04-10 heldout rollout-quality probe

- Added diagnostic entrypoint:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_heldout_rollout_quality_probe.py`
- Duplicate-run guard:
  - reuse existing `output_json` by default
  - rerun only with explicit `--overwrite`
- Contract:
  - same lightweight contract as the train rollout-quality probe
  - one restored-policy heldout-suite `collect_rollouts()`
  - reuses canonical zero/pre baselines and heldout subset deltas
  - no full rebuild

- Artifact:
  - `/home/chen/RLPFN/artifacts/phase3_heldout_rollout_quality_probe.json`

- Locked numbers:
  - runtime wall: `104.70s`
  - full-heldout coupling:
    - `corr_pre_suffix_gap_vs_positive_norm_mass = -0.1206`
    - `corr_value_corr_vs_positive_norm_mass = +0.2778`
    - `corr_terminal_resets_vs_positive_norm_mass = +0.9237`
    - `corr_last16_positive_share_vs_positive_norm_mass = -0.1284`
  - subset alignment:
    - `corr_suffix_gap_delta_vs_pre_suffix_gap = -0.7658`
    - `corr_suffix_gap_delta_vs_positive_norm_mass = +0.1908`
    - `corr_suffix_gap_delta_vs_value_corr = -0.0615`
  - highest heldout pre-gap envs:
    - env `12`: `pre_gap = 27.6898`, `positive_mass = 0.0`
    - env `11`: `pre_gap = 11.0164`, `positive_mass = 0.3290`

- Locked conclusions:
  - the heldout suite is not simply “hard and signal-free”; strong heldout pre-gap environments exist
  - however positive update mass on heldout is coupled much more strongly to terminal resets than to pre-gap or critic alignment
  - the cleanest high-pre-gap heldout environments are nearly starved of positive mass
  - this indicates a heldout-side selection problem inside the current many-env update, not just weak representation or absent task signal

## 2026-04-10 Phase 3 root-cause synthesis

- Added diagnostic entrypoint:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_root_cause_probe.py`
- Duplicate-run guard:
  - reuse existing `output_json` by default
  - rerun only with explicit `--overwrite`
- Contract:
  - pure synthesis over canonical diagnostic artifacts
  - no rollout collection
  - no rebuild

- Artifact:
  - `/home/chen/RLPFN/artifacts/phase3_root_cause_probe.json`

- Locked numbers:
  - runtime wall: `0.0027s`
  - sequence-layout probe:
    - `collect_path_stable = true`
  - transfer pattern:
    - `all_bucket_alignment_signs_flip_between_train_and_heldout = true`
    - `phase3_transfer_pattern_is_mode_inversion_not_simple_weakening = true`
  - heldout top-2 pre-gap env capture:
    - envs `[12, 11]`
    - mean pre-gap `19.3531`
    - positive-mass share `0.000626`
  - train top-2 pre-gap env capture:
    - envs `[3, 14]`
    - mean pre-gap `6.0138`
    - positive-mass share `0.0270`

- Locked conclusions:
  - current evidence supports an **optimization selection bias**
  - current evidence does **not** support:
    - an active collect/layout bug
    - a “task is simply unlearnable” reading
  - the reset-heavy heldout bias is already visible in **raw** positive actor-advantage mass:
    - heldout `corr_raw_positive_mass_vs_terminal_resets = +0.9203`
    - heldout `corr_raw_positive_mass_vs_pre_suffix_gap = -0.1274`
  - therefore global advantage normalization is **not** the primary driver of the heldout failure mode
  - the sharpened Phase 3 root-cause reading is:
    - train side: early-bucket low-quality mass and weak early-bucket alignment
    - heldout side: update mass is overly reset-driven and starves the cleanest high-pre-gap environments
    - transfer failure appears as mode inversion relative to the train-side late-bucket success signature
  - this remains diagnostic-only and does **not** establish a new trusted Phase 3 baseline

## 2026-04-10 Phase 3 heldout reset-semantics probe

- Added diagnostic entrypoint:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_heldout_reset_semantics_probe.py`
- Duplicate-run guard:
  - script reuses existing `output_json` unless `--overwrite`
  - probe stays on the same lightweight heldout contract:
    - reuse existing zero-control baseline
    - reuse existing pre-policy baseline
    - reuse existing heldout subset delta artifact
    - one heldout rollout only; no full rebuild
- Test status:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_phase3_heldout_reset_semantics_probe.py`
  - focused test pass: `2 passed`
- Artifact:
  - `/home/chen/RLPFN/artifacts/phase3_heldout_reset_semantics_probe.json`
- Locked runtime:
  - probe wall `100.97s`

- Locked conclusions:
  - in reset-heavy heldout envs, positive actor-advantage mass is **not** primarily concentrated in:
    - `post_first_reset`
    - `terminal_tail`
    - `post_reset_terminal_tail`
  - aggregate reset-heavy segment means:
    - `first_episode.actor_adv_positive_mass_share = 0.5327`
    - `post_first_reset.actor_adv_positive_mass_share = 0.4673`
    - `terminal_tail.actor_adv_positive_mass_share = 0.4323`
    - `non_tail.actor_adv_positive_mass_share = 0.5677`
    - `post_reset_non_tail.actor_adv_positive_mass_share = 0.3037`
    - `post_reset_terminal_tail.actor_adv_positive_mass_share = 0.1636`
  - raw residual positive mass is even more concentrated away from terminal-tail:
    - `non_tail.raw_residual_positive_mass_share = 0.8454`
    - `terminal_tail.raw_residual_positive_mass_share = 0.1546`
    - `post_reset_non_tail.raw_residual_positive_mass_share = 0.5312`
    - `post_reset_terminal_tail.raw_residual_positive_mass_share = 0.0095`
  - sharpened reading:
    - the heldout failure mode is **not** best described as a pure terminal-tail semantics bug
    - it is also **not** best described as a pure “post-first-reset dominates the update” bug
    - current deepest Phase 3 reading is:
      - reset-conditioned **non-tail** objective weighting / baseline interaction
      - with `post_reset_non_tail` as the strongest candidate sink, not `post_reset_terminal_tail`
  - diagnostic-only; no new trusted Phase 3 baseline

## 2026-04-10 Phase 3 heldout layered mass decomposition

- Added diagnostic entrypoint:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_heldout_layered_mass_probe.py`
- Duplicate-run guard:
  - script reuses existing `output_json` unless `--overwrite`
  - probe stays on the same lightweight heldout contract (reuse zero/pre/subset artifacts)
- Test status:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_phase3_heldout_layered_mass_probe.py`
  - focused run: `2 passed`
- Artifact:
  - `/home/chen/RLPFN/artifacts/phase3_heldout_layered_mass_probe.json`
- Runtime:
  - wall `101.91s`

- Locked layer conclusion:
  - earliest reset-heavy bias appears at **raw actor-advantage** mass:
    - `earliest_reset_bias_layer = raw_actor_adv_positive_mass`
  - supported flags:
    - `raw_actor_objective_bias_supported = true`
    - `return_target_shape_bias_supported = false`
    - `critic_baseline_bias_supported = false`
    - `normalization_primary_driver_supported = false`
  - interpretation:
    - the heldout reset-heavy bias is created **after** return/residual,
      **before** normalization, inside the raw objective/advantage construction
    - therefore the primary culprit is **objective weighting / mask semantics**,
      not return target shape or critic baseline
  - diagnostic-only; no new trusted Phase 3 baseline

## 2026-04-10 Phase 3 heldout layered mass decomposition probe

- Added diagnostic entrypoint:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_heldout_layered_mass_probe.py`
- Duplicate-run guard:
  - script reuses existing `output_json` unless `--overwrite`
  - one heldout rollout only; no full rebuild
- Probe implementation note:
  - initial run exposed a real probe bug:
    - `episode_starts` had been indexed with the wrong shape during terminal-reset counting
  - fixed before canonical rerun; canonical artifact below is the only valid result
- Test status:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_phase3_heldout_layered_mass_probe.py`
  - focused test pass: `2 passed`
- Artifact:
  - `/home/chen/RLPFN/artifacts/phase3_heldout_layered_mass_probe.json`
- Locked runtime:
  - wall `101.91s`

- Locked layer decomposition:
  - earliest supported reset-bias layer:
    - `raw_actor_adv_positive_mass`
  - supported flags:
    - `return_target_shape_bias_supported = false`
    - `critic_baseline_bias_supported = false`
    - `raw_actor_objective_bias_supported = true`
    - `normalization_primary_driver_supported = false`
  - layer couplings:
    - `raw_return_positive_mass`:
      - `corr_vs_terminal_resets = -0.2130`
      - `corr_vs_pre_suffix_gap = -0.2128`
      - top-2 pre-gap capture `0.0538`
    - `raw_residual_positive_mass`:
      - `corr_vs_terminal_resets = -0.3746`
      - `corr_vs_pre_suffix_gap = +0.2117`
      - top-2 pre-gap capture `0.1321`
    - `raw_actor_adv_positive_mass`:
      - `corr_vs_terminal_resets = +0.8820`
      - `corr_vs_pre_suffix_gap = -0.2393`
      - top-2 pre-gap capture `0.0021`
    - `normalized_actor_adv_positive_mass`:
      - `corr_vs_terminal_resets = +0.8808`
      - `corr_vs_pre_suffix_gap = -0.2424`
      - top-2 pre-gap capture `0.0030`

- Locked conclusions:
  - the heldout reset-heavy bias does **not** originate in:
    - raw return target shape
    - raw residual / critic-baseline subtraction
  - the bias first becomes visible at the **raw actor-advantage mass** layer
  - normalization preserves the same ordering but is not the primary source
  - current deepest Phase 3 reading is now:
    - reset-conditioned non-tail objective weighting / actor-advantage construction bias
    - more than pure return-shape bias
    - more than pure critic-baseline bias
  - diagnostic-only; no new trusted Phase 3 baseline

## 2026-04-10 Phase 3 heldout segment ablation

- Added diagnostic entrypoint:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_heldout_segment_ablation_probe.py`
- Duplicate-run guard:
  - script reuses existing `output_json` unless `--overwrite`
  - probe stays on the same lightweight heldout contract (reuse zero/pre/subset artifacts)
- Test status:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_phase3_heldout_segment_ablation_probe.py`
  - focused run: `1 passed`
- Artifact:
  - `/home/chen/RLPFN/artifacts/phase3_heldout_segment_ablation_probe.json`
- Runtime:
  - wall `~102s`

- Locked ablation conclusion (raw actor-adv positive mass):
  - base:
    - `corr_vs_terminal_resets = +0.7272`
    - `corr_vs_pre_suffix_gap = -0.2241`
    - `top2_pre_gap_capture = 0.0049`
  - removing `terminal_tail` yields the largest reset-correlation drop:
    - `corr_vs_terminal_resets = +0.3358`
    - `corr_vs_pre_suffix_gap = -0.2405`
    - `top2_pre_gap_capture = 0.0027`
  - removing `post_reset_terminal_tail` also reduces reset-correlation:
    - `corr_vs_terminal_resets = +0.5026`
  - removing `post_reset_non_tail` is a smaller effect:
    - `corr_vs_terminal_resets = +0.6373`
    - `corr_vs_pre_suffix_gap = -0.2179`
  - removing `non_tail` or `first_episode_non_tail` **worsens** reset-correlation

- Interpretation:
  - reset-heavy bias in raw actor-advantage mass is **most sensitive to terminal-tail tokens**
  - this does **not** improve pre-gap capture, so it is a bias driver, not a cure
  - therefore the immediate culprit is still objective-mask semantics in terminal-tail,
    even if terminal-tail is not the majority mass holder
  - diagnostic-only; no new trusted Phase 3 baseline

## 2026-04-10 Phase 3 terminal-tail mask decomposition

- Added diagnostic entrypoint:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_terminal_tail_mask_probe.py`
- Duplicate-run guard:
  - script reuses existing `output_json` unless `--overwrite`
  - same lightweight heldout contract (reuse zero/pre/subset artifacts)
- Test status:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_phase3_terminal_tail_mask_probe.py`
  - focused run: `1 passed`
- Artifact:
  - `/home/chen/RLPFN/artifacts/phase3_terminal_tail_mask_probe.json`
- Runtime:
  - wall `101.92s`

- Locked decomposition (remaining positive mass after removing tail variants):
  - `tail_len=1`:
    - `terminal_tail corr_reset = +0.3745`, `corr_pre = -0.0433`
    - `first_terminal_tail corr_reset = +0.4046`
    - `post_reset_terminal_tail corr_reset = +0.3886`
  - `tail_len=2`:
    - `terminal_tail corr_reset = +0.3317`, `corr_pre = -0.0299`
    - `first_terminal_tail corr_reset = +0.3919`
    - `post_reset_terminal_tail corr_reset = +0.3615`
  - `tail_len=4`:
    - `terminal_tail corr_reset = +0.2414`, `corr_pre = -0.0017`
    - `first_terminal_tail corr_reset = +0.3644`
    - `post_reset_terminal_tail corr_reset = +0.3093`
  - `tail_len=8`:
    - `terminal_tail corr_reset = +0.0402`, `corr_pre = +0.0430`
    - `first_terminal_tail corr_reset = +0.3000`
    - `post_reset_terminal_tail corr_reset = +0.2062`

- Interpretation:
  - removing **larger** terminal tails rapidly collapses the reset-correlation
  - the strongest bias driver is the **full terminal_tail**, not only post-reset tail
  - first-terminal tails are slightly more biased than post-reset tails
  - this reinforces that terminal-tail objective mask semantics are the dominant bias source
  - diagnostic-only; no new trusted Phase 3 baseline

## 2026-04-10 Tail-bias propagation (raw return vs GAE-adv)

- Added diagnostic entrypoint:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_tail_bias_propagation_probe.py`
- Duplicate-run guard:
  - script reuses existing `output_json` unless `--overwrite`
  - same lightweight heldout contract (reuse zero/pre/subset artifacts)
- Test status:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_phase3_tail_bias_propagation_probe.py`
  - focused run: `1 passed`
- Artifact:
  - `/home/chen/RLPFN/artifacts/phase3_tail_bias_propagation_probe.json`
- Runtime:
  - wall `103.69s`

- Locked propagation conclusion:
  - raw return tail bias is already present:
    - `raw corr_reset = +0.5282` across tail variants (all tail_len)
  - GAE-adv reduces but does **not** remove tail bias:
    - `gae corr_reset` stays positive and sizable:
      - `tail_len=8` terminal_tail `+0.2578`
      - `first_terminal_tail` `+0.4061`
      - `post_reset_terminal_tail` `+0.2122`
  - therefore the bias is **not** created by GAE propagation; it is **amplified/reshaped**, not invented
  - the root layer remains objective mask semantics in terminal-tail
  - diagnostic-only; no new trusted Phase 3 baseline

## 2026-04-10 Terminal-tail rule decomposition (mask interactions)

- Added diagnostic entrypoint:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_terminal_tail_mask_probe.py`
- Duplicate-run guard:
  - script reuses existing `output_json` unless `--overwrite`
  - same lightweight heldout contract (reuse zero/pre/subset artifacts)
- Test status:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_phase3_terminal_tail_mask_probe.py`
  - focused run: `1 passed`
- Artifact:
  - `/home/chen/RLPFN/artifacts/phase3_terminal_tail_mask_probe.json`
- Runtime:
  - wall `105.75s`

- Locked rule decomposition:
  - `first_episode_only` is the strongest bias driver:
    - `corr_reset = +0.8456`
    - `corr_pre = -0.2792`
    - `top2_pre_gap_capture = 0.000084`
  - `post_reset_only` is comparatively benign:
    - `corr_reset = +0.2237`
    - `corr_pre = +0.2354`
    - `top2_pre_gap_capture = 0.1750`
  - tail variants still matter, but less than first-episode-only:
    - `terminal_tail corr_reset` drops from `+0.6844` (tail_len=1) to `+0.2018` (tail_len=8)
    - `first_terminal_tail` remains higher than `post_reset_terminal_tail`

- Interpretation:
  - the dominant bias is **first-episode objective weighting**, not post-reset segments
  - terminal-tail interacts with first-episode dominance but is not the sole source
  - immediate culprit is the combination of:
    - first-episode-only mask
    - terminal-tail mask inside first episode
  - diagnostic-only; no new trusted Phase 3 baseline

## 2026-04-10 First-episode-only ablation verify

- Added diagnostic entrypoint:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_first_episode_ablation_verify.py`
- Duplicate-run guard:
  - script reuses existing `output_json` unless `--overwrite`
  - no new rollout; reads the existing terminal-tail mask probe output
- Test status:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_phase3_first_episode_ablation_verify.py`
  - focused run: `1 passed`
- Artifact:
  - `/home/chen/RLPFN/artifacts/phase3_first_episode_ablation_verify.json`

- Locked verification:
  - base (full objective):
    - `corr_reset = +0.7225`
    - `corr_pre = -0.0421`
  - after removing `first_episode_only` (keep post-reset only):
    - `corr_reset = +0.8456`
    - `corr_pre = -0.2792`
  - deltas:
    - `corr_reset_delta = +0.1230`
    - `corr_pre_delta = -0.2370`

- Interpretation:
  - removing `first_episode_only` does **not** reduce bias
  - therefore it is **not** a unique necessary condition
  - bias persists (and worsens) in the remaining post-reset-only mass
  - diagnostic-only; no new trusted Phase 3 baseline

## 2026-04-10 Post-reset-only ablation verify

- Added diagnostic entrypoint:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_post_reset_ablation_verify.py`
- Duplicate-run guard:
  - script reuses existing `output_json` unless `--overwrite`
  - no new rollout; reads the existing terminal-tail mask probe output
- Test status:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_phase3_post_reset_ablation_verify.py`
  - focused run: `1 passed` (conda env `rlpfn`)
- Artifact:
  - `/home/chen/RLPFN/artifacts/phase3_post_reset_ablation_verify.json`

- Locked verification:
  - base (full objective):
    - `corr_reset = +0.7225`
    - `corr_pre = -0.0421`
  - after removing `post_reset_only` (keep first-episode-only):
    - `corr_reset = +0.2237`
    - `corr_pre = +0.2354`
  - deltas:
    - `corr_reset_delta = -0.4988`
    - `corr_pre_delta = +0.2775`

- Interpretation:
  - removing `post_reset_only` **reduces** reset bias and **improves** pre-gap alignment
  - `post_reset_only` is **not** a necessary condition for the bias
  - the dominant bias still tracks to first-episode-only weighting
  - diagnostic-only; no new trusted Phase 3 baseline

## 2026-04-10 Tail interaction probe (objective_mask ∧ tail)

- Added diagnostic entrypoint:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_terminal_tail_interaction_probe.py`
- Duplicate-run guard:
  - output reuse is default
  - no new rollout; reads terminal-tail mask probe output
- Test status:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_phase3_terminal_tail_interaction_probe.py`
  - focused run: `1 passed`
- Artifact:
  - `/home/chen/RLPFN/artifacts/phase3_terminal_tail_interaction_probe.json`

- Locked interaction breakdown (tail_len=8):
  - bias-dominant bucket by reset-correlation:
    - `post_reset_tail_mass`:
      - `corr_reset = +0.9952`
      - `corr_pre = -0.2229`
      - `mass_share = 0.1765`
  - `terminal_tail_mass` overall is also strongly reset-correlated:
    - `corr_reset = +0.9317`
  - `first_episode_non_tail_mass` is weakly anti-correlated with reset count:
    - `corr_reset = -0.0789`
  - `post_reset_non_tail_mass` remains reset-biased but lower than post-reset tail:
    - `corr_reset = +0.4155`

- Interpretation:
  - the strongest reset-correlation is concentrated in **post-reset terminal-tail** tokens
  - this isolates the interaction term:
    - `post_reset_only ∧ terminal_tail`
  - even though `post_reset_only` alone is not a necessary condition, its **tail** subset is the sharpest bias driver
  - next correction should focus on the post-reset terminal-tail objective slice
  - diagnostic-only; no new trusted Phase 3 baseline

## 2026-04-10 Post-reset terminal-tail ablation verify

- Added diagnostic entrypoint:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_post_reset_tail_ablation_verify.py`
- Duplicate-run guard:
  - output reuse is default
  - no new rollout; reads terminal-tail mask probe output
- Test status:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_phase3_post_reset_tail_ablation_verify.py`
  - focused run: `1 passed` (conda env `rlpfn`)
- Artifact:
  - `/home/chen/RLPFN/artifacts/phase3_post_reset_tail_ablation_verify.json`

- Locked verification:
  - base (full objective):
    - `corr_reset = +0.7225`
    - `corr_pre = -0.0421`
  - after removing `post_reset_terminal_tail`:
    - `corr_reset = +0.4034`
    - `corr_pre = +0.0713`
  - deltas:
    - `corr_reset_delta = -0.3191`
    - `corr_pre_delta = +0.1134`

- Interpretation:
  - removing `post_reset_terminal_tail` **reduces** reset-bias and **improves** pre-gap alignment
  - this confirms the interaction term is a strong contributor to the bias
  - however bias is not fully eliminated, so it is not the sole source
  - diagnostic-only; no new trusted Phase 3 baseline

## 2026-04-10 First-episode tail vs non-tail ablation verify

- Added diagnostic entrypoint:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_first_episode_tail_ablation_verify.py`
- Duplicate-run guard:
  - output reuse is default
  - no new rollout; reads terminal-tail mask probe output
- Test status:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_phase3_first_episode_tail_ablation_verify.py`
  - focused run: `1 passed` (conda env `rlpfn`)
- Artifact:
  - `/home/chen/RLPFN/artifacts/phase3_first_episode_tail_ablation_verify.json`

- Locked verification:
  - base (full objective):
    - `corr_reset = +0.7225`
    - `corr_pre = -0.0421`
  - after removing `first_episode_tail`:
    - `corr_reset = +0.6790`
    - `corr_pre = -0.1011`
  - after removing `first_episode_non_tail`:
    - `corr_reset = +0.8677`
    - `corr_pre = -0.1745`
  - deltas:
    - tail: `corr_reset_delta = -0.0436`, `corr_pre_delta = -0.0589`
    - non-tail: `corr_reset_delta = +0.1451`, `corr_pre_delta = -0.1324`

- Interpretation:
  - removing `first_episode_tail` slightly reduces reset-bias but **worsens** pre-gap alignment
  - removing `first_episode_non_tail` worsens both reset-bias and pre-gap alignment
  - therefore neither first-episode tail nor non-tail is a minimal necessary bias source
  - the dominant remaining lever remains post-reset terminal-tail
  - diagnostic-only; no new trusted Phase 3 baseline

## 2026-04-10 Post-reset terminal-tail band probe

- Added diagnostic entrypoint:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_post_reset_tail_band_probe.py`
- Duplicate-run guard:
  - output reuse is default
  - no new rollout; reads terminal-tail mask probe output
- Test status:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_phase3_post_reset_tail_band_probe.py`
  - focused run: `1 passed` (conda env `rlpfn`)
- Artifact:
  - `/home/chen/RLPFN/artifacts/phase3_post_reset_tail_band_probe.json`

- Locked band breakdown:
  - all tail bands are strongly reset-correlated
  - strongest reset-correlation band:
    - `tail_band_1` (last 1 token)
      - `corr_reset = +0.9952`
      - `corr_pre = -0.2169`
      - `mass_total = 5.3773`
  - remaining bands:
    - `tail_band_2` `corr_reset = +0.9947`, `mass_total = 4.9849`
    - `tail_band_4` `corr_reset = +0.9950`, `mass_total = 9.2921`
    - `tail_band_8` `corr_reset = +0.9936`, `mass_total = 16.0618`

- Interpretation:
  - reset-bias is present in **every** post-reset tail band
  - the last-token band is marginally the most correlated but not uniquely
  - there is no single tail_len cutoff that isolates all bias
  - diagnostic-only; no new trusted Phase 3 baseline

## 2026-04-10 Post-reset tail split by episode index / reset position

- Added diagnostic entrypoint:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_post_reset_tail_episode_probe.py`
- Rollout-required probe:
  - uses the heldout fixed suite to collect one rollout
  - tail_len fixed at `8`
- Test status:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_phase3_post_reset_tail_episode_probe.py`
  - focused run: `1 passed` (conda env `rlpfn`)
- Artifact:
  - `/home/chen/RLPFN/artifacts/phase3_post_reset_tail_episode_probe.json`
- Runtime:
  - wall `103.94s`

- Locked split summaries:
  - episode index:
    - `post_reset_tail_ep2`:
      - `corr_reset = +0.99997`
      - `corr_pre = -0.1198`
      - `mass_total = 5.6693`
    - `post_reset_tail_ep3plus`:
      - `corr_reset = +1.0000`
      - `corr_pre = -0.1184`
      - `mass_total = 7.2463`
  - reset position (segment length median split):
    - `post_reset_tail_early`:
      - `corr_reset = +0.99997`
      - `corr_pre = -0.1198`
      - `mass_total = 5.6693`
    - `post_reset_tail_late`:
      - `corr_reset = +1.0000`
      - `corr_pre = -0.1184`
      - `mass_total = 7.2463`

- Interpretation:
  - both episode-2 and episode-3+ tails are equally reset-biased
  - early vs late reset positions do not separate the bias either
  - this split does not reveal a narrower necessary condition
  - diagnostic-only; no new trusted Phase 3 baseline

## 2026-04-10 Post-reset tail distance/step probe

- Added diagnostic entrypoint:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_post_reset_tail_distance_probe.py`
- Rollout-required probe:
  - per-token distance bins inside post-reset tail
  - reset-step split: first `k` steps vs later (`k ∈ {1,2,4}`)
  - tail_len fixed at `8`
- Test status:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_phase3_post_reset_tail_distance_probe.py`
  - focused run: `1 passed` (conda env `rlpfn`)
- Artifact:
  - `/home/chen/RLPFN/artifacts/phase3_post_reset_tail_distance_probe.json`
- Runtime:
  - wall `102.74s`

- Locked distance-bin summaries:
  - distance 1-4 (closest to terminal) show the strongest reset correlation:
    - `dist_1 corr_reset = +0.9077`
    - `dist_2 corr_reset = +0.9176`
    - `dist_3 corr_reset = +0.9172`
    - `dist_4 corr_reset = +0.8998`
  - distances 5-8 remain reset-biased but weaker:
    - `dist_5 corr_reset = +0.8208`
    - `dist_6 corr_reset = +0.8145`
    - `dist_7 corr_reset = +0.8191`
    - `dist_8 corr_reset = +0.8307`

- Locked reset-step summaries:
  - early steps carry less mass and weaker bias:
    - `early_k1 corr_reset = +0.6133`, `mass_total = 1.2140`
    - `early_k2 corr_reset = +0.6332`, `mass_total = 2.5279`
    - `early_k4 corr_reset = +0.6474`, `mass_total = 5.3245`
  - late steps dominate bias and mass:
    - `late_k1 corr_reset = +0.8775`, `mass_total = 56.6618`
    - `late_k2 corr_reset = +0.8689`, `mass_total = 55.3480`
    - `late_k4 corr_reset = +0.8463`, `mass_total = 52.5513`

- Interpretation:
  - the highest reset-bias concentrates in the **closest 1-4 tail tokens**
  - post-reset bias is dominated by **late** steps after reset (beyond first `k`)
  - this suggests the most promising narrow condition is:
    - post-reset tail, distance ≤ 4, late-step subset
  - diagnostic-only; no new trusted Phase 3 baseline

## 2026-04-10 Post-reset tail dist≤4 & late-step ablation verify

- Added diagnostic entrypoint:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_post_reset_tail_dist4_late_ablation_verify.py`
- Rollout-required ablation:
  - removes post-reset tail tokens with distance ≤ 4 and position ≥ 4
  - tail_len fixed at 8
- Test status:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_phase3_post_reset_tail_dist4_late_ablation_verify.py`
  - focused run: `1 passed` (conda env `rlpfn`)
- Artifact:
  - `/home/chen/RLPFN/artifacts/phase3_post_reset_tail_dist4_late_ablation_verify.json`
- Runtime:
  - wall `105.24s`

- Locked verification:
  - base (full objective):
    - `corr_reset = +0.6351`
    - `corr_pre = +0.1887`
  - after ablation:
    - `corr_reset = +0.5718`
    - `corr_pre = +0.2196`
  - deltas:
    - `corr_reset_delta = -0.0633`
    - `corr_pre_delta = +0.0309`

- Interpretation:
  - removing post-reset tail dist≤4 & late-step **reduces** reset-bias
  - it also **improves** pre-gap alignment
  - effect is material but not a full removal, so this is a strong contributor, not the sole source
  - diagnostic-only; no new trusted Phase 3 baseline

## 2026-04-10 Post-reset tail dist≤2 & late-step ablation verify

- Added diagnostic entrypoint:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_post_reset_tail_dist2_late_ablation_verify.py`
- Rollout-required ablation:
  - removes post-reset tail tokens with distance ≤ 2 and position ≥ 4
  - tail_len fixed at 8
- Test status:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_phase3_post_reset_tail_dist2_late_ablation_verify.py`
  - focused run: `1 passed` (conda env `rlpfn`)
- Artifact:
  - `/home/chen/RLPFN/artifacts/phase3_post_reset_tail_dist2_late_ablation_verify.json`
- Runtime:
  - wall `103.70s`

- Locked verification:
  - base (full objective):
    - `corr_reset = -0.1344`
    - `corr_pre = +0.2776`
  - after ablation:
    - `corr_reset = -0.1344`
    - `corr_pre = +0.2788`
  - deltas:
    - `corr_reset_delta = -0.000033`
    - `corr_pre_delta = +0.001187`

- Interpretation:
  - dist≤2 late-step removal produces **negligible** change
  - therefore the minimal necessary subset is **larger than dist≤2**
  - diagnostic-only; no new trusted Phase 3 baseline

## 2026-04-10 Next Probe Plan: Cross-Suite Repro + Semantic Contrast

Goal:
- replace distance-threshold search with **cross-suite reproducibility** and **semantic-contrast** checks
- verify whether the bias mechanism is stable across suites and tied to reset semantics (not a tail cutoff)

Plan A: Cross-suite reproducibility
- run the same post-reset tail probes on **two additional heldout suites** with distinct suite seeds
- compare:
  - bias direction (corr vs reset)
  - bias strength
  - mass concentration
- success criterion:
  - bias direction is consistent across suites
  - magnitude differences are explainable by reset frequency or episode length distribution

Plan B: Semantic contrast (reset semantics)
- run a controlled contrast with reset semantics altered:
  - disable terminal reset OR
  - switch tail mask logic to null (no terminal-tail masking)
- compare bias metrics to baseline suite
- success criterion:
  - bias collapses or flips when reset semantics are removed or altered

Plan C: Statistical mechanism summary
- regress bias magnitude against:
  - reset count
  - average episode length
  - tail occupancy ratio
- use this to summarize a **suite-invariant** mechanism

Guardrails:
- keep Phase 2 pack green as hard gate
- do not modify training semantics
- diagnostic-only; no new trusted Phase 3 baseline until cross-suite agreement holds

## 2026-04-10 Phase 3 shortrun many-env sequence-layout source probe

- Added diagnostic entrypoint:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_shortrun_sequence_layout_source_probe.py`
- Duplicate-run guard:
  - script reuses existing `output_json` unless `--overwrite`
  - same strict numeric contract as the maintained Phase 2 lineage, with only many-env fixed-suite batching added
- Test status:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_phase3_shortrun_sequence_layout_source_probe.py`
  - current relevant total: `22 passed`
- Artifact:
  - `/home/chen/RLPFN/artifacts/phase3_shortrun_sequence_layout_source_probe.json`

- Locked runtime:
  - repeated collect wall:
    - run1 `114.82s`
    - run2 `114.16s`
  - runtime stays in the same band across repeated runs; no new timing anomaly appeared inside the probe itself

- Locked comparisons:
  - step level:
    - `episode_starts = stable`
    - `dones = stable`
    - `rewards = stable`
    - `actions_valid = stable`
    - `actions_tail = stable`
  - buffer level:
    - `episode_starts = stable`
    - `returns = stable`
    - `advantages = stable`
    - `actor_advantages = stable`
  - flat batch level:
    - `seq_start_indices = stable`
    - `seq_lengths = stable`
    - `episode_starts = stable`
    - `returns = stable`

- Locked diagnostics:
  - `episode_starts_follow_previous_dones = true`
  - `action_tail_nonzero_count = 0`
  - `seq_layout_matches_buffer_batch = true`

- New conclusion:
  - under the current correct many-env shortrun contract:
    - fixed pair1 train suite
    - `strict_fixed_env_mode = true`
    - `deterministic_actor_sampling = true`
    - `deterministic_batch_plan = true`
    - `restore_validation_policy_state = true`
  - the suspected sequence-layout instability does **not** reproduce
  - therefore:
    - terminal-reset / `episode_starts` drift is **not** the active source
    - seq packing / flatten order drift is **not** the active source
    - masked action tail is **not** affecting layout or return path
  - the old quarantined shortrun instability artifact remains historical-only and must not be used to characterize the current many-env fixed-suite collect path
  - this removes one suspected bottleneck candidate, but still does **not** establish a new trusted Phase 3 baseline

## 2026-04-11 Pair1 suite-matched repair + regime anchor read

- Added repair utility:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_suite_matched_gap_repair.py`
- Added anchor probe:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_regime_anchor_env_probe.py`
- Added tests:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_phase3_suite_matched_gap_repair.py`
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_phase3_regime_anchor_env_probe.py`
  - targeted run in `conda` env `rlpfn`: `8 passed`
- New artifacts:
  - `/home/chen/RLPFN/artifacts/phase3_post_reset_tail_distance_probe_pair1_suite_matched.json`
  - `/home/chen/RLPFN/artifacts/phase3_terminal_tail_mask_probe_pair1_suite_matched.json`
  - `/home/chen/RLPFN/artifacts/phase3_post_reset_tail_ablation_verify_pair1_suite_matched.json`
  - `/home/chen/RLPFN/artifacts/phase3_post_reset_ablation_verify_pair1_suite_matched.json`
  - `/home/chen/RLPFN/artifacts/phase3_cross_suite_regime_summary_pair1_suite_matched.json`
  - `/home/chen/RLPFN/artifacts/phase3_regime_mixed_update_probe_pair1_suite_matched.json`
  - `/home/chen/RLPFN/artifacts/phase3_regime_env_concentration_probe_pair1_suite_matched.json`
  - `/home/chen/RLPFN/artifacts/phase3_regime_anchor_env_probe.json`

- Locked bug:
  - pair1 distance/tail-mask probes reused `/home/chen/RLPFN/artifacts/phase3_env_delta_concentration_probe_max4.json`
  - envs outside that partial subset were silently assigned `pre_vs_zero_suffix_gap = 0.0`
  - repair metadata confirms `mismatch_count = 12`
  - root-cause guard now added in code:
    - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_subset_delta_fallback.py`
    - partial heldout subset coverage now raises immediately

- Locked pair1 repair effect:
  - repaired pair1 `dist1_to_4_mean_corr_pre = +0.4612`
  - old negative sign was an artifact of stale partial subset deltas
  - repaired pair1 tail ablations no longer improve pre-gap:
    - post-reset-terminal-tail: `corr_pre_delta = -0.2590`
    - post-reset-only: `corr_pre_delta = -0.2647`

- Locked cross-suite/regime effect after repair:
  - pooled all-env `corr_mass_vs_pre_gap = +0.3137`
  - positive regime `local_corr_mass_vs_pre_gap = +0.3832`
  - nonpositive regime `local_corr_mass_vs_pre_gap = +0.0495`
  - therefore the earlier "regime sign flip" interpretation no longer holds once pair1 is repaired
  - what remains stable is:
    - reset tracking stays positive across regimes
    - positive regime still dominates mass and pooled covariance

- Locked anchor interpretation:
  - pair1 heavy mass is sparse and reset-active:
    - top3 share within pair1 = `0.9567`
    - top envs by mass: env10, env12, env5
  - pair1 heavy mass is semantically mixed:
    - env10/full gap `+101.02`, suffix gap `+5.74`
    - env12/full gap `+577.02`, suffix gap `+27.69`
    - env5/full gap `-13.81`, suffix gap `-0.64`
  - pair1 also has zero-mass/high-gain anchors:
    - env11 has `dist1_to_4_mass = 0.0` but suffix gap `+11.02`
  - pair2 env12 is a real gain anchor, not a metric glitch:
    - compare-contract fields match between zero/PPO baselines
    - all nonfinite counts are zero
    - suffix gap `+21.45`, full gap `+414.92`
  - pair2 env12 is specifically a proxy outlier:
    - mass rank `3`
    - mass share within pair2 `0.0117`
    - pre-gap rank `1`

- Current bottleneck read:
  - `dist1_to_4_mass` is a sparse reset-sensitive proxy, not a full gain proxy
  - optimization/fitting ambiguity now sits between:
    - real reset-active gain anchors
    - reset-active negative anchors
    - zero-mass or tiny-mass positive-gain anchors that the proxy does not capture well
  - this is closer to the actual Phase 3 bottleneck than the old pooled sign-flip narrative

## 2026-04-11 Positive-regime captured-vs-missed gain-anchor structure

- Added canonical pair2 tail-mask artifact:
  - `/home/chen/RLPFN/artifacts/phase3_terminal_tail_mask_probe_pair2.json`
- Added compare probe:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_positive_regime_anchor_structure_probe.py`
- Added test:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_phase3_positive_regime_anchor_structure_probe.py`
  - targeted run in `conda` env `rlpfn`: `4 passed`
- New artifact:
  - `/home/chen/RLPFN/artifacts/phase3_positive_regime_anchor_structure_probe.json`
- Runtime:
  - pair2 tail-mask rollout wall `103.36s`
  - structure compare itself is artifact-only

- Locked grouping contract:
  - positive suites only:
    - pair1 repaired
    - pair2 canonical
    - seed13579 canonical
  - strong gain anchor:
    - `pre_vs_zero_suffix_gap > 1.0`
  - proxy-captured:
    - `dist_mass_share_within_suite >= 0.05`
  - proxy-missed:
    - below that threshold

- Locked result:
  - strong gain anchors total: `12`
  - captured: `2`
    - both from pair1: env10, env12
  - missed: `10`
    - pair1: env2, env11, env14
    - pair2: env0, env1, env4, env7, env12, env13, env15

- Locked structural difference:
  - captured anchors:
    - mean suffix gap `16.7130`
    - mean full gap `339.0199`
    - mean reset count `1.5`
    - mean base positive mass `29.2010`
    - mean dist-mass share `0.3641`
    - mean `post_reset_terminal_tail_removed_share = 0.4638`
    - mean `first_episode_only_removed_share = 0.3643`
  - missed anchors:
    - mean suffix gap `4.8114`
    - mean full gap `85.1207`
    - mean reset count `0.4`
    - mean base positive mass `6.1615`
    - mean dist-mass share `0.0015`
    - mean `post_reset_terminal_tail_removed_share = 0.0281`
    - mean `first_episode_only_removed_share = 0.7468`

- Locked interpretation:
  - captured anchors are reset-heavy, high-mass, post-reset-tail-weighted
  - missed anchors are still genuinely positive on return, but are mostly first-episode / non-reset dominated
  - therefore the current many-env proxy is not merely noisy; it is structurally incomplete
  - the current objective over-exposes reset-tail positive mass and underweights first-episode positive-gain anchors

- Immediate design implication:
  - the next minimal objective/weighting exploration should not delete reset-tail weighting entirely
  - it should add a second coverage channel for first-episode / non-reset positive-gain anchors, then test whether heldout positive anchors become less proxy-missed

## 2026-04-11 Dual-channel anchor coverage check

- Added probe:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_dual_channel_anchor_coverage_probe.py`
- Added test:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_phase3_dual_channel_anchor_coverage_probe.py`
  - targeted run in `conda` env `rlpfn`: `5 passed`
- New artifact:
  - `/home/chen/RLPFN/artifacts/phase3_dual_channel_anchor_coverage_probe.json`
- Runtime:
  - artifact-only; no new rollout

- Locked dual-channel definition:
  - keep reset-tail channel:
    - `dist1_to_4_mass`
  - add first-episode / non-reset coverage channel:
    - `first_episode_non_tail_mass = first_episode_only_mass - first_terminal_tail_mass`
  - anchor threshold:
    - `pre_vs_zero_suffix_gap > 1.0`
  - capture threshold:
    - per-suite channel share `>= 0.05`

- Locked result:
  - baseline reset-tail-only capture:
    - `2 / 12`
  - dual-channel capture:
    - `6 / 12`
  - coverage gain:
    - `+4`
  - recovered anchors:
    - pair1 env2
    - pair2 env0
    - pair2 env13
    - pair2 env15
  - residual missed anchors:
    - pair1 env11
    - pair1 env14
    - pair2 env1
    - pair2 env4
    - pair2 env7
    - pair2 env12

- Locked structural read:
  - recovered anchors:
    - mean base positive mass `14.8114`
    - mean first-episode-non-tail share `0.1546`
    - mean reset-tail share `0.0000`
  - residual missed anchors:
    - mean base positive mass `0.3949`
    - mean first-episode-non-tail share `0.0080`
    - mean reset-tail share `0.0025`
  - pair2 env12 remains the clearest residual miss:
    - suffix gap `+21.45`
    - reset-tail share `0.0117`
    - first-episode-non-tail share `0.0000`
    - base positive mass `0.0`

- Updated bottleneck read:
  - coverage is part of the problem, and a first-episode / non-reset channel materially helps
  - but coverage is no longer the whole story
  - the remaining blocker is that some real-gain anchors still fail to produce meaningful positive actor-adv mass at all
  - therefore the next fix class is not just proxy reweighting; it likely requires checking why those anchors land near-zero or wrong-sign actor objective mass

## 2026-04-11 Residual missed-anchor sign-formation probe

- Added probe:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_residual_missed_anchor_sign_probe.py`
- Added test:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_phase3_residual_missed_anchor_sign_probe.py`
  - targeted run in `conda` env `rlpfn`: `5 passed`
- New artifact:
  - `/home/chen/RLPFN/artifacts/phase3_residual_missed_anchor_sign_probe.json`

- Locked contract:
  - anchor source:
    - `dual_channel_residual_missed`
  - residual missed set:
    - `6` anchors
    - pair1: env11, env14
    - pair2: env1, env4, env7, env12
  - rollout contract:
    - Phase 2 green preflight
    - `trusted_sep_reset_mainline`
    - `n_samples = 2048`
    - `single_eval_pos = 1946`
    - fixed heldout suite replay per pair

- Runtime:
  - pair1 rollout wall:
    - `100.71s`
  - pair2 rollout wall:
    - `99.88s`
  - total wall:
    - `203.30s`

- Locked result:
  - residual missed anchors mean suffix gap:
    - `6.8109`
  - residual missed anchors mean full gap:
    - `126.8925`
  - final normalized actor-adv sign:
    - positive: `1 / 6`
    - wrong-sign: `5 / 6`
  - within-suite positive-mass share collapses mainly at residual -> actor stage:
    - raw return:
      - `0.06275`
    - raw residual:
      - `0.06276`
    - raw actor-adv:
      - `0.01778`
    - normalized actor-adv:
      - `0.01762`
  - mean flipped positive-mass share:
    - return -> residual:
      - `0.0030`
    - residual -> actor:
      - `0.7663`
    - actor -> normalization:
      - `0.1783`
  - diagnosis counts:
    - `raw_return_shape`: `5`
    - `baseline_cancellation`: `1`

- Locked interpretation:
  - residual missed anchors are not primarily a normalization problem
  - they are also not primarily a broad baseline-over-subtraction problem:
    - return -> residual flip share is near zero on average
  - the main bottleneck is positive actor-adv sign formation after the residual stage:
    - dominant share-drop stage is usually `actor_adv`
    - dominant flip stage is usually `actor_adv`
  - pair2 residual misses are mostly raw-return wrong-sign at objective tokens, then remain underweighted or wrong-sign after actor-adv formation
  - pair1 residual misses are critic/baseline-side harder:
    - env11 becomes a normalization-outlier after already degraded residual/actor mass
    - env14 collapses to zero actor positive mass before normalization
  - pair2 env12 is the main exception:
    - raw return positive mass is zero
    - residual/actor stages recover a positive signal
    - but final actor mass rank still trails pre-gap rank by `+4`

- Updated bottleneck read:
  - after dual-channel coverage repair, the next root-cause target is no longer “which semantic bucket is missing”
  - it is “why residual positive signal fails to survive into actor-adv mass on residual missed anchors”
  - the next minimal probe should inspect the residual -> actor transition itself:
    - GAE propagation
    - bootstrap placement
    - advantage sign resolution under the current mask/episode contract

## 2026-04-12 Residual-anchor GAE path probe

- Added probe:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_residual_anchor_gae_path_probe.py`
- Added test:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_phase3_residual_anchor_gae_path_probe.py`
  - targeted run in `conda` env `rlpfn`: `4 passed`
- New artifact:
  - `/home/chen/RLPFN/artifacts/phase3_residual_anchor_gae_path_probe.json`

- Locked contract:
  - anchor set:
    - inherited from `/home/chen/RLPFN/artifacts/phase3_residual_missed_anchor_sign_probe.json`
    - `anchor_source = dual_channel_residual_missed`
    - fixed `6` anchors
  - suite replay contract:
    - Phase 2 green
    - `trusted_sep_reset_mainline`
    - `n_samples = 2048`
    - `single_eval_pos = 1946`
  - probe computes, per objective token:
    - raw rollout residual
    - normalized reward term
    - bootstrap term
    - one-step delta
    - recursive GAE future carry
    - actor-adv reconstruction error

- Runtime:
  - pair1 rollout wall:
    - `100.23s`
  - pair2 rollout wall:
    - `99.84s`
  - total wall:
    - `202.87s`

- Locked math guard:
  - actor recurrence reconstruction exact:
    - max error `4.96e-08`
  - delta decomposition exact:
    - max error `0.0`
  - therefore this artifact can be trusted as an exact read of the current PPO GAE path, not a loose approximation

- Locked result:
  - across all 6 residual anchors:
    - positive raw rollout residual tokens:
      - `227`
    - positive-raw-residual but wrong-sign actor-adv tokens:
      - `121`
    - flip share:
      - `0.5330`
  - aggregate flip causes:
    - `bootstrap_term_negative = 57`
    - `delta_negative_mixed = 12`
    - `negative_gae_future_carry = 52`
  - aggregate mean on flip tokens:
    - reward term:
      - `-0.00455`
    - bootstrap term:
      - `-0.00568`
    - delta:
      - `-0.01022`
    - GAE future carry:
      - `-0.07964`
    - MC-equivalent future carry:
      - `+0.12710`

- Locked env-level split:
  - pair2 env12:
    - positive residual tokens `102`
    - wrong-sign flips `44`
    - dominant cause:
      - `negative_gae_future_carry`
  - pair2 env1:
    - positive residual tokens `17`
    - wrong-sign flips `16`
    - dominant cause:
      - `negative_gae_future_carry`
  - pair2 env7:
    - positive residual tokens `89`
    - wrong-sign flips `43`
    - dominant cause:
      - `bootstrap_term_negative`
  - pair2 env4:
    - positive residual tokens `19`
    - wrong-sign flips `18`
    - dominant cause:
      - `bootstrap_term_negative`
  - pair1 env11:
    - positive residual tokens `0`
  - pair1 env14:
    - positive residual tokens `0`

- Locked interpretation:
  - pair2 residual anchors are the real “positive residual gets turned into wrong-sign actor-adv” cases
  - for those pair2 anchors, there is no single universal cause:
    - env12 / env1 are more future-carry dominated
    - env7 / env4 are more bootstrap-negative dominated
  - aggregate counts are slightly bootstrap-heavy, but the strongest heldout anchor `pair2 env12` is future-carry dominated
  - pair1 env11 / env14 are not in this category at all:
    - on the current PPO rollout they do not produce positive raw rollout residual tokens on the objective suffix
    - so their failure happens earlier than GAE sign flipping
  - non-objective-step gap is not the main hidden bug here:
    - mean `next_step_nonobjective_flip_share` is only `0.0273`

- Updated bottleneck read:
  - the residual-missed set actually contains two regimes:
    - `pair2`:
      - true sign-formation failures inside bootstrap / GAE
    - `pair1`:
      - no positive raw rollout residual on the current objective suffix, so the issue is upstream of GAE sign propagation
  - next minimal probe should therefore split by regime instead of treating all residual anchors as one class
  - the most productive immediate follow-up is:
    - isolate `pair2` positive-residual flip tokens
    - compare bootstrap-dominant vs future-carry-dominant tokens structurally

## 2026-04-12 Pair2 flip-mode split probe

- Added probe:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_pair2_flip_mode_split_probe.py`
- Added test:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_phase3_pair2_flip_mode_split_probe.py`
  - targeted run in `conda` env `rlpfn`: `3 passed`
- New artifact:
  - `/home/chen/RLPFN/artifacts/phase3_pair2_flip_mode_split_probe.json`

- Locked contract:
  - input artifact only:
    - `/home/chen/RLPFN/artifacts/phase3_residual_anchor_gae_path_probe.json`
  - no rollout replay
  - no new runtime randomness
  - target scope:
    - `suite_name = pair2`
    - `positive_residual_wrong_sign_actor = true`
  - split rule:
    - `future_carry_dominant`:
      - `flip_cause = negative_gae_future_carry`
    - `bootstrap_dominant`:
      - `flip_cause in {bootstrap_term_negative, delta_negative_mixed}`

- Locked result:
  - target token count:
    - `121`
  - split counts:
    - `bootstrap_dominant = 69`
    - `future_carry_dominant = 52`
  - env distribution:
    - bootstrap-dominant:
      - `env7 = 27`
      - `env12 = 21`
      - `env4 = 12`
      - `env1 = 9`
    - future-carry-dominant:
      - `env12 = 23`
      - `env7 = 16`
      - `env1 = 7`
      - `env4 = 6`

- Locked structural contrast:
  - future-carry tokens carry the larger heldout-gain anchors:
    - mean `pre_vs_zero_suffix_gap = 10.7195`
    - vs bootstrap `8.0105`
  - future-carry tokens start from stronger positive residual:
    - mean `raw_rollout_residual_norm = 0.3147`
    - vs bootstrap `0.1984`
  - future-carry tokens keep local delta positive on average:
    - mean `reward_norm = +0.00654`
    - mean `bootstrap_term_norm = +0.02383`
    - mean `delta_norm = +0.03037`
  - bootstrap-dominant tokens are locally negative already:
    - mean `reward_norm = -0.02089`
    - mean `bootstrap_term_norm = -0.03685`
    - mean `delta_norm = -0.05774`
  - future-carry tokens are flipped mainly by recursive carry:
    - mean `gae_future_carry_norm = -0.21387`
    - vs bootstrap `-0.12538`
  - final wrong-sign strength is nearly the same in both groups:
    - mean `actor_adv_norm = -0.18349` for future-carry
    - mean `actor_adv_norm = -0.18312` for bootstrap

- Locked temporal read:
  - future-carry tokens are slightly earlier in episode index:
    - mean `objective_episode_index = 0.0192`
    - vs bootstrap `0.0580`
  - but they are not a cleaner “early-token” bucket:
    - mean `token_in_objective_episode = 55.44`
    - vs bootstrap `57.33`
  - and they are actually slightly closer to episode end:
    - mean `tokens_to_objective_episode_end = 25.42`
    - vs bootstrap `29.28`
  - therefore token position alone is not the root separator

- Locked anomaly read:
  - next-step non-objective gap remains minor and is confined to bootstrap mode:
    - bootstrap `next_step_nonobjective_share = 0.0580`
    - future-carry `0.0`
  - this is still too small to explain the main pair2 bottleneck

- Updated bottleneck read:
  - pair2 sign-formation failure is not a single mechanism
  - bootstrap mode explains slightly more tokens by count
  - but future-carry mode carries stronger raw residual and stronger heldout-gain anchors, especially around `env12`
  - since both modes land at almost identical final wrong-sign actor-adv magnitude, the next useful probe should not collapse them back together
  - the next minimal probe should stay on pair2 and trace the future-carry chain itself:
    - which downstream objective tokens contribute the negative recursive carry on `env12`-like anchors
    - whether that carry is produced by genuinely negative future residual, or by a baseline / tail-specific semantic mismatch downstream

## 2026-04-12 Future-carry source-chain probe

- Added probe:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_future_carry_source_chain_probe.py`
- Added test:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_phase3_future_carry_source_chain_probe.py`
  - targeted run in `conda` env `rlpfn`: `3 passed`
- New artifact:
  - `/home/chen/RLPFN/artifacts/phase3_future_carry_source_chain_probe_pair2_env12.json`

- Locked contract:
  - input artifacts only:
    - `/home/chen/RLPFN/artifacts/phase3_residual_anchor_gae_path_probe.json`
    - `/home/chen/RLPFN/artifacts/phase3_pair2_flip_mode_split_probe.json`
  - no rollout replay
  - target scope:
    - `suite_name = pair2`
    - `env_index = 12`
    - `flip_mode = future_carry_dominant`
  - source classification:
    - `true_negative_future_residual`:
      - downstream token has `raw_rollout_residual_raw <= 0`
    - `terminal_reset_tail_semantic_mismatch`:
      - downstream token has positive raw residual but `next_non_terminal = 0`
    - `baseline_bootstrap_semantic_mismatch`:
      - downstream token has positive raw residual and `next_non_terminal = 1`
  - guardrails before interpreting source:
    - objective episode chain must be contiguous in both local index and global step
    - inferred recurrence multiplier must stay stable
    - weighted downstream `delta` sum must reconstruct `gae_future_carry_norm`

- Locked math / layout guard:
  - env12 objective episodes are contiguous:
    - episode0:
      - local `0 -> 56`
      - global step `1946 -> 2002`
      - terminal token `next_non_terminal = 0`
    - episode1:
      - local `57 -> 101`
      - global step `2003 -> 2047`
      - terminal token `next_non_terminal = 1`
  - inferred effective recursion factor:
    - mean `0.9499999361`
    - spread `1.43e-08`
  - source-chain reconstruction error:
    - max `6.52e-07`
    - mean `4.34e-07`
  - target starts with non-objective next step:
    - `0`

- Locked result:
  - target start tokens:
    - `23`
  - episode split:
    - episode0:
      - `22`
    - episode1:
      - `1`
  - total negative future-carry source mass:
    - `15.7387`
  - total positive offset mass:
    - `5.8226`
  - negative source mass split:
    - `true_negative_future_residual = 0.0`
    - `terminal_reset_tail_semantic_mismatch = 11.6735`
    - `baseline_bootstrap_semantic_mismatch = 4.0652`
  - negative source shares:
    - terminal-reset tail semantic mismatch:
      - `0.7417`
    - baseline/bootstrap semantic mismatch:
      - `0.2583`
    - true negative future residual:
      - `0.0`
  - dominant source by start token:
    - terminal-reset tail semantic mismatch:
      - `18`
    - baseline/bootstrap semantic mismatch:
      - `5`

- Locked top contributors:
  - overwhelmingly dominant source token:
    - episode0 local `56`
    - global step `2002`
    - source:
      - `terminal_reset_tail_semantic_mismatch`
    - aggregated negative contribution:
      - `11.6735`
    - local delta:
      - `-1.3474`
    - raw residual stays strongly positive:
      - `6.3613`
  - leading nonterminal baseline/bootstrap source tokens:
    - episode0 local `29`:
      - aggregated negative contribution `0.4548`
    - episode0 local `51`:
      - `0.4178`
    - episode0 local `38`:
      - `0.3658`
  - episode1 does not end in terminal reset:
    - local `101` is a nonterminal rollout-horizon bootstrap source
    - it belongs to `baseline_bootstrap_semantic_mismatch`, not terminal tail

- Locked interpretation:
  - for `pair2 env12` future-carry starts, negative carry is not coming from genuinely negative downstream future residual
  - inside the current objective suffix, every negative source token still has positive raw rollout residual
  - the negative future-carry chain is therefore entirely produced by downstream semantic / bootstrap effects
  - most of that mass comes from one terminal-reset tail token at episode0 local `56`
  - the remaining mass comes from a smaller set of nonterminal bootstrap-negative tokens, plus one rollout-horizon bootstrap endpoint in episode1

- Updated bottleneck read:
  - the strongest heldout future-carry anchor is not failing because the downstream future is actually bad
  - it fails because current GAE propagation is importing negative semantic/bootstrap deltas from downstream tokens that still have positive raw residual
  - this narrows the next root-cause step:
    - inspect why those downstream tokens receive negative local `delta_norm`
    - especially:
      - episode0 local `56` terminal-reset tail token
      - the nonterminal bootstrap-negative cluster around locals `29`, `38`, `51`

## 2026-04-12 Env12 negative-delta probe

- Added probe:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_env12_negative_delta_probe.py`
- Added test:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_phase3_env12_negative_delta_probe.py`
  - targeted run in `conda` env `rlpfn`: `3 passed`
- New artifact:
  - `/home/chen/RLPFN/artifacts/phase3_env12_negative_delta_probe.json`

- Locked contract:
  - input artifacts only:
    - `/home/chen/RLPFN/artifacts/phase3_residual_anchor_gae_path_probe.json`
    - `/home/chen/RLPFN/artifacts/phase3_future_carry_source_chain_probe_pair2_env12.json`
  - no rollout replay
  - focus tokens:
    - episode0 local `56`
    - episode0 locals `29`, `38`, `51`
  - root-cause decomposition is done only in normalized TD space:
    - `reward_norm`
    - `boundary_shift_norm`
    - `bootstrap_term_norm`
  - raw-space value fields are treated as analysis-side auxiliary data only

- Locked normalized-space guard:
  - inferred env std from nonterminal reward scaling:
    - `14.0860414505`
  - std spread:
    - `1.78e-15`
  - inferred terminal boundary mean/std shift:
    - `mean_over_std = 1.1599998240`
  - terminal-reset rows used for that inference:
    - `1`

- Locked result:
  - focus driver split:
    - `terminal_reset_boundary_shift_dominant = 1`
    - `bootstrap_stepdown_overrides_positive_reward = 2`
    - `bootstrap_negative_with_nonpositive_reward = 1`
  - nonterminal cluster summary:
    - token count:
      - `3`
    - mean reward term:
      - `+0.00928`
    - mean bootstrap term:
      - `-0.06993`
    - mean delta:
      - `-0.06065`
    - positive-reward but still negative-delta tokens:
      - `2 / 3`
    - bootstrap-dominant tokens:
      - `3 / 3`

- Locked token-level read:
  - local `56`:
    - `delta_norm = -1.3474`
    - immediate reward-over-std:
      - `-0.02787`
    - terminal boundary shift:
      - `-1.1600`
    - bootstrap term:
      - `-0.15949`
    - dominant driver:
      - `terminal_reset_boundary_shift_dominant`
    - conclusion:
      - its negative local TD sign is mostly created by the terminal-reset boundary shift, not by the immediate reward itself
  - local `29`:
    - `delta_norm = -0.06808`
    - reward term:
      - `-0.00061`
    - bootstrap term:
      - `-0.06747`
    - dominant driver:
      - `bootstrap_negative_with_nonpositive_reward`
  - local `38`:
    - `delta_norm = -0.04881`
    - reward term:
      - `+0.00896`
    - bootstrap term:
      - `-0.05776`
    - dominant driver:
      - `bootstrap_stepdown_overrides_positive_reward`
  - local `51`:
    - `delta_norm = -0.06507`
    - reward term:
      - `+0.01948`
    - bootstrap term:
      - `-0.08455`
    - dominant driver:
      - `bootstrap_stepdown_overrides_positive_reward`

- Locked anomaly:
  - focus-token raw value contract inconsistency is present:
    - max `raw_value_contract_error = 20.3240`
  - therefore:
    - `raw_value_raw`
    - `bootstrap_term_raw`
    - `delta_raw`
    - should not be used as the primary explanation for this step
  - the current probe therefore locks conclusions only in normalized TD space, which is the space actually used by PPO GAE here

- Updated bottleneck read:
  - env12 downstream negative-delta tokens do not form one class
  - terminal-reset tail local `56` is a boundary-shift problem
  - nonterminal locals `29 / 38 / 51` are bootstrap stepdown problems
  - two of those nonterminal tokens already have positive reward contribution, yet still flip negative because the value-stepdown term is larger in magnitude
  - the next root-cause probe should therefore split:
    - why terminal-reset local `56` gets such a large negative boundary-normalized reward term
    - why the nonterminal cluster has such a strong local value stepdown despite remaining future gain

## 2026-04-12 Env12 boundary-stepdown root probe

- Added probe:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_env12_boundary_stepdown_root_probe.py`
- Added test:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_phase3_env12_boundary_stepdown_root_probe.py`
  - targeted run in `conda` env `rlpfn`: `2 passed`
- New artifact:
  - `/home/chen/RLPFN/artifacts/phase3_env12_boundary_stepdown_root_probe.json`

- Locked contract:
  - input artifacts only:
    - `/home/chen/RLPFN/artifacts/phase3_env12_negative_delta_probe.json`
    - `/home/chen/RLPFN/artifacts/phase3_residual_anchor_gae_path_probe.json`
  - no rollout replay
  - scope is fixed:
    - `suite_name = pair2`
    - `env_index = 12`
    - boundary focus:
      - local `56`
      - local `101` as nonterminal contrast
    - bootstrap-stepdown focus:
      - locals `29 / 38 / 51`
  - output reuse remains default unless `--overwrite` is passed

- Locked result:
  - local `56` terminal boundary:
    - boundary shift vs immediate reward-over-std ratio:
      - `41.6161x`
    - boundary share of `reward_norm` magnitude:
      - `0.9765`
    - boundary share of `delta_norm` magnitude:
      - `0.8609`
    - terminal-only presence:
      - boundary shift is present on local `56`
      - absent on nonterminal contrast local `101`
  - locals `29 / 38 / 51` cluster:
    - all three are local value peaks
    - mean previous upswing:
      - `+0.04618`
    - mean next-step drop:
      - `+0.06993`
    - mean reward term:
      - `+0.00928`
    - mean bootstrap term:
      - `-0.06993`
    - mean local delta:
      - `-0.06065`
    - positive-reward but negative-delta tokens:
      - `2 / 3`
    - subpattern split:
      - oscillatory peaks:
        - locals `29`, `38`
      - late-preterminal peak:
        - local `51`

- Locked token-pattern read:
  - local `56`:
    - the large negative boundary-normalized reward term is almost entirely the terminal boundary shift term, not the raw immediate reward-over-std
    - local `101` shows that the same tail position without terminal reset does not create this penalty
  - local `29`:
    - small negative reward term
    - large negative value stepdown
    - rebounds above the current local peak within two steps
    - pattern class:
      - `oscillatory_peak`
  - local `38`:
    - positive reward term
    - larger negative stepdown term
    - rebounds above the current local peak within two steps
    - pattern class:
      - `oscillatory_peak`
  - local `51`:
    - positive reward term
    - largest negative stepdown term in the cluster
    - no fast rebound
    - close to episode end
    - pattern class:
      - `late_preterminal_peak`

- Updated bottleneck read:
  - the local `56` failure is now narrowed to terminal-reset boundary normalization semantics, not generic tail behavior
  - the locals `29 / 38 / 51` failures are now narrowed to local value-peak stepdown semantics, not lack of downstream gain
  - the nonterminal cluster is heterogeneous:
    - an oscillatory over-peak regime at `29 / 38`
    - a late-preterminal stepdown regime at `51`
  - the next root-cause step, if we continue on this branch, should inspect why these local value peaks are produced under the current value contract rather than re-aggregating them into one generic heldout failure bucket

## 2026-04-12 Env12 local56 boundary-contract probe

- Added probe:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_env12_local56_boundary_contract_probe.py`
- Added test:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_phase3_env12_local56_boundary_contract_probe.py`
  - targeted run in `conda` env `rlpfn`: `2 passed`
- New artifact:
  - `/home/chen/RLPFN/artifacts/phase3_env12_local56_boundary_contract_probe.json`

- Locked contract:
  - input artifacts only:
    - `/home/chen/RLPFN/artifacts/phase3_env12_negative_delta_probe.json`
    - `/home/chen/RLPFN/artifacts/phase3_residual_anchor_gae_path_probe.json`
  - no rollout replay
  - scope is fixed:
    - `suite_name = pair2`
    - `env_index = 12`
    - terminal focus:
      - local `56`
    - nonterminal contrasts:
      - locals `55 / 57 / 101`
  - boundary contract must use full-rollout normalization stats from the TD contract
    - not objective-subset return stats

- Locked result:
  - full rollout contract:
    - `mean/std = 1.1600`
    - `mean_raw = 16.3398`
    - `std_raw = 14.0860`
    - nonterminal boundary rows imply:
      - `gamma = 1.0`
  - local `56`:
    - actual boundary shift:
      - `-1.1600`
    - expected full-contract boundary shift:
      - `-1.1600`
    - contract error:
      - `0.0`
    - reward-over-std:
      - `-0.02787`
    - delta with boundary:
      - `-1.34736`
    - delta without boundary:
      - `-0.18736`
    - delta recovery from removing only boundary:
      - `+1.1600`
    - boundary share of delta magnitude:
      - `0.86094`
    - full-rollout mean to immediate reward abs ratio:
      - `41.6161x`
  - objective-subset contrast:
    - objective-only `mean/std = 1.6895`
    - gap vs full contract:
      - `+0.52946`

- Locked interpretation:
  - `local56` gets the large negative boundary term because the current normalized TD contract subtracts full-rollout `mean/std` at terminal-reset when `next_non_terminal = 0`
  - under the current nonterminal rows, `gamma = 1`, so the terminal boundary term is exactly:
    - `-mean/std`
  - the large negative sign is therefore not created by immediate reward magnitude
  - it is created by applying a positive full-rollout centering constant at the terminal boundary
  - objective-only token rows are not a valid estimator for this boundary constant and must not be used to explain `local56`

## 2026-04-12 Env12 value peak-drop contract probe

- Added probe:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_env12_value_peak_drop_contract_probe.py`
- Added test:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_phase3_env12_value_peak_drop_contract_probe.py`
  - targeted run in `conda` env `rlpfn`: `2 passed`
- New artifact:
  - `/home/chen/RLPFN/artifacts/phase3_env12_value_peak_drop_contract_probe.json`

- Locked contract:
  - input artifacts only:
    - `/home/chen/RLPFN/artifacts/phase3_env12_negative_delta_probe.json`
    - `/home/chen/RLPFN/artifacts/phase3_residual_anchor_gae_path_probe.json`
  - no rollout replay
  - scope is fixed:
    - `suite_name = pair2`
    - `env_index = 12`
    - locals:
      - `29 / 38 / 51`
  - analysis is done in normalized value space:
    - full-rollout target:
      - `(rollout_return_raw - mean_raw) / std_raw`
    - value head prediction:
      - `value_norm`
    - one-step local slope:
      - `value_norm_t - value_norm_{t+1}`

- Locked result:
  - contract checks all pass:
    - reward matches target step-change:
      - max error `3.17e-08`
    - delta matches reward minus value-stepdown:
      - max error `0.0`
    - stepdown excess equals error-drop:
      - max error `0.0`
  - aggregate:
    - mean target step-change:
      - `+0.00928`
    - mean value stepdown:
      - `+0.06993`
    - mean stepdown excess:
      - `+0.06065`
    - mean current value error:
      - `+0.79089`
    - mean next value error:
      - `+0.73024`
    - future raw return after token stays positive for all three:
      - mean `8.2351`
  - pattern split:
    - `local29`:
      - `hallucinated_peak_against_target_trend`
      - target step-change:
        - `-0.000611`
      - value stepdown:
        - `+0.06747`
    - `local38`:
      - `amplified_target_peak`
      - target step-change:
        - `+0.008959`
      - value stepdown:
        - `+0.05776`
      - amplification factor:
        - `6.45x`
    - `local51`:
      - `amplified_target_peak`
      - target step-change:
        - `+0.01948`
      - value stepdown:
        - `+0.08455`
      - amplification factor:
        - `4.34x`

- Locked interpretation:
  - the `29 / 38 / 51` problem is not “future gain disappears”
  - all three tokens still have positive next-step raw future return
  - the immediate cause of negative local `delta_norm` is:
    - `value_stepdown > target_step_change`
  - more specifically:
    - `local29` is a hallucinated local peak against the target trend
    - `locals38 / 51` have real target peaks, but the value head amplifies those peaks far beyond target slope
  - therefore the current failure bucket must stay split:
    - terminal boundary-centering on `local56`
    - value-error peak/drop on `29 / 38 / 51`

## 2026-04-12 Env12 local56 mean-source probe

- Added probe:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_env12_local56_mean_source_probe.py`
- Added test:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_phase3_env12_local56_mean_source_probe.py`
  - targeted run in `conda` env `rlpfn`: `2 passed`
- New artifact:
  - `/home/chen/RLPFN/artifacts/phase3_env12_local56_mean_source_probe.json`

- Locked contract:
  - input artifacts only:
    - `/home/chen/RLPFN/artifacts/phase3_env12_local56_boundary_contract_probe.json`
    - `/home/chen/RLPFN/artifacts/phase3_residual_anchor_gae_path_probe.json`
  - no rollout replay
  - no retraining
  - scope is fixed:
    - `suite_name = pair2`
    - `env_index = 12`
    - target question:
      - why full-rollout `mean_raw = 16.3398` is high enough to create `-mean/std = -1.1600` at `local56`
  - this probe is a weighted-average source inference:
    - not a direct terminal-bonus trace probe

- Locked result:
  - timing contract:
    - `n_samples = 2048`
    - `single_eval_pos = 1946`
    - objective rows run from global step `1946` to `2047`
    - objective token count:
      - `102`
    - omitted prefix token count before objective:
      - `1946`
    - objective token share:
      - `4.98%`
    - prefix token share:
      - `95.02%`
  - full vs objective means:
    - full-rollout mean raw:
      - `16.3398`
    - objective-only mean raw:
      - `6.6663`
    - implied omitted-prefix mean raw:
      - `16.8468`
    - full minus objective gap:
      - `+9.6735`
    - prefix minus objective gap:
      - `+10.1806`
  - objective episode readout:
    - episode0:
      - token count `57`
      - mean rollout return raw `9.7042`
      - terminal rows `1`
    - episode1 post-reset:
      - token count `45`
      - mean rollout return raw `2.8182`
      - terminal rows `0`
  - local `56` itself:
    - rollout return raw:
      - `5.6887`
    - reward raw:
      - `-0.3926`
    - return minus full mean:
      - `-10.6511`
    - return minus objective mean:
      - `-0.9775`

- Locked interpretation:
  - the high boundary constant at `local56` is not being set by the objective tail itself
  - it is mainly set by the long omitted rollout prefix before the objective window
  - the post-reset episode does not raise the objective mean:
    - it lowers it sharply from `9.7042` to `2.8182`
  - `local56` itself is also not a positive mean source:
    - its reward is negative
    - its return is below both the full-rollout mean and the objective-only mean
  - therefore the large `-mean/std` boundary term is best explained as a full-rollout centering-scope effect, not a local terminal bonus spike and not a post-reset objective inflation effect

- Locked anomaly:
  - `/home/chen/RLPFN/artifacts/phase3_terminal_tail_mask_probe_pair2.json` reports:
    - `pair2 env12 objective_terminal_reset_count = 0`
  - this disagrees with the trusted local56 chain:
    - local `56` is a terminal row
    - `/home/chen/RLPFN/artifacts/phase3_post_reset_tail_distance_probe_pair2.json` reports env12 terminal reset count `1`
  - consequence:
    - `phase3_terminal_tail_mask_probe_pair2.json` is excluded from the local56 mean-source evidence chain until reconciled

## 2026-04-12 Env12 local56 boundary-scope counterfactual probe

- Added probe:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_env12_local56_boundary_scope_counterfactual_probe.py`
- Added test:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_phase3_env12_local56_boundary_scope_counterfactual_probe.py`
  - targeted run in `conda` env `rlpfn`: `2 passed`
- New artifact:
  - `/home/chen/RLPFN/artifacts/phase3_env12_local56_boundary_scope_counterfactual_probe.json`

- Locked contract:
  - input artifacts only:
    - `/home/chen/RLPFN/artifacts/phase3_env12_local56_boundary_contract_probe.json`
    - `/home/chen/RLPFN/artifacts/phase3_residual_anchor_gae_path_probe.json`
  - no rollout replay
  - no retraining
  - target question:
    - if `local56` used an objective-aligned `mean/std`, would terminal boundary penalty shrink

- Locked result:
  - current full-rollout scope:
    - boundary shift:
      - `-1.1600`
    - reward-over-std:
      - `-0.02787`
    - reward norm before value term:
      - `-1.18787`
  - objective-all scope:
    - boundary shift:
      - `-1.68946`
    - ratio vs current:
      - `1.456x`
  - objective-episode0-only scope:
    - boundary shift:
      - `-4.55267`
    - ratio vs current:
      - `3.925x`
  - objective-episode1-only scope:
    - boundary shift:
      - `-1.61690`
    - ratio vs current:
      - `1.394x`

- Locked interpretation:
  - no objective-aligned centering scope shrinks the local56 terminal boundary penalty
  - all tested objective-aligned scopes make the penalty more negative
  - the naive intuition:
    - “align centering scope to objective tokens and the boundary penalty should shrink”
    - is false on the current trusted env12 contract
  - therefore:
    - boundary normalization scope must not be changed to objective-aligned centering as a direct fix candidate without a deeper contract redesign

## 2026-04-12 Env12 local56 boundary mean-std-interaction probe

- Added probe:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_env12_local56_boundary_mean_std_interaction_probe.py`
- Added test:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_phase3_env12_local56_boundary_mean_std_interaction_probe.py`
  - targeted run in `conda` env `rlpfn`: `2 passed`
- New artifact:
  - `/home/chen/RLPFN/artifacts/phase3_env12_local56_boundary_mean_std_interaction_probe.json`

- Locked contract:
  - input artifact only:
    - `/home/chen/RLPFN/artifacts/phase3_env12_local56_boundary_scope_counterfactual_probe.json`
  - no rollout replay
  - no retraining
  - exact decomposition target:
    - boundary magnitude delta
      - `mean_1 / std_1 - mean_0 / std_0`
    - exact split:
      - `mean_only = (mean_1 - mean_0) / std_0`
      - `std_only = mean_0 * (1/std_1 - 1/std_0)`
      - `interaction = (mean_1 - mean_0) * (1/std_1 - 1/std_0)`

- Locked result:
  - objective-all:
    - total worsening:
      - `+0.52946`
    - mean-only:
      - `-0.68674`
    - std-only:
      - `+2.98105`
    - interaction:
      - `-1.76485`
    - dominant worsening driver:
      - `std_too_small`
  - objective-episode0-only:
    - total worsening:
      - `+3.39267`
    - mean-only:
      - `-0.47108`
    - std-only:
      - `+6.50571`
    - interaction:
      - `-2.64196`
    - dominant worsening driver:
      - `std_too_small`
  - objective-episode1-only:
    - total worsening:
      - `+0.45690`
    - mean-only:
      - `-0.95993`
    - std-only:
      - `+8.21453`
    - interaction:
      - `-6.79771`
    - dominant worsening driver:
      - `std_too_small`

- Locked interpretation:
  - objective-aligned scope gets worse for `local56` because `std` collapses, not because `mean` grows
  - in every tested objective-aligned scope:
    - `mean` change alone would shrink the terminal boundary penalty
    - `std` change alone would strongly enlarge it
    - interaction is negative and offsets part of the `std` damage
  - therefore the root answer to:
    - “is mean too large, is std too small, or is interaction to blame?”
    - is:
      - `std too small` is the dominant driver
      - `mean` is not the driver on this branch
      - interaction is not the driver either; it partially mitigates the worsening

## 2026-04-12 Env12 local56 scale-collapse probe

- Added probe:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_env12_local56_scale_collapse_probe.py`
- Added test:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_phase3_env12_local56_scale_collapse_probe.py`
  - targeted run in `conda` env `rlpfn`: `2 passed`
- New artifact:
  - `/home/chen/RLPFN/artifacts/phase3_env12_local56_scale_collapse_probe.json`

- Locked contract:
  - input artifacts only:
    - `/home/chen/RLPFN/artifacts/phase3_env12_local56_mean_source_probe.json`
    - `/home/chen/RLPFN/artifacts/phase3_env12_local56_boundary_scope_counterfactual_probe.json`
  - no rollout replay
  - no retraining
  - target question:
    - why objective-aligned scope has much smaller `std` than full rollout for env12 local `56`
    - which return-distribution segment actually creates that scale collapse
  - decomposition is two-layer only:
    - full rollout:
      - omitted prefix vs objective tail
    - objective tail:
      - episode0 vs post-reset episode1

- Locked result:
  - scope stats:
    - full rollout:
      - token count `2048`
      - mean raw `16.3398`
      - std raw `14.0860`
      - variance raw `198.4166`
    - omitted prefix before objective:
      - token count `1946`
      - mean raw `16.8468`
      - std raw `14.2421`
      - variance raw `202.8386`
    - objective all:
      - token count `102`
      - mean raw `6.6663`
      - std raw `3.9458`
      - variance raw `15.5694`
    - objective episode0 only:
      - token count `57`
      - mean raw `9.7042`
      - std raw `2.1315`
    - objective episode1 only:
      - token count `45`
      - mean raw `2.8182`
      - std raw `1.7430`
  - scale-collapse readout:
    - objective std ratio vs full:
      - `0.2801x`
    - objective variance ratio vs full:
      - `0.07847x`
    - full-to-objective variance multiple:
      - `12.7440x`
    - prefix std ratio vs full:
      - `1.0111x`
  - full variance decomposition:
    - within prefix contribution:
      - `192.7363`
      - share `97.1372%`
    - within objective contribution:
      - `0.7754`
      - share `0.3908%`
    - between prefix/objective means:
      - `4.9049`
      - share `2.4720%`
    - reconstruction error:
      - `0.0`
  - objective variance decomposition:
    - within episode0 contribution:
      - `2.5390`
      - share `16.3077%`
    - within episode1 contribution:
      - `1.3403`
      - share `8.6086%`
    - between episode means:
      - `11.6901`
      - share `75.0837%`
    - reconstruction error:
      - `5.33e-15`

- Locked interpretation:
  - objective-aligned scope has much smaller `std` mainly because it truncates away the high-variance omitted prefix
  - the omitted prefix is not the compressed slice:
    - its `std` is slightly larger than the full rollout
    - `14.2421 > 14.0860`
  - the scale-collapsed slice is the objective tail itself:
    - `std = 3.9458`
    - only `28.01%` of the full-rollout scale
  - within that objective tail, each episode is even narrower than the mixed objective window:
    - episode0 `std = 2.1315`
    - episode1 `std = 1.7430`
  - most of the objective-window variance that remains is not within-episode spread:
    - it is the mean jump between episode0 and episode1
    - share `75.08%`
  - therefore the current answer to:
    - “why does objective-aligned scope make local56 scale worse?”
    - is:
      - because it throws away the high-variance rollout prefix and leaves a short low-variance tail slice
      - not because the prefix itself is collapsed
      - and not because the objective window contains large within-episode spread

## 2026-04-12 Cross-suite replication: pair1 env12 terminal-local scale chain

- Added probe:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_terminal_local_scale_chain_probe.py`
- Added test:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_phase3_terminal_local_scale_chain_probe.py`
  - targeted run in `conda` env `rlpfn`: `4 passed`
- New artifact:
  - `/home/chen/RLPFN/artifacts/phase3_pair1_env12_terminal_scale_chain_probe.json`

- Locked contract:
  - fixed suite/env:
    - `suite_name = pair1`
    - `env_index = 12`
    - `n_samples = 2048`
    - `single_eval_pos = 1946`
  - no retraining
  - one fixed-suite heldout rollout only
  - full boundary contract must be inferred from token rows:
    - nonterminal `reward_raw / reward_norm` for full-rollout `std`
    - terminal boundary shift for full-rollout `mean/std`
  - do not use direct env-level discounted-return moments as the boundary contract:
    - those are diagnostic-only in this probe
  - do not force-pick one terminal local when the rollout contains multiple objective terminal rows:
    - env-level scale-collapse and boundary-scope conclusions must work with `selected_target_local_index = null`

- Locked result:
  - terminal locals in this heldout rollout:
    - `[17, 79]`
  - important semantics:
    - trusted distance probe still reports `objective_terminal_reset_count = 1`
    - this does not imply one terminal token row
    - token-row terminal count here is `2`
  - boundary contract stability:
    - inferred full-rollout `std raw = 7.7068`
    - inferred full-rollout `mean/std = -1.1874`
    - `inferred_env_std_max_abs_deviation = 8.88e-16`
    - `inferred_terminal_mean_over_std_max_abs_deviation = 0.0`
    - all terminal candidates match that full-rollout centering contract
  - scope worsening:
    - all terminal candidates worsen under objective-only scope
  - scale-collapse readout:
    - full rollout:
      - mean raw `-9.1513`
      - std raw `7.7068`
    - omitted prefix:
      - mean raw `-9.3163`
      - std raw `7.8370`
    - objective all:
      - mean raw `-6.0038`
      - std raw `3.2181`
    - objective std ratio vs full:
      - `0.4176x`
    - full-to-objective variance multiple:
      - `5.7351x`
    - prefix std ratio vs full:
      - `1.0169x`
  - full variance decomposition:
    - within prefix share:
      - `98.2573%`
    - within objective share:
      - `0.8684%`
    - between prefix/objective means share:
      - `0.8743%`
  - objective variance decomposition:
    - between episode means share:
      - `75.2916%`
    - episode0 std:
      - `0.7426`
    - episode1 std:
      - `1.9438`
    - episode2 std:
      - `0.8744`
  - mean-source sign:
    - `prefix_mean_exceeds_objective_mean = false`
    - both scopes are negative-return regimes here
    - the omitted prefix is more negative than the objective tail:
      - prefix mean `-9.3163`
      - objective mean `-6.0038`

- Locked interpretation:
  - historical read under the old collect helper was:
    - objective scope again discards a high-variance omitted prefix
    - objective scope again keeps a shorter low-variance tail slice
  - but this pair1 env12 chain is no longer a trusted official replication point by itself:
    - see the later `pair1 env12 objective-reset contract probe`
    - under the current official collect contract, this env can replay as:
      - `objective_terminal_reset_count_from_segments = 0`
      - `terminal_row_count_within_objective = 0`
  - therefore this section is preserved as historical evidence from the pre-repair collect path
  - the repaired boolean matters:
    - `objective_scope_std_collapse_is_explained_by_truncating_away_high_variance_prefix = true`
    - even when `full std` can differ from `prefix std` due to group-mean structure
  - therefore the stable cross-suite read is:
    - not based on this pair1 env12 artifact alone anymore
    - mean-source direction is not cross-suite invariant

## 2026-04-12 Seed13579 env12 objective-reset contract probe

- Added probe:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_objective_reset_contract_probe.py`
- Added test:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_phase3_objective_reset_contract_probe.py`
  - targeted run in `conda` env `rlpfn`: `3 passed`
- New artifact:
  - `/home/chen/RLPFN/artifacts/phase3_seed13579_env12_objective_reset_contract_probe.json`
- Repeated formal rerun artifact:
  - `/home/chen/RLPFN/artifacts/phase3_seed13579_env12_objective_reset_contract_probe_repeatB.json`
- Preserved first fresh direct-collect mismatch:
  - `/home/chen/RLPFN/artifacts/phase3_seed13579_env12_objective_reset_contract_debug_runA.json`

- Locked contract:
  - fixed suite/env:
    - `suite_name = seed13579`
    - `env_index = 12`
    - `n_samples = 2048`
    - `single_eval_pos = 1946`
  - compare only:
    - stored distance-probe `objective_terminal_reset_count`
    - fresh rollout compressed-segment reset count
    - fresh rollout in-objective terminal-row count
  - fresh collect helper is now locked to suite seeds:
    - `strict_fixed_env_mode = true`
    - `env_rng_seeds = heldout_suite.env_seeds`
    - `rollout_rng_seeds = heldout_suite.rollout_seeds`
    - `deterministic_batch_plan = true`
  - no retraining
  - no full scale-chain interpretation on this suite/env until reset contract is replay-stable

- Locked result:
  - stored distance artifact says:
    - `objective_terminal_reset_count = 1`
  - historical pre-fix direct collect (`debug_runA`) says:
    - compressed segments `[[0, 102]]`
    - `objective_terminal_reset_count_from_segments = 0`
    - `terminal_row_count_within_objective = 0`
  - first post-fix formal probe run says:
    - compressed segments `[[0, 91], [91, 102]]`
    - `objective_terminal_reset_count_from_segments = 1`
    - `terminal_row_count_within_objective = 1`
    - terminal row global step `2036`
  - second post-fix formal probe run says the same thing exactly on the reset-contract fields:
    - compressed segments `[[0, 91], [91, 102]]`
    - `objective_terminal_reset_count_from_segments = 1`
    - `terminal_row_count_within_objective = 1`
    - terminal row global step `2036`
  - post-fix replay-stable fields across the two formal runs:
    - compressed segment starts `[0, 91]`
    - compressed segments `[[0, 91], [91, 102]]`
    - `episode_starts_on_objective_positions = [91]`
    - `objective_terminal_reset_count_from_segments = 1`
    - `terminal_rows_within_objective_global_steps = [2036]`

- Locked interpretation:
  - the old fresh collect helper was unstable on this suite/env:
    - before strict seed wiring, `seed13579 env12` could flip from:
      - no objective reset
      - to one objective reset with an in-objective terminal row
  - wiring suite `env_seeds/rollout_seeds` into the official collect path fixed that specific reset-contract drift:
    - two post-fix formal reruns now agree exactly on the reset-contract fields
  - this does not yet promote `seed13579 env12` to a trusted third scale-chain replication point:
    - only the reset-contract replay bug is addressed here
    - the terminal-local scale chain itself has not been rerun and locked on this suite/env
  - therefore:
    - keep `debug_runA` as historical evidence of the pre-fix bug
    - use the post-fix formal probe artifacts as the current reset-contract baseline for this suite/env
    - still do not extend the cross-suite variance-collapse claim with `seed13579 env12` until the scale-chain probe itself is rerun under the repaired collect contract

## 2026-04-12 Pair1 env12 objective-reset contract probe under repaired collect contract

- New artifact:
  - `/home/chen/RLPFN/artifacts/phase3_pair1_env12_objective_reset_contract_probe.json`

- Locked result:
  - stored pair1 distance artifact says:
    - `objective_terminal_reset_count = 1`
  - fresh official collect under the repaired suite-seed contract says:
    - compressed segments `[[0, 102]]`
    - `objective_terminal_reset_count_from_segments = 0`
    - `terminal_row_count_within_objective = 0`
  - key booleans:
    - `fresh_reset_count_matches_stored_distance_probe = false`
    - `stored_distance_probe_claims_reset_but_fresh_rollout_has_none = true`
    - `stored_distance_probe_claims_reset_but_fresh_rollout_has_no_terminal_row = true`

- Locked interpretation:
  - `pair1 env12` is currently a replay-instability sentinel under the current official collect contract
  - the historical `pair1 env12` terminal-local scale chain must not be promoted as a current trusted replication point
  - the current official contract does not reproduce the required objective reset on this env
  - therefore:
    - quarantine the historical `pair1 env12` scale-chain artifact from the official cross-suite stable set
    - do not use `pair1 env12` in future cross-suite mechanism summaries until this reset-contract mismatch is explained or removed

## 2026-04-12 Pair1 env12 reset-contract replay-diff probe

- New artifact:
  - `/home/chen/RLPFN/artifacts/phase3_pair1_env12_reset_contract_replay_diff_probe.json`

- Probe design:
  - fixed target:
    - `suite_name = pair1`
    - `env_index = 12`
    - `n_samples = 2048`
    - `single_eval_pos = 1946`
  - compare three fresh collects on the same suite/env:
    - `legacy_distance_probe_runA`
    - `legacy_distance_probe_runB`
    - `official_strict`

- Locked result:
  - stored distance row still says:
    - `objective_terminal_reset_count = 1`
    - `env_seed = 787425143`
    - `rollout_seed = 1419953343`
  - `legacy_distance_probe_runA`:
    - matches stored reset count:
      - compressed segments `[[0, 63], [63, 102]]`
      - `objective_terminal_reset_count_from_segments = 1`
      - terminal row global step `2008`
  - `legacy_distance_probe_runB`:
    - same code path, different fresh layout:
      - compressed segments `[[0, 102]]`
      - `objective_terminal_reset_count_from_segments = 0`
      - `terminal_row_count_within_objective = 0`
  - `official_strict`:
    - strict suite env/rollout seeds are explicitly wired and replayed:
      - `env_rng_seed_spec == suite_env_seeds`
      - `rollout_rng_seed_spec == suite_rollout_seeds`
      - `last_collect_rollout_env_rng_seeds == suite_env_seeds`
      - `last_collect_rollout_rollout_rng_seeds == suite_rollout_seeds`
    - fresh layout is:
      - compressed segments `[[0, 102]]`
      - `objective_terminal_reset_count_from_segments = 0`
      - `terminal_row_count_within_objective = 0`

- Locked interpretation:
  - pair1 matches the same source class as pair2:
    - stored pair1 distance row belongs to the legacy unseeded collect contract
    - current official collect belongs to the strict suite-seeded contract
  - evidence:
    - legacy path omits explicit env/rollout seed wiring entirely
    - legacy path is itself unstable across immediate repeats on the same suite/env:
      - runA reset count `1`
      - runB reset count `0`
    - official strict path deterministically replays suite seeds and yields reset count `0`
  - therefore:
    - `pair1 env12` mismatch is not a new semantic bug inside the current official path
    - it is also a contract drift between:
      - historical legacy distance artifacts
      - current strict fresh collect
    - do not mix those two contracts when reasoning about reset counts or cross-suite replication
  - trust boundary after this probe:
    - historical `pair1 env12` scale-chain remains legacy artifact-scoped evidence only
    - it remains excluded from the current official fresh-collect stable set

## 2026-04-12 Seed13579 env12 terminal-local scale chain under repaired collect contract

- Reused probe:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_terminal_local_scale_chain_probe.py`
- Repaired probe output bug before rerun:
  - `objective_variance_decomposition.dominant_objective_variance_source`
  - it is no longer hard-coded to `between_episode_means`
- Updated test:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_phase3_terminal_local_scale_chain_probe.py`
  - targeted related run in `conda` env `rlpfn`: `9 passed`
- New artifact:
  - `/home/chen/RLPFN/artifacts/phase3_seed13579_env12_terminal_scale_chain_probe.json`
- Repeated formal rerun artifact:
  - `/home/chen/RLPFN/artifacts/phase3_seed13579_env12_terminal_scale_chain_probe_repeatB.json`

- Locked contract:
  - fixed suite/env:
    - `suite_name = seed13579`
    - `env_index = 12`
    - `n_samples = 2048`
    - `single_eval_pos = 1946`
  - uses the repaired fresh collect helper:
    - strict fixed-suite env/rollout seeds are wired into `build_recurrent_ppo()`
  - interpret this artifact only after the reset-contract fix:
    - it supersedes the earlier failed `seed13579 env12` scale-chain attempt

- Locked result:
  - reset/boundary shape:
    - `objective_terminal_reset_count_from_distance_probe = 1`
    - `objective_terminal_row_count_from_token_rows = 1`
    - `terminal_local_indices = [90]`
    - `selected_target_local_index = 90`
    - terminal global step `2036`
  - repeated formal rerun agrees exactly on the structural and scale-chain fields:
    - `terminal_local_indices = [90]`
    - `selected_target_local_index = 90`
    - same selected target local readout:
      - global step `2036`
      - objective local `90`
    - same boundary contract core
    - same mean-source summary
    - same scale-collapse readout
    - same objective-variance decomposition
  - boundary contract:
    - all terminal candidates match full-rollout centering contract
    - objective-only scope worsens the boundary penalty
    - full `mean/std = -1.1450`
  - scale-collapse readout:
    - full rollout:
      - mean raw `-28.0773`
      - std raw `24.5206`
    - omitted prefix:
      - mean raw `-29.3377`
      - std raw `24.5059`
    - objective all:
      - mean raw `-4.0308`
      - std raw `2.5495`
    - objective/full std ratio:
      - `0.1040x`
    - objective/full variance ratio:
      - `0.01081x`
    - full-to-objective variance multiple:
      - `92.5017x`
    - prefix/full std ratio:
      - `0.9994x`
  - full variance decomposition:
    - within-prefix share:
      - `94.9054%`
    - within-objective share:
      - `0.0538%`
    - between prefix/objective means share:
      - `5.0408%`
  - objective variance decomposition:
    - dominant source:
      - `within_episode0`
    - between-episode-means share:
      - `16.7959%`
    - within-episode0 contribution:
      - `5.3732`
    - within-episode1 contribution:
      - `0.0351`
  - mean-source sign:
    - `prefix_mean_exceeds_objective_mean = false`
    - both scopes are negative-return regimes here
    - omitted prefix is much more negative than objective:
      - prefix mean `-29.3377`
      - objective mean `-4.0308`

- Locked interpretation:
  - under the repaired collect contract, `seed13579 env12` now supports the same high-level variance mechanism as pair2/pair1:
    - objective scope discards a high-variance prefix
    - objective scope keeps a much lower-variance tail slice
  - and this repaired fresh-collect point is replay-stable under the repaired collect contract:
    - the repeatB artifact matches the original repaired artifact exactly on the key semantic fields
  - what does not stay invariant across suites is the internal structure of the remaining objective variance:
    - here the dominant remainder is `within_episode0`
    - not `between_episode_means`
  - therefore the current trusted cross-suite summary is:
    - stable:
      - high-variance prefix vs low-variance objective-tail split
    - unstable:
      - which objective sub-bucket dominates the residual variance once the tail is isolated

## 2026-04-12 Pair2 env12 objective-reset contract probe under repaired collect contract

- Trigger:
  - the generic fresh-collect terminal-local scale-chain rerun for `pair2 env12` failed before producing an artifact:
    - `ValueError: Could not infer full-rollout mean/std from terminal token rows.`
  - therefore the next narrow check was the formal reset-contract probe only

- New artifact:
  - `/home/chen/RLPFN/artifacts/phase3_pair2_env12_objective_reset_contract_probe.json`

- Locked result:
  - stored pair2 distance artifact says:
    - `objective_terminal_reset_count = 1`
  - fresh official collect under the repaired suite-seed contract says:
    - compressed segments `[[0, 102]]`
    - `objective_terminal_reset_count_from_segments = 0`
    - `terminal_row_count_within_objective = 0`
  - key booleans:
    - `fresh_reset_count_matches_stored_distance_probe = false`
    - `stored_distance_probe_claims_reset_but_fresh_rollout_has_none = true`
    - `stored_distance_probe_claims_reset_but_fresh_rollout_has_no_terminal_row = true`
  - runtime wall:
    - `112.8491s`

- Locked interpretation:
  - `pair2 env12` is now also a replay-instability sentinel under the current official collect contract
  - the historical `pair2 env12 local56` evidence chain is not erased, but its trust boundary is now narrower:
    - it remains artifact-scoped historical evidence tied to the stored pair2 contract
    - it is no longer a current official fresh-collect replication point
  - therefore:
    - quarantine `pair2 env12` from the current official cross-suite fresh-collect stable set
    - do not use `pair2 env12` to claim official replay-stable replication of the terminal-local scale chain until this reset-contract mismatch is explained or removed
  - at that stage, the current official fresh-collect stable set was:
    - `seed13579 env12` only

## 2026-04-12 Pair2 env12 reset-contract replay-diff probe

- New probe:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_reset_contract_replay_diff_probe.py`
- New tests:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_phase3_reset_contract_replay_diff_probe.py`
  - targeted related run in `conda` env `rlpfn`: `12 passed`
- New artifact:
  - `/home/chen/RLPFN/artifacts/phase3_pair2_env12_reset_contract_replay_diff_probe.json`

- Probe design:
  - fixed target:
    - `suite_name = pair2`
    - `env_index = 12`
    - `n_samples = 2048`
    - `single_eval_pos = 1946`
  - compare three fresh collects on the same suite/env:
    - `legacy_distance_probe_runA`
    - `legacy_distance_probe_runB`
    - `official_strict`
  - record both:
    - reset layout readout
    - runtime collect contract actually wired into `build_recurrent_ppo()`

- Locked result:
  - stored distance row still says:
    - `objective_terminal_reset_count = 1`
    - `env_seed = 379681280`
    - `rollout_seed = 2114629823`
  - `legacy_distance_probe_runA`:
    - matches stored reset count:
      - compressed segments `[[0, 54], [54, 102]]`
      - `objective_terminal_reset_count_from_segments = 1`
      - terminal row global step `1999`
  - `legacy_distance_probe_runB`:
    - same code path, different fresh layout:
      - compressed segments `[[0, 69], [69, 101], [101, 102]]`
      - `objective_terminal_reset_count_from_segments = 2`
      - terminal row global steps `2014`, `2046`
  - `official_strict`:
    - strict suite env/rollout seeds are explicitly wired and replayed:
      - `env_rng_seed_spec == suite_env_seeds`
      - `rollout_rng_seed_spec == suite_rollout_seeds`
      - `last_collect_rollout_env_rng_seeds == suite_env_seeds`
      - `last_collect_rollout_rollout_rng_seeds == suite_rollout_seeds`
    - fresh layout is:
      - compressed segments `[[0, 102]]`
      - `objective_terminal_reset_count_from_segments = 0`
      - `terminal_row_count_within_objective = 0`

- Locked interpretation:
  - the replay diff is now reduced to a single collect-contract source:
    - stored pair2 distance row belongs to the legacy unseeded collect contract
    - current official collect belongs to the strict suite-seeded contract
  - evidence for that reduction:
    - legacy path omits explicit env/rollout seed wiring entirely:
      - no `env_rng_seed_spec`
      - no `rollout_rng_seed_spec`
      - no `last_collect_rollout_*` seed trace
    - legacy path is itself unstable across immediate repeats on the same suite/env:
      - runA reset count `1`
      - runB reset count `2`
    - official strict path deterministically replays the suite seeds and yields reset count `0`
  - therefore:
    - `pair2 env12` mismatch is not a new semantic bug inside the current official path
    - it is a contract drift between:
      - historical legacy distance artifacts
      - current strict fresh collect
    - do not mix those two contracts when reasoning about reset counts or terminal-local replication
  - trust boundary after this probe:
    - `pair2 env12 local56` remains valid only as legacy artifact-scoped evidence
    - it remains excluded from the current official fresh-collect stable set

## 2026-04-12 Official-strict reset census for pair1/pair2

- New probe:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_official_strict_reset_census.py`
- New tests:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_phase3_official_strict_reset_census.py`
  - targeted related run in `conda` env `rlpfn`: `15 passed`
- New artifact:
  - `/home/chen/RLPFN/artifacts/phase3_official_strict_reset_census_pair1_pair2.json`

- Probe design:
  - suites:
    - `pair1`
    - `pair2`
  - collect contract:
    - current official strict suite-seeded collect only
  - per heldout env record:
    - `official_strict_objective_terminal_reset_count`
    - `terminal_row_count_within_objective`
    - `compressed_segments`
    - whether the env can actually enter the terminal-local scale-chain by the real probe preconditions

- Locked result:
  - global conclusion:
    - `any_official_strict_scale_chain_entry = true`
    - `all_pair1_pair2_envs_fail_official_strict_scale_chain_entry = false`
  - official fresh-collect scale-chain entry candidates found by the census:
    - `pair1 env10`
      - distance reset count `2`
      - official strict reset count `1`
      - terminal row count `1`
      - selected target local `80`
      - compressed segments `[[0, 81], [81, 102]]`
    - `pair1 env13`
      - distance reset count `1`
      - official strict reset count `1`
      - terminal row count `1`
      - selected target local `22`
      - compressed segments `[[0, 23], [23, 102]]`
    - `pair2 env2`
      - distance reset count `2`
      - official strict reset count `1`
      - terminal row count `1`
      - selected target local `82`
      - compressed segments `[[0, 83], [83, 102]]`
    - `pair2 env9`
      - distance reset count `1`
      - official strict reset count `1`
      - terminal row count `1`
      - selected target local `25`
      - compressed segments `[[0, 26], [26, 102]]`
    - `pair2 env13`
      - distance reset count `1`
      - official strict reset count `1`
      - terminal row count `1`
      - selected target local `37`
      - compressed segments `[[0, 38], [38, 102]]`
  - suite-level official strict counts:
    - `pair1`
      - reset-positive envs: `10`
      - terminal-row-positive envs: `10`
      - scale-chain-entry envs: `2`
      - entry env indices: `[10, 13]`
    - `pair2`
      - reset-positive envs: `4`
      - terminal-row-positive envs: `4`
      - scale-chain-entry envs: `3`
      - entry env indices: `[2, 9, 13]`
  - important non-entry failures with positive legacy distance reset count:
    - `pair1`: env `5`, env `12`
    - `pair2`: env `1`, env `5`, env `7`, env `12`
    - common failure reason:
      - `Could not infer full-rollout mean/std from terminal token rows.`

- Locked interpretation:
  - the previous fear was too strong:
    - current official strict collect is not limited to `seed13579 env12` as the only place where terminal-local scale-chain structure exists
  - however the census does **not** by itself promote these five envs to replay-stable official replication points:
    - it proves entry viability under one current official strict run
    - it does not yet provide per-env saved formal scale-chain artifacts plus repeatB stability
  - therefore the new trust boundary is:
    - official saved/replay-stable scale-chain point still:
      - `seed13579 env12`
    - official strict fresh-collect candidates now available for next-step formalization:
      - `pair1 env10`
      - `pair1 env13`
      - `pair2 env2`
      - `pair2 env9`
      - `pair2 env13`
  - this materially changes the optimization-bottleneck plan:
    - a larger repair is now more plausible again, because the official contract does expose multiple candidate envs
    - but the next step still must be narrow:
      - formally rerun and repeatB one candidate env first, not redesign the objective yet

## 2026-04-12 Pair2 env13 terminal-local scale chain under current official strict collect

- Reused probe:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_terminal_local_scale_chain_probe.py`
- New artifacts:
  - `/home/chen/RLPFN/artifacts/phase3_pair2_env13_terminal_scale_chain_probe.json`
  - `/home/chen/RLPFN/artifacts/phase3_pair2_env13_terminal_scale_chain_probe_repeatB.json`

- Locked contract:
  - fixed suite/env:
    - `suite_name = pair2`
    - `env_index = 13`
    - `n_samples = 2048`
    - `single_eval_pos = 1946`
  - official strict current collect:
    - distance reset count `1`
    - official strict reset count `1`
    - terminal row count `1`
    - selected target local `37`
    - terminal global step `1983`

- Locked result:
  - repeatB agrees exactly on the key semantic fields:
    - config subset
    - selected target local readout
    - boundary contract
    - mean-source summary
    - scale-collapse block
    - top-level conclusions
  - runtime wall:
    - runA `111.5424s`
    - repeatB `113.2419s`
  - boundary contract core:
    - full mean raw `-3.8278`
    - full std raw `3.1259`
    - objective mean raw `-0.3871`
    - objective std raw `0.2284`
    - objective-vs-full mean/std gap `-0.4702`
  - scale-collapse readout:
    - objective/full std ratio `0.07307x`
    - objective/full variance ratio `0.005339x`
    - full/objective variance multiple `187.3048x`
    - within-prefix share `93.6230%`
    - between prefix/objective means share `6.3504%`
  - objective tail remainder:
    - dominant objective variance source `within_episode1`
    - between-episode-means share `0.4328%`

- Locked interpretation:
  - `pair2 env13` is now a second official saved/replay-stable terminal-local scale-chain replication point under the current strict collect contract
  - it supports the same high-level variance mechanism as `seed13579 env12`:
    - objective scope discards a high-variance prefix
    - objective scope isolates a much lower-variance tail slice
    - objective-scope boundary penalty worsens relative to full-rollout centering
  - what still does **not** become a cross-suite invariant:
    - objective-internal dominant remainder source
    - here it is `within_episode1`, whereas `seed13579 env12` was `within_episode0`
  - current official saved/replay-stable scale-chain set is now:
    - `seed13579 env12`
    - `pair2 env13`

## 2026-04-12 Cross-suite replication: pair1 env13 terminal-local scale chain under current official strict collect

- Commands run:

```bash
conda run -n rlpfn python reinforce-terminal-explore/ticl/analysis/phase3_terminal_local_scale_chain_probe.py \
  /home/chen/RLPFN/rwkv/models_diff/rlpfn_03_30_2026_22_37_53_epoch_13.cpkt \
  --suite-name pair1 \
  --env-index 13 \
  --output-json /home/chen/RLPFN/artifacts/phase3_pair1_env13_terminal_scale_chain_probe.json \
  --overwrite
```

```bash
conda run -n rlpfn python reinforce-terminal-explore/ticl/analysis/phase3_terminal_local_scale_chain_probe.py \
  /home/chen/RLPFN/rwkv/models_diff/rlpfn_03_30_2026_22_37_53_epoch_13.cpkt \
  --suite-name pair1 \
  --env-index 13 \
  --output-json /home/chen/RLPFN/artifacts/phase3_pair1_env13_terminal_scale_chain_probe_repeatB.json \
  --overwrite
```

- Preconditions from official-strict reset census:
  - `pair1 env13` was one of the cleanest current official candidates:
    - distance reset count `1`
    - official reset count `1`
    - terminal-row count within objective `1`
    - selected terminal-local index `22`
    - terminal global step `1968`

- Repeatability check:
  - runA artifact:
    - `/home/chen/RLPFN/artifacts/phase3_pair1_env13_terminal_scale_chain_probe.json`
  - repeatB artifact:
    - `/home/chen/RLPFN/artifacts/phase3_pair1_env13_terminal_scale_chain_probe_repeatB.json`
  - exact-match fields across runA and repeatB:
    - config subset
    - selected-target-local readout
    - boundary contract
    - mean source
    - scale collapse
    - conclusions
  - wall time:
    - runA `115.3087s`
    - repeatB `112.3406s`

- Locked readout:
  - selected target local:
    - `selected_target_local_index = 22`
    - terminal global step `1968`
  - boundary contract core:
    - full mean raw `0.5028`
    - full std raw `0.5088`
    - objective mean raw `0.2310`
    - objective std raw `0.1214`
    - objective-vs-full mean/std gap `+0.9142`
  - scale-collapse readout:
    - objective/full std ratio `0.23858x`
    - objective/full variance ratio `0.05692x`
    - full/objective variance multiple `17.5680x`
    - within-prefix share `98.2202%`
    - between prefix/objective means share `1.4963%`
  - objective tail remainder:
    - dominant objective variance source `within_episode1`
    - between-episode-means share `5.8790%`

- Locked interpretation:
  - `pair1 env13` is now a third official saved/replay-stable terminal-local scale-chain replication point under the current strict collect contract
  - it supports the same high-level mechanism already seen in:
    - `seed13579 env12`
    - `pair2 env13`
  - the stable cross-suite invariant remains:
    - objective scope discards a high-variance prefix
    - objective scope isolates a lower-variance tail slice
    - objective-scope boundary penalty worsens relative to full-rollout centering
  - what still does **not** become a cross-suite invariant:
    - the objective-internal dominant remainder source
    - here it is again `within_episode1`
  - current official saved/replay-stable scale-chain set is now:
    - `seed13579 env12`
    - `pair2 env13`
    - `pair1 env13`

- Implication for repair scope:
  - the evidence threshold is now materially better for a small repair:
    - we no longer rely on a single official point
    - we now have replay-stable support from three suite views
  - this still justifies only a narrow repair target:
    - repair the objective/boundary scope contract that amplifies low-variance terminal tails
    - do not jump to a broad many-env objective rewrite yet

## 2026-04-12 Official boundary-scope counterfactual pack across saved/replay-stable points

- Command run:

```bash
conda run -n rlpfn python reinforce-terminal-explore/ticl/analysis/phase3_official_boundary_scope_counterfactual_pack.py \
  --output-json /home/chen/RLPFN/artifacts/phase3_official_boundary_scope_counterfactual_pack.json \
  --overwrite
```

- Artifact:
  - `/home/chen/RLPFN/artifacts/phase3_official_boundary_scope_counterfactual_pack.json`

- Scope:
  - read-only pack over the three current official saved/replay-stable terminal-local scale-chain points:
    - `seed13579 env12`
    - `pair2 env13`
    - `pair1 env13`
  - comparison performed at each point:
    - current `current_full_rollout` boundary centering
    - counterfactual `objective_all_tokens` boundary centering

- Aggregate results:
  - replay stability:
    - all three points passed canonical-vs-repeatB locked-field checks
  - contract status:
    - all three current points already match the full-rollout centering contract
    - all three objective-scope counterfactuals worsen boundary penalty
    - all three objective-scope counterfactuals worsen pre-value reward-norm magnitude
  - shrink if using full-rollout over objective scope:
    - boundary abs shrink:
      - min `0.4360`
      - mean `0.6068`
      - max `0.9142`
    - reward-norm abs shrink:
      - min `0.2850`
      - mean `1.4054`
      - max `2.1711`
  - shared mechanism strength:
    - objective/full std ratio:
      - min `0.07307x`
      - mean `0.13854x`
      - max `0.23858x`
    - within-prefix share:
      - min `93.6230%`
      - mean `95.5829%`
      - max `98.2202%`

- Locked interpretation:
  - this pack upgrades the boundary-scope claim from a pointwise observation to a three-point guardrail:
    - `full-rollout centering` is the consistently better contract
    - `objective_all_tokens` is a consistently worse counterfactual
  - the pack supports a small-scope repair target only:
    - keep terminal boundary centering locked to the full-rollout contract wherever this semantic path appears
    - do not promote objective-scope terminal centering
  - the pack does **not** justify a broad many-env objective rewrite:
    - the saved official points already use the better contract
    - so this is guardrail evidence first, not proof that a new runtime fix has already been identified
  - next repair-oriented step must therefore narrow further:
    - find any active Phase 3 path that can still drift away from this contract
    - or prove no such drift remains and move to the next bottleneck

## 2026-04-12 Runtime drift scan: current Phase 3 train collect path

- Command run:

```bash
conda run -n rlpfn python reinforce-terminal-explore/ticl/analysis/phase3_runtime_terminal_boundary_drift_scan.py \
  /home/chen/RLPFN/rwkv/models_diff/rlpfn_03_30_2026_22_37_53_epoch_13.cpkt \
  --suite-split train \
  --output-json /home/chen/RLPFN/artifacts/phase3_runtime_terminal_boundary_drift_scan_train.json \
  --overwrite
```

- Artifact:
  - `/home/chen/RLPFN/artifacts/phase3_runtime_terminal_boundary_drift_scan_train.json`

- Scope:
  - current official strict collect path
  - active `train_suite` only
  - suites scanned:
    - `pair1`
    - `pair2`
    - `seed13579`
  - token class scanned:
    - all objective terminal rows in the collected train rollout
  - contract checked:
    - recovered runtime boundary shift from live buffer contents
    - versus expected `- full_rollout_mean / full_rollout_std`

- Aggregate results:
  - runtime wall time:
    - total `333.0787s`
  - coverage:
    - suite count `3`
    - objective terminal token count `17`
    - envs with objective terminal tokens `14`
  - drift:
    - boundary shift mismatch count `0`
    - max abs boundary-shift contract error `8.6939e-08`
    - mean abs boundary-shift contract error `3.3567e-08`
  - mismatch suites:
    - none

- Per-suite readout:
  - `pair1 train_suite`
    - terminal tokens `8`
    - envs with terminal tokens `6`
    - max abs error `8.6939e-08`
    - runtime `110.7687s`
  - `pair2 train_suite`
    - terminal tokens `3`
    - envs with terminal tokens `3`
    - max abs error `2.5343e-08`
    - runtime `110.0625s`
  - `seed13579 train_suite`
    - terminal tokens `6`
    - envs with terminal tokens `5`
    - max abs error `6.1974e-08`
    - runtime `108.7922s`

- Locked interpretation:
  - no runtime terminal-boundary drift was found in the current Phase 3 train collect path
  - the active train collector already respects the locked `full-rollout centering` guardrail
  - this narrows the repair search:
    - a small runtime fix is not currently localized in train-time terminal boundary centering
    - continuing to “fix boundary scope in collect” would now be chasing a problem that is already absent in the live train path
  - therefore the previously identified boundary-scope issue remains:
    - valid as a guardrail against regressions
    - not yet a new active bottleneck inside current train collect

## Phase 3 buffer transport contract scan

- Canonical command:

```bash
conda run -n rlpfn python reinforce-terminal-explore/ticl/analysis/phase3_shortrun_buffer_transport_contract_scan.py \
  /home/chen/RLPFN/rwkv/models_diff/rlpfn_03_30_2026_22_37_53_epoch_13.cpkt \
  --overwrite
```

- Artifact:
  - `/home/chen/RLPFN/artifacts/phase3_shortrun_buffer_transport_contract_scan.json`

- Scope:
  - current official Phase 3 train collect path
  - suites scanned:
    - `pair1`
    - `pair2`
    - `seed13579`
  - transport contract checked:
    - `collect_rollouts()` raw buffer contents
    - `get_gpu_flat()` public flat minibatch transport
    - `get_gpu()` public padded minibatch transport
    - train-side subbatch slicing via the same sequence-slice contract
  - fields scanned:
    - `objective_masks`
    - `episode_starts`
    - `rollout_return_means`
    - `rollout_return_stds`

- Aggregate results:
  - suite count `3`
  - batch count `384`
  - subbatch count `384`
  - transport contract mismatch counts:
    - source -> flat `0`
    - source -> padded valid `0`
    - flat -> padded valid `0`
    - train-side subbatch flat -> padded `0`
    - seq layout mismatch `0`
  - max abs diff by field:
    - `objective_masks = 0.0`
    - `episode_starts = 0.0`
    - `rollout_return_means = 0.0`
    - `rollout_return_stds = 0.0`

- Per-suite readout:
  - `pair1`
    - runtime `116.6407s`
    - batch count `128`
    - subbatch count `128`
    - transport contract ok `true`
  - `pair2`
    - runtime `116.8599s`
    - batch count `128`
    - subbatch count `128`
    - transport contract ok `true`
  - `seed13579`
    - runtime `115.7648s`
    - batch count `128`
    - subbatch count `128`
    - transport contract ok `true`

- Locked interpretation:
  - the current official Phase 3 train path preserves the four transport-critical fields exactly from raw rollout buffer through public getter transport and train-side slice reads
  - there is no evidence of buffer packing / minibatch transport drift in the active runtime path
  - this means the next small fix is not inside `objective_masks / episode_starts / rollout_return_means / rollout_return_stds` transport
  - if a later change touches buffer packing or train-side slicing, this scan is now the required regression gate
## Phase 3 nonpositive regime train-side contrast

- Completed the complementary regime-labeled train-side contrast for `seed24680` using the same wrapper shape as the positive-regime artifact.

- Canonical artifacts:
  - `/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_24680/post_policy_bundle.pt`
  - `/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_24680/phase3_env_delta_concentration_probe_seed24680_max4.json`
  - `/home/chen/RLPFN/artifacts/phase3_train_rollout_quality_probe_seed24680_with_delta.json`
  - `/home/chen/RLPFN/artifacts/phase3_train_env_quality_contrast_regime_seed24680_nonpositive.json`

- Locked reading:
  - suite context:
    - `suite_name = seed24680`
    - `suite_regime = nonpositive`
    - `train_suite_fingerprint = d7c6a96c0f77cd30`
    - fingerprint match verified against the seed24680 rollout-quality artifact
    - regime membership verified against the cross-suite regime summary
  - complementary train-side split:
    - `high_mass_nonpositive = [1, 2]`
    - `low_mass_positive = [0, 3]`
  - contrast readout:
    - `pre_suffix_gap = +0.1286`
    - `raw_value_return_corr = +0.0070`
    - `first16_positive_share = +0.0055`
    - `last16_positive_share = +0.0288`
    - `episode_count = 0.0`
    - `terminal_reset_count = 0.0`
  - conclusion:
    - high positive mass alone is not sufficient for positive delta in the nonpositive regime
    - the train-side separation remains regime-aware, not a single pooled global rule
    - this complement closes the missing regime side of the diagnostic; it does not promote a new Phase 3 baseline

- Operational rule:
  - when comparing train-side update concentration, split by suite regime first
  - do not pool positive and nonpositive suites into one token-quality / gap-correlation summary
  - keep the regime-labeled wrapper as the stable entrypoint for future complementary contrasts

## Phase 3 regime comparison summary pack

- Completed the small compare pack that reads the positive and nonpositive regime-labeled contrasts side by side.

- Canonical artifact:
  - `/home/chen/RLPFN/artifacts/phase3_train_env_quality_regime_compare_pack.json`

- Locked reading:
  - shared signals that survive both regimes:
    - `high_positive_mass_alone_is_not_sufficient_for_positive_delta = true`
    - `episode_segmentation_not_primary_separator = true`
    - `low_mass_positive_envs_have_higher_pre_suffix_gap_than_high_mass_nonpositive = true`
    - `low_mass_positive_envs_have_higher_value_corr_than_high_mass_nonpositive = true`
  - regime-specific separator:
    - `early_mass_distinguishes_better_than_tail_mass = true` only in the positive regime
    - false in the nonpositive regime
  - regime amplification:
    - `pre_suffix_gap` ratio positive/nonpositive `45.655x`
    - `raw_value_return_corr` ratio positive/nonpositive `8.437x`
    - `first16_positive_share` ratio positive/nonpositive `32.667x`
    - `last16_positive_share` flips sign between the two regimes
- recommendation:
  - `weighting_repair_candidate_supported = false`
  - `regime_split_guardrail_required = true`
  - pooled weighting repair is not yet justified by the available evidence

- Operational rule:
  - use the compare pack as the decision gate before any weighting tweak
  - if future evidence adds more regime-labeled contrasts, rerun the same comparison pattern before changing the recommendation

## Phase 3 root-cause closure

- The current bottleneck stack is now stable enough to stop broad exploration:
  - active collect / buffer transport is stable
  - regime split is required and should remain the default guardrail
  - train/heldout transfer is a mode inversion, not simple weakening
  - residual positive anchors fail in actor mass because of GAE / bootstrap / boundary-stepdown semantics, not because the task is unlearnable

- The narrow repair boundary is also fixed:
  - pooled weighting is not yet justified
  - the positive-only early-mass separator is a regime-specific signal, not a pooled global rule
  - the env12 local-56 boundary stepdown is a local semantic failure mode, not a broad new Phase 3 baseline

- Current conclusion for the optimization bottleneck:
  - the root cause is not one single missing token bucket
  - it is a combination of regime-mismatched update transport plus residual sign-formation failures on a subset of anchors
  - the only safe default at this point is to keep regime split as a guardrail and wait for a new regime-labeled contrast before changing the weighting story

## Phase 3 residual flip-mode compare pack

- Completed a new active-search compare pack on the residual sign-formation layer.

- Canonical artifact:
  - `/home/chen/RLPFN/artifacts/phase3_residual_flip_mode_compare_pack.json`

- Locked reading:
  - `future_carry_dominant` residual flips:
    - stronger heldout-gain anchors
    - stronger raw residuals
    - more negative recursive carry
    - nearly the same final actor-adv magnitude as bootstrap-dominant flips
  - `bootstrap_dominant` residual flips:
    - more locally negative reward / delta terms
    - weaker residual and pre-gap signal
  - the two modes are not equivalent, but they end at nearly the same wrong-sign actor-adv magnitude

- Recommendation:
  - `candidate_small_fix_target = future_carry_source_chain`
  - `bootstrap_only_fix_insufficient = true`
  - `regime_split_guardrail_required = true`

- Active-search interpretation:
  - this is the narrowest remaining place where a small fix could still have broad effect
  - the repair target is now more specific than “pooled weighting”:
    - it is the future-carry recursion / downstream semantic chain for the strongest residual positive anchors

## Phase 3 future-carry source-chain compare pack

- Completed a focused compare pack on the future-carry source-chain layer using the pair2 env12 and env1 chains.

- Canonical artifact:
  - `/home/chen/RLPFN/artifacts/phase3_future_carry_source_chain_compare_pack.json`

- Locked reading:
  - the future-carry source chain is not a single uniform bucket across envs:
    - env12 is terminal-reset-tail dominant
    - env1 is true-negative-future-residual dominant
  - the only shared negative source bucket across env12 and env1 is:
    - `baseline_bootstrap_semantic_mismatch`
  - the shared bucket is present at similar share in both envs:
    - env12 share `0.2583`
    - env1 share `0.2668`
    - mean share `0.2625`

- Recommendation:
  - `candidate_small_fix_target = baseline_bootstrap_semantic_mismatch`
  - `branch_specific_followup_needed = true`
  - `shared_candidate_supported = true`

- Active-search interpretation:
  - this is the first shared repair axis inside the future-carry chain that survives both env12 and env1
  - it is narrower than pooled weighting and broader than the env12-only terminal-reset tail branch
  - the remaining branch-specific failures still need separate handling, but the shared bootstrap mismatch is now the best small-fix candidate with cross-anchor coverage

## Phase 3 baseline-bootstrap semantic mismatch compare pack

- Completed a follow-up compare pack that only splits the shared `baseline_bootstrap_semantic_mismatch` axis into:
  - a near-terminal tail core
  - an env12-only mid-episode extension

- Canonical artifact:
  - `/home/chen/RLPFN/artifacts/phase3_baseline_bootstrap_semantic_mismatch_compare_pack.json`

- Locked reading:
  - both env12 and env1 share a near-terminal baseline-bootstrap core
  - env1 is tail-only for this bucket
  - env12 carries an additional mid-episode extension that env1 does not have
  - the shared small-fix target is still `baseline_bootstrap_semantic_mismatch`
  - the narrow repair scope is now `near_terminal_tail_core`

- Recommendation:
  - `candidate_small_fix_target = baseline_bootstrap_semantic_mismatch`
  - `narrow_fix_scope = near_terminal_tail_core`
  - `branch_specific_followup_needed = true`

- Active-search interpretation:
  - this is the smallest shared structure we have found inside the future-carry / bootstrap chain
  - it is tighter than the full bootstrap bucket and should be the only remaining candidate axis for a small repair

## Phase 3 baseline-bootstrap terminal-adjacent core compare pack

- Completed a final shrink of the shared baseline-bootstrap axis to the terminal-adjacent core only.

- Canonical artifact:
  - `/home/chen/RLPFN/artifacts/phase3_baseline_bootstrap_terminal_adjacent_core_compare_pack.json`

- Locked reading:
  - both env12 and env1 share the terminal-adjacent baseline-bootstrap core
  - env12 still has a mid-episode extension, but the shared terminal-adjacent row is common
  - the smallest shared repair axis is now `terminal_adjacent_core`

- Recommendation:
  - `candidate_small_fix_target = baseline_bootstrap_semantic_mismatch`
  - `narrow_fix_scope = terminal_adjacent_core`
  - `branch_specific_followup_needed = true`

- Active-search interpretation:
  - this is the narrowest shared repair point we have found so far inside the future-carry / bootstrap chain
  - it is smaller than the previous near-terminal tail-core scope and still preserves cross-anchor coverage
  - operationally, it should be treated as a guardrail rather than a standalone pooled fix
  - no narrower cross-env shared repair axis remains; the remaining actionable direction is branch-specific env12 mid-episode extension handling

## Phase 3 env12 mid-episode extension compare pack

- Completed the env12-only branch-specific compare pack for the mid-episode extension block.

- Canonical artifact:
  - `/home/chen/RLPFN/artifacts/phase3_env12_mid_episode_extension_compare_pack.json`

- Locked reading:
  - the env12 mid-episode extension is a contiguous four-token block
  - it is uniformly dominated by `baseline_bootstrap_semantic_mismatch`
  - no finer split inside the block is supported by the current evidence
  - the block is branch-specific and distinct from the shared terminal-adjacent core guardrail

- Recommendation:
  - `candidate_small_control_target = env12_mid_episode_extension_block`
  - `branch_specific_followup_needed = true`
  - `shared_guardrail_unchanged = true`

- Active-search interpretation:
  - this is the remaining narrow branch-specific control direction
  - it is not a new shared root cause; the shared guardrail remains `terminal_adjacent_core`

## Phase 3 env12 mid-episode extension counterfactual pack

- Completed a read-only counterfactual pack on the env12-only mid-episode extension block.

- Canonical artifact:
  - `/home/chen/RLPFN/artifacts/phase3_env12_mid_episode_extension_counterfactual_pack.json`

- Locked reading:
  - replacing `env12_mid_episode_extension_block` with the `terminal_adjacent_core` anchor reduces absolute GAE future-carry magnitude by `0.02045608926564456` (`22.68%`)
  - the same counterfactual reduces raw residual magnitude by `0.7392120670826595` (`91.93%`)
  - weighted delta magnitude shrinks by `0.020455745239239626` (`22.68%`)
  - this is material enough to keep the block as a narrow branch-specific control candidate
  - the shared guardrail remains unchanged at `terminal_adjacent_core`

- Recommendation:
  - `candidate_small_control_target = env12_mid_episode_extension_block`
  - `counterfactual_anchor = terminal_adjacent_core`
  - `branch_specific_followup_needed = true`
  - `shared_guardrail_unchanged = true`

- Active-search interpretation:
  - this is still not a pooled fix
  - it is the narrowest branch-specific control candidate currently supported by read-only evidence

## Phase 3 env12 mid-episode extension runtime control probe

- Completed a runtime semantic control probe for the env12 mid-episode extension block.

- Canonical artifact:
  - `/home/chen/RLPFN/artifacts/phase3_env12_mid_episode_extension_runtime_control_probe.json`

- Locked reading:
  - the env12 mid-episode extension is the narrowest supported runtime control unit
  - it can be realized as a contiguous four-token branch-specific gate
  - it requires no shared-core edit
  - the shared guardrail remains unchanged at `terminal_adjacent_core`
  - the counterfactual collapse materially shrinks both future-carry and raw residual

- Recommendation:
  - `candidate_small_control_target = env12_mid_episode_extension_block`
  - `runtime_control_shape = contiguous_four_token_branch_specific_gate`
  - `branch_specific_followup_needed = true`
  - `shared_guardrail_unchanged = true`

- Active-search interpretation:
  - this is a verified narrow control block, not a pooled fix
  - it is the current best candidate for a small, branch-specific runtime-level control

## Phase 3 env12 mid-episode extension runtime gate feasibility probe

- Completed a runtime feasibility probe for the env12 mid-episode extension block.

- Canonical artifact:
  - `/home/chen/RLPFN/artifacts/phase3_env12_mid_episode_extension_runtime_gate_feasibility_probe.json`

- Locked reading:
  - the current runtime mode set does not exactly encode this internal four-token block
  - the smallest runtime edit surface is `MaskedRecurrentPPO._apply_actor_objective_postprocess`
  - no shared-core edit is required
  - the branch-specific control remains feasible as a new actor-objective mode
  - the nearest built-in family is `tokenwise_drop_q1_early`, but it is too broad to serve as the exact control

- Recommendation:
  - `candidate_small_control_target = env12_mid_episode_extension_block`
  - `runtime_gate_kind = new_branch_specific_actor_objective_mode`
  - `minimal_runtime_edit_surface = MaskedRecurrentPPO._apply_actor_objective_postprocess`
  - `shared_guardrail_unchanged = true`
  - `branch_specific_followup_needed = true`

- Active-search interpretation:
  - this is still a small control, not a shared fix
  - the runtime path can host it, but only with a tiny branch-specific mode extension

## Phase 3 env12 mid-episode extension runtime mode regression

- Completed a minimal env12 runtime-mode regression for the branch-specific mid-episode extension block.

- Canonical artifact:
  - `/home/chen/RLPFN/artifacts/phase3_env12_mid_episode_extension_runtime_mode_regression.json`

- Locked reading:
  - the branch-specific control is now realizable by the runtime actor-objective mode `tokenwise_scale_env12_mid_episode_extension_block`
  - only objective positions `18..21` in objective episode `0` are scaled
  - the runtime regression reproduces the same negative-carry shrink on env12:
    - block `mean_start_gae_future_carry_norm` shrinks from `0.0901851886883378` to `0.06972909718751907`
    - the shrink is `0.020456090569496155`
  - the shared core anchor remains unchanged:
    - `core_before = core_after = -0.06972909718751907`

- Recommendation:
  - `candidate_small_control_target = env12_mid_episode_extension_block`
  - `runtime_mode_name = tokenwise_scale_env12_mid_episode_extension_block`
  - `shared_guardrail_unchanged = true`
  - `branch_specific_followup_needed = true`

- Active-search interpretation:
  - the branch-specific control is now runtime-feasible in the minimal postprocess hook
  - the control remains narrow and does not require a shared-core edit

- Strict train-side A/B on `seed24680`:
  - baseline and override were rerun under the same strict collect contract
  - contract settings:
    - `strict_fixed_env_mode = true`
    - `deterministic_actor_sampling = true`
    - `deterministic_batch_plan = true`
    - train suite env/rollout seeds were wired directly from the suite artifact
  - compare result:
    - `max_abs_metric_delta = 0.0`
    - `runtime_mode_no_other_suite_side_effects = true`
  - locked reading:
    - the branch-specific runtime mode stays local under a strict A/B contract
    - no additional shared-core edit is required

- Strict train-side A/B on `seed13579`:
  - baseline and override were rerun under the same strict collect contract
  - contract settings:
    - `strict_fixed_env_mode = true`
    - `deterministic_actor_sampling = true`
    - `deterministic_batch_plan = true`
    - train suite env/rollout seeds were wired directly from the suite artifact
  - compare result:
    - `max_abs_metric_delta = 0.0`
    - `runtime_mode_no_other_suite_side_effects = true`
  - locked reading:
    - the branch-specific runtime mode stays local under a strict A/B contract on a second non-target suite
    - this is now frozen as a runtime-feasible branch-specific control candidate

- Strict train-side A/B on `phase3_cross_env_baseline_pair2`:
  - baseline and override were rerun under the same strict collect contract on the pair2 suite path
  - contract settings:
    - `strict_fixed_env_mode = true`
    - `deterministic_actor_sampling = true`
    - `deterministic_batch_plan = true`
    - train suite env/rollout seeds were wired directly from the suite artifact
  - compare result:
    - `max_abs_metric_delta = 0.0`
    - `runtime_mode_no_other_suite_side_effects = true`
  - locked reading:
    - the branch-specific runtime mode stays local on a third non-target suite path
    - the control candidate can now be treated as frozen for the current runtime contract

## Phase 3 pair2 proxy-to-train transfer audit

- Completed a read-only audit to explain why the `env12_mid_episode_extension_block` local proxy gain did not convert into train-side gain.

- Canonical artifact:
  - `/home/chen/RLPFN/artifacts/phase3_pair2_proxy_to_train_transfer_audit.json`

- Locked reading:
  - the local proxy improvement is real:
    - counterfactual absolute future-carry shrink per row is `0.02045608926564456`
    - across the four-token block, total shrink is `0.08182435706257823`
  - but the block is tiny relative to the pair2 training batch:
    - block token fraction of batch = `0.2451%`
    - block shrink share of total normalized actor-advantage abs mass = `0.0079%`
    - block shrink share of total normalized negative mass = `0.0158%`
    - block shrink share of env12 negative mass = `0.0862%`
  - train-side A/B result stays consistent with that dilution:
    - `train_full_return_delta_change = -0.018476486206054688`
    - `train_suffix_return_delta_change = 0.0`
    - `train_history_identical = true`

- Recommendation:
  - keep `env12_mid_episode_extension_block` as a runtime-feasible branch-specific control candidate
  - do not reinterpret its local proxy gain as evidence of expected train-side improvement
  - the next root-cause direction is `proxy_to_train_update_transfer_semantics`
  - inspect update aggregation semantics next:
    - advantage normalization
    - PPO clipping
    - many-env batch mixing

- Active-search interpretation:
  - the mismatch is not surprising under the current batch-weighted contract
  - the block changes a local unweighted proxy, but its direct contribution is too small to reliably move the aggregated PPO update

## Phase 3 train outer-batch snapshot hook

- Completed a runtime-safe, read-only hook in `MaskedRecurrentPPO.train()` for the flat recurrent outer-batch path.

- Hook contract:
  - writes at most once per audit run
  - captures:
    - pre-normalization `actor_advantages`
    - post-normalization `advantages`
    - `ratio`
    - `clipped_objective`
    - `objective_mask`
    - flat env/step/objective-local position metadata
  - does not change train semantics
  - does not touch shared core
  - does not change pooled weighting

- Strict scope:
  - the path is formally wired only through the pair2 strict A/B audit contract
  - requested snapshot paths are now explicit runtime config, not implicit scratch files:
    - `/home/chen/RLPFN/artifacts/phase3_pair2_train_update_ab_baseline_outer_batch_snapshot.json`
    - `/home/chen/RLPFN/artifacts/phase3_pair2_train_update_ab_override_outer_batch_snapshot.json`

- Verification:
  - targeted test pack passed after wiring:
    - `test_phase3_multi_env_optimization_audit.py`
    - `test_phase3_pair2_train_update_ab_compare_pack.py`
    - `test_phase3_train_outer_batch_snapshot_hook.py`
  - result: `20 passed`

- Locked reading:
  - the codebase can now produce an exact train-side update snapshot for the strict pair2 A/B path
  - the next update-aggregation audit no longer needs to rely on batch-mass approximations alone

- Next step:
  - rerun the strict pair2 target-side A/B once with the snapshot hook enabled
  - then build the exact update-aggregation audit on the saved outer-batch snapshot:
    - block share before normalization
    - share after normalization
    - share after ratio/clipping
    - share relative to whole-batch and whole-env12

## Phase 3 pair2 exact update aggregation audit

- Completed the strict pair2 target-side A/B rerun only far enough to obtain both outer-batch snapshots:
  - `/home/chen/RLPFN/artifacts/phase3_pair2_train_update_ab_baseline_outer_batch_snapshot.json`
  - `/home/chen/RLPFN/artifacts/phase3_pair2_train_update_ab_override_outer_batch_snapshot.json`
- The long compare-pack process was then stopped after both snapshots were written, because the exact next step was read-only snapshot analysis, not another full post-train compare.

- Canonical exact-audit artifact:
  - `/home/chen/RLPFN/artifacts/phase3_pair2_exact_update_aggregation_audit.json`

- Locked reading:
  - the strict runtime hook itself is real and writes valid baseline/override snapshots
  - but the current one-shot capture policy is still wrong for the env12 branch-specific question:
    - baseline and override snapshots both captured only one objective segment:
      - `env_index = 0`
      - `objective_episode_index = 2`
      - `objective_position = 0..101`
    - the requested target block is absent:
      - `env12 objective token count in snapshot = 0`
      - `block_token_count = 0`
      - `requested_target_block_present_in_captured_outer_batch = false`
  - therefore the current snapshot contract is sufficient for exact many-env aggregation on the captured batch, but insufficient for exact env12-block transfer analysis

- Important implication:
  - the bottleneck has moved from "we lack train-side exact numbers" to "the runtime hook captures the first outer batch, not a selected outer batch containing the target block"
  - this is a narrow instrumentation-contract issue, not a shared-core optimization conclusion

- Next step:
  - do not reopen pooled weighting
  - do not retune the block
  - only refine the snapshot trigger/selection contract so that strict pair2 can capture the outer batch that actually contains the env12 branch-specific block

## Phase 3 selectable snapshot hook

- Implemented a narrow selector contract on top of the existing read-only outer-batch snapshot hook.

- New selector surface:
  - optional `target_outer_batch_idx`
  - optional `target_env_index`
  - optional `target_objective_episode_index`
  - optional `target_objective_position_start`
  - optional `target_objective_position_end`

- Runtime behavior:
  - if no selector is given, the hook keeps the old behavior and writes the first eligible outer batch
  - if a selector is given, the hook writes only when the current flat outer batch contains matching objective tokens
  - this remains read-only and does not touch shared core or pooled weighting

- Verification:
  - focused regression pack passed after wiring:
    - snapshot hook metadata/selector tests
    - audit plumbing tests
    - pair2 compare-pack plumbing tests
    - exact aggregation audit tests
  - result: `25 passed`

- Runtime validation status:
  - attempted strict `pair2` baseline-only audit with selector:
    - `env_index = 12`
    - `objective_episode_index = 0`
    - `objective_position = 18..21`
  - the audit was manually stopped after exceeding the expected small-control validation budget:
    - no snapshot file written
    - no output audit JSON written
  - current reading:
    - the selector logic is unit-tested and formally wired
    - but strict runtime confirmation is still open because the selected baseline audit did not finish within the expected runtime envelope

- Open issue:
  - treat this as a runtime-observability/runtime-phase problem, not as evidence that the selector is semantically wrong
  - next step, if this line continues, is to add a minimal read-only per-batch selector trace or equivalent phase-local profile so we can tell whether:
    - the target env12 block never appears in the train-side outer-batch stream
    - or the job is simply spending too long in later runtime phases before returning the audit artifact

## Phase 3 pair2 outer-batch selector trace

- Completed a minimal read-only per-batch selector trace on the strict pair2 train-side outer-batch path.

- Canonical artifact:
  - `/home/chen/RLPFN/artifacts/phase3_pair2_train_side_selector_trace.json`

- Implementation scope:
  - added a flat-path selector trace in `MaskedRecurrentPPO.train()`
  - reused the same selector contract as the snapshot hook:
    - `target_outer_batch_idx`
    - `target_env_index`
    - `target_objective_episode_index`
    - `target_objective_position_start`
    - `target_objective_position_end`
  - the trace writes one row per outer batch and stays read-only:
    - no shared-core edits
    - no pooled-weighting edits
    - no actor-objective semantic change

- Verification:
  - focused regression pack passed after the trace wiring:
    - `test_phase3_train_outer_batch_snapshot_hook.py`
    - `test_phase3_multi_env_optimization_audit.py`
    - `test_phase3_pair2_train_update_ab_compare_pack.py`
  - result: `24 passed`

- Runtime result under the strict pair2 baseline contract:
  - selector target:
    - `env_index = 12`
    - `objective_episode_index = 0`
    - `objective_position = 18..21`
  - trace summary:
    - `row_count = 128`
    - `target_appeared = false`
    - `first_match = null`
  - status breakdown:
    - `no_objective_tokens = 112`
    - `target_env_absent = 15`
    - `target_episode_absent = 1`
  - there was exactly one outer batch containing any `env12` objective tokens:
    - `outer_batch_idx = 104`
    - `target_env_objective_token_count = 102`
    - `target_env_objective_episode_indices_present = [9, 10, 11]`
    - spans:
      - `episode 9 -> positions 0..7`
      - `episode 10 -> positions 0..29`
      - `episode 11 -> positions 0..63`
    - `objective_episode_index = 0` was absent there

- Locked reading:
  - the env12 branch-specific block exact-aggregation audit is still not available on the current strict pair2 train-side runtime path
  - the reason is now explicit:
    - this is not merely a bad one-shot snapshot trigger
    - the requested block does not appear anywhere in the traced outer-batch stream under its expected identity
  - the current bottleneck has moved again:
    - from snapshot selection
    - to train-side segment identity / rollout-to-batch transport semantics

- Next step:
  - do not reopen pooled weighting
  - do not retune the branch-specific block
  - only trace why the saved env12 local proxy block is transformed into train-side `objective_episode_index in {9,10,11}` before batching, instead of remaining `episode 0`

## Phase 3 pair2 train segment identity probe

- Completed a read-only chain probe on the exact path:
  - `rollout -> rollout_buffer -> get_gpu_flat -> outer batch 104`

- Canonical artifact:
  - `/home/chen/RLPFN/artifacts/phase3_pair2_train_segment_identity_probe.json`

- Implementation scope:
  - added a standalone analysis probe only:
    - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_pair2_train_segment_identity_probe.py`
  - no new runtime hook in `sb3_recurrent_ppo.py`
  - no change to train semantics, shared core, or pooled weighting

- Verification:
  - focused regression pack passed:
    - `test_phase3_pair2_train_segment_identity_probe.py`
    - `test_phase3_train_outer_batch_snapshot_hook.py`
  - result: `7 passed`

- Locked reading:
  - source block from the saved env12 artifact:
    - global steps `1964..1967`
    - source objective identity:
      - `objective_episode_index = 0`
      - `objective_local_index = 18..21`
  - current strict pair2 train-side rollout at the same global steps:
    - raw rollout identity:
      - `raw_objective_episode_index = 10`
      - `raw_objective_episode_position = 10..13`
      - `raw_objective_global_position = 18..21`
    - compressed objective identity on the current strict rollout:
      - `compressed_objective_episode_index = 1`
      - `compressed_objective_episode_position = 10..13`
  - current strict rollout objective window for env12 is segmented as:
    - raw episodes `9, 10, 11`
    - compressed objective episodes `0, 1, 2`
    - spans:
      - `1946..1953`
      - `1954..1983`
      - `1984..2047`

- Transport conclusion:
  - `get_gpu_flat` does not introduce the renumbering:
    - flatten stage preserves the rollout raw identity exactly
  - outer batch `104` does not introduce the renumbering:
    - outer batch `104` preserves the flat raw identity exactly
  - therefore the mismatch is already present at rollout stage

- Semantic conclusion:
  - the saved env12 block is not aligned to the current strict train-side episode-local numbering
  - its `18..21` indices align with the current strict rollout's objective-global positions `18..21`, not with current episode-local positions
  - this is why the source block does not survive as `episode 0 / positions 18..21` on the strict train-side path
  - the bottleneck is now narrowed to:
    - source artifact segment identity / reset contract
    - versus current strict rollout segment identity
  - it is no longer a `get_gpu_flat` or outer-batch transport bug hypothesis

- Next step:
  - do not add more batch transport instrumentation
  - if this line continues, only do a read-only source-vs-current env12 reset/segment contrast on the `1946..2002` window to explain why the saved artifact had one long `episode 0` there while current strict collect splits it into shorter segments

## Phase 3 pair2 env12 window reset/segment compare

- Completed a read-only source-vs-current strict compare on the exact env12 window:
  - `global_step = 1946..2002`

- Canonical artifact:
  - `/home/chen/RLPFN/artifacts/phase3_pair2_env12_window_reset_segment_compare.json`

- Implementation scope:
  - reused a standalone analysis probe:
    - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_pair2_env12_window_reset_segment_compare.py`
  - added a focused unit test:
    - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_phase3_pair2_env12_window_reset_segment_compare.py`
  - no change to `sb3_recurrent_ppo.py`
  - no change to collect/train semantics, shared core, or pooled weighting

- Verification:
  - focused regression pack passed:
    - `test_phase3_pair2_env12_window_reset_segment_compare.py`
  - result: `3 passed`

- Locked reading:
  - source artifact contract inside `1946..2002` is one continuous objective segment:
    - `episode 0`
    - `local 0..56`
    - no internal boundary
  - current strict collect on the same env/window is split into three segments:
    - `1946..1953` -> raw `episode 9`, compressed `episode 0`, local `0..7`
    - `1954..1983` -> raw `episode 10`, compressed `episode 1`, local `0..29`
    - `1984..2002` -> raw `episode 11`, compressed `episode 2`, local `0..18`
  - therefore the current strict extra boundaries inside the source window are exactly:
    - `1954`
    - `1984`
  - target proxy block `1964..1967` lands at:
    - source local `18..21`
    - current strict raw/compressed local `10..13`
    - current strict raw global `18..21`
  - the local shift is constant:
    - `+8` source-local minus current-episode-local
    - which exactly equals the length of the first strict split block `1946..1953`

- Root-cause narrowing:
  - the mismatch is not produced by `get_gpu_flat`
  - the mismatch is not produced by outer-batch transport
  - the mismatch is already present in the rollout reset/segment contract
  - more precisely:
    - source artifact local numbering in this window behaves like one long objective segment
    - current strict collect resets episode-local numbering at `1954` and `1984`
    - so the saved source block survives only under objective-global positions, not under current episode-local positions

- Operational rule:
  - do not reuse source `episode 0 / local 18..21` identity as if it were still valid on current strict pair2 runtime
  - any exact block-level train-side audit must first reindex against the current strict reset/segment contract or compare in objective-global coordinates

- Next step:
  - if this line continues, do not reopen pooled weighting or shared-core repair
  - only compare the source artifact generation contract against the current strict collect contract at the actual boundary sources `1954` and `1984`

## Phase 3 pair2 env12 source-vs-strict suite parity correction

- Completed the missing same-suite correction:
  - reran the same window compare on the `pair2 heldout suite`, which is the suite actually used by the source artifact generation chain

- Canonical artifact:
  - `/home/chen/RLPFN/artifacts/phase3_pair2_env12_window_reset_segment_compare_heldout_suite.json`

- Locked reading:
  - source artifact generation chain in `phase3_residual_anchor_gae_path_probe.py` binds to:
    - `heldout_suite_path`
    - not `train_suite_path`
  - the earlier current-strict compare that produced internal boundaries `1954` and `1984` was run on:
    - `/home/chen/RLPFN/artifacts/phase3_cross_env_baseline_pair2/suites/train_suite.pt`
  - but `train suite env12` and `heldout suite env12` are not the same environment:
    - different `env_seed`
    - different `rollout_seed`
    - different sampled `h`
  - after correcting the compare to use the same `heldout suite`, current official strict collect matches the source window contract exactly:
    - `1946..2002` remains one continuous objective segment
    - no internal boundary at `1954`
    - no internal boundary at `1984`
    - target block `1964..1967` stays at local `18..21`

- Corrected root-cause statement:
  - the previously observed `1954/1984` split is not evidence that current strict collect cuts the same source env/window differently
  - it is evidence that we had compared:
    - source artifact on `pair2 heldout env12`
    - against current strict collect on `pair2 train env12`
  - so the earlier mismatch was a suite mismatch, not yet a reset-contract mismatch inside the same environment

- Operational rule:
  - do not compare source env-local coordinates against current strict `train suite env12` when the source chain came from `heldout suite env12`
  - any future source-vs-current contract audit must lock:
    - same suite
    - same env index
    - same suite seed wiring

- Next step:
  - if this line continues, only trace where the analysis path selected `train_suite_path` instead of the source chain's `heldout_suite_path`
  - do not reinterpret `1954/1984` as a runtime reset bug unless it reproduces under the same heldout suite contract

## Phase 3 analysis suite-selection contract lock

- Completed a narrow analysis-only guardrail to prevent mixed-suite source/current compares.

- Scope:
  - updated only:
    - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_pair2_env12_window_reset_segment_compare.py`
    - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_phase3_pair2_env12_window_reset_segment_compare.py`
  - no change to:
    - `sb3_recurrent_ppo.py`
    - rollout semantics
    - shared core
    - pooled weighting

- Contract now locked as:
  - source-side selection is keyed by:
    - `source_suite_name + env_index`
    - not just `env_index`
  - current compare-side suite path is resolved by source artifact contract:
    - `phase3_residual_anchor_gae_path_probe` implies `heldout` compare suite role
  - default canonical run now auto-resolves to:
    - `/home/chen/RLPFN/artifacts/phase3_cross_env_baseline_pair2/suites/heldout_suite.pt`
  - explicit `train_suite.pt` compare is rejected unless the caller opts into an intentional mismatch probe

- Verification:
  - focused regression pack passed:
    - `test_phase3_pair2_env12_window_reset_segment_compare.py`
  - result: `6 passed`
  - runtime contract check passed:
    - default run writes canonical artifact with:
      - `compare_suite_role = heldout`
      - `source_expected_compare_suite_role = heldout`
      - `allow_source_suite_mismatch = false`
    - explicit `train_suite.pt` run now fails fast with a suite-role mismatch error

- Canonical artifact after the lock:
  - `/home/chen/RLPFN/artifacts/phase3_pair2_env12_window_reset_segment_compare.json`
  - this artifact is now same-suite by default, not a mixed-suite compare

- Operational rule:
  - do not trust any source-vs-current env-local compare unless:
    - source suite name is explicit
    - compare suite role matches the source artifact contract
  - mixed-suite compare is now treated as an intentional counterfactual only, not as the default diagnostic path

- Next step:
  - keep this contract fixed
  - if we continue on the root-cause line, only check whether any other analysis probes still silently select suite by `env_index` or by misleading `train_suite_path` naming

## Phase 3 analysis suite-contract audit

- Completed a narrow audit over the remaining analysis probes for the same bug class:
  - source row selection by `env_index` alone on multi-suite payloads
  - misleading suite-path defaults in source-vs-current compare probes

- Canonical audit note:
  - `/home/chen/RLPFN/artifacts/phase3_analysis_suite_contract_audit.md`

- Locked findings:
  - one real vulnerability existed in:
    - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_pair2_env12_window_reset_segment_compare.py`
  - that vulnerability is already fixed and regression-protected
  - audited `token_row_bundles` readers that are safe in this bug class already filter by:
    - `suite_name + env_index`
  - audited safe readers include:
    - `phase3_pair2_flip_mode_split_probe.py`
    - `phase3_future_carry_source_chain_probe.py`
    - `phase3_env12_negative_delta_probe.py`
    - `phase3_env12_value_peak_drop_contract_probe.py`
    - `phase3_env12_local56_boundary_contract_probe.py`
    - `phase3_env12_boundary_stepdown_root_probe.py`
    - `phase3_env12_local56_boundary_scope_counterfactual_probe.py`
    - `phase3_env12_local56_mean_source_probe.py`
  - intentional exception:
    - `phase3_pair2_train_segment_identity_probe.py`
    - this remains a deliberate train-side identity probe, not a same-suite env-local reset-contract compare

- Operational rule:
  - do not reopen this bug class unless a new analysis probe is found to:
    - read a multi-suite source payload by `env_index` alone
    - or silently default a source/current compare to the wrong suite role

- Next step:
  - keep suite-selection contract fixed
  - if root-cause work continues, move to a different bottleneck axis rather than re-auditing this same suite-selection class

## Phase 3 root-cause probe trust matrix

- Completed one analysis-only cleanup step before returning to the optimization bottleneck axis:
  - classified the current pair2 root-cause probes by trust level under the locked suite contract

- Canonical note:
  - `/home/chen/RLPFN/artifacts/phase3_root_cause_probe_trust_matrix.md`

- Locked reading:
  - trusted:
    - `phase3_pair2_proxy_to_train_transfer_audit.json`
      - valid for the train-side mass-dilution conclusion
    - `phase3_pair2_train_segment_identity_probe.json`
      - valid as a train-side identity/transport probe
  - conditionally trusted:
    - `phase3_pair2_exact_update_aggregation_audit.json`
      - valid for batch-level dilution stages
      - not valid for exact env12 block attribution because the requested block is absent from the captured outer batch
  - guardrail-only:
    - `phase3_pair2_env12_window_reset_segment_compare.json`
      - use it to enforce same-suite compare semantics, not as a direct optimization bottleneck artifact

- Root-cause interpretation after the lock:
  - the strongest still-trusted bottleneck axis is now:
    - `proxy_to_train_update_transfer_semantics`
  - meaning:
    - local proxy gain can be real
    - but still fail to survive many-env train-side update aggregation

- Operational rule:
  - do not use the current exact update aggregation artifact as proof about the env12 four-token block
  - do use it as evidence that normalization / clipping are real dilution stages on the captured strict batch

- Next step:
  - if this bottleneck line continues, only do a train-side update-transfer semantics probe on the actually captured strict batch scope
  - do not reopen mixed-suite compare
  - do not retune the branch-specific block itself yet

## Phase 3 pair2 captured-scope update-transfer probe

- Completed a stricter re-interpretation of the current exact update aggregation artifact:
  - target is no longer the absent env12 four-token block
  - target is the actual strict outer-batch scope that was really captured

- Canonical artifact:
  - `/home/chen/RLPFN/artifacts/phase3_pair2_captured_scope_update_transfer_probe.json`

- Scope and contract:
  - implemented in:
    - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_pair2_captured_scope_update_transfer_probe.py`
    - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_phase3_pair2_captured_scope_update_transfer_probe.py`
  - this probe is snapshot-only:
    - no training rerun
    - no runtime semantic change
    - no `sb3_recurrent_ppo.py` change
  - it is only valid when the selected captured scope already equals the whole-batch objective
    - that contract holds for the current strict pair2 exact aggregation artifact

- Locked reading:
  - the current captured strict outer batch contains exactly one objective segment:
    - `env_index = 0`
    - `objective_episode_index = 2`
    - `objective_position = 0..101`
    - `token_count = 102`
  - that single segment already equals the whole-batch objective
  - baseline and override are exactly identical on this captured scope across:
    - `pre_normalization_actor_advantages`
    - `post_normalization_advantages`
    - `policy_num_from_clipped_objective`
    - `ratio`
    - `clip_active_fraction`
  - therefore:
    - the current captured strict batch is not the transfer site for the env12 branch-specific gain
    - this artifact cannot be used to claim in-batch many-env dilution on that batch, because there is no many-env mixture inside the captured objective scope

- Verification:
  - focused regression pack passed:
    - `test_phase3_pair2_captured_scope_update_transfer_probe.py`
  - result:
    - `2 passed`

- Operational rule:
  - do not interpret the current exact update aggregation artifact as evidence that env12 gain was diluted inside the captured batch
  - interpret it as evidence that:
    - the captured batch is a non-target env0 batch
    - the branch-specific override leaves that batch completely unchanged

- Next step:
  - if this bottleneck line continues, the next narrow move is:
    - locate or capture a strict train-side outer batch where the target env/scope is actually present
  - do not retune the block from this artifact
  - do not reopen pooled weighting from this artifact

## Phase 3 pair2 target-present strict train-side outer batch capture

- Completed the next narrow step on the `proxy_to_train_update_transfer_semantics` axis:
  - located one strict train-side outer batch where the env12 target scope is actually present
  - captured the matching outer-batch snapshot directly

- Canonical artifacts:
  - `/home/chen/RLPFN/artifacts/phase3_pair2_target_present_outer_batch_selector_trace.json`
  - `/home/chen/RLPFN/artifacts/phase3_pair2_target_present_outer_batch_snapshot.json`

- Contract used:
  - strict fixed-env mode
  - deterministic actor sampling
  - deterministic batch plan
  - runtime suite name:
    - `pair2`
  - target selector was updated from the stale source identity
    - old source-local selector:
      - `env12 / episode0 / pos18..21`
    - current strict train-side raw identity:
      - `env12 / raw episode10 / pos10..13`
      - `outer_batch_idx = 104`

- Locked reading:
  - selector trace confirms one real match:
    - `target_appeared = true`
    - first match:
      - `epoch_idx = 1`
      - `outer_batch_idx = 104`
      - `target_match_count = 4`
  - captured snapshot confirms the target block is physically present in that batch:
    - snapshot summary:
      - `outer_batch_idx = 104`
      - `objective_total = 102`
      - `actor_objective_mode = tokenwise`
      - `runtime_current_suite_name = pair2`
    - target selector in snapshot:
      - `target_env_index = 12`
      - `target_objective_episode_index = 10`
      - `target_objective_position = 10..13`
      - `target_outer_batch_idx = 104`
    - snapshot target match count:
      - `4`
    - matched rows carry:
      - `raw_objective_global_position = 18..21`

- Runtime note:
  - the full audit command kept running after the snapshot and selector trace had already been written
  - for this task, those two artifacts were sufficient
  - the background process was stopped after confirming the files existed and the selector/snapshot contracts matched
  - this avoided spending additional runtime on post-capture work not needed for the current step
  - important contract caveat:
    - this capture reused the current pair2 many-env audit contract
    - so it ran with:
      - `n_samples = 2048`
      - `single_eval_pos = 1946`
      - `outer_epochs = 1`
    - that is not the trusted Phase 2 runtime reference contract
    - therefore this capture is valid for:
      - target-present identity localization
      - target-present snapshot capture
    - but it is not valid for:
      - Phase 2-equivalent runtime comparison
      - any claim about same-order training time versus the trusted Phase 2 pack

- Trusted Phase 2 runtime reference:
  - `/home/chen/RLPFN/artifacts/phase2_regression_pack/phase2_regression_suite_summary.json`
  - locked contract:
    - `n_steps = 256`
    - `batch_size = 256`
    - `outer_epochs = 2`
    - `single_eval_pos = 64`
    - `strict_fixed_env_mode = true`
    - `deterministic_actor_sampling = true`
    - `deterministic_batch_plan = true`

- Operational rule:
  - do not keep using the old `episode0 / pos18..21` selector for strict train-side capture
  - for the current pair2 strict path, the target-present train-side identity is:
    - `outer_batch 104 / env12 / raw episode10 / pos10..13`
  - do not use this `2048`-sample capture as a runtime-scale reference against Phase 2
  - any runtime-sensitive Phase 3 validation must be re-expressed in Phase 2-equivalent scale first

- Next step:
  - with the target-present snapshot now available, the next narrow move can be:
    - exact update-aggregation audit on this captured target-present batch
  - do not reopen pooled weighting
  - do not retune the branch-specific block itself from this step alone

## Phase 3 pair2 Phase-2-scale target-present capture

- Re-ran the target-present capture under the trusted Phase 2-equivalent runtime contract instead of the larger pair2 many-env audit scale.

- Canonical artifacts:
  - `/home/chen/RLPFN/artifacts/phase3_pair2_phase2_scale_target_present_capture.json`
  - `/home/chen/RLPFN/artifacts/phase3_pair2_phase2_scale_target_present_selector_trace.json`

- Locked contract:
  - `n_steps = 256`
  - `batch_size = 256`
  - `outer_epochs = 2`
  - `single_eval_pos = 64`
  - `strict_fixed_env_mode = true`
  - `deterministic_actor_sampling = true`
  - `deterministic_batch_plan = true`
  - no extra zero/pre/heldout eval was included in this probe path

- Locked reading:
  - this Phase-2-scale run does not preserve the previous target-present hit relationship
  - selector trace result:
    - `target_appeared = false`
    - `row_count = 16`
    - no target-present outer batch exists for:
      - `env12 / objective_episode_index 10 / objective_position 10..13`
  - the only batch containing env12 objective tokens is:
    - `outer_batch_idx = 13`
  - within that batch:
    - env12 objective episodes present are `8..18`
    - target episode `10` exists
    - but only positions `0..7` exist
    - so the failure mode is:
      - `target_status = target_position_range_absent`
  - therefore:
    - under Phase 2-equivalent scale, the old current-strict target selector does not reappear
    - no snapshot is written

- Runtime reading:
  - elapsed wall time:
    - `24.02s`
  - last update wall time:
    - `5.63s`
  - this is back in a Phase-2-like order of magnitude, unlike the earlier `2048 / 1946` many-env audit path

- Operational rule:
  - do not assume the target-present selector from the large pair2 many-env contract survives Phase 2-equivalent scaling
  - under trusted Phase 2-equivalent scale, treat:
    - `env12 / episode10 / pos10..13`
    as absent

- Next step:
  - if this line continues, the next narrow move is not a new runtime repair
  - it is to re-identify the Phase-2-scale train-side target selector first, then decide whether an exact aggregation audit is still meaningful at trusted scale

## Phase 3 pair2 Phase-2-scale selector reidentification

- Completed the next trusted-scale analysis step:
  - reidentified the env12 train-side selector directly from the Phase-2-scale selector trace
  - no new training run was needed for this step

- Canonical artifact:
  - `/home/chen/RLPFN/artifacts/phase3_pair2_phase2_scale_selector_reidentified.json`

- Implemented in:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_pair2_phase2_scale_selector_reidentify.py`
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_phase3_pair2_phase2_scale_selector_reidentify.py`

- Locked reading:
  - the stale requested trusted-scale selector was:
    - `env12 / objective_episode_index 10 / position 10..13`
  - it is absent under the trusted Phase-2-scale trace
  - the closest present selector that preserves env and episode identity is:
    - `outer_batch_idx = 13`
    - `env_index = 12`
    - `objective_episode_index = 10`
    - `objective_position = 0..7`
    - `token_count = 8`
  - so the trusted-scale difference is not:
    - env mismatch
    - episode mismatch
  - it is:
    - position-range shrink

- Verification:
  - focused regression pack passed:
    - `test_phase3_pair2_phase2_scale_selector_reidentify.py`
  - result:
    - `2 passed`

- Operational rule:
  - do not carry the larger-contract selector `episode10 / pos10..13` into the trusted Phase-2-scale path
  - trusted Phase-2-scale selector is now:
    - `outer_batch13 / env12 / episode10 / pos0..7`

- Next step:
  - if this line continues, the next narrow move is:
    - Phase-2-scale target-present recapture using the reidentified selector
  - only after that should exact update aggregation be retried at trusted scale

## Phase 3 pair2 Phase-2-scale trusted selector recapture

- Re-ran the Phase-2-scale target-present capture using the reidentified trusted selector instead of the stale larger-contract selector.

- Artifacts:
  - `/home/chen/RLPFN/artifacts/phase3_pair2_phase2_scale_target_present_recapture.json`
  - `/home/chen/RLPFN/artifacts/phase3_pair2_phase2_scale_target_present_recapture_selector_trace.json`
  - `/home/chen/RLPFN/artifacts/phase3_pair2_phase2_scale_target_present_recapture_snapshot.json`
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_pair2_phase2_scale_target_present_capture.py`
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_phase3_pair2_phase2_scale_target_present_capture.py`

- Trusted selector used:
  - `outer_batch_idx = 13`
  - `env_index = 12`
  - `objective_episode_index = 10`
  - `objective_position = 0..7`

- Capture result:
  - `target_appeared = true`
  - first match:
    - `epoch_idx = 1`
    - `outer_batch_idx = 13`
    - `target_match_count = 8`
  - snapshot write:
    - `snapshot_written = true`
    - `snapshot_target_match_count = 8`
    - `snapshot_objective_total = 192`

- Snapshot-level trusted readout:
  - target rows are exactly:
    - `env12 / objective_episode_index 10 / episode_pos 0..7`
  - corresponding objective-global positions are:
    - `22..29`
  - target rows are all objective-active:
    - `objective_mask = 1`
  - target rows carry negative actor mass in this captured trusted-scale batch:
    - pre-normalization advantages are all negative
    - post-normalization advantages are all negative
    - clipped objectives are all negative

- Runtime:
  - full recapture wall time:
    - `23.308994076913223 s`
  - last update wall time:
    - `5.456477575004101 s`
  - this remains same-order as the trusted Phase 2 single-collect timing reference and does not indicate a new runtime anomaly by itself

- Verification:
  - focused regression pack passed:
    - `test_phase3_pair2_phase2_scale_target_present_capture.py`
  - result:
    - `2 passed`

- Locked conclusion:
  - the trusted Phase-2-scale train-side target-present fixed point now exists and is captured
  - do not use the stale larger-contract selector `env12 / episode10 / pos10..13`
  - use the trusted selector and snapshot instead:
    - `outer_batch13 / env12 / episode10 / pos0..7 / global22..29`

- Next step:
  - if this line continues, the next narrow move is:
    - exact trusted-scale update-aggregation audit on this captured scope
  - that audit should stay read-only and operate on the saved trusted-scale snapshot, not on a larger-contract batch

## Phase 3 pair2 Phase-2-scale exact update aggregation audit

- Built a read-only exact update aggregation audit directly from the trusted-scale target-present snapshot.

- Artifacts:
  - `/home/chen/RLPFN/artifacts/phase3_pair2_phase2_scale_exact_update_aggregation_audit.json`
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_pair2_phase2_scale_exact_update_aggregation_audit.py`
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_phase3_pair2_phase2_scale_exact_update_aggregation_audit.py`

- Trusted target scope audited:
  - `outer_batch13 / env12 / episode10 / pos0..7 / global22..29`
  - `token_count = 8`

- Exact stage readout:
  - target negative mass:
    - pre-normalization:
      - `7.032025098800659`
    - post-normalization:
      - `14.139300107955933`
    - policy-num-from-clipped-objective:
      - `13.329894781112671`
  - stage deltas:
    - normalization delta:
      - `+7.107275009155273`
    - ratio/clipping delta:
      - `-0.8094053268432617`
  - retention:
    - `post_over_pre_retention = 2.0107010298310013`
    - `policy_over_post_retention = 0.9427549227569034`
    - `policy_over_pre_retention = 1.8955982940655516`

- Share readout:
  - target share of whole-batch negative mass:
    - pre:
      - `0.17903176345129235`
    - post:
      - `0.17812002157705045`
    - policy-num:
      - `0.17802543705133486`
  - target share of env12 negative mass:
    - pre:
      - `0.17903176345129235`
    - post:
      - `0.17812002157705045`
    - policy-num:
      - `0.17802543705133486`

- Ratio/clipping readout:
  - target scope:
    - `clip_active_fraction = 1.0`
    - `ratio_mean = 0.9427475333213806`
  - whole batch:
    - `clip_active_fraction = 0.6822916666666666`
    - `ratio_mean = 0.9429953281457225`

- Locked interpretation:
  - this captured trusted-scale outer batch is not a many-env mixture at the objective-token level:
    - whole-batch and env12 stage metrics are identical
    - `whole_batch.token_count = env12.token_count = 192`
  - therefore this batch cannot be used as evidence for cross-env dilution
  - inside this exact trusted batch:
    - normalization amplifies the target block's negative mass strongly
    - ratio/clipping removes only a small fraction of that amplified negative mass
    - target block remains a material share of the batch negative mass:
      - about `17.8%` after clipping

- Verification:
  - focused regression pack passed:
    - `test_phase3_pair2_phase2_scale_exact_update_aggregation_audit.py`
  - result:
    - `2 passed`

- Next step:
  - if this line continues, the next narrow move is:
    - interpret why this trusted target scope still fails to convert into train-side gain even though it is not being diluted by cross-env mixing inside the captured batch
  - that next step should remain read-only and should compare:
    - local target block sign/mass
    - whole-episode sign/mass
    - epoch-level or cross-batch aggregation context

## Phase 3 pair2 Phase-2-scale batch-to-epoch aggregation probe

- Built a read-only batch-to-epoch aggregation probe to answer the narrower question:
  - if the trusted target block is not weak inside its own batch, why does it still fail to become train-side gain?

- Artifacts:
  - `/home/chen/RLPFN/artifacts/phase3_pair2_phase2_scale_batch_to_epoch_aggregation_probe.json`
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_pair2_phase2_scale_batch_to_epoch_aggregation_probe.py`
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_phase3_pair2_phase2_scale_batch_to_epoch_aggregation_probe.py`

- Locked epoch layout:
  - total outer batches in the trusted-scale epoch:
    - `16`
  - objective layout:
    - every outer batch has exactly one env:
      - `singleton_env_outer_batches = true`
    - every outer batch has the same objective token count:
      - `objective_total_per_outer_batch = 192`
    - outer-batch env order is fixed:
      - `env0, env1, ..., env15`
  - target block batch:
    - `outer_batch13 = env12`

- Locked train-loop contract:
  - `optimizer_step_per_outer_batch = true`
  - `outer_batch_loss_normalized_by_that_batch_objective_total = true`
  - `sequential_outer_batch_updates_within_epoch = true`

- Target scope epoch context:
  - target scope hits:
    - exactly one outer batch per epoch
  - target batch position in epoch:
    - `12` batches before
    - `3` batches after
  - target step fraction of epoch:
    - `1 / 16 = 0.0625`
  - target share of its own batch negative mass after clipping:
    - `0.17802543705133486`
  - uniform-step-weight heuristic for target influence at epoch scale:
    - `0.011126589815708429`
    - this is only a heuristic compression of:
      - `target_batch_share * 1/16`
    - it is not an exact gradient share claim

- Locked interpretation:
  - the trusted target block is not failing because it is mixed with other envs inside the same batch
  - the more plausible bottleneck is:
    - cross-batch overwrite across sequential optimizer steps
  - in the trusted Phase-2-scale path, env12 contributes one single-env optimizer step inside a 16-step epoch
  - therefore the correct next explanation line is:
    - not intra-batch dilution
    - but whether the local env12 update is later counteracted by subsequent outer-batch steps

- Verification:
  - focused regression pack passed:
    - `test_phase3_pair2_phase2_scale_batch_to_epoch_aggregation_probe.py`
  - result:
    - `2 passed`

- Next step:
  - if this line continues, the next narrow move is:
    - a read-only cross-batch overwrite probe
  - that probe should compare:
    - the target env12 step
    - the immediately following outer batches
    - whether later sequential steps plausibly reverse the same local direction

## Phase 3 pair2 Phase-2-scale cross-batch overwrite probe

- Built a read-only sequential actor-only overwrite probe on the trusted Phase-2-scale pair2 runtime.

- Artifacts:
  - `/home/chen/RLPFN/artifacts/phase3_pair2_phase2_scale_cross_batch_overwrite_probe.json`
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_pair2_phase2_scale_cross_batch_overwrite_probe.py`
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_phase3_pair2_phase2_scale_cross_batch_overwrite_probe.py`

- Probe contract:
  - collect one trusted Phase-2-scale pair2 rollout
  - keep the rollout fixed
  - replay the training order sequentially
  - simulate actor-only optimizer steps through outer batches `1..16`
  - inspect actual parameter-step deltas for:
    - target batch `13`
    - downstream batches `14..16`
  - this is read-only:
    - no checkpoint writes
    - no training-path code changes

- Runtime:
  - wall time:
    - `23.347357381135225 s`
  - same order as other trusted-scale probes
  - no new runtime anomaly observed

- Exact overwrite result:
  - target batch:
    - `outer_batch13 = env12`
    - `delta_norm = 0.008737072348594666`
  - downstream pairwise alignment vs target:
    - `outer_batch14 = env13`
      - `cosine = 0.7779969016740984`
      - `projection_on_target = 0.6548209835704135`
    - `outer_batch15 = env14`
      - `cosine = 0.85268642774419`
      - `projection_on_target = 0.668908968923087`
    - `outer_batch16 = env15`
      - `cosine = 0.8921939052438096`
      - `projection_on_target = 0.6320691143122877`
  - downstream cumulative alignment vs target:
    - `cosine = 0.8566248050119049`
    - `projection_on_target = 1.955839623062271`
  - net target retention after applying downstream `14..16`:
    - `projection_on_target = 2.9583765580495096`

- Locked interpretation:
  - the immediate actor-only downstream steps do **not** overwrite the env12 target step
  - they are strongly aligned with it
  - therefore the current `14..16 overwrite` hypothesis is falsified on the trusted actor-only line
  - this means the failure to convert local proxy gain into train-side gain is **not** explained by:
    - intra-batch cross-env mixing
    - immediate downstream actor-only overwrite from `outer_batch14..16`

- Verification:
  - focused regression pack passed:
    - `test_phase3_pair2_phase2_scale_cross_batch_overwrite_probe.py`
  - result:
    - `2 passed`

- Next step:
  - if this line continues, the next narrow move should be:
    - compare the trusted target step under:
      - actor-only update
      - full PPO main-loss update
  - reason:
    - the current overwrite hypothesis is false on the actor-only line
    - so the next plausible narrow bottleneck is:
      - shared-core/full-loss interference
      - not later actor-only batches

## Phase 3 pair2 Phase-2-scale local step direction probe

- Built a read-only local-step branch compare on the trusted `outer_batch13 = env12` step.

- Artifacts:
  - `/home/chen/RLPFN/artifacts/phase3_pair2_phase2_scale_local_step_direction_probe.json`
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_pair2_phase2_scale_local_step_direction_probe.py`
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_phase3_pair2_phase2_scale_local_step_direction_probe.py`

- Probe contract:
  - collect one trusted Phase-2-scale pair2 rollout
  - replay `outer_batch1..12` with full main-loss to reconstruct the real pre-`batch13` state
  - branch on the same prestate:
    - actor-only step on `batch13`
    - `policy + value (+ entropy if enabled)` main-loss step on `batch13`
  - current compare scope explicitly excludes aux losses
  - this matters because current trusted runtime still has:
    - `aux_flow_weight = 0.2`

- Runtime:
  - wall time:
    - `27.70612142700702 s`
  - same order as the other trusted-scale probes
  - no new runtime anomaly observed

- Exact local-step compare:
  - target batch:
    - `outer_batch13 = env12`
    - `objective_total = 192`
  - actor-only:
    - `policy_loss = 0.00015528748917859048`
    - `value_loss = 1.149062991142273`
    - `grad_norm_pre_clip_flat = 0.11291873455047607`
    - `delta_norm = 0.0360880121588707`
  - full main-loss:
    - `policy_loss = 0.00015528748917859048`
    - `value_loss = 1.149062991142273`
    - `vf_coef = 0.5`
    - `ent_coef = 0.0`
    - `grad_norm_pre_clip_flat = 6.695580005645752`
    - `delta_norm = 0.04133513942360878`

- Direction readout:
  - actor vs full gradient:
    - `cosine = 0.017400420572603232`
    - full gradient norm is about `59x` the actor-only gradient norm
  - actor vs full parameter delta:
    - `cosine = 0.8892663378371908`
    - `projection_on_actor = 1.0185639457613322`
  - residual (`full - actor`) vs actor delta:
    - `cosine = 0.031602329955936204`
    - `projection_on_actor = 0.016680717798082487`

- Locked interpretation:
  - this does **not** support a simple opposite-direction shared-core interference story
  - the local full main-loss step is not anti-aligned with the actor-only step
  - however, the full main-loss gradient is dominated by a very large non-actor component that is mostly orthogonal to the actor gradient
  - because:
    - `ent_coef = 0.0`
    - compare scope excludes aux losses
  - the non-actor component in this probe is effectively the shared value-path contribution
  - therefore the current trusted local-step evidence is:
    - not "value pushes opposite to actor"
    - but "value injects a large mostly orthogonal shared-core gradient"

- Important caveat:
  - the actual trusted runtime still has `aux_flow_weight = 0.2`
  - so this probe does **not** yet close the auxiliary shared-core line

- Verification:
  - focused regression pack passed:
    - `test_phase3_pair2_phase2_scale_local_step_direction_probe.py`
  - result:
    - `2 passed`

- Next step:
  - if this line continues, the next narrow move should be:
    - compare `main-loss` vs `main-loss + aux-flow` on the same trusted local step
  - reason:
    - immediate actor-only overwrite is already falsified
    - simple opposite-direction value interference is not supported
    - the remaining unchecked shared-core branch in the trusted runtime is the auxiliary flow path

2026-04-14 Phase 3 contract correction: lock all trusted many-env work to the Phase 2 shared-backbone baseline

- Problem:
  - the previous Phase 3 "trusted" line was not actually contract-equivalent to the only trusted Phase 2 milestone
  - concrete drift:
    - `train_profile = trusted_sep_reset_mainline`
    - `ppo_normalize_advantage = True`
    - runtime q/flow overrides left as `None`
    - several `phase2_scale` wrappers still built env cfg via `_resolve_audit_env_config(...)`
  - this means earlier "Phase-2-scale" pair2 probes were collected under a contaminated contract

- Locked code changes:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_multi_env_optimization_audit.py`
    - default `train_profile` changed to `phase2_shared_backbone_contract`
    - CLI default `--train-profile` changed to `phase2_shared_backbone_contract`
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_pair2_phase2_scale_target_present_capture.py`
    - switched from `_resolve_audit_env_config(...)` to `_build_audit_env_cfg(...)`
    - switched profile from `trusted_sep_reset_mainline` to `phase2_shared_backbone_contract`
    - artifact config now records:
      - `train_profile`
      - `ppo_normalize_advantage`
      - runtime q/flow overrides
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_pair2_phase2_scale_cross_batch_overwrite_probe.py`
    - same contract correction as above
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_pair2_phase2_scale_local_step_direction_probe.py`
    - same contract correction as above
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_pair2_train_update_ab_compare_pack.py`
    - pair2 target-side A/B now calls the audit with `train_profile = phase2_shared_backbone_contract`

- Locked test coverage:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_phase3_multi_env_optimization_audit.py`
    - added:
      - `test_resolve_train_profile_phase2_shared_backbone_contract_matches_phase2_milestone`
      - `test_run_phase3_audit_phase2_shared_backbone_contract_uses_phase2_env_cfg`
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_phase3_pair2_train_update_ab_compare_pack.py`
    - updated fixture contract strings to `phase2_shared_backbone_contract`

- Verification:
  - `python -m py_compile` passed for the touched analysis/test files
  - `conda run -n rlpfn python -m pytest reinforce-terminal-explore/ticl/tests/test_phase3_multi_env_optimization_audit.py reinforce-terminal-explore/ticl/tests/test_phase3_pair2_train_update_ab_compare_pack.py -q`
  - result:
    - `21 passed`

- Operational consequence:
  - from now on, the only acceptable Phase 3 baseline is:
    - Phase 2 shared actor-critic backbone logic
    - plus many-env
    - plus the narrow adjustment under test
    - and nothing else
  - do not treat previously generated `phase3_pair2_phase2_scale_*` JSON artifacts as trusted milestone evidence until they are re-run under the corrected contract

- Next step:
  - re-run exactly one many-env target-side A/B under `phase2_shared_backbone_contract`
  - compare:
    - baseline: shared-backbone many-env with no narrow adjustment
    - narrow-adjusted: same contract plus only the narrow adjustment
  - do not add actor-only, aux-flow, pooled-weighting, or other side branches into that baseline

2026-04-14 Phase 2 milestone vs current Phase 3 baseline gradient-level contract compare

- New artifact:
  - `/home/chen/RLPFN/artifacts/phase3_phase2_milestone_contract_gradient_compare.json`
- New code:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_phase2_milestone_contract_gradient_compare.py`
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_phase3_phase2_milestone_contract_gradient_compare.py`

- Purpose:
  - prove whether the current Phase 3 baseline is truly the Phase 2 shared-backbone milestone logic, after collapsing back to:
    - one env
    - one fixed frozen_h
    - same train env seed / rollout seed
    - same PPO hyperparameters
  - only if this compare is clean should many-env + narrow-adjustment A/B be interpreted as isolated

- Locked results:
  - `signature_diff = {}`
  - initial policy state now matches exactly after seeding the two builds identically:
    - `policy_param_max_abs_diff = 0.0`
  - main-loss / rollout / gradient compare is extremely close, but not exact:
    - max rollout tensor diffs:
      - `actions = 5.4867e-05`
      - `old_values = 0.0019040`
      - `returns = 2.0778e-04`
      - `advantages = 0.0019382`
    - scalar diffs:
      - `policy_loss = 5.3942e-06`
      - `total_loss = 5.3644e-06`
      - `value_loss = 0.0`
    - gradient diffs:
      - `max_abs_grad_delta = 5.4240e-06`
      - `l2_delta_norm = 6.3272e-05`
      - `cosine_similarity = 1.0001656`

- Interpretation:
  - the previous large mismatch was not a Phase 2/Phase 3 contract drift in logic; it was just unsynchronized random initialization of PPO heads during the compare harness
  - after fixing that, the remaining drift is very small
  - but it is still nonzero, so the current codebase is not yet proven to be bitwise-identical to the milestone path
  - because:
    - signature is exact
    - initial policy state is exact
    - observations / masks / old log-prob already match exactly
  - the remaining live suspect is now narrow:
    - one-env fixed-suite collect/reset transport
    - i.e. the difference between the milestone single-env reset path and the Phase 3 `_bind_vec_env_to_fixed_suite(...)` path

- Decision:
  - do not yet trust many-env target-side A/B as fully isolated
  - first close this tiny residual drift

- Next step:
  - do one read-only one-env fixed-suite collect drift probe
  - compare the milestone path and the Phase 3 fixed-suite path at the earliest possible point:
    - reset output
    - first rollout step actor outputs
    - first value prediction / action tensor
  - the goal is to attribute the residual `~1e-3 rollout / ~5e-6 gradient` drift to a single remaining runtime path difference

2026-04-14 One-env fixed-suite collect drift probe

- New artifact:
  - `/home/chen/RLPFN/artifacts/phase3_phase2_one_env_fixed_suite_collect_drift_probe.json`
- New code:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_phase2_one_env_fixed_suite_collect_drift_probe.py`
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_phase3_phase2_one_env_fixed_suite_collect_drift_probe.py`

- Purpose:
  - localize the remaining Phase 2 milestone vs current Phase 3 baseline drift at the earliest collect-relevant stage
  - compare, under the trusted one-env contract:
    - `vec_env.reset()` outputs
    - first strict collect policy-step inputs
    - first-step `obs_full`
    - first-step actor outputs
    - first stored rollout-buffer step

- Locked results:
  - signature and initial policy state remain exact:
    - `signature_diff = {}`
    - `policy_param_max_abs_diff = 0.0`
  - `vec_env.reset()` does drift numerically:
    - reset obs max diff: `0.2776488`
    - internal `_state_t` max diff: `0.2776488`
    - internal `_action_t` max diff: `0.1761909`
  - but that reset drift is not the first collect-relevant divergence:
    - first policy-step scalar/tensor inputs are exact:
      - `obs_t = 0.0`
      - `action_t = 0.0`
      - `reward_t = 0.0`
      - `reward_mask_t = 0.0`
    - first policy-step cache input diff: `0.0`
    - all captured `env_info` tensor leaves are exact across `29` leaves:
      - `first_policy_step_env_info_all_tensor_max_abs_diff = 0.0`
    - `_build_obs_from_rollout_step_inputs` output is exact:
      - `first_policy_step_obs_full_max_abs_diff = 0.0`
  - the earliest collect-relevant divergence is therefore inside the first policy forward path itself:
    - `drift_localization.stage = model_forward_path`
    - first actor-output diffs:
      - `action_mean = 3.3885e-05`
      - `value_logits = 0.00362635`
      - `values = 1.3649e-04`
    - first stored rollout step then inherits that same scale:
      - `actions = 3.3885e-05`
      - `values = 1.3649e-04`
      - `rewards = 1.0595e-05`
      - `next_states = 1.4305e-05`

- Interpretation:
  - the current residual milestone drift is no longer best explained as:
    - Phase 3 profile mismatch
    - initialization mismatch
    - fixed-suite env binding mismatch at the visible reset/input layer
  - it is now localized more narrowly:
    - strict collect reaches the first policy step with exact visible inputs and exact `obs_full`
    - but actor outputs already differ at that point
  - so the live suspect is now the first policy forward path itself:
    - token encode / RWKV core forward_step / latent heads / distribution-value heads
  - the `vec_env.reset()` mismatch is real, but it is currently a side path for this compare, not the earliest proven source of the collect drift

- Verification:
  - `python -m py_compile reinforce-terminal-explore/ticl/analysis/phase3_phase2_one_env_fixed_suite_collect_drift_probe.py reinforce-terminal-explore/ticl/tests/test_phase3_phase2_one_env_fixed_suite_collect_drift_probe.py`
  - `conda run -n rlpfn python -m pytest reinforce-terminal-explore/ticl/tests/test_phase3_phase2_one_env_fixed_suite_collect_drift_probe.py -q`
  - result:
    - `4 passed`

- Decision:
  - still do not start trusted many-env narrow-adjustment A/B
  - the remaining baseline drift is now narrower than reset/input transport, but not yet closed

- Next step:
  - keep the trusted one-env contract fixed
  - do one more read-only first-step forward-subpath probe
  - directly compare inside the first policy step:
    - encoded token
    - RWKV hidden state after `forward_step`
    - `latent_pi`
    - `latent_vf`
    - action-distribution mean
    - value logits
  - the goal is to reduce the remaining drift from `model_forward_path` to one concrete internal subpath

2026-04-15 One-env first-step forward-subpath probe

- New artifact:
  - `/home/chen/RLPFN/artifacts/phase3_phase2_one_env_first_step_forward_subpath_probe.json`
- New code:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_phase2_one_env_first_step_forward_subpath_probe.py`
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_phase3_phase2_one_env_first_step_forward_subpath_probe.py`

- Method:
  - same trusted one-env contract as the milestone compare
  - no training-path edits
  - no second-pass recomputation
  - instead, the probe wraps the real first call inside `make_vectorized_rollout_step_fn()` and traces the actual first-call subpath:
    - `_encode_train_token`
    - `_cast_token_for_rwkv_core`
    - `rwkv_core.forward_step`
    - `mlp_extractor.forward_actor`
    - `mlp_extractor.forward_critic`
    - `_get_action_dist_from_latent`
    - `_value_logits_from_latent`

- Locked results:
  - upstream tokenization is exact:
    - `encoded_token max diff = 0.0`
    - `rwkv_input_token max diff = 0.0`
  - the first actual divergence is now localized to the RWKV core step itself:
    - `forward_subpath_localization.stage = rwkv_forward_step_path`
    - `rwkv_forward_hidden max diff = 0.015625`
  - downstream tensors inherit the same first-call drift:
    - `latent_pi max diff = 0.015625`
    - `latent_vf max diff = 0.015625`
    - `action_distribution_mean max diff = 3.3885e-05`
    - `value_logits max diff = 0.00362635`
  - the returned step outputs match those downstream diffs exactly:
    - `returned_action_mean max diff = 3.3885e-05`
    - `returned_value_logits max diff = 0.00362635`
    - `returned_values max diff = 1.3649e-04`

- Interpretation:
  - the previous “recomputed forward subpath is exact” reading was methodologically too weak because it measured a second-pass replay after the first call
  - after switching to real first-call tracing, the residual trusted drift is no longer ambiguous:
    - not env suite/profile drift
    - not token encoding drift
    - not token cast drift
    - first proven split is inside `rwkv_core.forward_step`
  - this is the narrowest defended localization so far for the remaining milestone-vs-current baseline mismatch

- Verification:
  - `python -m py_compile reinforce-terminal-explore/ticl/analysis/phase3_phase2_one_env_first_step_forward_subpath_probe.py reinforce-terminal-explore/ticl/tests/test_phase3_phase2_one_env_first_step_forward_subpath_probe.py`
  - `conda run -n rlpfn python -m pytest reinforce-terminal-explore/ticl/tests/test_phase3_phase2_one_env_first_step_forward_subpath_probe.py -q`
  - result:
    - `3 passed`

- Decision:
  - still do not start trusted many-env narrow-adjustment A/B
  - the live blocker is now specifically the first-call RWKV core step under the current baseline path

- Next step:
  - stay on the same trusted one-env contract
  - do one read-only RWKV-core step probe
  - compare only what can explain a `forward_step` split despite exact input token:
    - input token dtype / cast dtype
    - actor cache tensors just before `forward_step`
    - `rwkv_core.forward_step` output dtype
    - if possible, layerwise hidden/state outputs inside the core
  - the goal is to decide whether the remaining drift is:
    - cache-state construction
    - dtype / precision path
    - or the core kernel path itself

2026-04-15 One-env RWKV core-step cache/dtype/kernel probe

- New artifact:
  - `/home/chen/RLPFN/artifacts/phase3_phase2_one_env_rwkv_core_step_probe.json`
- New code:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_phase2_one_env_rwkv_core_step_probe.py`
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_phase3_phase2_one_env_rwkv_core_step_probe.py`

- Purpose:
  - keep the trusted one-env Phase 2 contract fixed
  - explain the remaining `~1e-3` rollout / `~5e-6` gradient drift before any trusted many-env A/B
  - isolate whether the first `rwkv_core.forward_step` split is caused by:
    - core input token
    - init/cache state
    - official eval flat-state conversion
    - official eval weight snapshot
    - native core replay
    - actual official eval kernel path

- Locked results:
  - runtime was normal for this probe:
    - `elapsed_wall_time_sec = 17.2676`
    - this is below the Phase 2 standard-test runtime scale of about `103s`, so this run does not indicate a hang or phase-level runtime anomaly
  - signature and initial policy state remain exact:
    - `signature_diff = {}`
    - `policy_param_max_abs_diff = 0.0`
  - the real first RWKV core call uses the same path flags:
    - `path_flags_exact_match = true`
    - `batch_size = 1`
    - `token_dtype = torch.float32`
    - `core_training = false`
    - `torch_grad_enabled = false`
    - `uses_official_eval_path = true`
    - `official_eval_core_cached_before = false`
    - `official_eval_core_cached_after = true`
  - all upstream core inputs and official-eval materialization checks are exact:
    - `token_arg_max_abs_diff = 0.0`
    - `init_state_max_abs_diff = 0.0`
    - `state_arg_max_abs_diff = 0.0`
    - `official_flat_state_input_max_abs_diff = 0.0`
    - `official_weight_snapshot_max_abs_diff = 0.0`
  - native core replay is exact across Phase 2 and current Phase 3 contract:
    - `native_replay_hidden_max_abs_diff = 0.0`
  - actual first official eval `forward_step` output is not exact:
    - `actual_hidden_max_abs_diff = 0.015625`
  - localization:
    - `core_step_localization.stage = official_eval_kernel_path`
    - reason: `token/state/weights/native replay match, but actual official eval forward_step differs`

- Interpretation:
  - this closes several prior suspects:
    - not config/profile drift
    - not policy initialization drift
    - not visible reset/input transport drift
    - not tokenization or cast drift
    - not cache-state construction drift
    - not official eval weight snapshot drift
    - not native PyTorch RWKV core replay drift
  - the remaining trusted one-env baseline mismatch is now isolated to the CUDA official eval shortcut used by `rwkv_core.forward_step`
  - this means the next control is not an actor objective repair and not a Phase 3 many-env weighting change
  - it is a deterministic runtime-contract question:
    - whether the trusted baseline should use the native core path for the milestone-compatible single-env rollout
    - or whether the official eval shortcut needs its own numeric guardrail / bypass under strict regression mode

- Verification:
  - `python -m py_compile reinforce-terminal-explore/ticl/analysis/phase3_phase2_one_env_rwkv_core_step_probe.py reinforce-terminal-explore/ticl/tests/test_phase3_phase2_one_env_rwkv_core_step_probe.py`
  - `conda run -n rlpfn python -m pytest reinforce-terminal-explore/ticl/tests/test_phase3_phase2_one_env_rwkv_core_step_probe.py -q`
  - result:
    - `3 passed`
  - final probe command:
    - `conda run -n rlpfn python reinforce-terminal-explore/ticl/analysis/phase3_phase2_one_env_rwkv_core_step_probe.py --output-json /home/chen/RLPFN/artifacts/phase3_phase2_one_env_rwkv_core_step_probe.json`

- Decision:
  - still do not start trusted many-env narrow-adjustment A/B
  - the remaining Phase 2 vs current one-env drift is small but real and now has a single defended frontier:
    - actual official eval kernel path
  - no training semantics were changed by this probe

- Next step:
  - do exactly one read-only official-eval-vs-native milestone contract probe
  - under the same one-env trusted contract, force or simulate the native RWKV core path for the current baseline and compare whether:
    - first-step hidden drift closes from `0.015625` to `0.0`
    - rollout/value drift shrinks from the previous `~1e-3` scale
    - gradient drift shrinks from `5.4240e-06`
  - if this closes the mismatch, the smallest safe control is a strict-regression guardrail or bypass for the official eval shortcut, not any Phase 3 objective change

2026-04-15 Official-eval vs native milestone contract probe

- New artifact:
  - `/home/chen/RLPFN/artifacts/phase3_phase2_official_eval_vs_native_contract_probe.json`
- New code:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_phase2_official_eval_vs_native_contract_probe.py`
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_phase3_phase2_official_eval_vs_native_contract_probe.py`

- Purpose:
  - test the previously justified narrow control point, not a new objective idea
  - keep the Phase 2 one-env trusted contract fixed
  - temporarily force both Phase 2 and current Phase 3 contract builds through the native RWKV core path
  - check whether bypassing the batch=1/no-grad/eval official eval shortcut closes:
    - first-step hidden drift
    - rollout/value/advantage drift
    - main-loss gradient drift

- Method:
  - read-only probe-level monkeypatch only
  - no production training semantics changed
  - no actor-only path
  - no aux-flow path
  - no pooled weighting
  - both sides still use the Phase 2 shared actor-critic backbone contract:
    - `n_envs = 1`
    - `n_steps = 256`
    - `batch_size = 256`
    - `n_epochs = 1`
    - `actor_baseline_mode = learned`
    - `actor_gae_space = normalized`
    - `actor_objective_mode = tokenwise`
    - `reset_env_state_at_sep = true`
    - `separate_value_backbone = false`
    - `strict_fixed_env_mode = true`
    - deterministic actor sampling and deterministic batch plan enabled

- Locked results:
  - runtime was normal:
    - `elapsed_wall_time_sec = 23.7160`
    - this is below the Phase 2 standard-test runtime scale of about `103s`
  - setup stayed exact:
    - `signature_diff = {}`
    - `policy_param_max_abs_diff = 0.0`
  - the first call on both sides would have used official eval without the probe override:
    - `phase2_would_use_official = true`
    - `phase3_would_use_official = true`
  - the forced-native path was actually used for the rollout calls:
    - `phase2_native_calls = 256`
    - `phase3_native_calls = 256`
    - `phase2_official_eval_state_seen_count = 0`
    - `phase3_official_eval_state_seen_count = 0`
  - previous default drift reference:
    - `default_first_hidden_max_abs_diff = 0.015625`
    - `default_rollout_max_abs_diff = 0.0019381940`
    - `default_old_values_max_abs_diff = 0.0019039959`
    - `default_returns_max_abs_diff = 0.0002077818`
    - `default_advantages_max_abs_diff = 0.0019381940`
    - `default_grad_max_abs_delta = 5.4240227e-06`
    - `default_grad_l2_delta_norm = 6.3271909e-05`
  - forced native closes all compared layers exactly:
    - `native_first_hidden_max_abs_diff = 0.0`
    - `native_rollout_max_abs_diff = 0.0`
    - `native_grad_max_abs_delta = 0.0`
    - `grad_l2_delta_norm = 0.0`
    - all rollout tensor diffs are `0.0`:
      - actions
      - old_values
      - returns
      - advantages
      - old_log_prob
      - objective_masks
      - episode_starts
      - observations
    - all scalar loss/stat diffs are `0.0`

- Interpretation:
  - this is the strongest current root-cause result in the Phase 2-to-Phase 3 baseline alignment line
  - the remaining one-env milestone drift is caused by the batch=1/no-grad/eval official eval shortcut, not by:
    - Phase 3 profile drift
    - environment reset/input transport
    - tokenization
    - cache construction
    - native RWKV core math
    - PPO objective or many-env weighting
  - the minimal control is therefore runtime-contract level:
    - guard or bypass the official eval shortcut under strict regression / trusted compare mode
  - this does not justify changing Phase 3 objective semantics

- Verification:
  - `python -m py_compile reinforce-terminal-explore/ticl/analysis/phase3_phase2_official_eval_vs_native_contract_probe.py reinforce-terminal-explore/ticl/tests/test_phase3_phase2_official_eval_vs_native_contract_probe.py`
  - `conda run -n rlpfn python -m pytest reinforce-terminal-explore/ticl/tests/test_phase3_phase2_official_eval_vs_native_contract_probe.py -q`
  - result:
    - `2 passed`
  - final probe command:
    - `conda run -n rlpfn python reinforce-terminal-explore/ticl/analysis/phase3_phase2_official_eval_vs_native_contract_probe.py --output-json /home/chen/RLPFN/artifacts/phase3_phase2_official_eval_vs_native_contract_probe.json`

- Decision:
  - this narrow modification is validated as effective for the Phase 2 one-env contract mismatch
  - still do not start many-env A/B until the control is formalized as a guarded runtime option rather than an analysis-only monkeypatch
  - the official eval shortcut must not be allowed to silently affect trusted numeric/gradient compares

- Next step:
  - implement exactly one minimal production guard:
    - a strict-regression/native-rollout switch that bypasses the official eval shortcut only when explicitly enabled for trusted comparisons
  - then rerun the original Phase 2 milestone gradient compare without analysis monkeypatch
  - acceptance criterion:
    - `signature_diff = {}`
    - `policy_param_max_abs_diff = 0.0`
    - first-step / rollout / scalar / gradient diffs all remain `0.0`
  - only after that should Phase 3 many-env baseline vs narrow-adjustment A/B be trusted

2026-04-15 Formal strict-native rollout guard and milestone recheck

- New / changed code:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/models/rwkv7_pfn.py`
    - added default-off `force_native_eval_forward_step`
    - `RWKV7Core.forward_step()` now bypasses the batch=1/no-grad/eval official eval shortcut only when this flag is true
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/sb3_recurrent_ppo.py`
    - added `build_recurrent_ppo(..., strict_native_rollout=False)`
    - production default remains false
    - when true, the policy actor core and optional separate value core are both switched to native forward-step mode
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_phase2_milestone_contract_gradient_compare.py`
    - trusted milestone compare now defaults to `strict_native_rollout=True`
    - the switch is included in both Phase 2 and Phase 3 signatures
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_phase3_phase2_milestone_contract_gradient_compare.py`
    - added a signature guard for `strict_native_rollout`

- Updated canonical artifact:
  - `/home/chen/RLPFN/artifacts/phase3_phase2_milestone_contract_gradient_compare.json`

- Purpose:
  - convert the successful analysis-only monkeypatch into the smallest formal runtime guard
  - keep normal training default behavior unchanged
  - make trusted numeric compares explicit about bypassing the official eval shortcut
  - rerun the original milestone gradient compare without analysis monkeypatch

- Locked results:
  - observed command wall time:
    - about `28.6s`
    - this is still below the Phase 2 standard-test runtime scale of about `103s`
  - trusted contract:
    - `strict_native_rollout = true`
    - `signature_diff = {}`
    - `policy_param_max_abs_diff = 0.0`
  - rollout tensor diffs are all exactly zero:
    - `actions = 0.0`
    - `advantages = 0.0`
    - `episode_starts = 0.0`
    - `objective_masks = 0.0`
    - `observations = 0.0`
    - `old_log_prob = 0.0`
    - `old_values = 0.0`
    - `returns = 0.0`
  - scalar loss/stat diffs are all exactly zero:
    - `policy_loss = 0.0`
    - `value_loss = 0.0`
    - `total_loss = 0.0`
    - `grad_norm = 0.0`
    - `approx_kl = 0.0`
    - `clip_fraction = 0.0`
    - `entropy_loss = 0.0`
    - `objective_total = 0.0`
    - `aux_q_weight = 0.0`
    - `aux_flow_weight = 0.0`
  - gradient compare closes exactly:
    - `max_abs_grad_delta = 0.0`
    - `l2_delta_norm = 0.0`
    - `phase2_grad_norm = 5.755584239959717`
    - `phase3_grad_norm = 5.755584239959717`
  - note:
    - reported cosine is slightly above `1.0` due to floating-point reduction display
    - exactness is determined by `max_abs_grad_delta = 0.0` and `l2_delta_norm = 0.0`

- Interpretation:
  - the Phase 2 milestone vs current single-env trusted contract is now numerically closed
  - the official eval shortcut issue is controlled by an explicit strict-regression switch
  - this is a small runtime-contract guard, not an objective change
  - normal production default still uses the official eval shortcut unless the caller opts into `strict_native_rollout=True`

- Verification:
  - `python -m py_compile reinforce-terminal-explore/ticl/models/rwkv7_pfn.py reinforce-terminal-explore/ticl/sb3_recurrent_ppo.py reinforce-terminal-explore/ticl/analysis/phase3_phase2_milestone_contract_gradient_compare.py`
  - `conda run -n rlpfn python -m pytest reinforce-terminal-explore/ticl/tests/test_phase3_phase2_milestone_contract_gradient_compare.py reinforce-terminal-explore/ticl/tests/test_phase3_phase2_official_eval_vs_native_contract_probe.py -q`
  - result:
    - `5 passed`
  - default fastpath preservation check:
    - `conda run -n rlpfn python -m pytest reinforce-terminal-explore/ticl/tests/test_rwkv7_rlpfn.py::test_rwkv7_core_cuda_batch1_official_eval_fastpath_matches_raw_step -q`
    - result: `1 passed`
  - milestone compare command:
    - `conda run -n rlpfn python reinforce-terminal-explore/ticl/analysis/phase3_phase2_milestone_contract_gradient_compare.py --output-json /home/chen/RLPFN/artifacts/phase3_phase2_milestone_contract_gradient_compare.json`
  - JSON acceptance guard:
    - all conclusion booleans true
    - max rollout diff `0.0`
    - max scalar diff `0.0`
    - max grad diff `0.0`

- Decision:
  - Phase 2 regression baseline is green again under the formal strict-native rollout guard
  - Phase 3 may now proceed only from this guarded Phase 2 shared-backbone contract
  - allowed Phase 3 differences remain limited to:
    - many-env rollout/training
    - the single narrow adjustment being tested
  - still forbid:
    - actor-only substitutions
    - aux-flow drift
    - pooled weighting repairs
    - unguarded official eval shortcut in trusted numeric compares

- Next step:
  - do one Phase 3 preflight contract check before target-side A/B:
    - build the many-env baseline with Phase 2 shared-backbone settings plus `strict_native_rollout=True`
    - verify config/signature contains no extra drift
    - verify single-env collapsed path remains exactly green against the updated milestone artifact
  - after that, run the many-env baseline vs the previously justified narrow adjustment as the only A/B difference

2026-04-15 Phase 3 pair2 preflight contract check

- New artifact:
  - `/home/chen/RLPFN/artifacts/phase3_preflight_contract_check_pair2.json`
- New code:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_preflight_contract_check.py`
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_phase3_preflight_contract_check.py`
- Updated code:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_multi_env_optimization_audit.py`
    - now accepts and reports `strict_native_rollout`
    - post-policy bundle contract now includes `strict_native_rollout`
    - old bundles without this field are not treated as current trusted-contract bundles

- Purpose:
  - build the Phase 3 many-env baseline contract without training or evaluation
  - verify that the only baseline differences from the guarded Phase 2 milestone are the many-env identity/seed fields
  - require the guarded runtime control:
    - `strict_native_rollout = true`
  - block many-env target-side A/B if any non-target config drift appears

- Pair2 preflight inputs:
  - train suite:
    - `/home/chen/RLPFN/artifacts/phase3_cross_env_baseline_pair2/suites/train_suite.pt`
  - heldout suite:
    - `/home/chen/RLPFN/artifacts/phase3_cross_env_baseline_pair2/suites/heldout_suite.pt`
  - guarded Phase 2 milestone:
    - `/home/chen/RLPFN/artifacts/phase3_phase2_milestone_contract_gradient_compare.json`

- Locked results:
  - runtime was normal:
    - `elapsed_wall_time_sec = 1.6969`
    - this is far below the Phase 2 standard-test runtime scale of about `103s`
  - `preflight_passed = true`
  - Phase 3 baseline contract:
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
    - `runtime_normalized_q_value_weight_override = null`
    - `runtime_next_state_flow_matching_weight_override = null`
    - `strict_fixed_env_mode = true`
    - `deterministic_actor_sampling = true`
    - `deterministic_batch_plan = true`
    - `strict_native_rollout = true`
  - built runtime confirms the guard is active:
    - `algo_strict_native_rollout = true`
    - `actor_core_force_native_eval_forward_step = true`
    - `rollout_buffer_actor_objective_mode = tokenwise`
    - `vec_env_num_envs = 16`
  - contract diff from Phase 2 milestone contains only allowed many-env differences:
    - `n_envs`
    - `env_rng_seeds`
    - `rollout_rng_seeds`
  - unexpected drift:
    - `unexpected_diff = {}`

- Verification:
  - `python -m py_compile reinforce-terminal-explore/ticl/analysis/phase3_multi_env_optimization_audit.py reinforce-terminal-explore/ticl/analysis/phase3_preflight_contract_check.py reinforce-terminal-explore/ticl/tests/test_phase3_preflight_contract_check.py reinforce-terminal-explore/ticl/tests/test_phase3_multi_env_optimization_audit.py`
  - `conda run -n rlpfn python -m pytest reinforce-terminal-explore/ticl/tests/test_phase3_preflight_contract_check.py reinforce-terminal-explore/ticl/tests/test_phase3_multi_env_optimization_audit.py::test_post_policy_bundle_roundtrip reinforce-terminal-explore/ticl/tests/test_phase3_multi_env_optimization_audit.py::test_run_phase3_audit_resumes_post_policy_bundle_without_train_loop reinforce-terminal-explore/ticl/tests/test_phase3_multi_env_optimization_audit.py::test_run_phase3_audit_forwards_pair2_runtime_scope_and_strict_contract -q`
  - result:
    - `5 passed`
  - preflight command:
    - `conda run -n rlpfn python reinforce-terminal-explore/ticl/analysis/phase3_preflight_contract_check.py --output-json /home/chen/RLPFN/artifacts/phase3_preflight_contract_check_pair2.json`
  - JSON acceptance guard:
    - `preflight_passed = true`
    - `unexpected_contract_diff_free = true`
    - only diff keys are `env_rng_seeds`, `n_envs`, `rollout_rng_seeds`

- Decision:
  - pair2 Phase 3 many-env baseline contract is green
  - the next trusted experiment may be a target-side A/B only if both arms share this exact preflight contract
  - the only allowed A/B difference is the previously justified narrow actor-objective adjustment

- Next step:
  - update the pair2 target-side A/B driver so both baseline and override pass:
    - `train_profile = phase2_shared_backbone_contract`
    - `n_samples = 256`
    - `single_eval_pos = 64`
    - `strict_fixed_env_mode = true`
    - `deterministic_actor_sampling = true`
    - `deterministic_batch_plan = true`
    - `strict_native_rollout = true`
  - then run one minimal target-side A/B:
    - baseline: `actor_objective_mode = tokenwise`
    - override: previously justified narrow mode only
  - do not add any other profile or objective changes

2026-04-15 Phase 3 pair2 target-side A/B under guarded Phase 2 contract

Update:
- The initial numeric result from this section was superseded by the deterministic-eval rerun below.
- Root cause:
  - `deterministic_actor_sampling = true` was part of the trusted contract, but policy eval did not forward it into `_build_audit_ppo_policy_step_fn`.
  - That left pre/post policy evaluation stochastic even though collect/train was strict.
  - The large earlier target-side delta must not be used as a stable result.
- Code fix:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_multi_env_optimization_audit.py`
    - `_run_policy_eval()` now accepts `deterministic_actor_sampling`
    - all pre/post policy eval calls forward the flag
  - post-policy bundle contracts now include `actor_objective_mode`
    - this prevents baseline and branch-specific override bundles from being accidentally interchanged

- New artifact:
  - `/home/chen/RLPFN/artifacts/phase3_pair2_train_update_ab_compare_pack.json`
- Updated code:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_pair2_train_update_ab_compare_pack.py`
    - now loads and enforces `/home/chen/RLPFN/artifacts/phase3_preflight_contract_check_pair2.json`
    - forces Phase-2-scale settings:
      - `n_samples = 256`
      - `single_eval_pos = 64`
      - `batch_size = 256`
      - `n_epochs = 1`
      - `learning_rate = 0.0002`
      - `target_kl = 0.03`
      - `strict_native_rollout = true`
    - rejects baseline runtime objective scope
    - allows override to differ only by:
      - `actor_objective_mode_override = tokenwise_scale_env12_mid_episode_extension_block`
      - `actor_objective_runtime_current_suite_name = pair2`
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_phase3_pair2_train_update_ab_compare_pack.py`
    - covers the strict shared-contract and runtime-scope checks

- Runtime profile:
  - complete command wall time:
    - `real = 449.56s`
  - this is much slower than the one-env Phase 2 milestone because the trusted Phase 3 A/B performs serial policy evaluation over 16 train environments under `strict_native_rollout`
  - train/update itself is not the slow phase:
    - baseline `collect_rollouts = 15.070s`
    - baseline `algo.train = 3.819s`
    - baseline `audit_train_loop = 19.064s`
    - override `collect_rollouts = 13.889s`
    - override `algo.train = 4.491s`
    - override `audit_train_loop = 18.480s`
  - eval dominates:
    - baseline `zero_train_eval = 18.203s`
    - baseline `zero_heldout_eval = 16.969s`
    - baseline `pre_train_eval = 121.400s`
    - baseline `post_train_eval = 122.456s`
    - override `post_train_eval = 122.029s`
  - the 120s faulthandler stacks were in serial policy eval:
    - `evaluate_prior_suite -> _collect_suite_rewards_serial -> _rollout_single -> policy_step_fn -> RWKV forward_step`
  - no evidence of collect/train deadlock or runaway PPO update was found

- Contract checks:
  - preflight reused:
    - `preflight_passed = true`
    - `unexpected_diff = {}`
  - shared baseline/override context:
    - `train_profile = phase2_shared_backbone_contract`
    - `n_samples = 256`
    - `eval_n_samples = 256`
    - `single_eval_pos = 64`
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
  - only override target fields differ:
    - `actor_objective_mode_override = tokenwise_scale_env12_mid_episode_extension_block`
    - `actor_objective_runtime_current_suite_name = pair2`
  - heldout eval was intentionally skipped in this target-side A/B:
    - this artifact does not establish heldout generalization

- Superseded initial target-side trace:
  - keep this only as evidence for the eval-randomness bug above
  - do not use these numbers as the current trusted target-side result
  - pre metrics are unchanged:
    - `pre_train_vs_zero_gap_delta.full_return_gap = 0.0`
    - `pre_train_vs_zero_gap_delta.suffix_return_gap = 0.0`
  - baseline post-train vs zero:
    - `full_return_gap = -0.7805871964`
    - `suffix_return_gap = -0.6102252007`
  - override post-train vs zero:
    - `full_return_gap = -0.1433005333`
    - `suffix_return_gap = -0.1367490292`
  - override improvement over baseline:
    - `post_train_vs_zero_gap_delta.full_return_gap = +0.6372866631`
    - `post_train_vs_zero_gap_delta.suffix_return_gap = +0.4734761715`
  - train delta also improves:
    - baseline `train_full_return_delta = -0.7792921066`
    - override `train_full_return_delta = -0.1420054436`
    - baseline `train_suffix_return_delta = -0.6040790081`
    - override `train_suffix_return_delta = -0.1306028366`
  - critic quality remains high in both arms:
    - baseline `critic_raw_corr = 0.9927887321`
    - override `critic_raw_corr = 0.9915908575`
    - baseline `critic_explained_variance_raw = 0.9725700617`
    - override `critic_explained_variance_raw = 0.9603631496`
  - snapshot selector did not hit in either arm:
    - `baseline.train_outer_batch_snapshot.written = false`
    - `override.train_outer_batch_snapshot.written = false`
    - this run should not be interpreted as an exact env12 block aggregation audit

- Decision:
  - the narrow pair2 branch-specific objective adjustment has a real target-side effect under the guarded Phase 2 shared-backbone contract
  - it substantially reduces the negative post-train gap, but it does not make pair2 train reward positive in this one-epoch trusted A/B
  - the result supports keeping this as a runtime-feasible small-control candidate, not promoting it to a default Phase 3 fix
  - no actor-only, aux-flow, pooled weighting, normalized-advantage, separate-value-backbone, or unguarded official-eval shortcut was introduced

- Next step:
  - do not add a larger repair
  - if continuing this branch, the next single useful check is an eval-only heldout/non-target side-effect pass using saved post policies or an equivalent no-retrain bundle path
  - if saved post-policy bundles are not available, add bundle saving first; otherwise repeating this A/B just to evaluate heldout would waste about seven minutes of strict serial eval/runtime

2026-04-15 Phase 3 pair2 deterministic-eval A/B with post-policy bundle reuse

- Updated artifacts:
  - `/home/chen/RLPFN/artifacts/phase3_pair2_train_update_ab_compare_pack.json`
  - `/home/chen/RLPFN/artifacts/phase3_pair2_post_policy_side_effect_eval.json`
- New post-policy bundles:
  - `/home/chen/RLPFN/artifacts/phase3_pair2_train_update_ab_baseline_post_policy_bundle.pt`
  - `/home/chen/RLPFN/artifacts/phase3_pair2_train_update_ab_override_post_policy_bundle.pt`
- New code:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_pair2_post_policy_side_effect_eval.py`
    - loads saved post-policy bundles
    - runs eval-only heldout/non-target side-effect comparison
    - does not run PPO collect or train

- Deterministic eval contract fix:
  - `deterministic_actor_sampling = true` now applies to policy eval, not only PPO collect/train
  - bundle contract now records:
    - baseline `actor_objective_mode = tokenwise`
    - override `actor_objective_mode = tokenwise_scale_env12_mid_episode_extension_block`
  - this is a guardrail/instrumentation fix, not an objective repair

- Verification:
  - `python -m py_compile reinforce-terminal-explore/ticl/analysis/phase3_multi_env_optimization_audit.py reinforce-terminal-explore/ticl/analysis/phase3_pair2_train_update_ab_compare_pack.py reinforce-terminal-explore/ticl/analysis/phase3_pair2_post_policy_side_effect_eval.py`
  - `conda run -n rlpfn python -m pytest reinforce-terminal-explore/ticl/tests/test_phase3_multi_env_optimization_audit.py::test_post_policy_bundle_roundtrip reinforce-terminal-explore/ticl/tests/test_phase3_multi_env_optimization_audit.py::test_run_phase3_audit_resumes_post_policy_bundle_without_train_loop reinforce-terminal-explore/ticl/tests/test_phase3_pair2_train_update_ab_compare_pack.py -q`
  - result:
    - `4 passed`
  - JSON validation:
    - both updated artifacts pass `python -m json.tool`

- A/B runtime:
  - complete deterministic-eval A/B command:
    - `real = 456.12s`
  - train/update remains normal:
    - baseline `collect_rollouts = 15.168s`
    - baseline `algo.train = 4.643s`
    - baseline `audit_train_loop = 19.981s`
    - override `collect_rollouts = 13.993s`
    - override `algo.train = 4.354s`
    - override `audit_train_loop = 18.448s`
  - strict serial eval still dominates runtime:
    - `zero_train_eval = 18.169s`
    - `zero_heldout_eval = 17.257s`
    - baseline `pre_train_eval = 122.541s`
    - baseline `post_train_eval = 123.717s`
    - override `post_train_eval = 124.880s`

- Deterministic target-side result:
  - pre metrics unchanged:
    - `pre_train_vs_zero_gap_delta.full_return_gap = 0.0`
    - `pre_train_vs_zero_gap_delta.suffix_return_gap = 0.0`
  - baseline post-train vs zero:
    - `full_return_gap = -0.0539064407`
    - `suffix_return_gap = -0.0732917786`
  - override post-train vs zero:
    - `full_return_gap = -0.0136370659`
    - `suffix_return_gap = -0.0299243927`
  - override improvement over baseline:
    - `post_train_vs_zero_gap_delta.full_return_gap = +0.0402693748`
    - `post_train_vs_zero_gap_delta.suffix_return_gap = +0.0433673859`
  - interpretation:
    - the branch-specific control still improves target-side train reward under the trusted contract
    - the effect is modest and does not turn the train gap positive
    - it remains a candidate small control, not a default Phase 3 fix

- Eval-only heldout/non-target side-effect:
  - command runtime:
    - `real = 254.01s`
  - no train loop was run:
    - `eval_only = true`
    - `no_train_loop = true`
  - heldout mean side-effect:
    - `override_minus_baseline_full_return_mean = +0.0055379868`
    - `override_minus_baseline_suffix_return_mean = +0.0107862949`
  - per-env heldout side-effect is mixed:
    - full-return delta:
      - `positive_count = 9`
      - `negative_count = 7`
      - `min = -0.9088620096`
      - `max = +0.7089591026`
    - suffix-return delta:
      - `positive_count = 9`
      - `negative_count = 7`
      - `min = -0.6861140132`
      - `max = +0.5092029572`

- Decision:
  - the post-policy bundle save/reuse mechanism is now usable under the guarded Phase 2 contract
  - heldout/non-target average side-effect is small and slightly positive, but per-env effects are mixed
  - this does not justify promoting the branch-specific adjustment to default
  - the next step should not be a larger repair

- Next step:
  - if continuing this line, do one repeat-only determinism check using the saved bundles:
    - rerun `/home/chen/RLPFN/artifacts/phase3_pair2_post_policy_side_effect_eval.json` without retraining
    - require identical heldout metrics under deterministic eval
  - only after that should we decide whether this branch-specific control deserves a second suite target-side A/B
