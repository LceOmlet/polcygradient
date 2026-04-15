# Phase 3 Cross-Suite Probe Entrypoints

This file lists the fixed-suite configs and commands to run the cross-suite probes.

## Critical Rule

Do not reuse pair1 heldout-delta artifacts on other suites.

In particular, do not pass:

- `/home/chen/RLPFN/artifacts/phase3_env_delta_concentration_probe_max4.json`

for any suite other than the canonical pair1 contract.

Cross-suite probes must either:

- use a suite-matched `reuse_heldout_subset_delta_json`, or
- omit that flag entirely and let the probe derive heldout per-env `pre_vs_zero_suffix_gap`
  from the suite-matched zero/PPO baselines

The current official path is the second one.

Runtime note:

- baseline commands below intentionally include `--no-action-reachability-probe`
- this is required for runtime sanity on new suites
- reason:
  - action-reachability is not consumed by the cross-suite baseline reuse contract
  - leaving it enabled adds extra serial rollout work and obscures whether slowdowns are real PPO-eval cost or just auxiliary probe overhead

## Verified Reference Suites

### Pair 1 Canonical

- Train suite: `/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/train_suite.pt`
- Heldout suite: `/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/heldout_suite.pt`
- Zero baseline: `/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/zero_cross_env_control.json`
- PPO baseline: `/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/ppo_cross_env_baseline.json`

### Pair 2 Verified Cross-Suite Reproduction

- Train suite: `/home/chen/RLPFN/artifacts/phase3_cross_env_baseline_pair2/suites/train_suite.pt`
- Heldout suite: `/home/chen/RLPFN/artifacts/phase3_cross_env_baseline_pair2/suites/heldout_suite.pt`
- Zero baseline: `/home/chen/RLPFN/artifacts/phase3_regression_pack/phase3_cross_env_baseline_pair2_zero.json`
- PPO baseline: `/home/chen/RLPFN/artifacts/phase3_regression_pack/phase3_cross_env_baseline_pair2_ppo.json`

Verified commands:

```bash
conda run -n rlpfn python reinforce-terminal-explore/ticl/analysis/phase3_post_reset_tail_distance_probe.py \
  /home/chen/RLPFN/rwkv/models_diff/rlpfn_03_30_2026_22_37_53_epoch_13.cpkt \
  --train-suite-path /home/chen/RLPFN/artifacts/phase3_cross_env_baseline_pair2/suites/train_suite.pt \
  --heldout-suite-path /home/chen/RLPFN/artifacts/phase3_cross_env_baseline_pair2/suites/heldout_suite.pt \
  --reuse-zero-control-json /home/chen/RLPFN/artifacts/phase3_regression_pack/phase3_cross_env_baseline_pair2_zero.json \
  --reuse-pre-policy-json /home/chen/RLPFN/artifacts/phase3_regression_pack/phase3_cross_env_baseline_pair2_ppo.json \
  --n-samples 2048 --single-eval-pos 1946 \
  --output-json /home/chen/RLPFN/artifacts/phase3_post_reset_tail_distance_probe_pair2.json
```

```bash
conda run -n rlpfn python reinforce-terminal-explore/ticl/analysis/phase3_post_reset_tail_dist4_late_ablation_verify.py \
  /home/chen/RLPFN/rwkv/models_diff/rlpfn_03_30_2026_22_37_53_epoch_13.cpkt \
  --train-suite-path /home/chen/RLPFN/artifacts/phase3_cross_env_baseline_pair2/suites/train_suite.pt \
  --heldout-suite-path /home/chen/RLPFN/artifacts/phase3_cross_env_baseline_pair2/suites/heldout_suite.pt \
  --reuse-zero-control-json /home/chen/RLPFN/artifacts/phase3_regression_pack/phase3_cross_env_baseline_pair2_zero.json \
  --reuse-pre-policy-json /home/chen/RLPFN/artifacts/phase3_regression_pack/phase3_cross_env_baseline_pair2_ppo.json \
  --n-samples 2048 --single-eval-pos 1946 \
  --output-json /home/chen/RLPFN/artifacts/phase3_post_reset_tail_dist4_late_ablation_pair2.json
```

### Suite `seed_24680` Completed

- Zero baseline:
  - `/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_24680/zero_cross_env_control.json`
- PPO baseline:
  - `/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_24680/ppo_cross_env_baseline.json`
- Distance probe:
  - `/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_24680/phase3_post_reset_tail_distance_probe.json`
- Terminal-tail semantic contrast:
  - `/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_24680/phase3_terminal_tail_mask_probe.json`
  - `/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_24680/phase3_post_reset_tail_ablation_verify.json`
  - `/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_24680/phase3_post_reset_ablation_verify.json`

Locked reading:

- suite-level heldout `PPO-vs-zero` flips negative on this suite
- near-terminal `post_reset_tail_dist_1..4` still track reset density
- but their `corr_vs_pre_suffix_gap` no longer matches pair1/pair2
- removing post-reset mass lowers `corr_reset` while making `corr_pre` more negative
- therefore this suite should be treated as evidence that:
  - reset-bias and pre-gap alignment have decoupled
  - pair1/pair2 token-bias interpretation is not yet suite-invariant

### Suite `seed_13579` Completed

- Zero baseline:
  - `/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_13579/zero_cross_env_control.json`
- PPO baseline:
  - `/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_13579/ppo_cross_env_baseline.json`
- Distance probe:
  - `/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_13579/phase3_post_reset_tail_distance_probe.json`
- Terminal-tail semantic contrast:
  - `/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_13579/phase3_terminal_tail_mask_probe.json`
  - `/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_13579/phase3_post_reset_tail_ablation_verify.json`
  - `/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_13579/phase3_post_reset_ablation_verify.json`

Locked reading:

- suite-level heldout `PPO-vs-zero` is weakly positive on this suite:
  - `ppo_minus_zero_suffix = +0.0147`
- `dist1..4` remains reset-tracking and mostly negative on `pre-gap`
- but removing `post_reset_terminal_tail` has zero effect
- removing whole `post_reset_only` has only a tiny effect
- therefore this suite belongs to the positive regime by sign, but has much weaker semantic-ablation leverage than pair1

### Current Cross-Suite Root-Cause Summary

- canonical summary artifact:
  - `/home/chen/RLPFN/artifacts/phase3_cross_suite_regime_summary.json`
- canonical entrypoint:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_cross_suite_regime_summary.py`
- canonical mixed-update artifact:
  - `/home/chen/RLPFN/artifacts/phase3_regime_mixed_update_probe.json`
- canonical mixed-update entrypoint:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_regime_mixed_update_probe.py`
- canonical suite/env concentration artifact:
  - `/home/chen/RLPFN/artifacts/phase3_regime_env_concentration_probe.json`
- canonical suite/env concentration entrypoint:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_regime_env_concentration_probe.py`
- current locked root cause:
  - reset-heavy near-terminal mass is fitted consistently across suites
  - but the mapping from that mass to heldout gain is regime-dependent
  - future analyses must stratify by suite-level `ppo_minus_zero_suffix` sign before comparing token-quality correlations
  - pooled all-env token-quality correlations are not sufficient by themselves
  - current rerun on `2026-04-12` preserved the same regime split and concentration profile
  - the train-side contrast is now also labeled and verified as the pair1-positive regime
  - expected artifact booleans:
    - `regime_split_visible_in_raw_correlation = true`
    - `pooled_sign_hides_regime_split = true`
    - `positive_regime_dominates_pooled_update = true`
  - future probes must also report:
    - per-regime mass share
    - per-regime covariance contribution to the pooled update signal
    - top envs by update-mass proxy
    - top envs by pooled covariance contribution

## Suite A (heldout_seed=24680)

- Config: `/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_24680/suite_config.json`
- Train suite: `/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_24680/train_suite.pt`
- Heldout suite: `/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_24680/heldout_suite.pt`
- Zero baseline target: `/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_24680/zero_cross_env_control.json`
- PPO baseline target: `/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_24680/ppo_cross_env_baseline.json`

### Baselines

```bash
conda run -n rlpfn python reinforce-terminal-explore/ticl/analysis/prior_generalization_audit.py \
  --checkpoint /home/chen/RLPFN/rwkv/models_diff/rlpfn_03_30_2026_22_37_53_epoch_13.cpkt \
  --policy-mode zero \
  --train-suite-path /home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_24680/train_suite.pt \
  --heldout-suite-path /home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_24680/heldout_suite.pt \
  --rollout-backend serial \
  --no-action-reachability-probe \
  --n-samples 2048 --single-eval-pos 1946 \
  --output /home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_24680/zero_cross_env_control.json
```

```bash
conda run -n rlpfn python reinforce-terminal-explore/ticl/analysis/prior_generalization_audit.py \
  --checkpoint /home/chen/RLPFN/rwkv/models_diff/rlpfn_03_30_2026_22_37_53_epoch_13.cpkt \
  --policy-mode ppo \
  --train-suite-path /home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_24680/train_suite.pt \
  --heldout-suite-path /home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_24680/heldout_suite.pt \
  --rollout-backend serial \
  --no-action-reachability-probe \
  --n-samples 2048 --single-eval-pos 1946 \
  --output /home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_24680/ppo_cross_env_baseline.json
```

### Probes

```bash
conda run -n rlpfn python reinforce-terminal-explore/ticl/analysis/phase3_post_reset_tail_episode_probe.py \
  /home/chen/RLPFN/rwkv/models_diff/rlpfn_03_30_2026_22_37_53_epoch_13.cpkt \
  --train-suite-path /home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_24680/train_suite.pt \
  --heldout-suite-path /home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_24680/heldout_suite.pt \
  --reuse-zero-control-json /home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_24680/zero_cross_env_control.json \
  --reuse-pre-policy-json /home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_24680/ppo_cross_env_baseline.json \
  --n-samples 2048 --single-eval-pos 1946 \
  --output-json /home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_24680/phase3_post_reset_tail_episode_probe.json
```

```bash
conda run -n rlpfn python reinforce-terminal-explore/ticl/analysis/phase3_post_reset_tail_distance_probe.py \
  /home/chen/RLPFN/rwkv/models_diff/rlpfn_03_30_2026_22_37_53_epoch_13.cpkt \
  --train-suite-path /home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_24680/train_suite.pt \
  --heldout-suite-path /home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_24680/heldout_suite.pt \
  --reuse-zero-control-json /home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_24680/zero_cross_env_control.json \
  --reuse-pre-policy-json /home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_24680/ppo_cross_env_baseline.json \
  --n-samples 2048 --single-eval-pos 1946 \
  --output-json /home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_24680/phase3_post_reset_tail_distance_probe.json
```

```bash
conda run -n rlpfn python reinforce-terminal-explore/ticl/analysis/phase3_post_reset_tail_dist4_late_ablation_verify.py \
  /home/chen/RLPFN/rwkv/models_diff/rlpfn_03_30_2026_22_37_53_epoch_13.cpkt \
  --train-suite-path /home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_24680/train_suite.pt \
  --heldout-suite-path /home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_24680/heldout_suite.pt \
  --reuse-zero-control-json /home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_24680/zero_cross_env_control.json \
  --reuse-pre-policy-json /home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_24680/ppo_cross_env_baseline.json \
  --n-samples 2048 --single-eval-pos 1946 \
  --output-json /home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_24680/phase3_post_reset_tail_dist4_late_ablation.json
```

## Suite B (heldout_seed=13579)

- Config: `/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_13579/suite_config.json`
- Train suite: `/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_13579/train_suite.pt`
- Heldout suite: `/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_13579/heldout_suite.pt`
- Zero baseline target: `/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_13579/zero_cross_env_control.json`
- PPO baseline target: `/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_13579/ppo_cross_env_baseline.json`

### Baselines

```bash
conda run -n rlpfn python reinforce-terminal-explore/ticl/analysis/prior_generalization_audit.py \
  --checkpoint /home/chen/RLPFN/rwkv/models_diff/rlpfn_03_30_2026_22_37_53_epoch_13.cpkt \
  --policy-mode zero \
  --train-suite-path /home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_13579/train_suite.pt \
  --heldout-suite-path /home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_13579/heldout_suite.pt \
  --rollout-backend serial \
  --no-action-reachability-probe \
  --n-samples 2048 --single-eval-pos 1946 \
  --output /home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_13579/zero_cross_env_control.json
```

```bash
conda run -n rlpfn python reinforce-terminal-explore/ticl/analysis/prior_generalization_audit.py \
  --checkpoint /home/chen/RLPFN/rwkv/models_diff/rlpfn_03_30_2026_22_37_53_epoch_13.cpkt \
  --policy-mode ppo \
  --train-suite-path /home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_13579/train_suite.pt \
  --heldout-suite-path /home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_13579/heldout_suite.pt \
  --rollout-backend serial \
  --no-action-reachability-probe \
  --n-samples 2048 --single-eval-pos 1946 \
  --output /home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_13579/ppo_cross_env_baseline.json
```

### Probes

```bash
conda run -n rlpfn python reinforce-terminal-explore/ticl/analysis/phase3_post_reset_tail_episode_probe.py \
  /home/chen/RLPFN/rwkv/models_diff/rlpfn_03_30_2026_22_37_53_epoch_13.cpkt \
  --train-suite-path /home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_13579/train_suite.pt \
  --heldout-suite-path /home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_13579/heldout_suite.pt \
  --reuse-zero-control-json /home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_13579/zero_cross_env_control.json \
  --reuse-pre-policy-json /home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_13579/ppo_cross_env_baseline.json \
  --n-samples 2048 --single-eval-pos 1946 \
  --output-json /home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_13579/phase3_post_reset_tail_episode_probe.json
```

```bash
conda run -n rlpfn python reinforce-terminal-explore/ticl/analysis/phase3_post_reset_tail_distance_probe.py \
  /home/chen/RLPFN/rwkv/models_diff/rlpfn_03_30_2026_22_37_53_epoch_13.cpkt \
  --train-suite-path /home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_13579/train_suite.pt \
  --heldout-suite-path /home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_13579/heldout_suite.pt \
  --reuse-zero-control-json /home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_13579/zero_cross_env_control.json \
  --reuse-pre-policy-json /home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_13579/ppo_cross_env_baseline.json \
  --n-samples 2048 --single-eval-pos 1946 \
  --output-json /home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_13579/phase3_post_reset_tail_distance_probe.json
```

```bash
conda run -n rlpfn python reinforce-terminal-explore/ticl/analysis/phase3_post_reset_tail_dist4_late_ablation_verify.py \
  /home/chen/RLPFN/rwkv/models_diff/rlpfn_03_30_2026_22_37_53_epoch_13.cpkt \
  --train-suite-path /home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_13579/train_suite.pt \
  --heldout-suite-path /home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_13579/heldout_suite.pt \
  --reuse-zero-control-json /home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_13579/zero_cross_env_control.json \
  --reuse-pre-policy-json /home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_13579/ppo_cross_env_baseline.json \
  --n-samples 2048 --single-eval-pos 1946 \
  --output-json /home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_13579/phase3_post_reset_tail_dist4_late_ablation.json
```

## Pair1 suite-matched repair + anchor checks

Repair the historical pair1 artifacts before any new regime-level summary:

```bash
conda run -n rlpfn python reinforce-terminal-explore/ticl/analysis/phase3_suite_matched_gap_repair.py \
  --artifact-json /home/chen/RLPFN/artifacts/phase3_post_reset_tail_distance_probe.json \
  --zero-json /home/chen/RLPFN/artifacts/phase3_cross_env_baseline/zero_cross_env_control.json \
  --ppo-json /home/chen/RLPFN/artifacts/phase3_cross_env_baseline/ppo_cross_env_baseline.json \
  --output-json /home/chen/RLPFN/artifacts/phase3_post_reset_tail_distance_probe_pair1_suite_matched.json \
  --overwrite
```

```bash
conda run -n rlpfn python reinforce-terminal-explore/ticl/analysis/phase3_suite_matched_gap_repair.py \
  --artifact-json /home/chen/RLPFN/artifacts/phase3_terminal_tail_mask_probe.json \
  --zero-json /home/chen/RLPFN/artifacts/phase3_cross_env_baseline/zero_cross_env_control.json \
  --ppo-json /home/chen/RLPFN/artifacts/phase3_cross_env_baseline/ppo_cross_env_baseline.json \
  --output-json /home/chen/RLPFN/artifacts/phase3_terminal_tail_mask_probe_pair1_suite_matched.json \
  --overwrite
```

```bash
conda run -n rlpfn python reinforce-terminal-explore/ticl/analysis/phase3_post_reset_tail_ablation_verify.py \
  --tail-mask-probe-json /home/chen/RLPFN/artifacts/phase3_terminal_tail_mask_probe_pair1_suite_matched.json \
  --output-json /home/chen/RLPFN/artifacts/phase3_post_reset_tail_ablation_verify_pair1_suite_matched.json \
  --overwrite
```

```bash
conda run -n rlpfn python reinforce-terminal-explore/ticl/analysis/phase3_post_reset_ablation_verify.py \
  --tail-mask-probe-json /home/chen/RLPFN/artifacts/phase3_terminal_tail_mask_probe_pair1_suite_matched.json \
  --output-json /home/chen/RLPFN/artifacts/phase3_post_reset_ablation_verify_pair1_suite_matched.json \
  --overwrite
```

Then rebuild the regime summaries with the repaired pair1 default:

```bash
conda run -n rlpfn python reinforce-terminal-explore/ticl/analysis/phase3_cross_suite_regime_summary.py \
  --output-json /home/chen/RLPFN/artifacts/phase3_cross_suite_regime_summary_pair1_suite_matched.json \
  --overwrite
```

```bash
conda run -n rlpfn python reinforce-terminal-explore/ticl/analysis/phase3_regime_mixed_update_probe.py \
  --output-json /home/chen/RLPFN/artifacts/phase3_regime_mixed_update_probe_pair1_suite_matched.json \
  --overwrite
```

```bash
conda run -n rlpfn python reinforce-terminal-explore/ticl/analysis/phase3_regime_env_concentration_probe.py \
  --output-json /home/chen/RLPFN/artifacts/phase3_regime_env_concentration_probe_pair1_suite_matched.json \
  --overwrite
```

Anchor probe for pair1 heavy-mass envs and pair2 env12:

```bash
conda run -n rlpfn python reinforce-terminal-explore/ticl/analysis/phase3_regime_anchor_env_probe.py \
  --output-json /home/chen/RLPFN/artifacts/phase3_regime_anchor_env_probe.json \
  --overwrite
```

Canonical pair2 tail-mask probe:

```bash
conda run -n rlpfn python reinforce-terminal-explore/ticl/analysis/phase3_terminal_tail_mask_probe.py \
  /home/chen/RLPFN/rwkv/models_diff/rlpfn_03_30_2026_22_37_53_epoch_13.cpkt \
  --train-suite-path /home/chen/RLPFN/artifacts/phase3_cross_env_baseline_pair2/suites/train_suite.pt \
  --heldout-suite-path /home/chen/RLPFN/artifacts/phase3_cross_env_baseline_pair2/suites/heldout_suite.pt \
  --reuse-zero-control-json /home/chen/RLPFN/artifacts/phase3_cross_env_baseline_pair2/zero_cross_env_control.json \
  --reuse-pre-policy-json /home/chen/RLPFN/artifacts/phase3_cross_env_baseline_pair2/ppo_cross_env_baseline.json \
  --n-samples 2048 --single-eval-pos 1946 \
  --output-json /home/chen/RLPFN/artifacts/phase3_terminal_tail_mask_probe_pair2.json \
  --overwrite
```

Positive-regime captured-vs-missed anchor structure compare:

```bash
conda run -n rlpfn python reinforce-terminal-explore/ticl/analysis/phase3_positive_regime_anchor_structure_probe.py \
  --output-json /home/chen/RLPFN/artifacts/phase3_positive_regime_anchor_structure_probe.json \
  --overwrite
```

Dual-channel anchor coverage compare:

```bash
conda run -n rlpfn python reinforce-terminal-explore/ticl/analysis/phase3_dual_channel_anchor_coverage_probe.py \
  --output-json /home/chen/RLPFN/artifacts/phase3_dual_channel_anchor_coverage_probe.json \
  --overwrite
```

Residual missed-anchor sign-formation probe:

```bash
conda run -n rlpfn python reinforce-terminal-explore/ticl/analysis/phase3_residual_missed_anchor_sign_probe.py \
  /home/chen/RLPFN/rwkv/models_diff/rlpfn_03_30_2026_22_37_53_epoch_13.cpkt \
  --anchor-source dual_channel_residual_missed \
  --output-json /home/chen/RLPFN/artifacts/phase3_residual_missed_anchor_sign_probe.json \
  --overwrite
```

Residual-anchor GAE path probe:

```bash
conda run -n rlpfn python reinforce-terminal-explore/ticl/analysis/phase3_residual_anchor_gae_path_probe.py \
  /home/chen/RLPFN/rwkv/models_diff/rlpfn_03_30_2026_22_37_53_epoch_13.cpkt \
  --output-json /home/chen/RLPFN/artifacts/phase3_residual_anchor_gae_path_probe.json \
  --overwrite
```

Pair2 flip-mode split probe:

```bash
conda run -n rlpfn python reinforce-terminal-explore/ticl/analysis/phase3_pair2_flip_mode_split_probe.py \
  --output-json /home/chen/RLPFN/artifacts/phase3_pair2_flip_mode_split_probe.json \
  --overwrite
```

Residual flip-mode compare pack:

```bash
conda run -n rlpfn python reinforce-terminal-explore/ticl/analysis/phase3_residual_flip_mode_compare_pack.py \
  --output-json /home/chen/RLPFN/artifacts/phase3_residual_flip_mode_compare_pack.json \
  --overwrite
```

Future-carry source-chain compare pack:

```bash
conda run -n rlpfn python reinforce-terminal-explore/ticl/analysis/phase3_future_carry_source_chain_compare_pack.py \
  --output-json /home/chen/RLPFN/artifacts/phase3_future_carry_source_chain_compare_pack.json \
  --overwrite
```

Baseline-bootstrap semantic mismatch compare pack:

```bash
conda run -n rlpfn python reinforce-terminal-explore/ticl/analysis/phase3_baseline_bootstrap_semantic_mismatch_compare_pack.py \
  --output-json /home/chen/RLPFN/artifacts/phase3_baseline_bootstrap_semantic_mismatch_compare_pack.json \
  --overwrite
```

Baseline-bootstrap terminal-adjacent core compare pack:

```bash
conda run -n rlpfn python reinforce-terminal-explore/ticl/analysis/phase3_baseline_bootstrap_semantic_mismatch_compare_pack.py \
  --near-terminal-window 1 \
  --output-json /home/chen/RLPFN/artifacts/phase3_baseline_bootstrap_terminal_adjacent_core_compare_pack.json \
  --overwrite
```

Env12 mid-episode extension compare pack:

```bash
conda run -n rlpfn python reinforce-terminal-explore/ticl/analysis/phase3_env12_mid_episode_extension_compare_pack.py \
  --output-json /home/chen/RLPFN/artifacts/phase3_env12_mid_episode_extension_compare_pack.json \
  --overwrite
```

Env12 mid-episode extension counterfactual pack:

```bash
conda run -n rlpfn python reinforce-terminal-explore/ticl/analysis/phase3_env12_mid_episode_extension_counterfactual_pack.py \
  --output-json /home/chen/RLPFN/artifacts/phase3_env12_mid_episode_extension_counterfactual_pack.json \
  --overwrite
```

Env12 mid-episode extension runtime control probe:

```bash
conda run -n rlpfn python reinforce-terminal-explore/ticl/analysis/phase3_env12_mid_episode_extension_runtime_control_probe.py \
  --output-json /home/chen/RLPFN/artifacts/phase3_env12_mid_episode_extension_runtime_control_probe.json \
  --overwrite
```

Env12 mid-episode extension runtime gate feasibility probe:

```bash
conda run -n rlpfn python reinforce-terminal-explore/ticl/analysis/phase3_env12_mid_episode_extension_runtime_gate_feasibility_probe.py \
  --output-json /home/chen/RLPFN/artifacts/phase3_env12_mid_episode_extension_runtime_gate_feasibility_probe.json \
  --overwrite
```

Env12 mid-episode extension runtime mode regression:

```bash
conda run -n rlpfn python reinforce-terminal-explore/ticl/analysis/phase3_env12_mid_episode_extension_runtime_mode_regression.py \
  --output-json /home/chen/RLPFN/artifacts/phase3_env12_mid_episode_extension_runtime_mode_regression.json \
  --overwrite
```

Future-carry source-chain probe for pair2 env12:

```bash
conda run -n rlpfn python reinforce-terminal-explore/ticl/analysis/phase3_future_carry_source_chain_probe.py \
  --output-json /home/chen/RLPFN/artifacts/phase3_future_carry_source_chain_probe_pair2_env12.json \
  --overwrite
```

Env12 negative-delta probe:

```bash
conda run -n rlpfn python reinforce-terminal-explore/ticl/analysis/phase3_env12_negative_delta_probe.py \
  --output-json /home/chen/RLPFN/artifacts/phase3_env12_negative_delta_probe.json \
  --overwrite
```

Env12 boundary-stepdown root probe:

```bash
conda run -n rlpfn python reinforce-terminal-explore/ticl/analysis/phase3_env12_boundary_stepdown_root_probe.py \
  --output-json /home/chen/RLPFN/artifacts/phase3_env12_boundary_stepdown_root_probe.json \
  --overwrite
```

Env12 local56 boundary-contract probe:

```bash
conda run -n rlpfn python reinforce-terminal-explore/ticl/analysis/phase3_env12_local56_boundary_contract_probe.py \
  --output-json /home/chen/RLPFN/artifacts/phase3_env12_local56_boundary_contract_probe.json \
  --overwrite
```

Env12 value peak-drop contract probe:

```bash
conda run -n rlpfn python reinforce-terminal-explore/ticl/analysis/phase3_env12_value_peak_drop_contract_probe.py \
  --output-json /home/chen/RLPFN/artifacts/phase3_env12_value_peak_drop_contract_probe.json \
  --overwrite
```

Env12 local56 mean-source probe:

```bash
conda run -n rlpfn python reinforce-terminal-explore/ticl/analysis/phase3_env12_local56_mean_source_probe.py \
  --output-json /home/chen/RLPFN/artifacts/phase3_env12_local56_mean_source_probe.json \
  --overwrite
```

Env12 local56 boundary-scope counterfactual probe:

```bash
conda run -n rlpfn python reinforce-terminal-explore/ticl/analysis/phase3_env12_local56_boundary_scope_counterfactual_probe.py \
  --output-json /home/chen/RLPFN/artifacts/phase3_env12_local56_boundary_scope_counterfactual_probe.json \
  --overwrite
```

Env12 local56 boundary mean-std-interaction probe:

```bash
conda run -n rlpfn python reinforce-terminal-explore/ticl/analysis/phase3_env12_local56_boundary_mean_std_interaction_probe.py \
  --output-json /home/chen/RLPFN/artifacts/phase3_env12_local56_boundary_mean_std_interaction_probe.json \
  --overwrite
```

Env12 local56 scale-collapse probe:

```bash
conda run -n rlpfn python reinforce-terminal-explore/ticl/analysis/phase3_env12_local56_scale_collapse_probe.py \
  --output-json /home/chen/RLPFN/artifacts/phase3_env12_local56_scale_collapse_probe.json \
  --overwrite
```

Generic terminal-local scale-chain replication for pair1 env12:

```bash
conda run -n rlpfn python reinforce-terminal-explore/ticl/analysis/phase3_terminal_local_scale_chain_probe.py \
  /home/chen/RLPFN/rwkv/models_diff/rlpfn_03_30_2026_22_37_53_epoch_13.cpkt \
  --suite-name pair1 \
  --env-index 12 \
  --output-json /home/chen/RLPFN/artifacts/phase3_pair1_env12_terminal_scale_chain_probe.json \
  --overwrite
```

Notes:

- this command is now historical only
- current guardrail says do not treat pair1 env12 as an official replication point
- reason:
  - `/home/chen/RLPFN/artifacts/phase3_pair1_env12_objective_reset_contract_probe.json`
  - fresh official collect currently reports zero objective resets on this env

Pair1 env12 reset-contract replay-diff probe:

```bash
conda run -n rlpfn python reinforce-terminal-explore/ticl/analysis/phase3_reset_contract_replay_diff_probe.py \
  /home/chen/RLPFN/rwkv/models_diff/rlpfn_03_30_2026_22_37_53_epoch_13.cpkt \
  --suite-name pair1 \
  --env-index 12 \
  --output-json /home/chen/RLPFN/artifacts/phase3_pair1_env12_reset_contract_replay_diff_probe.json \
  --overwrite
```

Notes:

- this is now the narrow source-localization probe for the pair1 reset mismatch
- current locked read:
  - stored pair1 distance row belongs to the legacy unseeded collect contract
  - `legacy_distance_probe_runA` reproduces reset count `1`
  - `legacy_distance_probe_runB` drifts to reset count `0`
  - `official_strict` wires suite env/rollout seeds and yields reset count `0`
- therefore:
  - do not interpret the pair1 mismatch as a new official-path semantic regression
  - interpret it as contract drift between legacy distance artifacts and current strict fresh collect

Seed13579 env12 objective-reset contract probe:

```bash
conda run -n rlpfn python reinforce-terminal-explore/ticl/analysis/phase3_objective_reset_contract_probe.py \
  /home/chen/RLPFN/rwkv/models_diff/rlpfn_03_30_2026_22_37_53_epoch_13.cpkt \
  --suite-name seed13579 \
  --env-index 12 \
  --output-json /home/chen/RLPFN/artifacts/phase3_seed13579_env12_objective_reset_contract_probe.json \
  --overwrite
```

Notes:

- this probe is now the official post-fix reset-contract guardrail for `seed13579 env12`
- repeated rerun target:
  - `/home/chen/RLPFN/artifacts/phase3_seed13579_env12_objective_reset_contract_probe_repeatB.json`
- current locked read:
  - pre-fix helper was unstable
  - post-fix formal reruns agree on reset-contract fields
- the repaired terminal-local scale-chain rerun is now complete:
  - see the official fresh-collect replication entry immediately below

Seed13579 env12 terminal-local scale chain under repaired collect contract:

```bash
conda run -n rlpfn python reinforce-terminal-explore/ticl/analysis/phase3_terminal_local_scale_chain_probe.py \
  /home/chen/RLPFN/rwkv/models_diff/rlpfn_03_30_2026_22_37_53_epoch_13.cpkt \
  --suite-name seed13579 \
  --env-index 12 \
  --output-json /home/chen/RLPFN/artifacts/phase3_seed13579_env12_terminal_scale_chain_probe.json \
  --overwrite
```

Notes:

- this remains an official saved/replay-stable replication point for the high-level variance-collapse mechanism
- repeated rerun target:
  - `/home/chen/RLPFN/artifacts/phase3_seed13579_env12_terminal_scale_chain_probe_repeatB.json`
- current locked read:
  - repaired collect contract yields replay-stable scale-chain fields on this suite/env
- do not reuse its objective-internal dominant-source label as a cross-suite invariant:
  - here the dominant residual bucket is `within_episode0`

Pair2 env12 objective-reset contract probe:

```bash
conda run -n rlpfn python reinforce-terminal-explore/ticl/analysis/phase3_objective_reset_contract_probe.py \
  /home/chen/RLPFN/rwkv/models_diff/rlpfn_03_30_2026_22_37_53_epoch_13.cpkt \
  --suite-name pair2 \
  --env-index 12 \
  --output-json /home/chen/RLPFN/artifacts/phase3_pair2_env12_objective_reset_contract_probe.json \
  --overwrite
```

Notes:

- this is now the formal guardrail for the failed pair2 fresh-collect rerun
- current locked read:
  - stored pair2 distance row claims one objective reset
  - fresh official collect reports zero objective resets and zero objective terminal rows on this env
- therefore:
  - do not treat pair2 env12 as a current official fresh-collect replication point
  - do not rerun the generic pair2 env12 scale-chain as an official stable-set check until this mismatch is resolved
  - the older pair2 local56 chain remains historical artifact-scoped evidence only

Pair2 env12 reset-contract replay-diff probe:

```bash
conda run -n rlpfn python reinforce-terminal-explore/ticl/analysis/phase3_reset_contract_replay_diff_probe.py \
  /home/chen/RLPFN/rwkv/models_diff/rlpfn_03_30_2026_22_37_53_epoch_13.cpkt \
  --suite-name pair2 \
  --env-index 12 \
  --output-json /home/chen/RLPFN/artifacts/phase3_pair2_env12_reset_contract_replay_diff_probe.json \
  --overwrite
```

Notes:

- this is now the narrow source-localization probe for the pair2 reset mismatch
- current locked read:
  - stored pair2 distance row belongs to the legacy unseeded collect contract
  - `legacy_distance_probe_runA` reproduces reset count `1`
  - `legacy_distance_probe_runB` drifts to reset count `2`
  - `official_strict` wires suite env/rollout seeds and yields reset count `0`
- therefore:
  - do not interpret the pair2 mismatch as a new official-path semantic regression
  - interpret it as contract drift between legacy distance artifacts and current strict fresh collect

Official-strict reset census for pair1/pair2:

```bash
conda run -n rlpfn python reinforce-terminal-explore/ticl/analysis/phase3_official_strict_reset_census.py \
  /home/chen/RLPFN/rwkv/models_diff/rlpfn_03_30_2026_22_37_53_epoch_13.cpkt \
  --suite-name pair1 \
  --suite-name pair2 \
  --output-json /home/chen/RLPFN/artifacts/phase3_official_strict_reset_census_pair1_pair2.json \
  --overwrite
```

Notes:

- current locked read:
  - official strict scale-chain entry candidates now exist on pair1/pair2:
    - `pair1 env10`
    - `pair1 env13`
    - `pair2 env2`
    - `pair2 env9`
    - `pair2 env13`
- but census is still only an entry-screen:
  - do not promote these envs to replay-stable official replication points until one is rerun as a full saved scale-chain artifact and then repeatB-confirmed

Pair2 env13 terminal-local scale chain under current official strict collect:

```bash
conda run -n rlpfn python reinforce-terminal-explore/ticl/analysis/phase3_terminal_local_scale_chain_probe.py \
  /home/chen/RLPFN/rwkv/models_diff/rlpfn_03_30_2026_22_37_53_epoch_13.cpkt \
  --suite-name pair2 \
  --env-index 13 \
  --output-json /home/chen/RLPFN/artifacts/phase3_pair2_env13_terminal_scale_chain_probe.json \
  --overwrite
```

RepeatB:

```bash
conda run -n rlpfn python reinforce-terminal-explore/ticl/analysis/phase3_terminal_local_scale_chain_probe.py \
  /home/chen/RLPFN/rwkv/models_diff/rlpfn_03_30_2026_22_37_53_epoch_13.cpkt \
  --suite-name pair2 \
  --env-index 13 \
  --output-json /home/chen/RLPFN/artifacts/phase3_pair2_env13_terminal_scale_chain_probe_repeatB.json \
  --overwrite
```

Notes:

- this is now the second official saved/replay-stable scale-chain replication point
- current locked read:
  - selected target local `37`
  - terminal global step `1983`
  - repeatB matches runA exactly on:
    - boundary contract
    - mean source
    - scale collapse
    - conclusions
- do not reuse its objective-internal dominant-source label as a cross-suite invariant:
  - here the dominant residual bucket is `within_episode1`

Official strict terminal-local scale-chain for pair1 env13:

```bash
conda run -n rlpfn python reinforce-terminal-explore/ticl/analysis/phase3_terminal_local_scale_chain_probe.py \
  /home/chen/RLPFN/rwkv/models_diff/rlpfn_03_30_2026_22_37_53_epoch_13.cpkt \
  --suite-name pair1 \
  --env-index 13 \
  --output-json /home/chen/RLPFN/artifacts/phase3_pair1_env13_terminal_scale_chain_probe.json \
  --overwrite
```

RepeatB:

```bash
conda run -n rlpfn python reinforce-terminal-explore/ticl/analysis/phase3_terminal_local_scale_chain_probe.py \
  /home/chen/RLPFN/rwkv/models_diff/rlpfn_03_30_2026_22_37_53_epoch_13.cpkt \
  --suite-name pair1 \
  --env-index 13 \
  --output-json /home/chen/RLPFN/artifacts/phase3_pair1_env13_terminal_scale_chain_probe_repeatB.json \
  --overwrite
```

Notes:

- this is now the third official saved/replay-stable scale-chain replication point
- current locked read:
  - selected target local `22`
  - terminal global step `1968`
  - repeatB matches runA exactly on:
    - boundary contract
    - mean source
    - scale collapse
    - conclusions
- do not reuse its objective-internal dominant-source label as a cross-suite invariant:
  - here the dominant residual bucket is `within_episode1`

Official boundary-scope counterfactual pack across saved/replay-stable points:

```bash
conda run -n rlpfn python reinforce-terminal-explore/ticl/analysis/phase3_official_boundary_scope_counterfactual_pack.py \
  --output-json /home/chen/RLPFN/artifacts/phase3_official_boundary_scope_counterfactual_pack.json \
  --overwrite
```

Notes:

- this is a read-only pack over:
  - `seed13579 env12`
  - `pair2 env13`
  - `pair1 env13`
- current locked read:
  - all three current points already use `current_full_rollout`
  - all three `objective_all_tokens` counterfactuals worsen boundary penalty
  - all three `objective_all_tokens` counterfactuals worsen pre-value reward-norm magnitude
- use this pack as a guardrail for a narrow contract lock:
  - preserve full-rollout terminal boundary centering
  - do not treat objective-scope terminal centering as a valid fix path

Runtime terminal-boundary drift scan for current Phase 3 train collect path:

```bash
conda run -n rlpfn python reinforce-terminal-explore/ticl/analysis/phase3_runtime_terminal_boundary_drift_scan.py \
  /home/chen/RLPFN/rwkv/models_diff/rlpfn_03_30_2026_22_37_53_epoch_13.cpkt \
  --suite-split train \
  --output-json /home/chen/RLPFN/artifacts/phase3_runtime_terminal_boundary_drift_scan_train.json \
  --overwrite
```

Notes:

- this scans the live train collect path, not saved point artifacts
- current locked read:
  - suites scanned:
    - `pair1`
    - `pair2`
    - `seed13579`
  - objective terminal token count `17`
  - boundary shift mismatch count `0`
  - max abs contract error `8.6939e-08`
- use this to prove whether the current train collector still violates the full-rollout boundary guardrail:
  - current answer is no

Runtime buffer transport contract scan for current Phase 3 train collect path:

```bash
conda run -n rlpfn python reinforce-terminal-explore/ticl/analysis/phase3_shortrun_buffer_transport_contract_scan.py \
  /home/chen/RLPFN/rwkv/models_diff/rlpfn_03_30_2026_22_37_53_epoch_13.cpkt \
  --overwrite
```

Notes:

- this scans the live train collect path and the minibatch transport path
- current locked read:
  - suites scanned:
    - `pair1`
    - `pair2`
    - `seed13579`
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
- use this as the regression gate for any future minibatch packing or train-side slice changes
Regime-labeled train-side contrast complement:

```bash
conda run -n rlpfn python reinforce-terminal-explore/ticl/analysis/phase3_train_env_quality_contrast_probe.py \
  --rollout-quality-json /home/chen/RLPFN/artifacts/phase3_train_rollout_quality_probe_seed24680_with_delta.json \
  --suite-name seed24680 \
  --suite-regime nonpositive \
  --expected-train-suite-summary-json /home/chen/RLPFN/artifacts/phase3_train_rollout_quality_probe_seed24680.json \
  --suite-regime-summary-json /home/chen/RLPFN/artifacts/phase3_cross_suite_regime_summary.json \
  --output-json /home/chen/RLPFN/artifacts/phase3_train_env_quality_contrast_regime_seed24680_nonpositive.json \
  --overwrite
```

- Notes:
  - this is the complementary nonpositive-side train contrast for the same wrapper shape used on the positive regime
  - it uses the recovered seed24680 post-policy bundle plus the reused subset-delta layer
  - locked reading:
    - `suite_name = seed24680`
    - `suite_regime = nonpositive`
    - `train_suite_fingerprint = d7c6a96c0f77cd30`
  - `high_mass_nonpositive = [1, 2]`
  - `low_mass_positive = [0, 3]`
  - `high positive mass alone is not sufficient for positive delta`
  - use this as the guardrailed nonpositive counterpart when comparing train-side update concentration across regimes

Regime comparison summary pack:

```bash
conda run -n rlpfn python reinforce-terminal-explore/ticl/analysis/phase3_train_env_quality_regime_compare_pack.py \
  --output-json /home/chen/RLPFN/artifacts/phase3_train_env_quality_regime_compare_pack.json \
  --overwrite
```

- Notes:
  - compares the positive and nonpositive regime-labeled contrast artifacts side by side
  - locked reading:
    - shared signals are stable across both regimes
    - early-mass separation is positive-only
    - pooled weighting repair is not yet justified
  - use this summary pack as the decision gate before any weighting tweak

Strict train-side A/B for the branch-specific runtime mode:

```bash
conda run -n rlpfn python -m ticl.analysis.phase3_train_rollout_quality_probe \
  /home/chen/RLPFN/rwkv/models_diff/rlpfn_03_30_2026_22_37_53_epoch_13.cpkt \
  --train-suite-path /home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_24680/train_suite.pt \
  --heldout-suite-path /home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_24680/heldout_suite.pt \
  --reuse-zero-control-json /home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_24680/zero_cross_env_control.json \
  --reuse-pre-policy-json /home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_24680/ppo_cross_env_baseline.json \
  --output-json /home/chen/RLPFN/artifacts/phase3_train_rollout_quality_probe_seed24680.json \
  --overwrite \
  --device cuda:0

conda run -n rlpfn python -m ticl.analysis.phase3_train_rollout_quality_probe \
  /home/chen/RLPFN/rwkv/models_diff/rlpfn_03_30_2026_22_37_53_epoch_13.cpkt \
  --train-suite-path /home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_24680/train_suite.pt \
  --heldout-suite-path /home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_24680/heldout_suite.pt \
  --reuse-zero-control-json /home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_24680/zero_cross_env_control.json \
  --reuse-pre-policy-json /home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_24680/ppo_cross_env_baseline.json \
  --output-json /home/chen/RLPFN/artifacts/phase3_train_rollout_quality_probe_seed24680_env12_mode.json \
  --overwrite \
  --device cuda:0 \
  --actor-objective-mode-override tokenwise_scale_env12_mid_episode_extension_block

conda run -n rlpfn python -m ticl.analysis.phase3_train_rollout_quality_mode_side_effect_compare_pack \
  --baseline-json /home/chen/RLPFN/artifacts/phase3_train_rollout_quality_probe_seed24680.json \
  --override-json /home/chen/RLPFN/artifacts/phase3_train_rollout_quality_probe_seed24680_env12_mode.json \
  --output-json /home/chen/RLPFN/artifacts/phase3_train_rollout_quality_mode_side_effect_compare_pack_seed24680.json \
  --overwrite
```

- Notes:
  - this is the strict A/B contract for the runtime-feasible branch-specific control
  - locked reading:
    - `max_abs_metric_delta = 0.0`
    - `runtime_mode_no_other_suite_side_effects = true`
    - `side_effects_are_detectable_in_train_side_ab = false`
  - use this as the no-leak regression check before any future runtime use of `tokenwise_scale_env12_mid_episode_extension_block`

Second non-target strict A/B check:

```bash
conda run -n rlpfn python -m ticl.analysis.phase3_train_rollout_quality_probe \
  /home/chen/RLPFN/rwkv/models_diff/rlpfn_03_30_2026_22_37_53_epoch_13.cpkt \
  --train-suite-path /home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_13579/train_suite.pt \
  --heldout-suite-path /home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_13579/heldout_suite.pt \
  --reuse-zero-control-json /home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_13579/zero_cross_env_control.json \
  --reuse-pre-policy-json /home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_13579/ppo_cross_env_baseline.json \
  --output-json /home/chen/RLPFN/artifacts/phase3_train_rollout_quality_probe_seed13579.json \
  --overwrite \
  --device cuda:0

conda run -n rlpfn python -m ticl.analysis.phase3_train_rollout_quality_probe \
  /home/chen/RLPFN/rwkv/models_diff/rlpfn_03_30_2026_22_37_53_epoch_13.cpkt \
  --train-suite-path /home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_13579/train_suite.pt \
  --heldout-suite-path /home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_13579/heldout_suite.pt \
  --reuse-zero-control-json /home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_13579/zero_cross_env_control.json \
  --reuse-pre-policy-json /home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_13579/ppo_cross_env_baseline.json \
  --output-json /home/chen/RLPFN/artifacts/phase3_train_rollout_quality_probe_seed13579_env12_mode.json \
  --overwrite \
  --device cuda:0 \
  --actor-objective-mode-override tokenwise_scale_env12_mid_episode_extension_block

conda run -n rlpfn python -m ticl.analysis.phase3_train_rollout_quality_mode_side_effect_compare_pack \
  --baseline-json /home/chen/RLPFN/artifacts/phase3_train_rollout_quality_probe_seed13579.json \
  --override-json /home/chen/RLPFN/artifacts/phase3_train_rollout_quality_probe_seed13579_env12_mode.json \
  --output-json /home/chen/RLPFN/artifacts/phase3_train_rollout_quality_mode_side_effect_compare_pack_seed13579.json \
  --overwrite
```

- Notes:
  - second non-target suite confirms the runtime mode stays local
  - locked reading:
    - `max_abs_metric_delta = 0.0`
    - `runtime_mode_no_other_suite_side_effects = true`
  - freeze the mode as a verified small control block once no further local changes are needed

Third non-target strict A/B check on the pair2 suite path:

```bash
conda run -n rlpfn python -m ticl.analysis.phase3_train_rollout_quality_probe \
  /home/chen/RLPFN/rwkv/models_diff/rlpfn_03_30_2026_22_37_53_epoch_13.cpkt \
  --train-suite-path /home/chen/RLPFN/artifacts/phase3_cross_env_baseline_pair2/suites/train_suite.pt \
  --heldout-suite-path /home/chen/RLPFN/artifacts/phase3_cross_env_baseline_pair2/suites/heldout_suite.pt \
  --reuse-zero-control-json /home/chen/RLPFN/artifacts/phase3_cross_env_baseline_pair2/zero_cross_env_control.json \
  --reuse-pre-policy-json /home/chen/RLPFN/artifacts/phase3_cross_env_baseline_pair2/ppo_cross_env_baseline.json \
  --output-json /home/chen/RLPFN/artifacts/phase3_train_rollout_quality_probe_pair2.json \
  --overwrite \
  --device cuda:0

conda run -n rlpfn python -m ticl.analysis.phase3_train_rollout_quality_probe \
  /home/chen/RLPFN/rwkv/models_diff/rlpfn_03_30_2026_22_37_53_epoch_13.cpkt \
  --train-suite-path /home/chen/RLPFN/artifacts/phase3_cross_env_baseline_pair2/suites/train_suite.pt \
  --heldout-suite-path /home/chen/RLPFN/artifacts/phase3_cross_env_baseline_pair2/suites/heldout_suite.pt \
  --reuse-zero-control-json /home/chen/RLPFN/artifacts/phase3_cross_env_baseline_pair2/zero_cross_env_control.json \
  --reuse-pre-policy-json /home/chen/RLPFN/artifacts/phase3_cross_env_baseline_pair2/ppo_cross_env_baseline.json \
  --output-json /home/chen/RLPFN/artifacts/phase3_train_rollout_quality_probe_pair2_env12_mode.json \
  --overwrite \
  --device cuda:0 \
  --actor-objective-mode-override tokenwise_scale_env12_mid_episode_extension_block

conda run -n rlpfn python -m ticl.analysis.phase3_train_rollout_quality_mode_side_effect_compare_pack \
  --baseline-json /home/chen/RLPFN/artifacts/phase3_train_rollout_quality_probe_pair2.json \
  --override-json /home/chen/RLPFN/artifacts/phase3_train_rollout_quality_probe_pair2_env12_mode.json \
  --output-json /home/chen/RLPFN/artifacts/phase3_train_rollout_quality_mode_side_effect_compare_pack_pair2.json \
  --overwrite
```

- Notes:
  - third non-target suite path also returns `max_abs_metric_delta = 0.0`
  - runtime mode is frozen as the verified runtime-feasible branch-specific control candidate

Proxy-to-train transfer audit for the pair2 runtime control:

```bash
conda run -n rlpfn python /home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_pair2_proxy_to_train_transfer_audit.py \
  --pair2-train-update-ab-compare-json /home/chen/RLPFN/artifacts/phase3_pair2_train_update_ab_compare_pack.json \
  --pair2-train-rollout-quality-json /home/chen/RLPFN/artifacts/phase3_train_rollout_quality_probe_pair2.json \
  --env12-counterfactual-json /home/chen/RLPFN/artifacts/phase3_env12_mid_episode_extension_counterfactual_pack.json \
  --target-env-index 12 \
  --output-json /home/chen/RLPFN/artifacts/phase3_pair2_proxy_to_train_transfer_audit.json
```

- Notes:
  - use this after the pair2 target-side A/B has been written
  - locked reading:
    - local proxy shrink is real but too small relative to pair2 batch-weighted actor-advantage mass
    - this explains why the runtime control can be locally valid yet fail to improve train-side gain
  - use this to route the next analysis toward update aggregation semantics:
    - advantage normalization
    - PPO clipping
    - many-env batch mixing
