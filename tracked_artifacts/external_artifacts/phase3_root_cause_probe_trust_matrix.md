## Phase 3 Root-Cause Probe Trust Matrix

### Why this matrix exists

- The current bottleneck work is only worth继续 if the probes we rely on are semantically aligned.
- After the suite-selection contract lock, the main question is no longer "do we have more probes?"
- It is "which existing probes still support root-cause conclusions under the locked contract?"

### Trusted probes

- [phase3_pair2_proxy_to_train_transfer_audit.json](/home/chen/RLPFN/artifacts/phase3_pair2_proxy_to_train_transfer_audit.json)
  - Status: trusted for the train-side mass-dilution conclusion
  - Reason:
    - it is explicitly train-side
    - it checks `train_suite_fingerprint` against the rollout-quality artifact before drawing conclusions
    - it does not rely on same-suite env-local source/current identity matching
  - Locked reading:
    - local proxy gain can be real
    - yet still be too small relative to whole-batch / whole-env mass to change train-side gain

- [phase3_pair2_train_segment_identity_probe.json](/home/chen/RLPFN/artifacts/phase3_pair2_train_segment_identity_probe.json)
  - Status: trusted as a train-side identity probe
  - Reason:
    - it intentionally compares a saved source block against the strict train-side runtime path
    - it is not a same-suite reset-contract probe
    - its conclusions are about train-side identity / transport semantics only
  - Locked reading:
    - flatten and outer-batch transport preserve rollout identity
    - the relevant mismatch already exists at train-side rollout identity

### Conditionally trusted probes

- [phase3_pair2_exact_update_aggregation_audit.json](/home/chen/RLPFN/artifacts/phase3_pair2_exact_update_aggregation_audit.json)
  - Status: conditionally trusted
  - Trusted scope:
    - stage-wise batch-level dilution exists
    - normalization and ratio/clipping are real dilution stages on the captured strict snapshot
  - Not trusted scope:
    - exact env12 branch-block attribution
  - Reason:
    - the requested target block is absent from the captured outer batch
    - the artifact itself says:
      - `current_single_snapshot_is_insufficient_for_env12_block_audit = true`
      - `requested_target_block_present_in_captured_outer_batch = false`
  - Locked reading:
    - use it for batch-level transfer semantics
    - do not use it as exact proof about the env12 four-token block

### Closed / superseded confusion source

- [phase3_pair2_env12_window_reset_segment_compare.json](/home/chen/RLPFN/artifacts/phase3_pair2_env12_window_reset_segment_compare.json)
  - Status: trusted after suite-contract lock
  - Role:
    - not a direct optimization bottleneck probe
    - now only a guardrail proving that source/current compare defaults to same-source-suite parity
  - Locked reading:
    - do not use mixed-suite compare artifacts to infer optimization bottlenecks

### Current bottleneck axis that remains valid

- The strongest still-trusted root-cause line is:
  - `proxy_to_train_update_transfer_semantics`
- In plain terms:
  - the branch-specific local control can improve a local proxy
  - but the many-env train-side update may still ignore or dilute that gain before it becomes useful policy movement

### What is not yet justified

- It is not yet justified to claim:
  - the env12 branch block is exactly neutralized at a known train-side update stage
- Because the exact target block is not yet present in the captured strict outer batch snapshot

### Single next-step candidate

- If the bottleneck line continues, the next single probe should be:
  - a train-side update-transfer semantics probe that targets the actually captured strict batch scope
  - not a return to mixed-suite source/current compare
  - not more tuning of the branch-specific block itself
