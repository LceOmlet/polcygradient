## Phase 3 Analysis Suite Contract Audit

### Scope

- Only audited the same class of analysis-layer suite-selection bugs:
  - source artifact readers that might select rows by `env_index` alone
  - source-vs-current compare probes that might silently use a misleading `train_suite_path`
- Did not reopen training-path or rollout-path semantics.

### Result

- Confirmed one real vulnerability:
  - [phase3_pair2_env12_window_reset_segment_compare.py](/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_pair2_env12_window_reset_segment_compare.py)
  - This was the only audited probe found to have both:
    - source-side selection by `env_index` alone
    - a source/current compare path whose suite argument could silently default to the wrong suite role
  - Status:
    - fixed
    - covered by focused tests
    - canonical artifact now defaults to same-source-suite parity

- Confirmed safe source-row readers:
  - [phase3_pair2_flip_mode_split_probe.py](/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_pair2_flip_mode_split_probe.py)
    - filters `token_row_bundles` by `suite_name` before reading rows
  - [phase3_future_carry_source_chain_probe.py](/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_future_carry_source_chain_probe.py)
    - loads token rows by `suite_name + env_index`
  - [phase3_env12_negative_delta_probe.py](/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_env12_negative_delta_probe.py)
    - loads token rows by `suite_name + env_index`
  - [phase3_env12_value_peak_drop_contract_probe.py](/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_env12_value_peak_drop_contract_probe.py)
    - loads token rows by `suite_name + env_index`
  - [phase3_env12_local56_boundary_contract_probe.py](/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_env12_local56_boundary_contract_probe.py)
    - loads token rows by `suite_name + env_index`
  - [phase3_env12_boundary_stepdown_root_probe.py](/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_env12_boundary_stepdown_root_probe.py)
    - loads token rows by `suite_name + env_index`
  - [phase3_env12_local56_boundary_scope_counterfactual_probe.py](/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_env12_local56_boundary_scope_counterfactual_probe.py)
    - loads token rows by `suite_name + env_index`
  - [phase3_env12_local56_mean_source_probe.py](/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_env12_local56_mean_source_probe.py)
    - loads token rows by `suite_name + env_index`

### Intentional Exception

- [phase3_pair2_train_segment_identity_probe.py](/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_pair2_train_segment_identity_probe.py)
  - This probe intentionally compares:
    - a saved source block identity
    - against the strict train-side runtime path
  - It is not a same-suite env-local compare probe.
  - Therefore its use of `train_suite_path` is intentional, not a silent suite-selection bug.
  - Operational reading:
    - do not interpret its output as a same-suite reset-contract compare
    - interpret it as a train-side transport/identity probe only

### Out of Scope

- Scripts that read `distance_probe_json` rows by `env_index` only were not treated as the same bug class here.
  - Reason:
    - those artifacts are suite-specific at the file level
    - the current audit only targeted multi-suite source payloads such as `token_row_bundles`

### Locked Conclusion

- After this audit, no remaining audited `token_row_bundles` reader was found to silently select source rows by `env_index` alone.
- No remaining audited source-vs-current compare probe was found to silently default to `train_suite_path` for a source artifact that implies `heldout` suite role.
- The one confirmed vulnerability in this class has already been fixed and regression-protected.
