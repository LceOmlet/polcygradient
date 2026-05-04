import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_SELECTOR_TRACE_JSON = (
    "/home/chen/RLPFN/artifacts/phase3_pair2_guarded_target_present_selector_trace.json"
)
DEFAULT_EXACT_AUDIT_JSON = (
    "/home/chen/RLPFN/artifacts/phase3_pair2_exact_update_aggregation_audit.json"
)
DEFAULT_COMPARE_PACK_JSON = (
    "/home/chen/RLPFN/artifacts/phase3_pair2_train_update_ab_compare_pack.json"
)
DEFAULT_OUTPUT_JSON = (
    "/home/chen/RLPFN/artifacts/phase3_pair2_repaired_selector_epoch_dilution_probe.json"
)


def _load_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def _float(value: Any) -> float:
    return float(value)


def build_pair2_repaired_selector_epoch_dilution_probe(
    *,
    selector_trace_json: str,
    exact_audit_json: str,
    compare_pack_json: str,
) -> dict[str, Any]:
    selector_trace = _load_json(selector_trace_json)
    exact_audit = _load_json(exact_audit_json)
    compare_pack = _load_json(compare_pack_json)

    trace_conclusions = dict(selector_trace["conclusions"])
    compare_conclusions = dict(compare_pack["conclusions"])
    exact_conclusions = dict(exact_audit["conclusions"])
    if not bool(trace_conclusions.get("target_present_in_baseline", False)):
        raise ValueError("selector trace baseline arm does not contain the repaired target selector")
    if not bool(trace_conclusions.get("target_present_in_override", False)):
        raise ValueError("selector trace override arm does not contain the repaired target selector")
    if bool(trace_conclusions.get("current_selector_definition_cannot_fire_under_trusted_contract", True)):
        raise ValueError("selector trace still reports the repaired selector as unfireable")
    if not bool(compare_conclusions.get("target_side_compare_is_efficacy_eligible", False)):
        raise ValueError("compare pack is not efficacy-eligible under the repaired selector contract")
    if not bool(exact_conclusions.get("requested_target_block_present_in_captured_outer_batch", False)):
        raise ValueError("exact aggregation audit does not contain the repaired target block")

    baseline_trace = dict(selector_trace["baseline"]["selector_trace"])
    rows = list(baseline_trace["rows"])
    if not rows:
        raise ValueError("selector trace has no rows")
    total_outer_batches = int(rows[0]["total_outer_batches"])
    if int(len(rows)) != int(total_outer_batches):
        raise ValueError("selector trace row count does not match total_outer_batches")
    if any(int(row["total_outer_batches"]) != int(total_outer_batches) for row in rows):
        raise ValueError("selector trace total_outer_batches is not constant")

    objective_totals = [int(row["objective_total"]) for row in rows]
    if any(int(v) != int(objective_totals[0]) for v in objective_totals):
        raise ValueError("objective_total is not constant across the trusted selector trace rows")
    objective_total_per_outer_batch = int(objective_totals[0])

    target_rows = [dict(row) for row in rows if int(row["target_match_count"]) > 0]
    if len(target_rows) != 1:
        raise ValueError(
            f"expected exactly one target-present outer batch under the repaired selector contract, got {len(target_rows)}"
        )
    target_row = target_rows[0]
    target_outer_batch_idx = int(target_row["outer_batch_idx"])

    target_selector = dict(selector_trace["target_selector"])
    if int(target_selector["target_outer_batch_idx"]) != int(target_outer_batch_idx):
        raise ValueError("selector trace target_outer_batch_idx does not match the actual target-present row")
    if int(target_selector["target_env_index"]) != int(exact_audit["target_block"]["env_index"]):
        raise ValueError("selector trace target env does not match exact audit target block env")

    exact_contract = dict(exact_audit["strict_contract"])
    if int(exact_contract["outer_batch_idx"]) != int(target_outer_batch_idx):
        raise ValueError("exact aggregation outer_batch_idx does not match the selector trace target row")
    if int(exact_contract["objective_total"]) != int(objective_total_per_outer_batch):
        raise ValueError("exact aggregation objective_total does not match selector trace objective_total")
    if str(exact_contract["runtime_suite_name"]) != "pair2":
        raise ValueError("exact aggregation audit is not on the repaired pair2 runtime scope")

    target_token_count = int(exact_audit["target_block"]["block_token_count"])
    total_epoch_objective_tokens = int(total_outer_batches * objective_total_per_outer_batch)
    target_token_fraction_of_target_batch = _float(target_token_count / objective_total_per_outer_batch)
    target_token_fraction_of_epoch_objective = _float(target_token_count / total_epoch_objective_tokens)
    target_step_fraction_of_epoch = _float(1.0 / float(total_outer_batches))
    target_epoch_position_fraction = _float(target_outer_batch_idx / total_outer_batches)
    remaining_outer_batches_after_target = int(total_outer_batches - target_outer_batch_idx)
    remaining_outer_batch_fraction_after_target = _float(
        remaining_outer_batches_after_target / float(total_outer_batches)
    )

    policy_delta = dict(exact_audit["delta"]["policy_num_from_clipped_objective"])
    pre_delta = dict(exact_audit["delta"]["pre_normalization_actor_advantages"])
    policy_negative_mass_shrink_single_batch = _float(policy_delta["block_negative_mass_shrink"])
    pre_negative_mass_shrink_single_batch = _float(pre_delta["block_negative_mass_shrink"])
    policy_batch_negative_mass_share_delta_single_batch = abs(
        _float(policy_delta["block_share_of_batch_negative_mass_delta"])
    )
    pre_batch_negative_mass_share_delta_single_batch = abs(
        _float(pre_delta["block_share_of_batch_negative_mass_delta"])
    )
    policy_negative_mass_shrink_epoch_uniform_heuristic = _float(
        policy_negative_mass_shrink_single_batch / float(total_outer_batches)
    )
    pre_negative_mass_shrink_epoch_uniform_heuristic = _float(
        pre_negative_mass_shrink_single_batch / float(total_outer_batches)
    )
    policy_batch_negative_mass_share_epoch_uniform_heuristic = _float(
        policy_batch_negative_mass_share_delta_single_batch / float(total_outer_batches)
    )
    pre_batch_negative_mass_share_epoch_uniform_heuristic = _float(
        pre_batch_negative_mass_share_delta_single_batch / float(total_outer_batches)
    )

    compare_delta = dict(compare_pack["comparison"])
    train_history_delta = dict(compare_delta["train_history_delta"])
    train_full_return_delta_change = _float(compare_delta["train_full_return_delta_change"])
    train_suffix_return_delta_change = _float(compare_delta["train_suffix_return_delta_change"])
    numeric_noise_eps = 1e-4

    return {
        "probe_entry": "phase3_pair2_repaired_selector_epoch_dilution_probe",
        "inputs": {
            "selector_trace_json": str(Path(selector_trace_json).expanduser().resolve()),
            "exact_audit_json": str(Path(exact_audit_json).expanduser().resolve()),
            "compare_pack_json": str(Path(compare_pack_json).expanduser().resolve()),
        },
        "trusted_contract": {
            "checkpoint_path": str(compare_pack["shared_contract_context"]["checkpoint_path"]),
            "train_suite_path": str(compare_pack["shared_contract_context"]["train_suite_path"]),
            "single_eval_pos": int(compare_pack["shared_contract_context"]["single_eval_pos"]),
            "n_samples": int(compare_pack["shared_contract_context"]["n_samples"]),
            "strict_fixed_env_mode": bool(compare_pack["shared_contract_context"]["strict_fixed_env_mode"]),
            "deterministic_actor_sampling": bool(compare_pack["shared_contract_context"]["deterministic_actor_sampling"]),
            "deterministic_batch_plan": bool(compare_pack["shared_contract_context"]["deterministic_batch_plan"]),
            "strict_native_rollout": bool(compare_pack["shared_contract_context"]["strict_native_rollout"]),
            "ppo_restore_validation_policy_head_state": bool(
                compare_pack["shared_contract_context"]["ppo_restore_validation_policy_head_state"]
            ),
            "runtime_suite_name": str(exact_contract["runtime_suite_name"]),
        },
        "epoch_layout": {
            "total_outer_batches": int(total_outer_batches),
            "objective_total_per_outer_batch": int(objective_total_per_outer_batch),
            "total_epoch_objective_tokens": int(total_epoch_objective_tokens),
            "target_present_outer_batches_per_epoch": int(len(target_rows)),
            "target_outer_batch_idx": int(target_outer_batch_idx),
            "remaining_outer_batches_after_target": int(remaining_outer_batches_after_target),
            "target_step_fraction_of_epoch": float(target_step_fraction_of_epoch),
            "target_epoch_position_fraction": float(target_epoch_position_fraction),
            "remaining_outer_batch_fraction_after_target": float(
                remaining_outer_batch_fraction_after_target
            ),
        },
        "target_scope": {
            "selector": {
                "target_outer_batch_idx": int(target_selector["target_outer_batch_idx"]),
                "target_env_index": int(target_selector["target_env_index"]),
                "target_objective_episode_index": int(target_selector["target_objective_episode_index"]),
                "target_objective_position_start": int(target_selector["target_objective_position_start"]),
                "target_objective_position_end": int(target_selector["target_objective_position_end"]),
            },
            "token_count": int(target_token_count),
            "target_match_count": int(target_row["target_match_count"]),
            "target_token_fraction_of_target_batch": float(target_token_fraction_of_target_batch),
            "target_token_fraction_of_epoch_objective": float(target_token_fraction_of_epoch_objective),
        },
        "local_to_epoch_dilution": {
            "pre_negative_mass_shrink_single_batch": float(pre_negative_mass_shrink_single_batch),
            "policy_negative_mass_shrink_single_batch": float(policy_negative_mass_shrink_single_batch),
            "pre_negative_mass_shrink_epoch_uniform_heuristic": float(
                pre_negative_mass_shrink_epoch_uniform_heuristic
            ),
            "policy_negative_mass_shrink_epoch_uniform_heuristic": float(
                policy_negative_mass_shrink_epoch_uniform_heuristic
            ),
            "pre_batch_negative_mass_share_delta_single_batch_abs": float(
                pre_batch_negative_mass_share_delta_single_batch
            ),
            "policy_batch_negative_mass_share_delta_single_batch_abs": float(
                policy_batch_negative_mass_share_delta_single_batch
            ),
            "pre_batch_negative_mass_share_epoch_uniform_heuristic": float(
                pre_batch_negative_mass_share_epoch_uniform_heuristic
            ),
            "policy_batch_negative_mass_share_epoch_uniform_heuristic": float(
                policy_batch_negative_mass_share_epoch_uniform_heuristic
            ),
        },
        "observed_train_side_outcome": {
            "target_side_effect_is_only_numeric_noise": bool(
                compare_conclusions["target_side_effect_is_only_numeric_noise"]
            ),
            "target_side_train_update_changes_trajectory": bool(
                compare_conclusions["target_side_train_update_changes_trajectory"]
            ),
            "train_history_identical": bool(train_history_delta["histories_identical"]),
            "train_history_max_abs_metric_delta": float(train_history_delta["max_abs_metric_delta"]),
            "train_full_return_delta_change": float(train_full_return_delta_change),
            "train_suffix_return_delta_change": float(train_suffix_return_delta_change),
        },
        "conclusions": {
            "target_scope_perturbs_exactly_one_outer_batch_per_epoch": bool(len(target_rows) == 1),
            "target_scope_is_applied_in_only_one_optimizer_step_per_epoch": bool(len(target_rows) == 1),
            "target_scope_enters_only_in_the_final_quarter_of_epoch": bool(
                target_epoch_position_fraction >= 0.75
            ),
            "target_scope_directly_touches_about_one_per_thousand_of_epoch_objective_tokens": bool(
                target_token_fraction_of_epoch_objective <= 0.0015
            ),
            "epoch_weighted_policy_negative_mass_share_perturbation_is_subpermille": bool(
                policy_batch_negative_mass_share_epoch_uniform_heuristic < 1e-3
            ),
            "later_same_contract_outer_batches_can_overwrite_the_single_target_step": bool(
                remaining_outer_batches_after_target > 0
            ),
            "local_shrink_to_train_noise_is_explained_by_single_late_step_dilution": bool(
                bool(compare_conclusions["target_side_effect_is_only_numeric_noise"])
                and bool(train_history_delta["histories_identical"])
                and policy_batch_negative_mass_share_epoch_uniform_heuristic < 1e-3
                and len(target_rows) == 1
                and target_epoch_position_fraction >= 0.75
            ),
        },
        "recommendation": {
            "next_focus": "shared_core_plus_branch_specific_extension_boundary",
            "reason": (
                "Under the repaired trusted contract, the env12 branch-specific gate is not being erased by a stale "
                "selector bug. Its train-side effect is narrow because it perturbs one late outer-batch step, touches "
                "about one-per-thousand of epoch objective tokens, and produces only a subpermille epoch-weighted "
                "policy negative-mass share change. The next small-control question is therefore about which class of "
                "branch-specific extensions can clear this transfer floor without widening into a shared-core repair."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Explain why the repaired pair2 local env12 control remains train-side numeric noise."
    )
    parser.add_argument("--selector-trace-json", type=str, default=DEFAULT_SELECTOR_TRACE_JSON)
    parser.add_argument("--exact-audit-json", type=str, default=DEFAULT_EXACT_AUDIT_JSON)
    parser.add_argument("--compare-pack-json", type=str, default=DEFAULT_COMPARE_PACK_JSON)
    parser.add_argument("--output-json", type=str, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output_json).expanduser().resolve()
    if output_path.exists() and not bool(args.overwrite):
        print(output_path.read_text(encoding="utf-8"))
        return

    report = build_pair2_repaired_selector_epoch_dilution_probe(
        selector_trace_json=args.selector_trace_json,
        exact_audit_json=args.exact_audit_json,
        compare_pack_json=args.compare_pack_json,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
