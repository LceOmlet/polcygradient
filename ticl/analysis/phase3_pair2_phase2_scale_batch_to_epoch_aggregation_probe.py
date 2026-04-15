import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_SELECTOR_TRACE_JSON = (
    "/home/chen/RLPFN/artifacts/phase3_pair2_phase2_scale_target_present_recapture_selector_trace.json"
)
DEFAULT_EXACT_AUDIT_JSON = (
    "/home/chen/RLPFN/artifacts/phase3_pair2_phase2_scale_exact_update_aggregation_audit.json"
)
DEFAULT_OUTPUT_JSON = (
    "/home/chen/RLPFN/artifacts/phase3_pair2_phase2_scale_batch_to_epoch_aggregation_probe.json"
)


def _load_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).expanduser().resolve().read_text())


def build_pair2_phase2_scale_batch_to_epoch_aggregation_probe(
    *,
    selector_trace_json: str,
    exact_audit_json: str,
) -> dict[str, Any]:
    selector_trace = _load_json(selector_trace_json)
    exact_audit = _load_json(exact_audit_json)

    rows = list(selector_trace["rows"])
    if not rows:
        raise ValueError("selector trace has no rows")

    target_scope = dict(exact_audit["target_scope"])
    target_selector = dict(target_scope["selector"])
    target_outer_batch_idx = int(target_selector["outer_batch_idx"])
    target_env_index = int(target_selector["env_index"])

    total_outer_batches = int(rows[0]["total_outer_batches"])
    if any(int(row["total_outer_batches"]) != total_outer_batches for row in rows):
        raise ValueError("selector trace total_outer_batches is not constant")
    if int(len(rows)) != total_outer_batches:
        raise ValueError("selector trace row count does not match total_outer_batches")

    objective_totals = [int(row["objective_total"]) for row in rows]
    objective_env_lists = [tuple(int(v) for v in row["objective_env_indices_present"]) for row in rows]
    singleton_env_outer_batches = all(len(envs) == 1 for envs in objective_env_lists)
    constant_objective_total = all(v == objective_totals[0] for v in objective_totals)

    target_rows = [dict(row) for row in rows if int(row["outer_batch_idx"]) == target_outer_batch_idx]
    if len(target_rows) != 1:
        raise ValueError(f"expected exactly one target outer batch row, got {len(target_rows)}")
    target_row = target_rows[0]
    if int(target_row["objective_env_indices_present"][0]) != target_env_index:
        raise ValueError("target outer batch env identity does not match exact audit selector")

    target_env_batches = [
        int(row["outer_batch_idx"])
        for row in rows
        if int(len(row["objective_env_indices_present"])) == 1
        and int(row["objective_env_indices_present"][0]) == target_env_index
    ]
    target_present_batches = [
        int(row["outer_batch_idx"])
        for row in rows
        if int(row["target_match_count"]) > 0
    ]

    target_step_fraction = float(1.0 / float(total_outer_batches))
    target_share_of_batch_negative_mass_after_clipping = float(
        exact_audit["stage_transition"]["target_share_of_batch_negative_mass"]["policy_num"]
    )
    uniform_step_weight_epoch_share_heuristic = float(
        target_share_of_batch_negative_mass_after_clipping * target_step_fraction
    )

    target_outer_batch_idx_1based = int(target_outer_batch_idx)
    target_batches_after = int(total_outer_batches - target_outer_batch_idx_1based)
    target_batches_before = int(target_outer_batch_idx_1based - 1)

    return {
        "probe_entry": "phase3_pair2_phase2_scale_batch_to_epoch_aggregation_probe",
        "inputs": {
            "selector_trace_json": str(Path(selector_trace_json).expanduser().resolve()),
            "exact_audit_json": str(Path(exact_audit_json).expanduser().resolve()),
        },
        "train_loop_contract": {
            "optimizer_step_per_outer_batch": True,
            "outer_batch_loss_normalized_by_that_batch_objective_total": True,
            "sequential_outer_batch_updates_within_epoch": True,
        },
        "epoch_layout": {
            "total_outer_batches": total_outer_batches,
            "singleton_env_outer_batches": bool(singleton_env_outer_batches),
            "constant_objective_total": bool(constant_objective_total),
            "objective_total_per_outer_batch": int(objective_totals[0]),
            "outer_batch_env_order": [
                {
                    "outer_batch_idx": int(row["outer_batch_idx"]),
                    "env_index": int(row["objective_env_indices_present"][0]) if row["objective_env_indices_present"] else None,
                }
                for row in rows
            ],
        },
        "target_scope_epoch_context": {
            "target_outer_batch_idx": target_outer_batch_idx_1based,
            "target_env_index": target_env_index,
            "target_env_batches_per_epoch": int(len(target_env_batches)),
            "target_present_batches_per_epoch": int(len(target_present_batches)),
            "target_batches_before_in_epoch": target_batches_before,
            "target_batches_after_in_epoch": target_batches_after,
            "target_step_fraction_of_epoch": target_step_fraction,
            "target_share_of_batch_negative_mass_after_clipping": target_share_of_batch_negative_mass_after_clipping,
            "uniform_step_weight_epoch_share_heuristic": uniform_step_weight_epoch_share_heuristic,
            "target_batch_objective_episode_spans": list(target_row["target_env_objective_episode_spans"]),
        },
        "conclusions": {
            "no_intra_batch_cross_env_mixing_in_trusted_epoch": bool(singleton_env_outer_batches),
            "target_scope_hits_exactly_one_outer_batch_per_epoch": bool(len(target_present_batches) == 1),
            "target_scope_is_a_sequential_single_step_effect_within_epoch": True,
            "cross_batch_overwrite_is_a_more_plausible_bottleneck_than_intra_batch_mixing": bool(singleton_env_outer_batches),
        },
        "recommendation": {
            "next_focus": "cross_batch_overwrite_read_only_probe",
            "reason": (
                "The trusted target scope occupies one single-env outer batch and is applied in one optimizer step. "
                "The next narrow question is whether later outer-batch steps counteract or overwrite that local step."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Explain trusted-scale target-block behavior at the batch-to-epoch aggregation level."
    )
    parser.add_argument("--selector-trace-json", type=str, default=DEFAULT_SELECTOR_TRACE_JSON)
    parser.add_argument("--exact-audit-json", type=str, default=DEFAULT_EXACT_AUDIT_JSON)
    parser.add_argument("--output-json", type=str, default=DEFAULT_OUTPUT_JSON)
    args = parser.parse_args()

    report = build_pair2_phase2_scale_batch_to_epoch_aggregation_probe(
        selector_trace_json=args.selector_trace_json,
        exact_audit_json=args.exact_audit_json,
    )
    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
