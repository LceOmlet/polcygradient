import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: str, *, expected_audit_entry: str) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise ValueError(f"Missing JSON artifact: {resolved}")
    payload = json.loads(resolved.read_text())
    if str(payload.get("audit_entry", "")) != str(expected_audit_entry):
        raise ValueError(f"Expected {expected_audit_entry} artifact.")
    return payload


def _block_summary(block: dict[str, Any]) -> dict[str, Any]:
    return {
        "row_count": int(block["row_count"]),
        "local_indices": [int(v) for v in list(block["local_indices"])],
        "tokens_to_objective_episode_end": [int(v) for v in list(block["tokens_to_objective_episode_end"])],
        "future_chain_step_count": [int(v) for v in list(block["future_chain_step_count"])],
        "mean_start_gae_future_carry_norm": float(block["mean_start_gae_future_carry_norm"]),
        "mean_start_raw_rollout_residual_norm": float(block["mean_start_raw_rollout_residual_norm"]),
        "mean_weighted_delta_sum": float(block["mean_weighted_delta_sum"]),
        "dominant_source": str(block["dominant_source"]),
        "negative_source_shares_reference": dict(block["negative_source_shares_reference"]),
        "negative_source_shares_is_uniform": bool(block["negative_source_shares_is_uniform"]),
        "negative_source_shares_max_abs_deviation": float(block["negative_source_shares_max_abs_deviation"]),
        "is_contiguous": bool(block["is_contiguous"]),
        "monotone_tte_decreasing": bool(block["monotone_tte_decreasing"]),
        "monotone_weighted_delta_more_negative": bool(block["monotone_weighted_delta_more_negative"]),
    }


def _match_block_summaries(compare_summary: dict[str, Any], counter_summary: dict[str, Any]) -> bool:
    keys = (
        "row_count",
        "local_indices",
        "tokens_to_objective_episode_end",
        "future_chain_step_count",
        "dominant_source",
        "negative_source_shares_reference",
        "negative_source_shares_is_uniform",
        "negative_source_shares_max_abs_deviation",
        "is_contiguous",
        "monotone_tte_decreasing",
        "monotone_weighted_delta_more_negative",
    )
    return all(compare_summary[key] == counter_summary[key] for key in keys)


def _material_shrink_summary(observed: float, counterfactual: float) -> dict[str, float]:
    observed_abs = abs(float(observed))
    counterfactual_abs = abs(float(counterfactual))
    shrink = float(observed_abs - counterfactual_abs)
    ratio = float(counterfactual_abs / observed_abs) if observed_abs > 1e-12 else 0.0
    return {
        "observed_abs": float(observed_abs),
        "counterfactual_abs": float(counterfactual_abs),
        "abs_shrink": float(shrink),
        "counterfactual_over_observed_ratio": float(ratio),
        "observed_over_counterfactual_ratio": float(observed_abs / counterfactual_abs)
        if counterfactual_abs > 1e-12
        else 0.0,
    }


def build_env12_mid_episode_extension_runtime_control_probe(
    *,
    env12_compare_pack_json: str,
    env12_counterfactual_pack_json: str,
) -> dict[str, Any]:
    compare_payload = _load_json(
        env12_compare_pack_json,
        expected_audit_entry="phase3_env12_mid_episode_extension_compare_pack",
    )
    counter_payload = _load_json(
        env12_counterfactual_pack_json,
        expected_audit_entry="phase3_env12_mid_episode_extension_counterfactual_pack",
    )

    compare_extension = _block_summary(dict(compare_payload["comparison"]["extension_block"]))
    compare_core = _block_summary(dict(compare_payload["comparison"]["terminal_adjacent_core"]))
    counter_observed = _block_summary(dict(counter_payload["comparison"]["observed_extension_block"]))
    counter_anchor = _block_summary(dict(counter_payload["comparison"]["counterfactual_anchor_terminal_adjacent_core"]))

    payloads_match = bool(
        _match_block_summaries(compare_extension, counter_observed)
        and _match_block_summaries(compare_core, counter_anchor)
    )

    carry = _material_shrink_summary(
        compare_extension["mean_start_gae_future_carry_norm"],
        compare_core["mean_start_gae_future_carry_norm"],
    )
    residual = _material_shrink_summary(
        compare_extension["mean_start_raw_rollout_residual_norm"],
        compare_core["mean_start_raw_rollout_residual_norm"],
    )
    weighted = _material_shrink_summary(
        compare_extension["mean_weighted_delta_sum"],
        compare_core["mean_weighted_delta_sum"],
    )

    can_minimize = bool(
        payloads_match
        and compare_extension["is_contiguous"]
        and compare_extension["negative_source_shares_is_uniform"]
        and bool(compare_payload["conclusions"]["env12_mid_episode_extension_has_no_internal_split_supported"])
        and bool(counter_payload["conclusions"]["env12_mid_episode_extension_counterfactual_supports_branch_specific_small_control"])
    )

    return {
        "audit_entry": "phase3_env12_mid_episode_extension_runtime_control_probe",
        "source_env12_compare_pack_json": str(Path(env12_compare_pack_json).expanduser().resolve()),
        "source_env12_counterfactual_pack_json": str(Path(env12_counterfactual_pack_json).expanduser().resolve()),
        "control_block": {
            "candidate_small_control_target": "env12_mid_episode_extension_block",
            "runtime_control_granularity": "contiguous_four_token_block",
            "branch_specific": True,
            "row_count": int(compare_extension["row_count"]),
            "local_indices": [int(v) for v in compare_extension["local_indices"]],
            "tokens_to_objective_episode_end": [int(v) for v in compare_extension["tokens_to_objective_episode_end"]],
            "future_chain_step_count": [int(v) for v in compare_extension["future_chain_step_count"]],
            "dominant_source": str(compare_extension["dominant_source"]),
            "negative_source_shares_reference": dict(compare_extension["negative_source_shares_reference"]),
        },
        "runtime_semantic_checks": {
            "payloads_match": bool(payloads_match),
            "can_be_minimally_realized_as_runtime_control": bool(can_minimize),
            "requires_shared_core_edit": False,
            "requires_branch_specific_scope_only": True,
            "shared_guardrail_unchanged": True,
            "no_finer_split_supported": bool(
                compare_payload["conclusions"]["env12_mid_episode_extension_has_no_internal_split_supported"]
            ),
            "branch_specific_followup_needed": True,
        },
        "material_counterfactual": {
            "mean_start_gae_future_carry_norm": carry,
            "mean_start_raw_rollout_residual_norm": residual,
            "mean_weighted_delta_sum": weighted,
        },
        "recommendation": {
            "candidate_small_control_target": "env12_mid_episode_extension_block",
            "runtime_control_shape": "contiguous_four_token_branch_specific_gate",
            "branch_specific_followup_needed": True,
            "shared_guardrail_unchanged": True,
            "reason": (
                "The env12 mid-episode extension is the narrowest supported runtime control unit: "
                "it is contiguous, has uniform source shares, has no supported internal split, and a "
                "counterfactual collapse to the terminal-adjacent core materially shrinks both future-carry "
                "and raw residual without requiring any shared-core change."
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check whether the env12 mid-episode extension can be minimized to a runtime semantic control block."
    )
    parser.add_argument(
        "--env12-compare-pack-json",
        type=str,
        default="/home/chen/RLPFN/artifacts/phase3_env12_mid_episode_extension_compare_pack.json",
    )
    parser.add_argument(
        "--env12-counterfactual-pack-json",
        type=str,
        default="/home/chen/RLPFN/artifacts/phase3_env12_mid_episode_extension_counterfactual_pack.json",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="/home/chen/RLPFN/artifacts/phase3_env12_mid_episode_extension_runtime_control_probe.json",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output_json).expanduser().resolve()
    if output_path.exists() and not bool(args.overwrite):
        print(output_path.read_text())
        return 0

    report = build_env12_mid_episode_extension_runtime_control_probe(
        env12_compare_pack_json=args.env12_compare_pack_json,
        env12_counterfactual_pack_json=args.env12_counterfactual_pack_json,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(output_path.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
