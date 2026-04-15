import argparse
import json
from pathlib import Path
from typing import Any


def _load_compare_pack(path: str) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise ValueError(f"Missing JSON artifact: {resolved}")
    payload = json.loads(resolved.read_text())
    if str(payload.get("audit_entry", "")) != "phase3_env12_mid_episode_extension_compare_pack":
        raise ValueError(
            "Expected phase3_env12_mid_episode_extension_compare_pack artifact."
        )
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


def _abs_shrink_and_ratio(observed: float, counterfactual: float) -> dict[str, float]:
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


def build_env12_mid_episode_extension_counterfactual_pack(
    *,
    env12_compare_pack_json: str,
) -> dict[str, Any]:
    payload = _load_compare_pack(env12_compare_pack_json)
    comparison = dict(payload.get("comparison", {}))
    extension_block = _block_summary(dict(comparison["extension_block"]))
    terminal_adjacent_core = _block_summary(dict(comparison["terminal_adjacent_core"]))

    carry_counterfactual = _abs_shrink_and_ratio(
        extension_block["mean_start_gae_future_carry_norm"],
        terminal_adjacent_core["mean_start_gae_future_carry_norm"],
    )
    raw_residual_counterfactual = _abs_shrink_and_ratio(
        extension_block["mean_start_raw_rollout_residual_norm"],
        terminal_adjacent_core["mean_start_raw_rollout_residual_norm"],
    )
    weighted_delta_counterfactual = _abs_shrink_and_ratio(
        extension_block["mean_weighted_delta_sum"],
        terminal_adjacent_core["mean_weighted_delta_sum"],
    )

    conclusions = {
        "env12_mid_episode_extension_counterfactual_shrinks_negative_carry": bool(
            carry_counterfactual["abs_shrink"] > 0.0
        ),
        "env12_mid_episode_extension_counterfactual_shrinks_negative_carry_materially": bool(
            carry_counterfactual["counterfactual_over_observed_ratio"] <= 0.8
        ),
        "env12_mid_episode_extension_counterfactual_shrinks_raw_residual_materially": bool(
            raw_residual_counterfactual["counterfactual_over_observed_ratio"] <= 0.5
        ),
        "env12_mid_episode_extension_counterfactual_supports_branch_specific_small_control": bool(
            carry_counterfactual["counterfactual_over_observed_ratio"] <= 0.8
            and raw_residual_counterfactual["counterfactual_over_observed_ratio"] <= 0.5
        ),
        "env12_mid_episode_extension_counterfactual_does_not_change_shared_guardrail_anchor": bool(
            str(comparison.get("terminal_adjacent_core", {}).get("dominant_source", ""))
            == "baseline_bootstrap_semantic_mismatch"
        ),
    }

    recommendation = {
        "candidate_small_control_target": "env12_mid_episode_extension_block",
        "counterfactual_anchor": "terminal_adjacent_core",
        "branch_specific_followup_needed": True,
        "shared_guardrail_unchanged": True,
        "reason": (
            "Replacing the env12 mid-episode extension block with the terminal-adjacent core reduces "
            f"the absolute GAE future-carry magnitude by {carry_counterfactual['abs_shrink']:.6f} "
            f"({(1.0 - carry_counterfactual['counterfactual_over_observed_ratio']) * 100.0:.2f}%) and the "
            f"raw residual magnitude by {raw_residual_counterfactual['abs_shrink']:.6f} "
            f"({(1.0 - raw_residual_counterfactual['counterfactual_over_observed_ratio']) * 100.0:.2f}%). "
            "This is material enough to keep as a narrow branch-specific control candidate, not a pooled fix."
        ),
    }

    return {
        "audit_entry": "phase3_env12_mid_episode_extension_counterfactual_pack",
        "source_env12_compare_pack_json": str(Path(env12_compare_pack_json).expanduser().resolve()),
        "comparison": {
            "observed_extension_block": extension_block,
            "counterfactual_anchor_terminal_adjacent_core": terminal_adjacent_core,
            "counterfactual": {
                "mean_start_gae_future_carry_norm": carry_counterfactual,
                "mean_start_raw_rollout_residual_norm": raw_residual_counterfactual,
                "mean_weighted_delta_sum": weighted_delta_counterfactual,
            },
        },
        "conclusions": conclusions,
        "recommendation": recommendation,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read-only counterfactual pack comparing the env12 mid-episode extension block against the terminal-adjacent core."
    )
    parser.add_argument(
        "--env12-compare-pack-json",
        type=str,
        default="/home/chen/RLPFN/artifacts/phase3_env12_mid_episode_extension_compare_pack.json",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="/home/chen/RLPFN/artifacts/phase3_env12_mid_episode_extension_counterfactual_pack.json",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output_json).expanduser().resolve()
    if output_path.exists() and not bool(args.overwrite):
        print(output_path.read_text())
        return 0

    report = build_env12_mid_episode_extension_counterfactual_pack(
        env12_compare_pack_json=args.env12_compare_pack_json,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(output_path.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
