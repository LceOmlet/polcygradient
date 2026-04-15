import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_PAIR2_TRAIN_UPDATE_AB_COMPARE_JSON = (
    "/home/chen/RLPFN/artifacts/phase3_pair2_train_update_ab_compare_pack.json"
)
DEFAULT_PAIR2_TRAIN_ROLLOUT_QUALITY_JSON = (
    "/home/chen/RLPFN/artifacts/phase3_train_rollout_quality_probe_pair2.json"
)
DEFAULT_ENV12_COUNTERFACTUAL_JSON = (
    "/home/chen/RLPFN/artifacts/phase3_env12_mid_episode_extension_counterfactual_pack.json"
)
DEFAULT_OUTPUT_JSON = (
    "/home/chen/RLPFN/artifacts/phase3_pair2_proxy_to_train_transfer_audit.json"
)


def _load_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).expanduser().resolve().read_text())


def _float(value: Any) -> float:
    return float(value)


def build_pair2_proxy_to_train_transfer_audit(
    *,
    pair2_train_update_ab_compare_json: str,
    pair2_train_rollout_quality_json: str,
    env12_counterfactual_json: str,
    target_env_index: int = 12,
) -> dict[str, Any]:
    compare_payload = _load_json(pair2_train_update_ab_compare_json)
    rollout_payload = _load_json(pair2_train_rollout_quality_json)
    counterfactual_payload = _load_json(env12_counterfactual_json)

    suite_context = dict(compare_payload["suite_context"])
    rollout_suite_summary = dict(rollout_payload["train_suite_summary"])
    if str(suite_context["train_suite_fingerprint"]) != str(rollout_suite_summary["fingerprint"]):
        raise ValueError(
            "Suite fingerprint mismatch between target-side A/B compare pack and rollout-quality artifact: "
            f"{suite_context['train_suite_fingerprint']!r} vs {rollout_suite_summary['fingerprint']!r}"
        )

    env_items = list(rollout_payload["env_items"])
    if not (0 <= int(target_env_index) < len(env_items)):
        raise ValueError(
            f"target_env_index out of range: {target_env_index} for env_items={len(env_items)}"
        )
    target_env = dict(env_items[int(target_env_index)])

    total_objective_tokens = int(sum(int(row["objective_token_count"]) for row in env_items))
    total_norm_abs_mass = _float(sum(_float(row["normalized_actor_adv_abs_mass"]) for row in env_items))
    total_norm_pos_mass = _float(sum(_float(row["normalized_actor_adv_positive_mass"]) for row in env_items))
    total_norm_neg_mass = _float(sum(_float(row["normalized_actor_adv_negative_mass"]) for row in env_items))

    observed_extension = dict(counterfactual_payload["comparison"]["observed_extension_block"])
    counterfactual = dict(counterfactual_payload["comparison"]["counterfactual"])
    block_row_count = int(observed_extension["row_count"])
    block_abs_future_carry_shrink_per_row = _float(
        counterfactual["mean_start_gae_future_carry_norm"]["abs_shrink"]
    )
    block_abs_future_carry_shrink_total = _float(block_row_count * block_abs_future_carry_shrink_per_row)

    env12_negative_mass = _float(target_env["normalized_actor_adv_negative_mass"])
    env12_positive_mass = _float(target_env["normalized_actor_adv_positive_mass"])
    env12_abs_mass = _float(target_env["normalized_actor_adv_abs_mass"])

    block_token_fraction_of_batch = _float(block_row_count / total_objective_tokens)
    block_abs_shrink_share_of_batch_abs_mass = _float(block_abs_future_carry_shrink_total / total_norm_abs_mass)
    block_abs_shrink_share_of_batch_neg_mass = _float(block_abs_future_carry_shrink_total / total_norm_neg_mass)
    block_abs_shrink_share_of_env12_neg_mass = _float(block_abs_future_carry_shrink_total / env12_negative_mass)

    compare_section = dict(compare_payload["comparison"])
    post_gap_delta = dict(compare_section["post_train_vs_zero_gap_delta"])
    pre_gap_delta = dict(compare_section["pre_train_vs_zero_gap_delta"])

    return {
        "audit_entry": "phase3_pair2_proxy_to_train_transfer_audit",
        "inputs": {
            "pair2_train_update_ab_compare_json": str(Path(pair2_train_update_ab_compare_json).expanduser().resolve()),
            "pair2_train_rollout_quality_json": str(Path(pair2_train_rollout_quality_json).expanduser().resolve()),
            "env12_counterfactual_json": str(Path(env12_counterfactual_json).expanduser().resolve()),
            "target_env_index": int(target_env_index),
        },
        "suite_context": {
            "train_suite_fingerprint": str(suite_context["train_suite_fingerprint"]),
            "heldout_suite_fingerprint": str(suite_context["heldout_suite_fingerprint"]),
            "n_samples": int(suite_context["n_samples"]),
            "eval_n_samples": int(suite_context["eval_n_samples"]),
            "single_eval_pos": int(suite_context["single_eval_pos"]),
            "strict_fixed_env_mode": bool(suite_context["strict_fixed_env_mode"]),
            "deterministic_actor_sampling": bool(suite_context["deterministic_actor_sampling"]),
            "deterministic_batch_plan": bool(suite_context["deterministic_batch_plan"]),
        },
        "batch_mass_context": {
            "env_count": int(len(env_items)),
            "total_objective_tokens": int(total_objective_tokens),
            "total_normalized_actor_adv_abs_mass": float(total_norm_abs_mass),
            "total_normalized_actor_adv_positive_mass": float(total_norm_pos_mass),
            "total_normalized_actor_adv_negative_mass": float(total_norm_neg_mass),
        },
        "target_env_context": {
            "env_index": int(target_env_index),
            "env_seed": int(target_env["env_seed"]),
            "rollout_seed": int(target_env["rollout_seed"]),
            "objective_token_count": int(target_env["objective_token_count"]),
            "objective_terminal_reset_count": int(target_env["objective_terminal_reset_count"]),
            "normalized_actor_adv_abs_mass": float(env12_abs_mass),
            "normalized_actor_adv_positive_mass": float(env12_positive_mass),
            "normalized_actor_adv_negative_mass": float(env12_negative_mass),
            "normalized_actor_adv_abs_mass_share_of_batch": float(env12_abs_mass / total_norm_abs_mass),
            "normalized_actor_adv_positive_mass_share_of_batch": float(env12_positive_mass / total_norm_pos_mass),
            "normalized_actor_adv_negative_mass_share_of_batch": float(env12_negative_mass / total_norm_neg_mass),
            "pre_vs_zero_full_gap": float(target_env["pre_vs_zero_full_gap"]),
            "pre_vs_zero_suffix_gap": float(target_env["pre_vs_zero_suffix_gap"]),
            "raw_value_return_corr": float(target_env["raw_value_return_corr"]),
        },
        "local_proxy_vs_batch_weighted_transfer": {
            "block_row_count": int(block_row_count),
            "block_token_fraction_of_batch": float(block_token_fraction_of_batch),
            "block_abs_future_carry_shrink_per_row": float(block_abs_future_carry_shrink_per_row),
            "block_abs_future_carry_shrink_total": float(block_abs_future_carry_shrink_total),
            "block_abs_shrink_share_of_batch_abs_mass": float(block_abs_shrink_share_of_batch_abs_mass),
            "block_abs_shrink_share_of_batch_negative_mass": float(block_abs_shrink_share_of_batch_neg_mass),
            "block_abs_shrink_share_of_env12_negative_mass": float(block_abs_shrink_share_of_env12_neg_mass),
        },
        "train_side_outcome": {
            "pre_train_vs_zero_gap_delta": {
                "full_return_gap": float(pre_gap_delta["full_return_gap"]),
                "suffix_return_gap": float(pre_gap_delta["suffix_return_gap"]),
            },
            "post_train_vs_zero_gap_delta": {
                "full_return_gap": float(post_gap_delta["full_return_gap"]),
                "suffix_return_gap": float(post_gap_delta["suffix_return_gap"]),
            },
            "train_full_return_delta_change": float(compare_section["train_full_return_delta_change"]),
            "train_suffix_return_delta_change": float(compare_section["train_suffix_return_delta_change"]),
            "train_history_identical": bool(compare_section["train_history_delta"]["histories_identical"]),
        },
        "conclusions": {
            "local_proxy_improvement_is_real": bool(
                counterfactual_payload["conclusions"]["env12_mid_episode_extension_counterfactual_shrinks_negative_carry"]
            ),
            "local_proxy_improvement_is_too_small_relative_to_batch_mass": bool(
                block_abs_shrink_share_of_batch_abs_mass < 1e-3
            ),
            "local_proxy_improvement_is_too_small_to_flip_env12_negative_mass": bool(
                block_abs_shrink_share_of_env12_neg_mass < 0.01
            ),
            "local_proxy_gain_did_not_move_suite_suffix_metric": bool(
                abs(_float(compare_section["train_suffix_return_delta_change"])) <= 1e-12
            ),
            "target_side_train_update_changed_but_not_in_the_desired_direction": bool(
                _float(compare_section["train_full_return_delta_change"]) < 0.0
            ),
            "proxy_to_train_transfer_failure_is_not_surprising_given_mass_dilution": bool(
                block_abs_shrink_share_of_batch_abs_mass < 1e-3
                and _float(compare_section["train_full_return_delta_change"]) < 0.0
                and abs(_float(compare_section["train_suffix_return_delta_change"])) <= 1e-12
            ),
        },
        "recommendation": {
            "candidate_small_control_target": "env12_mid_episode_extension_block",
            "reason": (
                "The env12 branch-specific gate materially improves a local carry proxy, but the affected four-token "
                "block accounts for only a tiny fraction of the pair2 batch-weighted actor-advantage mass. "
                "The next root-cause step should inspect update aggregation semantics "
                "(advantage normalization, clipping, and many-env mixing), not tune the block itself."
            ),
            "shared_guardrail_unchanged": True,
            "next_focus": "proxy_to_train_update_transfer_semantics",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quantify why the pair2 env12 local proxy gain failed to turn into train-side gain."
    )
    parser.add_argument(
        "--pair2-train-update-ab-compare-json",
        type=str,
        default=DEFAULT_PAIR2_TRAIN_UPDATE_AB_COMPARE_JSON,
    )
    parser.add_argument(
        "--pair2-train-rollout-quality-json",
        type=str,
        default=DEFAULT_PAIR2_TRAIN_ROLLOUT_QUALITY_JSON,
    )
    parser.add_argument(
        "--env12-counterfactual-json",
        type=str,
        default=DEFAULT_ENV12_COUNTERFACTUAL_JSON,
    )
    parser.add_argument("--target-env-index", type=int, default=12)
    parser.add_argument("--output-json", type=str, default=DEFAULT_OUTPUT_JSON)
    args = parser.parse_args()

    report = build_pair2_proxy_to_train_transfer_audit(
        pair2_train_update_ab_compare_json=args.pair2_train_update_ab_compare_json,
        pair2_train_rollout_quality_json=args.pair2_train_rollout_quality_json,
        env12_counterfactual_json=args.env12_counterfactual_json,
        target_env_index=int(args.target_env_index),
    )
    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
