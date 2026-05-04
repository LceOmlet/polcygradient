import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_CURRENT_BASELINE_SNAPSHOT_JSON = (
    "/home/chen/RLPFN/artifacts/phase3_pair2_train_update_ab_baseline_outer_batch_snapshot.json"
)
DEFAULT_CURRENT_EXACT_AUDIT_JSON = (
    "/home/chen/RLPFN/artifacts/phase3_pair2_exact_update_aggregation_audit.json"
)
DEFAULT_CURRENT_EPOCH_DILUTION_JSON = (
    "/home/chen/RLPFN/artifacts/phase3_pair2_repaired_selector_epoch_dilution_probe.json"
)
DEFAULT_PRIOR_EXACT_AUDIT_JSON = (
    "/home/chen/RLPFN/artifacts/phase3_pair2_phase2_scale_exact_update_aggregation_audit.json"
)
DEFAULT_PRIOR_BATCH_TO_EPOCH_JSON = (
    "/home/chen/RLPFN/artifacts/phase3_pair2_phase2_scale_batch_to_epoch_aggregation_probe.json"
)
DEFAULT_OUTPUT_JSON = (
    "/home/chen/RLPFN/artifacts/phase3_pair2_branch_specific_extension_class_probe.json"
)


def _load_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def _float(value: Any) -> float:
    return float(value)


def _share(numerator: float, denominator: float) -> float:
    denominator = float(denominator)
    if abs(denominator) <= 1e-12:
        return 0.0
    return float(float(numerator) / denominator)


def _as_int_array(payload: dict[str, Any], key: str) -> np.ndarray:
    return np.asarray(payload[key], dtype=np.int64).reshape(-1)


def _as_float_array(payload: dict[str, Any], key: str) -> np.ndarray:
    return np.asarray(payload[key], dtype=np.float64).reshape(-1)


def _masked_stage_metrics(values: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    masked = np.asarray(values, dtype=np.float64)[np.asarray(mask, dtype=bool)]
    negative = masked[masked < 0.0]
    return {
        "token_count": int(masked.size),
        "signed_sum": float(masked.sum()) if masked.size > 0 else 0.0,
        "abs_mass": float(np.abs(masked).sum()) if masked.size > 0 else 0.0,
        "negative_mass": float(np.abs(negative).sum()) if negative.size > 0 else 0.0,
        "mean": float(masked.mean()) if masked.size > 0 else 0.0,
    }


def _build_current_episode_scope(
    *,
    current_baseline_snapshot: dict[str, Any],
    current_exact_audit: dict[str, Any],
    current_epoch_dilution: dict[str, Any],
) -> dict[str, Any]:
    exact_target_selector = dict(current_epoch_dilution["target_scope"]["selector"])
    exact_target_block = dict(current_exact_audit["target_block"])
    env_index = int(exact_target_block["env_index"])
    objective_episode_index = int(exact_target_selector["target_objective_episode_index"])
    block_position_start = int(exact_target_selector["target_objective_position_start"])
    block_position_end = int(exact_target_selector["target_objective_position_end"])

    objective_mask = _as_int_array(current_baseline_snapshot, "objective_mask") > 0
    env_indices = _as_int_array(current_baseline_snapshot, "flat_env_indices")
    objective_episode_indices = _as_int_array(current_baseline_snapshot, "flat_objective_episode_indices")
    objective_episode_positions = _as_int_array(current_baseline_snapshot, "flat_objective_episode_positions")

    if not bool(np.any(objective_mask)):
        raise ValueError("current baseline snapshot has no objective tokens")

    episode_mask = (
        objective_mask
        & (env_indices == env_index)
        & (objective_episode_indices == objective_episode_index)
    )
    if not bool(np.any(episode_mask)):
        raise ValueError("current repaired target episode is absent from the current baseline snapshot")

    block_mask = (
        episode_mask
        & (objective_episode_positions >= block_position_start)
        & (objective_episode_positions <= block_position_end)
    )
    if not bool(np.any(block_mask)):
        raise ValueError("current repaired target block is absent from the current baseline snapshot")
    if int(block_mask.sum()) != int(exact_target_block["block_token_count"]):
        raise ValueError("current repaired target block count does not match the exact audit block_token_count")

    pre = _as_float_array(current_baseline_snapshot, "pre_normalization_actor_advantages")
    post = _as_float_array(current_baseline_snapshot, "post_normalization_advantages")
    clipped = _as_float_array(current_baseline_snapshot, "clipped_objective")

    objective_pre = _masked_stage_metrics(pre, objective_mask)
    objective_post = _masked_stage_metrics(post, objective_mask)
    objective_policy = _masked_stage_metrics(clipped, objective_mask)
    block_policy = _masked_stage_metrics(clipped, block_mask)
    episode_pre = _masked_stage_metrics(pre, episode_mask)
    episode_post = _masked_stage_metrics(post, episode_mask)
    episode_policy = _masked_stage_metrics(clipped, episode_mask)

    return {
        "selector": {
            "env_index": int(env_index),
            "objective_episode_index": int(objective_episode_index),
            "objective_position_start": int(objective_episode_positions[episode_mask].min()),
            "objective_position_end": int(objective_episode_positions[episode_mask].max()),
        },
        "token_count": int(episode_mask.sum()),
        "stage_metrics": {
            "pre_normalization_actor_advantages": {
                **episode_pre,
                "share_of_batch_negative_mass": _share(
                    episode_pre["negative_mass"], objective_pre["negative_mass"]
                ),
            },
            "post_normalization_advantages": {
                **episode_post,
                "share_of_batch_negative_mass": _share(
                    episode_post["negative_mass"], objective_post["negative_mass"]
                ),
            },
            "policy_num_from_clipped_objective": {
                **episode_policy,
                "share_of_batch_negative_mass": _share(
                    episode_policy["negative_mass"], objective_policy["negative_mass"]
                ),
            },
        },
        "microblock_relation": {
            "block_token_count": int(block_mask.sum()),
            "block_share_of_episode_token_count": _share(int(block_mask.sum()), int(episode_mask.sum())),
            "block_policy_negative_mass": float(block_policy["negative_mass"]),
            "episode_policy_negative_mass": float(episode_policy["negative_mass"]),
            "block_share_of_episode_policy_negative_mass": _share(
                block_policy["negative_mass"], episode_policy["negative_mass"]
            ),
        },
    }


def build_pair2_branch_specific_extension_class_probe(
    *,
    current_baseline_snapshot_json: str,
    current_exact_audit_json: str,
    current_epoch_dilution_json: str,
    prior_exact_audit_json: str,
    prior_batch_to_epoch_json: str,
) -> dict[str, Any]:
    current_baseline_snapshot = _load_json(current_baseline_snapshot_json)
    current_exact_audit = _load_json(current_exact_audit_json)
    current_epoch_dilution = _load_json(current_epoch_dilution_json)
    prior_exact_audit = _load_json(prior_exact_audit_json)
    prior_batch_to_epoch = _load_json(prior_batch_to_epoch_json)

    current_epoch_layout = dict(current_epoch_dilution["epoch_layout"])
    current_selector = dict(current_epoch_dilution["target_scope"]["selector"])
    current_block_metrics = dict(
        current_exact_audit["baseline"]["policy_num_from_clipped_objective"]
    )
    current_full_episode = _build_current_episode_scope(
        current_baseline_snapshot=current_baseline_snapshot,
        current_exact_audit=current_exact_audit,
        current_epoch_dilution=current_epoch_dilution,
    )

    prior_target_scope = dict(prior_exact_audit["target_scope"])
    prior_stage_transition = dict(prior_exact_audit["stage_transition"])
    prior_epoch_context = dict(prior_batch_to_epoch["target_scope_epoch_context"])
    prior_epoch_layout = dict(prior_batch_to_epoch["epoch_layout"])

    if str(current_exact_audit["strict_contract"]["runtime_suite_name"]) != "pair2":
        raise ValueError("current exact audit is not running on pair2 runtime scope")
    if int(current_epoch_layout["target_outer_batch_idx"]) != int(current_exact_audit["strict_contract"]["outer_batch_idx"]):
        raise ValueError("current epoch dilution target_outer_batch_idx does not match current exact audit")
    if int(prior_epoch_context["target_outer_batch_idx"]) != int(prior_target_scope["selector"]["outer_batch_idx"]):
        raise ValueError("prior target_outer_batch_idx does not match prior exact target scope selector")
    if int(current_epoch_layout["total_outer_batches"]) != int(prior_epoch_layout["total_outer_batches"]):
        raise ValueError("current vs prior total_outer_batches mismatch")
    if int(current_epoch_layout["target_outer_batch_idx"]) != int(prior_epoch_context["target_outer_batch_idx"]):
        raise ValueError("current vs prior target_outer_batch_idx mismatch")

    current_total_outer_batches = int(current_epoch_layout["total_outer_batches"])
    current_microblock_policy_share = _float(current_block_metrics["block_share_of_batch_negative_mass"])
    current_microblock_epoch_uniform_heuristic = _float(
        current_microblock_policy_share / float(current_total_outer_batches)
    )
    current_episode_policy_share = _float(
        current_full_episode["stage_metrics"]["policy_num_from_clipped_objective"]["share_of_batch_negative_mass"]
    )
    current_episode_epoch_uniform_heuristic = _float(
        current_episode_policy_share / float(current_total_outer_batches)
    )
    prior_policy_share = _float(
        prior_stage_transition["target_share_of_batch_negative_mass"]["policy_num"]
    )
    prior_epoch_uniform_heuristic = _float(
        prior_epoch_context["uniform_step_weight_epoch_share_heuristic"]
    )

    current_target_outer_batch_fraction = _float(
        current_epoch_layout["target_outer_batch_idx"] / float(current_total_outer_batches)
    )
    prior_target_outer_batch_fraction = _float(
        prior_epoch_context["target_outer_batch_idx"] / float(prior_epoch_layout["total_outer_batches"])
    )

    return {
        "probe_entry": "phase3_pair2_branch_specific_extension_class_probe",
        "inputs": {
            "current_baseline_snapshot_json": str(Path(current_baseline_snapshot_json).expanduser().resolve()),
            "current_exact_audit_json": str(Path(current_exact_audit_json).expanduser().resolve()),
            "current_epoch_dilution_json": str(Path(current_epoch_dilution_json).expanduser().resolve()),
            "prior_exact_audit_json": str(Path(prior_exact_audit_json).expanduser().resolve()),
            "prior_batch_to_epoch_json": str(Path(prior_batch_to_epoch_json).expanduser().resolve()),
        },
        "current_trusted_contract": {
            "runtime_suite_name": str(current_exact_audit["strict_contract"]["runtime_suite_name"]),
            "target_outer_batch_idx": int(current_epoch_layout["target_outer_batch_idx"]),
            "total_outer_batches": int(current_total_outer_batches),
            "target_outer_batch_fraction": float(current_target_outer_batch_fraction),
            "objective_total_per_outer_batch": int(current_epoch_layout["objective_total_per_outer_batch"]),
            "remaining_outer_batches_after_target": int(current_epoch_layout["remaining_outer_batches_after_target"]),
        },
        "prior_effective_looking_contract": {
            "runtime_suite_name": str(prior_exact_audit["strict_contract"]["runtime_suite_name"]),
            "target_outer_batch_idx": int(prior_epoch_context["target_outer_batch_idx"]),
            "total_outer_batches": int(prior_epoch_layout["total_outer_batches"]),
            "target_outer_batch_fraction": float(prior_target_outer_batch_fraction),
            "objective_total_per_outer_batch": int(prior_epoch_layout["objective_total_per_outer_batch"]),
            "remaining_outer_batches_after_target": int(prior_epoch_context["target_batches_after_in_epoch"]),
        },
        "scope_classes": {
            "current_repaired_microblock": {
                "selector": {
                    "env_index": int(current_exact_audit["target_block"]["env_index"]),
                    "objective_episode_index": int(current_selector["target_objective_episode_index"]),
                    "objective_position_start": int(current_selector["target_objective_position_start"]),
                    "objective_position_end": int(current_selector["target_objective_position_end"]),
                },
                "token_count": int(current_exact_audit["target_block"]["block_token_count"]),
                "policy_negative_mass": float(
                    current_exact_audit["baseline"]["policy_num_from_clipped_objective"]["block"]["negative_mass"]
                ),
                "policy_share_of_batch_negative_mass": float(current_microblock_policy_share),
                "epoch_uniform_share_heuristic": float(current_microblock_epoch_uniform_heuristic),
            },
            "current_selector_matched_full_episode": {
                "selector": dict(current_full_episode["selector"]),
                "token_count": int(current_full_episode["token_count"]),
                "policy_negative_mass": float(
                    current_full_episode["stage_metrics"]["policy_num_from_clipped_objective"]["negative_mass"]
                ),
                "policy_share_of_batch_negative_mass": float(current_episode_policy_share),
                "epoch_uniform_share_heuristic": float(current_episode_epoch_uniform_heuristic),
                "microblock_relation": dict(current_full_episode["microblock_relation"]),
            },
            "prior_phase2_scale_target_scope": {
                "selector": dict(prior_target_scope["selector"]),
                "token_count": int(prior_target_scope["token_count"]),
                "policy_negative_mass": float(prior_stage_transition["target_negative_mass"]["policy_num"]),
                "policy_share_of_batch_negative_mass": float(prior_policy_share),
                "epoch_uniform_share_heuristic": float(prior_epoch_uniform_heuristic),
            },
        },
        "cross_scope_compare": {
            "old_vs_current_same_outer_batch_timing": bool(
                int(current_epoch_layout["target_outer_batch_idx"]) == int(prior_epoch_context["target_outer_batch_idx"])
                and int(current_total_outer_batches) == int(prior_epoch_layout["total_outer_batches"])
            ),
            "current_full_episode_vs_microblock_token_ratio": float(
                _share(
                    current_full_episode["token_count"],
                    current_exact_audit["target_block"]["block_token_count"],
                )
            ),
            "current_full_episode_vs_microblock_policy_share_ratio": float(
                _share(current_episode_policy_share, current_microblock_policy_share)
            ),
            "current_microblock_vs_prior_policy_share_ratio": float(
                _share(current_microblock_policy_share, prior_policy_share)
            ),
            "current_full_episode_vs_prior_policy_share_ratio": float(
                _share(current_episode_policy_share, prior_policy_share)
            ),
            "current_full_episode_vs_prior_epoch_uniform_ratio": float(
                _share(current_episode_epoch_uniform_heuristic, prior_epoch_uniform_heuristic)
            ),
            "current_microblock_vs_prior_epoch_uniform_ratio": float(
                _share(current_microblock_epoch_uniform_heuristic, prior_epoch_uniform_heuristic)
            ),
        },
        "conclusions": {
            "previous_effective_looking_scope_and_current_microblock_are_not_the_same_size_class": bool(
                current_full_episode["token_count"]
                > int(current_exact_audit["target_block"]["block_token_count"]) * 2
                and current_microblock_policy_share < prior_policy_share * 0.5
            ),
            "current_microblock_covers_only_small_fraction_of_current_target_episode_policy_mass": bool(
                current_full_episode["microblock_relation"]["block_share_of_episode_policy_negative_mass"] < 0.25
            ),
            "old_vs_current_difference_is_not_explained_by_later_epoch_timing": bool(
                int(current_epoch_layout["target_outer_batch_idx"]) == int(prior_epoch_context["target_outer_batch_idx"])
                and int(current_total_outer_batches) == int(prior_epoch_layout["total_outer_batches"])
            ),
            "selector_matched_full_episode_is_scale_matched_to_or_larger_than_the_prior_effective_scope": bool(
                current_episode_policy_share >= prior_policy_share * 0.9
            ),
            "scope_drift_is_a_better_explanation_for_the_surprise_than_new_shared_core_failure": bool(
                int(current_epoch_layout["target_outer_batch_idx"]) == int(prior_epoch_context["target_outer_batch_idx"])
                and int(current_total_outer_batches) == int(prior_epoch_layout["total_outer_batches"])
                and current_microblock_policy_share < prior_policy_share * 0.5
                and current_episode_policy_share >= prior_policy_share * 0.9
            ),
            "smallest_current_branch_specific_extension_class_that_can_clear_the_single_step_floor_is_episode_complete": bool(
                current_microblock_policy_share < prior_policy_share
                and current_episode_policy_share >= prior_policy_share * 0.9
            ),
        },
        "recommendation": {
            "next_focus": "episode_complete_branch_specific_extension_class",
            "reason": (
                "The old effective-looking signal and the current repaired microblock share the same late outer-batch "
                "timing, so overwrite timing alone does not explain the surprise. The main difference is scope class: "
                "the repaired 4-token microblock is far smaller than the old phase2-scale target scope, while the "
                "current selector-matched full objective episode is already size-matched to that old scope. The next "
                "small-control decomposition should therefore keep the shared core fixed and treat the episode-complete "
                "selector-matched extension as the first admissible branch-specific class to evaluate."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare current repaired microblock scope against the prior effective-looking branch-specific scope class."
    )
    parser.add_argument("--current-baseline-snapshot-json", type=str, default=DEFAULT_CURRENT_BASELINE_SNAPSHOT_JSON)
    parser.add_argument("--current-exact-audit-json", type=str, default=DEFAULT_CURRENT_EXACT_AUDIT_JSON)
    parser.add_argument("--current-epoch-dilution-json", type=str, default=DEFAULT_CURRENT_EPOCH_DILUTION_JSON)
    parser.add_argument("--prior-exact-audit-json", type=str, default=DEFAULT_PRIOR_EXACT_AUDIT_JSON)
    parser.add_argument("--prior-batch-to-epoch-json", type=str, default=DEFAULT_PRIOR_BATCH_TO_EPOCH_JSON)
    parser.add_argument("--output-json", type=str, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output_json).expanduser().resolve()
    if output_path.exists() and not bool(args.overwrite):
        print(output_path.read_text(encoding="utf-8"))
        return

    report = build_pair2_branch_specific_extension_class_probe(
        current_baseline_snapshot_json=args.current_baseline_snapshot_json,
        current_exact_audit_json=args.current_exact_audit_json,
        current_epoch_dilution_json=args.current_epoch_dilution_json,
        prior_exact_audit_json=args.prior_exact_audit_json,
        prior_batch_to_epoch_json=args.prior_batch_to_epoch_json,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
