import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_BASELINE_SNAPSHOT_JSON = (
    "/home/chen/RLPFN/artifacts/phase3_pair2_train_update_ab_baseline_outer_batch_snapshot.json"
)
DEFAULT_OVERRIDE_SNAPSHOT_JSON = (
    "/home/chen/RLPFN/artifacts/phase3_pair2_train_update_ab_override_outer_batch_snapshot.json"
)
DEFAULT_OUTPUT_JSON = (
    "/home/chen/RLPFN/artifacts/phase3_pair2_exact_update_aggregation_audit.json"
)

TARGET_ENV_INDEX = 12
TARGET_OBJECTIVE_GLOBAL_POSITION_START = 18
TARGET_OBJECTIVE_GLOBAL_POSITION_END = 21


def _load_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).expanduser().resolve().read_text())


def _as_float_array(payload: dict[str, Any], key: str) -> np.ndarray:
    return np.asarray(payload[key], dtype=np.float64).reshape(-1)


def _as_int_array(payload: dict[str, Any], key: str) -> np.ndarray:
    return np.asarray(payload[key], dtype=np.int64).reshape(-1)


def _masked_stage_metrics(values: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    masked = np.asarray(values, dtype=np.float64)[np.asarray(mask, dtype=bool)]
    positive = masked[masked > 0.0]
    negative = masked[masked < 0.0]
    return {
        "token_count": int(masked.size),
        "signed_sum": float(masked.sum()) if masked.size > 0 else 0.0,
        "abs_mass": float(np.abs(masked).sum()) if masked.size > 0 else 0.0,
        "positive_mass": float(positive.sum()) if positive.size > 0 else 0.0,
        "negative_mass": float(np.abs(negative).sum()) if negative.size > 0 else 0.0,
        "mean": float(masked.mean()) if masked.size > 0 else 0.0,
    }


def _share(numerator: float, denominator: float) -> float:
    denominator = float(denominator)
    if abs(denominator) <= 1e-12:
        return 0.0
    return float(float(numerator) / denominator)


def _clip_active_fraction(post_advantages: np.ndarray, ratio: np.ndarray, clipped_objective: np.ndarray, mask: np.ndarray) -> float:
    active = np.asarray(mask, dtype=bool)
    if not bool(np.any(active)):
        return 0.0
    unclipped = np.asarray(post_advantages, dtype=np.float64) * np.asarray(ratio, dtype=np.float64)
    clipped = np.asarray(clipped_objective, dtype=np.float64)
    return float(np.mean(np.abs(clipped[active] - unclipped[active]) > 1e-8))


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    active = np.asarray(mask, dtype=bool)
    if not bool(np.any(active)):
        return 0.0
    return float(np.mean(np.asarray(values, dtype=np.float64)[active]))


def _build_snapshot_arrays(snapshot: dict[str, Any]) -> dict[str, np.ndarray]:
    objective_mask = _as_int_array(snapshot, "objective_mask") > 0
    arrays = {
        "objective_mask": objective_mask,
        "env_indices": _as_int_array(snapshot, "flat_env_indices"),
        "step_indices": _as_int_array(snapshot, "flat_step_indices"),
        "objective_episode_indices": _as_int_array(snapshot, "flat_objective_episode_indices"),
        "objective_episode_positions": _as_int_array(snapshot, "flat_objective_episode_positions"),
        "objective_global_positions": _as_int_array(snapshot, "flat_objective_global_positions"),
        "pre": _as_float_array(snapshot, "pre_normalization_actor_advantages"),
        "post": _as_float_array(snapshot, "post_normalization_advantages"),
        "ratio": _as_float_array(snapshot, "ratio"),
        "clipped": _as_float_array(snapshot, "clipped_objective"),
    }
    expected_len = int(arrays["pre"].shape[0])
    for key, value in arrays.items():
        if int(value.shape[0]) != expected_len:
            raise ValueError(f"snapshot field length mismatch for {key}: expected {expected_len}, got {value.shape[0]}")
    return arrays


def _build_scope_masks(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    objective_mask = np.asarray(arrays["objective_mask"], dtype=bool)
    env12_mask = objective_mask & (np.asarray(arrays["env_indices"], dtype=np.int64) == int(TARGET_ENV_INDEX))
    block_mask = (
        env12_mask
        & (np.asarray(arrays["objective_global_positions"], dtype=np.int64) >= int(TARGET_OBJECTIVE_GLOBAL_POSITION_START))
        & (np.asarray(arrays["objective_global_positions"], dtype=np.int64) <= int(TARGET_OBJECTIVE_GLOBAL_POSITION_END))
    )
    return {
        "objective": objective_mask,
        "env12": env12_mask,
        "block": block_mask,
    }


def _build_available_snapshot_scope(arrays: dict[str, np.ndarray], objective_mask: np.ndarray) -> list[dict[str, int]]:
    rows: list[dict[str, int]] = []
    active = np.asarray(objective_mask, dtype=bool)
    env_indices = np.asarray(arrays["env_indices"], dtype=np.int64)
    objective_episode_indices = np.asarray(arrays["objective_episode_indices"], dtype=np.int64)
    objective_episode_positions = np.asarray(arrays["objective_episode_positions"], dtype=np.int64)
    for env_idx in sorted(set(int(v) for v in env_indices[active])):
        env_mask = active & (env_indices == int(env_idx))
        for objective_episode_index in sorted(set(int(v) for v in objective_episode_indices[env_mask])):
            episode_mask = env_mask & (objective_episode_indices == int(objective_episode_index))
            positions = objective_episode_positions[episode_mask]
            rows.append(
                {
                    "env_index": int(env_idx),
                    "objective_episode_index": int(objective_episode_index),
                    "token_count": int(positions.shape[0]),
                    "objective_position_min": int(positions.min()),
                    "objective_position_max": int(positions.max()),
                }
            )
    return rows


def _build_stage_report(arrays: dict[str, np.ndarray], masks: dict[str, np.ndarray]) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    for stage_name, key in (
        ("pre_normalization_actor_advantages", "pre"),
        ("post_normalization_advantages", "post"),
        ("policy_num_from_clipped_objective", "clipped"),
    ):
        values = np.asarray(arrays[key], dtype=np.float64)
        batch_metrics = _masked_stage_metrics(values, masks["objective"])
        env12_metrics = _masked_stage_metrics(values, masks["env12"])
        block_metrics = _masked_stage_metrics(values, masks["block"])
        reports[stage_name] = {
            "whole_batch": batch_metrics,
            "env12": env12_metrics,
            "block": block_metrics,
            "block_share_of_batch_abs_mass": _share(block_metrics["abs_mass"], batch_metrics["abs_mass"]),
            "block_share_of_batch_negative_mass": _share(block_metrics["negative_mass"], batch_metrics["negative_mass"]),
            "block_share_of_env12_abs_mass": _share(block_metrics["abs_mass"], env12_metrics["abs_mass"]),
            "block_share_of_env12_negative_mass": _share(block_metrics["negative_mass"], env12_metrics["negative_mass"]),
        }
    reports["ratio_and_clipping"] = {
        "whole_batch_clip_active_fraction": _clip_active_fraction(
            arrays["post"],
            arrays["ratio"],
            arrays["clipped"],
            masks["objective"],
        ),
        "env12_clip_active_fraction": _clip_active_fraction(
            arrays["post"],
            arrays["ratio"],
            arrays["clipped"],
            masks["env12"],
        ),
        "block_clip_active_fraction": _clip_active_fraction(
            arrays["post"],
            arrays["ratio"],
            arrays["clipped"],
            masks["block"],
        ),
        "whole_batch_ratio_mean": _masked_mean(arrays["ratio"], masks["objective"]),
        "env12_ratio_mean": _masked_mean(arrays["ratio"], masks["env12"]),
        "block_ratio_mean": _masked_mean(arrays["ratio"], masks["block"]),
    }
    return reports


def _build_delta_report(baseline: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for stage_name in (
        "pre_normalization_actor_advantages",
        "post_normalization_advantages",
        "policy_num_from_clipped_objective",
    ):
        base_stage = baseline[stage_name]
        over_stage = override[stage_name]
        block_pre = float(base_stage["block"]["negative_mass"])
        block_over = float(over_stage["block"]["negative_mass"])
        block_shrink = float(block_pre - block_over)
        report[stage_name] = {
            "block_negative_mass_shrink": block_shrink,
            "block_abs_mass_delta": float(over_stage["block"]["abs_mass"] - base_stage["block"]["abs_mass"]),
            "block_signed_sum_delta": float(over_stage["block"]["signed_sum"] - base_stage["block"]["signed_sum"]),
            "block_share_of_batch_abs_mass_delta": float(
                over_stage["block_share_of_batch_abs_mass"] - base_stage["block_share_of_batch_abs_mass"]
            ),
            "block_share_of_batch_negative_mass_delta": float(
                over_stage["block_share_of_batch_negative_mass"] - base_stage["block_share_of_batch_negative_mass"]
            ),
            "block_share_of_env12_negative_mass_delta": float(
                over_stage["block_share_of_env12_negative_mass"] - base_stage["block_share_of_env12_negative_mass"]
            ),
        }

    pre_shrink = float(report["pre_normalization_actor_advantages"]["block_negative_mass_shrink"])
    post_shrink = float(report["post_normalization_advantages"]["block_negative_mass_shrink"])
    policy_shrink = float(report["policy_num_from_clipped_objective"]["block_negative_mass_shrink"])
    report["shrink_retention"] = {
        "post_over_pre_negative_mass_shrink_retention": _share(post_shrink, pre_shrink),
        "policy_over_pre_negative_mass_shrink_retention": _share(policy_shrink, pre_shrink),
        "policy_over_post_negative_mass_shrink_retention": _share(policy_shrink, post_shrink),
    }
    report["ratio_and_clipping"] = {
        "whole_batch_clip_active_fraction_delta": float(
            override["ratio_and_clipping"]["whole_batch_clip_active_fraction"]
            - baseline["ratio_and_clipping"]["whole_batch_clip_active_fraction"]
        ),
        "env12_clip_active_fraction_delta": float(
            override["ratio_and_clipping"]["env12_clip_active_fraction"]
            - baseline["ratio_and_clipping"]["env12_clip_active_fraction"]
        ),
        "block_clip_active_fraction_delta": float(
            override["ratio_and_clipping"]["block_clip_active_fraction"]
            - baseline["ratio_and_clipping"]["block_clip_active_fraction"]
        ),
        "block_ratio_mean_delta": float(
            override["ratio_and_clipping"]["block_ratio_mean"] - baseline["ratio_and_clipping"]["block_ratio_mean"]
        ),
    }
    return report


def build_pair2_exact_update_aggregation_audit(
    *,
    baseline_snapshot_json: str,
    override_snapshot_json: str,
) -> dict[str, Any]:
    baseline_snapshot = _load_json(baseline_snapshot_json)
    override_snapshot = _load_json(override_snapshot_json)

    baseline_summary = dict(baseline_snapshot["summary"])
    override_summary = dict(override_snapshot["summary"])
    for key in ("outer_batch_idx", "epoch_idx", "batch_size_flat", "objective_total", "normalize_advantage"):
        if baseline_summary[key] != override_summary[key]:
            raise ValueError(f"snapshot contract mismatch on {key}: {baseline_summary[key]!r} vs {override_summary[key]!r}")
    baseline_runtime_suite_name = str(
        baseline_summary.get("actor_objective_runtime_current_suite_name", "")
    )
    override_runtime_suite_name = str(
        override_summary.get("actor_objective_runtime_current_suite_name", "")
    )

    baseline_arrays = _build_snapshot_arrays(baseline_snapshot)
    override_arrays = _build_snapshot_arrays(override_snapshot)
    baseline_masks = _build_scope_masks(baseline_arrays)
    override_masks = _build_scope_masks(override_arrays)

    baseline_block_count = int(baseline_masks["block"].sum())
    override_block_count = int(override_masks["block"].sum())
    target_block_present = bool(baseline_block_count > 0 or override_block_count > 0)

    if target_block_present and (baseline_block_count != 4 or override_block_count != 4):
        raise ValueError(
            "expected the env12 branch-specific block to contain exactly four objective tokens, got "
            f"baseline={baseline_block_count}, override={override_block_count}"
        )
    if target_block_present and not np.array_equal(baseline_masks["block"], override_masks["block"]):
        raise ValueError("baseline/override block masks must match under the strict pair2 contract")
    if not np.array_equal(baseline_masks["env12"], override_masks["env12"]):
        raise ValueError("baseline/override env12 masks must match under the strict pair2 contract")
    if not np.array_equal(baseline_masks["objective"], override_masks["objective"]):
        raise ValueError("baseline/override objective masks must match under the strict pair2 contract")

    baseline_stage_report = _build_stage_report(baseline_arrays, baseline_masks)
    override_stage_report = _build_stage_report(override_arrays, override_masks)
    delta_report = _build_delta_report(baseline_stage_report, override_stage_report)

    return {
        "audit_entry": "phase3_pair2_exact_update_aggregation_audit",
        "inputs": {
            "baseline_snapshot_json": str(Path(baseline_snapshot_json).expanduser().resolve()),
            "override_snapshot_json": str(Path(override_snapshot_json).expanduser().resolve()),
        },
        "strict_contract": {
            "outer_batch_idx": int(baseline_summary["outer_batch_idx"]),
            "epoch_idx": int(baseline_summary["epoch_idx"]),
            "batch_size_flat": int(baseline_summary["batch_size_flat"]),
            "objective_total": int(baseline_summary["objective_total"]),
            "normalize_advantage": bool(baseline_summary["normalize_advantage"]),
            "baseline_actor_objective_mode": str(baseline_summary["actor_objective_mode"]),
            "override_actor_objective_mode": str(override_summary["actor_objective_mode"]),
            "runtime_suite_name": override_runtime_suite_name,
            "baseline_runtime_suite_name": baseline_runtime_suite_name,
            "override_runtime_suite_name": override_runtime_suite_name,
        },
        "target_block": {
            "env_index": int(TARGET_ENV_INDEX),
            "objective_global_position_start": int(TARGET_OBJECTIVE_GLOBAL_POSITION_START),
            "objective_global_position_end": int(TARGET_OBJECTIVE_GLOBAL_POSITION_END),
            "block_token_count": int(baseline_block_count),
            "env12_objective_token_count": int(baseline_masks["env12"].sum()),
            "batch_objective_token_count": int(baseline_masks["objective"].sum()),
            "target_block_present_in_snapshot": bool(target_block_present),
        },
        "available_snapshot_scope": {
            "objective_segments": _build_available_snapshot_scope(baseline_arrays, baseline_masks["objective"]),
        },
        "baseline": baseline_stage_report,
        "override": override_stage_report,
        "delta": delta_report,
        "conclusions": {
            "requested_target_block_present_in_captured_outer_batch": bool(target_block_present),
            "block_exact_pre_norm_negative_mass_shrinks": bool(
                delta_report["pre_normalization_actor_advantages"]["block_negative_mass_shrink"] > 0.0
            ),
            "block_exact_post_norm_negative_mass_shrink_retained": bool(
                delta_report["post_normalization_advantages"]["block_negative_mass_shrink"] > 0.0
            ),
            "block_exact_policy_num_negative_mass_shrink_retained": bool(
                delta_report["policy_num_from_clipped_objective"]["block_negative_mass_shrink"] > 0.0
            ),
            "normalization_is_a_major_dilution_stage": bool(
                delta_report["shrink_retention"]["post_over_pre_negative_mass_shrink_retention"] < 0.5
            ),
            "ratio_and_clipping_further_reduce_transfer": bool(
                delta_report["shrink_retention"]["policy_over_post_negative_mass_shrink_retention"] < 0.9
            ),
            "current_single_snapshot_is_insufficient_for_env12_block_audit": bool(not target_block_present),
        },
        "recommendation": {
            "next_focus": "exact_update_aggregation_audit",
            "reason": (
                "If the captured strict outer batch contains the env12 block, the exact stage-wise contribution can be "
                "measured directly from the snapshot. If the target block is absent, the hook must be refined to "
                "capture a selected outer batch instead of the first one that reaches train()."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an exact train-side update aggregation audit from strict pair2 outer-batch snapshots."
    )
    parser.add_argument("--baseline-snapshot-json", type=str, default=DEFAULT_BASELINE_SNAPSHOT_JSON)
    parser.add_argument("--override-snapshot-json", type=str, default=DEFAULT_OVERRIDE_SNAPSHOT_JSON)
    parser.add_argument("--output-json", type=str, default=DEFAULT_OUTPUT_JSON)
    args = parser.parse_args()

    report = build_pair2_exact_update_aggregation_audit(
        baseline_snapshot_json=args.baseline_snapshot_json,
        override_snapshot_json=args.override_snapshot_json,
    )
    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
