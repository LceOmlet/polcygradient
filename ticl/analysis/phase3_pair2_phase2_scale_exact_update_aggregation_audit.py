import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_SNAPSHOT_JSON = (
    "/home/chen/RLPFN/artifacts/phase3_pair2_phase2_scale_target_present_recapture_snapshot.json"
)
DEFAULT_OUTPUT_JSON = (
    "/home/chen/RLPFN/artifacts/phase3_pair2_phase2_scale_exact_update_aggregation_audit.json"
)


def _load_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).expanduser().resolve().read_text())


def _as_float_array(payload: dict[str, Any], key: str) -> np.ndarray:
    return np.asarray(payload[key], dtype=np.float64).reshape(-1)


def _as_int_array(payload: dict[str, Any], key: str) -> np.ndarray:
    return np.asarray(payload[key], dtype=np.int64).reshape(-1)


def _share(numerator: float, denominator: float) -> float:
    denominator = float(denominator)
    if abs(denominator) <= 1e-12:
        return 0.0
    return float(float(numerator) / denominator)


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


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    active = np.asarray(mask, dtype=bool)
    if not bool(np.any(active)):
        return 0.0
    return float(np.mean(np.asarray(values, dtype=np.float64)[active]))


def _clip_active_fraction(post_advantages: np.ndarray, ratio: np.ndarray, clipped_objective: np.ndarray, mask: np.ndarray) -> float:
    active = np.asarray(mask, dtype=bool)
    if not bool(np.any(active)):
        return 0.0
    unclipped = np.asarray(post_advantages, dtype=np.float64) * np.asarray(ratio, dtype=np.float64)
    clipped = np.asarray(clipped_objective, dtype=np.float64)
    return float(np.mean(np.abs(clipped[active] - unclipped[active]) > 1e-8))


def _build_snapshot_arrays(snapshot: dict[str, Any]) -> dict[str, np.ndarray]:
    arrays = {
        "objective_mask": _as_int_array(snapshot, "objective_mask") > 0,
        "target_mask": _as_int_array(snapshot, "snapshot_target_match_mask") > 0,
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


def _build_masks(snapshot: dict[str, Any], arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    objective_mask = np.asarray(arrays["objective_mask"], dtype=bool)
    target_mask = np.asarray(arrays["target_mask"], dtype=bool)
    if not bool(np.any(target_mask)):
        raise ValueError("trusted-scale snapshot has no target-present rows")
    if not bool(np.all(~target_mask | objective_mask)):
        raise ValueError("snapshot_target_match_mask must be a subset of objective_mask")

    selector = dict(snapshot["snapshot_target_selector"])
    env_index = int(selector["target_env_index"])
    objective_episode_index = int(selector["target_objective_episode_index"])
    env_mask = objective_mask & (np.asarray(arrays["env_indices"], dtype=np.int64) == env_index)
    episode_mask = env_mask & (
        np.asarray(arrays["objective_episode_indices"], dtype=np.int64) == objective_episode_index
    )
    if not bool(np.all(~target_mask | episode_mask)):
        raise ValueError("target mask rows must share selector env and objective episode identity")

    return {
        "objective": objective_mask,
        "env": env_mask,
        "episode": episode_mask,
        "target": target_mask,
    }


def _build_target_scope(arrays: dict[str, np.ndarray], masks: dict[str, np.ndarray], snapshot: dict[str, Any]) -> dict[str, Any]:
    selector = dict(snapshot["snapshot_target_selector"])
    target = np.asarray(masks["target"], dtype=bool)
    return {
        "selector": {
            "outer_batch_idx": int(selector["target_outer_batch_idx"]),
            "env_index": int(selector["target_env_index"]),
            "objective_episode_index": int(selector["target_objective_episode_index"]),
            "objective_position_start": int(selector["target_objective_position_start"]),
            "objective_position_end": int(selector["target_objective_position_end"]),
        },
        "token_count": int(target.sum()),
        "objective_position_min": int(np.asarray(arrays["objective_episode_positions"], dtype=np.int64)[target].min()),
        "objective_position_max": int(np.asarray(arrays["objective_episode_positions"], dtype=np.int64)[target].max()),
        "objective_global_position_min": int(np.asarray(arrays["objective_global_positions"], dtype=np.int64)[target].min()),
        "objective_global_position_max": int(np.asarray(arrays["objective_global_positions"], dtype=np.int64)[target].max()),
        "step_index_min": int(np.asarray(arrays["step_indices"], dtype=np.int64)[target].min()),
        "step_index_max": int(np.asarray(arrays["step_indices"], dtype=np.int64)[target].max()),
    }


def _build_stage_report(arrays: dict[str, np.ndarray], masks: dict[str, np.ndarray]) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    for stage_name, key in (
        ("pre_normalization_actor_advantages", "pre"),
        ("post_normalization_advantages", "post"),
        ("policy_num_from_clipped_objective", "clipped"),
    ):
        values = np.asarray(arrays[key], dtype=np.float64)
        batch_metrics = _masked_stage_metrics(values, masks["objective"])
        env_metrics = _masked_stage_metrics(values, masks["env"])
        episode_metrics = _masked_stage_metrics(values, masks["episode"])
        target_metrics = _masked_stage_metrics(values, masks["target"])
        reports[stage_name] = {
            "whole_batch": batch_metrics,
            "env12": env_metrics,
            "episode": episode_metrics,
            "target_scope": target_metrics,
            "target_scope_share_of_batch_abs_mass": _share(target_metrics["abs_mass"], batch_metrics["abs_mass"]),
            "target_scope_share_of_batch_negative_mass": _share(target_metrics["negative_mass"], batch_metrics["negative_mass"]),
            "target_scope_share_of_env12_abs_mass": _share(target_metrics["abs_mass"], env_metrics["abs_mass"]),
            "target_scope_share_of_env12_negative_mass": _share(target_metrics["negative_mass"], env_metrics["negative_mass"]),
            "target_scope_share_of_episode_abs_mass": _share(target_metrics["abs_mass"], episode_metrics["abs_mass"]),
            "target_scope_share_of_episode_negative_mass": _share(target_metrics["negative_mass"], episode_metrics["negative_mass"]),
        }
    reports["ratio_and_clipping"] = {
        "whole_batch_clip_active_fraction": _clip_active_fraction(arrays["post"], arrays["ratio"], arrays["clipped"], masks["objective"]),
        "env12_clip_active_fraction": _clip_active_fraction(arrays["post"], arrays["ratio"], arrays["clipped"], masks["env"]),
        "episode_clip_active_fraction": _clip_active_fraction(arrays["post"], arrays["ratio"], arrays["clipped"], masks["episode"]),
        "target_scope_clip_active_fraction": _clip_active_fraction(arrays["post"], arrays["ratio"], arrays["clipped"], masks["target"]),
        "whole_batch_ratio_mean": _masked_mean(arrays["ratio"], masks["objective"]),
        "env12_ratio_mean": _masked_mean(arrays["ratio"], masks["env"]),
        "episode_ratio_mean": _masked_mean(arrays["ratio"], masks["episode"]),
        "target_scope_ratio_mean": _masked_mean(arrays["ratio"], masks["target"]),
    }
    return reports


def _build_stage_transition(report: dict[str, Any]) -> dict[str, Any]:
    pre = report["pre_normalization_actor_advantages"]["target_scope"]
    post = report["post_normalization_advantages"]["target_scope"]
    policy = report["policy_num_from_clipped_objective"]["target_scope"]
    return {
        "target_negative_mass": {
            "pre": float(pre["negative_mass"]),
            "post": float(post["negative_mass"]),
            "policy_num": float(policy["negative_mass"]),
            "normalization_delta": float(post["negative_mass"] - pre["negative_mass"]),
            "ratio_and_clipping_delta": float(policy["negative_mass"] - post["negative_mass"]),
            "pipeline_delta": float(policy["negative_mass"] - pre["negative_mass"]),
            "post_over_pre_retention": _share(post["negative_mass"], pre["negative_mass"]),
            "policy_over_post_retention": _share(policy["negative_mass"], post["negative_mass"]),
            "policy_over_pre_retention": _share(policy["negative_mass"], pre["negative_mass"]),
        },
        "target_share_of_batch_negative_mass": {
            "pre": float(report["pre_normalization_actor_advantages"]["target_scope_share_of_batch_negative_mass"]),
            "post": float(report["post_normalization_advantages"]["target_scope_share_of_batch_negative_mass"]),
            "policy_num": float(report["policy_num_from_clipped_objective"]["target_scope_share_of_batch_negative_mass"]),
            "normalization_delta": float(
                report["post_normalization_advantages"]["target_scope_share_of_batch_negative_mass"]
                - report["pre_normalization_actor_advantages"]["target_scope_share_of_batch_negative_mass"]
            ),
            "ratio_and_clipping_delta": float(
                report["policy_num_from_clipped_objective"]["target_scope_share_of_batch_negative_mass"]
                - report["post_normalization_advantages"]["target_scope_share_of_batch_negative_mass"]
            ),
        },
        "target_share_of_env12_negative_mass": {
            "pre": float(report["pre_normalization_actor_advantages"]["target_scope_share_of_env12_negative_mass"]),
            "post": float(report["post_normalization_advantages"]["target_scope_share_of_env12_negative_mass"]),
            "policy_num": float(report["policy_num_from_clipped_objective"]["target_scope_share_of_env12_negative_mass"]),
            "normalization_delta": float(
                report["post_normalization_advantages"]["target_scope_share_of_env12_negative_mass"]
                - report["pre_normalization_actor_advantages"]["target_scope_share_of_env12_negative_mass"]
            ),
            "ratio_and_clipping_delta": float(
                report["policy_num_from_clipped_objective"]["target_scope_share_of_env12_negative_mass"]
                - report["post_normalization_advantages"]["target_scope_share_of_env12_negative_mass"]
            ),
        },
    }


def build_pair2_phase2_scale_exact_update_aggregation_audit(*, snapshot_json: str) -> dict[str, Any]:
    snapshot = _load_json(snapshot_json)
    summary = dict(snapshot["summary"])
    arrays = _build_snapshot_arrays(snapshot)
    masks = _build_masks(snapshot, arrays)
    target_scope = _build_target_scope(arrays, masks, snapshot)
    stage_report = _build_stage_report(arrays, masks)
    stage_transition = _build_stage_transition(stage_report)

    return {
        "audit_entry": "phase3_pair2_phase2_scale_exact_update_aggregation_audit",
        "inputs": {
            "snapshot_json": str(Path(snapshot_json).expanduser().resolve()),
        },
        "strict_contract": {
            "outer_batch_idx": int(summary["outer_batch_idx"]),
            "epoch_idx": int(summary["epoch_idx"]),
            "batch_size_flat": int(summary["batch_size_flat"]),
            "objective_total": int(summary["objective_total"]),
            "normalize_advantage": bool(summary["normalize_advantage"]),
            "actor_objective_mode": str(summary["actor_objective_mode"]),
            "runtime_suite_name": str(summary["actor_objective_runtime_current_suite_name"]),
        },
        "target_scope": target_scope,
        "stage_report": stage_report,
        "stage_transition": stage_transition,
        "conclusions": {
            "trusted_target_scope_present_in_snapshot": True,
            "target_scope_negative_mass_is_amplified_by_normalization": bool(
                stage_transition["target_negative_mass"]["normalization_delta"] > 0.0
            ),
            "ratio_and_clipping_preserve_most_target_negative_mass": bool(
                stage_transition["target_negative_mass"]["policy_over_post_retention"] > 0.9
            ),
            "target_scope_is_a_small_fraction_of_whole_batch_negative_mass_after_clipping": bool(
                stage_transition["target_share_of_batch_negative_mass"]["policy_num"] < 0.1
            ),
        },
        "recommendation": {
            "next_focus": "trusted_scale_update_transfer_interpretation",
            "reason": (
                "The trusted snapshot now pins the exact train-side target scope. The next read-only step is to "
                "interpret whether this scope is too small to move the whole batch, or whether later aggregation "
                "stages still suppress it."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an exact trusted-scale update aggregation audit from the Phase-2-scale target-present snapshot."
    )
    parser.add_argument("--snapshot-json", type=str, default=DEFAULT_SNAPSHOT_JSON)
    parser.add_argument("--output-json", type=str, default=DEFAULT_OUTPUT_JSON)
    args = parser.parse_args()

    report = build_pair2_phase2_scale_exact_update_aggregation_audit(snapshot_json=args.snapshot_json)
    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
