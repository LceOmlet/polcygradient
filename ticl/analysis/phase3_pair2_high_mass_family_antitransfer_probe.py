import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_HIGH_MASS_HELDOUT_TRANSFER_COMPARE_JSON = (
    "/home/chen/RLPFN/artifacts/phase3_pair2_high_mass_episode_family_heldout_transfer_compare_pack.json"
)
DEFAULT_BASELINE_RESUME_EVAL_JSON = (
    "/home/chen/RLPFN/artifacts/phase3_pair2_high_mass_episode_family_heldout_transfer_baseline_resume_eval.json"
)
DEFAULT_OVERRIDE_RESUME_EVAL_JSON = (
    "/home/chen/RLPFN/artifacts/phase3_pair2_high_mass_episode_family_heldout_transfer_override_resume_eval.json"
)
DEFAULT_POSITIVE_REGIME_ANCHOR_STRUCTURE_JSON = (
    "/home/chen/RLPFN/artifacts/phase3_positive_regime_anchor_structure_probe.json"
)
DEFAULT_DUAL_CHANNEL_COVERAGE_JSON = (
    "/home/chen/RLPFN/artifacts/phase3_dual_channel_anchor_coverage_probe.json"
)
DEFAULT_BASELINE_SNAPSHOT_JSON = (
    "/home/chen/RLPFN/artifacts/phase3_pair2_episode_complete_train_update_ab_baseline_outer_batch_snapshot.json"
)
DEFAULT_FAMILY_CANDIDATE_JSON = (
    "/home/chen/RLPFN/artifacts/phase3_pair2_branch_specific_family_candidate_probe.json"
)
DEFAULT_OUTPUT_JSON = (
    "/home/chen/RLPFN/artifacts/phase3_pair2_high_mass_family_antitransfer_probe.json"
)
TAIL_LEN = 8


def _load_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(float(v) for v in values) / len(values))


def _pair2_env_rows(summary_payload: dict[str, Any], key: str) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in list(summary_payload.get(key, {}).get("env_items", []))
        if str(row.get("suite_name", "")) == "pair2"
    ]


def _per_env_transfer_rows(
    *,
    baseline_resume_eval: dict[str, Any],
    override_resume_eval: dict[str, Any],
) -> list[dict[str, Any]]:
    baseline_full = list(baseline_resume_eval["post"]["heldout"]["full_return_per_env"])
    override_full = list(override_resume_eval["post"]["heldout"]["full_return_per_env"])
    baseline_suffix = list(baseline_resume_eval["post"]["heldout"]["suffix_return_per_env"])
    override_suffix = list(override_resume_eval["post"]["heldout"]["suffix_return_per_env"])
    if not (
        len(baseline_full)
        == len(override_full)
        == len(baseline_suffix)
        == len(override_suffix)
    ):
        raise RuntimeError("Heldout per-env metric lengths do not match between baseline and override.")
    rows = []
    for env_index, (base_full, over_full, base_suffix, over_suffix) in enumerate(
        zip(baseline_full, override_full, baseline_suffix, override_suffix)
    ):
        rows.append(
            {
                "env_index": int(env_index),
                "full_return_delta": float(float(over_full) - float(base_full)),
                "suffix_return_delta": float(float(over_suffix) - float(base_suffix)),
            }
        )
    return rows


def _group_transfer_summary(
    *,
    rows: list[dict[str, Any]],
    env_indices: set[int],
) -> dict[str, Any]:
    kept = [dict(row) for row in rows if int(row["env_index"]) in {int(v) for v in env_indices}]
    suffix_values = [float(row["suffix_return_delta"]) for row in kept]
    full_values = [float(row["full_return_delta"]) for row in kept]
    return {
        "count": int(len(kept)),
        "env_indices": sorted(int(v) for v in env_indices),
        "mean_suffix_return_delta": _mean(suffix_values),
        "mean_full_return_delta": _mean(full_values),
        "positive_suffix_count": int(sum(float(v) > 0.0 for v in suffix_values)),
        "negative_suffix_count": int(sum(float(v) < 0.0 for v in suffix_values)),
        "env_rows": sorted(kept, key=lambda row: float(row["suffix_return_delta"])),
    }


def _env12_episode_metrics(snapshot_payload: dict[str, Any]) -> dict[str, Any]:
    objective_mask = np.asarray(snapshot_payload["objective_mask"], dtype=np.int64).reshape(-1) > 0
    env_indices = np.asarray(snapshot_payload["flat_env_indices"], dtype=np.int64).reshape(-1)
    episode_indices = np.asarray(snapshot_payload["flat_objective_episode_indices"], dtype=np.int64).reshape(-1)
    positions = np.asarray(snapshot_payload["flat_objective_episode_positions"], dtype=np.int64).reshape(-1)
    clip = np.asarray(snapshot_payload["clipped_objective"], dtype=np.float64).reshape(-1)

    env12_mask = objective_mask & (env_indices == 12)
    if not bool(np.any(env12_mask)):
        raise RuntimeError("Expected env12 objective tokens in the pair2 baseline snapshot.")

    episode_spans = []
    for episode_index in sorted({int(v) for v in episode_indices[env12_mask].tolist()}):
        mask = env12_mask & (episode_indices == int(episode_index))
        episode_positions = positions[mask]
        token_count = int(mask.sum())
        negative_mass = float(np.abs(clip[mask & (clip < 0.0)]).sum()) if token_count > 0 else 0.0
        episode_spans.append(
            {
                "episode_index": int(episode_index),
                "token_count": int(token_count),
                "position_min": int(episode_positions.min()),
                "position_max": int(episode_positions.max()),
                "negative_mass": float(negative_mass),
                "non_tail_token_count": int(max(0, token_count - TAIL_LEN)),
            }
        )
    first_episode = dict(episode_spans[0])
    return {
        "episode_spans": episode_spans,
        "first_objective_episode_index": int(first_episode["episode_index"]),
        "first_objective_episode_token_count": int(first_episode["token_count"]),
        "first_objective_episode_non_tail_token_count": int(first_episode["non_tail_token_count"]),
        "first_objective_episode_all_tail_under_tail_len_8": bool(
            int(first_episode["non_tail_token_count"]) == 0
        ),
    }


def build_pair2_high_mass_family_antitransfer_probe(
    *,
    high_mass_heldout_transfer_compare_json: str,
    baseline_resume_eval_json: str,
    override_resume_eval_json: str,
    positive_regime_anchor_structure_json: str,
    dual_channel_coverage_json: str,
    baseline_snapshot_json: str,
    family_candidate_json: str,
) -> dict[str, Any]:
    transfer_compare = _load_json(high_mass_heldout_transfer_compare_json)
    baseline_resume_eval = _load_json(baseline_resume_eval_json)
    override_resume_eval = _load_json(override_resume_eval_json)
    positive_regime = _load_json(positive_regime_anchor_structure_json)
    dual_channel = _load_json(dual_channel_coverage_json)
    baseline_snapshot = _load_json(baseline_snapshot_json)
    family_candidate = _load_json(family_candidate_json)

    if (
        str(transfer_compare.get("audit_entry", ""))
        != "phase3_pair2_high_mass_episode_family_heldout_transfer_compare_pack"
    ):
        raise RuntimeError("Expected canonical pair2 high-mass-family heldout-transfer compare artifact.")
    if not bool(transfer_compare["conclusions"]["train_side_reproduction_matches_locked_compare"]):
        raise RuntimeError("Heldout-transfer compare did not preserve the locked train-side reproduction contract.")

    pair2_missed_rows = _pair2_env_rows(positive_regime, "missed_summary")
    pair2_captured_rows = _pair2_env_rows(positive_regime, "captured_summary")
    pair2_recovered_rows = _pair2_env_rows(dual_channel, "recovered_anchor_summary")
    pair2_residual_rows = _pair2_env_rows(dual_channel, "residual_missed_summary")

    per_env_rows = _per_env_transfer_rows(
        baseline_resume_eval=baseline_resume_eval,
        override_resume_eval=override_resume_eval,
    )
    recovered_envs = {int(row["env_index"]) for row in pair2_recovered_rows}
    residual_envs = {int(row["env_index"]) for row in pair2_residual_rows}
    missed_envs = {int(row["env_index"]) for row in pair2_missed_rows}
    all_envs = {int(row["env_index"]) for row in per_env_rows}
    other_envs = all_envs - recovered_envs - residual_envs

    env12_metrics = _env12_episode_metrics(baseline_snapshot)
    first_episode_index = int(env12_metrics["first_objective_episode_index"])

    candidate_by_name = {
        str(row["name"]): dict(row) for row in list(family_candidate.get("candidate_families", []))
    }
    first_episode_candidate = None
    for row in list(family_candidate.get("candidate_families", [])):
        if list(row.get("episode_indices", [])) == [first_episode_index]:
            first_episode_candidate = dict(row)
            break
    if first_episode_candidate is None:
        first_episode_candidate = {
            "name": f"first_objective_episode_{first_episode_index}_complete",
            "episode_indices": [int(first_episode_index)],
            "token_count": int(env12_metrics["first_objective_episode_token_count"]),
            "policy_share_of_batch_negative_mass": float(
                next(
                    span["negative_mass"]
                    for span in list(env12_metrics["episode_spans"])
                    if int(span["episode_index"]) == int(first_episode_index)
                )
                / float(candidate_by_name["all_env12_objective_tokens"]["policy_negative_mass"])
                * float(candidate_by_name["all_env12_objective_tokens"]["policy_share_of_batch_negative_mass"])
                if float(candidate_by_name["all_env12_objective_tokens"]["policy_negative_mass"]) > 0.0
                else 0.0
            ),
        }

    high_mass_candidate = dict(candidate_by_name["high_mass_complete_episodes_4_7_11"])
    locked_train_side_source = dict(transfer_compare.get("locked_train_side_source", {}))
    locked_episode_indices = [
        int(v)
        for v in list(locked_train_side_source.get("locked_objective_episode_indices", [4, 7, 11]))
    ]
    current_family_summary = {
        "name": "env12_high_mass_objective_episode_family",
        "locked_objective_episode_indices": list(locked_episode_indices),
        "token_count": int(high_mass_candidate["token_count"]),
        "policy_share_of_batch_negative_mass": float(
            high_mass_candidate["policy_share_of_batch_negative_mass"]
        ),
    }

    fused_episode_indices = sorted(
        {int(first_episode_index)} | {int(v) for v in current_family_summary["locked_objective_episode_indices"]}
    )
    fused_candidate_mask = (
        np.asarray(baseline_snapshot["objective_mask"], dtype=np.int64).reshape(-1) > 0
    ) & (
        np.asarray(baseline_snapshot["flat_env_indices"], dtype=np.int64).reshape(-1) == 12
    ) & np.isin(
        np.asarray(baseline_snapshot["flat_objective_episode_indices"], dtype=np.int64).reshape(-1),
        np.asarray(fused_episode_indices, dtype=np.int64),
    )
    clip = np.asarray(baseline_snapshot["clipped_objective"], dtype=np.float64).reshape(-1)
    objective_mask = np.asarray(baseline_snapshot["objective_mask"], dtype=np.int64).reshape(-1) > 0
    objective_negative_mass = float(np.abs(clip[objective_mask & (clip < 0.0)]).sum())
    fused_negative_mass = float(np.abs(clip[fused_candidate_mask & (clip < 0.0)]).sum())
    fused_candidate = {
        "name": f"dual_channel_fused_first_episode_{first_episode_index}_plus_high_mass",
        "episode_indices": fused_episode_indices,
        "token_count": int(fused_candidate_mask.sum()),
        "policy_share_of_batch_negative_mass": float(
            fused_negative_mass / objective_negative_mass if objective_negative_mass > 0.0 else 0.0
        ),
    }

    recovered_summary = _group_transfer_summary(rows=per_env_rows, env_indices=recovered_envs)
    residual_summary = _group_transfer_summary(rows=per_env_rows, env_indices=residual_envs)
    missed_summary = _group_transfer_summary(rows=per_env_rows, env_indices=missed_envs)
    other_summary = _group_transfer_summary(rows=per_env_rows, env_indices=other_envs)

    return {
        "probe_entry": "phase3_pair2_high_mass_family_antitransfer_probe",
        "inputs": {
            "high_mass_heldout_transfer_compare_json": str(
                Path(high_mass_heldout_transfer_compare_json).expanduser().resolve()
            ),
            "baseline_resume_eval_json": str(Path(baseline_resume_eval_json).expanduser().resolve()),
            "override_resume_eval_json": str(Path(override_resume_eval_json).expanduser().resolve()),
            "positive_regime_anchor_structure_json": str(
                Path(positive_regime_anchor_structure_json).expanduser().resolve()
            ),
            "dual_channel_coverage_json": str(Path(dual_channel_coverage_json).expanduser().resolve()),
            "baseline_snapshot_json": str(Path(baseline_snapshot_json).expanduser().resolve()),
            "family_candidate_json": str(Path(family_candidate_json).expanduser().resolve()),
        },
        "locked_high_mass_transfer": {
            "train_full_return_delta_change": float(
                transfer_compare["comparison"]["train_full_return_delta_change"]
            ),
            "train_suffix_return_delta_change": float(
                transfer_compare["comparison"]["train_suffix_return_delta_change"]
            ),
            "heldout_full_return_delta_change": float(
                transfer_compare["comparison"]["heldout_full_return_delta_change"]
            ),
            "heldout_suffix_return_delta_change": float(
                transfer_compare["comparison"]["heldout_suffix_return_delta_change"]
            ),
        },
        "pair2_anchor_structure": {
            "captured_reset_tail_anchor_count": int(len(pair2_captured_rows)),
            "missed_anchor_count": int(len(pair2_missed_rows)),
            "recovered_dual_channel_anchor_count": int(len(pair2_recovered_rows)),
            "residual_missed_anchor_count": int(len(pair2_residual_rows)),
            "captured_reset_tail_env_indices": sorted(int(row["env_index"]) for row in pair2_captured_rows),
            "missed_env_indices": sorted(int(row["env_index"]) for row in pair2_missed_rows),
            "recovered_dual_channel_env_indices": sorted(int(row["env_index"]) for row in pair2_recovered_rows),
            "residual_missed_env_indices": sorted(int(row["env_index"]) for row in pair2_residual_rows),
        },
        "heldout_transfer_groups": {
            "recovered_dual_channel_anchors": recovered_summary,
            "residual_missed_anchors": residual_summary,
            "all_missed_anchors": missed_summary,
            "other_envs": other_summary,
        },
        "current_env12_family_class": {
            "current_high_mass_candidate": current_family_summary,
            "env12_episode_spans": env12_metrics["episode_spans"],
            "first_objective_episode_candidate": first_episode_candidate,
            "first_objective_episode_index": int(first_episode_index),
            "first_objective_episode_token_count": int(
                env12_metrics["first_objective_episode_token_count"]
            ),
            "first_objective_episode_non_tail_token_count": int(
                env12_metrics["first_objective_episode_non_tail_token_count"]
            ),
            "first_objective_episode_all_tail_under_tail_len_8": bool(
                env12_metrics["first_objective_episode_all_tail_under_tail_len_8"]
            ),
            "dual_channel_fused_candidate": fused_candidate,
        },
        "comparisons": {
            "recovered_minus_residual_mean_suffix_delta": float(
                recovered_summary["mean_suffix_return_delta"] - residual_summary["mean_suffix_return_delta"]
            ),
            "fused_minus_high_mass_policy_share_of_batch_negative_mass": float(
                float(fused_candidate["policy_share_of_batch_negative_mass"])
                - float(current_family_summary["policy_share_of_batch_negative_mass"])
            ),
            "fused_minus_high_mass_token_count": int(
                int(fused_candidate["token_count"]) - int(current_family_summary["token_count"])
            ),
        },
        "conclusions": {
            "pair2_positive_gain_anchors_are_not_reset_tail_captured": bool(len(pair2_captured_rows) == 0),
            "pair2_dual_channel_recovers_nonreset_anchor_subset": bool(len(pair2_recovered_rows) > 0),
            "current_high_mass_family_harms_recovered_dual_channel_anchors_on_average": bool(
                float(recovered_summary["mean_suffix_return_delta"]) < 0.0
            ),
            "current_high_mass_family_rescues_env12_outlier_but_not_anchor_set_broadly": bool(
                any(
                    int(row["env_index"]) == 12 and float(row["suffix_return_delta"]) > 0.0
                    for row in residual_summary["env_rows"]
                )
                and int(residual_summary["negative_suffix_count"]) >= 3
            ),
            "current_env12_local_family_class_has_no_first_episode_non_tail_support": bool(
                env12_metrics["first_objective_episode_all_tail_under_tail_len_8"]
            ),
            "simply_retuning_episode_indices_inside_env12_is_unlikely_to_add_the_missing_channel": bool(
                env12_metrics["first_objective_episode_all_tail_under_tail_len_8"]
            ),
            "next_family_should_leave_current_env12_local_reset_channel_class": bool(
                len(pair2_captured_rows) == 0
                and len(pair2_recovered_rows) > 0
                and env12_metrics["first_objective_episode_all_tail_under_tail_len_8"]
            ),
        },
        "recommendation": {
            "current_high_mass_family_status": "train_positive_but_heldout_negative",
            "recommended_next_family_class": (
                "pair2_dual_channel_family_with_explicit_first_objective_non_tail_coverage_outside_current_env12_local_class"
            ),
            "do_not_repeat": [
                "the same high-mass-family heldout-transfer compare",
                "env12-only episode-retuning inside the current all-tail first-objective-episode class",
            ],
            "reason": (
                "Pair2 gain anchors are missed by the reset-tail channel, dual-channel recovery exists on pair2, "
                "and the current env12-local class has no first-objective non-tail tokens to encode that missing channel."
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read-only pair2 probe explaining why the high-mass family anti-transfers on heldout."
    )
    parser.add_argument(
        "--high-mass-heldout-transfer-compare-json",
        type=str,
        default=DEFAULT_HIGH_MASS_HELDOUT_TRANSFER_COMPARE_JSON,
    )
    parser.add_argument("--baseline-resume-eval-json", type=str, default=DEFAULT_BASELINE_RESUME_EVAL_JSON)
    parser.add_argument("--override-resume-eval-json", type=str, default=DEFAULT_OVERRIDE_RESUME_EVAL_JSON)
    parser.add_argument(
        "--positive-regime-anchor-structure-json",
        type=str,
        default=DEFAULT_POSITIVE_REGIME_ANCHOR_STRUCTURE_JSON,
    )
    parser.add_argument(
        "--dual-channel-coverage-json",
        type=str,
        default=DEFAULT_DUAL_CHANNEL_COVERAGE_JSON,
    )
    parser.add_argument("--baseline-snapshot-json", type=str, default=DEFAULT_BASELINE_SNAPSHOT_JSON)
    parser.add_argument("--family-candidate-json", type=str, default=DEFAULT_FAMILY_CANDIDATE_JSON)
    parser.add_argument("--output-json", type=str, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output_json).expanduser().resolve()
    if output_path.exists() and not bool(args.overwrite):
        print(output_path.read_text(encoding="utf-8"))
        return 0

    report = build_pair2_high_mass_family_antitransfer_probe(
        high_mass_heldout_transfer_compare_json=args.high_mass_heldout_transfer_compare_json,
        baseline_resume_eval_json=args.baseline_resume_eval_json,
        override_resume_eval_json=args.override_resume_eval_json,
        positive_regime_anchor_structure_json=args.positive_regime_anchor_structure_json,
        dual_channel_coverage_json=args.dual_channel_coverage_json,
        baseline_snapshot_json=args.baseline_snapshot_json,
        family_candidate_json=args.family_candidate_json,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(output_path.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
