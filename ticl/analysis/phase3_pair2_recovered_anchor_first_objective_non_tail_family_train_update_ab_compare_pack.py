import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from ticl.analysis.phase3_pair2_train_update_ab_compare_pack import (
    PAIR2_PREFLIGHT_JSON,
    PAIR2_TRAIN_UPDATE_EVAL_NSAMPLES,
    PAIR2_TRAIN_UPDATE_MATERIAL_EPS,
    _build_reused_metrics_payload,
    _gap_delta,
    _has_material_delta,
    _load_pair2_preflight,
    _run_audit,
    _shared_contract_context,
    _suite_context,
    _train_history_delta,
    _write_reused_metrics_json,
)


PAIR2_CURRENT_CONTRACT_DUAL_CHANNEL_FAMILY_PROBE_JSON = (
    "/home/chen/RLPFN/artifacts/phase3_pair2_current_contract_dual_channel_family_probe.json"
)
PAIR2_RECOVERED_ANCHOR_NON_TAIL_BRANCH_SPECIFIC_MODE = (
    "tokenwise_scale_pair2_recovered_anchor_first_objective_non_tail_family"
)
PAIR2_RECOVERED_ANCHOR_NON_TAIL_BASELINE_TRAIN_OUTER_BATCH_SNAPSHOT_JSON = (
    "/home/chen/RLPFN/artifacts/phase3_pair2_recovered_anchor_first_objective_non_tail_family_train_update_ab_baseline_outer_batch_snapshot.json"
)
PAIR2_RECOVERED_ANCHOR_NON_TAIL_OVERRIDE_TRAIN_OUTER_BATCH_SNAPSHOT_JSON = (
    "/home/chen/RLPFN/artifacts/phase3_pair2_recovered_anchor_first_objective_non_tail_family_train_update_ab_override_outer_batch_snapshot.json"
)
PAIR2_RECOVERED_ANCHOR_NON_TAIL_BASELINE_POST_POLICY_BUNDLE_PATH = (
    "/home/chen/RLPFN/artifacts/phase3_pair2_recovered_anchor_first_objective_non_tail_family_train_update_ab_baseline_post_policy_bundle.pt"
)
PAIR2_RECOVERED_ANCHOR_NON_TAIL_OVERRIDE_POST_POLICY_BUNDLE_PATH = (
    "/home/chen/RLPFN/artifacts/phase3_pair2_recovered_anchor_first_objective_non_tail_family_train_update_ab_override_post_policy_bundle.pt"
)


def _load_recovered_anchor_non_tail_family_spec() -> dict[str, Any]:
    path = Path(PAIR2_CURRENT_CONTRACT_DUAL_CHANNEL_FAMILY_PROBE_JSON).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if str(payload.get("probe_entry", "")) != "phase3_pair2_current_contract_dual_channel_family_probe":
        raise RuntimeError(f"Expected current-contract dual-channel family probe artifact at {path}")
    recommendation = dict(payload.get("recommendation", {}))
    if str(recommendation.get("recommended_next_family", "")) != "recovered_anchor_first_objective_non_tail_family":
        raise RuntimeError(
            "Current-contract probe does not currently recommend recovered_anchor_first_objective_non_tail_family."
        )
    candidate = dict(recommendation.get("candidate", {}))
    distribution = dict(payload.get("recovered_anchor_first_objective_non_tail_outer_batch_distribution", {}))
    dominant_selector = recommendation.get("recommended_snapshot_selector", None)
    if dominant_selector is None:
        raise RuntimeError(
            "Current-contract probe did not emit a dominant snapshot selector for the recovered-anchor family."
        )
    env_position_end = {
        int(k): int(v)
        for k, v in dict(distribution.get("target_env_position_end_by_env", {})).items()
    }
    return {
        "family_probe_source_path": str(path),
        "target_env_indices": [int(v) for v in candidate.get("env_indices", [])],
        "target_objective_episode_index": 0,
        "target_env_position_end_by_env": env_position_end,
        "family_token_count": int(candidate["token_count"]),
        "family_raw_negative_share_of_batch": float(candidate["raw_negative_share_of_batch"]),
        "dominant_snapshot_selector": {
            "target_outer_batch_idx": int(dominant_selector["target_outer_batch_idx"]),
            "target_env_index": int(dominant_selector["target_env_index"]),
            "target_objective_episode_index": int(dominant_selector["target_objective_episode_index"]),
            "target_objective_position_start": int(dominant_selector["target_objective_position_start"]),
            "target_objective_position_end": int(dominant_selector["target_objective_position_end"]),
        },
        "outer_batch_distribution": dict(distribution),
    }


def _require_snapshot_written_with_selector(
    report: dict[str, Any],
    *,
    expected_selector: dict[str, int],
    arm_name: str,
) -> dict[str, Any]:
    snapshot = dict(report.get("train_outer_batch_snapshot", {}))
    if not bool(snapshot.get("written", False)):
        raise RuntimeError(f"Pair2 recovered-anchor non-tail {arm_name} snapshot was not written.")
    selector = dict(snapshot.get("selector", {}))
    if selector != expected_selector:
        raise RuntimeError(
            f"Pair2 recovered-anchor non-tail {arm_name} snapshot selector drifted: "
            f"{selector!r} vs {expected_selector!r}"
        )
    if snapshot.get("written_path", None) in {None, ""}:
        raise RuntimeError(
            f"Pair2 recovered-anchor non-tail {arm_name} snapshot reports written=True but has no path."
        )
    if int(snapshot.get("snapshot_target_match_count", 0)) <= 0:
        raise RuntimeError(
            f"Pair2 recovered-anchor non-tail {arm_name} snapshot matched no dominant family tokens."
        )
    return snapshot


def _read_dominant_snapshot_summary(
    snapshot_path: str,
    *,
    selector: dict[str, int],
) -> dict[str, Any]:
    payload = json.loads(Path(snapshot_path).expanduser().resolve().read_text(encoding="utf-8"))
    objective_mask = np.asarray(payload["objective_mask"], dtype=np.int64).reshape(-1) > 0
    env_indices = np.asarray(payload["flat_env_indices"], dtype=np.int64).reshape(-1)
    objective_episode_indices = np.asarray(
        payload["flat_objective_episode_indices"],
        dtype=np.int64,
    ).reshape(-1)
    objective_episode_positions = np.asarray(
        payload["flat_objective_episode_positions"],
        dtype=np.int64,
    ).reshape(-1)
    clip = np.asarray(payload["clipped_objective"], dtype=np.float64).reshape(-1)
    target_mask = (
        objective_mask
        & (env_indices == int(selector["target_env_index"]))
        & (objective_episode_indices == int(selector["target_objective_episode_index"]))
        & (objective_episode_positions >= int(selector["target_objective_position_start"]))
        & (objective_episode_positions <= int(selector["target_objective_position_end"]))
    )
    token_count = int(target_mask.sum())
    negative_mass = float(np.abs(clip[target_mask & (clip < 0.0)]).sum()) if token_count > 0 else 0.0
    objective_negative_mass = float(np.abs(clip[objective_mask & (clip < 0.0)]).sum())
    return {
        "target_env_index": int(selector["target_env_index"]),
        "target_objective_episode_index": int(selector["target_objective_episode_index"]),
        "target_objective_position_start": int(selector["target_objective_position_start"]),
        "target_objective_position_end": int(selector["target_objective_position_end"]),
        "token_count": int(token_count),
        "negative_mass": float(negative_mass),
        "policy_share_of_snapshot_negative_mass": float(
            negative_mass / objective_negative_mass if objective_negative_mass > 0.0 else 0.0
        ),
    }


def build_pair2_recovered_anchor_first_objective_non_tail_family_train_update_ab_compare_pack() -> dict[str, Any]:
    preflight = _load_pair2_preflight()
    family_spec = _load_recovered_anchor_non_tail_family_spec()
    dominant_selector = dict(family_spec["dominant_snapshot_selector"])

    print("[pair2-recovered-anchor-nontail-family-ab] start baseline audit", flush=True)
    baseline = _run_audit(
        actor_objective_mode_override=None,
        restore_validation_policy_head_state=True,
        save_post_policy_bundle_path=PAIR2_RECOVERED_ANCHOR_NON_TAIL_BASELINE_POST_POLICY_BUNDLE_PATH,
        eval_n_samples=PAIR2_TRAIN_UPDATE_EVAL_NSAMPLES,
        train_outer_batch_snapshot_json=PAIR2_RECOVERED_ANCHOR_NON_TAIL_BASELINE_TRAIN_OUTER_BATCH_SNAPSHOT_JSON,
        train_outer_batch_snapshot_target_outer_batch_idx=dominant_selector["target_outer_batch_idx"],
        train_outer_batch_snapshot_target_env_index=dominant_selector["target_env_index"],
        train_outer_batch_snapshot_target_objective_episode_index=dominant_selector[
            "target_objective_episode_index"
        ],
        train_outer_batch_snapshot_target_objective_position_start=dominant_selector[
            "target_objective_position_start"
        ],
        train_outer_batch_snapshot_target_objective_position_end=dominant_selector[
            "target_objective_position_end"
        ],
    )
    print("[pair2-recovered-anchor-nontail-family-ab] start override audit", flush=True)
    with tempfile.TemporaryDirectory(prefix="pair2_recovered_anchor_nontail_family_ab_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        zero_payload = _build_reused_metrics_payload(
            report=baseline,
            policy_mode="zero",
            train_metrics=baseline["zero_control"]["train"],
            heldout_metrics=baseline["zero_control"]["heldout"],
        )
        pre_payload = _build_reused_metrics_payload(
            report=baseline,
            policy_mode="ppo",
            train_metrics=baseline["pre"]["train"],
        )
        reuse_zero_control_json = _write_reused_metrics_json(tmpdir_path / "zero_control.json", zero_payload)
        reuse_pre_policy_json = _write_reused_metrics_json(tmpdir_path / "pre_policy.json", pre_payload)
        override = _run_audit(
            actor_objective_mode_override=PAIR2_RECOVERED_ANCHOR_NON_TAIL_BRANCH_SPECIFIC_MODE,
            restore_validation_policy_head_state=True,
            reuse_zero_control_json=reuse_zero_control_json,
            reuse_pre_policy_json=reuse_pre_policy_json,
            save_post_policy_bundle_path=PAIR2_RECOVERED_ANCHOR_NON_TAIL_OVERRIDE_POST_POLICY_BUNDLE_PATH,
            eval_n_samples=PAIR2_TRAIN_UPDATE_EVAL_NSAMPLES,
            train_outer_batch_snapshot_json=PAIR2_RECOVERED_ANCHOR_NON_TAIL_OVERRIDE_TRAIN_OUTER_BATCH_SNAPSHOT_JSON,
            train_outer_batch_snapshot_target_outer_batch_idx=dominant_selector["target_outer_batch_idx"],
            train_outer_batch_snapshot_target_env_index=dominant_selector["target_env_index"],
            train_outer_batch_snapshot_target_objective_episode_index=dominant_selector[
                "target_objective_episode_index"
            ],
            train_outer_batch_snapshot_target_objective_position_start=dominant_selector[
                "target_objective_position_start"
            ],
            train_outer_batch_snapshot_target_objective_position_end=dominant_selector[
                "target_objective_position_end"
            ],
        )
    print("[pair2-recovered-anchor-nontail-family-ab] both audits complete; assembling compare pack", flush=True)

    baseline_ctx = _suite_context(baseline)
    override_ctx = _suite_context(override)
    if _shared_contract_context(baseline_ctx) != _shared_contract_context(override_ctx):
        raise ValueError(
            "Baseline and override recovered-anchor non-tail family audits must share the same suite contract, "
            f"got {baseline_ctx!r} vs {override_ctx!r}"
        )
    if baseline_ctx["actor_objective_runtime_current_suite_name"] not in {"", "none"}:
        raise ValueError(
            "Baseline recovered-anchor non-tail audit should not activate a runtime-scoped objective mode: "
            f"{baseline_ctx['actor_objective_runtime_current_suite_name']!r}"
        )
    if baseline_ctx["actor_objective_mode_override"] is not None:
        raise ValueError(
            f"Baseline recovered-anchor non-tail audit should use tokenwise objective without override: {baseline_ctx!r}"
        )
    if (
        str(override_ctx["actor_objective_mode_override"])
        != PAIR2_RECOVERED_ANCHOR_NON_TAIL_BRANCH_SPECIFIC_MODE
    ):
        raise ValueError(
            "Override recovered-anchor non-tail audit did not activate expected branch-specific mode: "
            f"{override_ctx['actor_objective_mode_override']!r}"
        )
    if override_ctx["actor_objective_runtime_current_suite_name"] != "pair2":
        raise ValueError(
            "Override recovered-anchor non-tail audit did not target pair2 runtime scope: "
            f"{override_ctx['actor_objective_runtime_current_suite_name']!r}"
        )

    baseline_snapshot = _require_snapshot_written_with_selector(
        baseline,
        expected_selector=dominant_selector,
        arm_name="baseline",
    )
    override_snapshot = _require_snapshot_written_with_selector(
        override,
        expected_selector=dominant_selector,
        arm_name="override",
    )
    baseline_dominant_summary = _read_dominant_snapshot_summary(
        baseline_snapshot["written_path"],
        selector=dominant_selector,
    )
    override_dominant_summary = _read_dominant_snapshot_summary(
        override_snapshot["written_path"],
        selector=dominant_selector,
    )

    baseline_comparison = dict(baseline["comparison"])
    override_comparison = dict(override["comparison"])
    train_gap_delta = _gap_delta(
        dict(baseline_comparison["pre_train_vs_zero"]),
        dict(override_comparison["pre_train_vs_zero"]),
    )
    post_train_gap_delta = _gap_delta(
        dict(baseline_comparison["post_train_vs_zero"]),
        dict(override_comparison["post_train_vs_zero"]),
    )
    train_history_delta = _train_history_delta(
        list(baseline["train_history"]),
        list(override["train_history"]),
    )

    return {
        "audit_entry": "phase3_pair2_recovered_anchor_first_objective_non_tail_family_train_update_ab_compare_pack",
        "preflight": {
            "source_path": PAIR2_PREFLIGHT_JSON,
            "preflight_passed": bool(preflight.get("preflight_passed", False)),
            "contract_args": dict(preflight.get("contract_args", {})),
            "allowed_diff": dict(dict(preflight.get("contract_diff", {})).get("allowed_diff", {})),
            "unexpected_diff": dict(dict(preflight.get("contract_diff", {})).get("unexpected_diff", {})),
        },
        "suite_context": baseline_ctx,
        "shared_contract_context": _shared_contract_context(baseline_ctx),
        "override_target_context": {
            "actor_objective_runtime_current_suite_name": override_ctx["actor_objective_runtime_current_suite_name"],
            "actor_objective_mode_override": override_ctx["actor_objective_mode_override"],
        },
        "target_control": {
            "candidate_small_control_target": "recovered_anchor_first_objective_non_tail_family",
            "runtime_control_shape": "pair2_recovered_anchor_first_objective_non_tail_family_branch_specific_gate",
            "branch_specific": True,
            "family_probe_source_path": family_spec["family_probe_source_path"],
            "target_env_indices": list(family_spec["target_env_indices"]),
            "target_objective_episode_index": int(family_spec["target_objective_episode_index"]),
            "target_env_position_end_by_env": dict(family_spec["target_env_position_end_by_env"]),
            "family_token_count": int(family_spec["family_token_count"]),
            "family_raw_negative_share_of_batch": float(
                family_spec["family_raw_negative_share_of_batch"]
            ),
            "dominant_snapshot_selector": dominant_selector,
            "runtime_mode_activated": bool(
                override_ctx["actor_objective_runtime_current_suite_name"] == "pair2"
                and override_ctx["actor_objective_mode_override"]
                == PAIR2_RECOVERED_ANCHOR_NON_TAIL_BRANCH_SPECIFIC_MODE
            ),
        },
        "baseline": {
            "comparison": baseline_comparison,
            "train_history": list(baseline["train_history"]),
        },
        "override": {
            "comparison": override_comparison,
            "train_history": list(override["train_history"]),
        },
        "comparison": {
            "pre_train_vs_zero_gap_delta": train_gap_delta,
            "post_train_vs_zero_gap_delta": post_train_gap_delta,
            "train_full_return_delta_change": float(
                override_comparison["train_full_return_delta"] - baseline_comparison["train_full_return_delta"]
            ),
            "train_suffix_return_delta_change": float(
                override_comparison["train_suffix_return_delta"] - baseline_comparison["train_suffix_return_delta"]
            ),
            "train_history_delta": train_history_delta,
            "heldout_eval_skipped": bool(
                baseline.get("heldout_eval_skipped", True) and override.get("heldout_eval_skipped", True)
            ),
        },
        "train_outer_batch_snapshots": {
            "baseline_requested_path": PAIR2_RECOVERED_ANCHOR_NON_TAIL_BASELINE_TRAIN_OUTER_BATCH_SNAPSHOT_JSON,
            "override_requested_path": PAIR2_RECOVERED_ANCHOR_NON_TAIL_OVERRIDE_TRAIN_OUTER_BATCH_SNAPSHOT_JSON,
            "dominant_snapshot_selector": dominant_selector,
            "baseline": baseline_snapshot,
            "override": override_snapshot,
            "baseline_dominant_segment_summary": baseline_dominant_summary,
            "override_dominant_segment_summary": override_dominant_summary,
        },
        "post_policy_bundles": {
            "baseline_requested_path": PAIR2_RECOVERED_ANCHOR_NON_TAIL_BASELINE_POST_POLICY_BUNDLE_PATH,
            "override_requested_path": PAIR2_RECOVERED_ANCHOR_NON_TAIL_OVERRIDE_POST_POLICY_BUNDLE_PATH,
            "baseline": dict(baseline.get("post_policy_bundle", {})),
            "override": dict(override.get("post_policy_bundle", {})),
        },
        "family_distribution": dict(family_spec["outer_batch_distribution"]),
        "conclusions": {
            "target_side_train_update_changes_trajectory": bool(
                not train_history_delta["histories_identical"]
                or _has_material_delta(train_gap_delta)
                or _has_material_delta(post_train_gap_delta)
                or abs(
                    float(
                        override_comparison["train_full_return_delta"]
                        - baseline_comparison["train_full_return_delta"]
                    )
                )
                > float(PAIR2_TRAIN_UPDATE_MATERIAL_EPS)
                or abs(
                    float(
                        override_comparison["train_suffix_return_delta"]
                        - baseline_comparison["train_suffix_return_delta"]
                    )
                )
                > float(PAIR2_TRAIN_UPDATE_MATERIAL_EPS)
            ),
            "pre_metrics_unchanged": bool(
                all(abs(float(v)) <= 1e-6 for v in train_gap_delta.values())
            ),
            "target_side_effect_is_only_numeric_noise": bool(
                train_history_delta["histories_identical"]
                and not _has_material_delta(train_gap_delta)
                and not _has_material_delta(post_train_gap_delta)
                and abs(
                    float(
                        override_comparison["train_full_return_delta"]
                        - baseline_comparison["train_full_return_delta"]
                    )
                )
                <= float(PAIR2_TRAIN_UPDATE_MATERIAL_EPS)
                and abs(
                    float(
                        override_comparison["train_suffix_return_delta"]
                        - baseline_comparison["train_suffix_return_delta"]
                    )
                )
                <= float(PAIR2_TRAIN_UPDATE_MATERIAL_EPS)
            ),
            "dominant_snapshot_present_in_both_arms": True,
            "target_side_compare_is_efficacy_eligible": True,
            "family_spans_multiple_outer_batches": bool(
                family_spec["outer_batch_distribution"]["family_spans_multiple_outer_batches"]
            ),
            "single_snapshot_only_covers_dominant_family_segment": bool(
                not family_spec["outer_batch_distribution"]["single_outer_batch_covers_full_family"]
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pair2 train-side A/B compare for recovered-anchor first-objective non-tail branch-specific family."
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=(
            "/home/chen/RLPFN/artifacts/"
            "phase3_pair2_recovered_anchor_first_objective_non_tail_family_train_update_ab_compare_pack.json"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output_json).expanduser().resolve()
    if output_path.exists() and not bool(args.overwrite):
        print(output_path.read_text(encoding="utf-8"))
        return 0

    report = build_pair2_recovered_anchor_first_objective_non_tail_family_train_update_ab_compare_pack()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(output_path.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
