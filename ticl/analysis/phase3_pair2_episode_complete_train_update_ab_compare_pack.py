import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

from ticl.analysis.phase3_pair2_train_update_ab_compare_pack import (
    PAIR2_CHECKPOINT_PATH,
    PAIR2_HELDOUT_SUITE_PATH,
    PAIR2_PREFLIGHT_JSON,
    PAIR2_TRAIN_SUITE_PATH,
    PAIR2_TRAIN_UPDATE_EVAL_NSAMPLES,
    PAIR2_TRAIN_UPDATE_MATERIAL_EPS,
    PAIR2_TRAIN_UPDATE_NSAMPLES,
    PAIR2_TRAIN_UPDATE_SINGLE_EVAL_POS,
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


PAIR2_EPISODE_COMPLETE_SCOPE_PROBE_JSON = (
    "/home/chen/RLPFN/artifacts/phase3_pair2_branch_specific_extension_class_probe.json"
)
PAIR2_EPISODE_COMPLETE_BRANCH_SPECIFIC_MODE = (
    "tokenwise_scale_env12_selector_matched_objective_episode"
)
PAIR2_EPISODE_COMPLETE_BASELINE_TRAIN_OUTER_BATCH_SNAPSHOT_JSON = (
    "/home/chen/RLPFN/artifacts/phase3_pair2_episode_complete_train_update_ab_baseline_outer_batch_snapshot.json"
)
PAIR2_EPISODE_COMPLETE_OVERRIDE_TRAIN_OUTER_BATCH_SNAPSHOT_JSON = (
    "/home/chen/RLPFN/artifacts/phase3_pair2_episode_complete_train_update_ab_override_outer_batch_snapshot.json"
)
PAIR2_EPISODE_COMPLETE_BASELINE_POST_POLICY_BUNDLE_PATH = (
    "/home/chen/RLPFN/artifacts/phase3_pair2_episode_complete_train_update_ab_baseline_post_policy_bundle.pt"
)
PAIR2_EPISODE_COMPLETE_OVERRIDE_POST_POLICY_BUNDLE_PATH = (
    "/home/chen/RLPFN/artifacts/phase3_pair2_episode_complete_train_update_ab_override_post_policy_bundle.pt"
)


def _load_pair2_episode_complete_locked_selector() -> dict[str, int]:
    path = Path(PAIR2_EPISODE_COMPLETE_SCOPE_PROBE_JSON).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if str(payload.get("probe_entry", "")) != "phase3_pair2_branch_specific_extension_class_probe":
        raise RuntimeError(f"Expected pair2 episode-complete scope probe artifact at {path}")
    scope = dict(payload.get("scope_classes", {}).get("current_selector_matched_full_episode", {}))
    selector = dict(scope.get("selector", {}))
    contract = dict(payload.get("current_trusted_contract", {}))
    return {
        "target_outer_batch_idx": int(contract["target_outer_batch_idx"]),
        "target_env_index": int(selector["env_index"]),
        "target_objective_episode_index": int(selector["objective_episode_index"]),
        "target_objective_position_start": int(selector["objective_position_start"]),
        "target_objective_position_end": int(selector["objective_position_end"]),
    }


def _require_target_snapshot_written(
    report: dict[str, Any],
    *,
    arm_name: str,
    locked_selector: dict[str, int],
) -> dict[str, Any]:
    snapshot = dict(report.get("train_outer_batch_snapshot", {}))
    if not bool(snapshot.get("written", False)):
        raise RuntimeError(
            f"Pair2 episode-complete {arm_name} train-side snapshot was not written."
        )
    selector = dict(snapshot.get("selector", {}))
    if selector != locked_selector:
        raise RuntimeError(
            f"Pair2 episode-complete {arm_name} snapshot selector drifted from the locked selector: {selector!r}"
        )
    if snapshot.get("written_path", None) in {None, ""}:
        raise RuntimeError(
            f"Pair2 episode-complete {arm_name} snapshot reports written=True but has no written_path."
        )
    target_match_count = int(snapshot.get("snapshot_target_match_count", 0))
    if target_match_count <= 0:
        raise RuntimeError(
            f"Pair2 episode-complete {arm_name} snapshot was written but matched no objective tokens "
            f"(snapshot_target_match_count={target_match_count})."
        )
    return snapshot


def build_pair2_episode_complete_train_update_ab_compare_pack() -> dict[str, Any]:
    preflight = _load_pair2_preflight()
    locked_selector = _load_pair2_episode_complete_locked_selector()
    print("[pair2-episode-complete-ab] start baseline audit", flush=True)
    baseline = _run_audit(
        actor_objective_mode_override=None,
        restore_validation_policy_head_state=True,
        save_post_policy_bundle_path=PAIR2_EPISODE_COMPLETE_BASELINE_POST_POLICY_BUNDLE_PATH,
        eval_n_samples=PAIR2_TRAIN_UPDATE_EVAL_NSAMPLES,
        train_outer_batch_snapshot_json=PAIR2_EPISODE_COMPLETE_BASELINE_TRAIN_OUTER_BATCH_SNAPSHOT_JSON,
        train_outer_batch_snapshot_target_outer_batch_idx=locked_selector["target_outer_batch_idx"],
        train_outer_batch_snapshot_target_env_index=locked_selector["target_env_index"],
        train_outer_batch_snapshot_target_objective_episode_index=locked_selector[
            "target_objective_episode_index"
        ],
        train_outer_batch_snapshot_target_objective_position_start=locked_selector[
            "target_objective_position_start"
        ],
        train_outer_batch_snapshot_target_objective_position_end=locked_selector[
            "target_objective_position_end"
        ],
    )
    print("[pair2-episode-complete-ab] start override audit", flush=True)
    with tempfile.TemporaryDirectory(prefix="pair2_episode_complete_ab_") as tmpdir:
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
            actor_objective_mode_override=PAIR2_EPISODE_COMPLETE_BRANCH_SPECIFIC_MODE,
            restore_validation_policy_head_state=True,
            reuse_zero_control_json=reuse_zero_control_json,
            reuse_pre_policy_json=reuse_pre_policy_json,
            save_post_policy_bundle_path=PAIR2_EPISODE_COMPLETE_OVERRIDE_POST_POLICY_BUNDLE_PATH,
            eval_n_samples=PAIR2_TRAIN_UPDATE_EVAL_NSAMPLES,
            train_outer_batch_snapshot_json=PAIR2_EPISODE_COMPLETE_OVERRIDE_TRAIN_OUTER_BATCH_SNAPSHOT_JSON,
            train_outer_batch_snapshot_target_outer_batch_idx=locked_selector["target_outer_batch_idx"],
            train_outer_batch_snapshot_target_env_index=locked_selector["target_env_index"],
            train_outer_batch_snapshot_target_objective_episode_index=locked_selector[
                "target_objective_episode_index"
            ],
            train_outer_batch_snapshot_target_objective_position_start=locked_selector[
                "target_objective_position_start"
            ],
            train_outer_batch_snapshot_target_objective_position_end=locked_selector[
                "target_objective_position_end"
            ],
        )
    print("[pair2-episode-complete-ab] both audits complete; assembling compare pack", flush=True)

    baseline_ctx = _suite_context(baseline)
    override_ctx = _suite_context(override)
    if _shared_contract_context(baseline_ctx) != _shared_contract_context(override_ctx):
        raise ValueError(
            "Baseline and override pair2 episode-complete audits must share the same suite contract, "
            f"got {baseline_ctx!r} vs {override_ctx!r}"
        )
    if baseline_ctx["actor_objective_runtime_current_suite_name"] not in {"", "none", "None"}:
        raise ValueError(
            "Baseline episode-complete audit should not activate a runtime-scoped objective mode: "
            f"{baseline_ctx['actor_objective_runtime_current_suite_name']!r}"
        )
    if baseline_ctx["actor_objective_mode_override"] is not None:
        raise ValueError(
            f"Baseline episode-complete audit should use tokenwise objective without override: {baseline_ctx!r}"
        )
    if str(override_ctx["actor_objective_mode_override"]) != PAIR2_EPISODE_COMPLETE_BRANCH_SPECIFIC_MODE:
        raise ValueError(
            "Override episode-complete audit did not activate expected branch-specific mode: "
            f"{override_ctx['actor_objective_mode_override']!r}"
        )
    if override_ctx["actor_objective_runtime_current_suite_name"] != "pair2":
        raise ValueError(
            "Override episode-complete audit did not target pair2 runtime scope: "
            f"{override_ctx['actor_objective_runtime_current_suite_name']!r}"
        )
    baseline_snapshot = _require_target_snapshot_written(
        baseline,
        arm_name="baseline",
        locked_selector=locked_selector,
    )
    override_snapshot = _require_target_snapshot_written(
        override,
        arm_name="override",
        locked_selector=locked_selector,
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
        "audit_entry": "phase3_pair2_episode_complete_train_update_ab_compare_pack",
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
            "candidate_small_control_target": "env12_selector_matched_objective_episode",
            "runtime_control_shape": "selector_matched_objective_episode_branch_specific_gate",
            "branch_specific": True,
            "locked_selector": locked_selector,
            "locked_selector_source_path": PAIR2_EPISODE_COMPLETE_SCOPE_PROBE_JSON,
            "runtime_mode_activated": bool(
                override_ctx["actor_objective_runtime_current_suite_name"] == "pair2"
                and override_ctx["actor_objective_mode_override"]
                == PAIR2_EPISODE_COMPLETE_BRANCH_SPECIFIC_MODE
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
            "baseline_requested_path": PAIR2_EPISODE_COMPLETE_BASELINE_TRAIN_OUTER_BATCH_SNAPSHOT_JSON,
            "override_requested_path": PAIR2_EPISODE_COMPLETE_OVERRIDE_TRAIN_OUTER_BATCH_SNAPSHOT_JSON,
            "selector": locked_selector,
            "baseline": baseline_snapshot,
            "override": override_snapshot,
        },
        "post_policy_bundles": {
            "baseline_requested_path": PAIR2_EPISODE_COMPLETE_BASELINE_POST_POLICY_BUNDLE_PATH,
            "override_requested_path": PAIR2_EPISODE_COMPLETE_OVERRIDE_POST_POLICY_BUNDLE_PATH,
            "baseline": dict(baseline.get("post_policy_bundle", {})),
            "override": dict(override.get("post_policy_bundle", {})),
        },
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
            "target_selector_present_in_both_arms": True,
            "target_side_compare_is_efficacy_eligible": True,
            "runtime_mode_local_to_pair2": True,
            "shared_core_unchanged": True,
        },
        "recommendation": {
            "candidate_small_control_target": "env12_selector_matched_objective_episode",
            "branch_specific_scope_gate_needed": True,
            "shared_core_unchanged": True,
            "reason": (
                "This is the first size-matched branch-specific extension class to evaluate under the same guarded "
                "pair2 contract. Keep the shared core fixed and treat the result as efficacy evidence only when the "
                "locked selector-matched full objective episode is present in both arms."
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pair2 target-side train-update A/B compare pack for the env12 selector-matched full objective episode control."
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="/home/chen/RLPFN/artifacts/phase3_pair2_episode_complete_train_update_ab_compare_pack.json",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output_json).expanduser().resolve()
    if output_path.exists() and not bool(args.overwrite):
        print(output_path.read_text(encoding="utf-8"))
        return 0

    report = build_pair2_episode_complete_train_update_ab_compare_pack()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
