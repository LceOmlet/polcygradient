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


PAIR2_FAMILY_CANDIDATE_PROBE_JSON = (
    "/home/chen/RLPFN/artifacts/phase3_pair2_branch_specific_family_candidate_probe.json"
)
PAIR2_HIGH_MASS_FAMILY_BRANCH_SPECIFIC_MODE = (
    "tokenwise_scale_env12_high_mass_objective_episode_family"
)
PAIR2_HIGH_MASS_FAMILY_BASELINE_TRAIN_OUTER_BATCH_SNAPSHOT_JSON = (
    "/home/chen/RLPFN/artifacts/phase3_pair2_high_mass_episode_family_train_update_ab_baseline_outer_batch_snapshot.json"
)
PAIR2_HIGH_MASS_FAMILY_OVERRIDE_TRAIN_OUTER_BATCH_SNAPSHOT_JSON = (
    "/home/chen/RLPFN/artifacts/phase3_pair2_high_mass_episode_family_train_update_ab_override_outer_batch_snapshot.json"
)
PAIR2_HIGH_MASS_FAMILY_BASELINE_POST_POLICY_BUNDLE_PATH = (
    "/home/chen/RLPFN/artifacts/phase3_pair2_high_mass_episode_family_train_update_ab_baseline_post_policy_bundle.pt"
)
PAIR2_HIGH_MASS_FAMILY_OVERRIDE_POST_POLICY_BUNDLE_PATH = (
    "/home/chen/RLPFN/artifacts/phase3_pair2_high_mass_episode_family_train_update_ab_override_post_policy_bundle.pt"
)


def _load_pair2_high_mass_family_spec() -> dict[str, Any]:
    path = Path(PAIR2_FAMILY_CANDIDATE_PROBE_JSON).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if str(payload.get("probe_entry", "")) != "phase3_pair2_branch_specific_family_candidate_probe":
        raise RuntimeError(f"Expected pair2 family candidate probe artifact at {path}")
    family = dict(payload.get("recommended_family", {}))
    return {
        "target_outer_batch_idx": int(family["locked_outer_batch_idx"]),
        "target_env_index": int(family["locked_env_index"]),
        "target_objective_episode_indices": [int(v) for v in family["locked_objective_episode_indices"]],
        "target_anchor_episode_index": int(family["locked_objective_episode_indices"][0]),
    }


def _read_family_snapshot_summary(
    snapshot_path: str,
    *,
    outer_batch_idx: int,
    env_index: int,
    objective_episode_indices: list[int],
) -> dict[str, Any]:
    payload = json.loads(Path(snapshot_path).expanduser().resolve().read_text(encoding="utf-8"))
    objective_mask = np.asarray(payload["objective_mask"], dtype=np.int64).reshape(-1) > 0
    env_indices = np.asarray(payload["flat_env_indices"], dtype=np.int64).reshape(-1)
    objective_episode_indices_flat = np.asarray(
        payload["flat_objective_episode_indices"], dtype=np.int64
    ).reshape(-1)
    clip = np.asarray(payload["clipped_objective"], dtype=np.float64).reshape(-1)
    target_episodes = {int(v) for v in objective_episode_indices}
    family_mask = (
        objective_mask
        & (env_indices == int(env_index))
        & np.isin(objective_episode_indices_flat, np.asarray(sorted(target_episodes), dtype=np.int64))
    )
    family_token_count = int(family_mask.sum())
    family_negative_mass = float(np.abs(clip[family_mask & (clip < 0.0)]).sum()) if family_token_count > 0 else 0.0
    objective_negative_mass = float(np.abs(clip[objective_mask & (clip < 0.0)]).sum())
    return {
        "outer_batch_idx": int(outer_batch_idx),
        "env_index": int(env_index),
        "objective_episode_indices": sorted(int(v) for v in target_episodes),
        "family_token_count": int(family_token_count),
        "family_negative_mass": float(family_negative_mass),
        "family_policy_share_of_batch_negative_mass": float(
            family_negative_mass / objective_negative_mass if objective_negative_mass > 0.0 else 0.0
        ),
    }


def build_pair2_high_mass_episode_family_train_update_ab_compare_pack() -> dict[str, Any]:
    preflight = _load_pair2_preflight()
    family_spec = _load_pair2_high_mass_family_spec()
    anchor_selector = {
        "target_outer_batch_idx": int(family_spec["target_outer_batch_idx"]),
        "target_env_index": int(family_spec["target_env_index"]),
        "target_objective_episode_index": int(family_spec["target_anchor_episode_index"]),
        "target_objective_position_start": 0,
        "target_objective_position_end": 28,
    }

    print("[pair2-high-mass-family-ab] start baseline audit", flush=True)
    baseline = _run_audit(
        actor_objective_mode_override=None,
        restore_validation_policy_head_state=True,
        save_post_policy_bundle_path=PAIR2_HIGH_MASS_FAMILY_BASELINE_POST_POLICY_BUNDLE_PATH,
        eval_n_samples=PAIR2_TRAIN_UPDATE_EVAL_NSAMPLES,
        train_outer_batch_snapshot_json=PAIR2_HIGH_MASS_FAMILY_BASELINE_TRAIN_OUTER_BATCH_SNAPSHOT_JSON,
        train_outer_batch_snapshot_target_outer_batch_idx=anchor_selector["target_outer_batch_idx"],
        train_outer_batch_snapshot_target_env_index=anchor_selector["target_env_index"],
        train_outer_batch_snapshot_target_objective_episode_index=anchor_selector[
            "target_objective_episode_index"
        ],
        train_outer_batch_snapshot_target_objective_position_start=anchor_selector[
            "target_objective_position_start"
        ],
        train_outer_batch_snapshot_target_objective_position_end=anchor_selector[
            "target_objective_position_end"
        ],
    )
    print("[pair2-high-mass-family-ab] start override audit", flush=True)
    with tempfile.TemporaryDirectory(prefix="pair2_high_mass_family_ab_") as tmpdir:
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
            actor_objective_mode_override=PAIR2_HIGH_MASS_FAMILY_BRANCH_SPECIFIC_MODE,
            restore_validation_policy_head_state=True,
            reuse_zero_control_json=reuse_zero_control_json,
            reuse_pre_policy_json=reuse_pre_policy_json,
            save_post_policy_bundle_path=PAIR2_HIGH_MASS_FAMILY_OVERRIDE_POST_POLICY_BUNDLE_PATH,
            eval_n_samples=PAIR2_TRAIN_UPDATE_EVAL_NSAMPLES,
            train_outer_batch_snapshot_json=PAIR2_HIGH_MASS_FAMILY_OVERRIDE_TRAIN_OUTER_BATCH_SNAPSHOT_JSON,
            train_outer_batch_snapshot_target_outer_batch_idx=anchor_selector["target_outer_batch_idx"],
            train_outer_batch_snapshot_target_env_index=anchor_selector["target_env_index"],
            train_outer_batch_snapshot_target_objective_episode_index=anchor_selector[
                "target_objective_episode_index"
            ],
            train_outer_batch_snapshot_target_objective_position_start=anchor_selector[
                "target_objective_position_start"
            ],
            train_outer_batch_snapshot_target_objective_position_end=anchor_selector[
                "target_objective_position_end"
            ],
        )
    print("[pair2-high-mass-family-ab] both audits complete; assembling compare pack", flush=True)

    baseline_ctx = _suite_context(baseline)
    override_ctx = _suite_context(override)
    if _shared_contract_context(baseline_ctx) != _shared_contract_context(override_ctx):
        raise ValueError(
            "Baseline and override pair2 high-mass-family audits must share the same suite contract, "
            f"got {baseline_ctx!r} vs {override_ctx!r}"
        )
    if baseline_ctx["actor_objective_runtime_current_suite_name"] not in {"", "none"}:
        raise ValueError(
            "Baseline high-mass-family audit should not activate a runtime-scoped objective mode: "
            f"{baseline_ctx['actor_objective_runtime_current_suite_name']!r}"
        )
    if baseline_ctx["actor_objective_mode_override"] is not None:
        raise ValueError(
            f"Baseline high-mass-family audit should use tokenwise objective without override: {baseline_ctx!r}"
        )
    if str(override_ctx["actor_objective_mode_override"]) != PAIR2_HIGH_MASS_FAMILY_BRANCH_SPECIFIC_MODE:
        raise ValueError(
            "Override high-mass-family audit did not activate expected branch-specific mode: "
            f"{override_ctx['actor_objective_mode_override']!r}"
        )
    if override_ctx["actor_objective_runtime_current_suite_name"] != "pair2":
        raise ValueError(
            "Override high-mass-family audit did not target pair2 runtime scope: "
            f"{override_ctx['actor_objective_runtime_current_suite_name']!r}"
        )

    baseline_snapshot_meta = dict(baseline["train_outer_batch_snapshot"])
    override_snapshot_meta = dict(override["train_outer_batch_snapshot"])
    if not bool(baseline_snapshot_meta.get("written", False)) or not bool(override_snapshot_meta.get("written", False)):
        raise RuntimeError("High-mass-family compare requires written train-side snapshots in both arms.")
    baseline_family = _read_family_snapshot_summary(
        baseline_snapshot_meta["written_path"],
        outer_batch_idx=family_spec["target_outer_batch_idx"],
        env_index=family_spec["target_env_index"],
        objective_episode_indices=family_spec["target_objective_episode_indices"],
    )
    override_family = _read_family_snapshot_summary(
        override_snapshot_meta["written_path"],
        outer_batch_idx=family_spec["target_outer_batch_idx"],
        env_index=family_spec["target_env_index"],
        objective_episode_indices=family_spec["target_objective_episode_indices"],
    )
    if int(baseline_family["family_token_count"]) <= 0 or int(override_family["family_token_count"]) <= 0:
        raise RuntimeError("High-mass-family compare matched no family tokens in one or both arms.")

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
        "audit_entry": "phase3_pair2_high_mass_episode_family_train_update_ab_compare_pack",
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
            "candidate_small_control_target": "env12_high_mass_objective_episode_family",
            "runtime_control_shape": "repeated_high_mass_complete_episode_family_branch_specific_gate",
            "branch_specific": True,
            "locked_outer_batch_idx": int(family_spec["target_outer_batch_idx"]),
            "locked_env_index": int(family_spec["target_env_index"]),
            "locked_objective_episode_indices": list(family_spec["target_objective_episode_indices"]),
            "family_probe_source_path": PAIR2_FAMILY_CANDIDATE_PROBE_JSON,
            "runtime_mode_activated": bool(
                override_ctx["actor_objective_runtime_current_suite_name"] == "pair2"
                and override_ctx["actor_objective_mode_override"]
                == PAIR2_HIGH_MASS_FAMILY_BRANCH_SPECIFIC_MODE
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
            "baseline_requested_path": PAIR2_HIGH_MASS_FAMILY_BASELINE_TRAIN_OUTER_BATCH_SNAPSHOT_JSON,
            "override_requested_path": PAIR2_HIGH_MASS_FAMILY_OVERRIDE_TRAIN_OUTER_BATCH_SNAPSHOT_JSON,
            "anchor_selector": anchor_selector,
            "family_spec": family_spec,
            "baseline": {
                **baseline_snapshot_meta,
                "family_summary": baseline_family,
            },
            "override": {
                **override_snapshot_meta,
                "family_summary": override_family,
            },
        },
        "post_policy_bundles": {
            "baseline_requested_path": PAIR2_HIGH_MASS_FAMILY_BASELINE_POST_POLICY_BUNDLE_PATH,
            "override_requested_path": PAIR2_HIGH_MASS_FAMILY_OVERRIDE_POST_POLICY_BUNDLE_PATH,
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
            "target_family_present_in_both_arms": True,
            "target_side_compare_is_efficacy_eligible": True,
            "runtime_mode_local_to_pair2": True,
            "shared_core_unchanged": True,
        },
        "recommendation": {
            "candidate_small_control_target": "env12_high_mass_objective_episode_family",
            "branch_specific_scope_gate_needed": True,
            "shared_core_unchanged": True,
            "reason": (
                "This is the first larger-than-episode-complete branch-specific family chosen under the trusted pair2 "
                "contract to increase coverage without collapsing into whole-batch scope."
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pair2 guarded compare pack for the next larger high-mass episode family branch-specific control."
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="/home/chen/RLPFN/artifacts/phase3_pair2_high_mass_episode_family_train_update_ab_compare_pack.json",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output_json).expanduser().resolve()
    if output_path.exists() and not bool(args.overwrite):
        print(output_path.read_text(encoding="utf-8"))
        return 0

    report = build_pair2_high_mass_episode_family_train_update_ab_compare_pack()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(output_path.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
