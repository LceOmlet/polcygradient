import argparse
import faulthandler
import json
import os
import tempfile
from pathlib import Path
from typing import Any

faulthandler.enable(all_threads=True)
faulthandler.dump_traceback_later(120, repeat=True)

from ticl.analysis.phase2_guardrail import DEFAULT_PHASE2_SUMMARY
from ticl.analysis.phase3_multi_env_optimization_audit import run_phase3_multi_env_optimization_audit
from ticl.analysis.phase3_pair2_train_update_ab_compare_pack import (
    PAIR2_BRANCH_SPECIFIC_MODE,
    _build_reused_metrics_payload,
    _gap_delta,
    _shared_contract_context,
    _suite_context,
    _train_history_delta,
    _write_reused_metrics_json,
)


PAIR1_CHECKPOINT_PATH = "/home/chen/RLPFN/rwkv/models_diff/rlpfn_03_30_2026_22_37_53_epoch_13.cpkt"
PAIR1_TRAIN_SUITE_PATH = "/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/train_suite.pt"
PAIR1_HELDOUT_SUITE_PATH = "/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/heldout_suite.pt"
PAIR1_PREFLIGHT_JSON = "/home/chen/RLPFN/artifacts/phase3_preflight_contract_check_pair1.json"

PAIR1_TRAIN_UPDATE_EVAL_NSAMPLES = 256
PAIR1_TRAIN_UPDATE_NSAMPLES = 256
PAIR1_TRAIN_UPDATE_SINGLE_EVAL_POS = 64

PAIR1_BASELINE_POST_POLICY_BUNDLE_PATH = (
    "/home/chen/RLPFN/artifacts/phase3_pair1_train_update_ab_baseline_post_policy_bundle.pt"
)
PAIR1_OVERRIDE_POST_POLICY_BUNDLE_PATH = (
    "/home/chen/RLPFN/artifacts/phase3_pair1_train_update_ab_override_post_policy_bundle.pt"
)


def _load_pair1_preflight() -> dict[str, Any]:
    path = Path(PAIR1_PREFLIGHT_JSON).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not bool(payload.get("preflight_passed", False)):
        raise RuntimeError(f"Pair1 preflight is not green: {path}")
    signature = dict(payload.get("phase3_many_env_signature", {}))
    required = {
        "n_envs": 16,
        "n_steps": PAIR1_TRAIN_UPDATE_NSAMPLES,
        "batch_size": 256,
        "n_epochs": 1,
        "learning_rate": 2e-4,
        "target_kl": 0.03,
        "actor_objective_mode": "tokenwise",
        "strict_native_rollout": True,
        "normalize_advantage": False,
        "separate_value_backbone": False,
        "runtime_normalized_q_value_weight_override": None,
        "runtime_next_state_flow_matching_weight_override": None,
    }
    for key, expected in required.items():
        observed = signature.get(key, None)
        if observed != expected:
            raise RuntimeError(
                f"Pair1 preflight mismatch for {key}: observed={observed!r}, expected={expected!r}"
            )
    if dict(payload.get("contract_diff", {})).get("unexpected_diff", None) != {}:
        raise RuntimeError(f"Pair1 preflight has unexpected contract diff: {path}")
    return payload


def _run_audit(
    *,
    actor_objective_mode_override: str | None,
    restore_validation_policy_head_state: bool = False,
    reuse_zero_control_json: str | None = None,
    reuse_pre_policy_json: str | None = None,
    save_post_policy_bundle_path: str | None = None,
    resume_post_policy_bundle_path: str | None = None,
    eval_n_samples: int = PAIR1_TRAIN_UPDATE_EVAL_NSAMPLES,
) -> dict[str, Any]:
    mode_label = "baseline" if actor_objective_mode_override is None else str(actor_objective_mode_override)
    print(f"[pair1-train-update-ab] start audit mode={mode_label}", flush=True)
    os.environ["TICL_PHASE3_PROFILE_TIMING"] = "1"
    report = run_phase3_multi_env_optimization_audit(
        checkpoint_path=PAIR1_CHECKPOINT_PATH,
        train_suite_path=PAIR1_TRAIN_SUITE_PATH,
        heldout_suite_path=PAIR1_HELDOUT_SUITE_PATH,
        device="cuda:0",
        n_samples=PAIR1_TRAIN_UPDATE_NSAMPLES,
        single_eval_pos=PAIR1_TRAIN_UPDATE_SINGLE_EVAL_POS,
        batch_size=256,
        outer_epochs=1,
        n_epochs=1,
        learning_rate=2e-4,
        target_kl=0.03,
        rollout_backend="serial",
        train_profile="phase2_shared_backbone_contract",
        phase2_summary_path=DEFAULT_PHASE2_SUMMARY,
        reuse_zero_control_json=reuse_zero_control_json,
        reuse_pre_policy_json=reuse_pre_policy_json,
        save_post_policy_bundle_path=save_post_policy_bundle_path,
        resume_post_policy_bundle_path=resume_post_policy_bundle_path,
        strict_fixed_env_mode=True,
        deterministic_actor_sampling=True,
        deterministic_batch_plan=True,
        strict_native_rollout=True,
        restore_validation_policy_head_state=bool(restore_validation_policy_head_state),
        actor_objective_mode_override=actor_objective_mode_override,
        actor_objective_runtime_current_suite_name=(
            "pair1" if actor_objective_mode_override is not None else None
        ),
        skip_heldout_eval=True,
        eval_n_samples=int(eval_n_samples),
    )
    print(f"[pair1-train-update-ab] done audit mode={mode_label}", flush=True)
    return report


def build_pair1_train_update_ab_compare_pack() -> dict[str, Any]:
    preflight = _load_pair1_preflight()
    print("[pair1-train-update-ab] start baseline audit", flush=True)
    baseline = _run_audit(
        actor_objective_mode_override=None,
        restore_validation_policy_head_state=True,
        save_post_policy_bundle_path=PAIR1_BASELINE_POST_POLICY_BUNDLE_PATH,
        eval_n_samples=PAIR1_TRAIN_UPDATE_EVAL_NSAMPLES,
    )
    print("[pair1-train-update-ab] start override audit", flush=True)
    with tempfile.TemporaryDirectory(prefix="pair1_train_update_ab_") as tmpdir:
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
            actor_objective_mode_override=PAIR2_BRANCH_SPECIFIC_MODE,
            restore_validation_policy_head_state=True,
            reuse_zero_control_json=reuse_zero_control_json,
            reuse_pre_policy_json=reuse_pre_policy_json,
            save_post_policy_bundle_path=PAIR1_OVERRIDE_POST_POLICY_BUNDLE_PATH,
            eval_n_samples=PAIR1_TRAIN_UPDATE_EVAL_NSAMPLES,
        )
    print("[pair1-train-update-ab] both audits complete; assembling compare pack", flush=True)

    baseline_ctx = _suite_context(baseline)
    override_ctx = _suite_context(override)
    if _shared_contract_context(baseline_ctx) != _shared_contract_context(override_ctx):
        raise ValueError(
            "Baseline and override pair1 train-update audits must share the same suite contract, "
            f"got {baseline_ctx!r} vs {override_ctx!r}"
        )
    if baseline_ctx["actor_objective_runtime_current_suite_name"] not in {"", "none", "None"}:
        raise ValueError(
            "Baseline audit should not activate a runtime-scoped objective mode: "
            f"{baseline_ctx['actor_objective_runtime_current_suite_name']!r}"
        )
    if baseline_ctx["actor_objective_mode_override"] is not None:
        raise ValueError(
            f"Baseline audit should use tokenwise objective without override: {baseline_ctx!r}"
        )
    if str(override_ctx["actor_objective_mode_override"]) != PAIR2_BRANCH_SPECIFIC_MODE:
        raise ValueError(
            "Override audit did not request expected branch-specific mode: "
            f"{override_ctx['actor_objective_mode_override']!r}"
        )
    if override_ctx["actor_objective_runtime_current_suite_name"] != "pair1":
        raise ValueError(
            "Override audit did not target pair1 runtime scope: "
            f"{override_ctx['actor_objective_runtime_current_suite_name']!r}"
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
    no_target_spillover_observed = bool(
        train_history_delta["histories_identical"]
        and all(abs(float(v)) <= 1e-6 for v in train_gap_delta.values())
        and all(abs(float(v)) <= 1e-6 for v in post_train_gap_delta.values())
        and abs(
            float(override_comparison["train_full_return_delta"] - baseline_comparison["train_full_return_delta"])
        )
        <= 1e-6
        and abs(
            float(override_comparison["train_suffix_return_delta"] - baseline_comparison["train_suffix_return_delta"])
        )
        <= 1e-6
    )

    return {
        "audit_entry": "phase3_pair1_train_update_ab_compare_pack",
        "preflight": {
            "source_path": PAIR1_PREFLIGHT_JSON,
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
            "candidate_small_control_target": "env12_mid_episode_extension_block",
            "runtime_control_shape": "contiguous_four_token_branch_specific_gate",
            "branch_specific": True,
            "runtime_mode_requested_on_suite": "pair1",
            "runtime_mode_target_suite": "pair2",
            "runtime_mode_expected_active": False,
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
        "post_policy_bundles": {
            "baseline_requested_path": PAIR1_BASELINE_POST_POLICY_BUNDLE_PATH,
            "override_requested_path": PAIR1_OVERRIDE_POST_POLICY_BUNDLE_PATH,
            "baseline": dict(baseline.get("post_policy_bundle", {})),
            "override": dict(override.get("post_policy_bundle", {})),
        },
        "conclusions": {
            "target_side_train_update_changes_trajectory": bool(
                not train_history_delta["histories_identical"]
                or any(abs(float(v)) > 1e-6 for v in train_gap_delta.values())
                or any(abs(float(v)) > 1e-6 for v in post_train_gap_delta.values())
            ),
            "pre_metrics_unchanged": bool(
                all(abs(float(v)) <= 1e-6 for v in train_gap_delta.values())
            ),
            "runtime_mode_local_to_pair2": True,
            "shared_core_unchanged": True,
            "non_target_suite_noop_expected": True,
            "non_target_suite_noop_observed": no_target_spillover_observed,
        },
        "recommendation": {
            "candidate_small_control_target": "env12_mid_episode_extension_block",
            "branch_specific_scope_gate_needed": True,
            "shared_core_unchanged": True,
            "reason": (
                "Compare pair1 baseline and pair1 override audits under the same strict contract to confirm "
                "that the pair2-scoped branch-specific control does not create train-side spillover on another suite."
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pair1 target-side A/B compare pack under the Phase 2 shared-backbone contract."
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="/home/chen/RLPFN/artifacts/phase3_pair1_train_update_ab_compare_pack.json",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output_json).expanduser().resolve()
    if output_path.exists() and not bool(args.overwrite):
        print(output_path.read_text())
        return 0

    report = build_pair1_train_update_ab_compare_pack()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
