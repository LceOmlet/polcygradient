import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

from ticl.analysis.phase3_pair1_train_update_ab_compare_pack import (
    PAIR1_PREFLIGHT_JSON,
    PAIR1_TRAIN_UPDATE_EVAL_NSAMPLES,
    _load_pair1_preflight,
    _run_audit,
)
from ticl.analysis.phase3_pair2_train_update_ab_compare_pack import (
    _build_reused_metrics_payload,
    _gap_delta,
    _shared_contract_context,
    _suite_context,
    _train_history_delta,
    _write_reused_metrics_json,
)


PAIR1_BASELINE_REPEAT_PRIMARY_POST_POLICY_BUNDLE_PATH = (
    "/home/chen/RLPFN/artifacts/phase3_pair1_baseline_repeat_primary_post_policy_bundle.pt"
)
PAIR1_BASELINE_REPEAT_REPEAT_POST_POLICY_BUNDLE_PATH = (
    "/home/chen/RLPFN/artifacts/phase3_pair1_baseline_repeat_repeat_post_policy_bundle.pt"
)


def build_pair1_baseline_repeat_compare_pack() -> dict[str, Any]:
    preflight = _load_pair1_preflight()
    print("[pair1-baseline-repeat] start primary baseline audit", flush=True)
    primary = _run_audit(
        actor_objective_mode_override=None,
        restore_validation_policy_head_state=True,
        save_post_policy_bundle_path=PAIR1_BASELINE_REPEAT_PRIMARY_POST_POLICY_BUNDLE_PATH,
        eval_n_samples=PAIR1_TRAIN_UPDATE_EVAL_NSAMPLES,
    )
    print("[pair1-baseline-repeat] start repeat baseline audit", flush=True)
    with tempfile.TemporaryDirectory(prefix="pair1_baseline_repeat_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        zero_payload = _build_reused_metrics_payload(
            report=primary,
            policy_mode="zero",
            train_metrics=primary["zero_control"]["train"],
            heldout_metrics=primary["zero_control"]["heldout"],
        )
        pre_payload = _build_reused_metrics_payload(
            report=primary,
            policy_mode="ppo",
            train_metrics=primary["pre"]["train"],
        )
        reuse_zero_control_json = _write_reused_metrics_json(tmpdir_path / "zero_control.json", zero_payload)
        reuse_pre_policy_json = _write_reused_metrics_json(tmpdir_path / "pre_policy.json", pre_payload)
        repeat = _run_audit(
            actor_objective_mode_override=None,
            restore_validation_policy_head_state=True,
            reuse_zero_control_json=reuse_zero_control_json,
            reuse_pre_policy_json=reuse_pre_policy_json,
            save_post_policy_bundle_path=PAIR1_BASELINE_REPEAT_REPEAT_POST_POLICY_BUNDLE_PATH,
            eval_n_samples=PAIR1_TRAIN_UPDATE_EVAL_NSAMPLES,
        )
    print("[pair1-baseline-repeat] both audits complete; assembling compare pack", flush=True)

    primary_ctx = _suite_context(primary)
    repeat_ctx = _suite_context(repeat)
    if _shared_contract_context(primary_ctx) != _shared_contract_context(repeat_ctx):
        raise ValueError(
            "Primary and repeat pair1 baseline audits must share the same suite contract, "
            f"got {primary_ctx!r} vs {repeat_ctx!r}"
        )
    for name, ctx in [("primary", primary_ctx), ("repeat", repeat_ctx)]:
        if ctx["actor_objective_runtime_current_suite_name"] not in {"", "none", "None"}:
            raise ValueError(
                f"{name} baseline audit should not activate a runtime-scoped objective mode: "
                f"{ctx['actor_objective_runtime_current_suite_name']!r}"
            )
        if ctx["actor_objective_mode_override"] is not None:
            raise ValueError(
                f"{name} baseline audit should use tokenwise objective without override: {ctx!r}"
            )

    primary_comparison = dict(primary["comparison"])
    repeat_comparison = dict(repeat["comparison"])
    train_gap_delta = _gap_delta(
        dict(primary_comparison["pre_train_vs_zero"]),
        dict(repeat_comparison["pre_train_vs_zero"]),
    )
    post_train_gap_delta = _gap_delta(
        dict(primary_comparison["post_train_vs_zero"]),
        dict(repeat_comparison["post_train_vs_zero"]),
    )
    train_history_delta = _train_history_delta(
        list(primary["train_history"]),
        list(repeat["train_history"]),
    )
    same_arm_repeat_is_stable = bool(
        train_history_delta["histories_identical"]
        and all(abs(float(v)) <= 1e-6 for v in train_gap_delta.values())
        and all(abs(float(v)) <= 1e-6 for v in post_train_gap_delta.values())
        and abs(float(repeat_comparison["train_full_return_delta"] - primary_comparison["train_full_return_delta"]))
        <= 1e-6
        and abs(
            float(repeat_comparison["train_suffix_return_delta"] - primary_comparison["train_suffix_return_delta"])
        )
        <= 1e-6
    )

    return {
        "audit_entry": "phase3_pair1_baseline_repeat_compare_pack",
        "preflight": {
            "source_path": PAIR1_PREFLIGHT_JSON,
            "preflight_passed": bool(preflight.get("preflight_passed", False)),
            "contract_args": dict(preflight.get("contract_args", {})),
            "allowed_diff": dict(dict(preflight.get("contract_diff", {})).get("allowed_diff", {})),
            "unexpected_diff": dict(dict(preflight.get("contract_diff", {})).get("unexpected_diff", {})),
        },
        "suite_context": primary_ctx,
        "shared_contract_context": _shared_contract_context(primary_ctx),
        "primary": {
            "comparison": primary_comparison,
            "train_history": list(primary["train_history"]),
        },
        "repeat": {
            "comparison": repeat_comparison,
            "train_history": list(repeat["train_history"]),
        },
        "comparison": {
            "pre_train_vs_zero_gap_delta": train_gap_delta,
            "post_train_vs_zero_gap_delta": post_train_gap_delta,
            "train_full_return_delta_change": float(
                repeat_comparison["train_full_return_delta"] - primary_comparison["train_full_return_delta"]
            ),
            "train_suffix_return_delta_change": float(
                repeat_comparison["train_suffix_return_delta"] - primary_comparison["train_suffix_return_delta"]
            ),
            "train_history_delta": train_history_delta,
            "heldout_eval_skipped": bool(
                primary.get("heldout_eval_skipped", True) and repeat.get("heldout_eval_skipped", True)
            ),
        },
        "post_policy_bundles": {
            "primary_requested_path": PAIR1_BASELINE_REPEAT_PRIMARY_POST_POLICY_BUNDLE_PATH,
            "repeat_requested_path": PAIR1_BASELINE_REPEAT_REPEAT_POST_POLICY_BUNDLE_PATH,
            "primary": dict(primary.get("post_policy_bundle", {})),
            "repeat": dict(repeat.get("post_policy_bundle", {})),
        },
        "conclusions": {
            "same_arm_repeat_is_stable": same_arm_repeat_is_stable,
            "pre_metrics_unchanged": bool(all(abs(float(v)) <= 1e-6 for v in train_gap_delta.values())),
            "same_contract_many_env_train_nondeterminism_observed": bool(not same_arm_repeat_is_stable),
        },
        "recommendation": {
            "reason": (
                "Run pair1 baseline twice under the same guarded contract to distinguish same-contract "
                "many-env train nondeterminism from branch-specific mode leakage."
            ),
            "if_stable_then_pair1_override_anomaly_points_to_mode_leakage": True,
            "if_unstable_then_pair1_override_anomaly_cannot_be_attributed_to_mode_alone": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pair1 same-arm baseline repeat compare pack under the guarded Phase 2 contract."
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="/home/chen/RLPFN/artifacts/phase3_pair1_baseline_repeat_compare_pack.json",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output_json).expanduser().resolve()
    if output_path.exists() and not bool(args.overwrite):
        print(output_path.read_text())
        return 0

    report = build_pair1_baseline_repeat_compare_pack()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
