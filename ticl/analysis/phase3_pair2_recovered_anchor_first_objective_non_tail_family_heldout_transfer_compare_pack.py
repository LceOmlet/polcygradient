import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import torch

from ticl.analysis.phase2_guardrail import DEFAULT_PHASE2_SUMMARY
from ticl.analysis.phase3_multi_env_optimization_audit import run_phase3_multi_env_optimization_audit
from ticl.analysis.phase3_pair2_recovered_anchor_first_objective_non_tail_family_train_update_ab_compare_pack import (
    PAIR2_RECOVERED_ANCHOR_NON_TAIL_BASELINE_POST_POLICY_BUNDLE_PATH,
    PAIR2_RECOVERED_ANCHOR_NON_TAIL_BRANCH_SPECIFIC_MODE,
    PAIR2_RECOVERED_ANCHOR_NON_TAIL_OVERRIDE_POST_POLICY_BUNDLE_PATH,
)
from ticl.analysis.phase3_pair2_train_update_ab_compare_pack import (
    PAIR2_CHECKPOINT_PATH,
    PAIR2_HELDOUT_SUITE_PATH,
    PAIR2_PRE_JSON,
    PAIR2_PREFLIGHT_JSON,
    PAIR2_TRAIN_SUITE_PATH,
    PAIR2_TRAIN_UPDATE_EVAL_NSAMPLES,
    PAIR2_TRAIN_UPDATE_MATERIAL_EPS,
    PAIR2_TRAIN_UPDATE_NSAMPLES,
    PAIR2_TRAIN_UPDATE_SINGLE_EVAL_POS,
    PAIR2_ZERO_JSON,
    _build_reused_metrics_payload,
    _gap_delta,
    _has_material_delta,
    _load_pair2_preflight,
    _shared_contract_context,
    _suite_context,
    _train_history_delta,
    _write_reused_metrics_json,
)


PAIR2_RECOVERED_ANCHOR_NON_TAIL_LOCKED_TRAIN_COMPARE_JSON = (
    "/home/chen/RLPFN/artifacts/phase3_pair2_recovered_anchor_first_objective_non_tail_family_train_update_ab_compare_pack.json"
)
PAIR2_RECOVERED_ANCHOR_NON_TAIL_HELDOUT_TRANSFER_BASELINE_RESUME_EVAL_JSON = (
    "/home/chen/RLPFN/artifacts/phase3_pair2_recovered_anchor_first_objective_non_tail_family_heldout_transfer_baseline_resume_eval.json"
)
PAIR2_RECOVERED_ANCHOR_NON_TAIL_HELDOUT_TRANSFER_OVERRIDE_RESUME_EVAL_JSON = (
    "/home/chen/RLPFN/artifacts/phase3_pair2_recovered_anchor_first_objective_non_tail_family_heldout_transfer_override_resume_eval.json"
)


def _persist_json(path: str, payload: dict[str, Any]) -> str:
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return str(output_path)


def _load_locked_train_compare() -> dict[str, Any]:
    path = Path(PAIR2_RECOVERED_ANCHOR_NON_TAIL_LOCKED_TRAIN_COMPARE_JSON).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        str(payload.get("audit_entry", ""))
        != "phase3_pair2_recovered_anchor_first_objective_non_tail_family_train_update_ab_compare_pack"
    ):
        raise RuntimeError(f"Expected locked recovered-anchor non-tail train compare artifact at {path}")
    conclusions = dict(payload.get("conclusions", {}))
    if not bool(conclusions.get("target_side_compare_is_efficacy_eligible", False)):
        raise RuntimeError(f"Locked recovered-anchor non-tail train compare is not efficacy-eligible: {path}")
    if not bool(conclusions.get("target_side_train_update_changes_trajectory", False)):
        raise RuntimeError(
            f"Locked recovered-anchor non-tail train compare does not show train-side trajectory change: {path}"
        )
    if bool(payload.get("comparison", {}).get("heldout_eval_skipped", False)) is not True:
        raise RuntimeError(
            "Locked recovered-anchor non-tail train compare should remain train-side only; "
            f"heldout_eval_skipped was not true in {path}"
        )
    return payload


def _load_bundle_contract(bundle_path: str) -> dict[str, Any]:
    payload = torch.load(str(Path(bundle_path).expanduser().resolve()), map_location="cpu")
    contract = dict(payload.get("contract", {}))
    if not contract:
        raise RuntimeError(f"Bundle at {bundle_path} has no contract payload")
    return contract


def _assert_bundle_contract_matches_expected(
    contract: dict[str, Any], *, expected_actor_objective_mode: str, arm_name: str
) -> None:
    required = {
        "checkpoint_path": str(Path(PAIR2_CHECKPOINT_PATH).expanduser().resolve()),
        "n_samples": int(PAIR2_TRAIN_UPDATE_NSAMPLES),
        "single_eval_pos": int(PAIR2_TRAIN_UPDATE_SINGLE_EVAL_POS),
        "rollout_backend": "serial",
        "train_profile": "phase2_shared_backbone_contract",
        "batch_size": 256,
        "n_epochs": 1,
        "learning_rate": 2e-4,
        "target_kl": 0.03,
        "actor_objective_mode": str(expected_actor_objective_mode),
        "ppo_actor_baseline_mode": "learned",
        "ppo_normalize_advantage": False,
        "ppo_actor_gae_space": "normalized",
        "ppo_vf_coef": 0.5,
        "ppo_reset_env_state_at_sep": True,
        "ppo_separate_value_backbone": False,
        "ppo_restore_validation_policy_state": False,
        "ppo_restore_validation_policy_head_state": True,
        "strict_native_rollout": True,
    }
    for key, expected in required.items():
        observed = contract.get(key, None)
        if observed != expected:
            raise RuntimeError(
                f"Pair2 recovered-anchor non-tail {arm_name} bundle contract mismatch for {key}: "
                f"observed={observed!r}, expected={expected!r}"
            )


def _load_reuse_contract_summary(path: str) -> dict[str, Any]:
    payload = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    config = dict(payload.get("config", {}))
    return {
        "path": str(Path(path).expanduser().resolve()),
        "n_samples": int(config.get("n_samples", -1)),
        "single_eval_pos": int(config.get("single_eval_pos", -1)),
        "rollout_backend": str(config.get("rollout_backend", "")),
        "reuse_eligible_for_current_guarded_contract": bool(
            int(config.get("n_samples", -1)) == int(PAIR2_TRAIN_UPDATE_EVAL_NSAMPLES)
            and int(config.get("single_eval_pos", -1)) == int(PAIR2_TRAIN_UPDATE_SINGLE_EVAL_POS)
            and str(config.get("rollout_backend", "")) == "serial"
        ),
    }


def _assert_close_scalar(name: str, actual: float, expected: float, *, atol: float = 1e-6) -> float:
    diff = abs(float(actual) - float(expected))
    if diff > float(atol):
        raise RuntimeError(
            f"{name} drifted beyond tolerance: actual={float(actual)!r}, expected={float(expected)!r}, diff={diff!r}"
        )
    return float(diff)


def _assert_close_dict(
    name: str, actual: dict[str, Any], expected: dict[str, Any], *, atol: float = 1e-6
) -> float:
    if set(actual) != set(expected):
        raise RuntimeError(
            f"{name} keys drifted: actual={sorted(actual)}, expected={sorted(expected)}"
        )
    max_abs_diff = 0.0
    for key in sorted(actual):
        max_abs_diff = max(
            max_abs_diff,
            _assert_close_scalar(f"{name}.{key}", float(actual[key]), float(expected[key]), atol=atol),
        )
    return float(max_abs_diff)


def _assert_train_side_reproduction(
    *,
    locked_report: dict[str, Any],
    baseline_report: dict[str, Any],
    override_report: dict[str, Any],
) -> dict[str, Any]:
    locked_baseline = dict(dict(locked_report["baseline"])["comparison"])
    locked_override = dict(dict(locked_report["override"])["comparison"])
    baseline_comparison = dict(baseline_report["comparison"])
    override_comparison = dict(override_report["comparison"])

    train_side_max_abs_diff = 0.0
    train_side_max_abs_diff = max(
        train_side_max_abs_diff,
        _assert_close_dict(
            "baseline.pre_train_vs_zero",
            dict(baseline_comparison["pre_train_vs_zero"]),
            dict(locked_baseline["pre_train_vs_zero"]),
        ),
        _assert_close_dict(
            "baseline.post_train_vs_zero",
            dict(baseline_comparison["post_train_vs_zero"]),
            dict(locked_baseline["post_train_vs_zero"]),
        ),
        _assert_close_scalar(
            "baseline.train_full_return_delta",
            float(baseline_comparison["train_full_return_delta"]),
            float(locked_baseline["train_full_return_delta"]),
        ),
        _assert_close_scalar(
            "baseline.train_suffix_return_delta",
            float(baseline_comparison["train_suffix_return_delta"]),
            float(locked_baseline["train_suffix_return_delta"]),
        ),
        _assert_close_dict(
            "override.pre_train_vs_zero",
            dict(override_comparison["pre_train_vs_zero"]),
            dict(locked_override["pre_train_vs_zero"]),
        ),
        _assert_close_dict(
            "override.post_train_vs_zero",
            dict(override_comparison["post_train_vs_zero"]),
            dict(locked_override["post_train_vs_zero"]),
        ),
        _assert_close_scalar(
            "override.train_full_return_delta",
            float(override_comparison["train_full_return_delta"]),
            float(locked_override["train_full_return_delta"]),
        ),
        _assert_close_scalar(
            "override.train_suffix_return_delta",
            float(override_comparison["train_suffix_return_delta"]),
            float(locked_override["train_suffix_return_delta"]),
        ),
    )

    locked_baseline_history = list(dict(locked_report["baseline"])["train_history"])
    locked_override_history = list(dict(locked_report["override"])["train_history"])
    baseline_history = list(baseline_report["train_history"])
    override_history = list(override_report["train_history"])
    if baseline_history != locked_baseline_history:
        raise RuntimeError("Baseline resumed train_history drifted from locked train compare artifact.")
    if override_history != locked_override_history:
        raise RuntimeError("Override resumed train_history drifted from locked train compare artifact.")

    return {
        "max_abs_metric_diff": float(train_side_max_abs_diff),
        "baseline_history_matches_locked": True,
        "override_history_matches_locked": True,
    }


def _run_resumed_eval_audit(
    *,
    actor_objective_mode_override: str | None,
    resume_post_policy_bundle_path: str,
    reuse_zero_control_json: str | None,
    reuse_pre_policy_json: str | None,
    output_json_path: str,
) -> dict[str, Any]:
    mode_label = (
        "baseline_resume"
        if actor_objective_mode_override is None
        else str(actor_objective_mode_override)
    )
    print(f"[pair2-recovered-anchor-nontail-heldout] start audit mode={mode_label}", flush=True)
    report = run_phase3_multi_env_optimization_audit(
        checkpoint_path=PAIR2_CHECKPOINT_PATH,
        train_suite_path=PAIR2_TRAIN_SUITE_PATH,
        heldout_suite_path=PAIR2_HELDOUT_SUITE_PATH,
        device="cuda:0",
        n_samples=PAIR2_TRAIN_UPDATE_NSAMPLES,
        single_eval_pos=PAIR2_TRAIN_UPDATE_SINGLE_EVAL_POS,
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
        strict_fixed_env_mode=True,
        deterministic_actor_sampling=True,
        deterministic_batch_plan=True,
        strict_native_rollout=True,
        restore_validation_policy_head_state=True,
        actor_objective_mode_override=actor_objective_mode_override,
        actor_objective_runtime_current_suite_name=(
            "pair2" if actor_objective_mode_override is not None else None
        ),
        skip_heldout_eval=False,
        eval_n_samples=int(PAIR2_TRAIN_UPDATE_EVAL_NSAMPLES),
        resume_post_policy_bundle_path=str(resume_post_policy_bundle_path),
    )
    resolved_path = _persist_json(output_json_path, report)
    print(
        f"[pair2-recovered-anchor-nontail-heldout] done audit mode={mode_label}; wrote {resolved_path}",
        flush=True,
    )
    return report


def build_pair2_recovered_anchor_first_objective_non_tail_family_heldout_transfer_compare_pack() -> dict[str, Any]:
    preflight = _load_pair2_preflight()
    locked_train_compare = _load_locked_train_compare()

    baseline_bundle_contract = _load_bundle_contract(
        PAIR2_RECOVERED_ANCHOR_NON_TAIL_BASELINE_POST_POLICY_BUNDLE_PATH
    )
    override_bundle_contract = _load_bundle_contract(
        PAIR2_RECOVERED_ANCHOR_NON_TAIL_OVERRIDE_POST_POLICY_BUNDLE_PATH
    )
    _assert_bundle_contract_matches_expected(
        baseline_bundle_contract,
        expected_actor_objective_mode="tokenwise",
        arm_name="baseline",
    )
    _assert_bundle_contract_matches_expected(
        override_bundle_contract,
        expected_actor_objective_mode=PAIR2_RECOVERED_ANCHOR_NON_TAIL_BRANCH_SPECIFIC_MODE,
        arm_name="override",
    )

    canonical_zero_contract = _load_reuse_contract_summary(PAIR2_ZERO_JSON)
    canonical_pre_contract = _load_reuse_contract_summary(PAIR2_PRE_JSON)

    print("[pair2-recovered-anchor-nontail-heldout] start baseline resumed eval", flush=True)
    baseline = _run_resumed_eval_audit(
        actor_objective_mode_override=None,
        resume_post_policy_bundle_path=PAIR2_RECOVERED_ANCHOR_NON_TAIL_BASELINE_POST_POLICY_BUNDLE_PATH,
        reuse_zero_control_json=None,
        reuse_pre_policy_json=None,
        output_json_path=PAIR2_RECOVERED_ANCHOR_NON_TAIL_HELDOUT_TRANSFER_BASELINE_RESUME_EVAL_JSON,
    )

    print("[pair2-recovered-anchor-nontail-heldout] start override resumed eval", flush=True)
    with tempfile.TemporaryDirectory(prefix="pair2_recovered_anchor_nontail_heldout_") as tmpdir:
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
            heldout_metrics=baseline["pre"]["heldout"],
        )
        reuse_zero_control_json = _write_reused_metrics_json(tmpdir_path / "zero_control.json", zero_payload)
        reuse_pre_policy_json = _write_reused_metrics_json(tmpdir_path / "pre_policy.json", pre_payload)
        override = _run_resumed_eval_audit(
            actor_objective_mode_override=PAIR2_RECOVERED_ANCHOR_NON_TAIL_BRANCH_SPECIFIC_MODE,
            resume_post_policy_bundle_path=PAIR2_RECOVERED_ANCHOR_NON_TAIL_OVERRIDE_POST_POLICY_BUNDLE_PATH,
            reuse_zero_control_json=reuse_zero_control_json,
            reuse_pre_policy_json=reuse_pre_policy_json,
            output_json_path=PAIR2_RECOVERED_ANCHOR_NON_TAIL_HELDOUT_TRANSFER_OVERRIDE_RESUME_EVAL_JSON,
        )

    baseline_ctx = _suite_context(baseline)
    override_ctx = _suite_context(override)
    if _shared_contract_context(baseline_ctx) != _shared_contract_context(override_ctx):
        raise ValueError(
            "Baseline and override recovered-anchor non-tail heldout-transfer audits must share the same suite contract, "
            f"got {baseline_ctx!r} vs {override_ctx!r}"
        )
    if baseline_ctx["actor_objective_runtime_current_suite_name"] not in {"", "none"}:
        raise ValueError(
            "Baseline heldout-transfer audit should not activate a runtime-scoped objective mode: "
            f"{baseline_ctx['actor_objective_runtime_current_suite_name']!r}"
        )
    if baseline_ctx["actor_objective_mode_override"] is not None:
        raise ValueError(
            f"Baseline heldout-transfer audit should use tokenwise objective without override: {baseline_ctx!r}"
        )
    if (
        str(override_ctx["actor_objective_mode_override"])
        != PAIR2_RECOVERED_ANCHOR_NON_TAIL_BRANCH_SPECIFIC_MODE
    ):
        raise ValueError(
            "Override heldout-transfer audit did not activate expected branch-specific mode: "
            f"{override_ctx['actor_objective_mode_override']!r}"
        )
    if override_ctx["actor_objective_runtime_current_suite_name"] != "pair2":
        raise ValueError(
            "Override heldout-transfer audit did not target pair2 runtime scope: "
            f"{override_ctx['actor_objective_runtime_current_suite_name']!r}"
        )

    reproduction = _assert_train_side_reproduction(
        locked_report=locked_train_compare,
        baseline_report=baseline,
        override_report=override,
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
    pre_heldout_gap_delta = _gap_delta(
        dict(baseline_comparison["pre_heldout_vs_zero"]),
        dict(override_comparison["pre_heldout_vs_zero"]),
    )
    post_heldout_gap_delta = _gap_delta(
        dict(baseline_comparison["post_heldout_vs_zero"]),
        dict(override_comparison["post_heldout_vs_zero"]),
    )
    train_history_delta = _train_history_delta(
        list(baseline["train_history"]),
        list(override["train_history"]),
    )

    train_full_return_delta_change = float(
        override_comparison["train_full_return_delta"] - baseline_comparison["train_full_return_delta"]
    )
    train_suffix_return_delta_change = float(
        override_comparison["train_suffix_return_delta"] - baseline_comparison["train_suffix_return_delta"]
    )
    heldout_full_return_delta_change = float(
        override_comparison["heldout_full_return_delta"] - baseline_comparison["heldout_full_return_delta"]
    )
    heldout_suffix_return_delta_change = float(
        override_comparison["heldout_suffix_return_delta"] - baseline_comparison["heldout_suffix_return_delta"]
    )

    return {
        "audit_entry": "phase3_pair2_recovered_anchor_first_objective_non_tail_family_heldout_transfer_compare_pack",
        "preflight": {
            "source_path": PAIR2_PREFLIGHT_JSON,
            "preflight_passed": bool(preflight.get("preflight_passed", False)),
            "contract_args": dict(preflight.get("contract_args", {})),
            "allowed_diff": dict(dict(preflight.get("contract_diff", {})).get("allowed_diff", {})),
            "unexpected_diff": dict(dict(preflight.get("contract_diff", {})).get("unexpected_diff", {})),
        },
        "locked_train_side_source": {
            "source_path": PAIR2_RECOVERED_ANCHOR_NON_TAIL_LOCKED_TRAIN_COMPARE_JSON,
            "train_suffix_return_delta_change": float(
                locked_train_compare["comparison"]["train_suffix_return_delta_change"]
            ),
            "train_full_return_delta_change": float(
                locked_train_compare["comparison"]["train_full_return_delta_change"]
            ),
        },
        "bundle_contracts": {
            "baseline_bundle_path": PAIR2_RECOVERED_ANCHOR_NON_TAIL_BASELINE_POST_POLICY_BUNDLE_PATH,
            "override_bundle_path": PAIR2_RECOVERED_ANCHOR_NON_TAIL_OVERRIDE_POST_POLICY_BUNDLE_PATH,
            "baseline": baseline_bundle_contract,
            "override": override_bundle_contract,
        },
        "reuse_guard": {
            "canonical_pair2_zero": canonical_zero_contract,
            "canonical_pair2_pre": canonical_pre_contract,
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
            "runtime_mode_activated": bool(
                override_ctx["actor_objective_runtime_current_suite_name"] == "pair2"
                and override_ctx["actor_objective_mode_override"]
                == PAIR2_RECOVERED_ANCHOR_NON_TAIL_BRANCH_SPECIFIC_MODE
            ),
        },
        "baseline_eval_report": {
            "path": PAIR2_RECOVERED_ANCHOR_NON_TAIL_HELDOUT_TRANSFER_BASELINE_RESUME_EVAL_JSON,
            "comparison": baseline_comparison,
            "train_history": list(baseline["train_history"]),
        },
        "override_eval_report": {
            "path": PAIR2_RECOVERED_ANCHOR_NON_TAIL_HELDOUT_TRANSFER_OVERRIDE_RESUME_EVAL_JSON,
            "comparison": override_comparison,
            "train_history": list(override["train_history"]),
        },
        "comparison": {
            "pre_train_vs_zero_gap_delta": train_gap_delta,
            "post_train_vs_zero_gap_delta": post_train_gap_delta,
            "pre_heldout_vs_zero_gap_delta": pre_heldout_gap_delta,
            "post_heldout_vs_zero_gap_delta": post_heldout_gap_delta,
            "train_full_return_delta_change": train_full_return_delta_change,
            "train_suffix_return_delta_change": train_suffix_return_delta_change,
            "heldout_full_return_delta_change": heldout_full_return_delta_change,
            "heldout_suffix_return_delta_change": heldout_suffix_return_delta_change,
            "heldout_full_over_train_full_ratio": None
            if abs(train_full_return_delta_change) <= 1e-12
            else float(heldout_full_return_delta_change / train_full_return_delta_change),
            "heldout_suffix_over_train_suffix_ratio": None
            if abs(train_suffix_return_delta_change) <= 1e-12
            else float(heldout_suffix_return_delta_change / train_suffix_return_delta_change),
            "train_history_delta": train_history_delta,
            "train_side_reproduction": reproduction,
        },
        "conclusions": {
            "train_side_reproduction_matches_locked_compare": True,
            "pre_metrics_unchanged": bool(
                all(abs(float(v)) <= 1e-6 for v in train_gap_delta.values())
                and all(abs(float(v)) <= 1e-6 for v in pre_heldout_gap_delta.values())
            ),
            "heldout_transfer_compare_is_efficacy_eligible": True,
            "heldout_transfer_material": bool(
                _has_material_delta(post_heldout_gap_delta)
                or abs(float(heldout_full_return_delta_change)) > float(PAIR2_TRAIN_UPDATE_MATERIAL_EPS)
                or abs(float(heldout_suffix_return_delta_change)) > float(PAIR2_TRAIN_UPDATE_MATERIAL_EPS)
            ),
            "heldout_transfer_is_only_numeric_noise": bool(
                train_history_delta["histories_identical"]
                and all(abs(float(v)) <= 1e-6 for v in pre_heldout_gap_delta.values())
                and not _has_material_delta(post_heldout_gap_delta)
                and abs(float(heldout_full_return_delta_change)) <= float(PAIR2_TRAIN_UPDATE_MATERIAL_EPS)
                and abs(float(heldout_suffix_return_delta_change)) <= float(PAIR2_TRAIN_UPDATE_MATERIAL_EPS)
            ),
            "heldout_suffix_same_sign_as_train_suffix": bool(
                train_suffix_return_delta_change == 0.0
                or heldout_suffix_return_delta_change == 0.0
                or (
                    train_suffix_return_delta_change > 0.0
                    and heldout_suffix_return_delta_change > 0.0
                )
                or (
                    train_suffix_return_delta_change < 0.0
                    and heldout_suffix_return_delta_change < 0.0
                )
            ),
            "heldout_full_same_sign_as_train_full": bool(
                train_full_return_delta_change == 0.0
                or heldout_full_return_delta_change == 0.0
                or (
                    train_full_return_delta_change > 0.0
                    and heldout_full_return_delta_change > 0.0
                )
                or (
                    train_full_return_delta_change < 0.0
                    and heldout_full_return_delta_change < 0.0
                )
            ),
            "heldout_effect_smaller_than_train_effect": bool(
                abs(float(heldout_full_return_delta_change)) < abs(float(train_full_return_delta_change))
                and abs(float(heldout_suffix_return_delta_change)) < abs(float(train_suffix_return_delta_change))
            ),
            "canonical_pair2_cross_env_baselines_reuse_ineligible": bool(
                not canonical_zero_contract["reuse_eligible_for_current_guarded_contract"]
                and not canonical_pre_contract["reuse_eligible_for_current_guarded_contract"]
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Heldout-transfer compare for the pair2 recovered-anchor first-objective non-tail branch-specific family."
        )
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=(
            "/home/chen/RLPFN/artifacts/"
            "phase3_pair2_recovered_anchor_first_objective_non_tail_family_heldout_transfer_compare_pack.json"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output_json).expanduser().resolve()
    if output_path.exists() and not bool(args.overwrite):
        print(output_path.read_text(encoding="utf-8"))
        return 0

    report = build_pair2_recovered_anchor_first_objective_non_tail_family_heldout_transfer_compare_pack()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(output_path.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
