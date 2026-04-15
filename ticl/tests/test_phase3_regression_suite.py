import json
from argparse import Namespace
from pathlib import Path

from ticl.analysis.phase2_guardrail import phase2_green_failures
from ticl.analysis.phase3_regression_suite import (
    _load_existing_summary_if_reusable,
    _pair_gap_summary,
    _source_artifact_fingerprints,
    _summary_matches_invocation,
    _validate,
)


CANONICAL_MANIFEST = (
    "/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_regression_manifest.json"
)


def _load(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _default_args() -> Namespace:
    return Namespace(
        manifest_path=CANONICAL_MANIFEST,
    )


def _build_manifest_from_current_artifacts() -> dict:
    longrun = _load("/home/chen/RLPFN/artifacts/phase3_longrun_gradient_compare.json")
    short = _load("/home/chen/RLPFN/artifacts/phase3_shortrun_profile_gradient_regression_v2.json")
    trusted = _load("/home/chen/RLPFN/artifacts/phase3_multi_env_trusted_pair1.json")
    pair1_gaps = _pair_gap_summary(
        ppo_report=_load("/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/ppo_cross_env_baseline.json"),
        zero_report=_load("/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/zero_cross_env_control.json"),
    )
    pair2_gaps = _pair_gap_summary(
        ppo_report=_load("/home/chen/RLPFN/artifacts/phase3_cross_env_baseline_pair2/ppo_cross_env_baseline.json"),
        zero_report=_load("/home/chen/RLPFN/artifacts/phase3_cross_env_baseline_pair2/zero_cross_env_control.json"),
    )
    return {
        "trusted_longrun_gradient_compare": {
            "exact_match_required": True,
            "cosine_similarity_min": 0.9999,
            "l2_delta_norm_max": float(longrun["grad_compare"]["l2_delta_norm"]) + 1e-3,
            "resume_grad_norm_expected": float(longrun["grad_compare"]["resume_grad_norm"]),
            "phase3_isolated_grad_norm_expected": float(longrun["grad_compare"]["phase3_isolated_grad_norm"]),
            "collect_wall_s_max": max(
                float(longrun["modes"]["phase3_isolated_semantics"]["collect_wall_s"]),
                float(longrun["modes"]["resume_checkpoint_semantics"]["collect_wall_s"]),
            ) + 1.0,
            "backward_wall_s_max": max(
                float(longrun["modes"]["phase3_isolated_semantics"]["backward_wall_s"]),
                float(longrun["modes"]["resume_checkpoint_semantics"]["backward_wall_s"]),
            ) + 1.0,
            "collect_ratio_max": 2.0,
            "backward_ratio_max": 2.0,
            "abs_tol": 1e-6,
        },
        "legacy_shortrun_profile_gradient_compare": {
            "exact_match_required": True,
            "cosine_similarity_min": 0.9999,
            "l2_delta_norm_max": float(short["grad_compare"]["l2_delta_norm"]) + 1e-3,
            "driver_grad_norm_expected": float(short["grad_compare"]["driver_grad_norm"]),
            "reference_grad_norm_expected": float(short["grad_compare"]["reference_grad_norm"]),
            "collect_wall_s_max": max(
                float(short["modes"]["driver_profile_shortrun_aligned"]["collect_wall_s"]),
                float(short["modes"]["manual_reference_shortrun_aligned"]["collect_wall_s"]),
            ) + 1.0,
            "backward_wall_s_max": max(
                float(short["modes"]["driver_profile_shortrun_aligned"]["backward_wall_s"]),
                float(short["modes"]["manual_reference_shortrun_aligned"]["backward_wall_s"]),
            ) + 1.0,
            "collect_ratio_max": 2.0,
            "backward_ratio_max": 2.0,
            "abs_tol": 1e-6,
        },
        "checkpoint_generalization": {
            "abs_tol": 1e-6,
            "pair1": {
                "train_full_gap": float(pair1_gaps["train_suite"]["full_gap"]),
                "train_suffix_gap": float(pair1_gaps["train_suite"]["suffix_gap"]),
                "heldout_full_gap": float(pair1_gaps["heldout_suite"]["full_gap"]),
                "heldout_suffix_gap": float(pair1_gaps["heldout_suite"]["suffix_gap"]),
            },
            "pair2": {
                "train_full_gap": float(pair2_gaps["train_suite"]["full_gap"]),
                "train_suffix_gap": float(pair2_gaps["train_suite"]["suffix_gap"]),
                "heldout_full_gap": float(pair2_gaps["heldout_suite"]["full_gap"]),
                "heldout_suffix_gap": float(pair2_gaps["heldout_suite"]["suffix_gap"]),
            },
        },
        "trusted_pair1_optimization": {
            "abs_tol": 1e-6,
            "train_full_return_delta": float(trusted["comparison"]["train_full_return_delta"]),
            "train_suffix_return_delta": float(trusted["comparison"]["train_suffix_return_delta"]),
            "heldout_full_return_delta": float(trusted["comparison"]["heldout_full_return_delta"]),
            "heldout_suffix_return_delta": float(trusted["comparison"]["heldout_suffix_return_delta"]),
        },
    }, longrun, short, trusted, pair1_gaps, pair2_gaps


def test_phase3_manifest_validation_passes_on_current_artifacts_with_matching_manifest():
    manifest, longrun, short, trusted, pair1_gaps, pair2_gaps = _build_manifest_from_current_artifacts()

    validation = _validate(
        manifest=manifest,
        longrun_grad=longrun,
        shortrun_profile=short,
        pair1_gaps=pair1_gaps,
        pair2_gaps=pair2_gaps,
        trusted_pair1=trusted,
        manifest_path=CANONICAL_MANIFEST,
    )
    assert validation["all_checks_pass"] is True
    assert validation["failures"] == []


def test_phase3_summary_matches_invocation_for_matching_synthetic_summary():
    summary = {
        "manifest_path": str(Path(CANONICAL_MANIFEST).resolve()),
        "validation": {"all_checks_pass": True},
        "source_artifact_fingerprints": _source_artifact_fingerprints(),
    }
    assert _summary_matches_invocation(summary=summary, args=_default_args()) is True


def test_phase3_existing_summary_reuse_respects_manifest_mismatch(tmp_path):
    summary = {
        "manifest_path": str(Path(CANONICAL_MANIFEST).resolve()),
        "validation": {"all_checks_pass": True},
        "source_artifact_fingerprints": _source_artifact_fingerprints(),
    }
    summary_path = tmp_path / "phase3_regression_suite_summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    reused = _load_existing_summary_if_reusable(summary_path=summary_path, args=_default_args())
    assert reused is not None

    bad_args = Namespace(manifest_path="/tmp/not-the-same-manifest.json")
    rejected = _load_existing_summary_if_reusable(summary_path=summary_path, args=bad_args)
    assert rejected is None


def test_phase3_existing_summary_reuse_rejects_source_artifact_drift(tmp_path):
    summary = {
        "manifest_path": str(Path(CANONICAL_MANIFEST).resolve()),
        "validation": {"all_checks_pass": True},
        "source_artifact_fingerprints": _source_artifact_fingerprints(),
    }
    first_key = next(iter(summary["source_artifact_fingerprints"]))
    summary["source_artifact_fingerprints"][first_key]["sha256"] = "broken"
    summary_path = tmp_path / "phase3_regression_suite_summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    rejected = _load_existing_summary_if_reusable(summary_path=summary_path, args=_default_args())
    assert rejected is None


def test_phase3_validation_can_be_forced_into_diagnostic_only_mode():
    manifest, longrun, short, trusted, pair1_gaps, pair2_gaps = _build_manifest_from_current_artifacts()
    manifest["phase3_claims_trusted"] = False
    manifest["phase3_trust_reason"] = "diagnostic only"

    validation = _validate(
        manifest=manifest,
        longrun_grad=longrun,
        shortrun_profile=short,
        pair1_gaps=pair1_gaps,
        pair2_gaps=pair2_gaps,
        trusted_pair1=trusted,
        manifest_path=CANONICAL_MANIFEST,
    )

    assert validation["all_checks_pass"] is True
    assert validation["diagnostic_only"] is True
    assert validation["phase3_claims_trusted"] is False


def test_phase2_guardrail_accepts_current_phase2_pack():
    summary = _load("/home/chen/RLPFN/artifacts/phase2_regression_pack/phase2_regression_suite_summary.json")
    assert phase2_green_failures(summary) == []
