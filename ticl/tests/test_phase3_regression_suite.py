import json
from argparse import Namespace
from pathlib import Path

from ticl.analysis.phase3_regression_suite import (
    _load_existing_summary_if_reusable,
    _load_json,
    _source_artifact_fingerprints,
    _summary_matches_invocation,
    _validate,
)


CANONICAL_MANIFEST = (
    "/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_regression_manifest.json"
)
PHASE2_LAUNCH_ANCHOR = (
    "/home/chen/RLPFN/artifacts/"
    "phase2_launch_anchor_shared_seed12345_current_contract_quick.json"
)
PHASE2_TO_PHASE3_COMPARE = (
    "/home/chen/RLPFN/artifacts/"
    "phase3_phase2_milestone_contract_gradient_compare_2026_04_17_rerun.json"
)
PHASE3_ANCHOR_GRADIENT = "/home/chen/RLPFN/artifacts/phase3_longrun_gradient_compare.json"
FIT_MODEL_ALIGNMENT = "/home/chen/RLPFN/artifacts/fit_model_phase3_anchor_alignment_probe.json"


def _default_args() -> Namespace:
    return Namespace(
        manifest_path=CANONICAL_MANIFEST,
    )


def _current_inputs() -> tuple[dict, dict, dict, dict, dict]:
    return (
        _load_json(CANONICAL_MANIFEST),
        _load_json(PHASE2_LAUNCH_ANCHOR),
        _load_json(PHASE2_TO_PHASE3_COMPARE),
        _load_json(PHASE3_ANCHOR_GRADIENT),
        _load_json(FIT_MODEL_ALIGNMENT),
    )


def test_phase3_unique_anchor_manifest_validation_passes_on_current_artifacts():
    manifest, phase2_launch, phase2_to_phase3, phase3_anchor, fit_model = _current_inputs()

    validation = _validate(
        manifest=manifest,
        phase2_launch_anchor=phase2_launch,
        phase2_to_phase3_compare=phase2_to_phase3,
        phase3_anchor_gradient_compare=phase3_anchor,
        fit_model_alignment=fit_model,
        manifest_path=CANONICAL_MANIFEST,
    )

    assert validation["all_checks_pass"] is True
    assert validation["failures"] == []
    assert validation["validated_sections"] == [
        "phase2_launch_anchor",
        "phase2_to_phase3_anchor_compare",
        "phase3_anchor_gradient_compare",
        "fit_model_anchor_alignment",
    ]


def test_phase3_summary_matches_invocation_for_matching_synthetic_summary():
    summary = {
        "manifest_path": str(Path(CANONICAL_MANIFEST).resolve()),
        "validation": {"all_checks_pass": True},
        "source_artifact_fingerprints": _source_artifact_fingerprints(),
    }
    assert _summary_matches_invocation(summary=summary, args=_default_args()) is True


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


def test_phase3_validation_detects_phase2_launch_anchor_drift():
    manifest, phase2_launch, phase2_to_phase3, phase3_anchor, fit_model = _current_inputs()
    phase2_launch["modes"]["learned"]["delta_smp_gap"] -= 1.0

    validation = _validate(
        manifest=manifest,
        phase2_launch_anchor=phase2_launch,
        phase2_to_phase3_compare=phase2_to_phase3,
        phase3_anchor_gradient_compare=phase3_anchor,
        fit_model_alignment=fit_model,
        manifest_path=CANONICAL_MANIFEST,
    )

    assert "phase2_launch_anchor.learned.delta_smp_gap drift" in validation["failures"]


def test_phase3_validation_detects_unexpected_fit_model_signature_diff():
    manifest, phase2_launch, phase2_to_phase3, phase3_anchor, fit_model = _current_inputs()
    fit_model["training_signature"]["diff"]["unexpected.key"] = {"left": 1, "right": 2}

    validation = _validate(
        manifest=manifest,
        phase2_launch_anchor=phase2_launch,
        phase2_to_phase3_compare=phase2_to_phase3,
        phase3_anchor_gradient_compare=phase3_anchor,
        fit_model_alignment=fit_model,
        manifest_path=CANONICAL_MANIFEST,
    )

    assert (
        "fit_model_anchor_alignment.training_signature unexpected diff keys"
        in validation["failures"]
    )
