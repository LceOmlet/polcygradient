import json
from argparse import Namespace
from pathlib import Path

from ticl.analysis.phase2_regression_suite import (
    _load_existing_summary_if_reusable,
    _load_manifest,
    _milestone_summary,
    _summary_matches_invocation,
    _validate_against_manifest,
)


CANONICAL_REGRESSION = "/home/chen/RLPFN/artifacts/critic_official_vs_monkey_regression_seed12345.json"
CANONICAL_MILESTONE = (
    "/home/chen/RLPFN/artifacts/"
    "critic_help_vs_zero_state_reset_shared_seed12345_strict_fixed_env_deterministic_quick.json"
)
CANONICAL_MANIFEST = (
    "/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase2_regression_manifest.json"
)


def _default_args() -> Namespace:
    return Namespace(
        checkpoint_path="/home/chen/RLPFN/rwkv/models_diff/rlpfn_03_30_2026_22_37_53_epoch_13.cpkt",
        device="cuda:0",
        frozen_h_seed=12345,
        train_env_seed=2020,
        train_rollout_seed=4040,
        single_eval_pos=64,
        n_steps=256,
        batch_size=256,
        n_epochs=1,
        outer_epochs=2,
        learning_rate=2e-4,
        target_kl=0.03,
        reuse_canonical_official_monkey=True,
        reuse_canonical_shared_milestone=True,
        manifest_path=CANONICAL_MANIFEST,
    )


def test_phase2_manifest_validation_passes_on_canonical_artifacts():
    regression = json.loads(Path(CANONICAL_REGRESSION).read_text(encoding="utf-8"))
    audit = json.loads(Path(CANONICAL_MILESTONE).read_text(encoding="utf-8"))
    milestone = _milestone_summary(audit)
    manifest = _load_manifest(CANONICAL_MANIFEST)

    validation = _validate_against_manifest(
        regression=regression,
        milestone=milestone,
        manifest=manifest,
        manifest_path=CANONICAL_MANIFEST,
    )

    assert validation["all_checks_pass"] is True
    assert validation["failures"] == []


def test_phase2_summary_matches_invocation_for_canonical_fast_pack():
    args = _default_args()
    summary = json.loads(
        Path("/home/chen/RLPFN/artifacts/phase2_regression_pack/phase2_regression_suite_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert _summary_matches_invocation(summary=summary, args=args) is True


def test_phase2_existing_summary_reuse_respects_invocation_mismatch(tmp_path):
    args = _default_args()
    summary = json.loads(
        Path("/home/chen/RLPFN/artifacts/phase2_regression_pack/phase2_regression_suite_summary.json").read_text(
            encoding="utf-8"
        )
    )
    summary_path = tmp_path / "phase2_regression_suite_summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    reused = _load_existing_summary_if_reusable(summary_path=summary_path, args=args)
    assert reused is not None

    args_bad = _default_args()
    args_bad.train_rollout_seed = 5050
    rejected = _load_existing_summary_if_reusable(summary_path=summary_path, args=args_bad)
    assert rejected is None
