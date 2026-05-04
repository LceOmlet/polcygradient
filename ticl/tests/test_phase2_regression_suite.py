import json
from argparse import Namespace
from pathlib import Path

from ticl.analysis.phase2_regression_suite import (
    _load_existing_summary_if_reusable,
    _load_manifest,
    _milestone_summary,
    _python_executable,
    _summary_matches_invocation,
    _validate_against_manifest,
)
from ticl.analysis.phase2_guardrail import (
    TRUSTED_PHASE2_LAUNCH_SUMMARY,
    assert_phase2_launch_chain_green,
    phase2_launch_chain_failures,
)


CANONICAL_REGRESSION = "/home/chen/RLPFN/artifacts/critic_official_vs_monkey_regression_seed12345.json"
CANONICAL_MILESTONE = (
    "/home/chen/RLPFN/artifacts/"
    "phase2_launch_anchor_shared_seed12345_current_contract_quick.json"
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
        build_seed=4040,
        reuse_canonical_official_monkey=True,
        reuse_canonical_shared_milestone=False,
        manifest_path=CANONICAL_MANIFEST,
    )


def _canonical_reusable_summary() -> dict:
    args = _default_args()
    return {
        "trusted_contract": {
            "checkpoint_path": str(Path(args.checkpoint_path).expanduser().resolve()),
            "device": args.device,
            "frozen_h_seed": args.frozen_h_seed,
            "train_env_seed": args.train_env_seed,
            "train_rollout_seed": args.train_rollout_seed,
            "single_eval_pos": args.single_eval_pos,
            "n_steps": args.n_steps,
            "batch_size": args.batch_size,
            "n_epochs": args.n_epochs,
            "outer_epochs": args.outer_epochs,
            "learning_rate": args.learning_rate,
            "target_kl": args.target_kl,
            "build_seed": args.build_seed,
            "ppo_reset_env_state_at_sep": True,
            "strict_fixed_env_mode": True,
            "deterministic_actor_sampling": True,
            "deterministic_batch_plan": True,
            "strict_native_rollout": False,
            "restore_validation_policy_state": False,
            "shared_actor_critic_backbone": True,
            "record_suite_critic_quality": True,
            "frozen_h_seed_mode": "current",
            "zero_eval_prior_reuses_sampling_state": False,
        },
        "assembly_mode": {
            "reuse_canonical_official_monkey": args.reuse_canonical_official_monkey,
            "reuse_canonical_shared_milestone": args.reuse_canonical_shared_milestone,
        },
        "manifest_path": str(Path(args.manifest_path).expanduser().resolve()),
        "validation": {
            "all_checks_pass": True,
        },
    }


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
    summary = _canonical_reusable_summary()
    assert _summary_matches_invocation(summary=summary, args=args) is True


def test_phase2_existing_summary_reuse_respects_invocation_mismatch(tmp_path):
    args = _default_args()
    summary = _canonical_reusable_summary()
    summary_path = tmp_path / "phase2_regression_suite_summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    reused = _load_existing_summary_if_reusable(summary_path=summary_path, args=args)
    assert reused is not None

    args_bad = _default_args()
    args_bad.train_rollout_seed = 5050
    rejected = _load_existing_summary_if_reusable(summary_path=summary_path, args=args_bad)
    assert rejected is None


def test_phase2_python_executable_prefers_current_interpreter(monkeypatch):
    monkeypatch.delenv("TICL_PHASE_REGRESSION_PYTHON", raising=False)
    assert _python_executable() == str(Path(__import__("sys").executable).resolve())


def test_phase2_launch_chain_guard_accepts_stable_summary():
    summary = assert_phase2_launch_chain_green(TRUSTED_PHASE2_LAUNCH_SUMMARY)
    assert summary["validation"]["all_checks_pass"] is True
    assert summary["trusted_contract"]["strict_native_rollout"] is False
    assert summary["trusted_contract"]["restore_validation_policy_state"] is False


def test_phase2_launch_chain_guard_detects_source_of_truth_drift():
    summary = assert_phase2_launch_chain_green(TRUSTED_PHASE2_LAUNCH_SUMMARY)
    summary["source_of_truth"]["phase2_launch_anchor_artifact"] = "/tmp/bad.json"
    failures = phase2_launch_chain_failures(summary)
    assert "source_of_truth.phase2_launch_anchor_artifact drift" in failures
