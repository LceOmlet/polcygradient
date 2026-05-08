import argparse
import os
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = (
    "/home/chen/RLPFN/rwkv/models_diff/rlpfn_03_30_2026_22_37_53_epoch_13.cpkt"
)
DEFAULT_OUTPUT_DIR = "/home/chen/RLPFN/artifacts/phase2_regression_pack"
DEFAULT_LEGACY_GRAD_SNAPSHOT = (
    "/home/chen/RLPFN/artifacts/legacy_frozen_h_seed12345_sep_state_reset_gradient_snapshot.json"
)
DEFAULT_CANONICAL_MILESTONE = (
    "/home/chen/RLPFN/artifacts/"
    "phase2_launch_anchor_shared_seed12345_current_contract_quick.json"
)
DEFAULT_CANONICAL_OFFICIAL_MONKEY = (
    "/home/chen/RLPFN/artifacts/critic_official_vs_monkey_regression_seed12345.json"
)
DEFAULT_MANIFEST = (
    "/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase2_regression_manifest.json"
)
PREFERRED_PYTHON = "/home/liangchen/miniconda3/envs/rlpfn/bin/python"
PYTHON_OVERRIDE_ENV = "TICL_PHASE_REGRESSION_PYTHON"


def _default_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _python_executable() -> str:
    override = os.environ.get(PYTHON_OVERRIDE_ENV, "").strip()
    if override:
        override_path = Path(override).expanduser().resolve()
        if override_path.exists():
            return str(override_path)
    current = Path(sys.executable).resolve()
    if current.exists():
        return str(current)
    preferred = Path(PREFERRED_PYTHON)
    if preferred.exists():
        return str(preferred)
    return str(current)


def _subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    repo_pythonpath = str(REPO_ROOT)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        repo_pythonpath
        if not existing
        else f"{repo_pythonpath}:{existing}"
    )
    python_dir = str(Path(_python_executable()).resolve().parent)
    existing_path = env.get("PATH", "")
    env["PATH"] = python_dir if not existing_path else f"{python_dir}:{existing_path}"
    return env


def _run_json_command(*, cmd: list[str], output_path: Path) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        check=True,
        env=_subprocess_env(),
    )
    return json.loads(output_path.read_text(encoding="utf-8"))


def _reuse_json(*, source_path: str, output_path: Path) -> dict:
    source = Path(source_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, output_path)
    return json.loads(output_path.read_text(encoding="utf-8"))


def _milestone_summary(report: dict) -> dict:
    zero_suffix = float(report["zero_suffix_return_mean"])
    learned = dict(report["modes"]["learned"])
    zero = dict(report["modes"]["zero"])
    learned["post_return"] = float(zero_suffix + float(learned["post_smp_gap"]))
    learned["pre_return"] = float(zero_suffix + float(learned["pre_smp_gap"]))
    zero["post_return"] = float(zero_suffix + float(zero["post_smp_gap"]))
    zero["pre_return"] = float(zero_suffix + float(zero["pre_smp_gap"]))
    return {
        "zero_suffix_return_mean": zero_suffix,
        "frozen_h_seed_mode": report.get("frozen_h_seed_mode"),
        "strict_native_rollout": report.get("strict_native_rollout"),
        "restore_validation_policy_state": report.get("restore_validation_policy_state"),
        "record_suite_critic_quality": report.get("record_suite_critic_quality"),
        "fixed_env_contract": dict(report.get("fixed_env_contract", {})),
        "learned": learned,
        "zero": zero,
    }


def _load_manifest(path: str) -> dict:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def _is_pid_alive(pid: int) -> bool:
    if int(pid) <= 0:
        return False
    try:
        os.kill(int(pid), 0)
    except OSError:
        return False
    return True


def _acquire_lock(lock_path: Path) -> dict:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if lock_path.exists():
        try:
            lock_payload = json.loads(lock_path.read_text(encoding="utf-8"))
        except Exception:
            lock_payload = {}
        existing_pid = int(lock_payload.get("pid", -1))
        if _is_pid_alive(existing_pid):
            raise RuntimeError(
                f"Phase 2 regression pack is already running under pid={existing_pid}: {lock_path}"
            )
        lock_path.unlink(missing_ok=True)
    payload = {
        "pid": int(os.getpid()),
        "created_unix_s": float(time.time()),
        "argv": list(sys.argv),
    }
    lock_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _release_lock(lock_path: Path) -> None:
    lock_path.unlink(missing_ok=True)


def _check_close(*, actual: float, expected: float, abs_tol: float) -> bool:
    return abs(float(actual) - float(expected)) <= float(abs_tol)


def _validate_against_manifest(*, regression: dict, milestone: dict, manifest: dict, manifest_path: str) -> dict:
    failures: list[str] = []
    regression_core = regression.get("regression_compare_after_sync", regression)
    timing = regression.get("timing_compare_after_sync", {})
    first_step = regression.get("first_step_compare_after_sync", {})
    synced = regression.get("synced_param_diff", {})
    unsynced = regression.get("unsynced_param_diff", {})

    reg_spec = manifest["official_vs_monkey"]
    for key in reg_spec["exact_zero_keys"]:
        if float(regression_core.get(key, 1.0)) != 0.0:
            failures.append(f"official_vs_monkey.{key} != 0")
    if float(regression_core.get("grad_cosine", 0.0)) < float(reg_spec["grad_cosine_min"]):
        failures.append("official_vs_monkey.grad_cosine below threshold")
    timing_ratio = float(timing.get("official_over_monkey_ratio", float("inf")))
    if timing_ratio < float(reg_spec["timing_ratio_min"]):
        failures.append("official_vs_monkey.official_over_monkey_ratio too low")
    if timing_ratio > float(reg_spec["timing_ratio_max"]):
        failures.append("official_vs_monkey.official_over_monkey_ratio too high")
    if float(first_step.get("action_mean_max_abs_diff", float("inf"))) > float(
        reg_spec["first_step_action_mean_max_abs_diff_max"]
    ):
        failures.append("official_vs_monkey.first_step.action_mean drift too high")
    if float(first_step.get("values_max_abs_diff", float("inf"))) > float(
        reg_spec["first_step_values_max_abs_diff_max"]
    ):
        failures.append("official_vs_monkey.first_step.values drift too high")
    if float(synced.get("policy_param_max_abs_diff", float("inf"))) != 0.0:
        failures.append("official_vs_monkey.synced_param_diff not zero")
    actual_unsynced_names = [str(item["name"]) for item in unsynced.get("differing_tensors", [])]
    if actual_unsynced_names != list(reg_spec["expected_unsynced_tensor_names"]):
        failures.append("official_vs_monkey.unsynced differing tensor set changed")

    milestone_spec = manifest["shared_milestone"]
    tol = float(milestone_spec["abs_tol"])
    for key, expected in milestone_spec.get("required_report_fields", {}).items():
        if milestone.get(key) != expected:
            failures.append(f"shared_milestone.{key} drift")
    fixed_env_contract = dict(milestone.get("fixed_env_contract", {}))
    for key, expected in milestone_spec.get("required_fixed_env_contract", {}).items():
        if fixed_env_contract.get(key) != expected:
            failures.append(f"shared_milestone.fixed_env_contract.{key} drift")
    if not _check_close(
        actual=float(milestone["zero_suffix_return_mean"]),
        expected=float(milestone_spec["zero_suffix_return_mean"]),
        abs_tol=tol,
    ):
        failures.append("shared_milestone.zero_suffix_return_mean drift")
    for mode_name in ("learned", "zero"):
        for key, expected in milestone_spec[mode_name].items():
            actual = float(milestone[mode_name][key])
            if not _check_close(actual=actual, expected=float(expected), abs_tol=tol):
                failures.append(f"shared_milestone.{mode_name}.{key} drift")
    if not (float(milestone["learned"]["delta_smp_gap"]) > float(milestone["zero"]["delta_smp_gap"])):
        failures.append("shared_milestone.learned no longer beats zero")
    if not (float(milestone["learned"]["delta_smp_gap"]) > 0.0):
        failures.append("shared_milestone.learned delta not positive")
    heldout_quality_spec = dict(milestone_spec.get("heldout_critic_quality", {}))
    for mode_name, quality_spec in heldout_quality_spec.items():
        quality = milestone.get(mode_name, {}).get("eval_suite_critic_quality_after_train")
        if not isinstance(quality, dict):
            failures.append(f"shared_milestone.{mode_name}.eval_suite_critic_quality_after_train missing")
            continue
        if int(quality.get("objective_total", -1)) != int(quality_spec["objective_total"]):
            failures.append(f"shared_milestone.{mode_name}.eval_suite_critic_quality_after_train.objective_total drift")

    return {
        "manifest_path": str(Path(manifest_path).expanduser().resolve()),
        "all_checks_pass": bool(len(failures) == 0),
        "failures": failures,
        "checked_items": {
            "official_vs_monkey_exact_zero_keys": list(reg_spec["exact_zero_keys"]),
            "official_vs_monkey_timing_ratio_range": [
                float(reg_spec["timing_ratio_min"]),
                float(reg_spec["timing_ratio_max"]),
            ],
            "shared_milestone_abs_tol": tol,
        },
        "disabled_untrusted_metrics": [
            "raw_corr",
            "explained_variance_raw",
            "raw_corr_min",
            "explained_variance_raw_min",
        ],
    }


def _pass_flags(*, regression: dict, milestone: dict) -> dict:
    learned = milestone["learned"]
    zero = milestone["zero"]
    regression_core = regression.get("regression_compare_after_sync", regression)
    heldout_quality_recorded = bool(
        isinstance(learned.get("eval_suite_critic_quality_after_train"), dict)
        and isinstance(zero.get("eval_suite_critic_quality_after_train"), dict)
    )
    return {
        "official_vs_monkey_numeric_match": bool(
            float(regression_core["policy_loss_abs_diff"]) == 0.0
            and float(regression_core["grad_l2_delta"]) == 0.0
            and float(regression_core["actions_max_abs_diff"]) == 0.0
            and float(regression_core["returns_max_abs_diff"]) == 0.0
            and float(regression_core["values_max_abs_diff"]) == 0.0
            and float(regression_core["advantages_max_abs_diff"]) == 0.0
            and float(regression_core["log_probs_max_abs_diff"]) == 0.0
        ),
        "shared_learned_beats_zero": bool(
            float(learned["delta_smp_gap"]) > float(zero["delta_smp_gap"])
        ),
        "shared_learned_delta_positive": bool(float(learned["delta_smp_gap"]) > 0.0),
        "shared_heldout_critic_quality_recorded": heldout_quality_recorded,
    }


def _summary_matches_invocation(*, summary: dict, args) -> bool:
    contract = summary.get("trusted_contract", {})
    assembly = summary.get("assembly_mode", {})
    return all(
        [
            str(contract.get("checkpoint_path", "")) == str(Path(args.checkpoint_path).expanduser().resolve()),
            str(contract.get("device", "")) == str(args.device or _default_device()),
            int(contract.get("frozen_h_seed", -1)) == int(args.frozen_h_seed),
            int(contract.get("train_env_seed", -1)) == int(args.train_env_seed),
            int(contract.get("train_rollout_seed", -1)) == int(args.train_rollout_seed),
            int(contract.get("single_eval_pos", -1)) == int(args.single_eval_pos),
            int(contract.get("n_steps", -1)) == int(args.n_steps),
            int(contract.get("batch_size", -1)) == int(args.batch_size),
            int(contract.get("n_epochs", -1)) == int(args.n_epochs),
            int(contract.get("outer_epochs", -1)) == int(args.outer_epochs),
            float(contract.get("learning_rate", float("nan"))) == float(args.learning_rate),
            float(contract.get("target_kl", float("nan"))) == float(args.target_kl),
            int(contract.get("build_seed", -1)) == int(args.build_seed),
            bool(contract.get("ppo_reset_env_state_at_sep", False)) is True,
            bool(contract.get("strict_fixed_env_mode", False)) is True,
            bool(contract.get("deterministic_actor_sampling", False)) is True,
            bool(contract.get("deterministic_batch_plan", False)) is True,
            bool(contract.get("strict_native_rollout", True)) is False,
            bool(contract.get("restore_validation_policy_state", True)) is False,
            bool(contract.get("shared_actor_critic_backbone", False)) is True,
            bool(contract.get("record_suite_critic_quality", False)) is True,
            str(contract.get("frozen_h_seed_mode", "")) == "current",
            bool(contract.get("zero_eval_prior_reuses_sampling_state", True)) is False,
            str(summary.get("manifest_path", "")) == str(Path(args.manifest_path).expanduser().resolve()),
            bool(assembly.get("reuse_canonical_official_monkey", False))
            == bool(args.reuse_canonical_official_monkey),
            bool(assembly.get("reuse_canonical_shared_milestone", False))
            == bool(args.reuse_canonical_shared_milestone),
        ]
    )


def _load_existing_summary_if_reusable(*, summary_path: Path, args) -> dict | None:
    if not summary_path.exists():
        return None
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not _summary_matches_invocation(summary=summary, args=args):
        return None
    validation = summary.get("validation", {})
    if bool(validation.get("all_checks_pass", False)) is not True:
        return None
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the maintained Phase 2 regression pack on the trusted fixed-env contract."
    )
    parser.add_argument("--checkpoint-path", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--frozen-h-seed", type=int, default=12345)
    parser.add_argument("--train-env-seed", type=int, default=2020)
    parser.add_argument("--train-rollout-seed", type=int, default=4040)
    parser.add_argument("--single-eval-pos", type=int, default=64)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=1)
    parser.add_argument("--outer-epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument("--build-seed", type=int, default=4040)
    parser.add_argument("--timing-repeats", type=int, default=3)
    parser.add_argument(
        "--reuse-canonical-official-monkey",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--reuse-canonical-shared-milestone",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--manifest-path", type=str, default=DEFAULT_MANIFEST)
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--allow-validation-failure", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    device = str(args.device or _default_device())
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    official_vs_monkey_path = output_dir / "critic_official_vs_monkey_regression.json"
    shared_milestone_path = output_dir / "shared_backbone_milestone_quick.json"
    summary_path = output_dir / "phase2_regression_suite_summary.json"
    lock_path = output_dir / ".phase2_regression_suite.lock"
    manifest = _load_manifest(args.manifest_path)

    if not bool(args.force_rerun):
        existing_summary = _load_existing_summary_if_reusable(summary_path=summary_path, args=args)
        if existing_summary is not None:
            existing_summary["summary_reused"] = True
            print(json.dumps(existing_summary, sort_keys=True))
            return 0

    _acquire_lock(lock_path)
    try:
        if bool(args.reuse_canonical_official_monkey):
            regression = _reuse_json(
                source_path=DEFAULT_CANONICAL_OFFICIAL_MONKEY,
                output_path=official_vs_monkey_path,
            )
        else:
            regression = _run_json_command(
                cmd=[
                    str(Path(_python_executable()).resolve()),
                    "ticl/analysis/critic_official_vs_monkey_regression.py",
                    str(Path(args.checkpoint_path).expanduser().resolve()),
                    "--device",
                    device,
                    "--frozen-h-seed",
                    str(args.frozen_h_seed),
                    "--train-env-seed",
                    str(args.train_env_seed),
                    "--train-rollout-seed",
                    str(args.train_rollout_seed),
                    "--single-eval-pos",
                    str(args.single_eval_pos),
                    "--n-steps",
                    str(args.n_steps),
                    "--batch-size",
                    str(args.batch_size),
                    "--n-epochs",
                    str(args.n_epochs),
                    "--learning-rate",
                    str(args.learning_rate),
                    "--target-kl",
                    str(args.target_kl),
                    "--timing-repeats",
                    str(args.timing_repeats),
                    "--output-json",
                    str(official_vs_monkey_path),
                ],
                output_path=official_vs_monkey_path,
            )

        if bool(args.reuse_canonical_shared_milestone):
            audit = _reuse_json(
                source_path=DEFAULT_CANONICAL_MILESTONE,
                output_path=shared_milestone_path,
            )
        else:
            audit = _run_json_command(
                cmd=[
                    str(Path(_python_executable()).resolve()),
                    "ticl/analysis/critic_free_single_env_audit.py",
                    str(Path(args.checkpoint_path).expanduser().resolve()),
                    "--device",
                    device,
                    "--frozen-h-seed",
                    str(args.frozen_h_seed),
                    "--train-env-seed",
                    str(args.train_env_seed),
                    "--eval-env-seed-start",
                    "3000",
                    "--eval-rollout-seed-start",
                    "4000",
                    "--eval-env-count",
                    "8",
                    "--single-eval-pos",
                    str(args.single_eval_pos),
                    "--n-steps",
                    str(args.n_steps),
                    "--outer-epochs",
                    str(args.outer_epochs),
                    "--learning-rate",
                    str(args.learning_rate),
                    "--batch-size",
                    str(args.batch_size),
                    "--n-epochs",
                    str(args.n_epochs),
                    "--target-kl",
                    str(args.target_kl),
                    "--build-seed",
                    str(args.build_seed),
                    "--strict-fixed-env-mode",
                    "--train-rollout-seed",
                    str(args.train_rollout_seed),
                    "--frozen-h-seed-mode",
                    "current",
                    "--deterministic-actor-sampling",
                    "--ppo-reset-env-state-at-sep",
                    "--no-restore-validation-policy-state",
                    "--record-suite-critic-quality",
                    "--modes",
                    "learned",
                    "zero",
                    "--output-json",
                    str(shared_milestone_path),
                ],
                output_path=shared_milestone_path,
            )
    finally:
        _release_lock(lock_path)

    milestone = _milestone_summary(audit)
    pass_flags = _pass_flags(regression=regression, milestone=milestone)
    validation = _validate_against_manifest(
        regression=regression,
        milestone=milestone,
        manifest=manifest,
        manifest_path=args.manifest_path,
    )

    summary = {
        "suite_name": "phase2_regression_pack",
        "phase2_frozen": True,
        "trusted_contract": {
            "checkpoint_path": str(Path(args.checkpoint_path).expanduser().resolve()),
            "device": device,
            "frozen_h_seed": int(args.frozen_h_seed),
            "train_env_seed": int(args.train_env_seed),
            "train_rollout_seed": int(args.train_rollout_seed),
            "single_eval_pos": int(args.single_eval_pos),
            "n_steps": int(args.n_steps),
            "batch_size": int(args.batch_size),
            "n_epochs": int(args.n_epochs),
            "outer_epochs": int(args.outer_epochs),
            "learning_rate": float(args.learning_rate),
            "target_kl": float(args.target_kl),
            "build_seed": int(args.build_seed),
            "ppo_reset_env_state_at_sep": True,
            "strict_fixed_env_mode": True,
            "deterministic_actor_sampling": True,
            "deterministic_batch_plan": True,
            "strict_native_rollout": False,
            "restore_validation_policy_state": False,
            "record_suite_critic_quality": True,
            "frozen_h_seed_mode": "current",
            "zero_eval_prior_reuses_sampling_state": False,
            "shared_actor_critic_backbone": True,
        },
        "legacy_artifacts": {
            "sep_state_reset_gradient_snapshot": DEFAULT_LEGACY_GRAD_SNAPSHOT,
            "canonical_official_vs_monkey": DEFAULT_CANONICAL_OFFICIAL_MONKEY,
            "canonical_shared_milestone": DEFAULT_CANONICAL_MILESTONE,
        },
        "generated_artifacts": {
            "official_vs_monkey_regression": str(official_vs_monkey_path),
            "shared_backbone_milestone_quick": str(shared_milestone_path),
        },
        "assembly_mode": {
            "reuse_canonical_official_monkey": bool(args.reuse_canonical_official_monkey),
            "reuse_canonical_shared_milestone": bool(args.reuse_canonical_shared_milestone),
        },
        "summary_reused": False,
        "manifest_path": str(Path(args.manifest_path).expanduser().resolve()),
        "official_vs_monkey_regression": regression,
        "shared_backbone_milestone": milestone,
        "pass_flags": pass_flags,
        "validation": validation,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True))
    if not bool(validation["all_checks_pass"]) and not bool(args.allow_validation_failure):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
