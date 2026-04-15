import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from phase2_guardrail import DEFAULT_PHASE2_SUMMARY, assert_phase2_green
else:
    from ticl.analysis.phase2_guardrail import (
        DEFAULT_PHASE2_SUMMARY,
        assert_phase2_green,
    )


DEFAULT_OUTPUT_DIR = "/home/chen/RLPFN/artifacts/phase3_regression_pack"
DEFAULT_MANIFEST = (
    "/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_regression_manifest.json"
)
CANONICAL_LONGRUN_GRADIENT = "/home/chen/RLPFN/artifacts/phase3_longrun_gradient_compare.json"
CANONICAL_SHORTRUN_PROFILE = (
    "/home/chen/RLPFN/artifacts/phase3_shortrun_profile_gradient_regression_v2.json"
)
CANONICAL_TRUSTED_PAIR1 = "/home/chen/RLPFN/artifacts/phase3_multi_env_trusted_pair1.json"
CANONICAL_PAIR1_PPO = "/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/ppo_cross_env_baseline.json"
CANONICAL_PAIR1_ZERO = "/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/zero_cross_env_control.json"
CANONICAL_PAIR2_PPO = (
    "/home/chen/RLPFN/artifacts/phase3_cross_env_baseline_pair2/ppo_cross_env_baseline.json"
)
CANONICAL_PAIR2_ZERO = (
    "/home/chen/RLPFN/artifacts/phase3_cross_env_baseline_pair2/zero_cross_env_control.json"
)


def _canonical_sources() -> dict[str, str]:
    return {
        "phase3_longrun_gradient_compare": CANONICAL_LONGRUN_GRADIENT,
        "phase3_shortrun_profile_gradient_regression_v2": CANONICAL_SHORTRUN_PROFILE,
        "phase3_multi_env_trusted_pair1": CANONICAL_TRUSTED_PAIR1,
        "phase3_cross_env_baseline_pair1_ppo": CANONICAL_PAIR1_PPO,
        "phase3_cross_env_baseline_pair1_zero": CANONICAL_PAIR1_ZERO,
        "phase3_cross_env_baseline_pair2_ppo": CANONICAL_PAIR2_PPO,
        "phase3_cross_env_baseline_pair2_zero": CANONICAL_PAIR2_ZERO,
    }


def _load_json(path: str | Path) -> dict:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _source_artifact_fingerprints() -> dict[str, dict[str, int | str]]:
    out: dict[str, dict[str, int | str]] = {}
    for name, source_path in _canonical_sources().items():
        path = Path(source_path).expanduser().resolve()
        stat = path.stat()
        out[name] = {
            "path": str(path),
            "mtime_ns": int(stat.st_mtime_ns),
            "size": int(stat.st_size),
            "sha256": _file_sha256(path),
        }
    return out


def _is_pid_alive(pid: int) -> bool:
    if int(pid) <= 0:
        return False
    try:
        os.kill(int(pid), 0)
    except OSError:
        return False
    return True


def _acquire_lock(lock_path: Path) -> None:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if lock_path.exists():
        try:
            payload = json.loads(lock_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        existing_pid = int(payload.get("pid", -1))
        if _is_pid_alive(existing_pid):
            raise RuntimeError(f"Phase 3 regression pack already running under pid={existing_pid}")
        lock_path.unlink(missing_ok=True)
    payload = {
        "pid": int(os.getpid()),
        "created_unix_s": float(time.time()),
        "argv": list(sys.argv),
    }
    lock_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _release_lock(lock_path: Path) -> None:
    lock_path.unlink(missing_ok=True)


def _reuse_json(*, source_path: str, output_path: Path) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(str(Path(source_path).expanduser().resolve()), output_path)
    return _load_json(output_path)


def _check_close(*, actual: float, expected: float, abs_tol: float) -> bool:
    return abs(float(actual) - float(expected)) <= float(abs_tol)


def _pair_gap_summary(*, ppo_report: dict, zero_report: dict) -> dict:
    out = {}
    for split in ("train_suite", "heldout_suite"):
        pm = ppo_report[split]["metrics"]
        zm = zero_report[split]["metrics"]
        out[split] = {
            "full_gap": float(pm["full_return_mean"] - zm["full_return_mean"]),
            "suffix_gap": float(pm["suffix_return_mean"] - zm["suffix_return_mean"]),
        }
    return out


def _summary_matches_invocation(*, summary: dict, args) -> bool:
    return str(summary.get("manifest_path", "")) == str(Path(args.manifest_path).expanduser().resolve())


def _load_existing_summary_if_reusable(*, summary_path: Path, args) -> dict | None:
    if not summary_path.exists():
        return None
    try:
        summary = _load_json(summary_path)
    except Exception:
        return None
    if not _summary_matches_invocation(summary=summary, args=args):
        return None
    if bool(summary.get("validation", {}).get("all_checks_pass", False)) is not True:
        return None
    if summary.get("source_artifact_fingerprints") != _source_artifact_fingerprints():
        return None
    return summary


def _mode_runtime_ratios(artifact: dict, mode_names: tuple[str, str]) -> dict[str, float]:
    left = artifact["modes"][mode_names[0]]
    right = artifact["modes"][mode_names[1]]
    collect_left = float(left["collect_wall_s"])
    collect_right = float(right["collect_wall_s"])
    backward_left = float(left["backward_wall_s"])
    backward_right = float(right["backward_wall_s"])
    return {
        "collect_ratio": max(collect_left, collect_right) / max(min(collect_left, collect_right), 1e-12),
        "backward_ratio": max(backward_left, backward_right) / max(min(backward_left, backward_right), 1e-12),
    }


def _validate(*, manifest: dict, longrun_grad: dict, shortrun_profile: dict, pair1_gaps: dict, pair2_gaps: dict, trusted_pair1: dict, manifest_path: str) -> dict:
    failures: list[str] = []
    phase3_claims_trusted = bool(manifest.get("phase3_claims_trusted", True))
    phase3_trust_reason = str(manifest.get("phase3_trust_reason", "")).strip()

    if not phase3_claims_trusted:
        return {
            "manifest_path": str(Path(manifest_path).expanduser().resolve()),
            "all_checks_pass": True,
            "failures": [],
            "diagnostic_only": True,
            "phase3_claims_trusted": False,
            "phase3_trust_reason": phase3_trust_reason,
        }

    longrun_spec = manifest["trusted_longrun_gradient_compare"]
    if bool(longrun_grad["training_signature"]["exact_match"]) is not bool(longrun_spec["exact_match_required"]):
        failures.append("trusted_longrun_gradient_compare.training_signature mismatch")
    if float(longrun_grad["grad_compare"]["cosine_similarity"]) < float(longrun_spec["cosine_similarity_min"]):
        failures.append("trusted_longrun_gradient_compare.cosine too low")
    if float(longrun_grad["grad_compare"]["l2_delta_norm"]) > float(longrun_spec["l2_delta_norm_max"]):
        failures.append("trusted_longrun_gradient_compare.l2 too high")
    tol = float(longrun_spec["abs_tol"])
    if not _check_close(
        actual=float(longrun_grad["grad_compare"]["resume_grad_norm"]),
        expected=float(longrun_spec["resume_grad_norm_expected"]),
        abs_tol=tol,
    ):
        failures.append("trusted_longrun_gradient_compare.resume_grad_norm drift")
    if not _check_close(
        actual=float(longrun_grad["grad_compare"]["phase3_isolated_grad_norm"]),
        expected=float(longrun_spec["phase3_isolated_grad_norm_expected"]),
        abs_tol=tol,
    ):
        failures.append("trusted_longrun_gradient_compare.phase3_isolated_grad_norm drift")
    if "collect_wall_s_max" in longrun_spec:
        for mode_name, mode_report in longrun_grad["modes"].items():
            if float(mode_report["collect_wall_s"]) > float(longrun_spec["collect_wall_s_max"]):
                failures.append(f"trusted_longrun_gradient_compare.{mode_name}.collect_wall_s too high")
    if "backward_wall_s_max" in longrun_spec:
        for mode_name, mode_report in longrun_grad["modes"].items():
            if float(mode_report["backward_wall_s"]) > float(longrun_spec["backward_wall_s_max"]):
                failures.append(f"trusted_longrun_gradient_compare.{mode_name}.backward_wall_s too high")
    longrun_ratios = _mode_runtime_ratios(
        longrun_grad,
        ("phase3_isolated_semantics", "resume_checkpoint_semantics"),
    )
    if "collect_ratio_max" in longrun_spec and float(longrun_ratios["collect_ratio"]) > float(longrun_spec["collect_ratio_max"]):
        failures.append("trusted_longrun_gradient_compare.collect_ratio too high")
    if "backward_ratio_max" in longrun_spec and float(longrun_ratios["backward_ratio"]) > float(longrun_spec["backward_ratio_max"]):
        failures.append("trusted_longrun_gradient_compare.backward_ratio too high")

    short_spec = manifest["legacy_shortrun_profile_gradient_compare"]
    if bool(short_spec.get("enabled", True)):
        if bool(shortrun_profile["training_signature"]["exact_match"]) is not bool(short_spec["exact_match_required"]):
            failures.append("legacy_shortrun_profile_gradient_compare.training_signature mismatch")
        if float(shortrun_profile["grad_compare"]["cosine_similarity"]) < float(short_spec["cosine_similarity_min"]):
            failures.append("legacy_shortrun_profile_gradient_compare.cosine too low")
        if float(shortrun_profile["grad_compare"]["l2_delta_norm"]) > float(short_spec["l2_delta_norm_max"]):
            failures.append("legacy_shortrun_profile_gradient_compare.l2 too high")
        tol = float(short_spec["abs_tol"])
        if not _check_close(
            actual=float(shortrun_profile["grad_compare"]["driver_grad_norm"]),
            expected=float(short_spec["driver_grad_norm_expected"]),
            abs_tol=tol,
        ):
            failures.append("legacy_shortrun_profile_gradient_compare.driver_grad_norm drift")
        if not _check_close(
            actual=float(shortrun_profile["grad_compare"]["reference_grad_norm"]),
            expected=float(short_spec["reference_grad_norm_expected"]),
            abs_tol=tol,
        ):
            failures.append("legacy_shortrun_profile_gradient_compare.reference_grad_norm drift")
        if "collect_wall_s_max" in short_spec:
            for mode_name, mode_report in shortrun_profile["modes"].items():
                if float(mode_report["collect_wall_s"]) > float(short_spec["collect_wall_s_max"]):
                    failures.append(f"legacy_shortrun_profile_gradient_compare.{mode_name}.collect_wall_s too high")
        if "backward_wall_s_max" in short_spec:
            for mode_name, mode_report in shortrun_profile["modes"].items():
                if float(mode_report["backward_wall_s"]) > float(short_spec["backward_wall_s_max"]):
                    failures.append(f"legacy_shortrun_profile_gradient_compare.{mode_name}.backward_wall_s too high")
        shortrun_ratios = _mode_runtime_ratios(
            shortrun_profile,
            ("driver_profile_shortrun_aligned", "manual_reference_shortrun_aligned"),
        )
        if "collect_ratio_max" in short_spec and float(shortrun_ratios["collect_ratio"]) > float(short_spec["collect_ratio_max"]):
            failures.append("legacy_shortrun_profile_gradient_compare.collect_ratio too high")
        if "backward_ratio_max" in short_spec and float(shortrun_ratios["backward_ratio"]) > float(short_spec["backward_ratio_max"]):
            failures.append("legacy_shortrun_profile_gradient_compare.backward_ratio too high")

    for pair_name, gaps in (("pair1", pair1_gaps), ("pair2", pair2_gaps)):
        spec = manifest["checkpoint_generalization"][pair_name]
        tol = float(manifest["checkpoint_generalization"]["abs_tol"])
        for split, split_name in (("train_suite", "train"), ("heldout_suite", "heldout")):
            for key in ("full_gap", "suffix_gap"):
                expected = float(spec[f"{split_name}_{key}"])
                actual = float(gaps[split][key])
                if not _check_close(actual=actual, expected=expected, abs_tol=tol):
                    failures.append(f"checkpoint_generalization.{pair_name}.{split_name}.{key} drift")

    legacy_pair1_spec = manifest.get("legacy_pair1_optimization", manifest.get("trusted_pair1_optimization", None))
    if legacy_pair1_spec is None:
        raise KeyError("Manifest must contain legacy_pair1_optimization or trusted_pair1_optimization")
    if bool(legacy_pair1_spec.get("enabled", True)):
        tol = float(legacy_pair1_spec["abs_tol"])
        for key in (
            "train_full_return_delta",
            "train_suffix_return_delta",
            "heldout_full_return_delta",
            "heldout_suffix_return_delta",
        ):
            actual = float(trusted_pair1["comparison"][key])
            expected = float(legacy_pair1_spec[key])
            if not _check_close(actual=actual, expected=expected, abs_tol=tol):
                failures.append(f"legacy_pair1_optimization.{key} drift")

    return {
        "manifest_path": str(Path(manifest_path).expanduser().resolve()),
        "all_checks_pass": bool(len(failures) == 0),
        "failures": failures,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Assemble and validate the trusted Phase 3 regression pack.")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest-path", type=str, default=DEFAULT_MANIFEST)
    parser.add_argument("--phase2-summary-path", type=str, default=DEFAULT_PHASE2_SUMMARY)
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--allow-validation-failure", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "phase3_regression_suite_summary.json"
    lock_path = output_dir / ".phase3_regression_suite.lock"

    if not bool(args.force_rerun):
        existing = _load_existing_summary_if_reusable(summary_path=summary_path, args=args)
        if existing is not None:
            existing["summary_reused"] = True
            print(json.dumps(existing, sort_keys=True))
            return 0

    _acquire_lock(lock_path)
    try:
        phase2_summary = assert_phase2_green(args.phase2_summary_path)
        manifest = _load_json(args.manifest_path)
        copied = {
            "phase3_longrun_gradient_compare": _reuse_json(
                source_path=CANONICAL_LONGRUN_GRADIENT,
                output_path=output_dir / "phase3_longrun_gradient_compare.json",
            ),
            "phase3_shortrun_profile_gradient_regression_v2": _reuse_json(
                source_path=CANONICAL_SHORTRUN_PROFILE,
                output_path=output_dir / "phase3_shortrun_profile_gradient_regression_v2.json",
            ),
            "phase3_multi_env_trusted_pair1": _reuse_json(
                source_path=CANONICAL_TRUSTED_PAIR1,
                output_path=output_dir / "phase3_multi_env_trusted_pair1.json",
            ),
            "phase3_cross_env_baseline_pair1_ppo": _reuse_json(
                source_path=CANONICAL_PAIR1_PPO,
                output_path=output_dir / "phase3_cross_env_baseline_pair1_ppo.json",
            ),
            "phase3_cross_env_baseline_pair1_zero": _reuse_json(
                source_path=CANONICAL_PAIR1_ZERO,
                output_path=output_dir / "phase3_cross_env_baseline_pair1_zero.json",
            ),
            "phase3_cross_env_baseline_pair2_ppo": _reuse_json(
                source_path=CANONICAL_PAIR2_PPO,
                output_path=output_dir / "phase3_cross_env_baseline_pair2_ppo.json",
            ),
            "phase3_cross_env_baseline_pair2_zero": _reuse_json(
                source_path=CANONICAL_PAIR2_ZERO,
                output_path=output_dir / "phase3_cross_env_baseline_pair2_zero.json",
            ),
        }
    finally:
        _release_lock(lock_path)

    pair1_gaps = _pair_gap_summary(
        ppo_report=copied["phase3_cross_env_baseline_pair1_ppo"],
        zero_report=copied["phase3_cross_env_baseline_pair1_zero"],
    )
    pair2_gaps = _pair_gap_summary(
        ppo_report=copied["phase3_cross_env_baseline_pair2_ppo"],
        zero_report=copied["phase3_cross_env_baseline_pair2_zero"],
    )
    validation = _validate(
        manifest=manifest,
        longrun_grad=copied["phase3_longrun_gradient_compare"],
        shortrun_profile=copied["phase3_shortrun_profile_gradient_regression_v2"],
        pair1_gaps=pair1_gaps,
        pair2_gaps=pair2_gaps,
        trusted_pair1=copied["phase3_multi_env_trusted_pair1"],
        manifest_path=args.manifest_path,
    )

    summary = {
        "suite_name": "phase3_regression_pack",
        "phase3_claims_trusted": bool(manifest.get("phase3_claims_trusted", True)),
        "phase3_trust_reason": str(manifest.get("phase3_trust_reason", "")),
        "pack_role": (
            "trusted_gate" if bool(manifest.get("phase3_claims_trusted", True)) else "diagnostic_only"
        ),
        "phase2_preflight": {
            "required": True,
            "summary_path": str(Path(args.phase2_summary_path).expanduser().resolve()),
            "phase2_all_checks_pass": bool(phase2_summary["validation"]["all_checks_pass"]),
            "phase2_pass_flags": dict(phase2_summary["pass_flags"]),
        },
        "phase3_correct_contract": {
            "batch_many_env": True,
            "one_fixed_n_samples_rollout_per_env": True,
            "internal_multi_episode_rollouts": True,
            "checkpoint_generalization_source": "fixed cross-env suites",
            "trusted_training_profile": "trusted_sep_reset_mainline",
        },
        "manifest_path": str(Path(args.manifest_path).expanduser().resolve()),
        "summary_reused": False,
        "generated_artifacts": {
            key: str((output_dir / f"{key}.json").resolve())
            for key in [
                "phase3_longrun_gradient_compare",
                "phase3_shortrun_profile_gradient_regression_v2",
                "phase3_multi_env_trusted_pair1",
                "phase3_cross_env_baseline_pair1_ppo",
                "phase3_cross_env_baseline_pair1_zero",
                "phase3_cross_env_baseline_pair2_ppo",
                "phase3_cross_env_baseline_pair2_zero",
            ]
        },
        "source_artifact_fingerprints": _source_artifact_fingerprints(),
        "checkpoint_generalization": {
            "pair1": pair1_gaps,
            "pair2": pair2_gaps,
        },
        "legacy_pair1_optimization": copied["phase3_multi_env_trusted_pair1"]["comparison"],
        "legacy_pair1_optimization_trusted": bool(manifest["legacy_pair1_optimization"].get("enabled", True)),
        "trusted_longrun_gradient_compare": copied["phase3_longrun_gradient_compare"]["grad_compare"],
        "legacy_shortrun_profile_gradient_compare": copied["phase3_shortrun_profile_gradient_regression_v2"]["grad_compare"],
        "legacy_shortrun_profile_gradient_compare_trusted": bool(manifest["legacy_shortrun_profile_gradient_compare"].get("enabled", True)),
        "validation": validation,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    if not bool(validation["all_checks_pass"]) and not bool(args.allow_validation_failure):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
