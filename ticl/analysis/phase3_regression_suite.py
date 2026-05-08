import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path


DEFAULT_OUTPUT_DIR = "/home/chen/RLPFN/artifacts/phase3_regression_pack"
DEFAULT_MANIFEST = (
    "/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_regression_manifest.json"
)
CANONICAL_PHASE2_LAUNCH_ANCHOR = (
    "/home/chen/RLPFN/artifacts/"
    "phase2_launch_anchor_shared_seed12345_current_contract_quick.json"
)
CANONICAL_PHASE2_TO_PHASE3_COMPARE = (
    "/home/chen/RLPFN/artifacts/"
    "phase3_phase2_milestone_contract_gradient_compare_2026_04_17_rerun.json"
)
CANONICAL_PHASE3_ANCHOR_GRADIENT = (
    "/home/chen/RLPFN/artifacts/phase3_longrun_gradient_compare.json"
)
CANONICAL_FIT_MODEL_ALIGNMENT = (
    "/home/chen/RLPFN/artifacts/fit_model_phase3_anchor_alignment_probe.json"
)


def _canonical_sources() -> dict[str, str]:
    return {
        "phase2_launch_anchor": CANONICAL_PHASE2_LAUNCH_ANCHOR,
        "phase2_to_phase3_anchor_compare": CANONICAL_PHASE2_TO_PHASE3_COMPARE,
        "phase3_anchor_gradient_compare": CANONICAL_PHASE3_ANCHOR_GRADIENT,
        "fit_model_phase3_anchor_alignment_probe": CANONICAL_FIT_MODEL_ALIGNMENT,
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
        "collect_ratio": max(collect_left, collect_right)
        / max(min(collect_left, collect_right), 1e-12),
        "backward_ratio": max(backward_left, backward_right)
        / max(min(backward_left, backward_right), 1e-12),
    }


def _validate(
    *,
    manifest: dict,
    phase2_launch_anchor: dict,
    phase2_to_phase3_compare: dict,
    phase3_anchor_gradient_compare: dict,
    fit_model_alignment: dict,
    manifest_path: str,
) -> dict:
    failures: list[str] = []
    validated_sections: list[str] = []
    phase3_trust_reason = str(manifest.get("phase3_trust_reason", "")).strip()

    launch_spec = manifest["phase2_launch_anchor"]
    validated_sections.append("phase2_launch_anchor")
    launch_tol = float(launch_spec["abs_tol"])
    if not _check_close(
        actual=float(phase2_launch_anchor["zero_suffix_return_mean"]),
        expected=float(launch_spec["zero_suffix_return_mean"]),
        abs_tol=launch_tol,
    ):
        failures.append("phase2_launch_anchor.zero_suffix_return_mean drift")
    for mode_name in ("learned", "zero"):
        mode_report = phase2_launch_anchor["modes"][mode_name]
        spec_report = launch_spec[mode_name]
        for key, expected in spec_report.items():
            if not _check_close(
                actual=float(mode_report[key]),
                expected=float(expected),
                abs_tol=launch_tol,
            ):
                failures.append(f"phase2_launch_anchor.{mode_name}.{key} drift")
    if float(phase2_launch_anchor["modes"]["learned"]["delta_smp_gap"]) <= float(
        phase2_launch_anchor["modes"]["zero"]["delta_smp_gap"]
    ):
        failures.append("phase2_launch_anchor.learned delta no longer beats zero")
    for key, expected in launch_spec["required_contract"].items():
        if phase2_launch_anchor.get(key) != expected:
            failures.append(f"phase2_launch_anchor.{key} drift")
    fixed_env_contract = dict(phase2_launch_anchor.get("fixed_env_contract", {}))
    for key, expected in launch_spec.get("required_fixed_env_contract", {}).items():
        if fixed_env_contract.get(key) != expected:
            failures.append(f"phase2_launch_anchor.fixed_env_contract.{key} drift")
    if bool(phase2_launch_anchor.get("ppo_separate_value_backbone", True)) is not False:
        failures.append("phase2_launch_anchor.shared_backbone drift")
    heldout_quality_spec = dict(launch_spec.get("heldout_critic_quality", {}))
    for mode_name, quality_spec in heldout_quality_spec.items():
        quality = phase2_launch_anchor["modes"][mode_name].get("eval_suite_critic_quality_after_train")
        if not isinstance(quality, dict):
            failures.append(f"phase2_launch_anchor.{mode_name}.eval_suite_critic_quality_after_train missing")
            continue
        if int(quality.get("objective_total", -1)) != int(quality_spec["objective_total"]):
            failures.append(
                f"phase2_launch_anchor.{mode_name}.eval_suite_critic_quality_after_train.objective_total drift"
            )

    bridge_spec = manifest["phase2_to_phase3_anchor_compare"]
    validated_sections.append("phase2_to_phase3_anchor_compare")
    bridge_conclusions = dict(phase2_to_phase3_compare.get("conclusions", {}))
    for key in (
        "phase2_guardrail_passed",
        "gradient_exact_match",
        "rollout_tensors_exact_match",
        "scalar_measures_exact_match",
        "signature_exact_match",
    ):
        expected = bool(bridge_spec[key])
        if key == "phase2_guardrail_passed":
            actual = bool(phase2_to_phase3_compare.get(key, False))
        else:
            actual = bool(bridge_conclusions.get(key, False))
        if actual is not expected:
            failures.append(f"phase2_to_phase3_anchor_compare.{key} mismatch")
    bridge_tol = float(bridge_spec["abs_tol"])
    for key in ("phase2_grad_norm", "phase3_grad_norm"):
        if not _check_close(
            actual=float(phase2_to_phase3_compare["grad_compare"][key]),
            expected=float(bridge_spec["grad_compare"][key]),
            abs_tol=bridge_tol,
        ):
            failures.append(f"phase2_to_phase3_anchor_compare.{key} drift")
    if float(phase2_to_phase3_compare["grad_compare"]["cosine_similarity"]) < float(
        bridge_spec["grad_compare"]["cosine_similarity_min"]
    ):
        failures.append("phase2_to_phase3_anchor_compare.cosine too low")
    if float(phase2_to_phase3_compare["grad_compare"]["l2_delta_norm"]) > float(
        bridge_spec["grad_compare"]["l2_delta_norm_max"]
    ):
        failures.append("phase2_to_phase3_anchor_compare.l2 too high")
    for signature_name in ("phase2_signature", "phase3_signature"):
        signature = phase2_to_phase3_compare[signature_name]
        for key, expected in bridge_spec["required_signature_fields"].items():
            if signature.get(key) != expected:
                failures.append(
                    f"phase2_to_phase3_anchor_compare.{signature_name}.{key} drift"
                )

    phase3_spec = manifest["phase3_anchor_gradient_compare"]
    validated_sections.append("phase3_anchor_gradient_compare")
    if bool(phase3_anchor_gradient_compare["training_signature"]["exact_match"]) is not bool(
        phase3_spec["exact_match_required"]
    ):
        failures.append("phase3_anchor_gradient_compare.training_signature mismatch")
    if float(phase3_anchor_gradient_compare["grad_compare"]["cosine_similarity"]) < float(
        phase3_spec["cosine_similarity_min"]
    ):
        failures.append("phase3_anchor_gradient_compare.cosine too low")
    if float(phase3_anchor_gradient_compare["grad_compare"]["l2_delta_norm"]) > float(
        phase3_spec["l2_delta_norm_max"]
    ):
        failures.append("phase3_anchor_gradient_compare.l2 too high")
    phase3_tol = float(phase3_spec["abs_tol"])
    for key in ("resume_grad_norm", "phase3_isolated_grad_norm"):
        if not _check_close(
            actual=float(phase3_anchor_gradient_compare["grad_compare"][key]),
            expected=float(phase3_spec[key]),
            abs_tol=phase3_tol,
        ):
            failures.append(f"phase3_anchor_gradient_compare.{key} drift")
    if (
        str(phase3_anchor_gradient_compare["fixed_env_contract"]["frozen_h_fingerprint"])
        != str(phase3_spec["frozen_h_fingerprint"])
    ):
        failures.append("phase3_anchor_gradient_compare.frozen_h_fingerprint drift")
    if int(phase3_anchor_gradient_compare["fixed_env_contract"]["frozen_h_leaf_count"]) != int(
        phase3_spec["frozen_h_leaf_count"]
    ):
        failures.append("phase3_anchor_gradient_compare.frozen_h_leaf_count drift")
    for signature_name in ("phase3_isolated_semantics", "resume_checkpoint_semantics"):
        signature = phase3_anchor_gradient_compare["training_signature"][signature_name]
        for key, expected in phase3_spec["required_signature_fields"].items():
            if signature.get(key) != expected:
                failures.append(
                    f"phase3_anchor_gradient_compare.{signature_name}.{key} drift"
                )
    phase3_ratios = _mode_runtime_ratios(
        phase3_anchor_gradient_compare,
        ("phase3_isolated_semantics", "resume_checkpoint_semantics"),
    )
    if float(phase3_ratios["collect_ratio"]) > float(phase3_spec["collect_ratio_max"]):
        failures.append("phase3_anchor_gradient_compare.collect_ratio too high")
    if float(phase3_ratios["backward_ratio"]) > float(phase3_spec["backward_ratio_max"]):
        failures.append("phase3_anchor_gradient_compare.backward_ratio too high")

    fit_spec = manifest["fit_model_anchor_alignment"]
    validated_sections.append("fit_model_anchor_alignment")
    if str(fit_model_alignment["localized_read"]["stage"]) != str(fit_spec["expected_stage"]):
        failures.append("fit_model_anchor_alignment.localized_read.stage drift")
    training_signature = fit_model_alignment["training_signature"]
    if bool(training_signature["exact_match"]) is not bool(fit_spec["exact_match_required"]):
        failures.append("fit_model_anchor_alignment.training_signature mismatch")
    allowed_diff_keys = set(fit_spec["allowed_training_signature_diff_keys"])
    actual_diff_keys = set(training_signature.get("diff", {}).keys())
    if not actual_diff_keys.issubset(allowed_diff_keys):
        failures.append("fit_model_anchor_alignment.training_signature unexpected diff keys")
    fit_grad = fit_model_alignment["training_compare"]["grad_compare"]
    if float(fit_grad["cosine_similarity"]) < float(fit_spec["grad_cosine_similarity_min"]):
        failures.append("fit_model_anchor_alignment.grad cosine too low")
    if float(fit_grad["l2_delta_norm"]) > float(fit_spec["grad_l2_delta_norm_max"]):
        failures.append("fit_model_anchor_alignment.grad l2 too high")
    fit_tol = float(fit_spec["abs_tol"])
    for key in ("fit_model_grad_norm", "anchor_grad_norm"):
        if not _check_close(
            actual=float(fit_grad[key]),
            expected=float(fit_spec[key]),
            abs_tol=fit_tol,
        ):
            failures.append(f"fit_model_anchor_alignment.{key} drift")
    if bool(
        fit_model_alignment["training_compare"]["fit_model_vs_anchor_action_head"]["exact_match"]
    ) is not True:
        failures.append("fit_model_anchor_alignment.action_head not exact")
    if bool(
        fit_model_alignment["training_compare"]["fit_model_vs_anchor_value_head"]["exact_match"]
    ) is not True:
        failures.append("fit_model_anchor_alignment.value_head not exact")
    if bool(
        fit_model_alignment["training_compare"]["saved_policy_state_compare"]["exact_match"]
    ) is not True:
        failures.append("fit_model_anchor_alignment.saved_policy_state not exact")
    for key in ("action_mean", "action_std", "value_logits", "values"):
        if bool(fit_model_alignment["validation_compare"][key]["exact_match"]) is not True:
            failures.append(f"fit_model_anchor_alignment.validation_compare.{key} not exact")
    for key, expected in fit_spec["required_runtime_fields"].items():
        if fit_model_alignment["fit_model_runtime"].get(key) != expected:
            failures.append(f"fit_model_anchor_alignment.fit_model_runtime.{key} drift")
        if fit_model_alignment["anchor_runtime"].get(key) != expected:
            failures.append(f"fit_model_anchor_alignment.anchor_runtime.{key} drift")
    for signature_name in ("fit_model_active_bridge", "phase3_anchor_reference"):
        signature = training_signature[signature_name]
        for key, expected in fit_spec["required_training_signature_fields"].items():
            if signature.get(key) != expected:
                failures.append(
                    f"fit_model_anchor_alignment.{signature_name}.{key} drift"
                )

    return {
        "manifest_path": str(Path(manifest_path).expanduser().resolve()),
        "all_checks_pass": bool(len(failures) == 0),
        "failures": failures,
        "validated_sections": validated_sections,
        "phase3_claims_trusted": True,
        "phase3_trust_reason": phase3_trust_reason,
        "diagnostic_only": False,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Assemble and validate the unique trusted Phase 2 -> Phase 3 -> fit_model anchor chain."
    )
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest-path", type=str, default=DEFAULT_MANIFEST)
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
        manifest = _load_json(args.manifest_path)
        copied = {
            "phase2_launch_anchor": _reuse_json(
                source_path=CANONICAL_PHASE2_LAUNCH_ANCHOR,
                output_path=output_dir / "phase2_launch_anchor.json",
            ),
            "phase2_to_phase3_anchor_compare": _reuse_json(
                source_path=CANONICAL_PHASE2_TO_PHASE3_COMPARE,
                output_path=output_dir / "phase2_to_phase3_anchor_compare.json",
            ),
            "phase3_anchor_gradient_compare": _reuse_json(
                source_path=CANONICAL_PHASE3_ANCHOR_GRADIENT,
                output_path=output_dir / "phase3_anchor_gradient_compare.json",
            ),
            "fit_model_phase3_anchor_alignment_probe": _reuse_json(
                source_path=CANONICAL_FIT_MODEL_ALIGNMENT,
                output_path=output_dir / "fit_model_phase3_anchor_alignment_probe.json",
            ),
        }
    finally:
        _release_lock(lock_path)

    validation = _validate(
        manifest=manifest,
        phase2_launch_anchor=copied["phase2_launch_anchor"],
        phase2_to_phase3_compare=copied["phase2_to_phase3_anchor_compare"],
        phase3_anchor_gradient_compare=copied["phase3_anchor_gradient_compare"],
        fit_model_alignment=copied["fit_model_phase3_anchor_alignment_probe"],
        manifest_path=args.manifest_path,
    )

    summary = {
        "suite_name": "phase3_regression_pack",
        "phase3_claims_trusted": True,
        "phase3_trust_reason": str(manifest.get("phase3_trust_reason", "")),
        "pack_role": "trusted_unique_anchor_chain",
        "unique_trusted_chain": [
            "phase2_launch_anchor",
            "phase2_to_phase3_anchor_compare",
            "phase3_anchor_gradient_compare",
            "fit_model_anchor_alignment",
        ],
        "historical_traceback_only": list(manifest.get("historical_traceback_only", [])),
        "manifest_path": str(Path(args.manifest_path).expanduser().resolve()),
        "summary_reused": False,
        "generated_artifacts": {
            "phase2_launch_anchor": str((output_dir / "phase2_launch_anchor.json").resolve()),
            "phase2_to_phase3_anchor_compare": str(
                (output_dir / "phase2_to_phase3_anchor_compare.json").resolve()
            ),
            "phase3_anchor_gradient_compare": str(
                (output_dir / "phase3_anchor_gradient_compare.json").resolve()
            ),
            "fit_model_phase3_anchor_alignment_probe": str(
                (output_dir / "fit_model_phase3_anchor_alignment_probe.json").resolve()
            ),
        },
        "source_artifact_fingerprints": _source_artifact_fingerprints(),
        "phase2_launch_anchor": {
            "zero_suffix_return_mean": float(copied["phase2_launch_anchor"]["zero_suffix_return_mean"]),
            "learned_delta_smp_gap": float(
                copied["phase2_launch_anchor"]["modes"]["learned"]["delta_smp_gap"]
            ),
            "zero_delta_smp_gap": float(
                copied["phase2_launch_anchor"]["modes"]["zero"]["delta_smp_gap"]
            ),
            "disabled_untrusted_metrics": [
                "raw_corr",
                "explained_variance_raw",
                "raw_corr_min",
                "explained_variance_raw_min",
            ],
        },
        "phase2_to_phase3_anchor_compare": copied["phase2_to_phase3_anchor_compare"][
            "grad_compare"
        ],
        "phase3_anchor_gradient_compare": copied["phase3_anchor_gradient_compare"][
            "grad_compare"
        ],
        "fit_model_anchor_alignment": {
            "localized_read": dict(copied["fit_model_phase3_anchor_alignment_probe"]["localized_read"]),
            "grad_compare": dict(
                copied["fit_model_phase3_anchor_alignment_probe"]["training_compare"][
                    "grad_compare"
                ]
            ),
            "fit_model_runtime": dict(
                copied["fit_model_phase3_anchor_alignment_probe"]["fit_model_runtime"]
            ),
            "anchor_runtime": dict(
                copied["fit_model_phase3_anchor_alignment_probe"]["anchor_runtime"]
            ),
        },
        "validation": validation,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    if not bool(validation["all_checks_pass"]) and not bool(args.allow_validation_failure):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
