import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: str, *, expected_audit_entry: str) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise ValueError(f"Missing JSON artifact: {resolved}")
    payload = json.loads(resolved.read_text())
    if str(payload.get("audit_entry", "")) != str(expected_audit_entry):
        raise ValueError(f"Expected {expected_audit_entry} artifact.")
    return payload


def _suite_context(payload: dict[str, Any]) -> dict[str, Any]:
    config = payload["config"]
    ctx = {
        "train_suite_fingerprint": str(payload["train_suite_summary"]["fingerprint"]),
        "heldout_suite_fingerprint": str(payload["heldout_suite_summary"]["fingerprint"]),
        "train_suite_path": str(config["train_suite_path"]),
        "heldout_suite_path": str(config["heldout_suite_path"]),
        "n_samples": int(config["n_samples"]),
        "single_eval_pos": int(config["single_eval_pos"]),
        "strict_fixed_env_mode": bool(config.get("strict_fixed_env_mode", False)),
        "deterministic_actor_sampling": bool(config.get("deterministic_actor_sampling", False)),
        "deterministic_batch_plan": bool(config.get("deterministic_batch_plan", False)),
        "train_env_rng_seeds": [int(v) for v in config.get("train_env_rng_seeds", [])],
        "train_rollout_rng_seeds": [int(v) for v in config.get("train_rollout_rng_seeds", [])],
    }
    return ctx


def _env_rows(payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    rows = payload.get("env_items", [])
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Missing env_items in {payload.get('audit_entry')!r}")
    indexed: dict[int, dict[str, Any]] = {}
    for row in rows:
        env_index = int(row["env_index"])
        if env_index in indexed:
            raise ValueError(f"Duplicate env_index={env_index} in {payload.get('audit_entry')!r}")
        indexed[env_index] = dict(row)
    return indexed


def _max_abs_delta(base: dict[int, dict[str, Any]], other: dict[int, dict[str, Any]], key: str) -> float:
    common = sorted(set(base) & set(other))
    if not common:
        return 0.0
    return float(max(abs(float(other[idx][key]) - float(base[idx][key])) for idx in common))


def _mean_delta(base: dict[int, dict[str, Any]], other: dict[int, dict[str, Any]], key: str) -> float:
    common = sorted(set(base) & set(other))
    if not common:
        return 0.0
    return float(sum(float(other[idx][key]) - float(base[idx][key]) for idx in common) / float(len(common)))


def build_train_rollout_quality_mode_side_effect_compare_pack(
    *,
    baseline_json: str,
    override_json: str,
) -> dict[str, Any]:
    baseline = _load_json(baseline_json, expected_audit_entry="phase3_train_rollout_quality_probe")
    override = _load_json(override_json, expected_audit_entry="phase3_train_rollout_quality_probe")

    baseline_ctx = _suite_context(baseline)
    override_ctx = _suite_context(override)
    if baseline_ctx != override_ctx:
        raise ValueError(
            "Baseline and override rollout quality artifacts must share the same suite contract, "
            f"got {baseline_ctx!r} vs {override_ctx!r}"
        )

    baseline_rows = _env_rows(baseline)
    override_rows = _env_rows(override)
    common_envs = sorted(set(baseline_rows) & set(override_rows))
    if not common_envs:
        raise ValueError("No overlapping env indices between baseline and override artifacts.")

    key_metrics = [
        "normalized_actor_adv_positive_mass",
        "normalized_actor_adv_abs_mass",
        "normalized_actor_adv_first16_positive_share",
        "normalized_actor_adv_last16_positive_share",
        "raw_actor_adv_positive_mass",
        "raw_actor_adv_abs_mass",
        "raw_value_return_corr",
        "pre_vs_zero_suffix_gap",
        "objective_episode_count",
        "objective_terminal_reset_count",
    ]
    metric_deltas: dict[str, dict[str, float]] = {}
    for key in key_metrics:
        metric_deltas[key] = {
            "max_abs_delta": _max_abs_delta(baseline_rows, override_rows, key),
            "mean_delta": _mean_delta(baseline_rows, override_rows, key),
        }

    max_abs_metric_delta = max(float(v["max_abs_delta"]) for v in metric_deltas.values())
    side_effect_free = bool(max_abs_metric_delta <= 1e-6)
    most_shifted_metric = max(metric_deltas.items(), key=lambda kv: float(kv[1]["max_abs_delta"]))
    most_shifted_env = max(
        common_envs,
        key=lambda idx: abs(
            float(override_rows[idx]["normalized_actor_adv_positive_mass"])
            - float(baseline_rows[idx]["normalized_actor_adv_positive_mass"])
        ),
    )

    return {
        "audit_entry": "phase3_train_rollout_quality_mode_side_effect_compare_pack",
        "baseline_json": str(Path(baseline_json).expanduser().resolve()),
        "override_json": str(Path(override_json).expanduser().resolve()),
        "suite_context": baseline_ctx,
        "comparison": {
            "metric_deltas": metric_deltas,
            "max_abs_metric_delta": float(max_abs_metric_delta),
            "most_shifted_metric": {
                "name": str(most_shifted_metric[0]),
                "max_abs_delta": float(most_shifted_metric[1]["max_abs_delta"]),
                "mean_delta": float(most_shifted_metric[1]["mean_delta"]),
            },
            "most_shifted_env_index": int(most_shifted_env),
        },
        "conclusions": {
            "runtime_mode_no_other_suite_side_effects": bool(side_effect_free),
            "side_effects_are_detectable_in_train_side_ab": bool(not side_effect_free),
            "suite_contract_unmodified": True,
            "same_suite_fingerprint_checked": True,
        },
        "recommendation": {
            "candidate_small_control_target": "env12_mid_episode_extension_block",
            "branch_specific_scope_gate_needed": bool(not side_effect_free),
            "shared_core_unchanged": True,
            "reason": (
                "Compare baseline and runtime-mode train-side rollout quality artifacts on the same suite contract "
                "to verify whether the branch-specific mode stays local or leaks into other suites."
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare baseline and override train-side rollout-quality artifacts for mode side effects."
    )
    parser.add_argument(
        "--baseline-json",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--override-json",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output-json",
        type=str,
        required=True,
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output_json).expanduser().resolve()
    if output_path.exists() and not bool(args.overwrite):
        print(output_path.read_text())
        return 0

    report = build_train_rollout_quality_mode_side_effect_compare_pack(
        baseline_json=args.baseline_json,
        override_json=args.override_json,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
