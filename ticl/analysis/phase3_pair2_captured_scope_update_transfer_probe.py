import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_EXACT_UPDATE_AGGREGATION_AUDIT_JSON = (
    "/home/chen/RLPFN/artifacts/phase3_pair2_exact_update_aggregation_audit.json"
)
DEFAULT_OUTPUT_JSON = (
    "/home/chen/RLPFN/artifacts/phase3_pair2_captured_scope_update_transfer_probe.json"
)

STAGE_NAMES = (
    "pre_normalization_actor_advantages",
    "post_normalization_advantages",
    "policy_num_from_clipped_objective",
)


def _load_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).expanduser().resolve().read_text())


def _sorted_objective_segments(payload: dict[str, Any]) -> list[dict[str, int]]:
    segments = list(payload["available_snapshot_scope"]["objective_segments"])
    return sorted(
        (
            {
                "env_index": int(row["env_index"]),
                "objective_episode_index": int(row["objective_episode_index"]),
                "objective_position_min": int(row["objective_position_min"]),
                "objective_position_max": int(row["objective_position_max"]),
                "token_count": int(row["token_count"]),
            }
            for row in segments
        ),
        key=lambda row: (
            -int(row["token_count"]),
            int(row["env_index"]),
            int(row["objective_episode_index"]),
            int(row["objective_position_min"]),
            int(row["objective_position_max"]),
        ),
    )


def _select_captured_scope(payload: dict[str, Any]) -> tuple[dict[str, int], str]:
    segments = _sorted_objective_segments(payload)
    if not segments:
        raise ValueError("Exact update aggregation audit has no available_snapshot_scope.objective_segments")
    if len(segments) == 1:
        return dict(segments[0]), "single_available_objective_segment"
    return dict(segments[0]), "largest_objective_segment"


def _scope_covers_whole_batch_objective(*, strict_contract: dict[str, Any], captured_scope: dict[str, int]) -> bool:
    objective_total = int(strict_contract["objective_total"])
    scope_token_count = int(captured_scope["token_count"])
    scope_min = int(captured_scope["objective_position_min"])
    scope_max = int(captured_scope["objective_position_max"])
    return (
        scope_token_count == objective_total
        and scope_min == 0
        and scope_max == max(objective_total - 1, 0)
    )


def _float(value: Any) -> float:
    return float(value)


def _captured_scope_stage_metrics(payload: dict[str, Any], stage_name: str) -> dict[str, float]:
    return {
        key: (_float(value) if key != "token_count" else int(value))
        for key, value in payload[stage_name]["whole_batch"].items()
    }


def _captured_scope_ratio_and_clipping(payload: dict[str, Any]) -> dict[str, float]:
    section = dict(payload["ratio_and_clipping"])
    return {
        "clip_active_fraction": _float(section["whole_batch_clip_active_fraction"]),
        "ratio_mean": _float(section["whole_batch_ratio_mean"]),
    }


def _stage_delta(baseline: dict[str, float], override: dict[str, float]) -> dict[str, float]:
    numeric_keys = (
        "signed_sum",
        "abs_mass",
        "positive_mass",
        "negative_mass",
        "mean",
    )
    return {f"{key}_delta": _float(override[key]) - _float(baseline[key]) for key in numeric_keys}


def build_pair2_captured_scope_update_transfer_probe(
    *,
    exact_update_aggregation_audit_json: str,
) -> dict[str, Any]:
    payload = _load_json(exact_update_aggregation_audit_json)
    strict_contract = {
        "outer_batch_idx": int(payload["strict_contract"]["outer_batch_idx"]),
        "epoch_idx": int(payload["strict_contract"]["epoch_idx"]),
        "batch_size_flat": int(payload["strict_contract"]["batch_size_flat"]),
        "objective_total": int(payload["strict_contract"]["objective_total"]),
        "normalize_advantage": bool(payload["strict_contract"]["normalize_advantage"]),
        "runtime_suite_name": str(payload["strict_contract"]["runtime_suite_name"]),
        "baseline_actor_objective_mode": str(payload["strict_contract"]["baseline_actor_objective_mode"]),
        "override_actor_objective_mode": str(payload["strict_contract"]["override_actor_objective_mode"]),
    }
    captured_scope, selection_reason = _select_captured_scope(payload)
    scope_is_whole_batch = _scope_covers_whole_batch_objective(
        strict_contract=strict_contract,
        captured_scope=captured_scope,
    )
    if not scope_is_whole_batch:
        raise ValueError(
            "Current captured-scope probe only supports the case where the selected captured scope equals the "
            "whole-batch objective. This exact audit does not satisfy that contract."
        )

    baseline = dict(payload["baseline"])
    override = dict(payload["override"])
    baseline_stage_metrics = {
        stage_name: _captured_scope_stage_metrics(baseline, stage_name)
        for stage_name in STAGE_NAMES
    }
    override_stage_metrics = {
        stage_name: _captured_scope_stage_metrics(override, stage_name)
        for stage_name in STAGE_NAMES
    }
    delta_stage_metrics = {
        stage_name: _stage_delta(
            baseline_stage_metrics[stage_name],
            override_stage_metrics[stage_name],
        )
        for stage_name in STAGE_NAMES
    }
    baseline_ratio = _captured_scope_ratio_and_clipping(baseline)
    override_ratio = _captured_scope_ratio_and_clipping(override)
    ratio_delta = {
        "clip_active_fraction_delta": _float(override_ratio["clip_active_fraction"]) - _float(baseline_ratio["clip_active_fraction"]),
        "ratio_mean_delta": _float(override_ratio["ratio_mean"]) - _float(baseline_ratio["ratio_mean"]),
    }

    eps = 1e-12
    return {
        "probe_entry": "phase3_pair2_captured_scope_update_transfer_probe",
        "inputs": {
            "exact_update_aggregation_audit_json": str(
                Path(exact_update_aggregation_audit_json).expanduser().resolve()
            ),
        },
        "strict_contract": strict_contract,
        "available_snapshot_scope": {
            "objective_segments": _sorted_objective_segments(payload),
            "selected_scope": dict(captured_scope),
            "selected_scope_reason": str(selection_reason),
            "selected_scope_equals_whole_batch_objective": bool(scope_is_whole_batch),
            "selected_scope_is_single_env_single_episode": True,
        },
        "baseline": {
            "captured_scope": baseline_stage_metrics,
            "ratio_and_clipping": baseline_ratio,
        },
        "override": {
            "captured_scope": override_stage_metrics,
            "ratio_and_clipping": override_ratio,
        },
        "delta": {
            "captured_scope": delta_stage_metrics,
            "ratio_and_clipping": ratio_delta,
        },
        "conclusions": {
            "captured_scope_is_non_target_env": bool(int(captured_scope["env_index"]) != 12),
            "captured_scope_equals_whole_batch_objective": bool(scope_is_whole_batch),
            "captured_scope_pre_stage_changed": bool(
                abs(delta_stage_metrics["pre_normalization_actor_advantages"]["abs_mass_delta"]) > eps
                or abs(delta_stage_metrics["pre_normalization_actor_advantages"]["signed_sum_delta"]) > eps
            ),
            "captured_scope_post_stage_changed": bool(
                abs(delta_stage_metrics["post_normalization_advantages"]["abs_mass_delta"]) > eps
                or abs(delta_stage_metrics["post_normalization_advantages"]["signed_sum_delta"]) > eps
            ),
            "captured_scope_policy_stage_changed": bool(
                abs(delta_stage_metrics["policy_num_from_clipped_objective"]["abs_mass_delta"]) > eps
                or abs(delta_stage_metrics["policy_num_from_clipped_objective"]["signed_sum_delta"]) > eps
            ),
            "captured_scope_ratio_or_clip_changed": bool(
                abs(ratio_delta["clip_active_fraction_delta"]) > eps
                or abs(ratio_delta["ratio_mean_delta"]) > eps
            ),
            "current_snapshot_is_not_the_transfer_site_for_env12_branch_specific_gain": bool(
                int(captured_scope["env_index"]) != 12
                and all(
                    abs(delta_stage_metrics[stage_name]["abs_mass_delta"]) <= eps
                    and abs(delta_stage_metrics[stage_name]["signed_sum_delta"]) <= eps
                    for stage_name in STAGE_NAMES
                )
                and abs(ratio_delta["clip_active_fraction_delta"]) <= eps
                and abs(ratio_delta["ratio_mean_delta"]) <= eps
            ),
            "current_snapshot_cannot_support_in_batch_many_env_dilution_claim": bool(scope_is_whole_batch),
        },
        "recommendation": {
            "next_focus": "target_present_train_side_selector_trace_or_snapshot",
            "reason": (
                "The current captured strict outer batch is a single env0 objective segment that already equals the "
                "whole-batch objective, and the branch-specific override leaves its pre/post/policy stages unchanged. "
                "This snapshot can rule out transfer occurring inside this captured batch, but it cannot quantify "
                "many-env mixing for env12 because env12 is not present here."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-interpret the strict pair2 exact update audit around the actual captured train-side scope."
    )
    parser.add_argument(
        "--exact-update-aggregation-audit-json",
        type=str,
        default=DEFAULT_EXACT_UPDATE_AGGREGATION_AUDIT_JSON,
    )
    parser.add_argument("--output-json", type=str, default=DEFAULT_OUTPUT_JSON)
    args = parser.parse_args()

    report = build_pair2_captured_scope_update_transfer_probe(
        exact_update_aggregation_audit_json=args.exact_update_aggregation_audit_json,
    )
    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
