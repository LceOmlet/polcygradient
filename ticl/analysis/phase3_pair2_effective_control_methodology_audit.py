import argparse
import json
import time
from pathlib import Path
from typing import Any

from ticl.analysis.phase2_guardrail import DEFAULT_PHASE2_SUMMARY, assert_phase2_green


DEFAULT_PRIOR_EXACT_AUDIT_JSON = (
    "/home/chen/RLPFN/artifacts/phase3_pair2_phase2_scale_exact_update_aggregation_audit.json"
)
DEFAULT_PRIOR_BATCH_TO_EPOCH_JSON = (
    "/home/chen/RLPFN/artifacts/phase3_pair2_phase2_scale_batch_to_epoch_aggregation_probe.json"
)
DEFAULT_PRIOR_LOCAL_STEP_DIRECTION_JSON = (
    "/home/chen/RLPFN/artifacts/phase3_pair2_phase2_scale_local_step_direction_probe.json"
)
DEFAULT_MICROBLOCK_COMPARE_JSON = (
    "/home/chen/RLPFN/artifacts/phase3_pair2_train_update_ab_compare_pack.json"
)
DEFAULT_EPISODE_COMPLETE_COMPARE_JSON = (
    "/home/chen/RLPFN/artifacts/phase3_pair2_episode_complete_train_update_ab_compare_pack.json"
)
DEFAULT_BRANCH_CLASS_PROBE_JSON = (
    "/home/chen/RLPFN/artifacts/phase3_pair2_branch_specific_extension_class_probe.json"
)
DEFAULT_OUTPUT_JSON = (
    "/home/chen/RLPFN/artifacts/phase3_pair2_effective_control_methodology_audit.json"
)
MATERIAL_EPS = 1e-4
SUBMILLISCALE_EPS = 1e-3


def _load_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def _require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _float(value: Any) -> float:
    return float(value)


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    denominator = float(denominator)
    if abs(denominator) <= 1e-12:
        return None
    return float(float(numerator) / denominator)


def _prior_scope_entry(branch_probe: dict[str, Any]) -> dict[str, Any]:
    scope_classes = dict(branch_probe["scope_classes"])
    if "prior_phase2_scale_target_scope" in scope_classes:
        return dict(scope_classes["prior_phase2_scale_target_scope"])
    if "prior_effective_looking_phase2_scale_scope" in scope_classes:
        return dict(scope_classes["prior_effective_looking_phase2_scale_scope"])
    raise KeyError("Expected prior phase2-scale scope entry in branch class probe.")


def build_pair2_effective_control_methodology_audit(
    *,
    phase2_summary_path: str,
    prior_exact_audit_json: str,
    prior_batch_to_epoch_json: str,
    prior_local_step_direction_json: str,
    microblock_compare_json: str,
    episode_complete_compare_json: str,
    branch_class_probe_json: str,
) -> dict[str, Any]:
    phase2_summary = assert_phase2_green(phase2_summary_path)
    prior_exact = _load_json(prior_exact_audit_json)
    prior_batch = _load_json(prior_batch_to_epoch_json)
    prior_local = _load_json(prior_local_step_direction_json)
    microblock_compare = _load_json(microblock_compare_json)
    episode_compare = _load_json(episode_complete_compare_json)
    branch_probe = _load_json(branch_class_probe_json)

    _require(
        str(branch_probe.get("probe_entry", "")) == "phase3_pair2_branch_specific_extension_class_probe",
        "Expected branch-specific extension class probe artifact.",
    )
    _require(
        bool(prior_exact["conclusions"]["trusted_target_scope_present_in_snapshot"]),
        "Prior phase2-scale exact audit no longer confirms trusted target presence.",
    )
    _require(
        bool(prior_batch["conclusions"]["target_scope_hits_exactly_one_outer_batch_per_epoch"]),
        "Prior phase2-scale batch-to-epoch probe no longer supports single-step in-epoch targeting.",
    )
    _require(
        bool(prior_batch["conclusions"]["target_scope_is_a_sequential_single_step_effect_within_epoch"]),
        "Prior phase2-scale batch-to-epoch probe no longer supports sequential single-step effect.",
    )
    _require(
        bool(prior_local["conclusions"]["full_main_loss_delta_is_not_opposed_to_actor_only_step"]),
        "Prior phase2-scale local-step probe no longer supports actor-only direction compatibility.",
    )
    _require(
        bool(microblock_compare["conclusions"]["target_side_compare_is_efficacy_eligible"]),
        "Current repaired microblock compare is not efficacy-eligible.",
    )
    _require(
        bool(episode_compare["conclusions"]["target_side_compare_is_efficacy_eligible"]),
        "Episode-complete compare is not efficacy-eligible.",
    )
    _require(
        bool(branch_probe["conclusions"]["scope_drift_is_a_better_explanation_for_the_surprise_than_new_shared_core_failure"]),
        "Branch-class probe does not support the current scope-drift explanation.",
    )

    microblock_suffix_delta = abs(
        _float(microblock_compare["comparison"]["train_suffix_return_delta_change"])
    )
    episode_suffix_delta = abs(
        _float(episode_compare["comparison"]["train_suffix_return_delta_change"])
    )
    microblock_full_delta = abs(
        _float(microblock_compare["comparison"]["train_full_return_delta_change"])
    )
    episode_full_delta = abs(
        _float(episode_compare["comparison"]["train_full_return_delta_change"])
    )
    microblock_policy_share = _float(
        branch_probe["scope_classes"]["current_repaired_microblock"]["policy_share_of_batch_negative_mass"]
    )
    episode_policy_share = _float(
        branch_probe["scope_classes"]["current_selector_matched_full_episode"][
            "policy_share_of_batch_negative_mass"
        ]
    )
    prior_policy_share = _float(_prior_scope_entry(branch_probe)["policy_share_of_batch_negative_mass"])

    methodology_local_mechanism_supported = bool(
        prior_exact["conclusions"]["trusted_target_scope_present_in_snapshot"]
        and prior_exact["conclusions"]["ratio_and_clipping_preserve_most_target_negative_mass"]
        and prior_batch["conclusions"]["target_scope_hits_exactly_one_outer_batch_per_epoch"]
        and prior_batch["conclusions"]["target_scope_is_a_sequential_single_step_effect_within_epoch"]
        and prior_local["conclusions"]["full_main_loss_delta_is_not_opposed_to_actor_only_step"]
    )

    episode_complete_above_numeric_floor = bool(episode_suffix_delta > MATERIAL_EPS)
    episode_complete_submilliscale = bool(episode_suffix_delta < SUBMILLISCALE_EPS)

    return {
        "audit_entry": "phase3_pair2_effective_control_methodology_audit",
        "config": {
            "material_eps": float(MATERIAL_EPS),
            "submilliscale_eps": float(SUBMILLISCALE_EPS),
            "phase2_summary_path": str(Path(phase2_summary_path).expanduser().resolve()),
            "prior_exact_audit_json": str(Path(prior_exact_audit_json).expanduser().resolve()),
            "prior_batch_to_epoch_json": str(Path(prior_batch_to_epoch_json).expanduser().resolve()),
            "prior_local_step_direction_json": str(
                Path(prior_local_step_direction_json).expanduser().resolve()
            ),
            "microblock_compare_json": str(Path(microblock_compare_json).expanduser().resolve()),
            "episode_complete_compare_json": str(
                Path(episode_complete_compare_json).expanduser().resolve()
            ),
            "branch_class_probe_json": str(Path(branch_class_probe_json).expanduser().resolve()),
        },
        "phase2_preflight": {
            "required": True,
            "phase2_all_checks_pass": bool(phase2_summary["validation"]["all_checks_pass"]),
            "phase2_pass_flags": dict(phase2_summary["pass_flags"]),
        },
        "evidence_layers": {
            "prior_effective_looking_method": {
                "scientific_for_local_mechanism_hypothesis": bool(methodology_local_mechanism_supported),
                "trusted_target_scope_present_in_snapshot": bool(
                    prior_exact["conclusions"]["trusted_target_scope_present_in_snapshot"]
                ),
                "single_step_in_epoch_supported": bool(
                    prior_batch["conclusions"]["target_scope_is_a_sequential_single_step_effect_within_epoch"]
                ),
                "main_loss_not_opposed_to_actor_only_step": bool(
                    prior_local["conclusions"]["full_main_loss_delta_is_not_opposed_to_actor_only_step"]
                ),
                "scientific_for_direct_microblock_efficacy": False,
            },
            "current_repaired_microblock_same_contract_compare": {
                "efficacy_eligible": bool(
                    microblock_compare["conclusions"]["target_side_compare_is_efficacy_eligible"]
                ),
                "noise_only": bool(microblock_compare["conclusions"]["target_side_effect_is_only_numeric_noise"]),
                "train_suffix_return_delta_change": float(microblock_suffix_delta),
                "train_full_return_delta_change": float(microblock_full_delta),
            },
            "current_episode_complete_same_contract_compare": {
                "efficacy_eligible": bool(
                    episode_compare["conclusions"]["target_side_compare_is_efficacy_eligible"]
                ),
                "noise_only": bool(episode_compare["conclusions"]["target_side_effect_is_only_numeric_noise"]),
                "train_suffix_return_delta_change": float(episode_suffix_delta),
                "train_full_return_delta_change": float(episode_full_delta),
                "runtime_mode_local_to_pair2": bool(
                    episode_compare["conclusions"]["runtime_mode_local_to_pair2"]
                ),
                "shared_core_unchanged": bool(episode_compare["conclusions"]["shared_core_unchanged"]),
            },
        },
        "scope_compare": {
            "scope_drift_supported": bool(
                branch_probe["conclusions"]["scope_drift_is_a_better_explanation_for_the_surprise_than_new_shared_core_failure"]
            ),
            "microblock_policy_share_of_batch_negative_mass": float(microblock_policy_share),
            "episode_complete_policy_share_of_batch_negative_mass": float(episode_policy_share),
            "prior_effective_looking_policy_share_of_batch_negative_mass": float(prior_policy_share),
            "episode_complete_vs_microblock_policy_share_ratio": _safe_ratio(
                episode_policy_share, microblock_policy_share
            ),
            "episode_complete_vs_prior_policy_share_ratio": _safe_ratio(
                episode_policy_share, prior_policy_share
            ),
            "microblock_vs_prior_policy_share_ratio": _safe_ratio(
                microblock_policy_share, prior_policy_share
            ),
        },
        "effect_scale_compare": {
            "episode_complete_vs_microblock_suffix_delta_ratio": _safe_ratio(
                episode_suffix_delta, microblock_suffix_delta
            ),
            "episode_complete_vs_microblock_full_delta_ratio": _safe_ratio(
                episode_full_delta, microblock_full_delta
            ),
            "episode_complete_above_numeric_floor": bool(episode_complete_above_numeric_floor),
            "episode_complete_submilliscale": bool(episode_complete_submilliscale),
        },
        "conclusions": {
            "old_methodology_is_scientific_for_local_mechanism_only": bool(
                methodology_local_mechanism_supported
            ),
            "old_methodology_is_not_sufficient_for_direct_microblock_efficacy": True,
            "scope_drift_was_the_missing_methodology_guard": bool(
                branch_probe["conclusions"]["scope_drift_is_a_better_explanation_for_the_surprise_than_new_shared_core_failure"]
            ),
            "episode_complete_same_contract_compare_is_the_first_correct_same_class_efficacy_test": bool(
                episode_compare["target_control"]["candidate_small_control_target"]
                == "env12_selector_matched_objective_episode"
                and episode_compare["conclusions"]["target_side_compare_is_efficacy_eligible"]
            ),
            "episode_complete_clears_microblock_noise_floor": bool(
                microblock_compare["conclusions"]["target_side_effect_is_only_numeric_noise"]
                and episode_suffix_delta > microblock_suffix_delta
            ),
            "episode_complete_is_above_numeric_floor_but_not_large_effect": bool(
                episode_complete_above_numeric_floor and episode_complete_submilliscale
            ),
            "small_control_large_effect_not_yet_supported": bool(episode_complete_submilliscale),
            "shared_core_rewrite_not_justified": bool(
                episode_compare["conclusions"]["runtime_mode_local_to_pair2"]
                and episode_compare["conclusions"]["shared_core_unchanged"]
            ),
            "next_search_space_should_remain_branch_specific_extension_families": True,
        },
        "recommendation": {
            "shared_core_policy": "freeze",
            "repeat_guard": (
                "Do not reuse the old phase2-scale mechanism probes as direct efficacy evidence for the repaired "
                "4-token microblock. Do not rerun the repaired microblock A/B as if the class question were still open."
            ),
            "next_narrow_direction": (
                "Stay inside shared-core + branch-specific extension decomposition. Treat episode-complete as the "
                "first correct same-class efficacy test, and only widen further within selector-matched branch-specific "
                "families if the goal is to clear the submilliscale effect regime."
            ),
        },
    }


def main() -> int:
    t0 = time.perf_counter()
    parser = argparse.ArgumentParser(
        description="Audit whether the earlier pair2 'effective-looking' control evidence was scientific, and what it actually proved."
    )
    parser.add_argument("--phase2-summary-path", type=str, default=DEFAULT_PHASE2_SUMMARY)
    parser.add_argument(
        "--prior-exact-audit-json",
        type=str,
        default=DEFAULT_PRIOR_EXACT_AUDIT_JSON,
    )
    parser.add_argument(
        "--prior-batch-to-epoch-json",
        type=str,
        default=DEFAULT_PRIOR_BATCH_TO_EPOCH_JSON,
    )
    parser.add_argument(
        "--prior-local-step-direction-json",
        type=str,
        default=DEFAULT_PRIOR_LOCAL_STEP_DIRECTION_JSON,
    )
    parser.add_argument(
        "--microblock-compare-json",
        type=str,
        default=DEFAULT_MICROBLOCK_COMPARE_JSON,
    )
    parser.add_argument(
        "--episode-complete-compare-json",
        type=str,
        default=DEFAULT_EPISODE_COMPLETE_COMPARE_JSON,
    )
    parser.add_argument(
        "--branch-class-probe-json",
        type=str,
        default=DEFAULT_BRANCH_CLASS_PROBE_JSON,
    )
    parser.add_argument("--output-json", type=str, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output_json).expanduser().resolve()
    if output_path.exists() and not bool(args.overwrite):
        print(output_path.read_text(encoding="utf-8"))
        return 0

    report = build_pair2_effective_control_methodology_audit(
        phase2_summary_path=args.phase2_summary_path,
        prior_exact_audit_json=args.prior_exact_audit_json,
        prior_batch_to_epoch_json=args.prior_batch_to_epoch_json,
        prior_local_step_direction_json=args.prior_local_step_direction_json,
        microblock_compare_json=args.microblock_compare_json,
        episode_complete_compare_json=args.episode_complete_compare_json,
        branch_class_probe_json=args.branch_class_probe_json,
    )
    report["config"]["runtime_wall_s"] = float(time.perf_counter() - t0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(output_path.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
