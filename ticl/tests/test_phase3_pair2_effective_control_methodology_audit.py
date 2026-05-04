import json

from ticl.analysis import phase3_pair2_effective_control_methodology_audit as audit_mod
from ticl.analysis.phase3_pair2_effective_control_methodology_audit import (
    build_pair2_effective_control_methodology_audit,
)


def _write(tmp_path, name: str, payload: dict) -> str:
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def test_pair2_effective_control_methodology_audit_distinguishes_mechanism_vs_efficacy(tmp_path, monkeypatch):
    phase2_summary_json = _write(tmp_path, "phase2_summary.json", {"unused": True})
    monkeypatch.setattr(
        audit_mod,
        "assert_phase2_green",
        lambda path: {
            "validation": {"all_checks_pass": True},
            "pass_flags": {"official_vs_monkey_numeric_match": True},
        },
    )
    prior_exact_json = _write(
        tmp_path,
        "prior_exact.json",
        {
            "conclusions": {
                "trusted_target_scope_present_in_snapshot": True,
                "ratio_and_clipping_preserve_most_target_negative_mass": True,
            }
        },
    )
    prior_batch_json = _write(
        tmp_path,
        "prior_batch.json",
        {
            "conclusions": {
                "target_scope_hits_exactly_one_outer_batch_per_epoch": True,
                "target_scope_is_a_sequential_single_step_effect_within_epoch": True,
            }
        },
    )
    prior_local_json = _write(
        tmp_path,
        "prior_local.json",
        {
            "conclusions": {
                "full_main_loss_delta_is_not_opposed_to_actor_only_step": True,
            }
        },
    )
    microblock_compare_json = _write(
        tmp_path,
        "microblock_compare.json",
        {
            "comparison": {
                "train_suffix_return_delta_change": 5.0e-05,
                "train_full_return_delta_change": 1.0e-05,
            },
            "conclusions": {
                "target_side_compare_is_efficacy_eligible": True,
                "target_side_effect_is_only_numeric_noise": True,
            },
        },
    )
    episode_compare_json = _write(
        tmp_path,
        "episode_compare.json",
        {
            "comparison": {
                "train_suffix_return_delta_change": 4.0e-04,
                "train_full_return_delta_change": 5.0e-04,
            },
            "conclusions": {
                "target_side_compare_is_efficacy_eligible": True,
                "target_side_effect_is_only_numeric_noise": False,
                "runtime_mode_local_to_pair2": True,
                "shared_core_unchanged": True,
            },
            "target_control": {
                "candidate_small_control_target": "env12_selector_matched_objective_episode",
            },
        },
    )
    branch_probe_json = _write(
        tmp_path,
        "branch_probe.json",
        {
            "probe_entry": "phase3_pair2_branch_specific_extension_class_probe",
            "scope_classes": {
                "current_repaired_microblock": {
                    "policy_share_of_batch_negative_mass": 0.03,
                },
                "current_selector_matched_full_episode": {
                    "policy_share_of_batch_negative_mass": 0.21,
                },
                "prior_effective_looking_phase2_scale_scope": {
                    "policy_share_of_batch_negative_mass": 0.18,
                },
            },
            "conclusions": {
                "scope_drift_is_a_better_explanation_for_the_surprise_than_new_shared_core_failure": True,
            },
        },
    )

    report = build_pair2_effective_control_methodology_audit(
        phase2_summary_path=phase2_summary_json,
        prior_exact_audit_json=prior_exact_json,
        prior_batch_to_epoch_json=prior_batch_json,
        prior_local_step_direction_json=prior_local_json,
        microblock_compare_json=microblock_compare_json,
        episode_complete_compare_json=episode_compare_json,
        branch_class_probe_json=branch_probe_json,
    )

    assert report["conclusions"]["old_methodology_is_scientific_for_local_mechanism_only"] is True
    assert report["conclusions"]["old_methodology_is_not_sufficient_for_direct_microblock_efficacy"] is True
    assert report["conclusions"]["episode_complete_same_contract_compare_is_the_first_correct_same_class_efficacy_test"] is True
    assert report["conclusions"]["episode_complete_clears_microblock_noise_floor"] is True
    assert report["conclusions"]["episode_complete_is_above_numeric_floor_but_not_large_effect"] is True
    assert report["conclusions"]["shared_core_rewrite_not_justified"] is True
    assert report["scope_compare"]["episode_complete_vs_microblock_policy_share_ratio"] == 7.0
    assert report["effect_scale_compare"]["episode_complete_vs_microblock_suffix_delta_ratio"] == 8.0


def test_pair2_effective_control_methodology_audit_rejects_missing_branch_class_support(tmp_path, monkeypatch):
    phase2_summary_json = _write(tmp_path, "phase2_summary.json", {"unused": True})
    monkeypatch.setattr(
        audit_mod,
        "assert_phase2_green",
        lambda path: {
            "validation": {"all_checks_pass": True},
            "pass_flags": {"official_vs_monkey_numeric_match": True},
        },
    )
    prior_exact_json = _write(
        tmp_path,
        "prior_exact.json",
        {
            "conclusions": {
                "trusted_target_scope_present_in_snapshot": True,
                "ratio_and_clipping_preserve_most_target_negative_mass": True,
            }
        },
    )
    prior_batch_json = _write(
        tmp_path,
        "prior_batch.json",
        {
            "conclusions": {
                "target_scope_hits_exactly_one_outer_batch_per_epoch": True,
                "target_scope_is_a_sequential_single_step_effect_within_epoch": True,
            }
        },
    )
    prior_local_json = _write(
        tmp_path,
        "prior_local.json",
        {"conclusions": {"full_main_loss_delta_is_not_opposed_to_actor_only_step": True}},
    )
    microblock_compare_json = _write(
        tmp_path,
        "microblock_compare.json",
        {
            "comparison": {
                "train_suffix_return_delta_change": 1.0e-05,
                "train_full_return_delta_change": 1.0e-05,
            },
            "conclusions": {
                "target_side_compare_is_efficacy_eligible": True,
                "target_side_effect_is_only_numeric_noise": True,
            },
        },
    )
    episode_compare_json = _write(
        tmp_path,
        "episode_compare.json",
        {
            "comparison": {
                "train_suffix_return_delta_change": 2.0e-04,
                "train_full_return_delta_change": 2.0e-04,
            },
            "conclusions": {
                "target_side_compare_is_efficacy_eligible": True,
                "target_side_effect_is_only_numeric_noise": False,
                "runtime_mode_local_to_pair2": True,
                "shared_core_unchanged": True,
            },
            "target_control": {
                "candidate_small_control_target": "env12_selector_matched_objective_episode",
            },
        },
    )
    branch_probe_json = _write(
        tmp_path,
        "branch_probe.json",
        {
            "probe_entry": "phase3_pair2_branch_specific_extension_class_probe",
            "scope_classes": {
                "current_repaired_microblock": {
                    "policy_share_of_batch_negative_mass": 0.03,
                },
                "current_selector_matched_full_episode": {
                    "policy_share_of_batch_negative_mass": 0.21,
                },
                "prior_effective_looking_phase2_scale_scope": {
                    "policy_share_of_batch_negative_mass": 0.18,
                },
            },
            "conclusions": {
                "scope_drift_is_a_better_explanation_for_the_surprise_than_new_shared_core_failure": False,
            },
        },
    )

    try:
        build_pair2_effective_control_methodology_audit(
            phase2_summary_path=phase2_summary_json,
            prior_exact_audit_json=prior_exact_json,
            prior_batch_to_epoch_json=prior_batch_json,
            prior_local_step_direction_json=prior_local_json,
            microblock_compare_json=microblock_compare_json,
            episode_complete_compare_json=episode_compare_json,
            branch_class_probe_json=branch_probe_json,
        )
    except ValueError as exc:
        assert "scope-drift explanation" in str(exc)
    else:
        raise AssertionError("expected missing scope-drift support to raise")
