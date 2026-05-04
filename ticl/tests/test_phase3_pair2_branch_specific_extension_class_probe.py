import json

from ticl.analysis.phase3_pair2_branch_specific_extension_class_probe import (
    build_pair2_branch_specific_extension_class_probe,
)


def _write(tmp_path, name: str, payload: dict) -> str:
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def _build_current_baseline_snapshot_payload() -> dict:
    return {
        "objective_mask": [1, 1, 1, 1, 1, 1, 1, 1],
        "flat_env_indices": [12, 12, 12, 12, 12, 12, 9, 9],
        "flat_objective_episode_indices": [4, 4, 4, 4, 4, 4, 1, 1],
        "flat_objective_episode_positions": [0, 1, 2, 3, 4, 5, 0, 1],
        "pre_normalization_actor_advantages": [-1.0, -2.0, -1.5, -1.5, -2.0, -2.0, -1.0, -1.0],
        "post_normalization_advantages": [-1.0, -2.0, -1.5, -1.5, -2.0, -2.0, -1.0, -1.0],
        "clipped_objective": [-1.0, -2.0, -1.5, -1.5, -2.0, -2.0, -1.0, -1.0],
    }


def _build_current_exact_audit_payload() -> dict:
    return {
        "strict_contract": {
            "outer_batch_idx": 13,
            "runtime_suite_name": "pair2",
        },
        "target_block": {
            "env_index": 12,
            "block_token_count": 1,
        },
        "baseline": {
            "policy_num_from_clipped_objective": {
                "block": {
                    "negative_mass": 1.0,
                },
                "block_share_of_batch_negative_mass": 0.08333333333333333,
            }
        },
    }


def _build_current_epoch_dilution_payload() -> dict:
    return {
        "epoch_layout": {
            "target_outer_batch_idx": 13,
            "total_outer_batches": 16,
            "objective_total_per_outer_batch": 8,
            "remaining_outer_batches_after_target": 3,
        },
        "target_scope": {
            "selector": {
                "target_objective_episode_index": 4,
                "target_objective_position_start": 0,
                "target_objective_position_end": 0,
            }
        },
    }


def _build_prior_exact_audit_payload() -> dict:
    return {
        "strict_contract": {
            "runtime_suite_name": "pair2",
        },
        "target_scope": {
            "selector": {
                "outer_batch_idx": 13,
                "env_index": 12,
                "objective_episode_index": 10,
                "objective_position_start": 0,
                "objective_position_end": 3,
            },
            "token_count": 4,
        },
        "stage_transition": {
            "target_negative_mass": {
                "policy_num": 3.0,
            },
            "target_share_of_batch_negative_mass": {
                "policy_num": 0.25,
            },
        },
    }


def _build_prior_batch_to_epoch_payload() -> dict:
    return {
        "epoch_layout": {
            "total_outer_batches": 16,
            "objective_total_per_outer_batch": 8,
        },
        "target_scope_epoch_context": {
            "target_outer_batch_idx": 13,
            "target_batches_after_in_epoch": 3,
            "uniform_step_weight_epoch_share_heuristic": 0.015625,
        },
    }


def test_branch_specific_extension_class_probe_identifies_scope_drift_and_episode_complete_candidate(tmp_path):
    current_baseline_snapshot_json = _write(
        tmp_path, "current_baseline.json", _build_current_baseline_snapshot_payload()
    )
    current_exact_audit_json = _write(
        tmp_path, "current_exact.json", _build_current_exact_audit_payload()
    )
    current_epoch_dilution_json = _write(
        tmp_path, "current_epoch.json", _build_current_epoch_dilution_payload()
    )
    prior_exact_audit_json = _write(tmp_path, "prior_exact.json", _build_prior_exact_audit_payload())
    prior_batch_to_epoch_json = _write(
        tmp_path, "prior_batch.json", _build_prior_batch_to_epoch_payload()
    )

    report = build_pair2_branch_specific_extension_class_probe(
        current_baseline_snapshot_json=current_baseline_snapshot_json,
        current_exact_audit_json=current_exact_audit_json,
        current_epoch_dilution_json=current_epoch_dilution_json,
        prior_exact_audit_json=prior_exact_audit_json,
        prior_batch_to_epoch_json=prior_batch_to_epoch_json,
    )

    assert report["cross_scope_compare"]["old_vs_current_same_outer_batch_timing"] is True
    assert abs(
        report["scope_classes"]["current_repaired_microblock"]["policy_share_of_batch_negative_mass"]
        - (1.0 / 12.0)
    ) <= 1e-12
    assert abs(
        report["scope_classes"]["current_selector_matched_full_episode"]["policy_share_of_batch_negative_mass"]
        - (10.0 / 12.0)
    ) <= 1e-12
    assert report["scope_classes"]["current_selector_matched_full_episode"]["microblock_relation"][
        "block_share_of_episode_policy_negative_mass"
    ] == 0.1
    assert report["conclusions"][
        "previous_effective_looking_scope_and_current_microblock_are_not_the_same_size_class"
    ] is True
    assert report["conclusions"][
        "selector_matched_full_episode_is_scale_matched_to_or_larger_than_the_prior_effective_scope"
    ] is True
    assert report["conclusions"][
        "scope_drift_is_a_better_explanation_for_the_surprise_than_new_shared_core_failure"
    ] is True
    assert report["conclusions"][
        "smallest_current_branch_specific_extension_class_that_can_clear_the_single_step_floor_is_episode_complete"
    ] is True


def test_branch_specific_extension_class_probe_rejects_missing_block_inside_selected_episode(tmp_path):
    current_baseline = _build_current_baseline_snapshot_payload()
    current_baseline["flat_objective_episode_positions"][0] = 8
    current_baseline_snapshot_json = _write(tmp_path, "current_baseline.json", current_baseline)
    current_exact_audit_json = _write(
        tmp_path, "current_exact.json", _build_current_exact_audit_payload()
    )
    current_epoch_dilution_json = _write(
        tmp_path, "current_epoch.json", _build_current_epoch_dilution_payload()
    )
    prior_exact_audit_json = _write(tmp_path, "prior_exact.json", _build_prior_exact_audit_payload())
    prior_batch_to_epoch_json = _write(
        tmp_path, "prior_batch.json", _build_prior_batch_to_epoch_payload()
    )

    try:
        build_pair2_branch_specific_extension_class_probe(
            current_baseline_snapshot_json=current_baseline_snapshot_json,
            current_exact_audit_json=current_exact_audit_json,
            current_epoch_dilution_json=current_epoch_dilution_json,
            prior_exact_audit_json=prior_exact_audit_json,
            prior_batch_to_epoch_json=prior_batch_to_epoch_json,
        )
    except ValueError as exc:
        assert "target block is absent" in str(exc)
    else:
        raise AssertionError("expected missing repaired block inside selected episode to raise")
