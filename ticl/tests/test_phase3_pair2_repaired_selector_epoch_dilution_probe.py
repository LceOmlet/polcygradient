import json

from ticl.analysis.phase3_pair2_repaired_selector_epoch_dilution_probe import (
    build_pair2_repaired_selector_epoch_dilution_probe,
)


def _write(tmp_path, name: str, payload: dict) -> str:
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def _build_selector_trace_payload() -> dict:
    rows = []
    for outer_batch_idx in range(1, 17):
        rows.append(
            {
                "epoch_idx": 1,
                "outer_batch_idx": outer_batch_idx,
                "total_outer_batches": 16,
                "objective_total": 192,
                "objective_env_indices_present": [outer_batch_idx - 1],
                "target_match_count": 4 if outer_batch_idx == 13 else 0,
            }
        )
    return {
        "conclusions": {
            "target_present_in_baseline": True,
            "target_present_in_override": True,
            "current_selector_definition_cannot_fire_under_trusted_contract": False,
        },
        "baseline": {
            "selector_trace": {
                "rows": rows,
            }
        },
        "target_selector": {
            "target_outer_batch_idx": 13,
            "target_env_index": 12,
            "target_objective_episode_index": 4,
            "target_objective_position_start": 15,
            "target_objective_position_end": 18,
        },
    }


def _build_exact_audit_payload() -> dict:
    return {
        "conclusions": {
            "requested_target_block_present_in_captured_outer_batch": True,
        },
        "strict_contract": {
            "outer_batch_idx": 13,
            "objective_total": 192,
            "runtime_suite_name": "pair2",
        },
        "target_block": {
            "env_index": 12,
            "block_token_count": 4,
        },
        "delta": {
            "pre_normalization_actor_advantages": {
                "block_negative_mass_shrink": 0.3358079493045807,
                "block_share_of_batch_negative_mass_delta": -0.007103350007505348,
            },
            "policy_num_from_clipped_objective": {
                "block_negative_mass_shrink": 0.28959451615810394,
                "block_share_of_batch_negative_mass_delta": -0.007103594506918991,
            },
        },
    }


def _build_compare_pack_payload() -> dict:
    return {
        "shared_contract_context": {
            "checkpoint_path": "/tmp/checkpoint.cpkt",
            "train_suite_path": "/tmp/train_suite.pt",
            "single_eval_pos": 64,
            "n_samples": 256,
            "strict_fixed_env_mode": True,
            "deterministic_actor_sampling": True,
            "deterministic_batch_plan": True,
            "strict_native_rollout": True,
            "ppo_restore_validation_policy_head_state": True,
        },
        "conclusions": {
            "target_side_compare_is_efficacy_eligible": True,
            "target_side_effect_is_only_numeric_noise": True,
            "target_side_train_update_changes_trajectory": False,
        },
        "comparison": {
            "train_history_delta": {
                "histories_identical": True,
                "max_abs_metric_delta": 0.0,
            },
            "train_full_return_delta_change": 5.7220458984375e-06,
            "train_suffix_return_delta_change": 5.8650970458984375e-05,
        },
    }


def test_repaired_selector_epoch_dilution_probe_quantifies_single_late_step_dilution(tmp_path):
    selector_trace_json = _write(tmp_path, "trace.json", _build_selector_trace_payload())
    exact_audit_json = _write(tmp_path, "exact.json", _build_exact_audit_payload())
    compare_pack_json = _write(tmp_path, "compare.json", _build_compare_pack_payload())

    report = build_pair2_repaired_selector_epoch_dilution_probe(
        selector_trace_json=selector_trace_json,
        exact_audit_json=exact_audit_json,
        compare_pack_json=compare_pack_json,
    )

    assert report["epoch_layout"]["total_outer_batches"] == 16
    assert report["epoch_layout"]["target_outer_batch_idx"] == 13
    assert report["epoch_layout"]["remaining_outer_batches_after_target"] == 3
    assert abs(report["target_scope"]["target_token_fraction_of_target_batch"] - (4.0 / 192.0)) <= 1e-12
    assert abs(report["target_scope"]["target_token_fraction_of_epoch_objective"] - (4.0 / (192.0 * 16.0))) <= 1e-12
    assert abs(report["local_to_epoch_dilution"]["policy_batch_negative_mass_share_epoch_uniform_heuristic"] - (0.007103594506918991 / 16.0)) <= 1e-12
    assert report["conclusions"]["target_scope_perturbs_exactly_one_outer_batch_per_epoch"] is True
    assert report["conclusions"]["target_scope_enters_only_in_the_final_quarter_of_epoch"] is True
    assert report["conclusions"]["epoch_weighted_policy_negative_mass_share_perturbation_is_subpermille"] is True
    assert report["conclusions"]["local_shrink_to_train_noise_is_explained_by_single_late_step_dilution"] is True


def test_repaired_selector_epoch_dilution_probe_rejects_selector_vs_exact_mismatch(tmp_path):
    selector_payload = _build_selector_trace_payload()
    selector_payload["target_selector"]["target_outer_batch_idx"] = 12
    selector_trace_json = _write(tmp_path, "trace.json", selector_payload)
    exact_audit_json = _write(tmp_path, "exact.json", _build_exact_audit_payload())
    compare_pack_json = _write(tmp_path, "compare.json", _build_compare_pack_payload())

    try:
        build_pair2_repaired_selector_epoch_dilution_probe(
            selector_trace_json=selector_trace_json,
            exact_audit_json=exact_audit_json,
            compare_pack_json=compare_pack_json,
        )
    except ValueError as exc:
        assert "target_outer_batch_idx" in str(exc)
    else:
        raise AssertionError("expected selector vs exact outer-batch mismatch to raise")
