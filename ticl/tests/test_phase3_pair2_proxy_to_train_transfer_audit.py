import json

from ticl.analysis.phase3_pair2_proxy_to_train_transfer_audit import (
    build_pair2_proxy_to_train_transfer_audit,
)


def _write(tmp_path, name: str, payload: dict) -> str:
    path = tmp_path / name
    path.write_text(json.dumps(payload))
    return str(path)


def test_build_pair2_proxy_to_train_transfer_audit_quantifies_mass_dilution(tmp_path):
    compare_json = _write(
        tmp_path,
        "compare.json",
        {
            "suite_context": {
                "train_suite_fingerprint": "trainfp",
                "heldout_suite_fingerprint": "heldfp",
                "n_samples": 2048,
                "eval_n_samples": 256,
                "single_eval_pos": 1946,
                "strict_fixed_env_mode": True,
                "deterministic_actor_sampling": True,
                "deterministic_batch_plan": True,
            },
            "comparison": {
                "pre_train_vs_zero_gap_delta": {"full_return_gap": 0.0, "suffix_return_gap": 0.0},
                "post_train_vs_zero_gap_delta": {"full_return_gap": -0.018, "suffix_return_gap": 0.0},
                "train_full_return_delta_change": -0.018,
                "train_suffix_return_delta_change": 0.0,
                "train_history_delta": {"histories_identical": True},
            },
        },
    )
    rollout_json = _write(
        tmp_path,
        "rollout.json",
        {
            "train_suite_summary": {"fingerprint": "trainfp"},
            "env_items": [
                {
                    "env_seed": 11,
                    "rollout_seed": 101,
                    "objective_token_count": 102,
                    "objective_terminal_reset_count": 0,
                    "normalized_actor_adv_abs_mass": 100.0,
                    "normalized_actor_adv_positive_mass": 55.0,
                    "normalized_actor_adv_negative_mass": 45.0,
                    "pre_vs_zero_full_gap": 1.0,
                    "pre_vs_zero_suffix_gap": 0.2,
                    "raw_value_return_corr": 0.1,
                },
                {
                    "env_seed": 12,
                    "rollout_seed": 102,
                    "objective_token_count": 102,
                    "objective_terminal_reset_count": 2,
                    "normalized_actor_adv_abs_mass": 120.0,
                    "normalized_actor_adv_positive_mass": 20.0,
                    "normalized_actor_adv_negative_mass": 100.0,
                    "pre_vs_zero_full_gap": 10.0,
                    "pre_vs_zero_suffix_gap": 5.0,
                    "raw_value_return_corr": 0.02,
                },
            ],
        },
    )
    counterfactual_json = _write(
        tmp_path,
        "counterfactual.json",
        {
            "comparison": {
                "observed_extension_block": {"row_count": 4},
                "counterfactual": {
                    "mean_start_gae_future_carry_norm": {"abs_shrink": 0.02}
                },
            },
            "conclusions": {
                "env12_mid_episode_extension_counterfactual_shrinks_negative_carry": True
            },
        },
    )

    report = build_pair2_proxy_to_train_transfer_audit(
        pair2_train_update_ab_compare_json=compare_json,
        pair2_train_rollout_quality_json=rollout_json,
        env12_counterfactual_json=counterfactual_json,
        target_env_index=1,
    )

    transfer = report["local_proxy_vs_batch_weighted_transfer"]
    assert transfer["block_row_count"] == 4
    assert transfer["block_token_fraction_of_batch"] == 4 / 204
    assert transfer["block_abs_future_carry_shrink_total"] == 0.08
    assert report["conclusions"]["local_proxy_improvement_is_real"] is True
    assert report["conclusions"]["local_proxy_improvement_is_too_small_relative_to_batch_mass"] is True
    assert report["conclusions"]["local_proxy_improvement_is_too_small_to_flip_env12_negative_mass"] is True
    assert report["conclusions"]["local_proxy_gain_did_not_move_suite_suffix_metric"] is True
    assert report["conclusions"]["target_side_train_update_changed_but_not_in_the_desired_direction"] is True
    assert report["conclusions"]["proxy_to_train_transfer_failure_is_not_surprising_given_mass_dilution"] is True


def test_build_pair2_proxy_to_train_transfer_audit_rejects_suite_mismatch(tmp_path):
    compare_json = _write(
        tmp_path,
        "compare.json",
        {
            "suite_context": {
                "train_suite_fingerprint": "compare-fp",
                "heldout_suite_fingerprint": "held-fp",
                "n_samples": 2048,
                "eval_n_samples": 256,
                "single_eval_pos": 1946,
                "strict_fixed_env_mode": True,
                "deterministic_actor_sampling": True,
                "deterministic_batch_plan": True,
            },
            "comparison": {
                "pre_train_vs_zero_gap_delta": {"full_return_gap": 0.0, "suffix_return_gap": 0.0},
                "post_train_vs_zero_gap_delta": {"full_return_gap": 0.0, "suffix_return_gap": 0.0},
                "train_full_return_delta_change": 0.0,
                "train_suffix_return_delta_change": 0.0,
                "train_history_delta": {"histories_identical": True},
            },
        },
    )
    rollout_json = _write(
        tmp_path,
        "rollout.json",
        {
            "train_suite_summary": {"fingerprint": "other-fp"},
            "env_items": [
                {
                    "env_seed": 0,
                    "rollout_seed": 0,
                    "objective_token_count": 102,
                    "objective_terminal_reset_count": 0,
                    "normalized_actor_adv_abs_mass": 1.0,
                    "normalized_actor_adv_positive_mass": 1.0,
                    "normalized_actor_adv_negative_mass": 0.0,
                    "pre_vs_zero_full_gap": 0.0,
                    "pre_vs_zero_suffix_gap": 0.0,
                    "raw_value_return_corr": 0.0,
                }
            ],
        },
    )
    counterfactual_json = _write(
        tmp_path,
        "counterfactual.json",
        {
            "comparison": {
                "observed_extension_block": {"row_count": 4},
                "counterfactual": {
                    "mean_start_gae_future_carry_norm": {"abs_shrink": 0.02}
                },
            },
            "conclusions": {
                "env12_mid_episode_extension_counterfactual_shrinks_negative_carry": True
            },
        },
    )

    try:
        build_pair2_proxy_to_train_transfer_audit(
            pair2_train_update_ab_compare_json=compare_json,
            pair2_train_rollout_quality_json=rollout_json,
            env12_counterfactual_json=counterfactual_json,
            target_env_index=0,
        )
    except ValueError as exc:
        assert "Suite fingerprint mismatch" in str(exc)
    else:
        raise AssertionError("expected suite mismatch to raise")
