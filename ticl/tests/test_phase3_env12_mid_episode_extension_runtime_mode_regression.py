import json

from ticl.analysis.phase3_env12_mid_episode_extension_runtime_mode_regression import (
    build_env12_mid_episode_extension_runtime_mode_regression,
)


def _write(tmp_path, name: str, payload: dict) -> str:
    path = tmp_path / name
    path.write_text(json.dumps(payload))
    return str(path)


def test_build_env12_mid_episode_extension_runtime_mode_regression_matches_counterfactual_abs(tmp_path):
    runtime_control = _write(
        tmp_path,
        "runtime_control.json",
        {
            "audit_entry": "phase3_env12_mid_episode_extension_runtime_control_probe",
            "control_block": {
                "candidate_small_control_target": "env12_mid_episode_extension_block",
                "runtime_control_granularity": "contiguous_four_token_block",
                "branch_specific": True,
                "row_count": 4,
                "local_indices": [18, 19, 20, 21],
                "tokens_to_objective_episode_end": [38, 37, 36, 35],
                "future_chain_step_count": [38, 37, 36, 35],
                "dominant_source": "baseline_bootstrap_semantic_mismatch",
                "negative_source_shares_reference": {
                    "baseline_bootstrap_semantic_mismatch": 0.5491065753155423,
                    "terminal_reset_tail_semantic_mismatch": 0.45089342468445764,
                },
            },
            "material_counterfactual": {
                "mean_start_gae_future_carry_norm": {
                    "observed_abs": 0.0901851886883378,
                    "counterfactual_abs": 0.06972909942269324,
                    "abs_shrink": 0.02045608926564456,
                    "counterfactual_over_observed_ratio": 0.7731768424154796,
                    "observed_over_counterfactual_ratio": 1.2933651722882735,
                },
                "mean_start_raw_rollout_residual_norm": {
                    "observed_abs": 0.80408083907995,
                    "counterfactual_abs": 0.0648687719972905,
                    "abs_shrink": 0.7392120670826595,
                    "counterfactual_over_observed_ratio": 0.08067444073349021,
                    "observed_over_counterfactual_ratio": 12.39549962674699,
                },
                "mean_weighted_delta_sum": {
                    "observed_abs": 0.09018484419889922,
                    "counterfactual_abs": 0.06972909895965959,
                    "abs_shrink": 0.020455745239239626,
                    "counterfactual_over_observed_ratio": 0.7731797906738602,
                    "observed_over_counterfactual_ratio": 1.2933602404797155,
                },
            },
        },
    )
    counterfactual = _write(
        tmp_path,
        "counterfactual.json",
        {
            "audit_entry": "phase3_env12_mid_episode_extension_counterfactual_pack",
            "comparison": {
                "counterfactual": {
                    "mean_start_gae_future_carry_norm": {
                        "observed_abs": 0.0901851886883378,
                        "counterfactual_abs": 0.06972909942269324,
                        "abs_shrink": 0.02045608926564456,
                        "counterfactual_over_observed_ratio": 0.7731768424154796,
                        "observed_over_counterfactual_ratio": 1.2933651722882735,
                    }
                }
            },
        },
    )

    report = build_env12_mid_episode_extension_runtime_mode_regression(
        env12_runtime_control_json=runtime_control,
        env12_counterfactual_pack_json=counterfactual,
    )

    assert report["audit_entry"] == "phase3_env12_mid_episode_extension_runtime_mode_regression"
    assert report["runtime_mode_name"] == "tokenwise_scale_env12_mid_episode_extension_block"
    assert report["conclusions"]["branch_specific_gate_supported_in_runtime"] is True
    assert report["conclusions"]["env12_mid_episode_extension_block_reproduces_counterfactual_carry_shrink"] is True
    assert abs(report["regression"]["block_after_abs"] - 0.06972909942269324) < 1e-6
    assert abs(report["regression"]["abs_shrink"] - 0.02045608926564456) < 1e-6
    assert report["regression"]["shared_core_unchanged"] is True


def test_build_env12_mid_episode_extension_runtime_mode_regression_rejects_wrong_artifact(tmp_path):
    bad_runtime_control = _write(
        tmp_path,
        "bad.json",
        {"audit_entry": "wrong", "control_block": {}},
    )
    bad_counterfactual = _write(
        tmp_path,
        "bad_counterfactual.json",
        {"audit_entry": "wrong", "comparison": {}},
    )
    try:
        build_env12_mid_episode_extension_runtime_mode_regression(
            env12_runtime_control_json=bad_runtime_control,
            env12_counterfactual_pack_json=bad_counterfactual,
        )
    except ValueError as exc:
        assert "Expected phase3_env12_mid_episode_extension_runtime_control_probe artifact." in str(exc)
    else:
        raise AssertionError("Expected artifact validation to fail.")
