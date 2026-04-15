import json
import sys

from ticl.analysis.phase3_env12_mid_episode_extension_runtime_control_probe import (
    build_env12_mid_episode_extension_runtime_control_probe,
)


def _write(tmp_path, name: str, payload: dict) -> str:
    path = tmp_path / name
    path.write_text(json.dumps(payload))
    return str(path)


def test_build_env12_mid_episode_extension_runtime_control_probe_reports_minimal_runtime_control(tmp_path):
    compare_pack = _write(
        tmp_path,
        "compare.json",
        {
            "audit_entry": "phase3_env12_mid_episode_extension_compare_pack",
            "comparison": {
                "extension_block": {
                    "dominant_source": "baseline_bootstrap_semantic_mismatch",
                    "future_chain_step_count": [38, 37, 36, 35],
                    "is_contiguous": True,
                    "local_indices": [18, 19, 20, 21],
                    "mean_start_gae_future_carry_norm": -0.0901851886883378,
                    "mean_start_raw_rollout_residual_norm": 0.80408083907995,
                    "mean_weighted_delta_sum": -0.09018484419889922,
                    "negative_source_shares_is_uniform": True,
                    "negative_source_shares_max_abs_deviation": 1.1102230246251565e-16,
                    "negative_source_shares_reference": {
                        "baseline_bootstrap_semantic_mismatch": 0.5491065753155423,
                        "terminal_reset_tail_semantic_mismatch": 0.45089342468445764,
                    },
                    "monotone_tte_decreasing": True,
                    "monotone_weighted_delta_more_negative": True,
                    "row_count": 4,
                    "tokens_to_objective_episode_end": [38, 37, 36, 35],
                },
                "terminal_adjacent_core": {
                    "dominant_source": "baseline_bootstrap_semantic_mismatch",
                    "future_chain_step_count": [1],
                    "is_contiguous": True,
                    "local_indices": [100],
                    "mean_start_gae_future_carry_norm": -0.06972909942269324,
                    "mean_start_raw_rollout_residual_norm": 0.0648687719972905,
                    "mean_weighted_delta_sum": -0.06972909895965959,
                    "negative_source_shares_is_uniform": True,
                    "negative_source_shares_max_abs_deviation": 0.0,
                    "negative_source_shares_reference": {
                        "baseline_bootstrap_semantic_mismatch": 1.0,
                    },
                    "monotone_tte_decreasing": True,
                    "monotone_weighted_delta_more_negative": True,
                    "row_count": 1,
                    "tokens_to_objective_episode_end": [1],
                },
            },
            "conclusions": {
                "env12_mid_episode_extension_has_no_internal_split_supported": True,
            },
        },
    )
    counterfactual_pack = _write(
        tmp_path,
        "counterfactual.json",
        {
            "audit_entry": "phase3_env12_mid_episode_extension_counterfactual_pack",
            "comparison": {
                "observed_extension_block": {
                    "dominant_source": "baseline_bootstrap_semantic_mismatch",
                    "future_chain_step_count": [38, 37, 36, 35],
                    "is_contiguous": True,
                    "local_indices": [18, 19, 20, 21],
                    "mean_start_gae_future_carry_norm": -0.0901851886883378,
                    "mean_start_raw_rollout_residual_norm": 0.80408083907995,
                    "mean_weighted_delta_sum": -0.09018484419889922,
                    "negative_source_shares_is_uniform": True,
                    "negative_source_shares_max_abs_deviation": 1.1102230246251565e-16,
                    "negative_source_shares_reference": {
                        "baseline_bootstrap_semantic_mismatch": 0.5491065753155423,
                        "terminal_reset_tail_semantic_mismatch": 0.45089342468445764,
                    },
                    "monotone_tte_decreasing": True,
                    "monotone_weighted_delta_more_negative": True,
                    "row_count": 4,
                    "tokens_to_objective_episode_end": [38, 37, 36, 35],
                },
                "counterfactual_anchor_terminal_adjacent_core": {
                    "dominant_source": "baseline_bootstrap_semantic_mismatch",
                    "future_chain_step_count": [1],
                    "is_contiguous": True,
                    "local_indices": [100],
                    "mean_start_gae_future_carry_norm": -0.06972909942269324,
                    "mean_start_raw_rollout_residual_norm": 0.0648687719972905,
                    "mean_weighted_delta_sum": -0.06972909895965959,
                    "negative_source_shares_is_uniform": True,
                    "negative_source_shares_max_abs_deviation": 0.0,
                    "negative_source_shares_reference": {
                        "baseline_bootstrap_semantic_mismatch": 1.0,
                    },
                    "monotone_tte_decreasing": True,
                    "monotone_weighted_delta_more_negative": True,
                    "row_count": 1,
                    "tokens_to_objective_episode_end": [1],
                },
                "counterfactual": {
                    "mean_start_gae_future_carry_norm": {
                        "abs_shrink": 0.02045608926564456,
                        "counterfactual_abs": 0.06972909942269324,
                        "counterfactual_over_observed_ratio": 0.7731768424154796,
                        "observed_abs": 0.0901851886883378,
                        "observed_over_counterfactual_ratio": 1.2933651722882735,
                    },
                    "mean_start_raw_rollout_residual_norm": {
                        "abs_shrink": 0.7392120670826595,
                        "counterfactual_abs": 0.0648687719972905,
                        "counterfactual_over_observed_ratio": 0.08067444073349021,
                        "observed_abs": 0.80408083907995,
                        "observed_over_counterfactual_ratio": 12.39549962674699,
                    },
                    "mean_weighted_delta_sum": {
                        "abs_shrink": 0.020455745239239626,
                        "counterfactual_abs": 0.06972909895965959,
                        "counterfactual_over_observed_ratio": 0.7731797906738602,
                        "observed_abs": 0.09018484419889922,
                        "observed_over_counterfactual_ratio": 1.2933602404797155,
                    },
                },
            },
            "conclusions": {
                "env12_mid_episode_extension_counterfactual_supports_branch_specific_small_control": True,
            },
        },
    )

    report = build_env12_mid_episode_extension_runtime_control_probe(
        env12_compare_pack_json=compare_pack,
        env12_counterfactual_pack_json=counterfactual_pack,
    )

    assert report["audit_entry"] == "phase3_env12_mid_episode_extension_runtime_control_probe"
    assert report["runtime_semantic_checks"]["payloads_match"] is True
    assert report["runtime_semantic_checks"]["can_be_minimally_realized_as_runtime_control"] is True
    assert report["runtime_semantic_checks"]["requires_shared_core_edit"] is False
    assert report["runtime_semantic_checks"]["requires_branch_specific_scope_only"] is True
    assert report["runtime_semantic_checks"]["no_finer_split_supported"] is True
    assert report["recommendation"]["runtime_control_shape"] == "contiguous_four_token_branch_specific_gate"
    assert report["material_counterfactual"]["mean_start_gae_future_carry_norm"]["abs_shrink"] > 0.0
    assert report["material_counterfactual"]["mean_start_raw_rollout_residual_norm"]["abs_shrink"] > 0.0
    assert report["material_counterfactual"]["mean_weighted_delta_sum"]["abs_shrink"] > 0.0


def test_build_env12_mid_episode_extension_runtime_control_probe_rejects_wrong_input(tmp_path):
    bad_counterfactual = _write(
        tmp_path,
        "bad.json",
        {
            "audit_entry": "phase3_env12_mid_episode_extension_compare_pack",
            "comparison": {},
        },
    )
    compare_pack = _write(
        tmp_path,
        "compare.json",
        {
            "audit_entry": "phase3_env12_mid_episode_extension_compare_pack",
            "comparison": {},
            "conclusions": {
                "env12_mid_episode_extension_has_no_internal_split_supported": True,
            },
        },
    )

    try:
        build_env12_mid_episode_extension_runtime_control_probe(
            env12_compare_pack_json=compare_pack,
            env12_counterfactual_pack_json=bad_counterfactual,
        )
    except ValueError as exc:
        assert "Expected phase3_env12_mid_episode_extension_counterfactual_pack artifact." in str(exc)
    else:
        raise AssertionError("Expected artifact validation to fail.")


def test_phase3_env12_mid_episode_extension_runtime_control_probe_reuses_existing_output(
    tmp_path, monkeypatch, capsys
):
    output_path = tmp_path / "phase3_env12_mid_episode_extension_runtime_control_probe.json"
    payload = {"ok": True, "kind": "phase3_env12_mid_episode_extension_runtime_control_probe"}
    output_path.write_text(json.dumps(payload))

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "phase3_env12_mid_episode_extension_runtime_control_probe.py",
            "--output-json",
            str(output_path),
        ],
    )

    from ticl.analysis import phase3_env12_mid_episode_extension_runtime_control_probe as probe_mod

    assert probe_mod.main() == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == payload
