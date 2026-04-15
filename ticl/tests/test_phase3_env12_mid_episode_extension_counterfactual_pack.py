import json
import sys

from ticl.analysis.phase3_env12_mid_episode_extension_counterfactual_pack import (
    build_env12_mid_episode_extension_counterfactual_pack,
)


def _write(tmp_path, name: str, payload: dict) -> str:
    path = tmp_path / name
    path.write_text(json.dumps(payload))
    return str(path)


def test_build_env12_mid_episode_extension_counterfactual_pack_reports_material_shrink(tmp_path):
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
        },
    )

    report = build_env12_mid_episode_extension_counterfactual_pack(
        env12_compare_pack_json=compare_pack
    )

    assert report["audit_entry"] == "phase3_env12_mid_episode_extension_counterfactual_pack"
    carry = report["comparison"]["counterfactual"]["mean_start_gae_future_carry_norm"]
    residual = report["comparison"]["counterfactual"]["mean_start_raw_rollout_residual_norm"]
    weighted = report["comparison"]["counterfactual"]["mean_weighted_delta_sum"]
    assert carry["abs_shrink"] > 0.0
    assert carry["counterfactual_over_observed_ratio"] < 1.0
    assert residual["abs_shrink"] > 0.0
    assert residual["counterfactual_over_observed_ratio"] < 1.0
    assert weighted["abs_shrink"] > 0.0
    assert report["conclusions"]["env12_mid_episode_extension_counterfactual_shrinks_negative_carry"] is True
    assert report["conclusions"]["env12_mid_episode_extension_counterfactual_supports_branch_specific_small_control"] is True
    assert report["recommendation"]["candidate_small_control_target"] == "env12_mid_episode_extension_block"


def test_build_env12_mid_episode_extension_counterfactual_pack_rejects_wrong_input(tmp_path):
    bad_compare_pack = _write(
        tmp_path,
        "bad.json",
        {
            "audit_entry": "phase3_env12_mid_episode_extension_compare_pack",
            "comparison": {},
        },
    )

    try:
        build_env12_mid_episode_extension_counterfactual_pack(
            env12_compare_pack_json=bad_compare_pack
        )
    except KeyError:
        pass
    else:
        raise AssertionError("Expected missing comparison block to fail.")


def test_phase3_env12_mid_episode_extension_counterfactual_pack_reuses_existing_output(
    tmp_path, monkeypatch, capsys
):
    output_path = tmp_path / "phase3_env12_mid_episode_extension_counterfactual_pack.json"
    payload = {"ok": True, "kind": "phase3_env12_mid_episode_extension_counterfactual_pack"}
    output_path.write_text(json.dumps(payload))

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "phase3_env12_mid_episode_extension_counterfactual_pack.py",
            "--output-json",
            str(output_path),
        ],
    )

    from ticl.analysis import phase3_env12_mid_episode_extension_counterfactual_pack as pack_mod

    assert pack_mod.main() == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == payload
