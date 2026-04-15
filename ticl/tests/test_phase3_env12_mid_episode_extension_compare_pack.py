import json

from ticl.analysis.phase3_env12_mid_episode_extension_compare_pack import (
    build_env12_mid_episode_extension_compare_pack,
)


def _write(tmp_path, name: str, payload: dict) -> str:
    path = tmp_path / name
    path.write_text(json.dumps(payload))
    return str(path)


def test_build_env12_mid_episode_extension_compare_pack_reports_branch_specific_block(tmp_path):
    env12 = _write(
        tmp_path,
        "env12.json",
        {
            "config": {
                "suite_name": "pair2",
                "env_index": 12,
                "flip_mode": "future_carry_dominant",
            },
            "start_rows": [
                {
                    "dominant_negative_source": "baseline_bootstrap_semantic_mismatch",
                    "start_objective_local_index": 18,
                    "start_tokens_to_objective_episode_end": 38,
                    "start_raw_rollout_residual_norm": 0.8,
                    "start_gae_future_carry_norm": -0.06,
                    "weighted_delta_sum": -0.06,
                    "negative_source_shares": {"baseline_bootstrap_semantic_mismatch": 0.55, "terminal_reset_tail_semantic_mismatch": 0.45},
                },
                {
                    "dominant_negative_source": "baseline_bootstrap_semantic_mismatch",
                    "start_objective_local_index": 19,
                    "start_tokens_to_objective_episode_end": 37,
                    "start_raw_rollout_residual_norm": 0.79,
                    "start_gae_future_carry_norm": -0.07,
                    "weighted_delta_sum": -0.07,
                    "negative_source_shares": {"baseline_bootstrap_semantic_mismatch": 0.55, "terminal_reset_tail_semantic_mismatch": 0.45},
                },
                {
                    "dominant_negative_source": "baseline_bootstrap_semantic_mismatch",
                    "start_objective_local_index": 20,
                    "start_tokens_to_objective_episode_end": 36,
                    "start_raw_rollout_residual_norm": 0.78,
                    "start_gae_future_carry_norm": -0.08,
                    "weighted_delta_sum": -0.08,
                    "negative_source_shares": {"baseline_bootstrap_semantic_mismatch": 0.55, "terminal_reset_tail_semantic_mismatch": 0.45},
                },
                {
                    "dominant_negative_source": "baseline_bootstrap_semantic_mismatch",
                    "start_objective_local_index": 21,
                    "start_tokens_to_objective_episode_end": 35,
                    "start_raw_rollout_residual_norm": 0.77,
                    "start_gae_future_carry_norm": -0.09,
                    "weighted_delta_sum": -0.09,
                    "negative_source_shares": {"baseline_bootstrap_semantic_mismatch": 0.55, "terminal_reset_tail_semantic_mismatch": 0.45},
                },
                {
                    "dominant_negative_source": "baseline_bootstrap_semantic_mismatch",
                    "start_objective_local_index": 100,
                    "start_tokens_to_objective_episode_end": 1,
                    "start_raw_rollout_residual_norm": 0.06,
                    "start_gae_future_carry_norm": -0.07,
                    "weighted_delta_sum": -0.07,
                    "negative_source_shares": {"baseline_bootstrap_semantic_mismatch": 1.0},
                },
            ],
        },
    )

    report = build_env12_mid_episode_extension_compare_pack(env12_json=env12)

    assert report["audit_entry"] == "phase3_env12_mid_episode_extension_compare_pack"
    assert report["comparison"]["extension_block"]["row_count"] == 4
    assert report["comparison"]["extension_block"]["is_contiguous"] is True
    assert report["comparison"]["extension_block"]["negative_source_shares_is_uniform"] is True
    assert report["comparison"]["extension_block"]["dominant_source"] == "baseline_bootstrap_semantic_mismatch"
    assert report["conclusions"]["env12_mid_episode_extension_is_branch_specific"] is True
    assert report["conclusions"]["env12_mid_episode_extension_root_source_is_baseline_bootstrap_semantic_mismatch"] is True
    assert report["conclusions"]["env12_mid_episode_extension_has_no_internal_split_supported"] is True
    assert report["recommendation"]["candidate_small_control_target"] == "env12_mid_episode_extension_block"


def test_build_env12_mid_episode_extension_compare_pack_rejects_wrong_suite(tmp_path):
    bad_env12 = _write(
        tmp_path,
        "bad.json",
        {
            "config": {
                "suite_name": "pair1",
                "env_index": 12,
                "flip_mode": "future_carry_dominant",
            },
            "start_rows": [
                {
                    "dominant_negative_source": "terminal_reset_tail_semantic_mismatch",
                    "start_objective_local_index": 18,
                    "start_tokens_to_objective_episode_end": 38,
                    "start_raw_rollout_residual_norm": 0.8,
                    "start_gae_future_carry_norm": -0.06,
                    "weighted_delta_sum": -0.06,
                    "negative_source_shares": {"baseline_bootstrap_semantic_mismatch": 0.55, "terminal_reset_tail_semantic_mismatch": 0.45},
                }
            ],
        },
    )

    try:
        build_env12_mid_episode_extension_compare_pack(env12_json=bad_env12)
    except ValueError as exc:
        assert "Unexpected suite_name" in str(exc)
    else:
        raise AssertionError("expected suite validation to fail")
