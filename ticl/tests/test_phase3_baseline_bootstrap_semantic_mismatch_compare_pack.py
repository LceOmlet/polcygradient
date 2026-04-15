import json

from ticl.analysis.phase3_baseline_bootstrap_semantic_mismatch_compare_pack import (
    build_baseline_bootstrap_semantic_mismatch_compare_pack,
)


def _write(tmp_path, name: str, payload: dict) -> str:
    path = tmp_path / name
    path.write_text(json.dumps(payload))
    return str(path)


def test_build_baseline_bootstrap_semantic_mismatch_compare_pack_finds_tail_core_and_mid_extension(tmp_path):
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
                    "negative_source_shares": {"baseline_bootstrap_semantic_mismatch": 0.55},
                },
                {
                    "dominant_negative_source": "baseline_bootstrap_semantic_mismatch",
                    "start_objective_local_index": 100,
                    "start_tokens_to_objective_episode_end": 1,
                    "start_raw_rollout_residual_norm": 0.06,
                    "start_gae_future_carry_norm": -0.07,
                    "negative_source_shares": {"baseline_bootstrap_semantic_mismatch": 1.0},
                },
            ],
        },
    )
    env1 = _write(
        tmp_path,
        "env1.json",
        {
            "config": {
                "suite_name": "pair2",
                "env_index": 1,
                "flip_mode": "future_carry_dominant",
            },
            "start_rows": [
                {
                    "dominant_negative_source": "baseline_bootstrap_semantic_mismatch",
                    "start_objective_local_index": 97,
                    "start_tokens_to_objective_episode_end": 4,
                    "start_raw_rollout_residual_norm": 0.011,
                    "start_gae_future_carry_norm": -0.018,
                    "negative_source_shares": {"baseline_bootstrap_semantic_mismatch": 1.0},
                },
                {
                    "dominant_negative_source": "baseline_bootstrap_semantic_mismatch",
                    "start_objective_local_index": 100,
                    "start_tokens_to_objective_episode_end": 1,
                    "start_raw_rollout_residual_norm": 0.019,
                    "start_gae_future_carry_norm": -0.04,
                    "negative_source_shares": {"baseline_bootstrap_semantic_mismatch": 1.0},
                },
            ],
        },
    )

    report = build_baseline_bootstrap_semantic_mismatch_compare_pack(
        env12_json=env12,
        env1_json=env1,
    )

    assert report["audit_entry"] == "phase3_baseline_bootstrap_semantic_mismatch_compare_pack"
    assert report["comparison"]["shared_tail_core_supported"] is True
    assert report["comparison"]["env12"]["baseline_bootstrap_rows"]["tail_core"]["row_count"] == 1
    assert report["comparison"]["env12"]["baseline_bootstrap_rows"]["mid_episode_extension"]["row_count"] == 1
    assert report["comparison"]["env1"]["baseline_bootstrap_rows"]["tail_core"]["row_count"] == 2
    assert report["comparison"]["env1"]["baseline_bootstrap_rows"]["mid_episode_extension"]["row_count"] == 0
    assert report["recommendation"]["candidate_small_fix_target"] == "baseline_bootstrap_semantic_mismatch"
    assert report["recommendation"]["narrow_fix_scope"] == "near_terminal_tail_core"
    assert report["recommendation"]["branch_specific_followup_needed"] is True
    assert report["conclusions"]["tail_core_shared_across_env12_and_env1"] is True


def test_build_baseline_bootstrap_semantic_mismatch_compare_pack_can_shrink_to_terminal_adjacent_core(tmp_path):
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
                    "start_objective_local_index": 100,
                    "start_tokens_to_objective_episode_end": 1,
                    "start_raw_rollout_residual_norm": 0.06,
                    "start_gae_future_carry_norm": -0.07,
                    "negative_source_shares": {"baseline_bootstrap_semantic_mismatch": 1.0},
                }
            ],
        },
    )
    env1 = _write(
        tmp_path,
        "env1.json",
        {
            "config": {
                "suite_name": "pair2",
                "env_index": 1,
                "flip_mode": "future_carry_dominant",
            },
            "start_rows": [
                {
                    "dominant_negative_source": "baseline_bootstrap_semantic_mismatch",
                    "start_objective_local_index": 97,
                    "start_tokens_to_objective_episode_end": 4,
                    "start_raw_rollout_residual_norm": 0.011,
                    "start_gae_future_carry_norm": -0.018,
                    "negative_source_shares": {"baseline_bootstrap_semantic_mismatch": 1.0},
                },
                {
                    "dominant_negative_source": "baseline_bootstrap_semantic_mismatch",
                    "start_objective_local_index": 99,
                    "start_tokens_to_objective_episode_end": 2,
                    "start_raw_rollout_residual_norm": 0.013,
                    "start_gae_future_carry_norm": -0.017,
                    "negative_source_shares": {"baseline_bootstrap_semantic_mismatch": 1.0},
                },
                {
                    "dominant_negative_source": "baseline_bootstrap_semantic_mismatch",
                    "start_objective_local_index": 100,
                    "start_tokens_to_objective_episode_end": 1,
                    "start_raw_rollout_residual_norm": 0.019,
                    "start_gae_future_carry_norm": -0.04,
                    "negative_source_shares": {"baseline_bootstrap_semantic_mismatch": 1.0},
                },
            ],
        },
    )

    report = build_baseline_bootstrap_semantic_mismatch_compare_pack(
        env12_json=env12,
        env1_json=env1,
        near_terminal_window=1,
    )

    assert report["comparison"]["shared_tail_core_supported"] is True
    assert report["comparison"]["env12"]["baseline_bootstrap_rows"]["tail_core"]["row_count"] == 1
    assert report["comparison"]["env12"]["baseline_bootstrap_rows"]["mid_episode_extension"]["row_count"] == 0
    assert report["comparison"]["env1"]["baseline_bootstrap_rows"]["tail_core"]["row_count"] == 1
    assert report["comparison"]["env1"]["baseline_bootstrap_rows"]["mid_episode_extension"]["row_count"] == 2
    assert report["recommendation"]["narrow_fix_scope"] == "terminal_adjacent_core"
    assert report["conclusions"]["tail_core_shared_across_env12_and_env1"] is True


def test_build_baseline_bootstrap_semantic_mismatch_compare_pack_rejects_bad_env_index(tmp_path):
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
                    "negative_source_shares": {"baseline_bootstrap_semantic_mismatch": 0.55},
                }
            ],
        },
    )
    bad_env1 = _write(
        tmp_path,
        "bad_env1.json",
        {
            "config": {
                "suite_name": "pair2",
                "env_index": 2,
                "flip_mode": "future_carry_dominant",
            },
            "start_rows": [
                {
                    "dominant_negative_source": "baseline_bootstrap_semantic_mismatch",
                    "start_objective_local_index": 97,
                    "start_tokens_to_objective_episode_end": 4,
                    "start_raw_rollout_residual_norm": 0.011,
                    "start_gae_future_carry_norm": -0.018,
                    "negative_source_shares": {"baseline_bootstrap_semantic_mismatch": 1.0},
                }
            ],
        },
    )

    try:
        build_baseline_bootstrap_semantic_mismatch_compare_pack(env12_json=env12, env1_json=bad_env1)
    except ValueError as exc:
        assert "Unexpected env_index" in str(exc)
    else:
        raise AssertionError("expected env index mismatch to raise")
