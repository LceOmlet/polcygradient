import json

from ticl.analysis.phase3_future_carry_source_chain_compare_pack import (
    build_future_carry_source_chain_compare_pack,
)


def _write(tmp_path, name: str, payload: dict) -> str:
    path = tmp_path / name
    path.write_text(json.dumps(payload))
    return str(path)


def test_build_future_carry_source_chain_compare_pack_isolates_shared_candidate(tmp_path):
    env12 = _write(
        tmp_path,
        "env12.json",
        {
            "config": {"suite_name": "pair2", "env_index": 12, "flip_mode": "future_carry_dominant"},
            "aggregate": {
                "dominant_negative_source": "terminal_reset_tail_semantic_mismatch",
                "negative_source_shares": {
                    "baseline_bootstrap_semantic_mismatch": 0.25829536613335724,
                    "terminal_reset_tail_semantic_mismatch": 0.7417046338666428,
                },
                "start_token_count": 23,
                "negative_contribution_total": 15.0,
            },
        },
    )
    env1 = _write(
        tmp_path,
        "env1.json",
        {
            "config": {"suite_name": "pair2", "env_index": 1, "flip_mode": "future_carry_dominant"},
            "aggregate": {
                "dominant_negative_source": "true_negative_future_residual",
                "negative_source_shares": {
                    "baseline_bootstrap_semantic_mismatch": 0.26676411147412926,
                    "true_negative_future_residual": 0.7332358885258707,
                },
                "start_token_count": 7,
                "negative_contribution_total": 1.0,
            },
        },
    )

    report = build_future_carry_source_chain_compare_pack(env12_json=env12, env1_json=env1)

    assert report["audit_entry"] == "phase3_future_carry_source_chain_compare_pack"
    assert report["comparison"]["shared_negative_sources"] == ["baseline_bootstrap_semantic_mismatch"]
    assert report["conclusions"]["shared_small_fix_candidate_is_baseline_bootstrap_semantic_mismatch"] is True
    assert report["conclusions"]["future_carry_source_chain_is_not_single_dominant_bucket"] is True
    assert report["conclusions"]["env12_branch_specific_dominant_source_is_terminal_reset_tail"] is True
    assert report["conclusions"]["env1_branch_specific_dominant_source_is_true_negative_future_residual"] is True
    assert report["recommendation"]["candidate_small_fix_target"] == "baseline_bootstrap_semantic_mismatch"
    assert report["recommendation"]["branch_specific_followup_needed"] is True


def test_build_future_carry_source_chain_compare_pack_rejects_wrong_env(tmp_path):
    env12 = _write(
        tmp_path,
        "env12.json",
        {
            "config": {"suite_name": "pair2", "env_index": 12, "flip_mode": "future_carry_dominant"},
            "aggregate": {"dominant_negative_source": "terminal_reset_tail_semantic_mismatch"},
        },
    )
    bad_env1 = _write(
        tmp_path,
        "bad_env1.json",
        {
            "config": {"suite_name": "pair2", "env_index": 2, "flip_mode": "future_carry_dominant"},
            "aggregate": {"dominant_negative_source": "true_negative_future_residual"},
        },
    )

    try:
        build_future_carry_source_chain_compare_pack(env12_json=env12, env1_json=bad_env1)
    except ValueError as exc:
        assert "Unexpected env_index" in str(exc)
    else:
        raise AssertionError("expected env mismatch to raise")
