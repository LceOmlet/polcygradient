import json

from ticl.analysis.phase3_residual_flip_mode_compare_pack import (
    build_residual_flip_mode_compare_pack,
)


def _write(tmp_path, name: str, payload: dict) -> str:
    path = tmp_path / name
    path.write_text(json.dumps(payload))
    return str(path)


def test_build_residual_flip_mode_compare_pack_extracts_active_search_target(tmp_path):
    split = _write(
        tmp_path,
        "split.json",
        {
            "mode_rows": [
                {
                    "flip_mode": "bootstrap_dominant",
                    "mean_pre_vs_zero_suffix_gap": 8.0,
                    "mean_raw_rollout_residual_norm": 0.2,
                    "mean_reward_norm": -0.02,
                    "mean_bootstrap_term_norm": -0.04,
                    "mean_delta_norm": -0.05,
                    "mean_gae_future_carry_norm": -0.12,
                    "mean_actor_adv_norm": -0.183,
                    "mean_token_in_objective_episode": 57.0,
                    "mean_tokens_to_objective_episode_end": 29.0,
                    "next_step_nonobjective_share": 0.05,
                    "share_within_target_tokens": 0.57,
                },
                {
                    "flip_mode": "future_carry_dominant",
                    "mean_pre_vs_zero_suffix_gap": 10.7,
                    "mean_raw_rollout_residual_norm": 0.31,
                    "mean_reward_norm": 0.006,
                    "mean_bootstrap_term_norm": 0.02,
                    "mean_delta_norm": 0.03,
                    "mean_gae_future_carry_norm": -0.21,
                    "mean_actor_adv_norm": -0.1834,
                    "mean_token_in_objective_episode": 55.4,
                    "mean_tokens_to_objective_episode_end": 25.4,
                    "next_step_nonobjective_share": 0.0,
                    "share_within_target_tokens": 0.43,
                },
            ],
            "flip_token_rows": [
                {"flip_mode": "bootstrap_dominant", "env_index": 7},
                {"flip_mode": "future_carry_dominant", "env_index": 12},
                {"flip_mode": "future_carry_dominant", "env_index": 12},
            ],
        },
    )

    report = build_residual_flip_mode_compare_pack(residual_flip_split_json=split)

    assert report["audit_entry"] == "phase3_residual_flip_mode_compare_pack"
    assert report["conclusions"]["future_carry_chain_is_primary_active_search_target"] is True
    assert report["conclusions"]["bootstrap_only_fix_not_enough"] is True
    assert report["conclusions"]["future_carry_mode_has_stronger_anchor_and_residual_signal"] is True
    assert report["conclusions"]["future_carry_mode_carries_more_negative_recursive_signal"] is True
    assert report["recommendation"]["candidate_small_fix_target"] == "future_carry_source_chain"
    assert report["recommendation"]["regime_split_guardrail_required"] is True
    assert report["comparison"]["shared_actor_adv_magnitude"] is True
    assert report["token_distribution"]["future_carry_dominant"]["12"] == 2


def test_build_residual_flip_mode_compare_pack_rejects_missing_modes(tmp_path):
    split = _write(
        tmp_path,
        "split.json",
        {
            "mode_rows": [
                {
                    "flip_mode": "bootstrap_dominant",
                    "mean_pre_vs_zero_suffix_gap": 8.0,
                    "mean_raw_rollout_residual_norm": 0.2,
                    "mean_reward_norm": -0.02,
                    "mean_bootstrap_term_norm": -0.04,
                    "mean_delta_norm": -0.05,
                    "mean_gae_future_carry_norm": -0.12,
                    "mean_actor_adv_norm": -0.183,
                    "mean_token_in_objective_episode": 57.0,
                    "mean_tokens_to_objective_episode_end": 29.0,
                    "next_step_nonobjective_share": 0.05,
                    "share_within_target_tokens": 0.57,
                }
            ],
            "flip_token_rows": [{"flip_mode": "bootstrap_dominant", "env_index": 7}],
        },
    )

    try:
        build_residual_flip_mode_compare_pack(residual_flip_split_json=split)
    except ValueError as exc:
        assert "Missing flip_mode rows" in str(exc)
    else:
        raise AssertionError("expected missing mode rows to raise")
