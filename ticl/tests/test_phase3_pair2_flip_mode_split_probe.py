import json
import sys

from ticl.analysis import phase3_pair2_flip_mode_split_probe as probe_mod


def test_classify_flip_mode_maps_gaepath_causes_to_two_buckets():
    assert probe_mod._classify_flip_mode({"flip_cause": "negative_gae_future_carry"}) == "future_carry_dominant"
    assert probe_mod._classify_flip_mode({"flip_cause": "bootstrap_term_negative"}) == "bootstrap_dominant"
    assert probe_mod._classify_flip_mode({"flip_cause": "delta_negative_mixed"}) == "bootstrap_dominant"


def test_build_probe_from_payload_filters_pair2_positive_residual_wrong_sign_tokens():
    payload = {
        "config": {
            "train_profile": "trusted_sep_reset_mainline",
            "n_samples": 2048,
            "single_eval_pos": 1946,
        },
        "token_row_bundles": [
            {
                "suite_name": "pair2",
                "env_index": 12,
                "pre_vs_zero_suffix_gap": 21.4,
                "token_rows": [
                    {
                        "flip_cause": "negative_gae_future_carry",
                        "positive_residual_wrong_sign_actor": True,
                        "raw_rollout_residual_norm": 0.6,
                        "reward_norm": -0.01,
                        "bootstrap_term_norm": -0.02,
                        "delta_norm": 0.03,
                        "mc_future_carry_equiv_norm": 0.57,
                        "gae_future_carry_norm": -0.4,
                        "actor_adv_norm": -0.37,
                        "objective_episode_index": 0,
                        "token_in_objective_episode": 10,
                        "tokens_to_objective_episode_end": 40,
                        "steps_until_next_objective": 1,
                        "next_step_is_objective": True,
                        "objective_local_index": 10,
                    },
                    {
                        "flip_cause": "bootstrap_term_negative",
                        "positive_residual_wrong_sign_actor": True,
                        "raw_rollout_residual_norm": 0.1,
                        "reward_norm": -0.01,
                        "bootstrap_term_norm": -0.1,
                        "delta_norm": -0.11,
                        "mc_future_carry_equiv_norm": 0.21,
                        "gae_future_carry_norm": -0.02,
                        "actor_adv_norm": -0.13,
                        "objective_episode_index": 0,
                        "token_in_objective_episode": 70,
                        "tokens_to_objective_episode_end": 5,
                        "steps_until_next_objective": 1,
                        "next_step_is_objective": True,
                        "objective_local_index": 70,
                    },
                    {
                        "flip_cause": "not_flip",
                        "positive_residual_wrong_sign_actor": False,
                        "raw_rollout_residual_norm": 0.2,
                        "reward_norm": 0.1,
                        "bootstrap_term_norm": 0.0,
                        "delta_norm": 0.1,
                        "mc_future_carry_equiv_norm": 0.1,
                        "gae_future_carry_norm": 0.0,
                        "actor_adv_norm": 0.1,
                        "objective_episode_index": 0,
                        "token_in_objective_episode": 0,
                        "tokens_to_objective_episode_end": 50,
                        "steps_until_next_objective": 1,
                        "next_step_is_objective": True,
                        "objective_local_index": 0,
                    },
                ],
            },
            {
                "suite_name": "pair1",
                "env_index": 11,
                "pre_vs_zero_suffix_gap": 11.0,
                "token_rows": [
                    {
                        "flip_cause": "bootstrap_term_negative",
                        "positive_residual_wrong_sign_actor": True,
                        "raw_rollout_residual_norm": 0.9,
                        "reward_norm": -0.2,
                        "bootstrap_term_norm": -0.3,
                        "delta_norm": -0.5,
                        "mc_future_carry_equiv_norm": 1.4,
                        "gae_future_carry_norm": -0.1,
                        "actor_adv_norm": -0.6,
                        "objective_episode_index": 0,
                        "token_in_objective_episode": 20,
                        "tokens_to_objective_episode_end": 20,
                        "steps_until_next_objective": 1,
                        "next_step_is_objective": True,
                        "objective_local_index": 20,
                    }
                ],
            },
        ],
    }

    result = probe_mod._build_probe_from_payload(payload, "pair2")
    assert result["audit_entry"] == "phase3_pair2_flip_mode_split_probe"
    assert result["aggregate"]["target_token_count"] == 2
    assert result["aggregate"]["flip_mode_counts"] == {
        "bootstrap_dominant": 1,
        "future_carry_dominant": 1,
    }
    mode_by_name = {row["flip_mode"]: row for row in result["mode_rows"]}
    assert mode_by_name["future_carry_dominant"]["mean_token_in_objective_episode"] == 10.0
    assert mode_by_name["bootstrap_dominant"]["mean_token_in_objective_episode"] == 70.0
    assert result["conclusions"]["future_carry_tokens_are_more_front_loaded"] is True
    assert result["conclusions"]["bootstrap_tokens_are_more_tail_skewed"] is True
    assert all(row["suite_name"] == "pair2" for row in result["flip_token_rows"])


def test_phase3_pair2_flip_mode_split_probe_reuses_existing_output(tmp_path, monkeypatch, capsys):
    output_path = tmp_path / "pair2_flip_mode_split.json"
    payload = {"ok": True, "kind": "pair2_flip_mode_split"}
    output_path.write_text(json.dumps(payload))

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "phase3_pair2_flip_mode_split_probe.py",
            "--output-json",
            str(output_path),
        ],
    )

    assert probe_mod.main() == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == payload
