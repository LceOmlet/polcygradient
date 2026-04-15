import json
import sys

from ticl.analysis import phase3_env12_negative_delta_probe as probe_mod


def test_classify_negative_delta_driver_distinguishes_terminal_boundary_and_bootstrap_cases():
    assert (
        probe_mod._classify_negative_delta_driver(
            {
                "next_non_terminal": 0.0,
                "boundary_shift_norm": -1.2,
                "bootstrap_term_norm": -0.1,
                "reward_norm": -1.25,
            }
        )
        == "terminal_reset_boundary_shift_dominant"
    )
    assert (
        probe_mod._classify_negative_delta_driver(
            {
                "next_non_terminal": 1.0,
                "boundary_shift_norm": 0.0,
                "bootstrap_term_norm": -0.06,
                "reward_norm": 0.01,
            }
        )
        == "bootstrap_stepdown_overrides_positive_reward"
    )
    assert (
        probe_mod._classify_negative_delta_driver(
            {
                "next_non_terminal": 1.0,
                "boundary_shift_norm": 0.0,
                "bootstrap_term_norm": -0.07,
                "reward_norm": -0.001,
            }
        )
        == "bootstrap_negative_with_nonpositive_reward"
    )


def test_build_probe_summarizes_terminal_and_cluster_tokens():
    gae_payload = {
        "token_row_bundles": [
            {
                "suite_name": "pair2",
                "env_index": 12,
                "token_rows": [
                    {
                        "suite_name": "pair2",
                        "env_index": 12,
                        "global_step": 10,
                        "objective_episode_index": 0,
                        "objective_local_index": 29,
                        "token_in_objective_episode": 29,
                        "tokens_to_objective_episode_end": 27,
                        "next_non_terminal": 1.0,
                        "reward_raw": 0.10,
                        "reward_norm": 0.01,
                        "bootstrap_term_norm": -0.06,
                        "next_value_norm": 0.14,
                        "value_norm": 0.20,
                        "delta_norm": -0.05,
                        "raw_rollout_residual_norm": 0.70,
                        "raw_rollout_residual_raw": 7.0,
                        "raw_value_raw": -1.0,
                        "flip_cause": "bootstrap_term_negative",
                    },
                    {
                        "suite_name": "pair2",
                        "env_index": 12,
                        "global_step": 11,
                        "objective_episode_index": 0,
                        "objective_local_index": 38,
                        "token_in_objective_episode": 38,
                        "tokens_to_objective_episode_end": 18,
                        "next_non_terminal": 1.0,
                        "reward_raw": 0.20,
                        "reward_norm": 0.02,
                        "bootstrap_term_norm": -0.07,
                        "next_value_norm": 0.13,
                        "value_norm": 0.20,
                        "delta_norm": -0.05,
                        "raw_rollout_residual_norm": 0.80,
                        "raw_rollout_residual_raw": 8.0,
                        "raw_value_raw": -1.0,
                        "flip_cause": "bootstrap_term_negative",
                    },
                    {
                        "suite_name": "pair2",
                        "env_index": 12,
                        "global_step": 12,
                        "objective_episode_index": 0,
                        "objective_local_index": 51,
                        "token_in_objective_episode": 51,
                        "tokens_to_objective_episode_end": 5,
                        "next_non_terminal": 1.0,
                        "reward_raw": -0.01,
                        "reward_norm": -0.001,
                        "bootstrap_term_norm": -0.08,
                        "next_value_norm": 0.12,
                        "value_norm": 0.20,
                        "delta_norm": -0.081,
                        "raw_rollout_residual_norm": 0.90,
                        "raw_rollout_residual_raw": 9.0,
                        "raw_value_raw": -1.0,
                        "flip_cause": "bootstrap_term_negative",
                    },
                    {
                        "suite_name": "pair2",
                        "env_index": 12,
                        "global_step": 13,
                        "objective_episode_index": 0,
                        "objective_local_index": 56,
                        "token_in_objective_episode": 56,
                        "tokens_to_objective_episode_end": 0,
                        "next_non_terminal": 0.0,
                        "reward_raw": -11.7,
                        "reward_norm": -2.37,
                        "bootstrap_term_norm": -0.16,
                        "next_value_norm": 0.0,
                        "value_norm": 0.16,
                        "delta_norm": -1.34,
                        "raw_rollout_residual_norm": 1.00,
                        "raw_rollout_residual_raw": 10.0,
                        "raw_value_raw": -1.0,
                        "flip_cause": "delta_negative_mixed",
                    },
                ],
            }
        ]
    }
    source_payload = {
        "top_negative_source_tokens": [
            {
                "objective_episode_index": 0,
                "objective_local_index": 29,
                "aggregated_negative_contribution": 0.4,
            },
            {
                "objective_episode_index": 0,
                "objective_local_index": 38,
                "aggregated_negative_contribution": 0.3,
            },
            {
                "objective_episode_index": 0,
                "objective_local_index": 51,
                "aggregated_negative_contribution": 0.2,
            },
            {
                "objective_episode_index": 0,
                "objective_local_index": 56,
                "aggregated_negative_contribution": 5.0,
            },
        ]
    }

    result = probe_mod._build_probe(
        gae_payload=gae_payload,
        source_payload=source_payload,
        suite_name="pair2",
        env_index=12,
        token_local_indices=[56, 29, 38, 51],
    )
    assert result["audit_entry"] == "phase3_env12_negative_delta_probe"
    assert result["aggregate"]["cluster_summary"]["positive_reward_but_negative_delta_tokens"] == 2
    assert result["aggregate"]["cluster_summary"]["bootstrap_dominant_tokens"] == 3
    assert result["conclusions"]["local56_terminal_reset_boundary_shift_dominant"] is True
    assert result["conclusions"]["nonterminal_cluster_bootstrap_dominant"] is True
    focus_by_local = {row["objective_local_index"]: row for row in result["focus_rows"]}
    assert focus_by_local[29]["negative_delta_driver"] == "bootstrap_stepdown_overrides_positive_reward"
    assert focus_by_local[51]["negative_delta_driver"] == "bootstrap_negative_with_nonpositive_reward"
    assert focus_by_local[56]["aggregated_negative_source_contribution"] == 5.0


def test_phase3_env12_negative_delta_probe_reuses_existing_output(tmp_path, monkeypatch, capsys):
    output_path = tmp_path / "env12_negative_delta_probe.json"
    payload = {"ok": True, "kind": "env12_negative_delta_probe"}
    output_path.write_text(json.dumps(payload))

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "phase3_env12_negative_delta_probe.py",
            "--output-json",
            str(output_path),
        ],
    )

    assert probe_mod.main() == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == payload
