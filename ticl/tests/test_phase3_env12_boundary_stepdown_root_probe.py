import json
import sys

from ticl.analysis import phase3_env12_boundary_stepdown_root_probe as probe_mod


def test_build_local56_and_cluster_sections_from_minimal_payloads():
    negative_payload = {
        "audit_entry": "phase3_env12_negative_delta_probe",
        "guardrails": {
            "inferred_env_std": 10.0,
            "inferred_mean_over_std_from_terminal_reset": 1.2,
        },
        "focus_rows": [
            {
                "objective_local_index": 29,
                "reward_norm": -0.001,
                "reward_over_std": -0.001,
                "boundary_shift_norm": 0.0,
                "bootstrap_term_norm": -0.07,
                "delta_norm": -0.071,
                "value_norm": 0.20,
                "next_value_norm": 0.13,
                "aggregated_negative_source_contribution": 0.4,
                "tokens_to_objective_episode_end": 27,
            },
            {
                "objective_local_index": 38,
                "reward_norm": 0.01,
                "reward_over_std": 0.01,
                "boundary_shift_norm": 0.0,
                "bootstrap_term_norm": -0.06,
                "delta_norm": -0.05,
                "value_norm": 0.22,
                "next_value_norm": 0.16,
                "aggregated_negative_source_contribution": 0.3,
                "tokens_to_objective_episode_end": 18,
            },
            {
                "objective_local_index": 51,
                "reward_norm": 0.02,
                "reward_over_std": 0.02,
                "boundary_shift_norm": 0.0,
                "bootstrap_term_norm": -0.08,
                "delta_norm": -0.06,
                "value_norm": 0.23,
                "next_value_norm": 0.15,
                "aggregated_negative_source_contribution": 0.2,
                "tokens_to_objective_episode_end": 5,
            },
            {
                "objective_local_index": 56,
                "reward_norm": -1.25,
                "reward_over_std": -0.03,
                "boundary_shift_norm": -1.22,
                "bootstrap_term_norm": -0.16,
                "delta_norm": -1.38,
                "value_norm": 0.16,
                "next_value_norm": 0.0,
                "aggregated_negative_source_contribution": 5.0,
                "tokens_to_objective_episode_end": 0,
            },
        ],
    }
    env_rows = []
    for local, value, nxt, reward, delta, to_end, cause, nn in [
        (28, 0.18, 0.20, 0.01, 0.03, 28, "not_flip", 1.0),
        (29, 0.20, 0.13, -0.001, -0.071, 27, "bootstrap_term_negative", 1.0),
        (30, 0.13, 0.24, 0.02, 0.11, 26, "negative_gae_future_carry", 1.0),
        (31, 0.23, 0.18, 0.01, -0.04, 25, "bootstrap_term_negative", 1.0),
        (37, 0.16, 0.22, 0.01, 0.07, 19, "not_flip", 1.0),
        (38, 0.22, 0.16, 0.01, -0.05, 18, "bootstrap_term_negative", 1.0),
        (39, 0.16, 0.24, -0.01, 0.08, 17, "negative_gae_future_carry", 1.0),
        (40, 0.24, 0.20, -0.01, -0.05, 16, "bootstrap_term_negative", 1.0),
        (50, 0.17, 0.23, -0.02, 0.05, 6, "negative_gae_future_carry", 1.0),
        (51, 0.23, 0.15, 0.02, -0.06, 5, "bootstrap_term_negative", 1.0),
        (52, 0.15, 0.13, 0.01, -0.01, 4, "bootstrap_term_negative", 1.0),
        (53, 0.13, 0.16, 0.01, 0.04, 3, "negative_gae_future_carry", 1.0),
        (54, 0.16, 0.19, 0.03, 0.06, 2, "negative_gae_future_carry", 1.0),
        (55, 0.19, 0.16, 0.01, -0.02, 1, "bootstrap_term_negative", 1.0),
        (56, 0.16, 0.0, -1.25, -1.38, 0, "delta_negative_mixed", 0.0),
        (57, 0.17, 0.12, -0.01, -0.05, 44, "not_flip", 1.0),
        (58, 0.12, 0.19, 0.02, 0.09, 43, "not_flip", 1.0),
        (99, 0.12, 0.15, 0.01, 0.03, 2, "not_flip", 1.0),
        (100, 0.15, 0.20, 0.02, 0.07, 1, "negative_gae_future_carry", 1.0),
        (101, 0.20, 0.13, 0.001, -0.07, 0, "bootstrap_term_negative", 1.0),
    ]:
        env_rows.append(
            {
                "suite_name": "pair2",
                "env_index": 12,
                "objective_episode_index": 0 if local <= 58 else 1,
                "objective_local_index": local,
                "global_step": local,
                "reward_raw": reward * 10.0,
                "reward_norm": reward,
                "value_norm": value,
                "next_value_norm": nxt,
                "delta_norm": delta,
                "next_non_terminal": nn,
                "tokens_to_objective_episode_end": to_end,
                "flip_cause": cause,
            }
        )
    gae_payload = {
        "audit_entry": "phase3_residual_anchor_gae_path_probe",
        "token_row_bundles": [
            {
                "suite_name": "pair2",
                "env_index": 12,
                "token_rows": env_rows,
            }
        ],
    }

    result = probe_mod._build_probe(
        negative_payload=negative_payload,
        gae_payload=gae_payload,
        suite_name="pair2",
        env_index=12,
    )
    assert result["audit_entry"] == "phase3_env12_boundary_stepdown_root_probe"
    assert result["local56_boundary"]["conclusions"]["local56_boundary_shift_dwarfs_immediate_reward"] is True
    assert result["local56_boundary"]["conclusions"]["local101_has_no_terminal_boundary_shift"] is True
    assert result["cluster_stepdown"]["aggregate"]["all_local_peaks"] is True
    assert result["cluster_stepdown"]["aggregate"]["oscillatory_peak_count"] == 2
    assert result["cluster_stepdown"]["aggregate"]["late_preterminal_peak_count"] == 1
    assert result["conclusions"]["cluster_contains_multiple_subpatterns"] is True


def test_phase3_env12_boundary_stepdown_root_probe_reuses_existing_output(tmp_path, monkeypatch, capsys):
    output_path = tmp_path / "env12_boundary_stepdown_root_probe.json"
    payload = {"ok": True, "kind": "env12_boundary_stepdown_root_probe"}
    output_path.write_text(json.dumps(payload))

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "phase3_env12_boundary_stepdown_root_probe.py",
            "--output-json",
            str(output_path),
        ],
    )

    assert probe_mod.main() == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == payload
