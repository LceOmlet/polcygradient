import json
import sys

from ticl.analysis import phase3_env12_local56_mean_source_probe as probe_mod


def test_build_local56_mean_source_probe_from_minimal_payloads():
    boundary_payload = {
        "audit_entry": "phase3_env12_local56_boundary_contract_probe",
        "full_rollout_contract": {
            "full_rollout_mean_raw": 10.0,
            "full_rollout_std_raw": 5.0,
            "full_rollout_mean_over_std": 2.0,
        },
    }
    gae_payload = {
        "audit_entry": "phase3_residual_anchor_gae_path_probe",
        "config": {
            "n_samples": 10,
            "single_eval_pos": 6,
        },
        "token_row_bundles": [
            {
                "suite_name": "pair2",
                "env_index": 12,
                "token_rows": [
                    {
                        "objective_local_index": 0,
                        "global_step": 6,
                        "objective_episode_index": 0,
                        "rollout_return_raw": 7.0,
                        "reward_raw": 0.1,
                        "reward_norm": 0.02,
                        "next_non_terminal": 1.0,
                    },
                    {
                        "objective_local_index": 1,
                        "global_step": 7,
                        "objective_episode_index": 0,
                        "rollout_return_raw": 6.0,
                        "reward_raw": 0.2,
                        "reward_norm": 0.04,
                        "next_non_terminal": 0.0,
                    },
                    {
                        "objective_local_index": 2,
                        "global_step": 8,
                        "objective_episode_index": 1,
                        "rollout_return_raw": 3.0,
                        "reward_raw": 0.3,
                        "reward_norm": 0.06,
                        "next_non_terminal": 1.0,
                    },
                    {
                        "objective_local_index": 56,
                        "global_step": 9,
                        "objective_episode_index": 1,
                        "rollout_return_raw": 4.0,
                        "reward_raw": -0.5,
                        "reward_norm": -0.1,
                        "next_non_terminal": 0.0,
                    },
                ],
            }
        ],
    }

    result = probe_mod._build_probe(
        boundary_payload=boundary_payload,
        gae_payload=gae_payload,
        suite_name="pair2",
        env_index=12,
    )

    assert result["audit_entry"] == "phase3_env12_local56_mean_source_probe"
    assert result["timing_contract"]["objective_starts_at_single_eval_pos"] is True
    assert result["timing_contract"]["objective_is_tail_of_full_rollout"] is True
    assert result["conclusions"]["full_mean_is_prefix_dominated_not_objective_dominated"] is True
    assert result["conclusions"]["objective_tail_tokens_are_too_late_to_set_full_mean_constant"] is True
    assert result["conclusions"]["post_reset_episode_lowers_objective_mean_instead_of_raising_it"] is True
    assert result["conclusions"]["local56_terminal_row_is_not_a_positive_mean_source"] is True
    assert result["conclusions"]["large_terminal_boundary_constant_is_explained_by_full_rollout_centering_scope"] is True
    assert result["full_mean_decomposition"]["implied_prefix_mean_rollout_return_raw"] > result["full_mean_decomposition"]["full_rollout_mean_raw"]


def test_phase3_env12_local56_mean_source_probe_reuses_existing_output(tmp_path, monkeypatch, capsys):
    output_path = tmp_path / "env12_local56_mean_source_probe.json"
    payload = {"ok": True, "kind": "env12_local56_mean_source_probe"}
    output_path.write_text(json.dumps(payload))

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "phase3_env12_local56_mean_source_probe.py",
            "--output-json",
            str(output_path),
        ],
    )

    assert probe_mod.main() == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == payload
