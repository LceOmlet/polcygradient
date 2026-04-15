import json
import sys

from ticl.analysis import phase3_env12_local56_boundary_scope_counterfactual_probe as probe_mod


def test_build_boundary_scope_counterfactual_probe_from_minimal_payloads():
    boundary_payload = {
        "audit_entry": "phase3_env12_local56_boundary_contract_probe",
        "full_rollout_contract": {
            "full_rollout_mean_raw": 10.0,
            "full_rollout_std_raw": 10.0,
            "full_rollout_mean_over_std": 1.0,
        },
        "local56_boundary": {
            "actual_boundary_shift_norm": -1.0,
            "reward_over_std": -0.1,
            "reward_norm": -1.1,
        },
    }
    gae_payload = {
        "audit_entry": "phase3_residual_anchor_gae_path_probe",
        "config": {
            "n_samples": 10,
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
                        "rollout_return_raw": 6.0,
                        "reward_raw": 0.0,
                    },
                    {
                        "objective_local_index": 1,
                        "global_step": 7,
                        "objective_episode_index": 0,
                        "rollout_return_raw": 8.0,
                        "reward_raw": 0.0,
                    },
                    {
                        "objective_local_index": 56,
                        "global_step": 8,
                        "objective_episode_index": 0,
                        "rollout_return_raw": 10.0,
                        "reward_raw": -1.0,
                    },
                    {
                        "objective_local_index": 57,
                        "global_step": 9,
                        "objective_episode_index": 1,
                        "rollout_return_raw": 1.0,
                        "reward_raw": 0.0,
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

    assert result["audit_entry"] == "phase3_env12_local56_boundary_scope_counterfactual_probe"
    assert result["conclusions"]["objective_all_centering_worsens_boundary_penalty_vs_current"] is True
    assert result["conclusions"]["objective_episode0_centering_worsens_boundary_penalty_vs_current"] is True
    assert result["conclusions"]["objective_episode1_centering_worsens_boundary_penalty_vs_current"] is False
    assert result["conclusions"]["no_objective_aligned_scope_shrinks_local56_boundary_penalty"] is False


def test_phase3_env12_local56_boundary_scope_counterfactual_probe_reuses_existing_output(
    tmp_path, monkeypatch, capsys
):
    output_path = tmp_path / "env12_local56_boundary_scope_counterfactual_probe.json"
    payload = {"ok": True, "kind": "env12_local56_boundary_scope_counterfactual_probe"}
    output_path.write_text(json.dumps(payload))

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "phase3_env12_local56_boundary_scope_counterfactual_probe.py",
            "--output-json",
            str(output_path),
        ],
    )

    assert probe_mod.main() == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == payload
