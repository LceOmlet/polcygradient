import json
import math
import sys

from ticl.analysis import phase3_env12_local56_boundary_contract_probe as probe_mod


def test_build_local56_boundary_contract_probe_from_minimal_payloads():
    negative_payload = {
        "audit_entry": "phase3_env12_negative_delta_probe",
        "guardrails": {
            "inferred_env_std": 10.0,
            "inferred_mean_over_std_from_terminal_reset": 1.2,
        },
        "focus_rows": [
            {
                "objective_local_index": 56,
                "global_step": 123,
                "next_non_terminal": 0.0,
                "reward_raw": -0.3,
                "reward_norm": -1.23,
                "value_norm": 0.2,
                "delta_norm": -1.43,
            }
        ],
    }
    gae_payload = {
        "audit_entry": "phase3_residual_anchor_gae_path_probe",
        "token_row_bundles": [
            {
                "suite_name": "pair2",
                "env_index": 12,
                "token_rows": [
                    {
                        "objective_local_index": 55,
                        "reward_raw": 0.1,
                        "reward_norm": 0.01,
                        "next_non_terminal": 1.0,
                        "rollout_return_raw": 7.0,
                    },
                    {
                        "objective_local_index": 56,
                        "reward_raw": -0.3,
                        "reward_norm": -1.23,
                        "next_non_terminal": 0.0,
                        "rollout_return_raw": 6.0,
                    },
                    {
                        "objective_local_index": 57,
                        "reward_raw": -0.05,
                        "reward_norm": -0.005,
                        "next_non_terminal": 1.0,
                        "rollout_return_raw": 5.5,
                    },
                    {
                        "objective_local_index": 101,
                        "reward_raw": 0.02,
                        "reward_norm": 0.002,
                        "next_non_terminal": 1.0,
                        "rollout_return_raw": 4.5,
                    },
                ],
            }
        ],
    }

    result = probe_mod._build_probe(
        negative_payload=negative_payload,
        gae_payload=gae_payload,
        suite_name="pair2",
        env_index=12,
    )

    assert result["audit_entry"] == "phase3_env12_local56_boundary_contract_probe"
    assert result["conclusions"]["local56_boundary_shift_matches_full_rollout_centering_contract"] is True
    assert result["conclusions"]["nonterminal_boundary_rows_imply_gamma_one_contract"] is True
    assert result["conclusions"]["local56_boundary_shift_recovers_most_of_negative_delta_if_removed"] is True
    assert result["conclusions"]["objective_subset_stats_do_not_explain_terminal_boundary_constant"] is True
    assert math.isclose(result["local56_boundary"]["delta_without_boundary_shift"], -0.23, rel_tol=0.0, abs_tol=1e-12)


def test_phase3_env12_local56_boundary_contract_probe_reuses_existing_output(tmp_path, monkeypatch, capsys):
    output_path = tmp_path / "env12_local56_boundary_contract_probe.json"
    payload = {"ok": True, "kind": "env12_local56_boundary_contract_probe"}
    output_path.write_text(json.dumps(payload))

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "phase3_env12_local56_boundary_contract_probe.py",
            "--output-json",
            str(output_path),
        ],
    )

    assert probe_mod.main() == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == payload
