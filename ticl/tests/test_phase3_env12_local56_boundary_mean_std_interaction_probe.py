import json
import sys

from ticl.analysis import phase3_env12_local56_boundary_mean_std_interaction_probe as probe_mod


def test_build_boundary_mean_std_interaction_probe_from_minimal_payloads():
    scope_payload = {
        "audit_entry": "phase3_env12_local56_boundary_scope_counterfactual_probe",
        "scope_rows": [
            {
                "scope_name": "current_full_rollout",
                "mean_raw": 10.0,
                "std_raw": 10.0,
                "mean_over_std": 1.0,
                "local56_boundary_shift_norm": -1.0,
            },
            {
                "scope_name": "objective_all_tokens",
                "mean_raw": 8.0,
                "std_raw": 4.0,
                "mean_over_std": 2.0,
                "local56_boundary_shift_norm": -2.0,
            },
            {
                "scope_name": "objective_episode0_only",
                "mean_raw": 7.0,
                "std_raw": 2.0,
                "mean_over_std": 3.5,
                "local56_boundary_shift_norm": -3.5,
            },
            {
                "scope_name": "objective_episode1_only",
                "mean_raw": 6.0,
                "std_raw": 3.0,
                "mean_over_std": 2.0,
                "local56_boundary_shift_norm": -2.0,
            },
        ],
    }

    result = probe_mod._build_probe(scope_payload)

    assert result["audit_entry"] == "phase3_env12_local56_boundary_mean_std_interaction_probe"
    assert result["conclusions"]["objective_all_worsening_is_std_dominated"] is True
    assert result["conclusions"]["episode0_worsening_is_std_dominated"] is True
    assert result["conclusions"]["episode1_worsening_is_std_dominated"] is True
    assert result["conclusions"]["mean_change_alone_would_shrink_penalty_in_all_objective_scopes"] is True
    assert result["conclusions"]["std_change_alone_would_worsen_penalty_in_all_objective_scopes"] is True
    assert result["conclusions"]["interaction_offsets_part_of_std_worsening_in_all_objective_scopes"] is True


def test_phase3_env12_local56_boundary_mean_std_interaction_probe_reuses_existing_output(
    tmp_path, monkeypatch, capsys
):
    output_path = tmp_path / "env12_local56_boundary_mean_std_interaction_probe.json"
    payload = {"ok": True, "kind": "env12_local56_boundary_mean_std_interaction_probe"}
    output_path.write_text(json.dumps(payload))

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "phase3_env12_local56_boundary_mean_std_interaction_probe.py",
            "--output-json",
            str(output_path),
        ],
    )

    assert probe_mod.main() == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == payload
