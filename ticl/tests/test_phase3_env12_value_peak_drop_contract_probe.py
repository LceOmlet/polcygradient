import json
import sys

from ticl.analysis import phase3_env12_value_peak_drop_contract_probe as probe_mod


def test_build_value_peak_drop_contract_probe_from_minimal_payloads():
    negative_payload = {
        "audit_entry": "phase3_env12_negative_delta_probe",
        "guardrails": {
            "inferred_env_std": 10.0,
            "inferred_mean_over_std_from_terminal_reset": 1.2,
        },
    }
    token_rows = []
    for local, reward_norm, reward_raw, rollout_return_raw, value_norm, next_value_norm, delta_norm in [
        (28, 0.005, 0.05, 5.0, 0.10, 0.18, 0.085),
        (29, -0.005, -0.05, 5.05, 0.18, 0.11, -0.075),
        (30, 0.01, 0.10, 5.10, 0.11, 0.12, 0.0),
        (37, -0.01, -0.10, 4.60, 0.12, 0.20, 0.07),
        (38, 0.005, 0.05, 4.70, 0.20, 0.14, -0.055),
        (39, -0.015, -0.15, 4.65, 0.14, 0.13, -0.025),
        (50, 0.02, 0.20, 3.00, 0.15, 0.24, 0.11),
        (51, 0.015, 0.15, 3.20, 0.24, 0.16, -0.065),
        (52, -0.005, -0.05, 3.05, 0.16, 0.17, -0.015),
    ]:
        token_rows.append(
            {
                "suite_name": "pair2",
                "env_index": 12,
                "objective_local_index": local,
                "global_step": local,
                "reward_norm": reward_norm,
                "reward_raw": reward_raw,
                "rollout_return_raw": rollout_return_raw,
                "value_norm": value_norm,
                "next_value_norm": next_value_norm,
                "delta_norm": delta_norm,
                "next_non_terminal": 1.0,
            }
        )
    gae_payload = {
        "audit_entry": "phase3_residual_anchor_gae_path_probe",
        "token_row_bundles": [
            {
                "suite_name": "pair2",
                "env_index": 12,
                "token_rows": token_rows,
            }
        ],
    }

    result = probe_mod._build_probe(
        negative_payload=negative_payload,
        gae_payload=gae_payload,
        suite_name="pair2",
        env_index=12,
        target_locals=[29, 38, 51],
    )

    assert result["audit_entry"] == "phase3_env12_value_peak_drop_contract_probe"
    assert result["conclusions"]["reward_matches_target_step_change_under_gamma_one_contract"] is True
    assert result["conclusions"]["delta_matches_reward_minus_value_stepdown_contract"] is True
    assert result["conclusions"]["stepdown_excess_equals_error_drop_contract"] is True
    assert result["conclusions"]["all_tokens_keep_positive_future_raw_return"] is True
    assert result["conclusions"]["local29_is_hallucinated_peak_against_target"] is True
    assert result["conclusions"]["locals38_51_are_amplified_target_peaks"] is True
    assert result["aggregate"]["hallucinated_peak_count"] == 1
    assert result["aggregate"]["amplified_target_peak_count"] == 2


def test_phase3_env12_value_peak_drop_contract_probe_reuses_existing_output(tmp_path, monkeypatch, capsys):
    output_path = tmp_path / "env12_value_peak_drop_contract_probe.json"
    payload = {"ok": True, "kind": "env12_value_peak_drop_contract_probe"}
    output_path.write_text(json.dumps(payload))

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "phase3_env12_value_peak_drop_contract_probe.py",
            "--output-json",
            str(output_path),
        ],
    )

    assert probe_mod.main() == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == payload
