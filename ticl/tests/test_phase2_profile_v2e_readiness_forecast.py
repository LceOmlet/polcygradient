import json
from pathlib import Path

from scripts.exploratory import phase2_profile_v2e_readiness_forecast as forecast


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_v2e_readiness_forecast_flags_missing_fit_model_support(tmp_path: Path):
    spec = tmp_path / "spec.json"
    table = tmp_path / "table.json"
    smoke = tmp_path / "smoke.json"
    decision = tmp_path / "decision.json"
    maintained = tmp_path / "rlpfn_maintained_path.py"
    cli = tmp_path / "cli_parsing.py"
    train = tmp_path / "train.py"
    parity = tmp_path / "parity.py"
    v2e_parity = tmp_path / "v2e_parity.py"
    v2e_parity_json = tmp_path / "v2e_parity.json"
    generator_gap_json = tmp_path / "generator_gap.json"

    _write_json(
        spec,
        {
            "v2e_definition": {
                "short_name": forecast.V2E_ID,
                "does_not_include": ["hard gain quota"],
            },
            "selected_summary": {
                "gym_full_obs_action_range": {
                    "n": 64,
                    "joint_active_cells": 15,
                    "joint_max_cell_share": 0.18,
                }
            },
        },
    )
    _write_json(
        table,
        {
            "rows": [
                {"mode_id": "low_energy_weak_action_optimum", "v2e_mechanism": "profile_band_base"},
                {"mode_id": "sign_phase_sensitive_action", "v2e_mechanism": "profile_band_base"},
                {"mode_id": "state_conditioned_energy_injection", "v2e_mechanism": "profile_band_base"},
                {"mode_id": "locomotion_sustained_energy_tradeoff", "v2e_mechanism": "profile_band_base"},
                {
                    "mode_id": "context_identifiable_closed_loop_settle_feedback",
                    "v2e_mechanism": "feedback_source_pool_repair + feedback_near_best_gain_guard",
                },
                {
                    "mode_id": "mountaincar_upfront_investment_future_basin",
                    "v2e_mechanism": "mountaincar_future_basin_detector",
                },
            ]
        },
    )
    _write_json(
        smoke,
        {
            "overall": {
                "candidate_smoke_guard_pass": True,
                "profile_guard_pass": True,
                "diversity_guard_pass": True,
                "trainability_delta_guard_pass": True,
            },
            "scopes": {},
        },
    )
    _write_json(
        decision,
        {"v2e_candidate_read": {"profile_definition_close_enough_for_candidate": True}},
    )
    maintained.write_text("sampled_topology gated_reward_path_balance\n", encoding="utf-8")
    cli.write_text("choices=['sampled_topology']\n", encoding="utf-8")
    train.write_text("normalize_prior_milestone('sampled_topology')\n", encoding="utf-8")
    parity.write_text("sampled_topology\n", encoding="utf-8")
    v2e_parity.write_text("wrapper\n", encoding="utf-8")

    report = forecast.build_report(
        spec_json=spec,
        table_json=table,
        smoke_json=smoke,
        decision_json=decision,
        maintained_path=maintained,
        cli_parsing=cli,
        train_py=train,
        parity_script=parity,
        v2e_parity_script=v2e_parity,
        v2e_parity_json=v2e_parity_json,
        generator_gap_json=generator_gap_json,
    )

    assert report["contract"]["delta_sign_fraction_not_a_gate"] is True
    assert report["fit_model_support_read"]["fit_model_startup_option_present"] is False
    assert report["fit_model_support_read"]["pack_to_fit_model_v2e_parity_present"] is False
    assert report["fit_model_support_read"]["exploratory_v2e_parity_script_present"] is True
    assert report["v2e_numeric_parity_read"]["available"] is False
    assert report["generator_level_gap_read"]["available"] is False
    assert "pack-to-fit_model" in report["next_highest_value_step"]
    assert report["readiness"]["candidate_profile_readiness_estimate"] < 1.0


def test_v2e_readiness_forecast_recognizes_parity_and_startup_references(tmp_path: Path):
    spec = tmp_path / "spec.json"
    table = tmp_path / "table.json"
    smoke = tmp_path / "smoke.json"
    decision = tmp_path / "decision.json"
    maintained = tmp_path / "rlpfn_maintained_path.py"
    cli = tmp_path / "cli_parsing.py"
    train = tmp_path / "train.py"
    parity = tmp_path / "parity.py"
    v2e_parity = tmp_path / "v2e_parity.py"
    v2e_parity_json = tmp_path / "v2e_parity.json"
    generator_gap_json = tmp_path / "generator_gap.json"

    _write_json(spec, {"v2e_definition": {"short_name": forecast.V2E_ID}, "selected_summary": {}})
    _write_json(table, {"rows": []})
    _write_json(smoke, {"overall": {}, "scopes": {}})
    _write_json(decision, {"v2e_candidate_read": {}})
    for path in (maintained, cli, train, parity):
        path.write_text(f"{forecast.V2E_ID}\n", encoding="utf-8")
    v2e_parity.write_text("wrapper\n", encoding="utf-8")
    _write_json(
        v2e_parity_json,
        {
            "hard_pass": True,
            "cases": [
                {
                    "hard_pass": True,
                    "rule": "gym_full_obs_action_range",
                    "n_envs": 2,
                    "n_steps": 16,
                    "update_parity": {
                        "gradient_diff": {"grad_delta_max_abs": 0.0},
                        "one_train_update_compare": {
                            "post_update_param_diff": {"policy_param_max_abs_diff": 0.0}
                        },
                    },
                }
            ],
        },
    )
    _write_json(
        generator_gap_json,
        {
            "generator_level_read": {
                "generator_level_selector_ready": False,
                "offline_label_axes": ["locomotion_witness"],
                "online_label_axes": ["reward_path_topology_health"],
                "why_not_ready": "offline labels still need online ports",
            }
        },
    )

    report = forecast.build_report(
        spec_json=spec,
        table_json=table,
        smoke_json=smoke,
        decision_json=decision,
        maintained_path=maintained,
        cli_parsing=cli,
        train_py=train,
        parity_script=parity,
        v2e_parity_script=v2e_parity,
        v2e_parity_json=v2e_parity_json,
        generator_gap_json=generator_gap_json,
    )

    assert report["fit_model_support_read"]["fit_model_startup_option_present"] is True
    assert report["fit_model_support_read"]["pack_to_fit_model_v2e_parity_present"] is True
    assert report["v2e_numeric_parity_read"]["hard_pass"] is True
    assert report["generator_level_gap_read"]["generator_level_selector_ready"] is False
    assert "online candidate-pool" in report["next_highest_value_step"]
