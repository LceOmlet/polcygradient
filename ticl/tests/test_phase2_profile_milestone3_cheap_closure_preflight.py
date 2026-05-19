import json
from pathlib import Path

from scripts.exploratory import phase2_profile_milestone3_cheap_closure_preflight as preflight


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _source() -> dict:
    return {
        "sampling": {"raw_seen": 8192},
        "selected_count_by_rule": {
            "gym_full_obs_action_range": 384,
            "gym_q90_obs_action_range": 384,
        },
        "pendulum_settle_yield_axis": {
            "selected_counts_by_rule": {
                rule: {
                    "enabled": 96,
                    "bin_counts": {"bin_00": 24, "bin_01": 24, "bin_02": 24, "bin_03": 24},
                }
                for rule in preflight.RULES
            }
        },
    }


def _basin() -> dict:
    return {
        "decision": {"axis_supported_controllable_basin_ready": True},
        "lists": [
            {
                "coverage": {
                    "axis_supported_sustained_settle_count": 96,
                    "axis_supported_sustained_settle_rate": 0.25,
                }
            }
            for _ in preflight.RULES
        ],
    }


def _label() -> dict:
    return {
        "read": {
            "generator_cached_label_schema_ready": True,
            "pendulum_axis_support_populated": True,
            "pendulum_generator_label_populated": True,
        }
    }


def _selector() -> dict:
    return {
        "config": {"profile_version_id": "profile_band_v2e_source384_action_controls_position_cheap_state_guard"},
        "rules": {
            rule: {
                "guard_pass": True,
                "profile_guard_pass": True,
                "selected_summary": {
                    "strict_feedback_count": 20,
                    "strict_feedback_gain_counts": {"1.0": 20},
                    "pendulum_axis_supported_sustained_settle_count": 9,
                    "pendulum_settle_effective_support_count": 13,
                },
            }
            for rule in preflight.RULES
        },
    }


def _smoke() -> dict:
    return {
        "hard_pass": True,
        "scopes": {
            rule: {
                "hard_pass": True,
                "checks": {"repair_gap_positive": True},
                "settle_gap": {"gap_active": {"min": 0.014}},
                "transition_diversity": {"steps": 32},
            }
            for rule in preflight.RULES
        },
    }


def _parity() -> dict:
    return {
        "hard_pass": True,
        "cases": [
            {
                "case": rule,
                "rule": rule,
                "hard_pass": True,
                "selected_pendulum_settle_yield_repair_count": 4,
                "update_parity": {
                    "gradient_diff": {"grad_delta_max_abs": 0.0},
                    "one_train_update_compare": {
                        "post_update_param_diff": {"policy_param_max_abs_diff": 0.0}
                    },
                },
            }
            for rule in preflight.RULES
        ],
    }


def _contract() -> dict:
    return {
        "analysis_entry": "phase2_profile_milestone3_cheap_contract",
        "decision": {
            "cheap_contract_formalized": True,
            "mountaincar_scope_decided": True,
            "blockers": [],
        },
        "cheap_contract": {
            "selector_profile_version_id": "profile_band_v2e_source384_action_controls_position_cheap_state_guard",
            "sustained_smoke_contract": {
                "min_gap": 0.01,
                "old_one_step_min_gap_0_20_is_not_the_gate": True,
            },
            "stability_contract": {"same_fixed_group_min_steps": 128},
        },
        "mountaincar_scope": {"decision": "out_of_scope_for_cheap_m3"},
    }


def _top_level_closure() -> dict:
    return {
        "analysis_entry": "phase2_profile_closure_progress_plan",
        "decision": {
            "milestone_tracks": {
                "cheap_milestone3": {
                    "consumed_preflight": True,
                    "ready_to_land": False,
                    "hard_blockers": ["top_level_gate_not_integrated"],
                }
            }
        },
        "diagnostics": {
            "cheap_milestone3_preflight_read": {
                "analysis_entry": "phase2_profile_milestone3_cheap_closure_preflight"
            }
        },
    }


def _paths(tmp_path: Path) -> dict[str, Path]:
    payloads = {
        "source_json": _source(),
        "basin_json": _basin(),
        "label_json": _label(),
        "selector_json": _selector(),
        "smoke_json": _smoke(),
        "stability_smoke_json": {},
        "parity_json": _parity(),
    }
    return {key: _write(tmp_path / f"{key}.json", value) for key, value in payloads.items()}


def test_cheap_closure_preflight_keeps_formalization_and_stability_as_blockers(tmp_path: Path):
    report = preflight.build_report(**_paths(tmp_path))

    assert report["components"]["source_generation"]["passed"] is True
    assert report["components"]["action_state_basin_semantics"]["passed"] is True
    assert report["components"]["selector_materialization"]["passed"] is True
    assert report["components"]["integrated_health_smoke"]["passed"] is True
    assert report["components"]["pack_fit_parity"]["passed"] is True
    assert report["decision"]["ready_to_land_milestone3_cheap"] is False
    assert "cheap_contract_not_formalized" in report["decision"]["hard_blockers"]
    assert "cross_seed_or_longer_stability_missing" in report["decision"]["hard_blockers"]
    assert report["score"]["weighted_progress"] < report["score"]["milestone_threshold"]


def test_cheap_closure_preflight_accepts_longer_same_fixed_group_smoke(tmp_path: Path):
    paths = _paths(tmp_path)
    stability = _smoke()
    for scope in stability["scopes"].values():
        scope["transition_diversity"]["steps"] = 128
    paths["stability_smoke_json"].write_text(json.dumps(stability), encoding="utf-8")

    report = preflight.build_report(**paths)

    assert report["components"]["cross_seed_or_longer_stability"]["passed"] is True
    assert "cross_seed_or_longer_stability_missing" not in report["decision"]["hard_blockers"]


def test_cheap_closure_preflight_closes_after_contract_and_top_level_integration(tmp_path: Path):
    paths = _paths(tmp_path)
    stability = _smoke()
    for scope in stability["scopes"].values():
        scope["transition_diversity"]["steps"] = 128
    paths["stability_smoke_json"].write_text(json.dumps(stability), encoding="utf-8")
    paths["cheap_contract_json"] = _write(tmp_path / "contract.json", _contract())
    paths["top_level_closure_json"] = _write(tmp_path / "top_level.json", _top_level_closure())

    report = preflight.build_report(**paths)

    assert report["components"]["cheap_contract_formalization"]["passed"] is True
    assert report["components"]["top_level_gate_integration"]["passed"] is True
    assert report["components"]["mountaincar_scope_decision"]["passed"] is True
    assert report["decision"]["hard_blockers"] == []
    assert report["decision"]["ready_to_land_milestone3_cheap"] is True


def test_cheap_closure_preflight_blocks_nonuniform_axis_bins(tmp_path: Path):
    paths = _paths(tmp_path)
    source = _source()
    source["pendulum_settle_yield_axis"]["selected_counts_by_rule"]["gym_full_obs_action_range"]["bin_counts"][
        "bin_00"
    ] = 20
    paths["source_json"].write_text(json.dumps(source), encoding="utf-8")

    report = preflight.build_report(**paths)

    assert report["components"]["source_generation"]["passed"] is False
    assert "source_generation_not_milestone_ready" in report["decision"]["blockers"]
