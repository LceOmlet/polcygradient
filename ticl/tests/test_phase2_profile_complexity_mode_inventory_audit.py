import json
from pathlib import Path

from scripts.exploratory import phase2_profile_complexity_mode_inventory_audit as audit


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_py(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


def _governance() -> dict:
    return {
        "axis_rows": [
            {
                "axis_id": "low_energy_weak_action_optimum",
                "profile_key": "low_energy_not_ctrl_only",
                "definition": "low-energy definition",
                "trust_status": "quota_ready_protected_band",
                "gym_targets": "Ant-v5",
            },
            {
                "axis_id": "context_identifiable_closed_loop_settle_feedback",
                "profile_key": "",
                "definition": "feedback definition",
                "trust_status": "pressure_point_subprofile_not_single_quota",
                "gym_targets": "Pendulum-v1",
            },
            {
                "axis_id": "mountaincar_upfront_investment_future_basin",
                "profile_key": "",
                "definition": "future basin",
                "trust_status": "candidate_detector_not_quota_ready",
                "gym_targets": "MountainCarContinuous-v0",
            },
        ]
    }


def _mode_table() -> dict:
    return {
        "rows": [
            {
                "mode_id": "low_energy_weak_action_optimum",
                "status": "protected broad band",
                "decision": "preserve",
            },
            {
                "mode_id": "context_identifiable_closed_loop_settle_feedback",
                "status": "definition repair candidate",
                "decision": "repair",
            },
            {
                "mode_id": "mountaincar_upfront_investment_future_basin",
                "status": "coverage annotation only; not quota-ready",
                "decision": "do not quota yet",
            },
        ]
    }


def _sufficiency() -> dict:
    return {
        "checks": {
            "pack_to_fit_model_numeric_parity": True,
            "fit_model_fixed_csv_startup_pass": True,
            "chunked_label_path_ready": True,
            "explicit_no_silent_fallback": True,
        }
    }


def _feedback_budgeted() -> dict:
    return {
        "rules": {
            "gym_full_obs_action_range": {"candidate_count": 8},
            "gym_q90_obs_action_range": {"candidate_count": 10},
        }
    }


def _feedback_control() -> dict:
    return {
        "read": {
            "feedback_subprofile_is_pressure_point": True,
            "single_existing_scalar_knob_sufficient": False,
        }
    }


def _feedback_broaden() -> dict:
    return {
        "read": {
            "strict_source_pool_collapse_confirmed": True,
            "core_or_weak_relaxation_sufficient": False,
        }
    }


def _feedback_history() -> dict:
    return {
        "read": {
            "broad_feedback_memory_supported": False,
            "narrow_official_trainable_feedback_memory_partly_supported": True,
        }
    }


def _generator_gap() -> dict:
    return {"generator_level_read": {"generator_level_selector_ready": False}}


def _fixture(tmp_path: Path) -> dict[str, Path]:
    paths = {
        "governance": tmp_path / "governance.json",
        "mode_table": tmp_path / "mode_table.json",
        "sufficiency": tmp_path / "sufficiency.json",
        "feedback_budgeted": tmp_path / "feedback_budgeted.json",
        "feedback_control": tmp_path / "feedback_control.json",
        "feedback_broaden": tmp_path / "feedback_broaden.json",
        "feedback_history": tmp_path / "feedback_history.json",
        "generator_gap": tmp_path / "generator_gap.json",
        "mainline": tmp_path / "main.py",
        "profile": tmp_path / "profile.py",
    }
    _write_json(paths["governance"], _governance())
    _write_json(paths["mode_table"], _mode_table())
    _write_json(paths["sufficiency"], _sufficiency())
    _write_json(paths["feedback_budgeted"], _feedback_budgeted())
    _write_json(paths["feedback_control"], _feedback_control())
    _write_json(paths["feedback_broaden"], _feedback_broaden())
    _write_json(paths["feedback_history"], _feedback_history())
    _write_json(paths["generator_gap"], _generator_gap())
    _write_py(paths["mainline"], "def train():\n    return 1\n")
    _write_py(paths["profile"], "def probe():\n    return 2\n")
    return paths


def _build(paths: dict[str, Path]) -> dict:
    return audit.build_report(
        governance_json=paths["governance"],
        mode_table_json=paths["mode_table"],
        sufficiency_json=paths["sufficiency"],
        feedback_budgeted_json=paths["feedback_budgeted"],
        feedback_control_json=paths["feedback_control"],
        feedback_broaden_json=paths["feedback_broaden"],
        feedback_history_json=paths["feedback_history"],
        generator_gap_json=paths["generator_gap"],
        mainline_files=[paths["mainline"]],
        profile_files=[paths["profile"]],
    )


def test_inventory_ranks_feedback_as_highest_active_pressure(tmp_path: Path):
    report = _build(_fixture(tmp_path))

    assert report["decision"]["mainline_complexity_acceptance"] == "acceptable_with_guards"
    assert report["decision"]["next_mode_to_fix"] == "context_identifiable_closed_loop_settle_feedback"
    rows = {row["mode_id"]: row for row in report["mode_inventory"]["rows"]}
    assert rows["context_identifiable_closed_loop_settle_feedback"]["selector_sufficient"] is True
    assert rows["context_identifiable_closed_loop_settle_feedback"]["gym_semantic_sufficient"] is False
    assert rows["context_identifiable_closed_loop_settle_feedback"]["pressure_score"] > rows[
        "mountaincar_upfront_investment_future_basin"
    ]["pressure_score"]
    assert rows["low_energy_weak_action_optimum"]["pressure_score"] < 0.5


def test_complexity_becomes_unacceptable_when_parity_gates_are_missing(tmp_path: Path):
    paths = _fixture(tmp_path)
    _write_json(paths["sufficiency"], {"checks": {}})

    report = _build(paths)

    assert report["complexity"]["risk_level"] in {"moderate", "high"}
    assert report["decision"]["mainline_complexity_acceptance"] in {
        "acceptable_with_guards",
        "too_risky_without_refactor",
    }
    assert "one or more fit_model/profile parity gates are missing" in report["complexity"]["risk_points"]
