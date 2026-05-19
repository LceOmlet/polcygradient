import csv
import json
from pathlib import Path

from scripts.exploratory import phase2_profile_v2e_mechanism_spec as spec


def _write_selected(path: Path) -> None:
    rows = []
    for rule in ("gym_full_obs_action_range", "gym_q90_obs_action_range"):
        for idx in range(64):
            strict = idx < 20
            rows.append(
                {
                    "rule": rule,
                    "locomotion_witness": str(idx % 2 == 0),
                    "low_energy_not_ctrl_only": str(idx % 3 == 0),
                    "sign_or_phase_sensitive": str(idx % 4 == 0),
                    "state_conditioned_energy_injection": str(idx % 5 == 0),
                    "strict_feedback_candidate": str(strict),
                    "strict_feedback_source_pool_definition": (
                        "strict_or_time_profile_core_no_random" if strict else ""
                    ),
                    "strict_feedback_effective_source": (
                        "time_profile_core_no_random" if idx % 2 == 0 else "best_feedback"
                    ),
                    "strict_feedback_profile": "decay" if idx % 2 == 0 else "constant",
                    "strict_feedback_gain": "1.0" if idx % 3 == 0 else "2.0",
                    "v2e_strict_near_best_gain_1": str(strict and idx % 2 == 0),
                    "v2e_strict_near_best_gain_0.5": str(strict and idx % 4 == 0),
                    "mountaincar_future_basin_candidate": str(idx == 0),
                    "mountaincar_future_basin_strong": str(idx == 1),
                }
            )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_v2e_mechanism_spec_keeps_quota_and_annotation_roles_separate(tmp_path: Path):
    selected = tmp_path / "selected.csv"
    source = tmp_path / "source.json"
    near = tmp_path / "near.json"
    smoke = tmp_path / "smoke.json"
    decision = tmp_path / "decision.json"
    closure = tmp_path / "closure.json"
    reward_relevant = tmp_path / "reward_relevant.json"
    _write_selected(selected)
    _write_json(source, {"read": {"temporal_union_repairs_profile": True}})
    _write_json(near, {"overall_read": {"any_source_pool_high_gain_bias": False}})
    _write_json(smoke, {"overall": {"candidate_smoke_guard_pass": True}})
    _write_json(
        decision,
        {
            "v2e_candidate_read": {
                "profile_definition_close_enough_for_candidate": True,
                "exclude_from_v2e_for_now": ["hard gain quota"],
            }
        },
    )
    _write_json(
        closure,
        {
            "disallowed_repairs": [
                {"name": "simple_reference_inertia_highway_knob", "reason": "failed screen"},
                {"name": "full_feedback_grid_in_fit_model_startup", "reason": "offline only"},
            ],
            "allowed_repairs": [
                {"name": "cheap_clean_pendulum_settle_guard", "status": "necessary"},
                {"name": "bounded_generator_level_selector", "status": "necessary"},
            ],
        },
    )
    _write_json(
        reward_relevant,
        {"decision": {"cheap_suffix_state_settle_guard_ready": False, "min_count": 0}},
    )

    report = spec.build_report(
        selected_csv=selected,
        source_pool_json=source,
        near_best_json=near,
        smoke_json=smoke,
        decision_json=decision,
        closure_register_json=closure,
        reward_relevant_state_json=reward_relevant,
    )

    by_id = {row["id"]: row for row in report["mechanisms"]}
    assert by_id["profile_band_base"]["is_quota"] is True
    assert by_id["feedback_source_pool_repair"]["is_quota"] is False
    assert by_id["feedback_near_best_gain_guard"]["is_quota"] is False
    assert by_id["mountaincar_future_basin_detector"]["is_quota"] is False
    assert by_id["pendulum_closed_loop_settle_gap"]["is_quota"] is False
    assert by_id["generator_level_selector_orchestrator"]["is_quota"] is False
    assert "simple_reference_inertia_highway_knob" in by_id["pendulum_closed_loop_settle_gap"]["current_read"]["disallowed_repairs"]
    assert report["contract"]["delta_sign_fraction_not_a_gate"] is True
    assert report["selected_summary"]["gym_full_obs_action_range"]["strict_feedback"][
        "source_pool_definition"
    ] == {"strict_or_time_profile_core_no_random": 20}
