import json
from pathlib import Path

from scripts.exploratory import phase2_mode_coverage_table_v2e as table


def test_v2e_mode_table_keeps_mountaincar_and_survival_out_of_quota(tmp_path: Path):
    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps(
            {
                "selected_summary": {
                    "gym_full_obs_action_range": {
                        "broad_axis_rates": {
                            "locomotion_witness": 0.3,
                            "low_energy_not_ctrl_only": 0.5,
                            "sign_or_phase_sensitive": 0.6,
                            "state_conditioned_energy_injection": 0.4,
                        },
                        "strict_feedback": {
                            "strict_count": 20,
                            "profile": {"decay": 13, "constant": 7},
                            "near_best_gain1_rate": 0.5,
                        },
                        "mountaincar_candidate_count": 7,
                        "mountaincar_strong_count": 1,
                    },
                    "gym_q90_obs_action_range": {
                        "broad_axis_rates": {
                            "locomotion_witness": 0.3,
                            "low_energy_not_ctrl_only": 0.4,
                            "sign_or_phase_sensitive": 0.6,
                            "state_conditioned_energy_injection": 0.5,
                        },
                        "strict_feedback": {
                            "strict_count": 20,
                            "profile": {"decay": 15, "constant": 5},
                            "near_best_gain1_rate": 0.35,
                        },
                        "mountaincar_candidate_count": 4,
                        "mountaincar_strong_count": 6,
                    },
                },
                "mechanisms": [
                    {
                        "id": "pendulum_closed_loop_settle_gap",
                        "current_read": {
                            "reward_relevant_state_audit": {
                                "cheap_suffix_state_settle_guard_ready": False,
                                "min_count": 0,
                                "min_ratio": 0.0,
                            }
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = table.build_report(spec)
    by_id = {row["mode_id"]: row for row in report["rows"]}
    assert "not quota-ready" in by_id["mountaincar_upfront_investment_future_basin"]["status"]
    assert by_id["terminal_survival_action_sensitive"]["v2e_mechanism"] == "none in v2e"
    assert "hard gain quota" in by_id["context_identifiable_closed_loop_settle_feedback"]["decision"]
    assert "not quota-ready" in by_id["pendulum_sustained_closed_loop_settle"]["status"]
    assert report["read"]["largest_semantic_blocker"].startswith("Pendulum")
