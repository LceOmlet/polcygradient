from scripts.exploratory import phase2_profile_milestone_readiness as readiness


def test_readiness_keeps_feedback_and_mountaincar_as_blockers():
    gov = {
        "contract": {"does_not_modify_exact_scm": True},
        "axis_rows": [
            {"axis_id": "low_energy_weak_action_optimum", "trust_status": "quota_ready_protected_band"},
            {"axis_id": "sign_phase_sensitive_action", "trust_status": "quota_ready_protected_band_but_not_sufficient"},
            {"axis_id": "state_conditioned_energy_injection", "trust_status": "diagnostic_broad_axis"},
            {"axis_id": "locomotion_sustained_energy_tradeoff", "trust_status": "quota_ready_only_with_joint_profile_guard"},
            {"axis_id": "mountaincar_upfront_investment_future_basin", "trust_status": "missing_detector"},
        ],
        "profile_versions": {
            "milestone1_sampled_topology_conditioned": {"status": "protected_milestone"},
            "milestone2_gated_reward_path_balance": {"status": "protected_milestone"},
            "profile_balanced_push2": {
                "profile": {
                    "full": {"joint_active_cells": 15, "joint_max_cell_share": 0.18},
                    "q90": {"joint_active_cells": 14, "joint_max_cell_share": 0.18},
                },
                "guards": {
                    "full": {"diversity_guard_pass": True},
                    "q90": {"diversity_guard_pass": True},
                },
            },
            "strict_feedback_profile_gain_frontier": {
                "subprofile": {
                    "control_audit": {
                        "read": {"feedback_subprofile_is_pressure_point": True},
                        "simple_existing_scalar_knob_sufficient": False,
                    }
                }
            },
        },
    }
    selector = {
        "analysis_entry": "phase2_profile_band_selector",
        "contract": {"selection_wrapper_only": True, "raw_delta_not_used_in_selection": True},
    }

    report = readiness.build_report_from_loaded(gov, selector, {})

    by_id = {item["id"]: item for item in report["sufficiency_items"]}
    assert by_id["feedback_strict_subprofile_control_ready"]["score"] == 0.25
    assert by_id["mountaincar_future_basin_detector_ready"]["score"] == 0.0
    assert report["completion_band"] != "milestone_candidate_ready_for_parity"
