from pathlib import Path

import pytest

from scripts.exploratory import phase2_profile_governance_table as gov


def test_profile_stats_is_profile_first_and_does_not_consume_raw_delta():
    rows = [
        {
            "locomotion_witness": True,
            "low_energy_not_ctrl_only": False,
            "sign_or_phase_sensitive": True,
            "state_conditioned_energy_injection": False,
            "raw_reward_sum_delta": -1000.0,
        },
        {
            "locomotion_witness": False,
            "low_energy_not_ctrl_only": True,
            "sign_or_phase_sensitive": True,
            "state_conditioned_energy_injection": True,
            "raw_reward_sum_delta": 1000.0,
        },
    ]

    stats = gov._profile_stats(rows)

    assert stats["mode_rates"]["locomotion_witness"] == 0.5
    assert stats["mode_rates"]["low_energy_not_ctrl_only"] == 0.5
    assert stats["joint_active_cells"] == 2
    assert "raw_reward_sum_delta" not in stats


def test_feedback_and_mountaincar_are_not_single_scalar_quota_axes():
    by_id = {axis["axis_id"]: axis for axis in gov.AXIS_REGISTRY}

    feedback = by_id["context_identifiable_closed_loop_settle_feedback"]
    mountaincar = by_id["mountaincar_upfront_investment_future_basin"]

    assert feedback["profile_key"] is None
    assert feedback["control_role"] == "subprofile_guard_needed"
    assert "single_quota" in feedback["trust_status"]
    assert mountaincar["profile_key"] is None
    assert mountaincar["trust_status"] == "candidate_detector_not_quota_ready"
    assert mountaincar["control_role"] == "log_only_until_gym_anchor_and_health_guard"


def test_build_report_contract_marks_raw_delta_as_guard_when_artifacts_exist():
    required = [
        gov.UNIFIED_AUDIT_JSON,
        gov.PUSH2_CSV,
        gov.PUSH2_SMOKE_JSON,
        gov.STRICT_FRONTIER_JSON,
    ]
    missing = [str(path) for path in required if not Path(path).exists()]
    if missing:
        pytest.skip(f"local exploratory artifacts are absent: {missing}")

    report = gov.build_report()

    assert report["contract"]["raw_delta_is_guard_not_objective"] is True
    assert report["contract"]["does_not_train"] is True
    assert report["contract"]["does_not_modify_exact_scm"] is True
    assert report["profile_versions"]["profile_balanced_push2"]["status"] == (
        "exploratory_candidate_not_milestone"
    )
    assert report["next_read"]["closest_to_next_milestone"] is False
