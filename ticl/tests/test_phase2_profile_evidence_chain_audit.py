from scripts.exploratory import phase2_profile_evidence_chain_audit as audit


def _selector_report(*, sign_count: int, state_count: int, low_energy_count: int = 33):
    return {
        "rules": {
            "gym_full_obs_action_range": {
                "selected_summary": {
                    "n": 64,
                    "locomotion_witness_count": 19,
                    "low_energy_not_ctrl_only_count": low_energy_count,
                    "sign_or_phase_sensitive_count": sign_count,
                    "state_conditioned_energy_injection_count": state_count,
                    "strict_feedback_count": 20,
                }
            },
            "gym_q90_obs_action_range": {
                "selected_summary": {
                    "n": 64,
                    "locomotion_witness_count": 19,
                    "low_energy_not_ctrl_only_count": 26,
                    "sign_or_phase_sensitive_count": 45,
                    "state_conditioned_energy_injection_count": 36,
                    "strict_feedback_count": 20,
                }
            },
        }
    }


def _reference_summary(*, sign_count: int, state_count: int, low_energy_count: int = 21):
    return {
        "gym_full_obs_action_range": {
            "n": 64,
            "locomotion_witness": 19,
            "low_energy_not_ctrl_only": low_energy_count,
            "sign_or_phase_sensitive": sign_count,
            "state_conditioned_energy_injection": state_count,
            "strict_feedback_count": 16,
        },
        "gym_q90_obs_action_range": {
            "n": 64,
            "locomotion_witness": 17,
            "low_energy_not_ctrl_only": 19,
            "sign_or_phase_sensitive": 48,
            "state_conditioned_energy_injection": 29,
            "strict_feedback_count": 17,
        },
    }


def test_reference_more_sign_phase_is_conservative_not_failure():
    result = audit._cheap_reference_compare(
        _selector_report(sign_count=44, state_count=28),
        _reference_summary(sign_count=59, state_count=26),
    )

    assert result["hard_pass"] is True
    assert result["hard_failures"] == []
    assert result["low_energy_calibration_warning"] is True


def test_reference_non_low_energy_collapse_is_failure():
    result = audit._cheap_reference_compare(
        _selector_report(sign_count=44, state_count=28),
        _reference_summary(sign_count=59, state_count=10),
    )

    assert result["hard_pass"] is False
    assert result["hard_failures"] == [
        "gym_full_obs_action_range:state_conditioned_energy_injection:cheap=28,reference=10"
    ]
