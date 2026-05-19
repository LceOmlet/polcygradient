from scripts.exploratory import phase2_pendulum_like_signature_fixedlist_builder as builder


def test_strict_or_time_profile_definition_uses_effective_temporal_mode():
    row = {
        "pendulum_like_strict": False,
        "time_profile_core": True,
        "best_feedback_mode": "obs_linear_0x200",
        "best_time_profile_mode": "decay_obs_linear_1x100",
        "best_time_profile_env_return": 8.0,
        "best_open_loop_env_return": 3.0,
        "feedback_pair_phase_gap_env": 2.0,
        "feedback_suffix_gain_over_zero_env": 1.0,
    }

    assert builder._label_match(row, "strict_or_time_profile_core_no_random") is True
    assert builder._effective_feedback_source(row, "strict_or_time_profile_core_no_random") == (
        "time_profile_core_no_random"
    )
    assert builder._effective_feedback_mode(row, "strict_or_time_profile_core_no_random") == (
        "decay_obs_linear_1x100"
    )
    assert builder._score(row, "strict_or_time_profile_core_no_random") == 8.0


def test_strict_or_time_profile_definition_excludes_random_temporal_shortcut():
    row = {
        "pendulum_like_strict": False,
        "time_profile_core": True,
        "best_time_profile_mode": "early_random_then_zero",
    }

    assert builder._label_match(row, "strict_or_time_profile_core_no_random") is False
