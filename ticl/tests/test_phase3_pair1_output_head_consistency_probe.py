from ticl.analysis.phase3_pair1_output_head_consistency_probe import _localize


def test_localize_unrestored_validation_head_state() -> None:
    primary = {
        "config": {
            "restore_validation_policy_state": False,
        },
        "saved_validation_state_present": True,
    }
    repeat = {
        "config": {
            "restore_validation_policy_state": False,
        },
        "saved_validation_state_present": True,
    }
    compare = {
        "action_head_primary_vs_repeat": {"max_abs_diff": 0.5},
        "value_head_primary_vs_repeat": {"max_abs_diff": 0.25},
        "current_replay_primary_vs_repeat": {"max_abs_diff": 0.5},
        "primary_current_vs_saved_action_head": {"max_abs_diff": 0.5},
        "primary_current_vs_saved_value_head": {"max_abs_diff": 0.1},
        "repeat_current_vs_saved_action_head": {"max_abs_diff": 0.6},
        "repeat_current_vs_saved_value_head": {"max_abs_diff": 0.2},
    }

    localized = _localize(compare, primary, repeat)

    assert localized["stage"] == "unrestored_validation_head_state"
    assert localized["checkpoint_has_saved_validation_policy_state"] is True
    assert localized["contract_restore_validation_policy_state"] is False


def test_localize_head_consistent() -> None:
    primary = {
        "config": {
            "restore_validation_policy_state": False,
        },
        "saved_validation_state_present": True,
    }
    repeat = {
        "config": {
            "restore_validation_policy_state": False,
        },
        "saved_validation_state_present": True,
    }
    compare = {
        "action_head_primary_vs_repeat": {"max_abs_diff": 0.0},
        "value_head_primary_vs_repeat": {"max_abs_diff": 0.0},
        "current_replay_primary_vs_repeat": {"max_abs_diff": 0.0},
        "primary_current_vs_saved_action_head": {"max_abs_diff": 0.0},
        "primary_current_vs_saved_value_head": {"max_abs_diff": 0.0},
        "repeat_current_vs_saved_action_head": {"max_abs_diff": 0.0},
        "repeat_current_vs_saved_value_head": {"max_abs_diff": 0.0},
    }

    localized = _localize(compare, primary, repeat)

    assert localized["stage"] == "head_consistent"
