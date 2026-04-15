from ticl.analysis.phase3_phase2_one_env_fixed_suite_collect_drift_probe import (
    _compare_tensors,
    _localize_drift,
)


def test_compare_tensors_reports_exact_match_and_diff():
    same = _compare_tensors([1.0, 2.0], [1.0, 2.0])
    diff = _compare_tensors([1.0, 2.0], [1.0, 2.5])
    assert same["exact_match"] is True
    assert same["max_abs_diff"] == 0.0
    assert diff["exact_match"] is False
    assert diff["max_abs_diff"] == 0.5


def test_localize_drift_prefers_earliest_differing_stage():
    compare = {
        "reset_compare": {
            "last_obs": {"max_abs_diff": 0.25},
            "last_episode_starts": {"max_abs_diff": 0.0},
            "internal": {"_state_t": {"max_abs_diff": 0.0}},
            "reset_infos_exact_match": False,
        },
        "first_policy_step_compare": {
            "inputs": {"obs_t": {"max_abs_diff": 0.0}, "obs_full": {"max_abs_diff": 0.0}},
            "cache_inputs": {"root[0]": {"max_abs_diff": 0.0}},
            "env_info": {"action_dim_per_sample": {"max_abs_diff": 0.0}},
            "env_info_all_tensors": {"env_info.action_dim_per_sample": {"max_abs_diff": 0.0}},
            "actor_outputs": {"values": {"max_abs_diff": 1.25e-3}},
        },
        "buffer_step0_compare": {
            "actions": {"max_abs_diff": 0.0},
        },
    }
    result = _localize_drift(compare)
    assert result["stage"] == "model_forward_path"
    assert result["first_policy_step_actor_output_max_abs_diff"] == 1.25e-3


def test_localize_drift_marks_reset_as_side_path_when_first_step_matches():
    compare = {
        "reset_compare": {
            "last_obs": {"max_abs_diff": 0.25},
            "last_episode_starts": {"max_abs_diff": 0.0},
            "internal": {"_state_t": {"max_abs_diff": 0.1}},
            "reset_infos_exact_match": False,
        },
        "first_policy_step_compare": {
            "inputs": {"obs_t": {"max_abs_diff": 0.0}, "obs_full": {"max_abs_diff": 0.0}},
            "cache_inputs": {"root[0]": {"max_abs_diff": 0.0}},
            "env_info": {"action_dim_per_sample": {"max_abs_diff": 0.0}},
            "env_info_all_tensors": {"env_info.action_dim_per_sample": {"max_abs_diff": 0.0}},
            "actor_outputs": {"values": {"max_abs_diff": 0.0}},
        },
        "buffer_step0_compare": {
            "actions": {"max_abs_diff": 0.0},
        },
    }
    result = _localize_drift(compare)
    assert result["stage"] == "vec_env_reset_side_path"


def test_localize_drift_can_separate_obs_builder_and_model_forward():
    compare = {
        "reset_compare": {
            "last_obs": {"max_abs_diff": 0.0},
            "last_episode_starts": {"max_abs_diff": 0.0},
            "internal": {"_state_t": {"max_abs_diff": 0.0}},
            "reset_infos_exact_match": True,
        },
        "first_policy_step_compare": {
            "inputs": {"obs_t": {"max_abs_diff": 0.0}, "obs_full": {"max_abs_diff": 0.0}},
            "cache_inputs": {"root[0]": {"max_abs_diff": 0.0}},
            "env_info": {"action_dim_per_sample": {"max_abs_diff": 0.0}},
            "env_info_all_tensors": {"env_info.action_dim_per_sample": {"max_abs_diff": 0.0}},
            "actor_outputs": {"values": {"max_abs_diff": 0.25}},
        },
        "buffer_step0_compare": {
            "actions": {"max_abs_diff": 0.0},
        },
    }
    result = _localize_drift(compare)
    assert result["stage"] == "model_forward_path"
