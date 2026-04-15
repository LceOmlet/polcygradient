from ticl.analysis.phase3_phase2_one_env_rwkv_core_step_probe import (
    _localize_core_step_drift,
)


def _base_compare():
    return {
        "path_flags_exact_match": True,
        "token_arg": {"max_abs_diff": 0.0},
        "init_state_compare": {"max_abs_diff": 0.0},
        "state_arg_compare": {"max_abs_diff": 0.0},
        "official_flat_state_input_compare": {"max_abs_diff": 0.0},
        "official_weight_snapshot_compare": {"max_abs_diff": 0.0},
        "native_replay_hidden": {"max_abs_diff": 0.0},
        "actual_hidden": {"max_abs_diff": 0.0},
        "phase2_actual_vs_native_hidden": {"max_abs_diff": 0.0},
        "phase3_actual_vs_native_hidden": {"max_abs_diff": 0.0},
    }


def test_localize_core_step_drift_prefers_cache_state():
    compare = _base_compare()
    compare["state_arg_compare"] = {"max_abs_diff": 0.25}
    compare["actual_hidden"] = {"max_abs_diff": 0.5}
    result = _localize_core_step_drift(compare)
    assert result["stage"] == "core_cache_state"
    assert result["state_arg_max_abs_diff"] == 0.25


def test_localize_core_step_drift_can_pick_official_eval_kernel():
    compare = _base_compare()
    compare["actual_hidden"] = {"max_abs_diff": 0.015625}
    result = _localize_core_step_drift(compare)
    assert result["stage"] == "official_eval_kernel_path"
    assert result["actual_hidden_max_abs_diff"] == 0.015625


def test_localize_core_step_drift_can_pick_native_replay():
    compare = _base_compare()
    compare["native_replay_hidden"] = {"max_abs_diff": 1e-4}
    compare["actual_hidden"] = {"max_abs_diff": 0.015625}
    result = _localize_core_step_drift(compare)
    assert result["stage"] == "native_core_replay"
