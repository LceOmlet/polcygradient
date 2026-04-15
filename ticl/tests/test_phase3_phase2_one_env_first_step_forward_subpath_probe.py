from ticl.analysis.phase3_phase2_one_env_first_step_forward_subpath_probe import (
    _localize_forward_drift,
)


def test_localize_forward_drift_prefers_hidden_over_later_heads():
    compare = {
        "encoded_token": {"max_abs_diff": 0.0},
        "rwkv_input_token": {"max_abs_diff": 0.0},
        "rwkv_forward_hidden": {"max_abs_diff": 1.5e-3},
        "latent_pi": {"max_abs_diff": 2.0e-3},
        "latent_vf": {"max_abs_diff": 3.0e-3},
        "action_distribution_mean": {"max_abs_diff": 4.0e-3},
        "value_logits": {"max_abs_diff": 5.0e-3},
    }
    result = _localize_forward_drift(compare)
    assert result["stage"] == "rwkv_forward_step_path"
    assert result["rwkv_forward_hidden_max_abs_diff"] == 1.5e-3


def test_localize_forward_drift_can_pick_distribution_or_value_head():
    compare = {
        "encoded_token": {"max_abs_diff": 0.0},
        "rwkv_input_token": {"max_abs_diff": 0.0},
        "rwkv_forward_hidden": {"max_abs_diff": 0.0},
        "latent_pi": {"max_abs_diff": 0.0},
        "latent_vf": {"max_abs_diff": 0.0},
        "action_distribution_mean": {"max_abs_diff": 1.25e-5},
        "value_logits": {"max_abs_diff": 0.0},
    }
    result = _localize_forward_drift(compare)
    assert result["stage"] == "distribution_or_value_head_path"


def test_localize_forward_drift_can_pick_encoded_token():
    compare = {
        "encoded_token": {"max_abs_diff": 0.25},
        "rwkv_input_token": {"max_abs_diff": 0.0},
        "rwkv_forward_hidden": {"max_abs_diff": 0.0},
        "latent_pi": {"max_abs_diff": 0.0},
        "latent_vf": {"max_abs_diff": 0.0},
        "action_distribution_mean": {"max_abs_diff": 0.0},
        "value_logits": {"max_abs_diff": 0.0},
    }
    result = _localize_forward_drift(compare)
    assert result["stage"] == "encoded_token_path"
