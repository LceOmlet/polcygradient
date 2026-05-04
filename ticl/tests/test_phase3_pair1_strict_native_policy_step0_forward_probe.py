import torch

from ticl.analysis.phase3_pair1_strict_native_policy_step0_forward_probe import (
    _localize_forward_drift,
    _summarize_forward_compare,
)


def _capture_with(**overrides):
    base = {
        "step_idx": 0,
        "obs_full": torch.zeros((2, 3), dtype=torch.float32),
        "reward_token_y": torch.zeros((1, 2), dtype=torch.float32),
        "encoded_token": torch.zeros((1, 2, 4), dtype=torch.float32),
        "rwkv_input_token": torch.zeros((1, 2, 4), dtype=torch.float32),
        "rwkv_forward_token_arg": torch.zeros((2, 4), dtype=torch.float32),
        "rwkv_forward_hidden": torch.zeros((2, 4), dtype=torch.float32),
        "latent_pi": torch.zeros((2, 4), dtype=torch.float32),
        "latent_vf": torch.zeros((2, 4), dtype=torch.float32),
        "action_distribution_mean": torch.zeros((2, 2), dtype=torch.float32),
        "returned_action_mean": torch.zeros((2, 2), dtype=torch.float32),
        "value_logits": torch.zeros((2, 5), dtype=torch.float32),
        "returned_values": torch.zeros((2, 1), dtype=torch.float32),
        "action_dim_per_sample": torch.tensor([2, 2], dtype=torch.long),
        "init_state": [torch.zeros((2, 4), dtype=torch.float32)],
        "state_arg": [torch.zeros((2, 4), dtype=torch.float32)],
        "next_state": [torch.zeros((2, 4), dtype=torch.float32)],
        "path_flags": {
            "batch_size": 2,
            "token_device": "cpu",
            "token_dtype": "torch.float32",
            "token_is_cuda": False,
            "core_training": False,
            "torch_grad_enabled": False,
            "force_native_eval_forward_step": True,
        },
    }
    base.update(overrides)
    return {"capture": base}


def test_localize_encoded_token_drift() -> None:
    primary = _capture_with()
    repeat = _capture_with(encoded_token=torch.ones((1, 2, 4), dtype=torch.float32))

    compare = _summarize_forward_compare(primary, repeat)
    localized = _localize_forward_drift(compare)

    assert localized["stage"] == "encoded_token_path"
    assert localized["encoded_token_max_abs_diff"] > 0.0


def test_localize_rwkv_hidden_drift_when_earlier_paths_match() -> None:
    primary = _capture_with()
    repeat = _capture_with(rwkv_forward_hidden=torch.ones((2, 4), dtype=torch.float32))

    compare = _summarize_forward_compare(primary, repeat)
    localized = _localize_forward_drift(compare)

    assert localized["stage"] == "rwkv_forward_step_path"
    assert localized["rwkv_forward_hidden_max_abs_diff"] > 0.0
