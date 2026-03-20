from copy import deepcopy
import random

import numpy as np
import torch

from ticl.model_configs import get_model_default_config
from ticl.models.encoders import Linear
from ticl.models.tabpfn import TabPFN
from ticl.priors.environment_prior import EnvironmentPrior
from ticl.rlpfn_maintained_path import resolve_rlpfn_token_layout, validate_rlpfn_maintained_path_config
from ticl.train import _build_policy_step_fn


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _small_exact_scm_env_cfg():
    cfg = deepcopy(get_model_default_config("rlpfn"))
    env_cfg = deepcopy(cfg["prior"]["environment"])
    env_cfg.update(
        {
            "state_dim": {"distribution": "uniform_int", "min": 8, "max": 8},
            "obs_dim": {"distribution": "uniform_int", "min": 6, "max": 6},
            "action_dim": {"distribution": "uniform_int", "min": 3, "max": 3},
            "noise_dim": {"distribution": "uniform_int", "min": 2, "max": 2},
            "zero_pad_dim": {"distribution": "uniform_int", "min": 0, "max": 0},
            "reward_dropout_randomize": False,
            "reward_dropout_ratio_min": 0.5,
            "reward_dropout_ratio_max": 0.5,
            "action_noise_train_std": 0.05,
            "action_noise_eval_std": 0.03,
        }
    )
    return cfg, env_cfg


def _capture_exact_scm_get_batch_trace():
    _seed_everything(11)
    cfg, env_cfg = _small_exact_scm_env_cfg()
    prior = EnvironmentPrior(env_cfg)
    num_features = int(cfg["prior"]["num_features"])
    x, y, _ = prior.get_batch(
        batch_size=2,
        n_samples=12,
        num_features=num_features,
        device="cpu",
        single_eval_pos=5,
    )
    info = prior.last_runtime_info
    return {
        "x_shape": tuple(x.shape),
        "y_shape": tuple(y.shape),
        "x00_head": [round(float(v), 6) for v in x[0, 0, :10]],
        "x51_head": [round(float(v), 6) for v in x[5, 1, :10]],
        "x00_meta": [round(float(v), 6) for v in x[0, 0, 400:408]],
        "x51_meta": [round(float(v), 6) for v in x[5, 1, 400:408]],
        "y_head": [round(float(v), 6) for v in y[:6, 0]],
        "y_tail": [round(float(v), 6) for v in y[-3:, 1]],
        "x_sum": round(float(x.sum()), 6),
        "y_sum": round(float(y.sum()), 6),
        "info0": {
            "family": info[0]["family"],
            "transition_reference_mode": info[0]["transition_reference_mode"],
            "strict_joint_transition_enabled": bool(info[0]["strict_joint_transition_enabled"]),
            "reference_semantics_enabled": bool(info[0]["reference_semantics_enabled"]),
            "reward_drop_frac_realized": round(float(info[0]["reward_drop_frac_realized"]), 6),
        },
        "info1": {
            "family": info[1]["family"],
            "transition_reference_mode": info[1]["transition_reference_mode"],
            "strict_joint_transition_enabled": bool(info[1]["strict_joint_transition_enabled"]),
            "reference_semantics_enabled": bool(info[1]["reference_semantics_enabled"]),
            "reward_drop_frac_realized": round(float(info[1]["reward_drop_frac_realized"]), 6),
        },
    }


def _capture_policy_step_trace():
    _seed_everything(20260320)
    model = TabPFN(
        n_out=1,
        n_features=434,
        emsize=32,
        nhead=1,
        nhid_factor=2,
        nlayers=1,
        y_encoder_layer=Linear(1, emsize=32),
        classification_task=False,
        y_encoder="linear",
        x_encoder_type="split_obs_action",
        x_obs_dim=404,
        x_action_dim=30,
        single_eval_causal=True,
    )
    model.eval()
    step_fn = _build_policy_step_fn(
        model,
        num_features=434,
        max_cache_len=16,
        kv_cache_mode="immutable",
        kv_cache_page_size=None,
        allow_grad_mutable_cache=False,
        pg_torch_compile=False,
    )

    obs_t = torch.linspace(-1.0, 1.0, steps=800, dtype=torch.float32).reshape(2, 400)
    action_t = torch.linspace(-0.5, 0.5, steps=60, dtype=torch.float32).reshape(2, 30)
    reward_t = torch.tensor([[0.25], [-0.75]], dtype=torch.float32)
    reward_mask_t = torch.ones((2, 1), dtype=torch.float32)
    phase_t = torch.tensor([[0.0], [1.0]], dtype=torch.float32)
    terminal_t = torch.tensor([[0.0], [1.0]], dtype=torch.float32)
    env_info = {
        "obs_slot_dim": 400,
        "action_slot_dim": 30,
        "action_dim": 30,
        "phase_t": phase_t,
        "terminal_t": terminal_t,
    }

    action_1, cache_1 = step_fn(obs_t, action_t, reward_t, reward_mask_t, None, 0, env_info)
    action_2, cache_2 = step_fn(obs_t.flip(-1), -action_t, reward_t * -0.5, reward_mask_t, cache_1, 1, env_info)

    return {
        "a1_head": [round(float(v), 6) for v in action_1[0, :8].detach()],
        "a1_row1_head": [round(float(v), 6) for v in action_1[1, :8].detach()],
        "a2_head": [round(float(v), 6) for v in action_2[0, :8].detach()],
        "a2_row1_head": [round(float(v), 6) for v in action_2[1, :8].detach()],
        "a1_sum": round(float(action_1.detach().sum()), 6),
        "a2_sum": round(float(action_2.detach().sum()), 6),
        "cache1_type": type(cache_1).__name__,
        "cache2_type": type(cache_2).__name__,
    }


def test_rlpfn_maintained_path_default_contract():
    cfg = get_model_default_config("rlpfn")
    layout = validate_rlpfn_maintained_path_config(cfg)

    assert cfg["prior"]["prior_type"] == "environment_only"
    assert cfg["prior"]["num_features"] == 434
    assert cfg["prior"]["environment"]["family"] == {
        "distribution": "meta_choice",
        "choice_values": ["scm"],
    }
    assert cfg["prior"]["environment"]["strict_joint_transition_enabled"] is True
    assert cfg["prior"]["environment"]["batch_parallel_backend"] == "torch_vectorized"
    assert cfg["prior"]["environment"]["batch_vectorized_grouping"] == "family"
    assert cfg["prior"]["environment"]["pg_one_hop_replay_enabled"] is True
    assert cfg["prior"]["environment"]["alpha_grad_one_hop_replay_enabled"] is True
    assert cfg["prior"]["environment"]["pg_markov_adjacent_replay_enabled"] is True
    assert cfg["optimizer"]["rl_objective"] == "alpha_grad"
    assert cfg["transformer"]["x_encoder_type"] == "split_obs_action"
    assert cfg["transformer"]["x_obs_dim"] == 404
    assert cfg["transformer"]["x_action_dim"] == 30
    assert cfg["transformer"]["single_eval_causal"] is True
    assert layout["default_num_features"] == 434
    assert layout["x_obs_dim"] == 404
    assert layout["x_action_dim"] == 30


def test_rlpfn_maintained_token_layout_helper_matches_terminal_toggle():
    _, env_cfg = _small_exact_scm_env_cfg()
    with_terminal = resolve_rlpfn_token_layout(env_cfg)
    assert with_terminal["terminal_token_enabled"] is True
    assert with_terminal["x_obs_dim"] == 404
    assert with_terminal["default_num_features"] == 434

    env_cfg_no_terminal = deepcopy(env_cfg)
    env_cfg_no_terminal["terminal_reset_enabled"] = False
    without_terminal = resolve_rlpfn_token_layout(env_cfg_no_terminal)
    assert without_terminal["terminal_token_enabled"] is False
    assert without_terminal["x_obs_dim"] == 403
    assert without_terminal["default_num_features"] == 433


def test_rlpfn_maintained_path_exact_scm_get_batch_trace_matches_golden():
    expected = {
        "x_shape": (12, 2, 434),
        "y_shape": (12, 2),
        "x00_head": [0.007877, -0.016809, -0.007063, -0.010912, -0.019034, 0.024015, 0.0, 0.0, 0.0, 0.0],
        "x51_head": [1.068617, -1.159532, 0.061322, 1.817357, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "x00_meta": [0.0, 1.0, 0.0, 0.0, -0.044135, 0.191324, 0.028874, 0.0],
        "x51_meta": [0.895848, 1.0, 1.0, 0.0, 0.131627, -0.186655, 0.430441, 0.0],
        "y_head": [0.012206, 0.007162, 0.014864, 0.015489, 0.011253, 0.015846],
        "y_tail": [-0.367191, -1.10667, 1.206445],
        "x_sum": 52.351818,
        "y_sum": 4.157647,
        "info0": {
            "family": "scm",
            "transition_reference_mode": "scm_exact",
            "strict_joint_transition_enabled": True,
            "reference_semantics_enabled": True,
            "reward_drop_frac_realized": 0.0,
        },
        "info1": {
            "family": "scm",
            "transition_reference_mode": "scm_exact",
            "strict_joint_transition_enabled": True,
            "reference_semantics_enabled": True,
            "reward_drop_frac_realized": 0.0,
        },
    }

    assert _capture_exact_scm_get_batch_trace() == expected


def test_rlpfn_maintained_path_policy_step_trace_matches_golden():
    expected = {
        "a1_head": [-0.056622, -0.051079, -0.140691, -0.325196, 0.247738, -0.091861, -0.094656, 0.066978],
        "a1_row1_head": [0.016508, -0.106914, -0.203769, 0.047674, 0.240656, 0.216213, -0.165484, 0.05801],
        "a2_head": [0.057632, -0.017828, -0.11649, -0.306731, 0.240711, 0.103525, -0.366433, 0.115897],
        "a2_row1_head": [0.233536, -0.160705, -0.223101, -0.164249, 0.138238, 0.335276, -0.364536, -0.056289],
        "a1_sum": 1.541568,
        "a2_sum": 2.278063,
        "cache1_type": "list",
        "cache2_type": "list",
    }

    assert _capture_policy_step_trace() == expected
