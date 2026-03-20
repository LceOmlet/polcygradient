from copy import deepcopy
import random

import numpy as np
import torch

from ticl.model_configs import get_model_default_config
from ticl.models.encoders import Linear
from ticl.models.tabpfn import TabPFN
from ticl.priors.maintained_exact_scm import (
    coerce_bool,
    env_input_layout,
    env_obs_input_dim,
    env_uses_reference_semantics,
    expand_env_value_to_list,
    finalize_env_semantics_summary,
    merge_env_semantics_summary,
    new_env_semantics_accumulator,
    resolve_reference_semantics_enabled,
    resolve_state_full_rms_enabled,
    resolve_state_input_scale_enabled,
    resolve_strict_joint_transition_enabled,
    resolve_terminal_reset_enabled,
    summarize_env_semantics,
    transition_reference_mode,
)
from ticl.priors.environment_prior import EnvironmentPrior
from ticl.rlpfn_maintained_path import resolve_rlpfn_token_layout, validate_rlpfn_maintained_path_config
from ticl.train import _build_policy_step_fn


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _legacy_coerce_bool(value):
    if torch.is_tensor(value):
        if value.dtype == torch.bool:
            return value
        if value.numel() == 1:
            return bool(value.item())
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off", ""}
    return bool(value)


def _legacy_resolve_strict_joint_transition_enabled(h):
    enabled = h.get("strict_joint_transition_enabled", False)
    return bool(_legacy_coerce_bool(enabled))


def _legacy_resolve_reference_semantics_enabled(h):
    return _legacy_resolve_strict_joint_transition_enabled(h)


def _legacy_env_uses_reference_semantics(env):
    enabled = env.get(
        "reference_semantics_enabled",
        env.get("strict_joint_transition_enabled", False),
    )
    return bool(_legacy_coerce_bool(enabled))


def _legacy_transition_reference_mode(family, reference_semantics_enabled, gp_forward_mode=None):
    family_str = str(family)
    if family_str == "gp" and bool(reference_semantics_enabled):
        mode = str(gp_forward_mode or "exact").strip().lower()
        if mode == "fixed_cost":
            return "gp_fixed_cost"
        return "gp_exact"
    return f"{family_str}_{'exact' if bool(reference_semantics_enabled) else 'legacy'}"


def _legacy_env_obs_input_dim(obs_dim, *, reference_semantics_enabled=False):
    return 0 if bool(reference_semantics_enabled) else int(max(0, int(obs_dim)))


def _legacy_env_input_layout(
    state_dim,
    obs_dim,
    action_dim,
    noise_dim,
    zero_pad_dim,
    *,
    reference_semantics_enabled=False,
):
    state_dim = int(max(0, int(state_dim)))
    obs_dim = int(max(0, int(obs_dim)))
    action_dim = int(max(0, int(action_dim)))
    noise_dim = int(max(0, int(noise_dim)))
    zero_pad_dim = int(max(0, int(zero_pad_dim)))
    obs_input_dim = _legacy_env_obs_input_dim(
        obs_dim,
        reference_semantics_enabled=reference_semantics_enabled,
    )
    action_start = state_dim + obs_input_dim
    noise_start = action_start + action_dim
    zero_start = noise_start + noise_dim
    total_dim = zero_start + zero_pad_dim
    return {
        "total_dim": int(total_dim),
        "include_obs": bool(obs_input_dim > 0),
        "obs_input_dim": int(obs_input_dim),
        "obs_start": int(state_dim) if obs_input_dim > 0 else None,
        "action_start": int(action_start),
        "noise_start": int(noise_start),
        "zero_start": int(zero_start),
    }


def _legacy_resolve_state_input_scale_enabled(h):
    return bool(_legacy_coerce_bool(h.get("state_input_scale_enabled", False)))


def _legacy_resolve_state_full_rms_enabled(h):
    return bool(_legacy_coerce_bool(h.get("state_full_rms_enabled", False)))


def _legacy_resolve_terminal_reset_enabled(h):
    return bool(_legacy_coerce_bool(h.get("terminal_reset_enabled", False)))


def _legacy_expand_env_value_to_list(value, batch_size):
    batch_size = int(max(1, int(batch_size)))
    if torch.is_tensor(value):
        if value.ndim == 0:
            return [value.item()] * batch_size
        items = value.detach().cpu().reshape(-1).tolist()
    elif isinstance(value, np.ndarray):
        items = value.reshape(-1).tolist()
    elif isinstance(value, (list, tuple)):
        items = list(value)
    else:
        return [value] * batch_size
    if not items:
        return [None] * batch_size
    if len(items) == batch_size:
        return items
    if len(items) == 1:
        return items * batch_size
    if len(items) < batch_size:
        return items + [items[-1]] * (batch_size - len(items))
    return items[:batch_size]


def _legacy_new_env_semantics_accumulator():
    return {
        "env_count": 0,
        "strict_joint_transition_count": 0,
        "reference_semantics_count": 0,
        "exact_scm_count": 0,
        "exact_gp_count": 0,
        "fixed_gp_count": 0,
        "legacy_scm_count": 0,
        "legacy_gp_count": 0,
    }


def _legacy_finalize_env_semantics_summary(acc):
    if not isinstance(acc, dict):
        return None
    total = int(acc.get("env_count", 0) or 0)
    summary = dict(acc)
    summary["strict_joint_transition_share"] = float(
        float(summary.get("strict_joint_transition_count", 0) or 0) / float(max(1, total))
    )
    summary["reference_semantics_share"] = float(
        float(summary.get("reference_semantics_count", 0) or 0) / float(max(1, total))
    )
    modes = []
    if int(summary.get("exact_scm_count", 0) or 0) > 0:
        modes.append("scm_exact")
    if int(summary.get("exact_gp_count", 0) or 0) > 0:
        modes.append("gp_exact")
    if int(summary.get("fixed_gp_count", 0) or 0) > 0:
        modes.append("gp_fixed_cost")
    if int(summary.get("legacy_scm_count", 0) or 0) > 0:
        modes.append("scm_legacy")
    if int(summary.get("legacy_gp_count", 0) or 0) > 0:
        modes.append("gp_legacy")
    if len(modes) == 1:
        summary["transition_reference_mode"] = modes[0]
    elif len(modes) == 0:
        summary["transition_reference_mode"] = "unknown"
    else:
        summary["transition_reference_mode"] = "mixed"
    return summary


def _legacy_merge_env_semantics_summary(acc, summary):
    if not isinstance(summary, dict):
        return acc
    if acc is None:
        acc = _legacy_new_env_semantics_accumulator()
    for key in (
        "env_count",
        "strict_joint_transition_count",
        "reference_semantics_count",
        "exact_scm_count",
        "exact_gp_count",
        "fixed_gp_count",
        "legacy_scm_count",
        "legacy_gp_count",
    ):
        acc[key] = int(acc.get(key, 0) or 0) + int(summary.get(key, 0) or 0)
    return acc


def _legacy_summarize_env_semantics(env, batch_size):
    batch_size = int(max(1, int(batch_size)))
    families = _legacy_expand_env_value_to_list(env.get("family", "unknown"), batch_size)
    gp_modes = _legacy_expand_env_value_to_list(env.get("reference_gp_forward_mode", None), batch_size)
    references = [
        bool(v)
        for v in _legacy_expand_env_value_to_list(
            env.get("reference_semantics_enabled", env.get("strict_joint_transition_enabled", False)),
            batch_size,
        )
    ]
    stricts = [
        bool(v)
        for v in _legacy_expand_env_value_to_list(env.get("strict_joint_transition_enabled", False), batch_size)
    ]
    exact_scm_count = 0
    exact_gp_count = 0
    fixed_gp_count = 0
    legacy_scm_count = 0
    legacy_gp_count = 0
    for family, reference_enabled, gp_mode in zip(families, references, gp_modes):
        family_str = str(family)
        if family_str == "scm":
            if reference_enabled:
                exact_scm_count += 1
            else:
                legacy_scm_count += 1
        elif family_str == "gp":
            if reference_enabled:
                if str(gp_mode).strip().lower() == "fixed_cost":
                    fixed_gp_count += 1
                else:
                    exact_gp_count += 1
            else:
                legacy_gp_count += 1
    summary = {
        "env_count": int(batch_size),
        "strict_joint_transition_count": int(sum(1 for flag in stricts if flag)),
        "reference_semantics_count": int(sum(1 for flag in references if flag)),
        "exact_scm_count": int(exact_scm_count),
        "exact_gp_count": int(exact_gp_count),
        "fixed_gp_count": int(fixed_gp_count),
        "legacy_scm_count": int(legacy_scm_count),
        "legacy_gp_count": int(legacy_gp_count),
    }
    return _legacy_finalize_env_semantics_summary(summary)


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


def test_rlpfn_maintained_exact_scm_helper_contract():
    h = {
        "strict_joint_transition_enabled": "true",
    }
    env = {
        "strict_joint_transition_enabled": False,
        "reference_semantics_enabled": torch.tensor(True),
    }

    assert resolve_strict_joint_transition_enabled(h) is True
    assert resolve_reference_semantics_enabled(h) is True
    assert env_uses_reference_semantics(env) is True
    assert transition_reference_mode("scm", True) == "scm_exact"
    assert transition_reference_mode("scm", False) == "scm_legacy"
    assert transition_reference_mode("gp", True, gp_forward_mode="exact") == "gp_exact"
    assert transition_reference_mode("gp", True, gp_forward_mode="fixed_cost") == "gp_fixed_cost"


def test_rlpfn_maintained_exact_scm_helper_matches_legacy_inline_reference():
    bool_cases = [
        False,
        True,
        "true",
        "false",
        "off",
        "",
        0,
        1,
        torch.tensor(False),
        torch.tensor(True),
        torch.tensor([0, 1, 0], dtype=torch.int64),
        torch.tensor([True, False, True], dtype=torch.bool),
    ]
    for value in bool_cases:
        actual = coerce_bool(value)
        expected = _legacy_coerce_bool(value)
        if torch.is_tensor(expected):
            assert torch.equal(actual, expected)
        else:
            assert actual == expected

    h_cases = [
        {"strict_joint_transition_enabled": False},
        {"strict_joint_transition_enabled": "true"},
        {"strict_joint_transition_enabled": torch.tensor(True)},
        {"strict_joint_transition_enabled": 0},
    ]
    for h in h_cases:
        assert resolve_strict_joint_transition_enabled(h) == _legacy_resolve_strict_joint_transition_enabled(h)
        assert resolve_reference_semantics_enabled(h) == _legacy_resolve_reference_semantics_enabled(h)

    env_cases = [
        {"strict_joint_transition_enabled": False, "reference_semantics_enabled": torch.tensor(True)},
        {"strict_joint_transition_enabled": "true"},
        {"strict_joint_transition_enabled": False, "reference_semantics_enabled": False},
    ]
    for env in env_cases:
        assert env_uses_reference_semantics(env) == _legacy_env_uses_reference_semantics(env)

    mode_cases = [
        ("scm", True, None),
        ("scm", False, None),
        ("gp", True, "exact"),
        ("gp", True, "fixed_cost"),
        ("gp", False, "exact"),
    ]
    for family, enabled, gp_mode in mode_cases:
        assert transition_reference_mode(
            family,
            enabled,
            gp_forward_mode=gp_mode,
        ) == _legacy_transition_reference_mode(family, enabled, gp_forward_mode=gp_mode)


def test_rlpfn_maintained_exact_scm_layout_and_flag_helpers_match_legacy_inline_reference():
    layout_cases = [
        (8, 6, 3, 2, 0, False),
        (8, 6, 3, 2, 0, True),
        (0, 4, 1, 0, 2, False),
        (5, 12, 7, 4, 3, True),
    ]
    for state_dim, obs_dim, action_dim, noise_dim, zero_pad_dim, reference_semantics_enabled in layout_cases:
        assert env_obs_input_dim(
            obs_dim,
            reference_semantics_enabled=reference_semantics_enabled,
        ) == _legacy_env_obs_input_dim(
            obs_dim,
            reference_semantics_enabled=reference_semantics_enabled,
        )
        assert env_input_layout(
            state_dim,
            obs_dim,
            action_dim,
            noise_dim,
            zero_pad_dim,
            reference_semantics_enabled=reference_semantics_enabled,
        ) == _legacy_env_input_layout(
            state_dim,
            obs_dim,
            action_dim,
            noise_dim,
            zero_pad_dim,
            reference_semantics_enabled=reference_semantics_enabled,
        )
        assert EnvironmentPrior._env_input_layout(
            state_dim,
            obs_dim,
            action_dim,
            noise_dim,
            zero_pad_dim,
            reference_semantics_enabled=reference_semantics_enabled,
        ) == _legacy_env_input_layout(
            state_dim,
            obs_dim,
            action_dim,
            noise_dim,
            zero_pad_dim,
            reference_semantics_enabled=reference_semantics_enabled,
        )

    flag_cases = [
        {
            "state_input_scale_enabled": False,
            "state_full_rms_enabled": False,
            "terminal_reset_enabled": False,
        },
        {
            "state_input_scale_enabled": "true",
            "state_full_rms_enabled": "false",
            "terminal_reset_enabled": "1",
        },
        {
            "state_input_scale_enabled": torch.tensor(True),
            "state_full_rms_enabled": torch.tensor(False),
            "terminal_reset_enabled": torch.tensor(True),
        },
    ]
    for h in flag_cases:
        assert resolve_state_input_scale_enabled(h) == _legacy_resolve_state_input_scale_enabled(h)
        assert resolve_state_full_rms_enabled(h) == _legacy_resolve_state_full_rms_enabled(h)
        assert resolve_terminal_reset_enabled(h) == _legacy_resolve_terminal_reset_enabled(h)
        assert EnvironmentPrior._resolve_state_input_scale_enabled(h) == _legacy_resolve_state_input_scale_enabled(h)
        assert EnvironmentPrior._resolve_state_full_rms_enabled(h) == _legacy_resolve_state_full_rms_enabled(h)
        assert EnvironmentPrior._resolve_terminal_reset_enabled(h) == _legacy_resolve_terminal_reset_enabled(h)


def test_rlpfn_maintained_exact_scm_env_semantics_helpers_match_legacy_inline_reference():
    value_cases = [
        ("scm", 3),
        (torch.tensor(True), 2),
        (torch.tensor([1, 0, 1]), 3),
        (np.array(["exact", "fixed_cost"], dtype=object), 3),
        ([], 2),
        ([1], 4),
        ([1, 2], 4),
        ([1, 2, 3, 4], 2),
    ]
    for value, batch_size in value_cases:
        assert expand_env_value_to_list(value, batch_size) == _legacy_expand_env_value_to_list(value, batch_size)

    env = {
        "family": ["scm", "gp", "gp", "scm"],
        "reference_gp_forward_mode": [None, "exact", "fixed_cost", None],
        "reference_semantics_enabled": [True, True, True, False],
        "strict_joint_transition_enabled": [True, True, False, False],
    }
    expected_summary = _legacy_summarize_env_semantics(env, 4)
    assert summarize_env_semantics(env, 4) == expected_summary
    assert EnvironmentPrior._summarize_env_semantics(env, 4) == expected_summary

    acc_expected = _legacy_new_env_semantics_accumulator()
    acc_actual = new_env_semantics_accumulator()
    assert acc_actual == acc_expected
    assert EnvironmentPrior._new_env_semantics_accumulator() == acc_expected

    part_a = _legacy_summarize_env_semantics(
        {"family": ["scm", "gp"], "reference_semantics_enabled": [True, False], "strict_joint_transition_enabled": [True, False]},
        2,
    )
    part_b = _legacy_summarize_env_semantics(
        {"family": ["gp"], "reference_gp_forward_mode": ["fixed_cost"], "reference_semantics_enabled": [True], "strict_joint_transition_enabled": [True]},
        1,
    )
    merged_expected = _legacy_merge_env_semantics_summary(dict(acc_expected), part_a)
    merged_expected = _legacy_merge_env_semantics_summary(merged_expected, part_b)
    merged_actual = merge_env_semantics_summary(dict(acc_actual), part_a)
    merged_actual = merge_env_semantics_summary(merged_actual, part_b)
    assert merged_actual == merged_expected
    assert EnvironmentPrior._merge_env_semantics_summary(dict(acc_actual), part_a) == _legacy_merge_env_semantics_summary(
        dict(acc_expected),
        part_a,
    )
    assert finalize_env_semantics_summary(merged_actual) == _legacy_finalize_env_semantics_summary(merged_expected)
    assert EnvironmentPrior._finalize_env_semantics_summary(merged_actual) == _legacy_finalize_env_semantics_summary(
        merged_expected,
    )


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
