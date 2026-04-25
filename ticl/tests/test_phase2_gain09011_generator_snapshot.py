import copy
import hashlib
import json
import random
from contextlib import contextmanager
from typing import Any

import numpy as np
import pytest
import torch

from ticl.model_configs import get_model_default_config
from ticl.priors.environment_prior import EnvironmentPrior
from ticl.rlpfn_maintained_path import validate_rlpfn_maintained_path_config
from ticl.train import _freeze_env_h_list_for_replay


BUILD_SEED = 26024
FROZEN_H_SEED = 26026
TRAIN_ENV_SEED = 26024
FROZEN_H_OVERRIDES = {
    "reward_state_input_gain_fraction_rejection_min": 0.58,
    "reward_state_input_gain_fraction_rejection_max_tries": 128,
}

EXPECTED_LEGACY_PROJECTED_FINGERPRINT = (
    "5206d9b9db9e9ef0a6599b1a9fe85ab11c44bc471b1d23a14364fe62aa878f55"
)
EXPECTED_LEGACY_PROJECTED_KEY_COUNT = 99
CURRENT_ONLY_INERT_KEYS = {
    "fixed_frozen_h_json",
    "reward_state_input_gain_fraction_conditioned_sampling_enabled",
    "reward_state_input_gain_fraction_conditioned_min",
}

EXPECTED_SAMPLED_ENV_SNAPSHOT = {
    "family": "scm",
    "state_dim": 364,
    "obs_dim": 260,
    "action_dim": 17,
    "noise_dim": 23,
    "zero_pad_dim": 13,
    "prior_mlp_activations": "sin",
    "prior_mlp_hidden_dim": 139,
    "num_layers": 3,
    "scm_standard_linear_init_enabled": True,
    "init_std": 0.6615218157850955,
    "noise_std": 0.017524186325965533,
    "reference_state_inertia_enabled": False,
    "alpha": 1.0,
    "state_highway_enabled": False,
    "state_highway_lambda": 0.0,
    "state_noise_std": 0.0,
    "reward_scale": 1.0,
    "ctrl_reward_enabled": True,
    "survival_reward_enabled": False,
    "terminal_reset_enabled": True,
    "terminal_reset_count_target": 10.60502950392072,
    "reward_state_input_gain_fraction": 0.901058195877859,
    "reward_action_input_gain_fraction": 0.04187411811399192,
    "reward_noise_input_gain_fraction": 0.05706768600814905,
    "reward_state_input_gain_fraction_rejection_min": 0.58,
    "reward_state_input_gain_fraction_rejection_attempt": 1,
    "reward_state_input_gain_fraction_rejection_max_tries": 128,
    "reward_state_input_gain_fraction_rejection_accepted": True,
}


def _seed_all(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


@contextmanager
def _temporary_seed(seed: int):
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        _seed_all(seed)
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_state)


def _to_builtin(value: Any):
    if torch.is_tensor(value):
        if value.ndim == 0:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _summarize_frozen_h(frozen_h: dict[str, Any]) -> dict[str, Any]:
    frozen_h_builtin = _to_builtin(frozen_h)
    payload = json.dumps(frozen_h_builtin, sort_keys=True, ensure_ascii=True)
    return {
        "frozen_h_fingerprint": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "top_level_key_count": len(frozen_h_builtin),
    }


def _build_audit_env_cfg(base_env_cfg: dict[str, Any]) -> dict[str, Any]:
    env_cfg = copy.deepcopy(base_env_cfg)
    env_cfg["batch_parallel_backend"] = "python_thread"
    env_cfg["batch_shared_environment"] = False
    env_cfg["reward_dropout_enabled"] = False
    env_cfg["reward_dropout_randomize"] = False
    env_cfg["reward_dropout_ratio"] = 0.0
    env_cfg["reinforce_reward_transform"] = "none"
    env_cfg["action_noise_train_std"] = 0.0
    env_cfg["action_noise_eval_std"] = 0.0
    env_cfg["normalized_q_value_weight"] = 0.0
    env_cfg["next_state_flow_matching_weight"] = 0.0
    return env_cfg


def _build_gain09011_env_cfg_and_frozen_h():
    _seed_all(BUILD_SEED)
    config = copy.deepcopy(get_model_default_config("rlpfn"))
    validate_rlpfn_maintained_path_config(config)
    env_cfg = _build_audit_env_cfg(config["prior"]["environment"])
    prior = EnvironmentPrior(copy.deepcopy(env_cfg))
    setattr(prior, "_rwkv_reset_env_state_at_sep_keep_actor_history", True)
    with _temporary_seed(FROZEN_H_SEED):
        frozen_h = _freeze_env_h_list_for_replay(
            prior,
            list(prior._sample_batch_hypers(1)),
        )[0]
    frozen_h = copy.deepcopy(frozen_h)
    frozen_h.update(copy.deepcopy(FROZEN_H_OVERRIDES))
    return env_cfg, frozen_h


def _sampled_env_snapshot(env: dict[str, Any], frozen_h: dict[str, Any]) -> dict[str, Any]:
    def _get(name: str):
        value = env.get(name)
        if value is None:
            value = frozen_h.get(name)
        return value

    return {
        "family": env.get("family"),
        "state_dim": int(env.get("state_dim")),
        "obs_dim": int(env.get("obs_dim")),
        "action_dim": int(env.get("action_dim")),
        "noise_dim": int(env.get("noise_dim")),
        "zero_pad_dim": int(env.get("zero_pad_dim")),
        "prior_mlp_activations": env.get("prior_mlp_activations", frozen_h.get("prior_mlp_activations")),
        "prior_mlp_hidden_dim": int(env.get("prior_mlp_hidden_dim", frozen_h.get("prior_mlp_hidden_dim"))),
        "num_layers": int(env.get("num_layers", frozen_h.get("num_layers"))),
        "scm_standard_linear_init_enabled": bool(_get("scm_standard_linear_init_enabled")),
        "init_std": float(_get("init_std")),
        "noise_std": float(_get("noise_std")),
        "reference_state_inertia_enabled": bool(env.get("reference_state_inertia_enabled")),
        "alpha": float(env.get("alpha")),
        "state_highway_enabled": bool(env.get("state_highway_enabled")),
        "state_highway_lambda": float(env.get("state_highway_lambda")),
        "state_noise_std": float(env.get("state_noise_std")),
        "reward_scale": float(env.get("reward_scale")),
        "ctrl_reward_enabled": bool(env.get("ctrl_reward_enabled")),
        "survival_reward_enabled": bool(env.get("survival_reward_enabled")),
        "terminal_reset_enabled": bool(env.get("terminal_reset_enabled")),
        "terminal_reset_count_target": float(_get("terminal_reset_count_target")),
        "reward_state_input_gain_fraction": float(env.get("reward_state_input_gain_fraction")),
        "reward_action_input_gain_fraction": float(env.get("reward_action_input_gain_fraction")),
        "reward_noise_input_gain_fraction": float(env.get("reward_noise_input_gain_fraction")),
        "reward_state_input_gain_fraction_rejection_min": float(
            env.get("reward_state_input_gain_fraction_rejection_min")
        ),
        "reward_state_input_gain_fraction_rejection_attempt": int(
            env.get("reward_state_input_gain_fraction_rejection_attempt")
        ),
        "reward_state_input_gain_fraction_rejection_max_tries": int(
            env.get("reward_state_input_gain_fraction_rejection_max_tries")
        ),
        "reward_state_input_gain_fraction_rejection_accepted": bool(
            env.get("reward_state_input_gain_fraction_rejection_accepted")
        ),
    }


def test_gain09011_generator_legacy_frozen_h_snapshot_is_stable():
    _, frozen_h = _build_gain09011_env_cfg_and_frozen_h()

    present_current_only_keys = set(frozen_h) & CURRENT_ONLY_INERT_KEYS
    projected_h = {k: v for k, v in frozen_h.items() if k not in present_current_only_keys}
    summary = _summarize_frozen_h(projected_h)
    assert summary["frozen_h_fingerprint"] == EXPECTED_LEGACY_PROJECTED_FINGERPRINT
    assert summary["top_level_key_count"] == EXPECTED_LEGACY_PROJECTED_KEY_COUNT


@pytest.mark.skipif(not torch.cuda.is_available(), reason="gain09011 sampled-env snapshot is CUDA-path specific")
def test_gain09011_generator_cuda_sampled_env_snapshot_is_stable():
    env_cfg, frozen_h = _build_gain09011_env_cfg_and_frozen_h()
    device = torch.device("cuda:0")
    env = EnvironmentPrior(copy.deepcopy(env_cfg))._sample_environment(
        copy.deepcopy(frozen_h),
        device=device,
        rng_seed=TRAIN_ENV_SEED,
    )
    snapshot = _sampled_env_snapshot(env, frozen_h)

    assert set(snapshot) == set(EXPECTED_SAMPLED_ENV_SNAPSHOT)
    for key, expected_value in EXPECTED_SAMPLED_ENV_SNAPSHOT.items():
        actual_value = snapshot[key]
        if isinstance(expected_value, float):
            assert actual_value == pytest.approx(expected_value, rel=0.0, abs=1e-15), key
        else:
            assert actual_value == expected_value, key
