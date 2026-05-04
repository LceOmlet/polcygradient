import importlib
import sys
import types

import numpy as np
import pytest
import torch


def _install_fake_tracking_modules(monkeypatch):
    mlflow_mod = types.ModuleType("mlflow")
    mlflow_mod.log_metric = lambda *args, **kwargs: None
    mlflow_mod.log_params = lambda *args, **kwargs: None
    mlflow_mod.set_tracking_uri = lambda *args, **kwargs: None
    mlflow_mod.search_runs = lambda *args, **kwargs: {"run_id": []}

    class _Run:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    mlflow_mod.start_run = lambda *args, **kwargs: _Run()
    wandb_mod = types.ModuleType("wandb")
    wandb_mod.run = None
    wandb_mod.init = lambda *args, **kwargs: None
    wandb_mod.log = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "mlflow", mlflow_mod)
    monkeypatch.setitem(sys.modules, "wandb", wandb_mod)


class _FakeSpace:
    pass


class _FakeBox(_FakeSpace):
    def __init__(self, low, high, shape=None, dtype=np.float32):
        del shape
        self.low = np.asarray(low, dtype=dtype)
        self.high = np.asarray(high, dtype=dtype)
        self.shape = self.low.shape
        self.dtype = dtype


class _FakeDiscrete(_FakeSpace):
    def __init__(self, n):
        self.n = int(n)


class _FakeMultiDiscrete(_FakeSpace):
    def __init__(self, nvec):
        self.nvec = np.asarray(nvec)


class _FakeMultiBinary(_FakeSpace):
    def __init__(self, n):
        self.n = int(n)


class _FakeDict(_FakeSpace):
    def __init__(self, spaces):
        self.spaces = dict(spaces)


class _FakeTuple(_FakeSpace):
    def __init__(self, spaces):
        self.spaces = tuple(spaces)


class _FakeWrapper:
    def __init__(self, env=None):
        self.env = env


class _FastContinuousEnv:
    def __init__(self, name, obs_dim=4, action_dim=1, horizon=2):
        self.name = name
        self._obs_dim = int(obs_dim)
        self._horizon = int(horizon)
        self._step_idx = 0
        self.action_space = _FakeBox(
            low=-np.ones((int(action_dim),), dtype=np.float32),
            high=np.ones((int(action_dim),), dtype=np.float32),
        )

    def reset(self, seed=None):
        del seed
        self._step_idx = 0
        return np.zeros((self._obs_dim,), dtype=np.float32), {}

    def step(self, action):
        del action
        self._step_idx += 1
        terminated = bool(self._step_idx >= self._horizon)
        obs = np.full((self._obs_dim,), float(self._step_idx), dtype=np.float32)
        reward = 1.0
        return obs, reward, terminated, False, {}

    def close(self):
        return None


def _install_fake_gym(monkeypatch):
    import gymnasium as gym_mod

    monkeypatch.setattr(gym_mod, "make", lambda env_name: _FastContinuousEnv(env_name))


def _tiny_rlpfn_extra_config():
    return {
        "transformer": {
            "emsize": 32,
            "nlayers": 1,
            "nhid_factor": 1,
            "nhead": 1,
            "recompute_attn": False,
            "x_obs_dim": 9,
            "x_action_dim": 1,
        },
        "prior": {
            "n_samples": 16,
            "eval_positions": [8],
            "num_features": 10,
            "environment": {
                "action_dim": 1,
                "state_dim": 4,
                "obs_dim": 4,
                "noise_dim": 2,
                "zero_pad_dim": 0,
                "obs_slot_dim": 5,
                "action_slot_dim": 1,
                "reward_dropout_enabled": False,
                "batch_parallel_workers": 1,
                "batch_parallel_backend": "torch_vectorized",
                "terminal_reset_enabled": True,
            },
        },
        "dataloader": {
            "batch_size": 2,
            "num_steps": 1,
        },
        "optimizer": {
            "epochs": 1,
            "stop_after_epochs": 1,
            "adaptive_batch_size": False,
            "learning_rate": 1e-4,
            "policy_rollout_chunk_size": 1,
            "policy_rollout_checkpoint": False,
            "policy_rollout_checkpoint_reentrant": False,
            "pg_grad_mutable_kv_cache": False,
            "pg_saved_tensors_cpu_offload": False,
            "pg_saved_tensors_pin_memory": False,
            "pg_tbptt_window": 4,
            "pg_env_replay_steps": 1,
            "pg_kv_cache_mode": "auto",
            "pg_kv_cache_page_size": None,
            "train_host_rss_limit_gib": None,
        },
    }


def test_short_rlpfn_training_runs_new_validation_path_across_all_default_envs(monkeypatch, tmp_path):
    import time

    if not torch.cuda.is_available():
        pytest.skip("CUDA required for strict official RLPFN validation smoke.")

    _install_fake_tracking_modules(monkeypatch)
    _install_fake_gym(monkeypatch)
    monkeypatch.setenv("TICL_POLICY_ENVGEN_CHECKPOINT", "0")
    monkeypatch.setenv("TICL_POLICY_ENVGEN_CHECKPOINT_REENTRANT", "0")

    rl_validation = importlib.import_module("ticl.rl_validation")
    utils_mod = importlib.import_module("ticl.utils")
    fit_model_mod = importlib.import_module("ticl.fit_model")

    validation_stats = {"calls": 0}
    original_validate = utils_mod.evaluate_rlpfn_on_gym_envs

    def _timed_validate(*args, **kwargs):
        import time

        validation_stats["calls"] += 1
        t0 = time.perf_counter()
        mean_ret, per_env = original_validate(*args, **kwargs)
        t1 = time.perf_counter()
        validation_stats["validation_sec"] = t1 - t0
        validation_stats["mean_return"] = float(mean_ret)
        validation_stats["per_env"] = per_env
        return mean_ret, per_env

    def _unexpected_old_path(*args, **kwargs):
        raise AssertionError("rl validation unexpectedly fell back to candidate-action scoring")

    monkeypatch.setattr(utils_mod, "evaluate_rlpfn_on_gym_envs", _timed_validate)
    monkeypatch.setattr(rl_validation, "_score_candidate_action_jobs", _unexpected_old_path)

    argv = [
        "rlpfn",
        "--seed-everything",
        "False",
        "--validate",
        "True",
        "--save-every",
        "1",
        "--extra-fast-test",
        "--gpu-id",
        "0",
        "--train-mixed-precision",
        "False",
        "--rl-validate-episodes",
        "1",
        "--rl-validate-max-steps",
        "2",
        "--rl-validate-context-lower-bound",
        "1",
        "-B",
        str(tmp_path),
    ]

    t0 = time.perf_counter()
    results = fit_model_mod.main(argv, extra_config=_tiny_rlpfn_extra_config())
    total_sec = time.perf_counter() - t0

    assert results["config"]["model_type"] == "rlpfn"
    assert int(results["epoch"]) == 1
    assert validation_stats["calls"] == 1
    assert validation_stats["validation_sec"] > 0.0
    assert validation_stats["mean_return"] == pytest.approx(2.0)
    assert set(validation_stats["per_env"]) == set(rl_validation.RLPFN_DEFAULT_OOP_ENVS)
    for env_name in rl_validation.RLPFN_DEFAULT_OOP_ENVS:
        env_metrics = validation_stats["per_env"][env_name]
        assert env_metrics["make_failed"] == 0
        assert env_metrics["return"] == pytest.approx(2.0)
        assert env_metrics["return_mean"] == pytest.approx(2.0)

    print(
        "rlpfn_short_train_validation_smoke "
        f"total_sec={total_sec:.4f} "
        f"validation_sec={validation_stats['validation_sec']:.4f} "
        f"validated_envs={len(validation_stats['per_env'])}"
    )
