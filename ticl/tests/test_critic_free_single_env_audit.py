from types import SimpleNamespace

import numpy as np
import torch

from ticl.analysis import critic_free_single_env_audit as audit_mod
from ticl.analysis.critic_free_single_env_audit import (
    _build_audit_ppo_policy_step_fn,
    _build_frozen_h_from_mode,
    _masked_corrcoef_np,
    _measure_rollout_critic_quality,
    _run_extra_critic_updates_with_scope,
)
from ticl.sb3_recurrent_ppo import _explained_variance_with_mask


class _DummyPolicy:
    def __init__(self):
        self.calls = []

    def make_vectorized_rollout_step_fn(self):
        def _step_fn(obs_t, action_t, reward_t, reward_mask_t, cache, step_idx, env_info):
            self.calls.append(
                {
                    "obs_t": obs_t.clone(),
                    "action_t": action_t.clone(),
                    "reward_t": reward_t.clone(),
                    "reward_mask_t": reward_mask_t.clone(),
                    "cache": cache,
                    "step_idx": int(step_idx),
                    "terminal_t": env_info.get("terminal_t"),
                }
            )
            return {"ok": True}

        return _step_fn


def test_build_audit_ppo_policy_step_fn_clear_first_eval_reward_terminal_history_only():
    policy = _DummyPolicy()
    step_fn = _build_audit_ppo_policy_step_fn(
        policy,
        sampled=True,
        single_eval_pos=4,
        boundary_contract_mode="clear_first_eval_reward_terminal_history",
    )

    obs_t = torch.ones(2, 3)
    action_t = torch.full((2, 2), 7.0)
    reward_t = torch.full((2,), 5.0)
    reward_mask_t = torch.ones(2)
    terminal_t = torch.ones(2)

    step_fn(
        obs_t,
        action_t,
        reward_t,
        reward_mask_t,
        cache={"k": 1},
        step_idx=4,
        env_info={"action_dim": 2, "terminal_t": terminal_t},
    )

    call = policy.calls[-1]
    assert torch.allclose(call["action_t"], action_t)
    assert torch.allclose(call["reward_t"], torch.zeros_like(reward_t))
    assert torch.allclose(call["reward_mask_t"], torch.zeros_like(reward_mask_t))
    assert torch.allclose(call["terminal_t"], torch.zeros_like(terminal_t))
    assert call["cache"] == {"k": 1}


def test_build_audit_ppo_policy_step_fn_state_reset_plus_hidden_keeps_step_history():
    policy = _DummyPolicy()
    step_fn = _build_audit_ppo_policy_step_fn(
        policy,
        sampled=True,
        single_eval_pos=4,
        boundary_contract_mode="reset_env_state_at_sep_and_reset_hidden_keep_actor_history",
    )

    obs_t = torch.ones(2, 3)
    action_t = torch.full((2, 2), 7.0)
    reward_t = torch.full((2,), 5.0)
    reward_mask_t = torch.ones(2)
    terminal_t = torch.ones(2)

    step_fn(
        obs_t,
        action_t,
        reward_t,
        reward_mask_t,
        cache={"k": 1},
        step_idx=4,
        env_info={"action_dim": 2, "terminal_t": terminal_t},
    )

    call = policy.calls[-1]
    assert torch.allclose(call["action_t"], action_t)
    assert torch.allclose(call["reward_t"], reward_t)
    assert torch.allclose(call["reward_mask_t"], reward_mask_t)
    assert torch.allclose(call["terminal_t"], terminal_t)
    assert call["cache"] is None


def test_masked_corrcoef_np_uses_only_masked_entries():
    x = [1.0, 2.0, 100.0]
    y = [2.0, 4.0, -999.0]
    mask = [True, True, False]
    assert _masked_corrcoef_np(x, y, mask) > 0.999


def test_build_frozen_h_from_mode_routes_current_and_legacy_numpy(monkeypatch):
    calls = []

    def _fake_build_fixed_env_h(*, prior_cfg, frozen_h_seed, seed_mode="current"):
        calls.append(
            {
                "prior_cfg": prior_cfg,
                "frozen_h_seed": int(frozen_h_seed),
                "seed_mode": str(seed_mode),
            }
        )
        return {"seed_mode": str(seed_mode)}

    monkeypatch.setattr(audit_mod, "build_fixed_env_h", _fake_build_fixed_env_h)

    current = _build_frozen_h_from_mode(
        prior_cfg={"dummy": True},
        frozen_h_seed=12345,
        seed_mode="current",
    )
    legacy = _build_frozen_h_from_mode(
        prior_cfg={"dummy": True},
        frozen_h_seed=23456,
        seed_mode="legacy_numpy_only",
    )

    assert current == {"seed_mode": "current"}
    assert legacy == {"seed_mode": "legacy_numpy_only"}
    assert calls == [
        {"prior_cfg": {"dummy": True}, "frozen_h_seed": 12345, "seed_mode": "current"},
        {"prior_cfg": {"dummy": True}, "frozen_h_seed": 23456, "seed_mode": "legacy_numpy_only"},
    ]


def test_build_frozen_h_from_mode_routes_legacy_same_prior_numpy_only(monkeypatch):
    calls = []
    prior = object()

    def _fake_sample_fixed_env_h_from_prior(*, prior, frozen_h_seed, seed_mode="current"):
        calls.append(
            {
                "prior": prior,
                "frozen_h_seed": int(frozen_h_seed),
                "seed_mode": str(seed_mode),
            }
        )
        return {"prior_id": id(prior), "seed_mode": str(seed_mode)}

    monkeypatch.setattr(audit_mod, "sample_fixed_env_h_from_prior", _fake_sample_fixed_env_h_from_prior)

    out = _build_frozen_h_from_mode(
        prior_cfg={"dummy": True},
        frozen_h_seed=34567,
        seed_mode="legacy_same_prior_numpy_only",
        prior=prior,
    )

    assert out == {"prior_id": id(prior), "seed_mode": "legacy_numpy_only"}
    assert calls == [
        {
            "prior": prior,
            "frozen_h_seed": 34567,
            "seed_mode": "legacy_numpy_only",
        }
    ]


def test_seed_all_makes_torch_and_numpy_draws_repeatable():
    audit_mod._seed_all(4040)
    first_torch = torch.randn(4)
    first_numpy = audit_mod.np.random.randn(4)

    audit_mod._seed_all(4040)
    second_torch = torch.randn(4)
    second_numpy = audit_mod.np.random.randn(4)

    assert torch.allclose(first_torch, second_torch)
    assert audit_mod.np.allclose(first_numpy, second_numpy)


def test_audit_train_loop_records_post_train_rollout_quality_when_enabled(monkeypatch):
    class _DummyRolloutBuffer:
        _actor_baseline_mode = "learned"

    class _DummyPolicy:
        def _dummy_states(self, n_envs):
            return {"n_envs": n_envs}

    class _DummyAlgo:
        def __init__(self):
            self.n_steps = 4
            self.rollout_buffer = _DummyRolloutBuffer()
            self.policy = _DummyPolicy()
            self._last_obs = object()
            self.trained = 0

        def _update_current_progress_remaining(self, *_args, **_kwargs):
            return None

        def collect_rollouts(self, *_args, **_kwargs):
            return True

        def train(self):
            self.trained += 1
            return None

    algo = _DummyAlgo()

    monkeypatch.setattr(
        audit_mod,
        "_measure_rollout_critic_quality",
        lambda _algo: {
            "objective_total": 12,
            "raw_corr": -0.25,
            "explained_variance_normalized": -0.125,
            "explained_variance_raw": -0.5,
            "normalized_value_std": 0.01,
            "normalized_return_std": 0.02,
            "raw_value_std": 0.03,
            "raw_return_std": 0.04,
        },
    )
    monkeypatch.setattr(
        audit_mod,
        "_measure_current_policy_rollout_critic_quality",
        lambda _algo: {
            "objective_total": 12,
            "raw_corr": 0.75,
            "explained_variance_normalized": 0.25,
            "explained_variance_raw": 0.5,
            "normalized_value_std": 0.11,
            "normalized_return_std": 0.22,
            "raw_value_std": 0.33,
            "raw_return_std": 0.44,
        },
    )

    history = audit_mod._audit_train_loop(
        algo,
        callback=object(),
        vec_env=object(),
        outer_epochs=1,
        record_post_train_rollout_critic_quality=True,
    )

    assert len(history) == 1
    row = history[0]
    assert row["critic_raw_corr"] == -0.25
    assert row["critic_explained_variance_normalized"] == -0.125
    assert row["critic_explained_variance_raw"] == -0.5
    assert row["critic_normalized_value_std"] == 0.01
    assert row["critic_normalized_return_std"] == 0.02
    assert row["critic_raw_value_std"] == 0.03
    assert row["critic_raw_return_std"] == 0.04
    assert row["critic_raw_corr_post_train_rollout"] == 0.75
    assert row["critic_explained_variance_normalized_post_train_rollout"] == 0.25
    assert row["critic_explained_variance_raw_post_train_rollout"] == 0.5
    assert row["critic_objective_total_post_train_rollout"] == 12
    assert row["critic_normalized_value_std_post_train_rollout"] == 0.11
    assert row["critic_normalized_return_std_post_train_rollout"] == 0.22
    assert row["critic_raw_value_std_post_train_rollout"] == 0.33
    assert row["critic_raw_return_std_post_train_rollout"] == 0.44


def test_run_extra_critic_updates_with_scope_prefers_bucket_targets_when_available():
    class _RecordingBarDist:
        def __init__(self):
            self.calls = []

        def __call__(self, logits, returns):
            self.calls.append(("returns", returns.detach().clone()))
            return (logits.reshape(-1) - returns.reshape(-1)).square()

        def nll_from_bucket_idx(self, logits, bucket_idx):
            self.calls.append(("bucket", bucket_idx.detach().clone()))
            return (logits.reshape(-1) - bucket_idx.reshape(-1).float()).square()

    class _DummyPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.value_head = torch.nn.Linear(1, 1, bias=False)
            self.optimizer = torch.optim.SGD(self.parameters(), lr=0.1)
            self._bardist = _RecordingBarDist()

        def evaluate_actions_with_hidden_flat(self, observations, actions, *, seq_lengths, action_masks):
            del actions, seq_lengths, action_masks
            return {"value_logits": self.value_head(observations)}

        def get_value_bardist(self):
            return self._bardist

        def compute_value_loss_from_logits(
            self,
            *,
            value_logits,
            targets,
            objective_mask,
            value_target_bucket_idx=None,
        ):
            objective_mask = objective_mask.to(dtype=torch.bool)
            if value_target_bucket_idx is not None:
                losses = self._bardist.nll_from_bucket_idx(value_logits, value_target_bucket_idx)
            else:
                losses = self._bardist(value_logits, targets)
            losses = losses.reshape(-1)
            mask_f = objective_mask.reshape(-1).to(device=losses.device, dtype=losses.dtype)
            return (losses * mask_f).sum() / mask_f.sum().clamp_min(1.0)

    class _DummyRolloutBuffer:
        def __init__(self):
            self._deterministic_batch_plan = False

        def get_gpu_flat(self, _batch_size):
            yield SimpleNamespace(
                observations=torch.tensor([[1.0], [2.0]], dtype=torch.float32),
                actions=torch.zeros((2, 1), dtype=torch.float32),
                seq_lengths=np.array([2], dtype=np.int64),
                action_masks=None,
                objective_masks=torch.ones(2, dtype=torch.float32),
                returns=torch.tensor([1.0, 2.0], dtype=torch.float32),
                value_target_bucket_idx=torch.tensor([3, 4], dtype=torch.int64),
            )

        def set_deterministic_batch_plan(self, enabled=True):
            self._deterministic_batch_plan = bool(enabled)

    algo = SimpleNamespace(
        rollout_buffer=_DummyRolloutBuffer(),
        policy=_DummyPolicy(),
        batch_size=2,
        max_grad_norm=0.5,
    )

    _run_extra_critic_updates_with_scope(algo, extra_updates=1, scope="shared")

    assert len(algo.policy._bardist.calls) == 1
    kind, bucket_idx = algo.policy._bardist.calls[0]
    assert kind == "bucket"
    assert torch.equal(bucket_idx, torch.tensor([3, 4], dtype=torch.int64))


def test_measure_suite_critic_quality_with_policy_state_passes_vf_coef_override(monkeypatch):
    class _DummyPrior:
        def __init__(self, _cfg):
            self.cfg = _cfg

    class _DummyPolicy:
        def load_state_dict(self, _state_dict, strict=True):
            assert strict is True
            return None

        def _dummy_states(self, n_envs):
            return {"n_envs": int(n_envs)}

    class _DummyAlgo:
        def __init__(self):
            self.policy = _DummyPolicy()
            self.rollout_buffer = object()
            self.n_steps = 4

        def _update_current_progress_remaining(self, *_args, **_kwargs):
            return None

        def collect_rollouts(self, *_args, **_kwargs):
            return True

    class _DummyCallback:
        def init_callback(self, _algo):
            return None

    class _DummyVecEnv:
        num_envs = 1

        def reset(self):
            return {"obs": 1}

        def close(self):
            return None

    captured = {}

    def _fake_load_model(_checkpoint_path, device=None, verbose=False):
        del device, verbose
        return object(), {"optimizer": {"ppo_vf_coef": 0.5, "ppo_gamma": 0.98, "ppo_gae_lambda": 0.90, "ppo_ent_coef": 0.0, "ppo_max_grad_norm": 0.5}}

    _fake_load_model.cache_clear = lambda: None  # type: ignore[attr-defined]

    def _fake_build_recurrent_ppo(**kwargs):
        captured["vf_coef"] = float(kwargs["vf_coef"])
        return _DummyAlgo(), _DummyCallback(), _DummyVecEnv()

    monkeypatch.setattr(audit_mod, "load_model", _fake_load_model)
    monkeypatch.setattr(audit_mod, "EnvironmentPrior", _DummyPrior)
    monkeypatch.setattr(audit_mod, "build_recurrent_ppo", _fake_build_recurrent_ppo)
    monkeypatch.setattr(
        audit_mod,
        "_measure_rollout_critic_quality",
        lambda _algo: {
            "objective_total": 7,
            "raw_corr": 0.25,
            "explained_variance_normalized": 0.125,
            "explained_variance_raw": 0.5,
        },
    )
    monkeypatch.setattr(audit_mod, "_apply_boundary_contract_mode_flags", lambda *args, **kwargs: None)
    monkeypatch.setattr(audit_mod, "_apply_sep_state_reset_flag", lambda *args, **kwargs: None)
    monkeypatch.setattr(audit_mod, "_make_deterministic_batch_plan", lambda *_args, **_kwargs: None)

    out = audit_mod._measure_suite_critic_quality_with_policy_state(
        checkpoint_path="/tmp/fake.cpkt",
        device_obj=torch.device("cpu"),
        env_cfg={"dummy": True},
        num_features=4,
        suite={"batch_size": 1, "h_list": [{"a": 1}], "env_seeds": [11], "rollout_seeds": [22]},
        single_eval_pos=64,
        n_steps=4,
        learning_rate=2e-4,
        batch_size=4,
        n_epochs=1,
        target_kl=0.03,
        ppo_reset_env_state_at_sep=True,
        ppo_separate_value_backbone=False,
        deterministic_actor_sampling=True,
        deterministic_batch_plan=True,
        strict_native_rollout=False,
        actor_gae_space="normalized",
        actor_baseline_mode="learned",
        actor_objective_mode="tokenwise",
        boundary_contract_mode="normal",
        policy_state_dict={},
        build_seed=4040,
        vf_coef_override=1.25,
    )

    assert captured["vf_coef"] == 1.25
    assert out["objective_total"] == 7
    assert out["raw_corr"] == 0.25
    assert out["explained_variance_normalized"] == 0.125
    assert out["explained_variance_raw"] == 0.5


def test_measure_suite_critic_quality_with_policy_state_sets_policy_loss_coef_override(monkeypatch):
    class _DummyPrior:
        def __init__(self, _cfg):
            self.cfg = _cfg

    class _DummyPolicy:
        def load_state_dict(self, _state_dict, strict=True):
            assert strict is True
            return None

        def _dummy_states(self, n_envs):
            return {"n_envs": int(n_envs)}

    class _DummyAlgo:
        def __init__(self):
            self.policy = _DummyPolicy()
            self.rollout_buffer = object()
            self.n_steps = 4
            self._rwkv_policy_loss_coef = None

        def _update_current_progress_remaining(self, *_args, **_kwargs):
            return None

        def collect_rollouts(self, *_args, **_kwargs):
            return True

    class _DummyCallback:
        def init_callback(self, _algo):
            return None

    class _DummyVecEnv:
        num_envs = 1

        def reset(self):
            return {"obs": 1}

        def close(self):
            return None

    algo_holder = {}

    def _fake_load_model(_checkpoint_path, device=None, verbose=False):
        del device, verbose
        return object(), {"optimizer": {"ppo_vf_coef": 0.5, "ppo_gamma": 0.98, "ppo_gae_lambda": 0.90, "ppo_ent_coef": 0.0, "ppo_max_grad_norm": 0.5}}

    _fake_load_model.cache_clear = lambda: None  # type: ignore[attr-defined]

    def _fake_build_recurrent_ppo(**_kwargs):
        algo = _DummyAlgo()
        algo_holder["algo"] = algo
        return algo, _DummyCallback(), _DummyVecEnv()

    monkeypatch.setattr(audit_mod, "load_model", _fake_load_model)
    monkeypatch.setattr(audit_mod, "EnvironmentPrior", _DummyPrior)
    monkeypatch.setattr(audit_mod, "build_recurrent_ppo", _fake_build_recurrent_ppo)
    monkeypatch.setattr(
        audit_mod,
        "_measure_rollout_critic_quality",
        lambda _algo: {
            "objective_total": 7,
            "raw_corr": 0.25,
            "explained_variance_normalized": 0.125,
            "explained_variance_raw": 0.5,
        },
    )
    monkeypatch.setattr(audit_mod, "_apply_boundary_contract_mode_flags", lambda *args, **kwargs: None)
    monkeypatch.setattr(audit_mod, "_apply_sep_state_reset_flag", lambda *args, **kwargs: None)
    monkeypatch.setattr(audit_mod, "_make_deterministic_batch_plan", lambda *_args, **_kwargs: None)

    audit_mod._measure_suite_critic_quality_with_policy_state(
        checkpoint_path="/tmp/fake.cpkt",
        device_obj=torch.device("cpu"),
        env_cfg={"dummy": True},
        num_features=4,
        suite={"batch_size": 1, "h_list": [{"a": 1}], "env_seeds": [11], "rollout_seeds": [22]},
        single_eval_pos=64,
        n_steps=4,
        learning_rate=2e-4,
        batch_size=4,
        n_epochs=1,
        target_kl=0.03,
        ppo_reset_env_state_at_sep=True,
        ppo_separate_value_backbone=False,
        deterministic_actor_sampling=True,
        deterministic_batch_plan=True,
        strict_native_rollout=False,
        actor_gae_space="normalized",
        actor_baseline_mode="learned",
        actor_objective_mode="tokenwise",
        boundary_contract_mode="normal",
        policy_state_dict={},
        build_seed=4040,
        policy_loss_coef_override=0.25,
    )

    assert algo_holder["algo"]._rwkv_policy_loss_coef == 0.25


def test_measure_rollout_critic_quality_aligns_multi_env_flatten_order():
    class _DummyRolloutBuffer:
        def __init__(self):
            self.generator_ready = False
            self.objective_masks = np.array([[1.0, 1.0], [0.0, 0.0]], dtype=np.float32)
            self.values = np.array([[1.0, 20.0], [3.0, 40.0]], dtype=np.float32)
            self.returns = np.array([[1.0, 2.0], [30.0, 40.0]], dtype=np.float32)
            self._raw_values = np.array([[10.0, 200.0], [30.0, 400.0]], dtype=np.float32)
            self._raw_returns = np.array([[10.0, 20.0], [300.0, 400.0]], dtype=np.float32)

        def _ensure_generator_ready(self):
            if self.generator_ready:
                return
            self.generator_ready = True
            self.objective_masks = self.objective_masks.swapaxes(0, 1).reshape(-1)
            self.values = self.values.swapaxes(0, 1).reshape(-1)
            self.returns = self.returns.swapaxes(0, 1).reshape(-1)
            self._raw_values = self._raw_values.swapaxes(0, 1).reshape(-1)
            self._raw_returns = self._raw_returns.swapaxes(0, 1).reshape(-1)

        def _get_flat_raw_values_tensor(self, *, dtype=torch.float32, device=None):
            self._ensure_generator_ready()
            return torch.as_tensor(self._raw_values, dtype=dtype, device=device)

        def _get_flat_raw_returns_tensor(self, *, dtype=torch.float32, device=None):
            self._ensure_generator_ready()
            return torch.as_tensor(self._raw_returns, dtype=dtype, device=device)

    algo = SimpleNamespace(rollout_buffer=_DummyRolloutBuffer())
    out = _measure_rollout_critic_quality(algo)

    expected_norm_ev = float(
        _explained_variance_with_mask(
            [1.0, 3.0, 20.0, 40.0],
            [1.0, 30.0, 2.0, 40.0],
            mask=[1, 0, 1, 0],
        )
    )
    expected_raw_ev = float(
        _explained_variance_with_mask(
            [10.0, 30.0, 200.0, 400.0],
            [10.0, 300.0, 20.0, 400.0],
            mask=[1, 0, 1, 0],
        )
    )

    assert algo.rollout_buffer.generator_ready is True
    assert out["objective_total"] == 2
    assert out["explained_variance_normalized"] == expected_norm_ev
    assert out["explained_variance_raw"] == expected_raw_ev
