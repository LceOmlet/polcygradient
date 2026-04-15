import json
import sys

import torch

from ticl.analysis import phase3_multi_env_optimization_audit as audit_mod
from ticl.analysis import phase3_env_delta_concentration_probe as env_delta_probe_mod
from ticl.analysis import phase3_post_eval_hotspot_profile as hotspot_profile_mod
from ticl.analysis import phase3_post_eval_profile as post_profile_mod
from ticl.analysis import phase3_post_eval_subpath_profile as subpath_profile_mod
from ticl.analysis import phase3_train_env_quality_contrast_probe as train_env_contrast_probe_mod
from ticl.analysis import phase3_train_token_bucket_probe as train_token_bucket_probe_mod
from ticl.analysis import phase3_train_rollout_quality_probe as train_quality_probe_mod
from ticl.analysis import prior_generalization_audit as pga_mod


class _DummyPolicy:
    def make_vectorized_rollout_step_fn(self):
        def _step_fn(obs_t, action_t, reward_t, reward_mask_t, cache, step_idx, env_info):
            return {
                "action": action_t,
                "cache": cache,
            }

        return _step_fn


def test_run_policy_eval_forwards_rollout_backend(monkeypatch):
    captured = {}

    def _fake_run_eval(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(audit_mod, "_run_eval", _fake_run_eval)

    metrics = audit_mod._run_policy_eval(
        prior=object(),
        suite={"batch_size": 1},
        policy=_DummyPolicy(),
        sampled=True,
        n_samples=32,
        num_features=8,
        single_eval_pos=16,
        device_obj=torch.device("cpu"),
        rollout_backend="family_vectorized",
    )

    assert metrics == {"ok": True}
    assert captured["rollout_backend"] == "family_vectorized"
    assert captured["n_samples"] == 32
    assert captured["num_features"] == 8
    assert captured["single_eval_pos"] == 16


def test_resolve_train_profile_phase2_shared_backbone_contract_matches_phase2_milestone():
    profile = audit_mod._resolve_train_profile(
        optimizer_cfg={},
        train_profile="phase2_shared_backbone_contract",
        batch_size=None,
        n_epochs=None,
        learning_rate=None,
        target_kl=None,
    )

    assert profile["train_profile"] == "phase2_shared_backbone_contract"
    assert profile["batch_size"] == 256
    assert profile["n_epochs"] == 1
    assert profile["learning_rate"] == 2e-4
    assert profile["target_kl"] == 0.03
    assert profile["normalize_advantage"] is False
    assert profile["actor_gae_space"] == "normalized"
    assert profile["actor_baseline_mode"] == "learned"
    assert profile["actor_objective_mode"] == "tokenwise"
    assert profile["reset_env_state_at_sep"] is True
    assert profile["separate_value_backbone"] is False
    assert profile["runtime_normalized_q_value_weight_override"] is None
    assert profile["runtime_next_state_flow_matching_weight_override"] is None
    assert profile["vf_coef"] == 0.5
    assert profile["restore_validation_policy_state"] is False


def test_load_reused_zero_control_metrics_accepts_matching_contract(tmp_path):
    payload = {
        "policy": "zero",
        "config": {
            "n_samples": 2048,
            "single_eval_pos": 1946,
            "rollout_backend": "serial",
            "core_a": False,
            "reference_semantics_enabled": False,
        },
        "train_suite": {
            "summary": {"fingerprint": "train_fp"},
            "metrics": {"suffix_return_mean": 1.0, "full_return_mean": 2.0},
        },
        "heldout_suite": {
            "summary": {"fingerprint": "heldout_fp"},
            "metrics": {"suffix_return_mean": 3.0, "full_return_mean": 4.0},
        },
    }
    path = tmp_path / "zero.json"
    path.write_text(__import__("json").dumps(payload))

    reused = audit_mod._load_reused_suite_metrics(
        report_json_path=str(path),
        expected_policy_mode="zero",
        train_suite_summary={"fingerprint": "train_fp"},
        heldout_suite_summary={"fingerprint": "heldout_fp"},
        n_samples=2048,
        single_eval_pos=1946,
        rollout_backend="serial",
    )

    assert reused["train"]["suffix_return_mean"] == 1.0
    assert reused["heldout"]["full_return_mean"] == 4.0


def test_load_reused_zero_control_metrics_rejects_mismatched_contract(tmp_path):
    payload = {
        "policy": "zero",
        "config": {
            "n_samples": 2048,
            "single_eval_pos": 1946,
            "rollout_backend": "serial",
            "core_a": False,
            "reference_semantics_enabled": False,
        },
        "train_suite": {
            "summary": {"fingerprint": "train_fp"},
            "metrics": {"suffix_return_mean": 1.0, "full_return_mean": 2.0},
        },
        "heldout_suite": {
            "summary": {"fingerprint": "heldout_fp"},
            "metrics": {"suffix_return_mean": 3.0, "full_return_mean": 4.0},
        },
    }
    path = tmp_path / "zero_bad.json"
    path.write_text(__import__("json").dumps(payload))

    try:
        audit_mod._load_reused_suite_metrics(
            report_json_path=str(path),
            expected_policy_mode="zero",
            train_suite_summary={"fingerprint": "different_train"},
            heldout_suite_summary={"fingerprint": "heldout_fp"},
            n_samples=2048,
            single_eval_pos=1946,
            rollout_backend="serial",
        )
    except ValueError as exc:
        assert "fingerprint mismatch" in str(exc)
    else:
        raise AssertionError("expected fingerprint mismatch to raise")


def test_load_reused_suite_metrics_accepts_ppo_policy_dict(tmp_path):
    payload = {
        "policy": {"requested_mode": "ppo", "resolved_mode": "ppo", "device": "cpu"},
        "config": {
            "n_samples": 2048,
            "single_eval_pos": 1946,
            "rollout_backend": "serial",
            "core_a": False,
            "reference_semantics_enabled": False,
        },
        "train_suite": {
            "summary": {"fingerprint": "train_fp"},
            "metrics": {"suffix_return_mean": 1.0, "full_return_mean": 2.0},
        },
        "heldout_suite": {
            "summary": {"fingerprint": "heldout_fp"},
            "metrics": {"suffix_return_mean": 3.0, "full_return_mean": 4.0},
        },
    }
    path = tmp_path / "ppo.json"
    path.write_text(__import__("json").dumps(payload))

    reused = audit_mod._load_reused_suite_metrics(
        report_json_path=str(path),
        expected_policy_mode="ppo",
        train_suite_summary={"fingerprint": "train_fp"},
        heldout_suite_summary={"fingerprint": "heldout_fp"},
        n_samples=2048,
        single_eval_pos=1946,
        rollout_backend="serial",
    )

    assert reused["train"]["full_return_mean"] == 2.0


def test_post_policy_bundle_roundtrip(tmp_path):
    policy = torch.nn.Linear(2, 2, bias=False)
    algo = type("Algo", (), {"policy": policy})()
    contract = {
        "checkpoint_path": "/tmp/checkpoint.cpkt",
        "train_suite_fingerprint": "train_fp",
        "heldout_suite_fingerprint": "heldout_fp",
        "n_samples": 2048,
        "single_eval_pos": 1946,
        "rollout_backend": "serial",
        "train_profile": "trusted_sep_reset_mainline",
        "batch_size": 256,
        "n_epochs": 1,
        "learning_rate": 2e-4,
        "target_kl": 0.03,
        "actor_objective_mode": "tokenwise",
        "ppo_actor_baseline_mode": "learned",
        "ppo_normalize_advantage": True,
        "ppo_actor_gae_space": "normalized",
        "ppo_vf_coef": 0.5,
        "ppo_reset_env_state_at_sep": True,
        "ppo_separate_value_backbone": False,
        "ppo_restore_validation_policy_state": True,
        "strict_native_rollout": False,
    }
    bundle_path = tmp_path / "post_policy_bundle.pt"
    train_history = {"history": [1, 2, 3]}

    saved_path = audit_mod._save_post_policy_bundle(
        bundle_path=str(bundle_path),
        algo=algo,
        contract=contract,
        train_history=train_history,
    )

    loaded = audit_mod._load_post_policy_bundle(
        bundle_path=saved_path,
        checkpoint_path="/tmp/checkpoint.cpkt",
        train_suite_summary={"fingerprint": "train_fp"},
        heldout_suite_summary={"fingerprint": "heldout_fp"},
        n_samples=2048,
        single_eval_pos=1946,
        rollout_backend="serial",
        profile_cfg={
            "train_profile": "trusted_sep_reset_mainline",
            "batch_size": 256,
            "n_epochs": 1,
            "learning_rate": 2e-4,
            "target_kl": 0.03,
            "actor_objective_mode": "tokenwise",
            "actor_baseline_mode": "learned",
            "normalize_advantage": True,
                "actor_gae_space": "normalized",
                "vf_coef": 0.5,
                "reset_env_state_at_sep": True,
                "separate_value_backbone": False,
                "restore_validation_policy_state": True,
            },
            strict_native_rollout=False,
        )

    assert loaded["source_path"] == str(bundle_path.resolve())
    assert loaded["train_history"] == train_history
    assert sorted(loaded["policy_state_dict"].keys()) == sorted(policy.state_dict().keys())


def test_run_phase3_audit_resumes_post_policy_bundle_without_train_loop(tmp_path, monkeypatch):
    bundle_policy = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        bundle_policy.weight.fill_(3.0)
    bundle_path = tmp_path / "bundle.pt"
    audit_mod._save_post_policy_bundle(
        bundle_path=str(bundle_path),
        algo=type("Algo", (), {"policy": bundle_policy})(),
        contract={
            "checkpoint_path": str((tmp_path / "checkpoint.cpkt").resolve()),
            "train_suite_fingerprint": "train_fp",
            "heldout_suite_fingerprint": "heldout_fp",
            "n_samples": 2048,
            "single_eval_pos": 1946,
            "rollout_backend": "serial",
            "train_profile": "trusted_sep_reset_mainline",
            "batch_size": 256,
            "n_epochs": 1,
            "learning_rate": 2e-4,
            "target_kl": 0.03,
            "actor_objective_mode": "tokenwise",
            "ppo_actor_baseline_mode": "learned",
            "ppo_normalize_advantage": True,
                "ppo_actor_gae_space": "normalized",
                "ppo_vf_coef": 0.5,
                "ppo_reset_env_state_at_sep": True,
                "ppo_separate_value_backbone": False,
                "ppo_restore_validation_policy_state": True,
                "strict_native_rollout": False,
            },
            train_history={"resumed": True},
        )

    class _VecEnv:
        def close(self):
            return None

    class _Algo:
        def __init__(self):
            self.policy = torch.nn.Linear(2, 2, bias=False)
            self.rollout_buffer = object()

    monkeypatch.setattr(audit_mod, "assert_phase2_green", lambda path: {"validation": {"all_checks_pass": True}, "pass_flags": {}})
    monkeypatch.setattr(audit_mod, "_load_required_suite", lambda path: {"batch_size": 1, "h_list": [], "env_seeds": [1], "rollout_seeds": [2]})
    def _fake_load_model(checkpoint_path, device, verbose=False):
        return object(), {"prior": {"num_features": 8}, "optimizer": {}}

    _fake_load_model.cache_clear = lambda: None
    monkeypatch.setattr(audit_mod, "load_model", _fake_load_model)
    monkeypatch.setattr(audit_mod, "_resolve_audit_env_config", lambda config, core_a=False, reference_semantics_enabled=False: {})
    monkeypatch.setattr(audit_mod, "EnvironmentPrior", lambda cfg: type("Prior", (), {})())
    monkeypatch.setattr(audit_mod, "_bind_vec_env_to_fixed_suite", lambda *args, **kwargs: None)
    monkeypatch.setattr(audit_mod, "_make_deterministic_batch_plan", lambda rollout_buffer: None)
    monkeypatch.setattr(audit_mod, "_summarize_suite", lambda prior, suite: {"fingerprint": "train_fp" if suite is train_suite else "heldout_fp"})

    algo = _Algo()
    monkeypatch.setattr(audit_mod, "build_recurrent_ppo", lambda **kwargs: (algo, object(), _VecEnv()))
    monkeypatch.setattr(
        audit_mod,
        "_load_reused_suite_metrics",
        lambda **kwargs: {
            "source_path": str(tmp_path / "reuse.json"),
            "train": {"full_return_mean": 1.0, "suffix_return_mean": 2.0},
            "heldout": {"full_return_mean": 3.0, "suffix_return_mean": 4.0},
        },
    )
    monkeypatch.setattr(
        audit_mod,
        "_run_policy_eval",
        lambda **kwargs: {"full_return_mean": 5.0, "suffix_return_mean": 6.0},
    )
    monkeypatch.setattr(
        audit_mod,
        "_audit_train_loop",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("train_loop should not run when resuming post policy bundle")),
    )

    train_suite = {"batch_size": 1, "h_list": [], "env_seeds": [1], "rollout_seeds": [2]}
    heldout_suite = {"batch_size": 1, "h_list": [], "env_seeds": [3], "rollout_seeds": [4]}
    monkeypatch.setattr(
        audit_mod,
        "_load_required_suite",
        lambda path: train_suite if "train" in str(path) else heldout_suite,
    )

    report = audit_mod.run_phase3_multi_env_optimization_audit(
        checkpoint_path=str((tmp_path / "checkpoint.cpkt").resolve()),
        train_suite_path=str(tmp_path / "train_suite.pt"),
        heldout_suite_path=str(tmp_path / "heldout_suite.pt"),
        device="cpu",
        n_samples=2048,
        single_eval_pos=1946,
        outer_epochs=1,
        rollout_backend="serial",
        train_profile="trusted_sep_reset_mainline",
        phase2_summary_path=str(tmp_path / "phase2.json"),
        reuse_zero_control_json=str(tmp_path / "zero.json"),
        reuse_pre_policy_json=str(tmp_path / "pre.json"),
        resume_post_policy_bundle_path=str(bundle_path),
    )

    assert report["post_policy_bundle"]["resumed"] is True
    assert report["train_history"] == {"resumed": True}
    assert torch.allclose(algo.policy.weight.detach(), torch.full_like(algo.policy.weight.detach(), 3.0))


def test_run_phase3_audit_forwards_pair2_runtime_scope_and_strict_contract(tmp_path, monkeypatch):
    captured = {}

    class _VecEnv:
        def close(self):
            return None

    class _Algo:
        def __init__(self):
            self.policy = torch.nn.Linear(2, 2, bias=False)
            self.rollout_buffer = object()

    train_suite = {"batch_size": 1, "h_list": [], "env_seeds": [11], "rollout_seeds": [22]}
    heldout_suite = {"batch_size": 1, "h_list": [], "env_seeds": [33], "rollout_seeds": [44]}

    monkeypatch.setattr(audit_mod, "assert_phase2_green", lambda path: {"validation": {"all_checks_pass": True}, "pass_flags": {}})
    monkeypatch.setattr(
        audit_mod,
        "_load_required_suite",
        lambda path: train_suite if "train" in str(path) else heldout_suite,
    )
    def _fake_load_model(checkpoint_path, device, verbose=False):
        return object(), {"prior": {"num_features": 8}, "optimizer": {}}

    _fake_load_model.cache_clear = lambda: None
    monkeypatch.setattr(audit_mod, "load_model", _fake_load_model)
    monkeypatch.setattr(audit_mod, "_resolve_audit_env_config", lambda config, core_a=False, reference_semantics_enabled=False: {})
    monkeypatch.setattr(audit_mod, "EnvironmentPrior", lambda cfg: type("Prior", (), {})())
    monkeypatch.setattr(audit_mod, "_bind_vec_env_to_fixed_suite", lambda *args, **kwargs: None)
    monkeypatch.setattr(audit_mod, "_make_deterministic_batch_plan", lambda rollout_buffer: None)
    monkeypatch.setattr(audit_mod, "_summarize_suite", lambda prior, suite: {"fingerprint": "train_fp" if suite is train_suite else "heldout_fp"})
    monkeypatch.setattr(
        audit_mod,
        "_load_reused_suite_metrics",
        lambda **kwargs: {
            "source_path": str(tmp_path / "reuse.json"),
            "train": {"full_return_mean": 1.0, "suffix_return_mean": 2.0},
            "heldout": {"full_return_mean": 3.0, "suffix_return_mean": 4.0},
        },
    )
    monkeypatch.setattr(
        audit_mod,
        "_run_policy_eval",
        lambda **kwargs: {"full_return_mean": 5.0, "suffix_return_mean": 6.0},
    )
    monkeypatch.setattr(audit_mod, "_audit_train_loop", lambda *args, **kwargs: [{"outer_epoch": 1}])

    def _fake_build_recurrent_ppo(**kwargs):
        captured.update(kwargs)
        return _Algo(), object(), _VecEnv()

    monkeypatch.setattr(audit_mod, "build_recurrent_ppo", _fake_build_recurrent_ppo)

    report = audit_mod.run_phase3_multi_env_optimization_audit(
        checkpoint_path=str((tmp_path / "checkpoint.cpkt").resolve()),
        train_suite_path=str(tmp_path / "phase3_cross_env_baseline_pair2/suites/train_suite.pt"),
        heldout_suite_path=str(tmp_path / "phase3_cross_env_baseline_pair2/suites/heldout_suite.pt"),
        device="cpu",
        n_samples=2048,
        single_eval_pos=1946,
        outer_epochs=1,
        rollout_backend="serial",
        train_profile="trusted_sep_reset_mainline",
        phase2_summary_path=str(tmp_path / "phase2.json"),
        reuse_zero_control_json=str(tmp_path / "zero.json"),
        reuse_pre_policy_json=str(tmp_path / "pre.json"),
        strict_fixed_env_mode=True,
        deterministic_actor_sampling=True,
        deterministic_batch_plan=True,
        actor_objective_mode_override="tokenwise_scale_env12_mid_episode_extension_block",
        actor_objective_runtime_current_suite_name="pair2",
        train_outer_batch_snapshot_json=str(tmp_path / "outer_batch_snapshot.json"),
        train_outer_batch_selector_trace_json=str(tmp_path / "outer_batch_selector_trace.json"),
        train_outer_batch_snapshot_target_env_index=12,
        train_outer_batch_snapshot_target_objective_episode_index=0,
        train_outer_batch_snapshot_target_objective_position_start=18,
        train_outer_batch_snapshot_target_objective_position_end=21,
    )

    assert captured["strict_fixed_env_mode"] is True
    assert captured["deterministic_actor_sampling"] is True
    assert captured["deterministic_batch_plan"] is True
    assert captured["strict_native_rollout"] is False
    assert captured["actor_objective_runtime_current_suite_name"] == "pair2"
    assert captured["actor_objective_mode"] == "tokenwise_scale_env12_mid_episode_extension_block"
    assert captured["env_rng_seeds"] == [11]
    assert captured["rollout_rng_seeds"] == [22]
    assert report["config"]["actor_objective_runtime_current_suite_name"] == "pair2"
    assert report["config"]["actor_objective_mode_override"] == "tokenwise_scale_env12_mid_episode_extension_block"
    assert report["config"]["strict_native_rollout"] is False
    assert report["config"]["train_outer_batch_snapshot_json"] == str(
        (tmp_path / "outer_batch_snapshot.json").resolve()
    )
    assert report["config"]["train_outer_batch_selector_trace_json"] == str(
        (tmp_path / "outer_batch_selector_trace.json").resolve()
    )
    assert report["config"]["train_outer_batch_snapshot_target_env_index"] == 12
    assert report["config"]["train_outer_batch_snapshot_target_objective_episode_index"] == 0
    assert report["config"]["train_outer_batch_snapshot_target_objective_position_start"] == 18
    assert report["config"]["train_outer_batch_snapshot_target_objective_position_end"] == 21
    assert report["train_outer_batch_snapshot"]["requested_path"] == str(
        (tmp_path / "outer_batch_snapshot.json").resolve()
    )
    assert report["train_outer_batch_snapshot"]["selector"]["target_env_index"] == 12
    assert report["train_outer_batch_snapshot"]["selector"]["target_objective_episode_index"] == 0
    assert report["train_outer_batch_snapshot"]["selector"]["target_objective_position_start"] == 18
    assert report["train_outer_batch_snapshot"]["selector"]["target_objective_position_end"] == 21
    assert report["train_outer_batch_snapshot"]["written"] is False
    assert report["train_outer_batch_selector_trace"]["requested_path"] == str(
        (tmp_path / "outer_batch_selector_trace.json").resolve()
    )
    assert report["train_outer_batch_selector_trace"]["selector"]["target_env_index"] == 12
    assert report["train_outer_batch_selector_trace"]["selector"]["target_objective_episode_index"] == 0
    assert report["train_outer_batch_selector_trace"]["selector"]["target_objective_position_start"] == 18
    assert report["train_outer_batch_selector_trace"]["selector"]["target_objective_position_end"] == 21
    assert report["train_outer_batch_selector_trace"]["completed"] is False
    assert report["train_outer_batch_selector_trace"]["row_count"] == 0


def test_run_phase3_audit_phase2_shared_backbone_contract_uses_phase2_env_cfg(tmp_path, monkeypatch):
    captured = {}

    class _VecEnv:
        def close(self):
            return None

    class _Algo:
        def __init__(self):
            self.policy = torch.nn.Linear(2, 2, bias=False)
            self.rollout_buffer = object()

    train_suite = {"batch_size": 1, "h_list": [], "env_seeds": [11], "rollout_seeds": [22]}
    heldout_suite = {"batch_size": 1, "h_list": [], "env_seeds": [33], "rollout_seeds": [44]}

    monkeypatch.setattr(
        audit_mod,
        "assert_phase2_green",
        lambda path: {"validation": {"all_checks_pass": True}, "pass_flags": {}, "trusted_contract": {}},
    )
    monkeypatch.setattr(
        audit_mod,
        "_load_required_suite",
        lambda path: train_suite if "train" in str(path) else heldout_suite,
    )

    def _fake_load_model(checkpoint_path, device, verbose=False):
        return object(), {
            "prior": {"num_features": 8, "environment": {"family": "exact_scm"}},
            "optimizer": {},
        }

    _fake_load_model.cache_clear = lambda: None
    monkeypatch.setattr(audit_mod, "load_model", _fake_load_model)

    def _fake_build_audit_env_cfg(environment_cfg):
        captured["build_audit_env_cfg_arg"] = dict(environment_cfg)
        return {"phase2_env_cfg": True}

    monkeypatch.setattr(audit_mod, "_build_audit_env_cfg", _fake_build_audit_env_cfg)
    monkeypatch.setattr(
        audit_mod,
        "_resolve_audit_env_config",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected fallback env cfg path")),
    )
    monkeypatch.setattr(audit_mod, "EnvironmentPrior", lambda cfg: type("Prior", (), {})())
    monkeypatch.setattr(audit_mod, "_bind_vec_env_to_fixed_suite", lambda *args, **kwargs: None)
    monkeypatch.setattr(audit_mod, "_make_deterministic_batch_plan", lambda rollout_buffer: None)
    monkeypatch.setattr(
        audit_mod,
        "_summarize_suite",
        lambda prior, suite: {"fingerprint": "train_fp" if suite is train_suite else "heldout_fp"},
    )
    monkeypatch.setattr(
        audit_mod,
        "_load_reused_suite_metrics",
        lambda **kwargs: {
            "source_path": str(tmp_path / "reuse.json"),
            "train": {"full_return_mean": 1.0, "suffix_return_mean": 2.0},
            "heldout": {"full_return_mean": 3.0, "suffix_return_mean": 4.0},
        },
    )
    monkeypatch.setattr(
        audit_mod,
        "_run_policy_eval",
        lambda **kwargs: {"full_return_mean": 5.0, "suffix_return_mean": 6.0},
    )
    monkeypatch.setattr(audit_mod, "_audit_train_loop", lambda *args, **kwargs: [{"outer_epoch": 1}])

    def _fake_build_recurrent_ppo(**kwargs):
        captured["ppo_kwargs"] = dict(kwargs)
        return _Algo(), object(), _VecEnv()

    monkeypatch.setattr(audit_mod, "build_recurrent_ppo", _fake_build_recurrent_ppo)

    report = audit_mod.run_phase3_multi_env_optimization_audit(
        checkpoint_path=str((tmp_path / "checkpoint.cpkt").resolve()),
        train_suite_path=str(tmp_path / "train_suite.pt"),
        heldout_suite_path=str(tmp_path / "heldout_suite.pt"),
        device="cpu",
        n_samples=256,
        single_eval_pos=64,
        outer_epochs=1,
        rollout_backend="serial",
        train_profile="phase2_shared_backbone_contract",
        phase2_summary_path=str(tmp_path / "phase2.json"),
        reuse_zero_control_json=str(tmp_path / "zero.json"),
        reuse_pre_policy_json=str(tmp_path / "pre.json"),
        strict_fixed_env_mode=True,
        deterministic_actor_sampling=True,
        deterministic_batch_plan=True,
    )

    assert captured["build_audit_env_cfg_arg"] == {"family": "exact_scm"}
    assert captured["ppo_kwargs"]["normalize_advantage"] is False
    assert captured["ppo_kwargs"]["actor_baseline_mode"] == "learned"
    assert captured["ppo_kwargs"]["actor_objective_mode"] == "tokenwise"
    assert captured["ppo_kwargs"]["reset_env_state_at_sep"] is True
    assert captured["ppo_kwargs"]["separate_value_backbone"] is False
    assert captured["ppo_kwargs"]["restore_validation_policy_state"] is False
    assert captured["ppo_kwargs"]["runtime_normalized_q_value_weight_override"] is None
    assert captured["ppo_kwargs"]["runtime_next_state_flow_matching_weight_override"] is None
    assert captured["ppo_kwargs"]["env_rng_seeds"] == [11]
    assert captured["ppo_kwargs"]["rollout_rng_seeds"] == [22]
    assert report["train_profile"] == "phase2_shared_backbone_contract"
    assert report["config"]["ppo_normalize_advantage"] is False
    assert report["config"]["ppo_restore_validation_policy_state"] is False
    assert report["config"]["runtime_normalized_q_value_weight_override"] is None
    assert report["config"]["runtime_next_state_flow_matching_weight_override"] is None


def test_collect_suite_rewards_serial_profiled_reports_per_env_timings():
    class _FakePrior:
        def __init__(self):
            self.grad_enabled_flags = []

        def _sample_environment(self, h, device, rng_seed):
            return {"h": h, "seed": int(rng_seed)}

        def _rollout_single(
            self,
            *,
            env,
            n_samples,
            num_features,
            single_eval_pos,
            device,
            policy_step_fn,
            collect_x,
            collect_runtime_info,
            rng_seed,
            store_rewards,
            policy_objective_kind,
        ):
            del num_features, single_eval_pos, device, policy_step_fn, collect_x, collect_runtime_info, store_rewards, policy_objective_kind
            self.grad_enabled_flags.append(bool(torch.is_grad_enabled()))
            values = torch.arange(int(n_samples), dtype=torch.float32) + float(env["seed"] + int(rng_seed))
            return None, values, {"seed": int(rng_seed)}

    suite = {
        "batch_size": 2,
        "h_list": [{"id": 0}, {"id": 1}],
        "env_seeds": [10, 20],
        "rollout_seeds": [100, 200],
    }
    prior = _FakePrior()
    profile = pga_mod.collect_suite_rewards_serial_profiled(
        prior,
        suite=suite,
        policy_step_fn=lambda *args, **kwargs: {"action": torch.zeros(1)},
        n_samples=4,
        num_features=3,
        single_eval_pos=2,
        device=torch.device("cpu"),
    )

    assert profile["suite_batch_size"] == 2
    assert profile["profiled_env_count"] == 2
    assert len(profile["env_profiles"]) == 2
    assert profile["env_profiles"][0]["env_seed"] == 10
    assert profile["env_profiles"][1]["rollout_seed"] == 200
    assert profile["aggregate"]["total_wall_s"] >= 0.0
    assert prior.grad_enabled_flags == [False, False]


def test_phase3_post_eval_profile_reuses_existing_output(tmp_path, monkeypatch, capsys):
    output_path = tmp_path / "profile.json"
    payload = {"ok": True, "kind": "post_eval"}
    output_path.write_text(json.dumps(payload))

    monkeypatch.setattr(
        post_profile_mod,
        "assert_phase2_green",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not preflight when reusing output")),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "phase3_post_eval_profile.py",
            "/tmp/checkpoint.cpkt",
            "--train-suite-path",
            "/tmp/train.pt",
            "--heldout-suite-path",
            "/tmp/heldout.pt",
            "--post-policy-bundle-path",
            "/tmp/bundle.pt",
            "--suite-target",
            "train",
            "--output-json",
            str(output_path),
        ],
    )

    assert post_profile_mod.main() == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == payload


def test_phase3_post_eval_hotspot_profile_reuses_existing_output(tmp_path, monkeypatch, capsys):
    output_path = tmp_path / "hotspot.json"
    payload = {"ok": True, "kind": "hotspot"}
    output_path.write_text(json.dumps(payload))

    monkeypatch.setattr(
        hotspot_profile_mod,
        "assert_phase2_green",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not preflight when reusing output")),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "phase3_post_eval_hotspot_profile.py",
            "/tmp/checkpoint.cpkt",
            "--train-suite-path",
            "/tmp/train.pt",
            "--heldout-suite-path",
            "/tmp/heldout.pt",
            "--post-policy-bundle-path",
            "/tmp/bundle.pt",
            "--suite-target",
            "heldout",
            "--env-index",
            "0",
            "--output-json",
            str(output_path),
        ],
    )

    assert hotspot_profile_mod.main() == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == payload


def test_phase3_post_eval_subpath_profile_reuses_existing_output(tmp_path, monkeypatch, capsys):
    output_path = tmp_path / "subpath.json"
    payload = {"ok": True, "kind": "subpath"}
    output_path.write_text(json.dumps(payload))

    monkeypatch.setattr(
        subpath_profile_mod,
        "assert_phase2_green",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not preflight when reusing output")),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "phase3_post_eval_subpath_profile.py",
            "/tmp/checkpoint.cpkt",
            "--train-suite-path",
            "/tmp/train.pt",
            "--heldout-suite-path",
            "/tmp/heldout.pt",
            "--post-policy-bundle-path",
            "/tmp/bundle.pt",
            "--suite-target",
            "train",
            "--env-index",
            "0",
            "--output-json",
            str(output_path),
        ],
    )

    assert subpath_profile_mod.main() == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == payload


def test_phase3_env_delta_concentration_probe_reuses_existing_output(tmp_path, monkeypatch, capsys):
    output_path = tmp_path / "env_delta.json"
    payload = {"ok": True, "kind": "env_delta"}
    output_path.write_text(json.dumps(payload))

    monkeypatch.setattr(
        env_delta_probe_mod,
        "assert_phase2_green",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not preflight when reusing output")),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "phase3_env_delta_concentration_probe.py",
            "/tmp/checkpoint.cpkt",
            "--train-suite-path",
            "/tmp/train.pt",
            "--heldout-suite-path",
            "/tmp/heldout.pt",
            "--reuse-zero-control-json",
            "/tmp/zero.json",
            "--reuse-pre-policy-json",
            "/tmp/pre.json",
            "--post-policy-bundle-path",
            "/tmp/bundle.pt",
            "--output-json",
            str(output_path),
        ],
    )

    assert env_delta_probe_mod.main() == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == payload


def test_phase3_train_rollout_quality_probe_reuses_existing_output(tmp_path, monkeypatch, capsys):
    output_path = tmp_path / "train_quality.json"
    payload = {"ok": True, "kind": "train_rollout_quality"}
    output_path.write_text(json.dumps(payload))

    monkeypatch.setattr(
        train_quality_probe_mod,
        "assert_phase2_green",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not preflight when reusing output")),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "phase3_train_rollout_quality_probe.py",
            "/tmp/checkpoint.cpkt",
            "--train-suite-path",
            "/tmp/train.pt",
            "--heldout-suite-path",
            "/tmp/heldout.pt",
            "--reuse-zero-control-json",
            "/tmp/zero.json",
            "--reuse-pre-policy-json",
            "/tmp/pre.json",
            "--output-json",
            str(output_path),
        ],
    )

    assert train_quality_probe_mod.main() == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == payload


def test_phase3_train_rollout_quality_probe_collect_contract_uses_suite_seeds_and_determinism():
    train_suite = {
        "env_seeds": [11, 22, 33],
        "rollout_seeds": [44, 55, 66],
    }

    contract = train_quality_probe_mod._train_rollout_quality_collect_contract_kwargs(train_suite)

    assert contract == {
        "strict_fixed_env_mode": True,
        "env_rng_seeds": [11, 22, 33],
        "rollout_rng_seeds": [44, 55, 66],
        "deterministic_actor_sampling": True,
        "deterministic_batch_plan": True,
    }


def test_phase3_train_env_quality_contrast_probe_reuses_existing_output(tmp_path, capsys, monkeypatch):
    output_path = tmp_path / "train_env_contrast.json"
    payload = {"ok": True, "kind": "train_env_contrast"}
    output_path.write_text(json.dumps(payload))

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "phase3_train_env_quality_contrast_probe.py",
            "--rollout-quality-json",
            "/tmp/rollout_quality.json",
            "--output-json",
            str(output_path),
        ],
    )

    assert train_env_contrast_probe_mod.main() == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == payload


def test_phase3_train_env_quality_contrast_probe_labels_regime_context(tmp_path, capsys, monkeypatch):
    rollout_quality_path = tmp_path / "rollout_quality.json"
    expected_summary_path = tmp_path / "expected_summary.json"
    regime_summary_path = tmp_path / "regime_summary.json"
    output_path = tmp_path / "train_env_contrast_regime.json"

    rollout_quality_payload = {
        "train_suite_summary": {"fingerprint": "train_fp"},
        "env_items": [
            {
                "env_index": 0,
                "env_seed": 10,
                "normalized_actor_adv_positive_mass": 8.0,
                "normalized_actor_adv_first16_positive_share": 0.2,
                "normalized_actor_adv_last16_positive_share": 0.1,
                "objective_episode_count": 1,
                "objective_terminal_reset_count": 0,
                "pre_vs_zero_suffix_gap": 1.0,
                "raw_value_return_corr": 0.3,
                "suffix_gap_delta": 0.5,
            },
            {
                "env_index": 1,
                "env_seed": 11,
                "normalized_actor_adv_positive_mass": 16.0,
                "normalized_actor_adv_first16_positive_share": 0.1,
                "normalized_actor_adv_last16_positive_share": 0.2,
                "objective_episode_count": 1,
                "objective_terminal_reset_count": 0,
                "pre_vs_zero_suffix_gap": -1.0,
                "raw_value_return_corr": 0.1,
                "suffix_gap_delta": -0.25,
            },
        ],
    }
    rollout_quality_path.write_text(json.dumps(rollout_quality_payload))
    expected_summary_path.write_text(json.dumps({"train_suite_summary": {"fingerprint": "train_fp"}}))
    regime_summary_path.write_text(
        json.dumps({"regime_groups": {"positive": {"suite_names": ["pair1"]}, "nonpositive": {"suite_names": []}}})
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "phase3_train_env_quality_contrast_probe.py",
            "--rollout-quality-json",
            str(rollout_quality_path),
            "--suite-name",
            "pair1",
            "--suite-regime",
            "positive",
            "--expected-train-suite-summary-json",
            str(expected_summary_path),
            "--suite-regime-summary-json",
            str(regime_summary_path),
            "--output-json",
            str(output_path),
        ],
    )

    assert train_env_contrast_probe_mod.main() == 0
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result["suite_context"]["suite_name"] == "pair1"
    assert result["suite_context"]["suite_regime"] == "positive"
    assert result["suite_context"]["train_suite_summary_fingerprint_match"] is True
    assert result["suite_context"]["suite_regime_summary_membership_verified"] is True
    assert result["groups"]["high_mass_nonpositive"]["summary"]["count"] == 1
    assert result["groups"]["low_mass_positive"]["summary"]["count"] == 1


def test_phase3_train_token_bucket_probe_reuses_existing_output(tmp_path, capsys, monkeypatch):
    output_path = tmp_path / "train_token_bucket.json"
    payload = {"ok": True, "kind": "train_token_bucket"}
    output_path.write_text(json.dumps(payload))

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "phase3_train_token_bucket_probe.py",
            "/tmp/checkpoint.cpkt",
            "--train-suite-path",
            "/tmp/train.pt",
            "--heldout-suite-path",
            "/tmp/heldout.pt",
            "--reuse-zero-control-json",
            "/tmp/zero.json",
            "--reuse-pre-policy-json",
            "/tmp/pre.json",
            "--reuse-train-subset-delta-json",
            "/tmp/delta.json",
            "--output-json",
            str(output_path),
        ],
    )

    assert train_token_bucket_probe_mod.main() == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == payload
