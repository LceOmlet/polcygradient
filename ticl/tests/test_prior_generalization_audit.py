import math

import torch

from ticl.analysis.prior_generalization_audit import (
    build_rollout_resample_suite,
    evaluate_action_reachability,
    evaluate_prior_suite,
    run_prior_generalization_audit,
    sample_fixed_prior_suite,
)
from ticl.model_configs import get_model_default_config
from ticl.priors.environment_prior import EnvironmentPrior


def _make_core_a_prior():
    config = get_model_default_config("rlpfn")
    env_cfg = dict(config["prior"]["environment"])
    env_cfg.update(
        {
            "family": {"distribution": "meta_choice", "choice_values": ["scm"]},
            "strict_joint_transition_enabled": True,
            "terminal_reset_enabled": False,
            "reward_dropout_enabled": False,
            "reward_dropout_randomize": False,
            "reward_dropout_ratio": 0.0,
            "ctrl_reward_weight": 0.0,
            "ctrl_reward_enable_prob": 0.0,
            "survival_reward_weight": 0.0,
            "survival_reward_enable_prob": 0.0,
            "normalized_q_value_weight": 0.0,
            "next_state_flow_matching_weight": 0.0,
            "state_input_scale_enabled": False,
            "state_full_rms_enabled": False,
            "action_noise_train_std": 0.0,
            "action_noise_eval_std": 0.0,
            "reinforce_action_transform": "none",
            "reinforce_reward_transform": "none",
            "batch_parallel_backend": "torch_vectorized",
            "batch_vectorized_grouping": "family",
        }
    )
    return config, EnvironmentPrior(env_cfg)


def test_sample_fixed_prior_suite_is_reproducible():
    _, prior = _make_core_a_prior()
    suite_a = sample_fixed_prior_suite(prior, batch_size=4, suite_seed=1234)
    suite_b = sample_fixed_prior_suite(prior, batch_size=4, suite_seed=1234)
    suite_c = sample_fixed_prior_suite(prior, batch_size=4, suite_seed=5678)

    sig_a = [prior._environment_structure_signature(h) for h in suite_a["h_list"]]
    sig_b = [prior._environment_structure_signature(h) for h in suite_b["h_list"]]
    sig_c = [prior._environment_structure_signature(h) for h in suite_c["h_list"]]

    assert sig_a == sig_b
    assert suite_a["env_seeds"] == suite_b["env_seeds"]
    assert suite_a["rollout_seeds"] == suite_b["rollout_seeds"]
    assert sig_a != sig_c or suite_a["env_seeds"] != suite_c["env_seeds"]


def test_evaluate_prior_suite_zero_policy_reports_finite_metrics():
    config, prior = _make_core_a_prior()
    suite = sample_fixed_prior_suite(prior, batch_size=4, suite_seed=4321)

    def _zero_policy(obs_t, action_t, reward_t, reward_mask_t, cache, step_idx, env_info):
        del reward_t, reward_mask_t, cache, step_idx, env_info
        return torch.zeros_like(action_t)

    metrics = evaluate_prior_suite(
        prior,
        suite=suite,
        policy_step_fn=_zero_policy,
        n_samples=32,
        num_features=int(config["prior"]["num_features"]),
        single_eval_pos=16,
        device=torch.device("cpu"),
    )

    assert metrics["batch_size"] == 4
    assert metrics["single_eval_pos"] == 16
    assert math.isfinite(metrics["full_return_mean"])
    assert math.isfinite(metrics["suffix_return_mean"])
    assert len(metrics["full_return_per_env"]) == 4
    assert len(metrics["suffix_return_per_env"]) == 4


def test_evaluate_prior_suite_zero_policy_serial_reports_finite_metrics():
    config, prior = _make_core_a_prior()
    suite = sample_fixed_prior_suite(prior, batch_size=3, suite_seed=2468)

    def _zero_policy(obs_t, action_t, reward_t, reward_mask_t, cache, step_idx, env_info):
        del reward_t, reward_mask_t, cache, step_idx, env_info
        return torch.zeros_like(action_t)

    metrics = evaluate_prior_suite(
        prior,
        suite=suite,
        policy_step_fn=_zero_policy,
        n_samples=24,
        num_features=int(config["prior"]["num_features"]),
        single_eval_pos=12,
        device=torch.device("cpu"),
        rollout_backend="serial",
    )

    assert metrics["rollout_backend"] == "serial"
    assert metrics["batch_size"] == 3
    assert math.isfinite(metrics["full_return_mean"])
    assert math.isfinite(metrics["suffix_return_mean"])


def test_evaluate_action_reachability_serial_smoke():
    config, prior = _make_core_a_prior()
    suite = sample_fixed_prior_suite(prior, batch_size=2, suite_seed=8642)

    metrics = evaluate_action_reachability(
        prior,
        suite=suite,
        n_samples=24,
        num_features=int(config["prior"]["num_features"]),
        single_eval_pos=12,
        device=torch.device("cpu"),
    )

    assert metrics["probe_backend"] == "serial"
    assert len(metrics["reward_step0_delta_abs_per_env"]) == 2
    assert len(metrics["suffix_return_delta_abs_per_env"]) == 2
    assert len(metrics["obs_max_abs_delta_per_env"]) == 2
    assert 0.0 <= metrics["suffix_return_delta_abs_lt_1e_3_share"] <= 1.0
    assert 0.0 <= metrics["obs_max_abs_delta_gt_1e_2_share"] <= 1.0


def test_build_rollout_resample_suite_reuses_envs_and_changes_rollout_seeds():
    _, prior = _make_core_a_prior()
    base_suite = sample_fixed_prior_suite(prior, batch_size=3, suite_seed=777)
    train_suite = build_rollout_resample_suite(
        prior,
        base_suite=base_suite,
        rollout_suite_seed=111,
        rollouts_per_env=2,
    )
    heldout_suite = build_rollout_resample_suite(
        prior,
        base_suite=base_suite,
        rollout_suite_seed=222,
        rollouts_per_env=2,
    )

    assert train_suite["base_batch_size"] == 3
    assert train_suite["batch_size"] == 6
    assert heldout_suite["base_batch_size"] == 3
    assert heldout_suite["batch_size"] == 6
    assert train_suite["env_seeds"] == heldout_suite["env_seeds"]
    assert train_suite["rollout_seeds"] != heldout_suite["rollout_seeds"]
    assert train_suite["base_env_indices"] == [0, 0, 1, 1, 2, 2]
    assert heldout_suite["base_env_indices"] == [0, 0, 1, 1, 2, 2]


def test_run_prior_generalization_audit_zero_policy_core_a_smoke():
    report = run_prior_generalization_audit(
        checkpoint_path=None,
        policy_mode="zero",
        device="cpu",
        n_samples=24,
        num_features=434,
        single_eval_pos=12,
        train_suite_batch_size=3,
        heldout_suite_batch_size=3,
        train_suite_seed=101,
        heldout_suite_seed=202,
        core_a=True,
        reference_semantics_enabled=False,
    )

    assert report["audit_entry"] == "prior_generalization_audit"
    assert report["policy"]["resolved_mode"] == "zero"
    assert report["train_suite"]["metrics"]["batch_size"] == 3
    assert report["heldout_suite"]["metrics"]["batch_size"] == 3
    assert math.isfinite(report["comparison"]["heldout_minus_train_suffix_return_mean"])


def test_run_prior_generalization_audit_zero_policy_same_env_smoke():
    report = run_prior_generalization_audit(
        checkpoint_path=None,
        policy_mode="zero",
        device="cpu",
        n_samples=24,
        num_features=434,
        single_eval_pos=12,
        suite_regime="same_env",
        base_suite_batch_size=2,
        base_suite_seed=333,
        train_rollouts_per_env=2,
        heldout_rollouts_per_env=3,
        train_rollout_seed=444,
        heldout_rollout_seed=555,
        core_a=True,
        reference_semantics_enabled=False,
    )

    assert report["audit_entry"] == "prior_generalization_audit"
    assert report["config"]["suite_regime"] == "same_env"
    assert report["base_suite"]["summary"]["batch_size"] == 2
    assert report["train_suite"]["metrics"]["batch_size"] == 4
    assert report["heldout_suite"]["metrics"]["batch_size"] == 6
    assert report["paired_metrics"]["base_env_batch_size"] == 2
    assert len(report["paired_metrics"]["heldout_minus_train_suffix_return_mean_per_base_env"]) == 2


def test_run_prior_generalization_audit_zero_policy_same_env_serial_smoke():
    report = run_prior_generalization_audit(
        checkpoint_path=None,
        policy_mode="zero",
        device="cpu",
        n_samples=24,
        num_features=434,
        single_eval_pos=12,
        suite_regime="same_env",
        base_suite_batch_size=2,
        base_suite_seed=933,
        train_rollouts_per_env=1,
        heldout_rollouts_per_env=1,
        train_rollout_seed=944,
        heldout_rollout_seed=955,
        core_a=True,
        reference_semantics_enabled=False,
        rollout_backend="serial",
        include_action_reachability_probe=True,
    )

    assert report["config"]["rollout_backend"] == "serial"
    assert report["train_suite"]["metrics"]["rollout_backend"] == "serial"
    assert isinstance(report["base_suite"]["action_reachability"], dict)
    assert isinstance(report["train_suite"]["action_reachability"], dict)
