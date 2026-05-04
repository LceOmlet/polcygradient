from ticl.analysis.fit_model_phase3_anchor_alignment_probe import (
    _build_fit_model_anchor_compare_config,
    _localize,
    _validation_pair_summary,
)


def test_build_fit_model_anchor_compare_config_enables_strict_compare_bridge():
    config = _build_fit_model_anchor_compare_config(
        checkpoint_path="/home/chen/RLPFN/rwkv/models_diff/rlpfn_03_30_2026_22_37_53_epoch_13.cpkt",
        n_steps=256,
        batch_size=256,
        learning_rate=2e-4,
        train_env_seed=2020,
        rollout_seed=4040,
    )
    optimizer = config["optimizer"]

    assert optimizer["ppo_restore_validation_policy_state"] is True
    assert optimizer["ppo_strict_fixed_env_mode"] is True
    assert optimizer["ppo_env_rng_seeds"] == [2020]
    assert optimizer["ppo_rollout_rng_seeds"] == [4040]
    assert optimizer["ppo_deterministic_actor_sampling"] is True
    assert optimizer["ppo_deterministic_batch_plan"] is True
    assert optimizer["ppo_strict_native_rollout"] is True


def test_localize_prefers_unrestored_validation_policy_state_reason():
    report = {
        "fit_model_runtime": {
            "checkpoint_saved_validation_policy_state_present": True,
            "restore_validation_policy_state": False,
            "strict_native_rollout": False,
        },
        "training_compare": {
            "grad_compare": {
                "cosine_similarity": 0.9,
                "l2_delta_norm": 1.0,
            },
            "saved_policy_state_compare": {"max_abs_diff": 0.5},
        },
        "validation_compare": {
            "action_mean": {"max_abs_diff": 0.1},
            "value_logits": {"max_abs_diff": 0.2},
        },
    }

    localized = _localize(report)

    assert localized["stage"] == "unrestored_validation_policy_state"


def test_localize_detects_validation_e2e_semantic_drift_after_single_step_alignment():
    report = {
        "fit_model_runtime": {
            "checkpoint_saved_validation_policy_state_present": True,
            "restore_validation_policy_state": True,
            "strict_native_rollout": True,
        },
        "training_compare": {
            "grad_compare": {
                "cosine_similarity": 1.0,
                "l2_delta_norm": 0.0,
            },
            "saved_policy_state_compare": {"max_abs_diff": 0.0},
        },
        "validation_compare": {
            "action_mean": {"max_abs_diff": 0.0},
            "value_logits": {"max_abs_diff": 0.0},
        },
        "validation_e2e_compare": {
            "conclusion": {"no_validation_semantic_diff_detected": False},
        },
    }

    localized = _localize(report)

    assert localized["stage"] == "validation_e2e_semantic_drift"


def test_validation_pair_summary_reports_exact_action_and_return_match():
    summary = _validation_pair_summary(
        {
            "mean_return": 1.5,
            "env_return_mean": 1.5,
            "env_return": 3.0,
            "env_len_mean": 2.0,
            "action_log": [0.1, 0.2, 0.3],
        },
        {
            "mean_return": 1.5,
            "env_return_mean": 1.5,
            "env_return": 3.0,
            "env_len_mean": 2.0,
            "action_log": [0.1, 0.2, 0.3],
        },
    )

    assert summary["mean_return_diff"] == 0.0
    assert summary["env_return_mean_diff"] == 0.0
    assert summary["env_return_diff"] == 0.0
    assert summary["env_len_mean_diff"] == 0.0
    assert summary["exact_action_log_match"] is True
    assert summary["max_abs_action_log_diff"] == 0.0
