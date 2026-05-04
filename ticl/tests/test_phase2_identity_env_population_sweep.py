import numpy as np

from ticl.analysis.phase2_identity_env_population_sweep import (
    _horizon_features,
    _return_features,
    _snapshot_features,
    _stable_report_fingerprint,
    build_arg_parser,
)


def test_population_sweep_parser_accepts_identity_contract_knobs():
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--output-dir",
            "/tmp/pop",
            "--env-count",
            "3",
            "--reference-state-count",
            "8",
            "--continuation-repeats",
            "9",
            "--batch-size",
            "128",
            "--alpha-override",
            "0.5",
            "--reference-state-inertia-enabled-override",
            "true",
            "--terminal-reset-enabled-override",
            "none",
            "--horizon-diagnostics-steps",
            "2",
            "--save-full-reports",
            "false",
        ]
    )

    assert args.output_dir == "/tmp/pop"
    assert args.env_count == 3
    assert args.reference_state_count == 8
    assert args.continuation_repeats == 9
    assert args.batch_size == 128
    assert args.alpha_override == 0.5
    assert args.reference_state_inertia_enabled_override is True
    assert args.terminal_reset_enabled_override is None
    assert args.horizon_diagnostics_steps == "2"
    assert args.save_full_reports is False


def test_return_features_extract_between_and_within_state_structure():
    report = {
        "aggregate": {
            "identity_score": 0.8,
            "signal": 4.0,
            "noise": 1.0,
            "total_variance": 5.0,
            "mean_of_state_means": 3.0,
            "std_of_state_means": 2.0,
            "mean_within_state_std": 1.0,
        },
        "anchors": [
            {
                "continuation_return_mean": 1.0,
                "continuation_return_std": 0.5,
                "continuation_returns": [0.5, 1.5],
            },
            {
                "continuation_return_mean": 5.0,
                "continuation_return_std": 0.5,
                "continuation_returns": [4.5, 5.5],
            },
        ],
    }

    features = _return_features(report)

    assert np.isclose(features["identity_score"], 0.8)
    assert np.isclose(features["signal_to_noise"], 4.0)
    assert np.isclose(features["state_return_mean_q50"], 3.0)
    assert np.isclose(features["state_return_std_mean"], 0.5)
    assert np.isclose(features["all_return_q50"], 3.0)


def test_horizon_features_extracts_gk_diagnostics():
    report = {
        "horizon_diagnostics": [
            {
                "horizon_step": 2,
                "identity_score": 0.2,
                "bias_corrected_identity_score": 0.15,
                "signal": 1.0,
                "bias_corrected_signal": 0.75,
                "noise": 4.0,
                "finite_repeat_signal_bias_estimate": 0.25,
            }
        ]
    }
    features = _horizon_features(report)
    assert np.isclose(features["horizon_2_identity_score"], 0.2)
    assert np.isclose(features["horizon_2_bias_corrected_identity_score"], 0.15)
    assert np.isclose(features["horizon_2_bias_corrected_signal"], 0.75)


def test_snapshot_features_add_dimension_ratios_and_activation_name():
    snapshot = {
        "family": "scm",
        "state_dim": 300,
        "obs_dim": 150,
        "action_dim": 20,
        "noise_dim": 5,
        "zero_pad_dim": 75,
        "alpha": 1.0,
        "reference_state_inertia_enabled": True,
        "terminal_reset_enabled": True,
        "terminal_reset_count_target": 8.0,
        "prior_mlp_activations": "<class 'torch.nn.modules.activation.Tanh'>",
        "num_layers": 3,
        "prior_mlp_hidden_dim": 64,
        "scm_standard_linear_init_enabled": False,
        "init_std": 0.05,
        "noise_std": 0.01,
    }
    sampled = {
        "state_dim": 300,
        "obs_dim": 150,
        "action_dim": 20,
        "noise_dim": 5,
        "zero_pad_dim": 75,
        "state_noise_std": 0.0,
        "reward_scale": 1.0,
        "ctrl_reward_enabled": False,
        "survival_reward_enabled": True,
        "state_highway_enabled": False,
        "state_highway_lambda": 0.0,
    }

    features = _snapshot_features(snapshot, sampled, n_steps=256)

    assert features["activation"] == "tanh"
    assert np.isclose(features["state_dim_fraction"], 0.75)
    assert np.isclose(features["obs_to_state_ratio"], 0.5)
    assert np.isclose(features["terminal_reset_expected_prob_per_step"], 8.0 / 256.0)
    assert features["scm_standard_linear_init_enabled"] == 0
    assert features["survival_reward_enabled"] == 1


def test_stable_report_fingerprint_ignores_output_paths():
    base = {
        "aggregate": {"identity_score": 0.1},
        "anchors": [{"continuation_returns": [1.0, 2.0]}],
        "effective_frozen_h_snapshot": {"state_dim": 4},
        "sampled_env_snapshot": {"state_dim": 4},
        "fixed_env_contract": {"frozen_h_seed": 123},
        "frozen_h_seed": 123,
        "train_env_seed": 2020,
        "train_rollout_seed": 4040,
        "single_eval_pos": 64,
        "anchor_step": 128,
        "n_steps": 256,
        "reference_state_count": 1,
        "continuation_repeats": 2,
        "reference_rollout_seed_start": 5000,
        "continuation_rollout_seed_start": 10000,
        "discount_gamma": 0.99,
        "behavior_policy": "unit_gaussian",
        "sampled_policy": True,
        "horizon_diagnostics_steps": [2],
        "horizon_diagnostics": [{"horizon_step": 2, "identity_score": 0.2}],
        "save_full_frozen_h_json": "/tmp/a.json",
    }
    changed_path = dict(base)
    changed_path["save_full_frozen_h_json"] = "/tmp/b.json"

    assert _stable_report_fingerprint(base) == _stable_report_fingerprint(changed_path)
