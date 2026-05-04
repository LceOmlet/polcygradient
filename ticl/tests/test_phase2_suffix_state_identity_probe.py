import numpy as np

from ticl.analysis.phase2_suffix_state_identity_probe import (
    _compute_horizon_diagnostics,
    _compute_identity_metrics,
    _compute_identity_metrics_from_return_lists,
    _json_safe_value,
    _resolve_horizon_diagnostics_steps,
    _select_env_knob_snapshot,
    _validate_discount_gamma,
    build_arg_parser,
)


def test_compute_identity_metrics_matches_signal_noise_decomposition():
    returns = np.asarray(
        [
            [1.0, 1.0, 1.0],
            [3.0, 3.0, 3.0],
        ],
        dtype=np.float32,
    )
    metrics = _compute_identity_metrics(returns)
    assert metrics["noise"] == 0.0
    assert metrics["signal"] == 1.0
    assert metrics["identity_score"] == 1.0


def test_compute_identity_metrics_detects_within_state_noise():
    returns = np.asarray(
        [
            [0.0, 2.0],
            [4.0, 6.0],
        ],
        dtype=np.float32,
    )
    metrics = _compute_identity_metrics(returns)
    assert np.isclose(metrics["noise"], 1.0)
    assert np.isclose(metrics["signal"], 4.0)
    assert np.isclose(metrics["identity_score"], 0.8)


def test_partial_identity_metrics_match_dense_matrix_for_complete_data():
    returns = np.asarray(
        [
            [1.0, 3.0, 5.0],
            [2.0, 4.0, 6.0],
        ],
        dtype=np.float32,
    )
    dense_metrics = _compute_identity_metrics(returns)
    partial_metrics = _compute_identity_metrics_from_return_lists(
        [
            [1.0, 3.0, 5.0],
            [2.0, 4.0, 6.0],
        ]
    )
    for key in ("noise", "signal", "identity_score", "total_variance"):
        assert np.isclose(dense_metrics[key], partial_metrics[key])


def test_validate_discount_gamma_accepts_closed_interval():
    assert _validate_discount_gamma(0.0) == 0.0
    assert _validate_discount_gamma(0.99) == 0.99
    assert _validate_discount_gamma(1.0) == 1.0


def test_resolve_horizon_diagnostics_steps_accepts_powers_and_deduplicates():
    assert _resolve_horizon_diagnostics_steps("powers", remaining_horizon=9) == [1, 2, 4, 8, 9]
    assert _resolve_horizon_diagnostics_steps("1,2,2,4", remaining_horizon=8) == [1, 2, 4]
    assert _resolve_horizon_diagnostics_steps(None, remaining_horizon=8) == []


def test_compute_horizon_diagnostics_uses_discounted_prefix_returns():
    discounted_step_rewards = np.asarray(
        [
            [[1.0, 1.0, 0.0], [1.0, 3.0, 0.0]],
            [[3.0, 1.0, 0.0], [3.0, 3.0, 0.0]],
        ],
        dtype=np.float32,
    )
    diagnostics = _compute_horizon_diagnostics(
        discounted_step_rewards,
        horizon_steps=[1, 2],
        continuation_repeats=2,
    )
    assert diagnostics[0]["horizon_step"] == 1
    assert np.isclose(diagnostics[0]["signal"], 1.0)
    assert np.isclose(diagnostics[0]["noise"], 0.0)
    assert diagnostics[1]["horizon_step"] == 2
    assert np.isclose(diagnostics[1]["signal"], 1.0)
    assert np.isclose(diagnostics[1]["noise"], 1.0)
    assert np.isclose(diagnostics[1]["finite_repeat_signal_bias_estimate"], 0.5)


def test_identity_probe_parser_accepts_unit_gaussian_and_alpha_terminal_overrides():
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--from-scratch", "true",
            "--behavior-policy", "unit_gaussian",
            "--sampled-policy", "true",
            "--constrained-dim-sampling-total-budget-override", "72",
            "--state-dim-override", "57",
            "--action-dim-override", "28",
            "--noise-dim-override", "4",
            "--zero-pad-dim-override", "11",
            "--alpha-override", "0.2",
            "--reference-state-inertia-enabled-override", "true",
            "--terminal-reset-enabled-override", "false",
            "--terminal-reset-count-target-override", "1.5",
            "--fixed-frozen-h-json", "/tmp/frozen_h.json",
            "--save-full-frozen-h-json", "/tmp/frozen_h_out.json",
            "--discount-gamma", "0.97",
            "--horizon-diagnostics-steps", "1,2,4",
        ]
    )

    assert args.from_scratch is True
    assert args.behavior_policy == "unit_gaussian"
    assert args.sampled_policy is True
    assert args.constrained_dim_sampling_total_budget_override == 72
    assert args.state_dim_override == 57
    assert args.action_dim_override == 28
    assert args.noise_dim_override == 4
    assert args.zero_pad_dim_override == 11
    assert args.alpha_override == 0.2
    assert args.reference_state_inertia_enabled_override is True
    assert args.terminal_reset_enabled_override is False
    assert args.terminal_reset_count_target_override == 1.5
    assert args.fixed_frozen_h_json == "/tmp/frozen_h.json"
    assert args.save_full_frozen_h_json == "/tmp/frozen_h_out.json"
    assert args.discount_gamma == 0.97
    assert args.horizon_diagnostics_steps == "1,2,4"


def test_probe_snapshot_and_json_safe_helpers_keep_init_knobs_serializable():
    payload = {
        "prior_mlp_activations": "tanh",
        "init_std": np.float64(0.05),
        "noise_std": np.float32(0.01),
        "terminal_reset_enabled": np.bool_(True),
    }
    safe_payload = _json_safe_value(payload)
    assert safe_payload["prior_mlp_activations"] == "tanh"
    assert np.isclose(safe_payload["init_std"], 0.05)
    assert np.isclose(safe_payload["noise_std"], 0.01)
    assert safe_payload["terminal_reset_enabled"] is True

    snapshot = _select_env_knob_snapshot(payload)
    assert snapshot["prior_mlp_activations"] == "tanh"
    assert np.isclose(snapshot["init_std"], 0.05)
    assert np.isclose(snapshot["noise_std"], 0.01)
