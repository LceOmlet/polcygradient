from ticl.analysis.phase2_alpha_temporal_control_probe import _parse_alpha_list, build_arg_parser


def test_parse_alpha_list_trims_and_preserves_order():
    values = _parse_alpha_list(" 1.0, 0.2 ,0.05 ")
    assert values == [1.0, 0.2, 0.05]


def test_alpha_temporal_control_probe_parser_accepts_overrides():
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--from-scratch", "true",
            "--rollout-count", "1024",
            "--constrained-dim-sampling-total-budget-override", "400",
            "--state-dim-override", "300",
            "--fixed-frozen-h-json", "/tmp/frozen_h.json",
            "--prior-mlp-activations-override", "sin",
            "--init-std-override", "0.05",
            "--noise-std-override", "0.01",
            "--scm-standard-linear-init-enabled-override", "false",
            "--alpha-list", "1.0,0.2,0.05",
            "--reference-state-inertia-enabled-override", "true",
            "--terminal-reset-enabled-override", "true",
            "--terminal-reset-count-target-override", "3.0",
            "--action-delta", "0.2",
        ]
    )

    assert args.from_scratch is True
    assert args.rollout_count == 1024
    assert args.constrained_dim_sampling_total_budget_override == 400
    assert args.state_dim_override == 300
    assert args.fixed_frozen_h_json == "/tmp/frozen_h.json"
    assert args.prior_mlp_activations_override == "sin"
    assert args.init_std_override == 0.05
    assert args.noise_std_override == 0.01
    assert args.scm_standard_linear_init_enabled_override is False
    assert args.alpha_list == "1.0,0.2,0.05"
    assert args.reference_state_inertia_enabled_override is True
    assert args.terminal_reset_enabled_override is True
    assert args.terminal_reset_count_target_override == 3.0
    assert args.action_delta == 0.2
