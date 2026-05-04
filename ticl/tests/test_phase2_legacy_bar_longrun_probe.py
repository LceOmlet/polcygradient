from ticl.analysis.phase2_legacy_bar_longrun_probe import build_arg_parser


def test_phase2_legacy_bar_longrun_parser_defaults_match_phase2_contract():
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "/tmp/mock_checkpoint.cpkt",
        ]
    )

    assert args.checkpoint_path == "/tmp/mock_checkpoint.cpkt"
    assert args.from_scratch is False
    assert args.from_scratch_model_type == "rlpfn"
    assert args.outer_epochs == 8
    assert args.ppo_reset_env_state_at_sep is True
    assert args.strict_native_rollout is False
    assert args.ppo_separate_value_backbone is False
    assert args.value_head_impl == "legacy_bar"
    assert args.value_path_adapter_impl == "none"
    assert args.value_head_mlp_hidden_dim == 256
    assert args.space_contract == "normalized"
    assert args.vf_coef_override is None
    assert args.policy_loss_coef_override is None
    assert args.behavior_policy == "learned"
    assert args.modes == ["learned", "zero"]
    assert args.output_json.endswith("phase2_legacy_bar_longrun_probe.json")


def test_phase2_legacy_bar_longrun_parser_accepts_from_scratch_without_checkpoint():
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--from-scratch", "true",
            "--from-scratch-model-type", "rlpfn",
        ]
    )

    assert args.checkpoint_path is None
    assert args.from_scratch is True
    assert args.from_scratch_model_type == "rlpfn"
    assert args.strict_native_rollout is False
    assert args.ppo_separate_value_backbone is False
    assert args.value_head_impl == "legacy_bar"
    assert args.value_path_adapter_impl == "none"
    assert args.value_head_mlp_hidden_dim == 256
    assert args.space_contract == "normalized"
    assert args.behavior_policy == "learned"


def test_phase2_legacy_bar_longrun_parser_accepts_loss_overrides():
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--from-scratch", "true",
            "--policy-loss-coef-override", "0.0",
            "--vf-coef-override", "1.0",
        ]
    )

    assert args.from_scratch is True
    assert args.policy_loss_coef_override == 0.0
    assert args.vf_coef_override == 1.0


def test_phase2_legacy_bar_longrun_parser_accepts_unit_gaussian_behavior_policy():
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--from-scratch", "true",
            "--behavior-policy", "unit_gaussian",
        ]
    )

    assert args.from_scratch is True
    assert args.behavior_policy == "unit_gaussian"


def test_phase2_legacy_bar_longrun_parser_accepts_separate_value_backbone():
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--from-scratch", "true",
            "--ppo-separate-value-backbone", "true",
        ]
    )

    assert args.from_scratch is True
    assert args.ppo_separate_value_backbone is True
