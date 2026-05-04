from ticl.analysis import phase3_pair2_episode_complete_train_update_ab_compare_pack as compare_mod


def test_build_pair2_episode_complete_train_update_ab_compare_pack_detects_trajectory_change(monkeypatch):
    selector = {
        "target_outer_batch_idx": 13,
        "target_env_index": 12,
        "target_objective_episode_index": 4,
        "target_objective_position_start": 0,
        "target_objective_position_end": 28,
    }
    baseline = {
        "config": {
            "checkpoint_path": "/tmp/checkpoint.cpkt",
            "train_suite_path": "/tmp/pair2/train_suite.pt",
            "heldout_suite_path": "/tmp/pair2/heldout_suite.pt",
            "n_samples": 256,
            "eval_n_samples": 256,
            "single_eval_pos": 64,
            "train_profile": "phase2_shared_backbone_contract",
            "strict_fixed_env_mode": True,
            "deterministic_actor_sampling": True,
            "deterministic_batch_plan": True,
            "strict_native_rollout": True,
            "rollout_backend": "serial",
            "actor_objective_runtime_current_suite_name": None,
            "actor_objective_mode_override": None,
            "ppo_actor_baseline_mode": "learned",
            "ppo_actor_gae_space": "normalized",
            "ppo_normalize_advantage": False,
            "ppo_vf_coef": 0.5,
            "ppo_reset_env_state_at_sep": True,
            "ppo_separate_value_backbone": False,
            "ppo_restore_validation_policy_head_state": True,
            "runtime_normalized_q_value_weight_override": None,
            "runtime_next_state_flow_matching_weight_override": None,
        },
        "train_suite_summary": {"fingerprint": "train_fp"},
        "heldout_suite_summary": {"fingerprint": "heldout_fp"},
        "zero_control": {
            "train": {"full_return_mean": 0.0, "suffix_return_mean": 0.0, "full_return_gap": 0.0, "suffix_return_gap": 0.0},
            "heldout": {"full_return_mean": 0.0, "suffix_return_mean": 0.0, "full_return_gap": 0.0, "suffix_return_gap": 0.0},
        },
        "pre": {
            "train": {"full_return_mean": 0.0, "suffix_return_mean": 0.0, "full_return_gap": 0.0, "suffix_return_gap": 0.0},
            "heldout": {"full_return_mean": 0.0, "suffix_return_mean": 0.0, "full_return_gap": 0.0, "suffix_return_gap": 0.0},
        },
        "comparison": {
            "pre_train_vs_zero": {"full_return_gap": 1.0, "suffix_return_gap": 2.0},
            "pre_heldout_vs_zero": {"full_return_gap": 3.0, "suffix_return_gap": 4.0},
            "post_train_vs_zero": {"full_return_gap": 5.0, "suffix_return_gap": 6.0},
            "post_heldout_vs_zero": {"full_return_gap": 7.0, "suffix_return_gap": 8.0},
            "train_full_return_delta": 0.5,
            "train_suffix_return_delta": 0.25,
            "heldout_full_return_delta": -0.5,
            "heldout_suffix_return_delta": -0.25,
        },
        "train_history": [
            {
                "outer_epoch": 1,
                "critic_raw_corr": 0.1,
                "critic_explained_variance_raw": 0.2,
                "critic_objective_total": 16,
                "actor_baseline_mode": "learned",
                "switches_to_learned_next": False,
            }
        ],
        "train_outer_batch_snapshot": {
            "requested_path": "/tmp/baseline_snapshot.json",
            "selector": dict(selector),
            "written": True,
            "written_path": "/tmp/baseline_snapshot.json",
            "snapshot_target_match_count": 29,
        },
    }
    override = {
        **baseline,
        "config": {
            **baseline["config"],
            "actor_objective_mode_override": compare_mod.PAIR2_EPISODE_COMPLETE_BRANCH_SPECIFIC_MODE,
            "actor_objective_runtime_current_suite_name": "pair2",
        },
        "comparison": {
            "pre_train_vs_zero": {"full_return_gap": 1.0, "suffix_return_gap": 2.0},
            "pre_heldout_vs_zero": {"full_return_gap": 3.0, "suffix_return_gap": 4.0},
            "post_train_vs_zero": {"full_return_gap": 5.5, "suffix_return_gap": 6.75},
            "post_heldout_vs_zero": {"full_return_gap": 7.0, "suffix_return_gap": 8.0},
            "train_full_return_delta": 0.75,
            "train_suffix_return_delta": 0.5,
            "heldout_full_return_delta": -0.5,
            "heldout_suffix_return_delta": -0.25,
        },
        "train_history": [
            {
                "outer_epoch": 1,
                "critic_raw_corr": 0.2,
                "critic_explained_variance_raw": 0.35,
                "critic_objective_total": 16,
                "actor_baseline_mode": "learned",
                "switches_to_learned_next": True,
            }
        ],
        "train_outer_batch_snapshot": {
            "requested_path": "/tmp/override_snapshot.json",
            "selector": dict(selector),
            "written": True,
            "written_path": "/tmp/override_snapshot.json",
            "snapshot_target_match_count": 29,
        },
    }

    monkeypatch.setattr(
        compare_mod,
        "_load_pair2_episode_complete_locked_selector",
        lambda: dict(selector),
    )
    monkeypatch.setattr(
        compare_mod,
        "_run_audit",
        lambda **kwargs: baseline if kwargs["actor_objective_mode_override"] is None else override,
    )
    monkeypatch.setattr(
        compare_mod,
        "_load_pair2_preflight",
        lambda: {
            "preflight_passed": True,
            "contract_args": {},
            "contract_diff": {"allowed_diff": {}, "unexpected_diff": {}},
        },
    )

    report = compare_mod.build_pair2_episode_complete_train_update_ab_compare_pack()

    assert report["target_control"]["candidate_small_control_target"] == "env12_selector_matched_objective_episode"
    assert report["target_control"]["locked_selector"] == selector
    assert report["target_control"]["runtime_mode_activated"] is True
    assert report["conclusions"]["target_selector_present_in_both_arms"] is True
    assert report["conclusions"]["target_side_compare_is_efficacy_eligible"] is True
    assert report["conclusions"]["target_side_train_update_changes_trajectory"] is True
    assert report["comparison"]["train_history_delta"]["histories_identical"] is False
    assert abs(report["comparison"]["post_train_vs_zero_gap_delta"]["full_return_gap"] - 0.5) <= 1e-12
    assert abs(report["comparison"]["post_train_vs_zero_gap_delta"]["suffix_return_gap"] - 0.75) <= 1e-12


def test_build_pair2_episode_complete_train_update_ab_compare_pack_rejects_zero_target_match_snapshot(monkeypatch):
    selector = {
        "target_outer_batch_idx": 13,
        "target_env_index": 12,
        "target_objective_episode_index": 4,
        "target_objective_position_start": 0,
        "target_objective_position_end": 28,
    }
    baseline = {
        "config": {
            "checkpoint_path": "/tmp/checkpoint.cpkt",
            "train_suite_path": "/tmp/pair2/train_suite.pt",
            "heldout_suite_path": "/tmp/pair2/heldout_suite.pt",
            "n_samples": 256,
            "eval_n_samples": 256,
            "single_eval_pos": 64,
            "train_profile": "phase2_shared_backbone_contract",
            "strict_fixed_env_mode": True,
            "deterministic_actor_sampling": True,
            "deterministic_batch_plan": True,
            "strict_native_rollout": True,
            "rollout_backend": "serial",
            "actor_objective_runtime_current_suite_name": None,
            "actor_objective_mode_override": None,
            "ppo_actor_baseline_mode": "learned",
            "ppo_actor_gae_space": "normalized",
            "ppo_normalize_advantage": False,
            "ppo_vf_coef": 0.5,
            "ppo_reset_env_state_at_sep": True,
            "ppo_separate_value_backbone": False,
            "ppo_restore_validation_policy_head_state": True,
            "runtime_normalized_q_value_weight_override": None,
            "runtime_next_state_flow_matching_weight_override": None,
        },
        "train_suite_summary": {"fingerprint": "train_fp"},
        "heldout_suite_summary": {"fingerprint": "heldout_fp"},
        "zero_control": {
            "train": {"full_return_mean": 0.0, "suffix_return_mean": 0.0, "full_return_gap": 0.0, "suffix_return_gap": 0.0},
            "heldout": {"full_return_mean": 0.0, "suffix_return_mean": 0.0, "full_return_gap": 0.0, "suffix_return_gap": 0.0},
        },
        "pre": {
            "train": {"full_return_mean": 0.0, "suffix_return_mean": 0.0, "full_return_gap": 0.0, "suffix_return_gap": 0.0},
            "heldout": {"full_return_mean": 0.0, "suffix_return_mean": 0.0, "full_return_gap": 0.0, "suffix_return_gap": 0.0},
        },
        "comparison": {
            "pre_train_vs_zero": {"full_return_gap": 1.0, "suffix_return_gap": 2.0},
            "pre_heldout_vs_zero": {"full_return_gap": 3.0, "suffix_return_gap": 4.0},
            "post_train_vs_zero": {"full_return_gap": 5.0, "suffix_return_gap": 6.0},
            "post_heldout_vs_zero": {"full_return_gap": 7.0, "suffix_return_gap": 8.0},
            "train_full_return_delta": 0.5,
            "train_suffix_return_delta": 0.25,
            "heldout_full_return_delta": -0.5,
            "heldout_suffix_return_delta": -0.25,
        },
        "train_history": [
            {
                "outer_epoch": 1,
                "critic_raw_corr": 0.1,
                "critic_explained_variance_raw": 0.2,
                "critic_objective_total": 16,
                "actor_baseline_mode": "learned",
                "switches_to_learned_next": False,
            }
        ],
        "train_outer_batch_snapshot": {
            "requested_path": "/tmp/baseline_snapshot.json",
            "selector": dict(selector),
            "written": True,
            "written_path": "/tmp/baseline_snapshot.json",
            "snapshot_target_match_count": 0,
        },
    }
    override = {
        **baseline,
        "config": {
            **baseline["config"],
            "actor_objective_mode_override": compare_mod.PAIR2_EPISODE_COMPLETE_BRANCH_SPECIFIC_MODE,
            "actor_objective_runtime_current_suite_name": "pair2",
        },
        "train_outer_batch_snapshot": {
            "requested_path": "/tmp/override_snapshot.json",
            "selector": dict(selector),
            "written": True,
            "written_path": "/tmp/override_snapshot.json",
            "snapshot_target_match_count": 0,
        },
    }

    monkeypatch.setattr(
        compare_mod,
        "_load_pair2_episode_complete_locked_selector",
        lambda: dict(selector),
    )
    monkeypatch.setattr(
        compare_mod,
        "_run_audit",
        lambda **kwargs: baseline if kwargs["actor_objective_mode_override"] is None else override,
    )
    monkeypatch.setattr(
        compare_mod,
        "_load_pair2_preflight",
        lambda: {
            "preflight_passed": True,
            "contract_args": {},
            "contract_diff": {"allowed_diff": {}, "unexpected_diff": {}},
        },
    )

    try:
        compare_mod.build_pair2_episode_complete_train_update_ab_compare_pack()
    except RuntimeError as exc:
        assert "matched no objective tokens" in str(exc)
    else:
        raise AssertionError("expected zero target-match snapshot to raise")
