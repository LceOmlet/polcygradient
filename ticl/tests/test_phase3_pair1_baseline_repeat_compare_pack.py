from ticl.analysis import phase3_pair1_baseline_repeat_compare_pack as compare_mod


def test_build_pair1_baseline_repeat_compare_pack_detects_same_arm_drift(monkeypatch):
    primary = {
        "config": {
            "checkpoint_path": "/tmp/checkpoint.cpkt",
            "train_suite_path": "/tmp/pair1/train_suite.pt",
            "heldout_suite_path": "/tmp/pair1/heldout_suite.pt",
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
            "post_train_vs_zero": {"full_return_gap": 5.0, "suffix_return_gap": 6.0},
            "train_full_return_delta": 0.5,
            "train_suffix_return_delta": 0.25,
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
        "heldout_eval_skipped": True,
        "post_policy_bundle": {"saved": True, "save_path": "/tmp/primary.pt", "resumed": False, "source_path": None},
    }
    repeat = {
        **primary,
        "comparison": {
            "pre_train_vs_zero": {"full_return_gap": 1.0, "suffix_return_gap": 2.0},
            "post_train_vs_zero": {"full_return_gap": 5.3, "suffix_return_gap": 6.1},
            "train_full_return_delta": 0.8,
            "train_suffix_return_delta": 0.35,
        },
        "train_history": [
            {
                "outer_epoch": 1,
                "critic_raw_corr": 0.11,
                "critic_explained_variance_raw": 0.25,
                "critic_objective_total": 16,
                "actor_baseline_mode": "learned",
                "switches_to_learned_next": False,
            }
        ],
        "post_policy_bundle": {"saved": True, "save_path": "/tmp/repeat.pt", "resumed": False, "source_path": None},
    }

    calls = []

    def _fake_run_audit(
        *,
        actor_objective_mode_override,
        restore_validation_policy_head_state=False,
        reuse_zero_control_json=None,
        reuse_pre_policy_json=None,
        save_post_policy_bundle_path=None,
        resume_post_policy_bundle_path=None,
        eval_n_samples=None,
    ):
        calls.append(
            {
                "actor_objective_mode_override": actor_objective_mode_override,
                "restore_validation_policy_head_state": restore_validation_policy_head_state,
                "reuse_zero_control_json": reuse_zero_control_json,
                "reuse_pre_policy_json": reuse_pre_policy_json,
                "save_post_policy_bundle_path": save_post_policy_bundle_path,
                "resume_post_policy_bundle_path": resume_post_policy_bundle_path,
                "eval_n_samples": eval_n_samples,
            }
        )
        return primary if len(calls) == 1 else repeat

    monkeypatch.setattr(compare_mod, "_run_audit", _fake_run_audit)
    monkeypatch.setattr(
        compare_mod,
        "_load_pair1_preflight",
        lambda: {
            "preflight_passed": True,
            "contract_args": {},
            "contract_diff": {"allowed_diff": {}, "unexpected_diff": {}},
        },
    )

    report = compare_mod.build_pair1_baseline_repeat_compare_pack()

    assert calls[0]["actor_objective_mode_override"] is None
    assert calls[0]["restore_validation_policy_head_state"] is True
    assert calls[0]["reuse_zero_control_json"] is None
    assert calls[0]["reuse_pre_policy_json"] is None
    assert calls[0]["save_post_policy_bundle_path"] == compare_mod.PAIR1_BASELINE_REPEAT_PRIMARY_POST_POLICY_BUNDLE_PATH
    assert calls[1]["actor_objective_mode_override"] is None
    assert calls[1]["restore_validation_policy_head_state"] is True
    assert calls[1]["reuse_zero_control_json"] is not None
    assert calls[1]["reuse_pre_policy_json"] is not None
    assert calls[1]["save_post_policy_bundle_path"] == compare_mod.PAIR1_BASELINE_REPEAT_REPEAT_POST_POLICY_BUNDLE_PATH
    assert report["conclusions"]["pre_metrics_unchanged"] is True
    assert report["conclusions"]["same_arm_repeat_is_stable"] is False
    assert report["conclusions"]["same_contract_many_env_train_nondeterminism_observed"] is True
    assert abs(report["comparison"]["post_train_vs_zero_gap_delta"]["full_return_gap"] - 0.3) <= 1e-12
    assert abs(report["comparison"]["train_history_delta"]["epoch_deltas"][0]["critic_raw_corr_delta"] - 0.01) <= 1e-12


def test_build_pair1_baseline_repeat_compare_pack_rejects_override_leak(monkeypatch):
    payload = {
        "config": {
            "checkpoint_path": "/tmp/checkpoint.cpkt",
            "train_suite_path": "/tmp/pair1/train_suite.pt",
            "heldout_suite_path": "/tmp/pair1/heldout_suite.pt",
            "n_samples": 256,
            "eval_n_samples": 256,
            "single_eval_pos": 64,
            "train_profile": "phase2_shared_backbone_contract",
            "strict_fixed_env_mode": True,
            "deterministic_actor_sampling": True,
            "deterministic_batch_plan": True,
            "strict_native_rollout": True,
            "rollout_backend": "serial",
            "actor_objective_runtime_current_suite_name": "pair1",
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
        "zero_control": {"train": {}, "heldout": {}},
        "pre": {"train": {}, "heldout": {}},
        "comparison": {
            "pre_train_vs_zero": {"full_return_gap": 1.0, "suffix_return_gap": 2.0},
            "post_train_vs_zero": {"full_return_gap": 5.0, "suffix_return_gap": 6.0},
            "train_full_return_delta": 0.5,
            "train_suffix_return_delta": 0.25,
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
        "heldout_eval_skipped": True,
        "post_policy_bundle": {"saved": True, "save_path": "/tmp/x.pt", "resumed": False, "source_path": None},
    }

    def _fake_run_audit(
        *,
        actor_objective_mode_override,
        restore_validation_policy_head_state=False,
        reuse_zero_control_json=None,
        reuse_pre_policy_json=None,
        save_post_policy_bundle_path=None,
        resume_post_policy_bundle_path=None,
        eval_n_samples=None,
    ):
        return payload

    monkeypatch.setattr(compare_mod, "_run_audit", _fake_run_audit)
    monkeypatch.setattr(
        compare_mod,
        "_load_pair1_preflight",
        lambda: {
            "preflight_passed": True,
            "contract_args": {},
            "contract_diff": {"allowed_diff": {}, "unexpected_diff": {}},
        },
    )

    try:
        compare_mod.build_pair1_baseline_repeat_compare_pack()
    except ValueError as exc:
        assert "runtime-scoped objective mode" in str(exc)
    else:
        raise AssertionError("expected repeat compare to reject runtime-scoped baseline")
