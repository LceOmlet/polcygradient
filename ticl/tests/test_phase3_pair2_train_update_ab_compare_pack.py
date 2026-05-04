from ticl.analysis import phase3_pair2_train_update_ab_compare_pack as compare_mod


def test_build_pair2_train_update_ab_compare_pack_detects_target_side_trajectory_change(monkeypatch):
    selector = {
        "target_outer_batch_idx": compare_mod.PAIR2_SNAPSHOT_TARGET_OUTER_BATCH_IDX,
        "target_env_index": compare_mod.PAIR2_SNAPSHOT_TARGET_ENV_INDEX,
        "target_objective_episode_index": compare_mod.PAIR2_SNAPSHOT_TARGET_OBJECTIVE_EPISODE_INDEX,
        "target_objective_position_start": compare_mod.PAIR2_SNAPSHOT_TARGET_OBJECTIVE_POSITION_START,
        "target_objective_position_end": compare_mod.PAIR2_SNAPSHOT_TARGET_OBJECTIVE_POSITION_END,
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
            "snapshot_target_match_count": 4,
        },
    }
    override = {
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
            "actor_objective_runtime_current_suite_name": "pair2",
            "actor_objective_mode_override": compare_mod.PAIR2_BRANCH_SPECIFIC_MODE,
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
            "post_train_vs_zero": {"full_return_gap": 5.75, "suffix_return_gap": 6.5},
            "post_heldout_vs_zero": {"full_return_gap": 7.5, "suffix_return_gap": 8.5},
            "train_full_return_delta": 0.75,
            "train_suffix_return_delta": 0.5,
            "heldout_full_return_delta": -0.25,
            "heldout_suffix_return_delta": 0.25,
        },
        "train_history": [
            {
                "outer_epoch": 1,
                "critic_raw_corr": 0.15,
                "critic_explained_variance_raw": 0.27,
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
            "snapshot_target_match_count": 4,
        },
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
        train_outer_batch_snapshot_json=None,
        train_outer_batch_snapshot_target_outer_batch_idx=None,
        train_outer_batch_snapshot_target_env_index=None,
        train_outer_batch_snapshot_target_objective_episode_index=None,
        train_outer_batch_snapshot_target_objective_position_start=None,
        train_outer_batch_snapshot_target_objective_position_end=None,
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
                "train_outer_batch_snapshot_json": train_outer_batch_snapshot_json,
                "train_outer_batch_snapshot_target_outer_batch_idx": train_outer_batch_snapshot_target_outer_batch_idx,
                "train_outer_batch_snapshot_target_env_index": train_outer_batch_snapshot_target_env_index,
                "train_outer_batch_snapshot_target_objective_episode_index": train_outer_batch_snapshot_target_objective_episode_index,
                "train_outer_batch_snapshot_target_objective_position_start": train_outer_batch_snapshot_target_objective_position_start,
                "train_outer_batch_snapshot_target_objective_position_end": train_outer_batch_snapshot_target_objective_position_end,
            }
        )
        return baseline if actor_objective_mode_override is None else override

    monkeypatch.setattr(compare_mod, "_run_audit", _fake_run_audit)
    monkeypatch.setattr(
        compare_mod,
        "_load_pair2_preflight",
        lambda: {
            "preflight_passed": True,
            "contract_args": {},
            "contract_diff": {"allowed_diff": {}, "unexpected_diff": {}},
        },
    )
    monkeypatch.setattr(compare_mod, "_load_pair2_guarded_selector_reidentified", lambda: dict(selector))

    report = compare_mod.build_pair2_train_update_ab_compare_pack()

    assert calls[0]["actor_objective_mode_override"] is None
    assert calls[0]["restore_validation_policy_head_state"] is True
    assert calls[0]["reuse_zero_control_json"] is None
    assert calls[0]["reuse_pre_policy_json"] is None
    assert calls[0]["save_post_policy_bundle_path"] == compare_mod.PAIR2_BASELINE_POST_POLICY_BUNDLE_PATH
    assert calls[0]["resume_post_policy_bundle_path"] is None
    assert calls[0]["eval_n_samples"] == compare_mod.PAIR2_TRAIN_UPDATE_EVAL_NSAMPLES
    assert (
        calls[0]["train_outer_batch_snapshot_json"]
        == compare_mod.PAIR2_BASELINE_TRAIN_OUTER_BATCH_SNAPSHOT_JSON
    )
    assert (
        calls[0]["train_outer_batch_snapshot_target_outer_batch_idx"]
        == compare_mod.PAIR2_SNAPSHOT_TARGET_OUTER_BATCH_IDX
    )
    assert calls[0]["train_outer_batch_snapshot_target_env_index"] == compare_mod.PAIR2_SNAPSHOT_TARGET_ENV_INDEX
    assert (
        calls[0]["train_outer_batch_snapshot_target_objective_episode_index"]
        == compare_mod.PAIR2_SNAPSHOT_TARGET_OBJECTIVE_EPISODE_INDEX
    )
    assert (
        calls[0]["train_outer_batch_snapshot_target_objective_position_start"]
        == compare_mod.PAIR2_SNAPSHOT_TARGET_OBJECTIVE_POSITION_START
    )
    assert (
        calls[0]["train_outer_batch_snapshot_target_objective_position_end"]
        == compare_mod.PAIR2_SNAPSHOT_TARGET_OBJECTIVE_POSITION_END
    )
    assert calls[1]["actor_objective_mode_override"] == compare_mod.PAIR2_BRANCH_SPECIFIC_MODE
    assert calls[1]["restore_validation_policy_head_state"] is True
    assert calls[1]["reuse_zero_control_json"] is not None
    assert calls[1]["reuse_pre_policy_json"] is not None
    assert calls[1]["save_post_policy_bundle_path"] == compare_mod.PAIR2_OVERRIDE_POST_POLICY_BUNDLE_PATH
    assert calls[1]["resume_post_policy_bundle_path"] is None
    assert calls[1]["eval_n_samples"] == compare_mod.PAIR2_TRAIN_UPDATE_EVAL_NSAMPLES
    assert (
        calls[1]["train_outer_batch_snapshot_json"]
        == compare_mod.PAIR2_OVERRIDE_TRAIN_OUTER_BATCH_SNAPSHOT_JSON
    )
    assert (
        calls[1]["train_outer_batch_snapshot_target_outer_batch_idx"]
        == compare_mod.PAIR2_SNAPSHOT_TARGET_OUTER_BATCH_IDX
    )
    assert calls[1]["train_outer_batch_snapshot_target_env_index"] == compare_mod.PAIR2_SNAPSHOT_TARGET_ENV_INDEX
    assert (
        calls[1]["train_outer_batch_snapshot_target_objective_episode_index"]
        == compare_mod.PAIR2_SNAPSHOT_TARGET_OBJECTIVE_EPISODE_INDEX
    )
    assert (
        calls[1]["train_outer_batch_snapshot_target_objective_position_start"]
        == compare_mod.PAIR2_SNAPSHOT_TARGET_OBJECTIVE_POSITION_START
    )
    assert (
        calls[1]["train_outer_batch_snapshot_target_objective_position_end"]
        == compare_mod.PAIR2_SNAPSHOT_TARGET_OBJECTIVE_POSITION_END
    )
    assert report["target_control"]["runtime_mode_activated"] is True
    assert report["conclusions"]["target_side_train_update_changes_trajectory"] is True
    assert report["conclusions"]["pre_metrics_unchanged"] is True
    assert report["conclusions"]["target_selector_present_in_both_arms"] is True
    assert report["conclusions"]["target_side_compare_is_efficacy_eligible"] is True
    assert report["comparison"]["heldout_eval_skipped"] is True
    assert report["post_policy_bundles"]["baseline_requested_path"] == compare_mod.PAIR2_BASELINE_POST_POLICY_BUNDLE_PATH
    assert report["post_policy_bundles"]["override_requested_path"] == compare_mod.PAIR2_OVERRIDE_POST_POLICY_BUNDLE_PATH
    assert report["comparison"]["train_history_delta"]["histories_identical"] is False
    assert abs(report["comparison"]["train_history_delta"]["epoch_deltas"][0]["critic_raw_corr_delta"] - 0.05) <= 1e-12
    assert abs(report["comparison"]["post_train_vs_zero_gap_delta"]["full_return_gap"] - 0.75) <= 1e-12
    assert abs(report["comparison"]["post_train_vs_zero_gap_delta"]["suffix_return_gap"] - 0.5) <= 1e-12


def test_build_pair2_train_update_ab_compare_pack_rejects_runtime_scope_mismatch(monkeypatch):
    selector = {
        "target_outer_batch_idx": compare_mod.PAIR2_SNAPSHOT_TARGET_OUTER_BATCH_IDX,
        "target_env_index": compare_mod.PAIR2_SNAPSHOT_TARGET_ENV_INDEX,
        "target_objective_episode_index": compare_mod.PAIR2_SNAPSHOT_TARGET_OBJECTIVE_EPISODE_INDEX,
        "target_objective_position_start": compare_mod.PAIR2_SNAPSHOT_TARGET_OBJECTIVE_POSITION_START,
        "target_objective_position_end": compare_mod.PAIR2_SNAPSHOT_TARGET_OBJECTIVE_POSITION_END,
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
            "snapshot_target_match_count": 4,
        },
    }
    broken_override = dict(baseline)
    broken_override["config"] = dict(baseline["config"])
    broken_override["config"]["actor_objective_mode_override"] = compare_mod.PAIR2_BRANCH_SPECIFIC_MODE
    broken_override["config"]["actor_objective_runtime_current_suite_name"] = "suites"

    def _fake_run_audit(
        *,
        actor_objective_mode_override,
        restore_validation_policy_head_state=False,
        reuse_zero_control_json=None,
        reuse_pre_policy_json=None,
        save_post_policy_bundle_path=None,
        resume_post_policy_bundle_path=None,
        eval_n_samples=None,
        train_outer_batch_snapshot_json=None,
        train_outer_batch_snapshot_target_outer_batch_idx=None,
        train_outer_batch_snapshot_target_env_index=None,
        train_outer_batch_snapshot_target_objective_episode_index=None,
        train_outer_batch_snapshot_target_objective_position_start=None,
        train_outer_batch_snapshot_target_objective_position_end=None,
    ):
        return baseline if actor_objective_mode_override is None else broken_override

    monkeypatch.setattr(compare_mod, "_run_audit", _fake_run_audit)
    monkeypatch.setattr(
        compare_mod,
        "_load_pair2_preflight",
        lambda: {
            "preflight_passed": True,
            "contract_args": {},
            "contract_diff": {"allowed_diff": {}, "unexpected_diff": {}},
        },
    )
    monkeypatch.setattr(compare_mod, "_load_pair2_guarded_selector_reidentified", lambda: dict(selector))

    try:
        compare_mod.build_pair2_train_update_ab_compare_pack()
    except ValueError as exc:
        assert "pair2 runtime scope" in str(exc)
    else:
        raise AssertionError("expected runtime-scope mismatch to raise")


def test_build_pair2_train_update_ab_compare_pack_rejects_absent_target_snapshot(monkeypatch):
    selector = {
        "target_outer_batch_idx": compare_mod.PAIR2_SNAPSHOT_TARGET_OUTER_BATCH_IDX,
        "target_env_index": compare_mod.PAIR2_SNAPSHOT_TARGET_ENV_INDEX,
        "target_objective_episode_index": compare_mod.PAIR2_SNAPSHOT_TARGET_OBJECTIVE_EPISODE_INDEX,
        "target_objective_position_start": compare_mod.PAIR2_SNAPSHOT_TARGET_OBJECTIVE_POSITION_START,
        "target_objective_position_end": compare_mod.PAIR2_SNAPSHOT_TARGET_OBJECTIVE_POSITION_END,
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
            "written": False,
            "written_path": None,
        },
    }
    override = dict(baseline)
    override["config"] = dict(baseline["config"])
    override["config"]["actor_objective_mode_override"] = compare_mod.PAIR2_BRANCH_SPECIFIC_MODE
    override["config"]["actor_objective_runtime_current_suite_name"] = "pair2"
    override["train_outer_batch_snapshot"] = {
        "requested_path": "/tmp/override_snapshot.json",
        "selector": dict(selector),
        "written": False,
        "written_path": None,
    }

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
    monkeypatch.setattr(compare_mod, "_load_pair2_guarded_selector_reidentified", lambda: dict(selector))

    try:
        compare_mod.build_pair2_train_update_ab_compare_pack()
    except RuntimeError as exc:
        assert "not written" in str(exc)
    else:
        raise AssertionError("expected absent target snapshot to raise")


def test_build_pair2_train_update_ab_compare_pack_treats_tiny_target_side_delta_as_numeric_noise(
    monkeypatch,
):
    selector = {
        "target_outer_batch_idx": compare_mod.PAIR2_SNAPSHOT_TARGET_OUTER_BATCH_IDX,
        "target_env_index": compare_mod.PAIR2_SNAPSHOT_TARGET_ENV_INDEX,
        "target_objective_episode_index": compare_mod.PAIR2_SNAPSHOT_TARGET_OBJECTIVE_EPISODE_INDEX,
        "target_objective_position_start": compare_mod.PAIR2_SNAPSHOT_TARGET_OBJECTIVE_POSITION_START,
        "target_objective_position_end": compare_mod.PAIR2_SNAPSHOT_TARGET_OBJECTIVE_POSITION_END,
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
            "snapshot_target_match_count": 4,
        },
    }
    override = {
        **baseline,
        "config": {**baseline["config"], "actor_objective_mode_override": compare_mod.PAIR2_BRANCH_SPECIFIC_MODE, "actor_objective_runtime_current_suite_name": "pair2"},
        "comparison": {
            "pre_train_vs_zero": {"full_return_gap": 1.0, "suffix_return_gap": 2.0},
            "pre_heldout_vs_zero": {"full_return_gap": 3.0, "suffix_return_gap": 4.0},
            "post_train_vs_zero": {"full_return_gap": 5.00005, "suffix_return_gap": 6.00006},
            "post_heldout_vs_zero": {"full_return_gap": 7.0, "suffix_return_gap": 8.0},
            "train_full_return_delta": 0.50005,
            "train_suffix_return_delta": 0.25006,
            "heldout_full_return_delta": -0.5,
            "heldout_suffix_return_delta": -0.25,
        },
        "train_outer_batch_snapshot": {
            "requested_path": "/tmp/override_snapshot.json",
            "selector": dict(selector),
            "written": True,
            "written_path": "/tmp/override_snapshot.json",
            "snapshot_target_match_count": 4,
        },
    }

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
    monkeypatch.setattr(compare_mod, "_load_pair2_guarded_selector_reidentified", lambda: dict(selector))

    report = compare_mod.build_pair2_train_update_ab_compare_pack()

    assert report["conclusions"]["target_side_train_update_changes_trajectory"] is False
    assert report["conclusions"]["target_side_effect_is_only_numeric_noise"] is True


def test_build_pair2_train_update_ab_compare_pack_rejects_zero_target_match_snapshot(monkeypatch):
    selector = {
        "target_outer_batch_idx": compare_mod.PAIR2_SNAPSHOT_TARGET_OUTER_BATCH_IDX,
        "target_env_index": compare_mod.PAIR2_SNAPSHOT_TARGET_ENV_INDEX,
        "target_objective_episode_index": compare_mod.PAIR2_SNAPSHOT_TARGET_OBJECTIVE_EPISODE_INDEX,
        "target_objective_position_start": compare_mod.PAIR2_SNAPSHOT_TARGET_OBJECTIVE_POSITION_START,
        "target_objective_position_end": compare_mod.PAIR2_SNAPSHOT_TARGET_OBJECTIVE_POSITION_END,
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
            "actor_objective_mode_override": compare_mod.PAIR2_BRANCH_SPECIFIC_MODE,
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
    monkeypatch.setattr(compare_mod, "_load_pair2_guarded_selector_reidentified", lambda: dict(selector))

    try:
        compare_mod.build_pair2_train_update_ab_compare_pack()
    except RuntimeError as exc:
        assert "matched no objective tokens" in str(exc)
    else:
        raise AssertionError("expected zero target-match snapshot to raise")
