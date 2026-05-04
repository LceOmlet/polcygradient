import json

from ticl.analysis import (
    phase3_pair2_recovered_anchor_first_objective_non_tail_family_train_update_ab_compare_pack as compare_mod,
)


def _write_snapshot(path, *, env_indices, episode_indices, positions):
    payload = {
        "objective_mask": [1] * len(env_indices),
        "flat_env_indices": list(env_indices),
        "flat_objective_episode_indices": list(episode_indices),
        "flat_objective_episode_positions": list(positions),
        "clipped_objective": [-1.0] * len(env_indices),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_build_pair2_recovered_anchor_first_objective_non_tail_family_train_update_ab_compare_pack_detects_trajectory_change(
    monkeypatch, tmp_path
):
    family_spec = {
        "family_probe_source_path": "/tmp/current_contract_probe.json",
        "target_env_indices": [0, 15],
        "target_objective_episode_index": 0,
        "target_env_position_end_by_env": {0: 46, 15: 31},
        "family_token_count": 79,
        "family_raw_negative_share_of_batch": 0.023,
        "dominant_snapshot_selector": {
            "target_outer_batch_idx": 1,
            "target_env_index": 0,
            "target_objective_episode_index": 0,
            "target_objective_position_start": 0,
            "target_objective_position_end": 46,
        },
        "outer_batch_distribution": {
            "family_spans_multiple_outer_batches": True,
            "single_outer_batch_covers_full_family": False,
        },
    }
    baseline_snapshot_path = tmp_path / "baseline_snapshot.json"
    override_snapshot_path = tmp_path / "override_snapshot.json"
    _write_snapshot(
        baseline_snapshot_path,
        env_indices=[0, 0, 1, 1],
        episode_indices=[0, 0, 0, 0],
        positions=[0, 46, 0, 1],
    )
    _write_snapshot(
        override_snapshot_path,
        env_indices=[0, 0, 0, 1],
        episode_indices=[0, 0, 0, 0],
        positions=[0, 20, 46, 1],
    )

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
            "requested_path": str(baseline_snapshot_path),
            "selector": dict(family_spec["dominant_snapshot_selector"]),
            "written": True,
            "written_path": str(baseline_snapshot_path),
            "snapshot_target_match_count": 2,
        },
    }
    override = {
        **baseline,
        "config": {
            **baseline["config"],
            "actor_objective_mode_override": compare_mod.PAIR2_RECOVERED_ANCHOR_NON_TAIL_BRANCH_SPECIFIC_MODE,
            "actor_objective_runtime_current_suite_name": "pair2",
        },
        "comparison": {
            **baseline["comparison"],
            "post_train_vs_zero": {"full_return_gap": 5.4, "suffix_return_gap": 6.5},
            "train_full_return_delta": 0.9,
            "train_suffix_return_delta": 0.55,
        },
        "train_outer_batch_snapshot": {
            "requested_path": str(override_snapshot_path),
            "selector": dict(family_spec["dominant_snapshot_selector"]),
            "written": True,
            "written_path": str(override_snapshot_path),
            "snapshot_target_match_count": 3,
        },
    }

    monkeypatch.setattr(compare_mod, "_load_recovered_anchor_non_tail_family_spec", lambda: dict(family_spec))
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

    report = compare_mod.build_pair2_recovered_anchor_first_objective_non_tail_family_train_update_ab_compare_pack()

    assert report["target_control"]["candidate_small_control_target"] == "recovered_anchor_first_objective_non_tail_family"
    assert report["conclusions"]["target_side_compare_is_efficacy_eligible"] is True
    assert report["conclusions"]["target_side_train_update_changes_trajectory"] is True
    assert report["conclusions"]["family_spans_multiple_outer_batches"] is True
    assert report["conclusions"]["single_snapshot_only_covers_dominant_family_segment"] is True
    assert report["train_outer_batch_snapshots"]["baseline_dominant_segment_summary"]["token_count"] == 2
    assert report["train_outer_batch_snapshots"]["override_dominant_segment_summary"]["token_count"] == 3


def test_build_pair2_recovered_anchor_first_objective_non_tail_family_train_update_ab_compare_pack_rejects_missing_dominant_snapshot_tokens(
    monkeypatch, tmp_path
):
    family_spec = {
        "family_probe_source_path": "/tmp/current_contract_probe.json",
        "target_env_indices": [0, 15],
        "target_objective_episode_index": 0,
        "target_env_position_end_by_env": {0: 46, 15: 31},
        "family_token_count": 79,
        "family_raw_negative_share_of_batch": 0.023,
        "dominant_snapshot_selector": {
            "target_outer_batch_idx": 1,
            "target_env_index": 0,
            "target_objective_episode_index": 0,
            "target_objective_position_start": 0,
            "target_objective_position_end": 46,
        },
        "outer_batch_distribution": {
            "family_spans_multiple_outer_batches": True,
            "single_outer_batch_covers_full_family": False,
        },
    }
    baseline_snapshot_path = tmp_path / "baseline_snapshot.json"
    override_snapshot_path = tmp_path / "override_snapshot.json"
    _write_snapshot(
        baseline_snapshot_path,
        env_indices=[1, 1, 1],
        episode_indices=[0, 0, 0],
        positions=[0, 1, 2],
    )
    _write_snapshot(
        override_snapshot_path,
        env_indices=[1, 1, 1],
        episode_indices=[0, 0, 0],
        positions=[0, 1, 2],
    )

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
            "requested_path": str(baseline_snapshot_path),
            "selector": dict(family_spec["dominant_snapshot_selector"]),
            "written": True,
            "written_path": str(baseline_snapshot_path),
            "snapshot_target_match_count": 0,
        },
    }
    override = {
        **baseline,
        "config": {
            **baseline["config"],
            "actor_objective_mode_override": compare_mod.PAIR2_RECOVERED_ANCHOR_NON_TAIL_BRANCH_SPECIFIC_MODE,
            "actor_objective_runtime_current_suite_name": "pair2",
        },
        "train_outer_batch_snapshot": {
            "requested_path": str(override_snapshot_path),
            "selector": dict(family_spec["dominant_snapshot_selector"]),
            "written": True,
            "written_path": str(override_snapshot_path),
            "snapshot_target_match_count": 0,
        },
    }

    monkeypatch.setattr(compare_mod, "_load_recovered_anchor_non_tail_family_spec", lambda: dict(family_spec))
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
        compare_mod.build_pair2_recovered_anchor_first_objective_non_tail_family_train_update_ab_compare_pack()
    except RuntimeError as exc:
        assert "matched no dominant family tokens" in str(exc)
    else:
        raise AssertionError("Expected compare pack to reject snapshots that miss the dominant family segment.")
