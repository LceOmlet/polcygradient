import json

import pytest

from ticl.analysis import (
    phase3_pair2_recovered_anchor_first_objective_non_tail_family_heldout_transfer_compare_pack as compare_mod,
)


def _make_report(
    *,
    runtime_suite_name,
    actor_objective_mode_override,
    post_train_gap,
    post_heldout_gap,
    train_delta,
    heldout_delta,
):
    return {
        "config": {
            "checkpoint_path": "/home/chen/RLPFN/rwkv/models_diff/rlpfn_03_30_2026_22_37_53_epoch_13.cpkt",
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
            "actor_objective_runtime_current_suite_name": runtime_suite_name,
            "actor_objective_mode_override": actor_objective_mode_override,
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
            "train": {"full_return_mean": 0.0, "suffix_return_mean": 0.0},
            "heldout": {"full_return_mean": 0.0, "suffix_return_mean": 0.0},
        },
        "pre": {
            "train": {"full_return_mean": 1.0, "suffix_return_mean": 2.0},
            "heldout": {"full_return_mean": 3.0, "suffix_return_mean": 4.0},
        },
        "comparison": {
            "pre_train_vs_zero": {"full_return_gap": 1.0, "suffix_return_gap": 2.0},
            "pre_heldout_vs_zero": {"full_return_gap": 3.0, "suffix_return_gap": 4.0},
            "post_train_vs_zero": dict(post_train_gap),
            "post_heldout_vs_zero": dict(post_heldout_gap),
            "train_full_return_delta": float(train_delta["full"]),
            "train_suffix_return_delta": float(train_delta["suffix"]),
            "heldout_full_return_delta": float(heldout_delta["full"]),
            "heldout_suffix_return_delta": float(heldout_delta["suffix"]),
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
    }


def test_build_pair2_recovered_anchor_first_objective_non_tail_family_heldout_transfer_compare_pack_detects_material_transfer(
    monkeypatch, tmp_path
):
    locked_train = {
        "audit_entry": "phase3_pair2_recovered_anchor_first_objective_non_tail_family_train_update_ab_compare_pack",
        "comparison": {
            "train_full_return_delta_change": 0.008572101593017578,
            "train_suffix_return_delta_change": 0.005451679229736328,
            "heldout_eval_skipped": True,
        },
        "conclusions": {
            "target_side_compare_is_efficacy_eligible": True,
            "target_side_train_update_changes_trajectory": True,
        },
        "baseline": {
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
        },
        "override": {
            "comparison": {
                "pre_train_vs_zero": {"full_return_gap": 1.0, "suffix_return_gap": 2.0},
                "post_train_vs_zero": {"full_return_gap": 5.008572101593018, "suffix_return_gap": 6.005451679229736},
                "train_full_return_delta": 0.5085721015930176,
                "train_suffix_return_delta": 0.25545167922973633,
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
        },
    }
    baseline_report = _make_report(
        runtime_suite_name=None,
        actor_objective_mode_override=None,
        post_train_gap={"full_return_gap": 5.0, "suffix_return_gap": 6.0},
        post_heldout_gap={"full_return_gap": 7.0, "suffix_return_gap": 8.0},
        train_delta={"full": 0.5, "suffix": 0.25},
        heldout_delta={"full": 0.2, "suffix": 0.1},
    )
    override_report = _make_report(
        runtime_suite_name="pair2",
        actor_objective_mode_override=compare_mod.PAIR2_RECOVERED_ANCHOR_NON_TAIL_BRANCH_SPECIFIC_MODE,
        post_train_gap={"full_return_gap": 5.008572101593018, "suffix_return_gap": 6.005451679229736},
        post_heldout_gap={"full_return_gap": 7.02, "suffix_return_gap": 8.015},
        train_delta={"full": 0.5085721015930176, "suffix": 0.25545167922973633},
        heldout_delta={"full": 0.23, "suffix": 0.115},
    )

    canonical_zero = tmp_path / "zero.json"
    canonical_zero.write_text(
        json.dumps({"config": {"n_samples": 2048, "single_eval_pos": 1946, "rollout_backend": "serial"}}),
        encoding="utf-8",
    )
    canonical_pre = tmp_path / "pre.json"
    canonical_pre.write_text(
        json.dumps({"config": {"n_samples": 2048, "single_eval_pos": 1946, "rollout_backend": "serial"}}),
        encoding="utf-8",
    )

    monkeypatch.setattr(compare_mod, "PAIR2_ZERO_JSON", str(canonical_zero))
    monkeypatch.setattr(compare_mod, "PAIR2_PRE_JSON", str(canonical_pre))
    monkeypatch.setattr(
        compare_mod,
        "_load_pair2_preflight",
        lambda: {"preflight_passed": True, "contract_args": {}, "contract_diff": {"allowed_diff": {}, "unexpected_diff": {}}},
    )
    monkeypatch.setattr(compare_mod, "_load_locked_train_compare", lambda: dict(locked_train))
    monkeypatch.setattr(
        compare_mod,
        "_load_bundle_contract",
        lambda path: {
            "checkpoint_path": "/home/chen/RLPFN/rwkv/models_diff/rlpfn_03_30_2026_22_37_53_epoch_13.cpkt",
            "n_samples": 256,
            "single_eval_pos": 64,
            "rollout_backend": "serial",
            "train_profile": "phase2_shared_backbone_contract",
            "batch_size": 256,
            "n_epochs": 1,
            "learning_rate": 2e-4,
            "target_kl": 0.03,
            "actor_objective_mode": (
                "tokenwise"
                if "baseline" in str(path)
                else compare_mod.PAIR2_RECOVERED_ANCHOR_NON_TAIL_BRANCH_SPECIFIC_MODE
            ),
            "ppo_actor_baseline_mode": "learned",
            "ppo_normalize_advantage": False,
            "ppo_actor_gae_space": "normalized",
            "ppo_vf_coef": 0.5,
            "ppo_reset_env_state_at_sep": True,
            "ppo_separate_value_backbone": False,
            "ppo_restore_validation_policy_state": False,
            "ppo_restore_validation_policy_head_state": True,
            "strict_native_rollout": True,
        },
    )
    monkeypatch.setattr(
        compare_mod,
        "_run_resumed_eval_audit",
        lambda **kwargs: baseline_report
        if kwargs["actor_objective_mode_override"] is None
        else override_report,
    )

    report = compare_mod.build_pair2_recovered_anchor_first_objective_non_tail_family_heldout_transfer_compare_pack()

    assert report["conclusions"]["train_side_reproduction_matches_locked_compare"] is True
    assert report["conclusions"]["heldout_transfer_compare_is_efficacy_eligible"] is True
    assert report["conclusions"]["heldout_transfer_material"] is True
    assert report["conclusions"]["heldout_transfer_is_only_numeric_noise"] is False
    assert report["conclusions"]["canonical_pair2_cross_env_baselines_reuse_ineligible"] is True
    assert report["comparison"]["heldout_full_return_delta_change"] == pytest.approx(0.03)
    assert report["comparison"]["heldout_suffix_return_delta_change"] == pytest.approx(0.015)


def test_build_pair2_recovered_anchor_first_objective_non_tail_family_heldout_transfer_compare_pack_rejects_train_side_reproduction_drift(
    monkeypatch, tmp_path
):
    locked_train = {
        "audit_entry": "phase3_pair2_recovered_anchor_first_objective_non_tail_family_train_update_ab_compare_pack",
        "comparison": {
            "train_full_return_delta_change": 0.008572101593017578,
            "train_suffix_return_delta_change": 0.005451679229736328,
            "heldout_eval_skipped": True,
        },
        "conclusions": {
            "target_side_compare_is_efficacy_eligible": True,
            "target_side_train_update_changes_trajectory": True,
        },
        "baseline": {
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
        },
        "override": {
            "comparison": {
                "pre_train_vs_zero": {"full_return_gap": 1.0, "suffix_return_gap": 2.0},
                "post_train_vs_zero": {"full_return_gap": 5.008572101593018, "suffix_return_gap": 6.005451679229736},
                "train_full_return_delta": 0.5085721015930176,
                "train_suffix_return_delta": 0.25545167922973633,
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
        },
    }
    baseline_report = _make_report(
        runtime_suite_name=None,
        actor_objective_mode_override=None,
        post_train_gap={"full_return_gap": 5.1, "suffix_return_gap": 6.0},
        post_heldout_gap={"full_return_gap": 7.0, "suffix_return_gap": 8.0},
        train_delta={"full": 0.5, "suffix": 0.25},
        heldout_delta={"full": 0.2, "suffix": 0.1},
    )
    override_report = _make_report(
        runtime_suite_name="pair2",
        actor_objective_mode_override=compare_mod.PAIR2_RECOVERED_ANCHOR_NON_TAIL_BRANCH_SPECIFIC_MODE,
        post_train_gap={"full_return_gap": 5.008572101593018, "suffix_return_gap": 6.005451679229736},
        post_heldout_gap={"full_return_gap": 7.02, "suffix_return_gap": 8.015},
        train_delta={"full": 0.5085721015930176, "suffix": 0.25545167922973633},
        heldout_delta={"full": 0.23, "suffix": 0.115},
    )

    canonical_zero = tmp_path / "zero.json"
    canonical_zero.write_text(
        json.dumps({"config": {"n_samples": 2048, "single_eval_pos": 1946, "rollout_backend": "serial"}}),
        encoding="utf-8",
    )
    canonical_pre = tmp_path / "pre.json"
    canonical_pre.write_text(
        json.dumps({"config": {"n_samples": 2048, "single_eval_pos": 1946, "rollout_backend": "serial"}}),
        encoding="utf-8",
    )

    monkeypatch.setattr(compare_mod, "PAIR2_ZERO_JSON", str(canonical_zero))
    monkeypatch.setattr(compare_mod, "PAIR2_PRE_JSON", str(canonical_pre))
    monkeypatch.setattr(
        compare_mod,
        "_load_pair2_preflight",
        lambda: {"preflight_passed": True, "contract_args": {}, "contract_diff": {"allowed_diff": {}, "unexpected_diff": {}}},
    )
    monkeypatch.setattr(compare_mod, "_load_locked_train_compare", lambda: dict(locked_train))
    monkeypatch.setattr(
        compare_mod,
        "_load_bundle_contract",
        lambda path: {
            "checkpoint_path": "/home/chen/RLPFN/rwkv/models_diff/rlpfn_03_30_2026_22_37_53_epoch_13.cpkt",
            "n_samples": 256,
            "single_eval_pos": 64,
            "rollout_backend": "serial",
            "train_profile": "phase2_shared_backbone_contract",
            "batch_size": 256,
            "n_epochs": 1,
            "learning_rate": 2e-4,
            "target_kl": 0.03,
            "actor_objective_mode": (
                "tokenwise"
                if "baseline" in str(path)
                else compare_mod.PAIR2_RECOVERED_ANCHOR_NON_TAIL_BRANCH_SPECIFIC_MODE
            ),
            "ppo_actor_baseline_mode": "learned",
            "ppo_normalize_advantage": False,
            "ppo_actor_gae_space": "normalized",
            "ppo_vf_coef": 0.5,
            "ppo_reset_env_state_at_sep": True,
            "ppo_separate_value_backbone": False,
            "ppo_restore_validation_policy_state": False,
            "ppo_restore_validation_policy_head_state": True,
            "strict_native_rollout": True,
        },
    )
    monkeypatch.setattr(
        compare_mod,
        "_run_resumed_eval_audit",
        lambda **kwargs: baseline_report
        if kwargs["actor_objective_mode_override"] is None
        else override_report,
    )

    with pytest.raises(RuntimeError, match="baseline.post_train_vs_zero.full_return_gap"):
        compare_mod.build_pair2_recovered_anchor_first_objective_non_tail_family_heldout_transfer_compare_pack()


def test_build_pair2_recovered_anchor_first_objective_non_tail_family_heldout_transfer_compare_pack_rejects_bundle_contract_mismatch(
    monkeypatch, tmp_path
):
    canonical_zero = tmp_path / "zero.json"
    canonical_zero.write_text(
        json.dumps({"config": {"n_samples": 2048, "single_eval_pos": 1946, "rollout_backend": "serial"}}),
        encoding="utf-8",
    )
    canonical_pre = tmp_path / "pre.json"
    canonical_pre.write_text(
        json.dumps({"config": {"n_samples": 2048, "single_eval_pos": 1946, "rollout_backend": "serial"}}),
        encoding="utf-8",
    )

    monkeypatch.setattr(compare_mod, "PAIR2_ZERO_JSON", str(canonical_zero))
    monkeypatch.setattr(compare_mod, "PAIR2_PRE_JSON", str(canonical_pre))
    monkeypatch.setattr(
        compare_mod,
        "_load_pair2_preflight",
        lambda: {"preflight_passed": True, "contract_args": {}, "contract_diff": {"allowed_diff": {}, "unexpected_diff": {}}},
    )
    monkeypatch.setattr(
        compare_mod,
        "_load_locked_train_compare",
        lambda: {
            "audit_entry": "phase3_pair2_recovered_anchor_first_objective_non_tail_family_train_update_ab_compare_pack",
            "comparison": {
                "train_full_return_delta_change": 0.008572101593017578,
                "train_suffix_return_delta_change": 0.005451679229736328,
                "heldout_eval_skipped": True,
            },
            "conclusions": {
                "target_side_compare_is_efficacy_eligible": True,
                "target_side_train_update_changes_trajectory": True,
            },
        },
    )
    monkeypatch.setattr(
        compare_mod,
        "_load_bundle_contract",
        lambda path: {
            "checkpoint_path": "/home/chen/RLPFN/rwkv/models_diff/rlpfn_03_30_2026_22_37_53_epoch_13.cpkt",
            "n_samples": 256,
            "single_eval_pos": 64,
            "rollout_backend": "serial",
            "train_profile": "phase2_shared_backbone_contract",
            "batch_size": 256,
            "n_epochs": 1,
            "learning_rate": 2e-4,
            "target_kl": 0.03,
            "actor_objective_mode": "wrong_mode",
            "ppo_actor_baseline_mode": "learned",
            "ppo_normalize_advantage": False,
            "ppo_actor_gae_space": "normalized",
            "ppo_vf_coef": 0.5,
            "ppo_reset_env_state_at_sep": True,
            "ppo_separate_value_backbone": False,
            "ppo_restore_validation_policy_state": False,
            "ppo_restore_validation_policy_head_state": True,
            "strict_native_rollout": True,
        },
    )

    with pytest.raises(RuntimeError, match="bundle contract mismatch for actor_objective_mode"):
        compare_mod.build_pair2_recovered_anchor_first_objective_non_tail_family_heldout_transfer_compare_pack()
