import json

from ticl.analysis import phase3_pair2_high_mass_family_antitransfer_probe as probe_mod


def test_build_pair2_high_mass_family_antitransfer_probe_locks_channel_mismatch(tmp_path):
    transfer_compare = {
        "audit_entry": "phase3_pair2_high_mass_episode_family_heldout_transfer_compare_pack",
        "conclusions": {
            "train_side_reproduction_matches_locked_compare": True,
        },
        "comparison": {
            "train_full_return_delta_change": 0.12,
            "train_suffix_return_delta_change": 0.09,
            "heldout_full_return_delta_change": -0.05,
            "heldout_suffix_return_delta_change": -0.03,
        },
    }
    baseline_resume = {
        "post": {
            "heldout": {
                "full_return_per_env": [1.0, 2.0, 3.0, 4.0],
                "suffix_return_per_env": [0.5, 0.6, 0.7, 0.8],
            }
        }
    }
    override_resume = {
        "post": {
            "heldout": {
                "full_return_per_env": [0.9, 1.7, 3.4, 4.2],
                "suffix_return_per_env": [0.4, 0.4, 0.9, 0.7],
            }
        }
    }
    positive_regime = {
        "missed_summary": {
            "env_items": [
                {"suite_name": "pair2", "env_index": 0},
                {"suite_name": "pair2", "env_index": 1},
                {"suite_name": "pair1", "env_index": 12},
            ]
        },
        "captured_summary": {
            "env_items": [
                {"suite_name": "pair1", "env_index": 10},
            ]
        },
    }
    dual_channel = {
        "recovered_anchor_summary": {
            "env_items": [
                {"suite_name": "pair2", "env_index": 0},
                {"suite_name": "pair2", "env_index": 3},
            ]
        },
        "residual_missed_summary": {
            "env_items": [
                {"suite_name": "pair2", "env_index": 1},
            ]
        },
    }
    snapshot = {
        "objective_mask": [1, 1, 1, 1, 1],
        "flat_env_indices": [12, 12, 12, 12, 12],
        "flat_objective_episode_indices": [3, 3, 3, 4, 4],
        "flat_objective_episode_positions": [0, 1, 2, 0, 1],
        "clipped_objective": [-1.0, -1.0, -0.5, -2.0, -2.0],
    }
    family_candidate = {
        "candidate_families": [
            {
                "name": "high_mass_complete_episodes_4_7_11",
                "token_count": 2,
                "policy_share_of_batch_negative_mass": 0.6153846153846154,
            },
            {
                "name": "all_env12_objective_tokens",
                "token_count": 5,
                "policy_negative_mass": 6.5,
                "policy_share_of_batch_negative_mass": 1.0,
            },
        ]
    }

    paths = {}
    for name, payload in [
        ("transfer_compare", transfer_compare),
        ("baseline_resume", baseline_resume),
        ("override_resume", override_resume),
        ("positive_regime", positive_regime),
        ("dual_channel", dual_channel),
        ("snapshot", snapshot),
        ("family_candidate", family_candidate),
    ]:
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths[name] = str(path)

    report = probe_mod.build_pair2_high_mass_family_antitransfer_probe(
        high_mass_heldout_transfer_compare_json=paths["transfer_compare"],
        baseline_resume_eval_json=paths["baseline_resume"],
        override_resume_eval_json=paths["override_resume"],
        positive_regime_anchor_structure_json=paths["positive_regime"],
        dual_channel_coverage_json=paths["dual_channel"],
        baseline_snapshot_json=paths["snapshot"],
        family_candidate_json=paths["family_candidate"],
    )

    assert report["pair2_anchor_structure"]["captured_reset_tail_anchor_count"] == 0
    assert report["pair2_anchor_structure"]["recovered_dual_channel_anchor_count"] == 2
    assert report["heldout_transfer_groups"]["recovered_dual_channel_anchors"]["mean_suffix_return_delta"] < 0.0
    assert report["current_env12_family_class"]["first_objective_episode_token_count"] == 3
    assert report["current_env12_family_class"]["first_objective_episode_non_tail_token_count"] == 0
    assert report["conclusions"]["current_env12_local_family_class_has_no_first_episode_non_tail_support"] is True
    assert report["conclusions"]["next_family_should_leave_current_env12_local_reset_channel_class"] is True


def test_build_pair2_high_mass_family_antitransfer_probe_rejects_noncanonical_transfer_compare(tmp_path):
    bad_transfer = {
        "audit_entry": "wrong_entry",
        "conclusions": {
            "train_side_reproduction_matches_locked_compare": True,
        },
        "comparison": {
            "train_full_return_delta_change": 0.0,
            "train_suffix_return_delta_change": 0.0,
            "heldout_full_return_delta_change": 0.0,
            "heldout_suffix_return_delta_change": 0.0,
        },
    }
    dummy = {"post": {"heldout": {"full_return_per_env": [0.0], "suffix_return_per_env": [0.0]}}}
    summary = {"missed_summary": {"env_items": []}, "captured_summary": {"env_items": []}}
    dual = {"recovered_anchor_summary": {"env_items": []}, "residual_missed_summary": {"env_items": []}}
    snapshot = {
        "objective_mask": [1],
        "flat_env_indices": [12],
        "flat_objective_episode_indices": [3],
        "flat_objective_episode_positions": [0],
        "clipped_objective": [-1.0],
    }
    family_candidate = {
        "candidate_families": [
            {
                "name": "high_mass_complete_episodes_4_7_11",
                "token_count": 1,
                "policy_share_of_batch_negative_mass": 1.0,
            },
            {
                "name": "all_env12_objective_tokens",
                "token_count": 1,
                "policy_negative_mass": 1.0,
                "policy_share_of_batch_negative_mass": 1.0,
            },
        ]
    }

    payloads = {
        "bad_transfer": bad_transfer,
        "baseline": dummy,
        "override": dummy,
        "positive": summary,
        "dual": dual,
        "snapshot": snapshot,
        "family": family_candidate,
    }
    paths = {}
    for name, payload in payloads.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths[name] = str(path)

    try:
        probe_mod.build_pair2_high_mass_family_antitransfer_probe(
            high_mass_heldout_transfer_compare_json=paths["bad_transfer"],
            baseline_resume_eval_json=paths["baseline"],
            override_resume_eval_json=paths["override"],
            positive_regime_anchor_structure_json=paths["positive"],
            dual_channel_coverage_json=paths["dual"],
            baseline_snapshot_json=paths["snapshot"],
            family_candidate_json=paths["family"],
        )
    except RuntimeError as exc:
        assert "Expected canonical pair2 high-mass-family heldout-transfer compare artifact." in str(exc)
    else:
        raise AssertionError("Expected noncanonical transfer compare to be rejected")
