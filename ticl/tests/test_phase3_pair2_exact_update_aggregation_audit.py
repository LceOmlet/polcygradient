import json

from ticl.analysis.phase3_pair2_exact_update_aggregation_audit import (
    build_pair2_exact_update_aggregation_audit,
)


def _write(tmp_path, name: str, payload: dict) -> str:
    path = tmp_path / name
    path.write_text(json.dumps(payload))
    return str(path)


def test_build_pair2_exact_update_aggregation_audit_tracks_three_stage_transfer(tmp_path):
    base_snapshot = _write(
        tmp_path,
        "baseline.json",
        {
            "summary": {
                "outer_batch_idx": 8,
                "epoch_idx": 1,
                "batch_size_flat": 8,
                "objective_total": 6,
                "normalize_advantage": True,
                "actor_objective_mode": "tokenwise",
                "actor_objective_runtime_current_suite_name": "pair2",
            },
            "objective_mask": [1, 1, 1, 1, 1, 0, 0, 1],
            "flat_env_indices": [12, 12, 12, 12, 12, 0, 0, 1],
            "flat_step_indices": [10, 11, 12, 13, 14, 0, 1, 2],
            "flat_objective_episode_indices": [0, 0, 0, 0, 0, -1, -1, 0],
            "flat_objective_episode_positions": [18, 19, 20, 21, 30, -1, -1, 0],
            "flat_objective_global_positions": [18, 19, 20, 21, 30, -1, -1, 0],
            "pre_normalization_actor_advantages": [-2.0, -2.0, -2.0, -2.0, -1.0, 0.0, 0.0, 4.0],
            "post_normalization_advantages": [-1.0, -1.0, -1.0, -1.0, -0.5, 0.0, 0.0, 2.0],
            "ratio": [1.0, 1.0, 1.0, 1.0, 1.1, 1.0, 1.0, 0.5],
            "clipped_objective": [-1.0, -1.0, -1.0, -1.0, -0.55, 0.0, 0.0, 1.0],
        },
    )
    override_snapshot = _write(
        tmp_path,
        "override.json",
        {
            "summary": {
                "outer_batch_idx": 8,
                "epoch_idx": 1,
                "batch_size_flat": 8,
                "objective_total": 6,
                "normalize_advantage": True,
                "actor_objective_mode": "tokenwise_scale_env12_mid_episode_extension_block",
                "actor_objective_runtime_current_suite_name": "pair2",
            },
            "objective_mask": [1, 1, 1, 1, 1, 0, 0, 1],
            "flat_env_indices": [12, 12, 12, 12, 12, 0, 0, 1],
            "flat_step_indices": [10, 11, 12, 13, 14, 0, 1, 2],
            "flat_objective_episode_indices": [0, 0, 0, 0, 0, -1, -1, 0],
            "flat_objective_episode_positions": [18, 19, 20, 21, 30, -1, -1, 0],
            "flat_objective_global_positions": [18, 19, 20, 21, 30, -1, -1, 0],
            "pre_normalization_actor_advantages": [-1.5, -1.5, -1.5, -1.5, -1.0, 0.0, 0.0, 4.0],
            "post_normalization_advantages": [-0.7, -0.7, -0.7, -0.7, -0.5, 0.0, 0.0, 2.0],
            "ratio": [1.0, 1.0, 1.0, 1.0, 1.1, 1.0, 1.0, 0.5],
            "clipped_objective": [-0.6, -0.6, -0.6, -0.6, -0.55, 0.0, 0.0, 1.0],
        },
    )

    report = build_pair2_exact_update_aggregation_audit(
        baseline_snapshot_json=base_snapshot,
        override_snapshot_json=override_snapshot,
    )

    assert report["strict_contract"]["runtime_suite_name"] == "pair2"
    assert report["strict_contract"]["baseline_runtime_suite_name"] == "pair2"
    assert report["strict_contract"]["override_runtime_suite_name"] == "pair2"
    assert report["target_block"]["block_token_count"] == 4
    assert report["baseline"]["pre_normalization_actor_advantages"]["block"]["negative_mass"] == 8.0
    assert report["override"]["pre_normalization_actor_advantages"]["block"]["negative_mass"] == 6.0
    assert report["delta"]["pre_normalization_actor_advantages"]["block_negative_mass_shrink"] == 2.0
    assert abs(report["delta"]["post_normalization_advantages"]["block_negative_mass_shrink"] - 1.2) <= 1e-12
    assert abs(report["delta"]["policy_num_from_clipped_objective"]["block_negative_mass_shrink"] - 1.6) <= 1e-12
    assert report["conclusions"]["block_exact_pre_norm_negative_mass_shrinks"] is True
    assert report["conclusions"]["block_exact_post_norm_negative_mass_shrink_retained"] is True
    assert report["conclusions"]["block_exact_policy_num_negative_mass_shrink_retained"] is True


def test_build_pair2_exact_update_aggregation_audit_rejects_block_mismatch(tmp_path):
    baseline_snapshot = _write(
        tmp_path,
        "baseline.json",
        {
            "summary": {
                "outer_batch_idx": 8,
                "epoch_idx": 1,
                "batch_size_flat": 4,
                "objective_total": 4,
                "normalize_advantage": True,
                "actor_objective_mode": "tokenwise",
                "actor_objective_runtime_current_suite_name": "pair2",
            },
            "objective_mask": [1, 1, 1, 1],
            "flat_env_indices": [12, 12, 12, 12],
            "flat_step_indices": [0, 1, 2, 3],
            "flat_objective_episode_indices": [0, 0, 0, 0],
            "flat_objective_episode_positions": [18, 19, 20, 21],
            "flat_objective_global_positions": [18, 19, 20, 21],
            "pre_normalization_actor_advantages": [-1, -1, -1, -1],
            "post_normalization_advantages": [-1, -1, -1, -1],
            "ratio": [1, 1, 1, 1],
            "clipped_objective": [-1, -1, -1, -1],
        },
    )
    override_snapshot = _write(
        tmp_path,
        "override.json",
        {
            "summary": {
                "outer_batch_idx": 8,
                "epoch_idx": 1,
                "batch_size_flat": 4,
                "objective_total": 4,
                "normalize_advantage": True,
                "actor_objective_mode": "tokenwise_scale_env12_mid_episode_extension_block",
                "actor_objective_runtime_current_suite_name": "pair2",
            },
            "objective_mask": [1, 1, 1, 0],
            "flat_env_indices": [12, 12, 12, 12],
            "flat_step_indices": [0, 1, 2, 3],
            "flat_objective_episode_indices": [0, 0, 0, 0],
            "flat_objective_episode_positions": [18, 19, 20, 21],
            "flat_objective_global_positions": [18, 19, 20, 21],
            "pre_normalization_actor_advantages": [-1, -1, -1, -1],
            "post_normalization_advantages": [-1, -1, -1, -1],
            "ratio": [1, 1, 1, 1],
            "clipped_objective": [-1, -1, -1, -1],
        },
    )

    try:
        build_pair2_exact_update_aggregation_audit(
            baseline_snapshot_json=baseline_snapshot,
            override_snapshot_json=override_snapshot,
        )
    except ValueError as exc:
        assert "branch-specific block" in str(exc) or "objective masks must match" in str(exc)
    else:
        raise AssertionError("expected block/objective mismatch to raise")


def test_build_pair2_exact_update_aggregation_audit_reports_missing_target_block(tmp_path):
    baseline_snapshot = _write(
        tmp_path,
        "baseline.json",
        {
            "summary": {
                "outer_batch_idx": 8,
                "epoch_idx": 1,
                "batch_size_flat": 4,
                "objective_total": 4,
                "normalize_advantage": True,
                "actor_objective_mode": "tokenwise",
                "actor_objective_runtime_current_suite_name": "pair2",
            },
            "objective_mask": [1, 1, 1, 1],
            "flat_env_indices": [0, 0, 0, 0],
            "flat_step_indices": [100, 101, 102, 103],
            "flat_objective_episode_indices": [2, 2, 2, 2],
            "flat_objective_episode_positions": [0, 1, 2, 3],
            "flat_objective_global_positions": [0, 1, 2, 3],
            "pre_normalization_actor_advantages": [-1, -1, -1, -1],
            "post_normalization_advantages": [-1, -1, -1, -1],
            "ratio": [1, 1, 1, 1],
            "clipped_objective": [-1, -1, -1, -1],
        },
    )
    override_snapshot = _write(
        tmp_path,
        "override.json",
        {
            "summary": {
                "outer_batch_idx": 8,
                "epoch_idx": 1,
                "batch_size_flat": 4,
                "objective_total": 4,
                "normalize_advantage": True,
                "actor_objective_mode": "tokenwise_scale_env12_mid_episode_extension_block",
                "actor_objective_runtime_current_suite_name": "pair2",
            },
            "objective_mask": [1, 1, 1, 1],
            "flat_env_indices": [0, 0, 0, 0],
            "flat_step_indices": [100, 101, 102, 103],
            "flat_objective_episode_indices": [2, 2, 2, 2],
            "flat_objective_episode_positions": [0, 1, 2, 3],
            "flat_objective_global_positions": [0, 1, 2, 3],
            "pre_normalization_actor_advantages": [-1, -1, -1, -1],
            "post_normalization_advantages": [-1, -1, -1, -1],
            "ratio": [1, 1, 1, 1],
            "clipped_objective": [-1, -1, -1, -1],
        },
    )

    report = build_pair2_exact_update_aggregation_audit(
        baseline_snapshot_json=baseline_snapshot,
        override_snapshot_json=override_snapshot,
    )

    assert report["strict_contract"]["runtime_suite_name"] == "pair2"
    assert report["strict_contract"]["baseline_runtime_suite_name"] == "pair2"
    assert report["strict_contract"]["override_runtime_suite_name"] == "pair2"
    assert report["target_block"]["target_block_present_in_snapshot"] is False
    assert report["conclusions"]["requested_target_block_present_in_captured_outer_batch"] is False
    assert report["conclusions"]["current_single_snapshot_is_insufficient_for_env12_block_audit"] is True
    assert report["available_snapshot_scope"]["objective_segments"] == [
        {
            "env_index": 0,
            "objective_episode_index": 2,
            "token_count": 4,
            "objective_position_min": 0,
            "objective_position_max": 3,
        }
    ]


def test_build_pair2_exact_update_aggregation_audit_tracks_baseline_vs_override_runtime_scope(tmp_path):
    baseline_snapshot = _write(
        tmp_path,
        "baseline.json",
        {
            "summary": {
                "outer_batch_idx": 8,
                "epoch_idx": 1,
                "batch_size_flat": 4,
                "objective_total": 4,
                "normalize_advantage": False,
                "actor_objective_mode": "tokenwise",
                "actor_objective_runtime_current_suite_name": "None",
            },
            "objective_mask": [1, 1, 1, 1],
            "flat_env_indices": [12, 12, 12, 12],
            "flat_step_indices": [0, 1, 2, 3],
            "flat_objective_episode_indices": [0, 0, 0, 0],
            "flat_objective_episode_positions": [18, 19, 20, 21],
            "flat_objective_global_positions": [18, 19, 20, 21],
            "pre_normalization_actor_advantages": [-1, -1, -1, -1],
            "post_normalization_advantages": [-1, -1, -1, -1],
            "ratio": [1, 1, 1, 1],
            "clipped_objective": [-1, -1, -1, -1],
        },
    )
    override_snapshot = _write(
        tmp_path,
        "override.json",
        {
            "summary": {
                "outer_batch_idx": 8,
                "epoch_idx": 1,
                "batch_size_flat": 4,
                "objective_total": 4,
                "normalize_advantage": False,
                "actor_objective_mode": "tokenwise_scale_env12_mid_episode_extension_block",
                "actor_objective_runtime_current_suite_name": "pair2",
            },
            "objective_mask": [1, 1, 1, 1],
            "flat_env_indices": [12, 12, 12, 12],
            "flat_step_indices": [0, 1, 2, 3],
            "flat_objective_episode_indices": [0, 0, 0, 0],
            "flat_objective_episode_positions": [18, 19, 20, 21],
            "flat_objective_global_positions": [18, 19, 20, 21],
            "pre_normalization_actor_advantages": [-0.9, -0.9, -0.9, -0.9],
            "post_normalization_advantages": [-0.9, -0.9, -0.9, -0.9],
            "ratio": [1, 1, 1, 1],
            "clipped_objective": [-0.9, -0.9, -0.9, -0.9],
        },
    )

    report = build_pair2_exact_update_aggregation_audit(
        baseline_snapshot_json=baseline_snapshot,
        override_snapshot_json=override_snapshot,
    )

    assert report["strict_contract"]["runtime_suite_name"] == "pair2"
    assert report["strict_contract"]["baseline_runtime_suite_name"] == "None"
    assert report["strict_contract"]["override_runtime_suite_name"] == "pair2"
