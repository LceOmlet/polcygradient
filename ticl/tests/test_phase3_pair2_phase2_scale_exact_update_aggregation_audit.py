import json
from pathlib import Path

import pytest

from ticl.analysis.phase3_pair2_phase2_scale_exact_update_aggregation_audit import (
    build_pair2_phase2_scale_exact_update_aggregation_audit,
)


def _write_snapshot(tmp_path: Path, payload: dict) -> str:
    path = tmp_path / "snapshot.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return str(path)


def test_trusted_scale_exact_update_audit_reports_stage_metrics_and_retention(tmp_path: Path) -> None:
    snapshot = {
        "summary": {
            "outer_batch_idx": 13,
            "epoch_idx": 1,
            "batch_size_flat": 6,
            "objective_total": 6,
            "normalize_advantage": True,
            "actor_objective_mode": "tokenwise",
            "actor_objective_runtime_current_suite_name": "pair2",
        },
        "snapshot_target_selector": {
            "target_outer_batch_idx": 13,
            "target_env_index": 12,
            "target_objective_episode_index": 10,
            "target_objective_position_start": 0,
            "target_objective_position_end": 1,
        },
        "objective_mask": [1, 1, 1, 1, 1, 1],
        "snapshot_target_match_mask": [0, 1, 1, 0, 0, 0],
        "flat_env_indices": [12, 12, 12, 12, 1, 1],
        "flat_step_indices": [10, 11, 12, 13, 14, 15],
        "flat_objective_episode_indices": [9, 10, 10, 10, 0, 0],
        "flat_objective_episode_positions": [4, 0, 1, 2, 0, 1],
        "flat_objective_global_positions": [20, 22, 23, 24, 0, 1],
        "pre_normalization_actor_advantages": [0.1, -1.0, -2.0, 0.5, -0.5, 0.3],
        "post_normalization_advantages": [0.2, -1.5, -2.5, 0.4, -0.4, 0.2],
        "ratio": [1.0, 0.9, 0.95, 1.0, 1.0, 1.0],
        "clipped_objective": [0.2, -1.2, -2.0, 0.4, -0.4, 0.2],
    }

    report = build_pair2_phase2_scale_exact_update_aggregation_audit(
        snapshot_json=_write_snapshot(tmp_path, snapshot)
    )

    assert report["target_scope"]["selector"]["outer_batch_idx"] == 13
    assert report["target_scope"]["token_count"] == 2
    assert report["target_scope"]["objective_global_position_min"] == 22
    assert report["target_scope"]["objective_global_position_max"] == 23
    assert report["stage_report"]["pre_normalization_actor_advantages"]["target_scope"]["negative_mass"] == pytest.approx(3.0)
    assert report["stage_report"]["post_normalization_advantages"]["target_scope"]["negative_mass"] == pytest.approx(4.0)
    assert report["stage_transition"]["target_negative_mass"]["normalization_delta"] == pytest.approx(1.0)
    assert report["stage_transition"]["target_negative_mass"]["ratio_and_clipping_delta"] == pytest.approx(-0.8)
    assert report["conclusions"]["trusted_target_scope_present_in_snapshot"] is True
    assert report["conclusions"]["target_scope_negative_mass_is_amplified_by_normalization"] is True


def test_trusted_scale_exact_update_audit_rejects_empty_target_scope(tmp_path: Path) -> None:
    snapshot = {
        "summary": {
            "outer_batch_idx": 13,
            "epoch_idx": 1,
            "batch_size_flat": 2,
            "objective_total": 2,
            "normalize_advantage": True,
            "actor_objective_mode": "tokenwise",
            "actor_objective_runtime_current_suite_name": "pair2",
        },
        "snapshot_target_selector": {
            "target_outer_batch_idx": 13,
            "target_env_index": 12,
            "target_objective_episode_index": 10,
            "target_objective_position_start": 0,
            "target_objective_position_end": 1,
        },
        "objective_mask": [1, 1],
        "snapshot_target_match_mask": [0, 0],
        "flat_env_indices": [12, 12],
        "flat_step_indices": [10, 11],
        "flat_objective_episode_indices": [10, 10],
        "flat_objective_episode_positions": [0, 1],
        "flat_objective_global_positions": [22, 23],
        "pre_normalization_actor_advantages": [-1.0, -2.0],
        "post_normalization_advantages": [-1.5, -2.5],
        "ratio": [0.9, 0.95],
        "clipped_objective": [-1.2, -2.0],
    }

    with pytest.raises(ValueError, match="no target-present rows"):
        build_pair2_phase2_scale_exact_update_aggregation_audit(
            snapshot_json=_write_snapshot(tmp_path, snapshot)
        )
