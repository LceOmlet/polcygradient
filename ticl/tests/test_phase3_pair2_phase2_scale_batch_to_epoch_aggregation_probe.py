import json
from pathlib import Path

from ticl.analysis.phase3_pair2_phase2_scale_batch_to_epoch_aggregation_probe import (
    build_pair2_phase2_scale_batch_to_epoch_aggregation_probe,
)


def _write_json(tmp_path: Path, name: str, payload: dict) -> str:
    path = tmp_path / name
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return str(path)


def test_batch_to_epoch_probe_locks_singleton_env_epoch_context(tmp_path: Path) -> None:
    selector_trace = {
        "rows": [
            {
                "outer_batch_idx": 1,
                "total_outer_batches": 4,
                "objective_total": 192,
                "objective_env_indices_present": [0],
                "target_match_count": 0,
                "target_env_objective_episode_spans": [],
            },
            {
                "outer_batch_idx": 2,
                "total_outer_batches": 4,
                "objective_total": 192,
                "objective_env_indices_present": [1],
                "target_match_count": 1,
                "target_env_objective_episode_spans": [{"objective_episode_index": 10, "token_count": 8}],
            },
            {
                "outer_batch_idx": 3,
                "total_outer_batches": 4,
                "objective_total": 192,
                "objective_env_indices_present": [2],
                "target_match_count": 0,
                "target_env_objective_episode_spans": [],
            },
            {
                "outer_batch_idx": 4,
                "total_outer_batches": 4,
                "objective_total": 192,
                "objective_env_indices_present": [3],
                "target_match_count": 0,
                "target_env_objective_episode_spans": [],
            },
        ]
    }
    exact_audit = {
        "target_scope": {
            "selector": {
                "outer_batch_idx": 2,
                "env_index": 1,
                "objective_episode_index": 10,
                "objective_position_start": 0,
                "objective_position_end": 7,
            }
        },
        "stage_transition": {
            "target_share_of_batch_negative_mass": {
                "policy_num": 0.25,
            }
        },
    }

    report = build_pair2_phase2_scale_batch_to_epoch_aggregation_probe(
        selector_trace_json=_write_json(tmp_path, "selector_trace.json", selector_trace),
        exact_audit_json=_write_json(tmp_path, "exact_audit.json", exact_audit),
    )
    assert report["epoch_layout"]["singleton_env_outer_batches"] is True
    assert report["target_scope_epoch_context"]["target_step_fraction_of_epoch"] == 0.25
    assert report["target_scope_epoch_context"]["uniform_step_weight_epoch_share_heuristic"] == 0.0625
    assert report["conclusions"]["cross_batch_overwrite_is_a_more_plausible_bottleneck_than_intra_batch_mixing"] is True


def test_batch_to_epoch_probe_rejects_missing_target_batch(tmp_path: Path) -> None:
    selector_trace = {
        "rows": [
            {
                "outer_batch_idx": 1,
                "total_outer_batches": 1,
                "objective_total": 192,
                "objective_env_indices_present": [0],
                "target_match_count": 0,
                "target_env_objective_episode_spans": [],
            },
        ]
    }
    exact_audit = {
        "target_scope": {
            "selector": {
                "outer_batch_idx": 2,
                "env_index": 1,
                "objective_episode_index": 10,
                "objective_position_start": 0,
                "objective_position_end": 7,
            }
        },
        "stage_transition": {
            "target_share_of_batch_negative_mass": {
                "policy_num": 0.25,
            }
        },
    }

    try:
        build_pair2_phase2_scale_batch_to_epoch_aggregation_probe(
            selector_trace_json=_write_json(tmp_path, "selector_trace.json", selector_trace),
            exact_audit_json=_write_json(tmp_path, "exact_audit.json", exact_audit),
        )
    except ValueError as exc:
        assert "expected exactly one target outer batch row" in str(exc)
    else:
        raise AssertionError("expected ValueError")
