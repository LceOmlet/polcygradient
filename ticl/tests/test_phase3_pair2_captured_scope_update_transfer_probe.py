from pathlib import Path

import pytest

from ticl.analysis.phase3_pair2_captured_scope_update_transfer_probe import (
    build_pair2_captured_scope_update_transfer_probe,
)


def _write_exact_audit(tmp_path: Path, payload: dict) -> str:
    path = tmp_path / "exact_audit.json"
    path.write_text(__import__("json").dumps(payload, indent=2, sort_keys=True))
    return str(path)


def test_captured_scope_probe_uses_single_available_segment_and_reports_no_change(tmp_path: Path) -> None:
    payload = {
        "strict_contract": {
            "outer_batch_idx": 8,
            "epoch_idx": 1,
            "batch_size_flat": 256,
            "objective_total": 4,
            "normalize_advantage": True,
            "runtime_suite_name": "pair2",
            "baseline_actor_objective_mode": "tokenwise",
            "override_actor_objective_mode": "branch_mode",
        },
        "available_snapshot_scope": {
            "objective_segments": [
                {
                    "env_index": 0,
                    "objective_episode_index": 2,
                    "objective_position_min": 0,
                    "objective_position_max": 3,
                    "token_count": 4,
                }
            ]
        },
        "baseline": {
            "pre_normalization_actor_advantages": {
                "whole_batch": {
                    "token_count": 4,
                    "signed_sum": -1.0,
                    "abs_mass": 1.0,
                    "positive_mass": 0.0,
                    "negative_mass": 1.0,
                    "mean": -0.25,
                }
            },
            "post_normalization_advantages": {
                "whole_batch": {
                    "token_count": 4,
                    "signed_sum": 0.0,
                    "abs_mass": 2.0,
                    "positive_mass": 1.0,
                    "negative_mass": 1.0,
                    "mean": 0.0,
                }
            },
            "policy_num_from_clipped_objective": {
                "whole_batch": {
                    "token_count": 4,
                    "signed_sum": 0.1,
                    "abs_mass": 1.5,
                    "positive_mass": 0.8,
                    "negative_mass": 0.7,
                    "mean": 0.025,
                }
            },
            "ratio_and_clipping": {
                "whole_batch_clip_active_fraction": 0.5,
                "whole_batch_ratio_mean": 0.99,
            },
        },
        "override": {
            "pre_normalization_actor_advantages": {
                "whole_batch": {
                    "token_count": 4,
                    "signed_sum": -1.0,
                    "abs_mass": 1.0,
                    "positive_mass": 0.0,
                    "negative_mass": 1.0,
                    "mean": -0.25,
                }
            },
            "post_normalization_advantages": {
                "whole_batch": {
                    "token_count": 4,
                    "signed_sum": 0.0,
                    "abs_mass": 2.0,
                    "positive_mass": 1.0,
                    "negative_mass": 1.0,
                    "mean": 0.0,
                }
            },
            "policy_num_from_clipped_objective": {
                "whole_batch": {
                    "token_count": 4,
                    "signed_sum": 0.1,
                    "abs_mass": 1.5,
                    "positive_mass": 0.8,
                    "negative_mass": 0.7,
                    "mean": 0.025,
                }
            },
            "ratio_and_clipping": {
                "whole_batch_clip_active_fraction": 0.5,
                "whole_batch_ratio_mean": 0.99,
            },
        },
    }
    report = build_pair2_captured_scope_update_transfer_probe(
        exact_update_aggregation_audit_json=_write_exact_audit(tmp_path, payload)
    )
    assert report["available_snapshot_scope"]["selected_scope"]["env_index"] == 0
    assert report["available_snapshot_scope"]["selected_scope_equals_whole_batch_objective"] is True
    assert report["conclusions"]["current_snapshot_is_not_the_transfer_site_for_env12_branch_specific_gain"] is True
    assert report["conclusions"]["captured_scope_pre_stage_changed"] is False
    assert report["conclusions"]["captured_scope_policy_stage_changed"] is False


def test_captured_scope_probe_rejects_non_whole_batch_scope(tmp_path: Path) -> None:
    payload = {
        "strict_contract": {
            "outer_batch_idx": 8,
            "epoch_idx": 1,
            "batch_size_flat": 256,
            "objective_total": 6,
            "normalize_advantage": True,
            "runtime_suite_name": "pair2",
            "baseline_actor_objective_mode": "tokenwise",
            "override_actor_objective_mode": "branch_mode",
        },
        "available_snapshot_scope": {
            "objective_segments": [
                {
                    "env_index": 0,
                    "objective_episode_index": 2,
                    "objective_position_min": 0,
                    "objective_position_max": 3,
                    "token_count": 4,
                },
                {
                    "env_index": 1,
                    "objective_episode_index": 0,
                    "objective_position_min": 0,
                    "objective_position_max": 1,
                    "token_count": 2,
                },
            ]
        },
        "baseline": {
            stage_name: {
                "whole_batch": {
                    "token_count": 6,
                    "signed_sum": 0.0,
                    "abs_mass": 1.0,
                    "positive_mass": 0.5,
                    "negative_mass": 0.5,
                    "mean": 0.0,
                }
            }
            for stage_name in (
                "pre_normalization_actor_advantages",
                "post_normalization_advantages",
                "policy_num_from_clipped_objective",
            )
        },
        "override": {
            stage_name: {
                "whole_batch": {
                    "token_count": 6,
                    "signed_sum": 0.0,
                    "abs_mass": 1.0,
                    "positive_mass": 0.5,
                    "negative_mass": 0.5,
                    "mean": 0.0,
                }
            }
            for stage_name in (
                "pre_normalization_actor_advantages",
                "post_normalization_advantages",
                "policy_num_from_clipped_objective",
            )
        },
    }
    payload["baseline"]["ratio_and_clipping"] = {
        "whole_batch_clip_active_fraction": 0.0,
        "whole_batch_ratio_mean": 1.0,
    }
    payload["override"]["ratio_and_clipping"] = {
        "whole_batch_clip_active_fraction": 0.0,
        "whole_batch_ratio_mean": 1.0,
    }

    with pytest.raises(ValueError, match="whole-batch objective"):
        build_pair2_captured_scope_update_transfer_probe(
            exact_update_aggregation_audit_json=_write_exact_audit(tmp_path, payload)
        )
