import json

import pytest

from ticl.analysis.phase3_subset_delta_fallback import resolve_heldout_subset_deltas


def test_resolve_heldout_subset_deltas_derives_from_reused_metrics():
    zero_metrics = {
        "heldout": {
            "suffix_return_per_env": [1.0, 2.0],
            "full_return_per_env": [10.0, 20.0],
        }
    }
    pre_metrics = {
        "heldout": {
            "suffix_return_per_env": [1.5, 1.0],
            "full_return_per_env": [12.0, 18.0],
        }
    }

    out = resolve_heldout_subset_deltas(
        reuse_heldout_subset_delta_json=None,
        zero_metrics=zero_metrics,
        pre_metrics=pre_metrics,
        reuse_zero_control_json=None,
        reuse_pre_policy_json=None,
        train_suite_summary={},
        heldout_suite_summary={},
        n_samples=2048,
        single_eval_pos=1946,
        rollout_backend="serial",
    )

    assert out[0]["pre_vs_zero_suffix_gap"] == 0.5
    assert out[1]["pre_vs_zero_suffix_gap"] == -1.0
    assert out[0]["pre_vs_zero_full_gap"] == 2.0
    assert out[1]["pre_vs_zero_full_gap"] == -2.0


def test_resolve_heldout_subset_deltas_rejects_partial_subset_json(tmp_path):
    subset_path = tmp_path / "subset.json"
    subset_path.write_text(
        json.dumps(
            {
                "heldout_subset": {
                    "combined": {
                        "env_items": [
                            {
                                "env_index": 0,
                                "pre_vs_zero_suffix_gap": 0.5,
                                "post_vs_zero_suffix_gap": 1.0,
                                "suffix_return_delta": 0.0,
                            }
                        ]
                    }
                }
            }
        )
    )
    zero_metrics = {
        "heldout": {
            "suffix_return_per_env": [1.0, 2.0],
            "full_return_per_env": [10.0, 20.0],
        }
    }
    pre_metrics = {
        "heldout": {
            "suffix_return_per_env": [1.5, 1.0],
            "full_return_per_env": [12.0, 18.0],
        }
    }

    with pytest.raises(ValueError, match="does not cover the full heldout suite"):
        resolve_heldout_subset_deltas(
            reuse_heldout_subset_delta_json=str(subset_path),
            zero_metrics=zero_metrics,
            pre_metrics=pre_metrics,
            reuse_zero_control_json=None,
            reuse_pre_policy_json=None,
            train_suite_summary={},
            heldout_suite_summary={},
            n_samples=2048,
            single_eval_pos=1946,
            rollout_backend="serial",
        )
