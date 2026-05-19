import json
from pathlib import Path

from scripts.exploratory import phase2_pendulum_settle_state_blocker_decomposition as audit


def _row(**kwargs):
    row = {
        "reward_condition": True,
        "suffix_reward_condition": True,
        "action_decay_condition": True,
        "best_decay_state_drift_prefix": 1.0,
        "best_decay_state_drift_suffix": 0.99,
        "best_open_state_drift_suffix": 0.96,
    }
    row.update(kwargs)
    return row


def _write_preflight(path: Path):
    lists = []
    for label in ("full", "q90"):
        rows = [_row() for _ in range(8)]
        lists.append(
            {
                "label": label,
                "classification": {
                    "coverage": {
                        "n_envs": 8,
                        "reward_condition_count": 8,
                        "suffix_reward_condition_count": 8,
                        "action_decay_condition_count": 8,
                        "self_state_settle_condition_count": 0,
                        "open_state_settle_condition_count": 0,
                        "cheap_settle_guard_count": 0,
                    },
                    "scores": {
                        "best_decay_suffix_drift_over_prefix_drift": {"mean": 0.99},
                        "best_decay_suffix_drift_over_open_suffix_drift": {"mean": 1.03},
                    },
                    "per_env": rows,
                },
            }
        )
    path.write_text(
        json.dumps(
            {
                "decision": {"cheap_suffix_state_settle_guard_ready": False},
                "lists": lists,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_decomposition_rejects_loose_non_worsening_as_semantic_closure(tmp_path: Path):
    path = _write_preflight(tmp_path / "preflight.json")

    report = audit.build_report(preflight_json=path, min_count=8)

    assert report["decision"]["dominant_blocker_is_state_basin"] is True
    assert report["decision"]["strict_contraction_count_safe_under_grid"] is False
    assert report["decision"]["loose_non_worsening_count_safe_under_grid"] is True
    assert report["decision"]["selector_threshold_repair_allowed"] is False
    assert report["decision"]["loose_guard_not_semantically_sufficient"] is True
