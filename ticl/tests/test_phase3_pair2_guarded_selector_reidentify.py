import json
from pathlib import Path

from ticl.analysis.phase3_pair2_guarded_selector_reidentify import (
    build_pair2_guarded_selector_reidentified,
)


def _write_json(tmp_path: Path, name: str, payload: dict) -> str:
    path = tmp_path / name
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return str(path)


def test_build_pair2_guarded_selector_reidentified_maps_global_anchor_to_current_episode(tmp_path: Path) -> None:
    guarded_trace = {
        "probe_entry": "phase3_pair2_guarded_target_present_selector_trace",
        "target_selector": {
            "target_outer_batch_idx": None,
            "target_env_index": 12,
            "target_objective_episode_index": 0,
            "target_objective_position_start": 18,
            "target_objective_position_end": 21,
        },
        "baseline": {
            "selector_trace": {
                "env_present_rows": [
                    {
                        "epoch_idx": 1,
                        "outer_batch_idx": 13,
                        "target_status": "target_episode_absent",
                        "objective_env_indices_present": [12],
                        "target_env_objective_token_count": 192,
                        "target_env_objective_episode_indices_present": [3, 4, 5],
                        "target_env_objective_episode_spans": [
                            {
                                "objective_episode_index": 3,
                                "objective_position_min": 0,
                                "objective_position_max": 2,
                                "token_count": 3,
                            },
                            {
                                "objective_episode_index": 4,
                                "objective_position_min": 0,
                                "objective_position_max": 28,
                                "token_count": 29,
                            },
                            {
                                "objective_episode_index": 5,
                                "objective_position_min": 0,
                                "objective_position_max": 1,
                                "token_count": 2,
                            },
                        ],
                    }
                ]
            }
        },
    }
    source_compare_pack = {
        "audit_entry": "phase3_env12_mid_episode_extension_compare_pack",
        "comparison": {
            "extension_block": {
                "env_index": 12,
                "local_indices": [18, 19, 20, 21],
                "global_steps": [1964, 1965, 1966, 1967],
            }
        },
    }

    report = build_pair2_guarded_selector_reidentified(
        guarded_selector_trace_json=_write_json(tmp_path, "guarded_trace.json", guarded_trace),
        source_compare_pack_json=_write_json(tmp_path, "source_compare_pack.json", source_compare_pack),
    )

    assert report["probe_entry"] == "phase3_pair2_guarded_selector_reidentified"
    assert report["reidentified_selector"]["outer_batch_idx"] == 13
    assert report["reidentified_selector"]["target_env_index"] == 12
    assert report["reidentified_selector"]["target_objective_episode_index"] == 4
    assert report["reidentified_selector"]["target_objective_position_start"] == 15
    assert report["reidentified_selector"]["target_objective_position_end"] == 18
    assert [row["objective_global_position"] for row in report["reidentified_selector"]["mapped_rows"]] == [
        18,
        19,
        20,
        21,
    ]
    assert [row["objective_episode_position"] for row in report["reidentified_selector"]["mapped_rows"]] == [
        15,
        16,
        17,
        18,
    ]
    assert report["conclusions"]["reidentified_selector_requires_episode_shift"] is True
    assert report["conclusions"]["reidentified_selector_requires_position_shift"] is True
    assert report["conclusions"]["reidentified_selector_requires_outer_batch_lock"] is True


def test_build_pair2_guarded_selector_reidentified_rejects_anchor_split_across_multiple_episodes(
    tmp_path: Path,
) -> None:
    guarded_trace = {
        "probe_entry": "phase3_pair2_guarded_target_present_selector_trace",
        "target_selector": {
            "target_outer_batch_idx": None,
            "target_env_index": 12,
            "target_objective_episode_index": 0,
            "target_objective_position_start": 18,
            "target_objective_position_end": 21,
        },
        "baseline": {
            "selector_trace": {
                "env_present_rows": [
                    {
                        "epoch_idx": 1,
                        "outer_batch_idx": 13,
                        "target_status": "target_episode_absent",
                        "objective_env_indices_present": [12],
                        "target_env_objective_token_count": 192,
                        "target_env_objective_episode_indices_present": [3, 4],
                        "target_env_objective_episode_spans": [
                            {
                                "objective_episode_index": 3,
                                "objective_position_min": 0,
                                "objective_position_max": 18,
                                "token_count": 19,
                            },
                            {
                                "objective_episode_index": 4,
                                "objective_position_min": 0,
                                "objective_position_max": 2,
                                "token_count": 3,
                            },
                        ],
                    }
                ]
            }
        },
    }
    source_compare_pack = {
        "audit_entry": "phase3_env12_mid_episode_extension_compare_pack",
        "comparison": {
            "extension_block": {
                "env_index": 12,
                "local_indices": [18, 19, 20, 21],
                "global_steps": [1964, 1965, 1966, 1967],
            }
        },
    }

    try:
        build_pair2_guarded_selector_reidentified(
            guarded_selector_trace_json=_write_json(tmp_path, "guarded_trace.json", guarded_trace),
            source_compare_pack_json=_write_json(tmp_path, "source_compare_pack.json", source_compare_pack),
        )
    except ValueError as exc:
        assert "spans multiple guarded episodes" in str(exc)
    else:
        raise AssertionError("expected split semantic anchor to raise")
