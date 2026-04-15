import json
from pathlib import Path

from ticl.analysis.phase3_pair2_env12_window_reset_segment_compare import (
    _extract_source_window_rows,
    _resolve_compare_suite_path,
    _segment_boundary_steps,
    _source_expected_compare_suite_role,
    _summarize_window_segments,
)


def test_extract_source_window_rows_filters_env_and_window(tmp_path):
    payload = {
        "token_row_bundles": [
            {
                "suite_name": "pair1",
                "env_index": 11,
                "token_rows": [
                    {
                        "global_step": 1946,
                        "objective_episode_index": 7,
                        "objective_local_index": 0,
                        "tokens_to_objective_episode_end": 2,
                    }
                ],
            },
            {
                "suite_name": "pair2",
                "env_index": 12,
                "token_rows": [
                    {
                        "global_step": 1945,
                        "objective_episode_index": 0,
                        "objective_local_index": 99,
                        "tokens_to_objective_episode_end": 3,
                    },
                    {
                        "global_step": 1946,
                        "objective_episode_index": 0,
                        "objective_local_index": 0,
                        "tokens_to_objective_episode_end": 2,
                    },
                    {
                        "global_step": 1947,
                        "objective_episode_index": 0,
                        "objective_local_index": 1,
                        "tokens_to_objective_episode_end": 1,
                    },
                    {
                        "global_step": 1948,
                        "objective_episode_index": 1,
                        "objective_local_index": 0,
                        "tokens_to_objective_episode_end": 4,
                    },
                ],
            },
        ]
    }
    source_json = tmp_path / "source.json"
    source_json.write_text(json.dumps(payload))

    rows = _extract_source_window_rows(
        source_json=str(source_json),
        source_suite_name="pair2",
        env_index=12,
        global_step_start=1946,
        global_step_end=1947,
    )

    assert rows == [
        {
            "global_step": 1946,
            "objective_episode_index": 0,
            "objective_local_index": 0,
            "tokens_to_objective_episode_end": 2,
        },
        {
            "global_step": 1947,
            "objective_episode_index": 0,
            "objective_local_index": 1,
            "tokens_to_objective_episode_end": 1,
        },
    ]


def test_summarize_window_segments_splits_on_episode_or_noncontiguous_local():
    rows = [
        {"global_step": 1946, "objective_episode_index": 0, "objective_local_index": 0},
        {"global_step": 1947, "objective_episode_index": 0, "objective_local_index": 1},
        {"global_step": 1948, "objective_episode_index": 0, "objective_local_index": 3},
        {"global_step": 1949, "objective_episode_index": 1, "objective_local_index": 0},
        {"global_step": 1950, "objective_episode_index": 1, "objective_local_index": 1},
    ]

    segments = _summarize_window_segments(
        rows,
        episode_key="objective_episode_index",
        local_key="objective_local_index",
    )

    assert segments == [
        {
            "episode_index": 0,
            "global_step_start": 1946,
            "global_step_end": 1947,
            "local_index_start": 0,
            "local_index_end": 1,
            "token_count": 2,
        },
        {
            "episode_index": 0,
            "global_step_start": 1948,
            "global_step_end": 1948,
            "local_index_start": 3,
            "local_index_end": 3,
            "token_count": 1,
        },
        {
            "episode_index": 1,
            "global_step_start": 1949,
            "global_step_end": 1950,
            "local_index_start": 0,
            "local_index_end": 1,
            "token_count": 2,
        },
    ]
    assert _segment_boundary_steps(segments) == [1948, 1949]


def test_extract_source_window_rows_raises_if_env_missing(tmp_path):
    source_json = tmp_path / "source.json"
    source_json.write_text(
        json.dumps({"token_row_bundles": [{"suite_name": "pair1", "env_index": 1, "token_rows": []}]})
    )

    try:
        _extract_source_window_rows(
            source_json=str(source_json),
            source_suite_name="pair2",
            env_index=12,
            global_step_start=1946,
            global_step_end=2002,
        )
    except ValueError as exc:
        assert "suite_name='pair2'" in str(exc)
        assert "env_index=12" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing env bundle")


def test_source_expected_compare_suite_role_is_heldout_for_residual_anchor_probe(tmp_path):
    source_json = tmp_path / "source.json"
    source_json.write_text(json.dumps({"audit_entry": "phase3_residual_anchor_gae_path_probe"}))

    assert _source_expected_compare_suite_role(str(source_json)) == "heldout"


def test_resolve_compare_suite_path_defaults_to_heldout_for_residual_anchor_source(tmp_path):
    source_json = tmp_path / "source.json"
    source_json.write_text(json.dumps({"audit_entry": "phase3_residual_anchor_gae_path_probe"}))

    path, expected_role, actual_role = _resolve_compare_suite_path(
        compare_suite_path=None,
        source_json=str(source_json),
        allow_source_suite_mismatch=False,
    )

    assert path.endswith("/heldout_suite.pt")
    assert expected_role == "heldout"
    assert actual_role == "heldout"


def test_resolve_compare_suite_path_rejects_train_suite_for_heldout_source(tmp_path):
    source_json = tmp_path / "source.json"
    source_json.write_text(json.dumps({"audit_entry": "phase3_residual_anchor_gae_path_probe"}))

    try:
        _resolve_compare_suite_path(
            compare_suite_path="/tmp/train_suite.pt",
            source_json=str(source_json),
            allow_source_suite_mismatch=False,
        )
    except ValueError as exc:
        assert "expects compare suite role 'heldout'" in str(exc)
        assert "role 'train'" in str(exc)
    else:
        raise AssertionError("Expected suite-role mismatch to raise")
