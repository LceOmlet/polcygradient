import json

from ticl.analysis.phase3_pair2_branch_specific_family_candidate_probe import (
    build_pair2_branch_specific_family_candidate_probe,
)


def _write(tmp_path, name: str, payload: dict) -> str:
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def test_pair2_branch_specific_family_candidate_probe_recommends_high_mass_family(tmp_path):
    baseline_snapshot_json = _write(
        tmp_path,
        "baseline_snapshot.json",
        {
            "objective_mask": [1] * 22,
            "flat_env_indices": [12] * 22,
            "flat_objective_episode_indices": [
                4, 4, 4, 4, 4, 4,
                7, 7, 7,
                11, 11, 11, 11,
                14, 14, 14, 14, 14,
                3, 5, 6, 12,
            ],
            "flat_objective_episode_positions": [
                0, 1, 2, 3, 15, 16,
                0, 1, 15,
                0, 1, 15, 18,
                0, 1, 15, 16, 17,
                0, 0, 0, 0,
            ],
            "clipped_objective": [
                -1.0, -1.0, -0.5, -0.5, -1.0, -1.0,
                -2.0, -2.0, -0.2,
                -1.5, -1.5, -0.2, -0.2,
                -0.01, 0.1, 0.2, 0.3, 0.4,
                -0.5, -0.4, -0.3, -0.2,
            ],
        },
    )
    scope_probe_json = _write(
        tmp_path,
        "scope_probe.json",
        {
            "probe_entry": "phase3_pair2_branch_specific_extension_class_probe",
            "current_trusted_contract": {
                "target_outer_batch_idx": 13,
                "total_outer_batches": 16,
            },
        },
    )

    report = build_pair2_branch_specific_family_candidate_probe(
        baseline_snapshot_json=baseline_snapshot_json,
        scope_class_probe_json=scope_probe_json,
    )

    assert report["conclusions"]["motif_family_is_not_the_next_larger_candidate"] is True
    assert report["conclusions"]["high_mass_family_is_meaningfully_larger_than_episode_complete"] is True
    assert report["conclusions"]["pretail_family_is_too_close_to_whole_batch_for_next_small_control"] is True
    assert report["recommended_family"]["candidate_small_control_target"] == "env12_high_mass_objective_episode_family"
    assert report["recommended_family"]["locked_objective_episode_indices"] == [4, 7, 11]


def test_pair2_branch_specific_family_candidate_probe_rejects_wrong_scope_probe(tmp_path):
    baseline_snapshot_json = _write(
        tmp_path,
        "baseline_snapshot.json",
        {
            "objective_mask": [1],
            "flat_env_indices": [12],
            "flat_objective_episode_indices": [4],
            "flat_objective_episode_positions": [0],
            "clipped_objective": [-1.0],
        },
    )
    scope_probe_json = _write(tmp_path, "scope_probe.json", {"probe_entry": "wrong"})

    try:
        build_pair2_branch_specific_family_candidate_probe(
            baseline_snapshot_json=baseline_snapshot_json,
            scope_class_probe_json=scope_probe_json,
        )
    except ValueError as exc:
        assert "Expected canonical branch-specific extension class probe artifact" in str(exc)
    else:
        raise AssertionError("expected wrong scope probe to raise")
