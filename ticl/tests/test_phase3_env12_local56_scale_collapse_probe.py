import json
import sys

from ticl.analysis import phase3_env12_local56_scale_collapse_probe as probe_mod


def test_build_scale_collapse_probe_from_minimal_payloads():
    mean_source_payload = {
        "audit_entry": "phase3_env12_local56_mean_source_probe",
        "config": {
            "suite_name": "pair2",
            "env_index": 12,
        },
        "full_mean_decomposition": {
            "n_samples": 10,
            "objective_token_count": 4,
            "prefix_token_count_before_objective": 6,
            "objective_mean_rollout_return_raw": 5.0,
            "implied_prefix_mean_rollout_return_raw": 13.333333333333334,
        },
        "objective_episode_summaries": [
            {
                "objective_episode_index": 0,
                "token_count": 2,
                "mean_rollout_return_raw": 7.0,
            },
            {
                "objective_episode_index": 1,
                "token_count": 2,
                "mean_rollout_return_raw": 3.0,
            },
        ],
    }
    scope_payload = {
        "audit_entry": "phase3_env12_local56_boundary_scope_counterfactual_probe",
        "scope_rows": [
            {
                "scope_name": "current_full_rollout",
                "token_count": 10,
                "mean_raw": 10.0,
                "std_raw": 15.0,
            },
            {
                "scope_name": "objective_all_tokens",
                "token_count": 4,
                "mean_raw": 5.0,
                "std_raw": 2.23606797749979,
            },
            {
                "scope_name": "objective_episode0_only",
                "token_count": 2,
                "mean_raw": 7.0,
                "std_raw": 1.0,
            },
            {
                "scope_name": "objective_episode1_only",
                "token_count": 2,
                "mean_raw": 3.0,
                "std_raw": 1.5,
            },
        ],
    }

    result = probe_mod._build_probe(mean_source_payload, scope_payload)

    assert result["audit_entry"] == "phase3_env12_local56_scale_collapse_probe"
    assert result["full_variance_decomposition"]["dominant_full_variance_source"] == "within_prefix"
    assert result["objective_variance_decomposition"]["dominant_objective_variance_source"] == "between_episode_means"
    assert result["conclusions"]["objective_scope_std_collapse_is_explained_by_truncating_away_high_variance_prefix"] is True
    assert result["conclusions"]["objective_window_is_low_variance_slice_not_high_variance_source"] is True
    assert result["conclusions"]["episode0_and_episode1_are_each_more_scale_collapsed_than_objective_all"] is True
    assert result["conclusions"]["remaining_objective_variance_is_mostly_between_episode_means"] is True
    assert result["conclusions"]["episode1_only_is_most_scale_collapsed_objective_subscope"] is False


def test_phase3_env12_local56_scale_collapse_probe_reuses_existing_output(tmp_path, monkeypatch, capsys):
    output_path = tmp_path / "env12_local56_scale_collapse_probe.json"
    payload = {"ok": True, "kind": "env12_local56_scale_collapse_probe"}
    output_path.write_text(json.dumps(payload))

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "phase3_env12_local56_scale_collapse_probe.py",
            "--output-json",
            str(output_path),
        ],
    )

    assert probe_mod.main() == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == payload
