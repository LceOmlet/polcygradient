import json
import sys

from ticl.analysis import phase3_pair2_current_contract_dual_channel_family_probe as probe_mod


def test_pair2_current_contract_dual_channel_family_probe_reuses_existing_output(
    tmp_path, monkeypatch, capsys
):
    output_path = tmp_path / "pair2_dual_channel_probe.json"
    payload = {"ok": True, "kind": "pair2_dual_channel"}
    output_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "phase3_pair2_current_contract_dual_channel_family_probe.py",
            "--output-json",
            str(output_path),
        ],
    )

    assert probe_mod.main() == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == payload


def test_candidate_metrics_tracks_cross_env_family_mass():
    objective_mask = [1, 1, 1, 1, 1, 1]
    env_indices = [12, 12, 0, 0, 13, 15]
    objective_episode_indices = [4, 7, 0, 0, 0, 0]
    raw_actor_advantages = [-2.0, -3.0, -1.0, 2.0, -4.0, -5.0]
    mask = [1, 0, 1, 0, 1, 1]

    metrics = probe_mod._candidate_metrics(
        name="candidate",
        description="synthetic",
        mask=mask,
        objective_mask=objective_mask,
        env_indices=env_indices,
        objective_episode_indices=objective_episode_indices,
        raw_actor_advantages=raw_actor_advantages,
    )

    assert metrics["token_count"] == 4
    assert metrics["env_indices"] == [0, 12, 13, 15]
    assert metrics["episode_indices"] == [0, 4]
    assert abs(metrics["raw_negative_mass"] - 12.0) < 1e-8
    assert abs(metrics["raw_positive_mass"] - 0.0) < 1e-8
    assert 0.0 < metrics["raw_negative_share_of_batch"] <= 1.0


def test_summarize_recovered_anchor_first_objective_non_tail_outer_batches_detects_split_family():
    objective_mask = [1] * 12
    env_indices = [15, 15, 1, 1, 15, 15, 0, 0, 2, 2, 0, 0]
    objective_episode_indices = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
    objective_episode_positions = [0, 1, 0, 1, 2, 3, 0, 1, 0, 1, 4, 5]
    raw_actor_advantages = [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, 1.0, 1.0, 1.0, 1.0, -7.0, -8.0]

    summary = probe_mod._summarize_recovered_anchor_first_objective_non_tail_outer_batches(
        objective_mask=objective_mask,
        env_indices=env_indices,
        objective_episode_indices=objective_episode_indices,
        objective_episode_positions=objective_episode_positions,
        raw_actor_advantages=raw_actor_advantages,
        target_env_position_end_by_env={0: 4, 15: 3},
        batch_size=6,
    )

    assert summary["family_spans_multiple_outer_batches"] is True
    assert summary["single_outer_batch_covers_full_family"] is False
    assert summary["family_is_split_across_distinct_env_local_outer_batches"] is True
    assert summary["recommended_snapshot_selector"] == {
        "target_outer_batch_idx": 1,
        "target_env_index": 15,
        "target_objective_episode_index": 0,
        "target_objective_position_start": 0,
        "target_objective_position_end": 3,
    }
    assert [row["outer_batch_idx"] for row in summary["rows"]] == [1, 2]
