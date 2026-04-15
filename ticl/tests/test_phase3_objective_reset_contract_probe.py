import json
import sys

from ticl.analysis import phase3_objective_reset_contract_probe as probe_mod
from ticl.analysis import phase3_residual_anchor_gae_path_probe as anchor_probe_mod


def test_summarize_objective_reset_contract_detects_segment_without_terminal_row():
    objective_mask = [0, 1, 1, 0, 1, 1, 0]
    episode_starts = [0, 0, 0, 0, 1, 0, 0]
    result = probe_mod._summarize_objective_reset_contract(
        objective_mask=objective_mask,
        episode_starts=episode_starts,
        last_done=False,
    )
    assert result["objective_token_count"] == 4
    assert result["compressed_segments"] == [[0, 2], [2, 4]]
    assert result["objective_terminal_reset_count_from_segments"] == 1
    assert result["episode_starts_on_objective_positions"] == [2]
    assert result["terminal_row_count_within_objective"] == 0
    assert result["terminal_rows_within_objective_global_steps"] == []


def test_summarize_objective_reset_contract_detects_in_objective_terminal_row():
    objective_mask = [0, 0, 1, 1, 1]
    episode_starts = [0, 0, 0, 0, 0]
    result = probe_mod._summarize_objective_reset_contract(
        objective_mask=objective_mask,
        episode_starts=episode_starts,
        last_done=True,
    )
    assert result["objective_token_count"] == 3
    assert result["compressed_segments"] == [[0, 3]]
    assert result["objective_terminal_reset_count_from_segments"] == 0
    assert result["terminal_row_count_within_objective"] == 1
    assert result["terminal_rows_within_objective_global_steps"] == [4]


def test_objective_reset_contract_probe_reuses_existing_output(tmp_path, monkeypatch, capsys):
    output_path = tmp_path / "objective_reset_contract_probe.json"
    payload = {"ok": True, "kind": "objective_reset_contract_probe"}
    output_path.write_text(json.dumps(payload))

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "phase3_objective_reset_contract_probe.py",
            "dummy.ckpt",
            "--suite-name",
            "pair1",
            "--env-index",
            "12",
            "--output-json",
            str(output_path),
        ],
    )

    assert probe_mod.main() == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == payload


def test_fixed_suite_collect_contract_kwargs_exposes_strict_seeded_collect_mode():
    suite = {
        "env_seeds": [11, 22],
        "rollout_seeds": [33, 44],
    }
    result = anchor_probe_mod._fixed_suite_collect_contract_kwargs(suite)
    assert result == {
        "strict_fixed_env_mode": True,
        "env_rng_seeds": [11, 22],
        "rollout_rng_seeds": [33, 44],
        "deterministic_batch_plan": True,
    }
