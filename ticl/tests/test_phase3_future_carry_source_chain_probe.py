import json
import sys

from ticl.analysis import phase3_future_carry_source_chain_probe as probe_mod


def test_classify_negative_source_separates_true_negative_vs_terminal_vs_baseline():
    assert (
        probe_mod._classify_negative_source(
            {"raw_rollout_residual_raw": -0.1, "next_non_terminal": 1.0}
        )
        == "true_negative_future_residual"
    )
    assert (
        probe_mod._classify_negative_source(
            {"raw_rollout_residual_raw": 0.1, "next_non_terminal": 0.0}
        )
        == "terminal_reset_tail_semantic_mismatch"
    )
    assert (
        probe_mod._classify_negative_source(
            {"raw_rollout_residual_raw": 0.1, "next_non_terminal": 1.0}
        )
        == "baseline_bootstrap_semantic_mismatch"
    )


def test_trace_start_source_chain_reconstructs_future_carry_and_preserves_horizon_vs_terminal_split():
    effective_factor = 0.95
    episode_rows = [
        {
            "suite_name": "pair2",
            "env_index": 12,
            "objective_episode_index": 0,
            "objective_local_index": 0,
            "global_step": 100,
            "token_in_objective_episode": 0,
            "tokens_to_objective_episode_end": 3,
            "raw_rollout_residual_norm": 0.8,
            "raw_rollout_residual_raw": 8.0,
            "delta_norm": 0.05,
            "gae_future_carry_norm": 0.95 * (-0.1) + (0.95**2) * (-0.2) + (0.95**3) * 0.03,
            "next_non_terminal": 1.0,
            "next_step_is_objective": True,
            "steps_until_next_objective": 1,
            "actor_adv_norm": -0.2155375,
            "flip_cause": "negative_gae_future_carry",
        },
        {
            "suite_name": "pair2",
            "env_index": 12,
            "objective_episode_index": 0,
            "objective_local_index": 1,
            "global_step": 101,
            "token_in_objective_episode": 1,
            "tokens_to_objective_episode_end": 2,
            "raw_rollout_residual_norm": 0.7,
            "raw_rollout_residual_raw": 7.0,
            "delta_norm": -0.1,
            "gae_future_carry_norm": (0.95) * (-0.2) + (0.95**2) * 0.03,
            "next_non_terminal": 1.0,
            "next_step_is_objective": True,
            "steps_until_next_objective": 1,
            "actor_adv_norm": -0.172925,
            "flip_cause": "bootstrap_term_negative",
        },
        {
            "suite_name": "pair2",
            "env_index": 12,
            "objective_episode_index": 0,
            "objective_local_index": 2,
            "global_step": 102,
            "token_in_objective_episode": 2,
            "tokens_to_objective_episode_end": 1,
            "raw_rollout_residual_norm": 0.6,
            "raw_rollout_residual_raw": 6.0,
            "delta_norm": -0.2,
            "gae_future_carry_norm": (0.95) * 0.03,
            "next_non_terminal": 0.0,
            "next_step_is_objective": True,
            "steps_until_next_objective": 1,
            "actor_adv_norm": -0.1715,
            "flip_cause": "bootstrap_term_negative",
        },
        {
            "suite_name": "pair2",
            "env_index": 12,
            "objective_episode_index": 0,
            "objective_local_index": 3,
            "global_step": 103,
            "token_in_objective_episode": 3,
            "tokens_to_objective_episode_end": 0,
            "raw_rollout_residual_norm": 0.5,
            "raw_rollout_residual_raw": 5.0,
            "delta_norm": 0.03,
            "gae_future_carry_norm": 0.0,
            "next_non_terminal": 1.0,
            "next_step_is_objective": False,
            "steps_until_next_objective": -1,
            "actor_adv_norm": 0.03,
            "flip_cause": "not_flip",
        },
    ]

    summary = probe_mod._trace_start_source_chain(
        start_row=episode_rows[0],
        episode_rows=episode_rows,
        effective_factor=effective_factor,
    )
    assert abs(summary["weighted_delta_sum"] - episode_rows[0]["gae_future_carry_norm"]) < 1e-12
    assert summary["dominant_negative_source"] == "terminal_reset_tail_semantic_mismatch"
    assert abs(summary["negative_source_contributions"]["baseline_bootstrap_semantic_mismatch"] - 0.095) < 1e-12
    assert abs(summary["negative_source_contributions"]["terminal_reset_tail_semantic_mismatch"] - 0.1805) < 1e-12
    assert abs(summary["positive_offset_contribution_total"] - (0.95**3) * 0.03) < 1e-12


def test_phase3_future_carry_source_chain_probe_reuses_existing_output(tmp_path, monkeypatch, capsys):
    output_path = tmp_path / "future_carry_source_chain.json"
    payload = {"ok": True, "kind": "future_carry_source_chain"}
    output_path.write_text(json.dumps(payload))

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "phase3_future_carry_source_chain_probe.py",
            "--output-json",
            str(output_path),
        ],
    )

    assert probe_mod.main() == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == payload
