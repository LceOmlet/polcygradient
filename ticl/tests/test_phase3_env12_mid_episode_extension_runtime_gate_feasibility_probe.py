import json
import sys

from ticl.analysis.phase3_env12_mid_episode_extension_runtime_gate_feasibility_probe import (
    build_env12_mid_episode_extension_runtime_gate_feasibility_probe,
)


def _write(tmp_path, name: str, payload: dict) -> str:
    path = tmp_path / name
    path.write_text(json.dumps(payload))
    return str(path)


def test_build_env12_mid_episode_extension_runtime_gate_feasibility_probe_reports_no_exact_builtin_mode(tmp_path):
    runtime_control = _write(
        tmp_path,
        "runtime_control.json",
        {
            "audit_entry": "phase3_env12_mid_episode_extension_runtime_control_probe",
            "control_block": {
                "candidate_small_control_target": "env12_mid_episode_extension_block",
                "runtime_control_granularity": "contiguous_four_token_block",
                "branch_specific": True,
                "row_count": 4,
                "local_indices": [18, 19, 20, 21],
                "tokens_to_objective_episode_end": [38, 37, 36, 35],
                "future_chain_step_count": [38, 37, 36, 35],
                "dominant_source": "baseline_bootstrap_semantic_mismatch",
                "negative_source_shares_reference": {
                    "baseline_bootstrap_semantic_mismatch": 0.5491065753155423,
                    "terminal_reset_tail_semantic_mismatch": 0.45089342468445764,
                },
            },
            "runtime_semantic_checks": {
                "payloads_match": True,
                "can_be_minimally_realized_as_runtime_control": True,
                "requires_shared_core_edit": False,
                "requires_branch_specific_scope_only": True,
                "shared_guardrail_unchanged": True,
                "no_finer_split_supported": True,
                "branch_specific_followup_needed": True,
            },
            "material_counterfactual": {
                "mean_start_gae_future_carry_norm": {
                    "abs_shrink": 0.02045608926564456,
                    "counterfactual_abs": 0.06972909942269324,
                    "counterfactual_over_observed_ratio": 0.7731768424154796,
                    "observed_abs": 0.0901851886883378,
                    "observed_over_counterfactual_ratio": 1.2933651722882735,
                }
            },
        },
    )

    report = build_env12_mid_episode_extension_runtime_gate_feasibility_probe(
        runtime_control_json=runtime_control
    )

    assert report["audit_entry"] == "phase3_env12_mid_episode_extension_runtime_gate_feasibility_probe"
    assert report["conclusions"]["exact_existing_actor_objective_mode_support_absent"] is True
    assert report["conclusions"]["runtime_gate_feasible_without_shared_core_edit"] is True
    assert report["conclusions"]["runtime_gate_requires_new_actor_objective_mode"] is True
    assert report["conclusions"]["runtime_gate_stays_branch_specific"] is True
    assert report["recommendation"]["runtime_gate_kind"] == "new_branch_specific_actor_objective_mode"


def test_build_env12_mid_episode_extension_runtime_gate_feasibility_probe_rejects_wrong_input(tmp_path):
    bad_runtime_control = _write(
        tmp_path,
        "bad.json",
        {"audit_entry": "wrong", "control_block": {}},
    )
    try:
        build_env12_mid_episode_extension_runtime_gate_feasibility_probe(
            runtime_control_json=bad_runtime_control
        )
    except ValueError as exc:
        assert "Expected phase3_env12_mid_episode_extension_runtime_control_probe artifact." in str(exc)
    else:
        raise AssertionError("Expected artifact validation to fail.")


def test_phase3_env12_mid_episode_extension_runtime_gate_feasibility_probe_reuses_existing_output(
    tmp_path, monkeypatch, capsys
):
    output_path = tmp_path / "phase3_env12_mid_episode_extension_runtime_gate_feasibility_probe.json"
    payload = {"ok": True, "kind": "phase3_env12_mid_episode_extension_runtime_gate_feasibility_probe"}
    output_path.write_text(json.dumps(payload))

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "phase3_env12_mid_episode_extension_runtime_gate_feasibility_probe.py",
            "--output-json",
            str(output_path),
        ],
    )

    from ticl.analysis import phase3_env12_mid_episode_extension_runtime_gate_feasibility_probe as probe_mod

    assert probe_mod.main() == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == payload
