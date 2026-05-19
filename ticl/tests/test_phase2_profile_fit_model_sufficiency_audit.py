import csv
import json
from pathlib import Path

import pytest

from scripts.exploratory import phase2_profile_fit_model_sufficiency_audit as audit


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"rule": rule, "env_seed": str(1000 + idx), "full_frozen_h_json": f"/tmp/{rule}_{idx}.json"}
        for rule in audit.RULES
        for idx in range(2)
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["rule", "env_seed", "full_frozen_h_json"])
        writer.writeheader()
        writer.writerows(rows)


def _source_payload(seed: int) -> dict:
    return {
        "sampling": {
            "seed": seed,
            "target_per_rule": 64,
            "raw_seen": 4096,
            "env_evaluated": 256,
        },
        "selected_count_by_rule": {rule: 64 for rule in audit.RULES},
    }


def _preflight_payload() -> dict:
    return {
        "contract": {"explicit_failure_no_unprofiled_fallback": True},
        "existing_evidence": {
            "broad_labels_ready": True,
            "feedback_support_ready": True,
            "selector": {"hard_pass": True},
        },
        "bounded_generator_plan": {
            "feedback_eval_budget": {"execution_blocked_by_default": False},
            "stage_plan": [
                {"name": "source_pool", "ready": True},
                {"name": "fixed_groups", "ready": True},
                {"name": "profile_selector", "ready": True},
            ],
        },
        "fit_model_2048_scale_projection": {
            "chunked_probe": {"memory_safe": True},
            "naive_all_at_once": {"memory_safe": False},
        },
    }


def _selector_payload() -> dict:
    return {
        "config": {"profile_version_id": "profile_band_online_feedback_support_prototype"},
        "rules": {
            rule: {
                "guard_pass": True,
                "profile_guard_pass": True,
                "selected_summary": {
                    "n": 16,
                    "joint_profile_active_cells": 8,
                    "joint_profile_dominant_rate": 0.25,
                    "joint_profile_entropy_norm": 0.7,
                    "locomotion_witness_rate": 0.375,
                    "low_energy_not_ctrl_only_rate": 0.5,
                    "sign_or_phase_sensitive_rate": 0.8125,
                    "state_conditioned_energy_injection_rate": 0.5,
                    "strict_feedback_rate": 0.5,
                },
            }
            for rule in audit.RULES
        },
    }


def _chunk_payload() -> dict:
    return {
        "chunk_count": 2,
        "contract": {"chunked_memory_bounded_label_path": True},
        "execution": {"executed_count": 8, "skipped": [], "skipped_count": 0, "selected_chunk_count": 2},
        "feedback_eval_budget": {"chunk_size_within_feedback_budget": True},
        "source_counts": {rule: 64 for rule in audit.RULES},
        "merged_outputs": {
            "locomotion": {"row_count": 128},
            "feedback_near_best": {"row_count": 128},
        },
    }


def _parity_payload(*, hard_pass: bool = True) -> dict:
    return {
        "hard_pass": hard_pass,
        "cases": [
            {
                "case": f"case:{rule}",
                "rule": rule,
                "n_envs": 2,
                "n_steps": 32,
                "hard_pass": hard_pass,
                "collection_parity": {"hard_pass": hard_pass},
                "update_parity": {
                    "hard_pass": hard_pass,
                    "gradient_diff": {"grad_delta_max_abs": 0.0},
                    "one_train_update_compare": {
                        "post_update_param_diff": {"policy_param_max_abs_diff": 0.0}
                    },
                },
            }
            for rule in audit.RULES
        ],
    }


def _write_fixture(tmp_path: Path, *, parity_pass: bool = True) -> dict[str, Path]:
    paths = {
        "current_source": tmp_path / "current/source.json",
        "ref_source": tmp_path / "ref/source.json",
        "preflight": tmp_path / "preflight.json",
        "chunk": tmp_path / "chunk.json",
        "selector": tmp_path / "selector.json",
        "chunked_selector": tmp_path / "chunked_selector.json",
        "original_csv": tmp_path / "original.csv",
        "chunked_csv": tmp_path / "chunked.csv",
        "parity": tmp_path / "parity.json",
        "fit_log": tmp_path / "fit.log",
    }
    _write_json(paths["current_source"], _source_payload(9602))
    _write_json(paths["ref_source"], _source_payload(6261))
    _write_json(paths["preflight"], _preflight_payload())
    _write_json(paths["chunk"], _chunk_payload())
    _write_json(paths["selector"], _selector_payload())
    _write_json(paths["chunked_selector"], _selector_payload())
    _write_csv(paths["original_csv"])
    _write_csv(paths["chunked_csv"])
    _write_json(paths["parity"], _parity_payload(hard_pass=parity_pass))
    paths["fit_log"].write_text(
        "\n".join(
            [
                "Using trusted canonical PPO pack runner with official RWKV train path",
                "Starting training of model test",
                "end of epoch 1",
                "ppo-phases rollout_s +3.215e+00 | update_s +3.413e-01 |",
            ]
        ),
        encoding="utf-8",
    )
    return paths


def _build(paths: dict[str, Path]) -> dict:
    return audit.build_report(
        current_source_report=paths["current_source"],
        milestone2_source_reports=[paths["ref_source"]],
        preflight_json=paths["preflight"],
        chunk_contract_json=paths["chunk"],
        original_selector_json=paths["selector"],
        original_selected_csv=paths["original_csv"],
        chunked_selector_json=paths["chunked_selector"],
        chunked_selected_csv=paths["chunked_csv"],
        parity_json=paths["parity"],
        fit_model_log=paths["fit_log"],
    )


def test_build_report_marks_profile_fit_model_path_as_milestone_candidate(tmp_path: Path):
    report = _build(_write_fixture(tmp_path))

    assert report["score"]["score"] == pytest.approx(1.0)
    assert report["decision"]["seal_as_new_engineering_milestone_candidate"] is True
    assert report["source_pool"]["same_order_as_milestone2"] is True
    assert report["chunked_label_path"]["chunked_selector_semantic_parity"] is True
    assert report["fit_model_startup"]["hard_pass"] is True
    assert report["pack_to_fit_model_parity"]["hard_pass"] is True


def test_build_report_blocks_milestone_candidate_when_numeric_parity_fails(tmp_path: Path):
    report = _build(_write_fixture(tmp_path, parity_pass=False))

    assert report["checks"]["pack_to_fit_model_numeric_parity"] is False
    assert report["score"]["score"] < 0.92
    assert report["decision"]["seal_as_new_engineering_milestone_candidate"] is False
    assert "pack_to_fit_model_numeric_parity" in report["remaining_work"]
