import csv
import json
from pathlib import Path

import pytest

from scripts.exploratory import phase2_profile_online_selector_preflight as preflight


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_selected_csv(path: Path, n_per_rule: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for rule in preflight.RULES:
        for idx in range(n_per_rule):
            rows.append({"rule": rule, "env_seed": str(1000 + idx), "full_frozen_h_json": f"/tmp/{rule}_{idx}.json"})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["rule", "env_seed", "full_frozen_h_json"])
        writer.writeheader()
        writer.writerows(rows)


def _selector_payload() -> dict:
    return {
        "config": {"profile_version_id": "profile_band_online_feedback_support_prototype"},
        "rules": {
            rule: {
                "guard_pass": True,
                "source_summary": {"n": 64},
                "selected_summary": {
                    "n": 16,
                    "joint_profile_active_cells": 8,
                    "joint_profile_dominant_rate": 0.25,
                    "joint_profile_entropy_norm": 0.7,
                    "locomotion_witness_rate": 0.375,
                    "low_energy_not_ctrl_only_rate": 0.5,
                    "sign_or_phase_sensitive_rate": 0.875,
                    "state_conditioned_energy_injection_rate": 0.3125,
                    "strict_feedback_rate": 0.625,
                    "strict_feedback_profile_counts": {"decay": 10},
                    "strict_feedback_gain_counts": {"1.0": 10},
                },
            }
            for rule in preflight.RULES
        },
    }


def test_build_report_scores_existing_pool64_ready(tmp_path: Path):
    pool = tmp_path / "pool"
    selector = tmp_path / "selector"
    _write_selected_csv(pool / "source_pool/selected_prior_envs.csv", 64)
    _write_json(
        pool / "source_pool/reward_group_balance_probe_report.json",
        {"sampling": {"raw_seen": 4096, "target_per_rule": 64}},
    )
    _write_json(pool / "action_mode_probe/report.json", {"ok": True})
    (pool / "locomotion_probe").mkdir(parents=True)
    (pool / "locomotion_probe/locomotion_full_temporal_per_env.csv").write_text("scope,env_index\n", encoding="utf-8")
    (pool / "feedback_near_best_source_pool_corrected").mkdir(parents=True)
    (pool / "feedback_near_best_source_pool_corrected/feedback_near_best_gain_per_env.csv").write_text(
        "rule,source_index\n", encoding="utf-8"
    )
    _write_json(selector / "profile_band_selector_report.json", _selector_payload())

    report = preflight.build_report(
        existing_pool_root=pool,
        existing_selector_dir=selector,
        planned_run_root=tmp_path / "new_run",
        target_per_rule=16,
        candidate_multiplier=4,
        seed=9601,
        env_seed_start=9601000,
        device="cuda:0",
        profile_config=tmp_path / "profile.json",
        max_raw_samples=262144,
        raw_batch_size=4096,
        env_eval_batch_size=64,
        n_steps_action=256,
        n_steps_locomotion=256,
        n_steps_feedback=64,
        feedback_label_mode="budgeted_guard",
        feedback_baseline_modes="zero,random,neg_random,half_random",
        feedback_profiles="decay",
        feedback_gains="1.0",
        feedback_obs_linear_count=2,
        max_feedback_eval_units=500_000,
        fit_model_target_envs=2048,
        reference_probe_envs=64,
        reference_probe_rss_gib=1.75,
        reference_probe_gpu_delta_gib=1.65,
        probe_chunk_envs=64,
        host_rss_limit_gib=32.0,
        gpu0_mem_limit_gib=49.0,
        pendulum_settle_yield_axis_profile="settle_axis_v1",
        pendulum_settle_yield_axis_rate=0.25,
        pendulum_settle_yield_axis_springs="0.0,0.2",
        pendulum_settle_yield_axis_position_couplings="0.0,1.0",
        pendulum_settle_yield_axis_action_controls_positions="false,true",
    )

    assert report["contract"]["delta_positive_fraction_not_a_gate"] is True
    assert report["existing_evidence"]["source_pool"]["source_ready_for_candidate_multiplier"] is True
    assert report["existing_evidence"]["selector"]["hard_pass"] is True
    assert report["sufficiency_forecast"]["score"] == pytest.approx(0.88)
    assert report["sufficiency_forecast"]["new_seed_generator_execution_ready"] is False
    assert report["sufficiency_forecast"]["fit_model_selector_path_parity_ready"] is False
    scale = report["fit_model_2048_scale_projection"]
    assert scale["naive_all_at_once"]["memory_safe"] is False
    assert scale["chunked_probe"]["memory_safe"] is True
    assert scale["chunked_probe"]["chunk_count"] == 128
    budget = report["bounded_generator_plan"]["feedback_eval_budget"]
    assert budget["label_mode"] == "budgeted_guard"
    assert budget["mode_count_per_rule"] == 8
    assert budget["execution_blocked_by_default"] is False
    feedback_stage = [
        item for item in report["bounded_generator_plan"]["stage_plan"]
        if item["name"] == "feedback_budgeted_guard_labels"
    ][0]
    assert "--feedback-profiles" in feedback_stage["command"]
    assert "--profiles" not in feedback_stage["command"]
    source_stage = [
        item for item in report["bounded_generator_plan"]["stage_plan"]
        if item["name"] == "source_pool"
    ][0]
    assert "--pendulum-settle-yield-axis-rate" in source_stage["command"]
    settle_axis = report["bounded_generator_plan"]["pendulum_settle_yield_axis"]
    assert settle_axis["default_path_enabled"] is True
    assert settle_axis["standalone_generator"] is False
    assert settle_axis["uses_existing_state_action_slots"] is True
    assert settle_axis["springs"] == [0.0, 0.2]
    assert settle_axis["position_couplings"] == [0.0, 1.0]
    assert settle_axis["action_controls_positions"] == [False, True]
    assert "--pendulum-settle-yield-axis-action-controls-positions" in source_stage["command"]
    assert "profile_band_selector.py" in report["bounded_generator_plan"]["stage_plan"][-1]["command_text"]


def test_build_report_marks_missing_labels_as_remaining(tmp_path: Path):
    pool = tmp_path / "pool"
    selector = tmp_path / "selector"
    _write_selected_csv(pool / "source_pool/selected_prior_envs.csv", 16)
    _write_json(pool / "source_pool/reward_group_balance_probe_report.json", {"sampling": {}})

    report = preflight.build_report(
        existing_pool_root=pool,
        existing_selector_dir=selector,
        planned_run_root=tmp_path / "new_run",
        target_per_rule=16,
        candidate_multiplier=4,
        seed=9601,
        env_seed_start=9601000,
        device="cuda:0",
        profile_config=tmp_path / "profile.json",
        max_raw_samples=262144,
        raw_batch_size=4096,
        env_eval_batch_size=64,
        n_steps_action=256,
        n_steps_locomotion=256,
        n_steps_feedback=64,
        feedback_label_mode="budgeted_guard",
        feedback_baseline_modes="zero,random,neg_random,half_random",
        feedback_profiles="decay",
        feedback_gains="1.0",
        feedback_obs_linear_count=2,
        max_feedback_eval_units=500_000,
        fit_model_target_envs=2048,
        reference_probe_envs=64,
        reference_probe_rss_gib=1.75,
        reference_probe_gpu_delta_gib=1.65,
        probe_chunk_envs=64,
        host_rss_limit_gib=32.0,
        gpu0_mem_limit_gib=49.0,
    )

    assert report["existing_evidence"]["source_pool"]["source_ready_for_candidate_multiplier"] is False
    assert report["existing_evidence"]["broad_labels_ready"] is False
    assert report["existing_evidence"]["feedback_support_ready"] is False
    assert report["existing_evidence"]["selector"]["available"] is False
    assert 0.0 < report["sufficiency_forecast"]["score"] < 1.0
