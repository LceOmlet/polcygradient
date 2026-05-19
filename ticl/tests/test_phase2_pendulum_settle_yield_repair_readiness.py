import json
from pathlib import Path

from scripts.exploratory import phase2_pendulum_settle_yield_repair_readiness as readiness


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _preflight(min_count: int, *, required: int = 4) -> dict:
    return {
        "decision": {
            "cheap_suffix_state_settle_guard_ready": min_count >= required,
            "min_count": min_count,
            "min_ratio": min_count / 32.0,
            "required_min_count": required,
            "required_min_ratio": 0.1,
        },
        "config": {"h_overrides": {"state_output_scale": 0.85}},
        "lists": [
            {
                "label": "full",
                "classification": {
                    "coverage": {
                        "n_envs": 32,
                        "cheap_settle_guard_count": min_count,
                        "cheap_settle_guard_ratio": min_count / 32.0,
                        "reward_condition_count": 9,
                        "suffix_reward_condition_count": 9,
                        "action_decay_condition_count": 12,
                        "self_state_settle_condition_count": 2,
                        "open_state_settle_condition_count": 2,
                    }
                },
            },
            {
                "label": "q90",
                "classification": {
                    "coverage": {
                        "n_envs": 32,
                        "cheap_settle_guard_count": min_count,
                        "cheap_settle_guard_ratio": min_count / 32.0,
                        "reward_condition_count": 10,
                        "suffix_reward_condition_count": 6,
                        "action_decay_condition_count": 25,
                        "self_state_settle_condition_count": 1,
                        "open_state_settle_condition_count": 2,
                    }
                },
            },
        ],
    }


def test_readiness_allows_only_prior_derived_probe_when_specimen_is_independent(tmp_path: Path):
    current = _write_json(tmp_path / "current.json", _preflight(0))
    trial = _write_json(tmp_path / "phase2_pendulum_settle_state_label_state_output_scale_085_0512" / "p.json", _preflight(0))
    blocker = _write_json(
        tmp_path
        / "phase2_pendulum_settle_state_blocker_state_output_scale_085_0512"
        / "b.json",
        {
            "decision": {
                "dominant_blocker_is_state_basin": True,
                "selector_threshold_repair_allowed": False,
                "strict_contraction_count_safe_under_grid": False,
            }
        },
    )
    specimen = _write_json(
        tmp_path / "specimen.json",
        {
            "decision": {"specimen_preflight_pass": True},
            "aggregate": {
                "count_safe": True,
                "diversity_pass": True,
                "n_envs": 256,
                "required_min_guard_count": 64,
                "required_min_guard_ratio": 0.25,
                "condition_counts": {"guard_pass": 160},
                "condition_ratios": {
                    "guard_pass": 0.625,
                    "settle_pass": 0.9,
                    "action_decay_pass": 0.8,
                    "action_sensitive_pass": 0.9,
                },
            },
            "mechanism": {"id": "controllable_quadratic_settle_factor"},
        },
    )
    calibration = _write_json(
        tmp_path / "calibration.json",
        {
            "decision": {"cheap_guard_viable": False, "short_temporal_guard_viable": True},
            "static_analytic_best": {"precision": 0.2, "recall": 0.1, "predicted_ratio": 0.3},
            "short_temporal_best": {
                "precision": 0.95,
                "recall": 0.96,
                "predicted_ratio": 0.65,
                "truth_ratio": 0.64,
                "f1": 0.955,
            },
        },
    )
    selector = _write_json(
        tmp_path / "selector.json",
        {
            "rules": {
                "full": {
                    "profile_guard_pass": True,
                    "selected_summary": {
                        "n": 16,
                        "joint_profile_active_cells": 8,
                        "joint_profile_dominant_rate": 0.31,
                        "joint_profile_entropy_norm": 0.67,
                        "pendulum_label_available_rate": 1.0,
                        "pendulum_short_temporal_settle_guard_count": 1,
                        "pendulum_short_temporal_settle_guard_rate": 0.0625,
                    },
                },
                "q90": {
                    "profile_guard_pass": True,
                    "selected_summary": {
                        "n": 16,
                        "joint_profile_active_cells": 9,
                        "joint_profile_dominant_rate": 0.25,
                        "joint_profile_entropy_norm": 0.73,
                        "pendulum_label_available_rate": 1.0,
                        "pendulum_short_temporal_settle_guard_count": 0,
                        "pendulum_short_temporal_settle_guard_rate": 0.0,
                    },
                },
            }
        },
    )
    closure = _write_json(
        tmp_path / "closure.json",
        {
            "progress": {"weighted_progress": 0.72},
            "decision": {
                "largest_remaining_blocker": "pendulum_settle_semantics",
                "second_blocker": "generator_level_selector",
            },
        },
    )

    report = readiness.build_report(
        current_preflight_json=current,
        preflight_paths=[str(trial)],
        blocker_paths=[str(blocker)],
        specimen_json=specimen,
        calibration_paths=[str(calibration)],
        selector_json=selector,
        closure_json=closure,
    )

    assert report["existing_knob_trials"]["existing_knobs_falsified"] is True
    assert report["quadratic_settle_mechanism"]["structural_mechanism_viable"] is True
    assert report["quadratic_settle_mechanism"]["standalone_specimen_integration_allowed"] is False
    assert report["cheap_or_cached_guard"]["short_temporal_cached_label_viable"] is True
    assert report["decision"]["ready_for_generator_level_prototype"] is False
    assert report["decision"]["ready_for_prior_derived_repair_probe"] is True
    assert report["decision"]["third_milestone_ready"] is False


def test_readiness_blocks_when_cached_label_is_not_precise_enough(tmp_path: Path):
    current = _write_json(tmp_path / "current.json", _preflight(0))
    specimen = _write_json(
        tmp_path / "specimen.json",
        {
            "decision": {"specimen_preflight_pass": True},
            "aggregate": {
                "count_safe": True,
                "diversity_pass": True,
                "condition_counts": {"guard_pass": 64},
                "condition_ratios": {"guard_pass": 0.25},
            },
        },
    )
    calibration = _write_json(
        tmp_path / "calibration.json",
        {
            "decision": {"cheap_guard_viable": False, "short_temporal_guard_viable": True},
            "short_temporal_best": {"precision": 0.4, "predicted_ratio": 0.65, "f1": 0.5},
        },
    )
    selector = _write_json(
        tmp_path / "selector.json",
        {"rules": {"full": {"profile_guard_pass": True, "selected_summary": {}}}},
    )
    closure = _write_json(tmp_path / "closure.json", {"progress": {}, "decision": {}})

    report = readiness.build_report(
        current_preflight_json=current,
        preflight_paths=[],
        blocker_paths=[],
        specimen_json=specimen,
        calibration_paths=[str(calibration)],
        selector_json=selector,
        closure_json=closure,
    )

    assert report["cheap_or_cached_guard"]["short_temporal_cached_label_viable"] is False
    assert report["decision"]["ready_for_generator_level_prototype"] is False


def test_calibration_summary_uses_latest_protocol_family(tmp_path: Path):
    old = _write_json(
        tmp_path / "seed_1_n512" / "calibration.json",
        {
            "decision": {"cheap_guard_viable": False, "short_temporal_guard_viable": False},
            "short_temporal_best": {"precision": 0.1, "predicted_ratio": 0.1, "f1": 0.1},
        },
    )
    new = _write_json(
        tmp_path / "seed_1_n512_v3" / "calibration.json",
        {
            "decision": {"cheap_guard_viable": False, "short_temporal_guard_viable": True},
            "short_temporal_best": {"precision": 0.9, "predicted_ratio": 0.5, "f1": 0.9},
        },
    )

    summary = readiness._calibration_summary([str(old), str(new)])

    assert summary["selected_protocol_version"] == 3
    assert summary["selected_protocol_run_count"] == 1
    assert summary["short_temporal_cached_label_viable"] is True
