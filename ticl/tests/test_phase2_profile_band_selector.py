from pathlib import Path
import csv
import json

import pytest

from scripts.exploratory import phase2_profile_band_selector as selector


def test_default_profile_config_logs_feedback_and_mountaincar_not_quota():
    cfg = selector._load_profile_config(None)

    assert cfg["background_milestone"] == "gated_reward_path_balance"
    assert cfg["axes"]["locomotion_witness"]["role"] == "raised_band"
    assert cfg["axes"]["low_energy_not_ctrl_only"]["role"] == "protected_band"
    assert cfg["prior_pack_optimizability_guard"]["enabled"] is False
    assert cfg["logged_not_quota_axes"]["context_identifiable_closed_loop_settle_feedback"][
        "trust_status"
    ] == "pressure_point_subprofile_not_single_quota"
    assert cfg["logged_not_quota_axes"]["mountaincar_upfront_investment_future_basin"][
        "trust_status"
    ] == "missing_detector"


def test_summary_uses_joint_profile_cells_not_raw_delta():
    rows = [
        {
            "locomotion_witness": True,
            "low_energy_not_ctrl_only": False,
            "sign_or_phase_sensitive": True,
            "state_conditioned_energy_injection": False,
            "raw_reward_sum_delta": -10.0,
            "terminal_reset_count_target": 5.0,
            "balanced_reward_action_input_gain_fraction": 0.25,
        },
        {
            "locomotion_witness": True,
            "low_energy_not_ctrl_only": False,
            "sign_or_phase_sensitive": True,
            "state_conditioned_energy_injection": False,
            "raw_reward_sum_delta": 10.0,
            "terminal_reset_count_target": 25.0,
            "balanced_reward_action_input_gain_fraction": 0.15,
        },
        {
            "locomotion_witness": False,
            "low_energy_not_ctrl_only": True,
            "sign_or_phase_sensitive": False,
            "state_conditioned_energy_injection": True,
            "raw_reward_sum_delta": 0.0,
            "terminal_reset_count_target": 15.0,
            "balanced_reward_action_input_gain_fraction": 0.20,
        },
    ]

    summary = selector._summary(rows, rule="synthetic")

    assert summary["joint_profile_active_cells"] == 2
    assert summary["joint_profile_dominant_count"] == 2
    assert summary["locomotion_witness_count"] == 2
    assert summary["terminal_reset_count_target"]["mean"] == pytest.approx(15.0)
    assert summary["terminal_reset_count_target_gt20_count"] == 1
    assert summary["balanced_reward_action_input_gain_fraction"]["max"] == pytest.approx(0.25)
    assert "raw_reward_sum_delta" not in summary


def test_prior_pack_optimizability_guard_adds_terminal_and_proxy_constraints():
    rows = [
        {"terminal_reset_count_target": 5.0, "balanced_reward_action_input_gain_fraction": 0.25},
        {"terminal_reset_count_target": 10.0, "balanced_reward_action_input_gain_fraction": 0.19},
        {"terminal_reset_count_target": 30.0, "balanced_reward_action_input_gain_fraction": 0.30},
        {"terminal_reset_count_target": 40.0, "balanced_reward_action_input_gain_fraction": 0.10},
    ]
    config = {
        "prior_pack_optimizability_guard": {
            "enabled": True,
            "terminal_reset_count_target_max": 20.0,
            "balanced_action_gain_min": 0.18,
            "balanced_action_gain_min_by_rule": {"synthetic": 0.20},
            "min_proxy_count_by_rule": {"synthetic": 1},
            "max_high_terminal_count_by_rule": {"synthetic": 1},
            "min_low_terminal_count_by_rule": {"synthetic": 2},
            "trust_status": "unit_test",
        }
    }
    a_rows = []
    lb = []
    ub = []
    details = {}

    proxy, high_terminal, action_gain = selector._add_prior_pack_optimizability_constraints(
        rows=rows,
        rule="synthetic",
        target_n=3,
        config=config,
        a_rows=a_rows,
        lb=lb,
        ub=ub,
        details=details,
    )

    assert proxy.tolist() == [1.0, 0.0, 0.0, 0.0]
    assert high_terminal.tolist() == [0.0, 0.0, 1.0, 1.0]
    assert action_gain.tolist() == pytest.approx([0.25, 0.19, 0.30, 0.10])
    assert lb == [1.0, 0.0, 2.0]
    assert ub == [3.0, 1.0, 3.0]
    assert details["prior_pack_optimizability_guard"]["source_proxy_count"] == 1
    assert details["prior_pack_optimizability_guard"]["balanced_action_gain_min"] == 0.20


def test_prior_pack_optimizability_guard_fails_when_proxy_support_is_absent():
    rows = [
        {"terminal_reset_count_target": 5.0, "balanced_reward_action_input_gain_fraction": 0.10},
        {"terminal_reset_count_target": 30.0, "balanced_reward_action_input_gain_fraction": 0.30},
    ]
    config = {
        "prior_pack_optimizability_guard": {
            "enabled": True,
            "terminal_reset_count_target_max": 20.0,
            "balanced_action_gain_min": 0.18,
            "min_proxy_count": 1,
        }
    }

    with pytest.raises(ValueError, match="prior-pack optimizable proxy"):
        selector._add_prior_pack_optimizability_constraints(
            rows=rows,
            rule="synthetic",
            target_n=2,
            config=config,
            a_rows=[],
            lb=[],
            ub=[],
            details={},
        )


def test_prior_pack_optimizability_guard_adds_terminal_distribution_bins():
    rows = [
        {"terminal_reset_count_target": 5.0, "balanced_reward_action_input_gain_fraction": 0.25},
        {"terminal_reset_count_target": 15.0, "balanced_reward_action_input_gain_fraction": 0.21},
        {"terminal_reset_count_target": 25.0, "balanced_reward_action_input_gain_fraction": 0.30},
        {"terminal_reset_count_target": 35.0, "balanced_reward_action_input_gain_fraction": 0.10},
    ]
    config = {
        "prior_pack_optimizability_guard": {
            "enabled": True,
            "terminal_reset_count_target_max": 20.0,
            "balanced_action_gain_min": 0.18,
            "min_proxy_count": 1,
            "max_high_terminal_count": 2,
            "min_low_terminal_count": 2,
            "terminal_distribution_guard": {
                "enabled": True,
                "edges": [0.0, 20.0, 40.0],
                "min_count": [1, 1],
                "max_count": [2, 2],
                "trust_status": "unit_test_uniform_terminal",
            },
        }
    }
    a_rows = []
    lb = []
    ub = []
    details = {}

    selector._add_prior_pack_optimizability_constraints(
        rows=rows,
        rule="synthetic",
        target_n=4,
        config=config,
        a_rows=a_rows,
        lb=lb,
        ub=ub,
        details=details,
    )

    assert lb == [1.0, 0.0, 2.0, 1.0, 1.0]
    assert ub == [4.0, 2.0, 4.0, 2.0, 2.0]
    bins = details["prior_pack_optimizability_guard"]["terminal_distribution_guard"]["bins"]
    assert [(row["lower"], row["upper"], row["source_count"]) for row in bins] == [
        (0.0, 20.0, 2),
        (20.0, 40.0, 2),
    ]


def test_terminal_distribution_guard_adds_per_bin_action_and_mode_support():
    rows = [
        {
            "terminal_reset_count_target": 5.0,
            "balanced_reward_action_input_gain_fraction": 0.25,
            "locomotion_witness": True,
        },
        {
            "terminal_reset_count_target": 15.0,
            "balanced_reward_action_input_gain_fraction": 0.10,
            "locomotion_witness": False,
        },
        {
            "terminal_reset_count_target": 25.0,
            "balanced_reward_action_input_gain_fraction": 0.30,
            "locomotion_witness": True,
        },
        {
            "terminal_reset_count_target": 35.0,
            "balanced_reward_action_input_gain_fraction": 0.12,
            "locomotion_witness": False,
        },
    ]
    config = {
        "prior_pack_optimizability_guard": {
            "enabled": True,
            "terminal_reset_count_target_max": 20.0,
            "balanced_action_gain_min": 0.18,
            "min_proxy_count": 1,
            "max_high_terminal_count": 2,
            "min_low_terminal_count": 2,
            "terminal_distribution_guard": {
                "enabled": True,
                "edges": [0.0, 20.0, 40.0],
                "min_count": [1, 1],
                "max_count": [2, 2],
                "action_min": 0.20,
                "min_action_count": [1, 1],
                "mode_min_count": {"locomotion_witness": [1, 1]},
            },
        }
    }
    a_rows = []
    lb = []
    ub = []
    details = {}

    selector._add_prior_pack_optimizability_constraints(
        rows=rows,
        rule="synthetic",
        target_n=4,
        config=config,
        a_rows=a_rows,
        lb=lb,
        ub=ub,
        details=details,
    )

    assert lb == [1.0, 0.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    bins = details["prior_pack_optimizability_guard"]["terminal_distribution_guard"]["bins"]
    assert bins[0]["source_action_count"] == 1
    assert bins[1]["source_action_count"] == 1
    assert bins[0]["mode_min_counts"]["locomotion_witness"]["source_count"] == 1
    assert bins[1]["mode_min_counts"]["locomotion_witness"]["source_count"] == 1


def test_profile_band_selector_materializes_local_artifact_when_inputs_exist(tmp_path):
    required = [
        selector.DEFAULT_SELECTED_CSV,
        selector.DEFAULT_LOCOMOTION_CSV,
        selector.DEFAULT_ACTION_MODE_JSON,
    ]
    missing = [str(path) for path in required if not Path(path).exists()]
    if missing:
        pytest.skip(f"local exploratory artifacts are absent: {missing}")

    class Args:
        selected_csv = str(selector.DEFAULT_SELECTED_CSV)
        locomotion_csv = str(selector.DEFAULT_LOCOMOTION_CSV)
        action_mode_json = str(selector.DEFAULT_ACTION_MODE_JSON)
        strict_feedback_csv = str(selector.DEFAULT_STRICT_FEEDBACK_CSV)
        profile_config = None
        output_dir = str(tmp_path)

    selector.run(Args())

    report = tmp_path / "profile_band_selector_report.json"
    selected = tmp_path / "selected_profile_band_prior_envs.csv"
    assert report.exists()
    assert selected.exists()
    assert "raw_delta_not_used_in_selection" in report.read_text(encoding="utf-8")


def test_parse_feedback_mode_extracts_subprofile_gain_and_sign():
    parsed = selector._parse_feedback_mode("decay_obs_linear_neg_3x200")

    assert parsed["strict_feedback_profile"] == "decay"
    assert parsed["strict_feedback_gain"] == 2.0
    assert parsed["strict_feedback_sign"] == "neg"
    assert parsed["strict_feedback_feature"] == 3


def test_strict_feedback_loader_prefers_effective_mode(tmp_path):
    h_path = tmp_path / "h.json"
    h_path.write_text(json.dumps({"alpha": 1.0, "beta": [2, 3]}), encoding="utf-8")
    csv_path = tmp_path / "feedback.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source_tag",
                "env_index",
                "full_frozen_h_json",
                "feedback_source_pool_definition",
                "effective_feedback_source",
                "effective_feedback_mode",
                "best_feedback_mode",
                "effective_feedback_minus_open_env",
                "signature_score",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "source_tag": "seed6262_full",
                "env_index": 0,
                "full_frozen_h_json": str(h_path),
                "feedback_source_pool_definition": "strict_or_time_profile_core_no_random",
                "effective_feedback_source": "time_profile_core_no_random",
                "effective_feedback_mode": "decay_obs_linear_1x100",
                "best_feedback_mode": "obs_linear_0x200",
                "effective_feedback_minus_open_env": "5.0",
                "signature_score": "7.0",
            }
        )

    labels = selector._load_strict_feedback_labels(csv_path)
    digest = selector._frozen_h_digest(h_path)
    row = labels[digest]

    assert row["strict_feedback_profile"] == "decay"
    assert row["strict_feedback_gain"] == 1.0
    assert row["strict_feedback_source_pool_definition"] == "strict_or_time_profile_core_no_random"
    assert row["strict_feedback_effective_source"] == "time_profile_core_no_random"


def test_strict_feedback_summary_counts_subprofile_and_gain():
    rows = [
        {
            "locomotion_witness": False,
            "low_energy_not_ctrl_only": False,
            "sign_or_phase_sensitive": False,
            "state_conditioned_energy_injection": False,
            "strict_feedback_candidate": True,
            "strict_feedback_profile": "constant",
            "strict_feedback_gain": 2.0,
        },
        {
            "locomotion_witness": False,
            "low_energy_not_ctrl_only": False,
            "sign_or_phase_sensitive": False,
            "state_conditioned_energy_injection": False,
            "strict_feedback_candidate": True,
            "strict_feedback_profile": "decay",
            "strict_feedback_gain": 1.0,
        },
        {
            "locomotion_witness": False,
            "low_energy_not_ctrl_only": False,
            "sign_or_phase_sensitive": False,
            "state_conditioned_energy_injection": False,
            "strict_feedback_candidate": False,
        },
    ]

    summary = selector._summary(rows, rule="synthetic")

    assert summary["strict_feedback_count"] == 2
    assert summary["strict_feedback_profile_counts"] == {"constant": 1, "decay": 1}
    assert summary["strict_feedback_gain_counts"] == {"2.0": 1, "1.0": 1}


def test_future_basin_cached_labels_are_logged_not_profile_axes(tmp_path):
    csv_path = tmp_path / "future.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "rule",
                "source_index",
                "pendulum_label_available",
                "pendulum_axis_support",
                "pendulum_short_gap",
                "pendulum_sustained_settle",
                "pendulum_profile_quota_ready",
                "mountaincar_label_available",
                "mountaincar_short_future_basin_guard",
                "mountaincar_label_use",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "rule": "gym_full_obs_action_range",
                "source_index": 0,
                "pendulum_label_available": "True",
                "pendulum_axis_support": "True",
                "pendulum_short_gap": "False",
                "pendulum_sustained_settle": "False",
                "pendulum_profile_quota_ready": "False",
                "mountaincar_label_available": "True",
                "mountaincar_short_future_basin_guard": "False",
                "mountaincar_label_use": "coverage_only_not_quota",
            }
        )

    labels = selector._load_future_basin_cached_labels(csv_path)
    row = {
        "locomotion_witness": False,
        "low_energy_not_ctrl_only": False,
        "sign_or_phase_sensitive": False,
        "state_conditioned_energy_injection": False,
        **labels[("gym_full_obs_action_range", 0)],
    }
    summary = selector._summary([row], rule="gym_full_obs_action_range")

    assert "pendulum_short_temporal_settle_guard" not in selector.MODE_KEYS
    assert summary["pendulum_axis_support_count"] == 1
    assert summary["pendulum_short_gap_count"] == 0
    assert summary["pendulum_sustained_settle_count"] == 0
    assert summary["pendulum_profile_quota_ready_count"] == 0
    assert summary["pendulum_short_temporal_settle_guard_count"] == 0
    assert summary["mountaincar_label_available_count"] == 1


def test_pendulum_settle_repair_branch_materializes_prior_internal_patch(tmp_path):
    h_path = tmp_path / "h.json"
    h_path.write_text(
        json.dumps(
            {
                "family": "scm",
                "state_dim": 2,
                "obs_dim": 2,
                "action_dim": 1,
            }
        ),
        encoding="utf-8",
    )
    rows = [
        {
            "_source_index": 0,
            "full_frozen_h_json": str(h_path),
            "env_seed": 123,
            "state_dim": 2,
            "action_dim": 1,
            "locomotion_witness": False,
            "low_energy_not_ctrl_only": False,
            "sign_or_phase_sensitive": True,
            "state_conditioned_energy_injection": False,
            "pendulum_label_available": True,
            "pendulum_axis_support": False,
            "pendulum_profile_quota_ready": False,
        }
    ]
    cfg = {
        "target_per_rule": 1,
        "pendulum_settle_yield_repair_branch": {
            "enabled": True,
            "exact_count": 1,
            "require_cached_label_available": True,
            "prefer_missing_cached_guard": True,
        },
    }

    details = selector._assign_pendulum_settle_repair_branch(
        rows,
        rule="gym_full_obs_action_range",
        config=cfg,
    )
    materialized = selector._materialize(tmp_path / "out", "gym_full_obs_action_range", rows)
    h_list = json.loads(Path(materialized["fixed_h_list_json"]).read_text(encoding="utf-8"))

    assert details["selected_count"] == 1
    assert rows[0]["pendulum_settle_yield_repair_planned"] is True
    assert h_list[0]["pendulum_settle_yield_repair_enabled"] is True
    assert h_list[0]["pendulum_settle_yield_repair_prob"] == 1.0
    assert h_list[0]["pendulum_settle_yield_action_controls_position"] is True
    assert h_list[0]["pendulum_settle_yield_repair_standalone_generator"] is False
    assert h_list[0]["pendulum_settle_yield_repair_uses_existing_state_action_slots"] is True
    assert rows[0]["full_frozen_h_json"] != str(h_path)


def test_pendulum_settle_unplanned_repair_is_cleared_and_csv_points_materialized_h(tmp_path):
    h_path = tmp_path / "h.json"
    h_path.write_text(
        json.dumps(
            {
                "family": "scm",
                "state_dim": 2,
                "obs_dim": 2,
                "action_dim": 1,
                "pendulum_settle_yield_repair_enabled": True,
                "pendulum_settle_yield_repair_prob": 1.0,
                "pendulum_settle_yield_action_controls_position": True,
                "pendulum_settle_yield_repair_expected_guard": True,
            }
        ),
        encoding="utf-8",
    )
    rows = [
        {
            "_source_index": 0,
            "full_frozen_h_json": str(h_path),
            "env_seed": 123,
            "pendulum_settle_yield_repair_planned": False,
            "frozen_h_fingerprint": "source",
        }
    ]

    materialized = selector._materialize(tmp_path / "out", "gym_full_obs_action_range", rows)
    h_list = json.loads(Path(materialized["fixed_h_list_json"]).read_text(encoding="utf-8"))
    materialized_h = json.loads(Path(rows[0]["full_frozen_h_json"]).read_text(encoding="utf-8"))

    assert rows[0]["full_frozen_h_json"] != str(h_path)
    assert rows[0]["frozen_h_fingerprint"] != "source"
    assert h_list[0]["pendulum_settle_yield_repair_enabled"] is False
    assert h_list[0]["pendulum_settle_yield_repair_prob"] == 0.0
    assert h_list[0]["pendulum_settle_yield_action_controls_position"] is False
    assert materialized_h == h_list[0]


def test_pendulum_settle_source_axis_repair_is_preserved_when_labeled(tmp_path):
    h_path = tmp_path / "h.json"
    h_path.write_text(
        json.dumps(
            {
                "family": "scm",
                "state_dim": 2,
                "obs_dim": 2,
                "action_dim": 1,
                "pendulum_settle_yield_repair_enabled": True,
                "pendulum_settle_yield_repair_prob": 1.0,
                "pendulum_settle_yield_action_controls_position": True,
                "pendulum_settle_yield_repair_source": "generator_level_profile_axis",
            }
        ),
        encoding="utf-8",
    )
    rows = [
        {
            "_source_index": 0,
            "full_frozen_h_json": str(h_path),
            "env_seed": 123,
            "pendulum_axis_support": True,
            "pendulum_axis_supported_sustained_settle": True,
            "pendulum_settle_yield_repair_planned": False,
        }
    ]

    materialized = selector._materialize(tmp_path / "out", "gym_full_obs_action_range", rows)
    h_list = json.loads(Path(materialized["fixed_h_list_json"]).read_text(encoding="utf-8"))

    assert h_list[0]["pendulum_settle_yield_repair_enabled"] is True
    assert h_list[0]["pendulum_settle_yield_repair_source"] == "generator_level_profile_axis"
    assert h_list[0]["pendulum_settle_yield_repair_standalone_generator"] is False
    assert h_list[0]["pendulum_settle_yield_repair_uses_existing_state_action_slots"] is True
