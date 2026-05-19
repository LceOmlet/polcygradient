from scripts.exploratory import phase2_action_conditioned_future_basin_spectrum as spectrum


def test_gym_anchors_place_pendulum_and_mountaincar_in_same_spectrum():
    anchors = {row["env"]: row for row in spectrum.gym_anchors()}

    pendulum = anchors["Pendulum-v1"]
    mountain = anchors["MountainCarContinuous-v0"]

    assert pendulum["profile"]["valence"] == mountain["profile"]["valence"] == "attractive"
    assert pendulum["profile"]["control_phase"] == "settle"
    assert mountain["profile"]["control_phase"] == "invest_then_basin"
    assert pendulum["profile"]["credit_horizon"] != mountain["profile"]["credit_horizon"]


def test_report_marks_static_guard_as_non_solution_and_short_guard_as_evidence():
    report = spectrum.build_report(
        unified={"modes": []},
        mountaincar={"all_summary": {}},
        settle_specimen={"aggregate": {"condition_ratios": {"guard_pass": 0.6}}},
        settle_calibration={
            "decision": {"short_temporal_guard_viable": True},
            "static_analytic_best": {"precision": 0.3},
            "short_temporal_best": {"precision": 0.95, "recall": 0.98},
        },
    )

    pendulum = next(row for row in report["spectrum_rows"] if row["spectrum_id"].startswith("nearby"))

    assert pendulum["prior_evidence"]["static_precision"] == 0.3
    assert pendulum["prior_evidence"]["short_temporal_guard_viable"] is True
    assert "static analytic" in " ".join(report["remaining_progress"]["do_not_do"])


def test_mountaincar_delayed_mode_is_detected_as_sparse_annotation():
    report = spectrum.build_report(
        unified={"modes": []},
        mountaincar={
            "all_summary": {
                "mountaincar_future_basin_candidate_rate": 0.05,
                "mountaincar_future_basin_strong_rate": 0.08,
                "mountaincar_delayed_candidate_rate": 0.01,
                "mountaincar_upfront_action_decay_rate": 0.9,
            },
            "read": {"quota_ready": False},
        },
        settle_specimen=None,
        settle_calibration=None,
    )

    mountain = next(row for row in report["spectrum_rows"] if row["spectrum_id"].startswith("delayed"))

    assert mountain["prior_evidence"]["delayed_candidate_rate"] == 0.01
    assert mountain["prior_evidence"]["quota_ready"] is False
    assert "sparse" in mountain["coverage_read"]
