from scripts.exploratory import phase2_future_basin_profile_pressure_audit as audit


def test_pressure_audit_prioritizes_actionable_pendulum_label_gap():
    spectrum = {
        "spectrum_rows": [
            {
                "spectrum_id": "nearby_attractive_closed_loop_settle",
                "prior_evidence": {
                    "short_temporal_guard_viable": True,
                    "short_temporal_precision": 0.95,
                    "short_temporal_recall": 0.97,
                },
            },
            {
                "spectrum_id": "delayed_upfront_investment_future_basin",
                "prior_evidence": {
                    "delayed_candidate_rate": 0.01,
                    "future_basin_candidate_rate": 0.05,
                    "quota_ready": False,
                },
            },
        ]
    }
    selector = {
        "missing_generator_labels": [
            "pendulum_short_temporal_settle_guard",
            "mountaincar_delayed_future_basin_guard",
        ]
    }

    rows = audit.pressure_rows(spectrum, selector, {"read": {"gym_anchor_pass": True}})

    largest_actionable = next(row for row in rows if row["actionability"] >= 0.7)
    assert largest_actionable["pressure_id"] == "pendulum_settle_generator_label_gap"


def test_mountaincar_anchor_raises_actionability_without_making_it_quota_ready():
    spectrum = {
        "spectrum_rows": [
            {
                "spectrum_id": "delayed_upfront_investment_future_basin",
                "prior_evidence": {
                    "delayed_candidate_rate": 0.01,
                    "future_basin_candidate_rate": 0.05,
                    "quota_ready": False,
                },
            }
        ]
    }
    selector = {"missing_generator_labels": ["mountaincar_delayed_future_basin_guard"]}

    rows = audit.pressure_rows(spectrum, selector, {"read": {"gym_anchor_pass": True}})
    mountain = next(row for row in rows if row["pressure_id"].startswith("mountaincar"))

    assert mountain["actionability"] > 0.6
    assert mountain["evidence"]["gym_anchor_pass"] is True
    assert mountain["evidence"]["quota_ready"] is False


def test_progress_forecast_decreases_when_pressure_is_high():
    rows = [{"score": 0.9}]
    selector = {"read": {"selector_mechanics_ready": True, "fully_unified_sampling_ready": False}}

    forecast = audit.progress_forecast(rows, selector)

    assert forecast["selector_mechanics"] == 0.75
    assert forecast["generator_level_labels"] == 0.0
    assert forecast["overall_rough_progress"] < 0.72


def test_scarcity_pressure_zero_when_rate_above_target():
    assert audit._scarcity_pressure(0.3, 0.2) == 0.0
    assert audit._scarcity_pressure(0.0, 0.2) == 1.0
