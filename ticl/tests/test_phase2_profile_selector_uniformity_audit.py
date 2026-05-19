from scripts.exploratory import phase2_profile_selector_uniformity_audit as audit


def test_uniformity_audit_marks_low_pressure_when_selected_matches_target():
    selector = {
        "missing_generator_labels": ["pendulum_short_temporal_settle_guard"],
        "scope_reports": {
            "full": {
                "pool_rates": {"a": 0.20, "terminal_activity_first_done": 1.0},
                "target_rates": {"a": 0.20, "terminal_activity_first_done": 1.0},
                "selected_rates": {"a": 0.22, "terminal_activity_first_done": 1.0},
            }
        },
    }

    out = audit.audit_selector_uniformity(selector)

    assert out["pressure_level"] == "low"
    assert out["read"]["missing_generator_labels_are_larger_pressure"] is True


def test_uniformity_audit_flags_label_dominance():
    selector = {
        "scope_reports": {
            "q90": {
                "pool_rates": {"feedback": 0.30},
                "target_rates": {"feedback": 0.30},
                "selected_rates": {"feedback": 0.90},
            }
        }
    }

    out = audit.audit_selector_uniformity(selector, dominance_threshold=0.75)

    assert out["pressure_level"] == "high"
    assert out["read"]["profile_implementation_uniformity_is_current_bottleneck"] is True
    assert out["dominance_rows"][0]["label"] == "feedback"
