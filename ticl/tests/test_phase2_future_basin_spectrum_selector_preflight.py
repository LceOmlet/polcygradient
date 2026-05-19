from scripts.exploratory import phase2_future_basin_spectrum_selector_preflight as selector


def test_balanced_select_increases_rare_label_when_available():
    rows = []
    for idx in range(8):
        rows.append({"env_idx": idx, "labels": ["common"], "common": True, "rare": False})
    for idx in range(8, 12):
        rows.append({"env_idx": idx, "labels": ["common", "rare"], "common": True, "rare": True})

    selected = selector.balanced_select(rows, n_select=4, labels=("common", "rare"), seed=1)
    rates = selector.label_rates(selected, labels=("common", "rare"))

    assert rates["rare"] >= 0.5


def test_balanced_select_avoids_common_label_domination_when_possible():
    rows = []
    for idx in range(12):
        rows.append({"env_idx": idx, "labels": ["common"], "common": True, "rare": False})
    for idx in range(12, 16):
        rows.append({"env_idx": idx, "labels": ["rare"], "common": False, "rare": True})

    selected = selector.balanced_select(rows, n_select=8, labels=("common", "rare"), seed=2)
    rates = selector.label_rates(selected, labels=("common", "rare"))

    assert rates["common"] <= 0.75
    assert rates["rare"] >= 0.25


def test_candidate_rows_preserve_scope_env_and_labels():
    unified = {
        "per_env_rows": [
            {
                "scope": "full",
                "env_idx": 3,
                "locomotion_witness": True,
                "low_energy_not_ctrl_only": False,
                "sign_or_phase_sensitive": True,
            }
        ]
    }

    rows = selector._candidate_rows(unified)

    assert rows[0]["scope"] == "full"
    assert rows[0]["env_idx"] == 3
    assert "locomotion_witness" in rows[0]["labels"]
    assert "sign_or_phase_sensitive" in rows[0]["labels"]
    assert "low_energy_not_ctrl_only" not in rows[0]["labels"]


def test_missing_generator_labels_are_explicit():
    assert "pendulum_short_temporal_settle_guard" in selector.MISSING_GENERATOR_LABELS
    assert "mountaincar_delayed_future_basin_guard" in selector.MISSING_GENERATOR_LABELS
