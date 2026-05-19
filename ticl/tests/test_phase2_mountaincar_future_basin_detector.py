from pathlib import Path

import pytest

from scripts.exploratory import phase2_mountaincar_future_basin_detector as detector


def test_mountaincar_detector_separates_suffix_basin_from_prefix_action():
    class Args:
        abs_margin = 5.0
        threshold_fraction = 0.5
        action_decay_margin = 0.02
        min_prefix_action_abs = 0.02
        delayed_margin_fraction = 0.25

    row = {
        "threshold": 20.0,
        "best_early_only_action_abs_prefix": 0.2,
        "best_early_only_action_abs_suffix": 0.0,
        "best_early_only_prefix_env_return": -10.0,
        "zero_prefix_env_return": 0.0,
        "early_only_suffix_gain_over_zero_env": 30.0,
        "early_only_suffix_gain_over_open_env": 25.0,
        "early_only_beats_zero_env": True,
        "early_only_beats_open_loop_env": True,
    }

    out = detector._classify_row(row, args=Args())

    assert out["mountaincar_upfront_action_decay"] is True
    assert out["mountaincar_future_basin_candidate"] is True
    assert out["mountaincar_future_basin_strong"] is True
    assert out["mountaincar_delayed_candidate"] is True


def test_mountaincar_detector_rejects_prefix_action_without_future_basin():
    class Args:
        abs_margin = 5.0
        threshold_fraction = 0.5
        action_decay_margin = 0.02
        min_prefix_action_abs = 0.02
        delayed_margin_fraction = 0.25

    row = {
        "threshold": 20.0,
        "best_early_only_action_abs_prefix": 0.2,
        "best_early_only_action_abs_suffix": 0.0,
        "best_early_only_prefix_env_return": 30.0,
        "zero_prefix_env_return": 0.0,
        "early_only_suffix_gain_over_zero_env": 3.0,
        "early_only_suffix_gain_over_open_env": 2.0,
        "early_only_beats_zero_env": True,
        "early_only_beats_open_loop_env": True,
    }

    out = detector._classify_row(row, args=Args())

    assert out["mountaincar_upfront_action_decay"] is True
    assert out["mountaincar_future_basin_vs_zero"] is False
    assert out["mountaincar_future_basin_candidate"] is False


def test_mountaincar_detector_materializes_local_report_when_probe_exists(tmp_path):
    if not Path(detector.DEFAULT_INPUT_JSON).exists():
        pytest.skip("local time-profile probe artifact is absent")

    class Args:
        input_json = str(detector.DEFAULT_INPUT_JSON)
        output_dir = str(tmp_path)
        abs_margin = 5.0
        threshold_fraction = 0.5
        action_decay_margin = 0.02
        min_prefix_action_abs = 0.02
        delayed_margin_fraction = 0.25

    detector.run(Args())

    assert (tmp_path / "mountaincar_future_basin_detector_report.json").exists()
    assert (tmp_path / "mountaincar_future_basin_per_env.csv").exists()
