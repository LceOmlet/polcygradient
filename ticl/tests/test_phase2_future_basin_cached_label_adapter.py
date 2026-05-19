import csv
from pathlib import Path

from scripts.exploratory import phase2_future_basin_cached_label_adapter as adapter


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_cached_label_adapter_maps_by_rule_and_source_index(tmp_path: Path):
    selected = tmp_path / "source_pool" / "selected_prior_envs.csv"
    _write_csv(
        selected,
        [
            {"rule": "gym_full_obs_action_range", "env_seed": "10"},
            {"rule": "gym_full_obs_action_range", "env_seed": "11"},
            {"rule": "gym_q90_obs_action_range", "env_seed": "20"},
        ],
    )
    mountaincar = tmp_path / "mountaincar.csv"
    _write_csv(
        mountaincar,
        [
            {
                "source_label": "gym_full_obs_action_range",
                "env_index": "1",
                "mountaincar_future_basin_candidate": "true",
                "mountaincar_delayed_candidate": "false",
                "mountaincar_future_basin_strong": "true",
                "mountaincar_upfront_action_decay": "true",
                "mountaincar_margin": "5.0",
                "best_early_only_mode": "early_obs_linear_pos_then_zero",
            },
            {
                "source_label": "gym_q90_obs_action_range",
                "env_index": "0",
                "mountaincar_future_basin_candidate": "false",
                "mountaincar_delayed_candidate": "true",
                "mountaincar_future_basin_strong": "false",
                "mountaincar_upfront_action_decay": "true",
                "mountaincar_margin": "7.0",
                "best_early_only_mode": "early_random_then_zero",
            },
        ],
    )
    pendulum = tmp_path / "pendulum.csv"
    _write_csv(
        pendulum,
        [
            {
                "rule": "gym_full_obs_action_range",
                "source_index": "0",
                "pendulum_axis_support": "true",
                "pendulum_short_gap": "true",
                "pendulum_medium_settle": "true",
                "pendulum_strict_settle": "false",
                "pendulum_sustained_settle": "false",
                "pendulum_axis_supported_sustained_settle": "true",
                "pendulum_profile_quota_ready": "false",
                "precision_class": "short_temporal_calibrated",
            }
        ],
    )

    report = adapter.run(
        type(
            "Args",
            (),
            {
                "selected_csv": str(selected),
                "output_dir": str(tmp_path / "out"),
                "pendulum_settle_label_csv": str(pendulum),
                "mountaincar_per_env_csv": str(mountaincar),
            },
        )()
    )

    rows = list(csv.DictReader(Path(report["outputs"]["csv"]).open("r", encoding="utf-8")))
    assert rows[0]["pendulum_label_available"] == "True"
    assert rows[0]["pendulum_axis_support"] == "True"
    assert rows[0]["pendulum_short_gap"] == "True"
    assert rows[0]["pendulum_medium_settle"] == "True"
    assert rows[0]["pendulum_strict_settle"] == "False"
    assert rows[0]["pendulum_sustained_settle"] == "True"
    assert rows[0]["pendulum_axis_supported_sustained_settle"] == "True"
    assert rows[0]["pendulum_profile_quota_ready"] == "False"
    assert rows[0]["pendulum_short_temporal_settle_guard"] == "False"
    assert rows[1]["mountaincar_label_available"] == "True"
    assert rows[1]["mountaincar_short_future_basin_guard"] == "True"
    assert rows[2]["mountaincar_delayed_future_basin_guard"] == "True"
    assert report["read"]["generator_cached_label_schema_ready"] is True
    assert report["read"]["pendulum_axis_support_populated"] is True
    assert report["read"]["pendulum_profile_quota_ready"] is False
    assert report["read"]["mountaincar_quota_ready"] is False


def test_missing_labels_are_explicit_not_silent_negatives(tmp_path: Path):
    selected = tmp_path / "selected.csv"
    _write_csv(selected, [{"rule": "gym_full_obs_action_range", "env_seed": "10"}])

    report = adapter.run(
        type(
            "Args",
            (),
            {
                "selected_csv": str(selected),
                "output_dir": str(tmp_path / "out"),
                "pendulum_settle_label_csv": "",
                "mountaincar_per_env_csv": "",
            },
        )()
    )

    rows = list(csv.DictReader(Path(report["outputs"]["csv"]).open("r", encoding="utf-8")))
    assert rows[0]["pendulum_label_available"] == "False"
    assert rows[0]["mountaincar_label_available"] == "False"
    assert report["contract"]["missing_labels_are_explicit"] is True


def test_selected_pool_axis_support_is_not_confused_with_sustained_settle(tmp_path: Path):
    selected = tmp_path / "source_pool" / "selected_prior_envs.csv"
    _write_csv(
        selected,
        [
            {
                "rule": "gym_full_obs_action_range",
                "env_seed": "10",
                "pendulum_settle_yield_axis_enabled": "true",
            }
        ],
    )
    pendulum = tmp_path / "pendulum.csv"
    _write_csv(
        pendulum,
        [
            {
                "rule": "gym_full_obs_action_range",
                "source_index": "0",
                "pendulum_short_temporal_settle_guard": "false",
                "pendulum_label_available": "true",
                "precision_class": "suffix_state_bounded_guard",
            }
        ],
    )

    report = adapter.run(
        type(
            "Args",
            (),
            {
                "selected_csv": str(selected),
                "output_dir": str(tmp_path / "out"),
                "pendulum_settle_label_csv": str(pendulum),
                "mountaincar_per_env_csv": "",
            },
        )()
    )

    rows = list(csv.DictReader(Path(report["outputs"]["csv"]).open("r", encoding="utf-8")))
    assert rows[0]["pendulum_axis_support"] == "True"
    assert rows[0]["pendulum_axis_support_source"] == "selected_pool_axis"
    assert rows[0]["pendulum_sustained_settle"] == "False"
    assert rows[0]["pendulum_axis_supported_sustained_settle"] == "False"
    assert rows[0]["pendulum_profile_quota_ready"] == "False"
    assert rows[0]["pendulum_short_temporal_settle_guard"] == "False"
