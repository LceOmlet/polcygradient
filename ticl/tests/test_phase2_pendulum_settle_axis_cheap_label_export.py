import csv
import json
from pathlib import Path

from scripts.exploratory import phase2_pendulum_settle_axis_cheap_label_export as export


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


def test_selected_csv_axis_labels_are_ordinal_by_rule(tmp_path: Path):
    selected = tmp_path / "selected.csv"
    _write_csv(
        selected,
        [
            {"rule": "gym_full_obs_action_range", "pendulum_settle_yield_axis_enabled": "true"},
            {"rule": "gym_full_obs_action_range", "pendulum_settle_yield_axis_enabled": "false"},
            {"rule": "gym_q90_obs_action_range", "pendulum_settle_yield_axis_enabled": "true"},
        ],
    )

    rows = export.rows_from_selected_csv(selected)

    assert rows[0]["source_index"] == 0
    assert rows[1]["source_index"] == 1
    assert rows[2]["source_index"] == 0
    assert rows[0]["pendulum_axis_support"] is True
    assert rows[0]["pendulum_short_gap"] is False
    assert rows[0]["pendulum_sustained_settle"] is False
    assert rows[0]["pendulum_profile_quota_ready"] is False
    assert rows[0]["pendulum_short_temporal_settle_guard"] is False
    assert rows[1]["pendulum_axis_support"] is False


def test_fixed_h_export_and_reference_calibration(tmp_path: Path):
    fixed = tmp_path / "gym_full_obs_action_range" / "prior_fixed_h_list.json"
    fixed.parent.mkdir(parents=True)
    fixed.write_text(
        json.dumps(
            [
                {"pendulum_settle_yield_repair_enabled": True},
                {"pendulum_settle_yield_repair_enabled": False},
            ]
        ),
        encoding="utf-8",
    )
    reference = tmp_path / "reference.json"
    reference.write_text(
        json.dumps(
            {
                "lists": [
                    {
                        "label": "gym_full_obs_action_range",
                        "classification": {
                            "per_env": [
                                {"env_index": 0, "cheap_settle_guard": False},
                                {"env_index": 1, "cheap_settle_guard": False},
                            ]
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    report = export.run(
        type(
            "Args",
            (),
            {
                "selected_csv": "",
                "fixed_h_list_json": [str(fixed)],
                "reference_preflight_json": str(reference),
                "output_dir": str(tmp_path / "out"),
            },
        )()
    )

    rows = list(csv.DictReader(Path(report["outputs"]["csv"]).open("r", encoding="utf-8")))
    assert rows[0]["pendulum_axis_support"] == "True"
    assert rows[0]["pendulum_short_temporal_settle_guard"] == "False"
    assert report["calibration"]["by_rule"]["gym_full_obs_action_range"]["fp"] == 1
    assert report["read"]["cheap_label_quota_ready"] is False
    assert report["read"]["axis_support_label_populated"] is True
    assert report["read"]["sustained_settle_label_populated"] is False
