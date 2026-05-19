import csv
from pathlib import Path

from scripts.exploratory import phase2_pendulum_settle_cached_label_export as export


def test_export_cached_labels_preserves_rule_and_source_index():
    report = {
        "lists": [
            {
                "label": "gym_full_obs_action_range",
                "classification": {
                    "per_env": [
                        {
                            "env_index": 2,
                            "medium_settle_guard": True,
                            "cheap_settle_guard": True,
                            "reward_condition": True,
                            "suffix_reward_condition": True,
                            "action_decay_condition": True,
                            "self_state_settle_condition": True,
                            "open_state_settle_condition": True,
                            "best_decay_mode": "decay_obs_linear_0x100",
                        }
                    ]
                },
            }
        ]
    }

    rows = export.export_cached_labels(report)

    assert rows[0]["rule"] == "gym_full_obs_action_range"
    assert rows[0]["source_index"] == 2
    assert rows[0]["pendulum_short_temporal_settle_guard"] is True
    assert rows[0]["pendulum_medium_settle"] is True
    assert rows[0]["pendulum_strict_settle"] is True
    assert rows[0]["pendulum_sustained_settle"] is True
    assert rows[0]["pendulum_profile_quota_ready"] is False
    assert rows[0]["pendulum_label_available"] is True


def test_run_writes_cached_label_csv(tmp_path: Path):
    input_json = tmp_path / "in.json"
    input_json.write_text(
        """
        {
          "lists": [
            {
              "label": "gym_q90_obs_action_range",
              "classification": {
                "per_env": [
                  {"env_index": 0, "cheap_settle_guard": false},
                  {"env_index": 1, "cheap_settle_guard": true}
                ]
              }
            }
          ]
        }
        """,
        encoding="utf-8",
    )

    report = export.run(
        type(
            "Args",
            (),
            {
                "input_json": str(input_json),
                "output_dir": str(tmp_path / "out"),
                "min_count": 1,
                "min_rate": 0.1,
            },
        )()
    )

    rows = list(csv.DictReader(Path(report["outputs"]["csv"]).open("r", encoding="utf-8")))
    assert rows[1]["pendulum_short_temporal_settle_guard"] == "True"
    assert rows[1]["pendulum_strict_settle"] == "True"
    assert report["read"]["cached_label_populated"] is True
