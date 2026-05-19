import csv
import json
from pathlib import Path

from scripts.exploratory import phase2_profile_chunked_label_contract as contract


def _write_selected(path: Path, n_per_rule: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for rule in contract.RULES:
        for idx in range(n_per_rule):
            h = path.parent / f"{rule}_{idx}.json"
            h.write_text("{}", encoding="utf-8")
            rows.append({"rule": rule, "env_seed": str(1000 + idx), "full_frozen_h_json": str(h)})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["rule", "env_seed", "full_frozen_h_json"])
        writer.writeheader()
        writer.writerows(rows)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0]) if rows else ["x"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def test_chunk_manifest_preserves_source_ranges(tmp_path: Path):
    selected = tmp_path / "selected.csv"
    _write_selected(selected, n_per_rule=5)

    report = contract.run(
        type(
            "Args",
            (),
            {
                "selected_csv": str(selected),
                "output_dir": str(tmp_path / "out"),
                "chunk_size": 2,
                "device": "cuda:0",
                "n_steps_action": 16,
                "n_steps_locomotion": 16,
                "n_steps_feedback": 8,
                "feedback_label_mode": "budgeted_guard",
                "feedback_baseline_modes": "zero,random,neg_random,half_random",
                "feedback_profiles": "decay",
                "feedback_gains": "1.0",
                "feedback_obs_linear_count": 2,
                "max_feedback_eval_units_per_chunk": 500_000,
                "merge_existing": False,
            },
        )()
    )

    assert report["chunk_count"] == 6
    first = report["chunks"][0]
    assert first["rule"] == "gym_full_obs_action_range"
    assert first["source_start_index"] == 0
    assert first["source_end_index_exclusive"] == 2
    assert first["count"] == 2
    assert "phase2_prior_action_mode_coverage_probe.py" in first["command_text"]["action_mode_labels"]
    assert "--feedback-profiles" in first["commands"]["feedback_near_best_labels"]
    assert "--profiles" not in first["commands"]["feedback_near_best_labels"]
    assert "phase2_future_basin_cached_label_adapter.py" in first["command_text"]["future_basin_cached_labels"]
    assert report["feedback_eval_budget"]["label_mode"] == "budgeted_guard"
    assert report["feedback_eval_budget"]["gains"] == [1.0]
    assert report["feedback_eval_budget"]["modes_per_rule"] == 8
    assert report["feedback_eval_budget"]["chunk_size_within_feedback_budget"] is True


def test_merge_existing_offsets_chunk_indices(tmp_path: Path):
    selected = tmp_path / "selected.csv"
    _write_selected(selected, n_per_rule=3)
    args = type(
        "Args",
        (),
        {
            "selected_csv": str(selected),
            "output_dir": str(tmp_path / "out"),
            "chunk_size": 2,
            "device": "cuda:0",
            "n_steps_action": 16,
            "n_steps_locomotion": 16,
            "n_steps_feedback": 8,
            "feedback_label_mode": "budgeted_guard",
            "feedback_baseline_modes": "zero,random,neg_random,half_random",
            "feedback_profiles": "decay",
            "feedback_gains": "1.0",
            "feedback_obs_linear_count": 2,
            "max_feedback_eval_units_per_chunk": 500_000,
            "merge_existing": False,
        },
    )()
    report = contract.run(args)

    # Populate only two chunk outputs.  Merge should offset env/source indices
    # by each chunk's original source_start_index.
    for chunk in report["chunks"]:
        if chunk["rule"] != "gym_full_obs_action_range":
            continue
        action_path = Path(chunk["action_mode_json"])
        action_path.parent.mkdir(parents=True, exist_ok=True)
        action_path.write_text(
            json.dumps(
                {
                    "lists": [
                        {
                            "label": chunk["rule"],
                            "classification": {
                                "per_env": [
                                    {"env_index": idx, "low_energy_not_ctrl_only": True}
                                    for idx in range(chunk["count"])
                                ]
                            },
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        _write_csv(
            Path(chunk["locomotion_csv"]),
            [{"scope": chunk["rule"], "env_index": idx, "sustained_locomotion_temporal_witness": "true"} for idx in range(chunk["count"])],
        )
        _write_csv(
            Path(chunk["feedback_near_best_csv"]),
            [{"rule": chunk["rule"], "source_index": idx, "env_idx": idx, "profile": "decay"} for idx in range(chunk["count"])],
        )
        _write_csv(
            Path(chunk["future_basin_cached_labels_csv"]),
            [
                {
                    "rule": chunk["rule"],
                    "source_index": idx,
                    "pendulum_label_available": "False",
                    "mountaincar_label_available": "True",
                }
                for idx in range(chunk["count"])
            ],
        )

    report["merged_outputs"] = contract._merge_existing(report, tmp_path / "out")
    merged_action = json.loads(Path(report["merged_outputs"]["action_mode"]["path"]).read_text(encoding="utf-8"))
    per_env = merged_action["lists"][0]["classification"]["per_env"]
    assert [row["env_index"] for row in per_env] == [0, 1, 2]

    with Path(report["merged_outputs"]["locomotion"]["path"]).open("r", encoding="utf-8") as handle:
        loco_rows = list(csv.DictReader(handle))
    assert [int(row["env_index"]) for row in loco_rows] == [0, 1, 2]

    with Path(report["merged_outputs"]["feedback_near_best"]["path"]).open("r", encoding="utf-8") as handle:
        feedback_rows = list(csv.DictReader(handle))
    assert [int(row["source_index"]) for row in feedback_rows] == [0, 1, 2]

    with Path(report["merged_outputs"]["future_basin_cached_labels"]["path"]).open("r", encoding="utf-8") as handle:
        future_rows = list(csv.DictReader(handle))
    assert [int(row["source_index"]) for row in future_rows] == [0, 1, 2]


def test_merge_existing_preserves_global_source_indices(tmp_path: Path):
    selected = tmp_path / "selected.csv"
    _write_selected(selected, n_per_rule=3)
    args = type(
        "Args",
        (),
        {
            "selected_csv": str(selected),
            "output_dir": str(tmp_path / "out"),
            "chunk_size": 2,
            "device": "cuda:0",
            "n_steps_action": 16,
            "n_steps_locomotion": 16,
            "n_steps_feedback": 8,
            "feedback_label_mode": "budgeted_guard",
            "feedback_baseline_modes": "zero,random,neg_random,half_random",
            "feedback_profiles": "decay",
            "feedback_gains": "1.0",
            "feedback_obs_linear_count": 2,
            "max_feedback_eval_units_per_chunk": 500_000,
            "merge_existing": False,
        },
    )()
    report = contract.run(args)

    # Real chunked feedback/future-basin labels inherit `_source_index` from
    # the chunk CSV, so source_index is already global.  The merge must not
    # offset it a second time; only env_idx remains local for feedback rows.
    for chunk in report["chunks"]:
        if chunk["rule"] != "gym_full_obs_action_range":
            continue
        offset = int(chunk["source_start_index"])
        _write_csv(
            Path(chunk["feedback_near_best_csv"]),
            [
                {
                    "rule": chunk["rule"],
                    "source_index": offset + idx,
                    "env_idx": idx,
                    "profile": "decay",
                }
                for idx in range(chunk["count"])
            ],
        )
        _write_csv(
            Path(chunk["future_basin_cached_labels_csv"]),
            [
                {
                    "rule": chunk["rule"],
                    "source_index": offset + idx,
                    "pendulum_label_available": "False",
                    "mountaincar_label_available": "True",
                }
                for idx in range(chunk["count"])
            ],
        )

    report["merged_outputs"] = contract._merge_existing(report, tmp_path / "out")

    with Path(report["merged_outputs"]["feedback_near_best"]["path"]).open("r", encoding="utf-8") as handle:
        feedback_rows = list(csv.DictReader(handle))
    assert [int(row["source_index"]) for row in feedback_rows] == [0, 1, 2]
    assert [int(row["env_idx"]) for row in feedback_rows] == [0, 1, 2]

    with Path(report["merged_outputs"]["future_basin_cached_labels"]["path"]).open("r", encoding="utf-8") as handle:
        future_rows = list(csv.DictReader(handle))
    assert [int(row["source_index"]) for row in future_rows] == [0, 1, 2]


def test_execute_missing_runs_bounded_stage_subset(tmp_path: Path, monkeypatch):
    selected = tmp_path / "selected.csv"
    _write_selected(selected, n_per_rule=3)
    calls = []

    def _fake_run(command, cwd, check):
        calls.append({"command": list(command), "cwd": cwd, "check": check})

    monkeypatch.setattr(contract.subprocess, "run", _fake_run)

    report = contract.run(
        type(
            "Args",
            (),
            {
                "selected_csv": str(selected),
                "output_dir": str(tmp_path / "out"),
                "chunk_size": 2,
                "device": "cuda:0",
                "n_steps_action": 16,
                "n_steps_locomotion": 16,
                "n_steps_feedback": 8,
                "feedback_label_mode": "budgeted_guard",
                "feedback_baseline_modes": "zero,random,neg_random,half_random",
                "feedback_profiles": "decay",
                "feedback_gains": "1.0",
                "feedback_obs_linear_count": 2,
                "max_feedback_eval_units_per_chunk": 500_000,
                "execute_missing": True,
                "execute_stages": "fixed_groups",
                "max_execute_chunks": 1,
                "execute_rule": "gym_full_obs_action_range",
                "allow_expensive_feedback": False,
                "merge_existing": False,
            },
        )()
    )

    assert len(calls) == 1
    assert "phase2_selected_prior_envs_to_fixed_groups.py" in calls[0]["command"][1]
    assert calls[0]["check"] is True
    assert report["execution"]["selected_chunk_count"] == 1
    assert report["execution"]["executed_count"] == 1
