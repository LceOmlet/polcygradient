import json
from pathlib import Path

from scripts.exploratory import phase2_profile_next_pressure_audit as audit


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.write_text(
        ",".join(keys)
        + "\n"
        + "\n".join(",".join(str(row.get(key, "")) for key in keys) for row in rows)
        + "\n",
        encoding="utf-8",
    )


def test_next_pressure_audit_keeps_delta_positive_as_monitor(tmp_path: Path):
    selected = []
    mountaincar = []
    smoke = []
    for rule in audit.RULES:
        for idx in range(2):
            selected.append(
                {
                    "candidate_rule": rule,
                    "profile_band_rank": idx,
                    "_source_index": idx,
                }
            )
            mountaincar.append(
                {
                    "source_label": rule,
                    "env_index": idx,
                    "mountaincar_future_basin_candidate": idx == 0,
                    "mountaincar_future_basin_strong": idx == 0,
                    "mountaincar_delayed_candidate": False,
                    "mountaincar_future_basin_vs_zero": idx == 0,
                    "mountaincar_future_basin_vs_open": idx == 0,
                    "best_early_only_mode": "early_random_then_zero",
                }
            )
            smoke.append(
                {
                    "rule": rule,
                    "env_idx": idx,
                    "raw_reward_sum_delta": 1.0 if idx == 0 else -1.0,
                    "reward_env_sum_delta": 2.0 if idx == 0 else -2.0,
                }
            )
    selected_csv = tmp_path / "selected.csv"
    mountaincar_csv = tmp_path / "mountaincar.csv"
    smoke_csv = tmp_path / "smoke.csv"
    _write_csv(selected_csv, selected)
    _write_csv(mountaincar_csv, mountaincar)
    _write_csv(smoke_csv, smoke)

    feedback_control = tmp_path / "feedback_control.json"
    feedback_control.write_text(json.dumps({"simple_knob_sufficient": False}), encoding="utf-8")
    strict = tmp_path / "strict.json"
    strict.write_text(
        json.dumps(
            {
                "pool_summary": {
                    "profile_dominance": {"dominant": "constant", "dominant_share": 0.9},
                    "gain_dominance": {"dominant": "2.0", "dominant_share": 0.8},
                },
                "decision": {"selection_only_is_sufficient": False},
            }
        ),
        encoding="utf-8",
    )

    report, delta_rows = audit.build_report(
        selected_csv=selected_csv,
        mountaincar_csv=mountaincar_csv,
        smoke_csvs=[smoke_csv],
        feedback_control_json=feedback_control,
        strict_frontier_json=strict,
    )

    assert report["contract"]["delta_positive_fraction_monitor_only"] is True
    assert report["feedback_source_pool_read"]["pressure_type"] == "source_pool_subprofile_collapse"
    assert report["mountaincar_prior_fixedlist_guard"]["guard_pass"] is False
    assert {row["mountaincar_key"] for row in delta_rows} == set(audit.MOUNTAINCAR_KEYS)


def test_next_pressure_audit_can_run_prior_coverage_without_smoke_delta(tmp_path: Path):
    selected = []
    mountaincar = []
    for rule in audit.RULES:
        selected.append({"candidate_rule": rule, "profile_band_rank": 0, "_source_index": 0})
        mountaincar.append(
            {
                "source_label": rule,
                "env_index": 0,
                "mountaincar_future_basin_candidate": True,
                "mountaincar_future_basin_strong": False,
                "mountaincar_delayed_candidate": False,
                "mountaincar_future_basin_vs_zero": True,
                "mountaincar_future_basin_vs_open": False,
            }
        )
    selected_csv = tmp_path / "selected.csv"
    mountaincar_csv = tmp_path / "mountaincar.csv"
    _write_csv(selected_csv, selected)
    _write_csv(mountaincar_csv, mountaincar)
    feedback_control = tmp_path / "feedback_control.json"
    feedback_control.write_text(json.dumps({"simple_knob_sufficient": False}), encoding="utf-8")
    strict = tmp_path / "strict.json"
    strict.write_text(
        json.dumps(
            {
                "pool_summary": {
                    "profile_dominance": {"dominant": "constant", "dominant_share": 0.9},
                    "gain_dominance": {"dominant": "2.0", "dominant_share": 0.8},
                },
                "decision": {"selection_only_is_sufficient": False},
            }
        ),
        encoding="utf-8",
    )

    report, delta_rows = audit.build_report(
        selected_csv=selected_csv,
        mountaincar_csv=mountaincar_csv,
        smoke_csvs=[],
        feedback_control_json=feedback_control,
        strict_frontier_json=strict,
    )

    assert delta_rows == []
    assert report["contract"]["profile_quality_first"] is True
    assert "selected prior fixed lists" in report["mountaincar_prior_fixedlist_guard"]["read"]
