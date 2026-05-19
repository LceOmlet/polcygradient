from pathlib import Path

from scripts.exploratory import phase2_profile_v2e_candidate_annotator as annotator


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.write_text(
        ",".join(fields)
        + "\n"
        + "\n".join(",".join(str(row.get(key, "")) for key in fields) for row in rows)
        + "\n",
        encoding="utf-8",
    )


def test_v2e_candidate_annotator_merges_near_best_and_mountaincar_without_selection(tmp_path):
    selected = []
    near = []
    mc = []
    for rule in annotator.RULES:
        selected.append(
            {
                "candidate_rule": rule,
                "_source_index": 7,
                "profile_band_rank": 0,
                "env_seed": 123,
                "full_frozen_h_json": tmp_path / f"{rule}.json",
                "strict_feedback_candidate": True,
                "strict_feedback_profile": "decay",
                "locomotion_witness": True,
                "low_energy_not_ctrl_only": False,
                "sign_or_phase_sensitive": True,
                "state_conditioned_energy_injection": False,
            }
        )
        Path(selected[-1]["full_frozen_h_json"]).write_text('{"a": 1}\n', encoding="utf-8")
        near.append(
            {
                "rule": rule,
                "source_index": 7,
                "profile": "decay",
                "best_gain": 2.0,
                "near_best_gain_1": True,
                "near_best_gain_0.5": False,
                "gap_gain_1": 1.0,
                "gap_gain_0.5": 9.0,
            }
        )
        mc.append(
            {
                "source_label": rule,
                "env_index": 7,
                "mountaincar_future_basin_candidate": True,
                "mountaincar_future_basin_strong": False,
                "mountaincar_delayed_candidate": False,
            }
        )
    selected_csv = tmp_path / "selected.csv"
    near_csv = tmp_path / "near.csv"
    mc_csv = tmp_path / "mc.csv"
    _write_csv(selected_csv, selected)
    _write_csv(near_csv, near)
    _write_csv(mc_csv, mc)

    rows, report = annotator.build_candidate(
        selected_csv=selected_csv,
        near_best_csv=near_csv,
        mountaincar_csv=mc_csv,
    )

    assert len(rows) == 2
    assert rows[0]["profile_version_id"] == "profile_band_v2e_thin_annotated_candidate"
    assert rows[0]["v2e_strict_near_best_gain_1"] == "True"
    assert rows[0]["mountaincar_future_basin_candidate"] == "True"
    assert report["contract"]["annotation_only"] is True
    assert report["read"]["complexity_safe"] is True
    assert report["read"]["v2e_profile_annotation_complete"] is True
