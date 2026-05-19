import csv
from pathlib import Path

from scripts.exploratory import phase2_profile_v2e_pack_fit_parity_preflight as preflight


def _write_selected(path: Path, frozen: Path) -> None:
    rows = []
    for rule in preflight.RULES:
        for idx in range(64):
            rows.append(
                {
                    "rule": rule,
                    "env_seed": str(1000 + idx),
                    "full_frozen_h_json": str(frozen),
                    "profile_version_id": "profile_band_v2e_thin_annotated_candidate",
                    "profile_band_guard": "unified_profile_band_selector",
                    "locomotion_witness": str(idx % 2 == 0),
                    "low_energy_not_ctrl_only": str(idx % 3 == 0),
                    "sign_or_phase_sensitive": str(idx % 4 == 0),
                    "state_conditioned_energy_injection": str(idx % 5 == 0),
                }
            )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_v2e_pack_fit_preflight_checks_fixed_list_and_helpers(tmp_path: Path):
    frozen = tmp_path / "h.json"
    frozen.write_text("{}", encoding="utf-8")
    selected = tmp_path / "selected.csv"
    parity = tmp_path / "parity.py"
    gated = tmp_path / "gated.py"
    _write_selected(selected, frozen)
    parity.write_text(
        "\n".join(f"def {helper}(): pass" for helper in preflight.REQUIRED_HELPERS),
        encoding="utf-8",
    )
    gated.write_text("selected_prior_envs.csv\n_load_selected_fixed_list\n", encoding="utf-8")

    report = preflight.build_report(
        selected_csv=selected,
        parity_script=parity,
        gated_parity_script=gated,
    )

    assert report["preflight_pass"] is True
    assert report["selected_csv_read"]["rule_summary"]["gym_full_obs_action_range"][
        "can_form_n64_fixed_list"
    ] is True
    assert all(report["parity_helper_read"]["required_helpers_present"].values())


def test_v2e_pack_fit_preflight_fails_missing_columns(tmp_path: Path):
    selected = tmp_path / "selected.csv"
    parity = tmp_path / "parity.py"
    gated = tmp_path / "gated.py"
    selected.write_text("rule,env_seed\nx,1\n", encoding="utf-8")
    parity.write_text("", encoding="utf-8")
    gated.write_text("", encoding="utf-8")

    report = preflight.build_report(
        selected_csv=selected,
        parity_script=parity,
        gated_parity_script=gated,
    )

    assert report["preflight_pass"] is False
    assert "full_frozen_h_json" in report["selected_csv_read"]["missing_required_columns"]
