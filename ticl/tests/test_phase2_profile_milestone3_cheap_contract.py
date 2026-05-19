import json
from pathlib import Path

from scripts.exploratory import phase2_profile_milestone3_cheap_contract as contract


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _label() -> dict:
    return {
        "read": {
            "generator_cached_label_schema_ready": True,
            "pendulum_axis_support_populated": True,
            "pendulum_generator_label_populated": True,
            "mountaincar_guard_coverage_ready": False,
            "mountaincar_quota_ready": False,
        }
    }


def _selector() -> dict:
    return {
        "config": {
            "profile_version_id": "profile_band_v2e_source384_action_controls_position_cheap_state_guard"
        }
    }


def _smoke(*, steps: int = 128, gap: float = 0.014) -> dict:
    return {
        "hard_pass": True,
        "scopes": {
            rule: {
                "hard_pass": True,
                "settle_gap": {"gap_active": {"min": gap}},
                "transition_diversity": {"steps": steps},
            }
            for rule in contract.RULES
        },
    }


def _paths(tmp_path: Path, *, stability_steps: int = 128) -> dict[str, Path]:
    return {
        "label_json": _write(tmp_path / "label.json", _label()),
        "selector_json": _write(tmp_path / "selector.json", _selector()),
        "smoke_json": _write(tmp_path / "smoke.json", _smoke(steps=32)),
        "stability_smoke_json": _write(tmp_path / "stability.json", _smoke(steps=stability_steps)),
    }


def test_cheap_contract_formalizes_out_of_scope_mountaincar(tmp_path: Path):
    report = contract.build_report(**_paths(tmp_path))

    assert report["decision"]["cheap_contract_formalized"] is True
    assert report["decision"]["mountaincar_scope_decided"] is True
    assert report["cheap_contract"]["sustained_smoke_contract"]["old_one_step_min_gap_0_20_is_not_the_gate"] is True
    assert report["mountaincar_scope"]["decision"] == "out_of_scope_for_cheap_m3"


def test_cheap_contract_requires_stability_support(tmp_path: Path):
    report = contract.build_report(**_paths(tmp_path, stability_steps=32))

    assert report["decision"]["cheap_contract_formalized"] is False
    assert "cheap_stability_requirement_not_supported" in report["decision"]["blockers"]


def test_cheap_contract_rejects_mountaincar_quota_without_same_pool_labels(tmp_path: Path):
    report = contract.build_report(
        **_paths(tmp_path),
        mountaincar_scope="same_pool_labels_and_guards_attached",
    )

    assert report["decision"]["cheap_contract_formalized"] is False
    assert "mountaincar_scope_not_decided" in report["decision"]["blockers"]
