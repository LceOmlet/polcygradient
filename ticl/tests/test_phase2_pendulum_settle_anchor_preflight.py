import json
from pathlib import Path

from scripts.exploratory import phase2_pendulum_settle_anchor_preflight as preflight


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _summary(ret: float, late_upright: float, early_action: float, late_action: float) -> dict:
    return {
        "raw_episode_return": {"mean": ret},
        "late_upright_frac_abs_theta_lt_0_5": {"mean": late_upright},
        "early_abs_action": {"mean": early_action},
        "late_abs_action": {"mean": late_action},
        "late_abs_theta": {"mean": 0.1 if late_upright > 0.8 else 2.0},
    }


def test_pendulum_settle_anchor_defined_but_not_currently_reproduced(tmp_path: Path):
    mechanism = _write_json(
        tmp_path / "mechanism.json",
        {
            "summaries": {
                "official_sb3_single_env_policy": _summary(-150.0, 1.0, 0.9, 0.34),
                "fit_policy": _summary(-1200.0, 0.02, 0.02, 0.02),
                "zero": _summary(-1100.0, 0.08, 0.0, 0.0),
                "random": _summary(-1000.0, 0.10, 1.0, 1.0),
            }
        },
    )
    early = _write_json(
        tmp_path / "official_earlycarry_frac025.json",
        {"summaries": {"official_sb3_single_env_policy_early_then_zero": _summary(-550.0, 0.55, 0.9, 0.0)}},
    )
    control = _write_json(
        tmp_path / "control.json",
        {
            "current_evidence": {
                "current_read": {
                    "mode_is_not_clean_enough": "not clean",
                    "pendulum_failure_location": "needs sustained settle",
                }
            }
        },
    )

    report = preflight.build_report(mechanism_json=mechanism, early_jsons=[early], control_audit_json=control)

    assert report["read"]["closed_loop_settle_anchor_defined"] is True
    assert report["read"]["closed_loop_settle_anchor_pass"] is False
    assert report["read"]["fit_current_fails_anchor"] is True
    assert report["read"]["early_only_insufficient"] is True
    assert "no_profile_or_fit_candidate_reproduces_sustained_settle" in report["read"]["blockers"]


def test_pendulum_settle_anchor_requires_early_only_falsification(tmp_path: Path):
    mechanism = _write_json(
        tmp_path / "mechanism.json",
        {
            "summaries": {
                "official_sb3_single_env_policy": _summary(-150.0, 1.0, 0.9, 0.34),
                "fit_policy": _summary(-1200.0, 0.02, 0.02, 0.02),
                "zero": _summary(-1100.0, 0.08, 0.0, 0.0),
                "random": _summary(-1000.0, 0.10, 1.0, 1.0),
            }
        },
    )
    early = _write_json(
        tmp_path / "official_earlycarry_frac050.json",
        {"summaries": {"official_sb3_single_env_policy_early_then_zero": _summary(-250.0, 0.90, 0.9, 0.0)}},
    )
    control = _write_json(tmp_path / "control.json", {"current_evidence": {"current_read": {}}})

    report = preflight.build_report(mechanism_json=mechanism, early_jsons=[early], control_audit_json=control)

    assert report["read"]["closed_loop_settle_anchor_defined"] is False
    assert "early_only_not_falsified" in report["read"]["blockers"]
