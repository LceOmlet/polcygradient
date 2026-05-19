import json
from pathlib import Path

from scripts.exploratory import phase2_feedback_source_pool_broaden_audit as audit


def _mode(profile: str, gain: str = "200") -> str:
    if profile == "constant":
        return f"obs_linear_0x{gain}"
    if profile == "decay":
        return f"decay_obs_linear_0x{gain}"
    if profile == "early_then_zero":
        return f"early_obs_linear_0x{gain}"
    raise AssertionError(profile)


def test_feedback_source_pool_audit_distinguishes_label_broadening_from_gain_repair(tmp_path: Path):
    rows = []
    for idx in range(160):
        rows.append(
            {
                "pendulum_like_strict": True,
                "pendulum_like_core": True,
                "pendulum_like_weak": True,
                "time_profile_core": False,
                "best_feedback_mode": _mode("constant"),
                "best_time_profile_mode": "None",
                "best_feedback_minus_open_env": 2.0,
                "feedback_pair_phase_gap_env": 1.0,
                "feedback_suffix_gain_over_zero_env": 1.0,
            }
        )
    for profile in ("decay", "early_then_zero"):
        for idx in range(80):
            rows.append(
                {
                    "pendulum_like_strict": False,
                    "pendulum_like_core": False,
                    "pendulum_like_weak": False,
                    "time_profile_core": True,
                    "best_feedback_mode": _mode("constant"),
                    "best_time_profile_mode": _mode(profile),
                    "best_time_profile_env_return": 4.0,
                    "best_open_loop_env_return": 1.0,
                }
            )
    path = tmp_path / "signature.json"
    path.write_text(
        json.dumps({"lists": [{"label": "full_synthetic_seed6260", "classification": {"per_env": rows}}]}),
        encoding="utf-8",
    )

    report = audit.build_report([path])

    assert report["contract"]["does_not_train"] is True
    assert report["contract"]["profile_quality_first"] is True
    assert report["read"]["strict_source_pool_collapse_confirmed"] is True
    assert report["read"]["core_or_weak_relaxation_sufficient"] is False
    assert report["read"]["temporal_union_repairs_profile"] is True
    assert report["read"]["temporal_union_repairs_gain"] is False
    assert "gain=2.0" in report["read"]["recommended_next"]


def test_feedback_source_pool_markdown_has_no_delta_positive_guard_language(tmp_path: Path):
    rows = [
        {
            "pendulum_like_strict": True,
            "pendulum_like_core": True,
            "pendulum_like_weak": True,
            "time_profile_core": False,
            "best_feedback_mode": _mode("constant"),
        }
    ]
    path = tmp_path / "signature.json"
    path.write_text(
        json.dumps({"lists": [{"label": "q90_synthetic_seed6261", "classification": {"per_env": rows}}]}),
        encoding="utf-8",
    )
    report = audit.build_report([path])

    text = audit._markdown(report)

    assert "delta+" not in text.lower()
    assert "raw positive" not in text.lower()
    assert "profile dominant" in text
