import json
from pathlib import Path

from scripts.exploratory import phase2_pendulum_settle_light_guard_audit as audit


def _env(**overrides):
    row = {
        "pendulum_like_settle_strict": False,
        "time_profile_core": False,
        "time_profile_suffix_supported": False,
        "feedback_action_decay": False,
        "suffix_supported": False,
        "phase_sensitive_feedback": False,
        "feedback_beats_open_loop_env": False,
        "pendulum_like_core": False,
        "early_only_causal_carryover_core": False,
        "early_only_strong_carryover_core": False,
    }
    row.update(overrides)
    return row


def _write_report(path: Path, rows):
    path.write_text(
        json.dumps(
            {
                "lists": [
                    {
                        "label": "toy",
                        "classification": {"per_env": rows},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return path


def test_light_guard_audit_rejects_broad_impure_guard(tmp_path: Path):
    rows = [
        _env(pendulum_like_settle_strict=True, time_profile_core=True, time_profile_suffix_supported=True),
        _env(time_profile_core=True, time_profile_suffix_supported=True),
        _env(time_profile_core=True, time_profile_suffix_supported=True),
    ]
    path = _write_report(tmp_path / "sig.json", rows)

    report = audit.build_report(
        signature_reports=[path],
        min_count=3,
        min_rate=0.5,
        min_precision=0.5,
        min_recall=0.8,
    )

    assert report["by_guard"]["time_core_suffix"]["count_ready"] is True
    assert report["by_guard"]["time_core_suffix"]["clean_ready"] is False
    assert "time_core_suffix" in report["decision"]["broad_but_impure_guards"]
    assert report["decision"]["cheap_guard_ready"] is False


def test_light_guard_audit_accepts_broad_clean_guard(tmp_path: Path):
    rows = [
        _env(
            pendulum_like_settle_strict=True,
            feedback_action_decay=True,
            suffix_supported=True,
        ),
        _env(
            pendulum_like_settle_strict=True,
            feedback_action_decay=True,
            suffix_supported=True,
        ),
    ]
    path = _write_report(tmp_path / "sig.json", rows)

    report = audit.build_report(
        signature_reports=[path],
        min_count=2,
        min_rate=0.5,
        min_precision=0.9,
        min_recall=0.9,
    )

    assert report["by_guard"]["feedback_decay_suffix"]["candidate_ready"] is True
    assert "feedback_decay_suffix" in report["decision"]["ready_guards"]
    assert report["decision"]["cheap_guard_ready"] is True
