import json
from pathlib import Path

from scripts.exploratory import phase2_pendulum_settle_guard_search as search


def _write_report(path: Path) -> Path:
    lists = []
    for label in ("full", "q90"):
        rows = []
        for idx in range(16):
            target = idx < 4
            rows.append(
                {
                    "env_index": idx,
                    "pendulum_like_settle_strict": target,
                    "feedback_action_decay": target,
                    "suffix_supported": target or idx in {4, 5},
                    "time_profile_core": idx < 10,
                    "feedback_suffix_gain_over_zero_env": float(10 - idx),
                    "feedback_suffix_gain_over_open_env": float(8 - idx),
                    "best_feedback_action_abs_prefix_minus_suffix": float(4 - idx),
                }
            )
        lists.append({"label": label, "classification": {"per_env": rows}})
    path.write_text(json.dumps({"lists": lists}), encoding="utf-8")
    return path


def test_guard_search_finds_simple_ready_formula(tmp_path: Path):
    report_path = _write_report(tmp_path / "report.json")

    report = search.build_report(
        signature_reports=[report_path],
        max_conjunction_size=2,
        min_count=4,
        min_rate=0.20,
        min_precision=0.90,
        min_recall=0.90,
    )

    assert report["contract"]["bounded_search_only"] is True
    assert report["decision"]["cheap_formula_guard_ready"] is True
    assert report["ready_candidates"]


def test_guard_search_respects_count_and_clean_thresholds(tmp_path: Path):
    report_path = _write_report(tmp_path / "report.json")

    report = search.build_report(
        signature_reports=[report_path],
        max_conjunction_size=1,
        min_count=12,
        min_rate=0.50,
        min_precision=0.90,
        min_recall=0.90,
    )

    assert report["decision"]["cheap_formula_guard_ready"] is False
    assert "No simple bounded formula" in report["decision"]["read"]
