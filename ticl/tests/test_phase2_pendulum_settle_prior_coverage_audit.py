import json
from pathlib import Path

from scripts.exploratory import phase2_pendulum_settle_prior_coverage_audit as audit


def _write_report(path: Path, counts: list[int]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "lists": [
                    {
                        "label": f"list_{idx}",
                        "classification": {
                            "coverage": {
                                "n_envs": 64,
                                "pendulum_like_settle_strict_count": count,
                                "pendulum_like_strict_count": 20,
                                "time_profile_core_count": 18,
                                "suffix_supported_count": 30,
                            }
                        },
                    }
                    for idx, count in enumerate(counts)
                ]
            }
        ),
        encoding="utf-8",
    )
    return path


def test_pendulum_settle_prior_coverage_blocks_sparse_candidates(tmp_path: Path):
    report_path = _write_report(tmp_path / "sig.json", [4, 8])

    report = audit.build_report(signature_reports=[report_path])

    assert report["read"]["settle_candidates_exist"] is True
    assert report["read"]["quota_ready"] is False
    assert report["read"]["min_settle_count"] == 4
    assert report["read"]["blockers"] == ["settle_strict_prior_coverage_too_sparse"]


def test_pendulum_settle_prior_coverage_passes_when_all_lists_have_quota(tmp_path: Path):
    report_path = _write_report(tmp_path / "sig.json", [12, 14])

    report = audit.build_report(signature_reports=[report_path])

    assert report["read"]["quota_ready"] is True
    assert report["read"]["blockers"] == []
