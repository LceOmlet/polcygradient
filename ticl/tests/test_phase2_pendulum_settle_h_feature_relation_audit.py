import json
from pathlib import Path

from scripts.exploratory import phase2_pendulum_settle_h_feature_relation_audit as audit


def _write_signature(tmp_path: Path, h_values, labels):
    h_path = tmp_path / "h.json"
    sig_path = tmp_path / "sig.json"
    h_path.write_text(
        json.dumps([{"feature": value, "other": idx} for idx, value in enumerate(h_values)]),
        encoding="utf-8",
    )
    sig_path.write_text(
        json.dumps(
            {
                "lists": [
                    {
                        "label": "toy",
                        "fixed_h_list_json": str(h_path),
                        "classification": {
                            "per_env": [
                                {
                                    "pendulum_like_settle_strict": label,
                                    "pendulum_like_core": label,
                                    "feedback_action_decay": label,
                                    "suffix_supported": label,
                                    "time_profile_core": label,
                                    "time_profile_suffix_supported": label,
                                }
                                for label in labels
                            ]
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return sig_path


def test_h_feature_relation_detects_ready_simple_split(tmp_path: Path):
    sig = _write_signature(
        tmp_path,
        h_values=[0, 1, 2, 3, 4, 5],
        labels=[False, False, False, True, True, True],
    )

    report = audit.build_report(
        signature_reports=[sig],
        min_selected_rate=0.5,
        min_lift=0.2,
        min_label_selected_n=2,
        min_label_target_rate=0.5,
    )

    assert report["targets"]["strict_current"]["simple_h_split_ready"] is True
    assert report["decision"]["strict_current_simple_h_knob_ready"] is True


def test_h_feature_relation_rejects_weak_split(tmp_path: Path):
    sig = _write_signature(
        tmp_path,
        h_values=[0, 1, 2, 3, 4, 5],
        labels=[False, True, False, True, False, True],
    )

    report = audit.build_report(
        signature_reports=[sig],
        min_selected_rate=0.9,
        min_lift=0.4,
        min_label_selected_n=2,
        min_label_target_rate=0.9,
    )

    assert report["targets"]["strict_current"]["simple_h_split_ready"] is False
    assert report["decision"]["strict_current_simple_h_knob_ready"] is False
