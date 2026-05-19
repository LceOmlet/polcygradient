import math
from pathlib import Path

import pytest

from scripts.exploratory import phase2_feedback_subprofile_branch_selector as branch


def _row(idx, *, profile="constant", gain=2.0, sign="pos", feature=0, source="s0", score=1.0):
    if profile == "constant":
        prefix = "obs_linear"
    elif profile == "decay":
        prefix = "decay_obs_linear"
    elif profile == "early_then_zero":
        prefix = "early_obs_linear"
    else:
        prefix = "obs_linear"
    sign_part = "_neg" if sign == "neg" else ""
    gain_part = {0.5: "50", 1.0: "100", 2.0: "200"}[gain]
    mode = f"{prefix}{sign_part}_{feature}x{gain_part}"
    return {
        "source_tag": source,
        "seed": str(1000 + idx),
        "full_frozen_h_json": f"/tmp/frozen_{idx}.json",
        "signature_score": str(score),
        "best_feedback_mode": mode,
    }


def test_parse_feedback_mode_profiles_and_gain():
    assert branch._parse_mode("obs_linear_neg_0x200") == {
        "parsed_mode": "obs_linear_neg_0x200",
        "parsed_profile": "constant",
        "parsed_gain": 2.0,
        "parsed_sign": "neg",
        "parsed_feature": 0,
    }
    assert branch._parse_mode("decay_obs_linear_3x100")["parsed_profile"] == "decay"
    assert branch._parse_mode("early_obs_linear_neg_2x50")["parsed_profile"] == "early_then_zero"
    assert math.isnan(branch._parse_mode("bad")["parsed_gain"])


def test_branch_selection_reduces_constant_and_gain2_on_synthetic_pool():
    rows = []
    idx = 0
    for source in ("a", "b", "c"):
        for _ in range(5):
            rows.append(_row(idx, profile="constant", gain=2.0, sign="pos", source=source, score=idx))
            idx += 1
        rows.append(_row(idx, profile="decay", gain=1.0, sign="neg", source=source, score=idx))
        idx += 1
        rows.append(_row(idx, profile="early_then_zero", gain=0.5, sign="neg", source=source, score=idx))
        idx += 1
    loaded = []
    for source_idx, row in enumerate(rows):
        parsed = branch._parse_mode(row["best_feedback_mode"])
        row.update(parsed)
        row["source_index"] = source_idx
        row["env_seed"] = int(row["seed"])
        row["signature_score_float"] = float(row["signature_score"])
        row["is_constant_profile"] = parsed["parsed_profile"] == "constant"
        row["is_gain2"] = abs(parsed["parsed_gain"] - 2.0) < 1e-9
        row["is_nonconstant_non_gain2"] = (not row["is_constant_profile"]) and (not row["is_gain2"])
        loaded.append(row)
    cfg = dict(branch.DEFAULT_CONFIG)
    cfg["target_n"] = 9
    cfg["profile_constant_max_rate"] = 1 / 3
    cfg["gain2_max_rate"] = 1 / 3
    cfg["sign_min_rate"] = 1 / 3
    cfg["sign_max_rate"] = 2 / 3
    cfg["source_tag_max_rate"] = 0.45
    cfg["feature_max_rate"] = 1.0

    selected, info = branch._select(loaded, cfg)

    summary = branch._summary(selected)
    assert info["guard_pass"] is True
    assert summary["profile_counts"]["constant"] <= 3
    assert summary["gain_counts"]["2.0"] <= 3
    assert summary["nonconstant_non_gain2_count"] == 6


def test_branch_selector_materializes_local_artifact_when_input_exists(tmp_path):
    if not Path(branch.DEFAULT_INPUT_CSV).exists():
        pytest.skip("local strict feedback candidate pool is absent")

    class Args:
        input_csv = str(branch.DEFAULT_INPUT_CSV)
        branch_config = None
        target_n = 32
        output_dir = str(tmp_path)

    branch.run(Args())

    report = tmp_path / "feedback_strict_subprofile_branch_report.json"
    selected = tmp_path / "selected_feedback_strict_subprofile_branch.csv"
    assert report.exists()
    assert selected.exists()
    text = report.read_text(encoding="utf-8")
    assert "feedback_quantity_not_raised" in text
    assert "selection_only_frontier" in text
