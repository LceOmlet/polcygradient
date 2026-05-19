import numpy as np

from scripts.exploratory import phase2_feedback_near_best_gain_audit as audit


def test_mode_name_and_parse_roundtrip():
    mode = audit._mode_name(profile="decay", feature=3, gain=1.0, sign="neg")
    parsed = audit._parse_mode(mode)

    assert mode == "decay_obs_linear_neg_3x100"
    assert parsed["profile"] == "decay"
    assert parsed["gain"] == 1.0
    assert parsed["feature"] == 3
    assert parsed["sign"] == "neg"


def test_near_best_gain_classification_detects_argmax_hard_case():
    returns = {
        "obs_linear_0x50": np.asarray([9.0, 0.0], dtype=np.float32),
        "obs_linear_neg_0x50": np.asarray([8.0, 0.0], dtype=np.float32),
        "obs_linear_0x100": np.asarray([9.5, 0.0], dtype=np.float32),
        "obs_linear_neg_0x100": np.asarray([9.4, 0.0], dtype=np.float32),
        "obs_linear_0x200": np.asarray([10.0, 10.0], dtype=np.float32),
        "obs_linear_neg_0x200": np.asarray([9.8, 9.0], dtype=np.float32),
    }
    rows = [
        {"_source_index": 0, "strict_feedback_candidate": True},
        {"_source_index": 1, "strict_feedback_candidate": True},
    ]

    per_env, summary = audit._classify_profile(
        returns_by_mode=returns,
        profile="constant",
        gains=[0.5, 1.0, 2.0],
        margin=np.asarray([1.0, 1.0], dtype=np.float32),
        rule="synthetic",
        rows=rows,
    )

    assert len(per_env) == 2
    assert summary["best_gain2_rate"] == 1.0
    assert summary["strict_near_best_gain_1_rate"] == 0.5
    assert audit._read_from_summary(summary, gains=[0.5, 1.0, 2.0])["verdict"] == (
        "argmax_hard_near_best_definition_likely_sufficient"
    )


def test_near_best_gain_classification_detects_source_pool_bias_case():
    returns = {
        "obs_linear_0x50": np.asarray([0.0, 0.0], dtype=np.float32),
        "obs_linear_neg_0x50": np.asarray([0.0, 0.0], dtype=np.float32),
        "obs_linear_0x100": np.asarray([0.0, 0.0], dtype=np.float32),
        "obs_linear_neg_0x100": np.asarray([0.0, 0.0], dtype=np.float32),
        "obs_linear_0x200": np.asarray([10.0, 10.0], dtype=np.float32),
        "obs_linear_neg_0x200": np.asarray([9.0, 9.0], dtype=np.float32),
    }
    rows = [
        {"_source_index": 0, "strict_feedback_candidate": True},
        {"_source_index": 1, "strict_feedback_candidate": True},
    ]

    _, summary = audit._classify_profile(
        returns_by_mode=returns,
        profile="constant",
        gains=[0.5, 1.0, 2.0],
        margin=np.asarray([1.0, 1.0], dtype=np.float32),
        rule="synthetic",
        rows=rows,
    )

    assert audit._read_from_summary(summary, gains=[0.5, 1.0, 2.0])["verdict"] == (
        "source_pool_or_generator_high_gain_bias"
    )
