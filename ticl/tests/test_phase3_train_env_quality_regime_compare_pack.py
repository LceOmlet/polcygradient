import json

from ticl.analysis.phase3_train_env_quality_regime_compare_pack import (
    build_train_env_quality_regime_compare_pack,
)


def _write(tmp_path, name: str, payload: dict) -> str:
    path = tmp_path / name
    path.write_text(json.dumps(payload))
    return str(path)


def test_build_train_env_quality_regime_compare_pack_extracts_shared_and_regime_specific_signals(tmp_path):
    positive = _write(
        tmp_path,
        "positive.json",
        {
            "suite_context": {
                "suite_name": "pair1",
                "suite_regime": "positive",
                "train_suite_summary_fingerprint_match": True,
                "suite_regime_summary_membership_verified": True,
            },
            "conclusions": {
                "high_positive_mass_alone_is_not_sufficient_for_positive_delta": True,
                "episode_segmentation_not_primary_separator": True,
                "low_mass_positive_envs_have_higher_pre_suffix_gap_than_high_mass_nonpositive": True,
                "low_mass_positive_envs_have_higher_value_corr_than_high_mass_nonpositive": True,
                "early_mass_distinguishes_better_than_tail_mass": True,
            },
            "contrasts": {
                "low_mass_positive_minus_high_mass_nonpositive": {
                    "episode_count": 0.0,
                    "first16_positive_share": 2.0,
                    "last16_positive_share": -1.0,
                    "pre_suffix_gap": 4.0,
                    "raw_value_return_corr": 0.8,
                    "terminal_reset_count": 0.0,
                }
            },
        },
    )
    nonpositive = _write(
        tmp_path,
        "nonpositive.json",
        {
            "suite_context": {
                "suite_name": "seed24680",
                "suite_regime": "nonpositive",
                "train_suite_summary_fingerprint_match": True,
                "suite_regime_summary_membership_verified": True,
            },
            "conclusions": {
                "high_positive_mass_alone_is_not_sufficient_for_positive_delta": True,
                "episode_segmentation_not_primary_separator": True,
                "low_mass_positive_envs_have_higher_pre_suffix_gap_than_high_mass_nonpositive": True,
                "low_mass_positive_envs_have_higher_value_corr_than_high_mass_nonpositive": True,
                "early_mass_distinguishes_better_than_tail_mass": False,
            },
            "contrasts": {
                "low_mass_positive_minus_high_mass_nonpositive": {
                    "episode_count": 0.0,
                    "first16_positive_share": 1.0,
                    "last16_positive_share": 0.25,
                    "pre_suffix_gap": 0.5,
                    "raw_value_return_corr": 0.1,
                    "terminal_reset_count": 0.0,
                }
            },
        },
    )

    report = build_train_env_quality_regime_compare_pack(
        positive_contrast_json=positive,
        nonpositive_contrast_json=nonpositive,
    )

    assert report["audit_entry"] == "phase3_train_env_quality_regime_compare_pack"
    assert report["shared_signals"]["high_positive_mass_alone_is_not_sufficient_for_positive_delta"] is True
    assert report["shared_signals"]["episode_segmentation_not_primary_separator"] is True
    assert report["shared_signals"]["low_mass_positive_envs_have_higher_pre_suffix_gap_than_high_mass_nonpositive"] is True
    assert report["shared_signals"]["low_mass_positive_envs_have_higher_value_corr_than_high_mass_nonpositive"] is True
    assert report["regime_specific_signals"]["positive_only_early_mass_distinguishes_better_than_tail_mass"] is True
    assert report["regime_specific_signals"]["nonpositive_early_mass_distinguishes_better_than_tail_mass"] is False
    assert report["regime_specific_signals"]["contrast_amplification"]["pre_suffix_gap_ratio"] == 8.0
    assert report["regime_specific_signals"]["contrast_amplification"]["last16_positive_share_sign_flip"] is True
    assert report["recommendation"]["regime_split_guardrail_required"] is True
    assert report["recommendation"]["weighting_repair_candidate_supported"] is False
    assert report["conclusions"]["shared_signals_consistent_across_regimes"] is True
    assert report["conclusions"]["regime_specific_early_mass_separator_is_positive_only"] is True


def test_build_train_env_quality_regime_compare_pack_rejects_bad_regime_context(tmp_path):
    positive = _write(
        tmp_path,
        "positive.json",
        {
            "suite_context": {
                "suite_name": "pair1",
                "suite_regime": "positive",
                "train_suite_summary_fingerprint_match": True,
                "suite_regime_summary_membership_verified": True,
            },
            "conclusions": {
                "high_positive_mass_alone_is_not_sufficient_for_positive_delta": True,
                "episode_segmentation_not_primary_separator": True,
                "low_mass_positive_envs_have_higher_pre_suffix_gap_than_high_mass_nonpositive": True,
                "low_mass_positive_envs_have_higher_value_corr_than_high_mass_nonpositive": True,
                "early_mass_distinguishes_better_than_tail_mass": True,
            },
            "contrasts": {
                "low_mass_positive_minus_high_mass_nonpositive": {
                    "episode_count": 0.0,
                    "first16_positive_share": 1.0,
                    "last16_positive_share": 0.0,
                    "pre_suffix_gap": 1.0,
                    "raw_value_return_corr": 0.1,
                    "terminal_reset_count": 0.0,
                }
            },
        },
    )
    bad_nonpositive = _write(
        tmp_path,
        "bad_nonpositive.json",
        {
            "suite_context": {
                "suite_name": "seed24680",
                "suite_regime": "positive",
                "train_suite_summary_fingerprint_match": True,
                "suite_regime_summary_membership_verified": True,
            },
            "conclusions": {
                "high_positive_mass_alone_is_not_sufficient_for_positive_delta": True,
                "episode_segmentation_not_primary_separator": True,
                "low_mass_positive_envs_have_higher_pre_suffix_gap_than_high_mass_nonpositive": True,
                "low_mass_positive_envs_have_higher_value_corr_than_high_mass_nonpositive": True,
                "early_mass_distinguishes_better_than_tail_mass": False,
            },
            "contrasts": {
                "low_mass_positive_minus_high_mass_nonpositive": {
                    "episode_count": 0.0,
                    "first16_positive_share": 1.0,
                    "last16_positive_share": 0.0,
                    "pre_suffix_gap": 1.0,
                    "raw_value_return_corr": 0.1,
                    "terminal_reset_count": 0.0,
                }
            },
        },
    )

    try:
        build_train_env_quality_regime_compare_pack(
            positive_contrast_json=positive,
            nonpositive_contrast_json=bad_nonpositive,
        )
    except ValueError as exc:
        assert "Unexpected suite regime" in str(exc)
    else:
        raise AssertionError("expected suite regime mismatch to raise")
