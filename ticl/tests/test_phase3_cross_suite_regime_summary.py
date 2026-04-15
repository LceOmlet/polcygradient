from ticl.analysis.phase3_cross_suite_regime_summary import (
    DEFAULT_SUITES,
    build_cross_suite_regime_summary,
)


def test_default_pair1_suite_uses_suite_matched_artifacts():
    pair1 = next(spec for spec in DEFAULT_SUITES if spec["suite_name"] == "pair1")
    assert pair1["distance_probe_json"].endswith("_pair1_suite_matched.json")
    assert pair1["tail_ablation_json"].endswith("_pair1_suite_matched.json")
    assert pair1["post_reset_ablation_json"].endswith("_pair1_suite_matched.json")


def test_build_cross_suite_regime_summary_splits_positive_and_nonpositive_regimes():
    suite_specs = [
        {
            "suite_name": "a",
            "zero_json": None,
            "ppo_json": None,
        }
    ]
    summary = build_cross_suite_regime_summary([])
    assert summary["available_suite_count"] == 0
    assert summary["regime_groups"]["positive"]["count"] == 0
    assert summary["regime_groups"]["nonpositive"]["count"] == 0


def test_build_cross_suite_regime_summary_detects_inconsistent_pre_alignment(tmp_path):
    def write(name, payload):
        path = tmp_path / name
        path.write_text(payload)
        return str(path)

    zero_a = write(
        "zero_a.json",
        """{
          "heldout_suite": {"metrics": {"suffix_return_mean": 0.0, "full_return_mean": 0.0}}
        }""",
    )
    ppo_a = write(
        "ppo_a.json",
        """{
          "heldout_suite": {"metrics": {"suffix_return_mean": 1.0, "full_return_mean": 1.0}}
        }""",
    )
    dist_a = write(
        "dist_a.json",
        """{
          "summaries": {
            "post_reset_tail_dist_1": {"corr_vs_terminal_resets": 0.5, "corr_vs_pre_suffix_gap": -0.1},
            "post_reset_tail_dist_2": {"corr_vs_terminal_resets": 0.6, "corr_vs_pre_suffix_gap": -0.2},
            "post_reset_tail_dist_3": {"corr_vs_terminal_resets": 0.7, "corr_vs_pre_suffix_gap": -0.1},
            "post_reset_tail_dist_4": {"corr_vs_terminal_resets": 0.8, "corr_vs_pre_suffix_gap": -0.2}
          }
        }""",
    )
    zero_b = write(
        "zero_b.json",
        """{
          "heldout_suite": {"metrics": {"suffix_return_mean": 1.0, "full_return_mean": 1.0}}
        }""",
    )
    ppo_b = write(
        "ppo_b.json",
        """{
          "heldout_suite": {"metrics": {"suffix_return_mean": 0.5, "full_return_mean": 0.0}}
        }""",
    )
    dist_b = write(
        "dist_b.json",
        """{
          "summaries": {
            "post_reset_tail_dist_1": {"corr_vs_terminal_resets": 0.5, "corr_vs_pre_suffix_gap": 0.1},
            "post_reset_tail_dist_2": {"corr_vs_terminal_resets": 0.6, "corr_vs_pre_suffix_gap": 0.2},
            "post_reset_tail_dist_3": {"corr_vs_terminal_resets": 0.7, "corr_vs_pre_suffix_gap": 0.1},
            "post_reset_tail_dist_4": {"corr_vs_terminal_resets": 0.8, "corr_vs_pre_suffix_gap": 0.2}
          }
        }""",
    )

    summary = build_cross_suite_regime_summary(
        [
            {"suite_name": "a", "zero_json": zero_a, "ppo_json": ppo_a, "distance_probe_json": dist_a},
            {"suite_name": "b", "zero_json": zero_b, "ppo_json": ppo_b, "distance_probe_json": dist_b},
        ]
    )

    assert summary["available_suite_count"] == 2
    assert summary["conclusions"]["dist1_to_4_reset_tracking_consistent"] is True
    assert summary["conclusions"]["dist1_to_4_pre_alignment_consistent"] is False
    assert summary["regime_groups"]["positive"]["suite_names"] == ["a"]
    assert summary["regime_groups"]["nonpositive"]["suite_names"] == ["b"]
