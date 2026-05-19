from scripts.exploratory import phase2_profile_v2e_decision_summary as summary


def test_v2e_summary_keeps_mountaincar_out_of_quota_and_delta_out_of_gate(tmp_path):
    selector = tmp_path / "selector.json"
    selector.write_text(
        """{
          "rules": {
            "gym_full_obs_action_range": {
              "guard_pass": true,
              "selected_summary": {
                "strict_feedback_count": 20,
                "strict_feedback_profile_counts": {"decay": 13, "constant": 7},
                "strict_feedback_gain_counts": {"2.0": 16, "1.0": 2, "0.5": 2},
                "joint_profile_dominant_rate": 0.18,
                "joint_profile_entropy_norm": 0.86
              },
              "band_details": {"strict_feedback_subprofile_branch": {"selected_nonconstant_count": 13, "selected_non_gain2_count": 4}}
            }
          }
        }""",
        encoding="utf-8",
    )
    near = tmp_path / "near.json"
    near.write_text(
        """{
          "overall_read": {"any_source_pool_high_gain_bias": false, "any_argmax_hard": true},
          "rules": {
            "gym_full_obs_action_range": {
              "profiles": {
                "constant": {
                  "read": {"best_gain2_rate": 0.8, "verdict": "argmax_hard_near_best_definition_likely_sufficient"},
                  "strict_near_best_gain_1_rate": 0.6,
                  "strict_near_best_gain_0.5_rate": 0.3
                }
              }
            }
          }
        }""",
        encoding="utf-8",
    )
    pressure = tmp_path / "pressure.json"
    pressure.write_text(
        """{"mountaincar_prior_fixedlist_guard": {"guard_pass": false, "scope_summaries": {}}}""",
        encoding="utf-8",
    )
    source = tmp_path / "source.json"
    source.write_text(
        """{"read": {"temporal_union_repairs_profile": true}}""",
        encoding="utf-8",
    )

    report = summary.build_report(
        selector_json=selector,
        near_best_json=near,
        next_pressure_json=pressure,
        source_pool_json=source,
    )

    assert report["contract"]["delta_sign_fraction_not_a_gate"] is True
    assert "MountainCar quota" in report["v2e_candidate_read"]["exclude_from_v2e_for_now"]
    assert "hard gain quota" in report["v2e_candidate_read"]["exclude_from_v2e_for_now"]
    assert report["v2e_candidate_read"]["profile_definition_close_enough_for_candidate"] is True
    assert report["integrated_smoke"]["available"] is False


def test_v2e_summary_records_integrated_smoke_when_available(tmp_path):
    selector = tmp_path / "selector.json"
    selector.write_text(
        """{
          "rules": {
            "gym_full_obs_action_range": {
              "guard_pass": true,
              "selected_summary": {},
              "band_details": {"strict_feedback_subprofile_branch": {}}
            }
          }
        }""",
        encoding="utf-8",
    )
    near = tmp_path / "near.json"
    near.write_text(
        """{"overall_read": {"any_source_pool_high_gain_bias": false}, "rules": {}}""",
        encoding="utf-8",
    )
    pressure = tmp_path / "pressure.json"
    pressure.write_text(
        """{"mountaincar_prior_fixedlist_guard": {"guard_pass": false}}""",
        encoding="utf-8",
    )
    source = tmp_path / "source.json"
    source.write_text("""{"read": {"temporal_union_repairs_profile": true}}""", encoding="utf-8")
    smoke = tmp_path / "smoke.json"
    smoke.write_text(
        """{
          "overall": {
            "candidate_smoke_guard_pass": true,
            "profile_guard_pass": true,
            "diversity_guard_pass": true,
            "trainability_delta_guard_pass": true
          },
          "scopes": {
            "gym_full_obs_action_range": {
              "smoke": {
                "raw_reward_sum_delta": {"mean": 1.0},
                "reward_env_sum_delta": {"mean": 2.0},
                "diversity_guard": {"pass": true}
              }
            }
          }
        }""",
        encoding="utf-8",
    )

    report = summary.build_report(
        selector_json=selector,
        near_best_json=near,
        next_pressure_json=pressure,
        source_pool_json=source,
        smoke_json=smoke,
    )

    assert report["integrated_smoke"]["available"] is True
    assert report["integrated_smoke"]["guard_pass"] is True
    assert "complete" in report["v2e_candidate_read"]["remaining_before_milestone"][0]
