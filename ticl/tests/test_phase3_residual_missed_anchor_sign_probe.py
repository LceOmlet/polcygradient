import json
import sys

from ticl.analysis import phase3_residual_missed_anchor_sign_probe as probe_mod


def test_sign_state_classifies_positive_balanced_and_wrong_sign():
    assert probe_mod._sign_state(3.0, 1.0) == "positive"
    assert probe_mod._sign_state(1.0, 1.0) == "near_zero_or_balanced"
    assert probe_mod._sign_state(1.0, 3.0) == "wrong_sign"
    assert probe_mod._sign_state(0.0, 0.0) == "zero"


def test_build_report_summarizes_missed_anchor_sign_diagnoses():
    anchor_payload = {
        "strong_gain_anchor_count": 3,
        "captured_summary": {"count": 1},
        "missed_summary": {
            "count": 2,
            "suite_counts": {"synthetic": 2},
            "env_items": [
                {
                    "suite_name": "synthetic",
                    "env_index": 0,
                    "dist_mass_share_within_suite": 0.0,
                    "dist_mass_rank_within_suite": 3,
                },
                {
                    "suite_name": "synthetic",
                    "env_index": 1,
                    "dist_mass_share_within_suite": 0.01,
                    "dist_mass_rank_within_suite": 2,
                },
            ],
        },
    }
    suite_rows = {
        "synthetic": [
            {
                "suite_name": "synthetic",
                "env_index": 0,
                "env_seed": 10,
                "rollout_seed": 20,
                "pre_vs_zero_suffix_gap": 5.0,
                "pre_vs_zero_full_gap": 9.0,
                "pre_gap_rank_within_suite": 1,
                "normalized_actor_adv_positive_mass_rank_within_suite": 3,
                "rank_gap_pre_to_norm_adv": 2,
                "objective_token_count": 100,
                "objective_terminal_reset_count": 1,
                "raw_value_return_corr": 0.2,
                "raw_return_sign_state": "positive",
                "raw_residual_sign_state": "wrong_sign",
                "raw_actor_adv_sign_state": "wrong_sign",
                "normalized_actor_adv_sign_state": "wrong_sign",
                "raw_return_positive_mass_share_within_suite": 0.40,
                "raw_residual_positive_mass_share_within_suite": 0.05,
                "raw_actor_adv_positive_mass_share_within_suite": 0.03,
                "normalized_actor_adv_positive_mass_share_within_suite": 0.02,
                "return_to_residual_flipped_positive_mass_share": 0.75,
                "residual_to_actor_flipped_positive_mass_share": 0.20,
                "actor_to_norm_flipped_positive_mass_share": 0.05,
                "dominant_share_drop_stage": "baseline",
                "dominant_flip_stage": "baseline",
                "diagnosis": "baseline_cancellation",
            },
            {
                "suite_name": "synthetic",
                "env_index": 1,
                "env_seed": 11,
                "rollout_seed": 21,
                "pre_vs_zero_suffix_gap": 3.0,
                "pre_vs_zero_full_gap": 6.0,
                "pre_gap_rank_within_suite": 2,
                "normalized_actor_adv_positive_mass_rank_within_suite": 2,
                "rank_gap_pre_to_norm_adv": 0,
                "objective_token_count": 100,
                "objective_terminal_reset_count": 0,
                "raw_value_return_corr": 0.6,
                "raw_return_sign_state": "positive",
                "raw_residual_sign_state": "positive",
                "raw_actor_adv_sign_state": "positive",
                "normalized_actor_adv_sign_state": "near_zero_or_balanced",
                "raw_return_positive_mass_share_within_suite": 0.15,
                "raw_residual_positive_mass_share_within_suite": 0.14,
                "raw_actor_adv_positive_mass_share_within_suite": 0.13,
                "normalized_actor_adv_positive_mass_share_within_suite": 0.04,
                "return_to_residual_flipped_positive_mass_share": 0.05,
                "residual_to_actor_flipped_positive_mass_share": 0.10,
                "actor_to_norm_flipped_positive_mass_share": 0.55,
                "dominant_share_drop_stage": "normalization",
                "dominant_flip_stage": "normalization",
                "diagnosis": "adv_normalization",
            },
        ]
    }
    suite_summaries = [
        {
            "suite_name": "synthetic",
            "env_count": 2,
            "objective_token_total": 200,
            "suite_runtime_wall_s": 1.0,
            "suite_pre_minus_zero_suffix": 1.0,
        }
    ]

    report = probe_mod._build_report(
        anchor_source="positive_regime_missed",
        anchor_payload=anchor_payload,
        suite_rows=suite_rows,
        suite_summaries=suite_summaries,
    )

    assert report["aggregate"]["count"] == 2
    assert report["aggregate"]["diagnosis_counts"]["baseline_cancellation"] == 1
    assert report["aggregate"]["diagnosis_counts"]["adv_normalization"] == 1
    assert report["aggregate"]["final_normalized_actor_adv_sign_state_counts"]["wrong_sign"] == 1
    assert report["aggregate"]["final_normalized_actor_adv_sign_state_counts"]["near_zero_or_balanced"] == 1
    assert report["conclusions"]["missed_anchor_count_matches_anchor_probe"] is True
    assert report["conclusions"]["wrong_sign_exists_among_missed_anchors"] is True
    assert report["conclusions"]["near_zero_or_balanced_exists_among_missed_anchors"] is True


def test_phase3_residual_missed_anchor_sign_probe_reuses_existing_output(tmp_path, monkeypatch, capsys):
    output_path = tmp_path / "residual_missed_anchor_sign.json"
    payload = {"ok": True, "kind": "residual_missed_anchor_sign"}
    output_path.write_text(json.dumps(payload))

    monkeypatch.setattr(
        probe_mod,
        "assert_phase2_green",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not preflight when reusing output")),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "phase3_residual_missed_anchor_sign_probe.py",
            "/tmp/checkpoint.cpkt",
            "--anchor-source",
            "dual_channel_residual_missed",
            "--output-json",
            str(output_path),
        ],
    )

    assert probe_mod.main() == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == payload
