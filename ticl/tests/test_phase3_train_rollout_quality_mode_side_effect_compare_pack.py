import json

from ticl.analysis.phase3_train_rollout_quality_mode_side_effect_compare_pack import (
    build_train_rollout_quality_mode_side_effect_compare_pack,
)


def _write(tmp_path, name: str, payload: dict) -> str:
    path = tmp_path / name
    path.write_text(json.dumps(payload))
    return str(path)


def _payload(*, positive_mass_delta: float) -> dict:
    return {
        "audit_entry": "phase3_train_rollout_quality_probe",
        "config": {
            "train_suite_path": "/tmp/train_suite.pt",
            "heldout_suite_path": "/tmp/heldout_suite.pt",
            "n_samples": 2048,
            "single_eval_pos": 1946,
        },
        "train_suite_summary": {"fingerprint": "train_fp"},
        "heldout_suite_summary": {"fingerprint": "heldout_fp"},
        "env_items": [
            {
                "env_index": 0,
                "normalized_actor_adv_positive_mass": 10.0,
                "normalized_actor_adv_abs_mass": 12.0,
                "normalized_actor_adv_first16_positive_share": 0.1,
                "normalized_actor_adv_last16_positive_share": 0.2,
                "raw_actor_adv_positive_mass": 3.0,
                "raw_actor_adv_abs_mass": 4.0,
                "raw_value_return_corr": 0.25,
                "pre_vs_zero_suffix_gap": 1.0,
                "objective_episode_count": 1,
                "objective_terminal_reset_count": 0,
            },
            {
                "env_index": 1,
                "normalized_actor_adv_positive_mass": 20.0 + positive_mass_delta,
                "normalized_actor_adv_abs_mass": 22.0 + positive_mass_delta,
                "normalized_actor_adv_first16_positive_share": 0.3,
                "normalized_actor_adv_last16_positive_share": 0.4,
                "raw_actor_adv_positive_mass": 6.0,
                "raw_actor_adv_abs_mass": 7.0,
                "raw_value_return_corr": 0.45,
                "pre_vs_zero_suffix_gap": 2.0,
                "objective_episode_count": 2,
                "objective_terminal_reset_count": 1,
            },
        ],
    }


def test_build_train_rollout_quality_mode_side_effect_compare_pack_detects_nonzero_delta(tmp_path):
    baseline = _write(tmp_path, "baseline.json", _payload(positive_mass_delta=0.0))
    override = _write(tmp_path, "override.json", _payload(positive_mass_delta=2.5))

    report = build_train_rollout_quality_mode_side_effect_compare_pack(
        baseline_json=baseline,
        override_json=override,
    )

    assert report["audit_entry"] == "phase3_train_rollout_quality_mode_side_effect_compare_pack"
    assert report["suite_context"]["train_suite_fingerprint"] == "train_fp"
    assert report["conclusions"]["runtime_mode_no_other_suite_side_effects"] is False
    assert report["conclusions"]["side_effects_are_detectable_in_train_side_ab"] is True
    assert report["comparison"]["most_shifted_env_index"] == 1
    assert report["comparison"]["metric_deltas"]["normalized_actor_adv_positive_mass"]["max_abs_delta"] == 2.5


def test_build_train_rollout_quality_mode_side_effect_compare_pack_rejects_suite_contract_mismatch(tmp_path):
    baseline = _write(tmp_path, "baseline.json", _payload(positive_mass_delta=0.0))
    mismatch = _payload(positive_mass_delta=0.0)
    mismatch["train_suite_summary"] = {"fingerprint": "different"}
    override = _write(tmp_path, "override.json", mismatch)

    try:
        build_train_rollout_quality_mode_side_effect_compare_pack(
            baseline_json=baseline,
            override_json=override,
        )
    except ValueError as exc:
        assert "same suite contract" in str(exc)
    else:
        raise AssertionError("expected suite contract mismatch to raise")
