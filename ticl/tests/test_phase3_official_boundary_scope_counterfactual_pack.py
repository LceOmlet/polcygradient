import json
import sys

from ticl.analysis import phase3_official_boundary_scope_counterfactual_pack as pack_mod


def _scale_chain_payload(
    *,
    suite_name: str,
    env_index: int,
    selected_target_local_index: int,
    terminal_global_step: int,
    full_mean_raw: float,
    full_std_raw: float,
    objective_mean_raw: float,
    objective_std_raw: float,
    current_boundary_shift_norm: float,
    objective_boundary_shift_norm: float,
    current_reward_norm: float,
    objective_reward_norm: float,
    objective_std_ratio_vs_full: float,
    full_to_objective_variance_multiple: float,
    within_prefix_share: float,
    between_prefix_objective_share: float,
    dominant_objective_variance_source: str,
    between_episode_means_share: float,
) -> dict:
    return {
        "audit_entry": "phase3_terminal_local_scale_chain_probe",
        "config": {
            "suite_name": suite_name,
            "env_index": env_index,
            "selected_target_local_index": selected_target_local_index,
            "single_eval_pos": 1946,
            "objective_terminal_reset_count_from_distance_probe": 1,
            "objective_terminal_row_count_from_token_rows": 1,
        },
        "selected_target_local_readout": {
            "global_step": terminal_global_step,
            "objective_episode_index": 0,
            "objective_local_index": selected_target_local_index,
        },
        "boundary_contract": {
            "full_rollout_mean_raw": full_mean_raw,
            "full_rollout_std_raw": full_std_raw,
            "full_rollout_mean_over_std": full_mean_raw / full_std_raw,
            "objective_subset_mean_raw": objective_mean_raw,
            "objective_subset_std_raw": objective_std_raw,
            "objective_subset_mean_over_std": objective_mean_raw / objective_std_raw,
            "objective_subset_gap_vs_full_mean_over_std": (objective_mean_raw / objective_std_raw)
            - (full_mean_raw / full_std_raw),
        },
        "mean_source": {
            "objective_mean_rollout_return_raw": objective_mean_raw,
            "implied_prefix_mean_rollout_return_raw": full_mean_raw,
        },
        "scale_collapse": {
            "scale_collapse_readout": {
                "objective_std_ratio_vs_full": objective_std_ratio_vs_full,
                "objective_variance_ratio_vs_full": objective_std_ratio_vs_full * objective_std_ratio_vs_full,
                "full_to_objective_variance_multiple": full_to_objective_variance_multiple,
            },
            "full_variance_decomposition": {
                "within_prefix_share": within_prefix_share,
                "between_prefix_objective_share": between_prefix_objective_share,
            },
            "objective_variance_decomposition": {
                "dominant_objective_variance_source": dominant_objective_variance_source,
                "between_episode_means_share": between_episode_means_share,
            },
        },
        "conclusions": {
            "all_terminal_candidates_match_full_rollout_centering_contract": True,
            "all_terminal_candidates_worsen_under_objective_scope": True,
            "objective_scope_std_collapse_is_explained_by_truncating_away_high_variance_prefix": True,
            "objective_window_is_low_variance_slice_not_high_variance_source": True,
        },
        "runtime_wall_s_total": 12.5,
        "terminal_local_sections": [
            {
                "scope_rows": [
                    {
                        "scope_name": "current_full_rollout",
                        "local_boundary_shift_norm": current_boundary_shift_norm,
                        "local_reward_norm_before_value_term": current_reward_norm,
                    },
                    {
                        "scope_name": "objective_all_tokens",
                        "local_boundary_shift_norm": objective_boundary_shift_norm,
                        "local_reward_norm_before_value_term": objective_reward_norm,
                    },
                ]
            }
        ],
    }


def test_build_pack_from_minimal_official_saved_payloads(monkeypatch):
    seed = _scale_chain_payload(
        suite_name="seed13579",
        env_index=12,
        selected_target_local_index=90,
        terminal_global_step=2036,
        full_mean_raw=-28.0,
        full_std_raw=24.0,
        objective_mean_raw=-4.0,
        objective_std_raw=2.5,
        current_boundary_shift_norm=1.1,
        objective_boundary_shift_norm=1.5,
        current_reward_norm=1.0,
        objective_reward_norm=1.4,
        objective_std_ratio_vs_full=0.1,
        full_to_objective_variance_multiple=100.0,
        within_prefix_share=0.97,
        between_prefix_objective_share=0.02,
        dominant_objective_variance_source="within_episode0",
        between_episode_means_share=0.03,
    )
    pair2 = _scale_chain_payload(
        suite_name="pair2",
        env_index=13,
        selected_target_local_index=37,
        terminal_global_step=1983,
        full_mean_raw=-3.8,
        full_std_raw=3.1,
        objective_mean_raw=-0.38,
        objective_std_raw=0.23,
        current_boundary_shift_norm=1.22,
        objective_boundary_shift_norm=1.69,
        current_reward_norm=1.35,
        objective_reward_norm=3.52,
        objective_std_ratio_vs_full=0.07,
        full_to_objective_variance_multiple=187.0,
        within_prefix_share=0.93,
        between_prefix_objective_share=0.06,
        dominant_objective_variance_source="within_episode1",
        between_episode_means_share=0.004,
    )
    pair1 = _scale_chain_payload(
        suite_name="pair1",
        env_index=13,
        selected_target_local_index=22,
        terminal_global_step=1968,
        full_mean_raw=0.50,
        full_std_raw=0.51,
        objective_mean_raw=0.23,
        objective_std_raw=0.12,
        current_boundary_shift_norm=-0.99,
        objective_boundary_shift_norm=-1.90,
        current_reward_norm=-1.25,
        objective_reward_norm=-3.01,
        objective_std_ratio_vs_full=0.24,
        full_to_objective_variance_multiple=17.5,
        within_prefix_share=0.98,
        between_prefix_objective_share=0.015,
        dominant_objective_variance_source="within_episode1",
        between_episode_means_share=0.059,
    )
    payload_map = {
        "seed_can.json": seed,
        "seed_rep.json": seed,
        "pair2_can.json": pair2,
        "pair2_rep.json": pair2,
        "pair1_can.json": pair1,
        "pair1_rep.json": pair1,
    }

    monkeypatch.setattr(
        pack_mod,
        "OFFICIAL_SAVED_POINTS",
        (
            {"point_name": "seed13579_env12", "canonical_json": "seed_can.json", "repeat_json": "seed_rep.json"},
            {"point_name": "pair2_env13", "canonical_json": "pair2_can.json", "repeat_json": "pair2_rep.json"},
            {"point_name": "pair1_env13", "canonical_json": "pair1_can.json", "repeat_json": "pair1_rep.json"},
        ),
    )

    def _fake_loader(path: str) -> dict:
        return payload_map[path]

    monkeypatch.setattr(pack_mod, "_load_scale_chain_payload", _fake_loader)

    result = pack_mod._build_pack()

    assert result["audit_entry"] == "phase3_official_boundary_scope_counterfactual_pack"
    assert result["aggregate"]["official_point_count"] == 3
    assert result["aggregate"]["all_points_replay_stable"] is True
    assert result["aggregate"]["all_points_current_match_full_rollout_centering_contract"] is True
    assert result["aggregate"]["all_points_objective_counterfactual_worsens_boundary_penalty"] is True
    assert result["aggregate"]["all_points_would_shrink_boundary_abs_under_full_vs_objective"] is True
    assert result["aggregate"]["all_points_would_shrink_reward_norm_abs_under_full_vs_objective"] is True
    assert result["conclusions"]["official_saved_points_support_full_rollout_centering_as_cross_point_guardrail"] is True
    assert result["conclusions"]["pack_is_guardrail_evidence_not_yet_new_runtime_fix"] is True


def test_point_summary_raises_when_repeat_locked_fields_drift():
    payload = _scale_chain_payload(
        suite_name="pair1",
        env_index=13,
        selected_target_local_index=22,
        terminal_global_step=1968,
        full_mean_raw=0.50,
        full_std_raw=0.51,
        objective_mean_raw=0.23,
        objective_std_raw=0.12,
        current_boundary_shift_norm=-0.99,
        objective_boundary_shift_norm=-1.90,
        current_reward_norm=-1.25,
        objective_reward_norm=-3.01,
        objective_std_ratio_vs_full=0.24,
        full_to_objective_variance_multiple=17.5,
        within_prefix_share=0.98,
        between_prefix_objective_share=0.015,
        dominant_objective_variance_source="within_episode1",
        between_episode_means_share=0.059,
    )
    repeat = json.loads(json.dumps(payload))
    repeat["boundary_contract"]["objective_subset_std_raw"] = 0.13

    try:
        pack_mod._point_summary("pair1_env13", payload, repeat)
    except ValueError as exc:
        assert "did not match on locked fields" in str(exc)
    else:
        raise AssertionError("Expected locked-field mismatch to raise.")


def test_main_reuses_existing_output(tmp_path, monkeypatch, capsys):
    output_path = tmp_path / "phase3_official_boundary_scope_counterfactual_pack.json"
    payload = {"ok": True, "kind": "phase3_official_boundary_scope_counterfactual_pack"}
    output_path.write_text(json.dumps(payload))

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "phase3_official_boundary_scope_counterfactual_pack.py",
            "--output-json",
            str(output_path),
        ],
    )

    assert pack_mod.main() == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == payload
