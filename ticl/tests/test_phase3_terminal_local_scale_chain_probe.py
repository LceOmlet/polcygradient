import json
import sys

from ticl.analysis import phase3_terminal_local_scale_chain_probe as probe_mod


def test_resolve_target_local_index_requires_or_infers_terminal_local():
    rows = [
        {"objective_local_index": 10, "next_non_terminal": 1.0},
        {"objective_local_index": 11, "next_non_terminal": 0.0},
        {"objective_local_index": 12, "next_non_terminal": 1.0},
    ]
    assert probe_mod._resolve_target_local_index(rows, None) == 11
    assert probe_mod._resolve_target_local_index(rows, 11) == 11


def test_build_terminal_local_scale_chain_probe_from_minimal_payloads():
    full_returns = [[0.0], [30.0]] * 48 + [[7.0], [6.0], [3.0], [1.0]]
    flat_returns = [float(v[0]) for v in full_returns]
    full_mean = probe_mod._mean(flat_returns)
    full_std = probe_mod._std(flat_returns)

    distance_row = {
        "env_index": 0,
        "env_seed": 1,
        "rollout_seed": 2,
        "pre_vs_zero_suffix_gap": 3.0,
        "objective_terminal_reset_count": 1,
    }
    suite_payload = {
        "rollout_discounted_raw_returns": full_returns
    }
    suite_summary = {"suite_runtime_wall_s": 1.25}
    token_rows = [
        {
            "global_step": 96,
            "objective_local_index": 0,
            "objective_episode_index": 0,
            "reward_raw": 0.2,
            "reward_norm": float(0.2 / full_std),
            "rollout_return_raw": 7.0,
            "value_norm": 0.3,
            "delta_norm": 0.1,
            "next_non_terminal": 1.0,
        },
        {
            "global_step": 97,
            "objective_local_index": 1,
            "objective_episode_index": 0,
            "reward_raw": -0.1,
            "reward_norm": float((-0.1 / full_std) - (full_mean / full_std)),
            "rollout_return_raw": 6.0,
            "value_norm": 0.2,
            "delta_norm": -1.2,
            "next_non_terminal": 0.0,
        },
        {
            "global_step": 98,
            "objective_local_index": 2,
            "objective_episode_index": 1,
            "reward_raw": 0.3,
            "reward_norm": float(0.3 / full_std),
            "rollout_return_raw": 3.0,
            "value_norm": 0.1,
            "delta_norm": 0.2,
            "next_non_terminal": 1.0,
        },
        {
            "global_step": 99,
            "objective_local_index": 3,
            "objective_episode_index": 1,
            "reward_raw": 0.2,
            "reward_norm": float(0.2 / full_std),
            "rollout_return_raw": 1.0,
            "value_norm": 0.05,
            "delta_norm": 0.1,
            "next_non_terminal": 1.0,
        },
    ]

    result = probe_mod._build_probe(
        suite_name="pair1",
        env_index=0,
        distance_row=distance_row,
        suite_payload=suite_payload,
        suite_summary=suite_summary,
        token_rows=token_rows,
        explicit_target_local_index=None,
    )

    assert result["audit_entry"] == "phase3_terminal_local_scale_chain_probe"
    assert result["config"]["selected_target_local_index"] == 1
    assert result["config"]["terminal_local_indices"] == [1]
    assert result["conclusions"]["single_terminal_local"] is True
    assert result["conclusions"]["selected_target_local_available"] is True
    assert result["conclusions"]["all_terminal_candidates_match_full_rollout_centering_contract"] is True
    assert result["conclusions"]["all_terminal_candidates_worsen_under_objective_scope"] is True
    assert result["conclusions"]["prefix_mean_exceeds_objective_mean"] is True
    assert result["conclusions"]["objective_scope_std_collapse_is_explained_by_truncating_away_high_variance_prefix"] is True
    assert result["conclusions"]["objective_window_is_low_variance_slice_not_high_variance_source"] is True


def test_prefix_dominance_conclusion_allows_full_std_to_exceed_prefix_std_via_between_group_gap():
    prefix_returns = [[10.0], [50.0]] * 973
    objective_returns = [[9.0], [8.0], [7.0], [6.0]] * 25 + [[6.0], [6.0]]
    full_returns = prefix_returns + objective_returns
    flat_returns = [float(v[0]) for v in full_returns]
    full_mean = probe_mod._mean(flat_returns)
    full_std = probe_mod._std(flat_returns)

    distance_row = {
        "env_index": 0,
        "env_seed": 1,
        "rollout_seed": 2,
        "pre_vs_zero_suffix_gap": 3.0,
        "objective_terminal_reset_count": 1,
    }
    suite_payload = {
        "rollout_discounted_raw_returns": full_returns
    }
    suite_summary = {"suite_runtime_wall_s": 1.25}
    token_rows = []
    for idx, entry in enumerate(objective_returns):
        reward_raw = 0.2 if idx % 2 == 0 else -0.1
        is_terminal = idx == 68
        token_rows.append(
            {
                "global_step": 1946 + idx,
                "objective_local_index": idx,
                "objective_episode_index": 0 if idx <= 68 else 1,
                "reward_raw": reward_raw,
                "reward_norm": float(
                    (reward_raw / full_std) - (full_mean / full_std) if is_terminal else (reward_raw / full_std)
                ),
                "rollout_return_raw": float(entry[0]),
                "value_norm": 0.0,
                "delta_norm": 0.0,
                "next_non_terminal": 0.0 if is_terminal else 1.0,
            }
        )

    result = probe_mod._build_probe(
        suite_name="pair1",
        env_index=0,
        distance_row=distance_row,
        suite_payload=suite_payload,
        suite_summary=suite_summary,
        token_rows=token_rows,
        explicit_target_local_index=None,
    )

    readout = result["scale_collapse"]["scale_collapse_readout"]
    assert readout["prefix_std_ratio_vs_full"] < 1.0
    assert result["scale_collapse"]["full_variance_decomposition"]["within_prefix_share"] > 0.9
    assert result["conclusions"]["objective_scope_std_collapse_is_explained_by_truncating_away_high_variance_prefix"] is True


def test_terminal_local_scale_chain_probe_reuses_existing_output(tmp_path, monkeypatch, capsys):
    output_path = tmp_path / "terminal_local_scale_chain_probe.json"
    payload = {"ok": True, "kind": "terminal_local_scale_chain_probe"}
    output_path.write_text(json.dumps(payload))

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "phase3_terminal_local_scale_chain_probe.py",
            "dummy.ckpt",
            "--suite-name",
            "pair1",
            "--env-index",
            "12",
            "--output-json",
            str(output_path),
        ],
    )

    assert probe_mod.main() == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == payload


def test_dominant_objective_variance_source_prefers_largest_component():
    assert (
        probe_mod._dominant_objective_variance_source(
            between_episode_means=1.0,
            within_episode_contributions=[
                {"episode_index": 0, "contribution": 5.0},
                {"episode_index": 1, "contribution": 0.5},
            ],
        )
        == "within_episode0"
    )
    assert (
        probe_mod._dominant_objective_variance_source(
            between_episode_means=6.0,
            within_episode_contributions=[
                {"episode_index": 0, "contribution": 5.0},
                {"episode_index": 1, "contribution": 0.5},
            ],
        )
        == "between_episode_means"
    )
