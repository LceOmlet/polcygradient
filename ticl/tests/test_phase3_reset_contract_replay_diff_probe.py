from ticl.analysis import phase3_reset_contract_replay_diff_probe as probe_mod
from ticl.analysis import phase3_residual_anchor_gae_path_probe as anchor_probe_mod


def test_collect_contract_plan_for_mode_exposes_legacy_vs_official_seed_wiring():
    suite = {
        "env_seeds": [11, 22],
        "rollout_seeds": [33, 44],
    }
    legacy = anchor_probe_mod._collect_contract_plan_for_mode(
        suite,
        collect_contract_mode="legacy_distance_probe",
    )
    official = anchor_probe_mod._collect_contract_plan_for_mode(
        suite,
        collect_contract_mode="official_strict",
    )
    assert legacy["mode"] == "legacy_distance_probe"
    assert legacy["build_kwargs"] == {}
    assert official["mode"] == "official_strict"
    assert official["build_kwargs"] == {
        "strict_fixed_env_mode": True,
        "env_rng_seeds": [11, 22],
        "rollout_rng_seeds": [33, 44],
        "deterministic_batch_plan": True,
    }


def test_resolve_likely_single_source_attributes_diff_to_legacy_unseeded_collect():
    legacy_a = {
        "matches_stored_reset_count": True,
        "collect_contract_debug": {
            "algo_runtime_flags": {
                "strict_fixed_env_mode": False,
                "env_rng_seed_spec": None,
                "rollout_rng_seed_spec": None,
                "last_collect_rollout_env_rng_seeds": None,
                "last_collect_rollout_rollout_rng_seeds": None,
            }
        },
        "fresh_rollout_summary": {
            "compressed_segments": [[0, 90], [90, 102]],
            "episode_starts_on_objective_positions": [90],
            "objective_terminal_reset_count_from_segments": 1,
            "terminal_rows_within_objective_global_steps": [2036],
            "terminal_row_count_within_objective": 1,
        },
    }
    legacy_b = {
        "matches_stored_reset_count": False,
        "collect_contract_debug": legacy_a["collect_contract_debug"],
        "fresh_rollout_summary": {
            "compressed_segments": [[0, 102]],
            "episode_starts_on_objective_positions": [],
            "objective_terminal_reset_count_from_segments": 0,
            "terminal_rows_within_objective_global_steps": [],
            "terminal_row_count_within_objective": 0,
        },
    }
    official = {
        "matches_stored_reset_count": False,
        "collect_contract_debug": {
            "algo_runtime_flags": {
                "strict_fixed_env_mode": True,
                "env_rng_seed_spec": [11, 22],
                "rollout_rng_seed_spec": [33, 44],
                "last_collect_rollout_env_rng_seeds": [11, 22],
                "last_collect_rollout_rollout_rng_seeds": [33, 44],
            }
        },
        "fresh_rollout_summary": legacy_b["fresh_rollout_summary"],
    }
    result = probe_mod._resolve_likely_single_source(
        legacy_a=legacy_a,
        legacy_b=legacy_b,
        official=official,
    )
    assert "legacy unseeded collect contract" in result
    assert "current official collect explicitly wires suite env/rollout seeds" in result


def test_legacy_repeats_disagree_detects_layout_change():
    legacy_a = {
        "fresh_rollout_summary": {
            "compressed_segments": [[0, 90], [90, 102]],
            "episode_starts_on_objective_positions": [90],
            "objective_terminal_reset_count_from_segments": 1,
            "terminal_rows_within_objective_global_steps": [2036],
            "terminal_row_count_within_objective": 1,
        }
    }
    legacy_b = {
        "fresh_rollout_summary": {
            "compressed_segments": [[0, 102]],
            "episode_starts_on_objective_positions": [],
            "objective_terminal_reset_count_from_segments": 0,
            "terminal_rows_within_objective_global_steps": [],
            "terminal_row_count_within_objective": 0,
        }
    }
    assert probe_mod._legacy_repeats_disagree(legacy_a, legacy_b) is True
