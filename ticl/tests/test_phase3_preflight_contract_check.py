from ticl.analysis.phase3_preflight_contract_check import (
    ALLOWED_PHASE3_BASELINE_DIFF_KEYS,
    _contract_diff_report,
)


def test_contract_diff_report_allows_only_many_env_seed_diffs() -> None:
    phase2 = {
        "n_envs": 1,
        "n_steps": 256,
        "batch_size": 256,
        "actor_objective_mode": "tokenwise",
        "strict_native_rollout": True,
        "env_rng_seeds": [2020],
        "rollout_rng_seeds": [4040],
    }
    phase3 = {
        **phase2,
        "n_envs": 16,
        "env_rng_seeds": list(range(16)),
        "rollout_rng_seeds": list(range(100, 116)),
    }
    result = _contract_diff_report(
        phase2_signature=phase2,
        phase3_signature=phase3,
        allowed_diff_keys=ALLOWED_PHASE3_BASELINE_DIFF_KEYS,
    )
    assert result["unexpected_diff_free"] is True
    assert sorted(result["allowed_diff"].keys()) == [
        "env_rng_seeds",
        "n_envs",
        "rollout_rng_seeds",
    ]


def test_contract_diff_report_rejects_objective_drift_in_baseline() -> None:
    phase2 = {
        "n_envs": 1,
        "actor_objective_mode": "tokenwise",
        "strict_native_rollout": True,
    }
    phase3 = {
        "n_envs": 16,
        "actor_objective_mode": "tokenwise_scale_env12_mid_episode_extension_block",
        "strict_native_rollout": True,
    }
    result = _contract_diff_report(
        phase2_signature=phase2,
        phase3_signature=phase3,
        allowed_diff_keys=ALLOWED_PHASE3_BASELINE_DIFF_KEYS,
    )
    assert result["unexpected_diff_free"] is False
    assert "actor_objective_mode" in result["unexpected_diff"]
