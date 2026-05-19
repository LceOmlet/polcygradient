from copy import deepcopy

import pytest

from ticl.analysis.phase2_m4_potential_progress_milestone import (
    M4_POTENTIAL_PROGRESS_MILESTONE,
    m4_potential_progress_contract,
    install_m4_potential_progress_live_milestone,
    normalize_prior_milestone,
)
from ticl.model_configs import get_model_default_config
from ticl.priors.environment_prior import EnvironmentPrior
from ticl.rlpfn_maintained_path import validate_rlpfn_maintained_path_config


RETIRED_REWARD_GROUP_BALANCE_KEY = "exploratory_reward_group_balance_enabled"


def test_normalize_prior_milestone_is_m4_only_for_maintained_runner():
    assert normalize_prior_milestone("m4") == M4_POTENTIAL_PROGRESS_MILESTONE
    assert normalize_prior_milestone("potential_progress") == M4_POTENTIAL_PROGRESS_MILESTONE
    with pytest.raises(ValueError):
        normalize_prior_milestone("gated_reward_path_balance")
    with pytest.raises(ValueError):
        normalize_prior_milestone("gated_reward_path_balance_terminal_coverage")


def test_m4_contract_excludes_retired_balanced_runner_paths():
    contract = m4_potential_progress_contract()

    assert contract["milestone"] == M4_POTENTIAL_PROGRESS_MILESTONE
    assert contract["runner_path"] == "family_vectorized_live"
    assert "reward_group_balance_runtime" in contract["retired_paths"]
    assert "gated_reward_path_balance" in contract["retired_paths"]
    assert "no_runtime_reward_group_balance_patch" in contract["non_degradation_guards"]


def test_m4_potential_progress_installer_authorizes_live_source_labels_only():
    cfg = deepcopy(get_model_default_config("rlpfn"))
    cfg["optimizer"]["ppo_pack_prior_milestone"] = M4_POTENTIAL_PROGRESS_MILESTONE
    validate_rlpfn_maintained_path_config(cfg)

    prior = EnvironmentPrior(deepcopy(cfg["prior"]["environment"]))
    install_m4_potential_progress_live_milestone(prior, max_attempts=256)

    h_list, _env, _seeds = prior._sample_batch_hypers_and_environment_family_coarse_batch_conditioned(
        16,
        device="cpu",
        rng_seeds=list(range(8001, 8017)),
        build_policy_generator=False,
        return_final_env=False,
    )

    targets = {str(h["potential_progress_reward_axis_target"]) for h in h_list}
    survival_targets = {str(h["survival_component_axis_target"]) for h in h_list}
    assert targets == {"progress_only", "potential_delta_progress"}
    assert survival_targets == {"survival_off", "survival_on"}
    assert sum(str(h["survival_component_axis_target"]) == "survival_on" for h in h_list) == 8
    assert sum(str(h["survival_component_axis_target"]) == "survival_off" for h in h_list) == 8
    assert all(bool(h["exact_scm_gym_lowtail_family_selected"]) for h in h_list)
    assert all(float(h["exact_scm_gym_lowtail_reward_mix"]) == pytest.approx(0.35) for h in h_list)
    assert all(float(h["base_reward_retention_fraction"]) == pytest.approx(0.65) for h in h_list)
    assert all(bool(h["survival_component_preserves_base_reward"]) for h in h_list)
    assert all(
        0.005 <= float(h["survival_reward_weight"]) <= 0.03
        for h in h_list
        if str(h["survival_component_axis_target"]) == "survival_on"
    )
    assert all(
        float(h["survival_reward_weight"]) == 0.0
        for h in h_list
        if str(h["survival_component_axis_target"]) == "survival_off"
    )
    assert all(str(h["m4_topology_guard_metric"]) == "reward_density_ratio" for h in h_list)
    assert all(str(h["constrained_dim_sampling_policy"]) == "state_first" for h in h_list)
    assert all(str(h["constrained_dim_noise_policy"]) == "gym_low" for h in h_list)
    assert all(bool(h["reference_scm_zero_pad_inactive_init_enabled"]) for h in h_list)
    assert not any(bool(h.get(RETIRED_REWARD_GROUP_BALANCE_KEY, False)) for h in h_list)


def test_m4_potential_progress_source_label_conditioning_uses_formula_precheck(monkeypatch):
    cfg = deepcopy(get_model_default_config("rlpfn"))
    cfg["optimizer"]["ppo_pack_prior_milestone"] = M4_POTENTIAL_PROGRESS_MILESTONE
    validate_rlpfn_maintained_path_config(cfg)

    prior = EnvironmentPrior(deepcopy(cfg["prior"]["environment"]))
    install_m4_potential_progress_live_milestone(prior, max_attempts=256)

    def _unexpected_env_build(*_args, **_kwargs):
        raise AssertionError("M4 source-label topology precheck should not build a candidate env")

    monkeypatch.setattr(prior, "_sample_environment_family_coarse_batch", _unexpected_env_build)
    h_list, env, seeds = prior._sample_batch_hypers_and_environment_family_coarse_batch_conditioned(
        16,
        device="cpu",
        rng_seeds=list(range(8101, 8117)),
        build_policy_generator=False,
        return_final_env=False,
    )

    assert env is None
    assert len(h_list) == 16
    assert len(seeds) == 16
    assert all(bool(h["exact_scm_gym_lowtail_family_selected"]) for h in h_list)
    assert {str(h["survival_component_axis_target"]) for h in h_list} == {"survival_off", "survival_on"}


def test_m4_potential_progress_maintained_path_rejects_fixed_frozen_list_training():
    cfg = deepcopy(get_model_default_config("rlpfn"))
    cfg["optimizer"]["ppo_pack_prior_milestone"] = M4_POTENTIAL_PROGRESS_MILESTONE
    cfg["optimizer"]["ppo_pack_prior_mode"] = "fixed_frozen_list"
    cfg["optimizer"]["ppo_pack_fixed_frozen_h_list_csv"] = "/tmp/m4_fixed.csv"
    cfg["optimizer"]["ppo_pack_fixed_env_group_across_updates"] = True

    with pytest.raises(ValueError, match="sampled_topology"):
        validate_rlpfn_maintained_path_config(cfg)
