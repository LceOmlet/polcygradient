import torch
import pytest

from ticl.train import (
    _parse_ppo_seed_spec,
    _looks_like_maintained_rlpfn_env_prior,
    _resolve_environment_prior,
    _resolve_recurrent_ppo_training_bridge_kwargs,
)
from ticl.priors.environment_prior import EnvironmentPrior


class _Leaf:
    pass


def _dummy_model_with_saved_validation_state():
    model = _Leaf()
    model.__dict__["_validation_ppo_policy_state"] = {
        "action_net.weight": torch.ones((2, 2), dtype=torch.float32),
    }
    return model


def test_resolve_environment_prior_unwraps_classification_adapter_chain():
    env_prior = EnvironmentPrior({})
    wrapper = _Leaf()
    wrapper.base_prior = env_prior

    assert _resolve_environment_prior(wrapper) is env_prior


def test_legacy_environment_policy_gradient_loss_is_disabled():
    env_prior = EnvironmentPrior({})

    with pytest.raises(RuntimeError, match="rollout_policy_gradient_loss is disabled"):
        env_prior.rollout_policy_gradient_loss()


def test_maintained_rlpfn_env_prior_detection_requires_trusted_contract_markers():
    env_prior = EnvironmentPrior(
        {
            "strict_joint_transition_enabled": True,
            "reinforce_sequence_replay_enabled": True,
        }
    )
    plain_prior = EnvironmentPrior({"strict_joint_transition_enabled": True})

    assert _looks_like_maintained_rlpfn_env_prior(env_prior) is True
    assert _looks_like_maintained_rlpfn_env_prior(plain_prior) is False


def test_parse_ppo_seed_spec_accepts_comma_and_json_list():
    assert _parse_ppo_seed_spec("101,202", name="env") == [101, 202]
    assert _parse_ppo_seed_spec("[303, 404]", name="rollout") == [303, 404]
    assert _parse_ppo_seed_spec("505", name="single") == 505
    assert _parse_ppo_seed_spec(None, name="none") is None


def test_resolve_recurrent_ppo_training_bridge_kwargs_auto_restores_saved_validation_state():
    kwargs, summary = _resolve_recurrent_ppo_training_bridge_kwargs(
        model=_dummy_model_with_saved_validation_state(),
        env_prior=object(),
        device="cpu",
        num_features=16,
        n_envs=1,
        n_steps=8,
        learning_rate=1e-4,
        batch_size=8,
        n_epochs=1,
        gamma=1.0,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        normalize_advantage=True,
        actor_gae_space="normalized",
        actor_baseline_mode="learned",
        separate_value_backbone=False,
        reset_env_state_at_sep=True,
        runtime_normalized_q_value_weight_override=None,
        runtime_next_state_flow_matching_weight_override=None,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=None,
        verbose=0,
        ppo_restore_validation_policy_state=None,
        ppo_strict_fixed_env_mode=True,
        ppo_env_rng_seeds="101,202",
        ppo_rollout_rng_seeds="[303, 404]",
        ppo_deterministic_actor_sampling=True,
        ppo_deterministic_batch_plan=True,
        ppo_strict_native_rollout=True,
    )

    assert kwargs["restore_validation_policy_state"] is True
    assert kwargs["env_rng_seeds"] == [101, 202]
    assert kwargs["rollout_rng_seeds"] == [303, 404]
    assert kwargs["strict_native_rollout"] is True
    assert summary["checkpoint_saved_validation_policy_state_present"] is True
    assert summary["restore_validation_policy_state"] is True


def test_resolve_recurrent_ppo_training_bridge_kwargs_respects_explicit_restore_opt_out():
    kwargs, summary = _resolve_recurrent_ppo_training_bridge_kwargs(
        model=_dummy_model_with_saved_validation_state(),
        env_prior=object(),
        device="cpu",
        num_features=16,
        n_envs=1,
        n_steps=8,
        learning_rate=1e-4,
        batch_size=8,
        n_epochs=1,
        gamma=1.0,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        normalize_advantage=True,
        actor_gae_space="normalized",
        actor_baseline_mode="learned",
        separate_value_backbone=False,
        reset_env_state_at_sep=True,
        runtime_normalized_q_value_weight_override=None,
        runtime_next_state_flow_matching_weight_override=None,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=None,
        verbose=0,
        ppo_restore_validation_policy_state=False,
        ppo_strict_fixed_env_mode=False,
        ppo_env_rng_seeds=None,
        ppo_rollout_rng_seeds=None,
        ppo_deterministic_actor_sampling=False,
        ppo_deterministic_batch_plan=False,
        ppo_strict_native_rollout=False,
    )

    assert kwargs["restore_validation_policy_state"] is False
    assert summary["checkpoint_saved_validation_policy_state_present"] is True
    assert summary["restore_validation_policy_state"] is False


def test_resolve_recurrent_ppo_training_bridge_kwargs_passes_legacy_compare_knobs():
    kwargs, summary = _resolve_recurrent_ppo_training_bridge_kwargs(
        model=_Leaf(),
        env_prior=object(),
        device="cpu",
        num_features=432,
        n_envs=1,
        n_steps=256,
        learning_rate=2e-4,
        batch_size=256,
        n_epochs=1,
        gamma=0.98,
        gae_lambda=0.90,
        clip_range=0.2,
        clip_range_vf=None,
        normalize_advantage=False,
        space_contract="normalized",
        actor_baseline_mode="learned",
        separate_value_backbone=False,
        reset_env_state_at_sep=True,
        value_head_impl="vendor_official",
        value_path_adapter_impl="none",
        value_head_mlp_hidden_dim=256,
        runtime_normalized_q_value_weight_override=None,
        runtime_next_state_flow_matching_weight_override=None,
        ent_coef=0.0,
        vf_coef=0.1,
        max_grad_norm=0.5,
        target_kl=0.03,
        verbose=0,
        ppo_restore_validation_policy_state=False,
        ppo_strict_fixed_env_mode=True,
        ppo_env_rng_seeds="26024",
        ppo_rollout_rng_seeds="4040",
        ppo_deterministic_actor_sampling=True,
        ppo_deterministic_batch_plan=True,
        ppo_strict_native_rollout=False,
    )

    assert kwargs["batch_size"] == 256
    assert kwargs["normalize_advantage"] is False
    assert kwargs["space_contract"] == "normalized"
    assert kwargs["value_head_impl"] == "vendor_official"
    assert kwargs["value_path_adapter_impl"] == "none"
    assert kwargs["value_head_mlp_hidden_dim"] == 256
    assert kwargs["vf_coef"] == 0.1
    assert kwargs["target_kl"] == 0.03
    assert kwargs["env_rng_seeds"] == 26024
    assert kwargs["rollout_rng_seeds"] == 4040
    assert summary["value_head_impl"] == "vendor_official"
    assert summary["strict_fixed_env_mode"] is True
