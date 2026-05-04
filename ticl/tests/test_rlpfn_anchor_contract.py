from argparse import Namespace

import pytest

from ticl.fit_model import _apply_rlpfn_anchor_compare_contract_if_requested
from ticl.model_configs import get_model_default_config
from ticl.rlpfn_anchor_contract import apply_rlpfn_anchor_compare_contract


def test_apply_rlpfn_anchor_compare_contract_sets_only_compare_critical_bridge_fields():
    config = get_model_default_config("rlpfn")
    baseline_batch_size = int(config["dataloader"]["batch_size"])
    baseline_n_samples = int(config["prior"]["n_samples"])
    baseline_lr = float(config["optimizer"]["learning_rate"])

    apply_rlpfn_anchor_compare_contract(
        config,
        env_rng_seeds=[2020],
        rollout_rng_seeds=[4040],
    )

    optimizer = config["optimizer"]
    assert optimizer["rl_objective"] == "ppo"
    assert optimizer["ppo_restore_validation_policy_state"] is True
    assert optimizer["ppo_strict_fixed_env_mode"] is True
    assert optimizer["ppo_deterministic_actor_sampling"] is True
    assert optimizer["ppo_deterministic_batch_plan"] is True
    assert optimizer["ppo_strict_native_rollout"] is True
    assert optimizer["ppo_env_rng_seeds"] == [2020]
    assert optimizer["ppo_rollout_rng_seeds"] == [4040]
    assert float(optimizer["learning_rate"]) == baseline_lr
    assert int(config["dataloader"]["batch_size"]) == baseline_batch_size
    assert int(config["prior"]["n_samples"]) == baseline_n_samples


def test_apply_rlpfn_anchor_compare_contract_can_pin_learning_rate_without_touching_scale():
    config = get_model_default_config("rlpfn")
    apply_rlpfn_anchor_compare_contract(config, learning_rate=2e-4)

    assert float(config["optimizer"]["learning_rate"]) == pytest.approx(2e-4)
    assert int(config["dataloader"]["batch_size"]) == 2048
    assert int(config["prior"]["n_samples"]) == 2048


def test_fit_model_anchor_compare_contract_requires_warm_start_checkpoint():
    config = get_model_default_config("rlpfn")
    args = Namespace(
        model_type="rlpfn",
        orchestration=Namespace(
            rlpfn_anchor_compare_contract=True,
            warm_start_from=None,
        ),
    )

    with pytest.raises(ValueError, match="requires --warm-start-from"):
        _apply_rlpfn_anchor_compare_contract_if_requested(config, args)
