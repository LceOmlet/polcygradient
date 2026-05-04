from __future__ import annotations

from copy import deepcopy
from typing import Any


RLPFN_ANCHOR_COMPARE_OPTIMIZER_OVERLAY = {
    "rl_objective": "ppo",
    "ppo_n_epochs": 4,
    "ppo_actor_gae_space": "normalized",
    "ppo_actor_baseline_mode": "learned",
    "ppo_separate_value_backbone": False,
    "ppo_reset_env_state_at_sep": True,
    "ppo_restore_validation_policy_state": True,
    "ppo_strict_fixed_env_mode": True,
    "ppo_deterministic_actor_sampling": True,
    "ppo_deterministic_batch_plan": True,
    "ppo_strict_native_rollout": True,
    "ppo_vf_coef": 0.5,
}


def apply_rlpfn_anchor_compare_contract(
    config: dict[str, Any],
    *,
    learning_rate: float | None = None,
    env_rng_seeds: Any = None,
    rollout_rng_seeds: Any = None,
) -> dict[str, Any]:
    optimizer_cfg = config.setdefault("optimizer", {})
    optimizer_cfg.update(deepcopy(RLPFN_ANCHOR_COMPARE_OPTIMIZER_OVERLAY))
    if learning_rate is not None:
        optimizer_cfg["learning_rate"] = float(learning_rate)
    if env_rng_seeds is not None:
        optimizer_cfg["ppo_env_rng_seeds"] = deepcopy(env_rng_seeds)
    if rollout_rng_seeds is not None:
        optimizer_cfg["ppo_rollout_rng_seeds"] = deepcopy(rollout_rng_seeds)
    return config
