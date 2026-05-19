import torch

from scripts.exploratory.phase2_pendulum_controllable_basin_guard import evaluate_h_list
from ticl.priors.environment_prior import EnvironmentPrior
from ticl.tests.test_phase2_pendulum_settle_prior_internal_repair import _base_env_cfg


def _h_list(*, action_controls_position: bool, n: int = 2):
    prior = EnvironmentPrior(
        _base_env_cfg(
            pendulum_settle_yield_repair_enabled=True,
            pendulum_settle_yield_action_controls_position=action_controls_position,
            pendulum_settle_yield_action_gain=1.0,
            pendulum_settle_yield_ctrl_weight=0.0,
            batch_parallel_backend="torch_vectorized",
        )
    )
    h_list = prior._sample_batch_hypers(n)
    for h in h_list:
        h["pendulum_settle_yield_repair_enabled"] = True
        h["pendulum_settle_yield_repair_prob"] = 1.0
        h["pendulum_settle_yield_action_controls_position"] = action_controls_position
        h["pendulum_settle_yield_action_gain"] = 1.0
        h["pendulum_settle_yield_ctrl_weight"] = 0.0
    return h_list


def test_controllable_basin_guard_marks_prior_internal_position_control_positive():
    item = evaluate_h_list(
        _h_list(action_controls_position=True, n=2),
        label="unit",
        seed=11,
        device=torch.device("cpu"),
        n_steps=48,
        split_frac=0.5,
        gains=[(1.0, 0.0), (2.0, 1.0)],
        init_states=[(1.0, 0.5), (-1.0, -0.5)],
        reward_margin=0.10,
        suffix_margin=0.05,
        action_decay_margin=0.005,
        settle_ratio=0.85,
    )

    assert item["coverage"]["axis_support_count"] == 2
    assert item["coverage"]["sustained_settle_count"] == 2
    assert item["coverage"]["axis_supported_sustained_settle_count"] == 2
    assert all(row["pendulum_axis_support"] for row in item["per_env"])
    assert all(row["pendulum_sustained_settle"] for row in item["per_env"])
    assert all(row["pendulum_axis_supported_sustained_settle"] for row in item["per_env"])


def test_controllable_basin_guard_keeps_axis_support_separate_from_short_gap():
    item = evaluate_h_list(
        _h_list(action_controls_position=False, n=2),
        label="unit",
        seed=12,
        device=torch.device("cpu"),
        n_steps=32,
        split_frac=0.5,
        gains=[(1.0, 0.0), (2.0, 1.0)],
        init_states=[(1.0, 0.5)],
        reward_margin=0.10,
        suffix_margin=0.05,
        action_decay_margin=0.005,
        settle_ratio=0.85,
    )

    assert item["coverage"]["axis_support_count"] == 0
    assert item["coverage"]["axis_supported_sustained_settle_count"] == 0
    assert all(not row["pendulum_axis_support"] for row in item["per_env"])
    assert all(not row["pendulum_axis_supported_sustained_settle"] for row in item["per_env"])
    assert all("pendulum_short_gap" in row for row in item["per_env"])
