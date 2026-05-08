import pytest

from scripts.exploratory.phase2_gated_reward_path_generator_wrapper import (
    RewardPathBalanceProfile,
    TopologyThresholds,
    apply_decision_to_row,
    decide_gated_reward_path_repair,
    row_with_balanced_topology,
    select_h_for_decision,
)
from scripts.exploratory.phase2_reward_group_balance_runtime import (
    BALANCE_ACTION_SCALE_KEY,
    BALANCE_ENABLED_KEY,
    BALANCE_NOISE_SCALE_KEY,
    BALANCE_PROFILE_KEY,
    BALANCE_STATE_SCALE_KEY,
)
from ticl.analysis.phase2_gated_reward_path_milestone import (
    GATED_REWARD_PATH_MILESTONE,
    GATED_REWARD_PATH_TERMINAL_COVERAGE_MILESTONE,
    SAMPLED_TOPOLOGY_MILESTONE,
    _terminal_coverage_bucket_for_slot,
    _terminal_coverage_target_for_bucket,
    gated_reward_path_milestone_contract,
    normalize_prior_milestone,
)


def _row(
    *,
    state: float,
    action: float,
    noise: float,
) -> dict[str, float]:
    return {
        "reward_state_input_gain_fraction": float(state),
        "reward_action_input_gain_fraction": float(action),
        "reward_noise_input_gain_fraction": float(noise),
        "reward_state_to_action_gain_ratio": float(state) / max(float(action), 1e-12),
    }


def test_normalize_prior_milestone_accepts_terminal_coverage_aliases():
    assert normalize_prior_milestone("sampled") == SAMPLED_TOPOLOGY_MILESTONE
    assert normalize_prior_milestone("gated") == GATED_REWARD_PATH_MILESTONE
    assert (
        normalize_prior_milestone("gated_reward_path_terminal_coverage")
        == GATED_REWARD_PATH_TERMINAL_COVERAGE_MILESTONE
    )
    assert (
        normalize_prior_milestone("gated_reward_path_balance_terminal")
        == GATED_REWARD_PATH_TERMINAL_COVERAGE_MILESTONE
    )
    with pytest.raises(ValueError):
        normalize_prior_milestone("unknown")


def test_formal_gated_reward_path_contract_excludes_terminal_and_dimension_changes():
    contract = gated_reward_path_milestone_contract()

    assert contract["milestone"] == GATED_REWARD_PATH_MILESTONE
    assert contract["modified_path"] == "reward_path_only"
    assert contract["terminal_coverage_in_formal_milestone"] is False
    assert "state_transition_path_unchanged" in contract["non_degradation_guards"]
    assert "terminal_survival_path_unchanged" in contract["non_degradation_guards"]
    assert "dimension_samplers_unchanged" in contract["non_degradation_guards"]


def test_gated_reward_path_keeps_original_when_already_healthy():
    profile = RewardPathBalanceProfile()
    thresholds = TopologyThresholds()
    row = row_with_balanced_topology(
        _row(state=0.80, action=0.10, noise=0.10),
        profile=profile,
    )

    decision = decide_gated_reward_path_repair(row, thresholds=thresholds)
    audited = apply_decision_to_row(row, decision)
    selected_h = select_h_for_decision(
        {"state_dim": 12, "action_dim": 4, "terminal_reset_count_target": 7.0},
        decision=decision,
        profile=profile,
    )

    assert audited["gated_topology_pass"] is True
    assert audited["gated_repair_kind"] == "original_healthy_keep"
    assert audited["selected_balance_applied"] is False
    assert BALANCE_ENABLED_KEY not in selected_h
    assert selected_h["state_dim"] == 12
    assert selected_h["action_dim"] == 4
    assert selected_h["terminal_reset_count_target"] == pytest.approx(7.0)


def test_gated_reward_path_balances_width_bias_and_preserves_non_reward_fields():
    profile = RewardPathBalanceProfile()
    thresholds = TopologyThresholds()
    row = row_with_balanced_topology(
        _row(state=0.70, action=0.02, noise=0.28),
        profile=profile,
    )

    decision = decide_gated_reward_path_repair(row, thresholds=thresholds)
    audited = apply_decision_to_row(row, decision)
    selected_h = select_h_for_decision(
        {
            "state_dim": 16,
            "action_dim": 4,
            "env_obs_input_dim": 24,
            "terminal_reset_count_target": 11.0,
        },
        decision=decision,
        profile=profile,
    )

    assert audited["original_topology_pass"] is False
    assert audited["balanced_topology_pass"] is True
    assert audited["gated_topology_pass"] is True
    assert audited["gated_repair_kind"] == "width_bias_balance"
    assert audited["selected_balance_applied"] is True
    assert selected_h[BALANCE_ENABLED_KEY] is True
    assert selected_h[BALANCE_STATE_SCALE_KEY] == pytest.approx(profile.state_scale)
    assert selected_h[BALANCE_ACTION_SCALE_KEY] == pytest.approx(profile.action_scale)
    assert selected_h[BALANCE_NOISE_SCALE_KEY] == pytest.approx(profile.noise_scale)
    assert selected_h[BALANCE_PROFILE_KEY] == profile.name
    assert selected_h["state_dim"] == 16
    assert selected_h["action_dim"] == 4
    assert selected_h["env_obs_input_dim"] == 24
    assert selected_h["terminal_reset_count_target"] == pytest.approx(11.0)


def test_gated_reward_path_rejects_when_balancing_still_fails():
    profile = RewardPathBalanceProfile()
    thresholds = TopologyThresholds()
    row = row_with_balanced_topology(
        _row(state=0.99, action=0.001, noise=0.009),
        profile=profile,
    )

    decision = decide_gated_reward_path_repair(row, thresholds=thresholds)
    audited = apply_decision_to_row(row, decision)

    assert audited["original_topology_pass"] is False
    assert audited["balanced_topology_pass"] is False
    assert audited["gated_topology_pass"] is False
    assert audited["gated_repair_kind"] == "reject_still_fail"
    assert audited["selected_repair_kind"] is None
    with pytest.raises(ValueError):
        select_h_for_decision({}, decision=decision, profile=profile)


def test_terminal_coverage_quota_keeps_most_slots_original():
    buckets = [_terminal_coverage_bucket_for_slot(i, 64) for i in range(64)]

    assert buckets.count("high_50_80") == 8
    assert buckets.count("mid_20_50") == 8
    assert buckets.count("original") == 48


def test_terminal_coverage_target_ranges_are_deterministic():
    high_a = _terminal_coverage_target_for_bucket(
        original_target=7.0,
        bucket="high_50_80",
        seed=123,
        slot_idx=5,
    )
    high_b = _terminal_coverage_target_for_bucket(
        original_target=7.0,
        bucket="high_50_80",
        seed=123,
        slot_idx=5,
    )
    mid = _terminal_coverage_target_for_bucket(
        original_target=7.0,
        bucket="mid_20_50",
        seed=123,
        slot_idx=5,
    )
    original = _terminal_coverage_target_for_bucket(
        original_target=7.0,
        bucket="original",
        seed=123,
        slot_idx=5,
    )

    assert high_a == pytest.approx(high_b)
    assert 50.0 <= high_a <= 80.0
    assert 20.0 <= mid <= 50.0
    assert original == pytest.approx(7.0)
