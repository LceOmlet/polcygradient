import numpy as np

from scripts.exploratory import phase2_pendulum_quadratic_settle_specimen_preflight as preflight


def test_lqr_gain_stabilizes_controllable_specimen():
    env = preflight.QuadraticSettleEnv(
        dt=0.08,
        rho=0.92,
        beta=0.2,
        q_weight=1.0,
        v_weight=0.3,
        ctrl_weight=0.03,
        action_limit=1.0,
        q0=1.0,
        v0=0.2,
        noise_std=0.0,
    )

    gain = preflight._lqr_gain(env)

    assert preflight._controllability_rank(env) == 2
    assert preflight._closed_loop_radius(env, gain) < 1.0
    assert gain.shape == (2,)


def test_lqr_witness_beats_zero_and_settles_for_easy_case():
    args = type(
        "Args",
        (),
        {
            "split_frac": 0.5,
            "closed_loop_radius_max": 0.995,
            "abs_reward_margin": 1.0,
            "rel_reward_margin": 0.01,
            "abs_suffix_margin": 0.5,
            "rel_suffix_margin": 0.01,
            "action_decay_margin": 0.01,
            "self_energy_rel_max": 0.8,
            "open_energy_rel_max": 0.8,
            "saturation_max": 0.5,
            "action_sensitive_delta_min": 0.01,
        },
    )()
    env = preflight.QuadraticSettleEnv(
        dt=0.08,
        rho=0.9,
        beta=0.25,
        q_weight=1.4,
        v_weight=0.5,
        ctrl_weight=0.04,
        action_limit=1.1,
        q0=1.0,
        v0=0.4,
        noise_std=0.0,
    )

    row = preflight._one_env_report(env, n_steps=160, seed=123, args=args)

    assert row["conditions"]["analytic_pass"] is True
    assert row["conditions"]["witness_reward_pass"] is True
    assert row["conditions"]["settle_pass"] is True
    assert row["conditions"]["guard_pass"] is True
    assert row["score"]["lqr_minus_best_baseline_return"] > 0
    assert row["score"]["suffix_energy_over_prefix"] < 1.0


def test_aggregate_requires_count_and_diversity():
    args = type("Args", (), {"min_guard_count": 2, "min_guard_ratio": 0.5})()
    template = {
        "conditions": {
            "analytic_pass": True,
            "witness_reward_pass": True,
            "action_decay_pass": True,
            "settle_pass": True,
            "saturation_pass": True,
            "action_sensitive_pass": True,
            "guard_pass": True,
        },
        "score": {
            "lqr_minus_best_baseline_return": 1.0,
            "lqr_minus_best_baseline_suffix_return": 1.0,
            "suffix_energy_over_prefix": 0.5,
            "suffix_energy_over_best_baseline_suffix": 0.5,
        },
        "closed_loop_radius": 0.8,
        "lqr_action_abs": {"suffix": 0.1},
    }
    rows = []
    for idx, rho in enumerate([0.87, 0.91, 0.94, 0.97]):
        row = dict(template)
        row["env"] = {
            "dt": 0.08,
            "rho": rho,
            "beta": [0.12, 0.2, 0.27, 0.33][idx],
            "q_weight": 1.0,
            "v_weight": 0.4,
            "ctrl_weight": [0.015, 0.035, 0.055, 0.075][idx],
            "q0": [-1.0, 1.0, -0.5, 0.5][idx],
            "v0": [-0.3, 0.3, 0.4, -0.4][idx],
        }
        rows.append(row)

    report = preflight._aggregate(rows, args=args)

    assert report["count_safe"] is True
    assert report["diversity_pass"] is True
    assert report["condition_counts"]["guard_pass"] == 4
