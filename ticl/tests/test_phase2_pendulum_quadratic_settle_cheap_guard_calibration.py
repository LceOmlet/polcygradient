from scripts.exploratory import phase2_pendulum_quadratic_settle_cheap_guard_calibration as cheap
from scripts.exploratory import phase2_pendulum_quadratic_settle_specimen_preflight as specimen


def test_cheap_features_are_analytic_and_finite():
    env = specimen.QuadraticSettleEnv(
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

    features = cheap._cheap_features(env)

    assert features["rank"] == 2.0
    assert features["radius"] < 1.0
    assert features["first_step_delta_v"] > 0.0
    assert features["initial_energy"] > 0.0


def test_cheap_guard_respects_core_thresholds():
    features = {
        "rank": 2.0,
        "radius": 0.8,
        "initial_action_ratio": 0.5,
        "initial_action_abs": 0.2,
        "first_step_delta_v": 0.1,
        "initial_energy": 0.4,
    }
    cfg = cheap.CheapGuardConfig(
        radius_max=0.9,
        initial_action_ratio_max=1.0,
        initial_action_abs_min=0.05,
        first_step_delta_v_min=0.05,
        initial_energy_min=0.05,
    )

    assert cheap._cheap_guard(features, cfg) is True

    weak_features = dict(features)
    weak_features["first_step_delta_v"] = 0.01
    assert cheap._cheap_guard(weak_features, cfg) is False


def test_confusion_counts_match_expected_labels():
    cfg = cheap.CheapGuardConfig(
        radius_max=0.9,
        initial_action_ratio_max=1.0,
        initial_action_abs_min=0.05,
        first_step_delta_v_min=0.05,
        initial_energy_min=0.05,
    )
    good = {
        "rank": 2.0,
        "radius": 0.8,
        "initial_action_ratio": 0.5,
        "initial_action_abs": 0.2,
        "first_step_delta_v": 0.1,
        "initial_energy": 0.4,
    }
    bad = dict(good)
    bad["radius"] = 0.95
    rows = [
        {"cheap_features": good, "full_guard_pass": True},
        {"cheap_features": good, "full_guard_pass": False},
        {"cheap_features": bad, "full_guard_pass": True},
        {"cheap_features": bad, "full_guard_pass": False},
    ]

    out = cheap._confusion(rows, cfg)

    assert out["tp"] == 1
    assert out["fp"] == 1
    assert out["fn"] == 1
    assert out["tn"] == 1
