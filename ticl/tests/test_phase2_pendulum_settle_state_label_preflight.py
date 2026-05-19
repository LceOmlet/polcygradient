import numpy as np

from scripts.exploratory import phase2_pendulum_settle_state_label_preflight as preflight


def _payload(full, prefix, suffix, action_prefix, action_suffix, drift_prefix, drift_suffix):
    arr = lambda x: np.asarray(x, dtype=np.float32)
    return {
        "reward_env": {
            "full": arr(full),
            "prefix": arr(prefix),
            "suffix": arr(suffix),
        },
        "action_abs": {
            "full": arr(action_prefix),
            "prefix": arr(action_prefix),
            "suffix": arr(action_suffix),
        },
        "state_drift": {
            "full": arr(drift_prefix),
            "prefix": arr(drift_prefix),
            "suffix": arr(drift_suffix),
        },
    }


def test_state_drift_uses_masked_next_state_differences():
    state = np.asarray(
        [
            [[0.0, 0.0], [1.0, 10.0]],
            [[3.0, 4.0], [1.0, 20.0]],
        ],
        dtype=np.float32,
    )
    mask = np.asarray(
        [
            [[1.0, 1.0], [1.0, 0.0]],
            [[1.0, 1.0], [1.0, 0.0]],
        ],
        dtype=np.float32,
    )

    drift = preflight._state_drift(state, mask)

    assert drift.shape == (2, 2)
    assert np.allclose(drift[0], drift[1])
    assert np.isclose(drift[1, 0], np.sqrt((3.0**2 + 4.0**2) / 2.0))
    assert np.isclose(drift[1, 1], 0.0)


def test_classify_marks_only_reward_action_and_state_settle_cases():
    args = type(
        "Args",
        (),
        {
            "abs_margin": 1.0,
            "rel_margin": 0.1,
            "action_decay_margin": 0.2,
            "self_drift_rel_max": 0.85,
            "open_drift_rel_max": 0.9,
            "drift_eps": 1e-6,
        },
    )()
    payloads = {
        "zero": _payload([0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [1, 1], [1, 1]),
        "random": _payload([1, 1], [0.5, 0.5], [0.5, 0.5], [1, 1], [1, 1], [2, 2], [2, 2]),
        "neg_random": _payload([0, 0], [0, 0], [0, 0], [1, 1], [1, 1], [2, 2], [2, 2]),
        "half_random": _payload([0.5, 0.5], [0.25, 0.25], [0.25, 0.25], [0.5, 0.5], [0.5, 0.5], [2, 2], [2, 2]),
        "decay_obs_linear_0x100": _payload([5, 5], [2, 4.4], [3, 0.6], [1, 1], [0.1, 0.1], [2, 2], [1, 2.5]),
        "decay_obs_linear_neg_0x100": _payload([2, 2], [1, 1], [1, 1], [1, 1], [0.1, 0.1], [2, 2], [2, 2]),
    }

    report = preflight._classify(payloads, args=args)

    assert report["coverage"]["cheap_settle_guard_count"] == 1
    assert report["coverage"]["medium_settle_guard_count"] == 1
    assert report["per_env"][0]["cheap_settle_guard"] is True
    assert report["per_env"][0]["medium_settle_guard"] is True
    assert report["per_env"][1]["cheap_settle_guard"] is False
    assert report["per_env"][1]["suffix_reward_condition"] is False


def test_build_decision_requires_all_lists_count_safe():
    lists = [
        {"classification": {"coverage": {"cheap_settle_guard_count": 8, "cheap_settle_guard_ratio": 0.125}}},
        {"classification": {"coverage": {"cheap_settle_guard_count": 7, "cheap_settle_guard_ratio": 0.109}}},
    ]

    report = preflight.build_decision(lists, min_count=8, min_ratio=0.1)

    assert report["cheap_suffix_state_settle_guard_ready"] is False
    assert report["min_count"] == 7


def test_apply_h_overrides_only_touches_requested_transition_knobs():
    args = type(
        "Args",
        (),
        {
            "override_alpha": 0.1,
            "override_state_highway_lambda": 0.4,
            "override_state_output_scale": None,
            "override_reference_state_inertia_enabled": True,
        },
    )()
    h_list = [{"alpha": 0.3, "state_highway_enabled": False, "state_highway_lambda": 0.0, "reward_scale": 2.0}]

    out = preflight._apply_h_overrides(h_list, args)

    assert out[0]["alpha"] == 0.1
    assert out[0]["state_highway_enabled"] is True
    assert out[0]["state_highway_lambda"] == 0.4
    assert out[0]["reference_state_inertia_enabled"] is True
    assert out[0]["reward_scale"] == 2.0
    assert h_list[0]["alpha"] == 0.3


def test_apply_prior_config_overrides_sets_vectorized_runtime_source():
    args = type(
        "Args",
        (),
        {
            "override_alpha": 0.2,
            "override_state_highway_lambda": 0.6,
            "override_state_output_scale": 0.9,
            "override_reference_state_inertia_enabled": True,
        },
    )()
    prior = type("Prior", (), {"config": {}})()

    preflight._apply_prior_config_overrides(prior, args)

    assert prior.config["alpha"] == 0.2
    assert prior.config["state_highway_enabled"] is True
    assert prior.config["state_highway_lambda"] == 0.6
    assert prior.config["state_output_scale"] == 0.9
    assert prior.config["reference_state_inertia_enabled"] is True
