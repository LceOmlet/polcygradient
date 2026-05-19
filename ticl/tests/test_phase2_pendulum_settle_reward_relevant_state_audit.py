import numpy as np

from scripts.exploratory import phase2_pendulum_settle_reward_relevant_state_audit as audit


def test_masked_corr_weights_focus_reward_correlated_state_dim():
    reward = np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
    state = np.asarray(
        [
            [[0.0, 1.0]],
            [[1.0, 1.0]],
            [[2.0, 1.0]],
            [[3.0, 1.0]],
        ],
        dtype=np.float32,
    )
    mask = np.ones_like(state, dtype=np.float32)

    weights = audit._masked_corr_weights(
        reward_stack=reward,
        state_stack=state,
        mask_stack=mask,
        eps=1e-6,
    )

    assert weights.shape == (1, 2)
    assert weights[0, 0] > 0.99
    assert weights[0, 1] < 0.01


def test_reward_relevant_state_drift_ignores_unweighted_large_motion():
    next_state = np.asarray(
        [
            [[0.0, 0.0]],
            [[1.0, 100.0]],
        ],
        dtype=np.float32,
    )
    mask = np.ones_like(next_state, dtype=np.float32)
    weights = np.asarray([[1.0, 0.0]], dtype=np.float32)

    drift = audit._reward_relevant_state_drift(next_state, mask, weights, eps=1e-6)

    assert drift.shape == (2, 1)
    assert np.allclose(drift[0], drift[1])
    assert np.isclose(drift[1, 0], 1.0)


def test_reward_relevant_classify_reuses_reward_action_settle_contract():
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

    def payload(full, suffix, action_prefix, action_suffix, drift_prefix, drift_suffix):
        arr = lambda x: np.asarray(x, dtype=np.float32)
        return {
            "reward_env": {"full": arr(full), "prefix": arr(full), "suffix": arr(suffix)},
            "action_abs": {"full": arr(action_prefix), "prefix": arr(action_prefix), "suffix": arr(action_suffix)},
            "state_drift": {"full": arr(drift_prefix), "prefix": arr(drift_prefix), "suffix": arr(drift_suffix)},
        }

    payloads = {
        "zero": payload([0], [0], [0], [0], [1], [1]),
        "random": payload([1], [0.5], [1], [1], [2], [2]),
        "neg_random": payload([0], [0], [1], [1], [2], [2]),
        "half_random": payload([0.5], [0.25], [0.5], [0.5], [2], [2]),
        "decay_obs_linear_0x100": payload([5], [3], [1], [0.1], [2], [1]),
        "decay_obs_linear_neg_0x100": payload([2], [1], [1], [0.1], [2], [2]),
    }

    report = audit._classify(payloads, args=args)

    assert report["coverage"]["reward_relevant_settle_guard_count"] == 1
    assert report["per_env"][0]["reward_relevant_settle_guard"] is True
    assert report["contract"]["offline_audit_only"] is True
