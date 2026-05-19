import numpy as np

from scripts.exploratory.phase2_action_sensitive_terminal_counterfactual import (
    _freeze_counterfactual_h_list,
    _official_buffer_outputs,
    _pearson,
    first_done_steps,
    future_survival_lengths,
)


def test_first_done_steps_and_future_survival_lengths_are_terminal_only():
    dones = np.asarray(
        [
            [False, False, True],
            [False, True, False],
            [True, False, False],
            [False, False, False],
        ],
        dtype=bool,
    )

    assert first_done_steps(dones).tolist() == [2, 1, 0]
    assert future_survival_lengths(dones).tolist() == [
        [3.0, 2.0, 1.0],
        [2.0, 1.0, 3.0],
        [1.0, 2.0, 2.0],
        [1.0, 1.0, 1.0],
    ]


def test_official_buffer_survival_reward_outputs_track_future_survival_length():
    dones = np.asarray(
        [
            [False, False],
            [False, True],
            [True, False],
            [False, False],
        ],
        dtype=bool,
    )
    future_len = future_survival_lengths(dones)
    rewards = np.ones_like(future_len, dtype=np.float32)
    episode_starts = np.zeros_like(future_len, dtype=np.float32)
    episode_starts[0] = 1.0
    episode_starts[1:] = dones[:-1].astype(np.float32)

    official = _official_buffer_outputs(
        rewards=rewards,
        episode_starts=episode_starts,
        dones_last=dones[-1],
        action_dim=2,
        gamma=0.98,
        gae_lambda=0.90,
        value_target_space="raw",
        actor_gae_space="raw",
    )

    assert float(_pearson(official["returns"], future_len)) > 0.95
    assert float(_pearson(official["advantages"], future_len)) > 0.95
    assert float(_pearson(official["actor_advantages"], future_len)) > 0.95


def test_freeze_counterfactual_h_list_materializes_lazy_rollout_latents():
    class FakePrior:
        def _sample_dims(self, h):
            h.setdefault("_constrained_obs_u", 0.25)
            h.setdefault("_constrained_noise_u", 0.75)
            return 4, 2, 1, 1, 0

        @staticmethod
        def _sample_reward_dropout_ratio(h):
            assert h["reward_dropout_randomize"]
            return 0.42

        @staticmethod
        def _sample_exact_scm_reward_term_enabled(h, *, prob_key, latent_key):
            prob = float(h.get(prob_key, 0.0))
            if prob > 0.0:
                h.setdefault(latent_key, 0.1)
            return prob > 0.0

    h_list = [
        {
            "reward_dropout_enabled": True,
            "reward_dropout_randomize": True,
            "ctrl_reward_enable_prob": 0.5,
            "survival_reward_enable_prob": 0.0,
        }
    ]

    frozen = _freeze_counterfactual_h_list(FakePrior(), h_list)

    assert frozen[0]["_constrained_obs_u"] == 0.25
    assert frozen[0]["_constrained_noise_u"] == 0.75
    assert frozen[0]["reward_dropout_randomize"] is False
    assert frozen[0]["reward_dropout_ratio"] == 0.42
    assert frozen[0]["_ctrl_reward_enable_u"] == 0.1
    assert "_survival_reward_enable_u" not in frozen[0]
    assert "_constrained_obs_u" not in h_list[0]
