import numpy as np

from scripts.exploratory.phase2_survival_candidate_probe import (
    _field_diffs_except_survival,
    _make_survival_candidate_h_list,
    _reward_view_survival_candidate_rollout,
    _terminal_sensitive_mask,
    _write_fixed_list_csvs,
)


def test_terminal_sensitive_mask_uses_action_counterfactual_spread():
    payloads = {
        "zero": {
            "first_done_steps": np.asarray([10, 20, 30], dtype=np.int64),
            "terminal_counts": np.asarray([1, 2, 3], dtype=np.int64),
        },
        "random": {
            "first_done_steps": np.asarray([14, 20, 44], dtype=np.int64),
            "terminal_counts": np.asarray([1, 3, 4], dtype=np.int64),
        },
        "obs_linear_policy": {
            "first_done_steps": np.asarray([15, 22, 32], dtype=np.int64),
            "terminal_counts": np.asarray([1, 2, 3], dtype=np.int64),
        },
    }

    mask = _terminal_sensitive_mask(
        payloads,
        first_done_spread_threshold=5,
        require_terminal_count_change=True,
    )

    assert mask.tolist() == [False, False, True]


def test_survival_candidate_only_changes_survival_fields():
    h_list = [
        {"state_dim": 4, "action_dim": 2, "terminal_reset_count_target": 3.0},
        {"state_dim": 5, "action_dim": 3, "terminal_reset_count_target": 4.0},
    ]

    candidate = _make_survival_candidate_h_list(
        h_list,
        np.asarray([True, False], dtype=bool),
        survival_weight=0.1,
    )

    assert candidate[0]["survival_reward_weight"] == 0.1
    assert candidate[0]["survival_reward_enable_prob"] == 1.0
    assert candidate[0]["_survival_reward_enable_u"] == 0.0
    assert "survival_reward_weight" not in candidate[1]
    assert _field_diffs_except_survival(h_list, candidate) == []


def test_reward_view_survival_candidate_keeps_dynamics_and_non_sensitive_rewards():
    base = {
        "rewards": np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "dones": np.asarray([[False, False], [True, False]], dtype=bool),
        "episode_starts": np.asarray([[1.0, 1.0], [0.0, 0.0]], dtype=np.float32),
        "final_terminal": np.asarray([True, False], dtype=bool),
        "actions": np.ones((2, 2, 3), dtype=np.float32),
        "next_states": np.arange(12, dtype=np.float32).reshape(2, 2, 3),
    }

    candidate = _reward_view_survival_candidate_rollout(
        base,
        np.asarray([True, False], dtype=bool),
        survival_weight=0.25,
    )

    np.testing.assert_array_equal(candidate["dones"], base["dones"])
    np.testing.assert_array_equal(candidate["episode_starts"], base["episode_starts"])
    np.testing.assert_array_equal(candidate["actions"], base["actions"])
    np.testing.assert_array_equal(candidate["next_states"], base["next_states"])
    np.testing.assert_allclose(candidate["rewards"][:, 0], base["rewards"][:, 0] + 0.25)
    np.testing.assert_allclose(candidate["rewards"][:, 1], base["rewards"][:, 1])


def test_write_fixed_list_csvs_emits_pack_runner_columns(tmp_path):
    class FakePrior:
        @staticmethod
        def _sample_dims(h):
            return h["state_dim"], h["obs_dim"], h["action_dim"], h.get("noise_dim", 1), h.get("zero_pad_dim", 0)

    base = [
        {"state_dim": 4, "obs_dim": 3, "action_dim": 2, "noise_dim": 1},
        {"state_dim": 5, "obs_dim": 2, "action_dim": 1, "noise_dim": 2},
    ]
    candidate = _make_survival_candidate_h_list(
        base,
        np.asarray([True, False], dtype=bool),
        survival_weight=0.5,
    )

    out = _write_fixed_list_csvs(
        prior=FakePrior(),
        base_h_list=base,
        candidate_h_list=candidate,
        env_seeds=[11, 12],
        sensitive_mask=np.asarray([True, False], dtype=bool),
        output_dir=tmp_path,
    )

    base_csv = tmp_path / "base_fixed_list.csv"
    cand_csv = tmp_path / "candidate_fixed_list.csv"
    assert base_csv.exists()
    assert cand_csv.exists()
    assert "full_frozen_h_json" in base_csv.read_text(encoding="utf-8").splitlines()[0]
    assert out["paired_diff_contract"]["unexpected_non_survival_h_field_diff_count"] == 0
    assert out["candidate"]["count"] == 2
