import json
import sys

from ticl.analysis import phase3_residual_anchor_gae_path_probe as probe_mod


def test_classify_flip_cause_distinguishes_bootstrap_vs_future_carry():
    assert (
        probe_mod._classify_flip_cause(
            raw_rollout_residual_raw=1.0,
            actor_adv_norm=-0.1,
            delta_norm=0.2,
            reward_norm=0.3,
            bootstrap_term_norm=-0.1,
        )
        == "negative_gae_future_carry"
    )
    assert (
        probe_mod._classify_flip_cause(
            raw_rollout_residual_raw=1.0,
            actor_adv_norm=-0.1,
            delta_norm=-0.2,
            reward_norm=0.1,
            bootstrap_term_norm=-0.3,
        )
        == "bootstrap_term_negative"
    )
    assert (
        probe_mod._classify_flip_cause(
            raw_rollout_residual_raw=1.0,
            actor_adv_norm=-0.1,
            delta_norm=-0.2,
            reward_norm=-0.05,
            bootstrap_term_norm=-0.15,
        )
        == "bootstrap_term_negative"
    )
    assert (
        probe_mod._classify_flip_cause(
            raw_rollout_residual_raw=1.0,
            actor_adv_norm=-0.1,
            delta_norm=-0.2,
            reward_norm=-0.3,
            bootstrap_term_norm=0.1,
        )
        == "delta_negative_mixed"
    )
    assert (
        probe_mod._classify_flip_cause(
            raw_rollout_residual_raw=-0.1,
            actor_adv_norm=-0.1,
            delta_norm=-0.2,
            reward_norm=0.0,
            bootstrap_term_norm=0.0,
        )
        == "not_flip"
    )


def test_build_objective_position_maps_tracks_episode_positions():
    local_by_step, episode_idx, token_in_episode, to_end = probe_mod._build_objective_position_maps(
        [False, True, True, False, True, True, True],
        [(0, 2), (2, 5)],
    )
    assert local_by_step == {1: 0, 2: 1, 4: 2, 5: 3, 6: 4}
    assert episode_idx == {0: 0, 1: 0, 2: 1, 3: 1, 4: 1}
    assert token_in_episode == {0: 0, 1: 1, 2: 0, 3: 1, 4: 2}
    assert to_end == {0: 1, 1: 0, 2: 2, 3: 1, 4: 0}


def test_reshape_like_rewards_accepts_flat_storage():
    reshaped = probe_mod._reshape_like_rewards(
        [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]],
        (2, 3),
        name="episode_starts",
    )
    assert reshaped.shape == (2, 3)
    assert reshaped.tolist() == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]


def test_phase3_residual_anchor_gae_path_probe_reuses_existing_output(tmp_path, monkeypatch, capsys):
    output_path = tmp_path / "residual_anchor_gae_path.json"
    payload = {"ok": True, "kind": "residual_anchor_gae_path"}
    output_path.write_text(json.dumps(payload))

    monkeypatch.setattr(
        probe_mod,
        "assert_phase2_green",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not preflight when reusing output")),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "phase3_residual_anchor_gae_path_probe.py",
            "/tmp/checkpoint.cpkt",
            "--output-json",
            str(output_path),
        ],
    )

    assert probe_mod.main() == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == payload
