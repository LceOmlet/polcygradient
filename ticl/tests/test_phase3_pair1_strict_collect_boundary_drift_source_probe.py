from ticl.analysis import phase3_pair1_strict_collect_boundary_drift_source_probe as probe_mod


def test_build_pair1_strict_collect_boundary_drift_source_probe_classifies_downstream_boundary(monkeypatch):
    base = {
        "config": {
            "checkpoint_path": "/tmp/checkpoint.cpkt",
            "train_suite_path": "/tmp/pair1/train_suite.pt",
            "device": "cuda:0",
            "n_steps": 256,
            "batch_size": 256,
            "single_eval_pos": 64,
            "strict_fixed_env_mode": True,
            "deterministic_actor_sampling": True,
            "deterministic_batch_plan": True,
            "strict_native_rollout": True,
            "actor_objective_mode": "tokenwise",
        },
        "runtime": {"elapsed_wall_time_sec": 1.0, "rollout_wall_time_sec": 0.5},
        "within_run_contract": {
            "episode_starts_follow_previous_dones": True,
            "step0_episode_starts_all_ones": True,
        },
        "step_arrays": {
            "obs": [[[1.0]], [[2.0]], [[3.0]], [[4.0]]],
            "action": [[[0.1]], [[0.2]], [[0.3]], [[0.4]]],
            "reward": [[0.0], [0.0], [1.0], [0.0]],
            "values": [[0.5], [0.6], [0.7], [0.8]],
            "log_probs": [[0.9], [0.9], [0.9], [0.9]],
            "dones": [[0], [0], [1], [0]],
            "episode_starts": [[1.0], [0.0], [0.0], [1.0]],
            "objective_masks": [[0], [0], [1], [1]],
            "objective_episode_indices": [[0], [0], [1], [2]],
            "objective_episode_positions": [[-1], [-1], [0], [0]],
        },
    }
    repeat = {
        **base,
        "step_arrays": {
            "obs": [[[1.0]], [[2.0]], [[3.0]], [[4.0]]],
            "action": [[[0.1]], [[0.25]], [[0.3]], [[0.4]]],
            "reward": [[0.0], [1.0], [0.0], [0.0]],
            "values": [[0.5], [0.65], [0.7], [0.8]],
            "log_probs": [[0.9], [0.9], [0.9], [0.9]],
            "dones": [[0], [1], [0], [0]],
            "episode_starts": [[1.0], [0.0], [1.0], [0.0]],
            "objective_masks": [[0], [0], [1], [1]],
            "objective_episode_indices": [[0], [0], [1], [1]],
            "objective_episode_positions": [[-1], [-1], [0], [1]],
        },
    }

    calls = []

    def _fake_capture(**kwargs):
        calls.append(dict(kwargs))
        return base if len(calls) == 1 else repeat

    monkeypatch.setattr(probe_mod, "_capture_one_collect", _fake_capture)
    monkeypatch.setattr(
        probe_mod,
        "_load_pair1_preflight",
        lambda: {
            "preflight_passed": True,
            "contract_args": {},
            "contract_diff": {"allowed_diff": {}, "unexpected_diff": {}},
        },
    )

    report = probe_mod.build_pair1_strict_collect_boundary_drift_source_probe()

    assert report["field_compare"]["obs"]["identical"] is True
    assert report["field_compare"]["objective_masks"]["identical"] is True
    assert report["field_compare"]["action"]["first_diff_step"] == 1
    assert report["field_compare"]["dones"]["first_diff_step"] == 1
    assert report["field_compare"]["episode_starts"]["first_diff_step"] == 2
    assert report["field_compare"]["objective_episode_indices"]["first_diff_step"] == 3
    assert report["conclusions"]["boundary_drift_is_downstream_of_action_drift"] is True


def test_build_pair1_strict_collect_boundary_drift_source_probe_detects_non_action_boundary_origin(monkeypatch):
    capture = {
        "config": {
            "checkpoint_path": "/tmp/checkpoint.cpkt",
            "train_suite_path": "/tmp/pair1/train_suite.pt",
            "device": "cuda:0",
            "n_steps": 256,
            "batch_size": 256,
            "single_eval_pos": 64,
            "strict_fixed_env_mode": True,
            "deterministic_actor_sampling": True,
            "deterministic_batch_plan": True,
            "strict_native_rollout": True,
            "actor_objective_mode": "tokenwise",
        },
        "runtime": {"elapsed_wall_time_sec": 1.0, "rollout_wall_time_sec": 0.5},
        "within_run_contract": {
            "episode_starts_follow_previous_dones": True,
            "step0_episode_starts_all_ones": True,
        },
        "step_arrays": {
            "obs": [[[1.0]], [[2.0]]],
            "action": [[[0.1]], [[0.2]]],
            "reward": [[0.0], [0.0]],
            "values": [[0.5], [0.6]],
            "log_probs": [[0.9], [0.9]],
            "dones": [[0], [0]],
            "episode_starts": [[1.0], [0.0]],
            "objective_masks": [[0], [1]],
            "objective_episode_indices": [[0], [0]],
            "objective_episode_positions": [[-1], [0]],
        },
    }
    repeat = {
        **capture,
        "step_arrays": {
            **capture["step_arrays"],
            "episode_starts": [[1.0], [1.0]],
            "objective_episode_indices": [[0], [1]],
            "objective_episode_positions": [[-1], [0]],
        },
    }

    calls = []

    def _fake_capture(**kwargs):
        calls.append(dict(kwargs))
        return capture if len(calls) == 1 else repeat

    monkeypatch.setattr(probe_mod, "_capture_one_collect", _fake_capture)
    monkeypatch.setattr(
        probe_mod,
        "_load_pair1_preflight",
        lambda: {
            "preflight_passed": True,
            "contract_args": {},
            "contract_diff": {"allowed_diff": {}, "unexpected_diff": {}},
        },
    )

    report = probe_mod.build_pair1_strict_collect_boundary_drift_source_probe()

    assert report["field_compare"]["action"]["identical"] is True
    assert report["field_compare"]["episode_starts"]["first_diff_step"] == 1
    assert report["conclusions"]["boundary_drift_is_downstream_of_action_drift"] is False
