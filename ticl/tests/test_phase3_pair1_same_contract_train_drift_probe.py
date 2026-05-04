from ticl.analysis import phase3_pair1_same_contract_train_drift_probe as probe_mod


def test_build_pair1_same_contract_train_drift_probe_reports_collect_origin(monkeypatch):
    base_capture = {
        "phase2_reference_contract": {"n_steps": 256},
        "config": {
            "checkpoint_path": "/tmp/checkpoint.cpkt",
            "train_suite_path": "/tmp/pair1/train_suite.pt",
            "device": "cuda:0",
            "n_steps": 256,
            "batch_size": 256,
            "outer_epochs": 1,
            "single_eval_pos": 64,
            "train_profile": "phase2_shared_backbone_contract",
            "strict_fixed_env_mode": True,
            "deterministic_actor_sampling": True,
            "deterministic_batch_plan": True,
            "strict_native_rollout": True,
            "actor_objective_mode": "tokenwise",
            "actor_baseline_mode": "learned",
            "actor_gae_space": "normalized",
            "normalize_advantage": False,
        },
        "runtime": {"elapsed_wall_time_sec": 1.0, "rollout_wall_time_sec": 0.5, "update_wall_time_sec": 0.25},
        "collect_snapshot_path": "/tmp/collect.json",
        "train_snapshot_path": "/tmp/train.json",
        "collect_snapshot": {
            "summary": {"objective_total": 128},
            "array_digests": {
                key: {"sha256": "same"} for key in probe_mod._COLLECT_DIGEST_KEYS
            },
        },
        "train_snapshot": {
            "summary": {"outer_batch_idx": 1, "objective_total": 128},
            "flat_batch_indices": [0, 1],
            "flat_env_indices": [0, 0],
            "flat_step_indices": [0, 1],
            "flat_objective_episode_indices": [0, 0],
            "flat_objective_episode_positions": [0, 1],
            "flat_objective_global_positions": [0, 1],
            "objective_mask": [1, 1],
            "episode_starts": [1, 0],
            "seq_start_indices": [0],
            "seq_lengths": [2],
            "actions": [[0.1], [0.2]],
            "old_values": [0.3, 0.4],
            "old_log_prob": [0.5, 0.6],
            "returns": [0.7, 0.8],
            "pre_normalization_actor_advantages": [0.9, 1.0],
            "post_normalization_advantages": [0.9, 1.0],
        },
    }
    primary = base_capture
    repeat = {
        **base_capture,
        "collect_snapshot": {
            "summary": {"objective_total": 128},
            "array_digests": {
                key: {"sha256": ("diff" if key == "log_probs" else "same")}
                for key in probe_mod._COLLECT_DIGEST_KEYS
            },
        },
    }

    calls = []

    def _fake_run_single_capture(**kwargs):
        calls.append(dict(kwargs))
        return primary if len(calls) == 1 else repeat

    monkeypatch.setattr(probe_mod, "_run_single_capture", _fake_run_single_capture)
    monkeypatch.setattr(
        probe_mod,
        "_load_pair1_preflight",
        lambda: {
            "preflight_passed": True,
            "contract_args": {},
            "contract_diff": {"allowed_diff": {}, "unexpected_diff": {}},
        },
    )

    report = probe_mod.build_pair1_same_contract_train_drift_probe()

    assert calls[0]["collect_snapshot_json"] == probe_mod.PRIMARY_COLLECT_SNAPSHOT_JSON
    assert calls[0]["train_snapshot_json"] == probe_mod.PRIMARY_TRAIN_SNAPSHOT_JSON
    assert calls[1]["collect_snapshot_json"] == probe_mod.REPEAT_COLLECT_SNAPSHOT_JSON
    assert calls[1]["train_snapshot_json"] == probe_mod.REPEAT_TRAIN_SNAPSHOT_JSON
    assert report["conclusions"]["drift_origin"] == "collect"
    assert report["conclusions"]["collect_identical"] is False
    assert report["conclusions"]["first_train_batch_input_identical"] is True
    assert report["comparison"]["collect"]["differing_keys"] == ["log_probs"]


def test_build_pair1_same_contract_train_drift_probe_reports_post_collect_origin(monkeypatch):
    capture = {
        "phase2_reference_contract": {"n_steps": 256},
        "config": {
            "checkpoint_path": "/tmp/checkpoint.cpkt",
            "train_suite_path": "/tmp/pair1/train_suite.pt",
            "device": "cuda:0",
            "n_steps": 256,
            "batch_size": 256,
            "outer_epochs": 1,
            "single_eval_pos": 64,
            "train_profile": "phase2_shared_backbone_contract",
            "strict_fixed_env_mode": True,
            "deterministic_actor_sampling": True,
            "deterministic_batch_plan": True,
            "strict_native_rollout": True,
            "actor_objective_mode": "tokenwise",
            "actor_baseline_mode": "learned",
            "actor_gae_space": "normalized",
            "normalize_advantage": False,
        },
        "runtime": {"elapsed_wall_time_sec": 1.0, "rollout_wall_time_sec": 0.5, "update_wall_time_sec": 0.25},
        "collect_snapshot_path": "/tmp/collect.json",
        "train_snapshot_path": "/tmp/train.json",
        "collect_snapshot": {
            "summary": {"objective_total": 128},
            "array_digests": {
                key: {"sha256": "same"} for key in probe_mod._COLLECT_DIGEST_KEYS
            },
        },
        "train_snapshot": {
            "summary": {"outer_batch_idx": 1, "objective_total": 128},
            "flat_batch_indices": [0, 1],
            "flat_env_indices": [0, 0],
            "flat_step_indices": [0, 1],
            "flat_objective_episode_indices": [0, 0],
            "flat_objective_episode_positions": [0, 1],
            "flat_objective_global_positions": [0, 1],
            "objective_mask": [1, 1],
            "episode_starts": [1, 0],
            "seq_start_indices": [0],
            "seq_lengths": [2],
            "actions": [[0.1], [0.2]],
            "old_values": [0.3, 0.4],
            "old_log_prob": [0.5, 0.6],
            "returns": [0.7, 0.8],
            "pre_normalization_actor_advantages": [0.9, 1.0],
            "post_normalization_advantages": [0.9, 1.0],
        },
    }
    repeat = {
        **capture,
        "train_snapshot": {
            **capture["train_snapshot"],
            "old_log_prob": [0.5, 0.61],
        },
    }

    calls = []

    def _fake_run_single_capture(**kwargs):
        calls.append(dict(kwargs))
        return capture if len(calls) == 1 else repeat

    monkeypatch.setattr(probe_mod, "_run_single_capture", _fake_run_single_capture)
    monkeypatch.setattr(
        probe_mod,
        "_load_pair1_preflight",
        lambda: {
            "preflight_passed": True,
            "contract_args": {},
            "contract_diff": {"allowed_diff": {}, "unexpected_diff": {}},
        },
    )

    report = probe_mod.build_pair1_same_contract_train_drift_probe()

    assert report["conclusions"]["collect_identical"] is True
    assert report["conclusions"]["first_train_batch_input_identical"] is False
    assert report["conclusions"]["drift_origin"] == "train_input_transport"
    assert report["comparison"]["train"]["differing_float_keys"] == ["old_log_prob"]
