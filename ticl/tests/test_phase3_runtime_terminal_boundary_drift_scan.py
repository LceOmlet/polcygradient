import json
import sys

from ticl.analysis import phase3_runtime_terminal_boundary_drift_scan as probe_mod


def _suite_payload(*, mismatch: bool = False) -> dict:
    # Three rollout steps, one env, terminal objective token at final step.
    # mean/std = 4/2 => expected boundary shift = -2.0
    # reward_raw/std = -1/2 = -0.5
    # recovered reward_norm = -2.5
    # advantage = reward_norm - value = -2.5 - 0.3 = -2.8
    advantage = -2.8 if not mismatch else -2.55
    return {
        "suite_name": "pair1",
        "suite_split": "train",
        "suite_path": "/tmp/train_suite.pt",
        "batch_size": 1,
        "env_seeds": [101],
        "rollout_seeds": [202],
        "rewards": [[0.0], [0.0], [-1.0]],
        "values_norm": [[0.0], [0.0], [0.3]],
        "advantages_norm": [[0.0], [0.0], [advantage]],
        "objective_masks": [[0.0], [1.0], [1.0]],
        "episode_starts": [[1.0], [0.0], [0.0]],
        "rollout_return_means": [[4.0], [4.0], [4.0]],
        "rollout_return_stds": [[2.0], [2.0], [2.0]],
        "last_dones": [True],
        "runtime_wall_s": 1.0,
        "collect_contract_debug": {
            "strict_fixed_env_mode": True,
            "env_rng_seed_spec": [101],
            "rollout_rng_seed_spec": [202],
            "deterministic_actor_sampling": False,
            "deterministic_batch_plan": True,
            "last_collect_rollout_env_rng_seeds": [101],
            "last_collect_rollout_rollout_rng_seeds": [202],
        },
    }


def test_analyze_suite_payload_detects_zero_drift_when_contract_matches():
    result = probe_mod._analyze_suite_payload(_suite_payload(mismatch=False), tol=1e-6)
    suite_summary = result["suite_summary"]
    terminal_row = result["terminal_rows"][0]

    assert suite_summary["objective_terminal_token_count"] == 1
    assert suite_summary["boundary_shift_mismatch_count"] == 0
    assert abs(float(terminal_row["boundary_shift_contract_error"])) <= 1e-6
    assert terminal_row["contract_matches_full_rollout_centering"] is True


def test_analyze_suite_payload_flags_runtime_boundary_drift():
    result = probe_mod._analyze_suite_payload(_suite_payload(mismatch=True), tol=1e-6)
    suite_summary = result["suite_summary"]
    terminal_row = result["terminal_rows"][0]

    assert suite_summary["boundary_shift_mismatch_count"] == 1
    assert abs(float(terminal_row["boundary_shift_contract_error"])) > 1e-6
    assert terminal_row["contract_matches_full_rollout_centering"] is False


def test_main_reuses_existing_output(tmp_path, monkeypatch, capsys):
    output_path = tmp_path / "phase3_runtime_terminal_boundary_drift_scan.json"
    payload = {"ok": True, "kind": "phase3_runtime_terminal_boundary_drift_scan"}
    output_path.write_text(json.dumps(payload))

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "phase3_runtime_terminal_boundary_drift_scan.py",
            "/tmp/mock.ckpt",
            "--output-json",
            str(output_path),
        ],
    )

    assert probe_mod.main() == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == payload
