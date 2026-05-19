import json
from pathlib import Path

from scripts.exploratory import phase2_profile_unprofiled_platform_audit as audit


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.write_text(
        ",".join(keys)
        + "\n"
        + "\n".join(",".join(str(row.get(key, "")) for key in keys) for row in rows)
        + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_unprofiled_platform_uses_effective_pendulum_axis_not_short_gap():
    row = {
        "pendulum_axis_support": False,
        "pendulum_short_gap": True,
        "low_energy_not_ctrl_only": False,
        "state_conditioned_energy_injection": False,
        "strict_feedback_candidate": False,
    }

    assert audit.is_unprofiled_platform(row) is True

    row["pendulum_axis_support"] = True
    assert audit.is_unprofiled_platform(row) is False


def test_build_report_attributes_delta_pressure_to_unprofiled_cell(tmp_path: Path):
    selected_csv = tmp_path / "selected.csv"
    _write_csv(
        selected_csv,
        [
            {
                "candidate_rule": "gym_full_obs_action_range",
                "profile_band_rank": 0,
                "_source_index": 0,
                "pendulum_axis_support": False,
                "low_energy_not_ctrl_only": False,
                "state_conditioned_energy_injection": False,
                "strict_feedback_candidate": False,
                "locomotion_witness": False,
                "sign_or_phase_sensitive": True,
                "terminal_reset_count_target": 10,
                "action_dim": 2,
                "obs_dim": 4,
                "state_dim": 8,
            },
            {
                "candidate_rule": "gym_full_obs_action_range",
                "profile_band_rank": 1,
                "_source_index": 1,
                "pendulum_axis_support": False,
                "low_energy_not_ctrl_only": True,
                "state_conditioned_energy_injection": False,
                "strict_feedback_candidate": False,
                "locomotion_witness": False,
                "sign_or_phase_sensitive": False,
                "terminal_reset_count_target": 30,
                "action_dim": 6,
                "obs_dim": 12,
                "state_dim": 16,
            },
        ],
    )
    source_csv = tmp_path / "source.csv"
    _write_csv(
        source_csv,
        [
            {
                "rule": "gym_full_obs_action_range",
                "_source_index": 0,
                "unsafe_nonpend_triple_false": True,
                "terminal_reset_count_target": 10,
                "action_dim": 2,
                "obs_dim": 4,
                "state_dim": 8,
            },
            {
                "rule": "gym_full_obs_action_range",
                "_source_index": 1,
                "unsafe_nonpend_triple_false": False,
                "terminal_reset_count_target": 30,
                "action_dim": 6,
                "obs_dim": 12,
                "state_dim": 16,
            },
        ],
    )
    run_dir = tmp_path / "run"
    _write_jsonl(
        run_dir / "semantic_sidecars" / "prior_update_000000_envs.jsonl",
        [
            {"env_idx": 0, "training_reward_sum": 10.0, "raw_reward_sum": 100.0},
            {"env_idx": 1, "training_reward_sum": 1.0, "raw_reward_sum": 10.0},
        ],
    )
    _write_jsonl(
        run_dir / "semantic_sidecars" / "prior_update_000001_envs.jsonl",
        [
            {"env_idx": 0, "training_reward_sum": 2.0, "raw_reward_sum": 40.0},
            {"env_idx": 1, "training_reward_sum": 3.0, "raw_reward_sum": 20.0},
        ],
    )
    _write_jsonl(
        run_dir / "prior_progress.jsonl",
        [
            {
                "update": {
                    "logger": {
                        "train/approx_kl": 0.01,
                        "train/clip_fraction": 0.2,
                        "train/explained_variance": 0.5,
                    },
                    "param_delta": {"param_delta_l2": 1.0},
                }
            }
        ],
    )

    report, group_rows, env_rows = audit.build_report(
        source_annotated_csv=source_csv,
        run_specs=[
            audit.RunSpec(
                "unit",
                selected_csv,
                run_dir,
                "gym_full_obs_action_range",
            )
        ],
    )

    groups = {row["group"]: row for row in group_rows}
    assert report["contract"]["negative_pressure_label_not_selector_quota"] is True
    assert report["source_pool"]["unprofiled_platform_count"] == 1
    assert groups["unprofiled_platform"]["n"] == 1
    assert groups["unprofiled_platform"]["train_delta_mean"] == -8.0
    assert groups["unprofiled_platform"]["raw_improved_count"] == 0
    assert groups["supported_nonpend"]["train_delta_mean"] == 2.0
    assert groups["supported_nonpend"]["raw_improved_count"] == 1
    assert len(env_rows) == 2
