import csv
import json
from argparse import Namespace
from pathlib import Path

from scripts.exploratory import phase2_profile_v2e_smoke_summary as summary


def _write_csv(path: Path) -> None:
    rows = []
    for rule in ("gym_full_obs_action_range", "gym_q90_obs_action_range"):
        for idx in range(20):
            rows.append(
                {
                    "rule": rule,
                    "locomotion_witness": str(idx % 2 == 0),
                    "low_energy_not_ctrl_only": str(idx % 3 == 0),
                    "sign_or_phase_sensitive": str(idx % 4 == 0),
                    "state_conditioned_energy_injection": str(idx % 5 == 0),
                    "strict_feedback_candidate": "True",
                    "strict_feedback_profile": "decay" if idx >= 5 else "constant",
                    "strict_feedback_gain": "1.0" if idx < 10 else "2.0",
                    "v2e_strict_near_best_gain_1": str(idx % 2 == 0),
                    "v2e_strict_near_best_gain_0.5": str(idx % 4 == 0),
                    "mountaincar_future_basin_candidate": str(idx == 0),
                    "mountaincar_future_basin_strong": str(idx == 1),
                }
            )
        for idx in range(44):
            rows.append(
                {
                    "rule": rule,
                    "locomotion_witness": str(idx % 2 == 0),
                    "low_energy_not_ctrl_only": str(idx % 3 == 0),
                    "sign_or_phase_sensitive": str(idx % 4 == 0),
                    "state_conditioned_energy_injection": str(idx % 5 == 0),
                    "strict_feedback_candidate": "False",
                    "mountaincar_future_basin_candidate": "False",
                    "mountaincar_future_basin_strong": "False",
                }
            )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sidecar_row(idx: int, update: int) -> dict:
    return {
        "env_idx": idx,
        "env_group_env_seed": 1000 + idx,
        "raw_reward_sum": float(idx + update),
        "reward_env_sum": float(idx + update),
        "training_reward_sum": float(idx + update),
        "done_count": float(idx % 4 + update % 2),
        "first_done_step": float(10 + idx % 7 + update),
        "action_value_absmean": float(0.2 + 0.01 * idx + 0.1 * update),
        "action_per_dim_temporal_std_mean": float(0.1 + 0.001 * idx + 0.01 * update),
        "next_state_step_l2_delta_mean": float(0.5 + 0.01 * idx + 0.02 * update),
        "raw_reward_std": float(0.2 + 0.01 * idx),
        "training_reward_std": float(0.2 + 0.01 * idx),
        "next_state_rms": float(1.0 + 0.01 * idx),
        "raw_policy_action_unit_clip_saturation_fraction": 0.0,
        "action_constant_dim_fraction_std_lt_0p05": 0.1,
    }


def _write_sidecars(run_dir: Path) -> None:
    sidecar = run_dir / "semantic_sidecars"
    sidecar.mkdir(parents=True)
    for update in (0, 3):
        path = sidecar / f"prior_update_{update:06d}_envs.jsonl"
        path.write_text(
            "".join(json.dumps(_sidecar_row(idx, update)) + "\n" for idx in range(64)),
            encoding="utf-8",
        )


def test_v2e_smoke_summary_reports_profile_not_delta_positive(tmp_path: Path):
    selected = tmp_path / "selected.csv"
    full = tmp_path / "full"
    q90 = tmp_path / "q90"
    out = tmp_path / "out"
    _write_csv(selected)
    _write_sidecars(full)
    _write_sidecars(q90)

    report = summary.run(
        Namespace(
            output_dir=str(out),
            selected_csv=str(selected),
            full_run_dir=str(full),
            q90_run_dir=str(q90),
        )
    )

    assert (
        report["scopes"]["gym_full_obs_action_range"]["selected_profile"][
            "strict_feedback_count"
        ]
        == 20
    )
    assert report["overall"]["diversity_guard_pass"] is True
    assert report["contract"]["delta_sign_fraction_not_reported"] is True
    payload = (out / "profile_v2e_smoke_summary.json").read_text(encoding="utf-8")
    assert "positive_fraction" not in payload
