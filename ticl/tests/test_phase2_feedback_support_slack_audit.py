import json
from pathlib import Path

from scripts.exploratory import phase2_feedback_support_slack_audit as audit


DEFAULT_FEEDBACK = [
    "decay_obs_linear_0x100",
    "decay_obs_linear_neg_0x100",
    "decay_obs_linear_1x100",
    "decay_obs_linear_neg_1x100",
]
EXPANDED_FEEDBACK = DEFAULT_FEEDBACK + [
    "early_obs_linear_0x100_then_zero",
    "early_obs_linear_neg_0x100_then_zero",
    "early_obs_linear_1x100_then_zero",
    "early_obs_linear_neg_1x100_then_zero",
]


def _write_guard(path: Path, *, selected_csv: str, feedback_modes: list[str], full: int, q90: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "inputs": {"selected_csv": selected_csv},
                "config": {
                    "n_steps": 64,
                    "n_envs": 64,
                    "mode_count_per_rule": len(feedback_modes) + 4,
                    "baseline_modes": ["zero", "random", "neg_random", "half_random"],
                    "feedback_modes": feedback_modes,
                },
                "rules": {
                    "gym_full_obs_action_range": {
                        "candidate_count": full,
                        "candidate_rate": full / 64.0,
                        "candidate_gain_counts": {"1.0": full},
                    },
                    "gym_q90_obs_action_range": {
                        "candidate_count": q90,
                        "candidate_rate": q90 / 64.0,
                        "candidate_gain_counts": {"1.0": q90},
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_short_guard(path: Path, *, selected_csv: str, full: int, q90: int) -> Path:
    out = _write_guard(path, selected_csv=selected_csv, feedback_modes=DEFAULT_FEEDBACK, full=full, q90=q90)
    payload = json.loads(out.read_text(encoding="utf-8"))
    payload["config"]["n_steps"] = 32
    out.write_text(json.dumps(payload), encoding="utf-8")
    return out


def test_support_slack_dedupes_and_blocks_exact_selector_floor(tmp_path: Path):
    seed1 = _write_guard(
        tmp_path / "seed9601" / "feedback_budgeted_support_guard.json",
        selected_csv="/tmp/seed9601/selected_prior_envs.csv",
        feedback_modes=DEFAULT_FEEDBACK,
        full=10,
        q90=10,
    )
    duplicate_seed1 = _write_guard(
        tmp_path / "duplicate" / "feedback_budgeted_support_guard.json",
        selected_csv="/tmp/seed9601/selected_prior_envs.csv",
        feedback_modes=DEFAULT_FEEDBACK,
        full=10,
        q90=10,
    )
    seed2 = _write_guard(
        tmp_path / "seed9602" / "feedback_budgeted_support_guard.json",
        selected_csv="/tmp/seed9602/selected_prior_envs.csv",
        feedback_modes=DEFAULT_FEEDBACK,
        full=8,
        q90=10,
    )
    short_probe = _write_short_guard(
        tmp_path / "seed9603_s32" / "feedback_budgeted_support_guard.json",
        selected_csv="/tmp/seed9603/selected_prior_envs.csv",
        full=1,
        q90=1,
    )

    report = audit.build_report(artifact_jsons=[seed1, duplicate_seed1, seed2, short_probe])

    group = report["groups"]["default_decay_obs2"]
    assert report["inputs"]["raw_record_count"] == 4
    assert report["inputs"]["deduped_record_count"] == 3
    assert report["inputs"]["slack_eligible_record_count"] == 2
    assert group["all_meet_selector_minimum"] is True
    assert group["all_meet_slack_target"] is False
    assert group["by_rule"]["gym_full_obs_action_range"]["min"] == 8
    assert report["decision"]["ready_to_clear_slack_blocker"] is False


def test_support_slack_passes_only_with_cross_seed_slack(tmp_path: Path):
    seed1 = _write_guard(
        tmp_path / "seed9601" / "feedback_budgeted_support_guard.json",
        selected_csv="/tmp/seed9601/selected_prior_envs.csv",
        feedback_modes=EXPANDED_FEEDBACK,
        full=12,
        q90=13,
    )
    seed2 = _write_guard(
        tmp_path / "seed9602" / "feedback_budgeted_support_guard.json",
        selected_csv="/tmp/seed9602/selected_prior_envs.csv",
        feedback_modes=EXPANDED_FEEDBACK,
        full=14,
        q90=12,
    )

    report = audit.build_report(artifact_jsons=[seed1, seed2])

    group = report["groups"]["decay_plus_early_obs2_g1"]
    assert group["cross_seed"] is True
    assert group["all_meet_slack_target"] is True
    assert group["ready_to_clear_slack_blocker"] is True
    assert report["decision"]["ready_to_clear_slack_blocker"] is True


def test_support_slack_blocks_dominated_multigain_candidates(tmp_path: Path):
    feedback = EXPANDED_FEEDBACK + [
        "decay_obs_linear_0x200",
        "decay_obs_linear_neg_0x200",
        "decay_obs_linear_1x200",
        "decay_obs_linear_neg_1x200",
    ]
    seed1 = _write_guard(
        tmp_path / "seed9601" / "feedback_budgeted_support_guard.json",
        selected_csv="/tmp/seed9601/selected_prior_envs.csv",
        feedback_modes=feedback,
        full=16,
        q90=18,
    )
    seed2 = _write_guard(
        tmp_path / "seed9602" / "feedback_budgeted_support_guard.json",
        selected_csv="/tmp/seed9602/selected_prior_envs.csv",
        feedback_modes=feedback,
        full=16,
        q90=18,
    )
    for path in (seed1, seed2):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["rules"]["gym_full_obs_action_range"]["candidate_gain_counts"] = {"1.0": 1, "2.0": 15}
        payload["rules"]["gym_q90_obs_action_range"]["candidate_gain_counts"] = {"1.0": 3, "2.0": 15}
        path.write_text(json.dumps(payload), encoding="utf-8")

    report = audit.build_report(artifact_jsons=[seed1, seed2])

    group = report["groups"]["decay_plus_early_obs2_g1_2"]
    assert group["all_meet_slack_target"] is True
    assert group["gain_diversity_guard_pass"] is False
    assert group["ready_to_clear_slack_blocker"] is False
    assert "slack_candidate_gain_dominance" in report["decision"]["blockers"]
