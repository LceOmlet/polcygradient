import json
from pathlib import Path

from scripts.exploratory import phase2_pendulum_settle_prior_derived_repair_probe as probe


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _label_rows(guards: list[bool]) -> list[dict]:
    rows = []
    for idx, guard in enumerate(guards):
        rows.append(
            {
                "env_index": idx,
                "cheap_settle_guard": guard,
                "reward_condition": idx % 2 == 0,
                "suffix_reward_condition": idx % 3 == 0,
                "action_decay_condition": idx % 2 == 1,
                "self_state_settle_condition": guard,
                "open_state_settle_condition": guard,
            }
        )
    return rows


def _h_rows(n: int) -> list[dict]:
    return [
        {
            "alpha": 0.05 + 0.01 * idx,
            "state_output_scale": 1.0,
            "ctrl_reward_weight": 0.01 + 0.001 * idx,
            "family": "scm",
        }
        for idx in range(n)
    ]


def test_probe_blocks_h_only_when_existing_positive_count_is_too_low(tmp_path: Path):
    preflight = _write_json(
        tmp_path / "preflight.json",
        {
            "decision": {"required_min_count": 2},
            "lists": [
                {"label": "gym_full_obs_action_range", "classification": {"per_env": _label_rows([True, False, False])}},
                {"label": "gym_q90_obs_action_range", "classification": {"per_env": _label_rows([False, False, False])}},
            ],
        },
    )
    readiness = _write_json(
        tmp_path / "readiness.json",
        {
            "existing_knob_trials": {"existing_knobs_falsified": True},
            "quadratic_settle_mechanism": {"standalone_specimen_integration_allowed": False},
            "cheap_or_cached_guard": {"short_temporal_cached_label_viable": True},
        },
    )
    full_h = _write_json(tmp_path / "full_h.json", _h_rows(3))
    q90_h = _write_json(tmp_path / "q90_h.json", _h_rows(3))

    report = probe.build_report(
        preflight_json=preflight,
        readiness_json=readiness,
        full_h_list_json=full_h,
        q90_h_list_json=q90_h,
    )

    assert report["target_counts"]["min_guard_count"] == 0
    assert report["decision"]["can_close_with_h_only_selection"] is False
    assert report["decision"]["safe_next_repair_class"] == "prior_internal_exact_scm_mechanism_required"
    assert report["decision"]["can_use_standalone_quadratic_generator"] is False


def test_probe_allows_h_only_when_each_rule_has_enough_existing_positive_h(tmp_path: Path):
    preflight = _write_json(
        tmp_path / "preflight.json",
        {
            "decision": {"required_min_count": 2},
            "lists": [
                {"label": "gym_full_obs_action_range", "classification": {"per_env": _label_rows([True, True, False])}},
                {"label": "gym_q90_obs_action_range", "classification": {"per_env": _label_rows([True, False, True])}},
            ],
        },
    )
    readiness = _write_json(tmp_path / "readiness.json", {"existing_knob_trials": {}})
    full_h = _write_json(tmp_path / "full_h.json", _h_rows(3))
    q90_h = _write_json(tmp_path / "q90_h.json", _h_rows(3))

    report = probe.build_report(
        preflight_json=preflight,
        readiness_json=readiness,
        full_h_list_json=full_h,
        q90_h_list_json=q90_h,
    )

    assert report["target_counts"]["min_guard_count"] == 2
    assert report["decision"]["can_close_with_h_only_selection"] is True
    assert report["decision"]["safe_next_repair_class"] == "h_only_selector_possible"
