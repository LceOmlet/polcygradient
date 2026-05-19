import json
from pathlib import Path

from scripts.exploratory import phase2_pendulum_settle_reduction_spec as spec


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_reduction_spec_defines_mechanism_before_tests_or_generator_changes(tmp_path: Path):
    closure = _write_json(
        tmp_path / "closure.json",
        {
            "disallowed_repairs": [
                {"name": "raw_delta_or_raw_positive_quota", "reason": "monitor only"},
                {"name": "simple_reference_inertia_highway_knob", "reason": "zero support"},
            ],
            "allowed_repairs": [
                {"name": "cheap_clean_pendulum_settle_guard", "status": "necessary"}
            ],
        },
    )
    reward_relevant = _write_json(
        tmp_path / "reward_relevant.json",
        {"decision": {"cheap_suffix_state_settle_guard_ready": False, "min_count": 0}},
    )

    report = spec.build_report(
        closure_register_json=closure,
        reward_relevant_audit_json=reward_relevant,
    )

    assert report["contract"]["does_not_modify_exact_scm"] is True
    assert report["contract"]["test_is_only_for_falsification"] is True
    assert report["reduction_decision"]["should_define_new_reduction_before_more_tests"] is True
    assert report["reduction_decision"]["should_implement_generator_now"] is False
    assert report["mechanism"]["id"] == "controllable_quadratic_settle_factor"
    necessary = {item["id"] for item in report["necessary_conditions"]}
    assert {"controllable_subspace", "reward_state_alignment", "context_identifiability"} <= necessary
    sufficient = {item["id"] for item in report["sufficient_conditions_for_candidate"]}
    assert {"analytic_witness", "settle_witness", "startup_safe"} <= sufficient
    assert report["minimal_test_plan"][0]["stage"] == "analytic_preflight"
