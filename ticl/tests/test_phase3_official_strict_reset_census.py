from ticl.analysis import phase3_official_strict_reset_census as census_mod


def test_scale_chain_entry_status_rejects_env_without_distance_probe_reset():
    result = census_mod._scale_chain_entry_status(
        suite_name="pair1",
        env_index=3,
        distance_row={"objective_terminal_reset_count": 0},
        suite_payload={},
        suite_summary={},
        token_rows=[],
    )
    assert result["can_enter_terminal_local_scale_chain"] is False
    assert result["entry_failure_reason"] == "distance_probe_has_no_objective_terminal_reset"
    assert result["selected_target_local_available"] is False
    assert result["selected_target_local_index"] is None


def test_scale_chain_entry_status_reports_success_from_probe(monkeypatch):
    monkeypatch.setattr(
        census_mod,
        "_build_probe",
        lambda **kwargs: {
            "config": {"selected_target_local_index": 17},
            "conclusions": {
                "selected_target_local_available": True,
                "single_terminal_local": True,
            },
        },
    )
    token_rows = [
        {"objective_local_index": 17, "next_non_terminal": 0.0},
        {"objective_local_index": 5, "next_non_terminal": 1.0},
    ]
    result = census_mod._scale_chain_entry_status(
        suite_name="pair2",
        env_index=12,
        distance_row={"objective_terminal_reset_count": 1},
        suite_payload={},
        suite_summary={},
        token_rows=token_rows,
    )
    assert result["can_enter_terminal_local_scale_chain"] is True
    assert result["entry_failure_reason"] is None
    assert result["terminal_local_indices"] == [17]
    assert result["selected_target_local_available"] is True
    assert result["selected_target_local_index"] == 17
    assert result["single_terminal_local"] is True


def test_scale_chain_entry_status_captures_probe_failure(monkeypatch):
    monkeypatch.setattr(
        census_mod,
        "_build_probe",
        lambda **kwargs: (_ for _ in ()).throw(ValueError("Could not infer full-rollout mean/std from terminal token rows.")),
    )
    token_rows = [
        {"objective_local_index": 17, "next_non_terminal": 0.0},
        {"objective_local_index": 5, "next_non_terminal": 1.0},
    ]
    result = census_mod._scale_chain_entry_status(
        suite_name="pair2",
        env_index=12,
        distance_row={"objective_terminal_reset_count": 1},
        suite_payload={},
        suite_summary={},
        token_rows=token_rows,
    )
    assert result["can_enter_terminal_local_scale_chain"] is False
    assert "Could not infer full-rollout mean/std" in result["entry_failure_reason"]
    assert result["terminal_local_indices"] == [17]
