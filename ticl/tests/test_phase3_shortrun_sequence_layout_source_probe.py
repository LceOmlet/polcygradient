import json
import sys

from ticl.analysis import phase3_shortrun_sequence_layout_source_probe as probe_mod


def test_classify_instability_marks_tail_only_drift_as_cosmetic():
    comparisons = {
        "step_hashes.episode_starts": True,
        "step_hashes.dones": True,
        "step_hashes.rewards": True,
        "step_hashes.actions_valid": True,
        "step_hashes.actions_tail": False,
        "buffer_hashes.episode_starts": True,
        "buffer_hashes.returns": True,
        "batch_plan_hashes.batch_inds": True,
        "batch_plan_hashes.env_change": True,
        "flat_batch_hashes.seq_start_indices": True,
        "flat_batch_hashes.seq_lengths": True,
    }
    out = probe_mod._classify_instability(comparisons)
    assert out["masked_action_tail_is_cosmetic_only"] is True
    assert out["terminal_reset_or_episode_starts_instability"] is False
    assert out["seq_packing_or_flatten_order_instability"] is False
    assert out["return_path_driver"] == "stable"


def test_phase3_shortrun_sequence_layout_source_probe_reuses_existing_output(tmp_path, monkeypatch, capsys):
    output_path = tmp_path / "probe.json"
    payload = {"ok": True, "kind": "sequence_layout_source"}
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
            "phase3_shortrun_sequence_layout_source_probe.py",
            "/tmp/checkpoint.cpkt",
            "--output-json",
            str(output_path),
        ],
    )

    assert probe_mod.main() == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == payload
