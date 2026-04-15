import json
import sys

from ticl.analysis import phase3_terminal_tail_interaction_probe as probe_mod


def test_phase3_terminal_tail_interaction_probe_reuses_existing_output(tmp_path, monkeypatch, capsys):
    output_path = tmp_path / "tail_interaction_probe.json"
    payload = {"ok": True, "kind": "tail_interaction_probe"}
    output_path.write_text(json.dumps(payload))

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "phase3_terminal_tail_interaction_probe.py",
            "--tail-mask-probe-json",
            "/tmp/tail_mask.json",
            "--output-json",
            str(output_path),
        ],
    )

    assert probe_mod.main() == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == payload
