import json
import sys

from ticl.analysis import phase3_root_cause_probe as probe_mod


def test_phase3_root_cause_probe_reuses_existing_output(tmp_path, monkeypatch, capsys):
    output_path = tmp_path / "phase3_root_cause_probe.json"
    payload = {"ok": True, "kind": "phase3_root_cause"}
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
            "phase3_root_cause_probe.py",
            "--output-json",
            str(output_path),
        ],
    )

    assert probe_mod.main() == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == payload
