import json
import sys

from ticl.analysis import phase3_transfer_bottleneck_probe as probe_mod


def test_phase3_transfer_bottleneck_probe_reuses_existing_output(tmp_path, monkeypatch, capsys):
    output_path = tmp_path / "phase3_transfer_bottleneck_probe.json"
    payload = {"ok": True, "kind": "phase3_transfer_bottleneck"}
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
            "phase3_transfer_bottleneck_probe.py",
            "--output-json",
            str(output_path),
        ],
    )

    assert probe_mod.main() == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == payload
