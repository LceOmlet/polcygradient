import json
import sys

from ticl.analysis import phase3_post_reset_tail_band_probe as probe_mod


def test_phase3_post_reset_tail_band_probe_reuses_existing_output(tmp_path, monkeypatch, capsys):
    output_path = tmp_path / "post_reset_tail_band_probe.json"
    payload = {"ok": True, "kind": "post_reset_tail_band_probe"}
    output_path.write_text(json.dumps(payload))

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "phase3_post_reset_tail_band_probe.py",
            "--tail-mask-probe-json",
            "/tmp/tail_mask.json",
            "--output-json",
            str(output_path),
        ],
    )

    assert probe_mod.main() == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == payload
