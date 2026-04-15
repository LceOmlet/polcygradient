import json
import sys

from ticl.analysis import phase3_tail_bias_propagation_probe as probe_mod


def test_phase3_tail_bias_propagation_probe_reuses_existing_output(tmp_path, monkeypatch, capsys):
    output_path = tmp_path / "tail_bias_propagation.json"
    payload = {"ok": True, "kind": "tail_bias_propagation"}
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
            "phase3_tail_bias_propagation_probe.py",
            "/tmp/checkpoint.cpkt",
            "--train-suite-path",
            "/tmp/train.pt",
            "--heldout-suite-path",
            "/tmp/heldout.pt",
            "--reuse-zero-control-json",
            "/tmp/zero.json",
            "--reuse-pre-policy-json",
            "/tmp/pre.json",
            "--output-json",
            str(output_path),
        ],
    )

    assert probe_mod.main() == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == payload
