import json
import sys

from ticl.analysis import phase3_post_reset_tail_dist2_late_ablation_verify as probe_mod


def test_phase3_post_reset_tail_dist2_late_ablation_verify_reuses_existing_output(tmp_path, monkeypatch, capsys):
    output_path = tmp_path / "post_reset_tail_dist2_late_ablation_verify.json"
    payload = {"ok": True, "kind": "post_reset_tail_dist2_late_ablation_verify"}
    output_path.write_text(json.dumps(payload))

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "phase3_post_reset_tail_dist2_late_ablation_verify.py",
            "/tmp/checkpoint.cpkt",
            "--train-suite-path",
            "/tmp/train_suite.json",
            "--heldout-suite-path",
            "/tmp/heldout_suite.json",
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
