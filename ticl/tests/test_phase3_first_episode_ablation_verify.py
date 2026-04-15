import json
import sys

from ticl.analysis import phase3_first_episode_ablation_verify as probe_mod


def test_phase3_first_episode_ablation_verify_reuses_existing_output(tmp_path, monkeypatch, capsys):
    output_path = tmp_path / "first_episode_ablation_verify.json"
    payload = {"ok": True, "kind": "first_episode_ablation_verify"}
    output_path.write_text(json.dumps(payload))

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "phase3_first_episode_ablation_verify.py",
            "--tail-mask-probe-json",
            "/tmp/tail_mask.json",
            "--output-json",
            str(output_path),
        ],
    )

    assert probe_mod.main() == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == payload
