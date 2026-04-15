import json
import sys

import numpy as np

from ticl.analysis import phase3_heldout_reset_semantics_probe as probe_mod


def test_phase3_heldout_reset_semantics_probe_reuses_existing_output(tmp_path, monkeypatch, capsys):
    output_path = tmp_path / "heldout_reset_semantics.json"
    payload = {"ok": True, "kind": "heldout_reset_semantics"}
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
            "phase3_heldout_reset_semantics_probe.py",
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


def test_objective_episode_segments_and_masks():
    objective_mask = np.array([0, 1, 1, 1, 1, 1, 1, 1], dtype=bool)
    episode_starts = np.array([0, 1, 0, 0, 1, 0, 0, 1], dtype=bool)
    segments = probe_mod._objective_episode_segments(objective_mask, episode_starts)
    assert segments == [(0, 3), (3, 6), (6, 7)]

    masks = probe_mod._segment_masks(7, segments, tail_len=2)
    assert masks["first_episode"].astype(int).tolist() == [1, 1, 1, 0, 0, 0, 0]
    assert masks["post_first_reset"].astype(int).tolist() == [0, 0, 0, 1, 1, 1, 1]
    assert masks["terminal_tail"].astype(int).tolist() == [0, 1, 1, 0, 1, 1, 1]
    assert masks["non_tail"].astype(int).tolist() == [1, 0, 0, 1, 0, 0, 0]


def test_objective_terminal_reset_count_uses_segment_definition():
    objective_mask = np.array([0, 1, 1, 1, 1, 1], dtype=bool)
    episode_starts = np.array([0, 0, 0, 1, 0, 0], dtype=bool)
    segments = probe_mod._objective_episode_segments(objective_mask, episode_starts)

    assert segments == [(0, 2), (2, 5)]
    assert probe_mod._objective_terminal_reset_count_from_segments(segments) == 1
