import json
import sys

import numpy as np

from ticl.analysis import phase3_shortrun_buffer_transport_contract_scan as probe_mod


def test_compare_transport_field_flags_perfect_alignment_and_mismatch():
    source = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    result = probe_mod._compare_transport_field(
        source_flat=source,
        flat_batch_field=source.copy(),
        padded_batch_field=source.copy(),
        padded_valid_mask=np.asarray([True, True, True, True], dtype=bool),
    )
    assert result["source_to_flat"]["exact_match"] is True
    assert result["source_to_padded_valid"]["exact_match"] is True
    assert result["flat_to_padded_valid"]["exact_match"] is True

    perturbed = np.asarray([0.0, 1.0, 2.0, 3.5], dtype=np.float32)
    result = probe_mod._compare_transport_field(
        source_flat=source,
        flat_batch_field=source.copy(),
        padded_batch_field=perturbed,
        padded_valid_mask=np.asarray([True, True, True, True], dtype=bool),
    )
    assert result["source_to_flat"]["exact_match"] is True
    assert result["source_to_padded_valid"]["exact_match"] is False
    assert result["flat_to_padded_valid"]["mismatch_count"] == 1


def test_compare_seq_layout_recovers_starts_and_lengths():
    flat_seq_start_indices = np.asarray([0, 2], dtype=np.int64)
    flat_seq_lengths = np.asarray([2, 3], dtype=np.int64)
    padded_mask = np.asarray([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)
    result = probe_mod._compare_seq_layout(flat_seq_start_indices, flat_seq_lengths, padded_mask)
    assert result["seq_start_indices_exact_match"] is True
    assert result["seq_lengths_exact_match"] is True
    assert result["padded_valid_count_match"] is True


def test_phase3_shortrun_buffer_transport_contract_scan_reuses_existing_output(tmp_path, monkeypatch, capsys):
    output_path = tmp_path / "probe.json"
    payload = {"ok": True, "kind": "buffer_transport_contract"}
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
            "phase3_shortrun_buffer_transport_contract_scan.py",
            "/tmp/checkpoint.cpkt",
            "--output-json",
            str(output_path),
        ],
    )

    assert probe_mod.main() == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == payload
