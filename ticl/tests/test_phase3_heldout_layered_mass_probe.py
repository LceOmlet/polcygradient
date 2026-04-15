import json
import sys

from ticl.analysis import phase3_heldout_layered_mass_probe as probe_mod


def test_phase3_heldout_layered_mass_probe_reuses_existing_output(tmp_path, monkeypatch, capsys):
    output_path = tmp_path / "heldout_layered_mass.json"
    payload = {"ok": True, "kind": "heldout_layered_mass"}
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
            "phase3_heldout_layered_mass_probe.py",
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


def test_first_supported_layer_prefers_earliest_matching_stage():
    layer_summaries = {
        "raw_return_positive_mass": {
            "corr_vs_terminal_resets": 0.61,
            "corr_vs_pre_suffix_gap": -0.10,
            "top2_pre_gap_capture": {"mass_share": 0.01},
        },
        "raw_residual_positive_mass": {
            "corr_vs_terminal_resets": 0.95,
            "corr_vs_pre_suffix_gap": -0.20,
            "top2_pre_gap_capture": {"mass_share": 0.001},
        },
        "raw_actor_adv_positive_mass": {
            "corr_vs_terminal_resets": 0.95,
            "corr_vs_pre_suffix_gap": -0.20,
            "top2_pre_gap_capture": {"mass_share": 0.001},
        },
        "normalized_actor_adv_positive_mass": {
            "corr_vs_terminal_resets": 0.95,
            "corr_vs_pre_suffix_gap": -0.20,
            "top2_pre_gap_capture": {"mass_share": 0.001},
        },
    }
    assert probe_mod._first_supported_layer(layer_summaries) == "raw_return_positive_mass"
