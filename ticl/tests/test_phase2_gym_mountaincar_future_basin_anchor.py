import json

from scripts.exploratory import phase2_gym_mountaincar_future_basin_anchor as anchor


def test_early_velocity_pump_decays_after_prefix():
    policy = anchor._make_policy("early_velocity_pump", prefix_steps=3)

    assert policy([0.0, 0.1], 0) == 1.0
    assert policy([0.0, -0.1], 2) == -1.0
    assert policy([0.0, 0.1], 3) == 0.0


def test_mountaincar_anchor_materializes_small_report(tmp_path):
    class Args:
        output_dir = str(tmp_path)
        seed_start = 0
        n_seeds = 4
        prefix_steps = 100
        max_steps = 999
        min_prefix_action_abs = 0.5
        max_suffix_action_abs = 0.05
        min_suffix_gain_vs_zero = 50.0
        min_total_gain_vs_zero = 50.0
        min_suffix_gain_vs_open = 0.0
        min_anchor_rate = 0.5

    anchor.run(Args())

    report_path = tmp_path / "gym_mountaincar_future_basin_anchor_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["analysis_entry"] == "phase2_gym_mountaincar_future_basin_anchor"
    assert report["contract"]["gym_anchor_only"] is True
    assert report["per_seed_summary"]["anchor_rate"] >= 0.5
