from ticl.analysis.phase3_pair1_sign_flip_explanation_probe import (
    _leave_one_out_means,
    _random_partition_sign_flip_rate,
    _trimmed_mean,
)


def test_trimmed_mean_handles_heavy_tail():
    values = [-100.0, -2.0, -1.0, 1.0, 2.0, 200.0]
    assert _trimmed_mean(values, 1) == 0.0
    assert _trimmed_mean(values, 2) == 0.0


def test_leave_one_out_means_capture_outlier_sensitivity():
    values = [-100.0, 1.0, 2.0, 3.0]
    loo = dict(_leave_one_out_means(values))
    assert loo[0] > 0.0
    assert loo[3] < 0.0


def test_random_partition_sign_flip_rate_is_positive_for_mixed_heavy_tail_distribution():
    combined = [-50.0, -40.0, -30.0, 1.0, 2.0, 3.0, 80.0, 90.0]
    result = _random_partition_sign_flip_rate(combined, group_size=4, n_trials=2000, seed=0)
    assert result["sign_flip_rate"] > 0.0
    assert result["both_pos_rate"] > 0.0
