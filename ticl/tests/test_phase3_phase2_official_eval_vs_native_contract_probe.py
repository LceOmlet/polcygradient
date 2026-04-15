from ticl.analysis.phase3_phase2_official_eval_vs_native_contract_probe import (
    _compare_reduction,
)


def test_compare_reduction_reports_exact_native_closure():
    reference = {
        "default_first_hidden_max_abs_diff": 0.015625,
        "default_rollout_max_abs_diff": 0.001,
        "default_grad_max_abs_delta": 5e-6,
    }
    native_report = {
        "first_call_hidden_compare": {"max_abs_diff": 0.0},
        "native_measure_compare": {
            "rollout_tensor_max_abs_diff": {"actions": 0.0, "returns": 0.0},
            "grad_compare": {"max_abs_grad_delta": 0.0},
        },
    }
    result = _compare_reduction(reference, native_report)
    assert result["first_hidden_closed_exactly"] is True
    assert result["rollout_tensors_closed_exactly"] is True
    assert result["gradient_closed_exactly"] is True
    assert result["first_hidden_ratio_vs_default"] == 0.0
    assert result["rollout_max_ratio_vs_default"] == 0.0
    assert result["grad_max_ratio_vs_default"] == 0.0


def test_compare_reduction_keeps_partial_ratios():
    reference = {
        "default_first_hidden_max_abs_diff": 0.015625,
        "default_rollout_max_abs_diff": 0.004,
        "default_grad_max_abs_delta": 8e-6,
    }
    native_report = {
        "first_call_hidden_compare": {"max_abs_diff": 0.0078125},
        "native_measure_compare": {
            "rollout_tensor_max_abs_diff": {"actions": 0.001, "returns": 0.002},
            "grad_compare": {"max_abs_grad_delta": 2e-6},
        },
    }
    result = _compare_reduction(reference, native_report)
    assert result["first_hidden_closed_exactly"] is False
    assert result["rollout_tensors_closed_exactly"] is False
    assert result["gradient_closed_exactly"] is False
    assert result["first_hidden_ratio_vs_default"] == 0.5
    assert result["rollout_max_ratio_vs_default"] == 0.5
    assert result["grad_max_ratio_vs_default"] == 0.25
