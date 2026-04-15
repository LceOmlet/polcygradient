from ticl.analysis.phase3_phase2_milestone_contract_gradient_compare import (
    _build_phase2_milestone_signature,
    _max_abs_diff_tensor,
    _signature_diff,
)


def test_signature_diff_reports_only_changed_keys() -> None:
    diff = _signature_diff(
        {"a": 1, "b": 2, "c": None},
        {"a": 1, "b": 3, "d": 4},
    )
    assert diff == {
        "b": {"phase2": 2, "phase3": 3},
        "d": {"phase2": None, "phase3": 4},
    }


def test_max_abs_diff_tensor_is_zero_for_equal_tensors() -> None:
    torch = __import__("torch")
    a = torch.tensor([1.0, -2.0], dtype=torch.float32)
    b = torch.tensor([1.0, -2.0], dtype=torch.float32)
    assert _max_abs_diff_tensor(a, b) == 0.0


def test_phase2_milestone_signature_locks_strict_native_rollout() -> None:
    args = {
        "n_steps": 256,
        "learning_rate": 2e-4,
        "batch_size": 256,
        "n_epochs": 1,
        "train_env_seed": 2020,
        "train_rollout_seed": 4040,
        "target_kl": 0.03,
        "strict_native_rollout": True,
    }
    signature = _build_phase2_milestone_signature(config={"optimizer": {}}, args=args)
    assert signature["strict_native_rollout"] is True
