from ticl.analysis.phase3_phase2_milestone_contract_gradient_compare import (
    _build_phase2_milestone_signature,
    _install_batch1_non_native_compare_shim,
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


def test_install_batch1_non_native_compare_shim_forces_native_on_batch1_cuda(monkeypatch) -> None:
    torch = __import__("torch")

    class DummyCore:
        def __init__(self):
            self.training = False
            self.force_native_eval_forward_step = False
            self.calls = []

        def forward_step(self, token, state=None):
            self.calls.append(bool(self.force_native_eval_forward_step))
            return token, state

    class DummyModel:
        def __init__(self):
            self.rwkv_core = DummyCore()

    class DummyPolicy:
        def __init__(self):
            self.rlpfn_model = DummyModel()

    class DummyAlgo:
        def __init__(self):
            self.policy = DummyPolicy()

    algo = DummyAlgo()
    meta = _install_batch1_non_native_compare_shim(algo)
    assert meta["batch1_non_native_compare_shim_applied"] is True

    token = torch.zeros((1, 4), dtype=torch.float32)

    class FakeCudaTensor:
        def __init__(self, base):
            self._base = base

        @property
        def ndim(self):
            return self._base.ndim

        @property
        def shape(self):
            return self._base.shape

        @property
        def is_cuda(self):
            return True

        def __getattr__(self, name):
            return getattr(self._base, name)

    out, _ = algo.policy.rlpfn_model.rwkv_core.forward_step(FakeCudaTensor(token), None)
    assert torch.equal(out._base, token)
    assert algo.policy.rlpfn_model.rwkv_core.calls == [False]
