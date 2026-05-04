from types import SimpleNamespace

import numpy as np
import torch

from ticl.analysis.critic_value_fit_probe import _critic_only_step


class _DummyBarDist:
    def __call__(self, logits: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
        return (logits.reshape(-1) - returns.reshape(-1)).square()

    def nll_from_bucket_idx(self, logits: torch.Tensor, bucket_idx: torch.Tensor) -> torch.Tensor:
        return (logits.reshape(-1) - bucket_idx.reshape(-1).float()).square()


class _DummyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.value_head = torch.nn.Linear(1, 1, bias=False)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.1)

    def evaluate_actions_with_hidden_flat(self, observations, actions, *, seq_lengths, action_masks):
        del actions, seq_lengths, action_masks
        return {"value_logits": self.value_head(observations)}

    def get_value_bardist(self):
        return _DummyBarDist()


def test_critic_only_step_switches_to_train_and_clips_gradients(monkeypatch):
    policy = _DummyPolicy()
    policy.eval()
    algo = SimpleNamespace(policy=policy, max_grad_norm=0.5)
    rollout_data = SimpleNamespace(
        observations=torch.tensor([[1.0], [2.0]], dtype=torch.float32),
        actions=torch.zeros((2, 1), dtype=torch.float32),
        seq_lengths=np.array([2], dtype=np.int64),
        action_masks=None,
        objective_masks=torch.ones(2, dtype=torch.float32),
        returns=torch.tensor([1.0, 2.0], dtype=torch.float32),
    )

    clipped = {}
    orig_clip = torch.nn.utils.clip_grad_norm_

    def _record_clip(parameters, max_norm, *args, **kwargs):
        params = list(parameters)
        clipped["called"] = True
        clipped["max_norm"] = float(max_norm)
        clipped["num_params"] = len(params)
        return orig_clip(params, max_norm, *args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", _record_clip)

    before = policy.value_head.weight.detach().clone()
    loss = _critic_only_step(algo, rollout_data)
    after = policy.value_head.weight.detach().clone()

    assert float(loss) >= 0.0
    assert policy.training is True
    assert clipped == {
        "called": True,
        "max_norm": 0.5,
        "num_params": 1,
    }
    assert not torch.allclose(before, after)


def test_critic_only_step_prefers_bucket_targets_when_available():
    class _RecordingBarDist(_DummyBarDist):
        def __init__(self):
            self.calls = []

        def __call__(self, logits: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
            self.calls.append(("returns", returns.detach().clone()))
            return super().__call__(logits, returns)

        def nll_from_bucket_idx(self, logits: torch.Tensor, bucket_idx: torch.Tensor) -> torch.Tensor:
            self.calls.append(("bucket", bucket_idx.detach().clone()))
            return super().nll_from_bucket_idx(logits, bucket_idx)

    class _BucketPolicy(_DummyPolicy):
        def __init__(self):
            super().__init__()
            self._bardist = _RecordingBarDist()

        def get_value_bardist(self):
            return self._bardist

    policy = _BucketPolicy()
    algo = SimpleNamespace(policy=policy, max_grad_norm=0.5)
    rollout_data = SimpleNamespace(
        observations=torch.tensor([[1.0], [2.0]], dtype=torch.float32),
        actions=torch.zeros((2, 1), dtype=torch.float32),
        seq_lengths=np.array([2], dtype=np.int64),
        action_masks=None,
        objective_masks=torch.ones(2, dtype=torch.float32),
        returns=torch.tensor([1.0, 2.0], dtype=torch.float32),
        value_target_bucket_idx=torch.tensor([3, 4], dtype=torch.int64),
    )

    _critic_only_step(algo, rollout_data)

    assert len(policy._bardist.calls) == 1
    kind, bucket_idx = policy._bardist.calls[0]
    assert kind == "bucket"
    assert torch.equal(bucket_idx, torch.tensor([3, 4], dtype=torch.int64))
