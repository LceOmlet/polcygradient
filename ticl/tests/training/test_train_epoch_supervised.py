import torch
from torch import nn

from ticl.train import train_epoch


class _DummySupervisedModel(nn.Module):
    def __init__(self, in_features=4, out_features=1):
        super().__init__()
        self.proj = nn.Linear(in_features, out_features)

    def forward(self, src, single_eval_pos=None):
        assert single_eval_pos is not None
        return self.proj(src[single_eval_pos:])


def test_train_epoch_supervised_does_not_require_pg_artifacts():
    model = _DummySupervisedModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss(reduction="none")

    t, b, f = 4, 2, 4
    data = torch.randn(t, b, f)
    targets = torch.randn(t, b, 1)
    dl = [(data, targets, 1)]

    mean_loss, nan_share, ignore_share = train_epoch(
        model=model,
        aggregate_k_gradients=1,
        using_dist=False,
        scaler=None,
        dl=dl,
        device="cpu",
        optimizer=optimizer,
        criterion=criterion,
        n_out=1,
        epoch_idx=0,
        progress_bar=False,
        epoch_profiler=None,
        epoch_start_time=None,
        train_profiler_log_every_batches=0,
        verbose=False,
        kernel_profiler=None,
    )

    assert torch.isfinite(torch.tensor(mean_loss))
    assert nan_share == 0.0
    assert ignore_share == 0.0
