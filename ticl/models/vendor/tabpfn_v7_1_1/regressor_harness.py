"""Thin TabPFN v7.1.1 regressor harness for TICL backbones."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from torch import nn

from .bar_distribution import FullSupportBarDistribution
from .runtime_regression_loss import _compute_regression_loss
from .runtime_regression_utils import translate_probs_across_borders

REGRESSION_CONSTANT_TARGET_BORDER_EPSILON = 1e-5


def make_standardized_full_support_bar_distribution(
    *,
    num_buckets: int = 100,
    value_range: float = 5.0,
    ignore_nan_targets: bool = True,
) -> FullSupportBarDistribution:
    if int(num_buckets) < 2:
        raise ValueError(f"num_buckets must be at least 2, got {num_buckets}")
    if float(value_range) <= 0.0:
        raise ValueError(f"value_range must be positive, got {value_range}")
    borders = torch.linspace(
        -float(value_range),
        float(value_range),
        int(num_buckets) + 1,
        dtype=torch.float32,
    )
    return FullSupportBarDistribution(
        borders,
        ignore_nan_targets=ignore_nan_targets,
    )


class TabPFNOfficialRegressorHarness(nn.Module):
    """Official TabPFN regression head/loss/decode glue for a backbone hidden state."""

    def __init__(
        self,
        *,
        emsize: int,
        num_buckets: int = 100,
        value_range: float = 5.0,
        ignore_nan_targets: bool = True,
    ) -> None:
        super().__init__()
        self.emsize = int(emsize)
        self.num_buckets = int(num_buckets)
        self.value_range = float(value_range)
        self.ignore_nan_targets = bool(ignore_nan_targets)

        self.output_projection = nn.Sequential(
            nn.Linear(self.emsize, 2 * self.emsize),
            nn.GELU(),
            nn.Linear(2 * self.emsize, self.num_buckets),
        )
        self.reset_target_statistics()

    def reset_target_statistics(self) -> None:
        self.znorm_space_bardist_ = make_standardized_full_support_bar_distribution(
            num_buckets=self.num_buckets,
            value_range=self.value_range,
            ignore_nan_targets=self.ignore_nan_targets,
        )
        self.raw_space_bardist_ = FullSupportBarDistribution(
            self.znorm_space_bardist_.borders.detach().clone(),
            ignore_nan_targets=self.ignore_nan_targets,
        )
        self.y_train_mean_ = 0.0
        self.y_train_std_ = 1.0
        self.is_constant_target_ = False
        self.constant_value_ = None

    def set_target_statistics(self, y_train: torch.Tensor | np.ndarray | Sequence[float]) -> None:
        y_tensor = torch.as_tensor(y_train, dtype=torch.float32).reshape(-1)
        if int(y_tensor.numel()) <= 0:
            raise ValueError("y_train must contain at least one target.")

        valid = y_tensor[torch.isfinite(y_tensor)]
        if int(valid.numel()) <= 0:
            raise ValueError("y_train must contain at least one finite target.")

        unique = torch.unique(valid)
        self.is_constant_target_ = bool(int(unique.numel()) == 1)
        self.constant_value_ = float(valid[0].item()) if self.is_constant_target_ else None

        if self.is_constant_target_:
            border_adjustment = max(
                abs(self.constant_value_ * REGRESSION_CONSTANT_TARGET_BORDER_EPSILON),
                REGRESSION_CONSTANT_TARGET_BORDER_EPSILON,
            )
            self.znorm_space_bardist_ = FullSupportBarDistribution(
                borders=torch.tensor(
                    [
                        self.constant_value_ - border_adjustment,
                        self.constant_value_ + border_adjustment,
                    ],
                    dtype=torch.float32,
                ),
                ignore_nan_targets=self.ignore_nan_targets,
            )
            self.raw_space_bardist_ = self.znorm_space_bardist_
            self.y_train_mean_ = float(self.constant_value_)
            self.y_train_std_ = 1.0
            return

        mean = valid.mean()
        std = valid.std(unbiased=False)
        self.y_train_mean_ = float(mean.item())
        self.y_train_std_ = float(std.item()) + 1e-20
        raw_borders = self.znorm_space_bardist_.borders * self.y_train_std_ + self.y_train_mean_
        self.raw_space_bardist_ = FullSupportBarDistribution(
            raw_borders.to(dtype=torch.float32),
            ignore_nan_targets=self.ignore_nan_targets,
        ).float()

    def forward_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        if int(hidden.shape[-1]) != self.emsize:
            raise ValueError(
                f"Expected hidden last dim {self.emsize}, got {tuple(hidden.shape)}"
            )
        head_param = next(self.output_projection.parameters())
        if hidden.dtype != head_param.dtype or hidden.device != head_param.device:
            hidden = hidden.to(device=head_param.device, dtype=head_param.dtype)
        logits = self.output_projection(hidden)
        if int(logits.shape[-1]) != self.num_buckets:
            raise RuntimeError(
                f"Official regressor head must emit {self.num_buckets} buckets, got {tuple(logits.shape)}"
            )
        return logits

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.forward_logits(hidden)

    def compute_regression_loss(
        self,
        *,
        hidden: torch.Tensor | None = None,
        logits: torch.Tensor | None = None,
        targets_BQ: torch.Tensor,
        ce_loss_weight: float = 0.0,
        crps_loss_weight: float = 1.0,
        crls_loss_weight: float = 0.0,
        mse_loss_weight: float = 1.0,
        mse_loss_clip: float | None = None,
        mae_loss_weight: float = 0.0,
        mae_loss_clip: float | None = None,
    ) -> torch.Tensor:
        if logits is None:
            if hidden is None:
                raise ValueError("Either hidden or logits must be provided.")
            logits = self.forward_logits(hidden)
        if int(logits.ndim) != 3:
            raise ValueError(f"logits must have shape (B, Q, L), got {tuple(logits.shape)}")
        if tuple(logits.shape[:-1]) != tuple(targets_BQ.shape):
            raise ValueError(
                f"targets_BQ must match logits leading dims, got {tuple(targets_BQ.shape)} for {tuple(logits.shape)}"
            )
        return _compute_regression_loss(
            logits_BQL=logits.to(dtype=torch.float32),
            targets_BQ=targets_BQ.to(device=logits.device, dtype=torch.float32),
            bardist_loss_fn=self.znorm_space_bardist_,
            ce_loss_weight=ce_loss_weight,
            crps_loss_weight=crps_loss_weight,
            crls_loss_weight=crls_loss_weight,
            mse_loss_weight=mse_loss_weight,
            mse_loss_clip=mse_loss_clip,
            mae_loss_weight=mae_loss_weight,
            mae_loss_clip=mae_loss_clip,
        )

    def aggregate_logits_across_borders(
        self,
        *,
        logits_list: Sequence[torch.Tensor],
        from_borders: Sequence[torch.Tensor | FullSupportBarDistribution],
        use_raw_space: bool = False,
        average_before_softmax: bool = False,
    ) -> torch.Tensor:
        if len(logits_list) != len(from_borders):
            raise ValueError("logits_list and from_borders must have the same length.")
        if len(logits_list) == 0:
            raise ValueError("Need at least one logits tensor to aggregate.")

        criterion = self.raw_space_bardist_ if use_raw_space else self.znorm_space_bardist_
        canonical_borders = criterion.borders
        translated_logits = []
        for logits, source in zip(logits_list, from_borders):
            source_borders = source.borders if isinstance(source, FullSupportBarDistribution) else source
            probs = translate_probs_across_borders(
                logits,
                frm=source_borders.to(device=logits.device, dtype=logits.dtype),
                to=canonical_borders.to(device=logits.device, dtype=logits.dtype),
            )
            translated_logits.append(probs.log() if average_before_softmax else probs)

        stacked = torch.stack(translated_logits, dim=0)
        if average_before_softmax:
            return (stacked.mean(dim=0)).log_softmax(dim=-1)
        return stacked.mean(dim=0).log()

    def _criterion(self, *, use_raw_space: bool) -> FullSupportBarDistribution:
        return self.raw_space_bardist_ if use_raw_space else self.znorm_space_bardist_

    def _constant_output(self, logits: torch.Tensor | None, *, quantiles: Sequence[float]) -> dict[str, Any]:
        if logits is not None:
            base_shape = tuple(logits.shape[:-1])
            sample_shape = base_shape
            device = logits.device
        else:
            sample_shape = tuple()
            device = self.znorm_space_bardist_.borders.device
        value = torch.full(sample_shape, float(self.constant_value_), device=device, dtype=torch.float32)
        return {
            "logits": torch.zeros((*sample_shape, 1), device=device, dtype=torch.float32),
            "criterion": self.znorm_space_bardist_,
            "mean": value,
            "median": value.clone(),
            "mode": value.clone(),
            "quantiles": [value.clone() for _ in quantiles],
        }

    def predict_full(
        self,
        *,
        hidden: torch.Tensor | None = None,
        logits: torch.Tensor | None = None,
        quantiles: Sequence[float] | None = None,
        use_raw_space: bool = True,
    ) -> dict[str, Any]:
        if quantiles is None:
            quantiles = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
        if logits is None:
            if hidden is None:
                raise ValueError("Either hidden or logits must be provided.")
            logits = self.forward_logits(hidden)
        if self.is_constant_target_:
            return self._constant_output(logits, quantiles=quantiles)
        criterion = self._criterion(use_raw_space=use_raw_space)
        return {
            "logits": logits,
            "criterion": criterion,
            "mean": criterion.mean(logits),
            "median": criterion.median(logits),
            "mode": criterion.mode(logits),
            "quantiles": [criterion.icdf(logits, float(q)) for q in quantiles],
        }

    def predict_mean(self, *, hidden: torch.Tensor | None = None, logits: torch.Tensor | None = None, use_raw_space: bool = True) -> torch.Tensor:
        return self.predict_full(hidden=hidden, logits=logits, use_raw_space=use_raw_space)["mean"]

    def predict_median(self, *, hidden: torch.Tensor | None = None, logits: torch.Tensor | None = None, use_raw_space: bool = True) -> torch.Tensor:
        return self.predict_full(hidden=hidden, logits=logits, use_raw_space=use_raw_space)["median"]

    def predict_mode(self, *, hidden: torch.Tensor | None = None, logits: torch.Tensor | None = None, use_raw_space: bool = True) -> torch.Tensor:
        return self.predict_full(hidden=hidden, logits=logits, use_raw_space=use_raw_space)["mode"]

    def predict_quantiles(
        self,
        *,
        hidden: torch.Tensor | None = None,
        logits: torch.Tensor | None = None,
        quantiles: Sequence[float] = (0.1, 0.5, 0.9),
        use_raw_space: bool = True,
    ) -> list[torch.Tensor]:
        return self.predict_full(
            hidden=hidden,
            logits=logits,
            quantiles=quantiles,
            use_raw_space=use_raw_space,
        )["quantiles"]

    def pdf(
        self,
        ys: torch.Tensor,
        *,
        hidden: torch.Tensor | None = None,
        logits: torch.Tensor | None = None,
        use_raw_space: bool = True,
    ) -> torch.Tensor:
        if logits is None:
            if hidden is None:
                raise ValueError("Either hidden or logits must be provided.")
            logits = self.forward_logits(hidden)
        return self._criterion(use_raw_space=use_raw_space).pdf(logits, ys)

    def cdf(
        self,
        ys: torch.Tensor,
        *,
        hidden: torch.Tensor | None = None,
        logits: torch.Tensor | None = None,
        use_raw_space: bool = True,
    ) -> torch.Tensor:
        if logits is None:
            if hidden is None:
                raise ValueError("Either hidden or logits must be provided.")
            logits = self.forward_logits(hidden)
        return self._criterion(use_raw_space=use_raw_space).cdf(logits, ys)

    def icdf(
        self,
        q: float,
        *,
        hidden: torch.Tensor | None = None,
        logits: torch.Tensor | None = None,
        use_raw_space: bool = True,
    ) -> torch.Tensor:
        if logits is None:
            if hidden is None:
                raise ValueError("Either hidden or logits must be provided.")
            logits = self.forward_logits(hidden)
        return self._criterion(use_raw_space=use_raw_space).icdf(logits, float(q))
