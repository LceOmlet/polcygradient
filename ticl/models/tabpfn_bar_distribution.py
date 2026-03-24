# Copyright (c) Prior Labs GmbH 2025.
#
# Copied and minimally adapted from the official TabPFN implementation:
# https://github.com/PriorLabs/TabPFN/blob/main/src/tabpfn/architectures/base/bar_distribution.py

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from typing_extensions import override

if TYPE_CHECKING:
    import matplotlib.pyplot as plt


class BarDistribution(nn.Module):
    def __init__(self, borders: torch.Tensor, *, ignore_nan_targets: bool = True):
        super().__init__()
        assert len(borders.shape) == 1
        borders = borders.contiguous()
        self.register_buffer("borders", borders)
        full_width = self.bucket_widths.sum()

        assert (1 - (full_width / (self.borders[-1] - self.borders[0]))).abs() < 1e-2, (
            f"diff: {full_width - (self.borders[-1] - self.borders[0])} with"
            f" {full_width} {self.borders[-1]} {self.borders[0]}"
        )
        assert (self.bucket_widths >= 0.0).all(), "Please provide sorted borders!"

        self.ignore_nan_targets = ignore_nan_targets
        self.to(borders.device)

    def has_equal_borders(self, other: "BarDistribution") -> bool:
        return torch.equal(self.borders, other.borders)

    @property
    def bucket_widths(self) -> torch.Tensor:
        return self.borders[1:] - self.borders[:-1]

    @property
    def num_bars(self) -> int:
        return len(self.borders) - 1

    def cdf(self, logits: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
        if len(ys.shape) < len(logits.shape) and len(ys.shape) == 1:
            ys = ys.repeat((*logits.shape[:-1], 1))
        else:
            assert ys.shape[:-1] == logits.shape[:-1], (
                f"ys.shape: {ys.shape} logits.shape: {logits.shape}"
            )
        probs = torch.softmax(logits, dim=-1)
        buckets_of_ys = self.map_to_bucket_idx(ys).clamp(0, self.num_bars - 1)

        prob_so_far = torch.cumsum(probs, dim=-1) - probs
        prob_left_of_bucket = prob_so_far.gather(-1, buckets_of_ys)

        share_of_bucket_left = (
            (ys - self.borders[buckets_of_ys]) / self.bucket_widths[buckets_of_ys]
        ).clamp(0.0, 1.0)
        prob_in_bucket = probs.gather(-1, buckets_of_ys) * share_of_bucket_left

        prob_left_of_ys = prob_left_of_bucket + prob_in_bucket
        prob_left_of_ys[ys <= self.borders[0]] = 0.0
        prob_left_of_ys[ys >= self.borders[-1]] = 1.0
        assert not torch.isnan(prob_left_of_ys).any()

        return prob_left_of_ys.clip(0.0, 1.0)

    def get_probs_for_different_borders(
        self,
        logits: torch.Tensor,
        new_borders: torch.Tensor,
    ) -> torch.Tensor:
        if (len(self.borders) == len(new_borders)) and (
            self.borders == new_borders
        ).all():
            return logits.softmax(-1)

        prob_left_of_borders = self.cdf(logits, new_borders)
        prob_left_of_borders[..., 0] = 0.0
        prob_left_of_borders[..., -1] = 1.0

        return (
            prob_left_of_borders[..., 1:] - prob_left_of_borders[..., :-1]
        ).clamp_min(0.0)

    def average_bar_distributions_into_this(
        self,
        list_of_bar_distributions: Sequence["BarDistribution"],
        list_of_logits: Sequence[torch.Tensor],
        *,
        average_logits: bool = False,
    ) -> torch.Tensor:
        probs = torch.stack(
            [
                bar_dist.get_probs_for_different_borders(logits, self.borders)
                for bar_dist, logits in zip(list_of_bar_distributions, list_of_logits)
            ],
            dim=0,
        )

        if average_logits:
            probs = probs.log().mean(dim=0).softmax(-1)
        else:
            probs = probs.mean(dim=0)

        return probs.log()

    def __setstate__(self, state: dict) -> None:
        state.pop("bucket_widths", None)
        super().__setstate__(state)
        self.__dict__.setdefault("append_mean_pred", False)

    def map_to_bucket_idx(self, y: torch.Tensor) -> torch.Tensor:
        assert (self.borders[1:] - self.borders[:-1] >= 0.0).all()
        target_sample = torch.searchsorted(self.borders, y) - 1
        target_sample[y == self.borders[0]] = 0
        target_sample[y == self.borders[-1]] = self.num_bars - 1
        return target_sample

    def ignore_init(self, y: torch.Tensor) -> torch.Tensor:
        ignore_loss_mask = torch.isnan(y)
        if ignore_loss_mask.any() and not self.ignore_nan_targets:
            raise ValueError(f"Found NaN in target {y}")

        y[ignore_loss_mask] = self.borders[0]
        return ignore_loss_mask

    def compute_scaled_log_probs(self, logits: torch.Tensor) -> torch.Tensor:
        bucket_log_probs = torch.log_softmax(logits, -1)
        return bucket_log_probs - torch.log(self.bucket_widths)

    def full_ce(self, logits: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        return -(probs * torch.log_softmax(logits, -1)).sum(-1)

    def forward(
        self,
        logits: torch.Tensor,
        y: torch.Tensor,
        mean_prediction_logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        y = y.clone().view(*logits.shape[:-1])
        ignore_loss_mask = self.ignore_init(y)
        target_sample = self.map_to_bucket_idx(y)
        assert (target_sample >= 0).all()
        assert (target_sample < self.num_bars).all(), (
            f"y {y} not in support set for borders (min_y, max_y) {self.borders}"
        )

        last_dim = logits.shape[-1]
        assert last_dim == self.num_bars, f"{last_dim} v {self.num_bars}"

        scaled_bucket_log_probs = self.compute_scaled_log_probs(logits)
        nll_loss = -scaled_bucket_log_probs.gather(
            -1,
            target_sample[..., None],
        ).squeeze(-1)

        if mean_prediction_logits is not None:
            nll_loss = torch.cat(
                (nll_loss, self.mean_loss(logits, mean_prediction_logits)),
                0,
            )

        nll_loss[ignore_loss_mask] = 0.0
        return nll_loss

    def mean_loss(
        self,
        logits: torch.Tensor,
        mean_prediction_logits: torch.Tensor,
    ) -> torch.Tensor:
        scaled_mean_log_probs = self.compute_scaled_log_probs(mean_prediction_logits)
        assert len(logits.shape) == 3, len(logits.shape)
        assert len(scaled_mean_log_probs.shape) == 2, len(scaled_mean_log_probs.shape)

        means = self.mean(logits).detach()
        target_mean = self.map_to_bucket_idx(means).clamp_(0, self.num_bars - 1)
        return -scaled_mean_log_probs.gather(1, target_mean.T).mean(1).unsqueeze(0)

    def mean(self, logits: torch.Tensor) -> torch.Tensor:
        bucket_means = self.borders[:-1] + self.bucket_widths / 2
        p = torch.softmax(logits, -1)
        return p @ bucket_means

    def median(self, logits: torch.Tensor) -> torch.Tensor:
        return self.icdf(logits, 0.5)

    def icdf(self, logits: torch.Tensor, left_prob: float) -> torch.Tensor:
        probs = logits.softmax(-1)
        cumprobs = torch.cumsum(probs, -1)
        idx = (
            torch.searchsorted(
                cumprobs,
                left_prob * torch.ones(*cumprobs.shape[:-1], 1, device=logits.device),
            )
            .squeeze(-1)
            .clamp(0, cumprobs.shape[-1] - 1)
        )

        cumprobs = torch.cat(
            [torch.zeros(*cumprobs.shape[:-1], 1, device=logits.device), cumprobs],
            -1,
        )
        rest_prob = left_prob - cumprobs.gather(-1, idx[..., None]).squeeze(-1)
        left_border = self.borders[idx]
        right_border = self.borders[idx + 1]
        return left_border + (right_border - left_border) * rest_prob / probs.gather(
            -1,
            idx[..., None],
        ).squeeze(-1)

    def quantile(
        self,
        logits: torch.Tensor,
        center_prob: float = 0.682,
    ) -> torch.Tensor:
        side_probs = (1.0 - center_prob) / 2
        return torch.stack(
            (self.icdf(logits, side_probs), self.icdf(logits, 1.0 - side_probs)),
            -1,
        )

    def ucb(
        self,
        logits: torch.Tensor,
        best_f: float,
        rest_prob: float = (1 - 0.682) / 2,
        *,
        maximize: bool = True,
    ) -> torch.Tensor:
        if maximize:
            rest_prob = 1 - rest_prob
        return self.icdf(logits, rest_prob)

    def mode(self, logits: torch.Tensor) -> torch.Tensor:
        density = logits.softmax(-1) / self.bucket_widths
        mode_inds = density.argmax(-1)
        bucket_means = self.borders[:-1] + self.bucket_widths / 2
        return bucket_means[mode_inds]

    def mean_of_square(self, logits: torch.Tensor) -> torch.Tensor:
        left_borders = self.borders[:-1]
        right_borders = self.borders[1:]
        bucket_mean_of_square = (
            left_borders.square() + right_borders.square() + left_borders * right_borders
        ) / 3.0
        p = torch.softmax(logits, -1)
        return p @ bucket_mean_of_square

    def variance(self, logits: torch.Tensor) -> torch.Tensor:
        return self.mean_of_square(logits) - self.mean(logits).square()

    def plot(
        self,
        logits: torch.Tensor,
        ax: plt.Axes | None = None,
        zoom_to_quantile: float | None = None,
        **kwargs: Any,
    ) -> plt.Axes:
        import matplotlib.pyplot as plt  # noqa: PLC0415

        logits = logits.squeeze()
        assert logits.dim() == 1, "logits should be 1d, at least after squeezing."
        if ax is None:
            ax = plt.gca()
        if zoom_to_quantile is not None:
            lower_bounds, upper_bounds = self.quantile(
                logits,
                zoom_to_quantile,
            ).transpose(0, -1)
            lower_bound = lower_bounds.min().item()
            upper_bound = upper_bounds.max().item()
            ax.set_xlim(lower_bound, upper_bound)
            border_mask = (self.borders[:-1] >= lower_bound) & (
                self.borders[1:] <= upper_bound
            )
        else:
            border_mask = slice(None)

        p = torch.softmax(logits, -1) / self.bucket_widths
        ax.bar(
            self.borders[:-1][border_mask],
            p[border_mask],
            self.bucket_widths[border_mask],
            **kwargs,
        )
        return ax


class FullSupportBarDistribution(BarDistribution):
    def __init__(
        self,
        borders: torch.Tensor,
        **kwargs: Any,
    ):
        super().__init__(borders, **kwargs)
        self.assert_support(allow_zero_bucket_left=False)
        losses_per_bucket = torch.zeros_like(self.bucket_widths)
        self.register_buffer("losses_per_bucket", losses_per_bucket)

    def assert_support(self, *, allow_zero_bucket_left: bool = False) -> None:
        if allow_zero_bucket_left:
            assert self.bucket_widths[-1] > 0, (
                f"Half Normal weight must be > 0 (got -1:{self.bucket_widths[-1]})."
            )
            if self.bucket_widths[0] == 0:
                self.borders[0] = self.borders[0] - 1
                self.bucket_widths[0] = 1.0
        else:
            assert self.bucket_widths[0] > 0
        assert self.bucket_widths[-1] > 0

    @staticmethod
    def halfnormal_with_p_weight_before(
        range_max: float,
        p: float = 0.5,
    ) -> torch.distributions.HalfNormal:
        s = range_max / torch.distributions.HalfNormal(torch.tensor(1.0)).icdf(
            torch.tensor(p),
        )
        return torch.distributions.HalfNormal(s)

    @override
    def forward(
        self,
        logits: torch.Tensor,
        y: torch.Tensor,
        mean_prediction_logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert self.num_bars > 1
        y = y.clone().view(*logits.shape[:-1])
        ignore_loss_mask = self.ignore_init(y)
        target_sample = self.map_to_bucket_idx(y)
        target_sample.clamp_(0, self.num_bars - 1)
        assert logits.shape[-1] == self.num_bars, (
            f"{logits.shape[-1]} vs {self.num_bars}"
        )
        assert (target_sample >= 0).all()
        assert (target_sample < self.num_bars).all(), (
            f"y {y} not in support set for borders (min_y, max_y) {self.borders}"
        )
        last_dim = logits.shape[-1]
        assert last_dim == self.num_bars, f"{last_dim} vs {self.num_bars}"

        scaled_bucket_log_probs = self.compute_scaled_log_probs(logits)
        assert len(scaled_bucket_log_probs) == len(target_sample), (
            len(scaled_bucket_log_probs),
            len(target_sample),
        )
        log_probs = scaled_bucket_log_probs.gather(
            -1,
            target_sample.unsqueeze(-1),
        ).squeeze(-1)

        side_normals = (
            self.halfnormal_with_p_weight_before(self.bucket_widths[0]),
            self.halfnormal_with_p_weight_before(self.bucket_widths[-1]),
        )

        log_probs[target_sample == 0] += side_normals[0].log_prob(
            (self.borders[1] - y[target_sample == 0]).clamp(min=0.00000001),
        ) + torch.log(self.bucket_widths[0])
        log_probs[target_sample == self.num_bars - 1] += side_normals[1].log_prob(
            (y[target_sample == self.num_bars - 1] - self.borders[-2]).clamp(
                min=0.00000001,
            ),
        ) + torch.log(self.bucket_widths[-1])

        nll_loss = -log_probs

        if mean_prediction_logits is not None:
            assert not ignore_loss_mask.any(), (
                "Ignoring examples is not implemented with mean pred."
            )
            nll_loss = torch.cat(
                (nll_loss, self.mean_loss(logits, mean_prediction_logits)),
                0,
            )
        if ignore_loss_mask.any():
            nll_loss[ignore_loss_mask] = 0.0
        self.losses_per_bucket += (
            torch.scatter(
                self.losses_per_bucket,
                0,
                target_sample[~ignore_loss_mask].flatten(),
                nll_loss[~ignore_loss_mask].flatten().detach(),
            )
            / target_sample[~ignore_loss_mask].numel()
        )
        return nll_loss

    def pdf(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.exp(-self.forward(logits, y))

    def sample(self, logits: torch.Tensor, t: float = 1.0) -> torch.Tensor:
        p_cdf = torch.rand(*logits.shape[:-1])
        return torch.tensor(
            [self.icdf(logits[i, :] / t, p) for i, p in enumerate(p_cdf.tolist())],
        )

    @override
    def mean(self, logits: torch.Tensor) -> torch.Tensor:
        bucket_means = self.borders[:-1] + self.bucket_widths / 2
        p = torch.softmax(logits, -1)
        side_normals = (
            self.halfnormal_with_p_weight_before(self.bucket_widths[0]),
            self.halfnormal_with_p_weight_before(self.bucket_widths[-1]),
        )
        bucket_means[0] = -side_normals[0].mean + self.borders[1]
        bucket_means[-1] = side_normals[1].mean + self.borders[-2]
        return p @ bucket_means.to(logits.device).type(logits.dtype)

    @override
    def mean_of_square(self, logits: torch.Tensor) -> torch.Tensor:
        left_borders = self.borders[:-1]
        right_borders = self.borders[1:]
        bucket_mean_of_square = (
            left_borders.square() + right_borders.square() + left_borders * right_borders
        ) / 3.0
        side_normals = (
            self.halfnormal_with_p_weight_before(self.bucket_widths[0]),
            self.halfnormal_with_p_weight_before(self.bucket_widths[-1]),
        )
        bucket_mean_of_square[0] = (
            side_normals[0].variance + (-side_normals[0].mean + self.borders[1]).square()
        )
        bucket_mean_of_square[-1] = (
            side_normals[1].variance + (side_normals[1].mean + self.borders[-2]).square()
        )
        p = torch.softmax(logits, -1)
        return p @ bucket_mean_of_square


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
