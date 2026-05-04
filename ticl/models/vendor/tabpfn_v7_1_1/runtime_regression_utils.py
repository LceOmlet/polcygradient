"""Runtime-stable extracts from TabPFN v7.1.1 regression utility code.

Source of truth:
- upstream/src/tabpfn/utils.py
- upstream/src/tabpfn/constants.py
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch
from sklearn.base import TransformerMixin

REGRESSION_NAN_BORDER_LIMIT_UPPER = 1e3
REGRESSION_NAN_BORDER_LIMIT_LOWER = -1e3


def _repair_borders(borders: np.ndarray, *, inplace: bool) -> None:
    if inplace is not True:
        raise NotImplementedError("Only inplace is supported")

    if np.isnan(borders[-1]):
        nans = np.isnan(borders)
        largest = borders[~nans].max()
        borders[nans] = largest
        borders[-1] = borders[-1] * 2

    if borders[-1] - borders[-2] < 1e-6:
        borders[-1] = borders[-1] * 1.1

    if borders[0] == borders[1]:
        borders[0] -= np.abs(borders[0] * 0.1)


def _cancel_nan_borders(
    *,
    borders: np.ndarray,
    broken_mask: npt.NDArray[np.bool_],
) -> tuple[np.ndarray, npt.NDArray[np.bool_]]:
    borders = borders.copy()
    num_right_borders = (broken_mask[:-1] > broken_mask[1:]).sum()
    num_left_borders = (broken_mask[1:] > broken_mask[:-1]).sum()
    assert num_left_borders <= 1
    assert num_right_borders <= 1

    if num_right_borders:
        assert bool(broken_mask[0]) is True
        rightmost_nan_of_left = np.where(broken_mask[:-1] > broken_mask[1:])[0][0] + 1
        borders[:rightmost_nan_of_left] = borders[rightmost_nan_of_left]
        borders[0] = borders[1] - 1.0

    if num_left_borders:
        assert bool(broken_mask[-1]) is True
        leftmost_nan_of_right = np.where(broken_mask[1:] > broken_mask[:-1])[0][0]
        borders[leftmost_nan_of_right + 1 :] = borders[leftmost_nan_of_right]
        borders[-1] = borders[-2] + 1.0

    logit_cancel_mask = broken_mask[1:] | broken_mask[:-1]
    return borders, logit_cancel_mask


def _map_to_bucket_ix(y: torch.Tensor, borders: torch.Tensor) -> torch.Tensor:
    ix = torch.searchsorted(sorted_sequence=borders, input=y) - 1
    ix[y == borders[0]] = 0
    ix[y == borders[-1]] = len(borders) - 2
    return ix


def _cdf(logits: torch.Tensor, borders: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
    ys = ys.repeat((*logits.shape[:-1], 1))
    n_bars = len(borders) - 1
    y_buckets = _map_to_bucket_ix(ys, borders).clamp(0, n_bars - 1).to(logits.device)

    probs = torch.softmax(logits, dim=-1)
    prob_so_far = torch.cumsum(probs, dim=-1) - probs
    prob_left_of_bucket = prob_so_far.gather(index=y_buckets, dim=-1)

    bucket_widths = borders[1:] - borders[:-1]
    share_of_bucket_left = (ys - borders[y_buckets]) / bucket_widths[y_buckets]
    share_of_bucket_left = share_of_bucket_left.clamp(0.0, 1.0)

    prob_in_bucket = probs.gather(index=y_buckets, dim=-1) * share_of_bucket_left
    prob_left_of_ys = prob_left_of_bucket + prob_in_bucket

    prob_left_of_ys[ys <= borders[0]] = 0.0
    prob_left_of_ys[ys >= borders[-1]] = 1.0
    return prob_left_of_ys.clip(0.0, 1.0)


def translate_probs_across_borders(
    logits: torch.Tensor,
    *,
    frm: torch.Tensor,
    to: torch.Tensor,
) -> torch.Tensor:
    prob_left = _cdf(logits, borders=frm, ys=to)
    prob_left[..., 0] = 0.0
    prob_left[..., -1] = 1.0
    return (prob_left[..., 1:] - prob_left[..., :-1]).clamp_min(0.0)


def transform_borders_one(
    borders: np.ndarray,
    target_transform: TransformerMixin,
    *,
    repair_nan_borders_after_transform: bool,
) -> tuple[npt.NDArray[np.bool_] | None, bool, np.ndarray]:
    borders_t = target_transform.inverse_transform(borders.reshape(-1, 1)).squeeze()  # type: ignore[arg-type]

    logit_cancel_mask: npt.NDArray[np.bool_] | None = None
    if repair_nan_borders_after_transform:
        broken_mask = (
            ~np.isfinite(borders_t)
            | (borders_t > REGRESSION_NAN_BORDER_LIMIT_UPPER)
            | (borders_t < REGRESSION_NAN_BORDER_LIMIT_LOWER)
        )
        if broken_mask.any():
            borders_t, logit_cancel_mask = _cancel_nan_borders(
                borders=borders_t,
                broken_mask=broken_mask,
            )

    _repair_borders(borders_t, inplace=True)

    reversed_order = np.arange(len(borders_t) - 1, -1, -1)
    descending_borders = (np.argsort(borders_t) == reversed_order).all()
    if descending_borders:
        borders_t = borders_t[::-1]
        logit_cancel_mask = (
            logit_cancel_mask[::-1] if logit_cancel_mask is not None else None
        )

    return logit_cancel_mask, descending_borders, borders_t
