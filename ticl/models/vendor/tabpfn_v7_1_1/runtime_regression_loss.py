"""Runtime-stable extracts from TabPFN v7.1.1 regression loss code.

Source of truth:
- upstream/src/tabpfn/finetuning/finetuned_regressor.py
"""

from __future__ import annotations

from typing import Any, Literal

import torch


def _compute_regression_loss(  # noqa: C901
    *,
    logits_BQL: torch.Tensor,
    targets_BQ: torch.Tensor,
    bardist_loss_fn: Any,
    ce_loss_weight: float = 1.0,
    crps_loss_weight: float = 0.0,
    crls_loss_weight: float = 0.0,
    mse_loss_weight: float = 0.0,
    mse_loss_clip: float | None = None,
    mae_loss_weight: float = 0.0,
    mae_loss_clip: float | None = None,
) -> torch.Tensor:
    """Official TabPFN regression loss combination."""
    weights_to_validate = {
        "ce_loss_weight": ce_loss_weight,
        "crps_loss_weight": crps_loss_weight,
        "crls_loss_weight": crls_loss_weight,
        "mse_loss_weight": mse_loss_weight,
        "mae_loss_weight": mae_loss_weight,
    }
    for weight_name, weight in weights_to_validate.items():
        if weight < 0.0:
            raise ValueError(f"{weight_name} must be >= 0.0")

    total_loss = torch.tensor(0.0, device=logits_BQL.device)
    valid_mask_BQ = ~torch.isnan(targets_BQ)

    if ce_loss_weight > 0.0:
        ce_losses_BQ = bardist_loss_fn(logits_BQL, targets_BQ)
        total_loss = total_loss + ce_loss_weight * ce_losses_BQ.mean()

    if crps_loss_weight > 0.0:
        crps_loss = _ranked_probability_score_loss_from_bar_logits(
            logits_BQL=logits_BQL,
            targets_BQ=targets_BQ,
            bardist_loss_fn=bardist_loss_fn,
            loss_type="crps",
        )
        total_loss = total_loss + crps_loss_weight * crps_loss

    if crls_loss_weight > 0.0:
        crls_loss = _ranked_probability_score_loss_from_bar_logits(
            logits_BQL=logits_BQL,
            targets_BQ=targets_BQ,
            bardist_loss_fn=bardist_loss_fn,
            loss_type="crls",
        )
        total_loss = total_loss + crls_loss_weight * crls_loss

    if mse_loss_weight > 0.0 or mae_loss_weight > 0.0:
        predictions_mean_BQ = bardist_loss_fn.mean(logits_BQL)
        diffs_BQ = predictions_mean_BQ - targets_BQ

        if mse_loss_weight > 0.0:
            mse_terms_BQ = torch.where(
                valid_mask_BQ, diffs_BQ.square(), torch.zeros_like(diffs_BQ)
            )
            if mse_loss_clip is not None:
                mse_terms_BQ = mse_terms_BQ.clamp(max=mse_loss_clip)
            total_loss = total_loss + mse_loss_weight * mse_terms_BQ.mean()

        if mae_loss_weight > 0.0:
            mae_terms_BQ = torch.where(
                valid_mask_BQ, diffs_BQ.abs(), torch.zeros_like(diffs_BQ)
            )
            if mae_loss_clip is not None:
                mae_terms_BQ = mae_terms_BQ.clamp(max=mae_loss_clip)
            total_loss = total_loss + mae_loss_weight * mae_terms_BQ.mean()

    return total_loss


def _ranked_probability_score_loss_from_bar_logits(
    *,
    logits_BQL: torch.Tensor,
    targets_BQ: torch.Tensor,
    bardist_loss_fn: Any,
    loss_type: Literal["crps", "crls"] = "crps",
) -> torch.Tensor:
    """Official TabPFN ordered-bar score."""
    bucket_widths_L = bardist_loss_fn.bucket_widths.to(logits_BQL.device)
    assert bucket_widths_L.shape == (logits_BQL.shape[-1],), (
        f"bucket_widths_L.shape: {bucket_widths_L.shape} "
        f"logits_BQL.shape: {logits_BQL.shape}"
    )
    probs_BQL = torch.softmax(logits_BQL, dim=-1)
    pred_cdf_BQL = torch.cumsum(probs_BQL, dim=-1)

    ignore_loss_mask_BQ = torch.isnan(targets_BQ)
    filled_targets_BQ = torch.where(
        ignore_loss_mask_BQ, torch.zeros_like(targets_BQ), targets_BQ
    )

    target_bins_BQ = bardist_loss_fn.map_to_bucket_idx(filled_targets_BQ).clamp(
        0, bardist_loss_fn.num_bars - 1
    )
    bin_indices_L = torch.arange(probs_BQL.shape[-1], device=probs_BQL.device)
    target_cdf_BQL = (bin_indices_L.view(1, 1, -1) >= target_bins_BQ.unsqueeze(-1)).to(
        probs_BQL.dtype
    )

    if loss_type == "crps":
        cdf_diff_BQL = pred_cdf_BQL - target_cdf_BQL
        cdf_term_losses_BQL = cdf_diff_BQL.square()
    else:
        eps = torch.finfo(pred_cdf_BQL.dtype).eps
        cdf = pred_cdf_BQL.clamp(eps, 1 - eps)
        cdf_term_losses_BQL = target_cdf_BQL * (-torch.log(cdf)) + (
            1 - target_cdf_BQL
        ) * (-torch.log1p(-cdf))

    weighted_term_losses_BQL = cdf_term_losses_BQL * bucket_widths_L.view(1, 1, -1)
    crps_losses_BQ = weighted_term_losses_BQL.sum(dim=-1)

    if ignore_loss_mask_BQ.any():
        crps_losses_BQ[ignore_loss_mask_BQ] = 0.0

    return crps_losses_BQ.mean()
