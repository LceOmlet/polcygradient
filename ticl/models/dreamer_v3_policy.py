import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _dreamer_trunc_normal_in_(tensor: torch.Tensor, *, outscale: float = 1.0) -> torch.Tensor:
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(tensor)
    std = 1.1368 / math.sqrt(max(1, fan_in))
    nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)
    if outscale != 1.0:
        tensor.mul_(float(outscale))
    return tensor


class _DreamerMLPBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.RMSNorm(out_dim)
        self.act = nn.SiLU()
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            _dreamer_trunc_normal_in_(self.linear.weight, outscale=1.0)
            nn.init.zeros_(self.linear.bias)
            nn.init.ones_(self.norm.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.norm(x)
        return self.act(x)


class DreamerV3ContinuousActorHead(nn.Module):
    """PyTorch port of DreamerV3 continuous bounded_normal actor head."""

    def __init__(
        self,
        in_dim: int,
        action_dim: int,
        *,
        layers: int = 3,
        units: int = 1024,
        minstd: float = 0.1,
        maxstd: float = 1.0,
        outscale: float = 0.01,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.action_dim = int(action_dim)
        self.layers = int(layers)
        self.units = int(units)
        self.minstd = float(minstd)
        self.maxstd = float(maxstd)
        self.outscale = float(outscale)

        dims = [self.in_dim] + [self.units] * self.layers
        self.mlp = nn.Sequential(*[
            _DreamerMLPBlock(dims[i], dims[i + 1]) for i in range(self.layers)
        ])
        self.mean = nn.Linear(self.units, self.action_dim)
        self.stddev = nn.Linear(self.units, self.action_dim)
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            _dreamer_trunc_normal_in_(self.mean.weight, outscale=self.outscale)
            _dreamer_trunc_normal_in_(self.stddev.weight, outscale=self.outscale)
            nn.init.zeros_(self.mean.bias)
            nn.init.zeros_(self.stddev.bias)

    def forward(self, hidden: torch.Tensor) -> Dict[str, torch.Tensor]:
        if hidden.shape[-1] != self.in_dim:
            raise ValueError(
                f"DreamerV3ContinuousActorHead expected hidden last-dim {self.in_dim}, got {hidden.shape[-1]}"
            )
        torso = self.mlp(hidden)
        raw_mean = self.mean(torso)
        raw_std = self.stddev(torso)
        action_mean = torch.tanh(raw_mean)
        action_std = (self.maxstd - self.minstd) * torch.sigmoid(raw_std + 2.0) + self.minstd
        action_log_std = torch.log(action_std)
        return {
            "action_mean": action_mean,
            "action_std": action_std,
            "action_log_std": action_log_std,
            "raw_action_mean": raw_mean,
            "raw_action_std": raw_std,
        }

    @staticmethod
    def mode(actor_outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        return actor_outputs["action_mean"]

    @staticmethod
    def sample(actor_outputs: Dict[str, torch.Tensor], *, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        mean = actor_outputs["action_mean"]
        std = actor_outputs["action_std"]
        if noise is None:
            noise = torch.randn_like(mean)
        else:
            noise = noise.to(device=mean.device, dtype=mean.dtype)
        return mean + noise * std

    @staticmethod
    def log_prob(
        actor_outputs: Dict[str, torch.Tensor],
        action: torch.Tensor,
        *,
        mask: Optional[torch.Tensor] = None,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        mean = actor_outputs["action_mean"]
        std = actor_outputs["action_std"].clamp_min(float(max(1e-12, eps)))
        if action.shape != mean.shape:
            raise ValueError(
                f"action and action_mean must have identical shape, got {tuple(action.shape)} and {tuple(mean.shape)}"
            )
        log_std = torch.log(std)
        centered = (action - mean) / std
        per_dim = -0.5 * (centered.square() + 2.0 * log_std + math.log(2.0 * math.pi))
        if mask is not None:
            mask_t = mask.to(device=action.device, dtype=action.dtype)
            while mask_t.ndim < per_dim.ndim:
                mask_t = mask_t.unsqueeze(0)
            per_dim = per_dim * mask_t
        return per_dim.sum(dim=-1)

    @staticmethod
    def score_wrt_mean(
        actor_outputs: Dict[str, torch.Tensor],
        action: torch.Tensor,
        *,
        mask: Optional[torch.Tensor] = None,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        mean = actor_outputs["action_mean"]
        std = actor_outputs["action_std"].clamp_min(float(max(1e-12, eps)))
        score = (action - mean) / std.square()
        if mask is not None:
            mask_t = mask.to(device=action.device, dtype=action.dtype)
            while mask_t.ndim < score.ndim:
                mask_t = mask_t.unsqueeze(0)
            score = score * mask_t
        return score

    @staticmethod
    def decomposition_stats(
        actor_outputs: Dict[str, torch.Tensor],
        action: torch.Tensor,
        *,
        mask: Optional[torch.Tensor] = None,
        eps: float = 1e-6,
    ) -> Dict[str, torch.Tensor]:
        mean = actor_outputs["action_mean"]
        std = actor_outputs["action_std"].clamp_min(float(max(1e-12, eps)))
        centered_sq = ((action - mean) / std).square()
        log_std = torch.log(std)
        if mask is None:
            mask_t = torch.ones_like(action, dtype=torch.bool)
        else:
            mask_t = mask.to(device=action.device, dtype=torch.bool)
            while mask_t.ndim < action.ndim:
                mask_t = mask_t.unsqueeze(0)
            mask_t = mask_t.expand_as(action)
        active_counts = mask_t.to(dtype=torch.float32).sum(dim=-1)
        active_total = int(mask_t.sum().item())

        def _safe_mean(values: torch.Tensor) -> torch.Tensor:
            if active_total <= 0:
                return torch.zeros((), device=action.device, dtype=torch.float32)
            return values.masked_select(mask_t).to(dtype=torch.float32).mean().detach()

        def _safe_min(values: torch.Tensor) -> torch.Tensor:
            if active_total <= 0:
                return torch.zeros((), device=action.device, dtype=torch.float32)
            return values.masked_select(mask_t).to(dtype=torch.float32).min().detach()

        def _safe_max(values: torch.Tensor) -> torch.Tensor:
            if active_total <= 0:
                return torch.zeros((), device=action.device, dtype=torch.float32)
            return values.masked_select(mask_t).to(dtype=torch.float32).max().detach()

        return {
            "reinforce_action_dim_mean": active_counts.mean().detach(),
            "reinforce_action_dim_min": active_counts.min().detach(),
            "reinforce_action_dim_max": active_counts.max().detach(),
            "reinforce_action_std_mean": _safe_mean(std),
            "reinforce_action_std_min": _safe_min(std),
            "reinforce_action_std_max": _safe_max(std),
            "reinforce_logprob_log_std_mean": _safe_mean(log_std),
            "reinforce_logprob_log_std_min": _safe_min(log_std),
            "reinforce_logprob_log_std_max": _safe_max(log_std),
            "reinforce_logprob_z2_mean": _safe_mean(centered_sq),
            "reinforce_logprob_z2_max": _safe_max(centered_sq),
        }
