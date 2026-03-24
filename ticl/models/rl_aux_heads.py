from typing import Callable, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_two_layer_mlp_head(in_dim: int, hidden_dim: int, out_dim: int):
    return nn.Sequential(
        nn.Linear(int(in_dim), int(hidden_dim)),
        nn.GELU(),
        nn.Linear(int(hidden_dim), int(out_dim)),
    )


def _resolve_activation(activation: Union[Callable, str]):
    if callable(activation):
        return activation
    if str(activation).lower() == "relu":
        return F.relu
    if str(activation).lower() == "gelu":
        return F.gelu
    if str(activation).lower() == "silu":
        return F.silu
    raise ValueError(f"Unsupported activation: {activation}")


def _prepare_flow_inputs(hidden: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor, *, target_dim: int):
    if tuple(hidden.shape[:-1]) != tuple(x_t.shape[:-1]):
        raise ValueError(
            "hidden and x_t must share leading dims, "
            f"got {tuple(hidden.shape)} and {tuple(x_t.shape)}"
        )
    if int(x_t.shape[-1]) != int(target_dim):
        raise ValueError(
            f"x_t last dim must be {int(target_dim)}, got {tuple(x_t.shape)}"
        )
    if t.ndim == hidden.ndim - 1:
        t = t.unsqueeze(-1)
    if t.ndim != hidden.ndim or int(t.shape[-1]) != 1:
        raise ValueError(
            "t must broadcast as (..., 1), "
            f"got {tuple(t.shape)} for hidden {tuple(hidden.shape)}"
        )
    if tuple(t.shape[:-1]) != tuple(hidden.shape[:-1]):
        t = t.expand(*hidden.shape[:-1], 1)
    leading_shape = tuple(hidden.shape[:-1])
    hidden_flat = hidden.reshape(-1, int(hidden.shape[-1]))
    x_t_flat = x_t.reshape(-1, int(x_t.shape[-1]))
    t_flat = t.reshape(-1, 1)
    return hidden_flat, x_t_flat, t_flat, leading_shape


class TimeEmbeddingNet(nn.Module):
    """CFMI-style sinusoidal time embedding with two Linear+SiLU projections."""

    def __init__(self, embedding_dim: int, projection_dim: int = None, frequency_multiplier: int = None):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        if self.embedding_dim % 2 != 0:
            raise ValueError(f"embedding_dim must be even, got {self.embedding_dim}")
        self.projection_dim = int(self.embedding_dim if projection_dim is None else projection_dim)
        self.frequency_multiplier = frequency_multiplier
        self.projection1 = nn.Linear(self.embedding_dim, self.projection_dim)
        self.projection2 = nn.Linear(self.projection_dim, self.projection_dim)

    def embed_time(self, t: torch.Tensor):
        half_dim = int(self.embedding_dim // 2)
        denom = max(1, half_dim - 1)
        frequencies = (10.0 ** (torch.arange(half_dim, device=t.device, dtype=t.dtype) / denom * 4.0)).unsqueeze(0)
        table = t * frequencies
        if self.frequency_multiplier is not None:
            table = table * float(self.frequency_multiplier - 1)
        return torch.cat([torch.sin(table), torch.cos(table)], dim=-1)

    def forward(self, t: torch.Tensor):
        x = self.embed_time(t)
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x


class ResidualBlock(nn.Module):
    """CFMI residual block for vector-field decoding."""

    def __init__(
        self,
        features: int,
        *,
        activation: Union[Callable, str] = "relu",
        dropout_probability: float = 0.0,
        use_layer_norm: bool = False,
    ):
        super().__init__()
        self.features = int(features)
        self.activation = _resolve_activation(activation)
        self.linear_layers = nn.ModuleList([nn.Linear(self.features, self.features) for _ in range(2)])
        self.dropout = nn.Dropout(p=float(dropout_probability))
        self.layer_norm = nn.LayerNorm(self.features) if bool(use_layer_norm) else None

    def forward(self, inputs: torch.Tensor):
        out = self.linear_layers[0](inputs)
        if self.layer_norm is not None:
            out = self.layer_norm(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.linear_layers[1](out)
        return inputs + out


class ResidualFCNetwork(nn.Module):
    """CFMI residual MLP used for tabular conditional flow matching."""

    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        num_residual_blocks: int,
        residual_block_dim: int,
        activation: Union[Callable, str] = "relu",
        dropout_probability: float = 0.0,
        use_layer_norm: bool = False,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.residual_block_dim = int(residual_block_dim)
        self.activation = _resolve_activation(activation)
        self.initial_layer = nn.Linear(self.input_dim, self.residual_block_dim)
        self.blocks = nn.ModuleList(
            [
                ResidualBlock(
                    self.residual_block_dim,
                    activation=self.activation,
                    dropout_probability=dropout_probability,
                    use_layer_norm=use_layer_norm,
                )
                for _ in range(int(num_residual_blocks))
            ]
        )
        self.final_layer = nn.Linear(self.residual_block_dim, self.output_dim)

    def forward(self, inputs: torch.Tensor):
        out = self.initial_layer(inputs)
        out = self.activation(out)
        for block in self.blocks:
            out = block(out)
        out = self.activation(out)
        return self.final_layer(out)


class ConditionalFlowMatchingHead(nn.Module):
    """Two-layer conditional velocity head for official affine/CondOT flow matching."""

    def __init__(self, *, hidden_dim: int, target_dim: int, mlp_hidden_dim: int):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.target_dim = int(target_dim)
        self.net = build_two_layer_mlp_head(
            self.hidden_dim + self.target_dim + 1,
            int(mlp_hidden_dim),
            self.target_dim,
        )

    def forward(self, hidden: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor):
        hidden_flat, x_t_flat, t_flat, leading_shape = _prepare_flow_inputs(
            hidden,
            x_t,
            t,
            target_dim=self.target_dim,
        )
        head_param = next(self.net.parameters())
        inputs = torch.cat(
            [
                hidden_flat.to(dtype=head_param.dtype),
                x_t_flat.to(device=hidden_flat.device, dtype=head_param.dtype),
                t_flat.to(device=hidden_flat.device, dtype=head_param.dtype),
            ],
            dim=-1,
        )
        out = self.net(inputs)
        return out.reshape(*leading_shape, self.target_dim)


class CFMIResidualFlowMatchingHead(nn.Module):
    """CFMI-style residual MLP head with 4 residual blocks of width 256."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        target_dim: int,
        time_embedding_dim: int = 128,
        num_residual_blocks: int = 4,
        residual_block_dim: int = 256,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.target_dim = int(target_dim)
        self.time_embedding_dim = int(time_embedding_dim)
        self.time_embedding_net = TimeEmbeddingNet(
            embedding_dim=self.time_embedding_dim,
            projection_dim=self.time_embedding_dim,
        )
        self.net = ResidualFCNetwork(
            input_dim=self.time_embedding_dim + self.target_dim + self.hidden_dim,
            output_dim=self.target_dim,
            num_residual_blocks=int(num_residual_blocks),
            residual_block_dim=int(residual_block_dim),
            activation="relu",
            dropout_probability=0.0,
            use_layer_norm=True,
        )

    def forward(self, hidden: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor):
        hidden_flat, x_t_flat, t_flat, leading_shape = _prepare_flow_inputs(
            hidden,
            x_t,
            t,
            target_dim=self.target_dim,
        )
        head_param = next(self.net.parameters())
        hidden_in = hidden_flat.to(dtype=head_param.dtype)
        x_t_in = x_t_flat.to(device=hidden_in.device, dtype=head_param.dtype)
        t_in = t_flat.to(device=hidden_in.device, dtype=head_param.dtype)
        t_emb = self.time_embedding_net(t_in)
        out = self.net(torch.cat([t_emb, x_t_in, hidden_in], dim=-1))
        return out.reshape(*leading_shape, self.target_dim)
