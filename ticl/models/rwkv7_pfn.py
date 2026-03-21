import math
import os
import importlib.util
import sys
import types
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import wandb

from ticl.models.encoders import Linear, SplitObsActionEncoder
from ticl.utils import SeqBN


def _is_torch_compiling():
    compiler_mod = getattr(torch, "compiler", None)
    if compiler_mod is not None and hasattr(compiler_mod, "is_compiling"):
        try:
            if bool(compiler_mod.is_compiling()):
                return True
        except Exception:
            pass
    dynamo_mod = getattr(torch, "_dynamo", None)
    if dynamo_mod is not None and hasattr(dynamo_mod, "is_compiling"):
        try:
            return bool(dynamo_mod.is_compiling())
        except Exception:
            return False
    return False


_SPLIT_ENCODE_FUSION_ENV = str(os.environ.get("TICL_POLICY_SPLIT_ENCODE_FUSION", "1")).strip().lower()
_SPLIT_ENCODE_FUSION_ENABLED = _SPLIT_ENCODE_FUSION_ENV not in {"0", "false", "no", "off"}


def _group_norm_last_dim(x: torch.Tensor, *, num_groups: int, weight: torch.Tensor, bias: torch.Tensor, eps: float):
    flat = x.reshape(-1, x.shape[-1])
    normed = F.group_norm(flat, num_groups=num_groups, weight=weight, bias=bias, eps=eps)
    return normed.reshape_as(x)


def _default_mlp_head(in_dim: int, hidden_dim: int, out_dim: int):
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, out_dim),
    )


_OFFICIAL_RWKV7_DEMO_RNN = None
_OFFICIAL_RWKV7_TRAIN_TEMP = {}


def _load_official_rwkv7_demo_rnn():
    global _OFFICIAL_RWKV7_DEMO_RNN
    if _OFFICIAL_RWKV7_DEMO_RNN is not None:
        return _OFFICIAL_RWKV7_DEMO_RNN
    module_path = Path(__file__).resolve().parent / "vendor" / "rwkv_lm_official" / "RWKV-v7" / "rwkv_v7_demo_rnn.py"
    source = module_path.read_text(encoding="utf-8")
    cutoff_marker = "@MyStatic\ndef sample_logits"
    cutoff = source.find(cutoff_marker)
    if cutoff == -1:
        raise ImportError(f"Unable to isolate reusable portion of official RWKV-7 demo_rnn module at {module_path}")
    reusable_source = source[:cutoff]
    spec = importlib.util.spec_from_file_location("ticl_vendor_rwkv7_demo_rnn", str(module_path))
    if spec is None:
        raise ImportError(f"Unable to construct spec for official RWKV-7 demo_rnn module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    exec(compile(reusable_source, str(module_path), "exec"), module.__dict__)
    _OFFICIAL_RWKV7_DEMO_RNN = module
    return module


@contextmanager
def _pushd(path: Path):
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def _load_official_rwkv7_train_temp(head_size: int):
    head_size = int(head_size)
    if head_size != 64:
        raise ValueError(
            f"Official RWKV-7 train_temp CUDA path currently requires rwkv_head_size=64, got {head_size}."
        )
    cached = _OFFICIAL_RWKV7_TRAIN_TEMP.get(head_size)
    if cached is not None:
        return cached
    module_path = Path(__file__).resolve().parent / "vendor" / "rwkv_lm_official" / "RWKV-v7" / "train_temp" / "src" / "model.py"
    spec = importlib.util.spec_from_file_location(f"ticl_vendor_rwkv7_train_temp_h{head_size}", str(module_path))
    if spec is None:
        raise ImportError(f"Unable to construct spec for official RWKV-7 train_temp module from {module_path}")
    module = importlib.util.module_from_spec(spec)

    prev_env = {
        "RWKV_JIT_ON": os.environ.get("RWKV_JIT_ON"),
        "RWKV_MY_TESTING": os.environ.get("RWKV_MY_TESTING"),
        "RWKV_HEAD_SIZE": os.environ.get("RWKV_HEAD_SIZE"),
    }
    os.environ["RWKV_JIT_ON"] = "0"
    os.environ["RWKV_MY_TESTING"] = "x070"
    os.environ["RWKV_HEAD_SIZE"] = str(head_size)

    injected = {}
    if "pytorch_lightning" not in sys.modules:
        pl_mod = types.ModuleType("pytorch_lightning")
        pl_mod.LightningModule = nn.Module
        injected["pytorch_lightning"] = pl_mod
    if "pytorch_lightning.utilities" not in sys.modules:
        utilities_mod = types.ModuleType("pytorch_lightning.utilities")
        utilities_mod.rank_zero_info = lambda *args, **kwargs: None
        utilities_mod.rank_zero_only = lambda fn: fn
        injected["pytorch_lightning.utilities"] = utilities_mod
    if "pytorch_lightning.strategies" not in sys.modules:
        strategies_mod = types.ModuleType("pytorch_lightning.strategies")
        strategies_mod.DeepSpeedStrategy = object
        injected["pytorch_lightning.strategies"] = strategies_mod

    old_modules = {}
    for name, mod in injected.items():
        old_modules[name] = sys.modules.get(name)
        sys.modules[name] = mod
    try:
        with _pushd(module_path.parent.parent):
            spec.loader.exec_module(module)
    finally:
        for key, value in prev_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        for name in injected:
            previous = old_modules[name]
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous

    if not hasattr(module, "RWKV7_CLAMPW_CUDA"):
        raise RuntimeError(
            "Official RWKV-7 train_temp module did not expose RWKV7_CLAMPW_CUDA; "
            "refusing to fall back."
        )
    _OFFICIAL_RWKV7_TRAIN_TEMP[head_size] = module
    return module


def _build_official_rwkv7_args(*, emb_dim: int, nlayers: int, head_size: int):
    return types.SimpleNamespace(
        n_embd=int(emb_dim),
        n_layer=max(2, int(nlayers)),
        dim_att=int(emb_dim),
        head_size=int(head_size),
        my_testing="",
    )


_OFFICIAL_RWKV7_SCRIPT_API = _load_official_rwkv7_demo_rnn()
_OFFICIAL_RWKV7_TIME_MIXING = _OFFICIAL_RWKV7_SCRIPT_API.time_mixing
_OFFICIAL_RWKV7_CHANNEL_MIXING = _OFFICIAL_RWKV7_SCRIPT_API.channel_mixing


class OfficialRWKV7EvalCore(torch.jit.ScriptModule):
    z: Dict[str, torch.Tensor]
    n_embd: int
    n_layer: int
    n_head: int
    head_size: int
    has_ln0: bool

    def __init__(
        self,
        *,
        z: Dict[str, torch.Tensor],
        n_embd: int,
        n_layer: int,
        n_head: int,
        head_size: int,
        has_ln0: bool,
    ):
        super().__init__()
        self.z = torch.jit.Attribute(z, Dict[str, torch.Tensor])
        self.n_embd = int(n_embd)
        self.n_layer = int(n_layer)
        self.n_head = int(n_head)
        self.head_size = int(head_size)
        self.has_ln0 = bool(has_ln0)
        self.eval()

    @torch.jit.script_method
    def forward(self, x: torch.Tensor, state: List[torch.Tensor]):
        with torch.no_grad():
            z = self.z
            if self.has_ln0:
                x = F.layer_norm(
                    x,
                    (self.n_embd,),
                    weight=z["blocks.0.ln0.weight"],
                    bias=z["blocks.0.ln0.bias"],
                )

            v_first = torch.empty_like(x)
            for i in range(self.n_layer):
                bbb = "blocks." + str(i) + "."
                att = bbb + "att."
                ffn = bbb + "ffn."

                xx = F.layer_norm(
                    x,
                    (self.n_embd,),
                    weight=z[bbb + "ln1.weight"],
                    bias=z[bbb + "ln1.bias"],
                )

                xx, state[i * 3 + 0], state[i * 3 + 1], v_first = _OFFICIAL_RWKV7_TIME_MIXING(
                    i,
                    self.n_head,
                    self.head_size,
                    xx,
                    state[i * 3 + 0],
                    v_first,
                    state[i * 3 + 1],
                    z[att + "x_r"],
                    z[att + "x_w"],
                    z[att + "x_k"],
                    z[att + "x_v"],
                    z[att + "x_a"],
                    z[att + "x_g"],
                    z[att + "w0"],
                    z[att + "w1"],
                    z[att + "w2"],
                    z[att + "a0"],
                    z[att + "a1"],
                    z[att + "a2"],
                    z[att + "v0"],
                    z[att + "v1"],
                    z[att + "v2"],
                    z[att + "g1"],
                    z[att + "g2"],
                    z[att + "k_k"],
                    z[att + "k_a"],
                    z[att + "r_k"],
                    z[att + "key.weight"],
                    z[att + "value.weight"],
                    z[att + "receptance.weight"],
                    z[att + "output.weight"],
                    z[att + "ln_x.weight"],
                    z[att + "ln_x.bias"],
                )
                x = x + xx

                xx = F.layer_norm(
                    x,
                    (self.n_embd,),
                    weight=z[bbb + "ln2.weight"],
                    bias=z[bbb + "ln2.bias"],
                )
                xx, state[i * 3 + 2] = _OFFICIAL_RWKV7_CHANNEL_MIXING(
                    xx,
                    state[i * 3 + 2],
                    z[ffn + "x_k"],
                    z[ffn + "key.weight"],
                    z[ffn + "value.weight"],
                )
                x = x + xx

            x = F.layer_norm(
                x,
                (self.n_embd,),
                weight=z["ln_out.weight"],
                bias=z["ln_out.bias"],
            )
            return x, state


class OfficialRWKV7Block(nn.Module):
    def __init__(self, *, emb_dim: int, layer_id: int, num_layers: int, head_size: int):
        super().__init__()
        official_train = _load_official_rwkv7_train_temp(head_size)
        args = _build_official_rwkv7_args(emb_dim=emb_dim, nlayers=num_layers, head_size=head_size)
        block = official_train.Block(args, int(layer_id))
        self.layer_id = int(layer_id)
        self.emb_dim = int(emb_dim)
        self.head_size = int(head_size)
        self.n_head = self.emb_dim // self.head_size
        if hasattr(block, "ln0"):
            self.ln0 = block.ln0
        self.ln1 = block.ln1
        self.ln2 = block.ln2
        self.att = block.att
        self.ffn = block.ffn

    def init_state(self, batch_size: int, *, device: torch.device, dtype: torch.dtype):
        att_x_prev = torch.zeros((batch_size, self.emb_dim), device=device, dtype=dtype)
        att_kv = torch.zeros(
            (batch_size, self.n_head, self.head_size, self.head_size),
            device=device,
            dtype=torch.float32,
        )
        ffn_x_prev = torch.zeros((batch_size, self.emb_dim), device=device, dtype=dtype)
        return att_x_prev, att_kv, ffn_x_prev

    def _forward_step_batched_no_grad(
        self,
        x: torch.Tensor,
        state,
        v_first: Optional[torch.Tensor],
    ):
        if hasattr(self, "ln0"):
            x = self.ln0(x)
        if v_first is None:
            v_first = torch.zeros_like(x)
        att_x_prev, att_kv, ffn_x_prev = state

        att_in = self.ln1(x)
        xx = att_x_prev - att_in

        x_r = self.att.x_r.squeeze(0).squeeze(0)
        x_w = self.att.x_w.squeeze(0).squeeze(0)
        x_k = self.att.x_k.squeeze(0).squeeze(0)
        x_v = self.att.x_v.squeeze(0).squeeze(0)
        x_a = self.att.x_a.squeeze(0).squeeze(0)
        x_g = self.att.x_g.squeeze(0).squeeze(0)
        w0 = self.att.w0.squeeze(0).squeeze(0)
        a0 = self.att.a0.squeeze(0).squeeze(0)
        v0 = self.att.v0.squeeze(0).squeeze(0)
        k_k = self.att.k_k.squeeze(0).squeeze(0)
        k_a = self.att.k_a.squeeze(0).squeeze(0)
        r_k = self.att.r_k.reshape(1, self.n_head, self.head_size)

        xr = att_in + xx * x_r
        xw = att_in + xx * x_w
        xk = att_in + xx * x_k
        xv = att_in + xx * x_v
        xa = att_in + xx * x_a
        xg = att_in + xx * x_g

        r = F.linear(xr, self.att.receptance.weight)
        w = torch.tanh(xw @ self.att.w1) @ self.att.w2
        k = F.linear(xk, self.att.key.weight)
        v = F.linear(xv, self.att.value.weight)
        a = torch.sigmoid(a0 + (xa @ self.att.a1) @ self.att.a2)
        g = torch.sigmoid(xg @ self.att.g1) @ self.att.g2

        kk = k * k_k
        kk = F.normalize(kk.view(-1, self.n_head, self.head_size), dim=-1, p=2.0)
        k = k.view(-1, self.n_head, self.head_size) * (
            1 + (a.view(-1, self.n_head, self.head_size) - 1) * k_a.view(1, self.n_head, self.head_size)
        )
        v_heads = v.view(-1, self.n_head, self.head_size)

        if self.layer_id == 0:
            v_first_next = v
        else:
            v = v + (v_first - v) * torch.sigmoid(v0 + (xv @ self.att.v1) @ self.att.v2)
            v_heads = v.view(-1, self.n_head, self.head_size)
            v_first_next = v_first

        w = w0 + w.float()
        w = torch.exp(-0.606531 * torch.sigmoid(w)).view(-1, self.n_head, self.head_size)

        vk = v_heads.unsqueeze(-1) @ k.unsqueeze(-2)
        ab = (-kk).unsqueeze(-1) @ (kk * a.view(-1, self.n_head, self.head_size)).unsqueeze(-2)
        att_kv_next = att_kv * w.unsqueeze(-2) + (att_kv @ ab.float()) + vk.float()
        out = (att_kv_next.to(dtype=att_in.dtype) @ r.view(-1, self.n_head, self.head_size, 1)).view(-1, self.emb_dim)
        out = F.group_norm(
            out,
            num_groups=self.n_head,
            weight=self.att.ln_x.weight,
            bias=self.att.ln_x.bias,
            eps=64e-5,
        )
        out = out + ((r.view(-1, self.n_head, self.head_size) * k * r_k).sum(dim=-1, keepdim=True) * v_heads).view(
            -1, self.emb_dim
        )
        att_out = F.linear(out * g, self.att.output.weight)
        x = x + att_out

        ffn_in = self.ln2(x)
        ffn_xx = ffn_x_prev - ffn_in
        ffn_k = ffn_in + ffn_xx * self.ffn.x_k.squeeze(0).squeeze(0)
        ffn_k = torch.relu(F.linear(ffn_k, self.ffn.key.weight)) ** 2
        ffn_out = F.linear(ffn_k, self.ffn.value.weight)
        x = x + ffn_out
        return x, (att_in, att_kv_next, ffn_in), v_first_next

    def forward_step(self, x: torch.Tensor, state, v_first: Optional[torch.Tensor]):
        official = _load_official_rwkv7_demo_rnn()
        batch_size = int(x.shape[0])
        if batch_size > 1 and (not torch.is_grad_enabled()):
            return self._forward_step_batched_no_grad(x, state, v_first)
        if hasattr(self, "ln0"):
            x = self.ln0(x)
        if v_first is None:
            v_first = torch.zeros_like(x)
        att_x_prev, att_kv, ffn_x_prev = state

        def _single_att_step(x_i, x_prev_i, kv_state_i, v_first_i):
            return official.time_mixing__(
                int(self.layer_id),
                int(self.n_head),
                int(self.head_size),
                x_i,
                x_prev_i,
                v_first_i,
                kv_state_i,
                self.att.x_r.squeeze(0).squeeze(0),
                self.att.x_w.squeeze(0).squeeze(0),
                self.att.x_k.squeeze(0).squeeze(0),
                self.att.x_v.squeeze(0).squeeze(0),
                self.att.x_a.squeeze(0).squeeze(0),
                self.att.x_g.squeeze(0).squeeze(0),
                self.att.w0.squeeze(0).squeeze(0),
                self.att.w1,
                self.att.w2,
                self.att.a0.squeeze(0).squeeze(0),
                self.att.a1,
                self.att.a2,
                self.att.v0.squeeze(0).squeeze(0),
                self.att.v1,
                self.att.v2,
                self.att.g1,
                self.att.g2,
                self.att.k_k.squeeze(0).squeeze(0),
                self.att.k_a.squeeze(0).squeeze(0),
                self.att.r_k.reshape(-1),
                self.att.key.weight,
                self.att.value.weight,
                self.att.receptance.weight,
                self.att.output.weight,
                self.att.ln_x.weight,
                self.att.ln_x.bias,
            )

        def _single_ffn_step(x_i, x_prev_i):
            return official.channel_mixing__(
                x_i,
                x_prev_i,
                self.ffn.x_k.squeeze(0).squeeze(0),
                self.ffn.key.weight,
                self.ffn.value.weight,
            )

        att_in = self.ln1(x)
        if batch_size == 1:
            att_out_i, att_x_prev_next_i, att_kv_next_i, v_first_next_i = official.time_mixing__(
                int(self.layer_id),
                int(self.n_head),
                int(self.head_size),
                att_in[0],
                att_x_prev[0],
                v_first[0],
                att_kv[0],
                self.att.x_r.squeeze(0).squeeze(0),
                self.att.x_w.squeeze(0).squeeze(0),
                self.att.x_k.squeeze(0).squeeze(0),
                self.att.x_v.squeeze(0).squeeze(0),
                self.att.x_a.squeeze(0).squeeze(0),
                self.att.x_g.squeeze(0).squeeze(0),
                self.att.w0.squeeze(0).squeeze(0),
                self.att.w1,
                self.att.w2,
                self.att.a0.squeeze(0).squeeze(0),
                self.att.a1,
                self.att.a2,
                self.att.v0.squeeze(0).squeeze(0),
                self.att.v1,
                self.att.v2,
                self.att.g1,
                self.att.g2,
                self.att.k_k.squeeze(0).squeeze(0),
                self.att.k_a.squeeze(0).squeeze(0),
                self.att.r_k.reshape(-1),
                self.att.key.weight,
                self.att.value.weight,
                self.att.receptance.weight,
                self.att.output.weight,
                self.att.ln_x.weight,
                self.att.ln_x.bias,
            )
            att_out = att_out_i.unsqueeze(0)
            att_x_prev_next = att_x_prev_next_i.unsqueeze(0)
            att_kv_next = att_kv_next_i.unsqueeze(0)
            v_first_next = v_first_next_i.unsqueeze(0)
        else:
            att_out, att_x_prev_next, att_kv_next, v_first_next = torch.vmap(
                _single_att_step, in_dims=(0, 0, 0, 0), out_dims=(0, 0, 0, 0)
            )(att_in, att_x_prev, att_kv, v_first)
        x = x + att_out

        ffn_in = self.ln2(x)
        if batch_size == 1:
            ffn_out_i, ffn_x_prev_next_i = official.channel_mixing__(
                ffn_in[0],
                ffn_x_prev[0],
                self.ffn.x_k.squeeze(0).squeeze(0),
                self.ffn.key.weight,
                self.ffn.value.weight,
            )
            ffn_out = ffn_out_i.unsqueeze(0)
            ffn_x_prev_next = ffn_x_prev_next_i.unsqueeze(0)
        else:
            ffn_out, ffn_x_prev_next = torch.vmap(_single_ffn_step, in_dims=(0, 0), out_dims=(0, 0))(ffn_in, ffn_x_prev)
        x = x + ffn_out
        return x, (att_x_prev_next, att_kv_next, ffn_x_prev_next), v_first_next

    def forward_sequence(self, x: torch.Tensor, v_first: Optional[torch.Tensor]):
        official_train = _load_official_rwkv7_train_temp(self.head_size)
        if hasattr(self, "ln0"):
            x = self.ln0(x)
        x_att = self.ln1(x)
        bsz, seq_len, channels = x_att.shape
        xx = self.att.time_shift(x_att) - x_att

        xr = x_att + xx * self.att.x_r
        xw = x_att + xx * self.att.x_w
        xk = x_att + xx * self.att.x_k
        xv = x_att + xx * self.att.x_v
        xa = x_att + xx * self.att.x_a
        xg = x_att + xx * self.att.x_g

        r = self.att.receptance(xr)
        w = self.att.w0 + torch.tanh(xw @ self.att.w1) @ self.att.w2
        k = self.att.key(xk)
        v = self.att.value(xv)
        if self.layer_id == 0:
            v_first = v
        else:
            v = v + (v_first - v) * torch.sigmoid(self.att.v0 + (xv @ self.att.v1) @ self.att.v2)
        a = torch.sigmoid(self.att.a0 + (xa @ self.att.a1) @ self.att.a2)
        g = torch.sigmoid(xg @ self.att.g1) @ self.att.g2

        kk = k * self.att.k_k
        kk = F.normalize(kk.view(bsz, seq_len, self.n_head, -1), dim=-1, p=2.0).view(bsz, seq_len, channels)
        k = k * (1 + (a - 1) * self.att.k_a)

        att_out = official_train.RWKV7_CLAMPW_CUDA(
            r.to(dtype=torch.bfloat16),
            w.to(dtype=torch.bfloat16),
            k.to(dtype=torch.bfloat16),
            v.to(dtype=torch.bfloat16),
            (-kk).to(dtype=torch.bfloat16),
            (kk * a).to(dtype=torch.bfloat16),
        )
        att_out = self.att.ln_x(att_out.view(bsz * seq_len, channels)).view(bsz, seq_len, channels)
        att_out = att_out + (
            (r.view(bsz, seq_len, self.n_head, -1) * k.view(bsz, seq_len, self.n_head, -1) * self.att.r_k)
            .sum(dim=-1, keepdim=True)
            * v.view(bsz, seq_len, self.n_head, -1)
        ).view(bsz, seq_len, channels)
        att_out = self.att.output(att_out * g)
        x = x + att_out
        x = x + self.ffn(self.ln2(x))
        return x, v_first


class RWKV7Core(nn.Module):
    def __init__(
        self,
        *,
        emb_dim: int,
        nlayers: int,
        head_size: int,
        ffn_mult: int,
        sequence_replay_checkpoint: bool = False,
    ):
        super().__init__()
        self.emb_dim = int(emb_dim)
        self.nlayers = int(nlayers)
        self.head_size = int(head_size)
        self.ffn_mult = int(ffn_mult)
        self.sequence_replay_checkpoint = bool(sequence_replay_checkpoint)
        if self.head_size != 64:
            raise ValueError(
                f"Official RWKV-7 maintained path requires rwkv_head_size=64, got {self.head_size}."
            )
        if self.ffn_mult != 4:
            raise ValueError(
                f"Official RWKV-7 integration currently requires rwkv_ffn_mult=4, got {self.ffn_mult}."
            )
        self.blocks = nn.ModuleList(
            [
                OfficialRWKV7Block(
                    emb_dim=self.emb_dim,
                    layer_id=layer_id,
                    num_layers=self.nlayers,
                    head_size=self.head_size,
                )
                for layer_id in range(self.nlayers)
            ]
        )
        self.ln_out = nn.LayerNorm(self.emb_dim)
        self._official_eval_core = None
        self._official_eval_core_device = None

    def train(self, mode: bool = True):
        self._official_eval_core = None
        self._official_eval_core_device = None
        return super().train(mode)

    def load_state_dict(self, state_dict, strict: bool = True):
        self._official_eval_core = None
        self._official_eval_core_device = None
        return super().load_state_dict(state_dict, strict=strict)

    def _flatten_official_eval_state(self, state):
        flat_state = []
        for att_x_prev, att_kv, ffn_x_prev in state:
            flat_state.append(att_x_prev[0].contiguous())
            flat_state.append(att_kv[0].contiguous())
            flat_state.append(ffn_x_prev[0].contiguous())
        return flat_state

    def _is_official_eval_state(self, state) -> bool:
        if not isinstance(state, list):
            return False
        if len(state) != int(self.nlayers) * 3:
            return False
        return all(torch.is_tensor(t) for t in state)

    def _init_official_eval_state(self, *, device: torch.device, dtype: torch.dtype):
        return self._flatten_official_eval_state(self.init_state(1, device=device, dtype=dtype))

    def _build_official_eval_weights(self, *, device: torch.device):
        def _snapshot(name: str, tensor: torch.Tensor):
            tensor = tensor.detach().to(device=device)
            if name.endswith("att.w0"):
                tensor = tensor.to(dtype=torch.float32)
            else:
                tensor = tensor.to(dtype=torch.bfloat16)
            return tensor.contiguous()

        z: Dict[str, torch.Tensor] = {
            "ln_out.weight": _snapshot("ln_out.weight", self.ln_out.weight),
            "ln_out.bias": _snapshot("ln_out.bias", self.ln_out.bias),
        }
        for layer_id, block in enumerate(self.blocks):
            prefix = f"blocks.{layer_id}."
            if hasattr(block, "ln0"):
                z[prefix + "ln0.weight"] = _snapshot(prefix + "ln0.weight", block.ln0.weight)
                z[prefix + "ln0.bias"] = _snapshot(prefix + "ln0.bias", block.ln0.bias)
            z[prefix + "ln1.weight"] = _snapshot(prefix + "ln1.weight", block.ln1.weight)
            z[prefix + "ln1.bias"] = _snapshot(prefix + "ln1.bias", block.ln1.bias)
            z[prefix + "ln2.weight"] = _snapshot(prefix + "ln2.weight", block.ln2.weight)
            z[prefix + "ln2.bias"] = _snapshot(prefix + "ln2.bias", block.ln2.bias)

            att_prefix = prefix + "att."
            z[att_prefix + "x_r"] = _snapshot(att_prefix + "x_r", block.att.x_r.squeeze(0).squeeze(0))
            z[att_prefix + "x_w"] = _snapshot(att_prefix + "x_w", block.att.x_w.squeeze(0).squeeze(0))
            z[att_prefix + "x_k"] = _snapshot(att_prefix + "x_k", block.att.x_k.squeeze(0).squeeze(0))
            z[att_prefix + "x_v"] = _snapshot(att_prefix + "x_v", block.att.x_v.squeeze(0).squeeze(0))
            z[att_prefix + "x_a"] = _snapshot(att_prefix + "x_a", block.att.x_a.squeeze(0).squeeze(0))
            z[att_prefix + "x_g"] = _snapshot(att_prefix + "x_g", block.att.x_g.squeeze(0).squeeze(0))
            z[att_prefix + "w0"] = _snapshot(att_prefix + "w0", block.att.w0.squeeze(0).squeeze(0))
            z[att_prefix + "w1"] = _snapshot(att_prefix + "w1", block.att.w1)
            z[att_prefix + "w2"] = _snapshot(att_prefix + "w2", block.att.w2)
            z[att_prefix + "a0"] = _snapshot(att_prefix + "a0", block.att.a0.squeeze(0).squeeze(0))
            z[att_prefix + "a1"] = _snapshot(att_prefix + "a1", block.att.a1)
            z[att_prefix + "a2"] = _snapshot(att_prefix + "a2", block.att.a2)
            z[att_prefix + "v0"] = _snapshot(att_prefix + "v0", block.att.v0.squeeze(0).squeeze(0))
            z[att_prefix + "v1"] = _snapshot(att_prefix + "v1", block.att.v1)
            z[att_prefix + "v2"] = _snapshot(att_prefix + "v2", block.att.v2)
            z[att_prefix + "g1"] = _snapshot(att_prefix + "g1", block.att.g1)
            z[att_prefix + "g2"] = _snapshot(att_prefix + "g2", block.att.g2)
            z[att_prefix + "k_k"] = _snapshot(att_prefix + "k_k", block.att.k_k.squeeze(0).squeeze(0))
            z[att_prefix + "k_a"] = _snapshot(att_prefix + "k_a", block.att.k_a.squeeze(0).squeeze(0))
            z[att_prefix + "r_k"] = _snapshot(att_prefix + "r_k", block.att.r_k.reshape(-1))
            z[att_prefix + "key.weight"] = _snapshot(att_prefix + "key.weight", block.att.key.weight)
            z[att_prefix + "value.weight"] = _snapshot(att_prefix + "value.weight", block.att.value.weight)
            z[att_prefix + "receptance.weight"] = _snapshot(att_prefix + "receptance.weight", block.att.receptance.weight)
            z[att_prefix + "output.weight"] = _snapshot(att_prefix + "output.weight", block.att.output.weight)
            z[att_prefix + "ln_x.weight"] = _snapshot(att_prefix + "ln_x.weight", block.att.ln_x.weight)
            z[att_prefix + "ln_x.bias"] = _snapshot(att_prefix + "ln_x.bias", block.att.ln_x.bias)

            ffn_prefix = prefix + "ffn."
            z[ffn_prefix + "x_k"] = _snapshot(ffn_prefix + "x_k", block.ffn.x_k.squeeze(0).squeeze(0))
            z[ffn_prefix + "key.weight"] = _snapshot(ffn_prefix + "key.weight", block.ffn.key.weight)
            z[ffn_prefix + "value.weight"] = _snapshot(ffn_prefix + "value.weight", block.ffn.value.weight)
        return z

    def _get_official_eval_core(self, *, device: torch.device):
        if self._official_eval_core is not None and self._official_eval_core_device == device:
            return self._official_eval_core
        z = self._build_official_eval_weights(device=device)
        self._official_eval_core = OfficialRWKV7EvalCore(
            z=z,
            n_embd=self.emb_dim,
            n_layer=self.nlayers,
            n_head=self.emb_dim // self.head_size,
            head_size=self.head_size,
            has_ln0=hasattr(self.blocks[0], "ln0"),
        ).to(device=device)
        self._official_eval_core_device = device
        return self._official_eval_core

    def _forward_step_official_eval(self, token: torch.Tensor, state=None):
        output_dtype = token.dtype
        token = token.to(dtype=torch.bfloat16)
        if state is None:
            flat_state = self._init_official_eval_state(device=token.device, dtype=token.dtype)
        elif self._is_official_eval_state(state):
            flat_state = state
        else:
            flat_state = self._flatten_official_eval_state(state)
        compiler_mod = getattr(torch, "compiler", None)
        if compiler_mod is not None and hasattr(compiler_mod, "cudagraph_mark_step_begin"):
            compiler_mod.cudagraph_mark_step_begin()
        eval_core = self._get_official_eval_core(device=token.device)
        hidden, flat_state = eval_core(token[0], flat_state)
        return hidden.unsqueeze(0).to(dtype=output_dtype), flat_state

    def init_state(self, batch_size: int, *, device: torch.device, dtype: torch.dtype):
        return [block.init_state(batch_size, device=device, dtype=dtype) for block in self.blocks]

    def forward_step(self, token: torch.Tensor, state=None):
        if token.ndim != 2:
            raise ValueError(f"RWKV7Core.forward_step expects (B, C), got {tuple(token.shape)}")
        batch_size = int(token.shape[0])
        if (
            batch_size == 1
            and token.is_cuda
            and (not self.training)
            and (not torch.is_grad_enabled())
        ):
            return self._forward_step_official_eval(token, state)
        if state is None:
            state = self.init_state(batch_size, device=token.device, dtype=token.dtype)
        x = token
        new_state = []
        v_first = None
        autocast_enabled = bool(token.is_cuda)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
            for block, block_state in zip(self.blocks, state):
                x, block_state_next, v_first = block.forward_step(x, block_state, v_first)
                new_state.append(block_state_next)
            x = self.ln_out(x)
        return x, new_state

    def forward_tokens(self, tokens: torch.Tensor, state=None):
        if tokens.ndim != 3:
            raise ValueError(f"RWKV7Core.forward_tokens expects (T, B, C), got {tuple(tokens.shape)}")
        if tokens.shape[0] == 0:
            if state is None:
                state = self.init_state(int(tokens.shape[1]), device=tokens.device, dtype=tokens.dtype)
            return tokens, state
        outputs = []
        current_state = state
        for t in range(int(tokens.shape[0])):
            out_t, current_state = self.forward_step(tokens[t], current_state)
            outputs.append(out_t.unsqueeze(0))
        return torch.cat(outputs, dim=0), current_state

    def forward_tokens_sequence_only(self, tokens: torch.Tensor):
        if tokens.ndim != 3:
            raise ValueError(f"RWKV7Core.forward_tokens_sequence_only expects (T, B, C), got {tuple(tokens.shape)}")
        if tokens.shape[0] == 0:
            return tokens
        if not tokens.is_cuda:
            raise RuntimeError("Official RWKV-7 training sequence path requires CUDA tensors.")
        seq_len = int(tokens.shape[0])
        pad_len = int((-seq_len) % 16)
        if pad_len > 0:
            tokens = torch.cat(
                [
                    tokens,
                    torch.zeros(
                        (pad_len, int(tokens.shape[1]), int(tokens.shape[2])),
                        device=tokens.device,
                        dtype=tokens.dtype,
                    ),
                ],
                dim=0,
            )
        x = tokens.transpose(0, 1)
        v_first = None
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for block in self.blocks:
                if self.sequence_replay_checkpoint and torch.is_grad_enabled():
                    if v_first is None:
                        def _run_block_no_vfirst(x_in, *, _block=block):
                            return _block.forward_sequence(x_in, None)

                        x, v_first = torch.utils.checkpoint.checkpoint(
                            _run_block_no_vfirst,
                            x,
                            use_reentrant=False,
                        )
                    else:
                        def _run_block_with_vfirst(x_in, v_first_in, *, _block=block):
                            return _block.forward_sequence(x_in, v_first_in)

                        x, v_first = torch.utils.checkpoint.checkpoint(
                            _run_block_with_vfirst,
                            x,
                            v_first,
                            use_reentrant=False,
                        )
                else:
                    x, v_first = block.forward_sequence(x, v_first)
            x = self.ln_out(x)
        x = x.transpose(0, 1)
        if pad_len > 0:
            x = x[:seq_len]
        return x


class RWKV7PFN(nn.Module):
    def __init__(
        self,
        *,
        n_out,
        emsize,
        nhead,
        nhid_factor,
        nlayers,
        n_features,
        dropout=0.0,
        y_encoder_layer=None,
        decoder=None,
        input_normalization=False,
        init_method=None,
        pre_norm=False,
        activation="gelu",
        recompute_attn=False,
        classification_task=True,
        all_layers_same_init=False,
        efficient_eval_masking=True,
        y_encoder=None,
        tabpfn_zero_weights=False,
        x_encoder_type="single",
        x_obs_dim=None,
        x_action_dim=None,
        single_eval_causal=False,
        backbone="rwkv7",
        rwkv_head_size=64,
        rwkv_ffn_mult=4,
        rwkv_sequence_replay_checkpoint=False,
        rwkv_sequence_replay_batch_chunk_size=None,
        rwkv_sequence_replay_token_budget=None,
    ):
        del dropout, pre_norm, activation, recompute_attn, all_layers_same_init, y_encoder
        super().__init__()
        self.classification_task = classification_task
        self.y_encoder = y_encoder_layer
        self.emsize = int(emsize)
        self.x_encoder_type = x_encoder_type
        self.backbone = str(backbone)
        self.rwkv_head_size = int(rwkv_head_size)
        self.rwkv_ffn_mult = int(rwkv_ffn_mult)
        self.rwkv_sequence_replay_checkpoint = bool(rwkv_sequence_replay_checkpoint)
        self.rwkv_sequence_replay_batch_chunk_size = (
            None if rwkv_sequence_replay_batch_chunk_size in (None, 0, False)
            else int(rwkv_sequence_replay_batch_chunk_size)
        )
        self.rwkv_sequence_replay_token_budget = (
            None if rwkv_sequence_replay_token_budget in (None, 0, False)
            else int(rwkv_sequence_replay_token_budget)
        )

        if self.x_encoder_type == "single":
            self.encoder = Linear(n_features, self.emsize, replace_nan_by_zero=True)
        elif self.x_encoder_type == "split_obs_action":
            if x_obs_dim is None or x_action_dim is None:
                raise ValueError("x_obs_dim and x_action_dim must be set for split_obs_action encoder.")
            split_total = int(x_obs_dim) + int(x_action_dim)
            if int(n_features) < split_total:
                raise ValueError(
                    f"n_features={n_features} is smaller than split total {split_total} "
                    f"(x_obs_dim={x_obs_dim}, x_action_dim={x_action_dim})"
                )
            self.encoder = SplitObsActionEncoder(
                obs_dim=int(x_obs_dim),
                action_dim=int(x_action_dim),
                emsize=self.emsize,
                replace_nan_by_zero=True,
            )
        else:
            raise ValueError(f"Unknown x_encoder_type: {self.x_encoder_type}")

        self.rwkv_core = RWKV7Core(
            emb_dim=self.emsize,
            nlayers=int(nlayers),
            head_size=self.rwkv_head_size,
            ffn_mult=self.rwkv_ffn_mult,
            sequence_replay_checkpoint=self.rwkv_sequence_replay_checkpoint,
        )
        backbone_size = sum(p.numel() for p in self.rwkv_core.parameters())
        if wandb.run:
            wandb.log({"backbone_size": backbone_size})
        print("Number of parameters in backbone: ", backbone_size)

        nhid = self.emsize * int(nhid_factor)
        self.decoder = decoder(self.emsize, nhid, n_out) if decoder is not None else _default_mlp_head(self.emsize, nhid, n_out)
        self.policy_action_dim = int(x_action_dim) if x_action_dim is not None else None
        self.policy_action_head = None
        if self.policy_action_dim is not None and self.policy_action_dim > 0:
            self.policy_action_head = (
                decoder(self.emsize, nhid, self.policy_action_dim)
                if decoder is not None
                else _default_mlp_head(self.emsize, nhid, self.policy_action_dim)
            )

        self.input_ln = SeqBN(self.emsize) if input_normalization else None
        self.init_method = init_method
        self.efficient_eval_masking = efficient_eval_masking
        self.tabpfn_zero_weights = bool(tabpfn_zero_weights)
        self.single_eval_causal = bool(single_eval_causal)
        self.n_out = int(n_out)
        self.nhid = int(nhid)
        self._policy_step_profile_enabled = False

    def _rwkv_core_param(self):
        return next(self.rwkv_core.parameters())

    def _ensure_rwkv_core_runtime_dtype(self, device: torch.device):
        del device
        return

    def _cast_token_for_rwkv_core(self, token: torch.Tensor):
        return token

    def has_policy_action_head(self):
        return isinstance(self.policy_action_head, nn.Module)

    def policy_action_head_required(self):
        return bool(self.x_encoder_type == "split_obs_action" and self.policy_action_dim is not None and self.policy_action_dim > 0)

    @staticmethod
    def _infer_head_out_dim(head):
        if isinstance(head, nn.Linear):
            return int(head.out_features)
        if isinstance(head, nn.Sequential):
            for module in reversed(head):
                out_features = getattr(module, "out_features", None)
                if out_features is not None:
                    return int(out_features)
        for module in reversed(list(head.modules())):
            if module is head:
                continue
            out_features = getattr(module, "out_features", None)
            if out_features is not None:
                return int(out_features)
        return None

    def has_correct_policy_action_head(self):
        if not self.policy_action_head_required():
            return True
        if not self.has_policy_action_head():
            return False
        expected_dim = int(self.policy_action_dim)
        actual_dim = self._infer_head_out_dim(self.policy_action_head)
        return actual_dim is None or int(actual_dim) == expected_dim

    def require_policy_action_head(self):
        if not self.policy_action_head_required():
            return True
        if not self.has_policy_action_head():
            raise ValueError(
                "split_obs_action policy rollout requires policy_action_head; "
                "refusing to fall back to the scalar decoder."
            )
        expected_dim = int(self.policy_action_dim)
        actual_dim = self._infer_head_out_dim(self.policy_action_head)
        if actual_dim is not None and int(actual_dim) != expected_dim:
            raise ValueError(
                "split_obs_action policy rollout requires policy_action_head "
                f"output width {expected_dim}, got {int(actual_dim)}."
            )
        return True

    def _decode_policy_action(self, hidden):
        target_dtype = None
        if self.policy_action_head_required():
            self.require_policy_action_head()
            head_param = next(self.policy_action_head.parameters())
            target_dtype = head_param.dtype
            if hidden.dtype != target_dtype:
                hidden = hidden.to(dtype=target_dtype)
            return self.policy_action_head(hidden)
        head_param = next(self.decoder.parameters())
        target_dtype = head_param.dtype
        if hidden.dtype != target_dtype:
            hidden = hidden.to(dtype=target_dtype)
        return self.decoder(hidden)

    def consume_policy_step_profile(self):
        return None

    def policy_fastpath_compile_active(self):
        return False

    def get_policy_fastpath_compile_config(self):
        return {"finalize_torch_compile": False}

    def warmup_policy_fastpaths(self, batch_size: int):
        del batch_size
        return False

    def _encode_xy(self, src):
        if len(src) == 3:
            _, x_src, y_src = src
        else:
            x_src, y_src = src
        x_enc = self.encoder(x_src)
        if self.y_encoder is None:
            y_enc = torch.zeros_like(x_enc)
        else:
            y_enc = self.y_encoder(y_src.unsqueeze(-1) if len(y_src.shape) < len(x_enc.shape) else y_src)
        return x_enc, y_enc

    def _encode_train_token(self, x_token, y_token):
        x_enc = self.encoder(x_token)
        if self.y_encoder is None:
            y_enc = torch.zeros_like(x_enc)
        else:
            y_enc = self.y_encoder(y_token.unsqueeze(-1) if len(y_token.shape) < len(x_enc.shape) else y_token)
        token = x_enc + y_enc
        if self.input_ln is not None:
            token = self.input_ln(token)
        return token

    def _encode_query_token(self, x_token):
        x_enc = self.encoder(x_token)
        if self.input_ln is not None:
            x_enc = self.input_ln(x_enc)
        return x_enc

    def _encode_split_train_token(
        self,
        obs_t,
        action_t,
        reward_t,
        reward_mask_t,
        phase_t=None,
        terminal_t=None,
    ):
        if self.x_encoder_type != "split_obs_action" or (not isinstance(self.encoder, SplitObsActionEncoder)):
            raise ValueError("_encode_split_train_token requires split_obs_action encoder.")
        if obs_t.ndim != 2 or action_t.ndim != 2:
            raise ValueError(
                f"obs_t/action_t must have shape (B, D), got {tuple(obs_t.shape)} / {tuple(action_t.shape)}"
            )
        batch_size = int(obs_t.shape[0])
        if int(action_t.shape[0]) != batch_size:
            raise ValueError("obs_t and action_t batch size mismatch")

        obs_dtype = obs_t.dtype
        obs_device = obs_t.device
        reward_scalar = reward_t.reshape(batch_size).to(dtype=obs_dtype, device=obs_device)
        reward_mask_scalar = reward_mask_t.reshape(batch_size).to(dtype=obs_dtype, device=obs_device)
        phase_scalar = None if phase_t is None else phase_t.reshape(batch_size).to(dtype=obs_dtype, device=obs_device)
        terminal_scalar = None if terminal_t is None else terminal_t.reshape(batch_size).to(dtype=obs_dtype, device=obs_device)

        obs_dim = int(self.encoder.obs_dim)
        action_dim = int(self.encoder.action_dim)
        extra_scalar_slots = int(max(0, obs_dim - int(obs_t.shape[-1]) - 2))
        phase_token_enabled = bool(phase_scalar is not None)
        terminal_token_enabled = bool(terminal_scalar is not None)
        remaining_scalar_slots = int(max(0, extra_scalar_slots - int(phase_token_enabled) - int(terminal_token_enabled)))
        if remaining_scalar_slots > 0 and phase_scalar is None:
            phase_scalar = torch.zeros((batch_size,), device=obs_device, dtype=obs_dtype)
            phase_token_enabled = True
            remaining_scalar_slots -= 1
        if remaining_scalar_slots > 0 and terminal_scalar is None:
            terminal_scalar = torch.zeros((batch_size,), device=obs_device, dtype=obs_dtype)
            terminal_token_enabled = True
            remaining_scalar_slots -= 1
        obs_slot_dim = int(max(0, obs_dim - (2 + int(phase_token_enabled) + int(terminal_token_enabled))))

        obs_features = torch.zeros((batch_size, obs_dim), device=obs_device, dtype=obs_dtype)
        obs_copy = int(min(int(obs_t.shape[-1]), obs_slot_dim))
        if obs_copy > 0:
            obs_src = (
                torch.nan_to_num(obs_t, nan=0.0)
                if bool(getattr(self.encoder.obs_encoder, "replace_nan_by_zero", False))
                else obs_t
            )
            obs_features[:, :obs_copy] = obs_src[:, :obs_copy]
        reward_idx = obs_slot_dim
        mask_idx = obs_slot_dim + 1
        if reward_idx < obs_dim:
            obs_features[:, reward_idx] = reward_scalar
        if mask_idx < obs_dim:
            obs_features[:, mask_idx] = reward_mask_scalar
        phase_idx = obs_slot_dim + 2
        if phase_token_enabled and phase_idx < obs_dim:
            obs_features[:, phase_idx] = phase_scalar
        terminal_idx = obs_slot_dim + 2 + int(phase_token_enabled)
        if terminal_token_enabled and terminal_idx < obs_dim:
            obs_features[:, terminal_idx] = terminal_scalar

        action_features = torch.zeros((batch_size, action_dim), device=obs_device, dtype=obs_dtype)
        action_copy = int(min(int(action_t.shape[-1]), action_dim))
        if action_copy > 0:
            action_src = (
                torch.nan_to_num(action_t, nan=0.0)
                if bool(getattr(self.encoder.action_encoder, "replace_nan_by_zero", False))
                else action_t
            )
            action_features[:, :action_copy] = action_src[:, :action_copy]

        x_enc = self.encoder.obs_encoder(obs_features) + self.encoder.action_encoder(action_features)
        if self.y_encoder is None:
            y_enc = torch.zeros_like(x_enc)
        else:
            y_enc = self.y_encoder(reward_scalar.reshape(batch_size, 1))
        token = x_enc + y_enc
        if self.input_ln is not None:
            token = self.input_ln(token)
        return token

    def forward(self, src, single_eval_pos=None):
        assert isinstance(src, tuple), "inputs (src) have to be given as (x,y) or (style,x,y) tuple"
        if single_eval_pos is None:
            raise ValueError("single_eval_pos has to be given, instead of None.")
        if not self.single_eval_causal:
            raise ValueError("RWKV7PFN currently supports single_eval_causal=True only.")

        x_src, y_src = self._encode_xy(src)
        train_x = x_src[:single_eval_pos] + y_src[:single_eval_pos]
        query_x = x_src[single_eval_pos:]
        full_tokens = torch.cat([train_x, query_x], dim=0)
        if self.input_ln is not None:
            full_tokens = self.input_ln(full_tokens)
        full_tokens = self._cast_token_for_rwkv_core(full_tokens)
        hidden_all = self.rwkv_core.forward_tokens_sequence_only(full_tokens)
        hidden_q = hidden_all[single_eval_pos:]
        return self._decode_policy_action(hidden_q)

    def init_kv_cache(self, x_train, y_train):
        if not self.single_eval_causal:
            raise ValueError("RWKV state cache requires single_eval_causal=True.")
        token = self._encode_train_token(x_train, y_train)
        token = self._cast_token_for_rwkv_core(token)
        _, state = self.rwkv_core.forward_tokens(token, None)
        return state

    def append_train_token_to_kv(
        self,
        x_token,
        y_token,
        kv_cache,
        max_cache_len=None,
        kv_cache_mode: str = "auto",
        kv_cache_page_size=None,
        allow_grad_mutable_cache: bool = False,
        allow_grad_inplace_paged_cache: bool = False,
    ):
        del max_cache_len, kv_cache_mode, kv_cache_page_size, allow_grad_mutable_cache, allow_grad_inplace_paged_cache
        if not self.single_eval_causal:
            raise ValueError("RWKV state cache requires single_eval_causal=True.")
        token = self._encode_train_token(x_token, y_token)
        token = self._cast_token_for_rwkv_core(token)
        _, state = self.rwkv_core.forward_tokens(token, kv_cache)
        return state

    def predict_query_with_kv(self, x_query, kv_cache):
        if not self.single_eval_causal:
            raise ValueError("RWKV state cache requires single_eval_causal=True.")
        token = self._encode_query_token(x_query)
        token = self._cast_token_for_rwkv_core(token)
        hidden, _ = self.rwkv_core.forward_tokens(token, kv_cache)
        return self._decode_policy_action(hidden)

    def replay_policy_sequence_tokens(self, x_tokens, y_tokens, *, eval_start: int = 0):
        if not self.single_eval_causal:
            raise ValueError("replay_policy_sequence_tokens requires single_eval_causal=True.")
        if x_tokens.ndim != 3:
            raise ValueError(
                f"replay_policy_sequence_tokens expects x_tokens with shape (T, B, F), got {tuple(x_tokens.shape)}"
            )
        if y_tokens.ndim != 2:
            raise ValueError(
                f"replay_policy_sequence_tokens expects y_tokens with shape (T, B), got {tuple(y_tokens.shape)}"
            )
        if tuple(x_tokens.shape[:2]) != tuple(y_tokens.shape):
            raise ValueError(
                "replay_policy_sequence_tokens expects matching leading dims, "
                f"got {tuple(x_tokens.shape[:2])} and {tuple(y_tokens.shape)}"
            )
        eval_start = int(max(0, min(int(x_tokens.shape[0]), int(eval_start))))
        total_batch = int(x_tokens.shape[1])
        batch_chunk_size = self.resolve_replay_batch_chunk_size(
            seq_len=int(x_tokens.shape[0]),
            total_batch=total_batch,
        )
        if batch_chunk_size >= total_batch:
            tokens = self._encode_train_token(x_tokens, y_tokens)
            tokens = self._cast_token_for_rwkv_core(tokens)
            hidden_all = self.rwkv_core.forward_tokens_sequence_only(tokens)
            return self._decode_policy_action(hidden_all[eval_start:])
        if self.input_ln is not None:
            raise RuntimeError(
                "RWKV sequence replay batch microbatching requires input_normalization=False "
                "to preserve exact BatchNorm semantics."
            )
        decoded_chunks = []
        for start in range(0, total_batch, int(batch_chunk_size)):
            end = min(total_batch, start + int(batch_chunk_size))
            tokens_chunk = self._encode_train_token(
                x_tokens[:, start:end],
                y_tokens[:, start:end],
            )
            tokens_chunk = self._cast_token_for_rwkv_core(tokens_chunk)
            hidden_chunk = self.rwkv_core.forward_tokens_sequence_only(tokens_chunk)
            decoded_chunks.append(self._decode_policy_action(hidden_chunk[eval_start:]))
        return torch.cat(decoded_chunks, dim=1)

    def resolve_replay_batch_chunk_size(self, *, seq_len: int, total_batch: int):
        seq_len = int(max(1, seq_len))
        total_batch = int(max(1, total_batch))
        effective_chunk = total_batch
        if self.rwkv_sequence_replay_batch_chunk_size is not None:
            effective_chunk = min(effective_chunk, int(max(1, self.rwkv_sequence_replay_batch_chunk_size)))
        if self.rwkv_sequence_replay_token_budget is not None:
            token_budget = int(max(1, self.rwkv_sequence_replay_token_budget))
            effective_chunk = min(effective_chunk, int(max(1, token_budget // seq_len)))
        return int(max(1, effective_chunk))

    def forward_policy_step(
        self,
        x_token,
        y_token,
        kv_cache=None,
        max_cache_len=None,
        kv_cache_mode: str = "auto",
        kv_cache_page_size=None,
        allow_grad_mutable_cache: bool = False,
        allow_grad_inplace_paged_cache: bool = False,
    ):
        del max_cache_len, kv_cache_mode, kv_cache_page_size, allow_grad_mutable_cache, allow_grad_inplace_paged_cache
        if not self.single_eval_causal:
            raise ValueError("forward_policy_step requires single_eval_causal=True.")
        if x_token.ndim != 3 or x_token.shape[0] != 1:
            raise ValueError(f"x_token must have shape (1, B, F), got {tuple(x_token.shape)}")
        token = self._encode_train_token(x_token, y_token)
        token = self._cast_token_for_rwkv_core(token)
        hidden, kv_cache = self.rwkv_core.forward_step(token[0], kv_cache)
        out = self._decode_policy_action(hidden.unsqueeze(0))
        return out, kv_cache

    def forward_policy_step_split(
        self,
        obs_t,
        action_t,
        reward_t,
        reward_mask_t,
        phase_t=None,
        terminal_t=None,
        kv_cache=None,
        max_cache_len=None,
        kv_cache_mode: str = "auto",
        kv_cache_page_size=None,
        allow_grad_mutable_cache: bool = False,
        allow_grad_inplace_paged_cache: bool = False,
    ):
        del max_cache_len, kv_cache_mode, kv_cache_page_size, allow_grad_mutable_cache, allow_grad_inplace_paged_cache
        if not self.single_eval_causal:
            raise ValueError("forward_policy_step_split requires single_eval_causal=True.")
        if self.x_encoder_type != "split_obs_action" or (not isinstance(self.encoder, SplitObsActionEncoder)):
            raise ValueError("forward_policy_step_split requires split_obs_action encoder.")
        token = self._encode_split_train_token(
            obs_t,
            action_t,
            reward_t,
            reward_mask_t,
            phase_t=phase_t,
            terminal_t=terminal_t,
        )
        token = self._cast_token_for_rwkv_core(token)
        hidden, kv_cache = self.rwkv_core.forward_step(token, kv_cache)
        out = self._decode_policy_action(hidden.unsqueeze(0))
        return out, kv_cache

    def forward_with_kv(self, src, single_eval_pos=None):
        assert isinstance(src, tuple), "inputs (src) have to be given as (x,y) or (style,x,y) tuple"
        if single_eval_pos is None:
            raise ValueError("single_eval_pos has to be given, instead of None.")
        if not self.single_eval_causal:
            raise ValueError("forward_with_kv requires single_eval_causal=True.")
        if len(src) == 3:
            _, x_src, y_src = src
        else:
            x_src, y_src = src
        kv_cache = self.init_kv_cache(x_src[:single_eval_pos], y_src[:single_eval_pos])
        return self.predict_query_with_kv(x_src[single_eval_pos:], kv_cache)
