import math
import os
import time
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.transformer import (Dropout, LayerNorm, Linear, Module,
                                          _get_activation_fn)
from torch.utils.checkpoint import checkpoint
from torch.nn import MultiheadAttention

from torch.nn import TransformerEncoder


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


_CAT_FUSION_ENV = str(os.environ.get("TICL_POLICY_CAT_FUSION", "1")).strip().lower()
_CAT_FUSION_ENABLED = _CAT_FUSION_ENV not in {"0", "false", "no", "off"}


class BiAttentionEncoderLayer(Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu",
                 layer_norm_eps=1e-5, batch_first=True, pre_norm=False,
                 device=None, dtype=None, recompute_attn=False):
        super().__init__()
        self.cross_feature_attention = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, batch_first=batch_first)
        self.cross_sample_attention = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, batch_first=batch_first)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None) -> Tensor:
        # src_mask is in with eval position, applies only to samples
        # src comes in as samples x batch x feature x emsize
        # reshape to features x (samples * batch) x emsize for cross-feature attention
        post_feature_attention = self.cross_feature_attention(src.reshape(-1, *src.shape[2:]).transpose(0, 1), src_mask)
        # from cross-feature attention, we get features x (samples * batch) x emsize
        # reshape back to original, then reshape to samples x (batch * feature) x emsize
        reshaped = post_feature_attention.transpose(0, 1).reshape(src.shape)
        reshaped = reshaped.reshape(src.shape[0], -1, src.shape[-1])
        res = self.cross_sample_attention(reshaped, src_mask)
        return res.reshape(src.shape)


class LinearBiAttentionEncoderLayer(Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu",
                 layer_norm_eps=1e-5, batch_first=True, pre_norm=False,
                 device=None, dtype=None, recompute_attn=False):
        super().__init__()
        self.cross_feature_attention = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, batch_first=batch_first)
        self.cross_sample_attention = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, batch_first=batch_first)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None) -> Tensor:
        # src_mask is in with eval position, applies only to samples
        # src comes in as samples x batch x feature x emsize
        # reshape to features x (samples * batch) x emsize for cross-feature attention
        post_feature_attention = self.cross_feature_attention(src.reshape(-1, *src.shape[2:]).transpose(0, 1), src_mask)
        # from cross-feature attention, we get features x (samples * batch) x emsize
        # reshape back to original, then reshape to samples x (batch * feature) x emsize
        reshaped = post_feature_attention.transpose(0, 1).reshape(src.shape)
        reshaped = reshaped.reshape(src.shape[0], -1, src.shape[-1])
        res = self.cross_sample_attention(reshaped, src_mask)
        return res.reshape(src.shape)



class TransformerEncoderLayer(Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False``.

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)

    Alternatively, when ``batch_first`` is ``True``:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> src = torch.rand(32, 10, 512)
        >>> out = encoder_layer(src)
    """
    __constants__ = ['batch_first']

    def __init__(
        self, 
        d_model, 
        nhead, 
        dim_feedforward=2048, 
        dropout=0.1, 
        activation="relu",
        layer_norm_eps=1e-5, 
        batch_first=True, 
        pre_norm=False,
        device=None, 
        dtype=None, 
        recompute_attn=False,
        attn_name = 'default',
        norm_output = False,
        single_eval_causal: bool = False,
    ) -> None:
        # batch_first is set to True for using flash attention II
        # check the details of when flash attention can be triggered here: https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
        
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.self_attn = MultiheadAttention(
                d_model, 
                nhead, 
                dropout=dropout, 
                batch_first=batch_first,
                **factory_kwargs,
            )
        self.attn_name = attn_name
        
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.pre_norm = pre_norm
        self.recompute_attn = recompute_attn
        self.single_eval_causal = bool(single_eval_causal)
        paged_mode_env = str(os.environ.get("TICL_POLICY_PAGED_ATTN_TRAIN_MODE", "auto")).strip().lower()
        if paged_mode_env not in {"auto", "dense", "flash_merge", "flash_prefix"}:
            paged_mode_env = "auto"
        self.paged_attn_train_mode = paged_mode_env
        try:
            chunk_tokens_env = int(os.environ.get("TICL_POLICY_PAGED_ATTN_FLASHMERGE_CHUNK_TOKENS", "0"))
        except Exception:
            chunk_tokens_env = 0
        self.paged_attn_train_chunk_tokens = int(max(0, chunk_tokens_env))
        try:
            flashprefix_page_size_env = int(os.environ.get("TICL_POLICY_PAGED_ATTN_FLASHPREFIX_PAGE_SIZE", "32"))
        except Exception:
            flashprefix_page_size_env = 32
        self.paged_attn_flashprefix_page_size = int(max(1, flashprefix_page_size_env))
        try:
            flashprefix_dense_tokens_env = int(
                os.environ.get("TICL_POLICY_PAGED_ATTN_FLASHPREFIX_DENSE_MAX_TOKENS", "64")
            )
        except Exception:
            flashprefix_dense_tokens_env = 64
        self.paged_attn_flashprefix_dense_max_tokens = int(max(0, flashprefix_dense_tokens_env))
        try:
            dense_page_size_env = int(os.environ.get("TICL_POLICY_PAGED_ATTN_DENSE_PAGE_SIZE", "128"))
        except Exception:
            dense_page_size_env = 48
        self.paged_attn_dense_page_size = int(max(1, dense_page_size_env))
        force_flash_single_page_env = str(os.environ.get("TICL_POLICY_FORCE_FLASH_SINGLE_PAGE", "0")).strip().lower()
        self.force_flash_single_page = force_flash_single_page_env in {"1", "true", "yes", "on"}
        flash_prefix_async_env = str(os.environ.get("TICL_POLICY_FLASH_PREFIX_ASYNC", "1")).strip().lower()
        self.flash_prefix_async = flash_prefix_async_env in {"1", "true", "yes", "on"}
        self._flash_prefix_streams = {}
        prefix_compact_env = str(os.environ.get("TICL_POLICY_PAGED_PREFIX_COMPACT_PAGES", "1")).strip().lower()
        self.paged_prefix_compact_pages = prefix_compact_env not in {"0", "false", "no", "off"}
        prefix_auto_route_env = str(os.environ.get("TICL_POLICY_PAGED_PREFIX_AUTO_ROUTE", "0")).strip().lower()
        self.paged_prefix_auto_route = prefix_auto_route_env in {"1", "true", "yes", "on"}
        paged_recompute_env = str(os.environ.get("TICL_POLICY_PAGED_RECOMPUTE_ATTN", "0")).strip().lower()
        self.paged_recompute_attn = paged_recompute_env not in {"0", "false", "no", "off"}
        step_profile_flag = str(os.environ.get("TICL_TRANSFORMER_LAYER_STEP_PROFILE", "")).strip().lower()
        self.layer_step_profile_enabled = step_profile_flag in {"1", "true", "yes", "on"}
        finalize_2d_flag = str(os.environ.get("TICL_POLICY_FINALIZE_2D_FASTPATH", "1")).strip().lower()
        self.finalize_2d_fastpath = finalize_2d_flag not in {"0", "false", "no", "off"}
        cache_reuse_flag = str(os.environ.get("TICL_POLICY_CACHE_CONTAINER_REUSE", "1")).strip().lower()
        self.cache_container_reuse = cache_reuse_flag not in {"0", "false", "no", "off"}
        self._layer_step_profile_stats = {
            "calls": 0,
            "proj_wall_s": 0.0,
            "cache_wall_s": 0.0,
            "attnff_wall_s": 0.0,
            "attn_core_wall_s": 0.0,
            "finalize_wall_s": 0.0,
            "finalize_attn_outproj_wall_s": 0.0,
            "finalize_ffn_wall_s": 0.0,
            "paged_path_single_page": 0,
            "paged_path_flash_prefix": 0,
            "paged_path_flash_merge": 0,
            "paged_path_dense": 0,
            "paged_recompute_calls": 0,
            "total_wall_s": 0.0,
        }

        self.activation = _get_activation_fn(activation)

    def consume_forward_step_profile(self):
        if not bool(self.layer_step_profile_enabled):
            return None
        stats = dict(self._layer_step_profile_stats)
        self._layer_step_profile_stats = {
            "calls": 0,
            "proj_wall_s": 0.0,
            "cache_wall_s": 0.0,
            "attnff_wall_s": 0.0,
            "attn_core_wall_s": 0.0,
            "finalize_wall_s": 0.0,
            "finalize_attn_outproj_wall_s": 0.0,
            "finalize_ffn_wall_s": 0.0,
            "paged_path_single_page": 0,
            "paged_path_flash_prefix": 0,
            "paged_path_flash_merge": 0,
            "paged_path_dense": 0,
            "paged_recompute_calls": 0,
            "total_wall_s": 0.0,
        }
        return stats

    def _project_qkv(self, x_bld: Tensor):
        # True projection fusion: one GEMM on (B, L, E) -> (B, L, 3E),
        # then split Q/K/V on the feature axis.
        qkv = F.linear(
            x_bld,
            self.self_attn.in_proj_weight,
            self.self_attn.in_proj_bias,
        )
        return qkv.chunk(3, dim=-1)

    def _project_q(self, x_bld: Tensor):
        w_q = self.self_attn.in_proj_weight[: self.self_attn.embed_dim]
        if self.self_attn.in_proj_bias is not None:
            b_q = self.self_attn.in_proj_bias[: self.self_attn.embed_dim]
        else:
            b_q = None
        return F.linear(x_bld, w_q, b_q)

    def _project_kv(self, x_bld: Tensor):
        embed_dim = self.self_attn.embed_dim
        w_kv = self.self_attn.in_proj_weight[embed_dim:]
        if self.self_attn.in_proj_bias is not None:
            b_kv = self.self_attn.in_proj_bias[embed_dim:]
        else:
            b_kv = None
        kv = F.linear(x_bld, w_kv, b_kv)
        return kv.chunk(2, dim=-1)

    def _step_profile_enabled_now(self):
        return bool(self.layer_step_profile_enabled) and (not _is_torch_compiling())

    def _split_heads(self, x_bld: Tensor):
        bsz, seq_len, emsize = x_bld.shape
        n_heads = int(self.self_attn.num_heads)
        head_dim = emsize // n_heads
        return x_bld.view(bsz, seq_len, n_heads, head_dim).transpose(1, 2)

    @staticmethod
    def _merge_heads(x_bhld: Tensor):
        bsz, n_heads, seq_len, head_dim = x_bhld.shape
        return x_bhld.transpose(1, 2).contiguous().view(bsz, seq_len, n_heads * head_dim)

    def _finalize_forward_step(self, src_step: Tensor, attn_bhld: Tensor):
        profile_enabled = self._step_profile_enabled_now()
        outproj_t0 = time.perf_counter() if profile_enabled else None
        input_was_3d = src_step.ndim == 3
        if input_was_3d:
            src_step_2d = src_step.squeeze(0)
            src_step_3d = src_step
        else:
            src_step_2d = src_step
            src_step_3d = src_step.unsqueeze(0)
        use_2d_fastpath = bool(self.finalize_2d_fastpath) and int(attn_bhld.shape[2]) == 1
        if use_2d_fastpath:
            # forward_step hot path uses single-token query (L=1); keep
            # finalize in 2D (B, E) to avoid extra permute/contiguous overhead.
            attn_bld = attn_bhld.squeeze(2).reshape(attn_bhld.shape[0], -1)  # (B, E)
            attn_bld = F.linear(attn_bld, self.self_attn.out_proj.weight, self.self_attn.out_proj.bias)  # (B, E)
            if self.training and float(self.dropout1.p) > 0.0:
                attn_bld = F.dropout(attn_bld, p=float(self.dropout1.p), training=True)
            src = src_step_2d + attn_bld  # (B, E)
        else:
            attn_bld = self._merge_heads(attn_bhld)
            attn_bld = F.linear(attn_bld, self.self_attn.out_proj.weight, self.self_attn.out_proj.bias)
            src2 = attn_bld.permute(1, 0, 2)  # (1, B, E)
            src = src_step_3d + self.dropout1(src2)
        if not self.pre_norm:
            src = F.layer_norm(
                src,
                self.norm1.normalized_shape,
                self.norm1.weight,
                self.norm1.bias,
                self.norm1.eps,
            )
        outproj_dt = (time.perf_counter() - outproj_t0) if outproj_t0 is not None else 0.0

        ffn_t0 = time.perf_counter() if profile_enabled else None
        if self.pre_norm:
            src_ff = F.layer_norm(
                src,
                self.norm2.normalized_shape,
                self.norm2.weight,
                self.norm2.bias,
                self.norm2.eps,
            )
        else:
            src_ff = src
        src2_ff = F.linear(src_ff, self.linear1.weight, self.linear1.bias)
        src2_ff = self.activation(src2_ff)
        if self.training and float(self.dropout.p) > 0.0:
            src2_ff = F.dropout(src2_ff, p=float(self.dropout.p), training=True)
        src2_ff = F.linear(src2_ff, self.linear2.weight, self.linear2.bias)
        if self.training and float(self.dropout2.p) > 0.0:
            src2_ff = F.dropout(src2_ff, p=float(self.dropout2.p), training=True)
        src = src + src2_ff

        if not self.pre_norm:
            src = F.layer_norm(
                src,
                self.norm2.normalized_shape,
                self.norm2.weight,
                self.norm2.bias,
                self.norm2.eps,
            )
        ffn_dt = (time.perf_counter() - ffn_t0) if ffn_t0 is not None else 0.0
        if profile_enabled:
            stats = self._layer_step_profile_stats
            stats["finalize_attn_outproj_wall_s"] += float(outproj_dt)
            stats["finalize_ffn_wall_s"] += float(ffn_dt)
        if use_2d_fastpath:
            return src.unsqueeze(0) if input_was_3d else src
        return src if input_was_3d else src.squeeze(0)

    def _forward_step_attn_ff(self, src_step: Tensor, q_bhld: Tensor, k_all: Tensor, v_all: Tensor):
        attn_dropout = float(self.self_attn.dropout) if self.training else 0.0
        profile_enabled = self._step_profile_enabled_now()
        attn_t0 = time.perf_counter() if profile_enabled else None
        attn_bhld = F.scaled_dot_product_attention(
            q_bhld,
            k_all,
            v_all,
            attn_mask=None,
            dropout_p=attn_dropout,
            is_causal=False,
        )
        attn_dt = (time.perf_counter() - attn_t0) if attn_t0 is not None else 0.0
        finalize_t0 = time.perf_counter() if profile_enabled else None
        src = self._finalize_forward_step(src_step, attn_bhld)
        finalize_dt = (time.perf_counter() - finalize_t0) if finalize_t0 is not None else 0.0
        if profile_enabled:
            stats = self._layer_step_profile_stats
            stats["attn_core_wall_s"] += float(attn_dt)
            stats["finalize_wall_s"] += float(finalize_dt)
        return src

    @staticmethod
    def _concat_dim2(chunks):
        if not chunks:
            raise ValueError("concat requires at least one chunk.")
        if len(chunks) == 1:
            return chunks[0]
        if _CAT_FUSION_ENABLED:
            # Prefer a single cat kernel over multiple explicit slice-copy launches.
            return torch.cat(tuple(chunks), dim=2)
        total_len = int(sum(int(t.shape[2]) for t in chunks))
        out = chunks[0].new_empty(
            (chunks[0].shape[0], chunks[0].shape[1], total_len, chunks[0].shape[3])
        )
        offset = 0
        for t in chunks:
            take = int(t.shape[2])
            out[:, :, offset: offset + take, :] = t
            offset += take
        return out

    @staticmethod
    def _resolve_paged_chunk_tokens(q_bhld: Tensor, valid_len: int):
        # Bound temporary K/V concat + score tensors while still preferring
        # larger chunks to avoid many tiny attention kernels.
        batch_size, num_heads, q_len, head_dim = q_bhld.shape
        elem_size = int(max(1, q_bhld.element_size()))
        kv_bytes_per_token = int(batch_size) * int(num_heads) * int(head_dim) * elem_size * 2
        score_bytes_per_token = int(batch_size) * int(num_heads) * int(max(1, q_len)) * elem_size
        bytes_per_token = int(max(1, kv_bytes_per_token + score_bytes_per_token))
        target_bytes = 32 * 1024 * 1024 if torch.is_grad_enabled() else 64 * 1024 * 1024
        token_cap = int(max(1, target_bytes // bytes_per_token))
        return int(max(1, min(int(valid_len), token_cap)))

    def _resolve_train_flashmerge_chunk_tokens(self, q_bhld: Tensor, valid_len: int):
        if self.paged_attn_train_chunk_tokens > 0:
            return int(max(1, min(int(valid_len), int(self.paged_attn_train_chunk_tokens))))
        return self._resolve_paged_chunk_tokens(q_bhld, valid_len)

    @staticmethod
    def _normalize_flash_lse(lse: Tensor, q_len: int):
        # PyTorch flash/efficient kernels return LSE padded to a backend-dependent
        # token multiple (often 16). We only need the first q_len entries.
        if lse.ndim != 3:
            raise ValueError(f"unexpected flash LSE rank: {lse.ndim}")
        lse_trim = lse[:, :, :int(q_len)]
        return lse_trim.unsqueeze(-1).to(dtype=torch.float32)

    @staticmethod
    def _flash_sdpa_chunk_with_lse(q_bhld: Tensor, k_chunk: Tensor, v_chunk: Tensor):
        # Prefer flash backend and fallback to efficient backend if needed.
        try:
            out, lse, *_ = torch.ops.aten._scaled_dot_product_flash_attention.default(
                q_bhld,
                k_chunk,
                v_chunk,
                0.0,
                False,
                False,
            )
            return out, lse
        except Exception:
            out, lse = torch.ops.aten._scaled_dot_product_efficient_attention.default(
                q_bhld,
                k_chunk,
                v_chunk,
                True,
                False,
            )
            return out, lse

    def _get_flash_prefix_stream(self, device):
        if (not bool(self.flash_prefix_async)) or (device.type != "cuda") or (not torch.cuda.is_available()):
            return None
        device_index = device.index if device.index is not None else torch.cuda.current_device()
        stream = self._flash_prefix_streams.get(int(device_index), None)
        if stream is None:
            stream = torch.cuda.Stream(device=device)
            self._flash_prefix_streams[int(device_index)] = stream
        return stream

    def _forward_step_attn_ff_paged_flash_merge(
        self,
        src_step: Tensor,
        q_bhld: Tensor,
        k_pages,
        v_pages,
        valid_len: int,
        chunk_token_cap: int,
        clone_kv_for_grad: bool = False,
    ):
        valid_len = int(valid_len)
        if valid_len <= 0:
            raise ValueError("Paged KV attention requires valid_len > 0.")

        chunk_token_cap = int(max(1, min(valid_len, int(chunk_token_cap))))
        q_len = int(q_bhld.shape[2])
        merged_out = None
        merged_lse = None

        profile_enabled = self._step_profile_enabled_now()
        core_t0 = time.perf_counter() if profile_enabled else None
        consumed = 0
        for k_chunk, v_chunk in self._iter_paged_kv_chunks(
            k_pages,
            v_pages,
            valid_len=valid_len,
            chunk_token_cap=chunk_token_cap,
        ):
            consumed += int(k_chunk.shape[2])
            if bool(clone_kv_for_grad):
                # In-place paged-grad mode mutates cache pages every step.
                # Clone read views so autograd sees stable versions.
                k_chunk = k_chunk.clone()
                v_chunk = v_chunk.clone()

            out_chunk, lse_chunk = self._flash_sdpa_chunk_with_lse(q_bhld, k_chunk, v_chunk)
            lse_chunk_f32 = self._normalize_flash_lse(lse_chunk, q_len=q_len)
            out_chunk_f32 = out_chunk.to(dtype=torch.float32)

            if merged_out is None:
                merged_out = out_chunk_f32
                merged_lse = lse_chunk_f32
            else:
                merged_lse_next = torch.logaddexp(merged_lse, lse_chunk_f32)
                prev_scale = torch.exp(merged_lse - merged_lse_next)
                curr_scale = torch.exp(lse_chunk_f32 - merged_lse_next)
                merged_out = merged_out * prev_scale + out_chunk_f32 * curr_scale
                merged_lse = merged_lse_next

        core_dt = (time.perf_counter() - core_t0) if core_t0 is not None else 0.0
        if int(consumed) != int(valid_len):
            raise ValueError("Paged KV cache has inconsistent valid_len/pages.")
        if merged_out is None:
            raise ValueError("Paged KV attention produced no chunk output.")
        finalize_t0 = time.perf_counter() if profile_enabled else None
        out = self._finalize_forward_step(src_step, merged_out.to(dtype=q_bhld.dtype))
        finalize_dt = (time.perf_counter() - finalize_t0) if finalize_t0 is not None else 0.0
        if profile_enabled:
            stats = self._layer_step_profile_stats
            stats["attn_core_wall_s"] += float(core_dt)
            stats["finalize_wall_s"] += float(finalize_dt)
        return out

    def _forward_step_attn_ff_paged_flash_prefix(
        self,
        src_step: Tensor,
        q_bhld: Tensor,
        k_pages,
        v_pages,
        valid_len: int,
        prefix_k: Optional[Tensor] = None,
        prefix_v: Optional[Tensor] = None,
        clone_kv_for_grad: bool = False,
    ):
        profile_enabled = self._step_profile_enabled_now()
        core_t0 = time.perf_counter() if profile_enabled else None
        valid_len = int(valid_len)
        if valid_len <= 0:
            raise ValueError("Paged KV attention requires valid_len > 0.")
        prefix_len = 0
        if prefix_k is not None and prefix_v is not None:
            prefix_len = int(prefix_k.shape[2])
        if prefix_len >= valid_len:
            raise ValueError("Flash-prefix attention requires prefix_len < valid_len.")

        tail_take = int(valid_len - prefix_len)
        tail_k = k_pages[-1][:, :, :tail_take, :]
        tail_v = v_pages[-1][:, :, :tail_take, :]
        if bool(clone_kv_for_grad):
            tail_k = tail_k.clone()
            tail_v = tail_v.clone()
        q_len = int(q_bhld.shape[2])
        prefix_out = None
        prefix_lse = None
        prefix_stream = None
        if prefix_len > 0:
            prefix_chunk_k = prefix_k
            prefix_chunk_v = prefix_v
            if bool(clone_kv_for_grad):
                prefix_chunk_k = prefix_chunk_k.clone()
                prefix_chunk_v = prefix_chunk_v.clone()
            prefix_stream = self._get_flash_prefix_stream(q_bhld.device)
            if prefix_stream is not None:
                with torch.cuda.stream(prefix_stream):
                    prefix_out, prefix_lse = self._flash_sdpa_chunk_with_lse(q_bhld, prefix_chunk_k, prefix_chunk_v)
            else:
                prefix_out, prefix_lse = self._flash_sdpa_chunk_with_lse(q_bhld, prefix_chunk_k, prefix_chunk_v)

        tail_out, tail_lse = self._flash_sdpa_chunk_with_lse(q_bhld, tail_k, tail_v)
        merged_out = tail_out.to(dtype=torch.float32)
        merged_lse = self._normalize_flash_lse(tail_lse, q_len=q_len)

        if prefix_out is not None and prefix_lse is not None:
            if prefix_stream is not None:
                torch.cuda.current_stream(device=q_bhld.device).wait_stream(prefix_stream)
            prefix_out = prefix_out.to(dtype=torch.float32)
            prefix_lse = self._normalize_flash_lse(prefix_lse, q_len=q_len)
            merged_lse_next = torch.logaddexp(prefix_lse, merged_lse)
            prefix_scale = torch.exp(prefix_lse - merged_lse_next)
            tail_scale = torch.exp(merged_lse - merged_lse_next)
            merged_out = prefix_out * prefix_scale + merged_out * tail_scale

        core_dt = (time.perf_counter() - core_t0) if core_t0 is not None else 0.0
        finalize_t0 = time.perf_counter() if profile_enabled else None
        out = self._finalize_forward_step(src_step, merged_out.to(dtype=q_bhld.dtype))
        finalize_dt = (time.perf_counter() - finalize_t0) if finalize_t0 is not None else 0.0
        if profile_enabled:
            stats = self._layer_step_profile_stats
            stats["attn_core_wall_s"] += float(core_dt)
            stats["finalize_wall_s"] += float(finalize_dt)
        return out

    def _forward_step_attn_ff_paged(
        self,
        src_step: Tensor,
        q_bhld: Tensor,
        k_pages,
        v_pages,
        valid_len: int,
        clone_kv_for_grad: bool = False,
        prefix_k: Optional[Tensor] = None,
        prefix_v: Optional[Tensor] = None,
    ):
        # Online softmax accumulation over KV pages (FlashAttention-style reduction),
        # avoids materializing full concatenated K/V every step.
        valid_len = int(valid_len)
        if valid_len <= 0:
            raise ValueError("Paged KV attention requires valid_len > 0.")
        profile_enabled = self._step_profile_enabled_now()
        stats = self._layer_step_profile_stats if profile_enabled else None
        attn_dropout = float(self.self_attn.dropout) if self.training else 0.0
        train_mode = self.paged_attn_train_mode
        if train_mode == "auto":
            # Keep runtime routing consistent with training heuristic.
            train_mode = "flash_prefix" if int(q_bhld.shape[0]) >= 32 else "dense"
        if q_bhld.device.type != "cuda":
            train_mode = "dense"

        # Fast path: single-page cache can directly dispatch to SDPA.
        # This is important for no-grad/inference runs where we keep one large
        # page and avoid the per-page python loop entirely.
        if len(k_pages) == 1:
            if (
                bool(self.force_flash_single_page)
                and
                torch.is_grad_enabled()
                and attn_dropout <= 0.0
                and train_mode == "flash_prefix"
                and (not bool(clone_kv_for_grad))
            ):
                if stats is not None:
                    stats["paged_path_flash_prefix"] += 1
                return self._forward_step_attn_ff_paged_flash_prefix(
                    src_step,
                    q_bhld,
                    k_pages,
                    v_pages,
                    valid_len,
                    prefix_k=prefix_k,
                    prefix_v=prefix_v,
                    clone_kv_for_grad=False,
                )
            k_all = k_pages[0][:, :, :valid_len, :]
            v_all = v_pages[0][:, :, :valid_len, :]
            if bool(clone_kv_for_grad) and torch.is_grad_enabled():
                # In in-place paged-grad mode, cache pages are mutated every step.
                # Clone read views so backward does not observe version bumps.
                k_all = k_all.clone()
                v_all = v_all.clone()
            if stats is not None:
                stats["paged_path_single_page"] += 1
            return self._forward_step_attn_ff(src_step, q_bhld, k_all, v_all)

        # Training throughput route: dispatch fused SDPA on dense/flash views.
        # This removes many tiny per-page kernels in the grad-enabled hot path.
        if torch.is_grad_enabled():
            if (
                attn_dropout <= 0.0
                and train_mode == "flash_prefix"
                and (not bool(clone_kv_for_grad))
            ):
                dense_token_cap = int(max(0, self.paged_attn_flashprefix_dense_max_tokens))
                if (
                    dense_token_cap > 0
                    and int(valid_len) <= dense_token_cap
                    and (prefix_k is not None)
                    and (prefix_v is not None)
                ):
                    prefix_len = int(prefix_k.shape[2])
                    if prefix_len >= int(valid_len):
                        k_all = prefix_k[:, :, :int(valid_len), :]
                        v_all = prefix_v[:, :, :int(valid_len), :]
                    else:
                        tail_take = int(valid_len) - prefix_len
                        tail_k = k_pages[-1][:, :, :tail_take, :]
                        tail_v = v_pages[-1][:, :, :tail_take, :]
                        k_all = self._concat_dim2([prefix_k, tail_k])
                        v_all = self._concat_dim2([prefix_v, tail_v])
                    if stats is not None:
                        stats["paged_path_dense"] += 1
                    return self._forward_step_attn_ff(src_step, q_bhld, k_all, v_all)
                if stats is not None:
                    stats["paged_path_flash_prefix"] += 1
                return self._forward_step_attn_ff_paged_flash_prefix(
                    src_step,
                    q_bhld,
                    k_pages,
                    v_pages,
                    valid_len,
                    prefix_k=prefix_k,
                    prefix_v=prefix_v,
                    clone_kv_for_grad=False,
                )
            if attn_dropout <= 0.0 and train_mode == "flash_merge":
                if stats is not None:
                    stats["paged_path_flash_merge"] += 1
                return self._forward_step_attn_ff_paged_flash_merge(
                    src_step,
                    q_bhld,
                    k_pages,
                    v_pages,
                    valid_len,
                    chunk_token_cap=self._resolve_train_flashmerge_chunk_tokens(q_bhld, valid_len),
                    clone_kv_for_grad=bool(clone_kv_for_grad),
                )
            if (
                train_mode == "dense"
                and (not bool(clone_kv_for_grad))
                and (prefix_k is not None)
                and (prefix_v is not None)
            ):
                prefix_len = int(prefix_k.shape[2])
                if prefix_len >= int(valid_len):
                    k_all = prefix_k[:, :, :int(valid_len), :]
                    v_all = prefix_v[:, :, :int(valid_len), :]
                else:
                    tail_take = int(valid_len) - prefix_len
                    tail_k = k_pages[-1][:, :, :tail_take, :]
                    tail_v = v_pages[-1][:, :, :tail_take, :]
                    k_all = self._concat_dim2([prefix_k, tail_k])
                    v_all = self._concat_dim2([prefix_v, tail_v])
                if stats is not None:
                    stats["paged_path_dense"] += 1
                return self._forward_step_attn_ff(src_step, q_bhld, k_all, v_all)
            k_all, v_all = self._build_paged_views(k_pages, v_pages, valid_len)
            if bool(clone_kv_for_grad):
                # Multi-page fallback for in-place paged-grad mode.
                k_all = k_all.clone()
                v_all = v_all.clone()
            if stats is not None:
                stats["paged_path_dense"] += 1
            return self._forward_step_attn_ff(src_step, q_bhld, k_all, v_all)

        if attn_dropout > 0.0:
            k_all, v_all = self._build_paged_views(k_pages, v_pages, valid_len)
            if stats is not None:
                stats["paged_path_dense"] += 1
            return self._forward_step_attn_ff(src_step, q_bhld, k_all, v_all)

        scale = 1.0 / math.sqrt(float(q_bhld.shape[-1]))
        running_max = None
        running_denom = None
        running_num = None

        chunk_token_cap = self._resolve_paged_chunk_tokens(q_bhld, valid_len)
        consumed = 0
        for k_chunk, v_chunk in self._iter_paged_kv_chunks(
            k_pages,
            v_pages,
            valid_len=valid_len,
            chunk_token_cap=chunk_token_cap,
        ):
            consumed += int(k_chunk.shape[2])

            scores = torch.matmul(q_bhld, k_chunk.transpose(-2, -1)) * scale
            local_max = scores.amax(dim=-1, keepdim=True)
            local_probs = torch.exp(scores - local_max)
            local_denom = local_probs.sum(dim=-1, keepdim=True)
            local_num = torch.matmul(local_probs, v_chunk)

            if running_num is None:
                running_max = local_max
                running_denom = local_denom
                running_num = local_num
            else:
                merged_max = torch.maximum(running_max, local_max)
                old_scale = torch.exp(running_max - merged_max)
                new_scale = torch.exp(local_max - merged_max)
                running_num = running_num * old_scale + local_num * new_scale
                running_denom = running_denom * old_scale + local_denom * new_scale
                running_max = merged_max

        if int(consumed) != int(valid_len):
            raise ValueError("Paged KV cache has inconsistent valid_len/pages.")
        attn_bhld = running_num / running_denom.clamp_min(1e-12)
        return self._finalize_forward_step(src_step, attn_bhld)

    @staticmethod
    def _cache_capacity_from_tensor(k_tensor: Tensor, requested_max_len: Optional[int]):
        if requested_max_len is None:
            return int(max(1, k_tensor.shape[2]))
        return int(max(1, requested_max_len))

    @staticmethod
    def _cache_views_from_store(k_store: Tensor, v_store: Tensor, valid_len: int):
        k_view = k_store[:, :, :valid_len, :]
        v_view = v_store[:, :, :valid_len, :]
        return k_view, v_view

    @staticmethod
    def _normalize_kv_cache_mode(kv_cache_mode):
        mode = "auto" if kv_cache_mode is None else str(kv_cache_mode).strip().lower()
        if mode not in {"auto", "immutable", "static", "paged"}:
            raise ValueError(
                f"Unknown kv_cache_mode={kv_cache_mode!r}. "
                f"Expected one of: auto, immutable, static, paged."
            )
        return mode

    def _resolve_kv_cache_mode(self, kv_cache, kv_cache_mode, max_cache_len, allow_grad_mutable_cache=False):
        requested_mode = self._normalize_kv_cache_mode(kv_cache_mode)
        if torch.is_grad_enabled() and requested_mode in {"static", "paged"} and not bool(allow_grad_mutable_cache):
            # Preserve training semantics: never use mutable cache updates with autograd enabled.
            requested_mode = "immutable"
        existing_mode = None
        if kv_cache is not None and "cache_mode" in kv_cache:
            existing_mode = self._normalize_kv_cache_mode(kv_cache.get("cache_mode"))
            if existing_mode == "auto":
                existing_mode = None

        if existing_mode is not None:
            if requested_mode not in {"auto", existing_mode}:
                raise ValueError(
                    f"kv_cache_mode mismatch: requested={requested_mode}, existing cache mode={existing_mode}"
                )
            return existing_mode

        if requested_mode == "auto":
            if torch.is_grad_enabled():
                return "immutable"
            return "static" if max_cache_len is not None else "immutable"
        return requested_mode

    def _append_to_kv_store(
        self,
        k_store: Tensor,
        v_store: Tensor,
        valid_len: int,
        k_new_bhld: Tensor,
        v_new_bhld: Tensor,
        requested_max_len: Optional[int],
    ):
        if valid_len >= k_store.shape[2]:
            requested = int(max(1, requested_max_len)) if requested_max_len is not None else None
            if requested is not None and k_store.shape[2] >= requested:
                raise ValueError(f"KV cache capacity exceeded: {valid_len} >= max_cache_len={requested}.")
            new_cap = max(int(k_store.shape[2]) * 2, int(valid_len) + 1)
            if requested is not None:
                new_cap = min(new_cap, requested)
            new_k_store = k_store.new_empty((k_store.shape[0], k_store.shape[1], new_cap, k_store.shape[3]))
            new_v_store = v_store.new_empty((v_store.shape[0], v_store.shape[1], new_cap, v_store.shape[3]))
            if valid_len > 0:
                new_k_store[:, :, :valid_len, :] = k_store[:, :, :valid_len, :]
                new_v_store[:, :, :valid_len, :] = v_store[:, :, :valid_len, :]
            k_store = new_k_store
            v_store = new_v_store

        k_store[:, :, valid_len: valid_len + 1, :] = k_new_bhld
        v_store[:, :, valid_len: valid_len + 1, :] = v_new_bhld
        valid_len = int(valid_len) + 1
        return k_store, v_store, valid_len

    @staticmethod
    def _build_paged_views(k_pages, v_pages, valid_len: int):
        if valid_len <= 0:
            raise ValueError("Paged KV cache requires valid_len > 0")
        remaining = int(valid_len)
        k_chunks = []
        v_chunks = []
        for k_page, v_page in zip(k_pages, v_pages):
            if remaining <= 0:
                break
            cap = int(k_page.shape[2])
            take = min(cap, remaining)
            if take == cap:
                k_chunks.append(k_page)
                v_chunks.append(v_page)
            else:
                k_chunks.append(k_page[:, :, :take, :])
                v_chunks.append(v_page[:, :, :take, :])
            remaining -= take
        if remaining != 0:
            raise ValueError("Paged KV cache has inconsistent valid_len/pages.")
        if len(k_chunks) == 1:
            return k_chunks[0], v_chunks[0]
        return TransformerEncoderLayer._concat_dim2(k_chunks), TransformerEncoderLayer._concat_dim2(v_chunks)

    @staticmethod
    def _iter_paged_kv_chunks(k_pages, v_pages, valid_len: int, chunk_token_cap: int):
        remaining = int(valid_len)
        if remaining <= 0:
            raise ValueError("Paged KV cache requires valid_len > 0")
        cap = int(max(1, chunk_token_cap))
        page_idx = 0
        page_offset = 0
        num_pages = int(len(k_pages))

        while remaining > 0:
            if page_idx >= num_pages:
                raise ValueError("Paged KV cache has inconsistent valid_len/pages.")
            k_chunks = []
            v_chunks = []
            chunk_tokens = 0
            while remaining > 0 and chunk_tokens < cap:
                if page_idx >= num_pages:
                    raise ValueError("Paged KV cache has inconsistent valid_len/pages.")
                k_page = k_pages[page_idx]
                v_page = v_pages[page_idx]
                page_cap = int(k_page.shape[2])
                if page_offset >= page_cap:
                    page_idx += 1
                    page_offset = 0
                    continue
                take = min(page_cap - page_offset, remaining, cap - chunk_tokens)
                end = int(page_offset + take)
                if page_offset == 0 and end == page_cap:
                    k_chunks.append(k_page)
                    v_chunks.append(v_page)
                else:
                    k_chunks.append(k_page[:, :, page_offset:end, :])
                    v_chunks.append(v_page[:, :, page_offset:end, :])
                chunk_tokens += int(take)
                remaining -= int(take)
                page_offset = end
                if page_offset >= page_cap:
                    page_idx += 1
                    page_offset = 0

            if not k_chunks:
                raise ValueError("Paged KV chunk iterator produced empty chunk.")
            if len(k_chunks) == 1:
                yield k_chunks[0], v_chunks[0]
            else:
                yield TransformerEncoderLayer._concat_dim2(k_chunks), TransformerEncoderLayer._concat_dim2(v_chunks)

    @staticmethod
    def _append_to_kv_pages(
        k_pages,
        v_pages,
        valid_len: int,
        k_new_bhld: Tensor,
        v_new_bhld: Tensor,
        page_size: int,
        max_cache_len: int,
    ):
        if valid_len >= max_cache_len:
            raise ValueError(f"KV cache capacity exceeded: {valid_len} >= max_cache_len={max_cache_len}.")

        remaining_idx = int(valid_len)
        page_idx = None
        offset = None
        for idx, k_page in enumerate(k_pages):
            cap = int(k_page.shape[2])
            if remaining_idx < cap:
                page_idx = idx
                offset = remaining_idx
                break
            remaining_idx -= cap

        if page_idx is None:
            used_cap = sum(int(page.shape[2]) for page in k_pages)
            remaining_cap = int(max_cache_len) - used_cap
            if remaining_cap <= 0:
                raise ValueError(f"KV cache capacity exceeded: {valid_len} >= max_cache_len={max_cache_len}.")
            page_cap = min(int(page_size), int(remaining_cap))
            k_page = k_new_bhld.new_empty((k_new_bhld.shape[0], k_new_bhld.shape[1], page_cap, k_new_bhld.shape[3]))
            v_page = v_new_bhld.new_empty((v_new_bhld.shape[0], v_new_bhld.shape[1], page_cap, v_new_bhld.shape[3]))
            k_pages.append(k_page)
            v_pages.append(v_page)
            page_idx = len(k_pages) - 1
            offset = 0

        k_pages[page_idx][:, :, offset: offset + 1, :] = k_new_bhld
        v_pages[page_idx][:, :, offset: offset + 1, :] = v_new_bhld
        return k_pages, v_pages, int(valid_len) + 1

    @staticmethod
    def _append_to_kv_pages_cow(
        k_pages,
        v_pages,
        valid_len: int,
        k_new_bhld: Tensor,
        v_new_bhld: Tensor,
        page_size: int,
        max_cache_len: int,
        freeze_existing_tail: bool = False,
    ):
        """
        Copy-on-write append for paged KV cache.
        No in-place mutation of previous page tensors or page lists so
        reentrant checkpoint can safely replay backward.
        """
        if valid_len >= max_cache_len:
            raise ValueError(f"KV cache capacity exceeded: {valid_len} >= max_cache_len={max_cache_len}.")
        new_k_pages = list(k_pages)
        new_v_pages = list(v_pages)

        total_cap = sum(int(page.shape[2]) for page in k_pages)
        has_preallocated_slack = int(total_cap) > int(valid_len)

        if has_preallocated_slack:
            # Compatibility path for caches converted from dense tensors:
            # keep the old offset-based COW semantics.
            remaining_idx = int(valid_len)
            page_idx = None
            offset = None
            for idx, k_page in enumerate(k_pages):
                cap = int(k_page.shape[2])
                if remaining_idx < cap:
                    page_idx = idx
                    offset = remaining_idx
                    break
                remaining_idx -= cap
            if page_idx is None:
                raise ValueError("Paged KV cache has inconsistent valid_len/pages.")
            old_k_page = k_pages[page_idx]
            old_v_page = v_pages[page_idx]
            k_page = old_k_page.clone()
            v_page = old_v_page.clone()
            k_page[:, :, offset: offset + 1, :] = k_new_bhld
            v_page[:, :, offset: offset + 1, :] = v_new_bhld
            new_k_pages[page_idx] = k_page
            new_v_pages[page_idx] = v_page
            return new_k_pages, new_v_pages, int(valid_len) + 1

        if bool(freeze_existing_tail):
            # TBPTT detach can leave a long immutable tail page. Start a fresh
            # mutable page for the new window to avoid repeatedly copying that
            # detached history on every COW append.
            new_k_pages.append(k_new_bhld.clone())
            new_v_pages.append(v_new_bhld.clone())
            return new_k_pages, new_v_pages, int(valid_len) + 1

        # Packed-growth path: pages only store filled tokens.
        # This avoids cloning full preallocated page capacity on every append.
        if new_k_pages:
            last_cap = int(new_k_pages[-1].shape[2])
            if last_cap < int(page_size):
                prev_k = new_k_pages[-1]
                prev_v = new_v_pages[-1]
                grown_k = TransformerEncoderLayer._concat_dim2((prev_k, k_new_bhld))
                grown_v = TransformerEncoderLayer._concat_dim2((prev_v, v_new_bhld))
                new_k_pages[-1] = grown_k
                new_v_pages[-1] = grown_v
                return new_k_pages, new_v_pages, int(valid_len) + 1

        new_k_pages.append(k_new_bhld.clone())
        new_v_pages.append(v_new_bhld.clone())
        return new_k_pages, new_v_pages, int(valid_len) + 1

    @staticmethod
    def _paged_views_are_packed(k_pages, valid_len: int):
        """
        Return True when every stored page has no preallocated slack
        (i.e. total page capacity == valid_len).
        """
        if valid_len <= 0:
            return False
        total_cap = 0
        for page in k_pages:
            total_cap += int(page.shape[2])
        return int(total_cap) == int(valid_len)

    @staticmethod
    def _append_to_kv_pages_cow_packed(
        k_pages,
        v_pages,
        valid_len: int,
        k_new_bhld: Tensor,
        v_new_bhld: Tensor,
        page_size: int,
        max_cache_len: int,
        freeze_existing_tail: bool = False,
    ):
        """
        Fast COW append for packed pages (no preallocated slack pages).
        Avoids page-scan and capacity-scan in the hot rollout path.
        """
        if valid_len >= max_cache_len:
            raise ValueError(f"KV cache capacity exceeded: {valid_len} >= max_cache_len={max_cache_len}.")
        new_k_pages = list(k_pages)
        new_v_pages = list(v_pages)

        if bool(freeze_existing_tail):
            new_k_pages.append(k_new_bhld.clone())
            new_v_pages.append(v_new_bhld.clone())
            return new_k_pages, new_v_pages, int(valid_len) + 1

        if new_k_pages:
            prev_k = new_k_pages[-1]
            prev_v = new_v_pages[-1]
            prev_len = int(prev_k.shape[2])
            if prev_len < int(page_size):
                grown_k = TransformerEncoderLayer._concat_dim2((prev_k, k_new_bhld))
                grown_v = TransformerEncoderLayer._concat_dim2((prev_v, v_new_bhld))
                new_k_pages[-1] = grown_k
                new_v_pages[-1] = grown_v
                return new_k_pages, new_v_pages, int(valid_len) + 1

        new_k_pages.append(k_new_bhld.clone())
        new_v_pages.append(v_new_bhld.clone())
        return new_k_pages, new_v_pages, int(valid_len) + 1

    def forward_step(
        self,
        src_step: Tensor,
        kv_cache: Optional[dict] = None,
        append_to_cache: bool = True,
        max_cache_len: Optional[int] = None,
        kv_cache_mode: str = "auto",
        kv_cache_page_size: Optional[int] = None,
        allow_grad_mutable_cache: bool = False,
        allow_grad_inplace_paged_cache: bool = False,
    ):
        """
        Incremental forward for a single token (seq length = 1) with KV cache.
        Only valid under single-directional (causal) single-eval semantics.
        """
        if not self.single_eval_causal:
            raise ValueError("forward_step requires single_eval_causal=True.")
        if src_step.ndim == 3:
            if src_step.shape[0] != 1:
                raise ValueError(f"src_step must have shape (1, B, E), got {tuple(src_step.shape)}")
        elif src_step.ndim != 2:
            raise ValueError(f"src_step must have shape (1, B, E) or (B, E), got {tuple(src_step.shape)}")
        profile_enabled = self._step_profile_enabled_now()
        total_t0 = time.perf_counter() if profile_enabled else None

        if self.pre_norm:
            src_norm = self.norm1(src_step)
        else:
            src_norm = src_step

        if src_norm.ndim == 3:
            src_norm_bld = src_norm.permute(1, 0, 2)  # (B, 1, E)
        else:
            src_norm_bld = src_norm.unsqueeze(1)  # (B, 1, E)
        proj_t0 = time.perf_counter() if profile_enabled else None
        if append_to_cache:
            q_bld, k_new_bld, v_new_bld = self._project_qkv(src_norm_bld)
            q_bhld = self._split_heads(q_bld)
            k_new_bhld = self._split_heads(k_new_bld)
            v_new_bhld = self._split_heads(v_new_bld)
        else:
            q_bld = self._project_q(src_norm_bld)
            q_bhld = self._split_heads(q_bld)
        proj_dt = (time.perf_counter() - proj_t0) if proj_t0 is not None else 0.0
        if max_cache_len is None and kv_cache is not None:
            max_cache_len = kv_cache.get("max_cache_len", None)
        if kv_cache_page_size is None and kv_cache is not None:
            kv_cache_page_size = kv_cache.get("page_size", None)

        if kv_cache is not None and "allow_grad_mutable_cache" in kv_cache:
            allow_grad_mutable_cache = bool(kv_cache.get("allow_grad_mutable_cache"))
        if kv_cache is not None and "allow_grad_inplace_paged_cache" in kv_cache:
            allow_grad_inplace_paged_cache = bool(kv_cache.get("allow_grad_inplace_paged_cache"))
        prefix_compact_pages = bool(self.paged_prefix_compact_pages)
        if kv_cache is not None and "prefix_compact_pages" in kv_cache:
            prefix_compact_pages = bool(kv_cache.get("prefix_compact_pages"))
        cache_mode = self._resolve_kv_cache_mode(
            kv_cache,
            kv_cache_mode,
            max_cache_len,
            allow_grad_mutable_cache=allow_grad_mutable_cache,
        )
        if cache_mode in {"static", "paged"} and max_cache_len is None:
            raise ValueError(f"kv_cache_mode={cache_mode} requires max_cache_len.")
        if max_cache_len is not None:
            max_cache_len = int(max_cache_len)
        runtime_paged_train_mode = self.paged_attn_train_mode
        if runtime_paged_train_mode == "auto":
            runtime_paged_train_mode = "flash_prefix" if int(q_bhld.shape[0]) >= 32 else "dense"
        if cache_mode == "paged":
            if kv_cache_page_size is None:
                kv_cache_page_size = 128
            kv_cache_page_size = int(max(1, kv_cache_page_size))
        mutable_paged_grad = (
            cache_mode == "paged"
            and torch.is_grad_enabled()
            and bool(allow_grad_mutable_cache)
        )
        inplace_paged_grad = bool(mutable_paged_grad and bool(allow_grad_inplace_paged_cache))
        if cache_mode == "paged":
            # Throughput route:
            # - no-grad path (including reentrant checkpoint forward pass) uses
            #   a single dense page to avoid tiny-page python overhead;
            # - grad + mutable path keeps bounded page size but avoids
            #   pathological page_size=1 behavior.
            if (not torch.is_grad_enabled()) and (max_cache_len is not None):
                effective_kv_page_size = int(max(1, max_cache_len))
            elif mutable_paged_grad:
                if inplace_paged_grad and (max_cache_len is not None):
                    # Keep one dense page in in-place mode so append stays
                    # O(1) and avoids page-growth cat churn in TBPTT rollout.
                    effective_kv_page_size = int(max(1, max_cache_len))
                else:
                    if runtime_paged_train_mode == "flash_prefix":
                        # In COW mode, large pages amplify per-step cat/clone
                        # growth on the mutable tail page. Use a smaller
                        # training page only for flash-prefix mode to reduce
                        # this copy overhead while keeping page-locality.
                        target_page = int(max(8, self.paged_attn_flashprefix_page_size))
                        effective_kv_page_size = int(max(8, min(int(kv_cache_page_size), target_page)))
                    elif runtime_paged_train_mode == "dense":
                        # Dense paged-attn still rebuilds multi-page views per step.
                        # Cap mutable COW page growth to reduce tail-page cat cost.
                        target_page = int(max(8, self.paged_attn_dense_page_size))
                        effective_kv_page_size = int(max(8, min(int(kv_cache_page_size), target_page)))
                    else:
                        effective_kv_page_size = int(max(8, kv_cache_page_size))
            else:
                effective_kv_page_size = kv_cache_page_size
        else:
            effective_kv_page_size = kv_cache_page_size

        cache_t0 = time.perf_counter() if profile_enabled else None
        paged_packed = False
        if cache_mode != "paged":
            prefix_compact_pages = False
        if kv_cache is None:
            if not append_to_cache:
                raise ValueError("predict-only step requires a non-empty kv_cache.")
            tail_frozen = False
            k_prefix = None
            v_prefix = None
            prefix_pages = 0
            if cache_mode == "static":
                cap = self._cache_capacity_from_tensor(k_new_bhld, max_cache_len)
                k_store = k_new_bhld.new_empty((k_new_bhld.shape[0], k_new_bhld.shape[1], cap, k_new_bhld.shape[3]))
                v_store = v_new_bhld.new_empty((v_new_bhld.shape[0], v_new_bhld.shape[1], cap, v_new_bhld.shape[3]))
                k_store[:, :, :1, :] = k_new_bhld
                v_store[:, :, :1, :] = v_new_bhld
                valid_len = 1
                k_all, v_all = self._cache_views_from_store(k_store, v_store, valid_len)
                k_pages = None
                v_pages = None
            elif cache_mode == "paged":
                if torch.is_grad_enabled():
                    if inplace_paged_grad:
                        page_cap = min(int(effective_kv_page_size), int(max_cache_len))
                    else:
                        # Start mutable COW pages with one filled token; page growth
                        # happens via explicit preallocated copy in _append_to_kv_pages_cow.
                        page_cap = 1
                else:
                    page_cap = min(int(effective_kv_page_size), int(max_cache_len))
                k_page = k_new_bhld.new_empty((k_new_bhld.shape[0], k_new_bhld.shape[1], page_cap, k_new_bhld.shape[3]))
                v_page = v_new_bhld.new_empty((v_new_bhld.shape[0], v_new_bhld.shape[1], page_cap, v_new_bhld.shape[3]))
                k_page[:, :, :1, :] = k_new_bhld
                v_page[:, :, :1, :] = v_new_bhld
                k_pages = [k_page]
                v_pages = [v_page]
                valid_len = 1
                k_all = None
                v_all = None
                paged_packed = bool(torch.is_grad_enabled() and (not inplace_paged_grad))
                k_store = None
                v_store = None
            else:
                k_all = k_new_bhld
                v_all = v_new_bhld
                k_store = None
                v_store = None
                k_pages = None
                v_pages = None
                valid_len = int(k_all.shape[2])
        else:
            k_prev = kv_cache.get("k", None)
            v_prev = kv_cache.get("v", None)
            k_store = kv_cache.get("k_store", None)
            v_store = kv_cache.get("v_store", None)
            k_pages = kv_cache.get("k_pages", None)
            v_pages = kv_cache.get("v_pages", None)
            k_prefix = kv_cache.get("k_prefix", None)
            v_prefix = kv_cache.get("v_prefix", None)
            tail_frozen = bool(kv_cache.get("tail_frozen", False))
            paged_packed = bool(kv_cache.get("paged_packed", False))
            try:
                prefix_pages = int(kv_cache.get("prefix_pages", 0))
            except Exception:
                prefix_pages = 0
            if k_prev is None:
                valid_len = int(kv_cache.get("valid_len", 0))
            else:
                valid_len = int(kv_cache.get("valid_len", k_prev.shape[2]))
            if append_to_cache:
                if cache_mode == "static":
                    if k_store is None or v_store is None:
                        if k_prev is None or v_prev is None:
                            raise ValueError("static cache requires dense k/v tensor when k_store/v_store are missing.")
                        cap = self._cache_capacity_from_tensor(k_prev, max_cache_len)
                        cap = max(cap, int(valid_len) + 1)
                        k_store = k_prev.new_empty((k_prev.shape[0], k_prev.shape[1], cap, k_prev.shape[3]))
                        v_store = v_prev.new_empty((v_prev.shape[0], v_prev.shape[1], cap, v_prev.shape[3]))
                        if valid_len > 0:
                            k_store[:, :, :valid_len, :] = k_prev[:, :, :valid_len, :]
                            v_store[:, :, :valid_len, :] = v_prev[:, :, :valid_len, :]
                    k_store, v_store, valid_len = self._append_to_kv_store(
                        k_store=k_store,
                        v_store=v_store,
                        valid_len=valid_len,
                        k_new_bhld=k_new_bhld,
                        v_new_bhld=v_new_bhld,
                        requested_max_len=max_cache_len,
                    )
                    k_all, v_all = self._cache_views_from_store(k_store, v_store, valid_len)
                    k_pages = None
                    v_pages = None
                elif cache_mode == "paged":
                    if k_pages is None or v_pages is None:
                        if k_prev is None or v_prev is None:
                            raise ValueError("paged cache requires either k_pages/v_pages or dense k/v tensors.")
                        k_pages = []
                        v_pages = []
                        remaining = int(valid_len)
                        offset = 0
                        while remaining > 0:
                            page_cap = min(int(effective_kv_page_size), int(max_cache_len) - offset)
                            take = min(page_cap, remaining)
                            k_page = k_prev.new_empty((k_prev.shape[0], k_prev.shape[1], page_cap, k_prev.shape[3]))
                            v_page = v_prev.new_empty((v_prev.shape[0], v_prev.shape[1], page_cap, v_prev.shape[3]))
                            k_page[:, :, :take, :] = k_prev[:, :, offset: offset + take, :]
                            v_page[:, :, :take, :] = v_prev[:, :, offset: offset + take, :]
                            k_pages.append(k_page)
                            v_pages.append(v_page)
                            offset += take
                            remaining -= take
                        paged_packed = self._paged_views_are_packed(k_pages, valid_len)
                    if torch.is_grad_enabled() and (not inplace_paged_grad):
                        if bool(paged_packed):
                            k_pages, v_pages, valid_len = self._append_to_kv_pages_cow_packed(
                                k_pages=k_pages,
                                v_pages=v_pages,
                                valid_len=valid_len,
                                k_new_bhld=k_new_bhld,
                                v_new_bhld=v_new_bhld,
                                page_size=effective_kv_page_size,
                                max_cache_len=max_cache_len,
                                freeze_existing_tail=bool(tail_frozen),
                            )
                        else:
                            k_pages, v_pages, valid_len = self._append_to_kv_pages_cow(
                                k_pages=k_pages,
                                v_pages=v_pages,
                                valid_len=valid_len,
                                k_new_bhld=k_new_bhld,
                                v_new_bhld=v_new_bhld,
                                page_size=effective_kv_page_size,
                                max_cache_len=max_cache_len,
                                freeze_existing_tail=bool(tail_frozen),
                            )
                            paged_packed = self._paged_views_are_packed(k_pages, valid_len)
                        tail_frozen = False
                    else:
                        k_pages, v_pages, valid_len = self._append_to_kv_pages(
                            k_pages=k_pages,
                            v_pages=v_pages,
                            valid_len=valid_len,
                            k_new_bhld=k_new_bhld,
                            v_new_bhld=v_new_bhld,
                            page_size=effective_kv_page_size,
                            max_cache_len=max_cache_len,
                        )
                        tail_frozen = False
                        paged_packed = False
                    k_all = None
                    v_all = None
                    k_store = None
                    v_store = None
                else:
                    k_all = self._concat_dim2([k_prev, k_new_bhld])
                    v_all = self._concat_dim2([v_prev, v_new_bhld])
                    k_store = None
                    v_store = None
                    k_pages = None
                    v_pages = None
                    valid_len = int(k_all.shape[2])
            else:
                if cache_mode == "static" and (k_store is not None) and (v_store is not None):
                    k_all, v_all = self._cache_views_from_store(k_store, v_store, valid_len)
                    k_pages = None
                    v_pages = None
                elif cache_mode == "paged" and (k_pages is not None) and (v_pages is not None):
                    k_all = None
                    v_all = None
                    k_store = None
                    v_store = None
                else:
                    if k_prev is None or v_prev is None:
                        raise ValueError("predict-only step requires dense k/v tensors or paged cache pages.")
                    k_all = k_prev
                    v_all = v_prev
                    k_store = None
                    v_store = None
                    k_pages = None
                    v_pages = None
                    valid_len = int(k_all.shape[2])

        prefix_train_mode = runtime_paged_train_mode if bool(self.paged_prefix_auto_route) else self.paged_attn_train_mode
        if (
            cache_mode == "paged"
            and torch.is_grad_enabled()
            and (not bool(inplace_paged_grad))
            and prefix_train_mode in {"flash_prefix", "dense"}
            and (k_pages is not None)
            and (v_pages is not None)
        ):
            # Maintain concatenated full-page prefix so hot path attention can
            # consume [prefix + tail] instead of rebuilding all-page cat views.
            full_pages = int(max(0, len(k_pages) - 1))
            if full_pages <= 0:
                k_prefix = None
                v_prefix = None
                prefix_pages = 0
            else:
                rebuild_prefix = (
                    (k_prefix is None)
                    or (v_prefix is None)
                    or (prefix_pages <= 0)
                    or (full_pages < prefix_pages)
                )
                if rebuild_prefix:
                    if full_pages == 1:
                        k_prefix = k_pages[0]
                        v_prefix = v_pages[0]
                    else:
                        k_prefix = self._concat_dim2(k_pages[:full_pages])
                        v_prefix = self._concat_dim2(v_pages[:full_pages])
                    prefix_pages = full_pages
                elif full_pages > prefix_pages:
                    for idx in range(prefix_pages, full_pages):
                        k_full = k_pages[idx]
                        v_full = v_pages[idx]
                        if k_prefix is None or v_prefix is None:
                            k_prefix = k_full
                            v_prefix = v_full
                        else:
                            k_prefix = self._concat_dim2([k_prefix, k_full])
                            v_prefix = self._concat_dim2([v_prefix, v_full])
                    prefix_pages = full_pages
            if (
                bool(prefix_compact_pages)
                and (k_prefix is not None)
                and (v_prefix is not None)
                and len(k_pages) > 1
            ):
                # Compact representation: keep full history in a single prefix page
                # plus one mutable tail page. This removes redundant storage of
                # all historical full pages in k_pages/v_pages.
                prefix_len = int(k_prefix.shape[2])
                tail_take = int(max(0, int(valid_len) - prefix_len))
                if tail_take > 0:
                    tail_k = k_pages[-1][:, :, :tail_take, :]
                    tail_v = v_pages[-1][:, :, :tail_take, :]
                    k_pages = [k_prefix, tail_k]
                    v_pages = [v_prefix, tail_v]
                else:
                    k_pages = [k_prefix]
                    v_pages = [v_prefix]
                prefix_pages = 1
        else:
            k_prefix = None
            v_prefix = None
            prefix_pages = 0

        cache_dt = (time.perf_counter() - cache_t0) if cache_t0 is not None else 0.0
        attnff_t0 = time.perf_counter() if profile_enabled else None
        if cache_mode == "paged" and (k_pages is not None) and (v_pages is not None):
            paged_recompute_active = bool(
                self.recompute_attn
                and self.paged_recompute_attn
                and torch.is_grad_enabled()
                and (not bool(inplace_paged_grad))
                and (not _is_torch_compiling())
            )
            if paged_recompute_active:
                # Snapshot page references so checkpoint replay reads the same
                # immutable page tensors even if outer cache containers advance.
                k_pages_snapshot = tuple(k_pages)
                v_pages_snapshot = tuple(v_pages)

                def _paged_attn_ff_ckpt(src_in: Tensor, q_in: Tensor):
                    return self._forward_step_attn_ff_paged(
                        src_in,
                        q_in,
                        k_pages_snapshot,
                        v_pages_snapshot,
                        valid_len,
                        clone_kv_for_grad=False,
                        prefix_k=k_prefix,
                        prefix_v=v_prefix,
                    )

                src = checkpoint(
                    _paged_attn_ff_ckpt,
                    src_step,
                    q_bhld,
                    use_reentrant=False,
                )
                if profile_enabled:
                    self._layer_step_profile_stats["paged_recompute_calls"] += 1
            else:
                src = self._forward_step_attn_ff_paged(
                    src_step,
                    q_bhld,
                    k_pages,
                    v_pages,
                    valid_len,
                    clone_kv_for_grad=bool(inplace_paged_grad),
                    prefix_k=k_prefix,
                    prefix_v=v_prefix,
                )
        elif self.recompute_attn and torch.is_grad_enabled():
            src = checkpoint(
                self._forward_step_attn_ff,
                src_step,
                q_bhld,
                k_all,
                v_all,
                use_reentrant=True,
            )
        else:
            src = self._forward_step_attn_ff(src_step, q_bhld, k_all, v_all)
        attnff_dt = (time.perf_counter() - attnff_t0) if attnff_t0 is not None else 0.0

        store_dense_paged_views = bool(
            (cache_mode == "paged")
            and (k_pages is None or v_pages is None)
        )
        cache_k = (k_all if (cache_mode != "paged" or store_dense_paged_views) else None)
        cache_v = (v_all if (cache_mode != "paged" or store_dense_paged_views) else None)
        reuse_cache_container = bool(
            self.cache_container_reuse
            and isinstance(kv_cache, dict)
            and bool(append_to_cache)
            and bool(allow_grad_mutable_cache)
        )
        if reuse_cache_container:
            new_cache = kv_cache
            new_cache["k"] = cache_k
            new_cache["v"] = cache_v
            new_cache["cache_mode"] = cache_mode
            new_cache["max_cache_len"] = int(max_cache_len) if max_cache_len is not None else None
            new_cache["k_store"] = k_store
            new_cache["v_store"] = v_store
            new_cache["k_pages"] = k_pages
            new_cache["v_pages"] = v_pages
            new_cache["page_size"] = int(effective_kv_page_size) if effective_kv_page_size is not None else None
            new_cache["valid_len"] = int(valid_len)
            new_cache["k_prefix"] = k_prefix
            new_cache["v_prefix"] = v_prefix
            new_cache["prefix_pages"] = int(prefix_pages)
            new_cache["allow_grad_mutable_cache"] = bool(allow_grad_mutable_cache)
            new_cache["allow_grad_inplace_paged_cache"] = bool(allow_grad_inplace_paged_cache)
            new_cache["tail_frozen"] = bool(tail_frozen) if cache_mode == "paged" else False
            new_cache["paged_packed"] = bool(paged_packed) if cache_mode == "paged" else False
            new_cache["prefix_compact_pages"] = bool(prefix_compact_pages) if cache_mode == "paged" else False
        else:
            new_cache = {
                "k": cache_k,
                "v": cache_v,
                "cache_mode": cache_mode,
                "max_cache_len": int(max_cache_len) if max_cache_len is not None else None,
                "k_store": k_store,
                "v_store": v_store,
                "k_pages": k_pages,
                "v_pages": v_pages,
                "page_size": int(effective_kv_page_size) if effective_kv_page_size is not None else None,
                "valid_len": int(valid_len),
                "k_prefix": k_prefix,
                "v_prefix": v_prefix,
                "prefix_pages": int(prefix_pages),
                "allow_grad_mutable_cache": bool(allow_grad_mutable_cache),
                "allow_grad_inplace_paged_cache": bool(allow_grad_inplace_paged_cache),
                "tail_frozen": bool(tail_frozen) if cache_mode == "paged" else False,
                "paged_packed": bool(paged_packed) if cache_mode == "paged" else False,
                "prefix_compact_pages": bool(prefix_compact_pages) if cache_mode == "paged" else False,
            }
        if total_t0 is not None:
            stats = self._layer_step_profile_stats
            stats["calls"] += 1
            stats["proj_wall_s"] += float(proj_dt)
            stats["cache_wall_s"] += float(cache_dt)
            stats["attnff_wall_s"] += float(attnff_dt)
            stats["total_wall_s"] += float(time.perf_counter() - total_t0)
        return src, new_cache

    def forward_causal_prefix(self, src_prefix: Tensor):
        """
        Process a training prefix in one pass under causal attention and
        return both transformed output and KV cache for this layer.
        """
        if not self.single_eval_causal:
            raise ValueError("forward_causal_prefix requires single_eval_causal=True.")
        if src_prefix.ndim != 3:
            raise ValueError(f"src_prefix must have shape (T, B, E), got {tuple(src_prefix.shape)}")
        if src_prefix.shape[0] == 0:
            raise ValueError("src_prefix must contain at least one token.")

        if self.pre_norm:
            src_norm = self.norm1(src_prefix)
        else:
            src_norm = src_prefix

        src_norm_bld = src_norm.permute(1, 0, 2)  # (B, T, E)
        q_bld = self._project_q(src_norm_bld)
        k_bld, v_bld = self._project_kv(src_norm_bld)
        k_bhld = self._split_heads(k_bld)
        v_bhld = self._split_heads(v_bld)

        t_len = src_prefix.shape[0]
        causal_mask = torch.ones((t_len, t_len), device=src_prefix.device, dtype=torch.bool).triu(1)
        # Use the same MHA path as full forward for exact causal-mask semantics.
        attn_out = self.self_attn(
            src_norm_bld,
            src_norm_bld,
            src_norm_bld,
            attn_mask=causal_mask,
            is_causal=False,
            need_weights=False,
        )[0]
        src2 = attn_out.permute(1, 0, 2)

        src = src_prefix + self.dropout1(src2)
        if not self.pre_norm:
            src = self.norm1(src)

        if self.pre_norm:
            src_ff = self.norm2(src)
        else:
            src_ff = src
        src2_ff = self.linear2(self.dropout(self.activation(self.linear1(src_ff))))
        src = src + self.dropout2(src2_ff)

        if not self.pre_norm:
            src = self.norm2(src)

        cache = {"k": k_bhld, "v": v_bhld}
        return src, cache

    def forward_query(self, src_query: Tensor, kv_cache: dict):
        """
        Process query tokens in one pass against fixed training KV cache.
        Query-query interaction is disabled (only attends to cached train keys).
        """
        if not self.single_eval_causal:
            raise ValueError("forward_query requires single_eval_causal=True.")
        if src_query.ndim != 3:
            raise ValueError(f"src_query must have shape (Tq, B, E), got {tuple(src_query.shape)}")
        if src_query.shape[0] == 0:
            return src_query

        if self.pre_norm:
            src_norm = self.norm1(src_query)
        else:
            src_norm = src_query

        src_norm_bld = src_norm.permute(1, 0, 2)  # (B, Tq, E)
        q_bld = self._project_q(src_norm_bld)
        q_bhld = self._split_heads(q_bld)
        k_pages = kv_cache.get("k_pages", None)
        v_pages = kv_cache.get("v_pages", None)
        if (k_pages is not None) and (v_pages is not None):
            valid_len = int(kv_cache.get("valid_len", 0))
            k_bhld, v_bhld = self._build_paged_views(k_pages, v_pages, valid_len)
        else:
            k_bhld = kv_cache["k"]
            v_bhld = kv_cache["v"]

        attn_dropout = float(self.self_attn.dropout) if self.training else 0.0
        attn_bhld = F.scaled_dot_product_attention(
            q_bhld,
            k_bhld,
            v_bhld,
            attn_mask=None,
            dropout_p=attn_dropout,
            is_causal=False,
        )
        attn_bld = self._merge_heads(attn_bhld)
        attn_bld = F.linear(attn_bld, self.self_attn.out_proj.weight, self.self_attn.out_proj.bias)
        src2 = attn_bld.permute(1, 0, 2)

        src = src_query + self.dropout1(src2)
        if not self.pre_norm:
            src = self.norm1(src)

        if self.pre_norm:
            src_ff = self.norm2(src)
        else:
            src_ff = src
        src2_ff = self.linear2(self.dropout(self.activation(self.linear1(src_ff))))
        src = src + self.dropout2(src2_ff)

        if not self.pre_norm:
            src = self.norm2(src)
        return src

    def forward(
        self, 
        src: Tensor, 
        src_mask: Optional[Tensor] = None, 
    ) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).

        Shape:
            see the docs in Transformer class.
        """
        if self.pre_norm:
            src_ = self.norm1(src)
        else:
            src_ = src

        if isinstance(src_mask, tuple):
            return NotImplementedError
        elif isinstance(src_mask, int):
            single_eval_position = src_mask
            train_mask = None
            test_mask = None
            test_is_causal = False

            # split the training and testing samples
            src_train = src_[:single_eval_position]
            src_test = src_[single_eval_position:]
            # since we set batch_first = True, the shape of src_ is (batch, seq, feature)
            src_train = src_train.permute(1, 0, 2)
            src_test = src_test.permute(1, 0, 2)
            if self.single_eval_causal:
                train_len = src_train.shape[1]
                train_mask = torch.ones(
                    (train_len, train_len),
                    device=src_train.device,
                    dtype=torch.bool,
                ).triu(1)

            # the training samples are only attend to themselves
            src_left = self.self_attn(
                src_train, 
                src_train, 
                src_train, 
                attn_mask=train_mask,
                is_causal=False,
                need_weights=False,
            )[0]

            # the testing samples attend to training samples
            src_right = self.self_attn(
                src_test, 
                src_train, 
                src_train,
                attn_mask=test_mask,
                is_causal=test_is_causal,
                need_weights=False,
            )[0]

            # permute them back to (seq, batch, feature)
            src_left = src_left.permute(1, 0, 2)
            src_right = src_right.permute(1, 0, 2)
            src2 = torch.cat([src_left, src_right], dim=0)
        else:
            if self.recompute_attn:
                # this might have some problems, double check
                # https://github.com/pytorch/pytorch/issues/99282

                src2 = checkpoint(
                    self.self_attn, 
                    src_, # query: Tensor,
                    src_, # key: Tensor,
                    src_, # value: Tensor,
                    None, # key_padding_mask: Optional[Tensor] = None,
                    False, # need_weights: bool = True,
                    src_mask, # attn_mask: Optional[Tensor] = None,
                    True, # average_attn_weights: bool = True,
                    False, # is_causal : bool = False) -> Tuple[Tensor, Optional[Tensor]]:
                    use_reentrant=True,
                )[0]
            else:
                src2 = self.self_attn(
                    query = src_, 
                    key = src_, 
                    value = src_, 
                    attn_mask = src_mask,
                    is_causal = False,
                    need_weights=False,
                )[0]
        
        # residual connection
        src = src + self.dropout1(src2)
        if not self.pre_norm:
            src = self.norm1(src)

        if self.pre_norm:
            src_ = self.norm2(src)
        else:
            src_ = src
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src_))))
        src = src + self.dropout2(src2)

        if not self.pre_norm:
            src = self.norm2(src)
        return src


class TransformerEncoderSimple(Module):
    r"""TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer_creator: a function generating objects of TransformerEncoderLayer class without args (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer_creator, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([encoder_layer_creator() for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def consume_step_profile(self):
        calls = 0
        proj_wall_s = 0.0
        cache_wall_s = 0.0
        attnff_wall_s = 0.0
        attn_core_wall_s = 0.0
        finalize_wall_s = 0.0
        finalize_attn_outproj_wall_s = 0.0
        finalize_ffn_wall_s = 0.0
        paged_path_single_page = 0
        paged_path_flash_prefix = 0
        paged_path_flash_merge = 0
        paged_path_dense = 0
        paged_recompute_calls = 0
        total_wall_s = 0.0
        enabled = False
        for layer in self.layers:
            consume_fn = getattr(layer, "consume_forward_step_profile", None)
            if not callable(consume_fn):
                continue
            layer_stats = consume_fn()
            if not isinstance(layer_stats, dict):
                continue
            enabled = True
            calls += int(layer_stats.get("calls", 0) or 0)
            proj_wall_s += float(layer_stats.get("proj_wall_s", 0.0) or 0.0)
            cache_wall_s += float(layer_stats.get("cache_wall_s", 0.0) or 0.0)
            attnff_wall_s += float(layer_stats.get("attnff_wall_s", 0.0) or 0.0)
            attn_core_wall_s += float(layer_stats.get("attn_core_wall_s", 0.0) or 0.0)
            finalize_wall_s += float(layer_stats.get("finalize_wall_s", 0.0) or 0.0)
            finalize_attn_outproj_wall_s += float(
                layer_stats.get("finalize_attn_outproj_wall_s", 0.0) or 0.0
            )
            finalize_ffn_wall_s += float(layer_stats.get("finalize_ffn_wall_s", 0.0) or 0.0)
            paged_path_single_page += int(layer_stats.get("paged_path_single_page", 0) or 0)
            paged_path_flash_prefix += int(layer_stats.get("paged_path_flash_prefix", 0) or 0)
            paged_path_flash_merge += int(layer_stats.get("paged_path_flash_merge", 0) or 0)
            paged_path_dense += int(layer_stats.get("paged_path_dense", 0) or 0)
            paged_recompute_calls += int(layer_stats.get("paged_recompute_calls", 0) or 0)
            total_wall_s += float(layer_stats.get("total_wall_s", 0.0) or 0.0)
        if not enabled:
            return None
        return {
            "calls": int(calls),
            "proj_wall_s": float(proj_wall_s),
            "cache_wall_s": float(cache_wall_s),
            "attnff_wall_s": float(attnff_wall_s),
            "attn_core_wall_s": float(attn_core_wall_s),
            "finalize_wall_s": float(finalize_wall_s),
            "finalize_attn_outproj_wall_s": float(finalize_attn_outproj_wall_s),
            "finalize_ffn_wall_s": float(finalize_ffn_wall_s),
            "paged_path_single_page": int(paged_path_single_page),
            "paged_path_flash_prefix": int(paged_path_flash_prefix),
            "paged_path_flash_merge": int(paged_path_flash_merge),
            "paged_path_dense": int(paged_path_dense),
            "paged_recompute_calls": int(paged_recompute_calls),
            "total_wall_s": float(total_wall_s),
        }

    def forward(
        self, 
        src: Tensor, 
        mask: Optional[Tensor] = None, 
    ) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src

        for mod in self.layers:
            output = mod(
                output, 
                src_mask=mask,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output

    def forward_step(
        self,
        src_step: Tensor,
        kv_cache=None,
        append_to_cache: bool = True,
        max_cache_len: Optional[int] = None,
        kv_cache_mode: str = "auto",
        kv_cache_page_size: Optional[int] = None,
        allow_grad_mutable_cache: bool = False,
        allow_grad_inplace_paged_cache: bool = False,
    ):
        output = src_step
        if kv_cache is None:
            kv_cache = [None] * len(self.layers)
        if len(kv_cache) != len(self.layers):
            raise ValueError(f"kv_cache length {len(kv_cache)} != num_layers {len(self.layers)}")

        reuse_cache_list = bool(
            isinstance(kv_cache, list)
            and bool(append_to_cache)
            and bool(allow_grad_mutable_cache)
            and bool(self.layers[0].cache_container_reuse if len(self.layers) > 0 else True)
        )
        if reuse_cache_list:
            new_cache = kv_cache
            layer_iter = zip(range(len(self.layers)), self.layers, kv_cache)
            for layer_idx, layer, cache in layer_iter:
                output, cache_next = layer.forward_step(
                    output,
                    kv_cache=cache,
                    append_to_cache=append_to_cache,
                    max_cache_len=max_cache_len,
                    kv_cache_mode=kv_cache_mode,
                    kv_cache_page_size=kv_cache_page_size,
                    allow_grad_mutable_cache=allow_grad_mutable_cache,
                    allow_grad_inplace_paged_cache=allow_grad_inplace_paged_cache,
                )
                new_cache[layer_idx] = cache_next
        else:
            new_cache = []
            for layer, cache in zip(self.layers, kv_cache):
                output, cache_next = layer.forward_step(
                    output,
                    kv_cache=cache,
                    append_to_cache=append_to_cache,
                    max_cache_len=max_cache_len,
                    kv_cache_mode=kv_cache_mode,
                    kv_cache_page_size=kv_cache_page_size,
                    allow_grad_mutable_cache=allow_grad_mutable_cache,
                    allow_grad_inplace_paged_cache=allow_grad_inplace_paged_cache,
                )
                new_cache.append(cache_next)

        if self.norm is not None:
            output = self.norm(output)
        return output, new_cache

    def encode_prefix_to_kv(self, src_prefix: Tensor):
        output = src_prefix
        kv_cache = []
        for layer in self.layers:
            output, layer_cache = layer.forward_causal_prefix(output)
            kv_cache.append(layer_cache)
        if self.norm is not None:
            output = self.norm(output)
        return kv_cache, output

    def forward_query(self, src_query: Tensor, kv_cache):
        if len(kv_cache) != len(self.layers):
            raise ValueError(f"kv_cache length {len(kv_cache)} != num_layers {len(self.layers)}")
        output = src_query
        for layer, layer_cache in zip(self.layers, kv_cache):
            output = layer.forward_query(output, layer_cache)
        if self.norm is not None:
            output = self.norm(output)
        return output

    def forward_with_prefix_cache(self, src_prefix: Tensor, src_query: Tensor):
        kv_cache, _ = self.encode_prefix_to_kv(src_prefix)
        output_q = self.forward_query(src_query, kv_cache) if src_query.shape[0] > 0 else src_query
        return output_q, kv_cache
