import math
import inspect
import os
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
try:
    import triton
    import triton.language as tl
    import triton.language.extra.cuda.libdevice as tl_libdevice
except Exception:  # pragma: no cover - optional CUDA path
    triton = None
    tl = None
    tl_libdevice = None

from ticl.distributions import parse_distributions, sample_distributions
from ticl.utils import default_device


if triton is not None:

    @triton.jit
    def _ragged_affine_fwd_kernel(
        x_ptr,
        w_ptr,
        b_ptr,
        input_index_ptr,
        in_sizes_ptr,
        out_sizes_ptr,
        out_ptr,
        k_cap,
        stride_xb,
        stride_xi,
        stride_wb,
        stride_wi,
        stride_wo,
        stride_bb,
        stride_bo,
        stride_ib,
        stride_ii,
        stride_ob,
        stride_oo,
        BLOCK_O: tl.constexpr,
        BLOCK_K: tl.constexpr,
        HAS_BIAS: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_o = tl.program_id(1)
        offs_o = pid_o * BLOCK_O + tl.arange(0, BLOCK_O)
        out_size = tl.load(out_sizes_ptr + pid_b)
        out_mask = offs_o < out_size
        in_size = tl.load(in_sizes_ptr + pid_b)
        acc = tl.zeros((BLOCK_O,), dtype=tl.float32)
        if HAS_BIAS:
            bias_ptrs = b_ptr + pid_b * stride_bb + offs_o * stride_bo
            acc += tl.load(bias_ptrs, mask=out_mask, other=0.0).to(tl.float32)
        x_base = x_ptr + pid_b * stride_xb
        w_base = w_ptr + pid_b * stride_wb
        input_index_base = input_index_ptr + pid_b * stride_ib
        for k_start in tl.range(0, in_size, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            k_mask = offs_k < in_size
            input_idx = tl.load(input_index_base + offs_k * stride_ii, mask=k_mask, other=0).to(tl.int32)
            x_vals = tl.load(x_base + input_idx * stride_xi, mask=k_mask, other=0.0).to(tl.float32)
            w_ptrs = w_base + input_idx[:, None] * stride_wi + offs_o[None, :] * stride_wo
            w_vals = tl.load(w_ptrs, mask=k_mask[:, None] & out_mask[None, :], other=0.0).to(tl.float32)
            acc += tl.sum(w_vals * x_vals[:, None], axis=0)
        out_ptrs = out_ptr + pid_b * stride_ob + offs_o * stride_oo
        tl.store(out_ptrs, acc, mask=out_mask)


    @triton.jit
    def _ragged_affine_bwd_input_kernel(
        grad_out_ptr,
        w_ptr,
        input_index_ptr,
        in_sizes_ptr,
        out_sizes_ptr,
        grad_x_ptr,
        o_cap,
        stride_gob,
        stride_goi,
        stride_wb,
        stride_wi,
        stride_wo,
        stride_ib,
        stride_ii,
        stride_gxb,
        stride_gxi,
        BLOCK_O: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_k = tl.program_id(1)
        offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        in_size = tl.load(in_sizes_ptr + pid_b)
        k_mask = offs_k < in_size
        input_index_base = input_index_ptr + pid_b * stride_ib
        input_idx = tl.load(input_index_base + offs_k * stride_ii, mask=k_mask, other=0).to(tl.int32)
        out_size = tl.load(out_sizes_ptr + pid_b)
        acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
        grad_out_base = grad_out_ptr + pid_b * stride_gob
        w_base = w_ptr + pid_b * stride_wb
        for o_start in tl.range(0, out_size, BLOCK_O):
            offs_o = o_start + tl.arange(0, BLOCK_O)
            out_mask = offs_o < out_size
            grad_vals = tl.load(grad_out_base + offs_o * stride_goi, mask=out_mask, other=0.0).to(tl.float32)
            w_ptrs = w_base + input_idx[:, None] * stride_wi + offs_o[None, :] * stride_wo
            w_vals = tl.load(w_ptrs, mask=k_mask[:, None] & out_mask[None, :], other=0.0).to(tl.float32)
            acc += tl.sum(w_vals * grad_vals[None, :], axis=1)
        grad_x_ptrs = grad_x_ptr + pid_b * stride_gxb + input_idx * stride_gxi
        tl.store(grad_x_ptrs, acc, mask=k_mask)


    class _RaggedBatchAffineFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, w, b, input_index, in_sizes, out_sizes):
            if triton is None:
                raise RuntimeError("ragged affine Triton path requested but Triton is unavailable")
            x_contig = x.contiguous()
            w_contig = w.contiguous()
            b_contig = None if b is None else b.contiguous()
            input_index_i32 = input_index.to(dtype=torch.int32).contiguous()
            in_sizes_i32 = in_sizes.to(dtype=torch.int32).contiguous()
            out_sizes_i32 = out_sizes.to(dtype=torch.int32).contiguous()
            batch_size = int(x_contig.shape[0])
            o_cap = int(w_contig.shape[2])
            k_cap = int(input_index_i32.shape[1])
            out = torch.zeros((batch_size, o_cap), device=x_contig.device, dtype=x_contig.dtype)
            if batch_size > 0 and o_cap > 0 and k_cap > 0:
                grid = (batch_size, triton.cdiv(o_cap, 32))
                _ragged_affine_fwd_kernel[grid](
                    x_contig,
                    w_contig,
                    b_contig,
                    input_index_i32,
                    in_sizes_i32,
                    out_sizes_i32,
                    out,
                    k_cap,
                    x_contig.stride(0),
                    x_contig.stride(1),
                    w_contig.stride(0),
                    w_contig.stride(1),
                    w_contig.stride(2),
                    0 if b_contig is None else b_contig.stride(0),
                    0 if b_contig is None else b_contig.stride(1),
                    input_index_i32.stride(0),
                    input_index_i32.stride(1),
                    out.stride(0),
                    out.stride(1),
                    BLOCK_O=32,
                    BLOCK_K=32,
                    HAS_BIAS=bool(b_contig is not None),
                    num_warps=4,
                )
            ctx.save_for_backward(w_contig, input_index_i32, in_sizes_i32, out_sizes_i32)
            ctx.x_shape = tuple(x_contig.shape)
            return out

        @staticmethod
        def backward(ctx, grad_out):
            if triton is None:
                raise RuntimeError("ragged affine Triton backward requested but Triton is unavailable")
            w_contig, input_index_i32, in_sizes_i32, out_sizes_i32 = ctx.saved_tensors
            grad_out_contig = grad_out.contiguous()
            grad_x = torch.zeros(ctx.x_shape, device=grad_out_contig.device, dtype=grad_out_contig.dtype)
            batch_size = int(grad_out_contig.shape[0])
            o_cap = int(grad_out_contig.shape[1])
            k_cap = int(input_index_i32.shape[1])
            if batch_size > 0 and o_cap > 0 and k_cap > 0:
                grid = (batch_size, triton.cdiv(k_cap, 32))
                _ragged_affine_bwd_input_kernel[grid](
                    grad_out_contig,
                    w_contig,
                    input_index_i32,
                    in_sizes_i32,
                    out_sizes_i32,
                    grad_x,
                    o_cap,
                    grad_out_contig.stride(0),
                    grad_out_contig.stride(1),
                    w_contig.stride(0),
                    w_contig.stride(1),
                    w_contig.stride(2),
                    input_index_i32.stride(0),
                    input_index_i32.stride(1),
                    grad_x.stride(0),
                    grad_x.stride(1),
                    BLOCK_O=32,
                    BLOCK_K=32,
                    num_warps=4,
                )
            return grad_x, None, None, None, None, None


if triton is not None:

    @triton.jit
    def _ragged_tiled_affine_fwd_kernel(
        x_ptr,
        w_ptr,
        b_ptr,
        input_index_ptr,
        in_sizes_ptr,
        out_sizes_ptr,
        activation_code_ptr,
        tile_batch_ptr,
        tile_out_offset_ptr,
        pre_out_ptr,
        out_ptr,
        k_cap,
        stride_xb,
        stride_xi,
        stride_wb,
        stride_wi,
        stride_wo,
        stride_bb,
        stride_bo,
        stride_ib,
        stride_ii,
        stride_sb,
        stride_ab,
        stride_tb,
        stride_to,
        stride_pob,
        stride_poo,
        stride_ob,
        stride_oo,
        BLOCK_O: tl.constexpr,
        BLOCK_K: tl.constexpr,
        HAS_BIAS: tl.constexpr,
    ):
        pid_t = tl.program_id(0)
        pid_b = tl.load(tile_batch_ptr + pid_t * stride_tb).to(tl.int32)
        o_start = tl.load(tile_out_offset_ptr + pid_t * stride_to).to(tl.int32)
        offs_o = o_start + tl.arange(0, BLOCK_O)
        out_size = tl.load(out_sizes_ptr + pid_b * stride_sb)
        out_mask = offs_o < out_size
        in_size = tl.load(in_sizes_ptr + pid_b * stride_sb)
        acc = tl.zeros((BLOCK_O,), dtype=tl.float32)
        if HAS_BIAS:
            bias_ptrs = b_ptr + pid_b * stride_bb + offs_o * stride_bo
            acc += tl.load(bias_ptrs, mask=out_mask, other=0.0).to(tl.float32)
        x_base = x_ptr + pid_b * stride_xb
        w_base = w_ptr + pid_b * stride_wb
        input_index_base = input_index_ptr + pid_b * stride_ib
        for k_start in tl.range(0, in_size, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            k_mask = offs_k < in_size
            input_idx = tl.load(input_index_base + offs_k * stride_ii, mask=k_mask, other=0).to(tl.int32)
            x_vals = tl.load(x_base + input_idx * stride_xi, mask=k_mask, other=0.0).to(tl.float32)
            w_ptrs = w_base + input_idx[:, None] * stride_wi + offs_o[None, :] * stride_wo
            w_vals = tl.load(w_ptrs, mask=k_mask[:, None] & out_mask[None, :], other=0.0).to(tl.float32)
            acc += tl.sum(w_vals * x_vals[:, None], axis=0)
        pre_out_ptrs = pre_out_ptr + pid_b * stride_pob + offs_o * stride_poo
        tl.store(pre_out_ptrs, acc, mask=out_mask)
        activation_code = tl.load(activation_code_ptr + pid_b * stride_ab).to(tl.int32)
        if activation_code == 0:
            acc = tl_libdevice.tanh(acc)
        elif activation_code == 1:
            acc = tl.maximum(acc, 0.0)
        elif activation_code == 3:
            acc = tl_libdevice.cos(acc)
        out_ptrs = out_ptr + pid_b * stride_ob + offs_o * stride_oo
        tl.store(out_ptrs, acc, mask=out_mask)


    @triton.jit
    def _ragged_tiled_affine_bwd_input_kernel(
        grad_out_ptr,
        w_ptr,
        input_index_ptr,
        in_sizes_ptr,
        out_sizes_ptr,
        tile_batch_ptr,
        tile_in_offset_ptr,
        grad_x_ptr,
        o_cap,
        stride_gob,
        stride_goi,
        stride_wb,
        stride_wi,
        stride_wo,
        stride_ib,
        stride_ii,
        stride_sb,
        stride_tb,
        stride_ti,
        stride_gxb,
        stride_gxi,
        BLOCK_O: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_t = tl.program_id(0)
        pid_b = tl.load(tile_batch_ptr + pid_t * stride_tb).to(tl.int32)
        k_start = tl.load(tile_in_offset_ptr + pid_t * stride_ti).to(tl.int32)
        offs_k = k_start + tl.arange(0, BLOCK_K)
        in_size = tl.load(in_sizes_ptr + pid_b * stride_sb)
        out_size = tl.load(out_sizes_ptr + pid_b * stride_sb)
        k_mask = offs_k < in_size
        input_index_base = input_index_ptr + pid_b * stride_ib
        input_idx = tl.load(input_index_base + offs_k * stride_ii, mask=k_mask, other=0).to(tl.int32)
        acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
        grad_out_base = grad_out_ptr + pid_b * stride_gob
        w_base = w_ptr + pid_b * stride_wb
        for o_start in tl.range(0, out_size, BLOCK_O):
            offs_o = o_start + tl.arange(0, BLOCK_O)
            out_mask = offs_o < out_size
            grad_vals = tl.load(grad_out_base + offs_o * stride_goi, mask=out_mask, other=0.0).to(tl.float32)
            w_ptrs = w_base + input_idx[:, None] * stride_wi + offs_o[None, :] * stride_wo
            w_vals = tl.load(w_ptrs, mask=k_mask[:, None] & out_mask[None, :], other=0.0).to(tl.float32)
            acc += tl.sum(w_vals * grad_vals[None, :], axis=1)
        grad_x_ptrs = grad_x_ptr + pid_b * stride_gxb + input_idx * stride_gxi
        tl.store(grad_x_ptrs, acc, mask=k_mask)

class _TiledRaggedBatchAffineFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        w,
        b,
        input_index,
        in_sizes,
        out_sizes,
        activation_codes,
        out_tile_batch,
        out_tile_offsets,
        in_tile_batch,
        in_tile_offsets,
        block_o,
        block_k,
        num_warps,
    ):
        if triton is None:
            raise RuntimeError("tiled ragged affine Triton path requested but Triton is unavailable")
        block_o = int(block_o)
        block_k = int(block_k)
        num_warps = int(num_warps)
        x_contig = x.contiguous()
        w_contig = w.contiguous()
        b_contig = None if b is None else b.contiguous()
        input_index_i32 = input_index.to(dtype=torch.int32).contiguous()
        in_sizes_i32 = in_sizes.to(dtype=torch.int32).contiguous()
        out_sizes_i32 = out_sizes.to(dtype=torch.int32).contiguous()
        activation_codes_i32 = activation_codes.to(dtype=torch.int32).contiguous()
        out_tile_batch_i32 = out_tile_batch.to(dtype=torch.int32).contiguous()
        out_tile_offsets_i32 = out_tile_offsets.to(dtype=torch.int32).contiguous()
        in_tile_batch_i32 = in_tile_batch.to(dtype=torch.int32).contiguous()
        in_tile_offsets_i32 = in_tile_offsets.to(dtype=torch.int32).contiguous()
        batch_size = int(x_contig.shape[0])
        o_cap = int(w_contig.shape[2])
        k_cap = int(input_index_i32.shape[1])
        pre_out = torch.zeros((batch_size, o_cap), device=x_contig.device, dtype=x_contig.dtype)
        out = torch.zeros((batch_size, o_cap), device=x_contig.device, dtype=x_contig.dtype)
        if batch_size > 0 and o_cap > 0 and k_cap > 0 and int(out_tile_batch_i32.numel()) > 0:
            grid = (int(out_tile_batch_i32.numel()),)
            _ragged_tiled_affine_fwd_kernel[grid](
                x_contig,
                w_contig,
                b_contig,
                input_index_i32,
                in_sizes_i32,
                out_sizes_i32,
                activation_codes_i32,
                out_tile_batch_i32,
                out_tile_offsets_i32,
                pre_out,
                out,
                k_cap,
                x_contig.stride(0),
                x_contig.stride(1),
                w_contig.stride(0),
                w_contig.stride(1),
                w_contig.stride(2),
                0 if b_contig is None else b_contig.stride(0),
                0 if b_contig is None else b_contig.stride(1),
                input_index_i32.stride(0),
                input_index_i32.stride(1),
                in_sizes_i32.stride(0),
                activation_codes_i32.stride(0),
                out_tile_batch_i32.stride(0),
                out_tile_offsets_i32.stride(0),
                pre_out.stride(0),
                pre_out.stride(1),
                out.stride(0),
                out.stride(1),
                BLOCK_O=block_o,
                BLOCK_K=block_k,
                HAS_BIAS=bool(b_contig is not None),
                num_warps=num_warps,
            )
        ctx.save_for_backward(
            w_contig,
            input_index_i32,
            in_sizes_i32,
            out_sizes_i32,
            activation_codes_i32,
            in_tile_batch_i32,
            in_tile_offsets_i32,
            pre_out,
            out,
        )
        ctx.x_shape = tuple(x_contig.shape)
        ctx.block_o = block_o
        ctx.block_k = block_k
        ctx.num_warps = num_warps
        return out

    @staticmethod
    def backward(ctx, grad_out):
        if triton is None:
            raise RuntimeError("tiled ragged affine Triton backward requested but Triton is unavailable")
        (
            w_contig,
            input_index_i32,
            in_sizes_i32,
            out_sizes_i32,
            activation_codes_i32,
            in_tile_batch_i32,
            in_tile_offsets_i32,
            pre_out,
            out_saved,
        ) = ctx.saved_tensors
        grad_out_contig = grad_out.contiguous()
        if int(activation_codes_i32.numel()) > 0:
            activation_codes = activation_codes_i32.to(device=grad_out_contig.device)
            no_activation_mask = activation_codes < 0
            relu_mask = activation_codes == 1
            tanh_mask = activation_codes == 0
            cos_mask = activation_codes == 3
            if bool(torch.any(relu_mask)):
                grad_out_contig = torch.where(
                    relu_mask.unsqueeze(1),
                    grad_out_contig * (out_saved > 0).to(dtype=grad_out_contig.dtype),
                    grad_out_contig,
                )
            if bool(torch.any(tanh_mask)):
                grad_out_contig = torch.where(
                    tanh_mask.unsqueeze(1),
                    grad_out_contig * (1.0 - out_saved.square()),
                    grad_out_contig,
                )
            if bool(torch.any(cos_mask)):
                grad_out_contig = torch.where(
                    cos_mask.unsqueeze(1),
                    grad_out_contig * (-torch.sin(pre_out)),
                    grad_out_contig,
                )
            if bool(torch.any(no_activation_mask)):
                grad_out_contig = torch.where(no_activation_mask.unsqueeze(1), grad_out.contiguous(), grad_out_contig)
        grad_x = torch.zeros(ctx.x_shape, device=grad_out_contig.device, dtype=grad_out_contig.dtype)
        o_cap = int(w_contig.shape[2])
        if int(grad_out_contig.shape[0]) > 0 and o_cap > 0 and int(in_tile_batch_i32.numel()) > 0:
            grid = (int(in_tile_batch_i32.numel()),)
            _ragged_tiled_affine_bwd_input_kernel[grid](
                grad_out_contig,
                w_contig,
                input_index_i32,
                in_sizes_i32,
                out_sizes_i32,
                in_tile_batch_i32,
                in_tile_offsets_i32,
                grad_x,
                o_cap,
                grad_out_contig.stride(0),
                grad_out_contig.stride(1),
                w_contig.stride(0),
                w_contig.stride(1),
                w_contig.stride(2),
                input_index_i32.stride(0),
                input_index_i32.stride(1),
                in_sizes_i32.stride(0),
                in_tile_batch_i32.stride(0),
                in_tile_offsets_i32.stride(0),
                grad_x.stride(0),
                grad_x.stride(1),
                BLOCK_O=int(ctx.block_o),
                BLOCK_K=int(ctx.block_k),
                num_warps=int(ctx.num_warps),
            )
        return grad_x, None, None, None, None, None, None, None, None, None, None, None, None, None

if triton is not None:

    @triton.jit
    def _prefix_tiled_affine_fwd_kernel(
        x_ptr,
        w_ptr,
        b_ptr,
        in_sizes_ptr,
        out_sizes_ptr,
        activation_code_ptr,
        tile_batch_ptr,
        tile_out_offset_ptr,
        pre_out_ptr,
        out_ptr,
        k_cap,
        stride_xb,
        stride_xi,
        stride_wb,
        stride_wi,
        stride_wo,
        stride_bb,
        stride_bo,
        stride_sb,
        stride_ab,
        stride_tb,
        stride_to,
        stride_pob,
        stride_poo,
        stride_ob,
        stride_oo,
        BLOCK_O: tl.constexpr,
        BLOCK_K: tl.constexpr,
        HAS_BIAS: tl.constexpr,
    ):
        pid_t = tl.program_id(0)
        pid_b = tl.load(tile_batch_ptr + pid_t * stride_tb).to(tl.int32)
        o_start = tl.load(tile_out_offset_ptr + pid_t * stride_to).to(tl.int32)
        offs_o = o_start + tl.arange(0, BLOCK_O)
        out_size = tl.load(out_sizes_ptr + pid_b * stride_sb)
        out_mask = offs_o < out_size
        in_size = tl.load(in_sizes_ptr + pid_b * stride_sb)
        acc = tl.zeros((BLOCK_O,), dtype=tl.float32)
        if HAS_BIAS:
            bias_ptrs = b_ptr + pid_b * stride_bb + offs_o * stride_bo
            acc += tl.load(bias_ptrs, mask=out_mask, other=0.0).to(tl.float32)
        x_base = x_ptr + pid_b * stride_xb
        w_base = w_ptr + pid_b * stride_wb
        for k_start in tl.range(0, in_size, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            k_mask = offs_k < in_size
            x_vals = tl.load(x_base + offs_k * stride_xi, mask=k_mask, other=0.0).to(tl.float32)
            w_ptrs = w_base + offs_k[:, None] * stride_wi + offs_o[None, :] * stride_wo
            w_vals = tl.load(w_ptrs, mask=k_mask[:, None] & out_mask[None, :], other=0.0).to(tl.float32)
            acc += tl.sum(w_vals * x_vals[:, None], axis=0)
        pre_out_ptrs = pre_out_ptr + pid_b * stride_pob + offs_o * stride_poo
        tl.store(pre_out_ptrs, acc, mask=out_mask)
        activation_code = tl.load(activation_code_ptr + pid_b * stride_ab).to(tl.int32)
        if activation_code == 0:
            acc = tl_libdevice.tanh(acc)
        elif activation_code == 1:
            acc = tl.maximum(acc, 0.0)
        elif activation_code == 3:
            acc = tl_libdevice.cos(acc)
        out_ptrs = out_ptr + pid_b * stride_ob + offs_o * stride_oo
        tl.store(out_ptrs, acc, mask=out_mask)


    @triton.jit
    def _prefix_tiled_affine_bwd_input_kernel(
        grad_out_ptr,
        w_ptr,
        in_sizes_ptr,
        out_sizes_ptr,
        tile_batch_ptr,
        tile_in_offset_ptr,
        grad_x_ptr,
        o_cap,
        stride_gob,
        stride_goi,
        stride_wb,
        stride_wi,
        stride_wo,
        stride_sb,
        stride_tb,
        stride_ti,
        stride_gxb,
        stride_gxi,
        BLOCK_O: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_t = tl.program_id(0)
        pid_b = tl.load(tile_batch_ptr + pid_t * stride_tb).to(tl.int32)
        k_start = tl.load(tile_in_offset_ptr + pid_t * stride_ti).to(tl.int32)
        offs_k = k_start + tl.arange(0, BLOCK_K)
        in_size = tl.load(in_sizes_ptr + pid_b * stride_sb)
        out_size = tl.load(out_sizes_ptr + pid_b * stride_sb)
        k_mask = offs_k < in_size
        acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
        grad_out_base = grad_out_ptr + pid_b * stride_gob
        w_base = w_ptr + pid_b * stride_wb
        for o_start in tl.range(0, out_size, BLOCK_O):
            offs_o = o_start + tl.arange(0, BLOCK_O)
            out_mask = offs_o < out_size
            grad_vals = tl.load(grad_out_base + offs_o * stride_goi, mask=out_mask, other=0.0).to(tl.float32)
            w_ptrs = w_base + offs_k[:, None] * stride_wi + offs_o[None, :] * stride_wo
            w_vals = tl.load(w_ptrs, mask=k_mask[:, None] & out_mask[None, :], other=0.0).to(tl.float32)
            acc += tl.sum(w_vals * grad_vals[None, :], axis=1)
        grad_x_ptrs = grad_x_ptr + pid_b * stride_gxb + offs_k * stride_gxi
        tl.store(grad_x_ptrs, acc, mask=k_mask)

class _PrefixTiledBatchAffineFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        w,
        b,
        in_sizes,
        out_sizes,
        activation_codes,
        out_tile_batch,
        out_tile_offsets,
        in_tile_batch,
        in_tile_offsets,
        block_o,
        block_k,
        num_warps,
    ):
        if triton is None:
            raise RuntimeError("prefix tiled affine Triton path requested but Triton is unavailable")
        block_o = int(block_o)
        block_k = int(block_k)
        num_warps = int(num_warps)
        x_contig = x.contiguous()
        w_contig = w.contiguous()
        b_contig = None if b is None else b.contiguous()
        in_sizes_i32 = in_sizes.to(dtype=torch.int32).contiguous()
        out_sizes_i32 = out_sizes.to(dtype=torch.int32).contiguous()
        activation_codes_i32 = activation_codes.to(dtype=torch.int32).contiguous()
        out_tile_batch_i32 = out_tile_batch.to(dtype=torch.int32).contiguous()
        out_tile_offsets_i32 = out_tile_offsets.to(dtype=torch.int32).contiguous()
        in_tile_batch_i32 = in_tile_batch.to(dtype=torch.int32).contiguous()
        in_tile_offsets_i32 = in_tile_offsets.to(dtype=torch.int32).contiguous()
        batch_size = int(x_contig.shape[0])
        k_cap = int(w_contig.shape[1])
        o_cap = int(w_contig.shape[2])
        pre_out = torch.zeros((batch_size, o_cap), device=x_contig.device, dtype=x_contig.dtype)
        out = torch.zeros((batch_size, o_cap), device=x_contig.device, dtype=x_contig.dtype)
        if batch_size > 0 and k_cap > 0 and o_cap > 0 and int(out_tile_batch_i32.numel()) > 0:
            grid = (int(out_tile_batch_i32.numel()),)
            _prefix_tiled_affine_fwd_kernel[grid](
                x_contig,
                w_contig,
                b_contig,
                in_sizes_i32,
                out_sizes_i32,
                activation_codes_i32,
                out_tile_batch_i32,
                out_tile_offsets_i32,
                pre_out,
                out,
                k_cap,
                x_contig.stride(0),
                x_contig.stride(1),
                w_contig.stride(0),
                w_contig.stride(1),
                w_contig.stride(2),
                0 if b_contig is None else b_contig.stride(0),
                0 if b_contig is None else b_contig.stride(1),
                in_sizes_i32.stride(0),
                activation_codes_i32.stride(0),
                out_tile_batch_i32.stride(0),
                out_tile_offsets_i32.stride(0),
                pre_out.stride(0),
                pre_out.stride(1),
                out.stride(0),
                out.stride(1),
                BLOCK_O=block_o,
                BLOCK_K=block_k,
                HAS_BIAS=bool(b_contig is not None),
                num_warps=num_warps,
            )
        ctx.save_for_backward(
            w_contig,
            in_sizes_i32,
            out_sizes_i32,
            activation_codes_i32,
            in_tile_batch_i32,
            in_tile_offsets_i32,
            pre_out,
            out,
        )
        ctx.x_shape = tuple(x_contig.shape)
        ctx.block_o = block_o
        ctx.block_k = block_k
        ctx.num_warps = num_warps
        return out

    @staticmethod
    def backward(ctx, grad_out):
        if triton is None:
            raise RuntimeError("prefix tiled affine Triton backward requested but Triton is unavailable")
        (
            w_contig,
            in_sizes_i32,
            out_sizes_i32,
            activation_codes_i32,
            in_tile_batch_i32,
            in_tile_offsets_i32,
            pre_out,
            out_saved,
        ) = ctx.saved_tensors
        grad_out_contig = grad_out.contiguous()
        if int(activation_codes_i32.numel()) > 0:
            activation_codes = activation_codes_i32.to(device=grad_out_contig.device)
            no_activation_mask = activation_codes < 0
            relu_mask = activation_codes == 1
            tanh_mask = activation_codes == 0
            cos_mask = activation_codes == 3
            if bool(torch.any(relu_mask)):
                grad_out_contig = torch.where(
                    relu_mask.unsqueeze(1),
                    grad_out_contig * (out_saved > 0).to(dtype=grad_out_contig.dtype),
                    grad_out_contig,
                )
            if bool(torch.any(tanh_mask)):
                grad_out_contig = torch.where(
                    tanh_mask.unsqueeze(1),
                    grad_out_contig * (1.0 - out_saved.square()),
                    grad_out_contig,
                )
            if bool(torch.any(cos_mask)):
                grad_out_contig = torch.where(
                    cos_mask.unsqueeze(1),
                    grad_out_contig * (-torch.sin(pre_out)),
                    grad_out_contig,
                )
            if bool(torch.any(no_activation_mask)):
                grad_out_contig = torch.where(no_activation_mask.unsqueeze(1), grad_out.contiguous(), grad_out_contig)
        grad_x = torch.zeros(ctx.x_shape, device=grad_out_contig.device, dtype=grad_out_contig.dtype)
        o_cap = int(w_contig.shape[2])
        if int(grad_out_contig.shape[0]) > 0 and o_cap > 0 and int(in_tile_batch_i32.numel()) > 0:
            grid = (int(in_tile_batch_i32.numel()),)
            _prefix_tiled_affine_bwd_input_kernel[grid](
                grad_out_contig,
                w_contig,
                in_sizes_i32,
                out_sizes_i32,
                in_tile_batch_i32,
                in_tile_offsets_i32,
                grad_x,
                o_cap,
                grad_out_contig.stride(0),
                grad_out_contig.stride(1),
                w_contig.stride(0),
                w_contig.stride(1),
                w_contig.stride(2),
                in_sizes_i32.stride(0),
                in_tile_batch_i32.stride(0),
                in_tile_offsets_i32.stride(0),
                grad_x.stride(0),
                grad_x.stride(1),
                BLOCK_O=int(ctx.block_o),
                BLOCK_K=int(ctx.block_k),
                num_warps=int(ctx.num_warps),
            )
        return grad_x, None, None, None, None, None, None, None, None, None, None, None, None

class EnvironmentPrior:
    """
    Environment prior with explicit SCM/GP input partition:
      [state | obs(subset of state) | action | noise | zero_pad]

    Semantics:
    - `s_{t+1}` is sampled by the X-style generator (same role as X in SCM/GP priors).
    - `r_{t+1}` is sampled by the Y-style generator (same role as Y in SCM/GP priors).
    - PFN token at step `t` is `(obs_t, a_t, r_t)` with fixed-width projection by pad/truncate.
    """

    def __init__(self, config=None):
        cfg = dict(config or {})

        # Family sampling (SCM/GP) follows existing parse/sample_distributions flow.
        cfg.setdefault("family", {"distribution": "meta_choice", "choice_values": ["scm", "gp"]})

        # Required dimension randomization ranges.
        cfg.setdefault("action_dim", {"distribution": "uniform_int", "min": 1, "max": 30})
        cfg.setdefault("state_dim", {"distribution": "uniform_int", "min": 1, "max": 400})
        cfg.setdefault("obs_dim", {"distribution": "uniform_int", "min": 1, "max": 400})
        cfg.setdefault("noise_dim", {"distribution": "uniform_int", "min": 1, "max": 64})
        cfg.setdefault("zero_pad_dim", {"distribution": "uniform_int", "min": 0, "max": 400})
        cfg.setdefault("obs_slot_dim", 400)
        cfg.setdefault("action_slot_dim", 30)
        cfg.setdefault("reward_dropout_enabled", True)
        cfg.setdefault("reward_dropout_randomize", True)
        cfg.setdefault("reward_dropout_ratio", 0.0)
        cfg.setdefault("reward_dropout_ratio_min", 0.1)
        cfg.setdefault("reward_dropout_ratio_max", 1.0)
        cfg.setdefault("reward_dropout_impute_zero", True)
        # Batch-level rollout parallelism for get_batch():
        # each batch column is independent and can be generated concurrently.
        cfg.setdefault("batch_parallel_workers", 4)
        cfg.setdefault("batch_parallel_backend", "python_thread")
        cfg.setdefault("batch_shared_environment", False)
        cfg.setdefault("batch_vectorized_strict_rng_match", False)
        # torch_vectorized grouping mode:
        # - "structure": strict homogeneous grouping (legacy semantics path)
        # - "family": coarser grouping by family to enlarge policy-step batch width
        cfg.setdefault("batch_vectorized_grouping", "structure")

        # Dynamics / rollout knobs.
        cfg.setdefault("alpha", {"distribution": "uniform", "min": 0.05, "max": 0.35})
        cfg.setdefault("init_state_std", {"distribution": "log_uniform", "min": 1e-3, "max": 1.0})
        cfg.setdefault("init_action_std", {"distribution": "log_uniform", "min": 1e-3, "max": 1.0})
        cfg.setdefault("state_noise_std", {"distribution": "log_uniform", "min": 1e-4, "max": 0.2})
        cfg.setdefault("action_noise_train_std", {"distribution": "log_uniform", "min": 1e-4, "max": 0.2})
        cfg.setdefault("action_noise_eval_std", {"distribution": "log_uniform", "min": 1e-4, "max": 0.1})
        # Reward-scale is sampled per environment; final rewards are clipped
        # to keep rollout targets in a stable bounded range.
        cfg.setdefault("reward_scale", {"distribution": "uniform", "min": 0.1, "max": 10.0})
        cfg.setdefault("reward_clip", 10.0)
        cfg.setdefault("state_clip", 8.0)
        # Optional residual highway on state update:
        # s_{t+1} <- lambda * s_t + (1-lambda) * tanh(clamp(s_{t+1})).
        cfg.setdefault("state_highway_enabled", False)
        cfg.setdefault("state_highway_lambda", 0.0)
        state_highway_enabled_env = os.environ.get("TICL_STATE_HIGHWAY_ENABLED")
        if state_highway_enabled_env is not None:
            cfg["state_highway_enabled"] = state_highway_enabled_env.strip().lower() not in {
                "0",
                "false",
                "no",
                "off",
                "",
            }
        state_highway_lambda_env = os.environ.get("TICL_STATE_HIGHWAY_LAMBDA")
        if state_highway_lambda_env is not None:
            try:
                cfg["state_highway_lambda"] = float(state_highway_lambda_env)
            except ValueError:
                pass
        # anti-explosion&vanishing-v2:
        # two-sided corridor regularization on log gain of consecutive
        # state increments.
        cfg.setdefault("anti_explosion_vanishing_v2_enabled", False)
        cfg.setdefault("anti_explosion_vanishing_v2_lambda", 0.05)
        cfg.setdefault("anti_explosion_vanishing_v2_gain_lo", 0.85)
        cfg.setdefault("anti_explosion_vanishing_v2_gain_hi", 1.15)
        cfg.setdefault("anti_explosion_vanishing_v2_huber_delta", 0.05)
        cfg.setdefault("anti_explosion_vanishing_v2_eps", 1e-6)
        cfg.setdefault("anti_explosion_vanishing_v2_detach_reference", True)
        # anti-explosion&vanishing-v3:
        # decoupled drift+tail regularization on log gain of consecutive
        # latent-state increments.
        cfg.setdefault("anti_explosion_vanishing_v3_enabled", False)
        cfg.setdefault("anti_explosion_vanishing_v3_lambda_drift", 0.02)
        cfg.setdefault("anti_explosion_vanishing_v3_lambda_tail", 0.05)
        cfg.setdefault("anti_explosion_vanishing_v3_gain_lo", 0.85)
        cfg.setdefault("anti_explosion_vanishing_v3_gain_hi", 1.15)
        cfg.setdefault("anti_explosion_vanishing_v3_tail_tau", 0.02)
        cfg.setdefault("anti_explosion_vanishing_v3_eps", 1e-6)
        cfg.setdefault("anti_explosion_vanishing_v3_detach_reference", True)
        # anti-explosion&vanishing-v4:
        # controlled highway-subspace residual update + gain regularization.
        cfg.setdefault("anti_explosion_vanishing_v4_enabled", False)
        cfg.setdefault("anti_explosion_vanishing_v4_lambda_drift", 0.08)
        cfg.setdefault("anti_explosion_vanishing_v4_lambda_tail", 0.25)
        cfg.setdefault("anti_explosion_vanishing_v4_gain_lo", 0.97)
        cfg.setdefault("anti_explosion_vanishing_v4_gain_hi", 1.03)
        cfg.setdefault("anti_explosion_vanishing_v4_tail_tau", 0.010)
        cfg.setdefault("anti_explosion_vanishing_v4_eps", 1e-6)
        cfg.setdefault("anti_explosion_vanishing_v4_detach_reference", True)
        cfg.setdefault("anti_explosion_vanishing_v4_highway_ratio", 0.25)
        cfg.setdefault("anti_explosion_vanishing_v4_update_scale", 0.08)
        cfg.setdefault("anti_explosion_vanishing_v4_update_clip", 0.0)
        # anti-explosion&vanishing-v5:
        # detached reward-signal thermostat. This rescales PG loss magnitude
        # using per-batch reward std while preserving ascent direction.
        cfg.setdefault("anti_explosion_vanishing_v5_enabled", False)
        cfg.setdefault("anti_explosion_vanishing_v5_target_std", 0.25)
        cfg.setdefault("anti_explosion_vanishing_v5_scale_lo", 0.5)
        cfg.setdefault("anti_explosion_vanishing_v5_scale_hi", 4.0)
        cfg.setdefault("anti_explosion_vanishing_v5_eps", 1e-6)
        cfg.setdefault("anti_explosion_vanishing_v5_detach_reference", True)

        # Reward normalization / policy-gradient stability.
        # When False, PG optimizes raw discounted reward mean directly.
        cfg.setdefault("policy_gradient_normalize_rewards", False)
        cfg.setdefault("reward_norm_eps", 1e-6)
        cfg.setdefault("reward_norm_clip", 10.0)
        cfg.setdefault("discount", 1.0)
        # Lipschitz safeguards for differentiable rollout stability.
        # Enabling this projects sampled linear maps by Frobenius norm:
        # ||W||_2 <= ||W||_F <= lipschitz_weight_fro_norm_max.
        cfg.setdefault("lipschitz_enforce", False)
        cfg.setdefault("lipschitz_weight_fro_norm_max", 1.0)
        # GP effective output-scale cap used in Jacobian bound.
        cfg.setdefault("lipschitz_gp_outputscale_max", 1.0)

        # SCM-style knobs (aligned with names in priors/mlp.py).
        cfg.setdefault(
            "num_layers",
            {"distribution": "meta_gamma", "max_alpha": 2, "max_scale": 3, "round": True, "lower_bound": 2},
        )
        cfg.setdefault(
            "prior_mlp_hidden_dim",
            {"distribution": "meta_gamma", "max_alpha": 3, "max_scale": 128, "round": True, "lower_bound": 8},
        )
        cfg.setdefault(
            "prior_mlp_activations",
            {"distribution": "meta_choice", "choice_values": [torch.nn.Tanh, torch.nn.ReLU, torch.nn.Identity]},
        )
        cfg.setdefault("init_std", {"distribution": "log_uniform", "min": 1e-3, "max": 1.0})
        cfg.setdefault("noise_std", {"distribution": "log_uniform", "min": 1e-4, "max": 0.2})

        # GP-style knobs (aligned with names in priors/fast_gp.py).
        cfg.setdefault("lengthscale", {"distribution": "log_uniform", "min": 1e-5, "max": 8.0})
        cfg.setdefault("outputscale", {"distribution": "log_uniform", "min": 1e-5, "max": 8.0})
        cfg.setdefault("noise", {"distribution": "meta_choice", "choice_values": [1e-5, 1e-4, 1e-2]})
        cfg.setdefault("gp_rff_features", {"distribution": "uniform_int", "min": 32, "max": 256})

        self.config = parse_distributions(cfg)
        self.last_runtime_info = []
        self.last_rollout_profile = None
        self.last_rollout_v2 = None
        self.last_rollout_v3 = None
        self.last_rollout_v4 = None
        self._rollout_executor = None
        self._rollout_executor_workers = 0
        envgen_bmm_flag = str(os.environ.get("TICL_POLICY_ENVGEN_BMM", "1")).strip().lower()
        self.envgen_bmm = envgen_bmm_flag not in {"0", "false", "no", "off"}
        envgen_ragged_affine_flag = str(
            os.environ.get("TICL_POLICY_ENVGEN_RAGGED_AFFINE", "0")
        ).strip().lower()
        self.envgen_ragged_affine = (
            envgen_ragged_affine_flag not in {"0", "false", "no", "off"}
            and (triton is not None)
        )
        fused_transition_flag = str(
            os.environ.get("TICL_POLICY_FUSED_TRANSITION_GENERATOR", "1")
        ).strip().lower()
        self.fused_transition_generator = fused_transition_flag not in {"0", "false", "no", "off"}
        envgen_checkpoint_flag = str(
            os.environ.get("TICL_POLICY_ENVGEN_CHECKPOINT", "0")
        ).strip().lower()
        self.envgen_checkpoint = envgen_checkpoint_flag not in {"0", "false", "no", "off"}
        envgen_checkpoint_reentrant_flag = str(
            os.environ.get("TICL_POLICY_ENVGEN_CHECKPOINT_REENTRANT", "0")
        ).strip().lower()
        self.envgen_checkpoint_reentrant = envgen_checkpoint_reentrant_flag in {
            "1",
            "true",
            "yes",
            "on",
        }
        fused_transition_scm_hidden_fused_flag = str(
            os.environ.get("TICL_POLICY_FUSED_TRANSITION_SCM_HIDDEN_FUSED", "0")
        ).strip().lower()
        self.fused_transition_scm_hidden_fused = fused_transition_scm_hidden_fused_flag not in {
            "0",
            "false",
            "no",
            "off",
        }
        fused_transition_gp_input_fused_flag = str(
            os.environ.get("TICL_POLICY_FUSED_TRANSITION_GP_INPUT_FUSED", "0")
        ).strip().lower()
        self.fused_transition_gp_input_fused = fused_transition_gp_input_fused_flag not in {
            "0",
            "false",
            "no",
            "off",
        }
        fused_transition_gp_output_fused_flag = str(
            os.environ.get("TICL_POLICY_FUSED_TRANSITION_GP_OUTPUT_FUSED", "0")
        ).strip().lower()
        self.fused_transition_gp_output_fused = fused_transition_gp_output_fused_flag not in {
            "0",
            "false",
            "no",
            "off",
        }
        fused_transition_gp_output_subgraph_flag = str(
            os.environ.get("TICL_POLICY_FUSED_TRANSITION_GP_OUTPUT_SUBGRAPH", "0")
        ).strip().lower()
        self.fused_transition_gp_output_subgraph = fused_transition_gp_output_subgraph_flag not in {
            "0",
            "false",
            "no",
            "off",
        }
        fused_transition_gp_rff_fused_flag = str(
            os.environ.get("TICL_POLICY_FUSED_TRANSITION_GP_RFF_FUSED", "0")
        ).strip().lower()
        self.fused_transition_gp_rff_fused = fused_transition_gp_rff_fused_flag not in {
            "0",
            "false",
            "no",
            "off",
        }
        fused_transition_gp_packed_env_input_flag = str(
            os.environ.get("TICL_POLICY_FUSED_TRANSITION_GP_PACKED_ENV_INPUT", "0")
        ).strip().lower()
        self.fused_transition_gp_packed_env_input = fused_transition_gp_packed_env_input_flag not in {
            "0",
            "false",
            "no",
            "off",
        }
        fused_transition_gp_shared_first_proj_flag = str(
            os.environ.get("TICL_POLICY_FUSED_TRANSITION_GP_SHARED_FIRST_PROJ", "0")
        ).strip().lower()
        self.fused_transition_gp_shared_first_proj = fused_transition_gp_shared_first_proj_flag not in {
            "0",
            "false",
            "no",
            "off",
        }
        self.gp_rff_tiled_block_o = self._resolve_tiled_block_size(
            os.environ.get("TICL_POLICY_GP_RFF_BLOCK_O", "32"),
            default=32,
        )
        self.gp_rff_tiled_block_k = self._resolve_tiled_block_size(
            os.environ.get("TICL_POLICY_GP_RFF_BLOCK_K", "32"),
            default=32,
        )
        self.gp_rff_tiled_num_warps = self._resolve_tiled_num_warps(
            os.environ.get("TICL_POLICY_GP_RFF_NUM_WARPS", "4"),
            default=4,
        )
        profile_gp_projection_timing_flag = str(
            os.environ.get("TICL_PROFILE_GP_PROJECTION_TIMING", "0")
        ).strip().lower()
        self.profile_gp_projection_timing = profile_gp_projection_timing_flag in {
            "1",
            "true",
            "yes",
            "on",
        }
        transition_inner_grouping = str(
            os.environ.get("TICL_POLICY_TRANSITION_INNER_GROUPING", "family")
        ).strip().lower()
        if transition_inner_grouping in {"", "1", "true", "yes", "on"}:
            transition_inner_grouping = "structure"
        if transition_inner_grouping in {"0", "false", "no", "off"}:
            transition_inner_grouping = "family"
        if transition_inner_grouping not in {"family", "structure", "pow2", "pow2_no_depth"}:
            transition_inner_grouping = "family"
        self.transition_inner_grouping = transition_inner_grouping
        try:
            transition_inner_min_bucket = int(os.environ.get("TICL_POLICY_TRANSITION_INNER_MIN_BUCKET", "0"))
        except Exception:
            transition_inner_min_bucket = 0
        self.transition_inner_min_bucket = max(0, transition_inner_min_bucket)

    def __del__(self):
        executor = getattr(self, "_rollout_executor", None)
        if executor is not None:
            executor.shutdown(wait=False)

    @staticmethod
    def _resolve_activation(activation):
        if isinstance(activation, nn.Module):
            return activation
        if isinstance(activation, type) and issubclass(activation, nn.Module):
            return activation()
        if isinstance(activation, str):
            act = activation.strip().lower()
            if act == "relu":
                return nn.ReLU()
            if act in {"identity", "linear", "none"}:
                return nn.Identity()
            if act == "tanh":
                return nn.Tanh()
        return nn.Tanh()

    @staticmethod
    def _resolve_scalar(value):
        return float(sample_distributions({"value": value})["value"])

    @staticmethod
    def _optional_positive_scalar(value):
        if value is None:
            return None
        try:
            v = float(value)
        except (TypeError, ValueError):
            return None
        if (not math.isfinite(v)) or v <= 0.0:
            return None
        return v

    @staticmethod
    def _project_matrix_fro_norm(matrix, max_fro_norm):
        """
        Project matrix/matrix-batch to a Frobenius-norm ball.
        This guarantees ||W||_2 <= ||W||_F <= max_fro_norm.
        """
        if max_fro_norm is None:
            return matrix
        if not torch.is_floating_point(matrix):
            return matrix
        eps = float(torch.finfo(matrix.dtype).eps)
        if matrix.ndim == 2:
            max_norm = float(max_fro_norm)
            if (not math.isfinite(max_norm)) or max_norm <= 0.0:
                return matrix
            fro = torch.linalg.matrix_norm(matrix, ord="fro")
            scale = torch.clamp(torch.as_tensor(max_norm, device=matrix.device, dtype=matrix.dtype) / (fro + eps), max=1.0)
            return matrix * scale
        if matrix.ndim == 3:
            if torch.is_tensor(max_fro_norm):
                max_norm = max_fro_norm.to(device=matrix.device, dtype=matrix.dtype).reshape(-1, 1)
            else:
                max_norm = torch.full(
                    (matrix.shape[0], 1),
                    float(max_fro_norm),
                    device=matrix.device,
                    dtype=matrix.dtype,
                )
            fro = torch.linalg.vector_norm(matrix.reshape(matrix.shape[0], -1), dim=1, keepdim=True)
            scale = torch.clamp(max_norm / (fro + eps), max=1.0)
            return matrix * scale.reshape(-1, 1, 1)
        raise ValueError(f"expected 2D or 3D matrix tensor, got shape={tuple(matrix.shape)}")

    @staticmethod
    def _resolve_lipschitz_weight_cap(h):
        if not bool(h.get("lipschitz_enforce", False)):
            return None
        return EnvironmentPrior._optional_positive_scalar(h.get("lipschitz_weight_fro_norm_max", 1.0))

    @staticmethod
    def _resolve_lipschitz_gp_outputscale_cap(h):
        if not bool(h.get("lipschitz_enforce", False)):
            return None
        return EnvironmentPrior._optional_positive_scalar(h.get("lipschitz_gp_outputscale_max", 1.0))

    def _build_scm_fn(self, in_dim, out_dim, h, device, generator=None):
        depth = max(2, int(h["num_layers"]))
        hidden = max(int(out_dim), int(h["prior_mlp_hidden_dim"]))
        init_std = float(h["init_std"])
        noise_std = float(h["noise_std"])
        activation = self._resolve_activation(h["prior_mlp_activations"])
        weight_cap = self._resolve_lipschitz_weight_cap(h)

        layer_dims = [in_dim] + [hidden] * (depth - 1) + [out_dim]
        weights = []
        biases = []
        for d_in, d_out in zip(layer_dims[:-1], layer_dims[1:]):
            if generator is None:
                w = torch.randn(d_in, d_out, device=device) * (init_std / math.sqrt(max(1, d_in)))
                b = torch.randn(d_out, device=device) * (init_std * 0.1)
            else:
                w = torch.randn(d_in, d_out, device=device, generator=generator) * (init_std / math.sqrt(max(1, d_in)))
                b = torch.randn(d_out, device=device, generator=generator) * (init_std * 0.1)
            w = self._project_matrix_fro_norm(w, weight_cap)
            weights.append(w)
            biases.append(b)

        def fn(x, generator=None):
            z = x
            for i, (w, b) in enumerate(zip(weights, biases)):
                z = z @ w + b
                if i < len(weights) - 1:
                    z = activation(z)
            if noise_std > 0:
                if generator is None:
                    z = z + torch.randn_like(z) * noise_std
                else:
                    z = z + torch.randn(z.shape, device=z.device, dtype=z.dtype, generator=generator) * noise_std
            return torch.tanh(z)

        return fn

    @staticmethod
    def _build_gp_fn(in_dim, out_dim, h, device, generator=None):
        m = max(8, int(h["gp_rff_features"]))
        lengthscale = max(1e-6, float(h["lengthscale"]))
        outputscale = float(h["outputscale"])
        noise = float(h["noise"])
        weight_cap = EnvironmentPrior._resolve_lipschitz_weight_cap(h)
        outputscale_cap = EnvironmentPrior._resolve_lipschitz_gp_outputscale_cap(h)
        if outputscale_cap is not None:
            outputscale = math.copysign(min(abs(outputscale), float(outputscale_cap)), outputscale)

        if generator is None:
            w = torch.randn(in_dim, m, device=device) / lengthscale
            b = 2.0 * math.pi * torch.rand(m, device=device)
            a = torch.randn(m, out_dim, device=device) / math.sqrt(max(1, m))
        else:
            w = torch.randn(in_dim, m, device=device, generator=generator) / lengthscale
            b = 2.0 * math.pi * torch.rand(m, device=device, generator=generator)
            a = torch.randn(m, out_dim, device=device, generator=generator) / math.sqrt(max(1, m))
        w = EnvironmentPrior._project_matrix_fro_norm(w, weight_cap)
        a = EnvironmentPrior._project_matrix_fro_norm(a, weight_cap)

        def fn(x, generator=None):
            phi = torch.cos(x @ w + b)
            y = outputscale * (phi @ a)
            if noise > 0:
                if generator is None:
                    y = y + torch.randn_like(y) * noise
                else:
                    y = y + torch.randn(y.shape, device=y.device, dtype=y.dtype, generator=generator) * noise
            return torch.tanh(y)

        return fn

    @staticmethod
    def _fit_to_num_features(x, num_features):
        d = int(x.shape[-1])
        if d == num_features:
            return x
        if d > num_features:
            return x[..., :num_features]
        pad = torch.zeros(
            x.shape[0],
            num_features - d,
            device=x.device,
            dtype=x.dtype,
        )
        return torch.cat([x, pad], dim=-1)

    @staticmethod
    def _resolve_tiled_block_size(v, default=32):
        try:
            v = int(v)
        except Exception:
            v = int(default)
        if v not in {16, 32, 64, 128}:
            v = int(default)
        return int(v)

    @staticmethod
    def _resolve_tiled_num_warps(v, default=4):
        try:
            v = int(v)
        except Exception:
            v = int(default)
        if v not in {1, 2, 4, 8}:
            v = int(default)
        return int(v)

    @staticmethod
    def _clamp_int(v, lo, hi):
        return int(max(lo, min(hi, int(v))))

    def _sample_single_eval_pos(self, n_samples, single_eval_pos):
        if single_eval_pos is None:
            single_eval_pos = np.random.randint(1, n_samples)
        return int(max(1, min(int(n_samples) - 1, int(single_eval_pos))))

    @staticmethod
    def _resolve_rollout_noise_block_size(n_samples):
        stream_flag = str(os.environ.get("TICL_POLICY_ROLLOUT_NOISE_STREAM", "1")).strip().lower()
        if stream_flag in {"0", "false", "no", "off"}:
            return 0
        try:
            block_size = int(os.environ.get("TICL_POLICY_ROLLOUT_NOISE_BLOCK_SIZE", "64"))
        except Exception:
            block_size = 64
        if block_size <= 0:
            return 0
        n_samples = int(max(1, n_samples))
        block_size = int(min(n_samples, block_size))
        if block_size >= n_samples:
            return 0
        return block_size

    def _sample_dims(self, h):
        action_dim = self._clamp_int(h["action_dim"], 1, 30)
        state_dim = self._clamp_int(h["state_dim"], 1, 400)
        obs_dim = self._clamp_int(h["obs_dim"], 1, 400)
        obs_dim = min(obs_dim, state_dim)  # obs is subset of state.
        noise_dim = max(1, int(h["noise_dim"]))
        zero_pad_dim = max(0, int(h["zero_pad_dim"]))
        return state_dim, obs_dim, action_dim, noise_dim, zero_pad_dim

    @staticmethod
    def _pack_env_input(state_t, obs_t, action_t, noise_t, zero_pad_t):
        return torch.cat([state_t, obs_t, action_t, noise_t, zero_pad_t], dim=-1)

    @staticmethod
    def _fit_to_slot(x, slot_dim):
        d = int(x.shape[-1])
        slot_dim = int(slot_dim)
        if d == slot_dim:
            return x
        if d > slot_dim:
            return x[..., :slot_dim]
        pad_shape = list(x.shape)
        pad_shape[-1] = slot_dim - d
        pad = torch.zeros(pad_shape, device=x.device, dtype=x.dtype)
        return torch.cat([x, pad], dim=-1)

    @staticmethod
    def _pack_pfn_input(obs_slot_t, action_slot_t, reward_t, reward_mask_t):
        reward_vec = reward_t.reshape(1)
        mask_vec = reward_mask_t.reshape(1)
        # Two-head layout:
        #   head-1: [obs_slot, reward, reward_mask]
        #   head-2: [action_slot]
        return torch.cat([obs_slot_t, reward_vec, mask_vec, action_slot_t], dim=-1)

    @staticmethod
    def _sample_reward_dropout_ratio(h):
        if not bool(h.get("reward_dropout_enabled", True)):
            return 0.0
        if bool(h.get("reward_dropout_randomize", True)):
            lo = float(h.get("reward_dropout_ratio_min", 0.1))
            hi = float(h.get("reward_dropout_ratio_max", 1.0))
            lo, hi = min(lo, hi), max(lo, hi)
            return float(np.random.uniform(lo, hi))
        return float(h.get("reward_dropout_ratio", 0.0))

    @staticmethod
    def _coerce_bool(value):
        if torch.is_tensor(value):
            if value.dtype == torch.bool:
                return value
            if value.numel() == 1:
                return bool(value.item())
            return value != 0
        if isinstance(value, str):
            return value.strip().lower() not in {"0", "false", "no", "off", ""}
        return bool(value)

    @staticmethod
    def _resolve_state_highway_enabled(h):
        return EnvironmentPrior._coerce_bool(h.get("state_highway_enabled", False))

    @staticmethod
    def _resolve_state_highway_lambda(h):
        v = EnvironmentPrior._resolve_scalar(h.get("state_highway_lambda", 0.0))
        if not math.isfinite(v):
            return 0.0
        return float(min(1.0, max(0.0, v)))

    @staticmethod
    def _apply_state_postprocess(
        state_next_raw,
        state_prev,
        state_clip,
        state_highway_enabled=False,
        state_highway_lambda=0.0,
    ):
        clip = state_clip
        if not torch.is_tensor(clip):
            clip = torch.as_tensor(clip, device=state_next_raw.device, dtype=state_next_raw.dtype)
        else:
            clip = clip.to(device=state_next_raw.device, dtype=state_next_raw.dtype)
        if clip.ndim == state_next_raw.ndim - 1:
            clip = clip.unsqueeze(-1)

        state_bounded = torch.maximum(torch.minimum(state_next_raw, clip), -clip)
        state_bounded = torch.tanh(state_bounded)

        enabled = EnvironmentPrior._coerce_bool(state_highway_enabled)
        if torch.is_tensor(enabled):
            enabled_mask = enabled.to(device=state_next_raw.device)
            if enabled_mask.dtype != torch.bool:
                enabled_mask = enabled_mask != 0
            if enabled_mask.ndim == state_next_raw.ndim - 1:
                enabled_mask = enabled_mask.unsqueeze(-1)
            if not bool(enabled_mask.any().item()):
                return state_bounded
        elif not bool(enabled):
            return state_bounded
        else:
            enabled_mask = None

        lam = state_highway_lambda
        if torch.is_tensor(lam):
            lam_t = lam.to(device=state_next_raw.device, dtype=state_next_raw.dtype)
            if lam_t.ndim == state_next_raw.ndim - 1:
                lam_t = lam_t.unsqueeze(-1)
            lam_t = lam_t.clamp(0.0, 1.0)
        else:
            lam_t = float(lam)
            if not math.isfinite(lam_t):
                lam_t = 0.0
            lam_t = min(1.0, max(0.0, lam_t))

        state_mixed = lam_t * state_prev + (1.0 - lam_t) * state_bounded
        if enabled_mask is not None:
            return torch.where(enabled_mask, state_mixed, state_bounded)
        return state_mixed

    def _resolve_aev2_config(self):
        enabled = self._coerce_bool(self.config.get("anti_explosion_vanishing_v2_enabled", False))
        enabled = bool(enabled)
        lam = self._resolve_scalar(self.config.get("anti_explosion_vanishing_v2_lambda", 0.05))
        if (not math.isfinite(lam)) or lam < 0.0:
            lam = 0.0
        gain_lo = self._resolve_scalar(self.config.get("anti_explosion_vanishing_v2_gain_lo", 0.85))
        gain_hi = self._resolve_scalar(self.config.get("anti_explosion_vanishing_v2_gain_hi", 1.15))
        if (not math.isfinite(gain_lo)) or gain_lo <= 0.0:
            gain_lo = 0.85
        if (not math.isfinite(gain_hi)) or gain_hi <= 0.0:
            gain_hi = 1.15
        gain_lo = max(1e-6, float(gain_lo))
        gain_hi = max(gain_lo + 1e-6, float(gain_hi))
        huber_delta = self._resolve_scalar(self.config.get("anti_explosion_vanishing_v2_huber_delta", 0.05))
        if (not math.isfinite(huber_delta)) or huber_delta < 0.0:
            huber_delta = 0.05
        eps = self._resolve_scalar(self.config.get("anti_explosion_vanishing_v2_eps", 1e-6))
        if (not math.isfinite(eps)) or eps <= 0.0:
            eps = 1e-6
        detach_reference = self._coerce_bool(
            self.config.get("anti_explosion_vanishing_v2_detach_reference", True)
        )
        return {
            "enabled": bool(enabled),
            "lambda": float(lam),
            "gain_lo": float(gain_lo),
            "gain_hi": float(gain_hi),
            "log_gain_lo": float(math.log(gain_lo)),
            "log_gain_hi": float(math.log(gain_hi)),
            "huber_delta": float(huber_delta),
            "eps": float(eps),
            "detach_reference": bool(detach_reference),
        }

    @staticmethod
    def _aev2_new_accumulator(enabled, device, dtype):
        return {
            "enabled": bool(enabled),
            "penalty_sum": torch.zeros((), device=device, dtype=dtype),
            "pairs": 0,
            "gain_sum": torch.zeros((), device=device, dtype=torch.float64),
            "gain_sumsq": torch.zeros((), device=device, dtype=torch.float64),
            "gain_count": 0,
            "gain_min": None,
            "gain_max": None,
        }

    @staticmethod
    def _aev2_update_accumulator(acc, prev_delta, curr_delta, aev2_cfg):
        if (not bool(acc.get("enabled", False))) or (prev_delta is None):
            return

        prev = prev_delta
        curr = curr_delta
        if prev.ndim == 1:
            prev = prev.unsqueeze(0)
            curr = curr.unsqueeze(0)

        eps = float(aev2_cfg["eps"])
        prev_rms = torch.sqrt(torch.mean(prev * prev, dim=-1) + eps)
        if bool(aev2_cfg.get("detach_reference", True)):
            prev_rms = prev_rms.detach()
        curr_rms = torch.sqrt(torch.mean(curr * curr, dim=-1) + eps)
        gain = curr_rms / (prev_rms + eps)
        log_gain = torch.log(gain + eps)

        v_hi = F.relu(log_gain - float(aev2_cfg["log_gain_hi"]))
        v_lo = F.relu(float(aev2_cfg["log_gain_lo"]) - log_gain)
        violation = v_hi + v_lo
        huber_delta = float(aev2_cfg["huber_delta"])
        if huber_delta > 0.0:
            delta_t = torch.as_tensor(huber_delta, device=violation.device, dtype=violation.dtype)
            penalty_vec = torch.where(
                violation <= delta_t,
                0.5 * violation * violation / delta_t,
                violation - 0.5 * delta_t,
            )
        else:
            penalty_vec = violation * violation

        acc["penalty_sum"] = acc["penalty_sum"] + penalty_vec.mean()
        acc["pairs"] = int(acc["pairs"]) + 1

        gain_det = gain.detach()
        gain_det64 = gain_det.to(dtype=torch.float64)
        acc["gain_sum"] = acc["gain_sum"] + gain_det64.sum()
        acc["gain_sumsq"] = acc["gain_sumsq"] + (gain_det64 * gain_det64).sum()
        acc["gain_count"] = int(acc["gain_count"]) + int(gain_det64.numel())
        gain_min = gain_det.min().detach()
        gain_max = gain_det.max().detach()
        acc["gain_min"] = gain_min if acc["gain_min"] is None else torch.minimum(acc["gain_min"], gain_min)
        acc["gain_max"] = gain_max if acc["gain_max"] is None else torch.maximum(acc["gain_max"], gain_max)

    @staticmethod
    def _aev2_finalize_accumulator(acc, device, dtype, detach_penalty=False):
        enabled = bool(acc.get("enabled", False))
        penalty_sum = acc.get("penalty_sum", None)
        pairs = int(acc.get("pairs", 0))
        if penalty_sum is None:
            penalty_mean = torch.zeros((), device=device, dtype=dtype)
        elif pairs > 0:
            penalty_mean = penalty_sum / float(max(1, pairs))
        else:
            penalty_mean = torch.zeros((), device=device, dtype=penalty_sum.dtype)
        if detach_penalty and torch.is_tensor(penalty_mean):
            penalty_mean = penalty_mean.detach()

        gain_count = int(acc.get("gain_count", 0))
        gain_sum = acc.get("gain_sum", torch.zeros((), device=device, dtype=torch.float64)).detach()
        gain_sumsq = acc.get("gain_sumsq", torch.zeros((), device=device, dtype=torch.float64)).detach()
        if gain_count > 0:
            gain_mean64 = gain_sum / float(gain_count)
            gain_var64 = (gain_sumsq / float(gain_count)) - (gain_mean64 * gain_mean64)
            gain_std64 = torch.sqrt(torch.clamp(gain_var64, min=0.0))
            gain_mean = gain_mean64.to(dtype=dtype).detach()
            gain_std = gain_std64.to(dtype=dtype).detach()
        else:
            gain_mean = torch.zeros((), device=device, dtype=dtype)
            gain_std = torch.zeros((), device=device, dtype=dtype)

        gain_min = acc.get("gain_min", None)
        if gain_min is None:
            gain_min = torch.zeros((), device=device, dtype=dtype)
        else:
            gain_min = gain_min.to(device=device, dtype=dtype).detach()
        gain_max = acc.get("gain_max", None)
        if gain_max is None:
            gain_max = torch.zeros((), device=device, dtype=dtype)
        else:
            gain_max = gain_max.to(device=device, dtype=dtype).detach()

        return {
            "enabled": int(enabled),
            "penalty_mean": penalty_mean,
            "pairs": int(pairs),
            "gain_sum": gain_sum,
            "gain_sumsq": gain_sumsq,
            "gain_count": int(gain_count),
            "gain_mean": gain_mean,
            "gain_std": gain_std,
            "gain_min": gain_min,
            "gain_max": gain_max,
        }

    @staticmethod
    def _aev2_accumulate_window_summary(det_acc, window_summary, device):
        if (not isinstance(det_acc, dict)) or (not isinstance(window_summary, dict)):
            return
        det_acc["pairs"] = int(det_acc.get("pairs", 0)) + int(window_summary.get("pairs", 0))
        penalty_w = window_summary.get("penalty_mean", None)
        if torch.is_tensor(penalty_w):
            det_acc["penalty_sum"] = det_acc["penalty_sum"] + (
                penalty_w.detach() * float(max(1, int(window_summary.get("pairs", 0))))
            )
        gain_sum_w = window_summary.get("gain_sum", None)
        if torch.is_tensor(gain_sum_w):
            det_acc["gain_sum"] = det_acc["gain_sum"] + gain_sum_w.detach().to(
                device=device, dtype=torch.float64
            )
        gain_sumsq_w = window_summary.get("gain_sumsq", None)
        if torch.is_tensor(gain_sumsq_w):
            det_acc["gain_sumsq"] = det_acc["gain_sumsq"] + gain_sumsq_w.detach().to(
                device=device, dtype=torch.float64
            )
        det_acc["gain_count"] = int(det_acc.get("gain_count", 0)) + int(window_summary.get("gain_count", 0))
        gain_min_w = window_summary.get("gain_min", None)
        if torch.is_tensor(gain_min_w):
            det_acc["gain_min"] = (
                gain_min_w.detach()
                if det_acc.get("gain_min", None) is None
                else torch.minimum(det_acc["gain_min"], gain_min_w.detach())
            )
        gain_max_w = window_summary.get("gain_max", None)
        if torch.is_tensor(gain_max_w):
            det_acc["gain_max"] = (
                gain_max_w.detach()
                if det_acc.get("gain_max", None) is None
                else torch.maximum(det_acc["gain_max"], gain_max_w.detach())
            )

    @staticmethod
    def _aev2_merge_rollout_summary(acc, summary, batch_weight, device, dtype):
        if not isinstance(summary, dict):
            return acc
        if int(summary.get("enabled", 0)) == 0:
            return acc

        w = float(max(0.0, batch_weight))
        if acc is None:
            acc = {
                "enabled": 1,
                "penalty_mean_weighted": None,
                "gain_sum": torch.zeros((), device=device, dtype=torch.float64),
                "gain_sumsq": torch.zeros((), device=device, dtype=torch.float64),
                "gain_count": 0,
                "gain_min": None,
                "gain_max": None,
            }

        penalty_mean = summary.get("penalty_mean", None)
        if torch.is_tensor(penalty_mean):
            term = penalty_mean * w
            if acc["penalty_mean_weighted"] is None:
                acc["penalty_mean_weighted"] = term
            else:
                acc["penalty_mean_weighted"] = acc["penalty_mean_weighted"] + term

        gain_sum = summary.get("gain_sum", None)
        if torch.is_tensor(gain_sum):
            acc["gain_sum"] = acc["gain_sum"] + gain_sum.to(device=device, dtype=torch.float64)
        gain_sumsq = summary.get("gain_sumsq", None)
        if torch.is_tensor(gain_sumsq):
            acc["gain_sumsq"] = acc["gain_sumsq"] + gain_sumsq.to(device=device, dtype=torch.float64)
        acc["gain_count"] = int(acc["gain_count"]) + int(summary.get("gain_count", 0))

        gain_min = summary.get("gain_min", None)
        if torch.is_tensor(gain_min):
            gain_min = gain_min.to(device=device, dtype=dtype)
            acc["gain_min"] = gain_min if acc["gain_min"] is None else torch.minimum(acc["gain_min"], gain_min)
        gain_max = summary.get("gain_max", None)
        if torch.is_tensor(gain_max):
            gain_max = gain_max.to(device=device, dtype=dtype)
            acc["gain_max"] = gain_max if acc["gain_max"] is None else torch.maximum(acc["gain_max"], gain_max)
        return acc

    @staticmethod
    def _aev2_finalize_rollout_summary(acc, device, dtype):
        if acc is None:
            return None
        penalty_weighted = acc.get("penalty_mean_weighted", None)
        if penalty_weighted is None:
            penalty_weighted = torch.zeros((), device=device, dtype=dtype)
        gain_count = int(acc.get("gain_count", 0))
        gain_sum = acc.get("gain_sum", torch.zeros((), device=device, dtype=torch.float64)).detach()
        gain_sumsq = acc.get("gain_sumsq", torch.zeros((), device=device, dtype=torch.float64)).detach()
        if gain_count > 0:
            gain_mean64 = gain_sum / float(gain_count)
            gain_var64 = (gain_sumsq / float(gain_count)) - (gain_mean64 * gain_mean64)
            gain_std64 = torch.sqrt(torch.clamp(gain_var64, min=0.0))
            gain_mean = gain_mean64.to(dtype=dtype).detach()
            gain_std = gain_std64.to(dtype=dtype).detach()
        else:
            gain_mean = torch.zeros((), device=device, dtype=dtype)
            gain_std = torch.zeros((), device=device, dtype=dtype)
        gain_min = acc.get("gain_min", None)
        if gain_min is None:
            gain_min = torch.zeros((), device=device, dtype=dtype)
        else:
            gain_min = gain_min.detach()
        gain_max = acc.get("gain_max", None)
        if gain_max is None:
            gain_max = torch.zeros((), device=device, dtype=dtype)
        else:
            gain_max = gain_max.detach()
        return {
            "enabled": 1,
            "penalty_mean": penalty_weighted,
            "gain_sum": gain_sum,
            "gain_sumsq": gain_sumsq,
            "gain_count": gain_count,
            "gain_mean": gain_mean,
            "gain_std": gain_std,
            "gain_min": gain_min,
            "gain_max": gain_max,
        }

    def _resolve_aev3_config(self):
        enabled = self._coerce_bool(self.config.get("anti_explosion_vanishing_v3_enabled", False))
        enabled = bool(enabled)
        lam_drift = self._resolve_scalar(self.config.get("anti_explosion_vanishing_v3_lambda_drift", 0.02))
        if (not math.isfinite(lam_drift)) or lam_drift < 0.0:
            lam_drift = 0.0
        lam_tail = self._resolve_scalar(self.config.get("anti_explosion_vanishing_v3_lambda_tail", 0.05))
        if (not math.isfinite(lam_tail)) or lam_tail < 0.0:
            lam_tail = 0.0
        gain_lo = self._resolve_scalar(self.config.get("anti_explosion_vanishing_v3_gain_lo", 0.85))
        gain_hi = self._resolve_scalar(self.config.get("anti_explosion_vanishing_v3_gain_hi", 1.15))
        if (not math.isfinite(gain_lo)) or gain_lo <= 0.0:
            gain_lo = 0.85
        if (not math.isfinite(gain_hi)) or gain_hi <= 0.0:
            gain_hi = 1.15
        gain_lo = max(1e-6, float(gain_lo))
        gain_hi = max(gain_lo + 1e-6, float(gain_hi))
        tail_tau = self._resolve_scalar(self.config.get("anti_explosion_vanishing_v3_tail_tau", 0.02))
        if (not math.isfinite(tail_tau)) or tail_tau <= 0.0:
            tail_tau = 0.02
        eps = self._resolve_scalar(self.config.get("anti_explosion_vanishing_v3_eps", 1e-6))
        if (not math.isfinite(eps)) or eps <= 0.0:
            eps = 1e-6
        detach_reference = self._coerce_bool(
            self.config.get("anti_explosion_vanishing_v3_detach_reference", True)
        )
        return {
            "enabled": bool(enabled),
            "lambda_drift": float(lam_drift),
            "lambda_tail": float(lam_tail),
            "gain_lo": float(gain_lo),
            "gain_hi": float(gain_hi),
            "log_gain_lo": float(math.log(gain_lo)),
            "log_gain_hi": float(math.log(gain_hi)),
            "tail_tau": float(tail_tau),
            "eps": float(eps),
            "detach_reference": bool(detach_reference),
        }

    @staticmethod
    def _aev3_new_accumulator(enabled, device, dtype, aev3_cfg=None):
        if aev3_cfg is None:
            aev3_cfg = {}
        return {
            "enabled": bool(enabled),
            "lambda_drift": float(aev3_cfg.get("lambda_drift", 0.0)),
            "lambda_tail": float(aev3_cfg.get("lambda_tail", 0.0)),
            "gain_lo": float(aev3_cfg.get("gain_lo", 0.0)),
            "gain_hi": float(aev3_cfg.get("gain_hi", 0.0)),
            "log_gain_lo": float(aev3_cfg.get("log_gain_lo", 0.0)),
            "log_gain_hi": float(aev3_cfg.get("log_gain_hi", 0.0)),
            "drift_sum": torch.zeros((), device=device, dtype=dtype),
            "tail_sum": torch.zeros((), device=device, dtype=dtype),
            "pairs": 0,
            "log_gain_sum": torch.zeros((), device=device, dtype=torch.float64),
            "log_gain_sumsq": torch.zeros((), device=device, dtype=torch.float64),
            "log_gain_count": 0,
            "tail_low_count": 0,
            "tail_high_count": 0,
            "gain_sum": torch.zeros((), device=device, dtype=torch.float64),
            "gain_sumsq": torch.zeros((), device=device, dtype=torch.float64),
            "gain_count": 0,
            "gain_min": None,
            "gain_max": None,
        }

    @staticmethod
    def _aev3_update_accumulator(acc, prev_delta, curr_delta, aev3_cfg):
        if (not bool(acc.get("enabled", False))) or (prev_delta is None):
            return

        prev = prev_delta
        curr = curr_delta
        if prev.ndim == 1:
            prev = prev.unsqueeze(0)
            curr = curr.unsqueeze(0)

        eps = float(aev3_cfg["eps"])
        prev_rms = torch.sqrt(torch.mean(prev * prev, dim=-1) + eps)
        if bool(aev3_cfg.get("detach_reference", True)):
            prev_rms = prev_rms.detach()
        curr_rms = torch.sqrt(torch.mean(curr * curr, dim=-1) + eps)
        gain = curr_rms / (prev_rms + eps)
        log_gain = torch.log(gain + eps)

        tau = float(aev3_cfg.get("tail_tau", 0.02))
        if tau <= 0.0:
            tau = 1e-6
        tau_t = torch.as_tensor(tau, device=log_gain.device, dtype=log_gain.dtype)
        low_margin = (float(aev3_cfg["log_gain_lo"]) - log_gain) / tau_t
        high_margin = (log_gain - float(aev3_cfg["log_gain_hi"])) / tau_t
        tail_penalty_vec = (F.softplus(low_margin) + F.softplus(high_margin)) * tau_t

        acc["drift_sum"] = acc["drift_sum"] + log_gain.mean()
        acc["tail_sum"] = acc["tail_sum"] + tail_penalty_vec.mean()
        acc["pairs"] = int(acc["pairs"]) + 1

        log_gain_det64 = log_gain.detach().to(dtype=torch.float64)
        acc["log_gain_sum"] = acc["log_gain_sum"] + log_gain_det64.sum()
        acc["log_gain_sumsq"] = acc["log_gain_sumsq"] + (log_gain_det64 * log_gain_det64).sum()
        acc["log_gain_count"] = int(acc["log_gain_count"]) + int(log_gain_det64.numel())
        acc["tail_low_count"] = int(acc["tail_low_count"]) + int(
            (log_gain_det64 < float(aev3_cfg["log_gain_lo"])).sum().item()
        )
        acc["tail_high_count"] = int(acc["tail_high_count"]) + int(
            (log_gain_det64 > float(aev3_cfg["log_gain_hi"])).sum().item()
        )

        gain_det = gain.detach()
        gain_det64 = gain_det.to(dtype=torch.float64)
        acc["gain_sum"] = acc["gain_sum"] + gain_det64.sum()
        acc["gain_sumsq"] = acc["gain_sumsq"] + (gain_det64 * gain_det64).sum()
        acc["gain_count"] = int(acc["gain_count"]) + int(gain_det64.numel())
        gain_min = gain_det.min().detach()
        gain_max = gain_det.max().detach()
        acc["gain_min"] = gain_min if acc["gain_min"] is None else torch.minimum(acc["gain_min"], gain_min)
        acc["gain_max"] = gain_max if acc["gain_max"] is None else torch.maximum(acc["gain_max"], gain_max)

    @staticmethod
    def _aev3_finalize_accumulator(acc, device, dtype, detach_penalty=False):
        enabled = bool(acc.get("enabled", False))
        pairs = int(acc.get("pairs", 0))
        drift_sum = acc.get("drift_sum", None)
        tail_sum = acc.get("tail_sum", None)
        if pairs > 0 and torch.is_tensor(drift_sum):
            drift_mean = drift_sum / float(pairs)
        else:
            drift_mean = torch.zeros((), device=device, dtype=dtype)
        if pairs > 0 and torch.is_tensor(tail_sum):
            tail_mean = tail_sum / float(pairs)
        else:
            tail_mean = torch.zeros((), device=device, dtype=dtype)

        penalty_drift = drift_mean * drift_mean
        penalty_tail = tail_mean
        lambda_drift = float(acc.get("lambda_drift", 0.0))
        lambda_tail = float(acc.get("lambda_tail", 0.0))
        penalty_mean = (penalty_drift * lambda_drift) + (penalty_tail * lambda_tail)
        if detach_penalty:
            penalty_drift = penalty_drift.detach()
            penalty_tail = penalty_tail.detach()
            penalty_mean = penalty_mean.detach()

        log_gain_count = int(acc.get("log_gain_count", 0))
        log_gain_sum = acc.get("log_gain_sum", torch.zeros((), device=device, dtype=torch.float64)).detach()
        log_gain_sumsq = acc.get("log_gain_sumsq", torch.zeros((), device=device, dtype=torch.float64)).detach()
        if log_gain_count > 0:
            log_gain_mean64 = log_gain_sum / float(log_gain_count)
            log_gain_var64 = (log_gain_sumsq / float(log_gain_count)) - (log_gain_mean64 * log_gain_mean64)
            log_gain_std64 = torch.sqrt(torch.clamp(log_gain_var64, min=0.0))
            log_gain_mean = log_gain_mean64.to(dtype=dtype).detach()
            log_gain_std = log_gain_std64.to(dtype=dtype).detach()
        else:
            log_gain_mean = torch.zeros((), device=device, dtype=dtype)
            log_gain_std = torch.zeros((), device=device, dtype=dtype)

        tail_low_count = int(acc.get("tail_low_count", 0))
        tail_high_count = int(acc.get("tail_high_count", 0))
        if log_gain_count > 0:
            tail_low_share = torch.as_tensor(
                float(tail_low_count) / float(log_gain_count),
                device=device,
                dtype=dtype,
            )
            tail_high_share = torch.as_tensor(
                float(tail_high_count) / float(log_gain_count),
                device=device,
                dtype=dtype,
            )
        else:
            tail_low_share = torch.zeros((), device=device, dtype=dtype)
            tail_high_share = torch.zeros((), device=device, dtype=dtype)

        gain_count = int(acc.get("gain_count", 0))
        gain_sum = acc.get("gain_sum", torch.zeros((), device=device, dtype=torch.float64)).detach()
        gain_sumsq = acc.get("gain_sumsq", torch.zeros((), device=device, dtype=torch.float64)).detach()
        if gain_count > 0:
            gain_mean64 = gain_sum / float(gain_count)
            gain_var64 = (gain_sumsq / float(gain_count)) - (gain_mean64 * gain_mean64)
            gain_std64 = torch.sqrt(torch.clamp(gain_var64, min=0.0))
            gain_mean = gain_mean64.to(dtype=dtype).detach()
            gain_std = gain_std64.to(dtype=dtype).detach()
        else:
            gain_mean = torch.zeros((), device=device, dtype=dtype)
            gain_std = torch.zeros((), device=device, dtype=dtype)

        gain_min = acc.get("gain_min", None)
        if gain_min is None:
            gain_min = torch.zeros((), device=device, dtype=dtype)
        else:
            gain_min = gain_min.to(device=device, dtype=dtype).detach()
        gain_max = acc.get("gain_max", None)
        if gain_max is None:
            gain_max = torch.zeros((), device=device, dtype=dtype)
        else:
            gain_max = gain_max.to(device=device, dtype=dtype).detach()

        return {
            "enabled": int(enabled),
            "pairs": int(pairs),
            "lambda_drift": float(lambda_drift),
            "lambda_tail": float(lambda_tail),
            "gain_lo": float(acc.get("gain_lo", 0.0)),
            "gain_hi": float(acc.get("gain_hi", 0.0)),
            "penalty_drift": penalty_drift,
            "penalty_tail": penalty_tail,
            "penalty_mean": penalty_mean,
            "log_gain_sum": log_gain_sum,
            "log_gain_sumsq": log_gain_sumsq,
            "log_gain_count": int(log_gain_count),
            "log_gain_mean": log_gain_mean,
            "log_gain_std": log_gain_std,
            "tail_low_count": int(tail_low_count),
            "tail_high_count": int(tail_high_count),
            "tail_low_share": tail_low_share,
            "tail_high_share": tail_high_share,
            "gain_sum": gain_sum,
            "gain_sumsq": gain_sumsq,
            "gain_count": int(gain_count),
            "gain_mean": gain_mean,
            "gain_std": gain_std,
            "gain_min": gain_min,
            "gain_max": gain_max,
        }

    @staticmethod
    def _aev3_accumulate_window_summary(det_acc, window_summary, device):
        if (not isinstance(det_acc, dict)) or (not isinstance(window_summary, dict)):
            return
        pairs_w = int(window_summary.get("pairs", 0))
        det_acc["pairs"] = int(det_acc.get("pairs", 0)) + pairs_w
        penalty_drift_w = window_summary.get("penalty_drift", None)
        if torch.is_tensor(penalty_drift_w):
            det_acc["drift_sum"] = det_acc["drift_sum"] + (penalty_drift_w.detach() * float(max(1, pairs_w)))
        penalty_tail_w = window_summary.get("penalty_tail", None)
        if torch.is_tensor(penalty_tail_w):
            det_acc["tail_sum"] = det_acc["tail_sum"] + (penalty_tail_w.detach() * float(max(1, pairs_w)))

        log_gain_sum_w = window_summary.get("log_gain_sum", None)
        if torch.is_tensor(log_gain_sum_w):
            det_acc["log_gain_sum"] = det_acc["log_gain_sum"] + log_gain_sum_w.detach().to(
                device=device, dtype=torch.float64
            )
        log_gain_sumsq_w = window_summary.get("log_gain_sumsq", None)
        if torch.is_tensor(log_gain_sumsq_w):
            det_acc["log_gain_sumsq"] = det_acc["log_gain_sumsq"] + log_gain_sumsq_w.detach().to(
                device=device, dtype=torch.float64
            )
        det_acc["log_gain_count"] = int(det_acc.get("log_gain_count", 0)) + int(window_summary.get("log_gain_count", 0))
        det_acc["tail_low_count"] = int(det_acc.get("tail_low_count", 0)) + int(window_summary.get("tail_low_count", 0))
        det_acc["tail_high_count"] = int(det_acc.get("tail_high_count", 0)) + int(window_summary.get("tail_high_count", 0))

        gain_sum_w = window_summary.get("gain_sum", None)
        if torch.is_tensor(gain_sum_w):
            det_acc["gain_sum"] = det_acc["gain_sum"] + gain_sum_w.detach().to(device=device, dtype=torch.float64)
        gain_sumsq_w = window_summary.get("gain_sumsq", None)
        if torch.is_tensor(gain_sumsq_w):
            det_acc["gain_sumsq"] = det_acc["gain_sumsq"] + gain_sumsq_w.detach().to(
                device=device, dtype=torch.float64
            )
        det_acc["gain_count"] = int(det_acc.get("gain_count", 0)) + int(window_summary.get("gain_count", 0))
        gain_min_w = window_summary.get("gain_min", None)
        if torch.is_tensor(gain_min_w):
            det_acc["gain_min"] = (
                gain_min_w.detach()
                if det_acc.get("gain_min", None) is None
                else torch.minimum(det_acc["gain_min"], gain_min_w.detach())
            )
        gain_max_w = window_summary.get("gain_max", None)
        if torch.is_tensor(gain_max_w):
            det_acc["gain_max"] = (
                gain_max_w.detach()
                if det_acc.get("gain_max", None) is None
                else torch.maximum(det_acc["gain_max"], gain_max_w.detach())
            )

    @staticmethod
    def _aev3_merge_rollout_summary(acc, summary, batch_weight, device, dtype):
        if not isinstance(summary, dict):
            return acc
        if int(summary.get("enabled", 0)) == 0:
            return acc

        w = float(max(0.0, batch_weight))
        if acc is None:
            acc = {
                "enabled": 1,
                "lambda_drift": float(summary.get("lambda_drift", 0.0)),
                "lambda_tail": float(summary.get("lambda_tail", 0.0)),
                "gain_lo": float(summary.get("gain_lo", 0.0)),
                "gain_hi": float(summary.get("gain_hi", 0.0)),
                "penalty_mean_weighted": None,
                "penalty_drift_weighted": None,
                "penalty_tail_weighted": None,
                "log_gain_sum": torch.zeros((), device=device, dtype=torch.float64),
                "log_gain_sumsq": torch.zeros((), device=device, dtype=torch.float64),
                "log_gain_count": 0,
                "tail_low_count": 0,
                "tail_high_count": 0,
                "gain_sum": torch.zeros((), device=device, dtype=torch.float64),
                "gain_sumsq": torch.zeros((), device=device, dtype=torch.float64),
                "gain_count": 0,
                "gain_min": None,
                "gain_max": None,
            }

        penalty_mean = summary.get("penalty_mean", None)
        if torch.is_tensor(penalty_mean):
            term = penalty_mean * w
            if acc["penalty_mean_weighted"] is None:
                acc["penalty_mean_weighted"] = term
            else:
                acc["penalty_mean_weighted"] = acc["penalty_mean_weighted"] + term
        penalty_drift = summary.get("penalty_drift", None)
        if torch.is_tensor(penalty_drift):
            term = penalty_drift * w
            if acc["penalty_drift_weighted"] is None:
                acc["penalty_drift_weighted"] = term
            else:
                acc["penalty_drift_weighted"] = acc["penalty_drift_weighted"] + term
        penalty_tail = summary.get("penalty_tail", None)
        if torch.is_tensor(penalty_tail):
            term = penalty_tail * w
            if acc["penalty_tail_weighted"] is None:
                acc["penalty_tail_weighted"] = term
            else:
                acc["penalty_tail_weighted"] = acc["penalty_tail_weighted"] + term

        log_gain_sum = summary.get("log_gain_sum", None)
        if torch.is_tensor(log_gain_sum):
            acc["log_gain_sum"] = acc["log_gain_sum"] + log_gain_sum.to(device=device, dtype=torch.float64)
        log_gain_sumsq = summary.get("log_gain_sumsq", None)
        if torch.is_tensor(log_gain_sumsq):
            acc["log_gain_sumsq"] = acc["log_gain_sumsq"] + log_gain_sumsq.to(device=device, dtype=torch.float64)
        acc["log_gain_count"] = int(acc["log_gain_count"]) + int(summary.get("log_gain_count", 0))
        acc["tail_low_count"] = int(acc["tail_low_count"]) + int(summary.get("tail_low_count", 0))
        acc["tail_high_count"] = int(acc["tail_high_count"]) + int(summary.get("tail_high_count", 0))

        gain_sum = summary.get("gain_sum", None)
        if torch.is_tensor(gain_sum):
            acc["gain_sum"] = acc["gain_sum"] + gain_sum.to(device=device, dtype=torch.float64)
        gain_sumsq = summary.get("gain_sumsq", None)
        if torch.is_tensor(gain_sumsq):
            acc["gain_sumsq"] = acc["gain_sumsq"] + gain_sumsq.to(device=device, dtype=torch.float64)
        acc["gain_count"] = int(acc["gain_count"]) + int(summary.get("gain_count", 0))
        gain_min = summary.get("gain_min", None)
        if torch.is_tensor(gain_min):
            gain_min = gain_min.to(device=device, dtype=dtype)
            acc["gain_min"] = gain_min if acc["gain_min"] is None else torch.minimum(acc["gain_min"], gain_min)
        gain_max = summary.get("gain_max", None)
        if torch.is_tensor(gain_max):
            gain_max = gain_max.to(device=device, dtype=dtype)
            acc["gain_max"] = gain_max if acc["gain_max"] is None else torch.maximum(acc["gain_max"], gain_max)
        return acc

    @staticmethod
    def _aev3_finalize_rollout_summary(acc, device, dtype):
        if acc is None:
            return None
        penalty_mean = acc.get("penalty_mean_weighted", None)
        if penalty_mean is None:
            penalty_mean = torch.zeros((), device=device, dtype=dtype)
        penalty_drift = acc.get("penalty_drift_weighted", None)
        if penalty_drift is None:
            penalty_drift = torch.zeros((), device=device, dtype=dtype)
        penalty_tail = acc.get("penalty_tail_weighted", None)
        if penalty_tail is None:
            penalty_tail = torch.zeros((), device=device, dtype=dtype)

        log_gain_count = int(acc.get("log_gain_count", 0))
        log_gain_sum = acc.get("log_gain_sum", torch.zeros((), device=device, dtype=torch.float64)).detach()
        log_gain_sumsq = acc.get("log_gain_sumsq", torch.zeros((), device=device, dtype=torch.float64)).detach()
        if log_gain_count > 0:
            log_gain_mean64 = log_gain_sum / float(log_gain_count)
            log_gain_var64 = (log_gain_sumsq / float(log_gain_count)) - (log_gain_mean64 * log_gain_mean64)
            log_gain_std64 = torch.sqrt(torch.clamp(log_gain_var64, min=0.0))
            log_gain_mean = log_gain_mean64.to(dtype=dtype).detach()
            log_gain_std = log_gain_std64.to(dtype=dtype).detach()
            tail_low_share = torch.as_tensor(
                float(acc.get("tail_low_count", 0)) / float(log_gain_count),
                device=device,
                dtype=dtype,
            )
            tail_high_share = torch.as_tensor(
                float(acc.get("tail_high_count", 0)) / float(log_gain_count),
                device=device,
                dtype=dtype,
            )
        else:
            log_gain_mean = torch.zeros((), device=device, dtype=dtype)
            log_gain_std = torch.zeros((), device=device, dtype=dtype)
            tail_low_share = torch.zeros((), device=device, dtype=dtype)
            tail_high_share = torch.zeros((), device=device, dtype=dtype)

        gain_count = int(acc.get("gain_count", 0))
        gain_sum = acc.get("gain_sum", torch.zeros((), device=device, dtype=torch.float64)).detach()
        gain_sumsq = acc.get("gain_sumsq", torch.zeros((), device=device, dtype=torch.float64)).detach()
        if gain_count > 0:
            gain_mean64 = gain_sum / float(gain_count)
            gain_var64 = (gain_sumsq / float(gain_count)) - (gain_mean64 * gain_mean64)
            gain_std64 = torch.sqrt(torch.clamp(gain_var64, min=0.0))
            gain_mean = gain_mean64.to(dtype=dtype).detach()
            gain_std = gain_std64.to(dtype=dtype).detach()
        else:
            gain_mean = torch.zeros((), device=device, dtype=dtype)
            gain_std = torch.zeros((), device=device, dtype=dtype)

        gain_min = acc.get("gain_min", None)
        if gain_min is None:
            gain_min = torch.zeros((), device=device, dtype=dtype)
        else:
            gain_min = gain_min.detach()
        gain_max = acc.get("gain_max", None)
        if gain_max is None:
            gain_max = torch.zeros((), device=device, dtype=dtype)
        else:
            gain_max = gain_max.detach()

        return {
            "enabled": 1,
            "lambda_drift": float(acc.get("lambda_drift", 0.0)),
            "lambda_tail": float(acc.get("lambda_tail", 0.0)),
            "gain_lo": float(acc.get("gain_lo", 0.0)),
            "gain_hi": float(acc.get("gain_hi", 0.0)),
            "penalty_mean": penalty_mean,
            "penalty_drift": penalty_drift,
            "penalty_tail": penalty_tail,
            "log_gain_sum": log_gain_sum,
            "log_gain_sumsq": log_gain_sumsq,
            "log_gain_count": int(log_gain_count),
            "log_gain_mean": log_gain_mean,
            "log_gain_std": log_gain_std,
            "tail_low_count": int(acc.get("tail_low_count", 0)),
            "tail_high_count": int(acc.get("tail_high_count", 0)),
            "tail_low_share": tail_low_share,
            "tail_high_share": tail_high_share,
            "gain_sum": gain_sum,
            "gain_sumsq": gain_sumsq,
            "gain_count": int(gain_count),
            "gain_mean": gain_mean,
            "gain_std": gain_std,
            "gain_min": gain_min,
            "gain_max": gain_max,
        }

    def _resolve_aev4_config(self):
        enabled = self._coerce_bool(self.config.get("anti_explosion_vanishing_v4_enabled", False))
        enabled = bool(enabled)
        lam_drift = self._resolve_scalar(self.config.get("anti_explosion_vanishing_v4_lambda_drift", 0.08))
        if (not math.isfinite(lam_drift)) or lam_drift < 0.0:
            lam_drift = 0.0
        lam_tail = self._resolve_scalar(self.config.get("anti_explosion_vanishing_v4_lambda_tail", 0.25))
        if (not math.isfinite(lam_tail)) or lam_tail < 0.0:
            lam_tail = 0.0
        gain_lo = self._resolve_scalar(self.config.get("anti_explosion_vanishing_v4_gain_lo", 0.97))
        gain_hi = self._resolve_scalar(self.config.get("anti_explosion_vanishing_v4_gain_hi", 1.03))
        if (not math.isfinite(gain_lo)) or gain_lo <= 0.0:
            gain_lo = 0.97
        if (not math.isfinite(gain_hi)) or gain_hi <= 0.0:
            gain_hi = 1.03
        gain_lo = max(1e-6, float(gain_lo))
        gain_hi = max(gain_lo + 1e-6, float(gain_hi))
        tail_tau = self._resolve_scalar(self.config.get("anti_explosion_vanishing_v4_tail_tau", 0.010))
        if (not math.isfinite(tail_tau)) or tail_tau <= 0.0:
            tail_tau = 0.010
        eps = self._resolve_scalar(self.config.get("anti_explosion_vanishing_v4_eps", 1e-6))
        if (not math.isfinite(eps)) or eps <= 0.0:
            eps = 1e-6
        detach_reference = self._coerce_bool(
            self.config.get("anti_explosion_vanishing_v4_detach_reference", True)
        )
        highway_ratio = self._resolve_scalar(self.config.get("anti_explosion_vanishing_v4_highway_ratio", 0.25))
        if (not math.isfinite(highway_ratio)) or highway_ratio <= 0.0:
            highway_ratio = 0.25
        highway_ratio = float(min(1.0, max(0.01, highway_ratio)))
        update_scale = self._resolve_scalar(self.config.get("anti_explosion_vanishing_v4_update_scale", 0.08))
        if (not math.isfinite(update_scale)) or update_scale <= 0.0:
            update_scale = 0.08
        update_scale = float(min(1.0, max(1e-6, update_scale)))
        update_clip = self._resolve_scalar(self.config.get("anti_explosion_vanishing_v4_update_clip", 0.0))
        if not math.isfinite(update_clip):
            update_clip = 0.0
        update_clip = float(max(0.0, update_clip))
        return {
            "enabled": bool(enabled),
            "lambda_drift": float(lam_drift),
            "lambda_tail": float(lam_tail),
            "gain_lo": float(gain_lo),
            "gain_hi": float(gain_hi),
            "log_gain_lo": float(math.log(gain_lo)),
            "log_gain_hi": float(math.log(gain_hi)),
            "tail_tau": float(tail_tau),
            "eps": float(eps),
            "detach_reference": bool(detach_reference),
            "highway_ratio": float(highway_ratio),
            "update_scale": float(update_scale),
            "update_clip": float(update_clip),
        }

    def _resolve_aev5_config(self):
        enabled = self._coerce_bool(self.config.get("anti_explosion_vanishing_v5_enabled", False))
        enabled = bool(enabled)
        target_std = self._resolve_scalar(self.config.get("anti_explosion_vanishing_v5_target_std", 0.25))
        if (not math.isfinite(target_std)) or target_std <= 0.0:
            target_std = 0.25
        scale_lo = self._resolve_scalar(self.config.get("anti_explosion_vanishing_v5_scale_lo", 0.5))
        scale_hi = self._resolve_scalar(self.config.get("anti_explosion_vanishing_v5_scale_hi", 4.0))
        if (not math.isfinite(scale_lo)) or scale_lo <= 0.0:
            scale_lo = 0.5
        if (not math.isfinite(scale_hi)) or scale_hi <= 0.0:
            scale_hi = 4.0
        scale_lo = max(1e-6, float(scale_lo))
        scale_hi = max(scale_lo, float(scale_hi))
        eps = self._resolve_scalar(self.config.get("anti_explosion_vanishing_v5_eps", 1e-6))
        if (not math.isfinite(eps)) or eps <= 0.0:
            eps = 1e-6
        detach_reference = self._coerce_bool(
            self.config.get("anti_explosion_vanishing_v5_detach_reference", True)
        )
        return {
            "enabled": bool(enabled),
            "target_std": float(target_std),
            "scale_lo": float(scale_lo),
            "scale_hi": float(scale_hi),
            "eps": float(eps),
            "detach_reference": bool(detach_reference),
        }

    @staticmethod
    def _aev4_highway_dim(last_dim, aev4_cfg):
        d = int(last_dim)
        if d <= 0:
            return 0
        ratio = float(aev4_cfg.get("highway_ratio", 0.25))
        if not math.isfinite(ratio):
            ratio = 0.25
        ratio = min(1.0, max(0.01, ratio))
        return int(min(d, max(1, round(float(d) * ratio))))

    @staticmethod
    def _apply_aev4_state_update(state_prev, state_next_post, aev4_cfg):
        if not bool(aev4_cfg.get("enabled", False)):
            return state_next_post, None
        if state_next_post.shape[-1] <= 0:
            return state_next_post, None
        highway_dim = EnvironmentPrior._aev4_highway_dim(state_next_post.shape[-1], aev4_cfg)
        if highway_dim <= 0:
            return state_next_post, None

        update_scale = float(aev4_cfg.get("update_scale", 0.12))
        if (not math.isfinite(update_scale)) or update_scale <= 0.0:
            return state_next_post, None
        update_scale = min(1.0, max(1e-6, update_scale))
        update_clip = float(aev4_cfg.get("update_clip", 0.0))
        if (not math.isfinite(update_clip)) or update_clip < 0.0:
            update_clip = 0.0

        residual = state_next_post[..., :highway_dim] - state_prev[..., :highway_dim]
        clip_hit_share = torch.zeros((), device=state_next_post.device, dtype=torch.float32)
        if update_clip > 0.0:
            clip_abs = torch.as_tensor(update_clip, device=residual.device, dtype=residual.dtype)
            clip_hit_share = (residual.detach().abs() > clip_abs).to(torch.float32).mean()
            residual = torch.clamp(residual, min=-clip_abs, max=clip_abs)
        controlled = state_prev[..., :highway_dim] + (residual * update_scale)
        out = state_next_post.clone()
        out[..., :highway_dim] = controlled
        return out, {
            "highway_dim": int(highway_dim),
            "clip_hit_share": clip_hit_share,
        }

    @staticmethod
    def _aev4_new_accumulator(enabled, device, dtype, aev4_cfg=None):
        if aev4_cfg is None:
            aev4_cfg = {}
        return {
            "enabled": bool(enabled),
            "lambda_drift": float(aev4_cfg.get("lambda_drift", 0.0)),
            "lambda_tail": float(aev4_cfg.get("lambda_tail", 0.0)),
            "gain_lo": float(aev4_cfg.get("gain_lo", 0.0)),
            "gain_hi": float(aev4_cfg.get("gain_hi", 0.0)),
            "log_gain_lo": float(aev4_cfg.get("log_gain_lo", 0.0)),
            "log_gain_hi": float(aev4_cfg.get("log_gain_hi", 0.0)),
            "highway_ratio": float(aev4_cfg.get("highway_ratio", 0.25)),
            "update_scale": float(aev4_cfg.get("update_scale", 0.12)),
            "update_clip": float(aev4_cfg.get("update_clip", 0.0)),
            "drift_sum": torch.zeros((), device=device, dtype=dtype),
            "tail_sum": torch.zeros((), device=device, dtype=dtype),
            "pairs": 0,
            "log_gain_sum": torch.zeros((), device=device, dtype=torch.float64),
            "log_gain_sumsq": torch.zeros((), device=device, dtype=torch.float64),
            "log_gain_count": 0,
            "tail_low_count": 0,
            "tail_high_count": 0,
            "gain_sum": torch.zeros((), device=device, dtype=torch.float64),
            "gain_sumsq": torch.zeros((), device=device, dtype=torch.float64),
            "gain_count": 0,
            "gain_min": None,
            "gain_max": None,
            "update_rms_sum": torch.zeros((), device=device, dtype=torch.float64),
            "update_rms_sumsq": torch.zeros((), device=device, dtype=torch.float64),
            "update_rms_count": 0,
            "clip_hit_sum": torch.zeros((), device=device, dtype=torch.float64),
            "clip_hit_count": 0,
        }

    @staticmethod
    def _aev4_update_accumulator(acc, prev_delta, curr_delta, aev4_cfg, step_aux=None):
        if (not bool(acc.get("enabled", False))) or (prev_delta is None):
            return
        if curr_delta is None or curr_delta.shape[-1] <= 0:
            return
        highway_dim = EnvironmentPrior._aev4_highway_dim(curr_delta.shape[-1], aev4_cfg)
        if highway_dim <= 0:
            return

        prev = prev_delta[..., :highway_dim]
        curr = curr_delta[..., :highway_dim]
        if prev.ndim == 1:
            prev = prev.unsqueeze(0)
            curr = curr.unsqueeze(0)

        eps = float(aev4_cfg.get("eps", 1e-6))
        prev_rms = torch.sqrt(torch.mean(prev * prev, dim=-1) + eps)
        if bool(aev4_cfg.get("detach_reference", True)):
            prev_rms = prev_rms.detach()
        curr_rms = torch.sqrt(torch.mean(curr * curr, dim=-1) + eps)
        gain = curr_rms / (prev_rms + eps)
        log_gain = torch.log(gain + eps)

        tau = float(aev4_cfg.get("tail_tau", 0.016))
        if tau <= 0.0:
            tau = 1e-6
        tau_t = torch.as_tensor(tau, device=log_gain.device, dtype=log_gain.dtype)
        low_margin = (float(aev4_cfg["log_gain_lo"]) - log_gain) / tau_t
        high_margin = (log_gain - float(aev4_cfg["log_gain_hi"])) / tau_t
        tail_penalty_vec = (F.softplus(low_margin) + F.softplus(high_margin)) * tau_t

        acc["drift_sum"] = acc["drift_sum"] + log_gain.mean()
        acc["tail_sum"] = acc["tail_sum"] + tail_penalty_vec.mean()
        acc["pairs"] = int(acc["pairs"]) + 1

        log_gain_det64 = log_gain.detach().to(dtype=torch.float64)
        acc["log_gain_sum"] = acc["log_gain_sum"] + log_gain_det64.sum()
        acc["log_gain_sumsq"] = acc["log_gain_sumsq"] + (log_gain_det64 * log_gain_det64).sum()
        acc["log_gain_count"] = int(acc["log_gain_count"]) + int(log_gain_det64.numel())
        acc["tail_low_count"] = int(acc["tail_low_count"]) + int(
            (log_gain_det64 < float(aev4_cfg["log_gain_lo"])).sum().item()
        )
        acc["tail_high_count"] = int(acc["tail_high_count"]) + int(
            (log_gain_det64 > float(aev4_cfg["log_gain_hi"])).sum().item()
        )

        gain_det = gain.detach()
        gain_det64 = gain_det.to(dtype=torch.float64)
        acc["gain_sum"] = acc["gain_sum"] + gain_det64.sum()
        acc["gain_sumsq"] = acc["gain_sumsq"] + (gain_det64 * gain_det64).sum()
        acc["gain_count"] = int(acc["gain_count"]) + int(gain_det64.numel())
        gain_min = gain_det.min().detach()
        gain_max = gain_det.max().detach()
        acc["gain_min"] = gain_min if acc["gain_min"] is None else torch.minimum(acc["gain_min"], gain_min)
        acc["gain_max"] = gain_max if acc["gain_max"] is None else torch.maximum(acc["gain_max"], gain_max)

        curr_rms_det = curr_rms.detach().to(dtype=torch.float64)
        acc["update_rms_sum"] = acc["update_rms_sum"] + curr_rms_det.sum()
        acc["update_rms_sumsq"] = acc["update_rms_sumsq"] + (curr_rms_det * curr_rms_det).sum()
        acc["update_rms_count"] = int(acc["update_rms_count"]) + int(curr_rms_det.numel())
        if isinstance(step_aux, dict):
            clip_hit_share = step_aux.get("clip_hit_share", None)
            if torch.is_tensor(clip_hit_share):
                acc["clip_hit_sum"] = acc["clip_hit_sum"] + clip_hit_share.detach().to(
                    device=curr.device, dtype=torch.float64
                )
                acc["clip_hit_count"] = int(acc["clip_hit_count"]) + 1

    @staticmethod
    def _aev4_finalize_accumulator(acc, device, dtype, detach_penalty=False):
        enabled = bool(acc.get("enabled", False))
        pairs = int(acc.get("pairs", 0))
        drift_sum = acc.get("drift_sum", None)
        tail_sum = acc.get("tail_sum", None)
        if pairs > 0 and torch.is_tensor(drift_sum):
            drift_mean = drift_sum / float(pairs)
        else:
            drift_mean = torch.zeros((), device=device, dtype=dtype)
        if pairs > 0 and torch.is_tensor(tail_sum):
            tail_mean = tail_sum / float(pairs)
        else:
            tail_mean = torch.zeros((), device=device, dtype=dtype)

        penalty_drift = drift_mean * drift_mean
        penalty_tail = tail_mean
        lambda_drift = float(acc.get("lambda_drift", 0.0))
        lambda_tail = float(acc.get("lambda_tail", 0.0))
        penalty_mean = (penalty_drift * lambda_drift) + (penalty_tail * lambda_tail)
        if detach_penalty:
            penalty_drift = penalty_drift.detach()
            penalty_tail = penalty_tail.detach()
            penalty_mean = penalty_mean.detach()

        log_gain_count = int(acc.get("log_gain_count", 0))
        log_gain_sum = acc.get("log_gain_sum", torch.zeros((), device=device, dtype=torch.float64)).detach()
        log_gain_sumsq = acc.get("log_gain_sumsq", torch.zeros((), device=device, dtype=torch.float64)).detach()
        if log_gain_count > 0:
            log_gain_mean64 = log_gain_sum / float(log_gain_count)
            log_gain_var64 = (log_gain_sumsq / float(log_gain_count)) - (log_gain_mean64 * log_gain_mean64)
            log_gain_std64 = torch.sqrt(torch.clamp(log_gain_var64, min=0.0))
            log_gain_mean = log_gain_mean64.to(dtype=dtype).detach()
            log_gain_std = log_gain_std64.to(dtype=dtype).detach()
            tail_low_share = torch.as_tensor(
                float(acc.get("tail_low_count", 0)) / float(log_gain_count),
                device=device,
                dtype=dtype,
            )
            tail_high_share = torch.as_tensor(
                float(acc.get("tail_high_count", 0)) / float(log_gain_count),
                device=device,
                dtype=dtype,
            )
        else:
            log_gain_mean = torch.zeros((), device=device, dtype=dtype)
            log_gain_std = torch.zeros((), device=device, dtype=dtype)
            tail_low_share = torch.zeros((), device=device, dtype=dtype)
            tail_high_share = torch.zeros((), device=device, dtype=dtype)

        gain_count = int(acc.get("gain_count", 0))
        gain_sum = acc.get("gain_sum", torch.zeros((), device=device, dtype=torch.float64)).detach()
        gain_sumsq = acc.get("gain_sumsq", torch.zeros((), device=device, dtype=torch.float64)).detach()
        if gain_count > 0:
            gain_mean64 = gain_sum / float(gain_count)
            gain_var64 = (gain_sumsq / float(gain_count)) - (gain_mean64 * gain_mean64)
            gain_std64 = torch.sqrt(torch.clamp(gain_var64, min=0.0))
            gain_mean = gain_mean64.to(dtype=dtype).detach()
            gain_std = gain_std64.to(dtype=dtype).detach()
        else:
            gain_mean = torch.zeros((), device=device, dtype=dtype)
            gain_std = torch.zeros((), device=device, dtype=dtype)

        gain_min = acc.get("gain_min", None)
        if gain_min is None:
            gain_min = torch.zeros((), device=device, dtype=dtype)
        else:
            gain_min = gain_min.to(device=device, dtype=dtype).detach()
        gain_max = acc.get("gain_max", None)
        if gain_max is None:
            gain_max = torch.zeros((), device=device, dtype=dtype)
        else:
            gain_max = gain_max.to(device=device, dtype=dtype).detach()

        update_rms_count = int(acc.get("update_rms_count", 0))
        update_rms_sum = acc.get("update_rms_sum", torch.zeros((), device=device, dtype=torch.float64)).detach()
        update_rms_sumsq = acc.get("update_rms_sumsq", torch.zeros((), device=device, dtype=torch.float64)).detach()
        if update_rms_count > 0:
            update_rms_mean64 = update_rms_sum / float(update_rms_count)
            update_rms_var64 = (
                (update_rms_sumsq / float(update_rms_count)) - (update_rms_mean64 * update_rms_mean64)
            )
            update_rms_std64 = torch.sqrt(torch.clamp(update_rms_var64, min=0.0))
            update_rms_mean = update_rms_mean64.to(dtype=dtype).detach()
            update_rms_std = update_rms_std64.to(dtype=dtype).detach()
        else:
            update_rms_mean = torch.zeros((), device=device, dtype=dtype)
            update_rms_std = torch.zeros((), device=device, dtype=dtype)

        clip_hit_count = int(acc.get("clip_hit_count", 0))
        clip_hit_sum = acc.get("clip_hit_sum", torch.zeros((), device=device, dtype=torch.float64)).detach()
        if clip_hit_count > 0:
            clip_hit_share = (clip_hit_sum / float(clip_hit_count)).to(dtype=dtype).detach()
        else:
            clip_hit_share = torch.zeros((), device=device, dtype=dtype)

        return {
            "enabled": int(enabled),
            "pairs": int(pairs),
            "lambda_drift": float(lambda_drift),
            "lambda_tail": float(lambda_tail),
            "gain_lo": float(acc.get("gain_lo", 0.0)),
            "gain_hi": float(acc.get("gain_hi", 0.0)),
            "highway_ratio": float(acc.get("highway_ratio", 0.25)),
            "update_scale": float(acc.get("update_scale", 0.12)),
            "update_clip": float(acc.get("update_clip", 0.0)),
            "drift_mean": drift_mean,
            "tail_mean": tail_mean,
            "penalty_drift": penalty_drift,
            "penalty_tail": penalty_tail,
            "penalty_mean": penalty_mean,
            "log_gain_sum": log_gain_sum,
            "log_gain_sumsq": log_gain_sumsq,
            "log_gain_count": int(log_gain_count),
            "log_gain_mean": log_gain_mean,
            "log_gain_std": log_gain_std,
            "tail_low_count": int(acc.get("tail_low_count", 0)),
            "tail_high_count": int(acc.get("tail_high_count", 0)),
            "tail_low_share": tail_low_share,
            "tail_high_share": tail_high_share,
            "gain_sum": gain_sum,
            "gain_sumsq": gain_sumsq,
            "gain_count": int(gain_count),
            "gain_mean": gain_mean,
            "gain_std": gain_std,
            "gain_min": gain_min,
            "gain_max": gain_max,
            "update_rms_sum": update_rms_sum,
            "update_rms_sumsq": update_rms_sumsq,
            "update_rms_count": int(update_rms_count),
            "update_rms_mean": update_rms_mean,
            "update_rms_std": update_rms_std,
            "clip_hit_sum": clip_hit_sum,
            "clip_hit_count": int(clip_hit_count),
            "clip_hit_share": clip_hit_share,
        }

    @staticmethod
    def _aev4_accumulate_window_summary(det_acc, window_summary, device):
        if (not isinstance(det_acc, dict)) or (not isinstance(window_summary, dict)):
            return
        pairs_w = int(window_summary.get("pairs", 0))
        det_acc["pairs"] = int(det_acc.get("pairs", 0)) + pairs_w
        drift_mean_w = window_summary.get("drift_mean", None)
        if torch.is_tensor(drift_mean_w):
            det_acc["drift_sum"] = det_acc["drift_sum"] + (drift_mean_w.detach() * float(max(1, pairs_w)))
        tail_mean_w = window_summary.get("tail_mean", None)
        if torch.is_tensor(tail_mean_w):
            det_acc["tail_sum"] = det_acc["tail_sum"] + (tail_mean_w.detach() * float(max(1, pairs_w)))

        log_gain_sum_w = window_summary.get("log_gain_sum", None)
        if torch.is_tensor(log_gain_sum_w):
            det_acc["log_gain_sum"] = det_acc["log_gain_sum"] + log_gain_sum_w.detach().to(
                device=device, dtype=torch.float64
            )
        log_gain_sumsq_w = window_summary.get("log_gain_sumsq", None)
        if torch.is_tensor(log_gain_sumsq_w):
            det_acc["log_gain_sumsq"] = det_acc["log_gain_sumsq"] + log_gain_sumsq_w.detach().to(
                device=device, dtype=torch.float64
            )
        det_acc["log_gain_count"] = int(det_acc.get("log_gain_count", 0)) + int(window_summary.get("log_gain_count", 0))
        det_acc["tail_low_count"] = int(det_acc.get("tail_low_count", 0)) + int(window_summary.get("tail_low_count", 0))
        det_acc["tail_high_count"] = int(det_acc.get("tail_high_count", 0)) + int(window_summary.get("tail_high_count", 0))

        gain_sum_w = window_summary.get("gain_sum", None)
        if torch.is_tensor(gain_sum_w):
            det_acc["gain_sum"] = det_acc["gain_sum"] + gain_sum_w.detach().to(device=device, dtype=torch.float64)
        gain_sumsq_w = window_summary.get("gain_sumsq", None)
        if torch.is_tensor(gain_sumsq_w):
            det_acc["gain_sumsq"] = det_acc["gain_sumsq"] + gain_sumsq_w.detach().to(
                device=device, dtype=torch.float64
            )
        det_acc["gain_count"] = int(det_acc.get("gain_count", 0)) + int(window_summary.get("gain_count", 0))
        gain_min_w = window_summary.get("gain_min", None)
        if torch.is_tensor(gain_min_w):
            det_acc["gain_min"] = (
                gain_min_w.detach()
                if det_acc.get("gain_min", None) is None
                else torch.minimum(det_acc["gain_min"], gain_min_w.detach())
            )
        gain_max_w = window_summary.get("gain_max", None)
        if torch.is_tensor(gain_max_w):
            det_acc["gain_max"] = (
                gain_max_w.detach()
                if det_acc.get("gain_max", None) is None
                else torch.maximum(det_acc["gain_max"], gain_max_w.detach())
            )

        update_rms_sum_w = window_summary.get("update_rms_sum", None)
        if torch.is_tensor(update_rms_sum_w):
            det_acc["update_rms_sum"] = det_acc["update_rms_sum"] + update_rms_sum_w.detach().to(
                device=device, dtype=torch.float64
            )
        update_rms_sumsq_w = window_summary.get("update_rms_sumsq", None)
        if torch.is_tensor(update_rms_sumsq_w):
            det_acc["update_rms_sumsq"] = det_acc["update_rms_sumsq"] + update_rms_sumsq_w.detach().to(
                device=device, dtype=torch.float64
            )
        det_acc["update_rms_count"] = int(det_acc.get("update_rms_count", 0)) + int(window_summary.get("update_rms_count", 0))
        clip_hit_sum_w = window_summary.get("clip_hit_sum", None)
        if torch.is_tensor(clip_hit_sum_w):
            det_acc["clip_hit_sum"] = det_acc["clip_hit_sum"] + clip_hit_sum_w.detach().to(
                device=device, dtype=torch.float64
            )
        det_acc["clip_hit_count"] = int(det_acc.get("clip_hit_count", 0)) + int(window_summary.get("clip_hit_count", 0))

    @staticmethod
    def _aev4_merge_rollout_summary(acc, summary, batch_weight, device, dtype):
        if not isinstance(summary, dict):
            return acc
        if int(summary.get("enabled", 0)) == 0:
            return acc

        w = float(max(0.0, batch_weight))
        if acc is None:
            acc = {
                "enabled": 1,
                "lambda_drift": float(summary.get("lambda_drift", 0.0)),
                "lambda_tail": float(summary.get("lambda_tail", 0.0)),
                "gain_lo": float(summary.get("gain_lo", 0.0)),
                "gain_hi": float(summary.get("gain_hi", 0.0)),
                "highway_ratio": float(summary.get("highway_ratio", 0.25)),
                "update_scale": float(summary.get("update_scale", 0.12)),
                "update_clip": float(summary.get("update_clip", 0.0)),
                "penalty_mean_weighted": None,
                "penalty_drift_weighted": None,
                "penalty_tail_weighted": None,
                "log_gain_sum": torch.zeros((), device=device, dtype=torch.float64),
                "log_gain_sumsq": torch.zeros((), device=device, dtype=torch.float64),
                "log_gain_count": 0,
                "tail_low_count": 0,
                "tail_high_count": 0,
                "gain_sum": torch.zeros((), device=device, dtype=torch.float64),
                "gain_sumsq": torch.zeros((), device=device, dtype=torch.float64),
                "gain_count": 0,
                "gain_min": None,
                "gain_max": None,
                "update_rms_sum": torch.zeros((), device=device, dtype=torch.float64),
                "update_rms_sumsq": torch.zeros((), device=device, dtype=torch.float64),
                "update_rms_count": 0,
                "clip_hit_sum": torch.zeros((), device=device, dtype=torch.float64),
                "clip_hit_count": 0,
            }

        penalty_mean = summary.get("penalty_mean", None)
        if torch.is_tensor(penalty_mean):
            term = penalty_mean * w
            if acc["penalty_mean_weighted"] is None:
                acc["penalty_mean_weighted"] = term
            else:
                acc["penalty_mean_weighted"] = acc["penalty_mean_weighted"] + term
        penalty_drift = summary.get("penalty_drift", None)
        if torch.is_tensor(penalty_drift):
            term = penalty_drift * w
            if acc["penalty_drift_weighted"] is None:
                acc["penalty_drift_weighted"] = term
            else:
                acc["penalty_drift_weighted"] = acc["penalty_drift_weighted"] + term
        penalty_tail = summary.get("penalty_tail", None)
        if torch.is_tensor(penalty_tail):
            term = penalty_tail * w
            if acc["penalty_tail_weighted"] is None:
                acc["penalty_tail_weighted"] = term
            else:
                acc["penalty_tail_weighted"] = acc["penalty_tail_weighted"] + term

        log_gain_sum = summary.get("log_gain_sum", None)
        if torch.is_tensor(log_gain_sum):
            acc["log_gain_sum"] = acc["log_gain_sum"] + log_gain_sum.to(device=device, dtype=torch.float64)
        log_gain_sumsq = summary.get("log_gain_sumsq", None)
        if torch.is_tensor(log_gain_sumsq):
            acc["log_gain_sumsq"] = acc["log_gain_sumsq"] + log_gain_sumsq.to(device=device, dtype=torch.float64)
        acc["log_gain_count"] = int(acc["log_gain_count"]) + int(summary.get("log_gain_count", 0))
        acc["tail_low_count"] = int(acc["tail_low_count"]) + int(summary.get("tail_low_count", 0))
        acc["tail_high_count"] = int(acc["tail_high_count"]) + int(summary.get("tail_high_count", 0))

        gain_sum = summary.get("gain_sum", None)
        if torch.is_tensor(gain_sum):
            acc["gain_sum"] = acc["gain_sum"] + gain_sum.to(device=device, dtype=torch.float64)
        gain_sumsq = summary.get("gain_sumsq", None)
        if torch.is_tensor(gain_sumsq):
            acc["gain_sumsq"] = acc["gain_sumsq"] + gain_sumsq.to(device=device, dtype=torch.float64)
        acc["gain_count"] = int(acc["gain_count"]) + int(summary.get("gain_count", 0))
        gain_min = summary.get("gain_min", None)
        if torch.is_tensor(gain_min):
            gain_min = gain_min.to(device=device, dtype=dtype)
            acc["gain_min"] = gain_min if acc["gain_min"] is None else torch.minimum(acc["gain_min"], gain_min)
        gain_max = summary.get("gain_max", None)
        if torch.is_tensor(gain_max):
            gain_max = gain_max.to(device=device, dtype=dtype)
            acc["gain_max"] = gain_max if acc["gain_max"] is None else torch.maximum(acc["gain_max"], gain_max)

        update_rms_sum = summary.get("update_rms_sum", None)
        if torch.is_tensor(update_rms_sum):
            acc["update_rms_sum"] = acc["update_rms_sum"] + update_rms_sum.to(device=device, dtype=torch.float64)
        update_rms_sumsq = summary.get("update_rms_sumsq", None)
        if torch.is_tensor(update_rms_sumsq):
            acc["update_rms_sumsq"] = acc["update_rms_sumsq"] + update_rms_sumsq.to(
                device=device, dtype=torch.float64
            )
        acc["update_rms_count"] = int(acc["update_rms_count"]) + int(summary.get("update_rms_count", 0))
        clip_hit_sum = summary.get("clip_hit_sum", None)
        if torch.is_tensor(clip_hit_sum):
            acc["clip_hit_sum"] = acc["clip_hit_sum"] + clip_hit_sum.to(device=device, dtype=torch.float64)
        acc["clip_hit_count"] = int(acc["clip_hit_count"]) + int(summary.get("clip_hit_count", 0))
        return acc

    @staticmethod
    def _aev4_finalize_rollout_summary(acc, device, dtype):
        if acc is None:
            return None
        penalty_mean = acc.get("penalty_mean_weighted", None)
        if penalty_mean is None:
            penalty_mean = torch.zeros((), device=device, dtype=dtype)
        penalty_drift = acc.get("penalty_drift_weighted", None)
        if penalty_drift is None:
            penalty_drift = torch.zeros((), device=device, dtype=dtype)
        penalty_tail = acc.get("penalty_tail_weighted", None)
        if penalty_tail is None:
            penalty_tail = torch.zeros((), device=device, dtype=dtype)

        log_gain_count = int(acc.get("log_gain_count", 0))
        log_gain_sum = acc.get("log_gain_sum", torch.zeros((), device=device, dtype=torch.float64)).detach()
        log_gain_sumsq = acc.get("log_gain_sumsq", torch.zeros((), device=device, dtype=torch.float64)).detach()
        if log_gain_count > 0:
            log_gain_mean64 = log_gain_sum / float(log_gain_count)
            log_gain_var64 = (log_gain_sumsq / float(log_gain_count)) - (log_gain_mean64 * log_gain_mean64)
            log_gain_std64 = torch.sqrt(torch.clamp(log_gain_var64, min=0.0))
            log_gain_mean = log_gain_mean64.to(dtype=dtype).detach()
            log_gain_std = log_gain_std64.to(dtype=dtype).detach()
            tail_low_share = torch.as_tensor(
                float(acc.get("tail_low_count", 0)) / float(log_gain_count),
                device=device,
                dtype=dtype,
            )
            tail_high_share = torch.as_tensor(
                float(acc.get("tail_high_count", 0)) / float(log_gain_count),
                device=device,
                dtype=dtype,
            )
        else:
            log_gain_mean = torch.zeros((), device=device, dtype=dtype)
            log_gain_std = torch.zeros((), device=device, dtype=dtype)
            tail_low_share = torch.zeros((), device=device, dtype=dtype)
            tail_high_share = torch.zeros((), device=device, dtype=dtype)

        gain_count = int(acc.get("gain_count", 0))
        gain_sum = acc.get("gain_sum", torch.zeros((), device=device, dtype=torch.float64)).detach()
        gain_sumsq = acc.get("gain_sumsq", torch.zeros((), device=device, dtype=torch.float64)).detach()
        if gain_count > 0:
            gain_mean64 = gain_sum / float(gain_count)
            gain_var64 = (gain_sumsq / float(gain_count)) - (gain_mean64 * gain_mean64)
            gain_std64 = torch.sqrt(torch.clamp(gain_var64, min=0.0))
            gain_mean = gain_mean64.to(dtype=dtype).detach()
            gain_std = gain_std64.to(dtype=dtype).detach()
        else:
            gain_mean = torch.zeros((), device=device, dtype=dtype)
            gain_std = torch.zeros((), device=device, dtype=dtype)

        gain_min = acc.get("gain_min", None)
        if gain_min is None:
            gain_min = torch.zeros((), device=device, dtype=dtype)
        else:
            gain_min = gain_min.detach()
        gain_max = acc.get("gain_max", None)
        if gain_max is None:
            gain_max = torch.zeros((), device=device, dtype=dtype)
        else:
            gain_max = gain_max.detach()

        update_rms_count = int(acc.get("update_rms_count", 0))
        update_rms_sum = acc.get("update_rms_sum", torch.zeros((), device=device, dtype=torch.float64)).detach()
        update_rms_sumsq = acc.get("update_rms_sumsq", torch.zeros((), device=device, dtype=torch.float64)).detach()
        if update_rms_count > 0:
            update_rms_mean64 = update_rms_sum / float(update_rms_count)
            update_rms_var64 = (
                (update_rms_sumsq / float(update_rms_count)) - (update_rms_mean64 * update_rms_mean64)
            )
            update_rms_std64 = torch.sqrt(torch.clamp(update_rms_var64, min=0.0))
            update_rms_mean = update_rms_mean64.to(dtype=dtype).detach()
            update_rms_std = update_rms_std64.to(dtype=dtype).detach()
        else:
            update_rms_mean = torch.zeros((), device=device, dtype=dtype)
            update_rms_std = torch.zeros((), device=device, dtype=dtype)

        clip_hit_count = int(acc.get("clip_hit_count", 0))
        clip_hit_sum = acc.get("clip_hit_sum", torch.zeros((), device=device, dtype=torch.float64)).detach()
        if clip_hit_count > 0:
            clip_hit_share = (clip_hit_sum / float(clip_hit_count)).to(dtype=dtype).detach()
        else:
            clip_hit_share = torch.zeros((), device=device, dtype=dtype)

        return {
            "enabled": 1,
            "lambda_drift": float(acc.get("lambda_drift", 0.0)),
            "lambda_tail": float(acc.get("lambda_tail", 0.0)),
            "gain_lo": float(acc.get("gain_lo", 0.0)),
            "gain_hi": float(acc.get("gain_hi", 0.0)),
            "highway_ratio": float(acc.get("highway_ratio", 0.25)),
            "update_scale": float(acc.get("update_scale", 0.12)),
            "update_clip": float(acc.get("update_clip", 0.0)),
            "penalty_mean": penalty_mean,
            "penalty_drift": penalty_drift,
            "penalty_tail": penalty_tail,
            "log_gain_sum": log_gain_sum,
            "log_gain_sumsq": log_gain_sumsq,
            "log_gain_count": int(log_gain_count),
            "log_gain_mean": log_gain_mean,
            "log_gain_std": log_gain_std,
            "tail_low_count": int(acc.get("tail_low_count", 0)),
            "tail_high_count": int(acc.get("tail_high_count", 0)),
            "tail_low_share": tail_low_share,
            "tail_high_share": tail_high_share,
            "gain_sum": gain_sum,
            "gain_sumsq": gain_sumsq,
            "gain_count": int(gain_count),
            "gain_mean": gain_mean,
            "gain_std": gain_std,
            "gain_min": gain_min,
            "gain_max": gain_max,
            "update_rms_sum": update_rms_sum,
            "update_rms_sumsq": update_rms_sumsq,
            "update_rms_count": int(update_rms_count),
            "update_rms_mean": update_rms_mean,
            "update_rms_std": update_rms_std,
            "clip_hit_sum": clip_hit_sum,
            "clip_hit_count": int(clip_hit_count),
            "clip_hit_share": clip_hit_share,
        }

    def _sample_environment(self, h, device, rng_seed=None):
        family = str(h["family"]).lower()
        if family not in {"scm", "gp"}:
            family = "scm"

        state_dim, obs_dim, action_dim, noise_dim, zero_pad_dim = self._sample_dims(h)
        in_dim = int(state_dim + obs_dim + action_dim + noise_dim + zero_pad_dim)
        builder = self._build_scm_fn if family == "scm" else self._build_gp_fn
        local_generator = None
        if rng_seed is not None:
            local_generator = torch.Generator(device=device)
            local_generator.manual_seed(int(rng_seed))

        # X-style / Y-style generators.
        x_generator = builder(in_dim, state_dim, h, device, generator=local_generator)  # for s_{t+1}
        y_generator = builder(in_dim, 1, h, device, generator=local_generator)          # for r_{t+1}
        policy_generator = builder(in_dim, action_dim, h, device, generator=local_generator)
        aev4_cfg = self._resolve_aev4_config()

        env = {
            "family": family,
            "state_dim": state_dim,
            "obs_dim": obs_dim,
            "action_dim": action_dim,
            "noise_dim": noise_dim,
            "zero_pad_dim": zero_pad_dim,
            "obs_slot_dim": int(max(1, h.get("obs_slot_dim", 400))),
            "action_slot_dim": int(max(1, h.get("action_slot_dim", 30))),
            "x_generator": x_generator,
            "y_generator": y_generator,
            "policy_generator": policy_generator,
            "alpha": float(max(1e-4, min(1.0, float(h["alpha"])))),
            "init_state_std": float(h["init_state_std"]),
            "init_action_std": float(h["init_action_std"]),
            "state_noise_std": float(h["state_noise_std"]),
            "action_noise_train_std": float(h["action_noise_train_std"]),
            "action_noise_eval_std": float(h["action_noise_eval_std"]),
            "reward_scale": float(h["reward_scale"]),
            "reward_clip": float(max(0.1, self._resolve_scalar(h.get("reward_clip", 10.0)))),
            "state_clip": float(max(1.0, self._resolve_scalar(h.get("state_clip", 8.0)))),
            "state_highway_enabled": bool(self._resolve_state_highway_enabled(h)),
            "state_highway_lambda": float(self._resolve_state_highway_lambda(h)),
            "aev4_enabled": bool(aev4_cfg.get("enabled", False)),
            "aev4_highway_ratio": float(aev4_cfg.get("highway_ratio", 0.25)),
            "aev4_update_scale": float(aev4_cfg.get("update_scale", 0.12)),
            "aev4_update_clip": float(aev4_cfg.get("update_clip", 0.0)),
            "reward_dropout_enabled": bool(h.get("reward_dropout_enabled", True)),
            "reward_dropout_impute_zero": bool(h.get("reward_dropout_impute_zero", True)),
            "reward_dropout_ratio": float(max(0.0, min(1.0, self._sample_reward_dropout_ratio(h)))),
        }
        return env

    def _sample_vectorized_batch_hypers(self, batch_size):
        return self._sample_batch_hypers(batch_size)

    def _sample_batch_hypers(self, batch_size):
        batch_size = int(batch_size)
        if batch_size <= 0:
            return []
        return [sample_distributions(self.config) for _ in range(batch_size)]

    def _environment_structure_signature(self, h):
        family = self._normalize_family(h.get("family", "scm"))
        state_dim, obs_dim, action_dim, noise_dim, zero_pad_dim = self._sample_dims(h)
        obs_slot_dim = int(max(1, h.get("obs_slot_dim", 400)))
        action_slot_dim = int(max(1, h.get("action_slot_dim", 30)))
        signature = (
            family,
            state_dim,
            obs_dim,
            action_dim,
            noise_dim,
            zero_pad_dim,
            obs_slot_dim,
            action_slot_dim,
        )
        if family == "scm":
            signature += (
                int(h["num_layers"]),
                int(h["prior_mlp_hidden_dim"]),
                h["prior_mlp_activations"],
            )
        else:
            signature += (int(h["gp_rff_features"]),)
        return signature

    def _environment_transition_signature(self, h):
        family = self._normalize_family(h.get("family", "scm"))
        state_dim, obs_dim, action_dim, noise_dim, zero_pad_dim = self._sample_dims(h)
        signature = (
            family,
            state_dim,
            obs_dim,
            action_dim,
            noise_dim,
            zero_pad_dim,
        )
        if family == "scm":
            signature += (
                int(max(2, int(h["num_layers"]))),
                int(max(state_dim, int(h["prior_mlp_hidden_dim"]))),
                self._activation_name(h["prior_mlp_activations"]),
            )
        else:
            signature += (int(max(8, int(h["gp_rff_features"]))),)
        return signature

    @staticmethod
    def _bucket_ceil_pow2(value):
        value = max(1, int(value))
        return 1 << (value - 1).bit_length()

    @staticmethod
    def _resolve_transition_balanced_bucket_count(grouping_mode):
        mode = str(grouping_mode).strip().lower()
        if not mode.startswith("balanced"):
            return None
        suffix = mode[len("balanced"):]
        if suffix == "":
            return 2
        try:
            return max(1, int(suffix))
        except Exception:
            return None

    def _environment_transition_bucket_signature(self, h, grouping_mode):
        grouping_mode = str(grouping_mode).strip().lower()
        family = self._normalize_family(h.get("family", "scm"))
        if grouping_mode == "family":
            return (family,)
        if grouping_mode == "structure":
            return self._environment_transition_signature(h)
        if self._resolve_transition_balanced_bucket_count(grouping_mode) is not None:
            return (family,)
        state_dim, obs_dim, action_dim, noise_dim, zero_pad_dim = self._sample_dims(h)
        total_dim = state_dim + obs_dim + action_dim + noise_dim + zero_pad_dim
        if family == "scm":
            signature = (
                family,
                self._bucket_ceil_pow2(total_dim),
                self._bucket_ceil_pow2(state_dim),
                self._bucket_ceil_pow2(max(state_dim, int(h["prior_mlp_hidden_dim"]))),
                self._activation_name(h["prior_mlp_activations"]),
            )
            if grouping_mode == "pow2":
                signature += (int(min(8, max(2, int(h["num_layers"])))),)
            return signature
        return (
            family,
            self._bucket_ceil_pow2(total_dim),
            self._bucket_ceil_pow2(state_dim),
            self._bucket_ceil_pow2(max(8, int(h["gp_rff_features"]))),
        )

    def _estimate_transition_sample_work(self, h):
        family = self._normalize_family(h.get("family", "scm"))
        state_dim, obs_dim, action_dim, noise_dim, zero_pad_dim = self._sample_dims(h)
        in_dim = int(state_dim + obs_dim + action_dim + noise_dim + zero_pad_dim)
        state_dim = int(state_dim)
        if family == "scm":
            hidden_dim = int(max(state_dim, int(h["prior_mlp_hidden_dim"])))
            depth = int(max(2, int(h["num_layers"])))
            hidden_layers = max(0, depth - 2)
            return float(in_dim * hidden_dim + hidden_layers * hidden_dim * hidden_dim + hidden_dim * state_dim) + float(
                in_dim * hidden_dim + hidden_layers * hidden_dim * hidden_dim + hidden_dim
            )
        rff_dim = int(max(8, int(h["gp_rff_features"])))
        return float(2 * in_dim * rff_dim + rff_dim * (state_dim + 1))

    def _transition_balanced_sort_key(self, h):
        family = self._normalize_family(h.get("family", "scm"))
        state_dim, obs_dim, action_dim, noise_dim, zero_pad_dim = self._sample_dims(h)
        in_dim = int(state_dim + obs_dim + action_dim + noise_dim + zero_pad_dim)
        if family == "scm":
            hidden_dim = int(max(state_dim, int(h["prior_mlp_hidden_dim"])))
            depth = int(max(2, int(h["num_layers"])))
            act_name = self._activation_name(h["prior_mlp_activations"])
            act_rank = {"identity": 0, "tanh": 1, "relu": 2}.get(act_name, 1)
            return (
                int(self._bucket_ceil_pow2(hidden_dim)),
                int(self._bucket_ceil_pow2(in_dim)),
                int(self._bucket_ceil_pow2(state_dim)),
                int(depth),
                int(act_rank),
            )
        rff_dim = int(max(8, int(h["gp_rff_features"])))
        return (
            int(self._bucket_ceil_pow2(rff_dim)),
            int(self._bucket_ceil_pow2(in_dim)),
            int(self._bucket_ceil_pow2(state_dim)),
        )

    def _build_balanced_transition_bucket_groups(self, family_group, bucket_count):
        bucket_count = int(max(1, bucket_count))
        if bucket_count <= 1 or len(family_group) <= 1:
            family = self._normalize_family(family_group[0][1].get("family", "scm"))
            return {(family, "__balanced__0"): list(family_group)}
        sorted_group = sorted(
            family_group,
            key=lambda pair: (self._transition_balanced_sort_key(pair[1]), self._estimate_transition_sample_work(pair[1])),
            reverse=True,
        )
        sample_work = [self._estimate_transition_sample_work(h) for _, h in sorted_group]
        total_work = float(sum(sample_work))
        if total_work <= 0.0:
            total_work = float(len(sorted_group))
            sample_work = [1.0] * len(sorted_group)
        target_work = total_work / float(bucket_count)
        family = self._normalize_family(sorted_group[0][1].get("family", "scm"))
        groups = {}
        start = 0
        acc_work = 0.0
        bucket_idx = 0
        for idx, item_work in enumerate(sample_work):
            remaining_items = len(sorted_group) - idx - 1
            remaining_buckets = bucket_count - bucket_idx - 1
            acc_work += float(item_work)
            if remaining_buckets <= 0:
                continue
            if remaining_items < remaining_buckets:
                continue
            if acc_work < target_work:
                continue
            groups[(family, f"__balanced__{bucket_idx}")] = sorted_group[start: idx + 1]
            start = idx + 1
            acc_work = 0.0
            bucket_idx += 1
        if start < len(sorted_group):
            groups[(family, f"__balanced__{bucket_idx}")] = sorted_group[start:]
        return groups

    def _estimate_transition_group_work(self, h_list):
        if not h_list:
            return 0.0, 0.0
        family = self._normalize_family(h_list[0].get("family", "scm"))
        dims = [self._sample_dims(h) for h in h_list]
        in_dims = [int(s + o + a + n + z) for s, o, a, n, z in dims]
        state_dims = [int(s) for s, _, _, _, _ in dims]
        batch_size = int(len(h_list))
        if family == "scm":
            hidden_dims = [
                int(max(state_dims[idx], int(h["prior_mlp_hidden_dim"])))
                for idx, h in enumerate(h_list)
            ]
            depths = [int(max(2, int(h["num_layers"]))) for h in h_list]
            in_cap = max(in_dims)
            hidden_cap = max(hidden_dims)
            depth_cap = max(depths)
            state_cap = max(state_dims)
            actual = 0.0
            for in_i, state_i, hidden_i, depth_i in zip(in_dims, state_dims, hidden_dims, depths):
                hidden_layers = max(0, depth_i - 2)
                actual += float(in_i * hidden_i + hidden_layers * hidden_i * hidden_i + hidden_i * state_i)
                actual += float(in_i * hidden_i + hidden_layers * hidden_i * hidden_i + hidden_i)
            padded = float(
                2 * batch_size * (in_cap * hidden_cap + max(0, depth_cap - 2) * hidden_cap * hidden_cap + hidden_cap * state_cap)
            )
            return actual, padded
        m_dims = [int(max(8, int(h["gp_rff_features"]))) for h in h_list]
        in_cap = max(in_dims)
        m_cap = max(m_dims)
        state_cap = max(state_dims)
        actual = 0.0
        for in_i, state_i, m_i in zip(in_dims, state_dims, m_dims):
            actual += float(2 * in_i * m_i + m_i * (state_i + 1))
        padded = float(2 * batch_size * (in_cap * m_cap + m_cap * state_cap))
        return actual, padded

    @staticmethod
    def _normalize_family(family_value):
        family = str(family_value).lower()
        if family not in {"scm", "gp"}:
            family = "scm"
        return family

    @staticmethod
    def _activation_name(activation_value):
        if isinstance(activation_value, str):
            name = activation_value.strip().lower()
        elif isinstance(activation_value, nn.Module):
            name = activation_value.__class__.__name__.lower()
        elif isinstance(activation_value, type) and issubclass(activation_value, nn.Module):
            name = activation_value.__name__.lower()
        else:
            name = "tanh"
        if "relu" in name:
            return "relu"
        if "identity" in name or "linear" in name or name == "none":
            return "identity"
        return "tanh"

    def _environment_group_signature(self, h, grouping_mode):
        grouping_mode = str(grouping_mode).strip().lower()
        if grouping_mode == "family":
            # Throughput-first coarse grouping:
            # keep slot layout homogeneous so reward/mask/action token positions
            # remain valid, but do not split by sampled family at this stage.
            # `_rollout_family_group_vectorized_with_policy` already performs
            # inner structure grouping (family) for transition
            # generators, so mixing families here preserves semantics while
            # avoiding tiny outer rollout groups.
            return (
                int(max(1, h.get("obs_slot_dim", 400))),
                int(max(1, h.get("action_slot_dim", 30))),
            )
        return self._environment_structure_signature(h)

    def _build_scm_hetero_batch_fn(
        self,
        in_dims,
        out_dims,
        h_list,
        device,
        depth_values,
        activation_names,
        generators=None,
        input_mask=None,
    ):
        batch_size = len(h_list)
        device_obj = device if isinstance(device, torch.device) else torch.device(str(device))
        ragged_affine_enabled = bool(self.envgen_ragged_affine and device_obj.type == "cuda")
        if torch.is_tensor(depth_values):
            depth_per_sample = depth_values.to(device=device, dtype=torch.long)
        elif isinstance(depth_values, (list, tuple)):
            depth_per_sample = torch.tensor(
                [max(2, int(d)) for d in depth_values],
                device=device,
                dtype=torch.long,
            )
        else:
            depth_per_sample = torch.full(
                (batch_size,),
                int(max(2, int(depth_values))),
                device=device,
                dtype=torch.long,
            )
        if int(depth_per_sample.numel()) != batch_size:
            raise ValueError("depth_values must match h_list length")
        depth = int(depth_per_sample.max().item())
        in_dims = torch.as_tensor(in_dims, device=device, dtype=torch.long)
        out_dims = torch.as_tensor(out_dims, device=device, dtype=torch.long)
        if input_mask is not None:
            input_mask = torch.as_tensor(input_mask, device=device, dtype=torch.float32)
            in_dims = input_mask.to(dtype=torch.long).sum(dim=1)
        hidden_dims = torch.tensor(
            [max(int(out_dims[bi].item()), int(h["prior_mlp_hidden_dim"])) for bi, h in enumerate(h_list)],
            device=device,
            dtype=torch.long,
        )
        init_std = torch.tensor([float(h["init_std"]) for h in h_list], device=device, dtype=torch.float32)
        noise_std = torch.tensor([float(h["noise_std"]) for h in h_list], device=device, dtype=torch.float32)
        weight_cap_values = []
        for h in h_list:
            cap = self._resolve_lipschitz_weight_cap(h)
            weight_cap_values.append(float(cap) if cap is not None else float("inf"))
        weight_cap = torch.tensor(weight_cap_values, device=device, dtype=torch.float32)
        if isinstance(activation_names, str):
            activation_values = [self._activation_name(activation_names)] * batch_size
        else:
            activation_values = [self._activation_name(v) for v in activation_names]
        if len(activation_values) != batch_size:
            raise ValueError("activation_names must match h_list length")
        activation_codes = []
        for name in activation_values:
            if name == "relu":
                activation_codes.append(1)
            elif name == "identity":
                activation_codes.append(2)
            else:
                activation_codes.append(0)
        activation_codes = torch.tensor(activation_codes, device=device, dtype=torch.long)
        activation_mixed = not bool(torch.all(activation_codes == activation_codes[0]))
        activation_relu_mask = (activation_codes == 1).unsqueeze(1)
        activation_identity_mask = (activation_codes == 2).unsqueeze(1)
        activation_single = int(activation_codes[0].item())

        layers = []
        for layer_idx in range(depth):
            hidden_active = layer_idx < (depth_per_sample - 1)
            if layer_idx == 0:
                if input_mask is None:
                    in_layer = in_dims
                    in_cap = int(in_layer.max().item())
                    in_mask = torch.zeros((batch_size, in_cap), device=device, dtype=torch.float32)
                else:
                    in_layer = in_dims
                    in_cap = int(input_mask.shape[1])
                    in_mask = input_mask.clone()
            else:
                in_layer = torch.where(
                    layer_idx <= (depth_per_sample - 1),
                    hidden_dims,
                    out_dims,
                )
                in_cap = int(in_layer.max().item())
                in_mask = torch.zeros((batch_size, in_cap), device=device, dtype=torch.float32)
            out_layer = torch.where(hidden_active, hidden_dims, out_dims)

            out_cap = int(out_layer.max().item())
            w = torch.zeros((batch_size, in_cap, out_cap), device=device, dtype=torch.float32)
            b = torch.zeros((batch_size, out_cap), device=device, dtype=torch.float32)
            out_mask = torch.zeros((batch_size, out_cap), device=device, dtype=torch.float32)
            ragged_input_index = None
            if ragged_affine_enabled:
                ragged_input_index = torch.zeros((batch_size, in_cap), device=device, dtype=torch.long)
            for bi in range(batch_size):
                if layer_idx == 0 and input_mask is not None:
                    active_idx = torch.nonzero(input_mask[bi] > 0, as_tuple=False).squeeze(1)
                    in_i = int(active_idx.numel())
                else:
                    in_i = int(in_layer[bi].item())
                    active_idx = torch.arange(in_i, device=device, dtype=torch.long)
                out_i = int(out_layer[bi].item())
                if in_i <= 0 or out_i <= 0:
                    continue
                post_layer = layer_idx > int(depth_per_sample[bi].item() - 1)
                if post_layer:
                    d = min(in_i, out_i)
                    if d > 0:
                        eye_idx = torch.arange(d, device=device, dtype=torch.long)
                        w[bi, active_idx[:d], eye_idx] = 1.0
                else:
                    g = None if generators is None else generators[bi]
                    if g is None:
                        w_b = torch.randn((in_i, out_i), device=device, dtype=torch.float32)
                        b_b = torch.randn((out_i,), device=device, dtype=torch.float32)
                    else:
                        w_b = torch.randn((in_i, out_i), device=device, dtype=torch.float32, generator=g)
                        b_b = torch.randn((out_i,), device=device, dtype=torch.float32, generator=g)
                    w_b = w_b * (init_std[bi] / math.sqrt(max(1, in_i)))
                    w_b = self._project_matrix_fro_norm(w_b, float(weight_cap[bi].item()))
                    w[bi, active_idx, :out_i] = w_b
                    b[bi, :out_i] = b_b * (init_std[bi] * 0.1)
                if layer_idx > 0 or input_mask is None:
                    in_mask[bi, :in_i] = 1.0
                out_mask[bi, :out_i] = 1.0
                if ragged_input_index is not None and in_i > 0:
                    ragged_input_index[bi, :in_i] = active_idx
            layers.append(
                {
                    "w": w,
                    "b": b,
                    "in_mask": in_mask,
                    "out_mask": out_mask,
                    "in_cap": in_cap,
                    "activation_mask": hidden_active,
                    "ragged_input_index": ragged_input_index,
                    "ragged_in_sizes": in_layer.clone() if ragged_affine_enabled else None,
                    "ragged_out_sizes": out_layer.clone() if ragged_affine_enabled else None,
                }
            )

        final_out_mask = layers[-1]["out_mask"]

        def _core_fn(x):
            z = x
            for li, layer in enumerate(layers):
                z_in = z[:, :layer["in_cap"]]
                if layer["ragged_input_index"] is None:
                    z = self._batch_affine(z_in * layer["in_mask"], layer["w"], layer["b"])
                else:
                    z = self._batch_affine_with_layout(
                        z_in,
                        layer["w"],
                        layer["b"],
                        input_index=layer["ragged_input_index"],
                        in_sizes=layer["ragged_in_sizes"],
                        out_sizes=layer["ragged_out_sizes"],
                    )
                z = z * layer["out_mask"]
                if li < (len(layers) - 1):
                    activation_mask = layer["activation_mask"].unsqueeze(1)
                    if torch.any(activation_mask):
                        if not activation_mixed:
                            if activation_single == 1:
                                z_act = torch.relu(z)
                            elif activation_single == 2:
                                z_act = z
                            else:
                                z_act = torch.tanh(z)
                        else:
                            z_linear = z
                            z_tanh = torch.tanh(z_linear)
                            z_relu = torch.relu(z_linear)
                            z_act = torch.where(activation_relu_mask, z_relu, z_tanh)
                            z_act = torch.where(activation_identity_mask, z_linear, z_act)
                        z = torch.where(activation_mask, z_act, z)
            return z

        def fn(x, generators_for_noise=None, noise_eps=None, stable_input=False):
            if self._envgen_checkpoint_active(x):
                # Rollout reuses the env-input buffer across timesteps; checkpoint
                # needs a stable snapshot for backward recompute.
                x_checkpoint = x if bool(stable_input) else x.clone()
                z = checkpoint(
                    _core_fn,
                    x_checkpoint,
                    use_reentrant=bool(self.envgen_checkpoint_reentrant),
                    preserve_rng_state=False,
                )
            else:
                z = _core_fn(x)
            if (noise_eps is None) and torch.any(noise_std > 0):
                noise_eps = self._sample_scaled_noise_batch(
                    generators_for_noise,
                    scale=noise_std,
                    width=z.shape[1],
                    device=z.device,
                    dtype=z.dtype,
                )
            if noise_eps is not None:
                z = z + noise_eps
            z = torch.tanh(z)
            return z * final_out_mask

        fn._envgen_checkpoint_enabled = bool(self.envgen_checkpoint)
        fn._noise_scale = noise_std
        fn._out_width = int(final_out_mask.shape[1])

        return fn

    @staticmethod
    def _consume_init_random_tensor(shape, device, *, generator=None, uniform=False):
        if uniform:
            if generator is None:
                torch.rand(shape, device=device, dtype=torch.float32)
            else:
                torch.rand(shape, device=device, dtype=torch.float32, generator=generator)
        else:
            if generator is None:
                torch.randn(shape, device=device, dtype=torch.float32)
            else:
                torch.randn(shape, device=device, dtype=torch.float32, generator=generator)

    def _consume_scm_hetero_batch_init_rng(
        self,
        in_dims,
        out_dims,
        h_list,
        device,
        depth_values,
        generators=None,
        input_mask=None,
    ):
        batch_size = int(len(h_list))
        if batch_size <= 0:
            return
        if torch.is_tensor(depth_values):
            depth_per_sample = depth_values.to(device=device, dtype=torch.long)
        elif isinstance(depth_values, (list, tuple)):
            depth_per_sample = torch.tensor(
                [max(2, int(d)) for d in depth_values],
                device=device,
                dtype=torch.long,
            )
        else:
            depth_per_sample = torch.full(
                (batch_size,),
                int(max(2, int(depth_values))),
                device=device,
                dtype=torch.long,
            )
        in_dims = torch.as_tensor(in_dims, device=device, dtype=torch.long)
        out_dims = torch.as_tensor(out_dims, device=device, dtype=torch.long)
        if input_mask is not None:
            input_mask = torch.as_tensor(input_mask, device=device, dtype=torch.float32)
            in_dims = input_mask.to(dtype=torch.long).sum(dim=1)
        hidden_dims = torch.tensor(
            [max(int(out_dims[bi].item()), int(h["prior_mlp_hidden_dim"])) for bi, h in enumerate(h_list)],
            device=device,
            dtype=torch.long,
        )
        depth = int(depth_per_sample.max().item())
        for layer_idx in range(depth):
            hidden_active = layer_idx < (depth_per_sample - 1)
            if layer_idx == 0:
                in_layer = in_dims
            else:
                in_layer = torch.where(
                    layer_idx <= (depth_per_sample - 1),
                    hidden_dims,
                    out_dims,
                )
            out_layer = torch.where(hidden_active, hidden_dims, out_dims)
            for bi in range(batch_size):
                in_i = int(in_layer[bi].item())
                out_i = int(out_layer[bi].item())
                if in_i <= 0 or out_i <= 0:
                    continue
                if layer_idx > int(depth_per_sample[bi].item() - 1):
                    continue
                g = None if generators is None else generators[bi]
                self._consume_init_random_tensor((in_i, out_i), device, generator=g, uniform=False)
                self._consume_init_random_tensor((out_i,), device, generator=g, uniform=False)

    def _consume_gp_hetero_batch_init_rng(
        self,
        in_dims,
        out_dims,
        h_list,
        device,
        generators=None,
        input_mask=None,
        sample_in_dims=None,
    ):
        batch_size = int(len(h_list))
        if batch_size <= 0:
            return
        in_dims = torch.as_tensor(in_dims, device=device, dtype=torch.long)
        if sample_in_dims is None:
            sample_in_dims = in_dims
        else:
            sample_in_dims = torch.as_tensor(sample_in_dims, device=device, dtype=torch.long)
        out_dims = torch.as_tensor(out_dims, device=device, dtype=torch.long)
        if input_mask is not None:
            input_mask = torch.as_tensor(input_mask, device=device, dtype=torch.float32)
            in_dims = input_mask.to(dtype=torch.long).sum(dim=1)
        m_dims = torch.tensor(
            [max(8, int(h["gp_rff_features"])) for h in h_list],
            device=device,
            dtype=torch.long,
        )
        for bi in range(batch_size):
            sample_in_i = int(sample_in_dims[bi].item())
            m_i = int(m_dims[bi].item())
            out_i = int(out_dims[bi].item())
            g = None if generators is None else generators[bi]
            self._consume_init_random_tensor((sample_in_i, m_i), device, generator=g, uniform=False)
            self._consume_init_random_tensor((m_i,), device, generator=g, uniform=True)
            self._consume_init_random_tensor((m_i, out_i), device, generator=g, uniform=False)

    def _build_gp_hetero_batch_fn(
        self,
        in_dims,
        out_dims,
        h_list,
        device,
        generators=None,
        input_mask=None,
        sample_in_dims=None,
    ):
        batch_size = len(h_list)
        device_obj = device if isinstance(device, torch.device) else torch.device(str(device))
        ragged_affine_enabled = bool(self.envgen_ragged_affine and device_obj.type == "cuda")
        gp_input_fused_enabled = bool(self.fused_transition_gp_input_fused and device_obj.type == "cuda")
        gp_output_fused_enabled = bool(self.fused_transition_gp_output_fused and device_obj.type == "cuda")
        gp_rff_fused_enabled = bool(
            self.fused_transition_gp_rff_fused and triton is not None and device_obj.type == "cuda"
        )
        gp_projection_timing_enabled = bool(self.profile_gp_projection_timing)
        in_dims = torch.as_tensor(in_dims, device=device, dtype=torch.long)
        if sample_in_dims is None:
            sample_in_dims = in_dims
        else:
            sample_in_dims = torch.as_tensor(sample_in_dims, device=device, dtype=torch.long)
        out_dims = torch.as_tensor(out_dims, device=device, dtype=torch.long)
        if input_mask is not None:
            input_mask = torch.as_tensor(input_mask, device=device, dtype=torch.float32)
            in_dims = input_mask.to(dtype=torch.long).sum(dim=1)
        m_dims = torch.tensor([max(8, int(h["gp_rff_features"])) for h in h_list], device=device, dtype=torch.long)
        lengthscale = torch.tensor(
            [max(1e-6, float(h["lengthscale"])) for h in h_list],
            device=device,
            dtype=torch.float32,
        )
        outputscale = torch.tensor([float(h["outputscale"]) for h in h_list], device=device, dtype=torch.float32)
        noise = torch.tensor([float(h["noise"]) for h in h_list], device=device, dtype=torch.float32)
        weight_cap_values = []
        outputscale_cap_values = []
        for h in h_list:
            w_cap = self._resolve_lipschitz_weight_cap(h)
            s_cap = self._resolve_lipschitz_gp_outputscale_cap(h)
            weight_cap_values.append(float(w_cap) if w_cap is not None else float("inf"))
            outputscale_cap_values.append(float(s_cap) if s_cap is not None else float("inf"))
        weight_cap = torch.tensor(weight_cap_values, device=device, dtype=torch.float32)
        outputscale_cap = torch.tensor(outputscale_cap_values, device=device, dtype=torch.float32)
        outputscale = torch.sign(outputscale) * torch.minimum(outputscale.abs(), outputscale_cap)

        if input_mask is not None:
            in_cap = int(input_mask.shape[1])
            in_mask = input_mask.clone()
        else:
            in_cap = int(in_dims.max().item())
            in_mask = torch.zeros((batch_size, in_cap), device=device, dtype=torch.float32)
        w_in_cap = int(max(1, int(in_dims.max().item()))) if gp_input_fused_enabled else int(in_cap)
        m_cap = int(m_dims.max().item())
        out_cap = int(out_dims.max().item())
        w = torch.zeros((batch_size, w_in_cap, m_cap), device=device, dtype=torch.float32)
        b = torch.zeros((batch_size, m_cap), device=device, dtype=torch.float32)
        a = torch.zeros((batch_size, m_cap, out_cap), device=device, dtype=torch.float32)
        m_mask = torch.zeros((batch_size, m_cap), device=device, dtype=torch.float32)
        out_mask = torch.zeros((batch_size, out_cap), device=device, dtype=torch.float32)
        w_source_index = None
        w_layout_index = None
        a_input_index = None
        if gp_input_fused_enabled:
            w_source_index = torch.zeros((batch_size, w_in_cap), device=device, dtype=torch.long)
        elif ragged_affine_enabled or (gp_rff_fused_enabled and input_mask is not None):
            w_layout_index = torch.zeros((batch_size, in_cap), device=device, dtype=torch.long)
        if ragged_affine_enabled and not gp_output_fused_enabled:
            a_input_index = torch.zeros((batch_size, m_cap), device=device, dtype=torch.long)

        for bi in range(batch_size):
            if input_mask is not None:
                active_idx = torch.nonzero(input_mask[bi] > 0, as_tuple=False).squeeze(1)
                in_i = int(active_idx.numel())
            else:
                in_i = int(in_dims[bi].item())
                active_idx = torch.arange(in_i, device=device, dtype=torch.long)
            sample_in_i = int(sample_in_dims[bi].item())
            m_i = int(m_dims[bi].item())
            out_i = int(out_dims[bi].item())
            g = None if generators is None else generators[bi]
            if g is None:
                w_b = torch.randn((sample_in_i, m_i), device=device, dtype=torch.float32)
                b_b = torch.rand((m_i,), device=device, dtype=torch.float32)
                a_b = torch.randn((m_i, out_i), device=device, dtype=torch.float32)
            else:
                w_b = torch.randn((sample_in_i, m_i), device=device, dtype=torch.float32, generator=g)
                b_b = torch.rand((m_i,), device=device, dtype=torch.float32, generator=g)
                a_b = torch.randn((m_i, out_i), device=device, dtype=torch.float32, generator=g)
            w_b = w_b / lengthscale[bi]
            w_b = self._project_matrix_fro_norm(w_b, float(weight_cap[bi].item()))
            if sample_in_i != in_i:
                w_b = w_b[:in_i]
            a_b = a_b / math.sqrt(max(1, m_i))
            a_b = self._project_matrix_fro_norm(a_b, float(weight_cap[bi].item()))
            if gp_input_fused_enabled:
                w[bi, :in_i, :m_i] = w_b
            else:
                w[bi, active_idx, :m_i] = w_b
            b[bi, :m_i] = 2.0 * math.pi * b_b
            a[bi, :m_i, :out_i] = a_b
            if input_mask is None:
                in_mask[bi, :in_i] = 1.0
            m_mask[bi, :m_i] = 1.0
            out_mask[bi, :out_i] = 1.0
            if gp_input_fused_enabled and in_i > 0:
                w_source_index[bi, :in_i] = active_idx
            elif w_layout_index is not None and in_i > 0:
                w_layout_index[bi, :in_i] = active_idx
            if a_input_index is not None and m_i > 0:
                a_input_index[bi, :m_i] = torch.arange(m_i, device=device, dtype=torch.long)

        a_projection_activation_codes = None
        a_out_tile_batch = None
        a_out_tile_offsets = None
        a_in_tile_batch = None
        a_in_tile_offsets = None
        if gp_output_fused_enabled:
            a_projection_activation_codes = torch.full((batch_size,), -1, device=device, dtype=torch.long)
            a_out_tile_batch, a_out_tile_offsets = self._build_active_tile_map(out_dims, 32)
            a_in_tile_batch, a_in_tile_offsets = self._build_active_tile_map(m_dims, 32)
        rff_activation_codes = None
        rff_out_tile_batch = None
        rff_out_tile_offsets = None
        rff_in_tile_batch = None
        rff_in_tile_offsets = None
        rff_block_o = int(self.gp_rff_tiled_block_o)
        rff_block_k = int(self.gp_rff_tiled_block_k)
        rff_num_warps = int(self.gp_rff_tiled_num_warps)
        if gp_rff_fused_enabled:
            rff_activation_codes = torch.full((batch_size,), 3, device=device, dtype=torch.long)
            rff_out_tile_batch, rff_out_tile_offsets = self._build_active_tile_map(m_dims, rff_block_o)
            rff_in_tile_batch, rff_in_tile_offsets = self._build_active_tile_map(in_dims, rff_block_k)
        gp_projection_profile = {
            "first_projection_wall_s": 0.0,
            "second_projection_wall_s": 0.0,
            "call_count": 0,
            "rff_fused_call_count": 0,
            "shared_total_wall_s": 0.0,
            "shared_core_wall_s": 0.0,
            "shared_noise_wall_s": 0.0,
            "shared_checkpoint_wall_s": 0.0,
            "shared_post_wall_s": 0.0,
            "shared_call_count": 0,
        }

        def _consume_gp_projection_profile():
            stats = dict(gp_projection_profile)
            for key in gp_projection_profile:
                gp_projection_profile[key] = 0.0 if "wall_s" in key else 0
            return stats

        def _core_fn(x):
            x_in = x[:, :in_cap]
            first_proj_t0 = time.perf_counter() if gp_projection_timing_enabled else None
            if gp_input_fused_enabled:
                x_packed = torch.gather(x_in, 1, w_source_index)
                phi_pre = self._batch_affine(x_packed[:, :w_in_cap], w, b)
                phi = torch.cos(phi_pre) * m_mask
            elif gp_rff_fused_enabled:
                if w_layout_index is None:
                    phi = self._batch_affine_prefix_tiled(
                        x_in[:, :in_cap] * in_mask,
                        w,
                        b,
                        in_sizes=in_dims,
                        out_sizes=m_dims,
                        activation_codes=rff_activation_codes,
                        out_tile_batch=rff_out_tile_batch,
                        out_tile_offsets=rff_out_tile_offsets,
                        in_tile_batch=rff_in_tile_batch,
                        in_tile_offsets=rff_in_tile_offsets,
                        block_o=rff_block_o,
                        block_k=rff_block_k,
                        num_warps=rff_num_warps,
                    )
                else:
                    phi = self._batch_affine_with_layout_tiled(
                        x_in,
                        w,
                        b,
                        input_index=w_layout_index,
                        in_sizes=in_dims,
                        out_sizes=m_dims,
                        activation_codes=rff_activation_codes,
                        out_tile_batch=rff_out_tile_batch,
                        out_tile_offsets=rff_out_tile_offsets,
                        in_tile_batch=rff_in_tile_batch,
                        in_tile_offsets=rff_in_tile_offsets,
                        block_o=rff_block_o,
                        block_k=rff_block_k,
                        num_warps=rff_num_warps,
                    )
                phi = phi * m_mask
            elif w_layout_index is None:
                phi_pre = self._batch_affine(x_in * in_mask, w, b)
                phi = torch.cos(phi_pre) * m_mask
            else:
                phi_pre = self._batch_affine_with_layout(
                    x_in,
                    w,
                    b,
                    input_index=w_layout_index,
                    in_sizes=in_dims,
                    out_sizes=m_dims,
                )
                phi = torch.cos(phi_pre) * m_mask
            if first_proj_t0 is not None:
                gp_projection_profile["first_projection_wall_s"] += (time.perf_counter() - first_proj_t0)
                gp_projection_profile["call_count"] += 1
                if gp_rff_fused_enabled:
                    gp_projection_profile["rff_fused_call_count"] += 1
            second_proj_t0 = time.perf_counter() if gp_projection_timing_enabled else None
            if gp_output_fused_enabled:
                y = outputscale[:, None] * self._batch_affine_prefix_tiled(
                    phi[:, :m_cap],
                    a,
                    None,
                    in_sizes=m_dims,
                    out_sizes=out_dims,
                    activation_codes=a_projection_activation_codes,
                    out_tile_batch=a_out_tile_batch,
                    out_tile_offsets=a_out_tile_offsets,
                    in_tile_batch=a_in_tile_batch,
                    in_tile_offsets=a_in_tile_offsets,
                )
            elif a_input_index is None:
                y = outputscale[:, None] * self._batch_affine(phi, a, None)
            else:
                y = outputscale[:, None] * self._batch_affine_with_layout(
                    phi,
                    a,
                    None,
                    input_index=a_input_index,
                    in_sizes=m_dims,
                    out_sizes=out_dims,
                )
            if second_proj_t0 is not None:
                gp_projection_profile["second_projection_wall_s"] += (time.perf_counter() - second_proj_t0)
            y = y * out_mask
            return y

        def fn(x, generators_for_noise=None, noise_eps=None, stable_input=False):
            if self._envgen_checkpoint_active(x):
                # Rollout reuses the env-input buffer across timesteps; checkpoint
                # needs a stable snapshot for backward recompute.
                x_checkpoint = x if bool(stable_input) else x.clone()
                y = checkpoint(
                    _core_fn,
                    x_checkpoint,
                    use_reentrant=bool(self.envgen_checkpoint_reentrant),
                    preserve_rng_state=False,
                )
            else:
                y = _core_fn(x)
            if (noise_eps is None) and torch.any(noise > 0):
                noise_eps = self._sample_scaled_noise_batch(
                    generators_for_noise,
                    scale=noise,
                    width=y.shape[1],
                    device=y.device,
                    dtype=y.dtype,
                )
            if noise_eps is not None:
                y = y + noise_eps
            y = torch.tanh(y)
            return y * out_mask

        fn._envgen_checkpoint_enabled = bool(self.envgen_checkpoint)
        fn._noise_scale = noise
        fn._out_width = int(out_mask.shape[1])
        fn._gp_input_rff_fused = bool(gp_input_fused_enabled)
        fn._gp_output_projection_fused = bool(gp_output_fused_enabled)
        fn._gp_rff_fused = bool(gp_rff_fused_enabled)
        fn._gp_rff_block_o = int(rff_block_o)
        fn._gp_rff_block_k = int(rff_block_k)
        fn._gp_rff_num_warps = int(rff_num_warps)
        fn._consume_gp_projection_profile = _consume_gp_projection_profile

        return fn

    def _build_scm_hetero_paired_transition_fns(
        self,
        in_dims,
        state_dims,
        h_list,
        device,
        depth_values,
        activation_names,
        input_mask=None,
    ):
        batch_size = len(h_list)
        reward_dims = torch.ones((batch_size,), device=device, dtype=torch.long)
        if torch.is_tensor(depth_values):
            depth_per_sample = depth_values.to(device=device, dtype=torch.long)
        elif isinstance(depth_values, (list, tuple)):
            depth_per_sample = torch.tensor(
                [max(2, int(d)) for d in depth_values],
                device=device,
                dtype=torch.long,
            )
        else:
            depth_per_sample = torch.full(
                (batch_size,),
                int(max(2, int(depth_values))),
                device=device,
                dtype=torch.long,
            )
        if int(depth_per_sample.numel()) != batch_size:
            raise ValueError("depth_values must match h_list length")
        depth = int(depth_per_sample.max().item())
        in_dims = torch.as_tensor(in_dims, device=device, dtype=torch.long)
        state_dims = torch.as_tensor(state_dims, device=device, dtype=torch.long)
        if input_mask is not None:
            input_mask = torch.as_tensor(input_mask, device=device, dtype=torch.float32)
            in_dims = input_mask.to(dtype=torch.long).sum(dim=1)
        hidden_state_dims = torch.tensor(
            [max(int(state_dims[bi].item()), int(h["prior_mlp_hidden_dim"])) for bi, h in enumerate(h_list)],
            device=device,
            dtype=torch.long,
        )
        hidden_reward_dims = torch.tensor(
            [max(1, int(h["prior_mlp_hidden_dim"])) for h in h_list],
            device=device,
            dtype=torch.long,
        )
        init_std = torch.tensor([float(h["init_std"]) for h in h_list], device=device, dtype=torch.float32)
        noise_std = torch.tensor([float(h["noise_std"]) for h in h_list], device=device, dtype=torch.float32)
        weight_cap = torch.tensor(
            [
                float(self._resolve_lipschitz_weight_cap(h))
                if self._resolve_lipschitz_weight_cap(h) is not None
                else float("inf")
                for h in h_list
            ],
            device=device,
            dtype=torch.float32,
        )
        if isinstance(activation_names, str):
            activation_values = [self._activation_name(activation_names)] * batch_size
        else:
            activation_values = [self._activation_name(v) for v in activation_names]
        if len(activation_values) != batch_size:
            raise ValueError("activation_names must match h_list length")
        activation_codes = []
        for name in activation_values:
            if name == "relu":
                activation_codes.append(1)
            elif name == "identity":
                activation_codes.append(2)
            else:
                activation_codes.append(0)
        activation_codes = torch.tensor(activation_codes, device=device, dtype=torch.long)
        activation_mixed = not bool(torch.all(activation_codes == activation_codes[0]))
        activation_relu_mask = (activation_codes == 1).unsqueeze(1)
        activation_identity_mask = (activation_codes == 2).unsqueeze(1)
        activation_single = int(activation_codes[0].item())

        state_layers = []
        reward_layers = []
        for layer_idx in range(depth):
            hidden_active = layer_idx < (depth_per_sample - 1)
            if layer_idx == 0:
                if input_mask is None:
                    state_in_layer = in_dims
                    reward_in_layer = in_dims
                    state_in_cap = int(state_in_layer.max().item())
                    reward_in_cap = int(reward_in_layer.max().item())
                    state_in_mask = torch.zeros((batch_size, state_in_cap), device=device, dtype=torch.float32)
                    reward_in_mask = torch.zeros((batch_size, reward_in_cap), device=device, dtype=torch.float32)
                else:
                    state_in_layer = in_dims
                    reward_in_layer = in_dims
                    state_in_cap = int(input_mask.shape[1])
                    reward_in_cap = int(input_mask.shape[1])
                    state_in_mask = input_mask.clone()
                    reward_in_mask = input_mask.clone()
            else:
                state_in_layer = torch.where(layer_idx <= (depth_per_sample - 1), hidden_state_dims, state_dims)
                reward_in_layer = torch.where(layer_idx <= (depth_per_sample - 1), hidden_reward_dims, reward_dims)
                state_in_cap = int(state_in_layer.max().item())
                reward_in_cap = int(reward_in_layer.max().item())
                state_in_mask = torch.zeros((batch_size, state_in_cap), device=device, dtype=torch.float32)
                reward_in_mask = torch.zeros((batch_size, reward_in_cap), device=device, dtype=torch.float32)

            state_out_layer = torch.where(hidden_active, hidden_state_dims, state_dims)
            reward_out_layer = torch.where(hidden_active, hidden_reward_dims, reward_dims)
            state_out_cap = int(state_out_layer.max().item())
            reward_out_cap = int(reward_out_layer.max().item())

            state_w = torch.zeros((batch_size, state_in_cap, state_out_cap), device=device, dtype=torch.float32)
            state_b = torch.zeros((batch_size, state_out_cap), device=device, dtype=torch.float32)
            state_out_mask = torch.zeros((batch_size, state_out_cap), device=device, dtype=torch.float32)
            reward_w = torch.zeros((batch_size, reward_in_cap, reward_out_cap), device=device, dtype=torch.float32)
            reward_b = torch.zeros((batch_size, reward_out_cap), device=device, dtype=torch.float32)
            reward_out_mask = torch.zeros((batch_size, reward_out_cap), device=device, dtype=torch.float32)

            for bi in range(batch_size):
                if layer_idx == 0 and input_mask is not None:
                    active_idx = torch.nonzero(input_mask[bi] > 0, as_tuple=False).squeeze(1)
                    state_in_i = int(active_idx.numel())
                else:
                    state_in_i = int(state_in_layer[bi].item())
                    active_idx = torch.arange(state_in_i, device=device, dtype=torch.long)
                state_out_i = int(state_out_layer[bi].item())
                if state_in_i > 0 and state_out_i > 0:
                    post_layer = layer_idx > int(depth_per_sample[bi].item() - 1)
                    if post_layer:
                        d = min(state_in_i, state_out_i)
                        if d > 0:
                            eye_idx = torch.arange(d, device=device, dtype=torch.long)
                            state_w[bi, active_idx[:d], eye_idx] = 1.0
                    else:
                        w_b = torch.randn((state_in_i, state_out_i), device=device, dtype=torch.float32)
                        b_b = torch.randn((state_out_i,), device=device, dtype=torch.float32)
                        w_b = w_b * (init_std[bi] / math.sqrt(max(1, state_in_i)))
                        w_b = self._project_matrix_fro_norm(w_b, float(weight_cap[bi].item()))
                        state_w[bi, active_idx, :state_out_i] = w_b
                        state_b[bi, :state_out_i] = b_b * (init_std[bi] * 0.1)
                    if layer_idx > 0 or input_mask is None:
                        state_in_mask[bi, :state_in_i] = 1.0
                    state_out_mask[bi, :state_out_i] = 1.0

            for bi in range(batch_size):
                if layer_idx == 0 and input_mask is not None:
                    active_idx = torch.nonzero(input_mask[bi] > 0, as_tuple=False).squeeze(1)
                    reward_in_i = int(active_idx.numel())
                else:
                    reward_in_i = int(reward_in_layer[bi].item())
                    active_idx = torch.arange(reward_in_i, device=device, dtype=torch.long)
                reward_out_i = int(reward_out_layer[bi].item())
                if reward_in_i > 0 and reward_out_i > 0:
                    post_layer = layer_idx > int(depth_per_sample[bi].item() - 1)
                    if post_layer:
                        d = min(reward_in_i, reward_out_i)
                        if d > 0:
                            eye_idx = torch.arange(d, device=device, dtype=torch.long)
                            reward_w[bi, active_idx[:d], eye_idx] = 1.0
                    else:
                        w_b = torch.randn((reward_in_i, reward_out_i), device=device, dtype=torch.float32)
                        b_b = torch.randn((reward_out_i,), device=device, dtype=torch.float32)
                        w_b = w_b * (init_std[bi] / math.sqrt(max(1, reward_in_i)))
                        w_b = self._project_matrix_fro_norm(w_b, float(weight_cap[bi].item()))
                        reward_w[bi, active_idx, :reward_out_i] = w_b
                        reward_b[bi, :reward_out_i] = b_b * (init_std[bi] * 0.1)
                    if layer_idx > 0 or input_mask is None:
                        reward_in_mask[bi, :reward_in_i] = 1.0
                    reward_out_mask[bi, :reward_out_i] = 1.0

            state_layers.append(
                {
                    "w": state_w,
                    "b": state_b,
                    "in_mask": state_in_mask,
                    "out_mask": state_out_mask,
                    "in_cap": state_in_cap,
                    "activation_mask": hidden_active,
                }
            )
            reward_layers.append(
                {
                    "w": reward_w,
                    "b": reward_b,
                    "in_mask": reward_in_mask,
                    "out_mask": reward_out_mask,
                    "in_cap": reward_in_cap,
                    "activation_mask": hidden_active,
                }
            )

        def _apply_layers(x, layers, final_out_mask):
            z = x
            for li, layer in enumerate(layers):
                z_in = z[:, :layer["in_cap"]] * layer["in_mask"]
                z = self._batch_affine(z_in, layer["w"], layer["b"])
                z = z * layer["out_mask"]
                if li < (len(layers) - 1):
                    activation_mask = layer["activation_mask"].unsqueeze(1)
                    if torch.any(activation_mask):
                        if not activation_mixed:
                            if activation_single == 1:
                                z_act = torch.relu(z)
                            elif activation_single == 2:
                                z_act = z
                            else:
                                z_act = torch.tanh(z)
                        else:
                            z_linear = z
                            z_tanh = torch.tanh(z_linear)
                            z_relu = torch.relu(z_linear)
                            z_act = torch.where(activation_relu_mask, z_relu, z_tanh)
                            z_act = torch.where(activation_identity_mask, z_linear, z_act)
                        z = torch.where(activation_mask, z_act, z)
            return z * final_out_mask

        state_final_out_mask = state_layers[-1]["out_mask"]
        reward_final_out_mask = reward_layers[-1]["out_mask"]

        def _make_branch_fn(layers, final_out_mask):
            def _core_fn(x):
                return _apply_layers(x, layers, final_out_mask)

            def fn(x, generators_for_noise=None, noise_eps=None, stable_input=False):
                if self._envgen_checkpoint_active(x):
                    x_checkpoint = x if bool(stable_input) else x.clone()
                    z = checkpoint(
                        _core_fn,
                        x_checkpoint,
                        use_reentrant=bool(self.envgen_checkpoint_reentrant),
                        preserve_rng_state=False,
                    )
                else:
                    z = _core_fn(x)
                if (noise_eps is None) and torch.any(noise_std > 0):
                    noise_eps = self._sample_scaled_noise_batch(
                        generators_for_noise,
                        scale=noise_std,
                        width=z.shape[1],
                        device=z.device,
                        dtype=z.dtype,
                    )
                if noise_eps is not None:
                    z = z + noise_eps
                z = torch.tanh(z)
                return z * final_out_mask

            fn._envgen_checkpoint_enabled = bool(self.envgen_checkpoint)
            fn._noise_scale = noise_std
            fn._out_width = int(final_out_mask.shape[1])
            return fn

        return (
            _make_branch_fn(state_layers, state_final_out_mask),
            _make_branch_fn(reward_layers, reward_final_out_mask),
        )

    def _build_gp_hetero_paired_transition_fns(self, in_dims, state_dims, h_list, device, input_mask=None):
        batch_size = len(h_list)
        reward_dims = torch.ones((batch_size,), device=device, dtype=torch.long)
        in_dims = torch.as_tensor(in_dims, device=device, dtype=torch.long)
        state_dims = torch.as_tensor(state_dims, device=device, dtype=torch.long)
        if input_mask is not None:
            input_mask = torch.as_tensor(input_mask, device=device, dtype=torch.float32)
            in_dims = input_mask.to(dtype=torch.long).sum(dim=1)
        m_dims = torch.tensor([max(8, int(h["gp_rff_features"])) for h in h_list], device=device, dtype=torch.long)
        lengthscale = torch.tensor(
            [max(1e-6, float(h["lengthscale"])) for h in h_list],
            device=device,
            dtype=torch.float32,
        )
        outputscale = torch.tensor([float(h["outputscale"]) for h in h_list], device=device, dtype=torch.float32)
        noise = torch.tensor([float(h["noise"]) for h in h_list], device=device, dtype=torch.float32)
        weight_cap_values = []
        outputscale_cap_values = []
        for h in h_list:
            w_cap = self._resolve_lipschitz_weight_cap(h)
            s_cap = self._resolve_lipschitz_gp_outputscale_cap(h)
            weight_cap_values.append(float(w_cap) if w_cap is not None else float("inf"))
            outputscale_cap_values.append(float(s_cap) if s_cap is not None else float("inf"))
        weight_cap = torch.tensor(weight_cap_values, device=device, dtype=torch.float32)
        outputscale_cap = torch.tensor(outputscale_cap_values, device=device, dtype=torch.float32)
        outputscale = torch.sign(outputscale) * torch.minimum(outputscale.abs(), outputscale_cap)

        if input_mask is not None:
            in_cap = int(input_mask.shape[1])
            in_mask = input_mask.clone()
        else:
            in_cap = int(in_dims.max().item())
            in_mask = torch.zeros((batch_size, in_cap), device=device, dtype=torch.float32)
        m_cap = int(m_dims.max().item())
        state_out_cap = int(state_dims.max().item())
        reward_out_cap = 1
        state_w = torch.zeros((batch_size, in_cap, m_cap), device=device, dtype=torch.float32)
        state_b = torch.zeros((batch_size, m_cap), device=device, dtype=torch.float32)
        state_a = torch.zeros((batch_size, m_cap, state_out_cap), device=device, dtype=torch.float32)
        reward_w = torch.zeros((batch_size, in_cap, m_cap), device=device, dtype=torch.float32)
        reward_b = torch.zeros((batch_size, m_cap), device=device, dtype=torch.float32)
        reward_a = torch.zeros((batch_size, m_cap, reward_out_cap), device=device, dtype=torch.float32)
        m_mask = torch.zeros((batch_size, m_cap), device=device, dtype=torch.float32)
        state_out_mask = torch.zeros((batch_size, state_out_cap), device=device, dtype=torch.float32)
        reward_out_mask = torch.zeros((batch_size, reward_out_cap), device=device, dtype=torch.float32)

        for bi in range(batch_size):
            if input_mask is not None:
                active_idx = torch.nonzero(input_mask[bi] > 0, as_tuple=False).squeeze(1)
                in_i = int(active_idx.numel())
            else:
                in_i = int(in_dims[bi].item())
                active_idx = torch.arange(in_i, device=device, dtype=torch.long)
            m_i = int(m_dims[bi].item())
            out_i = int(state_dims[bi].item())
            w_b = torch.randn((in_i, m_i), device=device, dtype=torch.float32)
            b_b = torch.rand((m_i,), device=device, dtype=torch.float32)
            a_b = torch.randn((m_i, out_i), device=device, dtype=torch.float32)
            w_b = w_b / lengthscale[bi]
            w_b = self._project_matrix_fro_norm(w_b, float(weight_cap[bi].item()))
            a_b = a_b / math.sqrt(max(1, m_i))
            a_b = self._project_matrix_fro_norm(a_b, float(weight_cap[bi].item()))
            state_w[bi, active_idx, :m_i] = w_b
            state_b[bi, :m_i] = 2.0 * math.pi * b_b
            state_a[bi, :m_i, :out_i] = a_b
            if input_mask is None:
                in_mask[bi, :in_i] = 1.0
            m_mask[bi, :m_i] = 1.0
            state_out_mask[bi, :out_i] = 1.0

        for bi in range(batch_size):
            if input_mask is not None:
                active_idx = torch.nonzero(input_mask[bi] > 0, as_tuple=False).squeeze(1)
                in_i = int(active_idx.numel())
            else:
                in_i = int(in_dims[bi].item())
                active_idx = torch.arange(in_i, device=device, dtype=torch.long)
            m_i = int(m_dims[bi].item())
            w_b = torch.randn((in_i, m_i), device=device, dtype=torch.float32)
            b_b = torch.rand((m_i,), device=device, dtype=torch.float32)
            a_b = torch.randn((m_i, 1), device=device, dtype=torch.float32)
            w_b = w_b / lengthscale[bi]
            w_b = self._project_matrix_fro_norm(w_b, float(weight_cap[bi].item()))
            a_b = a_b / math.sqrt(max(1, m_i))
            a_b = self._project_matrix_fro_norm(a_b, float(weight_cap[bi].item()))
            reward_w[bi, active_idx, :m_i] = w_b
            reward_b[bi, :m_i] = 2.0 * math.pi * b_b
            reward_a[bi, :m_i, :1] = a_b
            reward_out_mask[bi, :1] = 1.0

        def _make_branch_fn(w, b, a, out_mask):
            def _core_fn(x):
                x_in = x[:, :in_cap] * in_mask
                phi = torch.cos(self._batch_affine(x_in, w, b)) * m_mask
                y = outputscale[:, None] * self._batch_affine(phi, a, None)
                return y * out_mask

            def fn(x, generators_for_noise=None, noise_eps=None, stable_input=False):
                if self._envgen_checkpoint_active(x):
                    x_checkpoint = x if bool(stable_input) else x.clone()
                    y = checkpoint(
                        _core_fn,
                        x_checkpoint,
                        use_reentrant=bool(self.envgen_checkpoint_reentrant),
                        preserve_rng_state=False,
                    )
                else:
                    y = _core_fn(x)
                if (noise_eps is None) and torch.any(noise > 0):
                    noise_eps = self._sample_scaled_noise_batch(
                        generators_for_noise,
                        scale=noise,
                        width=y.shape[1],
                        device=y.device,
                        dtype=y.dtype,
                    )
                if noise_eps is not None:
                    y = y + noise_eps
                y = torch.tanh(y)
                return y * out_mask

            fn._envgen_checkpoint_enabled = bool(self.envgen_checkpoint)
            fn._noise_scale = noise
            fn._out_width = int(out_mask.shape[1])
            return fn

        return (
            _make_branch_fn(state_w, state_b, state_a, state_out_mask),
            _make_branch_fn(reward_w, reward_b, reward_a, reward_out_mask),
        )

    def _build_gp_hetero_output_subgraph_transition_fn(self, in_dims, state_dims, h_list, device, input_mask=None):
        batch_size = int(len(h_list))
        if batch_size <= 0:
            return None
        reward_dims = torch.ones((batch_size,), device=device, dtype=torch.long)
        in_dims = torch.as_tensor(in_dims, device=device, dtype=torch.long)
        state_dims = torch.as_tensor(state_dims, device=device, dtype=torch.long)
        if input_mask is not None:
            input_mask = torch.as_tensor(input_mask, device=device, dtype=torch.float32)
            in_dims = input_mask.to(dtype=torch.long).sum(dim=1)
        m_dims = torch.tensor([max(8, int(h["gp_rff_features"])) for h in h_list], device=device, dtype=torch.long)
        lengthscale = torch.tensor(
            [max(1e-6, float(h["lengthscale"])) for h in h_list],
            device=device,
            dtype=torch.float32,
        )
        outputscale = torch.tensor([float(h["outputscale"]) for h in h_list], device=device, dtype=torch.float32)
        noise = torch.tensor([float(h["noise"]) for h in h_list], device=device, dtype=torch.float32)
        weight_cap_values = []
        outputscale_cap_values = []
        for h in h_list:
            w_cap = self._resolve_lipschitz_weight_cap(h)
            s_cap = self._resolve_lipschitz_gp_outputscale_cap(h)
            weight_cap_values.append(float(w_cap) if w_cap is not None else float("inf"))
            outputscale_cap_values.append(float(s_cap) if s_cap is not None else float("inf"))
        weight_cap = torch.tensor(weight_cap_values, device=device, dtype=torch.float32)
        outputscale_cap = torch.tensor(outputscale_cap_values, device=device, dtype=torch.float32)
        outputscale = torch.sign(outputscale) * torch.minimum(outputscale.abs(), outputscale_cap)

        if input_mask is not None:
            in_cap = int(input_mask.shape[1])
            in_mask = input_mask.clone()
        else:
            in_cap = int(in_dims.max().item())
            in_mask = torch.zeros((batch_size, in_cap), device=device, dtype=torch.float32)
        m_cap = int(m_dims.max().item())
        state_out_cap = int(state_dims.max().item())
        state_w = torch.zeros((batch_size, in_cap, m_cap), device=device, dtype=torch.float32)
        state_b = torch.zeros((batch_size, m_cap), device=device, dtype=torch.float32)
        state_a = torch.zeros((batch_size, m_cap, state_out_cap), device=device, dtype=torch.float32)
        reward_w = torch.zeros((batch_size, in_cap, m_cap), device=device, dtype=torch.float32)
        reward_b = torch.zeros((batch_size, m_cap), device=device, dtype=torch.float32)
        reward_a_vec = torch.zeros((batch_size, m_cap), device=device, dtype=torch.float32)
        m_mask = torch.zeros((batch_size, m_cap), device=device, dtype=torch.float32)
        state_out_mask = torch.zeros((batch_size, state_out_cap), device=device, dtype=torch.float32)
        reward_dual_mask = torch.zeros((batch_size, state_out_cap), device=device, dtype=torch.float32)

        for bi in range(batch_size):
            if input_mask is not None:
                active_idx = torch.nonzero(input_mask[bi] > 0, as_tuple=False).squeeze(1)
                in_i = int(active_idx.numel())
            else:
                in_i = int(in_dims[bi].item())
                active_idx = torch.arange(in_i, device=device, dtype=torch.long)
            m_i = int(m_dims[bi].item())
            out_i = int(state_dims[bi].item())
            w_b = torch.randn((in_i, m_i), device=device, dtype=torch.float32)
            b_b = torch.rand((m_i,), device=device, dtype=torch.float32)
            a_b = torch.randn((m_i, out_i), device=device, dtype=torch.float32)
            w_b = w_b / lengthscale[bi]
            w_b = self._project_matrix_fro_norm(w_b, float(weight_cap[bi].item()))
            a_b = a_b / math.sqrt(max(1, m_i))
            a_b = self._project_matrix_fro_norm(a_b, float(weight_cap[bi].item()))
            state_w[bi, active_idx, :m_i] = w_b
            state_b[bi, :m_i] = 2.0 * math.pi * b_b
            state_a[bi, :m_i, :out_i] = a_b
            if input_mask is None:
                in_mask[bi, :in_i] = 1.0
            m_mask[bi, :m_i] = 1.0
            state_out_mask[bi, :out_i] = 1.0

        for bi in range(batch_size):
            if input_mask is not None:
                active_idx = torch.nonzero(input_mask[bi] > 0, as_tuple=False).squeeze(1)
                in_i = int(active_idx.numel())
            else:
                in_i = int(in_dims[bi].item())
                active_idx = torch.arange(in_i, device=device, dtype=torch.long)
            m_i = int(m_dims[bi].item())
            w_b = torch.randn((in_i, m_i), device=device, dtype=torch.float32)
            b_b = torch.rand((m_i,), device=device, dtype=torch.float32)
            a_b = torch.randn((m_i,), device=device, dtype=torch.float32)
            w_b = w_b / lengthscale[bi]
            w_b = self._project_matrix_fro_norm(w_b, float(weight_cap[bi].item()))
            a_b = a_b / math.sqrt(max(1, m_i))
            a_b = self._project_matrix_fro_norm(a_b.unsqueeze(1), float(weight_cap[bi].item())).squeeze(1)
            reward_w[bi, active_idx, :m_i] = w_b
            reward_b[bi, :m_i] = 2.0 * math.pi * b_b
            reward_a_vec[bi, :m_i] = a_b
            reward_dual_mask[bi, :1] = 1.0

        dual_out_mask = torch.cat([state_out_mask, reward_dual_mask], dim=0)
        checkpoint_enabled = bool(self.envgen_checkpoint)

        def _core_dual(x_state, x_reward):
            state_in = x_state[:, :in_cap] * in_mask
            reward_in = x_reward[:, :in_cap] * in_mask
            state_phi = torch.cos(self._batch_affine(state_in, state_w, state_b)) * m_mask
            reward_phi = torch.cos(self._batch_affine(reward_in, reward_w, reward_b)) * m_mask
            state_y = outputscale[:, None] * self._batch_affine(state_phi, state_a, None)
            reward_y = outputscale[:, None] * (reward_phi * reward_a_vec).sum(dim=1, keepdim=True)
            out_dual = torch.zeros((2 * batch_size, state_out_cap), device=x_state.device, dtype=x_state.dtype)
            out_dual[:batch_size] = state_y * state_out_mask
            out_dual[batch_size:, :1] = reward_y
            return out_dual

        def transition_fn(x, generators_for_noise=None, x_is_dual_packed=False):
            if bool(x_is_dual_packed):
                x_state = x[:batch_size]
                x_reward = x[batch_size: batch_size * 2]
                state_input = x_state
                reward_input = x_reward
            else:
                x_state = x
                x_reward = x
                state_input = x_state
                reward_input = x_reward
            if checkpoint_enabled and self._envgen_checkpoint_active(state_input):
                if bool(x_is_dual_packed):
                    checkpoint_state = state_input
                    checkpoint_reward = reward_input
                else:
                    shared_input = state_input.clone()
                    checkpoint_state = shared_input
                    checkpoint_reward = shared_input
                out_dual = checkpoint(
                    _core_dual,
                    checkpoint_state,
                    checkpoint_reward,
                    use_reentrant=bool(self.envgen_checkpoint_reentrant),
                    preserve_rng_state=False,
                )
            else:
                out_dual = _core_dual(state_input, reward_input)
            dual_noise = None
            dual_scale = torch.cat([noise, noise], dim=0)
            if generators_for_noise is not None:
                generators_list = list(generators_for_noise)
                if len(generators_list) != batch_size:
                    raise ValueError("generators_for_noise must match batch size")
                if bool(torch.any(dual_scale > 0)):
                    dual_noise = self._sample_scaled_noise_batch(
                        generators_list + generators_list,
                        scale=dual_scale,
                        width=state_out_cap,
                        device=out_dual.device,
                        dtype=out_dual.dtype,
                    )
            elif bool(torch.any(dual_scale > 0)):
                dual_noise = self._sample_scaled_noise_batch(
                    None,
                    scale=dual_scale,
                    width=state_out_cap,
                    device=out_dual.device,
                    dtype=out_dual.dtype,
                )
            if dual_noise is not None:
                out_dual = out_dual + dual_noise
            out_dual = torch.tanh(out_dual)
            out_dual = out_dual * dual_out_mask
            return out_dual[:batch_size, :state_out_cap], out_dual[batch_size:, :1]

        transition_fn._envgen_checkpoint_enabled = checkpoint_enabled
        transition_fn._gp_output_subgraph_fused = True
        return transition_fn

    def _build_scm_hetero_hidden_fused_transition_fn(
        self,
        in_dims,
        state_dims,
        h_list,
        device,
        depth_values,
        activation_names,
        input_mask=None,
    ):
        batch_size = int(len(h_list))
        if batch_size <= 0:
            return None
        if torch.is_tensor(depth_values):
            depth_per_sample = depth_values.to(device=device, dtype=torch.long)
        elif isinstance(depth_values, (list, tuple)):
            depth_per_sample = torch.tensor(
                [max(2, int(d)) for d in depth_values],
                device=device,
                dtype=torch.long,
            )
        else:
            depth_per_sample = torch.full(
                (batch_size,),
                int(max(2, int(depth_values))),
                device=device,
                dtype=torch.long,
            )
        if int(depth_per_sample.numel()) != batch_size:
            raise ValueError("depth_values must match h_list length")
        in_dims = torch.as_tensor(in_dims, device=device, dtype=torch.long)
        state_dims = torch.as_tensor(state_dims, device=device, dtype=torch.long)
        if input_mask is not None:
            input_mask = torch.as_tensor(input_mask, device=device, dtype=torch.float32)
            in_dims = input_mask.to(dtype=torch.long).sum(dim=1)
        hidden_state_dims = torch.tensor(
            [max(int(state_dims[bi].item()), int(h["prior_mlp_hidden_dim"])) for bi, h in enumerate(h_list)],
            device=device,
            dtype=torch.long,
        )
        hidden_reward_dims = torch.tensor(
            [max(1, int(h["prior_mlp_hidden_dim"])) for h in h_list],
            device=device,
            dtype=torch.long,
        )
        init_std = torch.tensor([float(h["init_std"]) for h in h_list], device=device, dtype=torch.float32)
        noise_std = torch.tensor([float(h["noise_std"]) for h in h_list], device=device, dtype=torch.float32)
        weight_cap_values = []
        for h in h_list:
            cap = self._resolve_lipschitz_weight_cap(h)
            weight_cap_values.append(float(cap) if cap is not None else float("inf"))
        weight_cap = torch.tensor(weight_cap_values, device=device, dtype=torch.float32)
        if isinstance(activation_names, str):
            activation_values = [self._activation_name(activation_names)] * batch_size
        else:
            activation_values = [self._activation_name(v) for v in activation_names]
        if len(activation_values) != batch_size:
            raise ValueError("activation_names must match h_list length")
        activation_codes = []
        for name in activation_values:
            if name == "relu":
                activation_codes.append(1)
            elif name == "identity":
                activation_codes.append(2)
            else:
                activation_codes.append(0)
        activation_codes = torch.tensor(activation_codes, device=device, dtype=torch.long)
        activation_mixed = not bool(torch.all(activation_codes == activation_codes[0]))
        activation_relu_mask = (activation_codes == 1).unsqueeze(1)
        activation_identity_mask = (activation_codes == 2).unsqueeze(1)
        activation_single = int(activation_codes[0].item())

        input_cap = int(input_mask.shape[1]) if input_mask is not None else int(in_dims.max().item())
        state_hidden_cap = int(hidden_state_dims.max().item())
        reward_hidden_cap = int(hidden_reward_dims.max().item())
        packed_branch_cap = int(max(state_hidden_cap, reward_hidden_cap, int(state_dims.max().item()), 1))
        state_cap = int(state_dims.max().item())
        packed_batch_size = int(batch_size * 2)

        first_in_mask = (
            input_mask.clone()
            if input_mask is not None
            else torch.zeros((batch_size, input_cap), device=device, dtype=torch.float32)
        )
        first_input_index = torch.zeros((batch_size, input_cap), device=device, dtype=torch.long)
        first_state_w = torch.zeros((batch_size, input_cap, state_hidden_cap), device=device, dtype=torch.float32)
        first_state_b = torch.zeros((batch_size, state_hidden_cap), device=device, dtype=torch.float32)
        first_reward_w = torch.zeros((batch_size, input_cap, reward_hidden_cap), device=device, dtype=torch.float32)
        first_reward_b = torch.zeros((batch_size, reward_hidden_cap), device=device, dtype=torch.float32)

        for bi in range(batch_size):
            if input_mask is not None:
                active_idx = torch.nonzero(input_mask[bi] > 0, as_tuple=False).squeeze(1)
                in_i = int(active_idx.numel())
            else:
                in_i = int(in_dims[bi].item())
                active_idx = torch.arange(in_i, device=device, dtype=torch.long)
                first_in_mask[bi, :in_i] = 1.0
            if in_i > 0:
                first_input_index[bi, :in_i] = active_idx
            out_i = int(hidden_state_dims[bi].item())
            if in_i <= 0 or out_i <= 0:
                continue
            w_b = torch.randn((in_i, out_i), device=device, dtype=torch.float32)
            b_b = torch.randn((out_i,), device=device, dtype=torch.float32)
            w_b = w_b * (init_std[bi] / math.sqrt(max(1, in_i)))
            w_b = self._project_matrix_fro_norm(w_b, float(weight_cap[bi].item()))
            first_state_w[bi, active_idx, :out_i] = w_b
            first_state_b[bi, :out_i] = b_b * (init_std[bi] * 0.1)

        for bi in range(batch_size):
            if input_mask is not None:
                active_idx = torch.nonzero(input_mask[bi] > 0, as_tuple=False).squeeze(1)
                in_i = int(active_idx.numel())
            else:
                in_i = int(in_dims[bi].item())
                active_idx = torch.arange(in_i, device=device, dtype=torch.long)
            out_i = int(hidden_reward_dims[bi].item())
            if in_i <= 0 or out_i <= 0:
                continue
            w_b = torch.randn((in_i, out_i), device=device, dtype=torch.float32)
            b_b = torch.randn((out_i,), device=device, dtype=torch.float32)
            w_b = w_b * (init_std[bi] / math.sqrt(max(1, in_i)))
            w_b = self._project_matrix_fro_norm(w_b, float(weight_cap[bi].item()))
            first_reward_w[bi, active_idx, :out_i] = w_b
            first_reward_b[bi, :out_i] = b_b * (init_std[bi] * 0.1)

        packed_activation_codes = torch.cat([activation_codes, activation_codes], dim=0)
        packed_activation_mixed = not bool(torch.all(packed_activation_codes == packed_activation_codes[0]))
        packed_activation_relu_mask = (packed_activation_codes == 1).unsqueeze(1)
        packed_activation_identity_mask = (packed_activation_codes == 2).unsqueeze(1)
        packed_activation_single = int(packed_activation_codes[0].item())

        packed_layers = []
        max_depth = int(depth_per_sample.max().item())
        for layer_idx in range(1, max_depth):
            layer_in_sizes = torch.zeros((packed_batch_size,), device=device, dtype=torch.long)
            layer_out_sizes = torch.zeros((packed_batch_size,), device=device, dtype=torch.long)
            layer_activation_codes = torch.full((packed_batch_size,), -1, device=device, dtype=torch.long)

            for bi in range(batch_size):
                depth_i = int(depth_per_sample[bi].item())
                state_i = int(state_dims[bi].item())
                state_hidden_i = int(hidden_state_dims[bi].item())
                reward_hidden_i = int(hidden_reward_dims[bi].item())
                hidden_out = layer_idx < (depth_i - 1)
                state_in_i = state_hidden_i if layer_idx <= (depth_i - 1) else state_i
                reward_in_i = reward_hidden_i if layer_idx <= (depth_i - 1) else 1
                state_out_i = state_hidden_i if hidden_out else state_i
                reward_out_i = reward_hidden_i if hidden_out else 1
                layer_in_sizes[bi] = state_in_i
                layer_out_sizes[bi] = state_out_i
                layer_in_sizes[batch_size + bi] = reward_in_i
                layer_out_sizes[batch_size + bi] = reward_out_i
                if bool(hidden_out):
                    layer_activation_codes[bi] = packed_activation_codes[bi]
                    layer_activation_codes[batch_size + bi] = packed_activation_codes[batch_size + bi]

            layer_in_cap = int(max(1, int(layer_in_sizes.max().item())))
            layer_out_cap = int(max(1, int(layer_out_sizes.max().item())))
            layer_w = torch.zeros((packed_batch_size, layer_in_cap, layer_out_cap), device=device, dtype=torch.float32)
            layer_b = torch.zeros((packed_batch_size, layer_out_cap), device=device, dtype=torch.float32)

            for bi in range(batch_size):
                depth_i = int(depth_per_sample[bi].item())
                state_row = bi
                state_in_i = int(layer_in_sizes[state_row].item())
                state_out_i = int(layer_out_sizes[state_row].item())
                post_layer = layer_idx > (depth_i - 1)
                if state_in_i <= 0 or state_out_i <= 0:
                    continue
                if post_layer:
                    d = min(state_in_i, state_out_i)
                    if d > 0:
                        eye_idx = torch.arange(d, device=device, dtype=torch.long)
                        layer_w[state_row, eye_idx, eye_idx] = 1.0
                    continue
                w_b = torch.randn((state_in_i, state_out_i), device=device, dtype=torch.float32)
                b_b = torch.randn((state_out_i,), device=device, dtype=torch.float32)
                w_b = w_b * (init_std[bi] / math.sqrt(max(1, state_in_i)))
                w_b = self._project_matrix_fro_norm(w_b, float(weight_cap[bi].item()))
                layer_w[state_row, :state_in_i, :state_out_i] = w_b
                layer_b[state_row, :state_out_i] = b_b * (init_std[bi] * 0.1)

            for bi in range(batch_size):
                depth_i = int(depth_per_sample[bi].item())
                reward_row = batch_size + bi
                reward_in_i = int(layer_in_sizes[reward_row].item())
                reward_out_i = int(layer_out_sizes[reward_row].item())
                post_layer = layer_idx > (depth_i - 1)
                if reward_in_i <= 0 or reward_out_i <= 0:
                    continue
                if post_layer:
                    d = min(reward_in_i, reward_out_i)
                    if d > 0:
                        eye_idx = torch.arange(d, device=device, dtype=torch.long)
                        layer_w[reward_row, eye_idx, eye_idx] = 1.0
                    continue
                w_b = torch.randn((reward_in_i, reward_out_i), device=device, dtype=torch.float32)
                b_b = torch.randn((reward_out_i,), device=device, dtype=torch.float32)
                w_b = w_b * (init_std[bi] / math.sqrt(max(1, reward_in_i)))
                w_b = self._project_matrix_fro_norm(w_b, float(weight_cap[bi].item()))
                layer_w[reward_row, :reward_in_i, :reward_out_i] = w_b
                layer_b[reward_row, :reward_out_i] = b_b * (init_std[bi] * 0.1)

            out_tile_batch, out_tile_offsets = self._build_active_tile_map(layer_out_sizes, 32)
            in_tile_batch, in_tile_offsets = self._build_active_tile_map(layer_in_sizes, 32)
            packed_layers.append(
                {
                    "w": layer_w,
                    "b": layer_b,
                    "in_sizes": layer_in_sizes,
                    "out_sizes": layer_out_sizes,
                    "in_cap": layer_in_cap,
                    "out_cap": layer_out_cap,
                    "out_tile_batch": out_tile_batch,
                    "out_tile_offsets": out_tile_offsets,
                    "in_tile_batch": in_tile_batch,
                    "in_tile_offsets": in_tile_offsets,
                    "activation_codes": layer_activation_codes,
                }
            )

        state_final_mask = torch.zeros((batch_size, state_cap), device=device, dtype=torch.float32)
        for bi in range(batch_size):
            state_final_mask[bi, :int(state_dims[bi].item())] = 1.0
        reward_final_mask = torch.ones((batch_size, 1), device=device, dtype=torch.float32)

        def _apply_activation(z, mask):
            if not bool(torch.any(mask)):
                return z
            mask = mask.unsqueeze(1)
            if not packed_activation_mixed:
                if packed_activation_single == 1:
                    z_act = torch.relu(z)
                elif packed_activation_single == 2:
                    z_act = z
                else:
                    z_act = torch.tanh(z)
            else:
                z_linear = z
                z_tanh = torch.tanh(z_linear)
                z_relu = torch.relu(z_linear)
                z_act = torch.where(packed_activation_relu_mask, z_relu, z_tanh)
                z_act = torch.where(packed_activation_identity_mask, z_linear, z_act)
            return torch.where(mask, z_act, z)

        def _core_fn(x_state, x_reward):
            state_hidden = self._batch_affine_with_layout(
                x_state[:, :input_cap] * first_in_mask,
                first_state_w,
                first_state_b,
                input_index=first_input_index,
                in_sizes=in_dims,
                out_sizes=hidden_state_dims,
            )
            reward_hidden = self._batch_affine_with_layout(
                x_reward[:, :input_cap] * first_in_mask,
                first_reward_w,
                first_reward_b,
                input_index=first_input_index,
                in_sizes=in_dims,
                out_sizes=hidden_reward_dims,
            )
            z = torch.zeros((packed_batch_size, packed_branch_cap), device=x_state.device, dtype=x_state.dtype)
            z[:batch_size, :state_hidden_cap] = state_hidden
            z[batch_size:, :reward_hidden_cap] = reward_hidden
            z = _apply_activation(z, torch.ones((packed_batch_size,), device=z.device, dtype=torch.bool))
            for layer in packed_layers:
                z = self._batch_affine_prefix_tiled(
                    z[:, :layer["in_cap"]],
                    layer["w"],
                    layer["b"],
                    in_sizes=layer["in_sizes"],
                    out_sizes=layer["out_sizes"],
                    activation_codes=layer["activation_codes"],
                    out_tile_batch=layer["out_tile_batch"],
                    out_tile_offsets=layer["out_tile_offsets"],
                    in_tile_batch=layer["in_tile_batch"],
                    in_tile_offsets=layer["in_tile_offsets"],
                )
            return z

        checkpoint_enabled = bool(self.envgen_checkpoint)

        def transition_fn(x, generators_for_noise=None, x_is_dual_packed=False):
            if bool(x_is_dual_packed):
                x_state_src = x[:batch_size]
                x_reward_src = x[batch_size: batch_size * 2]
            else:
                x_state_src = x
                x_reward_src = x
            if checkpoint_enabled and self._envgen_checkpoint_active(x_state_src):
                if bool(x_is_dual_packed):
                    x_state = x_state_src.clone()
                    x_reward = x_reward_src.clone()
                else:
                    shared_input = x_state_src.clone()
                    x_state = shared_input
                    x_reward = shared_input
                z = checkpoint(
                    _core_fn,
                    x_state,
                    x_reward,
                    use_reentrant=bool(self.envgen_checkpoint_reentrant),
                    preserve_rng_state=False,
                )
            else:
                z = _core_fn(x_state_src, x_reward_src)

            state_pre = z[:batch_size, :state_cap]
            reward_pre = z[batch_size:, :1]
            if torch.any(noise_std > 0):
                dual_noise = None
                if generators_for_noise is None:
                    dual_scale = torch.cat([noise_std, noise_std], dim=0)
                    dual_noise = self._sample_scaled_noise_batch(
                        None,
                        scale=dual_scale,
                        width=state_cap,
                        device=z.device,
                        dtype=z.dtype,
                    )
                else:
                    generators_list = list(generators_for_noise)
                    if len(generators_list) != batch_size:
                        raise ValueError("generators_for_noise must match batch size")
                    dual_scale = torch.cat([noise_std, noise_std], dim=0)
                    dual_noise = self._sample_scaled_noise_batch(
                        generators_list + generators_list,
                        scale=dual_scale,
                        width=state_cap,
                        device=z.device,
                        dtype=z.dtype,
                    )
                if dual_noise is not None:
                    state_pre = state_pre + dual_noise[:batch_size, :state_cap]
                    reward_pre = reward_pre + dual_noise[batch_size:, :1]

            state_out = torch.tanh(state_pre) * state_final_mask
            reward_out = torch.tanh(reward_pre) * reward_final_mask
            return state_out, reward_out

        transition_fn._envgen_checkpoint_enabled = checkpoint_enabled
        transition_fn._scm_hidden_fused_specialized = True
        return transition_fn

    def _build_scm_hetero_transition_batch_fn(
        self,
        in_dims,
        state_dims,
        h_list,
        device,
        depth_values,
        activation_names,
        input_mask=None,
    ):
        batch_size = int(len(h_list))
        if batch_size <= 0:
            return None
        in_dims = torch.as_tensor(in_dims, device=device, dtype=torch.long)
        state_dims = torch.as_tensor(state_dims, device=device, dtype=torch.long)
        if bool(self.fused_transition_scm_hidden_fused):
            return self._build_scm_hetero_hidden_fused_transition_fn(
                in_dims=in_dims,
                state_dims=state_dims,
                h_list=h_list,
                device=device,
                depth_values=depth_values,
                activation_names=activation_names,
                input_mask=input_mask,
            )
        reward_dims = torch.ones((batch_size,), device=device, dtype=torch.long)
        dual_in_dims = torch.cat([in_dims, in_dims], dim=0)
        dual_out_dims = torch.cat([state_dims, reward_dims], dim=0)
        if torch.is_tensor(depth_values):
            depth_values_dual = torch.cat(
                [
                    depth_values.to(device=device, dtype=torch.long),
                    depth_values.to(device=device, dtype=torch.long),
                ],
                dim=0,
            )
        elif isinstance(depth_values, (list, tuple)):
            depth_list = [max(2, int(d)) for d in depth_values]
            depth_values_dual = depth_list + depth_list
        else:
            depth_scalar = int(max(2, int(depth_values)))
            depth_values_dual = [depth_scalar] * (2 * batch_size)
        if isinstance(activation_names, str):
            activation_names_dual = [activation_names] * (2 * batch_size)
        else:
            activation_list = list(activation_names)
            if len(activation_list) != batch_size:
                raise ValueError("activation_names must match h_list length")
            activation_names_dual = activation_list + activation_list
        input_mask_dual = None
        if input_mask is not None:
            input_mask_t = torch.as_tensor(input_mask, device=device, dtype=torch.float32)
            input_mask_dual = torch.cat([input_mask_t, input_mask_t], dim=0)
        h_list_dual = list(h_list) + list(h_list)
        dual_fn = self._build_scm_hetero_batch_fn(
            in_dims=dual_in_dims,
            out_dims=dual_out_dims,
            h_list=h_list_dual,
            device=device,
            depth_values=depth_values_dual,
            activation_names=activation_names_dual,
            generators=None,
            input_mask=input_mask_dual,
        )
        state_cap = int(state_dims.max().item())

        def transition_fn(x, generators_for_noise=None, x_is_dual_packed=False):
            x_dual = x if bool(x_is_dual_packed) else torch.cat([x, x], dim=0)
            generators_dual = None
            if generators_for_noise is not None:
                generators_list = list(generators_for_noise)
                if len(generators_list) != batch_size:
                    raise ValueError("generators_for_noise must match batch size")
                generators_dual = generators_list + generators_list
            out_dual = dual_fn(
                x_dual,
                generators_for_noise=generators_dual,
                stable_input=True,
            )
            x_next = out_dual[:batch_size, :state_cap]
            reward_next = out_dual[batch_size:, :1]
            return x_next, reward_next

        transition_fn._gp_input_rff_fused = bool(getattr(dual_fn, "_gp_input_rff_fused", False))
        transition_fn._gp_output_projection_fused = bool(
            getattr(dual_fn, "_gp_output_projection_fused", False)
        )
        transition_fn._gp_rff_fused = bool(getattr(dual_fn, "_gp_rff_fused", False))
        if callable(getattr(dual_fn, "_consume_gp_projection_profile", None)):
            transition_fn._consume_gp_projection_profile = dual_fn._consume_gp_projection_profile

        return transition_fn

    def _build_gp_hetero_transition_batch_fn(self, in_dims, state_dims, h_list, device, input_mask=None):
        batch_size = int(len(h_list))
        if batch_size <= 0:
            return None
        in_dims = torch.as_tensor(in_dims, device=device, dtype=torch.long)
        state_dims = torch.as_tensor(state_dims, device=device, dtype=torch.long)
        reward_dims = torch.ones((batch_size,), device=device, dtype=torch.long)
        device_obj = device if isinstance(device, torch.device) else torch.device(str(device))
        gp_packed_env_input_enabled = bool(
            self.fused_transition_gp_packed_env_input
            and input_mask is not None
            and device_obj.type == "cuda"
        )
        if bool(self.fused_transition_gp_output_subgraph):
            return self._build_gp_hetero_output_subgraph_transition_fn(
                in_dims=in_dims,
                state_dims=state_dims,
                h_list=h_list,
                device=device,
                input_mask=input_mask,
            )
        dual_in_dims = torch.cat([in_dims, in_dims], dim=0)
        dual_out_dims = torch.cat([state_dims, reward_dims], dim=0)
        input_mask_dual = None
        packed_input_cap = int(max(1, int(in_dims.max().item())))
        if input_mask is not None:
            input_mask_t = torch.as_tensor(input_mask, device=device, dtype=torch.float32)
            input_mask_dual = torch.cat([input_mask_t, input_mask_t], dim=0)
        h_list_dual = list(h_list) + list(h_list)
        gp_shared_first_proj_enabled = bool(
            self.fused_transition_gp_shared_first_proj
            and device_obj.type == "cuda"
            and not gp_packed_env_input_enabled
        )
        cpu_rng_state_pre_legacy = None
        cuda_rng_state_pre_legacy = None
        if gp_shared_first_proj_enabled:
            cpu_rng_state_pre_legacy = torch.random.get_rng_state()
            cuda_rng_state_pre_legacy = torch.cuda.get_rng_state(device=device_obj)
        natural_dual_fn = self._build_gp_hetero_batch_fn(
            in_dims=dual_in_dims,
            out_dims=dual_out_dims,
            h_list=h_list_dual,
            device=device,
            generators=None,
            input_mask=input_mask_dual,
        )
        packed_dual_fn = None
        if gp_packed_env_input_enabled:
            packed_in_dims = torch.tensor(
                [sum(self._sample_dims(h)[:4]) for h in h_list],
                device=device,
                dtype=torch.long,
            )
            packed_input_cap = int(max(1, int(packed_in_dims.max().item())))
            packed_dual_in_dims = torch.cat([packed_in_dims, packed_in_dims], dim=0)
            packed_dual_fn = self._build_gp_hetero_batch_fn(
                in_dims=packed_dual_in_dims,
                out_dims=dual_out_dims,
                h_list=h_list_dual,
                device=device,
                generators=None,
                input_mask=None,
                sample_in_dims=dual_in_dims,
            )
        state_cap = int(state_dims.max().item())

        def legacy_transition_fn(x, generators_for_noise=None, x_is_dual_packed=False, x_input_is_packed=False):
            x_dual = x if bool(x_is_dual_packed) else torch.cat([x, x], dim=0)
            active_dual_fn = packed_dual_fn if (gp_packed_env_input_enabled and bool(x_input_is_packed)) else natural_dual_fn
            generators_dual = None
            if generators_for_noise is not None:
                generators_list = list(generators_for_noise)
                if len(generators_list) != batch_size:
                    raise ValueError("generators_for_noise must match batch size")
                generators_dual = generators_list + generators_list
            out_dual = active_dual_fn(
                x_dual,
                generators_for_noise=generators_dual,
                stable_input=True,
            )
            x_next = out_dual[:batch_size, :state_cap]
            reward_next = out_dual[batch_size:, :1]
            return x_next, reward_next

        shared_transition_fn = None
        if gp_shared_first_proj_enabled:
            cpu_rng_state_post_legacy = torch.random.get_rng_state()
            cuda_rng_state_post_legacy = torch.cuda.get_rng_state(device=device_obj)
            torch.random.set_rng_state(cpu_rng_state_pre_legacy)
            torch.cuda.set_rng_state(cuda_rng_state_pre_legacy, device=device_obj)
            shared_transition_fn = self._build_gp_hetero_shared_first_proj_transition_fn(
                in_dims=in_dims,
                state_dims=state_dims,
                h_list=h_list,
                device=device,
                input_mask=input_mask,
            )
            torch.random.set_rng_state(cpu_rng_state_post_legacy)
            torch.cuda.set_rng_state(cuda_rng_state_post_legacy, device=device_obj)

        if shared_transition_fn is None:
            transition_fn = legacy_transition_fn
        else:
            def transition_fn(x, generators_for_noise=None, x_is_dual_packed=False, x_input_is_packed=False):
                if bool(x_is_dual_packed) or bool(x_input_is_packed):
                    return legacy_transition_fn(
                        x,
                        generators_for_noise=generators_for_noise,
                        x_is_dual_packed=x_is_dual_packed,
                        x_input_is_packed=x_input_is_packed,
                    )
                return shared_transition_fn(
                    x,
                    generators_for_noise=generators_for_noise,
                    stable_input=True,
                )

        transition_fn._prefers_packed_env_input = bool((shared_transition_fn is None) and gp_packed_env_input_enabled)
        transition_fn._packed_input_cap = int(packed_input_cap)
        transition_fn._envgen_checkpoint_enabled = bool(
            getattr(natural_dual_fn, "_envgen_checkpoint_enabled", False)
            or getattr(packed_dual_fn, "_envgen_checkpoint_enabled", False)
            or getattr(shared_transition_fn, "_envgen_checkpoint_enabled", False)
        )
        transition_fn._gp_input_rff_fused = bool(
            getattr(natural_dual_fn, "_gp_input_rff_fused", False)
            or getattr(packed_dual_fn, "_gp_input_rff_fused", False)
        )
        transition_fn._gp_output_projection_fused = bool(
            getattr(natural_dual_fn, "_gp_output_projection_fused", False)
            or getattr(packed_dual_fn, "_gp_output_projection_fused", False)
        )
        transition_fn._gp_rff_fused = bool(
            getattr(natural_dual_fn, "_gp_rff_fused", False)
            or getattr(packed_dual_fn, "_gp_rff_fused", False)
        )
        transition_fn._gp_shared_first_proj_fused = bool(shared_transition_fn is not None)
        natural_profile_reader = getattr(natural_dual_fn, "_consume_gp_projection_profile", None)
        packed_profile_reader = getattr(packed_dual_fn, "_consume_gp_projection_profile", None)
        shared_profile_reader = getattr(shared_transition_fn, "_consume_gp_projection_profile", None)
        if callable(natural_profile_reader) or callable(packed_profile_reader) or callable(shared_profile_reader):
            def _consume_gp_projection_profile():
                stats = {
                    "first_projection_wall_s": 0.0,
                    "second_projection_wall_s": 0.0,
                    "call_count": 0,
                    "rff_fused_call_count": 0,
                    "shared_total_wall_s": 0.0,
                    "shared_core_wall_s": 0.0,
                    "shared_noise_wall_s": 0.0,
                    "shared_checkpoint_wall_s": 0.0,
                    "shared_post_wall_s": 0.0,
                    "shared_call_count": 0,
                }
                if callable(natural_profile_reader):
                    natural_stats = natural_profile_reader() or {}
                    stats["first_projection_wall_s"] += float(natural_stats.get("first_projection_wall_s", 0.0) or 0.0)
                    stats["second_projection_wall_s"] += float(natural_stats.get("second_projection_wall_s", 0.0) or 0.0)
                    stats["call_count"] += int(natural_stats.get("call_count", 0) or 0)
                    stats["rff_fused_call_count"] += int(natural_stats.get("rff_fused_call_count", 0) or 0)
                    stats["shared_total_wall_s"] += float(natural_stats.get("shared_total_wall_s", 0.0) or 0.0)
                    stats["shared_core_wall_s"] += float(natural_stats.get("shared_core_wall_s", 0.0) or 0.0)
                    stats["shared_noise_wall_s"] += float(natural_stats.get("shared_noise_wall_s", 0.0) or 0.0)
                    stats["shared_checkpoint_wall_s"] += float(
                        natural_stats.get("shared_checkpoint_wall_s", 0.0) or 0.0
                    )
                    stats["shared_post_wall_s"] += float(natural_stats.get("shared_post_wall_s", 0.0) or 0.0)
                    stats["shared_call_count"] += int(natural_stats.get("shared_call_count", 0) or 0)
                if callable(packed_profile_reader):
                    packed_stats = packed_profile_reader() or {}
                    stats["first_projection_wall_s"] += float(packed_stats.get("first_projection_wall_s", 0.0) or 0.0)
                    stats["second_projection_wall_s"] += float(packed_stats.get("second_projection_wall_s", 0.0) or 0.0)
                    stats["call_count"] += int(packed_stats.get("call_count", 0) or 0)
                    stats["rff_fused_call_count"] += int(packed_stats.get("rff_fused_call_count", 0) or 0)
                    stats["shared_total_wall_s"] += float(packed_stats.get("shared_total_wall_s", 0.0) or 0.0)
                    stats["shared_core_wall_s"] += float(packed_stats.get("shared_core_wall_s", 0.0) or 0.0)
                    stats["shared_noise_wall_s"] += float(packed_stats.get("shared_noise_wall_s", 0.0) or 0.0)
                    stats["shared_checkpoint_wall_s"] += float(
                        packed_stats.get("shared_checkpoint_wall_s", 0.0) or 0.0
                    )
                    stats["shared_post_wall_s"] += float(packed_stats.get("shared_post_wall_s", 0.0) or 0.0)
                    stats["shared_call_count"] += int(packed_stats.get("shared_call_count", 0) or 0)
                if callable(shared_profile_reader):
                    shared_stats = shared_profile_reader() or {}
                    stats["first_projection_wall_s"] += float(shared_stats.get("first_projection_wall_s", 0.0) or 0.0)
                    stats["second_projection_wall_s"] += float(shared_stats.get("second_projection_wall_s", 0.0) or 0.0)
                    stats["call_count"] += int(shared_stats.get("call_count", 0) or 0)
                    stats["rff_fused_call_count"] += int(shared_stats.get("rff_fused_call_count", 0) or 0)
                    stats["shared_total_wall_s"] += float(shared_stats.get("shared_total_wall_s", 0.0) or 0.0)
                    stats["shared_core_wall_s"] += float(shared_stats.get("shared_core_wall_s", 0.0) or 0.0)
                    stats["shared_noise_wall_s"] += float(shared_stats.get("shared_noise_wall_s", 0.0) or 0.0)
                    stats["shared_checkpoint_wall_s"] += float(
                        shared_stats.get("shared_checkpoint_wall_s", 0.0) or 0.0
                    )
                    stats["shared_post_wall_s"] += float(shared_stats.get("shared_post_wall_s", 0.0) or 0.0)
                    stats["shared_call_count"] += int(shared_stats.get("shared_call_count", 0) or 0)
                return stats

            transition_fn._consume_gp_projection_profile = _consume_gp_projection_profile

        return transition_fn

    def _build_paired_transition_fn(self, state_fn, reward_fn, batch_size, state_cap):
        batch_size = int(batch_size)
        state_cap = int(max(1, state_cap))
        state_out_width = int(max(1, int(getattr(state_fn, "_out_width", state_cap))))
        reward_out_width = int(max(1, int(getattr(reward_fn, "_out_width", 1))))
        state_noise_scale = getattr(state_fn, "_noise_scale", None)
        reward_noise_scale = getattr(reward_fn, "_noise_scale", None)
        checkpoint_enabled = bool(
            getattr(state_fn, "_envgen_checkpoint_enabled", False)
            or getattr(reward_fn, "_envgen_checkpoint_enabled", False)
        )

        def transition_fn(x, generators_for_noise=None, x_is_dual_packed=False):
            if bool(x_is_dual_packed):
                x_state = x[:batch_size]
                x_reward = x[batch_size: batch_size * 2]
                state_input = x_state
                reward_input = x_reward
                state_stable_input = True
                reward_stable_input = True
            else:
                x_state = x
                x_reward = x
                if checkpoint_enabled and self._envgen_checkpoint_active(x_state):
                    shared_input = x_state.clone()
                    state_input = shared_input
                    reward_input = shared_input
                    state_stable_input = True
                    reward_stable_input = True
                else:
                    state_input = x_state
                    reward_input = x_reward
                    state_stable_input = False
                    reward_stable_input = False
            state_noise_eps = None
            reward_noise_eps = None
            state_generators = generators_for_noise
            reward_generators = generators_for_noise
            if generators_for_noise is not None:
                generators_list = list(generators_for_noise)
                if len(generators_list) != batch_size:
                    raise ValueError("generators_for_noise must match batch size")
                dual_scale = None
                if torch.is_tensor(state_noise_scale) and torch.is_tensor(reward_noise_scale):
                    dual_scale = torch.cat([state_noise_scale, reward_noise_scale], dim=0)
                if dual_scale is not None and bool(torch.any(dual_scale > 0)):
                    dual_noise = self._sample_scaled_noise_batch(
                        generators_list + generators_list,
                        scale=dual_scale,
                        width=state_cap,
                        device=x_state.device,
                        dtype=x_state.dtype,
                    )
                    if dual_noise is not None:
                        state_noise_eps = dual_noise[:batch_size, :state_out_width]
                        reward_noise_eps = dual_noise[batch_size:, :reward_out_width]
                        state_generators = None
                        reward_generators = None
            x_next = state_fn(
                state_input,
                generators_for_noise=state_generators,
                noise_eps=state_noise_eps,
                stable_input=state_stable_input,
            )
            reward_next = reward_fn(
                reward_input,
                generators_for_noise=reward_generators,
                noise_eps=reward_noise_eps,
                stable_input=reward_stable_input,
            )
            return x_next[:, :state_cap], reward_next[:, :1]

        transition_fn._envgen_checkpoint_enabled = checkpoint_enabled
        return transition_fn

    def _build_gp_hetero_shared_first_proj_transition_fn(self, in_dims, state_dims, h_list, device, input_mask=None):
        batch_size = int(len(h_list))
        if batch_size <= 0:
            return None
        in_dims = torch.as_tensor(in_dims, device=device, dtype=torch.long)
        state_dims = torch.as_tensor(state_dims, device=device, dtype=torch.long)
        if input_mask is not None:
            input_mask = torch.as_tensor(input_mask, device=device, dtype=torch.float32)
            in_dims = input_mask.to(dtype=torch.long).sum(dim=1)
        reward_dims = torch.ones((batch_size,), device=device, dtype=torch.long)
        m_dims = torch.tensor([max(8, int(h["gp_rff_features"])) for h in h_list], device=device, dtype=torch.long)
        lengthscale = torch.tensor(
            [max(1e-6, float(h["lengthscale"])) for h in h_list],
            device=device,
            dtype=torch.float32,
        )
        outputscale = torch.tensor([float(h["outputscale"]) for h in h_list], device=device, dtype=torch.float32)
        noise = torch.tensor([float(h["noise"]) for h in h_list], device=device, dtype=torch.float32)
        weight_cap_values = []
        outputscale_cap_values = []
        for h in h_list:
            w_cap = self._resolve_lipschitz_weight_cap(h)
            s_cap = self._resolve_lipschitz_gp_outputscale_cap(h)
            weight_cap_values.append(float(w_cap) if w_cap is not None else float("inf"))
            outputscale_cap_values.append(float(s_cap) if s_cap is not None else float("inf"))
        weight_cap = torch.tensor(weight_cap_values, device=device, dtype=torch.float32)
        outputscale_cap = torch.tensor(outputscale_cap_values, device=device, dtype=torch.float32)
        outputscale = torch.sign(outputscale) * torch.minimum(outputscale.abs(), outputscale_cap)

        if input_mask is not None:
            in_cap = int(input_mask.shape[1])
            in_mask = input_mask.clone()
        else:
            in_cap = int(in_dims.max().item())
            in_mask = torch.zeros((batch_size, in_cap), device=device, dtype=torch.float32)
        m_cap = int(m_dims.max().item())
        dual_m_cap = int(2 * m_cap)
        state_cap = int(state_dims.max().item())
        dual_out_cap = int(state_cap + 1)
        w_dual = torch.zeros((batch_size, in_cap, dual_m_cap), device=device, dtype=torch.float32)
        b_dual = torch.zeros((batch_size, dual_m_cap), device=device, dtype=torch.float32)
        dual_m_mask = torch.zeros((batch_size, dual_m_cap), device=device, dtype=torch.float32)
        a_dual = torch.zeros((batch_size, dual_m_cap, dual_out_cap), device=device, dtype=torch.float32)
        state_out_mask = torch.zeros((batch_size, state_cap), device=device, dtype=torch.float32)
        reward_out_mask = torch.zeros((batch_size, 1), device=device, dtype=torch.float32)

        active_indices = []
        for bi in range(batch_size):
            if input_mask is not None:
                active_idx = torch.nonzero(input_mask[bi] > 0, as_tuple=False).squeeze(1)
                in_i = int(active_idx.numel())
            else:
                in_i = int(in_dims[bi].item())
                active_idx = torch.arange(in_i, device=device, dtype=torch.long)
                if in_i > 0:
                    in_mask[bi, :in_i] = 1.0
            active_indices.append((active_idx, in_i))

        def _sample_branch_params(in_i, m_i, out_i):
            w_b = torch.randn((in_i, m_i), device=device, dtype=torch.float32)
            b_b = torch.rand((m_i,), device=device, dtype=torch.float32)
            a_b = torch.randn((m_i, out_i), device=device, dtype=torch.float32)
            return w_b, b_b, a_b

        # Preserve legacy RNG order: state branch params for all samples, then reward branch params.
        for bi in range(batch_size):
            active_idx, in_i = active_indices[bi]
            m_i = int(m_dims[bi].item())
            out_i = int(state_dims[bi].item())
            if in_i <= 0 or m_i <= 0 or out_i <= 0:
                continue
            w_b, b_b, a_b = _sample_branch_params(in_i, m_i, out_i)
            w_b = w_b / lengthscale[bi]
            w_b = self._project_matrix_fro_norm(w_b, float(weight_cap[bi].item()))
            a_b = a_b / math.sqrt(max(1, m_i))
            a_b = self._project_matrix_fro_norm(a_b, float(weight_cap[bi].item()))
            w_dual[bi, active_idx, :m_i] = w_b
            b_dual[bi, :m_i] = 2.0 * math.pi * b_b
            dual_m_mask[bi, :m_i] = 1.0
            a_dual[bi, :m_i, :out_i] = a_b
            state_out_mask[bi, :out_i] = 1.0

        for bi in range(batch_size):
            active_idx, in_i = active_indices[bi]
            m_i = int(m_dims[bi].item())
            out_i = int(reward_dims[bi].item())
            if in_i <= 0 or m_i <= 0 or out_i <= 0:
                continue
            w_b, b_b, a_b = _sample_branch_params(in_i, m_i, out_i)
            w_b = w_b / lengthscale[bi]
            w_b = self._project_matrix_fro_norm(w_b, float(weight_cap[bi].item()))
            a_b = a_b / math.sqrt(max(1, m_i))
            a_b = self._project_matrix_fro_norm(a_b, float(weight_cap[bi].item()))
            w_dual[bi, active_idx, m_cap : m_cap + m_i] = w_b
            b_dual[bi, m_cap : m_cap + m_i] = 2.0 * math.pi * b_b
            dual_m_mask[bi, m_cap : m_cap + m_i] = 1.0
            a_dual[bi, m_cap : m_cap + m_i, state_cap : state_cap + out_i] = a_b
            reward_out_mask[bi, :out_i] = 1.0

        gp_projection_profile = {
            "first_projection_wall_s": 0.0,
            "second_projection_wall_s": 0.0,
            "call_count": 0,
            "rff_fused_call_count": 0,
            "shared_total_wall_s": 0.0,
            "shared_core_wall_s": 0.0,
            "shared_noise_wall_s": 0.0,
            "shared_checkpoint_wall_s": 0.0,
            "shared_post_wall_s": 0.0,
            "shared_call_count": 0,
        }

        def _consume_gp_projection_profile():
            stats = dict(gp_projection_profile)
            for key in gp_projection_profile:
                gp_projection_profile[key] = 0.0 if "wall_s" in key else 0
            return stats

        def _core_fn(x):
            core_t0 = time.perf_counter() if self.profile_gp_projection_timing else None
            x_in = x[:, :in_cap] * in_mask
            first_proj_t0 = time.perf_counter() if self.profile_gp_projection_timing else None
            phi_dual = torch.cos(self._batch_affine(x_in, w_dual, b_dual)) * dual_m_mask
            if first_proj_t0 is not None:
                gp_projection_profile["first_projection_wall_s"] += (time.perf_counter() - first_proj_t0)
                gp_projection_profile["call_count"] += 1
            second_proj_t0 = time.perf_counter() if self.profile_gp_projection_timing else None
            y_dual = outputscale[:, None] * self._batch_affine(phi_dual, a_dual, None)
            if second_proj_t0 is not None:
                gp_projection_profile["second_projection_wall_s"] += (time.perf_counter() - second_proj_t0)
            if core_t0 is not None:
                gp_projection_profile["shared_core_wall_s"] += (time.perf_counter() - core_t0)
            state_y = y_dual[:, :state_cap]
            reward_y = y_dual[:, state_cap : state_cap + 1]
            return state_y * state_out_mask, reward_y * reward_out_mask

        def _sample_noise_pair(generators_for_noise, *, dtype, device_obj):
            if not bool(torch.any(noise > 0)):
                return None, None
            if generators_for_noise is None:
                state_eps = torch.randn((batch_size, state_cap), device=device_obj, dtype=dtype) * noise[:, None]
                reward_eps = torch.randn((batch_size, state_cap), device=device_obj, dtype=dtype) * noise[:, None]
                return state_eps, reward_eps[:, :1]
            generators_list = list(generators_for_noise)
            if len(generators_list) != batch_size:
                raise ValueError("generators_for_noise must match batch size")
            state_eps = torch.zeros((batch_size, state_cap), device=device_obj, dtype=dtype)
            reward_eps = torch.zeros((batch_size, 1), device=device_obj, dtype=dtype)
            for bi in range(batch_size):
                if float(noise[bi].item()) <= 0.0:
                    continue
                g = generators_list[bi]
                if g is None:
                    e_state = torch.randn((state_cap,), device=device_obj, dtype=dtype)
                    e_reward_full = torch.randn((state_cap,), device=device_obj, dtype=dtype)
                else:
                    e_state = torch.randn((state_cap,), device=device_obj, dtype=dtype, generator=g)
                    e_reward_full = torch.randn((state_cap,), device=device_obj, dtype=dtype, generator=g)
                state_eps[bi] = e_state * noise[bi]
                reward_eps[bi, 0] = e_reward_full[0] * noise[bi]
            return state_eps, reward_eps

        def fn(x, generators_for_noise=None, stable_input=False):
            total_t0 = time.perf_counter() if self.profile_gp_projection_timing else None
            checkpoint_t0 = time.perf_counter() if self.profile_gp_projection_timing else None
            if self._envgen_checkpoint_active(x):
                x_checkpoint = x if bool(stable_input) else x.clone()
                state_y, reward_y = checkpoint(
                    _core_fn,
                    x_checkpoint,
                    use_reentrant=bool(self.envgen_checkpoint_reentrant),
                    preserve_rng_state=False,
                )
            else:
                state_y, reward_y = _core_fn(x)
            if checkpoint_t0 is not None:
                gp_projection_profile["shared_checkpoint_wall_s"] += (time.perf_counter() - checkpoint_t0)
            noise_t0 = time.perf_counter() if self.profile_gp_projection_timing else None
            state_noise_eps, reward_noise_eps = _sample_noise_pair(
                generators_for_noise,
                dtype=state_y.dtype,
                device_obj=state_y.device,
            )
            if state_noise_eps is not None:
                state_y = state_y + state_noise_eps
            if reward_noise_eps is not None:
                reward_y = reward_y + reward_noise_eps
            if noise_t0 is not None:
                gp_projection_profile["shared_noise_wall_s"] += (time.perf_counter() - noise_t0)
            post_t0 = time.perf_counter() if self.profile_gp_projection_timing else None
            state_y = torch.tanh(state_y)
            reward_y = torch.tanh(reward_y)
            if post_t0 is not None:
                gp_projection_profile["shared_post_wall_s"] += (time.perf_counter() - post_t0)
            if total_t0 is not None:
                gp_projection_profile["shared_total_wall_s"] += (time.perf_counter() - total_t0)
                gp_projection_profile["shared_call_count"] += 1
            return state_y * state_out_mask, reward_y * reward_out_mask

        fn._envgen_checkpoint_enabled = bool(self.envgen_checkpoint)
        fn._consume_gp_projection_profile = _consume_gp_projection_profile
        fn._gp_shared_first_proj_fused = True
        return fn

    def _sample_environment_family_coarse_batch(
        self,
        h_list,
        device,
        rng_seeds=None,
        *,
        build_x_generator=True,
        build_y_generator=True,
        build_policy_generator=True,
        prefer_transition_only=False,
        preserve_skipped_generator_rng=True,
    ):
        if not h_list:
            raise ValueError("h_list must be non-empty")
        family = self._normalize_family(h_list[0].get("family", "scm"))
        for h in h_list[1:]:
            if self._normalize_family(h.get("family", "scm")) != family:
                raise ValueError("family-coarse batch expects same family")

        batch_size = len(h_list)
        if rng_seeds is not None and len(rng_seeds) != batch_size:
            raise ValueError("rng_seeds must match h_list length")
        generators = None
        if rng_seeds is not None:
            generators = []
            for s in rng_seeds:
                g = torch.Generator(device=device)
                g.manual_seed(int(s))
                generators.append(g)
        family_build_wall_t0 = time.perf_counter()
        generator_build_wall_s = 0.0
        transition_generator_build_wall_s = 0.0
        gp_shared_transition_build_wall_s = 0.0

        dims = [self._sample_dims(h) for h in h_list]
        state_dims = torch.tensor([d[0] for d in dims], device=device, dtype=torch.long)
        obs_dims = torch.tensor([d[1] for d in dims], device=device, dtype=torch.long)
        action_dims = torch.tensor([d[2] for d in dims], device=device, dtype=torch.long)
        noise_dims = torch.tensor([d[3] for d in dims], device=device, dtype=torch.long)
        zero_pad_dims = torch.tensor([d[4] for d in dims], device=device, dtype=torch.long)

        in_dims = state_dims + obs_dims + action_dims + noise_dims + zero_pad_dims
        max_state_dim = int(state_dims.max().item())
        max_obs_dim = int(obs_dims.max().item())
        max_action_dim = int(action_dims.max().item())
        max_noise_dim = int(noise_dims.max().item())
        max_zero_pad_dim = int(zero_pad_dims.max().item())
        in_cap = int(max_state_dim + max_obs_dim + max_action_dim + max_noise_dim + max_zero_pad_dim)
        input_mask = torch.zeros((batch_size, in_cap), device=device, dtype=torch.float32)
        obs_start = max_state_dim
        action_start = max_state_dim + max_obs_dim
        noise_start = action_start + max_action_dim
        zero_start = noise_start + max_noise_dim
        for bi in range(batch_size):
            s_i = int(state_dims[bi].item())
            o_i = int(obs_dims[bi].item())
            a_i = int(action_dims[bi].item())
            n_i = int(noise_dims[bi].item())
            z_i = int(zero_pad_dims[bi].item())
            if s_i > 0:
                input_mask[bi, :s_i] = 1.0
            if o_i > 0:
                input_mask[bi, obs_start: obs_start + o_i] = 1.0
            if a_i > 0:
                input_mask[bi, action_start: action_start + a_i] = 1.0
            if n_i > 0:
                input_mask[bi, noise_start: noise_start + n_i] = 1.0
            if z_i > 0:
                input_mask[bi, zero_start: zero_start + z_i] = 1.0

        device_obj = device if isinstance(device, torch.device) else torch.device(str(device))
        enable_fused_transition = bool(
            self.fused_transition_generator
            and device_obj.type == "cuda"
            and generators is None
        )
        transition_only_build_enabled = bool(prefer_transition_only and enable_fused_transition)
        build_x_generator = bool(build_x_generator) and not transition_only_build_enabled
        build_y_generator = bool(build_y_generator) and not transition_only_build_enabled
        build_policy_generator = bool(build_policy_generator) and not transition_only_build_enabled
        preserve_skipped_generator_rng = bool(preserve_skipped_generator_rng)
        transition_generator = None
        x_generator = None
        y_generator = None
        policy_generator = None
        skipped_non_transition_generator_count = 0
        skipped_non_transition_rng_preserve_count = 0

        def _preserve_skipped_rng(build_needed, out_dims):
            nonlocal skipped_non_transition_generator_count
            nonlocal skipped_non_transition_rng_preserve_count
            if bool(build_needed):
                return
            skipped_non_transition_generator_count += 1
            if not preserve_skipped_generator_rng:
                return
            skipped_non_transition_rng_preserve_count += 1
            if family == "scm":
                self._consume_scm_hetero_batch_init_rng(
                    in_dims=in_dims,
                    out_dims=out_dims,
                    h_list=h_list,
                    device=device,
                    depth_values=depth_values,
                    generators=generators,
                    input_mask=input_mask,
                )
            else:
                self._consume_gp_hetero_batch_init_rng(
                    in_dims=in_dims,
                    out_dims=out_dims,
                    h_list=h_list,
                    device=device,
                    generators=generators,
                    input_mask=input_mask,
                )

        if family == "scm":
            depth_values = [max(2, int(h["num_layers"])) for h in h_list]
            activation_values = [self._activation_name(h["prior_mlp_activations"]) for h in h_list]
            if enable_fused_transition:
                build_t0 = time.perf_counter()
                transition_generator = self._build_scm_hetero_transition_batch_fn(
                    in_dims=in_dims,
                    state_dims=state_dims,
                    h_list=h_list,
                    device=device,
                    depth_values=depth_values,
                    activation_names=activation_values,
                    input_mask=input_mask,
                )
                transition_generator_build_wall_s += (time.perf_counter() - build_t0)
            if build_x_generator:
                build_t0 = time.perf_counter()
                x_generator = self._build_scm_hetero_batch_fn(
                    in_dims=in_dims,
                    out_dims=state_dims,
                    h_list=h_list,
                    device=device,
                    depth_values=depth_values,
                    activation_names=activation_values,
                    generators=generators,
                    input_mask=input_mask,
                )
                generator_build_wall_s += (time.perf_counter() - build_t0)
            else:
                _preserve_skipped_rng(build_x_generator, state_dims)
            if build_y_generator:
                build_t0 = time.perf_counter()
                y_generator = self._build_scm_hetero_batch_fn(
                    in_dims=in_dims,
                    out_dims=torch.ones((batch_size,), device=device, dtype=torch.long),
                    h_list=h_list,
                    device=device,
                    depth_values=depth_values,
                    activation_names=activation_values,
                    generators=generators,
                    input_mask=input_mask,
                )
                generator_build_wall_s += (time.perf_counter() - build_t0)
            else:
                _preserve_skipped_rng(
                    build_y_generator,
                    torch.ones((batch_size,), device=device, dtype=torch.long),
                )
            if build_policy_generator:
                build_t0 = time.perf_counter()
                policy_generator = self._build_scm_hetero_batch_fn(
                    in_dims=in_dims,
                    out_dims=action_dims,
                    h_list=h_list,
                    device=device,
                    depth_values=depth_values,
                    activation_names=activation_values,
                    generators=generators,
                    input_mask=input_mask,
                )
                generator_build_wall_s += (time.perf_counter() - build_t0)
            else:
                _preserve_skipped_rng(build_policy_generator, action_dims)
        else:
            if enable_fused_transition:
                build_t0 = time.perf_counter()
                transition_generator = self._build_gp_hetero_transition_batch_fn(
                    in_dims=in_dims,
                    state_dims=state_dims,
                    h_list=h_list,
                    device=device,
                    input_mask=input_mask,
                )
                build_dt = (time.perf_counter() - build_t0)
                transition_generator_build_wall_s += build_dt
                if bool(getattr(transition_generator, "_gp_shared_first_proj_fused", False)):
                    gp_shared_transition_build_wall_s += build_dt
            if build_x_generator:
                build_t0 = time.perf_counter()
                x_generator = self._build_gp_hetero_batch_fn(
                    in_dims=in_dims,
                    out_dims=state_dims,
                    h_list=h_list,
                    device=device,
                    generators=generators,
                    input_mask=input_mask,
                )
                generator_build_wall_s += (time.perf_counter() - build_t0)
            else:
                _preserve_skipped_rng(build_x_generator, state_dims)
            if build_y_generator:
                build_t0 = time.perf_counter()
                y_generator = self._build_gp_hetero_batch_fn(
                    in_dims=in_dims,
                    out_dims=torch.ones((batch_size,), device=device, dtype=torch.long),
                    h_list=h_list,
                    device=device,
                    generators=generators,
                    input_mask=input_mask,
                )
                generator_build_wall_s += (time.perf_counter() - build_t0)
            else:
                _preserve_skipped_rng(
                    build_y_generator,
                    torch.ones((batch_size,), device=device, dtype=torch.long),
                )
            if build_policy_generator:
                build_t0 = time.perf_counter()
                policy_generator = self._build_gp_hetero_batch_fn(
                    in_dims=in_dims,
                    out_dims=action_dims,
                    h_list=h_list,
                    device=device,
                    generators=generators,
                    input_mask=input_mask,
                )
                generator_build_wall_s += (time.perf_counter() - build_t0)
            else:
                _preserve_skipped_rng(build_policy_generator, action_dims)

        alpha = torch.tensor(
            [float(max(1e-4, min(1.0, float(h["alpha"])))) for h in h_list],
            device=device,
            dtype=torch.float32,
        )
        init_state_std = torch.tensor([float(h["init_state_std"]) for h in h_list], device=device, dtype=torch.float32)
        init_action_std = torch.tensor([float(h["init_action_std"]) for h in h_list], device=device, dtype=torch.float32)
        state_noise_std = torch.tensor([float(h["state_noise_std"]) for h in h_list], device=device, dtype=torch.float32)
        action_noise_train_std = torch.tensor(
            [float(h["action_noise_train_std"]) for h in h_list],
            device=device,
            dtype=torch.float32,
        )
        action_noise_eval_std = torch.tensor(
            [float(h["action_noise_eval_std"]) for h in h_list],
            device=device,
            dtype=torch.float32,
        )
        reward_scale = torch.tensor([float(h["reward_scale"]) for h in h_list], device=device, dtype=torch.float32)
        reward_clip = torch.tensor(
            [float(max(0.1, self._resolve_scalar(h.get("reward_clip", 10.0)))) for h in h_list],
            device=device,
            dtype=torch.float32,
        )
        state_clip = torch.tensor(
            [float(max(1.0, self._resolve_scalar(h.get("state_clip", 8.0)))) for h in h_list],
            device=device,
            dtype=torch.float32,
        )
        state_highway_enabled = torch.tensor(
            [bool(self._resolve_state_highway_enabled(h)) for h in h_list],
            device=device,
            dtype=torch.bool,
        )
        state_highway_lambda = torch.tensor(
            [float(self._resolve_state_highway_lambda(h)) for h in h_list],
            device=device,
            dtype=torch.float32,
        )
        aev4_cfg = self._resolve_aev4_config()
        aev4_enabled = torch.full(
            (batch_size,),
            bool(aev4_cfg.get("enabled", False)),
            device=device,
            dtype=torch.bool,
        )
        aev4_highway_ratio = torch.full(
            (batch_size,),
            float(aev4_cfg.get("highway_ratio", 0.25)),
            device=device,
            dtype=torch.float32,
        )
        aev4_update_scale = torch.full(
            (batch_size,),
            float(aev4_cfg.get("update_scale", 0.12)),
            device=device,
            dtype=torch.float32,
        )
        aev4_update_clip = torch.full(
            (batch_size,),
            float(aev4_cfg.get("update_clip", 0.0)),
            device=device,
            dtype=torch.float32,
        )
        reward_dropout_enabled = torch.tensor(
            [bool(h.get("reward_dropout_enabled", True)) for h in h_list],
            device=device,
            dtype=torch.bool,
        )
        reward_dropout_impute_zero = torch.tensor(
            [bool(h.get("reward_dropout_impute_zero", True)) for h in h_list],
            device=device,
            dtype=torch.bool,
        )
        reward_dropout_ratio = torch.tensor(
            [float(max(0.0, min(1.0, self._sample_reward_dropout_ratio(h)))) for h in h_list],
            device=device,
            dtype=torch.float32,
        )
        obs_slot_dims = torch.tensor(
            [int(max(1, h.get("obs_slot_dim", 400))) for h in h_list],
            device=device,
            dtype=torch.long,
        )
        action_slot_dims = torch.tensor(
            [int(max(1, h.get("action_slot_dim", 30))) for h in h_list],
            device=device,
            dtype=torch.long,
        )

        env = {
            "family": family,
            "state_dim": int(max_state_dim),
            "obs_dim": int(max_obs_dim),
            "action_dim": int(max_action_dim),
            "noise_dim": int(max_noise_dim),
            "zero_pad_dim": int(max_zero_pad_dim),
            "state_dim_per_sample": state_dims,
            "obs_dim_per_sample": obs_dims,
            "action_dim_per_sample": action_dims,
            "noise_dim_per_sample": noise_dims,
            "zero_pad_dim_per_sample": zero_pad_dims,
            "obs_slot_dim_per_sample": obs_slot_dims,
            "action_slot_dim_per_sample": action_slot_dims,
            "obs_slot_dim": int(obs_slot_dims.max().item()),
            "action_slot_dim": int(action_slot_dims.max().item()),
            "x_generator": x_generator,
            "y_generator": y_generator,
            "transition_generator": transition_generator,
            "policy_generator": policy_generator,
            "_build_profile": {
                "family_build_wall_s": float(time.perf_counter() - family_build_wall_t0),
                "generator_build_wall_s": float(generator_build_wall_s),
                "transition_generator_build_wall_s": float(transition_generator_build_wall_s),
                "gp_shared_transition_build_wall_s": float(gp_shared_transition_build_wall_s),
                "transition_only_build_enabled": int(transition_only_build_enabled),
                "skipped_non_transition_generator_count": int(skipped_non_transition_generator_count),
                "skipped_non_transition_rng_preserve_count": int(
                    skipped_non_transition_rng_preserve_count
                ),
                "non_transition_generator_build_count": int(
                    int(x_generator is not None)
                    + int(y_generator is not None)
                    + int(policy_generator is not None)
                ),
                "non_transition_generator_skip_count": int(
                    3
                    - int(x_generator is not None)
                    - int(y_generator is not None)
                    - int(policy_generator is not None)
                ),
            },
            "alpha": alpha,
            "init_state_std": init_state_std,
            "init_action_std": init_action_std,
            "state_noise_std": state_noise_std,
            "action_noise_train_std": action_noise_train_std,
            "action_noise_eval_std": action_noise_eval_std,
            "reward_scale": reward_scale,
            "reward_clip": reward_clip,
            "state_clip": state_clip,
            "state_highway_enabled": state_highway_enabled,
            "state_highway_lambda": state_highway_lambda,
            "aev4_enabled": aev4_enabled,
            "aev4_highway_ratio": aev4_highway_ratio,
            "aev4_update_scale": aev4_update_scale,
            "aev4_update_clip": aev4_update_clip,
            "reward_dropout_enabled": reward_dropout_enabled,
            "reward_dropout_impute_zero": reward_dropout_impute_zero,
            "reward_dropout_ratio": reward_dropout_ratio,
        }
        return env

    def _build_scm_batch_fn(self, in_dim, out_dim, h_list, device, generators=None):
        batch_size = len(h_list)
        depth = max(2, int(h_list[0]["num_layers"]))
        hidden = max(int(out_dim), int(h_list[0]["prior_mlp_hidden_dim"]))
        activation = self._resolve_activation(h_list[0]["prior_mlp_activations"])

        init_std = torch.tensor(
            [float(h["init_std"]) for h in h_list],
            device=device,
            dtype=torch.float32,
        )
        noise_std = torch.tensor(
            [float(h["noise_std"]) for h in h_list],
            device=device,
            dtype=torch.float32,
        )
        weight_cap_values = []
        for h in h_list:
            cap = self._resolve_lipschitz_weight_cap(h)
            weight_cap_values.append(float(cap) if cap is not None else float("inf"))
        weight_cap = torch.tensor(weight_cap_values, device=device, dtype=torch.float32)

        layer_dims = [in_dim] + [hidden] * (depth - 1) + [out_dim]
        weights = []
        biases = []
        if generators is None:
            for d_in, d_out in zip(layer_dims[:-1], layer_dims[1:]):
                scale = init_std[:, None, None] / math.sqrt(max(1, d_in))
                w = torch.randn((batch_size, d_in, d_out), device=device, dtype=torch.float32) * scale
                w = self._project_matrix_fro_norm(w, weight_cap)
                b = torch.randn((batch_size, d_out), device=device, dtype=torch.float32) * (init_std[:, None] * 0.1)
                weights.append(w)
                biases.append(b)
        else:
            for d_in, d_out in zip(layer_dims[:-1], layer_dims[1:]):
                w = torch.empty((batch_size, d_in, d_out), device=device, dtype=torch.float32)
                b = torch.empty((batch_size, d_out), device=device, dtype=torch.float32)
                for bi in range(batch_size):
                    g = generators[bi]
                    w_b = torch.randn((d_in, d_out), device=device, dtype=torch.float32, generator=g)
                    b_b = torch.randn((d_out,), device=device, dtype=torch.float32, generator=g)
                    w_b = w_b * (init_std[bi] / math.sqrt(max(1, d_in)))
                    w[bi] = self._project_matrix_fro_norm(w_b, float(weight_cap[bi].item()))
                    b[bi] = b_b * (init_std[bi] * 0.1)
                weights.append(w)
                biases.append(b)

        def fn(x, generators_for_noise=None):
            z = x
            for i, (w, b) in enumerate(zip(weights, biases)):
                z = self._batch_affine(z, w, b)
                if i < len(weights) - 1:
                    z = activation(z)
            if torch.any(noise_std > 0):
                if generators_for_noise is None:
                    z = z + torch.randn_like(z) * noise_std[:, None]
                else:
                    noise = torch.empty_like(z)
                    for bi in range(batch_size):
                        if noise_std[bi] <= 0:
                            noise[bi].zero_()
                            continue
                        g = generators_for_noise[bi]
                        if g is None:
                            noise_b = torch.randn((z.shape[1],), device=z.device, dtype=z.dtype)
                        else:
                            noise_b = torch.randn((z.shape[1],), device=z.device, dtype=z.dtype, generator=g)
                        noise[bi] = noise_b * noise_std[bi]
                    z = z + noise
            return torch.tanh(z)

        return fn

    def _build_gp_batch_fn(self, in_dim, out_dim, h_list, device, generators=None):
        batch_size = len(h_list)
        m = max(8, int(h_list[0]["gp_rff_features"]))
        lengthscale = torch.tensor(
            [max(1e-6, float(h["lengthscale"])) for h in h_list],
            device=device,
            dtype=torch.float32,
        )
        outputscale = torch.tensor(
            [float(h["outputscale"]) for h in h_list],
            device=device,
            dtype=torch.float32,
        )
        noise = torch.tensor(
            [float(h["noise"]) for h in h_list],
            device=device,
            dtype=torch.float32,
        )
        weight_cap_values = []
        outputscale_cap_values = []
        for h in h_list:
            w_cap = self._resolve_lipschitz_weight_cap(h)
            s_cap = self._resolve_lipschitz_gp_outputscale_cap(h)
            weight_cap_values.append(float(w_cap) if w_cap is not None else float("inf"))
            outputscale_cap_values.append(float(s_cap) if s_cap is not None else float("inf"))
        weight_cap = torch.tensor(weight_cap_values, device=device, dtype=torch.float32)
        outputscale_cap = torch.tensor(outputscale_cap_values, device=device, dtype=torch.float32)
        outputscale = torch.sign(outputscale) * torch.minimum(outputscale.abs(), outputscale_cap)
        if generators is None:
            w = torch.randn((batch_size, in_dim, m), device=device, dtype=torch.float32) / lengthscale[:, None, None]
            b = 2.0 * math.pi * torch.rand((batch_size, m), device=device, dtype=torch.float32)
            a = torch.randn((batch_size, m, out_dim), device=device, dtype=torch.float32) / math.sqrt(max(1, m))
            w = self._project_matrix_fro_norm(w, weight_cap)
            a = self._project_matrix_fro_norm(a, weight_cap)
        else:
            w = torch.empty((batch_size, in_dim, m), device=device, dtype=torch.float32)
            b = torch.empty((batch_size, m), device=device, dtype=torch.float32)
            a = torch.empty((batch_size, m, out_dim), device=device, dtype=torch.float32)
            for bi in range(batch_size):
                g = generators[bi]
                w_b = torch.randn((in_dim, m), device=device, dtype=torch.float32, generator=g)
                b_b = torch.rand((m,), device=device, dtype=torch.float32, generator=g)
                a_b = torch.randn((m, out_dim), device=device, dtype=torch.float32, generator=g)
                w_b = w_b / lengthscale[bi]
                w[bi] = self._project_matrix_fro_norm(w_b, float(weight_cap[bi].item()))
                b[bi] = 2.0 * math.pi * b_b
                a_b = a_b / math.sqrt(max(1, m))
                a[bi] = self._project_matrix_fro_norm(a_b, float(weight_cap[bi].item()))

        def fn(x, generators_for_noise=None):
            phi = torch.cos(self._batch_affine(x, w, b))
            y = outputscale[:, None] * self._batch_affine(phi, a, None)
            if torch.any(noise > 0):
                if generators_for_noise is None:
                    y = y + torch.randn_like(y) * noise[:, None]
                else:
                    eps = torch.empty_like(y)
                    for bi in range(batch_size):
                        if noise[bi] <= 0:
                            eps[bi].zero_()
                            continue
                        g = generators_for_noise[bi]
                        if g is None:
                            e_b = torch.randn((y.shape[1],), device=y.device, dtype=y.dtype)
                        else:
                            e_b = torch.randn((y.shape[1],), device=y.device, dtype=y.dtype, generator=g)
                        eps[bi] = e_b * noise[bi]
                    y = y + eps
            return torch.tanh(y)

        return fn

    def _sample_environment_batch(self, h_list, device, rng_seeds=None):
        if not h_list:
            raise ValueError("h_list must be non-empty for vectorized rollout")
        batch_size = int(len(h_list))
        schema_sig = self._environment_structure_signature(h_list[0])
        for h in h_list[1:]:
            if self._environment_structure_signature(h) != schema_sig:
                raise ValueError("h_list must be structurally homogeneous for batch vectorization")

        family = str(h_list[0]["family"]).lower()
        if family not in {"scm", "gp"}:
            family = "scm"

        state_dim, obs_dim, action_dim, noise_dim, zero_pad_dim = self._sample_dims(h_list[0])
        in_dim = int(state_dim + obs_dim + action_dim + noise_dim + zero_pad_dim)
        builder = self._build_scm_batch_fn if family == "scm" else self._build_gp_batch_fn
        generators = None
        if rng_seeds is not None:
            if len(rng_seeds) != len(h_list):
                raise ValueError("rng_seeds must match h_list length")
            generators = []
            for s in rng_seeds:
                g = torch.Generator(device=device)
                g.manual_seed(int(s))
                generators.append(g)

        x_generator = builder(in_dim, state_dim, h_list, device, generators=generators)
        y_generator = builder(in_dim, 1, h_list, device, generators=generators)
        policy_generator = builder(in_dim, action_dim, h_list, device, generators=generators)

        alpha = torch.tensor(
            [float(max(1e-4, min(1.0, float(h["alpha"])))) for h in h_list],
            device=device,
            dtype=torch.float32,
        )
        init_state_std = torch.tensor([float(h["init_state_std"]) for h in h_list], device=device, dtype=torch.float32)
        init_action_std = torch.tensor([float(h["init_action_std"]) for h in h_list], device=device, dtype=torch.float32)
        state_noise_std = torch.tensor([float(h["state_noise_std"]) for h in h_list], device=device, dtype=torch.float32)
        action_noise_train_std = torch.tensor(
            [float(h["action_noise_train_std"]) for h in h_list],
            device=device,
            dtype=torch.float32,
        )
        action_noise_eval_std = torch.tensor(
            [float(h["action_noise_eval_std"]) for h in h_list],
            device=device,
            dtype=torch.float32,
        )
        reward_scale = torch.tensor([float(h["reward_scale"]) for h in h_list], device=device, dtype=torch.float32)
        reward_clip = torch.tensor(
            [float(max(0.1, self._resolve_scalar(h.get("reward_clip", 10.0)))) for h in h_list],
            device=device,
            dtype=torch.float32,
        )
        state_clip = torch.tensor(
            [float(max(1.0, self._resolve_scalar(h.get("state_clip", 8.0)))) for h in h_list],
            device=device,
            dtype=torch.float32,
        )
        state_highway_enabled = torch.tensor(
            [bool(self._resolve_state_highway_enabled(h)) for h in h_list],
            device=device,
            dtype=torch.bool,
        )
        state_highway_lambda = torch.tensor(
            [float(self._resolve_state_highway_lambda(h)) for h in h_list],
            device=device,
            dtype=torch.float32,
        )
        aev4_cfg = self._resolve_aev4_config()
        aev4_enabled = torch.full(
            (batch_size,),
            bool(aev4_cfg.get("enabled", False)),
            device=device,
            dtype=torch.bool,
        )
        aev4_highway_ratio = torch.full(
            (batch_size,),
            float(aev4_cfg.get("highway_ratio", 0.25)),
            device=device,
            dtype=torch.float32,
        )
        aev4_update_scale = torch.full(
            (batch_size,),
            float(aev4_cfg.get("update_scale", 0.12)),
            device=device,
            dtype=torch.float32,
        )
        aev4_update_clip = torch.full(
            (batch_size,),
            float(aev4_cfg.get("update_clip", 0.0)),
            device=device,
            dtype=torch.float32,
        )
        reward_dropout_enabled = torch.tensor(
            [bool(h.get("reward_dropout_enabled", True)) for h in h_list],
            device=device,
            dtype=torch.bool,
        )
        reward_dropout_impute_zero = torch.tensor(
            [bool(h.get("reward_dropout_impute_zero", True)) for h in h_list],
            device=device,
            dtype=torch.bool,
        )
        reward_dropout_ratio = torch.tensor(
            [float(max(0.0, min(1.0, self._sample_reward_dropout_ratio(h)))) for h in h_list],
            device=device,
            dtype=torch.float32,
        )

        env = {
            "family": family,
            "state_dim": state_dim,
            "obs_dim": obs_dim,
            "action_dim": action_dim,
            "noise_dim": noise_dim,
            "zero_pad_dim": zero_pad_dim,
            "obs_slot_dim": int(max(1, h_list[0].get("obs_slot_dim", 400))),
            "action_slot_dim": int(max(1, h_list[0].get("action_slot_dim", 30))),
            "x_generator": x_generator,
            "y_generator": y_generator,
            "policy_generator": policy_generator,
            "alpha": alpha,
            "init_state_std": init_state_std,
            "init_action_std": init_action_std,
            "state_noise_std": state_noise_std,
            "action_noise_train_std": action_noise_train_std,
            "action_noise_eval_std": action_noise_eval_std,
            "reward_scale": reward_scale,
            "reward_clip": reward_clip,
            "state_clip": state_clip,
            "state_highway_enabled": state_highway_enabled,
            "state_highway_lambda": state_highway_lambda,
            "aev4_enabled": aev4_enabled,
            "aev4_highway_ratio": aev4_highway_ratio,
            "aev4_update_scale": aev4_update_scale,
            "aev4_update_clip": aev4_update_clip,
            "reward_dropout_enabled": reward_dropout_enabled,
            "reward_dropout_impute_zero": reward_dropout_impute_zero,
            "reward_dropout_ratio": reward_dropout_ratio,
        }
        return env

    @staticmethod
    def _make_generators_from_seeds(rng_seeds, batch_size, device):
        if rng_seeds is None:
            return None
        if len(rng_seeds) != int(batch_size):
            raise ValueError("rng_seeds length must match batch_size")
        generators = []
        for s in rng_seeds:
            g = torch.Generator(device=device)
            g.manual_seed(int(s))
            generators.append(g)
        return generators

    @staticmethod
    def _stack_randn_with_generators(generators, shape, device, dtype):
        if generators is None:
            return torch.randn(shape, device=device, dtype=dtype)
        cols = [
            torch.randn(shape[1:], device=device, dtype=dtype, generator=g)
            for g in generators
        ]
        return torch.stack(cols, dim=0)

    @staticmethod
    def _stack_rand_with_generators(generators, shape, device, dtype):
        if generators is None:
            return torch.rand(shape, device=device, dtype=dtype)
        cols = [
            torch.rand(shape[1:], device=device, dtype=dtype, generator=g)
            for g in generators
        ]
        return torch.stack(cols, dim=0)

    @staticmethod
    def _sample_scaled_noise_batch(generators, scale, width, device, dtype):
        scale = torch.as_tensor(scale, device=device, dtype=dtype)
        batch_size = int(scale.shape[0])
        width = int(width)
        if batch_size <= 0 or width <= 0 or (not bool(torch.any(scale > 0))):
            return None
        if generators is None:
            return torch.randn((batch_size, width), device=device, dtype=dtype) * scale[:, None]
        eps = torch.zeros((batch_size, width), device=device, dtype=dtype)
        generators_list = list(generators)
        for bi in range(batch_size):
            if float(scale[bi]) <= 0.0:
                continue
            g = generators_list[bi] if bi < len(generators_list) else None
            if g is None:
                e_b = torch.randn((width,), device=device, dtype=dtype)
            else:
                e_b = torch.randn((width,), device=device, dtype=dtype, generator=g)
            eps[bi] = e_b * scale[bi]
        return eps

    def _envgen_checkpoint_active(self, x):
        return bool(
            self.envgen_checkpoint
            and torch.is_grad_enabled()
            and torch.is_tensor(x)
            and bool(x.requires_grad)
        )

    @staticmethod
    def _build_active_tile_map(sizes, block):
        sizes_tensor = torch.as_tensor(sizes, dtype=torch.long)
        if int(sizes_tensor.numel()) <= 0:
            empty = torch.empty((0,), device=sizes_tensor.device, dtype=torch.int32)
            return empty, empty
        block = int(max(1, int(block)))
        batch_index = []
        offsets = []
        for bi, size_i in enumerate(sizes_tensor.detach().to(device="cpu", dtype=torch.long).tolist()):
            for start in range(0, max(0, int(size_i)), block):
                batch_index.append(int(bi))
                offsets.append(int(start))
        if not batch_index:
            empty = torch.empty((0,), device=sizes_tensor.device, dtype=torch.int32)
            return empty, empty
        return (
            torch.tensor(batch_index, device=sizes_tensor.device, dtype=torch.int32),
            torch.tensor(offsets, device=sizes_tensor.device, dtype=torch.int32),
        )

    @staticmethod
    def _build_segment_write_map(start_offsets, sizes, width_cap):
        sizes_t = torch.as_tensor(sizes, dtype=torch.long)
        if int(sizes_t.numel()) <= 0 or int(width_cap) <= 0:
            empty = torch.empty((0,), device=sizes_t.device, dtype=torch.long)
            return empty, empty, empty
        starts_t = torch.as_tensor(start_offsets, device=sizes_t.device, dtype=torch.long)
        src_cols = torch.arange(int(width_cap), device=sizes_t.device, dtype=torch.long).unsqueeze(0)
        valid = src_cols < sizes_t.unsqueeze(1)
        if not bool(torch.any(valid)):
            empty = torch.empty((0,), device=sizes_t.device, dtype=torch.long)
            return empty, empty, empty
        rows, src = torch.nonzero(valid, as_tuple=True)
        dst = starts_t[rows] + src
        return rows, dst, src
    def _batch_affine_prefix_tiled(
        self,
        x,
        w,
        b=None,
        in_sizes=None,
        out_sizes=None,
        activation_codes=None,
        out_tile_batch=None,
        out_tile_offsets=None,
        in_tile_batch=None,
        in_tile_offsets=None,
        block_o=32,
        block_k=32,
        num_warps=4,
    ):
        if not (
            triton is not None
            and torch.is_tensor(x)
            and torch.is_tensor(w)
            and torch.is_tensor(in_sizes)
            and torch.is_tensor(out_sizes)
            and torch.is_tensor(activation_codes)
            and torch.is_tensor(out_tile_batch)
            and torch.is_tensor(out_tile_offsets)
            and torch.is_tensor(in_tile_batch)
            and torch.is_tensor(in_tile_offsets)
            and x.device.type == "cuda"
            and w.device.type == "cuda"
            and in_sizes.device.type == "cuda"
            and out_sizes.device.type == "cuda"
            and activation_codes.device.type == "cuda"
            and out_tile_batch.device.type == "cuda"
            and out_tile_offsets.device.type == "cuda"
            and in_tile_batch.device.type == "cuda"
            and in_tile_offsets.device.type == "cuda"
            and x.dtype == torch.float32
            and w.dtype == torch.float32
            and x.ndim == 2
            and w.ndim == 3
            and int(x.shape[0]) == int(w.shape[0])
            and int(in_sizes.shape[0]) == int(x.shape[0])
            and int(out_sizes.shape[0]) == int(x.shape[0])
        ):
            in_cap = int(w.shape[1])
            out_cap = int(w.shape[2])
            x_in = x[:, :in_cap]
            in_mask = (
                torch.arange(in_cap, device=x.device, dtype=torch.long).unsqueeze(0)
                < in_sizes.to(device=x.device, dtype=torch.long).unsqueeze(1)
            ).to(dtype=x.dtype)
            out = self._batch_affine(x_in * in_mask, w, b)
            out_mask = (
                torch.arange(out_cap, device=out.device, dtype=torch.long).unsqueeze(0)
                < out_sizes.to(device=out.device, dtype=torch.long).unsqueeze(1)
            ).to(dtype=out.dtype)
            out = out * out_mask
            if torch.is_tensor(activation_codes) and int(activation_codes.numel()) == int(out.shape[0]):
                activation_codes = activation_codes.to(device=out.device, dtype=torch.long)
                relu_mask = activation_codes == 1
                tanh_mask = activation_codes == 0
                cos_mask = activation_codes == 3
                if bool(torch.any(relu_mask)):
                    out = torch.where(relu_mask.unsqueeze(1), torch.relu(out), out)
                if bool(torch.any(tanh_mask)):
                    out = torch.where(tanh_mask.unsqueeze(1), torch.tanh(out), out)
                if bool(torch.any(cos_mask)):
                    out = torch.where(cos_mask.unsqueeze(1), torch.cos(out), out)
                    out = out * out_mask
            return out
        return _PrefixTiledBatchAffineFn.apply(
            x,
            w,
            b,
            in_sizes,
            out_sizes,
            activation_codes,
            out_tile_batch,
            out_tile_offsets,
            in_tile_batch,
            in_tile_offsets,
            int(block_o),
            int(block_k),
            int(num_warps),
        )

    def _batch_affine_with_layout_tiled(
        self,
        x,
        w,
        b=None,
        input_index=None,
        in_sizes=None,
        out_sizes=None,
        activation_codes=None,
        out_tile_batch=None,
        out_tile_offsets=None,
        in_tile_batch=None,
        in_tile_offsets=None,
        block_o=32,
        block_k=32,
        num_warps=4,
    ):
        if not (
            triton is not None
            and torch.is_tensor(x)
            and torch.is_tensor(w)
            and torch.is_tensor(input_index)
            and torch.is_tensor(in_sizes)
            and torch.is_tensor(out_sizes)
            and torch.is_tensor(activation_codes)
            and torch.is_tensor(out_tile_batch)
            and torch.is_tensor(out_tile_offsets)
            and torch.is_tensor(in_tile_batch)
            and torch.is_tensor(in_tile_offsets)
            and x.device.type == "cuda"
            and w.device.type == "cuda"
            and input_index.device.type == "cuda"
            and in_sizes.device.type == "cuda"
            and out_sizes.device.type == "cuda"
            and activation_codes.device.type == "cuda"
            and out_tile_batch.device.type == "cuda"
            and out_tile_offsets.device.type == "cuda"
            and in_tile_batch.device.type == "cuda"
            and in_tile_offsets.device.type == "cuda"
            and x.dtype == torch.float32
            and w.dtype == torch.float32
            and x.ndim == 2
            and w.ndim == 3
            and int(x.shape[0]) == int(w.shape[0])
            and int(input_index.shape[0]) == int(x.shape[0])
            and int(in_sizes.shape[0]) == int(x.shape[0])
            and int(out_sizes.shape[0]) == int(x.shape[0])
        ):
            out = self._batch_affine_with_layout(
                x,
                w,
                b,
                input_index=input_index,
                in_sizes=in_sizes,
                out_sizes=out_sizes,
            )
            if torch.is_tensor(activation_codes) and int(activation_codes.numel()) == int(out.shape[0]):
                activation_codes = activation_codes.to(device=out.device, dtype=torch.long)
                tanh_mask = activation_codes == 0
                relu_mask = activation_codes == 1
                cos_mask = activation_codes == 3
                if bool(torch.any(tanh_mask)):
                    out = torch.where(tanh_mask.unsqueeze(1), torch.tanh(out), out)
                if bool(torch.any(relu_mask)):
                    out = torch.where(relu_mask.unsqueeze(1), torch.relu(out), out)
                if bool(torch.any(cos_mask)):
                    out = torch.where(cos_mask.unsqueeze(1), torch.cos(out), out)
                if bool(torch.any(cos_mask)):
                    out_mask = (
                        torch.arange(int(out.shape[1]), device=out.device, dtype=torch.long).unsqueeze(0)
                        < out_sizes.to(device=out.device, dtype=torch.long).unsqueeze(1)
                    ).to(dtype=out.dtype)
                    out = out * out_mask
            return out
        return _TiledRaggedBatchAffineFn.apply(
            x,
            w,
            b,
            input_index,
            in_sizes,
            out_sizes,
            activation_codes,
            out_tile_batch,
            out_tile_offsets,
            in_tile_batch,
            in_tile_offsets,
            int(block_o),
            int(block_k),
            int(num_warps),
        )

    def _batch_affine_with_layout(self, x, w, b=None, input_index=None, in_sizes=None, out_sizes=None):
        if not bool(
            self.envgen_ragged_affine
            and triton is not None
            and torch.is_tensor(x)
            and torch.is_tensor(w)
            and torch.is_tensor(input_index)
            and torch.is_tensor(in_sizes)
            and torch.is_tensor(out_sizes)
            and x.device.type == "cuda"
            and w.device.type == "cuda"
            and input_index.device.type == "cuda"
            and in_sizes.device.type == "cuda"
            and out_sizes.device.type == "cuda"
            and x.dtype == torch.float32
            and w.dtype == torch.float32
            and x.ndim == 2
            and w.ndim == 3
            and x.shape[0] == w.shape[0]
            and int(input_index.shape[0]) == int(x.shape[0])
            and int(in_sizes.shape[0]) == int(x.shape[0])
            and int(out_sizes.shape[0]) == int(x.shape[0])
        ):
            return self._batch_affine(x, w, b)
        return _RaggedBatchAffineFn.apply(x, w, b, input_index, in_sizes, out_sizes)

    def _batch_affine(self, x, w, b=None):
        """
        Batched affine map for per-sample weights.
        x: (B, I), w: (B, I, O), b: (B, O) or None.
        """
        if (x.device.type != "cuda") or (not bool(self.envgen_bmm)):
            out = torch.einsum("bi,bij->bj", x, w)
            return out if b is None else (out + b)
        x3 = x.unsqueeze(1)
        if b is None:
            return torch.bmm(x3, w).squeeze(1)
        return torch.baddbmm(b.unsqueeze(1), x3, w).squeeze(1)

    @staticmethod
    def _sample_seed_list(batch_size):
        batch_size = int(batch_size)
        if batch_size <= 0:
            return []
        return [int(x) for x in torch.randint(0, 2**31 - 1, (batch_size,), device="cpu").tolist()]

    @staticmethod
    def _build_vectorized_runtime_info(env, reward_values, state_abs_max, reward_drop_frac, single_eval_pos):
        # Pull reduced stats to CPU once per metric instead of syncing once per batch column.
        batch_size = int(reward_values.shape[1])
        reward_min_cpu = reward_values.min(dim=0).values.detach().cpu().tolist()
        reward_max_cpu = reward_values.max(dim=0).values.detach().cpu().tolist()
        reward_std_cpu = reward_values.std(dim=0, unbiased=False).detach().cpu().tolist()
        state_abs_max_cpu = state_abs_max.max(dim=0).values.detach().cpu().tolist()
        reward_drop_frac_cpu = reward_drop_frac.detach().cpu().tolist()
        reward_dropout_ratio = env["reward_dropout_ratio"]
        if torch.is_tensor(reward_dropout_ratio):
            reward_dropout_ratio_cpu = reward_dropout_ratio.detach().cpu().tolist()
        elif isinstance(reward_dropout_ratio, (list, tuple)):
            reward_dropout_ratio_cpu = [float(v) for v in reward_dropout_ratio]
        else:
            reward_dropout_ratio_cpu = [float(reward_dropout_ratio)] * batch_size
        state_highway_enabled = env.get("state_highway_enabled", False)
        if torch.is_tensor(state_highway_enabled):
            state_highway_enabled_cpu = state_highway_enabled.detach().cpu().tolist()
        elif isinstance(state_highway_enabled, (list, tuple)):
            state_highway_enabled_cpu = [bool(v) for v in state_highway_enabled]
        else:
            state_highway_enabled_cpu = [bool(state_highway_enabled)] * batch_size
        state_highway_lambda = env.get("state_highway_lambda", 0.0)
        if torch.is_tensor(state_highway_lambda):
            state_highway_lambda_cpu = state_highway_lambda.detach().cpu().tolist()
        elif isinstance(state_highway_lambda, (list, tuple)):
            state_highway_lambda_cpu = [float(v) for v in state_highway_lambda]
        else:
            state_highway_lambda_cpu = [float(state_highway_lambda)] * batch_size
        aev4_enabled = env.get("aev4_enabled", False)
        if torch.is_tensor(aev4_enabled):
            aev4_enabled_cpu = aev4_enabled.detach().cpu().tolist()
        elif isinstance(aev4_enabled, (list, tuple)):
            aev4_enabled_cpu = [bool(v) for v in aev4_enabled]
        else:
            aev4_enabled_cpu = [bool(aev4_enabled)] * batch_size
        aev4_highway_ratio = env.get("aev4_highway_ratio", 0.25)
        if torch.is_tensor(aev4_highway_ratio):
            aev4_highway_ratio_cpu = aev4_highway_ratio.detach().cpu().tolist()
        elif isinstance(aev4_highway_ratio, (list, tuple)):
            aev4_highway_ratio_cpu = [float(v) for v in aev4_highway_ratio]
        else:
            aev4_highway_ratio_cpu = [float(aev4_highway_ratio)] * batch_size
        aev4_update_scale = env.get("aev4_update_scale", 0.12)
        if torch.is_tensor(aev4_update_scale):
            aev4_update_scale_cpu = aev4_update_scale.detach().cpu().tolist()
        elif isinstance(aev4_update_scale, (list, tuple)):
            aev4_update_scale_cpu = [float(v) for v in aev4_update_scale]
        else:
            aev4_update_scale_cpu = [float(aev4_update_scale)] * batch_size
        aev4_update_clip = env.get("aev4_update_clip", 0.0)
        if torch.is_tensor(aev4_update_clip):
            aev4_update_clip_cpu = aev4_update_clip.detach().cpu().tolist()
        elif isinstance(aev4_update_clip, (list, tuple)):
            aev4_update_clip_cpu = [float(v) for v in aev4_update_clip]
        else:
            aev4_update_clip_cpu = [float(aev4_update_clip)] * batch_size

        def _env_value_at(key, b, cast_int=False):
            value = env[key]
            if torch.is_tensor(value):
                if value.ndim == 0:
                    item = value.item()
                else:
                    item = value[b].item()
            elif isinstance(value, (list, tuple)):
                item = value[b]
            else:
                item = value
            return int(item) if cast_int else item

        infos = []
        for b in range(batch_size):
            infos.append(
                {
                    "family": str(_env_value_at("family", b)),
                    "state_dim": _env_value_at("state_dim", b, cast_int=True),
                    "obs_dim": _env_value_at("obs_dim", b, cast_int=True),
                    "action_dim": _env_value_at("action_dim", b, cast_int=True),
                    "noise_dim": _env_value_at("noise_dim", b, cast_int=True),
                    "zero_pad_dim": _env_value_at("zero_pad_dim", b, cast_int=True),
                    "obs_slot_dim": _env_value_at("obs_slot_dim", b, cast_int=True),
                    "action_slot_dim": _env_value_at("action_slot_dim", b, cast_int=True),
                    "single_eval_pos": int(single_eval_pos),
                    "reward_min": float(reward_min_cpu[b]),
                    "reward_max": float(reward_max_cpu[b]),
                    "reward_std": float(reward_std_cpu[b]),
                    "state_abs_max": float(state_abs_max_cpu[b]),
                    "reward_dropout_ratio": float(reward_dropout_ratio_cpu[b]),
                    "reward_drop_frac_realized": float(reward_drop_frac_cpu[b]),
                    "state_highway_enabled": bool(state_highway_enabled_cpu[b]),
                    "state_highway_lambda": float(state_highway_lambda_cpu[b]),
                    "aev4_enabled": bool(aev4_enabled_cpu[b]),
                    "aev4_highway_ratio": float(aev4_highway_ratio_cpu[b]),
                    "aev4_update_scale": float(aev4_update_scale_cpu[b]),
                    "aev4_update_clip": float(aev4_update_clip_cpu[b]),
                }
            )
        return infos

    def _rollout_distinct_envs_vectorized(
        self,
        env,
        batch_size,
        n_samples,
        num_features,
        single_eval_pos,
        device,
        collect_x=True,
        rng_seeds=None,
    ):
        if not collect_x:
            raise ValueError("get_batch vectorized rollout requires collect_x=True")

        n_samples = int(n_samples)
        batch_size = int(batch_size)
        num_features = int(num_features)

        state_dim = int(env["state_dim"])
        obs_dim = int(env["obs_dim"])
        action_dim = int(env["action_dim"])
        noise_dim = int(env["noise_dim"])
        zero_pad_dim = int(env["zero_pad_dim"])
        obs_slot_dim = int(env["obs_slot_dim"])
        action_slot_dim = int(env["action_slot_dim"])
        rollout_generators = self._make_generators_from_seeds(rng_seeds, batch_size, device)

        state_t = self._stack_randn_with_generators(
            rollout_generators,
            (batch_size, state_dim),
            device=device,
            dtype=torch.float32,
        ) * env["init_state_std"][:, None]
        action_t = self._stack_randn_with_generators(
            rollout_generators,
            (batch_size, action_dim),
            device=device,
            dtype=torch.float32,
        ) * env["init_action_std"][:, None]
        reward_t = torch.zeros((batch_size,), device=device, dtype=torch.float32)
        reward_mask_t = torch.ones((batch_size,), device=device, dtype=torch.float32)

        x_steps = torch.empty((n_samples, batch_size, num_features), device=device, dtype=torch.float32)
        y_steps = torch.empty((n_samples, batch_size), device=device, dtype=torch.float32)
        state_abs_max = torch.empty((n_samples, batch_size), device=device, dtype=torch.float32)
        reward_values = torch.empty((n_samples, batch_size), device=device, dtype=torch.float32)
        reward_drop_count = torch.zeros((batch_size,), device=device, dtype=torch.int64)

        transition_noise = self._stack_randn_with_generators(
            rollout_generators,
            (batch_size, n_samples, noise_dim),
            device=device,
            dtype=torch.float32,
        ).transpose(0, 1)
        action_noise_train = None
        action_noise_eval = None
        state_noise = None
        if torch.any(env["action_noise_train_std"] > 0):
            action_noise_train = self._stack_randn_with_generators(
                rollout_generators,
                (batch_size, n_samples, action_dim),
                device=device,
                dtype=torch.float32,
            ).transpose(0, 1)
        if torch.any(env["action_noise_eval_std"] > 0):
            action_noise_eval = self._stack_randn_with_generators(
                rollout_generators,
                (batch_size, n_samples, action_dim),
                device=device,
                dtype=torch.float32,
            ).transpose(0, 1)
        if torch.any(env["state_noise_std"] > 0):
            state_noise = self._stack_randn_with_generators(
                rollout_generators,
                (batch_size, n_samples, state_dim),
                device=device,
                dtype=torch.float32,
            ).transpose(0, 1)

        dropout_active = env["reward_dropout_enabled"] & (env["reward_dropout_ratio"] > 0.0)
        dropout_draws = None
        if torch.any(dropout_active):
            dropout_draws = self._stack_rand_with_generators(
                rollout_generators,
                (batch_size, n_samples),
                device=device,
                dtype=torch.float32,
            ).transpose(0, 1)

        token_reward_idx = obs_slot_dim
        token_mask_idx = obs_slot_dim + 1
        token_action_start = obs_slot_dim + 2
        token_obs_cap = min(obs_dim, obs_slot_dim)
        token_action_cap = min(action_dim, action_slot_dim)

        env_total_dim = state_dim + obs_dim + action_dim + noise_dim + zero_pad_dim
        env_in = torch.zeros((batch_size, env_total_dim), device=device, dtype=torch.float32)
        env_obs_start = state_dim
        env_action_start = state_dim + obs_dim
        env_noise_start = env_action_start + action_dim
        for t in range(n_samples):
            obs_t = state_t[:, :obs_dim]
            token_row = x_steps[t]
            token_row.zero_()
            if num_features > 0 and token_obs_cap > 0:
                obs_write = min(token_obs_cap, num_features)
                token_row[:, :obs_write] = obs_t[:, :obs_write]
            if token_reward_idx < num_features:
                token_row[:, token_reward_idx] = reward_t
            if token_mask_idx < num_features:
                token_row[:, token_mask_idx] = reward_mask_t
            if token_action_start < num_features and token_action_cap > 0:
                action_write = min(token_action_cap, num_features - token_action_start)
                token_row[:, token_action_start: token_action_start + action_write] = action_t[:, :action_write]

            noise_t = transition_noise[t]
            env_in[:, :state_dim] = state_t
            env_in[:, env_obs_start: env_obs_start + obs_dim] = obs_t
            env_in[:, env_action_start: env_action_start + action_dim] = action_t
            env_in[:, env_noise_start: env_noise_start + noise_dim] = noise_t
            action_next = torch.tanh(env["policy_generator"](env_in, generators_for_noise=rollout_generators))

            if t < single_eval_pos:
                if action_noise_train is not None:
                    action_next = torch.tanh(action_next + action_noise_train[t] * env["action_noise_train_std"][:, None])
            elif action_noise_eval is not None:
                action_next = torch.tanh(action_next + action_noise_eval[t] * env["action_noise_eval_std"][:, None])

            env_in[:, env_action_start: env_action_start + action_dim] = action_next
            reward_next_raw = env["reward_scale"] * env["y_generator"](
                env_in,
                generators_for_noise=rollout_generators,
            ).reshape(batch_size)
            reward_next = torch.maximum(
                torch.minimum(reward_next_raw, env["reward_clip"]),
                -env["reward_clip"],
            )
            reward_mask_next = torch.ones((batch_size,), device=device, dtype=torch.float32)

            if dropout_draws is not None:
                drop_mask = dropout_active & (dropout_draws[t] < env["reward_dropout_ratio"])
                reward_drop_count = reward_drop_count + drop_mask.to(dtype=torch.int64)
                reward_mask_next = torch.where(drop_mask, torch.zeros_like(reward_mask_next), reward_mask_next)
                impute_mask = drop_mask & env["reward_dropout_impute_zero"]
                reward_next = torch.where(impute_mask, torch.zeros_like(reward_next), reward_next)

            x_next = env["x_generator"](env_in, generators_for_noise=rollout_generators)
            state_next = (1.0 - env["alpha"][:, None]) * state_t + env["alpha"][:, None] * x_next
            if state_noise is not None:
                state_next = state_next + state_noise[t] * env["state_noise_std"][:, None]
            state_next = self._apply_state_postprocess(
                state_next_raw=state_next,
                state_prev=state_t,
                state_clip=env["state_clip"],
                state_highway_enabled=env.get("state_highway_enabled", False),
                state_highway_lambda=env.get("state_highway_lambda", 0.0),
            )

            y_steps[t] = reward_next
            reward_values[t] = reward_next.detach()
            state_abs_max[t] = state_next.detach().abs().amax(dim=1)

            state_t = state_next
            action_t = action_next
            reward_t = reward_next
            reward_mask_t = reward_mask_next

        reward_drop_frac = reward_drop_count.to(dtype=torch.float32) / float(max(1, n_samples))
        infos = self._build_vectorized_runtime_info(
            env=env,
            reward_values=reward_values,
            state_abs_max=state_abs_max,
            reward_drop_frac=reward_drop_frac,
            single_eval_pos=single_eval_pos,
        )

        return x_steps, y_steps, infos

    @staticmethod
    def _policy_step_accepts_reward_mask(policy_step_fn):
        try:
            sig = inspect.signature(policy_step_fn)
            params = list(sig.parameters.values())
            has_var_positional = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params)
            positional_count = sum(
                p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                for p in params
            )
            return bool(has_var_positional or positional_count >= 7)
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _detach_policy_cache(cache, clone_tensors=False):
        if cache is None:
            return None
        if torch.is_tensor(cache):
            out = cache.detach()
            return out.clone() if bool(clone_tensors) else out
        if isinstance(cache, list):
            return [EnvironmentPrior._detach_policy_cache(v, clone_tensors=clone_tensors) for v in cache]
        if isinstance(cache, tuple):
            return tuple(EnvironmentPrior._detach_policy_cache(v, clone_tensors=clone_tensors) for v in cache)
        if isinstance(cache, dict):
            detached = {
                k: EnvironmentPrior._detach_policy_cache(v, clone_tensors=clone_tensors)
                for k, v in cache.items()
            }
            # After TBPTT detach, keep old paged tail immutable and let the next
            # window start from a fresh mutable page to avoid COW-copying long
            # detached history on every append.
            tail_freeze_env = str(os.environ.get("TICL_POLICY_TAIL_FREEZE", "1")).strip().lower()
            tail_freeze_enabled = tail_freeze_env not in {"0", "false", "no", "off"}
            if tail_freeze_enabled and detached.get("cache_mode", None) == "paged":
                if isinstance(detached.get("k_pages", None), list) and isinstance(
                    detached.get("v_pages", None), list
                ):
                    compact_prefix_env = str(
                        os.environ.get("TICL_POLICY_PREFIX_COMPACT_ON_TBPTT_DETACH", "1")
                    ).strip().lower()
                    compact_prefix_enabled = compact_prefix_env not in {"0", "false", "no", "off"}
                    if compact_prefix_enabled:
                        k_pages = detached.get("k_pages", None)
                        v_pages = detached.get("v_pages", None)
                        if len(k_pages) > 1:
                            full_k_pages = k_pages[:-1]
                            full_v_pages = v_pages[:-1]
                            k_prefix = detached.get("k_prefix", None)
                            v_prefix = detached.get("v_prefix", None)
                            if (k_prefix is None) or (v_prefix is None):
                                if len(full_k_pages) == 1:
                                    k_prefix = full_k_pages[0]
                                    v_prefix = full_v_pages[0]
                                elif len(full_k_pages) > 1:
                                    k_prefix = torch.cat(full_k_pages, dim=2)
                                    v_prefix = torch.cat(full_v_pages, dim=2)
                            detached["k_prefix"] = k_prefix
                            detached["v_prefix"] = v_prefix
                            detached["prefix_base_len"] = (
                                int(k_prefix.shape[2]) if torch.is_tensor(k_prefix) else 0
                            )
                            detached["prefix_pages"] = 0
                            detached["k_pages"] = [k_pages[-1]]
                            detached["v_pages"] = [v_pages[-1]]
                            detached["paged_packed"] = False
                        elif "prefix_base_len" not in detached:
                            k_prefix = detached.get("k_prefix", None)
                            detached["prefix_base_len"] = (
                                int(k_prefix.shape[2]) if torch.is_tensor(k_prefix) else 0
                            )
                    k_pages = detached.get("k_pages", None)
                    v_pages = detached.get("v_pages", None)
                    if isinstance(k_pages, list) and isinstance(v_pages, list) and len(k_pages) > 0:
                        try:
                            valid_len = int(detached.get("valid_len", 0))
                        except Exception:
                            valid_len = 0
                        k_prefix = detached.get("k_prefix", None)
                        v_prefix = detached.get("v_prefix", None)
                        try:
                            prefix_base_len = int(detached.get("prefix_base_len", 0))
                        except Exception:
                            prefix_base_len = 0
                        prefix_covered_len = int(prefix_base_len)
                        if torch.is_tensor(k_prefix) and torch.is_tensor(v_prefix):
                            prefix_covered_len = int(min(int(valid_len), int(k_prefix.shape[2])))
                        try:
                            prefix_pages = int(detached.get("prefix_pages", 0))
                        except Exception:
                            prefix_pages = 0
                        prefix_pages = int(max(0, min(int(prefix_pages), int(len(k_pages)))))
                        tail_valid_len = int(max(0, valid_len - prefix_covered_len))
                        if tail_valid_len > 0:
                            trimmed_k_pages = []
                            trimmed_v_pages = []
                            if prefix_pages > 0:
                                trimmed_k_pages.extend(k_pages[:prefix_pages])
                                trimmed_v_pages.extend(v_pages[:prefix_pages])
                            remaining = int(tail_valid_len)
                            for k_page, v_page in zip(k_pages[prefix_pages:], v_pages[prefix_pages:]):
                                if remaining <= 0:
                                    break
                                take = min(int(k_page.shape[2]), remaining)
                                trimmed_k_pages.append(k_page[:, :, :take, :])
                                trimmed_v_pages.append(v_page[:, :, :take, :])
                                remaining -= take
                            if trimmed_k_pages and trimmed_v_pages:
                                detached["k_pages"] = trimmed_k_pages
                                detached["v_pages"] = trimmed_v_pages
                                detached["paged_packed"] = bool(
                                    sum(int(page.shape[2]) for page in trimmed_k_pages[prefix_pages:])
                                    == int(tail_valid_len)
                                )
                    detached["tail_frozen"] = True
            return detached
        return cache

    def _rollout_distinct_envs_vectorized_with_policy(
        self,
        env,
        policy_step_fn,
        batch_size,
        n_samples,
        num_features,
        single_eval_pos,
        device,
        collect_x=True,
        collect_runtime_info=True,
        rng_seeds=None,
        tbptt_window=None,
        tbptt_reward_sink=None,
        tbptt_reward_sink_supports_aux=False,
        store_rewards=True,
    ):
        n_samples = int(n_samples)
        batch_size = int(batch_size)
        num_features = int(num_features)
        collect_runtime_info = bool(collect_runtime_info)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        state_dim = int(env["state_dim"])
        obs_dim = int(env["obs_dim"])
        action_dim = int(env["action_dim"])
        noise_dim = int(env["noise_dim"])
        zero_pad_dim = int(env["zero_pad_dim"])
        obs_slot_dim = int(env["obs_slot_dim"])
        action_slot_dim = int(env["action_slot_dim"])
        rollout_generators = self._make_generators_from_seeds(rng_seeds, batch_size, device)
        policy_accepts_reward_mask = self._policy_step_accepts_reward_mask(policy_step_fn)
        profile_rollout_breakdown_flag = str(os.environ.get("TICL_PROFILE_ROLLOUT_BREAKDOWN", "")).strip().lower()
        profile_rollout_breakdown = profile_rollout_breakdown_flag in {"1", "true", "yes", "on"}
        profile_rollout_timing_flag = str(os.environ.get("TICL_PROFILE_ROLLOUT_TIMING", "")).strip().lower()
        profile_rollout_timing = profile_rollout_timing_flag in {"1", "true", "yes", "on"}
        device_obj = device if isinstance(device, torch.device) else torch.device(str(device))
        profile_rollout_breakdown_cuda = bool(
            profile_rollout_breakdown
            and device_obj.type == "cuda"
            and torch.cuda.is_available()
        )
        policy_cuda_pairs = []
        transition_cuda_pairs = []
        policy_wall_s = 0.0
        transition_wall_s = 0.0
        transition_y_wall_s = 0.0
        transition_x_wall_s = 0.0
        transition_group_wall_s = 0.0
        transition_env_pack_wall_s = 0.0
        transition_state_update_wall_s = 0.0
        transition_noise_wall_s = 0.0

        state_t = self._stack_randn_with_generators(
            rollout_generators,
            (batch_size, state_dim),
            device=device,
            dtype=torch.float32,
        ) * env["init_state_std"][:, None]
        action_t = self._stack_randn_with_generators(
            rollout_generators,
            (batch_size, action_dim),
            device=device,
            dtype=torch.float32,
        ) * env["init_action_std"][:, None]
        reward_t = torch.zeros((batch_size,), device=device, dtype=torch.float32)
        reward_mask_t = torch.ones((batch_size,), device=device, dtype=torch.float32)
        cache = None

        x_steps = (
            torch.empty((n_samples, batch_size, num_features), device=device, dtype=torch.float32)
            if collect_x
            else None
        )
        y_steps = (
            torch.empty((n_samples, batch_size), device=device, dtype=torch.float32)
            if bool(store_rewards)
            else None
        )
        state_abs_max = (
            torch.empty((n_samples, batch_size), device=device, dtype=torch.float32)
            if collect_runtime_info
            else None
        )
        reward_values = (
            torch.empty((n_samples, batch_size), device=device, dtype=torch.float32)
            if collect_runtime_info
            else None
        )
        reward_drop_count = (
            torch.zeros((batch_size,), device=device, dtype=torch.int64)
            if collect_runtime_info
            else None
        )
        tbptt_window_active = False
        tbptt_window_size = n_samples
        if tbptt_window is not None:
            w = int(tbptt_window)
            if 0 < w < n_samples:
                tbptt_window_active = True
                tbptt_window_size = w
        tbptt_reward_buffer = [] if tbptt_window_active else None
        aev2_cfg = self._resolve_aev2_config()
        aev2_enabled = bool(aev2_cfg.get("enabled", False))
        aev2_prev_delta = None
        aev3_cfg = self._resolve_aev3_config()
        aev3_enabled = bool(aev3_cfg.get("enabled", False))
        aev3_prev_delta = None
        aev4_cfg = self._resolve_aev4_config()
        aev4_reg_enabled = bool(aev4_cfg.get("enabled", False))
        aev4_prev_delta = None
        aev2_streaming_sink = bool(
            aev2_enabled
            and tbptt_window_active
            and (tbptt_reward_sink is not None)
            and bool(tbptt_reward_sink_supports_aux)
        )
        aev3_streaming_sink = bool(
            aev3_enabled
            and tbptt_window_active
            and (tbptt_reward_sink is not None)
            and bool(tbptt_reward_sink_supports_aux)
        )
        aev4_streaming_sink = bool(
            aev4_reg_enabled
            and tbptt_window_active
            and (tbptt_reward_sink is not None)
            and bool(tbptt_reward_sink_supports_aux)
        )
        aev2_acc = self._aev2_new_accumulator(aev2_enabled, device=device, dtype=torch.float32)
        aev2_det_acc = self._aev2_new_accumulator(aev2_enabled, device=device, dtype=torch.float32)
        aev3_acc = self._aev3_new_accumulator(aev3_enabled, device=device, dtype=torch.float32, aev3_cfg=aev3_cfg)
        aev3_det_acc = self._aev3_new_accumulator(
            aev3_enabled, device=device, dtype=torch.float32, aev3_cfg=aev3_cfg
        )
        aev4_acc = self._aev4_new_accumulator(
            aev4_reg_enabled, device=device, dtype=torch.float32, aev4_cfg=aev4_cfg
        )
        aev4_det_acc = self._aev4_new_accumulator(
            aev4_reg_enabled, device=device, dtype=torch.float32, aev4_cfg=aev4_cfg
        )

        strict_seed_mode = rollout_generators is not None
        transition_noise = None
        action_noise_train = None
        action_noise_eval = None
        state_noise = None
        dropout_draws = None
        noise_block_size = 0
        noise_streaming_mode = False
        noise_block_start = 0
        noise_block_end = 0
        transition_noise_block = None
        action_noise_train_block = None
        action_noise_eval_block = None
        state_noise_block = None
        dropout_draws_block = None

        transition_noise_generators = None
        action_noise_train_generators = None
        action_noise_eval_generators = None
        state_noise_generators = None
        dropout_draw_generators = None
        env_noise_generators = rollout_generators

        dropout_active = env["reward_dropout_enabled"] & (env["reward_dropout_ratio"] > 0.0)

        def _clone_generator_state(gen):
            cloned = torch.Generator(device=device)
            cloned.set_state(gen.get_state())
            return cloned

        def _advance_generator_randn(gen, numel):
            remaining = int(max(0, numel))
            chunk = 1 << 15
            while remaining > 0:
                take = min(chunk, remaining)
                torch.randn((take,), device=device, dtype=torch.float32, generator=gen)
                remaining -= take

        def _advance_generator_rand(gen, numel):
            remaining = int(max(0, numel))
            chunk = 1 << 15
            while remaining > 0:
                take = min(chunk, remaining)
                torch.rand((take,), device=device, dtype=torch.float32, generator=gen)
                remaining -= take

        def _draw_step_randn_with_optional_generators(generators, width):
            out = torch.empty((batch_size, width), device=device, dtype=torch.float32)
            for bi, g in enumerate(generators):
                if g is None:
                    out[bi].zero_()
                else:
                    out[bi] = torch.randn((width,), device=device, dtype=torch.float32, generator=g)
            return out

        def _draw_step_rand_with_optional_generators(generators):
            out = torch.empty((batch_size,), device=device, dtype=torch.float32)
            for bi, g in enumerate(generators):
                if g is None:
                    out[bi] = 0.0
                else:
                    out[bi] = torch.rand((), device=device, dtype=torch.float32, generator=g)
            return out

        if strict_seed_mode:
            transition_noise_generators = [_clone_generator_state(g) for g in rollout_generators]
            for g in rollout_generators:
                _advance_generator_randn(g, n_samples * noise_dim)

            if torch.any(env["action_noise_train_std"] > 0):
                action_noise_train_generators = [None] * batch_size
                for bi, g in enumerate(rollout_generators):
                    if float(env["action_noise_train_std"][bi]) > 0.0:
                        action_noise_train_generators[bi] = _clone_generator_state(g)
                        _advance_generator_randn(g, n_samples * action_dim)

            if torch.any(env["action_noise_eval_std"] > 0):
                action_noise_eval_generators = [None] * batch_size
                for bi, g in enumerate(rollout_generators):
                    if float(env["action_noise_eval_std"][bi]) > 0.0:
                        action_noise_eval_generators[bi] = _clone_generator_state(g)
                        _advance_generator_randn(g, n_samples * action_dim)

            if torch.any(env["state_noise_std"] > 0):
                state_noise_generators = [None] * batch_size
                for bi, g in enumerate(rollout_generators):
                    if float(env["state_noise_std"][bi]) > 0.0:
                        state_noise_generators[bi] = _clone_generator_state(g)
                        _advance_generator_randn(g, n_samples * state_dim)

            if torch.any(dropout_active):
                dropout_draw_generators = [None] * batch_size
                for bi, g in enumerate(rollout_generators):
                    if bool(dropout_active[bi]):
                        dropout_draw_generators[bi] = _clone_generator_state(g)
                        _advance_generator_rand(g, n_samples)
            env_noise_generators = rollout_generators
        else:
            noise_block_size = self._resolve_rollout_noise_block_size(n_samples)
            noise_streaming_mode = bool(noise_block_size > 0)

            def _refresh_noise_block(block_start_idx):
                nonlocal noise_block_start, noise_block_end
                nonlocal transition_noise_block, action_noise_train_block, action_noise_eval_block
                nonlocal state_noise_block, dropout_draws_block
                block_start_idx = int(block_start_idx)
                block_len = int(min(noise_block_size, n_samples - block_start_idx))
                noise_block_start = block_start_idx
                noise_block_end = block_start_idx + block_len
                transition_noise_block = self._stack_randn_with_generators(
                    rollout_generators,
                    (batch_size, block_len, noise_dim),
                    device=device,
                    dtype=torch.float32,
                ).transpose(0, 1)
                if torch.any(env["action_noise_train_std"] > 0):
                    action_noise_train_block = self._stack_randn_with_generators(
                        rollout_generators,
                        (batch_size, block_len, action_dim),
                        device=device,
                        dtype=torch.float32,
                    ).transpose(0, 1)
                else:
                    action_noise_train_block = None
                if torch.any(env["action_noise_eval_std"] > 0):
                    action_noise_eval_block = self._stack_randn_with_generators(
                        rollout_generators,
                        (batch_size, block_len, action_dim),
                        device=device,
                        dtype=torch.float32,
                    ).transpose(0, 1)
                else:
                    action_noise_eval_block = None
                if torch.any(env["state_noise_std"] > 0):
                    state_noise_block = self._stack_randn_with_generators(
                        rollout_generators,
                        (batch_size, block_len, state_dim),
                        device=device,
                        dtype=torch.float32,
                    ).transpose(0, 1)
                else:
                    state_noise_block = None
                if torch.any(dropout_active):
                    dropout_draws_block = self._stack_rand_with_generators(
                        rollout_generators,
                        (batch_size, block_len),
                        device=device,
                        dtype=torch.float32,
                    ).transpose(0, 1)
                else:
                    dropout_draws_block = None

            if noise_streaming_mode:
                _refresh_noise_block(0)
            else:
                transition_noise = self._stack_randn_with_generators(
                    rollout_generators,
                    (batch_size, n_samples, noise_dim),
                    device=device,
                    dtype=torch.float32,
                ).transpose(0, 1)
                if torch.any(env["action_noise_train_std"] > 0):
                    action_noise_train = self._stack_randn_with_generators(
                        rollout_generators,
                        (batch_size, n_samples, action_dim),
                        device=device,
                        dtype=torch.float32,
                    ).transpose(0, 1)
                if torch.any(env["action_noise_eval_std"] > 0):
                    action_noise_eval = self._stack_randn_with_generators(
                        rollout_generators,
                        (batch_size, n_samples, action_dim),
                        device=device,
                        dtype=torch.float32,
                    ).transpose(0, 1)
                if torch.any(env["state_noise_std"] > 0):
                    state_noise = self._stack_randn_with_generators(
                        rollout_generators,
                        (batch_size, n_samples, state_dim),
                        device=device,
                        dtype=torch.float32,
                    ).transpose(0, 1)
                if torch.any(dropout_active):
                    dropout_draws = self._stack_rand_with_generators(
                        rollout_generators,
                        (batch_size, n_samples),
                        device=device,
                        dtype=torch.float32,
                    ).transpose(0, 1)

        token_reward_idx = obs_slot_dim
        token_mask_idx = obs_slot_dim + 1
        token_action_start = obs_slot_dim + 2
        token_obs_cap = min(obs_dim, obs_slot_dim)
        token_action_cap = min(action_dim, action_slot_dim)

        env_total_dim = state_dim + obs_dim + action_dim + noise_dim + zero_pad_dim
        env_in = torch.zeros((batch_size, env_total_dim), device=device, dtype=torch.float32)
        env_obs_start = state_dim
        env_action_start = state_dim + obs_dim
        env_noise_start = env_action_start + action_dim
        for t in range(n_samples):
            obs_t = state_t[:, :obs_dim]
            if collect_x:
                with torch.no_grad():
                    token_row = x_steps[t]
                    token_row.zero_()
                    if num_features > 0 and token_obs_cap > 0:
                        obs_write = min(token_obs_cap, num_features)
                        token_row[:, :obs_write] = obs_t[:, :obs_write].detach()
                    if token_reward_idx < num_features:
                        token_row[:, token_reward_idx] = reward_t.detach()
                    if token_mask_idx < num_features:
                        token_row[:, token_mask_idx] = reward_mask_t.detach()
                    if token_action_start < num_features and token_action_cap > 0:
                        action_write = min(token_action_cap, num_features - token_action_start)
                        token_row[:, token_action_start: token_action_start + action_write] = action_t[:, :action_write].detach()

            policy_out = None
            policy_cuda_start = None
            if profile_rollout_breakdown_cuda:
                policy_cuda_start = torch.cuda.Event(enable_timing=True)
                policy_cuda_start.record()
            if policy_accepts_reward_mask:
                policy_out = policy_step_fn(
                    obs_t,
                    action_t,
                    reward_t.reshape(batch_size, 1),
                    reward_mask_t.reshape(batch_size, 1),
                    cache,
                    t,
                    env,
                )
            else:
                policy_out = policy_step_fn(
                    obs_t,
                    action_t,
                    reward_t.reshape(batch_size, 1),
                    cache,
                    t,
                    env,
                )
            if policy_cuda_start is not None:
                policy_cuda_end = torch.cuda.Event(enable_timing=True)
                policy_cuda_end.record()
                policy_cuda_pairs.append((policy_cuda_start, policy_cuda_end))
            if isinstance(policy_out, tuple):
                action_next, cache = policy_out
            else:
                action_next = policy_out

            if action_next.ndim == 1:
                action_next = action_next.reshape(batch_size, 1)
            if action_next.ndim != 2 or action_next.shape[0] != batch_size:
                raise ValueError(
                    f"policy action batch mismatch: expected ({batch_size}, {action_dim}), got {tuple(action_next.shape)}"
                )
            if action_next.shape[-1] != action_dim:
                raise ValueError(
                    f"policy action dim mismatch: expected {action_dim}, got {action_next.shape[-1]}"
                )
            action_next = torch.tanh(action_next)

            noise_timing_t0 = time.perf_counter() if profile_rollout_timing else None
            noise_block_idx = None
            if noise_streaming_mode:
                if t >= noise_block_end:
                    _refresh_noise_block(t)
                noise_block_idx = int(t - noise_block_start)

            if t < single_eval_pos:
                if strict_seed_mode and action_noise_train_generators is not None:
                    action_noise_train_t = _draw_step_randn_with_optional_generators(
                        action_noise_train_generators,
                        action_dim,
                    )
                    action_next = torch.tanh(action_next + action_noise_train_t * env["action_noise_train_std"][:, None])
                elif noise_streaming_mode and action_noise_train_block is not None and noise_block_idx is not None:
                    action_next = torch.tanh(
                        action_next + action_noise_train_block[noise_block_idx] * env["action_noise_train_std"][:, None]
                    )
                elif action_noise_train is not None:
                    action_next = torch.tanh(action_next + action_noise_train[t] * env["action_noise_train_std"][:, None])
            else:
                if strict_seed_mode and action_noise_eval_generators is not None:
                    action_noise_eval_t = _draw_step_randn_with_optional_generators(
                        action_noise_eval_generators,
                        action_dim,
                    )
                    action_next = torch.tanh(action_next + action_noise_eval_t * env["action_noise_eval_std"][:, None])
                elif noise_streaming_mode and action_noise_eval_block is not None and noise_block_idx is not None:
                    action_next = torch.tanh(
                        action_next + action_noise_eval_block[noise_block_idx] * env["action_noise_eval_std"][:, None]
                    )
                elif action_noise_eval is not None:
                    action_next = torch.tanh(action_next + action_noise_eval[t] * env["action_noise_eval_std"][:, None])

            transition_cuda_start = None
            if profile_rollout_breakdown_cuda:
                transition_cuda_start = torch.cuda.Event(enable_timing=True)
                transition_cuda_start.record()
            if strict_seed_mode:
                noise_t = _draw_step_randn_with_optional_generators(
                    transition_noise_generators,
                    noise_dim,
                )
            elif noise_streaming_mode and noise_block_idx is not None:
                noise_t = transition_noise_block[noise_block_idx]
            else:
                noise_t = transition_noise[t]
            if profile_rollout_timing and noise_timing_t0 is not None:
                transition_noise_wall_s += (time.perf_counter() - noise_timing_t0)
            env_in[:, :state_dim] = state_t
            env_in[:, env_obs_start: env_obs_start + obs_dim] = obs_t
            env_in[:, env_action_start: env_action_start + action_dim] = action_next
            env_in[:, env_noise_start: env_noise_start + noise_dim] = noise_t

            reward_next_raw = env["reward_scale"] * env["y_generator"](
                env_in,
                generators_for_noise=env_noise_generators,
            ).reshape(batch_size)
            reward_next = torch.maximum(
                torch.minimum(reward_next_raw, env["reward_clip"]),
                -env["reward_clip"],
            )
            reward_mask_next = torch.ones((batch_size,), device=device, dtype=torch.float32)

            dropout_timing_t0 = time.perf_counter() if profile_rollout_timing else None
            dropout_timed = False
            if strict_seed_mode and dropout_draw_generators is not None:
                dropout_timed = True
                drop_draw = _draw_step_rand_with_optional_generators(dropout_draw_generators)
                drop_mask = dropout_active & (drop_draw < env["reward_dropout_ratio"])
                if collect_runtime_info:
                    reward_drop_count = reward_drop_count + drop_mask.to(dtype=torch.int64)
                reward_mask_next = torch.where(drop_mask, torch.zeros_like(reward_mask_next), reward_mask_next)
                impute_mask = drop_mask & env["reward_dropout_impute_zero"]
                reward_next = torch.where(impute_mask, torch.zeros_like(reward_next), reward_next)
            elif noise_streaming_mode and dropout_draws_block is not None and noise_block_idx is not None:
                dropout_timed = True
                drop_mask = dropout_active & (dropout_draws_block[noise_block_idx] < env["reward_dropout_ratio"])
                if collect_runtime_info:
                    reward_drop_count = reward_drop_count + drop_mask.to(dtype=torch.int64)
                reward_mask_next = torch.where(drop_mask, torch.zeros_like(reward_mask_next), reward_mask_next)
                impute_mask = drop_mask & env["reward_dropout_impute_zero"]
                reward_next = torch.where(impute_mask, torch.zeros_like(reward_next), reward_next)
            elif dropout_draws is not None:
                dropout_timed = True
                drop_mask = dropout_active & (dropout_draws[t] < env["reward_dropout_ratio"])
                if collect_runtime_info:
                    reward_drop_count = reward_drop_count + drop_mask.to(dtype=torch.int64)
                reward_mask_next = torch.where(drop_mask, torch.zeros_like(reward_mask_next), reward_mask_next)
                impute_mask = drop_mask & env["reward_dropout_impute_zero"]
                reward_next = torch.where(impute_mask, torch.zeros_like(reward_next), reward_next)
            if profile_rollout_timing and dropout_timed and dropout_timing_t0 is not None:
                transition_noise_wall_s += (time.perf_counter() - dropout_timing_t0)

            x_next = env["x_generator"](env_in, generators_for_noise=env_noise_generators)
            state_next = (1.0 - env["alpha"][:, None]) * state_t + env["alpha"][:, None] * x_next
            state_noise_timing_t0 = time.perf_counter() if profile_rollout_timing else None
            state_noise_timed = False
            if strict_seed_mode and state_noise_generators is not None:
                state_noise_timed = True
                state_noise_t = _draw_step_randn_with_optional_generators(
                    state_noise_generators,
                    state_dim,
                )
                state_next = state_next + state_noise_t * env["state_noise_std"][:, None]
            elif noise_streaming_mode and state_noise_block is not None and noise_block_idx is not None:
                state_noise_timed = True
                state_next = state_next + state_noise_block[noise_block_idx] * env["state_noise_std"][:, None]
            elif state_noise is not None:
                state_noise_timed = True
                state_next = state_next + state_noise[t] * env["state_noise_std"][:, None]
            if profile_rollout_timing and state_noise_timed and state_noise_timing_t0 is not None:
                transition_noise_wall_s += (time.perf_counter() - state_noise_timing_t0)
            state_next = self._apply_state_postprocess(
                state_next_raw=state_next,
                state_prev=state_t,
                state_clip=env["state_clip"],
                state_highway_enabled=env.get("state_highway_enabled", False),
                state_highway_lambda=env.get("state_highway_lambda", 0.0),
            )
            aev4_step_aux = None
            if aev4_reg_enabled:
                state_next, aev4_step_aux = self._apply_aev4_state_update(
                    state_prev=state_t,
                    state_next_post=state_next,
                    aev4_cfg=aev4_cfg,
                )
            state_delta = state_next - state_t
            if aev2_enabled and (aev2_prev_delta is not None):
                self._aev2_update_accumulator(aev2_acc, aev2_prev_delta, state_delta, aev2_cfg)
            aev2_prev_delta = state_delta
            if aev3_enabled and (aev3_prev_delta is not None):
                self._aev3_update_accumulator(aev3_acc, aev3_prev_delta, state_delta, aev3_cfg)
            aev3_prev_delta = state_delta
            if aev4_reg_enabled and (aev4_prev_delta is not None):
                self._aev4_update_accumulator(aev4_acc, aev4_prev_delta, state_delta, aev4_cfg, step_aux=aev4_step_aux)
            aev4_prev_delta = state_delta
            if transition_cuda_start is not None:
                transition_cuda_end = torch.cuda.Event(enable_timing=True)
                transition_cuda_end.record()
                transition_cuda_pairs.append((transition_cuda_start, transition_cuda_end))

            if tbptt_window_active:
                if y_steps is not None:
                    y_steps[t] = reward_next.detach()
                tbptt_reward_buffer.append(reward_next)
            else:
                if y_steps is not None:
                    y_steps[t] = reward_next
            if collect_runtime_info:
                reward_values[t] = reward_next.detach()
                state_abs_max[t] = state_next.detach().abs().amax(dim=1)

            state_t = state_next
            action_t = action_next
            reward_t = reward_next
            reward_mask_t = reward_mask_next

            if tbptt_window_active:
                is_window_end = (len(tbptt_reward_buffer) >= tbptt_window_size) or (t == (n_samples - 1))
                if is_window_end:
                    rewards_window = torch.stack(tbptt_reward_buffer, dim=0)
                    tbptt_reward_buffer = []
                    if t < (n_samples - 1):
                        state_t = state_t.detach()
                        action_t = action_t.detach()
                        reward_t = reward_t.detach()
                        reward_mask_t = reward_mask_t.detach()
                        env_in = env_in.detach()
                        if aev2_prev_delta is not None:
                            aev2_prev_delta = aev2_prev_delta.detach()
                        if aev3_prev_delta is not None:
                            aev3_prev_delta = aev3_prev_delta.detach()
                        if aev4_prev_delta is not None:
                            aev4_prev_delta = aev4_prev_delta.detach()
                        cache = self._detach_policy_cache(cache, clone_tensors=(tbptt_reward_sink is None))
                    if tbptt_reward_sink is not None:
                        if aev2_streaming_sink or aev3_streaming_sink or aev4_streaming_sink:
                            payload_aux = {}
                            if aev2_streaming_sink:
                                aev2_window_summary = self._aev2_finalize_accumulator(
                                    aev2_acc,
                                    device=device,
                                    dtype=torch.float32,
                                    detach_penalty=False,
                                )
                                payload_aux["aev2"] = aev2_window_summary
                                self._aev2_accumulate_window_summary(aev2_det_acc, aev2_window_summary, device=device)
                                aev2_acc = self._aev2_new_accumulator(aev2_enabled, device=device, dtype=torch.float32)
                            if aev3_streaming_sink:
                                aev3_window_summary = self._aev3_finalize_accumulator(
                                    aev3_acc,
                                    device=device,
                                    dtype=torch.float32,
                                    detach_penalty=False,
                                )
                                payload_aux["aev3"] = aev3_window_summary
                                self._aev3_accumulate_window_summary(aev3_det_acc, aev3_window_summary, device=device)
                                aev3_acc = self._aev3_new_accumulator(
                                    aev3_enabled,
                                    device=device,
                                    dtype=torch.float32,
                                    aev3_cfg=aev3_cfg,
                                )
                            if aev4_streaming_sink:
                                aev4_window_summary = self._aev4_finalize_accumulator(
                                    aev4_acc,
                                    device=device,
                                    dtype=torch.float32,
                                    detach_penalty=False,
                                )
                                payload_aux["aev4"] = aev4_window_summary
                                self._aev4_accumulate_window_summary(aev4_det_acc, aev4_window_summary, device=device)
                                aev4_acc = self._aev4_new_accumulator(
                                    aev4_reg_enabled,
                                    device=device,
                                    dtype=torch.float32,
                                    aev4_cfg=aev4_cfg,
                                )
                            if ("aev2" in payload_aux) and (len(payload_aux) == 1):
                                tbptt_reward_sink((rewards_window, payload_aux["aev2"]))
                            else:
                                tbptt_reward_sink((rewards_window, payload_aux))
                        else:
                            tbptt_reward_sink(rewards_window)

        if collect_runtime_info:
            reward_drop_frac = reward_drop_count.to(dtype=torch.float32) / float(max(1, n_samples))
            infos = self._build_vectorized_runtime_info(
                env=env,
                reward_values=reward_values,
                state_abs_max=state_abs_max,
                reward_drop_frac=reward_drop_frac,
                single_eval_pos=single_eval_pos,
            )
        else:
            infos = [None] * batch_size
        rollout_profile = None
        if profile_rollout_breakdown_cuda and (policy_cuda_pairs or transition_cuda_pairs):
            torch.cuda.synchronize(device=device_obj)
            policy_cuda_ms = float(sum(start.elapsed_time(end) for start, end in policy_cuda_pairs))
            transition_cuda_ms = float(sum(start.elapsed_time(end) for start, end in transition_cuda_pairs))
            noise_mode = "strict_seed" if strict_seed_mode else ("block_stream" if noise_streaming_mode else "full_prealloc")
            rollout_profile = {
                "policy_cuda_ms": policy_cuda_ms,
                "transition_cuda_ms": transition_cuda_ms,
                "steps": int(n_samples),
                "batch_size": int(batch_size),
                "noise_mode": noise_mode,
                "noise_block_size": int(noise_block_size) if noise_streaming_mode else 0,
                "transition_noise_wall_ms": float(transition_noise_wall_s * 1000.0),
            }
        self.last_rollout_profile = rollout_profile
        if aev2_enabled:
            if aev2_streaming_sink:
                self.last_rollout_v2 = self._aev2_finalize_accumulator(
                    aev2_det_acc,
                    device=device,
                    dtype=torch.float32,
                    detach_penalty=True,
                )
            else:
                self.last_rollout_v2 = self._aev2_finalize_accumulator(
                    aev2_acc,
                    device=device,
                    dtype=torch.float32,
                    detach_penalty=False,
                )
            self.last_rollout_v2["lambda"] = float(aev2_cfg.get("lambda", 0.0))
            self.last_rollout_v2["gain_lo"] = float(aev2_cfg.get("gain_lo", 0.0))
            self.last_rollout_v2["gain_hi"] = float(aev2_cfg.get("gain_hi", 0.0))
        else:
            self.last_rollout_v2 = None
        if aev3_enabled:
            if aev3_streaming_sink:
                self.last_rollout_v3 = self._aev3_finalize_accumulator(
                    aev3_det_acc,
                    device=device,
                    dtype=torch.float32,
                    detach_penalty=True,
                )
            else:
                self.last_rollout_v3 = self._aev3_finalize_accumulator(
                    aev3_acc,
                    device=device,
                    dtype=torch.float32,
                    detach_penalty=False,
                )
        else:
            self.last_rollout_v3 = None
        if aev4_reg_enabled:
            if aev4_streaming_sink:
                self.last_rollout_v4 = self._aev4_finalize_accumulator(
                    aev4_det_acc,
                    device=device,
                    dtype=torch.float32,
                    detach_penalty=True,
                )
            else:
                self.last_rollout_v4 = self._aev4_finalize_accumulator(
                    aev4_acc,
                    device=device,
                    dtype=torch.float32,
                    detach_penalty=False,
                )
        else:
            self.last_rollout_v4 = None
        if y_steps is None:
            y_steps = torch.empty((0, batch_size), device=device, dtype=torch.float32)
        return x_steps, y_steps, infos

    def _rollout_family_group_vectorized_with_policy(
        self,
        h_list,
        policy_step_fn,
        n_samples,
        num_features,
        single_eval_pos,
        device,
        collect_x=True,
        collect_runtime_info=True,
        env_rng_seeds=None,
        rollout_rng_seeds=None,
        tbptt_window=None,
        tbptt_reward_sink=None,
        tbptt_reward_sink_supports_aux=False,
        store_rewards=True,
    ):
        n_samples = int(n_samples)
        batch_size = int(len(h_list))
        num_features = int(num_features)
        collect_runtime_info = bool(collect_runtime_info)
        if batch_size <= 0:
            raise ValueError("h_list must be non-empty for family-group rollout")

        # Preserve per-sample reward-dropout sampling order independent of
        # structure subgrouping.
        h_list_effective = []
        for h in h_list:
            h_eff = dict(h)
            if bool(h_eff.get("reward_dropout_enabled", True)) and bool(h_eff.get("reward_dropout_randomize", True)):
                sampled_ratio = float(self._sample_reward_dropout_ratio(h_eff))
                h_eff["reward_dropout_randomize"] = False
                h_eff["reward_dropout_ratio"] = sampled_ratio
            h_list_effective.append(h_eff)

        family_groups = {}
        transition_inner_grouping = str(getattr(self, "transition_inner_grouping", "family")).strip().lower()
        if transition_inner_grouping not in {"family", "structure", "pow2", "pow2_no_depth"}:
            transition_inner_grouping = "family"
        transition_inner_min_bucket = int(max(0, int(getattr(self, "transition_inner_min_bucket", 0) or 0)))
        for bi, h in enumerate(h_list_effective):
            family = self._normalize_family(h.get("family", "scm"))
            sig = (family,)
            family_groups.setdefault(sig, []).append((bi, h))

        rollout_generators = self._make_generators_from_seeds(rollout_rng_seeds, batch_size, device)
        profile_rollout_breakdown_flag = str(os.environ.get("TICL_PROFILE_ROLLOUT_BREAKDOWN", "")).strip().lower()
        profile_rollout_breakdown = profile_rollout_breakdown_flag in {"1", "true", "yes", "on"}
        profile_rollout_timing_flag = str(os.environ.get("TICL_PROFILE_ROLLOUT_TIMING", "")).strip().lower()
        profile_rollout_timing = profile_rollout_timing_flag in {"1", "true", "yes", "on"}
        device_obj = device if isinstance(device, torch.device) else torch.device(str(device))
        profile_rollout_breakdown_cuda = bool(
            profile_rollout_breakdown
            and device_obj.type == "cuda"
            and torch.cuda.is_available()
        )
        transition_stream_fusion_flag = str(os.environ.get("TICL_POLICY_TRANSITION_STREAM_FUSION", "1")).strip().lower()
        base_transition_stream_fusion = bool(
            transition_stream_fusion_flag in {"1", "true", "yes", "on"}
            and device_obj.type == "cuda"
            and torch.cuda.is_available()
            and rollout_generators is None
        )
        try:
            transition_stream_fusion_max_groups = int(
                max(1, int(os.environ.get("TICL_POLICY_TRANSITION_STREAM_FUSION_MAX_GROUPS", "2")))
            )
        except Exception:
            transition_stream_fusion_max_groups = 2
        async_group_commit_flag = str(
            os.environ.get("TICL_POLICY_ASYNC_GROUP_COMMIT_IN_STREAM", "auto")
        ).strip().lower()
        if async_group_commit_flag in {"1", "true", "yes", "on"}:
            async_group_commit_in_stream = True
        elif async_group_commit_flag in {"0", "false", "no", "off"}:
            async_group_commit_in_stream = False
        else:
            # Auto: when transition-group stream fusion is active, commit each
            # group update in its stream and only synchronize once per step.
            async_group_commit_in_stream = bool(base_transition_stream_fusion)
        tbptt_window_active = False
        tbptt_window_size = n_samples
        if tbptt_window is not None:
            w = int(tbptt_window)
            if 0 < w < n_samples:
                tbptt_window_active = True
                tbptt_window_size = w
        policy_cuda_pairs = []
        transition_cuda_pairs = []
        policy_wall_s = 0.0
        transition_wall_s = 0.0
        transition_y_wall_s = 0.0
        transition_x_wall_s = 0.0
        transition_group_wall_s = 0.0
        transition_group_launch_wall_s = 0.0
        transition_group_sync_wall_s = 0.0
        transition_env_pack_wall_s = 0.0
        transition_state_update_wall_s = 0.0
        transition_noise_wall_s = 0.0
        transition_fused_wall_s = 0.0
        transition_fused_launch_wall_s = 0.0
        transition_gp_first_projection_wall_s = 0.0
        transition_gp_second_projection_wall_s = 0.0
        transition_gp_projection_call_count = 0
        transition_gp_rff_fused_call_count = 0
        transition_gp_profile_group_count = 0
        transition_gp_profile_sync_group_count = 0
        transition_gp_shared_total_wall_s = 0.0
        transition_gp_shared_core_wall_s = 0.0
        transition_gp_shared_noise_wall_s = 0.0
        transition_gp_shared_checkpoint_wall_s = 0.0
        transition_gp_shared_post_wall_s = 0.0
        transition_gp_shared_call_count = 0
        transition_packed_env_input_group_count = 0
        transition_packed_env_input_call_count = 0
        transition_only_build_group_count = 0
        transition_only_skipped_generator_count = 0
        transition_fused_call_count = 0
        transition_fused_group_count = 0
        transition_checkpoint_call_count = 0
        transition_group_work_actual = 0.0
        transition_group_work_padded = 0.0
        transition_group_max_batch = 0
        transition_setup_wall_s = 0.0
        transition_family_build_wall_s = 0.0
        transition_generator_build_wall_s = 0.0
        transition_gp_shared_build_wall_s = 0.0

        state_dims = torch.empty((batch_size,), device=device, dtype=torch.long)
        obs_dims = torch.empty((batch_size,), device=device, dtype=torch.long)
        action_dims = torch.empty((batch_size,), device=device, dtype=torch.long)
        noise_dims = torch.empty((batch_size,), device=device, dtype=torch.long)
        zero_pad_dims = torch.empty((batch_size,), device=device, dtype=torch.long)
        obs_slot_dims = torch.empty((batch_size,), device=device, dtype=torch.long)
        action_slot_dims = torch.empty((batch_size,), device=device, dtype=torch.long)

        init_state_std = torch.empty((batch_size,), device=device, dtype=torch.float32)
        init_action_std = torch.empty((batch_size,), device=device, dtype=torch.float32)
        state_noise_std = torch.empty((batch_size,), device=device, dtype=torch.float32)
        action_noise_train_std = torch.empty((batch_size,), device=device, dtype=torch.float32)
        action_noise_eval_std = torch.empty((batch_size,), device=device, dtype=torch.float32)
        reward_scale = torch.empty((batch_size,), device=device, dtype=torch.float32)
        reward_clip = torch.empty((batch_size,), device=device, dtype=torch.float32)
        alpha = torch.empty((batch_size,), device=device, dtype=torch.float32)
        state_clip = torch.empty((batch_size,), device=device, dtype=torch.float32)
        state_highway_enabled = torch.empty((batch_size,), device=device, dtype=torch.bool)
        state_highway_lambda = torch.empty((batch_size,), device=device, dtype=torch.float32)
        aev4_enabled = torch.empty((batch_size,), device=device, dtype=torch.bool)
        aev4_highway_ratio = torch.empty((batch_size,), device=device, dtype=torch.float32)
        aev4_update_scale = torch.empty((batch_size,), device=device, dtype=torch.float32)
        aev4_update_clip = torch.empty((batch_size,), device=device, dtype=torch.float32)
        reward_dropout_enabled = torch.empty((batch_size,), device=device, dtype=torch.bool)
        reward_dropout_impute_zero = torch.empty((batch_size,), device=device, dtype=torch.bool)
        reward_dropout_ratio = torch.empty((batch_size,), device=device, dtype=torch.float32)
        family_list = [None] * batch_size

        transition_groups = []
        transition_family_group_count = int(len(family_groups))
        for family_group in family_groups.values():
            transition_bucket_groups = {}
            balanced_bucket_count = self._resolve_transition_balanced_bucket_count(transition_inner_grouping)
            if balanced_bucket_count is not None:
                raw_transition_bucket_groups = self._build_balanced_transition_bucket_groups(
                    family_group=family_group,
                    bucket_count=balanced_bucket_count,
                )
                if transition_inner_min_bucket > 1:
                    family = self._normalize_family(family_group[0][1].get("family", "scm"))
                    merged_family_bucket = []
                    for sig, bucket_group in raw_transition_bucket_groups.items():
                        if len(bucket_group) >= transition_inner_min_bucket:
                            transition_bucket_groups[sig] = bucket_group
                        else:
                            merged_family_bucket.extend(bucket_group)
                    if merged_family_bucket:
                        transition_bucket_groups[(family, "__fallback__")] = merged_family_bucket
                else:
                    transition_bucket_groups = raw_transition_bucket_groups
            elif transition_inner_grouping != "family":
                raw_transition_bucket_groups = {}
                for global_idx, h in family_group:
                    sig = self._environment_transition_bucket_signature(h, transition_inner_grouping)
                    raw_transition_bucket_groups.setdefault(sig, []).append((global_idx, h))
                if transition_inner_min_bucket > 1:
                    family = self._normalize_family(family_group[0][1].get("family", "scm"))
                    merged_family_bucket = []
                    for sig, bucket_group in raw_transition_bucket_groups.items():
                        if len(bucket_group) >= transition_inner_min_bucket:
                            transition_bucket_groups[sig] = bucket_group
                        else:
                            merged_family_bucket.extend(bucket_group)
                    if merged_family_bucket:
                        transition_bucket_groups[(family, "__fallback__")] = merged_family_bucket
                else:
                    transition_bucket_groups = raw_transition_bucket_groups
            else:
                family = self._normalize_family(family_group[0][1].get("family", "scm"))
                transition_bucket_groups[(family,)] = list(family_group)

            for group in transition_bucket_groups.values():
                group_indices = [idx for idx, _ in group]
                group_h_list = [h for _, h in group]
                group_env_seeds = (
                    [env_rng_seeds[idx] for idx in group_indices]
                    if env_rng_seeds is not None
                    else None
                )
                auto_elide_non_transition_generators = bool(
                    self.fused_transition_generator
                    and device_obj.type == "cuda"
                    and group_env_seeds is None
                )
                setup_t0 = time.perf_counter() if profile_rollout_timing else None
                env_batch = self._sample_environment_family_coarse_batch(
                    h_list=group_h_list,
                    device=device,
                    rng_seeds=group_env_seeds,
                    build_x_generator=(not auto_elide_non_transition_generators),
                    build_y_generator=(not auto_elide_non_transition_generators),
                    build_policy_generator=False,
                    prefer_transition_only=False,
                    preserve_skipped_generator_rng=True,
                )
                if setup_t0 is not None:
                    transition_setup_wall_s += (time.perf_counter() - setup_t0)
                build_profile = env_batch.get("_build_profile", None)
                if isinstance(build_profile, dict):
                    transition_family_build_wall_s += float(build_profile.get("family_build_wall_s", 0.0) or 0.0)
                    transition_generator_build_wall_s += float(
                        build_profile.get("transition_generator_build_wall_s", 0.0) or 0.0
                    )
                    transition_gp_shared_build_wall_s += float(
                        build_profile.get("gp_shared_transition_build_wall_s", 0.0) or 0.0
                    )
                    transition_only_build_group_count += int(
                        build_profile.get("transition_only_build_enabled", 0) or 0
                    )
                    transition_only_skipped_generator_count += int(
                        build_profile.get("non_transition_generator_skip_count", 0) or 0
                    )
                group_idx = torch.tensor(group_indices, device=device, dtype=torch.long)
                group_bs = int(group_idx.numel())
                transition_group_max_batch = max(transition_group_max_batch, group_bs)
                work_actual, work_padded = self._estimate_transition_group_work(group_h_list)
                transition_group_work_actual += float(work_actual)
                transition_group_work_padded += float(max(work_actual, work_padded))

                state_dims_group = env_batch.get("state_dim_per_sample", None)
                obs_dims_group = env_batch.get("obs_dim_per_sample", None)
                action_dims_group = env_batch.get("action_dim_per_sample", None)
                noise_dims_group = env_batch.get("noise_dim_per_sample", None)
                zero_pad_dims_group = env_batch.get("zero_pad_dim_per_sample", None)
                obs_slot_dims_group = env_batch.get("obs_slot_dim_per_sample", None)
                action_slot_dims_group = env_batch.get("action_slot_dim_per_sample", None)
                if state_dims_group is None:
                    state_dims_group = torch.full((group_bs,), int(env_batch["state_dim"]), device=device, dtype=torch.long)
                if obs_dims_group is None:
                    obs_dims_group = torch.full((group_bs,), int(env_batch["obs_dim"]), device=device, dtype=torch.long)
                if action_dims_group is None:
                    action_dims_group = torch.full((group_bs,), int(env_batch["action_dim"]), device=device, dtype=torch.long)
                if noise_dims_group is None:
                    noise_dims_group = torch.full((group_bs,), int(env_batch["noise_dim"]), device=device, dtype=torch.long)
                if zero_pad_dims_group is None:
                    zero_pad_dims_group = torch.full((group_bs,), int(env_batch["zero_pad_dim"]), device=device, dtype=torch.long)
                if obs_slot_dims_group is None:
                    obs_slot_dims_group = torch.full((group_bs,), int(env_batch["obs_slot_dim"]), device=device, dtype=torch.long)
                if action_slot_dims_group is None:
                    action_slot_dims_group = torch.full((group_bs,), int(env_batch["action_slot_dim"]), device=device, dtype=torch.long)

                state_dim_g = int(state_dims_group.max().item())
                obs_dim_g = int(obs_dims_group.max().item())
                action_dim_g = int(action_dims_group.max().item())
                noise_dim_g = int(noise_dims_group.max().item())
                zero_pad_dim_g = int(zero_pad_dims_group.max().item())

                state_dims[group_idx] = state_dims_group
                obs_dims[group_idx] = obs_dims_group
                action_dims[group_idx] = action_dims_group
                noise_dims[group_idx] = noise_dims_group
                zero_pad_dims[group_idx] = zero_pad_dims_group
                obs_slot_dims[group_idx] = obs_slot_dims_group
                action_slot_dims[group_idx] = action_slot_dims_group

                init_state_std[group_idx] = env_batch["init_state_std"]
                init_action_std[group_idx] = env_batch["init_action_std"]
                state_noise_std[group_idx] = env_batch["state_noise_std"]
                action_noise_train_std[group_idx] = env_batch["action_noise_train_std"]
                action_noise_eval_std[group_idx] = env_batch["action_noise_eval_std"]
                reward_scale[group_idx] = env_batch["reward_scale"]
                reward_clip[group_idx] = env_batch["reward_clip"]
                alpha[group_idx] = env_batch["alpha"]
                state_clip[group_idx] = env_batch["state_clip"]
                state_highway_enabled[group_idx] = env_batch["state_highway_enabled"]
                state_highway_lambda[group_idx] = env_batch["state_highway_lambda"]
                aev4_enabled[group_idx] = env_batch.get("aev4_enabled", False)
                aev4_highway_ratio[group_idx] = env_batch.get("aev4_highway_ratio", 0.25)
                aev4_update_scale[group_idx] = env_batch.get("aev4_update_scale", 0.12)
                aev4_update_clip[group_idx] = env_batch.get("aev4_update_clip", 0.0)
                reward_dropout_enabled[group_idx] = env_batch["reward_dropout_enabled"]
                reward_dropout_impute_zero[group_idx] = env_batch["reward_dropout_impute_zero"]
                reward_dropout_ratio[group_idx] = env_batch["reward_dropout_ratio"]
                for global_idx in group_indices:
                    family_list[global_idx] = str(env_batch["family"])

                group_rollout_generators = None
                if rollout_generators is not None:
                    group_rollout_generators = [rollout_generators[idx] for idx in group_indices]
                transition_generator = env_batch.get("transition_generator", None)
                use_fused_transition = bool(callable(transition_generator))
                if use_fused_transition:
                    transition_fused_group_count += 1
                gp_projection_profile_capable = bool(
                    callable(getattr(transition_generator, "_consume_gp_projection_profile", None))
                )
                group_env_total_dim = state_dim_g + obs_dim_g + action_dim_g + noise_dim_g + zero_pad_dim_g
                packed_input_enabled = bool(
                    use_fused_transition and bool(getattr(transition_generator, "_prefers_packed_env_input", False))
                )
                packed_input_cap = int(
                    max(
                        1,
                        int(
                            getattr(
                                transition_generator,
                                "_packed_input_cap",
                                int((state_dims_group + obs_dims_group + action_dims_group + noise_dims_group).max().item()),
                            )
                        ),
                    )
                )
                packed_state_mask = None
                packed_obs_rows = None
                packed_obs_dst_cols = None
                packed_obs_src_cols = None
                packed_action_rows = None
                packed_action_dst_cols = None
                packed_action_src_cols = None
                packed_noise_rows = None
                packed_noise_dst_cols = None
                packed_noise_src_cols = None
                packed_input_buf = None
                if packed_input_enabled:
                    transition_packed_env_input_group_count += 1
                    state_cols = torch.arange(state_dim_g, device=device, dtype=torch.long).unsqueeze(0)
                    packed_state_mask = (state_cols < state_dims_group.unsqueeze(1)).to(dtype=torch.float32)
                    packed_obs_rows, packed_obs_dst_cols, packed_obs_src_cols = self._build_segment_write_map(
                        state_dims_group,
                        obs_dims_group,
                        obs_dim_g,
                    )
                    packed_action_rows, packed_action_dst_cols, packed_action_src_cols = self._build_segment_write_map(
                        state_dims_group + obs_dims_group,
                        action_dims_group,
                        action_dim_g,
                    )
                    packed_noise_rows, packed_noise_dst_cols, packed_noise_src_cols = self._build_segment_write_map(
                        state_dims_group + obs_dims_group + action_dims_group,
                        noise_dims_group,
                        noise_dim_g,
                    )
                    packed_input_buf = torch.zeros((group_bs, packed_input_cap), device=device, dtype=torch.float32)
                transition_checkpoint_enabled = int(
                    bool(getattr(transition_generator, "_envgen_checkpoint_enabled", False))
                )
                transition_groups.append(
                    {
                        "indices": group_idx,
                        "env": env_batch,
                        "batch_size": group_bs,
                        "state_dim": state_dim_g,
                        "obs_dim": obs_dim_g,
                        "action_dim": action_dim_g,
                        "noise_dim": noise_dim_g,
                        "env_obs_start": state_dim_g,
                        "env_action_start": state_dim_g + obs_dim_g,
                        "env_noise_start": state_dim_g + obs_dim_g + action_dim_g,
                        "env_in": torch.zeros((group_bs, group_env_total_dim), device=device, dtype=torch.float32),
                        "packed_input_enabled": packed_input_enabled,
                        "packed_input_cap": packed_input_cap,
                        "packed_input": packed_input_buf,
                        "packed_state_mask": packed_state_mask,
                        "packed_obs_rows": packed_obs_rows,
                        "packed_obs_dst_cols": packed_obs_dst_cols,
                        "packed_obs_src_cols": packed_obs_src_cols,
                        "packed_action_rows": packed_action_rows,
                        "packed_action_dst_cols": packed_action_dst_cols,
                        "packed_action_src_cols": packed_action_src_cols,
                        "packed_noise_rows": packed_noise_rows,
                        "packed_noise_dst_cols": packed_noise_dst_cols,
                        "packed_noise_src_cols": packed_noise_src_cols,
                        "rollout_generators": group_rollout_generators,
                        "state_noise_active": bool(torch.any(env_batch["state_noise_std"] > 0).item()),
                        "family": str(env_batch["family"]),
                        "gp_projection_profile_capable": gp_projection_profile_capable,
                        "transition_generator": transition_generator,
                        "use_fused_transition": use_fused_transition,
                        "transition_checkpoint_enabled": transition_checkpoint_enabled,
                        "stream_transition": None,
                        "stream_y": None,
                        "stream_x": None,
                    }
                )

        # Keep transition subgroups contiguous in memory to avoid per-step
        # index_select/scatter overhead in the rollout hot loop.
        identity_perm = torch.arange(batch_size, device=device, dtype=torch.long)
        perm = torch.cat([g["indices"] for g in transition_groups], dim=0)
        if int(perm.numel()) != batch_size:
            raise RuntimeError("family-group rollout internal permutation size mismatch")
        keep_policy_order = bool(len(transition_groups) > 2)
        needs_unpermute = False
        inv_perm = identity_perm
        if (not keep_policy_order) and (not bool(torch.equal(perm, identity_perm))):
            needs_unpermute = True
            inv_perm = torch.empty_like(perm)
            inv_perm.scatter_(0, perm, identity_perm)
            perm_cpu = perm.detach().cpu().tolist()
            state_dims = state_dims.index_select(0, perm)
            obs_dims = obs_dims.index_select(0, perm)
            action_dims = action_dims.index_select(0, perm)
            noise_dims = noise_dims.index_select(0, perm)
            zero_pad_dims = zero_pad_dims.index_select(0, perm)
            obs_slot_dims = obs_slot_dims.index_select(0, perm)
            action_slot_dims = action_slot_dims.index_select(0, perm)
            init_state_std = init_state_std.index_select(0, perm)
            init_action_std = init_action_std.index_select(0, perm)
            state_noise_std = state_noise_std.index_select(0, perm)
            action_noise_train_std = action_noise_train_std.index_select(0, perm)
            action_noise_eval_std = action_noise_eval_std.index_select(0, perm)
            reward_scale = reward_scale.index_select(0, perm)
            reward_clip = reward_clip.index_select(0, perm)
            alpha = alpha.index_select(0, perm)
            state_clip = state_clip.index_select(0, perm)
            state_highway_enabled = state_highway_enabled.index_select(0, perm)
            state_highway_lambda = state_highway_lambda.index_select(0, perm)
            aev4_enabled = aev4_enabled.index_select(0, perm)
            aev4_highway_ratio = aev4_highway_ratio.index_select(0, perm)
            aev4_update_scale = aev4_update_scale.index_select(0, perm)
            aev4_update_clip = aev4_update_clip.index_select(0, perm)
            reward_dropout_enabled = reward_dropout_enabled.index_select(0, perm)
            reward_dropout_impute_zero = reward_dropout_impute_zero.index_select(0, perm)
            reward_dropout_ratio = reward_dropout_ratio.index_select(0, perm)
            family_list = [family_list[int(i)] for i in perm_cpu]
            if rollout_generators is not None:
                rollout_generators = [rollout_generators[int(i)] for i in perm_cpu]

        group_cursor = 0
        for group in transition_groups:
            group_bs = int(group["indices"].numel())
            group["start"] = int(group_cursor)
            group["end"] = int(group_cursor + group_bs)
            if (not keep_policy_order) and rollout_generators is not None:
                group["rollout_generators"] = rollout_generators[group_cursor: group_cursor + group_bs]
            elif (not keep_policy_order):
                group["rollout_generators"] = None
            group_cursor += group_bs
        if group_cursor != batch_size:
            raise RuntimeError("family-group rollout internal subgroup cursor mismatch")
        transition_stream_fusion = bool(
            (not keep_policy_order)
            and base_transition_stream_fusion
            and len(transition_groups) > 1
            and len(transition_groups) <= int(transition_stream_fusion_max_groups)
        )
        if transition_stream_fusion:
            for group in transition_groups:
                if bool(group.get("use_fused_transition", False)):
                    group["stream_transition"] = torch.cuda.Stream(device=device_obj)
                else:
                    group["stream_y"] = torch.cuda.Stream(device=device_obj)
                    group["stream_x"] = torch.cuda.Stream(device=device_obj)

        max_state_dim = int(state_dims.max().item())
        max_obs_dim = int(obs_dims.max().item())
        max_action_dim = int(action_dims.max().item())
        max_noise_dim = int(noise_dims.max().item())

        for group in transition_groups:
            state_dim_g = int(group["state_dim"])
            if keep_policy_order:
                group["slice"] = None
                group["reward_scale_view"] = reward_scale.index_select(0, group["indices"])
                group["alpha_view"] = alpha.index_select(0, group["indices"]).unsqueeze(1)
            else:
                start = int(group["start"])
                end = int(group["end"])
                group["slice"] = slice(start, end)
                group["reward_scale_view"] = reward_scale[start:end]
                group["alpha_view"] = alpha[start:end].unsqueeze(1)

        state_idx = torch.arange(max_state_dim, device=device, dtype=torch.long)
        obs_idx = torch.arange(max_obs_dim, device=device, dtype=torch.long)
        action_idx = torch.arange(max_action_dim, device=device, dtype=torch.long)
        noise_idx = torch.arange(max_noise_dim, device=device, dtype=torch.long)

        state_mask = (state_idx.unsqueeze(0) < state_dims.unsqueeze(1)).to(dtype=torch.float32)
        obs_mask = (obs_idx.unsqueeze(0) < obs_dims.unsqueeze(1)).to(dtype=torch.float32)
        action_mask = (action_idx.unsqueeze(0) < action_dims.unsqueeze(1)).to(dtype=torch.float32)
        noise_mask = (noise_idx.unsqueeze(0) < noise_dims.unsqueeze(1)).to(dtype=torch.float32)

        dropout_active = reward_dropout_enabled & (reward_dropout_ratio > 0.0)

        state_t = self._stack_randn_with_generators(
            rollout_generators,
            (batch_size, max_state_dim),
            device=device,
            dtype=torch.float32,
        ) * init_state_std[:, None]
        state_t = state_t * state_mask
        action_t = self._stack_randn_with_generators(
            rollout_generators,
            (batch_size, max_action_dim),
            device=device,
            dtype=torch.float32,
        ) * init_action_std[:, None]
        action_t = action_t * action_mask
        reward_t = torch.zeros((batch_size,), device=device, dtype=torch.float32)
        reward_mask_t = torch.ones((batch_size,), device=device, dtype=torch.float32)
        cache = None

        x_steps = (
            torch.empty((n_samples, batch_size, num_features), device=device, dtype=torch.float32)
            if collect_x
            else None
        )
        y_steps = (
            torch.empty((n_samples, batch_size), device=device, dtype=torch.float32)
            if bool(store_rewards)
            else None
        )
        state_abs_max = (
            torch.empty((n_samples, batch_size), device=device, dtype=torch.float32)
            if collect_runtime_info
            else None
        )
        reward_values = (
            torch.empty((n_samples, batch_size), device=device, dtype=torch.float32)
            if collect_runtime_info
            else None
        )
        reward_drop_count = (
            torch.zeros((batch_size,), device=device, dtype=torch.int64)
            if collect_runtime_info
            else None
        )

        tbptt_reward_buffer = [] if tbptt_window_active else None
        aev2_cfg = self._resolve_aev2_config()
        aev2_enabled = bool(aev2_cfg.get("enabled", False))
        aev2_prev_delta = None
        aev3_cfg = self._resolve_aev3_config()
        aev3_enabled = bool(aev3_cfg.get("enabled", False))
        aev3_prev_delta = None
        aev4_cfg = self._resolve_aev4_config()
        aev4_reg_enabled = bool(aev4_cfg.get("enabled", False))
        aev4_prev_delta = None
        aev2_streaming_sink = bool(
            aev2_enabled
            and tbptt_window_active
            and (tbptt_reward_sink is not None)
            and bool(tbptt_reward_sink_supports_aux)
        )
        aev3_streaming_sink = bool(
            aev3_enabled
            and tbptt_window_active
            and (tbptt_reward_sink is not None)
            and bool(tbptt_reward_sink_supports_aux)
        )
        aev4_streaming_sink = bool(
            aev4_reg_enabled
            and tbptt_window_active
            and (tbptt_reward_sink is not None)
            and bool(tbptt_reward_sink_supports_aux)
        )
        aev2_acc = self._aev2_new_accumulator(aev2_enabled, device=device, dtype=torch.float32)
        aev2_det_acc = self._aev2_new_accumulator(aev2_enabled, device=device, dtype=torch.float32)
        aev3_acc = self._aev3_new_accumulator(aev3_enabled, device=device, dtype=torch.float32, aev3_cfg=aev3_cfg)
        aev3_det_acc = self._aev3_new_accumulator(
            aev3_enabled, device=device, dtype=torch.float32, aev3_cfg=aev3_cfg
        )
        aev4_acc = self._aev4_new_accumulator(
            aev4_reg_enabled, device=device, dtype=torch.float32, aev4_cfg=aev4_cfg
        )
        aev4_det_acc = self._aev4_new_accumulator(
            aev4_reg_enabled, device=device, dtype=torch.float32, aev4_cfg=aev4_cfg
        )

        strict_seed_mode = rollout_generators is not None
        noise_block_size = 0
        noise_streaming_mode = False
        noise_block_start = 0
        noise_block_end = 0
        transition_noise_block = None
        action_noise_train_block = None
        action_noise_eval_block = None
        state_noise_block = None
        dropout_draws_block = None
        if not strict_seed_mode:
            noise_block_size = self._resolve_rollout_noise_block_size(n_samples)
            noise_streaming_mode = bool(noise_block_size > 0)

        transition_noise = None
        action_noise_train = None
        action_noise_eval = None
        state_noise = None
        dropout_draws = None

        def _refresh_noise_block(block_start_idx):
            nonlocal noise_block_start, noise_block_end
            nonlocal transition_noise_block, action_noise_train_block, action_noise_eval_block
            nonlocal state_noise_block, dropout_draws_block
            block_start_idx = int(block_start_idx)
            block_len = int(min(noise_block_size, n_samples - block_start_idx))
            noise_block_start = block_start_idx
            noise_block_end = block_start_idx + block_len
            transition_noise_block = self._stack_randn_with_generators(
                rollout_generators,
                (batch_size, block_len, max_noise_dim),
                device=device,
                dtype=torch.float32,
            ).transpose(0, 1) * noise_mask.unsqueeze(0)
            if torch.any(action_noise_train_std > 0):
                action_noise_train_block = self._stack_randn_with_generators(
                    rollout_generators,
                    (batch_size, block_len, max_action_dim),
                    device=device,
                    dtype=torch.float32,
                ).transpose(0, 1) * action_mask.unsqueeze(0)
            else:
                action_noise_train_block = None
            if torch.any(action_noise_eval_std > 0):
                action_noise_eval_block = self._stack_randn_with_generators(
                    rollout_generators,
                    (batch_size, block_len, max_action_dim),
                    device=device,
                    dtype=torch.float32,
                ).transpose(0, 1) * action_mask.unsqueeze(0)
            else:
                action_noise_eval_block = None
            if torch.any(state_noise_std > 0):
                state_noise_block = self._stack_randn_with_generators(
                    rollout_generators,
                    (batch_size, block_len, max_state_dim),
                    device=device,
                    dtype=torch.float32,
                ).transpose(0, 1) * state_mask.unsqueeze(0)
            else:
                state_noise_block = None
            if torch.any(dropout_active):
                dropout_draws_block = self._stack_rand_with_generators(
                    rollout_generators,
                    (batch_size, block_len),
                    device=device,
                    dtype=torch.float32,
                ).transpose(0, 1)
            else:
                dropout_draws_block = None

        if noise_streaming_mode:
            _refresh_noise_block(0)
        else:
            transition_noise = self._stack_randn_with_generators(
                rollout_generators,
                (batch_size, n_samples, max_noise_dim),
                device=device,
                dtype=torch.float32,
            ).transpose(0, 1) * noise_mask.unsqueeze(0)
            if torch.any(action_noise_train_std > 0):
                action_noise_train = self._stack_randn_with_generators(
                    rollout_generators,
                    (batch_size, n_samples, max_action_dim),
                    device=device,
                    dtype=torch.float32,
                ).transpose(0, 1) * action_mask.unsqueeze(0)
            if torch.any(action_noise_eval_std > 0):
                action_noise_eval = self._stack_randn_with_generators(
                    rollout_generators,
                    (batch_size, n_samples, max_action_dim),
                    device=device,
                    dtype=torch.float32,
                ).transpose(0, 1) * action_mask.unsqueeze(0)
            if torch.any(state_noise_std > 0):
                state_noise = self._stack_randn_with_generators(
                    rollout_generators,
                    (batch_size, n_samples, max_state_dim),
                    device=device,
                    dtype=torch.float32,
                ).transpose(0, 1) * state_mask.unsqueeze(0)
            if torch.any(dropout_active):
                dropout_draws = self._stack_rand_with_generators(
                    rollout_generators,
                    (batch_size, n_samples),
                    device=device,
                    dtype=torch.float32,
                ).transpose(0, 1)

        policy_accepts_reward_mask = self._policy_step_accepts_reward_mask(policy_step_fn)
        obs_slot_dim_max = int(obs_slot_dims.max().item())
        action_slot_dim_max = int(action_slot_dims.max().item())
        env_info = {
            "family": family_list,
            "state_dim": state_dims,
            "obs_dim": obs_dims,
            "action_dim": int(max_action_dim),
            "action_dim_per_sample": action_dims,
            "noise_dim": noise_dims,
            "zero_pad_dim": zero_pad_dims,
            "obs_slot_dim": obs_slot_dim_max,
            "action_slot_dim": action_slot_dim_max,
            "reward_dropout_ratio": reward_dropout_ratio,
            "reward_dropout_enabled": reward_dropout_enabled,
            "reward_dropout_impute_zero": reward_dropout_impute_zero,
            "state_highway_enabled": state_highway_enabled,
            "state_highway_lambda": state_highway_lambda,
            "aev4_enabled": aev4_enabled,
            "aev4_highway_ratio": aev4_highway_ratio,
            "aev4_update_scale": aev4_update_scale,
            "aev4_update_clip": aev4_update_clip,
        }

        # Precompute token-layout scatter metadata once and reuse in the rollout
        # loop. This is semantically equivalent and reduces per-step index work.
        token_layout_prepack_flag = str(os.environ.get("TICL_POLICY_TOKEN_LAYOUT_PREPACK", "1")).strip().lower()
        token_layout_prepack = token_layout_prepack_flag not in {"0", "false", "no", "off"}
        token_obs_write_cap = 0
        token_obs_valid = None
        token_reward_rows = None
        token_reward_cols = None
        token_mask_rows = None
        token_mask_cols = None
        token_action_write_cap = 0
        token_action_dst_rows = None
        token_action_dst_cols = None
        token_action_src_cols = None
        if collect_x and num_features > 0 and token_layout_prepack:
            token_obs_write_cap = int(min(max_obs_dim, num_features))
            if token_obs_write_cap > 0:
                obs_cap = torch.minimum(obs_dims, obs_slot_dims).unsqueeze(1)
                obs_cols = torch.arange(token_obs_write_cap, device=device, dtype=torch.long).unsqueeze(0)
                token_obs_valid = (obs_cols < obs_cap).to(dtype=torch.float32)

            token_reward_cols = obs_slot_dims
            token_reward_rows = torch.nonzero(token_reward_cols < num_features, as_tuple=False).squeeze(1)

            token_mask_cols = obs_slot_dims + 1
            token_mask_rows = torch.nonzero(token_mask_cols < num_features, as_tuple=False).squeeze(1)

            token_action_write_cap = int(min(max_action_dim, num_features))
            if token_action_write_cap > 0:
                action_positions = torch.arange(token_action_write_cap, device=device, dtype=torch.long).unsqueeze(0)
                action_start = (obs_slot_dims + 2).unsqueeze(1)
                action_cap = torch.minimum(action_dims, action_slot_dims).unsqueeze(1)
                token_action_cols = action_start + action_positions
                token_action_valid = (action_positions < action_cap) & (token_action_cols < num_features)
                if torch.any(token_action_valid):
                    token_action_assign = torch.nonzero(token_action_valid, as_tuple=False)
                    token_action_dst_rows = token_action_assign[:, 0]
                    token_action_src_cols = token_action_assign[:, 1]
                    token_action_dst_cols = token_action_cols[token_action_valid]

        def _accumulate_gp_projection_profile(fn_obj):
            nonlocal transition_gp_first_projection_wall_s
            nonlocal transition_gp_second_projection_wall_s
            nonlocal transition_gp_projection_call_count
            nonlocal transition_gp_rff_fused_call_count
            nonlocal transition_gp_shared_total_wall_s
            nonlocal transition_gp_shared_core_wall_s
            nonlocal transition_gp_shared_noise_wall_s
            nonlocal transition_gp_shared_checkpoint_wall_s
            nonlocal transition_gp_shared_post_wall_s
            nonlocal transition_gp_shared_call_count
            if not profile_rollout_timing:
                return
            reader = getattr(fn_obj, "_consume_gp_projection_profile", None)
            if not callable(reader):
                return
            gp_stats = reader()
            if not isinstance(gp_stats, dict):
                return
            transition_gp_first_projection_wall_s += float(gp_stats.get("first_projection_wall_s", 0.0) or 0.0)
            transition_gp_second_projection_wall_s += float(gp_stats.get("second_projection_wall_s", 0.0) or 0.0)
            transition_gp_projection_call_count += int(gp_stats.get("call_count", 0) or 0)
            transition_gp_rff_fused_call_count += int(gp_stats.get("rff_fused_call_count", 0) or 0)
            transition_gp_shared_total_wall_s += float(gp_stats.get("shared_total_wall_s", 0.0) or 0.0)
            transition_gp_shared_core_wall_s += float(gp_stats.get("shared_core_wall_s", 0.0) or 0.0)
            transition_gp_shared_noise_wall_s += float(gp_stats.get("shared_noise_wall_s", 0.0) or 0.0)
            transition_gp_shared_checkpoint_wall_s += float(
                gp_stats.get("shared_checkpoint_wall_s", 0.0) or 0.0
            )
            transition_gp_shared_post_wall_s += float(gp_stats.get("shared_post_wall_s", 0.0) or 0.0)
            transition_gp_shared_call_count += int(gp_stats.get("shared_call_count", 0) or 0)

        for t in range(n_samples):
            obs_t = state_t[:, :max_obs_dim] * obs_mask
            if collect_x:
                with torch.no_grad():
                    token_row = x_steps[t]
                    token_row.zero_()
                    if num_features > 0:
                        if token_layout_prepack:
                            if token_obs_valid is not None:
                                token_row[:, :token_obs_write_cap] = (
                                    obs_t[:, :token_obs_write_cap].detach() * token_obs_valid
                                )

                            if token_reward_rows is not None and token_reward_rows.numel() > 0:
                                token_row[token_reward_rows, token_reward_cols[token_reward_rows]] = (
                                    reward_t[token_reward_rows].detach()
                                )

                            if token_mask_rows is not None and token_mask_rows.numel() > 0:
                                token_row[token_mask_rows, token_mask_cols[token_mask_rows]] = (
                                    reward_mask_t[token_mask_rows].detach()
                                )

                            if token_action_dst_rows is not None:
                                action_src = action_t[:, :token_action_write_cap].detach()
                                token_row[token_action_dst_rows, token_action_dst_cols] = action_src[
                                    token_action_dst_rows,
                                    token_action_src_cols,
                                ]
                        else:
                            obs_cap = torch.minimum(obs_dims, obs_slot_dims)
                            obs_write_cap = min(max_obs_dim, num_features)
                            if obs_write_cap > 0:
                                obs_cols = torch.arange(obs_write_cap, device=device).unsqueeze(0)
                                obs_valid = (obs_cols < obs_cap.unsqueeze(1)).to(dtype=obs_t.dtype)
                                token_row[:, :obs_write_cap] = obs_t[:, :obs_write_cap].detach() * obs_valid

                            reward_cols = obs_slot_dims
                            reward_rows = torch.nonzero(reward_cols < num_features, as_tuple=False).squeeze(1)
                            if reward_rows.numel() > 0:
                                token_row[reward_rows, reward_cols[reward_rows]] = reward_t[reward_rows].detach()

                            mask_cols = obs_slot_dims + 1
                            mask_rows = torch.nonzero(mask_cols < num_features, as_tuple=False).squeeze(1)
                            if mask_rows.numel() > 0:
                                token_row[mask_rows, mask_cols[mask_rows]] = reward_mask_t[mask_rows].detach()

                            action_write_cap = min(max_action_dim, num_features)
                            if action_write_cap > 0:
                                action_positions = torch.arange(action_write_cap, device=device).unsqueeze(0)
                                action_start = (obs_slot_dims + 2).unsqueeze(1)
                                action_cap = torch.minimum(action_dims, action_slot_dims).unsqueeze(1)
                                action_cols = action_start + action_positions
                                action_valid = (action_positions < action_cap) & (action_cols < num_features)
                                if torch.any(action_valid):
                                    action_rows = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, action_write_cap)
                                    action_src = action_t[:, :action_write_cap].detach()
                                    token_row[action_rows[action_valid], action_cols[action_valid]] = action_src[action_valid]

            if policy_accepts_reward_mask:
                policy_cuda_start = None
                policy_wall_t0 = time.perf_counter() if profile_rollout_timing else None
                if profile_rollout_breakdown_cuda:
                    policy_cuda_start = torch.cuda.Event(enable_timing=True)
                    policy_cuda_start.record()
                policy_out = policy_step_fn(
                    obs_t,
                    action_t,
                    reward_t.reshape(batch_size, 1),
                    reward_mask_t.reshape(batch_size, 1),
                    cache,
                    t,
                    env_info,
                )
            else:
                policy_cuda_start = None
                policy_wall_t0 = time.perf_counter() if profile_rollout_timing else None
                if profile_rollout_breakdown_cuda:
                    policy_cuda_start = torch.cuda.Event(enable_timing=True)
                    policy_cuda_start.record()
                policy_out = policy_step_fn(
                    obs_t,
                    action_t,
                    reward_t.reshape(batch_size, 1),
                    cache,
                    t,
                    env_info,
                )
            if profile_rollout_timing and policy_wall_t0 is not None:
                policy_wall_s += (time.perf_counter() - policy_wall_t0)
            if policy_cuda_start is not None:
                policy_cuda_end = torch.cuda.Event(enable_timing=True)
                policy_cuda_end.record()
                policy_cuda_pairs.append((policy_cuda_start, policy_cuda_end))
            if isinstance(policy_out, tuple):
                action_next, cache = policy_out
            else:
                action_next = policy_out
            if action_next.ndim == 1:
                action_next = action_next.reshape(batch_size, 1)
            if action_next.ndim != 2 or action_next.shape[0] != batch_size:
                raise ValueError(
                    f"policy action batch mismatch: expected ({batch_size}, {max_action_dim}), got {tuple(action_next.shape)}"
                )
            if action_next.shape[-1] != max_action_dim:
                raise ValueError(
                    f"policy action dim mismatch: expected {max_action_dim}, got {action_next.shape[-1]}"
                )
            action_next = torch.tanh(action_next) * action_mask

            noise_timing_t0 = time.perf_counter() if profile_rollout_timing else None
            noise_block_idx = None
            if noise_streaming_mode:
                if t >= noise_block_end:
                    _refresh_noise_block(t)
                noise_block_idx = int(t - noise_block_start)

            if t < single_eval_pos:
                if noise_streaming_mode and action_noise_train_block is not None and noise_block_idx is not None:
                    action_next = torch.tanh(
                        action_next
                        + action_noise_train_block[noise_block_idx] * action_noise_train_std[:, None]
                    ) * action_mask
                elif action_noise_train is not None:
                    action_next = torch.tanh(
                        action_next + action_noise_train[t] * action_noise_train_std[:, None]
                    ) * action_mask
            elif noise_streaming_mode and action_noise_eval_block is not None and noise_block_idx is not None:
                action_next = torch.tanh(
                    action_next
                    + action_noise_eval_block[noise_block_idx] * action_noise_eval_std[:, None]
                ) * action_mask
            elif action_noise_eval is not None:
                action_next = torch.tanh(
                    action_next + action_noise_eval[t] * action_noise_eval_std[:, None]
                ) * action_mask

            if noise_streaming_mode and noise_block_idx is not None:
                noise_t = transition_noise_block[noise_block_idx]
                state_noise_t = (
                    None
                    if state_noise_block is None
                    else state_noise_block[noise_block_idx]
                )
                dropout_draw_t = (
                    None
                    if dropout_draws_block is None
                    else dropout_draws_block[noise_block_idx]
                )
            else:
                noise_t = transition_noise[t]
                state_noise_t = None if state_noise is None else state_noise[t]
                dropout_draw_t = None if dropout_draws is None else dropout_draws[t]
            if profile_rollout_timing and noise_timing_t0 is not None:
                transition_noise_wall_s += (time.perf_counter() - noise_timing_t0)

            reward_next_raw = torch.empty((batch_size,), device=device, dtype=torch.float32)
            reward_mask_next = torch.ones((batch_size,), device=device, dtype=torch.float32)
            state_next = torch.zeros((batch_size, max_state_dim), device=device, dtype=torch.float32)

            transition_cuda_start = None
            transition_wall_t0 = time.perf_counter() if profile_rollout_timing else None
            if profile_rollout_breakdown_cuda:
                transition_cuda_start = torch.cuda.Event(enable_timing=True)
                transition_cuda_start.record()
            pending_async_group_ops = None
            pending_async_group_count = 0
            pending_async_streams = None
            if transition_stream_fusion:
                if async_group_commit_in_stream:
                    pending_async_streams = []
                else:
                    pending_async_group_ops = [None] * int(len(transition_groups))
            for group in transition_groups:
                group_wall_t0 = time.perf_counter() if profile_rollout_timing else None
                group_slice = group["slice"]
                group_indices = group["indices"]
                env_g = group["env"]
                group_bs = int(group.get("batch_size", 0) or 0)
                state_dim_g = int(group["state_dim"])
                obs_dim_g = int(group["obs_dim"])
                action_dim_g = int(group["action_dim"])
                noise_dim_g = int(group["noise_dim"])
                pack_wall_t0 = time.perf_counter() if profile_rollout_timing else None
                if group_slice is None:
                    state_in = state_t.index_select(0, group_indices)[:, :state_dim_g]
                    obs_in = obs_t.index_select(0, group_indices)[:, :obs_dim_g]
                    action_in = action_next.index_select(0, group_indices)[:, :action_dim_g]
                    noise_in = noise_t.index_select(0, group_indices)[:, :noise_dim_g]
                else:
                    state_in = state_t[group_slice, :state_dim_g]
                    obs_in = obs_t[group_slice, :obs_dim_g]
                    action_in = action_next[group_slice, :action_dim_g]
                    noise_in = noise_t[group_slice, :noise_dim_g]
                env_obs_start = int(group["env_obs_start"])
                env_action_start = int(group["env_action_start"])
                env_noise_start = int(group["env_noise_start"])

                reward_scale_g = group["reward_scale_view"]
                alpha_g = group["alpha_view"]
                transition_generator_g = group.get("transition_generator", None)
                use_fused_transition = bool(group.get("use_fused_transition", False)) and callable(
                    transition_generator_g
                )
                stream_transition = group.get("stream_transition", None)
                stream_y = group.get("stream_y", None)
                stream_x = group.get("stream_x", None)
                transition_checkpoint_enabled = bool(group.get("transition_checkpoint_enabled", 0))
                reward_next_raw_g = None
                x_next_g = None
                transition_input = None
                gp_projection_profile_capable = bool(group.get("gp_projection_profile_capable", False))
                if gp_projection_profile_capable and not bool(group.get("_gp_projection_profile_seen", False)):
                    transition_gp_profile_group_count += 1
                    group["_gp_projection_profile_seen"] = True
                force_sync_gp_projection_profile = bool(
                    profile_rollout_timing
                    and self.profile_gp_projection_timing
                    and gp_projection_profile_capable
                )
                if (
                    force_sync_gp_projection_profile
                    and (stream_transition is not None)
                    and not bool(group.get("_gp_projection_profile_sync_seen", False))
                ):
                    transition_gp_profile_sync_group_count += 1
                    group["_gp_projection_profile_sync_seen"] = True
                if force_sync_gp_projection_profile and (stream_transition is not None):
                    stream_transition = None
                packed_transition_input_enabled = bool(group.get("packed_input_enabled", False))
                packed_transition_input = None
                if packed_transition_input_enabled:
                    transition_packed_env_input_call_count += 1
                    packed_transition_input = group["packed_input"]
                    packed_transition_input.zero_()
                    packed_transition_input[:, :state_dim_g] = state_in * group["packed_state_mask"]
                    packed_obs_rows = group.get("packed_obs_rows", None)
                    if packed_obs_rows is not None and int(packed_obs_rows.numel()) > 0:
                        packed_transition_input[packed_obs_rows, group["packed_obs_dst_cols"]] = obs_in[
                            packed_obs_rows, group["packed_obs_src_cols"]
                        ]
                    packed_action_rows = group.get("packed_action_rows", None)
                    if packed_action_rows is not None and int(packed_action_rows.numel()) > 0:
                        packed_transition_input[packed_action_rows, group["packed_action_dst_cols"]] = action_in[
                            packed_action_rows, group["packed_action_src_cols"]
                        ]
                    packed_noise_rows = group.get("packed_noise_rows", None)
                    if packed_noise_rows is not None and int(packed_noise_rows.numel()) > 0:
                        packed_transition_input[packed_noise_rows, group["packed_noise_dst_cols"]] = noise_in[
                            packed_noise_rows, group["packed_noise_src_cols"]
                        ]
                if packed_transition_input_enabled:
                    transition_input = packed_transition_input
                else:
                    env_in = group["env_in"]
                    env_in[:, :state_dim_g] = state_in
                    env_in[:, env_obs_start: env_obs_start + obs_dim_g] = obs_in
                    env_in[:, env_action_start: env_action_start + action_dim_g] = action_in
                    env_in[:, env_noise_start: env_noise_start + noise_dim_g] = noise_in
                    transition_input = env_in
                if profile_rollout_timing and pack_wall_t0 is not None:
                    transition_env_pack_wall_s += (time.perf_counter() - pack_wall_t0)
                if use_fused_transition:
                    if transition_checkpoint_enabled and bool(transition_input.requires_grad):
                        transition_checkpoint_call_count += 1
                    if stream_transition is not None:
                        launch_wall_t0 = time.perf_counter() if profile_rollout_timing else None
                        with torch.cuda.stream(stream_transition):
                            if packed_transition_input_enabled:
                                x_next_g, reward_next_raw_g = transition_generator_g(
                                    transition_input,
                                    generators_for_noise=group["rollout_generators"],
                                    x_is_dual_packed=False,
                                    x_input_is_packed=True,
                                )
                            else:
                                x_next_g, reward_next_raw_g = transition_generator_g(
                                    transition_input,
                                    generators_for_noise=group["rollout_generators"],
                                    x_is_dual_packed=False,
                                )
                            reward_next_raw_g = reward_scale_g * reward_next_raw_g.reshape(-1)
                            if async_group_commit_in_stream:
                                state_next_g = (1.0 - alpha_g) * state_in + alpha_g * x_next_g
                                reward_next_raw[group_slice] = reward_next_raw_g
                                state_next[group_slice, :state_dim_g] = state_next_g
                        if async_group_commit_in_stream:
                            pending_async_streams.append((stream_transition,))
                        else:
                            pending_async_group_ops[pending_async_group_count] = (
                                "deferred_commit",
                                (group_slice, alpha_g, state_dim_g, x_next_g, reward_next_raw_g),
                                (stream_transition,),
                            )
                            pending_async_group_count += 1
                        transition_fused_call_count += 1
                        if profile_rollout_timing and launch_wall_t0 is not None:
                            launch_dt = time.perf_counter() - launch_wall_t0
                            transition_group_launch_wall_s += float(launch_dt)
                            transition_fused_wall_s += float(launch_dt)
                            transition_fused_launch_wall_s += float(launch_dt)
                    else:
                        fused_wall_t0 = time.perf_counter() if profile_rollout_timing else None
                        if packed_transition_input_enabled:
                            x_next_g, reward_next_raw_g = transition_generator_g(
                                transition_input,
                                generators_for_noise=group["rollout_generators"],
                                x_is_dual_packed=False,
                                x_input_is_packed=True,
                            )
                        else:
                            x_next_g, reward_next_raw_g = transition_generator_g(
                                transition_input,
                                generators_for_noise=group["rollout_generators"],
                                x_is_dual_packed=False,
                            )
                        _accumulate_gp_projection_profile(transition_generator_g)
                        reward_next_raw_g = reward_scale_g * reward_next_raw_g.reshape(-1)
                        if profile_rollout_timing and fused_wall_t0 is not None:
                            transition_fused_wall_s += (time.perf_counter() - fused_wall_t0)
                        transition_fused_call_count += 1
                        state_next_g = (1.0 - alpha_g) * state_in + alpha_g * x_next_g
                        if group_slice is None:
                            reward_next_raw.index_copy_(0, group_indices, reward_next_raw_g)
                            state_next[group_indices, :state_dim_g] = state_next_g
                        else:
                            reward_next_raw[group_slice] = reward_next_raw_g
                            state_next[group_slice, :state_dim_g] = state_next_g
                elif (stream_y is not None) and (stream_x is not None):
                    launch_wall_t0 = time.perf_counter() if profile_rollout_timing else None
                    with torch.cuda.stream(stream_y):
                        reward_next_raw_g = reward_scale_g * env_g["y_generator"](
                            transition_input,
                            generators_for_noise=group["rollout_generators"],
                        ).reshape(-1)
                        _accumulate_gp_projection_profile(env_g["y_generator"])
                        if async_group_commit_in_stream:
                            reward_next_raw[group_slice] = reward_next_raw_g
                    with torch.cuda.stream(stream_x):
                        x_next_g = env_g["x_generator"](
                            transition_input,
                            generators_for_noise=group["rollout_generators"],
                        )
                        _accumulate_gp_projection_profile(env_g["x_generator"])
                        if async_group_commit_in_stream:
                            state_next_g = (1.0 - alpha_g) * state_in + alpha_g * x_next_g
                            state_next[group_slice, :state_dim_g] = state_next_g
                    if async_group_commit_in_stream:
                        pending_async_streams.append((stream_y, stream_x))
                    else:
                        pending_async_group_ops[pending_async_group_count] = (
                            "deferred_commit",
                            (group_slice, alpha_g, state_dim_g, x_next_g, reward_next_raw_g),
                            (stream_y, stream_x),
                        )
                        pending_async_group_count += 1
                    if profile_rollout_timing and launch_wall_t0 is not None:
                        launch_dt = time.perf_counter() - launch_wall_t0
                        transition_group_launch_wall_s += float(launch_dt)
                else:
                    y_wall_t0 = time.perf_counter() if profile_rollout_timing else None
                    reward_next_raw_g = reward_scale_g * env_g["y_generator"](
                        transition_input,
                        generators_for_noise=group["rollout_generators"],
                    ).reshape(-1)
                    _accumulate_gp_projection_profile(env_g["y_generator"])
                    if profile_rollout_timing and y_wall_t0 is not None:
                        transition_y_wall_s += (time.perf_counter() - y_wall_t0)
                    x_wall_t0 = time.perf_counter() if profile_rollout_timing else None
                    x_next_g = env_g["x_generator"](
                        transition_input,
                        generators_for_noise=group["rollout_generators"],
                    )
                    _accumulate_gp_projection_profile(env_g["x_generator"])
                    if profile_rollout_timing and x_wall_t0 is not None:
                        transition_x_wall_s += (time.perf_counter() - x_wall_t0)
                    state_next_g = (1.0 - alpha_g) * state_in + alpha_g * x_next_g
                    if group_slice is None:
                        reward_next_raw.index_copy_(0, group_indices, reward_next_raw_g)
                        state_next[group_indices, :state_dim_g] = state_next_g
                    else:
                        reward_next_raw[group_slice] = reward_next_raw_g
                        state_next[group_slice, :state_dim_g] = state_next_g
                if profile_rollout_timing and group_wall_t0 is not None:
                    transition_group_wall_s += (time.perf_counter() - group_wall_t0)

            if pending_async_streams:
                sync_wall_t0 = time.perf_counter() if profile_rollout_timing else None
                cur_stream = torch.cuda.current_stream(device=device_obj)
                for stream_tuple in pending_async_streams:
                    for stream in stream_tuple:
                        cur_stream.wait_stream(stream)
                if profile_rollout_timing and sync_wall_t0 is not None:
                    sync_dt = time.perf_counter() - sync_wall_t0
                    transition_group_sync_wall_s += float(sync_dt)
                    transition_group_wall_s += float(sync_dt)

            if pending_async_group_count > 0:
                sync_wall_t0 = time.perf_counter() if profile_rollout_timing else None
                cur_stream = torch.cuda.current_stream(device=device_obj)
                for op_idx in range(pending_async_group_count):
                    op_meta = pending_async_group_ops[op_idx]
                    stream_tuple = op_meta[-1]
                    for stream in stream_tuple:
                        cur_stream.wait_stream(stream)
                state_update_wall_t0 = time.perf_counter() if profile_rollout_timing else None
                for op_idx in range(pending_async_group_count):
                    op_meta = pending_async_group_ops[op_idx]
                    if op_meta[0] != "deferred_commit":
                        continue
                    group_slice, alpha_g, state_dim_g, x_next_g, reward_next_raw_g = op_meta[1]
                    state_in = state_t[group_slice, :state_dim_g]
                    state_next_g = (1.0 - alpha_g) * state_in + alpha_g * x_next_g
                    reward_next_raw[group_slice] = reward_next_raw_g
                    state_next[group_slice, :state_dim_g] = state_next_g
                if profile_rollout_timing and state_update_wall_t0 is not None:
                    transition_state_update_wall_s += (time.perf_counter() - state_update_wall_t0)
                if profile_rollout_timing and sync_wall_t0 is not None:
                    sync_dt = time.perf_counter() - sync_wall_t0
                    transition_group_sync_wall_s += float(sync_dt)
                    transition_group_wall_s += float(sync_dt)

            reward_next = torch.maximum(
                torch.minimum(reward_next_raw, reward_clip),
                -reward_clip,
            )
            if dropout_draw_t is not None:
                dropout_timing_t0 = time.perf_counter() if profile_rollout_timing else None
                drop_mask = dropout_active & (dropout_draw_t < reward_dropout_ratio)
                if collect_runtime_info:
                    reward_drop_count = reward_drop_count + drop_mask.to(dtype=torch.int64)
                reward_mask_next = torch.where(drop_mask, torch.zeros_like(reward_mask_next), reward_mask_next)
                impute_mask = drop_mask & reward_dropout_impute_zero
                reward_next = torch.where(impute_mask, torch.zeros_like(reward_next), reward_next)
                if profile_rollout_timing and dropout_timing_t0 is not None:
                    transition_noise_wall_s += (time.perf_counter() - dropout_timing_t0)
            if state_noise_t is not None:
                state_noise_timing_t0 = time.perf_counter() if profile_rollout_timing else None
                state_next = state_next + state_noise_t * state_noise_std[:, None]
                if profile_rollout_timing and state_noise_timing_t0 is not None:
                    transition_noise_wall_s += (time.perf_counter() - state_noise_timing_t0)
            state_next = self._apply_state_postprocess(
                state_next_raw=state_next,
                state_prev=state_t,
                state_clip=state_clip,
                state_highway_enabled=state_highway_enabled,
                state_highway_lambda=state_highway_lambda,
            )
            aev4_step_aux = None
            if aev4_reg_enabled:
                # Family-group vectorization uses one global v4 config.
                state_next, aev4_step_aux = self._apply_aev4_state_update(
                    state_prev=state_t,
                    state_next_post=state_next,
                    aev4_cfg=aev4_cfg,
                )
            if transition_cuda_start is not None:
                transition_cuda_end = torch.cuda.Event(enable_timing=True)
                transition_cuda_end.record()
                transition_cuda_pairs.append((transition_cuda_start, transition_cuda_end))
            if profile_rollout_timing and transition_wall_t0 is not None:
                transition_wall_s += (time.perf_counter() - transition_wall_t0)

            state_next = state_next * state_mask
            state_delta = state_next - state_t
            if aev2_enabled and (aev2_prev_delta is not None):
                self._aev2_update_accumulator(aev2_acc, aev2_prev_delta, state_delta, aev2_cfg)
            aev2_prev_delta = state_delta
            if aev3_enabled and (aev3_prev_delta is not None):
                self._aev3_update_accumulator(aev3_acc, aev3_prev_delta, state_delta, aev3_cfg)
            aev3_prev_delta = state_delta
            if aev4_reg_enabled and (aev4_prev_delta is not None):
                self._aev4_update_accumulator(aev4_acc, aev4_prev_delta, state_delta, aev4_cfg, step_aux=aev4_step_aux)
            aev4_prev_delta = state_delta
            action_next = action_next * action_mask
            if tbptt_window_active:
                if y_steps is not None:
                    y_steps[t] = reward_next.detach()
                tbptt_reward_buffer.append(reward_next)
            else:
                if y_steps is not None:
                    y_steps[t] = reward_next
            if collect_runtime_info:
                reward_values[t] = reward_next.detach()
                state_abs_max[t] = state_next.detach().abs().amax(dim=1)

            state_t = state_next
            action_t = action_next
            reward_t = reward_next
            reward_mask_t = reward_mask_next

            if tbptt_window_active:
                is_window_end = (len(tbptt_reward_buffer) >= tbptt_window_size) or (t == (n_samples - 1))
                if is_window_end:
                    rewards_window = torch.stack(tbptt_reward_buffer, dim=0)
                    tbptt_reward_buffer = []
                    if t < (n_samples - 1):
                        state_t = state_t.detach()
                        action_t = action_t.detach()
                        reward_t = reward_t.detach()
                        reward_mask_t = reward_mask_t.detach()
                        if aev2_prev_delta is not None:
                            aev2_prev_delta = aev2_prev_delta.detach()
                        if aev3_prev_delta is not None:
                            aev3_prev_delta = aev3_prev_delta.detach()
                        if aev4_prev_delta is not None:
                            aev4_prev_delta = aev4_prev_delta.detach()
                        for group in transition_groups:
                            group["env_in"] = group["env_in"].detach()
                        cache = self._detach_policy_cache(cache, clone_tensors=(tbptt_reward_sink is None))
                    if tbptt_reward_sink is not None:
                        if aev2_streaming_sink or aev3_streaming_sink or aev4_streaming_sink:
                            payload_aux = {}
                            if aev2_streaming_sink:
                                aev2_window_summary = self._aev2_finalize_accumulator(
                                    aev2_acc,
                                    device=device,
                                    dtype=torch.float32,
                                    detach_penalty=False,
                                )
                                payload_aux["aev2"] = aev2_window_summary
                                self._aev2_accumulate_window_summary(aev2_det_acc, aev2_window_summary, device=device)
                                aev2_acc = self._aev2_new_accumulator(aev2_enabled, device=device, dtype=torch.float32)
                            if aev3_streaming_sink:
                                aev3_window_summary = self._aev3_finalize_accumulator(
                                    aev3_acc,
                                    device=device,
                                    dtype=torch.float32,
                                    detach_penalty=False,
                                )
                                payload_aux["aev3"] = aev3_window_summary
                                self._aev3_accumulate_window_summary(aev3_det_acc, aev3_window_summary, device=device)
                                aev3_acc = self._aev3_new_accumulator(
                                    aev3_enabled,
                                    device=device,
                                    dtype=torch.float32,
                                    aev3_cfg=aev3_cfg,
                                )
                            if aev4_streaming_sink:
                                aev4_window_summary = self._aev4_finalize_accumulator(
                                    aev4_acc,
                                    device=device,
                                    dtype=torch.float32,
                                    detach_penalty=False,
                                )
                                payload_aux["aev4"] = aev4_window_summary
                                self._aev4_accumulate_window_summary(aev4_det_acc, aev4_window_summary, device=device)
                                aev4_acc = self._aev4_new_accumulator(
                                    aev4_reg_enabled,
                                    device=device,
                                    dtype=torch.float32,
                                    aev4_cfg=aev4_cfg,
                                )
                            payload_rewards = rewards_window.index_select(1, inv_perm) if needs_unpermute else rewards_window
                            if ("aev2" in payload_aux) and (len(payload_aux) == 1):
                                tbptt_reward_sink((payload_rewards, payload_aux["aev2"]))
                            else:
                                tbptt_reward_sink((payload_rewards, payload_aux))
                        else:
                            if needs_unpermute:
                                tbptt_reward_sink(rewards_window.index_select(1, inv_perm))
                            else:
                                tbptt_reward_sink(rewards_window)

        if needs_unpermute:
            if y_steps is not None:
                y_steps = y_steps.index_select(1, inv_perm)
            if collect_x:
                x_steps = x_steps.index_select(1, inv_perm)

        if collect_runtime_info:
            if needs_unpermute:
                state_abs_max = state_abs_max.index_select(1, inv_perm)
                reward_values = reward_values.index_select(1, inv_perm)
                reward_drop_count = reward_drop_count.index_select(0, inv_perm)
                state_dims_meta = state_dims.index_select(0, inv_perm)
                obs_dims_meta = obs_dims.index_select(0, inv_perm)
                action_dims_meta = action_dims.index_select(0, inv_perm)
                noise_dims_meta = noise_dims.index_select(0, inv_perm)
                zero_pad_dims_meta = zero_pad_dims.index_select(0, inv_perm)
                obs_slot_dims_meta = obs_slot_dims.index_select(0, inv_perm)
                action_slot_dims_meta = action_slot_dims.index_select(0, inv_perm)
                reward_dropout_ratio_meta = reward_dropout_ratio.index_select(0, inv_perm)
                aev4_enabled_meta = aev4_enabled.index_select(0, inv_perm)
                aev4_highway_ratio_meta = aev4_highway_ratio.index_select(0, inv_perm)
                aev4_update_scale_meta = aev4_update_scale.index_select(0, inv_perm)
                aev4_update_clip_meta = aev4_update_clip.index_select(0, inv_perm)
                inv_perm_cpu = inv_perm.detach().cpu().tolist()
                family_meta = [family_list[int(i)] for i in inv_perm_cpu]
            else:
                state_dims_meta = state_dims
                obs_dims_meta = obs_dims
                action_dims_meta = action_dims
                noise_dims_meta = noise_dims
                zero_pad_dims_meta = zero_pad_dims
                obs_slot_dims_meta = obs_slot_dims
                action_slot_dims_meta = action_slot_dims
                reward_dropout_ratio_meta = reward_dropout_ratio
                aev4_enabled_meta = aev4_enabled
                aev4_highway_ratio_meta = aev4_highway_ratio
                aev4_update_scale_meta = aev4_update_scale
                aev4_update_clip_meta = aev4_update_clip
                family_meta = family_list

            reward_drop_frac = reward_drop_count.to(dtype=torch.float32) / float(max(1, n_samples))
            env_meta = {
                "family": family_meta,
                "state_dim": state_dims_meta,
                "obs_dim": obs_dims_meta,
                "action_dim": action_dims_meta,
                "noise_dim": noise_dims_meta,
                "zero_pad_dim": zero_pad_dims_meta,
                "obs_slot_dim": obs_slot_dims_meta,
                "action_slot_dim": action_slot_dims_meta,
                "reward_dropout_ratio": reward_dropout_ratio_meta,
                "aev4_enabled": aev4_enabled_meta,
                "aev4_highway_ratio": aev4_highway_ratio_meta,
                "aev4_update_scale": aev4_update_scale_meta,
                "aev4_update_clip": aev4_update_clip_meta,
            }
            infos = self._build_vectorized_runtime_info(
                env=env_meta,
                reward_values=reward_values,
                state_abs_max=state_abs_max,
                reward_drop_frac=reward_drop_frac,
                single_eval_pos=single_eval_pos,
            )
        else:
            infos = [None] * batch_size
        rollout_profile = None
        if (profile_rollout_breakdown_cuda and (policy_cuda_pairs or transition_cuda_pairs)) or profile_rollout_timing:
            noise_mode = "strict_seed" if strict_seed_mode else ("block_stream" if noise_streaming_mode else "full_prealloc")
            rollout_profile = {
                "steps": int(n_samples),
                "batch_size": int(batch_size),
                "noise_mode": noise_mode,
                "noise_block_size": int(noise_block_size) if noise_streaming_mode else 0,
            }
            if profile_rollout_breakdown_cuda and (policy_cuda_pairs or transition_cuda_pairs):
                torch.cuda.synchronize(device=device_obj)
                policy_cuda_ms = float(sum(start.elapsed_time(end) for start, end in policy_cuda_pairs))
                transition_cuda_ms = float(sum(start.elapsed_time(end) for start, end in transition_cuda_pairs))
                rollout_profile["policy_cuda_ms"] = policy_cuda_ms
                rollout_profile["transition_cuda_ms"] = transition_cuda_ms
            if profile_rollout_timing:
                rollout_profile["policy_wall_ms"] = float(policy_wall_s * 1000.0)
                rollout_profile["transition_wall_ms"] = float(transition_wall_s * 1000.0)
                rollout_profile["transition_y_wall_ms"] = float(transition_y_wall_s * 1000.0)
                rollout_profile["transition_x_wall_ms"] = float(transition_x_wall_s * 1000.0)
                rollout_profile["transition_group_wall_ms"] = float(transition_group_wall_s * 1000.0)
                rollout_profile["transition_group_launch_wall_ms"] = float(
                    transition_group_launch_wall_s * 1000.0
                )
                rollout_profile["transition_group_sync_wall_ms"] = float(
                    transition_group_sync_wall_s * 1000.0
                )
                rollout_profile["transition_env_pack_wall_ms"] = float(transition_env_pack_wall_s * 1000.0)
                rollout_profile["transition_state_update_wall_ms"] = float(transition_state_update_wall_s * 1000.0)
                rollout_profile["transition_noise_wall_ms"] = float(transition_noise_wall_s * 1000.0)
                rollout_profile["transition_fused_wall_ms"] = float(transition_fused_wall_s * 1000.0)
                rollout_profile["transition_fused_launch_wall_ms"] = float(
                    transition_fused_launch_wall_s * 1000.0
                )
                rollout_profile["transition_gp_first_projection_wall_ms"] = float(
                    transition_gp_first_projection_wall_s * 1000.0
                )
                rollout_profile["transition_gp_second_projection_wall_ms"] = float(
                    transition_gp_second_projection_wall_s * 1000.0
                )
                rollout_profile["transition_gp_projection_call_count"] = int(
                    transition_gp_projection_call_count
                )
                rollout_profile["transition_gp_rff_fused_call_count"] = int(
                    transition_gp_rff_fused_call_count
                )
                rollout_profile["transition_gp_profile_group_count"] = int(
                    transition_gp_profile_group_count
                )
                rollout_profile["transition_gp_profile_sync_group_count"] = int(
                    transition_gp_profile_sync_group_count
                )
                rollout_profile["transition_gp_shared_total_wall_ms"] = float(
                    transition_gp_shared_total_wall_s * 1000.0
                )
                rollout_profile["transition_gp_shared_core_wall_ms"] = float(
                    transition_gp_shared_core_wall_s * 1000.0
                )
                rollout_profile["transition_gp_shared_noise_wall_ms"] = float(
                    transition_gp_shared_noise_wall_s * 1000.0
                )
                rollout_profile["transition_gp_shared_checkpoint_wall_ms"] = float(
                    transition_gp_shared_checkpoint_wall_s * 1000.0
                )
                rollout_profile["transition_gp_shared_post_wall_ms"] = float(
                    transition_gp_shared_post_wall_s * 1000.0
                )
                rollout_profile["transition_gp_shared_call_count"] = int(transition_gp_shared_call_count)
                rollout_profile["transition_packed_env_input_group_count"] = int(
                    transition_packed_env_input_group_count
                )
                rollout_profile["transition_packed_env_input_call_count"] = int(
                    transition_packed_env_input_call_count
                )
                rollout_profile["transition_only_build_group_count"] = int(
                    transition_only_build_group_count
                )
                rollout_profile["transition_only_skipped_generator_count"] = int(
                    transition_only_skipped_generator_count
                )
                rollout_profile["transition_setup_wall_ms"] = float(transition_setup_wall_s * 1000.0)
                rollout_profile["transition_family_build_wall_ms"] = float(transition_family_build_wall_s * 1000.0)
                rollout_profile["transition_generator_build_wall_ms"] = float(
                    transition_generator_build_wall_s * 1000.0
                )
                rollout_profile["transition_gp_shared_build_wall_ms"] = float(
                    transition_gp_shared_build_wall_s * 1000.0
                )
                rollout_profile["transition_fused_call_count"] = int(transition_fused_call_count)
                rollout_profile["transition_fused_group_count"] = int(transition_fused_group_count)
                rollout_profile["transition_fused_enabled"] = int(transition_fused_group_count > 0)
                rollout_profile["transition_checkpoint_enabled"] = int(bool(self.envgen_checkpoint))
                rollout_profile["transition_checkpoint_call_count"] = int(transition_checkpoint_call_count)
                rollout_profile["transition_group_count"] = int(len(transition_groups))
                rollout_profile["transition_family_group_count"] = int(transition_family_group_count)
                rollout_profile["transition_inner_grouping_structure_enabled"] = int(
                    transition_inner_grouping == "structure"
                )
                rollout_profile["transition_inner_min_bucket"] = int(transition_inner_min_bucket)
                rollout_profile["transition_bucket_max_batch"] = int(transition_group_max_batch)
                rollout_profile["transition_bucket_mean_batch"] = float(
                    batch_size / max(1, len(transition_groups))
                )
                rollout_profile["transition_work_actual_est"] = float(transition_group_work_actual)
                rollout_profile["transition_work_padded_est"] = float(transition_group_work_padded)
                rollout_profile["transition_work_fill_ratio"] = float(
                    transition_group_work_actual / max(1e-9, transition_group_work_padded)
                )
                rollout_profile["transition_async_enabled"] = int(bool(transition_stream_fusion))
                rollout_profile["transition_async_commit_in_stream"] = int(
                    bool(transition_stream_fusion and async_group_commit_in_stream)
                )
        self.last_rollout_profile = rollout_profile
        if aev2_enabled:
            if aev2_streaming_sink:
                self.last_rollout_v2 = self._aev2_finalize_accumulator(
                    aev2_det_acc,
                    device=device,
                    dtype=torch.float32,
                    detach_penalty=True,
                )
            else:
                self.last_rollout_v2 = self._aev2_finalize_accumulator(
                    aev2_acc,
                    device=device,
                    dtype=torch.float32,
                    detach_penalty=False,
                )
            self.last_rollout_v2["lambda"] = float(aev2_cfg.get("lambda", 0.0))
            self.last_rollout_v2["gain_lo"] = float(aev2_cfg.get("gain_lo", 0.0))
            self.last_rollout_v2["gain_hi"] = float(aev2_cfg.get("gain_hi", 0.0))
        else:
            self.last_rollout_v2 = None
        if aev3_enabled:
            if aev3_streaming_sink:
                self.last_rollout_v3 = self._aev3_finalize_accumulator(
                    aev3_det_acc,
                    device=device,
                    dtype=torch.float32,
                    detach_penalty=True,
                )
            else:
                self.last_rollout_v3 = self._aev3_finalize_accumulator(
                    aev3_acc,
                    device=device,
                    dtype=torch.float32,
                    detach_penalty=False,
                )
        else:
            self.last_rollout_v3 = None
        if aev4_reg_enabled:
            if aev4_streaming_sink:
                self.last_rollout_v4 = self._aev4_finalize_accumulator(
                    aev4_det_acc,
                    device=device,
                    dtype=torch.float32,
                    detach_penalty=True,
                )
            else:
                self.last_rollout_v4 = self._aev4_finalize_accumulator(
                    aev4_acc,
                    device=device,
                    dtype=torch.float32,
                    detach_penalty=False,
                )
        else:
            self.last_rollout_v4 = None
        if y_steps is None:
            y_steps = torch.empty((0, batch_size), device=device, dtype=torch.float32)
        return x_steps, y_steps, infos

    def _rollout_single(
        self,
        env,
        n_samples,
        num_features,
        single_eval_pos,
        device,
        policy_step_fn=None,
        collect_x=True,
        collect_runtime_info=True,
        rng_seed=None,
        tbptt_window=None,
        tbptt_reward_sink=None,
        tbptt_reward_sink_supports_aux=False,
        store_rewards=True,
    ):
        n_samples = int(n_samples)
        num_features = int(num_features)
        collect_runtime_info = bool(collect_runtime_info)

        local_generator = None
        if rng_seed is not None:
            local_generator = torch.Generator(device=device)
            local_generator.manual_seed(int(rng_seed))

        def _randn(shape, *, dtype):
            if local_generator is None:
                return torch.randn(shape, device=device, dtype=dtype)
            return torch.randn(shape, device=device, dtype=dtype, generator=local_generator)

        def _rand(shape, *, dtype):
            if local_generator is None:
                return torch.rand(shape, device=device, dtype=dtype)
            return torch.rand(shape, device=device, dtype=dtype, generator=local_generator)

        policy_accepts_reward_mask = False
        if policy_step_fn is not None:
            policy_accepts_reward_mask = self._policy_step_accepts_reward_mask(policy_step_fn)

        state_dim = int(env["state_dim"])
        obs_dim = int(env["obs_dim"])
        action_dim = int(env["action_dim"])
        noise_dim = int(env["noise_dim"])
        zero_pad_dim = int(env["zero_pad_dim"])
        obs_slot_dim = int(env["obs_slot_dim"])
        action_slot_dim = int(env["action_slot_dim"])

        state_t = _randn((state_dim,), dtype=torch.float32) * env["init_state_std"]   # s_0 Gaussian
        action_t = _randn((action_dim,), dtype=torch.float32) * env["init_action_std"]  # a_0 Gaussian
        # r_0 placeholder token input
        reward_t = torch.zeros((), device=device)
        reward_mask_t = torch.ones((), device=device)
        cache = None

        x_steps = torch.empty((n_samples, num_features), device=device, dtype=state_t.dtype) if collect_x else None
        y_steps = (
            torch.empty((n_samples,), device=device, dtype=state_t.dtype)
            if bool(store_rewards)
            else None
        )
        state_abs_max = torch.empty((n_samples,), device=device, dtype=state_t.dtype) if collect_runtime_info else None
        reward_values = torch.empty((n_samples,), device=device, dtype=state_t.dtype) if collect_runtime_info else None
        reward_drop_count = 0 if collect_runtime_info else None
        tbptt_window_active = False
        tbptt_window_size = n_samples
        if (policy_step_fn is not None) and (tbptt_window is not None):
            w = int(tbptt_window)
            if 0 < w < n_samples:
                tbptt_window_active = True
                tbptt_window_size = w
        tbptt_reward_buffer = [] if tbptt_window_active else None
        aev2_cfg = self._resolve_aev2_config()
        aev2_enabled = bool(aev2_cfg.get("enabled", False))
        aev2_prev_delta = None
        aev3_cfg = self._resolve_aev3_config()
        aev3_enabled = bool(aev3_cfg.get("enabled", False))
        aev3_prev_delta = None
        aev4_cfg = self._resolve_aev4_config()
        aev4_enabled = bool(aev4_cfg.get("enabled", False))
        aev4_prev_delta = None
        aev2_streaming_sink = bool(
            aev2_enabled
            and tbptt_window_active
            and (tbptt_reward_sink is not None)
            and bool(tbptt_reward_sink_supports_aux)
        )
        aev3_streaming_sink = bool(
            aev3_enabled
            and tbptt_window_active
            and (tbptt_reward_sink is not None)
            and bool(tbptt_reward_sink_supports_aux)
        )
        aev4_streaming_sink = bool(
            aev4_enabled
            and tbptt_window_active
            and (tbptt_reward_sink is not None)
            and bool(tbptt_reward_sink_supports_aux)
        )
        aev2_acc = self._aev2_new_accumulator(aev2_enabled, device=device, dtype=state_t.dtype)
        aev2_det_acc = self._aev2_new_accumulator(aev2_enabled, device=device, dtype=state_t.dtype)
        aev3_acc = self._aev3_new_accumulator(aev3_enabled, device=device, dtype=state_t.dtype, aev3_cfg=aev3_cfg)
        aev3_det_acc = self._aev3_new_accumulator(
            aev3_enabled, device=device, dtype=state_t.dtype, aev3_cfg=aev3_cfg
        )
        aev4_acc = self._aev4_new_accumulator(aev4_enabled, device=device, dtype=state_t.dtype, aev4_cfg=aev4_cfg)
        aev4_det_acc = self._aev4_new_accumulator(
            aev4_enabled, device=device, dtype=state_t.dtype, aev4_cfg=aev4_cfg
        )

        zero_pad_t = torch.zeros(zero_pad_dim, device=device, dtype=state_t.dtype)

        def _clone_generator_state(gen):
            if gen is None:
                return None
            cloned = torch.Generator(device=device)
            cloned.set_state(gen.get_state())
            return cloned

        def _advance_generator_randn(gen, numel):
            if gen is None:
                return
            remaining = int(max(0, numel))
            chunk = 1 << 15
            while remaining > 0:
                take = min(chunk, remaining)
                torch.randn((take,), device=device, dtype=state_t.dtype, generator=gen)
                remaining -= take

        def _advance_generator_rand(gen, numel):
            if gen is None:
                return
            remaining = int(max(0, numel))
            chunk = 1 << 15
            while remaining > 0:
                take = min(chunk, remaining)
                torch.rand((take,), device=device, dtype=state_t.dtype, generator=gen)
                remaining -= take

        # `x` token is not used by policy-gradient objective (loss uses rewards only),
        # so we can safely detach writes here to avoid building useless autograd graph.
        use_fast_env_in = policy_step_fn is None
        token_reward_idx = obs_slot_dim
        token_mask_idx = obs_slot_dim + 1
        token_action_start = obs_slot_dim + 2
        token_obs_cap = min(obs_dim, obs_slot_dim)
        token_action_cap = min(action_dim, action_slot_dim)

        transition_noise = None
        action_noise_train = None
        action_noise_eval = None
        state_noise = None
        dropout_draws = None
        transition_noise_generator = None
        action_noise_train_generator = None
        action_noise_eval_generator = None
        state_noise_generator = None
        dropout_draw_generator = None
        if use_fast_env_in:
            # Keep legacy rollout RNG ordering for data-generation path so
            # strict serial/vectorized semantic checks remain identical.
            transition_noise = _randn((n_samples, noise_dim), dtype=state_t.dtype)
            if env["action_noise_train_std"] > 0:
                action_noise_train = _randn((n_samples, action_dim), dtype=state_t.dtype)
            if env["action_noise_eval_std"] > 0:
                action_noise_eval = _randn((n_samples, action_dim), dtype=state_t.dtype)
            if env["state_noise_std"] > 0:
                state_noise = _randn((n_samples, state_dim), dtype=state_t.dtype)
            if env["reward_dropout_enabled"] and env["reward_dropout_ratio"] > 0.0:
                dropout_draws = _rand((n_samples,), dtype=state_t.dtype)
        else:
            # Stream rollout noise per step to reduce persistent memory in
            # policy-gradient rollout path.
            if local_generator is not None:
                transition_noise_generator = _clone_generator_state(local_generator)
                _advance_generator_randn(local_generator, n_samples * noise_dim)
                if env["action_noise_train_std"] > 0:
                    action_noise_train_generator = _clone_generator_state(local_generator)
                    _advance_generator_randn(local_generator, n_samples * action_dim)
                if env["action_noise_eval_std"] > 0:
                    action_noise_eval_generator = _clone_generator_state(local_generator)
                    _advance_generator_randn(local_generator, n_samples * action_dim)
                if env["state_noise_std"] > 0:
                    state_noise_generator = _clone_generator_state(local_generator)
                    _advance_generator_randn(local_generator, n_samples * state_dim)
                if env["reward_dropout_enabled"] and env["reward_dropout_ratio"] > 0.0:
                    dropout_draw_generator = _clone_generator_state(local_generator)
                    _advance_generator_rand(local_generator, n_samples)

        env_in_flat = None
        if use_fast_env_in:
            env_in_flat = torch.zeros(
                state_dim + obs_dim + action_dim + noise_dim + zero_pad_dim,
                device=device,
                dtype=state_t.dtype,
            )
            env_obs_start = state_dim
            env_action_start = state_dim + obs_dim
            env_noise_start = env_action_start + action_dim

        for t in range(n_samples):
            obs_t = state_t[:obs_dim]  # observable subset of state
            if collect_x:
                with torch.no_grad():
                    token_row = x_steps[t]
                    token_row.zero_()
                    if num_features > 0 and token_obs_cap > 0:
                        obs_write = min(token_obs_cap, num_features)
                        token_row[:obs_write] = obs_t[:obs_write].detach()
                    if token_reward_idx < num_features:
                        token_row[token_reward_idx] = reward_t.detach()
                    if token_mask_idx < num_features:
                        token_row[token_mask_idx] = reward_mask_t.detach()
                    if token_action_start < num_features and token_action_cap > 0:
                        action_write = min(token_action_cap, num_features - token_action_start)
                        token_row[token_action_start: token_action_start + action_write] = action_t[:action_write].detach()

            if transition_noise is not None:
                noise_t = transition_noise[t]
            elif transition_noise_generator is None:
                noise_t = _randn((noise_dim,), dtype=state_t.dtype)
            else:
                noise_t = torch.randn(
                    (noise_dim,),
                    device=device,
                    dtype=state_t.dtype,
                    generator=transition_noise_generator,
                )

            if use_fast_env_in:
                env_in_flat[:state_dim] = state_t
                env_in_flat[env_obs_start: env_obs_start + obs_dim] = obs_t
                env_in_flat[env_action_start: env_action_start + action_dim] = action_t
                env_in_flat[env_noise_start: env_noise_start + noise_dim] = noise_t
                policy_in = env_in_flat.unsqueeze(0)
                action_next = torch.tanh(env["policy_generator"](policy_in, generator=local_generator)).squeeze(0)
            else:
                if policy_accepts_reward_mask:
                    policy_out = policy_step_fn(
                        obs_t.unsqueeze(0),
                        action_t.unsqueeze(0),
                        reward_t.reshape(1, 1),
                        reward_mask_t.reshape(1, 1),
                        cache,
                        t,
                        env,
                    )
                else:
                    policy_out = policy_step_fn(
                        obs_t.unsqueeze(0),
                        action_t.unsqueeze(0),
                        reward_t.reshape(1, 1),
                        cache,
                        t,
                        env,
                    )
                if isinstance(policy_out, tuple):
                    action_next, cache = policy_out
                else:
                    action_next = policy_out
                if action_next.ndim == 2 and action_next.shape[0] == 1:
                    action_next = action_next.squeeze(0)
                if action_next.shape[-1] != env["action_dim"]:
                    raise ValueError(
                        f"policy action dim mismatch: expected {env['action_dim']}, got {action_next.shape[-1]}"
                    )
                action_next = torch.tanh(action_next)

            action_noise_std = env["action_noise_train_std"] if t < single_eval_pos else env["action_noise_eval_std"]
            if action_noise_std > 0:
                if t < single_eval_pos and action_noise_train is not None:
                    action_next = torch.tanh(action_next + action_noise_train[t] * action_noise_std)
                elif t >= single_eval_pos and action_noise_eval is not None:
                    action_next = torch.tanh(action_next + action_noise_eval[t] * action_noise_std)
                elif t < single_eval_pos and env["action_noise_train_std"] > 0:
                    if action_noise_train_generator is None:
                        action_noise_t = _randn((action_dim,), dtype=state_t.dtype)
                    else:
                        action_noise_t = torch.randn(
                            (action_dim,),
                            device=device,
                            dtype=state_t.dtype,
                            generator=action_noise_train_generator,
                        )
                    action_next = torch.tanh(action_next + action_noise_t * action_noise_std)
                elif t >= single_eval_pos and env["action_noise_eval_std"] > 0:
                    if action_noise_eval_generator is None:
                        action_noise_t = _randn((action_dim,), dtype=state_t.dtype)
                    else:
                        action_noise_t = torch.randn(
                            (action_dim,),
                            device=device,
                            dtype=state_t.dtype,
                            generator=action_noise_eval_generator,
                        )
                    action_next = torch.tanh(action_next + action_noise_t * action_noise_std)

            if use_fast_env_in:
                env_in_flat[env_action_start: env_action_start + action_dim] = action_next
                env_in = env_in_flat.unsqueeze(0)
            else:
                env_in = self._pack_env_input(state_t, obs_t, action_next, noise_t, zero_pad_t).unsqueeze(0)

            reward_next_raw = env["reward_scale"] * env["y_generator"](env_in, generator=local_generator).reshape(())
            reward_next = torch.clamp(
                reward_next_raw,
                -float(env["reward_clip"]),
                float(env["reward_clip"]),
            )
            reward_mask_next = torch.ones((), device=device, dtype=reward_next_raw.dtype)
            if dropout_draws is not None:
                if bool(dropout_draws[t] < float(env["reward_dropout_ratio"])):
                    if collect_runtime_info:
                        reward_drop_count += 1
                    reward_mask_next = torch.zeros((), device=device, dtype=reward_next_raw.dtype)
                    if env["reward_dropout_impute_zero"]:
                        reward_next = torch.zeros_like(reward_next_raw)
            elif env["reward_dropout_enabled"] and env["reward_dropout_ratio"] > 0.0:
                if dropout_draw_generator is None:
                    drop_draw = _rand((), dtype=state_t.dtype)
                else:
                    drop_draw = torch.rand(
                        (),
                        device=device,
                        dtype=state_t.dtype,
                        generator=dropout_draw_generator,
                    )
                if bool(drop_draw < float(env["reward_dropout_ratio"])):
                    if collect_runtime_info:
                        reward_drop_count += 1
                    reward_mask_next = torch.zeros((), device=device, dtype=reward_next_raw.dtype)
                    if env["reward_dropout_impute_zero"]:
                        reward_next = torch.zeros_like(reward_next_raw)

            x_next = env["x_generator"](env_in, generator=local_generator).squeeze(0)
            state_next = (1.0 - env["alpha"]) * state_t + env["alpha"] * x_next
            if state_noise is not None:
                state_next = state_next + state_noise[t] * env["state_noise_std"]
            elif env["state_noise_std"] > 0:
                if state_noise_generator is None:
                    state_noise_t = _randn((state_dim,), dtype=state_t.dtype)
                else:
                    state_noise_t = torch.randn(
                        (state_dim,),
                        device=device,
                        dtype=state_t.dtype,
                        generator=state_noise_generator,
                    )
                state_next = state_next + state_noise_t * env["state_noise_std"]
            state_next = self._apply_state_postprocess(
                state_next_raw=state_next,
                state_prev=state_t,
                state_clip=env["state_clip"],
                state_highway_enabled=env.get("state_highway_enabled", False),
                state_highway_lambda=env.get("state_highway_lambda", 0.0),
            )
            aev4_step_aux = None
            if aev4_enabled:
                state_next, aev4_step_aux = self._apply_aev4_state_update(
                    state_prev=state_t,
                    state_next_post=state_next,
                    aev4_cfg=aev4_cfg,
                )
            state_delta = state_next - state_t
            if aev2_enabled and (aev2_prev_delta is not None):
                self._aev2_update_accumulator(aev2_acc, aev2_prev_delta, state_delta, aev2_cfg)
            aev2_prev_delta = state_delta
            if aev3_enabled and (aev3_prev_delta is not None):
                self._aev3_update_accumulator(aev3_acc, aev3_prev_delta, state_delta, aev3_cfg)
            aev3_prev_delta = state_delta
            if aev4_enabled and (aev4_prev_delta is not None):
                self._aev4_update_accumulator(aev4_acc, aev4_prev_delta, state_delta, aev4_cfg, step_aux=aev4_step_aux)
            aev4_prev_delta = state_delta

            if tbptt_window_active:
                if y_steps is not None:
                    y_steps[t] = reward_next.detach()
                tbptt_reward_buffer.append(reward_next)
            else:
                if y_steps is not None:
                    y_steps[t] = reward_next
            if collect_runtime_info:
                reward_values[t] = reward_next.detach()
                state_abs_max[t] = torch.abs(state_next).max().detach()

            state_t = state_next
            action_t = action_next
            reward_t = reward_next.reshape(())
            reward_mask_t = reward_mask_next.reshape(())

            if tbptt_window_active:
                is_window_end = (len(tbptt_reward_buffer) >= tbptt_window_size) or (t == (n_samples - 1))
                if is_window_end:
                    rewards_window = torch.stack(tbptt_reward_buffer, dim=0).reshape(-1, 1)
                    tbptt_reward_buffer = []
                    if t < (n_samples - 1):
                        state_t = state_t.detach()
                        action_t = action_t.detach()
                        reward_t = reward_t.detach()
                        reward_mask_t = reward_mask_t.detach()
                        if aev2_prev_delta is not None:
                            aev2_prev_delta = aev2_prev_delta.detach()
                        if aev3_prev_delta is not None:
                            aev3_prev_delta = aev3_prev_delta.detach()
                        if aev4_prev_delta is not None:
                            aev4_prev_delta = aev4_prev_delta.detach()
                        cache = self._detach_policy_cache(cache, clone_tensors=(tbptt_reward_sink is None))
                    if tbptt_reward_sink is not None:
                        if aev2_streaming_sink or aev3_streaming_sink or aev4_streaming_sink:
                            payload_aux = {}
                            if aev2_streaming_sink:
                                aev2_window_summary = self._aev2_finalize_accumulator(
                                    aev2_acc,
                                    device=device,
                                    dtype=state_t.dtype,
                                    detach_penalty=False,
                                )
                                payload_aux["aev2"] = aev2_window_summary
                                self._aev2_accumulate_window_summary(aev2_det_acc, aev2_window_summary, device=device)
                                aev2_acc = self._aev2_new_accumulator(
                                    aev2_enabled, device=device, dtype=state_t.dtype
                                )
                            if aev3_streaming_sink:
                                aev3_window_summary = self._aev3_finalize_accumulator(
                                    aev3_acc,
                                    device=device,
                                    dtype=state_t.dtype,
                                    detach_penalty=False,
                                )
                                payload_aux["aev3"] = aev3_window_summary
                                self._aev3_accumulate_window_summary(aev3_det_acc, aev3_window_summary, device=device)
                                aev3_acc = self._aev3_new_accumulator(
                                    aev3_enabled,
                                    device=device,
                                    dtype=state_t.dtype,
                                    aev3_cfg=aev3_cfg,
                                )
                            if aev4_streaming_sink:
                                aev4_window_summary = self._aev4_finalize_accumulator(
                                    aev4_acc,
                                    device=device,
                                    dtype=state_t.dtype,
                                    detach_penalty=False,
                                )
                                payload_aux["aev4"] = aev4_window_summary
                                self._aev4_accumulate_window_summary(aev4_det_acc, aev4_window_summary, device=device)
                                aev4_acc = self._aev4_new_accumulator(
                                    aev4_enabled,
                                    device=device,
                                    dtype=state_t.dtype,
                                    aev4_cfg=aev4_cfg,
                                )
                            if ("aev2" in payload_aux) and (len(payload_aux) == 1):
                                tbptt_reward_sink((rewards_window, payload_aux["aev2"]))
                            else:
                                tbptt_reward_sink((rewards_window, payload_aux))
                        else:
                            tbptt_reward_sink(rewards_window)

        x = x_steps
        y = y_steps
        if collect_runtime_info:
            rewards_for_stats = reward_values
            state_abs_for_stats = state_abs_max
            info = {
                "family": env["family"],
                "state_dim": env["state_dim"],
                "obs_dim": env["obs_dim"],
                "action_dim": env["action_dim"],
                "noise_dim": env["noise_dim"],
                "zero_pad_dim": env["zero_pad_dim"],
                "obs_slot_dim": env["obs_slot_dim"],
                "action_slot_dim": env["action_slot_dim"],
                "single_eval_pos": int(single_eval_pos),
                "reward_min": float(rewards_for_stats.min().cpu()),
                "reward_max": float(rewards_for_stats.max().cpu()),
                "reward_std": float(rewards_for_stats.std(unbiased=False).cpu()),
                "state_abs_max": float(state_abs_for_stats.max().cpu()),
                "reward_dropout_ratio": float(env["reward_dropout_ratio"]),
                "reward_drop_frac_realized": float(reward_drop_count / max(1, int(n_samples))),
                "state_highway_enabled": bool(env.get("state_highway_enabled", False)),
                "state_highway_lambda": float(env.get("state_highway_lambda", 0.0)),
                "aev4_enabled": bool(env.get("aev4_enabled", False)),
                "aev4_highway_ratio": float(env.get("aev4_highway_ratio", 0.25)),
                "aev4_update_scale": float(env.get("aev4_update_scale", 0.12)),
                "aev4_update_clip": float(env.get("aev4_update_clip", 0.0)),
            }
        else:
            info = None
        if aev2_enabled:
            if aev2_streaming_sink:
                self.last_rollout_v2 = self._aev2_finalize_accumulator(
                    aev2_det_acc,
                    device=device,
                    dtype=state_t.dtype,
                    detach_penalty=True,
                )
            else:
                self.last_rollout_v2 = self._aev2_finalize_accumulator(
                    aev2_acc,
                    device=device,
                    dtype=state_t.dtype,
                    detach_penalty=False,
                )
            self.last_rollout_v2["lambda"] = float(aev2_cfg.get("lambda", 0.0))
            self.last_rollout_v2["gain_lo"] = float(aev2_cfg.get("gain_lo", 0.0))
            self.last_rollout_v2["gain_hi"] = float(aev2_cfg.get("gain_hi", 0.0))
        else:
            self.last_rollout_v2 = None
        if aev3_enabled:
            if aev3_streaming_sink:
                self.last_rollout_v3 = self._aev3_finalize_accumulator(
                    aev3_det_acc,
                    device=device,
                    dtype=state_t.dtype,
                    detach_penalty=True,
                )
            else:
                self.last_rollout_v3 = self._aev3_finalize_accumulator(
                    aev3_acc,
                    device=device,
                    dtype=state_t.dtype,
                    detach_penalty=False,
                )
        else:
            self.last_rollout_v3 = None
        if aev4_enabled:
            if aev4_streaming_sink:
                self.last_rollout_v4 = self._aev4_finalize_accumulator(
                    aev4_det_acc,
                    device=device,
                    dtype=state_t.dtype,
                    detach_penalty=True,
                )
            else:
                self.last_rollout_v4 = self._aev4_finalize_accumulator(
                    aev4_acc,
                    device=device,
                    dtype=state_t.dtype,
                    detach_penalty=False,
                )
        else:
            self.last_rollout_v4 = None
        return x, y, info

    def _get_rollout_executor(self, workers):
        workers = int(max(1, workers))
        if self._rollout_executor is None or self._rollout_executor_workers != workers:
            if self._rollout_executor is not None:
                self._rollout_executor.shutdown(wait=True)
            self._rollout_executor = ThreadPoolExecutor(
                max_workers=workers,
                thread_name_prefix="envprior-rollout",
            )
            self._rollout_executor_workers = workers
        return self._rollout_executor

    def _resolve_batch_parallel_workers(self, batch_size):
        workers_cfg = self.config.get("batch_parallel_workers", 1)
        if isinstance(workers_cfg, dict) and "distribution" in workers_cfg:
            workers = int(sample_distributions({"v": workers_cfg})["v"])
        else:
            workers = int(workers_cfg)
        workers = max(1, workers)
        return min(int(batch_size), workers)

    def _resolve_batch_parallel_backend(self):
        backend_cfg = self.config.get("batch_parallel_backend", "python_thread")
        if isinstance(backend_cfg, dict) and "distribution" in backend_cfg:
            backend = sample_distributions({"v": backend_cfg})["v"]
        else:
            backend = backend_cfg
        backend = str(backend).strip().lower()
        if backend not in {"python_thread", "torch_vectorized"}:
            backend = "python_thread"
        return backend

    def _resolve_batch_shared_environment(self):
        shared_cfg = self.config.get("batch_shared_environment", False)
        if isinstance(shared_cfg, dict) and "distribution" in shared_cfg:
            shared = sample_distributions({"v": shared_cfg})["v"]
        else:
            shared = shared_cfg
        if isinstance(shared, str):
            token = shared.strip().lower()
            if token in {"1", "true", "yes", "on"}:
                return True
            if token in {"0", "false", "no", "off"}:
                return False
        return bool(shared)

    def _resolve_batch_vectorized_strict_rng_match(self):
        strict_cfg = self.config.get("batch_vectorized_strict_rng_match", False)
        if isinstance(strict_cfg, dict) and "distribution" in strict_cfg:
            strict = sample_distributions({"v": strict_cfg})["v"]
        else:
            strict = strict_cfg
        if isinstance(strict, str):
            token = strict.strip().lower()
            if token in {"1", "true", "yes", "on"}:
                return True
            if token in {"0", "false", "no", "off"}:
                return False
        return bool(strict)

    def _resolve_batch_vectorized_grouping(self):
        grouping_cfg = self.config.get("batch_vectorized_grouping", "structure")
        if isinstance(grouping_cfg, dict) and "distribution" in grouping_cfg:
            grouping = sample_distributions({"v": grouping_cfg})["v"]
        else:
            grouping = grouping_cfg
        grouping = str(grouping).strip().lower()
        if grouping not in {"structure", "family"}:
            grouping = "structure"
        return grouping

    def _rollout_shared_env_vectorized(
        self,
        env,
        batch_size,
        n_samples,
        num_features,
        single_eval_pos,
        device,
        collect_x=True,
        rng_seeds=None,
    ):
        if not collect_x:
            raise ValueError("get_batch vectorized rollout requires collect_x=True")

        n_samples = int(n_samples)
        batch_size = int(batch_size)
        num_features = int(num_features)
        rollout_generators = self._make_generators_from_seeds(rng_seeds, batch_size, device)

        state_dim = int(env["state_dim"])
        obs_dim = int(env["obs_dim"])
        action_dim = int(env["action_dim"])
        noise_dim = int(env["noise_dim"])
        zero_pad_dim = int(env["zero_pad_dim"])
        obs_slot_dim = int(env["obs_slot_dim"])
        action_slot_dim = int(env["action_slot_dim"])

        state_t = self._stack_randn_with_generators(
            rollout_generators,
            (batch_size, state_dim),
            device=device,
            dtype=torch.float32,
        ) * env["init_state_std"]
        action_t = self._stack_randn_with_generators(
            rollout_generators,
            (batch_size, action_dim),
            device=device,
            dtype=torch.float32,
        ) * env["init_action_std"]
        reward_t = torch.zeros((batch_size,), device=device, dtype=torch.float32)
        reward_mask_t = torch.ones((batch_size,), device=device, dtype=torch.float32)

        x_steps = torch.empty((n_samples, batch_size, num_features), device=device, dtype=torch.float32)
        y_steps = torch.empty((n_samples, batch_size), device=device, dtype=torch.float32)
        state_abs_max = torch.empty((n_samples, batch_size), device=device, dtype=torch.float32)
        reward_values = torch.empty((n_samples, batch_size), device=device, dtype=torch.float32)
        reward_drop_count = torch.zeros((batch_size,), device=device, dtype=torch.int64)

        transition_noise = self._stack_randn_with_generators(
            rollout_generators,
            (batch_size, n_samples, noise_dim),
            device=device,
            dtype=torch.float32,
        ).transpose(0, 1)
        action_noise_train = None
        action_noise_eval = None
        state_noise = None
        if env["action_noise_train_std"] > 0:
            action_noise_train = self._stack_randn_with_generators(
                rollout_generators,
                (batch_size, n_samples, action_dim),
                device=device,
                dtype=torch.float32,
            ).transpose(0, 1)
        if env["action_noise_eval_std"] > 0:
            action_noise_eval = self._stack_randn_with_generators(
                rollout_generators,
                (batch_size, n_samples, action_dim),
                device=device,
                dtype=torch.float32,
            ).transpose(0, 1)
        if env["state_noise_std"] > 0:
            state_noise = self._stack_randn_with_generators(
                rollout_generators,
                (batch_size, n_samples, state_dim),
                device=device,
                dtype=torch.float32,
            ).transpose(0, 1)
        dropout_draws = None
        if env["reward_dropout_enabled"] and env["reward_dropout_ratio"] > 0.0:
            dropout_draws = self._stack_rand_with_generators(
                rollout_generators,
                (batch_size, n_samples),
                device=device,
                dtype=torch.float32,
            ).transpose(0, 1)

        token_reward_idx = obs_slot_dim
        token_mask_idx = obs_slot_dim + 1
        token_action_start = obs_slot_dim + 2
        token_obs_cap = min(obs_dim, obs_slot_dim)
        token_action_cap = min(action_dim, action_slot_dim)

        env_total_dim = state_dim + obs_dim + action_dim + noise_dim + zero_pad_dim
        env_in = torch.zeros((batch_size, env_total_dim), device=device, dtype=torch.float32)
        env_obs_start = state_dim
        env_action_start = state_dim + obs_dim
        env_noise_start = env_action_start + action_dim

        alpha = float(env["alpha"])
        reward_scale = float(env["reward_scale"])
        reward_clip = float(env.get("reward_clip", 10.0))
        state_clip = float(env["state_clip"])
        action_noise_train_std = float(env["action_noise_train_std"])
        action_noise_eval_std = float(env["action_noise_eval_std"])
        state_noise_std = float(env["state_noise_std"])
        reward_dropout_ratio = float(env["reward_dropout_ratio"])
        reward_impute_zero = bool(env["reward_dropout_impute_zero"])
        for t in range(n_samples):
            obs_t = state_t[:, :obs_dim]
            token_row = x_steps[t]
            token_row.zero_()
            if num_features > 0 and token_obs_cap > 0:
                obs_write = min(token_obs_cap, num_features)
                token_row[:, :obs_write] = obs_t[:, :obs_write]
            if token_reward_idx < num_features:
                token_row[:, token_reward_idx] = reward_t
            if token_mask_idx < num_features:
                token_row[:, token_mask_idx] = reward_mask_t
            if token_action_start < num_features and token_action_cap > 0:
                action_write = min(token_action_cap, num_features - token_action_start)
                token_row[:, token_action_start: token_action_start + action_write] = action_t[:, :action_write]

            noise_t = transition_noise[t]
            env_in[:, :state_dim] = state_t
            env_in[:, env_obs_start: env_obs_start + obs_dim] = obs_t
            env_in[:, env_action_start: env_action_start + action_dim] = action_t
            env_in[:, env_noise_start: env_noise_start + noise_dim] = noise_t
            if rollout_generators is None:
                action_next = torch.tanh(env["policy_generator"](env_in))
            else:
                action_next = torch.empty((batch_size, action_dim), device=device, dtype=torch.float32)
                for bi, g in enumerate(rollout_generators):
                    action_next[bi] = torch.tanh(env["policy_generator"](env_in[bi: bi + 1], generator=g).squeeze(0))

            if t < single_eval_pos:
                if action_noise_train is not None:
                    action_next = torch.tanh(action_next + action_noise_train[t] * action_noise_train_std)
            elif action_noise_eval is not None:
                action_next = torch.tanh(action_next + action_noise_eval[t] * action_noise_eval_std)

            env_in[:, env_action_start: env_action_start + action_dim] = action_next
            if rollout_generators is None:
                reward_next_raw = reward_scale * env["y_generator"](env_in).reshape(batch_size)
            else:
                reward_next_raw = torch.empty((batch_size,), device=device, dtype=torch.float32)
                for bi, g in enumerate(rollout_generators):
                    reward_next_raw[bi] = reward_scale * env["y_generator"](env_in[bi: bi + 1], generator=g).reshape(())
            reward_next = torch.clamp(reward_next_raw, -reward_clip, reward_clip)
            reward_mask_next = torch.ones((batch_size,), device=device, dtype=torch.float32)

            if dropout_draws is not None:
                drop_mask = dropout_draws[t] < reward_dropout_ratio
                reward_drop_count = reward_drop_count + drop_mask.to(dtype=torch.int64)
                reward_mask_next = torch.where(drop_mask, torch.zeros_like(reward_mask_next), reward_mask_next)
                if reward_impute_zero:
                    reward_next = torch.where(drop_mask, torch.zeros_like(reward_next), reward_next)

            if rollout_generators is None:
                x_next = env["x_generator"](env_in)
            else:
                x_next = torch.empty((batch_size, state_dim), device=device, dtype=torch.float32)
                for bi, g in enumerate(rollout_generators):
                    x_next[bi] = env["x_generator"](env_in[bi: bi + 1], generator=g).squeeze(0)
            state_next = (1.0 - alpha) * state_t + alpha * x_next
            if state_noise is not None:
                state_next = state_next + state_noise[t] * state_noise_std
            state_next = self._apply_state_postprocess(
                state_next_raw=state_next,
                state_prev=state_t,
                state_clip=state_clip,
                state_highway_enabled=env.get("state_highway_enabled", False),
                state_highway_lambda=env.get("state_highway_lambda", 0.0),
            )

            y_steps[t] = reward_next
            reward_values[t] = reward_next.detach()
            state_abs_max[t] = state_next.detach().abs().amax(dim=1)

            state_t = state_next
            action_t = action_next
            reward_t = reward_next
            reward_mask_t = reward_mask_next

        reward_drop_frac = reward_drop_count.to(dtype=torch.float32) / float(max(1, n_samples))
        infos = self._build_vectorized_runtime_info(
            env=env,
            reward_values=reward_values,
            state_abs_max=state_abs_max,
            reward_drop_frac=reward_drop_frac,
            single_eval_pos=single_eval_pos,
        )

        if y_steps is None:
            y_steps = torch.empty((0,), device=device, dtype=state_t.dtype)
        return x_steps, y_steps, infos

    def get_batch(self, batch_size, n_samples, num_features, device=default_device, epoch=None, single_eval_pos=None):
        with torch.no_grad():
            single_eval_pos = self._sample_single_eval_pos(int(n_samples), single_eval_pos)

            n_samples = int(n_samples)
            batch_size = int(batch_size)
            num_features = int(num_features)
            self.last_runtime_info = [None] * batch_size
            backend = self._resolve_batch_parallel_backend()
            shared_environment = self._resolve_batch_shared_environment()
            strict_rng_match = self._resolve_batch_vectorized_strict_rng_match()
            h_list = self._sample_batch_hypers(batch_size)
            env_seeds = self._sample_seed_list(batch_size) if strict_rng_match else None
            rollout_seeds = self._sample_seed_list(batch_size) if strict_rng_match else None

            if backend == "torch_vectorized":
                if shared_environment:
                    h = h_list[0]
                    shared_env = self._sample_environment(
                        h=h,
                        device=device,
                        rng_seed=(env_seeds[0] if env_seeds is not None else None),
                    )
                    x_vec, y_vec, infos = self._rollout_shared_env_vectorized(
                        env=shared_env,
                        batch_size=batch_size,
                        n_samples=n_samples,
                        num_features=num_features,
                        single_eval_pos=single_eval_pos,
                        device=device,
                        collect_x=True,
                        rng_seeds=rollout_seeds,
                    )
                else:
                    grouped = {}
                    for b, h in enumerate(h_list):
                        sig = self._environment_structure_signature(h)
                        grouped.setdefault(sig, []).append((b, h))

                    x_vec = torch.empty((n_samples, batch_size, num_features), device=device, dtype=torch.float32)
                    y_vec = torch.empty((n_samples, batch_size), device=device, dtype=torch.float32)
                    infos = [None] * batch_size
                    for group in grouped.values():
                        group_indices = [idx for idx, _ in group]
                        group_h_list = [h for _, h in group]
                        group_env_seeds = (
                            [env_seeds[idx] for idx in group_indices]
                            if env_seeds is not None
                            else None
                        )
                        group_rollout_seeds = (
                            [rollout_seeds[idx] for idx in group_indices]
                            if rollout_seeds is not None
                            else None
                        )
                        env_batch = self._sample_environment_batch(
                            h_list=group_h_list,
                            device=device,
                            rng_seeds=group_env_seeds,
                        )
                        x_group, y_group, infos_group = self._rollout_distinct_envs_vectorized(
                            env=env_batch,
                            batch_size=len(group_indices),
                            n_samples=n_samples,
                            num_features=num_features,
                            single_eval_pos=single_eval_pos,
                            device=device,
                            collect_x=True,
                            rng_seeds=group_rollout_seeds,
                        )
                        x_vec[:, group_indices] = x_group
                        y_vec[:, group_indices] = y_group
                        for local_idx, global_idx in enumerate(group_indices):
                            infos[global_idx] = infos_group[local_idx]
                self.last_runtime_info = infos
                return x_vec, y_vec, y_vec

            tasks = []
            x = torch.empty((n_samples, batch_size, num_features), device=device, dtype=torch.float32)
            y = torch.empty((n_samples, batch_size), device=device, dtype=torch.float32)
            shared_env = None
            if shared_environment:
                h = h_list[0]
                shared_env = self._sample_environment(
                    h=h,
                    device=device,
                    rng_seed=(env_seeds[0] if env_seeds is not None else None),
                )

            for b in range(batch_size):
                env = shared_env
                if env is None:
                    h = h_list[b]
                    env = self._sample_environment(
                        h=h,
                        device=device,
                        rng_seed=(env_seeds[b] if env_seeds is not None else None),
                    )
                seed = rollout_seeds[b] if rollout_seeds is not None else None
                tasks.append((b, env, seed))

            workers = self._resolve_batch_parallel_workers(batch_size)
            if backend != "python_thread":
                # Avoid non-torch parallel backends when torch_vectorized is requested
                # but strict vectorization is unavailable (e.g., non-shared environments).
                workers = 1

            if workers <= 1:
                for b, env, seed in tasks:
                    x_one, y_one, info = self._rollout_single(
                        env=env,
                        n_samples=n_samples,
                        num_features=num_features,
                        single_eval_pos=single_eval_pos,
                        device=device,
                        policy_step_fn=None,
                        collect_x=True,
                        rng_seed=seed,
                    )
                    x[:, b] = x_one
                    y[:, b] = y_one
                    self.last_runtime_info[b] = info
            else:
                executor = self._get_rollout_executor(workers)
                futures = []
                for b, env, seed in tasks:
                    futures.append(
                        executor.submit(
                            self._rollout_single,
                            env,
                            n_samples,
                            num_features,
                            single_eval_pos,
                            device,
                            None,
                            True,
                            True,
                            seed,
                        )
                    )
                for b, fut in enumerate(futures):
                    x_one, y_one, info = fut.result()
                    x[:, b] = x_one
                    y[:, b] = y_one
                    self.last_runtime_info[b] = info

        return x, y, y

    def rollout_with_policy(
        self,
        policy_step_fn,
        batch_size,
        n_samples,
        num_features,
        device=default_device,
        epoch=None,
        single_eval_pos=None,
        collect_x=True,
        collect_runtime_info=True,
        tbptt_window=None,
        tbptt_reward_sink=None,
        tbptt_reward_sink_supports_aux=False,
        h_list_override=None,
        env_seeds_override=None,
        rollout_seeds_override=None,
        store_rewards=True,
    ):
        """
        Differentiable rollout for policy optimization.

        policy_step_fn signature:
            preferred:
              (obs_t, action_t, reward_t, reward_mask_t, cache, step_idx, env_info)
                -> action_next or (action_next, cache)
            backward-compatible:
              (obs_t, action_t, reward_t, cache, step_idx, env_info)
        """
        single_eval_pos = self._sample_single_eval_pos(int(n_samples), single_eval_pos)
        n_samples = int(n_samples)
        batch_size = int(batch_size)
        num_features = int(num_features)
        collect_runtime_info = bool(collect_runtime_info)
        self.last_rollout_profile = None
        self.last_rollout_v2 = None
        self.last_rollout_v3 = None
        store_rewards = bool(store_rewards)
        x = torch.empty((n_samples, batch_size, num_features), device=device, dtype=torch.float32) if collect_x else None
        rewards = (
            torch.empty((n_samples, batch_size), device=device, dtype=torch.float32)
            if store_rewards
            else torch.empty((0, batch_size), device=device, dtype=torch.float32)
        )
        infos = [None] * batch_size

        backend = self._resolve_batch_parallel_backend()
        strict_rng_match = self._resolve_batch_vectorized_strict_rng_match()
        grouping_mode = self._resolve_batch_vectorized_grouping()
        if h_list_override is not None:
            if len(h_list_override) != batch_size:
                raise ValueError("h_list_override length must match batch_size")
            h_list = list(h_list_override)
        else:
            h_list = self._sample_batch_hypers(batch_size)
        if env_seeds_override is not None:
            if len(env_seeds_override) != batch_size:
                raise ValueError("env_seeds_override length must match batch_size")
            env_seeds = [int(s) for s in env_seeds_override]
        else:
            env_seeds = self._sample_seed_list(batch_size) if strict_rng_match else None
        if rollout_seeds_override is not None:
            if len(rollout_seeds_override) != batch_size:
                raise ValueError("rollout_seeds_override length must match batch_size")
            rollout_seeds = [int(s) for s in rollout_seeds_override]
        else:
            rollout_seeds = self._sample_seed_list(batch_size) if strict_rng_match else None

        if backend == "torch_vectorized":
            effective_grouping_mode = str(grouping_mode)
            # strict_rng_match path is used for semantic A/B tests and keeps the
            # legacy structural grouping behavior.
            if strict_rng_match:
                effective_grouping_mode = "structure"
            grouped = {}
            for b, h in enumerate(h_list):
                sig = self._environment_group_signature(h, effective_grouping_mode)
                grouped.setdefault(sig, []).append((b, h))
            rollout_profile_acc = None
            rollout_v2_acc = None
            rollout_v3_acc = None
            rollout_v4_acc = None

            for group in grouped.values():
                group_indices = [idx for idx, _ in group]
                group_h_list = [h for _, h in group]
                group_env_seeds = (
                    [env_seeds[idx] for idx in group_indices]
                    if env_seeds is not None
                    else None
                )
                group_rollout_seeds = (
                    [rollout_seeds[idx] for idx in group_indices]
                    if rollout_seeds is not None
                    else None
                )
                if effective_grouping_mode == "family":
                    x_group, y_group, infos_group = self._rollout_family_group_vectorized_with_policy(
                        h_list=group_h_list,
                        policy_step_fn=policy_step_fn,
                        n_samples=n_samples,
                        num_features=num_features,
                        single_eval_pos=single_eval_pos,
                        device=device,
                        collect_x=collect_x,
                        collect_runtime_info=collect_runtime_info,
                        env_rng_seeds=group_env_seeds,
                        rollout_rng_seeds=group_rollout_seeds,
                        tbptt_window=tbptt_window,
                        tbptt_reward_sink=tbptt_reward_sink,
                        tbptt_reward_sink_supports_aux=tbptt_reward_sink_supports_aux,
                        store_rewards=store_rewards,
                    )
                else:
                    env_batch = self._sample_environment_batch(
                        h_list=group_h_list,
                        device=device,
                        rng_seeds=group_env_seeds,
                    )
                    x_group, y_group, infos_group = self._rollout_distinct_envs_vectorized_with_policy(
                        env=env_batch,
                        policy_step_fn=policy_step_fn,
                        batch_size=len(group_indices),
                        n_samples=n_samples,
                        num_features=num_features,
                        single_eval_pos=single_eval_pos,
                        device=device,
                        collect_x=collect_x,
                        collect_runtime_info=collect_runtime_info,
                        rng_seeds=group_rollout_seeds,
                        tbptt_window=tbptt_window,
                        tbptt_reward_sink=tbptt_reward_sink,
                        tbptt_reward_sink_supports_aux=tbptt_reward_sink_supports_aux,
                        store_rewards=store_rewards,
                    )
                if collect_x:
                    x[:, group_indices] = x_group
                if store_rewards:
                    rewards[:, group_indices] = y_group
                for local_idx, global_idx in enumerate(group_indices):
                    infos[global_idx] = infos_group[local_idx]
                group_v2 = self.last_rollout_v2
                rollout_v2_acc = self._aev2_merge_rollout_summary(
                    rollout_v2_acc,
                    group_v2,
                    batch_weight=(float(len(group_indices)) / float(max(1, batch_size))),
                    device=device,
                    dtype=torch.float32,
                )
                group_v3 = self.last_rollout_v3
                rollout_v3_acc = self._aev3_merge_rollout_summary(
                    rollout_v3_acc,
                    group_v3,
                    batch_weight=(float(len(group_indices)) / float(max(1, batch_size))),
                    device=device,
                    dtype=torch.float32,
                )
                group_v4 = self.last_rollout_v4
                rollout_v4_acc = self._aev4_merge_rollout_summary(
                    rollout_v4_acc,
                    group_v4,
                    batch_weight=(float(len(group_indices)) / float(max(1, batch_size))),
                    device=device,
                    dtype=torch.float32,
                )
                group_profile = self.last_rollout_profile
                if isinstance(group_profile, dict):
                    if rollout_profile_acc is None:
                        rollout_profile_acc = {
                            "policy_cuda_ms": 0.0,
                            "transition_cuda_ms": 0.0,
                            "policy_wall_ms": 0.0,
                            "transition_wall_ms": 0.0,
                            "transition_y_wall_ms": 0.0,
                            "transition_x_wall_ms": 0.0,
                            "transition_group_wall_ms": 0.0,
                            "transition_group_launch_wall_ms": 0.0,
                            "transition_group_sync_wall_ms": 0.0,
                            "transition_env_pack_wall_ms": 0.0,
                            "transition_state_update_wall_ms": 0.0,
                            "transition_noise_wall_ms": 0.0,
                            "transition_fused_wall_ms": 0.0,
                            "transition_fused_launch_wall_ms": 0.0,
                            "transition_gp_first_projection_wall_ms": 0.0,
                            "transition_gp_second_projection_wall_ms": 0.0,
                            "transition_gp_projection_call_count": 0,
                            "transition_gp_rff_fused_call_count": 0,
                            "transition_gp_profile_group_count": 0,
                            "transition_gp_profile_sync_group_count": 0,
                            "transition_gp_shared_total_wall_ms": 0.0,
                            "transition_gp_shared_core_wall_ms": 0.0,
                            "transition_gp_shared_noise_wall_ms": 0.0,
                            "transition_gp_shared_checkpoint_wall_ms": 0.0,
                            "transition_gp_shared_post_wall_ms": 0.0,
                            "transition_gp_shared_call_count": 0,
                            "transition_packed_env_input_group_count": 0,
                            "transition_packed_env_input_call_count": 0,
                            "transition_only_build_group_count": 0,
                            "transition_only_skipped_generator_count": 0,
                            "transition_setup_wall_ms": 0.0,
                            "transition_family_build_wall_ms": 0.0,
                            "transition_generator_build_wall_ms": 0.0,
                            "transition_gp_shared_build_wall_ms": 0.0,
                            "transition_fused_call_count": 0,
                            "transition_fused_group_count": 0,
                            "transition_fused_enabled": 0,
                            "transition_checkpoint_enabled": 0,
                            "transition_checkpoint_call_count": 0,
                            "transition_group_count": 0,
                            "transition_family_group_count": 0,
                            "transition_inner_grouping_structure_enabled": 0,
                            "transition_inner_min_bucket": 0,
                            "transition_bucket_max_batch": 0,
                            "transition_work_actual_est": 0.0,
                            "transition_work_padded_est": 0.0,
                            "transition_async_enabled": 0,
                            "noise_mode": None,
                            "noise_block_size": 0,
                            "steps": int(n_samples),
                            "batch_size": int(batch_size),
                        }
                    rollout_profile_acc["policy_cuda_ms"] += float(group_profile.get("policy_cuda_ms", 0.0))
                    rollout_profile_acc["transition_cuda_ms"] += float(group_profile.get("transition_cuda_ms", 0.0))
                    rollout_profile_acc["policy_wall_ms"] += float(group_profile.get("policy_wall_ms", 0.0))
                    rollout_profile_acc["transition_wall_ms"] += float(group_profile.get("transition_wall_ms", 0.0))
                    rollout_profile_acc["transition_y_wall_ms"] += float(group_profile.get("transition_y_wall_ms", 0.0))
                    rollout_profile_acc["transition_x_wall_ms"] += float(group_profile.get("transition_x_wall_ms", 0.0))
                    rollout_profile_acc["transition_group_wall_ms"] += float(
                        group_profile.get("transition_group_wall_ms", 0.0)
                    )
                    rollout_profile_acc["transition_group_launch_wall_ms"] += float(
                        group_profile.get("transition_group_launch_wall_ms", 0.0)
                    )
                    rollout_profile_acc["transition_group_sync_wall_ms"] += float(
                        group_profile.get("transition_group_sync_wall_ms", 0.0)
                    )
                    rollout_profile_acc["transition_env_pack_wall_ms"] += float(
                        group_profile.get("transition_env_pack_wall_ms", 0.0)
                    )
                    rollout_profile_acc["transition_state_update_wall_ms"] += float(
                        group_profile.get("transition_state_update_wall_ms", 0.0)
                    )
                    rollout_profile_acc["transition_noise_wall_ms"] += float(
                        group_profile.get("transition_noise_wall_ms", 0.0)
                    )
                    rollout_profile_acc["transition_fused_wall_ms"] += float(
                        group_profile.get("transition_fused_wall_ms", 0.0)
                    )
                    rollout_profile_acc["transition_fused_launch_wall_ms"] += float(
                        group_profile.get("transition_fused_launch_wall_ms", 0.0)
                    )
                    rollout_profile_acc["transition_gp_first_projection_wall_ms"] += float(
                        group_profile.get("transition_gp_first_projection_wall_ms", 0.0)
                    )
                    rollout_profile_acc["transition_gp_second_projection_wall_ms"] += float(
                        group_profile.get("transition_gp_second_projection_wall_ms", 0.0)
                    )
                    rollout_profile_acc["transition_gp_projection_call_count"] += int(
                        group_profile.get("transition_gp_projection_call_count", 0) or 0
                    )
                    rollout_profile_acc["transition_gp_rff_fused_call_count"] += int(
                        group_profile.get("transition_gp_rff_fused_call_count", 0) or 0
                    )
                    rollout_profile_acc["transition_gp_profile_group_count"] += int(
                        group_profile.get("transition_gp_profile_group_count", 0) or 0
                    )
                    rollout_profile_acc["transition_gp_profile_sync_group_count"] += int(
                        group_profile.get("transition_gp_profile_sync_group_count", 0) or 0
                    )
                    rollout_profile_acc["transition_gp_shared_total_wall_ms"] += float(
                        group_profile.get("transition_gp_shared_total_wall_ms", 0.0) or 0.0
                    )
                    rollout_profile_acc["transition_gp_shared_core_wall_ms"] += float(
                        group_profile.get("transition_gp_shared_core_wall_ms", 0.0) or 0.0
                    )
                    rollout_profile_acc["transition_gp_shared_noise_wall_ms"] += float(
                        group_profile.get("transition_gp_shared_noise_wall_ms", 0.0) or 0.0
                    )
                    rollout_profile_acc["transition_gp_shared_checkpoint_wall_ms"] += float(
                        group_profile.get("transition_gp_shared_checkpoint_wall_ms", 0.0) or 0.0
                    )
                    rollout_profile_acc["transition_gp_shared_post_wall_ms"] += float(
                        group_profile.get("transition_gp_shared_post_wall_ms", 0.0) or 0.0
                    )
                    rollout_profile_acc["transition_gp_shared_call_count"] += int(
                        group_profile.get("transition_gp_shared_call_count", 0) or 0
                    )
                    rollout_profile_acc["transition_packed_env_input_group_count"] += int(
                        group_profile.get("transition_packed_env_input_group_count", 0) or 0
                    )
                    rollout_profile_acc["transition_packed_env_input_call_count"] += int(
                        group_profile.get("transition_packed_env_input_call_count", 0) or 0
                    )
                    rollout_profile_acc["transition_only_build_group_count"] += int(
                        group_profile.get("transition_only_build_group_count", 0) or 0
                    )
                    rollout_profile_acc["transition_only_skipped_generator_count"] += int(
                        group_profile.get("transition_only_skipped_generator_count", 0) or 0
                    )
                    rollout_profile_acc["transition_setup_wall_ms"] += float(
                        group_profile.get("transition_setup_wall_ms", 0.0) or 0.0
                    )
                    rollout_profile_acc["transition_family_build_wall_ms"] += float(
                        group_profile.get("transition_family_build_wall_ms", 0.0) or 0.0
                    )
                    rollout_profile_acc["transition_generator_build_wall_ms"] += float(
                        group_profile.get("transition_generator_build_wall_ms", 0.0) or 0.0
                    )
                    rollout_profile_acc["transition_gp_shared_build_wall_ms"] += float(
                        group_profile.get("transition_gp_shared_build_wall_ms", 0.0) or 0.0
                    )
                    rollout_profile_acc["transition_fused_call_count"] += int(
                        group_profile.get("transition_fused_call_count", 0)
                    )
                    rollout_profile_acc["transition_fused_group_count"] += int(
                        group_profile.get("transition_fused_group_count", 0)
                    )
                    rollout_profile_acc["transition_fused_enabled"] = int(
                        max(
                            int(rollout_profile_acc.get("transition_fused_enabled", 0) or 0),
                            int(group_profile.get("transition_fused_enabled", 0) or 0),
                        )
                    )
                    rollout_profile_acc["transition_checkpoint_enabled"] = int(
                        max(
                            int(rollout_profile_acc.get("transition_checkpoint_enabled", 0) or 0),
                            int(group_profile.get("transition_checkpoint_enabled", 0) or 0),
                        )
                    )
                    rollout_profile_acc["transition_checkpoint_call_count"] += int(
                        group_profile.get("transition_checkpoint_call_count", 0) or 0
                    )
                    rollout_profile_acc["transition_group_count"] += int(group_profile.get("transition_group_count", 0))
                    rollout_profile_acc["transition_family_group_count"] += int(
                        group_profile.get("transition_family_group_count", 0) or 0
                    )
                    rollout_profile_acc["transition_inner_grouping_structure_enabled"] = int(
                        max(
                            int(rollout_profile_acc.get("transition_inner_grouping_structure_enabled", 0) or 0),
                            int(group_profile.get("transition_inner_grouping_structure_enabled", 0) or 0),
                        )
                    )
                    rollout_profile_acc["transition_inner_min_bucket"] = int(
                        max(
                            int(rollout_profile_acc.get("transition_inner_min_bucket", 0) or 0),
                            int(group_profile.get("transition_inner_min_bucket", 0) or 0),
                        )
                    )
                    rollout_profile_acc["transition_bucket_max_batch"] = int(
                        max(
                            int(rollout_profile_acc.get("transition_bucket_max_batch", 0) or 0),
                            int(group_profile.get("transition_bucket_max_batch", 0) or 0),
                        )
                    )
                    rollout_profile_acc["transition_work_actual_est"] += float(
                        group_profile.get("transition_work_actual_est", 0.0) or 0.0
                    )
                    rollout_profile_acc["transition_work_padded_est"] += float(
                        group_profile.get("transition_work_padded_est", 0.0) or 0.0
                    )
                    rollout_profile_acc["transition_async_enabled"] = int(
                        max(
                            int(rollout_profile_acc.get("transition_async_enabled", 0) or 0),
                            int(group_profile.get("transition_async_enabled", 0) or 0),
                        )
                    )
                    group_noise_mode = group_profile.get("noise_mode", None)
                    if group_noise_mode is not None:
                        group_noise_mode = str(group_noise_mode)
                        current_noise_mode = rollout_profile_acc.get("noise_mode", None)
                        if current_noise_mode is None:
                            rollout_profile_acc["noise_mode"] = group_noise_mode
                        elif str(current_noise_mode) != group_noise_mode:
                            rollout_profile_acc["noise_mode"] = "mixed"
                    try:
                        group_noise_block_size = int(group_profile.get("noise_block_size", 0) or 0)
                    except Exception:
                        group_noise_block_size = 0
                    if group_noise_block_size > int(rollout_profile_acc.get("noise_block_size", 0) or 0):
                        rollout_profile_acc["noise_block_size"] = group_noise_block_size
            if rollout_profile_acc is not None:
                rollout_profile_acc["transition_bucket_mean_batch"] = float(
                    batch_size / max(1, int(rollout_profile_acc.get("transition_group_count", 0) or 0))
                )
                rollout_profile_acc["transition_work_fill_ratio"] = float(
                    float(rollout_profile_acc.get("transition_work_actual_est", 0.0) or 0.0)
                    / max(1e-9, float(rollout_profile_acc.get("transition_work_padded_est", 0.0) or 0.0))
                )
            self.last_rollout_profile = rollout_profile_acc
            self.last_rollout_v2 = self._aev2_finalize_rollout_summary(
                rollout_v2_acc,
                device=device,
                dtype=torch.float32,
            )
            self.last_rollout_v3 = self._aev3_finalize_rollout_summary(
                rollout_v3_acc,
                device=device,
                dtype=torch.float32,
            )
            self.last_rollout_v4 = self._aev4_finalize_rollout_summary(
                rollout_v4_acc,
                device=device,
                dtype=torch.float32,
            )
            self.last_runtime_info = infos if collect_runtime_info else [None] * batch_size
            return {
                "x": x,
                "rewards": rewards,
                "info": self.last_runtime_info,
                "single_eval_pos": single_eval_pos,
                "rollout_profile": self.last_rollout_profile,
                "aev2": self.last_rollout_v2,
                "aev3": self.last_rollout_v3,
                "aev4": self.last_rollout_v4,
            }

        # Fallback serial baseline for explicit non-vectorized backend.
        rollout_v2_acc = None
        rollout_v3_acc = None
        rollout_v4_acc = None
        for b, h in enumerate(h_list):
            env = self._sample_environment(
                h=h,
                device=device,
                rng_seed=(env_seeds[b] if env_seeds is not None else None),
            )
            x_one, y_one, info = self._rollout_single(
                env=env,
                n_samples=n_samples,
                num_features=num_features,
                single_eval_pos=single_eval_pos,
                device=device,
                policy_step_fn=policy_step_fn,
                collect_x=collect_x,
                collect_runtime_info=collect_runtime_info,
                rng_seed=(rollout_seeds[b] if rollout_seeds is not None else None),
                tbptt_window=tbptt_window,
                tbptt_reward_sink=tbptt_reward_sink,
                tbptt_reward_sink_supports_aux=tbptt_reward_sink_supports_aux,
                store_rewards=store_rewards,
            )
            if collect_x:
                x[:, b] = x_one
            if store_rewards:
                rewards[:, b] = y_one
            infos[b] = info
            rollout_v2_acc = self._aev2_merge_rollout_summary(
                rollout_v2_acc,
                self.last_rollout_v2,
                batch_weight=(1.0 / float(max(1, batch_size))),
                device=device,
                dtype=torch.float32,
            )
            rollout_v3_acc = self._aev3_merge_rollout_summary(
                rollout_v3_acc,
                self.last_rollout_v3,
                batch_weight=(1.0 / float(max(1, batch_size))),
                device=device,
                dtype=torch.float32,
            )
            rollout_v4_acc = self._aev4_merge_rollout_summary(
                rollout_v4_acc,
                self.last_rollout_v4,
                batch_weight=(1.0 / float(max(1, batch_size))),
                device=device,
                dtype=torch.float32,
            )
        self.last_runtime_info = infos if collect_runtime_info else [None] * batch_size
        self.last_rollout_profile = None
        self.last_rollout_v2 = self._aev2_finalize_rollout_summary(
            rollout_v2_acc,
            device=device,
            dtype=torch.float32,
        )
        self.last_rollout_v3 = self._aev3_finalize_rollout_summary(
            rollout_v3_acc,
            device=device,
            dtype=torch.float32,
        )
        self.last_rollout_v4 = self._aev4_finalize_rollout_summary(
            rollout_v4_acc,
            device=device,
            dtype=torch.float32,
        )
        return {
            "x": x,
            "rewards": rewards,
            "info": self.last_runtime_info,
            "single_eval_pos": single_eval_pos,
            "rollout_profile": self.last_rollout_profile,
            "aev2": self.last_rollout_v2,
            "aev3": self.last_rollout_v3,
            "aev4": self.last_rollout_v4,
        }

    def normalize_rewards(self, rewards, eps=None, clip=None, detach_stats=True, return_stats=False):
        if rewards.ndim != 2:
            raise ValueError(f"rewards must have shape (T, B), got {tuple(rewards.shape)}")
        if eps is None:
            eps = self._resolve_scalar(self.config.get("reward_norm_eps", 1e-6))
        if clip is None:
            clip = self._resolve_scalar(self.config.get("reward_norm_clip", 10.0))

        mean = rewards.mean(dim=0, keepdim=True)
        std = rewards.std(dim=0, unbiased=False, keepdim=True).clamp_min(float(eps))
        if detach_stats:
            mean = mean.detach()
            std = std.detach()
        normalized_preclip = (rewards - mean) / std
        normalized = normalized_preclip
        normalized_clip_hit_share = torch.zeros((), device=rewards.device, dtype=torch.float32)
        if clip is not None and float(clip) > 0:
            clip_f = float(clip)
            normalized_clip_hit_share = (normalized_preclip.abs() > clip_f).to(torch.float32).mean()
            normalized = normalized_preclip.clamp(-clip_f, clip_f)
        if return_stats:
            return normalized, {
                "normalized_clip_hit_share": normalized_clip_hit_share.detach(),
            }
        return normalized

    def policy_gradient_loss_signature(
        self,
        normalize,
        discount,
        detach_stats,
        eps,
        clip,
    ):
        def _fmt_float(x):
            try:
                return f"{float(x):.6g}"
            except Exception:
                return "na"

        reward_clip_cfg = self.config.get("reward_clip", None)
        reward_clip_value = None
        if reward_clip_cfg is not None:
            try:
                reward_clip_value = float(max(0.0, self._resolve_scalar(reward_clip_cfg)))
            except Exception:
                reward_clip_value = None

        # Bump this tag whenever PG loss form changes.
        loss_form = str(self.config.get("policy_gradient_loss_form", "pg_v2"))
        aev2_cfg = self._resolve_aev2_config()
        aev3_cfg = self._resolve_aev3_config()
        aev4_cfg = self._resolve_aev4_config()
        aev5_cfg = self._resolve_aev5_config()
        return (
            f"{loss_form}"
            f"|norm={int(bool(normalize))}"
            f"|disc={_fmt_float(discount)}"
            f"|detach={int(bool(detach_stats))}"
            f"|eps={_fmt_float(eps)}"
            f"|clip={_fmt_float(clip)}"
            f"|rclip={_fmt_float(reward_clip_value)}"
            f"|aev2={int(bool(aev2_cfg.get('enabled', False)))}"
            f"|aev2lam={_fmt_float(aev2_cfg.get('lambda', 0.0))}"
            f"|aev2glo={_fmt_float(aev2_cfg.get('gain_lo', 0.0))}"
            f"|aev2ghi={_fmt_float(aev2_cfg.get('gain_hi', 0.0))}"
            f"|aev3={int(bool(aev3_cfg.get('enabled', False)))}"
            f"|aev3ld={_fmt_float(aev3_cfg.get('lambda_drift', 0.0))}"
            f"|aev3lt={_fmt_float(aev3_cfg.get('lambda_tail', 0.0))}"
            f"|aev3glo={_fmt_float(aev3_cfg.get('gain_lo', 0.0))}"
            f"|aev3ghi={_fmt_float(aev3_cfg.get('gain_hi', 0.0))}"
            f"|aev4={int(bool(aev4_cfg.get('enabled', False)))}"
            f"|aev4ld={_fmt_float(aev4_cfg.get('lambda_drift', 0.0))}"
            f"|aev4lt={_fmt_float(aev4_cfg.get('lambda_tail', 0.0))}"
            f"|aev4glo={_fmt_float(aev4_cfg.get('gain_lo', 0.0))}"
            f"|aev4ghi={_fmt_float(aev4_cfg.get('gain_hi', 0.0))}"
            f"|aev4u={_fmt_float(aev4_cfg.get('update_scale', 0.0))}"
            f"|aev5={int(bool(aev5_cfg.get('enabled', False)))}"
            f"|aev5t={_fmt_float(aev5_cfg.get('target_std', 0.0))}"
            f"|aev5slo={_fmt_float(aev5_cfg.get('scale_lo', 0.0))}"
            f"|aev5shi={_fmt_float(aev5_cfg.get('scale_hi', 0.0))}"
        )

    def policy_gradient_loss_from_rewards(
        self,
        rewards,
        normalize=True,
        discount=None,
        detach_stats=True,
        eps=None,
        clip=None,
        aev5_cfg=None,
    ):
        if rewards.ndim != 2:
            raise ValueError(f"rewards must have shape (T, B), got {tuple(rewards.shape)}")
        if discount is None:
            discount = self._resolve_scalar(self.config.get("discount", 1.0))
        discount = float(max(0.0, min(1.0, discount)))
        if eps is None:
            eps = self._resolve_scalar(self.config.get("reward_norm_eps", 1e-6))
        if clip is None:
            clip = self._resolve_scalar(self.config.get("reward_norm_clip", 10.0))

        weighted = rewards
        if discount < 1.0:
            t = torch.arange(rewards.shape[0], device=rewards.device, dtype=rewards.dtype)
            weights = (discount ** t).unsqueeze(1)
            weighted = rewards * weights

        reward_min = rewards.min().detach()
        reward_max = rewards.max().detach()
        reward_abs_max = rewards.abs().max().detach()
        reward_clip_hit_share = torch.zeros((), device=rewards.device, dtype=torch.float32)
        reward_clip_cfg = self.config.get("reward_clip", None)
        if reward_clip_cfg is not None:
            try:
                reward_clip_bound = float(max(0.0, self._resolve_scalar(reward_clip_cfg)))
            except Exception:
                reward_clip_bound = 0.0
            if reward_clip_bound > 0.0:
                reward_clip_hit_share = (
                    rewards.detach().abs() >= max(0.0, reward_clip_bound - 1e-6)
                ).to(torch.float32).mean()

        norm_clip_hit_share = torch.zeros((), device=rewards.device, dtype=torch.float32)
        if normalize:
            objective_tensor, norm_stats = self.normalize_rewards(
                weighted,
                eps=eps,
                clip=clip,
                detach_stats=detach_stats,
                return_stats=True,
            )
            norm_clip_hit_share = norm_stats["normalized_clip_hit_share"].detach()
        else:
            objective_tensor = weighted
        objective = objective_tensor.mean()
        loss = -objective
        reward_std_for_scale = rewards.std(unbiased=False)

        if aev5_cfg is None:
            aev5_cfg = self._resolve_aev5_config()
        aev5_enabled = bool(aev5_cfg.get("enabled", False))
        aev5_scale_raw = torch.ones((), device=rewards.device, dtype=rewards.dtype)
        aev5_scale = torch.ones((), device=rewards.device, dtype=rewards.dtype)
        if aev5_enabled:
            std_ref = reward_std_for_scale
            if bool(aev5_cfg.get("detach_reference", True)):
                std_ref = std_ref.detach()
            eps_aev5 = float(aev5_cfg.get("eps", 1e-6))
            target_std = float(aev5_cfg.get("target_std", 0.25))
            scale_lo = float(aev5_cfg.get("scale_lo", 0.5))
            scale_hi = float(aev5_cfg.get("scale_hi", 4.0))
            aev5_scale_raw = torch.as_tensor(
                target_std,
                device=std_ref.device,
                dtype=std_ref.dtype,
            ) / (std_ref + float(max(1e-12, eps_aev5)))
            aev5_scale = torch.clamp(aev5_scale_raw, min=scale_lo, max=scale_hi)
            aev5_scale = torch.nan_to_num(
                aev5_scale,
                nan=1.0,
                posinf=float(scale_hi),
                neginf=float(scale_lo),
            )
            loss = loss * aev5_scale

        stats = {
            "objective": objective.detach(),
            "reward_mean": rewards.mean().detach(),
            "reward_std": reward_std_for_scale.detach(),
            "reward_min": reward_min,
            "reward_max": reward_max,
            "reward_abs_max": reward_abs_max,
            "reward_clip_hit_share": reward_clip_hit_share.detach(),
            "reward_norm_clip_hit_share": norm_clip_hit_share.detach(),
        }
        if aev5_enabled:
            stats["aev5_enabled"] = int(aev5_enabled)
            stats["aev5_target_std"] = float(aev5_cfg.get("target_std", 0.25))
            stats["aev5_scale_lo"] = float(aev5_cfg.get("scale_lo", 0.5))
            stats["aev5_scale_hi"] = float(aev5_cfg.get("scale_hi", 4.0))
            stats["aev5_loss_mul"] = aev5_scale.detach().to(dtype=torch.float32)
            stats["aev5_scale"] = aev5_scale.detach().to(dtype=torch.float32)
            stats["aev5_scale_raw"] = aev5_scale_raw.detach().to(dtype=torch.float32)
            stats["aev5_reward_std_ref"] = (
                reward_std_for_scale.detach().to(dtype=torch.float32)
            )
            stats["objective_with_aev5"] = (
                objective.detach() * aev5_scale.detach().to(dtype=objective.detach().dtype)
            ).to(dtype=torch.float32)
        return loss, stats

    def rollout_policy_gradient_loss(
        self,
        policy_step_fn,
        batch_size,
        n_samples,
        num_features,
        device=default_device,
        epoch=None,
        single_eval_pos=None,
        normalize=None,
        discount=None,
        detach_stats=True,
        eps=None,
        clip=None,
        collect_x=False,
        tbptt_window=None,
        tbptt_loss_sink=None,
        h_list_override=None,
        env_seeds_override=None,
        rollout_seeds_override=None,
    ):
        n_samples = int(n_samples)
        batch_size = int(batch_size)
        if normalize is None:
            normalize = bool(self.config.get("policy_gradient_normalize_rewards", False))
        tbptt_window_active = False
        tbptt_window_size = n_samples
        if tbptt_window is not None:
            w = int(tbptt_window)
            if 0 < w < n_samples:
                tbptt_window_active = True
                tbptt_window_size = w
        aev2_cfg = self._resolve_aev2_config()
        aev2_enabled = bool(aev2_cfg.get("enabled", False))
        aev2_lambda = float(aev2_cfg.get("lambda", 0.0))
        aev3_cfg = self._resolve_aev3_config()
        aev3_enabled = bool(aev3_cfg.get("enabled", False))
        aev3_lambda_drift = float(aev3_cfg.get("lambda_drift", 0.0))
        aev3_lambda_tail = float(aev3_cfg.get("lambda_tail", 0.0))
        aev4_cfg = self._resolve_aev4_config()
        aev4_enabled = bool(aev4_cfg.get("enabled", False))
        aev4_lambda_drift = float(aev4_cfg.get("lambda_drift", 0.0))
        aev4_lambda_tail = float(aev4_cfg.get("lambda_tail", 0.0))
        aev5_cfg = self._resolve_aev5_config()
        aev5_enabled = bool(aev5_cfg.get("enabled", False))

        if not tbptt_window_active:
            rollout = self.rollout_with_policy(
                policy_step_fn=policy_step_fn,
                batch_size=batch_size,
                n_samples=n_samples,
                num_features=num_features,
                device=device,
                epoch=epoch,
                single_eval_pos=single_eval_pos,
                collect_x=collect_x,
                collect_runtime_info=False,
                tbptt_reward_sink_supports_aux=False,
                h_list_override=h_list_override,
                env_seeds_override=env_seeds_override,
                rollout_seeds_override=rollout_seeds_override,
            )
            loss, stats = self.policy_gradient_loss_from_rewards(
                rewards=rollout["rewards"],
                normalize=normalize,
                discount=discount,
                detach_stats=detach_stats,
                eps=eps,
                clip=clip,
                aev5_cfg=aev5_cfg,
            )
            if aev2_enabled:
                aev2_rollout = rollout.get("aev2", None)
                aev2_penalty = None
                if isinstance(aev2_rollout, dict):
                    aev2_penalty = aev2_rollout.get("penalty_mean", None)
                if torch.is_tensor(aev2_penalty):
                    if aev2_lambda > 0.0:
                        loss = loss + (aev2_penalty * aev2_lambda)
                    stats["aev2_penalty"] = aev2_penalty.detach()
                    stats["aev2_loss_add"] = (aev2_penalty.detach() * float(aev2_lambda))
                    stats["objective_with_aev2"] = (
                        stats["objective"] - (aev2_penalty.detach() * float(aev2_lambda))
                    )
                if isinstance(aev2_rollout, dict):
                    stats["aev2_gain_mean"] = aev2_rollout.get(
                        "gain_mean",
                        torch.zeros((), device=rollout["rewards"].device, dtype=torch.float32),
                    )
                    stats["aev2_gain_std"] = aev2_rollout.get(
                        "gain_std",
                        torch.zeros((), device=rollout["rewards"].device, dtype=torch.float32),
                    )
                    stats["aev2_gain_min"] = aev2_rollout.get(
                        "gain_min",
                        torch.zeros((), device=rollout["rewards"].device, dtype=torch.float32),
                    )
                    stats["aev2_gain_max"] = aev2_rollout.get(
                        "gain_max",
                        torch.zeros((), device=rollout["rewards"].device, dtype=torch.float32),
                    )
                stats["aev2_enabled"] = int(aev2_enabled)
                stats["aev2_lambda"] = float(aev2_lambda)
                stats["aev2_gain_lo"] = float(aev2_cfg.get("gain_lo", 0.0))
                stats["aev2_gain_hi"] = float(aev2_cfg.get("gain_hi", 0.0))
                if "aev2_penalty" not in stats:
                    zero_t = torch.zeros((), device=rollout["rewards"].device, dtype=torch.float32)
                    stats["aev2_penalty"] = zero_t
                    stats["aev2_loss_add"] = zero_t
                    stats["objective_with_aev2"] = stats["objective"]
                if "aev2_gain_mean" not in stats:
                    zero_t = torch.zeros((), device=rollout["rewards"].device, dtype=torch.float32)
                    stats["aev2_gain_mean"] = zero_t
                    stats["aev2_gain_std"] = zero_t
                    stats["aev2_gain_min"] = zero_t
                    stats["aev2_gain_max"] = zero_t
            if aev3_enabled:
                aev3_rollout = rollout.get("aev3", None)
                aev3_penalty = None
                aev3_penalty_drift = None
                aev3_penalty_tail = None
                if isinstance(aev3_rollout, dict):
                    aev3_penalty = aev3_rollout.get("penalty_mean", None)
                    aev3_penalty_drift = aev3_rollout.get("penalty_drift", None)
                    aev3_penalty_tail = aev3_rollout.get("penalty_tail", None)
                if torch.is_tensor(aev3_penalty):
                    loss = loss + aev3_penalty
                    stats["aev3_penalty"] = aev3_penalty.detach()
                    stats["aev3_loss_add"] = aev3_penalty.detach()
                    stats["objective_with_aev3"] = stats["objective"] - aev3_penalty.detach()
                if torch.is_tensor(aev3_penalty_drift):
                    stats["aev3_penalty_drift"] = aev3_penalty_drift.detach()
                if torch.is_tensor(aev3_penalty_tail):
                    stats["aev3_penalty_tail"] = aev3_penalty_tail.detach()
                if isinstance(aev3_rollout, dict):
                    stats["aev3_log_gain_mean"] = aev3_rollout.get(
                        "log_gain_mean",
                        torch.zeros((), device=rollout["rewards"].device, dtype=torch.float32),
                    )
                    stats["aev3_log_gain_std"] = aev3_rollout.get(
                        "log_gain_std",
                        torch.zeros((), device=rollout["rewards"].device, dtype=torch.float32),
                    )
                    stats["aev3_tail_low_share"] = aev3_rollout.get(
                        "tail_low_share",
                        torch.zeros((), device=rollout["rewards"].device, dtype=torch.float32),
                    )
                    stats["aev3_tail_high_share"] = aev3_rollout.get(
                        "tail_high_share",
                        torch.zeros((), device=rollout["rewards"].device, dtype=torch.float32),
                    )
                    stats["aev3_gain_mean"] = aev3_rollout.get(
                        "gain_mean",
                        torch.zeros((), device=rollout["rewards"].device, dtype=torch.float32),
                    )
                    stats["aev3_gain_std"] = aev3_rollout.get(
                        "gain_std",
                        torch.zeros((), device=rollout["rewards"].device, dtype=torch.float32),
                    )
                    stats["aev3_gain_min"] = aev3_rollout.get(
                        "gain_min",
                        torch.zeros((), device=rollout["rewards"].device, dtype=torch.float32),
                    )
                    stats["aev3_gain_max"] = aev3_rollout.get(
                        "gain_max",
                        torch.zeros((), device=rollout["rewards"].device, dtype=torch.float32),
                    )
                stats["aev3_enabled"] = int(aev3_enabled)
                stats["aev3_lambda_drift"] = float(aev3_lambda_drift)
                stats["aev3_lambda_tail"] = float(aev3_lambda_tail)
                stats["aev3_gain_lo"] = float(aev3_cfg.get("gain_lo", 0.0))
                stats["aev3_gain_hi"] = float(aev3_cfg.get("gain_hi", 0.0))
                if "aev3_penalty" not in stats:
                    zero_t = torch.zeros((), device=rollout["rewards"].device, dtype=torch.float32)
                    stats["aev3_penalty"] = zero_t
                    stats["aev3_loss_add"] = zero_t
                    stats["objective_with_aev3"] = stats["objective"]
                if "aev3_penalty_drift" not in stats:
                    stats["aev3_penalty_drift"] = torch.zeros(
                        (), device=rollout["rewards"].device, dtype=torch.float32
                    )
                if "aev3_penalty_tail" not in stats:
                    stats["aev3_penalty_tail"] = torch.zeros(
                        (), device=rollout["rewards"].device, dtype=torch.float32
                    )
                if "aev3_log_gain_mean" not in stats:
                    zero_t = torch.zeros((), device=rollout["rewards"].device, dtype=torch.float32)
                    stats["aev3_log_gain_mean"] = zero_t
                    stats["aev3_log_gain_std"] = zero_t
                    stats["aev3_tail_low_share"] = zero_t
                    stats["aev3_tail_high_share"] = zero_t
                    stats["aev3_gain_mean"] = zero_t
                    stats["aev3_gain_std"] = zero_t
                    stats["aev3_gain_min"] = zero_t
                    stats["aev3_gain_max"] = zero_t
            if aev4_enabled:
                aev4_rollout = rollout.get("aev4", None)
                aev4_penalty = None
                aev4_penalty_drift = None
                aev4_penalty_tail = None
                if isinstance(aev4_rollout, dict):
                    aev4_penalty = aev4_rollout.get("penalty_mean", None)
                    aev4_penalty_drift = aev4_rollout.get("penalty_drift", None)
                    aev4_penalty_tail = aev4_rollout.get("penalty_tail", None)
                if torch.is_tensor(aev4_penalty):
                    loss = loss + aev4_penalty
                    stats["aev4_penalty"] = aev4_penalty.detach()
                    stats["aev4_loss_add"] = aev4_penalty.detach()
                    stats["objective_with_aev4"] = stats["objective"] - aev4_penalty.detach()
                if torch.is_tensor(aev4_penalty_drift):
                    stats["aev4_penalty_drift"] = aev4_penalty_drift.detach()
                if torch.is_tensor(aev4_penalty_tail):
                    stats["aev4_penalty_tail"] = aev4_penalty_tail.detach()
                if isinstance(aev4_rollout, dict):
                    stats["aev4_log_gain_mean"] = aev4_rollout.get(
                        "log_gain_mean",
                        torch.zeros((), device=rollout["rewards"].device, dtype=torch.float32),
                    )
                    stats["aev4_log_gain_std"] = aev4_rollout.get(
                        "log_gain_std",
                        torch.zeros((), device=rollout["rewards"].device, dtype=torch.float32),
                    )
                    stats["aev4_tail_low_share"] = aev4_rollout.get(
                        "tail_low_share",
                        torch.zeros((), device=rollout["rewards"].device, dtype=torch.float32),
                    )
                    stats["aev4_tail_high_share"] = aev4_rollout.get(
                        "tail_high_share",
                        torch.zeros((), device=rollout["rewards"].device, dtype=torch.float32),
                    )
                    stats["aev4_gain_mean"] = aev4_rollout.get(
                        "gain_mean",
                        torch.zeros((), device=rollout["rewards"].device, dtype=torch.float32),
                    )
                    stats["aev4_gain_std"] = aev4_rollout.get(
                        "gain_std",
                        torch.zeros((), device=rollout["rewards"].device, dtype=torch.float32),
                    )
                    stats["aev4_gain_min"] = aev4_rollout.get(
                        "gain_min",
                        torch.zeros((), device=rollout["rewards"].device, dtype=torch.float32),
                    )
                    stats["aev4_gain_max"] = aev4_rollout.get(
                        "gain_max",
                        torch.zeros((), device=rollout["rewards"].device, dtype=torch.float32),
                    )
                    stats["aev4_update_rms_mean"] = aev4_rollout.get(
                        "update_rms_mean",
                        torch.zeros((), device=rollout["rewards"].device, dtype=torch.float32),
                    )
                    stats["aev4_update_rms_std"] = aev4_rollout.get(
                        "update_rms_std",
                        torch.zeros((), device=rollout["rewards"].device, dtype=torch.float32),
                    )
                    stats["aev4_clip_hit_share"] = aev4_rollout.get(
                        "clip_hit_share",
                        torch.zeros((), device=rollout["rewards"].device, dtype=torch.float32),
                    )
                stats["aev4_enabled"] = int(aev4_enabled)
                stats["aev4_lambda_drift"] = float(aev4_lambda_drift)
                stats["aev4_lambda_tail"] = float(aev4_lambda_tail)
                stats["aev4_gain_lo"] = float(aev4_cfg.get("gain_lo", 0.0))
                stats["aev4_gain_hi"] = float(aev4_cfg.get("gain_hi", 0.0))
                stats["aev4_highway_ratio"] = float(aev4_cfg.get("highway_ratio", 0.25))
                stats["aev4_update_scale"] = float(aev4_cfg.get("update_scale", 0.12))
                stats["aev4_update_clip"] = float(aev4_cfg.get("update_clip", 0.0))
                if "aev4_penalty" not in stats:
                    zero_t = torch.zeros((), device=rollout["rewards"].device, dtype=torch.float32)
                    stats["aev4_penalty"] = zero_t
                    stats["aev4_loss_add"] = zero_t
                    stats["objective_with_aev4"] = stats["objective"]
                if "aev4_penalty_drift" not in stats:
                    stats["aev4_penalty_drift"] = torch.zeros(
                        (), device=rollout["rewards"].device, dtype=torch.float32
                    )
                if "aev4_penalty_tail" not in stats:
                    stats["aev4_penalty_tail"] = torch.zeros(
                        (), device=rollout["rewards"].device, dtype=torch.float32
                    )
                if "aev4_log_gain_mean" not in stats:
                    zero_t = torch.zeros((), device=rollout["rewards"].device, dtype=torch.float32)
                    stats["aev4_log_gain_mean"] = zero_t
                    stats["aev4_log_gain_std"] = zero_t
                    stats["aev4_tail_low_share"] = zero_t
                    stats["aev4_tail_high_share"] = zero_t
                    stats["aev4_gain_mean"] = zero_t
                    stats["aev4_gain_std"] = zero_t
                    stats["aev4_gain_min"] = zero_t
                    stats["aev4_gain_max"] = zero_t
                    stats["aev4_update_rms_mean"] = zero_t
                    stats["aev4_update_rms_std"] = zero_t
                    stats["aev4_clip_hit_share"] = zero_t
            if aev5_enabled:
                zero_t = torch.zeros((), device=rollout["rewards"].device, dtype=torch.float32)
                stats["aev5_enabled"] = int(aev5_enabled)
                stats["aev5_target_std"] = float(aev5_cfg.get("target_std", 0.25))
                stats["aev5_scale_lo"] = float(aev5_cfg.get("scale_lo", 0.5))
                stats["aev5_scale_hi"] = float(aev5_cfg.get("scale_hi", 4.0))
                if "aev5_loss_mul" not in stats:
                    stats["aev5_loss_mul"] = torch.ones((), device=rollout["rewards"].device, dtype=torch.float32)
                if "aev5_scale" not in stats:
                    stats["aev5_scale"] = stats["aev5_loss_mul"]
                if "aev5_scale_raw" not in stats:
                    stats["aev5_scale_raw"] = stats["aev5_scale"]
                if "aev5_reward_std_ref" not in stats:
                    stats["aev5_reward_std_ref"] = stats.get("reward_std", zero_t)
                if "objective_with_aev5" not in stats:
                    stats["objective_with_aev5"] = stats["objective"] * stats["aev5_loss_mul"]
            rollout_profile = rollout.get("rollout_profile", None)
            if isinstance(rollout_profile, dict):
                stats["rollout_policy_cuda_ms"] = float(rollout_profile.get("policy_cuda_ms", 0.0))
                stats["rollout_transition_cuda_ms"] = float(rollout_profile.get("transition_cuda_ms", 0.0))
                stats["rollout_policy_wall_ms"] = float(rollout_profile.get("policy_wall_ms", 0.0))
                stats["rollout_transition_wall_ms"] = float(rollout_profile.get("transition_wall_ms", 0.0))
                stats["rollout_transition_y_wall_ms"] = float(rollout_profile.get("transition_y_wall_ms", 0.0))
                stats["rollout_transition_x_wall_ms"] = float(rollout_profile.get("transition_x_wall_ms", 0.0))
                stats["rollout_transition_group_wall_ms"] = float(
                    rollout_profile.get("transition_group_wall_ms", 0.0)
                )
                stats["rollout_transition_group_launch_wall_ms"] = float(
                    rollout_profile.get("transition_group_launch_wall_ms", 0.0)
                )
                stats["rollout_transition_group_sync_wall_ms"] = float(
                    rollout_profile.get("transition_group_sync_wall_ms", 0.0)
                )
                stats["rollout_transition_env_pack_wall_ms"] = float(
                    rollout_profile.get("transition_env_pack_wall_ms", 0.0)
                )
                stats["rollout_transition_state_update_wall_ms"] = float(
                    rollout_profile.get("transition_state_update_wall_ms", 0.0)
                )
                stats["rollout_transition_noise_wall_ms"] = float(
                    rollout_profile.get("transition_noise_wall_ms", 0.0)
                )
                stats["rollout_transition_fused_wall_ms"] = float(
                    rollout_profile.get("transition_fused_wall_ms", 0.0)
                )
                stats["rollout_transition_fused_launch_wall_ms"] = float(
                    rollout_profile.get("transition_fused_launch_wall_ms", 0.0)
                )
                stats["rollout_transition_gp_first_projection_wall_ms"] = float(
                    rollout_profile.get("transition_gp_first_projection_wall_ms", 0.0)
                )
                stats["rollout_transition_gp_second_projection_wall_ms"] = float(
                    rollout_profile.get("transition_gp_second_projection_wall_ms", 0.0)
                )
                stats["rollout_transition_gp_projection_call_count"] = int(
                    rollout_profile.get("transition_gp_projection_call_count", 0) or 0
                )
                stats["rollout_transition_gp_rff_fused_call_count"] = int(
                    rollout_profile.get("transition_gp_rff_fused_call_count", 0) or 0
                )
                stats["rollout_transition_gp_profile_group_count"] = int(
                    rollout_profile.get("transition_gp_profile_group_count", 0) or 0
                )
                stats["rollout_transition_gp_profile_sync_group_count"] = int(
                    rollout_profile.get("transition_gp_profile_sync_group_count", 0) or 0
                )
                stats["rollout_transition_gp_shared_total_wall_ms"] = float(
                    rollout_profile.get("transition_gp_shared_total_wall_ms", 0.0) or 0.0
                )
                stats["rollout_transition_gp_shared_core_wall_ms"] = float(
                    rollout_profile.get("transition_gp_shared_core_wall_ms", 0.0) or 0.0
                )
                stats["rollout_transition_gp_shared_noise_wall_ms"] = float(
                    rollout_profile.get("transition_gp_shared_noise_wall_ms", 0.0) or 0.0
                )
                stats["rollout_transition_gp_shared_checkpoint_wall_ms"] = float(
                    rollout_profile.get("transition_gp_shared_checkpoint_wall_ms", 0.0) or 0.0
                )
                stats["rollout_transition_gp_shared_post_wall_ms"] = float(
                    rollout_profile.get("transition_gp_shared_post_wall_ms", 0.0) or 0.0
                )
                stats["rollout_transition_gp_shared_call_count"] = int(
                    rollout_profile.get("transition_gp_shared_call_count", 0) or 0
                )
                stats["rollout_transition_packed_env_input_group_count"] = int(
                    rollout_profile.get("transition_packed_env_input_group_count", 0) or 0
                )
                stats["rollout_transition_packed_env_input_call_count"] = int(
                    rollout_profile.get("transition_packed_env_input_call_count", 0) or 0
                )
                stats["rollout_transition_only_build_group_count"] = int(
                    rollout_profile.get("transition_only_build_group_count", 0) or 0
                )
                stats["rollout_transition_only_skipped_generator_count"] = int(
                    rollout_profile.get("transition_only_skipped_generator_count", 0) or 0
                )
                stats["rollout_transition_setup_wall_ms"] = float(
                    rollout_profile.get("transition_setup_wall_ms", 0.0) or 0.0
                )
                stats["rollout_transition_family_build_wall_ms"] = float(
                    rollout_profile.get("transition_family_build_wall_ms", 0.0) or 0.0
                )
                stats["rollout_transition_generator_build_wall_ms"] = float(
                    rollout_profile.get("transition_generator_build_wall_ms", 0.0) or 0.0
                )
                stats["rollout_transition_gp_shared_build_wall_ms"] = float(
                    rollout_profile.get("transition_gp_shared_build_wall_ms", 0.0) or 0.0
                )
                stats["rollout_transition_fused_call_count"] = int(
                    rollout_profile.get("transition_fused_call_count", 0) or 0
                )
                stats["rollout_transition_fused_group_count"] = int(
                    rollout_profile.get("transition_fused_group_count", 0) or 0
                )
                stats["rollout_transition_fused_enabled"] = int(
                    rollout_profile.get("transition_fused_enabled", 0) or 0
                )
                stats["rollout_transition_checkpoint_enabled"] = int(
                    rollout_profile.get("transition_checkpoint_enabled", 0) or 0
                )
                stats["rollout_transition_checkpoint_call_count"] = int(
                    rollout_profile.get("transition_checkpoint_call_count", 0) or 0
                )
                stats["rollout_transition_group_count"] = int(rollout_profile.get("transition_group_count", 0))
                stats["rollout_transition_family_group_count"] = int(
                    rollout_profile.get("transition_family_group_count", 0) or 0
                )
                stats["rollout_transition_inner_grouping_structure_enabled"] = int(
                    rollout_profile.get("transition_inner_grouping_structure_enabled", 0) or 0
                )
                stats["rollout_transition_inner_min_bucket"] = int(
                    rollout_profile.get("transition_inner_min_bucket", 0) or 0
                )
                stats["rollout_transition_bucket_max_batch"] = int(
                    rollout_profile.get("transition_bucket_max_batch", 0) or 0
                )
                stats["rollout_transition_bucket_mean_batch"] = float(
                    rollout_profile.get("transition_bucket_mean_batch", 0.0) or 0.0
                )
                stats["rollout_transition_work_actual_est"] = float(
                    rollout_profile.get("transition_work_actual_est", 0.0) or 0.0
                )
                stats["rollout_transition_work_padded_est"] = float(
                    rollout_profile.get("transition_work_padded_est", 0.0) or 0.0
                )
                stats["rollout_transition_work_fill_ratio"] = float(
                    rollout_profile.get("transition_work_fill_ratio", 0.0) or 0.0
                )
                stats["rollout_transition_async_enabled"] = int(
                    rollout_profile.get("transition_async_enabled", 0) or 0
                )
                stats["rollout_transition_async_commit_in_stream"] = int(
                    rollout_profile.get("transition_async_commit_in_stream", 0) or 0
                )
                stats["rollout_noise_mode"] = rollout_profile.get("noise_mode", None)
                stats["rollout_noise_block_size"] = int(rollout_profile.get("noise_block_size", 0) or 0)
            return loss, rollout, stats

        reward_sum = None
        reward_sumsq = None
        reward_count = 0
        reward_min_accum = None
        reward_max_accum = None
        reward_absmax_accum = None
        reward_clip_hit_accum = None
        reward_norm_clip_hit_accum = None
        weighted_losses = []
        objective_accum = None
        total_weight = 0.0
        aev2_penalty_accum = None
        aev2_gain_sum_accum = None
        aev2_gain_sumsq_accum = None
        aev2_gain_count_accum = 0
        aev2_gain_min_accum = None
        aev2_gain_max_accum = None
        aev3_penalty_accum = None
        aev3_penalty_drift_accum = None
        aev3_penalty_tail_accum = None
        aev3_log_gain_sum_accum = None
        aev3_log_gain_sumsq_accum = None
        aev3_log_gain_count_accum = 0
        aev3_tail_low_count_accum = 0
        aev3_tail_high_count_accum = 0
        aev3_gain_sum_accum = None
        aev3_gain_sumsq_accum = None
        aev3_gain_count_accum = 0
        aev3_gain_min_accum = None
        aev3_gain_max_accum = None
        aev4_penalty_accum = None
        aev4_penalty_drift_accum = None
        aev4_penalty_tail_accum = None
        aev4_log_gain_sum_accum = None
        aev4_log_gain_sumsq_accum = None
        aev4_log_gain_count_accum = 0
        aev4_tail_low_count_accum = 0
        aev4_tail_high_count_accum = 0
        aev4_gain_sum_accum = None
        aev4_gain_sumsq_accum = None
        aev4_gain_count_accum = 0
        aev4_gain_min_accum = None
        aev4_gain_max_accum = None
        aev4_update_rms_sum_accum = None
        aev4_update_rms_sumsq_accum = None
        aev4_update_rms_count_accum = 0
        aev4_clip_hit_sum_accum = None
        aev4_clip_hit_count_accum = 0
        aev5_scale_accum = None
        aev5_scale_raw_accum = None
        aev5_reward_std_ref_accum = None
        aev5_objective_scaled_accum = None
        n_samples_f = float(max(1, n_samples))
        batch_size_f = float(max(1, batch_size))

        def _tbptt_reward_sink(reward_payload):
            nonlocal reward_sum, reward_sumsq, reward_count
            nonlocal objective_accum, total_weight
            nonlocal reward_min_accum, reward_max_accum, reward_absmax_accum
            nonlocal reward_clip_hit_accum, reward_norm_clip_hit_accum
            nonlocal aev2_penalty_accum, aev2_gain_sum_accum, aev2_gain_sumsq_accum
            nonlocal aev2_gain_count_accum, aev2_gain_min_accum, aev2_gain_max_accum
            nonlocal aev3_penalty_accum, aev3_penalty_drift_accum, aev3_penalty_tail_accum
            nonlocal aev3_log_gain_sum_accum, aev3_log_gain_sumsq_accum, aev3_log_gain_count_accum
            nonlocal aev3_tail_low_count_accum, aev3_tail_high_count_accum
            nonlocal aev3_gain_sum_accum, aev3_gain_sumsq_accum, aev3_gain_count_accum
            nonlocal aev3_gain_min_accum, aev3_gain_max_accum
            nonlocal aev4_penalty_accum, aev4_penalty_drift_accum, aev4_penalty_tail_accum
            nonlocal aev4_log_gain_sum_accum, aev4_log_gain_sumsq_accum, aev4_log_gain_count_accum
            nonlocal aev4_tail_low_count_accum, aev4_tail_high_count_accum
            nonlocal aev4_gain_sum_accum, aev4_gain_sumsq_accum, aev4_gain_count_accum
            nonlocal aev4_gain_min_accum, aev4_gain_max_accum
            nonlocal aev4_update_rms_sum_accum, aev4_update_rms_sumsq_accum, aev4_update_rms_count_accum
            nonlocal aev4_clip_hit_sum_accum, aev4_clip_hit_count_accum
            nonlocal aev5_scale_accum, aev5_scale_raw_accum, aev5_reward_std_ref_accum, aev5_objective_scaled_accum
            aev2_window = None
            aev3_window = None
            aev4_window = None
            rewards_window = reward_payload
            if isinstance(reward_payload, tuple) and len(reward_payload) == 2:
                rewards_window, aux = reward_payload
                if isinstance(aux, dict) and (("aev2" in aux) or ("aev3" in aux) or ("aev4" in aux)):
                    aev2_window = aux.get("aev2", None)
                    aev3_window = aux.get("aev3", None)
                    aev4_window = aux.get("aev4", None)
                else:
                    if aev2_enabled:
                        aev2_window = aux
                    elif aev3_enabled:
                        aev3_window = aux
                    elif aev4_enabled:
                        aev4_window = aux
            loss_window, stats_window = self.policy_gradient_loss_from_rewards(
                rewards=rewards_window,
                normalize=normalize,
                discount=discount,
                detach_stats=detach_stats,
                eps=eps,
                clip=clip,
                aev5_cfg=aev5_cfg,
            )
            if aev2_enabled and isinstance(aev2_window, dict):
                aev2_penalty_window = aev2_window.get("penalty_mean", None)
                if torch.is_tensor(aev2_penalty_window):
                    if aev2_lambda > 0.0:
                        loss_window = loss_window + (aev2_penalty_window * aev2_lambda)
            if aev3_enabled and isinstance(aev3_window, dict):
                aev3_penalty_window = aev3_window.get("penalty_mean", None)
                if torch.is_tensor(aev3_penalty_window):
                    loss_window = loss_window + aev3_penalty_window
            if aev4_enabled and isinstance(aev4_window, dict):
                aev4_penalty_window = aev4_window.get("penalty_mean", None)
                if torch.is_tensor(aev4_penalty_window):
                    loss_window = loss_window + aev4_penalty_window
            time_weight = float(rewards_window.shape[0]) / n_samples_f
            batch_weight = float(rewards_window.shape[1]) / batch_size_f
            window_weight = time_weight * batch_weight
            weighted_loss = loss_window * window_weight
            if tbptt_loss_sink is None:
                weighted_losses.append(weighted_loss)
            else:
                tbptt_loss_sink(weighted_loss)

            objective_term = stats_window["objective"] * window_weight
            if objective_accum is None:
                objective_accum = objective_term.detach()
            else:
                objective_accum = objective_accum + objective_term.detach()
            total_weight += window_weight
            if aev5_enabled:
                aev5_scale_w = stats_window.get("aev5_scale", None)
                if torch.is_tensor(aev5_scale_w):
                    aev5_scale_term = aev5_scale_w.detach() * float(window_weight)
                    aev5_scale_accum = (
                        aev5_scale_term if aev5_scale_accum is None else (aev5_scale_accum + aev5_scale_term)
                    )
                aev5_scale_raw_w = stats_window.get("aev5_scale_raw", None)
                if torch.is_tensor(aev5_scale_raw_w):
                    aev5_scale_raw_term = aev5_scale_raw_w.detach() * float(window_weight)
                    aev5_scale_raw_accum = (
                        aev5_scale_raw_term
                        if aev5_scale_raw_accum is None
                        else (aev5_scale_raw_accum + aev5_scale_raw_term)
                    )
                aev5_std_ref_w = stats_window.get("aev5_reward_std_ref", None)
                if torch.is_tensor(aev5_std_ref_w):
                    aev5_std_ref_term = aev5_std_ref_w.detach() * float(window_weight)
                    aev5_reward_std_ref_accum = (
                        aev5_std_ref_term
                        if aev5_reward_std_ref_accum is None
                        else (aev5_reward_std_ref_accum + aev5_std_ref_term)
                    )
                aev5_obj_scaled_w = stats_window.get("objective_with_aev5", None)
                if torch.is_tensor(aev5_obj_scaled_w):
                    aev5_obj_scaled_term = aev5_obj_scaled_w.detach() * float(window_weight)
                    aev5_objective_scaled_accum = (
                        aev5_obj_scaled_term
                        if aev5_objective_scaled_accum is None
                        else (aev5_objective_scaled_accum + aev5_obj_scaled_term)
                    )
            if aev2_enabled and isinstance(aev2_window, dict):
                aev2_penalty_window = aev2_window.get("penalty_mean", None)
                if torch.is_tensor(aev2_penalty_window):
                    penalty_term = aev2_penalty_window.detach() * float(window_weight)
                    aev2_penalty_accum = (
                        penalty_term if aev2_penalty_accum is None else (aev2_penalty_accum + penalty_term)
                    )
                gain_sum_w = aev2_window.get("gain_sum", None)
                if torch.is_tensor(gain_sum_w):
                    gain_sum_w = gain_sum_w.detach().to(device=rewards_window.device, dtype=torch.float64)
                    aev2_gain_sum_accum = (
                        gain_sum_w if aev2_gain_sum_accum is None else (aev2_gain_sum_accum + gain_sum_w)
                    )
                gain_sumsq_w = aev2_window.get("gain_sumsq", None)
                if torch.is_tensor(gain_sumsq_w):
                    gain_sumsq_w = gain_sumsq_w.detach().to(device=rewards_window.device, dtype=torch.float64)
                    aev2_gain_sumsq_accum = (
                        gain_sumsq_w
                        if aev2_gain_sumsq_accum is None
                        else (aev2_gain_sumsq_accum + gain_sumsq_w)
                    )
                aev2_gain_count_accum += int(aev2_window.get("gain_count", 0))
                gain_min_w = aev2_window.get("gain_min", None)
                if torch.is_tensor(gain_min_w):
                    gain_min_w = gain_min_w.detach()
                    aev2_gain_min_accum = (
                        gain_min_w if aev2_gain_min_accum is None else torch.minimum(aev2_gain_min_accum, gain_min_w)
                    )
                gain_max_w = aev2_window.get("gain_max", None)
                if torch.is_tensor(gain_max_w):
                    gain_max_w = gain_max_w.detach()
                    aev2_gain_max_accum = (
                        gain_max_w if aev2_gain_max_accum is None else torch.maximum(aev2_gain_max_accum, gain_max_w)
                    )
            if aev3_enabled and isinstance(aev3_window, dict):
                aev3_penalty_window = aev3_window.get("penalty_mean", None)
                if torch.is_tensor(aev3_penalty_window):
                    penalty_term = aev3_penalty_window.detach() * float(window_weight)
                    aev3_penalty_accum = (
                        penalty_term if aev3_penalty_accum is None else (aev3_penalty_accum + penalty_term)
                    )
                aev3_penalty_drift_window = aev3_window.get("penalty_drift", None)
                if torch.is_tensor(aev3_penalty_drift_window):
                    penalty_drift_term = aev3_penalty_drift_window.detach() * float(window_weight)
                    aev3_penalty_drift_accum = (
                        penalty_drift_term
                        if aev3_penalty_drift_accum is None
                        else (aev3_penalty_drift_accum + penalty_drift_term)
                    )
                aev3_penalty_tail_window = aev3_window.get("penalty_tail", None)
                if torch.is_tensor(aev3_penalty_tail_window):
                    penalty_tail_term = aev3_penalty_tail_window.detach() * float(window_weight)
                    aev3_penalty_tail_accum = (
                        penalty_tail_term
                        if aev3_penalty_tail_accum is None
                        else (aev3_penalty_tail_accum + penalty_tail_term)
                    )
                log_gain_sum_w = aev3_window.get("log_gain_sum", None)
                if torch.is_tensor(log_gain_sum_w):
                    log_gain_sum_w = log_gain_sum_w.detach().to(device=rewards_window.device, dtype=torch.float64)
                    aev3_log_gain_sum_accum = (
                        log_gain_sum_w
                        if aev3_log_gain_sum_accum is None
                        else (aev3_log_gain_sum_accum + log_gain_sum_w)
                    )
                log_gain_sumsq_w = aev3_window.get("log_gain_sumsq", None)
                if torch.is_tensor(log_gain_sumsq_w):
                    log_gain_sumsq_w = log_gain_sumsq_w.detach().to(device=rewards_window.device, dtype=torch.float64)
                    aev3_log_gain_sumsq_accum = (
                        log_gain_sumsq_w
                        if aev3_log_gain_sumsq_accum is None
                        else (aev3_log_gain_sumsq_accum + log_gain_sumsq_w)
                    )
                aev3_log_gain_count_accum += int(aev3_window.get("log_gain_count", 0))
                aev3_tail_low_count_accum += int(aev3_window.get("tail_low_count", 0))
                aev3_tail_high_count_accum += int(aev3_window.get("tail_high_count", 0))
                gain_sum_w = aev3_window.get("gain_sum", None)
                if torch.is_tensor(gain_sum_w):
                    gain_sum_w = gain_sum_w.detach().to(device=rewards_window.device, dtype=torch.float64)
                    aev3_gain_sum_accum = (
                        gain_sum_w if aev3_gain_sum_accum is None else (aev3_gain_sum_accum + gain_sum_w)
                    )
                gain_sumsq_w = aev3_window.get("gain_sumsq", None)
                if torch.is_tensor(gain_sumsq_w):
                    gain_sumsq_w = gain_sumsq_w.detach().to(device=rewards_window.device, dtype=torch.float64)
                    aev3_gain_sumsq_accum = (
                        gain_sumsq_w
                        if aev3_gain_sumsq_accum is None
                        else (aev3_gain_sumsq_accum + gain_sumsq_w)
                    )
                aev3_gain_count_accum += int(aev3_window.get("gain_count", 0))
                gain_min_w = aev3_window.get("gain_min", None)
                if torch.is_tensor(gain_min_w):
                    gain_min_w = gain_min_w.detach()
                    aev3_gain_min_accum = (
                        gain_min_w if aev3_gain_min_accum is None else torch.minimum(aev3_gain_min_accum, gain_min_w)
                    )
                gain_max_w = aev3_window.get("gain_max", None)
                if torch.is_tensor(gain_max_w):
                    gain_max_w = gain_max_w.detach()
                    aev3_gain_max_accum = (
                        gain_max_w if aev3_gain_max_accum is None else torch.maximum(aev3_gain_max_accum, gain_max_w)
                    )
            if aev4_enabled and isinstance(aev4_window, dict):
                aev4_penalty_window = aev4_window.get("penalty_mean", None)
                if torch.is_tensor(aev4_penalty_window):
                    penalty_term = aev4_penalty_window.detach() * float(window_weight)
                    aev4_penalty_accum = (
                        penalty_term if aev4_penalty_accum is None else (aev4_penalty_accum + penalty_term)
                    )
                aev4_penalty_drift_window = aev4_window.get("penalty_drift", None)
                if torch.is_tensor(aev4_penalty_drift_window):
                    penalty_drift_term = aev4_penalty_drift_window.detach() * float(window_weight)
                    aev4_penalty_drift_accum = (
                        penalty_drift_term
                        if aev4_penalty_drift_accum is None
                        else (aev4_penalty_drift_accum + penalty_drift_term)
                    )
                aev4_penalty_tail_window = aev4_window.get("penalty_tail", None)
                if torch.is_tensor(aev4_penalty_tail_window):
                    penalty_tail_term = aev4_penalty_tail_window.detach() * float(window_weight)
                    aev4_penalty_tail_accum = (
                        penalty_tail_term
                        if aev4_penalty_tail_accum is None
                        else (aev4_penalty_tail_accum + penalty_tail_term)
                    )
                log_gain_sum_w = aev4_window.get("log_gain_sum", None)
                if torch.is_tensor(log_gain_sum_w):
                    log_gain_sum_w = log_gain_sum_w.detach().to(device=rewards_window.device, dtype=torch.float64)
                    aev4_log_gain_sum_accum = (
                        log_gain_sum_w
                        if aev4_log_gain_sum_accum is None
                        else (aev4_log_gain_sum_accum + log_gain_sum_w)
                    )
                log_gain_sumsq_w = aev4_window.get("log_gain_sumsq", None)
                if torch.is_tensor(log_gain_sumsq_w):
                    log_gain_sumsq_w = log_gain_sumsq_w.detach().to(device=rewards_window.device, dtype=torch.float64)
                    aev4_log_gain_sumsq_accum = (
                        log_gain_sumsq_w
                        if aev4_log_gain_sumsq_accum is None
                        else (aev4_log_gain_sumsq_accum + log_gain_sumsq_w)
                    )
                aev4_log_gain_count_accum += int(aev4_window.get("log_gain_count", 0))
                aev4_tail_low_count_accum += int(aev4_window.get("tail_low_count", 0))
                aev4_tail_high_count_accum += int(aev4_window.get("tail_high_count", 0))
                gain_sum_w = aev4_window.get("gain_sum", None)
                if torch.is_tensor(gain_sum_w):
                    gain_sum_w = gain_sum_w.detach().to(device=rewards_window.device, dtype=torch.float64)
                    aev4_gain_sum_accum = (
                        gain_sum_w if aev4_gain_sum_accum is None else (aev4_gain_sum_accum + gain_sum_w)
                    )
                gain_sumsq_w = aev4_window.get("gain_sumsq", None)
                if torch.is_tensor(gain_sumsq_w):
                    gain_sumsq_w = gain_sumsq_w.detach().to(device=rewards_window.device, dtype=torch.float64)
                    aev4_gain_sumsq_accum = (
                        gain_sumsq_w
                        if aev4_gain_sumsq_accum is None
                        else (aev4_gain_sumsq_accum + gain_sumsq_w)
                    )
                aev4_gain_count_accum += int(aev4_window.get("gain_count", 0))
                gain_min_w = aev4_window.get("gain_min", None)
                if torch.is_tensor(gain_min_w):
                    gain_min_w = gain_min_w.detach()
                    aev4_gain_min_accum = (
                        gain_min_w if aev4_gain_min_accum is None else torch.minimum(aev4_gain_min_accum, gain_min_w)
                    )
                gain_max_w = aev4_window.get("gain_max", None)
                if torch.is_tensor(gain_max_w):
                    gain_max_w = gain_max_w.detach()
                    aev4_gain_max_accum = (
                        gain_max_w if aev4_gain_max_accum is None else torch.maximum(aev4_gain_max_accum, gain_max_w)
                    )
                update_rms_sum_w = aev4_window.get("update_rms_sum", None)
                if torch.is_tensor(update_rms_sum_w):
                    update_rms_sum_w = update_rms_sum_w.detach().to(device=rewards_window.device, dtype=torch.float64)
                    aev4_update_rms_sum_accum = (
                        update_rms_sum_w
                        if aev4_update_rms_sum_accum is None
                        else (aev4_update_rms_sum_accum + update_rms_sum_w)
                    )
                update_rms_sumsq_w = aev4_window.get("update_rms_sumsq", None)
                if torch.is_tensor(update_rms_sumsq_w):
                    update_rms_sumsq_w = update_rms_sumsq_w.detach().to(device=rewards_window.device, dtype=torch.float64)
                    aev4_update_rms_sumsq_accum = (
                        update_rms_sumsq_w
                        if aev4_update_rms_sumsq_accum is None
                        else (aev4_update_rms_sumsq_accum + update_rms_sumsq_w)
                    )
                aev4_update_rms_count_accum += int(aev4_window.get("update_rms_count", 0))
                clip_hit_sum_w = aev4_window.get("clip_hit_sum", None)
                if torch.is_tensor(clip_hit_sum_w):
                    clip_hit_sum_w = clip_hit_sum_w.detach().to(device=rewards_window.device, dtype=torch.float64)
                    aev4_clip_hit_sum_accum = (
                        clip_hit_sum_w
                        if aev4_clip_hit_sum_accum is None
                        else (aev4_clip_hit_sum_accum + clip_hit_sum_w)
                    )
                aev4_clip_hit_count_accum += int(aev4_window.get("clip_hit_count", 0))
            reward_min_w = stats_window.get("reward_min", None)
            if reward_min_w is not None:
                reward_min_w = reward_min_w.detach()
                reward_min_accum = (
                    reward_min_w
                    if reward_min_accum is None
                    else torch.minimum(reward_min_accum, reward_min_w)
                )
            reward_max_w = stats_window.get("reward_max", None)
            if reward_max_w is not None:
                reward_max_w = reward_max_w.detach()
                reward_max_accum = (
                    reward_max_w
                    if reward_max_accum is None
                    else torch.maximum(reward_max_accum, reward_max_w)
                )
            reward_absmax_w = stats_window.get("reward_abs_max", None)
            if reward_absmax_w is not None:
                reward_absmax_w = reward_absmax_w.detach()
                reward_absmax_accum = (
                    reward_absmax_w
                    if reward_absmax_accum is None
                    else torch.maximum(reward_absmax_accum, reward_absmax_w)
                )
            reward_clip_hit_w = stats_window.get("reward_clip_hit_share", None)
            if reward_clip_hit_w is not None:
                reward_clip_hit_w = reward_clip_hit_w.detach() * float(window_weight)
                reward_clip_hit_accum = (
                    reward_clip_hit_w
                    if reward_clip_hit_accum is None
                    else reward_clip_hit_accum + reward_clip_hit_w
                )
            reward_norm_clip_hit_w = stats_window.get("reward_norm_clip_hit_share", None)
            if reward_norm_clip_hit_w is not None:
                reward_norm_clip_hit_w = reward_norm_clip_hit_w.detach() * float(window_weight)
                reward_norm_clip_hit_accum = (
                    reward_norm_clip_hit_w
                    if reward_norm_clip_hit_accum is None
                    else reward_norm_clip_hit_accum + reward_norm_clip_hit_w
                )

            rewards_det = rewards_window.detach().to(dtype=torch.float64)
            if reward_sum is None:
                reward_sum = rewards_det.sum()
                reward_sumsq = (rewards_det * rewards_det).sum()
            else:
                reward_sum = reward_sum + rewards_det.sum()
                reward_sumsq = reward_sumsq + (rewards_det * rewards_det).sum()
            reward_count += int(rewards_det.numel())

        rollout = self.rollout_with_policy(
            # Streaming-TBPTT consumes rewards window-by-window via sink; full
            # horizon reward tensor is optional and disabled by default.
            # Enable for diagnostics via TICL_POLICY_TBPTT_STORE_REWARDS=1.
            store_rewards=(
                str(os.environ.get("TICL_POLICY_TBPTT_STORE_REWARDS", "0")).strip().lower()
                in {"1", "true", "yes", "on"}
            ),
            policy_step_fn=policy_step_fn,
            batch_size=batch_size,
            n_samples=n_samples,
            num_features=num_features,
            device=device,
            epoch=epoch,
            single_eval_pos=single_eval_pos,
            collect_x=collect_x,
            collect_runtime_info=False,
            tbptt_window=tbptt_window_size,
            tbptt_reward_sink=_tbptt_reward_sink,
            tbptt_reward_sink_supports_aux=bool(aev2_enabled or aev3_enabled or aev4_enabled),
            h_list_override=h_list_override,
            env_seeds_override=env_seeds_override,
            rollout_seeds_override=rollout_seeds_override,
        )

        if tbptt_loss_sink is None:
            if not weighted_losses:
                raise RuntimeError("TBPTT rollout produced no window losses.")
            loss = torch.stack(weighted_losses).sum()
        else:
            loss = torch.zeros((), device=rollout["rewards"].device, dtype=torch.float32)

        if objective_accum is None:
            objective_accum = torch.zeros((), device=rollout["rewards"].device, dtype=torch.float32)
        if total_weight > 0.0:
            objective = (objective_accum / total_weight).detach()
        else:
            objective = objective_accum.detach()

        if reward_count <= 0 or reward_sum is None or reward_sumsq is None:
            reward_mean = torch.zeros((), device=rollout["rewards"].device, dtype=torch.float32)
            reward_std = torch.zeros((), device=rollout["rewards"].device, dtype=torch.float32)
        else:
            reward_mean64 = reward_sum / float(reward_count)
            reward_var64 = (reward_sumsq / float(reward_count)) - (reward_mean64 * reward_mean64)
            reward_std64 = torch.sqrt(torch.clamp(reward_var64, min=0.0))
            reward_mean = reward_mean64.to(dtype=torch.float32).detach()
            reward_std = reward_std64.to(dtype=torch.float32).detach()
        reward_min = (
            reward_min_accum
            if reward_min_accum is not None
            else torch.zeros((), device=rollout["rewards"].device, dtype=torch.float32)
        )
        reward_max = (
            reward_max_accum
            if reward_max_accum is not None
            else torch.zeros((), device=rollout["rewards"].device, dtype=torch.float32)
        )
        reward_absmax = (
            reward_absmax_accum
            if reward_absmax_accum is not None
            else torch.zeros((), device=rollout["rewards"].device, dtype=torch.float32)
        )
        reward_clip_hit = (
            reward_clip_hit_accum
            if reward_clip_hit_accum is not None
            else torch.zeros((), device=rollout["rewards"].device, dtype=torch.float32)
        )
        reward_norm_clip_hit = (
            reward_norm_clip_hit_accum
            if reward_norm_clip_hit_accum is not None
            else torch.zeros((), device=rollout["rewards"].device, dtype=torch.float32)
        )
        if total_weight > 0.0:
            reward_clip_hit = reward_clip_hit / float(total_weight)
            reward_norm_clip_hit = reward_norm_clip_hit / float(total_weight)

        aev2_penalty_mean = None
        aev2_gain_mean = None
        aev2_gain_std = None
        aev2_gain_min = None
        aev2_gain_max = None
        if aev2_enabled:
            if (aev2_penalty_accum is not None) and total_weight > 0.0:
                aev2_penalty_mean = (aev2_penalty_accum / float(total_weight)).detach()
            elif isinstance(rollout.get("aev2", None), dict):
                aev2_penalty_rollout = rollout["aev2"].get("penalty_mean", None)
                if torch.is_tensor(aev2_penalty_rollout):
                    aev2_penalty_mean = aev2_penalty_rollout.detach()
            if aev2_gain_count_accum > 0 and (aev2_gain_sum_accum is not None) and (aev2_gain_sumsq_accum is not None):
                gain_mean64 = aev2_gain_sum_accum / float(aev2_gain_count_accum)
                gain_var64 = (aev2_gain_sumsq_accum / float(aev2_gain_count_accum)) - (gain_mean64 * gain_mean64)
                gain_std64 = torch.sqrt(torch.clamp(gain_var64, min=0.0))
                aev2_gain_mean = gain_mean64.to(dtype=torch.float32).detach()
                aev2_gain_std = gain_std64.to(dtype=torch.float32).detach()
            elif isinstance(rollout.get("aev2", None), dict):
                aev2_gain_mean = rollout["aev2"].get("gain_mean", None)
                aev2_gain_std = rollout["aev2"].get("gain_std", None)
            if aev2_gain_min_accum is not None:
                aev2_gain_min = aev2_gain_min_accum.to(dtype=torch.float32).detach()
            elif isinstance(rollout.get("aev2", None), dict):
                aev2_gain_min = rollout["aev2"].get("gain_min", None)
            if aev2_gain_max_accum is not None:
                aev2_gain_max = aev2_gain_max_accum.to(dtype=torch.float32).detach()
            elif isinstance(rollout.get("aev2", None), dict):
                aev2_gain_max = rollout["aev2"].get("gain_max", None)
        aev3_penalty_mean = None
        aev3_penalty_drift_mean = None
        aev3_penalty_tail_mean = None
        aev3_log_gain_mean = None
        aev3_log_gain_std = None
        aev3_tail_low_share = None
        aev3_tail_high_share = None
        aev3_gain_mean = None
        aev3_gain_std = None
        aev3_gain_min = None
        aev3_gain_max = None
        if aev3_enabled:
            if (aev3_penalty_accum is not None) and total_weight > 0.0:
                aev3_penalty_mean = (aev3_penalty_accum / float(total_weight)).detach()
            elif isinstance(rollout.get("aev3", None), dict):
                aev3_penalty_rollout = rollout["aev3"].get("penalty_mean", None)
                if torch.is_tensor(aev3_penalty_rollout):
                    aev3_penalty_mean = aev3_penalty_rollout.detach()
            if (aev3_penalty_drift_accum is not None) and total_weight > 0.0:
                aev3_penalty_drift_mean = (aev3_penalty_drift_accum / float(total_weight)).detach()
            elif isinstance(rollout.get("aev3", None), dict):
                aev3_penalty_drift_rollout = rollout["aev3"].get("penalty_drift", None)
                if torch.is_tensor(aev3_penalty_drift_rollout):
                    aev3_penalty_drift_mean = aev3_penalty_drift_rollout.detach()
            if (aev3_penalty_tail_accum is not None) and total_weight > 0.0:
                aev3_penalty_tail_mean = (aev3_penalty_tail_accum / float(total_weight)).detach()
            elif isinstance(rollout.get("aev3", None), dict):
                aev3_penalty_tail_rollout = rollout["aev3"].get("penalty_tail", None)
                if torch.is_tensor(aev3_penalty_tail_rollout):
                    aev3_penalty_tail_mean = aev3_penalty_tail_rollout.detach()

            if (
                aev3_log_gain_count_accum > 0
                and (aev3_log_gain_sum_accum is not None)
                and (aev3_log_gain_sumsq_accum is not None)
            ):
                log_gain_mean64 = aev3_log_gain_sum_accum / float(aev3_log_gain_count_accum)
                log_gain_var64 = (
                    (aev3_log_gain_sumsq_accum / float(aev3_log_gain_count_accum)) - (log_gain_mean64 * log_gain_mean64)
                )
                log_gain_std64 = torch.sqrt(torch.clamp(log_gain_var64, min=0.0))
                aev3_log_gain_mean = log_gain_mean64.to(dtype=torch.float32).detach()
                aev3_log_gain_std = log_gain_std64.to(dtype=torch.float32).detach()
                aev3_tail_low_share = torch.as_tensor(
                    float(aev3_tail_low_count_accum) / float(aev3_log_gain_count_accum),
                    device=rollout["rewards"].device,
                    dtype=torch.float32,
                )
                aev3_tail_high_share = torch.as_tensor(
                    float(aev3_tail_high_count_accum) / float(aev3_log_gain_count_accum),
                    device=rollout["rewards"].device,
                    dtype=torch.float32,
                )
            elif isinstance(rollout.get("aev3", None), dict):
                aev3_log_gain_mean = rollout["aev3"].get("log_gain_mean", None)
                aev3_log_gain_std = rollout["aev3"].get("log_gain_std", None)
                aev3_tail_low_share = rollout["aev3"].get("tail_low_share", None)
                aev3_tail_high_share = rollout["aev3"].get("tail_high_share", None)

            if (
                aev3_gain_count_accum > 0
                and (aev3_gain_sum_accum is not None)
                and (aev3_gain_sumsq_accum is not None)
            ):
                gain_mean64 = aev3_gain_sum_accum / float(aev3_gain_count_accum)
                gain_var64 = (aev3_gain_sumsq_accum / float(aev3_gain_count_accum)) - (gain_mean64 * gain_mean64)
                gain_std64 = torch.sqrt(torch.clamp(gain_var64, min=0.0))
                aev3_gain_mean = gain_mean64.to(dtype=torch.float32).detach()
                aev3_gain_std = gain_std64.to(dtype=torch.float32).detach()
            elif isinstance(rollout.get("aev3", None), dict):
                aev3_gain_mean = rollout["aev3"].get("gain_mean", None)
                aev3_gain_std = rollout["aev3"].get("gain_std", None)
            if aev3_gain_min_accum is not None:
                aev3_gain_min = aev3_gain_min_accum.to(dtype=torch.float32).detach()
            elif isinstance(rollout.get("aev3", None), dict):
                aev3_gain_min = rollout["aev3"].get("gain_min", None)
            if aev3_gain_max_accum is not None:
                aev3_gain_max = aev3_gain_max_accum.to(dtype=torch.float32).detach()
            elif isinstance(rollout.get("aev3", None), dict):
                aev3_gain_max = rollout["aev3"].get("gain_max", None)

        aev4_penalty_mean = None
        aev4_penalty_drift_mean = None
        aev4_penalty_tail_mean = None
        aev4_log_gain_mean = None
        aev4_log_gain_std = None
        aev4_tail_low_share = None
        aev4_tail_high_share = None
        aev4_gain_mean = None
        aev4_gain_std = None
        aev4_gain_min = None
        aev4_gain_max = None
        aev4_update_rms_mean = None
        aev4_update_rms_std = None
        aev4_clip_hit_share = None
        if aev4_enabled:
            if (aev4_penalty_accum is not None) and total_weight > 0.0:
                aev4_penalty_mean = (aev4_penalty_accum / float(total_weight)).detach()
            elif isinstance(rollout.get("aev4", None), dict):
                aev4_penalty_rollout = rollout["aev4"].get("penalty_mean", None)
                if torch.is_tensor(aev4_penalty_rollout):
                    aev4_penalty_mean = aev4_penalty_rollout.detach()
            if (aev4_penalty_drift_accum is not None) and total_weight > 0.0:
                aev4_penalty_drift_mean = (aev4_penalty_drift_accum / float(total_weight)).detach()
            elif isinstance(rollout.get("aev4", None), dict):
                aev4_penalty_drift_rollout = rollout["aev4"].get("penalty_drift", None)
                if torch.is_tensor(aev4_penalty_drift_rollout):
                    aev4_penalty_drift_mean = aev4_penalty_drift_rollout.detach()
            if (aev4_penalty_tail_accum is not None) and total_weight > 0.0:
                aev4_penalty_tail_mean = (aev4_penalty_tail_accum / float(total_weight)).detach()
            elif isinstance(rollout.get("aev4", None), dict):
                aev4_penalty_tail_rollout = rollout["aev4"].get("penalty_tail", None)
                if torch.is_tensor(aev4_penalty_tail_rollout):
                    aev4_penalty_tail_mean = aev4_penalty_tail_rollout.detach()

            if (
                aev4_log_gain_count_accum > 0
                and (aev4_log_gain_sum_accum is not None)
                and (aev4_log_gain_sumsq_accum is not None)
            ):
                log_gain_mean64 = aev4_log_gain_sum_accum / float(aev4_log_gain_count_accum)
                log_gain_var64 = (
                    (aev4_log_gain_sumsq_accum / float(aev4_log_gain_count_accum)) - (log_gain_mean64 * log_gain_mean64)
                )
                log_gain_std64 = torch.sqrt(torch.clamp(log_gain_var64, min=0.0))
                aev4_log_gain_mean = log_gain_mean64.to(dtype=torch.float32).detach()
                aev4_log_gain_std = log_gain_std64.to(dtype=torch.float32).detach()
                aev4_tail_low_share = torch.as_tensor(
                    float(aev4_tail_low_count_accum) / float(aev4_log_gain_count_accum),
                    device=rollout["rewards"].device,
                    dtype=torch.float32,
                )
                aev4_tail_high_share = torch.as_tensor(
                    float(aev4_tail_high_count_accum) / float(aev4_log_gain_count_accum),
                    device=rollout["rewards"].device,
                    dtype=torch.float32,
                )
            elif isinstance(rollout.get("aev4", None), dict):
                aev4_log_gain_mean = rollout["aev4"].get("log_gain_mean", None)
                aev4_log_gain_std = rollout["aev4"].get("log_gain_std", None)
                aev4_tail_low_share = rollout["aev4"].get("tail_low_share", None)
                aev4_tail_high_share = rollout["aev4"].get("tail_high_share", None)

            if (
                aev4_gain_count_accum > 0
                and (aev4_gain_sum_accum is not None)
                and (aev4_gain_sumsq_accum is not None)
            ):
                gain_mean64 = aev4_gain_sum_accum / float(aev4_gain_count_accum)
                gain_var64 = (aev4_gain_sumsq_accum / float(aev4_gain_count_accum)) - (gain_mean64 * gain_mean64)
                gain_std64 = torch.sqrt(torch.clamp(gain_var64, min=0.0))
                aev4_gain_mean = gain_mean64.to(dtype=torch.float32).detach()
                aev4_gain_std = gain_std64.to(dtype=torch.float32).detach()
            elif isinstance(rollout.get("aev4", None), dict):
                aev4_gain_mean = rollout["aev4"].get("gain_mean", None)
                aev4_gain_std = rollout["aev4"].get("gain_std", None)
            if aev4_gain_min_accum is not None:
                aev4_gain_min = aev4_gain_min_accum.to(dtype=torch.float32).detach()
            elif isinstance(rollout.get("aev4", None), dict):
                aev4_gain_min = rollout["aev4"].get("gain_min", None)
            if aev4_gain_max_accum is not None:
                aev4_gain_max = aev4_gain_max_accum.to(dtype=torch.float32).detach()
            elif isinstance(rollout.get("aev4", None), dict):
                aev4_gain_max = rollout["aev4"].get("gain_max", None)

            if (
                aev4_update_rms_count_accum > 0
                and (aev4_update_rms_sum_accum is not None)
                and (aev4_update_rms_sumsq_accum is not None)
            ):
                update_rms_mean64 = aev4_update_rms_sum_accum / float(aev4_update_rms_count_accum)
                update_rms_var64 = (
                    (aev4_update_rms_sumsq_accum / float(aev4_update_rms_count_accum))
                    - (update_rms_mean64 * update_rms_mean64)
                )
                update_rms_std64 = torch.sqrt(torch.clamp(update_rms_var64, min=0.0))
                aev4_update_rms_mean = update_rms_mean64.to(dtype=torch.float32).detach()
                aev4_update_rms_std = update_rms_std64.to(dtype=torch.float32).detach()
            elif isinstance(rollout.get("aev4", None), dict):
                aev4_update_rms_mean = rollout["aev4"].get("update_rms_mean", None)
                aev4_update_rms_std = rollout["aev4"].get("update_rms_std", None)

            if (aev4_clip_hit_sum_accum is not None) and (aev4_clip_hit_count_accum > 0):
                aev4_clip_hit_share = (
                    aev4_clip_hit_sum_accum / float(max(1, aev4_clip_hit_count_accum))
                ).to(dtype=torch.float32).detach()
            elif isinstance(rollout.get("aev4", None), dict):
                aev4_clip_hit_share = rollout["aev4"].get("clip_hit_share", None)

        stats = {
            "objective": objective,
            "reward_mean": reward_mean,
            "reward_std": reward_std,
            "reward_min": reward_min,
            "reward_max": reward_max,
            "reward_abs_max": reward_absmax,
            "reward_clip_hit_share": reward_clip_hit,
            "reward_norm_clip_hit_share": reward_norm_clip_hit,
        }
        if aev2_enabled:
            zero_t = torch.zeros((), device=rollout["rewards"].device, dtype=torch.float32)
            if aev2_penalty_mean is None:
                aev2_penalty_mean = zero_t
            if aev2_gain_mean is None:
                aev2_gain_mean = zero_t
            if aev2_gain_std is None:
                aev2_gain_std = zero_t
            if aev2_gain_min is None:
                aev2_gain_min = zero_t
            if aev2_gain_max is None:
                aev2_gain_max = zero_t
            stats["aev2_enabled"] = int(aev2_enabled)
            stats["aev2_lambda"] = float(aev2_lambda)
            stats["aev2_gain_lo"] = float(aev2_cfg.get("gain_lo", 0.0))
            stats["aev2_gain_hi"] = float(aev2_cfg.get("gain_hi", 0.0))
            stats["aev2_penalty"] = aev2_penalty_mean
            stats["aev2_loss_add"] = aev2_penalty_mean * float(aev2_lambda)
            stats["objective_with_aev2"] = objective - (aev2_penalty_mean * float(aev2_lambda))
            stats["aev2_gain_mean"] = aev2_gain_mean
            stats["aev2_gain_std"] = aev2_gain_std
            stats["aev2_gain_min"] = aev2_gain_min
            stats["aev2_gain_max"] = aev2_gain_max
        if aev3_enabled:
            zero_t = torch.zeros((), device=rollout["rewards"].device, dtype=torch.float32)
            if aev3_penalty_mean is None:
                aev3_penalty_mean = zero_t
            if aev3_penalty_drift_mean is None:
                aev3_penalty_drift_mean = zero_t
            if aev3_penalty_tail_mean is None:
                aev3_penalty_tail_mean = zero_t
            if aev3_log_gain_mean is None:
                aev3_log_gain_mean = zero_t
            if aev3_log_gain_std is None:
                aev3_log_gain_std = zero_t
            if aev3_tail_low_share is None:
                aev3_tail_low_share = zero_t
            if aev3_tail_high_share is None:
                aev3_tail_high_share = zero_t
            if aev3_gain_mean is None:
                aev3_gain_mean = zero_t
            if aev3_gain_std is None:
                aev3_gain_std = zero_t
            if aev3_gain_min is None:
                aev3_gain_min = zero_t
            if aev3_gain_max is None:
                aev3_gain_max = zero_t
            stats["aev3_enabled"] = int(aev3_enabled)
            stats["aev3_lambda_drift"] = float(aev3_lambda_drift)
            stats["aev3_lambda_tail"] = float(aev3_lambda_tail)
            stats["aev3_gain_lo"] = float(aev3_cfg.get("gain_lo", 0.0))
            stats["aev3_gain_hi"] = float(aev3_cfg.get("gain_hi", 0.0))
            stats["aev3_penalty"] = aev3_penalty_mean
            stats["aev3_penalty_drift"] = aev3_penalty_drift_mean
            stats["aev3_penalty_tail"] = aev3_penalty_tail_mean
            stats["aev3_loss_add"] = aev3_penalty_mean
            stats["objective_with_aev3"] = objective - aev3_penalty_mean
            stats["aev3_log_gain_mean"] = aev3_log_gain_mean
            stats["aev3_log_gain_std"] = aev3_log_gain_std
            stats["aev3_tail_low_share"] = aev3_tail_low_share
            stats["aev3_tail_high_share"] = aev3_tail_high_share
            stats["aev3_gain_mean"] = aev3_gain_mean
            stats["aev3_gain_std"] = aev3_gain_std
            stats["aev3_gain_min"] = aev3_gain_min
            stats["aev3_gain_max"] = aev3_gain_max
        if aev4_enabled:
            zero_t = torch.zeros((), device=rollout["rewards"].device, dtype=torch.float32)
            if aev4_penalty_mean is None:
                aev4_penalty_mean = zero_t
            if aev4_penalty_drift_mean is None:
                aev4_penalty_drift_mean = zero_t
            if aev4_penalty_tail_mean is None:
                aev4_penalty_tail_mean = zero_t
            if aev4_log_gain_mean is None:
                aev4_log_gain_mean = zero_t
            if aev4_log_gain_std is None:
                aev4_log_gain_std = zero_t
            if aev4_tail_low_share is None:
                aev4_tail_low_share = zero_t
            if aev4_tail_high_share is None:
                aev4_tail_high_share = zero_t
            if aev4_gain_mean is None:
                aev4_gain_mean = zero_t
            if aev4_gain_std is None:
                aev4_gain_std = zero_t
            if aev4_gain_min is None:
                aev4_gain_min = zero_t
            if aev4_gain_max is None:
                aev4_gain_max = zero_t
            if aev4_update_rms_mean is None:
                aev4_update_rms_mean = zero_t
            if aev4_update_rms_std is None:
                aev4_update_rms_std = zero_t
            if aev4_clip_hit_share is None:
                aev4_clip_hit_share = zero_t
            stats["aev4_enabled"] = int(aev4_enabled)
            stats["aev4_lambda_drift"] = float(aev4_lambda_drift)
            stats["aev4_lambda_tail"] = float(aev4_lambda_tail)
            stats["aev4_gain_lo"] = float(aev4_cfg.get("gain_lo", 0.0))
            stats["aev4_gain_hi"] = float(aev4_cfg.get("gain_hi", 0.0))
            stats["aev4_highway_ratio"] = float(aev4_cfg.get("highway_ratio", 0.25))
            stats["aev4_update_scale"] = float(aev4_cfg.get("update_scale", 0.12))
            stats["aev4_update_clip"] = float(aev4_cfg.get("update_clip", 0.0))
            stats["aev4_penalty"] = aev4_penalty_mean
            stats["aev4_penalty_drift"] = aev4_penalty_drift_mean
            stats["aev4_penalty_tail"] = aev4_penalty_tail_mean
            stats["aev4_loss_add"] = aev4_penalty_mean
            stats["objective_with_aev4"] = objective - aev4_penalty_mean
            stats["aev4_log_gain_mean"] = aev4_log_gain_mean
            stats["aev4_log_gain_std"] = aev4_log_gain_std
            stats["aev4_tail_low_share"] = aev4_tail_low_share
            stats["aev4_tail_high_share"] = aev4_tail_high_share
            stats["aev4_gain_mean"] = aev4_gain_mean
            stats["aev4_gain_std"] = aev4_gain_std
            stats["aev4_gain_min"] = aev4_gain_min
            stats["aev4_gain_max"] = aev4_gain_max
            stats["aev4_update_rms_mean"] = aev4_update_rms_mean
            stats["aev4_update_rms_std"] = aev4_update_rms_std
            stats["aev4_clip_hit_share"] = aev4_clip_hit_share
        if aev5_enabled:
            zero_t = torch.zeros((), device=rollout["rewards"].device, dtype=torch.float32)
            if total_weight > 0.0:
                aev5_scale_mean = (
                    aev5_scale_accum / float(total_weight)
                    if aev5_scale_accum is not None
                    else torch.ones((), device=rollout["rewards"].device, dtype=torch.float32)
                )
                aev5_scale_raw_mean = (
                    aev5_scale_raw_accum / float(total_weight)
                    if aev5_scale_raw_accum is not None
                    else aev5_scale_mean
                )
                aev5_reward_std_ref_mean = (
                    aev5_reward_std_ref_accum / float(total_weight)
                    if aev5_reward_std_ref_accum is not None
                    else reward_std
                )
                aev5_objective_scaled_mean = (
                    aev5_objective_scaled_accum / float(total_weight)
                    if aev5_objective_scaled_accum is not None
                    else (objective * aev5_scale_mean)
                )
            else:
                aev5_scale_mean = torch.ones((), device=rollout["rewards"].device, dtype=torch.float32)
                aev5_scale_raw_mean = aev5_scale_mean
                aev5_reward_std_ref_mean = reward_std
                aev5_objective_scaled_mean = objective
            stats["aev5_enabled"] = int(aev5_enabled)
            stats["aev5_target_std"] = float(aev5_cfg.get("target_std", 0.25))
            stats["aev5_scale_lo"] = float(aev5_cfg.get("scale_lo", 0.5))
            stats["aev5_scale_hi"] = float(aev5_cfg.get("scale_hi", 4.0))
            stats["aev5_loss_mul"] = aev5_scale_mean if aev5_scale_mean is not None else torch.ones_like(zero_t)
            stats["aev5_scale"] = stats["aev5_loss_mul"]
            stats["aev5_scale_raw"] = aev5_scale_raw_mean if aev5_scale_raw_mean is not None else stats["aev5_scale"]
            stats["aev5_reward_std_ref"] = (
                aev5_reward_std_ref_mean if aev5_reward_std_ref_mean is not None else reward_std
            )
            stats["objective_with_aev5"] = (
                aev5_objective_scaled_mean if aev5_objective_scaled_mean is not None else objective
            )
        rollout_profile = rollout.get("rollout_profile", None)
        if isinstance(rollout_profile, dict):
            stats["rollout_policy_cuda_ms"] = float(rollout_profile.get("policy_cuda_ms", 0.0))
            stats["rollout_transition_cuda_ms"] = float(rollout_profile.get("transition_cuda_ms", 0.0))
            stats["rollout_policy_wall_ms"] = float(rollout_profile.get("policy_wall_ms", 0.0))
            stats["rollout_transition_wall_ms"] = float(rollout_profile.get("transition_wall_ms", 0.0))
            stats["rollout_transition_y_wall_ms"] = float(rollout_profile.get("transition_y_wall_ms", 0.0))
            stats["rollout_transition_x_wall_ms"] = float(rollout_profile.get("transition_x_wall_ms", 0.0))
            stats["rollout_transition_group_wall_ms"] = float(
                rollout_profile.get("transition_group_wall_ms", 0.0)
            )
            stats["rollout_transition_group_launch_wall_ms"] = float(
                rollout_profile.get("transition_group_launch_wall_ms", 0.0)
            )
            stats["rollout_transition_group_sync_wall_ms"] = float(
                rollout_profile.get("transition_group_sync_wall_ms", 0.0)
            )
            stats["rollout_transition_env_pack_wall_ms"] = float(
                rollout_profile.get("transition_env_pack_wall_ms", 0.0)
            )
            stats["rollout_transition_state_update_wall_ms"] = float(
                rollout_profile.get("transition_state_update_wall_ms", 0.0)
            )
            stats["rollout_transition_noise_wall_ms"] = float(
                rollout_profile.get("transition_noise_wall_ms", 0.0)
            )
            stats["rollout_transition_fused_wall_ms"] = float(
                rollout_profile.get("transition_fused_wall_ms", 0.0)
            )
            stats["rollout_transition_fused_launch_wall_ms"] = float(
                rollout_profile.get("transition_fused_launch_wall_ms", 0.0)
            )
            stats["rollout_transition_gp_first_projection_wall_ms"] = float(
                rollout_profile.get("transition_gp_first_projection_wall_ms", 0.0)
            )
            stats["rollout_transition_gp_second_projection_wall_ms"] = float(
                rollout_profile.get("transition_gp_second_projection_wall_ms", 0.0)
            )
            stats["rollout_transition_gp_projection_call_count"] = int(
                rollout_profile.get("transition_gp_projection_call_count", 0) or 0
            )
            stats["rollout_transition_gp_rff_fused_call_count"] = int(
                rollout_profile.get("transition_gp_rff_fused_call_count", 0) or 0
            )
            stats["rollout_transition_gp_profile_group_count"] = int(
                rollout_profile.get("transition_gp_profile_group_count", 0) or 0
            )
            stats["rollout_transition_gp_profile_sync_group_count"] = int(
                rollout_profile.get("transition_gp_profile_sync_group_count", 0) or 0
            )
            stats["rollout_transition_gp_shared_total_wall_ms"] = float(
                rollout_profile.get("transition_gp_shared_total_wall_ms", 0.0) or 0.0
            )
            stats["rollout_transition_gp_shared_core_wall_ms"] = float(
                rollout_profile.get("transition_gp_shared_core_wall_ms", 0.0) or 0.0
            )
            stats["rollout_transition_gp_shared_noise_wall_ms"] = float(
                rollout_profile.get("transition_gp_shared_noise_wall_ms", 0.0) or 0.0
            )
            stats["rollout_transition_gp_shared_checkpoint_wall_ms"] = float(
                rollout_profile.get("transition_gp_shared_checkpoint_wall_ms", 0.0) or 0.0
            )
            stats["rollout_transition_gp_shared_post_wall_ms"] = float(
                rollout_profile.get("transition_gp_shared_post_wall_ms", 0.0) or 0.0
            )
            stats["rollout_transition_gp_shared_call_count"] = int(
                rollout_profile.get("transition_gp_shared_call_count", 0) or 0
            )
            stats["rollout_transition_packed_env_input_group_count"] = int(
                rollout_profile.get("transition_packed_env_input_group_count", 0) or 0
            )
            stats["rollout_transition_packed_env_input_call_count"] = int(
                rollout_profile.get("transition_packed_env_input_call_count", 0) or 0
            )
            stats["rollout_transition_only_build_group_count"] = int(
                rollout_profile.get("transition_only_build_group_count", 0) or 0
            )
            stats["rollout_transition_only_skipped_generator_count"] = int(
                rollout_profile.get("transition_only_skipped_generator_count", 0) or 0
            )
            stats["rollout_transition_setup_wall_ms"] = float(
                rollout_profile.get("transition_setup_wall_ms", 0.0) or 0.0
            )
            stats["rollout_transition_family_build_wall_ms"] = float(
                rollout_profile.get("transition_family_build_wall_ms", 0.0) or 0.0
            )
            stats["rollout_transition_generator_build_wall_ms"] = float(
                rollout_profile.get("transition_generator_build_wall_ms", 0.0) or 0.0
            )
            stats["rollout_transition_gp_shared_build_wall_ms"] = float(
                rollout_profile.get("transition_gp_shared_build_wall_ms", 0.0) or 0.0
            )
            stats["rollout_transition_fused_call_count"] = int(
                rollout_profile.get("transition_fused_call_count", 0) or 0
            )
            stats["rollout_transition_fused_group_count"] = int(
                rollout_profile.get("transition_fused_group_count", 0) or 0
            )
            stats["rollout_transition_fused_enabled"] = int(
                rollout_profile.get("transition_fused_enabled", 0) or 0
            )
            stats["rollout_transition_checkpoint_enabled"] = int(
                rollout_profile.get("transition_checkpoint_enabled", 0) or 0
            )
            stats["rollout_transition_checkpoint_call_count"] = int(
                rollout_profile.get("transition_checkpoint_call_count", 0) or 0
            )
            stats["rollout_transition_group_count"] = int(rollout_profile.get("transition_group_count", 0))
            stats["rollout_transition_family_group_count"] = int(
                rollout_profile.get("transition_family_group_count", 0) or 0
            )
            stats["rollout_transition_inner_grouping_structure_enabled"] = int(
                rollout_profile.get("transition_inner_grouping_structure_enabled", 0) or 0
            )
            stats["rollout_transition_inner_min_bucket"] = int(
                rollout_profile.get("transition_inner_min_bucket", 0) or 0
            )
            stats["rollout_transition_bucket_max_batch"] = int(
                rollout_profile.get("transition_bucket_max_batch", 0) or 0
            )
            stats["rollout_transition_bucket_mean_batch"] = float(
                rollout_profile.get("transition_bucket_mean_batch", 0.0) or 0.0
            )
            stats["rollout_transition_work_actual_est"] = float(
                rollout_profile.get("transition_work_actual_est", 0.0) or 0.0
            )
            stats["rollout_transition_work_padded_est"] = float(
                rollout_profile.get("transition_work_padded_est", 0.0) or 0.0
            )
            stats["rollout_transition_work_fill_ratio"] = float(
                rollout_profile.get("transition_work_fill_ratio", 0.0) or 0.0
            )
            stats["rollout_transition_async_enabled"] = int(
                rollout_profile.get("transition_async_enabled", 0) or 0
            )
            stats["rollout_transition_async_commit_in_stream"] = int(
                rollout_profile.get("transition_async_commit_in_stream", 0) or 0
            )
            stats["rollout_noise_mode"] = rollout_profile.get("noise_mode", None)
            stats["rollout_noise_block_size"] = int(rollout_profile.get("noise_block_size", 0) or 0)
        return loss, rollout, stats

    def get_last_coverage(self):
        if not self.last_runtime_info:
            return {}
        return {
            "reward_min": min(x["reward_min"] for x in self.last_runtime_info),
            "reward_max": max(x["reward_max"] for x in self.last_runtime_info),
            "reward_std_mean": float(np.mean([x["reward_std"] for x in self.last_runtime_info])),
            "state_abs_max": max(x["state_abs_max"] for x in self.last_runtime_info),
        }
