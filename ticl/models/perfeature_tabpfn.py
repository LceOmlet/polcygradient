import math
import os
import time
from contextlib import contextmanager, nullcontext
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ticl.models.layer import TransformerEncoderLayer
from ticl.utils import SeqBN, get_init_method


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


def _tensor_storage_key(tensor):
    if not torch.is_tensor(tensor):
        return None
    try:
        storage = tensor.untyped_storage()
    except Exception:
        try:
            storage = tensor.storage()
        except Exception:
            return None
    try:
        data_ptr = int(storage.data_ptr())
    except Exception:
        return None
    if data_ptr == 0:
        return None
    device = getattr(tensor, "device", None)
    device_type = getattr(device, "type", "unknown")
    device_index = getattr(device, "index", None)
    return (str(device_type), int(-1 if device_index is None else device_index), data_ptr)


def _tensor_storage_nbytes(tensor):
    if not torch.is_tensor(tensor):
        return 0
    try:
        return int(tensor.untyped_storage().nbytes())
    except Exception:
        try:
            return int(tensor.storage().nbytes())
        except Exception:
            return int(tensor.numel()) * int(tensor.element_size())


def _tensor_nbytes(tensor):
    if not torch.is_tensor(tensor):
        return 0
    try:
        return int(tensor.numel()) * int(tensor.element_size())
    except Exception:
        return int(max(0, _tensor_storage_nbytes(tensor)))


def _tensor_tree_storage_stats(tree):
    stats = {
        "tensor_count": 0,
        "raw_bytes": 0,
        "unique_storage_bytes": 0,
    }
    seen_storage = set()

    def _visit(node):
        if node is None:
            return
        if torch.is_tensor(node):
            nbytes = int(max(0, _tensor_nbytes(node)))
            stats["tensor_count"] += 1
            stats["raw_bytes"] += nbytes
            storage_key = _tensor_storage_key(node)
            storage_nbytes = int(max(0, _tensor_storage_nbytes(node)))
            if storage_key is None or storage_key not in seen_storage:
                if storage_key is not None:
                    seen_storage.add(storage_key)
                stats["unique_storage_bytes"] += storage_nbytes
            return
        if isinstance(node, dict):
            for value in node.values():
                _visit(value)
            return
        if isinstance(node, (list, tuple)):
            for value in node:
                _visit(value)

    _visit(tree)
    return stats


def _module_storage_keys(module):
    keys = set()
    try:
        for tensor in list(module.parameters()) + list(module.buffers()):
            key = _tensor_storage_key(tensor)
            if key is not None:
                keys.add(key)
    except Exception:
        return set()
    return keys


@contextmanager
def _saved_tensors_profile_scope(*, module_storage_keys=None, enabled=False):
    if not bool(enabled):
        yield None
        return
    graph_mod = getattr(torch.autograd, "graph", None)
    if graph_mod is None or not hasattr(graph_mod, "saved_tensors_hooks"):
        yield None
        return
    stats = {
        "tensor_count": 0,
        "total_bytes": 0,
        "max_tensor_bytes": 0,
        "unique_storage_bytes": 0,
        "nonparam_raw_bytes": 0,
        "nonparam_unique_storage_bytes": 0,
        "param_ref_unique_storage_bytes": 0,
    }
    seen_storage = set()
    seen_nonparam_storage = set()
    seen_param_storage = set()
    module_storage_keys = set(module_storage_keys or ())

    def _pack(x):
        if torch.is_tensor(x):
            nbytes = int(max(0, _tensor_nbytes(x)))
            stats["tensor_count"] += 1
            stats["total_bytes"] += nbytes
            stats["max_tensor_bytes"] = max(int(stats["max_tensor_bytes"]), nbytes)
            storage_key = _tensor_storage_key(x)
            storage_nbytes = int(max(0, _tensor_storage_nbytes(x)))
            if storage_key is None or storage_key not in seen_storage:
                if storage_key is not None:
                    seen_storage.add(storage_key)
                stats["unique_storage_bytes"] += storage_nbytes
            is_param_storage = bool(storage_key is not None and storage_key in module_storage_keys)
            if is_param_storage:
                if storage_key not in seen_param_storage:
                    seen_param_storage.add(storage_key)
                    stats["param_ref_unique_storage_bytes"] += storage_nbytes
            else:
                stats["nonparam_raw_bytes"] += nbytes
                if storage_key is None or storage_key not in seen_nonparam_storage:
                    if storage_key is not None:
                        seen_nonparam_storage.add(storage_key)
                    stats["nonparam_unique_storage_bytes"] += storage_nbytes
        return x

    def _unpack(x):
        return x

    with graph_mod.saved_tensors_hooks(_pack, _unpack):
        yield stats


class GroupedLinearProjector(nn.Module):
    def __init__(self, num_features, emsize, group_size, replace_nan_by_zero=True):
        super().__init__()
        self.num_features = int(num_features)
        self.emsize = int(emsize)
        self.group_size = int(max(1, group_size))
        self.num_groups = int(math.ceil(self.num_features / self.group_size))
        self.replace_nan_by_zero = bool(replace_nan_by_zero)
        self.weight = nn.Parameter(torch.empty(self.num_groups, self.group_size, self.emsize))
        self.bias = nn.Parameter(torch.zeros(self.num_groups, self.emsize))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        if int(x.shape[-1]) < self.num_features:
            raise ValueError(
                f"GroupedLinearProjector expected at least {self.num_features} features, got {int(x.shape[-1])}"
            )
        x = x[..., : self.num_features]
        if self.replace_nan_by_zero:
            x = torch.nan_to_num(x, nan=0.0)
        padded = int(self.num_groups * self.group_size - self.num_features)
        if padded > 0:
            x = F.pad(x, (0, padded))
        x = x.view(*x.shape[:-1], self.num_groups, self.group_size)
        return torch.einsum("...gf,gfe->...ge", x, self.weight) + self.bias


class GroupedFeatureEncoder(nn.Module):
    def __init__(
        self,
        *,
        n_features,
        emsize,
        features_per_group,
        x_encoder_type="single",
        x_obs_dim=None,
        x_action_dim=None,
        replace_nan_by_zero=True,
        feature_positional_embedding=True,
    ):
        super().__init__()
        self.n_features = int(n_features)
        self.emsize = int(emsize)
        self.x_encoder_type = str(x_encoder_type)
        self.replace_nan_by_zero = bool(replace_nan_by_zero)
        self.feature_positional_embedding = bool(feature_positional_embedding)
        self.obs_dim = int(x_obs_dim) if x_obs_dim is not None else None
        self.action_dim = int(x_action_dim) if x_action_dim is not None else None

        if self.x_encoder_type == "single":
            self.main_projector = GroupedLinearProjector(
                self.n_features,
                self.emsize,
                features_per_group,
                replace_nan_by_zero=self.replace_nan_by_zero,
            )
            self.num_groups = int(self.main_projector.num_groups)
        elif self.x_encoder_type == "split_obs_action":
            if self.obs_dim is None or self.action_dim is None:
                raise ValueError("x_obs_dim/x_action_dim must be set for split_obs_action grouped encoder.")
            self.obs_projector = GroupedLinearProjector(
                self.obs_dim,
                self.emsize,
                features_per_group,
                replace_nan_by_zero=self.replace_nan_by_zero,
            )
            self.action_projector = GroupedLinearProjector(
                self.action_dim,
                self.emsize,
                features_per_group,
                replace_nan_by_zero=self.replace_nan_by_zero,
            )
            self.num_groups = int(self.obs_projector.num_groups + self.action_projector.num_groups)
        else:
            raise ValueError(f"Unknown x_encoder_type: {self.x_encoder_type}")

        if self.feature_positional_embedding:
            self.group_positional_embedding = nn.Parameter(torch.zeros(self.num_groups, self.emsize))
            nn.init.normal_(self.group_positional_embedding, std=0.02)
        else:
            self.register_parameter("group_positional_embedding", None)

    def forward(self, x):
        if self.x_encoder_type == "single":
            out = self.main_projector(x)
        else:
            required = int(self.obs_dim + self.action_dim)
            if int(x.shape[-1]) < required:
                raise ValueError(
                    f"GroupedFeatureEncoder expected at least {required} features, got {int(x.shape[-1])}"
                )
            obs = x[..., : self.obs_dim]
            action = x[..., self.obs_dim : self.obs_dim + self.action_dim]
            out = torch.cat([self.obs_projector(obs), self.action_projector(action)], dim=-2)
        if self.group_positional_embedding is not None:
            out = out + self.group_positional_embedding.view(*([1] * (out.ndim - 2)), self.num_groups, self.emsize)
        return out


class PerFeatureCausalEncoderLayer(nn.Module):
    _ITEM_CACHE_LAYOUT_MICROCHUNKED = "per_feature_item_microchunked"

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward,
        dropout=0.0,
        activation="gelu",
        pre_norm=False,
        recompute_attn=False,
        single_eval_causal=False,
    ):
        super().__init__()
        self.recompute_attn = bool(recompute_attn)
        self.recompute_attn_use_reentrant = True
        self.feature_block = TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation=activation,
            pre_norm=pre_norm,
            recompute_attn=False,
            single_eval_causal=False,
        )
        self.item_block = TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation=activation,
            pre_norm=pre_norm,
            recompute_attn=recompute_attn,
            single_eval_causal=single_eval_causal,
        )
        try:
            item_paged_cow_grow_max_tokens = int(
                os.environ.get("TICL_PERFEATURE_ITEM_PAGED_COW_GROW_MAX_TOKENS", "32")
            )
        except Exception:
            item_paged_cow_grow_max_tokens = 32
        self.item_block.paged_cow_grow_max_tokens = int(max(0, item_paged_cow_grow_max_tokens))
        try:
            item_step_max_columns = int(
                os.environ.get("TICL_PERFEATURE_ITEM_STEP_MAX_COLUMNS", "256")
            )
        except Exception:
            item_step_max_columns = 256
        self.item_step_max_columns = int(max(1, item_step_max_columns))
        try:
            item_step_max_columns_warmup = int(
                os.environ.get("TICL_PERFEATURE_ITEM_STEP_MAX_COLUMNS_WARMUP", "512")
            )
        except Exception:
            item_step_max_columns_warmup = 512
        self.item_step_max_columns_warmup = int(max(1, item_step_max_columns_warmup))
        try:
            item_step_warmup_valid_len = int(
                os.environ.get("TICL_PERFEATURE_ITEM_STEP_WARMUP_VALID_LEN", "32")
            )
        except Exception:
            item_step_warmup_valid_len = 32
        self.item_step_warmup_valid_len = int(max(0, item_step_warmup_valid_len))
        profile_flag = str(os.environ.get("TICL_POLICY_STEP_PROFILE", "")).strip().lower()
        self._forward_step_profile_enabled = profile_flag in {"1", "true", "yes", "on"}
        self._feature_block_storage_keys = _module_storage_keys(self.feature_block)
        self._item_block_storage_keys = _module_storage_keys(self.item_block)
        self._forward_step_profile_stats = self._new_forward_step_profile_stats()

    @staticmethod
    def _new_forward_step_profile_stats():
        return {
            "perfeature_layer_calls": 0,
            "perfeature_feature_wall_s": 0.0,
            "perfeature_item_wall_s": 0.0,
            "perfeature_feature_saved_nonparam_raw_bytes_sum": 0,
            "perfeature_feature_saved_nonparam_raw_bytes_last": 0,
            "perfeature_feature_saved_nonparam_unique_storage_bytes_sum": 0,
            "perfeature_feature_saved_nonparam_unique_storage_bytes_last": 0,
            "perfeature_item_saved_nonparam_raw_bytes_sum": 0,
            "perfeature_item_saved_nonparam_raw_bytes_last": 0,
            "perfeature_item_saved_nonparam_unique_storage_bytes_sum": 0,
            "perfeature_item_saved_nonparam_unique_storage_bytes_last": 0,
            "perfeature_item_chunk_calls": 0,
            "perfeature_item_chunk_wall_s": 0.0,
            "perfeature_item_chunk_batch_sum": 0,
            "perfeature_item_chunk_batch_max": 0,
            "perfeature_item_chunk_saved_nonparam_raw_bytes_sum": 0,
            "perfeature_item_chunk_saved_nonparam_raw_bytes_last": 0,
            "perfeature_item_chunk_saved_nonparam_unique_storage_bytes_sum": 0,
            "perfeature_item_chunk_saved_nonparam_unique_storage_bytes_last": 0,
            "perfeature_item_chunk_saved_nonparam_unique_storage_bytes_max": 0,
            "perfeature_item_chunk_cache_unique_storage_bytes_last": 0,
            "perfeature_item_chunk_cache_unique_storage_bytes_max": 0,
        }

    def _record_forward_step_segment_stats(self, prefix, *, wall_s, saved_stats):
        stats = self._forward_step_profile_stats
        stats[f"{prefix}_wall_s"] = float(stats.get(f"{prefix}_wall_s", 0.0) or 0.0) + float(wall_s)
        raw_bytes = 0
        unique_bytes = 0
        if isinstance(saved_stats, dict):
            raw_bytes = int(max(0, int(saved_stats.get("nonparam_raw_bytes", 0) or 0)))
            unique_bytes = int(
                max(0, int(saved_stats.get("nonparam_unique_storage_bytes", 0) or 0))
            )
        stats[f"{prefix}_saved_nonparam_raw_bytes_sum"] = int(
            stats.get(f"{prefix}_saved_nonparam_raw_bytes_sum", 0) or 0
        ) + raw_bytes
        stats[f"{prefix}_saved_nonparam_raw_bytes_last"] = int(raw_bytes)
        stats[f"{prefix}_saved_nonparam_unique_storage_bytes_sum"] = int(
            stats.get(f"{prefix}_saved_nonparam_unique_storage_bytes_sum", 0) or 0
        ) + unique_bytes
        stats[f"{prefix}_saved_nonparam_unique_storage_bytes_last"] = int(unique_bytes)

    def _record_item_chunk_profile_stats(self, *, chunk_bs, wall_s, saved_stats, cache_obj):
        stats = self._forward_step_profile_stats
        stats["perfeature_item_chunk_calls"] = int(stats.get("perfeature_item_chunk_calls", 0) or 0) + 1
        stats["perfeature_item_chunk_wall_s"] = float(
            stats.get("perfeature_item_chunk_wall_s", 0.0) or 0.0
        ) + float(wall_s)
        stats["perfeature_item_chunk_batch_sum"] = int(
            stats.get("perfeature_item_chunk_batch_sum", 0) or 0
        ) + int(max(1, chunk_bs))
        stats["perfeature_item_chunk_batch_max"] = max(
            int(stats.get("perfeature_item_chunk_batch_max", 0) or 0),
            int(max(1, chunk_bs)),
        )
        raw_bytes = 0
        unique_bytes = 0
        if isinstance(saved_stats, dict):
            raw_bytes = int(max(0, int(saved_stats.get("nonparam_raw_bytes", 0) or 0)))
            unique_bytes = int(max(0, int(saved_stats.get("nonparam_unique_storage_bytes", 0) or 0)))
        stats["perfeature_item_chunk_saved_nonparam_raw_bytes_sum"] = int(
            stats.get("perfeature_item_chunk_saved_nonparam_raw_bytes_sum", 0) or 0
        ) + raw_bytes
        stats["perfeature_item_chunk_saved_nonparam_raw_bytes_last"] = int(raw_bytes)
        stats["perfeature_item_chunk_saved_nonparam_unique_storage_bytes_sum"] = int(
            stats.get("perfeature_item_chunk_saved_nonparam_unique_storage_bytes_sum", 0) or 0
        ) + unique_bytes
        stats["perfeature_item_chunk_saved_nonparam_unique_storage_bytes_last"] = int(unique_bytes)
        stats["perfeature_item_chunk_saved_nonparam_unique_storage_bytes_max"] = max(
            int(stats.get("perfeature_item_chunk_saved_nonparam_unique_storage_bytes_max", 0) or 0),
            unique_bytes,
        )
        cache_stats = _tensor_tree_storage_stats(cache_obj)
        cache_unique = int(max(0, int(cache_stats.get("unique_storage_bytes", 0) or 0)))
        stats["perfeature_item_chunk_cache_unique_storage_bytes_last"] = int(cache_unique)
        stats["perfeature_item_chunk_cache_unique_storage_bytes_max"] = max(
            int(stats.get("perfeature_item_chunk_cache_unique_storage_bytes_max", 0) or 0),
            cache_unique,
        )

    @staticmethod
    def _item_cache_valid_len(layer_cache):
        if layer_cache is None:
            return 0
        if (
            isinstance(layer_cache, dict)
            and str(layer_cache.get("cache_layout", "")) == "per_feature_item_microchunked"
        ):
            item_chunk_caches = layer_cache.get("item_chunk_caches", None)
            if isinstance(item_chunk_caches, list) and item_chunk_caches:
                return int(PerFeatureCausalEncoderLayer._item_cache_valid_len(item_chunk_caches[0]))
            return 0
        if isinstance(layer_cache, dict):
            try:
                return int(max(0, int(layer_cache.get("valid_len", 0))))
            except Exception:
                return 0
        return 0

    def _resolve_item_step_max_columns(self, valid_len):
        valid_len = int(max(0, valid_len))
        max_item_columns = int(max(1, getattr(self, "item_step_max_columns", 256)))
        warmup_valid_len = int(max(0, getattr(self, "item_step_warmup_valid_len", 0)))
        warmup_max_columns = int(max(1, getattr(self, "item_step_max_columns_warmup", max_item_columns)))
        if warmup_valid_len > 0 and valid_len < warmup_valid_len:
            return int(max(max_item_columns, warmup_max_columns))
        return int(max_item_columns)

    def _item_step_batch_microchunk_size(self, batch_size, num_groups, valid_len=0):
        batch_size = int(max(1, batch_size))
        num_groups = int(max(1, num_groups))
        if batch_size <= 1:
            return batch_size
        max_item_columns = self._resolve_item_step_max_columns(valid_len)
        max_batch_from_columns = int(max(1, max_item_columns // max(1, num_groups)))
        return int(max(1, min(batch_size, max_batch_from_columns)))

    @staticmethod
    def _chunk_batch_sizes(batch_size, micro_bs):
        batch_size = int(max(1, batch_size))
        micro_bs = int(max(1, micro_bs))
        sizes = []
        covered = 0
        while covered < batch_size:
            take = int(min(micro_bs, batch_size - covered))
            sizes.append(take)
            covered += take
        return sizes

    @classmethod
    def _is_chunked_item_cache(cls, layer_cache):
        return (
            isinstance(layer_cache, dict)
            and str(layer_cache.get("cache_layout", "")) == cls._ITEM_CACHE_LAYOUT_MICROCHUNKED
            and isinstance(layer_cache.get("item_chunk_caches", None), list)
        )

    @staticmethod
    def _slice_single_item_cache_batch(layer_cache, start_col, end_col):
        if layer_cache is None:
            return None
        start_col = int(max(0, start_col))
        end_col = int(max(start_col, end_col))
        sliced = {}
        for key, value in layer_cache.items():
            if torch.is_tensor(value):
                sliced[key] = value[start_col:end_col]
            elif key in {"k_pages", "v_pages"} and value is not None:
                sliced[key] = [page[start_col:end_col] for page in value]
            else:
                sliced[key] = value
        return sliced

    @classmethod
    def _slice_item_cache_batch(cls, layer_cache, start_col, end_col):
        if layer_cache is None:
            return None
        if cls._is_chunked_item_cache(layer_cache):
            num_groups = int(max(1, int(layer_cache.get("num_groups", 1))))
            chunk_batch_sizes = [int(max(1, bs)) for bs in layer_cache.get("chunk_batch_sizes", [])]
            if not chunk_batch_sizes:
                raise ValueError("Chunked per-feature cache is missing chunk_batch_sizes.")
            batch_start = int(start_col // num_groups)
            batch_end = int((max(start_col, end_col) + num_groups - 1) // num_groups)
            chunk_caches = []
            selected_sizes = []
            covered = 0
            for chunk_bs, cache_chunk in zip(chunk_batch_sizes, layer_cache["item_chunk_caches"]):
                next_covered = covered + chunk_bs
                overlaps = (batch_start < next_covered) and (batch_end > covered)
                if overlaps:
                    if batch_start > covered or batch_end < next_covered:
                        raise ValueError("Chunked per-feature cache only supports chunk-aligned batch slicing.")
                    chunk_caches.append(cache_chunk)
                    selected_sizes.append(chunk_bs)
                covered = next_covered
            return {
                "cache_layout": cls._ITEM_CACHE_LAYOUT_MICROCHUNKED,
                "item_chunk_caches": chunk_caches,
                "chunk_batch_sizes": selected_sizes,
                "num_groups": num_groups,
                "batch_size": int(sum(selected_sizes)),
                "microbatch_size": int(layer_cache.get("microbatch_size", 1)),
            }
        return cls._slice_single_item_cache_batch(layer_cache, start_col, end_col)

    @classmethod
    def _wrap_chunked_item_cache(cls, cache_chunks, chunk_batch_sizes, num_groups, micro_bs):
        return {
            "cache_layout": cls._ITEM_CACHE_LAYOUT_MICROCHUNKED,
            "item_chunk_caches": list(cache_chunks),
            "chunk_batch_sizes": [int(max(1, bs)) for bs in chunk_batch_sizes],
            "num_groups": int(max(1, num_groups)),
            "batch_size": int(sum(int(max(1, bs)) for bs in chunk_batch_sizes)),
            "microbatch_size": int(max(1, micro_bs)),
        }

    def _forward_item_step_batched(
        self,
        feature_state,
        kv_cache=None,
        append_to_cache=True,
        max_cache_len=None,
        kv_cache_mode: str = "auto",
        kv_cache_page_size=None,
        allow_grad_mutable_cache: bool = False,
        allow_grad_inplace_paged_cache: bool = False,
    ):
        batch_size, num_groups, dim = feature_state.shape
        valid_len = self._item_cache_valid_len(kv_cache)
        micro_bs = self._item_step_batch_microchunk_size(batch_size, num_groups, valid_len=valid_len)
        if (not torch.is_grad_enabled()) or micro_bs >= batch_size:
            item_in = feature_state.reshape(1, batch_size * num_groups, dim)
            item_out, item_cache = self.item_block.forward_step(
                item_in,
                kv_cache=kv_cache,
                append_to_cache=append_to_cache,
                max_cache_len=max_cache_len,
                kv_cache_mode=kv_cache_mode,
                kv_cache_page_size=kv_cache_page_size,
                allow_grad_mutable_cache=allow_grad_mutable_cache,
                allow_grad_inplace_paged_cache=allow_grad_inplace_paged_cache,
            )
            return item_out.reshape(batch_size, num_groups, dim), item_cache

        if self._is_chunked_item_cache(kv_cache):
            chunk_batch_sizes = [int(max(1, bs)) for bs in kv_cache.get("chunk_batch_sizes", [])]
            prev_chunk_caches = list(kv_cache.get("item_chunk_caches", []))
            if int(sum(chunk_batch_sizes)) != int(batch_size):
                raise ValueError(
                    f"Chunked per-feature cache batch size {sum(chunk_batch_sizes)} != current batch size {batch_size}"
                )
            if int(kv_cache.get("num_groups", num_groups)) != int(num_groups):
                raise ValueError(
                    f"Chunked per-feature cache num_groups {kv_cache.get('num_groups')} != current num_groups {num_groups}"
                )
            if len(prev_chunk_caches) != len(chunk_batch_sizes):
                raise ValueError("Chunked per-feature cache chunk count mismatch.")
        else:
            chunk_batch_sizes = self._chunk_batch_sizes(batch_size, micro_bs)
            prev_chunk_caches = []
            covered = 0
            for chunk_bs in chunk_batch_sizes:
                if kv_cache is None:
                    prev_chunk_caches.append(None)
                else:
                    col_start = int(covered * num_groups)
                    col_end = int((covered + chunk_bs) * num_groups)
                    prev_chunk_caches.append(self._slice_single_item_cache_batch(kv_cache, col_start, col_end))
                covered += chunk_bs

        outputs = []
        cache_chunks = []
        batch_start = 0
        chunk_profile_enabled = bool(self._forward_step_profile_enabled) and torch.is_grad_enabled() and (not _is_torch_compiling())
        for chunk_bs, cache_chunk in zip(chunk_batch_sizes, prev_chunk_caches):
            batch_end = batch_start + int(chunk_bs)
            feature_chunk = feature_state[batch_start:batch_end]
            item_in = feature_chunk.reshape(1, (batch_end - batch_start) * num_groups, dim)
            item_chunk_t0 = time.perf_counter() if chunk_profile_enabled else None
            with (
                _saved_tensors_profile_scope(
                    module_storage_keys=self._item_block_storage_keys,
                    enabled=chunk_profile_enabled,
                )
                if chunk_profile_enabled
                else nullcontext(None)
            ) as item_chunk_saved_stats:
                item_out, cache_next = self.item_block.forward_step(
                    item_in,
                    kv_cache=cache_chunk,
                    append_to_cache=append_to_cache,
                    max_cache_len=max_cache_len,
                    kv_cache_mode=kv_cache_mode,
                    kv_cache_page_size=kv_cache_page_size,
                    allow_grad_mutable_cache=allow_grad_mutable_cache,
                    allow_grad_inplace_paged_cache=allow_grad_inplace_paged_cache,
                )
            if chunk_profile_enabled:
                self._record_item_chunk_profile_stats(
                    chunk_bs=int(batch_end - batch_start),
                    wall_s=(time.perf_counter() - item_chunk_t0),
                    saved_stats=item_chunk_saved_stats,
                    cache_obj=cache_next,
                )
            outputs.append(item_out.reshape(batch_end - batch_start, num_groups, dim))
            cache_chunks.append(cache_next)
            batch_start = batch_end
        return (
            torch.cat(outputs, dim=0),
            self._wrap_chunked_item_cache(cache_chunks, chunk_batch_sizes, num_groups, micro_bs),
        )

    def _feature_forward_impl(self, state):
        if state.ndim == 4:
            t_len, batch_size, num_groups, dim = state.shape
            feature_in = state.reshape(t_len * batch_size, num_groups, dim)
            feature_out = self.feature_block(feature_in, src_mask=None)
            return feature_out.reshape(t_len, batch_size, num_groups, dim)
        if state.ndim == 3:
            return self.feature_block(state, src_mask=None)
        raise ValueError(f"Expected 3D or 4D state, got {tuple(state.shape)}")

    def _feature_forward(self, state):
        if self.recompute_attn and torch.is_grad_enabled():
            return checkpoint(
                self._feature_forward_impl,
                state,
                use_reentrant=bool(getattr(self, "recompute_attn_use_reentrant", True)),
            )
        return self._feature_forward_impl(state)

    def forward(self, state, src_mask=None):
        feature_state = self._feature_forward(state)
        if feature_state.ndim != 4:
            raise ValueError("PerFeatureCausalEncoderLayer.forward expects state with shape (T, B, G, E).")
        t_len, batch_size, num_groups, dim = feature_state.shape
        item_in = feature_state.reshape(t_len, batch_size * num_groups, dim)
        item_out = self.item_block(item_in, src_mask=src_mask)
        return item_out.reshape(t_len, batch_size, num_groups, dim)

    def forward_step(
        self,
        src_step,
        kv_cache=None,
        append_to_cache=True,
        max_cache_len=None,
        kv_cache_mode: str = "auto",
        kv_cache_page_size=None,
        allow_grad_mutable_cache: bool = False,
        allow_grad_inplace_paged_cache: bool = False,
    ):
        keep_seq_dim = bool(src_step.ndim == 4)
        state = src_step.squeeze(0) if keep_seq_dim else src_step
        profile_enabled = bool(self._forward_step_profile_enabled) and torch.is_grad_enabled() and (not _is_torch_compiling())
        if profile_enabled:
            self._forward_step_profile_stats["perfeature_layer_calls"] = int(
                self._forward_step_profile_stats.get("perfeature_layer_calls", 0) or 0
            ) + 1
        feature_t0 = time.perf_counter() if profile_enabled else None
        with (
            _saved_tensors_profile_scope(
                module_storage_keys=self._feature_block_storage_keys,
                enabled=profile_enabled,
            )
            if profile_enabled
            else nullcontext(None)
        ) as feature_saved_stats:
            feature_state = self._feature_forward(state)
        if profile_enabled:
            self._record_forward_step_segment_stats(
                "perfeature_feature",
                wall_s=(time.perf_counter() - feature_t0),
                saved_stats=feature_saved_stats,
            )
        item_t0 = time.perf_counter() if profile_enabled else None
        with (
            _saved_tensors_profile_scope(
                module_storage_keys=self._item_block_storage_keys,
                enabled=profile_enabled,
            )
            if profile_enabled
            else nullcontext(None)
        ) as item_saved_stats:
            item_out, item_cache = self._forward_item_step_batched(
                feature_state,
                kv_cache=kv_cache,
                append_to_cache=append_to_cache,
                max_cache_len=max_cache_len,
                kv_cache_mode=kv_cache_mode,
                kv_cache_page_size=kv_cache_page_size,
                allow_grad_mutable_cache=allow_grad_mutable_cache,
                allow_grad_inplace_paged_cache=allow_grad_inplace_paged_cache,
            )
        if profile_enabled:
            self._record_forward_step_segment_stats(
                "perfeature_item",
                wall_s=(time.perf_counter() - item_t0),
                saved_stats=item_saved_stats,
            )
        out = item_out
        if keep_seq_dim:
            out = out.unsqueeze(0)
        return out, item_cache

    def forward_causal_prefix(self, src_prefix):
        if src_prefix.ndim != 4:
            raise ValueError(
                f"PerFeatureCausalEncoderLayer.forward_causal_prefix expects (T, B, G, E), got {tuple(src_prefix.shape)}"
            )
        feature_state = self._feature_forward(src_prefix)
        t_len, batch_size, num_groups, dim = feature_state.shape
        item_in = feature_state.reshape(t_len, batch_size * num_groups, dim)
        item_out, item_cache = self.item_block.forward_causal_prefix(item_in)
        return item_out.reshape(t_len, batch_size, num_groups, dim), item_cache

    def forward_query(self, src_query, kv_cache):
        if src_query.ndim != 4:
            raise ValueError(
                f"PerFeatureCausalEncoderLayer.forward_query expects (T, B, G, E), got {tuple(src_query.shape)}"
            )
        feature_state = self._feature_forward(src_query)
        t_len, batch_size, num_groups, dim = feature_state.shape
        item_in = feature_state.reshape(t_len, batch_size * num_groups, dim)
        item_out = self.item_block.forward_query(item_in, kv_cache)
        return item_out.reshape(t_len, batch_size, num_groups, dim)

    def consume_forward_step_profile(self):
        consume_fn = getattr(self.item_block, "consume_forward_step_profile", None)
        item_block_stats = consume_fn() if callable(consume_fn) else None
        local_stats = dict(self._forward_step_profile_stats)
        self._forward_step_profile_stats = self._new_forward_step_profile_stats()
        if not isinstance(item_block_stats, dict) and int(local_stats.get("perfeature_layer_calls", 0) or 0) <= 0:
            return None
        merged = dict(item_block_stats) if isinstance(item_block_stats, dict) else {}
        merged.update(local_stats)
        return merged


class PerFeatureTransformerEncoderSimple(nn.Module):
    def __init__(self, encoder_layer_creator, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([encoder_layer_creator() for _ in range(num_layers)])
        self.num_layers = int(num_layers)

    def forward(self, src, mask=None):
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask)
        return output

    def forward_step(
        self,
        src_step,
        kv_cache=None,
        append_to_cache=True,
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
        new_cache = kv_cache if (isinstance(kv_cache, list) and append_to_cache and allow_grad_mutable_cache) else []
        for layer_idx, (layer, layer_cache) in enumerate(zip(self.layers, kv_cache)):
            output, cache_next = layer.forward_step(
                output,
                kv_cache=layer_cache,
                append_to_cache=append_to_cache,
                max_cache_len=max_cache_len,
                kv_cache_mode=kv_cache_mode,
                kv_cache_page_size=kv_cache_page_size,
                allow_grad_mutable_cache=allow_grad_mutable_cache,
                allow_grad_inplace_paged_cache=allow_grad_inplace_paged_cache,
            )
            if new_cache is kv_cache:
                new_cache[layer_idx] = cache_next
            else:
                new_cache.append(cache_next)
        return output, new_cache

    def encode_prefix_to_kv(self, src_prefix):
        output = src_prefix
        kv_cache = []
        for layer in self.layers:
            output, layer_cache = layer.forward_causal_prefix(output)
            kv_cache.append(layer_cache)
        return kv_cache, output

    def forward_query(self, src_query, kv_cache):
        if len(kv_cache) != len(self.layers):
            raise ValueError(f"kv_cache length {len(kv_cache)} != num_layers {len(self.layers)}")
        output = src_query
        for layer, layer_cache in zip(self.layers, kv_cache):
            output = layer.forward_query(output, layer_cache)
        return output

    def forward_with_prefix_cache(self, src_prefix, src_query):
        kv_cache, _ = self.encode_prefix_to_kv(src_prefix)
        output_q = self.forward_query(src_query, kv_cache) if int(src_query.shape[0]) > 0 else src_query
        return output_q, kv_cache

    def consume_step_profile(self):
        calls = 0
        perfeature_layer_calls = 0
        perfeature_feature_wall_s = 0.0
        perfeature_item_wall_s = 0.0
        perfeature_feature_saved_nonparam_raw_bytes_sum = 0
        perfeature_feature_saved_nonparam_raw_bytes_last = 0
        perfeature_feature_saved_nonparam_unique_storage_bytes_sum = 0
        perfeature_feature_saved_nonparam_unique_storage_bytes_last = 0
        perfeature_item_saved_nonparam_raw_bytes_sum = 0
        perfeature_item_saved_nonparam_raw_bytes_last = 0
        perfeature_item_saved_nonparam_unique_storage_bytes_sum = 0
        perfeature_item_saved_nonparam_unique_storage_bytes_last = 0
        perfeature_item_chunk_calls = 0
        perfeature_item_chunk_wall_s = 0.0
        perfeature_item_chunk_batch_sum = 0
        perfeature_item_chunk_batch_max = 0
        perfeature_item_chunk_saved_nonparam_raw_bytes_sum = 0
        perfeature_item_chunk_saved_nonparam_raw_bytes_last = 0
        perfeature_item_chunk_saved_nonparam_unique_storage_bytes_sum = 0
        perfeature_item_chunk_saved_nonparam_unique_storage_bytes_last = 0
        perfeature_item_chunk_saved_nonparam_unique_storage_bytes_max = 0
        perfeature_item_chunk_cache_unique_storage_bytes_last = 0
        perfeature_item_chunk_cache_unique_storage_bytes_max = 0
        attnff_saved_nonparam_raw_bytes_last = 0
        attnff_saved_nonparam_unique_storage_bytes_last = 0
        finalize_saved_nonparam_raw_bytes_last = 0
        finalize_saved_nonparam_unique_storage_bytes_last = 0
        proj_wall_s = 0.0
        cache_wall_s = 0.0
        cache_append_wall_s = 0.0
        paged_grow_calls = 0
        paged_grow_prev_total_bytes_last = 0
        paged_grow_prev_total_bytes_max = 0
        paged_grow_new_total_bytes_last = 0
        paged_grow_new_total_bytes_max = 0
        paged_grow_out_total_bytes_last = 0
        paged_grow_out_total_bytes_max = 0
        paged_grow_working_set_total_bytes_last = 0
        paged_grow_working_set_total_bytes_max = 0
        paged_grow_k_out_bytes_last = 0
        paged_grow_k_out_bytes_max = 0
        paged_grow_v_out_bytes_last = 0
        paged_grow_v_out_bytes_max = 0
        cache_prefix_maint_wall_s = 0.0
        attnff_wall_s = 0.0
        attn_core_wall_s = 0.0
        paged_view_build_wall_s = 0.0
        paged_clone_wall_s = 0.0
        paged_dispatch_single_page_wall_s = 0.0
        paged_dispatch_flash_prefix_wall_s = 0.0
        paged_dispatch_flash_merge_wall_s = 0.0
        paged_dispatch_dense_wall_s = 0.0
        finalize_wall_s = 0.0
        flash_prefix_prepare_prefix_wall_s = 0.0
        flash_prefix_prefix_core_wall_s = 0.0
        flash_prefix_tail_core_wall_s = 0.0
        flash_prefix_merge_wall_s = 0.0
        flash_prefix_wait_stream_wall_s = 0.0
        flash_prefix_prefix_cast_norm_wall_s = 0.0
        flash_prefix_tail_cast_norm_wall_s = 0.0
        flash_prefix_merge_logaddexp_wall_s = 0.0
        flash_prefix_merge_scale_wall_s = 0.0
        flash_prefix_merge_blend_wall_s = 0.0
        flash_prefix_prefix_saved_nonparam_unique_storage_bytes_last = 0
        flash_prefix_tail_saved_nonparam_unique_storage_bytes_last = 0
        flash_prefix_merge_saved_nonparam_unique_storage_bytes_last = 0
        flash_prefix_prefix_saved_q_nonparam_unique_storage_bytes_last = 0
        flash_prefix_prefix_saved_k_nonparam_unique_storage_bytes_last = 0
        flash_prefix_prefix_saved_v_nonparam_unique_storage_bytes_last = 0
        flash_prefix_prefix_saved_other_nonparam_unique_storage_bytes_last = 0
        flash_prefix_tail_saved_q_nonparam_unique_storage_bytes_last = 0
        flash_prefix_tail_saved_k_nonparam_unique_storage_bytes_last = 0
        flash_prefix_tail_saved_v_nonparam_unique_storage_bytes_last = 0
        flash_prefix_tail_saved_other_nonparam_unique_storage_bytes_last = 0
        flash_merge_chunk_prep_wall_s = 0.0
        flash_merge_chunk_core_wall_s = 0.0
        flash_merge_lse_merge_wall_s = 0.0
        flash_merge_chunk_count_sum = 0
        finalize_attn_outproj_wall_s = 0.0
        finalize_attn_outproj_linear_wall_s = 0.0
        finalize_attn_outproj_norm_wall_s = 0.0
        finalize_ffn_wall_s = 0.0
        finalize_ffn_linear1_act_wall_s = 0.0
        finalize_ffn_linear2_residual_norm_wall_s = 0.0
        finalize_ffn_linear2_wall_s = 0.0
        finalize_ffn_residual_norm_wall_s = 0.0
        finalize_compiled_wall_s = 0.0
        paged_path_single_page = 0
        paged_path_flash_prefix = 0
        paged_path_flash_prefix_zero_fastpath = 0
        paged_path_flash_merge = 0
        paged_path_dense = 0
        paged_page_count_sum = 0
        paged_valid_len_sum = 0
        paged_last_page_tokens_sum = 0
        paged_prefix_len_sum = 0
        flash_prefix_valid_tokens_sum = 0
        flash_prefix_prefix_tokens_sum = 0
        flash_prefix_tail_tokens_sum = 0
        dense_valid_tokens_sum = 0
        dense_prefix_tokens_sum = 0
        dense_tail_tokens_sum = 0
        total_wall_s = 0.0
        enabled = False
        for layer in self.layers:
            layer_stats = layer.consume_forward_step_profile()
            if not isinstance(layer_stats, dict):
                continue
            enabled = True
            calls += int(layer_stats.get("calls", 0) or 0)
            perfeature_layer_calls += int(layer_stats.get("perfeature_layer_calls", 0) or 0)
            perfeature_feature_wall_s += float(layer_stats.get("perfeature_feature_wall_s", 0.0) or 0.0)
            perfeature_item_wall_s += float(layer_stats.get("perfeature_item_wall_s", 0.0) or 0.0)
            perfeature_feature_saved_nonparam_raw_bytes_sum += int(
                layer_stats.get("perfeature_feature_saved_nonparam_raw_bytes_sum", 0) or 0
            )
            perfeature_feature_saved_nonparam_raw_bytes_last += int(
                layer_stats.get("perfeature_feature_saved_nonparam_raw_bytes_last", 0) or 0
            )
            perfeature_feature_saved_nonparam_unique_storage_bytes_sum += int(
                layer_stats.get("perfeature_feature_saved_nonparam_unique_storage_bytes_sum", 0) or 0
            )
            perfeature_feature_saved_nonparam_unique_storage_bytes_last += int(
                layer_stats.get("perfeature_feature_saved_nonparam_unique_storage_bytes_last", 0) or 0
            )
            perfeature_item_saved_nonparam_raw_bytes_sum += int(
                layer_stats.get("perfeature_item_saved_nonparam_raw_bytes_sum", 0) or 0
            )
            perfeature_item_saved_nonparam_raw_bytes_last += int(
                layer_stats.get("perfeature_item_saved_nonparam_raw_bytes_last", 0) or 0
            )
            perfeature_item_saved_nonparam_unique_storage_bytes_sum += int(
                layer_stats.get("perfeature_item_saved_nonparam_unique_storage_bytes_sum", 0) or 0
            )
            perfeature_item_saved_nonparam_unique_storage_bytes_last += int(
                layer_stats.get("perfeature_item_saved_nonparam_unique_storage_bytes_last", 0) or 0
            )
            perfeature_item_chunk_calls += int(layer_stats.get("perfeature_item_chunk_calls", 0) or 0)
            perfeature_item_chunk_wall_s += float(layer_stats.get("perfeature_item_chunk_wall_s", 0.0) or 0.0)
            perfeature_item_chunk_batch_sum += int(layer_stats.get("perfeature_item_chunk_batch_sum", 0) or 0)
            perfeature_item_chunk_batch_max = max(
                int(perfeature_item_chunk_batch_max),
                int(layer_stats.get("perfeature_item_chunk_batch_max", 0) or 0),
            )
            perfeature_item_chunk_saved_nonparam_raw_bytes_sum += int(
                layer_stats.get("perfeature_item_chunk_saved_nonparam_raw_bytes_sum", 0) or 0
            )
            perfeature_item_chunk_saved_nonparam_raw_bytes_last += int(
                layer_stats.get("perfeature_item_chunk_saved_nonparam_raw_bytes_last", 0) or 0
            )
            perfeature_item_chunk_saved_nonparam_unique_storage_bytes_sum += int(
                layer_stats.get("perfeature_item_chunk_saved_nonparam_unique_storage_bytes_sum", 0) or 0
            )
            perfeature_item_chunk_saved_nonparam_unique_storage_bytes_last += int(
                layer_stats.get("perfeature_item_chunk_saved_nonparam_unique_storage_bytes_last", 0) or 0
            )
            perfeature_item_chunk_saved_nonparam_unique_storage_bytes_max = max(
                int(perfeature_item_chunk_saved_nonparam_unique_storage_bytes_max),
                int(layer_stats.get("perfeature_item_chunk_saved_nonparam_unique_storage_bytes_max", 0) or 0),
            )
            perfeature_item_chunk_cache_unique_storage_bytes_last += int(
                layer_stats.get("perfeature_item_chunk_cache_unique_storage_bytes_last", 0) or 0
            )
            perfeature_item_chunk_cache_unique_storage_bytes_max = max(
                int(perfeature_item_chunk_cache_unique_storage_bytes_max),
                int(layer_stats.get("perfeature_item_chunk_cache_unique_storage_bytes_max", 0) or 0),
            )
            attnff_saved_nonparam_raw_bytes_last += int(
                layer_stats.get("attnff_saved_nonparam_raw_bytes_last", 0) or 0
            )
            attnff_saved_nonparam_unique_storage_bytes_last += int(
                layer_stats.get("attnff_saved_nonparam_unique_storage_bytes_last", 0) or 0
            )
            finalize_saved_nonparam_raw_bytes_last += int(
                layer_stats.get("finalize_saved_nonparam_raw_bytes_last", 0) or 0
            )
            finalize_saved_nonparam_unique_storage_bytes_last += int(
                layer_stats.get("finalize_saved_nonparam_unique_storage_bytes_last", 0) or 0
            )
            proj_wall_s += float(layer_stats.get("proj_wall_s", 0.0) or 0.0)
            cache_wall_s += float(layer_stats.get("cache_wall_s", 0.0) or 0.0)
            cache_append_wall_s += float(layer_stats.get("cache_append_wall_s", 0.0) or 0.0)
            paged_grow_calls += int(layer_stats.get("paged_grow_calls", 0) or 0)
            paged_grow_prev_total_bytes_last = int(layer_stats.get("paged_grow_prev_total_bytes_last", 0) or 0)
            paged_grow_prev_total_bytes_max = max(
                int(paged_grow_prev_total_bytes_max),
                int(layer_stats.get("paged_grow_prev_total_bytes_max", 0) or 0),
            )
            paged_grow_new_total_bytes_last = int(layer_stats.get("paged_grow_new_total_bytes_last", 0) or 0)
            paged_grow_new_total_bytes_max = max(
                int(paged_grow_new_total_bytes_max),
                int(layer_stats.get("paged_grow_new_total_bytes_max", 0) or 0),
            )
            paged_grow_out_total_bytes_last = int(layer_stats.get("paged_grow_out_total_bytes_last", 0) or 0)
            paged_grow_out_total_bytes_max = max(
                int(paged_grow_out_total_bytes_max),
                int(layer_stats.get("paged_grow_out_total_bytes_max", 0) or 0),
            )
            paged_grow_working_set_total_bytes_last = int(
                layer_stats.get("paged_grow_working_set_total_bytes_last", 0) or 0
            )
            paged_grow_working_set_total_bytes_max = max(
                int(paged_grow_working_set_total_bytes_max),
                int(layer_stats.get("paged_grow_working_set_total_bytes_max", 0) or 0),
            )
            paged_grow_k_out_bytes_last = int(layer_stats.get("paged_grow_k_out_bytes_last", 0) or 0)
            paged_grow_k_out_bytes_max = max(
                int(paged_grow_k_out_bytes_max),
                int(layer_stats.get("paged_grow_k_out_bytes_max", 0) or 0),
            )
            paged_grow_v_out_bytes_last = int(layer_stats.get("paged_grow_v_out_bytes_last", 0) or 0)
            paged_grow_v_out_bytes_max = max(
                int(paged_grow_v_out_bytes_max),
                int(layer_stats.get("paged_grow_v_out_bytes_max", 0) or 0),
            )
            cache_prefix_maint_wall_s += float(layer_stats.get("cache_prefix_maint_wall_s", 0.0) or 0.0)
            attnff_wall_s += float(layer_stats.get("attnff_wall_s", 0.0) or 0.0)
            attn_core_wall_s += float(layer_stats.get("attn_core_wall_s", 0.0) or 0.0)
            paged_view_build_wall_s += float(layer_stats.get("paged_view_build_wall_s", 0.0) or 0.0)
            paged_clone_wall_s += float(layer_stats.get("paged_clone_wall_s", 0.0) or 0.0)
            paged_dispatch_single_page_wall_s += float(
                layer_stats.get("paged_dispatch_single_page_wall_s", 0.0) or 0.0
            )
            paged_dispatch_flash_prefix_wall_s += float(
                layer_stats.get("paged_dispatch_flash_prefix_wall_s", 0.0) or 0.0
            )
            paged_dispatch_flash_merge_wall_s += float(
                layer_stats.get("paged_dispatch_flash_merge_wall_s", 0.0) or 0.0
            )
            paged_dispatch_dense_wall_s += float(
                layer_stats.get("paged_dispatch_dense_wall_s", 0.0) or 0.0
            )
            finalize_wall_s += float(layer_stats.get("finalize_wall_s", 0.0) or 0.0)
            flash_prefix_prepare_prefix_wall_s += float(
                layer_stats.get("flash_prefix_prepare_prefix_wall_s", 0.0) or 0.0
            )
            flash_prefix_prefix_core_wall_s += float(
                layer_stats.get("flash_prefix_prefix_core_wall_s", 0.0) or 0.0
            )
            flash_prefix_tail_core_wall_s += float(
                layer_stats.get("flash_prefix_tail_core_wall_s", 0.0) or 0.0
            )
            flash_prefix_merge_wall_s += float(
                layer_stats.get("flash_prefix_merge_wall_s", 0.0) or 0.0
            )
            flash_prefix_wait_stream_wall_s += float(
                layer_stats.get("flash_prefix_wait_stream_wall_s", 0.0) or 0.0
            )
            flash_prefix_prefix_cast_norm_wall_s += float(
                layer_stats.get("flash_prefix_prefix_cast_norm_wall_s", 0.0) or 0.0
            )
            flash_prefix_tail_cast_norm_wall_s += float(
                layer_stats.get("flash_prefix_tail_cast_norm_wall_s", 0.0) or 0.0
            )
            flash_prefix_merge_logaddexp_wall_s += float(
                layer_stats.get("flash_prefix_merge_logaddexp_wall_s", 0.0) or 0.0
            )
            flash_prefix_merge_scale_wall_s += float(
                layer_stats.get("flash_prefix_merge_scale_wall_s", 0.0) or 0.0
            )
            flash_prefix_merge_blend_wall_s += float(
                layer_stats.get("flash_prefix_merge_blend_wall_s", 0.0) or 0.0
            )
            flash_prefix_prefix_saved_nonparam_unique_storage_bytes_last += int(
                layer_stats.get("flash_prefix_prefix_saved_nonparam_unique_storage_bytes_last", 0) or 0
            )
            flash_prefix_tail_saved_nonparam_unique_storage_bytes_last += int(
                layer_stats.get("flash_prefix_tail_saved_nonparam_unique_storage_bytes_last", 0) or 0
            )
            flash_prefix_merge_saved_nonparam_unique_storage_bytes_last += int(
                layer_stats.get("flash_prefix_merge_saved_nonparam_unique_storage_bytes_last", 0) or 0
            )
            flash_prefix_prefix_saved_q_nonparam_unique_storage_bytes_last += int(
                layer_stats.get("flash_prefix_prefix_saved_q_nonparam_unique_storage_bytes_last", 0) or 0
            )
            flash_prefix_prefix_saved_k_nonparam_unique_storage_bytes_last += int(
                layer_stats.get("flash_prefix_prefix_saved_k_nonparam_unique_storage_bytes_last", 0) or 0
            )
            flash_prefix_prefix_saved_v_nonparam_unique_storage_bytes_last += int(
                layer_stats.get("flash_prefix_prefix_saved_v_nonparam_unique_storage_bytes_last", 0) or 0
            )
            flash_prefix_prefix_saved_other_nonparam_unique_storage_bytes_last += int(
                layer_stats.get("flash_prefix_prefix_saved_other_nonparam_unique_storage_bytes_last", 0) or 0
            )
            flash_prefix_tail_saved_q_nonparam_unique_storage_bytes_last += int(
                layer_stats.get("flash_prefix_tail_saved_q_nonparam_unique_storage_bytes_last", 0) or 0
            )
            flash_prefix_tail_saved_k_nonparam_unique_storage_bytes_last += int(
                layer_stats.get("flash_prefix_tail_saved_k_nonparam_unique_storage_bytes_last", 0) or 0
            )
            flash_prefix_tail_saved_v_nonparam_unique_storage_bytes_last += int(
                layer_stats.get("flash_prefix_tail_saved_v_nonparam_unique_storage_bytes_last", 0) or 0
            )
            flash_prefix_tail_saved_other_nonparam_unique_storage_bytes_last += int(
                layer_stats.get("flash_prefix_tail_saved_other_nonparam_unique_storage_bytes_last", 0) or 0
            )
            flash_merge_chunk_prep_wall_s += float(
                layer_stats.get("flash_merge_chunk_prep_wall_s", 0.0) or 0.0
            )
            flash_merge_chunk_core_wall_s += float(
                layer_stats.get("flash_merge_chunk_core_wall_s", 0.0) or 0.0
            )
            flash_merge_lse_merge_wall_s += float(
                layer_stats.get("flash_merge_lse_merge_wall_s", 0.0) or 0.0
            )
            flash_merge_chunk_count_sum += int(layer_stats.get("flash_merge_chunk_count_sum", 0) or 0)
            finalize_attn_outproj_wall_s += float(layer_stats.get("finalize_attn_outproj_wall_s", 0.0) or 0.0)
            finalize_attn_outproj_linear_wall_s += float(
                layer_stats.get("finalize_attn_outproj_linear_wall_s", 0.0) or 0.0
            )
            finalize_attn_outproj_norm_wall_s += float(
                layer_stats.get("finalize_attn_outproj_norm_wall_s", 0.0) or 0.0
            )
            finalize_ffn_wall_s += float(layer_stats.get("finalize_ffn_wall_s", 0.0) or 0.0)
            finalize_ffn_linear1_act_wall_s += float(
                layer_stats.get("finalize_ffn_linear1_act_wall_s", 0.0) or 0.0
            )
            finalize_ffn_linear2_residual_norm_wall_s += float(
                layer_stats.get("finalize_ffn_linear2_residual_norm_wall_s", 0.0) or 0.0
            )
            finalize_ffn_linear2_wall_s += float(layer_stats.get("finalize_ffn_linear2_wall_s", 0.0) or 0.0)
            finalize_ffn_residual_norm_wall_s += float(
                layer_stats.get("finalize_ffn_residual_norm_wall_s", 0.0) or 0.0
            )
            finalize_compiled_wall_s += float(layer_stats.get("finalize_compiled_wall_s", 0.0) or 0.0)
            paged_path_single_page += int(layer_stats.get("paged_path_single_page", 0) or 0)
            paged_path_flash_prefix += int(layer_stats.get("paged_path_flash_prefix", 0) or 0)
            paged_path_flash_prefix_zero_fastpath += int(
                layer_stats.get("paged_path_flash_prefix_zero_fastpath", 0) or 0
            )
            paged_path_flash_merge += int(layer_stats.get("paged_path_flash_merge", 0) or 0)
            paged_path_dense += int(layer_stats.get("paged_path_dense", 0) or 0)
            paged_page_count_sum += int(layer_stats.get("paged_page_count_sum", 0) or 0)
            paged_valid_len_sum += int(layer_stats.get("paged_valid_len_sum", 0) or 0)
            paged_last_page_tokens_sum += int(layer_stats.get("paged_last_page_tokens_sum", 0) or 0)
            paged_prefix_len_sum += int(layer_stats.get("paged_prefix_len_sum", 0) or 0)
            flash_prefix_valid_tokens_sum += int(layer_stats.get("flash_prefix_valid_tokens_sum", 0) or 0)
            flash_prefix_prefix_tokens_sum += int(layer_stats.get("flash_prefix_prefix_tokens_sum", 0) or 0)
            flash_prefix_tail_tokens_sum += int(layer_stats.get("flash_prefix_tail_tokens_sum", 0) or 0)
            dense_valid_tokens_sum += int(layer_stats.get("dense_valid_tokens_sum", 0) or 0)
            dense_prefix_tokens_sum += int(layer_stats.get("dense_prefix_tokens_sum", 0) or 0)
            dense_tail_tokens_sum += int(layer_stats.get("dense_tail_tokens_sum", 0) or 0)
            total_wall_s += float(layer_stats.get("total_wall_s", 0.0) or 0.0)
        if not enabled:
            return None
        return {
            "calls": int(calls),
            "perfeature_layer_calls": int(perfeature_layer_calls),
            "perfeature_feature_wall_s": float(perfeature_feature_wall_s),
            "perfeature_item_wall_s": float(perfeature_item_wall_s),
            "perfeature_feature_saved_nonparam_raw_bytes_sum": int(
                perfeature_feature_saved_nonparam_raw_bytes_sum
            ),
            "perfeature_feature_saved_nonparam_raw_bytes_last": int(
                perfeature_feature_saved_nonparam_raw_bytes_last
            ),
            "perfeature_feature_saved_nonparam_unique_storage_bytes_sum": int(
                perfeature_feature_saved_nonparam_unique_storage_bytes_sum
            ),
            "perfeature_feature_saved_nonparam_unique_storage_bytes_last": int(
                perfeature_feature_saved_nonparam_unique_storage_bytes_last
            ),
            "perfeature_item_saved_nonparam_raw_bytes_sum": int(perfeature_item_saved_nonparam_raw_bytes_sum),
            "perfeature_item_saved_nonparam_raw_bytes_last": int(perfeature_item_saved_nonparam_raw_bytes_last),
            "perfeature_item_saved_nonparam_unique_storage_bytes_sum": int(
                perfeature_item_saved_nonparam_unique_storage_bytes_sum
            ),
            "perfeature_item_saved_nonparam_unique_storage_bytes_last": int(
                perfeature_item_saved_nonparam_unique_storage_bytes_last
            ),
            "perfeature_item_chunk_calls": int(perfeature_item_chunk_calls),
            "perfeature_item_chunk_wall_s": float(perfeature_item_chunk_wall_s),
            "perfeature_item_chunk_batch_sum": int(perfeature_item_chunk_batch_sum),
            "perfeature_item_chunk_batch_max": int(perfeature_item_chunk_batch_max),
            "perfeature_item_chunk_saved_nonparam_raw_bytes_sum": int(
                perfeature_item_chunk_saved_nonparam_raw_bytes_sum
            ),
            "perfeature_item_chunk_saved_nonparam_raw_bytes_last": int(
                perfeature_item_chunk_saved_nonparam_raw_bytes_last
            ),
            "perfeature_item_chunk_saved_nonparam_unique_storage_bytes_sum": int(
                perfeature_item_chunk_saved_nonparam_unique_storage_bytes_sum
            ),
            "perfeature_item_chunk_saved_nonparam_unique_storage_bytes_last": int(
                perfeature_item_chunk_saved_nonparam_unique_storage_bytes_last
            ),
            "perfeature_item_chunk_saved_nonparam_unique_storage_bytes_max": int(
                perfeature_item_chunk_saved_nonparam_unique_storage_bytes_max
            ),
            "perfeature_item_chunk_cache_unique_storage_bytes_last": int(
                perfeature_item_chunk_cache_unique_storage_bytes_last
            ),
            "perfeature_item_chunk_cache_unique_storage_bytes_max": int(
                perfeature_item_chunk_cache_unique_storage_bytes_max
            ),
            "attnff_saved_nonparam_raw_bytes_last": int(attnff_saved_nonparam_raw_bytes_last),
            "attnff_saved_nonparam_unique_storage_bytes_last": int(
                attnff_saved_nonparam_unique_storage_bytes_last
            ),
            "finalize_saved_nonparam_raw_bytes_last": int(finalize_saved_nonparam_raw_bytes_last),
            "finalize_saved_nonparam_unique_storage_bytes_last": int(
                finalize_saved_nonparam_unique_storage_bytes_last
            ),
            "proj_wall_s": float(proj_wall_s),
            "cache_wall_s": float(cache_wall_s),
            "cache_append_wall_s": float(cache_append_wall_s),
            "paged_grow_calls": int(paged_grow_calls),
            "paged_grow_prev_total_bytes_last": int(paged_grow_prev_total_bytes_last),
            "paged_grow_prev_total_bytes_max": int(paged_grow_prev_total_bytes_max),
            "paged_grow_new_total_bytes_last": int(paged_grow_new_total_bytes_last),
            "paged_grow_new_total_bytes_max": int(paged_grow_new_total_bytes_max),
            "paged_grow_out_total_bytes_last": int(paged_grow_out_total_bytes_last),
            "paged_grow_out_total_bytes_max": int(paged_grow_out_total_bytes_max),
            "paged_grow_working_set_total_bytes_last": int(paged_grow_working_set_total_bytes_last),
            "paged_grow_working_set_total_bytes_max": int(paged_grow_working_set_total_bytes_max),
            "paged_grow_k_out_bytes_last": int(paged_grow_k_out_bytes_last),
            "paged_grow_k_out_bytes_max": int(paged_grow_k_out_bytes_max),
            "paged_grow_v_out_bytes_last": int(paged_grow_v_out_bytes_last),
            "paged_grow_v_out_bytes_max": int(paged_grow_v_out_bytes_max),
            "cache_prefix_maint_wall_s": float(cache_prefix_maint_wall_s),
            "attnff_wall_s": float(attnff_wall_s),
            "attn_core_wall_s": float(attn_core_wall_s),
            "paged_view_build_wall_s": float(paged_view_build_wall_s),
            "paged_clone_wall_s": float(paged_clone_wall_s),
            "paged_dispatch_single_page_wall_s": float(paged_dispatch_single_page_wall_s),
            "paged_dispatch_flash_prefix_wall_s": float(paged_dispatch_flash_prefix_wall_s),
            "paged_dispatch_flash_merge_wall_s": float(paged_dispatch_flash_merge_wall_s),
            "paged_dispatch_dense_wall_s": float(paged_dispatch_dense_wall_s),
            "finalize_wall_s": float(finalize_wall_s),
            "flash_prefix_prepare_prefix_wall_s": float(flash_prefix_prepare_prefix_wall_s),
            "flash_prefix_prefix_core_wall_s": float(flash_prefix_prefix_core_wall_s),
            "flash_prefix_tail_core_wall_s": float(flash_prefix_tail_core_wall_s),
            "flash_prefix_merge_wall_s": float(flash_prefix_merge_wall_s),
            "flash_prefix_wait_stream_wall_s": float(flash_prefix_wait_stream_wall_s),
            "flash_prefix_prefix_cast_norm_wall_s": float(flash_prefix_prefix_cast_norm_wall_s),
            "flash_prefix_tail_cast_norm_wall_s": float(flash_prefix_tail_cast_norm_wall_s),
            "flash_prefix_merge_logaddexp_wall_s": float(flash_prefix_merge_logaddexp_wall_s),
            "flash_prefix_merge_scale_wall_s": float(flash_prefix_merge_scale_wall_s),
            "flash_prefix_merge_blend_wall_s": float(flash_prefix_merge_blend_wall_s),
            "flash_prefix_prefix_saved_nonparam_unique_storage_bytes_last": int(
                flash_prefix_prefix_saved_nonparam_unique_storage_bytes_last
            ),
            "flash_prefix_tail_saved_nonparam_unique_storage_bytes_last": int(
                flash_prefix_tail_saved_nonparam_unique_storage_bytes_last
            ),
            "flash_prefix_merge_saved_nonparam_unique_storage_bytes_last": int(
                flash_prefix_merge_saved_nonparam_unique_storage_bytes_last
            ),
            "flash_prefix_prefix_saved_q_nonparam_unique_storage_bytes_last": int(
                flash_prefix_prefix_saved_q_nonparam_unique_storage_bytes_last
            ),
            "flash_prefix_prefix_saved_k_nonparam_unique_storage_bytes_last": int(
                flash_prefix_prefix_saved_k_nonparam_unique_storage_bytes_last
            ),
            "flash_prefix_prefix_saved_v_nonparam_unique_storage_bytes_last": int(
                flash_prefix_prefix_saved_v_nonparam_unique_storage_bytes_last
            ),
            "flash_prefix_prefix_saved_other_nonparam_unique_storage_bytes_last": int(
                flash_prefix_prefix_saved_other_nonparam_unique_storage_bytes_last
            ),
            "flash_prefix_tail_saved_q_nonparam_unique_storage_bytes_last": int(
                flash_prefix_tail_saved_q_nonparam_unique_storage_bytes_last
            ),
            "flash_prefix_tail_saved_k_nonparam_unique_storage_bytes_last": int(
                flash_prefix_tail_saved_k_nonparam_unique_storage_bytes_last
            ),
            "flash_prefix_tail_saved_v_nonparam_unique_storage_bytes_last": int(
                flash_prefix_tail_saved_v_nonparam_unique_storage_bytes_last
            ),
            "flash_prefix_tail_saved_other_nonparam_unique_storage_bytes_last": int(
                flash_prefix_tail_saved_other_nonparam_unique_storage_bytes_last
            ),
            "flash_merge_chunk_prep_wall_s": float(flash_merge_chunk_prep_wall_s),
            "flash_merge_chunk_core_wall_s": float(flash_merge_chunk_core_wall_s),
            "flash_merge_lse_merge_wall_s": float(flash_merge_lse_merge_wall_s),
            "flash_merge_chunk_count_sum": int(flash_merge_chunk_count_sum),
            "finalize_attn_outproj_wall_s": float(finalize_attn_outproj_wall_s),
            "finalize_attn_outproj_linear_wall_s": float(finalize_attn_outproj_linear_wall_s),
            "finalize_attn_outproj_norm_wall_s": float(finalize_attn_outproj_norm_wall_s),
            "finalize_ffn_wall_s": float(finalize_ffn_wall_s),
            "finalize_ffn_linear1_act_wall_s": float(finalize_ffn_linear1_act_wall_s),
            "finalize_ffn_linear2_residual_norm_wall_s": float(finalize_ffn_linear2_residual_norm_wall_s),
            "finalize_ffn_linear2_wall_s": float(finalize_ffn_linear2_wall_s),
            "finalize_ffn_residual_norm_wall_s": float(finalize_ffn_residual_norm_wall_s),
            "finalize_compiled_wall_s": float(finalize_compiled_wall_s),
            "paged_path_single_page": int(paged_path_single_page),
            "paged_path_flash_prefix": int(paged_path_flash_prefix),
            "paged_path_flash_prefix_zero_fastpath": int(paged_path_flash_prefix_zero_fastpath),
            "paged_path_flash_merge": int(paged_path_flash_merge),
            "paged_path_dense": int(paged_path_dense),
            "paged_page_count_sum": int(paged_page_count_sum),
            "paged_valid_len_sum": int(paged_valid_len_sum),
            "paged_last_page_tokens_sum": int(paged_last_page_tokens_sum),
            "paged_prefix_len_sum": int(paged_prefix_len_sum),
            "flash_prefix_valid_tokens_sum": int(flash_prefix_valid_tokens_sum),
            "flash_prefix_prefix_tokens_sum": int(flash_prefix_prefix_tokens_sum),
            "flash_prefix_tail_tokens_sum": int(flash_prefix_tail_tokens_sum),
            "dense_valid_tokens_sum": int(dense_valid_tokens_sum),
            "dense_prefix_tokens_sum": int(dense_prefix_tokens_sum),
            "dense_tail_tokens_sum": int(dense_tail_tokens_sum),
            "total_wall_s": float(total_wall_s),
        }


class PerFeatureTabPFN(nn.Module):
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
        efficient_eval_masking=True,
        y_encoder=None,
        tabpfn_zero_weights=False,
        x_encoder_type="single",
        x_obs_dim=None,
        x_action_dim=None,
        single_eval_causal=False,
        backbone_variant=None,
        features_per_group=3,
        feature_positional_embedding=True,
    ):
        super().__init__()
        del backbone_variant, y_encoder
        self.classification_task = bool(classification_task)
        self.efficient_eval_masking = bool(efficient_eval_masking)
        self.single_eval_causal = bool(single_eval_causal)
        self.emsize = int(emsize)
        self.n_out = int(n_out)
        self.nhid = int(emsize * nhid_factor)
        self.x_encoder_type = str(x_encoder_type)
        self.y_encoder = y_encoder_layer

        self.encoder = GroupedFeatureEncoder(
            n_features=n_features,
            emsize=emsize,
            features_per_group=features_per_group,
            x_encoder_type=x_encoder_type,
            x_obs_dim=x_obs_dim,
            x_action_dim=x_action_dim,
            replace_nan_by_zero=True,
            feature_positional_embedding=feature_positional_embedding,
        )
        self.num_groups = int(self.encoder.num_groups)
        self.features_per_group = int(max(1, features_per_group))

        def encoder_layer_creator():
            return PerFeatureCausalEncoderLayer(
                emsize,
                nhead,
                self.nhid,
                dropout=dropout,
                activation=activation,
                pre_norm=pre_norm,
                recompute_attn=recompute_attn,
                single_eval_causal=self.single_eval_causal,
            )

        self.transformer_encoder = PerFeatureTransformerEncoderSimple(encoder_layer_creator, nlayers)
        self.decoder = (
            decoder(emsize, self.nhid, n_out)
            if decoder is not None
            else nn.Sequential(nn.Linear(emsize, self.nhid), nn.GELU(), nn.Linear(self.nhid, n_out))
        )
        self.input_ln = SeqBN(emsize) if input_normalization else None
        self.init_method = init_method
        self.tabpfn_zero_weights = bool(tabpfn_zero_weights)
        profile_flag = str(os.environ.get("TICL_POLICY_STEP_PROFILE", "")).strip().lower()
        self._policy_step_profile_enabled = profile_flag in {"1", "true", "yes", "on"}
        self._policy_step_profile_stats = self._new_policy_step_profile_stats()
        self.init_weights()

    @staticmethod
    def _new_policy_step_profile_stats():
        return {
            "calls": 0,
            "encode_wall_s": 0.0,
            "transformer_wall_s": 0.0,
            "decoder_wall_s": 0.0,
            "total_wall_s": 0.0,
            "transformer_layer_calls": 0,
            "transformer_layer_proj_wall_s": 0.0,
            "transformer_layer_cache_wall_s": 0.0,
            "transformer_layer_cache_append_wall_s": 0.0,
            "transformer_layer_paged_grow_calls": 0,
            "transformer_layer_paged_grow_prev_total_bytes_last": 0,
            "transformer_layer_paged_grow_prev_total_bytes_max": 0,
            "transformer_layer_paged_grow_new_total_bytes_last": 0,
            "transformer_layer_paged_grow_new_total_bytes_max": 0,
            "transformer_layer_paged_grow_out_total_bytes_last": 0,
            "transformer_layer_paged_grow_out_total_bytes_max": 0,
            "transformer_layer_paged_grow_working_set_total_bytes_last": 0,
            "transformer_layer_paged_grow_working_set_total_bytes_max": 0,
            "transformer_layer_paged_grow_k_out_bytes_last": 0,
            "transformer_layer_paged_grow_k_out_bytes_max": 0,
            "transformer_layer_paged_grow_v_out_bytes_last": 0,
            "transformer_layer_paged_grow_v_out_bytes_max": 0,
            "transformer_layer_cache_prefix_maint_wall_s": 0.0,
            "transformer_layer_attnff_wall_s": 0.0,
            "transformer_layer_attnff_saved_nonparam_raw_bytes_last": 0,
            "transformer_layer_attnff_saved_nonparam_unique_storage_bytes_last": 0,
            "transformer_layer_attn_core_wall_s": 0.0,
            "transformer_layer_paged_view_build_wall_s": 0.0,
            "transformer_layer_paged_clone_wall_s": 0.0,
            "transformer_layer_paged_dispatch_single_page_wall_s": 0.0,
            "transformer_layer_paged_dispatch_flash_prefix_wall_s": 0.0,
            "transformer_layer_paged_dispatch_flash_merge_wall_s": 0.0,
            "transformer_layer_paged_dispatch_dense_wall_s": 0.0,
            "transformer_layer_finalize_wall_s": 0.0,
            "transformer_layer_finalize_saved_nonparam_raw_bytes_last": 0,
            "transformer_layer_finalize_saved_nonparam_unique_storage_bytes_last": 0,
            "transformer_layer_flash_prefix_prepare_prefix_wall_s": 0.0,
            "transformer_layer_flash_prefix_prefix_core_wall_s": 0.0,
            "transformer_layer_flash_prefix_tail_core_wall_s": 0.0,
            "transformer_layer_flash_prefix_merge_wall_s": 0.0,
            "transformer_layer_flash_prefix_wait_stream_wall_s": 0.0,
            "transformer_layer_flash_prefix_prefix_cast_norm_wall_s": 0.0,
            "transformer_layer_flash_prefix_tail_cast_norm_wall_s": 0.0,
            "transformer_layer_flash_prefix_merge_logaddexp_wall_s": 0.0,
            "transformer_layer_flash_prefix_merge_scale_wall_s": 0.0,
            "transformer_layer_flash_prefix_merge_blend_wall_s": 0.0,
            "transformer_layer_flash_prefix_prefix_saved_nonparam_unique_storage_bytes_last": 0,
            "transformer_layer_flash_prefix_tail_saved_nonparam_unique_storage_bytes_last": 0,
            "transformer_layer_flash_prefix_merge_saved_nonparam_unique_storage_bytes_last": 0,
            "transformer_layer_flash_prefix_prefix_saved_q_nonparam_unique_storage_bytes_last": 0,
            "transformer_layer_flash_prefix_prefix_saved_k_nonparam_unique_storage_bytes_last": 0,
            "transformer_layer_flash_prefix_prefix_saved_v_nonparam_unique_storage_bytes_last": 0,
            "transformer_layer_flash_prefix_prefix_saved_other_nonparam_unique_storage_bytes_last": 0,
            "transformer_layer_flash_prefix_tail_saved_q_nonparam_unique_storage_bytes_last": 0,
            "transformer_layer_flash_prefix_tail_saved_k_nonparam_unique_storage_bytes_last": 0,
            "transformer_layer_flash_prefix_tail_saved_v_nonparam_unique_storage_bytes_last": 0,
            "transformer_layer_flash_prefix_tail_saved_other_nonparam_unique_storage_bytes_last": 0,
            "transformer_layer_flash_merge_chunk_prep_wall_s": 0.0,
            "transformer_layer_flash_merge_chunk_core_wall_s": 0.0,
            "transformer_layer_flash_merge_lse_merge_wall_s": 0.0,
            "transformer_layer_flash_merge_chunk_count_sum": 0,
            "transformer_layer_finalize_attn_outproj_wall_s": 0.0,
            "transformer_layer_finalize_attn_outproj_linear_wall_s": 0.0,
            "transformer_layer_finalize_attn_outproj_norm_wall_s": 0.0,
            "transformer_layer_finalize_ffn_wall_s": 0.0,
            "transformer_layer_finalize_ffn_linear1_act_wall_s": 0.0,
            "transformer_layer_finalize_ffn_linear2_residual_norm_wall_s": 0.0,
            "transformer_layer_finalize_ffn_linear2_wall_s": 0.0,
            "transformer_layer_finalize_ffn_residual_norm_wall_s": 0.0,
            "transformer_layer_finalize_compiled_wall_s": 0.0,
            "transformer_layer_paged_path_single_page": 0,
            "transformer_layer_paged_path_flash_prefix": 0,
            "transformer_layer_paged_path_flash_prefix_zero_fastpath": 0,
            "transformer_layer_paged_path_flash_merge": 0,
            "transformer_layer_paged_path_dense": 0,
            "transformer_layer_paged_page_count_sum": 0,
            "transformer_layer_paged_valid_len_sum": 0,
            "transformer_layer_paged_last_page_tokens_sum": 0,
            "transformer_layer_paged_prefix_len_sum": 0,
            "transformer_layer_flash_prefix_valid_tokens_sum": 0,
            "transformer_layer_flash_prefix_prefix_tokens_sum": 0,
            "transformer_layer_flash_prefix_tail_tokens_sum": 0,
            "transformer_layer_dense_valid_tokens_sum": 0,
            "transformer_layer_dense_prefix_tokens_sum": 0,
            "transformer_layer_dense_tail_tokens_sum": 0,
            "transformer_layer_total_wall_s": 0.0,
            "perfeature_layer_calls": 0,
            "perfeature_feature_wall_s": 0.0,
            "perfeature_item_wall_s": 0.0,
            "perfeature_feature_saved_nonparam_raw_bytes_sum": 0,
            "perfeature_feature_saved_nonparam_raw_bytes_last": 0,
            "perfeature_feature_saved_nonparam_unique_storage_bytes_sum": 0,
            "perfeature_feature_saved_nonparam_unique_storage_bytes_last": 0,
            "perfeature_item_saved_nonparam_raw_bytes_sum": 0,
            "perfeature_item_saved_nonparam_raw_bytes_last": 0,
            "perfeature_item_saved_nonparam_unique_storage_bytes_sum": 0,
            "perfeature_item_saved_nonparam_unique_storage_bytes_last": 0,
            "perfeature_item_chunk_calls": 0,
            "perfeature_item_chunk_wall_s": 0.0,
            "perfeature_item_chunk_batch_sum": 0,
            "perfeature_item_chunk_batch_max": 0,
            "perfeature_item_chunk_saved_nonparam_raw_bytes_sum": 0,
            "perfeature_item_chunk_saved_nonparam_raw_bytes_last": 0,
            "perfeature_item_chunk_saved_nonparam_unique_storage_bytes_sum": 0,
            "perfeature_item_chunk_saved_nonparam_unique_storage_bytes_last": 0,
            "perfeature_item_chunk_saved_nonparam_unique_storage_bytes_max": 0,
            "perfeature_item_chunk_cache_unique_storage_bytes_last": 0,
            "perfeature_item_chunk_cache_unique_storage_bytes_max": 0,
        }

    def init_weights(self):
        if self.init_method is not None:
            self.apply(get_init_method(self.init_method))
        if not self.tabpfn_zero_weights:
            return
        for layer in self.transformer_encoder.layers:
            for block in (layer.feature_block, layer.item_block):
                nn.init.zeros_(block.linear2.weight)
                nn.init.zeros_(block.linear2.bias)
                attns = block.self_attn if isinstance(block.self_attn, nn.ModuleList) else [block.self_attn]
                for attn in attns:
                    nn.init.zeros_(attn.out_proj.weight)
                    nn.init.zeros_(attn.out_proj.bias)

    def consume_policy_step_profile(self):
        if not bool(self._policy_step_profile_enabled):
            return None
        stats = dict(self._policy_step_profile_stats)
        self._policy_step_profile_stats = self._new_policy_step_profile_stats()
        return stats

    def policy_fastpath_compile_active(self):
        return False

    def get_policy_fastpath_compile_config(self):
        return {"finalize_torch_compile": False}

    def warmup_policy_fastpaths(self, batch_size: int):
        del batch_size
        return False

    def _apply_input_ln(self, x):
        if self.input_ln is None:
            return x
        return self.input_ln(x)

    def _encode_xy(self, src):
        if len(src) == 3:
            _, x_src, y_src = src
        else:
            x_src, y_src = src
        x_enc = self.encoder(x_src)
        y_enc = self.y_encoder(y_src.unsqueeze(-1) if len(y_src.shape) < 3 else y_src)
        return x_enc, y_enc

    @staticmethod
    def _pool_groups(hidden):
        return hidden.mean(dim=-2)

    def _forward_queries_with_kv_from_encoded(self, train_tokens, query_tokens):
        hidden_q, _ = self.transformer_encoder.forward_with_prefix_cache(train_tokens, query_tokens)
        return hidden_q

    def forward(self, src, single_eval_pos=None):
        assert isinstance(src, tuple), "inputs (src) have to be given as (x,y) or (style,x,y) tuple"
        if single_eval_pos is None:
            raise ValueError("single_eval_pos has to be given, instead of None.")
        x_src, y_src = self._encode_xy(src)
        train_x = x_src[:single_eval_pos] + y_src[:single_eval_pos].unsqueeze(-2)
        query_x = x_src[single_eval_pos:]
        train_x = self._apply_input_ln(train_x)
        query_x = self._apply_input_ln(query_x)
        if self.single_eval_causal:
            hidden_q = self._forward_queries_with_kv_from_encoded(train_x, query_x)
            return self.decoder(self._pool_groups(hidden_q))
        src_full = torch.cat([train_x, query_x], dim=0)
        output = self.transformer_encoder(src_full, single_eval_pos)
        output = self.decoder(self._pool_groups(output))
        return output[single_eval_pos:]

    def init_kv_cache(self, x_train, y_train):
        if not self.single_eval_causal:
            raise ValueError("KV cache requires single_eval_causal=True.")
        x_enc = self.encoder(x_train)
        y_enc = self.y_encoder(y_train.unsqueeze(-1) if len(y_train.shape) < 3 else y_train)
        train_tokens = self._apply_input_ln(x_enc + y_enc.unsqueeze(-2))
        kv_cache, _ = self.transformer_encoder.encode_prefix_to_kv(train_tokens)
        return kv_cache

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
        if not self.single_eval_causal:
            raise ValueError("KV cache requires single_eval_causal=True.")
        x_enc = self.encoder(x_token)
        y_enc = self.y_encoder(y_token.unsqueeze(-1) if len(y_token.shape) < 3 else y_token)
        token = self._apply_input_ln(x_enc + y_enc.unsqueeze(-2))
        _, kv_cache = self.transformer_encoder.forward_step(
            token,
            kv_cache=kv_cache,
            append_to_cache=True,
            max_cache_len=max_cache_len,
            kv_cache_mode=kv_cache_mode,
            kv_cache_page_size=kv_cache_page_size,
            allow_grad_mutable_cache=allow_grad_mutable_cache,
            allow_grad_inplace_paged_cache=allow_grad_inplace_paged_cache,
        )
        return kv_cache

    def predict_query_with_kv(self, x_query, kv_cache):
        if not self.single_eval_causal:
            raise ValueError("KV cache requires single_eval_causal=True.")
        x_enc = self._apply_input_ln(self.encoder(x_query))
        hidden = self.transformer_encoder.forward_query(x_enc, kv_cache)
        return self.decoder(self._pool_groups(hidden))

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
        if not self.single_eval_causal:
            raise ValueError("forward_policy_step requires single_eval_causal=True.")
        if x_token.ndim != 3 or int(x_token.shape[0]) != 1:
            raise ValueError(f"x_token must have shape (1, B, F), got {tuple(x_token.shape)}")
        total_t0 = time.perf_counter() if self._policy_step_profile_enabled else None
        encode_t0 = time.perf_counter() if self._policy_step_profile_enabled else None
        x_enc = self.encoder(x_token)
        y_enc = self.y_encoder(y_token.unsqueeze(-1) if len(y_token.shape) < 3 else y_token)
        token = self._apply_input_ln(x_enc + y_enc.unsqueeze(-2))
        if self._policy_step_profile_enabled:
            self._policy_step_profile_stats["encode_wall_s"] = float(
                self._policy_step_profile_stats.get("encode_wall_s", 0.0) or 0.0
            ) + float(time.perf_counter() - encode_t0)
        transformer_t0 = time.perf_counter() if self._policy_step_profile_enabled else None
        hidden, kv_cache = self.transformer_encoder.forward_step(
            token,
            kv_cache=kv_cache,
            append_to_cache=True,
            max_cache_len=max_cache_len,
            kv_cache_mode=kv_cache_mode,
            kv_cache_page_size=kv_cache_page_size,
            allow_grad_mutable_cache=allow_grad_mutable_cache,
            allow_grad_inplace_paged_cache=allow_grad_inplace_paged_cache,
        )
        step_profile = self.transformer_encoder.consume_step_profile() if self._policy_step_profile_enabled else None
        if self._policy_step_profile_enabled:
            self._policy_step_profile_stats["transformer_wall_s"] = float(
                self._policy_step_profile_stats.get("transformer_wall_s", 0.0) or 0.0
            ) + float(time.perf_counter() - transformer_t0)
            if isinstance(step_profile, dict):
                for key, value in step_profile.items():
                    if key == "calls":
                        self._policy_step_profile_stats["transformer_layer_calls"] = int(
                            self._policy_step_profile_stats.get("transformer_layer_calls", 0) or 0
                        ) + int(value or 0)
                    elif key.startswith("perfeature_"):
                        if key.endswith("_wall_s"):
                            self._policy_step_profile_stats[key] = float(
                                self._policy_step_profile_stats.get(key, 0.0) or 0.0
                            ) + float(value or 0.0)
                        elif key.endswith("_last"):
                            self._policy_step_profile_stats[key] = int(value or 0)
                        else:
                            self._policy_step_profile_stats[key] = int(
                                self._policy_step_profile_stats.get(key, 0) or 0
                            ) + int(value or 0)
                    elif key.endswith("_last"):
                        self._policy_step_profile_stats[f"transformer_layer_{key}"] = int(value or 0)
                    elif key.endswith("_wall_s"):
                        self._policy_step_profile_stats[f"transformer_layer_{key}"] = float(
                            self._policy_step_profile_stats.get(f"transformer_layer_{key}", 0.0) or 0.0
                        ) + float(value or 0.0)
                    elif key.startswith("paged_") or key.startswith("flash_prefix_") or key.startswith("dense_"):
                        self._policy_step_profile_stats[f"transformer_layer_{key}"] = int(
                            self._policy_step_profile_stats.get(f"transformer_layer_{key}", 0) or 0
                        ) + int(value or 0)
                    else:
                        self._policy_step_profile_stats[f"transformer_layer_{key}"] = int(
                            self._policy_step_profile_stats.get(f"transformer_layer_{key}", 0) or 0
                        ) + int(value or 0)
        decoder_t0 = time.perf_counter() if self._policy_step_profile_enabled else None
        out = self.decoder(self._pool_groups(hidden))
        if self._policy_step_profile_enabled:
            self._policy_step_profile_stats["calls"] = int(self._policy_step_profile_stats.get("calls", 0) or 0) + 1
            self._policy_step_profile_stats["decoder_wall_s"] = float(
                self._policy_step_profile_stats.get("decoder_wall_s", 0.0) or 0.0
            ) + float(time.perf_counter() - decoder_t0)
            self._policy_step_profile_stats["total_wall_s"] = float(
                self._policy_step_profile_stats.get("total_wall_s", 0.0) or 0.0
            ) + float(time.perf_counter() - total_t0)
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
        if not self.single_eval_causal:
            raise ValueError("forward_policy_step_split requires single_eval_causal=True.")
        if self.x_encoder_type != "split_obs_action":
            raise ValueError("forward_policy_step_split requires split_obs_action encoder.")
        batch_size = int(obs_t.shape[0])
        if int(action_t.shape[0]) != batch_size:
            raise ValueError("obs_t and action_t batch size mismatch")
        obs_dim = int(self.encoder.obs_dim)
        action_dim = int(self.encoder.action_dim)

        reward_scalar = reward_t.reshape(batch_size, 1).to(dtype=obs_t.dtype, device=obs_t.device)
        reward_mask_scalar = reward_mask_t.reshape(batch_size, 1).to(dtype=obs_t.dtype, device=obs_t.device)
        phase_scalar = (
            phase_t.reshape(batch_size, 1).to(dtype=obs_t.dtype, device=obs_t.device)
            if phase_t is not None
            else None
        )
        terminal_scalar = (
            terminal_t.reshape(batch_size, 1).to(dtype=obs_t.dtype, device=obs_t.device)
            if terminal_t is not None
            else None
        )

        extra_scalar_slots = int(max(0, obs_dim - int(obs_t.shape[-1]) - 2))
        phase_slot_expected = int(phase_scalar is not None)
        terminal_slot_expected = int(terminal_scalar is not None)
        remaining_slots = int(max(0, extra_scalar_slots - phase_slot_expected - terminal_slot_expected))
        if remaining_slots > 0 and phase_scalar is None:
            phase_scalar = torch.zeros((batch_size, 1), device=obs_t.device, dtype=obs_t.dtype)
            phase_slot_expected = 1
            remaining_slots -= 1
        if remaining_slots > 0 and terminal_scalar is None:
            terminal_scalar = torch.zeros((batch_size, 1), device=obs_t.device, dtype=obs_t.dtype)
            terminal_slot_expected = 1
            remaining_slots -= 1

        scalar_slots = 2 + phase_slot_expected + terminal_slot_expected
        obs_slot_dim = int(max(0, obs_dim - scalar_slots))
        reward_idx = obs_slot_dim
        mask_idx = obs_slot_dim + 1
        phase_idx = obs_slot_dim + 2 if phase_slot_expected else None
        terminal_idx = obs_slot_dim + 2 + phase_slot_expected if terminal_slot_expected else None
        obs_copy = int(min(int(obs_t.shape[-1]), obs_slot_dim))
        action_copy = int(min(int(action_t.shape[-1]), action_dim))

        x_token = torch.zeros((1, batch_size, obs_dim + action_dim), device=obs_t.device, dtype=obs_t.dtype)
        if obs_copy > 0:
            x_token[0, :, :obs_copy] = obs_t[:, :obs_copy]
        if action_copy > 0:
            x_token[0, :, obs_dim : obs_dim + action_copy] = action_t[:, :action_copy]
        x_token[0, :, reward_idx] = reward_scalar.reshape(-1)
        x_token[0, :, mask_idx] = reward_mask_scalar.reshape(-1)
        if phase_idx is not None and phase_scalar is not None:
            x_token[0, :, phase_idx] = phase_scalar.reshape(-1)
        if terminal_idx is not None and terminal_scalar is not None:
            x_token[0, :, terminal_idx] = terminal_scalar.reshape(-1)
        y_token = reward_scalar.reshape(1, batch_size)
        return self.forward_policy_step(
            x_token,
            y_token,
            kv_cache=kv_cache,
            max_cache_len=max_cache_len,
            kv_cache_mode=kv_cache_mode,
            kv_cache_page_size=kv_cache_page_size,
            allow_grad_mutable_cache=allow_grad_mutable_cache,
            allow_grad_inplace_paged_cache=allow_grad_inplace_paged_cache,
        )

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
