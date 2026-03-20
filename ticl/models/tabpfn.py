import os
import time
from collections import OrderedDict

import torch, wandb
import torch.nn as nn
import torch.nn.functional as F

from ticl.models.layer import TransformerEncoderLayer, TransformerEncoderSimple
from ticl.utils import SeqBN, get_init_method
from ticl.models.encoders import Linear, SplitObsActionEncoder


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


class TabPFN(nn.Module):
    def __init__(self, *, n_out, emsize, nhead, nhid_factor, nlayers, n_features, dropout=0.0,  y_encoder_layer=None,
                 decoder=None, input_normalization=False, init_method=None, pre_norm=False,
                 activation='gelu', recompute_attn=False, classification_task=True,
                 all_layers_same_init=False, efficient_eval_masking=True, y_encoder=None, tabpfn_zero_weights=False,
                 x_encoder_type='single', x_obs_dim=None, x_action_dim=None, single_eval_causal=False):
        super().__init__()
        self.classification_task = classification_task
        self.y_encoder = y_encoder_layer
        nhid = emsize * nhid_factor

        def encoder_layer_creator(): return TransformerEncoderLayer(
            emsize, 
            nhead, 
            nhid, 
                dropout, 
                activation=activation,
                pre_norm=pre_norm, 
                recompute_attn=recompute_attn,
                single_eval_causal=single_eval_causal,
            )
        self.transformer_encoder =  TransformerEncoderSimple(encoder_layer_creator, nlayers)
        backbone_size = sum(p.numel() for p in self.transformer_encoder.parameters())
        if wandb.run: wandb.log({"backbone_size": backbone_size})
        print("Number of parameters in backbone: ", backbone_size)

        self.emsize = emsize
        self.x_encoder_type = x_encoder_type
        if self.x_encoder_type == 'single':
            self.encoder = Linear(n_features, emsize, replace_nan_by_zero=True)
        elif self.x_encoder_type == 'split_obs_action':
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
                emsize=emsize,
                replace_nan_by_zero=True,
            )
        else:
            raise ValueError(f"Unknown x_encoder_type: {self.x_encoder_type}")
        self.decoder = decoder(emsize, nhid, n_out) if decoder is not None else nn.Sequential(nn.Linear(emsize, nhid), nn.GELU(), nn.Linear(nhid, n_out))
        self.policy_action_dim = int(x_action_dim) if x_action_dim is not None else None
        self.policy_action_head = None
        if self.policy_action_dim is not None and self.policy_action_dim > 0:
            self.policy_action_head = (
                decoder(emsize, nhid, self.policy_action_dim)
                if decoder is not None
                else nn.Sequential(nn.Linear(emsize, nhid), nn.GELU(), nn.Linear(nhid, self.policy_action_dim))
            )
        self.input_ln = SeqBN(emsize) if input_normalization else None
        self.init_method = init_method
        self.efficient_eval_masking = efficient_eval_masking
        self.tabpfn_zero_weights = tabpfn_zero_weights
        self.single_eval_causal = bool(single_eval_causal)
        self.n_out = n_out
        self.nhid = nhid
        profile_flag = str(os.environ.get("TICL_POLICY_STEP_PROFILE", "")).strip().lower()
        self._policy_step_profile_enabled = profile_flag in {"1", "true", "yes", "on"}
        self._policy_step_profile_stats = {
            "calls": 0,
            "encode_wall_s": 0.0,
            "transformer_wall_s": 0.0,
            "decoder_wall_s": 0.0,
            "total_wall_s": 0.0,
            "transformer_layer_calls": 0,
            "transformer_layer_proj_wall_s": 0.0,
            "transformer_layer_cache_wall_s": 0.0,
            "transformer_layer_attnff_wall_s": 0.0,
            "transformer_layer_attn_core_wall_s": 0.0,
            "transformer_layer_finalize_wall_s": 0.0,
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
        }
        self.init_weights()

    @staticmethod
    def _repeat_output_rows(tensor, out_rows):
        if tensor.ndim < 1:
            raise ValueError("Expected tensor with at least one dimension.")
        current_rows = int(tensor.shape[0])
        target_rows = int(out_rows)
        if current_rows == target_rows:
            return tensor.clone()
        if current_rows <= 0:
            raise ValueError("Cannot expand empty tensor rows.")
        repeat_rows = (target_rows + current_rows - 1) // current_rows
        repeat_shape = [repeat_rows] + [1] * (tensor.ndim - 1)
        return tensor.repeat(*repeat_shape)[:target_rows].clone()

    @staticmethod
    def _is_default_mlp_head(head):
        return (
            isinstance(head, nn.Sequential)
            and len(head) == 3
            and isinstance(head[0], nn.Linear)
            and isinstance(head[1], nn.GELU)
            and isinstance(head[2], nn.Linear)
        )

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
        if self.policy_action_head_required():
            self.require_policy_action_head()
            return self.policy_action_head(hidden)
        return self.decoder(hidden)

    def reset_policy_action_head_from_decoder_(self):
        if not self.has_policy_action_head():
            return False
        if not self._is_default_mlp_head(self.decoder):
            return False
        if not self._is_default_mlp_head(self.policy_action_head):
            return False
        src_first = self.decoder[0]
        dst_first = self.policy_action_head[0]
        src_last = self.decoder[2]
        dst_last = self.policy_action_head[2]
        if src_first.weight.shape != dst_first.weight.shape:
            return False
        if src_first.bias.shape != dst_first.bias.shape:
            return False
        if src_last.weight.shape[1] != dst_last.weight.shape[1]:
            return False
        with torch.no_grad():
            dst_first.weight.copy_(src_first.weight)
            dst_first.bias.copy_(src_first.bias)
            dst_last.weight.copy_(self._repeat_output_rows(src_last.weight, dst_last.weight.shape[0]))
            dst_last.bias.copy_(self._repeat_output_rows(src_last.bias, dst_last.bias.shape[0]))
        return True

    def _upgrade_legacy_policy_action_head_state_dict(self, state_dict):
        if not self.has_policy_action_head():
            return state_dict
        required_state = self.policy_action_head.state_dict()
        missing_keys = [
            f"policy_action_head.{key}"
            for key in required_state.keys()
            if f"policy_action_head.{key}" not in state_dict
        ]
        if not missing_keys:
            return state_dict

        upgraded_state = OrderedDict(state_dict)
        metadata = getattr(state_dict, "_metadata", None)
        if metadata is not None:
            upgraded_state._metadata = metadata

        can_bootstrap = (
            self._is_default_mlp_head(self.decoder)
            and self._is_default_mlp_head(self.policy_action_head)
            and "decoder.0.weight" in upgraded_state
            and "decoder.0.bias" in upgraded_state
            and "decoder.2.weight" in upgraded_state
            and "decoder.2.bias" in upgraded_state
        )
        if can_bootstrap:
            src_first_weight = upgraded_state["decoder.0.weight"]
            src_first_bias = upgraded_state["decoder.0.bias"]
            src_last_weight = upgraded_state["decoder.2.weight"]
            src_last_bias = upgraded_state["decoder.2.bias"]
            dst_first = self.policy_action_head[0]
            dst_last = self.policy_action_head[2]
            if (
                src_first_weight.shape == dst_first.weight.shape
                and src_first_bias.shape == dst_first.bias.shape
                and src_last_weight.shape[1] == dst_last.weight.shape[1]
            ):
                upgraded_state["policy_action_head.0.weight"] = src_first_weight.clone()
                upgraded_state["policy_action_head.0.bias"] = src_first_bias.clone()
                upgraded_state["policy_action_head.2.weight"] = self._repeat_output_rows(
                    src_last_weight,
                    dst_last.weight.shape[0],
                )
                upgraded_state["policy_action_head.2.bias"] = self._repeat_output_rows(
                    src_last_bias,
                    dst_last.bias.shape[0],
                )

        # Fall back to current initialization only for the new action head keys.
        current_policy_state = self.policy_action_head.state_dict()
        for key, value in current_policy_state.items():
            full_key = f"policy_action_head.{key}"
            if full_key not in upgraded_state:
                upgraded_state[full_key] = value.detach().clone()
        return upgraded_state

    def load_state_dict(self, state_dict, strict=True, assign=False):
        upgraded_state = self._upgrade_legacy_policy_action_head_state_dict(state_dict)
        return super().load_state_dict(upgraded_state, strict=strict, assign=assign)

    def consume_policy_step_profile(self):
        if not bool(self._policy_step_profile_enabled):
            return None
        stats = dict(self._policy_step_profile_stats)
        self._policy_step_profile_stats = {
            "calls": 0,
            "encode_wall_s": 0.0,
            "transformer_wall_s": 0.0,
            "decoder_wall_s": 0.0,
            "total_wall_s": 0.0,
            "transformer_layer_calls": 0,
            "transformer_layer_proj_wall_s": 0.0,
            "transformer_layer_cache_wall_s": 0.0,
            "transformer_layer_attnff_wall_s": 0.0,
            "transformer_layer_attn_core_wall_s": 0.0,
            "transformer_layer_finalize_wall_s": 0.0,
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
        }
        return stats

    def _encode_xy(self, src):
        if len(src) == 3:  # style is given
            _, x_src, y_src = src
        else:
            x_src, y_src = src
        x_enc = self.encoder(x_src)
        y_enc = self.y_encoder(y_src.unsqueeze(-1) if len(y_src.shape) < len(x_enc.shape) else y_src)
        return x_enc, y_enc

    def policy_fastpath_compile_active(self):
        fn = getattr(self.transformer_encoder, "step_fastpath_compile_active", None)
        return bool(fn()) if callable(fn) else False

    def get_policy_fastpath_compile_config(self):
        fn = getattr(self.transformer_encoder, "step_fastpath_compile_config", None)
        if callable(fn):
            cfg = fn()
            if isinstance(cfg, dict):
                return cfg
        return {"finalize_torch_compile": False}

    def warmup_policy_fastpaths(self, batch_size: int):
        warmup_fn = getattr(self.transformer_encoder, "warmup_step_fastpaths", None)
        warmed = bool(warmup_fn(batch_size=batch_size)) if callable(warmup_fn) else False
        if warmed:
            self.zero_grad(set_to_none=True)
        return warmed

    def init_weights(self):
        if self.init_method is not None:
            self.apply(get_init_method(self.init_method))
        if self.tabpfn_zero_weights:
            for layer in self.transformer_encoder.layers:
                nn.init.zeros_(layer.linear2.weight)
                nn.init.zeros_(layer.linear2.bias)
                attns = layer.self_attn if isinstance(layer.self_attn, nn.ModuleList) else [layer.self_attn]
                for attn in attns:
                    nn.init.zeros_(attn.out_proj.weight)
                    nn.init.zeros_(attn.out_proj.bias)

    def _forward_queries_with_kv_from_encoded(self, train_tokens, query_tokens):
        hidden_q, _ = self.transformer_encoder.forward_with_prefix_cache(train_tokens, query_tokens)
        return hidden_q

    def forward(self, src, single_eval_pos=None):
        assert isinstance(src, tuple), 'inputs (src) have to be given as (x,y) or (style,x,y) tuple'
        if single_eval_pos is None: raise ValueError('single_eval_pos has to be given, instead of None.')

        # x_src/y_src: (num_samples, batch_size, d_model)
        x_src, y_src = self._encode_xy(src)

        if self.efficient_eval_masking:
            src_mask = single_eval_pos
        else:
            raise NotImplementedError(f'efficient_eval_masking={self.efficient_eval_masking} is not implemented yet.')

        train_x = x_src[:single_eval_pos] + y_src[:single_eval_pos]
        query_x = x_src[single_eval_pos:]
        if self.input_ln is not None:
            train_x = self.input_ln(train_x)
            query_x = self.input_ln(query_x)

        if self.single_eval_causal:
            hidden_q = self._forward_queries_with_kv_from_encoded(train_x, query_x)
            return self._decode_policy_action(hidden_q)

        src = torch.cat([train_x, query_x], 0)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)  # decoder is position-wise
        return output[single_eval_pos:]

    def init_kv_cache(self, x_train, y_train):
        """
        Build KV cache from training prefix tokens (single-directional mode only).
        Shapes:
          x_train: (T_train, B, F)
          y_train: (T_train, B) or (T_train, B, 1)
        """
        if not self.single_eval_causal:
            raise ValueError("KV cache requires single_eval_causal=True.")
        x_enc = self.encoder(x_train)
        y_enc = self.y_encoder(y_train.unsqueeze(-1) if len(y_train.shape) < len(x_enc.shape) else y_train)
        train_tokens = x_enc + y_enc
        if self.input_ln is not None:
            train_tokens = self.input_ln(train_tokens)
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
        """
        Append a single realized training token to existing KV cache.
        x_token shape: (1, B, F)
        y_token shape: (1, B) or (1, B, 1)
        """
        if not self.single_eval_causal:
            raise ValueError("KV cache requires single_eval_causal=True.")
        x_enc = self.encoder(x_token)
        y_enc = self.y_encoder(y_token.unsqueeze(-1) if len(y_token.shape) < len(x_enc.shape) else y_token)
        token = x_enc + y_enc
        if self.input_ln is not None:
            token = self.input_ln(token)
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
        """
        Predict one or more query tokens against existing train-prefix KV cache.
        x_query shape: (T_query, B, F)
        """
        if not self.single_eval_causal:
            raise ValueError("KV cache requires single_eval_causal=True.")
        x_enc = self.encoder(x_query)
        if self.input_ln is not None:
            x_enc = self.input_ln(x_enc)
        hidden = self.transformer_encoder.forward_query(x_enc, kv_cache)
        return self._decode_policy_action(hidden)

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
        """
        Incremental single-step forward for autoregressive policy rollout.
        Shapes:
          x_token: (1, B, F)
          y_token: (1, B) or (1, B, 1)
        """
        if not self.single_eval_causal:
            raise ValueError("forward_policy_step requires single_eval_causal=True.")
        if x_token.ndim != 3 or x_token.shape[0] != 1:
            raise ValueError(f"x_token must have shape (1, B, F), got {tuple(x_token.shape)}")

        profile_enabled = bool(self._policy_step_profile_enabled) and (not _is_torch_compiling())
        total_t0 = time.perf_counter() if profile_enabled else None
        encode_t0 = time.perf_counter() if profile_enabled else None
        x_enc = self.encoder(x_token)
        y_enc = self.y_encoder(y_token.unsqueeze(-1) if len(y_token.shape) < len(x_enc.shape) else y_token)
        token = x_enc + y_enc
        if self.input_ln is not None:
            token = self.input_ln(token)
        encode_dt = (time.perf_counter() - encode_t0) if encode_t0 is not None else 0.0
        transformer_t0 = time.perf_counter() if profile_enabled else None
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
        transformer_dt = (time.perf_counter() - transformer_t0) if transformer_t0 is not None else 0.0
        transformer_layer_profile = None
        if profile_enabled:
            consume_tf_profile = getattr(self.transformer_encoder, "consume_step_profile", None)
            if callable(consume_tf_profile):
                transformer_layer_profile = consume_tf_profile()
        decoder_t0 = time.perf_counter() if profile_enabled else None
        out = self._decode_policy_action(hidden)
        decoder_dt = (time.perf_counter() - decoder_t0) if decoder_t0 is not None else 0.0
        if total_t0 is not None:
            stats = self._policy_step_profile_stats
            stats["calls"] += 1
            stats["encode_wall_s"] += float(encode_dt)
            stats["transformer_wall_s"] += float(transformer_dt)
            stats["decoder_wall_s"] += float(decoder_dt)
            stats["total_wall_s"] += float(time.perf_counter() - total_t0)
            if isinstance(transformer_layer_profile, dict):
                stats["transformer_layer_calls"] += int(transformer_layer_profile.get("calls", 0) or 0)
                stats["transformer_layer_proj_wall_s"] += float(transformer_layer_profile.get("proj_wall_s", 0.0) or 0.0)
                stats["transformer_layer_cache_wall_s"] += float(transformer_layer_profile.get("cache_wall_s", 0.0) or 0.0)
                stats["transformer_layer_attnff_wall_s"] += float(transformer_layer_profile.get("attnff_wall_s", 0.0) or 0.0)
                stats["transformer_layer_attn_core_wall_s"] += float(
                    transformer_layer_profile.get("attn_core_wall_s", 0.0) or 0.0
                )
                stats["transformer_layer_finalize_wall_s"] += float(
                    transformer_layer_profile.get("finalize_wall_s", 0.0) or 0.0
                )
                stats["transformer_layer_finalize_attn_outproj_wall_s"] += float(
                    transformer_layer_profile.get("finalize_attn_outproj_wall_s", 0.0) or 0.0
                )
                stats["transformer_layer_finalize_attn_outproj_linear_wall_s"] += float(
                    transformer_layer_profile.get("finalize_attn_outproj_linear_wall_s", 0.0) or 0.0
                )
                stats["transformer_layer_finalize_attn_outproj_norm_wall_s"] += float(
                    transformer_layer_profile.get("finalize_attn_outproj_norm_wall_s", 0.0) or 0.0
                )
                stats["transformer_layer_finalize_ffn_wall_s"] += float(
                    transformer_layer_profile.get("finalize_ffn_wall_s", 0.0) or 0.0
                )
                stats["transformer_layer_finalize_ffn_linear1_act_wall_s"] += float(
                    transformer_layer_profile.get("finalize_ffn_linear1_act_wall_s", 0.0) or 0.0
                )
                stats["transformer_layer_finalize_ffn_linear2_residual_norm_wall_s"] += float(
                    transformer_layer_profile.get("finalize_ffn_linear2_residual_norm_wall_s", 0.0) or 0.0
                )
                stats["transformer_layer_finalize_ffn_linear2_wall_s"] += float(
                    transformer_layer_profile.get("finalize_ffn_linear2_wall_s", 0.0) or 0.0
                )
                stats["transformer_layer_finalize_ffn_residual_norm_wall_s"] += float(
                    transformer_layer_profile.get("finalize_ffn_residual_norm_wall_s", 0.0) or 0.0
                )
                stats["transformer_layer_finalize_compiled_wall_s"] += float(
                    transformer_layer_profile.get("finalize_compiled_wall_s", 0.0) or 0.0
                )
                stats["transformer_layer_paged_path_single_page"] += int(
                    transformer_layer_profile.get("paged_path_single_page", 0) or 0
                )
                stats["transformer_layer_paged_path_flash_prefix"] += int(
                    transformer_layer_profile.get("paged_path_flash_prefix", 0) or 0
                )
                stats["transformer_layer_paged_path_flash_prefix_zero_fastpath"] += int(
                    transformer_layer_profile.get("paged_path_flash_prefix_zero_fastpath", 0) or 0
                )
                stats["transformer_layer_paged_path_flash_merge"] += int(
                    transformer_layer_profile.get("paged_path_flash_merge", 0) or 0
                )
                stats["transformer_layer_paged_path_dense"] += int(
                    transformer_layer_profile.get("paged_path_dense", 0) or 0
                )
                stats["transformer_layer_paged_page_count_sum"] += int(
                    transformer_layer_profile.get("paged_page_count_sum", 0) or 0
                )
                stats["transformer_layer_paged_valid_len_sum"] += int(
                    transformer_layer_profile.get("paged_valid_len_sum", 0) or 0
                )
                stats["transformer_layer_paged_last_page_tokens_sum"] += int(
                    transformer_layer_profile.get("paged_last_page_tokens_sum", 0) or 0
                )
                stats["transformer_layer_paged_prefix_len_sum"] += int(
                    transformer_layer_profile.get("paged_prefix_len_sum", 0) or 0
                )
                stats["transformer_layer_flash_prefix_valid_tokens_sum"] += int(
                    transformer_layer_profile.get("flash_prefix_valid_tokens_sum", 0) or 0
                )
                stats["transformer_layer_flash_prefix_prefix_tokens_sum"] += int(
                    transformer_layer_profile.get("flash_prefix_prefix_tokens_sum", 0) or 0
                )
                stats["transformer_layer_flash_prefix_tail_tokens_sum"] += int(
                    transformer_layer_profile.get("flash_prefix_tail_tokens_sum", 0) or 0
                )
                stats["transformer_layer_dense_valid_tokens_sum"] += int(
                    transformer_layer_profile.get("dense_valid_tokens_sum", 0) or 0
                )
                stats["transformer_layer_dense_prefix_tokens_sum"] += int(
                    transformer_layer_profile.get("dense_prefix_tokens_sum", 0) or 0
                )
                stats["transformer_layer_dense_tail_tokens_sum"] += int(
                    transformer_layer_profile.get("dense_tail_tokens_sum", 0) or 0
                )
                stats["transformer_layer_total_wall_s"] += float(transformer_layer_profile.get("total_wall_s", 0.0) or 0.0)
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
        """
        Incremental single-step forward optimized for split_obs_action layout.
        Inputs are per-step components (B, D) instead of a materialized (1, B, F) token.
        """
        if not self.single_eval_causal:
            raise ValueError("forward_policy_step_split requires single_eval_causal=True.")
        if self.x_encoder_type != "split_obs_action" or (not isinstance(self.encoder, SplitObsActionEncoder)):
            raise ValueError("forward_policy_step_split requires split_obs_action encoder.")
        if obs_t.ndim != 2 or action_t.ndim != 2:
            raise ValueError(
                f"obs_t/action_t must have shape (B, D), got {tuple(obs_t.shape)} / {tuple(action_t.shape)}"
            )

        profile_enabled = bool(self._policy_step_profile_enabled) and (not _is_torch_compiling())
        total_t0 = time.perf_counter() if profile_enabled else None
        encode_t0 = time.perf_counter() if profile_enabled else None
        batch_size = int(obs_t.shape[0])
        if int(action_t.shape[0]) != batch_size:
            raise ValueError("obs_t and action_t batch size mismatch")

        reward_scalar = reward_t.reshape(batch_size, 1).to(dtype=obs_t.dtype, device=obs_t.device)
        reward_mask_scalar = reward_mask_t.reshape(batch_size, 1).to(dtype=obs_t.dtype, device=obs_t.device)
        phase_scalar = None
        if phase_t is not None:
            phase_scalar = phase_t.reshape(batch_size, 1).to(dtype=obs_t.dtype, device=obs_t.device)
        terminal_scalar = None
        if terminal_t is not None:
            terminal_scalar = terminal_t.reshape(batch_size, 1).to(dtype=obs_t.dtype, device=obs_t.device)

        obs_encoder = self.encoder.obs_encoder
        action_encoder = self.encoder.action_encoder
        obs_weight = obs_encoder.weight
        obs_bias = obs_encoder.bias
        action_weight = action_encoder.weight
        action_bias = action_encoder.bias
        fuse_y_linear = isinstance(self.y_encoder, nn.Linear) and int(getattr(self.y_encoder, "in_features", 0)) == 1
        if fuse_y_linear:
            y_weight_vec = self.y_encoder.weight[:, 0]
            y_bias_vec = self.y_encoder.bias
        else:
            y_weight_vec = None
            y_bias_vec = None
        obs_dim = int(self.encoder.obs_dim)
        action_dim = int(self.encoder.action_dim)
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
        terminal_idx = (
            obs_slot_dim + 2 + phase_slot_expected
            if terminal_slot_expected
            else None
        )

        obs_copy = int(min(int(obs_t.shape[-1]), obs_slot_dim))
        action_copy = int(min(int(action_t.shape[-1]), action_dim))

        if bool(getattr(obs_encoder, "replace_nan_by_zero", False)):
            obs_src = torch.nan_to_num(obs_t, nan=0.0)
        else:
            obs_src = obs_t
        if bool(getattr(action_encoder, "replace_nan_by_zero", False)):
            action_src = torch.nan_to_num(action_t, nan=0.0)
        else:
            action_src = action_t

        if _SPLIT_ENCODE_FUSION_ENABLED:
            fused_inputs = []
            fused_weights = []

            if obs_copy > 0:
                fused_inputs.append(obs_src[:, :obs_copy])
                fused_weights.append(obs_weight[:, :obs_copy])
            if action_copy > 0:
                fused_inputs.append(action_src[:, :action_copy])
                fused_weights.append(action_weight[:, :action_copy])
            if reward_idx < obs_dim:
                reward_weight_col = obs_weight[:, reward_idx].unsqueeze(1)
                if y_weight_vec is not None:
                    reward_weight_col = reward_weight_col + y_weight_vec.unsqueeze(1)
                fused_inputs.append(reward_scalar)
                fused_weights.append(reward_weight_col)
            elif y_weight_vec is not None:
                fused_inputs.append(reward_scalar)
                fused_weights.append(y_weight_vec.unsqueeze(1))
            if mask_idx < obs_dim:
                fused_inputs.append(reward_mask_scalar)
                fused_weights.append(obs_weight[:, mask_idx].unsqueeze(1))
            if phase_idx is not None and phase_idx < obs_dim:
                fused_inputs.append(phase_scalar)
                fused_weights.append(obs_weight[:, phase_idx].unsqueeze(1))
            if terminal_idx is not None and terminal_idx < obs_dim:
                fused_inputs.append(terminal_scalar)
                fused_weights.append(obs_weight[:, terminal_idx].unsqueeze(1))

            fused_bias = obs_bias + action_bias
            if y_bias_vec is not None:
                fused_bias = fused_bias + y_bias_vec

            if fused_inputs:
                if len(fused_inputs) == 1:
                    fused_in = fused_inputs[0]
                    fused_w = fused_weights[0]
                else:
                    fused_in = torch.cat(fused_inputs, dim=1)
                    fused_w = torch.cat(fused_weights, dim=1)
                token_be = F.linear(fused_in, fused_w, fused_bias)
            else:
                token_be = fused_bias.unsqueeze(0).expand(batch_size, -1)
            token = token_be.unsqueeze(0)
        else:
            if obs_copy > 0:
                obs_enc = F.linear(obs_src[:, :obs_copy], obs_weight[:, :obs_copy], obs_bias)
            else:
                obs_enc = obs_bias.unsqueeze(0).expand(batch_size, -1)
            if reward_idx < obs_dim:
                reward_weight = obs_weight[:, reward_idx]
                if y_weight_vec is not None:
                    reward_weight = reward_weight + y_weight_vec
                obs_enc = obs_enc + reward_scalar * reward_weight.unsqueeze(0)
            elif y_weight_vec is not None:
                obs_enc = obs_enc + reward_scalar * y_weight_vec.unsqueeze(0)
            if mask_idx < obs_dim:
                obs_enc = obs_enc + reward_mask_scalar * obs_weight[:, mask_idx].unsqueeze(0)
            if phase_idx is not None and phase_idx < obs_dim:
                obs_enc = obs_enc + phase_scalar * obs_weight[:, phase_idx].unsqueeze(0)
            if terminal_idx is not None and terminal_idx < obs_dim:
                obs_enc = obs_enc + terminal_scalar * obs_weight[:, terminal_idx].unsqueeze(0)
            if y_bias_vec is not None:
                obs_enc = obs_enc + y_bias_vec.unsqueeze(0)

            if action_copy > 0:
                action_enc = F.linear(action_src[:, :action_copy], action_weight[:, :action_copy], action_bias)
            else:
                action_enc = action_bias.unsqueeze(0).expand(batch_size, -1)

            token = (obs_enc + action_enc).unsqueeze(0)
        if not fuse_y_linear:
            y_in = reward_scalar.reshape(1, batch_size)
            y_enc = self.y_encoder(y_in.unsqueeze(-1) if len(y_in.shape) < len(token.shape) else y_in)
            token = token + y_enc
        if self.input_ln is not None:
            token = self.input_ln(token)
        encode_dt = (time.perf_counter() - encode_t0) if encode_t0 is not None else 0.0
        transformer_t0 = time.perf_counter() if profile_enabled else None
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
        transformer_dt = (time.perf_counter() - transformer_t0) if transformer_t0 is not None else 0.0
        transformer_layer_profile = None
        if profile_enabled:
            consume_tf_profile = getattr(self.transformer_encoder, "consume_step_profile", None)
            if callable(consume_tf_profile):
                transformer_layer_profile = consume_tf_profile()
        decoder_t0 = time.perf_counter() if profile_enabled else None
        out = self._decode_policy_action(hidden)
        decoder_dt = (time.perf_counter() - decoder_t0) if decoder_t0 is not None else 0.0
        if total_t0 is not None:
            stats = self._policy_step_profile_stats
            stats["calls"] += 1
            stats["encode_wall_s"] += float(encode_dt)
            stats["transformer_wall_s"] += float(transformer_dt)
            stats["decoder_wall_s"] += float(decoder_dt)
            stats["total_wall_s"] += float(time.perf_counter() - total_t0)
            if isinstance(transformer_layer_profile, dict):
                stats["transformer_layer_calls"] += int(transformer_layer_profile.get("calls", 0) or 0)
                stats["transformer_layer_proj_wall_s"] += float(transformer_layer_profile.get("proj_wall_s", 0.0) or 0.0)
                stats["transformer_layer_cache_wall_s"] += float(transformer_layer_profile.get("cache_wall_s", 0.0) or 0.0)
                stats["transformer_layer_attnff_wall_s"] += float(transformer_layer_profile.get("attnff_wall_s", 0.0) or 0.0)
                stats["transformer_layer_attn_core_wall_s"] += float(
                    transformer_layer_profile.get("attn_core_wall_s", 0.0) or 0.0
                )
                stats["transformer_layer_finalize_wall_s"] += float(
                    transformer_layer_profile.get("finalize_wall_s", 0.0) or 0.0
                )
                stats["transformer_layer_finalize_attn_outproj_wall_s"] += float(
                    transformer_layer_profile.get("finalize_attn_outproj_wall_s", 0.0) or 0.0
                )
                stats["transformer_layer_finalize_attn_outproj_linear_wall_s"] += float(
                    transformer_layer_profile.get("finalize_attn_outproj_linear_wall_s", 0.0) or 0.0
                )
                stats["transformer_layer_finalize_attn_outproj_norm_wall_s"] += float(
                    transformer_layer_profile.get("finalize_attn_outproj_norm_wall_s", 0.0) or 0.0
                )
                stats["transformer_layer_finalize_ffn_wall_s"] += float(
                    transformer_layer_profile.get("finalize_ffn_wall_s", 0.0) or 0.0
                )
                stats["transformer_layer_finalize_ffn_linear1_act_wall_s"] += float(
                    transformer_layer_profile.get("finalize_ffn_linear1_act_wall_s", 0.0) or 0.0
                )
                stats["transformer_layer_finalize_ffn_linear2_residual_norm_wall_s"] += float(
                    transformer_layer_profile.get("finalize_ffn_linear2_residual_norm_wall_s", 0.0) or 0.0
                )
                stats["transformer_layer_finalize_ffn_linear2_wall_s"] += float(
                    transformer_layer_profile.get("finalize_ffn_linear2_wall_s", 0.0) or 0.0
                )
                stats["transformer_layer_finalize_ffn_residual_norm_wall_s"] += float(
                    transformer_layer_profile.get("finalize_ffn_residual_norm_wall_s", 0.0) or 0.0
                )
                stats["transformer_layer_finalize_compiled_wall_s"] += float(
                    transformer_layer_profile.get("finalize_compiled_wall_s", 0.0) or 0.0
                )
                stats["transformer_layer_paged_path_single_page"] += int(
                    transformer_layer_profile.get("paged_path_single_page", 0) or 0
                )
                stats["transformer_layer_paged_path_flash_prefix"] += int(
                    transformer_layer_profile.get("paged_path_flash_prefix", 0) or 0
                )
                stats["transformer_layer_paged_path_flash_prefix_zero_fastpath"] += int(
                    transformer_layer_profile.get("paged_path_flash_prefix_zero_fastpath", 0) or 0
                )
                stats["transformer_layer_paged_path_flash_merge"] += int(
                    transformer_layer_profile.get("paged_path_flash_merge", 0) or 0
                )
                stats["transformer_layer_paged_path_dense"] += int(
                    transformer_layer_profile.get("paged_path_dense", 0) or 0
                )
                stats["transformer_layer_paged_page_count_sum"] += int(
                    transformer_layer_profile.get("paged_page_count_sum", 0) or 0
                )
                stats["transformer_layer_paged_valid_len_sum"] += int(
                    transformer_layer_profile.get("paged_valid_len_sum", 0) or 0
                )
                stats["transformer_layer_paged_last_page_tokens_sum"] += int(
                    transformer_layer_profile.get("paged_last_page_tokens_sum", 0) or 0
                )
                stats["transformer_layer_paged_prefix_len_sum"] += int(
                    transformer_layer_profile.get("paged_prefix_len_sum", 0) or 0
                )
                stats["transformer_layer_flash_prefix_valid_tokens_sum"] += int(
                    transformer_layer_profile.get("flash_prefix_valid_tokens_sum", 0) or 0
                )
                stats["transformer_layer_flash_prefix_prefix_tokens_sum"] += int(
                    transformer_layer_profile.get("flash_prefix_prefix_tokens_sum", 0) or 0
                )
                stats["transformer_layer_flash_prefix_tail_tokens_sum"] += int(
                    transformer_layer_profile.get("flash_prefix_tail_tokens_sum", 0) or 0
                )
                stats["transformer_layer_dense_valid_tokens_sum"] += int(
                    transformer_layer_profile.get("dense_valid_tokens_sum", 0) or 0
                )
                stats["transformer_layer_dense_prefix_tokens_sum"] += int(
                    transformer_layer_profile.get("dense_prefix_tokens_sum", 0) or 0
                )
                stats["transformer_layer_dense_tail_tokens_sum"] += int(
                    transformer_layer_profile.get("dense_tail_tokens_sum", 0) or 0
                )
                stats["transformer_layer_total_wall_s"] += float(transformer_layer_profile.get("total_wall_s", 0.0) or 0.0)
        return out, kv_cache

    def forward_with_kv(self, src, single_eval_pos=None):
        """
        Convenience path: compute forward output via KV-cache incremental inference.
        Equivalent to forward(..., single_eval_pos=...) under single_eval_causal mode.
        """
        assert isinstance(src, tuple), 'inputs (src) have to be given as (x,y) or (style,x,y) tuple'
        if single_eval_pos is None:
            raise ValueError('single_eval_pos has to be given, instead of None.')
        if not self.single_eval_causal:
            raise ValueError("forward_with_kv requires single_eval_causal=True.")

        if len(src) == 3:
            _, x_src, y_src = src
        else:
            x_src, y_src = src
        kv_cache = self.init_kv_cache(x_src[:single_eval_pos], y_src[:single_eval_pos])
        return self.predict_query_with_kv(x_src[single_eval_pos:], kv_cache)
