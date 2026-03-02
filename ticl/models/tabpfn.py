
import torch, wandb
import torch.nn as nn

from ticl.models.layer import TransformerEncoderLayer, TransformerEncoderSimple
from ticl.utils import SeqBN, get_init_method
from ticl.models.encoders import Linear, SplitObsActionEncoder


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
        self.input_ln = SeqBN(emsize) if input_normalization else None
        self.init_method = init_method
        self.efficient_eval_masking = efficient_eval_masking
        self.tabpfn_zero_weights = tabpfn_zero_weights
        self.single_eval_causal = bool(single_eval_causal)
        self.n_out = n_out
        self.nhid = nhid
        self.init_weights()

    def _encode_xy(self, src):
        if len(src) == 3:  # style is given
            _, x_src, y_src = src
        else:
            x_src, y_src = src
        x_enc = self.encoder(x_src)
        y_enc = self.y_encoder(y_src.unsqueeze(-1) if len(y_src.shape) < len(x_enc.shape) else y_src)
        return x_enc, y_enc

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
            return self.decoder(hidden_q)

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
        return self.decoder(hidden)

    def forward_policy_step(
        self,
        x_token,
        y_token,
        kv_cache=None,
        max_cache_len=None,
        kv_cache_mode: str = "auto",
        kv_cache_page_size=None,
        allow_grad_mutable_cache: bool = False,
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

        x_enc = self.encoder(x_token)
        y_enc = self.y_encoder(y_token.unsqueeze(-1) if len(y_token.shape) < len(x_enc.shape) else y_token)
        token = x_enc + y_enc
        if self.input_ln is not None:
            token = self.input_ln(token)
        hidden, kv_cache = self.transformer_encoder.forward_step(
            token,
            kv_cache=kv_cache,
            append_to_cache=True,
            max_cache_len=max_cache_len,
            kv_cache_mode=kv_cache_mode,
            kv_cache_page_size=kv_cache_page_size,
            allow_grad_mutable_cache=allow_grad_mutable_cache,
        )
        return self.decoder(hidden), kv_cache

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
