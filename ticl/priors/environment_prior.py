import math
import inspect
import os
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from torch import nn

from ticl.distributions import parse_distributions, sample_distributions
from ticl.utils import default_device


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
        cfg.setdefault("reward_scale", {"distribution": "log_uniform", "min": 1e-3, "max": 4.0})
        cfg.setdefault("state_clip", 8.0)

        # Reward normalization / policy-gradient stability.
        cfg.setdefault("reward_norm_eps", 1e-6)
        cfg.setdefault("reward_norm_clip", 10.0)
        cfg.setdefault("discount", 1.0)

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
        self._rollout_executor = None
        self._rollout_executor_workers = 0

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

    def _build_scm_fn(self, in_dim, out_dim, h, device, generator=None):
        depth = max(2, int(h["num_layers"]))
        hidden = max(int(out_dim), int(h["prior_mlp_hidden_dim"]))
        init_std = float(h["init_std"])
        noise_std = float(h["noise_std"])
        activation = self._resolve_activation(h["prior_mlp_activations"])

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

        if generator is None:
            w = torch.randn(in_dim, m, device=device) / lengthscale
            b = 2.0 * math.pi * torch.rand(m, device=device)
            a = torch.randn(m, out_dim, device=device) / math.sqrt(max(1, m))
        else:
            w = torch.randn(in_dim, m, device=device, generator=generator) / lengthscale
            b = 2.0 * math.pi * torch.rand(m, device=device, generator=generator)
            a = torch.randn(m, out_dim, device=device, generator=generator) / math.sqrt(max(1, m))

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
    def _clamp_int(v, lo, hi):
        return int(max(lo, min(hi, int(v))))

    def _sample_single_eval_pos(self, n_samples, single_eval_pos):
        if single_eval_pos is None:
            single_eval_pos = np.random.randint(1, n_samples)
        return int(max(1, min(int(n_samples) - 1, int(single_eval_pos))))

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
            "state_clip": float(max(1.0, self._resolve_scalar(h.get("state_clip", 8.0)))),
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
                    w[bi, active_idx, :out_i] = w_b * (init_std[bi] / math.sqrt(max(1, in_i)))
                    b[bi, :out_i] = b_b * (init_std[bi] * 0.1)
                if layer_idx > 0 or input_mask is None:
                    in_mask[bi, :in_i] = 1.0
                out_mask[bi, :out_i] = 1.0
            layers.append(
                {
                    "w": w,
                    "b": b,
                    "in_mask": in_mask,
                    "out_mask": out_mask,
                    "in_cap": in_cap,
                    "activation_mask": hidden_active,
                }
            )

        final_out_mask = layers[-1]["out_mask"]

        def fn(x, generators_for_noise=None):
            z = x
            for li, layer in enumerate(layers):
                z_in = z[:, :layer["in_cap"]] * layer["in_mask"]
                z = torch.einsum("bi,bij->bj", z_in, layer["w"]) + layer["b"]
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
            if torch.any(noise_std > 0):
                if generators_for_noise is None:
                    z = z + torch.randn_like(z) * noise_std[:, None]
                else:
                    eps = torch.zeros_like(z)
                    for bi in range(batch_size):
                        if float(noise_std[bi]) <= 0.0:
                            continue
                        g = generators_for_noise[bi]
                        if g is None:
                            e_b = torch.randn((z.shape[1],), device=z.device, dtype=z.dtype)
                        else:
                            e_b = torch.randn((z.shape[1],), device=z.device, dtype=z.dtype, generator=g)
                        eps[bi] = e_b * noise_std[bi]
                    z = z + eps
            z = torch.tanh(z)
            return z * final_out_mask

        return fn

    def _build_gp_hetero_batch_fn(self, in_dims, out_dims, h_list, device, generators=None, input_mask=None):
        batch_size = len(h_list)
        in_dims = torch.as_tensor(in_dims, device=device, dtype=torch.long)
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

        if input_mask is not None:
            in_cap = int(input_mask.shape[1])
            in_mask = input_mask.clone()
        else:
            in_cap = int(in_dims.max().item())
            in_mask = torch.zeros((batch_size, in_cap), device=device, dtype=torch.float32)
        m_cap = int(m_dims.max().item())
        out_cap = int(out_dims.max().item())
        w = torch.zeros((batch_size, in_cap, m_cap), device=device, dtype=torch.float32)
        b = torch.zeros((batch_size, m_cap), device=device, dtype=torch.float32)
        a = torch.zeros((batch_size, m_cap, out_cap), device=device, dtype=torch.float32)
        m_mask = torch.zeros((batch_size, m_cap), device=device, dtype=torch.float32)
        out_mask = torch.zeros((batch_size, out_cap), device=device, dtype=torch.float32)

        for bi in range(batch_size):
            if input_mask is not None:
                active_idx = torch.nonzero(input_mask[bi] > 0, as_tuple=False).squeeze(1)
                in_i = int(active_idx.numel())
            else:
                in_i = int(in_dims[bi].item())
                active_idx = torch.arange(in_i, device=device, dtype=torch.long)
            m_i = int(m_dims[bi].item())
            out_i = int(out_dims[bi].item())
            g = None if generators is None else generators[bi]
            if g is None:
                w_b = torch.randn((in_i, m_i), device=device, dtype=torch.float32)
                b_b = torch.rand((m_i,), device=device, dtype=torch.float32)
                a_b = torch.randn((m_i, out_i), device=device, dtype=torch.float32)
            else:
                w_b = torch.randn((in_i, m_i), device=device, dtype=torch.float32, generator=g)
                b_b = torch.rand((m_i,), device=device, dtype=torch.float32, generator=g)
                a_b = torch.randn((m_i, out_i), device=device, dtype=torch.float32, generator=g)
            w[bi, active_idx, :m_i] = w_b / lengthscale[bi]
            b[bi, :m_i] = 2.0 * math.pi * b_b
            a[bi, :m_i, :out_i] = a_b / math.sqrt(max(1, m_i))
            if input_mask is None:
                in_mask[bi, :in_i] = 1.0
            m_mask[bi, :m_i] = 1.0
            out_mask[bi, :out_i] = 1.0

        def fn(x, generators_for_noise=None):
            x_in = x[:, :in_cap] * in_mask
            phi = torch.cos(torch.einsum("bi,bij->bj", x_in, w) + b) * m_mask
            y = outputscale[:, None] * torch.einsum("bi,bij->bj", phi, a)
            y = y * out_mask
            if torch.any(noise > 0):
                if generators_for_noise is None:
                    y = y + torch.randn_like(y) * noise[:, None]
                else:
                    eps = torch.zeros_like(y)
                    for bi in range(batch_size):
                        if float(noise[bi]) <= 0.0:
                            continue
                        g = generators_for_noise[bi]
                        if g is None:
                            e_b = torch.randn((y.shape[1],), device=y.device, dtype=y.dtype)
                        else:
                            e_b = torch.randn((y.shape[1],), device=y.device, dtype=y.dtype, generator=g)
                        eps[bi] = e_b * noise[bi]
                    y = y + eps
            y = torch.tanh(y)
            return y * out_mask

        return fn

    def _sample_environment_family_coarse_batch(self, h_list, device, rng_seeds=None):
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

        if family == "scm":
            depth_values = [max(2, int(h["num_layers"])) for h in h_list]
            activation_values = [self._activation_name(h["prior_mlp_activations"]) for h in h_list]
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
        else:
            x_generator = self._build_gp_hetero_batch_fn(
                in_dims=in_dims,
                out_dims=state_dims,
                h_list=h_list,
                device=device,
                generators=generators,
                input_mask=input_mask,
            )
            y_generator = self._build_gp_hetero_batch_fn(
                in_dims=in_dims,
                out_dims=torch.ones((batch_size,), device=device, dtype=torch.long),
                h_list=h_list,
                device=device,
                generators=generators,
                input_mask=input_mask,
            )
            policy_generator = self._build_gp_hetero_batch_fn(
                in_dims=in_dims,
                out_dims=action_dims,
                h_list=h_list,
                device=device,
                generators=generators,
                input_mask=input_mask,
            )

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
        state_clip = torch.tensor(
            [float(max(1.0, self._resolve_scalar(h.get("state_clip", 8.0)))) for h in h_list],
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
            "policy_generator": policy_generator,
            "alpha": alpha,
            "init_state_std": init_state_std,
            "init_action_std": init_action_std,
            "state_noise_std": state_noise_std,
            "action_noise_train_std": action_noise_train_std,
            "action_noise_eval_std": action_noise_eval_std,
            "reward_scale": reward_scale,
            "state_clip": state_clip,
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

        layer_dims = [in_dim] + [hidden] * (depth - 1) + [out_dim]
        weights = []
        biases = []
        if generators is None:
            for d_in, d_out in zip(layer_dims[:-1], layer_dims[1:]):
                scale = init_std[:, None, None] / math.sqrt(max(1, d_in))
                w = torch.randn((batch_size, d_in, d_out), device=device, dtype=torch.float32) * scale
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
                    w[bi] = w_b * (init_std[bi] / math.sqrt(max(1, d_in)))
                    b[bi] = b_b * (init_std[bi] * 0.1)
                weights.append(w)
                biases.append(b)

        def fn(x, generators_for_noise=None):
            z = x
            for i, (w, b) in enumerate(zip(weights, biases)):
                z = torch.einsum("bi,bij->bj", z, w) + b
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
        if generators is None:
            w = torch.randn((batch_size, in_dim, m), device=device, dtype=torch.float32) / lengthscale[:, None, None]
            b = 2.0 * math.pi * torch.rand((batch_size, m), device=device, dtype=torch.float32)
            a = torch.randn((batch_size, m, out_dim), device=device, dtype=torch.float32) / math.sqrt(max(1, m))
        else:
            w = torch.empty((batch_size, in_dim, m), device=device, dtype=torch.float32)
            b = torch.empty((batch_size, m), device=device, dtype=torch.float32)
            a = torch.empty((batch_size, m, out_dim), device=device, dtype=torch.float32)
            for bi in range(batch_size):
                g = generators[bi]
                w_b = torch.randn((in_dim, m), device=device, dtype=torch.float32, generator=g)
                b_b = torch.rand((m,), device=device, dtype=torch.float32, generator=g)
                a_b = torch.randn((m, out_dim), device=device, dtype=torch.float32, generator=g)
                w[bi] = w_b / lengthscale[bi]
                b[bi] = 2.0 * math.pi * b_b
                a[bi] = a_b / math.sqrt(max(1, m))

        def fn(x, generators_for_noise=None):
            phi = torch.cos(torch.einsum("bi,bij->bj", x, w) + b)
            y = outputscale[:, None] * torch.einsum("bi,bij->bj", phi, a)
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
        state_clip = torch.tensor(
            [float(max(1.0, self._resolve_scalar(h.get("state_clip", 8.0)))) for h in h_list],
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
            "state_clip": state_clip,
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
            reward_next = reward_next_raw
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
            state_clip = env["state_clip"][:, None]
            state_next = torch.maximum(torch.minimum(state_next, state_clip), -state_clip)
            state_next = torch.tanh(state_next)

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
            return {k: EnvironmentPrior._detach_policy_cache(v, clone_tensors=clone_tensors) for k, v in cache.items()}
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
        device_obj = device if isinstance(device, torch.device) else torch.device(str(device))
        profile_rollout_breakdown_cuda = bool(
            profile_rollout_breakdown
            and device_obj.type == "cuda"
            and torch.cuda.is_available()
        )
        policy_cuda_pairs = []
        transition_cuda_pairs = []

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
        y_steps = torch.empty((n_samples, batch_size), device=device, dtype=torch.float32)
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

        strict_seed_mode = rollout_generators is not None
        transition_noise = None
        action_noise_train = None
        action_noise_eval = None
        state_noise = None
        dropout_draws = None

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

            if t < single_eval_pos:
                if strict_seed_mode and action_noise_train_generators is not None:
                    action_noise_train_t = _draw_step_randn_with_optional_generators(
                        action_noise_train_generators,
                        action_dim,
                    )
                    action_next = torch.tanh(action_next + action_noise_train_t * env["action_noise_train_std"][:, None])
                elif action_noise_train is not None:
                    action_next = torch.tanh(action_next + action_noise_train[t] * env["action_noise_train_std"][:, None])
            else:
                if strict_seed_mode and action_noise_eval_generators is not None:
                    action_noise_eval_t = _draw_step_randn_with_optional_generators(
                        action_noise_eval_generators,
                        action_dim,
                    )
                    action_next = torch.tanh(action_next + action_noise_eval_t * env["action_noise_eval_std"][:, None])
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
            else:
                noise_t = transition_noise[t]
            env_in[:, :state_dim] = state_t
            env_in[:, env_obs_start: env_obs_start + obs_dim] = obs_t
            env_in[:, env_action_start: env_action_start + action_dim] = action_next
            env_in[:, env_noise_start: env_noise_start + noise_dim] = noise_t

            reward_next_raw = env["reward_scale"] * env["y_generator"](
                env_in,
                generators_for_noise=env_noise_generators,
            ).reshape(batch_size)
            reward_next = reward_next_raw
            reward_mask_next = torch.ones((batch_size,), device=device, dtype=torch.float32)

            if strict_seed_mode and dropout_draw_generators is not None:
                drop_draw = _draw_step_rand_with_optional_generators(dropout_draw_generators)
                drop_mask = dropout_active & (drop_draw < env["reward_dropout_ratio"])
                if collect_runtime_info:
                    reward_drop_count = reward_drop_count + drop_mask.to(dtype=torch.int64)
                reward_mask_next = torch.where(drop_mask, torch.zeros_like(reward_mask_next), reward_mask_next)
                impute_mask = drop_mask & env["reward_dropout_impute_zero"]
                reward_next = torch.where(impute_mask, torch.zeros_like(reward_next), reward_next)
            elif dropout_draws is not None:
                drop_mask = dropout_active & (dropout_draws[t] < env["reward_dropout_ratio"])
                if collect_runtime_info:
                    reward_drop_count = reward_drop_count + drop_mask.to(dtype=torch.int64)
                reward_mask_next = torch.where(drop_mask, torch.zeros_like(reward_mask_next), reward_mask_next)
                impute_mask = drop_mask & env["reward_dropout_impute_zero"]
                reward_next = torch.where(impute_mask, torch.zeros_like(reward_next), reward_next)

            x_next = env["x_generator"](env_in, generators_for_noise=env_noise_generators)
            state_next = (1.0 - env["alpha"][:, None]) * state_t + env["alpha"][:, None] * x_next
            if strict_seed_mode and state_noise_generators is not None:
                state_noise_t = _draw_step_randn_with_optional_generators(
                    state_noise_generators,
                    state_dim,
                )
                state_next = state_next + state_noise_t * env["state_noise_std"][:, None]
            elif state_noise is not None:
                state_next = state_next + state_noise[t] * env["state_noise_std"][:, None]
            state_clip = env["state_clip"][:, None]
            state_next = torch.maximum(torch.minimum(state_next, state_clip), -state_clip)
            state_next = torch.tanh(state_next)
            if transition_cuda_start is not None:
                transition_cuda_end = torch.cuda.Event(enable_timing=True)
                transition_cuda_end.record()
                transition_cuda_pairs.append((transition_cuda_start, transition_cuda_end))

            if tbptt_window_active:
                y_steps[t] = reward_next.detach()
                tbptt_reward_buffer.append(reward_next)
            else:
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
                        cache = self._detach_policy_cache(cache, clone_tensors=(tbptt_reward_sink is None))
                    if tbptt_reward_sink is not None:
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
            rollout_profile = {
                "policy_cuda_ms": policy_cuda_ms,
                "transition_cuda_ms": transition_cuda_ms,
                "steps": int(n_samples),
                "batch_size": int(batch_size),
            }
        self.last_rollout_profile = rollout_profile
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

        structure_groups = {}
        for bi, h in enumerate(h_list_effective):
            family = self._normalize_family(h.get("family", "scm"))
            sig = (family,)
            structure_groups.setdefault(sig, []).append((bi, h))

        rollout_generators = self._make_generators_from_seeds(rollout_rng_seeds, batch_size, device)
        profile_rollout_breakdown_flag = str(os.environ.get("TICL_PROFILE_ROLLOUT_BREAKDOWN", "")).strip().lower()
        profile_rollout_breakdown = profile_rollout_breakdown_flag in {"1", "true", "yes", "on"}
        device_obj = device if isinstance(device, torch.device) else torch.device(str(device))
        profile_rollout_breakdown_cuda = bool(
            profile_rollout_breakdown
            and device_obj.type == "cuda"
            and torch.cuda.is_available()
        )
        policy_cuda_pairs = []
        transition_cuda_pairs = []

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
        alpha = torch.empty((batch_size,), device=device, dtype=torch.float32)
        state_clip = torch.empty((batch_size,), device=device, dtype=torch.float32)
        reward_dropout_enabled = torch.empty((batch_size,), device=device, dtype=torch.bool)
        reward_dropout_impute_zero = torch.empty((batch_size,), device=device, dtype=torch.bool)
        reward_dropout_ratio = torch.empty((batch_size,), device=device, dtype=torch.float32)
        family_list = [None] * batch_size

        transition_groups = []
        for group in structure_groups.values():
            group_indices = [idx for idx, _ in group]
            group_h_list = [h for _, h in group]
            group_env_seeds = (
                [env_rng_seeds[idx] for idx in group_indices]
                if env_rng_seeds is not None
                else None
            )
            env_batch = self._sample_environment_family_coarse_batch(
                h_list=group_h_list,
                device=device,
                rng_seeds=group_env_seeds,
            )
            group_idx = torch.tensor(group_indices, device=device, dtype=torch.long)
            group_bs = int(group_idx.numel())

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
            alpha[group_idx] = env_batch["alpha"]
            state_clip[group_idx] = env_batch["state_clip"]
            reward_dropout_enabled[group_idx] = env_batch["reward_dropout_enabled"]
            reward_dropout_impute_zero[group_idx] = env_batch["reward_dropout_impute_zero"]
            reward_dropout_ratio[group_idx] = env_batch["reward_dropout_ratio"]
            for global_idx in group_indices:
                family_list[global_idx] = str(env_batch["family"])

            group_rollout_generators = None
            if rollout_generators is not None:
                group_rollout_generators = [rollout_generators[idx] for idx in group_indices]
            group_env_total_dim = state_dim_g + obs_dim_g + action_dim_g + noise_dim_g + zero_pad_dim_g
            transition_groups.append(
                {
                    "indices": group_idx,
                    "env": env_batch,
                    "state_dim": state_dim_g,
                    "obs_dim": obs_dim_g,
                    "action_dim": action_dim_g,
                    "noise_dim": noise_dim_g,
                    "env_obs_start": state_dim_g,
                    "env_action_start": state_dim_g + obs_dim_g,
                    "env_noise_start": state_dim_g + obs_dim_g + action_dim_g,
                    "env_in": torch.zeros((group_bs, group_env_total_dim), device=device, dtype=torch.float32),
                    "rollout_generators": group_rollout_generators,
                    "state_noise_active": bool(torch.any(env_batch["state_noise_std"] > 0).item()),
                }
            )

        # Keep transition subgroups contiguous in memory to avoid per-step
        # index_select/scatter overhead in the rollout hot loop.
        perm = torch.cat([g["indices"] for g in transition_groups], dim=0)
        if int(perm.numel()) != batch_size:
            raise RuntimeError("family-group rollout internal permutation size mismatch")
        identity_perm = torch.arange(batch_size, device=device, dtype=torch.long)
        needs_unpermute = not bool(torch.equal(perm, identity_perm))
        inv_perm = torch.empty_like(perm)
        inv_perm.scatter_(0, perm, identity_perm)
        if needs_unpermute:
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
            alpha = alpha.index_select(0, perm)
            state_clip = state_clip.index_select(0, perm)
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
            if rollout_generators is not None:
                group["rollout_generators"] = rollout_generators[group_cursor: group_cursor + group_bs]
            else:
                group["rollout_generators"] = None
            group_cursor += group_bs
        if group_cursor != batch_size:
            raise RuntimeError("family-group rollout internal subgroup cursor mismatch")

        max_state_dim = int(state_dims.max().item())
        max_obs_dim = int(obs_dims.max().item())
        max_action_dim = int(action_dims.max().item())
        max_noise_dim = int(noise_dims.max().item())

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
        y_steps = torch.empty((n_samples, batch_size), device=device, dtype=torch.float32)
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

        transition_noise = self._stack_randn_with_generators(
            rollout_generators,
            (batch_size, n_samples, max_noise_dim),
            device=device,
            dtype=torch.float32,
        ).transpose(0, 1) * noise_mask.unsqueeze(0)
        action_noise_train = None
        action_noise_eval = None
        state_noise = None
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
        dropout_draws = None
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
        }

        for t in range(n_samples):
            obs_t = state_t[:, :max_obs_dim] * obs_mask
            if collect_x:
                with torch.no_grad():
                    token_row = x_steps[t]
                    token_row.zero_()
                    if num_features > 0:
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

            if t < single_eval_pos:
                if action_noise_train is not None:
                    action_next = torch.tanh(action_next + action_noise_train[t] * action_noise_train_std[:, None]) * action_mask
            elif action_noise_eval is not None:
                action_next = torch.tanh(action_next + action_noise_eval[t] * action_noise_eval_std[:, None]) * action_mask

            noise_t = transition_noise[t]
            reward_next = torch.empty((batch_size,), device=device, dtype=torch.float32)
            reward_mask_next = torch.ones((batch_size,), device=device, dtype=torch.float32)
            state_next = torch.zeros((batch_size, max_state_dim), device=device, dtype=torch.float32)

            transition_cuda_start = None
            if profile_rollout_breakdown_cuda:
                transition_cuda_start = torch.cuda.Event(enable_timing=True)
                transition_cuda_start.record()
            for group in transition_groups:
                start = int(group["start"])
                end = int(group["end"])
                env_g = group["env"]
                state_dim_g = int(group["state_dim"])
                obs_dim_g = int(group["obs_dim"])
                action_dim_g = int(group["action_dim"])
                noise_dim_g = int(group["noise_dim"])

                state_in = state_t[start:end, :state_dim_g]
                obs_in = obs_t[start:end, :obs_dim_g]
                action_in = action_next[start:end, :action_dim_g]
                noise_in = noise_t[start:end, :noise_dim_g]

                env_in = group["env_in"]
                env_obs_start = int(group["env_obs_start"])
                env_action_start = int(group["env_action_start"])
                env_noise_start = int(group["env_noise_start"])
                env_in[:, :state_dim_g] = state_in
                env_in[:, env_obs_start: env_obs_start + obs_dim_g] = obs_in
                env_in[:, env_action_start: env_action_start + action_dim_g] = action_in
                env_in[:, env_noise_start: env_noise_start + noise_dim_g] = noise_in

                reward_scale_g = reward_scale[start:end]
                reward_next_raw = reward_scale_g * env_g["y_generator"](
                    env_in,
                    generators_for_noise=group["rollout_generators"],
                ).reshape(-1)
                reward_next_g = reward_next_raw
                reward_mask_next_g = torch.ones_like(reward_next_g)

                if dropout_draws is not None:
                    dropout_active_g = dropout_active[start:end]
                    ratio_g = reward_dropout_ratio[start:end]
                    drop_mask = dropout_active_g & (dropout_draws[t, start:end] < ratio_g)
                    if collect_runtime_info:
                        reward_drop_count[start:end] = reward_drop_count[start:end] + drop_mask.to(dtype=torch.int64)
                    reward_mask_next_g = torch.where(drop_mask, torch.zeros_like(reward_mask_next_g), reward_mask_next_g)
                    impute_mask = drop_mask & reward_dropout_impute_zero[start:end]
                    reward_next_g = torch.where(impute_mask, torch.zeros_like(reward_next_g), reward_next_g)

                x_next_g = env_g["x_generator"](
                    env_in,
                    generators_for_noise=group["rollout_generators"],
                )
                alpha_g = alpha[start:end].unsqueeze(1)
                state_next_g = (1.0 - alpha_g) * state_in + alpha_g * x_next_g
                if (state_noise is not None) and bool(group.get("state_noise_active", False)):
                    state_noise_std_g = state_noise_std[start:end]
                    state_next_g = state_next_g + state_noise[t, start:end, :state_dim_g] * state_noise_std_g[:, None]
                clip_g = state_clip[start:end].unsqueeze(1)
                state_next_g = torch.maximum(torch.minimum(state_next_g, clip_g), -clip_g)
                state_next_g = torch.tanh(state_next_g)

                state_next[start:end, :state_dim_g] = state_next_g
                reward_next[start:end] = reward_next_g
                reward_mask_next[start:end] = reward_mask_next_g
            if transition_cuda_start is not None:
                transition_cuda_end = torch.cuda.Event(enable_timing=True)
                transition_cuda_end.record()
                transition_cuda_pairs.append((transition_cuda_start, transition_cuda_end))

            state_next = state_next * state_mask
            action_next = action_next * action_mask
            if tbptt_window_active:
                y_steps[t] = reward_next.detach()
                tbptt_reward_buffer.append(reward_next)
            else:
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
                        for group in transition_groups:
                            group["env_in"] = group["env_in"].detach()
                        cache = self._detach_policy_cache(cache, clone_tensors=(tbptt_reward_sink is None))
                    if tbptt_reward_sink is not None:
                        if needs_unpermute:
                            tbptt_reward_sink(rewards_window.index_select(1, inv_perm))
                        else:
                            tbptt_reward_sink(rewards_window)

        if needs_unpermute:
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
        if profile_rollout_breakdown_cuda and (policy_cuda_pairs or transition_cuda_pairs):
            torch.cuda.synchronize(device=device_obj)
            policy_cuda_ms = float(sum(start.elapsed_time(end) for start, end in policy_cuda_pairs))
            transition_cuda_ms = float(sum(start.elapsed_time(end) for start, end in transition_cuda_pairs))
            rollout_profile = {
                "policy_cuda_ms": policy_cuda_ms,
                "transition_cuda_ms": transition_cuda_ms,
                "steps": int(n_samples),
                "batch_size": int(batch_size),
            }
        self.last_rollout_profile = rollout_profile
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
        y_steps = torch.empty((n_samples,), device=device, dtype=state_t.dtype)
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
            reward_next = reward_next_raw
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
            state_next = torch.clamp(state_next, -env["state_clip"], env["state_clip"])
            state_next = torch.tanh(state_next)

            if tbptt_window_active:
                y_steps[t] = reward_next.detach()
                tbptt_reward_buffer.append(reward_next)
            else:
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
                        cache = self._detach_policy_cache(cache, clone_tensors=(tbptt_reward_sink is None))
                    if tbptt_reward_sink is not None:
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
            }
        else:
            info = None
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
            reward_next = reward_next_raw
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
            state_next = torch.clamp(state_next, -state_clip, state_clip)
            state_next = torch.tanh(state_next)

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
        x = torch.empty((n_samples, batch_size, num_features), device=device, dtype=torch.float32) if collect_x else None
        rewards = torch.empty((n_samples, batch_size), device=device, dtype=torch.float32)
        infos = [None] * batch_size

        backend = self._resolve_batch_parallel_backend()
        strict_rng_match = self._resolve_batch_vectorized_strict_rng_match()
        grouping_mode = self._resolve_batch_vectorized_grouping()
        h_list = self._sample_batch_hypers(batch_size)
        env_seeds = self._sample_seed_list(batch_size) if strict_rng_match else None
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
                    )
                if collect_x:
                    x[:, group_indices] = x_group
                rewards[:, group_indices] = y_group
                for local_idx, global_idx in enumerate(group_indices):
                    infos[global_idx] = infos_group[local_idx]
                group_profile = self.last_rollout_profile
                if isinstance(group_profile, dict):
                    if rollout_profile_acc is None:
                        rollout_profile_acc = {
                            "policy_cuda_ms": 0.0,
                            "transition_cuda_ms": 0.0,
                            "steps": int(n_samples),
                            "batch_size": int(batch_size),
                        }
                    rollout_profile_acc["policy_cuda_ms"] += float(group_profile.get("policy_cuda_ms", 0.0))
                    rollout_profile_acc["transition_cuda_ms"] += float(group_profile.get("transition_cuda_ms", 0.0))
            self.last_rollout_profile = rollout_profile_acc
            self.last_runtime_info = infos if collect_runtime_info else [None] * batch_size
            return {
                "x": x,
                "rewards": rewards,
                "info": self.last_runtime_info,
                "single_eval_pos": single_eval_pos,
                "rollout_profile": self.last_rollout_profile,
            }

        # Fallback serial baseline for explicit non-vectorized backend.
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
            )
            if collect_x:
                x[:, b] = x_one
            rewards[:, b] = y_one
            infos[b] = info
        self.last_runtime_info = infos if collect_runtime_info else [None] * batch_size
        self.last_rollout_profile = None
        return {
            "x": x,
            "rewards": rewards,
            "info": self.last_runtime_info,
            "single_eval_pos": single_eval_pos,
            "rollout_profile": self.last_rollout_profile,
        }

    def normalize_rewards(self, rewards, eps=None, clip=None, detach_stats=True):
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
        normalized = (rewards - mean) / std
        if clip is not None and float(clip) > 0:
            normalized = normalized.clamp(-float(clip), float(clip))
        return normalized

    def policy_gradient_loss_from_rewards(
        self,
        rewards,
        normalize=True,
        discount=None,
        detach_stats=True,
        eps=None,
        clip=None,
    ):
        if rewards.ndim != 2:
            raise ValueError(f"rewards must have shape (T, B), got {tuple(rewards.shape)}")
        if discount is None:
            discount = self._resolve_scalar(self.config.get("discount", 1.0))
        discount = float(max(0.0, min(1.0, discount)))

        weighted = rewards
        if discount < 1.0:
            t = torch.arange(rewards.shape[0], device=rewards.device, dtype=rewards.dtype)
            weights = (discount ** t).unsqueeze(1)
            weighted = rewards * weights

        objective_tensor = (
            self.normalize_rewards(weighted, eps=eps, clip=clip, detach_stats=detach_stats)
            if normalize
            else weighted
        )
        objective = objective_tensor.mean()
        loss = -objective

        stats = {
            "objective": objective.detach(),
            "reward_mean": rewards.mean().detach(),
            "reward_std": rewards.std(unbiased=False).detach(),
        }
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
        normalize=True,
        discount=None,
        detach_stats=True,
        eps=None,
        clip=None,
        collect_x=False,
        tbptt_window=None,
        tbptt_loss_sink=None,
    ):
        n_samples = int(n_samples)
        batch_size = int(batch_size)
        tbptt_window_active = False
        tbptt_window_size = n_samples
        if tbptt_window is not None:
            w = int(tbptt_window)
            if 0 < w < n_samples:
                tbptt_window_active = True
                tbptt_window_size = w

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
            )
            loss, stats = self.policy_gradient_loss_from_rewards(
                rewards=rollout["rewards"],
                normalize=normalize,
                discount=discount,
                detach_stats=detach_stats,
                eps=eps,
                clip=clip,
            )
            rollout_profile = rollout.get("rollout_profile", None)
            if isinstance(rollout_profile, dict):
                stats["rollout_policy_cuda_ms"] = float(rollout_profile.get("policy_cuda_ms", 0.0))
                stats["rollout_transition_cuda_ms"] = float(rollout_profile.get("transition_cuda_ms", 0.0))
            return loss, rollout, stats

        reward_sum = None
        reward_sumsq = None
        reward_count = 0
        weighted_losses = []
        objective_accum = None
        total_weight = 0.0
        n_samples_f = float(max(1, n_samples))
        batch_size_f = float(max(1, batch_size))

        def _tbptt_reward_sink(rewards_window):
            nonlocal reward_sum, reward_sumsq, reward_count
            nonlocal objective_accum, total_weight
            loss_window, stats_window = self.policy_gradient_loss_from_rewards(
                rewards=rewards_window,
                normalize=normalize,
                discount=discount,
                detach_stats=detach_stats,
                eps=eps,
                clip=clip,
            )
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

            rewards_det = rewards_window.detach().to(dtype=torch.float64)
            if reward_sum is None:
                reward_sum = rewards_det.sum()
                reward_sumsq = (rewards_det * rewards_det).sum()
            else:
                reward_sum = reward_sum + rewards_det.sum()
                reward_sumsq = reward_sumsq + (rewards_det * rewards_det).sum()
            reward_count += int(rewards_det.numel())

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
            tbptt_window=tbptt_window_size,
            tbptt_reward_sink=_tbptt_reward_sink,
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

        stats = {
            "objective": objective,
            "reward_mean": reward_mean,
            "reward_std": reward_std,
        }
        rollout_profile = rollout.get("rollout_profile", None)
        if isinstance(rollout_profile, dict):
            stats["rollout_policy_cuda_ms"] = float(rollout_profile.get("policy_cuda_ms", 0.0))
            stats["rollout_transition_cuda_ms"] = float(rollout_profile.get("transition_cuda_ms", 0.0))
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
