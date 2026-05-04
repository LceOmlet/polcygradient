import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ticl.analysis.phase2_attention_core_origin_probe import (
    _feature_target_corr_summary,
    _project_stats,
    _safe_corrcoef,
)
from ticl.analysis.phase2_attention_gate_input_alignment_probe import (
    _direction_from_head,
    _safe_cosine,
)
from ticl.analysis.phase2_attention_internal_geometry_probe import (
    _evaluate_scalar_head,
    _geometry_summary,
)
from ticl.analysis.phase2_simple_scalar_value_head_probe import (
    _build_algo,
    _collect_rollout,
    _default_device,
    _fit_ridge_scalar_head,
)
from ticl.analysis.critic_free_single_env_audit import (
    _build_audit_env_cfg,
    _prepare_audit_fixed_env_contract,
    _seed_all,
)
from ticl.model_builder import load_model
from ticl.models.rwkv7_pfn import _load_official_rwkv7_train_temp


def _group_mean_subspace_mass(direction: torch.Tensor, n_head: int) -> dict:
    if direction is None:
        return {"mean_subspace_mass": 0.0, "residual_mass": 1.0}
    d = direction.detach().cpu().to(dtype=torch.float32).reshape(-1)
    c = int(d.numel())
    head_dim = int(c // int(n_head))
    if head_dim * int(n_head) != c:
        raise RuntimeError("Direction dim not divisible by n_head.")
    basis = []
    for h in range(int(n_head)):
        vec = torch.zeros((c,), dtype=torch.float32)
        start = h * head_dim
        vec[start : start + head_dim] = 1.0 / float(np.sqrt(head_dim))
        basis.append(vec)
    basis = torch.stack(basis, dim=1)  # [C, H]
    coeff = basis.T @ d
    mean_mass = float(coeff.square().sum().item())
    return {
        "mean_subspace_mass": mean_mass,
        "residual_mass": float(max(0.0, 1.0 - mean_mass)),
    }


def _safe_jvp_norm_ratio(module, x_sample: torch.Tensor, direction: torch.Tensor) -> float:
    if direction is None:
        return 0.0
    try:
        module_device = next(module.parameters()).device
    except StopIteration:
        module_device = x_sample.device
    x0 = x_sample.detach().clone().to(device=module_device, dtype=torch.float32).requires_grad_(True)
    v = direction.detach().clone().to(device=module_device, dtype=torch.float32)
    v = v / torch.clamp(torch.linalg.vector_norm(v), min=1e-12)

    def _fn(inp: torch.Tensor) -> torch.Tensor:
        return module(inp.unsqueeze(0)).squeeze(0)

    _, jvp = torch.autograd.functional.jvp(_fn, (x0,), (v,), create_graph=False, strict=False)
    denom = float(torch.linalg.vector_norm(v).item())
    num = float(torch.linalg.vector_norm(jvp.detach()).item())
    if denom <= 0.0:
        return 0.0
    return num / denom


def _random_direction_stats(module, samples: torch.Tensor, dim: int, *, n_draws: int = 8, max_samples: int = 32) -> dict:
    if int(samples.shape[0]) == 0:
        return {
            "mean_ratio": 0.0,
            "std_ratio": 0.0,
            "max_ratio": 0.0,
            "min_ratio": 0.0,
        }
    ratios = []
    gen = torch.Generator(device="cpu")
    gen.manual_seed(12345)
    sample_count = min(int(samples.shape[0]), int(max_samples))
    subset = samples[:sample_count].detach().cpu().to(dtype=torch.float32)
    for _ in range(int(n_draws)):
        v = torch.randn((dim,), generator=gen, dtype=torch.float32)
        v = v / torch.clamp(torch.linalg.vector_norm(v), min=1e-12)
        per_sample = []
        for idx in range(sample_count):
            per_sample.append(_safe_jvp_norm_ratio(module, subset[idx], v))
        ratios.append(float(np.mean(np.asarray(per_sample, dtype=np.float32))))
    arr = np.asarray(ratios, dtype=np.float32)
    return {
        "mean_ratio": float(arr.mean()) if arr.size > 0 else 0.0,
        "std_ratio": float(arr.std()) if arr.size > 0 else 0.0,
        "max_ratio": float(arr.max()) if arr.size > 0 else 0.0,
        "min_ratio": float(arr.min()) if arr.size > 0 else 0.0,
    }


def _direction_transform_stats(
    *,
    att_core_raw: torch.Tensor,
    gn_no_affine: torch.Tensor,
    ln_x_out: torch.Tensor,
    returns: torch.Tensor,
    mask: torch.Tensor,
    ln_x_module,
    n_head: int,
) -> dict:
    att_head = _fit_ridge_scalar_head(att_core_raw, returns, mask, ridge_lambda=1e-3)
    gn_head = _fit_ridge_scalar_head(gn_no_affine, returns, mask, ridge_lambda=1e-3)
    ln_head = _fit_ridge_scalar_head(ln_x_out, returns, mask, ridge_lambda=1e-3)
    d_att = _direction_from_head(att_head)
    d_gn = _direction_from_head(gn_head)
    d_ln = _direction_from_head(ln_head)

    mask_bool = mask.reshape(-1).to(dtype=torch.bool)
    y = returns[mask_bool].detach().cpu().numpy()

    def _proj(x: torch.Tensor, d: torch.Tensor | None) -> np.ndarray:
        if d is None:
            return np.zeros((int(mask_bool.sum().item()),), dtype=np.float32)
        centered = x[mask_bool].to(dtype=torch.float32) - x[mask_bool].to(dtype=torch.float32).mean(dim=0, keepdim=True)
        return (centered @ d).detach().cpu().numpy()

    p_att_att = _proj(att_core_raw, d_att)
    p_gn_att = _proj(gn_no_affine, d_att)
    p_ln_att = _proj(ln_x_out, d_att)
    p_gn_gn = _proj(gn_no_affine, d_gn)
    p_ln_ln = _proj(ln_x_out, d_ln)

    samples = att_core_raw[mask_bool].detach().cpu().to(dtype=torch.float32)
    sample_count = min(int(samples.shape[0]), 32)
    subset = samples[:sample_count]
    target_ratios = [_safe_jvp_norm_ratio(ln_x_module, subset[idx], d_att) for idx in range(sample_count)] if d_att is not None else [0.0]
    target_ratio_arr = np.asarray(target_ratios, dtype=np.float32)

    return {
        "direction_cosines": {
            "att_vs_gn": _safe_cosine(d_att, d_gn),
            "att_vs_ln": _safe_cosine(d_att, d_ln),
            "gn_vs_ln": _safe_cosine(d_gn, d_ln),
        },
        "group_mean_subspace_mass": {
            "att_direction": _group_mean_subspace_mass(d_att, int(n_head)),
            "gn_direction": _group_mean_subspace_mass(d_gn, int(n_head)),
            "ln_direction": _group_mean_subspace_mass(d_ln, int(n_head)),
        },
        "projection_stats": {
            "att_core_along_att": _project_stats(att_core_raw, d_att, mask),
            "gn_no_affine_along_att": _project_stats(gn_no_affine, d_att, mask),
            "ln_x_out_along_att": _project_stats(ln_x_out, d_att, mask),
            "gn_no_affine_along_gn": _project_stats(gn_no_affine, d_gn, mask),
            "ln_x_out_along_ln": _project_stats(ln_x_out, d_ln, mask),
        },
        "projection_corr_with_target": {
            "att_core_along_att": _safe_corrcoef(p_att_att, y),
            "gn_no_affine_along_att": _safe_corrcoef(p_gn_att, y),
            "ln_x_out_along_att": _safe_corrcoef(p_ln_att, y),
            "gn_no_affine_along_gn": _safe_corrcoef(p_gn_gn, y),
            "ln_x_out_along_ln": _safe_corrcoef(p_ln_ln, y),
        },
        "projection_std_ratios_vs_att": {
            "gn_no_affine_along_att": float(np.asarray(p_gn_att, dtype=np.float32).std() / max(np.asarray(p_att_att, dtype=np.float32).std(), 1e-12)),
            "ln_x_out_along_att": float(np.asarray(p_ln_att, dtype=np.float32).std() / max(np.asarray(p_att_att, dtype=np.float32).std(), 1e-12)),
            "ln_x_out_along_ln": float(np.asarray(p_ln_ln, dtype=np.float32).std() / max(np.asarray(p_att_att, dtype=np.float32).std(), 1e-12)),
        },
        "local_jvp": {
            "target_direction_mean_ratio": float(target_ratio_arr.mean()) if target_ratio_arr.size > 0 else 0.0,
            "target_direction_std_ratio": float(target_ratio_arr.std()) if target_ratio_arr.size > 0 else 0.0,
            "target_direction_min_ratio": float(target_ratio_arr.min()) if target_ratio_arr.size > 0 else 0.0,
            "target_direction_max_ratio": float(target_ratio_arr.max()) if target_ratio_arr.size > 0 else 0.0,
            "random_direction_stats": _random_direction_stats(ln_x_module, subset, int(att_core_raw.shape[-1])),
        },
    }


@torch.no_grad()
def _forward_pack_ln_x_stages(policy, bucket_obs: torch.Tensor, *, max_layer: int = 1) -> dict:
    tokens = policy._encode_obs_tokens(bucket_obs)
    tokens = policy.rlpfn_model._cast_token_for_rwkv_core(tokens)
    seq_len = int(tokens.shape[0])
    if seq_len == 0:
        raise RuntimeError("Empty token pack.")
    pad_len = int((-seq_len) % 16)
    if pad_len > 0:
        tokens_work = torch.cat(
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
    else:
        tokens_work = tokens

    x = tokens_work.transpose(0, 1)
    stages = {"token_encoder_output": tokens_work}
    stats = {}
    v_first = None
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=bool(tokens_work.is_cuda)):
        for layer_idx, block in enumerate(policy.rlpfn_model.rwkv_core.blocks):
            if layer_idx > int(max_layer):
                break
            official_train = _load_official_rwkv7_train_temp(block.head_size)
            if hasattr(block, "ln0"):
                x = block.ln0(x)

            x_att = block.ln1(x)
            bsz, seq_work_len, channels = x_att.shape
            xx = block.att.time_shift(x_att) - x_att

            xr = x_att + xx * block.att.x_r
            xw = x_att + xx * block.att.x_w
            xk = x_att + xx * block.att.x_k
            xv = x_att + xx * block.att.x_v
            xa = x_att + xx * block.att.x_a
            xg = x_att + xx * block.att.x_g

            r = block.att.receptance(xr)
            w = block.att.w0 + torch.tanh(xw @ block.att.w1) @ block.att.w2
            k_raw = block.att.key(xk)
            v_raw = block.att.value(xv)
            if layer_idx == 0:
                v = v_raw
                v_first = v_raw
            else:
                v = v_raw + (v_first - v_raw) * torch.sigmoid(block.att.v0 + (xv @ block.att.v1) @ block.att.v2)
            a = torch.sigmoid(block.att.a0 + (xa @ block.att.a1) @ block.att.a2)
            g = torch.sigmoid(xg @ block.att.g1) @ block.att.g2

            kk_norm = k_raw * block.att.k_k
            kk_norm = F.normalize(
                kk_norm.view(bsz, seq_work_len, block.n_head, -1),
                dim=-1,
                p=2.0,
            ).view(bsz, seq_work_len, channels)
            k = k_raw * (1 + (a - 1) * block.att.k_a)

            att_core = official_train.RWKV7_CLAMPW_CUDA(
                r.to(dtype=torch.bfloat16),
                w.to(dtype=torch.bfloat16),
                k.to(dtype=torch.bfloat16),
                v.to(dtype=torch.bfloat16),
                (-kk_norm).to(dtype=torch.bfloat16),
                (kk_norm * a).to(dtype=torch.bfloat16),
            )
            ln_x_out = block.att.ln_x(att_core.view(bsz * seq_work_len, channels)).view(bsz, seq_work_len, channels)
            gn_no_affine = F.group_norm(
                att_core.view(bsz * seq_work_len, channels).to(dtype=torch.float32),
                num_groups=int(block.n_head),
                weight=None,
                bias=None,
                eps=float(block.att.ln_x.eps),
            ).view(bsz, seq_work_len, channels)
            residual_term = (
                (r.view(bsz, seq_work_len, block.n_head, -1) * k.view(bsz, seq_work_len, block.n_head, -1) * block.att.r_k)
                .sum(dim=-1, keepdim=True)
                * v.view(bsz, seq_work_len, block.n_head, -1)
            ).view(bsz, seq_work_len, channels)
            pre_output_sum = ln_x_out + residual_term
            att_out = block.att.output(pre_output_sum * g)

            stages[f"layer{layer_idx}.att_core_raw"] = att_core.transpose(0, 1).to(dtype=torch.float32)
            stages[f"layer{layer_idx}.gn_no_affine"] = gn_no_affine.transpose(0, 1).to(dtype=torch.float32)
            stages[f"layer{layer_idx}.ln_x_out"] = ln_x_out.transpose(0, 1).to(dtype=torch.float32)
            stages[f"layer{layer_idx}.residual_term"] = residual_term.transpose(0, 1).to(dtype=torch.float32)
            stages[f"layer{layer_idx}.pre_output_sum"] = pre_output_sum.transpose(0, 1).to(dtype=torch.float32)
            stages[f"layer{layer_idx}.att_out"] = att_out.transpose(0, 1).to(dtype=torch.float32)
            stats[f"layer{layer_idx}"] = {
                "ln_x_weight": block.att.ln_x.weight.detach().cpu().to(dtype=torch.float32),
                "ln_x_bias": block.att.ln_x.bias.detach().cpu().to(dtype=torch.float32),
                "n_head": int(block.n_head),
                "eps": float(block.att.ln_x.eps),
            }

            x = x + att_out
            x = x + block.ffn(block.ln2(x))

    if pad_len > 0:
        for key in list(stages.keys()):
            stages[key] = stages[key][:seq_len]
    return {
        "stages": stages,
        "stats": stats,
    }


@torch.no_grad()
def _extract_ln_x_dataset(algo, *, max_layer: int = 1) -> dict:
    rollout_buffer = algo.rollout_buffer
    stage_latents = None
    layer_stats = None
    targets = []
    masks = []
    for rollout_data in rollout_buffer.get_gpu_flat(algo.batch_size):
        obs = rollout_data.observations
        seq_lengths = np.asarray(rollout_data.seq_lengths, dtype=np.int64)
        n_seq = int(len(seq_lengths))
        if int(seq_lengths.sum()) != int(obs.shape[0]):
            raise RuntimeError("seq_lengths sum does not match flat obs batch.")
        seq_start_indices = np.zeros((n_seq,), dtype=np.int64)
        if n_seq > 1:
            seq_start_indices[1:] = np.cumsum(seq_lengths[:-1], dtype=np.int64)

        def _resolve_pack_capacity(seq_len: int, total_batch: int) -> int:
            resolve_chunk = getattr(algo.policy.rlpfn_model, "resolve_replay_batch_chunk_size", None)
            if callable(resolve_chunk):
                batch_chunk_size = int(resolve_chunk(seq_len=int(seq_len), total_batch=int(total_batch)))
            else:
                batch_chunk_size = int(total_batch)
            return int(max(1, min(int(total_batch), batch_chunk_size)))

        stage_flat = None
        order = sorted(range(n_seq), key=lambda idx: int(seq_lengths[int(idx)]), reverse=True)
        cursor = 0
        while cursor < n_seq:
            first_seq_idx = int(order[cursor])
            pack_max_len = int(seq_lengths[first_seq_idx])
            remaining = int(n_seq - cursor)
            pack_capacity = _resolve_pack_capacity(seq_len=pack_max_len, total_batch=remaining)
            pack_seq_indices = [int(idx) for idx in order[cursor : cursor + pack_capacity]]
            pack_size = int(len(pack_seq_indices))
            bucket_obs = torch.zeros(
                (pack_max_len, pack_size, int(obs.shape[-1])),
                device=obs.device,
                dtype=obs.dtype,
            )
            for col, seq_idx in enumerate(pack_seq_indices):
                seg_start = int(seq_start_indices[int(seq_idx)])
                seg_len = int(seq_lengths[int(seq_idx)])
                bucket_obs[:seg_len, col].copy_(obs[seg_start : seg_start + seg_len])

            pack_result = _forward_pack_ln_x_stages(algo.policy, bucket_obs, max_layer=int(max_layer))
            stage_pack = pack_result["stages"]
            if stage_flat is None:
                stage_flat = {
                    key: torch.empty(
                        (int(obs.shape[0]), int(value.shape[-1])),
                        device=value.device,
                        dtype=value.dtype,
                    )
                    for key, value in stage_pack.items()
                }
                if stage_latents is None:
                    stage_latents = {key: [] for key in stage_pack.keys()}
                    layer_stats = pack_result["stats"]
            for pack_col, seq_idx in enumerate(pack_seq_indices):
                seg_start = int(seq_start_indices[int(seq_idx)])
                seg_len = int(seq_lengths[int(seq_idx)])
                for key in stage_pack.keys():
                    stage_flat[key][seg_start : seg_start + seg_len].copy_(stage_pack[key][:seg_len, pack_col])
            cursor += len(pack_seq_indices)

        for key in stage_latents.keys():
            stage_latents[key].append(stage_flat[key].detach().cpu())
        targets.append(rollout_data.returns.detach().cpu().reshape(-1))
        masks.append((rollout_data.objective_masks > 1e-8).detach().cpu().reshape(-1).to(dtype=torch.bool))

    return {
        "stages": {key: torch.cat(value, dim=0) for key, value in stage_latents.items()},
        "layer_stats": layer_stats,
        "returns": torch.cat(targets, dim=0),
        "objective_mask": torch.cat(masks, dim=0),
    }


def run_phase2_ln_x_direction_transform_probe(
    *,
    checkpoint_path: str,
    device: str | None = None,
    frozen_h_seed: int = 12345,
    train_env_seed: int = 2020,
    train_rollout_seed: int = 4040,
    single_eval_pos: int = 64,
    n_steps: int = 256,
    batch_size: int = 256,
    n_epochs: int = 1,
    learning_rate: float = 2e-4,
    target_kl: float = 0.03,
    outer_epochs: int = 1,
    build_seed: int = 4040,
    ppo_reset_env_state_at_sep: bool = True,
    actor_baseline_mode: str = "learned",
    max_layer: int = 1,
) -> dict:
    device_obj = torch.device(str(device or _default_device()))
    _seed_all(int(build_seed))
    load_model.cache_clear()
    _, config = load_model(checkpoint_path, device=device_obj, verbose=False)
    env_cfg = _build_audit_env_cfg(config["prior"]["environment"])
    num_features = int(config["prior"]["num_features"])
    fixed_bundle = _prepare_audit_fixed_env_contract(
        env_cfg=env_cfg,
        boundary_contract_mode="normal",
        ppo_reset_env_state_at_sep=bool(ppo_reset_env_state_at_sep),
        frozen_h_seed=int(frozen_h_seed),
        frozen_h_seed_mode="current",
    )
    frozen_h = fixed_bundle["frozen_h"]
    algo, callback, vec_env = _build_algo(
        checkpoint_path=checkpoint_path,
        device_obj=device_obj,
        frozen_h=frozen_h,
        env_cfg=env_cfg,
        num_features=int(num_features),
        train_env_seed=int(train_env_seed),
        train_rollout_seed=int(train_rollout_seed),
        single_eval_pos=int(single_eval_pos),
        n_steps=int(n_steps),
        batch_size=int(batch_size),
        n_epochs=int(n_epochs),
        learning_rate=float(learning_rate),
        target_kl=float(target_kl),
        ppo_reset_env_state_at_sep=bool(ppo_reset_env_state_at_sep),
        actor_baseline_mode=str(actor_baseline_mode),
        build_seed=int(build_seed),
    )
    total_timesteps = int(n_steps) * int(outer_epochs)
    try:
        _collect_rollout(algo, callback, vec_env, progress_timestep=0, total_timesteps=total_timesteps)
        algo.train()
        same_dataset = _extract_ln_x_dataset(algo, max_layer=int(max_layer))
        progress_next = min(int(n_steps), total_timesteps)
        _collect_rollout(algo, callback, vec_env, progress_timestep=progress_next, total_timesteps=total_timesteps)
        next_dataset = _extract_ln_x_dataset(algo, max_layer=int(max_layer))
    finally:
        vec_env.close()

    layer_results = {}
    for layer_idx in range(int(max_layer) + 1):
        prefix = f"layer{layer_idx}"
        same_att = same_dataset["stages"][f"{prefix}.att_core_raw"]
        same_gn = same_dataset["stages"][f"{prefix}.gn_no_affine"]
        same_ln = same_dataset["stages"][f"{prefix}.ln_x_out"]
        same_res = same_dataset["stages"][f"{prefix}.residual_term"]
        same_pre = same_dataset["stages"][f"{prefix}.pre_output_sum"]
        next_att = next_dataset["stages"][f"{prefix}.att_core_raw"]
        next_gn = next_dataset["stages"][f"{prefix}.gn_no_affine"]
        next_ln = next_dataset["stages"][f"{prefix}.ln_x_out"]
        next_res = next_dataset["stages"][f"{prefix}.residual_term"]
        next_pre = next_dataset["stages"][f"{prefix}.pre_output_sum"]
        ln_stats = same_dataset["layer_stats"][prefix]

        ln_module = torch.nn.GroupNorm(
            num_groups=int(ln_stats["n_head"]),
            num_channels=int(same_att.shape[-1]),
            eps=float(ln_stats["eps"]),
            affine=True,
        ).to(device=device_obj, dtype=torch.float32)
        ln_module.weight.data.copy_(ln_stats["ln_x_weight"].to(device=device_obj, dtype=torch.float32))
        ln_module.bias.data.copy_(ln_stats["ln_x_bias"].to(device=device_obj, dtype=torch.float32))
        ln_module.eval()

        layer_results[prefix] = {
            "ln_x_affine_summary": {
                "weight_abs_mean": float(ln_stats["ln_x_weight"].abs().mean().item()),
                "weight_mean": float(ln_stats["ln_x_weight"].mean().item()),
                "weight_min": float(ln_stats["ln_x_weight"].min().item()),
                "weight_max": float(ln_stats["ln_x_weight"].max().item()),
                "weight_std": float(ln_stats["ln_x_weight"].std(unbiased=False).item()),
                "bias_abs_mean": float(ln_stats["ln_x_bias"].abs().mean().item()),
                "bias_mean": float(ln_stats["ln_x_bias"].mean().item()),
                "bias_min": float(ln_stats["ln_x_bias"].min().item()),
                "bias_max": float(ln_stats["ln_x_bias"].max().item()),
                "bias_std": float(ln_stats["ln_x_bias"].std(unbiased=False).item()),
                "n_head": int(ln_stats["n_head"]),
                "eps": float(ln_stats["eps"]),
            },
            "stage_metrics": {
                "att_core_raw": {
                    "geometry_summary": _geometry_summary(same_att, same_dataset["objective_mask"]),
                    "feature_target_corr": _feature_target_corr_summary(same_att, same_dataset["returns"], same_dataset["objective_mask"]),
                    "same_rollout": _evaluate_scalar_head(same_att, same_dataset["returns"], same_dataset["objective_mask"]),
                    "next_rollout": _evaluate_scalar_head(next_att, next_dataset["returns"], next_dataset["objective_mask"]),
                },
                "gn_no_affine": {
                    "geometry_summary": _geometry_summary(same_gn, same_dataset["objective_mask"]),
                    "feature_target_corr": _feature_target_corr_summary(same_gn, same_dataset["returns"], same_dataset["objective_mask"]),
                    "same_rollout": _evaluate_scalar_head(same_gn, same_dataset["returns"], same_dataset["objective_mask"]),
                    "next_rollout": _evaluate_scalar_head(next_gn, next_dataset["returns"], next_dataset["objective_mask"]),
                },
                "ln_x_out": {
                    "geometry_summary": _geometry_summary(same_ln, same_dataset["objective_mask"]),
                    "feature_target_corr": _feature_target_corr_summary(same_ln, same_dataset["returns"], same_dataset["objective_mask"]),
                    "same_rollout": _evaluate_scalar_head(same_ln, same_dataset["returns"], same_dataset["objective_mask"]),
                    "next_rollout": _evaluate_scalar_head(next_ln, next_dataset["returns"], next_dataset["objective_mask"]),
                },
                "residual_term": {
                    "geometry_summary": _geometry_summary(same_res, same_dataset["objective_mask"]),
                    "feature_target_corr": _feature_target_corr_summary(same_res, same_dataset["returns"], same_dataset["objective_mask"]),
                    "same_rollout": _evaluate_scalar_head(same_res, same_dataset["returns"], same_dataset["objective_mask"]),
                    "next_rollout": _evaluate_scalar_head(next_res, next_dataset["returns"], next_dataset["objective_mask"]),
                },
                "pre_output_sum": {
                    "geometry_summary": _geometry_summary(same_pre, same_dataset["objective_mask"]),
                    "feature_target_corr": _feature_target_corr_summary(same_pre, same_dataset["returns"], same_dataset["objective_mask"]),
                    "same_rollout": _evaluate_scalar_head(same_pre, same_dataset["returns"], same_dataset["objective_mask"]),
                    "next_rollout": _evaluate_scalar_head(next_pre, next_dataset["returns"], next_dataset["objective_mask"]),
                },
            },
            "direction_transform": _direction_transform_stats(
                att_core_raw=same_att,
                gn_no_affine=same_gn,
                ln_x_out=same_ln,
                returns=same_dataset["returns"],
                mask=same_dataset["objective_mask"],
                ln_x_module=ln_module,
                n_head=int(ln_stats["n_head"]),
            ),
        }

    return {
        "checkpoint_path": str(Path(checkpoint_path).expanduser().resolve()),
        "contract": {
            "frozen_h_seed": int(frozen_h_seed),
            "train_env_seed": int(train_env_seed),
            "train_rollout_seed": int(train_rollout_seed),
            "single_eval_pos": int(single_eval_pos),
            "n_steps": int(n_steps),
            "batch_size": int(batch_size),
            "n_epochs": int(n_epochs),
            "learning_rate": float(learning_rate),
            "target_kl": float(target_kl),
            "build_seed": int(build_seed),
            "max_layer": int(max_layer),
        },
        "layers": layer_results,
    }


def main():
    parser = argparse.ArgumentParser(description="ln_x direction transform probe for Phase 2 critic path.")
    parser.add_argument("checkpoint_path", type=str)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-json", type=str, required=True)
    args = parser.parse_args()

    result = run_phase2_ln_x_direction_transform_probe(
        checkpoint_path=args.checkpoint_path,
        device=args.device,
    )
    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(str(output_path))


if __name__ == "__main__":
    main()
