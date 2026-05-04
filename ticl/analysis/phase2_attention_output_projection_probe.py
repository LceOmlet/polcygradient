import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ticl.analysis.phase2_attention_internal_geometry_probe import (
    _evaluate_scalar_head,
    _geometry_summary,
    _safe_corrcoef,
)
from ticl.analysis.phase2_hidden_geometry_whitening_probe import (
    _apply_raw_transform,
    _apply_zscore_transform,
    _evaluate_variant_on_transform,
    _fit_linear_geometry,
    _fit_official_distribution_head,
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


def _direction_from_head(head: dict) -> torch.Tensor:
    weight = head["weight"].detach().cpu().to(dtype=torch.float32).reshape(-1)
    norm = float(torch.linalg.vector_norm(weight).item())
    if norm <= 0.0:
        raise RuntimeError("Zero-norm direction.")
    return weight / norm


def _safe_std(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.size <= 1:
        return 0.0
    return float(x.std())


def _safe_cosine(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.detach().cpu().to(dtype=torch.float32).reshape(-1)
    y = y.detach().cpu().to(dtype=torch.float32).reshape(-1)
    denom = float(torch.linalg.vector_norm(x).item() * torch.linalg.vector_norm(y).item())
    if denom <= 0.0:
        return 0.0
    return float(torch.dot(x, y).item() / denom)


def _project_stats(x: torch.Tensor, direction: torch.Tensor, mask: torch.Tensor) -> dict:
    mask = mask.reshape(-1).to(dtype=torch.bool)
    x_obj = x[mask].to(dtype=torch.float32)
    centered = x_obj - x_obj.mean(dim=0, keepdim=True)
    proj = (centered @ direction.to(dtype=torch.float32)).detach().cpu().numpy()
    return {
        "projection_std": _safe_std(proj),
        "projection_mean_abs": float(np.asarray(np.abs(proj), dtype=np.float32).mean()) if proj.size > 0 else 0.0,
    }


def _weight_stats(weight: torch.Tensor) -> dict:
    weight = weight.detach().cpu().to(dtype=torch.float32)
    row_norm = torch.linalg.vector_norm(weight, dim=1)
    col_norm = torch.linalg.vector_norm(weight, dim=0)
    singular_values = torch.linalg.svdvals(weight)
    nonzero = singular_values[singular_values > 1e-12]
    cond = float((float(nonzero.max().item()) / float(nonzero.min().item())) if int(nonzero.numel()) > 0 else 0.0)
    fro = float(torch.linalg.vector_norm(weight).item())
    return {
        "shape": [int(v) for v in weight.shape],
        "row_norm_mean": float(row_norm.mean().item()),
        "row_norm_min": float(row_norm.min().item()),
        "row_norm_max": float(row_norm.max().item()),
        "col_norm_mean": float(col_norm.mean().item()),
        "col_norm_min": float(col_norm.min().item()),
        "col_norm_max": float(col_norm.max().item()),
        "fro_norm": fro,
        "singular_values_top8": [float(v) for v in singular_values[:8].tolist()],
        "singular_value_min_nonzero": float(nonzero.min().item()) if int(nonzero.numel()) > 0 else 0.0,
        "singular_value_max": float(nonzero.max().item()) if int(nonzero.numel()) > 0 else 0.0,
        "condition_nonzero": cond,
        "mean_singular_value": float(singular_values.mean().item()),
    }


@torch.no_grad()
def _forward_pack_output_projection_stages(policy, bucket_obs: torch.Tensor, *, max_layer: int = 1) -> dict:
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
    v_first = None
    g_stats = {}
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
            residual_term = (
                (r.view(bsz, seq_work_len, block.n_head, -1) * k.view(bsz, seq_work_len, block.n_head, -1) * block.att.r_k)
                .sum(dim=-1, keepdim=True)
                * v.view(bsz, seq_work_len, block.n_head, -1)
            ).view(bsz, seq_work_len, channels)
            pre_output_sum = ln_x_out + residual_term
            pre_gated = pre_output_sum * g
            att_out = block.att.output(pre_gated)

            stages[f"layer{layer_idx}.pre_output_sum"] = pre_output_sum.transpose(0, 1)
            stages[f"layer{layer_idx}.pre_gated"] = pre_gated.transpose(0, 1)
            stages[f"layer{layer_idx}.att_out"] = att_out.transpose(0, 1)
            g_stats[f"layer{layer_idx}.g"] = g.transpose(0, 1)

            x = x + att_out
            x = x + block.ffn(block.ln2(x))

    if pad_len > 0:
        for key in list(stages.keys()):
            stages[key] = stages[key][:seq_len]
        for key in list(g_stats.keys()):
            g_stats[key] = g_stats[key][:seq_len]
    return {
        "stages": stages,
        "gates": g_stats,
    }


@torch.no_grad()
def _extract_output_projection_dataset(algo, *, max_layer: int = 1) -> dict:
    rollout_buffer = algo.rollout_buffer
    stage_latents = None
    gate_latents = None
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
        gate_flat = None
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

            pack_result = _forward_pack_output_projection_stages(algo.policy, bucket_obs, max_layer=int(max_layer))
            stage_pack = pack_result["stages"]
            gate_pack = pack_result["gates"]
            if stage_flat is None:
                stage_dim = int(next(iter(stage_pack.values())).shape[-1])
                gate_dim = int(next(iter(gate_pack.values())).shape[-1])
                stage_flat = {
                    key: torch.empty(
                        (int(obs.shape[0]), stage_dim),
                        device=value.device,
                        dtype=value.dtype,
                    )
                    for key, value in stage_pack.items()
                }
                gate_flat = {
                    key: torch.empty(
                        (int(obs.shape[0]), gate_dim),
                        device=value.device,
                        dtype=value.dtype,
                    )
                    for key, value in gate_pack.items()
                }
                if stage_latents is None:
                    stage_latents = {key: [] for key in stage_pack.keys()}
                    gate_latents = {key: [] for key in gate_pack.keys()}
            for pack_col, seq_idx in enumerate(pack_seq_indices):
                seg_start = int(seq_start_indices[int(seq_idx)])
                seg_len = int(seq_lengths[int(seq_idx)])
                for key in stage_pack.keys():
                    stage_flat[key][seg_start : seg_start + seg_len].copy_(stage_pack[key][:seg_len, pack_col])
                for key in gate_pack.keys():
                    gate_flat[key][seg_start : seg_start + seg_len].copy_(gate_pack[key][:seg_len, pack_col])
            cursor += len(pack_seq_indices)

        for key in stage_latents.keys():
            stage_latents[key].append(stage_flat[key].detach().cpu())
        for key in gate_latents.keys():
            gate_latents[key].append(gate_flat[key].detach().cpu())
        targets.append(rollout_data.returns.detach().cpu().reshape(-1))
        masks.append((rollout_data.objective_masks > 1e-8).detach().cpu().reshape(-1).to(dtype=torch.bool))

    return {
        "stages": {key: torch.cat(value, dim=0) for key, value in stage_latents.items()},
        "gates": {key: torch.cat(value, dim=0) for key, value in gate_latents.items()},
        "returns": torch.cat(targets, dim=0),
        "objective_mask": torch.cat(masks, dim=0),
    }


def _make_scale_adapter(reference_x: torch.Tensor, target_x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> dict:
    mask = mask.reshape(-1).to(dtype=torch.bool)
    ref_obj = reference_x[mask].to(dtype=torch.float32)
    tgt_obj = target_x[mask].to(dtype=torch.float32)
    ref_std = ref_obj.std(dim=0, unbiased=False)
    tgt_std = tgt_obj.std(dim=0, unbiased=False).clamp_min(float(eps))
    scale = ref_std / tgt_std
    return {
        "scale": scale.detach().cpu(),
        "reference_feature_std_mean": float(ref_std.mean().item()),
        "target_feature_std_mean": float(tgt_std.mean().item()),
        "scale_mean": float(scale.mean().item()),
        "scale_min": float(scale.min().item()),
        "scale_max": float(scale.max().item()),
    }


def _apply_scale_adapter(x: torch.Tensor, adapter: dict) -> torch.Tensor:
    return x.to(dtype=torch.float32) * adapter["scale"].to(dtype=torch.float32)


def _direction_transport_stats(
    *,
    pre_output_sum: torch.Tensor,
    pre_gated: torch.Tensor,
    att_out: torch.Tensor,
    returns: torch.Tensor,
    mask: torch.Tensor,
    output_weight: torch.Tensor,
) -> dict:
    pre_head = _fit_ridge_scalar_head(pre_output_sum, returns, mask, ridge_lambda=1e-3)
    gated_head = _fit_ridge_scalar_head(pre_gated, returns, mask, ridge_lambda=1e-3)
    att_head = _fit_ridge_scalar_head(att_out, returns, mask, ridge_lambda=1e-3)
    d_pre = _direction_from_head(pre_head)
    d_gated = _direction_from_head(gated_head)
    d_att = _direction_from_head(att_head)

    weight = output_weight.detach().cpu().to(dtype=torch.float32)
    wd_pre = weight @ d_pre
    wd_gated = weight @ d_gated
    wd_pre_norm = wd_pre / torch.clamp(torch.linalg.vector_norm(wd_pre), min=1e-12)
    wd_gated_norm = wd_gated / torch.clamp(torch.linalg.vector_norm(wd_gated), min=1e-12)
    wt_datt = weight.T @ d_att
    wt_datt_norm = wt_datt / torch.clamp(torch.linalg.vector_norm(wt_datt), min=1e-12)

    pre_proj = _project_stats(pre_output_sum, d_pre, mask)
    gated_proj_on_pre = _project_stats(pre_gated, d_pre, mask)
    gated_proj = _project_stats(pre_gated, d_gated, mask)
    att_proj_from_pre = _project_stats(att_out, wd_pre_norm, mask)
    att_proj_from_gated = _project_stats(att_out, wd_gated_norm, mask)
    att_proj = _project_stats(att_out, d_att, mask)

    mask_np = mask.reshape(-1).cpu().numpy().astype(bool)
    pre_center = (pre_output_sum[mask].to(dtype=torch.float32) - pre_output_sum[mask].to(dtype=torch.float32).mean(dim=0, keepdim=True))
    gated_center = (pre_gated[mask].to(dtype=torch.float32) - pre_gated[mask].to(dtype=torch.float32).mean(dim=0, keepdim=True))
    att_center = (att_out[mask].to(dtype=torch.float32) - att_out[mask].to(dtype=torch.float32).mean(dim=0, keepdim=True))
    z_pre = (pre_center @ d_pre).detach().cpu().numpy()
    z_gated_same_dir = (gated_center @ d_pre).detach().cpu().numpy()
    z_att_from_pre = (att_center @ wd_pre_norm).detach().cpu().numpy()
    z_att = (att_center @ d_att).detach().cpu().numpy()
    return {
        "pre_direction": {
            "same_rollout": _evaluate_scalar_head(pre_output_sum, returns, mask),
            "projection_stats": pre_proj,
        },
        "gating_effect": {
            "gated_same_direction_projection": gated_proj_on_pre,
            "gated_best_direction_projection": gated_proj,
            "cos_pre_vs_gated_direction": _safe_cosine(d_pre, d_gated),
            "corr_pre_proj_vs_gated_same_dir": _safe_corrcoef(z_pre, z_gated_same_dir),
            "std_ratio_gated_same_dir_vs_pre": float(_safe_std(z_gated_same_dir) / max(_safe_std(z_pre), 1e-12)),
        },
        "output_projection_effect": {
            "gain_on_pre_direction": float(torch.linalg.vector_norm(wd_pre).item()),
            "gain_on_gated_direction": float(torch.linalg.vector_norm(wd_gated).item()),
            "cos_W_pre_vs_att_direction": _safe_cosine(wd_pre_norm, d_att),
            "cos_W_gated_vs_att_direction": _safe_cosine(wd_gated_norm, d_att),
            "cos_pre_vs_WT_att_direction": _safe_cosine(d_pre, wt_datt_norm),
            "corr_pre_proj_vs_att_proj_from_pre": _safe_corrcoef(z_pre, z_att_from_pre),
            "std_ratio_att_proj_from_pre_vs_pre": float(_safe_std(z_att_from_pre) / max(_safe_std(z_pre), 1e-12)),
            "att_projection_from_pre_direction": att_proj_from_pre,
            "att_projection_from_gated_direction": att_proj_from_gated,
            "att_best_direction_projection": att_proj,
            "corr_att_best_proj_vs_target": _safe_corrcoef(z_att, returns[mask].detach().cpu().numpy()),
        },
    }


def _fit_stage_variants(
    *,
    same_x: torch.Tensor,
    next_x: torch.Tensor,
    returns_same: torch.Tensor,
    returns_next: torch.Tensor,
    mask_same: torch.Tensor,
    mask_next: torch.Tensor,
    device: torch.device,
    num_buckets: int,
    value_range: float,
    fit_steps: int,
    lr: float,
    max_grad_norm: float,
) -> dict:
    geom = _fit_linear_geometry(same_x, mask_same)
    raw_same = _apply_raw_transform(same_x, geom)
    raw_next = _apply_raw_transform(next_x, geom)
    zscore_same = _apply_zscore_transform(same_x, geom)
    zscore_next = _apply_zscore_transform(next_x, geom)
    raw_variant = _fit_official_distribution_head(
        x=raw_same[mask_same].to(device=device, dtype=torch.float32),
        y=returns_same[mask_same].to(device=device, dtype=torch.float32),
        latent_dim=int(raw_same.shape[-1]),
        num_buckets=int(num_buckets),
        value_range=float(value_range),
        fit_steps=int(fit_steps),
        lr=float(lr),
        max_grad_norm=float(max_grad_norm),
        device=device,
    )
    zscore_variant = _fit_official_distribution_head(
        x=zscore_same[mask_same].to(device=device, dtype=torch.float32),
        y=returns_same[mask_same].to(device=device, dtype=torch.float32),
        latent_dim=int(zscore_same.shape[-1]),
        num_buckets=int(num_buckets),
        value_range=float(value_range),
        fit_steps=int(fit_steps),
        lr=float(lr),
        max_grad_norm=float(max_grad_norm),
        device=device,
    )
    same_dataset = {"returns": returns_same, "objective_mask": mask_same}
    next_dataset = {"returns": returns_next, "objective_mask": mask_next}
    return {
        "geometry_summary": _geometry_summary(same_x, mask_same),
        "raw_scalar_ridge": {
            "same_rollout": _evaluate_scalar_head(same_x, returns_same, mask_same),
            "next_rollout": _evaluate_scalar_head(next_x, returns_next, mask_next),
        },
        "distribution_head_metrics": {
            "raw_hidden": _evaluate_variant_on_transform(
                variant=raw_variant,
                same_x=raw_same,
                next_x=raw_next,
                same_dataset=same_dataset,
                next_dataset=next_dataset,
                device=device,
            ),
            "frozen_zscore_adapter": _evaluate_variant_on_transform(
                variant=zscore_variant,
                same_x=zscore_same,
                next_x=zscore_next,
                same_dataset=same_dataset,
                next_dataset=next_dataset,
                device=device,
            ),
        },
    }


def run_phase2_attention_output_projection_probe(
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
    fit_steps: int = 20,
    head_fit_learning_rate: float = 1e-3,
    head_fit_max_grad_norm: float = 0.5,
    num_buckets: int = 31,
    value_range: float = 5.0,
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
        same_dataset = _extract_output_projection_dataset(algo, max_layer=int(max_layer))
        progress_next = min(int(n_steps), total_timesteps)
        _collect_rollout(algo, callback, vec_env, progress_timestep=progress_next, total_timesteps=total_timesteps)
        next_dataset = _extract_output_projection_dataset(algo, max_layer=int(max_layer))

        layer_results = {}
        for layer_idx in range(int(max_layer) + 1):
            pre_key = f"layer{layer_idx}.pre_output_sum"
            gated_key = f"layer{layer_idx}.pre_gated"
            att_key = f"layer{layer_idx}.att_out"
            gate_key = f"layer{layer_idx}.g"
            same_pre = same_dataset["stages"][pre_key]
            same_gated = same_dataset["stages"][gated_key]
            same_att = same_dataset["stages"][att_key]
            next_pre = next_dataset["stages"][pre_key]
            next_gated = next_dataset["stages"][gated_key]
            next_att = next_dataset["stages"][att_key]
            same_gate = same_dataset["gates"][gate_key]
            output_weight = algo.policy.rlpfn_model.rwkv_core.blocks[layer_idx].att.output.weight.detach().cpu()
            scale_adapter = _make_scale_adapter(
                reference_x=same_gated,
                target_x=same_att,
                mask=same_dataset["objective_mask"],
            )
            same_att_scaled = _apply_scale_adapter(same_att, scale_adapter)
            next_att_scaled = _apply_scale_adapter(next_att, scale_adapter)

            pre_metrics = _fit_stage_variants(
                same_x=same_pre,
                next_x=next_pre,
                returns_same=same_dataset["returns"],
                returns_next=next_dataset["returns"],
                mask_same=same_dataset["objective_mask"],
                mask_next=next_dataset["objective_mask"],
                device=device_obj,
                num_buckets=int(num_buckets),
                value_range=float(value_range),
                fit_steps=int(fit_steps),
                lr=float(head_fit_learning_rate),
                max_grad_norm=float(head_fit_max_grad_norm),
            )
            gated_metrics = _fit_stage_variants(
                same_x=same_gated,
                next_x=next_gated,
                returns_same=same_dataset["returns"],
                returns_next=next_dataset["returns"],
                mask_same=same_dataset["objective_mask"],
                mask_next=next_dataset["objective_mask"],
                device=device_obj,
                num_buckets=int(num_buckets),
                value_range=float(value_range),
                fit_steps=int(fit_steps),
                lr=float(head_fit_learning_rate),
                max_grad_norm=float(head_fit_max_grad_norm),
            )
            att_metrics = _fit_stage_variants(
                same_x=same_att,
                next_x=next_att,
                returns_same=same_dataset["returns"],
                returns_next=next_dataset["returns"],
                mask_same=same_dataset["objective_mask"],
                mask_next=next_dataset["objective_mask"],
                device=device_obj,
                num_buckets=int(num_buckets),
                value_range=float(value_range),
                fit_steps=int(fit_steps),
                lr=float(head_fit_learning_rate),
                max_grad_norm=float(head_fit_max_grad_norm),
            )
            scaled_att_metrics = _fit_stage_variants(
                same_x=same_att_scaled,
                next_x=next_att_scaled,
                returns_same=same_dataset["returns"],
                returns_next=next_dataset["returns"],
                mask_same=same_dataset["objective_mask"],
                mask_next=next_dataset["objective_mask"],
                device=device_obj,
                num_buckets=int(num_buckets),
                value_range=float(value_range),
                fit_steps=int(fit_steps),
                lr=float(head_fit_learning_rate),
                max_grad_norm=float(head_fit_max_grad_norm),
            )

            gate_obj = same_gate[same_dataset["objective_mask"]].to(dtype=torch.float32)
            layer_results[f"layer{layer_idx}"] = {
                "gate_summary": {
                    "feature_mean_abs_mean": float(gate_obj.abs().mean().item()),
                    "feature_mean_mean": float(gate_obj.mean().item()),
                    "feature_std_mean": float(gate_obj.std(dim=0, unbiased=False).mean().item()),
                    "feature_min": float(gate_obj.min().item()),
                    "feature_max": float(gate_obj.max().item()),
                },
                "output_weight_stats": _weight_stats(output_weight),
                "direction_transport": _direction_transport_stats(
                    pre_output_sum=same_pre,
                    pre_gated=same_gated,
                    att_out=same_att,
                    returns=same_dataset["returns"],
                    mask=same_dataset["objective_mask"],
                    output_weight=output_weight,
                ),
                "stage_metrics": {
                    "pre_output_sum": pre_metrics,
                    "pre_gated": gated_metrics,
                    "att_out": att_metrics,
                "att_out_frozen_scale_adapter": {
                        **scaled_att_metrics,
                        "scale_adapter_summary": {
                            "reference_feature_std_mean": float(scale_adapter["reference_feature_std_mean"]),
                            "target_feature_std_mean": float(scale_adapter["target_feature_std_mean"]),
                            "scale_mean": float(scale_adapter["scale_mean"]),
                            "scale_min": float(scale_adapter["scale_min"]),
                            "scale_max": float(scale_adapter["scale_max"]),
                        },
                    },
                },
            }
    finally:
        vec_env.close()

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
            "fit_steps": int(fit_steps),
            "num_buckets": int(num_buckets),
            "value_range": float(value_range),
            "max_layer": int(max_layer),
            "geometry_adapter": "analysis_only_frozen_feature_scale_and_zscore",
        },
        "layers": layer_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Output projection bottleneck probe for Phase 2 critic attention path.")
    parser.add_argument("checkpoint_path", type=str)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-json", type=str, required=True)
    args = parser.parse_args()

    result = run_phase2_attention_output_projection_probe(
        checkpoint_path=args.checkpoint_path,
        device=args.device,
    )
    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(str(output_path))


if __name__ == "__main__":
    main()
