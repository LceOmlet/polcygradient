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
from ticl.analysis.phase2_attention_output_projection_probe import _weight_stats
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


def _direction_from_head(head: dict) -> torch.Tensor | None:
    weight = head["weight"].detach().cpu().to(dtype=torch.float32).reshape(-1)
    norm = float(torch.linalg.vector_norm(weight).item())
    if norm <= 0.0:
        return None
    return weight / norm


def _safe_cosine(x: torch.Tensor, y: torch.Tensor) -> float:
    if x is None or y is None:
        return 0.0
    x = x.detach().cpu().to(dtype=torch.float32).reshape(-1)
    y = y.detach().cpu().to(dtype=torch.float32).reshape(-1)
    denom = float(torch.linalg.vector_norm(x).item() * torch.linalg.vector_norm(y).item())
    if denom <= 0.0:
        return 0.0
    return float(torch.dot(x, y).item() / denom)


def _project_std(x: torch.Tensor, direction: torch.Tensor, mask: torch.Tensor) -> float:
    if direction is None:
        return 0.0
    mask = mask.reshape(-1).to(dtype=torch.bool)
    x_obj = x[mask].to(dtype=torch.float32)
    centered = x_obj - x_obj.mean(dim=0, keepdim=True)
    proj = (centered @ direction.to(dtype=torch.float32)).detach().cpu().numpy()
    if proj.size <= 1:
        return 0.0
    return float(np.asarray(proj, dtype=np.float32).std())


def _feature_target_corr_summary(x: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> dict:
    mask = mask.reshape(-1).to(dtype=torch.bool)
    x_obj = x[mask].to(dtype=torch.float32).detach().cpu().numpy()
    y = targets[mask].detach().cpu().numpy().reshape(-1)
    if x_obj.shape[0] <= 1:
        return {
            "mean_abs_corr": 0.0,
            "max_abs_corr": 0.0,
            "top5_abs_corr": [],
        }
    corrs = []
    for idx in range(int(x_obj.shape[1])):
        corrs.append(abs(_safe_corrcoef(x_obj[:, idx], y)))
    corrs = np.asarray(corrs, dtype=np.float32)
    order = np.argsort(-corrs)
    return {
        "mean_abs_corr": float(corrs.mean()),
        "max_abs_corr": float(corrs.max()),
        "top5_abs_corr": [float(v) for v in corrs[order[:5]].tolist()],
        "top5_idx": [int(v) for v in order[:5].tolist()],
    }


def _colspace_visibility(weight: torch.Tensor, direction: torch.Tensor) -> dict:
    if direction is None:
        return {
            "visible_mass_in_colspace": 0.0,
            "nullspace_mass": 1.0,
            "gain": 0.0,
            "top_left_singular_alignment_mass": {"1": 0.0, "2": 0.0, "4": 0.0, "8": 0.0, "16": 0.0, "32": 0.0},
            "cos_with_top_left_singular": 0.0,
            "top_singular_value": float(torch.linalg.svdvals(weight.detach().cpu().to(dtype=torch.float32))[0].item()) if weight.numel() > 0 else 0.0,
            "mean_singular_value": float(torch.linalg.svdvals(weight.detach().cpu().to(dtype=torch.float32)).mean().item()) if weight.numel() > 0 else 0.0,
        }
    weight = weight.detach().cpu().to(dtype=torch.float32)
    direction = direction.detach().cpu().to(dtype=torch.float32).reshape(-1)
    u, s, _ = torch.linalg.svd(weight, full_matrices=False)
    proj = u.T @ direction
    proj_mass = float((proj.square().sum()).item())
    gain = float(torch.linalg.vector_norm(weight.T @ direction).item())
    top_mass = {}
    coeff_sq = proj.square()
    coeff_sq = coeff_sq / torch.clamp(coeff_sq.sum(), min=1e-12)
    for k in (1, 2, 4, 8, 16, 32):
        top_mass[str(k)] = float(coeff_sq[: min(k, int(coeff_sq.numel()))].sum().item())
    return {
        "visible_mass_in_colspace": proj_mass,
        "nullspace_mass": float(max(0.0, 1.0 - proj_mass)),
        "gain": gain,
        "top_left_singular_alignment_mass": top_mass,
        "cos_with_top_left_singular": _safe_cosine(direction, u[:, 0]),
        "top_singular_value": float(s[0].item()) if int(s.numel()) > 0 else 0.0,
        "mean_singular_value": float(s.mean().item()) if int(s.numel()) > 0 else 0.0,
    }


def _sum_decomposition(a: torch.Tensor, b: torch.Tensor, total: torch.Tensor, returns: torch.Tensor, mask: torch.Tensor) -> dict:
    head_a = _fit_ridge_scalar_head(a, returns, mask, ridge_lambda=1e-3)
    head_b = _fit_ridge_scalar_head(b, returns, mask, ridge_lambda=1e-3)
    head_total = _fit_ridge_scalar_head(total, returns, mask, ridge_lambda=1e-3)
    d_a = _direction_from_head(head_a)
    d_b = _direction_from_head(head_b)
    d_total = _direction_from_head(head_total)
    mask_bool = mask.reshape(-1).to(dtype=torch.bool)
    a_center = a[mask_bool].to(dtype=torch.float32) - a[mask_bool].to(dtype=torch.float32).mean(dim=0, keepdim=True)
    b_center = b[mask_bool].to(dtype=torch.float32) - b[mask_bool].to(dtype=torch.float32).mean(dim=0, keepdim=True)
    total_center = total[mask_bool].to(dtype=torch.float32) - total[mask_bool].to(dtype=torch.float32).mean(dim=0, keepdim=True)
    a_proj = (a_center @ d_a).detach().cpu().numpy() if d_a is not None else np.zeros((int(a_center.shape[0]),), dtype=np.float32)
    b_proj = (b_center @ d_b).detach().cpu().numpy() if d_b is not None else np.zeros((int(b_center.shape[0]),), dtype=np.float32)
    total_proj = (total_center @ d_total).detach().cpu().numpy() if d_total is not None else np.zeros((int(total_center.shape[0]),), dtype=np.float32)
    y = returns[mask_bool].detach().cpu().numpy()
    return {
        "direction_cosines": {
            "a_vs_b": _safe_cosine(d_a, d_b),
            "a_vs_total": _safe_cosine(d_a, d_total),
            "b_vs_total": _safe_cosine(d_b, d_total),
        },
        "projection_std": {
            "a": float(np.asarray(a_proj, dtype=np.float32).std()) if a_proj.size > 1 else 0.0,
            "b": float(np.asarray(b_proj, dtype=np.float32).std()) if b_proj.size > 1 else 0.0,
            "total": float(np.asarray(total_proj, dtype=np.float32).std()) if total_proj.size > 1 else 0.0,
        },
        "projection_corr_with_target": {
            "a": _safe_corrcoef(a_proj, y),
            "b": _safe_corrcoef(b_proj, y),
            "total": _safe_corrcoef(total_proj, y),
        },
        "projection_corr_between_terms": {
            "a_vs_b": _safe_corrcoef(a_proj, b_proj),
            "a_vs_total": _safe_corrcoef(a_proj, total_proj),
            "b_vs_total": _safe_corrcoef(b_proj, total_proj),
        },
    }


@torch.no_grad()
def _forward_pack_gate_input_stages(policy, bucket_obs: torch.Tensor, *, max_layer: int = 1) -> dict:
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
    layer_stats = {}
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
            delta_xg = xx * block.att.x_g
            xg = x_att + delta_xg

            xr = x_att + xx * block.att.x_r
            xw = x_att + xx * block.att.x_w
            xk = x_att + xx * block.att.x_k
            xv = x_att + xx * block.att.x_v
            xa = x_att + xx * block.att.x_a

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
            gate_pre = xg @ block.att.g1
            gate_sigmoid = torch.sigmoid(gate_pre)
            g = gate_sigmoid @ block.att.g2

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

            stages[f"layer{layer_idx}.x_att"] = x_att.transpose(0, 1)
            stages[f"layer{layer_idx}.xx"] = xx.transpose(0, 1)
            stages[f"layer{layer_idx}.delta_xg"] = delta_xg.transpose(0, 1)
            stages[f"layer{layer_idx}.xg"] = xg.transpose(0, 1)
            stages[f"layer{layer_idx}.gate_pre"] = gate_pre.transpose(0, 1)
            stages[f"layer{layer_idx}.gate_sigmoid"] = gate_sigmoid.transpose(0, 1)
            stages[f"layer{layer_idx}.g"] = g.transpose(0, 1)
            stages[f"layer{layer_idx}.pre_output_sum"] = pre_output_sum.transpose(0, 1)
            stages[f"layer{layer_idx}.pre_gated"] = pre_gated.transpose(0, 1)
            stages[f"layer{layer_idx}.att_out"] = att_out.transpose(0, 1)
            layer_stats[f"layer{layer_idx}"] = {
                "x_g_param": block.att.x_g.detach().cpu().to(dtype=torch.float32).reshape(-1),
                "g1": block.att.g1.detach().cpu().to(dtype=torch.float32),
            }

            x = x + att_out
            x = x + block.ffn(block.ln2(x))

    if pad_len > 0:
        for key in list(stages.keys()):
            stages[key] = stages[key][:seq_len]
    return {
        "stages": stages,
        "layer_stats": layer_stats,
    }


@torch.no_grad()
def _extract_gate_input_dataset(algo, *, max_layer: int = 1) -> dict:
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

            pack_result = _forward_pack_gate_input_stages(algo.policy, bucket_obs, max_layer=int(max_layer))
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
                    layer_stats = pack_result["layer_stats"]
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


def run_phase2_attention_gate_input_alignment_probe(
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
        same_dataset = _extract_gate_input_dataset(algo, max_layer=int(max_layer))
        progress_next = min(int(n_steps), total_timesteps)
        _collect_rollout(algo, callback, vec_env, progress_timestep=progress_next, total_timesteps=total_timesteps)
        next_dataset = _extract_gate_input_dataset(algo, max_layer=int(max_layer))
    finally:
        vec_env.close()

    layer_results = {}
    for layer_idx in range(int(max_layer) + 1):
        prefix = f"layer{layer_idx}"
        same_x_att = same_dataset["stages"][f"{prefix}.x_att"]
        same_xx = same_dataset["stages"][f"{prefix}.xx"]
        same_delta = same_dataset["stages"][f"{prefix}.delta_xg"]
        same_xg = same_dataset["stages"][f"{prefix}.xg"]
        same_gate_pre = same_dataset["stages"][f"{prefix}.gate_pre"]
        same_pre = same_dataset["stages"][f"{prefix}.pre_output_sum"]

        next_x_att = next_dataset["stages"][f"{prefix}.x_att"]
        next_xx = next_dataset["stages"][f"{prefix}.xx"]
        next_delta = next_dataset["stages"][f"{prefix}.delta_xg"]
        next_xg = next_dataset["stages"][f"{prefix}.xg"]
        next_gate_pre = next_dataset["stages"][f"{prefix}.gate_pre"]
        next_pre = next_dataset["stages"][f"{prefix}.pre_output_sum"]

        xg_head = _fit_ridge_scalar_head(same_xg, same_dataset["returns"], same_dataset["objective_mask"], ridge_lambda=1e-3)
        xatt_head = _fit_ridge_scalar_head(same_x_att, same_dataset["returns"], same_dataset["objective_mask"], ridge_lambda=1e-3)
        delta_head = _fit_ridge_scalar_head(same_delta, same_dataset["returns"], same_dataset["objective_mask"], ridge_lambda=1e-3)
        pre_head = _fit_ridge_scalar_head(same_pre, same_dataset["returns"], same_dataset["objective_mask"], ridge_lambda=1e-3)

        d_xg = _direction_from_head(xg_head)
        d_xatt = _direction_from_head(xatt_head)
        d_delta = _direction_from_head(delta_head)
        d_pre = _direction_from_head(pre_head)
        g1 = same_dataset["layer_stats"][prefix]["g1"]
        x_g_param = same_dataset["layer_stats"][prefix]["x_g_param"]

        layer_results[prefix] = {
            "x_g_param_summary": {
                "abs_mean": float(x_g_param.abs().mean().item()),
                "mean": float(x_g_param.mean().item()),
                "min": float(x_g_param.min().item()),
                "max": float(x_g_param.max().item()),
                "std": float(x_g_param.std(unbiased=False).item()),
            },
            "g1_stats": _weight_stats(g1),
            "stage_metrics": {
                "x_att": {
                    "geometry_summary": _geometry_summary(same_x_att, same_dataset["objective_mask"]),
                    "feature_target_corr": _feature_target_corr_summary(same_x_att, same_dataset["returns"], same_dataset["objective_mask"]),
                    "same_rollout": _evaluate_scalar_head(same_x_att, same_dataset["returns"], same_dataset["objective_mask"]),
                    "next_rollout": _evaluate_scalar_head(next_x_att, next_dataset["returns"], next_dataset["objective_mask"]),
                },
                "xx": {
                    "geometry_summary": _geometry_summary(same_xx, same_dataset["objective_mask"]),
                    "feature_target_corr": _feature_target_corr_summary(same_xx, same_dataset["returns"], same_dataset["objective_mask"]),
                    "same_rollout": _evaluate_scalar_head(same_xx, same_dataset["returns"], same_dataset["objective_mask"]),
                    "next_rollout": _evaluate_scalar_head(next_xx, next_dataset["returns"], next_dataset["objective_mask"]),
                },
                "delta_xg": {
                    "geometry_summary": _geometry_summary(same_delta, same_dataset["objective_mask"]),
                    "feature_target_corr": _feature_target_corr_summary(same_delta, same_dataset["returns"], same_dataset["objective_mask"]),
                    "same_rollout": _evaluate_scalar_head(same_delta, same_dataset["returns"], same_dataset["objective_mask"]),
                    "next_rollout": _evaluate_scalar_head(next_delta, next_dataset["returns"], next_dataset["objective_mask"]),
                },
                "xg": {
                    "geometry_summary": _geometry_summary(same_xg, same_dataset["objective_mask"]),
                    "feature_target_corr": _feature_target_corr_summary(same_xg, same_dataset["returns"], same_dataset["objective_mask"]),
                    "same_rollout": _evaluate_scalar_head(same_xg, same_dataset["returns"], same_dataset["objective_mask"]),
                    "next_rollout": _evaluate_scalar_head(next_xg, next_dataset["returns"], next_dataset["objective_mask"]),
                },
                "gate_pre": {
                    "geometry_summary": _geometry_summary(same_gate_pre, same_dataset["objective_mask"]),
                    "feature_target_corr": _feature_target_corr_summary(same_gate_pre, same_dataset["returns"], same_dataset["objective_mask"]),
                    "same_rollout": _evaluate_scalar_head(same_gate_pre, same_dataset["returns"], same_dataset["objective_mask"]),
                    "next_rollout": _evaluate_scalar_head(next_gate_pre, next_dataset["returns"], next_dataset["objective_mask"]),
                },
                "pre_output_sum": {
                    "geometry_summary": _geometry_summary(same_pre, same_dataset["objective_mask"]),
                    "feature_target_corr": _feature_target_corr_summary(same_pre, same_dataset["returns"], same_dataset["objective_mask"]),
                    "same_rollout": _evaluate_scalar_head(same_pre, same_dataset["returns"], same_dataset["objective_mask"]),
                    "next_rollout": _evaluate_scalar_head(next_pre, next_dataset["returns"], next_dataset["objective_mask"]),
                },
            },
            "xg_decomposition": _sum_decomposition(
                same_x_att,
                same_delta,
                same_xg,
                same_dataset["returns"],
                same_dataset["objective_mask"],
            ),
            "g1_visibility": {
                "x_att_direction": _colspace_visibility(g1, d_xatt),
                "delta_xg_direction": _colspace_visibility(g1, d_delta),
                "xg_direction": _colspace_visibility(g1, d_xg),
                "pre_output_sum_direction": _colspace_visibility(g1, d_pre),
                "direction_cosines": {
                    "x_att_vs_xg": _safe_cosine(d_xatt, d_xg),
                    "delta_vs_xg": _safe_cosine(d_delta, d_xg),
                    "xg_vs_pre": _safe_cosine(d_xg, d_pre),
                    "x_att_vs_pre": _safe_cosine(d_xatt, d_pre),
                    "delta_vs_pre": _safe_cosine(d_delta, d_pre),
                },
                "projection_std": {
                    "x_att": _project_std(same_x_att, d_xatt, same_dataset["objective_mask"]),
                    "delta_xg": _project_std(same_delta, d_delta, same_dataset["objective_mask"]),
                    "xg": _project_std(same_xg, d_xg, same_dataset["objective_mask"]),
                    "pre_output_sum": _project_std(same_pre, d_pre, same_dataset["objective_mask"]),
                },
            },
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
    parser = argparse.ArgumentParser(description="Gate-input alignment probe for Phase 2 critic attention branch.")
    parser.add_argument("checkpoint_path", type=str)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-json", type=str, required=True)
    args = parser.parse_args()

    result = run_phase2_attention_gate_input_alignment_probe(
        checkpoint_path=args.checkpoint_path,
        device=args.device,
    )
    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(str(output_path))


if __name__ == "__main__":
    main()
