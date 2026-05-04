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
from ticl.analysis.phase2_attention_output_projection_probe import (
    _direction_from_head,
    _project_stats,
    _safe_cosine,
    _safe_std,
    _weight_stats,
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


@torch.no_grad()
def _forward_pack_gate_path_stages(policy, bucket_obs: torch.Tensor, *, max_layer: int = 1) -> dict:
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

            stages[f"layer{layer_idx}.xg"] = xg.transpose(0, 1)
            stages[f"layer{layer_idx}.gate_pre"] = gate_pre.transpose(0, 1)
            stages[f"layer{layer_idx}.gate_sigmoid"] = gate_sigmoid.transpose(0, 1)
            stages[f"layer{layer_idx}.g"] = g.transpose(0, 1)
            stages[f"layer{layer_idx}.pre_output_sum"] = pre_output_sum.transpose(0, 1)
            stages[f"layer{layer_idx}.pre_gated"] = pre_gated.transpose(0, 1)
            stages[f"layer{layer_idx}.att_out"] = att_out.transpose(0, 1)
            stats[f"layer{layer_idx}.gate_weights"] = {
                "g1": block.att.g1.detach().cpu().to(dtype=torch.float32),
                "g2": block.att.g2.detach().cpu().to(dtype=torch.float32),
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
def _extract_gate_path_dataset(algo, *, max_layer: int = 1) -> dict:
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

            pack_result = _forward_pack_gate_path_stages(algo.policy, bucket_obs, max_layer=int(max_layer))
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


def _gate_activation_summary(gate_pre: torch.Tensor, gate_sigmoid: torch.Tensor, g: torch.Tensor, mask: torch.Tensor) -> dict:
    mask = mask.reshape(-1).to(dtype=torch.bool)
    pre = gate_pre[mask].to(dtype=torch.float32)
    sig = gate_sigmoid[mask].to(dtype=torch.float32)
    g_obj = g[mask].to(dtype=torch.float32)
    deriv = sig * (1.0 - sig)
    frac_low = float((sig < 0.01).to(dtype=torch.float32).mean().item())
    frac_high = float((sig > 0.99).to(dtype=torch.float32).mean().item())
    return {
        "gate_pre_mean": float(pre.mean().item()),
        "gate_pre_std_mean": float(pre.std(dim=0, unbiased=False).mean().item()),
        "gate_pre_min": float(pre.min().item()),
        "gate_pre_max": float(pre.max().item()),
        "gate_sigmoid_mean": float(sig.mean().item()),
        "gate_sigmoid_std_mean": float(sig.std(dim=0, unbiased=False).mean().item()),
        "gate_sigmoid_min": float(sig.min().item()),
        "gate_sigmoid_max": float(sig.max().item()),
        "sigmoid_derivative_mean": float(deriv.mean().item()),
        "sigmoid_derivative_std_mean": float(deriv.std(dim=0, unbiased=False).mean().item()),
        "sigmoid_frac_lt_0_01": frac_low,
        "sigmoid_frac_gt_0_99": frac_high,
        "g_mean": float(g_obj.mean().item()),
        "g_abs_mean": float(g_obj.abs().mean().item()),
        "g_std_mean": float(g_obj.std(dim=0, unbiased=False).mean().item()),
        "g_min": float(g_obj.min().item()),
        "g_max": float(g_obj.max().item()),
    }


def _gate_directional_erosion(
    *,
    xg: torch.Tensor,
    gate_pre: torch.Tensor,
    gate_sigmoid: torch.Tensor,
    g: torch.Tensor,
    pre_output_sum: torch.Tensor,
    pre_gated: torch.Tensor,
    att_out: torch.Tensor,
    returns: torch.Tensor,
    mask: torch.Tensor,
) -> dict:
    pre_head = _fit_ridge_scalar_head(pre_output_sum, returns, mask, ridge_lambda=1e-3)
    xg_head = _fit_ridge_scalar_head(xg, returns, mask, ridge_lambda=1e-3)
    gate_pre_head = _fit_ridge_scalar_head(gate_pre, returns, mask, ridge_lambda=1e-3)
    gate_sig_head = _fit_ridge_scalar_head(gate_sigmoid, returns, mask, ridge_lambda=1e-3)
    g_head = _fit_ridge_scalar_head(g, returns, mask, ridge_lambda=1e-3)
    pre_gated_head = _fit_ridge_scalar_head(pre_gated, returns, mask, ridge_lambda=1e-3)
    att_head = _fit_ridge_scalar_head(att_out, returns, mask, ridge_lambda=1e-3)

    d_pre = _direction_from_head(pre_head)
    d_xg = _direction_from_head(xg_head)
    d_gate_pre = _direction_from_head(gate_pre_head)
    d_gate_sig = _direction_from_head(gate_sig_head)
    d_g = _direction_from_head(g_head)
    d_pre_gated = _direction_from_head(pre_gated_head)
    d_att = _direction_from_head(att_head)

    mask_bool = mask.reshape(-1).to(dtype=torch.bool)
    pre_obj = pre_output_sum[mask_bool].to(dtype=torch.float32)
    gated_obj = pre_gated[mask_bool].to(dtype=torch.float32)
    g_obj = g[mask_bool].to(dtype=torch.float32)
    returns_np = returns[mask_bool].detach().cpu().numpy()

    pre_center = pre_obj - pre_obj.mean(dim=0, keepdim=True)
    gated_center = gated_obj - gated_obj.mean(dim=0, keepdim=True)
    pre_proj = (pre_center @ d_pre).detach().cpu().numpy()
    gated_same_dir = (gated_center @ d_pre).detach().cpu().numpy()
    gated_best_dir = (gated_center @ d_pre_gated).detach().cpu().numpy()
    att_proj = ((att_out[mask_bool].to(dtype=torch.float32) - att_out[mask_bool].to(dtype=torch.float32).mean(dim=0, keepdim=True)) @ d_att).detach().cpu().numpy()

    dir_gate_gain = (g_obj * (d_pre.to(dtype=torch.float32) ** 2)).sum(dim=-1).detach().cpu().numpy()
    valid = np.abs(pre_proj) > 1e-8
    empirical_ratio = np.zeros_like(pre_proj, dtype=np.float32)
    empirical_ratio[valid] = gated_same_dir[valid] / pre_proj[valid]
    empirical_ratio_valid = empirical_ratio[valid]

    return {
        "direction_cosines": {
            "xg_vs_pre": _safe_cosine(d_xg, d_pre),
            "g_vs_pre": _safe_cosine(d_g, d_pre),
            "pre_gated_vs_pre": _safe_cosine(d_pre_gated, d_pre),
            "att_out_vs_pre": _safe_cosine(d_att, d_pre),
            "gate_sigmoid_vs_gate_pre": _safe_cosine(d_gate_sig, d_gate_pre),
        },
        "projection_preservation": {
            "pre_direction_stats": _project_stats(pre_output_sum, d_pre, mask),
            "gated_same_direction_stats": _project_stats(pre_gated, d_pre, mask),
            "gated_best_direction_stats": _project_stats(pre_gated, d_pre_gated, mask),
            "att_best_direction_stats": _project_stats(att_out, d_att, mask),
            "corr_pre_proj_vs_gated_same_dir": _safe_corrcoef(pre_proj, gated_same_dir),
            "corr_pre_proj_vs_gated_best_dir": _safe_corrcoef(pre_proj, gated_best_dir),
            "corr_gated_best_dir_vs_target": _safe_corrcoef(gated_best_dir, returns_np),
            "corr_att_best_dir_vs_target": _safe_corrcoef(att_proj, returns_np),
            "std_ratio_gated_same_dir_vs_pre": float(_safe_std(gated_same_dir) / max(_safe_std(pre_proj), 1e-12)),
            "std_ratio_gated_best_dir_vs_pre": float(_safe_std(gated_best_dir) / max(_safe_std(pre_proj), 1e-12)),
            "std_ratio_att_best_dir_vs_pre": float(_safe_std(att_proj) / max(_safe_std(pre_proj), 1e-12)),
        },
        "gate_gain_along_pre_direction": {
            "weighted_gain_mean": float(np.asarray(dir_gate_gain, dtype=np.float32).mean()),
            "weighted_gain_std": _safe_std(dir_gate_gain),
            "weighted_gain_min": float(np.asarray(dir_gate_gain, dtype=np.float32).min()) if dir_gate_gain.size > 0 else 0.0,
            "weighted_gain_max": float(np.asarray(dir_gate_gain, dtype=np.float32).max()) if dir_gate_gain.size > 0 else 0.0,
            "corr_weighted_gain_vs_target": _safe_corrcoef(dir_gate_gain, returns_np),
            "empirical_ratio_mean": float(np.asarray(empirical_ratio_valid, dtype=np.float32).mean()) if empirical_ratio_valid.size > 0 else 0.0,
            "empirical_ratio_std": _safe_std(empirical_ratio_valid),
            "empirical_ratio_min": float(np.asarray(empirical_ratio_valid, dtype=np.float32).min()) if empirical_ratio_valid.size > 0 else 0.0,
            "empirical_ratio_max": float(np.asarray(empirical_ratio_valid, dtype=np.float32).max()) if empirical_ratio_valid.size > 0 else 0.0,
            "corr_empirical_ratio_vs_target": _safe_corrcoef(empirical_ratio_valid, returns_np[valid]) if int(valid.sum()) > 1 else 0.0,
        },
    }


def run_phase2_attention_gate_path_probe(
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
        same_dataset = _extract_gate_path_dataset(algo, max_layer=int(max_layer))
        progress_next = min(int(n_steps), total_timesteps)
        _collect_rollout(algo, callback, vec_env, progress_timestep=progress_next, total_timesteps=total_timesteps)
        next_dataset = _extract_gate_path_dataset(algo, max_layer=int(max_layer))
    finally:
        vec_env.close()

    layer_results = {}
    for layer_idx in range(int(max_layer) + 1):
        prefix = f"layer{layer_idx}"
        same_xg = same_dataset["stages"][f"{prefix}.xg"]
        same_gate_pre = same_dataset["stages"][f"{prefix}.gate_pre"]
        same_gate_sig = same_dataset["stages"][f"{prefix}.gate_sigmoid"]
        same_g = same_dataset["stages"][f"{prefix}.g"]
        same_pre = same_dataset["stages"][f"{prefix}.pre_output_sum"]
        same_pre_gated = same_dataset["stages"][f"{prefix}.pre_gated"]
        same_att = same_dataset["stages"][f"{prefix}.att_out"]

        next_xg = next_dataset["stages"][f"{prefix}.xg"]
        next_gate_pre = next_dataset["stages"][f"{prefix}.gate_pre"]
        next_gate_sig = next_dataset["stages"][f"{prefix}.gate_sigmoid"]
        next_g = next_dataset["stages"][f"{prefix}.g"]
        next_pre = next_dataset["stages"][f"{prefix}.pre_output_sum"]
        next_pre_gated = next_dataset["stages"][f"{prefix}.pre_gated"]
        next_att = next_dataset["stages"][f"{prefix}.att_out"]

        weight_pack = same_dataset["layer_stats"][f"{prefix}.gate_weights"]
        layer_results[prefix] = {
            "gate_weight_stats": {
                "g1": _weight_stats(weight_pack["g1"]),
                "g2": _weight_stats(weight_pack["g2"]),
            },
            "stage_metrics": {
                "xg": {
                    "geometry_summary": _geometry_summary(same_xg, same_dataset["objective_mask"]),
                    "same_rollout": _evaluate_scalar_head(same_xg, same_dataset["returns"], same_dataset["objective_mask"]),
                    "next_rollout": _evaluate_scalar_head(next_xg, next_dataset["returns"], next_dataset["objective_mask"]),
                },
                "gate_pre": {
                    "geometry_summary": _geometry_summary(same_gate_pre, same_dataset["objective_mask"]),
                    "same_rollout": _evaluate_scalar_head(same_gate_pre, same_dataset["returns"], same_dataset["objective_mask"]),
                    "next_rollout": _evaluate_scalar_head(next_gate_pre, next_dataset["returns"], next_dataset["objective_mask"]),
                },
                "gate_sigmoid": {
                    "geometry_summary": _geometry_summary(same_gate_sig, same_dataset["objective_mask"]),
                    "same_rollout": _evaluate_scalar_head(same_gate_sig, same_dataset["returns"], same_dataset["objective_mask"]),
                    "next_rollout": _evaluate_scalar_head(next_gate_sig, next_dataset["returns"], next_dataset["objective_mask"]),
                },
                "g": {
                    "geometry_summary": _geometry_summary(same_g, same_dataset["objective_mask"]),
                    "same_rollout": _evaluate_scalar_head(same_g, same_dataset["returns"], same_dataset["objective_mask"]),
                    "next_rollout": _evaluate_scalar_head(next_g, next_dataset["returns"], next_dataset["objective_mask"]),
                },
                "pre_output_sum": {
                    "geometry_summary": _geometry_summary(same_pre, same_dataset["objective_mask"]),
                    "same_rollout": _evaluate_scalar_head(same_pre, same_dataset["returns"], same_dataset["objective_mask"]),
                    "next_rollout": _evaluate_scalar_head(next_pre, next_dataset["returns"], next_dataset["objective_mask"]),
                },
                "pre_gated": {
                    "geometry_summary": _geometry_summary(same_pre_gated, same_dataset["objective_mask"]),
                    "same_rollout": _evaluate_scalar_head(same_pre_gated, same_dataset["returns"], same_dataset["objective_mask"]),
                    "next_rollout": _evaluate_scalar_head(next_pre_gated, next_dataset["returns"], next_dataset["objective_mask"]),
                },
                "att_out": {
                    "geometry_summary": _geometry_summary(same_att, same_dataset["objective_mask"]),
                    "same_rollout": _evaluate_scalar_head(same_att, same_dataset["returns"], same_dataset["objective_mask"]),
                    "next_rollout": _evaluate_scalar_head(next_att, next_dataset["returns"], next_dataset["objective_mask"]),
                },
            },
            "activation_summary": _gate_activation_summary(
                same_gate_pre,
                same_gate_sig,
                same_g,
                same_dataset["objective_mask"],
            ),
            "directional_erosion": _gate_directional_erosion(
                xg=same_xg,
                gate_pre=same_gate_pre,
                gate_sigmoid=same_gate_sig,
                g=same_g,
                pre_output_sum=same_pre,
                pre_gated=same_pre_gated,
                att_out=same_att,
                returns=same_dataset["returns"],
                mask=same_dataset["objective_mask"],
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
    parser = argparse.ArgumentParser(description="Gate-path probe for Phase 2 critic attention branch.")
    parser.add_argument("checkpoint_path", type=str)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-json", type=str, required=True)
    args = parser.parse_args()

    result = run_phase2_attention_gate_path_probe(
        checkpoint_path=args.checkpoint_path,
        device=args.device,
    )
    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(str(output_path))


if __name__ == "__main__":
    main()
