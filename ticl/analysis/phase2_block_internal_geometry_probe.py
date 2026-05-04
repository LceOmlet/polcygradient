import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

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


def _safe_corrcoef(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    finite = np.isfinite(x) & np.isfinite(y)
    if int(finite.sum()) <= 1:
        return 0.0
    x = x[finite]
    y = y[finite]
    x = x - float(x.mean())
    y = y - float(y.mean())
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(x, y) / denom)


def _evaluate_scalar_head(x: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> dict:
    head = _fit_ridge_scalar_head(x, targets, mask, ridge_lambda=1e-3)
    preds = (x.to(dtype=torch.float32) @ head["weight"] + head["bias"]).detach().cpu().numpy().reshape(-1)
    target_np = targets.detach().cpu().numpy().reshape(-1)
    mask_np = mask.detach().cpu().numpy().reshape(-1).astype(bool)
    return {
        "pearson": _safe_corrcoef(preds[mask_np], target_np[mask_np]),
        "pred_std": float(np.asarray(preds[mask_np], dtype=np.float32).std()) if int(mask_np.sum()) > 1 else 0.0,
        "target_std": float(np.asarray(target_np[mask_np], dtype=np.float32).std()) if int(mask_np.sum()) > 1 else 0.0,
    }


def _geometry_summary(x: torch.Tensor, mask: torch.Tensor) -> dict:
    mask = mask.reshape(-1).to(dtype=torch.bool)
    x_obj = x[mask].to(dtype=torch.float32)
    feat_std = x_obj.std(dim=0, unbiased=False)
    centered = x_obj - x_obj.mean(dim=0, keepdim=True)
    cov = torch.cov(centered.T.to(dtype=torch.float64))
    eigvals = torch.linalg.eigvalsh(cov).real.clamp_min(0.0)
    trace = float(eigvals.sum().item())
    top = float(eigvals[-1].item()) if int(eigvals.numel()) > 0 else 0.0
    nonzero = eigvals[eigvals > 1e-12]
    cond = float((float(nonzero.max().item()) / float(nonzero.min().item())) if int(nonzero.numel()) > 0 else 0.0)
    return {
        "feature_std_mean": float(feat_std.mean().item()),
        "feature_std_min": float(feat_std.min().item()),
        "feature_std_max": float(feat_std.max().item()),
        "cov_trace": trace,
        "top_eigen_ratio": float(top / max(trace, 1e-12)),
        "cov_condition_nonzero": cond,
    }


@torch.no_grad()
def _forward_pack_block_stages(policy, bucket_obs: torch.Tensor) -> dict:
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
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=bool(tokens_work.is_cuda)):
        for layer_idx, block in enumerate(policy.rlpfn_model.rwkv_core.blocks):
            official_train = _load_official_rwkv7_train_temp(block.head_size)
            if hasattr(block, "ln0"):
                x = block.ln0(x)
                stages[f"layer{layer_idx}.after_ln0"] = x.transpose(0, 1)

            x_att = block.ln1(x)
            stages[f"layer{layer_idx}.ln1"] = x_att.transpose(0, 1)
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
            k = block.att.key(xk)
            v = block.att.value(xv)
            if layer_idx == 0:
                v_first = v
            else:
                v = v + (v_first - v) * torch.sigmoid(block.att.v0 + (xv @ block.att.v1) @ block.att.v2)
            a = torch.sigmoid(block.att.a0 + (xa @ block.att.a1) @ block.att.a2)
            g = torch.sigmoid(xg @ block.att.g1) @ block.att.g2

            kk = k * block.att.k_k
            kk = F.normalize(kk.view(bsz, seq_work_len, block.n_head, -1), dim=-1, p=2.0).view(bsz, seq_work_len, channels)
            k = k * (1 + (a - 1) * block.att.k_a)

            att_out = official_train.RWKV7_CLAMPW_CUDA(
                r.to(dtype=torch.bfloat16),
                w.to(dtype=torch.bfloat16),
                k.to(dtype=torch.bfloat16),
                v.to(dtype=torch.bfloat16),
                (-kk).to(dtype=torch.bfloat16),
                (kk * a).to(dtype=torch.bfloat16),
            )
            att_out = block.att.ln_x(att_out.view(bsz * seq_work_len, channels)).view(bsz, seq_work_len, channels)
            att_out = att_out + (
                (r.view(bsz, seq_work_len, block.n_head, -1) * k.view(bsz, seq_work_len, block.n_head, -1) * block.att.r_k)
                .sum(dim=-1, keepdim=True)
                * v.view(bsz, seq_work_len, block.n_head, -1)
            ).view(bsz, seq_work_len, channels)
            att_out = block.att.output(att_out * g)
            stages[f"layer{layer_idx}.att_out"] = att_out.transpose(0, 1)

            x_after_att = x + att_out
            stages[f"layer{layer_idx}.x_after_att"] = x_after_att.transpose(0, 1)

            ln2 = block.ln2(x_after_att)
            stages[f"layer{layer_idx}.ln2"] = ln2.transpose(0, 1)

            ffn_out = block.ffn(ln2)
            stages[f"layer{layer_idx}.ffn_out"] = ffn_out.transpose(0, 1)

            x = x_after_att + ffn_out
            stages[f"layer{layer_idx}.x_after_ffn"] = x.transpose(0, 1)

        post_ln = policy.rlpfn_model.rwkv_core.ln_out(x)
        stages["post_ln_out_hidden"] = post_ln.transpose(0, 1)

    if pad_len > 0:
        for key in list(stages.keys()):
            stages[key] = stages[key][:seq_len]
    return stages


@torch.no_grad()
def _extract_block_stage_dataset(algo) -> dict:
    rollout_buffer = algo.rollout_buffer
    stage_latents = None
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
            stage_pack = _forward_pack_block_stages(algo.policy, bucket_obs)
            if stage_flat is None:
                stage_dim = int(next(iter(stage_pack.values())).shape[-1])
                stage_flat = {
                    key: torch.empty(
                        (int(obs.shape[0]), stage_dim),
                        device=value.device,
                        dtype=value.dtype,
                    )
                    for key, value in stage_pack.items()
                }
                if stage_latents is None:
                    stage_latents = {key: [] for key in stage_pack.keys()}
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
        "returns": torch.cat(targets, dim=0),
        "objective_mask": torch.cat(masks, dim=0),
    }


def run_phase2_block_internal_geometry_probe(
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
        same_dataset = _extract_block_stage_dataset(algo)
        progress_next = min(int(n_steps), total_timesteps)
        _collect_rollout(algo, callback, vec_env, progress_timestep=progress_next, total_timesteps=total_timesteps)
        next_dataset = _extract_block_stage_dataset(algo)
    finally:
        vec_env.close()

    results = {}
    stage_names = list(same_dataset["stages"].keys())
    for stage_name in stage_names:
        same_x = same_dataset["stages"][stage_name]
        next_x = next_dataset["stages"][stage_name]
        geom = _fit_linear_geometry(same_x, same_dataset["objective_mask"])
        raw_same = _apply_raw_transform(same_x, geom)
        raw_next = _apply_raw_transform(next_x, geom)
        zscore_same = _apply_zscore_transform(same_x, geom)
        zscore_next = _apply_zscore_transform(next_x, geom)

        raw_variant = _fit_official_distribution_head(
            x=raw_same[same_dataset["objective_mask"]].to(device=device_obj, dtype=torch.float32),
            y=same_dataset["returns"][same_dataset["objective_mask"]].to(device=device_obj, dtype=torch.float32),
            latent_dim=int(raw_same.shape[-1]),
            num_buckets=int(num_buckets),
            value_range=float(value_range),
            fit_steps=int(fit_steps),
            lr=float(head_fit_learning_rate),
            max_grad_norm=float(head_fit_max_grad_norm),
            device=device_obj,
        )
        zscore_variant = _fit_official_distribution_head(
            x=zscore_same[same_dataset["objective_mask"]].to(device=device_obj, dtype=torch.float32),
            y=same_dataset["returns"][same_dataset["objective_mask"]].to(device=device_obj, dtype=torch.float32),
            latent_dim=int(zscore_same.shape[-1]),
            num_buckets=int(num_buckets),
            value_range=float(value_range),
            fit_steps=int(fit_steps),
            lr=float(head_fit_learning_rate),
            max_grad_norm=float(head_fit_max_grad_norm),
            device=device_obj,
        )
        raw_metrics = _evaluate_variant_on_transform(
            variant=raw_variant,
            same_x=raw_same,
            next_x=raw_next,
            same_dataset=same_dataset,
            next_dataset=next_dataset,
            device=device_obj,
        )
        zscore_metrics = _evaluate_variant_on_transform(
            variant=zscore_variant,
            same_x=zscore_same,
            next_x=zscore_next,
            same_dataset=same_dataset,
            next_dataset=next_dataset,
            device=device_obj,
        )
        results[stage_name] = {
            "geometry_summary": _geometry_summary(same_x, same_dataset["objective_mask"]),
            "raw_scalar_ridge": {
                "same_rollout": _evaluate_scalar_head(same_x, same_dataset["returns"], same_dataset["objective_mask"]),
                "next_rollout": _evaluate_scalar_head(next_x, next_dataset["returns"], next_dataset["objective_mask"]),
            },
            "distribution_head_metrics": {
                "raw_hidden": {
                    "same_rollout": raw_metrics["same_rollout"],
                    "next_rollout": raw_metrics["next_rollout"],
                    "fit_summary": raw_metrics["fit_summary"],
                },
                "frozen_zscore_adapter": {
                    "same_rollout": zscore_metrics["same_rollout"],
                    "next_rollout": zscore_metrics["next_rollout"],
                    "fit_summary": zscore_metrics["fit_summary"],
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
            "fit_steps": int(fit_steps),
            "num_buckets": int(num_buckets),
            "value_range": float(value_range),
            "geometry_adapter": "frozen_feature_zscore_from_same_rollout_objective_tokens",
        },
        "stages": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Internal RWKV block geometry probe for Phase 2 critic path.")
    parser.add_argument("checkpoint_path", type=str)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-json", type=str, required=True)
    args = parser.parse_args()

    result = run_phase2_block_internal_geometry_probe(
        checkpoint_path=args.checkpoint_path,
        device=args.device,
    )
    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(str(output_path))


if __name__ == "__main__":
    main()
