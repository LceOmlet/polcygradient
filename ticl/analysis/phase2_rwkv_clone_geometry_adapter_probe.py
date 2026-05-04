import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from ticl.analysis.phase2_hidden_geometry_whitening_probe import (
    _apply_raw_transform,
    _apply_zscore_transform,
    _evaluate_variant_on_transform,
    _fit_linear_geometry,
    _fit_official_distribution_head,
)
from ticl.analysis.phase2_simple_scalar_value_head_probe import (
    _build_algo,
    _build_suite_algo,
    _collect_rollout,
    _default_device,
    _extract_latent_dataset,
    _suite_from_seed_range,
)
from ticl.analysis.critic_free_single_env_audit import (
    _build_audit_env_cfg,
    _prepare_audit_fixed_env_contract,
    _seed_all,
)
from ticl.model_builder import load_model


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_head(repo_dir: Path) -> str:
    import subprocess

    return (
        subprocess.check_output(["git", "-C", str(repo_dir), "rev-parse", "HEAD"], text=True)
        .strip()
    )


def _manual_direct_core_hidden_flat(policy, obs: torch.Tensor, *, seq_lengths: np.ndarray) -> torch.Tensor:
    if obs.ndim != 2:
        raise ValueError(f"obs must be flat (B, F), got {tuple(obs.shape)}")
    seq_lengths = np.asarray(seq_lengths, dtype=np.int64)
    n_seq = int(len(seq_lengths))
    if n_seq <= 0:
        raise ValueError("Need at least one sequence.")
    if int(seq_lengths.sum()) != int(obs.shape[0]):
        raise ValueError(
            f"Flat obs batch {int(obs.shape[0])} must match sum(seq_lengths)={int(seq_lengths.sum())}"
        )
    seq_start_indices = np.zeros((n_seq,), dtype=np.int64)
    if n_seq > 1:
        seq_start_indices[1:] = np.cumsum(seq_lengths[:-1], dtype=np.int64)

    def _resolve_pack_capacity(seq_len: int, total_batch: int) -> int:
        resolve_chunk = getattr(policy.rlpfn_model, "resolve_replay_batch_chunk_size", None)
        if callable(resolve_chunk):
            batch_chunk_size = int(resolve_chunk(seq_len=int(seq_len), total_batch=int(total_batch)))
        else:
            batch_chunk_size = int(total_batch)
        return int(max(1, min(int(total_batch), batch_chunk_size)))

    def _forward_pack(pack_seq_indices: list[int]) -> torch.Tensor:
        pack_size = int(len(pack_seq_indices))
        pack_lengths = seq_lengths[np.asarray(pack_seq_indices, dtype=np.int64)]
        pack_max_len = int(pack_lengths.max())
        bucket_obs = torch.zeros(
            (pack_max_len, pack_size, int(obs.shape[-1])),
            device=obs.device,
            dtype=obs.dtype,
        )
        for col, seq_idx in enumerate(pack_seq_indices):
            seg_start = int(seq_start_indices[int(seq_idx)])
            seg_len = int(seq_lengths[int(seq_idx)])
            bucket_obs[:seg_len, col].copy_(obs[seg_start : seg_start + seg_len])
        tokens = policy._encode_obs_tokens(bucket_obs)
        tokens = policy.rlpfn_model._cast_token_for_rwkv_core(tokens)
        return policy.rlpfn_model.rwkv_core.forward_tokens_sequence_only(tokens)

    if np.all(seq_lengths == int(seq_lengths[0])):
        seq_len = int(seq_lengths[0])
        bucket_obs = obs.reshape((n_seq, seq_len, policy.num_features)).swapaxes(0, 1).contiguous()
        total_batch = int(bucket_obs.shape[1])
        batch_chunk_size = _resolve_pack_capacity(seq_len=seq_len, total_batch=total_batch)
        if batch_chunk_size >= total_batch:
            hidden_bucket = _forward_pack(list(range(n_seq)))
            return hidden_bucket.swapaxes(0, 1).reshape(-1, hidden_bucket.shape[-1])

    order = sorted(range(n_seq), key=lambda idx: int(seq_lengths[int(idx)]), reverse=True)
    hidden_flat = None
    cursor = 0
    while cursor < n_seq:
        first_seq_idx = int(order[cursor])
        pack_max_len = int(seq_lengths[first_seq_idx])
        remaining = int(n_seq - cursor)
        pack_capacity = _resolve_pack_capacity(seq_len=pack_max_len, total_batch=remaining)
        pack_seq_indices = [int(idx) for idx in order[cursor : cursor + pack_capacity]]
        hidden_pack = _forward_pack(pack_seq_indices)
        if hidden_flat is None:
            hidden_flat = torch.empty(
                (int(obs.shape[0]), int(hidden_pack.shape[-1])),
                device=hidden_pack.device,
                dtype=hidden_pack.dtype,
            )
        for pack_col, seq_idx in enumerate(pack_seq_indices):
            seg_start = int(seq_start_indices[int(seq_idx)])
            seg_len = int(seq_lengths[int(seq_idx)])
            hidden_flat[seg_start : seg_start + seg_len].copy_(hidden_pack[:seg_len, pack_col])
        cursor += len(pack_seq_indices)
    if hidden_flat is None:
        raise RuntimeError("Manual direct core hidden replay produced no hidden states.")
    return hidden_flat


@torch.no_grad()
def _hidden_origin_check(algo) -> dict:
    rollout_iter = algo.rollout_buffer.get_gpu_flat(algo.batch_size)
    rollout_data = next(rollout_iter)
    eval_outputs = algo.policy.evaluate_actions_with_hidden_flat(
        rollout_data.observations,
        rollout_data.actions,
        seq_lengths=rollout_data.seq_lengths,
        action_masks=rollout_data.action_masks,
    )
    hidden_eval = eval_outputs["hidden"]
    latent_vf = algo.policy.mlp_extractor.forward_critic(hidden_eval)
    hidden_direct = _manual_direct_core_hidden_flat(
        algo.policy,
        rollout_data.observations,
        seq_lengths=rollout_data.seq_lengths,
    )
    return {
        "hidden_shape": list(hidden_eval.shape),
        "direct_hidden_shape": list(hidden_direct.shape),
        "latent_vf_shape": list(latent_vf.shape),
        "max_abs_diff_hidden_vs_direct_core": float((hidden_eval - hidden_direct).abs().max().item()),
        "mean_abs_diff_hidden_vs_direct_core": float((hidden_eval - hidden_direct).abs().mean().item()),
        "max_abs_diff_hidden_vs_latent_vf": float((hidden_eval - latent_vf).abs().max().item()),
        "mean_abs_diff_hidden_vs_latent_vf": float((hidden_eval - latent_vf).abs().mean().item()),
    }


def run_phase2_rwkv_clone_geometry_adapter_probe(
    *,
    checkpoint_path: str,
    device: str | None = None,
    frozen_h_seed: int = 12345,
    train_env_seed: int = 2020,
    train_rollout_seed: int = 4040,
    eval_env_seed_start: int = 3000,
    eval_rollout_seed_start: int = 4000,
    eval_env_count: int = 4,
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
    rwkv_clone_dir: str = "/home/chen/RLPFN/external/RWKV-LM-official",
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
    rwkv_clone_path = Path(rwkv_clone_dir).expanduser().resolve()
    vendor_root = Path(
        "/home/chen/RLPFN/reinforce-terminal-explore/ticl/models/vendor/rwkv_lm_official"
    ).resolve()
    same_dataset = None
    next_dataset = None
    try:
        _collect_rollout(algo, callback, vec_env, progress_timestep=0, total_timesteps=total_timesteps)
        algo.train()
        same_dataset = _extract_latent_dataset(algo)
        hidden_origin = _hidden_origin_check(algo)
        progress_next = min(int(n_steps), total_timesteps)
        _collect_rollout(algo, callback, vec_env, progress_timestep=progress_next, total_timesteps=total_timesteps)
        next_dataset = _extract_latent_dataset(algo)
    finally:
        vec_env.close()

    geom = _fit_linear_geometry(same_dataset["latent_vf"], same_dataset["objective_mask"])
    raw_same = _apply_raw_transform(same_dataset["latent_vf"], geom)
    raw_next = _apply_raw_transform(next_dataset["latent_vf"], geom)
    zscore_same = _apply_zscore_transform(same_dataset["latent_vf"], geom)
    zscore_next = _apply_zscore_transform(next_dataset["latent_vf"], geom)

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

    suite = _suite_from_seed_range(
        frozen_h=frozen_h,
        env_seed_start=int(eval_env_seed_start),
        rollout_seed_start=int(eval_rollout_seed_start),
        count=int(eval_env_count),
    )
    suite_algo, suite_callback, suite_vec_env = _build_suite_algo(
        checkpoint_path=checkpoint_path,
        device_obj=device_obj,
        env_cfg=env_cfg,
        num_features=int(num_features),
        suite=suite,
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
    try:
        _collect_rollout(suite_algo, suite_callback, suite_vec_env, progress_timestep=0, total_timesteps=int(n_steps))
        heldout_dataset = _extract_latent_dataset(suite_algo)
    finally:
        suite_vec_env.close()

    heldout_raw = _apply_raw_transform(heldout_dataset["latent_vf"], geom)
    heldout_zscore = _apply_zscore_transform(heldout_dataset["latent_vf"], geom)
    heldout_raw_metrics = _evaluate_variant_on_transform(
        variant=raw_variant,
        same_x=heldout_raw,
        next_x=heldout_raw,
        same_dataset=heldout_dataset,
        next_dataset=heldout_dataset,
        device=device_obj,
    )["same_rollout"]
    heldout_zscore_metrics = _evaluate_variant_on_transform(
        variant=zscore_variant,
        same_x=heldout_zscore,
        next_x=heldout_zscore,
        same_dataset=heldout_dataset,
        next_dataset=heldout_dataset,
        device=device_obj,
    )["same_rollout"]

    official_files = {
        "rwkv_v7_demo_rnn.py": {
            "vendor": str(vendor_root / "RWKV-v7" / "rwkv_v7_demo_rnn.py"),
            "clone": str(rwkv_clone_path / "RWKV-v7" / "rwkv_v7_demo_rnn.py"),
        },
        "train_temp/src/model.py": {
            "vendor": str(vendor_root / "RWKV-v7" / "train_temp" / "src" / "model.py"),
            "clone": str(rwkv_clone_path / "RWKV-v7" / "train_temp" / "src" / "model.py"),
        },
    }
    file_hashes = {}
    for name, paths in official_files.items():
        vendor_path = Path(paths["vendor"])
        clone_path = Path(paths["clone"])
        file_hashes[name] = {
            "vendor_exists": bool(vendor_path.exists()),
            "clone_exists": bool(clone_path.exists()),
            "vendor_sha256": _sha256(vendor_path) if vendor_path.exists() else None,
            "clone_sha256": _sha256(clone_path) if clone_path.exists() else None,
            "hash_match": bool(vendor_path.exists() and clone_path.exists() and _sha256(vendor_path) == _sha256(clone_path)),
        }

    return {
        "checkpoint_path": str(Path(checkpoint_path).expanduser().resolve()),
        "contract": {
            "frozen_h_seed": int(frozen_h_seed),
            "train_env_seed": int(train_env_seed),
            "train_rollout_seed": int(train_rollout_seed),
            "eval_env_seed_start": int(eval_env_seed_start),
            "eval_rollout_seed_start": int(eval_rollout_seed_start),
            "eval_env_count": int(eval_env_count),
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
        "rwkv_official_clone": {
            "repo_dir": str(rwkv_clone_path),
            "git_head": _git_head(rwkv_clone_path),
            "file_hashes": file_hashes,
        },
        "hidden_origin_check": hidden_origin,
        "geometry_summary": {
            "raw_projection_std_proxy": float(
                same_dataset["latent_vf"][same_dataset["objective_mask"]].to(dtype=torch.float32).std(dim=0, unbiased=False).mean().item()
            ),
            "zscore_projection_std_proxy": float(
                zscore_same[same_dataset["objective_mask"]].to(dtype=torch.float32).std(dim=0, unbiased=False).mean().item()
            ),
        },
        "distribution_head_metrics": {
            "raw_hidden": {
                "same_rollout": raw_metrics["same_rollout"],
                "next_rollout": raw_metrics["next_rollout"],
                "heldout": heldout_raw_metrics,
                "fit_summary": raw_metrics["fit_summary"],
            },
            "frozen_zscore_adapter": {
                "same_rollout": zscore_metrics["same_rollout"],
                "next_rollout": zscore_metrics["next_rollout"],
                "heldout": heldout_zscore_metrics,
                "fit_summary": zscore_metrics["fit_summary"],
            },
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Clone-aware RWKV geometry adapter probe for Phase 2 critic hidden.")
    parser.add_argument("checkpoint_path", type=str)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-json", type=str, required=True)
    args = parser.parse_args()

    result = run_phase2_rwkv_clone_geometry_adapter_probe(
        checkpoint_path=args.checkpoint_path,
        device=args.device,
    )
    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(str(output_path))


if __name__ == "__main__":
    main()
