import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from ticl.analysis.phase2_distribution_head_root_cause_probe import _fit_distribution_variant
from ticl.analysis.phase2_simple_scalar_value_head_probe import (
    _build_algo,
    _collect_rollout,
    _default_device,
    _extract_latent_dataset,
    _fit_ridge_scalar_head,
)
from ticl.analysis.critic_free_single_env_audit import (
    _build_audit_env_cfg,
    _prepare_audit_fixed_env_contract,
)
from ticl.model_builder import load_model
from ticl.models.tabpfn_regressor_vendor import TabPFNOfficialRegressorHarness


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


def _evaluate_predictions(preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> dict:
    preds_np = preds.detach().cpu().numpy().reshape(-1)
    targets_np = targets.detach().cpu().numpy().reshape(-1)
    mask_np = mask.detach().cpu().numpy().reshape(-1).astype(bool)
    if int(mask_np.sum()) <= 1:
        return {
            "pearson": 0.0,
            "pred_std": 0.0,
            "target_std": 0.0,
        }
    p = preds_np[mask_np]
    t = targets_np[mask_np]
    return {
        "pearson": _safe_corrcoef(p, t),
        "pred_std": float(np.asarray(p, dtype=np.float32).std()),
        "target_std": float(np.asarray(t, dtype=np.float32).std()),
    }


def _apply_scalar_head(x: torch.Tensor, head: dict) -> torch.Tensor:
    return x.to(dtype=torch.float32) @ head["weight"] + head["bias"]


def _direction_from_ridge_head(head: dict) -> torch.Tensor:
    w = head["weight"].detach().cpu().to(dtype=torch.float32).reshape(-1)
    norm = float(torch.linalg.vector_norm(w).item())
    if norm <= 0.0:
        raise RuntimeError("Zero-norm ridge direction.")
    return w / norm


def _make_vendor_harness(*, latent_dim: int, num_buckets: int, value_range: float) -> TabPFNOfficialRegressorHarness:
    harness = TabPFNOfficialRegressorHarness(
        emsize=int(latent_dim),
        num_buckets=int(num_buckets),
        value_range=float(value_range),
        ignore_nan_targets=True,
    )
    harness.reset_target_statistics()
    return harness


def _fit_official_distribution_head(
    *,
    x: torch.Tensor,
    y: torch.Tensor,
    latent_dim: int,
    num_buckets: int,
    value_range: float,
    fit_steps: int,
    lr: float,
    max_grad_norm: float,
    device: torch.device,
) -> dict:
    harness = _make_vendor_harness(
        latent_dim=int(latent_dim),
        num_buckets=int(num_buckets),
        value_range=float(value_range),
    ).to(device)
    bucket_idx = harness.znorm_space_bardist_.map_to_bucket_idx(y)
    return _fit_distribution_variant(
        head_module=copy.deepcopy(harness.output_projection),
        bardist=copy.deepcopy(harness.znorm_space_bardist_),
        x=x,
        y=y,
        bucket_idx=bucket_idx,
        fit_steps=int(fit_steps),
        lr=float(lr),
        max_grad_norm=float(max_grad_norm),
        loss_name="vendor_official",
        official_harness=copy.deepcopy(harness),
    )


def _fit_linear_geometry(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> dict:
    mask = mask.reshape(-1).to(dtype=torch.bool)
    x_obj = x[mask].to(dtype=torch.float64)
    mean = x_obj.mean(dim=0, keepdim=True)
    centered = x_obj - mean
    feat_std = centered.std(dim=0, unbiased=False).clamp_min(float(eps))
    u, s, vh = torch.linalg.svd(centered, full_matrices=False)
    var = (s.square() / max(1, int(centered.shape[0]) - 1)).clamp_min(float(eps))
    return {
        "mean": mean.to(dtype=torch.float32).reshape(-1),
        "feature_std": feat_std.to(dtype=torch.float32).reshape(-1),
        "pc_basis": vh.to(dtype=torch.float32),  # [rank, dim]
        "pc_var": var.to(dtype=torch.float32),   # [rank]
        "singular_values": s.to(dtype=torch.float32),
        "rank": int(vh.shape[0]),
        "trace_var": float(var.sum().item()),
    }


def _apply_raw_transform(x: torch.Tensor, geom: dict) -> torch.Tensor:
    return x.to(dtype=torch.float32)


def _apply_zscore_transform(x: torch.Tensor, geom: dict) -> torch.Tensor:
    return (x.to(dtype=torch.float32) - geom["mean"]) / geom["feature_std"]


def _apply_pca_whiten_transform(x: torch.Tensor, geom: dict) -> torch.Tensor:
    centered = x.to(dtype=torch.float32) - geom["mean"]
    scores = centered @ geom["pc_basis"].T
    return scores / torch.sqrt(geom["pc_var"] + 1e-6)


def _variance_spectrum_summary(x: torch.Tensor, mask: torch.Tensor, direction: torch.Tensor, geom: dict) -> dict:
    mask = mask.reshape(-1).to(dtype=torch.bool)
    x_obj = x[mask].to(dtype=torch.float32)
    centered = x_obj - geom["mean"]
    proj = centered @ direction.to(dtype=torch.float32)
    proj_var = float(torch.var(proj, unbiased=False).item())
    pc_basis = geom["pc_basis"].to(dtype=torch.float32)
    coeff = pc_basis @ direction.to(dtype=torch.float32)
    coeff_sq = coeff.square()
    coeff_sq = coeff_sq / torch.clamp(coeff_sq.sum(), min=1e-12)
    topk = {}
    for k in (1, 2, 4, 8, 16, 32):
        topk[str(k)] = float(coeff_sq[: min(k, int(coeff_sq.numel()))].sum().item())
    return {
        "projection_std": float(torch.std(proj, unbiased=False).item()),
        "projection_var_ratio_to_trace": float(proj_var / max(geom["trace_var"], 1e-12)),
        "top_pc_alignment_mass": topk,
        "rank": int(geom["rank"]),
        "top5_pc_var": [float(v) for v in geom["pc_var"][:5].tolist()],
    }


def _evaluate_variant_on_transform(
    *,
    variant: dict,
    same_x: torch.Tensor,
    next_x: torch.Tensor,
    same_dataset: dict,
    next_dataset: dict,
    device: torch.device,
) -> dict:
    model = variant["model"].to(device)
    bardist = variant["bardist"].to(device)
    same_logits = model(same_x.to(device=device, dtype=torch.float32))
    next_logits = model(next_x.to(device=device, dtype=torch.float32))
    same_preds = bardist.mean(same_logits.to(dtype=torch.float32)).detach().cpu().reshape(-1)
    next_preds = bardist.mean(next_logits.to(dtype=torch.float32)).detach().cpu().reshape(-1)
    return {
        "fit_summary": dict(variant["fit_summary"]),
        "same_rollout": _evaluate_predictions(same_preds, same_dataset["returns"], same_dataset["objective_mask"]),
        "next_rollout": _evaluate_predictions(next_preds, next_dataset["returns"], next_dataset["objective_mask"]),
    }


def run_phase2_hidden_geometry_whitening_probe(
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
    fit_steps_schedule: tuple[int, ...] = (0, 1, 2, 5, 20),
    head_fit_learning_rate: float = 1e-3,
    head_fit_max_grad_norm: float = 0.5,
    num_buckets: int = 31,
    value_range: float = 5.0,
) -> dict:
    device_obj = torch.device(str(device or _default_device()))
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
    history = []
    try:
        _collect_rollout(algo, callback, vec_env, progress_timestep=0, total_timesteps=total_timesteps)
        for outer_idx in range(int(outer_epochs)):
            algo.train()
            same_dataset = _extract_latent_dataset(algo)
            hidden_scalar_head = _fit_ridge_scalar_head(
                same_dataset["latent_vf"],
                same_dataset["returns"],
                same_dataset["objective_mask"],
                ridge_lambda=1e-3,
            )
            hidden_direction = _direction_from_ridge_head(hidden_scalar_head)

            progress_next = min(int((outer_idx + 1) * n_steps), total_timesteps)
            _collect_rollout(algo, callback, vec_env, progress_timestep=progress_next, total_timesteps=total_timesteps)
            next_dataset = _extract_latent_dataset(algo)

            geom = _fit_linear_geometry(same_dataset["latent_vf"], same_dataset["objective_mask"])
            same_hidden_scalar = _apply_scalar_head(same_dataset["latent_vf"], hidden_scalar_head)
            next_hidden_scalar = _apply_scalar_head(next_dataset["latent_vf"], hidden_scalar_head)
            latent_dim = int(same_dataset["latent_vf"].shape[-1])

            transforms = {
                "raw_hidden": (
                    _apply_raw_transform(same_dataset["latent_vf"], geom),
                    _apply_raw_transform(next_dataset["latent_vf"], geom),
                ),
                "zscore_hidden": (
                    _apply_zscore_transform(same_dataset["latent_vf"], geom),
                    _apply_zscore_transform(next_dataset["latent_vf"], geom),
                ),
                "pca_whiten_hidden": (
                    _apply_pca_whiten_transform(same_dataset["latent_vf"], geom),
                    _apply_pca_whiten_transform(next_dataset["latent_vf"], geom),
                ),
                "hidden_scalar_control": (
                    torch.zeros((same_hidden_scalar.shape[0], latent_dim), dtype=torch.float32).index_copy(
                        1, torch.tensor([0]), same_hidden_scalar.reshape(-1, 1)
                    ),
                    torch.zeros((next_hidden_scalar.shape[0], latent_dim), dtype=torch.float32).index_copy(
                        1, torch.tensor([0]), next_hidden_scalar.reshape(-1, 1)
                    ),
                ),
            }

            transform_summaries = {
                "raw_hidden": _variance_spectrum_summary(
                    same_dataset["latent_vf"],
                    same_dataset["objective_mask"],
                    hidden_direction,
                    geom,
                ),
                "zscore_hidden": _variance_spectrum_summary(
                    transforms["zscore_hidden"][0],
                    same_dataset["objective_mask"],
                    _direction_from_ridge_head(
                        _fit_ridge_scalar_head(
                            transforms["zscore_hidden"][0],
                            same_dataset["returns"],
                            same_dataset["objective_mask"],
                            ridge_lambda=1e-3,
                        )
                    ),
                    _fit_linear_geometry(transforms["zscore_hidden"][0], same_dataset["objective_mask"]),
                ),
                "pca_whiten_hidden": _variance_spectrum_summary(
                    transforms["pca_whiten_hidden"][0],
                    same_dataset["objective_mask"],
                    _direction_from_ridge_head(
                        _fit_ridge_scalar_head(
                            transforms["pca_whiten_hidden"][0],
                            same_dataset["returns"],
                            same_dataset["objective_mask"],
                            ridge_lambda=1e-3,
                        )
                    ),
                    _fit_linear_geometry(transforms["pca_whiten_hidden"][0], same_dataset["objective_mask"]),
                ),
            }

            per_step = {}
            for fit_steps in [int(v) for v in fit_steps_schedule]:
                step_metrics = {}
                for name, (same_x, next_x) in transforms.items():
                    variant = _fit_official_distribution_head(
                        x=same_x[same_dataset["objective_mask"]].to(device=device_obj, dtype=torch.float32),
                        y=same_dataset["returns"][same_dataset["objective_mask"]].to(device=device_obj, dtype=torch.float32),
                        latent_dim=int(same_x.shape[-1]),
                        num_buckets=int(num_buckets),
                        value_range=float(value_range),
                        fit_steps=int(fit_steps),
                        lr=float(head_fit_learning_rate),
                        max_grad_norm=float(head_fit_max_grad_norm),
                        device=device_obj,
                    )
                    step_metrics[name] = _evaluate_variant_on_transform(
                        variant=variant,
                        same_x=same_x,
                        next_x=next_x,
                        same_dataset=same_dataset,
                        next_dataset=next_dataset,
                        device=device_obj,
                    )
                per_step[str(fit_steps)] = step_metrics

            history.append(
                {
                    "outer_epoch": int(outer_idx + 1),
                    "fit_steps_schedule": [int(v) for v in fit_steps_schedule],
                    "distribution_spec": {
                        "num_buckets": int(num_buckets),
                        "value_range": float(value_range),
                    },
                    "same_objective_total": int(same_dataset["objective_mask"].sum().item()),
                    "next_objective_total": int(next_dataset["objective_mask"].sum().item()),
                    "geometry_summary": transform_summaries,
                    "per_fit_steps": per_step,
                }
            )
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
            "outer_epochs": int(outer_epochs),
            "build_seed": int(build_seed),
            "actor_baseline_mode": str(actor_baseline_mode),
            "strict_fixed_env_mode": True,
            "deterministic_actor_sampling": True,
            "deterministic_batch_plan": True,
            "strict_native_rollout": False,
            "ppo_reset_env_state_at_sep": bool(ppo_reset_env_state_at_sep),
        },
        "history": history,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Disentangle whether official distribution head failure is due to critic hidden geometry or input/task semantics via linear geometry controls."
    )
    parser.add_argument("checkpoint_path", type=str)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--frozen-h-seed", type=int, default=12345)
    parser.add_argument("--train-env-seed", type=int, default=2020)
    parser.add_argument("--train-rollout-seed", type=int, default=4040)
    parser.add_argument("--single-eval-pos", type=int, default=64)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument("--outer-epochs", type=int, default=1)
    parser.add_argument("--build-seed", type=int, default=4040)
    parser.add_argument("--num-buckets", type=int, default=31)
    parser.add_argument("--value-range", type=float, default=5.0)
    parser.add_argument("--head-fit-learning-rate", type=float, default=1e-3)
    parser.add_argument("--head-fit-max-grad-norm", type=float, default=0.5)
    parser.add_argument("--output-json", type=str, required=True)
    args = parser.parse_args()

    result = run_phase2_hidden_geometry_whitening_probe(
        checkpoint_path=args.checkpoint_path,
        device=args.device,
        frozen_h_seed=int(args.frozen_h_seed),
        train_env_seed=int(args.train_env_seed),
        train_rollout_seed=int(args.train_rollout_seed),
        single_eval_pos=int(args.single_eval_pos),
        n_steps=int(args.n_steps),
        batch_size=int(args.batch_size),
        n_epochs=int(args.n_epochs),
        learning_rate=float(args.learning_rate),
        target_kl=float(args.target_kl),
        outer_epochs=int(args.outer_epochs),
        build_seed=int(args.build_seed),
        num_buckets=int(args.num_buckets),
        value_range=float(args.value_range),
        head_fit_learning_rate=float(args.head_fit_learning_rate),
        head_fit_max_grad_norm=float(args.head_fit_max_grad_norm),
    )
    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
