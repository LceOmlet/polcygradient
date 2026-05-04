import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ticl.analysis.phase2_distribution_head_arch_loss_probe import (
    _fit_variant as _fit_arch_loss_variant,
    _make_head,
)
from ticl.analysis.phase2_distribution_head_root_cause_probe import (
    _extract_latent_dataset_with_bucket_idx,
)
from ticl.analysis.phase2_simple_scalar_value_head_probe import (
    _build_algo,
    _collect_rollout,
    _default_device,
)
from ticl.analysis.critic_free_single_env_audit import (
    _build_audit_env_cfg,
    _prepare_audit_fixed_env_contract,
)
from ticl.model_builder import load_model
from ticl.models.tabpfn_regressor_vendor import TabPFNOfficialRegressorHarness


def _make_harness(*, latent_dim: int, num_buckets: int, value_range: float) -> TabPFNOfficialRegressorHarness:
    harness = TabPFNOfficialRegressorHarness(
        emsize=int(latent_dim),
        num_buckets=int(num_buckets),
        value_range=float(value_range),
        ignore_nan_targets=True,
    )
    harness.reset_target_statistics()
    return harness


def _occupancy_stats(
    *,
    bardist,
    returns: torch.Tensor,
    objective_mask: torch.Tensor,
) -> dict:
    mask = objective_mask.to(dtype=torch.bool).reshape(-1)
    y = returns[mask].to(device=bardist.borders.device, dtype=torch.float32)
    bucket_idx = bardist.map_to_bucket_idx(y).clamp(0, int(bardist.num_bars) - 1)
    counts = torch.bincount(bucket_idx, minlength=int(bardist.num_bars)).cpu().numpy()
    nonzero = np.flatnonzero(counts > 0)
    total = int(counts.sum())
    occupied_fraction = float(len(nonzero) / int(bardist.num_bars))
    top = sorted([(int(i), int(c)) for i, c in enumerate(counts.tolist()) if c > 0], key=lambda x: x[1], reverse=True)[:12]
    return {
        "objective_total": int(total),
        "num_buckets": int(bardist.num_bars),
        "occupied_bucket_count": int(len(nonzero)),
        "occupied_fraction": float(occupied_fraction),
        "first_occupied_bucket": None if len(nonzero) == 0 else int(nonzero[0]),
        "last_occupied_bucket": None if len(nonzero) == 0 else int(nonzero[-1]),
        "occupied_bucket_span": 0 if len(nonzero) == 0 else int(nonzero[-1] - nonzero[0] + 1),
        "bucket_histogram_top12": top,
        "return_min": float(y.min().item()) if int(y.numel()) > 0 else 0.0,
        "return_max": float(y.max().item()) if int(y.numel()) > 0 else 0.0,
        "return_std": float(y.std(unbiased=False).item()) if int(y.numel()) > 0 else 0.0,
        "return_mean": float(y.mean().item()) if int(y.numel()) > 0 else 0.0,
    }


def _fit_scalar_linear_head(
    *,
    x: torch.Tensor,
    y: torch.Tensor,
    fit_steps: int,
    lr: float,
    max_grad_norm: float,
) -> dict:
    head = nn.Linear(int(x.shape[-1]), 1).to(device=x.device, dtype=torch.float32)
    optimizer = torch.optim.Adam(head.parameters(), lr=float(lr))
    initial_loss = None
    final_loss = None
    for _ in range(int(fit_steps)):
        preds = head(x).squeeze(-1)
        loss = (preds - y).square().mean()
        if initial_loss is None:
            initial_loss = float(loss.detach().cpu().item())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), float(max_grad_norm))
        optimizer.step()
        final_loss = float(loss.detach().cpu().item())
    return {
        "model": head.cpu(),
        "fit_summary": {
            "fit_steps": int(fit_steps),
            "learning_rate": float(lr),
            "max_grad_norm": float(max_grad_norm),
            "initial_loss": None if initial_loss is None else float(initial_loss),
            "final_loss": None if final_loss is None else float(final_loss),
        },
    }


def _grad_stats_distribution(
    *,
    model: nn.Module,
    bardist,
    x: torch.Tensor,
    y: torch.Tensor,
    bucket_idx: torch.Tensor,
    loss_name: str,
    harness: TabPFNOfficialRegressorHarness,
) -> dict:
    model = copy.deepcopy(model).to(device=x.device, dtype=torch.float32)
    x_in = x.detach().clone().requires_grad_(True)
    logits = model(x_in)
    logits.retain_grad()
    if loss_name == "official_default":
        loss = harness.compute_regression_loss(
            logits=logits.unsqueeze(0),
            targets_BQ=y.unsqueeze(0),
        )
    elif loss_name == "official_mse_only":
        loss = harness.compute_regression_loss(
            logits=logits.unsqueeze(0),
            targets_BQ=y.unsqueeze(0),
            ce_loss_weight=0.0,
            crps_loss_weight=0.0,
            crls_loss_weight=0.0,
            mse_loss_weight=1.0,
            mse_loss_clip=None,
            mae_loss_weight=0.0,
            mae_loss_clip=None,
        )
    elif loss_name == "bucket_nll":
        loss = bardist(logits, y).mean()
    else:
        raise ValueError(loss_name)
    loss.backward()

    target_bucket = bucket_idx.clamp(0, int(bardist.num_bars) - 1)
    grad_abs = logits.grad.detach().abs()
    target_abs = grad_abs.gather(-1, target_bucket.unsqueeze(-1)).squeeze(-1)
    total_abs = grad_abs.sum(dim=-1)
    non_target_abs_mean = (total_abs - target_abs) / max(int(bardist.num_bars) - 1, 1)
    x_grad = x_in.grad.detach()
    x_grad_norm = x_grad.norm(dim=-1)
    param_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_sq += float(p.grad.detach().pow(2).sum().cpu().item())
    return {
        "loss": float(loss.detach().cpu().item()),
        "logits_abs_grad_mean": float(grad_abs.mean().cpu().item()),
        "logits_abs_grad_max": float(grad_abs.max().cpu().item()),
        "target_bucket_abs_grad_mean": float(target_abs.mean().cpu().item()),
        "non_target_abs_grad_mean": float(non_target_abs_mean.mean().cpu().item()),
        "target_to_non_target_grad_ratio": float(target_abs.mean().cpu().item() / max(non_target_abs_mean.mean().cpu().item(), 1e-12)),
        "latent_abs_grad_mean": float(x_grad.abs().mean().cpu().item()),
        "latent_grad_l2_mean": float(x_grad_norm.mean().cpu().item()),
        "latent_grad_l2_max": float(x_grad_norm.max().cpu().item()),
        "head_param_grad_l2": float(param_sq ** 0.5),
    }


def _grad_stats_scalar(
    *,
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
) -> dict:
    model = copy.deepcopy(model).to(device=x.device, dtype=torch.float32)
    x_in = x.detach().clone().requires_grad_(True)
    preds = model(x_in).squeeze(-1)
    preds.retain_grad()
    loss = (preds - y).square().mean()
    loss.backward()
    x_grad = x_in.grad.detach()
    x_grad_norm = x_grad.norm(dim=-1)
    pred_grad = preds.grad.detach().abs()
    param_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_sq += float(p.grad.detach().pow(2).sum().cpu().item())
    return {
        "loss": float(loss.detach().cpu().item()),
        "pred_abs_grad_mean": float(pred_grad.mean().cpu().item()),
        "pred_abs_grad_max": float(pred_grad.max().cpu().item()),
        "latent_abs_grad_mean": float(x_grad.abs().mean().cpu().item()),
        "latent_grad_l2_mean": float(x_grad_norm.mean().cpu().item()),
        "latent_grad_l2_max": float(x_grad_norm.max().cpu().item()),
        "head_param_grad_l2": float(param_sq ** 0.5),
    }


def run_phase2_distribution_target_gradient_probe(
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
    build_seed: int = 4040,
    ppo_reset_env_state_at_sep: bool = True,
    actor_baseline_mode: str = "learned",
    fit_steps: int = 80,
    head_fit_learning_rate: float = 1e-3,
    head_fit_max_grad_norm: float = 0.5,
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
    try:
        _collect_rollout(algo, callback, vec_env, progress_timestep=0, total_timesteps=int(n_steps))
        algo.train()
        dataset = _extract_latent_dataset_with_bucket_idx(algo)
        mask = dataset["objective_mask"]
        x = dataset["latent_vf"][mask].to(device=device_obj, dtype=torch.float32)
        y = dataset["returns"][mask].to(device=device_obj, dtype=torch.float32)

        occupancy = {}
        for num_buckets, value_range in [(100, 5.0), (31, 5.0), (11, 1.0)]:
            harness = _make_harness(
                latent_dim=int(x.shape[-1]),
                num_buckets=int(num_buckets),
                value_range=float(value_range),
            ).to(device_obj)
            occupancy[f"{num_buckets}b_r{str(value_range).replace('.', '_')}"] = _occupancy_stats(
                bardist=harness.znorm_space_bardist_,
                returns=dataset["returns"],
                objective_mask=dataset["objective_mask"],
            )

        harness31 = _make_harness(latent_dim=int(x.shape[-1]), num_buckets=31, value_range=5.0).to(device_obj)
        bucket_idx31 = harness31.znorm_space_bardist_.map_to_bucket_idx(y).clamp(0, 30)

        fitted_official_mlp_default = _fit_arch_loss_variant(
            head=_make_head(latent_dim=int(x.shape[-1]), num_buckets=31, arch="official_mlp").to(device_obj),
            harness=copy.deepcopy(harness31),
            x=x,
            y=y,
            bucket_idx=bucket_idx31,
            fit_steps=int(fit_steps),
            lr=float(head_fit_learning_rate),
            max_grad_norm=float(head_fit_max_grad_norm),
            loss_name="official_default",
        )
        fitted_official_mlp_mse = _fit_arch_loss_variant(
            head=_make_head(latent_dim=int(x.shape[-1]), num_buckets=31, arch="official_mlp").to(device_obj),
            harness=copy.deepcopy(harness31),
            x=x,
            y=y,
            bucket_idx=bucket_idx31,
            fit_steps=int(fit_steps),
            lr=float(head_fit_learning_rate),
            max_grad_norm=float(head_fit_max_grad_norm),
            loss_name="official_mse_only",
        )
        fitted_scalar = _fit_scalar_linear_head(
            x=x,
            y=y,
            fit_steps=int(fit_steps),
            lr=float(head_fit_learning_rate),
            max_grad_norm=float(head_fit_max_grad_norm),
        )

        gradient_stats = {
            "official_mlp_31b_r5__official_default": _grad_stats_distribution(
                model=fitted_official_mlp_default["model"],
                bardist=harness31.znorm_space_bardist_,
                x=x,
                y=y,
                bucket_idx=bucket_idx31,
                loss_name="official_default",
                harness=copy.deepcopy(harness31),
            ),
            "official_mlp_31b_r5__official_mse_only": _grad_stats_distribution(
                model=fitted_official_mlp_mse["model"],
                bardist=harness31.znorm_space_bardist_,
                x=x,
                y=y,
                bucket_idx=bucket_idx31,
                loss_name="official_mse_only",
                harness=copy.deepcopy(harness31),
            ),
            "scalar_linear__mse": _grad_stats_scalar(
                model=fitted_scalar["model"],
                x=x,
                y=y,
            ),
        }

        return {
            "audit_entry": "phase2_distribution_target_gradient_probe",
            "checkpoint_path": str(Path(checkpoint_path).expanduser().resolve()),
            "device": str(device_obj),
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
            "ppo_reset_env_state_at_sep": bool(ppo_reset_env_state_at_sep),
            "actor_baseline_mode": str(actor_baseline_mode),
            "fit_steps": int(fit_steps),
            "head_fit_learning_rate": float(head_fit_learning_rate),
            "head_fit_max_grad_norm": float(head_fit_max_grad_norm),
            "fixed_env_contract": dict(fixed_bundle["fixed_env_contract"]),
            "occupancy": occupancy,
            "gradient_stats": gradient_stats,
        }
    finally:
        vec_env.close()
        del algo, callback, vec_env
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--build-seed", type=int, default=4040)
    parser.add_argument("--ppo-reset-env-state-at-sep", action="store_true")
    parser.add_argument("--actor-baseline-mode", type=str, default="learned")
    parser.add_argument("--fit-steps", type=int, default=80)
    parser.add_argument("--head-fit-learning-rate", type=float, default=1e-3)
    parser.add_argument("--head-fit-max-grad-norm", type=float, default=0.5)
    parser.add_argument(
        "--output-json",
        type=str,
        default="/home/chen/RLPFN/artifacts/phase2_distribution_target_gradient_probe.json",
    )
    args = parser.parse_args()
    report = run_phase2_distribution_target_gradient_probe(
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
        build_seed=int(args.build_seed),
        ppo_reset_env_state_at_sep=bool(args.ppo_reset_env_state_at_sep),
        actor_baseline_mode=str(args.actor_baseline_mode),
        fit_steps=int(args.fit_steps),
        head_fit_learning_rate=float(args.head_fit_learning_rate),
        head_fit_max_grad_norm=float(args.head_fit_max_grad_norm),
    )
    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
