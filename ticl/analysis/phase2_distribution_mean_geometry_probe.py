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


def _evaluate_predictions(preds: np.ndarray, targets: np.ndarray, mask: np.ndarray) -> dict:
    preds = np.asarray(preds, dtype=np.float32).reshape(-1)
    targets = np.asarray(targets, dtype=np.float32).reshape(-1)
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if int(mask.sum()) <= 1:
        return {
            "pearson": 0.0,
            "pred_std": 0.0,
            "target_std": 0.0,
        }
    p = preds[mask]
    t = targets[mask]
    return {
        "pearson": _safe_corrcoef(p, t),
        "pred_std": float(np.asarray(p, dtype=np.float32).std()),
        "target_std": float(np.asarray(t, dtype=np.float32).std()),
    }


def _apply_scalar_head(x: torch.Tensor, head: dict) -> torch.Tensor:
    return x.to(dtype=torch.float32) @ head["weight"] + head["bias"]


def _make_vendor_harness(*, latent_dim: int, num_buckets: int, value_range: float) -> TabPFNOfficialRegressorHarness:
    harness = TabPFNOfficialRegressorHarness(
        emsize=int(latent_dim),
        num_buckets=int(num_buckets),
        value_range=float(value_range),
        ignore_nan_targets=True,
    )
    harness.reset_target_statistics()
    return harness


def _build_control_latent(source: torch.Tensor, latent_dim: int) -> torch.Tensor:
    source = source.detach().cpu().to(dtype=torch.float32).reshape(-1)
    latent = torch.zeros((int(source.shape[0]), int(latent_dim)), dtype=torch.float32)
    latent[:, 0] = source
    return latent


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


def _shell_contribution_stats(probs: np.ndarray, centers: np.ndarray, center_idx: int, mask: np.ndarray, max_shells: int = 4) -> dict:
    probs = np.asarray(probs, dtype=np.float32)
    centers = np.asarray(centers, dtype=np.float32).reshape(-1)
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    center_value = float(centers[int(center_idx)])
    bucket_width = float(centers[min(int(center_idx) + 1, int(len(centers) - 1))] - centers[int(center_idx)]) if len(centers) > 1 else 0.0
    out = {
        "center_value": center_value,
        "bucket_width": bucket_width,
        "shells": {},
    }
    if int(mask.sum()) <= 0:
        return out
    p = probs[mask]
    for k in range(1, int(max_shells) + 1):
        left = int(center_idx - k)
        right = int(center_idx + k)
        if left < 0 or right >= int(probs.shape[1]):
            continue
        prob_diff = p[:, right] - p[:, left]
        contrib = prob_diff * float(k) * bucket_width
        out["shells"][str(k)] = {
            "prob_diff_mean": float(prob_diff.mean()),
            "prob_diff_std": float(prob_diff.std()),
            "contribution_mean": float(contrib.mean()),
            "contribution_std": float(contrib.std()),
        }
    return out


def _mean_geometry_metrics(logits: torch.Tensor, bardist, targets: torch.Tensor, objective_mask: torch.Tensor) -> dict:
    logits = logits.to(dtype=torch.float32)
    probs = torch.softmax(logits, dim=-1)
    means = bardist.mean(logits).detach().cpu().numpy().reshape(-1)
    mask_np = objective_mask.detach().cpu().numpy().reshape(-1).astype(bool)
    probs_np = probs.detach().cpu().numpy()
    centers_np = (bardist.borders[:-1] + 0.5 * bardist.bucket_widths).detach().cpu().numpy()
    center_idx = int(bardist.num_bars // 2)
    center_value = float(centers_np[center_idx])
    mean_offset = means - center_value
    central_probs = probs[:, max(0, center_idx - 1) : min(int(bardist.num_bars), center_idx + 2)].sum(dim=-1)
    argmax_idx = logits.argmax(dim=-1)
    jac = probs * (torch.as_tensor(centers_np, device=logits.device, dtype=torch.float32) - bardist.mean(logits).unsqueeze(-1))
    jac_abs = jac.abs()
    jac_l2 = torch.linalg.vector_norm(jac, dim=-1)
    return {
        "prediction_metrics": _evaluate_predictions(
            means,
            targets.detach().cpu().numpy().reshape(-1),
            mask_np,
        ),
        "center_band_mass_mean": float(central_probs[objective_mask.to(device=logits.device, dtype=torch.bool)].mean().detach().cpu().item()),
        "center_band_mass_std": float(central_probs[objective_mask.to(device=logits.device, dtype=torch.bool)].std(unbiased=False).detach().cpu().item()),
        "center_argmax_fraction": float(
            (argmax_idx[objective_mask.to(device=logits.device, dtype=torch.bool)] == center_idx)
            .to(dtype=torch.float32)
            .mean()
            .detach()
            .cpu()
            .item()
        ),
        "mean_offset_from_center_mean": float(np.asarray(mean_offset[mask_np], dtype=np.float32).mean()) if int(mask_np.sum()) > 0 else 0.0,
        "mean_offset_from_center_std": float(np.asarray(mean_offset[mask_np], dtype=np.float32).std()) if int(mask_np.sum()) > 0 else 0.0,
        "jacobian_abs_mean": float(jac_abs[objective_mask.to(device=logits.device, dtype=torch.bool)].mean().detach().cpu().item()),
        "jacobian_l2_mean": float(jac_l2[objective_mask.to(device=logits.device, dtype=torch.bool)].mean().detach().cpu().item()),
        "shell_contributions": _shell_contribution_stats(
            probs_np,
            centers_np,
            center_idx,
            mask_np,
            max_shells=4,
        ),
    }


def run_phase2_distribution_mean_geometry_probe(
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
    fit_steps_schedule: tuple[int, ...] = (0, 1, 2, 5, 10, 20),
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
            progress_next = min(int((outer_idx + 1) * n_steps), total_timesteps)
            _collect_rollout(algo, callback, vec_env, progress_timestep=progress_next, total_timesteps=total_timesteps)
            next_dataset = _extract_latent_dataset(algo)

            latent_dim = int(same_dataset["latent_vf"].shape[-1])
            same_hidden_scalar_pred = _apply_scalar_head(same_dataset["latent_vf"], hidden_scalar_head)
            next_hidden_scalar_pred = _apply_scalar_head(next_dataset["latent_vf"], hidden_scalar_head)
            control_same_hidden_scalar = _build_control_latent(same_hidden_scalar_pred, latent_dim)
            control_next_hidden_scalar = _build_control_latent(next_hidden_scalar_pred, latent_dim)
            control_same_target_oracle = _build_control_latent(same_dataset["returns"], latent_dim)
            control_next_target_oracle = _build_control_latent(next_dataset["returns"], latent_dim)

            source_specs = {
                "critic_hidden": {
                    "same_x": same_dataset["latent_vf"],
                    "next_x": next_dataset["latent_vf"],
                },
                "hidden_scalar_control": {
                    "same_x": control_same_hidden_scalar,
                    "next_x": control_next_hidden_scalar,
                },
                "target_oracle_control": {
                    "same_x": control_same_target_oracle,
                    "next_x": control_next_target_oracle,
                },
            }

            per_step = {}
            for fit_steps in [int(v) for v in fit_steps_schedule]:
                step_metrics = {}
                for source_name, spec in source_specs.items():
                    variant = _fit_official_distribution_head(
                        x=spec["same_x"][same_dataset["objective_mask"]].to(device=device_obj, dtype=torch.float32),
                        y=same_dataset["returns"][same_dataset["objective_mask"]].to(device=device_obj, dtype=torch.float32),
                        latent_dim=latent_dim,
                        num_buckets=int(num_buckets),
                        value_range=float(value_range),
                        fit_steps=int(fit_steps),
                        lr=float(head_fit_learning_rate),
                        max_grad_norm=float(head_fit_max_grad_norm),
                        device=device_obj,
                    )
                    model = variant["model"].to(device_obj)
                    bardist = variant["bardist"].to(device_obj)
                    same_logits = model(spec["same_x"].to(device=device_obj, dtype=torch.float32))
                    next_logits = model(spec["next_x"].to(device=device_obj, dtype=torch.float32))
                    step_metrics[source_name] = {
                        "fit_summary": dict(variant["fit_summary"]),
                        "same_rollout": _mean_geometry_metrics(
                            same_logits,
                            bardist,
                            same_dataset["returns"],
                            same_dataset["objective_mask"],
                        ),
                        "next_rollout": _mean_geometry_metrics(
                            next_logits,
                            bardist,
                            next_dataset["returns"],
                            next_dataset["objective_mask"],
                        ),
                    }
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
        description="Analyze why bar mean geometry flattens scalar value differences and compare critic hidden against scalar-aligned controls."
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

    result = run_phase2_distribution_mean_geometry_probe(
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
