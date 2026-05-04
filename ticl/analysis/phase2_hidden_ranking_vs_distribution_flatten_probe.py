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
    _suite_from_seed_range,
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


def _safe_rank(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty_like(arr, dtype=np.float32)
    i = 0
    while i < int(order.size):
        j = i + 1
        while j < int(order.size) and float(arr[order[j]]) == float(arr[order[i]]):
            j += 1
        rank = 0.5 * float(i + j - 1) + 1.0
        ranks[order[i:j]] = rank
        i = j
    return ranks


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if int(x.size) <= 1 or int(y.size) <= 1:
        return 0.0
    return _safe_corrcoef(_safe_rank(x), _safe_rank(y))


def _pairwise_order_accuracy(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    n = int(x.size)
    if n <= 1:
        return 0.0
    idx_i, idx_j = np.triu_indices(n, k=1)
    dx = x[idx_i] - x[idx_j]
    dy = y[idx_i] - y[idx_j]
    valid = np.abs(dy) > 1e-8
    if int(valid.sum()) <= 0:
        return 0.0
    agree = np.sign(dx[valid]) == np.sign(dy[valid])
    return float(np.asarray(agree, dtype=np.float32).mean())


def _masked_stats(preds: np.ndarray, targets: np.ndarray, mask: np.ndarray) -> dict:
    preds = np.asarray(preds, dtype=np.float32).reshape(-1)
    targets = np.asarray(targets, dtype=np.float32).reshape(-1)
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if int(mask.sum()) <= 1:
        return {
            "count": int(mask.sum()),
            "pred_std": 0.0,
            "target_std": 0.0,
            "pearson": 0.0,
            "spearman": 0.0,
            "pairwise_order_acc": 0.0,
        }
    p = preds[mask]
    t = targets[mask]
    return {
        "count": int(mask.sum()),
        "pred_std": float(np.asarray(p, dtype=np.float32).std()),
        "target_std": float(np.asarray(t, dtype=np.float32).std()),
        "pearson": _safe_corrcoef(p, t),
        "spearman": _safe_spearman(p, t),
        "pairwise_order_acc": _pairwise_order_accuracy(p, t),
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


def _bucket_entropy(logits: torch.Tensor) -> float:
    probs = torch.softmax(logits.to(dtype=torch.float32), dim=-1)
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1).mean()
    return float(entropy.detach().cpu().item())


def _central_subset_mask(
    bardist,
    targets: torch.Tensor,
    objective_mask: torch.Tensor,
    *,
    radius: int,
) -> np.ndarray:
    target_device = bardist.borders.device
    bucket_idx = bardist.map_to_bucket_idx(targets.to(device=target_device, dtype=torch.float32))
    center = int(bardist.num_bars // 2)
    mask = (
        (bucket_idx >= max(0, center - int(radius)))
        & (bucket_idx <= min(int(bardist.num_bars) - 1, center + int(radius)))
        & objective_mask.to(device=target_device, dtype=torch.bool)
    )
    return mask.detach().cpu().numpy().reshape(-1).astype(bool)


@torch.no_grad()
def _evaluate_readout(
    preds: torch.Tensor,
    dataset: dict,
    *,
    central_mask_np: np.ndarray,
) -> dict:
    preds_np = preds.detach().cpu().numpy().reshape(-1)
    targets_np = dataset["returns"].detach().cpu().numpy().reshape(-1)
    objective_mask_np = dataset["objective_mask"].detach().cpu().numpy().reshape(-1).astype(bool)
    return {
        "overall": _masked_stats(preds_np, targets_np, objective_mask_np),
        "central_subset": _masked_stats(preds_np, targets_np, central_mask_np),
    }


def run_phase2_hidden_ranking_vs_distribution_flatten_probe(
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
    center_bucket_radius: int = 1,
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
    _suite_from_seed_range(
        frozen_h=frozen_h,
        env_seed_start=3000,
        rollout_seed_start=4000,
        count=4,
    )  # keep trusted contract assumptions aligned; suite not used directly here

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
            _collect_rollout(
                algo,
                callback,
                vec_env,
                progress_timestep=progress_next,
                total_timesteps=total_timesteps,
            )
            next_dataset = _extract_latent_dataset(algo)

            latent_dim = int(same_dataset["latent_vf"].shape[-1])
            harness = _make_vendor_harness(
                latent_dim=latent_dim,
                num_buckets=int(num_buckets),
                value_range=float(value_range),
            ).to(device_obj)
            x_same = same_dataset["latent_vf"][same_dataset["objective_mask"]].to(device=device_obj, dtype=torch.float32)
            y_same = same_dataset["returns"][same_dataset["objective_mask"]].to(device=device_obj, dtype=torch.float32)
            bucket_idx_same = harness.znorm_space_bardist_.map_to_bucket_idx(y_same)

            per_step = {}
            for fit_steps in [int(v) for v in fit_steps_schedule]:
                variant = _fit_distribution_variant(
                    head_module=copy.deepcopy(harness.output_projection),
                    bardist=copy.deepcopy(harness.znorm_space_bardist_),
                    x=x_same,
                    y=y_same,
                    bucket_idx=bucket_idx_same,
                    fit_steps=int(fit_steps),
                    lr=float(head_fit_learning_rate),
                    max_grad_norm=float(head_fit_max_grad_norm),
                    loss_name="vendor_official",
                    official_harness=copy.deepcopy(harness),
                )
                model = variant["model"].to(device_obj)
                bardist = variant["bardist"].to(device_obj)
                same_logits = model(same_dataset["latent_vf"].to(device=device_obj, dtype=torch.float32))
                next_logits = model(next_dataset["latent_vf"].to(device=device_obj, dtype=torch.float32))
                logits_scalar_head = _fit_ridge_scalar_head(
                    same_logits.detach().cpu().to(dtype=torch.float32),
                    same_dataset["returns"],
                    same_dataset["objective_mask"],
                    ridge_lambda=1e-3,
                )

                same_mean_preds = bardist.mean(same_logits.to(dtype=torch.float32)).detach().cpu().reshape(-1)
                next_mean_preds = bardist.mean(next_logits.to(dtype=torch.float32)).detach().cpu().reshape(-1)
                same_hidden_preds = _apply_scalar_head(same_dataset["latent_vf"], hidden_scalar_head)
                next_hidden_preds = _apply_scalar_head(next_dataset["latent_vf"], hidden_scalar_head)
                same_logits_preds = _apply_scalar_head(same_logits.detach().cpu().to(dtype=torch.float32), logits_scalar_head)
                next_logits_preds = _apply_scalar_head(next_logits.detach().cpu().to(dtype=torch.float32), logits_scalar_head)

                same_central_mask_np = _central_subset_mask(
                    bardist,
                    same_dataset["returns"],
                    same_dataset["objective_mask"],
                    radius=int(center_bucket_radius),
                )
                next_central_mask_np = _central_subset_mask(
                    bardist,
                    next_dataset["returns"],
                    next_dataset["objective_mask"],
                    radius=int(center_bucket_radius),
                )

                central_probs_same = torch.softmax(same_logits.to(dtype=torch.float32), dim=-1)
                center_idx = int(bardist.num_bars // 2)
                lo = max(0, center_idx - int(center_bucket_radius))
                hi = min(int(bardist.num_bars), center_idx + int(center_bucket_radius) + 1)
                central_band_mass_same = central_probs_same[:, lo:hi].sum(dim=-1)
                argmax_same = same_logits.argmax(dim=-1)

                per_step[str(fit_steps)] = {
                    "fit_summary": dict(variant["fit_summary"]),
                    "distribution_state": {
                        "bucket_entropy_same": _bucket_entropy(same_logits),
                        "mean_pred_std_same": float(np.asarray(same_mean_preds.numpy(), dtype=np.float32)[
                            same_dataset["objective_mask"].detach().cpu().numpy().astype(bool)
                        ].std()),
                        "central_band_mass_same": float(
                            central_band_mass_same[same_dataset["objective_mask"].to(device=device_obj, dtype=torch.bool)]
                            .mean()
                            .detach()
                            .cpu()
                            .item()
                        ),
                        "central_bucket_argmax_fraction_same": float(
                            (
                                argmax_same[same_dataset["objective_mask"].to(device=device_obj, dtype=torch.bool)] == center_idx
                            )
                            .to(dtype=torch.float32)
                            .mean()
                            .detach()
                            .cpu()
                            .item()
                        ),
                    },
                    "same_rollout": {
                        "hidden_scalar": _evaluate_readout(
                            same_hidden_preds,
                            same_dataset,
                            central_mask_np=same_central_mask_np,
                        ),
                        "distribution_mean": _evaluate_readout(
                            same_mean_preds,
                            same_dataset,
                            central_mask_np=same_central_mask_np,
                        ),
                        "logits_scalar": _evaluate_readout(
                            same_logits_preds,
                            same_dataset,
                            central_mask_np=same_central_mask_np,
                        ),
                    },
                    "next_rollout": {
                        "hidden_scalar": _evaluate_readout(
                            next_hidden_preds,
                            next_dataset,
                            central_mask_np=next_central_mask_np,
                        ),
                        "distribution_mean": _evaluate_readout(
                            next_mean_preds,
                            next_dataset,
                            central_mask_np=next_central_mask_np,
                        ),
                        "logits_scalar": _evaluate_readout(
                            next_logits_preds,
                            next_dataset,
                            central_mask_np=next_central_mask_np,
                        ),
                    },
                }

            history.append(
                {
                    "outer_epoch": int(outer_idx + 1),
                    "fit_steps_schedule": [int(v) for v in fit_steps_schedule],
                    "distribution_spec": {
                        "num_buckets": int(num_buckets),
                        "value_range": float(value_range),
                        "center_bucket_radius": int(center_bucket_radius),
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
        description="Probe whether ranking information exists in critic hidden and whether distribution mean decode flattens it too early."
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
    parser.add_argument("--center-bucket-radius", type=int, default=1)
    parser.add_argument("--head-fit-learning-rate", type=float, default=1e-3)
    parser.add_argument("--head-fit-max-grad-norm", type=float, default=0.5)
    parser.add_argument("--output-json", type=str, required=True)
    args = parser.parse_args()

    result = run_phase2_hidden_ranking_vs_distribution_flatten_probe(
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
        center_bucket_radius=int(args.center_bucket_radius),
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
