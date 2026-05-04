import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

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


def _direction_from_ridge_head(head: dict) -> torch.Tensor:
    w = head["weight"].detach().cpu().to(dtype=torch.float32).reshape(-1)
    norm = float(torch.linalg.vector_norm(w).item())
    if norm <= 0.0:
        raise RuntimeError("Ridge direction has zero norm.")
    return w / norm


def _correlate_output_with_direction(output: torch.Tensor, latent: torch.Tensor, direction: torch.Tensor, mask: torch.Tensor) -> dict:
    direction_proj = (latent.to(dtype=torch.float32) @ direction.to(dtype=torch.float32)).detach().cpu().numpy().reshape(-1)
    output_np = output.detach().cpu().numpy().reshape(-1)
    mask_np = mask.detach().cpu().numpy().reshape(-1).astype(bool)
    if int(mask_np.sum()) <= 1:
        return {
            "output_std": 0.0,
            "direction_proj_std": 0.0,
            "corr_with_direction_proj": 0.0,
        }
    return {
        "output_std": float(np.asarray(output_np[mask_np], dtype=np.float32).std()),
        "direction_proj_std": float(np.asarray(direction_proj[mask_np], dtype=np.float32).std()),
        "corr_with_direction_proj": _safe_corrcoef(output_np[mask_np], direction_proj[mask_np]),
    }


def _per_sample_gradients(latent: torch.Tensor, scalar_output: torch.Tensor) -> torch.Tensor:
    latent_var = latent.detach().clone().requires_grad_(True)
    scalar = scalar_output(latent_var)
    grads = torch.autograd.grad(
        outputs=scalar.sum(),
        inputs=latent_var,
        retain_graph=False,
        create_graph=False,
        allow_unused=False,
    )[0]
    return grads.detach().cpu()


def _gradient_alignment_summary(grads: torch.Tensor, direction: torch.Tensor, mask: torch.Tensor) -> dict:
    grads = grads.detach().cpu().to(dtype=torch.float32)
    direction = direction.detach().cpu().to(dtype=torch.float32).reshape(1, -1)
    mask_np = mask.detach().cpu().numpy().reshape(-1).astype(bool)
    if int(mask_np.sum()) <= 0:
        return {
            "grad_norm_mean": 0.0,
            "grad_norm_std": 0.0,
            "cosine_mean": 0.0,
            "cosine_abs_mean": 0.0,
            "directional_derivative_mean": 0.0,
            "directional_derivative_std": 0.0,
        }
    g = grads[mask_np]
    g_norm = torch.linalg.vector_norm(g, dim=-1)
    safe = torch.clamp(g_norm, min=1e-12)
    directional = (g * direction).sum(dim=-1)
    cosine = directional / safe
    return {
        "grad_norm_mean": float(g_norm.mean().item()),
        "grad_norm_std": float(g_norm.std(unbiased=False).item()),
        "cosine_mean": float(cosine.mean().item()),
        "cosine_abs_mean": float(cosine.abs().mean().item()),
        "directional_derivative_mean": float(directional.mean().item()),
        "directional_derivative_std": float(directional.std(unbiased=False).item()),
    }


def _mean_scalar_fn(model, bardist):
    def fn(latent: torch.Tensor) -> torch.Tensor:
        logits = model(latent.to(dtype=torch.float32))
        return bardist.mean(logits.to(dtype=torch.float32))

    return fn


def _shell_diff_scalar_fn(model, center_idx: int, k: int):
    def fn(latent: torch.Tensor) -> torch.Tensor:
        logits = model(latent.to(dtype=torch.float32)).to(dtype=torch.float32)
        return logits[:, center_idx + int(k)] - logits[:, center_idx - int(k)]

    return fn


@torch.no_grad()
def _evaluate_output_direction_relation(
    *,
    latent: torch.Tensor,
    mask: torch.Tensor,
    targets: torch.Tensor,
    direction: torch.Tensor,
    model,
    bardist,
) -> dict:
    latent_dev = latent.to(next(model.parameters()).device, dtype=torch.float32)
    logits = model(latent_dev).to(dtype=torch.float32)
    mean_out = bardist.mean(logits).detach().cpu()
    center_idx = int(bardist.num_bars // 2)
    out = {
        "mean_decode": {
            **_correlate_output_with_direction(mean_out, latent, direction, mask),
            "corr_with_target": _safe_corrcoef(
                mean_out.detach().cpu().numpy()[mask.detach().cpu().numpy().astype(bool)],
                targets.detach().cpu().numpy()[mask.detach().cpu().numpy().astype(bool)],
            ),
        }
    }
    for k in (1, 2, 3, 4):
        if center_idx - k < 0 or center_idx + k >= int(bardist.num_bars):
            continue
        shell_out = (logits[:, center_idx + k] - logits[:, center_idx - k]).detach().cpu()
        out[f"shell_diff_{k}"] = {
            **_correlate_output_with_direction(shell_out, latent, direction, mask),
            "corr_with_target": _safe_corrcoef(
                shell_out.detach().cpu().numpy()[mask.detach().cpu().numpy().astype(bool)],
                targets.detach().cpu().numpy()[mask.detach().cpu().numpy().astype(bool)],
            ),
        }
    return out


def run_phase2_hidden_direction_vs_output_projection_probe(
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

            latent_dim = int(same_dataset["latent_vf"].shape[-1])
            same_hidden_scalar_pred = _apply_scalar_head(same_dataset["latent_vf"], hidden_scalar_head)
            next_hidden_scalar_pred = _apply_scalar_head(next_dataset["latent_vf"], hidden_scalar_head)
            source_specs = {
                "critic_hidden": {
                    "same_x": same_dataset["latent_vf"],
                    "next_x": next_dataset["latent_vf"],
                },
                "hidden_scalar_control": {
                    "same_x": _build_control_latent(same_hidden_scalar_pred, latent_dim),
                    "next_x": _build_control_latent(next_hidden_scalar_pred, latent_dim),
                },
            }

            source_directions = {}
            for source_name, spec in source_specs.items():
                ridge = _fit_ridge_scalar_head(
                    spec["same_x"],
                    same_dataset["returns"],
                    same_dataset["objective_mask"],
                    ridge_lambda=1e-3,
                )
                source_directions[source_name] = _direction_from_ridge_head(ridge)

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
                    direction = source_directions[source_name]

                    same_relation = _evaluate_output_direction_relation(
                        latent=spec["same_x"],
                        mask=same_dataset["objective_mask"],
                        targets=same_dataset["returns"],
                        direction=direction,
                        model=model,
                        bardist=bardist,
                    )
                    next_relation = _evaluate_output_direction_relation(
                        latent=spec["next_x"],
                        mask=next_dataset["objective_mask"],
                        targets=next_dataset["returns"],
                        direction=direction,
                        model=model,
                        bardist=bardist,
                    )

                    same_latent_dev = spec["same_x"].to(device=device_obj, dtype=torch.float32)
                    mean_grads = _per_sample_gradients(
                        same_latent_dev,
                        _mean_scalar_fn(model, bardist),
                    )
                    grad_summary = {
                        "mean_decode": _gradient_alignment_summary(
                            mean_grads,
                            direction,
                            same_dataset["objective_mask"],
                        )
                    }
                    center_idx = int(bardist.num_bars // 2)
                    for k in (1, 2, 3, 4):
                        if center_idx - k < 0 or center_idx + k >= int(bardist.num_bars):
                            continue
                        shell_grads = _per_sample_gradients(
                            same_latent_dev,
                            _shell_diff_scalar_fn(model, center_idx, k),
                        )
                        grad_summary[f"shell_diff_{k}"] = _gradient_alignment_summary(
                            shell_grads,
                            direction,
                            same_dataset["objective_mask"],
                        )

                    step_metrics[source_name] = {
                        "fit_summary": dict(variant["fit_summary"]),
                        "direction_summary": {
                            "direction_norm": float(torch.linalg.vector_norm(direction).item()),
                            "same_direction_proj_std": float(
                                np.asarray(
                                    (spec["same_x"] @ direction).detach().cpu().numpy()[
                                        same_dataset["objective_mask"].detach().cpu().numpy().astype(bool)
                                    ],
                                    dtype=np.float32,
                                ).std()
                            ),
                            "next_direction_proj_std": float(
                                np.asarray(
                                    (spec["next_x"] @ direction).detach().cpu().numpy()[
                                        next_dataset["objective_mask"].detach().cpu().numpy().astype(bool)
                                    ],
                                    dtype=np.float32,
                                ).std()
                            ),
                        },
                        "same_rollout": same_relation,
                        "next_rollout": next_relation,
                        "same_rollout_gradient_alignment": grad_summary,
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
        description="Analyze which hidden directions carry ranking information and whether official output_projection aligns with them."
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

    result = run_phase2_hidden_direction_vs_output_projection_probe(
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
