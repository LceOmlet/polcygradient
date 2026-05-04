import argparse
import copy
import json
from pathlib import Path

import torch

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
from ticl.models.vendor.tabpfn_v7_1_1.runtime_regression_loss import (
    _ranked_probability_score_loss_from_bar_logits,
)


def _make_harness(*, latent_dim: int) -> TabPFNOfficialRegressorHarness:
    harness = TabPFNOfficialRegressorHarness(
        emsize=int(latent_dim),
        num_buckets=31,
        value_range=5.0,
        ignore_nan_targets=True,
    )
    harness.reset_target_statistics()
    return harness


def _mean_jacobian(logits: torch.Tensor, bucket_means: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    probs = torch.softmax(logits.to(dtype=torch.float32), dim=-1)
    mean = probs @ bucket_means
    jac = probs * (bucket_means.unsqueeze(0) - mean.unsqueeze(-1))
    return mean, jac


def _vector_stats(t: torch.Tensor) -> dict:
    flat = t.reshape(-1)
    return {
        "abs_mean": float(flat.abs().mean().detach().cpu().item()),
        "abs_max": float(flat.abs().max().detach().cpu().item()),
        "l2": float(flat.norm().detach().cpu().item()),
    }


def _samplewise_direction_stats(grad: torch.Tensor, ref: torch.Tensor) -> dict:
    grad_f = grad.reshape(grad.shape[0], -1)
    ref_f = ref.reshape(ref.shape[0], -1)
    grad_norm = grad_f.norm(dim=-1)
    ref_norm = ref_f.norm(dim=-1)
    denom = (grad_norm * ref_norm).clamp_min(1e-12)
    cosine = (grad_f * ref_f).sum(dim=-1) / denom
    projection = (grad_f * ref_f).sum(dim=-1) / ref_norm.clamp_min(1e-12)
    return {
        "cosine_mean": float(cosine.mean().detach().cpu().item()),
        "cosine_min": float(cosine.min().detach().cpu().item()),
        "cosine_max": float(cosine.max().detach().cpu().item()),
        "projection_mean": float(projection.mean().detach().cpu().item()),
        "projection_abs_mean": float(projection.abs().mean().detach().cpu().item()),
    }


def _pairwise_grad_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)
    denom = (a_flat.norm() * b_flat.norm()).clamp_min(1e-12)
    return float(((a_flat * b_flat).sum() / denom).detach().cpu().item())


def _loss_grad_logits(
    *,
    model,
    x: torch.Tensor,
    y: torch.Tensor,
    harness: TabPFNOfficialRegressorHarness,
) -> dict:
    model = copy.deepcopy(model).to(device=x.device, dtype=torch.float32)
    x_in = x.detach().clone().requires_grad_(True)
    logits = model(x_in)
    logits.retain_grad()
    bardist = harness.znorm_space_bardist_
    bucket_means = (bardist.borders[:-1] + 0.5 * bardist.bucket_widths).to(device=logits.device, dtype=logits.dtype)
    mean_pred, jac = _mean_jacobian(logits, bucket_means)

    losses = {}

    def run_loss(name: str, loss: torch.Tensor) -> None:
        model.zero_grad(set_to_none=True)
        if x_in.grad is not None:
            x_in.grad.zero_()
        if logits.grad is not None:
            logits.grad.zero_()
        loss.backward(retain_graph=True)
        g = logits.grad.detach().clone()
        losses[name] = {
            "loss": float(loss.detach().cpu().item()),
            "logits_grad": g,
            "logits_grad_stats": _vector_stats(g),
            "direction_vs_mean_jacobian": _samplewise_direction_stats(g, jac),
        }

    crps = _ranked_probability_score_loss_from_bar_logits(
        logits_BQL=logits.unsqueeze(0),
        targets_BQ=y.unsqueeze(0),
        bardist_loss_fn=bardist,
        loss_type="crps",
    )
    mse = (mean_pred - y).square().mean()
    nll = bardist(logits, y).mean()
    combined = crps + mse

    run_loss("crps", crps)
    run_loss("mse", mse)
    run_loss("nll", nll)
    run_loss("combined_crps_plus_mse", combined)

    losses["pairwise_cosine"] = {
        "crps_vs_mse": _pairwise_grad_cosine(losses["crps"]["logits_grad"], losses["mse"]["logits_grad"]),
        "crps_vs_nll": _pairwise_grad_cosine(losses["crps"]["logits_grad"], losses["nll"]["logits_grad"]),
        "mse_vs_nll": _pairwise_grad_cosine(losses["mse"]["logits_grad"], losses["nll"]["logits_grad"]),
        "combined_vs_mse": _pairwise_grad_cosine(losses["combined_crps_plus_mse"]["logits_grad"], losses["mse"]["logits_grad"]),
        "combined_vs_crps": _pairwise_grad_cosine(losses["combined_crps_plus_mse"]["logits_grad"], losses["crps"]["logits_grad"]),
    }

    losses["mean_jacobian_stats"] = {
        "jacobian_abs_mean": float(jac.abs().mean().detach().cpu().item()),
        "jacobian_abs_max": float(jac.abs().max().detach().cpu().item()),
        "jacobian_l2_mean": float(jac.reshape(jac.shape[0], -1).norm(dim=-1).mean().detach().cpu().item()),
        "mean_pred_mean": float(mean_pred.mean().detach().cpu().item()),
        "mean_pred_std": float(mean_pred.std(unbiased=False).detach().cpu().item()),
    }

    # Remove raw tensors from report payload.
    for key in ["crps", "mse", "nll", "combined_crps_plus_mse"]:
        losses[key].pop("logits_grad", None)
    return losses


def run_phase2_distribution_jacobian_loss_decomposition_probe(
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
        harness = _make_harness(latent_dim=int(x.shape[-1])).to(device_obj)
        bucket_idx = harness.znorm_space_bardist_.map_to_bucket_idx(y).clamp(0, 30)
        fitted = _fit_arch_loss_variant(
            head=_make_head(latent_dim=int(x.shape[-1]), num_buckets=31, arch="official_mlp").to(device_obj),
            harness=copy.deepcopy(harness),
            x=x,
            y=y,
            bucket_idx=bucket_idx,
            fit_steps=int(fit_steps),
            lr=float(head_fit_learning_rate),
            max_grad_norm=float(head_fit_max_grad_norm),
            loss_name="official_default",
        )
        decomp = _loss_grad_logits(
            model=fitted["model"],
            x=x,
            y=y,
            harness=copy.deepcopy(harness).to(device_obj),
        )
        return {
            "audit_entry": "phase2_distribution_jacobian_loss_decomposition_probe",
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
            "fit_summary": dict(fitted["fit_summary"]),
            "decomposition": decomp,
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
        default="/home/chen/RLPFN/artifacts/phase2_distribution_jacobian_loss_decomposition_probe.json",
    )
    args = parser.parse_args()
    report = run_phase2_distribution_jacobian_loss_decomposition_probe(
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
