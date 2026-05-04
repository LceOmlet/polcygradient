import argparse
import copy
import json
from pathlib import Path

import torch

from ticl.analysis.phase2_distribution_head_root_cause_probe import (
    _bucket_entropy,
    _extract_latent_dataset_with_bucket_idx,
)
from ticl.analysis.phase2_simple_scalar_value_head_probe import (
    _build_algo,
    _collect_rollout,
    _default_device,
    _evaluate_predictions,
)
from ticl.analysis.critic_free_single_env_audit import (
    _build_audit_env_cfg,
    _prepare_audit_fixed_env_contract,
)
from ticl.model_builder import load_model
from ticl.models.tabpfn_regressor_vendor import TabPFNOfficialRegressorHarness


def _make_harness(*, latent_dim: int, num_buckets: int = 31, value_range: float = 5.0) -> TabPFNOfficialRegressorHarness:
    harness = TabPFNOfficialRegressorHarness(
        emsize=int(latent_dim),
        num_buckets=int(num_buckets),
        value_range=float(value_range),
        ignore_nan_targets=True,
    )
    harness.reset_target_statistics()
    return harness


def _parse_checkpoint_steps(spec: str, fit_steps: int) -> list[int]:
    raw = [int(v.strip()) for v in str(spec).split(",") if v.strip()]
    steps = sorted({int(v) for v in raw if 0 <= int(v) <= int(fit_steps)})
    if 0 not in steps:
        steps = [0] + steps
    if int(fit_steps) not in steps:
        steps.append(int(fit_steps))
    return steps


def _objective_dataset(dataset: dict, *, device: torch.device) -> dict:
    mask = dataset["objective_mask"]
    x = dataset["latent_vf"][mask].to(device=device, dtype=torch.float32)
    y = dataset["returns"][mask].to(device=device, dtype=torch.float32)
    mask_full = torch.ones_like(y, dtype=torch.bool)
    return {
        "x": x,
        "y": y,
        "mask": mask_full,
    }


@torch.no_grad()
def _track_metrics(
    *,
    model,
    bardist,
    x: torch.Tensor,
    y: torch.Tensor,
) -> dict:
    logits = model(x).to(dtype=torch.float32)
    probs = torch.softmax(logits, dim=-1)
    mean_pred = bardist.mean(logits)
    center_idx = int(bardist.num_bars) // 2
    band_lo = max(center_idx - 1, 0)
    band_hi = min(center_idx + 2, int(bardist.num_bars))
    target_bucket = bardist.map_to_bucket_idx(y).clamp(0, int(bardist.num_bars) - 1)
    target_bucket_prob = probs.gather(-1, target_bucket.unsqueeze(-1)).squeeze(-1)
    pred_bucket = logits.argmax(dim=-1)

    eval_stats = _evaluate_predictions(
        mean_pred.detach().cpu(),
        y.detach().cpu(),
        torch.ones_like(y, dtype=torch.bool).cpu(),
    )
    loss = bardist(logits, y).mean()
    target_center_fraction = (
        ((target_bucket >= band_lo) & (target_bucket < band_hi)).to(dtype=torch.float32).mean()
    )
    pred_center_fraction = (
        ((pred_bucket >= band_lo) & (pred_bucket < band_hi)).to(dtype=torch.float32).mean()
    )
    return {
        "official_default_loss": float(loss.detach().cpu().item()),
        "logit_abs_mean": float(logits.abs().mean().detach().cpu().item()),
        "logit_std": float(logits.std(unbiased=False).detach().cpu().item()),
        "logit_l2_mean": float(logits.norm(dim=-1).mean().detach().cpu().item()),
        "bucket_entropy": float(_bucket_entropy(logits)),
        "mean_pred_mean": float(mean_pred.mean().detach().cpu().item()),
        "mean_pred_std": float(mean_pred.std(unbiased=False).detach().cpu().item()),
        "central_bucket_prob_mean": float(probs[:, center_idx].mean().detach().cpu().item()),
        "central_band_mass_mean": float(probs[:, band_lo:band_hi].sum(dim=-1).mean().detach().cpu().item()),
        "central_bucket_target_fraction": float(target_center_fraction.detach().cpu().item()),
        "central_bucket_argmax_fraction": float(pred_center_fraction.detach().cpu().item()),
        "target_bucket_prob_mean": float(target_bucket_prob.mean().detach().cpu().item()),
        **eval_stats,
    }


def run_phase2_distribution_attractor_trajectory_probe(
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
    checkpoint_steps: str = "0,1,2,5,10,20,40,80",
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
        same_dataset = _extract_latent_dataset_with_bucket_idx(algo)
        progress_next = int(n_steps)
        _collect_rollout(algo, callback, vec_env, progress_timestep=progress_next, total_timesteps=2 * int(n_steps))
        next_dataset = _extract_latent_dataset_with_bucket_idx(algo)

        same_obj = _objective_dataset(same_dataset, device=device_obj)
        next_obj = _objective_dataset(next_dataset, device=device_obj)
        latent_dim = int(same_obj["x"].shape[-1])
        harness = _make_harness(latent_dim=latent_dim).to(device_obj)
        bardist = harness.znorm_space_bardist_.to(device_obj)
        model = copy.deepcopy(harness.output_projection).to(device_obj)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(head_fit_learning_rate))

        checkpoints = _parse_checkpoint_steps(checkpoint_steps, int(fit_steps))
        history = []
        for step_idx in range(int(fit_steps) + 1):
            if step_idx in checkpoints:
                history.append(
                    {
                        "fit_step": int(step_idx),
                        "same_rollout": _track_metrics(
                            model=model,
                            bardist=bardist,
                            x=same_obj["x"],
                            y=same_obj["y"],
                        ),
                        "next_rollout": _track_metrics(
                            model=model,
                            bardist=bardist,
                            x=next_obj["x"],
                            y=next_obj["y"],
                        ),
                    }
                )
            if step_idx == int(fit_steps):
                break
            logits = model(same_obj["x"]).to(dtype=torch.float32)
            loss = harness.compute_regression_loss(
                logits=logits.unsqueeze(0),
                targets_BQ=same_obj["y"].unsqueeze(0),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(head_fit_max_grad_norm))
            optimizer.step()

        return {
            "audit_entry": "phase2_distribution_attractor_trajectory_probe",
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
            "checkpoint_steps": checkpoints,
            "head_fit_learning_rate": float(head_fit_learning_rate),
            "head_fit_max_grad_norm": float(head_fit_max_grad_norm),
            "head_spec": {
                "arch": "official_mlp",
                "num_buckets": 31,
                "value_range": 5.0,
                "loss": "official_default",
            },
            "fixed_env_contract": dict(fixed_bundle["fixed_env_contract"]),
            "history": history,
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
    parser.add_argument("--checkpoint-steps", type=str, default="0,1,2,5,10,20,40,80")
    parser.add_argument("--head-fit-learning-rate", type=float, default=1e-3)
    parser.add_argument("--head-fit-max-grad-norm", type=float, default=0.5)
    parser.add_argument(
        "--output-json",
        type=str,
        default="/home/chen/RLPFN/artifacts/phase2_distribution_attractor_trajectory_probe.json",
    )
    args = parser.parse_args()
    report = run_phase2_distribution_attractor_trajectory_probe(
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
        checkpoint_steps=str(args.checkpoint_steps),
        head_fit_learning_rate=float(args.head_fit_learning_rate),
        head_fit_max_grad_norm=float(args.head_fit_max_grad_norm),
    )
    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
