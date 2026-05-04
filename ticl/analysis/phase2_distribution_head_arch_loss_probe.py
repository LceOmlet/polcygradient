import argparse
import copy
import json
from pathlib import Path

import torch
from torch import nn

from ticl.analysis.phase2_distribution_head_resolution_decode_probe import (
    _evaluate_decodes_and_logits,
    _fit_logits_linear_probe,
)
from ticl.analysis.phase2_distribution_head_root_cause_probe import (
    _extract_latent_dataset_with_bucket_idx,
    _generic_nll_from_bucket_idx,
)
from ticl.analysis.phase2_simple_scalar_value_head_probe import (
    _build_algo,
    _build_suite_algo,
    _collect_rollout,
    _default_device,
    _suite_from_seed_range,
)
from ticl.analysis.critic_free_single_env_audit import (
    _build_audit_env_cfg,
    _prepare_audit_fixed_env_contract,
    _snapshot_policy_state_dict,
)
from ticl.model_builder import load_model
from ticl.models.tabpfn_regressor_vendor import TabPFNOfficialRegressorHarness


def _make_vendor_bardist(*, latent_dim: int, num_buckets: int, value_range: float) -> TabPFNOfficialRegressorHarness:
    harness = TabPFNOfficialRegressorHarness(
        emsize=int(latent_dim),
        num_buckets=int(num_buckets),
        value_range=float(value_range),
        ignore_nan_targets=True,
    )
    harness.reset_target_statistics()
    return harness


def _make_head(*, latent_dim: int, num_buckets: int, arch: str) -> nn.Module:
    arch = str(arch)
    if arch == "official_mlp":
        return nn.Sequential(
            nn.Linear(int(latent_dim), 2 * int(latent_dim)),
            nn.GELU(),
            nn.Linear(2 * int(latent_dim), int(num_buckets)),
        )
    if arch == "linear":
        return nn.Linear(int(latent_dim), int(num_buckets))
    raise ValueError(f"Unsupported arch: {arch}")


def _fit_variant(
    *,
    head: nn.Module,
    harness: TabPFNOfficialRegressorHarness,
    x: torch.Tensor,
    y: torch.Tensor,
    bucket_idx: torch.Tensor,
    fit_steps: int,
    lr: float,
    max_grad_norm: float,
    loss_name: str,
) -> dict:
    optimizer = torch.optim.Adam(head.parameters(), lr=float(lr))
    initial_loss = None
    final_loss = None
    for _ in range(int(fit_steps)):
        logits = head(x)
        logits_f = logits.to(dtype=torch.float32)
        if loss_name == "official_default":
            loss = harness.compute_regression_loss(
                logits=logits_f.unsqueeze(0),
                targets_BQ=y.unsqueeze(0),
            )
        elif loss_name == "official_mse_only":
            loss = harness.compute_regression_loss(
                logits=logits_f.unsqueeze(0),
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
            loss = _generic_nll_from_bucket_idx(
                harness.znorm_space_bardist_,
                logits_f,
                bucket_idx,
            ).mean()
        else:
            raise ValueError(f"Unsupported loss_name: {loss_name}")
        if initial_loss is None:
            initial_loss = float(loss.detach().cpu().item())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), float(max_grad_norm))
        optimizer.step()
        final_loss = float(loss.detach().cpu().item())
    return {
        "model": head.cpu(),
        "loss_name": str(loss_name),
        "fit_summary": {
            "fit_steps": int(fit_steps),
            "learning_rate": float(lr),
            "max_grad_norm": float(max_grad_norm),
            "initial_loss": None if initial_loss is None else float(initial_loss),
            "final_loss": None if final_loss is None else float(final_loss),
        },
    }


@torch.no_grad()
def _evaluate_variant(
    *,
    dataset: dict,
    variant: dict,
    harness: TabPFNOfficialRegressorHarness,
    device: torch.device,
    scalar_from_logits: dict,
    arch: str,
    loss_name: str,
) -> dict:
    model = variant["model"].to(device)
    bardist = harness.znorm_space_bardist_.to(device)
    latent_vf = dataset["latent_vf"].to(device=device, dtype=torch.float32)
    logits = model(latent_vf)
    metrics = _evaluate_decodes_and_logits(
        dataset=dataset,
        logits=logits,
        bardist=bardist,
        scalar_from_logits=scalar_from_logits,
    )
    metrics["arch"] = str(arch)
    metrics["loss_name"] = str(loss_name)
    metrics["fit_summary"] = dict(variant["fit_summary"])
    return metrics


def run_phase2_distribution_head_arch_loss_probe(
    *,
    checkpoint_path: str,
    device: str | None = None,
    frozen_h_seed: int = 12345,
    train_env_seed: int = 2020,
    train_rollout_seed: int = 4040,
    heldout_env_seed_start: int = 3000,
    heldout_rollout_seed_start: int = 4000,
    heldout_env_count: int = 4,
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
    heldout_suite = _suite_from_seed_range(
        frozen_h=frozen_h,
        env_seed_start=int(heldout_env_seed_start),
        rollout_seed_start=int(heldout_rollout_seed_start),
        count=int(heldout_env_count),
    )

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
            same_dataset = _extract_latent_dataset_with_bucket_idx(algo)
            x = same_dataset["latent_vf"][same_dataset["objective_mask"]].to(device=device_obj, dtype=torch.float32)
            y = same_dataset["returns"][same_dataset["objective_mask"]].to(device=device_obj, dtype=torch.float32)
            latent_dim = int(x.shape[-1])
            num_buckets = 31
            value_range = 5.0
            harness = _make_vendor_bardist(
                latent_dim=latent_dim,
                num_buckets=num_buckets,
                value_range=value_range,
            ).to(device_obj)
            bucket_idx = harness.znorm_space_bardist_.map_to_bucket_idx(y).clamp(0, num_buckets - 1)

            variant_specs = [
                ("official_mlp", "official_default"),
                ("official_mlp", "official_mse_only"),
                ("official_mlp", "bucket_nll"),
                ("linear", "official_default"),
                ("linear", "official_mse_only"),
                ("linear", "bucket_nll"),
            ]

            fitted = {}
            logits_probes = {}
            for arch, loss_name in variant_specs:
                variant = _fit_variant(
                    head=_make_head(latent_dim=latent_dim, num_buckets=num_buckets, arch=arch).to(device_obj),
                    harness=copy.deepcopy(harness),
                    x=x,
                    y=y,
                    bucket_idx=bucket_idx,
                    fit_steps=int(fit_steps),
                    lr=float(head_fit_learning_rate),
                    max_grad_norm=float(head_fit_max_grad_norm),
                    loss_name=loss_name,
                )
                fitted[(arch, loss_name)] = variant
                logits_same = variant["model"].to(device_obj)(
                    same_dataset["latent_vf"].to(device=device_obj, dtype=torch.float32)
                )
                logits_probes[(arch, loss_name)] = _fit_logits_linear_probe(logits_same, same_dataset)

            same_results = {}
            for arch, loss_name in variant_specs:
                same_results[f"{arch}__{loss_name}"] = _evaluate_variant(
                    dataset=same_dataset,
                    variant=fitted[(arch, loss_name)],
                    harness=copy.deepcopy(harness),
                    device=device_obj,
                    scalar_from_logits=logits_probes[(arch, loss_name)],
                    arch=arch,
                    loss_name=loss_name,
                )

            progress_next = min(int((outer_idx + 1) * n_steps), total_timesteps)
            _collect_rollout(algo, callback, vec_env, progress_timestep=progress_next, total_timesteps=total_timesteps)
            next_dataset = _extract_latent_dataset_with_bucket_idx(algo)
            next_results = {}
            for arch, loss_name in variant_specs:
                next_results[f"{arch}__{loss_name}"] = _evaluate_variant(
                    dataset=next_dataset,
                    variant=fitted[(arch, loss_name)],
                    harness=copy.deepcopy(harness),
                    device=device_obj,
                    scalar_from_logits=logits_probes[(arch, loss_name)],
                    arch=arch,
                    loss_name=loss_name,
                )

            policy_state_dict = _snapshot_policy_state_dict(algo.policy)
            suite_algo, suite_callback, suite_vec_env = _build_suite_algo(
                checkpoint_path=checkpoint_path,
                device_obj=device_obj,
                env_cfg=env_cfg,
                num_features=int(num_features),
                suite=heldout_suite,
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
                suite_algo.policy.load_state_dict(policy_state_dict, strict=True)
                _collect_rollout(suite_algo, suite_callback, suite_vec_env, progress_timestep=0, total_timesteps=int(n_steps))
                heldout_dataset = _extract_latent_dataset_with_bucket_idx(suite_algo)
                heldout_results = {}
                for arch, loss_name in variant_specs:
                    heldout_results[f"{arch}__{loss_name}"] = _evaluate_variant(
                        dataset=heldout_dataset,
                        variant=fitted[(arch, loss_name)],
                        harness=copy.deepcopy(harness),
                        device=device_obj,
                        scalar_from_logits=logits_probes[(arch, loss_name)],
                        arch=arch,
                        loss_name=loss_name,
                    )
            finally:
                suite_vec_env.close()
                del suite_algo, suite_callback, suite_vec_env
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            history.append(
                {
                    "outer_epoch": int(outer_idx + 1),
                    "same_rollout": same_results,
                    "next_rollout": next_results,
                    "heldout": heldout_results,
                }
            )

        return {
            "audit_entry": "phase2_distribution_head_arch_loss_probe",
            "checkpoint_path": str(Path(checkpoint_path).expanduser().resolve()),
            "device": str(device_obj),
            "frozen_h_seed": int(frozen_h_seed),
            "train_env_seed": int(train_env_seed),
            "train_rollout_seed": int(train_rollout_seed),
            "heldout_env_seed_start": int(heldout_env_seed_start),
            "heldout_rollout_seed_start": int(heldout_rollout_seed_start),
            "heldout_env_count": int(heldout_env_count),
            "single_eval_pos": int(single_eval_pos),
            "n_steps": int(n_steps),
            "batch_size": int(batch_size),
            "n_epochs": int(n_epochs),
            "learning_rate": float(learning_rate),
            "target_kl": float(target_kl),
            "outer_epochs": int(outer_epochs),
            "build_seed": int(build_seed),
            "ppo_reset_env_state_at_sep": bool(ppo_reset_env_state_at_sep),
            "actor_baseline_mode": str(actor_baseline_mode),
            "fit_steps": int(fit_steps),
            "head_fit_learning_rate": float(head_fit_learning_rate),
            "head_fit_max_grad_norm": float(head_fit_max_grad_norm),
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
    parser.add_argument("--heldout-env-seed-start", type=int, default=3000)
    parser.add_argument("--heldout-rollout-seed-start", type=int, default=4000)
    parser.add_argument("--heldout-env-count", type=int, default=4)
    parser.add_argument("--single-eval-pos", type=int, default=64)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument("--outer-epochs", type=int, default=1)
    parser.add_argument("--build-seed", type=int, default=4040)
    parser.add_argument("--ppo-reset-env-state-at-sep", action="store_true")
    parser.add_argument("--actor-baseline-mode", type=str, default="learned")
    parser.add_argument("--fit-steps", type=int, default=80)
    parser.add_argument("--head-fit-learning-rate", type=float, default=1e-3)
    parser.add_argument("--head-fit-max-grad-norm", type=float, default=0.5)
    parser.add_argument(
        "--output-json",
        type=str,
        default="/home/chen/RLPFN/artifacts/phase2_distribution_head_arch_loss_probe.json",
    )
    args = parser.parse_args()
    report = run_phase2_distribution_head_arch_loss_probe(
        checkpoint_path=args.checkpoint_path,
        device=args.device,
        frozen_h_seed=int(args.frozen_h_seed),
        train_env_seed=int(args.train_env_seed),
        train_rollout_seed=int(args.train_rollout_seed),
        heldout_env_seed_start=int(args.heldout_env_seed_start),
        heldout_rollout_seed_start=int(args.heldout_rollout_seed_start),
        heldout_env_count=int(args.heldout_env_count),
        single_eval_pos=int(args.single_eval_pos),
        n_steps=int(args.n_steps),
        batch_size=int(args.batch_size),
        n_epochs=int(args.n_epochs),
        learning_rate=float(args.learning_rate),
        target_kl=float(args.target_kl),
        outer_epochs=int(args.outer_epochs),
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
