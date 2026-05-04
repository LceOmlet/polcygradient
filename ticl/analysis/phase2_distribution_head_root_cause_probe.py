import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ticl.analysis.phase2_simple_scalar_value_head_probe import (
    _apply_scalar_head,
    _build_algo,
    _build_suite_algo,
    _collect_rollout,
    _default_device,
    _evaluate_predictions,
    _fit_ridge_scalar_head,
    _suite_from_seed_range,
)
from ticl.analysis.critic_free_single_env_audit import (
    _build_audit_env_cfg,
    _measure_rollout_critic_quality,
    _prepare_audit_fixed_env_contract,
    _snapshot_policy_state_dict,
)
from ticl.model_builder import load_model
from ticl.models.tabpfn_regressor_vendor import TabPFNOfficialRegressorHarness


@torch.no_grad()
def _extract_latent_dataset_with_bucket_idx(algo) -> dict:
    rollout_buffer = algo.rollout_buffer
    latents = []
    returns = []
    masks = []
    bardist_preds = []
    bucket_idx = []
    for rollout_data in rollout_buffer.get_gpu_flat(algo.batch_size):
        outputs = algo.policy.evaluate_actions_with_hidden_flat(
            rollout_data.observations,
            rollout_data.actions,
            seq_lengths=rollout_data.seq_lengths,
            action_masks=rollout_data.action_masks,
        )
        hidden = outputs["hidden"]
        latent_vf = algo.policy.mlp_extractor.forward_critic(hidden)
        latents.append(latent_vf.detach().cpu())
        returns.append(rollout_data.returns.detach().cpu().reshape(-1))
        masks.append((rollout_data.objective_masks > 1e-8).detach().cpu().reshape(-1))
        bardist_preds.append(outputs["values"].detach().cpu().reshape(-1))
        target_bucket = getattr(rollout_data, "value_target_bucket_idx", None)
        if target_bucket is None:
            target_bucket = torch.full_like(rollout_data.returns.detach().cpu().reshape(-1), -1, dtype=torch.long)
        else:
            target_bucket = target_bucket.detach().cpu().reshape(-1).to(dtype=torch.long)
        bucket_idx.append(target_bucket)
    return {
        "latent_vf": torch.cat(latents, dim=0),
        "returns": torch.cat(returns, dim=0),
        "objective_mask": torch.cat(masks, dim=0).to(dtype=torch.bool),
        "bardist_preds": torch.cat(bardist_preds, dim=0),
        "value_target_bucket_idx": torch.cat(bucket_idx, dim=0),
    }


def _bucket_entropy(logits: torch.Tensor) -> float:
    probs = torch.softmax(logits.to(dtype=torch.float32), dim=-1)
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1).mean()
    return float(entropy.detach().cpu().item())


def _generic_nll_from_bucket_idx(bardist, logits: torch.Tensor, bucket_idx: torch.Tensor) -> torch.Tensor:
    if hasattr(bardist, "nll_from_bucket_idx"):
        return bardist.nll_from_bucket_idx(logits, bucket_idx)
    target = bucket_idx.clone().view(*logits.shape[:-1]).to(device=logits.device, dtype=torch.long)
    scaled_bucket_log_probs = bardist.compute_scaled_log_probs(logits)
    return -scaled_bucket_log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)


def _clone_legacy_components(policy):
    latent_dim = int(policy.mlp_extractor.latent_dim_vf)
    bardist = copy.deepcopy(policy.get_value_bardist()).float()
    head = nn.Linear(latent_dim, int(bardist.num_bars))
    current_head = getattr(policy, "value_net", None)
    if not isinstance(current_head, nn.Linear):
        raise TypeError("Expected PPO legacy value head to be nn.Linear.")
    head.load_state_dict(copy.deepcopy(current_head.state_dict()))
    return head, bardist


def _make_vendor_harness(policy, *, value_range: float) -> TabPFNOfficialRegressorHarness:
    latent_dim = int(policy.mlp_extractor.latent_dim_vf)
    bardist = policy.get_value_bardist()
    return TabPFNOfficialRegressorHarness(
        emsize=latent_dim,
        num_buckets=int(bardist.num_bars),
        value_range=float(value_range),
        ignore_nan_targets=True,
    )


def _fit_distribution_variant(
    *,
    head_module: nn.Module,
    bardist,
    x: torch.Tensor,
    y: torch.Tensor,
    bucket_idx: torch.Tensor,
    fit_steps: int,
    lr: float,
    max_grad_norm: float,
    loss_name: str,
    official_harness: TabPFNOfficialRegressorHarness | None = None,
) -> dict:
    optimizer = torch.optim.Adam(head_module.parameters(), lr=float(lr))
    initial_loss = None
    final_loss = None
    for _ in range(int(fit_steps)):
        logits = head_module(x)
        if loss_name == "legacy_continuous_nll":
            loss = bardist(logits.to(dtype=torch.float32), y).mean()
        elif loss_name == "legacy_bucket_nll":
            loss = _generic_nll_from_bucket_idx(bardist, logits.to(dtype=torch.float32), bucket_idx).mean()
        elif loss_name == "legacy_mean_mse":
            preds = bardist.mean(logits.to(dtype=torch.float32))
            loss = (preds - y).square().mean()
        elif loss_name == "vendor_official":
            if official_harness is None:
                raise ValueError("vendor_official requires official_harness")
            loss = official_harness.compute_regression_loss(
                logits=logits.unsqueeze(0),
                targets_BQ=y.unsqueeze(0),
            )
        elif loss_name == "vendor_bucket_nll":
            loss = _generic_nll_from_bucket_idx(bardist, logits.to(dtype=torch.float32), bucket_idx).mean()
        elif loss_name == "vendor_mean_mse":
            preds = bardist.mean(logits.to(dtype=torch.float32))
            loss = (preds - y).square().mean()
        else:
            raise ValueError(f"Unsupported loss_name: {loss_name}")
        if initial_loss is None:
            initial_loss = float(loss.detach().cpu().item())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head_module.parameters(), float(max_grad_norm))
        optimizer.step()
        final_loss = float(loss.detach().cpu().item())
    return {
        "model": head_module.cpu(),
        "bardist": bardist.cpu(),
        "loss_name": str(loss_name),
        "fit_summary": {
            "fit_steps": int(fit_steps),
            "learning_rate": float(lr),
            "max_grad_norm": float(max_grad_norm),
            "initial_loss": None if initial_loss is None else float(initial_loss),
            "final_loss": None if final_loss is None else float(final_loss),
        },
    }


def _fit_variants(
    *,
    policy,
    dataset: dict,
    fit_steps: int,
    lr: float,
    max_grad_norm: float,
    device: torch.device,
) -> tuple[dict[str, dict], dict]:
    x = dataset["latent_vf"][dataset["objective_mask"]].to(device=device, dtype=torch.float32)
    y = dataset["returns"][dataset["objective_mask"]].to(device=device, dtype=torch.float32)
    bucket_idx = dataset["value_target_bucket_idx"][dataset["objective_mask"]].to(device=device, dtype=torch.long)
    scalar_head = _fit_ridge_scalar_head(
        dataset["latent_vf"],
        dataset["returns"],
        dataset["objective_mask"],
        ridge_lambda=1e-3,
    )

    legacy_head, legacy_bardist = _clone_legacy_components(policy)
    legacy_head = legacy_head.to(device)
    legacy_bardist = legacy_bardist.to(device)

    vendor_harness_default = _make_vendor_harness(policy, value_range=5.0)
    vendor_harness_default.reset_target_statistics()
    vendor_harness_default = vendor_harness_default.to(device)
    vendor_bardist_default = vendor_harness_default.znorm_space_bardist_

    vendor_harness_narrow = _make_vendor_harness(policy, value_range=1.5)
    vendor_harness_narrow.reset_target_statistics()
    vendor_harness_narrow = vendor_harness_narrow.to(device)
    vendor_bardist_narrow = vendor_harness_narrow.znorm_space_bardist_

    variants = {
        "legacy_checkpoint": {
            "kind": "checkpoint",
        },
        "legacy_continuous_nll": _fit_distribution_variant(
            head_module=copy.deepcopy(legacy_head),
            bardist=copy.deepcopy(legacy_bardist),
            x=x,
            y=y,
            bucket_idx=bucket_idx,
            fit_steps=int(fit_steps),
            lr=float(lr),
            max_grad_norm=float(max_grad_norm),
            loss_name="legacy_continuous_nll",
        ),
        "legacy_bucket_nll": _fit_distribution_variant(
            head_module=copy.deepcopy(legacy_head),
            bardist=copy.deepcopy(legacy_bardist),
            x=x,
            y=y,
            bucket_idx=bucket_idx,
            fit_steps=int(fit_steps),
            lr=float(lr),
            max_grad_norm=float(max_grad_norm),
            loss_name="legacy_bucket_nll",
        ),
        "legacy_mean_mse": _fit_distribution_variant(
            head_module=copy.deepcopy(legacy_head),
            bardist=copy.deepcopy(legacy_bardist),
            x=x,
            y=y,
            bucket_idx=bucket_idx,
            fit_steps=int(fit_steps),
            lr=float(lr),
            max_grad_norm=float(max_grad_norm),
            loss_name="legacy_mean_mse",
        ),
        "vendor_official_range5": _fit_distribution_variant(
            head_module=copy.deepcopy(vendor_harness_default.output_projection),
            bardist=copy.deepcopy(vendor_bardist_default),
            x=x,
            y=y,
            bucket_idx=bucket_idx,
            fit_steps=int(fit_steps),
            lr=float(lr),
            max_grad_norm=float(max_grad_norm),
            loss_name="vendor_official",
            official_harness=copy.deepcopy(vendor_harness_default),
        ),
        "vendor_bucket_nll_range5": _fit_distribution_variant(
            head_module=copy.deepcopy(vendor_harness_default.output_projection),
            bardist=copy.deepcopy(vendor_bardist_default),
            x=x,
            y=y,
            bucket_idx=bucket_idx,
            fit_steps=int(fit_steps),
            lr=float(lr),
            max_grad_norm=float(max_grad_norm),
            loss_name="vendor_bucket_nll",
        ),
        "vendor_mean_mse_range5": _fit_distribution_variant(
            head_module=copy.deepcopy(vendor_harness_default.output_projection),
            bardist=copy.deepcopy(vendor_bardist_default),
            x=x,
            y=y,
            bucket_idx=bucket_idx,
            fit_steps=int(fit_steps),
            lr=float(lr),
            max_grad_norm=float(max_grad_norm),
            loss_name="vendor_mean_mse",
        ),
        "vendor_mean_mse_range1_5": _fit_distribution_variant(
            head_module=copy.deepcopy(vendor_harness_narrow.output_projection),
            bardist=copy.deepcopy(vendor_bardist_narrow),
            x=x,
            y=y,
            bucket_idx=bucket_idx,
            fit_steps=int(fit_steps),
            lr=float(lr),
            max_grad_norm=float(max_grad_norm),
            loss_name="vendor_mean_mse",
        ),
    }
    return variants, scalar_head


@torch.no_grad()
def _evaluate_variant_on_dataset(dataset: dict, variant_name: str, variant: dict, *, device: torch.device) -> dict:
    if variant_name == "legacy_checkpoint":
        preds = dataset["bardist_preds"]
        metrics = _evaluate_predictions(preds, dataset["returns"], dataset["objective_mask"])
        metrics["head_type"] = "legacy_checkpoint"
        metrics["loss_type"] = "checkpoint_loaded"
        return metrics

    model = variant["model"].to(device)
    bardist = variant["bardist"].to(device)
    model.eval()
    latent_vf = dataset["latent_vf"].to(device=device, dtype=torch.float32)
    logits = model(latent_vf)
    preds = bardist.mean(logits.to(dtype=torch.float32)).detach().cpu().reshape(-1)
    metrics = _evaluate_predictions(preds, dataset["returns"], dataset["objective_mask"])
    metrics["loss_type"] = variant["loss_name"]
    metrics["fit_summary"] = dict(variant["fit_summary"])
    metrics["bucket_entropy"] = _bucket_entropy(logits)
    metrics["logits_shape"] = [int(v) for v in logits.shape]
    metrics["value_range"] = float(max(abs(float(bardist.borders[0].item())), abs(float(bardist.borders[-1].item()))))
    return metrics


def _evaluate_scalar_variant(dataset: dict, scalar_head: dict) -> dict:
    preds = _apply_scalar_head(dataset["latent_vf"], scalar_head)
    metrics = _evaluate_predictions(preds, dataset["returns"], dataset["objective_mask"])
    metrics["loss_type"] = "scalar_mse_ridge"
    return metrics


def run_phase2_distribution_head_root_cause_probe(
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
    fit_steps: int = 100,
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
            variants, scalar_head = _fit_variants(
                policy=algo.policy,
                dataset=same_dataset,
                fit_steps=int(fit_steps),
                lr=float(head_fit_learning_rate),
                max_grad_norm=float(head_fit_max_grad_norm),
                device=device_obj,
            )
            same_results = {name: _evaluate_variant_on_dataset(same_dataset, name, variant, device=device_obj) for name, variant in variants.items()}
            same_results["simple_scalar"] = _evaluate_scalar_variant(same_dataset, scalar_head)

            progress_next = min(int((outer_idx + 1) * n_steps), total_timesteps)
            _collect_rollout(algo, callback, vec_env, progress_timestep=progress_next, total_timesteps=total_timesteps)
            next_dataset = _extract_latent_dataset_with_bucket_idx(algo)
            next_results = {name: _evaluate_variant_on_dataset(next_dataset, name, variant, device=device_obj) for name, variant in variants.items()}
            next_results["simple_scalar"] = _evaluate_scalar_variant(next_dataset, scalar_head)

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
                heldout_results = {name: _evaluate_variant_on_dataset(heldout_dataset, name, variant, device=device_obj) for name, variant in variants.items()}
                heldout_results["simple_scalar"] = _evaluate_scalar_variant(heldout_dataset, scalar_head)
                heldout_bardist_quality = _measure_rollout_critic_quality(suite_algo)
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
                    "heldout_bardist_quality": heldout_bardist_quality,
                    "same_rollout_objective_total": int(same_dataset["objective_mask"].sum().item()),
                }
            )
        return {
            "audit_entry": "phase2_distribution_head_root_cause_probe",
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
    parser.add_argument("--fit-steps", type=int, default=100)
    parser.add_argument("--head-fit-learning-rate", type=float, default=1e-3)
    parser.add_argument("--head-fit-max-grad-norm", type=float, default=0.5)
    parser.add_argument(
        "--output-json",
        type=str,
        default="/home/chen/RLPFN/artifacts/phase2_distribution_head_root_cause_probe.json",
    )
    args = parser.parse_args()

    report = run_phase2_distribution_head_root_cause_probe(
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
    out_path = Path(args.output_json).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, sort_keys=True)
    out_path.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
