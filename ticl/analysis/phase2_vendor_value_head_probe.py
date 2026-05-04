import argparse
import copy
import json
from pathlib import Path

import torch
from torch import nn

from ticl.analysis.critic_free_single_env_audit import (
    _measure_rollout_critic_quality,
    _snapshot_policy_state_dict,
)
from ticl.analysis.phase2_simple_scalar_value_head_probe import (
    _apply_scalar_head,
    _build_algo,
    _build_suite_algo,
    _collect_rollout,
    _default_device,
    _evaluate_predictions,
    _extract_latent_dataset,
    _fit_ridge_scalar_head,
    _suite_from_seed_range,
)
from ticl.model_builder import load_model
from ticl.models.tabpfn_regressor_vendor import TabPFNOfficialRegressorHarness


def _clone_legacy_value_head(policy) -> tuple[nn.Linear, torch.nn.Module]:
    latent_dim = int(policy.mlp_extractor.latent_dim_vf)
    bardist = copy.deepcopy(policy.get_value_bardist()).float()
    head = nn.Linear(latent_dim, int(bardist.num_bars))
    current_head = getattr(policy, "value_net", None)
    if isinstance(current_head, nn.Linear):
        head.load_state_dict(copy.deepcopy(current_head.state_dict()))
    else:
        raise TypeError(
            "phase2_vendor_value_head_probe expects current legacy PPO value head to be nn.Linear."
        )
    return head, bardist


def _make_vendor_value_head(policy) -> TabPFNOfficialRegressorHarness:
    latent_dim = int(policy.mlp_extractor.latent_dim_vf)
    bardist = policy.get_value_bardist()
    borders = bardist.borders.detach().to(dtype=torch.float32)
    value_range = float(max(abs(float(borders[0].item())), abs(float(borders[-1].item()))))
    return TabPFNOfficialRegressorHarness(
        emsize=latent_dim,
        num_buckets=int(bardist.num_bars),
        value_range=float(value_range),
        ignore_nan_targets=True,
    )


def _fit_legacy_head(
    *,
    policy,
    dataset: dict,
    fit_steps: int,
    lr: float,
    max_grad_norm: float,
    device: torch.device,
) -> dict:
    head, bardist = _clone_legacy_value_head(policy)
    head = head.to(device)
    bardist = bardist.to(device)
    x = dataset["latent_vf"][dataset["objective_mask"]].to(device=device, dtype=torch.float32)
    y = dataset["returns"][dataset["objective_mask"]].to(device=device, dtype=torch.float32)
    optimizer = torch.optim.Adam(head.parameters(), lr=float(lr))
    initial_loss = None
    final_loss = None
    head.train()
    for step_idx in range(int(fit_steps)):
        logits = head(x)
        loss = bardist(logits.to(dtype=torch.float32), y).mean()
        if initial_loss is None:
            initial_loss = float(loss.detach().cpu().item())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), float(max_grad_norm))
        optimizer.step()
        final_loss = float(loss.detach().cpu().item())
    return {
        "head": head.cpu(),
        "bardist": bardist.cpu(),
        "loss_type": "legacy_bar_nll_mean",
        "fit_summary": {
            "fit_steps": int(fit_steps),
            "learning_rate": float(lr),
            "max_grad_norm": float(max_grad_norm),
            "initial_loss": None if initial_loss is None else float(initial_loss),
            "final_loss": None if final_loss is None else float(final_loss),
        },
    }


def _fit_vendor_head(
    *,
    policy,
    dataset: dict,
    fit_steps: int,
    lr: float,
    max_grad_norm: float,
    device: torch.device,
) -> dict:
    harness = _make_vendor_value_head(policy)
    harness.reset_target_statistics()
    harness = harness.to(device)
    x = dataset["latent_vf"][dataset["objective_mask"]].to(device=device, dtype=torch.float32)
    y = dataset["returns"][dataset["objective_mask"]].to(device=device, dtype=torch.float32)
    optimizer = torch.optim.Adam(harness.parameters(), lr=float(lr))
    initial_loss = None
    final_loss = None
    harness.train()
    for step_idx in range(int(fit_steps)):
        logits = harness.forward_logits(x).unsqueeze(0)
        loss = harness.compute_regression_loss(
            logits=logits,
            targets_BQ=y.unsqueeze(0),
        )
        if initial_loss is None:
            initial_loss = float(loss.detach().cpu().item())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(harness.parameters(), float(max_grad_norm))
        optimizer.step()
        final_loss = float(loss.detach().cpu().item())
    return {
        "head": harness.cpu(),
        "loss_type": "vendor_tabpfn_regression_loss",
        "fit_summary": {
            "fit_steps": int(fit_steps),
            "learning_rate": float(lr),
            "max_grad_norm": float(max_grad_norm),
            "initial_loss": None if initial_loss is None else float(initial_loss),
            "final_loss": None if final_loss is None else float(final_loss),
        },
    }


@torch.no_grad()
def _evaluate_legacy_head_on_dataset(dataset: dict, fitted: dict, *, device: torch.device) -> dict:
    head = fitted["head"].to(device)
    bardist = fitted["bardist"].to(device)
    head.eval()
    latent_vf = dataset["latent_vf"].to(device=device, dtype=torch.float32)
    logits = head(latent_vf)
    preds = bardist.mean(logits.to(dtype=torch.float32)).detach().cpu().reshape(-1)
    metrics = _evaluate_predictions(preds, dataset["returns"], dataset["objective_mask"])
    metrics["head_type"] = "legacy_refit"
    metrics["loss_type"] = fitted["loss_type"]
    metrics["logits_shape"] = [int(v) for v in logits.shape]
    return metrics


@torch.no_grad()
def _evaluate_vendor_head_on_dataset(dataset: dict, fitted: dict, *, device: torch.device) -> dict:
    harness = fitted["head"].to(device)
    harness.eval()
    latent_vf = dataset["latent_vf"].to(device=device, dtype=torch.float32)
    logits = harness.forward_logits(latent_vf)
    preds = harness.predict_mean(logits=logits, use_raw_space=False).detach().cpu().reshape(-1)
    full = harness.predict_full(
        logits=logits[: min(int(logits.shape[0]), 4)],
        quantiles=[0.1, 0.5, 0.9],
        use_raw_space=False,
    )
    metrics = _evaluate_predictions(preds, dataset["returns"], dataset["objective_mask"])
    metrics["head_type"] = "vendor_tabpfn"
    metrics["loss_type"] = fitted["loss_type"]
    metrics["logits_shape"] = [int(v) for v in logits.shape]
    metrics["decode_contract"] = {
        "keys": sorted(full.keys()),
        "criterion_num_bars": int(full["criterion"].num_bars),
        "mean_shape": [int(v) for v in full["mean"].shape],
        "median_shape": [int(v) for v in full["median"].shape],
        "mode_shape": [int(v) for v in full["mode"].shape],
        "quantile_shapes": [[int(v) for v in q.shape] for q in full["quantiles"]],
    }
    return metrics


def _evaluate_simple_scalar_head_on_dataset(dataset: dict, head: dict) -> dict:
    scalar_preds = _apply_scalar_head(dataset["latent_vf"], head)
    metrics = _evaluate_predictions(scalar_preds, dataset["returns"], dataset["objective_mask"])
    metrics["head_type"] = "simple_scalar_ridge"
    metrics["loss_type"] = "ridge_closed_form"
    return metrics


def _make_checkpoint_legacy_metrics(dataset: dict) -> dict:
    metrics = _evaluate_predictions(dataset["bardist_preds"], dataset["returns"], dataset["objective_mask"])
    metrics["head_type"] = "legacy_checkpoint"
    metrics["loss_type"] = "legacy_bar_nll_mean"
    return metrics


def _insufficient_objective_payload(*, head_type: str, loss_type: str, objective_total: int) -> dict:
    return {
        "head_type": str(head_type),
        "loss_type": str(loss_type),
        "status": "insufficient_objective_tokens",
        "objective_total": int(objective_total),
    }


def run_phase2_vendor_value_head_probe(
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
    outer_epochs: int = 2,
    build_seed: int = 4040,
    ppo_reset_env_state_at_sep: bool = True,
    actor_baseline_mode: str = "learned",
    legacy_fit_steps: int = 200,
    vendor_fit_steps: int = 200,
    head_fit_learning_rate: float = 1e-3,
    head_fit_max_grad_norm: float = 0.5,
    ridge_lambda: float = 1e-3,
) -> dict:
    device_obj = torch.device(str(device or _default_device()))
    load_model.cache_clear()
    _, config = load_model(checkpoint_path, device=device_obj, verbose=False)
    env_cfg = {
        **config["prior"]["environment"],
    }
    from ticl.analysis.critic_free_single_env_audit import (
        _build_audit_env_cfg,
        _prepare_audit_fixed_env_contract,
    )

    env_cfg = _build_audit_env_cfg(env_cfg)
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
            same_dataset = _extract_latent_dataset(algo)
            objective_total = int(same_dataset["objective_mask"].sum().item())
            checkpoint_same = _make_checkpoint_legacy_metrics(same_dataset)
            legacy_fit = _fit_legacy_head(
                policy=algo.policy,
                dataset=same_dataset,
                fit_steps=int(legacy_fit_steps),
                lr=float(head_fit_learning_rate),
                max_grad_norm=float(head_fit_max_grad_norm),
                device=device_obj,
            )
            vendor_fit = _fit_vendor_head(
                policy=algo.policy,
                dataset=same_dataset,
                fit_steps=int(vendor_fit_steps),
                lr=float(head_fit_learning_rate),
                max_grad_norm=float(head_fit_max_grad_norm),
                device=device_obj,
            )
            simple_head = None
            if int(objective_total) > 1:
                simple_head = _fit_ridge_scalar_head(
                    same_dataset["latent_vf"],
                    same_dataset["returns"],
                    same_dataset["objective_mask"],
                    ridge_lambda=float(ridge_lambda),
                )

            same_results = {
                "legacy_checkpoint": checkpoint_same,
                "legacy_refit": _evaluate_legacy_head_on_dataset(same_dataset, legacy_fit, device=device_obj),
                "vendor_refit": _evaluate_vendor_head_on_dataset(same_dataset, vendor_fit, device=device_obj),
                "simple_scalar": (
                    _evaluate_simple_scalar_head_on_dataset(same_dataset, simple_head)
                    if simple_head is not None
                    else _insufficient_objective_payload(
                        head_type="simple_scalar_ridge",
                        loss_type="ridge_closed_form",
                        objective_total=int(objective_total),
                    )
                ),
            }

            progress_next = min(int((outer_idx + 1) * n_steps), total_timesteps)
            _collect_rollout(
                algo,
                callback,
                vec_env,
                progress_timestep=progress_next,
                total_timesteps=total_timesteps,
            )
            next_dataset = _extract_latent_dataset(algo)
            next_results = {
                "legacy_checkpoint": _make_checkpoint_legacy_metrics(next_dataset),
                "legacy_refit": _evaluate_legacy_head_on_dataset(next_dataset, legacy_fit, device=device_obj),
                "vendor_refit": _evaluate_vendor_head_on_dataset(next_dataset, vendor_fit, device=device_obj),
                "simple_scalar": (
                    _evaluate_simple_scalar_head_on_dataset(next_dataset, simple_head)
                    if simple_head is not None
                    else _insufficient_objective_payload(
                        head_type="simple_scalar_ridge",
                        loss_type="ridge_closed_form",
                        objective_total=int(objective_total),
                    )
                ),
            }

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
                heldout_dataset = _extract_latent_dataset(suite_algo)
                heldout_quality = _measure_rollout_critic_quality(suite_algo)
                heldout_results = {
                    "legacy_checkpoint": _make_checkpoint_legacy_metrics(heldout_dataset),
                    "legacy_refit": _evaluate_legacy_head_on_dataset(heldout_dataset, legacy_fit, device=device_obj),
                    "vendor_refit": _evaluate_vendor_head_on_dataset(heldout_dataset, vendor_fit, device=device_obj),
                    "simple_scalar": (
                        _evaluate_simple_scalar_head_on_dataset(heldout_dataset, simple_head)
                        if simple_head is not None
                        else _insufficient_objective_payload(
                            head_type="simple_scalar_ridge",
                            loss_type="ridge_closed_form",
                            objective_total=int(objective_total),
                        )
                    ),
                }
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
                    "same_rollout_objective_total": int(same_dataset["objective_mask"].sum().item()),
                    "next_rollout_objective_total": int(next_dataset["objective_mask"].sum().item()),
                    "heldout_bardist_quality": heldout_quality,
                    "fit_summary": {
                        "legacy_refit": legacy_fit["fit_summary"],
                        "vendor_refit": vendor_fit["fit_summary"],
                        "simple_scalar": {
                            "loss_type": "ridge_closed_form",
                            "ridge_lambda": float(ridge_lambda),
                        },
                    },
                }
            )
        return {
            "audit_entry": "phase2_vendor_value_head_probe",
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
            "legacy_fit_steps": int(legacy_fit_steps),
            "vendor_fit_steps": int(vendor_fit_steps),
            "head_fit_learning_rate": float(head_fit_learning_rate),
            "head_fit_max_grad_norm": float(head_fit_max_grad_norm),
            "ridge_lambda": float(ridge_lambda),
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
    parser.add_argument("--outer-epochs", type=int, default=2)
    parser.add_argument("--build-seed", type=int, default=4040)
    parser.add_argument("--ppo-reset-env-state-at-sep", action="store_true")
    parser.add_argument("--actor-baseline-mode", type=str, default="learned")
    parser.add_argument("--legacy-fit-steps", type=int, default=200)
    parser.add_argument("--vendor-fit-steps", type=int, default=200)
    parser.add_argument("--head-fit-learning-rate", type=float, default=1e-3)
    parser.add_argument("--head-fit-max-grad-norm", type=float, default=0.5)
    parser.add_argument("--ridge-lambda", type=float, default=1e-3)
    parser.add_argument(
        "--output-json",
        type=str,
        default="/home/chen/RLPFN/artifacts/phase2_vendor_value_head_probe.json",
    )
    args = parser.parse_args()

    report = run_phase2_vendor_value_head_probe(
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
        legacy_fit_steps=int(args.legacy_fit_steps),
        vendor_fit_steps=int(args.vendor_fit_steps),
        head_fit_learning_rate=float(args.head_fit_learning_rate),
        head_fit_max_grad_norm=float(args.head_fit_max_grad_norm),
        ridge_lambda=float(args.ridge_lambda),
    )
    out_path = Path(args.output_json).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, sort_keys=True)
    out_path.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
