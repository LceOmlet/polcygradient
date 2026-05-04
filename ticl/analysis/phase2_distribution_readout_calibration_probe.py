import argparse
import json
from pathlib import Path

import torch

from ticl.analysis.phase2_distribution_head_resolution_decode_probe import (
    _fit_vendor_variant,
)
from ticl.analysis.phase2_distribution_head_root_cause_probe import (
    _extract_latent_dataset_with_bucket_idx,
)
from ticl.analysis.phase2_simple_scalar_value_head_probe import (
    _build_algo,
    _build_suite_algo,
    _collect_rollout,
    _default_device,
    _evaluate_predictions,
    _suite_from_seed_range,
)
from ticl.analysis.critic_free_single_env_audit import (
    _build_audit_env_cfg,
    _prepare_audit_fixed_env_contract,
    _snapshot_policy_state_dict,
)
from ticl.model_builder import load_model
from ticl.models.tabpfn_regressor_vendor import TabPFNOfficialRegressorHarness


def _make_harness(*, latent_dim: int) -> TabPFNOfficialRegressorHarness:
    harness = TabPFNOfficialRegressorHarness(
        emsize=int(latent_dim),
        num_buckets=31,
        value_range=5.0,
        ignore_nan_targets=True,
    )
    harness.reset_target_statistics()
    return harness


def _readout_predictions(
    *,
    bardist,
    logits: torch.Tensor,
    readout: str,
    temperature: float = 1.0,
    quantile: float = 0.5,
) -> torch.Tensor:
    logits = logits.to(dtype=torch.float32) / float(temperature)
    if readout == "mean":
        return bardist.mean(logits)
    if readout == "median":
        return bardist.median(logits)
    if readout == "mode":
        return bardist.mode(logits)
    if readout == "quantile":
        return bardist.icdf(logits, float(quantile))
    raise ValueError(f"Unsupported readout: {readout}")


def _score_dataset(
    *,
    preds: torch.Tensor,
    dataset: dict,
) -> dict:
    return _evaluate_predictions(
        preds.detach().cpu().reshape(-1),
        dataset["returns"],
        dataset["objective_mask"],
    )


@torch.no_grad()
def _collect_logits(
    *,
    variant: dict,
    dataset: dict,
    device: torch.device,
) -> torch.Tensor:
    model = variant["model"].to(device)
    latent_vf = dataset["latent_vf"].to(device=device, dtype=torch.float32)
    return model(latent_vf)


def _search_best_same_rollout_readouts(
    *,
    bardist,
    same_logits: torch.Tensor,
    same_dataset: dict,
) -> dict:
    candidates = {}

    base_mean = _score_dataset(
        preds=_readout_predictions(bardist=bardist, logits=same_logits, readout="mean"),
        dataset=same_dataset,
    )
    base_median = _score_dataset(
        preds=_readout_predictions(bardist=bardist, logits=same_logits, readout="median"),
        dataset=same_dataset,
    )
    base_mode = _score_dataset(
        preds=_readout_predictions(bardist=bardist, logits=same_logits, readout="mode"),
        dataset=same_dataset,
    )
    candidates["mean"] = {
        "readout": "mean",
        "temperature": 1.0,
        "quantile": None,
        "metrics": base_mean,
    }
    candidates["median"] = {
        "readout": "median",
        "temperature": 1.0,
        "quantile": 0.5,
        "metrics": base_median,
    }
    candidates["mode"] = {
        "readout": "mode",
        "temperature": 1.0,
        "quantile": None,
        "metrics": base_mode,
    }

    temp_grid = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0]
    best_temp_mean = None
    for t in temp_grid:
        metrics = _score_dataset(
            preds=_readout_predictions(
                bardist=bardist,
                logits=same_logits,
                readout="mean",
                temperature=float(t),
            ),
            dataset=same_dataset,
        )
        item = {
            "readout": "mean",
            "temperature": float(t),
            "quantile": None,
            "metrics": metrics,
        }
        if best_temp_mean is None or item["metrics"]["explained_variance_normalized"] > best_temp_mean["metrics"]["explained_variance_normalized"]:
            best_temp_mean = item
    candidates["best_tempered_mean"] = best_temp_mean

    q_grid = [0.05 * i for i in range(1, 20)]
    best_quantile = None
    for q in q_grid:
        metrics = _score_dataset(
            preds=_readout_predictions(
                bardist=bardist,
                logits=same_logits,
                readout="quantile",
                quantile=float(q),
            ),
            dataset=same_dataset,
        )
        item = {
            "readout": "quantile",
            "temperature": 1.0,
            "quantile": float(q),
            "metrics": metrics,
        }
        if best_quantile is None or item["metrics"]["explained_variance_normalized"] > best_quantile["metrics"]["explained_variance_normalized"]:
            best_quantile = item
    candidates["best_quantile"] = best_quantile

    best_tempered_quantile = None
    for t in temp_grid:
        for q in q_grid:
            metrics = _score_dataset(
                preds=_readout_predictions(
                    bardist=bardist,
                    logits=same_logits,
                    readout="quantile",
                    temperature=float(t),
                    quantile=float(q),
                ),
                dataset=same_dataset,
            )
            item = {
                "readout": "quantile",
                "temperature": float(t),
                "quantile": float(q),
                "metrics": metrics,
            }
            if best_tempered_quantile is None or item["metrics"]["explained_variance_normalized"] > best_tempered_quantile["metrics"]["explained_variance_normalized"]:
                best_tempered_quantile = item
    candidates["best_tempered_quantile"] = best_tempered_quantile
    return candidates


def _evaluate_candidate_on_dataset(
    *,
    candidate: dict,
    bardist,
    logits: torch.Tensor,
    dataset: dict,
) -> dict:
    preds = _readout_predictions(
        bardist=bardist,
        logits=logits,
        readout=str(candidate["readout"]),
        temperature=float(candidate["temperature"]),
        quantile=0.5 if candidate["quantile"] is None else float(candidate["quantile"]),
    )
    metrics = _score_dataset(preds=preds, dataset=dataset)
    return {
        "readout": str(candidate["readout"]),
        "temperature": float(candidate["temperature"]),
        "quantile": None if candidate["quantile"] is None else float(candidate["quantile"]),
        "metrics": metrics,
    }


def run_phase2_distribution_readout_calibration_probe(
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
            harness = _make_harness(latent_dim=int(x.shape[-1])).to(device_obj)
            variant = _fit_vendor_variant(
                latent_dim=int(x.shape[-1]),
                num_buckets=31,
                value_range=5.0,
                x=x,
                y=y,
                fit_steps=int(fit_steps),
                lr=float(head_fit_learning_rate),
                max_grad_norm=float(head_fit_max_grad_norm),
                device=device_obj,
            )
            bardist = harness.znorm_space_bardist_.to(device_obj)
            same_logits = _collect_logits(variant=variant, dataset=same_dataset, device=device_obj)
            chosen = _search_best_same_rollout_readouts(
                bardist=bardist,
                same_logits=same_logits,
                same_dataset=same_dataset,
            )
            same_metrics = {
                name: _evaluate_candidate_on_dataset(
                    candidate=candidate,
                    bardist=bardist,
                    logits=same_logits,
                    dataset=same_dataset,
                )
                for name, candidate in chosen.items()
            }

            progress_next = min(int((outer_idx + 1) * n_steps), total_timesteps)
            _collect_rollout(algo, callback, vec_env, progress_timestep=progress_next, total_timesteps=total_timesteps)
            next_dataset = _extract_latent_dataset_with_bucket_idx(algo)
            next_logits = _collect_logits(variant=variant, dataset=next_dataset, device=device_obj)
            next_metrics = {
                name: _evaluate_candidate_on_dataset(
                    candidate=candidate,
                    bardist=bardist,
                    logits=next_logits,
                    dataset=next_dataset,
                )
                for name, candidate in chosen.items()
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
                heldout_dataset = _extract_latent_dataset_with_bucket_idx(suite_algo)
                heldout_logits = _collect_logits(variant=variant, dataset=heldout_dataset, device=device_obj)
                heldout_metrics = {
                    name: _evaluate_candidate_on_dataset(
                        candidate=candidate,
                        bardist=bardist,
                        logits=heldout_logits,
                        dataset=heldout_dataset,
                    )
                    for name, candidate in chosen.items()
                }
            finally:
                suite_vec_env.close()
                del suite_algo, suite_callback, suite_vec_env
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            history.append(
                {
                    "outer_epoch": int(outer_idx + 1),
                    "chosen_on_same_rollout": chosen,
                    "same_rollout": same_metrics,
                    "next_rollout": next_metrics,
                    "heldout": heldout_metrics,
                }
            )

        return {
            "audit_entry": "phase2_distribution_readout_calibration_probe",
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
        default="/home/chen/RLPFN/artifacts/phase2_distribution_readout_calibration_probe.json",
    )
    args = parser.parse_args()
    report = run_phase2_distribution_readout_calibration_probe(
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
