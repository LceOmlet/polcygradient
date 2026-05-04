import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from ticl.analysis.critic_free_single_env_audit import (
    _apply_boundary_contract_mode_flags,
    _apply_sep_state_reset_flag,
    _build_audit_env_cfg,
    _make_deterministic_batch_plan,
    _measure_current_policy_rollout_critic_quality,
    _measure_rollout_critic_quality,
    _prepare_audit_fixed_env_contract,
    _seed_all,
    _snapshot_policy_state_dict,
)
from ticl.analysis.critic_value_fit_probe import _clone_h_repeated
from ticl.model_builder import load_model
from ticl.priors.environment_prior import EnvironmentPrior
from ticl.sb3_recurrent_ppo import _explained_variance_with_mask, build_recurrent_ppo
from stable_baselines3.common.logger import configure as configure_logger


def _default_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _masked_corrcoef_np(x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if int(mask.sum()) <= 1:
        return 0.0
    x_m = np.asarray(x, dtype=np.float32).reshape(-1)[mask]
    y_m = np.asarray(y, dtype=np.float32).reshape(-1)[mask]
    x_m = x_m - float(x_m.mean())
    y_m = y_m - float(y_m.mean())
    denom = float(np.linalg.norm(x_m) * np.linalg.norm(y_m))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(x_m, y_m) / denom)


def _suite_from_seed_range(*, frozen_h, env_seed_start: int, rollout_seed_start: int, count: int) -> dict:
    env_seeds = list(range(int(env_seed_start), int(env_seed_start) + int(count)))
    rollout_seeds = list(range(int(rollout_seed_start), int(rollout_seed_start) + int(count)))
    return {
        "h_list": _clone_h_repeated(frozen_h, len(env_seeds)),
        "env_seeds": [int(v) for v in env_seeds],
        "rollout_seeds": [int(v) for v in rollout_seeds],
        "batch_size": len(env_seeds),
        "suite_seed": None,
    }


def _build_algo(
    *,
    checkpoint_path: str,
    device_obj: torch.device,
    frozen_h,
    env_cfg: dict,
    num_features: int,
    train_env_seed: int,
    train_rollout_seed: int,
    single_eval_pos: int,
    n_steps: int,
    batch_size: int,
    n_epochs: int,
    learning_rate: float,
    target_kl: float,
    ppo_reset_env_state_at_sep: bool,
    actor_baseline_mode: str,
    build_seed: int,
):
    _seed_all(int(build_seed))
    load_model.cache_clear()
    model, config = load_model(checkpoint_path, device=device_obj, verbose=False)
    prior = EnvironmentPrior(copy.deepcopy(env_cfg))
    prior._sample_batch_hypers = lambda batch_n, _h=frozen_h: _clone_h_repeated(_h, int(batch_n))
    prior._sample_single_eval_pos = lambda n_samples_arg, rng_seed=None: int(single_eval_pos)
    _apply_boundary_contract_mode_flags(prior, "normal")
    _apply_sep_state_reset_flag(prior, bool(ppo_reset_env_state_at_sep))

    algo, callback, vec_env = build_recurrent_ppo(
        model=model,
        env_prior=prior,
        device=str(device_obj),
        num_features=int(num_features),
        n_envs=1,
        n_steps=int(n_steps),
        learning_rate=float(learning_rate),
        batch_size=int(batch_size),
        n_epochs=int(n_epochs),
        gamma=float(config["optimizer"].get("ppo_gamma", 0.98)),
        gae_lambda=float(config["optimizer"].get("ppo_gae_lambda", 0.90)),
        clip_range=config["optimizer"].get("ppo_clip_range", 0.2),
        clip_range_vf=None,
        normalize_advantage=False,
        actor_gae_space="normalized",
        actor_baseline_mode=str(actor_baseline_mode),
        actor_objective_mode="tokenwise",
        reset_env_state_at_sep=bool(ppo_reset_env_state_at_sep),
        separate_value_backbone=False,
        ent_coef=float(config["optimizer"].get("ppo_ent_coef", 0.0)),
        vf_coef=float(config["optimizer"].get("ppo_vf_coef", 0.5)),
        max_grad_norm=float(config["optimizer"].get("ppo_max_grad_norm", 0.5)),
        target_kl=float(target_kl),
        strict_fixed_env_mode=True,
        env_rng_seeds=[int(train_env_seed)],
        rollout_rng_seeds=[int(train_rollout_seed)],
        deterministic_actor_sampling=True,
        deterministic_batch_plan=True,
        strict_native_rollout=False,
        restore_validation_policy_state=False,
        verbose=0,
    )
    vec_env._sample_seed_list = lambda: [int(train_env_seed)]
    _make_deterministic_batch_plan(algo.rollout_buffer)
    algo.ep_info_buffer = []
    algo.ep_success_buffer = []
    algo.set_logger(configure_logger(folder=None, format_strings=[]))
    algo._last_obs = vec_env.reset()
    algo._last_episode_starts = np.ones((vec_env.num_envs,), dtype=bool)
    algo._last_lstm_states = algo.policy._dummy_states(vec_env.num_envs)
    callback.init_callback(algo)
    return algo, callback, vec_env


def _build_suite_algo(
    *,
    checkpoint_path: str,
    device_obj: torch.device,
    env_cfg: dict,
    num_features: int,
    suite: dict,
    single_eval_pos: int,
    n_steps: int,
    batch_size: int,
    n_epochs: int,
    learning_rate: float,
    target_kl: float,
    ppo_reset_env_state_at_sep: bool,
    actor_baseline_mode: str,
    build_seed: int,
):
    _seed_all(int(build_seed))
    load_model.cache_clear()
    model, config = load_model(checkpoint_path, device=device_obj, verbose=False)
    prior = EnvironmentPrior(copy.deepcopy(env_cfg))
    suite_h_list = [copy.deepcopy(h) for h in list(suite["h_list"])]
    prior._sample_batch_hypers = lambda batch_n, _h_list=suite_h_list: [
        copy.deepcopy(h) for h in _h_list[: int(batch_n)]
    ]
    prior._sample_single_eval_pos = lambda n_samples_arg, rng_seed=None: int(single_eval_pos)
    _apply_boundary_contract_mode_flags(prior, "normal")
    _apply_sep_state_reset_flag(prior, bool(ppo_reset_env_state_at_sep))

    algo, callback, vec_env = build_recurrent_ppo(
        model=model,
        env_prior=prior,
        device=str(device_obj),
        num_features=int(num_features),
        n_envs=int(suite["batch_size"]),
        n_steps=int(n_steps),
        learning_rate=float(learning_rate),
        batch_size=int(batch_size),
        n_epochs=int(n_epochs),
        gamma=float(config["optimizer"].get("ppo_gamma", 0.98)),
        gae_lambda=float(config["optimizer"].get("ppo_gae_lambda", 0.90)),
        clip_range=config["optimizer"].get("ppo_clip_range", 0.2),
        clip_range_vf=None,
        normalize_advantage=False,
        actor_gae_space="normalized",
        actor_baseline_mode=str(actor_baseline_mode),
        actor_objective_mode="tokenwise",
        reset_env_state_at_sep=bool(ppo_reset_env_state_at_sep),
        separate_value_backbone=False,
        ent_coef=float(config["optimizer"].get("ppo_ent_coef", 0.0)),
        vf_coef=float(config["optimizer"].get("ppo_vf_coef", 0.5)),
        max_grad_norm=float(config["optimizer"].get("ppo_max_grad_norm", 0.5)),
        target_kl=float(target_kl),
        strict_fixed_env_mode=True,
        env_rng_seeds=[int(v) for v in list(suite["env_seeds"])],
        rollout_rng_seeds=[int(v) for v in list(suite["rollout_seeds"])],
        deterministic_actor_sampling=True,
        deterministic_batch_plan=True,
        strict_native_rollout=False,
        restore_validation_policy_state=False,
        verbose=0,
    )
    vec_env._sample_seed_list = lambda _seeds=list(suite["env_seeds"]): [int(v) for v in list(_seeds)]
    _make_deterministic_batch_plan(algo.rollout_buffer)
    algo.ep_info_buffer = []
    algo.ep_success_buffer = []
    algo.set_logger(configure_logger(folder=None, format_strings=[]))
    algo._last_obs = vec_env.reset()
    algo._last_episode_starts = np.ones((vec_env.num_envs,), dtype=bool)
    algo._last_lstm_states = algo.policy._dummy_states(vec_env.num_envs)
    callback.init_callback(algo)
    return algo, callback, vec_env


def _collect_rollout(algo, callback, vec_env, *, progress_timestep: int = 0, total_timesteps: int | None = None) -> None:
    if total_timesteps is None:
        total_timesteps = int(algo.n_steps)
    algo._update_current_progress_remaining(int(progress_timestep), int(total_timesteps))
    ok = algo.collect_rollouts(vec_env, callback, algo.rollout_buffer, n_rollout_steps=algo.n_steps)
    if not ok:
        raise RuntimeError("collect_rollouts returned False during simple scalar value head probe.")


@torch.no_grad()
def _extract_latent_dataset(algo) -> dict:
    rollout_buffer = algo.rollout_buffer
    _make_deterministic_batch_plan(rollout_buffer)
    latents = []
    targets = []
    masks = []
    bardist_preds = []
    means = []
    stds = []
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
        targets.append(rollout_data.returns.detach().cpu())
        masks.append((rollout_data.objective_masks > 1e-8).detach().cpu())
        bardist_preds.append(outputs["values"].detach().cpu().reshape(-1))
        means.append(rollout_data.rollout_return_means.detach().cpu().reshape(-1))
        stds.append(rollout_data.rollout_return_stds.detach().cpu().reshape(-1))
    _make_deterministic_batch_plan(rollout_buffer)
    return {
        "latent_vf": torch.cat(latents, dim=0),
        "returns": torch.cat(targets, dim=0).reshape(-1),
        "objective_mask": torch.cat(masks, dim=0).reshape(-1).to(dtype=torch.bool),
        "bardist_preds": torch.cat(bardist_preds, dim=0).reshape(-1),
        "rollout_return_means": torch.cat(means, dim=0).reshape(-1),
        "rollout_return_stds": torch.cat(stds, dim=0).reshape(-1),
    }


def _fit_ridge_scalar_head(
    latent_vf: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    *,
    ridge_lambda: float = 1e-3,
) -> dict:
    mask = mask.reshape(-1).to(dtype=torch.bool)
    x = latent_vf[mask].to(dtype=torch.float64)
    y = targets[mask].reshape(-1, 1).to(dtype=torch.float64)
    if int(x.shape[0]) <= 1:
        raise RuntimeError("Not enough objective tokens for scalar head fit.")
    ones = torch.ones((int(x.shape[0]), 1), dtype=x.dtype, device=x.device)
    x_aug = torch.cat([x, ones], dim=1)
    gram = x_aug.T @ x_aug
    ridge = float(ridge_lambda) * torch.eye(int(gram.shape[0]), dtype=gram.dtype, device=gram.device)
    ridge[-1, -1] = 0.0  # do not regularize bias
    rhs = x_aug.T @ y
    weights = torch.linalg.solve(gram + ridge, rhs).reshape(-1)
    return {
        "weight": weights[:-1].to(dtype=torch.float32),
        "bias": weights[-1].to(dtype=torch.float32),
    }


def _apply_scalar_head(latent_vf: torch.Tensor, head: dict) -> torch.Tensor:
    return latent_vf.to(dtype=torch.float32) @ head["weight"] + head["bias"]


def _evaluate_predictions(
    preds: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> dict:
    preds_np = preds.detach().cpu().numpy().reshape(-1)
    targets_np = targets.detach().cpu().numpy().reshape(-1)
    mask_np = mask.detach().cpu().numpy().reshape(-1).astype(bool)
    if int(mask_np.sum()) > 1:
        value_std = float(np.asarray(preds_np[mask_np], dtype=np.float32).std())
        return_std = float(np.asarray(targets_np[mask_np], dtype=np.float32).std())
    else:
        value_std = 0.0
        return_std = 0.0
    return {
        "explained_variance_normalized": float(
            _explained_variance_with_mask(preds_np, targets_np, mask=mask_np)
        ),
        "raw_corr_normalized_space": float(_masked_corrcoef_np(preds_np, targets_np, mask_np)),
        "normalized_value_std": value_std,
        "normalized_return_std": return_std,
        "objective_total": int(mask_np.sum()),
    }


def _compare_head_design_on_dataset(dataset: dict, head: dict) -> dict:
    scalar_preds = _apply_scalar_head(dataset["latent_vf"], head)
    scalar_metrics = _evaluate_predictions(scalar_preds, dataset["returns"], dataset["objective_mask"])
    bardist_metrics = _evaluate_predictions(dataset["bardist_preds"], dataset["returns"], dataset["objective_mask"])
    return {
        "bardist_head": bardist_metrics,
        "simple_scalar_head": scalar_metrics,
        "improvement_ev_norm": float(
            scalar_metrics["explained_variance_normalized"] - bardist_metrics["explained_variance_normalized"]
        ),
        "improvement_corr_norm": float(
            scalar_metrics["raw_corr_normalized_space"] - bardist_metrics["raw_corr_normalized_space"]
        ),
    }


def run_phase2_simple_scalar_value_head_probe(
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
    ridge_lambda: float = 1e-3,
) -> dict:
    device_obj = torch.device(str(device or _default_device()))
    _seed_all(int(build_seed))
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
            same_dataset = _extract_latent_dataset(algo)
            simple_head = _fit_ridge_scalar_head(
                same_dataset["latent_vf"],
                same_dataset["returns"],
                same_dataset["objective_mask"],
                ridge_lambda=float(ridge_lambda),
            )
            same_compare = _compare_head_design_on_dataset(same_dataset, simple_head)

            progress_next = min(int((outer_idx + 1) * n_steps), total_timesteps)
            _collect_rollout(
                algo,
                callback,
                vec_env,
                progress_timestep=progress_next,
                total_timesteps=total_timesteps,
            )
            next_dataset = _extract_latent_dataset(algo)
            next_compare = _compare_head_design_on_dataset(next_dataset, simple_head)

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
                _collect_rollout(suite_algo, suite_callback, suite_vec_env)
                heldout_dataset = _extract_latent_dataset(suite_algo)
                heldout_compare = _compare_head_design_on_dataset(heldout_dataset, simple_head)
                heldout_quality = _measure_rollout_critic_quality(suite_algo)
            finally:
                suite_vec_env.close()
                del suite_algo, suite_callback, suite_vec_env
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            history.append(
                {
                    "outer_epoch": int(outer_idx + 1),
                    "same_rollout_compare": same_compare,
                    "next_rollout_compare": next_compare,
                    "heldout_compare": heldout_compare,
                    "heldout_bardist_quality": heldout_quality,
                }
            )
        return {
            "audit_entry": "phase2_simple_scalar_value_head_probe",
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
            "fixed_env_contract": dict(fixed_bundle["fixed_env_contract"]),
            "actor_baseline_mode": str(actor_baseline_mode),
            "ridge_lambda": float(ridge_lambda),
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
    parser.add_argument("--ridge-lambda", type=float, default=1e-3)
    parser.add_argument(
        "--output-json",
        type=str,
        default="/home/chen/RLPFN/artifacts/phase2_simple_scalar_value_head_probe.json",
    )
    args = parser.parse_args()

    report = run_phase2_simple_scalar_value_head_probe(
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
        ridge_lambda=float(args.ridge_lambda),
    )
    out_path = Path(args.output_json).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, sort_keys=True)
    out_path.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
