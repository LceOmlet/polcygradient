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
    _measure_rollout_critic_quality,
    _prepare_audit_fixed_env_contract,
    _recover_raw_from_value_space,
    _seed_all,
    _snapshot_policy_state_dict,
)
from ticl.analysis.critic_value_fit_probe import _clone_h_repeated
from ticl.analysis.phase2_walk_forward_critic_generalization_probe import _build_algo
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


def _build_suite_algo(
    *,
    checkpoint_path: str,
    device_obj: torch.device,
    env_cfg: dict,
    num_features: int,
    suite: dict,
    single_eval_pos: int,
    n_steps: int,
    learning_rate: float,
    batch_size: int,
    n_epochs: int,
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
        gamma=float(config["optimizer"].get("ppo_gamma", 1.0)),
        gae_lambda=float(config["optimizer"].get("ppo_gae_lambda", 0.95)),
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


def _collect_one_rollout(algo, callback, vec_env) -> None:
    algo._update_current_progress_remaining(0, int(algo.n_steps))
    ok = algo.collect_rollouts(vec_env, callback, algo.rollout_buffer, n_rollout_steps=algo.n_steps)
    if not ok:
        raise RuntimeError("collect_rollouts returned False during heldout raw metric decomposition probe.")


def _compute_mean_baseline_metrics(rollout_buffer) -> dict:
    objective_mask = np.asarray(rollout_buffer.objective_masks, dtype=np.float32).reshape(-1) > 1e-8
    normalized_returns = np.asarray(rollout_buffer.returns, dtype=np.float32).reshape(-1)
    normalized_values = np.asarray(rollout_buffer.values, dtype=np.float32).reshape(-1)
    means_flat = np.asarray(rollout_buffer.rollout_return_means, dtype=np.float32).reshape(-1)
    stds_flat = np.asarray(rollout_buffer.rollout_return_stds, dtype=np.float32).reshape(-1)
    raw_returns = rollout_buffer._get_flat_raw_returns_tensor(dtype=torch.float32).cpu().numpy().reshape(-1)
    raw_values = rollout_buffer._get_flat_raw_values_tensor(dtype=torch.float32).cpu().numpy().reshape(-1)

    mean_only_raw_values = means_flat
    zero_norm_raw_values = (
        _recover_raw_from_value_space(
            torch.zeros_like(torch.as_tensor(normalized_values, dtype=torch.float32)),
            value_means=torch.as_tensor(means_flat, dtype=torch.float32),
            value_stds=torch.as_tensor(stds_flat, dtype=torch.float32),
        )
        .cpu()
        .numpy()
        .reshape(-1)
    )
    if not np.allclose(mean_only_raw_values, zero_norm_raw_values):
        raise RuntimeError("mean-only raw reconstruction mismatch.")

    return {
        "normalized_actual": {
            "explained_variance": float(
                _explained_variance_with_mask(normalized_values, normalized_returns, mask=objective_mask)
            ),
            "raw_corr": float(_masked_corrcoef_np(normalized_values, normalized_returns, objective_mask)),
            "value_std": float(np.asarray(normalized_values[objective_mask], dtype=np.float32).std())
            if int(objective_mask.sum()) > 1
            else 0.0,
            "return_std": float(np.asarray(normalized_returns[objective_mask], dtype=np.float32).std())
            if int(objective_mask.sum()) > 1
            else 0.0,
        },
        "raw_actual": {
            "explained_variance": float(_explained_variance_with_mask(raw_values, raw_returns, mask=objective_mask)),
            "raw_corr": float(_masked_corrcoef_np(raw_values, raw_returns, objective_mask)),
            "value_std": float(np.asarray(raw_values[objective_mask], dtype=np.float32).std())
            if int(objective_mask.sum()) > 1
            else 0.0,
            "return_std": float(np.asarray(raw_returns[objective_mask], dtype=np.float32).std())
            if int(objective_mask.sum()) > 1
            else 0.0,
        },
        "raw_mean_only_baseline": {
            "explained_variance": float(
                _explained_variance_with_mask(mean_only_raw_values, raw_returns, mask=objective_mask)
            ),
            "raw_corr": float(_masked_corrcoef_np(mean_only_raw_values, raw_returns, objective_mask)),
            "value_std": float(np.asarray(mean_only_raw_values[objective_mask], dtype=np.float32).std())
            if int(objective_mask.sum()) > 1
            else 0.0,
            "return_std": float(np.asarray(raw_returns[objective_mask], dtype=np.float32).std())
            if int(objective_mask.sum()) > 1
            else 0.0,
        },
    }


def run_phase2_heldout_raw_metric_decomposition_probe(
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
    build_seed: int = 4040,
    ppo_reset_env_state_at_sep: bool = True,
    modes: list[str] | None = None,
) -> dict:
    modes = list(modes or ["learned", "zero"])
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

    report = {
        "audit_entry": "phase2_heldout_raw_metric_decomposition_probe",
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
        "build_seed": int(build_seed),
        "ppo_reset_env_state_at_sep": bool(ppo_reset_env_state_at_sep),
        "fixed_env_contract": dict(fixed_bundle["fixed_env_contract"]),
        "modes": {},
    }

    for mode in modes:
        train_algo, train_callback, train_vec_env, _, _ = _build_algo(
            checkpoint_path=checkpoint_path,
            device_obj=device_obj,
            frozen_h=frozen_h,
            train_env_seed=int(train_env_seed),
            train_rollout_seed=int(train_rollout_seed),
            single_eval_pos=int(single_eval_pos),
            n_steps=int(n_steps),
            batch_size=int(batch_size),
            n_epochs=int(n_epochs),
            learning_rate=float(learning_rate),
            target_kl=float(target_kl),
            ppo_reset_env_state_at_sep=bool(ppo_reset_env_state_at_sep),
            actor_baseline_mode=str(mode),
            build_seed=int(build_seed),
        )
        try:
            _collect_one_rollout(train_algo, train_callback, train_vec_env)
            train_algo.train()
            policy_state_dict = _snapshot_policy_state_dict(train_algo.policy)
        finally:
            train_vec_env.close()
            del train_algo, train_callback, train_vec_env
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        suite_algo, suite_callback, suite_vec_env = _build_suite_algo(
            checkpoint_path=checkpoint_path,
            device_obj=device_obj,
            env_cfg=env_cfg,
            num_features=int(num_features),
            suite=heldout_suite,
            single_eval_pos=int(single_eval_pos),
            n_steps=int(n_steps),
            learning_rate=float(learning_rate),
            batch_size=int(batch_size),
            n_epochs=int(n_epochs),
            target_kl=float(target_kl),
            ppo_reset_env_state_at_sep=bool(ppo_reset_env_state_at_sep),
            actor_baseline_mode=str(mode),
            build_seed=int(build_seed),
        )
        try:
            suite_algo.policy.load_state_dict(policy_state_dict, strict=True)
            _collect_one_rollout(suite_algo, suite_callback, suite_vec_env)
            report["modes"][str(mode)] = {
                "quality": _measure_rollout_critic_quality(suite_algo),
                "decomposition": _compute_mean_baseline_metrics(suite_algo.rollout_buffer),
            }
        finally:
            suite_vec_env.close()
            del suite_algo, suite_callback, suite_vec_env
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return report


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
    parser.add_argument("--build-seed", type=int, default=4040)
    parser.add_argument("--ppo-reset-env-state-at-sep", action="store_true")
    parser.add_argument("--modes", nargs="*", default=["learned", "zero"])
    parser.add_argument(
        "--output-json",
        type=str,
        default="/home/chen/RLPFN/artifacts/phase2_heldout_raw_metric_decomposition_probe.json",
    )
    args = parser.parse_args()

    report = run_phase2_heldout_raw_metric_decomposition_probe(
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
        build_seed=int(args.build_seed),
        ppo_reset_env_state_at_sep=bool(args.ppo_reset_env_state_at_sep),
        modes=list(args.modes),
    )
    out_path = Path(args.output_json).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, sort_keys=True)
    out_path.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
