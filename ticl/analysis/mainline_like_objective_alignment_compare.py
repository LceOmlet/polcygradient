import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from ticl.analysis.critic_free_single_env_audit import (
    _apply_boundary_contract_mode_flags,
    _build_audit_env_cfg,
    _build_audit_ppo_policy_step_fn,
    _make_deterministic_batch_plan,
    _make_suite,
    _run_eval,
)
from ticl.analysis.prior_generalization_audit import _temporary_seed
from ticl.model_builder import load_model
from ticl.priors.environment_prior import EnvironmentPrior
from ticl.sb3_recurrent_ppo import build_recurrent_ppo
from ticl.train import _freeze_env_h_list_for_replay


def _default_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _rankdata_simple(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if int(x.size) < 2 or int(y.size) < 2:
        return float("nan")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        return float("nan")
    x_std = float(x.std())
    y_std = float(y.std())
    if x_std <= 0.0 or y_std <= 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if int(x.size) < 2 or int(y.size) < 2:
        return float("nan")
    return _pearson(_rankdata_simple(x), _rankdata_simple(y))


def run_mainline_like_objective_alignment_compare(
    *,
    checkpoint_path: str,
    device: str | None = None,
    actor_objective_mode: str = "tokenwise",
    frozen_h_seeds: list[int] | None = None,
    train_env_seed_start: int = 2020,
    eval_rollout_seed_start: int = 5000,
    train_env_count: int = 8,
    single_eval_pos: int = 64,
    n_steps: int = 256,
    batch_size: int = 256,
    outer_epochs: int = 1,
    learning_rate: float = 2e-4,
    n_epochs: int = 1,
    target_kl: float = 0.03,
    boundary_contract_mode: str = "normal",
) -> dict:
    device_obj = torch.device(str(device or _default_device()))
    actor_objective_mode = str(actor_objective_mode).strip().lower()
    frozen_h_seeds = list(frozen_h_seeds or [12345, 23456, 34567, 45678])
    train_env_seeds = [int(train_env_seed_start) + i for i in range(int(train_env_count))]
    eval_rollout_seeds = [int(eval_rollout_seed_start) + i for i in range(int(train_env_count))]

    trajectory_rows = []

    for frozen_h_seed in frozen_h_seeds:
        load_model.cache_clear()
        model, config = load_model(checkpoint_path, device=device_obj, verbose=False)
        num_features = int(config["prior"]["num_features"])
        env_cfg = _build_audit_env_cfg(config["prior"]["environment"])

        prior_for_h = EnvironmentPrior(copy.deepcopy(env_cfg))
        _apply_boundary_contract_mode_flags(prior_for_h, boundary_contract_mode)
        with _temporary_seed(int(frozen_h_seed)):
            frozen_h = _freeze_env_h_list_for_replay(
                prior_for_h,
                list(prior_for_h._sample_batch_hypers(1)),
            )[0]

        prior = EnvironmentPrior(copy.deepcopy(env_cfg))
        _apply_boundary_contract_mode_flags(prior, boundary_contract_mode)
        prior._sample_batch_hypers = lambda batch_n, _h=frozen_h: [copy.deepcopy(_h) for _ in range(int(batch_n))]
        prior._sample_single_eval_pos = lambda n_samples_arg, rng_seed=None: int(single_eval_pos)

        algo, callback, vec_env = build_recurrent_ppo(
            model=model,
            env_prior=prior,
            device=str(device_obj),
            num_features=num_features,
            n_envs=len(train_env_seeds),
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
            actor_baseline_mode="learned",
            actor_objective_mode=str(actor_objective_mode),
            ent_coef=float(config["optimizer"].get("ppo_ent_coef", 0.0)),
            vf_coef=float(config["optimizer"].get("ppo_vf_coef", 0.5)),
            max_grad_norm=float(config["optimizer"].get("ppo_max_grad_norm", 0.5)),
            target_kl=float(target_kl),
            verbose=0,
        )
        vec_env._sample_seed_list = lambda _seeds=train_env_seeds: list(_seeds)
        _make_deterministic_batch_plan(algo.rollout_buffer)

        suite = _make_suite(
            frozen_h=frozen_h,
            env_seeds=train_env_seeds,
            rollout_seeds=eval_rollout_seeds,
        )
        pre_eval = _run_eval(
            prior=prior,
            suite=suite,
            policy_step_fn=_build_audit_ppo_policy_step_fn(
                algo.policy,
                sampled=True,
                boundary_contract_mode=boundary_contract_mode,
            ),
            n_samples=int(n_steps),
            num_features=num_features,
            single_eval_pos=int(single_eval_pos),
            device=device_obj,
        )

        captured = {}
        orig_train = algo.train

        def wrapped_train():
            rb = algo.rollout_buffer
            captured["actor_advantages"] = np.array(rb.actor_advantages, copy=True)
            captured["objective_masks"] = np.array(rb.objective_masks, copy=True)
            captured["returns"] = np.array(rb.returns, copy=True)
            captured["values"] = np.array(rb.values, copy=True)
            return orig_train()

        algo.train = wrapped_train
        algo.learn(total_timesteps=int(n_steps) * int(outer_epochs), callback=callback)

        post_eval = _run_eval(
            prior=prior,
            suite=suite,
            policy_step_fn=_build_audit_ppo_policy_step_fn(
                algo.policy,
                sampled=True,
                boundary_contract_mode=boundary_contract_mode,
            ),
            n_samples=int(n_steps),
            num_features=num_features,
            single_eval_pos=int(single_eval_pos),
            device=device_obj,
        )

        pre_suffix = list(pre_eval["suffix_return_per_env"])
        post_suffix = list(post_eval["suffix_return_per_env"])
        adv = captured["actor_advantages"]
        mask = captured["objective_masks"] > 1e-8

        for env_idx, env_seed in enumerate(train_env_seeds):
            env_mask = mask[:, env_idx]
            env_adv = adv[:, env_idx][env_mask]
            delta_suffix = float(post_suffix[env_idx] - pre_suffix[env_idx])
            trajectory_rows.append(
                {
                    "frozen_h_seed": int(frozen_h_seed),
                    "env_seed": int(env_seed),
                    "objective_count": int(env_adv.size),
                    "tokenwise_adv_mean": float(env_adv.mean()) if int(env_adv.size) > 0 else float("nan"),
                    "tokenwise_adv_pos_mass": (
                        float(np.clip(env_adv, 0.0, None).sum()) if int(env_adv.size) > 0 else 0.0
                    ),
                    "tokenwise_adv_pos_frac": float((env_adv > 0).mean()) if int(env_adv.size) > 0 else 0.0,
                    "trajectory_raw_suffix_return": float(pre_suffix[env_idx]),
                    "trajectory_post_suffix_return": float(post_suffix[env_idx]),
                    "trajectory_delta_suffix_return": float(delta_suffix),
                }
            )

    rows = trajectory_rows
    tokenwise_mean = np.asarray([r["tokenwise_adv_mean"] for r in rows], dtype=np.float64)
    tokenwise_pos_mass = np.asarray([r["tokenwise_adv_pos_mass"] for r in rows], dtype=np.float64)
    tokenwise_pos_frac = np.asarray([r["tokenwise_adv_pos_frac"] for r in rows], dtype=np.float64)
    traj_raw = np.asarray([r["trajectory_raw_suffix_return"] for r in rows], dtype=np.float64)
    traj_post = np.asarray([r["trajectory_post_suffix_return"] for r in rows], dtype=np.float64)
    traj_delta = np.asarray([r["trajectory_delta_suffix_return"] for r in rows], dtype=np.float64)

    valid_mask = (
        np.isfinite(tokenwise_mean)
        & np.isfinite(tokenwise_pos_mass)
        & np.isfinite(tokenwise_pos_frac)
        & np.isfinite(traj_raw)
        & np.isfinite(traj_post)
        & np.isfinite(traj_delta)
    )
    rows_valid = [r for r, keep in zip(rows, valid_mask.tolist()) if keep]
    tokenwise_mean = tokenwise_mean[valid_mask]
    tokenwise_pos_mass = tokenwise_pos_mass[valid_mask]
    tokenwise_pos_frac = tokenwise_pos_frac[valid_mask]
    traj_raw = traj_raw[valid_mask]
    traj_post = traj_post[valid_mask]
    traj_delta = traj_delta[valid_mask]

    def _quartile_stats(scores: np.ndarray, outcomes: np.ndarray) -> dict:
        if int(scores.size) <= 0:
            return {
                "high_count": 0,
                "high_nonpositive_delta_count": 0,
                "low_positive_delta_count": 0,
            }
        q75 = float(np.quantile(scores, 0.75))
        q25 = float(np.quantile(scores, 0.25))
        high = scores >= q75
        low = scores <= q25
        return {
            "high_count": int(high.sum()),
            "high_nonpositive_delta_count": int((outcomes[high] <= 0.0).sum()),
            "low_positive_delta_count": int((outcomes[low] > 0.0).sum()),
        }

    tokenwise_summary = {
        "corr_mean_vs_delta": _pearson(tokenwise_mean, traj_delta),
        "corr_mean_vs_post": _pearson(tokenwise_mean, traj_post),
        "spearman_mean_vs_delta": _spearman(tokenwise_mean, traj_delta),
        "spearman_mean_vs_post": _spearman(tokenwise_mean, traj_post),
        "corr_pos_mass_vs_delta": _pearson(tokenwise_pos_mass, traj_delta),
        "corr_pos_frac_vs_delta": _pearson(tokenwise_pos_frac, traj_delta),
    }
    tokenwise_summary.update(_quartile_stats(tokenwise_mean, traj_delta))

    trajectory_summary = {
        "corr_raw_suffix_vs_delta": _pearson(traj_raw, traj_delta),
        "corr_raw_suffix_vs_post": _pearson(traj_raw, traj_post),
        "spearman_raw_suffix_vs_delta": _spearman(traj_raw, traj_delta),
        "spearman_raw_suffix_vs_post": _spearman(traj_raw, traj_post),
    }
    trajectory_summary.update(_quartile_stats(traj_raw, traj_delta))

    token_q75 = float(np.quantile(tokenwise_mean, 0.75)) if int(tokenwise_mean.size) > 0 else float("nan")
    traj_q75 = float(np.quantile(traj_raw, 0.75)) if int(traj_raw.size) > 0 else float("nan")
    token_high_nonpositive = [
        r
        for r in rows_valid
        if float(r["tokenwise_adv_mean"]) >= token_q75 and float(r["trajectory_delta_suffix_return"]) <= 0.0
    ]
    traj_high_nonpositive = [
        r
        for r in rows_valid
        if float(r["trajectory_raw_suffix_return"]) >= traj_q75 and float(r["trajectory_delta_suffix_return"]) <= 0.0
    ]

    return {
        "audit_entry": "mainline_like_objective_alignment_compare",
        "setup": {
            "checkpoint_path": str(Path(checkpoint_path).expanduser().resolve()),
            "device": str(device_obj),
            "frozen_h_seeds": [int(v) for v in frozen_h_seeds],
            "train_env_seeds": [int(v) for v in train_env_seeds],
            "single_eval_pos": int(single_eval_pos),
            "n_steps": int(n_steps),
            "batch_size": int(batch_size),
            "outer_epochs": int(outer_epochs),
            "learning_rate": float(learning_rate),
            "n_epochs": int(n_epochs),
            "target_kl": float(target_kl),
            "actor_gae_space": "normalized",
            "actor_baseline_mode": "learned",
            "actor_objective_mode": str(actor_objective_mode),
            "boundary_contract_mode": str(boundary_contract_mode),
            "normalize_advantage": False,
        },
        "tokenwise_summary": tokenwise_summary,
        "trajectory_level_summary": trajectory_summary,
        "tokenwise_high_nonpositive_examples": token_high_nonpositive[:8],
        "trajectory_high_nonpositive_examples": traj_high_nonpositive[:8],
        "rows": rows_valid,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_path")
    parser.add_argument("--device", default=None)
    parser.add_argument("--actor-objective-mode", default="tokenwise")
    parser.add_argument("--boundary-contract-mode", default="normal")
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    report = run_mainline_like_objective_alignment_compare(
        checkpoint_path=args.checkpoint_path,
        device=args.device,
        actor_objective_mode=args.actor_objective_mode,
        boundary_contract_mode=args.boundary_contract_mode,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output_json:
        out_path = Path(args.output_json).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
