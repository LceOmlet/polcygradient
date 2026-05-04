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


def _bucket_name(bucket_idx: int) -> str:
    names = ["q1_early", "q2_mid_early", "q3_mid_late", "q4_late"]
    return names[int(bucket_idx)]


def run_mainline_like_suffix_position_probe(
    *,
    checkpoint_path: str,
    device: str | None = None,
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
    frozen_h_seeds = list(frozen_h_seeds or [12345, 23456, 34567, 45678])
    train_env_seeds = [int(train_env_seed_start) + i for i in range(int(train_env_count))]
    eval_rollout_seeds = [int(eval_rollout_seed_start) + i for i in range(int(train_env_count))]

    token_rows = []
    trajectory_rows = []

    for frozen_h_seed in frozen_h_seeds:
        load_model.cache_clear()
        model, config = load_model(checkpoint_path, device=device_obj, verbose=False)
        num_features = int(config["prior"]["num_features"])
        env_cfg = _build_audit_env_cfg(config["prior"]["environment"])

        prior_for_h = EnvironmentPrior(copy.deepcopy(env_cfg))
        with _temporary_seed(int(frozen_h_seed)):
            frozen_h = _freeze_env_h_list_for_replay(
                prior_for_h,
                list(prior_for_h._sample_batch_hypers(1)),
            )[0]

        prior = EnvironmentPrior(copy.deepcopy(env_cfg))
        prior._sample_batch_hypers = lambda batch_n, _h=frozen_h: [copy.deepcopy(_h) for _ in range(int(batch_n))]
        prior._sample_single_eval_pos = lambda n_samples_arg, rng_seed=None: int(single_eval_pos)
        _apply_boundary_contract_mode_flags(prior, boundary_contract_mode)

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
            actor_objective_mode="tokenwise",
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
                single_eval_pos=single_eval_pos,
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
            return orig_train()

        algo.train = wrapped_train
        algo.learn(total_timesteps=int(n_steps) * int(outer_epochs), callback=callback)

        post_eval = _run_eval(
            prior=prior,
            suite=suite,
            policy_step_fn=_build_audit_ppo_policy_step_fn(
                algo.policy,
                sampled=True,
                single_eval_pos=single_eval_pos,
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
            objective_count = int(env_adv.size)
            delta_suffix = float(post_suffix[env_idx] - pre_suffix[env_idx])
            trajectory_rows.append(
                {
                    "frozen_h_seed": int(frozen_h_seed),
                    "env_seed": int(env_seed),
                    "objective_count": objective_count,
                    "delta_suffix_return": delta_suffix,
                    "pre_suffix_return": float(pre_suffix[env_idx]),
                    "post_suffix_return": float(post_suffix[env_idx]),
                }
            )
            if objective_count <= 0:
                continue
            positions = np.arange(objective_count, dtype=np.float64)
            denom = float(max(objective_count - 1, 1))
            rel_pos = positions / denom
            bucket_ids = np.clip((rel_pos * 4.0).astype(np.int64), 0, 3)
            for tok_idx, (tok_adv, tok_rel, bucket_idx) in enumerate(zip(env_adv.tolist(), rel_pos.tolist(), bucket_ids.tolist())):
                token_rows.append(
                    {
                        "frozen_h_seed": int(frozen_h_seed),
                        "env_seed": int(env_seed),
                        "token_index": int(tok_idx),
                        "relative_suffix_position": float(tok_rel),
                        "position_bucket": _bucket_name(int(bucket_idx)),
                        "token_advantage": float(tok_adv),
                        "parent_delta_suffix_return": delta_suffix,
                        "parent_pre_suffix_return": float(pre_suffix[env_idx]),
                        "parent_post_suffix_return": float(post_suffix[env_idx]),
                    }
                )

    bucket_summary = {}
    for bucket_idx in range(4):
        bucket_name = _bucket_name(bucket_idx)
        bucket_rows = [r for r in token_rows if r["position_bucket"] == bucket_name]
        if not bucket_rows:
            bucket_summary[bucket_name] = {
                "count": 0,
                "corr_adv_vs_parent_delta": float("nan"),
                "high_adv_nonpositive_parent_count": 0,
                "high_adv_count": 0,
                "mean_adv_nonpositive_parent": float("nan"),
                "mean_adv_positive_parent": float("nan"),
                "nonpositive_parent_share": float("nan"),
            }
            continue
        adv = np.asarray([r["token_advantage"] for r in bucket_rows], dtype=np.float64)
        parent_delta = np.asarray([r["parent_delta_suffix_return"] for r in bucket_rows], dtype=np.float64)
        q75 = float(np.quantile(adv, 0.75))
        high = adv >= q75
        nonpositive_parent = parent_delta <= 0.0
        bucket_summary[bucket_name] = {
            "count": int(len(bucket_rows)),
            "corr_adv_vs_parent_delta": _pearson(adv, parent_delta),
            "high_adv_nonpositive_parent_count": int(np.logical_and(high, nonpositive_parent).sum()),
            "high_adv_count": int(high.sum()),
            "mean_adv_nonpositive_parent": float(adv[nonpositive_parent].mean()) if int(nonpositive_parent.sum()) > 0 else float("nan"),
            "mean_adv_positive_parent": float(adv[~nonpositive_parent].mean()) if int((~nonpositive_parent).sum()) > 0 else float("nan"),
            "nonpositive_parent_share": float(nonpositive_parent.mean()),
        }

    all_adv = np.asarray([r["token_advantage"] for r in token_rows], dtype=np.float64)
    all_parent_delta = np.asarray([r["parent_delta_suffix_return"] for r in token_rows], dtype=np.float64)
    q75_all = float(np.quantile(all_adv, 0.75)) if int(all_adv.size) > 0 else float("nan")
    high_all = all_adv >= q75_all if int(all_adv.size) > 0 else np.zeros((0,), dtype=bool)
    nonpositive_all = all_parent_delta <= 0.0 if int(all_parent_delta.size) > 0 else np.zeros((0,), dtype=bool)

    report = {
        "audit_entry": "mainline_like_suffix_position_probe",
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
            "boundary_contract_mode": str(boundary_contract_mode),
            "actor_objective_mode": "tokenwise",
            "actor_gae_space": "normalized",
            "actor_baseline_mode": "learned",
            "normalize_advantage": False,
        },
        "overall": {
            "token_count": int(len(token_rows)),
            "trajectory_count": int(len(trajectory_rows)),
            "corr_adv_vs_parent_delta": _pearson(all_adv, all_parent_delta),
            "high_adv_nonpositive_parent_count": int(np.logical_and(high_all, nonpositive_all).sum()),
            "high_adv_count": int(high_all.sum()),
            "nonpositive_parent_share": float(nonpositive_all.mean()) if int(nonpositive_all.size) > 0 else float("nan"),
        },
        "bucket_summary": bucket_summary,
        "token_rows": token_rows,
        "trajectory_rows": trajectory_rows,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_path")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--boundary-contract-mode", default="normal")
    args = parser.parse_args()

    report = run_mainline_like_suffix_position_probe(
        checkpoint_path=args.checkpoint_path,
        device=args.device,
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
