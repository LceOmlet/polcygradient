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
    run_critic_free_single_env_audit,
)
from ticl.analysis.critic_value_fit_probe import (
    _clone_h_repeated,
    _critic_only_step,
    _seed_all,
)
from ticl.model_builder import load_model
from ticl.priors.environment_prior import EnvironmentPrior
from ticl.sb3_recurrent_ppo import build_recurrent_ppo
from stable_baselines3.common.logger import configure as configure_logger


def _default_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _build_strict_shared_algo(
    *,
    checkpoint_path: str,
    device_obj: torch.device,
    frozen_h,
    train_env_seed: int,
    train_rollout_seed: int,
    single_eval_pos: int,
    n_steps: int,
    batch_size: int,
    n_epochs: int,
    learning_rate: float,
    target_kl: float,
    ppo_reset_env_state_at_sep: bool,
):
    load_model.cache_clear()
    model, config = load_model(checkpoint_path, device=device_obj, verbose=False)
    num_features = int(config["prior"]["num_features"])
    env_cfg = _build_audit_env_cfg(config["prior"]["environment"])
    prior = EnvironmentPrior(copy.deepcopy(env_cfg))
    prior._sample_batch_hypers = lambda batch_n, _h=frozen_h: _clone_h_repeated(_h, int(batch_n))
    prior._sample_single_eval_pos = lambda n_samples_arg, rng_seed=None: int(single_eval_pos)
    _apply_boundary_contract_mode_flags(prior, "normal")
    _apply_sep_state_reset_flag(prior, bool(ppo_reset_env_state_at_sep))

    algo, callback, vec_env = build_recurrent_ppo(
        model=model,
        env_prior=prior,
        device=str(device_obj),
        num_features=num_features,
        n_envs=1,
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
        reset_env_state_at_sep=bool(ppo_reset_env_state_at_sep),
        separate_value_backbone=False,
        ent_coef=float(config["optimizer"].get("ppo_ent_coef", 0.0)),
        vf_coef=1.0,
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
    algo._rwkv_policy_loss_coef = 0.0
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


def _run_critic_only_recollect_trajectory(
    *,
    checkpoint_path: str,
    device_obj: torch.device,
    frozen_h,
    train_env_seed: int,
    train_rollout_seed: int,
    single_eval_pos: int,
    n_steps: int,
    batch_size: int,
    n_epochs: int,
    learning_rate: float,
    target_kl: float,
    outer_epochs: int,
    critic_updates_per_outer_epoch: int,
    ppo_reset_env_state_at_sep: bool,
    build_seed: int,
) -> dict:
    _seed_all(int(build_seed))
    algo, callback, vec_env = _build_strict_shared_algo(
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
    )
    try:
        history: list[dict] = []
        total_timesteps = int(n_steps) * int(outer_epochs)
        for outer_idx in range(int(outer_epochs)):
            algo._update_current_progress_remaining(int(outer_idx * n_steps), total_timesteps)
            ok = algo.collect_rollouts(vec_env, callback, algo.rollout_buffer, n_rollout_steps=algo.n_steps)
            if not ok:
                raise RuntimeError("collect_rollouts returned False during critic-only recollect compare.")
            pre_quality = _measure_rollout_critic_quality(algo)
            rollout_data = next(algo.rollout_buffer.get_gpu_flat(algo.batch_size))
            for _ in range(int(critic_updates_per_outer_epoch)):
                _critic_only_step(algo, rollout_data)
            post_quality = _measure_current_policy_rollout_critic_quality(algo)
            history.append(
                {
                    "outer_epoch": int(outer_idx + 1),
                    "pre_ev_norm": float(pre_quality["explained_variance_normalized"]),
                    "pre_ev_raw": float(pre_quality["explained_variance_raw"]),
                    "pre_raw_corr": float(pre_quality["raw_corr"]),
                    "pre_norm_value_std": float(pre_quality["normalized_value_std"]),
                    "pre_norm_return_std": float(pre_quality["normalized_return_std"]),
                    "post_ev_norm": float(post_quality["explained_variance_normalized"]),
                    "post_ev_raw": float(post_quality["explained_variance_raw"]),
                    "post_raw_corr": float(post_quality["raw_corr"]),
                    "post_norm_value_std": float(post_quality["normalized_value_std"]),
                    "post_norm_return_std": float(post_quality["normalized_return_std"]),
                }
            )
        return {
            "critic_updates_per_outer_epoch": int(critic_updates_per_outer_epoch),
            "history": history,
        }
    finally:
        vec_env.close()
        del algo, callback, vec_env
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _zip_compare(actor_loss0_history: list[dict], critic_only_history: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for actor_row, critic_row in zip(actor_loss0_history, critic_only_history):
        rows.append(
            {
                "outer_epoch": int(actor_row["outer_epoch"]),
                "actor_loss0_post_ev_norm": float(actor_row["critic_explained_variance_normalized_post_train_rollout"]),
                "critic_only_post_ev_norm": float(critic_row["post_ev_norm"]),
                "actor_loss0_post_raw_corr": float(actor_row["critic_raw_corr_post_train_rollout"]),
                "critic_only_post_raw_corr": float(critic_row["post_raw_corr"]),
                "actor_loss0_post_norm_value_std": float(actor_row["critic_normalized_value_std_post_train_rollout"]),
                "critic_only_post_norm_value_std": float(critic_row["post_norm_value_std"]),
            }
        )
    return rows


def run_phase2_shared_value_only_recollect_compare(
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
    outer_epochs: int = 4,
    critic_updates_per_outer_epoch: int = 50,
    build_seed: int = 4040,
    ppo_reset_env_state_at_sep: bool = True,
) -> dict:
    device_obj = torch.device(str(device or _default_device()))
    _seed_all(int(build_seed))
    load_model.cache_clear()
    _, config = load_model(checkpoint_path, device=device_obj, verbose=False)
    env_cfg = _build_audit_env_cfg(config["prior"]["environment"])
    fixed_bundle = _prepare_audit_fixed_env_contract(
        env_cfg=env_cfg,
        boundary_contract_mode="normal",
        ppo_reset_env_state_at_sep=bool(ppo_reset_env_state_at_sep),
        frozen_h_seed=int(frozen_h_seed),
        frozen_h_seed_mode="current",
    )
    frozen_h = fixed_bundle["frozen_h"]

    critic_only = _run_critic_only_recollect_trajectory(
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
        outer_epochs=int(outer_epochs),
        critic_updates_per_outer_epoch=int(critic_updates_per_outer_epoch),
        ppo_reset_env_state_at_sep=bool(ppo_reset_env_state_at_sep),
        build_seed=int(build_seed),
    )
    actor_loss0_report = run_critic_free_single_env_audit(
        checkpoint_path=checkpoint_path,
        device=str(device_obj),
        frozen_h_seed=int(frozen_h_seed),
        train_env_seed=int(train_env_seed),
        eval_env_seed_start=3000,
        eval_rollout_seed_start=4000,
        eval_env_count=1,
        single_eval_pos=int(single_eval_pos),
        randomize_single_eval_pos=False,
        n_steps=int(n_steps),
        outer_epochs=int(outer_epochs),
        learning_rate=float(learning_rate),
        batch_size=int(batch_size),
        n_epochs=int(n_epochs),
        target_kl=float(target_kl),
        strict_fixed_env_mode=True,
        train_rollout_seed=int(train_rollout_seed),
        deterministic_actor_sampling=True,
        deterministic_batch_plan=True,
        strict_native_rollout=False,
        restore_validation_policy_state=False,
        build_seed=int(build_seed),
        frozen_h_seed_mode="current",
        ppo_reset_env_state_at_sep=bool(ppo_reset_env_state_at_sep),
        ppo_separate_value_backbone=False,
        extra_critic_updates_per_outer_epoch=int(max(0, critic_updates_per_outer_epoch - 1)),
        extra_critic_update_scope="shared",
        actor_gae_space="normalized",
        actor_objective_mode="tokenwise",
        boundary_contract_mode="normal",
        critic_warmup_min_outer_epochs=0,
        critic_warmup_raw_corr_threshold=0.0,
        record_suite_critic_quality=False,
        record_post_train_rollout_critic_quality=True,
        vf_coef_override=1.0,
        policy_loss_coef_override=0.0,
        modes=["learned"],
    )
    actor_history = actor_loss0_report["modes"]["learned"]["critic_train_history"]
    zipped = _zip_compare(actor_history, critic_only["history"])
    return {
        "audit_entry": "phase2_shared_value_only_recollect_compare",
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
        "outer_epochs": int(outer_epochs),
        "critic_updates_per_outer_epoch": int(critic_updates_per_outer_epoch),
        "build_seed": int(build_seed),
        "ppo_reset_env_state_at_sep": bool(ppo_reset_env_state_at_sep),
        "fixed_env_contract": dict(fixed_bundle["fixed_env_contract"]),
        "critic_only_recollect": critic_only,
        "actor_loss0_recollect": actor_loss0_report["modes"]["learned"],
        "side_by_side": zipped,
    }


def main():
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
    parser.add_argument("--outer-epochs", type=int, default=4)
    parser.add_argument("--critic-updates-per-outer-epoch", type=int, default=50)
    parser.add_argument("--build-seed", type=int, default=4040)
    parser.add_argument("--ppo-reset-env-state-at-sep", action="store_true")
    parser.add_argument(
        "--output-json",
        type=str,
        default="/home/chen/RLPFN/artifacts/phase2_shared_value_only_recollect_compare.json",
    )
    args = parser.parse_args()

    report = run_phase2_shared_value_only_recollect_compare(
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
        outer_epochs=int(args.outer_epochs),
        critic_updates_per_outer_epoch=int(args.critic_updates_per_outer_epoch),
        build_seed=int(args.build_seed),
        ppo_reset_env_state_at_sep=bool(args.ppo_reset_env_state_at_sep),
    )
    payload = json.dumps(report, sort_keys=True)
    out_path = Path(args.output_json).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
