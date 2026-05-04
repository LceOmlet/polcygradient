"""Official SB3 PPO single-Gym-env baseline with raw episodic eval returns.

This script is intentionally a thin experiment wrapper: policy optimization is
delegated to the vendored stable_baselines3.PPO implementation, and observation
/ reward normalization is delegated to stable_baselines3 VecNormalize.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _patch_cv2_for_vendored_sb3() -> None:
    try:
        import cv2  # type: ignore
    except Exception:
        return
    if hasattr(cv2, "ocl"):
        return

    class _OpenCLShim:
        @staticmethod
        def setUseOpenCL(flag: bool) -> None:
            del flag

    cv2.ocl = _OpenCLShim()  # type: ignore[attr-defined]


_patch_cv2_for_vendored_sb3()

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


DEFAULT_ENVS = [
    "BipedalWalker-v3",
    "LunarLanderContinuous-v3",
    "HalfCheetah-v5",
    "Pendulum-v1",
    "Reacher-v5",
]


def _parse_bool(text: str | int | bool) -> bool:
    if isinstance(text, bool):
        return text
    if isinstance(text, int):
        return bool(text)
    value = str(text).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid bool: {text!r}")


def _stats(values: list[float]) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "q10": float(np.quantile(arr, 0.10)),
        "q50": float(np.quantile(arr, 0.50)),
        "q90": float(np.quantile(arr, 0.90)),
        "max": float(np.max(arr)),
    }


def _make_train_env(
    env_id: str,
    *,
    n_envs: int,
    seed: int,
    monitor_dir: Path,
    norm_obs: bool,
    norm_reward: bool,
    clip_obs: float,
    clip_reward: float,
    gamma: float,
) -> VecNormalize:
    env = make_vec_env(
        env_id,
        n_envs=n_envs,
        seed=seed,
        monitor_dir=str(monitor_dir),
        vec_env_cls=DummyVecEnv,
    )
    return VecNormalize(
        env,
        training=True,
        norm_obs=norm_obs,
        norm_reward=norm_reward,
        clip_obs=clip_obs,
        clip_reward=clip_reward,
        gamma=gamma,
    )


def _make_eval_env(
    env_id: str,
    *,
    seed: int,
    train_env: VecNormalize,
    norm_obs: bool,
    clip_obs: float,
    gamma: float,
) -> VecNormalize:
    env = make_vec_env(
        env_id,
        n_envs=1,
        seed=seed,
        monitor_dir=None,
        vec_env_cls=DummyVecEnv,
    )
    eval_env = VecNormalize(
        env,
        training=False,
        norm_obs=norm_obs,
        norm_reward=False,
        clip_obs=clip_obs,
        gamma=gamma,
    )
    if norm_obs:
        eval_env.obs_rms = copy.deepcopy(train_env.obs_rms)
    return eval_env


def _evaluate_raw(
    model: PPO,
    env_id: str,
    *,
    train_env: VecNormalize,
    seed: int,
    n_eval_episodes: int,
    deterministic: bool,
    norm_obs: bool,
    clip_obs: float,
    gamma: float,
) -> dict[str, Any]:
    eval_env = _make_eval_env(
        env_id,
        seed=seed,
        train_env=train_env,
        norm_obs=norm_obs,
        clip_obs=clip_obs,
        gamma=gamma,
    )
    try:
        returns, lengths = evaluate_policy(
            model,
            eval_env,
            n_eval_episodes=n_eval_episodes,
            deterministic=deterministic,
            return_episode_rewards=True,
            warn=False,
        )
    finally:
        eval_env.close()
    return {
        "deterministic": bool(deterministic),
        "raw_episode_return": _stats([float(v) for v in returns]),
        "episode_length": _stats([float(v) for v in lengths]),
        "returns": [float(v) for v in returns],
        "lengths": [int(v) for v in lengths],
    }


def _train_one_env(args: argparse.Namespace, env_id: str, env_index: int, output_dir: Path) -> dict[str, Any]:
    env_seed = int(args.seed) + env_index * 10_000
    random.seed(env_seed)
    np.random.seed(env_seed)
    torch.manual_seed(env_seed)

    env_dir = output_dir / env_id
    env_dir.mkdir(parents=True, exist_ok=True)
    progress_path = env_dir / "progress.jsonl"

    train_env = _make_train_env(
        env_id,
        n_envs=int(args.n_envs),
        seed=env_seed,
        monitor_dir=env_dir / "monitor",
        norm_obs=bool(args.norm_obs),
        norm_reward=bool(args.norm_reward),
        clip_obs=float(args.clip_obs),
        clip_reward=float(args.clip_reward),
        gamma=float(args.gamma),
    )
    model = PPO(
        "MlpPolicy",
        train_env,
        seed=env_seed,
        device=str(args.device),
        n_steps=int(args.n_steps),
        batch_size=int(args.batch_size),
        n_epochs=int(args.n_epochs),
        gamma=float(args.gamma),
        gae_lambda=float(args.gae_lambda),
        learning_rate=float(args.learning_rate),
        ent_coef=float(args.ent_coef),
        vf_coef=float(args.vf_coef),
        clip_range=float(args.clip_range),
        verbose=0,
    )

    eval_updates = {0, int(args.updates)}
    eval_updates.update(range(int(args.eval_every_updates), int(args.updates) + 1, int(args.eval_every_updates)))
    update_records: list[dict[str, Any]] = []

    def write_eval(update_idx: int) -> None:
        record = {
            "env_id": env_id,
            "env_index": int(env_index),
            "seed": int(env_seed),
            "update_idx": int(update_idx),
            "timesteps": int(model.num_timesteps),
            "n_envs": int(args.n_envs),
            "n_steps": int(args.n_steps),
            "norm_obs": bool(args.norm_obs),
            "norm_reward_train": bool(args.norm_reward),
            "eval_reward_contract": "raw episodic env reward; eval VecNormalize has norm_reward=False",
            "deterministic_eval": _evaluate_raw(
                model,
                env_id,
                train_env=train_env,
                seed=env_seed + 1_000_000 + update_idx * 17,
                n_eval_episodes=int(args.n_eval_episodes),
                deterministic=True,
                norm_obs=bool(args.norm_obs),
                clip_obs=float(args.clip_obs),
                gamma=float(args.gamma),
            ),
            "stochastic_eval": _evaluate_raw(
                model,
                env_id,
                train_env=train_env,
                seed=env_seed + 2_000_000 + update_idx * 17,
                n_eval_episodes=int(args.n_eval_episodes),
                deterministic=False,
                norm_obs=bool(args.norm_obs),
                clip_obs=float(args.clip_obs),
                gamma=float(args.gamma),
            ),
        }
        with progress_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")
        update_records.append(record)

    try:
        write_eval(0)
        for update_idx in range(1, int(args.updates) + 1):
            model.learn(
                total_timesteps=int(args.n_envs) * int(args.n_steps),
                reset_num_timesteps=False,
                progress_bar=False,
            )
            if update_idx in eval_updates:
                write_eval(update_idx)
        model.save(env_dir / "final_model.zip")
        train_env.save(str(env_dir / "vecnormalize.pkl"))
    finally:
        train_env.close()

    first = update_records[0]
    last = update_records[-1]
    det_first = first["deterministic_eval"]["raw_episode_return"]["mean"]
    det_last = last["deterministic_eval"]["raw_episode_return"]["mean"]
    stoch_first = first["stochastic_eval"]["raw_episode_return"]["mean"]
    stoch_last = last["stochastic_eval"]["raw_episode_return"]["mean"]
    summary = {
        "env_id": env_id,
        "seed": int(env_seed),
        "updates": int(args.updates),
        "total_timesteps": int(model.num_timesteps),
        "progress_path": str(progress_path),
        "deterministic_pre_mean": float(det_first),
        "deterministic_post_mean": float(det_last),
        "deterministic_delta": float(det_last - det_first),
        "stochastic_pre_mean": float(stoch_first),
        "stochastic_post_mean": float(stoch_last),
        "stochastic_delta": float(stoch_last - stoch_first),
    }
    (env_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-ids", type=str, default=",".join(DEFAULT_ENVS))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=5050)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--eval-every-updates", type=int, default=10)
    parser.add_argument("--n-eval-episodes", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--norm-obs", type=_parse_bool, default=True)
    parser.add_argument("--norm-reward", type=_parse_bool, default=True)
    parser.add_argument("--clip-obs", type=float, default=10.0)
    parser.add_argument("--clip-reward", type=float, default=10.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    env_ids = [item.strip() for item in str(args.env_ids).split(",") if item.strip()]
    config = vars(args).copy()
    config["env_ids"] = env_ids
    config["output_dir"] = str(output_dir)
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    summaries = []
    for env_index, env_id in enumerate(env_ids):
        summary = _train_one_env(args, env_id, env_index, output_dir)
        summaries.append(summary)
        with (output_dir / "summary_partial.json").open("w", encoding="utf-8") as f:
            json.dump({"env_summaries": summaries}, f, indent=2, sort_keys=True)
            f.write("\n")

    report = {
        "contract": (
            "Official vendored stable_baselines3.PPO MlpPolicy single-env baselines; "
            "training may use VecNormalize obs/reward, eval returns are raw episodic env rewards."
        ),
        "env_summaries": summaries,
    }
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
