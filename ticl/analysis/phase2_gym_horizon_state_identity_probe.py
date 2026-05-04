import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from ticl.analysis.phase2_suffix_state_identity_probe import (
    _compute_horizon_diagnostics,
    _compute_identity_metrics,
    _resolve_horizon_diagnostics_steps,
    _validate_discount_gamma,
)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


def _make_env(env_id: str):
    import gymnasium as gym

    if str(env_id) == "Ant-v5_clean_obs":
        return gym.make(
            "Ant-v5",
            include_cfrc_ext_in_observation=False,
            terminate_when_unhealthy=False,
        )
    return gym.make(str(env_id))


def _capture_mujoco_state(env) -> dict[str, np.ndarray]:
    unwrapped = env.unwrapped
    return {
        "qpos": np.array(unwrapped.data.qpos, dtype=np.float64, copy=True),
        "qvel": np.array(unwrapped.data.qvel, dtype=np.float64, copy=True),
    }


def _restore_mujoco_state(env, state: dict[str, np.ndarray]) -> np.ndarray:
    unwrapped = env.unwrapped
    unwrapped.set_state(np.asarray(state["qpos"], dtype=np.float64), np.asarray(state["qvel"], dtype=np.float64))
    return np.asarray(unwrapped._get_obs(), dtype=np.float64)


class FixedRandomLinearGaussianPolicy:
    def __init__(
        self,
        *,
        obs_dim: int,
        action_dim: int,
        seed: int,
        sigma: float,
        policy_type: str,
    ) -> None:
        self.policy_type = str(policy_type)
        self.action_dim = int(action_dim)
        rng = np.random.default_rng(int(seed))
        self.weight = rng.standard_normal((int(action_dim), int(obs_dim))).astype(np.float64) / np.sqrt(
            max(1, int(obs_dim))
        )
        self.bias = 0.1 * rng.standard_normal((int(action_dim),)).astype(np.float64)
        self.sigma = float(sigma)

    def action(self, obs: np.ndarray, rng: np.random.Generator, low: np.ndarray, high: np.ndarray) -> np.ndarray:
        if self.policy_type == "unit_gaussian":
            return np.clip(rng.standard_normal((self.action_dim,)), low, high).astype(np.float32)
        mean = self.weight @ np.asarray(obs, dtype=np.float64).reshape(-1) + self.bias
        noisy = mean + float(self.sigma) * rng.standard_normal(mean.shape)
        return np.clip(noisy, low, high).astype(np.float32)


def _roll_to_anchor(
    *,
    env,
    policy: FixedRandomLinearGaussianPolicy,
    anchor_step: int,
    reference_seed: int,
) -> tuple[np.ndarray, dict[str, np.ndarray], float]:
    obs, _ = env.reset(seed=int(reference_seed))
    obs = np.asarray(obs, dtype=np.float64)
    rng = np.random.default_rng(int(reference_seed))
    action_low = np.asarray(env.action_space.low, dtype=np.float64)
    action_high = np.asarray(env.action_space.high, dtype=np.float64)
    last_reward = 0.0
    for _ in range(int(anchor_step)):
        action = policy.action(obs, rng, action_low, action_high)
        obs, reward, terminated, truncated, _ = env.step(action)
        obs = np.asarray(obs, dtype=np.float64)
        last_reward = float(reward)
        if bool(terminated) or bool(truncated):
            obs, _ = env.reset(seed=int(reference_seed) + 1)
            obs = np.asarray(obs, dtype=np.float64)
    state = _capture_mujoco_state(env)
    return obs, state, last_reward


def _rollout_discounted_rewards_from_anchor(
    *,
    env,
    policy: FixedRandomLinearGaussianPolicy,
    anchor_state: dict[str, np.ndarray],
    continuation_seed: int,
    horizon: int,
    gamma: float,
) -> np.ndarray:
    obs = _restore_mujoco_state(env, anchor_state)
    rng = np.random.default_rng(int(continuation_seed))
    action_low = np.asarray(env.action_space.low, dtype=np.float64)
    action_high = np.asarray(env.action_space.high, dtype=np.float64)
    rewards = np.zeros((int(horizon),), dtype=np.float32)
    running_discount = 1.0
    for step_idx in range(int(horizon)):
        action = policy.action(obs, rng, action_low, action_high)
        obs, reward, terminated, truncated, _ = env.step(action)
        obs = np.asarray(obs, dtype=np.float64)
        rewards[int(step_idx)] = float(running_discount) * float(reward)
        running_discount *= float(gamma)
        if bool(terminated) or bool(truncated):
            break
    return rewards


def run_probe(
    *,
    env_id: str,
    reference_state_count: int,
    continuation_repeats: int,
    anchor_step: int,
    n_steps: int,
    discount_gamma: float,
    policy_type: str,
    policy_seed: int,
    sigma: float,
    reference_seed_start: int,
    continuation_seed_start: int,
    horizon_diagnostics_steps: Any,
) -> dict[str, Any]:
    if int(n_steps) <= int(anchor_step):
        raise ValueError(f"n_steps must exceed anchor_step, got {n_steps=} and {anchor_step=}")
    gamma = _validate_discount_gamma(float(discount_gamma))
    horizon = int(n_steps) - int(anchor_step)
    horizon_steps = _resolve_horizon_diagnostics_steps(horizon_diagnostics_steps, remaining_horizon=horizon)

    env = _make_env(str(env_id))
    try:
        obs_dim = int(np.prod(env.observation_space.shape))
        action_dim = int(np.prod(env.action_space.shape))
        policy = FixedRandomLinearGaussianPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            seed=int(policy_seed),
            sigma=float(sigma),
            policy_type=str(policy_type),
        )
        discounted_step_rewards = np.zeros(
            (int(reference_state_count), int(continuation_repeats), int(horizon)),
            dtype=np.float32,
        )
        anchors = []
        restore_errors = []
        for state_idx in range(int(reference_state_count)):
            reference_seed = int(reference_seed_start) + int(state_idx)
            anchor_obs, anchor_state, anchor_reward = _roll_to_anchor(
                env=env,
                policy=policy,
                anchor_step=int(anchor_step),
                reference_seed=int(reference_seed),
            )
            restored_obs = _restore_mujoco_state(env, anchor_state)
            restore_errors.append(float(np.max(np.abs(restored_obs - anchor_obs))))
            for repeat_idx in range(int(continuation_repeats)):
                continuation_seed = (
                    int(continuation_seed_start) + int(state_idx) * int(continuation_repeats) + int(repeat_idx)
                )
                discounted_step_rewards[int(state_idx), int(repeat_idx)] = _rollout_discounted_rewards_from_anchor(
                    env=env,
                    policy=policy,
                    anchor_state=anchor_state,
                    continuation_seed=int(continuation_seed),
                    horizon=int(horizon),
                    gamma=float(gamma),
                )
            state_returns = discounted_step_rewards[int(state_idx)].sum(axis=1, dtype=np.float64)
            anchors.append(
                {
                    "anchor_index": int(state_idx),
                    "reference_seed": int(reference_seed),
                    "anchor_reward": float(anchor_reward),
                    "continuation_return_mean": float(np.mean(state_returns, dtype=np.float64)),
                    "continuation_return_std": float(np.std(state_returns, dtype=np.float64)),
                    "continuation_return_min": float(np.min(state_returns)),
                    "continuation_return_max": float(np.max(state_returns)),
                }
            )
        return_matrix = discounted_step_rewards.sum(axis=2, dtype=np.float32)
        return {
            "audit_entry": "phase2_gym_horizon_state_identity_probe",
            "env_id": str(env_id),
            "base_env_id": "Ant-v5" if str(env_id) == "Ant-v5_clean_obs" else str(env_id),
            "reference_state_count": int(reference_state_count),
            "continuation_repeats": int(continuation_repeats),
            "anchor_step": int(anchor_step),
            "n_steps": int(n_steps),
            "remaining_horizon": int(horizon),
            "discount_gamma": float(gamma),
            "policy": {
                "type": str(policy_type),
                "policy_seed": int(policy_seed),
                "sigma": float(sigma),
            },
            "obs_dim": int(obs_dim),
            "action_dim": int(action_dim),
            "obs_restore_maxerr_mean": float(np.mean(restore_errors, dtype=np.float64)),
            "obs_restore_maxerr_max": float(np.max(restore_errors)),
            "aggregate": _compute_identity_metrics(return_matrix),
            "horizon_diagnostics_steps": horizon_steps,
            "horizon_diagnostics": _compute_horizon_diagnostics(
                discounted_step_rewards,
                horizon_steps=horizon_steps,
                continuation_repeats=int(continuation_repeats),
            ),
            "anchors": anchors,
        }
    finally:
        env.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gym MuJoCo horizon-wise state-return identity probe.")
    parser.add_argument("--env-id", type=str, default="Ant-v5_clean_obs")
    parser.add_argument("--reference-state-count", type=int, default=64)
    parser.add_argument("--continuation-repeats", type=int, default=64)
    parser.add_argument("--anchor-step", type=int, default=150)
    parser.add_argument("--n-steps", type=int, default=300)
    parser.add_argument("--discount-gamma", type=float, default=0.99)
    parser.add_argument("--policy-type", type=str, default="unit_gaussian", choices=["unit_gaussian", "fixed_random_linear_gaussian"])
    parser.add_argument("--policy-seed", type=int, default=20260422)
    parser.add_argument("--sigma", type=float, default=0.25)
    parser.add_argument("--reference-seed-start", type=int, default=5000)
    parser.add_argument("--continuation-seed-start", type=int, default=10000)
    parser.add_argument("--horizon-diagnostics-steps", type=str, default="powers")
    parser.add_argument("--output-json", type=str, required=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = run_probe(
        env_id=str(args.env_id),
        reference_state_count=int(args.reference_state_count),
        continuation_repeats=int(args.continuation_repeats),
        anchor_step=int(args.anchor_step),
        n_steps=int(args.n_steps),
        discount_gamma=float(args.discount_gamma),
        policy_type=str(args.policy_type),
        policy_seed=int(args.policy_seed),
        sigma=float(args.sigma),
        reference_seed_start=int(args.reference_seed_start),
        continuation_seed_start=int(args.continuation_seed_start),
        horizon_diagnostics_steps=args.horizon_diagnostics_steps,
    )
    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_json_safe(report), sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(_json_safe(report), sort_keys=True))


if __name__ == "__main__":
    main()
