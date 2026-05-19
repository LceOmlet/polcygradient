#!/usr/bin/env python
"""Narrow Gym Pendulum mechanism-signature probe.

Exploratory/read-only only.  This script does not mutate the prior generator,
PPO implementation, fit_model defaults, checkpoints, or Gym environments.

The probe separates:

* Gym environment side: raw Pendulum-v1 observations, raw reward, and the exact
  Gym reward decomposition implied by observation/action.
* Policy side: either a trained official SB3 single-env PPO policy, the current
  fit_model validation PPO policy, or trivial zero/random actions.

The goal is not to define a new milestone.  It is to make the Pendulum failure
mode falsifiable: does the successful official policy exhibit a concrete
action-state/time signature that the fit_model policy lacks?
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.exploratory.phase2_gym_action_mode_counterfactual import (  # noqa: E402
    _drop_validation_value_only_state_for_action_probe,
)
from ticl.model_builder import load_model  # noqa: E402
from ticl.rl_validation import (  # noqa: E402
    _advance_validation_state_action_pairs,
    _build_validation_episode_states,
    _resolve_validation_action_bounds,
    _resolve_validation_recurrent_ppo_policy,
    _select_ppo_policy_actions_batched,
)
from ticl.rlpfn_maintained_path import resolve_rlpfn_token_layout  # noqa: E402


DEFAULT_FIT_CHECKPOINT = (
    "/home/chen/RLPFN/health/models_diff/"
    "rlpfn_ppopackpriormilestonegated_reward_path_balance_05_07_2026_01_53_23_epoch_29.cpkt"
)
DEFAULT_OFFICIAL_MODEL = (
    "/home/chen/RLPFN/artifacts/"
    "phase2_official_ppo_single_env_gym5_raweval_n8_u100_seed5050_gpu0_0502/"
    "Pendulum-v1/final_model.zip"
)
DEFAULT_OFFICIAL_VECNORMALIZE = (
    "/home/chen/RLPFN/artifacts/"
    "phase2_official_ppo_single_env_gym5_raweval_n8_u100_seed5050_gpu0_0502/"
    "Pendulum-v1/vecnormalize.pkl"
)
DEFAULT_OUTPUT_JSON = (
    "/home/chen/RLPFN/artifacts/phase2_gym_pendulum_mechanism_signature_probe_0509/"
    "fit_epoch29_vs_official_single_env.json"
)


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


def _parse_csv(value: str) -> list[str]:
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        out = float(value)
        return None if not math.isfinite(out) else out
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    return value


def _stats(values: list[float] | np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
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


def _corr(a: np.ndarray, b: np.ndarray) -> float | None:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 3:
        return None
    a = a[mask]
    b = b[mask]
    if float(np.std(a)) <= 1e-12 or float(np.std(b)) <= 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _pendulum_terms(obs: np.ndarray, action: np.ndarray) -> dict[str, float]:
    obs_arr = np.asarray(obs, dtype=np.float64).reshape(-1)
    action_arr = np.asarray(action, dtype=np.float64).reshape(-1)
    if obs_arr.size < 3:
        raise ValueError(f"Pendulum obs must have at least 3 dims, got {obs_arr.shape}")
    torque = float(np.clip(action_arr[0] if action_arr.size else 0.0, -2.0, 2.0))
    cos_theta = float(np.clip(obs_arr[0], -1.0, 1.0))
    sin_theta = float(np.clip(obs_arr[1], -1.0, 1.0))
    theta = float(np.arctan2(sin_theta, cos_theta))
    theta_dot = float(obs_arr[2])
    reward_angle = -(theta**2)
    reward_velocity = -0.1 * (theta_dot**2)
    reward_ctrl = -0.001 * (torque**2)
    return {
        "theta": theta,
        "abs_theta": abs(theta),
        "theta_dot": theta_dot,
        "abs_theta_dot": abs(theta_dot),
        "cos_theta": cos_theta,
        "sin_theta": sin_theta,
        "action": torque,
        "abs_action": abs(torque),
        "reward_angle": float(reward_angle),
        "reward_velocity": float(reward_velocity),
        "reward_ctrl": float(reward_ctrl),
        "reward_reconstructed": float(reward_angle + reward_velocity + reward_ctrl),
        # Classic swing-up energy-shaping policies often need phase-sensitive
        # torque relative to velocity/angle.  These are diagnostics only.
        "action_x_theta_dot": float(torque * theta_dot),
        "action_x_sin_theta": float(torque * sin_theta),
        "action_x_theta_dot_cos_theta": float(torque * theta_dot * cos_theta),
    }


def _record_step(
    records: list[dict[str, Any]],
    *,
    episode_idx: int,
    step_idx: int,
    obs: np.ndarray,
    action: np.ndarray,
    reward: float,
    terminated: bool,
    truncated: bool,
    source: str,
) -> None:
    terms = _pendulum_terms(obs, action)
    records.append(
        {
            "source": str(source),
            "episode_idx": int(episode_idx),
            "step_idx": int(step_idx),
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            **terms,
        }
    )


def _source_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"step_count": 0, "episode_count": 0}
    by_ep: dict[int, list[dict[str, Any]]] = {}
    for rec in records:
        by_ep.setdefault(int(rec["episode_idx"]), []).append(rec)
    returns = [float(sum(float(r["reward"]) for r in seq)) for seq in by_ep.values()]
    lengths = [float(len(seq)) for seq in by_ep.values()]
    rows = records

    def arr(key: str) -> np.ndarray:
        return np.asarray([float(r[key]) for r in rows], dtype=np.float64)

    early_rows: list[dict[str, Any]] = []
    late_rows: list[dict[str, Any]] = []
    middle_rows: list[dict[str, Any]] = []
    for seq in by_ep.values():
        n = max(1, len(seq))
        first_cut = max(1, int(math.ceil(0.25 * n)))
        late_cut = max(0, int(math.floor(0.75 * n)))
        early_rows.extend(seq[:first_cut])
        middle_rows.extend(seq[first_cut:late_cut])
        late_rows.extend(seq[late_cut:])

    def arr_from(seq: list[dict[str, Any]], key: str) -> np.ndarray:
        return np.asarray([float(r[key]) for r in seq], dtype=np.float64)

    action = arr("action")
    theta_dot = arr("theta_dot")
    sin_theta = arr("sin_theta")
    cos_theta = arr("cos_theta")
    abs_theta = arr("abs_theta")
    abs_action = arr("abs_action")
    reward = arr("reward")
    action_state_features = {
        "corr_action_theta_dot": _corr(action, theta_dot),
        "corr_action_sin_theta": _corr(action, sin_theta),
        "corr_action_theta_dot_cos_theta": _corr(action, theta_dot * cos_theta),
        "corr_abs_action_abs_theta": _corr(abs_action, abs_theta),
        "corr_abs_action_reward": _corr(abs_action, reward),
    }
    def corr_from(seq: list[dict[str, Any]]) -> dict[str, float | None]:
        if not seq:
            return {
                "corr_action_theta_dot": None,
                "corr_action_sin_theta": None,
                "corr_action_theta_dot_cos_theta": None,
                "corr_abs_action_abs_theta": None,
                "corr_abs_action_reward": None,
            }
        seq_action = arr_from(seq, "action")
        seq_theta_dot = arr_from(seq, "theta_dot")
        seq_sin_theta = arr_from(seq, "sin_theta")
        seq_cos_theta = arr_from(seq, "cos_theta")
        seq_abs_action = arr_from(seq, "abs_action")
        seq_abs_theta = arr_from(seq, "abs_theta")
        seq_reward = arr_from(seq, "reward")
        return {
            "corr_action_theta_dot": _corr(seq_action, seq_theta_dot),
            "corr_action_sin_theta": _corr(seq_action, seq_sin_theta),
            "corr_action_theta_dot_cos_theta": _corr(seq_action, seq_theta_dot * seq_cos_theta),
            "corr_abs_action_abs_theta": _corr(seq_abs_action, seq_abs_theta),
            "corr_abs_action_reward": _corr(seq_abs_action, seq_reward),
        }
    early_abs_action = arr_from(early_rows, "abs_action")
    late_abs_action = arr_from(late_rows, "abs_action")
    early_abs_theta = arr_from(early_rows, "abs_theta")
    late_abs_theta = arr_from(late_rows, "abs_theta")
    early_reward_angle = arr_from(early_rows, "reward_angle")
    late_reward_angle = arr_from(late_rows, "reward_angle")
    per_ep_late_upright_frac = []
    per_ep_late_action_abs = []
    per_ep_early_action_abs = []
    per_ep_late_abs_theta = []
    for seq in by_ep.values():
        n = max(1, len(seq))
        first_cut = max(1, int(math.ceil(0.25 * n)))
        late_cut = max(0, int(math.floor(0.75 * n)))
        late = seq[late_cut:]
        if late:
            per_ep_late_upright_frac.append(float(np.mean([float(r["abs_theta"]) < 0.5 for r in late])))
            per_ep_late_action_abs.append(float(np.mean([float(r["abs_action"]) for r in late])))
            per_ep_late_abs_theta.append(float(np.mean([float(r["abs_theta"]) for r in late])))
        early = seq[:first_cut]
        if early:
            per_ep_early_action_abs.append(float(np.mean([float(r["abs_action"]) for r in early])))

    return {
        "step_count": int(len(records)),
        "episode_count": int(len(by_ep)),
        "raw_episode_return": _stats(returns),
        "episode_length": _stats(lengths),
        "reward": _stats(reward),
        "reward_angle": _stats(arr("reward_angle")),
        "reward_velocity": _stats(arr("reward_velocity")),
        "reward_ctrl": _stats(arr("reward_ctrl")),
        "reward_reconstruction_abs_error": _stats(np.abs(arr("reward_reconstructed") - reward)),
        "abs_theta": _stats(abs_theta),
        "abs_theta_dot": _stats(arr("abs_theta_dot")),
        "abs_action": _stats(abs_action),
        "early_abs_action": _stats(early_abs_action),
        "middle_abs_action": _stats(arr_from(middle_rows, "abs_action")),
        "late_abs_action": _stats(late_abs_action),
        "early_abs_theta": _stats(early_abs_theta),
        "late_abs_theta": _stats(late_abs_theta),
        "early_reward_angle": _stats(early_reward_angle),
        "late_reward_angle": _stats(late_reward_angle),
        "late_upright_frac_abs_theta_lt_0_5": _stats(per_ep_late_upright_frac),
        "per_episode_early_minus_late_abs_action": _stats(
            np.asarray(per_ep_early_action_abs) - np.asarray(per_ep_late_action_abs)
            if len(per_ep_early_action_abs) == len(per_ep_late_action_abs)
            else []
        ),
        "per_episode_late_abs_theta": _stats(per_ep_late_abs_theta),
        "action_state_correlation": action_state_features,
        "early_action_state_correlation": corr_from(early_rows),
        "middle_action_state_correlation": corr_from(middle_rows),
        "late_action_state_correlation": corr_from(late_rows),
    }


def _early_active(step_idx: int, max_steps: int, early_frac: float) -> bool:
    cutoff = max(1, int(round(float(max_steps) * float(max(0.01, min(0.99, early_frac))))))
    return int(step_idx) < int(cutoff)


def _run_trivial_source(
    *,
    source: str,
    episodes: int,
    max_steps: int,
    seed: int,
    early_frac: float,
) -> list[dict[str, Any]]:
    import gymnasium as gym

    records: list[dict[str, Any]] = []
    for ep_idx in range(int(episodes)):
        env = gym.make("Pendulum-v1")
        rng = np.random.default_rng(int(seed) + int(ep_idx))
        obs, _ = env.reset(seed=int(seed) + int(ep_idx) * 1000)
        try:
            for step_idx in range(int(max_steps)):
                if source == "zero":
                    action = np.zeros((1,), dtype=np.float32)
                elif source in {"random", "random_early_then_zero"}:
                    action = rng.uniform(low=-2.0, high=2.0, size=(1,)).astype(np.float32)
                    if source == "random_early_then_zero" and not _early_active(step_idx, max_steps, early_frac):
                        action = np.zeros((1,), dtype=np.float32)
                else:
                    raise ValueError(f"unsupported trivial source={source!r}")
                obs_before = np.asarray(obs, dtype=np.float32).copy()
                obs, reward, terminated, truncated, _info = env.step(action)
                _record_step(
                    records,
                    episode_idx=ep_idx,
                    step_idx=step_idx,
                    obs=obs_before,
                    action=action,
                    reward=float(reward),
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                    source=source,
                )
                if bool(terminated or truncated):
                    break
        finally:
            env.close()
    return records


def _run_official_sb3_policy(
    *,
    source: str,
    model_zip: Path,
    vecnormalize_pkl: Path,
    episodes: int,
    max_steps: int,
    seed: int,
    early_frac: float,
) -> list[dict[str, Any]]:
    _patch_cv2_for_vendored_sb3()
    import gymnasium as gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    stats_env = DummyVecEnv([lambda: gym.make("Pendulum-v1")])
    vecnorm = VecNormalize.load(str(vecnormalize_pkl), stats_env)
    vecnorm.training = False
    vecnorm.norm_reward = False
    model = PPO.load(str(model_zip), device="cpu")
    records: list[dict[str, Any]] = []
    try:
        for ep_idx in range(int(episodes)):
            env = gym.make("Pendulum-v1")
            obs, _ = env.reset(seed=int(seed) + int(ep_idx) * 1000)
            try:
                for step_idx in range(int(max_steps)):
                    obs_before = np.asarray(obs, dtype=np.float32).copy()
                    norm_obs = vecnorm.normalize_obs(obs_before.reshape(1, -1))
                    action, _state = model.predict(norm_obs, deterministic=True)
                    action_np = np.asarray(action, dtype=np.float32).reshape(-1)
                    if source == "official_sb3_single_env_policy_early_then_zero" and not _early_active(
                        step_idx,
                        max_steps,
                        early_frac,
                    ):
                        action_np = np.zeros_like(action_np, dtype=np.float32)
                    obs, reward, terminated, truncated, _info = env.step(action_np)
                    _record_step(
                        records,
                        episode_idx=ep_idx,
                        step_idx=step_idx,
                        obs=obs_before,
                        action=action_np,
                        reward=float(reward),
                        terminated=bool(terminated),
                        truncated=bool(truncated),
                        source=source,
                    )
                    if bool(terminated or truncated):
                        break
            finally:
                env.close()
    finally:
        vecnorm.close()
    return records


def _transform_fit_action(
    mode: str,
    action: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    *,
    step_idx: int,
    max_steps: int,
    early_frac: float,
) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if mode in {"fit_policy", "fit_policy_early_then_zero"}:
        out = action
    elif mode == "fit_policy_half":
        out = 0.5 * action
    elif mode == "fit_policy_neg":
        out = -1.0 * action
    else:
        raise ValueError(f"unsupported fit source={mode!r}")
    if mode == "fit_policy_early_then_zero" and not _early_active(step_idx, max_steps, early_frac):
        out = np.zeros_like(out, dtype=np.float32)
    return np.clip(out, low, high).astype(np.float32)


def _run_fit_policy_sources(
    *,
    checkpoint: Path,
    sources: list[str],
    episodes: int,
    max_steps: int,
    context_lower_bound: int,
    seed: int,
    device: str,
    max_parallel_columns: int,
    early_frac: float,
) -> dict[str, list[dict[str, Any]]]:
    import gymnasium as gym

    model, config = load_model(str(checkpoint), device=device, verbose=False)
    state_filter = _drop_validation_value_only_state_for_action_probe(model)
    del state_filter
    config = copy.deepcopy(config)
    config["device"] = str(device)
    orch = config.setdefault("orchestration", {})
    orch["rl_validate_envs"] = "Pendulum-v1"
    orch["rl_validate_episodes"] = int(episodes)
    orch["rl_validate_max_steps"] = int(max_steps)
    orch["rl_validate_context_lower_bound"] = int(context_lower_bound)
    orch["rl_validate_seed"] = int(seed)
    orch["rl_validate_max_parallel_columns"] = int(max_parallel_columns)
    orch["rl_validate_enabled"] = True

    env_cfg = config.get("prior", {}).get("environment", {})
    layout = resolve_rlpfn_token_layout(env_cfg, num_features=config.get("prior", {}).get("num_features", None))
    obs_slot_dim = int(layout["obs_slot_dim"])
    action_slot_dim = int(layout["action_slot_dim"])
    terminal_token_enabled = bool(layout["terminal_token_enabled"])
    num_features = int(layout["num_features"])
    policy = _resolve_validation_recurrent_ppo_policy(
        model=model,
        config=config,
        env_cfg=env_cfg,
        device=device,
        num_features=num_features,
    )
    policy_step_fn = policy.make_vectorized_rollout_step_fn()

    probe_env = gym.make("Pendulum-v1")
    action_low, action_high, _used_fallback = _resolve_validation_action_bounds(probe_env.action_space)
    probe_env.close()

    out: dict[str, list[dict[str, Any]]] = {}
    for source in sources:
        states, env_batch = _build_validation_episode_states(
            gym=gym,
            env_name="Pendulum-v1",
            episodes=int(episodes),
            base_seed=int(seed),
            env_cfg=env_cfg,
            orch_cfg=orch,
            device=device,
            action_low=action_low,
            action_high=action_high,
            policy_step_enabled=True,
        )
        records: list[dict[str, Any]] = []
        try:
            while any(not bool(state["done"]) for state in states):
                active_states = []
                for state in states:
                    if bool(state["done"]):
                        continue
                    if int(state["current_rollout_len"]) >= int(max_steps):
                        active_states.append(state)
                        continue
                    active_states.append(state)
                raw_pairs = _select_ppo_policy_actions_batched(
                    states=active_states,
                    policy=policy,
                    policy_step_fn=policy_step_fn,
                    device=device,
                    obs_slot_dim=obs_slot_dim,
                    action_slot_dim=action_slot_dim,
                    terminal_token_enabled=terminal_token_enabled,
                    max_parallel_columns=max_parallel_columns,
                )
                pairs = []
                pre_records = []
                for state, raw_action in raw_pairs:
                    action = _transform_fit_action(
                        source,
                        raw_action,
                        np.asarray(state["action_low"], dtype=np.float32),
                        np.asarray(state["action_high"], dtype=np.float32),
                        step_idx=int(state["current_rollout_len"]),
                        max_steps=int(max_steps),
                        early_frac=float(early_frac),
                    )
                    should_record = bool(float(state["phase_flag"]) >= 1.0)
                    if should_record:
                        pre_records.append(
                            {
                                "state": state,
                                "episode_idx": int(state["env_idx"]),
                                "step_idx": int(state["current_rollout_len"]),
                                "obs": np.asarray(state["obs"], dtype=np.float32).copy(),
                                "action": action.copy(),
                            }
                        )
                    pairs.append((state, action))
                _advance_validation_state_action_pairs(
                    action_pairs=pairs,
                    device=device,
                    max_steps=max_steps,
                    context_lower_bound=context_lower_bound,
                )
                for item in pre_records:
                    state = item["state"]
                    _record_step(
                        records,
                        episode_idx=int(item["episode_idx"]),
                        step_idx=int(item["step_idx"]),
                        obs=np.asarray(item["obs"], dtype=np.float32),
                        action=np.asarray(item["action"], dtype=np.float32),
                        reward=float(state["prev_reward"]),
                        terminated=bool(float(state.get("prev_terminal", 0.0)) >= 1.0 and int(state["current_rollout_len"]) < int(max_steps)),
                        truncated=bool(int(state["current_rollout_len"]) >= int(max_steps)),
                        source=source,
                    )
        finally:
            closed = False
            if env_batch is not None:
                try:
                    env_batch.close()
                    closed = True
                except Exception:
                    pass
            if not closed:
                for state in states:
                    try:
                        state["env"].close()
                    except Exception:
                        pass
        out[source] = records
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-checkpoint", default=DEFAULT_FIT_CHECKPOINT)
    parser.add_argument("--official-model", default=DEFAULT_OFFICIAL_MODEL)
    parser.add_argument("--official-vecnormalize", default=DEFAULT_OFFICIAL_VECNORMALIZE)
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--context-lower-bound", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260509)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-parallel-columns", type=int, default=32)
    parser.add_argument("--early-frac", type=float, default=0.25)
    parser.add_argument(
        "--sources",
        default="zero,random,official_sb3_single_env_policy,fit_policy,fit_policy_half,fit_policy_neg",
    )
    args = parser.parse_args()

    device = str(args.device)
    if device.startswith("cuda") and not device.startswith("cuda:0"):
        raise ValueError("This probe is constrained to GPU0; pass --device cuda:0 or --device cpu.")

    sources = _parse_csv(args.sources)
    records_by_source: dict[str, list[dict[str, Any]]] = {}
    for source in sources:
        if source in {"zero", "random", "random_early_then_zero"}:
            records_by_source[source] = _run_trivial_source(
                source=source,
                episodes=int(args.episodes),
                max_steps=int(args.max_steps),
                seed=int(args.seed),
                early_frac=float(args.early_frac),
            )
        elif source in {"official_sb3_single_env_policy", "official_sb3_single_env_policy_early_then_zero"}:
            records_by_source[source] = _run_official_sb3_policy(
                source=source,
                model_zip=Path(args.official_model).expanduser().resolve(),
                vecnormalize_pkl=Path(args.official_vecnormalize).expanduser().resolve(),
                episodes=int(args.episodes),
                max_steps=int(args.max_steps),
                seed=int(args.seed),
                early_frac=float(args.early_frac),
            )

    fit_sources = [s for s in sources if s.startswith("fit_policy")]
    if fit_sources:
        records_by_source.update(
            _run_fit_policy_sources(
                checkpoint=Path(args.fit_checkpoint).expanduser().resolve(),
                sources=fit_sources,
                episodes=int(args.episodes),
                max_steps=int(args.max_steps),
                context_lower_bound=int(args.context_lower_bound),
                seed=int(args.seed),
                device=device,
                max_parallel_columns=int(args.max_parallel_columns),
                early_frac=float(args.early_frac),
            )
        )

    summaries = {source: _source_summary(records) for source, records in records_by_source.items()}
    report = {
        "schema": "phase2_gym_pendulum_mechanism_signature_probe.v1",
        "contract": {
            "exploratory_only": True,
            "gym_environment_side_raw_obs_reward_action": True,
            "ppo_algorithm_not_under_test": True,
            "prior_generator_not_mutated": True,
            "fit_model_not_mutated": True,
            "does_not_add_reward_terms": True,
            "pendulum_reward_decomposition": (
                "computed from raw obs/action as -(theta^2 + 0.1 * theta_dot^2 + 0.001 * torque^2)"
            ),
        },
        "config": {
            "fit_checkpoint": str(Path(args.fit_checkpoint).expanduser().resolve()),
            "official_model": str(Path(args.official_model).expanduser().resolve()),
            "official_vecnormalize": str(Path(args.official_vecnormalize).expanduser().resolve()),
            "episodes": int(args.episodes),
            "max_steps": int(args.max_steps),
            "context_lower_bound": int(args.context_lower_bound),
            "seed": int(args.seed),
            "device": device,
            "early_frac": float(args.early_frac),
            "sources": sources,
        },
        "summaries": summaries,
        "records_by_source": records_by_source,
    }
    output = Path(args.output_json).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    compact = {
        source: {
            "return_mean": summaries[source]["raw_episode_return"].get("mean"),
            "late_abs_theta_mean": summaries[source]["late_abs_theta"].get("mean"),
            "early_abs_action_mean": summaries[source]["early_abs_action"].get("mean"),
            "late_abs_action_mean": summaries[source]["late_abs_action"].get("mean"),
            "late_upright_frac": summaries[source]["late_upright_frac_abs_theta_lt_0_5"].get("mean"),
            "corr_action_theta_dot": summaries[source]["action_state_correlation"].get("corr_action_theta_dot"),
        }
        for source in summaries
    }
    print(json.dumps(compact, indent=2, sort_keys=True))
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
