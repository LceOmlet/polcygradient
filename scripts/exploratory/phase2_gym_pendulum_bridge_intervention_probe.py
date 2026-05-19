#!/usr/bin/env python
"""Pendulum fit-vs-official bridge audit and action intervention probe.

Exploratory/read-only only.  This script does not train, mutate PPO, mutate the
prior generator, or use zero action as a reference.  It is designed to falsify
the nearest-failure hypothesis from ``phase2_gym_pendulum_official_fit_action_alignment``:

1. Bridge/scale falsification:
   Is the fit_model action already tiny in the PPO actor mean before Gym-bound
   clipping, or is it made tiny by validation bridge/clip/action-slot plumbing?

2. Action intervention falsification:
   If the final eval rollout uses official single-env PPO actions on the same
   Gym observations, does the return recover under the same validation context
   construction?  This tests whether the proximal action mapping is a necessary
   failure point, without claiming it is the final root cause.
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
from scripts.exploratory.phase2_gym_pendulum_mechanism_signature_probe import (  # noqa: E402
    _json_safe,
    _patch_cv2_for_vendored_sb3,
    _stats,
)
from ticl.model_builder import load_model  # noqa: E402
from ticl.rl_validation import (  # noqa: E402
    _advance_validation_state_action_pairs,
    _build_validation_episode_states,
    _concat_cache_batch_dim,
    _finalize_validation_rollout,
    _resolve_validation_action_bounds,
    _resolve_validation_recurrent_ppo_policy,
    _slice_cache_batch_dim,
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
    "/home/chen/RLPFN/artifacts/phase2_gym_pendulum_bridge_intervention_probe_0509/"
    "fit_epoch29_bridge_intervention.json"
)


def _corr(a: list[float], b: list[float]) -> float | None:
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    mask = np.isfinite(aa) & np.isfinite(bb)
    if int(mask.sum()) < 3:
        return None
    aa = aa[mask]
    bb = bb[mask]
    if float(np.std(aa)) <= 1e-12 or float(np.std(bb)) <= 1e-12:
        return None
    return float(np.corrcoef(aa, bb)[0, 1])


def _load_official_policy(*, model_zip: Path, vecnormalize_pkl: Path):
    _patch_cv2_for_vendored_sb3()
    import gymnasium as gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    stats_env = DummyVecEnv([lambda: gym.make("Pendulum-v1")])
    vecnorm = VecNormalize.load(str(vecnormalize_pkl), stats_env)
    vecnorm.training = False
    vecnorm.norm_reward = False
    model = PPO.load(str(model_zip), device="cpu")
    return model, vecnorm


def _official_action(model: Any, vecnorm: Any, obs: np.ndarray) -> np.ndarray:
    norm_obs = vecnorm.normalize_obs(np.asarray(obs, dtype=np.float32).reshape(1, -1))
    action, _state = model.predict(norm_obs, deterministic=True)
    return np.asarray(action, dtype=np.float32).reshape(-1)


def _select_fit_actions_with_audit(
    *,
    states: list[dict[str, Any]],
    policy_step_fn: Any,
    device: str,
    obs_slot_dim: int,
    action_slot_dim: int,
    terminal_token_enabled: bool,
    max_parallel_columns: int,
    force_stochastic_eval: bool = False,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    if not states:
        return []

    sample_fn = getattr(policy_step_fn, "_policy_actor_sample_fn", None)
    if not callable(sample_fn):
        raise RuntimeError("PPO validation policy step function must expose _policy_actor_sample_fn.")

    out: list[tuple[dict[str, Any], dict[str, Any]]] = []
    batch_cap = int(max(1, max_parallel_columns))
    with torch.no_grad():
        for chunk_start in range(0, len(states), batch_cap):
            chunk_states = states[chunk_start : chunk_start + batch_cap]
            obs_dims = [int(state["policy_obs_dim"]) for state in chunk_states]
            action_dims = [int(state["policy_action_dim"]) for state in chunk_states]
            max_obs_dim = max(obs_dims)
            max_action_dim = max(action_dims)

            obs_t = torch.zeros((len(chunk_states), max_obs_dim), device=device, dtype=torch.float32)
            action_t = torch.zeros((len(chunk_states), max_action_dim), device=device, dtype=torch.float32)
            reward_t = torch.empty((len(chunk_states), 1), device=device, dtype=torch.float32)
            reward_mask_t = torch.empty((len(chunk_states), 1), device=device, dtype=torch.float32)
            phase_t = torch.empty((len(chunk_states), 1), device=device, dtype=torch.float32)
            terminal_t = (
                torch.empty((len(chunk_states), 1), device=device, dtype=torch.float32)
                if bool(terminal_token_enabled)
                else None
            )
            deterministic_mask = torch.empty((len(chunk_states),), device=device, dtype=torch.bool)

            cache_rows = []
            any_cache = False
            for row_idx, state in enumerate(chunk_states):
                obs_row = state["policy_obs_t"]
                if obs_row is None:
                    obs_arr = np.asarray(state["obs"], dtype=np.float32).reshape(-1)
                    obs_row = torch.from_numpy(obs_arr).to(device=device, dtype=torch.float32)
                    state["policy_obs_t"] = obs_row
                    state["policy_obs_dim"] = int(obs_arr.shape[0])
                obs_t[row_idx, : int(state["policy_obs_dim"])] = obs_row[: int(state["policy_obs_dim"])]
                prev_action_t = state["policy_action_t"].to(device=device, dtype=torch.float32).reshape(1, -1)
                action_t[row_idx, : int(prev_action_t.shape[-1])] = prev_action_t[0]
                reward_t[row_idx] = state["policy_reward_t"].to(device=device, dtype=torch.float32)[0]
                reward_mask_t[row_idx] = state["policy_reward_mask_t"].to(device=device, dtype=torch.float32)[0]
                phase_t[row_idx, 0] = float(state["phase_flag"])
                deterministic_mask[row_idx] = bool(float(state["phase_flag"]) >= 1.0) and not bool(
                    force_stochastic_eval
                )
                if terminal_t is not None:
                    terminal_t[row_idx] = state["policy_terminal_t"].to(device=device, dtype=torch.float32)[0]
                cache_rows.append(state["policy_cache"])
                any_cache = any_cache or (state["policy_cache"] is not None)

            env_info = {
                "obs_slot_dim": torch.full(
                    (len(chunk_states),), int(obs_slot_dim), device=device, dtype=torch.long
                ),
                "obs_dim": torch.as_tensor(obs_dims, device=device, dtype=torch.long),
                "action_slot_dim": torch.full(
                    (len(chunk_states),), int(action_slot_dim), device=device, dtype=torch.long
                ),
                "action_dim_per_sample": torch.as_tensor(action_dims, device=device, dtype=torch.long),
                "terminal_reset_enabled": torch.full(
                    (len(chunk_states),), bool(terminal_token_enabled), device=device, dtype=torch.bool
                ),
                "phase_t": phase_t,
            }
            if terminal_t is not None:
                env_info["terminal_t"] = terminal_t

            cache_in = _concat_cache_batch_dim(cache_rows) if any_cache else None
            actor_outputs, cache_next = policy_step_fn(obs_t, action_t, reward_t, reward_mask_t, cache_in, 0, env_info)
            action_mean = actor_outputs["action_mean"]
            action_std = actor_outputs.get("action_std", torch.zeros_like(action_mean))
            if bool(torch.all(deterministic_mask).item()):
                action_env = action_mean
            else:
                action_env = action_mean.clone()
                explore_rows = (~deterministic_mask).nonzero(as_tuple=False).reshape(-1)
                if int(explore_rows.numel()) > 0:
                    noise_t = torch.from_numpy(
                        np.stack(
                            [
                                chunk_states[int(row_idx)]["rng"].normal(
                                    size=(int(action_mean.shape[-1]),)
                                ).astype(np.float32)
                                for row_idx in explore_rows.detach().cpu().tolist()
                            ],
                            axis=0,
                        )
                    ).to(device=device, dtype=action_mean.dtype)
                    explore_outputs = {
                        "action_mean": actor_outputs["action_mean"].index_select(0, explore_rows),
                        "action_std": actor_outputs["action_std"].index_select(0, explore_rows),
                    }
                    action_env.index_copy_(0, explore_rows, sample_fn(explore_outputs, noise=noise_t))

            for row_idx, state in enumerate(chunk_states):
                action_dim = int(state["policy_action_dim"])
                mean_np = action_mean[row_idx, :action_dim].detach().to(dtype=torch.float32).cpu().numpy()
                std_np = action_std[row_idx, :action_dim].detach().to(dtype=torch.float32).cpu().numpy()
                preclip_np = action_env[row_idx, :action_dim].detach().to(dtype=torch.float32).cpu().numpy()
                clipped_np = np.clip(
                    preclip_np,
                    state["action_low"][:action_dim],
                    state["action_high"][:action_dim],
                ).astype(np.float32)
                state["policy_cache"] = _slice_cache_batch_dim(cache_next, row_idx)
                out.append(
                    (
                        state,
                        {
                            "fit_action_mean": mean_np,
                            "fit_action_std": std_np,
                            "fit_action_preclip": preclip_np.astype(np.float32),
                            "fit_action_clipped": clipped_np,
                        },
                    )
                )
    return out


def _summarize_records(records: list[dict[str, Any]], *, early_frac: float) -> dict[str, Any]:
    eval_records = [r for r in records if bool(r["is_eval"])]
    early_records = [r for r in eval_records if float(r["eval_step_frac"]) < float(early_frac)]

    def values(seq: list[dict[str, Any]], key: str) -> list[float]:
        return [float(r[key]) for r in seq if key in r and math.isfinite(float(r[key]))]

    def one(seq: list[dict[str, Any]]) -> dict[str, Any]:
        fit_mean = values(seq, "fit_action_mean")
        fit_preclip = values(seq, "fit_action_preclip")
        fit_clipped = values(seq, "fit_action_clipped")
        official = values(seq, "official_action")
        applied = values(seq, "applied_action")
        theta_dot = values(seq, "theta_dot")
        sin_theta = values(seq, "sin_theta")
        cos_theta = values(seq, "cos_theta")
        official_abs = [abs(x) for x in official]
        fit_mean_abs = [abs(x) for x in fit_mean]
        fit_preclip_abs = [abs(x) for x in fit_preclip]
        fit_clipped_abs = [abs(x) for x in fit_clipped]
        return {
            "count": int(len(seq)),
            "fit_action_mean_abs": _stats(fit_mean_abs),
            "fit_action_preclip_abs": _stats(fit_preclip_abs),
            "fit_action_clipped_abs": _stats(fit_clipped_abs),
            "official_action_abs": _stats(official_abs),
            "applied_action_abs": _stats([abs(x) for x in applied]),
            "fit_preclip_to_clipped_abs_diff": _stats(
                [abs(a - b) for a, b in zip(fit_preclip_abs, fit_clipped_abs)]
            ),
            "fit_mean_to_official_abs_ratio_mean": (
                None
                if not official_abs
                else float(np.mean(np.asarray(fit_mean_abs) / np.maximum(np.asarray(official_abs), 1e-6)))
            ),
            "low_fit_mean_high_official_rate": (
                None
                if not official_abs
                else float(np.mean((np.asarray(fit_mean_abs) < 0.1) & (np.asarray(official_abs) > 0.5)))
            ),
            "corr_fit_mean_official_action": _corr(fit_mean, official),
            "corr_fit_mean_theta_dot": _corr(fit_mean, theta_dot),
            "corr_official_theta_dot": _corr(official, theta_dot),
            "corr_fit_mean_theta_dot_cos_theta": _corr(
                fit_mean, [float(td) * float(c) for td, c in zip(theta_dot, cos_theta)]
            ),
            "corr_official_theta_dot_cos_theta": _corr(
                official, [float(td) * float(c) for td, c in zip(theta_dot, cos_theta)]
            ),
            "corr_fit_mean_sin_theta": _corr(fit_mean, sin_theta),
            "corr_official_sin_theta": _corr(official, sin_theta),
        }

    return {"eval": one(eval_records), "early_eval": one(early_records)}


def _pendulum_state_terms(obs: np.ndarray) -> dict[str, float]:
    obs_arr = np.asarray(obs, dtype=np.float32).reshape(-1)
    cos_theta = float(np.clip(obs_arr[0], -1.0, 1.0)) if obs_arr.size > 0 else 0.0
    sin_theta = float(np.clip(obs_arr[1], -1.0, 1.0)) if obs_arr.size > 1 else 0.0
    theta = float(np.arctan2(sin_theta, cos_theta))
    theta_dot = float(obs_arr[2]) if obs_arr.size > 2 else 0.0
    return {
        "cos_theta": cos_theta,
        "sin_theta": sin_theta,
        "theta": theta,
        "abs_theta": abs(theta),
        "theta_dot": theta_dot,
    }


def _run_mode(
    *,
    mode: str,
    fit_policy_step_fn: Any,
    official_model: Any,
    official_vecnorm: Any,
    gym: Any,
    env_cfg: dict[str, Any],
    orch: dict[str, Any],
    device: str,
    obs_slot_dim: int,
    action_slot_dim: int,
    terminal_token_enabled: bool,
    episodes: int,
    max_steps: int,
    context_lower_bound: int,
    seed: int,
    max_parallel_columns: int,
    official_prefix_frac: float,
) -> dict[str, Any]:
    probe_env = gym.make("Pendulum-v1")
    action_low, action_high, _used_fallback = _resolve_validation_action_bounds(probe_env.action_space)
    probe_env.close()
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
                    _finalize_validation_rollout(state, context_lower_bound)
                    continue
                active_states.append(state)
            audited = _select_fit_actions_with_audit(
                states=active_states,
                policy_step_fn=fit_policy_step_fn,
                device=device,
                obs_slot_dim=obs_slot_dim,
                action_slot_dim=action_slot_dim,
                terminal_token_enabled=terminal_token_enabled,
                max_parallel_columns=max_parallel_columns,
                force_stochastic_eval=(mode == "fit_stochastic_eval"),
            )
            pairs = []
            for state, audit in audited:
                obs = np.asarray(state["obs"], dtype=np.float32).reshape(-1).copy()
                official = _official_action(official_model, official_vecnorm, obs)
                fit_action = np.asarray(audit["fit_action_clipped"], dtype=np.float32).reshape(-1)
                is_eval = bool(float(state["phase_flag"]) >= 1.0)
                eval_step = int(state["current_rollout_len"])
                if mode == "fit":
                    applied = fit_action
                elif mode == "fit_stochastic_eval":
                    applied = fit_action
                elif mode == "official_eval_full":
                    applied = official if is_eval else fit_action
                elif mode == "official_eval_prefix_then_fit":
                    use_official = is_eval and (float(eval_step) / float(max(1, max_steps)) < float(official_prefix_frac))
                    applied = official if use_official else fit_action
                else:
                    raise ValueError(f"unsupported mode={mode!r}")
                applied = np.clip(
                    np.asarray(applied, dtype=np.float32).reshape(-1),
                    state["action_low"],
                    state["action_high"],
                ).astype(np.float32)
                terms = _pendulum_state_terms(obs)
                records.append(
                    {
                        "mode": mode,
                        "episode_idx": int(state["env_idx"]),
                        "rollout_step": int(eval_step),
                        "eval_step_frac": float(eval_step) / float(max(1, max_steps)),
                        "phase_flag": float(state["phase_flag"]),
                        "is_eval": is_eval,
                        "fit_action_mean": float(np.asarray(audit["fit_action_mean"]).reshape(-1)[0]),
                        "fit_action_std": float(np.asarray(audit["fit_action_std"]).reshape(-1)[0]),
                        "fit_action_preclip": float(np.asarray(audit["fit_action_preclip"]).reshape(-1)[0]),
                        "fit_action_clipped": float(fit_action.reshape(-1)[0]),
                        "official_action": float(official.reshape(-1)[0]),
                        "applied_action": float(applied.reshape(-1)[0]),
                        **terms,
                    }
                )
                pairs.append((state, applied))
            _advance_validation_state_action_pairs(
                action_pairs=pairs,
                device=device,
                max_steps=int(max_steps),
                context_lower_bound=int(context_lower_bound),
            )
    finally:
        if env_batch is not None:
            try:
                env_batch.close()
            except Exception:
                pass
        else:
            for state in states:
                try:
                    state["env"].close()
                except Exception:
                    pass

    returns = [float(state["reported_return"]) for state in states if state.get("reported_return") is not None]
    lengths = [float(state["reported_len"]) for state in states if state.get("reported_len") is not None]
    return {
        "return": _stats(returns),
        "return_mean": float(np.mean(returns)) if returns else None,
        "len": _stats(lengths),
        "len_mean": float(np.mean(lengths)) if lengths else None,
        "summary": _summarize_records(records, early_frac=float(official_prefix_frac)),
        "records": records,
    }


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
    parser.add_argument("--official-prefix-frac", type=float, default=0.25)
    parser.add_argument(
        "--modes",
        default="fit,official_eval_prefix_then_fit,official_eval_full",
        help=(
            "Comma-separated modes: fit, fit_stochastic_eval, "
            "official_eval_prefix_then_fit, official_eval_full."
        ),
    )
    args = parser.parse_args()

    device = str(args.device)
    if device.startswith("cuda") and not device.startswith("cuda:0"):
        raise ValueError("This probe is constrained to GPU0; pass --device cuda:0 or --device cpu.")

    import gymnasium as gym

    official_model, official_vecnorm = _load_official_policy(
        model_zip=Path(args.official_model).expanduser().resolve(),
        vecnormalize_pkl=Path(args.official_vecnormalize).expanduser().resolve(),
    )
    fit_model, config = load_model(str(Path(args.fit_checkpoint).expanduser().resolve()), device=device, verbose=False)
    state_filter = _drop_validation_value_only_state_for_action_probe(fit_model)
    config = copy.deepcopy(config)
    config["device"] = device
    orch = config.setdefault("orchestration", {})
    orch["rl_validate_envs"] = "Pendulum-v1"
    orch["rl_validate_episodes"] = int(args.episodes)
    orch["rl_validate_max_steps"] = int(args.max_steps)
    orch["rl_validate_context_lower_bound"] = int(args.context_lower_bound)
    orch["rl_validate_seed"] = int(args.seed)
    orch["rl_validate_max_parallel_columns"] = int(args.max_parallel_columns)
    orch["rl_validate_enabled"] = True

    env_cfg = config.get("prior", {}).get("environment", {})
    layout = resolve_rlpfn_token_layout(env_cfg, num_features=config.get("prior", {}).get("num_features", None))
    obs_slot_dim = int(layout["obs_slot_dim"])
    action_slot_dim = int(layout["action_slot_dim"])
    terminal_token_enabled = bool(layout["terminal_token_enabled"])
    num_features = int(layout["num_features"])
    fit_policy = _resolve_validation_recurrent_ppo_policy(
        model=fit_model,
        config=config,
        env_cfg=env_cfg,
        device=device,
        num_features=num_features,
    )
    fit_policy_step_fn = fit_policy.make_vectorized_rollout_step_fn()

    modes = [part.strip() for part in str(args.modes).split(",") if part.strip()]
    mode_results: dict[str, Any] = {}
    try:
        for mode in modes:
            mode_results[mode] = _run_mode(
                mode=mode,
                fit_policy_step_fn=fit_policy_step_fn,
                official_model=official_model,
                official_vecnorm=official_vecnorm,
                gym=gym,
                env_cfg=env_cfg,
                orch=orch,
                device=device,
                obs_slot_dim=obs_slot_dim,
                action_slot_dim=action_slot_dim,
                terminal_token_enabled=terminal_token_enabled,
                episodes=int(args.episodes),
                max_steps=int(args.max_steps),
                context_lower_bound=int(args.context_lower_bound),
                seed=int(args.seed),
                max_parallel_columns=int(args.max_parallel_columns),
                official_prefix_frac=float(args.official_prefix_frac),
            )
            print(
                json.dumps(
                    {
                        "mode": mode,
                        "return_mean": mode_results[mode].get("return_mean"),
                        "early_bridge": mode_results[mode].get("summary", {}).get("early_eval", {}),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        try:
            official_vecnorm.close()
        except Exception:
            pass

    report = {
        "schema": "phase2_gym_pendulum_bridge_intervention_probe.v1",
        "contract": {
            "exploratory_only": True,
            "no_training": True,
            "zero_action_not_reference": True,
            "bridge_audit_before_intervention": True,
            "official_reference": "SB3 PPO single-env Pendulum final_model.zip with saved VecNormalize obs stats",
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
            "official_prefix_frac": float(args.official_prefix_frac),
            "modes": modes,
            "state_filter": state_filter,
        },
        "mode_results": mode_results,
    }
    output = Path(args.output_json).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output_json": str(output), "modes": modes}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
