#!/usr/bin/env python
"""Frozen fit-hidden teacher probe for Gym Pendulum official PPO actions.

Exploratory/read-only only.  This does not train fit_model, mutate PPO, mutate
the prior generator, or use zero action as a reference.

The probe freezes the fit_model checkpoint, runs the normal fit validation
context on Pendulum, collects actor hidden/latent vectors on the final eval
rollouts, and fits tiny ridge-linear teacher heads to predict the official
single-env PPO action on the same raw Gym observations.

Interpretation contract:
* If heldout hidden/latent probes predict official actions well, the fit context
  contains the information and the failure is downstream of representation
  (policy head / PPO objective / readout).
* If they fail while raw observation controls succeed, the fit representation
  has lost or failed to expose the necessary current-state/control information.
* If even raw-observation controls fail, this probe is too weak for the target
  and should not be used to close the conclusion.
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
from scripts.exploratory.phase2_gym_pendulum_bridge_intervention_probe import (  # noqa: E402
    DEFAULT_FIT_CHECKPOINT,
    DEFAULT_OFFICIAL_MODEL,
    DEFAULT_OFFICIAL_VECNORMALIZE,
    _load_official_policy,
    _official_action,
)
from scripts.exploratory.phase2_gym_pendulum_mechanism_signature_probe import (  # noqa: E402
    _json_safe,
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


DEFAULT_OUTPUT_JSON = (
    "/home/chen/RLPFN/artifacts/phase2_gym_pendulum_frozen_hidden_teacher_probe_0509/"
    "fit_epoch29_hidden_teacher_e32.json"
)


def _corr(a: np.ndarray, b: np.ndarray) -> float | None:
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


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    mask = np.isfinite(yt) & np.isfinite(yp)
    if int(mask.sum()) < 3:
        return None
    yt = yt[mask]
    yp = yp[mask]
    denom = float(np.sum(np.square(yt - float(np.mean(yt)))))
    if denom <= 1e-12:
        return None
    return float(1.0 - (float(np.sum(np.square(yt - yp))) / denom))


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    err = yp - yt
    return {
        "count": int(yt.size),
        "target_abs": _stats(np.abs(yt).tolist()),
        "pred_abs": _stats(np.abs(yp).tolist()),
        "error_abs": _stats(np.abs(err).tolist()),
        "mse": float(np.mean(np.square(err))) if yt.size else None,
        "mae": float(np.mean(np.abs(err))) if yt.size else None,
        "corr": _corr(yt, yp),
        "r2": _r2(yt, yp),
        "sign_match_rate": (
            float(np.mean(np.sign(yt) == np.sign(yp))) if yt.size else None
        ),
        "low_pred_high_teacher_rate": (
            float(np.mean((np.abs(yp) < 0.1) & (np.abs(yt) > 0.5))) if yt.size else None
        ),
    }


def _standardize_fit_ridge(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    ridge: float,
) -> dict[str, np.ndarray]:
    x = np.asarray(x_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64).reshape(-1, 1)
    mean = np.mean(x, axis=0, keepdims=True)
    std = np.std(x, axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    xs = (x - mean) / std
    xa = np.concatenate([xs, np.ones((xs.shape[0], 1), dtype=np.float64)], axis=1)
    reg = np.eye(xa.shape[1], dtype=np.float64) * float(ridge)
    reg[-1, -1] = 0.0
    coef = np.linalg.solve(xa.T @ xa + reg, xa.T @ y)
    return {"mean": mean, "std": std, "coef": coef}


def _ridge_predict(model: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    xx = np.asarray(x, dtype=np.float64)
    xs = (xx - model["mean"]) / model["std"]
    xa = np.concatenate([xs, np.ones((xs.shape[0], 1), dtype=np.float64)], axis=1)
    return (xa @ model["coef"]).reshape(-1)


def _fit_eval_probe(
    name: str,
    x: np.ndarray,
    y: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    *,
    ridge: float,
    action_low: float = -2.0,
    action_high: float = 2.0,
) -> dict[str, Any]:
    model = _standardize_fit_ridge(x[train_mask], y[train_mask], ridge=float(ridge))
    pred_train = _ridge_predict(model, x[train_mask])
    pred_test = _ridge_predict(model, x[test_mask])
    pred_train_clipped = np.clip(pred_train, float(action_low), float(action_high))
    pred_test_clipped = np.clip(pred_test, float(action_low), float(action_high))
    return {
        "name": name,
        "feature_dim": int(x.shape[1]),
        "ridge": float(ridge),
        "train": _metrics(y[train_mask], pred_train),
        "heldout": _metrics(y[test_mask], pred_test),
        "train_action_clipped": _metrics(y[train_mask], pred_train_clipped),
        "heldout_action_clipped": _metrics(y[test_mask], pred_test_clipped),
        "coef_l2": float(np.linalg.norm(model["coef"][:-1])),
        "bias": float(model["coef"][-1, 0]),
    }


def _pendulum_obs_features(obs: np.ndarray) -> np.ndarray:
    obs_arr = np.asarray(obs, dtype=np.float32).reshape(-1)
    cos_theta = float(np.clip(obs_arr[0], -1.0, 1.0)) if obs_arr.size > 0 else 0.0
    sin_theta = float(np.clip(obs_arr[1], -1.0, 1.0)) if obs_arr.size > 1 else 0.0
    theta_dot = float(obs_arr[2]) if obs_arr.size > 2 else 0.0
    return np.asarray(
        [
            cos_theta,
            sin_theta,
            theta_dot,
            theta_dot * cos_theta,
            theta_dot * sin_theta,
        ],
        dtype=np.float32,
    )


def _select_fit_hidden_actions(
    *,
    states: list[dict[str, Any]],
    policy: Any,
    policy_step_fn: Any,
    device: str,
    obs_slot_dim: int,
    action_slot_dim: int,
    terminal_token_enabled: bool,
    max_parallel_columns: int,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    if not states:
        return []
    batch_cap = int(max(1, max_parallel_columns))
    out: list[tuple[dict[str, Any], dict[str, Any]]] = []
    sample_fn = getattr(policy_step_fn, "_policy_actor_sample_fn", None)
    if not callable(sample_fn):
        raise RuntimeError("PPO validation policy step function must expose _policy_actor_sample_fn.")

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
                deterministic_mask[row_idx] = bool(float(state["phase_flag"]) >= 1.0)
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
            obs_full = policy._build_obs_from_rollout_step_inputs(
                obs_t,
                action_t,
                reward_t,
                reward_mask_t,
                env_info,
            )
            action_dims_t = torch.as_tensor(action_dims, device=device, dtype=torch.long)
            starts = torch.zeros((len(chunk_states),), device=device, dtype=torch.float32)

            actor_cache = cache_in
            value_cache = None
            if bool(getattr(policy, "separate_value_backbone", False)) and isinstance(cache_in, dict):
                actor_cache = cache_in.get("actor")
                value_cache = cache_in.get("value")
            if actor_cache is None and bool(getattr(policy, "separate_value_backbone", False)):
                value_cache = None

            hidden, next_actor_cache = policy._official_rollout_hidden_from_cache(
                obs_full,
                starts,
                cache=actor_cache,
            )
            latent_pi = policy.mlp_extractor.forward_actor(hidden)
            if bool(getattr(policy, "separate_value_backbone", False)):
                with policy._value_backbone_context():
                    _value_hidden, next_value_cache = policy._official_rollout_hidden_from_cache(
                        obs_full,
                        starts,
                        cache=value_cache,
                    )
                cache_next = {"actor": next_actor_cache, "value": next_value_cache}
            else:
                cache_next = next_actor_cache

            distribution = policy._get_action_dist_from_latent(latent_pi)
            action_mean_full = distribution.distribution.mean
            action_std_full = distribution.distribution.stddev
            max_action_dim_out = int(max(1, min(int(action_mean_full.shape[-1]), int(action_dims_t.max().item()))))
            action_mean = action_mean_full[:, :max_action_dim_out].clone()
            action_std = action_std_full[:, :max_action_dim_out].clone()
            action_positions = torch.arange(max_action_dim_out, device=device, dtype=torch.long).unsqueeze(0)
            action_valid = action_positions < action_dims_t.unsqueeze(1)
            action_mean.masked_fill_(~action_valid, 0.0)
            action_std.masked_fill_(~action_valid, 0.0)

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
                        "action_mean": action_mean.index_select(0, explore_rows),
                        "action_std": action_std.index_select(0, explore_rows),
                    }
                    action_env.index_copy_(0, explore_rows, sample_fn(explore_outputs, noise=noise_t))

            for row_idx, state in enumerate(chunk_states):
                action_dim = int(state["policy_action_dim"])
                action_np = action_env[row_idx, :action_dim].detach().to(dtype=torch.float32).cpu().numpy()
                action_np = np.clip(
                    action_np,
                    state["action_low"][:action_dim],
                    state["action_high"][:action_dim],
                ).astype(np.float32)
                state["policy_cache"] = _slice_cache_batch_dim(cache_next, row_idx)
                out.append(
                    (
                        state,
                        {
                            "action": action_np,
                            "fit_action_mean": action_mean[row_idx, :action_dim]
                            .detach()
                            .to(dtype=torch.float32)
                            .cpu()
                            .numpy(),
                            "hidden": hidden[row_idx].detach().to(dtype=torch.float32).cpu().numpy(),
                            "latent_pi": latent_pi[row_idx].detach().to(dtype=torch.float32).cpu().numpy(),
                        },
                    )
                )
    return out


def _collect_dataset(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    import gymnasium as gym

    device = str(args.device)
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
    policy = _resolve_validation_recurrent_ppo_policy(
        model=fit_model,
        config=config,
        env_cfg=env_cfg,
        device=device,
        num_features=num_features,
    )
    policy_step_fn = policy.make_vectorized_rollout_step_fn()

    probe_env = gym.make("Pendulum-v1")
    action_low, action_high, _used_fallback = _resolve_validation_action_bounds(probe_env.action_space)
    probe_env.close()
    states, env_batch = _build_validation_episode_states(
        gym=gym,
        env_name="Pendulum-v1",
        episodes=int(args.episodes),
        base_seed=int(args.seed),
        env_cfg=env_cfg,
        orch_cfg=orch,
        device=device,
        action_low=action_low,
        action_high=action_high,
        policy_step_enabled=True,
    )

    rows: list[dict[str, Any]] = []
    hidden_rows: list[np.ndarray] = []
    latent_rows: list[np.ndarray] = []
    obs_rows: list[np.ndarray] = []
    obs_feature_rows: list[np.ndarray] = []
    target_rows: list[float] = []
    fit_mean_rows: list[float] = []
    episode_rows: list[int] = []
    try:
        while any(not bool(state["done"]) for state in states):
            active_states = []
            for state in states:
                if bool(state["done"]):
                    continue
                if int(state["current_rollout_len"]) >= int(args.max_steps):
                    _finalize_validation_rollout(state, int(args.context_lower_bound))
                    continue
                active_states.append(state)
            selected = _select_fit_hidden_actions(
                states=active_states,
                policy=policy,
                policy_step_fn=policy_step_fn,
                device=device,
                obs_slot_dim=obs_slot_dim,
                action_slot_dim=action_slot_dim,
                terminal_token_enabled=terminal_token_enabled,
                max_parallel_columns=int(args.max_parallel_columns),
            )
            pairs = []
            for state, payload in selected:
                obs = np.asarray(state["obs"], dtype=np.float32).reshape(-1).copy()
                official = _official_action(official_model, official_vecnorm, obs)
                is_eval = bool(float(state["phase_flag"]) >= 1.0)
                if is_eval:
                    hidden_rows.append(np.asarray(payload["hidden"], dtype=np.float32).reshape(-1))
                    latent_rows.append(np.asarray(payload["latent_pi"], dtype=np.float32).reshape(-1))
                    obs_rows.append(obs)
                    obs_feature_rows.append(_pendulum_obs_features(obs))
                    target_rows.append(float(official.reshape(-1)[0]))
                    fit_mean_rows.append(float(np.asarray(payload["fit_action_mean"]).reshape(-1)[0]))
                    episode_rows.append(int(state["env_idx"]))
                    rows.append(
                        {
                            "episode_idx": int(state["env_idx"]),
                            "rollout_step": int(state["current_rollout_len"]),
                            "official_action": float(official.reshape(-1)[0]),
                            "fit_action_mean": float(np.asarray(payload["fit_action_mean"]).reshape(-1)[0]),
                        }
                    )
                pairs.append((state, np.asarray(payload["action"], dtype=np.float32).reshape(-1)))
            _advance_validation_state_action_pairs(
                action_pairs=pairs,
                device=device,
                max_steps=int(args.max_steps),
                context_lower_bound=int(args.context_lower_bound),
            )
    finally:
        try:
            official_vecnorm.close()
        except Exception:
            pass
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

    metadata = {
        "state_filter": state_filter,
        "hidden_dim": int(hidden_rows[0].shape[0]) if hidden_rows else 0,
        "latent_pi_dim": int(latent_rows[0].shape[0]) if latent_rows else 0,
        "samples": int(len(target_rows)),
        "episodes": int(args.episodes),
    }
    arrays = {
        "hidden": np.stack(hidden_rows, axis=0).astype(np.float32),
        "latent_pi": np.stack(latent_rows, axis=0).astype(np.float32),
        "obs": np.stack(obs_rows, axis=0).astype(np.float32),
        "obs_features": np.stack(obs_feature_rows, axis=0).astype(np.float32),
        "official_action": np.asarray(target_rows, dtype=np.float32),
        "fit_action_mean": np.asarray(fit_mean_rows, dtype=np.float32),
        "episode_idx": np.asarray(episode_rows, dtype=np.int64),
    }
    metadata["record_sample"] = rows[: int(min(len(rows), 16))]
    return metadata, arrays


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-checkpoint", default=DEFAULT_FIT_CHECKPOINT)
    parser.add_argument("--official-model", default=DEFAULT_OFFICIAL_MODEL)
    parser.add_argument("--official-vecnormalize", default=DEFAULT_OFFICIAL_VECNORMALIZE)
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--context-lower-bound", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260509)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-parallel-columns", type=int, default=32)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--ridge-values", default=None)
    args = parser.parse_args()

    device = str(args.device)
    if device.startswith("cuda") and not device.startswith("cuda:0"):
        raise ValueError("This probe is constrained to GPU0; pass --device cuda:0 or --device cpu.")

    metadata, arrays = _collect_dataset(args)
    episodes = arrays["episode_idx"]
    unique_eps = np.unique(episodes)
    train_eps = set(int(x) for x in unique_eps[::2].tolist())
    test_eps = set(int(x) for x in unique_eps[1::2].tolist())
    train_mask = np.asarray([int(ep) in train_eps for ep in episodes], dtype=bool)
    test_mask = np.asarray([int(ep) in test_eps for ep in episodes], dtype=bool)
    if int(train_mask.sum()) <= 0 or int(test_mask.sum()) <= 0:
        raise RuntimeError("Need non-empty train and heldout episode splits.")

    y = arrays["official_action"].astype(np.float64)
    fit_mean = arrays["fit_action_mean"].astype(np.float64)
    if args.ridge_values is None:
        ridge_values = [float(args.ridge)]
    else:
        ridge_values = [
            float(part.strip())
            for part in str(args.ridge_values).split(",")
            if part.strip()
        ]
    probes = []
    for ridge_value in ridge_values:
        probes.extend(
            [
                _fit_eval_probe("hidden_ridge", arrays["hidden"], y, train_mask, test_mask, ridge=float(ridge_value)),
                _fit_eval_probe(
                    "latent_pi_ridge",
                    arrays["latent_pi"],
                    y,
                    train_mask,
                    test_mask,
                    ridge=float(ridge_value),
                ),
                _fit_eval_probe("raw_obs_ridge", arrays["obs"], y, train_mask, test_mask, ridge=float(ridge_value)),
                _fit_eval_probe(
                    "pendulum_obs_features_ridge",
                    arrays["obs_features"],
                    y,
                    train_mask,
                    test_mask,
                    ridge=float(ridge_value),
                ),
            ]
        )
    baseline = {
        "fit_action_mean_as_predictor": {
            "train": _metrics(y[train_mask], fit_mean[train_mask]),
            "heldout": _metrics(y[test_mask], fit_mean[test_mask]),
        },
        "constant_train_mean_as_predictor": {
            "train": _metrics(
                y[train_mask],
                np.full((int(train_mask.sum()),), float(np.mean(y[train_mask])), dtype=np.float64),
            ),
            "heldout": _metrics(
                y[test_mask],
                np.full((int(test_mask.sum()),), float(np.mean(y[train_mask])), dtype=np.float64),
            ),
        },
    }

    report = {
        "schema": "phase2_gym_pendulum_frozen_hidden_teacher_probe.v1",
        "contract": {
            "exploratory_only": True,
            "no_fit_model_training": True,
            "no_ppo_mutation": True,
            "no_prior_generator_mutation": True,
            "zero_action_not_reference": True,
            "episode_heldout_split": True,
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
            "ridge": float(args.ridge),
            "ridge_values": ridge_values,
            "train_episodes": sorted(train_eps),
            "heldout_episodes": sorted(test_eps),
        },
        "metadata": metadata,
        "target": {
            "official_action": _stats(y.tolist()),
            "fit_action_mean": _stats(fit_mean.tolist()),
        },
        "baseline": baseline,
        "probes": probes,
    }
    output = Path(args.output_json).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output_json": str(output),
                "samples": int(len(y)),
                "heldout": {f"{p['name']}_ridge{p['ridge']}": p["heldout"] for p in probes},
                "heldout_action_clipped": {
                    f"{p['name']}_ridge{p['ridge']}": p["heldout_action_clipped"] for p in probes
                },
                "fit_mean_heldout": baseline["fit_action_mean_as_predictor"]["heldout"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
