#!/usr/bin/env python
"""Closed-loop rollout intervention with a frozen-hidden official-action head.

Exploratory/read-only only.  This does not train fit_model, mutate PPO, mutate
the prior generator, or use zero action as a reference.

Purpose:
    Further localize why the fit_model checkpoint did not generalize to
    Pendulum.  We train a tiny ridge-linear head on frozen fit hidden states
    from one set of validation seeds to imitate the official single-env PPO
    action, then run a disjoint-seed closed-loop validation rollout where the
    final eval actions are taken from that ridge head instead of the saved PPO
    action head.

Interpretation:
    * If ridge-head rollout recovers return, the hidden representation contains
      usable Pendulum control information and the failure is pushed toward
      action head / readout / PPO objective generalization.
    * If ridge-head rollout does not recover return, the offline hidden teacher
      fit is not sufficient to explain closed-loop generalization failure.
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

from scripts.exploratory.phase2_gym_pendulum_bridge_intervention_probe import (  # noqa: E402
    DEFAULT_FIT_CHECKPOINT,
    DEFAULT_OFFICIAL_MODEL,
    DEFAULT_OFFICIAL_VECNORMALIZE,
    _load_official_policy,
    _official_action,
)
from scripts.exploratory.phase2_gym_pendulum_frozen_hidden_teacher_probe import (  # noqa: E402
    _collect_dataset,
    _metrics,
    _ridge_predict,
    _select_fit_hidden_actions,
    _standardize_fit_ridge,
)
from scripts.exploratory.phase2_gym_pendulum_mechanism_signature_probe import (  # noqa: E402
    _json_safe,
    _stats,
)
from ticl.model_builder import load_model  # noqa: E402
from ticl.rl_validation import (  # noqa: E402
    _advance_validation_state_action_pairs,
    _build_validation_episode_states,
    _finalize_validation_rollout,
    _resolve_validation_action_bounds,
    _resolve_validation_recurrent_ppo_policy,
)
from ticl.rlpfn_maintained_path import resolve_rlpfn_token_layout  # noqa: E402


DEFAULT_OUTPUT_JSON = (
    "/home/chen/RLPFN/artifacts/phase2_gym_pendulum_frozen_hidden_head_rollout_probe_0509/"
    "fit_epoch29_hidden_head_rollout_train32_eval16.json"
)


def _make_train_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        fit_checkpoint=args.fit_checkpoint,
        official_model=args.official_model,
        official_vecnormalize=args.official_vecnormalize,
        episodes=int(args.train_episodes),
        max_steps=int(args.max_steps),
        context_lower_bound=int(args.context_lower_bound),
        seed=int(args.train_seed),
        device=str(args.device),
        max_parallel_columns=int(args.max_parallel_columns),
        ridge=float(args.ridge),
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


def _summarize_records(records: list[dict[str, Any]], *, early_frac: float) -> dict[str, Any]:
    eval_records = [r for r in records if bool(r["is_eval"])]
    early_records = [r for r in eval_records if float(r["eval_step_frac"]) < float(early_frac)]

    def vals(seq: list[dict[str, Any]], key: str) -> list[float]:
        return [float(r[key]) for r in seq if key in r and math.isfinite(float(r[key]))]

    def one(seq: list[dict[str, Any]]) -> dict[str, Any]:
        fit = vals(seq, "fit_action_mean")
        official = vals(seq, "official_action")
        ridge = vals(seq, "ridge_action")
        applied = vals(seq, "applied_action")
        theta_dot = vals(seq, "theta_dot")
        return {
            "count": int(len(seq)),
            "fit_abs": _stats([abs(x) for x in fit]),
            "official_abs": _stats([abs(x) for x in official]),
            "ridge_abs": _stats([abs(x) for x in ridge]),
            "applied_abs": _stats([abs(x) for x in applied]),
            "ridge_vs_official": _metrics(np.asarray(official, dtype=np.float64), np.asarray(ridge, dtype=np.float64)),
            "fit_vs_official": _metrics(np.asarray(official, dtype=np.float64), np.asarray(fit, dtype=np.float64)),
            "corr_ridge_theta_dot": _corr(ridge, theta_dot),
            "corr_official_theta_dot": _corr(official, theta_dot),
            "corr_fit_theta_dot": _corr(fit, theta_dot),
        }

    return {"eval": one(eval_records), "early_eval": one(early_records)}


def _state_terms(obs: np.ndarray) -> dict[str, float]:
    obs_arr = np.asarray(obs, dtype=np.float32).reshape(-1)
    cos_theta = float(np.clip(obs_arr[0], -1.0, 1.0)) if obs_arr.size > 0 else 0.0
    sin_theta = float(np.clip(obs_arr[1], -1.0, 1.0)) if obs_arr.size > 1 else 0.0
    theta_dot = float(obs_arr[2]) if obs_arr.size > 2 else 0.0
    theta = float(np.arctan2(sin_theta, cos_theta))
    return {
        "cos_theta": cos_theta,
        "sin_theta": sin_theta,
        "theta": theta,
        "theta_dot": theta_dot,
    }


def _run_eval_mode(
    *,
    mode: str,
    ridge_head: dict[str, np.ndarray],
    feature_name: str,
    fit_policy: Any,
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
    early_frac: float,
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
                    _finalize_validation_rollout(state, int(context_lower_bound))
                    continue
                active_states.append(state)
            selected = _select_fit_hidden_actions(
                states=active_states,
                policy=fit_policy,
                policy_step_fn=fit_policy_step_fn,
                device=device,
                obs_slot_dim=obs_slot_dim,
                action_slot_dim=action_slot_dim,
                terminal_token_enabled=terminal_token_enabled,
                max_parallel_columns=int(max_parallel_columns),
            )
            pairs = []
            for state, payload in selected:
                obs = np.asarray(state["obs"], dtype=np.float32).reshape(-1).copy()
                official = _official_action(official_model, official_vecnorm, obs)
                is_eval = bool(float(state["phase_flag"]) >= 1.0)
                eval_step = int(state["current_rollout_len"])
                fit_action = np.asarray(payload["action"], dtype=np.float32).reshape(-1)
                fit_mean = np.asarray(payload["fit_action_mean"], dtype=np.float32).reshape(-1)
                features = np.asarray(payload[feature_name], dtype=np.float32).reshape(1, -1)
                ridge_action = np.clip(
                    _ridge_predict(ridge_head, features).reshape(-1).astype(np.float32),
                    state["action_low"],
                    state["action_high"],
                )
                if mode == "fit":
                    applied = fit_action
                elif mode == "official_eval_full":
                    applied = official if is_eval else fit_action
                elif mode == "hidden_head_eval_full":
                    applied = ridge_action if is_eval else fit_action
                else:
                    raise ValueError(f"unsupported mode={mode!r}")
                applied = np.clip(
                    np.asarray(applied, dtype=np.float32).reshape(-1),
                    state["action_low"],
                    state["action_high"],
                ).astype(np.float32)
                records.append(
                    {
                        "mode": mode,
                        "episode_idx": int(state["env_idx"]),
                        "rollout_step": int(eval_step),
                        "eval_step_frac": float(eval_step) / float(max(1, max_steps)),
                        "phase_flag": float(state["phase_flag"]),
                        "is_eval": bool(is_eval),
                        "fit_action_mean": float(fit_mean[0]),
                        "official_action": float(official.reshape(-1)[0]),
                        "ridge_action": float(ridge_action.reshape(-1)[0]),
                        "applied_action": float(applied.reshape(-1)[0]),
                        **_state_terms(obs),
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
        "summary": _summarize_records(records, early_frac=float(early_frac)),
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-checkpoint", default=DEFAULT_FIT_CHECKPOINT)
    parser.add_argument("--official-model", default=DEFAULT_OFFICIAL_MODEL)
    parser.add_argument("--official-vecnormalize", default=DEFAULT_OFFICIAL_VECNORMALIZE)
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--train-episodes", type=int, default=32)
    parser.add_argument("--eval-episodes", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--context-lower-bound", type=int, default=2048)
    parser.add_argument("--train-seed", type=int, default=20260509)
    parser.add_argument("--eval-seed", type=int, default=20270509)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-parallel-columns", type=int, default=32)
    parser.add_argument("--ridge", type=float, default=0.1)
    parser.add_argument("--feature", choices=["hidden", "latent_pi"], default="hidden")
    parser.add_argument("--early-frac", type=float, default=0.25)
    parser.add_argument("--modes", default="fit,hidden_head_eval_full,official_eval_full")
    args = parser.parse_args()

    device = str(args.device)
    if device.startswith("cuda") and not device.startswith("cuda:0"):
        raise ValueError("This probe is constrained to GPU0; pass --device cuda:0 or --device cpu.")

    metadata, arrays = _collect_dataset(_make_train_args(args))
    y_train = arrays["official_action"].astype(np.float64)
    x_train = arrays[str(args.feature)].astype(np.float64)
    ridge_head = _standardize_fit_ridge(x_train, y_train, ridge=float(args.ridge))
    train_pred = np.clip(_ridge_predict(ridge_head, x_train), -2.0, 2.0)
    train_metrics = _metrics(y_train, train_pred)

    import gymnasium as gym

    official_model, official_vecnorm = _load_official_policy(
        model_zip=Path(args.official_model).expanduser().resolve(),
        vecnormalize_pkl=Path(args.official_vecnormalize).expanduser().resolve(),
    )
    fit_model, config = load_model(str(Path(args.fit_checkpoint).expanduser().resolve()), device=device, verbose=False)
    config = copy.deepcopy(config)
    config["device"] = device
    orch = config.setdefault("orchestration", {})
    orch["rl_validate_envs"] = "Pendulum-v1"
    orch["rl_validate_episodes"] = int(args.eval_episodes)
    orch["rl_validate_max_steps"] = int(args.max_steps)
    orch["rl_validate_context_lower_bound"] = int(args.context_lower_bound)
    orch["rl_validate_seed"] = int(args.eval_seed)
    orch["rl_validate_max_parallel_columns"] = int(args.max_parallel_columns)
    orch["rl_validate_enabled"] = True
    env_cfg = config.get("prior", {}).get("environment", {})
    layout = resolve_rlpfn_token_layout(env_cfg, num_features=config.get("prior", {}).get("num_features", None))
    obs_slot_dim = int(layout["obs_slot_dim"])
    action_slot_dim = int(layout["action_slot_dim"])
    terminal_token_enabled = bool(layout["terminal_token_enabled"])
    policy = _resolve_validation_recurrent_ppo_policy(
        model=fit_model,
        config=config,
        env_cfg=env_cfg,
        device=device,
        num_features=int(layout["num_features"]),
    )
    fit_policy_step_fn = policy.make_vectorized_rollout_step_fn()

    mode_results = {}
    try:
        modes = [part.strip() for part in str(args.modes).split(",") if part.strip()]
        for mode in modes:
            mode_results[mode] = _run_eval_mode(
                mode=mode,
                ridge_head=ridge_head,
                feature_name=str(args.feature),
                fit_policy=policy,
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
                episodes=int(args.eval_episodes),
                max_steps=int(args.max_steps),
                context_lower_bound=int(args.context_lower_bound),
                seed=int(args.eval_seed),
                max_parallel_columns=int(args.max_parallel_columns),
                early_frac=float(args.early_frac),
            )
            print(
                json.dumps(
                    {
                        "mode": mode,
                        "return_mean": mode_results[mode]["return_mean"],
                        "early": mode_results[mode]["summary"]["early_eval"],
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
        "schema": "phase2_gym_pendulum_frozen_hidden_head_rollout_probe.v1",
        "contract": {
            "exploratory_only": True,
            "no_fit_model_training": True,
            "no_ppo_mutation": True,
            "no_prior_generator_mutation": True,
            "zero_action_not_reference": True,
            "train_eval_seed_disjoint": True,
            "closed_loop_intervention": True,
        },
        "config": {
            "fit_checkpoint": str(Path(args.fit_checkpoint).expanduser().resolve()),
            "official_model": str(Path(args.official_model).expanduser().resolve()),
            "official_vecnormalize": str(Path(args.official_vecnormalize).expanduser().resolve()),
            "feature": str(args.feature),
            "ridge": float(args.ridge),
            "train_episodes": int(args.train_episodes),
            "eval_episodes": int(args.eval_episodes),
            "max_steps": int(args.max_steps),
            "context_lower_bound": int(args.context_lower_bound),
            "train_seed": int(args.train_seed),
            "eval_seed": int(args.eval_seed),
            "device": device,
        },
        "train_dataset": metadata,
        "train_head_action_clipped_metrics": train_metrics,
        "mode_results": mode_results,
    }
    output = Path(args.output_json).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output_json": str(output), "modes": list(mode_results)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
