#!/usr/bin/env python
"""Gym action-mode counterfactual for missing cross-env generalization modes.

Exploratory/read-only only.  This probe does not mutate the exact SCM prior,
fit_model, PPO implementation, checkpoints, or Gym environments.

It runs the same Gym validation phase protocol as fit_model and compares:

* policy: checkpoint policy action
* policy_half: closed-loop 0.5 * policy action
* policy_neg: closed-loop -policy action
* zero: zero action
* random: uniform random action in Gym native bounds

The goal is to make the missing mode falsifiable.  If policy_half improves
HalfCheetah/Ant, action-amplitude/energy tradeoff is implicated.  If policy_neg
improves Pendulum/Swimmer, sign/phase/state-conditioning is implicated.  If
random beats policy while official single-env PPO learns, the gap is not env
unlearnability but cross-env action-policy generalization.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ticl.model_builder import load_model
from ticl.rl_validation import (
    _advance_validation_state_action_pairs,
    _build_validation_episode_states,
    _finalize_validation_rollout,
    _resolve_validation_action_bounds,
    _resolve_validation_recurrent_ppo_policy,
    _select_ppo_policy_actions_batched,
)
from ticl.rlpfn_maintained_path import resolve_rlpfn_token_layout


DEFAULT_ENVS = "Pendulum-v1,Swimmer-v5,HalfCheetah-v5,Ant-v5"
DEFAULT_MODES = "policy,policy_half,policy_neg,zero,random"


def _parse_csv(value: str) -> list[str]:
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


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
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    return value


def _stats(values: list[float]) -> dict[str, Any]:
    arr = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=np.float64)
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


def _drop_validation_value_only_state_for_action_probe(model: Any) -> dict[str, Any]:
    dropped: list[str] = []
    for target in (model, getattr(model, "module", None), getattr(model, "_orig_mod", None)):
        if target is None:
            continue
        state = getattr(target, "__dict__", {}).get("_validation_ppo_policy_state", None)
        if not isinstance(state, dict):
            continue
        filtered = {}
        for key, value in state.items():
            key_str = str(key)
            if key_str.startswith("value_vendor_head."):
                dropped.append(key_str)
                continue
            filtered[key_str] = value
        target.__dict__["_validation_ppo_policy_state"] = filtered
    return {
        "dropped_value_only_keys": sorted(set(dropped)),
        "dropped_value_only_key_count": int(len(set(dropped))),
    }


def _transform_mode_action(mode: str, action: np.ndarray, state: dict[str, Any]) -> np.ndarray:
    low = np.asarray(state["action_low"], dtype=np.float32)
    high = np.asarray(state["action_high"], dtype=np.float32)
    if mode == "policy":
        out = action
    elif mode == "policy_half":
        out = 0.5 * action
    elif mode == "policy_neg":
        out = -1.0 * action
    elif mode == "zero":
        out = np.zeros_like(low, dtype=np.float32)
    elif mode == "random":
        out = state["rng"].uniform(low=low, high=high).astype(np.float32)
    else:
        raise ValueError(f"unsupported mode={mode!r}")
    return np.clip(np.asarray(out, dtype=np.float32), low, high).astype(np.float32)


def _action_summary(actions: list[np.ndarray], lows: list[np.ndarray], highs: list[np.ndarray]) -> dict[str, Any]:
    if not actions:
        return {"count": 0}
    absmeans = []
    l2s = []
    l2_ratios = []
    saturations = []
    deltas = []
    by_episode: dict[int, list[np.ndarray]] = {}
    for idx, action in enumerate(actions):
        action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
        low = np.asarray(lows[idx], dtype=np.float32).reshape(-1)
        high = np.asarray(highs[idx], dtype=np.float32).reshape(-1)
        scale = np.maximum(np.maximum(np.abs(low), np.abs(high)), 1e-6)
        absmeans.append(float(np.mean(np.abs(action_arr))))
        l2 = float(np.linalg.norm(action_arr))
        l2s.append(l2)
        l2_ratios.append(float(l2 / max(float(np.linalg.norm(scale)), 1e-6)))
        saturations.append(float(np.mean(np.abs(action_arr) >= 0.95 * scale)))
        by_episode.setdefault(id(action), [])
    # The caller appends temporal deltas separately because actions are flattened
    # across envs.  Keep this hook so the JSON schema is stable.
    del by_episode
    deltas_arr = np.asarray(deltas, dtype=np.float64)
    return {
        "count": int(len(actions)),
        "absmean": _stats(absmeans),
        "l2": _stats(l2s),
        "l2_to_bound": _stats(l2_ratios),
        "saturation_frac": _stats(saturations),
        "temporal_delta_l2": _stats(deltas_arr.tolist()),
    }


def _compute_action_sets_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    actions = [np.asarray(r["action"], dtype=np.float32) for r in records]
    lows = [np.asarray(r["low"], dtype=np.float32) for r in records]
    highs = [np.asarray(r["high"], dtype=np.float32) for r in records]
    summary = _action_summary(actions, lows, highs)
    by_ep: dict[tuple[str, int], list[np.ndarray]] = {}
    for r in records:
        by_ep.setdefault((str(r["env_name"]), int(r["episode_idx"])), []).append(np.asarray(r["action"], dtype=np.float32))
    deltas = []
    per_dim_stds = []
    for seq in by_ep.values():
        if len(seq) >= 2:
            arr = np.stack(seq, axis=0)
            deltas.extend(np.linalg.norm(np.diff(arr, axis=0), axis=1).astype(np.float64).tolist())
            per_dim_stds.append(float(np.mean(np.std(arr, axis=0))))
    summary["temporal_delta_l2"] = _stats(deltas)
    summary["per_dim_temporal_std_mean"] = _stats(per_dim_stds)
    return summary


def _evaluate_mode(
    *,
    model: Any,
    config: dict[str, Any],
    envs: list[str],
    mode: str,
    device: str,
) -> dict[str, Any]:
    import gymnasium as gym

    orch = config.get("orchestration", {})
    episodes = int(orch.get("rl_validate_episodes", 3))
    max_steps = int(orch.get("rl_validate_max_steps", 1000))
    base_seed = int(orch.get("rl_validate_seed", 1))
    context_lower_bound = int(orch.get("rl_validate_context_lower_bound", 2048))
    max_parallel_columns = int(orch.get("rl_validate_max_parallel_columns", 96))
    env_cfg = config.get("prior", {}).get("environment", {})
    layout = resolve_rlpfn_token_layout(env_cfg, num_features=config.get("prior", {}).get("num_features", None))
    obs_slot_dim = int(layout["obs_slot_dim"])
    action_slot_dim = int(layout["action_slot_dim"])
    terminal_token_enabled = bool(layout["terminal_token_enabled"])
    num_features = int(layout["num_features"])

    use_policy = mode.startswith("policy")
    policy = None
    policy_step_fn = None
    if use_policy:
        policy = _resolve_validation_recurrent_ppo_policy(
            model=model,
            config=config,
            env_cfg=env_cfg,
            device=device,
            num_features=num_features,
        )
        policy_step_fn = policy.make_vectorized_rollout_step_fn()

    per_env: dict[str, Any] = {}
    env_states: dict[str, list[dict[str, Any]]] = {}
    env_batches: dict[str, Any] = {}
    action_records_all_by_env: dict[str, list[dict[str, Any]]] = {env_name: [] for env_name in envs}
    action_records_eval_by_env: dict[str, list[dict[str, Any]]] = {env_name: [] for env_name in envs}
    for env_name in envs:
        try:
            probe_env = gym.make(env_name)
            action_low, action_high, _used_fallback = _resolve_validation_action_bounds(probe_env.action_space)
            probe_env.close()
        except Exception as exc:
            per_env[env_name] = {"make_failed": 1, "error": str(exc)}
            continue

        states, env_batch = _build_validation_episode_states(
            gym=gym,
            env_name=env_name,
            episodes=episodes,
            base_seed=base_seed,
            env_cfg=env_cfg,
            orch_cfg=orch,
            device=device,
            action_low=action_low,
            action_high=action_high,
            policy_step_enabled=use_policy,
        )
        for state in states:
            state["__probe_env_name"] = env_name
        env_states[env_name] = states
        env_batches[env_name] = env_batch

    all_states = [state for states in env_states.values() for state in states]
    try:
        while any(not bool(state["done"]) for state in all_states):
            active_states = []
            for state in all_states:
                if bool(state["done"]):
                    continue
                if int(state["current_rollout_len"]) >= max_steps:
                    _finalize_validation_rollout(state, context_lower_bound)
                    continue
                active_states.append(state)

            if use_policy:
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
            else:
                raw_pairs = [(state, np.zeros_like(state["action_low"], dtype=np.float32)) for state in active_states]

            pairs = []
            for state, raw_action in raw_pairs:
                env_name = str(state["__probe_env_name"])
                action = _transform_mode_action(mode, raw_action, state)
                record = {
                    "env_name": env_name,
                    "episode_idx": int(state["env_idx"]),
                    "phase_flag": float(state["phase_flag"]),
                    "action": action.copy(),
                    "low": np.asarray(state["action_low"], dtype=np.float32).copy(),
                    "high": np.asarray(state["action_high"], dtype=np.float32).copy(),
                }
                action_records_all_by_env[env_name].append(record)
                if float(state["phase_flag"]) >= 1.0:
                    action_records_eval_by_env[env_name].append(record)
                pairs.append((state, action))

            _advance_validation_state_action_pairs(
                action_pairs=pairs,
                device=device,
                max_steps=max_steps,
                context_lower_bound=context_lower_bound,
            )
    finally:
        closed_batches = set()
        for env_batch in env_batches.values():
            if env_batch is not None and id(env_batch) not in closed_batches:
                try:
                    env_batch.close()
                except Exception:
                    pass
                closed_batches.add(id(env_batch))
        for state in all_states:
            if state.get("env_batch", None) is not None:
                continue
            try:
                state["env"].close()
            except Exception:
                pass

    for env_name, states in env_states.items():
        action_records_all = action_records_all_by_env.get(env_name, [])
        action_records_eval = action_records_eval_by_env.get(env_name, [])
        returns = [float(s["reported_return"]) for s in states if s.get("reported_return") is not None]
        lengths = [float(s["reported_len"]) for s in states if s.get("reported_len") is not None]
        terms: dict[str, list[float]] = {}
        for state in states:
            reported = state.get("reported_reward_terms", {})
            if not isinstance(reported, dict):
                continue
            for key, value in reported.items():
                fv = _finite_float(value)
                if fv is not None:
                    terms.setdefault(str(key), []).append(fv)
        item: dict[str, Any] = {
            "make_failed": 0,
            "return": _stats(returns),
            "len": _stats(lengths),
            "return_mean": float(np.mean(returns)) if returns else None,
            "len_mean": float(np.mean(lengths)) if lengths else None,
            "action_all": _compute_action_sets_summary(action_records_all),
            "action_eval": _compute_action_sets_summary(action_records_eval),
        }
        for key, values in terms.items():
            item[f"{key}_mean"] = float(np.mean(values))
            item[f"{key}"] = _stats(values)
        per_env[env_name] = item

    return per_env


def _best_trivial(modes: dict[str, Any], env: str) -> tuple[str | None, float | None]:
    candidates = []
    for name in ("zero", "random"):
        val = _finite_float((modes.get(name, {}).get(env, {}) or {}).get("return_mean"))
        if val is not None:
            candidates.append((name, val))
    if not candidates:
        return None, None
    return max(candidates, key=lambda item: item[1])


def _env_decision(modes: dict[str, Any], env: str) -> dict[str, Any]:
    policy_ret = _finite_float((modes.get("policy", {}).get(env, {}) or {}).get("return_mean"))
    half_ret = _finite_float((modes.get("policy_half", {}).get(env, {}) or {}).get("return_mean"))
    neg_ret = _finite_float((modes.get("policy_neg", {}).get(env, {}) or {}).get("return_mean"))
    trivial_name, trivial_ret = _best_trivial(modes, env)
    candidates = [
        ("policy", policy_ret),
        ("policy_half", half_ret),
        ("policy_neg", neg_ret),
        (trivial_name, trivial_ret),
    ]
    candidates = [(k, v) for k, v in candidates if k is not None and v is not None]
    best_name, best_ret = max(candidates, key=lambda item: item[1]) if candidates else (None, None)
    return {
        "policy_minus_best_trivial": None if policy_ret is None or trivial_ret is None else policy_ret - trivial_ret,
        "policy_half_minus_policy": None if policy_ret is None or half_ret is None else half_ret - policy_ret,
        "policy_neg_minus_policy": None if policy_ret is None or neg_ret is None else neg_ret - policy_ret,
        "best_mode": best_name,
        "best_return_mean": best_ret,
        "mechanism_hint": (
            "action_amplitude_energy_tradeoff"
            if half_ret is not None and policy_ret is not None and half_ret > policy_ret
            else "action_sign_or_phase_mismatch"
            if neg_ret is not None and policy_ret is not None and neg_ret > policy_ret
            else "policy_not_better_than_trivial"
            if trivial_ret is not None and policy_ret is not None and policy_ret <= trivial_ret
            else "not_resolved"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--env-ids", default=DEFAULT_ENVS)
    parser.add_argument("--modes", default=DEFAULT_MODES)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--context-lower-bound", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-parallel-columns", type=int, default=96)
    args = parser.parse_args()

    envs = _parse_csv(args.env_ids)
    modes = _parse_csv(args.modes)
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    model, config = load_model(str(checkpoint), device=args.device, verbose=False)
    state_filter = _drop_validation_value_only_state_for_action_probe(model)
    config = copy.deepcopy(config)
    config["device"] = str(args.device)
    orch = config.setdefault("orchestration", {})
    orch["rl_validate_envs"] = ",".join(envs)
    orch["rl_validate_episodes"] = int(args.episodes)
    orch["rl_validate_max_steps"] = int(args.max_steps)
    orch["rl_validate_context_lower_bound"] = int(args.context_lower_bound)
    orch["rl_validate_seed"] = int(args.seed)
    orch["rl_validate_max_parallel_columns"] = int(args.max_parallel_columns)
    orch["rl_validate_enabled"] = True

    mode_results = {}
    output = Path(args.output_json).expanduser().resolve()
    partial_output = output.with_suffix(output.suffix + ".partial")
    output.parent.mkdir(parents=True, exist_ok=True)
    for mode in modes:
        mode_results[mode] = _evaluate_mode(
            model=model,
            config=config,
            envs=envs,
            mode=mode,
            device=str(args.device),
        )
        partial_report = {
            "probe": "phase2_gym_action_mode_counterfactual",
            "partial": True,
            "checkpoint": str(checkpoint),
            "completed_modes": sorted(mode_results),
            "mode_results": mode_results,
        }
        partial_output.write_text(
            json.dumps(_json_safe(partial_report), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "completed_mode": mode,
                    "completed_modes": sorted(mode_results),
                    "partial_output": str(partial_output),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    decisions = {env: _env_decision(mode_results, env) for env in envs}
    report = {
        "probe": "phase2_gym_action_mode_counterfactual",
        "exploratory_only": True,
        "mutates_exact_scm": False,
        "mutates_prior_generator": False,
        "mutates_ppo": False,
        "does_not_add_reward_terms": True,
        "checkpoint": str(checkpoint),
        "state_filter": state_filter,
        "env_ids": envs,
        "modes": modes,
        "mode_results": mode_results,
        "decisions": decisions,
    }
    output.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output_json": str(output), "envs": len(envs), "modes": len(modes)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
