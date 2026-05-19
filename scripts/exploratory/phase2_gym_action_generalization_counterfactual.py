#!/usr/bin/env python
"""Narrow Gym action-generalization counterfactual for fit_model checkpoints.

This is an exploratory read-only probe.  It does not mutate the prior
generator, exact SCM defaults, PPO code, checkpoints, or Gym environments.

The probe compares the checkpoint policy against trivial action baselines under
the same validation reset/phase protocol:

* policy: the checkpoint's validation PPO policy path
* zero: zero action for both exploration and final evaluation rollouts
* random: uniform random action in the native Gym action bounds

The intended claim is deliberately narrow.  If a Gym env is learnable by
official single-env PPO, but the checkpoint policy is not better than zero or
random here, the missing piece is action-policy cross-env generalization rather
than terminal reward shaping.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
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
    evaluate_rlpfn_on_gym_envs,
)


DEFAULT_ENVS = "Pendulum-v1,Swimmer-v5,HalfCheetah-v5,Ant-v5"
DEFAULT_OFFICIAL_BASELINE_DIR = (
    "/home/chen/RLPFN/artifacts/phase2_official_ppo_single_env_gym5_raweval_n8_u100_seed5050_gpu0_0502"
)

RETURN_RE = re.compile(
    r"^Epoch\s+(?P<epoch>\d+)\s+return_(?P<env>.+)_(?P<metric>"
    r"return_mean|len_mean|reward_ctrl_mean|reward_forward_mean|reward_survive_mean|reward_contact_mean"
    r")\s+(?P<value>[-+0-9.eE]+)\s*$"
)


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


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


def _parse_envs(value: str) -> list[str]:
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _summarize_validation_result(per_env: dict[str, Any], envs: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for env in envs:
        item = per_env.get(env, {})
        out[env] = {
            "return_mean": _finite_float(item.get("return_mean")),
            "len_mean": _finite_float(item.get("len_mean")),
            "make_failed": int(item.get("make_failed", 0)) if isinstance(item, dict) else 1,
            "reward_ctrl_mean": _finite_float(item.get("reward_ctrl_mean")),
            "reward_forward_mean": _finite_float(item.get("reward_forward_mean")),
            "reward_survive_mean": _finite_float(item.get("reward_survive_mean")),
            "reward_contact_mean": _finite_float(item.get("reward_contact_mean")),
            "context_len_before_eval_mean": _finite_float(item.get("context_len_before_eval_mean")),
            "explore_rollout_count_mean": _finite_float(item.get("explore_rollout_count_mean")),
            "explore_rollout_len_mean": _finite_float(item.get("explore_rollout_len_mean")),
        }
    return out


def _evaluate_trivial_policy(
    *,
    envs: list[str],
    action_mode: str,
    config: dict[str, Any],
    device: str,
) -> dict[str, Any]:
    import gymnasium as gym

    orch = config.get("orchestration", {})
    episodes = int(orch.get("rl_validate_episodes", 3))
    max_steps = int(orch.get("rl_validate_max_steps", 1000))
    base_seed = int(orch.get("rl_validate_seed", 1))
    context_lower_bound = int(orch.get("rl_validate_context_lower_bound", 2048))
    env_cfg = config.get("prior", {}).get("environment", {})

    per_env: dict[str, Any] = {}
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
            policy_step_enabled=False,
        )
        all_actions: list[np.ndarray] = []
        try:
            while any(not bool(state["done"]) for state in states):
                pairs = []
                for state in states:
                    if bool(state["done"]):
                        continue
                    if int(state["current_rollout_len"]) >= max_steps:
                        _finalize_validation_rollout(state, context_lower_bound)
                        continue
                    low = np.asarray(state["action_low"], dtype=np.float32)
                    high = np.asarray(state["action_high"], dtype=np.float32)
                    if action_mode == "zero":
                        action = np.zeros_like(low, dtype=np.float32)
                    elif action_mode == "random":
                        action = state["rng"].uniform(low=low, high=high).astype(np.float32)
                    else:
                        raise ValueError(f"unsupported action_mode={action_mode!r}")
                    all_actions.append(action.copy())
                    pairs.append((state, action))
                _advance_validation_state_action_pairs(
                    action_pairs=pairs,
                    device=device,
                    max_steps=max_steps,
                    context_lower_bound=context_lower_bound,
                )
        finally:
            if env_batch is not None:
                try:
                    env_batch.close()
                except Exception:
                    pass
            for state in states:
                if state.get("env_batch", None) is not None:
                    continue
                try:
                    state["env"].close()
                except Exception:
                    pass

        returns = [float(s["reported_return"]) for s in states if s.get("reported_return") is not None]
        lengths = [float(s["reported_len"]) for s in states if s.get("reported_len") is not None]
        reward_terms: dict[str, list[float]] = {}
        for state in states:
            terms = state.get("reported_reward_terms", {})
            if not isinstance(terms, dict):
                continue
            for key, value in terms.items():
                fv = _finite_float(value)
                if fv is not None:
                    reward_terms.setdefault(str(key), []).append(fv)
        actions = np.concatenate([a.reshape(1, -1) for a in all_actions], axis=0) if all_actions else np.zeros((0, 0))
        item: dict[str, Any] = {
            "make_failed": 0,
            "return_mean": float(np.mean(returns)) if returns else None,
            "len_mean": float(np.mean(lengths)) if lengths else None,
            "returns": returns,
            "lengths": lengths,
            "action_absmean": float(np.mean(np.abs(actions))) if actions.size else None,
            "action_std": float(np.std(actions)) if actions.size else None,
            "action_absmax": float(np.max(np.abs(actions))) if actions.size else None,
        }
        for key, values in reward_terms.items():
            item[f"{key}_mean"] = float(np.mean(values))
        per_env[env_name] = item
    return per_env


def _parse_fit_log(path: str | Path, envs: list[str]) -> dict[str, Any]:
    parsed: dict[str, dict[int, dict[str, float]]] = {env: {} for env in envs}
    log_path = Path(path).expanduser()
    if not log_path.exists():
        return {"path": str(log_path), "exists": False, "envs": {}}
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = RETURN_RE.match(line.strip())
        if not match:
            continue
        env = str(match.group("env"))
        if env not in parsed:
            continue
        epoch = int(match.group("epoch"))
        parsed[env].setdefault(epoch, {})[str(match.group("metric"))] = float(match.group("value"))
    env_reports = {}
    for env, by_epoch in parsed.items():
        epochs = sorted(by_epoch)
        if not epochs:
            env_reports[env] = {"epochs": 0}
            continue
        first = by_epoch[epochs[0]]
        last = by_epoch[epochs[-1]]
        best_epoch = max(
            epochs,
            key=lambda ep: float(by_epoch[ep].get("return_mean", -float("inf"))),
        )
        best = by_epoch[best_epoch]
        env_reports[env] = {
            "epochs": len(epochs),
            "first_epoch": epochs[0],
            "last_epoch": epochs[-1],
            "best_epoch": best_epoch,
            "first": first,
            "last": last,
            "best": best,
            "delta_return_last_minus_first": (
                _finite_float(last.get("return_mean")) - _finite_float(first.get("return_mean"))
                if _finite_float(last.get("return_mean")) is not None
                and _finite_float(first.get("return_mean")) is not None
                else None
            ),
        }
    return {"path": str(log_path.resolve()), "exists": True, "envs": env_reports}


def _load_official_baselines(root: str | Path, envs: list[str]) -> dict[str, Any]:
    root_path = Path(root).expanduser()
    reports = {}
    for env in envs:
        progress = root_path / env / "progress.jsonl"
        if not progress.exists():
            reports[env] = {"exists": False, "progress_path": str(progress)}
            continue
        rows = [json.loads(line) for line in progress.read_text().splitlines() if line.strip()]
        if not rows:
            reports[env] = {"exists": True, "rows": 0, "progress_path": str(progress)}
            continue
        first = rows[0]
        last = rows[-1]

        def mean(row: dict[str, Any], mode: str) -> float | None:
            return _finite_float((row.get(mode) or {}).get("raw_episode_return", {}).get("mean"))

        det_first = mean(first, "deterministic_eval")
        det_last = mean(last, "deterministic_eval")
        stoch_first = mean(first, "stochastic_eval")
        stoch_last = mean(last, "stochastic_eval")
        reports[env] = {
            "exists": True,
            "rows": len(rows),
            "progress_path": str(progress),
            "first_update": first.get("update_idx"),
            "last_update": last.get("update_idx"),
            "deterministic_first": det_first,
            "deterministic_last": det_last,
            "deterministic_delta": None if det_first is None or det_last is None else det_last - det_first,
            "stochastic_first": stoch_first,
            "stochastic_last": stoch_last,
            "stochastic_delta": None if stoch_first is None or stoch_last is None else stoch_last - stoch_first,
        }
    return reports


def _drop_validation_value_only_state_for_action_probe(model: Any) -> dict[str, Any]:
    """Drop value-only vendor-head keys for this action-only offline probe.

    Fit-time validation can keep a live policy object.  When loading a checkpoint
    offline, the saved validation policy state may contain a vendor value head
    that is not constructed by the lightweight validation policy builder.  This
    probe only needs action selection; value predictions are not consumed by the
    Gym environment.  Keep all actor/RWKV/action-head keys and report the dropped
    value-only keys explicitly.
    """

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


def _decision_for_env(env: str, item: dict[str, Any]) -> dict[str, Any]:
    policy_ret = _finite_float((item.get("policy") or {}).get("return_mean"))
    zero_ret = _finite_float((item.get("zero") or {}).get("return_mean"))
    random_ret = _finite_float((item.get("random") or {}).get("return_mean"))
    official = item.get("official_single_env") or {}
    official_delta = _finite_float(official.get("deterministic_delta"))
    official_learnable = bool(official.get("exists")) and official_delta is not None and official_delta > 50.0
    trivial_best = max(v for v in [zero_ret, random_ret] if v is not None) if any(
        v is not None for v in [zero_ret, random_ret]
    ) else None
    policy_beats_trivial = (
        policy_ret is not None and trivial_best is not None and policy_ret > trivial_best
    )
    return {
        "env": env,
        "official_single_env_learnable_witness": official_learnable,
        "policy_beats_zero_or_random": policy_beats_trivial,
        "policy_minus_best_trivial": None if policy_ret is None or trivial_best is None else policy_ret - trivial_best,
        "narrow_conclusion": (
            "cross_env_action_policy_failure_supported"
            if official_learnable and not policy_beats_trivial
            else "not_sufficient_from_this_probe"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", required=True, help="Comma-separated fit_model .cpkt paths.")
    parser.add_argument("--validation-log", default="")
    parser.add_argument("--env-ids", default=DEFAULT_ENVS)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--context-lower-bound", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-parallel-columns", type=int, default=96)
    parser.add_argument("--official-baseline-dir", default=DEFAULT_OFFICIAL_BASELINE_DIR)
    args = parser.parse_args()

    envs = _parse_envs(args.env_ids)
    output = Path(args.output_json).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    official = _load_official_baselines(args.official_baseline_dir, envs)
    fit_log = _parse_fit_log(args.validation_log, envs) if args.validation_log else None
    checkpoint_reports = []

    for checkpoint_text in _parse_envs(args.checkpoints):
        checkpoint_path = Path(checkpoint_text).expanduser().resolve()
        model, config = load_model(str(checkpoint_path), device=args.device, verbose=False)
        action_probe_state_filter = _drop_validation_value_only_state_for_action_probe(model)
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

        global_mean, per_env = evaluate_rlpfn_on_gym_envs(model=model, config=config)
        policy_summary = _summarize_validation_result(per_env, envs)
        zero_summary = _evaluate_trivial_policy(
            envs=envs,
            action_mode="zero",
            config=config,
            device=str(args.device),
        )
        random_summary = _evaluate_trivial_policy(
            envs=envs,
            action_mode="random",
            config=config,
            device=str(args.device),
        )
        env_reports = {}
        for env in envs:
            env_item = {
                "policy": policy_summary.get(env, {}),
                "zero": zero_summary.get(env, {}),
                "random": random_summary.get(env, {}),
                "official_single_env": official.get(env, {}),
            }
            env_item["decision"] = _decision_for_env(env, env_item)
            env_reports[env] = env_item
        checkpoint_reports.append(
            {
                "checkpoint": str(checkpoint_path),
                "action_probe_state_filter": action_probe_state_filter,
                "global_policy_mean_return": _finite_float(global_mean),
                "envs": env_reports,
            }
        )
        del model
        if torch.cuda.is_available() and str(args.device).startswith("cuda"):
            torch.cuda.empty_cache()

    report = {
        "probe": "phase2_gym_action_generalization_counterfactual",
        "exploratory_only": True,
        "mutates_exact_scm": False,
        "mutates_prior_generator": False,
        "mutates_ppo": False,
        "contract": {
            "environment_side_consumes_actions_only": True,
            "policy_path_uses_existing_fit_model_validation_ppo_math": True,
            "zero_random_baselines_use_same_validation_reset_phase_protocol": True,
            "does_not_add_reward_terms": True,
        },
        "env_ids": envs,
        "official_baselines": official,
        "fit_log_context": fit_log,
        "checkpoints": checkpoint_reports,
    }
    output.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output_json": str(output), "checkpoints": len(checkpoint_reports)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
