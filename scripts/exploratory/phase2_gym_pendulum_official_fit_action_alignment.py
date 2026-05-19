#!/usr/bin/env python
"""Official-PPO vs fit_model early-action alignment on Gym Pendulum.

Exploratory/read-only only.  This probe does not train, does not mutate PPO,
does not mutate the prior generator, and does not use zero-action as the
reference.  It asks the nearest failure question:

    On the same raw Gym observations during the fit_model validation eval
    rollout, what action would official single-env PPO take, and what action
    does fit_model take?

This isolates the first-stage failure (early action selection) from later
future-state carryover/settle effects.
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
    _pendulum_terms,
    _stats,
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
    "/home/chen/RLPFN/artifacts/phase2_gym_pendulum_official_fit_action_alignment_0509/"
    "fit_epoch29_teacher_alignment.json"
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


def _summarize(records: list[dict[str, Any]], *, early_frac: float) -> dict[str, Any]:
    eval_records = [r for r in records if bool(r["is_eval"])]
    early_records = [
        r
        for r in eval_records
        if float(r["eval_step_frac"]) < float(max(0.0, min(1.0, early_frac)))
    ]

    def arr(seq: list[dict[str, Any]], key: str) -> list[float]:
        return [float(r[key]) for r in seq]

    def summarize_seq(seq: list[dict[str, Any]]) -> dict[str, Any]:
        fit_action = arr(seq, "fit_action")
        official_action = arr(seq, "official_action")
        fit_abs = [abs(v) for v in fit_action]
        official_abs = [abs(v) for v in official_action]
        theta_dot = arr(seq, "theta_dot")
        sin_theta = arr(seq, "sin_theta")
        cos_theta = arr(seq, "cos_theta")
        abs_diff = [abs(float(a) - float(b)) for a, b in zip(fit_action, official_action)]
        sign_match = [
            float(np.sign(float(a)) == np.sign(float(b)))
            for a, b in zip(fit_action, official_action)
            if abs(float(a)) > 1e-6 or abs(float(b)) > 1e-6
        ]
        low_fit_high_teacher = [
            float(abs(float(a)) < 0.1 and abs(float(b)) > 0.5)
            for a, b in zip(fit_action, official_action)
        ]
        return {
            "count": int(len(seq)),
            "fit_abs_action": _stats(fit_abs),
            "official_abs_action": _stats(official_abs),
            "abs_action_diff": _stats(abs_diff),
            "fit_action": _stats(fit_action),
            "official_action": _stats(official_action),
            "fit_to_official_abs_ratio_mean": (
                None
                if not official_abs
                else float(
                    np.mean(
                        np.asarray(fit_abs, dtype=np.float64)
                        / np.maximum(np.asarray(official_abs, dtype=np.float64), 1e-6)
                    )
                )
            ),
            "sign_match_rate": None if not sign_match else float(np.mean(sign_match)),
            "low_fit_high_teacher_rate": None if not low_fit_high_teacher else float(np.mean(low_fit_high_teacher)),
            "corr_fit_official_action": _corr(fit_action, official_action),
            "corr_fit_action_theta_dot": _corr(fit_action, theta_dot),
            "corr_official_action_theta_dot": _corr(official_action, theta_dot),
            "corr_fit_action_sin_theta": _corr(fit_action, sin_theta),
            "corr_official_action_sin_theta": _corr(official_action, sin_theta),
            "corr_fit_action_theta_dot_cos_theta": _corr(
                fit_action,
                [float(td) * float(c) for td, c in zip(theta_dot, cos_theta)],
            ),
            "corr_official_action_theta_dot_cos_theta": _corr(
                official_action,
                [float(td) * float(c) for td, c in zip(theta_dot, cos_theta)],
            ),
        }

    return {
        "eval": summarize_seq(eval_records),
        "early_eval": summarize_seq(early_records),
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
    parser.add_argument("--early-frac", type=float, default=0.25)
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

    records: list[dict[str, Any]] = []
    try:
        while any(not bool(state["done"]) for state in states):
            active_states = [
                state
                for state in states
                if not bool(state["done"]) and int(state["current_rollout_len"]) < int(args.max_steps)
            ]
            raw_pairs = _select_ppo_policy_actions_batched(
                states=active_states,
                policy=fit_policy,
                policy_step_fn=fit_policy_step_fn,
                device=device,
                obs_slot_dim=obs_slot_dim,
                action_slot_dim=action_slot_dim,
                terminal_token_enabled=terminal_token_enabled,
                max_parallel_columns=int(args.max_parallel_columns),
            )
            pairs = []
            for state, fit_action in raw_pairs:
                obs = np.asarray(state["obs"], dtype=np.float32).copy()
                official = _official_action(official_model, official_vecnorm, obs)
                fit_action = np.asarray(fit_action, dtype=np.float32).reshape(-1)
                terms = _pendulum_terms(obs, fit_action)
                eval_step = int(state["current_rollout_len"])
                records.append(
                    {
                        "episode_idx": int(state["env_idx"]),
                        "context_len": int(state["context_len"]),
                        "rollout_step": int(eval_step),
                        "eval_step_frac": float(eval_step) / float(max(1, int(args.max_steps))),
                        "phase_flag": float(state["phase_flag"]),
                        "is_eval": bool(float(state["phase_flag"]) >= 1.0),
                        "fit_action": float(fit_action[0]),
                        "official_action": float(official[0]),
                        "fit_abs_action": float(abs(float(fit_action[0]))),
                        "official_abs_action": float(abs(float(official[0]))),
                        "action_diff": float(float(fit_action[0]) - float(official[0])),
                        **terms,
                    }
                )
                pairs.append((state, fit_action))
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

    report = {
        "schema": "phase2_gym_pendulum_official_fit_action_alignment.v1",
        "contract": {
            "exploratory_only": True,
            "no_training": True,
            "zero_action_not_reference": True,
            "same_observation_teacher_student_action_compare": True,
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
            "early_frac": float(args.early_frac),
            "state_filter": state_filter,
        },
        "summary": _summarize(records, early_frac=float(args.early_frac)),
        "records": records,
    }
    output = Path(args.output_json).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    compact = report["summary"]["early_eval"]
    print(json.dumps(compact, indent=2, sort_keys=True))
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
