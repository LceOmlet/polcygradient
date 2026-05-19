#!/usr/bin/env python3
"""Fixed true-runner micro-family audit for terminal-event reward logging.

This audit deliberately does not sample a new training family.  It takes an
existing fixed h-list, runs the real EnvironmentPriorPPOBatchVecEnv step path,
and checks whether terminal-event reward is observable as a component without
dominating the base env reward.
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

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ticl.model_configs import get_model_default_config  # noqa: E402
from ticl.priors.environment_prior import EnvironmentPrior  # noqa: E402
from ticl.rlpfn_maintained_path import validate_rlpfn_maintained_path_config  # noqa: E402
from ticl.sb3_recurrent_ppo import EnvironmentPriorPPOBatchVecEnv  # noqa: E402

ART = Path("/home/chen/RLPFN/artifacts")
DEFAULT_H_LIST = (
    ART
    / "phase2_terminal_coverage_prior_sidecar_n64_s512_u1_rerun_0507"
    / "fixed_env_group"
    / "prior_fixed_h_list.json"
)
DEFAULT_SEEDS = (
    ART
    / "phase2_terminal_coverage_prior_sidecar_n64_s512_u1_rerun_0507"
    / "fixed_env_group"
    / "prior_fixed_env_seeds.json"
)
DEFAULT_OUT = ART / "phase2_terminal_event_true_runner_microfamily_audit_0518"


def _json_safe(value: Any) -> Any:
    if torch.is_tensor(value):
        if value.ndim == 0:
            return _json_safe(value.detach().cpu().item())
        return [_json_safe(v) for v in value.detach().cpu().tolist()]
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def _read_json(path: Path) -> Any:
    return json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))


def _finite_stats(arr: np.ndarray) -> dict[str, Any]:
    vals = np.asarray(arr, dtype=np.float64).reshape(-1)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {"n": 0, "mean": None, "std": None, "q50": None, "q90": None, "max_abs": None}
    return {
        "n": int(vals.size),
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "q50": float(np.quantile(vals, 0.50)),
        "q90": float(np.quantile(vals, 0.90)),
        "max_abs": float(np.max(np.abs(vals))),
    }


def _scale(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    return np.abs(np.mean(arr, axis=0)) + np.std(arr, axis=0)


def _max_abs(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.max(np.abs(np.asarray(x, dtype=np.float64))))


def _as_int(h: dict[str, Any], key: str, default: int) -> int:
    try:
        return int(h.get(key, default))
    except Exception:
        return int(default)


def _build_fixed_vec_env(
    *,
    h_list: list[dict[str, Any]],
    seeds: list[int],
    device: torch.device,
    n_steps: int,
) -> EnvironmentPriorPPOBatchVecEnv:
    cfg = copy.deepcopy(get_model_default_config("rlpfn"))
    validate_rlpfn_maintained_path_config(cfg)
    cfg["prior"]["environment"]["reward_topology_conditioned_sampling_enabled"] = False
    cfg["prior"]["environment"]["reward_state_input_gain_fraction_conditioned_sampling_enabled"] = False
    prior = EnvironmentPrior(copy.deepcopy(cfg["prior"]["environment"]))
    fixed_h = [copy.deepcopy(h) for h in h_list]
    prior._phase2_fixed_env_group_h_list = fixed_h
    prior._phase2_fixed_env_group_env_seeds = [int(s) for s in seeds]

    def _sample_fixed(batch_n: int, _fixed=fixed_h) -> list[dict[str, Any]]:
        if int(batch_n) != len(_fixed):
            raise RuntimeError(f"fixed micro-family requested {batch_n}, expected {len(_fixed)}")
        return [copy.deepcopy(h) for h in _fixed]

    prior._sample_batch_hypers = _sample_fixed  # type: ignore[method-assign]
    obs_slot_dim = max(_as_int(h, "obs_slot_dim", _as_int(h, "obs_dim", 1)) for h in h_list)
    action_slot_dim = max(_as_int(h, "action_slot_dim", _as_int(h, "action_dim", 1)) for h in h_list)
    action_dim = max(_as_int(h, "action_dim", action_slot_dim) for h in h_list)
    state_dim = max(_as_int(h, "state_dim", obs_slot_dim) for h in h_list)
    num_features = max(obs_slot_dim + 4 + action_slot_dim, obs_slot_dim + 1)
    return EnvironmentPriorPPOBatchVecEnv(
        env_prior=prior,
        device=device,
        num_features=int(num_features),
        n_steps=int(n_steps),
        num_envs=len(h_list),
        action_dim=int(action_dim),
        obs_slot_dim=int(obs_slot_dim),
        action_slot_dim=int(action_slot_dim),
        next_state_target_dim=int(state_dim),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    h_all = _read_json(Path(args.h_list))
    seed_all = _read_json(Path(args.seeds))
    n_envs = int(args.n_envs)
    h_list = [copy.deepcopy(h) for h in h_all[:n_envs]]
    seeds = [int(s) for s in seed_all[:n_envs]]
    device = torch.device(str(args.device))
    torch.manual_seed(int(args.action_seed))
    rng = np.random.default_rng(int(args.action_seed))

    vec_env = _build_fixed_vec_env(h_list=h_list, seeds=seeds, device=device, n_steps=int(args.n_steps))
    vec_env._full_reset_batch(seeds=seeds)
    action_dim = int(vec_env.action_dim)
    rewards = []
    reward_env = []
    reward_ctrl = []
    reward_survival = []
    reward_terminal_bonus = []
    dones = []
    component_missing_steps = 0
    for _step in range(int(args.n_steps)):
        actions = rng.normal(0.0, float(args.action_std), size=(n_envs, action_dim)).astype(np.float32)
        actions = np.clip(actions, -1.0, 1.0)
        vec_env.step_async(actions)
        _obs, reward, done, _infos = vec_env.step_wait()
        component_values = [
            getattr(vec_env, "_last_reward_env_np", None),
            getattr(vec_env, "_last_reward_ctrl_np", None),
            getattr(vec_env, "_last_reward_survival_np", None),
            getattr(vec_env, "_last_reward_terminal_bonus_np", None),
        ]
        if any(v is None or tuple(np.asarray(v).shape) != (n_envs,) for v in component_values):
            component_missing_steps += 1
            continue
        rewards.append(np.asarray(reward, dtype=np.float32))
        reward_env.append(np.asarray(component_values[0], dtype=np.float32))
        reward_ctrl.append(np.asarray(component_values[1], dtype=np.float32))
        reward_survival.append(np.asarray(component_values[2], dtype=np.float32))
        reward_terminal_bonus.append(np.asarray(component_values[3], dtype=np.float32))
        dones.append(np.asarray(done, dtype=bool))

    reward = np.stack(rewards, axis=0) if rewards else np.zeros((0, n_envs), dtype=np.float32)
    env = np.stack(reward_env, axis=0) if reward_env else np.zeros_like(reward)
    ctrl = np.stack(reward_ctrl, axis=0) if reward_ctrl else np.zeros_like(reward)
    survival = np.stack(reward_survival, axis=0) if reward_survival else np.zeros_like(reward)
    terminal = (
        np.stack(reward_terminal_bonus, axis=0)
        if reward_terminal_bonus
        else np.zeros_like(reward)
    )
    done_arr = np.stack(dones, axis=0) if dones else np.zeros_like(reward, dtype=bool)
    reconstructed = env + ctrl + survival + terminal
    reconstruction_max_abs_err = _max_abs(reward - reconstructed)
    env_scale = _scale(env)
    terminal_scale = _scale(terminal)
    terminal_to_env = terminal_scale / np.maximum(env_scale, 1e-9)
    terminal_abs = np.abs(terminal)
    terminal_nonzero = terminal_abs > float(args.nonzero_eps)
    terminal_event_observed = bool(np.any(terminal_nonzero))
    base_env_present_all = bool(np.all(env_scale > float(args.nonzero_eps)))
    terminal_scale_pass = bool(np.max(terminal_to_env) <= float(args.max_terminal_to_env_scale))
    reconstruction_pass = bool(reconstruction_max_abs_err <= float(args.reconstruction_atol))
    component_logging_pass = bool(component_missing_steps == 0 and reward.shape[0] == int(args.n_steps))
    hard_pass = bool(
        component_logging_pass
        and reconstruction_pass
        and terminal_event_observed
        and base_env_present_all
        and terminal_scale_pass
    )
    report = {
        "analysis_entry": "phase2_terminal_event_true_runner_microfamily_audit",
        "schema": "phase2_terminal_event_true_runner_microfamily_audit.v1",
        "contract": {
            "read_only": True,
            "fixed_h_list": True,
            "does_not_modify_generator": True,
            "does_not_train": True,
            "does_not_authorize_sampler_without_hard_pass": True,
        },
        "inputs": {
            "h_list": str(Path(args.h_list).expanduser().resolve()),
            "seeds": str(Path(args.seeds).expanduser().resolve()),
            "n_envs": int(n_envs),
            "n_steps": int(args.n_steps),
            "device": str(device),
            "action_std": float(args.action_std),
        },
        "read": {
            "hard_pass": hard_pass,
            "component_logging_pass": component_logging_pass,
            "component_missing_steps": int(component_missing_steps),
            "reconstruction_pass": reconstruction_pass,
            "reconstruction_max_abs_err": reconstruction_max_abs_err,
            "terminal_event_observed": terminal_event_observed,
            "terminal_nonzero_fraction": float(np.mean(terminal_nonzero)) if terminal.size else 0.0,
            "done_fraction": float(np.mean(done_arr)) if done_arr.size else 0.0,
            "base_env_present_all": base_env_present_all,
            "terminal_scale_pass": terminal_scale_pass,
            "max_terminal_to_env_scale": float(np.max(terminal_to_env)) if terminal_to_env.size else 0.0,
            "q90_terminal_to_env_scale": float(np.quantile(terminal_to_env, 0.90)) if terminal_to_env.size else 0.0,
            "thresholds": {
                "nonzero_eps": float(args.nonzero_eps),
                "reconstruction_atol": float(args.reconstruction_atol),
                "max_terminal_to_env_scale": float(args.max_terminal_to_env_scale),
            },
            "source_label_sampler_formula_path_authorized": hard_pass,
        },
        "stats": {
            "reward": _finite_stats(reward),
            "reward_env": _finite_stats(env),
            "reward_ctrl": _finite_stats(ctrl),
            "reward_survival": _finite_stats(survival),
            "reward_terminal_bonus": _finite_stats(terminal),
        },
    }
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "terminal_event_true_runner_microfamily_audit.json"
    md_path = out_dir / "terminal_event_true_runner_microfamily_audit.md"
    report["outputs"] = {"json": str(json_path), "markdown": str(md_path)}
    json_path.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md = [
        "# Terminal Event True-Runner Micro-Family Audit",
        "",
        f"- hard_pass: `{hard_pass}`",
        f"- component_logging_pass: `{component_logging_pass}`",
        f"- reconstruction_max_abs_err: `{reconstruction_max_abs_err}`",
        f"- terminal_event_observed: `{terminal_event_observed}`",
        f"- terminal_nonzero_fraction: `{report['read']['terminal_nonzero_fraction']}`",
        f"- base_env_present_all: `{base_env_present_all}`",
        f"- max_terminal_to_env_scale: `{report['read']['max_terminal_to_env_scale']}`",
        f"- q90_terminal_to_env_scale: `{report['read']['q90_terminal_to_env_scale']}`",
        f"- formula-path sampler gate authorized: `{hard_pass}`",
        "",
        "This gate only covers terminal-event formula-path logging and base-reward preservation.",
        "It is not a full training-family authorization by itself.",
    ]
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(report["read"], indent=2, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h-list", type=Path, default=DEFAULT_H_LIST)
    parser.add_argument("--seeds", type=Path, default=DEFAULT_SEEDS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--action-seed", type=int, default=180517)
    parser.add_argument("--action-std", type=float, default=0.7)
    parser.add_argument("--nonzero-eps", type=float, default=1e-8)
    parser.add_argument("--reconstruction-atol", type=float, default=1e-5)
    parser.add_argument("--max-terminal-to-env-scale", type=float, default=1.0)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
