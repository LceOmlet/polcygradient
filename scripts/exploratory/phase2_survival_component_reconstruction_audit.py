#!/usr/bin/env python3
"""True-runner audit for the survival reward component.

This is a read-only instrumentation gate. It builds paired Exact-SCM
environments where each pair shares the same sampled base h, seed, and action
sequence; the only intended difference is survival reward off vs on. The audit
checks that the runner-exposed component logs reconstruct total reward and that
survival does not alter base env reward, control reward, terminal bonus, done
events, or state transitions.
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

from ticl.analysis.phase2_m4_potential_progress_milestone import (  # noqa: E402
    install_m4_potential_progress_live_milestone,
)
from ticl.model_configs import get_model_default_config  # noqa: E402
from ticl.priors.environment_prior import EnvironmentPrior  # noqa: E402
from ticl.rlpfn_maintained_path import validate_rlpfn_maintained_path_config  # noqa: E402
from ticl.sb3_recurrent_ppo import EnvironmentPriorPPOBatchVecEnv  # noqa: E402


ART = Path("/home/chen/RLPFN/artifacts")
DEFAULT_OUT = ART / "phase2_survival_component_reconstruction_audit_0519"


def _json_safe(value: Any) -> Any:
    if torch.is_tensor(value):
        return _json_safe(value.detach().cpu().numpy())
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


def _max_abs(value: np.ndarray) -> float:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    return float(np.max(np.abs(arr)))


def _finite_stats(value: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0, "mean": None, "std": None, "q50": None, "q90": None, "max_abs": None}
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "q50": float(np.quantile(arr, 0.50)),
        "q90": float(np.quantile(arr, 0.90)),
        "max_abs": float(np.max(np.abs(arr))),
    }


def _as_int(h: dict[str, Any], key: str, default: int) -> int:
    try:
        return int(h.get(key, default))
    except Exception:
        return int(default)


def _diagnostic_base_h(h: dict[str, Any], idx: int) -> dict[str, Any]:
    out = copy.deepcopy(h)
    out["family"] = "scm"
    out["state_dim"] = 16
    out["obs_dim"] = 8
    out["obs_slot_dim"] = 8
    out["action_dim"] = 4
    out["action_slot_dim"] = 4
    out["noise_dim"] = 2
    out["zero_pad_dim"] = 0
    out["constrained_dim_sampling_enabled"] = False
    out["reward_state_input_gain_fraction_rejection_min"] = 0.0
    out["prior_mlp_hidden_dim"] = 24
    out["num_layers"] = 2
    out["strict_joint_transition_enabled"] = True
    out["reference_semantics_enabled"] = False
    out["reference_scm_zero_pad_inactive_init_enabled"] = True
    out["exact_scm_gym_lowtail_family_enabled"] = True
    out["exact_scm_gym_lowtail_family_prob"] = 1.0
    out["exact_scm_gym_lowtail_family_selected"] = True
    out["exact_scm_gym_lowtail_reward_mix"] = 0.35
    out["exact_scm_gym_lowtail_reward_linear_weight"] = 1.25
    out["exact_scm_gym_lowtail_reward_potential_delta_weight"] = 0.65 if idx % 2 == 0 else 0.0
    out["exact_scm_gym_lowtail_reward_state_cost"] = 0.05
    out["exact_scm_gym_lowtail_reward_ctrl_cost"] = 0.02
    out["pendulum_settle_yield_repair_enabled"] = False
    out["pendulum_settle_yield_repair_prob"] = 0.0
    out["potential_progress_reward_axis_target"] = (
        "potential_delta_progress" if idx % 2 == 0 else "progress_only"
    )
    out["reward_dropout_enabled"] = False
    out["reward_dropout_ratio"] = 0.0
    out["terminal_reset_enabled"] = False
    out["terminal_reset_count_target"] = 0.0
    out["reinforce_reward_transform"] = "none"
    out["reward_clip"] = 10.0
    out["ctrl_reward_weight"] = 0.03
    out["ctrl_reward_enable_prob"] = 1.0
    out["_ctrl_reward_enable_u"] = 0.0
    return out


def _paired_survival_h_list(
    *,
    base_h_list: list[dict[str, Any]],
    survival_weight: float,
) -> list[dict[str, Any]]:
    paired: list[dict[str, Any]] = []
    for pair_idx, base in enumerate(base_h_list):
        off = copy.deepcopy(base)
        off["survival_reward_weight"] = 0.0
        off["survival_reward_enable_prob"] = 0.0
        off["_survival_reward_enable_u"] = 1.0
        off["survival_component_axis_target"] = "survival_off"

        on = copy.deepcopy(base)
        on["survival_reward_weight"] = float(survival_weight)
        on["survival_reward_enable_prob"] = 1.0
        on["_survival_reward_enable_u"] = 0.0
        on["survival_component_axis_target"] = "survival_on"
        on["survival_component_axis_pair_index"] = int(pair_idx)
        off["survival_component_axis_pair_index"] = int(pair_idx)
        paired.extend([off, on])
    return paired


def _build_base_h_list(*, n_pairs: int, h_seed: int) -> list[dict[str, Any]]:
    np.random.seed(int(h_seed))
    torch.manual_seed(int(h_seed))
    cfg = copy.deepcopy(get_model_default_config("rlpfn"))
    validate_rlpfn_maintained_path_config(cfg)
    prior = EnvironmentPrior(copy.deepcopy(cfg["prior"]["environment"]))
    install_m4_potential_progress_live_milestone(prior, max_attempts=256)
    sampled = prior._sample_batch_hypers(int(n_pairs))
    return [_diagnostic_base_h(h, idx) for idx, h in enumerate(sampled)]


def _build_fixed_vec_env(
    *,
    h_list: list[dict[str, Any]],
    seeds: list[int],
    device: torch.device,
    n_steps: int,
) -> EnvironmentPriorPPOBatchVecEnv:
    cfg = copy.deepcopy(get_model_default_config("rlpfn"))
    validate_rlpfn_maintained_path_config(cfg)
    env_cfg = cfg["prior"]["environment"]
    env_cfg["batch_vectorized_strict_rng_match"] = True
    env_cfg["reward_topology_conditioned_sampling_enabled"] = False
    env_cfg["reward_state_input_gain_fraction_conditioned_sampling_enabled"] = False
    env_cfg["reward_dropout_enabled"] = False
    env_cfg["terminal_reset_enabled"] = False
    prior = EnvironmentPrior(copy.deepcopy(env_cfg))
    fixed_h = [copy.deepcopy(h) for h in h_list]
    prior._phase2_fixed_env_group_h_list = fixed_h
    prior._phase2_fixed_env_group_env_seeds = [int(s) for s in seeds]

    def _sample_fixed(batch_n: int, _fixed=fixed_h) -> list[dict[str, Any]]:
        if int(batch_n) != len(_fixed):
            raise RuntimeError(f"survival audit requested {batch_n}, expected {len(_fixed)}")
        return [copy.deepcopy(h) for h in _fixed]

    prior._sample_batch_hypers = _sample_fixed  # type: ignore[method-assign]
    obs_slot_dim = max(_as_int(h, "obs_slot_dim", _as_int(h, "obs_dim", 1)) for h in h_list)
    action_slot_dim = max(_as_int(h, "action_slot_dim", _as_int(h, "action_dim", 1)) for h in h_list)
    action_dim = max(_as_int(h, "action_dim", action_slot_dim) for h in h_list)
    state_dim = max(_as_int(h, "state_dim", obs_slot_dim) for h in h_list)
    terminal_token = any(bool(h.get("terminal_reset_enabled", False)) for h in h_list)
    num_features = max(obs_slot_dim + 3 + int(terminal_token) + action_slot_dim, obs_slot_dim + 1)
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
    n_pairs = int(args.n_pairs)
    if n_pairs <= 0:
        raise ValueError("--n-pairs must be positive")
    n_envs = int(n_pairs * 2)
    device = torch.device(str(args.device))
    base_h_list = _build_base_h_list(n_pairs=n_pairs, h_seed=int(args.h_seed))
    h_list = _paired_survival_h_list(
        base_h_list=base_h_list,
        survival_weight=float(args.survival_weight),
    )
    base_seeds = [int(args.env_seed) + i * 17 for i in range(n_pairs)]
    seeds = [seed for seed in base_seeds for _ in (0, 1)]
    vec_env = _build_fixed_vec_env(h_list=h_list, seeds=seeds, device=device, n_steps=int(args.n_steps))
    vec_env._defer_time_limit_reset_at_rollout_end = True
    vec_env._full_reset_batch(seeds=seeds)

    rng = np.random.default_rng(int(args.action_seed))
    action_dim = int(vec_env.action_dim)
    pair_off = np.arange(0, n_envs, 2, dtype=np.int64)
    pair_on = pair_off + 1

    rewards: list[np.ndarray] = []
    reward_envs: list[np.ndarray] = []
    reward_ctrls: list[np.ndarray] = []
    reward_survivals: list[np.ndarray] = []
    reward_terminals: list[np.ndarray] = []
    state_pair_deltas: list[np.ndarray] = []
    action_pair_deltas: list[np.ndarray] = []
    done_pair_mismatch = 0
    missing_component_steps = 0

    for _step in range(int(args.n_steps)):
        pair_actions = rng.normal(0.0, float(args.action_std), size=(n_pairs, action_dim)).astype(np.float32)
        pair_actions = np.clip(pair_actions, -1.0, 1.0)
        actions = np.repeat(pair_actions, 2, axis=0)
        vec_env.step_async(actions)
        _obs, reward, done, _infos = vec_env.step_wait()
        component_values = (
            getattr(vec_env, "_last_reward_env_np", None),
            getattr(vec_env, "_last_reward_ctrl_np", None),
            getattr(vec_env, "_last_reward_survival_np", None),
            getattr(vec_env, "_last_reward_terminal_bonus_np", None),
        )
        if any(value is None or tuple(np.asarray(value).shape) != (n_envs,) for value in component_values):
            missing_component_steps += 1
            continue
        rewards.append(np.asarray(reward, dtype=np.float32))
        reward_envs.append(np.asarray(component_values[0], dtype=np.float32))
        reward_ctrls.append(np.asarray(component_values[1], dtype=np.float32))
        reward_survivals.append(np.asarray(component_values[2], dtype=np.float32))
        reward_terminals.append(np.asarray(component_values[3], dtype=np.float32))
        state_t = getattr(vec_env, "_state_t", None)
        action_t = getattr(vec_env, "_action_t", None)
        if torch.is_tensor(state_t):
            state_np = state_t.detach().cpu().numpy().astype(np.float32, copy=False)
            state_pair_deltas.append(state_np[pair_on] - state_np[pair_off])
        if torch.is_tensor(action_t):
            action_np = action_t.detach().cpu().numpy().astype(np.float32, copy=False)
            action_pair_deltas.append(action_np[pair_on] - action_np[pair_off])
        done_arr = np.asarray(done, dtype=bool)
        done_pair_mismatch += int(np.count_nonzero(done_arr[pair_on] != done_arr[pair_off]))

    reward_arr = np.stack(rewards, axis=0) if rewards else np.zeros((0, n_envs), dtype=np.float32)
    env_arr = np.stack(reward_envs, axis=0) if reward_envs else np.zeros_like(reward_arr)
    ctrl_arr = np.stack(reward_ctrls, axis=0) if reward_ctrls else np.zeros_like(reward_arr)
    survival_arr = np.stack(reward_survivals, axis=0) if reward_survivals else np.zeros_like(reward_arr)
    terminal_arr = np.stack(reward_terminals, axis=0) if reward_terminals else np.zeros_like(reward_arr)
    state_pair_arr = (
        np.stack(state_pair_deltas, axis=0)
        if state_pair_deltas
        else np.zeros((0, n_pairs, 1), dtype=np.float32)
    )
    action_pair_arr = (
        np.stack(action_pair_deltas, axis=0)
        if action_pair_deltas
        else np.zeros((0, n_pairs, 1), dtype=np.float32)
    )

    reconstruction_err = _max_abs(reward_arr - (env_arr + ctrl_arr + survival_arr + terminal_arr))
    expected_survival = np.zeros_like(survival_arr, dtype=np.float32)
    if survival_arr.size:
        expected_survival[:, pair_on] = float(args.survival_weight)
    survival_formula_err = _max_abs(survival_arr - expected_survival)
    survival_observed = bool(_max_abs(survival_arr[:, pair_on] if survival_arr.size else survival_arr) > 0.0)

    env_pair_err = _max_abs(env_arr[:, pair_on] - env_arr[:, pair_off]) if env_arr.size else 0.0
    ctrl_pair_err = _max_abs(ctrl_arr[:, pair_on] - ctrl_arr[:, pair_off]) if ctrl_arr.size else 0.0
    terminal_pair_err = (
        _max_abs(terminal_arr[:, pair_on] - terminal_arr[:, pair_off]) if terminal_arr.size else 0.0
    )
    total_delta_err = (
        _max_abs((reward_arr[:, pair_on] - reward_arr[:, pair_off]) - float(args.survival_weight))
        if reward_arr.size
        else 0.0
    )
    state_pair_err = _max_abs(state_pair_arr)
    action_pair_err = _max_abs(action_pair_arr)
    finite_pass = bool(
        np.all(np.isfinite(reward_arr))
        and np.all(np.isfinite(env_arr))
        and np.all(np.isfinite(ctrl_arr))
        and np.all(np.isfinite(survival_arr))
        and np.all(np.isfinite(terminal_arr))
    )
    component_logging_pass = bool(missing_component_steps == 0 and reward_arr.shape[0] == int(args.n_steps))
    reconstruction_pass = bool(reconstruction_err <= float(args.reconstruction_atol))
    survival_formula_pass = bool(survival_formula_err <= float(args.reconstruction_atol) and survival_observed)
    base_preservation_pass = bool(
        env_pair_err <= float(args.pair_atol)
        and ctrl_pair_err <= float(args.pair_atol)
        and terminal_pair_err <= float(args.pair_atol)
        and state_pair_err <= float(args.pair_atol)
        and action_pair_err <= float(args.pair_atol)
        and total_delta_err <= float(args.reconstruction_atol)
        and done_pair_mismatch == 0
    )
    hard_pass = bool(
        finite_pass
        and component_logging_pass
        and reconstruction_pass
        and survival_formula_pass
        and base_preservation_pass
    )
    report = {
        "analysis_entry": "phase2_survival_component_reconstruction_audit",
        "schema": "phase2_survival_component_reconstruction_audit.v1",
        "contract": {
            "read_only": True,
            "does_not_modify_sampler": True,
            "does_not_train": True,
            "paired_only_difference": "survival_reward_weight/enable_prob/source_label",
            "source_label_sampler_authorized_only_on_hard_pass": True,
        },
        "inputs": {
            "n_pairs": int(n_pairs),
            "n_envs": int(n_envs),
            "n_steps": int(args.n_steps),
            "device": str(device),
            "h_seed": int(args.h_seed),
            "env_seed": int(args.env_seed),
            "action_seed": int(args.action_seed),
            "action_std": float(args.action_std),
            "survival_weight": float(args.survival_weight),
        },
        "read": {
            "hard_pass": hard_pass,
            "component_logging_pass": component_logging_pass,
            "finite_pass": finite_pass,
            "reconstruction_pass": reconstruction_pass,
            "survival_formula_pass": survival_formula_pass,
            "survival_observed": survival_observed,
            "base_preservation_pass": base_preservation_pass,
            "missing_component_steps": int(missing_component_steps),
            "done_pair_mismatch": int(done_pair_mismatch),
            "reconstruction_err": reconstruction_err,
            "survival_formula_err": survival_formula_err,
            "reward_env_pair_err": env_pair_err,
            "reward_ctrl_pair_err": ctrl_pair_err,
            "reward_terminal_pair_err": terminal_pair_err,
            "state_pair_err": state_pair_err,
            "action_pair_err": action_pair_err,
            "total_delta_minus_survival_err": total_delta_err,
            "source_label_sampler_authorized_after_this_audit": hard_pass,
            "thresholds": {
                "reconstruction_atol": float(args.reconstruction_atol),
                "pair_atol": float(args.pair_atol),
            },
        },
        "stats": {
            "reward": _finite_stats(reward_arr),
            "reward_env": _finite_stats(env_arr),
            "reward_ctrl": _finite_stats(ctrl_arr),
            "reward_survival": _finite_stats(survival_arr),
            "reward_terminal_bonus": _finite_stats(terminal_arr),
        },
        "sampled_axis_counts": {
            "survival_off": int(n_pairs),
            "survival_on": int(n_pairs),
        },
    }
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "survival_component_reconstruction_audit.json"
    md_path = out_dir / "survival_component_reconstruction_audit.md"
    report["outputs"] = {"json": str(json_path), "markdown": str(md_path)}
    json_path.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md = [
        "# Survival Component Reconstruction Audit",
        "",
        f"- hard_pass: `{hard_pass}`",
        f"- component_logging_pass: `{component_logging_pass}`",
        f"- reconstruction_pass: `{reconstruction_pass}`",
        f"- survival_formula_pass: `{survival_formula_pass}`",
        f"- base_preservation_pass: `{base_preservation_pass}`",
        f"- reconstruction_err: `{reconstruction_err}`",
        f"- survival_formula_err: `{survival_formula_err}`",
        f"- reward_env_pair_err: `{env_pair_err}`",
        f"- state_pair_err: `{state_pair_err}`",
        f"- total_delta_minus_survival_err: `{total_delta_err}`",
        f"- source-label sampler authorized after this audit: `{hard_pass}`",
        "",
        "This gate proves the existing survival component path is additive and reconstructable.",
        "It does not implement a survival source-label sampler or run training.",
    ]
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(report["read"], indent=2, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n-pairs", type=int, default=4)
    parser.add_argument("--n-steps", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--h-seed", type=int, default=190519)
    parser.add_argument("--env-seed", type=int, default=519000)
    parser.add_argument("--action-seed", type=int, default=519123)
    parser.add_argument("--action-std", type=float, default=0.7)
    parser.add_argument("--survival-weight", type=float, default=0.02)
    parser.add_argument("--reconstruction-atol", type=float, default=1e-5)
    parser.add_argument("--pair-atol", type=float, default=1e-5)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
