#!/usr/bin/env python3
"""True-runner reconstruction audit for potential/progress reward components.

This is an instrumentation gate only.  It materializes a fixed diagnostic
h-list, runs the real EnvironmentPriorPPOBatchVecEnv step path, and checks that
the logged lowtail reward components reconstruct the reward seen by the runner.
It does not modify the sampler and it does not train.
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
DEFAULT_OUT = ART / "phase2_potential_progress_component_reconstruction_audit_0518"

COMPONENT_KEYS = (
    "valid_mask",
    "override_mask",
    "reward_mix",
    "reward_unit_base_before_mix",
    "reward_bias_component",
    "reward_progress_component",
    "reward_potential_delta_component",
    "reward_state_cost_component",
    "reward_ctrl_cost_component",
    "reward_distribution_head_component",
    "reward_paired_response_component",
    "reward_base_formula_candidate",
    "reward_component_sum",
    "reward_candidate_final",
    "reward_unit_after_mix",
    "reward_reconstruction_residual",
    "reward_new_reconstruction_residual",
    "reward_scale",
    "reward_raw_scaled",
    "reward_env_transformed",
    "reward_total_return",
)


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


def _max_abs(arr: np.ndarray) -> float:
    vals = np.asarray(arr, dtype=np.float64).reshape(-1)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0
    return float(np.max(np.abs(vals)))


def _as_int(h: dict[str, Any], key: str, default: int) -> int:
    try:
        return int(h.get(key, default))
    except Exception:
        return int(default)


def _diagnostic_h(h: dict[str, Any], idx: int) -> dict[str, Any]:
    out = copy.deepcopy(h)
    # Keep the true runner path, but make the fixed witness small enough that
    # the audit remains an audit rather than an accidental stress test.
    out["state_dim"] = 12
    out["obs_dim"] = 6
    out["obs_slot_dim"] = 6
    out["action_dim"] = 3
    out["action_slot_dim"] = 3
    out["noise_dim"] = 2
    out["zero_pad_dim"] = 0
    out["prior_mlp_hidden_dim"] = 16
    out["num_layers"] = 2
    out["exact_scm_gym_lowtail_family_enabled"] = True
    out["exact_scm_gym_lowtail_family_selected"] = True
    out["exact_scm_gym_lowtail_family_prob"] = 1.0
    out["exact_scm_gym_lowtail_reward_mix"] = 1.0
    out["exact_scm_gym_lowtail_reward_linear_weight"] = 1.25
    out["exact_scm_gym_lowtail_reward_potential_delta_weight"] = 0.65 + 0.05 * (idx % 3)
    out["exact_scm_gym_lowtail_reward_state_cost"] = 0.05
    out["exact_scm_gym_lowtail_reward_ctrl_cost"] = 0.02
    out["exact_scm_gym_lowtail_reward_distribution_head_enabled"] = False
    out["exact_scm_gym_lowtail_paired_response_preimage_enabled"] = False
    out["exact_scm_gym_lowtail_clock_preimage_enabled"] = False
    out["exact_scm_gym_lowtail_materialized_theta_preimage_enabled"] = False
    out["reward_dropout_enabled"] = False
    return out


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
            raise RuntimeError(f"fixed reconstruction audit requested {batch_n}, expected {len(_fixed)}")
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
    h_list = [_diagnostic_h(h, idx) for idx, h in enumerate(h_all[:n_envs])]
    seeds = [int(s) for s in seed_all[:n_envs]]
    device = torch.device(str(args.device))
    rng = np.random.default_rng(int(args.action_seed))

    vec_env = _build_fixed_vec_env(h_list=h_list, seeds=seeds, device=device, n_steps=int(args.n_steps))
    vec_env._full_reset_batch(seeds=seeds)
    action_dim = int(vec_env.action_dim)
    component_series: dict[str, list[np.ndarray]] = {name: [] for name in COMPONENT_KEYS}
    rewards = []
    env_rewards = []
    ctrl_rewards = []
    survival_rewards = []
    terminal_rewards = []
    component_missing_steps = 0
    for _step in range(int(args.n_steps)):
        actions = rng.normal(0.0, float(args.action_std), size=(n_envs, action_dim)).astype(np.float32)
        actions = np.clip(actions, -1.0, 1.0)
        vec_env.step_async(actions)
        _obs, reward, _done, _infos = vec_env.step_wait()
        payload = getattr(vec_env, "_last_exact_scm_lowtail_reward_components_np", None)
        if not isinstance(payload, dict):
            component_missing_steps += 1
            continue
        step_ok = True
        for name in COMPONENT_KEYS:
            value = payload.get(name)
            if value is None or tuple(np.asarray(value).shape) != (n_envs,):
                step_ok = False
                break
        if not step_ok:
            component_missing_steps += 1
            continue
        for name in COMPONENT_KEYS:
            component_series[name].append(np.asarray(payload[name], dtype=np.float32))
        rewards.append(np.asarray(reward, dtype=np.float32))
        env_rewards.append(np.asarray(getattr(vec_env, "_last_reward_env_np"), dtype=np.float32))
        ctrl_rewards.append(np.asarray(getattr(vec_env, "_last_reward_ctrl_np"), dtype=np.float32))
        survival_rewards.append(np.asarray(getattr(vec_env, "_last_reward_survival_np"), dtype=np.float32))
        terminal_rewards.append(np.asarray(getattr(vec_env, "_last_reward_terminal_bonus_np"), dtype=np.float32))

    stacked = {
        name: np.stack(values, axis=0) if values else np.zeros((0, n_envs), dtype=np.float32)
        for name, values in component_series.items()
    }
    reward_arr = np.stack(rewards, axis=0) if rewards else np.zeros((0, n_envs), dtype=np.float32)
    env_arr = np.stack(env_rewards, axis=0) if env_rewards else np.zeros_like(reward_arr)
    ctrl_arr = np.stack(ctrl_rewards, axis=0) if ctrl_rewards else np.zeros_like(reward_arr)
    survival_arr = np.stack(survival_rewards, axis=0) if survival_rewards else np.zeros_like(reward_arr)
    terminal_arr = np.stack(terminal_rewards, axis=0) if terminal_rewards else np.zeros_like(reward_arr)

    named_component_sum = (
        stacked["reward_bias_component"]
        + stacked["reward_progress_component"]
        + stacked["reward_potential_delta_component"]
        + stacked["reward_state_cost_component"]
        + stacked["reward_ctrl_cost_component"]
        + stacked["reward_distribution_head_component"]
        + stacked["reward_paired_response_component"]
    )
    component_sum_err = _max_abs(stacked["reward_component_sum"] - named_component_sum)
    candidate_err = _max_abs(
        (1.0 - stacked["override_mask"]) * (stacked["reward_candidate_final"] - named_component_sum)
    )
    unit_mix_err = _max_abs(
        stacked["reward_unit_after_mix"]
        - (
            (1.0 - stacked["reward_mix"]) * stacked["reward_unit_base_before_mix"]
            + stacked["reward_mix"] * stacked["reward_candidate_final"]
        )
    )
    raw_scaled_err = _max_abs(
        stacked["reward_raw_scaled"] - stacked["reward_scale"] * stacked["reward_unit_after_mix"]
    )
    total_reconstruction_err = _max_abs(reward_arr - (env_arr + ctrl_arr + survival_arr + terminal_arr))
    component_logging_pass = bool(
        component_missing_steps == 0 and int(reward_arr.shape[0]) == int(args.n_steps)
    )
    no_override_pass = bool(_max_abs(stacked["override_mask"]) <= float(args.override_atol))
    progress_observed = bool(_max_abs(stacked["reward_progress_component"]) > float(args.nonzero_eps))
    potential_delta_observed = bool(
        _max_abs(stacked["reward_potential_delta_component"]) > float(args.nonzero_eps)
    )
    finite_pass = bool(
        np.all(np.isfinite(reward_arr))
        and all(np.all(np.isfinite(value)) for value in stacked.values())
    )
    reconstruction_pass = bool(
        component_sum_err <= float(args.reconstruction_atol)
        and candidate_err <= float(args.reconstruction_atol)
        and unit_mix_err <= float(args.reconstruction_atol)
        and raw_scaled_err <= float(args.reconstruction_atol)
    )
    base_preservation_pass = bool(
        finite_pass
        and total_reconstruction_err <= float(args.reconstruction_atol)
        and _max_abs(env_arr) > float(args.nonzero_eps)
    )
    hard_pass = bool(
        component_logging_pass
        and no_override_pass
        and progress_observed
        and potential_delta_observed
        and reconstruction_pass
        and base_preservation_pass
    )
    report = {
        "analysis_entry": "phase2_potential_progress_component_reconstruction_audit",
        "schema": "phase2_potential_progress_component_reconstruction_audit.v1",
        "contract": {
            "read_only": True,
            "fixed_diagnostic_h_list": True,
            "does_not_modify_sampler": True,
            "does_not_train": True,
            "authorizes_source_label_sampler_only_on_hard_pass": True,
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
            "no_override_pass": no_override_pass,
            "progress_observed": progress_observed,
            "potential_delta_observed": potential_delta_observed,
            "finite_pass": finite_pass,
            "reconstruction_pass": reconstruction_pass,
            "base_preservation_pass": base_preservation_pass,
            "component_sum_err": component_sum_err,
            "candidate_err": candidate_err,
            "unit_mix_err": unit_mix_err,
            "raw_scaled_err": raw_scaled_err,
            "total_reconstruction_err": total_reconstruction_err,
            "source_label_sampler_authorized_after_this_audit": hard_pass,
            "thresholds": {
                "reconstruction_atol": float(args.reconstruction_atol),
                "override_atol": float(args.override_atol),
                "nonzero_eps": float(args.nonzero_eps),
            },
        },
        "stats": {
            "reward": _finite_stats(reward_arr),
            "reward_env": _finite_stats(env_arr),
            "reward_progress_component": _finite_stats(stacked["reward_progress_component"]),
            "reward_potential_delta_component": _finite_stats(
                stacked["reward_potential_delta_component"]
            ),
            "reward_state_cost_component": _finite_stats(stacked["reward_state_cost_component"]),
            "reward_ctrl_cost_component": _finite_stats(stacked["reward_ctrl_cost_component"]),
        },
    }
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "potential_progress_component_reconstruction_audit.json"
    md_path = out_dir / "potential_progress_component_reconstruction_audit.md"
    report["outputs"] = {"json": str(json_path), "markdown": str(md_path)}
    json_path.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md = [
        "# Potential/Progress Component Reconstruction Audit",
        "",
        f"- hard_pass: `{hard_pass}`",
        f"- component_logging_pass: `{component_logging_pass}`",
        f"- no_override_pass: `{no_override_pass}`",
        f"- progress_observed: `{progress_observed}`",
        f"- potential_delta_observed: `{potential_delta_observed}`",
        f"- reconstruction_pass: `{reconstruction_pass}`",
        f"- base_preservation_pass: `{base_preservation_pass}`",
        f"- component_sum_err: `{component_sum_err}`",
        f"- candidate_err: `{candidate_err}`",
        f"- unit_mix_err: `{unit_mix_err}`",
        f"- raw_scaled_err: `{raw_scaled_err}`",
        f"- total_reconstruction_err: `{total_reconstruction_err}`",
        f"- source-label sampler authorized after this audit: `{hard_pass}`",
        "",
        "This gate only proves true-runner logging, reconstruction, and base preservation.",
        "It does not implement or run a potential/progress source-label sampler.",
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
    parser.add_argument("--action-seed", type=int, default=180518)
    parser.add_argument("--action-std", type=float, default=0.7)
    parser.add_argument("--nonzero-eps", type=float, default=1e-8)
    parser.add_argument("--reconstruction-atol", type=float, default=1e-5)
    parser.add_argument("--override-atol", type=float, default=1e-8)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
