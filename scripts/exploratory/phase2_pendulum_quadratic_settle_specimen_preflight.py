#!/usr/bin/env python
"""Specimen preflight for the controllable quadratic settle factor.

This is an exploratory mechanism test, not an Exact-SCM generator change.  It
constructs a small family of two-dimensional linear-quadratic systems and asks
whether the proposed Pendulum-like reduction is internally sound:

    z = [q, v]
    z[t+1] = A z[t] + B a[t] + eps[t]
    r[t] = c - z[t+1]^T Q z[t+1] - lambda * a[t]^2

The test has a deliberately narrow purpose.  It is not a raw-return objective
and it is not a milestone.  It verifies the structural conditions from
phase2_pendulum_settle_reduction_spec.py before any generator integration:

* controllability,
* reward-state alignment,
* bounded LQR/PD witness feasibility,
* action-sensitive basin entry,
* suffix action decay and reward-relevant state contraction,
* profile non-collapse over the sampled parameter family.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_OUTPUT_DIR = Path("/home/chen/RLPFN/artifacts/phase2_pendulum_quadratic_settle_specimen_preflight_0512")


@dataclass(frozen=True)
class QuadraticSettleEnv:
    dt: float
    rho: float
    beta: float
    q_weight: float
    v_weight: float
    ctrl_weight: float
    action_limit: float
    q0: float
    v0: float
    noise_std: float

    @property
    def A(self) -> np.ndarray:
        return np.asarray([[1.0, self.dt], [0.0, self.rho]], dtype=np.float64)

    @property
    def B(self) -> np.ndarray:
        return np.asarray([[0.0], [self.beta]], dtype=np.float64)

    @property
    def Q(self) -> np.ndarray:
        return np.asarray([[self.q_weight, 0.0], [0.0, self.v_weight]], dtype=np.float64)


def _sample_envs(n_envs: int, seed: int) -> list[QuadraticSettleEnv]:
    rng = np.random.default_rng(int(seed))
    envs: list[QuadraticSettleEnv] = []
    for _ in range(int(n_envs)):
        sign = -1.0 if rng.random() < 0.5 else 1.0
        envs.append(
            QuadraticSettleEnv(
                dt=float(rng.uniform(0.05, 0.12)),
                rho=float(rng.uniform(0.86, 0.98)),
                beta=float(sign * rng.uniform(0.10, 0.35)),
                q_weight=float(rng.uniform(0.8, 2.5)),
                v_weight=float(rng.uniform(0.15, 0.9)),
                ctrl_weight=float(rng.uniform(0.01, 0.08)),
                action_limit=float(rng.uniform(0.8, 1.2)),
                q0=float(rng.uniform(-1.25, 1.25)),
                v0=float(rng.uniform(-0.85, 0.85)),
                noise_std=float(rng.uniform(0.0, 0.01)),
            )
        )
    return envs


def _lqr_gain(env: QuadraticSettleEnv, *, iterations: int = 256) -> np.ndarray:
    A = env.A
    B = env.B
    Q = env.Q
    R = np.asarray([[env.ctrl_weight]], dtype=np.float64)
    P = Q.copy()
    for _ in range(int(iterations)):
        denom = R + B.T @ P @ B
        K = np.linalg.solve(denom, B.T @ P @ A)
        P_next = Q + A.T @ P @ A - A.T @ P @ B @ K
        if np.max(np.abs(P_next - P)) < 1e-12:
            P = P_next
            break
        P = P_next
    return np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A).reshape(2)


def _controllability_rank(env: QuadraticSettleEnv) -> int:
    mat = np.concatenate([env.B, env.A @ env.B], axis=1)
    return int(np.linalg.matrix_rank(mat, tol=1e-8))


def _closed_loop_radius(env: QuadraticSettleEnv, gain: np.ndarray) -> float:
    A_cl = env.A - env.B @ np.asarray(gain, dtype=np.float64).reshape(1, 2)
    return float(np.max(np.abs(np.linalg.eigvals(A_cl))))


def _simulate(
    env: QuadraticSettleEnv,
    *,
    policy: str,
    gain: np.ndarray,
    n_steps: int,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    z = np.asarray([env.q0, env.v0], dtype=np.float64)
    states = np.zeros((int(n_steps) + 1, 2), dtype=np.float64)
    actions = np.zeros((int(n_steps),), dtype=np.float64)
    rewards = np.zeros((int(n_steps),), dtype=np.float64)
    states[0] = z
    random_actions = rng.uniform(-env.action_limit, env.action_limit, size=int(n_steps))
    initial_lqr_action = float(np.clip(-float(np.dot(gain, z)), -env.action_limit, env.action_limit))
    for t in range(int(n_steps)):
        if policy == "zero":
            action = 0.0
        elif policy == "random":
            action = float(random_actions[t])
        elif policy == "open_loop_initial_lqr":
            action = initial_lqr_action
        elif policy == "lqr":
            action = float(np.clip(-float(np.dot(gain, z)), -env.action_limit, env.action_limit))
        else:
            raise ValueError(f"unknown policy: {policy}")
        eps = rng.normal(0.0, env.noise_std, size=(2,))
        z_next = env.A @ z + env.B[:, 0] * action + eps
        reward = 1.0 - float(z_next.T @ env.Q @ z_next) - float(env.ctrl_weight * action * action)
        actions[t] = action
        rewards[t] = reward
        states[t + 1] = z_next
        z = z_next
    return {"states": states, "actions": actions, "rewards": rewards}


def _split(arr: np.ndarray, split_step: int) -> tuple[np.ndarray, np.ndarray]:
    return arr[:split_step], arr[split_step:]


def _state_energy(env: QuadraticSettleEnv, states: np.ndarray) -> np.ndarray:
    return np.asarray([float(z.T @ env.Q @ z) for z in np.asarray(states, dtype=np.float64)], dtype=np.float64)


def _one_env_report(env: QuadraticSettleEnv, *, n_steps: int, seed: int, args: argparse.Namespace) -> dict[str, Any]:
    gain = _lqr_gain(env)
    rank = _controllability_rank(env)
    radius = _closed_loop_radius(env, gain)
    policy_payloads = {
        name: _simulate(env, policy=name, gain=gain, n_steps=int(n_steps), seed=int(seed) + idx * 1009)
        for idx, name in enumerate(("zero", "random", "open_loop_initial_lqr", "lqr"))
    }
    split_step = int(max(1, min(int(n_steps) - 1, round(float(args.split_frac) * float(n_steps)))))

    def returns(policy: str) -> tuple[float, float, float]:
        rewards = policy_payloads[policy]["rewards"]
        prefix, suffix = _split(rewards, split_step)
        return float(np.sum(rewards)), float(np.sum(prefix)), float(np.sum(suffix))

    def action_abs(policy: str) -> tuple[float, float, float, float]:
        actions = np.abs(policy_payloads[policy]["actions"])
        prefix, suffix = _split(actions, split_step)
        saturation = float(np.mean(actions >= (0.98 * env.action_limit)))
        suffix_saturation = float(np.mean(suffix >= (0.98 * env.action_limit)))
        return float(np.mean(actions)), float(np.mean(prefix)), float(np.mean(suffix)), max(saturation, suffix_saturation)

    def energy(policy: str) -> tuple[float, float, float]:
        # Use next states to align with the reward definition.
        state_energy = _state_energy(env, policy_payloads[policy]["states"][1:])
        prefix, suffix = _split(state_energy, split_step)
        return float(np.mean(state_energy)), float(np.mean(prefix)), float(np.mean(suffix))

    lqr_full, _lqr_prefix, lqr_suffix = returns("lqr")
    baseline_full = max(returns("zero")[0], returns("random")[0], returns("open_loop_initial_lqr")[0])
    baseline_suffix = max(returns("zero")[2], returns("random")[2], returns("open_loop_initial_lqr")[2])
    lqr_action_full, lqr_action_prefix, lqr_action_suffix, saturation = action_abs("lqr")
    lqr_energy_full, lqr_energy_prefix, lqr_energy_suffix = energy("lqr")
    best_baseline_suffix_energy = min(
        energy("zero")[2],
        energy("random")[2],
        energy("open_loop_initial_lqr")[2],
    )
    action_sensitive_delta = abs(
        _simulate(env, policy="open_loop_initial_lqr", gain=-gain, n_steps=1, seed=int(seed))["states"][1, 1]
        - policy_payloads["open_loop_initial_lqr"]["states"][1, 1]
    )

    analytic_pass = (
        rank == 2
        and radius < float(args.closed_loop_radius_max)
        and env.q_weight > 0
        and env.v_weight > 0
        and env.ctrl_weight > 0
    )
    reward_margin = max(float(args.abs_reward_margin), float(args.rel_reward_margin) * max(1.0, abs(baseline_full)))
    suffix_margin = max(float(args.abs_suffix_margin), float(args.rel_suffix_margin) * max(1.0, abs(baseline_suffix)))
    witness_reward_pass = (lqr_full - baseline_full) > reward_margin and (lqr_suffix - baseline_suffix) > suffix_margin
    action_decay_pass = (lqr_action_prefix - lqr_action_suffix) > float(args.action_decay_margin)
    settle_pass = (
        lqr_energy_suffix < float(args.self_energy_rel_max) * max(1e-9, lqr_energy_prefix)
        and lqr_energy_suffix < float(args.open_energy_rel_max) * max(1e-9, best_baseline_suffix_energy)
    )
    saturation_pass = saturation <= float(args.saturation_max)
    action_sensitive_pass = action_sensitive_delta >= float(args.action_sensitive_delta_min)
    guard_pass = bool(
        analytic_pass
        and witness_reward_pass
        and action_decay_pass
        and settle_pass
        and saturation_pass
        and action_sensitive_pass
    )

    return {
        "env": asdict(env),
        "lqr_gain": [float(gain[0]), float(gain[1])],
        "controllability_rank": int(rank),
        "closed_loop_radius": float(radius),
        "returns": {name: returns(name) for name in policy_payloads},
        "lqr_action_abs": {
            "full": lqr_action_full,
            "prefix": lqr_action_prefix,
            "suffix": lqr_action_suffix,
            "max_saturation_metric": float(saturation),
        },
        "lqr_state_energy": {
            "full": lqr_energy_full,
            "prefix": lqr_energy_prefix,
            "suffix": lqr_energy_suffix,
            "best_baseline_suffix": best_baseline_suffix_energy,
        },
        "action_sensitive_delta_v": float(action_sensitive_delta),
        "conditions": {
            "analytic_pass": bool(analytic_pass),
            "witness_reward_pass": bool(witness_reward_pass),
            "action_decay_pass": bool(action_decay_pass),
            "settle_pass": bool(settle_pass),
            "saturation_pass": bool(saturation_pass),
            "action_sensitive_pass": bool(action_sensitive_pass),
            "guard_pass": bool(guard_pass),
        },
        "score": {
            "lqr_minus_best_baseline_return": float(lqr_full - baseline_full),
            "lqr_minus_best_baseline_suffix_return": float(lqr_suffix - baseline_suffix),
            "suffix_energy_over_prefix": float(lqr_energy_suffix / max(1e-9, lqr_energy_prefix)),
            "suffix_energy_over_best_baseline_suffix": float(
                lqr_energy_suffix / max(1e-9, best_baseline_suffix_energy)
            ),
        },
    }


def _summary(values: list[float]) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"count": 0, "finite": True, "mean": None, "std": None, "min": None, "max": None}
    finite = np.isfinite(arr)
    arr_f = arr[finite]
    return {
        "count": int(arr.size),
        "finite": bool(np.all(finite)),
        "mean": float(np.mean(arr_f)) if arr_f.size else None,
        "std": float(np.std(arr_f)) if arr_f.size else None,
        "min": float(np.min(arr_f)) if arr_f.size else None,
        "max": float(np.max(arr_f)) if arr_f.size else None,
    }


def _bin_count(values: list[float], edges: list[float]) -> int:
    arr = np.asarray(values, dtype=np.float64)
    bins = np.digitize(arr, np.asarray(edges, dtype=np.float64), right=False)
    return int(len(set(int(v) for v in bins.tolist())))


def _aggregate(rows: list[dict[str, Any]], *, args: argparse.Namespace) -> dict[str, Any]:
    n = max(1, len(rows))
    condition_counts = {
        key: int(sum(bool(row["conditions"][key]) for row in rows))
        for key in (
            "analytic_pass",
            "witness_reward_pass",
            "action_decay_pass",
            "settle_pass",
            "saturation_pass",
            "action_sensitive_pass",
            "guard_pass",
        )
    }
    params = {
        "dt": [float(row["env"]["dt"]) for row in rows],
        "rho": [float(row["env"]["rho"]) for row in rows],
        "abs_beta": [abs(float(row["env"]["beta"])) for row in rows],
        "q_weight": [float(row["env"]["q_weight"]) for row in rows],
        "v_weight": [float(row["env"]["v_weight"]) for row in rows],
        "ctrl_weight": [float(row["env"]["ctrl_weight"]) for row in rows],
        "q0": [float(row["env"]["q0"]) for row in rows],
        "v0": [float(row["env"]["v0"]) for row in rows],
    }
    diversity = {
        "rho_bins": _bin_count(params["rho"], [0.89, 0.93, 0.96]),
        "beta_bins": _bin_count(params["abs_beta"], [0.16, 0.24, 0.30]),
        "ctrl_bins": _bin_count(params["ctrl_weight"], [0.025, 0.045, 0.065]),
        "q0_signs": int(len(set(1 if v >= 0 else -1 for v in params["q0"]))),
        "v0_signs": int(len(set(1 if v >= 0 else -1 for v in params["v0"]))),
    }
    diversity_pass = bool(
        diversity["rho_bins"] >= 3
        and diversity["beta_bins"] >= 3
        and diversity["ctrl_bins"] >= 3
        and diversity["q0_signs"] == 2
        and diversity["v0_signs"] == 2
    )
    guard_ratio = float(condition_counts["guard_pass"] / n)
    count_safe = bool(
        condition_counts["guard_pass"] >= int(args.min_guard_count)
        and guard_ratio >= float(args.min_guard_ratio)
    )
    return {
        "n_envs": int(len(rows)),
        "condition_counts": condition_counts,
        "condition_ratios": {key: float(value / n) for key, value in condition_counts.items()},
        "count_safe": count_safe,
        "required_min_guard_count": int(args.min_guard_count),
        "required_min_guard_ratio": float(args.min_guard_ratio),
        "diversity": diversity,
        "diversity_pass": diversity_pass,
        "parameter_summary": {key: _summary(values) for key, values in params.items()},
        "score_summary": {
            "lqr_minus_best_baseline_return": _summary(
                [float(row["score"]["lqr_minus_best_baseline_return"]) for row in rows]
            ),
            "lqr_minus_best_baseline_suffix_return": _summary(
                [float(row["score"]["lqr_minus_best_baseline_suffix_return"]) for row in rows]
            ),
            "suffix_energy_over_prefix": _summary(
                [float(row["score"]["suffix_energy_over_prefix"]) for row in rows]
            ),
            "suffix_energy_over_best_baseline_suffix": _summary(
                [float(row["score"]["suffix_energy_over_best_baseline_suffix"]) for row in rows]
            ),
            "closed_loop_radius": _summary([float(row["closed_loop_radius"]) for row in rows]),
            "lqr_suffix_action_abs": _summary([float(row["lqr_action_abs"]["suffix"]) for row in rows]),
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    envs = _sample_envs(int(args.n_envs), int(args.seed))
    rows = [
        _one_env_report(env, n_steps=int(args.n_steps), seed=int(args.seed) + idx * 17, args=args)
        for idx, env in enumerate(envs)
    ]
    aggregate = _aggregate(rows, args=args)
    decision = {
        "specimen_preflight_pass": bool(aggregate["count_safe"] and aggregate["diversity_pass"]),
        "should_modify_exact_scm_now": False,
        "next_action": (
            "Promote to an exploratory generator wrapper only if count_safe and diversity_pass are true; "
            "otherwise tune this specimen family before touching any milestone path."
        ),
        "read": (
            "The simplified mechanism is internally viable as a specimen."
            if bool(aggregate["count_safe"] and aggregate["diversity_pass"])
            else "The simplified mechanism is not yet viable; do not integrate it."
        ),
    }
    report = {
        "analysis_entry": "phase2_pendulum_quadratic_settle_specimen_preflight",
        "schema": "phase2_pendulum_quadratic_settle_specimen_preflight.v1",
        "contract": {
            "exploratory_only": True,
            "does_not_use_exact_scm": True,
            "does_not_modify_exact_scm": True,
            "does_not_modify_fit_model": True,
            "does_not_modify_ppo": True,
            "no_training": True,
            "not_a_raw_delta_gate": True,
            "test_is_for_structural_falsification": True,
        },
        "mechanism": {
            "id": "controllable_quadratic_settle_factor",
            "dynamics": "z[t+1] = A z[t] + B a[t] + eps[t]",
            "reward": "1 - z[t+1]^T Q z[t+1] - lambda * a[t]^2",
            "witness": "bounded discrete LQR feedback a=-Kz",
        },
        "config": {
            "seed": int(args.seed),
            "n_envs": int(args.n_envs),
            "n_steps": int(args.n_steps),
            "split_frac": float(args.split_frac),
        },
        "aggregate": aggregate,
        "decision": decision,
        "per_env": rows if bool(args.include_per_env) else rows[: int(args.per_env_preview)],
    }
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "pendulum_quadratic_settle_specimen_preflight.json"
    md_path = out_dir / "pendulum_quadratic_settle_specimen_preflight.md"
    report["outputs"] = {"json": str(json_path), "markdown": str(md_path)}
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({"decision": decision, "outputs": report["outputs"]}, indent=2, sort_keys=True))
    return report


def _markdown(report: dict[str, Any]) -> str:
    agg = report["aggregate"]
    d = report["decision"]
    lines = [
        "# Pendulum Quadratic Settle Specimen Preflight",
        "",
        f"- specimen preflight pass: `{d['specimen_preflight_pass']}`",
        f"- should modify Exact SCM now: `{d['should_modify_exact_scm_now']}`",
        f"- read: {d['read']}",
        "",
        "## Contract",
        "",
        "- exploratory specimen only",
        "- no Exact SCM, fit_model, or PPO modification",
        "- no training",
        "- not a raw-delta gate",
        "",
        "## Aggregate",
        "",
        f"- n envs: `{agg['n_envs']}`",
        f"- count safe: `{agg['count_safe']}`",
        f"- diversity pass: `{agg['diversity_pass']}`",
        "",
        "## Condition Counts",
        "",
        "| condition | count | ratio |",
        "|---|---:|---:|",
    ]
    for key, count in agg["condition_counts"].items():
        lines.append(f"| `{key}` | {count} | {agg['condition_ratios'][key]:.3f} |")
    lines.extend(["", "## Diversity", ""])
    for key, value in agg["diversity"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Score Summary", ""])
    for key, value in agg["score_summary"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Next Action", "", d["next_action"], "", "## Files", ""])
    for key, value in report.get("outputs", {}).items():
        lines.append(f"- `{key}`: `{value}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--seed", type=int, default=20260512)
    parser.add_argument("--n-envs", type=int, default=256)
    parser.add_argument("--n-steps", type=int, default=200)
    parser.add_argument("--split-frac", type=float, default=0.5)
    parser.add_argument("--closed-loop-radius-max", type=float, default=0.995)
    parser.add_argument("--abs-reward-margin", type=float, default=5.0)
    parser.add_argument("--rel-reward-margin", type=float, default=0.03)
    parser.add_argument("--abs-suffix-margin", type=float, default=1.0)
    parser.add_argument("--rel-suffix-margin", type=float, default=0.03)
    parser.add_argument("--action-decay-margin", type=float, default=0.03)
    parser.add_argument("--self-energy-rel-max", type=float, default=0.70)
    parser.add_argument("--open-energy-rel-max", type=float, default=0.70)
    parser.add_argument("--saturation-max", type=float, default=0.25)
    parser.add_argument("--action-sensitive-delta-min", type=float, default=0.05)
    parser.add_argument("--min-guard-count", type=int, default=64)
    parser.add_argument("--min-guard-ratio", type=float, default=0.25)
    parser.add_argument("--include-per-env", action="store_true")
    parser.add_argument("--per-env-preview", type=int, default=16)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
