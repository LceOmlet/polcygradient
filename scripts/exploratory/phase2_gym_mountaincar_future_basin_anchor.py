#!/usr/bin/env python
"""Gym anchor for the MountainCar future-basin profile item.

Exploratory/read-only.  This script does not train, sample prior environments,
or modify the generator.  It asks whether the detector concept has a clear Gym
reference case:

* upfront investment: action is large in a prefix and near zero afterwards;
* future basin payoff: suffix reward improves after that early action;
* full payoff: the full episode return repays the early control cost.

The anchor is deliberately small and deterministic so it can be used as a
health guard before any MountainCar-like quota is considered.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
import numpy as np


ART = Path("/home/chen/RLPFN/artifacts")
DEFAULT_OUTPUT_DIR = ART / "phase2_gym_mountaincar_future_basin_anchor_0511"


@dataclass(frozen=True)
class Rollout:
    mode: str
    seed: int
    total_return: float
    prefix_return: float
    suffix_return: float
    prefix_action_abs_mean: float
    suffix_action_abs_mean: float
    episode_len: int
    terminated: bool
    truncated: bool
    max_position: float
    max_velocity: float


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    return value


def _stats(values: list[float]) -> dict[str, Any]:
    arr = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=np.float64)
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "min": float(np.min(arr)),
        "q10": float(np.quantile(arr, 0.10)),
        "q50": float(np.quantile(arr, 0.50)),
        "q90": float(np.quantile(arr, 0.90)),
        "max": float(np.max(arr)),
        "positive_fraction": float(np.mean(arr > 0.0)),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _velocity_pump(obs: np.ndarray) -> float:
    velocity = float(obs[1])
    return 1.0 if velocity >= 0.0 else -1.0


def _make_policy(mode: str, *, prefix_steps: int) -> Callable[[np.ndarray, int], float]:
    def zero(obs: np.ndarray, step: int) -> float:
        return 0.0

    def velocity_pump(obs: np.ndarray, step: int) -> float:
        return _velocity_pump(obs)

    def early_velocity_pump(obs: np.ndarray, step: int) -> float:
        return _velocity_pump(obs) if step < prefix_steps else 0.0

    def early_pos(obs: np.ndarray, step: int) -> float:
        return 1.0 if step < prefix_steps else 0.0

    def early_neg(obs: np.ndarray, step: int) -> float:
        return -1.0 if step < prefix_steps else 0.0

    def early_neg_pos(obs: np.ndarray, step: int) -> float:
        if step >= prefix_steps:
            return 0.0
        return -1.0 if step < prefix_steps // 2 else 1.0

    def early_pos_neg(obs: np.ndarray, step: int) -> float:
        if step >= prefix_steps:
            return 0.0
        return 1.0 if step < prefix_steps // 2 else -1.0

    policies = {
        "zero": zero,
        "velocity_pump": velocity_pump,
        "early_velocity_pump": early_velocity_pump,
        "early_pos": early_pos,
        "early_neg": early_neg,
        "early_neg_pos": early_neg_pos,
        "early_pos_neg": early_pos_neg,
    }
    if mode not in policies:
        raise KeyError(f"unknown MountainCar anchor mode: {mode}")
    return policies[mode]


def _rollout(mode: str, *, seed: int, prefix_steps: int, max_steps: int) -> Rollout:
    env = gym.make("MountainCarContinuous-v0")
    obs, _ = env.reset(seed=int(seed))
    policy = _make_policy(mode, prefix_steps=prefix_steps)
    rewards: list[float] = []
    actions: list[float] = []
    positions: list[float] = []
    velocities: list[float] = []
    terminated = False
    truncated = False
    for step in range(max_steps):
        action_value = float(policy(np.asarray(obs, dtype=np.float64), step))
        obs, reward, terminated, truncated, _ = env.step(np.asarray([action_value], dtype=np.float32))
        rewards.append(float(reward))
        actions.append(abs(action_value))
        positions.append(float(obs[0]))
        velocities.append(float(abs(obs[1])))
        if terminated or truncated:
            break
    env.close()
    prefix_rewards = rewards[:prefix_steps]
    suffix_rewards = rewards[prefix_steps:]
    prefix_actions = actions[:prefix_steps]
    suffix_actions = actions[prefix_steps:]
    return Rollout(
        mode=mode,
        seed=int(seed),
        total_return=float(sum(rewards)),
        prefix_return=float(sum(prefix_rewards)),
        suffix_return=float(sum(suffix_rewards)),
        prefix_action_abs_mean=float(np.mean(prefix_actions)) if prefix_actions else 0.0,
        suffix_action_abs_mean=float(np.mean(suffix_actions)) if suffix_actions else 0.0,
        episode_len=len(rewards),
        terminated=bool(terminated),
        truncated=bool(truncated),
        max_position=float(max(positions)) if positions else float("nan"),
        max_velocity=float(max(velocities)) if velocities else float("nan"),
    )


def _as_row(item: Rollout) -> dict[str, Any]:
    return {
        "mode": item.mode,
        "seed": item.seed,
        "total_return": item.total_return,
        "prefix_return": item.prefix_return,
        "suffix_return": item.suffix_return,
        "prefix_action_abs_mean": item.prefix_action_abs_mean,
        "suffix_action_abs_mean": item.suffix_action_abs_mean,
        "episode_len": item.episode_len,
        "terminated": item.terminated,
        "truncated": item.truncated,
        "max_position": item.max_position,
        "max_velocity": item.max_velocity,
    }


def _summarize_mode(rows: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    subset = [row for row in rows if row["mode"] == mode]
    return {
        "mode": mode,
        "n": len(subset),
        "total_return": _stats([float(row["total_return"]) for row in subset]),
        "prefix_return": _stats([float(row["prefix_return"]) for row in subset]),
        "suffix_return": _stats([float(row["suffix_return"]) for row in subset]),
        "episode_len": _stats([float(row["episode_len"]) for row in subset]),
        "terminated_rate": float(np.mean([bool(row["terminated"]) for row in subset])) if subset else 0.0,
        "prefix_action_abs_mean": _stats([float(row["prefix_action_abs_mean"]) for row in subset]),
        "suffix_action_abs_mean": _stats([float(row["suffix_action_abs_mean"]) for row in subset]),
    }


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    modes = [
        "zero",
        "velocity_pump",
        "early_velocity_pump",
        "early_pos",
        "early_neg",
        "early_neg_pos",
        "early_pos_neg",
    ]
    seeds = list(range(int(args.seed_start), int(args.seed_start) + int(args.n_seeds)))
    rollouts = [
        _rollout(mode, seed=seed, prefix_steps=int(args.prefix_steps), max_steps=int(args.max_steps))
        for mode in modes
        for seed in seeds
    ]
    rows = [_as_row(item) for item in rollouts]
    by_mode = {mode: _summarize_mode(rows, mode) for mode in modes}
    zero_by_seed = {row["seed"]: row for row in rows if row["mode"] == "zero"}
    open_by_seed = {row["seed"]: row for row in rows if row["mode"] == "velocity_pump"}

    per_seed: list[dict[str, Any]] = []
    for seed in seeds:
        candidates = [
            row
            for row in rows
            if row["seed"] == seed
            and row["mode"].startswith("early_")
            and float(row["prefix_action_abs_mean"]) >= float(args.min_prefix_action_abs)
            and float(row["suffix_action_abs_mean"]) <= float(args.max_suffix_action_abs)
        ]
        zero = zero_by_seed[seed]
        open_loop = open_by_seed[seed]

        scored_candidates: list[tuple[tuple[float, float, float], dict[str, Any], dict[str, Any]]] = []
        for row in candidates:
            suffix_gain_zero = float(row["suffix_return"]) - float(zero["suffix_return"])
            total_gain_zero = float(row["total_return"]) - float(zero["total_return"])
            suffix_gain_open = float(row["suffix_return"]) - float(open_loop["suffix_return"])
            total_gain_open = float(row["total_return"]) - float(open_loop["total_return"])
            anchored = (
                suffix_gain_zero >= float(args.min_suffix_gain_vs_zero)
                and total_gain_zero >= float(args.min_total_gain_vs_zero)
                and float(row["prefix_return"]) < 0.0
            )
            metrics = {
                "suffix_gain_vs_zero": suffix_gain_zero,
                "total_gain_vs_zero": total_gain_zero,
                "suffix_gain_vs_open": suffix_gain_open,
                "total_gain_vs_open": total_gain_open,
                "anchored": anchored,
            }
            score = (1.0 if anchored else 0.0, suffix_gain_zero, float(row["total_return"]))
            scored_candidates.append((score, row, metrics))
        _, best, best_metrics = max(scored_candidates, key=lambda item: item[0])
        suffix_gain_zero = float(best_metrics["suffix_gain_vs_zero"])
        total_gain_zero = float(best_metrics["total_gain_vs_zero"])
        suffix_gain_open = float(best_metrics["suffix_gain_vs_open"])
        total_gain_open = float(best_metrics["total_gain_vs_open"])
        anchored = bool(best_metrics["anchored"])
        strong_vs_open = suffix_gain_open >= float(args.min_suffix_gain_vs_open)
        per_seed.append(
            {
                "seed": seed,
                "best_upfront_mode": best["mode"],
                "best_total_return": best["total_return"],
                "best_prefix_return": best["prefix_return"],
                "best_suffix_return": best["suffix_return"],
                "zero_total_return": zero["total_return"],
                "zero_suffix_return": zero["suffix_return"],
                "open_total_return": open_loop["total_return"],
                "open_suffix_return": open_loop["suffix_return"],
                "suffix_gain_vs_zero": suffix_gain_zero,
                "total_gain_vs_zero": total_gain_zero,
                "suffix_gain_vs_open": suffix_gain_open,
                "total_gain_vs_open": total_gain_open,
                "terminated": best["terminated"],
                "episode_len": best["episode_len"],
                "gym_mountaincar_future_basin_anchor": bool(anchored),
                "gym_mountaincar_strong_vs_open": bool(strong_vs_open),
            }
        )

    anchor_rate = float(np.mean([row["gym_mountaincar_future_basin_anchor"] for row in per_seed]))
    strong_rate = float(np.mean([row["gym_mountaincar_strong_vs_open"] for row in per_seed]))
    report = {
        "analysis_entry": "phase2_gym_mountaincar_future_basin_anchor",
        "contract": {
            "exploratory_only": True,
            "gym_anchor_only": True,
            "does_not_train": True,
            "does_not_sample_prior_envs": True,
            "does_not_modify_exact_scm": True,
            "does_not_modify_fit_model": True,
            "does_not_modify_ppo": True,
            "not_quota_ready_by_itself": True,
        },
        "definition": {
            "upfront_investment": "large action in the prefix and near-zero action in the suffix",
            "future_basin_payoff": "suffix reward after early action improves over zero",
            "full_payoff": "total return repays prefix control cost and beats zero",
            "anchor": "best early policy has negative prefix reward, positive suffix/total gains vs zero",
        },
        "config": vars(args),
        "mode_summaries": by_mode,
        "per_seed_summary": {
            "n": len(per_seed),
            "anchor_rate": anchor_rate,
            "strong_vs_open_rate": strong_rate,
            "suffix_gain_vs_zero": _stats([float(row["suffix_gain_vs_zero"]) for row in per_seed]),
            "total_gain_vs_zero": _stats([float(row["total_gain_vs_zero"]) for row in per_seed]),
            "suffix_gain_vs_open": _stats([float(row["suffix_gain_vs_open"]) for row in per_seed]),
            "total_gain_vs_open": _stats([float(row["total_gain_vs_open"]) for row in per_seed]),
            "best_mode_counts": {
                mode: int(sum(row["best_upfront_mode"] == mode for row in per_seed))
                for mode in sorted({row["best_upfront_mode"] for row in per_seed})
            },
        },
        "read": {
            "gym_anchor_pass": anchor_rate >= float(args.min_anchor_rate),
            "quota_ready": False,
            "why_not_quota_ready": (
                "This only anchors the detector against Gym MountainCar.  Prior fixed-list health/diversity "
                "and selector integration are still required before using it as a quota."
            ),
        },
    }
    per_mode_csv = out_dir / "gym_mountaincar_anchor_rollouts.csv"
    per_seed_csv = out_dir / "gym_mountaincar_anchor_per_seed.csv"
    report_json = out_dir / "gym_mountaincar_future_basin_anchor_report.json"
    report_md = out_dir / "gym_mountaincar_future_basin_anchor.md"
    _write_csv(per_mode_csv, rows)
    _write_csv(per_seed_csv, per_seed)
    report["outputs"] = {
        "rollouts_csv": str(per_mode_csv),
        "per_seed_csv": str(per_seed_csv),
        "report_json": str(report_json),
        "summary_md": str(report_md),
    }
    report_json.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Gym MountainCar Future-Basin Anchor",
        "",
        "Read-only Gym counterfactual anchor; not a prior quota by itself.",
        "",
        f"- prefix steps: `{args.prefix_steps}`",
        f"- seeds: `{args.n_seeds}` from `{args.seed_start}`",
        f"- anchor pass: `{report['read']['gym_anchor_pass']}`",
        f"- anchor rate: `{anchor_rate:.3f}`",
        f"- strong-vs-open rate: `{strong_rate:.3f}`",
        "",
        "| mode | return mean | suffix mean | prefix-action | suffix-action | terminated |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in modes:
        item = by_mode[mode]
        lines.append(
            f"| `{mode}` | {item['total_return'].get('mean')} | {item['suffix_return'].get('mean')} | "
            f"{item['prefix_action_abs_mean'].get('mean')} | {item['suffix_action_abs_mean'].get('mean')} | "
            f"{item['terminated_rate']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Decision: Gym anchoring only.  It can upgrade MountainCar detector trust, but quota needs prior fixed-list guard.",
            "",
            f"- JSON: `{report_json}`",
            f"- per-seed CSV: `{per_seed_csv}`",
            f"- rollouts CSV: `{per_mode_csv}`",
        ]
    )
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report["outputs"], indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--n-seeds", type=int, default=64)
    parser.add_argument("--prefix-steps", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=999)
    parser.add_argument("--min-prefix-action-abs", type=float, default=0.5)
    parser.add_argument("--max-suffix-action-abs", type=float, default=0.05)
    parser.add_argument("--min-suffix-gain-vs-zero", type=float, default=50.0)
    parser.add_argument("--min-total-gain-vs-zero", type=float, default=50.0)
    parser.add_argument("--min-suffix-gain-vs-open", type=float, default=0.0)
    parser.add_argument("--min-anchor-rate", type=float, default=0.8)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
