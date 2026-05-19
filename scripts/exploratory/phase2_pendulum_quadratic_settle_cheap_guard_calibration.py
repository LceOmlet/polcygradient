#!/usr/bin/env python
"""Calibrate a cheap guard for the quadratic settle specimen.

This is an offline calibration tool.  It compares simple analytic labels
against the full temporal specimen audit from
phase2_pendulum_quadratic_settle_specimen_preflight.py.

The intended split is:

* full temporal/LQR audit: offline profile calibration only;
* cheap guard: candidate generator-level predicate or cached label source.

No Exact SCM, fit_model, PPO, or milestone path is modified here.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from scripts.exploratory import phase2_pendulum_quadratic_settle_specimen_preflight as specimen


DEFAULT_OUTPUT_DIR = Path("/home/chen/RLPFN/artifacts/phase2_pendulum_quadratic_settle_cheap_guard_calibration_0512")


@dataclass(frozen=True)
class CheapGuardConfig:
    radius_max: float
    initial_action_ratio_max: float
    initial_action_abs_min: float
    first_step_delta_v_min: float
    initial_energy_min: float


@dataclass(frozen=True)
class ShortTemporalGuardConfig:
    radius_max: float
    reward_margin_min: float
    suffix_margin_min: float
    action_decay_min: float
    suffix_energy_rel_max: float
    first_step_delta_v_min: float
    saturation_max: float


def _initial_energy(env: specimen.QuadraticSettleEnv) -> float:
    z = np.asarray([env.q0, env.v0], dtype=np.float64)
    return float(z.T @ env.Q @ z)


def _cheap_features(env: specimen.QuadraticSettleEnv) -> dict[str, float]:
    gain = specimen._lqr_gain(env)
    z = np.asarray([env.q0, env.v0], dtype=np.float64)
    raw_action = abs(float(np.dot(gain, z)))
    clipped_action = min(raw_action, float(env.action_limit))
    return {
        "rank": float(specimen._controllability_rank(env)),
        "radius": float(specimen._closed_loop_radius(env, gain)),
        "initial_action_abs": float(clipped_action),
        "initial_action_ratio": float(raw_action / max(1e-9, float(env.action_limit))),
        "first_step_delta_v": float(2.0 * abs(float(env.beta)) * clipped_action),
        "initial_energy": _initial_energy(env),
        "abs_beta": abs(float(env.beta)),
        "ctrl_weight": float(env.ctrl_weight),
        "rho": float(env.rho),
    }


def _cheap_guard(features: dict[str, float], cfg: CheapGuardConfig) -> bool:
    return bool(
        features["rank"] == 2.0
        and features["radius"] <= cfg.radius_max
        and features["initial_action_ratio"] <= cfg.initial_action_ratio_max
        and features["initial_action_abs"] >= cfg.initial_action_abs_min
        and features["first_step_delta_v"] >= cfg.first_step_delta_v_min
        and features["initial_energy"] >= cfg.initial_energy_min
    )


def _short_temporal_features(
    env: specimen.QuadraticSettleEnv,
    *,
    n_steps: int,
    split_frac: float,
    seed: int,
) -> dict[str, float]:
    # Use a noiseless copy: the guard is meant to describe the controlled
    # mechanism, while stochasticity remains an offline audit concern.
    clean_env = replace(env, noise_std=0.0)
    gain = specimen._lqr_gain(clean_env)
    split_step = int(max(1, min(int(n_steps) - 1, round(float(split_frac) * float(n_steps)))))
    lqr = specimen._simulate(clean_env, policy="lqr", gain=gain, n_steps=int(n_steps), seed=int(seed))
    zero = specimen._simulate(clean_env, policy="zero", gain=gain, n_steps=int(n_steps), seed=int(seed) + 1009)
    lqr_rewards = lqr["rewards"]
    zero_rewards = zero["rewards"]
    lqr_actions = np.abs(lqr["actions"])
    lqr_energy = specimen._state_energy(clean_env, lqr["states"][1:])
    lqr_prefix_action = float(np.mean(lqr_actions[:split_step]))
    lqr_suffix_action = float(np.mean(lqr_actions[split_step:]))
    lqr_prefix_energy = float(np.mean(lqr_energy[:split_step]))
    lqr_suffix_energy = float(np.mean(lqr_energy[split_step:]))
    z = np.asarray([clean_env.q0, clean_env.v0], dtype=np.float64)
    first_action = min(abs(float(np.dot(gain, z))), float(clean_env.action_limit))
    return {
        "rank": float(specimen._controllability_rank(clean_env)),
        "radius": float(specimen._closed_loop_radius(clean_env, gain)),
        "lqr_minus_zero_return": float(np.sum(lqr_rewards) - np.sum(zero_rewards)),
        "lqr_minus_zero_suffix_return": float(np.sum(lqr_rewards[split_step:]) - np.sum(zero_rewards[split_step:])),
        "action_decay": float(lqr_prefix_action - lqr_suffix_action),
        "suffix_energy_over_prefix": float(lqr_suffix_energy / max(1e-9, lqr_prefix_energy)),
        "first_step_delta_v": float(2.0 * abs(float(clean_env.beta)) * first_action),
        "saturation": float(np.mean(lqr_actions >= 0.98 * float(clean_env.action_limit))),
    }


def _short_temporal_guard(features: dict[str, float], cfg: ShortTemporalGuardConfig) -> bool:
    return bool(
        features["rank"] == 2.0
        and features["radius"] <= cfg.radius_max
        and features["lqr_minus_zero_return"] >= cfg.reward_margin_min
        and features["lqr_minus_zero_suffix_return"] >= cfg.suffix_margin_min
        and features["action_decay"] >= cfg.action_decay_min
        and features["suffix_energy_over_prefix"] <= cfg.suffix_energy_rel_max
        and features["first_step_delta_v"] >= cfg.first_step_delta_v_min
        and features["saturation"] <= cfg.saturation_max
    )


def _confusion(rows: list[dict[str, Any]], cfg: CheapGuardConfig) -> dict[str, Any]:
    tp = fp = tn = fn = 0
    for row in rows:
        pred = _cheap_guard(row["cheap_features"], cfg)
        truth = bool(row["full_guard_pass"])
        if pred and truth:
            tp += 1
        elif pred and not truth:
            fp += 1
        elif not pred and truth:
            fn += 1
        else:
            tn += 1
    n = max(1, len(rows))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    predicted_ratio = (tp + fp) / n
    truth_ratio = (tp + fn) / n
    f1 = 2.0 * precision * recall / max(1e-9, precision + recall)
    return {
        "config": asdict(cfg),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "precision": float(precision),
        "recall": float(recall),
        "predicted_ratio": float(predicted_ratio),
        "truth_ratio": float(truth_ratio),
        "f1": float(f1),
    }


def _short_confusion(rows: list[dict[str, Any]], cfg: ShortTemporalGuardConfig) -> dict[str, Any]:
    tp = fp = tn = fn = 0
    for row in rows:
        pred = _short_temporal_guard(row["short_temporal_features"], cfg)
        truth = bool(row["full_guard_pass"])
        if pred and truth:
            tp += 1
        elif pred and not truth:
            fp += 1
        elif not pred and truth:
            fn += 1
        else:
            tn += 1
    n = max(1, len(rows))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    predicted_ratio = (tp + fp) / n
    truth_ratio = (tp + fn) / n
    f1 = 2.0 * precision * recall / max(1e-9, precision + recall)
    return {
        "config": asdict(cfg),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "precision": float(precision),
        "recall": float(recall),
        "predicted_ratio": float(predicted_ratio),
        "truth_ratio": float(truth_ratio),
        "f1": float(f1),
    }


def _candidate_grid() -> list[CheapGuardConfig]:
    candidates: list[CheapGuardConfig] = []
    for radius_max in (0.90, 0.93, 0.96, 0.98):
        for initial_action_ratio_max in (0.75, 1.0, 1.25, 1.5, 2.0):
            for initial_action_abs_min in (0.01, 0.03, 0.06, 0.10):
                for first_step_delta_v_min in (0.03, 0.05, 0.08, 0.12):
                    for initial_energy_min in (0.01, 0.03, 0.06, 0.10):
                        candidates.append(
                            CheapGuardConfig(
                                radius_max=radius_max,
                                initial_action_ratio_max=initial_action_ratio_max,
                                initial_action_abs_min=initial_action_abs_min,
                                first_step_delta_v_min=first_step_delta_v_min,
                                initial_energy_min=initial_energy_min,
                            )
                        )
    return candidates


def _short_candidate_grid() -> list[ShortTemporalGuardConfig]:
    candidates: list[ShortTemporalGuardConfig] = []
    for radius_max in (0.90, 0.93, 0.96, 0.98, 0.995):
        for reward_margin_min in (0.0, 1.0, 3.0, 5.0, 10.0, 20.0):
            for suffix_margin_min in (0.0, 0.5, 1.0, 2.0, 5.0, 10.0):
                for action_decay_min in (0.005, 0.015, 0.03, 0.05, 0.08):
                    for suffix_energy_rel_max in (0.25, 0.4, 0.55, 0.7, 0.85):
                        for first_step_delta_v_min in (0.03, 0.05, 0.08, 0.12, 0.18):
                            candidates.append(
                                ShortTemporalGuardConfig(
                                    radius_max=radius_max,
                                    reward_margin_min=reward_margin_min,
                                    suffix_margin_min=suffix_margin_min,
                                    action_decay_min=action_decay_min,
                                    suffix_energy_rel_max=suffix_energy_rel_max,
                                    first_step_delta_v_min=first_step_delta_v_min,
                                    saturation_max=0.25,
                                )
                            )
    return candidates


def _select_best(scored: list[dict[str, Any]], *, args: argparse.Namespace) -> dict[str, Any]:
    feasible = [
        item
        for item in scored
        if item["precision"] >= float(args.min_precision)
        and item["predicted_ratio"] >= float(args.min_predicted_ratio)
    ]
    pool = feasible if feasible else scored
    return max(
        pool,
        key=lambda item: (
            item["f1"],
            item["precision"],
            item["recall"],
            -abs(item["predicted_ratio"] - item["truth_ratio"]),
        ),
    )


def _summary(values: list[float]) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)) if arr.size else None,
        "std": float(np.std(arr)) if arr.size else None,
        "min": float(np.min(arr)) if arr.size else None,
        "max": float(np.max(arr)) if arr.size else None,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    full_args = argparse.Namespace(
        split_frac=float(args.split_frac),
        closed_loop_radius_max=0.995,
        abs_reward_margin=5.0,
        rel_reward_margin=0.03,
        abs_suffix_margin=1.0,
        rel_suffix_margin=0.03,
        action_decay_margin=0.03,
        self_energy_rel_max=0.70,
        open_energy_rel_max=0.70,
        saturation_max=0.25,
        action_sensitive_delta_min=0.05,
    )
    envs = specimen._sample_envs(int(args.n_envs), int(args.seed))
    rows: list[dict[str, Any]] = []
    for idx, env in enumerate(envs):
        full = specimen._one_env_report(
            env,
            n_steps=int(args.n_steps),
            seed=int(args.seed) + idx * 17,
            args=full_args,
        )
        rows.append(
            {
                "env": asdict(env),
                "cheap_features": _cheap_features(env),
                "short_temporal_features": _short_temporal_features(
                    env,
                    n_steps=int(args.short_n_steps),
                    split_frac=float(args.split_frac),
                    seed=int(args.seed) + idx * 17,
                ),
                "full_guard_pass": bool(full["conditions"]["guard_pass"]),
                "full_conditions": full["conditions"],
            }
        )

    scored = [_confusion(rows, cfg) for cfg in _candidate_grid()]
    best = _select_best(scored, args=args)
    short_scored = [_short_confusion(rows, cfg) for cfg in _short_candidate_grid()]
    best_short = _select_best(short_scored, args=args)
    selected_cfg = CheapGuardConfig(**best["config"])
    selected_rows = [row for row in rows if _cheap_guard(row["cheap_features"], selected_cfg)]
    short_selected_cfg = ShortTemporalGuardConfig(**best_short["config"])
    short_selected_rows = [
        row for row in rows if _short_temporal_guard(row["short_temporal_features"], short_selected_cfg)
    ]
    report = {
        "analysis_entry": "phase2_pendulum_quadratic_settle_cheap_guard_calibration",
        "schema": "phase2_pendulum_quadratic_settle_cheap_guard_calibration.v1",
        "contract": {
            "exploratory_only": True,
            "offline_full_audit_only": True,
            "cheap_guard_or_cached_label_for_generator_path": True,
            "does_not_modify_exact_scm": True,
            "does_not_modify_fit_model": True,
            "does_not_modify_ppo": True,
            "not_a_raw_delta_gate": True,
        },
        "config": {
            "seed": int(args.seed),
            "n_envs": int(args.n_envs),
            "n_steps": int(args.n_steps),
            "split_frac": float(args.split_frac),
            "short_n_steps": int(args.short_n_steps),
            "min_precision": float(args.min_precision),
            "min_predicted_ratio": float(args.min_predicted_ratio),
        },
        "static_analytic_best": best,
        "short_temporal_best": best_short,
        "selected_feature_summary": {
            key: _summary([float(row["cheap_features"][key]) for row in selected_rows])
            for key in (
                "radius",
                "initial_action_ratio",
                "initial_action_abs",
                "first_step_delta_v",
                "initial_energy",
                "abs_beta",
                "ctrl_weight",
                "rho",
            )
        },
        "short_selected_feature_summary": {
            key: _summary([float(row["short_temporal_features"][key]) for row in short_selected_rows])
            for key in (
                "radius",
                "lqr_minus_zero_return",
                "lqr_minus_zero_suffix_return",
                "action_decay",
                "suffix_energy_over_prefix",
                "first_step_delta_v",
                "saturation",
            )
        },
        "top10": sorted(scored, key=lambda item: (item["f1"], item["precision"], item["recall"]), reverse=True)[:10],
        "short_top10": sorted(
            short_scored,
            key=lambda item: (item["f1"], item["precision"], item["recall"]),
            reverse=True,
        )[:10],
        "decision": {
            "cheap_guard_viable": bool(
                best["precision"] >= float(args.min_precision)
                and best["predicted_ratio"] >= float(args.min_predicted_ratio)
            ),
            "short_temporal_guard_viable": bool(
                best_short["precision"] >= float(args.min_precision)
                and best_short["predicted_ratio"] >= float(args.min_predicted_ratio)
            ),
            "read": (
                "Short temporal guard is viable; static analytic guard remains an offline diagnostic."
                if best_short["precision"] >= float(args.min_precision)
                and best_short["predicted_ratio"] >= float(args.min_predicted_ratio)
                else "Neither cheap guard is calibrated enough; keep using cached full-audit labels for this profile."
            ),
        },
    }

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "pendulum_quadratic_settle_cheap_guard_calibration.json"
    md_path = out_dir / "pendulum_quadratic_settle_cheap_guard_calibration.md"
    report["outputs"] = {"json": str(json_path), "markdown": str(md_path)}
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "static_analytic_best": best,
                "short_temporal_best": best_short,
                "outputs": report["outputs"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return report


def _markdown(report: dict[str, Any]) -> str:
    best = report["static_analytic_best"]
    best_short = report["short_temporal_best"]
    lines = [
        "# Pendulum Quadratic Settle Cheap Guard Calibration",
        "",
        "## Contract",
        "",
        "- exploratory only",
        "- full temporal audit is offline calibration only",
        "- cheap guard or cached label is the only acceptable future generator path",
        "- no Exact SCM, fit_model, PPO, or milestone modification",
        "- not a raw-delta gate",
        "",
        "## Decision",
        "",
        f"- cheap guard viable: `{report['decision']['cheap_guard_viable']}`",
        f"- static analytic guard viable: `{report['decision']['cheap_guard_viable']}`",
        f"- short temporal guard viable: `{report['decision']['short_temporal_guard_viable']}`",
        f"- read: {report['decision']['read']}",
        "",
        "## Best Static Analytic Guard",
        "",
    ]
    for key, value in best["config"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Confusion Against Full Temporal Audit",
            "",
            f"- tp: `{best['tp']}`",
            f"- fp: `{best['fp']}`",
            f"- tn: `{best['tn']}`",
            f"- fn: `{best['fn']}`",
            f"- precision: `{best['precision']:.3f}`",
            f"- recall: `{best['recall']:.3f}`",
            f"- predicted ratio: `{best['predicted_ratio']:.3f}`",
            f"- full audit truth ratio: `{best['truth_ratio']:.3f}`",
            f"- f1: `{best['f1']:.3f}`",
            "",
            "## Best Short Temporal Guard",
            "",
        ]
    )
    for key, value in best_short["config"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Short Temporal Confusion Against Full Temporal Audit",
            "",
            f"- tp: `{best_short['tp']}`",
            f"- fp: `{best_short['fp']}`",
            f"- tn: `{best_short['tn']}`",
            f"- fn: `{best_short['fn']}`",
            f"- precision: `{best_short['precision']:.3f}`",
            f"- recall: `{best_short['recall']:.3f}`",
            f"- predicted ratio: `{best_short['predicted_ratio']:.3f}`",
            f"- full audit truth ratio: `{best_short['truth_ratio']:.3f}`",
            f"- f1: `{best_short['f1']:.3f}`",
            "",
            "## Selected Feature Summary",
            "",
        ]
    )
    for key, value in report["selected_feature_summary"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Short Temporal Selected Feature Summary", ""])
    for key, value in report["short_selected_feature_summary"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Files", ""])
    for key, value in report.get("outputs", {}).items():
        lines.append(f"- `{key}`: `{value}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--seed", type=int, default=20260512)
    parser.add_argument("--n-envs", type=int, default=512)
    parser.add_argument("--n-steps", type=int, default=200)
    parser.add_argument("--short-n-steps", type=int, default=80)
    parser.add_argument("--split-frac", type=float, default=0.5)
    parser.add_argument("--min-precision", type=float, default=0.80)
    parser.add_argument("--min-predicted-ratio", type=float, default=0.20)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
