#!/usr/bin/env python
"""Full temporal prior-side detector for sustained locomotion-like modes.

Read-only exploratory probe.  It does not mutate exact SCM defaults, fit_model,
PPO, checkpoints, or fixed-list inputs.

The detector is deliberately stricter than the earlier coverage proxy.  For each
fixed prior environment and matched rollout seed it compares several deterministic
action families and accepts a sustained-locomotion witness only when:

1. A nonzero action family is temporally persistent enough to be a sustained
   controller rather than one-step noise.
2. That action family improves environment reward versus zero action.
3. The improvement is not primarily explained by earlier/later terminal events.
4. The state trajectory changes nontrivially relative to zero action.

This is still not a generator repair.  It only estimates whether the current
milestone prior already contains a clean homologous family for Gym locomotion
tasks before we consider any balancing or parameter changes.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.exploratory.phase2_prior_action_mode_coverage_probe import (  # noqa: E402
    DEFAULT_FIXED_LISTS,
    ModeProbePolicy,
    _build_prior,
    _first_done_steps,
    _freeze_rollout_latents,
    _load_fixed_h_list,
    _mode_list,
    _seed_all,
)


DEFAULT_OUTPUT_DIR = (
    "/home/chen/RLPFN/artifacts/phase2_locomotion_full_temporal_detector_0510"
)


def _stats(values: np.ndarray | list[float]) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
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
    }


def _to_np_float(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy().astype(np.float32, copy=False)
    return np.asarray(x, dtype=np.float32)


def _to_np_bool(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy().astype(bool, copy=False)
    return np.asarray(x, dtype=bool)


def _run_mode_temporal(
    *,
    prior,
    h_list: list[dict[str, Any]],
    env_seeds: list[int],
    rollout_seeds: list[int],
    mode: str,
    args: argparse.Namespace,
    num_features: int,
    device: torch.device,
) -> dict[str, Any]:
    policy = ModeProbePolicy(mode, seed=int(args.seed), random_scale=float(args.random_action_scale))
    step_payloads: list[dict[str, np.ndarray]] = []

    def _step_sink(payload: dict[str, torch.Tensor]) -> bool:
        step_payloads.append(
            {
                "reward": _to_np_float(payload["reward"]),
                "reward_env": _to_np_float(payload["reward_env"]),
                "reward_ctrl": _to_np_float(payload["reward_ctrl"]),
                "reward_survival": _to_np_float(payload["reward_survival"]),
                "reward_terminal_bonus": _to_np_float(payload["reward_terminal_bonus"]),
                "dones": _to_np_bool(payload["dones"]),
                "action": _to_np_float(payload["action"]),
                "next_state": _to_np_float(payload["next_state"]),
                "next_state_mask": _to_np_bool(payload["next_state_mask"]),
            }
        )
        return True

    x_steps, rewards, _infos = prior._rollout_family_group_vectorized_with_policy(
        h_list,
        policy,
        int(args.n_steps),
        int(num_features),
        int(args.single_eval_pos),
        device,
        collect_x=True,
        collect_runtime_info=False,
        env_rng_seeds=list(env_seeds),
        rollout_rng_seeds=list(rollout_seeds),
        store_rewards=True,
        policy_objective_kind="reinforce",
        _policy_collect_log_probs=True,
        _policy_detach_action_in_env=False,
        _policy_force_no_grad=True,
        _policy_collect_ppo_trace=True,
        _policy_ppo_step_sink=_step_sink,
    )
    del x_steps
    if len(step_payloads) != int(args.n_steps):
        raise RuntimeError(f"{mode} emitted {len(step_payloads)} steps, expected {int(args.n_steps)}")

    reward_total = _to_np_float(rewards)
    reward_env = np.stack([p["reward_env"] for p in step_payloads], axis=0)
    reward_ctrl = np.stack([p["reward_ctrl"] for p in step_payloads], axis=0)
    reward_survival = np.stack([p["reward_survival"] for p in step_payloads], axis=0)
    reward_terminal = np.stack([p["reward_terminal_bonus"] for p in step_payloads], axis=0)
    dones = np.stack([p["dones"] for p in step_payloads], axis=0)
    actions = np.stack([p["action"] for p in step_payloads], axis=0)
    states = np.stack([p["next_state"] for p in step_payloads], axis=0)
    state_masks = np.stack([p["next_state_mask"] for p in step_payloads], axis=0).astype(np.float32, copy=False)
    states = states * state_masks

    action_abs_mean = np.mean(np.abs(actions), axis=(0, 2))
    action_norm = np.linalg.norm(actions, axis=2)
    if actions.shape[0] >= 2:
        action_delta = np.linalg.norm(np.diff(actions, axis=0), axis=2)
        action_delta_mean = np.mean(action_delta, axis=0)
        a0 = actions[:-1]
        a1 = actions[1:]
        denom = np.linalg.norm(a0, axis=2) * np.linalg.norm(a1, axis=2)
        cosine = np.sum(a0 * a1, axis=2) / np.maximum(denom, 1e-8)
        valid = denom > 1e-8
        action_temporal_cosine = np.where(valid, cosine, np.nan)
        action_temporal_cosine_mean = np.nanmean(action_temporal_cosine, axis=0)
    else:
        action_delta_mean = np.zeros((actions.shape[1],), dtype=np.float32)
        action_temporal_cosine_mean = np.zeros((actions.shape[1],), dtype=np.float32)

    if states.shape[0] >= 2:
        state_delta = np.linalg.norm(np.diff(states, axis=0), axis=2)
        state_path_l2 = np.sum(state_delta, axis=0)
        state_final_displacement = np.linalg.norm(states[-1] - states[0], axis=1)
    else:
        state_path_l2 = np.zeros((states.shape[1],), dtype=np.float32)
        state_final_displacement = np.zeros((states.shape[1],), dtype=np.float32)

    return {
        "mode": mode,
        "returns": {
            "total": np.sum(reward_total, axis=0),
            "env": np.sum(reward_env, axis=0),
            "ctrl": np.sum(reward_ctrl, axis=0),
            "survival": np.sum(reward_survival, axis=0),
            "terminal_bonus": np.sum(reward_terminal, axis=0),
        },
        "first_done_steps": _first_done_steps(dones),
        "done_count": np.sum(dones, axis=0),
        "action_abs_mean": action_abs_mean,
        "action_norm_mean": np.mean(action_norm, axis=0),
        "action_delta_mean": action_delta_mean,
        "action_temporal_cosine_mean": action_temporal_cosine_mean,
        "state_path_l2": state_path_l2,
        "state_final_displacement_l2": state_final_displacement,
    }


def _dynamic_margin(values: np.ndarray, *, abs_margin: float, rel_margin: float) -> np.ndarray:
    spread = np.nanmax(values, axis=0) - np.nanmin(values, axis=0)
    return np.maximum(float(abs_margin), float(rel_margin) * np.maximum(1.0, spread))


def _classify_temporal(mode_payloads: dict[str, dict[str, Any]], *, args: argparse.Namespace) -> dict[str, Any]:
    modes = list(mode_payloads.keys())
    n_envs = int(next(iter(mode_payloads.values()))["returns"]["total"].shape[0])
    total_stack = np.stack([mode_payloads[m]["returns"]["total"] for m in modes], axis=0)
    env_stack = np.stack([mode_payloads[m]["returns"]["env"] for m in modes], axis=0)
    margin = _dynamic_margin(env_stack, abs_margin=float(args.abs_margin), rel_margin=float(args.rel_margin))

    zero = mode_payloads["zero"]
    candidates = [m for m in modes if m != "zero"]
    per_env: list[dict[str, Any]] = []
    witness = np.zeros((n_envs,), dtype=bool)

    for env_idx in range(n_envs):
        best_mode = None
        best_score = -float("inf")
        best_row = None
        zero_env = float(zero["returns"]["env"][env_idx])
        zero_total = float(zero["returns"]["total"][env_idx])
        zero_first_done = float(zero["first_done_steps"][env_idx])
        zero_done = float(zero["done_count"][env_idx])
        zero_state_path = float(zero["state_path_l2"][env_idx])
        for mode in candidates:
            payload = mode_payloads[mode]
            action_abs = float(payload["action_abs_mean"][env_idx])
            action_cos = float(payload["action_temporal_cosine_mean"][env_idx])
            if not math.isfinite(action_cos):
                action_cos = 0.0
            env_gain = float(payload["returns"]["env"][env_idx]) - zero_env
            total_gain = float(payload["returns"]["total"][env_idx]) - zero_total
            first_done_delta = float(payload["first_done_steps"][env_idx]) - zero_first_done
            done_delta = float(payload["done_count"][env_idx]) - zero_done
            state_path_gain = float(payload["state_path_l2"][env_idx]) - zero_state_path
            terminal_invariant = bool(
                abs(done_delta) <= float(args.max_done_count_delta)
                and abs(first_done_delta) <= float(args.max_first_done_delta_frac) * float(args.n_steps)
            )
            sustained_action = bool(
                action_abs >= float(args.min_action_abs_mean)
                and action_cos >= float(args.min_action_temporal_cosine)
            )
            env_improves = bool(env_gain > float(margin[env_idx]))
            state_moves = bool(state_path_gain > float(args.min_state_path_gain))
            accepted = bool(sustained_action and env_improves and terminal_invariant and state_moves)
            score = env_gain
            row = {
                "mode": mode,
                "accepted": accepted,
                "env_reward_gain_vs_zero": env_gain,
                "total_reward_gain_vs_zero": total_gain,
                "first_done_delta_vs_zero": first_done_delta,
                "done_count_delta_vs_zero": done_delta,
                "action_abs_mean": action_abs,
                "action_temporal_cosine_mean": action_cos,
                "state_path_gain_vs_zero": state_path_gain,
                "terminal_invariant": terminal_invariant,
                "sustained_action": sustained_action,
                "env_improves": env_improves,
                "state_moves": state_moves,
            }
            if accepted and score > best_score:
                best_score = score
                best_mode = mode
                best_row = row
            elif best_mode is None and score > best_score:
                best_score = score
                best_mode = mode
                best_row = row
        accepted = bool(best_row is not None and best_row["accepted"])
        witness[env_idx] = accepted
        per_env.append(
            {
                "env_index": int(env_idx),
                "threshold": float(margin[env_idx]),
                "zero_env_return": zero_env,
                "zero_total_return": zero_total,
                "zero_first_done_step": zero_first_done,
                "zero_done_count": zero_done,
                "zero_state_path_l2": zero_state_path,
                "best_sustained_mode": best_mode,
                "sustained_locomotion_temporal_witness": accepted,
                **(best_row or {}),
            }
        )

    return {
        "coverage": {
            "n_envs": int(n_envs),
            "sustained_locomotion_temporal_witness_count": int(witness.sum()),
            "sustained_locomotion_temporal_witness_rate": float(np.mean(witness)),
        },
        "scores": {
            "accepted_env_reward_gain": _stats(
                [r["env_reward_gain_vs_zero"] for r in per_env if r["sustained_locomotion_temporal_witness"]]
            ),
            "all_best_env_reward_gain": _stats([r["env_reward_gain_vs_zero"] for r in per_env]),
            "all_best_action_abs_mean": _stats([r["action_abs_mean"] for r in per_env]),
            "all_best_action_temporal_cosine_mean": _stats([r["action_temporal_cosine_mean"] for r in per_env]),
            "all_best_state_path_gain": _stats([r["state_path_gain_vs_zero"] for r in per_env]),
        },
        "best_mode_counts": {
            mode: int(sum(1 for r in per_env if r["best_sustained_mode"] == mode))
            for mode in modes
        },
        "per_env": per_env,
    }


def _scan_list(
    *,
    label: str,
    path: Path,
    prior,
    args: argparse.Namespace,
    num_features: int,
    device: torch.device,
) -> dict[str, Any]:
    h_raw, seed_list = _load_fixed_h_list(path, limit=int(args.n_envs) if args.n_envs else None)
    h_list = _freeze_rollout_latents(prior, h_raw)
    env_seeds = [int(v) for v in seed_list] if seed_list is not None else [int(args.seed) + 10_000 + i for i in range(len(h_list))]
    rollout_seeds = [int(args.seed) + 20_000 + i for i in range(len(h_list))]
    modes = _mode_list(args)
    payloads = {}
    for mode in modes:
        print(f"[locomotion-temporal] {label} mode={mode} n_envs={len(h_list)} n_steps={int(args.n_steps)}", flush=True)
        payloads[mode] = _run_mode_temporal(
            prior=prior,
            h_list=h_list,
            env_seeds=env_seeds,
            rollout_seeds=rollout_seeds,
            mode=mode,
            args=args,
            num_features=int(num_features),
            device=device,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
    classification = _classify_temporal(payloads, args=args)
    return {
        "label": label,
        "fixed_h_list_json": str(path),
        "classification": classification,
        "mode_summaries": {
            mode: {
                "env_return": _stats(payload["returns"]["env"]),
                "total_return": _stats(payload["returns"]["total"]),
                "action_abs_mean": _stats(payload["action_abs_mean"]),
                "action_temporal_cosine_mean": _stats(payload["action_temporal_cosine_mean"]),
                "state_path_l2": _stats(payload["state_path_l2"]),
                "first_done_steps": _stats(payload["first_done_steps"]),
            }
            for mode, payload in payloads.items()
        },
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v for k, v in row.items()})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fixed-h-list-json", action="append", default=None)
    parser.add_argument("--seed", type=int, default=20260510)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-envs", type=int, default=64)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--single-eval-pos", type=int, default=0)
    parser.add_argument("--prior-milestone", default="gated_reward_path_balance")
    parser.add_argument("--random-action-scale", type=float, default=1.0)
    parser.add_argument("--obs-linear-count", type=int, default=4)
    parser.add_argument("--obs-linear-scales", default="0.5,1.0")
    parser.add_argument("--abs-margin", type=float, default=1.0)
    parser.add_argument("--rel-margin", type=float, default=0.10)
    parser.add_argument("--min-action-abs-mean", type=float, default=0.03)
    parser.add_argument("--min-action-temporal-cosine", type=float, default=0.25)
    parser.add_argument("--min-state-path-gain", type=float, default=1e-3)
    parser.add_argument("--max-done-count-delta", type=float, default=1.0)
    parser.add_argument("--max-first-done-delta-frac", type=float, default=0.10)
    parser.add_argument("--topology-state-gain-min", type=float, default=0.7)
    parser.add_argument("--topology-action-gain-min", type=float, default=0.06)
    parser.add_argument("--topology-state-to-action-ratio-max", type=float, default=10.0)
    parser.add_argument("--topology-max-attempts", type=int, default=4096)
    args = parser.parse_args()

    _seed_all(int(args.seed))
    device = torch.device(str(args.device))
    if device.type == "cuda" and int(device.index or 0) != 0:
        raise ValueError("This probe is constrained to GPU0 when CUDA is used; pass --device cuda:0 or cpu.")
    prior, num_features, _action_slot_dim = _build_prior(args)
    paths = [Path(p) for p in (args.fixed_h_list_json or DEFAULT_FIXED_LISTS)]
    reports = [
        _scan_list(
            label=path.parent.parent.name,
            path=path,
            prior=prior,
            args=args,
            num_features=int(num_features),
            device=device,
        )
        for path in paths
    ]
    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "analysis_entry": "phase2_locomotion_full_temporal_detector",
        "contract": {
            "exploratory_only": True,
            "does_not_modify_exact_scm": True,
            "does_not_modify_fit_model": True,
            "does_not_modify_ppo": True,
            "does_not_train": True,
            "same_h_list_env_seeds_rollout_seeds_across_modes": True,
        },
        "definition": {
            "sustained_locomotion_temporal_witness": (
                "nonzero temporally persistent action improves env reward over zero action, "
                "first_done/done_count remain approximately invariant, and state path length increases"
            ),
            "not_a_generator_repair": True,
        },
        "config": {
            "seed": int(args.seed),
            "device": str(device),
            "n_envs": int(args.n_envs),
            "n_steps": int(args.n_steps),
            "modes": _mode_list(args),
            "thresholds": {
                "abs_margin": float(args.abs_margin),
                "rel_margin": float(args.rel_margin),
                "min_action_abs_mean": float(args.min_action_abs_mean),
                "min_action_temporal_cosine": float(args.min_action_temporal_cosine),
                "min_state_path_gain": float(args.min_state_path_gain),
                "max_done_count_delta": float(args.max_done_count_delta),
                "max_first_done_delta_frac": float(args.max_first_done_delta_frac),
            },
        },
        "lists": reports,
    }
    report_path = out_dir / "locomotion_full_temporal_detector_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    flat_rows = []
    for item in reports:
        label = item["label"]
        for row in item["classification"]["per_env"]:
            flat = dict(row)
            flat["scope"] = label
            flat_rows.append(flat)
    _write_csv(out_dir / "locomotion_full_temporal_per_env.csv", flat_rows)
    md = [
        "# Locomotion Full Temporal Detector",
        "",
        "Read-only prior-side detector; no generator/PPO/fit_model changes.",
        "",
        "## Coverage",
        "",
        "| scope | witness_count | witness_rate | read |",
        "| --- | --- | --- | --- |",
    ]
    for item in reports:
        cov = item["classification"]["coverage"]
        n = int(cov["n_envs"])
        count = int(cov["sustained_locomotion_temporal_witness_count"])
        rate = float(cov["sustained_locomotion_temporal_witness_rate"])
        read = "present" if rate >= 0.15 else "scarce_or_detector_too_strict"
        md.append(f"| {item['label']} | {count}/{n} | {rate:.3f} | {read} |")
    md.extend(["", "## Next", "", "Stratify fixed-list PPO trainability by witness vs non-witness before any generator repair."])
    (out_dir / "locomotion_full_temporal_detector.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({item["label"]: item["classification"]["coverage"] for item in reports}, indent=2))
    print(report_path)


if __name__ == "__main__":
    main()
