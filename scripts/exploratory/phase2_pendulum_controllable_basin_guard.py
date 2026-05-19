#!/usr/bin/env python
"""Cached guard for prior-internal Pendulum controllable basins.

This is a narrow mechanism probe, not a PPO score and not a full feedback grid.
It tests whether an existing prior h exposes the missing Pendulum-like mechanism:

* a small bounded q+v feedback controller can improve future reward,
* the improvement persists in the suffix after the controller has settled,
* action magnitude decays from prefix to suffix,
* the suffix state is closer to the basin than its own prefix and zero-action.

The script is intentionally generator-level/cached-label shaped: it consumes
fixed h lists, uses only existing state/action slots, writes per-env labels, and
does not modify Exact SCM, fit_model, PPO, or selector quota decisions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ticl.priors.environment_prior import EnvironmentPrior  # noqa: E402


ART = Path("/home/chen/RLPFN/artifacts")
DEFAULT_OUTPUT_DIR = ART / "phase2_pendulum_controllable_basin_guard_0512"
BOOL_TRUE = {"1", "true", "yes", "y", "t", "on"}


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() in BOOL_TRUE


def _infer_label(path: str | Path) -> str:
    resolved = Path(path).expanduser().resolve()
    if resolved.parent.name == "fixed_env_group":
        return resolved.parent.parent.name
    return resolved.stem


def _parse_gain_grid(text: str) -> list[tuple[float, float]]:
    values: list[tuple[float, float]] = []
    for item in str(text).split(","):
        token = item.strip()
        if not token:
            continue
        if ":" not in token:
            raise ValueError(f"gain must be kq:kv, got {item!r}")
        kq, kv = token.split(":", 1)
        values.append((float(kq), float(kv)))
    if not values:
        raise ValueError("at least one gain is required")
    return values


def _parse_init_states(text: str) -> list[tuple[float, float]]:
    values: list[tuple[float, float]] = []
    for item in str(text).split(","):
        token = item.strip()
        if not token:
            continue
        if ":" not in token:
            raise ValueError(f"init state must be q:v, got {item!r}")
        q, v = token.split(":", 1)
        values.append((float(q), float(v)))
    if not values:
        raise ValueError("at least one init state is required")
    return values


def _unpack_transition(out: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(out, tuple) or len(out) < 2:
        raise TypeError(f"unexpected transition output: {type(out)!r}")
    return out[0], out[1]


def _build_batch_env(h_list: list[dict[str, Any]], *, device: torch.device, seed: int) -> dict[str, Any]:
    prior = EnvironmentPrior(dict(h_list[0]))
    rng_seeds = [int(seed) + idx for idx in range(len(h_list))]
    return prior._sample_environment_family_coarse_batch(
        h_list,
        device=str(device),
        rng_seeds=rng_seeds,
        build_policy_generator=False,
    )


def _rollout_mode(
    env: dict[str, Any],
    *,
    q0: float,
    v0: float,
    mode: str | tuple[float, float, int],
    n_steps: int,
    split_step: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    batch = int(env["pendulum_settle_yield_repair_active"].numel())
    x = torch.zeros((batch, int(env["env_input_dim"])), dtype=torch.float32, device=device)
    x[:, 0] = float(q0)
    x[:, 1] = float(v0)
    action_start = int(env["state_dim"])
    rewards: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    norms: list[torch.Tensor] = []
    for _ in range(int(n_steps)):
        q = x[:, 0]
        v = x[:, 1]
        if mode == "zero":
            action = torch.zeros_like(q)
        else:
            kq, kv, sign = mode
            action = float(sign) * (float(kq) * q + float(kv) * v)
        x[:, action_start] = action
        state, reward = _unpack_transition(env["transition_generator"](x))
        x[:, 0] = state[:, 0]
        x[:, 1] = state[:, 1]
        rewards.append(reward[:, 0].detach())
        actions.append(action.detach().abs())
        norms.append(torch.sqrt(state[:, 0].detach().square() + state[:, 1].detach().square()))
    reward_t = torch.stack(rewards, dim=0)
    action_t = torch.stack(actions, dim=0)
    norm_t = torch.stack(norms, dim=0)
    split_step = int(max(1, min(int(n_steps) - 1, int(split_step))))
    return {
        "full": reward_t.sum(dim=0),
        "suffix": reward_t[split_step:].sum(dim=0),
        "action_prefix": action_t[:split_step].mean(dim=0),
        "action_suffix": action_t[split_step:].mean(dim=0),
        "norm_prefix": norm_t[:split_step].mean(dim=0),
        "norm_suffix": norm_t[split_step:].mean(dim=0),
    }


def _average_rollouts(payloads: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        key: torch.stack([payload[key] for payload in payloads], dim=0).mean(dim=0)
        for key in payloads[0]
    }


def evaluate_h_list(
    h_list: list[dict[str, Any]],
    *,
    label: str,
    seed: int,
    device: torch.device,
    n_steps: int,
    split_frac: float,
    gains: list[tuple[float, float]],
    init_states: list[tuple[float, float]],
    reward_margin: float,
    suffix_margin: float,
    action_decay_margin: float,
    settle_ratio: float,
) -> dict[str, Any]:
    env = _build_batch_env(h_list, device=device, seed=seed)
    split_step = int(round(float(split_frac) * float(n_steps)))
    zero = _average_rollouts(
        [
            _rollout_mode(
                env,
                q0=q0,
                v0=v0,
                mode="zero",
                n_steps=n_steps,
                split_step=split_step,
                device=device,
            )
            for q0, v0 in init_states
        ]
    )

    best: dict[str, torch.Tensor] | None = None
    best_mode = [""] * len(h_list)
    for kq, kv in gains:
        for sign in (-1, 1):
            mode = (float(kq), float(kv), int(sign))
            current = _average_rollouts(
                [
                    _rollout_mode(
                        env,
                        q0=q0,
                        v0=v0,
                        mode=mode,
                        n_steps=n_steps,
                        split_step=split_step,
                        device=device,
                    )
                    for q0, v0 in init_states
                ]
            )
            if best is None:
                best = {key: value.clone() for key, value in current.items()}
                best_mode = [f"sgn{sign:+d}_kq{kq:g}_kv{kv:g}"] * len(h_list)
                continue
            replace = current["full"] > best["full"]
            for key in best:
                best[key] = torch.where(replace, current[key], best[key])
            for idx, flag in enumerate(replace.detach().cpu().tolist()):
                if bool(flag):
                    best_mode[idx] = f"sgn{sign:+d}_kq{kq:g}_kv{kv:g}"
    if best is None:
        raise RuntimeError("no controllability modes evaluated")

    reward_condition = best["full"] > zero["full"] + float(reward_margin)
    suffix_reward_condition = best["suffix"] > zero["suffix"] + float(suffix_margin)
    action_decay_condition = best["action_prefix"] > best["action_suffix"] + float(action_decay_margin)
    self_state_settle_condition = best["norm_suffix"] < best["norm_prefix"] * float(settle_ratio)
    open_state_settle_condition = best["norm_suffix"] < zero["norm_suffix"] * float(settle_ratio)
    sustained = (
        reward_condition
        & suffix_reward_condition
        & action_decay_condition
        & self_state_settle_condition
        & open_state_settle_condition
    )

    def _cpu_list(t: torch.Tensor) -> list[float]:
        return [float(v) for v in t.detach().cpu().tolist()]

    rows: list[dict[str, Any]] = []
    metrics = {
        "zero_full": _cpu_list(zero["full"]),
        "zero_suffix": _cpu_list(zero["suffix"]),
        "zero_norm_suffix": _cpu_list(zero["norm_suffix"]),
        "best_full": _cpu_list(best["full"]),
        "best_suffix": _cpu_list(best["suffix"]),
        "best_action_prefix": _cpu_list(best["action_prefix"]),
        "best_action_suffix": _cpu_list(best["action_suffix"]),
        "best_norm_prefix": _cpu_list(best["norm_prefix"]),
        "best_norm_suffix": _cpu_list(best["norm_suffix"]),
    }
    bools = {
        "reward_condition": reward_condition.detach().cpu().tolist(),
        "suffix_reward_condition": suffix_reward_condition.detach().cpu().tolist(),
        "action_decay_condition": action_decay_condition.detach().cpu().tolist(),
        "self_state_settle_condition": self_state_settle_condition.detach().cpu().tolist(),
        "open_state_settle_condition": open_state_settle_condition.detach().cpu().tolist(),
        "pendulum_sustained_settle": sustained.detach().cpu().tolist(),
    }
    for idx, h in enumerate(h_list):
        axis_support = _as_bool(h.get("pendulum_settle_yield_repair_enabled")) and _as_bool(
            h.get("pendulum_settle_yield_action_controls_position")
        )
        axis_supported_sustained = bool(axis_support) and bool(bools["pendulum_sustained_settle"][idx])
        row = {
            "rule": str(label),
            "source_index": idx,
            "env_index": idx,
            "pendulum_axis_support": bool(axis_support),
            "pendulum_short_gap": bool(bools["reward_condition"][idx]),
            "pendulum_sustained_settle": bool(bools["pendulum_sustained_settle"][idx]),
            "pendulum_axis_supported_sustained_settle": axis_supported_sustained,
            "pendulum_profile_quota_ready": False,
            "pendulum_medium_settle": axis_supported_sustained,
            "pendulum_strict_settle": axis_supported_sustained,
            "pendulum_short_temporal_settle_guard": axis_supported_sustained,
            "pendulum_label_available": True,
            "pendulum_settle_guard_precision_class": "transition_qv_controllable_basin_guard",
            "best_qv_feedback_mode": best_mode[idx],
            "label_key": f"{label}:{idx}",
        }
        for key, values in bools.items():
            row[key] = bool(values[idx])
        for key, values in metrics.items():
            row[key] = float(values[idx])
        rows.append(row)

    coverage = {
        "n_envs": len(rows),
        "axis_support_count": int(sum(bool(row["pendulum_axis_support"]) for row in rows)),
        "short_gap_count": int(sum(bool(row["pendulum_short_gap"]) for row in rows)),
        "sustained_settle_count": int(sum(bool(row["pendulum_sustained_settle"]) for row in rows)),
        "axis_supported_sustained_settle_count": int(
            sum(bool(row["pendulum_axis_supported_sustained_settle"]) for row in rows)
        ),
        "best_mode_counts": dict(Counter(str(row["best_qv_feedback_mode"]) for row in rows)),
    }
    for key in (
        "axis_support_count",
        "short_gap_count",
        "sustained_settle_count",
        "axis_supported_sustained_settle_count",
    ):
        coverage[key.replace("_count", "_rate")] = float(coverage[key] / max(1, len(rows)))
    return {
        "label": str(label),
        "coverage": coverage,
        "per_env": rows,
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


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Pendulum Controllable Basin Guard",
        "",
        "Transition-level q+v controllability cached label.",
        "",
        "## Coverage",
        "",
        "| rule | n | axis support | short gap | sustained settle | axis-supported sustained |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in report["lists"]:
        cov = item["coverage"]
        lines.append(
            f"| `{item['label']}` | {cov['n_envs']} | {cov['axis_support_count']} | "
            f"{cov['short_gap_count']} | {cov['sustained_settle_count']} | "
            f"{cov['axis_supported_sustained_settle_count']} |"
        )
    lines.extend(["", "## Read", "", report["read"]])
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(str(args.device))
    gains = _parse_gain_grid(args.feedback_gains)
    init_states = _parse_init_states(args.init_states)
    lists = []
    all_rows: list[dict[str, Any]] = []
    for idx, fixed in enumerate(args.fixed_h_list_json):
        h_list = list(_load_json(fixed))
        if int(args.limit_envs) > 0:
            h_list = h_list[: int(args.limit_envs)]
        label = args.labels[idx] if idx < len(args.labels) else _infer_label(fixed)
        item = evaluate_h_list(
            h_list,
            label=label,
            seed=int(args.seed) + 1009 * idx,
            device=device,
            n_steps=int(args.n_steps),
            split_frac=float(args.split_frac),
            gains=gains,
            init_states=init_states,
            reward_margin=float(args.reward_margin),
            suffix_margin=float(args.suffix_margin),
            action_decay_margin=float(args.action_decay_margin),
            settle_ratio=float(args.settle_ratio),
        )
        lists.append(item)
        all_rows.extend(item["per_env"])
    transition_ready = bool(
        lists
        and min(item["coverage"]["sustained_settle_count"] for item in lists) >= int(args.min_count)
        and min(item["coverage"]["sustained_settle_rate"] for item in lists) >= float(args.min_rate)
    )
    axis_supported_ready = bool(
        lists
        and min(item["coverage"]["axis_supported_sustained_settle_count"] for item in lists)
        >= int(args.min_count)
        and min(item["coverage"]["axis_supported_sustained_settle_rate"] for item in lists)
        >= float(args.min_rate)
    )
    report = {
        "analysis_entry": "phase2_pendulum_controllable_basin_guard",
        "schema": "phase2_pendulum_controllable_basin_guard.v1",
        "contract": {
            "environment_side_only": True,
            "transition_level_qv_guard": True,
            "no_training": True,
            "ppo_not_under_test": True,
            "raw_delta_not_used": True,
            "standalone_generator": False,
            "uses_existing_state_action_slots": True,
            "selector_quota_not_enabled_by_this_script": True,
        },
        "config": {
            "device": str(device),
            "n_steps": int(args.n_steps),
            "split_frac": float(args.split_frac),
            "feedback_gains": [[float(kq), float(kv)] for kq, kv in gains],
            "init_states": [[float(q), float(v)] for q, v in init_states],
            "reward_margin": float(args.reward_margin),
            "suffix_margin": float(args.suffix_margin),
            "action_decay_margin": float(args.action_decay_margin),
            "settle_ratio": float(args.settle_ratio),
        },
        "decision": {
            "controllable_basin_guard_ready": axis_supported_ready,
            "transition_sustained_settle_ready": transition_ready,
            "axis_supported_controllable_basin_ready": axis_supported_ready,
            "min_count": min(
                (item["coverage"]["axis_supported_sustained_settle_count"] for item in lists),
                default=0,
            ),
            "min_rate": min(
                (item["coverage"]["axis_supported_sustained_settle_rate"] for item in lists),
                default=0.0,
            ),
            "min_transition_sustained_count": min(
                (item["coverage"]["sustained_settle_count"] for item in lists),
                default=0,
            ),
            "min_transition_sustained_rate": min(
                (item["coverage"]["sustained_settle_rate"] for item in lists),
                default=0.0,
            ),
            "required_min_count": int(args.min_count),
            "required_min_rate": float(args.min_rate),
        },
        "read": (
            "Transition sustained-settle is diagnostic only: it can be true for old mechanisms under ideal q+v "
            "feedback. Cached selector readiness requires the prior-internal action-state basin axis and "
            "sustained-settle together; selector quota still needs separate health/diversity review."
        ),
        "lists": lists,
    }
    csv_path = out_dir / "pendulum_controllable_basin_per_env.csv"
    json_path = out_dir / "pendulum_controllable_basin_guard.json"
    md_path = out_dir / "pendulum_controllable_basin_guard.md"
    _write_csv(csv_path, all_rows)
    report["outputs"] = {"csv": str(csv_path), "json": str(json_path), "markdown": str(md_path)}
    json_path.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(_json_safe({"decision": report["decision"], "outputs": report["outputs"]}), sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--fixed-h-list-json", action="append", required=True)
    parser.add_argument("--labels", nargs="*", default=[])
    parser.add_argument("--seed", type=int, default=9724)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit-envs", type=int, default=0)
    parser.add_argument("--n-steps", type=int, default=64)
    parser.add_argument("--split-frac", type=float, default=0.5)
    parser.add_argument("--feedback-gains", default="1:0,2:1,3:1,3:2")
    parser.add_argument("--init-states", default="1:0.5,-1:-0.5")
    parser.add_argument("--reward-margin", type=float, default=0.25)
    parser.add_argument("--suffix-margin", type=float, default=0.10)
    parser.add_argument("--action-decay-margin", type=float, default=0.01)
    parser.add_argument("--settle-ratio", type=float, default=0.75)
    parser.add_argument("--min-count", type=int, default=4)
    parser.add_argument("--min-rate", type=float, default=0.10)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
