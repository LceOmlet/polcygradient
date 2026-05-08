#!/usr/bin/env python
"""Reward-path-only parity for exploratory gated reward balancing.

This is an environment-side parity check. It does not train PPO and does not
modify the exact-SCM/default prior path. For selected frozen-h entries it checks
the runtime wrapper contract directly:

* state_next is unchanged by reward balancing
* terminal_signal is unchanged by reward balancing
* terminal_bonus_base is unchanged by reward balancing
* reward equals an independently invoked balanced reward-path calculation
* original_healthy_keep rows are complete no-ops
* width_bias_balance rows actually change reward on at least one probe input
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from phase2_reward_group_balance_runtime import (  # noqa: E402
    apply_reward_group_balance_to_env,
    balance_scales_from_h_list,
    h_balance_enabled,
    _temporarily_scaled_transition_reward_path,
)
from ticl.model_configs import get_model_default_config  # noqa: E402
from ticl.priors.environment_prior import EnvironmentPrior  # noqa: E402
from ticl.rlpfn_maintained_path import validate_rlpfn_maintained_path_config  # noqa: E402


DEFAULT_PROBE_DIRS = (
    "/home/chen/RLPFN/artifacts/phase2_exploratory_reward_group_balance_gated_n64_0506,"
    "/home/chen/RLPFN/artifacts/phase2_exploratory_reward_group_balance_gated_n64_seed6261_0506"
)


def _json_safe(value: Any) -> Any:
    if torch.is_tensor(value):
        if value.ndim == 0:
            return _json_safe(value.detach().cpu().item())
        return [_json_safe(v) for v in value.detach().cpu().tolist()]
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _parse_csv_list(text: str) -> list[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def _load_selected_rows(probe_dir: Path) -> list[dict[str, Any]]:
    path = probe_dir / "selected_prior_envs.csv"
    with path.open("r", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _pick_rows(rows: list[dict[str, Any]], *, max_width: int, max_keep: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    buckets = [
        ("width_bias_balance", max_width),
        ("original_healthy_keep", max_keep),
    ]
    for kind, limit in buckets:
        if int(limit) <= 0:
            continue
        count = 0
        for row in rows:
            if str(row.get("selected_repair_kind")) != kind:
                continue
            h_path = str(row.get("full_frozen_h_json", ""))
            if not h_path or h_path in seen_paths:
                continue
            out.append(row)
            seen_paths.add(h_path)
            count += 1
            if count >= int(limit):
                break
    return out


def _rng_state_for_device(device: torch.device) -> dict[str, Any]:
    state: dict[str, Any] = {"cpu": torch.random.get_rng_state()}
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(device=device)
    return state


def _set_rng_state_for_device(state: dict[str, Any], device: torch.device) -> None:
    torch.random.set_rng_state(state["cpu"])
    if device.type == "cuda" and "cuda" in state:
        torch.cuda.set_rng_state(state["cuda"], device=device)


def _max_abs_diff(left: Any, right: Any) -> float:
    if left is None and right is None:
        return 0.0
    if left is None or right is None:
        return float("inf")
    if not torch.is_tensor(left):
        left = torch.as_tensor(left)
    if not torch.is_tensor(right):
        right = torch.as_tensor(right)
    if tuple(left.shape) != tuple(right.shape):
        return float("inf")
    if left.numel() == 0:
        return 0.0
    return float(torch.max(torch.abs(left.detach() - right.detach())).cpu().item())


def _per_env_max_abs_diff(left: Any, right: Any, n_envs: int) -> list[float]:
    if left is None and right is None:
        return [0.0 for _ in range(n_envs)]
    if left is None or right is None:
        return [float("inf") for _ in range(n_envs)]
    if not torch.is_tensor(left):
        left = torch.as_tensor(left)
    if not torch.is_tensor(right):
        right = torch.as_tensor(right)
    if tuple(left.shape) != tuple(right.shape) or left.ndim == 0:
        return [float("inf") for _ in range(n_envs)]
    diffs = torch.abs(left.detach() - right.detach()).reshape(n_envs, -1)
    return [float(v) for v in torch.max(diffs, dim=1).values.cpu().tolist()]


def _unpack(prior: EnvironmentPrior, out: Any) -> tuple[Any, Any, Any, Any]:
    return prior._unpack_transition_output(out)


def _build_env(
    *,
    prior: EnvironmentPrior,
    h_list: list[dict[str, Any]],
    seeds: list[int],
    device: torch.device,
) -> dict[str, Any]:
    return prior._sample_environment_family_coarse_batch(
        h_list=copy.deepcopy(h_list),
        device=device,
        rng_seeds=[int(s) for s in seeds],
        build_policy_generator=False,
        preserve_skipped_generator_rng=True,
        _disable_rejection=True,
    )


def _audit_probe_dir(probe_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    rows_all = _load_selected_rows(probe_dir)
    rows = _pick_rows(
        rows_all,
        max_width=int(args.max_width_rows_per_probe),
        max_keep=int(args.max_keep_rows_per_probe),
    )
    h_list = [_load_json(row["full_frozen_h_json"]) for row in rows]
    seeds = [int(row["env_seed"]) for row in rows]
    kinds = [str(row.get("selected_repair_kind")) for row in rows]
    balance_enabled = [bool(h_balance_enabled(h)) for h in h_list]
    if not rows:
        return {
            "probe_dir": str(probe_dir),
            "hard_pass": False,
            "reason": "no rows selected for parity",
        }

    device = torch.device(str(args.device))
    cfg = copy.deepcopy(get_model_default_config("rlpfn"))
    validate_rlpfn_maintained_path_config(cfg)
    prior = EnvironmentPrior(copy.deepcopy(cfg["prior"]["environment"]))
    env = _build_env(prior=prior, h_list=h_list, seeds=seeds, device=device)
    original_transition = env["transition_generator"]
    packed_width = int(getattr(original_transition, "_packed_input_cap", env.get("env_input_dim", 1)))
    n_envs = int(len(h_list))
    state_scales, action_scales, noise_scales, profiles = balance_scales_from_h_list(h_list)

    generator = torch.Generator(device=device)
    generator.manual_seed(int(args.input_seed))
    row_reward_change_max = [0.0 for _ in range(n_envs)]
    batch_reports: list[dict[str, Any]] = []
    max_state_diff = 0.0
    max_terminal_diff = 0.0
    max_terminal_bonus_diff = 0.0
    max_reward_expected_diff = 0.0
    max_keep_reward_diff = 0.0

    apply_reward_group_balance_to_env(env, h_list, fail_if_unavailable=True)
    wrapped_transition = env["transition_generator"]
    wrapper_enabled = bool(getattr(wrapped_transition, "_exploratory_reward_group_balance_enabled", False))

    for batch_idx in range(int(args.input_batches)):
        x = torch.randn((n_envs, packed_width), device=device, dtype=torch.float32, generator=generator)
        rng_state = _rng_state_for_device(device)
        original_out = original_transition(x, x_input_is_packed=True)
        _set_rng_state_for_device(rng_state, device)
        wrapped_out = wrapped_transition(x, x_input_is_packed=True)
        _set_rng_state_for_device(rng_state, device)
        with _temporarily_scaled_transition_reward_path(
            original_transition,
            env,
            h_list,
            state_scales=state_scales,
            action_scales=action_scales,
            noise_scales=noise_scales,
        ):
            expected_balanced_out = original_transition(x, x_input_is_packed=True)

        original_state, original_reward, original_terminal, original_bonus = _unpack(prior, original_out)
        wrapped_state, wrapped_reward, wrapped_terminal, wrapped_bonus = _unpack(prior, wrapped_out)
        _expected_state, expected_reward, _expected_terminal, _expected_bonus = _unpack(prior, expected_balanced_out)

        state_diff = _max_abs_diff(original_state, wrapped_state)
        terminal_diff = _max_abs_diff(original_terminal, wrapped_terminal)
        bonus_diff = _max_abs_diff(original_bonus, wrapped_bonus)
        reward_expected_diff = _max_abs_diff(expected_reward, wrapped_reward)
        reward_original_diffs = _per_env_max_abs_diff(original_reward, wrapped_reward, n_envs)
        for env_idx, diff in enumerate(reward_original_diffs):
            row_reward_change_max[env_idx] = max(float(row_reward_change_max[env_idx]), float(diff))
            if not bool(balance_enabled[env_idx]):
                max_keep_reward_diff = max(float(max_keep_reward_diff), float(diff))

        max_state_diff = max(max_state_diff, state_diff)
        max_terminal_diff = max(max_terminal_diff, terminal_diff)
        max_terminal_bonus_diff = max(max_terminal_bonus_diff, bonus_diff)
        max_reward_expected_diff = max(max_reward_expected_diff, reward_expected_diff)
        batch_reports.append(
            {
                "batch_idx": int(batch_idx),
                "state_next_max_abs_diff": state_diff,
                "terminal_signal_max_abs_diff": terminal_diff,
                "terminal_bonus_base_max_abs_diff": bonus_diff,
                "reward_vs_expected_balanced_max_abs_diff": reward_expected_diff,
                "reward_vs_original_max_abs_diff": max(reward_original_diffs) if reward_original_diffs else None,
            }
        )

    row_reports = []
    for idx, row in enumerate(rows):
        row_reports.append(
            {
                "idx": int(idx),
                "rule": str(row.get("rule")),
                "env_seed": int(row.get("env_seed")),
                "repair_kind": str(row.get("selected_repair_kind")),
                "balance_enabled": bool(balance_enabled[idx]),
                "balance_profile": str(profiles[idx]),
                "reward_vs_original_max_abs_diff": float(row_reward_change_max[idx]),
                "frozen_h_json": str(Path(row["full_frozen_h_json"]).expanduser().resolve()),
            }
        )

    exact_atol = float(args.exact_atol)
    min_reward_change = float(args.min_balanced_reward_change)
    exact_paths_pass = bool(
        max_state_diff <= exact_atol
        and max_terminal_diff <= exact_atol
        and max_terminal_bonus_diff <= exact_atol
        and max_reward_expected_diff <= exact_atol
    )
    keep_rows = [i for i, enabled in enumerate(balance_enabled) if not enabled]
    balanced_rows = [i for i, enabled in enumerate(balance_enabled) if enabled]
    keep_noop_pass = bool(all(row_reward_change_max[i] <= exact_atol for i in keep_rows))
    balanced_effective_pass = bool(
        balanced_rows
        and all(row_reward_change_max[i] >= min_reward_change for i in balanced_rows)
    )
    hard_pass = bool(wrapper_enabled and exact_paths_pass and keep_noop_pass and balanced_effective_pass)

    env["transition_generator"] = None
    env["policy_generator"] = None
    del env
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "probe_dir": str(probe_dir),
        "hard_pass": hard_pass,
        "wrapper_enabled": wrapper_enabled,
        "n_rows": int(len(rows)),
        "kind_counts": {kind: int(kinds.count(kind)) for kind in sorted(set(kinds))},
        "balance_enabled_count": int(sum(1 for v in balance_enabled if v)),
        "keep_count": int(sum(1 for v in balance_enabled if not v)),
        "packed_width": int(packed_width),
        "input_batches": int(args.input_batches),
        "exact_atol": exact_atol,
        "min_balanced_reward_change": min_reward_change,
        "max_state_next_abs_diff": float(max_state_diff),
        "max_terminal_signal_abs_diff": float(max_terminal_diff),
        "max_terminal_bonus_base_abs_diff": float(max_terminal_bonus_diff),
        "max_reward_vs_expected_balanced_abs_diff": float(max_reward_expected_diff),
        "max_keep_reward_vs_original_abs_diff": float(max_keep_reward_diff),
        "min_balanced_reward_vs_original_abs_diff": (
            min(float(row_reward_change_max[i]) for i in balanced_rows) if balanced_rows else None
        ),
        "checks": {
            "exact_paths_pass": exact_paths_pass,
            "keep_noop_pass": keep_noop_pass,
            "balanced_effective_pass": balanced_effective_pass,
        },
        "rows": row_reports,
        "batches": batch_reports,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(int(args.input_seed))
    np.random.seed(int(args.input_seed))
    torch.manual_seed(int(args.input_seed))
    probe_dirs = [Path(p).expanduser().resolve() for p in _parse_csv_list(args.probe_dirs)]
    runs = [_audit_probe_dir(path, args) for path in probe_dirs]
    hard_pass = bool(runs and all(bool(run.get("hard_pass")) for run in runs))
    report = {
        "analysis_entry": "phase2_gated_reward_path_only_parity",
        "exploratory_only": True,
        "hard_pass": hard_pass,
        "contract": {
            "does_not_modify_exact_scm": True,
            "does_not_modify_default_prior_generator": True,
            "does_not_train_ppo": True,
            "environment_side_only": True,
            "checked_paths": [
                "state_next unchanged",
                "terminal_signal unchanged",
                "terminal_bonus_base unchanged",
                "reward equals balanced reward-path calculation",
                "original_healthy_keep is no-op",
                "width_bias_balance changes reward",
            ],
        },
        "runs": runs,
    }
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "gated_reward_path_only_parity_report.json"
    report_path.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    md_lines = [
        "# Gated Reward-Path-Only Parity",
        "",
        f"hard_pass: {hard_pass}",
        "",
        "## Runs",
    ]
    for item in runs:
        md_lines.append(
            f"- {Path(str(item.get('probe_dir'))).name}: hard={item.get('hard_pass')}, "
            f"state_diff={item.get('max_state_next_abs_diff')}, "
            f"terminal_diff={item.get('max_terminal_signal_abs_diff')}, "
            f"bonus_diff={item.get('max_terminal_bonus_base_abs_diff')}, "
            f"reward_expected_diff={item.get('max_reward_vs_expected_balanced_abs_diff')}, "
            f"keep_reward_diff={item.get('max_keep_reward_vs_original_abs_diff')}, "
            f"min_balanced_reward_diff={item.get('min_balanced_reward_vs_original_abs_diff')}"
        )
    md_path = out_dir / "gated_reward_path_only_parity_summary.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(json.dumps({"hard_pass": hard_pass, "report": str(report_path)}, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-dirs", type=str, default=DEFAULT_PROBE_DIRS)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/chen/RLPFN/artifacts/phase2_exploratory_gated_reward_path_only_parity_0506",
    )
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--input-seed", type=int, default=927060)
    parser.add_argument("--input-batches", type=int, default=4)
    parser.add_argument("--max-width-rows-per-probe", type=int, default=4)
    parser.add_argument("--max-keep-rows-per-probe", type=int, default=2)
    parser.add_argument("--exact-atol", type=float, default=0.0)
    parser.add_argument("--min-balanced-reward-change", type=float, default=1e-8)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
