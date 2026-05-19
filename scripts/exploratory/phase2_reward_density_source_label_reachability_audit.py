#!/usr/bin/env python
"""Reachability audit for reward density source labels.

This script does not build a new generator and does not select a final h-list.
It answers the narrower reductionist question:

* Are state/action dominant reward-density allocations present in the current
  Exact-SCM formula image?
* Are they only missing because the maintained balance sampler rejects them?
* If promoted to a category axis, would the labels be badly coupled to basic
  structure axes?

Only after this audit passes should a source-label sampler be implemented.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch


ART = Path("/home/chen/RLPFN/artifacts")
REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ticl.model_configs import get_rlpfn_default_config  # noqa: E402
from ticl.priors.environment_prior import EnvironmentPrior  # noqa: E402
from ticl.rlpfn_maintained_path import apply_rlpfn_maintained_path_defaults  # noqa: E402


DEFAULT_OUTPUT_DIR = ART / "phase2_reward_density_source_label_reachability_audit_0518"
NATURAL_HARD_LABELS = ("density_balanced", "state_density_dominant", "action_density_dominant")
DIAGNOSTIC_LABELS = ("noise_density_dominant", "missing")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _seed_all(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _make_prior(*, seed: int, topology_conditioned: bool) -> EnvironmentPrior:
    _seed_all(seed)
    cfg = get_rlpfn_default_config()
    cfg["prior"]["prior_type"] = "environment_only"
    apply_rlpfn_maintained_path_defaults(cfg)
    env_cfg = cfg["prior"]["environment"]
    env_cfg["reward_topology_conditioned_sampling_enabled"] = bool(topology_conditioned)
    if not topology_conditioned:
        env_cfg["reward_state_input_gain_fraction_conditioned_sampling_enabled"] = False
        env_cfg["reward_state_input_gain_fraction_conditioned_min"] = 0.0
        env_cfg["reward_action_input_gain_fraction_conditioned_min"] = 0.0
        env_cfg["reward_noise_input_gain_fraction_conditioned_min"] = 0.0
        env_cfg["reward_state_to_action_gain_ratio_conditioned_max"] = 0.0
        env_cfg["reward_state_to_noise_gain_ratio_conditioned_max"] = 0.0
    return EnvironmentPrior(env_cfg)


def _tensor_list(env: dict[str, Any], key: str, n: int, default: float = float("nan")) -> list[float]:
    value = env.get(key, None)
    if torch.is_tensor(value):
        vals = value.detach().cpu().reshape(-1).tolist()
        if len(vals) == n:
            return [float(v) for v in vals]
        if len(vals) == 1:
            return [float(vals[0]) for _ in range(n)]
    if isinstance(value, (list, tuple)):
        if len(value) == n:
            return [float(v) for v in value]
        if len(value) == 1:
            return [float(value[0]) for _ in range(n)]
    try:
        scalar = float(value)
    except Exception:
        scalar = float(default)
    return [scalar for _ in range(n)]


def _int_tensor_list(env: dict[str, Any], key: str, n: int) -> list[int]:
    return [int(round(v)) for v in _tensor_list(env, key, n, default=0.0)]


def _density_label(
    state: float,
    action: float,
    noise: float,
    *,
    state_dim: int,
    action_dim: int,
    noise_dim: int,
    threshold: float,
) -> str:
    # Compare density only across input groups that actually exist.  A zero
    # noise dimension is a structural choice, not a missing reward-path signal.
    vals = {}
    if int(state_dim) > 0:
        vals["state"] = float(state)
    if int(action_dim) > 0:
        vals["action"] = float(action)
    if int(noise_dim) > 0:
        vals["noise"] = float(noise)
    finite = {k: v for k, v in vals.items() if math.isfinite(v) and v > 0.0}
    if len(finite) < 2:
        return "missing"
    ordered = sorted(finite.items(), key=lambda kv: kv[1], reverse=True)
    top_name, top_value = ordered[0]
    second_value = max(ordered[1][1], 1e-12)
    bottom_value = max(ordered[-1][1], 1e-12)
    if top_value / bottom_value <= float(threshold):
        return "density_balanced"
    if top_value / second_value >= float(threshold):
        return f"{top_name}_density_dominant"
    return "density_skewed_no_clear_dominant"


def _bin_action(value: int) -> str:
    if int(value) <= 10:
        return "action_low_01_10"
    if int(value) <= 20:
        return "action_mid_11_20"
    return "action_high_21_30"


def _bin_dim400(value: int, prefix: str) -> str:
    value = int(value)
    if value <= 133:
        return f"{prefix}_low"
    if value <= 266:
        return f"{prefix}_mid"
    return f"{prefix}_high"


def _bin_terminal(value: float) -> str:
    if float(value) <= 80.0 / 3.0:
        return "terminal_low_0_26p67"
    if float(value) <= 160.0 / 3.0:
        return "terminal_mid_26p67_53p33"
    return "terminal_high_53p33_80"


def _terminal_target(h: dict[str, Any]) -> float:
    value = h.get("terminal_reset_count_target", 0.0)
    try:
        return float(value)
    except Exception:
        return 0.0


def _cramers_v(xs: list[str], ys: list[str]) -> float:
    n = len(xs)
    if n == 0:
        return 0.0
    x_labels = sorted(set(xs))
    y_labels = sorted(set(ys))
    if len(x_labels) <= 1 or len(y_labels) <= 1:
        return 0.0
    xi = {label: idx for idx, label in enumerate(x_labels)}
    yi = {label: idx for idx, label in enumerate(y_labels)}
    table = [[0.0 for _ in y_labels] for _ in x_labels]
    for x, y in zip(xs, ys):
        table[xi[x]][yi[y]] += 1.0
    row_sum = [sum(row) for row in table]
    col_sum = [sum(table[i][j] for i in range(len(x_labels))) for j in range(len(y_labels))]
    chi2 = 0.0
    for i in range(len(x_labels)):
        for j in range(len(y_labels)):
            expected = row_sum[i] * col_sum[j] / n
            if expected > 0:
                chi2 += (table[i][j] - expected) ** 2 / expected
    denom = n * max(1, min(len(x_labels) - 1, len(y_labels) - 1))
    return math.sqrt(max(0.0, chi2 / denom)) if denom > 0 else 0.0


def _entropy(counts: Counter[str], labels: tuple[str, ...]) -> float:
    total = sum(counts.get(label, 0) for label in labels)
    if total <= 0 or len(labels) <= 1:
        return 0.0
    out = 0.0
    for label in labels:
        count = counts.get(label, 0)
        if count <= 0:
            continue
        p = count / total
        out -= p * math.log(p)
    return out / math.log(len(labels))


def _stats(values: list[float]) -> dict[str, Any]:
    vals = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=np.float64)
    if vals.size == 0:
        return {"n": 0, "mean": None, "q10": None, "q50": None, "q90": None}
    return {
        "n": int(vals.size),
        "mean": float(vals.mean()),
        "q10": float(np.quantile(vals, 0.10)),
        "q50": float(np.quantile(vals, 0.50)),
        "q90": float(np.quantile(vals, 0.90)),
    }


def _collect_direct(
    *,
    n: int,
    batch_size: int,
    seed: int,
    env_seed_base: int,
    device: torch.device,
    topology_conditioned_flag: bool,
    threshold: float,
) -> list[dict[str, Any]]:
    prior = _make_prior(seed=seed, topology_conditioned=topology_conditioned_flag)
    rows: list[dict[str, Any]] = []
    for start in range(0, int(n), int(batch_size)):
        batch_n = min(int(batch_size), int(n) - start)
        h_list = list(prior._sample_batch_hypers(batch_n))
        seeds = [int(env_seed_base) + start + i for i in range(batch_n)]
        env = prior._sample_environment_family_coarse_batch(
            h_list=h_list,
            device=device,
            rng_seeds=seeds,
            build_policy_generator=False,
            preserve_skipped_generator_rng=True,
        )
        rows.extend(_rows_from_env(
            mode=("direct_flag_on" if topology_conditioned_flag else "direct_unconditioned"),
            start=start,
            h_list=h_list,
            seeds=seeds,
            env=env,
            threshold=threshold,
        ))
        env["transition_generator"] = None
        env["policy_generator"] = None
        del env
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return rows


def _collect_conditioned(
    *,
    n: int,
    batch_size: int,
    seed: int,
    env_seed_base: int,
    device: torch.device,
    threshold: float,
) -> list[dict[str, Any]]:
    prior = _make_prior(seed=seed, topology_conditioned=True)
    rows: list[dict[str, Any]] = []
    for start in range(0, int(n), int(batch_size)):
        batch_n = min(int(batch_size), int(n) - start)
        base_seeds = [int(env_seed_base) + start + i for i in range(batch_n)]
        h_list, env, accepted_seeds = prior._sample_batch_hypers_and_environment_family_coarse_batch_conditioned(
            batch_n,
            device=device,
            rng_seeds=base_seeds,
            build_policy_generator=False,
            preserve_skipped_generator_rng=True,
            return_final_env=True,
        )
        rows.extend(_rows_from_env(
            mode="conditioned_balance",
            start=start,
            h_list=h_list,
            seeds=[int(v) for v in accepted_seeds],
            env=env,
            threshold=threshold,
        ))
        env["transition_generator"] = None
        env["policy_generator"] = None
        del env
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return rows


def _rows_from_env(
    *,
    mode: str,
    start: int,
    h_list: list[dict[str, Any]],
    seeds: list[int],
    env: dict[str, Any],
    threshold: float,
) -> list[dict[str, Any]]:
    n = len(h_list)
    state_density = _tensor_list(env, "reward_state_input_gain_density_ratio", n)
    action_density = _tensor_list(env, "reward_action_input_gain_density_ratio", n)
    noise_density = _tensor_list(env, "reward_noise_input_gain_density_ratio", n)
    state_frac = _tensor_list(env, "reward_state_input_gain_fraction", n)
    action_frac = _tensor_list(env, "reward_action_input_gain_fraction", n)
    noise_frac = _tensor_list(env, "reward_noise_input_gain_fraction", n)
    state_dim = _int_tensor_list(env, "state_dim_per_sample", n)
    obs_dim = _int_tensor_list(env, "obs_dim_per_sample", n)
    action_dim = _int_tensor_list(env, "action_dim_per_sample", n)
    noise_dim = _int_tensor_list(env, "noise_dim_per_sample", n)
    zero_dim = _int_tensor_list(env, "zero_pad_dim_per_sample", n)
    rows: list[dict[str, Any]] = []
    for i, h in enumerate(h_list):
        label = _density_label(
            state_density[i],
            action_density[i],
            noise_density[i],
            state_dim=state_dim[i],
            action_dim=action_dim[i],
            noise_dim=noise_dim[i],
            threshold=threshold,
        )
        rows.append(
            {
                "mode": str(mode),
                "env_idx": int(start + i),
                "env_seed": int(seeds[i]),
                "reward_density_topology": label,
                "natural_hard_label": bool(label in NATURAL_HARD_LABELS),
                "state_density_ratio": float(state_density[i]),
                "action_density_ratio": float(action_density[i]),
                "noise_density_ratio": float(noise_density[i]),
                "state_gain_fraction": float(state_frac[i]),
                "action_gain_fraction": float(action_frac[i]),
                "noise_gain_fraction": float(noise_frac[i]),
                "state_dim": int(state_dim[i]),
                "obs_dim": int(obs_dim[i]),
                "action_dim": int(action_dim[i]),
                "noise_dim": int(noise_dim[i]),
                "zero_pad_dim": int(zero_dim[i]),
                "state_dim_bin": _bin_dim400(state_dim[i], "state"),
                "obs_dim_bin": _bin_dim400(obs_dim[i], "obs"),
                "action_dim_bin": _bin_action(action_dim[i]),
                "noise_dim_bin": _bin_dim400(noise_dim[i], "noise"),
                "zero_pad_dim_bin": _bin_dim400(zero_dim[i], "zero"),
                "terminal_target": float(_terminal_target(h)),
                "terminal_target_bin": _bin_terminal(_terminal_target(h)),
            }
        )
    return rows


def _summarize_mode(rows: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    mode_rows = [row for row in rows if row["mode"] == mode]
    counts = Counter(str(row["reward_density_topology"]) for row in mode_rows)
    natural_counts = Counter({label: counts.get(label, 0) for label in NATURAL_HARD_LABELS})
    n = len(mode_rows)
    natural_occupied = sum(1 for label in NATURAL_HARD_LABELS if counts.get(label, 0) > 0)
    min_natural_frac = (
        min(counts.get(label, 0) for label in NATURAL_HARD_LABELS) / max(1, n)
        if n > 0
        else 0.0
    )
    coupling_rows: list[dict[str, Any]] = []
    xs = [str(row["reward_density_topology"]) for row in mode_rows]
    for axis in (
        "state_dim_bin",
        "obs_dim_bin",
        "action_dim_bin",
        "noise_dim_bin",
        "zero_pad_dim_bin",
        "terminal_target_bin",
    ):
        ys = [str(row[axis]) for row in mode_rows]
        v = _cramers_v(xs, ys)
        status = "high" if v >= 0.50 else ("medium" if v >= 0.25 else "low")
        if len(set(xs)) <= 1:
            status = "undefined_collapsed_density"
        coupling_rows.append(
            {
                "mode": mode,
                "axis": "reward_density_topology",
                "structure_axis": axis,
                "cramers_v": float(v),
                "status": status,
            }
        )
    return {
        "mode": mode,
        "n": n,
        "counts": {label: int(counts.get(label, 0)) for label in (*NATURAL_HARD_LABELS, *DIAGNOSTIC_LABELS, "density_skewed_no_clear_dominant")},
        "natural_occupied": int(natural_occupied),
        "natural_entropy": float(_entropy(natural_counts, NATURAL_HARD_LABELS)),
        "min_natural_label_fraction": float(min_natural_frac),
        "high_structure_coupling_count": int(sum(1 for row in coupling_rows if row["status"] == "high")),
        "coupling": coupling_rows,
        "stats": {
            "state_density_ratio": _stats([float(row["state_density_ratio"]) for row in mode_rows]),
            "action_density_ratio": _stats([float(row["action_density_ratio"]) for row in mode_rows]),
            "noise_density_ratio": _stats([float(row["noise_density_ratio"]) for row in mode_rows]),
            "state_gain_fraction": _stats([float(row["state_gain_fraction"]) for row in mode_rows]),
            "action_gain_fraction": _stats([float(row["action_gain_fraction"]) for row in mode_rows]),
            "noise_gain_fraction": _stats([float(row["noise_gain_fraction"]) for row in mode_rows]),
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    rows: list[dict[str, Any]] = []
    if "direct_unconditioned" in args.modes:
        rows.extend(
            _collect_direct(
                n=args.n,
                batch_size=args.batch_size,
                seed=args.seed,
                env_seed_base=args.env_seed_base,
                device=device,
                topology_conditioned_flag=False,
                threshold=args.dominance_threshold,
            )
        )
    if "direct_flag_on" in args.modes:
        rows.extend(
            _collect_direct(
                n=args.n,
                batch_size=args.batch_size,
                seed=args.seed + 17,
                env_seed_base=args.env_seed_base + 100000,
                device=device,
                topology_conditioned_flag=True,
                threshold=args.dominance_threshold,
            )
        )
    if "conditioned_balance" in args.modes:
        rows.extend(
            _collect_conditioned(
                n=args.n,
                batch_size=args.batch_size,
                seed=args.seed + 31,
                env_seed_base=args.env_seed_base + 200000,
                device=device,
                threshold=args.dominance_threshold,
            )
        )

    modes = [mode for mode in args.modes if any(row["mode"] == mode for row in rows)]
    mode_summaries = {mode: _summarize_mode(rows, mode) for mode in modes}
    coupling = [row for summary in mode_summaries.values() for row in summary["coupling"]]
    hard_labels_reachable_unconditioned = all(
        mode_summaries.get("direct_unconditioned", {}).get("counts", {}).get(label, 0) > 0
        for label in NATURAL_HARD_LABELS
    )
    hard_labels_reachable_conditioned = all(
        mode_summaries.get("conditioned_balance", {}).get("counts", {}).get(label, 0) > 0
        for label in NATURAL_HARD_LABELS
    )
    high_unconditioned = [
        row
        for row in mode_summaries.get("direct_unconditioned", {}).get("coupling", [])
        if row["status"] == "high"
    ]
    read = {
        "dominant_reward_allocation_hard_axis_decision": "not_promoted_by_current_evidence",
        "decision_reason": (
            "State/action density dominance are interpretable reward-allocation directions if the "
            "category contract explicitly promotes reward-density anisotropy.  Current Exact-SCM "
            "standard-init reward paths are intentionally near exchangeable per active input, and "
            "the current evidence does not require density dominance as a hard natural-training-family "
            "axis.  Noise-dominant remains diagnostic/stress only."
        ),
        "source_label_sampler_permitted": bool(hard_labels_reachable_unconditioned and len(high_unconditioned) == 0),
        "source_label_sampler_permitted_reason": (
            "unconditioned formula image reaches all natural hard labels without high structure coupling"
            if hard_labels_reachable_unconditioned and len(high_unconditioned) == 0
            else "do not implement sampler: density-dominant labels are not reachable in the current formula image or are structurally coupled; a sampler would become post-hoc filtering or require formula expansion"
        ),
        "conditioned_balance_suppresses_natural_labels": bool(
            hard_labels_reachable_unconditioned and not hard_labels_reachable_conditioned
        ),
        "if_promoted_minimal_labels": list(NATURAL_HARD_LABELS),
        "diagnostic_labels_not_hard": list(DIAGNOSTIC_LABELS),
    }
    report = {
        "analysis_entry": "phase2_reward_density_source_label_reachability_audit",
        "schema": "phase2_reward_density_source_label_reachability_audit.v1",
        "contract": {
            "read_only": True,
            "does_not_modify_generator": True,
            "does_not_select_final_h_list": True,
            "does_not_train": True,
            "tests_formula_image_before_sampler": True,
            "dominance_threshold": float(args.dominance_threshold),
        },
        "read": read,
        "mode_summaries": mode_summaries,
        "per_env": rows,
        "structure_coupling": coupling,
        "args": vars(args),
    }

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "reward_density_source_label_reachability_audit.json"
    md_path = out_dir / "reward_density_source_label_reachability_audit.md"
    per_env_path = out_dir / "reward_density_source_label_reachability_per_env.csv"
    coupling_path = out_dir / "reward_density_source_label_reachability_coupling.csv"
    report["outputs"] = {
        "json": str(json_path),
        "markdown": str(md_path),
        "per_env_csv": str(per_env_path),
        "coupling_csv": str(coupling_path),
    }
    json_path.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(per_env_path, rows)
    _write_csv(coupling_path, coupling)
    md_path.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report["outputs"], sort_keys=True))
    return report


def _markdown(report: dict[str, Any]) -> str:
    read = report["read"]
    lines = [
        "# Reward Density Source Label Reachability Audit",
        "",
        "Read-only audit.  No generator patch, no h-list selection, no training.",
        "",
        "## Decision",
        "",
        f"- dominant reward allocation hard axis: `{read['dominant_reward_allocation_hard_axis_decision']}`",
        f"- reason: {read['decision_reason']}",
        f"- source-label sampler permitted: `{read['source_label_sampler_permitted']}`",
        f"- sampler reason: {read['source_label_sampler_permitted_reason']}",
        f"- conditioned balance suppresses natural labels: `{read['conditioned_balance_suppresses_natural_labels']}`",
        f"- if promoted minimal labels: `{read['if_promoted_minimal_labels']}`",
        f"- diagnostic labels not hard: `{read['diagnostic_labels_not_hard']}`",
        "",
        "## Mode Counts",
        "",
        "| mode | n | balanced | state_dom | action_dom | noise_dom | skewed | missing | natural entropy | high coupling |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode, summary in report["mode_summaries"].items():
        counts = summary["counts"]
        lines.append(
            "| "
            f"`{mode}` | {summary['n']} | {counts.get('density_balanced', 0)} | "
            f"{counts.get('state_density_dominant', 0)} | {counts.get('action_density_dominant', 0)} | "
            f"{counts.get('noise_density_dominant', 0)} | {counts.get('density_skewed_no_clear_dominant', 0)} | "
            f"{counts.get('missing', 0)} | {summary['natural_entropy']:.4f} | "
            f"{summary['high_structure_coupling_count']} |"
        )
    lines.extend(["", "## Density Ratio q10/q50/q90", ""])
    for mode, summary in report["mode_summaries"].items():
        lines.append(f"### `{mode}`")
        lines.extend(["", "| metric | q10 | q50 | q90 |", "|---|---:|---:|---:|"])
        for key, stats in summary["stats"].items():
            lines.append(f"| `{key}` | {stats['q10']} | {stats['q50']} | {stats['q90']} |")
        lines.append("")
    highs = [row for row in report["structure_coupling"] if row["status"] == "high"]
    lines.extend(["## High Structure Coupling", ""])
    if highs:
        for row in highs:
            lines.append(
                f"- mode `{row['mode']}`: `{row['axis']}` vs `{row['structure_axis']}` V={row['cramers_v']:.4f}"
            )
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=51018)
    parser.add_argument("--env-seed-base", type=int, default=91018)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dominance-threshold", type=float, default=1.5)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["direct_unconditioned", "conditioned_balance"],
        choices=["direct_unconditioned", "direct_flag_on", "conditioned_balance"],
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
