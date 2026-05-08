#!/usr/bin/env python
"""Guard audit for exploratory gated fixed prior lists.

This script only reads the fixed-list probe outputs.  It does not sample new
environments, train PPO, or touch the exact-SCM/default prior path.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_PROBE_DIR = "/home/chen/RLPFN/artifacts/phase2_exploratory_reward_group_balance_gated_n64_0506"
DEFAULT_OUTPUT_DIR = "/home/chen/RLPFN/artifacts/phase2_exploratory_gated_fixedlist_guard_audit_0506"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _stats(values: list[float]) -> dict[str, Any]:
    arr = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=np.float64)
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "q10": float(np.quantile(arr, 0.10)),
        "q50": float(np.quantile(arr, 0.50)),
        "q90": float(np.quantile(arr, 0.90)),
        "max": float(np.max(arr)),
    }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _unique_count(rows: list[dict[str, str]], key: str) -> int:
    return int(len({str(row.get(key, "")) for row in rows}))


def _value_stats(rows: list[dict[str, str]], key: str) -> dict[str, Any]:
    values: list[float] = []
    for row in rows:
        try:
            values.append(float(row.get(key, "nan")))
        except Exception:
            pass
    return _stats(values)


def _rule_guard(
    *,
    rule: str,
    rows: list[dict[str, str]],
    report: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    diversity = ((report.get("transition_diversity_probe") or {}).get(rule) or {})
    n = int(len(rows))
    actual_state_gains: list[float] = []
    actual_action_gains: list[float] = []
    actual_ratios: list[float] = []
    for row in rows:
        use_balanced = str(row.get("selected_balance_applied", "")).strip().lower() in {"1", "true", "yes"}
        prefix = "balanced_" if use_balanced else ""
        try:
            actual_state_gains.append(float(row[f"{prefix}reward_state_input_gain_fraction"]))
            actual_action_gains.append(float(row[f"{prefix}reward_action_input_gain_fraction"]))
            actual_ratios.append(float(row[f"{prefix}reward_state_to_action_gain_ratio"]))
        except Exception:
            continue
    topology = {
        "state_gain_min": min(actual_state_gains) if actual_state_gains else None,
        "action_gain_min": min(actual_action_gains) if actual_action_gains else None,
        "state_to_action_ratio_max": max(actual_ratios) if actual_ratios else None,
        "actual_state_gain": _stats(actual_state_gains),
        "actual_action_gain": _stats(actual_action_gains),
        "actual_state_to_action_ratio": _stats(actual_ratios),
        "actual_topology_note": (
            "Per row, use balanced_* metrics only when selected_balance_applied=True; "
            "otherwise use original metrics because the frozen_h is kept original."
        ),
    }
    topology_pass = (
        topology["state_gain_min"] is not None
        and float(topology["state_gain_min"]) >= float(args.state_gain_min)
        and topology["action_gain_min"] is not None
        and float(topology["action_gain_min"]) >= float(args.action_gain_min)
        and topology["state_to_action_ratio_max"] is not None
        and float(topology["state_to_action_ratio_max"]) <= float(args.state_to_action_ratio_max)
    )
    reward_raw_std = diversity.get("reward_raw_std")
    state_delta_rms = diversity.get("state_delta_rms")
    terminal_signal_std = diversity.get("terminal_signal_std")
    finite_reward_fraction = diversity.get("finite_reward_fraction")
    finite_state_delta_fraction = diversity.get("finite_state_delta_fraction")
    diversity_pass = (
        finite_reward_fraction is not None
        and float(finite_reward_fraction) >= 1.0
        and finite_state_delta_fraction is not None
        and float(finite_state_delta_fraction) >= 1.0
        and reward_raw_std is not None
        and float(reward_raw_std) >= float(args.reward_std_min)
        and state_delta_rms is not None
        and float(state_delta_rms) >= float(args.state_delta_rms_min)
        and terminal_signal_std is not None
        and float(terminal_signal_std) >= float(args.terminal_signal_std_min)
    )
    support = {
        "state_dim_unique": _unique_count(rows, "state_dim"),
        "obs_dim_unique": _unique_count(rows, "obs_dim"),
        "action_dim_unique": _unique_count(rows, "action_dim"),
        "noise_dim_unique": _unique_count(rows, "noise_dim"),
        "repair_kind_counts": dict(Counter(str(row.get("selected_repair_kind", "")) for row in rows)),
        "balance_applied_counts": dict(Counter(str(row.get("selected_balance_applied", "")) for row in rows)),
        "state_dim": _value_stats(rows, "state_dim"),
        "obs_dim": _value_stats(rows, "obs_dim"),
        "action_dim": _value_stats(rows, "action_dim"),
        "noise_dim": _value_stats(rows, "noise_dim"),
    }
    support_pass = (
        n >= int(args.expected_n)
        and support["state_dim_unique"] >= int(args.min_state_dim_unique)
        and support["obs_dim_unique"] >= int(args.min_obs_dim_unique)
        and support["action_dim_unique"] >= int(args.min_action_dim_unique)
        and support["noise_dim_unique"] >= int(args.min_noise_dim_unique)
    )
    return {
        "rule": rule,
        "n": n,
        "hard_pass": bool(topology_pass and diversity_pass and support_pass),
        "topology_pass": bool(topology_pass),
        "diversity_pass": bool(diversity_pass),
        "support_pass": bool(support_pass),
        "topology": topology,
        "diversity": {
            "finite_reward_fraction": finite_reward_fraction,
            "finite_state_delta_fraction": finite_state_delta_fraction,
            "reward_raw_std": reward_raw_std,
            "state_delta_rms": state_delta_rms,
            "terminal_signal_std": terminal_signal_std,
            "terminal_signal_note": "Pre-smoke done proxy only; actual done_count is checked by the pack rollout smoke.",
            "reward_raw": diversity.get("reward_raw"),
            "state_delta_abs": diversity.get("state_delta_abs"),
            "terminal_signal": diversity.get("terminal_signal"),
        },
        "support": support,
        "thresholds": {
            "state_gain_min": float(args.state_gain_min),
            "action_gain_min": float(args.action_gain_min),
            "state_to_action_ratio_max": float(args.state_to_action_ratio_max),
            "reward_std_min": float(args.reward_std_min),
            "state_delta_rms_min": float(args.state_delta_rms_min),
            "terminal_signal_std_min": float(args.terminal_signal_std_min),
            "min_state_dim_unique": int(args.min_state_dim_unique),
            "min_obs_dim_unique": int(args.min_obs_dim_unique),
            "min_action_dim_unique": int(args.min_action_dim_unique),
            "min_noise_dim_unique": int(args.min_noise_dim_unique),
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    probe_dir = Path(args.probe_dir).expanduser().resolve()
    report = _load_json(probe_dir / "reward_group_balance_probe_report.json")
    rows = _load_csv(probe_dir / "selected_prior_envs.csv")
    rules = list(report.get("rules") or sorted({str(row.get("rule", "")) for row in rows}))
    rows_by_rule = {
        rule: [row for row in rows if str(row.get("rule", "")) == str(rule)]
        for rule in rules
    }
    rule_reports = {
        rule: _rule_guard(rule=rule, rows=rows_by_rule[rule], report=report, args=args)
        for rule in rules
    }
    hard_pass = all(bool(item["hard_pass"]) for item in rule_reports.values())
    out = {
        "analysis_entry": "phase2_gated_fixedlist_guard_audit",
        "exploratory_only": True,
        "hard_pass": bool(hard_pass),
        "contract": {
            "does_not_modify_exact_scm": True,
            "does_not_modify_default_prior_generator": True,
            "does_not_sample_environments": True,
            "does_not_run_ppo_training": True,
            "done_before_smoke_is_terminal_signal_proxy": True,
        },
        "input_artifacts": {
            "probe_dir": str(probe_dir),
            "probe_report": str(probe_dir / "reward_group_balance_probe_report.json"),
            "selected_csv": str(probe_dir / "selected_prior_envs.csv"),
        },
        "probe_sampling": report.get("sampling", {}),
        "selected_count_by_rule": report.get("selected_count_by_rule", {}),
        "rule_reports": rule_reports,
        "decision": {
            "short_smoke_allowed": bool(hard_pass),
            "reason": (
                "Topology, finite reward/state transition, terminal-signal proxy, and dimension-support "
                "guards passed." if hard_pass else "At least one guard failed; do not run PPO smoke yet."
            ),
        },
    }
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "gated_fixedlist_guard_audit_report.json"
    report_path.write_text(json.dumps(_json_safe(out), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_lines = [
        "# Gated Fixed-List Guard Audit",
        "",
        f"hard_pass: {hard_pass}",
        "",
        "## Rules",
    ]
    for rule, item in rule_reports.items():
        md_lines.append(
            f"- {rule}: hard={item['hard_pass']}, topology={item['topology_pass']}, "
            f"diversity={item['diversity_pass']}, support={item['support_pass']}, "
            f"reward_std={item['diversity']['reward_raw_std']}, "
            f"state_delta_rms={item['diversity']['state_delta_rms']}, "
            f"terminal_signal_std={item['diversity']['terminal_signal_std']}, "
            f"action_dim_unique={item['support']['action_dim_unique']}"
        )
    md_lines.extend(
        [
            "",
            "Note: pre-smoke done diversity is terminal-signal proxy only. Actual done_count is checked in the pack rollout smoke.",
        ]
    )
    md_path = out_dir / "gated_fixedlist_guard_audit_summary.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-dir", type=str, default=DEFAULT_PROBE_DIR)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--expected-n", type=int, default=64)
    parser.add_argument("--state-gain-min", type=float, default=0.7)
    parser.add_argument("--action-gain-min", type=float, default=0.06)
    parser.add_argument("--state-to-action-ratio-max", type=float, default=10.0)
    parser.add_argument("--reward-std-min", type=float, default=0.1)
    parser.add_argument("--state-delta-rms-min", type=float, default=0.05)
    parser.add_argument("--terminal-signal-std-min", type=float, default=0.05)
    parser.add_argument("--min-state-dim-unique", type=int, default=8)
    parser.add_argument("--min-obs-dim-unique", type=int, default=8)
    parser.add_argument("--min-action-dim-unique", type=int, default=4)
    parser.add_argument("--min-noise-dim-unique", type=int, default=8)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps(_json_safe({"hard_pass": report["hard_pass"], "output_dir": args.output_dir}), sort_keys=True))


if __name__ == "__main__":
    main()
