#!/usr/bin/env python
"""Decision parity check for the exploratory gated wrapper."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from phase2_gated_reward_path_generator_wrapper import (
    TopologyThresholds,
    decide_gated_reward_path_repair,
)


DEFAULT_PROBE_DIRS = (
    "/home/chen/RLPFN/artifacts/phase2_exploratory_reward_group_balance_gated_n64_0506,"
    "/home/chen/RLPFN/artifacts/phase2_exploratory_reward_group_balance_gated_n64_seed6261_0506"
)
DEFAULT_OUTPUT_DIR = "/home/chen/RLPFN/artifacts/phase2_exploratory_gated_wrapper_decision_parity_0506"


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _check_probe_dir(path: Path, thresholds: TopologyThresholds) -> dict[str, Any]:
    candidate_path = path / "reward_group_balance_candidates.csv"
    selected_path = path / "selected_prior_envs.csv"
    candidate_rows = _load_csv(candidate_path)
    selected_rows = _load_csv(selected_path)
    candidate_mismatches = []
    for idx, row in enumerate(candidate_rows):
        decision = decide_gated_reward_path_repair(row, thresholds=thresholds)
        expected = {
            "original_topology_pass": bool(decision.original_topology_pass),
            "balanced_topology_pass": bool(decision.balanced_topology_pass),
            "gated_topology_pass": bool(decision.gated_topology_pass),
            "gated_repair_kind": str(decision.repair_kind),
        }
        actual = {
            "original_topology_pass": _as_bool(row.get("original_topology_pass")),
            "balanced_topology_pass": _as_bool(row.get("balanced_topology_pass")),
            "gated_topology_pass": _as_bool(row.get("gated_topology_pass")),
            "gated_repair_kind": str(row.get("gated_repair_kind")),
        }
        if actual != expected:
            candidate_mismatches.append({"idx": int(idx), "actual": actual, "expected": expected})
    selected_mismatches = []
    for idx, row in enumerate(selected_rows):
        decision = decide_gated_reward_path_repair(row, thresholds=thresholds)
        actual_kind = str(row.get("selected_repair_kind"))
        actual_balance = _as_bool(row.get("selected_balance_applied"))
        if actual_kind != str(decision.repair_kind) or actual_balance != bool(decision.selected_balance_applied):
            selected_mismatches.append(
                {
                    "idx": int(idx),
                    "actual": {
                        "selected_repair_kind": actual_kind,
                        "selected_balance_applied": actual_balance,
                    },
                    "expected": {
                        "selected_repair_kind": str(decision.repair_kind),
                        "selected_balance_applied": bool(decision.selected_balance_applied),
                    },
                }
            )
    return {
        "probe_dir": str(path),
        "candidate_rows": int(len(candidate_rows)),
        "selected_rows": int(len(selected_rows)),
        "candidate_mismatch_count": int(len(candidate_mismatches)),
        "selected_mismatch_count": int(len(selected_mismatches)),
        "hard_pass": not candidate_mismatches and not selected_mismatches,
        "candidate_mismatches_head": candidate_mismatches[:10],
        "selected_mismatches_head": selected_mismatches[:10],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    thresholds = TopologyThresholds(
        state_gain_min=float(args.state_gain_min),
        action_gain_min=float(args.action_gain_min),
        state_to_action_ratio_max=float(args.state_to_action_ratio_max),
    )
    probe_dirs = [Path(part.strip()).expanduser().resolve() for part in str(args.probe_dirs).split(",") if part.strip()]
    checks = [_check_probe_dir(path, thresholds) for path in probe_dirs]
    report = {
        "analysis_entry": "phase2_gated_wrapper_decision_parity",
        "exploratory_only": True,
        "hard_pass": all(bool(check["hard_pass"]) for check in checks),
        "checks": checks,
    }
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "gated_wrapper_decision_parity_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-dirs", type=str, default=DEFAULT_PROBE_DIRS)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--state-gain-min", type=float, default=0.7)
    parser.add_argument("--action-gain-min", type=float, default=0.06)
    parser.add_argument("--state-to-action-ratio-max", type=float, default=10.0)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({"hard_pass": report["hard_pass"], "output_dir": args.output_dir}, sort_keys=True))


if __name__ == "__main__":
    main()
