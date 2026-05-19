#!/usr/bin/env python
"""Decompose the Pendulum suffix-state settle blocker.

Consumes the bounded suffix-state preflight and separates three possibilities:

1. reward/action evidence is missing,
2. strict state-settle evidence is missing,
3. a very loose non-worsening state proxy would pass, but is too weak to stand
   in for the Gym Pendulum closed-loop settle anchor.

This report is read-only.  It is a governance artifact for deciding whether the
next repair should be a selector threshold, a cached label, or a generator
mechanism change.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ART = Path("/home/chen/RLPFN/artifacts")
DEFAULT_PREFLIGHT_JSON = (
    ART / "phase2_pendulum_settle_state_label_preflight_0512/pendulum_settle_state_label_preflight.json"
)
DEFAULT_OUTPUT_DIR = ART / "phase2_pendulum_settle_state_blocker_decomposition_0512"


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def _count_loose(
    rows: list[dict[str, Any]],
    *,
    self_rel: float,
    open_rel: float,
    drift_eps: float = 1e-6,
) -> int:
    count = 0
    for row in rows:
        self_ok = float(row["best_decay_state_drift_suffix"]) <= float(self_rel) * max(
            float(drift_eps), float(row["best_decay_state_drift_prefix"])
        )
        open_ok = float(row["best_decay_state_drift_suffix"]) <= float(open_rel) * max(
            float(drift_eps), float(row["best_open_state_drift_suffix"])
        )
        if (
            bool(row["reward_condition"])
            and bool(row["suffix_reward_condition"])
            and bool(row["action_decay_condition"])
            and self_ok
            and open_ok
        ):
            count += 1
    return int(count)


def _dominant_bottleneck(coverage: dict[str, Any]) -> str:
    counts = {
        "reward_condition": int(coverage.get("reward_condition_count", 0)),
        "suffix_reward_condition": int(coverage.get("suffix_reward_condition_count", 0)),
        "action_decay_condition": int(coverage.get("action_decay_condition_count", 0)),
        "self_state_settle_condition": int(coverage.get("self_state_settle_condition_count", 0)),
        "open_state_settle_condition": int(coverage.get("open_state_settle_condition_count", 0)),
    }
    return min(counts, key=counts.get)


def build_report(
    *,
    preflight_json: str | Path,
    min_count: int = 8,
) -> dict[str, Any]:
    preflight = _read_json(preflight_json)
    list_reports = []
    threshold_grid: list[dict[str, Any]] = []
    self_grid = [0.85, 0.90, 0.95, 1.00, 1.05]
    open_grid = [0.90, 0.95, 1.00, 1.05]
    for self_rel in self_grid:
        for open_rel in open_grid:
            counts = [
                _count_loose(
                    item["classification"]["per_env"],
                    self_rel=float(self_rel),
                    open_rel=float(open_rel),
                )
                for item in preflight.get("lists", [])
            ]
            threshold_grid.append(
                {
                    "self_drift_rel_max": float(self_rel),
                    "open_drift_rel_max": float(open_rel),
                    "counts": counts,
                    "min_count": int(min(counts) if counts else 0),
                    "count_safe": bool(counts and min(counts) >= int(min_count)),
                    "semantic_strength": (
                        "strict_contraction"
                        if float(self_rel) < 1.0 and float(open_rel) < 1.0
                        else "non_worsening_or_looser"
                    ),
                }
            )
    for item in preflight.get("lists", []):
        coverage = item["classification"]["coverage"]
        list_reports.append(
            {
                "label": item["label"],
                "n_envs": int(coverage["n_envs"]),
                "condition_counts": {
                    "reward_condition": int(coverage["reward_condition_count"]),
                    "suffix_reward_condition": int(coverage["suffix_reward_condition_count"]),
                    "action_decay_condition": int(coverage["action_decay_condition_count"]),
                    "self_state_settle_condition": int(coverage["self_state_settle_condition_count"]),
                    "open_state_settle_condition": int(coverage["open_state_settle_condition_count"]),
                    "cheap_settle_guard": int(coverage["cheap_settle_guard_count"]),
                },
                "dominant_bottleneck": _dominant_bottleneck(coverage),
                "state_drift_ratio_means": {
                    "suffix_over_prefix": float(
                        item["classification"]["scores"]["best_decay_suffix_drift_over_prefix_drift"]["mean"]
                    ),
                    "suffix_over_open_suffix": float(
                        item["classification"]["scores"]["best_decay_suffix_drift_over_open_suffix_drift"]["mean"]
                    ),
                },
            }
        )
    strict_count_safe = bool(
        preflight.get("decision", {}).get("cheap_suffix_state_settle_guard_ready")
    )
    loose_count_safe = any(
        bool(row["count_safe"]) and row["semantic_strength"] == "non_worsening_or_looser"
        for row in threshold_grid
    )
    strict_contraction_count_safe = any(
        bool(row["count_safe"]) and row["semantic_strength"] == "strict_contraction"
        for row in threshold_grid
    )
    state_bottleneck = bool(
        list_reports
        and all(
            row["dominant_bottleneck"] in {"self_state_settle_condition", "open_state_settle_condition"}
            for row in list_reports
        )
    )
    return {
        "analysis_entry": "phase2_pendulum_settle_state_blocker_decomposition",
        "contract": {
            "read_only_existing_artifacts": True,
            "does_not_sample_environments": True,
            "does_not_train": True,
            "does_not_modify_exact_scm": True,
            "does_not_modify_fit_model": True,
            "not_a_raw_delta_gate": True,
        },
        "inputs": {"preflight_json": str(Path(preflight_json).expanduser().resolve())},
        "lists": list_reports,
        "threshold_sensitivity": threshold_grid,
        "decision": {
            "strict_suffix_state_guard_count_safe": strict_count_safe,
            "strict_contraction_count_safe_under_grid": strict_contraction_count_safe,
            "loose_non_worsening_count_safe_under_grid": loose_count_safe,
            "dominant_blocker_is_state_basin": state_bottleneck,
            "selector_threshold_repair_allowed": bool(strict_count_safe or strict_contraction_count_safe),
            "loose_guard_not_semantically_sufficient": bool(loose_count_safe and not strict_contraction_count_safe),
            "recommended_next": (
                "Do not close Pendulum by loosening state thresholds. The missing piece is a generator/yield "
                "mechanism for action-sensitive state-basin settling; a loose non-worsening guard is only an "
                "offline diagnostic."
                if state_bottleneck and not strict_contraction_count_safe
                else "A strict contraction guard may be viable; validate anchor precision before quota."
            ),
        },
    }


def _markdown(report: dict[str, Any]) -> str:
    d = report["decision"]
    lines = [
        "# Pendulum State-Settle Blocker Decomposition",
        "",
        f"- strict suffix-state guard count-safe: `{d['strict_suffix_state_guard_count_safe']}`",
        f"- strict contraction count-safe under grid: `{d['strict_contraction_count_safe_under_grid']}`",
        f"- loose non-worsening count-safe under grid: `{d['loose_non_worsening_count_safe_under_grid']}`",
        f"- dominant blocker is state basin: `{d['dominant_blocker_is_state_basin']}`",
        f"- selector threshold repair allowed: `{d['selector_threshold_repair_allowed']}`",
        f"- read: {d['recommended_next']}",
        "",
        "## Condition Counts",
        "",
        "| list | reward | suffix reward | action decay | self state settle | open state settle | guard | bottleneck | drift suffix/prefix | drift suffix/open |",
        "|---|---:|---:|---:|---:|---:|---:|---|---:|---:|",
    ]
    for row in report["lists"]:
        c = row["condition_counts"]
        ratios = row["state_drift_ratio_means"]
        lines.append(
            f"| `{row['label']}` | {c['reward_condition']} | {c['suffix_reward_condition']} | "
            f"{c['action_decay_condition']} | {c['self_state_settle_condition']} | "
            f"{c['open_state_settle_condition']} | {c['cheap_settle_guard']} | "
            f"`{row['dominant_bottleneck']}` | {ratios['suffix_over_prefix']:.3f} | "
            f"{ratios['suffix_over_open_suffix']:.3f} |"
        )
    lines.extend(["", "## Count-Safe Threshold Grid", ""])
    for row in report["threshold_sensitivity"]:
        if bool(row["count_safe"]):
            lines.append(
                f"- self<={row['self_drift_rel_max']:.2f}, open<={row['open_drift_rel_max']:.2f}: "
                f"counts={row['counts']} strength=`{row['semantic_strength']}`"
            )
    lines.extend(["", "## Files", ""])
    for key, value in report.get("outputs", {}).items():
        lines.append(f"- `{key}`: `{value}`")
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(preflight_json=args.preflight_json, min_count=int(args.min_count))
    json_path = out_dir / "pendulum_settle_state_blocker_decomposition.json"
    md_path = out_dir / "pendulum_settle_state_blocker_decomposition.md"
    report["outputs"] = {"json": str(json_path), "markdown": str(md_path)}
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report["decision"], indent=2, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-json", default=str(DEFAULT_PREFLIGHT_JSON))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--min-count", type=int, default=8)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
