#!/usr/bin/env python
"""Export Pendulum settle preflight JSON as generator-level cached labels.

The state/suffix settle probe already emits a bounded per-env classification
inside JSON.  This thin adapter turns that evidence into the same cached-label
CSV contract consumed by `phase2_future_basin_cached_label_adapter.py`.

It is intentionally boring: no sampling, no PPO, no selector decisions, and no
quota.  It only preserves per-rule source indices and makes label availability
explicit.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


ART = Path("/home/chen/RLPFN/artifacts")
DEFAULT_INPUT_JSON = (
    ART / "phase2_pendulum_settle_state_label_preflight_0512/pendulum_settle_state_label_preflight.json"
)
DEFAULT_OUTPUT_DIR = ART / "phase2_pendulum_settle_cached_label_export_0512"


def _read_json(path: str | Path) -> Any:
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


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    if not fields:
        fields = ["rule", "source_index"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _summaries(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for rule in sorted({str(row.get("rule")) for row in rows}):
        rule_rows = [row for row in rows if str(row.get("rule")) == rule]
        n = max(1, len(rule_rows))
        out[rule] = {
            "n": len(rule_rows),
            "pendulum_short_temporal_settle_guard_count": int(
                sum(bool(row.get("pendulum_short_temporal_settle_guard")) for row in rule_rows)
            ),
            "pendulum_short_temporal_settle_guard_rate": float(
                sum(bool(row.get("pendulum_short_temporal_settle_guard")) for row in rule_rows) / n
            ),
            "reward_condition_rate": float(sum(bool(row.get("reward_condition")) for row in rule_rows) / n),
            "suffix_reward_condition_rate": float(
                sum(bool(row.get("suffix_reward_condition")) for row in rule_rows) / n
            ),
            "action_decay_condition_rate": float(
                sum(bool(row.get("action_decay_condition")) for row in rule_rows) / n
            ),
            "pendulum_medium_settle_count": int(
                sum(bool(row.get("pendulum_medium_settle")) for row in rule_rows)
            ),
            "pendulum_medium_settle_rate": float(
                sum(bool(row.get("pendulum_medium_settle")) for row in rule_rows) / n
            ),
            "pendulum_strict_settle_count": int(
                sum(bool(row.get("pendulum_strict_settle")) for row in rule_rows)
            ),
            "pendulum_strict_settle_rate": float(
                sum(bool(row.get("pendulum_strict_settle")) for row in rule_rows) / n
            ),
            "medium_state_nonworse_condition_rate": float(
                sum(bool(row.get("medium_state_nonworse_condition")) for row in rule_rows) / n
            ),
            "self_state_settle_condition_rate": float(
                sum(bool(row.get("self_state_settle_condition")) for row in rule_rows) / n
            ),
            "open_state_settle_condition_rate": float(
                sum(bool(row.get("open_state_settle_condition")) for row in rule_rows) / n
            ),
        }
    return out


def export_cached_labels(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in report.get("lists", []) or []:
        rule = str(item.get("label", ""))
        classification = item.get("classification", {}) or {}
        for row in classification.get("per_env", []) or []:
            source_index = int(row.get("env_index", row.get("source_index", 0)))
            suffix_reward = bool(row.get("suffix_reward_condition", False))
            action_decay = bool(row.get("action_decay_condition", False))
            medium_state = bool(
                row.get(
                    "medium_state_nonworse_condition",
                    row.get("self_state_settle_condition", False),
                )
            )
            medium_settle = bool(
                row.get(
                    "medium_settle_guard",
                    suffix_reward and action_decay and medium_state,
                )
            )
            strict_settle = bool(row.get("strict_settle_guard", row.get("cheap_settle_guard", False)))
            rows.append(
                {
                    "rule": rule,
                    "source_index": source_index,
                    "env_index": source_index,
                    "pendulum_axis_support": False,
                    "pendulum_short_gap": suffix_reward,
                    "pendulum_medium_settle": medium_settle,
                    "pendulum_strict_settle": strict_settle,
                    "pendulum_sustained_settle": medium_settle,
                    "pendulum_profile_quota_ready": False,
                    "pendulum_short_temporal_settle_guard": strict_settle,
                    "pendulum_label_available": True,
                    "pendulum_settle_guard_precision_class": "medium_suffix_state_guard_with_strict_audit",
                    "reward_condition": bool(row.get("reward_condition", False)),
                    "suffix_reward_condition": suffix_reward,
                    "action_decay_condition": action_decay,
                    "medium_state_nonworse_condition": medium_state,
                    "self_state_settle_condition": bool(row.get("self_state_settle_condition", False)),
                    "open_state_settle_condition": bool(row.get("open_state_settle_condition", False)),
                    "best_decay_mode": row.get("best_decay_mode", ""),
                    "best_open_loop_mode": row.get("best_open_loop_mode", ""),
                    "best_decay_minus_open_env": row.get("best_decay_minus_open_env", ""),
                    "best_decay_suffix_minus_best_baseline_env": row.get(
                        "best_decay_suffix_minus_best_baseline_env", ""
                    ),
                    "best_decay_action_prefix_minus_suffix": row.get(
                        "best_decay_action_prefix_minus_suffix", ""
                    ),
                    "best_decay_state_drift_suffix": row.get("best_decay_state_drift_suffix", ""),
                    "best_open_state_drift_suffix": row.get("best_open_state_drift_suffix", ""),
                    "label_key": f"{rule}:{source_index}",
                }
            )
    return rows


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Pendulum Settle Cached Label Export",
        "",
        "Thin CSV export of bounded suffix-state settle labels.",
        "",
        "## Coverage",
        "",
        "| rule | n | medium | strict | reward | suffix reward | action decay | state nonworse | self settle | open settle |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rule, item in report["summaries"].items():
        lines.append(
            f"| `{rule}` | {item['n']} | {item['pendulum_medium_settle_rate']:.3f} | "
            f"{item['pendulum_strict_settle_rate']:.3f} | "
            f"{item['reward_condition_rate']:.3f} | {item['suffix_reward_condition_rate']:.3f} | "
            f"{item['action_decay_condition_rate']:.3f} | {item['medium_state_nonworse_condition_rate']:.3f} | "
            f"{item['self_state_settle_condition_rate']:.3f} | "
            f"{item['open_state_settle_condition_rate']:.3f} |"
        )
    lines.extend(["", "## Read", "", report["read"]["summary"]])
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    input_json = Path(args.input_json).expanduser().resolve()
    payload = _read_json(input_json)
    rows = export_cached_labels(payload)
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "pendulum_short_temporal_settle_per_env.csv"
    json_path = out_dir / "pendulum_settle_cached_label_export.json"
    md_path = out_dir / "pendulum_settle_cached_label_export.md"
    _write_csv(csv_path, rows)
    summaries = _summaries(rows)
    report = {
        "analysis_entry": "phase2_pendulum_settle_cached_label_export",
        "schema": "phase2_pendulum_settle_cached_label_export.v1",
        "contract": {
            "exploratory_only": True,
            "read_only_existing_preflight": True,
            "does_not_sample_environments": True,
            "does_not_train": True,
            "does_not_modify_exact_scm": True,
            "does_not_modify_fit_model": True,
            "does_not_modify_ppo": True,
            "profile_quality_first": True,
            "raw_delta_not_used": True,
            "missing_labels_are_explicit": True,
        },
        "inputs": {"input_json": str(input_json)},
        "summaries": summaries,
        "read": {
            "cached_label_populated": bool(rows),
            "guard_is_quota_ready": bool(
                summaries
                and min(item["pendulum_medium_settle_count"] for item in summaries.values())
                >= int(args.min_count)
                and min(item["pendulum_medium_settle_rate"] for item in summaries.values())
                >= float(args.min_rate)
            ),
            "summary": (
                "Pendulum cached label CSV populated from bounded suffix-state guard. "
                "Medium settle is evidence only; quota readiness still depends on health/diversity "
                "and profile-band protection."
            ),
        },
        "outputs": {"csv": str(csv_path), "json": str(json_path), "markdown": str(md_path)},
    }
    json_path.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report["outputs"], sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", default=str(DEFAULT_INPUT_JSON))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--min-count", type=int, default=8)
    parser.add_argument("--min-rate", type=float, default=0.10)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
