#!/usr/bin/env python
"""Search bounded cheap guards for Pendulum sustained-settle semantics.

This is read-only and intentionally conservative.  It searches simple
conjunctions of already-computed signature fields for a guard that is both
count-safe and clean relative to the current strict settle target
(`pendulum_like_settle_strict`).  It does not run rollouts, train PPO, or add a
new default generator path.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ART = Path("/home/chen/RLPFN/artifacts")
DEFAULT_SIGNATURE_REPORTS = (
    ART / "phase2_prior_pendulum_like_signature_probe_0509/gated_full_q90_n64_s128_obs4_scales3_timeprofile.json",
    ART
    / "phase2_prior_pendulum_like_signature_probe_seed6261_0509"
    / "gated_full_q90_seed6261_n64_s128_obs4_scales3_timeprofile.json",
    ART
    / "phase2_prior_pendulum_like_signature_probe_seed6262_0510"
    / "gated_full_q90_seed6262_n128_s128_obs4_scales3_timeprofile.json",
)
DEFAULT_OUTPUT_DIR = ART / "phase2_pendulum_settle_guard_search_0512"

BOOL_PRIMITIVES = (
    "nonzero_required_env",
    "feedback_beats_open_loop_env",
    "feedback_beats_zero_env",
    "phase_sensitive_feedback",
    "suffix_supported",
    "suffix_dominant_gain",
    "feedback_action_decay",
    "time_profile_core",
    "time_profile_suffix_supported",
    "time_profile_action_decay",
    "time_profile_beats_open_loop_env",
    "time_profile_beats_zero_env",
    "early_only_action_decay",
    "early_only_suffix_carryover_over_zero",
    "early_only_suffix_carryover_over_open",
    "early_only_causal_carryover_core",
    "early_only_strong_carryover_core",
    "pendulum_like_core",
    "pendulum_like_strict",
    "pendulum_like_weak",
    "prefix_oracle_is_feedback",
    "prefix_oracle_is_time_profile",
    "prefix_oracle_beats_open_loop",
)

NUMERIC_PRIMITIVES = (
    "feedback_suffix_gain_over_zero_env",
    "feedback_suffix_gain_over_open_env",
    "best_feedback_action_abs_prefix_minus_suffix",
    "best_time_profile_action_abs_prefix",
    "best_time_profile_action_abs_suffix",
    "best_time_profile_prefix_env_return",
    "best_time_profile_suffix_env_return",
    "best_early_only_suffix_gain_over_zero_env",
    "best_early_only_suffix_gain_over_open_env",
    "best_early_only_action_abs_prefix",
    "best_early_only_action_abs_suffix",
    "feedback_pair_phase_gap_env",
    "best_feedback_minus_open_env",
    "best_feedback_minus_zero_env",
)


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return math.nan
    return out if math.isfinite(out) else math.nan


def _load_rows(signature_reports: list[str | Path]) -> tuple[list[dict[str, Any]], np.ndarray, list[str]]:
    rows: list[dict[str, Any]] = []
    labels: list[str] = []
    for path in signature_reports:
        payload = _read_json(path)
        for item in payload.get("lists", []) or []:
            label = str(item.get("label", ""))
            for row in ((item.get("classification") or {}).get("per_env") or []):
                out = dict(row)
                out["_list_label"] = label
                out["_report_path"] = str(Path(path).expanduser().resolve())
                rows.append(out)
                labels.append(label)
    unique = {label: idx for idx, label in enumerate(dict.fromkeys(labels))}
    group = np.asarray([unique[label] for label in labels], dtype=np.int64)
    return rows, group, list(unique)


def _primitive_arrays(rows: list[dict[str, Any]], quantiles: tuple[float, ...]) -> list[dict[str, Any]]:
    primitives: list[dict[str, Any]] = []
    for key in BOOL_PRIMITIVES:
        values = np.asarray([bool(row.get(key)) for row in rows], dtype=bool)
        primitives.append(
            {
                "name": key,
                "kind": "bool",
                "field": key,
                "values": values,
                "complexity": 1,
            }
        )
    for key in NUMERIC_PRIMITIVES:
        raw = np.asarray([_safe_float(row.get(key)) for row in rows], dtype=np.float64)
        finite = raw[np.isfinite(raw)]
        if finite.size == 0:
            continue
        thresholds = sorted({float(np.quantile(finite, q)) for q in quantiles})
        for threshold in thresholds:
            for op in ("ge", "le"):
                if op == "ge":
                    values = raw >= threshold
                    symbol = ">="
                else:
                    values = raw <= threshold
                    symbol = "<="
                primitives.append(
                    {
                        "name": f"{key} {symbol} {threshold:.6g}",
                        "kind": op,
                        "field": key,
                        "threshold": threshold,
                        "values": values.astype(bool, copy=False),
                        "complexity": 1,
                    }
                )
    return primitives


def _metrics(pred: np.ndarray, target: np.ndarray, group: np.ndarray, labels: list[str]) -> dict[str, Any]:
    rows = []
    for idx, label in enumerate(labels):
        mask = group == idx
        p = pred[mask]
        y = target[mask]
        count = int(p.sum())
        target_count = int(y.sum())
        overlap = int(np.logical_and(p, y).sum())
        n = int(mask.sum())
        rows.append(
            {
                "label": label,
                "n_envs": n,
                "count": count,
                "rate": float(count / max(1, n)),
                "target_count": target_count,
                "overlap_with_target": overlap,
                "precision_vs_target": float(overlap / max(1, count)),
                "recall_vs_target": float(overlap / max(1, target_count)),
            }
        )
    return {
        "min_count": min((row["count"] for row in rows), default=0),
        "max_count": max((row["count"] for row in rows), default=0),
        "min_rate": min((row["rate"] for row in rows), default=0.0),
        "min_precision_vs_target": min((row["precision_vs_target"] for row in rows), default=0.0),
        "min_recall_vs_target": min((row["recall_vs_target"] for row in rows), default=0.0),
        "per_list": rows,
    }


def _formula_text(items: tuple[dict[str, Any], ...]) -> str:
    return " AND ".join(item["name"] for item in items)


def build_report(
    *,
    signature_reports: list[str | Path],
    max_conjunction_size: int = 2,
    min_count: int = 12,
    min_rate: float = 0.15,
    min_precision: float = 0.50,
    min_recall: float = 0.80,
    top_k: int = 25,
) -> dict[str, Any]:
    rows, group, labels = _load_rows(signature_reports)
    target = np.asarray([bool(row.get("pendulum_like_settle_strict")) for row in rows], dtype=bool)
    primitives = _primitive_arrays(rows, quantiles=(0.25, 0.5, 0.75))

    candidates: list[dict[str, Any]] = []
    ready: list[dict[str, Any]] = []
    for size in range(1, int(max_conjunction_size) + 1):
        for combo in itertools.combinations(primitives, size):
            pred = combo[0]["values"].copy()
            for item in combo[1:]:
                pred &= item["values"]
            metric = _metrics(pred, target, group, labels)
            if metric["max_count"] <= 0:
                continue
            score = (
                2.0 * metric["min_precision_vs_target"]
                + 1.5 * metric["min_recall_vs_target"]
                + min(metric["min_count"], min_count) / max(1, min_count)
                + 0.25 * metric["min_rate"]
                - 0.05 * (size - 1)
            )
            row = {
                "formula": _formula_text(combo),
                "size": size,
                "score": float(score),
                **metric,
            }
            candidates.append(row)
            if (
                metric["min_count"] >= min_count
                and metric["min_rate"] >= min_rate
                and metric["min_precision_vs_target"] >= min_precision
                and metric["min_recall_vs_target"] >= min_recall
            ):
                ready.append(row)

    candidates.sort(key=lambda item: item["score"], reverse=True)
    ready.sort(key=lambda item: item["score"], reverse=True)
    top = candidates[: int(top_k)]

    return {
        "analysis_entry": "phase2_pendulum_settle_guard_search",
        "contract": {
            "read_only_existing_signature_reports": True,
            "does_not_sample_environments": True,
            "does_not_train": True,
            "does_not_modify_exact_scm": True,
            "does_not_modify_fit_model": True,
            "does_not_modify_ppo": True,
            "bounded_search_only": True,
            "does_not_add_profile_axis": True,
        },
        "inputs": {"signature_reports": [str(Path(path).expanduser().resolve()) for path in signature_reports]},
        "thresholds": {
            "max_conjunction_size": int(max_conjunction_size),
            "min_count": int(min_count),
            "min_rate": float(min_rate),
            "min_precision_vs_target": float(min_precision),
            "min_recall_vs_target": float(min_recall),
            "target": "pendulum_like_settle_strict",
        },
        "data": {
            "n_rows": len(rows),
            "n_lists": len(labels),
            "list_labels": labels,
            "n_primitives": len(primitives),
            "n_candidates_scored": len(candidates),
            "target_count_by_list": {
                label: int(target[group == idx].sum()) for idx, label in enumerate(labels)
            },
        },
        "ready_candidates": ready[: int(top_k)],
        "top_candidates": top,
        "decision": {
            "cheap_formula_guard_ready": bool(ready),
            "best_formula": ready[0]["formula"] if ready else (top[0]["formula"] if top else None),
            "read": (
                "A bounded cheap formula guard exists under the configured thresholds."
                if ready
                else "No simple bounded formula over existing signature fields is both broad and clean enough. "
                "This further supports treating Pendulum settle as a generator/yield or new cheap-label problem, "
                "not as a threshold-only repair."
            ),
            "next_action": (
                "Validate the ready formula on a fresh full/q90 source pool and diversity smoke."
                if ready
                else "Do not quota existing broad guards.  Either implement a new cheap suffix-state settle label "
                "or repair the generator mechanism that creates clean feedback-decay suffix-supported cases."
            ),
        },
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Pendulum Settle Guard Search",
        "",
        "Bounded read-only search over existing signature fields.",
        "",
        "## Verdict",
        "",
        f"- cheap formula guard ready: `{report['decision']['cheap_formula_guard_ready']}`",
        f"- best formula: `{report['decision']['best_formula']}`",
        f"- read: {report['decision']['read']}",
        f"- next action: {report['decision']['next_action']}",
        "",
        "## Data",
        "",
    ]
    for key, value in report["data"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Top Candidates", ""])
    lines.append("| formula | min count | max count | min precision | min recall | min rate |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for item in report["top_candidates"][:10]:
        lines.append(
            f"| `{item['formula']}` | {item['min_count']} | {item['max_count']} | "
            f"{item['min_precision_vs_target']:.3f} | {item['min_recall_vs_target']:.3f} | "
            f"{item['min_rate']:.3f} |"
        )
    lines.extend(["", "## Files", ""])
    for key, value in report.get("outputs", {}).items():
        lines.append(f"- `{key}`: `{value}`")
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(
        signature_reports=list(args.signature_report),
        max_conjunction_size=args.max_conjunction_size,
        min_count=args.min_count,
        min_rate=args.min_rate,
        min_precision=args.min_precision,
        min_recall=args.min_recall,
        top_k=args.top_k,
    )
    json_path = out_dir / "pendulum_settle_guard_search.json"
    md_path = out_dir / "pendulum_settle_guard_search.md"
    report["outputs"] = {"json": str(json_path), "markdown": str(md_path)}
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report["outputs"], sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signature-report", action="append", default=[str(path) for path in DEFAULT_SIGNATURE_REPORTS])
    parser.add_argument("--max-conjunction-size", type=int, default=2)
    parser.add_argument("--min-count", type=int, default=12)
    parser.add_argument("--min-rate", type=float, default=0.15)
    parser.add_argument("--min-precision", type=float, default=0.50)
    parser.add_argument("--min-recall", type=float, default=0.80)
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
