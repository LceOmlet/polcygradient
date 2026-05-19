#!/usr/bin/env python
"""Audit whether existing generator parameters explain clean settle yield.

The goal is to avoid a false repair: if clean Pendulum-settle candidates are
mostly controlled by an existing h/metadata scalar, then a simple distribution
adjustment may be enough.  If no stable scalar split enriches clean settle
support, the next repair should target the generator/label mechanism directly.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Callable


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
DEFAULT_OUTPUT_DIR = ART / "phase2_pendulum_settle_h_feature_relation_audit_0512"


TARGETS: dict[str, Callable[[dict[str, Any]], bool]] = {
    "strict_current": lambda r: bool(r.get("pendulum_like_settle_strict")),
    "strict_no_suffix": lambda r: bool(r.get("pendulum_like_core")) and bool(r.get("feedback_action_decay")),
    "feedback_decay_suffix": lambda r: bool(r.get("feedback_action_decay")) and bool(r.get("suffix_supported")),
    "time_core_suffix": lambda r: bool(r.get("time_profile_core")) and bool(r.get("time_profile_suffix_supported")),
}


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def _is_num(value: Any) -> bool:
    return isinstance(value, (int, float, bool)) and math.isfinite(float(value))


def _flatten(signature_reports: list[str | Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in signature_reports:
        payload = _read_json(path)
        for item in payload.get("lists", []) or []:
            h_list = _read_json(item["fixed_h_list_json"])
            per_env = ((item.get("classification") or {}).get("per_env") or [])
            label = str(item.get("label", ""))
            for idx, (h, env) in enumerate(zip(h_list, per_env)):
                row = {
                    "report_path": str(Path(path).expanduser().resolve()),
                    "label": label,
                    "env_index": idx,
                }
                for key, fn in TARGETS.items():
                    row[key] = bool(fn(env))
                for key, value in h.items():
                    if _is_num(value):
                        row[f"h.{key}"] = float(value)
                rows.append(row)
    return rows


def _quantiles(values: list[float]) -> list[float]:
    if not values:
        return []
    sorted_values = sorted(values)
    out = []
    for frac in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        pos = int(round(frac * (len(sorted_values) - 1)))
        out.append(float(sorted_values[pos]))
    return sorted(set(out))


def _split_metrics(rows: list[dict[str, Any]], *, feature: str, target: str, side: str, threshold: float) -> dict[str, Any]:
    selected = []
    for row in rows:
        value = row.get(feature)
        if not _is_num(value):
            continue
        keep = float(value) <= threshold if side == "le" else float(value) >= threshold
        if keep:
            selected.append(row)
    target_count = sum(bool(row.get(target)) for row in selected)
    base_count = sum(bool(row.get(target)) for row in rows)
    by_label: dict[str, dict[str, Any]] = {}
    for label in sorted({str(row.get("label")) for row in rows}):
        all_label = [row for row in rows if str(row.get("label")) == label]
        selected_label = [row for row in selected if str(row.get("label")) == label]
        label_target = sum(bool(row.get(target)) for row in selected_label)
        by_label[label] = {
            "selected_n": len(selected_label),
            "target_count": label_target,
            "target_rate": label_target / max(1, len(selected_label)),
            "base_rate": sum(bool(row.get(target)) for row in all_label) / max(1, len(all_label)),
        }
    selected_rate = target_count / max(1, len(selected))
    base_rate = base_count / max(1, len(rows))
    return {
        "feature": feature,
        "target": target,
        "side": side,
        "threshold": threshold,
        "selected_n": len(selected),
        "target_count": target_count,
        "selected_rate": selected_rate,
        "base_rate": base_rate,
        "lift": selected_rate - base_rate,
        "by_label": by_label,
        "min_label_selected_n": min((item["selected_n"] for item in by_label.values()), default=0),
        "min_label_target_rate": min((item["target_rate"] for item in by_label.values()), default=0.0),
    }


def _numeric_features(rows: list[dict[str, Any]]) -> list[str]:
    features = []
    for key in sorted({key for row in rows for key in row}):
        if not key.startswith("h."):
            continue
        values = [float(row[key]) for row in rows if _is_num(row.get(key))]
        if len(values) < 0.95 * len(rows):
            continue
        if len({round(value, 10) for value in values}) <= 1:
            continue
        features.append(key)
    return features


def build_report(
    *,
    signature_reports: list[str | Path],
    min_selected_rate: float = 0.15,
    min_lift: float = 0.08,
    min_label_selected_n: int = 12,
    min_label_target_rate: float = 0.08,
) -> dict[str, Any]:
    rows = _flatten(signature_reports)
    features = _numeric_features(rows)
    target_reports: dict[str, Any] = {}
    for target in TARGETS:
        candidates: list[dict[str, Any]] = []
        for feature in features:
            values = [float(row[feature]) for row in rows if _is_num(row.get(feature))]
            for threshold in _quantiles(values):
                for side in ("le", "ge"):
                    metrics = _split_metrics(rows, feature=feature, target=target, side=side, threshold=threshold)
                    if metrics["selected_n"] >= min_label_selected_n:
                        candidates.append(metrics)
        candidates.sort(
            key=lambda item: (
                item["selected_rate"],
                item["lift"],
                item["min_label_target_rate"],
                item["target_count"],
            ),
            reverse=True,
        )
        ready = [
            item
            for item in candidates
            if item["selected_rate"] >= min_selected_rate
            and item["lift"] >= min_lift
            and item["min_label_selected_n"] >= min_label_selected_n
            and item["min_label_target_rate"] >= min_label_target_rate
        ]
        base_count = sum(bool(row.get(target)) for row in rows)
        target_reports[target] = {
            "base_count": base_count,
            "base_rate": base_count / max(1, len(rows)),
            "simple_h_split_ready": bool(ready),
            "ready_splits": ready[:5],
            "top_splits": candidates[:10],
        }
    strict_ready = bool(target_reports["strict_current"]["simple_h_split_ready"])
    return {
        "analysis_entry": "phase2_pendulum_settle_h_feature_relation_audit",
        "contract": {
            "read_only_existing_signature_reports": True,
            "does_not_sample_environments": True,
            "does_not_train": True,
            "does_not_modify_exact_scm": True,
            "does_not_modify_fit_model": True,
            "does_not_modify_ppo": True,
        },
        "thresholds": {
            "min_selected_rate": float(min_selected_rate),
            "min_lift": float(min_lift),
            "min_label_selected_n": int(min_label_selected_n),
            "min_label_target_rate": float(min_label_target_rate),
        },
        "inputs": {"signature_reports": [str(Path(path).expanduser().resolve()) for path in signature_reports]},
        "n_rows": len(rows),
        "n_features": len(features),
        "targets": target_reports,
        "decision": {
            "strict_current_simple_h_knob_ready": strict_ready,
            "read": (
                "No stable existing h scalar split is strong enough to explain strict clean-settle yield. "
                "This argues against repairing Pendulum settle by simply biasing an already-exposed scalar knob."
                if not strict_ready
                else "At least one existing h scalar split strongly enriches strict clean-settle candidates."
            ),
            "next_action": (
                "Treat clean settle yield as a mechanism/label problem: either add a cheap suffix-state settle label "
                "or modify the generator to produce more feedback-decay suffix-supported cases, then rerun this audit."
                if not strict_ready
                else "Audit diversity and fixed-list health under the proposed h split before using it as a generator control."
            ),
        },
    }


def _split_text(item: dict[str, Any]) -> str:
    return (
        f"`{item['feature']}` {item['side']} {item['threshold']:.6g}: "
        f"rate={item['selected_rate']:.3f}, lift={item['lift']:.3f}, "
        f"n={item['selected_n']}, min_label_rate={item['min_label_target_rate']:.3f}"
    )


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Pendulum Settle h-Feature Relation Audit",
        "",
        "Read-only audit of whether existing generator parameters explain clean settle yield.",
        "",
        "## Verdict",
        "",
        f"- strict current simple h-knob ready: `{report['decision']['strict_current_simple_h_knob_ready']}`",
        f"- rows/features: `{report['n_rows']}` / `{report['n_features']}`",
        f"- read: {report['decision']['read']}",
        f"- next action: {report['decision']['next_action']}",
        "",
        "## Targets",
        "",
        "| target | base count | base rate | simple split ready | top split |",
        "|---|---:|---:|---:|---|",
    ]
    for target, payload in report["targets"].items():
        top = payload["top_splits"][0] if payload["top_splits"] else None
        lines.append(
            f"| `{target}` | {payload['base_count']} | {payload['base_rate']:.3f} | "
            f"`{payload['simple_h_split_ready']}` | {_split_text(top) if top else 'none'} |"
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
        min_selected_rate=args.min_selected_rate,
        min_lift=args.min_lift,
        min_label_selected_n=args.min_label_selected_n,
        min_label_target_rate=args.min_label_target_rate,
    )
    json_path = out_dir / "pendulum_settle_h_feature_relation_audit.json"
    md_path = out_dir / "pendulum_settle_h_feature_relation_audit.md"
    report["outputs"] = {"json": str(json_path), "markdown": str(md_path)}
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report["outputs"], sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signature-report", action="append", default=[str(path) for path in DEFAULT_SIGNATURE_REPORTS])
    parser.add_argument("--min-selected-rate", type=float, default=0.15)
    parser.add_argument("--min-lift", type=float, default=0.08)
    parser.add_argument("--min-label-selected-n", type=int, default=12)
    parser.add_argument("--min-label-target-rate", type=float, default=0.08)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
