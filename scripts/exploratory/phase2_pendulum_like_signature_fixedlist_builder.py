#!/usr/bin/env python
"""Build fixed-list CSVs from Pendulum-like signature reports.

Exploratory only. It materializes already sampled frozen-h entries selected by
environment-side Pendulum-like labels into a trusted pack-runner CSV. It does
not sample new environments and does not modify exact SCM, fit_model, PPO, or
milestone defaults.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from scripts.exploratory.phase2_transition_causal_context_gate import _json_safe  # noqa: E402


DEFAULT_SIGNATURE_REPORTS = [
    "/home/chen/RLPFN/artifacts/phase2_prior_pendulum_like_signature_probe_0509/gated_full_q90_n64_s128_obs4_scales3_timeprofile.json",
    "/home/chen/RLPFN/artifacts/phase2_prior_pendulum_like_signature_probe_seed6261_0509/gated_full_q90_seed6261_n64_s128_obs4_scales3_timeprofile.json",
]
DEFAULT_OUTPUT_DIR = "/home/chen/RLPFN/artifacts/phase2_pendulum_like_strict_cross_seed_fixedlists_0509"


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def _safe_label(text: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")
    return value[:180] or "list"


def _source_tag(label: str, source_hint: str = "") -> str:
    text = f"{label} {source_hint}"
    seed_match = re.search(r"seed\d+", text)
    seed = seed_match.group(0) if seed_match is not None else "seed6260"
    scope = "q90" if "q90" in text else "full"
    return f"{seed}_{scope}"


def _seed_list(h_path: Path, n: int) -> list[int]:
    seed_path = h_path.parent / "prior_fixed_env_seeds.json"
    if seed_path.exists():
        return [int(v) for v in _load_json(seed_path)]
    return list(range(int(n)))


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _is_time_profile_core_no_random(row: dict[str, Any]) -> bool:
    return _truthy(row.get("time_profile_core")) and row.get("best_time_profile_mode") != "early_random_then_zero"


def _label_match(row: dict[str, Any], label_field: str) -> bool:
    if label_field == "strict_or_time_profile_core_no_random":
        return _truthy(row.get("pendulum_like_strict")) or _is_time_profile_core_no_random(row)
    if label_field == "strict_or_time_profile_core_no_random_or_early_strong":
        return (
            _truthy(row.get("pendulum_like_strict"))
            or _is_time_profile_core_no_random(row)
            or _truthy(row.get("early_only_strong_carryover_core"))
        )
    return _truthy(row.get(label_field))


def _effective_feedback_source(row: dict[str, Any], label_field: str) -> str:
    if label_field in {
        "strict_or_time_profile_core_no_random",
        "strict_or_time_profile_core_no_random_or_early_strong",
    } and _is_time_profile_core_no_random(row):
        return "time_profile_core_no_random"
    if (
        label_field == "strict_or_time_profile_core_no_random_or_early_strong"
        and _truthy(row.get("early_only_strong_carryover_core"))
    ):
        return "early_only_strong_carryover_core"
    return "best_feedback"


def _effective_feedback_mode(row: dict[str, Any], label_field: str) -> Any:
    source = _effective_feedback_source(row, label_field)
    if source == "time_profile_core_no_random":
        return row.get("best_time_profile_mode")
    if source == "early_only_strong_carryover_core":
        return row.get("best_early_only_mode")
    return row.get("best_feedback_mode")


def _effective_minus_open(row: dict[str, Any], label_field: str) -> float:
    source = _effective_feedback_source(row, label_field)
    if source == "time_profile_core_no_random":
        return float(row.get("best_time_profile_env_return", 0.0) or 0.0) - float(
            row.get("best_open_loop_env_return", 0.0) or 0.0
        )
    if source == "early_only_strong_carryover_core":
        return float(row.get("best_early_only_env_return", 0.0) or 0.0) - float(
            row.get("best_open_loop_env_return", 0.0) or 0.0
        )
    return float(row.get("best_feedback_minus_open_env", 0.0) or 0.0)


def _effective_action_abs_prefix(row: dict[str, Any], label_field: str) -> Any:
    source = _effective_feedback_source(row, label_field)
    if source == "time_profile_core_no_random":
        return row.get("best_time_profile_action_abs_prefix")
    if source == "early_only_strong_carryover_core":
        return row.get("best_early_only_action_abs_prefix")
    return row.get("best_feedback_action_abs_prefix")


def _effective_action_abs_suffix(row: dict[str, Any], label_field: str) -> Any:
    source = _effective_feedback_source(row, label_field)
    if source == "time_profile_core_no_random":
        return row.get("best_time_profile_action_abs_suffix")
    if source == "early_only_strong_carryover_core":
        return row.get("best_early_only_action_abs_suffix")
    return row.get("best_feedback_action_abs_suffix")


def _score(row: dict[str, Any], label_field: str) -> float:
    values = [
        _effective_minus_open(row, label_field),
        float(row.get("feedback_pair_phase_gap_env", 0.0) or 0.0),
        float(row.get("feedback_suffix_gain_over_zero_env", 0.0) or 0.0),
    ]
    return float(np.sum(values))


def _iter_candidates(signature_reports: list[str], label_field: str, scope: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for report_path in signature_reports:
        report = _load_json(report_path)
        for item in report.get("lists", []):
            label = str(item["label"])
            if scope != "all" and scope not in label:
                continue
            h_path = Path(item["fixed_h_list_json"]).expanduser().resolve()
            h_list = list(_load_json(h_path))
            seeds = _seed_list(h_path, len(h_list))
            source_tag = _source_tag(label, str(h_path))
            per_env = list((item.get("classification") or {}).get("per_env") or [])
            for row in per_env:
                env_idx = int(row.get("env_idx", row.get("env_index")))
                if not _label_match(row, label_field):
                    continue
                rows.append(
                    {
                        "source_label": label,
                        "source_tag": source_tag,
                        "env_index": env_idx,
                        "seed": int(seeds[env_idx]) if env_idx < len(seeds) else env_idx,
                        "h": h_list[env_idx],
                        "signature": row,
                        "score": _score(row, label_field),
                    }
                )
    return rows


def _balanced_select(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    by_source: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_source.setdefault(str(row["source_tag"]), []).append(row)
    for bucket in by_source.values():
        bucket.sort(key=lambda item: float(item["score"]), reverse=True)
    selected: list[dict[str, Any]] = []
    source_names = sorted(by_source)
    cursor = {name: 0 for name in source_names}
    while len(selected) < int(limit):
        progressed = False
        for name in source_names:
            bucket = by_source[name]
            idx = cursor[name]
            if idx >= len(bucket):
                continue
            selected.append(bucket[idx])
            cursor[name] = idx + 1
            progressed = True
            if len(selected) >= int(limit):
                break
        if not progressed:
            break
    return selected


def _stats(values: list[float]) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "min": float(np.min(arr)),
        "q50": float(np.quantile(arr, 0.5)),
        "max": float(np.max(arr)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signature-report", action="append", default=None)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--label-field", default="pendulum_like_strict")
    parser.add_argument("--scope", choices=["q90", "full", "all"], default="all")
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--rule-name", default=None)
    args = parser.parse_args()

    reports = list(args.signature_report or DEFAULT_SIGNATURE_REPORTS)
    all_candidates = _iter_candidates(reports, str(args.label_field), str(args.scope))
    selected = _balanced_select(all_candidates, int(args.limit))
    if len(selected) < int(args.limit):
        raise RuntimeError(
            f"Only selected {len(selected)} candidates for label={args.label_field!r}, "
            f"scope={args.scope!r}; requested {args.limit}."
        )

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rule = str(args.rule_name or f"{args.scope}_{args.label_field}_n{len(selected)}")
    group_dir = out_dir / _safe_label(rule)
    group_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for out_idx, item in enumerate(selected):
        sig = dict(item["signature"])
        h_path = group_dir / f"env_{out_idx:03d}_{item['source_tag']}_source_{int(item['env_index']):03d}.json"
        h_path.write_text(json.dumps(_json_safe(item["h"]), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        rows.append(
            {
                "rule": rule,
                "source_label": item["source_label"],
                "source_tag": item["source_tag"],
                "env_index": int(item["env_index"]),
                "seed": int(item["seed"]),
                "full_frozen_h_json": str(h_path),
                "signature_score": float(item["score"]),
                "feedback_source_pool_definition": str(args.label_field),
                "effective_feedback_source": _effective_feedback_source(sig, str(args.label_field)),
                "effective_feedback_mode": _effective_feedback_mode(sig, str(args.label_field)),
                "effective_feedback_minus_open_env": _effective_minus_open(sig, str(args.label_field)),
                "pendulum_like_weak": bool(sig.get("pendulum_like_weak")),
                "pendulum_like_core": bool(sig.get("pendulum_like_core")),
                "pendulum_like_strict": bool(sig.get("pendulum_like_strict")),
                "pendulum_like_settle_strict": bool(sig.get("pendulum_like_settle_strict")),
                "nonzero_required_env": bool(sig.get("nonzero_required_env")),
                "feedback_beats_open_loop_env": bool(sig.get("feedback_beats_open_loop_env")),
                "phase_sensitive_feedback": bool(sig.get("phase_sensitive_feedback")),
                "suffix_supported": bool(sig.get("suffix_supported")),
                "best_feedback_mode": sig.get("best_feedback_mode"),
                "best_time_profile_mode": sig.get("best_time_profile_mode"),
                "best_early_only_mode": sig.get("best_early_only_mode"),
                "feedback_pair_best_base_mode": sig.get("feedback_pair_best_base_mode"),
                "feedback_pair_best_sign": sig.get("feedback_pair_best_sign"),
                "best_feedback_minus_open_env": sig.get("best_feedback_minus_open_env"),
                "best_time_profile_minus_open_env": float(sig.get("best_time_profile_env_return", 0.0) or 0.0)
                - float(sig.get("best_open_loop_env_return", 0.0) or 0.0),
                "feedback_pair_phase_gap_env": sig.get("feedback_pair_phase_gap_env"),
                "feedback_suffix_gain_over_zero_env": sig.get("feedback_suffix_gain_over_zero_env"),
                "best_feedback_action_abs_prefix": _effective_action_abs_prefix(sig, str(args.label_field)),
                "best_feedback_action_abs_suffix": _effective_action_abs_suffix(sig, str(args.label_field)),
            }
        )

    csv_path = out_dir / "pendulum_like_signature_fixedlist.csv"
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    source_counts = {name: 0 for name in sorted({str(row["source_tag"]) for row in rows})}
    for row in rows:
        source_counts[str(row["source_tag"])] += 1
    report = {
        "analysis_entry": "phase2_pendulum_like_signature_fixedlist_builder",
        "contract": {
            "exploratory_only": True,
            "does_not_modify_exact_scm": True,
            "does_not_modify_fit_model": True,
            "does_not_modify_ppo": True,
            "does_not_sample_new_envs": True,
            "materializes_only_existing_fixed_h_entries": True,
        },
        "label_field": str(args.label_field),
        "scope": str(args.scope),
        "limit": int(args.limit),
        "candidate_count": int(len(all_candidates)),
        "selected_count": int(len(selected)),
        "source_counts": source_counts,
        "csv": str(csv_path),
        "rule": rule,
        "stats": {
            "signature_score": _stats([float(row["signature_score"]) for row in rows]),
            "best_feedback_minus_open_env": _stats([float(row["best_feedback_minus_open_env"]) for row in rows]),
            "feedback_pair_phase_gap_env": _stats([float(row["feedback_pair_phase_gap_env"]) for row in rows]),
            "feedback_suffix_gain_over_zero_env": _stats([float(row["feedback_suffix_gain_over_zero_env"]) for row in rows]),
        },
    }
    report_path = out_dir / "pendulum_like_signature_fixedlist_report.json"
    report_path.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Pendulum-Like Signature Fixed List",
        "",
        f"csv: {csv_path}",
        f"rule: {rule}",
        f"label_field: {args.label_field}",
        f"scope: {args.scope}",
        f"candidate_count: {len(all_candidates)}",
        f"selected_count: {len(selected)}",
        "",
        "## Source Counts",
        "",
    ]
    for name, count in source_counts.items():
        lines.append(f"- {name}: {count}")
    (out_dir / "pendulum_like_signature_fixedlist_summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"csv": str(csv_path), "rule": rule, "selected_count": len(selected)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
