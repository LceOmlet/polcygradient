#!/usr/bin/env python
"""Build fixed-list CSVs from official-trainable feedback audit reports.

Exploratory only. This script materializes already sampled frozen-h entries
from audit reports into CSV rows consumable by the trusted fixed_frozen_list
pack runner. It does not sample new environments and does not modify exact SCM,
fit_model, PPO, or milestone defaults.
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


DEFAULT_AUDIT_REPORTS = [
    "/home/chen/RLPFN/artifacts/phase2_official_trainable_feedback_audit_0509/official_trainable_feedback_audit_report.json",
    "/home/chen/RLPFN/artifacts/phase2_official_trainable_feedback_audit_seed6261_0509/official_trainable_feedback_audit_report.json",
]
DEFAULT_TRANSITION_REPORTS = [
    "/home/chen/RLPFN/artifacts/phase2_signed_transition_causal_path_probe_0509/signed_transition_causal_path_probe_report.json",
    "/home/chen/RLPFN/artifacts/phase2_signed_transition_causal_path_probe_seed6261_0509/signed_transition_causal_path_probe_report.json",
]
DEFAULT_OUTPUT_DIR = "/home/chen/RLPFN/artifacts/phase2_official_trainable_feedback_cross_seed_fixedlists_0509"


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def _safe_label(text: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")
    return value[:180] or "list"


def _seed_list_for_h_path(h_path: Path) -> list[int]:
    seed_path = h_path.parent / "prior_fixed_env_seeds.json"
    if seed_path.exists():
        return [int(v) for v in _load_json(seed_path)]
    h_list = _load_json(h_path)
    return list(range(len(h_list)))


def _transition_index(paths: list[str]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for path in paths:
        report = _load_json(path)
        for item in report.get("lists", []):
            label = str(item["label"])
            h_path = Path(item["fixed_h_list_json"]).expanduser().resolve()
            index[label] = {
                "fixed_h_list_json": str(h_path),
                "seeds": _seed_list_for_h_path(h_path),
                "h_list": _load_json(h_path),
            }
    return index


def _selected_indices(per_env: list[dict[str, Any]], rule_kind: str) -> list[int]:
    if rule_kind == "official_trainable_tp":
        return [
            int(row["env_index"])
            for row in per_env
            if bool(row.get("transition_and_primary"))
            and bool(row.get("official_trainable_feedback"))
        ]
    if rule_kind == "centered_supported_train_lost_tp":
        return [
            int(row["env_index"])
            for row in per_env
            if bool(row.get("transition_and_primary"))
            and bool(row.get("centered_supported_but_train_lost"))
        ]
    raise ValueError(f"Unknown rule kind: {rule_kind!r}")


def _write_rule_rows(
    *,
    out_dir: Path,
    rule: str,
    audit_label: str,
    source_tag: str,
    indices: list[int],
    per_env_by_idx: dict[int, dict[str, Any]],
    h_list: list[dict[str, Any]],
    seeds: list[int],
) -> list[dict[str, Any]]:
    group_dir = out_dir / _safe_label(rule)
    group_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    existing = len(list(group_dir.glob("env_*.json")))
    for local_i, env_idx in enumerate(indices):
        row_idx = existing + local_i
        h = h_list[int(env_idx)]
        seed = int(seeds[int(env_idx)]) if int(env_idx) < len(seeds) else int(env_idx)
        h_path = group_dir / f"env_{row_idx:03d}_{source_tag}_source_{int(env_idx):03d}.json"
        h_path.write_text(json.dumps(_json_safe(h), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        metrics = per_env_by_idx[int(env_idx)]
        rows.append(
            {
                "rule": rule,
                "source_label": audit_label,
                "source_tag": source_tag,
                "env_index": int(env_idx),
                "seed": seed,
                "full_frozen_h_json": str(h_path),
                "official_actor_advantage_train_normalized_alignment": metrics.get(
                    "official_actor_advantage_train_normalized_alignment"
                ),
                "official_value_gae_lambda_0.9_centered_alignment": metrics.get(
                    "official_value_gae_lambda_0.9_centered_alignment"
                ),
                "zero_value_gae_lambda_0.9_centered_alignment": metrics.get(
                    "zero_value_gae_lambda_0.9_centered_alignment"
                ),
                "best_late_state_pair_gap": metrics.get("best_late_state_pair_gap"),
                "best_suffix_reward_pair_gap": metrics.get("best_suffix_reward_pair_gap"),
                "best_action_pair_gap": metrics.get("best_action_pair_gap"),
                "best_done_pair_gap": metrics.get("best_done_pair_gap"),
                "best_pair_name": metrics.get("best_pair_name"),
                "best_pair_sign": metrics.get("best_pair_sign"),
                "transition_and_primary": bool(metrics.get("transition_and_primary")),
                "official_trainable_feedback": bool(metrics.get("official_trainable_feedback")),
                "centered_supported_but_train_lost": bool(metrics.get("centered_supported_but_train_lost")),
            }
        )
    return rows


def _source_tag(label: str) -> str:
    if "seed6261" in label:
        seed = "seed6261"
    else:
        seed = "seed6260"
    scope = "q90" if "q90" in label else "full"
    return f"{seed}_{scope}"


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
    parser.add_argument("--audit-report", action="append", default=None)
    parser.add_argument("--transition-report", action="append", default=None)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--scope", choices=["q90", "full", "all"], default="q90")
    args = parser.parse_args()

    audit_reports = list(args.audit_report or DEFAULT_AUDIT_REPORTS)
    transition_reports = list(args.transition_report or DEFAULT_TRANSITION_REPORTS)
    transition_by_label = _transition_index(transition_reports)
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    list_reports = []
    for audit_path in audit_reports:
        audit = _load_json(audit_path)
        for item in audit.get("lists", []):
            label = str(item["label"])
            if args.scope != "all" and args.scope not in label:
                continue
            if label not in transition_by_label:
                raise KeyError(f"Missing transition source for audit list {label!r}")
            source = transition_by_label[label]
            per_env = list(item["per_env"])
            per_env_by_idx = {int(row["env_index"]): row for row in per_env}
            source_tag = _source_tag(label)
            h_list = list(source["h_list"])
            seeds = list(source["seeds"])
            rule_counts: dict[str, int] = {}
            for rule_kind in ("official_trainable_tp", "centered_supported_train_lost_tp"):
                indices = _selected_indices(per_env, rule_kind)
                rule = f"cross_seed_{args.scope}_{rule_kind}" if args.scope != "all" else f"cross_seed_{rule_kind}"
                rule_rows = _write_rule_rows(
                    out_dir=out_dir,
                    rule=rule,
                    audit_label=label,
                    source_tag=source_tag,
                    indices=indices,
                    per_env_by_idx=per_env_by_idx,
                    h_list=h_list,
                    seeds=seeds,
                )
                rows.extend(rule_rows)
                rule_counts[rule] = len(rule_rows)
            list_reports.append(
                {
                    "audit_report": str(Path(audit_path).expanduser().resolve()),
                    "label": label,
                    "source_tag": source_tag,
                    "fixed_h_list_json": source["fixed_h_list_json"],
                    "rule_counts": rule_counts,
                    "counts": item.get("counts"),
                    "rates": item.get("rates"),
                }
            )

    csv_path = out_dir / "official_trainable_feedback_cross_seed_fixedlist.csv"
    fieldnames = [
        "rule",
        "source_label",
        "source_tag",
        "env_index",
        "seed",
        "full_frozen_h_json",
        "official_actor_advantage_train_normalized_alignment",
        "official_value_gae_lambda_0.9_centered_alignment",
        "zero_value_gae_lambda_0.9_centered_alignment",
        "best_late_state_pair_gap",
        "best_suffix_reward_pair_gap",
        "best_action_pair_gap",
        "best_done_pair_gap",
        "best_pair_name",
        "best_pair_sign",
        "transition_and_primary",
        "official_trainable_feedback",
        "centered_supported_but_train_lost",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    rule_summary = {}
    for rule in sorted({str(row["rule"]) for row in rows}):
        rule_rows = [row for row in rows if row["rule"] == rule]
        rule_summary[rule] = {
            "n": len(rule_rows),
            "sources": sorted({str(row["source_tag"]) for row in rule_rows}),
            "official_actor_alignment": _stats(
                [float(row["official_actor_advantage_train_normalized_alignment"]) for row in rule_rows]
            ),
            "suffix_reward_pair_gap": _stats([float(row["best_suffix_reward_pair_gap"]) for row in rule_rows]),
        }
    report = {
        "analysis_entry": "phase2_official_trainable_feedback_fixedlist_builder",
        "contract": {
            "exploratory_only": True,
            "does_not_modify_exact_scm": True,
            "does_not_modify_fit_model": True,
            "does_not_modify_ppo": True,
            "does_not_sample_new_envs": True,
            "materializes_only_existing_fixed_h_entries": True,
        },
        "scope": args.scope,
        "audit_reports": [str(Path(p).expanduser().resolve()) for p in audit_reports],
        "transition_reports": [str(Path(p).expanduser().resolve()) for p in transition_reports],
        "csv": str(csv_path),
        "lists": list_reports,
        "rules": rule_summary,
    }
    report_path = out_dir / "official_trainable_feedback_cross_seed_fixedlist_report.json"
    report_path.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Official-Trainable Feedback Cross-Seed Fixed Lists",
        "",
        f"csv: {csv_path}",
        "",
        "| rule | n | sources | actor alignment mean | suffix gap mean |",
        "|---|---:|---|---:|---:|",
    ]
    for rule, item in rule_summary.items():
        lines.append(
            f"| {rule} | {item['n']} | {','.join(item['sources'])} | "
            f"{item['official_actor_alignment'].get('mean')} | {item['suffix_reward_pair_gap'].get('mean')} |"
        )
    (out_dir / "official_trainable_feedback_cross_seed_fixedlist_summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"csv": str(csv_path), "rules": rule_summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
