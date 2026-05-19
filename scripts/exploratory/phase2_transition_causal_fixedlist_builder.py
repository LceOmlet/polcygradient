#!/usr/bin/env python
"""Build fixed-list CSVs for transition-causal candidate smoke tests.

Exploratory/read-only with respect to the generator.  It does not mutate exact
SCM, fit_model, PPO, or milestone defaults.  It only materializes selected
frozen-h entries from an already audited fixed-list into CSV rows consumable by
the trusted pack runner's fixed_frozen_list path.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

from phase2_transition_causal_context_gate import _candidate_labels, _json_safe  # type: ignore


DEFAULT_TRANSITION_REPORT = (
    "/home/chen/RLPFN/artifacts/phase2_signed_transition_causal_path_probe_0509/"
    "signed_transition_causal_path_probe_report.json"
)
DEFAULT_OUTPUT_DIR = "/home/chen/RLPFN/artifacts/phase2_transition_causal_fixedlists_0509"


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def _safe_label(text: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")
    return value[:160] or "list"


def _stats(values: list[float]) -> dict[str, Any]:
    arr = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=np.float64)
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "q50": float(np.quantile(arr, 0.50)),
        "max": float(np.max(arr)),
    }


def _write_group(
    *,
    out_dir: Path,
    source_label: str,
    rule: str,
    indices: list[int],
    h_list: list[dict[str, Any]],
    seeds: list[int],
    per_env: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    group_dir = out_dir / _safe_label(source_label) / _safe_label(rule)
    group_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    selected_h = []
    selected_seeds = []
    for row_idx, env_idx in enumerate(indices):
        h = h_list[int(env_idx)]
        seed = int(seeds[int(env_idx)])
        h_path = group_dir / f"env_{row_idx:03d}_source_{int(env_idx):03d}.json"
        h_path.write_text(json.dumps(_json_safe(h), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        selected_h.append(h)
        selected_seeds.append(seed)
        metrics = per_env[int(env_idx)]
        rows.append(
            {
                "rule": rule,
                "source_label": source_label,
                "env_index": int(env_idx),
                "seed": seed,
                "full_frozen_h_json": str(h_path),
                "primary_long_horizon_target": bool(metrics.get("primary_long_horizon_target", False)),
                "selector_hard_proxy": bool(metrics.get("selector_hard_proxy", False)),
                "best_late_state_pair_gap": metrics.get("best_late_state_pair_gap"),
                "best_suffix_reward_pair_gap": metrics.get("best_suffix_reward_pair_gap"),
                "best_action_pair_gap": metrics.get("best_action_pair_gap"),
                "best_done_pair_gap": metrics.get("best_done_pair_gap"),
                "best_pair_name": metrics.get("best_pair_name"),
                "best_pair_sign": metrics.get("best_pair_sign"),
            }
        )
    (group_dir / "prior_fixed_h_list.json").write_text(
        json.dumps(_json_safe(selected_h), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (group_dir / "prior_fixed_env_seeds.json").write_text(
        json.dumps([int(v) for v in selected_seeds], indent=2) + "\n",
        encoding="utf-8",
    )
    return rows


def _build_for_list(item: dict[str, Any], out_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    label = str(item["label"])
    h_path = Path(item["fixed_h_list_json"]).expanduser().resolve()
    seed_path = h_path.parent / "prior_fixed_env_seeds.json"
    h_list = _load_json(h_path)
    seeds = _load_json(seed_path) if seed_path.exists() else list(range(len(h_list)))
    per_env = list(item["per_env"])
    if not (len(h_list) == len(seeds) == len(per_env)):
        raise RuntimeError(f"{label}: h/seeds/per_env length mismatch")
    labels = _candidate_labels(per_env)
    candidate = np.asarray(labels["transition_causal_candidate"], dtype=bool)
    late = np.asarray([float(row["best_late_state_pair_gap"]) for row in per_env], dtype=np.float64)
    suffix = np.asarray([float(row["best_suffix_reward_pair_gap"]) for row in per_env], dtype=np.float64)
    score = late + suffix / max(float(np.nanstd(suffix)), 1.0)
    candidate_indices = np.flatnonzero(candidate).tolist()
    candidate_indices = sorted(candidate_indices, key=lambda idx: float(score[idx]), reverse=True)
    if int(args.limit) > 0:
        candidate_indices = candidate_indices[: int(args.limit)]
    low_mask = (late <= np.quantile(late, 1.0 / 3.0)) & (suffix <= np.quantile(suffix, 1.0 / 3.0)) & (~candidate)
    control_indices = np.flatnonzero(low_mask).tolist()
    control_indices = sorted(control_indices, key=lambda idx: float(score[idx]))
    control_indices = control_indices[: len(candidate_indices)]
    rules = {
        "transition_causal_candidate": candidate_indices,
        "transition_low_control": control_indices,
    }
    rows = []
    for rule, indices in rules.items():
        rows.extend(
            _write_group(
                out_dir=out_dir,
                source_label=label,
                rule=f"{_safe_label(label)}__{rule}",
                indices=indices,
                h_list=h_list,
                seeds=[int(v) for v in seeds],
                per_env=per_env,
            )
        )
    return {
        "label": label,
        "fixed_h_list_json": str(h_path),
        "fixed_env_seed_json": str(seed_path),
        "candidate_thresholds": labels["thresholds"],
        "counts": {name: int(len(indices)) for name, indices in rules.items()},
        "candidate_indices": [int(v) for v in candidate_indices],
        "control_indices": [int(v) for v in control_indices],
        "candidate_primary_rate": float(np.mean([bool(per_env[idx].get("primary_long_horizon_target", False)) for idx in candidate_indices]))
        if candidate_indices
        else None,
        "control_primary_rate": float(np.mean([bool(per_env[idx].get("primary_long_horizon_target", False)) for idx in control_indices]))
        if control_indices
        else None,
        "candidate_late_state_gap": _stats([float(per_env[idx]["best_late_state_pair_gap"]) for idx in candidate_indices]),
        "control_late_state_gap": _stats([float(per_env[idx]["best_late_state_pair_gap"]) for idx in control_indices]),
        "candidate_suffix_reward_gap": _stats([float(per_env[idx]["best_suffix_reward_pair_gap"]) for idx in candidate_indices]),
        "control_suffix_reward_gap": _stats([float(per_env[idx]["best_suffix_reward_pair_gap"]) for idx in control_indices]),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transition-report", default=DEFAULT_TRANSITION_REPORT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    transition = _load_json(args.transition_report)
    if any("per_env" not in item for item in transition.get("lists", [])):
        raise RuntimeError("Transition report lacks per_env metrics. Rerun phase2_signed_transition_causal_path_probe.py first.")
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    list_reports = [_build_for_list(item, out_dir, args) for item in transition.get("lists", [])]
    all_rows = [row for item in list_reports for row in item["rows"]]
    csv_path = out_dir / "transition_causal_fixedlist.csv"
    fieldnames = [
        "rule",
        "source_label",
        "env_index",
        "seed",
        "full_frozen_h_json",
        "primary_long_horizon_target",
        "selector_hard_proxy",
        "best_late_state_pair_gap",
        "best_suffix_reward_pair_gap",
        "best_action_pair_gap",
        "best_done_pair_gap",
        "best_pair_name",
        "best_pair_sign",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)
    report = {
        "analysis_entry": "phase2_transition_causal_fixedlist_builder",
        "contract": {
            "exploratory_only": True,
            "does_not_modify_exact_scm": True,
            "does_not_modify_fit_model": True,
            "does_not_modify_ppo": True,
            "materializes_only_existing_fixed_h_entries": True,
        },
        "input_transition_report": str(Path(args.transition_report).expanduser().resolve()),
        "csv": str(csv_path),
        "lists": [{k: v for k, v in item.items() if k != "rows"} for item in list_reports],
        "rules": sorted({str(row["rule"]) for row in all_rows}),
    }
    report_path = out_dir / "transition_causal_fixedlist_builder_report.json"
    report_path.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Transition-Causal Fixed Lists",
        "",
        f"csv: {csv_path}",
        "",
        "| source | candidate n | control n | candidate primary | control primary |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for item in list_reports:
        lines.append(
            f"| {item['label']} | {item['counts']['transition_causal_candidate']} | "
            f"{item['counts']['transition_low_control']} | {item['candidate_primary_rate']} | {item['control_primary_rate']} |"
        )
    (out_dir / "transition_causal_fixedlist_builder_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"csv": str(csv_path), "report": str(report_path), "rules": report["rules"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
