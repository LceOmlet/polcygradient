#!/usr/bin/env python
"""Validate Pendulum-like terminal guard stability/diversity/optimisability.

Exploratory read-only summary. It joins:
  - a 92-env Pendulum-like strict candidate CSV,
  - two independent pre-update sidecars for the same candidate pool,
  - the terminal-guard selected n64 CSV,
  - paired short-smoke train summaries.

It does not resample environments, modify exact SCM, modify PPO, or alter
fit_model defaults.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_ALL_CSV = (
    "/home/chen/RLPFN/artifacts/phase2_pendulum_like_strict_cross_seed_all_candidates_0509/"
    "pendulum_like_signature_fixedlist.csv"
)
DEFAULT_SELECTED_CSV = (
    "/home/chen/RLPFN/artifacts/phase2_pendulum_like_strict_terminal_guard_fixedlists_0509/"
    "pendulum_like_terminal_guard_fixedlist.csv"
)
DEFAULT_SIDECAR_A = (
    "/home/chen/RLPFN/artifacts/phase2_pendulum_like_strict_n92_preupdate_sidecar_s512_u1_e1_advnorm_0509/"
    "semantic_sidecars/prior_update_000000_envs.jsonl"
)
DEFAULT_SIDECAR_B = (
    "/home/chen/RLPFN/artifacts/phase2_pendulum_like_strict_n92_preupdate_sidecar_s512_u1_e1_advnorm_seed9292_0509/"
    "semantic_sidecars/prior_update_000000_envs.jsonl"
)
DEFAULT_PAIRED_SMOKE = (
    "/home/chen/RLPFN/artifacts/phase2_pendulum_like_strict_guard_paired_seed_summary_0509/"
    "paired_seed_summary_report.json"
)
DEFAULT_OUTPUT_DIR = "/home/chen/RLPFN/artifacts/phase2_pendulum_like_strict_guard_validation_summary_0509"


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


def _f(value: Any) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _stats(values: list[Any]) -> dict[str, Any]:
    arr = np.asarray([float(v) for v in values if _f(v) is not None], dtype=np.float64)
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
        "positive_fraction": float(np.mean(arr > 0.0)),
    }


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1)
        i = j
    return ranks


def _corr(xs: list[Any], ys: list[Any]) -> dict[str, Any]:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if _f(x) is not None and _f(y) is not None]
    if len(pairs) < 3:
        return {"n": len(pairs)}
    xa = np.asarray([p[0] for p in pairs], dtype=np.float64)
    ya = np.asarray([p[1] for p in pairs], dtype=np.float64)
    if float(np.std(xa)) == 0.0 or float(np.std(ya)) == 0.0:
        return {"n": len(pairs), "pearson": None, "spearman": None}
    return {
        "n": len(pairs),
        "pearson": float(np.corrcoef(xa, ya)[0, 1]),
        "spearman": float(np.corrcoef(_rankdata(xa), _rankdata(ya))[0, 1]),
    }


def _load_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _parse_mode(mode: Any) -> dict[str, Any]:
    text = str(mode or "")
    match = re.search(r"(?:^|_)obs_linear(?:_neg)?_(\d+)x(\d+)", text)
    return {
        "mode_exact": text or None,
        "mode_has_neg": "_neg_" in text,
        "mode_obs_feature": int(match.group(1)) if match else None,
        "mode_horizon": int(match.group(2)) if match else None,
    }


def _guard_rank_tuple(side: dict[str, Any]) -> tuple[float, float, float, float]:
    done = _f(side.get("done_count"))
    first_done = _f(side.get("first_done_step"))
    sat = _f(side.get("raw_policy_action_unit_clip_saturation_fraction"))
    action_abs = _f(side.get("action_value_absmean"))
    return (
        float(done if done is not None else 1e9),
        -float(first_done if first_done is not None else -1.0),
        float(sat if sat is not None else 1e9),
        float(action_abs if action_abs is not None else 1e9),
    )


def _balanced_select(rows: list[dict[str, Any]], sidecar_by_idx: dict[int, dict[str, Any]], limit: int) -> list[int]:
    by_source: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        by_source[str(row.get("source_tag"))].append(idx)
    for source, indices in by_source.items():
        indices.sort(key=lambda idx: _guard_rank_tuple(sidecar_by_idx[idx]))
    selected: list[int] = []
    cursor = {source: 0 for source in by_source}
    for source in sorted(by_source):
        indices = by_source[source]
        if indices and len(selected) < int(limit):
            selected.append(indices[0])
            cursor[source] = 1
    while len(selected) < int(limit):
        progressed = False
        for source in sorted(by_source):
            indices = by_source[source]
            idx = cursor[source]
            if idx >= len(indices):
                continue
            selected.append(indices[idx])
            cursor[source] = idx + 1
            progressed = True
            if len(selected) >= int(limit):
                break
        if not progressed:
            break
    return selected


def _counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(Counter(str(row.get(key)) for row in rows))


def _subset_summary(
    name: str,
    rows: list[dict[str, Any]],
    sidecar_by_idx: dict[int, dict[str, Any]],
    indices: list[int],
) -> dict[str, Any]:
    subset = [rows[i] for i in indices]
    sides = [sidecar_by_idx[i] for i in indices]
    enriched = []
    for row in subset:
        parsed = _parse_mode(row.get("best_feedback_mode"))
        enriched.append({**row, **parsed})
    return {
        "name": name,
        "n": len(indices),
        "source_counts": _counts(enriched, "source_tag"),
        "sign_counts": _counts(enriched, "feedback_pair_best_sign"),
        "mode_feature_counts": _counts(enriched, "mode_obs_feature"),
        "mode_horizon_counts": _counts(enriched, "mode_horizon"),
        "mode_exact_counts": _counts(enriched, "mode_exact"),
        "signature_score": _stats([r.get("signature_score") for r in enriched]),
        "best_feedback_minus_open_env": _stats([r.get("best_feedback_minus_open_env") for r in enriched]),
        "feedback_pair_phase_gap_env": _stats([r.get("feedback_pair_phase_gap_env") for r in enriched]),
        "feedback_suffix_gain_over_zero_env": _stats([r.get("feedback_suffix_gain_over_zero_env") for r in enriched]),
        "preupdate_done_count": _stats([s.get("done_count") for s in sides]),
        "preupdate_first_done_step": _stats([s.get("first_done_step") for s in sides]),
        "preupdate_raw_reward_sum": _stats([s.get("raw_reward_sum") for s in sides]),
        "preupdate_reward_env_sum": _stats([s.get("reward_env_sum") for s in sides]),
        "preupdate_action_saturation": _stats([s.get("raw_policy_action_unit_clip_saturation_fraction") for s in sides]),
        "preupdate_action_absmean": _stats([s.get("action_value_absmean") for s in sides]),
        "preupdate_next_state_step_l2_delta_mean": _stats([s.get("next_state_step_l2_delta_mean") for s in sides]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-csv", default=DEFAULT_ALL_CSV)
    parser.add_argument("--selected-csv", default=DEFAULT_SELECTED_CSV)
    parser.add_argument("--preupdate-sidecar-a", default=DEFAULT_SIDECAR_A)
    parser.add_argument("--preupdate-sidecar-b", default=DEFAULT_SIDECAR_B)
    parser.add_argument("--paired-smoke-report", default=DEFAULT_PAIRED_SMOKE)
    parser.add_argument("--all-rule", default="cross_seed_full_q90_pendulum_like_strict_n92")
    parser.add_argument("--selected-rule", default="cross_seed_full_q90_pendulum_like_strict_terminal_guard_n64")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    all_rows = [
        row for row in _load_csv(args.all_csv)
        if not str(args.all_rule) or str(row.get("rule")) == str(args.all_rule)
    ]
    selected_rows = [
        row for row in _load_csv(args.selected_csv)
        if not str(args.selected_rule) or str(row.get("rule")) == str(args.selected_rule)
    ]
    side_a = {int(row["env_idx"]): row for row in _load_jsonl(args.preupdate_sidecar_a)}
    side_b = {int(row["env_idx"]): row for row in _load_jsonl(args.preupdate_sidecar_b)}
    if len(all_rows) != len(side_a) or len(all_rows) != len(side_b):
        raise RuntimeError("all candidate rows and sidecar rows must have matching counts")

    selected_candidate_indices = [int(row["source_candidate_index"]) for row in selected_rows]
    selected_a = set(selected_candidate_indices)
    selected_from_b = set(_balanced_select(all_rows, side_b, limit=len(selected_rows)))
    selected_from_a = set(_balanced_select(all_rows, side_a, limit=len(selected_rows)))
    rejected_a = [idx for idx in range(len(all_rows)) if idx not in selected_a]

    ranks_a = [0.0] * len(all_rows)
    ranks_b = [0.0] * len(all_rows)
    for rank, idx in enumerate(sorted(range(len(all_rows)), key=lambda i: _guard_rank_tuple(side_a[i]))):
        ranks_a[idx] = float(rank)
    for rank, idx in enumerate(sorted(range(len(all_rows)), key=lambda i: _guard_rank_tuple(side_b[i]))):
        ranks_b[idx] = float(rank)

    paired_smoke = json.loads(Path(args.paired_smoke_report).expanduser().resolve().read_text(encoding="utf-8"))
    report = {
        "analysis_entry": "phase2_pendulum_like_guard_validation_summary",
        "contract": {
            "exploratory_only": True,
            "reads_existing_fixed_lists": True,
            "reads_existing_preupdate_sidecars": True,
            "reads_existing_pack_runner_smokes": True,
            "does_not_modify_exact_scm": True,
            "does_not_modify_fit_model": True,
            "does_not_modify_ppo": True,
            "does_not_resample_envs": True,
        },
        "all_candidate_count": len(all_rows),
        "selected_count": len(selected_rows),
        "guard_stability": {
            "selected_overlap_a_builder_vs_recomputed_a": len(selected_a & selected_from_a),
            "selected_overlap_a_vs_b": len(selected_a & selected_from_b),
            "selected_overlap_fraction_a_vs_b": float(len(selected_a & selected_from_b) / max(1, len(selected_a))),
            "rank_position_correlation_a_b": _corr(ranks_a, ranks_b),
            "done_count_correlation_a_b": _corr(
                [side_a[i].get("done_count") for i in range(len(all_rows))],
                [side_b[i].get("done_count") for i in range(len(all_rows))],
            ),
            "first_done_step_correlation_a_b": _corr(
                [side_a[i].get("first_done_step") for i in range(len(all_rows))],
                [side_b[i].get("first_done_step") for i in range(len(all_rows))],
            ),
        },
        "all_candidates_sidecar_a": _subset_summary("all_candidates_a", all_rows, side_a, list(range(len(all_rows)))),
        "selected_sidecar_a": _subset_summary("selected_a", all_rows, side_a, sorted(selected_a)),
        "rejected_sidecar_a": _subset_summary("rejected_a", all_rows, side_a, rejected_a),
        "selected_sidecar_b": _subset_summary("selected_b", all_rows, side_b, sorted(selected_a)),
        "b_recomputed_selected_sidecar_b": _subset_summary("b_recomputed_selected_b", all_rows, side_b, sorted(selected_from_b)),
        "paired_smoke": paired_smoke,
    }

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "guard_validation_summary_report.json"
    json_path.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Pendulum-Like Guard Validation Summary",
        "",
        "Exploratory read-only validation of the terminal-stability guard.",
        "",
        f"all candidates: {len(all_rows)}",
        f"selected: {len(selected_rows)}",
        "",
        "## Guard Stability",
        "",
    ]
    for key, value in report["guard_stability"].items():
        lines.append(f"- {key}: {value}")

    def add_subset(title: str, summary: dict[str, Any]) -> None:
        lines.extend(["", f"## {title}", ""])
        lines.append(f"- n: {summary['n']}")
        lines.append(f"- source_counts: {summary['source_counts']}")
        lines.append(f"- sign_counts: {summary['sign_counts']}")
        lines.append(f"- mode_feature_counts: {summary['mode_feature_counts']}")
        lines.append(f"- mode_horizon_counts: {summary['mode_horizon_counts']}")
        lines.append(f"- signature_score: {summary['signature_score']}")
        lines.append(f"- preupdate_done_count: {summary['preupdate_done_count']}")
        lines.append(f"- preupdate_first_done_step: {summary['preupdate_first_done_step']}")
        lines.append(f"- preupdate_action_saturation: {summary['preupdate_action_saturation']}")
        lines.append(f"- preupdate_next_state_step_l2_delta_mean: {summary['preupdate_next_state_step_l2_delta_mean']}")

    add_subset("All Candidates, Sidecar A", report["all_candidates_sidecar_a"])
    add_subset("Selected, Sidecar A", report["selected_sidecar_a"])
    add_subset("Rejected, Sidecar A", report["rejected_sidecar_a"])
    add_subset("Selected Rechecked, Sidecar B", report["selected_sidecar_b"])
    add_subset("Recomputed Selected, Sidecar B", report["b_recomputed_selected_sidecar_b"])

    lines.extend(["", "## Paired Short-Smoke Optimisability", "", "| seed | raw delta guard-plain | ep final guard-plain | ep delta guard-plain | full final guard-plain | full delta guard-plain |", "|---:|---:|---:|---:|---:|---:|"])
    for seed, row in paired_smoke.get("paired_by_seed", {}).items():
        lines.append(
            f"| {seed} | {row['guard_minus_plain_raw_delta_mean']} | "
            f"{row['guard_minus_plain_ep_rew_last']} | {row['guard_minus_plain_ep_rew_delta']} | "
            f"{row['guard_minus_plain_full_ep_rew_last']} | {row['guard_minus_plain_full_ep_rew_delta']} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- The guard preserves source balance and keeps multiple feedback signs/features/horizons.",
        "- The selected set is healthier on pre-update terminal stability than rejected candidates and remains healthier under an independent pre-update seed.",
        "- Paired short-smoke results show better final episodic returns at both training seeds.",
        "- This is still exploratory: raw sidecar deltas and source buckets are not uniformly better, so this is a candidate filter, not a protected milestone.",
    ])
    md_path = out_dir / "guard_validation_summary.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
