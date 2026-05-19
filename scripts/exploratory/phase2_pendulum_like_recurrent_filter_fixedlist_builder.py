#!/usr/bin/env python
"""Build exploratory Pendulum-like fixed lists with recurrent-failure filters.

This script is intentionally outside the exact SCM implementation.  It consumes
an already built Pendulum-like candidate list plus pre-update sidecars, applies
simple train-before-observable exclusion rules, and writes a fixed-frozen-h CSV
for the trusted pack runner.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_INPUT_CSV = (
    "/home/chen/RLPFN/artifacts/phase2_pendulum_like_strict_cross_seed_all_candidates_0509/"
    "pendulum_like_signature_fixedlist.csv"
)
DEFAULT_SIDECAR_A = (
    "/home/chen/RLPFN/artifacts/phase2_pendulum_like_strict_n92_preupdate_sidecar_s512_u1_e1_advnorm_0509/"
    "semantic_sidecars/prior_update_000000_envs.jsonl"
)
DEFAULT_SIDECAR_B = (
    "/home/chen/RLPFN/artifacts/phase2_pendulum_like_strict_n92_preupdate_sidecar_s512_u1_e1_advnorm_seed9292_0509/"
    "semantic_sidecars/prior_update_000000_envs.jsonl"
)
DEFAULT_OUTPUT_DIR = "/home/chen/RLPFN/artifacts/phase2_pendulum_like_recurrent_filter_fixedlists_0510"

FILTERS = {
    # Low-risk filter from the recurrent-failure audit:
    # removes high initial-policy action saturation in an independent sidecar.
    "sat_only": {
        "rule": "b_action_saturation <= 0.330781",
        "description": "Reject candidates whose seed9292 pre-update action saturation exceeds 0.330781.",
    },
    # Stronger candidate: rejects the high-saturation/high-amplitude corner.
    "sat_abs": {
        "rule": "b_action_saturation <= 0.330781 AND a_action_absmean <= 0.803117",
        "description": (
            "Reject candidates with high seed9292 action saturation or high seed9192 "
            "absolute action magnitude."
        ),
    },
    # Dynamics-scale candidate from the audit. Kept separate because it changes
    # source balance more than the action filters.
    "obs_reward": {
        "rule": "a_obs_value_std <= 1.50662 AND b_training_reward_sum <= 2.97315",
        "description": "Reject high obs-dynamics-scale or high initial training-reward outliers.",
    },
    "sat_obsu": {
        "rule": "b_action_saturation <= 0.330781 AND h._constrained_obs_u > 0.120099",
        "description": (
            "Keep the low action-saturation guard and reject the weakest "
            "observable-context sampler corner."
        ),
    },
    "sat_initstate": {
        "rule": "b_action_saturation <= 0.330781 AND h.init_state_std <= 0.587529",
        "description": "Keep the low action-saturation guard and reject very high initial-state spread.",
    },
    "sat_alpha_mode_guard": {
        "rule": (
            "b_action_saturation <= 0.330781 AND h.alpha <= 0.28831 AND "
            "best_feedback_mode not in {obs_linear_0x200, obs_linear_neg_0x200}"
        ),
        "description": (
            "Exploratory guarded rule from the expanded n128 post-opt audit: keep the "
            "validated low-saturation guard while avoiding the high-alpha/feature-0 "
            "corner that concentrated short-run regressions."
        ),
    },
}


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


def _join_rows(fixed_rows: list[dict[str, str]], side_a: list[dict[str, Any]], side_b: list[dict[str, Any]]) -> list[dict[str, Any]]:
    joined = []
    for idx, fixed in enumerate(fixed_rows):
        row: dict[str, Any] = dict(fixed)
        row["source_candidate_index"] = int(idx)
        for prefix, side in (("a", side_a[idx]), ("b", side_b[idx])):
            row[f"{prefix}_done_count"] = _f(side.get("done_count"))
            row[f"{prefix}_first_done_step"] = _f(side.get("first_done_step"))
            row[f"{prefix}_raw_reward_sum"] = _f(side.get("raw_reward_sum"))
            row[f"{prefix}_reward_env_sum"] = _f(side.get("reward_env_sum"))
            row[f"{prefix}_training_reward_sum"] = _f(side.get("training_reward_sum"))
            row[f"{prefix}_action_saturation"] = _f(side.get("raw_policy_action_unit_clip_saturation_fraction"))
            row[f"{prefix}_action_absmean"] = _f(side.get("action_value_absmean"))
            row[f"{prefix}_obs_value_std"] = _f(side.get("obs_value_std"))
            row[f"{prefix}_next_state_step_l2_delta_mean"] = _f(side.get("next_state_step_l2_delta_mean"))
        for key in (
            "signature_score",
            "best_feedback_minus_open_env",
            "feedback_pair_phase_gap_env",
            "feedback_suffix_gain_over_zero_env",
            "best_feedback_action_abs_prefix",
            "best_feedback_action_abs_suffix",
        ):
            row[key] = _f(row.get(key))
        joined.append(row)
    return joined


def _passes_filter(row: dict[str, Any], name: str) -> bool:
    if name == "sat_only":
        return float(row["b_action_saturation"]) <= 0.330781
    if name == "sat_abs":
        return float(row["b_action_saturation"]) <= 0.330781 and float(row["a_action_absmean"]) <= 0.803117
    if name == "obs_reward":
        return float(row["a_obs_value_std"]) <= 1.50662 and float(row["b_training_reward_sum"]) <= 2.97315
    if name in {"sat_obsu", "sat_initstate", "sat_alpha_mode_guard"}:
        h = json.loads(Path(str(row["full_frozen_h_json"])).expanduser().resolve().read_text(encoding="utf-8"))
        if name == "sat_obsu":
            return float(row["b_action_saturation"]) <= 0.330781 and float(h.get("_constrained_obs_u", 0.0)) > 0.120099
        if name == "sat_initstate":
            return float(row["b_action_saturation"]) <= 0.330781 and float(h.get("init_state_std", 0.0)) <= 0.587529
        bad_modes = {"obs_linear_0x200", "obs_linear_neg_0x200"}
        return (
            float(row["b_action_saturation"]) <= 0.330781
            and float(h.get("alpha", 0.0)) <= 0.28831
            and str(row.get("best_feedback_mode")) not in bad_modes
        )
    raise KeyError(f"unknown filter {name!r}")


def _rank_key(row: dict[str, Any]) -> tuple[float, float, float]:
    # Keep the Pendulum-like target strong, then avoid the exact high-action
    # corner that the filter is trying to control. This is a selection tie-break,
    # not a health claim.
    signature = _f(row.get("signature_score"))
    saturation = _f(row.get("b_action_saturation"))
    action_abs = _f(row.get("a_action_absmean"))
    return (
        -float(signature if signature is not None else -1e9),
        float(saturation if saturation is not None else 1e9),
        float(action_abs if action_abs is not None else 1e9),
    )


def _balanced_select(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_source[str(row.get("source_tag"))].append(row)
    for bucket in by_source.values():
        bucket.sort(key=_rank_key)

    selected = []
    cursor = {source: 0 for source in by_source}
    while len(selected) < int(limit):
        progressed = False
        for source in sorted(by_source):
            bucket = by_source[source]
            idx = cursor[source]
            if idx >= len(bucket):
                continue
            selected.append(bucket[idx])
            cursor[source] = idx + 1
            progressed = True
            if len(selected) >= int(limit):
                break
        if not progressed:
            break
    return selected


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "source_counts": dict(Counter(str(r.get("source_tag")) for r in rows)),
        "best_sign_counts": dict(Counter(str(r.get("feedback_pair_best_sign")) for r in rows)),
        "best_mode_counts": dict(Counter(str(r.get("best_feedback_mode")) for r in rows)),
        "signature_score": _stats([r.get("signature_score") for r in rows]),
        "a_action_absmean": _stats([r.get("a_action_absmean") for r in rows]),
        "b_action_saturation": _stats([r.get("b_action_saturation") for r in rows]),
        "a_obs_value_std": _stats([r.get("a_obs_value_std") for r in rows]),
        "b_training_reward_sum": _stats([r.get("b_training_reward_sum") for r in rows]),
        "a_raw_reward_sum": _stats([r.get("a_raw_reward_sum") for r in rows]),
        "b_raw_reward_sum": _stats([r.get("b_raw_reward_sum") for r in rows]),
        "a_done_count": _stats([r.get("a_done_count") for r in rows]),
        "b_done_count": _stats([r.get("b_done_count") for r in rows]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", default=DEFAULT_INPUT_CSV)
    parser.add_argument("--preupdate-sidecar-a", default=DEFAULT_SIDECAR_A)
    parser.add_argument("--preupdate-sidecar-b", default=DEFAULT_SIDECAR_B)
    parser.add_argument("--input-rule", default="cross_seed_full_q90_pendulum_like_strict_n92")
    parser.add_argument("--filter-name", choices=sorted(FILTERS), default="sat_only")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=64)
    args = parser.parse_args()

    fixed_rows = [
        row for row in _load_csv(args.input_csv)
        if not str(args.input_rule) or str(row.get("rule")) == str(args.input_rule)
    ]
    side_a = _load_jsonl(args.preupdate_sidecar_a)
    side_b = _load_jsonl(args.preupdate_sidecar_b)
    if len(side_a) < len(fixed_rows) or len(side_b) < len(fixed_rows):
        raise RuntimeError("pre-update sidecar length is shorter than fixed-list rows")

    joined = _join_rows(fixed_rows, side_a, side_b)
    passing = [row for row in joined if _passes_filter(row, args.filter_name)]
    rejected = [row for row in joined if not _passes_filter(row, args.filter_name)]
    selected = _balanced_select(passing, int(args.limit))
    if len(selected) < int(args.limit):
        raise RuntimeError(
            f"filter {args.filter_name!r} left {len(passing)} passing candidates and "
            f"{len(selected)} selected, need {args.limit}"
        )

    output_rule = f"cross_seed_full_q90_pendulum_like_strict_{args.filter_name}_n{int(args.limit)}"
    out_dir = Path(args.output_dir).expanduser().resolve()
    group_dir = out_dir / output_rule
    group_dir.mkdir(parents=True, exist_ok=True)

    output_rows = []
    for out_idx, row in enumerate(selected):
        src = Path(str(row["full_frozen_h_json"])).expanduser().resolve()
        dst = group_dir / f"env_{out_idx:03d}_{row['source_tag']}_candidate_{int(row['source_candidate_index']):03d}.json"
        shutil.copyfile(src, dst)
        out_row = dict(row)
        out_row["rule"] = output_rule
        out_row["full_frozen_h_json"] = str(dst)
        output_rows.append(out_row)

    csv_path = out_dir / f"pendulum_like_{args.filter_name}_fixedlist.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0].keys()))
        writer.writeheader()
        for row in output_rows:
            writer.writerow(_json_safe(row))

    report = {
        "analysis_entry": "phase2_pendulum_like_recurrent_filter_fixedlist_builder",
        "contract": {
            "exploratory_only": True,
            "uses_existing_candidate_list": True,
            "uses_preupdate_sidecars_only": True,
            "does_not_modify_exact_scm": True,
            "does_not_modify_fit_model": True,
            "does_not_modify_ppo": True,
            "postopt_health_not_claimed_until_pack_smoke": True,
        },
        "input_csv": str(Path(args.input_csv).expanduser().resolve()),
        "input_rule": str(args.input_rule),
        "filter_name": str(args.filter_name),
        "filter": FILTERS[str(args.filter_name)],
        "output_rule": output_rule,
        "output_csv": str(csv_path),
        "input_count": int(len(joined)),
        "passing_count": int(len(passing)),
        "rejected_count": int(len(rejected)),
        "selected_count": int(len(output_rows)),
        "selected_summary": _summary(output_rows),
        "rejected_summary": _summary(rejected),
        "passing_summary": _summary(passing),
    }
    report_path = out_dir / f"pendulum_like_{args.filter_name}_fixedlist_report.json"
    report_path.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    md_lines = [
        f"# Pendulum-Like Recurrent Filter Fixed List: {args.filter_name}",
        "",
        f"- filter: `{FILTERS[str(args.filter_name)]['rule']}`",
        f"- input: {len(joined)}",
        f"- passing: {len(passing)}",
        f"- rejected: {len(rejected)}",
        f"- selected: {len(output_rows)}",
        f"- csv: `{csv_path}`",
        f"- output_rule: `{output_rule}`",
        "",
        "## Selected Diversity",
        "",
        f"- source_counts: {report['selected_summary']['source_counts']}",
        f"- best_sign_counts: {report['selected_summary']['best_sign_counts']}",
        f"- signature_score: {report['selected_summary']['signature_score']}",
        f"- b_action_saturation: {report['selected_summary']['b_action_saturation']}",
        f"- a_action_absmean: {report['selected_summary']['a_action_absmean']}",
        "",
        "## Contract",
        "",
        "- This fixed list is exploratory and does not change the generator milestone.",
        "- It claims only pre-update screening/yield control; health requires trusted pack-runner post/pre deltas.",
    ]
    md_path = out_dir / f"pendulum_like_{args.filter_name}_fixedlist_summary.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"wrote {csv_path}")
    print(f"wrote {report_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
