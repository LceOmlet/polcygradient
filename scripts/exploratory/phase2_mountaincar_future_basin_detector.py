#!/usr/bin/env python
"""Candidate MountainCar-like upfront-investment/future-basin detector.

Exploratory/read-only.  This does not sample, train, or modify the prior
generator.  It reuses the existing time-profile mode probe and turns the
MountainCar-like idea into falsifiable per-env clauses:

1. upfront action investment: early-only action is active in the prefix and
   near-zero in the suffix;
2. future basin payoff: the suffix env reward is better than zero/open-loop by
   a dynamic margin;
3. full payoff: the full env return repays the early action and beats the same
   baseline by the same dynamic margin.

This detector is intentionally not quota-ready yet.  It is the second necessary
profile item after feedback strict subprofile control.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ART = Path("/home/chen/RLPFN/artifacts")
DEFAULT_INPUT_JSON = (
    ART
    / "phase2_prior_pendulum_like_signature_probe_seed6262_0510"
    / "gated_full_q90_seed6262_n128_s128_obs4_scales3_timeprofile.json"
)
DEFAULT_OUTPUT_DIR = ART / "phase2_mountaincar_future_basin_detector_v1_0511"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    return value


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _stats(values: list[float]) -> dict[str, Any]:
    arr = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=np.float64)
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "min": float(np.min(arr)),
        "q10": float(np.quantile(arr, 0.10)),
        "q50": float(np.quantile(arr, 0.50)),
        "q90": float(np.quantile(arr, 0.90)),
        "max": float(np.max(arr)),
        "positive_fraction": float(np.mean(arr > 0.0)),
    }


def _classify_row(row: dict[str, Any], *, args: argparse.Namespace) -> dict[str, Any]:
    threshold = _safe_float(row.get("threshold"))
    margin = max(float(args.abs_margin), float(args.threshold_fraction) * threshold)
    prefix_action = _safe_float(row.get("best_early_only_action_abs_prefix"))
    suffix_action = _safe_float(row.get("best_early_only_action_abs_suffix"))
    prefix_gain_zero = _safe_float(row.get("best_early_only_prefix_env_return")) - _safe_float(
        row.get("zero_prefix_env_return")
    )
    suffix_gain_zero = _safe_float(row.get("early_only_suffix_gain_over_zero_env"))
    suffix_gain_open = _safe_float(row.get("early_only_suffix_gain_over_open_env"))
    action_decay = prefix_action > max(float(args.min_prefix_action_abs), suffix_action + float(args.action_decay_margin))
    suffix_zero = suffix_gain_zero > margin
    suffix_open = suffix_gain_open > margin
    full_zero = bool(row.get("early_only_beats_zero_env"))
    full_open = bool(row.get("early_only_beats_open_loop_env"))
    delayed_zero = suffix_gain_zero > prefix_gain_zero + float(args.delayed_margin_fraction) * margin
    candidate = action_decay and suffix_zero and full_zero
    strong = action_decay and suffix_open and full_open
    delayed_candidate = candidate and delayed_zero
    return {
        "mountaincar_upfront_action_decay": bool(action_decay),
        "mountaincar_future_basin_vs_zero": bool(suffix_zero),
        "mountaincar_future_basin_vs_open": bool(suffix_open),
        "mountaincar_full_payoff_vs_zero": bool(full_zero),
        "mountaincar_full_payoff_vs_open": bool(full_open),
        "mountaincar_delayed_suffix_dominates_prefix": bool(delayed_zero),
        "mountaincar_future_basin_candidate": bool(candidate),
        "mountaincar_future_basin_strong": bool(strong),
        "mountaincar_delayed_candidate": bool(delayed_candidate),
        "mountaincar_margin": float(margin),
        "early_prefix_gain_vs_zero_env": float(prefix_gain_zero),
        "early_suffix_gain_vs_zero_env": float(suffix_gain_zero),
        "early_suffix_gain_vs_open_env": float(suffix_gain_open),
        "early_prefix_action_abs": float(prefix_action),
        "early_suffix_action_abs": float(suffix_action),
        "best_early_only_mode": row.get("best_early_only_mode"),
    }


def _summary(rows: list[dict[str, Any]], *, label: str) -> dict[str, Any]:
    n = len(rows)
    keys = [
        "mountaincar_upfront_action_decay",
        "mountaincar_future_basin_vs_zero",
        "mountaincar_future_basin_vs_open",
        "mountaincar_full_payoff_vs_zero",
        "mountaincar_full_payoff_vs_open",
        "mountaincar_future_basin_candidate",
        "mountaincar_future_basin_strong",
        "mountaincar_delayed_candidate",
    ]
    out: dict[str, Any] = {"label": label, "n": n}
    for key in keys:
        count = int(sum(bool(row.get(key)) for row in rows))
        out[f"{key}_count"] = count
        out[f"{key}_rate"] = float(count / max(1, n))
    out["suffix_gain_vs_zero"] = _stats([_safe_float(row.get("early_suffix_gain_vs_zero_env")) for row in rows])
    out["suffix_gain_vs_open"] = _stats([_safe_float(row.get("early_suffix_gain_vs_open_env")) for row in rows])
    out["prefix_gain_vs_zero"] = _stats([_safe_float(row.get("early_prefix_gain_vs_zero_env")) for row in rows])
    out["prefix_action_abs"] = _stats([_safe_float(row.get("early_prefix_action_abs")) for row in rows])
    return out


def run(args: argparse.Namespace) -> None:
    input_json = Path(args.input_json).expanduser().resolve()
    report = _read_json(input_json)
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    per_env: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for item in report.get("lists", []):
        label = str(item.get("label"))
        rows = []
        for raw in (item.get("classification") or {}).get("per_env", []):
            classified = {
                "source_label": label,
                "env_index": int(raw.get("env_index")),
                **_classify_row(raw, args=args),
            }
            rows.append(classified)
            per_env.append(classified)
        summaries[label] = _summary(rows, label=label)

    per_env_csv = out_dir / "mountaincar_future_basin_per_env.csv"
    report_json = out_dir / "mountaincar_future_basin_detector_report.json"
    report_md = out_dir / "mountaincar_future_basin_detector.md"
    _write_csv(per_env_csv, per_env)
    out = {
        "analysis_entry": "phase2_mountaincar_future_basin_detector",
        "contract": {
            "exploratory_only": True,
            "read_only_existing_time_profile_probe": True,
            "does_not_sample_environments": True,
            "does_not_train": True,
            "does_not_modify_exact_scm": True,
            "does_not_modify_fit_model": True,
            "does_not_modify_ppo": True,
            "not_quota_ready": True,
        },
        "definition": {
            "upfront_action_investment": "best early-only action has prefix action and decays to suffix",
            "future_basin_payoff": "suffix env reward improves over zero/open by dynamic margin",
            "full_payoff": "full env reward beats zero/open by the existing probe margin",
            "candidate": "upfront action decay + suffix payoff vs zero + full payoff vs zero",
            "strong": "upfront action decay + suffix payoff vs open-loop + full payoff vs open-loop",
        },
        "input_json": str(input_json),
        "config": {
            "abs_margin": float(args.abs_margin),
            "threshold_fraction": float(args.threshold_fraction),
            "action_decay_margin": float(args.action_decay_margin),
            "min_prefix_action_abs": float(args.min_prefix_action_abs),
            "delayed_margin_fraction": float(args.delayed_margin_fraction),
        },
        "summaries": summaries,
        "all_summary": _summary(per_env, label="all"),
        "outputs": {
            "per_env_csv": str(per_env_csv),
            "report_json": str(report_json),
            "summary_md": str(report_md),
        },
        "read": {
            "detector_candidate_exists": True,
            "quota_ready": False,
            "why_not_quota_ready": (
                "It reuses an existing finite time-profile probe and still needs Gym-side MountainCar "
                "counterfactual anchoring plus fixed-list health/diversity validation before entering profile bands."
            ),
        },
    }
    report_json.write_text(json.dumps(_json_safe(out), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# MountainCar Future-Basin Detector",
        "",
        "Exploratory/read-only detector candidate; not quota-ready.",
        "",
        "## Definition",
        "",
        "- `upfront_action_decay`: early-only policy has nontrivial prefix action and near-zero suffix action.",
        "- `future_basin_vs_zero/open`: suffix env reward improves over zero/open by a dynamic margin.",
        "- `candidate`: upfront action decay + suffix payoff vs zero + full payoff vs zero.",
        "- `strong`: upfront action decay + suffix payoff vs open-loop + full payoff vs open-loop.",
        "",
        "## Summary",
        "",
        "| label | n | candidate | strong | delayed | suffix-zero mean | suffix-open mean | prefix-action mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, item in summaries.items():
        lines.append(
            f"| `{label}` | {item['n']} | {item['mountaincar_future_basin_candidate_rate']:.3f} | "
            f"{item['mountaincar_future_basin_strong_rate']:.3f} | "
            f"{item['mountaincar_delayed_candidate_rate']:.3f} | "
            f"{item['suffix_gain_vs_zero'].get('mean')} | "
            f"{item['suffix_gain_vs_open'].get('mean')} | "
            f"{item['prefix_action_abs'].get('mean')} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "This establishes a maintainable candidate detector, but it is not sufficient for a quota. "
            "The next validation is Gym-side MountainCar anchoring and fixed-list health/diversity smoke.",
            "",
            f"- JSON: `{report_json}`",
            f"- per-env CSV: `{per_env_csv}`",
        ]
    )
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(out["outputs"], indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", default=str(DEFAULT_INPUT_JSON))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--abs-margin", type=float, default=5.0)
    parser.add_argument("--threshold-fraction", type=float, default=0.50)
    parser.add_argument("--action-decay-margin", type=float, default=0.02)
    parser.add_argument("--min-prefix-action-abs", type=float, default=0.02)
    parser.add_argument("--delayed-margin-fraction", type=float, default=0.25)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
