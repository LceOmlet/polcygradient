#!/usr/bin/env python
"""Audit next profile pressure points after milestone2/v2c.

Exploratory/read-only.  This script keeps the profile mainline explicit:

* post-pre delta mean and diversity are guards, not objectives;
* delta-positive fraction is monitoring only;
* feedback strict collapse is treated as a subprofile/source-pool issue;
* MountainCar future-basin is treated as Gym-anchored but not quota-ready until
  it passes a prior fixed-list guard.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


ART = Path("/home/chen/RLPFN/artifacts")
DEFAULT_OUTPUT_DIR = ART / "phase2_profile_next_pressure_audit_v2c_0511"
DEFAULT_SELECTED_CSV = (
    ART
    / "phase2_profile_band_selector_v2c_centered_protected_feedback_safe_0511"
    / "selected_profile_band_prior_envs.csv"
)
DEFAULT_MOUNTAINCAR_CSV = (
    ART
    / "phase2_mountaincar_future_basin_detector_v1_0511"
    / "mountaincar_future_basin_per_env.csv"
)
DEFAULT_SMOKE_CSVS = [
    ART
    / "phase2_profile_band_v2c_centered_protected_feedback_safe_delta_smoke_summary_0511"
    / "locomotion_guarded_smoke_per_env.csv",
    ART
    / "phase2_profile_band_v2c_centered_protected_feedback_safe_xseed_delta_smoke_summary_0511"
    / "locomotion_guarded_smoke_per_env.csv",
]
DEFAULT_FEEDBACK_CONTROL_JSON = (
    ART / "phase2_feedback_subprofile_control_audit_0511/feedback_subprofile_control_audit.json"
)
DEFAULT_STRICT_FRONTIER_JSON = (
    ART
    / "phase2_strict_profile_gain_balance_frontier_formal_0510"
    / "strict_profile_gain_balance_frontier_report.json"
)

RULES = ("gym_full_obs_action_range", "gym_q90_obs_action_range")
MOUNTAINCAR_KEYS = (
    "mountaincar_future_basin_candidate",
    "mountaincar_future_basin_strong",
    "mountaincar_delayed_candidate",
    "mountaincar_future_basin_vs_zero",
    "mountaincar_future_basin_vs_open",
)


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).expanduser().resolve().open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


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


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _safe_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _stats(values: list[Any]) -> dict[str, Any]:
    arr = np.asarray([float(v) for v in values if _safe_float(v) is not None], dtype=np.float64)
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "q10": float(np.quantile(arr, 0.10)),
        "q50": float(np.quantile(arr, 0.50)),
        "q90": float(np.quantile(arr, 0.90)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "delta_positive_fraction_monitor": float(np.mean(arr > 0.0)),
    }


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


def _selected_key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row["candidate_rule"]), int(float(row["profile_band_rank"]))


def _source_key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row["candidate_rule"]), int(float(row["_source_index"]))


def _smoke_key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row["rule"]), int(float(row["env_idx"]))


def _mountaincar_key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row["source_label"]), int(float(row["env_index"]))


def _scope_summary(
    *,
    selected: list[dict[str, str]],
    mountaincar_by_source: dict[tuple[str, int], dict[str, str]],
) -> dict[str, Any]:
    out: dict[str, Any] = {"n": len(selected)}
    for key in MOUNTAINCAR_KEYS:
        vals = [
            _safe_bool(mountaincar_by_source[_source_key(row)].get(key))
            for row in selected
            if _source_key(row) in mountaincar_by_source
        ]
        out[f"{key}_count"] = int(sum(vals))
        out[f"{key}_rate"] = float(sum(vals) / max(1, len(vals)))
    modes = [
        str(mountaincar_by_source[_source_key(row)].get("best_early_only_mode"))
        for row in selected
        if _source_key(row) in mountaincar_by_source
    ]
    out["best_early_only_mode_top"] = [
        {"mode": mode, "count": count, "rate": float(count / max(1, len(modes)))}
        for mode, count in Counter(modes).most_common(8)
    ]
    return out


def _delta_by_mountaincar(
    *,
    selected_by_rank: dict[tuple[str, int], dict[str, str]],
    mountaincar_by_source: dict[tuple[str, int], dict[str, str]],
    smoke_rows: list[dict[str, str]],
    run_name: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rule in RULES:
        scoped = [row for row in smoke_rows if str(row.get("rule")) == rule]
        for key in MOUNTAINCAR_KEYS:
            for value in (True, False):
                vals_raw = []
                vals_env = []
                n = 0
                for smoke in scoped:
                    selected = selected_by_rank.get(_smoke_key(smoke))
                    if selected is None:
                        continue
                    mc = mountaincar_by_source.get(_source_key(selected))
                    if mc is None or _safe_bool(mc.get(key)) != value:
                        continue
                    n += 1
                    vals_raw.append(smoke.get("raw_reward_sum_delta"))
                    vals_env.append(smoke.get("reward_env_sum_delta"))
                raw = _stats(vals_raw)
                env = _stats(vals_env)
                rows.append(
                    {
                        "run": run_name,
                        "rule": rule,
                        "mountaincar_key": key,
                        "value": value,
                        "n": n,
                        "raw_delta_mean": raw.get("mean"),
                        "env_delta_mean": env.get("mean"),
                        "raw_delta_positive_fraction_monitor": raw.get(
                            "delta_positive_fraction_monitor"
                        ),
                    }
                )
    return rows


def _feedback_read(strict: dict[str, Any], control: dict[str, Any]) -> dict[str, Any]:
    pool = strict.get("pool_summary", {})
    profile_dom = (pool.get("profile_dominance") or {})
    gain_dom = (pool.get("gain_dominance") or {})
    return {
        "pressure_type": "source_pool_subprofile_collapse",
        "profile_dominant": profile_dom.get("dominant"),
        "profile_dominant_share": profile_dom.get("dominant_share"),
        "gain_dominant": gain_dom.get("dominant"),
        "gain_dominant_share": gain_dom.get("dominant_share"),
        "selection_only_sufficient": strict.get("decision", {}).get(
            "selection_only_is_sufficient"
        ),
        "simple_existing_scalar_knob_sufficient": control.get("simple_knob_sufficient"),
        "recommended_control": (
            "do not raise global strict count; either broaden the behavioral label or add a narrow "
            "generator-level categorical profile/gain branch, then guard by delta/diversity"
        ),
    }


def build_report(
    *,
    selected_csv: str | Path,
    mountaincar_csv: str | Path,
    smoke_csvs: list[str | Path],
    feedback_control_json: str | Path,
    strict_frontier_json: str | Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selected = _read_csv(selected_csv)
    selected_by_rank = {_selected_key(row): row for row in selected}
    mountaincar_by_source = {_mountaincar_key(row): row for row in _read_csv(mountaincar_csv)}
    strict = _read_json(strict_frontier_json)
    control = _read_json(feedback_control_json)

    scope_summaries = {}
    for rule in RULES:
        scoped = [row for row in selected if str(row.get("candidate_rule")) == rule]
        scope_summaries[rule] = _scope_summary(
            selected=scoped,
            mountaincar_by_source=mountaincar_by_source,
        )

    delta_rows: list[dict[str, Any]] = []
    for path in smoke_csvs:
        smoke_path = Path(path).expanduser().resolve()
        run_name = smoke_path.parent.name
        delta_rows.extend(
            _delta_by_mountaincar(
                selected_by_rank=selected_by_rank,
                mountaincar_by_source=mountaincar_by_source,
                smoke_rows=_read_csv(smoke_path),
                run_name=run_name,
            )
        )

    full = scope_summaries["gym_full_obs_action_range"]
    q90 = scope_summaries["gym_q90_obs_action_range"]
    mc_guard_pass = bool(
        full["mountaincar_future_basin_candidate_count"] >= 8
        and q90["mountaincar_future_basin_candidate_count"] >= 8
        and full["mountaincar_future_basin_strong_count"] >= 4
        and q90["mountaincar_future_basin_strong_count"] >= 4
    )
    report = {
        "analysis_entry": "phase2_profile_next_pressure_audit",
        "contract": {
            "exploratory_only": True,
            "read_only_existing_artifacts": True,
            "does_not_sample_environments": True,
            "does_not_train": True,
            "does_not_modify_exact_scm": True,
            "does_not_modify_fit_model": True,
            "does_not_modify_ppo": True,
            "delta_positive_fraction_monitor_only": True,
            "profile_quality_first": True,
        },
        "inputs": {
            "selected_csv": str(Path(selected_csv).expanduser().resolve()),
            "mountaincar_csv": str(Path(mountaincar_csv).expanduser().resolve()),
            "smoke_csvs": [str(Path(path).expanduser().resolve()) for path in smoke_csvs],
            "feedback_control_json": str(Path(feedback_control_json).expanduser().resolve()),
            "strict_frontier_json": str(Path(strict_frontier_json).expanduser().resolve()),
        },
        "feedback_source_pool_read": _feedback_read(strict, control),
        "mountaincar_prior_fixedlist_guard": {
            "scope_summaries": scope_summaries,
            "guard_pass": mc_guard_pass,
            "guard_definition": (
                "candidate >= 8/64 and strong >= 4/64 in both full and q90 before entering quota"
            ),
            "read": (
                "MountainCar future-basin is Gym-anchored, but the selected prior fixed lists do not yet "
                "show enough stable prior-side coverage for quota control."
                if not mc_guard_pass
                else "MountainCar future-basin prior coverage is sufficient for a guarded quota candidate."
            ),
        },
        "next_pressure_ranking": [
            {
                "pressure": "feedback_strict_source_pool_collapse",
                "action": "fix definition/source pool before increasing strict fraction",
                "why": "strict source pool is dominated by constant profile and gain=2.0",
            },
            {
                "pressure": "mountaincar_future_basin_prior_guard_missing",
                "action": "validate or broaden prior detector/fixed-list coverage before quota",
                "why": "Gym anchor is strong, but selected prior coverage is sparse; keep it as detector/coverage guard",
            },
            {
                "pressure": "broad_axis_delta_association_unstable",
                "action": "avoid single-axis delta optimization; use joint profile/subprofile guards",
                "why": "existing cross-seed delta pressure reports show scope-dependent sign flips",
            },
        ],
        "delta_by_mountaincar_rows": delta_rows,
    }
    return report, delta_rows


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Profile Next Pressure Audit",
        "",
        "Read-only profile audit.  `delta+` is monitoring only; this report is about profile pressure and post-pre delta mean guards.",
        "",
        "## Feedback Source-Pool Read",
        "",
    ]
    fb = report["feedback_source_pool_read"]
    lines.extend(
        [
            f"- pressure type: `{fb['pressure_type']}`",
            f"- profile dominant: `{fb['profile_dominant']}` {100.0 * float(fb['profile_dominant_share']):.1f}%",
            f"- gain dominant: `{fb['gain_dominant']}` {100.0 * float(fb['gain_dominant_share']):.1f}%",
            f"- selection-only sufficient: `{fb['selection_only_sufficient']}`",
            f"- scalar knob sufficient: `{fb['simple_existing_scalar_knob_sufficient']}`",
            f"- recommendation: {fb['recommended_control']}",
            "",
            "## MountainCar Prior Fixed-List Guard",
            "",
            f"- guard pass: `{report['mountaincar_prior_fixedlist_guard']['guard_pass']}`",
            f"- read: {report['mountaincar_prior_fixedlist_guard']['read']}",
            "",
            "| rule | candidate | strong | delayed | future-vs-zero | future-vs-open |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for rule in RULES:
        s = report["mountaincar_prior_fixedlist_guard"]["scope_summaries"][rule]
        lines.append(
            "| {rule} | {cand}/{n} | {strong}/{n} | {delayed}/{n} | {zero}/{n} | {open}/{n} |".format(
                rule=rule,
                n=s["n"],
                cand=s["mountaincar_future_basin_candidate_count"],
                strong=s["mountaincar_future_basin_strong_count"],
                delayed=s["mountaincar_delayed_candidate_count"],
                zero=s["mountaincar_future_basin_vs_zero_count"],
                open=s["mountaincar_future_basin_vs_open_count"],
            )
        )
    lines.extend(
        [
            "",
            "## Next Pressure Ranking",
            "",
        ]
    )
    for item in report["next_pressure_ranking"]:
        lines.append(f"- `{item['pressure']}`: {item['action']} ({item['why']})")
    lines.extend(["", "## Files", ""])
    for key, value in report.get("outputs", {}).items():
        lines.append(f"- `{key}`: `{value}`")
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    smoke_csvs = [] if bool(getattr(args, "no_smoke_delta", False)) else list(args.smoke_csv)
    report, delta_rows = build_report(
        selected_csv=args.selected_csv,
        mountaincar_csv=args.mountaincar_csv,
        smoke_csvs=smoke_csvs,
        feedback_control_json=args.feedback_control_json,
        strict_frontier_json=args.strict_frontier_json,
    )
    json_path = out_dir / "profile_next_pressure_audit.json"
    md_path = out_dir / "profile_next_pressure_audit.md"
    delta_csv = out_dir / "mountaincar_delta_by_label.csv"
    report["outputs"] = {
        "json": str(json_path),
        "markdown": str(md_path),
        "mountaincar_delta_csv": str(delta_csv),
    }
    _write_csv(delta_csv, delta_rows)
    json_path.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report["outputs"], indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--selected-csv", default=str(DEFAULT_SELECTED_CSV))
    parser.add_argument("--mountaincar-csv", default=str(DEFAULT_MOUNTAINCAR_CSV))
    parser.add_argument("--smoke-csv", action="append", default=[str(path) for path in DEFAULT_SMOKE_CSVS])
    parser.add_argument(
        "--no-smoke-delta",
        action="store_true",
        help="Only audit selected prior coverage; do not join smoke delta rows from another fixed list.",
    )
    parser.add_argument("--feedback-control-json", default=str(DEFAULT_FEEDBACK_CONTROL_JSON))
    parser.add_argument("--strict-frontier-json", default=str(DEFAULT_STRICT_FRONTIER_JSON))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
