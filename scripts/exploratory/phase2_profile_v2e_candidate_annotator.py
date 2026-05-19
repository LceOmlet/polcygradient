#!/usr/bin/env python
"""Annotate the v2d fixed list as a v2e profile candidate.

This is a deliberately thin candidate layer:

* it reuses the v2d selected environments exactly;
* it adds near-best gain support annotations for feedback rows;
* it adds MountainCar future-basin detector coverage annotations;
* it does not resample, retrain, change exact SCM, or add quotas.

The output is a profile-governance artifact, not a new milestone.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


ART = Path("/home/chen/RLPFN/artifacts")
DEFAULT_SELECTED_CSV = (
    ART
    / "phase2_profile_band_selector_v2d_feedback_source_pool_repair_0511"
    / "selected_profile_band_prior_envs.csv"
)
DEFAULT_NEAR_BEST_CSV = (
    ART / "phase2_feedback_near_best_gain_audit_v2d_0511/feedback_near_best_gain_per_env.csv"
)
DEFAULT_MOUNTAINCAR_CSV = (
    ART / "phase2_mountaincar_future_basin_detector_v1_0511/mountaincar_future_basin_per_env.csv"
)
DEFAULT_OUTPUT_DIR = ART / "phase2_profile_v2e_candidate_annotated_0511"
RULES = ("gym_full_obs_action_range", "gym_q90_obs_action_range")
BROAD_AXES = (
    "locomotion_witness",
    "low_energy_not_ctrl_only",
    "sign_or_phase_sensitive",
    "state_conditioned_energy_injection",
)


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).expanduser().resolve().open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


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


def _safe_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _safe_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _safe_int(value: Any, default: int = -1) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def _near_key(row: dict[str, Any]) -> tuple[str, int, str]:
    return str(row.get("rule")), _safe_int(row.get("source_index", row.get("env_idx"))), str(row.get("profile"))


def _selected_source_key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row.get("candidate_rule")), _safe_int(row.get("_source_index"))


def _mc_key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row.get("source_label")), _safe_int(row.get("env_index"))


def _cell(row: dict[str, Any]) -> tuple[bool, ...]:
    return tuple(_safe_bool(row.get(axis)) for axis in BROAD_AXES)


def _entropy(cells: Counter[tuple[bool, ...]]) -> float:
    total = sum(cells.values())
    if total <= 0:
        return 0.0
    ent = 0.0
    for count in cells.values():
        p = count / total
        ent -= p * math.log(p + 1e-12)
    return float(ent / math.log(2 ** len(BROAD_AXES)))


def _materialize_fixed_groups(out_dir: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    manifest: dict[str, Any] = {}
    for rule in RULES:
        scoped = [row for row in rows if str(row.get("candidate_rule")) == rule]
        rule_dir = out_dir / rule / "fixed_env_group"
        frozen_dir = rule_dir / "frozen_h"
        frozen_dir.mkdir(parents=True, exist_ok=True)
        h_list = []
        seeds = []
        for idx, row in enumerate(scoped):
            src = Path(str(row["full_frozen_h_json"])).expanduser().resolve()
            h = _read_json(src)
            dst = frozen_dir / f"env_{idx:04d}_seed{_safe_int(row.get('env_seed'))}.json"
            shutil.copyfile(src, dst)
            h_list.append(h)
            seeds.append(_safe_int(row.get("env_seed")))
        h_path = rule_dir / "prior_fixed_h_list.json"
        seed_path = rule_dir / "prior_fixed_env_seeds.json"
        h_path.write_text(json.dumps(_json_safe(h_list), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        seed_path.write_text(json.dumps(seeds, indent=2) + "\n", encoding="utf-8")
        manifest[rule] = {
            "count": len(scoped),
            "fixed_h_list_json": str(h_path),
            "fixed_env_seed_json": str(seed_path),
        }
    return manifest


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    cells = Counter(_cell(row) for row in rows)
    strict = [row for row in rows if _safe_bool(row.get("strict_feedback_candidate"))]
    strict_with_near = [
        row
        for row in strict
        if row.get("v2e_feedback_profile_for_near_best") not in {"", None}
    ]
    mc_keys = [
        "mountaincar_future_basin_candidate",
        "mountaincar_future_basin_strong",
        "mountaincar_delayed_candidate",
        "mountaincar_future_basin_vs_zero",
        "mountaincar_future_basin_vs_open",
    ]
    out: dict[str, Any] = {
        "n": n,
        "joint_profile_active_cells": len(cells),
        "joint_profile_dominant_rate": max(cells.values()) / max(1, n) if cells else 0.0,
        "joint_profile_entropy_norm": _entropy(cells),
        "broad_axis_rates": {
            axis: float(sum(_safe_bool(row.get(axis)) for row in rows) / max(1, n))
            for axis in BROAD_AXES
        },
        "strict_feedback_count": len(strict),
        "strict_feedback_profile_counts": dict(Counter(str(row.get("strict_feedback_profile")) for row in strict)),
        "strict_feedback_gain_counts": dict(Counter(str(row.get("strict_feedback_gain")) for row in strict)),
        "strict_feedback_near_best_profile_count": len(strict_with_near),
        "strict_feedback_near_best_gain1_count": int(
            sum(_safe_bool(row.get("v2e_strict_near_best_gain_1")) for row in strict_with_near)
        ),
        "strict_feedback_near_best_gain1_rate": float(
            sum(_safe_bool(row.get("v2e_strict_near_best_gain_1")) for row in strict_with_near)
            / max(1, len(strict_with_near))
        ),
        "strict_feedback_near_best_gain05_count": int(
            sum(_safe_bool(row.get("v2e_strict_near_best_gain_0.5")) for row in strict_with_near)
        ),
        "strict_feedback_near_best_gain05_rate": float(
            sum(_safe_bool(row.get("v2e_strict_near_best_gain_0.5")) for row in strict_with_near)
            / max(1, len(strict_with_near))
        ),
        "mountaincar": {},
    }
    for key in mc_keys:
        count = int(sum(_safe_bool(row.get(key)) for row in rows))
        out["mountaincar"][f"{key}_count"] = count
        out["mountaincar"][f"{key}_rate"] = float(count / max(1, n))
    return out


def build_candidate(
    *,
    selected_csv: str | Path,
    near_best_csv: str | Path,
    mountaincar_csv: str | Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected = [dict(row) for row in _read_csv(selected_csv)]
    near = {_near_key(row): row for row in _read_csv(near_best_csv)}
    mountaincar = {_mc_key(row): row for row in _read_csv(mountaincar_csv)}
    rows: list[dict[str, Any]] = []
    for row in selected:
        out: dict[str, Any] = dict(row)
        out["profile_version_id"] = "profile_band_v2e_thin_annotated_candidate"
        out["v2e_candidate_layer"] = "annotation_only_reuses_v2d_fixed_list"
        out["v2e_near_best_gain_guard_available"] = False
        if _safe_bool(row.get("strict_feedback_candidate")):
            profile = str(row.get("strict_feedback_profile"))
            nb = near.get((str(row.get("candidate_rule")), _safe_int(row.get("_source_index")), profile))
            if nb is not None:
                out["v2e_feedback_profile_for_near_best"] = profile
                out["v2e_near_best_gain_guard_available"] = True
                out["v2e_near_best_best_gain"] = nb.get("best_gain")
                out["v2e_strict_near_best_gain_1"] = nb.get("near_best_gain_1")
                out["v2e_strict_near_best_gain_0.5"] = nb.get("near_best_gain_0.5")
                out["v2e_gap_gain_1"] = nb.get("gap_gain_1")
                out["v2e_gap_gain_0.5"] = nb.get("gap_gain_0.5")
        mc = mountaincar.get(_selected_source_key(row))
        if mc is not None:
            for key, value in mc.items():
                if key.startswith("mountaincar_") or key in {
                    "early_prefix_gain_vs_zero_env",
                    "early_suffix_gain_vs_zero_env",
                    "early_suffix_gain_vs_open_env",
                    "early_prefix_action_abs",
                    "early_suffix_action_abs",
                    "best_early_only_mode",
                }:
                    out[key] = value
        rows.append(out)
    report = {
        "analysis_entry": "phase2_profile_v2e_candidate_annotator",
        "contract": {
            "exploratory_only": True,
            "annotation_only": True,
            "reuses_v2d_fixed_list_exactly": True,
            "does_not_sample_environments": True,
            "does_not_train": True,
            "does_not_modify_exact_scm": True,
            "does_not_modify_fit_model": True,
            "does_not_modify_ppo": True,
            "delta_positive_not_a_gate": True,
            "profile_quality_first": True,
        },
        "inputs": {
            "selected_csv": str(Path(selected_csv).expanduser().resolve()),
            "near_best_csv": str(Path(near_best_csv).expanduser().resolve()),
            "mountaincar_csv": str(Path(mountaincar_csv).expanduser().resolve()),
        },
        "rules": {},
    }
    for rule in RULES:
        scoped = [row for row in rows if str(row.get("candidate_rule")) == rule]
        report["rules"][rule] = _summary(scoped)
    full = report["rules"]["gym_full_obs_action_range"]
    q90 = report["rules"]["gym_q90_obs_action_range"]
    report["read"] = {
        "complexity_safe": True,
        "v2e_profile_annotation_complete": bool(
            full["strict_feedback_near_best_profile_count"] == full["strict_feedback_count"]
            and q90["strict_feedback_near_best_profile_count"] == q90["strict_feedback_count"]
        ),
        "near_best_gain1_guard_read": (
            "available as a submetric; not a hard quota. full is stronger than q90, so use as guard/diagnostic first."
        ),
        "mountaincar_read": "coverage annotated only; still not quota-ready",
        "candidate_next_step": (
            "If this annotated candidate is used for training smoke, it is the same fixed env list as v2d; "
            "new value comes from profile governance/guards, not a distribution change."
        ),
    }
    return rows, report


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Profile v2e Thin Candidate",
        "",
        "Annotation-only candidate: reuses v2d fixed list exactly; no sampling or training.",
        "",
        "| rule | n | strict | near-best gain1 | near-best gain0.5 | MC candidate | MC strong | cells | dom cell |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rule in RULES:
        s = report["rules"][rule]
        mc = s["mountaincar"]
        lines.append(
            f"| `{rule}` | {s['n']} | {s['strict_feedback_count']} | "
            f"{s['strict_feedback_near_best_gain1_rate']:.3f} | "
            f"{s['strict_feedback_near_best_gain05_rate']:.3f} | "
            f"{mc['mountaincar_future_basin_candidate_count']} | "
            f"{mc['mountaincar_future_basin_strong_count']} | "
            f"{s['joint_profile_active_cells']} | {s['joint_profile_dominant_rate']:.3f} |"
        )
    lines.extend(["", "## Read", ""])
    for key, value in report["read"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Files", ""])
    for key, value in report.get("outputs", {}).items():
        lines.append(f"- `{key}`: `{value}`")
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, report = build_candidate(
        selected_csv=args.selected_csv,
        near_best_csv=args.near_best_csv,
        mountaincar_csv=args.mountaincar_csv,
    )
    selected_path = out_dir / "selected_profile_v2e_candidate_prior_envs.csv"
    json_path = out_dir / "profile_v2e_candidate_report.json"
    md_path = out_dir / "profile_v2e_candidate.md"
    _write_csv(selected_path, rows)
    report["fixed_groups"] = _materialize_fixed_groups(out_dir, rows)
    report["outputs"] = {
        "selected_csv": str(selected_path),
        "report_json": str(json_path),
        "markdown": str(md_path),
    }
    json_path.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report["outputs"], indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-csv", default=str(DEFAULT_SELECTED_CSV))
    parser.add_argument("--near-best-csv", default=str(DEFAULT_NEAR_BEST_CSV))
    parser.add_argument("--mountaincar-csv", default=str(DEFAULT_MOUNTAINCAR_CSV))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
