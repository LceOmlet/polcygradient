#!/usr/bin/env python
"""Build a joint-profile guarded prior candidate fixed list.

Exploratory only.  This differs from single-target band guards: the broad mode
profile itself is constrained.  The candidate raises the scarce locomotion mode
only after keeping low-energy / sign-phase / state-conditioned rates in bands
and capping every joint-profile cell.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


ARTIFACT_ROOT = Path("/home/chen/RLPFN/artifacts")
DEFAULT_SELECTED_CSV = (
    ARTIFACT_ROOT
    / "phase2_exploratory_reward_group_balance_gated_n128_seed6262_0510/selected_prior_envs.csv"
)
DEFAULT_LOCOMOTION_CSV = (
    ARTIFACT_ROOT
    / "phase2_locomotion_full_temporal_detector_gated_n128_gpu0_0510/locomotion_full_temporal_per_env.csv"
)
DEFAULT_ACTION_MODE_JSON = (
    ARTIFACT_ROOT
    / "phase2_prior_action_mode_coverage_gated_n128_gpu0_0510/gated_n128_action_mode_coverage.json"
)

RULES = ("gym_full_obs_action_range", "gym_q90_obs_action_range")
MODE_KEYS = (
    "locomotion_witness",
    "low_energy_not_ctrl_only",
    "sign_or_phase_sensitive",
    "state_conditioned_energy_injection",
)

# Milestone-2 profile rates from the unified mode audit.  Locomotion is the
# only raised target; the other modes are protected bands.
BASELINE_RATES = {
    "gym_full_obs_action_range": {
        "locomotion_witness": 0.078125,
        "low_energy_not_ctrl_only": 0.53125,
        "sign_or_phase_sensitive": 0.65625,
        "state_conditioned_energy_injection": 0.453125,
    },
    "gym_q90_obs_action_range": {
        "locomotion_witness": 0.171875,
        "low_energy_not_ctrl_only": 0.421875,
        "sign_or_phase_sensitive": 0.671875,
        "state_conditioned_energy_injection": 0.578125,
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
        value = float(value)
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    return value


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n", "", "none", "nan"}:
        return False
    return bool(value)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).expanduser().resolve().open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def _safe_label(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")[:160] or "rule"


def _load_action_labels(path: str | Path) -> dict[tuple[str, int], dict[str, bool]]:
    report = _read_json(path)
    labels: dict[tuple[str, int], dict[str, bool]] = {}
    for item in report.get("lists", []):
        label = str(item.get("label", ""))
        for row in item.get("classification", {}).get("per_env", []):
            idx = int(row["env_index"])
            labels[(label, idx)] = {
                "low_energy_not_ctrl_only": bool(row.get("low_energy_not_ctrl_only", False)),
                "sign_or_phase_sensitive": bool(row.get("sign_or_phase_sensitive_present", False)),
                "state_conditioned_energy_injection": bool(
                    row.get("state_conditioned_energy_injection_witness_present", False)
                ),
            }
    return labels


def _load_locomotion_labels(path: str | Path) -> dict[tuple[str, int], dict[str, Any]]:
    labels: dict[tuple[str, int], dict[str, Any]] = {}
    for row in _read_csv(path):
        scope = str(row.get("scope", ""))
        idx = int(float(row["env_index"]))
        labels[(scope, idx)] = {
            "locomotion_witness": _safe_bool(row.get("sustained_locomotion_temporal_witness")),
            "locomotion_env_gain_vs_zero": _safe_float(row.get("env_reward_gain_vs_zero")),
        }
    return labels


def _merge_rows(
    selected_rows: list[dict[str, str]],
    loco_labels: dict[tuple[str, int], dict[str, Any]],
    action_labels: dict[tuple[str, int], dict[str, bool]],
) -> dict[str, list[dict[str, Any]]]:
    by_rule: dict[str, list[dict[str, Any]]] = {rule: [] for rule in RULES}
    for row in selected_rows:
        rule = str(row.get("rule"))
        if rule not in by_rule:
            continue
        idx = len(by_rule[rule])
        merged: dict[str, Any] = dict(row)
        merged["_source_index"] = int(idx)
        merged.update(loco_labels.get((rule, idx), {}))
        merged.update(action_labels.get((rule, idx), {}))
        by_rule[rule].append(merged)
    return by_rule


def _profile(rows: list[dict[str, Any]], idx: int) -> tuple[bool, ...]:
    row = rows[idx]
    return tuple(_safe_bool(row.get(key)) for key in MODE_KEYS)


def _rate(rows: list[dict[str, Any]], key: str) -> float:
    return float(sum(_safe_bool(row.get(key)) for row in rows) / max(1, len(rows)))


def _summary(rows: list[dict[str, Any]], *, rule: str) -> dict[str, Any]:
    counter = Counter(tuple(_safe_bool(row.get(key)) for key in MODE_KEYS) for row in rows)
    dominant = max(counter.values()) if counter else 0
    out: dict[str, Any] = {
        "rule": rule,
        "n": len(rows),
        "joint_profile_active_cells": len(counter),
        "joint_profile_dominant_count": dominant,
        "joint_profile_dominant_rate": dominant / max(1, len(rows)),
        "top_joint_profiles": [
            {
                "profile": {key: value for key, value in zip(MODE_KEYS, profile)},
                "count": count,
                "rate": count / max(1, len(rows)),
            }
            for profile, count in counter.most_common(8)
        ],
    }
    for key in MODE_KEYS:
        out[f"{key}_count"] = int(sum(_safe_bool(row.get(key)) for row in rows))
        out[f"{key}_rate"] = _rate(rows, key)
    return out


def _count_band(rate: float, target_n: int, tolerance: float) -> tuple[int, int]:
    low = max(0, math.ceil((rate - tolerance) * target_n - 1e-9))
    high = min(target_n, math.floor((rate + tolerance) * target_n + 1e-9))
    return int(low), int(high)


def _select(
    rows: list[dict[str, Any]],
    *,
    rule: str,
    target_n: int,
    protected_tolerance: float,
    locomotion_min_rate: float,
    locomotion_max_rate: float,
    profile_cell_cap_rate: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    n = len(rows)
    if n < target_n:
        raise ValueError(f"{rule}: need {target_n}, got {n}")

    a_rows = [np.ones(n, dtype=np.float64)]
    lb = [float(target_n)]
    ub = [float(target_n)]

    band_details: dict[str, Any] = {}
    for key in MODE_KEYS:
        indicator = np.asarray([1.0 if _safe_bool(row.get(key)) else 0.0 for row in rows])
        if key == "locomotion_witness":
            low, high = _count_band((locomotion_min_rate + locomotion_max_rate) / 2.0, target_n, 1.0)
            low = max(low, math.ceil(locomotion_min_rate * target_n - 1e-9))
            high = min(high, math.floor(locomotion_max_rate * target_n + 1e-9))
            kind = "raised_target"
        else:
            low, high = _count_band(BASELINE_RATES[rule][key], target_n, protected_tolerance)
            kind = "protected"
        a_rows.append(indicator)
        lb.append(float(low))
        ub.append(float(high))
        band_details[key] = {
            "kind": kind,
            "min_count": int(low),
            "max_count": int(high),
            "source_count": int(indicator.sum()),
            "source_rate": float(indicator.mean()),
        }

    profile_cap = math.floor(profile_cell_cap_rate * target_n + 1e-9)
    profile_details: dict[str, Any] = {}
    for bits in sorted({tuple(_safe_bool(row.get(key)) for key in MODE_KEYS) for row in rows}):
        indicator = np.asarray([1.0 if _profile(rows, idx) == bits else 0.0 for idx in range(n)])
        a_rows.append(indicator)
        lb.append(0.0)
        ub.append(float(profile_cap))
        profile_details["".join("1" if b else "0" for b in bits)] = {
            "profile": {key: value for key, value in zip(MODE_KEYS, bits)},
            "source_count": int(indicator.sum()),
            "source_rate": float(indicator.mean()),
            "max_count": int(profile_cap),
        }

    # Objective is deliberately weak.  Constraints define the distribution;
    # the objective only prefers stronger locomotion witnesses within it.
    loco_gain = np.asarray([_safe_float(row.get("locomotion_env_gain_vs_zero")) for row in rows])
    gain_scale = float(np.std(loco_gain)) if float(np.std(loco_gain)) > 1e-9 else 1.0
    gain_z = (loco_gain - float(np.mean(loco_gain))) / gain_scale
    c = -1.0 * gain_z

    result = milp(
        c=c,
        integrality=np.ones(n, dtype=np.int8),
        bounds=Bounds(np.zeros(n), np.ones(n)),
        constraints=LinearConstraint(np.vstack(a_rows), np.asarray(lb), np.asarray(ub)),
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"{rule}: MILP failed: {result.message}")

    selected_indices = [idx for idx, value in enumerate(result.x) if value >= 0.5]
    selected = [dict(rows[idx]) for idx in selected_indices]
    selected.sort(key=lambda row: int(row["_source_index"]))
    for rank, row in enumerate(selected):
        row["candidate_rule"] = rule
        row["joint_profile_rank"] = int(rank)
        row["joint_profile_guard"] = "locomotion_raised_profile_capped"
        row["joint_profile_cell_cap_rate"] = float(profile_cell_cap_rate)

    selected_summary = _summary(selected, rule=rule)
    for key, detail in band_details.items():
        count = int(selected_summary[f"{key}_count"])
        detail["selected_count"] = count
        detail["selected_rate"] = selected_summary[f"{key}_rate"]
        detail["passed"] = bool(detail["min_count"] <= count <= detail["max_count"])
    dominant_pass = selected_summary["joint_profile_dominant_count"] <= profile_cap

    return selected, {
        "solver_success": bool(result.success),
        "solver_message": str(result.message),
        "solver_fun": float(result.fun),
        "band_details": band_details,
        "profile_cell_cap_count": int(profile_cap),
        "profile_details": profile_details,
        "profile_guard_pass": bool(dominant_pass),
        "guard_pass": bool(dominant_pass and all(detail["passed"] for detail in band_details.values())),
    }


def _materialize(out_dir: Path, rule: str, rows: list[dict[str, Any]]) -> dict[str, str]:
    rule_dir = out_dir / _safe_label(rule) / "fixed_env_group"
    frozen_dir = rule_dir / "frozen_h"
    frozen_dir.mkdir(parents=True, exist_ok=True)
    h_list = []
    seeds = []
    for idx, row in enumerate(rows):
        src = Path(str(row["full_frozen_h_json"])).expanduser().resolve()
        h = _read_json(src)
        dst = frozen_dir / f"env_{idx:04d}_seed{int(float(row['env_seed']))}.json"
        shutil.copyfile(src, dst)
        h_list.append(h)
        seeds.append(int(float(row["env_seed"])))
    h_path = rule_dir / "prior_fixed_h_list.json"
    seed_path = rule_dir / "prior_fixed_env_seeds.json"
    h_path.write_text(json.dumps(_json_safe(h_list), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    seed_path.write_text(json.dumps(seeds, indent=2) + "\n", encoding="utf-8")
    return {"fixed_h_list_json": str(h_path), "fixed_env_seed_json": str(seed_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-csv", default=str(DEFAULT_SELECTED_CSV))
    parser.add_argument("--locomotion-csv", default=str(DEFAULT_LOCOMOTION_CSV))
    parser.add_argument("--action-mode-json", default=str(DEFAULT_ACTION_MODE_JSON))
    parser.add_argument(
        "--output-dir",
        default=str(ARTIFACT_ROOT / "phase2_joint_profile_guarded_locomotion_candidate_gpu0_0511"),
    )
    parser.add_argument("--target-per-rule", type=int, default=64)
    parser.add_argument("--protected-tolerance", type=float, default=0.035)
    parser.add_argument("--locomotion-min-rate", type=float, default=0.25)
    parser.add_argument("--locomotion-max-rate", type=float, default=0.3125)
    parser.add_argument("--profile-cell-cap-rate", type=float, default=0.25)
    args = parser.parse_args()

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    by_rule = _merge_rows(
        _read_csv(args.selected_csv),
        _load_locomotion_labels(args.locomotion_csv),
        _load_action_labels(args.action_mode_json),
    )

    report: dict[str, Any] = {
        "analysis_entry": "phase2_joint_profile_guarded_candidate_builder",
        "contract": {
            "exploratory_only": True,
            "does_not_sample_new_envs": True,
            "does_not_modify_exact_scm": True,
            "does_not_modify_fit_model": True,
            "does_not_modify_ppo": True,
            "selection_wrapper_only": True,
        },
        "config": {
            "target_per_rule": int(args.target_per_rule),
            "protected_tolerance": float(args.protected_tolerance),
            "locomotion_min_rate": float(args.locomotion_min_rate),
            "locomotion_max_rate": float(args.locomotion_max_rate),
            "profile_cell_cap_rate": float(args.profile_cell_cap_rate),
            "mode_keys": list(MODE_KEYS),
        },
        "rules": {},
    }
    all_selected: list[dict[str, Any]] = []
    for rule in RULES:
        selected, info = _select(
            by_rule[rule],
            rule=rule,
            target_n=int(args.target_per_rule),
            protected_tolerance=float(args.protected_tolerance),
            locomotion_min_rate=float(args.locomotion_min_rate),
            locomotion_max_rate=float(args.locomotion_max_rate),
            profile_cell_cap_rate=float(args.profile_cell_cap_rate),
        )
        all_selected.extend(selected)
        report["rules"][rule] = {
            "source_summary": _summary(by_rule[rule], rule=rule),
            "selected_summary": _summary(selected, rule=rule),
            **info,
            "fixed_group": _materialize(out_dir, rule, selected),
        }

    selected_csv = out_dir / "selected_joint_profile_guarded_prior_envs.csv"
    report_json = out_dir / "joint_profile_guarded_candidate_report.json"
    report_md = out_dir / "joint_profile_guarded_candidate.md"
    _write_csv(selected_csv, all_selected)
    report["outputs"] = {"output_dir": str(out_dir), "selected_csv": str(selected_csv)}
    report_json.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Joint Profile Guarded Candidate",
        "",
        "Exploratory selection wrapper only; no exact SCM / fit_model / PPO changes.",
        "",
        "| rule | guard | locomotion | low-energy | sign/phase | state-cond | dominant cell | active cells |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rule in RULES:
        payload = report["rules"][rule]
        s = payload["selected_summary"]
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{rule}`",
                    "pass" if payload["guard_pass"] else "fail",
                    f"{s['locomotion_witness_rate']:.3f}",
                    f"{s['low_energy_not_ctrl_only_rate']:.3f}",
                    f"{s['sign_or_phase_sensitive_rate']:.3f}",
                    f"{s['state_conditioned_energy_injection_rate']:.3f}",
                    f"{s['joint_profile_dominant_rate']:.3f}",
                    str(s["joint_profile_active_cells"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Read: this candidate raises locomotion only under a joint-profile cap and protected broad-mode bands.",
            "It does not repair Pendulum strict subprofile/gain domination or MountainCar upfront-investment; those need separate detectors/controls.",
            "",
            f"- JSON: `{report_json}`",
            f"- selected CSV: `{selected_csv}`",
        ]
    )
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_json), "summary": str(report_md), "selected_csv": str(selected_csv)}, indent=2))


if __name__ == "__main__":
    main()
