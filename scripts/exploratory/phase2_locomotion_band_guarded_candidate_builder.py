#!/usr/bin/env python
"""Build a band-guarded locomotion-temporal candidate fixed list.

Exploratory only.  This script reads an already-materialized milestone-2 source
pool and read-only mode labels, then selects a fixed list that increases the
locomotion-temporal witness count while keeping protected mode rates inside a
small band around the milestone-2 baseline.

It does not sample new environments, train PPO, mutate exact SCM defaults, or
modify fit_model/PPO code paths.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


DEFAULT_SELECTED_CSV = (
    "/home/chen/RLPFN/artifacts/phase2_exploratory_reward_group_balance_gated_n128_seed6262_0510/"
    "selected_prior_envs.csv"
)
DEFAULT_LOCOMOTION_CSV = (
    "/home/chen/RLPFN/artifacts/phase2_locomotion_full_temporal_detector_gated_n128_gpu0_0510/"
    "locomotion_full_temporal_per_env.csv"
)
DEFAULT_ACTION_MODE_JSON = (
    "/home/chen/RLPFN/artifacts/phase2_prior_action_mode_coverage_gated_n128_gpu0_0510/"
    "gated_n128_action_mode_coverage.json"
)
DEFAULT_OUTPUT_DIR = (
    "/home/chen/RLPFN/artifacts/phase2_locomotion_band_guarded_candidate_tol0025_gated_n128_gpu0_0511"
)

RULES = ("gym_full_obs_action_range", "gym_q90_obs_action_range")
PROTECTED_KEYS = (
    "low_energy_not_ctrl_only",
    "sign_or_phase_sensitive",
    "state_conditioned_energy_injection",
)

PROTECTED_BASELINES = {
    "gym_full_obs_action_range": {
        "low_energy_not_ctrl_only": 0.53125,
        "sign_or_phase_sensitive": 0.65625,
        "state_conditioned_energy_injection": 0.453125,
    },
    "gym_q90_obs_action_range": {
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
        return float(default)
    return out if math.isfinite(out) else float(default)


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).expanduser().resolve().open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def _safe_label(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")[:160] or "rule"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True)
                    if isinstance(value, (dict, list, tuple))
                    else value
                    for key, value in row.items()
                }
            )


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
            "locomotion_state_path_gain_vs_zero": _safe_float(row.get("state_path_gain_vs_zero")),
            "locomotion_action_abs_mean": _safe_float(row.get("action_abs_mean")),
            "locomotion_best_mode": row.get("best_sustained_mode"),
            "locomotion_terminal_invariant": _safe_bool(row.get("terminal_invariant")),
        }
    return labels


def _stats(values: list[Any]) -> dict[str, Any]:
    arr = np.asarray([_safe_float(v) for v in values], dtype=np.float64)
    arr = arr[np.isfinite(arr)]
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
    }


def _rate(rows: list[dict[str, Any]], key: str) -> float | None:
    if not rows:
        return None
    return float(sum(1 for row in rows if bool(row.get(key))) / len(rows))


def _summary(rows: list[dict[str, Any]], *, rule: str, source_n: int) -> dict[str, Any]:
    out: dict[str, Any] = {
        "rule": rule,
        "n": int(len(rows)),
        "source_n": int(source_n),
        "locomotion_witness_count": int(sum(1 for row in rows if row.get("locomotion_witness"))),
        "locomotion_witness_rate": _rate(rows, "locomotion_witness"),
        "locomotion_env_gain_vs_zero": _stats(
            [row.get("locomotion_env_gain_vs_zero") for row in rows]
        ),
        "obs_dim": _stats([row.get("obs_dim") for row in rows]),
        "action_dim": _stats([row.get("action_dim") for row in rows]),
        "state_dim": _stats([row.get("state_dim") for row in rows]),
        "noise_dim": _stats([row.get("noise_dim") for row in rows]),
    }
    for key in PROTECTED_KEYS:
        out[f"{key}_count"] = int(sum(1 for row in rows if bool(row.get(key))))
        out[f"{key}_rate"] = _rate(rows, key)
    return out


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


def _band_counts(rule: str, key: str, target_n: int, tolerance: float) -> tuple[int, int]:
    baseline = float(PROTECTED_BASELINES[rule][key])
    low = max(0, math.ceil((baseline - tolerance) * target_n - 1e-9))
    high = min(target_n, math.floor((baseline + tolerance) * target_n + 1e-9))
    return int(low), int(high)


def _milp_select(
    rows: list[dict[str, Any]],
    *,
    rule: str,
    target_n: int,
    tolerance: float,
    gain_weight: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    n = len(rows)
    if n < target_n:
        raise ValueError(f"{rule}: need at least {target_n} rows, got {n}")

    witness = np.asarray([1.0 if row.get("locomotion_witness") else 0.0 for row in rows])
    gain = np.asarray([_safe_float(row.get("locomotion_env_gain_vs_zero")) for row in rows])
    gain_scale = float(np.std(gain)) if float(np.std(gain)) > 1e-9 else 1.0
    gain_z = (gain - float(np.mean(gain))) / gain_scale

    c = -1000.0 * witness - float(gain_weight) * gain_z
    constraints = []
    constraints.append((np.ones(n, dtype=np.float64), target_n, target_n))

    band_details: dict[str, Any] = {}
    for key in PROTECTED_KEYS:
        indicator = np.asarray([1.0 if row.get(key) else 0.0 for row in rows], dtype=np.float64)
        low, high = _band_counts(rule, key, target_n, tolerance)
        constraints.append((indicator, low, high))
        band_details[key] = {
            "baseline": float(PROTECTED_BASELINES[rule][key]),
            "min_count": int(low),
            "max_count": int(high),
            "min_rate": float(low / target_n),
            "max_rate": float(high / target_n),
            "source_count": int(indicator.sum()),
            "source_rate": float(indicator.mean()) if n else None,
        }

    a = np.vstack([item[0] for item in constraints])
    lb = np.asarray([float(item[1]) for item in constraints], dtype=np.float64)
    ub = np.asarray([float(item[2]) for item in constraints], dtype=np.float64)
    result = milp(
        c=c,
        integrality=np.ones(n, dtype=np.int8),
        bounds=Bounds(np.zeros(n), np.ones(n)),
        constraints=LinearConstraint(a, lb, ub),
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"{rule}: MILP failed: {result.message}")

    selected_indices = [idx for idx, value in enumerate(result.x) if value >= 0.5]
    selected = [dict(rows[idx]) for idx in selected_indices]
    selected.sort(key=lambda row: int(row["_source_index"]))
    for rank, row in enumerate(selected):
        row["candidate_rule"] = rule
        row["band_guard_rank"] = int(rank)
        row["band_guard_tolerance"] = float(tolerance)

    selected_summary = _summary(selected, rule=rule, source_n=n)
    for key in PROTECTED_KEYS:
        count = int(selected_summary[f"{key}_count"])
        detail = band_details[key]
        detail["selected_count"] = count
        detail["selected_rate"] = selected_summary[f"{key}_rate"]
        detail["passed"] = bool(detail["min_count"] <= count <= detail["max_count"])

    return selected, {
        "solver_success": bool(result.success),
        "solver_message": str(result.message),
        "solver_fun": float(result.fun),
        "band_details": band_details,
        "guard_pass": bool(all(detail["passed"] for detail in band_details.values())),
        "selection_rule": (
            "maximize locomotion-temporal witness count; tie-break by locomotion gain; "
            "keep protected mode counts inside baseline +/- tolerance bands"
        ),
    }


def _materialize_fixed_group(out_dir: Path, rule: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    rule_dir = out_dir / _safe_label(rule) / "fixed_env_group"
    frozen_dir = rule_dir / "frozen_h"
    frozen_dir.mkdir(parents=True, exist_ok=True)
    h_list = []
    seeds = []
    copied = []
    for idx, row in enumerate(rows):
        src = Path(str(row["full_frozen_h_json"])).expanduser().resolve()
        h = _read_json(src)
        dst = frozen_dir / f"env_{idx:04d}_seed{int(float(row['env_seed']))}.json"
        shutil.copyfile(src, dst)
        h_list.append(h)
        seeds.append(int(float(row["env_seed"])))
        copied.append(str(dst))
    h_path = rule_dir / "prior_fixed_h_list.json"
    seed_path = rule_dir / "prior_fixed_env_seeds.json"
    h_path.write_text(json.dumps(_json_safe(h_list), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    seed_path.write_text(json.dumps(seeds, indent=2) + "\n", encoding="utf-8")
    return {
        "fixed_h_list_json": str(h_path),
        "fixed_env_seed_json": str(seed_path),
        "copied_frozen_h_dir": str(frozen_dir),
        "copied_sample": copied[:5],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-csv", default=DEFAULT_SELECTED_CSV)
    parser.add_argument("--locomotion-csv", default=DEFAULT_LOCOMOTION_CSV)
    parser.add_argument("--action-mode-json", default=DEFAULT_ACTION_MODE_JSON)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-per-rule", type=int, default=64)
    parser.add_argument("--protected-rate-tolerance", type=float, default=0.025)
    parser.add_argument("--gain-weight", type=float, default=0.01)
    args = parser.parse_args()

    selected_rows = _read_csv(args.selected_csv)
    by_rule = _merge_rows(
        selected_rows,
        _load_locomotion_labels(args.locomotion_csv),
        _load_action_labels(args.action_mode_json),
    )

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "analysis_entry": "phase2_locomotion_band_guarded_candidate_builder",
        "contract": {
            "exploratory_only": True,
            "does_not_sample_new_envs": True,
            "does_not_modify_exact_scm": True,
            "does_not_modify_fit_model": True,
            "does_not_modify_ppo": True,
            "selection_wrapper_only": True,
        },
        "inputs": {
            "selected_csv": str(Path(args.selected_csv).expanduser().resolve()),
            "locomotion_csv": str(Path(args.locomotion_csv).expanduser().resolve()),
            "action_mode_json": str(Path(args.action_mode_json).expanduser().resolve()),
        },
        "config": {
            "target_per_rule": int(args.target_per_rule),
            "protected_rate_tolerance": float(args.protected_rate_tolerance),
            "gain_weight": float(args.gain_weight),
            "protected_baselines": PROTECTED_BASELINES,
        },
        "rules": {},
    }

    all_selected: list[dict[str, Any]] = []
    for rule in RULES:
        rows = by_rule[rule]
        selected, guard_info = _milp_select(
            rows,
            rule=rule,
            target_n=int(args.target_per_rule),
            tolerance=float(args.protected_rate_tolerance),
            gain_weight=float(args.gain_weight),
        )
        all_selected.extend(selected)
        report["rules"][rule] = {
            "source_summary": _summary(rows, rule=rule, source_n=len(rows)),
            "selected_summary": _summary(selected, rule=rule, source_n=len(rows)),
            **guard_info,
            "fixed_group": _materialize_fixed_group(out_dir, rule, selected),
        }

    selected_csv = out_dir / "selected_locomotion_band_guarded_prior_envs.csv"
    _write_csv(selected_csv, all_selected)
    report["outputs"] = {
        "selected_csv": str(selected_csv),
        "output_dir": str(out_dir),
    }

    report_path = out_dir / "locomotion_band_guarded_candidate_report.json"
    report_path.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Locomotion Band-Guarded Candidate",
        "",
        "Exploratory selection wrapper only; no generator/PPO/fit_model changes.",
        "",
        f"protected rate tolerance: `{float(args.protected_rate_tolerance):.4f}`",
        "",
        "| rule | source locomotion | selected locomotion | guard | low-energy | sign/phase | state-conditioned |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for rule in RULES:
        payload = report["rules"][rule]
        source = payload["source_summary"]
        selected = payload["selected_summary"]

        def fmt(value: Any) -> str:
            return "na" if value is None else f"{float(value):.3f}"

        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{rule}`",
                    fmt(source["locomotion_witness_rate"]),
                    fmt(selected["locomotion_witness_rate"]),
                    "pass" if payload["guard_pass"] else "fail",
                    fmt(selected.get("low_energy_not_ctrl_only_rate")),
                    fmt(selected.get("sign_or_phase_sensitive_rate")),
                    fmt(selected.get("state_conditioned_energy_injection_rate")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Read",
            "",
            "- This is the minimal repair after the lower-bound guarded candidate: it preserves both lower and upper protected-mode rates.",
            "- The next check is a canonical sidecar PPO smoke for raw delta positive rate and state/reward/done diversity.",
            "",
            "## Files",
            "",
            f"- JSON: `{report_path}`",
            f"- selected CSV: `{selected_csv}`",
        ]
    )
    md_path = out_dir / "locomotion_band_guarded_candidate.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path), "summary": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
