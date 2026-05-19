#!/usr/bin/env python
"""Build a mode-target band-guarded candidate fixed list.

Exploratory only.  This is a small generalization of the locomotion band guard:
choose from an already-materialized milestone-2 source pool, maximize one
explicit mode target, and keep all non-target protected modes inside narrow
baseline bands.

No new environments are sampled here, and no exact SCM, fit_model, or PPO path
is modified.
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

RULES = ("gym_full_obs_action_range", "gym_q90_obs_action_range")
MODE_KEYS = (
    "locomotion_witness",
    "low_energy_not_ctrl_only",
    "sign_or_phase_sensitive",
    "state_conditioned_energy_injection",
)

BASELINES = {
    "gym_full_obs_action_range": {
        "locomotion_witness": 0.1484375,
        "low_energy_not_ctrl_only": 0.53125,
        "sign_or_phase_sensitive": 0.65625,
        "state_conditioned_energy_injection": 0.453125,
    },
    "gym_q90_obs_action_range": {
        "locomotion_witness": 0.1484375,
        "low_energy_not_ctrl_only": 0.421875,
        "sign_or_phase_sensitive": 0.671875,
        "state_conditioned_energy_injection": 0.578125,
    },
}

TARGETS = {
    "energy": ("low_energy_not_ctrl_only",),
    "feedback": ("sign_or_phase_sensitive", "state_conditioned_energy_injection"),
    "locomotion_energy": ("locomotion_witness", "low_energy_not_ctrl_only"),
    "state_conditioned": ("state_conditioned_energy_injection",),
    "sign_phase": ("sign_or_phase_sensitive",),
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
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


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


def _rate(rows: list[dict[str, Any]], key: str) -> float:
    return float(sum(bool(row.get(key)) for row in rows) / max(1, len(rows)))


def _summary(rows: list[dict[str, Any]], *, rule: str) -> dict[str, Any]:
    out: dict[str, Any] = {"rule": rule, "n": int(len(rows))}
    for key in MODE_KEYS:
        out[f"{key}_count"] = int(sum(bool(row.get(key)) for row in rows))
        out[f"{key}_rate"] = _rate(rows, key)
    return out


def _band_counts(rule: str, key: str, target_n: int, tolerance: float) -> tuple[int, int]:
    baseline = float(BASELINES[rule][key])
    low = max(0, math.ceil((baseline - tolerance) * target_n - 1e-9))
    high = min(target_n, math.floor((baseline + tolerance) * target_n + 1e-9))
    return int(low), int(high)


def _select(
    rows: list[dict[str, Any]],
    *,
    rule: str,
    target_keys: tuple[str, ...],
    target_n: int,
    tolerance: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    n = len(rows)
    if n < target_n:
        raise ValueError(f"{rule}: need at least {target_n} rows, got {n}")
    target = np.asarray(
        [sum(1.0 for key in target_keys if bool(row.get(key))) for row in rows],
        dtype=np.float64,
    )
    loco_gain = np.asarray([_safe_float(row.get("locomotion_env_gain_vs_zero")) for row in rows])
    gain_scale = float(np.std(loco_gain)) if float(np.std(loco_gain)) > 1e-9 else 1.0
    gain_z = (loco_gain - float(np.mean(loco_gain))) / gain_scale
    c = -1000.0 * target - 0.01 * gain_z

    a_rows = [np.ones(n, dtype=np.float64)]
    lb = [float(target_n)]
    ub = [float(target_n)]
    band_details: dict[str, Any] = {}
    for key in MODE_KEYS:
        if key in target_keys:
            continue
        indicator = np.asarray([1.0 if bool(row.get(key)) else 0.0 for row in rows])
        low, high = _band_counts(rule, key, target_n, tolerance)
        a_rows.append(indicator)
        lb.append(float(low))
        ub.append(float(high))
        band_details[key] = {
            "baseline": float(BASELINES[rule][key]),
            "min_count": int(low),
            "max_count": int(high),
            "source_count": int(indicator.sum()),
            "source_rate": float(indicator.mean()),
        }

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
        row["mode_target_rank"] = int(rank)
        row["mode_target_keys"] = ",".join(target_keys)
        row["mode_target_guard_tolerance"] = float(tolerance)

    selected_summary = _summary(selected, rule=rule)
    for key, detail in band_details.items():
        count = int(selected_summary[f"{key}_count"])
        detail["selected_count"] = count
        detail["selected_rate"] = selected_summary[f"{key}_rate"]
        detail["passed"] = bool(detail["min_count"] <= count <= detail["max_count"])

    return selected, {
        "solver_success": bool(result.success),
        "solver_message": str(result.message),
        "solver_fun": float(result.fun),
        "band_details": band_details,
        "guard_pass": bool(all(detail["passed"] for detail in band_details.values())),
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
    parser.add_argument("--target", choices=sorted(TARGETS), required=True)
    parser.add_argument("--selected-csv", default=DEFAULT_SELECTED_CSV)
    parser.add_argument("--locomotion-csv", default=DEFAULT_LOCOMOTION_CSV)
    parser.add_argument("--action-mode-json", default=DEFAULT_ACTION_MODE_JSON)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-per-rule", type=int, default=64)
    parser.add_argument("--guard-tolerance", type=float, default=0.025)
    args = parser.parse_args()

    target_keys = TARGETS[str(args.target)]
    by_rule = _merge_rows(
        _read_csv(args.selected_csv),
        _load_locomotion_labels(args.locomotion_csv),
        _load_action_labels(args.action_mode_json),
    )
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "analysis_entry": "phase2_mode_target_band_guarded_candidate_builder",
        "contract": {
            "exploratory_only": True,
            "does_not_sample_new_envs": True,
            "does_not_modify_exact_scm": True,
            "does_not_modify_fit_model": True,
            "does_not_modify_ppo": True,
            "selection_wrapper_only": True,
        },
        "target": str(args.target),
        "target_keys": list(target_keys),
        "guard_tolerance": float(args.guard_tolerance),
        "rules": {},
    }
    all_selected: list[dict[str, Any]] = []
    for rule in RULES:
        rows = by_rule[rule]
        selected, info = _select(
            rows,
            rule=rule,
            target_keys=target_keys,
            target_n=int(args.target_per_rule),
            tolerance=float(args.guard_tolerance),
        )
        all_selected.extend(selected)
        report["rules"][rule] = {
            "source_summary": _summary(rows, rule=rule),
            "selected_summary": _summary(selected, rule=rule),
            **info,
            "fixed_group": _materialize(out_dir, rule, selected),
        }
    selected_csv = out_dir / f"selected_{args.target}_band_guarded_prior_envs.csv"
    _write_csv(selected_csv, all_selected)
    report["outputs"] = {"selected_csv": str(selected_csv), "output_dir": str(out_dir)}
    report_path = out_dir / f"{args.target}_band_guarded_candidate_report.json"
    report_path.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        f"# {args.target} Band-Guarded Candidate",
        "",
        "Exploratory selection wrapper only; no generator/PPO/fit_model changes.",
        "",
        f"target keys: `{','.join(target_keys)}`",
        "",
        "| rule | guard | locomotion | low-energy | sign/phase | state-conditioned |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
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
                    f"{float(s['locomotion_witness_rate']):.3f}",
                    f"{float(s['low_energy_not_ctrl_only_rate']):.3f}",
                    f"{float(s['sign_or_phase_sensitive_rate']):.3f}",
                    f"{float(s['state_conditioned_energy_injection_rate']):.3f}",
                ]
            )
            + " |"
        )
    lines.extend(["", f"- JSON: `{report_path}`", f"- selected CSV: `{selected_csv}`"])
    md_path = out_dir / f"{args.target}_band_guarded_candidate.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path), "summary": str(md_path), "selected_csv": str(selected_csv)}, indent=2))


if __name__ == "__main__":
    main()
