#!/usr/bin/env python
"""Build a guarded locomotion-temporal candidate fixed list.

Exploratory only.  This script consumes already materialized milestone-2
fixed-list candidates plus read-only mode probes, then selects a follow-up list
that raises sustained locomotion temporal witnesses while preserving the
currently protected Gym-like mode labels.

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
from collections import Counter
from pathlib import Path
from typing import Any


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
    "/home/chen/RLPFN/artifacts/phase2_locomotion_guarded_candidate_gated_n128_gpu0_0510"
)


RULES = ("gym_full_obs_action_range", "gym_q90_obs_action_range")

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
    p = Path(path).expanduser()
    if not p.exists():
        return {}
    report = _read_json(p)
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
    rows = _read_csv(path)
    labels: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
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


def _stats(values: list[float]) -> dict[str, Any]:
    vals = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not vals:
        return {"n": 0}
    n = len(vals)

    def q(frac: float) -> float:
        if n == 1:
            return vals[0]
        pos = max(0.0, min(1.0, float(frac))) * float(n - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return vals[lo]
        alpha = pos - lo
        return vals[lo] * (1.0 - alpha) + vals[hi] * alpha

    return {
        "n": int(n),
        "mean": float(sum(vals) / n),
        "min": float(vals[0]),
        "q10": float(q(0.10)),
        "q50": float(q(0.50)),
        "q90": float(q(0.90)),
        "max": float(vals[-1]),
    }


def _rate(rows: list[dict[str, Any]], key: str) -> float | None:
    valid = [row for row in rows if key in row and row[key] is not None]
    if not valid:
        return None
    return float(sum(1 for row in valid if bool(row[key])) / max(1, len(valid)))


def _summary(rows: list[dict[str, Any]], *, rule: str, source_n: int) -> dict[str, Any]:
    out = {
        "rule": rule,
        "n": int(len(rows)),
        "source_n": int(source_n),
        "locomotion_witness_count": int(sum(1 for row in rows if row.get("locomotion_witness"))),
        "locomotion_witness_rate": _rate(rows, "locomotion_witness"),
        "locomotion_env_gain_vs_zero": _stats(
            [_safe_float(row.get("locomotion_env_gain_vs_zero")) for row in rows]
        ),
        "selected_repair_kind_counts": dict(Counter(str(row.get("selected_repair_kind")) for row in rows)),
        "balance_profile_counts": dict(Counter(str(row.get("balance_profile")) for row in rows)),
        "obs_dim": _stats([_safe_float(row.get("obs_dim")) for row in rows]),
        "action_dim": _stats([_safe_float(row.get("action_dim")) for row in rows]),
        "state_dim": _stats([_safe_float(row.get("state_dim")) for row in rows]),
        "noise_dim": _stats([_safe_float(row.get("noise_dim")) for row in rows]),
    }
    for key in PROTECTED_BASELINES[rule]:
        out[f"{key}_rate"] = _rate(rows, key)
    return out


def _guard_min(rule: str, key: str, tolerance: float) -> float:
    baseline = float(PROTECTED_BASELINES[rule][key])
    return max(0.0, baseline - float(tolerance))


def _guard_ok(rows: list[dict[str, Any]], *, rule: str, tolerance: float) -> tuple[bool, dict[str, Any]]:
    details: dict[str, Any] = {}
    ok = True
    for key in PROTECTED_BASELINES[rule]:
        rate = _rate(rows, key)
        minimum = _guard_min(rule, key, tolerance)
        available = rate is not None
        passed = bool(available and float(rate) >= minimum)
        details[key] = {
            "rate": rate,
            "baseline": float(PROTECTED_BASELINES[rule][key]),
            "minimum": float(minimum),
            "passed": passed,
            "available": available,
        }
        ok = ok and passed
    return ok, details


def _score_row(row: dict[str, Any]) -> tuple[float, float, float, float]:
    protected = sum(
        1.0
        for key in ("low_energy_not_ctrl_only", "sign_or_phase_sensitive", "state_conditioned_energy_injection")
        if bool(row.get(key))
    )
    return (
        1.0 if bool(row.get("locomotion_witness")) else 0.0,
        protected,
        _safe_float(row.get("locomotion_env_gain_vs_zero")),
        -float(int(row.get("_source_index", 0))),
    )


def _select_guarded(
    rows: list[dict[str, Any]],
    *,
    rule: str,
    target_n: int,
    tolerance: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(rows) < int(target_n):
        raise ValueError(f"{rule}: need at least {target_n} rows, got {len(rows)}")

    sorted_rows = sorted(rows, key=_score_row, reverse=True)
    selected = sorted_rows[: int(target_n)]
    ok, guard = _guard_ok(selected, rule=rule, tolerance=tolerance)

    # If the witness-maximizing selection violates a protected rate, swap the
    # lowest-scoring selected rows for unselected rows carrying the missing mode.
    selected_ids = {int(row["_source_index"]) for row in selected}
    for _ in range(max(1, len(rows) * len(PROTECTED_BASELINES[rule]))):
        ok, guard = _guard_ok(selected, rule=rule, tolerance=tolerance)
        if ok:
            break
        failing = [
            key
            for key, payload in guard.items()
            if bool(payload.get("available")) and not bool(payload.get("passed"))
        ]
        if not failing:
            break
        key = failing[0]
        selected_false = [
            row
            for row in selected
            if not bool(row.get(key))
        ]
        pool_true = [
            row
            for row in sorted_rows
            if int(row["_source_index"]) not in selected_ids and bool(row.get(key))
        ]
        if not selected_false or not pool_true:
            break
        drop = sorted(selected_false, key=_score_row)[0]
        add = sorted(pool_true, key=_score_row, reverse=True)[0]
        selected = [row for row in selected if int(row["_source_index"]) != int(drop["_source_index"])]
        selected.append(add)
        selected_ids.remove(int(drop["_source_index"]))
        selected_ids.add(int(add["_source_index"]))

    selected = sorted(selected, key=lambda row: int(row["_source_index"]))
    ok, guard = _guard_ok(selected, rule=rule, tolerance=tolerance)
    return selected, {
        "guard_pass": bool(ok),
        "guard_details": guard,
        "selection_rule": (
            "maximize locomotion witness count, then protected-label coverage and locomotion gain; "
            "repair by swaps until protected rates pass"
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
    parser.add_argument("--protected-rate-tolerance", type=float, default=0.05)
    args = parser.parse_args()

    selected_rows = _read_csv(args.selected_csv)
    loco_labels = _load_locomotion_labels(args.locomotion_csv)
    action_labels = _load_action_labels(args.action_mode_json)
    action_labels_available = bool(action_labels)

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

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    all_selected: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "analysis_entry": "phase2_locomotion_guarded_candidate_builder",
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
            "action_labels_available": bool(action_labels_available),
        },
        "config": {
            "target_per_rule": int(args.target_per_rule),
            "protected_rate_tolerance": float(args.protected_rate_tolerance),
            "protected_baselines": PROTECTED_BASELINES,
        },
        "rules": {},
    }

    for rule in RULES:
        rows = by_rule[rule]
        if not action_labels_available:
            selected = sorted(rows, key=_score_row, reverse=True)[: int(args.target_per_rule)]
            selected = sorted(selected, key=lambda row: int(row["_source_index"]))
            guard_info = {
                "guard_pass": False,
                "guard_details": {
                    key: {
                        "available": False,
                        "baseline": float(value),
                        "minimum": _guard_min(rule, key, float(args.protected_rate_tolerance)),
                    }
                    for key, value in PROTECTED_BASELINES[rule].items()
                },
                "selection_rule": "locomotion-only provisional; protected action labels missing",
            }
        else:
            selected, guard_info = _select_guarded(
                rows,
                rule=rule,
                target_n=int(args.target_per_rule),
                tolerance=float(args.protected_rate_tolerance),
            )
        materialized = _materialize_fixed_group(out_dir, rule, selected)
        for row in selected:
            out_row = dict(row)
            out_row["candidate_rule"] = rule
            all_selected.append(out_row)
        report["rules"][rule] = {
            "source_summary": _summary(rows, rule=rule, source_n=len(rows)),
            "selected_summary": _summary(selected, rule=rule, source_n=len(rows)),
            **guard_info,
            "fixed_group": materialized,
        }

    selected_csv = out_dir / "selected_locomotion_guarded_prior_envs.csv"
    _write_csv(selected_csv, all_selected)
    report["outputs"] = {
        "selected_csv": str(selected_csv),
        "output_dir": str(out_dir),
    }
    report_path = out_dir / "locomotion_guarded_candidate_report.json"
    report_path.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Locomotion Guarded Candidate",
        "",
        "Exploratory selection wrapper only; no generator/PPO/fit_model changes.",
        "",
        f"action labels available: `{action_labels_available}`",
        "",
        "| rule | source locomotion | selected locomotion | guard | low-energy | sign/phase | state-conditioned |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for rule in RULES:
        payload = report["rules"][rule]
        source = payload["source_summary"]
        selected = payload["selected_summary"]
        guard = payload["guard_details"]

        def fmt(value: Any) -> str:
            return "na" if value is None else f"{float(value):.3f}"

        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{rule}`",
                    fmt(source["locomotion_witness_rate"]),
                    fmt(selected["locomotion_witness_rate"]),
                    "pass" if payload["guard_pass"] else "fail_or_unchecked",
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
            "## Next",
            "",
            "- If guard is unchecked, run the action-mode coverage probe for this n128 pool.",
            "- If guard passes, run sidecar short PPO smoke on the materialized fixed groups.",
        ]
    )
    md_path = out_dir / "locomotion_guarded_candidate.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path), "summary": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
