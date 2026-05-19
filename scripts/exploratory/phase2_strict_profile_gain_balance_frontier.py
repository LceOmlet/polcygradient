#!/usr/bin/env python
"""Audit and materialize strict Pendulum-like profile/gain balance frontiers.

Exploratory only. This script consumes already materialized strict
Pendulum-like fixed-list candidates and produces:

* profile/gain/source/sign/feature summaries for the pool,
* hard upper bounds for selection-only profile/gain balancing,
* best-effort balanced fixed-list CSVs for follow-up pack smokes,
* a compact mode-coverage table from the existing Gym-important audit.

It does not sample new environments, does not train PPO, and does not modify
exact SCM, fit_model, PPO, or milestone defaults.
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


DEFAULT_INPUT_CSV = (
    "/home/chen/RLPFN/artifacts/phase2_pendulum_like_strict_cross_seed6260_6261_6262_all_candidates_0510/"
    "pendulum_like_signature_fixedlist.csv"
)
DEFAULT_COVERAGE_REPORT = (
    "/home/chen/RLPFN/artifacts/phase2_gym_important_mode_prior_coverage_audit_0510/"
    "gym_important_mode_prior_coverage_audit_report.json"
)
DEFAULT_OUTPUT_DIR = "/home/chen/RLPFN/artifacts/phase2_strict_profile_gain_balance_frontier_formal_0510"

MODE_RE = re.compile(r"^(?P<profile>early_|decay_)?obs_linear(?P<neg>_neg)?_(?P<feature>\d+)x(?P<gain>\d+)")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    return value


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _parse_mode(mode: Any) -> dict[str, Any]:
    text = "" if mode is None else str(mode)
    match = MODE_RE.match(text)
    if not match:
        return {
            "parsed_mode": text,
            "parsed_profile": "unknown" if text else "none",
            "parsed_gain": None,
            "parsed_sign": "unknown" if text else "none",
            "parsed_feature": None,
        }
    raw_profile = match.group("profile") or ""
    profile = "constant"
    if raw_profile == "early_":
        profile = "early_then_zero"
    elif raw_profile == "decay_":
        profile = "decay"
    return {
        "parsed_mode": text,
        "parsed_profile": profile,
        "parsed_gain": float(match.group("gain")) / 100.0,
        "parsed_sign": "neg" if match.group("neg") else "pos",
        "parsed_feature": int(match.group("feature")),
    }


def _read_rows(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).expanduser().resolve().open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parsed = _parse_mode(row.get("best_feedback_mode"))
            row.update(parsed)
            rows.append(row)
    return rows


def _counter(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts[str(row.get(key))] += 1
    return dict(counts)


def _stats(values: list[float]) -> dict[str, Any]:
    vals = sorted(float(v) for v in values if math.isfinite(float(v)))
    n = len(vals)
    if n == 0:
        return {"n": 0}

    def q(frac: float) -> float:
        if n == 1:
            return float(vals[0])
        pos = max(0.0, min(1.0, frac)) * float(n - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return float(vals[lo])
        alpha = pos - lo
        return float(vals[lo] * (1.0 - alpha) + vals[hi] * alpha)

    return {
        "n": int(n),
        "mean": float(sum(vals) / n),
        "min": float(vals[0]),
        "q50": q(0.5),
        "max": float(vals[-1]),
    }


def _rates(counts: dict[str, int], total: int) -> dict[str, float]:
    denom = max(1, int(total))
    return {str(k): float(int(v) / denom) for k, v in sorted(counts.items())}


def _dominance(counts: dict[str, int]) -> dict[str, Any]:
    total = int(sum(int(v) for v in counts.values()))
    if total <= 0:
        return {"status": "empty", "dominant": None, "dominant_share": None}
    dominant, count = max(counts.items(), key=lambda item: int(item[1]))
    share = float(int(count) / total)
    if len([v for v in counts.values() if int(v) > 0]) <= 1:
        status = "collapsed"
    elif share >= 0.8:
        status = "dominated"
    elif share >= 0.65:
        status = "skewed"
    else:
        status = "ok"
    return {"status": status, "dominant": str(dominant), "dominant_share": share}


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    profile_counts = _counter(rows, "parsed_profile")
    gain_counts = _counter(rows, "parsed_gain")
    sign_counts = _counter(rows, "parsed_sign")
    feature_counts = _counter(rows, "parsed_feature")
    mode_counts = _counter(rows, "parsed_mode")
    source_counts = _counter(rows, "source_tag")
    return {
        "n": int(n),
        "source_counts": source_counts,
        "profile_counts": profile_counts,
        "profile_rates": _rates(profile_counts, n),
        "profile_dominance": _dominance(profile_counts),
        "gain_counts": gain_counts,
        "gain_rates": _rates(gain_counts, n),
        "gain_dominance": _dominance(gain_counts),
        "sign_counts": sign_counts,
        "feature_counts": feature_counts,
        "score": _stats([_safe_float(row.get("signature_score")) for row in rows]),
        "top_modes": sorted(mode_counts.items(), key=lambda item: int(item[1]), reverse=True)[:12],
    }


def _frontier(rows: list[dict[str, Any]], sizes: list[int]) -> list[dict[str, Any]]:
    total_nonconstant = sum(1 for row in rows if row.get("parsed_profile") != "constant")
    total_non_gain2 = sum(1 for row in rows if str(row.get("parsed_gain")) != "2.0")
    total_joint = sum(
        1
        for row in rows
        if row.get("parsed_profile") != "constant" and str(row.get("parsed_gain")) != "2.0"
    )
    out = []
    for n in sizes:
        n = int(n)
        out.append(
            {
                "n": n,
                "max_nonconstant_count": int(min(total_nonconstant, n)),
                "max_nonconstant_rate": float(min(total_nonconstant, n) / max(1, n)),
                "max_non_gain2_count": int(min(total_non_gain2, n)),
                "max_non_gain2_rate": float(min(total_non_gain2, n) / max(1, n)),
                "max_joint_nonconstant_non_gain2_count": int(min(total_joint, n)),
                "max_joint_nonconstant_non_gain2_rate": float(min(total_joint, n) / max(1, n)),
            }
        )
    return out


def _scarcity_tier(row: dict[str, Any]) -> tuple[int, float]:
    nonconstant = row.get("parsed_profile") != "constant"
    non_gain2 = str(row.get("parsed_gain")) != "2.0"
    if nonconstant and non_gain2:
        tier = 0
    elif nonconstant:
        tier = 1
    elif non_gain2:
        tier = 2
    else:
        tier = 3
    return tier, -_safe_float(row.get("signature_score"))


def _round_robin_take(bucket: list[dict[str, Any]], selected: list[dict[str, Any]], limit: int) -> None:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sorted(bucket, key=lambda item: _scarcity_tier(item)):
        by_source[str(row.get("source_tag"))].append(row)
    selected_paths = {str(row.get("full_frozen_h_json")) for row in selected}
    source_counts = Counter(str(row.get("source_tag")) for row in selected)
    while len(selected) < limit:
        candidates = [
            name
            for name, values in by_source.items()
            if any(str(row.get("full_frozen_h_json")) not in selected_paths for row in values)
        ]
        if not candidates:
            return
        name = min(candidates, key=lambda item: (source_counts[item], item))
        picked = None
        for row in by_source[name]:
            path = str(row.get("full_frozen_h_json"))
            if path not in selected_paths:
                picked = row
                break
        if picked is None:
            return
        selected.append(picked)
        selected_paths.add(str(picked.get("full_frozen_h_json")))
        source_counts[name] += 1


def _best_effort_select(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    tiers: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        tiers[_scarcity_tier(row)[0]].append(row)
    for tier in sorted(tiers):
        _round_robin_take(tiers[tier], selected, int(limit))
        if len(selected) >= int(limit):
            break
    if len(selected) < int(limit):
        remaining = [row for row in rows if row not in selected]
        _round_robin_take(remaining, selected, int(limit))
    return selected[: int(limit)]


def _write_csv(path: Path, rows: list[dict[str, Any]], *, rule: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    base_fields = list(rows[0].keys()) if rows else []
    parsed_fields = ["parsed_profile", "parsed_gain", "parsed_sign", "parsed_feature", "parsed_mode"]
    fields: list[str] = []
    for key in ["rule", *base_fields, *parsed_fields]:
        if key not in fields:
            fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["rule"] = rule
            writer.writerow({key: out.get(key) for key in fields})


def _mode_coverage_table(coverage_report: str | None) -> list[dict[str, Any]]:
    if not coverage_report:
        return []
    path = Path(coverage_report).expanduser().resolve()
    if not path.exists():
        return []
    report = _load_json(path)
    prior = report.get("prior_coverage") or {}
    long_horizon = report.get("long_horizon_profile_gain") or {}
    strict_pool = ((report.get("strict_candidate_pool_balance") or {}).get("all_candidates_seed6260_6261_6262") or {})
    rows: list[dict[str, Any]] = []
    for scope in ("full", "q90"):
        cov = (prior.get(scope) or {}).get("coverage") or {}
        lh = long_horizon.get(scope) or {}
        rows.extend(
            [
                {
                    "scope": scope,
                    "mode": "low_energy_not_ctrl_only",
                    "count": cov.get("low_energy_not_ctrl_only_count"),
                    "n": cov.get("n_envs"),
                    "rate": cov.get("low_energy_not_ctrl_only_ratio"),
                    "read": "broadly present",
                },
                {
                    "scope": scope,
                    "mode": "sign_or_phase_sensitive",
                    "count": cov.get("sign_or_phase_sensitive_present_count"),
                    "n": cov.get("n_envs"),
                    "rate": cov.get("sign_or_phase_sensitive_present_ratio"),
                    "read": "broadly present",
                },
                {
                    "scope": scope,
                    "mode": "state_conditioned_energy_injection",
                    "count": cov.get("state_conditioned_energy_injection_witness_count"),
                    "n": cov.get("n_envs"),
                    "rate": cov.get("state_conditioned_energy_injection_witness_ratio"),
                    "read": "present but should keep monitoring",
                },
                {
                    "scope": scope,
                    "mode": "long_horizon_closed_loop_strict_primary",
                    "count": lh.get("primary_n"),
                    "n": cov.get("n_envs"),
                    "rate": (
                        float(lh.get("primary_n", 0)) / max(1, int(cov.get("n_envs", 0) or 0))
                        if cov.get("n_envs")
                        else None
                    ),
                    "read": "present, but internal profile/gain is dominated",
                },
            ]
        )
    if strict_pool:
        rows.append(
            {
                "scope": "strict_pool",
                "mode": "profile_constant_domination",
                "count": (strict_pool.get("profile_counts") or {}).get("constant"),
                "n": strict_pool.get("n"),
                "rate": (strict_pool.get("profile_rates") or {}).get("constant"),
                "read": "largest clear simplicity risk",
            }
        )
        rows.append(
            {
                "scope": "strict_pool",
                "mode": "gain_2.0_domination",
                "count": (strict_pool.get("gain_counts") or {}).get("2.0"),
                "n": strict_pool.get("n"),
                "rate": (strict_pool.get("gain_rates") or {}).get("2.0"),
                "read": "largest clear simplicity risk",
            }
        )
    return rows


def _write_mode_table(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        fields = ["scope", "mode", "count", "n", "rate", "read"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def _write_summary(path: Path, report: dict[str, Any]) -> None:
    pool = report["pool_summary"]
    lines = [
        "# Strict Profile/Gain Balance Frontier",
        "",
        "Exploratory only. This is a read/materialization audit over existing strict candidates.",
        "",
        "## Definitions",
        "",
        "- `profile=constant`: the finite probe's best feedback controller uses the same memoryless observation feedback at every step.",
        "- `profile=decay`: the same feedback form is tapered through time.",
        "- `profile=early_then_zero`: feedback is active in the prefix and zero in the suffix.",
        "- `gain`: the finite probe's feedback scale, parsed from modes such as `obs_linear_neg_0x200` -> gain 2.0.",
        "",
        "These labels describe the finite probe controller family, not a proof of true global optimality.",
        "",
        "## Pool",
        "",
        f"- n: {pool['n']}",
        f"- profile counts: `{pool['profile_counts']}`",
        f"- gain counts: `{pool['gain_counts']}`",
        f"- profile dominance: `{pool['profile_dominance']}`",
        f"- gain dominance: `{pool['gain_dominance']}`",
        "",
        "## Selection-Only Frontier",
        "",
    ]
    for row in report["frontier_upper_bound"]:
        lines.append(
            f"- n={row['n']}: nonconstant <= {row['max_nonconstant_count']} "
            f"({row['max_nonconstant_rate']:.3f}), non-gain2 <= {row['max_non_gain2_count']} "
            f"({row['max_non_gain2_rate']:.3f}), joint <= {row['max_joint_nonconstant_non_gain2_count']} "
            f"({row['max_joint_nonconstant_non_gain2_rate']:.3f})"
        )
    lines.extend(["", "## Generated Lists", ""])
    for item in report["generated_lists"]:
        summary = item["summary"]
        lines.extend(
            [
                f"- {item['name']}: `{item['csv']}`",
                f"  - profile counts: `{summary['profile_counts']}`",
                f"  - gain counts: `{summary['gain_counts']}`",
            ]
        )
    if report.get("mode_coverage_csv"):
        lines.extend(["", "## Mode Coverage Table", "", f"- `{report['mode_coverage_csv']}`"])
    lines.extend(["", "## Decision", "", report["decision"]["read"]])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", default=DEFAULT_INPUT_CSV)
    parser.add_argument("--coverage-report", default=DEFAULT_COVERAGE_REPORT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sizes", default="16,32,64,128")
    parser.add_argument("--materialize-size", action="append", type=int, default=[64, 32])
    args = parser.parse_args()

    rows = _read_rows(args.input_csv)
    if not rows:
        raise RuntimeError(f"No rows read from {args.input_csv}")
    sizes = [int(part) for part in str(args.sizes).split(",") if str(part).strip()]
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    generated: list[dict[str, Any]] = []
    for n in list(dict.fromkeys(int(v) for v in args.materialize_size)):
        selected = _best_effort_select(rows, n)
        name = f"strict_profile_gain_best_effort_n{n}"
        csv_path = out_dir / f"{name}.csv"
        _write_csv(csv_path, selected, rule=name)
        generated.append({"name": name, "n": n, "csv": str(csv_path), "summary": _summary(selected)})

    mode_rows = _mode_coverage_table(args.coverage_report)
    mode_csv = out_dir / "gym_important_mode_coverage_table.csv"
    _write_mode_table(mode_csv, mode_rows)

    report = {
        "analysis_entry": "phase2_strict_profile_gain_balance_frontier",
        "contract": {
            "exploratory_only": True,
            "does_not_modify_exact_scm": True,
            "does_not_modify_fit_model": True,
            "does_not_modify_ppo": True,
            "does_not_sample_new_envs": True,
            "does_not_train": True,
            "materializes_only_existing_fixed_h_entries": True,
        },
        "input_csv": str(Path(args.input_csv).expanduser().resolve()),
        "coverage_report": str(Path(args.coverage_report).expanduser().resolve()) if args.coverage_report else None,
        "pool_summary": _summary(rows),
        "frontier_upper_bound": _frontier(rows, sizes),
        "generated_lists": generated,
        "mode_coverage_csv": str(mode_csv) if mode_rows else None,
        "mode_coverage_table": mode_rows,
        "decision": {
            "selection_only_is_sufficient": False,
            "read": (
                "The strict candidate pool is already dominated before selection. "
                "Best-effort post-selection can preserve aggregate optimizability as a guard, "
                "but it cannot make the strict subfamily rich enough without either broadening "
                "the behavioral label or adding a narrow generator-level profile/gain control."
            ),
            "next_repair_scope": (
                "If accepted, implement an exploratory generator-level wrapper inside the strict "
                "Pendulum-like subfamily only; keep the milestone2 background unchanged and guard "
                "state/reward/done diversity plus trusted pack-runner post-pre improvement."
            ),
        },
    }
    report_path = out_dir / "strict_profile_gain_balance_frontier_report.json"
    summary_path = out_dir / "strict_profile_gain_balance_frontier.md"
    report_path.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_summary(summary_path, report)
    print(json.dumps({"report": str(report_path), "summary": str(summary_path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
