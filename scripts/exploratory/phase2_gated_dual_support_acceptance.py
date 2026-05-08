#!/usr/bin/env python
"""Read-only acceptance audit for gated full/q90 prior candidates.

This script is intentionally small and non-invasive. It does not sample
environments, does not run PPO, and does not modify the exact-SCM/default prior
path. It only checks that the same exploratory gated reward-path wrapper has
already produced both support buckets:

* gym_full_obs_action_range
* gym_q90_obs_action_range

and that both buckets passed the existing topology/diversity guard plus a short
trusted-pack rollout smoke.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_RUN_SPECS = [
    {
        "name": "seed6260",
        "probe_dir": "/home/chen/RLPFN/artifacts/phase2_exploratory_reward_group_balance_gated_n64_0506",
        "guard_report": "/home/chen/RLPFN/artifacts/phase2_exploratory_gated_fixedlist_guard_audit_0506/gated_fixedlist_guard_audit_report.json",
        "full_progress": "/home/chen/RLPFN/artifacts/phase2_exploratory_reward_group_balance_gated_full_fixedlist_n64_s512_u4_e4_advnorm_0506/prior_progress.jsonl",
        "q90_progress": "/home/chen/RLPFN/artifacts/phase2_exploratory_reward_group_balance_gated_q90_fixedlist_n64_s512_u4_e4_advnorm_0506/prior_progress.jsonl",
    },
    {
        "name": "seed6261",
        "probe_dir": "/home/chen/RLPFN/artifacts/phase2_exploratory_reward_group_balance_gated_n64_seed6261_0506",
        "guard_report": "/home/chen/RLPFN/artifacts/phase2_exploratory_gated_fixedlist_guard_audit_seed6261_0506/gated_fixedlist_guard_audit_report.json",
        "full_progress": "/home/chen/RLPFN/artifacts/phase2_exploratory_reward_group_balance_gated_full_fixedlist_seed6261_n64_s512_u4_e4_advnorm_0506/prior_progress.jsonl",
        "q90_progress": "/home/chen/RLPFN/artifacts/phase2_exploratory_reward_group_balance_gated_q90_fixedlist_seed6261_n64_s512_u4_e4_advnorm_0506/prior_progress.jsonl",
    },
]

REQUIRED_RULES = ("gym_full_obs_action_range", "gym_q90_obs_action_range")


def _load_json(path: str | Path) -> dict[str, Any]:
    p = Path(path).expanduser().resolve()
    return json.loads(p.read_text(encoding="utf-8"))


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path).expanduser().resolve()
    rows: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def _progress_smoke(path: str | Path) -> dict[str, Any]:
    rows = _load_jsonl(path)
    if not rows:
        return {"hard_pass": False, "reason": "empty progress jsonl"}

    def stat(row: dict[str, Any], key: str) -> float | None:
        return _finite_float(((row.get("pack_summary") or {}).get("stats") or {}).get(key))

    first = rows[0]
    last = rows[-1]
    first_return = stat(first, "raw_reward_return_mean")
    last_return = stat(last, "raw_reward_return_mean")
    done_counts = [
        value
        for value in (stat(row, "done_count") for row in rows)
        if value is not None
    ]
    delta = None if first_return is None or last_return is None else last_return - first_return
    return {
        "n_rows": int(len(rows)),
        "first_raw_reward_return_mean": first_return,
        "last_raw_reward_return_mean": last_return,
        "raw_reward_return_mean_delta": delta,
        "done_count_min": min(done_counts) if done_counts else None,
        "done_count_max": max(done_counts) if done_counts else None,
        "path": str(Path(path).expanduser().resolve()),
    }


def _rule_guard_ok(guard: dict[str, Any], rule: str) -> bool:
    item = ((guard.get("rule_reports") or {}).get(rule) or {})
    return bool(
        item.get("hard_pass")
        and item.get("topology_pass")
        and item.get("diversity_pass")
        and item.get("support_pass")
    )


def _audit_run(spec: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    probe_dir = Path(spec["probe_dir"]).expanduser().resolve()
    probe = _load_json(probe_dir / "reward_group_balance_probe_report.json")
    guard = _load_json(spec["guard_report"])
    full_smoke = _progress_smoke(spec["full_progress"])
    q90_smoke = _progress_smoke(spec["q90_progress"])

    selected_counts = probe.get("selected_count_by_rule") or {}
    rule_count_ok = {
        rule: int(selected_counts.get(rule, 0)) >= int(args.expected_n)
        for rule in REQUIRED_RULES
    }
    guard_ok = {rule: _rule_guard_ok(guard, rule) for rule in REQUIRED_RULES}

    contract = probe.get("contract") or {}
    contract_ok = bool(
        probe.get("exploratory_only")
        and contract.get("does_not_modify_exact_scm")
        and contract.get("does_not_modify_default_prior_generator")
        and contract.get("dimension_features_are_filter_coordinates_not_generator_knobs")
    )

    def smoke_ok(smoke: dict[str, Any]) -> bool:
        delta = _finite_float(smoke.get("raw_reward_return_mean_delta"))
        done_min = _finite_float(smoke.get("done_count_min"))
        return bool(
            smoke.get("n_rows", 0) >= int(args.min_smoke_rows)
            and delta is not None
            and delta >= float(args.min_smoke_delta)
            and done_min is not None
            and done_min >= float(args.min_done_count)
        )

    smokes = {
        "gym_full_obs_action_range": full_smoke,
        "gym_q90_obs_action_range": q90_smoke,
    }
    smoke_ok_by_rule = {rule: smoke_ok(smokes[rule]) for rule in REQUIRED_RULES}

    hard_pass = bool(
        contract_ok
        and bool(guard.get("hard_pass"))
        and all(rule_count_ok.values())
        and all(guard_ok.values())
        and all(smoke_ok_by_rule.values())
    )
    return {
        "name": str(spec.get("name", probe_dir.name)),
        "hard_pass": hard_pass,
        "probe_dir": str(probe_dir),
        "guard_report": str(Path(spec["guard_report"]).expanduser().resolve()),
        "contract_ok": contract_ok,
        "guard_hard_pass": bool(guard.get("hard_pass")),
        "selected_count_by_rule": selected_counts,
        "rule_count_ok": rule_count_ok,
        "guard_ok_by_rule": guard_ok,
        "smoke_ok_by_rule": smoke_ok_by_rule,
        "smoke_by_rule": smokes,
        "balance_profile": probe.get("balance_profile"),
        "health_criteria": contract.get("health_criteria"),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.run_specs_json:
        specs = _load_json(args.run_specs_json)
        if isinstance(specs, dict):
            specs = specs.get("runs", [])
    else:
        specs = DEFAULT_RUN_SPECS
    runs = [_audit_run(spec, args) for spec in specs]
    profiles = {json.dumps(run.get("balance_profile"), sort_keys=True) for run in runs}
    criteria = {json.dumps(run.get("health_criteria"), sort_keys=True) for run in runs}
    hard_pass = bool(runs and all(run["hard_pass"] for run in runs) and len(profiles) == 1 and len(criteria) == 1)
    report = {
        "analysis_entry": "phase2_gated_dual_support_acceptance",
        "exploratory_only": True,
        "hard_pass": hard_pass,
        "contract": {
            "does_not_modify_exact_scm": True,
            "does_not_modify_default_prior_generator": True,
            "read_only_existing_artifacts": True,
            "single_reward_path_balance_profile_across_full_and_q90": len(profiles) == 1,
            "single_health_criteria_across_full_and_q90": len(criteria) == 1,
            "required_rules": list(REQUIRED_RULES),
        },
        "thresholds": {
            "expected_n": int(args.expected_n),
            "min_smoke_rows": int(args.min_smoke_rows),
            "min_smoke_delta": float(args.min_smoke_delta),
            "min_done_count": float(args.min_done_count),
        },
        "runs": runs,
    }
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "gated_dual_support_acceptance_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    md_lines = [
        "# Gated Dual-Support Acceptance",
        "",
        f"hard_pass: {hard_pass}",
        "",
        "## Runs",
    ]
    for item in runs:
        full = item["smoke_by_rule"]["gym_full_obs_action_range"]
        q90 = item["smoke_by_rule"]["gym_q90_obs_action_range"]
        md_lines.append(
            f"- {item['name']}: hard={item['hard_pass']}, guard={item['guard_hard_pass']}, "
            f"full_delta={full.get('raw_reward_return_mean_delta')}, "
            f"q90_delta={q90.get('raw_reward_return_mean_delta')}"
        )
    md_lines.extend(
        [
            "",
            "Decision rule: keep this as an exploratory candidate only when every run has both full and q90 selected, guarded, and smoke-positive under the same balance profile and health criteria.",
        ]
    )
    md_path = out_dir / "gated_dual_support_acceptance_summary.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(json.dumps({"hard_pass": hard_pass, "report": str(report_path)}, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/chen/RLPFN/artifacts/phase2_exploratory_gated_dual_support_acceptance_0506",
    )
    parser.add_argument("--run-specs-json", type=str, default="")
    parser.add_argument("--expected-n", type=int, default=64)
    parser.add_argument("--min-smoke-rows", type=int, default=4)
    parser.add_argument("--min-smoke-delta", type=float, default=0.0)
    parser.add_argument("--min-done-count", type=float, default=1.0)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
