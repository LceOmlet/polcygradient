#!/usr/bin/env python
"""Summarize the v2e profile-mainline decision.

Exploratory/read-only.  This intentionally does not introduce a new selector or
quota.  It consolidates the current evidence for the next candidate:

* v2d source-pool definition repairs feedback profile collapse;
* near-best gain audit decides whether gain=2.0 is an argmax-hard artifact;
* integrated v2e smoke checks profile/diversity/delta guards;
* MountainCar remains prior-side detector/coverage guard only.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


ART = Path("/home/chen/RLPFN/artifacts")
DEFAULT_SELECTOR_JSON = (
    ART / "phase2_profile_band_selector_v2d_feedback_source_pool_repair_0511/profile_band_selector_report.json"
)
DEFAULT_NEAR_BEST_JSON = (
    ART / "phase2_feedback_near_best_gain_audit_v2d_0511/feedback_near_best_gain_audit.json"
)
DEFAULT_NEXT_PRESSURE_JSON = (
    ART / "phase2_profile_next_pressure_audit_v2d_prior_coverage_0511/profile_next_pressure_audit.json"
)
DEFAULT_SOURCE_POOL_JSON = (
    ART / "phase2_feedback_source_pool_broaden_audit_0511/feedback_source_pool_broaden_audit.json"
)
DEFAULT_SMOKE_JSON = (
    ART / "phase2_profile_v2e_candidate_smoke_summary_0511/profile_v2e_smoke_summary.json"
)
DEFAULT_OUTPUT_DIR = ART / "phase2_profile_v2e_decision_summary_0511"


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def _read_json_optional(path: str | Path | None) -> Any:
    if path is None or str(path).strip() == "":
        return None
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        return None
    return json.loads(resolved.read_text(encoding="utf-8"))


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    return value


def _selector_read(selector: dict[str, Any]) -> dict[str, Any]:
    rules = selector.get("rules", {})
    per_rule = {}
    guards = []
    for rule, payload in rules.items():
        s = payload.get("selected_summary", {})
        branch = payload.get("band_details", {}).get("strict_feedback_subprofile_branch", {})
        guards.append(bool(payload.get("guard_pass")))
        per_rule[rule] = {
            "guard_pass": bool(payload.get("guard_pass")),
            "strict_feedback_count": s.get("strict_feedback_count"),
            "strict_feedback_profile_counts": s.get("strict_feedback_profile_counts"),
            "strict_feedback_gain_counts": s.get("strict_feedback_gain_counts"),
            "selected_nonconstant_count": branch.get("selected_nonconstant_count"),
            "selected_non_gain2_count": branch.get("selected_non_gain2_count"),
            "joint_profile_dominant_rate": s.get("joint_profile_dominant_rate"),
            "joint_profile_entropy_norm": s.get("joint_profile_entropy_norm"),
        }
    return {
        "all_guards_pass": bool(guards and all(guards)),
        "per_rule": per_rule,
        "read": (
            "v2d is a valid source-pool definition repair probe: unified profile guards pass and "
            "feedback non-constant support is restored. Gain=2.0 remains unresolved."
        ),
    }


def _near_best_read(near_best: dict[str, Any]) -> dict[str, Any]:
    overall = near_best.get("overall_read", {})
    per_rule_profile = {}
    for rule, payload in near_best.get("rules", {}).items():
        per_rule_profile[rule] = {}
        for profile, item in payload.get("profiles", {}).items():
            per_rule_profile[rule][profile] = {
                "best_gain2_rate": item.get("read", {}).get("best_gain2_rate"),
                "strict_near_best_gain1_rate": item.get("strict_near_best_gain_1_rate"),
                "strict_near_best_gain05_rate": item.get("strict_near_best_gain_0.5_rate"),
                "verdict": item.get("read", {}).get("verdict"),
            }
    return {
        "overall": overall,
        "per_rule_profile": per_rule_profile,
        "read": (
            "near-best gain audit does not support treating gain=2.0 as a proven generator bias. "
            "full is argmax-hard; q90 is mixed. v2e should track near-best gain support rather than add a hard gain quota."
        ),
    }


def _mountaincar_read(next_pressure: dict[str, Any]) -> dict[str, Any]:
    mc = next_pressure.get("mountaincar_prior_fixedlist_guard", {})
    return {
        "guard_pass": bool(mc.get("guard_pass")),
        "scope_summaries": mc.get("scope_summaries", {}),
        "read": (
            "MountainCar future-basin stays detector/coverage only. It is not quota-ready until prior-side "
            "coverage is stable in both full and q90."
        ),
    }


def build_report(
    *,
    selector_json: str | Path,
    near_best_json: str | Path,
    next_pressure_json: str | Path,
    source_pool_json: str | Path,
    smoke_json: str | Path | None = None,
) -> dict[str, Any]:
    selector = _read_json(selector_json)
    near_best = _read_json(near_best_json)
    next_pressure = _read_json(next_pressure_json)
    source_pool = _read_json(source_pool_json)
    smoke = _read_json_optional(smoke_json)
    selector_read = _selector_read(selector)
    near_best_read = _near_best_read(near_best)
    mountaincar_read = _mountaincar_read(next_pressure)
    v2e_ready = bool(
        selector_read["all_guards_pass"]
        and source_pool.get("read", {}).get("temporal_union_repairs_profile") is True
        and near_best.get("overall_read", {}).get("any_source_pool_high_gain_bias") is False
    )
    smoke_read = {
        "available": bool(smoke),
        "guard_pass": bool((smoke or {}).get("overall", {}).get("candidate_smoke_guard_pass")),
        "profile_guard_pass": bool((smoke or {}).get("overall", {}).get("profile_guard_pass")),
        "diversity_guard_pass": bool((smoke or {}).get("overall", {}).get("diversity_guard_pass")),
        "trainability_delta_guard_pass": bool(
            (smoke or {}).get("overall", {}).get("trainability_delta_guard_pass")
        ),
        "read": (
            "integrated full/q90 smoke passed profile, diversity, and delta-mean guards"
            if bool((smoke or {}).get("overall", {}).get("candidate_smoke_guard_pass"))
            else "integrated full/q90 smoke is missing or has not passed all guards"
        ),
    }
    if smoke:
        smoke_read["scopes"] = {
            rule: {
                "raw_delta_mean": payload.get("smoke", {})
                .get("raw_reward_sum_delta", {})
                .get("mean"),
                "env_delta_mean": payload.get("smoke", {})
                .get("reward_env_sum_delta", {})
                .get("mean"),
                "diversity_guard_pass": payload.get("smoke", {})
                .get("diversity_guard", {})
                .get("pass"),
            }
            for rule, payload in (smoke.get("scopes") or {}).items()
        }
    return {
        "analysis_entry": "phase2_profile_v2e_decision_summary",
        "contract": {
            "exploratory_only": True,
            "read_only_existing_artifacts": True,
            "does_not_sample_environments": True,
            "does_not_train": True,
            "does_not_modify_exact_scm": True,
            "does_not_modify_fit_model": True,
            "does_not_modify_ppo": True,
            "profile_quality_first": True,
            "delta_sign_fraction_not_a_gate": True,
        },
        "inputs": {
            "selector_json": str(Path(selector_json).expanduser().resolve()),
            "near_best_json": str(Path(near_best_json).expanduser().resolve()),
            "next_pressure_json": str(Path(next_pressure_json).expanduser().resolve()),
            "source_pool_json": str(Path(source_pool_json).expanduser().resolve()),
            "smoke_json": None if smoke_json is None else str(Path(smoke_json).expanduser().resolve()),
        },
        "feedback_source_pool": source_pool.get("read", {}),
        "selector_v2d": selector_read,
        "feedback_near_best_gain": near_best_read,
        "mountaincar": mountaincar_read,
        "integrated_smoke": smoke_read,
        "v2e_candidate_read": {
            "profile_definition_close_enough_for_candidate": v2e_ready,
            "include_in_v2e": [
                "feedback strict source-pool definition = strict_or_time_profile_core_no_random",
                "near-best gain support metric as profile submetric/guard",
                "keep v2c centered protected full/q90 bands and joint-profile cap",
            ],
            "exclude_from_v2e_for_now": [
                "hard gain quota",
                "MountainCar quota",
                "any delta-sign-count objective",
            ],
            "remaining_before_milestone": [
                (
                    "integrated full/q90 smoke is complete"
                    if smoke_read["guard_pass"]
                    else "run integrated full/q90 smoke after v2e selector materialization"
                ),
                (
                    "state/reward/done/action diversity guard passed"
                    if smoke_read["diversity_guard_pass"]
                    else "ensure state/reward/done/action diversity is not degraded"
                ),
                "only then consider pack-to-fit_model parity migration",
            ],
        },
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Profile v2e Decision Summary",
        "",
        "Read-only consolidation of profile-mainline evidence.",
        "",
        "## Decision",
        "",
        f"- candidate-ready profile definition: `{report['v2e_candidate_read']['profile_definition_close_enough_for_candidate']}`",
        "- include:",
    ]
    for item in report["v2e_candidate_read"]["include_in_v2e"]:
        lines.append(f"  - {item}")
    lines.append("- exclude for now:")
    for item in report["v2e_candidate_read"]["exclude_from_v2e_for_now"]:
        lines.append(f"  - {item}")
    lines.extend(["", "## Feedback", ""])
    lines.append(f"- source-pool read: {report['selector_v2d']['read']}")
    lines.append(f"- near-best read: {report['feedback_near_best_gain']['read']}")
    lines.extend(["", "| rule | profile | best gain2 | strict near gain1 | verdict |", "|---|---|---:|---:|---|"])
    for rule, profiles in report["feedback_near_best_gain"]["per_rule_profile"].items():
        for profile, item in profiles.items():
            lines.append(
                f"| `{rule}` | `{profile}` | {item['best_gain2_rate']:.3f} | "
                f"{item['strict_near_best_gain1_rate']:.3f} | `{item['verdict']}` |"
            )
    lines.extend(["", "## MountainCar", ""])
    lines.append(f"- guard pass: `{report['mountaincar']['guard_pass']}`")
    lines.append(f"- read: {report['mountaincar']['read']}")
    lines.extend(["", "## Integrated Smoke", ""])
    lines.append(f"- available: `{report['integrated_smoke']['available']}`")
    lines.append(f"- guard pass: `{report['integrated_smoke']['guard_pass']}`")
    lines.append(f"- read: {report['integrated_smoke']['read']}")
    lines.extend(["", "## Remaining", ""])
    for item in report["v2e_candidate_read"]["remaining_before_milestone"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Files", ""])
    for key, value in report.get("outputs", {}).items():
        lines.append(f"- `{key}`: `{value}`")
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(
        selector_json=args.selector_json,
        near_best_json=args.near_best_json,
        next_pressure_json=args.next_pressure_json,
        source_pool_json=args.source_pool_json,
        smoke_json=args.smoke_json,
    )
    json_path = out_dir / "profile_v2e_decision_summary.json"
    md_path = out_dir / "profile_v2e_decision_summary.md"
    report["outputs"] = {"json": str(json_path), "markdown": str(md_path)}
    json_path.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report["outputs"], indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selector-json", default=str(DEFAULT_SELECTOR_JSON))
    parser.add_argument("--near-best-json", default=str(DEFAULT_NEAR_BEST_JSON))
    parser.add_argument("--next-pressure-json", default=str(DEFAULT_NEXT_PRESSURE_JSON))
    parser.add_argument("--source-pool-json", default=str(DEFAULT_SOURCE_POOL_JSON))
    parser.add_argument("--smoke-json", default=str(DEFAULT_SMOKE_JSON))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
