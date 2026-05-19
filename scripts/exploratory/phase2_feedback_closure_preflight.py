#!/usr/bin/env python
"""Preflight the feedback closed-loop settle closure step.

Read-only consolidation.  This script checks whether the highest-priority
coverage milestone, context_identifiable_closed_loop_settle_feedback, is ready
to be called closed.  It intentionally distinguishes:

* branch/selector repair evidence,
* short sidecar health evidence,
* fit_model/pack parity preservation,
* actual Gym-style closed-loop settle anchors.

The expected outcome today is usually "not closed": that is useful because it
prevents over-crediting a feedback selector repair as semantic sufficiency.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


ART = Path("/home/chen/RLPFN/artifacts")
DEFAULT_OUTPUT_DIR = ART / "phase2_feedback_closure_preflight_0511"
DEFAULT_COVERAGE_JSON = (
    ART / "phase2_profile_coverage_closure_forecast_0511/profile_coverage_closure_forecast.json"
)
DEFAULT_SUFFICIENCY_JSON = (
    ART / "phase2_profile_fit_model_sufficiency_audit_0511/profile_fit_model_sufficiency_audit.json"
)
DEFAULT_BRANCH_JSON = (
    ART / "phase2_feedback_strict_subprofile_branch_v1_n32_0511/feedback_strict_subprofile_branch_report.json"
)
DEFAULT_BRANCH_SMOKE_JSON = (
    ART
    / "phase2_feedback_strict_subprofile_branch_v1_n32_sidecar_summary_seed9512_0511"
    / "pendulum_like_sidecar_delta_summary_report.json"
)
DEFAULT_BUDGETED_JSON = (
    ART / "phase2_feedback_support_slack_audit_0511/feedback_support_slack_audit.json"
)
DEFAULT_PENDULUM_JSON = (
    ART / "phase2_pendulum_settle_anchor_preflight_0511/pendulum_settle_anchor_preflight.json"
)
DEFAULT_REACHER_ANCHOR_JSON = (
    ART / "phase2_gym_reacher_feedback_settle_anchor_0511/reacher_feedback_settle_anchor_report.json"
)


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
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _component(name: str, weight: float, passed: bool, read: str, partial: float | None = None) -> dict[str, Any]:
    credit = weight * (1.0 if passed else max(0.0, min(1.0, float(partial or 0.0))))
    return {
        "id": name,
        "weight": weight,
        "passed": bool(passed),
        "partial": None if partial is None else max(0.0, min(1.0, float(partial))),
        "credit": credit,
        "read": read,
    }


def _budgeted_support_read(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("analysis_entry") == "phase2_feedback_support_slack_audit":
        groups = payload.get("groups", {}) or {}
        best_group_name = None
        best_group = None
        best_min = -1
        for name, group in groups.items():
            if not group.get("all_rules_present"):
                continue
            by_rule = group.get("by_rule", {}) or {}
            current_min = min(
                [
                    int((by_rule.get(rule, {}) or {}).get("min", 0) or 0)
                    for rule in ("gym_full_obs_action_range", "gym_q90_obs_action_range")
                ]
                or [0]
            )
            if current_min > best_min:
                best_min = current_min
                best_group_name = str(name)
                best_group = group
        decision = payload.get("decision", {}) or {}
        by_rule = (best_group or {}).get("by_rule", {}) or {}
        counts = {
            rule: list((by_rule.get(rule, {}) or {}).get("counts", []) or [])
            for rule in ("gym_full_obs_action_range", "gym_q90_obs_action_range")
        }
        min_count = min(
            [int((by_rule.get(rule, {}) or {}).get("min", 0) or 0) for rule in counts]
            or [0]
        )
        return {
            "source": "support_slack_audit",
            "candidate_counts": counts,
            "best_group": best_group_name,
            "min_candidate_count": min_count,
            "min_slack_vs_selector_need_8": min_count - 8,
            "has_selector_minimum": bool((best_group or {}).get("all_meet_selector_minimum")),
            "has_cross_seed_slack": bool(decision.get("ready_to_clear_slack_blocker")),
            "gain_diversity_guard_pass": bool((best_group or {}).get("gain_diversity_guard_pass")),
            "read": str(decision.get("read") or "feedback support slack audit was unavailable or inconclusive"),
        }
    rules = payload.get("rules", {}) or {}
    counts = {
        rule: int((item or {}).get("candidate_count", 0) or 0)
        for rule, item in rules.items()
    }
    min_count = min(counts.values()) if counts else 0
    min_slack = min_count - 8
    return {
        "candidate_counts": counts,
        "min_candidate_count": min_count,
        "min_slack_vs_selector_need_8": min_slack,
        "has_selector_minimum": min_count >= 8,
        "has_cross_seed_slack": min_count >= 12,
        "read": (
            "budgeted startup feedback support has useful slack"
            if min_count >= 12
            else "budgeted startup feedback support only meets or barely exceeds selector minimum"
        ),
    }


def build_report(
    *,
    coverage_json: str | Path,
    sufficiency_json: str | Path,
    branch_json: str | Path,
    branch_smoke_json: str | Path,
    budgeted_json: str | Path,
    pendulum_json: str | Path,
    reacher_anchor_json: str | Path,
) -> dict[str, Any]:
    coverage = _read_json_optional(coverage_json) or {}
    sufficiency = _read_json_optional(sufficiency_json) or {}
    branch = _read_json_optional(branch_json) or {}
    smoke = _read_json_optional(branch_smoke_json) or {}
    budgeted = _read_json_optional(budgeted_json) or {}
    pendulum = _read_json_optional(pendulum_json) or {}
    reacher = _read_json_optional(reacher_anchor_json) or {}

    branch_read = branch.get("read", {}) or {}
    branch_guard = bool((branch.get("selection", {}) or {}).get("guard_pass"))
    branch_reduces = bool(branch_read.get("branch_control_reduces_constant_gain2_dominance"))
    smoke_overall = smoke.get("overall", {}) or {}
    smoke_raw_pf = _safe_float((smoke_overall.get("raw_reward_sum_delta", {}) or {}).get("positive_fraction"))
    smoke_env_pf = _safe_float((smoke_overall.get("reward_env_sum_delta", {}) or {}).get("positive_fraction"))
    smoke_raw_mean = _safe_float((smoke_overall.get("raw_reward_sum_delta", {}) or {}).get("mean"))
    smoke_env_mean = _safe_float((smoke_overall.get("reward_env_sum_delta", {}) or {}).get("mean"))
    sidecar_pass = bool(smoke_raw_pf >= 0.60 and smoke_env_pf >= 0.60 and smoke_raw_mean > 0.0 and smoke_env_mean > 0.0)
    budgeted_read = _budgeted_support_read(budgeted)
    suff_checks = sufficiency.get("checks", {}) or {}
    parity_preserved = bool(
        suff_checks.get("selector_profile_guard_pass")
        and suff_checks.get("chunked_selector_semantic_parity")
        and suff_checks.get("pack_to_fit_model_numeric_parity")
    )
    pendulum_read = ((pendulum.get("current_evidence", {}) or {}).get("current_read", {}) or {})
    pendulum_gap_identified = bool(
        pendulum_read.get("mode_is_not_clean_enough")
        and pendulum_read.get("pendulum_failure_location")
    )
    pendulum_read_top = pendulum.get("read", {}) or {}
    pendulum_anchor_defined = bool(pendulum_read_top.get("closed_loop_settle_anchor_defined"))
    pendulum_closed = bool(pendulum_read_top.get("closed_loop_settle_anchor_pass"))
    reacher_read = reacher.get("read", {}) or {}
    reacher_available = bool(reacher)
    reacher_out_of_scope = bool(reacher_read.get("out_of_scope_for_current_closure"))
    reacher_closed = bool(reacher_read.get("closed_loop_settle_anchor_pass")) or reacher_out_of_scope
    coverage_next = (coverage.get("forecast", {}) or {}).get("next_step", {}) or {}
    expected_gain = _safe_float(coverage_next.get("estimated_coverage_gain"))

    components = [
        _component(
            "subprofile_branch_guard",
            0.15,
            branch_guard and branch_reduces,
            "strict feedback subprofile/gain branch reduces constant/gain2 dominance",
        ),
        _component(
            "sidecar_health_smoke",
            0.15,
            sidecar_pass,
            f"branch sidecar raw/env positive fractions {smoke_raw_pf:.3f}/{smoke_env_pf:.3f}",
        ),
        _component(
            "budgeted_startup_support_slack",
            0.12,
            budgeted_read["has_cross_seed_slack"],
            budgeted_read["read"],
            partial=0.5 if budgeted_read["has_selector_minimum"] else 0.0,
        ),
        _component(
            "selector_pack_parity_preserved",
            0.16,
            parity_preserved,
            "selector/chunked/pack-to-fit_model parity is preserved after current profile path",
        ),
        _component(
            "pendulum_gap_localized",
            0.12,
            pendulum_gap_identified,
            "Pendulum evidence localizes missing subtype to sustained closed-loop settling",
        ),
        _component(
            "pendulum_closed_loop_settle_anchor",
            0.16,
            pendulum_closed,
            "dedicated Pendulum closed-loop settle anchor hard-passes",
        ),
        _component(
            "reacher_closed_loop_settle_anchor",
            0.14,
            reacher_available and reacher_closed,
            "Reacher anchor hard-passes or is explicitly scoped out for this closure iteration",
        ),
    ]
    score = sum(float(item["credit"]) for item in components)
    hard_pass = bool(score >= 0.78 and all(item["passed"] for item in components[-2:]))
    blockers = [
        item["id"]
        for item in components
        if not item["passed"] and item["id"] in {"pendulum_closed_loop_settle_anchor", "reacher_closed_loop_settle_anchor", "budgeted_startup_support_slack"}
    ]
    next_actions = []
    if "pendulum_closed_loop_settle_anchor" in blockers:
        if pendulum_anchor_defined:
            next_actions.append(
                "add or find a profile/fit candidate that reproduces the defined Pendulum sustained settle anchor"
            )
        else:
            next_actions.append(
                "materialize a dedicated Pendulum settle-anchor report from paired zero/open-loop/decay-feedback rollouts"
            )
    if "reacher_closed_loop_settle_anchor" in blockers:
        next_actions.append(
            "add a Reacher feedback-settle anchor or explicitly mark Reacher out-of-scope for this closure iteration"
        )
    if "budgeted_startup_support_slack" in blockers:
        next_actions.append(
            "increase cross-seed budgeted feedback support slack without adding a hard gain quota"
        )
    return {
        "analysis_entry": "phase2_feedback_closure_preflight",
        "contract": {
            "read_only_existing_artifacts": True,
            "does_not_run_ppo": True,
            "does_not_train": True,
            "does_not_sample_environments": True,
            "does_not_modify_exact_scm": True,
            "does_not_modify_fit_model": True,
            "does_not_modify_ppo": True,
        },
        "inputs": {
            "coverage_json": str(Path(coverage_json).expanduser().resolve()),
            "sufficiency_json": str(Path(sufficiency_json).expanduser().resolve()),
            "branch_json": str(Path(branch_json).expanduser().resolve()),
            "branch_smoke_json": str(Path(branch_smoke_json).expanduser().resolve()),
            "budgeted_json": str(Path(budgeted_json).expanduser().resolve()),
            "pendulum_json": str(Path(pendulum_json).expanduser().resolve()),
            "reacher_anchor_json": str(Path(reacher_anchor_json).expanduser().resolve()),
        },
        "score": {
            "score": score,
            "remaining": max(0.0, 1.0 - score),
            "hard_pass": hard_pass,
            "components": components,
        },
        "evidence": {
            "branch_guard_pass": branch_guard,
            "branch_reduces_constant_gain2_dominance": branch_reduces,
            "sidecar": {
                "raw_positive_fraction": smoke_raw_pf,
                "env_positive_fraction": smoke_env_pf,
                "raw_mean_delta": smoke_raw_mean,
                "env_mean_delta": smoke_env_mean,
                "pass": sidecar_pass,
            },
            "budgeted_support": budgeted_read,
            "selector_pack_parity_preserved": parity_preserved,
            "pendulum_gap_identified": pendulum_gap_identified,
            "pendulum_closed_loop_settle_anchor_defined": pendulum_anchor_defined,
            "pendulum_closed_loop_settle_anchor_pass": pendulum_closed,
            "reacher_anchor_available": reacher_available,
            "reacher_out_of_scope_for_current_closure": reacher_out_of_scope,
            "reacher_closed_loop_settle_anchor_pass": reacher_closed,
        },
        "decision": {
            "ready_to_claim_feedback_closure": hard_pass,
            "expected_coverage_gain_if_closed": expected_gain,
            "blockers": blockers,
            "next_actions": next_actions,
            "read": (
                "feedback branch repair is promising, but closed-loop settle semantic coverage is not yet sufficient"
                if not hard_pass
                else "feedback closure has enough branch, sidecar, parity, and settle-anchor evidence"
            ),
        },
    }


def _markdown(report: dict[str, Any]) -> str:
    score = report["score"]
    decision = report["decision"]
    ev = report["evidence"]
    lines = [
        "# Feedback Closure Preflight",
        "",
        f"- closure score: `{score['score']:.3f}`",
        f"- ready to claim feedback closure: `{decision['ready_to_claim_feedback_closure']}`",
        f"- read: {decision['read']}",
        "",
        "## Components",
        "",
        "| component | weight | pass | credit | read |",
        "|---|---:|---|---:|---|",
    ]
    for item in score["components"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{item['id']}`",
                    f"{item['weight']:.2f}",
                    f"`{item['passed']}`",
                    f"{item['credit']:.3f}",
                    item["read"],
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Evidence",
            "",
            f"- branch guard pass: `{ev['branch_guard_pass']}`",
            f"- branch reduces constant/gain2 dominance: `{ev['branch_reduces_constant_gain2_dominance']}`",
            f"- sidecar raw/env positive fraction: `{ev['sidecar']['raw_positive_fraction']:.3f}` / `{ev['sidecar']['env_positive_fraction']:.3f}`",
            f"- budgeted support counts: `{json.dumps(ev['budgeted_support']['candidate_counts'], sort_keys=True)}`",
            f"- budgeted min slack vs 8: `{ev['budgeted_support']['min_slack_vs_selector_need_8']}`",
            f"- selector/pack parity preserved: `{ev['selector_pack_parity_preserved']}`",
            f"- Pendulum gap identified: `{ev['pendulum_gap_identified']}`",
            f"- Pendulum settle anchor defined: `{ev['pendulum_closed_loop_settle_anchor_defined']}`",
            f"- Pendulum settle anchor pass: `{ev['pendulum_closed_loop_settle_anchor_pass']}`",
            f"- Reacher anchor available/out-of-scope/pass: `{ev['reacher_anchor_available']}` / `{ev['reacher_out_of_scope_for_current_closure']}` / `{ev['reacher_closed_loop_settle_anchor_pass']}`",
            "",
            "## Next Actions",
            "",
        ]
    )
    for item in decision["next_actions"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Files", ""])
    for key, value in report.get("outputs", {}).items():
        lines.append(f"- `{key}`: `{value}`")
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(
        coverage_json=args.coverage_json,
        sufficiency_json=args.sufficiency_json,
        branch_json=args.branch_json,
        branch_smoke_json=args.branch_smoke_json,
        budgeted_json=args.budgeted_json,
        pendulum_json=args.pendulum_json,
        reacher_anchor_json=args.reacher_anchor_json,
    )
    json_path = out_dir / "feedback_closure_preflight.json"
    md_path = out_dir / "feedback_closure_preflight.md"
    report["outputs"] = {"json": str(json_path), "markdown": str(md_path)}
    json_path.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report["outputs"], sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage-json", default=str(DEFAULT_COVERAGE_JSON))
    parser.add_argument("--sufficiency-json", default=str(DEFAULT_SUFFICIENCY_JSON))
    parser.add_argument("--branch-json", default=str(DEFAULT_BRANCH_JSON))
    parser.add_argument("--branch-smoke-json", default=str(DEFAULT_BRANCH_SMOKE_JSON))
    parser.add_argument("--budgeted-json", default=str(DEFAULT_BUDGETED_JSON))
    parser.add_argument("--pendulum-json", default=str(DEFAULT_PENDULUM_JSON))
    parser.add_argument("--reacher-anchor-json", default=str(DEFAULT_REACHER_ANCHOR_JSON))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
