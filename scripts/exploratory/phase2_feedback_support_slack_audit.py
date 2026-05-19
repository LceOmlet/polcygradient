#!/usr/bin/env python
"""Audit cross-seed slack for the budgeted feedback support guard.

Read-only consolidation.  This is deliberately narrower than the feedback
guard itself: it does not roll out environments, train, or change profile
definitions.  It only answers whether existing guard artifacts contain enough
support beyond the selector minimum to remove the feedback-support blocker.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import re
from pathlib import Path
from typing import Any


ART = Path("/home/chen/RLPFN/artifacts")
DEFAULT_OUTPUT_DIR = ART / "phase2_feedback_support_slack_audit_0511"
DEFAULT_PATTERNS = (
    str(ART / "phase2_feedback_budgeted_support_guard*_0511/feedback_budgeted_support_guard.json"),
    str(
        ART
        / "phase2_profile_online_candidate_pool_feedback_support_generator_seed*_0511"
        / "feedback_budgeted_support_guard/feedback_budgeted_support_guard.json"
    ),
    str(
        ART
        / "phase2_profile_chunked_label*_seed*_0511/chunks/*/chunk_*/feedback_budgeted_support_guard"
        / "feedback_budgeted_support_guard.json"
    ),
)
RULES = ("gym_full_obs_action_range", "gym_q90_obs_action_range")
DEFAULT_DECAY_MODES = {
    "decay_obs_linear_0x100",
    "decay_obs_linear_neg_0x100",
    "decay_obs_linear_1x100",
    "decay_obs_linear_neg_1x100",
}
SEED_RE = re.compile(r"seed(?P<seed>\d+)")
MODE_RE = re.compile(
    r"^(?P<prefix>decay_|early_)?obs_linear(?P<neg>_neg)?_(?P<feature>\d+)x(?P<gain>\d+)"
)


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


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


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _discover(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    seen: set[str] = set()
    for pattern in patterns:
        for item in sorted(glob.glob(str(Path(pattern).expanduser()), recursive=True)):
            resolved = str(Path(item).resolve())
            if resolved not in seen:
                seen.add(resolved)
                paths.append(Path(resolved))
    return paths


def _seed_from(path: str | Path, payload: dict[str, Any]) -> int | None:
    selected = str((payload.get("inputs", {}) or {}).get("selected_csv", ""))
    text = f"{path} {selected}"
    match = SEED_RE.search(text)
    return None if match is None else int(match.group("seed"))


def _feedback_family(modes: list[str]) -> str:
    mode_set = set(modes)
    if mode_set == DEFAULT_DECAY_MODES:
        return "default_decay_obs2"
    profiles: set[str] = set()
    features: set[int] = set()
    gains: set[float] = set()
    for mode in modes:
        match = MODE_RE.match(mode)
        if match is None:
            continue
        prefix = match.group("prefix") or ""
        if prefix == "decay_":
            profiles.add("decay")
        elif prefix == "early_":
            profiles.add("early")
        else:
            profiles.add("constant")
        features.add(int(match.group("feature")))
        gains.add(float(match.group("gain")) / 100.0)
    if profiles:
        profile_part = "_plus_".join(profile for profile in ("decay", "early", "constant") if profile in profiles)
        obs_part = f"obs{max(features) + 1}" if features else "obs0"
        gain_part = "g" + "_".join(f"{gain:g}" for gain in sorted(gains))
        return f"{profile_part}_{obs_part}_{gain_part}"
    return "other"


def _run_key(path: Path, payload: dict[str, Any]) -> tuple[Any, ...]:
    config = payload.get("config", {}) or {}
    inputs = payload.get("inputs", {}) or {}
    selected_csv = str(inputs.get("selected_csv") or path)
    rules = tuple(sorted((payload.get("rules", {}) or {}).keys()))
    return (
        str(Path(selected_csv).expanduser()),
        _safe_int(config.get("n_steps")),
        tuple(str(mode) for mode in config.get("feedback_modes", []) or []),
        tuple(str(mode) for mode in config.get("baseline_modes", []) or []),
        rules,
    )


def _run_record(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    config = payload.get("config", {}) or {}
    rules_payload = payload.get("rules", {}) or {}
    counts = {
        rule: _safe_int((rules_payload.get(rule, {}) or {}).get("candidate_count"))
        for rule in RULES
        if rule in rules_payload
    }
    gain_counts = {
        rule: dict((rules_payload.get(rule, {}) or {}).get("candidate_gain_counts", {}) or {})
        for rule in RULES
        if rule in rules_payload
    }
    profile_counts = {
        rule: dict((rules_payload.get(rule, {}) or {}).get("candidate_profile_counts", {}) or {})
        for rule in RULES
        if rule in rules_payload
    }
    rates = {
        rule: _safe_float((rules_payload.get(rule, {}) or {}).get("candidate_rate"))
        for rule in RULES
        if rule in rules_payload
    }
    modes = [str(mode) for mode in config.get("feedback_modes", []) or []]
    return {
        "path": str(path),
        "seed": _seed_from(path, payload),
        "selected_csv": str((payload.get("inputs", {}) or {}).get("selected_csv", "")),
        "n_steps": _safe_int(config.get("n_steps")),
        "n_envs": _safe_int(config.get("n_envs")),
        "mode_count_per_rule": _safe_int(config.get("mode_count_per_rule")),
        "feedback_family": _feedback_family(modes),
        "feedback_modes": modes,
        "feedback_gain_set": sorted({float(match.group("gain")) / 100.0 for mode in modes if (match := MODE_RE.match(mode))}),
        "rules_present": sorted(counts),
        "candidate_counts": counts,
        "candidate_gain_counts": gain_counts,
        "candidate_profile_counts": profile_counts,
        "candidate_rates": rates,
        "dedupe_key": _run_key(path, payload),
    }


def _dedupe(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for record in records:
        key = tuple(record["dedupe_key"])
        if key in seen:
            continue
        seen.add(key)
        clean = dict(record)
        clean.pop("dedupe_key", None)
        out.append(clean)
    return out


def _group_summary(
    records: list[dict[str, Any]],
    *,
    selector_minimum: int,
    slack_target: int,
    min_cross_seed_runs: int,
    max_gain_dominance: float,
) -> dict[str, Any]:
    by_rule: dict[str, dict[str, Any]] = {}
    for rule in RULES:
        values = [_safe_int(record["candidate_counts"].get(rule)) for record in records if rule in record["candidate_counts"]]
        gain_dominance_shares = []
        for record in records:
            if rule not in record["candidate_counts"] or len(record.get("feedback_gain_set", [])) <= 1:
                continue
            counts = {
                str(key): _safe_int(value)
                for key, value in (record.get("candidate_gain_counts", {}).get(rule, {}) or {}).items()
            }
            total = sum(counts.values())
            if total > 0:
                gain_dominance_shares.append(max(counts.values()) / total)
        by_rule[rule] = {
            "counts": values,
            "run_count": len(values),
            "min": min(values) if values else 0,
            "max": max(values) if values else 0,
            "mean": (sum(values) / len(values)) if values else 0.0,
            "total": sum(values),
            "min_slack_vs_selector": (min(values) - selector_minimum) if values else -selector_minimum,
            "total_slack_vs_selector": sum(value - selector_minimum for value in values),
            "max_candidate_gain_dominance": max(gain_dominance_shares) if gain_dominance_shares else 0.0,
        }
    seed_values = sorted({record["seed"] for record in records if record.get("seed") is not None})
    all_rules_present = all(rule["run_count"] == len(records) and len(records) > 0 for rule in by_rule.values())
    all_meet_selector = all(rule["min"] >= selector_minimum for rule in by_rule.values())
    all_meet_slack = all(rule["min"] >= slack_target for rule in by_rule.values())
    cross_seed = len(seed_values) >= min_cross_seed_runs
    gain_diversity_guard_pass = all(
        float(rule["max_candidate_gain_dominance"]) <= float(max_gain_dominance)
        for rule in by_rule.values()
        if float(rule["max_candidate_gain_dominance"]) > 0.0
    )
    return {
        "run_count": len(records),
        "seeds": seed_values,
        "cross_seed": cross_seed,
        "all_rules_present": all_rules_present,
        "all_meet_selector_minimum": bool(all_rules_present and all_meet_selector),
        "all_meet_slack_target": bool(all_rules_present and all_meet_slack),
        "gain_diversity_guard_pass": bool(gain_diversity_guard_pass),
        "ready_to_clear_slack_blocker": bool(
            cross_seed and all_rules_present and all_meet_slack and gain_diversity_guard_pass
        ),
        "by_rule": by_rule,
    }


def _slack_eligible(record: dict[str, Any]) -> bool:
    return bool(
        set(record.get("candidate_counts", {})) == set(RULES)
        and _safe_int(record.get("n_steps")) >= 64
        and _safe_int(record.get("mode_count_per_rule")) >= 8
    )


def build_report(
    *,
    artifact_jsons: list[str | Path],
    selector_minimum: int = 8,
    slack_target: int = 12,
    min_cross_seed_runs: int = 2,
    max_gain_dominance: float = 0.85,
) -> dict[str, Any]:
    raw_records = [_run_record(Path(path).expanduser().resolve()) for path in artifact_jsons]
    records = _dedupe(raw_records)
    eligible_records = [record for record in records if _slack_eligible(record)]
    families = sorted({str(record["feedback_family"]) for record in records})
    grouped = {
        family: _group_summary(
            [record for record in eligible_records if record["feedback_family"] == family],
            selector_minimum=selector_minimum,
            slack_target=slack_target,
            min_cross_seed_runs=min_cross_seed_runs,
            max_gain_dominance=max_gain_dominance,
        )
        for family in families
    }
    default_group = grouped.get(
        "default_decay_obs2",
        _group_summary(
            [],
            selector_minimum=selector_minimum,
            slack_target=slack_target,
            min_cross_seed_runs=min_cross_seed_runs,
            max_gain_dominance=max_gain_dominance,
        ),
    )
    ready_groups = [name for name, group in grouped.items() if group["ready_to_clear_slack_blocker"]]
    near_groups = [
        name
        for name, group in grouped.items()
        if group["cross_seed"] and group["all_rules_present"] and group["all_meet_selector_minimum"]
    ]
    blockers: list[str] = []
    if not ready_groups:
        blockers.append("no_cross_seed_family_reaches_per_run_slack_target")
    if not default_group["all_meet_selector_minimum"]:
        blockers.append("default_decay_support_floor_regressed")
    if any(group["all_meet_slack_target"] and not group["gain_diversity_guard_pass"] for group in grouped.values()):
        blockers.append("slack_candidate_gain_dominance")
    next_actions = []
    if "no_cross_seed_family_reaches_per_run_slack_target" in blockers:
        next_actions.append(
            "run a bounded broadened feedback guard on at least two seeds, then re-audit for per-run min >= slack target"
        )
    if default_group["all_meet_selector_minimum"] and not default_group["all_meet_slack_target"]:
        next_actions.append(
            "do not claim closure from default decay alone: it reaches selector floor but has no full-rule slack on the weakest seed"
        )
    if "slack_candidate_gain_dominance" in blockers:
        next_actions.append(
            "treat gain=2 support as a diagnostic branch until a gain-diversity or near-best-gain guard passes"
        )
    return {
        "analysis_entry": "phase2_feedback_support_slack_audit",
        "contract": {
            "read_only_existing_guard_artifacts": True,
            "does_not_rollout": True,
            "does_not_train": True,
            "does_not_modify_exact_scm": True,
            "does_not_modify_fit_model": True,
            "does_not_modify_ppo": True,
        },
        "thresholds": {
            "selector_minimum": int(selector_minimum),
            "slack_target": int(slack_target),
            "min_cross_seed_runs": int(min_cross_seed_runs),
            "max_gain_dominance": float(max_gain_dominance),
        },
        "inputs": {
            "artifact_jsons": [str(Path(path).expanduser().resolve()) for path in artifact_jsons],
            "raw_record_count": len(raw_records),
            "deduped_record_count": len(records),
            "slack_eligible_record_count": len(eligible_records),
        },
        "records": records,
        "groups": grouped,
        "decision": {
            "ready_to_clear_slack_blocker": bool(ready_groups),
            "ready_groups": ready_groups,
            "near_but_not_slack_groups": [name for name in near_groups if name not in ready_groups],
            "blockers": blockers,
            "next_actions": next_actions,
            "read": (
                "at least one cross-seed feedback family has selector slack"
                if ready_groups
                else "feedback support is stable enough for the selector floor, but not enough to clear the slack blocker"
            ),
        },
    }


def _markdown(report: dict[str, Any]) -> str:
    decision = report["decision"]
    lines = [
        "# Feedback Support Slack Audit",
        "",
        f"- ready to clear slack blocker: `{decision['ready_to_clear_slack_blocker']}`",
        f"- read: {decision['read']}",
        f"- raw/deduped guard artifacts: `{report['inputs']['raw_record_count']}` / `{report['inputs']['deduped_record_count']}`",
        f"- selector minimum / slack target: `{report['thresholds']['selector_minimum']}` / `{report['thresholds']['slack_target']}`",
        f"- max gain dominance: `{report['thresholds']['max_gain_dominance']:.2f}`",
        "",
        "## Groups",
        "",
        "| family | runs | seeds | full min/mean | q90 min/mean | gain dom max | selector floor | slack target | ready |",
        "|---|---:|---|---:|---:|---:|---|---|---|",
    ]
    for family, group in sorted(report["groups"].items()):
        full = group["by_rule"].get("gym_full_obs_action_range", {})
        q90 = group["by_rule"].get("gym_q90_obs_action_range", {})
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{family}`",
                    str(group["run_count"]),
                    f"`{group['seeds']}`",
                    f"{int(full.get('min', 0))}/{float(full.get('mean', 0.0)):.1f}",
                    f"{int(q90.get('min', 0))}/{float(q90.get('mean', 0.0)):.1f}",
                    f"{max(float(full.get('max_candidate_gain_dominance', 0.0)), float(q90.get('max_candidate_gain_dominance', 0.0))):.2f}",
                    f"`{group['all_meet_selector_minimum']}`",
                    f"`{group['all_meet_slack_target']}`",
                    f"`{group['ready_to_clear_slack_blocker']}`",
                ]
            )
            + " |"
        )
    lines.extend(["", "## Next Actions", ""])
    for item in decision["next_actions"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Files", ""])
    for key, value in report.get("outputs", {}).items():
        lines.append(f"- `{key}`: `{value}`")
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    artifact_jsons = [Path(path).expanduser().resolve() for path in args.artifact_json]
    if not artifact_jsons:
        artifact_jsons = _discover(list(args.glob))
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(
        artifact_jsons=artifact_jsons,
        selector_minimum=int(args.selector_minimum),
        slack_target=int(args.slack_target),
        min_cross_seed_runs=int(args.min_cross_seed_runs),
        max_gain_dominance=float(args.max_gain_dominance),
    )
    json_path = out_dir / "feedback_support_slack_audit.json"
    md_path = out_dir / "feedback_support_slack_audit.md"
    report["outputs"] = {"json": str(json_path), "markdown": str(md_path)}
    json_path.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report["outputs"], sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-json", action="append", default=[])
    parser.add_argument("--glob", action="append", default=list(DEFAULT_PATTERNS))
    parser.add_argument("--selector-minimum", type=int, default=8)
    parser.add_argument("--slack-target", type=int, default=12)
    parser.add_argument("--min-cross-seed-runs", type=int, default=2)
    parser.add_argument("--max-gain-dominance", type=float, default=0.85)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
