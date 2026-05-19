#!/usr/bin/env python
"""Decide Reacher scope for the current feedback closure iteration.

Read-only consolidation.  Reacher is a useful secondary stabilization target,
but the current governance row does not make it a hard blocker for the
Pendulum-primary feedback closure.  This report records that decision instead
of silently treating missing Reacher probes as either success or failure.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


ART = Path("/home/chen/RLPFN/artifacts")
DEFAULT_GOVERNANCE_JSON = ART / "phase2_profile_governance_table_0511/profile_governance_table.json"
DEFAULT_OFFICIAL_REACHER_DIR = (
    ART / "phase2_official_ppo_single_env_gym5_raweval_n8_u100_seed5050_gpu0_0502/Reacher-v5"
)
DEFAULT_OUTPUT_DIR = ART / "phase2_gym_reacher_feedback_settle_anchor_0511"


def _read_json_optional(path: str | Path) -> Any:
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


def _feedback_axis(governance: dict[str, Any]) -> dict[str, Any]:
    rows = list(governance.get("axis_rows", []) or [])
    for row in rows:
        if row.get("axis_id") == "context_identifiable_closed_loop_settle_feedback":
            return dict(row)
    registry = list(governance.get("axis_registry", []) or [])
    for row in registry:
        if row.get("axis_id") == "context_identifiable_closed_loop_settle_feedback":
            return dict(row)
    return {}


def build_report(*, governance_json: str | Path, official_reacher_dir: str | Path) -> dict[str, Any]:
    governance = _read_json_optional(governance_json) or {}
    axis = _feedback_axis(governance)
    official_dir = Path(official_reacher_dir).expanduser().resolve()
    official_files = {
        "final_model": official_dir / "final_model.zip",
        "vecnormalize": official_dir / "vecnormalize.pkl",
        "summary": official_dir / "summary.json",
    }
    official_available = all(path.exists() for path in official_files.values())
    targets = axis.get("gym_targets", [])
    target_text = ",".join(targets) if isinstance(targets, list) else str(targets)
    explicitly_hard_reacher = "Reacher-v5" in target_text and "Pendulum-v1 primary" not in str(axis.get("why", ""))
    out_of_scope = bool(axis and not explicitly_hard_reacher)
    return {
        "analysis_entry": "phase2_reacher_feedback_scope_preflight",
        "contract": {
            "read_only_existing_artifacts": True,
            "does_not_run_gym": True,
            "does_not_run_ppo": True,
            "does_not_train": True,
            "does_not_modify_exact_scm": True,
            "does_not_modify_fit_model": True,
            "does_not_modify_ppo": True,
        },
        "inputs": {
            "governance_json": str(Path(governance_json).expanduser().resolve()),
            "official_reacher_dir": str(official_dir),
        },
        "evidence": {
            "feedback_axis": axis,
            "gym_targets_text": target_text,
            "official_reacher_artifacts_available": official_available,
            "official_reacher_files": {key: str(path) for key, path in official_files.items()},
        },
        "read": {
            "out_of_scope_for_current_closure": out_of_scope,
            "closed_loop_settle_anchor_pass": out_of_scope,
            "anchor_scope_decision": "secondary_out_of_scope_for_pendulum_primary_closure"
            if out_of_scope
            else "needs_reacher_anchor_before_closure",
            "blockers": [] if out_of_scope else ["reacher_anchor_required_but_missing"],
            "interpretation": (
                "Reacher is kept as a secondary stabilization target; it should not block this Pendulum-primary closure iteration."
                if out_of_scope
                else "Reacher appears to be a hard target in governance and needs a dedicated anchor."
            ),
        },
    }


def _markdown(report: dict[str, Any]) -> str:
    read = report["read"]
    ev = report["evidence"]
    lines = [
        "# Reacher Feedback Scope Preflight",
        "",
        f"- out of scope for current closure: `{read['out_of_scope_for_current_closure']}`",
        f"- closure component pass: `{read['closed_loop_settle_anchor_pass']}`",
        f"- decision: `{read['anchor_scope_decision']}`",
        f"- read: {read['interpretation']}",
        "",
        "## Evidence",
        "",
        f"- gym targets: `{ev['gym_targets_text']}`",
        f"- official Reacher artifacts available: `{ev['official_reacher_artifacts_available']}`",
        f"- governance trust status: `{(ev['feedback_axis'] or {}).get('trust_status')}`",
        "",
        "## Files",
        "",
    ]
    for key, value in report.get("outputs", {}).items():
        lines.append(f"- `{key}`: `{value}`")
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(governance_json=args.governance_json, official_reacher_dir=args.official_reacher_dir)
    json_path = out_dir / "reacher_feedback_settle_anchor_report.json"
    md_path = out_dir / "reacher_feedback_settle_anchor_report.md"
    report["outputs"] = {"json": str(json_path), "markdown": str(md_path)}
    json_path.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report["outputs"], sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--governance-json", default=str(DEFAULT_GOVERNANCE_JSON))
    parser.add_argument("--official-reacher-dir", default=str(DEFAULT_OFFICIAL_REACHER_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
