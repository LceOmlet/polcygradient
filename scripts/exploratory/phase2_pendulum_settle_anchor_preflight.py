#!/usr/bin/env python
"""Materialize a Pendulum closed-loop settle anchor from existing probes.

Read-only consolidation.  This script does not run Gym, PPO, fit_model, or the
prior generator.  It converts existing Pendulum mechanism reports into an
explicit closure gate so feedback coverage cannot be claimed from early action
or source-pool support alone.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any


ART = Path("/home/chen/RLPFN/artifacts")
DEFAULT_MECHANISM_JSON = (
    ART / "phase2_gym_pendulum_mechanism_signature_probe_0509/fit_epoch29_vs_official_single_env.json"
)
DEFAULT_EARLY_JSONS = (
    ART / "phase2_gym_pendulum_mechanism_signature_probe_0509/official_earlycarry_frac010.json",
    ART / "phase2_gym_pendulum_mechanism_signature_probe_0509/official_earlycarry_frac025.json",
    ART / "phase2_gym_pendulum_mechanism_signature_probe_0509/official_earlycarry_frac050.json",
)
DEFAULT_CONTROL_AUDIT_JSON = (
    ART / "phase2_pendulum_control_mode_sufficiency_audit_0509/pendulum_control_mode_sufficiency_audit_report.json"
)
DEFAULT_OUTPUT_DIR = ART / "phase2_pendulum_settle_anchor_preflight_0511"
FRAC_RE = re.compile(r"frac(?P<num>\d+)")


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


def _safe_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _mean(summary: dict[str, Any], key: str) -> float:
    value = summary.get(key, {}) if isinstance(summary, dict) else {}
    return _safe_float((value or {}).get("mean"))


def _source_read(payload: dict[str, Any], source: str) -> dict[str, Any]:
    summary = (payload.get("summaries", {}) or {}).get(source, {}) or {}
    return {
        "source": source,
        "raw_episode_return_mean": _mean(summary, "raw_episode_return"),
        "late_upright_frac_mean": _mean(summary, "late_upright_frac_abs_theta_lt_0_5"),
        "early_abs_action_mean": _mean(summary, "early_abs_action"),
        "late_abs_action_mean": _mean(summary, "late_abs_action"),
        "late_abs_theta_mean": _mean(summary, "late_abs_theta"),
    }


def _early_frac(path: str | Path) -> float | None:
    match = FRAC_RE.search(str(path))
    if match is None:
        return None
    return int(match.group("num")) / 100.0


def build_report(
    *,
    mechanism_json: str | Path,
    early_jsons: list[str | Path],
    control_audit_json: str | Path,
) -> dict[str, Any]:
    mechanism = _read_json(mechanism_json)
    control = _read_json(control_audit_json)
    official = _source_read(mechanism, "official_sb3_single_env_policy")
    fit = _source_read(mechanism, "fit_policy")
    zero = _source_read(mechanism, "zero")
    random = _source_read(mechanism, "random")
    early_reads = []
    for path in early_jsons:
        payload = _read_json(path)
        row = _source_read(payload, "official_sb3_single_env_policy_early_then_zero")
        row["early_fraction"] = _early_frac(path)
        row["path"] = str(Path(path).expanduser().resolve())
        early_reads.append(row)

    official_full_anchor = bool(
        official["raw_episode_return_mean"] >= -300.0
        and official["late_upright_frac_mean"] >= 0.90
        and official["late_abs_action_mean"] >= 0.10
    )
    fit_current_fails = bool(
        fit["raw_episode_return_mean"] <= -800.0
        and fit["late_upright_frac_mean"] <= 0.20
        and fit["late_abs_action_mean"] <= 0.10
    )
    early_only_insufficient = bool(
        early_reads
        and all(
            row["late_upright_frac_mean"] < 0.80 or row["raw_episode_return_mean"] < -300.0
            for row in early_reads
        )
    )
    anchor_defined = bool(official_full_anchor and fit_current_fails and early_only_insufficient)
    current_profile_reproduces_anchor = False
    current_read = ((control.get("current_evidence", {}) or {}).get("current_read", {}) or {})
    blockers = []
    if not official_full_anchor:
        blockers.append("official_full_policy_anchor_missing")
    if not early_only_insufficient:
        blockers.append("early_only_not_falsified")
    if fit_current_fails:
        blockers.append("current_fit_policy_does_not_settle")
    if anchor_defined and not current_profile_reproduces_anchor:
        blockers.append("no_profile_or_fit_candidate_reproduces_sustained_settle")
    return {
        "analysis_entry": "phase2_pendulum_settle_anchor_preflight",
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
            "mechanism_json": str(Path(mechanism_json).expanduser().resolve()),
            "early_jsons": [str(Path(path).expanduser().resolve()) for path in early_jsons],
            "control_audit_json": str(Path(control_audit_json).expanduser().resolve()),
        },
        "current_evidence": {"current_read": current_read},
        "evidence": {
            "official_full": official,
            "fit_policy": fit,
            "zero": zero,
            "random": random,
            "official_early_then_zero": early_reads,
        },
        "read": {
            "closed_loop_settle_anchor_defined": anchor_defined,
            "closed_loop_settle_anchor_pass": bool(anchor_defined and current_profile_reproduces_anchor),
            "official_full_anchor_pass": official_full_anchor,
            "fit_current_fails_anchor": fit_current_fails,
            "early_only_insufficient": early_only_insufficient,
            "current_profile_reproduces_anchor": current_profile_reproduces_anchor,
            "blockers": blockers,
            "interpretation": (
                "Pendulum anchor is defined, but current fit/profile evidence does not reproduce sustained closed-loop settle"
                if anchor_defined and not current_profile_reproduces_anchor
                else "Pendulum anchor definition is not yet sufficiently established"
            ),
        },
    }


def _markdown(report: dict[str, Any]) -> str:
    read = report["read"]
    ev = report["evidence"]
    lines = [
        "# Pendulum Settle Anchor Preflight",
        "",
        f"- anchor defined: `{read['closed_loop_settle_anchor_defined']}`",
        f"- closed-loop settle anchor pass: `{read['closed_loop_settle_anchor_pass']}`",
        f"- read: {read['interpretation']}",
        "",
        "## Main Signals",
        "",
        "| source | return mean | late upright | early abs action | late abs action |",
        "|---|---:|---:|---:|---:|",
    ]
    for key in ("official_full", "fit_policy", "zero", "random"):
        row = ev[key]
        lines.append(
            f"| `{row['source']}` | {row['raw_episode_return_mean']:.1f} | "
            f"{row['late_upright_frac_mean']:.3f} | {row['early_abs_action_mean']:.3f} | "
            f"{row['late_abs_action_mean']:.3f} |"
        )
    lines.extend(["", "## Early-Only Falsification", "", "| early fraction | return mean | late upright | late abs action |", "|---:|---:|---:|---:|"])
    for row in ev["official_early_then_zero"]:
        frac = row["early_fraction"]
        lines.append(
            f"| {0.0 if frac is None else float(frac):.2f} | {row['raw_episode_return_mean']:.1f} | "
            f"{row['late_upright_frac_mean']:.3f} | {row['late_abs_action_mean']:.3f} |"
        )
    lines.extend(["", "## Blockers", ""])
    for item in read["blockers"]:
        lines.append(f"- `{item}`")
    lines.extend(["", "## Files", ""])
    for key, value in report.get("outputs", {}).items():
        lines.append(f"- `{key}`: `{value}`")
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(
        mechanism_json=args.mechanism_json,
        early_jsons=list(args.early_json),
        control_audit_json=args.control_audit_json,
    )
    json_path = out_dir / "pendulum_settle_anchor_preflight.json"
    md_path = out_dir / "pendulum_settle_anchor_preflight.md"
    report["outputs"] = {"json": str(json_path), "markdown": str(md_path)}
    json_path.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report["outputs"], sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mechanism-json", default=str(DEFAULT_MECHANISM_JSON))
    parser.add_argument("--early-json", action="append", default=[str(path) for path in DEFAULT_EARLY_JSONS])
    parser.add_argument("--control-audit-json", default=str(DEFAULT_CONTROL_AUDIT_JSON))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
