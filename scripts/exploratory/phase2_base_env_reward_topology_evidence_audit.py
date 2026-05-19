#!/usr/bin/env python
"""Audit explicit evidence inside the base env reward bucket.

The maintained runner can already separate total reward into:

    base env reward + control cost + survival bonus

This audit asks a narrower question: which natural Gym reward topologies are
actually evidenced *inside* the base env reward bucket, and which are only
possible in principle through the generic Exact-SCM reward unit.

It is deliberately read-only.  Missing evidence here does not authorize a
sampler by itself; it identifies the smallest formula path that must be proven
before a sampler would be legitimate.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


ART = Path("/home/chen/RLPFN/artifacts")
DEFAULT_FIXED_CSV = (
    ART
    / "phase2_executable_formula_axis_fixed_h_list_0518"
    / "executable_formula_axis_fixed_h_list.csv"
)
DEFAULT_SIDECAR = (
    ART
    / "phase2_executable_formula_axis_runner_u2_s128_0518"
    / "semantic_sidecars"
    / "prior_update_000001_envs.jsonl"
)
DEFAULT_OUTPUT_DIR = ART / "phase2_base_env_reward_topology_evidence_audit_0518"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.expanduser().resolve().open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.expanduser().resolve().open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _stats(values: list[float]) -> dict[str, Any]:
    vals = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=np.float64)
    if vals.size == 0:
        return {"n": 0, "mean": None, "q10": None, "q50": None, "q90": None, "max": None}
    return {
        "n": int(vals.size),
        "mean": float(vals.mean()),
        "q10": float(np.quantile(vals, 0.10)),
        "q50": float(np.quantile(vals, 0.50)),
        "q90": float(np.quantile(vals, 0.90)),
        "max": float(vals.max()),
    }


def _load_h(row: dict[str, str]) -> dict[str, Any]:
    path = Path(row["full_frozen_h_json"]).expanduser().resolve()
    return json.loads(path.read_text(encoding="utf-8"))


def _nonzero_component(row: dict[str, Any], prefix: str, eps: float = 1e-8) -> bool:
    for suffix in ("std", "sum", "mean", "q90", "q50", "min", "max"):
        if abs(_safe_float(row.get(f"{prefix}_{suffix}"), 0.0)) > eps:
            return True
    return _safe_bool(row.get(f"{prefix}_q90_gt0"))


def _env_row(fixed: dict[str, str], side: dict[str, Any], h: dict[str, Any], idx: int) -> dict[str, Any]:
    lowtail_active = _safe_bool(side.get("exact_scm_gym_lowtail_family_active")) or _safe_bool(
        h.get("exact_scm_gym_lowtail_family_selected")
    )
    reward_env_std = _safe_float(side.get("reward_env_std"))
    state_gain = _safe_float(side.get("reward_state_input_gain_fraction"))
    action_gain = _safe_float(side.get("reward_action_input_gain_fraction"))
    noise_gain = _safe_float(side.get("reward_noise_input_gain_fraction"))
    return {
        "env_idx": idx,
        "dense_base_reward_present": reward_env_std > 1e-8,
        "generic_exact_scm_reward_unit": not lowtail_active,
        "lowtail_family_active": lowtail_active,
        "state_gain_fraction": state_gain,
        "action_gain_fraction": action_gain,
        "noise_gain_fraction": noise_gain,
        "state_readable": state_gain >= 0.05,
        "action_sensitive": action_gain >= 0.01,
        "noise_entangled": noise_gain >= 0.05,
        "explicit_potential_delta": lowtail_active
        and abs(_safe_float(h.get("exact_scm_gym_lowtail_reward_potential_delta_weight"))) > 1e-10,
        "explicit_state_cost": lowtail_active
        and abs(_safe_float(h.get("exact_scm_gym_lowtail_reward_state_cost"))) > 1e-10,
        "explicit_progress_linear": lowtail_active
        and abs(_safe_float(h.get("exact_scm_gym_lowtail_reward_linear_weight"))) > 1e-10,
        "explicit_distribution_head": lowtail_active
        and _safe_bool(h.get("exact_scm_gym_lowtail_reward_distribution_head_enabled")),
        "explicit_terminal_bonus": _nonzero_component(side, "reward_terminal_bonus"),
        "terminal_reset_enabled": _safe_bool(side.get("terminal_reset_enabled"))
        or _safe_bool(h.get("terminal_reset_enabled")),
        "reward_transform": str(h.get("reinforce_reward_transform", "none")),
        "reward_component_axis_target": fixed.get("reward_component_axis_target", ""),
        "terminal_formula_axis_target": fixed.get("terminal_formula_axis_target", ""),
    }


def _coverage(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    count = sum(1 for row in rows if bool(row.get(key)))
    return {"key": key, "true": count, "false": len(rows) - count, "fraction": count / max(1, len(rows))}


def _topology_decisions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cov = {row["key"]: row for row in [_coverage(rows, k) for k in (
        "dense_base_reward_present",
        "generic_exact_scm_reward_unit",
        "state_readable",
        "action_sensitive",
        "explicit_potential_delta",
        "explicit_state_cost",
        "explicit_progress_linear",
        "explicit_terminal_bonus",
    )]}

    def frac(key: str) -> float:
        return float(cov[key]["fraction"])

    return [
        {
            "topology": "dense_base_task_reward",
            "status": "covered" if frac("dense_base_reward_present") >= 1.0 else "gap",
            "evidence": "reward_env_std nonzero in runner sidecar",
            "sampler_authorized_now": False,
            "reason": "already present; no new sampler needed",
        },
        {
            "topology": "generic_state_transition_reward_unit",
            "status": "covered_as_generic_exact_scm" if frac("generic_exact_scm_reward_unit") >= 1.0 else "mixed",
            "evidence": "maintained family uses generic Exact-SCM reward unit rather than lowtail explicit reward formula",
            "sampler_authorized_now": False,
            "reason": "generic expressivity is not the same as explicit Gym topology coverage",
        },
        {
            "topology": "state_readable_reward",
            "status": "covered" if frac("state_readable") >= 0.9 else "weak",
            "evidence": "reward_state_input_gain_fraction >= 0.05",
            "sampler_authorized_now": False,
            "reason": "state-readable reward is broadly present; tune only if per-cell health fails",
        },
        {
            "topology": "action_sensitive_reward",
            "status": "covered" if frac("action_sensitive") >= 0.9 else "partial",
            "evidence": "reward_action_input_gain_fraction >= 0.01",
            "sampler_authorized_now": False,
            "reason": "action-sensitive reward is present; dominance is diagnostic, not a Gym topology hard axis",
        },
        {
            "topology": "potential_delta_or_progress",
            "status": "not_explicitly_evidenced" if frac("explicit_potential_delta") == 0.0 else "covered_explicit",
            "evidence": "lowtail potential_delta formula inactive in maintained runner image",
            "sampler_authorized_now": False,
            "reason": "needs true-runner formula path and component logging before source-label sampler",
        },
        {
            "topology": "state_cost_or_goal_distance",
            "status": "not_explicitly_evidenced" if frac("explicit_state_cost") == 0.0 else "covered_explicit",
            "evidence": "lowtail state_cost formula inactive in maintained runner image",
            "sampler_authorized_now": False,
            "reason": "may be implicit in generic reward unit, but not explicitly separable yet",
        },
        {
            "topology": "terminal_event_reward_or_penalty",
            "status": "missing" if frac("explicit_terminal_bonus") == 0.0 else "covered",
            "evidence": "reward_terminal_bonus component stats",
            "sampler_authorized_now": False,
            "reason": "requires explicit component path plus protection audit before sampler",
        },
    ]


def run(args: argparse.Namespace) -> dict[str, Any]:
    fixed_csv = Path(args.fixed_csv).expanduser().resolve()
    sidecar = Path(args.sidecar).expanduser().resolve()
    fixed_rows = _read_csv(fixed_csv)
    side_rows = _read_jsonl(sidecar)
    if len(fixed_rows) != len(side_rows):
        raise ValueError(f"fixed rows and sidecar rows differ: {len(fixed_rows)} != {len(side_rows)}")
    per_env = [_env_row(fixed, side, _load_h(fixed), idx) for idx, (fixed, side) in enumerate(zip(fixed_rows, side_rows))]
    coverage_rows = [
        _coverage(per_env, key)
        for key in [
            "dense_base_reward_present",
            "generic_exact_scm_reward_unit",
            "state_readable",
            "action_sensitive",
            "noise_entangled",
            "explicit_potential_delta",
            "explicit_state_cost",
            "explicit_progress_linear",
            "explicit_distribution_head",
            "explicit_terminal_bonus",
        ]
    ]
    decisions = _topology_decisions(per_env)
    report = {
        "analysis_entry": "phase2_base_env_reward_topology_evidence_audit",
        "schema": "phase2_base_env_reward_topology_evidence_audit.v1",
        "contract": {
            "read_only": True,
            "does_not_modify_generator": True,
            "does_not_train": True,
            "separates_generic_expressivity_from_explicit_topology_evidence": True,
        },
        "inputs": {"fixed_csv": str(fixed_csv), "sidecar": str(sidecar)},
        "read": {
            "n": len(per_env),
            "dense_base_reward_present_all": all(row["dense_base_reward_present"] for row in per_env),
            "generic_exact_scm_reward_unit_all": all(row["generic_exact_scm_reward_unit"] for row in per_env),
            "state_gain_fraction": _stats([row["state_gain_fraction"] for row in per_env]),
            "action_gain_fraction": _stats([row["action_gain_fraction"] for row in per_env]),
            "noise_gain_fraction": _stats([row["noise_gain_fraction"] for row in per_env]),
            "reward_transform_counts": dict(Counter(str(row["reward_transform"]) for row in per_env)),
            "sampler_authorized_now": [
                row["topology"] for row in decisions if bool(row["sampler_authorized_now"])
            ],
        },
        "coverage": coverage_rows,
        "decisions": decisions,
        "per_env": per_env,
    }
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": out_dir / "base_env_reward_topology_evidence_audit.json",
        "markdown": out_dir / "base_env_reward_topology_evidence_audit.md",
        "coverage_csv": out_dir / "base_env_reward_topology_coverage.csv",
        "per_env_csv": out_dir / "base_env_reward_topology_per_env.csv",
    }
    report["outputs"] = {key: str(value) for key, value in paths.items()}
    paths["json"].write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(paths["coverage_csv"], coverage_rows)
    _write_csv(paths["per_env_csv"], per_env)
    paths["markdown"].write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report["outputs"], sort_keys=True))
    return report


def _markdown(report: dict[str, Any]) -> str:
    read = report["read"]
    lines = [
        "# Base Env Reward Topology Evidence Audit",
        "",
        "Read-only audit of explicit topology evidence inside the base env reward bucket.",
        "",
        "## Read",
        "",
        f"- n: `{read['n']}`",
        f"- dense base reward present all: `{read['dense_base_reward_present_all']}`",
        f"- generic Exact-SCM reward unit all: `{read['generic_exact_scm_reward_unit_all']}`",
        f"- state gain fraction: `{read['state_gain_fraction']}`",
        f"- action gain fraction: `{read['action_gain_fraction']}`",
        f"- noise gain fraction: `{read['noise_gain_fraction']}`",
        f"- reward transform counts: `{read['reward_transform_counts']}`",
        f"- sampler authorized now: `{read['sampler_authorized_now']}`",
        "",
        "## Coverage",
        "",
        "| key | true | false | fraction |",
        "|---|---:|---:|---:|",
    ]
    for row in report["coverage"]:
        lines.append(f"| `{row['key']}` | {row['true']} | {row['false']} | {row['fraction']:.3f} |")
    lines.extend(
        [
            "",
            "## Decisions",
            "",
            "| topology | status | sampler now | evidence | reason |",
            "|---|---|---:|---|---|",
        ]
    )
    for row in report["decisions"]:
        lines.append(
            f"| `{row['topology']}` | `{row['status']}` | `{row['sampler_authorized_now']}` | "
            f"{row['evidence']} | {row['reason']} |"
        )
    lines.extend(["", "## Outputs", ""])
    for key, value in report["outputs"].items():
        lines.append(f"- `{key}`: `{value}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-csv", default=str(DEFAULT_FIXED_CSV))
    parser.add_argument("--sidecar", default=str(DEFAULT_SIDECAR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
