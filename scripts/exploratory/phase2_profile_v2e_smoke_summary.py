#!/usr/bin/env python
"""Summarize v2e profile-candidate smoke runs.

Read-only profile-governance summary.  The candidate is deliberately thin:
it reuses the v2d fixed environments and adds profile annotations only.  This
summary keeps the main line on profile quality and diversity; trainability
deltas are guards/diagnostics, not selection targets.
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
DEFAULT_OUTPUT_DIR = ART / "phase2_profile_v2e_candidate_smoke_summary_0511"
DEFAULT_SELECTED_CSV = (
    ART
    / "phase2_profile_v2e_candidate_annotated_0511"
    / "selected_profile_v2e_candidate_prior_envs.csv"
)
DEFAULT_FULL_RUN_DIR = (
    ART / "phase2_profile_v2e_candidate_smoke_full_n64_s512_u4_e4_advnorm_seed9760_gpu0_0511"
)
DEFAULT_Q90_RUN_DIR = (
    ART / "phase2_profile_v2e_candidate_smoke_q90_n64_s512_u4_e4_advnorm_seed9761_gpu0_0511"
)
RULES = ("gym_full_obs_action_range", "gym_q90_obs_action_range")
BROAD_AXES = (
    "locomotion_witness",
    "low_energy_not_ctrl_only",
    "sign_or_phase_sensitive",
    "state_conditioned_energy_injection",
)


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).expanduser().resolve().open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


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
        writer.writerows(rows)


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


def _f(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _near_best_gain(row: dict[str, str], gain: str) -> bool:
    legacy_key = f"v2e_strict_near_best_gain_{gain}"
    if legacy_key in row:
        return _bool(row.get(legacy_key))
    if not _bool(row.get("feedback_near_best_support_set")):
        return False
    try:
        return abs(float(row.get("strict_feedback_gain", "nan")) - float(gain)) < 1e-9
    except ValueError:
        return False


def _stats(values: list[Any]) -> dict[str, Any]:
    arr = np.asarray([float(v) for v in values if _f(v) is not None], dtype=np.float64)
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "q10": float(np.quantile(arr, 0.10)),
        "q50": float(np.quantile(arr, 0.50)),
        "q90": float(np.quantile(arr, 0.90)),
        "max": float(np.max(arr)),
        "negative_count": int(np.sum(arr < 0.0)),
    }


def _sidecar_pair(run_dir: str | Path) -> tuple[Path, Path]:
    sidecar_dir = Path(run_dir).expanduser().resolve() / "semantic_sidecars"
    files = sorted(sidecar_dir.glob("prior_update_*_envs.jsonl"))
    if len(files) < 2:
        raise RuntimeError(f"Need at least two prior sidecars in {sidecar_dir}, found {len(files)}")
    return files[0], files[-1]


def _delta(last: dict[str, Any], first: dict[str, Any], key: str) -> float | None:
    a = _f(first.get(key))
    b = _f(last.get(key))
    if a is None or b is None:
        return None
    return float(b - a)


def _selected_rows(selected_csv: str | Path, rule: str) -> list[dict[str, str]]:
    return [row for row in _read_csv(selected_csv) if str(row.get("rule", "")).strip() == rule]


def _profile_summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    n = len(rows)
    cells: Counter[tuple[bool, ...]] = Counter(
        tuple(_bool(row.get(axis)) for axis in BROAD_AXES) for row in rows
    )
    strict_rows = [row for row in rows if _bool(row.get("strict_feedback_candidate"))]
    profile_counts = Counter(str(row.get("strict_feedback_profile", "")).strip() for row in strict_rows)
    gain_counts = Counter(str(row.get("strict_feedback_gain", "")).strip() for row in strict_rows)
    max_cell = max(cells.values()) if cells else 0
    return {
        "n": int(n),
        "broad_axis_rates": {
            axis: float(sum(_bool(row.get(axis)) for row in rows) / max(1, n))
            for axis in BROAD_AXES
        },
        "joint_active_cells": int(len(cells)),
        "joint_max_cell_share": float(max_cell / max(1, n)),
        "strict_feedback_count": int(len(strict_rows)),
        "strict_feedback_profile_counts": dict(profile_counts),
        "strict_feedback_gain_counts": dict(gain_counts),
        "strict_feedback_near_best_gain1_rate": float(
            sum(_near_best_gain(row, "1") for row in strict_rows)
            / max(1, len(strict_rows))
        ),
        "strict_feedback_near_best_gain05_rate": float(
            sum(_near_best_gain(row, "0.5") for row in strict_rows)
            / max(1, len(strict_rows))
        ),
        "mountaincar_future_basin_candidate_count": int(
            sum(_bool(row.get("mountaincar_future_basin_candidate")) for row in rows)
        ),
        "mountaincar_future_basin_strong_count": int(
            sum(_bool(row.get("mountaincar_future_basin_strong")) for row in rows)
        ),
    }


def _per_env_rows(rule: str, run_dir: str | Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    first_path, last_path = _sidecar_pair(run_dir)
    first = _load_jsonl(first_path)
    last = _load_jsonl(last_path)
    if len(first) != len(last):
        raise RuntimeError(f"Sidecar length mismatch: {first_path} vs {last_path}")
    rows: list[dict[str, Any]] = []
    for idx, (a, b) in enumerate(zip(first, last, strict=True)):
        rows.append(
            {
                "rule": rule,
                "env_idx": int(b.get("env_idx", idx)),
                "env_seed": b.get("env_group_env_seed"),
                "raw_reward_sum_delta": _delta(b, a, "raw_reward_sum"),
                "reward_env_sum_delta": _delta(b, a, "reward_env_sum"),
                "training_reward_sum_delta": _delta(b, a, "training_reward_sum"),
                "done_count_delta": _delta(b, a, "done_count"),
                "first_done_step_delta": _delta(b, a, "first_done_step"),
                "action_value_absmean_delta": _delta(b, a, "action_value_absmean"),
                "action_temporal_std_delta": _delta(b, a, "action_per_dim_temporal_std_mean"),
                "next_state_step_l2_delta_mean_delta": _delta(
                    b, a, "next_state_step_l2_delta_mean"
                ),
                "last_raw_reward_std": _f(b.get("raw_reward_std")),
                "last_training_reward_std": _f(b.get("training_reward_std")),
                "last_done_count": _f(b.get("done_count")),
                "last_next_state_step_l2_delta_mean": _f(
                    b.get("next_state_step_l2_delta_mean")
                ),
                "last_next_state_rms": _f(b.get("next_state_rms")),
                "last_action_per_dim_temporal_std_mean": _f(
                    b.get("action_per_dim_temporal_std_mean")
                ),
                "last_action_constant_dim_fraction_std_lt_0p05": _f(
                    b.get("action_constant_dim_fraction_std_lt_0p05")
                ),
            }
        )
    return rows, {"first_sidecar": str(first_path), "last_sidecar": str(last_path)}


def _smoke_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {
        "n": int(len(rows)),
        "raw_reward_sum_delta": _stats([row["raw_reward_sum_delta"] for row in rows]),
        "reward_env_sum_delta": _stats([row["reward_env_sum_delta"] for row in rows]),
        "training_reward_sum_delta": _stats([row["training_reward_sum_delta"] for row in rows]),
        "done_count_delta": _stats([row["done_count_delta"] for row in rows]),
        "first_done_step_delta": _stats([row["first_done_step_delta"] for row in rows]),
        "action_value_absmean_delta": _stats([row["action_value_absmean_delta"] for row in rows]),
        "action_temporal_std_delta": _stats([row["action_temporal_std_delta"] for row in rows]),
        "next_state_step_l2_delta_mean_delta": _stats(
            [row["next_state_step_l2_delta_mean_delta"] for row in rows]
        ),
        "last_raw_reward_std": _stats([row["last_raw_reward_std"] for row in rows]),
        "last_training_reward_std": _stats(
            [row["last_training_reward_std"] for row in rows]
        ),
        "last_done_count": _stats([row["last_done_count"] for row in rows]),
        "last_next_state_step_l2_delta_mean": _stats(
            [row["last_next_state_step_l2_delta_mean"] for row in rows]
        ),
        "last_next_state_rms": _stats([row["last_next_state_rms"] for row in rows]),
        "last_action_per_dim_temporal_std_mean": _stats(
            [row["last_action_per_dim_temporal_std_mean"] for row in rows]
        ),
        "last_action_constant_dim_fraction_std_lt_0p05": _stats(
            [row["last_action_constant_dim_fraction_std_lt_0p05"] for row in rows]
        ),
    }
    checks = {
        "raw_reward_not_collapsed": out["last_raw_reward_std"].get("q50", 0.0) > 1e-6,
        "training_reward_not_collapsed": out["last_training_reward_std"].get("q50", 0.0) > 1e-6,
        "state_transition_not_collapsed": out["last_next_state_step_l2_delta_mean"].get("q50", 0.0)
        > 1e-6,
        "action_temporal_variation_not_collapsed": out[
            "last_action_per_dim_temporal_std_mean"
        ].get("q50", 0.0)
        > 1e-6,
        "constant_action_fraction_not_dominant": out[
            "last_action_constant_dim_fraction_std_lt_0p05"
        ].get("q90", 1.0)
        < 0.5,
        "done_channel_not_collapsed": out["last_done_count"].get("q90", 0.0)
        > out["last_done_count"].get("q10", 0.0),
    }
    out["diversity_guard"] = {"checks": checks, "pass": bool(all(checks.values()))}
    out["trainability_delta_guard"] = {
        "raw_reward_sum_delta_mean_nonnegative": bool(
            out["raw_reward_sum_delta"].get("mean", 0.0) >= 0.0
        ),
        "reward_env_sum_delta_mean_nonnegative": bool(
            out["reward_env_sum_delta"].get("mean", 0.0) >= 0.0
        ),
    }
    out["trainability_delta_guard"]["pass"] = bool(
        out["trainability_delta_guard"]["raw_reward_sum_delta_mean_nonnegative"]
        and out["trainability_delta_guard"]["reward_env_sum_delta_mean_nonnegative"]
    )
    return out


def _profile_guard(
    profile: dict[str, Any],
    *,
    min_active_cells: int | None,
    max_cell_share: float,
    min_strict_feedback: int | None,
    min_active_cell_rate: float,
    min_strict_feedback_rate: float,
) -> dict[str, Any]:
    n = int(profile.get("n") or 0)
    active_threshold = (
        int(min_active_cells)
        if min_active_cells is not None
        else min(14, max(1, math.ceil(float(min_active_cell_rate) * max(1, n))))
    )
    strict_threshold = (
        int(min_strict_feedback)
        if min_strict_feedback is not None
        else min(20, max(0, math.ceil(float(min_strict_feedback_rate) * max(1, n))))
    )
    checks = {
        "active_cells": int(profile.get("joint_active_cells", 0)) >= int(active_threshold),
        "max_cell_share": float(profile.get("joint_max_cell_share", 1.0)) <= float(max_cell_share),
        "strict_feedback": int(profile.get("strict_feedback_count", 0)) >= int(strict_threshold),
    }
    return {
        "pass": bool(all(checks.values())),
        "checks": checks,
        "thresholds": {
            "active_cells_min": int(active_threshold),
            "max_cell_share": float(max_cell_share),
            "strict_feedback_min": int(strict_threshold),
            "min_active_cell_rate": float(min_active_cell_rate),
            "min_strict_feedback_rate": float(min_strict_feedback_rate),
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dirs = {
        "gym_full_obs_action_range": args.full_run_dir,
        "gym_q90_obs_action_range": args.q90_run_dir,
    }
    per_env: list[dict[str, Any]] = []
    scopes: dict[str, Any] = {}
    for rule in RULES:
        selected = _selected_rows(args.selected_csv, rule)
        rows, sidecar_meta = _per_env_rows(rule, run_dirs[rule])
        if len(selected) != len(rows):
            raise RuntimeError(
                f"{rule} selected/smoke length mismatch: selected={len(selected)} smoke={len(rows)}"
            )
        scopes[rule] = {
            "selected_profile": _profile_summary(selected),
            "sidecars": sidecar_meta,
            "smoke": _smoke_summary(rows),
        }
        scopes[rule]["profile_guard"] = _profile_guard(
            scopes[rule]["selected_profile"],
            min_active_cells=getattr(args, "profile_min_active_cells", None),
            max_cell_share=float(getattr(args, "profile_max_cell_share", 0.25)),
            min_strict_feedback=getattr(args, "profile_min_strict_feedback", None),
            min_active_cell_rate=float(getattr(args, "profile_min_active_cell_rate", 0.5)),
            min_strict_feedback_rate=float(getattr(args, "profile_min_strict_feedback_rate", 0.5)),
        )
        per_env.extend(rows)
    report = {
        "analysis_entry": "phase2_profile_v2e_smoke_summary",
        "contract": {
            "exploratory_only": True,
            "does_not_train": True,
            "reads_sidecars_only": True,
            "does_not_modify_exact_scm": True,
            "does_not_modify_fit_model": True,
            "does_not_modify_ppo": True,
            "delta_sign_fraction_not_reported": True,
            "raw_score_not_used_as_objective": True,
        },
        "inputs": {
            "selected_csv": str(Path(args.selected_csv).expanduser().resolve()),
            "full_run_dir": str(Path(args.full_run_dir).expanduser().resolve()),
            "q90_run_dir": str(Path(args.q90_run_dir).expanduser().resolve()),
        },
        "scopes": scopes,
    }
    report["overall"] = {
        "profile_guard_pass": bool(
            all(scope["profile_guard"]["pass"] for scope in scopes.values())
        ),
        "diversity_guard_pass": bool(
            all(scope["smoke"]["diversity_guard"]["pass"] for scope in scopes.values())
        ),
        "trainability_delta_guard_pass": bool(
            all(scope["smoke"]["trainability_delta_guard"]["pass"] for scope in scopes.values())
        ),
    }
    report["overall"]["candidate_smoke_guard_pass"] = bool(
        report["overall"]["profile_guard_pass"]
        and report["overall"]["diversity_guard_pass"]
        and report["overall"]["trainability_delta_guard_pass"]
    )
    report_path = out_dir / "profile_v2e_smoke_summary.json"
    per_env_csv = out_dir / "profile_v2e_smoke_per_env.csv"
    _write_csv(per_env_csv, per_env)
    report["outputs"] = {"report_json": str(report_path), "per_env_csv": str(per_env_csv)}
    report_path.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n")
    lines = [
        "# Profile v2e Smoke Summary",
        "",
        "Read-only sidecar summary. Delta signs are guards/diagnostics, not profile-selection targets.",
        "",
        "| rule | cells | max cell | strict | near gain1 | MC cand/strong | raw delta mean | env delta mean | diversity | delta guard |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for rule, scope in scopes.items():
        prof = scope["selected_profile"]
        smoke = scope["smoke"]
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{rule}`",
                    str(prof["joint_active_cells"]),
                    f"{prof['joint_max_cell_share']:.3f}",
                    str(prof["strict_feedback_count"]),
                    f"{prof['strict_feedback_near_best_gain1_rate']:.3f}",
                    f"{prof['mountaincar_future_basin_candidate_count']}/{prof['mountaincar_future_basin_strong_count']}",
                    f"{smoke['raw_reward_sum_delta'].get('mean'):.6g}",
                    f"{smoke['reward_env_sum_delta'].get('mean'):.6g}",
                    "pass" if smoke["diversity_guard"]["pass"] else "fail",
                    "pass" if smoke["trainability_delta_guard"]["pass"] else "fail",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Overall",
            "",
            f"- profile guard pass: `{report['overall']['profile_guard_pass']}`",
            f"- diversity guard pass: `{report['overall']['diversity_guard_pass']}`",
            f"- trainability delta guard pass: `{report['overall']['trainability_delta_guard_pass']}`",
            f"- candidate smoke guard pass: `{report['overall']['candidate_smoke_guard_pass']}`",
            "",
            "## Files",
            "",
            f"- JSON: `{report_path}`",
            f"- per-env CSV: `{per_env_csv}`",
        ]
    )
    md_path = out_dir / "profile_v2e_smoke_summary.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path), "markdown": str(md_path)}, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--selected-csv", default=str(DEFAULT_SELECTED_CSV))
    parser.add_argument("--full-run-dir", default=str(DEFAULT_FULL_RUN_DIR))
    parser.add_argument("--q90-run-dir", default=str(DEFAULT_Q90_RUN_DIR))
    parser.add_argument("--profile-min-active-cells", type=int, default=None)
    parser.add_argument("--profile-max-cell-share", type=float, default=0.25)
    parser.add_argument("--profile-min-strict-feedback", type=int, default=None)
    parser.add_argument("--profile-min-active-cell-rate", type=float, default=0.5)
    parser.add_argument("--profile-min-strict-feedback-rate", type=float, default=0.5)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
