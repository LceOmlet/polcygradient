#!/usr/bin/env python3
"""Build Gym action-norm bound health-map tables from local Gym action spaces.

The runner stores raw policy actions in the canonical pack, then clips the
active Gym action to [-1, 1] before mapping it into the Gym Box action space.
This report therefore keeps two bounds separate:

* normalized_executed_action_l2_upper_bound: sqrt(action_dim), the bound for
  the clipped normalized action that reaches the Gym bridge.
* native_action_l2_upper_bound: the L2 norm implied by the environment's
  original Box action bounds after normalized-to-Box mapping.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ticl.rl_validation import RLPFN_DEFAULT_OOP_ENVS, parse_env_list  # noqa: E402


DEFAULT_GYM_DELTA_CSV = (
    "/home/chen/RLPFN/artifacts/phase2_clean_official_checkpoint_replay_sidecars_0501/"
    "gym_checkpoint_delta.csv"
)
DEFAULT_OUTPUT_DIR = "/home/chen/RLPFN/artifacts/phase2_gym_action_norm_bound_health_map_0501"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        out = float(value)
        return out if math.isfinite(out) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _float_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _mean(values: list[float]) -> float | None:
    arr = np.asarray([v for v in values if math.isfinite(float(v))], dtype=np.float64)
    return float(np.mean(arr)) if arr.size else None


def _max(values: list[float]) -> float | None:
    arr = np.asarray([v for v in values if math.isfinite(float(v))], dtype=np.float64)
    return float(np.max(arr)) if arr.size else None


def _gym_bound_catalog(env_ids: list[str]) -> list[dict[str, Any]]:
    import gymnasium as gym
    from gymnasium import spaces

    rows: list[dict[str, Any]] = []
    for env_id in env_ids:
        env = gym.make(str(env_id))
        try:
            if not isinstance(env.action_space, spaces.Box):
                raise ValueError(f"{env_id} action space is not Box: {env.action_space!r}")
            low = np.asarray(env.action_space.low, dtype=np.float64).reshape(-1)
            high = np.asarray(env.action_space.high, dtype=np.float64).reshape(-1)
            if low.shape != high.shape or low.ndim != 1 or low.size <= 0:
                raise ValueError(f"{env_id} action space is not a flat non-empty Box")
            if not np.all(np.isfinite(low)) or not np.all(np.isfinite(high)):
                raise ValueError(f"{env_id} action bounds are not finite")
            abs_bound = np.maximum(np.abs(low), np.abs(high))
            rows.append(
                {
                    "gym_env_id": str(env_id),
                    "action_dim": int(low.size),
                    "action_low_min": float(np.min(low)),
                    "action_low_max": float(np.max(low)),
                    "action_high_min": float(np.min(high)),
                    "action_high_max": float(np.max(high)),
                    "normalized_executed_action_l2_upper_bound": float(math.sqrt(float(low.size))),
                    "native_action_l2_upper_bound": float(np.linalg.norm(abs_bound)),
                    "bound_contract": "runner clips normalized policy action to [-1,1] before Gym Box mapping",
                }
            )
        finally:
            env.close()
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(_json_safe(rows))


def _augment_delta_rows(delta_csv: Path, catalog_by_env: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    if not delta_csv.exists():
        return []
    rows: list[dict[str, Any]] = []
    with delta_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            env_id = str(row.get("gym_env_id", "") or "")
            bound = catalog_by_env.get(env_id)
            out: dict[str, Any] = dict(row)
            if bound is None:
                rows.append(out)
                continue
            normalized_bound = float(bound["normalized_executed_action_l2_upper_bound"])
            native_bound = float(bound["native_action_l2_upper_bound"])
            out["action_dim"] = int(bound["action_dim"])
            out["normalized_executed_action_l2_upper_bound"] = normalized_bound
            out["native_action_l2_upper_bound"] = native_bound
            out["action_norm_bound_contract"] = str(bound["bound_contract"])
            for prefix in ("early", "late"):
                raw_norm = _float_or_none(row.get(f"{prefix}_action_norm_mean"))
                if raw_norm is not None:
                    out[f"{prefix}_raw_policy_action_norm_mean_to_gym_bound"] = raw_norm / max(normalized_bound, 1e-12)
                clipped_norm = _float_or_none(row.get(f"{prefix}_unit_clipped_action_norm_mean"))
                if clipped_norm is not None:
                    out[f"{prefix}_unit_clipped_action_norm_mean_to_gym_bound"] = (
                        clipped_norm / max(normalized_bound, 1e-12)
                    )
                raw_absmax = _float_or_none(row.get(f"{prefix}_action_value_absmax"))
                if raw_absmax is not None:
                    out[f"{prefix}_raw_policy_action_absmax_to_unit_clip"] = raw_absmax
            early_raw_ratio = _float_or_none(out.get("early_raw_policy_action_norm_mean_to_gym_bound"))
            late_raw_ratio = _float_or_none(out.get("late_raw_policy_action_norm_mean_to_gym_bound"))
            if early_raw_ratio is not None and late_raw_ratio is not None:
                out["delta_raw_policy_action_norm_mean_to_gym_bound"] = late_raw_ratio - early_raw_ratio
            early_clip_ratio = _float_or_none(out.get("early_unit_clipped_action_norm_mean_to_gym_bound"))
            late_clip_ratio = _float_or_none(out.get("late_unit_clipped_action_norm_mean_to_gym_bound"))
            if early_clip_ratio is not None and late_clip_ratio is not None:
                out["delta_unit_clipped_action_norm_mean_to_gym_bound"] = late_clip_ratio - early_clip_ratio
            rows.append(out)
    return rows


def _group_delta_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        env_id = str(row.get("gym_env_id", "") or "")
        grouped.setdefault(env_id, []).append(row)
    out: list[dict[str, Any]] = []
    for env_id, env_rows in sorted(grouped.items()):
        late_raw = [
            v
            for row in env_rows
            if (v := _float_or_none(row.get("late_raw_policy_action_norm_mean_to_gym_bound"))) is not None
        ]
        late_clip = [
            v
            for row in env_rows
            if (v := _float_or_none(row.get("late_unit_clipped_action_norm_mean_to_gym_bound"))) is not None
        ]
        late_absmax = [
            v
            for row in env_rows
            if (v := _float_or_none(row.get("late_raw_policy_action_absmax_to_unit_clip"))) is not None
        ]
        delta_reward = [
            v
            for row in env_rows
            if (v := _float_or_none(row.get("delta_reward_sum"))) is not None
        ]
        out.append(
            {
                "gym_env_id": env_id,
                "count": len(env_rows),
                "late_raw_policy_action_norm_to_bound_mean": _mean(late_raw),
                "late_raw_policy_action_norm_to_bound_max": _max(late_raw),
                "late_unit_clipped_action_norm_to_bound_mean": _mean(late_clip),
                "late_unit_clipped_action_norm_to_bound_max": _max(late_clip),
                "late_raw_policy_action_absmax_to_unit_clip_max": _max(late_absmax),
                "late_raw_policy_action_norm_mean_exceeds_bound_count": int(sum(v > 1.0 for v in late_raw)),
                "delta_reward_sum_mean": _mean(delta_reward),
                "delta_reward_sum_min": float(np.min(delta_reward)) if delta_reward else None,
            }
        )
    return out


def _write_markdown(path: Path, catalog_rows: list[dict[str, Any]], group_rows: list[dict[str, Any]], missing: list[str]) -> None:
    lines = [
        "# Gym Action Norm Bound Health Map",
        "",
        "Contract: runner stores raw policy actions in the pack, then clips active Gym actions to [-1,1] before Gym Box mapping.",
        "",
        "## Bound Catalog",
        "",
        "| env | dim | normalized executed L2 bound | native L2 bound |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in catalog_rows:
        lines.append(
            f"| {row['gym_env_id']} | {row['action_dim']} | "
            f"{float(row['normalized_executed_action_l2_upper_bound']):.6g} | "
            f"{float(row['native_action_l2_upper_bound']):.6g} |"
        )
    lines.extend(
        [
            "",
            "## Current Checkpoint Replay Rows",
            "",
            "| env | rows | late raw norm / bound mean | late raw norm / bound max | late clipped norm / bound mean | late raw absmax / unit max | mean reward delta |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in group_rows:
        lines.append(
            f"| {row['gym_env_id']} | {row['count']} | "
            f"{row['late_raw_policy_action_norm_to_bound_mean']:.6g} | "
            f"{row['late_raw_policy_action_norm_to_bound_max']:.6g} | "
            f"{row['late_unit_clipped_action_norm_to_bound_mean'] if row['late_unit_clipped_action_norm_to_bound_mean'] is not None else 'n/a'} | "
            f"{row['late_raw_policy_action_absmax_to_unit_clip_max']:.6g} | "
            f"{row['delta_reward_sum_mean']:.6g} |"
        )
    if missing:
        lines.extend(
            [
                "",
                "## Missing From Current Replay Delta",
                "",
                ", ".join(missing),
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Add Gym action norm bounds to a health-map artifact.")
    parser.add_argument("--gym-delta-csv", type=str, default=DEFAULT_GYM_DELTA_CSV)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--env-ids", type=str, default=",".join(RLPFN_DEFAULT_OOP_ENVS))
    args = parser.parse_args()

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    env_ids = parse_env_list(args.env_ids)
    catalog_rows = _gym_bound_catalog(env_ids)
    catalog_by_env = {str(row["gym_env_id"]): row for row in catalog_rows}
    augmented_rows = _augment_delta_rows(Path(args.gym_delta_csv).expanduser().resolve(), catalog_by_env)
    group_rows = _group_delta_summary(augmented_rows) if augmented_rows else []
    present_envs = sorted({str(row.get("gym_env_id", "") or "") for row in augmented_rows})
    missing = [env_id for env_id in env_ids if env_id not in present_envs]

    catalog_path = out_dir / "gym_action_norm_bound_catalog.csv"
    health_csv_path = out_dir / "gym_checkpoint_action_norm_health_map.csv"
    health_md_path = out_dir / "health_map.md"
    _write_csv(catalog_path, catalog_rows)
    if augmented_rows:
        _write_csv(health_csv_path, augmented_rows)
    _write_markdown(health_md_path, catalog_rows, group_rows, missing)
    summary = {
        "audit_entry": "phase2_gym_action_norm_bound_health_map",
        "contract": (
            "Raw policy action norms in sidecars are preclip; Gym executed normalized actions are clipped to [-1,1], "
            "so their L2 upper bound is sqrt(action_dim). Native bounds come from local gymnasium Box action spaces."
        ),
        "env_ids": env_ids,
        "delta_csv": str(Path(args.gym_delta_csv).expanduser().resolve()),
        "catalog_csv": str(catalog_path),
        "health_map_csv": str(health_csv_path) if augmented_rows else None,
        "health_map_md": str(health_md_path),
        "current_delta_unique_envs": present_envs,
        "missing_envs_in_current_delta": missing,
        "group_delta_summary": group_rows,
        "catalog": catalog_rows,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(_json_safe(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "health_map_md": str(health_md_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
