#!/usr/bin/env python3
"""Build a same-contract health map from Phase-2 per-env sidecars."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any


DEFAULT_INPUT_DIR = (
    "/home/chen/RLPFN/artifacts/"
    "phase2_clean_official_sidecar_gym_prior_n64_u100_obsrewardnorm_gpu0_0501"
)
DEFAULT_OUTPUT_DIR = "/home/chen/RLPFN/artifacts/phase2_unified_sidecar_health_map_0501"

HEALTH_FIELDS = [
    "arm",
    "env_idx",
    "gym_env_id",
    "early_update_idx",
    "late_update_idx",
    "early_reward_sum",
    "late_reward_sum",
    "delta_reward_sum",
    "early_training_reward_sum",
    "late_training_reward_sum",
    "delta_training_reward_sum",
    "early_raw_reward_sum",
    "late_raw_reward_sum",
    "delta_raw_reward_sum",
    "early_reward_env_sum",
    "late_reward_env_sum",
    "delta_reward_env_sum",
    "early_done_count",
    "late_done_count",
    "early_first_done_step",
    "late_first_done_step",
    "early_raw_action_norm_mean",
    "late_raw_action_norm_mean",
    "delta_raw_action_norm_mean",
    "early_raw_action_norm_to_unit_box_bound",
    "late_raw_action_norm_to_unit_box_bound",
    "delta_raw_action_norm_to_unit_box_bound",
    "early_unit_clipped_action_norm_mean",
    "late_unit_clipped_action_norm_mean",
    "delta_unit_clipped_action_norm_mean",
    "early_unit_clipped_action_norm_to_unit_box_bound",
    "late_unit_clipped_action_norm_to_unit_box_bound",
    "delta_unit_clipped_action_norm_to_unit_box_bound",
    "early_raw_policy_action_norm_mean_to_gym_bound",
    "late_raw_policy_action_norm_mean_to_gym_bound",
    "delta_raw_policy_action_norm_mean_to_gym_bound",
    "early_unit_clipped_action_norm_mean_to_gym_bound",
    "late_unit_clipped_action_norm_mean_to_gym_bound",
    "delta_unit_clipped_action_norm_mean_to_gym_bound",
    "early_action_saturation_fraction",
    "late_action_saturation_fraction",
    "delta_action_saturation_fraction",
    "early_action_unit_clip_excess_absmean",
    "late_action_unit_clip_excess_absmean",
    "delta_action_unit_clip_excess_absmean",
    "early_action_temporal_std_mean",
    "late_action_temporal_std_mean",
    "delta_action_temporal_std_mean",
    "early_constant_dim_fraction_std_lt_0p05",
    "late_constant_dim_fraction_std_lt_0p05",
    "delta_constant_dim_fraction_std_lt_0p05",
    "early_obs_per_dim_rms_mean",
    "late_obs_per_dim_rms_mean",
    "delta_obs_per_dim_rms_mean",
    "early_next_state_step_l2_delta_to_rms_ratio",
    "late_next_state_step_l2_delta_to_rms_ratio",
    "delta_next_state_step_l2_delta_to_rms_ratio",
    "reward_delta_sign",
    "training_reward_delta_sign",
    "raw_reward_delta_sign",
    "reward_env_delta_sign",
]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def _to_int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(float(value))
    except Exception:
        return None


def _delta(late: Any, early: Any) -> float | None:
    late_f = _to_float(late)
    early_f = _to_float(early)
    if late_f is None or early_f is None:
        return None
    return late_f - early_f


def _first_present(row: dict[str, Any], keys: str | tuple[str, ...]) -> Any:
    if isinstance(keys, str):
        return row.get(keys)
    for key in keys:
        if key in row and row.get(key) is not None:
            return row.get(key)
    return None


def _delta_sign(value: Any) -> str:
    value_f = _to_float(value)
    if value_f is None:
        return "missing"
    if value_f > 0:
        return "positive"
    if value_f < 0:
        return "negative"
    return "zero"


def _sidecar_files(input_dir: Path) -> list[Path]:
    sidecar_dir = input_dir / "semantic_sidecars"
    if not sidecar_dir.exists():
        sidecar_dir = input_dir
    return sorted(sidecar_dir.glob("*_envs.jsonl"))


def _load_sidecars(input_dir: Path) -> dict[str, dict[int, dict[int, dict[str, Any]]]]:
    by_arm_update_env: dict[str, dict[int, dict[int, dict[str, Any]]]] = {}
    pattern = re.compile(r"^(?P<arm>gym|prior)_(?:checkpoint_)?update_(?P<update>\d+)_envs\.jsonl$")
    for path in _sidecar_files(input_dir):
        match = pattern.match(path.name)
        if not match:
            continue
        arm = str(match.group("arm"))
        update_idx = int(match.group("update"))
        rows = _read_jsonl(path)
        by_env = {int(row["env_idx"]): row for row in rows}
        by_arm_update_env.setdefault(arm, {})[update_idx] = by_env
    return by_arm_update_env


def _merge_sidecars(input_dirs: list[Path]) -> dict[str, dict[int, dict[int, dict[str, Any]]]]:
    merged: dict[str, dict[int, dict[int, dict[str, Any]]]] = {}
    for input_dir in input_dirs:
        loaded = _load_sidecars(input_dir)
        for arm, by_update in loaded.items():
            arm_updates = merged.setdefault(arm, {})
            for update_idx, by_env in by_update.items():
                if update_idx not in arm_updates:
                    arm_updates[update_idx] = dict(by_env)
                    continue
                for env_idx, row in by_env.items():
                    existing = arm_updates[update_idx].get(env_idx)
                    if existing is not None and existing != row:
                        raise ValueError(
                            "conflicting sidecar row for "
                            f"arm={arm} update={update_idx} env_idx={env_idx}"
                        )
                    arm_updates[update_idx][env_idx] = row
    return merged


def _unit_box_ratio(row: dict[str, Any], key: str) -> float | None:
    value = _to_float(row.get(key))
    action_dim = _to_float(row.get("action_dim"))
    if value is None or action_dim is None or action_dim <= 0:
        return None
    return value / math.sqrt(action_dim)


def _make_health_row(arm: str, env_idx: int, early: dict[str, Any], late: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "arm": arm,
        "env_idx": int(env_idx),
        "gym_env_id": late.get("gym_env_id") or early.get("gym_env_id"),
        "early_update_idx": _to_int_or_none(early.get("update_idx")),
        "late_update_idx": _to_int_or_none(late.get("update_idx")),
        "early_first_done_step": _to_int_or_none(early.get("first_done_step")),
        "late_first_done_step": _to_int_or_none(late.get("first_done_step")),
    }
    direct_fields = {
        "reward_sum": "reward_sum",
        "training_reward_sum": ("training_reward_sum", "reward_sum"),
        "raw_reward_sum": "raw_reward_sum",
        "reward_env_sum": "reward_env_sum",
        "done_count": "done_count",
        "raw_action_norm_mean": "action_norm_mean",
        "unit_clipped_action_norm_mean": "unit_clipped_action_norm_mean",
        "raw_policy_action_norm_mean_to_gym_bound": "raw_policy_action_norm_mean_to_gym_bound",
        "unit_clipped_action_norm_mean_to_gym_bound": "unit_clipped_action_norm_mean_to_gym_bound",
        "action_saturation_fraction": "raw_policy_action_unit_clip_saturation_fraction",
        "action_unit_clip_excess_absmean": "raw_policy_action_unit_clip_excess_absmean",
        "action_temporal_std_mean": "action_per_dim_temporal_std_mean",
        "constant_dim_fraction_std_lt_0p05": "action_constant_dim_fraction_std_lt_0p05",
        "obs_per_dim_rms_mean": "obs_per_dim_rms_mean",
        "next_state_step_l2_delta_to_rms_ratio": "next_state_step_l2_delta_to_rms_ratio",
    }
    for label, source_key in direct_fields.items():
        early_value = _first_present(early, source_key)
        late_value = _first_present(late, source_key)
        out[f"early_{label}"] = _to_float(early_value)
        out[f"late_{label}"] = _to_float(late_value)
        out[f"delta_{label}"] = _delta(late_value, early_value)

    out["early_raw_action_norm_to_unit_box_bound"] = _unit_box_ratio(early, "action_norm_mean")
    out["late_raw_action_norm_to_unit_box_bound"] = _unit_box_ratio(late, "action_norm_mean")
    out["delta_raw_action_norm_to_unit_box_bound"] = _delta(
        out["late_raw_action_norm_to_unit_box_bound"],
        out["early_raw_action_norm_to_unit_box_bound"],
    )
    out["early_unit_clipped_action_norm_to_unit_box_bound"] = _unit_box_ratio(early, "unit_clipped_action_norm_mean")
    out["late_unit_clipped_action_norm_to_unit_box_bound"] = _unit_box_ratio(late, "unit_clipped_action_norm_mean")
    out["delta_unit_clipped_action_norm_to_unit_box_bound"] = _delta(
        out["late_unit_clipped_action_norm_to_unit_box_bound"],
        out["early_unit_clipped_action_norm_to_unit_box_bound"],
    )
    out["reward_delta_sign"] = _delta_sign(out.get("delta_reward_sum"))
    out["training_reward_delta_sign"] = _delta_sign(out.get("delta_training_reward_sum"))
    out["raw_reward_delta_sign"] = _delta_sign(out.get("delta_raw_reward_sum"))
    out["reward_env_delta_sign"] = _delta_sign(out.get("delta_reward_env_sum"))
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _parse_ints(text: str) -> list[int]:
    return [int(part.strip()) for part in str(text).split(",") if part.strip()]


def _mean(values: list[float]) -> float | None:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(sum(vals) / len(vals)) if vals else None


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_arm: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_arm.setdefault(str(row["arm"]), []).append(row)
    out: dict[str, Any] = {"row_count": int(len(rows)), "by_arm": {}}
    for arm, arm_rows in sorted(by_arm.items()):
        reward_deltas = [
            float(v)
            for row in arm_rows
            if (v := _to_float(row.get("delta_reward_sum"))) is not None
        ]
        training_reward_deltas = [
            float(v)
            for row in arm_rows
            if (v := _to_float(row.get("delta_training_reward_sum"))) is not None
        ]
        raw_reward_deltas = [
            float(v)
            for row in arm_rows
            if (v := _to_float(row.get("delta_raw_reward_sum"))) is not None
        ]
        reward_env_deltas = [
            float(v)
            for row in arm_rows
            if (v := _to_float(row.get("delta_reward_env_sum"))) is not None
        ]
        late_sat = [
            float(v)
            for row in arm_rows
            if (v := _to_float(row.get("late_action_saturation_fraction"))) is not None
        ]
        late_raw_unit = [
            float(v)
            for row in arm_rows
            if (v := _to_float(row.get("late_raw_action_norm_to_unit_box_bound"))) is not None
        ]
        late_clip_unit = [
            float(v)
            for row in arm_rows
            if (v := _to_float(row.get("late_unit_clipped_action_norm_to_unit_box_bound"))) is not None
        ]
        out["by_arm"][arm] = {
            "env_count": int(len(arm_rows)),
            "reward_delta_mean": _mean(reward_deltas),
            "reward_delta_negative_count": int(sum(v < 0 for v in reward_deltas)),
            "training_reward_delta_mean": _mean(training_reward_deltas),
            "training_reward_delta_negative_count": int(sum(v < 0 for v in training_reward_deltas)),
            "raw_reward_delta_mean": _mean(raw_reward_deltas),
            "raw_reward_delta_negative_count": int(sum(v < 0 for v in raw_reward_deltas)),
            "raw_reward_delta_count": int(len(raw_reward_deltas)),
            "reward_env_delta_mean": _mean(reward_env_deltas),
            "reward_env_delta_negative_count": int(sum(v < 0 for v in reward_env_deltas)),
            "reward_env_delta_count": int(len(reward_env_deltas)),
            "late_action_saturation_fraction_mean": _mean(late_sat),
            "late_raw_action_norm_to_unit_box_bound_mean": _mean(late_raw_unit),
            "late_unit_clipped_action_norm_to_unit_box_bound_mean": _mean(late_clip_unit),
        }
    return out


def _write_markdown(path: Path, *, summary: dict[str, Any], rows: list[dict[str, Any]], target_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Unified Sidecar Health Map",
        "",
        "Same-contract comparison uses the first and last available training sidecar for each arm/env.",
        "",
        "## Summary",
        "",
        "| arm | envs | mean training reward delta | mean raw reward delta | raw reward rows | negative raw deltas | late raw/unit-box ratio | late clipped/unit-box ratio | late saturation |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, item in sorted(summary.get("by_arm", {}).items()):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(arm),
                    str(item.get("env_count")),
                    f"{item.get('training_reward_delta_mean'):.6g}"
                    if item.get("training_reward_delta_mean") is not None
                    else "n/a",
                    f"{item.get('raw_reward_delta_mean'):.6g}"
                    if item.get("raw_reward_delta_mean") is not None
                    else "n/a",
                    str(item.get("raw_reward_delta_count")),
                    str(item.get("raw_reward_delta_negative_count")),
                    f"{item.get('late_raw_action_norm_to_unit_box_bound_mean'):.6g}"
                    if item.get("late_raw_action_norm_to_unit_box_bound_mean") is not None
                    else "n/a",
                    f"{item.get('late_unit_clipped_action_norm_to_unit_box_bound_mean'):.6g}"
                    if item.get("late_unit_clipped_action_norm_to_unit_box_bound_mean") is not None
                    else "n/a",
                    f"{item.get('late_action_saturation_fraction_mean'):.6g}"
                    if item.get("late_action_saturation_fraction_mean") is not None
                    else "n/a",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Prior Env 9/34/36",
            "",
            "| env | training reward delta | raw reward delta | late saturation | late temporal std | late const dim frac | late next drift/rms |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in target_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("env_idx")),
                    f"{row.get('delta_training_reward_sum'):.6g}"
                    if row.get("delta_training_reward_sum") is not None
                    else "n/a",
                    f"{row.get('delta_raw_reward_sum'):.6g}"
                    if row.get("delta_raw_reward_sum") is not None
                    else "n/a",
                    f"{row.get('late_action_saturation_fraction'):.6g}"
                    if row.get("late_action_saturation_fraction") is not None
                    else "n/a",
                    f"{row.get('late_action_temporal_std_mean'):.6g}"
                    if row.get("late_action_temporal_std_mean") is not None
                    else "n/a",
                    f"{row.get('late_constant_dim_fraction_std_lt_0p05'):.6g}"
                    if row.get("late_constant_dim_fraction_std_lt_0p05") is not None
                    else "n/a",
                    f"{row.get('late_next_state_step_l2_delta_to_rms_ratio'):.6g}"
                    if row.get("late_next_state_step_l2_delta_to_rms_ratio") is not None
                    else "n/a",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Field Contract",
            "",
            "- `raw_action_norm_to_unit_box_bound` is raw policy L2 norm divided by sqrt(action_dim), available for both prior and gym.",
            "- `*_to_gym_bound` is only populated for Gym rows with runner-derived Gym normalized action bounds.",
            "- `action_saturation_fraction` is the fraction of active raw policy action scalars with abs(action) > 1 before unit clipping.",
            "- `training reward delta` is the change in sidecar `training_reward_sum`, falling back to legacy `reward_sum` for older sidecars.",
            "- `raw reward delta` is populated only when sidecars contain `raw_reward_sum`; older sidecars will show missing raw reward rows.",
            "- `obs_per_dim_rms_mean` is the model-input observation-token RMS from the sidecar, not a new environment query.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, nargs="+", default=[Path(DEFAULT_INPUT_DIR)])
    parser.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--target-envs", type=str, default="9,34,36")
    args = parser.parse_args()

    input_dirs = [Path(p) for p in args.input_dir]
    by_arm_update_env = _merge_sidecars(input_dirs)
    rows: list[dict[str, Any]] = []
    for arm, by_update in sorted(by_arm_update_env.items()):
        updates = sorted(by_update)
        if len(updates) < 2:
            continue
        early_update, late_update = updates[0], updates[-1]
        early_env = by_update[early_update]
        late_env = by_update[late_update]
        for env_idx in sorted(set(early_env) & set(late_env)):
            rows.append(_make_health_row(arm, env_idx, early_env[env_idx], late_env[env_idx]))

    target_envs = set(_parse_ints(args.target_envs))
    target_rows = [row for row in rows if row.get("arm") == "prior" and int(row.get("env_idx", -1)) in target_envs]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    health_csv = args.output_dir / "unified_health_map.csv"
    target_csv = args.output_dir / "prior_env_9_34_36_same_contract.csv"
    summary_path = args.output_dir / "summary.json"
    md_path = args.output_dir / "health_map.md"
    _write_csv(health_csv, rows, HEALTH_FIELDS)
    _write_csv(target_csv, target_rows, HEALTH_FIELDS)
    summary = {
        "audit_entry": "phase2_unified_sidecar_health_map",
        "input_dirs": [str(path) for path in input_dirs],
        "health_csv": str(health_csv),
        "target_csv": str(target_csv),
        "health_map_md": str(md_path),
        "target_envs": sorted(target_envs),
        "summary": _summary(rows),
        "limitations": [
            "Uses available sidecar rows only; for training evidence, generate sidecars during training rather than checkpoint replay.",
            "Raw reward deltas require sidecars produced after raw_reward_sum logging was added.",
            "Eval-return deltas require a separate evaluation artifact; this map uses rollout sidecar evidence.",
            "Prior rows do not have Gym-native action bounds; unit-box ratios are generic action-scale diagnostics.",
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(md_path, summary=summary["summary"], rows=rows, target_rows=target_rows)
    print(json.dumps({"summary": str(summary_path), "health_map_md": str(md_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
