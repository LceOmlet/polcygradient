#!/usr/bin/env python
"""Audit category span from an h-list plus sidecar rows.

This is a read-only audit.  It does not propose or implement a new generator.
The purpose is narrower: turn the current fixed h-list and true-runner
sidecar into category cells, then report first-order coverage, selected
second-order coverage, structure-mode independence, and the health evidence
available for each cell.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable


ART = Path("/home/chen/RLPFN/artifacts")
DEFAULT_RUN_DIR = (
    ART
    / "phase2_m2_current_runner_probe_0515"
    / "prior_m2_q90_fixed64_s512_u4_e4_advnorm_b2048_0515"
)
DEFAULT_OUTPUT_DIR = ART / "phase2_category_span_audit_m2_q90_0518"


Json = dict[str, Any]
Rows = list[dict[str, Any]]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def _read_jsonl(path: str | Path) -> Rows:
    rows: Rows = []
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_csv(path: Path, rows: Rows) -> None:
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


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _finite(value: Any) -> bool:
    return math.isfinite(_safe_float(value))


def _mean(values: list[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return float("nan")
    return sum(vals) / len(vals)


def _quantile(values: list[float], q: float) -> float:
    vals = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not vals:
        return float("nan")
    if len(vals) == 1:
        return vals[0]
    pos = q * (len(vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def _stats(values: list[Any]) -> dict[str, Any]:
    vals = [_safe_float(v) for v in values]
    vals = [v for v in vals if math.isfinite(v)]
    if not vals:
        return {
            "n": 0,
            "mean": None,
            "min": None,
            "q25": None,
            "q50": None,
            "q75": None,
            "max": None,
        }
    return {
        "n": len(vals),
        "mean": _mean(vals),
        "min": min(vals),
        "q25": _quantile(vals, 0.25),
        "q50": _quantile(vals, 0.50),
        "q75": _quantile(vals, 0.75),
        "max": max(vals),
    }


def _entropy(counts: Counter[str]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    ent = 0.0
    for count in counts.values():
        if count:
            p = count / total
            ent -= p * math.log(p)
    return ent


def _normalized_entropy(counts: Counter[str], expected_bins: int) -> float:
    if expected_bins <= 1:
        return 1.0
    return _entropy(counts) / math.log(expected_bins)


def _cramers_v(labels_a: list[str], labels_b: list[str]) -> float:
    pairs = [(a, b) for a, b in zip(labels_a, labels_b) if a != "missing" and b != "missing"]
    if not pairs:
        return float("nan")
    a_vals = sorted({a for a, _ in pairs})
    b_vals = sorted({b for _, b in pairs})
    if len(a_vals) <= 1 or len(b_vals) <= 1:
        return 0.0
    a_idx = {v: i for i, v in enumerate(a_vals)}
    b_idx = {v: i for i, v in enumerate(b_vals)}
    table = [[0.0 for _ in b_vals] for _ in a_vals]
    for a, b in pairs:
        table[a_idx[a]][b_idx[b]] += 1.0
    row_totals = [sum(row) for row in table]
    col_totals = [sum(table[i][j] for i in range(len(a_vals))) for j in range(len(b_vals))]
    n = sum(row_totals)
    if n <= 0:
        return float("nan")
    chi2 = 0.0
    for i in range(len(a_vals)):
        for j in range(len(b_vals)):
            expected = row_totals[i] * col_totals[j] / n
            if expected > 0:
                chi2 += (table[i][j] - expected) ** 2 / expected
    denom = n * min(len(a_vals) - 1, len(b_vals) - 1)
    if denom <= 0:
        return 0.0
    return math.sqrt(chi2 / denom)


def _bin_linear(value: Any, edges: tuple[float, float], labels: tuple[str, str, str] = ("low", "mid", "high")) -> str:
    x = _safe_float(value)
    if not math.isfinite(x):
        return "missing"
    if x <= edges[0]:
        return labels[0]
    if x <= edges[1]:
        return labels[1]
    return labels[2]


def _bin_logish(
    value: Any,
    edges: tuple[float, float],
    labels: tuple[str, str, str] = ("tiny", "small", "large"),
) -> str:
    return _bin_linear(value, edges, labels)


def _boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n", "", "none", "nan"}:
        return False
    return bool(value)


def _activation(row: Json) -> str:
    text = str(row.get("prior_mlp_activations", "")).lower()
    if "tanh" in text:
        return "tanh"
    if "relu" in text:
        return "relu"
    if "sin" in text:
        return "sin"
    return text or "missing"


def _gain_dominance(row: Json) -> str:
    gains = {
        "state": _safe_float(row.get("reward_state_input_gain_fraction")),
        "action": _safe_float(row.get("reward_action_input_gain_fraction")),
        "noise": _safe_float(row.get("reward_noise_input_gain_fraction")),
    }
    gains = {k: v for k, v in gains.items() if math.isfinite(v)}
    if not gains:
        return "missing"
    return max(gains, key=gains.get)


def _reward_topology(row: Json) -> str:
    def _present(prefix: str) -> bool:
        for suffix in ("std", "sum", "mean", "q90", "q50", "min", "max"):
            value = _safe_float(row.get(f"{prefix}_{suffix}"))
            if math.isfinite(value) and abs(value) > 1e-8:
                return True
        value = _safe_float(row.get(f"{prefix}_positive_fraction"))
        return bool(math.isfinite(value) and value > 0.0)

    parts = []
    if _present("reward_env"):
        parts.append("env")
    if _present("reward_ctrl"):
        parts.append("ctrl")
    if _present("reward_survival"):
        parts.append("survival")
    return "+".join(parts) if parts else "zero"


def _terminal_event_reward_source(row: Json) -> str:
    label = str(row.get("h:terminal_event_reward_axis_target", "")).strip().lower()
    if label in {"terminal_event_reward_on", "on", "enabled", "true", "1"}:
        return "on"
    if label in {"terminal_event_reward_off", "off", "disabled", "false", "0"}:
        return "off"
    hi = _safe_float(row.get("h:terminal_bonus_scale_max"))
    lo = _safe_float(row.get("h:terminal_bonus_scale_min"))
    if math.isfinite(hi) or math.isfinite(lo):
        return "on" if max(hi if math.isfinite(hi) else 0.0, lo if math.isfinite(lo) else 0.0) > 0.0 else "off"
    return "missing"


def _potential_progress_reward_source(row: Json) -> str:
    label = str(row.get("h:potential_progress_reward_axis_target", "")).strip().lower()
    if label in {"progress_only", "progress"}:
        return "progress_only"
    if label in {"potential_delta_progress", "potential_progress", "potential_delta"}:
        return "potential_delta_progress"
    lowtail_enabled = _boolish(row.get("h:exact_scm_gym_lowtail_family_enabled", False)) or _boolish(
        row.get("h:exact_scm_gym_lowtail_family_selected", False)
    )
    if not lowtail_enabled:
        return "missing"
    potential = abs(_safe_float(row.get("h:exact_scm_gym_lowtail_reward_potential_delta_weight")))
    progress = abs(_safe_float(row.get("h:exact_scm_gym_lowtail_reward_linear_weight")))
    if potential > 1e-8:
        return "potential_delta_progress"
    if progress > 1e-8:
        return "progress_only"
    return "missing"


def _action_support(row: Json) -> str:
    gain = _safe_float(row.get("reward_action_input_gain_fraction"))
    if not math.isfinite(gain):
        return "missing"
    if gain < 0.01:
        return "weak"
    if gain < 0.04:
        return "present"
    return "strong"


def _done_bin(row: Json) -> str:
    count = _safe_float(row.get("done_count"))
    if not math.isfinite(count):
        return "missing"
    if count <= 0:
        return "none"
    if count <= 5:
        return "low"
    if count <= 20:
        return "mid"
    return "high"


def _first_done_bin(row: Json) -> str:
    step = _safe_float(row.get("first_done_step"))
    if not math.isfinite(step):
        return "missing"
    if step <= 32:
        return "early"
    if step <= 128:
        return "mid"
    if step <= 384:
        return "late"
    return "very_late_or_none"


def _latest_sidecar(run_dir: Path) -> Path:
    sidecar_dir = run_dir / "semantic_sidecars"
    candidates = sorted(sidecar_dir.glob("*_envs.jsonl"))
    if not candidates:
        raise FileNotFoundError(f"no sidecar jsonl found under {sidecar_dir}")
    return candidates[-1]


def _latest_progress(run_dir: Path) -> Path | None:
    path = run_dir / "prior_progress.jsonl"
    return path if path.exists() else None


def _join_rows(h_rows: Rows, side_rows: Rows) -> Rows:
    side_by_idx: dict[int, Json] = {}
    side_has_idx = all("env_idx" in row for row in side_rows)
    if side_has_idx:
        for row in side_rows:
            idx = int(row["env_idx"])
            side_by_idx[idx] = row

    joined: Rows = []
    for idx, h in enumerate(h_rows):
        side = side_by_idx.get(idx, side_rows[idx] if idx < len(side_rows) else {})
        row: Json = {"env_idx": idx}
        for key, value in h.items():
            row[f"h:{key}"] = value
        for key, value in side.items():
            row[key] = value
        joined.append(row)
    return joined


AxisFn = Callable[[Json], str]


def _axis_specs() -> dict[str, dict[str, Any]]:
    # Bins are category-contract bins, not quantiles of the current artifact.
    # They deliberately report missing high support when a selected subset only
    # occupies the low part of a configured uniform range.
    return {
        "state_dim_source": {
            "kind": "structure",
            "role": "required_category_axis",
            "bins": ["low", "mid", "high"],
            "fn": lambda r: _bin_linear(r.get("h:state_dim"), (133.0, 266.0)),
        },
        "obs_dim_source": {
            "kind": "structure",
            "role": "required_category_axis",
            "bins": ["low", "mid", "high"],
            "fn": lambda r: _bin_linear(r.get("h:obs_dim"), (133.0, 266.0)),
        },
        "action_dim_source": {
            "kind": "structure",
            "role": "required_category_axis",
            "bins": ["low", "mid", "high"],
            "fn": lambda r: _bin_linear(r.get("h:action_dim"), (10.0, 20.0)),
        },
        "noise_dim_source": {
            "kind": "structure",
            "role": "required_category_axis",
            "bins": ["low", "mid", "high"],
            "fn": lambda r: _bin_linear(r.get("h:noise_dim"), (20.0, 40.0)),
        },
        "zero_pad_dim_source": {
            "kind": "structure",
            "role": "required_category_axis",
            "bins": ["low", "mid", "high"],
            "fn": lambda r: _bin_linear(r.get("h:zero_pad_dim"), (133.0, 266.0)),
        },
        "terminal_target_source": {
            "kind": "structure",
            "role": "required_category_axis",
            "bins": ["low", "mid", "high"],
            "fn": lambda r: _bin_linear(r.get("h:terminal_reset_count_target"), (26.6667, 53.3333)),
        },
        "reward_scale_source": {
            "kind": "structure",
            "role": "required_category_axis",
            "bins": ["low", "mid", "high"],
            "fn": lambda r: _bin_linear(r.get("h:reward_scale"), (3.3333, 6.6667)),
        },
        "init_state_std_source": {
            "kind": "structure",
            "role": "required_category_axis",
            "bins": ["tiny", "small", "large"],
            "fn": lambda r: _bin_logish(r.get("h:init_state_std"), (0.01, 0.10)),
        },
        "state_noise_std_source": {
            "kind": "structure",
            "role": "required_category_axis",
            "bins": ["tiny", "small", "large"],
            "fn": lambda r: _bin_logish(r.get("h:state_noise_std"), (0.001, 0.01)),
        },
        "activation": {
            "kind": "mode",
            "role": "required_category_axis",
            "bins": ["tanh", "relu", "sin"],
            "fn": lambda r: _activation({k[2:]: v for k, v in r.items() if k.startswith("h:")}),
        },
        "reward_transform": {
            "kind": "mode",
            "role": "mechanism_diagnostic_not_required",
            "bins": ["none", "tanh", "clip", "other"],
            "fn": lambda r: str(r.get("h:reinforce_reward_transform", "missing")).lower() or "missing",
        },
        "action_transform": {
            "kind": "mode",
            "role": "mechanism_diagnostic_not_required",
            "bins": ["none", "tanh", "clip", "other"],
            "fn": lambda r: str(r.get("h:reinforce_action_transform", "missing")).lower() or "missing",
        },
        "dynamics_step_ratio": {
            "kind": "behavior",
            "role": "behavior_health_diagnostic_not_required_uniform",
            "bins": ["low", "mid", "high"],
            "fn": lambda r: _bin_linear(r.get("next_state_step_l2_delta_to_rms_ratio"), (0.8, 1.2)),
        },
        "dynamics_rms": {
            "kind": "behavior",
            "role": "behavior_health_diagnostic_not_required_uniform",
            "bins": ["low", "mid", "high"],
            "fn": lambda r: _bin_linear(r.get("next_state_rms"), (0.4, 1.0)),
        },
        "obs_std": {
            "kind": "behavior",
            "role": "behavior_health_diagnostic_not_required_uniform",
            "bins": ["low", "mid", "high"],
            "fn": lambda r: _bin_linear(r.get("obs_per_dim_std_q90"), (0.5, 1.0)),
        },
        "reward_token_std": {
            "kind": "behavior",
            "role": "behavior_health_diagnostic_not_required_uniform",
            "bins": ["low", "mid", "high"],
            "fn": lambda r: _bin_linear(r.get("input_reward_token_std"), (0.02, 0.06)),
        },
        "raw_reward_std": {
            "kind": "behavior",
            "role": "behavior_health_diagnostic_not_required_uniform",
            "bins": ["low", "mid", "high"],
            "fn": lambda r: _bin_linear(r.get("raw_reward_std"), (0.2, 0.8)),
        },
        "done_count": {
            "kind": "behavior",
            "role": "behavior_health_diagnostic_not_required_uniform",
            "bins": ["none", "low", "mid", "high"],
            "fn": _done_bin,
        },
        "first_done_step": {
            "kind": "behavior",
            "role": "behavior_health_diagnostic_not_required_uniform",
            "bins": ["early", "mid", "late", "very_late_or_none"],
            "fn": _first_done_bin,
        },
        "reward_gain_dominance": {
            "kind": "mode",
            "role": "candidate_mode_axis_needs_definition",
            "bins": ["state", "action", "noise"],
            "fn": _gain_dominance,
        },
        "reward_action_support": {
            "kind": "mode",
            "role": "candidate_mode_axis_needs_definition",
            "bins": ["weak", "present", "strong"],
            "fn": _action_support,
        },
        "reward_component_topology": {
            "kind": "mode",
            "role": "required_category_axis",
            "bins": [
                "env",
                "env+ctrl",
                "env+survival",
                "env+ctrl+survival",
            ],
            "fn": _reward_topology,
        },
        "terminal_event_reward_source": {
            "kind": "mode",
            "role": "required_category_axis",
            "bins": ["off", "on"],
            "fn": _terminal_event_reward_source,
        },
        "potential_progress_reward_source": {
            "kind": "mode",
            "role": "required_category_axis",
            "bins": ["progress_only", "potential_delta_progress"],
            "fn": _potential_progress_reward_source,
        },
        "action_saturation": {
            "kind": "behavior",
            "role": "health_diagnostic_not_required_uniform",
            "bins": ["low", "mid", "high"],
            "fn": lambda r: _bin_linear(r.get("raw_policy_action_unit_clip_saturation_fraction"), (0.05, 0.25)),
        },
        "policy_action_temporal_std": {
            "kind": "behavior",
            "role": "health_diagnostic_not_required_uniform",
            "bins": ["low", "unit", "high"],
            "fn": lambda r: _bin_linear(
                r.get("action_per_dim_temporal_std_mean"),
                (0.75, 1.25),
                ("low", "unit", "high"),
            ),
        },
    }


MISSING_EVIDENCE_AXES = [
    {
        "axis": "controllability_rank",
        "reason": "no per-env rank/Gramian evidence in h-list or sidecar",
    },
    {
        "axis": "controllability_delay",
        "reason": "no explicit action-to-state delay category in h-list or sidecar",
    },
    {
        "axis": "obs_projection_rank",
        "reason": "obs dim is present, but projection rank/conditioning is not logged",
    },
    {
        "axis": "done_topology_type",
        "reason": "done_count is logged, but boundary type/geometry is not logged",
    },
    {
        "axis": "reward_topology_type",
        "reason": "reward components/gains are logged, but reward-level topology labels are not complete",
    },
    {
        "axis": "ppo_response_cell",
        "reason": "progress rows are aggregate across env group; no per-env PPO response is present",
    },
    {
        "axis": "per_cell_optimizability",
        "reason": "no per-env pre/post train delta artifact is present in this run directory",
    },
]


PAIR_SPECS = [
    ("state_dim_source", "action_dim_source"),
    ("obs_dim_source", "action_dim_source"),
    ("action_dim_source", "terminal_target_source"),
    ("state_dim_source", "terminal_target_source"),
    ("obs_dim_source", "terminal_target_source"),
    ("noise_dim_source", "zero_pad_dim_source"),
    ("activation", "action_dim_source"),
    ("activation", "terminal_target_source"),
    ("state_dim_source", "dynamics_step_ratio"),
    ("obs_dim_source", "obs_std"),
    ("action_dim_source", "reward_action_support"),
    ("action_dim_source", "dynamics_step_ratio"),
    ("noise_dim_source", "reward_token_std"),
    ("terminal_target_source", "done_count"),
    ("reward_scale_source", "reward_token_std"),
    ("activation", "dynamics_step_ratio"),
    ("reward_gain_dominance", "reward_token_std"),
    ("obs_dim_source", "reward_token_std"),
    ("obs_std", "reward_token_std"),
    ("dynamics_step_ratio", "reward_token_std"),
    ("done_count", "reward_token_std"),
    ("terminal_event_reward_source", "action_dim_source"),
    ("terminal_event_reward_source", "terminal_target_source"),
    ("terminal_event_reward_source", "reward_component_topology"),
    ("potential_progress_reward_source", "action_dim_source"),
    ("potential_progress_reward_source", "terminal_target_source"),
    ("potential_progress_reward_source", "reward_component_topology"),
    ("potential_progress_reward_source", "terminal_event_reward_source"),
]

REQUIRED_PAIR_SET = {
    ("state_dim_source", "action_dim_source"),
    ("obs_dim_source", "action_dim_source"),
    ("action_dim_source", "terminal_target_source"),
    ("state_dim_source", "terminal_target_source"),
    ("obs_dim_source", "terminal_target_source"),
    ("noise_dim_source", "zero_pad_dim_source"),
    ("activation", "action_dim_source"),
    ("activation", "terminal_target_source"),
    ("terminal_event_reward_source", "action_dim_source"),
    ("terminal_event_reward_source", "terminal_target_source"),
    ("terminal_event_reward_source", "reward_component_topology"),
    ("potential_progress_reward_source", "action_dim_source"),
    ("potential_progress_reward_source", "terminal_target_source"),
    ("potential_progress_reward_source", "reward_component_topology"),
    ("potential_progress_reward_source", "terminal_event_reward_source"),
}


HEALTH_FIELDS = [
    "done_count",
    "truncated_count",
    "input_reward_token_std",
    "raw_reward_std",
    "training_reward_std",
    "obs_per_dim_std_q90",
    "obs_per_dim_rms_mean",
    "next_state_step_l2_delta_to_rms_ratio",
    "next_state_rms",
    "raw_policy_action_unit_clip_saturation_fraction",
    "action_per_dim_temporal_std_mean",
]


def _axis_values(rows: Rows, specs: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for name, spec in specs.items():
        fn = spec["fn"]
        vals = []
        for row in rows:
            try:
                val = str(fn(row))
            except Exception:
                val = "missing"
            vals.append(val if val else "missing")
        out[name] = vals
    return out


def _coverage_status(occupied: int, expected: int, normalized_entropy: float) -> str:
    if occupied <= 1:
        return "collapsed"
    if occupied < expected:
        return "partial"
    if normalized_entropy >= 0.85:
        return "pass_uniform"
    return "covered_skewed"


def _coverage_rows(axis_values: dict[str, list[str]], specs: dict[str, dict[str, Any]]) -> Rows:
    rows: Rows = []
    for name, vals in axis_values.items():
        bins = list(specs[name]["bins"])
        counts = Counter(vals)
        expected = len(bins)
        occupied = len([b for b in bins if counts.get(b, 0) > 0])
        missing = counts.get("missing", 0)
        total = sum(counts.values())
        normalized_entropy = _normalized_entropy(Counter({b: counts.get(b, 0) for b in bins}), expected)
        row: Json = {
            "axis": name,
            "kind": specs[name]["kind"],
            "role": specs[name]["role"],
            "n": total,
            "expected_bins": expected,
            "occupied_expected_bins": occupied,
            "coverage_fraction": occupied / expected if expected else 0.0,
            "normalized_entropy": normalized_entropy,
            "max_bin_fraction": max([counts.get(b, 0) / total for b in bins] or [0.0]) if total else 0.0,
            "missing_count": missing,
            "status": _coverage_status(occupied, expected, normalized_entropy),
        }
        for b in bins:
            row[f"count:{b}"] = counts.get(b, 0)
        if missing:
            row["count:missing"] = missing
        rows.append(row)
    return rows


def _pair_rows(axis_values: dict[str, list[str]], specs: dict[str, dict[str, Any]]) -> Rows:
    rows: Rows = []
    for a, b in PAIR_SPECS:
        if a not in axis_values or b not in axis_values:
            continue
        bins_a = specs[a]["bins"]
        bins_b = specs[b]["bins"]
        pairs = list(zip(axis_values[a], axis_values[b]))
        counts = Counter(pairs)
        expected = len(bins_a) * len(bins_b)
        occupied = sum(1 for x in bins_a for y in bins_b if counts.get((x, y), 0) > 0)
        total = len(pairs)
        pair_counts = Counter({f"{x}|{y}": counts.get((x, y), 0) for x in bins_a for y in bins_b})
        ent = _normalized_entropy(pair_counts, expected)
        row: Json = {
            "axis_a": a,
            "axis_b": b,
            "role": "required_category_pair" if (a, b) in REQUIRED_PAIR_SET else "diagnostic_pair",
            "n": total,
            "expected_cells": expected,
            "occupied_expected_cells": occupied,
            "coverage_fraction": occupied / expected if expected else 0.0,
            "normalized_entropy": ent,
            "max_cell_fraction": max([counts.get((x, y), 0) / total for x in bins_a for y in bins_b] or [0.0])
            if total
            else 0.0,
            "status": _coverage_status(occupied, expected, ent),
        }
        for x in bins_a:
            for y in bins_b:
                row[f"count:{x}|{y}"] = counts.get((x, y), 0)
        rows.append(row)
    return rows


def _independence_rows(axis_values: dict[str, list[str]], specs: dict[str, dict[str, Any]]) -> Rows:
    structures = [k for k, v in specs.items() if v["kind"] == "structure"]
    modes = [k for k, v in specs.items() if v["kind"] in {"mode", "behavior"}]
    rows: Rows = []
    for s in structures:
        for m in modes:
            if s == m:
                continue
            v = _cramers_v(axis_values[s], axis_values[m])
            if not math.isfinite(v):
                level = "missing"
            elif v >= 0.50:
                level = "high"
            elif v >= 0.30:
                level = "medium"
            else:
                level = "low"
            rows.append(
                {
                    "structure_axis": s,
                    "mode_or_behavior_axis": m,
                    "structure_role": specs[s]["role"],
                    "mode_or_behavior_role": specs[m]["role"],
                    "cramers_v": v,
                    "coupling_level": level,
                    "high_coupling": bool(level == "high"),
                    "hard_independence_pair": bool(
                        specs[s]["role"] == "required_category_axis"
                        and specs[m]["role"] == "required_category_axis"
                    ),
                }
            )
    return rows


def _aggregate_update_health(progress_path: Path | None) -> dict[str, Any]:
    if progress_path is None or not progress_path.exists():
        return {"available": False, "reason": "progress jsonl not found"}
    rows = _read_jsonl(progress_path)
    metrics: dict[str, list[float]] = defaultdict(list)
    exceptions: list[Any] = []
    for row in rows:
        update = row.get("update", {}) or {}
        logger = update.get("logger", {}) or {}
        train = logger.get("train", {}) or {}
        for key in (
            "approx_kl",
            "clip_fraction",
            "explained_variance",
            "loss",
            "policy_gradient_loss",
            "value_loss",
            "entropy_loss",
        ):
            value = _safe_float(train.get(key, logger.get(f"train/{key}")))
            if math.isfinite(value):
                metrics[key].append(value)
        pd = update.get("param_delta", {}) or {}
        for key in ("param_delta_l2", "param_delta_absmax", "param_after_l2"):
            value = _safe_float(pd.get(key))
            if math.isfinite(value):
                metrics[key].append(value)
        if update.get("exception"):
            exceptions.append(update.get("exception"))
    summary: Json = {
        "available": True,
        "progress_path": str(progress_path),
        "n_updates": len(rows),
        "exceptions": exceptions,
        "exception_count": len(exceptions),
    }
    for key, vals in metrics.items():
        summary[key] = _stats(vals)
    summary["hard_health_pass"] = bool(
        not exceptions
        and (not metrics.get("approx_kl") or max(metrics["approx_kl"]) < 0.03)
        and (not metrics.get("clip_fraction") or max(metrics["clip_fraction"]) < 0.30)
        and (not metrics.get("param_delta_l2") or all(math.isfinite(v) and v > 0 for v in metrics["param_delta_l2"]))
    )
    summary["per_cell_ppo_health_available"] = False
    summary["per_cell_optimizability_available"] = False
    summary["per_cell_limitation"] = (
        "progress rows contain aggregate PPO/update metrics across the fixed group; "
        "this run directory does not include per-env pre/post train deltas."
    )
    return summary


def _cell_health_rows(rows: Rows, axis_values: dict[str, list[str]], specs: dict[str, dict[str, Any]]) -> Rows:
    out: Rows = []
    for axis, vals in axis_values.items():
        for cell in specs[axis]["bins"]:
            idxs = [i for i, val in enumerate(vals) if val == cell]
            if not idxs:
                out.append(
                    {
                        "cell_scope": "axis",
                        "axis_a": axis,
                        "cell_a": cell,
                        "n": 0,
                        "per_cell_ppo_health_available": False,
                        "per_cell_optimizability_available": False,
                    }
                )
                continue
            row: Json = {
                "cell_scope": "axis",
                "axis_a": axis,
                "cell_a": cell,
                "n": len(idxs),
                "per_cell_ppo_health_available": False,
                "per_cell_optimizability_available": False,
            }
            for field in HEALTH_FIELDS:
                stat = _stats([rows[i].get(field) for i in idxs])
                row[f"{field}_mean"] = stat["mean"]
                row[f"{field}_q50"] = stat["q50"]
            row["finite_health_fields_fraction"] = _finite_health_fraction([rows[i] for i in idxs])
            out.append(row)
    for a, b in PAIR_SPECS:
        if a not in axis_values or b not in axis_values:
            continue
        for cell_a in specs[a]["bins"]:
            for cell_b in specs[b]["bins"]:
                idxs = [
                    i
                    for i, (va, vb) in enumerate(zip(axis_values[a], axis_values[b]))
                    if va == cell_a and vb == cell_b
                ]
                if not idxs:
                    out.append(
                        {
                            "cell_scope": "pair",
                            "axis_a": a,
                            "cell_a": cell_a,
                            "axis_b": b,
                            "cell_b": cell_b,
                            "n": 0,
                            "per_cell_ppo_health_available": False,
                            "per_cell_optimizability_available": False,
                        }
                    )
                    continue
                row = {
                    "cell_scope": "pair",
                    "axis_a": a,
                    "cell_a": cell_a,
                    "axis_b": b,
                    "cell_b": cell_b,
                    "n": len(idxs),
                    "per_cell_ppo_health_available": False,
                    "per_cell_optimizability_available": False,
                    "finite_health_fields_fraction": _finite_health_fraction([rows[i] for i in idxs]),
                }
                for field in HEALTH_FIELDS:
                    stat = _stats([rows[i].get(field) for i in idxs])
                    row[f"{field}_mean"] = stat["mean"]
                    row[f"{field}_q50"] = stat["q50"]
                out.append(row)
    return out


def _finite_health_fraction(rows: Rows) -> float:
    if not rows:
        return float("nan")
    denom = len(rows) * len(HEALTH_FIELDS)
    if denom <= 0:
        return float("nan")
    ok = 0
    for row in rows:
        for field in HEALTH_FIELDS:
            ok += int(_finite(row.get(field)))
    return ok / denom


def _axis_raw_stats(rows: Rows) -> Rows:
    fields = [
        ("h:state_dim", "state_dim_source"),
        ("h:obs_dim", "obs_dim_source"),
        ("h:action_dim", "action_dim_source"),
        ("h:noise_dim", "noise_dim_source"),
        ("h:zero_pad_dim", "zero_pad_dim_source"),
        ("h:terminal_reset_count_target", "terminal_target_source"),
        ("h:reward_scale", "reward_scale_source"),
        ("h:init_state_std", "init_state_std_source"),
        ("h:state_noise_std", "state_noise_std_source"),
        ("obs_per_dim_std_q90", "obs_std"),
        ("next_state_step_l2_delta_to_rms_ratio", "dynamics_step_ratio"),
        ("input_reward_token_std", "reward_token_std"),
        ("raw_reward_std", "raw_reward_std"),
        ("done_count", "done_count"),
        ("raw_policy_action_unit_clip_saturation_fraction", "action_saturation"),
    ]
    out: Rows = []
    for field, axis in fields:
        stat = _stats([row.get(field) for row in rows])
        stat.update({"field": field, "axis": axis})
        out.append(stat)
    return out


def _audit_read(
    axis_rows: Rows,
    pair_rows: Rows,
    independence_rows: Rows,
    aggregate_health: Json,
) -> Json:
    structural_fail = [
        row
        for row in axis_rows
        if row["role"] == "required_category_axis"
        and row["kind"] == "structure"
        and row["status"] not in {"pass_uniform", "covered_skewed"}
    ]
    behavior_fail = [
        row
        for row in axis_rows
        if row["role"] == "required_category_axis"
        and row["kind"] in {"behavior", "mode"}
        and row["status"] in {"collapsed", "partial"}
    ]
    diagnostic_collapses = [
        row
        for row in axis_rows
        if row["role"] != "required_category_axis" and row["status"] in {"collapsed", "partial"}
    ]
    pair_fail = [
        row
        for row in pair_rows
        if row.get("role") == "required_category_pair" and row["status"] in {"collapsed", "partial"}
    ]
    diagnostic_pair_fail = [
        row
        for row in pair_rows
        if row.get("role") != "required_category_pair" and row["status"] in {"collapsed", "partial"}
    ]
    high_coupling = [
        row
        for row in independence_rows
        if row["coupling_level"] == "high" and bool(row.get("hard_independence_pair"))
    ]
    medium_coupling = [
        row
        for row in independence_rows
        if row["coupling_level"] == "medium" and bool(row.get("hard_independence_pair"))
    ]
    diagnostic_high_coupling = [
        row
        for row in independence_rows
        if row["coupling_level"] == "high" and not bool(row.get("hard_independence_pair"))
    ]
    diagnostic_medium_coupling = [
        row
        for row in independence_rows
        if row["coupling_level"] == "medium" and not bool(row.get("hard_independence_pair"))
    ]
    hard_source_span_pass = bool(
        not structural_fail
        and not behavior_fail
        and not pair_fail
        and not high_coupling
        and bool(aggregate_health.get("hard_health_pass"))
    )
    return {
        "category_span_complete": False,
        "category_span_complete_reason": (
            "This audit found measured coverage gaps and missing evidence axes; "
            "therefore the current h-list cannot be treated as a fully spanned category."
        ),
        "hard_source_span_pass": hard_source_span_pass,
        "hard_source_span_pass_reason": (
            "Required source axes, required source pairs, hard structure-mode independence, "
            "and aggregate PPO health all pass. This does not certify missing evidence axes."
        ),
        "first_order_structural_fail_count": len(structural_fail),
        "first_order_behavior_or_mode_fail_count": len(behavior_fail),
        "diagnostic_not_required_collapse_count": len(diagnostic_collapses),
        "second_order_fail_count": len(pair_fail),
        "diagnostic_second_order_fail_count": len(diagnostic_pair_fail),
        "high_structure_mode_coupling_count": len(high_coupling),
        "medium_structure_mode_coupling_count": len(medium_coupling),
        "diagnostic_high_structure_mode_coupling_count": len(diagnostic_high_coupling),
        "diagnostic_medium_structure_mode_coupling_count": len(diagnostic_medium_coupling),
        "aggregate_ppo_health_pass": bool(aggregate_health.get("hard_health_pass")),
        "per_cell_ppo_health_available": bool(aggregate_health.get("per_cell_ppo_health_available")),
        "per_cell_optimizability_available": bool(aggregate_health.get("per_cell_optimizability_available")),
        "implementation_should_wait": True,
        "implementation_should_wait_reason": (
            "The task is category-span diagnosis. New family implementation is not licensed "
            "until missing axes and measured coupling/coverage gaps are resolved."
        ),
        "top_structural_failures": sorted(
            structural_fail,
            key=lambda r: (float(r["coverage_fraction"]), float(r["normalized_entropy"])),
        )[:10],
        "top_behavior_or_mode_failures": sorted(
            behavior_fail,
            key=lambda r: (float(r["coverage_fraction"]), float(r["normalized_entropy"])),
        )[:10],
        "top_second_order_failures": sorted(
            pair_fail,
            key=lambda r: (float(r["coverage_fraction"]), float(r["normalized_entropy"])),
        )[:12],
        "top_high_couplings": sorted(
            high_coupling,
            key=lambda r: -float(r["cramers_v"] if math.isfinite(float(r["cramers_v"])) else 0.0),
        )[:12],
        "top_diagnostic_high_couplings": sorted(
            diagnostic_high_coupling,
            key=lambda r: -float(r["cramers_v"] if math.isfinite(float(r["cramers_v"])) else 0.0),
        )[:12],
        "diagnostic_not_required_collapses": sorted(
            diagnostic_collapses,
            key=lambda r: (float(r["coverage_fraction"]), float(r["normalized_entropy"])),
        )[:12],
        "diagnostic_second_order_failures": sorted(
            diagnostic_pair_fail,
            key=lambda r: (float(r["coverage_fraction"]), float(r["normalized_entropy"])),
        )[:12],
    }


def _markdown(report: Json) -> str:
    read = report["read"]
    lines = [
        "# Category Span Audit",
        "",
        "Read-only audit from fixed h-list plus true-runner sidecar. It separates span/uniformity from generator design.",
        "",
        "## Read",
        "",
        f"- category span complete: `{read['category_span_complete']}`",
        f"- hard source-span pass: `{read['hard_source_span_pass']}`",
        f"- aggregate PPO health pass: `{read['aggregate_ppo_health_pass']}`",
        f"- per-cell PPO health available: `{read['per_cell_ppo_health_available']}`",
        f"- per-cell optimizability available: `{read['per_cell_optimizability_available']}`",
        f"- structural first-order failures: `{read['first_order_structural_fail_count']}`",
        f"- behavior/mode first-order failures: `{read['first_order_behavior_or_mode_fail_count']}`",
        f"- diagnostic-only collapses: `{read['diagnostic_not_required_collapse_count']}`",
        f"- second-order failures: `{read['second_order_fail_count']}`",
        f"- diagnostic second-order failures: `{read['diagnostic_second_order_fail_count']}`",
        f"- high structure-mode coupling pairs: `{read['high_structure_mode_coupling_count']}`",
        f"- medium structure-mode coupling pairs: `{read['medium_structure_mode_coupling_count']}`",
        f"- diagnostic high coupling pairs: `{read['diagnostic_high_structure_mode_coupling_count']}`",
        f"- diagnostic medium coupling pairs: `{read['diagnostic_medium_structure_mode_coupling_count']}`",
        "",
        "## First-Order Coverage",
        "",
        "| axis | kind | role | status | occupied/expected | entropy | max bin |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for row in report["axis_coverage"]:
        lines.append(
            f"| `{row['axis']}` | `{row['kind']}` | `{row['role']}` | "
            f"`{row['status']}` | "
            f"{row['occupied_expected_bins']}/{row['expected_bins']} | "
            f"{row['normalized_entropy']:.3f} | {row['max_bin_fraction']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Key Second-Order Coverage",
            "",
        "| axis A | axis B | role | status | occupied/expected | entropy | max cell |",
        "|---|---|---|---|---:|---:|---:|",
        ]
    )
    for row in sorted(report["pair_coverage"], key=lambda r: (r["coverage_fraction"], r["normalized_entropy"])):
        lines.append(
            f"| `{row['axis_a']}` | `{row['axis_b']}` | `{row['role']}` | `{row['status']}` | "
            f"{row['occupied_expected_cells']}/{row['expected_cells']} | "
            f"{row['normalized_entropy']:.3f} | {row['max_cell_fraction']:.3f} |"
        )
    lines.extend(["", "## Missing Evidence Axes", ""])
    for row in report["missing_evidence_axes"]:
        lines.append(f"- `{row['axis']}`: {row['reason']}")
    if read["top_high_couplings"]:
        lines.extend(["", "## High Coupling", ""])
        for row in read["top_high_couplings"]:
            lines.append(
                f"- `{row['structure_axis']}` vs `{row['mode_or_behavior_axis']}`: V={row['cramers_v']:.3f}"
            )
    if read["top_diagnostic_high_couplings"]:
        lines.extend(["", "## Diagnostic High Coupling", ""])
        for row in read["top_diagnostic_high_couplings"]:
            lines.append(
                f"- `{row['structure_axis']}` vs `{row['mode_or_behavior_axis']}` "
                f"({row['mode_or_behavior_role']}): V={row['cramers_v']:.3f}"
            )
    lines.extend(["", "## Outputs", ""])
    for key, value in report["outputs"].items():
        lines.append(f"- `{key}`: `{value}`")
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> Json:
    run_dir = Path(args.run_dir).expanduser().resolve()
    h_list_path = Path(args.h_list).expanduser().resolve() if args.h_list else run_dir / "fixed_env_group/prior_fixed_h_list.json"
    sidecar_path = Path(args.sidecar).expanduser().resolve() if args.sidecar else _latest_sidecar(run_dir)
    progress_path = Path(args.progress).expanduser().resolve() if args.progress else _latest_progress(run_dir)

    h_rows = _read_json(h_list_path)
    side_rows = _read_jsonl(sidecar_path)
    if not isinstance(h_rows, list):
        raise TypeError(f"h-list must be a JSON list: {h_list_path}")
    joined = _join_rows(h_rows, side_rows)
    specs = _axis_specs()
    axis_vals = _axis_values(joined, specs)
    axis_rows = _coverage_rows(axis_vals, specs)
    pair_rows = _pair_rows(axis_vals, specs)
    independence = _independence_rows(axis_vals, specs)
    cell_health = _cell_health_rows(joined, axis_vals, specs)
    raw_stats = _axis_raw_stats(joined)
    aggregate_health = _aggregate_update_health(progress_path)
    read = _audit_read(axis_rows, pair_rows, independence, aggregate_health)

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": out_dir / "category_span_audit.json",
        "markdown": out_dir / "category_span_audit.md",
        "axis_coverage_csv": out_dir / "category_span_axis_coverage.csv",
        "pair_coverage_csv": out_dir / "category_span_pair_coverage.csv",
        "independence_csv": out_dir / "category_span_structure_mode_independence.csv",
        "cell_health_csv": out_dir / "category_span_cell_health.csv",
        "raw_stats_csv": out_dir / "category_span_raw_stats.csv",
    }
    report: Json = {
        "analysis_entry": "phase2_category_span_audit",
        "schema": "phase2_category_span_audit.v1",
        "contract": {
            "read_only": True,
            "does_not_sample_environments": True,
            "does_not_train": True,
            "does_not_modify_generator": True,
            "does_not_modify_ppo": True,
            "proof_and_implementation_separated": True,
            "category_span_before_new_family": True,
        },
        "inputs": {
            "run_dir": str(run_dir),
            "h_list": str(h_list_path),
            "sidecar": str(sidecar_path),
            "progress": str(progress_path) if progress_path else None,
            "h_count": len(h_rows),
            "sidecar_count": len(side_rows),
            "joined_count": len(joined),
        },
        "axis_coverage": axis_rows,
        "pair_coverage": pair_rows,
        "structure_mode_independence": independence,
        "cell_health_summary_rows": cell_health,
        "raw_stats": raw_stats,
        "aggregate_update_health": aggregate_health,
        "missing_evidence_axes": MISSING_EVIDENCE_AXES,
        "read": read,
        "outputs": {k: str(v) for k, v in paths.items()},
    }

    paths["json"].write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths["markdown"].write_text(_markdown(report), encoding="utf-8")
    _write_csv(paths["axis_coverage_csv"], axis_rows)
    _write_csv(paths["pair_coverage_csv"], pair_rows)
    _write_csv(paths["independence_csv"], independence)
    _write_csv(paths["cell_health_csv"], cell_health)
    _write_csv(paths["raw_stats_csv"], raw_stats)
    print(json.dumps({k: str(v) for k, v in paths.items()}, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--h-list", default=None)
    parser.add_argument("--sidecar", default=None)
    parser.add_argument("--progress", default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
