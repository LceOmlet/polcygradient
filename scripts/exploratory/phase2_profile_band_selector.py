#!/usr/bin/env python
"""Unified profile-band selector for exploratory prior fixed lists.

This is the reductive replacement for one-wrapper-per-mode exploration.  It
selects fixed environments using one profile contract:

* marginal bands for trusted axes,
* a joint-profile cell cap,
* explicit detector trust/version metadata,
* health checks left to the canonical sidecar smoke runner.

Exploratory only.  The selector materializes fixed-list artifacts from existing
frozen environments; it does not sample new environments, train PPO, or modify
exact SCM / fit_model / PPO paths.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


ART = Path("/home/chen/RLPFN/artifacts")
DEFAULT_SELECTED_CSV = (
    ART / "phase2_exploratory_reward_group_balance_gated_n128_seed6262_0510/selected_prior_envs.csv"
)
DEFAULT_LOCOMOTION_CSV = (
    ART / "phase2_locomotion_full_temporal_detector_gated_n128_gpu0_0510/locomotion_full_temporal_per_env.csv"
)
DEFAULT_ACTION_MODE_JSON = (
    ART / "phase2_prior_action_mode_coverage_gated_n128_gpu0_0510/gated_n128_action_mode_coverage.json"
)
DEFAULT_STRICT_FEEDBACK_CSV = (
    ART
    / "phase2_pendulum_like_strict_cross_seed6260_6261_6262_all_candidates_0510"
    / "pendulum_like_signature_fixedlist.csv"
)
DEFAULT_FEEDBACK_NEAR_BEST_CSV = ""
DEFAULT_FUTURE_BASIN_CACHED_LABELS_CSV = ""
DEFAULT_OUTPUT_DIR = ART / "phase2_profile_band_selector_v1_push2_equiv_0511"

RULES = ("gym_full_obs_action_range", "gym_q90_obs_action_range")
MODE_KEYS = (
    "locomotion_witness",
    "low_energy_not_ctrl_only",
    "sign_or_phase_sensitive",
    "state_conditioned_energy_injection",
)

BASELINE_RATES = {
    "gym_full_obs_action_range": {
        "locomotion_witness": 0.078125,
        "low_energy_not_ctrl_only": 0.53125,
        "sign_or_phase_sensitive": 0.65625,
        "state_conditioned_energy_injection": 0.453125,
    },
    "gym_q90_obs_action_range": {
        "locomotion_witness": 0.171875,
        "low_energy_not_ctrl_only": 0.421875,
        "sign_or_phase_sensitive": 0.671875,
        "state_conditioned_energy_injection": 0.578125,
    },
}

DEFAULT_PROFILE_CONFIG = {
    "profile_version_id": "profile_band_v1_locomotion_raised_feedback_logged",
    "background_milestone": "gated_reward_path_balance",
    "target_per_rule": 64,
    "joint_profile_cell_cap_rate": 0.25,
    "axes": {
        "locomotion_witness": {
            "trust_status": "quota_ready_only_with_joint_profile_guard",
            "role": "raised_band",
            "min_rate": 0.296875,
            "max_rate": 0.296875,
            "objective_weight": 1.0,
        },
        "low_energy_not_ctrl_only": {
            "trust_status": "quota_ready_protected_band",
            "role": "protected_band",
            "baseline_tolerance": 0.035,
        },
        "sign_or_phase_sensitive": {
            "trust_status": "quota_ready_protected_band_but_not_sufficient",
            "role": "protected_band",
            "baseline_tolerance": 0.035,
        },
        "state_conditioned_energy_injection": {
            "trust_status": "diagnostic_broad_axis",
            "role": "protected_band",
            "baseline_tolerance": 0.035,
        },
    },
    "strict_feedback_subprofile_branch": {
        "enabled": False,
        "trust_status": "candidate_branch_control_only",
        "reason": "disabled in the default selector until explicitly requested by a candidate config",
    },
    "pendulum_settle_yield_repair_branch": {
        "enabled": False,
        "trust_status": "disabled_by_default_prior_internal_repair_only",
        "reason": (
            "disabled in the default selector; when enabled, selected frozen h rows are patched with "
            "the prior-internal settle-yield repair and then audited by cached guards/smoke"
        ),
    },
    "prior_pack_optimizability_guard": {
        "enabled": False,
        "trust_status": "disabled_by_default_requires_same_runner_evidence",
        "reason": "disabled by default; candidate configs may require a low-terminal/action-coupled PPO surface",
    },
    "logged_not_quota_axes": {
        "context_identifiable_closed_loop_settle_feedback": {
            "trust_status": "pressure_point_subprofile_not_single_quota",
            "reason": "strict feedback profile/gain dominance needs subprofile control, not broad count",
        },
        "mountaincar_upfront_investment_future_basin": {
            "trust_status": "missing_detector",
            "reason": "future-basin detector is not quota-ready",
        },
    },
}

MODE_RE = re.compile(r"^(?P<profile>early_|decay_)?obs_linear(?P<neg>_neg)?_(?P<feature>\d+)x(?P<gain>\d+)")
_FROZEN_H_CACHE: dict[str, Any] = {}


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


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n", "", "none", "nan"}:
        return False
    return bool(value)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).expanduser().resolve().open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def _read_frozen_h(path: str | Path) -> dict[str, Any]:
    resolved = str(Path(path).expanduser().resolve())
    if resolved not in _FROZEN_H_CACHE:
        payload = _read_json(resolved)
        if not isinstance(payload, dict):
            raise TypeError(f"frozen h must be a JSON object: {resolved}")
        _FROZEN_H_CACHE[resolved] = payload
    return dict(_FROZEN_H_CACHE[resolved])


def _safe_label(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")[:160] or "rule"


def _frozen_h_digest(path: str | Path) -> str:
    payload = _read_frozen_h(path)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _frozen_h_payload_digest(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _parse_feedback_mode(mode: Any) -> dict[str, Any]:
    text = "" if mode is None else str(mode)
    match = MODE_RE.match(text)
    if not match:
        return {
            "strict_feedback_profile": "unknown" if text else "none",
            "strict_feedback_gain": math.nan,
            "strict_feedback_sign": "unknown" if text else "none",
            "strict_feedback_feature": None,
            "strict_feedback_mode": text,
        }
    raw_profile = match.group("profile") or ""
    profile = "constant"
    if raw_profile == "early_":
        profile = "early_then_zero"
    elif raw_profile == "decay_":
        profile = "decay"
    return {
        "strict_feedback_profile": profile,
        "strict_feedback_gain": float(match.group("gain")) / 100.0,
        "strict_feedback_sign": "neg" if match.group("neg") else "pos",
        "strict_feedback_feature": int(match.group("feature")),
        "strict_feedback_mode": text,
    }


def _load_strict_feedback_labels(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    csv_path = Path(path).expanduser().resolve()
    if not csv_path.exists():
        return {}
    labels: dict[str, dict[str, Any]] = {}
    for row in _read_csv(csv_path):
        source_tag = str(row.get("source_tag", ""))
        if source_tag == "seed6262_full":
            rule = "gym_full_obs_action_range"
        elif source_tag == "seed6262_q90":
            rule = "gym_q90_obs_action_range"
        else:
            continue
        digest = _frozen_h_digest(row["full_frozen_h_json"])
        mode = row.get("effective_feedback_mode") or row.get("best_feedback_mode")
        parsed = _parse_feedback_mode(mode)
        labels[digest] = {
            "strict_feedback_candidate": True,
            "strict_feedback_rule": rule,
            "strict_feedback_source_tag": source_tag,
            "strict_feedback_source_env_index": row.get("env_index"),
            "strict_feedback_source_pool_definition": row.get(
                "feedback_source_pool_definition", "pendulum_like_strict"
            ),
            "strict_feedback_effective_source": row.get("effective_feedback_source", "best_feedback"),
            "strict_feedback_signature_score": _safe_float(row.get("signature_score")),
            "strict_feedback_pair_best_sign": row.get("feedback_pair_best_sign"),
            "strict_feedback_best_minus_open_env": _safe_float(
                row.get("effective_feedback_minus_open_env", row.get("best_feedback_minus_open_env"))
            ),
            **parsed,
        }
    return labels


def _load_action_labels(path: str | Path) -> dict[tuple[str, int], dict[str, bool]]:
    report = _read_json(path)
    labels: dict[tuple[str, int], dict[str, bool]] = {}
    for item in report.get("lists", []):
        label = str(item.get("label", ""))
        for row in item.get("classification", {}).get("per_env", []):
            idx = int(row["env_index"])
            labels[(label, idx)] = {
                "low_energy_not_ctrl_only": bool(row.get("low_energy_not_ctrl_only", False)),
                "sign_or_phase_sensitive": bool(row.get("sign_or_phase_sensitive_present", False)),
                "state_conditioned_energy_injection": bool(
                    row.get("state_conditioned_energy_injection_witness_present", False)
                ),
            }
    return labels


def _load_locomotion_labels(path: str | Path) -> dict[tuple[str, int], dict[str, Any]]:
    labels: dict[tuple[str, int], dict[str, Any]] = {}
    for row in _read_csv(path):
        scope = str(row.get("scope", ""))
        idx = int(float(row["env_index"]))
        labels[(scope, idx)] = {
            "locomotion_witness": _safe_bool(row.get("sustained_locomotion_temporal_witness")),
            "locomotion_env_gain_vs_zero": _safe_float(row.get("env_reward_gain_vs_zero")),
        }
    return labels


def _load_feedback_near_best_labels(path: str | Path | None) -> dict[tuple[str, int], dict[str, Any]]:
    if not path:
        return {}
    csv_path = Path(path).expanduser().resolve()
    if not csv_path.exists():
        return {}
    grouped: dict[tuple[str, int], list[dict[str, str]]] = {}
    for row in _read_csv(csv_path):
        rule = str(row.get("rule", ""))
        if rule not in RULES:
            continue
        source_index = int(float(row.get("source_index", row.get("env_idx", 0))))
        grouped.setdefault((rule, source_index), []).append(row)

    profile_priority = {"decay": 0, "early_then_zero": 1, "constant": 2}
    gain_priority = {1.0: 0, 0.5: 1}
    labels: dict[tuple[str, int], dict[str, Any]] = {}
    for key, rows in grouped.items():
        supports: list[tuple[int, int, float, dict[str, str], float]] = []
        for row in rows:
            profile = str(row.get("profile", "unknown"))
            for gain in (1.0, 0.5):
                gain_text = f"{gain:g}"
                if not _safe_bool(row.get(f"near_best_gain_{gain_text}")):
                    continue
                gap = _safe_float(row.get(f"gap_gain_{gain_text}"), default=0.0)
                supports.append(
                    (
                        profile_priority.get(profile, 9),
                        gain_priority.get(gain, 9),
                        gap,
                        row,
                        gain,
                    )
                )
        if not supports:
            labels[key] = {
                "strict_feedback_candidate": False,
                "feedback_near_best_support_set": False,
                "feedback_near_best_support_scope": "none",
            }
            continue
        supports.sort(key=lambda item: (item[0], item[1], item[2]))
        _profile_rank, _gain_rank, gap, chosen, gain = supports[0]
        profile = str(chosen.get("profile", "unknown"))
        labels[key] = {
            "strict_feedback_candidate": True,
            "strict_feedback_source_pool_definition": "near_best_support_set",
            "strict_feedback_effective_source": "feedback_near_best_medium_low_gain",
            "strict_feedback_profile": profile,
            "strict_feedback_gain": float(gain),
            "strict_feedback_sign": "support_set",
            "strict_feedback_feature": None,
            "strict_feedback_mode": f"near_best_{profile}_gain_{gain:g}",
            "feedback_near_best_support_set": True,
            "feedback_near_best_support_scope": "all_rows_or_strict_when_available",
            "feedback_near_best_chosen_gap": float(gap),
            "feedback_near_best_chosen_best_mode": chosen.get("best_mode", ""),
        }
    return labels


def _load_future_basin_cached_labels(path: str | Path | None) -> dict[tuple[str, int], dict[str, Any]]:
    if not path:
        return {}
    csv_path = Path(path).expanduser().resolve()
    if not csv_path.exists():
        return {}
    labels: dict[tuple[str, int], dict[str, Any]] = {}
    for row in _read_csv(csv_path):
        rule = str(row.get("rule", ""))
        if rule not in RULES:
            continue
        source_index = int(float(row.get("source_index", row.get("env_index", row.get("env_idx", 0)))))
        legacy_guard = _safe_bool(row.get("pendulum_short_temporal_settle_guard"))
        has_split_fields = any(
            key in row
            for key in (
                "pendulum_axis_support",
                "pendulum_short_gap",
                "pendulum_medium_settle",
                "pendulum_strict_settle",
                "pendulum_sustained_settle",
                "pendulum_axis_supported_sustained_settle",
                "pendulum_profile_quota_ready",
            )
        )
        sustained = _safe_bool(row.get("pendulum_sustained_settle")) or _safe_bool(
            row.get("pendulum_medium_settle")
        ) or (
            legacy_guard and not has_split_fields
        )
        labels[(rule, source_index)] = {
            "pendulum_axis_support": _safe_bool(row.get("pendulum_axis_support")) or (
                legacy_guard and not has_split_fields
            ),
            "pendulum_short_gap": _safe_bool(row.get("pendulum_short_gap")),
            "pendulum_medium_settle": _safe_bool(row.get("pendulum_medium_settle")),
            "pendulum_strict_settle": _safe_bool(row.get("pendulum_strict_settle")) or (
                legacy_guard and not has_split_fields
            ),
            "pendulum_sustained_settle": sustained,
            "pendulum_axis_supported_sustained_settle": _safe_bool(
                row.get("pendulum_axis_supported_sustained_settle")
            ),
            "pendulum_profile_quota_ready": _safe_bool(row.get("pendulum_profile_quota_ready")),
            "pendulum_label_available": _safe_bool(row.get("pendulum_label_available")),
            "pendulum_short_temporal_settle_guard": _safe_bool(row.get("pendulum_strict_settle")) or (
                legacy_guard and not has_split_fields
            ),
            "pendulum_settle_guard_precision_class": row.get("pendulum_settle_guard_precision_class", ""),
            "pendulum_axis_cheap_label_source": row.get("pendulum_axis_cheap_label_source", ""),
            "mountaincar_label_available": _safe_bool(row.get("mountaincar_label_available")),
            "mountaincar_short_future_basin_guard": _safe_bool(
                row.get("mountaincar_short_future_basin_guard")
            ),
            "mountaincar_delayed_future_basin_guard": _safe_bool(
                row.get("mountaincar_delayed_future_basin_guard")
            ),
            "mountaincar_upfront_action_decay": _safe_bool(row.get("mountaincar_upfront_action_decay")),
            "mountaincar_label_use": row.get("mountaincar_label_use", ""),
            "future_basin_cached_label_available": True,
        }
    return labels


def _merge_rows(
    selected_rows: list[dict[str, str]],
    loco_labels: dict[tuple[str, int], dict[str, Any]],
    action_labels: dict[tuple[str, int], dict[str, bool]],
    strict_feedback_labels: dict[str, dict[str, Any]] | None = None,
    feedback_near_best_labels: dict[tuple[str, int], dict[str, Any]] | None = None,
    future_basin_cached_labels: dict[tuple[str, int], dict[str, Any]] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    by_rule: dict[str, list[dict[str, Any]]] = {rule: [] for rule in RULES}
    strict_feedback_labels = strict_feedback_labels or {}
    feedback_near_best_labels = feedback_near_best_labels or {}
    future_basin_cached_labels = future_basin_cached_labels or {}
    for row in selected_rows:
        rule = str(row.get("rule"))
        if rule not in by_rule:
            continue
        idx = len(by_rule[rule])
        merged: dict[str, Any] = dict(row)
        merged["_source_index"] = int(idx)
        merged.update(loco_labels.get((rule, idx), {}))
        merged.update(action_labels.get((rule, idx), {}))
        frozen_h = _read_frozen_h(row["full_frozen_h_json"])
        merged["terminal_reset_count_target"] = _safe_float(frozen_h.get("terminal_reset_count_target"))
        digest = _frozen_h_payload_digest(frozen_h)
        if digest in strict_feedback_labels:
            merged.update(strict_feedback_labels[digest])
        else:
            merged["strict_feedback_candidate"] = False
        merged.update(feedback_near_best_labels.get((rule, idx), {}))
        merged.update(future_basin_cached_labels.get((rule, idx), {}))
        by_rule[rule].append(merged)
    return by_rule


def _count_band(rate: float, target_n: int, tolerance: float = 0.0) -> tuple[int, int]:
    low = max(0, math.ceil((rate - tolerance) * target_n - 1e-9))
    high = min(target_n, math.floor((rate + tolerance) * target_n + 1e-9))
    return int(low), int(high)


def _cell(row: dict[str, Any]) -> tuple[bool, ...]:
    return tuple(_safe_bool(row.get(key)) for key in MODE_KEYS)


def _numeric_stats(values: list[float]) -> dict[str, Any]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"count": 0, "mean": None, "min": None, "max": None}
    return {
        "count": int(len(finite)),
        "mean": float(np.mean(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def _summary(rows: list[dict[str, Any]], *, rule: str) -> dict[str, Any]:
    cells = Counter(_cell(row) for row in rows)
    dominant = max(cells.values()) if cells else 0
    total = len(rows)
    entropy = 0.0
    for count in cells.values():
        p = count / max(1, total)
        entropy -= p * math.log(p + 1e-12)
    entropy_norm = entropy / math.log(2 ** len(MODE_KEYS)) if total else 0.0
    out: dict[str, Any] = {
        "rule": rule,
        "n": total,
        "joint_profile_active_cells": len(cells),
        "joint_profile_dominant_count": dominant,
        "joint_profile_dominant_rate": dominant / max(1, total),
        "joint_profile_entropy_norm": entropy_norm,
        "top_joint_profiles": [
            {
                "cell": "".join("1" if bit else "0" for bit in cell),
                "profile": {key: bit for key, bit in zip(MODE_KEYS, cell)},
                "count": count,
                "rate": count / max(1, total),
            }
            for cell, count in cells.most_common(8)
        ],
    }
    for key in MODE_KEYS:
        out[f"{key}_count"] = int(sum(_safe_bool(row.get(key)) for row in rows))
        out[f"{key}_rate"] = out[f"{key}_count"] / max(1, total)
    terminal_values = [_safe_float(row.get("terminal_reset_count_target")) for row in rows]
    action_gain_values = [
        _safe_float(row.get("balanced_reward_action_input_gain_fraction")) for row in rows
    ]
    out["terminal_reset_count_target"] = _numeric_stats(terminal_values)
    out["terminal_reset_count_target_gt20_count"] = int(sum(value > 20.0 for value in terminal_values))
    out["terminal_reset_count_target_gt20_rate"] = (
        out["terminal_reset_count_target_gt20_count"] / max(1, total)
    )
    out["balanced_reward_action_input_gain_fraction"] = _numeric_stats(action_gain_values)
    strict_rows = [row for row in rows if _safe_bool(row.get("strict_feedback_candidate"))]
    out["strict_feedback_count"] = len(strict_rows)
    out["strict_feedback_rate"] = len(strict_rows) / max(1, total)
    out["strict_feedback_profile_counts"] = dict(Counter(str(row.get("strict_feedback_profile")) for row in strict_rows))
    out["strict_feedback_gain_counts"] = dict(Counter(str(row.get("strict_feedback_gain")) for row in strict_rows))
    logged_future_keys = (
        "pendulum_label_available",
        "pendulum_axis_support",
        "pendulum_short_gap",
        "pendulum_medium_settle",
        "pendulum_strict_settle",
        "pendulum_sustained_settle",
        "pendulum_axis_supported_sustained_settle",
        "pendulum_profile_quota_ready",
        "pendulum_short_temporal_settle_guard",
        "mountaincar_label_available",
        "mountaincar_upfront_action_decay",
        "mountaincar_short_future_basin_guard",
        "mountaincar_delayed_future_basin_guard",
    )
    for key in logged_future_keys:
        out[f"{key}_count"] = int(sum(_safe_bool(row.get(key)) for row in rows))
        out[f"{key}_rate"] = out[f"{key}_count"] / max(1, total)
    repair_count = int(sum(_safe_bool(row.get("pendulum_settle_yield_repair_planned")) for row in rows))
    effective_settle_count = int(
        sum(
            _safe_bool(row.get("pendulum_short_temporal_settle_guard"))
            or _safe_bool(row.get("pendulum_axis_support"))
            or _safe_bool(row.get("pendulum_settle_yield_repair_planned"))
            for row in rows
        )
    )
    out["pendulum_settle_yield_repair_planned_count"] = repair_count
    out["pendulum_settle_yield_repair_planned_rate"] = repair_count / max(1, total)
    out["pendulum_settle_effective_support_count"] = effective_settle_count
    out["pendulum_settle_effective_support_rate"] = effective_settle_count / max(1, total)
    return out


def _load_profile_config(path: str | None) -> dict[str, Any]:
    if path:
        cfg = _read_json(path)
        if not isinstance(cfg, dict):
            raise TypeError(f"profile config must be a JSON object: {path}")
        return cfg
    return json.loads(json.dumps(DEFAULT_PROFILE_CONFIG))


def _constraint_for_axis(
    *,
    config: dict[str, Any],
    rule: str,
    key: str,
    target_n: int,
) -> tuple[int, int, str]:
    spec = config["axes"][key]
    role = str(spec.get("role"))
    if role == "raised_band":
        return (
            math.ceil(float(spec["min_rate"]) * target_n - 1e-9),
            math.floor(float(spec["max_rate"]) * target_n + 1e-9),
            role,
        )
    if role == "protected_band":
        baseline = BASELINE_RATES[rule][key]
        tolerance = float(spec.get("baseline_tolerance", 0.0))
        low, high = _count_band(baseline, target_n, tolerance)
        return low, high, role
    raise ValueError(f"Unsupported axis role for {key}: {role}")


def _add_strict_feedback_constraints(
    *,
    rows: list[dict[str, Any]],
    rule: str,
    target_n: int,
    config: dict[str, Any],
    a_rows: list[np.ndarray],
    lb: list[float],
    ub: list[float],
    details: dict[str, Any],
) -> np.ndarray:
    spec = config.get("strict_feedback_subprofile_branch", {})
    strict = np.asarray([1.0 if _safe_bool(row.get("strict_feedback_candidate")) else 0.0 for row in rows])
    if not bool(spec.get("enabled", False)):
        return strict
    exact_counts = spec.get("exact_count_by_rule", {})
    min_counts = spec.get("min_count_by_rule", {})
    max_counts = spec.get("max_count_by_rule", {})
    if rule in exact_counts:
        low = high = int(exact_counts[rule])
    else:
        low = int(min_counts.get(rule, math.ceil(float(spec.get("min_rate", 0.0)) * target_n - 1e-9)))
        high = int(max_counts.get(rule, math.floor(float(spec.get("max_rate", 1.0)) * target_n + 1e-9)))
    a_rows.append(strict)
    lb.append(float(low))
    ub.append(float(high))

    strict_nonconstant = np.asarray(
        [
            1.0
            if _safe_bool(row.get("strict_feedback_candidate"))
            and str(row.get("strict_feedback_profile")) != "constant"
            else 0.0
            for row in rows
        ]
    )
    strict_nongain2 = np.asarray(
        [
            1.0
            if _safe_bool(row.get("strict_feedback_candidate"))
            and str(row.get("strict_feedback_gain")) != "2.0"
            else 0.0
            for row in rows
        ]
    )
    min_nonconstant_by_rule = spec.get("min_nonconstant_by_rule", {})
    min_non_gain2_by_rule = spec.get("min_non_gain2_by_rule", {})
    if bool(spec.get("include_all_nonconstant", True)):
        nonconstant_low = int(strict_nonconstant.sum())
    else:
        nonconstant_low = int(min_nonconstant_by_rule.get(rule, spec.get("min_nonconstant_count", 0)))
    if bool(spec.get("include_all_non_gain2", True)):
        nongain2_low = int(strict_nongain2.sum())
    else:
        nongain2_low = int(min_non_gain2_by_rule.get(rule, spec.get("min_non_gain2_count", 0)))
    if nonconstant_low > int(strict_nonconstant.sum()):
        raise ValueError(
            f"{rule}: requested {nonconstant_low} strict nonconstant rows, "
            f"but only {int(strict_nonconstant.sum())} are available"
        )
    if nongain2_low > int(strict_nongain2.sum()):
        raise ValueError(
            f"{rule}: requested {nongain2_low} strict non-gain2 rows, "
            f"but only {int(strict_nongain2.sum())} are available"
        )
    a_rows.append(strict_nonconstant)
    lb.append(float(nonconstant_low))
    ub.append(float(strict_nonconstant.sum()))
    a_rows.append(strict_nongain2)
    lb.append(float(nongain2_low))
    ub.append(float(strict_nongain2.sum()))

    details["strict_feedback_subprofile_branch"] = {
        "enabled": True,
        "trust_status": spec.get("trust_status"),
        "strict_min_count": low,
        "strict_max_count": high,
        "source_strict_count": int(strict.sum()),
        "source_nonconstant_count": int(strict_nonconstant.sum()),
        "source_non_gain2_count": int(strict_nongain2.sum()),
        "nonconstant_min_count": nonconstant_low,
        "non_gain2_min_count": nongain2_low,
    }
    return strict


def _count_from_rule_spec(
    spec: dict[str, Any],
    *,
    rule: str,
    key: str,
    rate_key: str,
    target_n: int,
    default: int,
) -> int:
    by_rule = spec.get(f"{key}_by_rule", {})
    if rule in by_rule:
        return max(0, min(int(target_n), int(by_rule[rule])))
    if key in spec:
        return max(0, min(int(target_n), int(spec[key])))
    rate_by_rule = spec.get(f"{rate_key}_by_rule", {})
    if rule in rate_by_rule:
        rate = float(rate_by_rule[rule])
        return max(0, min(int(target_n), int(math.ceil(rate * target_n - 1e-9))))
    if rate_key in spec:
        rate = float(spec[rate_key])
        return max(0, min(int(target_n), int(math.ceil(rate * target_n - 1e-9))))
    return max(0, min(int(target_n), int(default)))


def _float_from_rule_spec(
    spec: dict[str, Any],
    *,
    rule: str,
    key: str,
    default: float,
) -> float:
    by_rule = spec.get(f"{key}_by_rule", {})
    if rule in by_rule:
        return float(by_rule[rule])
    if key in spec:
        return float(spec[key])
    return float(default)


def _list_from_rule_spec(
    spec: dict[str, Any],
    *,
    rule: str,
    key: str,
    default: list[Any],
) -> list[Any]:
    by_rule = spec.get(f"{key}_by_rule", {})
    if rule in by_rule:
        value = by_rule[rule]
    else:
        value = spec.get(key, default)
    if not isinstance(value, list):
        raise TypeError(f"{key} for {rule} must be a list")
    return list(value)


def _dict_from_rule_spec(
    spec: dict[str, Any],
    *,
    rule: str,
    key: str,
    default: dict[str, Any],
) -> dict[str, Any]:
    by_rule = spec.get(f"{key}_by_rule", {})
    if rule in by_rule:
        value = by_rule[rule]
    else:
        value = spec.get(key, default)
    if not isinstance(value, dict):
        raise TypeError(f"{key} for {rule} must be a dict")
    return dict(value)


def _terminal_bin_indicator(terminals: np.ndarray, *, lo: float, hi: float, first: bool) -> np.ndarray:
    if first:
        return ((terminals >= lo) & (terminals <= hi)).astype(np.float64)
    return ((terminals > lo) & (terminals <= hi)).astype(np.float64)


def _add_prior_pack_optimizability_constraints(
    *,
    rows: list[dict[str, Any]],
    rule: str,
    target_n: int,
    config: dict[str, Any],
    a_rows: list[np.ndarray],
    lb: list[float],
    ub: list[float],
    details: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    spec = config.get("prior_pack_optimizability_guard", {}) or {}
    terminals = np.asarray([_safe_float(row.get("terminal_reset_count_target")) for row in rows])
    action_gains = np.asarray(
        [_safe_float(row.get("balanced_reward_action_input_gain_fraction")) for row in rows]
    )
    terminal_max = float(spec.get("terminal_reset_count_target_max", 20.0))
    action_min = _float_from_rule_spec(
        spec,
        rule=rule,
        key="balanced_action_gain_min",
        default=0.18,
    )
    low_terminal = (terminals <= terminal_max).astype(np.float64)
    high_terminal = (terminals > terminal_max).astype(np.float64)
    proxy = ((terminals <= terminal_max) & (action_gains >= action_min)).astype(np.float64)

    if not bool(spec.get("enabled", False)):
        return proxy, high_terminal, action_gains

    min_proxy = _count_from_rule_spec(
        spec,
        rule=rule,
        key="min_proxy_count",
        rate_key="min_proxy_rate",
        target_n=target_n,
        default=0,
    )
    max_high = _count_from_rule_spec(
        spec,
        rule=rule,
        key="max_high_terminal_count",
        rate_key="max_high_terminal_rate",
        target_n=target_n,
        default=target_n,
    )
    min_low = _count_from_rule_spec(
        spec,
        rule=rule,
        key="min_low_terminal_count",
        rate_key="min_low_terminal_rate",
        target_n=target_n,
        default=max(0, target_n - max_high),
    )

    source_proxy = int(proxy.sum())
    source_low = int(low_terminal.sum())
    if min_proxy > source_proxy:
        raise ValueError(
            f"{rule}: requested {min_proxy} prior-pack optimizable proxy rows "
            f"(terminal_reset_count_target <= {terminal_max:g}, "
            f"balanced_action_gain >= {action_min:g}), but only {source_proxy} are available"
        )
    if min_low > source_low:
        raise ValueError(
            f"{rule}: requested {min_low} low-terminal rows "
            f"(terminal_reset_count_target <= {terminal_max:g}), but only {source_low} are available"
        )
    a_rows.append(proxy)
    lb.append(float(min_proxy))
    ub.append(float(target_n))
    a_rows.append(high_terminal)
    lb.append(0.0)
    ub.append(float(max_high))
    a_rows.append(low_terminal)
    lb.append(float(min_low))
    ub.append(float(target_n))

    terminal_distribution_spec = spec.get("terminal_distribution_guard", {}) or {}
    terminal_bins: list[dict[str, Any]] = []
    if bool(terminal_distribution_spec.get("enabled", False)):
        edges = [float(value) for value in terminal_distribution_spec.get("edges", [])]
        if len(edges) < 2:
            raise ValueError("terminal_distribution_guard.edges must contain at least two values")
        bin_count = len(edges) - 1
        default_min = [0] * bin_count
        default_max = [target_n] * bin_count
        min_counts = [
            max(0, min(int(target_n), int(value)))
            for value in _list_from_rule_spec(
                terminal_distribution_spec,
                rule=rule,
                key="min_count",
                default=default_min,
            )
        ]
        max_counts = [
            max(0, min(int(target_n), int(value)))
            for value in _list_from_rule_spec(
                terminal_distribution_spec,
                rule=rule,
                key="max_count",
                default=default_max,
            )
        ]
        if len(min_counts) != bin_count or len(max_counts) != bin_count:
            raise ValueError(
                f"{rule}: terminal_distribution_guard count lists must match {bin_count} bins"
            )
        min_action_counts = [
            max(0, min(int(target_n), int(value)))
            for value in _list_from_rule_spec(
                terminal_distribution_spec,
                rule=rule,
                key="min_action_count",
                default=[0] * bin_count,
            )
        ]
        if len(min_action_counts) != bin_count:
            raise ValueError(
                f"{rule}: terminal_distribution_guard min_action_count must match {bin_count} bins"
            )
        bin_action_min = _float_from_rule_spec(
            terminal_distribution_spec,
            rule=rule,
            key="action_min",
            default=action_min,
        )
        mode_min_counts_raw = _dict_from_rule_spec(
            terminal_distribution_spec,
            rule=rule,
            key="mode_min_count",
            default={},
        )
        mode_min_counts: dict[str, list[int]] = {}
        for mode_key, counts in mode_min_counts_raw.items():
            if str(mode_key) not in MODE_KEYS and str(mode_key) not in {
                "strict_feedback_candidate",
                "pendulum_settle_effective_support",
            }:
                raise ValueError(f"{rule}: unsupported terminal bin mode key: {mode_key}")
            if not isinstance(counts, list) or len(counts) != bin_count:
                raise ValueError(
                    f"{rule}: terminal_distribution_guard mode_min_count[{mode_key}] "
                    f"must match {bin_count} bins"
                )
            mode_min_counts[str(mode_key)] = [
                max(0, min(int(target_n), int(value))) for value in counts
            ]
        for bin_idx, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            indicator = _terminal_bin_indicator(terminals, lo=lo, hi=hi, first=(bin_idx == 0))
            source_count = int(indicator.sum())
            if min_counts[bin_idx] > source_count:
                raise ValueError(
                    f"{rule}: requested {min_counts[bin_idx]} terminal rows in bin "
                    f"({lo:g}, {hi:g}], but only {source_count} are available"
                )
            a_rows.append(indicator)
            lb.append(float(min_counts[bin_idx]))
            ub.append(float(max_counts[bin_idx]))
            action_indicator = indicator * (action_gains >= bin_action_min).astype(np.float64)
            source_action_count = int(action_indicator.sum())
            if min_action_counts[bin_idx] > source_action_count:
                raise ValueError(
                    f"{rule}: requested {min_action_counts[bin_idx]} action-coupled terminal rows "
                    f"in bin ({lo:g}, {hi:g}] at action_min={bin_action_min:g}, "
                    f"but only {source_action_count} are available"
                )
            if min_action_counts[bin_idx] > 0:
                a_rows.append(action_indicator)
                lb.append(float(min_action_counts[bin_idx]))
                ub.append(float(target_n))
            mode_details: dict[str, Any] = {}
            for mode_key, counts in mode_min_counts.items():
                if mode_key == "strict_feedback_candidate":
                    mode_values = np.asarray(
                        [1.0 if _safe_bool(row.get("strict_feedback_candidate")) else 0.0 for row in rows]
                    )
                elif mode_key == "pendulum_settle_effective_support":
                    mode_values = np.asarray(
                        [
                            1.0
                            if (
                                _safe_bool(row.get("pendulum_short_temporal_settle_guard"))
                                or _safe_bool(row.get("pendulum_axis_support"))
                                or _safe_bool(row.get("pendulum_settle_yield_repair_planned"))
                            )
                            else 0.0
                            for row in rows
                        ]
                    )
                else:
                    mode_values = np.asarray(
                        [1.0 if _safe_bool(row.get(mode_key)) else 0.0 for row in rows]
                    )
                mode_indicator = indicator * mode_values
                source_mode_count = int(mode_indicator.sum())
                min_mode_count = int(counts[bin_idx])
                if min_mode_count > source_mode_count:
                    raise ValueError(
                        f"{rule}: requested {min_mode_count} {mode_key} rows in terminal bin "
                        f"({lo:g}, {hi:g}], but only {source_mode_count} are available"
                    )
                if min_mode_count > 0:
                    a_rows.append(mode_indicator)
                    lb.append(float(min_mode_count))
                    ub.append(float(target_n))
                mode_details[mode_key] = {
                    "min_count": int(min_mode_count),
                    "source_count": source_mode_count,
                }
            terminal_bins.append(
                {
                    "bin_index": int(bin_idx),
                    "lower": float(lo),
                    "upper": float(hi),
                    "include_lower": bool(bin_idx == 0),
                    "include_upper": True,
                    "min_count": int(min_counts[bin_idx]),
                    "max_count": int(max_counts[bin_idx]),
                    "source_count": source_count,
                    "action_min": float(bin_action_min),
                    "min_action_count": int(min_action_counts[bin_idx]),
                    "source_action_count": source_action_count,
                    "mode_min_counts": mode_details,
                }
            )

    details["prior_pack_optimizability_guard"] = {
        "enabled": True,
        "trust_status": spec.get("trust_status"),
        "terminal_reset_count_target_max": terminal_max,
        "balanced_action_gain_min": action_min,
        "min_proxy_count": int(min_proxy),
        "max_high_terminal_count": int(max_high),
        "min_low_terminal_count": int(min_low),
        "source_proxy_count": source_proxy,
        "source_low_terminal_count": source_low,
        "source_high_terminal_count": int(high_terminal.sum()),
        "source_terminal_reset_count_target_mean": float(np.mean(terminals)) if len(terminals) else 0.0,
        "source_balanced_action_gain_mean": float(np.mean(action_gains)) if len(action_gains) else 0.0,
        "terminal_distribution_guard": {
            "enabled": bool(terminal_distribution_spec.get("enabled", False)),
            "trust_status": terminal_distribution_spec.get("trust_status"),
            "bins": terminal_bins,
        },
    }
    return proxy, high_terminal, action_gains


def _repair_target_count(spec: dict[str, Any], *, rule: str, target_n: int) -> int:
    exact = spec.get("exact_count_by_rule", {})
    if rule in exact:
        return max(0, min(int(target_n), int(exact[rule])))
    if "exact_count" in spec:
        return max(0, min(int(target_n), int(spec["exact_count"])))
    min_by_rule = spec.get("min_count_by_rule", {})
    if rule in min_by_rule:
        return max(0, min(int(target_n), int(min_by_rule[rule])))
    rate = float(spec.get("rate", spec.get("min_rate", 0.0)))
    return max(0, min(int(target_n), int(math.ceil(rate * int(target_n) - 1e-9))))


def _pendulum_repair_eligible(row: dict[str, Any], spec: dict[str, Any]) -> bool:
    if bool(spec.get("require_cached_label_available", True)) and not _safe_bool(
        row.get("pendulum_label_available")
    ):
        return False
    if bool(spec.get("prefer_missing_cached_guard", True)) and _safe_bool(
        row.get("pendulum_profile_quota_ready")
    ):
        return False
    if _safe_int(row.get("state_dim"), 0) < 2:
        return False
    if _safe_int(row.get("action_dim"), 0) < 1:
        return False
    return True


def _assign_pendulum_settle_repair_branch(
    rows: list[dict[str, Any]],
    *,
    rule: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    spec = config.get("pendulum_settle_yield_repair_branch", {}) or {}
    for row in rows:
        row["pendulum_settle_yield_repair_planned"] = False
        row["pendulum_settle_yield_repair_branch"] = ""
    if not bool(spec.get("enabled", False)):
        return {"enabled": False, "selected_count": 0, "target_count": 0}

    target = _repair_target_count(spec, rule=rule, target_n=int(config["target_per_rule"]))
    eligible = [idx for idx, row in enumerate(rows) if _pendulum_repair_eligible(row, spec)]
    grouped: dict[tuple[bool, ...], list[int]] = defaultdict(list)
    for idx in eligible:
        grouped[_cell(rows[idx])].append(idx)
    for group in grouped.values():
        group.sort(key=lambda item: _safe_int(rows[item].get("_source_index"), item))

    chosen: list[int] = []
    cells = sorted(grouped, key=lambda cell: (len(grouped[cell]), "".join("1" if bit else "0" for bit in cell)))
    while len(chosen) < target and any(grouped.values()):
        progressed = False
        for cell in cells:
            if len(chosen) >= target:
                break
            if grouped[cell]:
                chosen.append(grouped[cell].pop(0))
                progressed = True
        if not progressed:
            break

    for idx in chosen:
        rows[idx]["pendulum_settle_yield_repair_planned"] = True
        rows[idx]["pendulum_settle_yield_repair_branch"] = str(
            spec.get("branch_id", "prior_internal_cached_settle_yield_repair")
        )
        rows[idx]["pendulum_settle_yield_repair_source"] = "profile_band_selector_materialize"
        rows[idx]["pendulum_settle_yield_repair_expected_guard"] = True
    return {
        "enabled": True,
        "trust_status": spec.get("trust_status"),
        "target_count": int(target),
        "eligible_count": int(len(eligible)),
        "selected_count": int(len(chosen)),
        "selected_source_indices": [int(_safe_int(rows[idx].get("_source_index"), idx)) for idx in chosen],
        "require_cached_label_available": bool(spec.get("require_cached_label_available", True)),
        "prefer_missing_cached_guard": bool(spec.get("prefer_missing_cached_guard", True)),
        "passed": bool(len(chosen) >= target),
    }


def _select(
    rows: list[dict[str, Any]],
    *,
    rule: str,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    target_n = int(config["target_per_rule"])
    if len(rows) < target_n:
        raise ValueError(f"{rule}: need {target_n}, got {len(rows)}")
    n = len(rows)

    a_rows = [np.ones(n, dtype=np.float64)]
    lb = [float(target_n)]
    ub = [float(target_n)]
    band_details: dict[str, Any] = {}
    objective = np.zeros(n, dtype=np.float64)

    for key in MODE_KEYS:
        indicator = np.asarray([1.0 if _safe_bool(row.get(key)) else 0.0 for row in rows])
        low, high, role = _constraint_for_axis(config=config, rule=rule, key=key, target_n=target_n)
        a_rows.append(indicator)
        lb.append(float(low))
        ub.append(float(high))
        weight = float(config["axes"][key].get("objective_weight", 0.0))
        if weight != 0.0:
            objective -= weight * indicator
        band_details[key] = {
            "role": role,
            "trust_status": config["axes"][key].get("trust_status"),
            "min_count": int(low),
            "max_count": int(high),
            "source_count": int(indicator.sum()),
            "source_rate": float(indicator.mean()),
        }

    # Optional locomotion gain tie-breaker only applies to the current
    # quota-ready raised axis.  It is deliberately small: constraints define
    # the profile, not this objective.
    if config["axes"].get("locomotion_witness", {}).get("role") == "raised_band":
        gains = np.asarray([_safe_float(row.get("locomotion_env_gain_vs_zero")) for row in rows])
        scale = float(np.std(gains))
        if scale > 1e-9:
            objective -= 0.05 * (gains - float(np.mean(gains))) / scale

    profile_cap = int(math.floor(float(config["joint_profile_cell_cap_rate"]) * target_n + 1e-9))
    profile_details: dict[str, Any] = {}
    for bits in sorted({_cell(row) for row in rows}):
        indicator = np.asarray([1.0 if _cell(row) == bits else 0.0 for row in rows])
        a_rows.append(indicator)
        lb.append(0.0)
        ub.append(float(profile_cap))
        profile_details["".join("1" if bit else "0" for bit in bits)] = {
            "profile": {key: bit for key, bit in zip(MODE_KEYS, bits)},
            "source_count": int(indicator.sum()),
            "source_rate": float(indicator.mean()),
            "max_count": profile_cap,
        }

    strict_indicator = _add_strict_feedback_constraints(
        rows=rows,
        rule=rule,
        target_n=target_n,
        config=config,
        a_rows=a_rows,
        lb=lb,
        ub=ub,
        details=band_details,
    )
    strict_spec = config.get("strict_feedback_subprofile_branch", {})
    if bool(strict_spec.get("enabled", False)):
        objective -= float(strict_spec.get("objective_weight", 0.0)) * strict_indicator
        objective -= float(strict_spec.get("nonconstant_objective_weight", 0.0)) * np.asarray(
            [
                1.0
                if _safe_bool(row.get("strict_feedback_candidate"))
                and str(row.get("strict_feedback_profile")) != "constant"
                else 0.0
                for row in rows
            ]
        )
        objective -= float(strict_spec.get("non_gain2_objective_weight", 0.0)) * np.asarray(
            [
                1.0
                if _safe_bool(row.get("strict_feedback_candidate"))
                and str(row.get("strict_feedback_gain")) != "2.0"
                else 0.0
                for row in rows
            ]
        )

    opt_proxy, high_terminal_indicator, action_gain_values = _add_prior_pack_optimizability_constraints(
        rows=rows,
        rule=rule,
        target_n=target_n,
        config=config,
        a_rows=a_rows,
        lb=lb,
        ub=ub,
        details=band_details,
    )
    opt_spec = config.get("prior_pack_optimizability_guard", {}) or {}
    if bool(opt_spec.get("enabled", False)):
        objective -= float(opt_spec.get("proxy_objective_weight", 0.0)) * opt_proxy
        objective += float(opt_spec.get("high_terminal_objective_weight", 0.0)) * high_terminal_indicator
        scale = float(np.std(action_gain_values))
        if scale > 1e-9:
            objective -= (
                float(opt_spec.get("action_gain_objective_weight", 0.0))
                * (action_gain_values - float(np.mean(action_gain_values)))
                / scale
            )

    result = milp(
        c=objective,
        integrality=np.ones(n, dtype=np.int8),
        bounds=Bounds(np.zeros(n), np.ones(n)),
        constraints=LinearConstraint(np.vstack(a_rows), np.asarray(lb), np.asarray(ub)),
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"{rule}: MILP failed: {result.message}")

    selected_indices = [idx for idx, value in enumerate(result.x) if value >= 0.5]
    selected = [dict(rows[idx]) for idx in selected_indices]
    selected.sort(key=lambda row: int(row["_source_index"]))
    for rank, row in enumerate(selected):
        row["candidate_rule"] = rule
        row["profile_band_rank"] = int(rank)
        row["profile_version_id"] = str(config["profile_version_id"])
        row["profile_band_guard"] = "unified_profile_band_selector"
        row["joint_profile_cell_cap_rate"] = float(config["joint_profile_cell_cap_rate"])
    pendulum_repair_details = _assign_pendulum_settle_repair_branch(
        selected,
        rule=rule,
        config=config,
    )

    selected_summary = _summary(selected, rule=rule)
    for key, detail in band_details.items():
        if key == "strict_feedback_subprofile_branch":
            summary = selected_summary
            strict_count = int(summary["strict_feedback_count"])
            nonconstant_count = int(
                sum(
                    count
                    for profile, count in summary["strict_feedback_profile_counts"].items()
                    if str(profile) != "constant"
                )
            )
            nongain2_count = int(
                sum(
                    count
                    for gain, count in summary["strict_feedback_gain_counts"].items()
                    if str(gain) != "2.0"
                )
            )
            detail["selected_strict_count"] = strict_count
            detail["selected_nonconstant_count"] = nonconstant_count
            detail["selected_non_gain2_count"] = nongain2_count
            detail["passed"] = bool(
                detail["strict_min_count"] <= strict_count <= detail["strict_max_count"]
                and nonconstant_count >= detail["nonconstant_min_count"]
                and nongain2_count >= detail["non_gain2_min_count"]
            )
            continue
        if key == "prior_pack_optimizability_guard":
            terminal_max = float(detail["terminal_reset_count_target_max"])
            action_min = float(detail["balanced_action_gain_min"])
            selected_proxy_count = int(
                sum(
                    _safe_float(row.get("terminal_reset_count_target")) <= terminal_max
                    and _safe_float(row.get("balanced_reward_action_input_gain_fraction")) >= action_min
                    for row in selected
                )
            )
            selected_high_terminal_count = int(
                sum(_safe_float(row.get("terminal_reset_count_target")) > terminal_max for row in selected)
            )
            selected_low_terminal_count = int(
                sum(_safe_float(row.get("terminal_reset_count_target")) <= terminal_max for row in selected)
            )
            detail["selected_proxy_count"] = selected_proxy_count
            detail["selected_high_terminal_count"] = selected_high_terminal_count
            detail["selected_low_terminal_count"] = selected_low_terminal_count
            detail["selected_terminal_reset_count_target_mean"] = selected_summary[
                "terminal_reset_count_target"
            ]["mean"]
            detail["selected_balanced_action_gain_mean"] = selected_summary[
                "balanced_reward_action_input_gain_fraction"
            ]["mean"]
            terminal_distribution = detail.get("terminal_distribution_guard", {})
            terminal_bins_pass = True
            for bin_detail in terminal_distribution.get("bins", []):
                lo = float(bin_detail["lower"])
                hi = float(bin_detail["upper"])
                bin_index = int(bin_detail["bin_index"])
                selected_in_bin = [
                    row
                    for row in selected
                    if bool(
                        _terminal_bin_indicator(
                            np.asarray([_safe_float(row.get("terminal_reset_count_target"))]),
                            lo=lo,
                            hi=hi,
                            first=(bin_index == 0),
                        )[0]
                    )
                ]
                selected_count = int(len(selected_in_bin))
                selected_action_count = int(
                    sum(
                        _safe_float(row.get("balanced_reward_action_input_gain_fraction"))
                        >= float(bin_detail.get("action_min", detail.get("balanced_action_gain_min", 0.18)))
                        for row in selected_in_bin
                    )
                )
                bin_detail["selected_count"] = selected_count
                bin_detail["selected_action_count"] = selected_action_count
                mode_bins_pass = True
                for mode_key, mode_detail in bin_detail.get("mode_min_counts", {}).items():
                    if mode_key == "strict_feedback_candidate":
                        selected_mode_count = int(
                            sum(_safe_bool(row.get("strict_feedback_candidate")) for row in selected_in_bin)
                        )
                    elif mode_key == "pendulum_settle_effective_support":
                        selected_mode_count = int(
                            sum(
                                _safe_bool(row.get("pendulum_short_temporal_settle_guard"))
                                or _safe_bool(row.get("pendulum_axis_support"))
                                or _safe_bool(row.get("pendulum_settle_yield_repair_planned"))
                                for row in selected_in_bin
                            )
                        )
                    else:
                        selected_mode_count = int(
                            sum(_safe_bool(row.get(mode_key)) for row in selected_in_bin)
                        )
                    mode_detail["selected_count"] = selected_mode_count
                    mode_detail["passed"] = bool(selected_mode_count >= int(mode_detail["min_count"]))
                    mode_bins_pass = mode_bins_pass and bool(mode_detail["passed"])
                bin_detail["passed"] = bool(
                    int(bin_detail["min_count"]) <= selected_count <= int(bin_detail["max_count"])
                    and selected_action_count >= int(bin_detail.get("min_action_count", 0))
                    and mode_bins_pass
                )
                terminal_bins_pass = terminal_bins_pass and bool(bin_detail["passed"])
            terminal_distribution["passed"] = bool(
                not terminal_distribution.get("enabled", False) or terminal_bins_pass
            )
            detail["passed"] = bool(
                selected_proxy_count >= detail["min_proxy_count"]
                and selected_high_terminal_count <= detail["max_high_terminal_count"]
                and selected_low_terminal_count >= detail["min_low_terminal_count"]
                and terminal_distribution.get("passed", True)
            )
            continue
        count = int(selected_summary[f"{key}_count"])
        detail["selected_count"] = count
        detail["selected_rate"] = selected_summary[f"{key}_rate"]
        detail["passed"] = bool(detail["min_count"] <= count <= detail["max_count"])

    profile_guard_pass = selected_summary["joint_profile_dominant_count"] <= profile_cap
    if pendulum_repair_details.get("enabled"):
        band_details["pendulum_settle_yield_repair_branch"] = pendulum_repair_details
    return selected, {
        "solver_success": bool(result.success),
        "solver_message": str(result.message),
        "solver_fun": float(result.fun),
        "band_details": band_details,
        "profile_cell_cap_count": profile_cap,
        "profile_details": profile_details,
        "profile_guard_pass": profile_guard_pass,
        "guard_pass": bool(
            profile_guard_pass
            and all(bool(item.get("passed", True)) for item in band_details.values())
        ),
    }


def _apply_pendulum_settle_yield_repair_to_h(h: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    if not _safe_bool(row.get("pendulum_settle_yield_repair_planned")):
        preserve_source_axis_repair = (
            _safe_bool(row.get("pendulum_axis_support"))
            or _safe_bool(row.get("pendulum_axis_supported_sustained_settle"))
        ) and _safe_bool(h.get("pendulum_settle_yield_repair_enabled"))
        if preserve_source_axis_repair:
            h["pendulum_settle_yield_repair_standalone_generator"] = False
            h["pendulum_settle_yield_repair_uses_existing_state_action_slots"] = True
            return h
        h["pendulum_settle_yield_repair_enabled"] = False
        h["pendulum_settle_yield_repair_prob"] = 0.0
        h["pendulum_settle_yield_action_controls_position"] = False
        h["pendulum_settle_yield_repair_expected_guard"] = False
        return h
    h["pendulum_settle_yield_repair_enabled"] = True
    h["pendulum_settle_yield_repair_prob"] = 1.0
    h["pendulum_settle_yield_action_controls_position"] = True
    h["pendulum_settle_yield_repair_source"] = str(
        row.get("pendulum_settle_yield_repair_source", "profile_band_selector_materialize")
    )
    h["pendulum_settle_yield_repair_branch"] = str(
        row.get("pendulum_settle_yield_repair_branch", "prior_internal_cached_settle_yield_repair")
    )
    h["pendulum_settle_yield_repair_expected_guard"] = True
    h["pendulum_settle_yield_repair_standalone_generator"] = False
    h["pendulum_settle_yield_repair_uses_existing_state_action_slots"] = True
    return h


def _materialize(out_dir: Path, rule: str, rows: list[dict[str, Any]]) -> dict[str, str]:
    rule_dir = out_dir / _safe_label(rule) / "fixed_env_group"
    frozen_dir = rule_dir / "frozen_h"
    frozen_dir.mkdir(parents=True, exist_ok=True)
    h_list = []
    seeds = []
    for idx, row in enumerate(rows):
        src = Path(str(row["full_frozen_h_json"])).expanduser().resolve()
        h = _read_json(src)
        h = _apply_pendulum_settle_yield_repair_to_h(h, row)
        dst = frozen_dir / f"env_{idx:04d}_seed{int(float(row['env_seed']))}.json"
        dst.write_text(json.dumps(_json_safe(h), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        row["full_frozen_h_json"] = str(dst)
        row["frozen_h_fingerprint"] = _frozen_h_payload_digest(h)
        h_list.append(h)
        seeds.append(int(float(row["env_seed"])))
    h_path = rule_dir / "prior_fixed_h_list.json"
    seed_path = rule_dir / "prior_fixed_env_seeds.json"
    h_path.write_text(json.dumps(_json_safe(h_list), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    seed_path.write_text(json.dumps(seeds, indent=2) + "\n", encoding="utf-8")
    return {"fixed_h_list_json": str(h_path), "fixed_env_seed_json": str(seed_path)}


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    config = _load_profile_config(args.profile_config)
    feedback_near_best_csv = getattr(args, "feedback_near_best_csv", DEFAULT_FEEDBACK_NEAR_BEST_CSV)
    future_basin_cached_labels_csv = getattr(
        args, "future_basin_cached_labels_csv", DEFAULT_FUTURE_BASIN_CACHED_LABELS_CSV
    )
    by_rule = _merge_rows(
        _read_csv(args.selected_csv),
        _load_locomotion_labels(args.locomotion_csv),
        _load_action_labels(args.action_mode_json),
        _load_strict_feedback_labels(args.strict_feedback_csv),
        _load_feedback_near_best_labels(feedback_near_best_csv),
        _load_future_basin_cached_labels(future_basin_cached_labels_csv),
    )

    report: dict[str, Any] = {
        "analysis_entry": "phase2_profile_band_selector",
        "contract": {
            "exploratory_only": True,
            "does_not_sample_new_envs": True,
            "does_not_modify_exact_scm": True,
            "does_not_modify_fit_model": True,
            "does_not_modify_ppo": True,
            "selection_wrapper_only": True,
            "raw_delta_not_used_in_selection": True,
        },
        "config": config,
        "inputs": {
            "selected_csv": str(Path(args.selected_csv).expanduser().resolve()),
            "locomotion_csv": str(Path(args.locomotion_csv).expanduser().resolve()),
            "action_mode_json": str(Path(args.action_mode_json).expanduser().resolve()),
            "strict_feedback_csv": str(Path(args.strict_feedback_csv).expanduser().resolve()) if args.strict_feedback_csv else None,
            "feedback_near_best_csv": (
                str(Path(feedback_near_best_csv).expanduser().resolve())
                if feedback_near_best_csv
                else None
            ),
            "future_basin_cached_labels_csv": (
                str(Path(future_basin_cached_labels_csv).expanduser().resolve())
                if future_basin_cached_labels_csv
                else None
            ),
            "profile_config": str(Path(args.profile_config).expanduser().resolve()) if args.profile_config else None,
        },
        "rules": {},
    }
    all_selected: list[dict[str, Any]] = []
    for rule in RULES:
        selected, info = _select(by_rule[rule], rule=rule, config=config)
        all_selected.extend(selected)
        report["rules"][rule] = {
            "source_summary": _summary(by_rule[rule], rule=rule),
            "selected_summary": _summary(selected, rule=rule),
            **info,
            "fixed_group": _materialize(out_dir, rule, selected),
        }

    selected_csv = out_dir / "selected_profile_band_prior_envs.csv"
    report_json = out_dir / "profile_band_selector_report.json"
    report_md = out_dir / "profile_band_selector.md"
    config_json = out_dir / "profile_band_selector_config.json"
    _write_csv(selected_csv, all_selected)
    config_json.write_text(json.dumps(_json_safe(config), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report["outputs"] = {
        "output_dir": str(out_dir),
        "selected_csv": str(selected_csv),
        "report_json": str(report_json),
        "summary_md": str(report_md),
        "config_json": str(config_json),
    }
    report_json.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Profile Band Selector",
        "",
        "Exploratory selection wrapper only; no exact SCM / fit_model / PPO changes.",
        "",
        f"- profile version: `{config['profile_version_id']}`",
        f"- background milestone: `{config['background_milestone']}`",
        f"- logged non-quota axes: `{list(config.get('logged_not_quota_axes', {}))}`",
        "",
        (
            "| rule | guard | locomotion | low-energy | sign/phase | state-cond | "
            "feedback | opt proxy | terminal>20 | pend-settle repair/effective | "
            "feedback profiles | feedback gains | dominant cell | entropy | active cells |"
        ),
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: |",
    ]
    for rule in RULES:
        payload = report["rules"][rule]
        s = payload["selected_summary"]
        feedback_profiles = ",".join(
            f"{key}:{value}" for key, value in sorted(s["strict_feedback_profile_counts"].items())
        ) or "none"
        feedback_gains = ",".join(
            f"{key}:{value}" for key, value in sorted(s["strict_feedback_gain_counts"].items())
        ) or "none"
        opt_detail = payload.get("band_details", {}).get("prior_pack_optimizability_guard", {})
        if opt_detail.get("enabled"):
            opt_proxy = (
                f"{opt_detail.get('selected_proxy_count', 0)}/"
                f"{opt_detail.get('min_proxy_count', 0)}"
            )
            terminal_high = (
                f"{opt_detail.get('selected_high_terminal_count', 0)}/"
                f"{opt_detail.get('max_high_terminal_count', 0)}"
            )
        else:
            opt_proxy = "off"
            terminal_high = f"{s['terminal_reset_count_target_gt20_count']}"
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{rule}`",
                    "pass" if payload["guard_pass"] else "fail",
                    f"{s['locomotion_witness_rate']:.3f}",
                    f"{s['low_energy_not_ctrl_only_rate']:.3f}",
                    f"{s['sign_or_phase_sensitive_rate']:.3f}",
                    f"{s['state_conditioned_energy_injection_rate']:.3f}",
                    f"{s['strict_feedback_rate']:.3f}",
                    f"`{opt_proxy}`",
                    f"`{terminal_high}`",
                    f"{s['pendulum_settle_yield_repair_planned_count']}/{s['pendulum_settle_effective_support_count']}",
                    f"`{feedback_profiles}`",
                    f"`{feedback_gains}`",
                    f"{s['joint_profile_dominant_rate']:.3f}",
                    f"{s['joint_profile_entropy_norm']:.3f}",
                    str(s["joint_profile_active_cells"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Read: this is a unified profile-band materialization, not a mode-specific result.",
            "Feedback strict subprofile may be branch-controlled only when enabled in the config. MountainCar future-basin remains logged until anchored.",
            "Pendulum/MountainCar future-basin cached labels are logged by default. If the pendulum repair branch is enabled, only materialized frozen h rows are patched; the source pool and default prior remain unchanged.",
            "",
            f"- JSON: `{report_json}`",
            f"- selected CSV: `{selected_csv}`",
            f"- config: `{config_json}`",
        ]
    )
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report["outputs"], indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-csv", default=str(DEFAULT_SELECTED_CSV))
    parser.add_argument("--locomotion-csv", default=str(DEFAULT_LOCOMOTION_CSV))
    parser.add_argument("--action-mode-json", default=str(DEFAULT_ACTION_MODE_JSON))
    parser.add_argument("--strict-feedback-csv", default=str(DEFAULT_STRICT_FEEDBACK_CSV))
    parser.add_argument("--feedback-near-best-csv", default=str(DEFAULT_FEEDBACK_NEAR_BEST_CSV))
    parser.add_argument("--future-basin-cached-labels-csv", default=str(DEFAULT_FUTURE_BASIN_CACHED_LABELS_CSV))
    parser.add_argument("--profile-config", default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
