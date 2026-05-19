#!/usr/bin/env python
"""Probe whether Pendulum settle yield can be repaired from existing prior h.

This is a read-only prior-derived repair gate.  It consumes exported milestone2
prior h lists plus the current Pendulum settle state-label preflight and asks:

1. Can a selector over existing h fields close the settle-yield gap?
2. Did existing h/parameter knob sweeps already fail?
3. If not h-only, what kind of repair remains allowed?

The script explicitly forbids using the standalone quadratic specimen as a
generator.  That specimen can define the missing shape, but a repair must be
implemented inside the existing prior/Exact-SCM generation family or derived
from exported h.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


ART = Path("/home/chen/RLPFN/artifacts")
DEFAULT_OUTPUT_DIR = ART / "phase2_pendulum_settle_prior_derived_repair_probe_0512"
DEFAULT_PREFLIGHT_JSON = (
    ART
    / "phase2_pendulum_settle_state_label_seed9511_gpu0_0512"
    / "pendulum_settle_state_label_preflight.json"
)
DEFAULT_READINESS_JSON = (
    ART
    / "phase2_pendulum_settle_yield_repair_readiness_0512"
    / "pendulum_settle_yield_repair_readiness.json"
)
DEFAULT_FULL_H_LIST_JSON = (
    ART
    / "phase2_profile_online_candidate_pool_broad_proto_seed9511_0511"
    / "fixed_groups"
    / "gym_full_obs_action_range"
    / "fixed_env_group"
    / "prior_fixed_h_list.json"
)
DEFAULT_Q90_H_LIST_JSON = (
    ART
    / "phase2_profile_online_candidate_pool_broad_proto_seed9511_0511"
    / "fixed_groups"
    / "gym_q90_obs_action_range"
    / "fixed_env_group"
    / "prior_fixed_h_list.json"
)

RULE_TO_DEFAULT_H_LIST = {
    "gym_full_obs_action_range": DEFAULT_FULL_H_LIST_JSON,
    "gym_q90_obs_action_range": DEFAULT_Q90_H_LIST_JSON,
}
TARGET_KEYS = (
    "cheap_settle_guard",
    "reward_condition",
    "suffix_reward_condition",
    "action_decay_condition",
    "self_state_settle_condition",
    "open_state_settle_condition",
)
EXCLUDED_H_KEYS = {
    "fixed_frozen_h_json",
    "family",
    "prior_mlp_activations",
    "reference_gp_forward_mode",
    "batch_parallel_backend",
    "batch_vectorized_grouping",
    "transition_inner_grouping",
}


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def _safe_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _load_h_rows(path: str | Path) -> list[dict[str, Any]]:
    payload = _read_json(path)
    if isinstance(payload, list):
        return [dict(row) for row in payload]
    if isinstance(payload, dict):
        for key in ("h_list", "prior_fixed_h_list", "rows"):
            value = payload.get(key)
            if isinstance(value, list):
                return [dict(row) for row in value]
    raise ValueError(f"Cannot find h list in {path}")


def _preflight_rows(preflight: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for item in preflight.get("lists", []) or []:
        label = str(item.get("label", ""))
        per_env = ((item.get("classification") or {}).get("per_env") or [])
        out[label] = [dict(row) for row in per_env]
    return out


def _merge_rows(
    *,
    preflight: dict[str, Any],
    rule_to_h_list: dict[str, str | Path],
) -> list[dict[str, Any]]:
    by_rule = _preflight_rows(preflight)
    merged: list[dict[str, Any]] = []
    for rule, h_path in rule_to_h_list.items():
        labels = by_rule.get(rule)
        if labels is None:
            continue
        h_rows = _load_h_rows(h_path)
        if len(labels) > len(h_rows):
            raise ValueError(f"{rule} has {len(labels)} labels but only {len(h_rows)} h rows")
        for idx, label in enumerate(labels):
            row = {
                "rule": rule,
                "env_index": int(label.get("env_index", idx)),
                "label": label,
                "h": h_rows[idx],
            }
            merged.append(row)
    return merged


def _target_counts(rows: list[dict[str, Any]], *, required_min_count: int) -> dict[str, Any]:
    by_rule: dict[str, dict[str, Any]] = {}
    for rule in sorted({str(row["rule"]) for row in rows}):
        subset = [row for row in rows if str(row["rule"]) == rule]
        counts = {
            key: int(sum(bool(row["label"].get(key)) for row in subset))
            for key in TARGET_KEYS
        }
        by_rule[rule] = {
            "n": int(len(subset)),
            "counts": counts,
            "guard_count_safe": bool(counts["cheap_settle_guard"] >= int(required_min_count)),
        }
    min_guard = min(
        (item["counts"]["cheap_settle_guard"] for item in by_rule.values()),
        default=0,
    )
    return {
        "required_min_count": int(required_min_count),
        "min_guard_count": int(min_guard),
        "can_close_by_selecting_existing_positive_h": bool(min_guard >= int(required_min_count)),
        "by_rule": by_rule,
    }


def _numeric_h_keys(rows: list[dict[str, Any]]) -> list[str]:
    keys = sorted({key for row in rows for key in row["h"].keys()})
    out = []
    for key in keys:
        if key in EXCLUDED_H_KEYS:
            continue
        values = [_safe_float(row["h"].get(key)) for row in rows]
        finite = [v for v in values if v is not None]
        if len(finite) < max(4, len(rows) // 3):
            continue
        if len({round(float(v), 12) for v in finite}) <= 1:
            continue
        out.append(key)
    return out


def _evaluate_selector(
    rows: list[dict[str, Any]],
    *,
    key: str,
    op: str,
    threshold: float,
    target: str,
) -> dict[str, Any]:
    selected = []
    for row in rows:
        value = _safe_float(row["h"].get(key))
        if value is None:
            continue
        keep = value <= threshold if op == "<=" else value >= threshold
        if keep:
            selected.append(row)
    positives = [row for row in selected if bool(row["label"].get(target))]
    truth = [row for row in rows if bool(row["label"].get(target))]
    return {
        "key": key,
        "op": op,
        "threshold": float(threshold),
        "selected_count": int(len(selected)),
        "positive_count": int(len(positives)),
        "truth_count": int(len(truth)),
        "precision": float(len(positives) / max(1, len(selected))),
        "recall": float(len(positives) / max(1, len(truth))),
        "selected_rate": float(len(selected) / max(1, len(rows))),
    }


def _best_univariate_selectors(rows: list[dict[str, Any]], *, target: str, top_k: int = 8) -> list[dict[str, Any]]:
    candidates = []
    for key in _numeric_h_keys(rows):
        values = sorted(
            {
                float(v)
                for row in rows
                for v in [_safe_float(row["h"].get(key))]
                if v is not None
            }
        )
        if len(values) < 2:
            continue
        cut_indices = sorted(
            {
                max(0, min(len(values) - 1, int(round(q * (len(values) - 1)))))
                for q in (0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9)
            }
        )
        for idx in cut_indices:
            threshold = values[idx]
            for op in ("<=", ">="):
                item = _evaluate_selector(rows, key=key, op=op, threshold=threshold, target=target)
                if item["selected_count"] < 3:
                    continue
                candidates.append(item)
    candidates.sort(
        key=lambda item: (
            item["positive_count"],
            item["precision"],
            item["recall"],
            -item["selected_count"],
        ),
        reverse=True,
    )
    return candidates[: int(top_k)]


def _summarize_h_distributions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    important = [
        "alpha",
        "state_output_scale",
        "state_highway_lambda",
        "reference_state_inertia_enabled",
        "state_full_rms_enabled",
        "state_full_rms_target",
        "init_state_std",
        "init_action_std",
        "state_noise_std",
        "noise_std",
        "outputscale",
        "lengthscale",
        "ctrl_reward_weight",
        "action_dim",
        "state_dim",
        "obs_dim",
    ]
    out = {}
    for key in important:
        values = [v for row in rows for v in [_safe_float(row["h"].get(key))] if v is not None]
        if not values:
            continue
        out[key] = {
            "count": int(len(values)),
            "min": float(min(values)),
            "max": float(max(values)),
            "mean": float(sum(values) / len(values)),
            "unique_count": int(len({round(float(v), 12) for v in values})),
        }
    return out


def build_report(
    *,
    preflight_json: str | Path,
    readiness_json: str | Path,
    full_h_list_json: str | Path,
    q90_h_list_json: str | Path,
    required_min_count: int | None = None,
) -> dict[str, Any]:
    preflight = _read_json(preflight_json)
    readiness = _read_json(readiness_json)
    required = int(required_min_count or (preflight.get("decision", {}) or {}).get("required_min_count", 8))
    rows = _merge_rows(
        preflight=preflight,
        rule_to_h_list={
            "gym_full_obs_action_range": full_h_list_json,
            "gym_q90_obs_action_range": q90_h_list_json,
        },
    )
    counts = _target_counts(rows, required_min_count=required)
    h_selectors = {
        target: _best_univariate_selectors(rows, target=target)
        for target in TARGET_KEYS
    }
    best_guard_selector = h_selectors["cheap_settle_guard"][0] if h_selectors["cheap_settle_guard"] else None
    knobs_falsified = bool((readiness.get("existing_knob_trials") or {}).get("existing_knobs_falsified"))
    standalone_allowed = bool(
        (readiness.get("quadratic_settle_mechanism") or {}).get("standalone_specimen_integration_allowed")
    )
    short_guard_ready = bool(
        (readiness.get("cheap_or_cached_guard") or {}).get("short_temporal_cached_label_viable")
    )
    h_only_can_close = bool(counts["can_close_by_selecting_existing_positive_h"])

    if h_only_can_close:
        repair_class = "h_only_selector_possible"
    elif knobs_falsified:
        repair_class = "prior_internal_exact_scm_mechanism_required"
    else:
        repair_class = "finish_existing_knob_falsification"

    return {
        "analysis_entry": "phase2_pendulum_settle_prior_derived_repair_probe",
        "schema": "phase2_pendulum_settle_prior_derived_repair_probe.v1",
        "contract": {
            "read_only_existing_artifacts": True,
            "uses_exported_prior_h": True,
            "does_not_sample_environments": True,
            "does_not_train": True,
            "does_not_modify_exact_scm": True,
            "does_not_modify_fit_model": True,
            "does_not_modify_ppo": True,
            "standalone_quadratic_generator_forbidden": True,
            "not_a_raw_delta_gate": True,
            "profile_quality_first": True,
        },
        "inputs": {
            "preflight_json": str(Path(preflight_json).expanduser().resolve()),
            "readiness_json": str(Path(readiness_json).expanduser().resolve()),
            "full_h_list_json": str(Path(full_h_list_json).expanduser().resolve()),
            "q90_h_list_json": str(Path(q90_h_list_json).expanduser().resolve()),
        },
        "target_counts": counts,
        "h_distributions": _summarize_h_distributions(rows),
        "h_only_selectors": {
            "best_guard_selector": best_guard_selector,
            "top_by_target": h_selectors,
        },
        "upstream_readiness": {
            "existing_knobs_falsified": knobs_falsified,
            "standalone_specimen_integration_allowed": standalone_allowed,
            "short_temporal_cached_label_ready": short_guard_ready,
        },
        "decision": {
            "can_use_standalone_quadratic_generator": False,
            "can_close_with_h_only_selection": h_only_can_close,
            "safe_next_repair_class": repair_class,
            "why": (
                "Existing exported h already contains enough positives for selection."
                if h_only_can_close
                else "Existing exported h does not contain enough settle-positive environments, so selection "
                "over h cannot create the missing yield. Since cheap existing knobs were falsified, the "
                "remaining allowed path is a prior-internal Exact-SCM mechanism repair, calibrated by "
                "the offline quadratic specimen but not replaced by it."
            ),
            "allowed_next_actions": [
                "search/implement an Exact-SCM-internal settle-yield mechanism using existing state/action slots",
                "derive labels from generated prior h via the short temporal cached guard",
                "keep MountainCar future-basin as coverage-only until Pendulum settle is prior-derived",
                "run full/q90 profile-band health/diversity smoke before any milestone promotion",
            ],
        },
    }


def _write_md(report: dict[str, Any], path: Path) -> None:
    decision = report["decision"]
    counts = report["target_counts"]
    lines = [
        "# Pendulum Settle Prior-Derived Repair Probe",
        "",
        f"- can use standalone quadratic generator: `{decision['can_use_standalone_quadratic_generator']}`",
        f"- can close with h-only selection: `{decision['can_close_with_h_only_selection']}`",
        f"- safe next repair class: `{decision['safe_next_repair_class']}`",
        f"- why: {decision['why']}",
        "",
        "## Counts",
        f"- required min guard count per rule: `{counts['required_min_count']}`",
        f"- observed min guard count: `{counts['min_guard_count']}`",
    ]
    for rule, item in counts["by_rule"].items():
        lines.append(f"- {rule}: `{item['counts']}`")
    lines.extend(["", "## Best h-only guard selector"])
    lines.append(f"`{json.dumps(report['h_only_selectors']['best_guard_selector'], sort_keys=True)}`")
    lines.extend(["", "## Allowed next actions"])
    lines.extend(f"- {item}" for item in decision["allowed_next_actions"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-json", default=str(DEFAULT_PREFLIGHT_JSON))
    parser.add_argument("--readiness-json", default=str(DEFAULT_READINESS_JSON))
    parser.add_argument("--full-h-list-json", default=str(DEFAULT_FULL_H_LIST_JSON))
    parser.add_argument("--q90-h-list-json", default=str(DEFAULT_Q90_H_LIST_JSON))
    parser.add_argument("--required-min-count", type=int, default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(
        preflight_json=args.preflight_json,
        readiness_json=args.readiness_json,
        full_h_list_json=args.full_h_list_json,
        q90_h_list_json=args.q90_h_list_json,
        required_min_count=args.required_min_count,
    )
    json_path = out_dir / "pendulum_settle_prior_derived_repair_probe.json"
    md_path = out_dir / "pendulum_settle_prior_derived_repair_probe.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_md(report, md_path)
    print(str(json_path))
    print(str(md_path))


if __name__ == "__main__":
    main()
