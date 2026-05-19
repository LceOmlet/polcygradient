#!/usr/bin/env python
"""Attach future-basin profile labels to generator-level candidate rows.

Exploratory adapter only.  This script does not sample environments, run PPO,
change Exact SCM, or change fit_model.  It defines the low-cost cached-label
contract that the generator-level selector may consume:

* `pendulum_axis_support`: file-only support that the prior-internal
  action/state reward-basin mechanism is present.
* `pendulum_short_gap`: calibrated short-horizon reward-gap evidence.
* `pendulum_sustained_settle`: calibrated closed-loop/suffix settle evidence.
* `pendulum_profile_quota_ready`: only true when sustained settle plus
  health/diversity evidence has been promoted for selector quota use.
* `mountaincar_short_future_basin_guard`: coverage-only label for upfront
  action investment that improves future basin reward.  It is explicitly not a
  quota until a separate health/diversity and Gym anchoring review promotes it.

Labels are merged by `(rule, source_index)`, where `source_index` is the
candidate's index within its rule/source pool.  Missing labels stay explicit so
the selector cannot silently treat unavailable evidence as a negative example.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ART = Path("/home/chen/RLPFN/artifacts")
DEFAULT_SELECTED_CSV = (
    ART / "phase2_profile_online_candidate_pool_broad_proto_seed9511_0511/source_pool/selected_prior_envs.csv"
)
DEFAULT_OUTPUT_DIR = ART / "phase2_future_basin_cached_label_adapter_0512"
RULES = ("gym_full_obs_action_range", "gym_q90_obs_action_range")
BOOL_TRUE = {"1", "true", "yes", "y", "t"}


def _read_csv_optional(path: str | Path | None) -> list[dict[str, str]]:
    if path is None or str(path).strip() == "":
        return []
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        return []
    with resolved.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    if not fields:
        fields = ["rule", "source_index"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


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


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() in BOOL_TRUE


def _selected_pendulum_axis_support(row: dict[str, Any]) -> bool:
    """Return file-only support that the prior-internal basin axis exists.

    This intentionally comes from the generator/source-pool row, not from the
    temporal settle guard.  The guard is evidence that the mechanism works as a
    sustained closed-loop mode; the selected row is evidence that the mechanism
    was actually present in the generated prior environment.
    """

    return any(
        _as_bool(row.get(key))
        for key in (
            "pendulum_axis_support",
            "pendulum_settle_yield_axis_enabled",
            "pendulum_settle_yield_repair_enabled",
            "pendulum_settle_yield_repair_planned",
        )
    )


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


def _candidate_source_indices(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Normalize selected rows to explicit per-rule source indices."""

    counters: dict[str, int] = defaultdict(int)
    out: list[dict[str, Any]] = []
    for fallback_idx, row in enumerate(rows):
        rule = str(row.get("rule") or row.get("scope") or "")
        if not rule:
            rule = "unknown"
        if str(row.get("_source_index", "")).strip() != "":
            source_index = _safe_int(row.get("_source_index"))
        elif str(row.get("source_index", "")).strip() != "":
            source_index = _safe_int(row.get("source_index"))
        elif str(row.get("env_index", "")).strip() != "":
            source_index = _safe_int(row.get("env_index"))
        elif str(row.get("env_idx", "")).strip() != "":
            source_index = _safe_int(row.get("env_idx"))
        else:
            source_index = counters[rule]
        counters[rule] = max(counters[rule], source_index + 1)
        out.append({"fallback_idx": fallback_idx, "rule": rule, "source_index": source_index, **row})
    return out


def _autodiscover_mountaincar(selected_csv: Path) -> Path | None:
    candidate = selected_csv.parent.parent / "mountaincar_future_basin_detector" / "mountaincar_future_basin_per_env.csv"
    return candidate if candidate.exists() else None


def _autodiscover_pendulum(selected_csv: Path) -> Path | None:
    candidate = (
        selected_csv.parent.parent
        / "pendulum_short_temporal_settle_guard"
        / "pendulum_short_temporal_settle_per_env.csv"
    )
    return candidate if candidate.exists() else None


def _index_mountaincar(rows: list[dict[str, str]]) -> dict[tuple[str, int], dict[str, Any]]:
    indexed = {}
    for row in rows:
        rule = str(row.get("source_label") or row.get("rule") or row.get("scope") or "")
        source_index = _safe_int(row.get("env_index", row.get("source_index", row.get("env_idx", 0))))
        indexed[(rule, source_index)] = {
            "mountaincar_short_future_basin_guard": _as_bool(row.get("mountaincar_future_basin_candidate")),
            "mountaincar_delayed_future_basin_guard": _as_bool(row.get("mountaincar_delayed_candidate")),
            "mountaincar_strong_future_basin_guard": _as_bool(row.get("mountaincar_future_basin_strong")),
            "mountaincar_upfront_action_decay": _as_bool(row.get("mountaincar_upfront_action_decay")),
            "mountaincar_guard_margin": _safe_float(row.get("mountaincar_margin")),
            "mountaincar_best_early_only_mode": row.get("best_early_only_mode", ""),
            "mountaincar_label_available": True,
        }
    return indexed


def _index_pendulum(rows: list[dict[str, str]]) -> dict[tuple[str, int], dict[str, Any]]:
    indexed = {}
    for row in rows:
        rule = str(row.get("source_label") or row.get("rule") or row.get("scope") or "")
        source_index = _safe_int(row.get("source_index", row.get("env_index", row.get("env_idx", 0))))
        legacy_guard = _as_bool(
            row.get(
                "pendulum_short_temporal_settle_guard",
                row.get("short_temporal_settle_guard", row.get("guard_pass", False)),
            )
        )
        axis_support = _as_bool(row.get("pendulum_axis_support")) or (
            legacy_guard
            and "pendulum_axis_support" not in row
            and "pendulum_sustained_settle" not in row
        )
        short_gap = _as_bool(row.get("pendulum_short_gap"))
        medium = _as_bool(row.get("pendulum_medium_settle"))
        strict = _as_bool(row.get("pendulum_strict_settle")) or (
            legacy_guard and "pendulum_sustained_settle" not in row
        )
        sustained = _as_bool(row.get("pendulum_sustained_settle")) or medium or (
            legacy_guard and "pendulum_sustained_settle" not in row
        )
        axis_supported_sustained = _as_bool(row.get("pendulum_axis_supported_sustained_settle"))
        quota_ready = _as_bool(row.get("pendulum_profile_quota_ready"))
        indexed[(rule, source_index)] = {
            "pendulum_axis_support": axis_support,
            "pendulum_short_gap": short_gap,
            "pendulum_medium_settle": medium,
            "pendulum_strict_settle": strict,
            "pendulum_sustained_settle": sustained,
            "pendulum_axis_supported_sustained_settle": axis_supported_sustained,
            "pendulum_profile_quota_ready": quota_ready,
            # Legacy compatibility; in the split schema this remains the
            # strict audit guard, not mere axis support or medium settle.
            "pendulum_short_temporal_settle_guard": strict,
            "pendulum_settle_guard_precision_class": row.get("precision_class", row.get("label_quality", "")),
            "pendulum_axis_cheap_label_source": row.get("pendulum_axis_cheap_label_source", ""),
            "pendulum_label_available": True,
        }
    return indexed


def _rate(rows: list[dict[str, Any]], key: str, *, available_key: str | None = None) -> float | None:
    if available_key is not None:
        rows = [row for row in rows if bool(row.get(available_key))]
    if not rows:
        return None
    return float(sum(bool(row.get(key)) for row in rows) / len(rows))


def _summaries(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_rule: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_rule[str(row.get("rule"))].append(row)
    out: dict[str, Any] = {}
    for rule, rule_rows in sorted(by_rule.items()):
        out[rule] = {
            "n": len(rule_rows),
            "pendulum_label_available_count": int(sum(bool(row.get("pendulum_label_available")) for row in rule_rows)),
            "pendulum_axis_support_rate": _rate(
                rule_rows,
                "pendulum_axis_support",
                available_key="pendulum_label_available",
            ),
            "pendulum_short_gap_rate": _rate(
                rule_rows,
                "pendulum_short_gap",
                available_key="pendulum_label_available",
            ),
            "pendulum_medium_settle_rate": _rate(
                rule_rows,
                "pendulum_medium_settle",
                available_key="pendulum_label_available",
            ),
            "pendulum_strict_settle_rate": _rate(
                rule_rows,
                "pendulum_strict_settle",
                available_key="pendulum_label_available",
            ),
            "pendulum_sustained_settle_rate": _rate(
                rule_rows,
                "pendulum_sustained_settle",
                available_key="pendulum_label_available",
            ),
            "pendulum_axis_supported_sustained_settle_rate": _rate(
                rule_rows,
                "pendulum_axis_supported_sustained_settle",
                available_key="pendulum_label_available",
            ),
            "pendulum_profile_quota_ready_rate": _rate(
                rule_rows,
                "pendulum_profile_quota_ready",
                available_key="pendulum_label_available",
            ),
            "pendulum_short_temporal_settle_guard_rate": _rate(
                rule_rows,
                "pendulum_short_temporal_settle_guard",
                available_key="pendulum_label_available",
            ),
            "mountaincar_label_available_count": int(
                sum(bool(row.get("mountaincar_label_available")) for row in rule_rows)
            ),
            "mountaincar_short_future_basin_guard_rate": _rate(
                rule_rows,
                "mountaincar_short_future_basin_guard",
                available_key="mountaincar_label_available",
            ),
            "mountaincar_delayed_future_basin_guard_rate": _rate(
                rule_rows,
                "mountaincar_delayed_future_basin_guard",
                available_key="mountaincar_label_available",
            ),
            "mountaincar_upfront_action_decay_rate": _rate(
                rule_rows,
                "mountaincar_upfront_action_decay",
                available_key="mountaincar_label_available",
            ),
            "mountaincar_best_early_only_mode_counts": dict(
                Counter(
                    str(row.get("mountaincar_best_early_only_mode") or "unavailable")
                    for row in rule_rows
                    if bool(row.get("mountaincar_label_available"))
                )
            ),
        }
    return out


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Future-Basin Cached Label Adapter",
        "",
        "Exploratory generator-level label contract. It attaches cached labels; it does not sample, train, or quota.",
        "",
        "## Contract",
        "",
        "- `pendulum_axis_support`: prior-internal action/state reward-basin mechanism exists.",
        "- `pendulum_short_gap`: calibrated short-horizon positive reward-gap label.",
        "- `pendulum_sustained_settle`: calibrated closed-loop/suffix settle label.",
        "- `pendulum_axis_supported_sustained_settle`: axis support and sustained settle together.",
        "- `pendulum_profile_quota_ready`: sustained settle plus health/diversity evidence; only this may become quota.",
        "- `mountaincar_short_future_basin_guard`: coverage-only upfront-investment/future-basin label; not a quota.",
        "- Missing label evidence remains explicit via `*_label_available`.",
        "",
        "## Coverage",
        "",
        "| rule | n | pend labels | axis support | short gap | medium | strict | sustained | axis+sustained | quota | mountaincar labels | mountaincar guard | mountaincar delayed |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rule, item in report["summaries"].items():
        lines.append(
            f"| `{rule}` | {item['n']} | {item['pendulum_label_available_count']} | "
            f"{item['pendulum_axis_support_rate']} | "
            f"{item['pendulum_short_gap_rate']} | "
            f"{item['pendulum_medium_settle_rate']} | "
            f"{item['pendulum_strict_settle_rate']} | "
            f"{item['pendulum_sustained_settle_rate']} | "
            f"{item['pendulum_axis_supported_sustained_settle_rate']} | "
            f"{item['pendulum_profile_quota_ready_rate']} | "
            f"{item['mountaincar_label_available_count']} | "
            f"{item['mountaincar_short_future_basin_guard_rate']} | "
            f"{item['mountaincar_delayed_future_basin_guard_rate']} |"
        )
    lines.extend(["", "## Read", "", report["read"]["summary"]])
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    selected_csv = Path(args.selected_csv).expanduser().resolve()
    selected_rows = _candidate_source_indices(_read_csv_optional(selected_csv))

    mountaincar_path = Path(args.mountaincar_per_env_csv).expanduser().resolve() if args.mountaincar_per_env_csv else None
    if mountaincar_path is None or not mountaincar_path.exists():
        mountaincar_path = _autodiscover_mountaincar(selected_csv)
    pendulum_path = Path(args.pendulum_settle_label_csv).expanduser().resolve() if args.pendulum_settle_label_csv else None
    if pendulum_path is None or not pendulum_path.exists():
        pendulum_path = _autodiscover_pendulum(selected_csv)

    mountaincar = _index_mountaincar(_read_csv_optional(mountaincar_path))
    pendulum = _index_pendulum(_read_csv_optional(pendulum_path))

    output_rows: list[dict[str, Any]] = []
    for row in selected_rows:
        key = (str(row["rule"]), int(row["source_index"]))
        m = mountaincar.get(key, {})
        p = pendulum.get(key, {})
        selected_axis_support = _selected_pendulum_axis_support(row)
        pendulum_axis_support = selected_axis_support or bool(p.get("pendulum_axis_support", False))
        pendulum_axis_supported_sustained = bool(p.get("pendulum_axis_supported_sustained_settle", False)) or (
            pendulum_axis_support and bool(p.get("pendulum_sustained_settle", False))
        )
        output_rows.append(
            {
                "rule": row["rule"],
                "source_index": int(row["source_index"]),
                "env_seed": row.get("env_seed", row.get("seed", "")),
                "candidate_idx": row.get("candidate_idx", ""),
                "pendulum_axis_support": pendulum_axis_support,
                "pendulum_short_gap": bool(p.get("pendulum_short_gap", False)),
                "pendulum_medium_settle": bool(p.get("pendulum_medium_settle", False)),
                "pendulum_strict_settle": bool(p.get("pendulum_strict_settle", False)),
                "pendulum_sustained_settle": bool(p.get("pendulum_sustained_settle", False)),
                "pendulum_axis_supported_sustained_settle": pendulum_axis_supported_sustained,
                "pendulum_profile_quota_ready": bool(p.get("pendulum_profile_quota_ready", False)),
                "pendulum_short_temporal_settle_guard": bool(p.get("pendulum_short_temporal_settle_guard", False)),
                "pendulum_label_available": bool(p.get("pendulum_label_available", False)),
                "pendulum_settle_guard_precision_class": p.get("pendulum_settle_guard_precision_class", ""),
                "pendulum_axis_cheap_label_source": p.get("pendulum_axis_cheap_label_source", ""),
                "pendulum_axis_support_source": (
                    "selected_pool_axis" if selected_axis_support else ("cached_label" if pendulum_axis_support else "")
                ),
                "mountaincar_short_future_basin_guard": bool(m.get("mountaincar_short_future_basin_guard", False)),
                "mountaincar_delayed_future_basin_guard": bool(
                    m.get("mountaincar_delayed_future_basin_guard", False)
                ),
                "mountaincar_strong_future_basin_guard": bool(m.get("mountaincar_strong_future_basin_guard", False)),
                "mountaincar_upfront_action_decay": bool(m.get("mountaincar_upfront_action_decay", False)),
                "mountaincar_guard_margin": m.get("mountaincar_guard_margin", ""),
                "mountaincar_best_early_only_mode": m.get("mountaincar_best_early_only_mode", ""),
                "mountaincar_label_available": bool(m.get("mountaincar_label_available", False)),
                "mountaincar_label_use": "coverage_only_not_quota" if bool(m.get("mountaincar_label_available")) else "",
                "label_key": f"{row['rule']}:{int(row['source_index'])}",
            }
        )

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "future_basin_cached_labels.csv"
    json_path = out_dir / "future_basin_cached_label_adapter.json"
    md_path = out_dir / "future_basin_cached_label_adapter.md"
    _write_csv(csv_path, output_rows)
    summaries = _summaries(output_rows)
    report = {
        "analysis_entry": "phase2_future_basin_cached_label_adapter",
        "schema": "phase2_future_basin_cached_label_adapter.v1",
        "contract": {
            "exploratory_only": True,
            "does_not_sample_environments": True,
            "does_not_train": True,
            "does_not_modify_exact_scm": True,
            "does_not_modify_fit_model": True,
            "does_not_modify_ppo": True,
            "profile_quality_first": True,
            "raw_delta_not_used": True,
            "mountaincar_guard_is_coverage_only": True,
            "missing_labels_are_explicit": True,
        },
        "inputs": {
            "selected_csv": str(selected_csv),
            "pendulum_settle_label_csv": str(pendulum_path) if pendulum_path else None,
            "mountaincar_per_env_csv": str(mountaincar_path) if mountaincar_path else None,
        },
        "summaries": summaries,
        "read": {
            "generator_cached_label_schema_ready": True,
            "pendulum_generator_label_populated": any(
                item["pendulum_label_available_count"] > 0 for item in summaries.values()
            ),
            "pendulum_axis_support_populated": any(
                (item.get("pendulum_axis_support_rate") or 0.0) > 0.0 for item in summaries.values()
            ),
            "pendulum_profile_quota_ready": any(
                (item.get("pendulum_profile_quota_ready_rate") or 0.0) > 0.0 for item in summaries.values()
            ),
            "mountaincar_guard_coverage_ready": any(
                item["mountaincar_label_available_count"] > 0 for item in summaries.values()
            ),
            "mountaincar_quota_ready": False,
            "summary": (
                "The cached-label schema is ready. Pendulum labels are split into axis_support, short_gap, "
                "sustained_settle, and profile_quota_ready; only profile_quota_ready may drive quota. "
                "MountainCar can be attached as coverage-only evidence when the per-env detector came from the same pool."
            ),
        },
        "outputs": {"csv": str(csv_path), "json": str(json_path), "markdown": str(md_path)},
    }
    json_path.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report["outputs"], sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-csv", default=str(DEFAULT_SELECTED_CSV))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--pendulum-settle-label-csv", default="")
    parser.add_argument("--mountaincar-per-env-csv", default="")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
