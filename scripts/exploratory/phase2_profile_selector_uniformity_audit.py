#!/usr/bin/env python
"""Audit whether the profile selector implementation is itself imbalanced.

Read-only exploratory audit.  It consumes a selector preflight report and
measures selected-vs-target deviation for each profile label.  This keeps the
profile mainline separate from PPO/raw-return monitoring: a selector can be
healthy even if downstream smoke needs more work, and a smoke can look good
while profile bands collapse.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


ART = Path("/home/chen/RLPFN/artifacts")
DEFAULT_SELECTOR_JSON = (
    ART
    / "phase2_future_basin_spectrum_selector_preflight_0512_v3"
    / "future_basin_spectrum_selector_preflight.json"
)
DEFAULT_OUTPUT_DIR = ART / "phase2_profile_selector_uniformity_audit_0512"
EXEMPT_DOMINANCE_LABELS = {"terminal_activity_first_done"}


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


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


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return 0.0
    return out if math.isfinite(out) else 0.0


def audit_selector_uniformity(
    selector: dict[str, Any],
    *,
    dominance_threshold: float = 0.75,
    high_deviation_threshold: float = 0.15,
    medium_deviation_threshold: float = 0.10,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    dominance_rows: list[dict[str, Any]] = []
    for scope, payload in (selector.get("scope_reports", {}) or {}).items():
        target = payload.get("target_rates", {}) or {}
        selected = payload.get("selected_rates", {}) or {}
        pool = payload.get("pool_rates", {}) or {}
        for label in sorted(set(target) | set(selected) | set(pool)):
            pool_rate = _safe_float(pool.get(label))
            target_rate = _safe_float(target.get(label))
            selected_rate = _safe_float(selected.get(label))
            deviation = selected_rate - target_rate
            row = {
                "scope": scope,
                "label": label,
                "pool_rate": pool_rate,
                "target_rate": target_rate,
                "selected_rate": selected_rate,
                "selected_minus_target": deviation,
                "abs_selected_minus_target": abs(deviation),
                "dominance": bool(
                    label not in EXEMPT_DOMINANCE_LABELS
                    and selected_rate >= float(dominance_threshold)
                    and target_rate < float(dominance_threshold)
                ),
            }
            rows.append(row)
            if row["dominance"]:
                dominance_rows.append(row)

    max_deviation = max([row["abs_selected_minus_target"] for row in rows] or [0.0])
    if max_deviation >= float(high_deviation_threshold) or dominance_rows:
        pressure = "high" if dominance_rows else "medium_high"
    elif max_deviation >= float(medium_deviation_threshold):
        pressure = "medium"
    else:
        pressure = "low"
    missing = list(selector.get("missing_generator_labels", []) or [])
    return {
        "label_rows": rows,
        "dominance_rows": dominance_rows,
        "max_abs_selected_minus_target": max_deviation,
        "pressure_level": pressure,
        "missing_generator_labels": missing,
        "read": {
            "selector_uniformity_pressure": pressure,
            "profile_implementation_uniformity_is_current_bottleneck": bool(pressure in {"high", "medium_high"}),
            "missing_generator_labels_are_larger_pressure": bool(missing and pressure == "low"),
        },
    }


def _markdown(report: dict[str, Any]) -> str:
    audit = report["uniformity"]
    lines = [
        "# Profile Selector Uniformity Audit",
        "",
        "Read-only audit of selected-vs-target profile rates. This does not use raw return.",
        "",
        "## Read",
        "",
        f"- selector uniformity pressure: `{audit['pressure_level']}`",
        f"- max abs selected-target deviation: `{audit['max_abs_selected_minus_target']:.3f}`",
        f"- missing generator labels larger pressure: `{audit['read']['missing_generator_labels_are_larger_pressure']}`",
        "",
        "## Label Deviations",
        "",
        "| scope | label | pool | target | selected | selected-target |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in sorted(
        audit["label_rows"],
        key=lambda item: (-float(item["abs_selected_minus_target"]), str(item["scope"]), str(item["label"])),
    ):
        lines.append(
            f"| `{row['scope']}` | `{row['label']}` | {row['pool_rate']:.3f} | "
            f"{row['target_rate']:.3f} | {row['selected_rate']:.3f} | {row['selected_minus_target']:.3f} |"
        )
    if audit["missing_generator_labels"]:
        lines.extend(["", "## Missing Generator Labels", ""])
        for label in audit["missing_generator_labels"]:
            lines.append(f"- `{label}`")
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    selector_json = Path(args.selector_json).expanduser().resolve()
    selector = _read_json(selector_json)
    uniformity = audit_selector_uniformity(
        selector,
        dominance_threshold=float(args.dominance_threshold),
        high_deviation_threshold=float(args.high_deviation_threshold),
        medium_deviation_threshold=float(args.medium_deviation_threshold),
    )
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "profile_selector_uniformity_audit.json"
    md_path = out_dir / "profile_selector_uniformity_audit.md"
    report = {
        "analysis_entry": "phase2_profile_selector_uniformity_audit",
        "schema": "phase2_profile_selector_uniformity_audit.v1",
        "contract": {
            "exploratory_only": True,
            "read_only_existing_selector_report": True,
            "does_not_sample_environments": True,
            "does_not_train": True,
            "does_not_modify_exact_scm": True,
            "does_not_modify_fit_model": True,
            "does_not_modify_ppo": True,
            "profile_quality_first": True,
            "raw_delta_not_used": True,
        },
        "inputs": {"selector_json": str(selector_json)},
        "uniformity": uniformity,
        "outputs": {"json": str(json_path), "markdown": str(md_path)},
    }
    json_path.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report["outputs"], sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selector-json", default=str(DEFAULT_SELECTOR_JSON))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--dominance-threshold", type=float, default=0.75)
    parser.add_argument("--high-deviation-threshold", type=float, default=0.15)
    parser.add_argument("--medium-deviation-threshold", type=float, default=0.10)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
