#!/usr/bin/env python
"""Preflight a balanced selector for the future-basin spectrum.

This is read-only and exploratory.  It validates the selector mechanics on the
currently available broad per-env labels, while explicitly reporting the labels
that are still missing for a fully unified Pendulum/MountainCar spectrum.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any


ART = Path("/home/chen/RLPFN/artifacts")
DEFAULT_OUTPUT_DIR = ART / "phase2_future_basin_spectrum_selector_preflight_0512"
DEFAULT_UNIFIED_PROFILE_JSON = (
    ART / "phase2_unified_gym_like_mode_profile_gpu0_0510/unified_gym_like_mode_profile_report.json"
)
DEFAULT_SPECTRUM_JSON = (
    ART
    / "phase2_action_conditioned_future_basin_spectrum_0512"
    / "action_conditioned_future_basin_spectrum.json"
)

AVAILABLE_LABELS = (
    "locomotion_witness",
    "low_energy_not_ctrl_only",
    "sign_or_phase_sensitive",
    "state_conditioned_energy_injection",
    "pendulum_like_primary_long_horizon",
    "pendulum_like_selector_hard_positive",
    "terminal_activity_first_done",
)

MISSING_GENERATOR_LABELS = (
    "pendulum_short_temporal_settle_guard",
    "mountaincar_delayed_future_basin_guard",
    "terminal_action_sensitive_trap_guard",
)


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _candidate_rows(unified: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in unified.get("per_env_rows", []) or []:
        labels = [label for label in AVAILABLE_LABELS if _as_bool(row.get(label))]
        rows.append(
            {
                "scope": str(row.get("scope")),
                "env_idx": int(row.get("env_idx")),
                "labels": labels,
                "raw_reward_sum_delta": row.get("raw_reward_sum_delta"),
                **{label: _as_bool(row.get(label)) for label in AVAILABLE_LABELS},
            }
        )
    return rows


def label_rates(rows: list[dict[str, Any]], labels: tuple[str, ...] = AVAILABLE_LABELS) -> dict[str, float]:
    n = max(1, len(rows))
    return {label: sum(bool(row.get(label)) for row in rows) / n for label in labels}


def _target_rates(rows: list[dict[str, Any]], labels: tuple[str, ...]) -> dict[str, float]:
    pool_rates = label_rates(rows, labels)
    targets: dict[str, float] = {}
    for label, rate in pool_rates.items():
        if rate <= 0.0 or rate >= 1.0:
            targets[label] = rate
        else:
            # Raise very sparse bins enough to make them visible, while capping
            # already-common parent/protection bins so co-occurrence cannot make
            # them dominate the selected list.
            targets[label] = min(0.60, max(0.20, rate))
    return targets


def balanced_select(
    rows: list[dict[str, Any]],
    *,
    n_select: int,
    labels: tuple[str, ...] = AVAILABLE_LABELS,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Greedy round-robin selector that favors underrepresented labels.

    It is intentionally simple: score candidates by the number of currently
    under-covered labels they carry, with a small deterministic shuffle for tie
    breaking.  This is a selector mechanics test, not a final quota policy.
    """

    rng = random.Random(int(seed))
    remaining = list(rows)
    rng.shuffle(remaining)
    selected: list[dict[str, Any]] = []
    target_rates = _target_rates(rows, labels)
    target_counts = {
        label: (math.ceil(target_rates[label] * int(n_select)) if target_rates[label] > 0.0 else 0.0)
        for label in labels
    }

    def objective(candidates: list[dict[str, Any]]) -> float:
        counts = Counter()
        for row in candidates:
            for label in labels:
                if row.get(label):
                    counts[label] += 1
        total = 0.0
        for label in labels:
            target = target_counts[label]
            if target <= 0.0:
                total += 4.0 * (counts[label] ** 2)
                continue
            diff = (counts[label] - target) / max(1.0, target)
            # Over-shooting a broad/protection band is more harmful than being
            # slightly short, because it makes the profile collapse.
            if diff > 0:
                diff *= 1.5
            total += diff * diff
        return total

    while remaining and len(selected) < int(n_select):
        best_idx = min(range(len(remaining)), key=lambda idx: objective([*selected, remaining[idx]]))
        selected.append(remaining.pop(best_idx))
    return selected


def _scope_report(rows: list[dict[str, Any]], *, n_select: int, seed: int) -> dict[str, Any]:
    selected = balanced_select(rows, n_select=n_select, seed=seed)
    return {
        "n_pool": len(rows),
        "n_selected": len(selected),
        "target_rates": _target_rates(rows, AVAILABLE_LABELS),
        "pool_rates": label_rates(rows),
        "selected_rates": label_rates(selected),
        "selected_envs": [
            {"scope": row["scope"], "env_idx": row["env_idx"], "labels": row["labels"]} for row in selected
        ],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    unified = _read_json(args.unified_profile_json)
    spectrum = _read_json(args.spectrum_json)
    rows = _candidate_rows(unified)
    by_scope = {
        scope: [row for row in rows if row["scope"] == scope]
        for scope in sorted({row["scope"] for row in rows})
    }
    scope_reports = {
        scope: _scope_report(scope_rows, n_select=int(args.n_select_per_scope), seed=int(args.seed) + idx)
        for idx, (scope, scope_rows) in enumerate(by_scope.items())
    }
    report = {
        "analysis_entry": "phase2_future_basin_spectrum_selector_preflight",
        "schema": "phase2_future_basin_spectrum_selector_preflight.v1",
        "contract": {
            "exploratory_only": True,
            "read_only_existing_artifacts": True,
            "does_not_modify_exact_scm": True,
            "does_not_modify_fit_model": True,
            "does_not_modify_ppo": True,
            "not_a_raw_delta_gate": True,
        },
        "spectrum_source": str(Path(args.spectrum_json).expanduser().resolve()),
        "available_labels": list(AVAILABLE_LABELS),
        "missing_generator_labels": list(MISSING_GENERATOR_LABELS),
        "scope_reports": scope_reports,
        "read": {
            "selector_mechanics_ready": True,
            "fully_unified_sampling_ready": False,
            "why_not_ready": (
                "broad label balancing works, but Pendulum short-temporal settle and MountainCar delayed "
                "future-basin labels are not yet present on the same generator-level candidate rows"
            ),
            "spectrum_open_items": (spectrum.get("remaining_progress", {}) or {}).get("open", []),
        },
    }
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "future_basin_spectrum_selector_preflight.json"
    md_path = out_dir / "future_basin_spectrum_selector_preflight.md"
    csv_path = out_dir / "future_basin_spectrum_selector_selected.csv"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(csv_path, scope_reports)
    md_path.write_text(_markdown(report) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output_dir": str(out_dir),
                "selector_mechanics_ready": True,
                "fully_unified_sampling_ready": False,
                "missing_generator_labels": list(MISSING_GENERATOR_LABELS),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return report


def _write_csv(path: Path, scope_reports: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["scope", "env_idx", "labels"])
        writer.writeheader()
        for scope, payload in scope_reports.items():
            for row in payload["selected_envs"]:
                writer.writerow({"scope": scope, "env_idx": row["env_idx"], "labels": ";".join(row["labels"])})


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Future-Basin Spectrum Selector Preflight",
        "",
        "Read-only selector mechanics test. It does not generate environments or optimize raw deltas.",
        "",
        "## Available Labels",
        "",
    ]
    for label in report["available_labels"]:
        lines.append(f"- `{label}`")
    lines.extend(["", "## Missing Generator-Level Labels", ""])
    for label in report["missing_generator_labels"]:
        lines.append(f"- `{label}`")
    lines.extend(
        [
            "",
            "## Scope Rates",
            "",
            "| scope | n pool | n selected | label | pool | target | selected |",
            "|---|---:|---:|---|---:|---:|---:|",
        ]
    )
    for scope, payload in report["scope_reports"].items():
        for label in report["available_labels"]:
            lines.append(
                f"| `{scope}` | {payload['n_pool']} | {payload['n_selected']} | `{label}` | "
                f"{payload['pool_rates'][label]:.3f} | {payload['target_rates'][label]:.3f} | "
                f"{payload['selected_rates'][label]:.3f} |"
            )
    lines.extend(["", "## Read", "", report["read"]["why_not_ready"]])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--unified-profile-json", default=str(DEFAULT_UNIFIED_PROFILE_JSON))
    parser.add_argument("--spectrum-json", default=str(DEFAULT_SPECTRUM_JSON))
    parser.add_argument("--n-select-per-scope", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260512)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
