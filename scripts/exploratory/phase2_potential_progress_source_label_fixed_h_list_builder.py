#!/usr/bin/env python3
"""Materialize a minimal potential/progress reward source-label fixed h-list.

This is a fixed h-list source-label sampler, not a generator rewrite.  It is
only licensed after the true-runner reconstruction/base-preservation audit has
proved that the lowtail progress and potential-delta formula path is visible in
the real runner image.

The sampler adds one source axis:

* progress_only: lowtail progress reward path active, potential-delta weight 0.
* potential_delta_progress: same path plus a mild potential-delta term.

The base Exact-SCM reward is deliberately retained through a bounded reward mix
instead of being overwritten.  Terminal structure, action/state/obs/noise/zero
dimensions, terminal-event reward labels, and PPO settings are left untouched.
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
DEFAULT_INPUT_CSV = (
    ART
    / "phase2_terminal_event_source_label_fixed_h_list_0518"
    / "terminal_event_source_label_fixed_h_list.csv"
)
DEFAULT_FORMULA_AUDIT = (
    ART
    / "phase2_potential_progress_component_reconstruction_audit_cuda_0518"
    / "potential_progress_component_reconstruction_audit.json"
)
DEFAULT_OUTPUT_DIR = ART / "phase2_potential_progress_source_label_fixed_h_list_0518"
LABELS = ["progress_only", "potential_delta_progress"]
RULE = "lowtail_progress_potential_delta_source_label_v1"
STRUCTURE_BIN_COLUMNS = [
    "action_dim_source_bin",
    "state_dim_source_bin",
    "obs_dim_source_bin",
    "terminal_target_source_bin",
    "realized_noise_dim_bin",
    "realized_zero_pad_dim_bin",
]
MODE_BIN_COLUMNS = [
    "terminal_event_reward_axis_target",
    "reward_component_axis_target",
    "terminal_formula_axis_target",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
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


def _read_json(path: Path) -> dict[str, Any]:
    with path.expanduser().resolve().open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_h(row: dict[str, str]) -> dict[str, Any]:
    raw = str(row.get("full_frozen_h_json", "")).strip()
    path = Path(raw).expanduser()
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return json.loads(raw)


def _safe_float(value: Any, default: float) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _assign_labels(n: int, seed: int) -> list[str]:
    order = list(range(n))
    random.Random(int(seed)).shuffle(order)
    out = [LABELS[0] for _ in range(n)]
    for rank, idx in enumerate(order):
        out[idx] = LABELS[rank % len(LABELS)]
    return out


def _cramers_v(xs: list[str], ys: list[str]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if x != "missing" and y != "missing"]
    if not pairs:
        return 0.0
    x_labels = sorted({x for x, _ in pairs})
    y_labels = sorted({y for _, y in pairs})
    if len(x_labels) <= 1 or len(y_labels) <= 1:
        return 0.0
    xi = {label: idx for idx, label in enumerate(x_labels)}
    yi = {label: idx for idx, label in enumerate(y_labels)}
    table = [[0.0 for _ in y_labels] for _ in x_labels]
    for x, y in pairs:
        table[xi[x]][yi[y]] += 1.0
    n = float(len(pairs))
    row_sum = [sum(row) for row in table]
    col_sum = [sum(table[i][j] for i in range(len(x_labels))) for j in range(len(y_labels))]
    chi2 = 0.0
    for i in range(len(x_labels)):
        for j in range(len(y_labels)):
            expected = row_sum[i] * col_sum[j] / n
            if expected > 0.0:
                chi2 += (table[i][j] - expected) ** 2 / expected
    denom = n * max(1, min(len(x_labels) - 1, len(y_labels) - 1))
    return math.sqrt(max(0.0, chi2 / denom)) if denom > 0.0 else 0.0


def _require_formula_audit_pass(path: Path) -> dict[str, Any]:
    audit = _read_json(path)
    read = audit.get("read", {}) or {}
    required = {
        "hard_pass": True,
        "component_logging_pass": True,
        "progress_observed": True,
        "potential_delta_observed": True,
        "reconstruction_pass": True,
        "base_preservation_pass": True,
        "source_label_sampler_authorized_after_this_audit": True,
    }
    missing = [key for key, expected in required.items() if bool(read.get(key)) is not bool(expected)]
    if missing:
        raise RuntimeError(
            "potential/progress source-label sampler is not authorized; "
            f"failed audit keys={missing} audit={path}"
        )
    return audit


def _apply_potential_progress_axis(
    h: dict[str, Any],
    label: str,
    *,
    reward_mix: float,
    linear_weight: float,
    potential_delta_weight: float,
    state_cost: float,
    ctrl_cost: float,
) -> None:
    if label not in set(LABELS):
        raise ValueError(f"unknown potential/progress source label: {label}")

    h["potential_progress_reward_axis_target"] = str(label)
    h["potential_progress_reward_axis_rule"] = RULE
    h["potential_progress_reward_formula_path"] = (
        "reward_progress_component + reward_potential_delta_component "
        "+ state_cost + ctrl_cost, through exact_scm_gym_lowtail wrapper"
    )
    h["potential_progress_preserves_base_reward_via_mix"] = True

    h["exact_scm_gym_lowtail_family_enabled"] = True
    h["exact_scm_gym_lowtail_family_selected"] = True
    h["exact_scm_gym_lowtail_family_prob"] = 1.0
    h["exact_scm_gym_lowtail_reward_mix"] = float(max(0.0, min(1.0, reward_mix)))
    h["exact_scm_gym_lowtail_reward_linear_weight"] = float(linear_weight)
    h["exact_scm_gym_lowtail_reward_potential_delta_weight"] = (
        0.0 if label == "progress_only" else float(potential_delta_weight)
    )
    h["exact_scm_gym_lowtail_reward_state_cost"] = float(max(0.0, state_cost))
    h["exact_scm_gym_lowtail_reward_ctrl_cost"] = float(max(0.0, ctrl_cost))

    # Keep this sampler focused on the audited formula path.  Other lowtail
    # preimage branches remain explicit future axes, not hidden confounders here.
    h["exact_scm_gym_lowtail_materialized_theta_preimage_enabled"] = False
    h["exact_scm_gym_lowtail_materialized_theta"] = None
    h["exact_scm_gym_lowtail_reward_distribution_head_enabled"] = False
    h["exact_scm_gym_lowtail_clock_preimage_enabled"] = False
    h["exact_scm_gym_lowtail_paired_response_preimage_enabled"] = False
    h["exact_scm_gym_lowtail_shared_energy_terminal_enabled"] = False


def _summarize_coupling(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    xs = [str(row["potential_progress_reward_axis_target"]) for row in rows]
    for axis in [*STRUCTURE_BIN_COLUMNS, *MODE_BIN_COLUMNS]:
        ys = [str(row.get(axis, "missing")) for row in rows]
        v = _cramers_v(xs, ys)
        out.append(
            {
                "source_axis": "potential_progress_reward_axis_target",
                "other_axis": axis,
                "cramers_v": v,
                "status": "high" if v >= 0.50 else ("medium" if v >= 0.25 else "low"),
            }
        )
    return out


def _best_independent_labels(rows: list[dict[str, str]], seed: int, trials: int) -> list[str]:
    axes = [*STRUCTURE_BIN_COLUMNS, *MODE_BIN_COLUMNS]
    n = len(rows)
    best_labels = _assign_labels(n, seed)
    best_score: tuple[int, int, float, float] | None = None
    rng = random.Random(int(seed))
    for trial in range(max(1, int(trials))):
        trial_seed = int(seed) + 104729 * trial + rng.randrange(0, 104729)
        labels = _assign_labels(n, trial_seed)
        tmp_rows = [dict(row, potential_progress_reward_axis_target=labels[idx]) for idx, row in enumerate(rows)]
        coupling = _summarize_coupling(tmp_rows)
        high = sum(1 for row in coupling if row["status"] == "high")
        medium = sum(1 for row in coupling if row["status"] == "medium")
        finite_vs = [float(row["cramers_v"]) for row in coupling if math.isfinite(float(row["cramers_v"]))]
        max_v = max(finite_vs or [0.0])
        sum_v = sum(finite_vs)
        score = (high, medium, max_v, sum_v)
        if best_score is None or score < best_score:
            best_score = score
            best_labels = labels
            if score[0] == 0 and score[1] == 0 and score[2] < 0.25:
                break
    return best_labels


def run(args: argparse.Namespace) -> dict[str, Any]:
    input_csv = Path(args.input_csv).expanduser().resolve()
    formula_audit_path = Path(args.formula_audit).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    h_dir = out_dir / "fixed_h"
    h_dir.mkdir(parents=True, exist_ok=True)

    formula_audit = _require_formula_audit_pass(formula_audit_path)
    input_rows = _read_csv(input_csv)
    labels = _best_independent_labels(input_rows, int(args.seed), int(args.assignment_trials))
    output_rows: list[dict[str, Any]] = []
    fixed_h_list: list[dict[str, Any]] = []
    fixed_env_seeds: list[int] = []

    for idx, row in enumerate(input_rows):
        h = _read_h(row)
        _apply_potential_progress_axis(
            h,
            labels[idx],
            reward_mix=float(args.reward_mix),
            linear_weight=_safe_float(
                h.get("exact_scm_gym_lowtail_reward_linear_weight"),
                float(args.linear_weight),
            ),
            potential_delta_weight=float(args.potential_delta_weight),
            state_cost=float(args.state_cost),
            ctrl_cost=float(args.ctrl_cost),
        )
        h_path = h_dir / f"potential_progress_source_env_{idx:04d}.json"
        h_path.write_text(json.dumps(_json_safe(h), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        env_seed = int(float(row.get("env_seed", h.get("seed", idx))))
        out_row: dict[str, Any] = dict(row)
        out_row["env_idx"] = idx
        out_row["env_seed"] = env_seed
        out_row["rule"] = RULE
        out_row["full_frozen_h_json"] = str(h_path)
        out_row["potential_progress_reward_axis_target"] = labels[idx]
        out_row["potential_progress_reward_axis_rule"] = RULE
        out_row["exact_scm_gym_lowtail_reward_mix"] = h["exact_scm_gym_lowtail_reward_mix"]
        out_row["exact_scm_gym_lowtail_reward_linear_weight"] = h[
            "exact_scm_gym_lowtail_reward_linear_weight"
        ]
        out_row["exact_scm_gym_lowtail_reward_potential_delta_weight"] = h[
            "exact_scm_gym_lowtail_reward_potential_delta_weight"
        ]
        out_row["base_reward_retention_fraction"] = 1.0 - float(h["exact_scm_gym_lowtail_reward_mix"])
        output_rows.append(out_row)
        fixed_h_list.append(h)
        fixed_env_seeds.append(env_seed)

    fixed_csv = out_dir / "potential_progress_source_label_fixed_h_list.csv"
    coupling_csv = out_dir / "potential_progress_source_label_coupling.csv"
    fixed_h_list_json = out_dir / "prior_fixed_h_list.json"
    fixed_env_seeds_json = out_dir / "prior_fixed_env_seeds.json"
    _write_csv(fixed_csv, output_rows)
    coupling_rows = _summarize_coupling(output_rows)
    _write_csv(coupling_csv, coupling_rows)
    fixed_h_list_json.write_text(
        json.dumps(_json_safe(fixed_h_list), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    fixed_env_seeds_json.write_text(
        json.dumps([int(v) for v in fixed_env_seeds], indent=2) + "\n",
        encoding="utf-8",
    )

    report = {
        "analysis_entry": "phase2_potential_progress_source_label_fixed_h_list_builder",
        "schema": "phase2_potential_progress_source_label_fixed_h_list_builder.v1",
        "contract": {
            "fixed_h_list_materialization_only": True,
            "does_not_modify_generator": True,
            "does_not_modify_ppo": True,
            "does_not_train": True,
            "formula_path_pre_authorized": True,
            "base_reward_preserved_by_reward_mix": True,
            "terminal_structure_untouched": True,
            "dim_sources_untouched": True,
        },
        "inputs": {
            "input_csv": str(input_csv),
            "formula_audit": str(formula_audit_path),
        },
        "outputs": {
            "fixed_csv": str(fixed_csv),
            "fixed_h_dir": str(h_dir),
            "fixed_h_list_json": str(fixed_h_list_json),
            "fixed_env_seeds_json": str(fixed_env_seeds_json),
            "coupling_csv": str(coupling_csv),
            "json": str(out_dir / "potential_progress_source_label_fixed_h_list_report.json"),
            "markdown": str(out_dir / "potential_progress_source_label_fixed_h_list_report.md"),
        },
        "read": {
            "label_counts": dict(Counter(labels)),
            "high_coupling_count": sum(1 for row in coupling_rows if row["status"] == "high"),
            "medium_coupling_count": sum(1 for row in coupling_rows if row["status"] == "medium"),
            "reward_mix": float(args.reward_mix),
            "base_reward_retention_fraction": 1.0 - float(args.reward_mix),
            "potential_delta_weight": float(args.potential_delta_weight),
            "formula_audit_hard_pass": bool((formula_audit.get("read", {}) or {}).get("hard_pass")),
            "source_label_sampler_materialized": True,
            "requires_next": [
                "true-runner sidecar rerun",
                "category span",
                "structure-mode independence",
                "pack-fit parity",
                "per-cell PPO health/optimizability",
            ],
        },
        "coupling": coupling_rows,
    }
    Path(report["outputs"]["json"]).write_text(
        json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Potential / Progress Source-Label Fixed H-List",
        "",
        f"- fixed csv: `{fixed_csv}`",
        f"- label counts: `{report['read']['label_counts']}`",
        f"- reward mix: `{report['read']['reward_mix']}`",
        f"- base reward retention: `{report['read']['base_reward_retention_fraction']}`",
        f"- high coupling count: `{report['read']['high_coupling_count']}`",
        f"- medium coupling count: `{report['read']['medium_coupling_count']}`",
        "- next: true-runner runner/audit gates before milestone use.",
    ]
    Path(report["outputs"]["markdown"]).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report["read"], indent=2, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--formula-audit", type=Path, default=DEFAULT_FORMULA_AUDIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=9520)
    parser.add_argument("--assignment-trials", type=int, default=4096)
    parser.add_argument("--reward-mix", type=float, default=0.35)
    parser.add_argument("--linear-weight", type=float, default=1.25)
    parser.add_argument("--potential-delta-weight", type=float, default=0.65)
    parser.add_argument("--state-cost", type=float, default=0.05)
    parser.add_argument("--ctrl-cost", type=float, default=0.02)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
