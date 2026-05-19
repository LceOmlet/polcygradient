#!/usr/bin/env python
"""Formula/topology label audit for the maintained fixed h-list family.

This is a read-only category-span audit.  It classifies formula axes from the
fixed h-list plus true-runner sidecar logs, without changing the generator or
PPO path.  The purpose is to separate:

* source/realized structural span, which is already audited elsewhere,
* formula topology span, such as reward component path and terminal mechanism,
* observed outcome density, such as done_count bins, which is not a generator
  formula axis by itself.

Collapsed diagnostic axes are reported, but they are not automatically treated
as generator failures.  A collapsed formula axis becomes a hard failure only if
the report marks it as required for the current category contract.
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
DEFAULT_RUN_DIR = ART / "phase2_stratified_source_fixed_h_list_runner_u2_s128_0518"
DEFAULT_RANK_DIR = ART / "phase2_category_controllability_rank_audit_stratified_0518"
DEFAULT_OUTPUT_DIR = ART / "phase2_category_topology_label_audit_stratified_0518"


Rows = list[dict[str, Any]]


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


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "on"}


def _latest_sidecar(run_dir: Path) -> Path:
    candidates = sorted((run_dir / "semantic_sidecars").glob("*_envs.jsonl"))
    if not candidates:
        raise FileNotFoundError(f"no sidecar found under {run_dir / 'semantic_sidecars'}")
    return candidates[-1]


def _entropy(counts: Counter[str]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    out = 0.0
    for count in counts.values():
        if count <= 0:
            continue
        p = count / total
        out -= p * math.log(p)
    return out


def _norm_entropy(counts: Counter[str], expected: int) -> float:
    if expected <= 1:
        return 1.0
    return _entropy(counts) / math.log(expected)


def _axis_status(counts: Counter[str], expected_labels: list[str]) -> str:
    if not expected_labels:
        expected_labels = sorted(counts)
    expected = len(expected_labels)
    occupied = sum(1 for label in expected_labels if counts.get(label, 0) > 0)
    ent = _norm_entropy(Counter({label: counts.get(label, 0) for label in expected_labels}), expected)
    if expected <= 1:
        return "single_allowed"
    if occupied == expected and ent >= 0.90:
        return "pass_uniform"
    if occupied == expected:
        return "covered_skewed"
    if occupied > 1:
        return "partial"
    return "collapsed"


def _bin3(value: Any, lo_hi: tuple[float, float]) -> str:
    x = _safe_float(value)
    if not math.isfinite(x):
        return "missing"
    if x <= lo_hi[0]:
        return "low"
    if x <= lo_hi[1]:
        return "mid"
    return "high"


def _component_present(row: dict[str, Any], prefix: str, *, eps: float = 1e-8) -> bool:
    for suffix in ("std", "sum", "mean", "q90"):
        if abs(_safe_float(row.get(f"side:{prefix}_{suffix}"), 0.0)) > eps:
            return True
    if _safe_bool(row.get(f"side:{prefix}_q90_gt0")):
        return True
    return False


def _reward_component_topology(row: dict[str, Any]) -> str:
    parts: list[str] = []
    if _component_present(row, "reward_env"):
        parts.append("env")
    if _component_present(row, "reward_ctrl"):
        parts.append("ctrl")
    if _component_present(row, "reward_survival"):
        parts.append("survival")
    return "+".join(parts) if parts else "zero"


def _reward_transform_topology(row: dict[str, Any]) -> str:
    log = row.get("side:reward_component_logging")
    if isinstance(log, dict):
        transform = log.get("reinforce_reward_transform")
    else:
        transform = row.get("h:reinforce_reward_transform")
    text = str(transform or "none").strip().lower()
    if text in {"", "none", "false", "0"}:
        return "none"
    if "tanh" in text:
        return "tanh"
    if "clip" in text:
        return "clip"
    return text


def _reward_density_topology(row: dict[str, Any]) -> str:
    vals = {}
    if _safe_float(row.get("side:state_dim"), 0.0) > 0.0:
        vals["state"] = _safe_float(row.get("side:reward_state_input_gain_density_ratio"))
    if _safe_float(row.get("side:action_dim"), 0.0) > 0.0:
        vals["action"] = _safe_float(row.get("side:reward_action_input_gain_density_ratio"))
    if _safe_float(row.get("side:noise_dim"), 0.0) > 0.0:
        vals["noise"] = _safe_float(row.get("side:reward_noise_input_gain_density_ratio"))
    finite = {k: v for k, v in vals.items() if math.isfinite(v) and v > 0.0}
    if len(finite) < 2:
        return "missing"
    ordered = sorted(finite.items(), key=lambda kv: kv[1], reverse=True)
    max_name, max_val = ordered[0]
    min_val = ordered[-1][1]
    if min_val <= 0.0:
        return f"{max_name}_singular"
    ratio = max_val / min_val
    if ratio <= 1.5:
        return "density_balanced"
    return f"{max_name}_density_dominant"


def _reward_value_shape(row: dict[str, Any]) -> str:
    std = _safe_float(row.get("side:raw_reward_std"), 0.0)
    q10 = _safe_float(row.get("side:raw_reward_q10"), 0.0)
    q90 = _safe_float(row.get("side:raw_reward_q90"), 0.0)
    if std <= 1e-8:
        return "constant"
    if q10 < 0.0 and q90 > 0.0:
        return "signed"
    if q90 <= 0.0:
        return "nonpositive"
    if q10 >= 0.0:
        return "nonnegative"
    return "mixed"


def _terminal_formula_topology(row: dict[str, Any]) -> str:
    if not _safe_bool(row.get("h:terminal_reset_enabled")) and not _safe_bool(row.get("side:terminal_reset_enabled")):
        return "terminal_off"
    lowtail_active = (
        _safe_bool(row.get("h:exact_scm_gym_lowtail_family_selected"))
        or _safe_bool(row.get("h:exact_scm_gym_lowtail_family_enabled"))
        or _safe_bool(row.get("side:exact_scm_gym_lowtail_family_active"))
    )
    if lowtail_active and (
        _safe_bool(row.get("h:exact_scm_gym_lowtail_shared_energy_terminal_enabled"))
        or _safe_bool(row.get("side:exact_scm_gym_lowtail_shared_energy_terminal_enabled"))
    ):
        return "shared_energy_terminal"
    if _safe_bool(row.get("h:terminal_reset_min_step_enabled")):
        return "min_step_hazard_reset"
    if _safe_bool(row.get("h:exact_scm_gym_lowtail_terminal_override_enabled")):
        return "override_hazard_reset"
    return "hazard_reset"


def _terminal_observed_topology(row: dict[str, Any]) -> str:
    done = _safe_float(row.get("side:done_count"), 0.0)
    trunc = _safe_float(row.get("side:truncated_count"), 0.0)
    if done <= 0.0 and trunc <= 0.0:
        return "no_episode_boundary_observed"
    if done <= 0.0 and trunc > 0.0:
        return "time_limit_only_observed"
    if trunc <= 0.0:
        return "done_only_observed"
    return "done_plus_time_limit_observed"


def _bin_done(row: dict[str, Any]) -> str:
    done = _safe_float(row.get("side:done_count"), 0.0)
    if done <= 0:
        return "zero"
    if done < 20:
        return "low"
    if done < 50:
        return "mid"
    return "high"


def _join(h_rows: Rows, side_rows: Rows, rank_rows: Rows) -> Rows:
    side_by_idx = {int(row.get("env_idx", idx)): row for idx, row in enumerate(side_rows)}
    rank_by_idx = {int(row.get("env_idx", idx)): row for idx, row in enumerate(rank_rows)}
    rows: Rows = []
    for idx, h in enumerate(h_rows):
        row: dict[str, Any] = {"env_idx": idx}
        for key, value in h.items():
            row[f"h:{key}"] = value
        for key, value in side_by_idx.get(idx, {}).items():
            row[f"side:{key}"] = value
        for key, value in rank_by_idx.get(idx, {}).items():
            row[f"rank:{key}"] = value
        rows.append(row)
    return rows


STRUCTURE_AXES = {
    "state_dim": ("side:state_dim", (133.0, 266.0)),
    "obs_dim": ("side:obs_dim", (133.0, 266.0)),
    "action_dim": ("side:action_dim", (10.0, 20.0)),
    "terminal_target": ("side:terminal_reset_count_target", (26.6667, 53.3333)),
    "noise_dim": ("side:noise_dim", (133.0, 266.0)),
    "zero_pad_dim": ("side:zero_pad_dim", (133.0, 266.0)),
}


TOPOLOGY_SPECS = [
    {
        "axis": "reward_component_topology",
        "kind": "formula",
        "required_for_current_contract": True,
        "expected_labels": [
            "env",
            "env+ctrl",
            "env+survival",
            "env+ctrl+survival",
        ],
        "minimal_repair_if_required": (
            "sample ctrl/survival component source labels explicitly while keeping base env reward present; "
            "do not treat reward_aux as an independent component because it is ctrl+survival aggregate logging"
        ),
    },
    {
        "axis": "reward_transform_topology",
        "kind": "formula",
        "required_for_current_contract": False,
        "expected_labels": ["none", "tanh", "clip"],
        "minimal_repair_if_required": "sample reward transform as an explicit source axis only if transform diversity is part of the category",
    },
    {
        "axis": "reward_density_topology",
        "kind": "formula",
        "required_for_current_contract": False,
        "expected_labels": [
            "density_balanced",
            "state_density_dominant",
            "action_density_dominant",
            "noise_density_dominant",
        ],
        "minimal_repair_if_required": "sample density-allocation topology directly; do not use dim fractions as a proxy",
    },
    {
        "axis": "reward_value_shape",
        "kind": "observed_formula_image",
        "required_for_current_contract": False,
        "expected_labels": ["signed", "nonpositive", "nonnegative", "constant"],
        "minimal_repair_if_required": "repair reward head/value-shape source after confirming the missing shape is required",
    },
    {
        "axis": "terminal_formula_topology",
        "kind": "formula",
        "required_for_current_contract": False,
        "expected_labels": [
            "hazard_reset",
            "override_hazard_reset",
            "min_step_hazard_reset",
            "shared_energy_terminal",
            "terminal_off",
        ],
        "minimal_repair_if_required": "sample terminal mechanism topology explicitly; do not tune done_count as a proxy",
    },
    {
        "axis": "terminal_observed_topology",
        "kind": "observed_outcome",
        "required_for_current_contract": False,
        "expected_labels": [
            "no_episode_boundary_observed",
            "time_limit_only_observed",
            "done_only_observed",
            "done_plus_time_limit_observed",
        ],
        "minimal_repair_if_required": "not a formula repair target by itself; first map back to terminal_formula_topology",
    },
    {
        "axis": "done_count_bin",
        "kind": "observed_outcome",
        "required_for_current_contract": False,
        "expected_labels": ["zero", "low", "mid", "high"],
        "minimal_repair_if_required": "not a formula repair target by itself; terminal target is the source axis",
    },
    {
        "axis": "controllability_rank",
        "kind": "formula_evidence",
        "required_for_current_contract": True,
        "expected_labels": ["full", "nonfull"],
        "minimal_repair_if_required": "repair transition action-coupling rank source only if this fails",
    },
    {
        "axis": "reward_action_dependency",
        "kind": "formula_evidence",
        "required_for_current_contract": True,
        "expected_labels": ["sensitive", "insensitive"],
        "minimal_repair_if_required": "repair reward action path only if this fails",
    },
]


def _labels_for_row(row: dict[str, Any]) -> dict[str, str]:
    state_rank = _safe_float(row.get("rank:state_action_rank_ratio"), 0.0)
    obs_rank = _safe_float(row.get("rank:obs_action_rank_ratio"), 0.0)
    reward_sensitive = _safe_bool(row.get("rank:reward_action_sensitive"))
    return {
        "reward_component_topology": _reward_component_topology(row),
        "reward_transform_topology": _reward_transform_topology(row),
        "reward_density_topology": _reward_density_topology(row),
        "reward_value_shape": _reward_value_shape(row),
        "terminal_formula_topology": _terminal_formula_topology(row),
        "terminal_observed_topology": _terminal_observed_topology(row),
        "done_count_bin": _bin_done(row),
        "controllability_rank": "full" if state_rank >= 1.0 and obs_rank >= 1.0 else "nonfull",
        "reward_action_dependency": "sensitive" if reward_sensitive else "insensitive",
    }


def _label_rows(rows: Rows) -> Rows:
    out: Rows = []
    for row in rows:
        labels = _labels_for_row(row)
        item = {"env_idx": int(row["env_idx"])}
        item.update(labels)
        for name, (key, edges) in STRUCTURE_AXES.items():
            item[f"struct:{name}"] = _bin3(row.get(key), edges)
        out.append(item)
    return out


def _axis_summary(label_rows: Rows) -> Rows:
    out: Rows = []
    specs = {spec["axis"]: spec for spec in TOPOLOGY_SPECS}
    for axis, spec in specs.items():
        expected = list(spec["expected_labels"])
        counts = Counter(str(row.get(axis, "missing")) for row in label_rows)
        labels = expected
        status = _axis_status(counts, labels)
        required = bool(spec["required_for_current_contract"])
        hard_pass = (not required) or status in {"single_allowed", "pass_uniform", "covered_skewed"} or (
            axis in {"controllability_rank", "reward_action_dependency"}
            and counts.get("full", 0) + counts.get("sensitive", 0) == len(label_rows)
        )
        item: dict[str, Any] = {
            "axis": axis,
            "kind": spec["kind"],
            "required_for_current_contract": required,
            "status": status,
            "hard_pass": bool(hard_pass),
            "n": len(label_rows),
            "occupied_expected_labels": sum(1 for label in labels if counts.get(label, 0) > 0),
            "expected_labels": len(labels),
            "normalized_entropy": _norm_entropy(Counter({label: counts.get(label, 0) for label in labels}), len(labels)),
            "max_label_fraction": max([counts.get(label, 0) / max(1, len(label_rows)) for label in labels] or [0.0]),
            "minimal_repair_if_required": spec["minimal_repair_if_required"],
        }
        for label in labels:
            item[f"count:{label}"] = counts.get(label, 0)
        extra = sorted(set(counts) - set(labels))
        if extra:
            item["extra_labels"] = ",".join(extra)
        out.append(item)
    return out


def _cramers_v(xs: list[str], ys: list[str]) -> float:
    n = len(xs)
    if n == 0:
        return 0.0
    x_labels = sorted(set(xs))
    y_labels = sorted(set(ys))
    if len(x_labels) <= 1 or len(y_labels) <= 1:
        return 0.0
    x_idx = {v: i for i, v in enumerate(x_labels)}
    y_idx = {v: i for i, v in enumerate(y_labels)}
    table = [[0.0 for _ in y_labels] for _ in x_labels]
    for x, y in zip(xs, ys):
        table[x_idx[x]][y_idx[y]] += 1.0
    row_sum = [sum(row) for row in table]
    col_sum = [sum(table[i][j] for i in range(len(x_labels))) for j in range(len(y_labels))]
    chi2 = 0.0
    for i in range(len(x_labels)):
        for j in range(len(y_labels)):
            expected = row_sum[i] * col_sum[j] / n
            if expected > 0:
                chi2 += (table[i][j] - expected) ** 2 / expected
    denom = n * max(1, min(len(x_labels) - 1, len(y_labels) - 1))
    return math.sqrt(max(0.0, chi2 / denom)) if denom > 0 else 0.0


def _coupling_summary(label_rows: Rows) -> Rows:
    out: Rows = []
    topology_axes = [spec["axis"] for spec in TOPOLOGY_SPECS if spec["kind"] in {"formula", "formula_evidence"}]
    for topo in topology_axes:
        xs = [str(row.get(topo, "missing")) for row in label_rows]
        for struct in STRUCTURE_AXES:
            ys = [str(row.get(f"struct:{struct}", "missing")) for row in label_rows]
            v = _cramers_v(xs, ys)
            status = "low"
            if len(set(xs)) <= 1:
                status = "undefined_collapsed_topology"
            elif v >= 0.50:
                status = "high"
            elif v >= 0.25:
                status = "medium"
            out.append(
                {
                    "topology_axis": topo,
                    "structure_axis": struct,
                    "cramers_v": float(v),
                    "status": status,
                }
            )
    return out


def _read(axis_rows: Rows, coupling_rows: Rows) -> dict[str, Any]:
    hard_fail = [row for row in axis_rows if bool(row["required_for_current_contract"]) and not bool(row["hard_pass"])]
    collapsed_formula_optional = [
        row
        for row in axis_rows
        if row["kind"] == "formula" and row["status"] == "collapsed" and not bool(row["required_for_current_contract"])
    ]
    high_coupling = [row for row in coupling_rows if row["status"] == "high"]
    return {
        "hard_topology_pass": bool(not hard_fail),
        "hard_topology_fail_axes": [row["axis"] for row in hard_fail],
        "optional_collapsed_formula_axes": [row["axis"] for row in collapsed_formula_optional],
        "structure_topology_high_coupling_count": len(high_coupling),
        "structure_topology_high_coupling_pairs": [
            [row["topology_axis"], row["structure_axis"], row["cramers_v"]] for row in high_coupling[:20]
        ],
        "interpretation": (
            "Required formula evidence is closed if hard_topology_pass is true. "
            "Optional collapsed formula axes are not automatic failures; they are candidates for a category decision."
        ),
    }


def _markdown(report: dict[str, Any]) -> str:
    read = report["read"]
    lines = [
        "# Category Topology Label Audit",
        "",
        "Read-only audit of formula topology labels from fixed h-list plus true-runner sidecar.",
        "",
        "## Read",
        "",
        f"- hard topology pass: `{read['hard_topology_pass']}`",
        f"- hard topology fail axes: `{read['hard_topology_fail_axes']}`",
        f"- optional collapsed formula axes: `{read['optional_collapsed_formula_axes']}`",
        f"- structure-topology high coupling count: `{read['structure_topology_high_coupling_count']}`",
        "",
        "## Axis Summary",
        "",
        "| axis | kind | required | status | hard pass | occupied/expected | entropy | max label |",
        "|---|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in report["axis_summary"]:
        lines.append(
            f"| `{row['axis']}` | `{row['kind']}` | `{row['required_for_current_contract']}` | "
            f"`{row['status']}` | `{row['hard_pass']}` | "
            f"{row['occupied_expected_labels']}/{row['expected_labels']} | "
            f"{row['normalized_entropy']:.3f} | {row['max_label_fraction']:.3f} |"
        )
    lines.extend(["", "## Minimal Repair Rules If Promoted To Required", ""])
    for row in report["axis_summary"]:
        if row["kind"] == "formula":
            lines.append(f"- `{row['axis']}`: {row['minimal_repair_if_required']}")
    lines.extend(["", "## High Coupling", ""])
    highs = [row for row in report["structure_topology_coupling"] if row["status"] == "high"]
    if not highs:
        lines.append("- none")
    else:
        for row in highs[:50]:
            lines.append(
                f"- `{row['topology_axis']}` vs `{row['structure_axis']}`: Cramer's V `{row['cramers_v']:.3f}`"
            )
    lines.extend(["", "## Outputs", ""])
    for key, value in report["outputs"].items():
        lines.append(f"- `{key}`: `{value}`")
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = Path(args.run_dir).expanduser().resolve()
    h_list_path = Path(args.h_list).expanduser().resolve() if args.h_list else run_dir / "fixed_env_group/prior_fixed_h_list.json"
    sidecar_path = Path(args.sidecar).expanduser().resolve() if args.sidecar else _latest_sidecar(run_dir)
    rank_path = (
        Path(args.rank_csv).expanduser().resolve()
        if args.rank_csv
        else Path(args.rank_dir).expanduser().resolve() / "category_controllability_rank_per_env.csv"
    )
    h_rows = _read_json(h_list_path)
    side_rows = _read_jsonl(sidecar_path)
    rank_rows: Rows = []
    if rank_path.exists():
        with rank_path.open("r", encoding="utf-8", newline="") as handle:
            rank_rows = list(csv.DictReader(handle))
    rows = _join(h_rows, side_rows, rank_rows)
    label_rows = _label_rows(rows)
    axis_rows = _axis_summary(label_rows)
    coupling_rows = _coupling_summary(label_rows)
    read = _read(axis_rows, coupling_rows)

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": out_dir / "category_topology_label_audit.json",
        "markdown": out_dir / "category_topology_label_audit.md",
        "per_env_csv": out_dir / "category_topology_label_per_env.csv",
        "axis_csv": out_dir / "category_topology_label_axis_summary.csv",
        "coupling_csv": out_dir / "category_topology_label_structure_coupling.csv",
    }
    report = {
        "analysis_entry": "phase2_category_topology_label_audit",
        "schema": "phase2_category_topology_label_audit.v1",
        "contract": {
            "read_only": True,
            "does_not_modify_generator": True,
            "does_not_modify_ppo": True,
            "separates_testing_from_development": True,
            "does_not_promote_optional_axes_to_failures": True,
        },
        "inputs": {
            "run_dir": str(run_dir),
            "h_list": str(h_list_path),
            "sidecar": str(sidecar_path),
            "rank_csv": str(rank_path),
            "h_count": len(h_rows),
            "sidecar_count": len(side_rows),
            "rank_count": len(rank_rows),
        },
        "read": read,
        "axis_summary": axis_rows,
        "structure_topology_coupling": coupling_rows,
        "per_env": label_rows,
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    paths["json"].write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths["markdown"].write_text(_markdown(report), encoding="utf-8")
    _write_csv(paths["per_env_csv"], label_rows)
    _write_csv(paths["axis_csv"], axis_rows)
    _write_csv(paths["coupling_csv"], coupling_rows)
    print(json.dumps({key: str(value) for key, value in paths.items()}, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--h-list", default=None)
    parser.add_argument("--sidecar", default=None)
    parser.add_argument("--rank-dir", default=str(DEFAULT_RANK_DIR))
    parser.add_argument("--rank-csv", default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
