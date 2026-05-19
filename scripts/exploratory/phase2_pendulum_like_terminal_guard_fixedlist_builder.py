#!/usr/bin/env python
"""Build a terminal-stability guarded Pendulum-like fixed-list CSV.

Exploratory only. This script reads an existing Pendulum-like fixed-list CSV and
an existing pre-update pack-runner sidecar. It does not resample environments,
does not edit exact SCM or PPO code, and uses only pre-update rollout stability
fields for the guard.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_INPUT_CSV = (
    "/home/chen/RLPFN/artifacts/phase2_pendulum_like_strict_cross_seed_all_candidates_0509/"
    "pendulum_like_signature_fixedlist.csv"
)
DEFAULT_SIDECAR = (
    "/home/chen/RLPFN/artifacts/phase2_pendulum_like_strict_n92_preupdate_sidecar_s512_u1_e1_advnorm_0509/"
    "semantic_sidecars/prior_update_000000_envs.jsonl"
)
DEFAULT_OUTPUT_DIR = "/home/chen/RLPFN/artifacts/phase2_pendulum_like_strict_terminal_guard_fixedlists_0509"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _f(value: Any) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _stats(values: list[Any]) -> dict[str, Any]:
    arr = np.asarray([float(v) for v in values if _f(v) is not None], dtype=np.float64)
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "min": float(np.min(arr)),
        "q10": float(np.quantile(arr, 0.10)),
        "q50": float(np.quantile(arr, 0.50)),
        "q90": float(np.quantile(arr, 0.90)),
        "max": float(np.max(arr)),
    }


def _load_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _rank_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    # Pre-update terminal stability guard:
    #   1. fewer terminations in the rollout,
    #   2. later first termination,
    #   3. lower action saturation under the initial policy,
    #   4. lower action growth pressure proxy from absolute action mean.
    # It intentionally does not use reward delta or post-update fields.
    done = _f(row.get("guard_done_count"))
    first_done = _f(row.get("guard_first_done_step"))
    sat = _f(row.get("guard_action_saturation"))
    act_abs = _f(row.get("guard_action_absmean"))
    return (
        float(done if done is not None else 1e9),
        -float(first_done if first_done is not None else -1.0),
        float(sat if sat is not None else 1e9),
        float(act_abs if act_abs is not None else 1e9),
    )


def _balanced_select(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_source[str(row.get("source_tag"))].append(row)
    for bucket in by_source.values():
        bucket.sort(key=_rank_key)

    selected = []
    cursors = {name: 0 for name in by_source}
    for source in sorted(by_source):
        bucket = by_source[source]
        if cursors[source] < len(bucket) and len(selected) < int(limit):
            selected.append(bucket[cursors[source]])
            cursors[source] += 1
    while len(selected) < int(limit):
        progressed = False
        for source in sorted(by_source):
            bucket = by_source[source]
            idx = cursors[source]
            if idx >= len(bucket):
                continue
            selected.append(bucket[idx])
            cursors[source] = idx + 1
            progressed = True
            if len(selected) >= int(limit):
                break
        if not progressed:
            break
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", default=DEFAULT_INPUT_CSV)
    parser.add_argument("--preupdate-sidecar", default=DEFAULT_SIDECAR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--input-rule", default="cross_seed_full_q90_pendulum_like_strict_n92")
    parser.add_argument("--output-rule", default="cross_seed_full_q90_pendulum_like_strict_terminal_guard_n64")
    parser.add_argument("--limit", type=int, default=64)
    args = parser.parse_args()

    input_csv = Path(args.input_csv).expanduser().resolve()
    sidecar_path = Path(args.preupdate_sidecar).expanduser().resolve()
    fixed_rows = [
        row for row in _load_csv(input_csv)
        if not str(args.input_rule) or str(row.get("rule")) == str(args.input_rule)
    ]
    sidecar_by_idx = {int(row["env_idx"]): row for row in _load_jsonl(sidecar_path)}
    if len(sidecar_by_idx) < len(fixed_rows):
        raise RuntimeError(
            f"sidecar rows {len(sidecar_by_idx)} fewer than fixed rows {len(fixed_rows)}"
        )

    joined = []
    for idx, fixed in enumerate(fixed_rows):
        side = sidecar_by_idx[int(idx)]
        row = dict(fixed)
        row.update(
            {
                "source_candidate_index": int(idx),
                "guard_done_count": side.get("done_count"),
                "guard_first_done_step": side.get("first_done_step"),
                "guard_action_saturation": side.get("raw_policy_action_unit_clip_saturation_fraction"),
                "guard_action_absmean": side.get("action_value_absmean"),
                "guard_raw_reward_sum": side.get("raw_reward_sum"),
                "guard_reward_env_sum": side.get("reward_env_sum"),
                "guard_training_reward_sum": side.get("training_reward_sum"),
            }
        )
        joined.append(row)

    selected = _balanced_select(joined, int(args.limit))
    if len(selected) < int(args.limit):
        raise RuntimeError(f"selected {len(selected)} rows, requested {args.limit}")

    out_dir = Path(args.output_dir).expanduser().resolve()
    group_dir = out_dir / str(args.output_rule)
    group_dir.mkdir(parents=True, exist_ok=True)

    output_rows = []
    for out_idx, row in enumerate(selected):
        src = Path(str(row["full_frozen_h_json"])).expanduser().resolve()
        dst = group_dir / f"env_{out_idx:03d}_{row['source_tag']}_candidate_{int(row['source_candidate_index']):03d}.json"
        shutil.copyfile(src, dst)
        out_row = dict(row)
        out_row["rule"] = str(args.output_rule)
        out_row["full_frozen_h_json"] = str(dst)
        output_rows.append(out_row)

    fieldnames = list(output_rows[0].keys())
    csv_path = out_dir / "pendulum_like_terminal_guard_fixedlist.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in output_rows:
            writer.writerow(row)

    rejected = [row for row in joined if row not in selected]
    report = {
        "analysis_entry": "phase2_pendulum_like_terminal_guard_fixedlist_builder",
        "contract": {
            "exploratory_only": True,
            "reads_existing_fixed_list_csv": True,
            "reads_existing_preupdate_sidecar": True,
            "does_not_modify_exact_scm": True,
            "does_not_modify_fit_model": True,
            "does_not_modify_ppo": True,
            "does_not_resample_envs": True,
            "guard_uses_preupdate_fields_only": True,
        },
        "input_csv": str(input_csv),
        "preupdate_sidecar": str(sidecar_path),
        "input_rule": str(args.input_rule),
        "output_rule": str(args.output_rule),
        "input_count": int(len(joined)),
        "selected_count": int(len(output_rows)),
        "rejected_count": int(len(rejected)),
        "source_counts_selected": dict(Counter(str(row.get("source_tag")) for row in output_rows)),
        "source_counts_input": dict(Counter(str(row.get("source_tag")) for row in joined)),
        "selected_guard_stats": {
            "done_count": _stats([r.get("guard_done_count") for r in output_rows]),
            "first_done_step": _stats([r.get("guard_first_done_step") for r in output_rows]),
            "action_saturation": _stats([r.get("guard_action_saturation") for r in output_rows]),
            "signature_score": _stats([r.get("signature_score") for r in output_rows]),
        },
        "rejected_guard_stats": {
            "done_count": _stats([r.get("guard_done_count") for r in rejected]),
            "first_done_step": _stats([r.get("guard_first_done_step") for r in rejected]),
            "action_saturation": _stats([r.get("guard_action_saturation") for r in rejected]),
            "signature_score": _stats([r.get("signature_score") for r in rejected]),
        },
        "csv": str(csv_path),
    }
    (out_dir / "pendulum_like_terminal_guard_fixedlist_report.json").write_text(
        json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Pendulum-Like Terminal Guard Fixed List",
        "",
        f"csv: {csv_path}",
        f"rule: {args.output_rule}",
        f"input_count: {len(joined)}",
        f"selected_count: {len(output_rows)}",
        f"rejected_count: {len(rejected)}",
        "",
        "## Guard Definition",
        "",
        "Rank candidates by pre-update terminal stability: lower `done_count`, higher `first_done_step`, lower action saturation, lower initial action absmean. Selection is source-balanced.",
        "",
        "## Source Counts Selected",
        "",
    ]
    for name, count in sorted(report["source_counts_selected"].items()):
        lines.append(f"- {name}: {count}")
    lines.extend(["", "## Selected Guard Stats", ""])
    for key, stats in report["selected_guard_stats"].items():
        lines.append(f"- {key}: {stats}")
    lines.extend(["", "## Rejected Guard Stats", ""])
    for key, stats in report["rejected_guard_stats"].items():
        lines.append(f"- {key}: {stats}")
    (out_dir / "pendulum_like_terminal_guard_fixedlist_summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"csv": str(csv_path), "rule": str(args.output_rule), "selected_count": len(output_rows)}, indent=2))


if __name__ == "__main__":
    main()
