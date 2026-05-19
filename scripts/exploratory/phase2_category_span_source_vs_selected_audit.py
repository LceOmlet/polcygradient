#!/usr/bin/env python
"""Reduce category-span failures to source-vs-selected loci.

This audit keeps testing separate from generator development.  It compares a
large maintained-prior h-only source sample against a selected/fixed h-list.
If the source spans an axis but the selected list does not, the sufficient fix
is fixed-group construction/stratification, not a new environment family.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


ART = Path("/home/chen/RLPFN/artifacts")
REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.exploratory.phase2_category_span_audit import (  # noqa: E402
    _axis_specs,
    _axis_values,
    _coverage_rows,
    _json_safe,
    _read_json,
)
from ticl.model_configs import get_rlpfn_default_config  # noqa: E402
from ticl.priors.environment_prior import EnvironmentPrior  # noqa: E402
from ticl.rlpfn_maintained_path import apply_rlpfn_maintained_path_defaults  # noqa: E402


DEFAULT_SELECTED_H_LIST = (
    ART
    / "phase2_m2_current_runner_probe_0515"
    / "prior_m2_q90_fixed64_s512_u4_e4_advnorm_b2048_0515"
    / "fixed_env_group"
    / "prior_fixed_h_list.json"
)
DEFAULT_OUTPUT_DIR = ART / "phase2_category_span_source_vs_selected_audit_0518"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _h_only_join(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for idx, h in enumerate(rows):
        row = {"env_idx": idx}
        for key, value in h.items():
            row[f"h:{key}"] = value
        out.append(row)
    return out


def _sample_maintained_source(count: int, seed: int) -> list[dict[str, Any]]:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    cfg = get_rlpfn_default_config()
    cfg["prior"]["prior_type"] = "environment_only"
    apply_rlpfn_maintained_path_defaults(cfg)
    prior = EnvironmentPrior(cfg["prior"]["environment"])
    return prior._sample_batch_hypers(int(count))


def _coverage_by_axis(h_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    specs = _axis_specs()
    required_specs = {
        name: spec
        for name, spec in specs.items()
        if spec.get("role") == "required_category_axis"
    }
    coverage = _coverage_rows(_axis_values(_h_only_join(h_rows), required_specs), required_specs)
    return {str(row["axis"]): row for row in coverage}


def _status_pass(row: dict[str, Any] | None) -> bool:
    if not row:
        return False
    return str(row.get("status")) in {"pass_uniform", "covered_skewed"}


def _reduction_rows(source_cov: dict[str, dict[str, Any]], selected_cov: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for axis in sorted(source_cov):
        source = source_cov.get(axis, {})
        selected = selected_cov.get(axis, {})
        source_pass = _status_pass(source)
        selected_pass = _status_pass(selected)
        if source_pass and not selected_pass:
            locus = "selected_fixed_group_construction"
            minimal_fix = (
                "draw/keep fixed h-list by stratified source sampling for this axis; "
                "do not change Exact SCM formula"
            )
            necessity = "selected training set must contain every required bin"
            sufficiency = "source already spans the axis, so preserving bins in fixed group is enough for this axis"
        elif not source_pass:
            locus = "source_sampler"
            minimal_fix = "repair source sampler range/bin law before any selected-list work"
            necessity = "no downstream selector can naturally cover a bin absent from source support"
            sufficiency = "source must cover all bins before fixed-list uniformity can be meaningful"
        else:
            locus = "none"
            minimal_fix = "no repair needed for this axis; protect it in future audits"
            necessity = "already satisfied in source and selected set"
            sufficiency = "keep existing path unchanged"
        rows.append(
            {
                "axis": axis,
                "source_status": source.get("status"),
                "source_coverage_fraction": source.get("coverage_fraction"),
                "source_entropy": source.get("normalized_entropy"),
                "selected_status": selected.get("status"),
                "selected_coverage_fraction": selected.get("coverage_fraction"),
                "selected_entropy": selected.get("normalized_entropy"),
                "repair_locus": locus,
                "minimal_fix_form": minimal_fix,
                "necessary_condition": necessity,
                "sufficient_condition": sufficiency,
            }
        )
    return rows


def _special_reductions() -> list[dict[str, Any]]:
    return [
        {
            "item": "reward_gain_dominance",
            "classification": "not_a_hard_required_axis_yet",
            "reason": (
                "Dominance is a sidecar-derived summary, not a source parameter. "
                "Action-sensitive reward can be captured by action support/gain-ratio "
                "without requiring action to dominate state/noise."
            ),
            "minimal_rule": (
                "Do not repair dominance directly. If a control-reward mode is required, "
                "define it as action reward support plus trainable PPO response, then audit it."
            ),
        },
        {
            "item": "done_count_none",
            "classification": "not_a_hard_required_axis_yet",
            "reason": (
                "Done count is rollout behavior. With continuous terminal target in [0, 80], "
                "exact none is not a uniform source bin unless terminal_reset_enabled itself is "
                "made a categorical source axis."
            ),
            "minimal_rule": (
                "Use terminal_reset_count_target as the hard source axis. Treat done_count as "
                "stability/semantic diagnostic unless terminal-off environments are formally added."
            ),
        },
    ]


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Category Span Source vs Selected Audit",
        "",
        "Read-only reduction audit. It identifies whether a gap belongs to the source sampler or to fixed-list selection.",
        "",
        "## Reduction",
        "",
        "| axis | source | selected | locus | minimal fix |",
        "|---|---|---|---|---|",
    ]
    for row in report["reduction_rows"]:
        lines.append(
            f"| `{row['axis']}` | `{row['source_status']}` | `{row['selected_status']}` | "
            f"`{row['repair_locus']}` | {row['minimal_fix_form']} |"
        )
    lines.extend(["", "## Non-Hard Diagnostics", ""])
    for row in report["special_reductions"]:
        lines.append(f"- `{row['item']}`: {row['minimal_rule']}")
    lines.extend(["", "## Outputs", ""])
    for key, value in report["outputs"].items():
        lines.append(f"- `{key}`: `{value}`")
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    selected_path = Path(args.selected_h_list).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    selected_rows = _read_json(selected_path)
    if not isinstance(selected_rows, list):
        raise TypeError(f"selected h-list must be a JSON list: {selected_path}")
    source_rows = _sample_maintained_source(int(args.source_sample_count), int(args.seed))
    source_cov = _coverage_by_axis(source_rows)
    selected_cov = _coverage_by_axis(selected_rows)
    reduction_rows = _reduction_rows(source_cov, selected_cov)

    json_path = output_dir / "category_span_source_vs_selected_audit.json"
    md_path = output_dir / "category_span_source_vs_selected_audit.md"
    report = {
        "analysis_entry": "phase2_category_span_source_vs_selected_audit",
        "schema": "phase2_category_span_source_vs_selected_audit.v1",
        "contract": {
            "read_only": True,
            "h_only_source_sample": True,
            "does_not_build_env": True,
            "does_not_train": True,
            "does_not_modify_generator": True,
            "test_and_development_separated": True,
        },
        "inputs": {
            "selected_h_list": str(selected_path),
            "source_sample_count": int(args.source_sample_count),
            "seed": int(args.seed),
        },
        "source_coverage": source_cov,
        "selected_coverage": selected_cov,
        "reduction_rows": reduction_rows,
        "special_reductions": _special_reductions(),
        "outputs": {"json": str(json_path), "markdown": str(md_path)},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write(json_path, json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n")
    _write(md_path, _markdown(report))
    print(json.dumps(report["outputs"], sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-h-list", default=str(DEFAULT_SELECTED_H_LIST))
    parser.add_argument("--source-sample-count", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260518)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
