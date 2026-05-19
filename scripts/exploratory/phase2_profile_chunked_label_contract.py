#!/usr/bin/env python
"""Build and merge chunked profile-label artifacts for large candidate pools.

This is a generator-level safety adapter for the unified profile mainline.  It
does not sample environments, train PPO, change the exact SCM, or change
fit_model.  It only decomposes a selected candidate pool into bounded chunks and
defines how probe labels are merged back by original source index.

Why this exists: the profile probes are safe at n=64 but cannot be called
all-at-once for fit_model-scale candidate pools.  Chunking turns the memory risk
into a bounded evaluation-time cost while preserving selector semantics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


ART = Path("/home/chen/RLPFN/artifacts")
REPO = Path(__file__).resolve().parents[2]
DEFAULT_SELECTED_CSV = (
    ART / "phase2_profile_online_candidate_pool_feedback_support_generator_seed9601_0511/source_pool/selected_prior_envs.csv"
)
DEFAULT_OUTPUT_DIR = ART / "phase2_profile_chunked_label_contract_seed9601_0511"
RULES = ("gym_full_obs_action_range", "gym_q90_obs_action_range")
STAGE_OUTPUT_KEYS = {
    "fixed_groups": ("fixed_h_list_json",),
    "action_mode_labels": ("action_mode_json",),
    "locomotion_labels": ("locomotion_csv",),
    "feedback_near_best_labels": ("feedback_near_best_csv",),
    "future_basin_cached_labels": ("future_basin_cached_labels_csv",),
}


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).expanduser().resolve().open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


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


def _safe_label(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")[:160] or "rule"


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def _python() -> str:
    return sys.executable


def _split_csv_floats(text: str) -> list[float]:
    return [float(item) for item in str(text).split(",") if str(item).strip()]


def _split_csv_text(text: str) -> list[str]:
    return [item.strip() for item in str(text).split(",") if item.strip()]


def _feedback_budget(
    *,
    feedback_label_mode: str,
    chunk_size: int,
    n_steps_feedback: int,
    feedback_baseline_modes: str,
    feedback_profiles: str,
    feedback_gains: str,
    feedback_obs_linear_count: int,
    max_feedback_eval_units_per_chunk: int,
) -> dict[str, Any]:
    baselines = _split_csv_text(feedback_baseline_modes)
    profiles = _split_csv_text(feedback_profiles)
    gains = _split_csv_floats(feedback_gains)
    feedback_modes_per_rule = len(profiles) * len(gains) * max(0, int(feedback_obs_linear_count)) * 2
    if str(feedback_label_mode) == "budgeted_guard":
        modes_per_rule = len(baselines) + feedback_modes_per_rule
    elif str(feedback_label_mode) == "full_audit":
        modes_per_rule = feedback_modes_per_rule
    else:
        raise ValueError(f"unsupported feedback_label_mode: {feedback_label_mode}")
    units_per_env = int(modes_per_rule * int(n_steps_feedback))
    units_per_chunk = int(max(0, int(chunk_size)) * units_per_env)
    max_envs_by_budget = (
        int(max_feedback_eval_units_per_chunk) // max(1, units_per_env)
        if units_per_env > 0
        else int(chunk_size)
    )
    return {
        "label_mode": str(feedback_label_mode),
        "baseline_modes": baselines,
        "profiles": profiles,
        "gains": gains,
        "obs_linear_count": int(feedback_obs_linear_count),
        "feedback_modes_per_rule": int(feedback_modes_per_rule),
        "modes_per_rule": int(modes_per_rule),
        "units_per_env": int(units_per_env),
        "units_per_chunk": int(units_per_chunk),
        "max_eval_units_per_chunk": int(max_feedback_eval_units_per_chunk),
        "chunk_size_within_feedback_budget": bool(units_per_chunk <= int(max_feedback_eval_units_per_chunk)),
        "recommended_max_feedback_chunk_envs": max(1, int(max_envs_by_budget)),
    }


def _command_text(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def _fixed_h_path(chunk_dir: Path, rule: str) -> Path:
    return chunk_dir / "fixed_groups" / _safe_label(rule) / "fixed_env_group" / "prior_fixed_h_list.json"


def _build_chunks(
    *,
    selected_csv: str | Path,
    output_dir: str | Path,
    chunk_size: int,
    device: str,
    n_steps_action: int,
    n_steps_locomotion: int,
    action_obs_linear_count: int,
    locomotion_obs_linear_count: int,
    n_steps_feedback: int,
    feedback_label_mode: str,
    feedback_baseline_modes: str,
    feedback_profiles: str,
    feedback_gains: str,
    feedback_obs_linear_count: int,
    max_feedback_eval_units_per_chunk: int,
    future_basin_pendulum_settle_label_csv: str = "",
    future_basin_mountaincar_per_env_csv: str = "",
) -> dict[str, Any]:
    rows = _read_csv(selected_csv)
    by_rule: dict[str, list[dict[str, Any]]] = {rule: [] for rule in RULES}
    for row in rows:
        rule = str(row.get("rule"))
        if rule not in by_rule:
            continue
        out = dict(row)
        out["_source_index"] = len(by_rule[rule])
        by_rule[rule].append(out)

    out_dir = Path(output_dir).expanduser().resolve()
    chunks: list[dict[str, Any]] = []
    for rule in RULES:
        rule_rows = by_rule[rule]
        for chunk_index, start in enumerate(range(0, len(rule_rows), int(chunk_size))):
            chunk_rows = rule_rows[start : start + int(chunk_size)]
            chunk_dir = out_dir / "chunks" / _safe_label(rule) / f"chunk_{chunk_index:04d}"
            chunk_csv = chunk_dir / "selected_prior_envs.csv"
            _write_csv(chunk_csv, chunk_rows)
            fixed_dir = chunk_dir / "fixed_groups"
            fixed_h = _fixed_h_path(chunk_dir, rule)
            action_json = chunk_dir / "action_mode_probe" / "report.json"
            locomotion_dir = chunk_dir / "locomotion_probe"
            feedback_dir = chunk_dir / "feedback_near_best_source_pool_corrected"
            future_basin_dir = chunk_dir / "future_basin_cached_labels"
            future_basin_csv = future_basin_dir / "future_basin_cached_labels.csv"
            if str(feedback_label_mode) == "budgeted_guard":
                feedback_dir = chunk_dir / "feedback_budgeted_support_guard"
                feedback_script = "phase2_feedback_budgeted_support_guard.py"
                feedback_csv = feedback_dir / "feedback_budgeted_support_guard_per_env.csv"
                feedback_command_extra = [
                    "--baseline-modes",
                    str(feedback_baseline_modes),
                    "--feedback-profiles",
                    str(feedback_profiles),
                    "--feedback-gains",
                    str(feedback_gains),
                ]
            elif str(feedback_label_mode) == "full_audit":
                feedback_script = "phase2_feedback_near_best_gain_audit.py"
                feedback_csv = feedback_dir / "feedback_near_best_gain_per_env.csv"
                feedback_command_extra = ["--profiles", str(feedback_profiles), "--gains", str(feedback_gains)]
            else:
                raise ValueError(f"unsupported feedback_label_mode: {feedback_label_mode}")
            commands = {
                "fixed_groups": [
                    _python(),
                    str(REPO / "scripts/exploratory/phase2_selected_prior_envs_to_fixed_groups.py"),
                    "--selected-csv",
                    str(chunk_csv),
                    "--output-dir",
                    str(fixed_dir),
                ],
                "action_mode_labels": [
                    _python(),
                    str(REPO / "scripts/exploratory/phase2_prior_action_mode_coverage_probe.py"),
                    "--output-json",
                    str(action_json),
                    "--fixed-h-list-json",
                    str(fixed_h),
                    "--n-envs",
                    str(len(chunk_rows)),
                    "--n-steps",
                    str(n_steps_action),
                "--device",
                str(device),
                "--prior-milestone",
                "gated_reward_path_balance",
                "--obs-linear-count",
                str(action_obs_linear_count),
            ],
                "locomotion_labels": [
                    _python(),
                    str(REPO / "scripts/exploratory/phase2_locomotion_full_temporal_detector.py"),
                    "--output-dir",
                    str(locomotion_dir),
                    "--fixed-h-list-json",
                    str(fixed_h),
                    "--n-envs",
                    str(len(chunk_rows)),
                "--n-steps",
                str(n_steps_locomotion),
                "--device",
                str(device),
                "--obs-linear-count",
                str(locomotion_obs_linear_count),
            ],
                "feedback_near_best_labels": [
                    _python(),
                    str(REPO / "scripts/exploratory" / feedback_script),
                    "--selected-csv",
                    str(chunk_csv),
                    "--output-dir",
                    str(feedback_dir),
                    "--n-envs",
                    str(len(chunk_rows)),
                    "--n-steps",
                    str(n_steps_feedback),
                    "--device",
                    str(device),
                    "--obs-linear-count",
                    str(feedback_obs_linear_count),
                ]
                + feedback_command_extra,
                "future_basin_cached_labels": [
                    _python(),
                    str(REPO / "scripts/exploratory/phase2_future_basin_cached_label_adapter.py"),
                    "--selected-csv",
                    str(chunk_csv),
                    "--output-dir",
                    str(future_basin_dir),
                    "--pendulum-settle-label-csv",
                    str(future_basin_pendulum_settle_label_csv),
                    "--mountaincar-per-env-csv",
                    str(future_basin_mountaincar_per_env_csv),
                ],
            }
            chunks.append(
                {
                    "rule": rule,
                    "chunk_index": int(chunk_index),
                    "source_start_index": int(start),
                    "source_end_index_exclusive": int(start + len(chunk_rows)),
                    "count": int(len(chunk_rows)),
                    "chunk_dir": str(chunk_dir),
                    "selected_csv": str(chunk_csv),
                    "fixed_h_list_json": str(fixed_h),
                    "action_mode_json": str(action_json),
                    "locomotion_csv": str(locomotion_dir / "locomotion_full_temporal_per_env.csv"),
                    "feedback_near_best_csv": str(feedback_csv),
                    "future_basin_cached_labels_csv": str(future_basin_csv),
                    "commands": commands,
                    "command_text": {key: _command_text(value) for key, value in commands.items()},
                }
            )
    budget = _feedback_budget(
        feedback_label_mode=feedback_label_mode,
        chunk_size=chunk_size,
        n_steps_feedback=n_steps_feedback,
        feedback_baseline_modes=feedback_baseline_modes,
        feedback_profiles=feedback_profiles,
        feedback_gains=feedback_gains,
        feedback_obs_linear_count=feedback_obs_linear_count,
        max_feedback_eval_units_per_chunk=max_feedback_eval_units_per_chunk,
    )
    return {
        "analysis_entry": "phase2_profile_chunked_label_contract",
        "contract": {
            "exploratory_only": True,
            "does_not_sample_environments": True,
            "does_not_train": True,
            "does_not_modify_exact_scm": True,
            "does_not_modify_fit_model": True,
            "does_not_modify_ppo": True,
            "profile_quality_first": True,
            "raw_delta_not_used": True,
            "chunked_memory_bounded_label_path": True,
        },
        "inputs": {
            "selected_csv": str(Path(selected_csv).expanduser().resolve()),
            "output_dir": str(out_dir),
            "chunk_size": int(chunk_size),
            "device": str(device),
            "n_steps_action": int(n_steps_action),
            "n_steps_locomotion": int(n_steps_locomotion),
            "action_obs_linear_count": int(action_obs_linear_count),
            "locomotion_obs_linear_count": int(locomotion_obs_linear_count),
            "n_steps_feedback": int(n_steps_feedback),
            "feedback_label_mode": str(feedback_label_mode),
            "feedback_baseline_modes": str(feedback_baseline_modes),
            "feedback_profiles": str(feedback_profiles),
                "feedback_gains": str(feedback_gains),
                "feedback_obs_linear_count": int(feedback_obs_linear_count),
                "future_basin_pendulum_settle_label_csv": str(future_basin_pendulum_settle_label_csv),
                "future_basin_mountaincar_per_env_csv": str(future_basin_mountaincar_per_env_csv),
            },
        "feedback_eval_budget": budget,
        "source_counts": {rule: len(by_rule[rule]) for rule in RULES},
        "chunk_count": len(chunks),
        "chunks_by_rule": {rule: sum(1 for item in chunks if item["rule"] == rule) for rule in RULES},
        "chunks": chunks,
        "merge_contract": {
            "action_mode": "merge report.json lists by label/rule and offset classification.per_env.env_index by chunk source_start_index",
            "locomotion": "merge locomotion_full_temporal_per_env.csv and offset env_index by chunk source_start_index",
            "feedback_near_best": "merge feedback_near_best_gain_per_env.csv and offset source_index/env_idx by chunk source_start_index",
            "future_basin_cached_labels": "merge future_basin_cached_labels.csv and offset source_index by chunk source_start_index; missing labels remain explicit",
            "selector_input": "merged artifacts must be semantically equivalent to all-at-once labels for phase2_profile_band_selector.py",
        },
    }


def _merge_action_reports(chunks: list[dict[str, Any]], output_json: Path) -> dict[str, Any]:
    lists_by_rule: dict[str, dict[str, Any]] = {}
    merged_inputs = []
    for chunk in chunks:
        path = Path(chunk["action_mode_json"])
        if not path.exists():
            continue
        report = _read_json(path)
        merged_inputs.append(str(path))
        for item in report.get("lists", []):
            label = str(item.get("label", chunk["rule"]))
            target = lists_by_rule.setdefault(
                chunk["rule"],
                {
                    **{key: item.get(key) for key in item if key != "classification"},
                    "label": chunk["rule"],
                    "classification": {"per_env": []},
                },
            )
            for row in item.get("classification", {}).get("per_env", []):
                out = dict(row)
                out["env_index"] = _safe_int(out.get("env_index")) + int(chunk["source_start_index"])
                out["_chunk_index"] = int(chunk["chunk_index"])
                target["classification"]["per_env"].append(out)
            # Keep coverage summaries out of the merged report unless recomputed;
            # phase2_profile_band_selector only consumes per_env labels.
            target["classification"].pop("coverage", None)
    merged = {
        "schema": "phase2_prior_action_mode_coverage_probe.chunked_merge.v1",
        "contract": {
            "chunked_merge_only": True,
            "does_not_recompute_labels": True,
            "offsets_restore_source_indices": True,
        },
        "inputs": merged_inputs,
        "lists": [lists_by_rule[rule] for rule in RULES if rule in lists_by_rule],
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(_json_safe(merged), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"path": str(output_json), "available_chunks": len(merged_inputs)}


def _merge_csv_with_offsets(
    *,
    chunks: list[dict[str, Any]],
    key: str,
    output_csv: Path,
    offset_columns: tuple[str, ...],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    inputs = []
    for chunk in chunks:
        path = Path(str(chunk[key]))
        if not path.exists():
            continue
        inputs.append(str(path))
        offset = int(chunk["source_start_index"])
        chunk_rows = _read_csv(path)
        already_global_columns: set[str] = set()
        for column in offset_columns:
            if column != "source_index" or offset <= 0:
                continue
            values = [
                _safe_int(row[column])
                for row in chunk_rows
                if column in row and str(row[column]).strip() != ""
            ]
            if values and min(values) >= offset and max(values) < int(chunk["source_end_index_exclusive"]):
                already_global_columns.add(column)
        for row in chunk_rows:
            out: dict[str, Any] = dict(row)
            for column in offset_columns:
                if column in out and str(out[column]).strip() != "":
                    if column in already_global_columns:
                        out[column] = _safe_int(out[column])
                    else:
                        out[column] = _safe_int(out[column]) + offset
            out["_chunk_index"] = int(chunk["chunk_index"])
            out["_chunk_source_start_index"] = offset
            rows.append(out)
    _write_csv(output_csv, rows)
    return {"path": str(output_csv), "available_chunks": len(inputs), "row_count": len(rows)}


def _merge_existing(manifest: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    chunks = list(manifest.get("chunks", []))
    merged_dir = output_dir / "merged_labels"
    action = _merge_action_reports(chunks, merged_dir / "action_mode_report.json")
    locomotion = _merge_csv_with_offsets(
        chunks=chunks,
        key="locomotion_csv",
        output_csv=merged_dir / "locomotion_full_temporal_per_env.csv",
        offset_columns=("env_index",),
    )
    feedback = _merge_csv_with_offsets(
        chunks=chunks,
        key="feedback_near_best_csv",
        output_csv=merged_dir / "feedback_near_best_gain_per_env.csv",
        # Real feedback guards copy the global `_source_index` from the chunk
        # CSV, while some tests and older cached labels emit local source
        # indices.  `_merge_csv_with_offsets` detects the global case and only
        # offsets local values.
        offset_columns=("source_index", "env_idx"),
    )
    future_basin = _merge_csv_with_offsets(
        chunks=chunks,
        key="future_basin_cached_labels_csv",
        output_csv=merged_dir / "future_basin_cached_labels.csv",
        offset_columns=("source_index",),
    )
    return {
        "action_mode": action,
        "locomotion": locomotion,
        "feedback_near_best": feedback,
        "future_basin_cached_labels": future_basin,
    }


def _split_stage_names(text: str) -> list[str]:
    stages = [part.strip() for part in str(text).split(",") if part.strip()]
    unknown = [stage for stage in stages if stage not in STAGE_OUTPUT_KEYS]
    if unknown:
        raise ValueError(f"unknown execute stage(s): {unknown}; expected one of {sorted(STAGE_OUTPUT_KEYS)}")
    return stages


def _stage_ready(chunk: dict[str, Any], stage: str) -> bool:
    return all(Path(str(chunk[key])).expanduser().exists() for key in STAGE_OUTPUT_KEYS[stage])


def _execute_missing_chunks(
    *,
    report: dict[str, Any],
    output_dir: Path,
    execute_stages: str,
    max_execute_chunks: int,
    execute_rule: str,
    allow_expensive_feedback: bool,
) -> dict[str, Any]:
    stages = _split_stage_names(execute_stages)
    chunks = list(report.get("chunks", []))
    rule_filter = str(execute_rule or "").strip()
    if rule_filter:
        if rule_filter not in RULES:
            raise ValueError(f"execute_rule must be one of {RULES}, got {execute_rule!r}")
        chunks = [chunk for chunk in chunks if chunk["rule"] == rule_filter]
    if int(max_execute_chunks) > 0:
        chunks = chunks[: int(max_execute_chunks)]

    feedback_budget = report.get("feedback_eval_budget", {}) or {}
    if (
        "feedback_near_best_labels" in stages
        and not bool(allow_expensive_feedback)
        and not bool(feedback_budget.get("chunk_size_within_feedback_budget", False))
    ):
        raise RuntimeError(
            "Refusing to execute feedback labels above the per-chunk budget: "
            f"units_per_chunk={feedback_budget.get('units_per_chunk')} "
            f"> max_eval_units_per_chunk={feedback_budget.get('max_eval_units_per_chunk')}. "
            "Reduce --chunk-size or pass --allow-expensive-feedback for an explicit research run."
        )

    executed: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for chunk in chunks:
        for stage in stages:
            if _stage_ready(chunk, stage):
                skipped.append(
                    {
                        "rule": chunk["rule"],
                        "chunk_index": int(chunk["chunk_index"]),
                        "stage": stage,
                        "reason": "outputs_exist",
                    }
                )
                continue
            command = list(chunk["commands"][stage])
            subprocess.run(command, cwd=str(REPO), check=True)
            executed.append(
                {
                    "rule": chunk["rule"],
                    "chunk_index": int(chunk["chunk_index"]),
                    "stage": stage,
                    "command_text": _command_text(command),
                }
            )

    out = {
        "enabled": True,
        "output_dir": str(output_dir),
        "execute_stages": stages,
        "execute_rule": rule_filter or None,
        "max_execute_chunks": int(max_execute_chunks),
        "selected_chunk_count": len(chunks),
        "executed_count": len(executed),
        "skipped_count": len(skipped),
        "executed": executed,
        "skipped": skipped,
    }
    (output_dir / "chunked_label_execution.json").write_text(
        json.dumps(_json_safe(out), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return out


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Profile Chunked Label Contract",
        "",
        "This artifact makes profile-label evaluation memory-bounded for large candidate pools.",
        "",
        "## Read",
        "",
        f"- chunk size: `{report['inputs']['chunk_size']}`",
        f"- chunk count: `{report['chunk_count']}`",
        f"- source counts: `{report['source_counts']}`",
        "- selector semantics: unchanged; labels are only offset and merged back to original source indices",
        "",
        "## Feedback Eval Budget",
        "",
        f"- label mode: `{report['feedback_eval_budget']['label_mode']}`",
        f"- baselines: `{report['feedback_eval_budget']['baseline_modes']}`",
        f"- feedback modes per rule: `{report['feedback_eval_budget']['feedback_modes_per_rule']}`",
        f"- modes per rule: `{report['feedback_eval_budget']['modes_per_rule']}`",
        f"- units per env: `{report['feedback_eval_budget']['units_per_env']}`",
        f"- units per chunk: `{report['feedback_eval_budget']['units_per_chunk']}`",
        f"- max units per chunk: `{report['feedback_eval_budget']['max_eval_units_per_chunk']}`",
        f"- chunk size within feedback budget: `{report['feedback_eval_budget']['chunk_size_within_feedback_budget']}`",
        f"- recommended max feedback chunk envs: `{report['feedback_eval_budget']['recommended_max_feedback_chunk_envs']}`",
        "",
        "## Chunks",
        "",
        "| rule | chunk | source range | count |",
        "|---|---:|---|---:|",
    ]
    for chunk in report["chunks"][:32]:
        lines.append(
            f"| `{chunk['rule']}` | {chunk['chunk_index']} | "
            f"{chunk['source_start_index']}:{chunk['source_end_index_exclusive']} | {chunk['count']} |"
        )
    if len(report["chunks"]) > 32:
        lines.append(f"| ... | ... | ... | {len(report['chunks']) - 32} more chunks |")
    lines.extend(["", "## Merge Contract", ""])
    for key, value in report["merge_contract"].items():
        lines.append(f"- `{key}`: {value}")
    if "merged_outputs" in report:
        lines.extend(["", "## Merged Outputs", ""])
        for key, value in report["merged_outputs"].items():
            lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Files", ""])
    for key, value in report.get("outputs", {}).items():
        lines.append(f"- `{key}`: `{value}`")
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report = _build_chunks(
        selected_csv=args.selected_csv,
        output_dir=out_dir,
        chunk_size=args.chunk_size,
        device=args.device,
        n_steps_action=args.n_steps_action,
        n_steps_locomotion=args.n_steps_locomotion,
        action_obs_linear_count=int(getattr(args, "action_obs_linear_count", 4)),
        locomotion_obs_linear_count=int(getattr(args, "locomotion_obs_linear_count", 4)),
        n_steps_feedback=args.n_steps_feedback,
        feedback_label_mode=args.feedback_label_mode,
        feedback_baseline_modes=args.feedback_baseline_modes,
        feedback_profiles=args.feedback_profiles,
        feedback_gains=args.feedback_gains,
        feedback_obs_linear_count=args.feedback_obs_linear_count,
        max_feedback_eval_units_per_chunk=args.max_feedback_eval_units_per_chunk,
        future_basin_pendulum_settle_label_csv=getattr(args, "future_basin_pendulum_settle_label_csv", ""),
        future_basin_mountaincar_per_env_csv=getattr(args, "future_basin_mountaincar_per_env_csv", ""),
    )
    if bool(getattr(args, "execute_missing", False)):
        report["execution"] = _execute_missing_chunks(
            report=report,
            output_dir=out_dir,
            execute_stages=str(
                getattr(
                    args,
                    "execute_stages",
                    "fixed_groups,action_mode_labels,locomotion_labels,feedback_near_best_labels",
                )
            ),
            max_execute_chunks=int(getattr(args, "max_execute_chunks", 0)),
            execute_rule=str(getattr(args, "execute_rule", "")),
            allow_expensive_feedback=bool(getattr(args, "allow_expensive_feedback", False)),
        )
    if bool(args.merge_existing):
        report["merged_outputs"] = _merge_existing(report, out_dir)
    json_path = out_dir / "profile_chunked_label_contract.json"
    md_path = out_dir / "profile_chunked_label_contract.md"
    report["outputs"] = {"json": str(json_path), "markdown": str(md_path)}
    json_path.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report["outputs"], sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-csv", default=str(DEFAULT_SELECTED_CSV))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-steps-action", type=int, default=256)
    parser.add_argument("--n-steps-locomotion", type=int, default=256)
    parser.add_argument("--action-obs-linear-count", type=int, default=4)
    parser.add_argument("--locomotion-obs-linear-count", type=int, default=4)
    parser.add_argument("--n-steps-feedback", type=int, default=64)
    parser.add_argument("--feedback-label-mode", choices=("budgeted_guard", "full_audit"), default="budgeted_guard")
    parser.add_argument("--feedback-baseline-modes", default="zero,random,neg_random,half_random")
    parser.add_argument("--feedback-profiles", default="decay")
    parser.add_argument("--feedback-gains", default="1.0")
    parser.add_argument("--feedback-obs-linear-count", type=int, default=2)
    parser.add_argument("--max-feedback-eval-units-per-chunk", type=int, default=500_000)
    parser.add_argument("--future-basin-pendulum-settle-label-csv", default="")
    parser.add_argument("--future-basin-mountaincar-per-env-csv", default="")
    parser.add_argument("--execute-missing", action="store_true")
    parser.add_argument(
        "--execute-stages",
        default="fixed_groups,action_mode_labels,locomotion_labels,feedback_near_best_labels",
        help="Comma-separated chunk stages to execute when --execute-missing is set.",
    )
    parser.add_argument("--max-execute-chunks", type=int, default=0)
    parser.add_argument("--execute-rule", default="")
    parser.add_argument("--allow-expensive-feedback", action="store_true")
    parser.add_argument("--merge-existing", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
