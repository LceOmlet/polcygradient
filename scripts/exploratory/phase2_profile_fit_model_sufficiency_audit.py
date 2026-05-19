#!/usr/bin/env python
"""Audit profile-to-fit_model sufficiency from existing artifacts.

Read-only consolidation.  This script does not run PPO, does not train, and
does not import torch.  It checks whether the profile mainline has the
properties needed to treat the current path as an engineering milestone:

1. source-pool generation is still milestone2-feasible,
2. profile labels are bounded and stage-ready,
3. chunked labels preserve selector semantics,
4. selected profile envs can enter fit_model startup,
5. pack-to-fit_model numeric parity passes,
6. fit_model-scale computation is memory-feasible only through chunking.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


ART = Path("/home/chen/RLPFN/artifacts")
RULES = ("gym_full_obs_action_range", "gym_q90_obs_action_range")

DEFAULT_CURRENT_SOURCE_REPORT = (
    ART
    / "phase2_profile_online_candidate_pool_feedback_support_generator_seed9602_0511"
    / "source_pool/reward_group_balance_probe_report.json"
)
DEFAULT_MILESTONE2_SOURCE_REPORTS = (
    ART / "phase2_exploratory_reward_group_balance_gated_n64_0506/reward_group_balance_probe_report.json",
    ART / "phase2_exploratory_reward_group_balance_gated_n64_seed6261_0506/reward_group_balance_probe_report.json",
)
DEFAULT_PREFLIGHT_JSON = (
    ART
    / "phase2_profile_budgeted_feedback_guard_preflight_execute_seed9602_0511"
    / "current_seed_evidence/profile_online_selector_preflight.json"
)
DEFAULT_CHUNK_CONTRACT_JSON = (
    ART / "phase2_profile_chunked_label_pool64_seed9602_0511/profile_chunked_label_contract.json"
)
DEFAULT_ORIGINAL_SELECTOR_JSON = (
    ART
    / "phase2_profile_online_candidate_pool_feedback_support_generator_seed9602_0511"
    / "profile_selector/profile_band_selector_report.json"
)
DEFAULT_ORIGINAL_SELECTED_CSV = (
    ART
    / "phase2_profile_online_candidate_pool_feedback_support_generator_seed9602_0511"
    / "profile_selector/selected_profile_band_prior_envs.csv"
)
DEFAULT_CHUNKED_SELECTOR_JSON = (
    ART
    / "phase2_profile_chunked_label_pool64_seed9602_0511"
    / "profile_selector_from_merged/profile_band_selector_report.json"
)
DEFAULT_CHUNKED_SELECTED_CSV = (
    ART
    / "phase2_profile_chunked_label_pool64_seed9602_0511"
    / "profile_selector_from_merged/selected_profile_band_prior_envs.csv"
)
DEFAULT_PARITY_JSON = (
    ART
    / "phase2_profile_chunked_label_pool64_seed9602_pack_fit_parity_0511"
    / "profile_v2e_pack_fit_model_parity_report.json"
)
DEFAULT_FIT_MODEL_LOG = ART / "fit_model_fixedlist_startup_smoke_0511/run2.log"
DEFAULT_OUTPUT_DIR = ART / "phase2_profile_fit_model_sufficiency_audit_0511"


WEIGHTS = {
    "source_pool_milestone2_feasible": 0.14,
    "bounded_profile_stages_ready": 0.10,
    "chunked_label_path_ready": 0.14,
    "selector_profile_guard_pass": 0.12,
    "chunked_selector_semantic_parity": 0.10,
    "fit_model_fixed_csv_startup_pass": 0.10,
    "pack_to_fit_model_numeric_parity": 0.15,
    "fit_model_scale_chunk_projection_safe": 0.10,
    "explicit_no_silent_fallback": 0.05,
}


def _read_json_optional(path: str | Path | None) -> Any:
    if path is None or str(path).strip() == "":
        return None
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        return None
    return json.loads(resolved.read_text(encoding="utf-8"))


def _read_text_optional(path: str | Path | None) -> str:
    if path is None or str(path).strip() == "":
        return ""
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        return ""
    return resolved.read_text(encoding="utf-8", errors="replace")


def _read_csv_optional(path: str | Path | None) -> list[dict[str, str]]:
    if path is None or str(path).strip() == "":
        return []
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        return []
    with resolved.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _sha256_optional(path: str | Path | None) -> str | None:
    if path is None or str(path).strip() == "":
        return None
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        return None
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _rule_counts_ok(counts: dict[str, Any], minimum: int) -> bool:
    return all(int(counts.get(rule, 0) or 0) >= int(minimum) for rule in RULES)


def _source_pool_read(current_path: str | Path, milestone2_paths: list[str | Path]) -> dict[str, Any]:
    current = _read_json_optional(current_path) or {}
    current_sampling = current.get("sampling", {}) or {}
    current_counts = current.get("selected_count_by_rule", {}) or {}
    refs = []
    for path in milestone2_paths:
        payload = _read_json_optional(path) or {}
        refs.append(
            {
                "path": str(Path(path).expanduser().resolve()),
                "available": bool(payload),
                "sampling": payload.get("sampling", {}) or {},
                "selected_count_by_rule": payload.get("selected_count_by_rule", {}) or {},
            }
        )

    comparable_refs = [
        ref
        for ref in refs
        if ref["available"]
        and int((ref["sampling"] or {}).get("target_per_rule", 0) or 0) == int(current_sampling.get("target_per_rule", 0) or 0)
    ]
    ref_raw = [int((ref["sampling"] or {}).get("raw_seen", 0) or 0) for ref in comparable_refs]
    ref_eval = [int((ref["sampling"] or {}).get("env_evaluated", 0) or 0) for ref in comparable_refs]
    raw_seen = int(current_sampling.get("raw_seen", 0) or 0)
    env_evaluated = int(current_sampling.get("env_evaluated", 0) or 0)
    target = int(current_sampling.get("target_per_rule", 0) or 0)
    selected_ok = _rule_counts_ok(current_counts, target)
    same_order_as_milestone2 = bool(
        comparable_refs
        and raw_seen <= max(ref_raw or [0])
        and env_evaluated <= max(ref_eval or [0])
    )
    hard_pass = bool(current and selected_ok and same_order_as_milestone2)
    return {
        "path": str(Path(current_path).expanduser().resolve()),
        "available": bool(current),
        "target_per_rule": target,
        "selected_count_by_rule": current_counts,
        "raw_seen": raw_seen,
        "env_evaluated": env_evaluated,
        "milestone2_reference": refs,
        "same_order_as_milestone2": same_order_as_milestone2,
        "hard_pass": hard_pass,
        "read": (
            "current source pool recovered milestone2 n64 compute/yield scale"
            if hard_pass
            else "source pool is missing, underfilled, or slower than the milestone2 reference set"
        ),
    }


def _preflight_read(path: str | Path) -> dict[str, Any]:
    payload = _read_json_optional(path) or {}
    evidence = payload.get("existing_evidence", {}) or {}
    plan = payload.get("bounded_generator_plan", {}) or {}
    stages = plan.get("stage_plan", []) or []
    stage_ready = {str(stage.get("name")): bool(stage.get("ready")) for stage in stages}
    budget = plan.get("feedback_eval_budget", {}) or {}
    projection = payload.get("fit_model_2048_scale_projection", {}) or {}
    contract = payload.get("contract", {}) or {}
    hard_pass = bool(
        payload
        and evidence.get("broad_labels_ready") is True
        and evidence.get("feedback_support_ready") is True
        and (evidence.get("selector", {}) or {}).get("hard_pass") is True
        and stage_ready
        and all(stage_ready.values())
        and not bool(budget.get("execution_blocked_by_default"))
    )
    return {
        "path": str(Path(path).expanduser().resolve()),
        "available": bool(payload),
        "stage_ready": stage_ready,
        "feedback_budget": budget,
        "fit_model_2048_scale_projection": projection,
        "contract": contract,
        "bounded_profile_stages_ready": hard_pass,
        "explicit_no_silent_fallback": bool(contract.get("explicit_failure_no_unprofiled_fallback")),
        "chunk_projection_safe": bool((projection.get("chunked_probe", {}) or {}).get("memory_safe")),
        "naive_all_at_once_safe": bool((projection.get("naive_all_at_once", {}) or {}).get("memory_safe")),
    }


def _selector_read(path: str | Path) -> dict[str, Any]:
    payload = _read_json_optional(path) or {}
    rules = payload.get("rules", {}) or {}
    per_rule: dict[str, Any] = {}
    for rule in RULES:
        item = rules.get(rule, {}) or {}
        selected = item.get("selected_summary", {}) or {}
        per_rule[rule] = {
            "guard_pass": bool(item.get("guard_pass")),
            "profile_guard_pass": bool(item.get("profile_guard_pass")),
            "n": int(selected.get("n", 0) or 0),
            "active_cells": int(selected.get("joint_profile_active_cells", 0) or 0),
            "dominant_rate": selected.get("joint_profile_dominant_rate"),
            "entropy_norm": selected.get("joint_profile_entropy_norm"),
            "locomotion_rate": selected.get("locomotion_witness_rate"),
            "low_energy_rate": selected.get("low_energy_not_ctrl_only_rate"),
            "sign_phase_rate": selected.get("sign_or_phase_sensitive_rate"),
            "state_conditioned_rate": selected.get("state_conditioned_energy_injection_rate"),
            "feedback_rate": selected.get("strict_feedback_rate"),
        }
    hard_pass = bool(payload and all(per_rule[rule]["guard_pass"] for rule in RULES))
    return {
        "path": str(Path(path).expanduser().resolve()),
        "available": bool(payload),
        "profile_version_id": (payload.get("config", {}) or {}).get("profile_version_id"),
        "per_rule": per_rule,
        "hard_pass": hard_pass,
    }


def _chunk_read(
    *,
    chunk_contract_json: str | Path,
    original_selected_csv: str | Path,
    chunked_selected_csv: str | Path,
) -> dict[str, Any]:
    payload = _read_json_optional(chunk_contract_json) or {}
    execution = payload.get("execution", {}) or {}
    merged = payload.get("merged_outputs", {}) or {}
    source_counts = payload.get("source_counts", {}) or {}
    feedback_budget = payload.get("feedback_eval_budget", {}) or {}
    contract = payload.get("contract", {}) or {}
    original_sha = _sha256_optional(original_selected_csv)
    chunked_sha = _sha256_optional(chunked_selected_csv)
    csv_equal = bool(original_sha and chunked_sha and original_sha == chunked_sha)
    merged_rows_ok = all(int((merged.get(name, {}) or {}).get("row_count", 0) or 0) >= sum(int(source_counts.get(rule, 0) or 0) for rule in RULES) for name in ("locomotion", "feedback_near_best"))
    hard_pass = bool(
        payload
        and contract.get("chunked_memory_bounded_label_path") is True
        and int(execution.get("executed_count", 0) or 0) > 0
        and not execution.get("skipped")
        and bool(feedback_budget.get("chunk_size_within_feedback_budget"))
        and _rule_counts_ok(source_counts, 64)
        and merged_rows_ok
    )
    return {
        "path": str(Path(chunk_contract_json).expanduser().resolve()),
        "available": bool(payload),
        "chunk_count": payload.get("chunk_count"),
        "execution": {
            "executed_count": execution.get("executed_count"),
            "skipped_count": execution.get("skipped_count"),
            "selected_chunk_count": execution.get("selected_chunk_count"),
        },
        "feedback_eval_budget": feedback_budget,
        "source_counts": source_counts,
        "merged_outputs": merged,
        "chunked_label_path_ready": hard_pass,
        "selector_csv_sha256": {
            "original": original_sha,
            "chunked": chunked_sha,
            "equal": csv_equal,
        },
        "chunked_selector_semantic_parity": csv_equal,
    }


def _parity_read(path: str | Path, *, tolerance: float = 0.0) -> dict[str, Any]:
    payload = _read_json_optional(path) or {}
    cases = payload.get("cases", []) or []
    case_reads = []
    for case in cases:
        update = case.get("update_parity", {}) or {}
        grad = (update.get("gradient_diff", {}) or {}).get("grad_delta_max_abs")
        post = (
            (update.get("one_train_update_compare", {}) or {})
            .get("post_update_param_diff", {})
            or {}
        ).get("policy_param_max_abs_diff")
        case_reads.append(
            {
                "case": case.get("case"),
                "rule": case.get("rule"),
                "n_envs": case.get("n_envs"),
                "n_steps": case.get("n_steps"),
                "hard_pass": bool(case.get("hard_pass")),
                "collection_hard_pass": bool((case.get("collection_parity", {}) or {}).get("hard_pass")),
                "update_hard_pass": bool(update.get("hard_pass")),
                "grad_delta_max_abs": grad,
                "post_param_delta_max_abs": post,
            }
        )
    hard_pass = bool(
        payload.get("hard_pass")
        and cases
        and all(item["hard_pass"] for item in case_reads)
        and all(_safe_float(item["grad_delta_max_abs"]) <= tolerance for item in case_reads)
        and all(_safe_float(item["post_param_delta_max_abs"]) <= tolerance for item in case_reads)
    )
    return {
        "path": str(Path(path).expanduser().resolve()),
        "available": bool(payload),
        "hard_pass": hard_pass,
        "case_count": len(cases),
        "cases": case_reads,
    }


def _fit_model_log_read(path: str | Path) -> dict[str, Any]:
    text = _read_text_optional(path)
    bad_patterns = ("Traceback", "CUDA out of memory", "RuntimeError:", "Exception")
    rollout_match = re.search(r"rollout_s\s+([+-]?[0-9.]+e?[+-]?[0-9]*)", text)
    update_match = re.search(r"update_s\s+([+-]?[0-9.]+e?[+-]?[0-9]*)", text)
    hard_pass = bool(
        text
        and "Using trusted canonical PPO pack runner" in text
        and "Starting training of model" in text
        and "end of epoch" in text
        and not any(pattern in text for pattern in bad_patterns)
    )
    return {
        "path": str(Path(path).expanduser().resolve()),
        "available": bool(text),
        "trusted_pack_runner": "Using trusted canonical PPO pack runner" in text,
        "fit_model_started": "Starting training of model" in text,
        "epoch_finished": "end of epoch" in text,
        "has_error_pattern": any(pattern in text for pattern in bad_patterns),
        "rollout_s": _safe_float(rollout_match.group(1), default=None) if rollout_match else None,
        "update_s": _safe_float(update_match.group(1), default=None) if update_match else None,
        "hard_pass": hard_pass,
    }


def _score(checks: dict[str, bool]) -> dict[str, Any]:
    components = []
    score = 0.0
    for key, weight in WEIGHTS.items():
        passed = bool(checks.get(key))
        score += weight if passed else 0.0
        components.append({"id": key, "weight": weight, "passed": passed, "credit": weight if passed else 0.0})
    return {
        "score": score,
        "remaining": max(0.0, 1.0 - score),
        "components": components,
    }


def build_report(
    *,
    current_source_report: str | Path,
    milestone2_source_reports: list[str | Path],
    preflight_json: str | Path,
    chunk_contract_json: str | Path,
    original_selector_json: str | Path,
    original_selected_csv: str | Path,
    chunked_selector_json: str | Path,
    chunked_selected_csv: str | Path,
    parity_json: str | Path,
    fit_model_log: str | Path,
) -> dict[str, Any]:
    source = _source_pool_read(current_source_report, milestone2_source_reports)
    preflight = _preflight_read(preflight_json)
    original_selector = _selector_read(original_selector_json)
    chunked_selector = _selector_read(chunked_selector_json)
    chunk = _chunk_read(
        chunk_contract_json=chunk_contract_json,
        original_selected_csv=original_selected_csv,
        chunked_selected_csv=chunked_selected_csv,
    )
    parity = _parity_read(parity_json)
    fit_model = _fit_model_log_read(fit_model_log)

    checks = {
        "source_pool_milestone2_feasible": bool(source["hard_pass"]),
        "bounded_profile_stages_ready": bool(preflight["bounded_profile_stages_ready"]),
        "chunked_label_path_ready": bool(chunk["chunked_label_path_ready"]),
        "selector_profile_guard_pass": bool(original_selector["hard_pass"] and chunked_selector["hard_pass"]),
        "chunked_selector_semantic_parity": bool(chunk["chunked_selector_semantic_parity"]),
        "fit_model_fixed_csv_startup_pass": bool(fit_model["hard_pass"]),
        "pack_to_fit_model_numeric_parity": bool(parity["hard_pass"]),
        "fit_model_scale_chunk_projection_safe": bool(
            preflight["chunk_projection_safe"] and not preflight["naive_all_at_once_safe"]
        ),
        "explicit_no_silent_fallback": bool(preflight["explicit_no_silent_fallback"]),
    }
    score = _score(checks)
    critical = (
        "source_pool_milestone2_feasible",
        "bounded_profile_stages_ready",
        "chunked_label_path_ready",
        "selector_profile_guard_pass",
        "chunked_selector_semantic_parity",
        "fit_model_fixed_csv_startup_pass",
        "pack_to_fit_model_numeric_parity",
        "fit_model_scale_chunk_projection_safe",
    )
    milestone_candidate = bool(score["score"] >= 0.92 and all(checks[item] for item in critical))
    gaps = []
    if not milestone_candidate:
        gaps.extend([key for key in critical if not checks[key]])
    gaps.extend(
        [
            "2048-scale profile labels are projected/chunked but not yet fully executed end-to-end",
            "cache-by-frozen_h fingerprint is a contract, not yet a fit_model startup cache implementation",
            "Gym-mode definition alignment remains partial: MountainCar future-basin and terminal/survival axes are not quota-ready",
            "numeric pack-to-fit_model parity is a tiny full/q90 smoke, not a long-train guarantee",
        ]
    )
    decision = {
        "seal_as_new_engineering_milestone_candidate": milestone_candidate,
        "recommended_branch": (
            "freeze the profile-fit_model integration contract as a new engineering milestone candidate; "
            "treat Gym-mode definition alignment as the next milestone track"
            if milestone_candidate
            else "continue profile-fit_model parity/feasibility work before returning to Gym-mode definition changes"
        ),
        "why": (
            "pipeline parity and bounded computation are hard-passing; remaining work is scale execution and Gym semantic validation"
            if milestone_candidate
            else "one or more critical pipeline properties are still missing"
        ),
    }
    return {
        "analysis_entry": "phase2_profile_fit_model_sufficiency_audit",
        "contract": {
            "read_only_existing_artifacts": True,
            "does_not_import_torch": True,
            "does_not_run_ppo": True,
            "does_not_train": True,
            "does_not_modify_exact_scm": True,
            "does_not_modify_fit_model": True,
            "does_not_modify_ppo": True,
        },
        "inputs": {
            "current_source_report": str(Path(current_source_report).expanduser().resolve()),
            "milestone2_source_reports": [str(Path(path).expanduser().resolve()) for path in milestone2_source_reports],
            "preflight_json": str(Path(preflight_json).expanduser().resolve()),
            "chunk_contract_json": str(Path(chunk_contract_json).expanduser().resolve()),
            "original_selector_json": str(Path(original_selector_json).expanduser().resolve()),
            "original_selected_csv": str(Path(original_selected_csv).expanduser().resolve()),
            "chunked_selector_json": str(Path(chunked_selector_json).expanduser().resolve()),
            "chunked_selected_csv": str(Path(chunked_selected_csv).expanduser().resolve()),
            "parity_json": str(Path(parity_json).expanduser().resolve()),
            "fit_model_log": str(Path(fit_model_log).expanduser().resolve()),
        },
        "checks": checks,
        "score": score,
        "source_pool": source,
        "preflight": preflight,
        "chunked_label_path": chunk,
        "selector": {
            "original": original_selector,
            "chunked_from_merged": chunked_selector,
        },
        "fit_model_startup": fit_model,
        "pack_to_fit_model_parity": parity,
        "decision": decision,
        "remaining_work": gaps,
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Profile-to-fit_model Sufficiency Audit",
        "",
        "Read-only audit from existing artifacts.",
        "",
        f"- pipeline score: `{report['score']['score']:.3f}`",
        f"- remaining pipeline gap: `{report['score']['remaining']:.3f}`",
        f"- new engineering milestone candidate: `{report['decision']['seal_as_new_engineering_milestone_candidate']}`",
        f"- decision: {report['decision']['recommended_branch']}",
        "",
        "## Checks",
        "",
        "| check | weight | pass |",
        "|---|---:|---|",
    ]
    for item in report["score"]["components"]:
        lines.append(f"| `{item['id']}` | {item['weight']:.2f} | `{item['passed']}` |")
    lines.extend(
        [
            "",
            "## Source Pool",
            "",
            f"- hard pass: `{report['source_pool']['hard_pass']}`",
            f"- current raw/env: `{report['source_pool']['raw_seen']}` / `{report['source_pool']['env_evaluated']}`",
            f"- selected counts: `{json.dumps(report['source_pool']['selected_count_by_rule'], sort_keys=True)}`",
            f"- read: {report['source_pool']['read']}",
            "",
            "## Profile/Selector",
            "",
            f"- preflight stages ready: `{report['preflight']['bounded_profile_stages_ready']}`",
            f"- no silent fallback: `{report['preflight']['explicit_no_silent_fallback']}`",
            f"- original selector hard pass: `{report['selector']['original']['hard_pass']}`",
            f"- chunked selector hard pass: `{report['selector']['chunked_from_merged']['hard_pass']}`",
            f"- selected CSV semantic parity: `{report['chunked_label_path']['selector_csv_sha256']['equal']}`",
            "",
            "## Fit Model",
            "",
            f"- fixed CSV startup smoke: `{report['fit_model_startup']['hard_pass']}`",
            f"- rollout/update seconds: `{report['fit_model_startup']['rollout_s']}` / `{report['fit_model_startup']['update_s']}`",
            f"- pack-to-fit_model parity: `{report['pack_to_fit_model_parity']['hard_pass']}`",
            f"- parity cases: `{report['pack_to_fit_model_parity']['case_count']}`",
            "",
            "## Remaining Work",
            "",
        ]
    )
    for item in report["remaining_work"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Files", ""])
    for key, value in report.get("outputs", {}).items():
        lines.append(f"- `{key}`: `{value}`")
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(
        current_source_report=args.current_source_report,
        milestone2_source_reports=args.milestone2_source_report,
        preflight_json=args.preflight_json,
        chunk_contract_json=args.chunk_contract_json,
        original_selector_json=args.original_selector_json,
        original_selected_csv=args.original_selected_csv,
        chunked_selector_json=args.chunked_selector_json,
        chunked_selected_csv=args.chunked_selected_csv,
        parity_json=args.parity_json,
        fit_model_log=args.fit_model_log,
    )
    json_path = out_dir / "profile_fit_model_sufficiency_audit.json"
    md_path = out_dir / "profile_fit_model_sufficiency_audit.md"
    report["outputs"] = {"json": str(json_path), "markdown": str(md_path)}
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report["outputs"], sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-source-report", default=str(DEFAULT_CURRENT_SOURCE_REPORT))
    parser.add_argument(
        "--milestone2-source-report",
        action="append",
        default=[str(path) for path in DEFAULT_MILESTONE2_SOURCE_REPORTS],
    )
    parser.add_argument("--preflight-json", default=str(DEFAULT_PREFLIGHT_JSON))
    parser.add_argument("--chunk-contract-json", default=str(DEFAULT_CHUNK_CONTRACT_JSON))
    parser.add_argument("--original-selector-json", default=str(DEFAULT_ORIGINAL_SELECTOR_JSON))
    parser.add_argument("--original-selected-csv", default=str(DEFAULT_ORIGINAL_SELECTED_CSV))
    parser.add_argument("--chunked-selector-json", default=str(DEFAULT_CHUNKED_SELECTOR_JSON))
    parser.add_argument("--chunked-selected-csv", default=str(DEFAULT_CHUNKED_SELECTED_CSV))
    parser.add_argument("--parity-json", default=str(DEFAULT_PARITY_JSON))
    parser.add_argument("--fit-model-log", default=str(DEFAULT_FIT_MODEL_LOG))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
