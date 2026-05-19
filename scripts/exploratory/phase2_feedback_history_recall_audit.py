#!/usr/bin/env python
"""Audit historical feedback evidence and whether it supports broad u64 claims.

Exploratory/read-only.  This does not train, sample, or modify any prior/PPO
path.  It separates broad feedback labels from narrower official-trainable
feedback labels, because those have different semantics.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


ART = Path("/home/chen/RLPFN/artifacts")
OUT = ART / "phase2_feedback_history_recall_audit_0511"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "na"
    try:
        f = float(value)
    except Exception:
        return str(value)
    if not math.isfinite(f):
        return "na"
    return f"{f:+.{digits}f}" if f < 0 else f"{f:.{digits}f}"


def _pct(value: Any) -> str:
    if value is None:
        return "na"
    try:
        f = float(value)
    except Exception:
        return str(value)
    return "na" if not math.isfinite(f) else f"{100.0 * f:.1f}%"


def _extract_broad_feedback() -> list[dict[str, Any]]:
    path = ART / "phase2_feedback_band_guarded_u4_smoke_summary_gpu0_0511/locomotion_guarded_smoke_summary.json"
    data = _read_json(path)
    rows = []
    for scope, payload in data.get("scopes", {}).items():
        cand = payload.get("candidate", {})
        rows.append(
            {
                "label": "broad_feedback_band_guarded_u4",
                "scope": scope,
                "n_envs": cand.get("n"),
                "updates": 4,
                "definition": "maximized broad sign_or_phase_sensitive + state_conditioned_energy_injection under other-mode bands",
                "raw_delta_mean": cand.get("raw_reward_sum_delta", {}).get("mean"),
                "raw_delta_positive_fraction": cand.get("raw_reward_sum_delta", {}).get("positive_fraction"),
                "diversity_pass": payload.get("diversity_guard", {}).get("pass"),
                "source": str(path),
            }
        )
    return rows


def _extract_official_feedback() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    single = ART / "phase2_official_trainable_feedback_smoke_summary_0509/official_trainable_feedback_smoke_summary_report.json"
    cross = ART / "phase2_official_trainable_feedback_cross_seed_smoke_summary_0509/official_trainable_feedback_smoke_summary_report.json"
    for path, label in ((single, "official_trainable_feedback_single_seed_u8"), (cross, "official_trainable_feedback_cross_seed_u8")):
        if not path.exists():
            continue
        data = _read_json(path)
        for name, payload in data.get("runs", {}).items():
            last = payload.get("last_row", {})
            contract = payload.get("contract", {})
            semantic = last.get("semantic_probe", {})
            pack = last.get("pack_summary", {})
            # Summary reports differ in exact field placement; prefer top-level
            # tables if present, fall back to pack semantic deltas when absent.
            rows.append(
                {
                    "label": label,
                    "scope": name,
                    "n_envs": contract.get("n_envs"),
                    "updates": contract.get("updates", 8),
                    "definition": "narrow official_actor_advantage_train_normalized_alignment > 0 within transition/primary candidates",
                    "raw_delta_mean": payload.get("raw_delta_mean")
                    or semantic.get("raw_reward_sum_delta", {}).get("mean")
                    or pack.get("stats", {}).get("raw_reward_return_mean_delta"),
                    "raw_delta_positive_fraction": payload.get("raw_delta_positive_fraction"),
                    "hard_pass": payload.get("hard_pass"),
                    "source": str(path),
                }
            )
    # The markdown summaries contain the clearest aggregate numbers.  Add them
    # explicitly so the audit is not dependent on internal report shape.
    rows.extend(
        [
            {
                "label": "official_trainable_feedback_single_seed_u8_table",
                "scope": "q90_official_trainable_tp",
                "n_envs": 6,
                "updates": 8,
                "definition": "narrow official-trainable TP",
                "raw_delta_mean": 6.10,
                "raw_delta_positive_fraction": None,
                "hard_pass": True,
                "source": str(ART / "phase2_official_trainable_feedback_smoke_summary_0509/official_trainable_feedback_smoke_summary.md"),
            },
            {
                "label": "official_trainable_feedback_single_seed_u8_table",
                "scope": "q90_centered_supported_train_lost_tp",
                "n_envs": 3,
                "updates": 8,
                "definition": "centered-supported but official-train-lost contrast",
                "raw_delta_mean": -7.52,
                "raw_delta_positive_fraction": None,
                "hard_pass": True,
                "source": str(ART / "phase2_official_trainable_feedback_smoke_summary_0509/official_trainable_feedback_smoke_summary.md"),
            },
            {
                "label": "official_trainable_feedback_cross_seed_u8_table",
                "scope": "cross_seed_q90_official_trainable_tp",
                "n_envs": 10,
                "updates": 8,
                "definition": "cross-seed narrow official-trainable TP",
                "raw_delta_mean": -7.856627113063581,
                "raw_delta_positive_fraction": None,
                "hard_pass": True,
                "source": str(ART / "phase2_official_trainable_feedback_cross_seed_smoke_summary_0509/official_trainable_feedback_smoke_summary.md"),
            },
            {
                "label": "official_trainable_feedback_cross_seed_u8_table",
                "scope": "cross_seed_q90_centered_supported_train_lost_tp",
                "n_envs": 5,
                "updates": 8,
                "definition": "cross-seed centered-supported train-lost contrast",
                "raw_delta_mean": 9.507566155749373,
                "raw_delta_positive_fraction": None,
                "hard_pass": True,
                "source": str(ART / "phase2_official_trainable_feedback_cross_seed_smoke_summary_0509/official_trainable_feedback_smoke_summary.md"),
            },
        ]
    )
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = _extract_broad_feedback() + _extract_official_feedback()
    report = {
        "analysis_entry": "phase2_feedback_history_recall_audit",
        "contract": {
            "exploratory_only": True,
            "read_only_existing_artifacts": True,
            "does_not_train": True,
            "does_not_modify_exact_scm": True,
            "does_not_modify_fit_model": True,
            "does_not_modify_ppo": True,
        },
        "rows": rows,
        "read": {
            "broad_feedback_memory_supported": False,
            "narrow_official_trainable_feedback_memory_partly_supported": True,
            "why": (
                "The strongest positive table is a small-n narrow official-trainable TP smoke. "
                "The broad feedback u64 candidate has only u4 evidence and full positive fraction 48.4%. "
                "The cross-seed narrow official feedback table is mixed/contradictory on raw delta. "
                "Therefore historical evidence supports refining the feedback grouping, not blindly pushing broad feedback."
            ),
        },
    }
    (OUT / "feedback_history_recall_audit.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# Feedback History Recall Audit",
        "",
        "Exploratory/read-only.  This separates broad feedback labels from narrow official-trainable feedback labels.",
        "",
        "| label | scope | n | updates | raw delta mean | raw delta+ | read |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {label} | {scope} | {n} | {u} | {raw} | {pos} | {definition} |".format(
                label=row["label"],
                scope=row["scope"],
                n=row.get("n_envs") or "na",
                u=row.get("updates") or "na",
                raw=_fmt(row.get("raw_delta_mean")),
                pos=_pct(row.get("raw_delta_positive_fraction")),
                definition=row.get("definition", ""),
            )
        )
    lines.extend(
        [
            "",
            "## Read",
            "",
            "- Your memory is directionally consistent for the **narrow official-trainable feedback** single-seed smoke: n=6, u8, raw delta +6.10.",
            "- It is **not supported as a claim about broad feedback u64**: the broad feedback candidate currently has u4 evidence only, and full raw-delta positive fraction is 48.4%.",
            "- Cross-seed narrow feedback is mixed on raw delta, so the useful conclusion is not “increase feedback broadly”; it is “feedback grouping is underdefined and needs a better profile/readout split.”",
        ]
    )
    md = OUT / "feedback_history_recall_audit.md"
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
