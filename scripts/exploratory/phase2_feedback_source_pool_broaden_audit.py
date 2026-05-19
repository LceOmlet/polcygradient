#!/usr/bin/env python
"""Audit feedback strict source-pool definitions.

Exploratory/read-only.  This answers whether the strict feedback collapse is a
selection problem, a too-narrow label problem, or a deeper source-pool/gain
problem.  It does not sample environments, train PPO, or modify exact SCM /
fit_model / PPO paths.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np


ART = Path("/home/chen/RLPFN/artifacts")
DEFAULT_SIGNATURE_REPORTS = [
    ART
    / "phase2_prior_pendulum_like_signature_probe_0509"
    / "gated_full_q90_n64_s128_obs4_scales3_timeprofile.json",
    ART
    / "phase2_prior_pendulum_like_signature_probe_seed6261_0509"
    / "gated_full_q90_seed6261_n64_s128_obs4_scales3_timeprofile.json",
    ART
    / "phase2_prior_pendulum_like_signature_probe_seed6262_0510"
    / "gated_full_q90_seed6262_n128_s128_obs4_scales3_timeprofile.json",
]
DEFAULT_OUTPUT_DIR = ART / "phase2_feedback_source_pool_broaden_audit_0511"


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    return value


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return math.nan
    return out if math.isfinite(out) else math.nan


def _is_true(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _source_tag(label: str, report_path: str | Path) -> str:
    text = f"{label} {report_path}"
    seed = "seed6260"
    for token in ("seed6262", "seed6261", "seed6260"):
        if token in text:
            seed = token
            break
    scope = "q90" if "q90" in text else "full"
    return f"{seed}_{scope}"


def _profile(mode: Any) -> str:
    text = str(mode or "")
    if text.startswith("early_") or "_then_zero" in text:
        return "early_then_zero"
    if text.startswith("decay_"):
        return "decay"
    if text.startswith("obs_linear"):
        return "constant"
    return "other"


def _gain(mode: Any) -> str:
    text = str(mode or "")
    if "x50" in text:
        return "0.5"
    if "x100" in text:
        return "1.0"
    if "x200" in text:
        return "2.0"
    return "other"


def _sign(mode: Any) -> str:
    text = str(mode or "")
    if "_neg_" in text or text.startswith("obs_linear_neg") or text.startswith("decay_obs_linear_neg"):
        return "neg"
    if text and text != "None":
        return "pos"
    return "none"


def _stats(values: list[Any]) -> dict[str, Any]:
    arr = np.asarray([_safe_float(v) for v in values], dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "q10": float(np.quantile(arr, 0.10)),
        "q50": float(np.quantile(arr, 0.50)),
        "q90": float(np.quantile(arr, 0.90)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _dominance(counter: Counter[str], total: int) -> dict[str, Any]:
    if total <= 0 or not counter:
        return {"dominant": None, "dominant_count": 0, "dominant_share": None}
    key, count = counter.most_common(1)[0]
    return {"dominant": key, "dominant_count": int(count), "dominant_share": float(count / total)}


def _rows(signature_reports: list[str | Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for report_path in signature_reports:
        report = _read_json(report_path)
        for item in report.get("lists", []):
            label = str(item.get("label"))
            source_tag = _source_tag(label, report_path)
            for row in (item.get("classification") or {}).get("per_env", []):
                enriched = dict(row)
                enriched["source_label"] = label
                enriched["source_tag"] = source_tag
                enriched["report_path"] = str(Path(report_path).expanduser().resolve())
                rows.append(enriched)
    return rows


def _mode_for(row: dict[str, Any], definition: str) -> Any:
    if definition in {"time_profile_core_no_random", "strict_or_time_profile_core_no_random"}:
        if _is_true(row.get("time_profile_core")) and row.get("best_time_profile_mode") != "early_random_then_zero":
            return row.get("best_time_profile_mode")
        return row.get("best_feedback_mode")
    if definition in {
        "strict_or_time_profile_core_no_random_or_early_strong",
        "time_profile_core_no_random_or_early_strong",
    }:
        if _is_true(row.get("time_profile_core")) and row.get("best_time_profile_mode") != "early_random_then_zero":
            return row.get("best_time_profile_mode")
        if _is_true(row.get("early_only_strong_carryover_core")):
            return row.get("best_early_only_mode")
        return row.get("best_feedback_mode")
    return row.get("best_feedback_mode")


def _definitions() -> dict[str, Callable[[dict[str, Any]], bool]]:
    return {
        "strict_current": lambda r: _is_true(r.get("pendulum_like_strict")),
        "core_relaxed": lambda r: _is_true(r.get("pendulum_like_core")),
        "weak_relaxed": lambda r: _is_true(r.get("pendulum_like_weak")),
        "time_profile_core_no_random": lambda r: _is_true(r.get("time_profile_core"))
        and r.get("best_time_profile_mode") != "early_random_then_zero",
        "strict_or_time_profile_core_no_random": lambda r: _is_true(r.get("pendulum_like_strict"))
        or (
            _is_true(r.get("time_profile_core"))
            and r.get("best_time_profile_mode") != "early_random_then_zero"
        ),
        "strict_or_time_profile_core_no_random_or_early_strong": lambda r: _is_true(
            r.get("pendulum_like_strict")
        )
        or (
            _is_true(r.get("time_profile_core"))
            and r.get("best_time_profile_mode") != "early_random_then_zero"
        )
        or _is_true(r.get("early_only_strong_carryover_core")),
    }


def _summary_for(name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    modes = [_mode_for(row, name) for row in rows]
    profile_counts = Counter(_profile(mode) for mode in modes)
    gain_counts = Counter(_gain(mode) for mode in modes)
    sign_counts = Counter(_sign(mode) for mode in modes)
    source_counts = Counter(str(row.get("source_tag")) for row in rows)
    mode_counts = Counter(str(mode) for mode in modes)
    n = len(rows)
    profile_dom = _dominance(profile_counts, n)
    gain_dom = _dominance(gain_counts, n)
    return {
        "n": int(n),
        "source_counts": dict(source_counts),
        "profile_counts": dict(profile_counts),
        "gain_counts": dict(gain_counts),
        "sign_counts": dict(sign_counts),
        "mode_top": [
            {"mode": mode, "count": int(count), "rate": float(count / max(1, n))}
            for mode, count in mode_counts.most_common(10)
        ],
        "profile_dominance": profile_dom,
        "gain_dominance": gain_dom,
        "profile_source_pool_repaired": bool(
            n >= 128 and (profile_dom.get("dominant_share") or 1.0) <= 0.75
        ),
        "gain_source_pool_repaired": bool(
            n >= 128 and (gain_dom.get("dominant_share") or 1.0) <= 0.75
        ),
        "score_stats": {
            "best_feedback_minus_open_env": _stats(
                [row.get("best_feedback_minus_open_env") for row in rows]
            ),
            "feedback_pair_phase_gap_env": _stats(
                [row.get("feedback_pair_phase_gap_env") for row in rows]
            ),
            "feedback_suffix_gain_over_zero_env": _stats(
                [row.get("feedback_suffix_gain_over_zero_env") for row in rows]
            ),
            "best_time_profile_minus_open_env": _stats(
                [
                    _safe_float(row.get("best_time_profile_env_return"))
                    - _safe_float(row.get("best_open_loop_env_return"))
                    for row in rows
                ]
            ),
        },
    }


def build_report(signature_reports: list[str | Path]) -> dict[str, Any]:
    rows = _rows(signature_reports)
    summaries = {}
    for name, pred in _definitions().items():
        summaries[name] = _summary_for(name, [row for row in rows if pred(row)])

    strict = summaries["strict_current"]
    core = summaries["core_relaxed"]
    weak = summaries["weak_relaxed"]
    temporal = summaries["strict_or_time_profile_core_no_random"]
    temporal_early = summaries["strict_or_time_profile_core_no_random_or_early_strong"]
    report = {
        "analysis_entry": "phase2_feedback_source_pool_broaden_audit",
        "contract": {
            "exploratory_only": True,
            "read_only_existing_signature_reports": True,
            "does_not_sample_environments": True,
            "does_not_train": True,
            "does_not_modify_exact_scm": True,
            "does_not_modify_fit_model": True,
            "does_not_modify_ppo": True,
            "profile_quality_first": True,
        },
        "inputs": {"signature_reports": [str(Path(path).expanduser().resolve()) for path in signature_reports]},
        "definitions": {
            "strict_current": "existing pendulum_like_strict using best_feedback_mode",
            "core_relaxed": "drops suffix support from strict; still phase-sensitive feedback",
            "weak_relaxed": "feedback beats zero rather than open-loop; still phase-sensitive",
            "time_profile_core_no_random": "time-profile feedback beats open-loop with action decay, excluding early_random_then_zero",
            "strict_or_time_profile_core_no_random": "current strict union temporal feedback core; candidate definition repair",
            "strict_or_time_profile_core_no_random_or_early_strong": "above plus early-only strong carryover; broader but semantically mixed",
        },
        "summaries": summaries,
        "read": {
            "strict_source_pool_collapse_confirmed": bool(
                (strict["profile_dominance"].get("dominant_share") or 0.0) > 0.8
                and (strict["gain_dominance"].get("dominant_share") or 0.0) > 0.8
            ),
            "core_or_weak_relaxation_sufficient": bool(
                core["profile_source_pool_repaired"]
                and core["gain_source_pool_repaired"]
                or weak["profile_source_pool_repaired"]
                and weak["gain_source_pool_repaired"]
            ),
            "temporal_union_repairs_profile": temporal["profile_source_pool_repaired"],
            "temporal_union_repairs_gain": temporal["gain_source_pool_repaired"],
            "broader_temporal_early_repairs_gain": temporal_early["gain_source_pool_repaired"],
            "recommended_next": (
                "Use strict_or_time_profile_core_no_random as a candidate source-pool definition repair "
                "for profile diversity, but do not call it sufficient for a milestone because gain=2.0 "
                "dominance remains.  The gain issue needs either a near-best gain detector or a narrow "
                "generator-level gain/profile branch guarded by delta/diversity."
            ),
        },
    }
    return report


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Feedback Source-Pool Broaden Audit",
        "",
        "Read-only definition audit.  It checks whether feedback strict collapse is solved by broadening the label.",
        "",
        "| definition | n | profile dominant | profile share | gain dominant | gain share | profile repaired | gain repaired |",
        "|---|---:|---|---:|---|---:|---|---|",
    ]
    for name, item in report["summaries"].items():
        pd = item["profile_dominance"]
        gd = item["gain_dominance"]
        lines.append(
            "| {name} | {n} | {pdom} | {pshare} | {gdom} | {gshare} | {prep} | {grep} |".format(
                name=name,
                n=item["n"],
                pdom=pd.get("dominant"),
                pshare="na"
                if pd.get("dominant_share") is None
                else f"{100.0 * float(pd['dominant_share']):.1f}%",
                gdom=gd.get("dominant"),
                gshare="na"
                if gd.get("dominant_share") is None
                else f"{100.0 * float(gd['dominant_share']):.1f}%",
                prep=item["profile_source_pool_repaired"],
                grep=item["gain_source_pool_repaired"],
            )
        )
    lines.extend(["", "## Read", ""])
    for key, value in report["read"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Files", ""])
    for key, value in report.get("outputs", {}).items():
        lines.append(f"- `{key}`: `{value}`")
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(args.signature_report)
    json_path = out_dir / "feedback_source_pool_broaden_audit.json"
    md_path = out_dir / "feedback_source_pool_broaden_audit.md"
    report["outputs"] = {"json": str(json_path), "markdown": str(md_path)}
    json_path.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report["outputs"], indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--signature-report",
        action="append",
        default=[str(path) for path in DEFAULT_SIGNATURE_REPORTS],
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
