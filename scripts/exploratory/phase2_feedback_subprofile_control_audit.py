#!/usr/bin/env python
"""Audit whether strict feedback subprofile collapse has simple controls.

Exploratory/read-only.  This parses the existing Pendulum-like strict candidate
pool, joins each row with its frozen h parameters, and checks whether
`profile != constant` or `gain != 2.0` can be explained by a few simple existing
generator parameters.  It does not sample, train, or mutate any generator path.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_INPUT_CSV = (
    "/home/chen/RLPFN/artifacts/phase2_pendulum_like_strict_cross_seed6260_6261_6262_all_candidates_0510/"
    "pendulum_like_signature_fixedlist.csv"
)
DEFAULT_OUTPUT_DIR = "/home/chen/RLPFN/artifacts/phase2_feedback_subprofile_control_audit_0511"

PROFILE_VALUES = ("constant", "decay", "early_then_zero")
CONTROL_PARAM_HINTS = (
    "reward_dropout_ratio",
    "terminal_reset_count_target",
    "reward_scale",
    "alpha",
    "ctrl_reward_weight",
    "state_noise_std",
    "init_state_std",
    "init_action_std",
    "outputscale",
    "lengthscale",
    "noise_std",
    "reward_tanh",
    "terminal_bonus",
)


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).expanduser().resolve().open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return math.nan
    return out if math.isfinite(out) else math.nan


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


def _parse_feedback_mode(mode: str) -> dict[str, Any]:
    text = str(mode or "")
    if text.startswith("early_") or "_then_zero" in text:
        profile = "early_then_zero"
    elif text.startswith("decay_"):
        profile = "decay"
    else:
        profile = "constant"
    sign = "neg" if "_neg_" in text or text.startswith("obs_linear_neg") else "pos"
    gain = math.nan
    if "x50" in text:
        gain = 0.5
    elif "x100" in text:
        gain = 1.0
    elif "x200" in text:
        gain = 2.0
    feature = None
    match = re.search(r"_(\d+)x(?:50|100|200)", text)
    if match:
        feature = int(match.group(1))
    return {"profile": profile, "sign": sign, "gain": gain, "feature": feature}


def _numeric_h_fields(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    values: dict[str, list[float]] = {}
    for row in rows:
        h = row["h"]
        for key, value in h.items():
            if isinstance(value, bool):
                continue
            val = _safe_float(value)
            if math.isfinite(val):
                values.setdefault(key, []).append(val)
    out: dict[str, np.ndarray] = {}
    n = len(rows)
    for key, vals in values.items():
        if len(vals) != n:
            continue
        arr = np.asarray(vals, dtype=np.float64)
        if arr.size == n and float(np.nanstd(arr)) > 1e-12:
            out[key] = arr
    return out


def _point_biserial(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3 or float(np.std(x)) < 1e-12 or float(np.std(y)) < 1e-12:
        return math.nan
    return float(np.corrcoef(x, y)[0, 1])


def _stats(values: list[float] | np.ndarray) -> dict[str, Any]:
    arr = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=np.float64)
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "q10": float(np.quantile(arr, 0.10)),
        "q50": float(np.quantile(arr, 0.50)),
        "q90": float(np.quantile(arr, 0.90)),
        "max": float(np.max(arr)),
    }


def _control_candidate_score(key: str, corr: float) -> float:
    hint = any(token in key for token in CONTROL_PARAM_HINTS)
    return abs(corr) + (0.05 if hint else 0.0)


def _parameter_associations(rows: list[dict[str, Any]], target_key: str) -> list[dict[str, Any]]:
    fields = _numeric_h_fields(rows)
    y = np.asarray([1.0 if row[target_key] else 0.0 for row in rows], dtype=np.float64)
    out = []
    for key, x in fields.items():
        corr = _point_biserial(x, y)
        if not math.isfinite(corr):
            continue
        pos = x[y > 0.5]
        neg = x[y <= 0.5]
        out.append(
            {
                "param": key,
                "corr": corr,
                "abs_corr": abs(corr),
                "mean_true": float(np.mean(pos)) if pos.size else math.nan,
                "mean_false": float(np.mean(neg)) if neg.size else math.nan,
                "diff_true_minus_false": float(np.mean(pos) - np.mean(neg)) if pos.size and neg.size else math.nan,
                "control_candidate_score": _control_candidate_score(key, corr),
                "looks_like_control_knob": any(token in key for token in CONTROL_PARAM_HINTS),
            }
        )
    out.sort(key=lambda item: item["control_candidate_score"], reverse=True)
    return out


def _categorical_counts(rows: list[dict[str, Any]], target_key: str, field: str) -> dict[str, Any]:
    total = Counter(str(row.get(field)) for row in rows)
    true = Counter(str(row.get(field)) for row in rows if row[target_key])
    false = Counter(str(row.get(field)) for row in rows if not row[target_key])
    return {
        "total": dict(total),
        "true": dict(true),
        "false": dict(false),
    }


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    enriched: list[dict[str, Any]] = []
    for row in _read_csv(args.input_csv):
        parsed = _parse_feedback_mode(row.get("best_feedback_mode", ""))
        h = _read_json(row["full_frozen_h_json"])
        enriched.append(
            {
                **row,
                "h": h,
                "parsed_profile": parsed["profile"],
                "parsed_gain": parsed["gain"],
                "parsed_sign": parsed["sign"],
                "parsed_feature": parsed["feature"],
                "nonconstant_profile": parsed["profile"] != "constant",
                "non_gain2": math.isfinite(parsed["gain"]) and abs(parsed["gain"] - 2.0) > 1e-9,
                "nonconstant_and_non_gain2": parsed["profile"] != "constant"
                and math.isfinite(parsed["gain"])
                and abs(parsed["gain"] - 2.0) > 1e-9,
            }
        )

    profile_counts = Counter(row["parsed_profile"] for row in enriched)
    gain_counts = Counter(str(row["parsed_gain"]) for row in enriched)
    labels = {
        "nonconstant_profile": _parameter_associations(enriched, "nonconstant_profile"),
        "non_gain2": _parameter_associations(enriched, "non_gain2"),
        "nonconstant_and_non_gain2": _parameter_associations(enriched, "nonconstant_and_non_gain2"),
    }
    strongest = {
        key: (items[0] if items else None)
        for key, items in labels.items()
    }
    max_control_corr = max(
        [
            abs(item["corr"])
            for items in labels.values()
            for item in items
            if item.get("looks_like_control_knob")
        ]
        or [math.nan]
    )
    simple_knob_sufficient = bool(math.isfinite(max_control_corr) and max_control_corr >= float(args.strong_corr))

    report = {
        "analysis_entry": "phase2_feedback_subprofile_control_audit",
        "contract": {
            "exploratory_only": True,
            "read_only_existing_artifacts": True,
            "does_not_sample_environments": True,
            "does_not_train": True,
            "does_not_modify_exact_scm": True,
            "does_not_modify_fit_model": True,
            "does_not_modify_ppo": True,
        },
        "input_csv": str(Path(args.input_csv).expanduser().resolve()),
        "n": len(enriched),
        "profile_counts": dict(profile_counts),
        "gain_counts": dict(gain_counts),
        "signature_score": _stats([_safe_float(row.get("signature_score")) for row in enriched]),
        "target_counts": {
            key: int(sum(bool(row[key]) for row in enriched))
            for key in ("nonconstant_profile", "non_gain2", "nonconstant_and_non_gain2")
        },
        "target_categorical_counts": {
            key: {
                "source_tag": _categorical_counts(enriched, key, "source_tag"),
                "parsed_sign": _categorical_counts(enriched, key, "parsed_sign"),
                "parsed_feature": _categorical_counts(enriched, key, "parsed_feature"),
            }
            for key in ("nonconstant_profile", "non_gain2", "nonconstant_and_non_gain2")
        },
        "top_parameter_associations": {
            key: items[:12]
            for key, items in labels.items()
        },
        "strongest_association": strongest,
        "simple_knob_sufficient": simple_knob_sufficient,
        "read": {
            "feedback_subprofile_is_pressure_point": True,
            "selection_only_already_known_insufficient": True,
            "single_existing_scalar_knob_sufficient": simple_knob_sufficient,
            "reason": (
                "A strong existing scalar parameter association would support a simple generator knob. "
                "If not present, the safer next step is a small explicit categorical subprofile branch "
                "inside the generator-level profile controller, still guarded by health/diversity/parity."
            ),
        },
    }

    report_path = out_dir / "feedback_subprofile_control_audit.json"
    md_path = out_dir / "feedback_subprofile_control_audit.md"
    report_path.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Feedback Subprofile Control Audit",
        "",
        "Exploratory/read-only.  It checks whether strict feedback profile/gain collapse has a simple existing parameter control.",
        "",
        f"- n: {len(enriched)}",
        f"- profile counts: `{dict(profile_counts)}`",
        f"- gain counts: `{dict(gain_counts)}`",
        f"- simple existing scalar knob sufficient: `{simple_knob_sufficient}`",
        "",
        "## Top Parameter Associations",
        "",
    ]
    for key, items in report["top_parameter_associations"].items():
        lines.append(f"### {key}")
        lines.append("")
        lines.append("| param | corr | true mean | false mean | diff | knob-like |")
        lines.append("|---|---:|---:|---:|---:|---|")
        for item in items[:8]:
            lines.append(
                "| {param} | {corr:.3f} | {mt:.4g} | {mf:.4g} | {diff:.4g} | {knob} |".format(
                    param=item["param"],
                    corr=item["corr"],
                    mt=item["mean_true"],
                    mf=item["mean_false"],
                    diff=item["diff_true_minus_false"],
                    knob=item["looks_like_control_knob"],
                )
            )
        lines.append("")
    lines.extend(
        [
            "## Read",
            "",
            "- The pressure point remains feedback strict subprofile collapse.",
            "- Existing scalar parameters are only candidates if their association is strong enough; otherwise use an explicit profile/subprofile controller rather than many ad hoc knobs.",
            f"- JSON: `{report_path}`",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path), "summary": str(md_path)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--strong-corr", type=float, default=0.35)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
