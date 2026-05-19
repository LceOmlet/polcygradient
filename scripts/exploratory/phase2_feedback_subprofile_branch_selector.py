#!/usr/bin/env python
"""Select a balanced strict-feedback branch from existing fixed candidates.

Exploratory only.  This is the first maintainable control point for the
Pendulum-like strict feedback pressure point: it does not try to increase the
amount of feedback environments.  Instead, within the already strict feedback
candidate pool, it reduces collapse onto:

* `profile=constant`: the same finite-probe feedback controller is best at
  every step;
* `gain=2.0`: the largest finite-probe gain is best.

The output is a fixed-list branch that can be passed to the trusted pack runner
for health/diversity/trainability guards.  No exact SCM, fit_model, PPO, or
milestone default path is modified here.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


ART = Path("/home/chen/RLPFN/artifacts")
DEFAULT_INPUT_CSV = (
    ART
    / "phase2_pendulum_like_strict_cross_seed6260_6261_6262_all_candidates_0510"
    / "pendulum_like_signature_fixedlist.csv"
)
DEFAULT_OUTPUT_DIR = ART / "phase2_feedback_strict_subprofile_branch_v1_n32_0511"
MODE_RE = re.compile(r"^(?P<profile>early_|decay_)?obs_linear(?P<neg>_neg)?_(?P<feature>\d+)x(?P<gain>\d+)")


DEFAULT_CONFIG = {
    "branch_version_id": "feedback_strict_subprofile_branch_v1",
    "background_milestone": "gated_reward_path_balance",
    "target_n": 32,
    "profile_constant_max_rate": 0.60,
    "gain2_max_rate": 0.375,
    "sign_min_rate": 0.40,
    "sign_max_rate": 0.60,
    "source_tag_max_rate": 0.34,
    "feature_max_rate": 0.60,
    "include_all_nonconstant_non_gain2": True,
    "objective": {
        "signature_score_weight": 1.0,
        "nonconstant_bonus": 250.0,
        "non_gain2_bonus": 120.0,
        "joint_nonconstant_non_gain2_bonus": 600.0,
        "rare_profile_bonus": 80.0,
        "rare_gain_bonus": 40.0,
    },
}


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


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _safe_label(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")[:160] or "rule"


def _read_csv(path: str | Path) -> list[dict[str, Any]]:
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


def _parse_mode(mode: Any) -> dict[str, Any]:
    text = "" if mode is None else str(mode)
    match = MODE_RE.match(text)
    if not match:
        return {
            "parsed_mode": text,
            "parsed_profile": "unknown" if text else "none",
            "parsed_gain": math.nan,
            "parsed_sign": "unknown" if text else "none",
            "parsed_feature": None,
        }
    raw_profile = match.group("profile") or ""
    profile = "constant"
    if raw_profile == "early_":
        profile = "early_then_zero"
    elif raw_profile == "decay_":
        profile = "decay"
    return {
        "parsed_mode": text,
        "parsed_profile": profile,
        "parsed_gain": float(match.group("gain")) / 100.0,
        "parsed_sign": "neg" if match.group("neg") else "pos",
        "parsed_feature": int(match.group("feature")),
    }


def _load_rows(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    for idx, row in enumerate(_read_csv(path)):
        parsed = _parse_mode(row.get("best_feedback_mode"))
        out: dict[str, Any] = dict(row)
        out.update(parsed)
        out["source_index"] = int(idx)
        out["env_seed"] = int(float(row.get("env_seed", row.get("seed", 0))))
        out["signature_score_float"] = _safe_float(row.get("signature_score"))
        out["is_constant_profile"] = parsed["parsed_profile"] == "constant"
        out["is_gain2"] = math.isfinite(parsed["parsed_gain"]) and abs(parsed["parsed_gain"] - 2.0) < 1e-9
        out["is_nonconstant_non_gain2"] = (not out["is_constant_profile"]) and (not out["is_gain2"])
        rows.append(out)
    return rows


def _counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(Counter(str(row.get(key)) for row in rows))


def _dominance(counts: dict[str, int]) -> dict[str, Any]:
    n = sum(int(v) for v in counts.values())
    if n <= 0:
        return {"dominant": None, "dominant_rate": None, "status": "empty"}
    label, count = max(counts.items(), key=lambda item: int(item[1]))
    rate = float(int(count) / n)
    if len([v for v in counts.values() if int(v) > 0]) <= 1:
        status = "collapsed"
    elif rate >= 0.80:
        status = "dominated"
    elif rate >= 0.65:
        status = "skewed"
    else:
        status = "ok"
    return {"dominant": str(label), "dominant_rate": rate, "status": status}


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    profile_counts = _counts(rows, "parsed_profile")
    gain_counts = _counts(rows, "parsed_gain")
    source_counts = _counts(rows, "source_tag")
    sign_counts = _counts(rows, "parsed_sign")
    feature_counts = _counts(rows, "parsed_feature")
    return {
        "n": n,
        "profile_counts": profile_counts,
        "profile_rates": {key: value / max(1, n) for key, value in profile_counts.items()},
        "profile_dominance": _dominance(profile_counts),
        "gain_counts": gain_counts,
        "gain_rates": {key: value / max(1, n) for key, value in gain_counts.items()},
        "gain_dominance": _dominance(gain_counts),
        "source_tag_counts": source_counts,
        "sign_counts": sign_counts,
        "feature_counts": feature_counts,
        "nonconstant_non_gain2_count": int(sum(bool(row.get("is_nonconstant_non_gain2")) for row in rows)),
        "nonconstant_non_gain2_rate": float(
            sum(bool(row.get("is_nonconstant_non_gain2")) for row in rows) / max(1, n)
        ),
        "signature_score_mean": float(np.mean([_safe_float(row.get("signature_score_float")) for row in rows]))
        if rows
        else None,
        "top_modes": dict(Counter(str(row.get("parsed_mode")) for row in rows).most_common(12)),
    }


def _selection_only_frontier(rows: list[dict[str, Any]], sizes: list[int]) -> list[dict[str, Any]]:
    total_nonconstant = sum(not bool(row["is_constant_profile"]) for row in rows)
    total_non_gain2 = sum(not bool(row["is_gain2"]) for row in rows)
    total_joint = sum(bool(row["is_nonconstant_non_gain2"]) for row in rows)
    out: list[dict[str, Any]] = []
    for size in sizes:
        n = int(size)
        min_constant = max(0, n - total_nonconstant)
        min_gain2 = max(0, n - total_non_gain2)
        out.append(
            {
                "n": n,
                "max_nonconstant_count": int(min(total_nonconstant, n)),
                "max_nonconstant_rate": float(min(total_nonconstant, n) / max(1, n)),
                "min_constant_count": int(min_constant),
                "min_constant_rate": float(min_constant / max(1, n)),
                "max_non_gain2_count": int(min(total_non_gain2, n)),
                "max_non_gain2_rate": float(min(total_non_gain2, n) / max(1, n)),
                "min_gain2_count": int(min_gain2),
                "min_gain2_rate": float(min_gain2 / max(1, n)),
                "max_joint_nonconstant_non_gain2_count": int(min(total_joint, n)),
                "max_joint_nonconstant_non_gain2_rate": float(min(total_joint, n) / max(1, n)),
            }
        )
    return out


def _constraint_indicator(rows: list[dict[str, Any]], key: str, value: Any) -> np.ndarray:
    return np.asarray([1.0 if str(row.get(key)) == str(value) else 0.0 for row in rows], dtype=np.float64)


def _add_cap(
    a_rows: list[np.ndarray],
    lb: list[float],
    ub: list[float],
    indicator: np.ndarray,
    *,
    cap: int,
) -> None:
    a_rows.append(indicator)
    lb.append(0.0)
    ub.append(float(cap))


def _select(rows: list[dict[str, Any]], config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    target_n = int(config["target_n"])
    if len(rows) < target_n:
        raise ValueError(f"need {target_n} rows, got {len(rows)}")
    n = len(rows)
    a_rows: list[np.ndarray] = [np.ones(n, dtype=np.float64)]
    lb: list[float] = [float(target_n)]
    ub: list[float] = [float(target_n)]
    details: list[dict[str, Any]] = []

    def add_named(name: str, indicator: np.ndarray, low: int, high: int) -> None:
        a_rows.append(indicator)
        lb.append(float(low))
        ub.append(float(high))
        details.append({"name": name, "min_count": int(low), "max_count": int(high), "source_count": int(indicator.sum())})

    constant = np.asarray([1.0 if row["is_constant_profile"] else 0.0 for row in rows], dtype=np.float64)
    gain2 = np.asarray([1.0 if row["is_gain2"] else 0.0 for row in rows], dtype=np.float64)
    add_named(
        "profile_constant_cap",
        constant,
        0,
        int(math.floor(float(config["profile_constant_max_rate"]) * target_n + 1e-9)),
    )
    add_named("gain2_cap", gain2, 0, int(math.floor(float(config["gain2_max_rate"]) * target_n + 1e-9)))

    signs = sorted({str(row.get("parsed_sign")) for row in rows if str(row.get("parsed_sign")) in {"pos", "neg"}})
    sign_low = int(math.ceil(float(config["sign_min_rate"]) * target_n - 1e-9))
    sign_high = int(math.floor(float(config["sign_max_rate"]) * target_n + 1e-9))
    for sign in signs:
        add_named(f"sign_{sign}_band", _constraint_indicator(rows, "parsed_sign", sign), sign_low, sign_high)

    source_cap = int(math.ceil(float(config["source_tag_max_rate"]) * target_n - 1e-9))
    for source in sorted({str(row.get("source_tag")) for row in rows}):
        add_named(f"source_{source}_cap", _constraint_indicator(rows, "source_tag", source), 0, source_cap)

    feature_cap = int(math.ceil(float(config["feature_max_rate"]) * target_n - 1e-9))
    for feature in sorted({str(row.get("parsed_feature")) for row in rows}):
        add_named(f"feature_{feature}_cap", _constraint_indicator(rows, "parsed_feature", feature), 0, feature_cap)

    if bool(config.get("include_all_nonconstant_non_gain2", True)):
        joint = np.asarray([1.0 if row["is_nonconstant_non_gain2"] else 0.0 for row in rows], dtype=np.float64)
        add_named("include_all_nonconstant_non_gain2", joint, int(joint.sum()), int(joint.sum()))

    scores = np.asarray([_safe_float(row.get("signature_score_float")) for row in rows], dtype=np.float64)
    scale = float(np.std(scores))
    if scale > 1e-12:
        norm_score = (scores - float(np.mean(scores))) / scale
    else:
        norm_score = np.zeros_like(scores)
    obj_cfg = config.get("objective", {})
    objective = -float(obj_cfg.get("signature_score_weight", 1.0)) * norm_score
    objective -= float(obj_cfg.get("nonconstant_bonus", 0.0)) * (1.0 - constant)
    objective -= float(obj_cfg.get("non_gain2_bonus", 0.0)) * (1.0 - gain2)
    objective -= float(obj_cfg.get("joint_nonconstant_non_gain2_bonus", 0.0)) * np.asarray(
        [1.0 if row["is_nonconstant_non_gain2"] else 0.0 for row in rows], dtype=np.float64
    )

    profile_counts = Counter(str(row.get("parsed_profile")) for row in rows)
    gain_counts = Counter(str(row.get("parsed_gain")) for row in rows)
    for idx, row in enumerate(rows):
        objective[idx] -= float(obj_cfg.get("rare_profile_bonus", 0.0)) / max(1, profile_counts[str(row["parsed_profile"])])
        objective[idx] -= float(obj_cfg.get("rare_gain_bonus", 0.0)) / max(1, gain_counts[str(row["parsed_gain"])])

    result = milp(
        c=objective,
        integrality=np.ones(n, dtype=np.int8),
        bounds=Bounds(np.zeros(n), np.ones(n)),
        constraints=LinearConstraint(np.vstack(a_rows), np.asarray(lb), np.asarray(ub)),
    )
    if not result.success or result.x is None:
        return [], {
            "solver_success": False,
            "solver_message": str(result.message),
            "constraints": details,
            "guard_pass": False,
        }

    selected_idx = [idx for idx, value in enumerate(result.x) if value >= 0.5]
    selected = [dict(rows[idx]) for idx in selected_idx]
    selected.sort(key=lambda row: (str(row.get("source_tag")), int(row.get("source_index", 0))))
    for rank, row in enumerate(selected):
        row["feedback_subprofile_branch_rank"] = int(rank)
        row["candidate_rule"] = f"{config['branch_version_id']}_n{target_n}"
        row["feedback_subprofile_branch_version"] = str(config["branch_version_id"])
        row["feedback_subprofile_branch_guard"] = "strict_feedback_subprofile_branch_selector"

    selected_summary = _summary(selected)
    for detail in details:
        if detail["name"] == "profile_constant_cap":
            selected_count = int(selected_summary["profile_counts"].get("constant", 0))
        elif detail["name"] == "gain2_cap":
            selected_count = int(selected_summary["gain_counts"].get("2.0", 0))
        elif detail["name"].startswith("sign_"):
            selected_count = int(selected_summary["sign_counts"].get(detail["name"].split("_")[1], 0))
        elif detail["name"].startswith("source_"):
            selected_count = int(selected_summary["source_tag_counts"].get(detail["name"].removeprefix("source_").removesuffix("_cap"), 0))
        elif detail["name"].startswith("feature_"):
            selected_count = int(selected_summary["feature_counts"].get(detail["name"].removeprefix("feature_").removesuffix("_cap"), 0))
        elif detail["name"] == "include_all_nonconstant_non_gain2":
            selected_count = int(selected_summary["nonconstant_non_gain2_count"])
        else:
            selected_count = 0
        detail["selected_count"] = selected_count
        detail["passed"] = bool(detail["min_count"] <= selected_count <= detail["max_count"])

    guard_pass = all(detail["passed"] for detail in details)
    return selected, {
        "solver_success": True,
        "solver_message": str(result.message),
        "solver_fun": float(result.fun),
        "constraints": details,
        "guard_pass": bool(guard_pass),
    }


def _materialize(out_dir: Path, rows: list[dict[str, Any]], *, rule: str) -> dict[str, str]:
    rule_dir = out_dir / _safe_label(rule) / "fixed_env_group"
    frozen_dir = rule_dir / "frozen_h"
    frozen_dir.mkdir(parents=True, exist_ok=True)
    h_list: list[Any] = []
    seeds: list[int] = []
    for idx, row in enumerate(rows):
        src = Path(str(row["full_frozen_h_json"])).expanduser().resolve()
        h = _read_json(src)
        seed = int(float(row["env_seed"]))
        dst = frozen_dir / f"env_{idx:04d}_seed{seed}.json"
        shutil.copyfile(src, dst)
        h_list.append(h)
        seeds.append(seed)
    h_path = rule_dir / "prior_fixed_h_list.json"
    seed_path = rule_dir / "prior_fixed_env_seeds.json"
    h_path.write_text(json.dumps(_json_safe(h_list), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    seed_path.write_text(json.dumps(seeds, indent=2) + "\n", encoding="utf-8")
    return {"fixed_h_list_json": str(h_path), "fixed_env_seed_json": str(seed_path)}


def _load_config(path: str | None) -> dict[str, Any]:
    if path:
        cfg = _read_json(path)
        if not isinstance(cfg, dict):
            raise TypeError(f"branch config must be a JSON object: {path}")
        return cfg
    return json.loads(json.dumps(DEFAULT_CONFIG))


def _write_summary_md(path: Path, report: dict[str, Any]) -> None:
    pool = report["pool_summary"]
    selected = report["selected_summary"]
    lines = [
        "# Feedback Strict Subprofile Branch Selector",
        "",
        "Exploratory branch materialization only.  It does not raise feedback count in the full prior distribution and does not modify milestone defaults.",
        "",
        "## Definitions",
        "",
        "- `strict feedback pool`: environments already passing the Pendulum-like strict feedback detector.",
        "- `profile=constant`: the best finite-probe feedback controller uses the same observation feedback throughout the rollout.",
        "- `profile=decay`: the same feedback form is tapered through time.",
        "- `profile=early_then_zero`: feedback is active early and then zero in the suffix.",
        "- `gain=2.0`: the finite probe's largest action gain wins.",
        "",
        "The goal is lower subprofile domination, not more feedback environments.",
        "",
        "## Pool vs Selected",
        "",
        "| set | n | profile counts | gain counts | profile dominance | gain dominance | joint nonconstant/non-gain2 |",
        "| --- | ---: | --- | --- | --- | --- | ---: |",
        (
            f"| pool | {pool['n']} | `{pool['profile_counts']}` | `{pool['gain_counts']}` | "
            f"`{pool['profile_dominance']}` | `{pool['gain_dominance']}` | "
            f"{pool['nonconstant_non_gain2_count']} |"
        ),
        (
            f"| selected | {selected['n']} | `{selected['profile_counts']}` | `{selected['gain_counts']}` | "
            f"`{selected['profile_dominance']}` | `{selected['gain_dominance']}` | "
            f"{selected['nonconstant_non_gain2_count']} |"
        ),
        "",
        "## Selection-Only Boundary",
        "",
    ]
    for row in report["selection_only_frontier"]:
        lines.append(
            f"- n={row['n']}: min constant {row['min_constant_count']}/{row['n']} "
            f"({row['min_constant_rate']:.3f}), min gain2 {row['min_gain2_count']}/{row['n']} "
            f"({row['min_gain2_rate']:.3f}), max joint nonconstant/non-gain2 "
            f"{row['max_joint_nonconstant_non_gain2_count']}/{row['n']} "
            f"({row['max_joint_nonconstant_non_gain2_rate']:.3f})"
        )
    lines.extend(
        [
            "",
        "## Guard Read",
        "",
        f"- branch solver guard: `{report['selection']['guard_pass']}`",
        "- health/diversity/raw-delta guards: not run by this selector; must be run with trusted pack fixed-list smoke.",
        "- full/q90 broad profile bands: preserved by construction because this is only a branch artifact, not a default distribution change.",
        "",
        "## Outputs",
        "",
        f"- selected CSV: `{report['outputs']['selected_csv']}`",
        f"- JSON: `{report['outputs']['report_json']}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    config = _load_config(args.branch_config)
    config["target_n"] = int(args.target_n or config["target_n"])
    rows = _load_rows(args.input_csv)
    selected, selection = _select(rows, config)
    if not selected:
        raise RuntimeError(f"feedback strict branch selection failed: {selection.get('solver_message')}")

    selected_csv = out_dir / "selected_feedback_strict_subprofile_branch.csv"
    report_json = out_dir / "feedback_strict_subprofile_branch_report.json"
    report_md = out_dir / "feedback_strict_subprofile_branch.md"
    config_json = out_dir / "feedback_strict_subprofile_branch_config.json"
    rule = f"{config['branch_version_id']}_n{int(config['target_n'])}"
    fixed_group = _materialize(out_dir, selected, rule=rule)
    _write_csv(selected_csv, selected)
    config_json.write_text(json.dumps(_json_safe(config), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report = {
        "analysis_entry": "phase2_feedback_subprofile_branch_selector",
        "contract": {
            "exploratory_only": True,
            "does_not_sample_new_envs": True,
            "does_not_train": True,
            "does_not_modify_exact_scm": True,
            "does_not_modify_fit_model": True,
            "does_not_modify_ppo": True,
            "does_not_change_milestone_defaults": True,
            "selection_branch_only": True,
            "feedback_quantity_not_raised": True,
            "health_guards_require_trusted_pack_smoke": True,
        },
        "inputs": {
            "input_csv": str(Path(args.input_csv).expanduser().resolve()),
            "branch_config": str(Path(args.branch_config).expanduser().resolve()) if args.branch_config else None,
        },
        "config": config,
        "pool_summary": _summary(rows),
        "selected_summary": _summary(selected),
        "selection_only_frontier": _selection_only_frontier(rows, [16, 32, 64, 128]),
        "selection": selection,
        "fixed_group": fixed_group,
        "outputs": {
            "output_dir": str(out_dir),
            "selected_csv": str(selected_csv),
            "report_json": str(report_json),
            "summary_md": str(report_md),
            "config_json": str(config_json),
        },
        "read": {
            "branch_control_reduces_constant_gain2_dominance": (
                _summary(selected)["profile_dominance"]["dominant_rate"]
                < _summary(rows)["profile_dominance"]["dominant_rate"]
                and _summary(selected)["gain_dominance"]["dominant_rate"]
                < _summary(rows)["gain_dominance"]["dominant_rate"]
            ),
            "eligible_for_health_smoke": bool(selection.get("guard_pass")),
            "eligible_for_milestone": False,
            "why_not_milestone": (
                "This only controls the strict feedback subbranch. It still needs trusted pack smoke and "
                "must later be integrated with full/q90 profile bands plus the MountainCar detector."
            ),
        },
    }
    report_json.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_summary_md(report_md, report)
    print(json.dumps(report["outputs"], indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", default=str(DEFAULT_INPUT_CSV))
    parser.add_argument("--branch-config", default=None)
    parser.add_argument("--target-n", type=int, default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
