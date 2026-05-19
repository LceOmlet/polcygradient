#!/usr/bin/env python
"""Summarize guarded locomotion-candidate short PPO smoke results.

Read-only exploratory analysis.  It compares first/last semantic sidecars for
the guarded candidate fixed lists and reports whether raw return deltas and
state/reward/done diversity remain healthy relative to the previous milestone-2
fixed-list smoke.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_OUTPUT_DIR = (
    "/home/chen/RLPFN/artifacts/phase2_locomotion_guarded_candidate_smoke_summary_gpu0_0511"
)
DEFAULT_SELECTED_CSV = (
    "/home/chen/RLPFN/artifacts/phase2_locomotion_guarded_candidate_gated_n128_gpu0_0510/"
    "selected_locomotion_guarded_prior_envs.csv"
)
DEFAULT_FULL_RUN_DIR = (
    "/home/chen/RLPFN/artifacts/"
    "phase2_locomotion_guarded_candidate_full_sidecar_s512_u4_e4_advnorm_gpu0_0511"
)
DEFAULT_Q90_RUN_DIR = (
    "/home/chen/RLPFN/artifacts/"
    "phase2_locomotion_guarded_candidate_q90_sidecar_s512_u4_e4_advnorm_gpu0_0511"
)
DEFAULT_BASELINE_FULL_RUN_DIR = (
    "/home/chen/RLPFN/artifacts/"
    "phase2_locomotion_trainability_full_sidecar_s512_u4_e4_advnorm_gpu0_0510"
)
DEFAULT_BASELINE_Q90_RUN_DIR = (
    "/home/chen/RLPFN/artifacts/"
    "phase2_locomotion_trainability_q90_sidecar_s512_u4_e4_advnorm_gpu0_0510"
)


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


def _load_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).expanduser().resolve().open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _f(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _stats(values: list[Any]) -> dict[str, Any]:
    arr = np.asarray([float(v) for v in values if _f(v) is not None], dtype=np.float64)
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
        "positive_fraction": float(np.mean(arr > 0.0)),
        "negative_count": int(np.sum(arr < 0.0)),
    }


def _sidecar_files(run_dir: str | Path) -> list[Path]:
    sidecar_dir = Path(run_dir).expanduser().resolve() / "semantic_sidecars"
    return sorted(sidecar_dir.glob("prior_update_*_envs.jsonl"))


def _sidecar_pair(run_dir: str | Path) -> tuple[Path, Path]:
    files = _sidecar_files(run_dir)
    if len(files) < 2:
        raise RuntimeError(f"Need at least two sidecars in {run_dir}, found {len(files)}")
    return files[0], files[-1]


def _load_pair(run_dir: str | Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Path, Path]:
    first_path, last_path = _sidecar_pair(run_dir)
    first = _load_jsonl(first_path)
    last = _load_jsonl(last_path)
    if len(first) != len(last):
        raise RuntimeError(f"Sidecar length mismatch: {first_path} vs {last_path}")
    return first, last, first_path, last_path


def _delta(last: dict[str, Any], first: dict[str, Any], key: str) -> float | None:
    a = _f(first.get(key))
    b = _f(last.get(key))
    if a is None or b is None:
        return None
    return float(b - a)


def _label_rows(selected_csv: str | Path, rule: str) -> dict[int, dict[str, Any]]:
    rows = [row for row in _load_csv(selected_csv) if str(row.get("candidate_rule") or row.get("rule")) == rule]
    out: dict[int, dict[str, Any]] = {}
    for idx, row in enumerate(rows):
        out[idx] = {
            "locomotion_witness": _bool(row.get("locomotion_witness")),
            "low_energy_not_ctrl_only": _bool(row.get("low_energy_not_ctrl_only")),
            "sign_or_phase_sensitive": _bool(row.get("sign_or_phase_sensitive")),
            "state_conditioned_energy_injection": _bool(row.get("state_conditioned_energy_injection")),
            "locomotion_env_gain_vs_zero": _f(row.get("locomotion_env_gain_vs_zero")),
            "source_index": int(float(row.get("_source_index", idx))),
            "env_seed": int(float(row.get("env_seed", -1))),
        }
    return out


def _per_env_rows(
    *,
    rule: str,
    run_dir: str | Path,
    selected_csv: str | Path | None,
    source_tag: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    first, last, first_path, last_path = _load_pair(run_dir)
    labels = _label_rows(selected_csv, rule) if selected_csv is not None else {}
    rows = []
    for idx, (a, b) in enumerate(zip(first, last, strict=True)):
        label = labels.get(idx, {})
        row = {
            "source_tag": source_tag,
            "rule": rule,
            "env_idx": int(b.get("env_idx", idx)),
            "env_seed": b.get("env_group_env_seed"),
            "action_dim": b.get("action_dim"),
            "obs_dim": b.get("obs_dim"),
            "locomotion_witness": bool(label.get("locomotion_witness", False)),
            "low_energy_not_ctrl_only": bool(label.get("low_energy_not_ctrl_only", False)),
            "sign_or_phase_sensitive": bool(label.get("sign_or_phase_sensitive", False)),
            "state_conditioned_energy_injection": bool(label.get("state_conditioned_energy_injection", False)),
            "raw_reward_sum_delta": _delta(b, a, "raw_reward_sum"),
            "reward_env_sum_delta": _delta(b, a, "reward_env_sum"),
            "training_reward_sum_delta": _delta(b, a, "training_reward_sum"),
            "done_count_delta": _delta(b, a, "done_count"),
            "first_done_step_delta": _delta(b, a, "first_done_step"),
            "next_state_step_l2_delta_mean_delta": _delta(b, a, "next_state_step_l2_delta_mean"),
            "next_state_rms_delta": _delta(b, a, "next_state_rms"),
            "action_value_absmean_delta": _delta(b, a, "action_value_absmean"),
            "action_saturation_delta": _delta(
                b,
                a,
                "raw_policy_action_unit_clip_saturation_fraction",
            ),
            "last_raw_reward_sum": _f(b.get("raw_reward_sum")),
            "last_reward_env_sum": _f(b.get("reward_env_sum")),
            "last_training_reward_sum": _f(b.get("training_reward_sum")),
            "last_done_count": _f(b.get("done_count")),
            "last_first_done_step": _f(b.get("first_done_step")),
            "last_next_state_step_l2_delta_mean": _f(b.get("next_state_step_l2_delta_mean")),
            "last_next_state_rms": _f(b.get("next_state_rms")),
            "last_raw_reward_std": _f(b.get("raw_reward_std")),
            "last_training_reward_std": _f(b.get("training_reward_std")),
            "last_action_per_dim_temporal_std_mean": _f(b.get("action_per_dim_temporal_std_mean")),
            "last_action_constant_dim_fraction_std_lt_0p05": _f(
                b.get("action_constant_dim_fraction_std_lt_0p05")
            ),
        }
        rows.append(row)
    meta = {
        "first_sidecar": str(first_path),
        "last_sidecar": str(last_path),
        "n_envs": int(len(rows)),
    }
    return rows, meta


def _group_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {
        "n": int(len(rows)),
        "raw_reward_sum_delta": _stats([r.get("raw_reward_sum_delta") for r in rows]),
        "reward_env_sum_delta": _stats([r.get("reward_env_sum_delta") for r in rows]),
        "training_reward_sum_delta": _stats([r.get("training_reward_sum_delta") for r in rows]),
        "done_count_delta": _stats([r.get("done_count_delta") for r in rows]),
        "first_done_step_delta": _stats([r.get("first_done_step_delta") for r in rows]),
        "next_state_step_l2_delta_mean_delta": _stats(
            [r.get("next_state_step_l2_delta_mean_delta") for r in rows]
        ),
        "next_state_rms_delta": _stats([r.get("next_state_rms_delta") for r in rows]),
        "action_value_absmean_delta": _stats([r.get("action_value_absmean_delta") for r in rows]),
        "action_saturation_delta": _stats([r.get("action_saturation_delta") for r in rows]),
        "last_raw_reward_std": _stats([r.get("last_raw_reward_std") for r in rows]),
        "last_training_reward_std": _stats([r.get("last_training_reward_std") for r in rows]),
        "last_done_count": _stats([r.get("last_done_count") for r in rows]),
        "last_next_state_step_l2_delta_mean": _stats(
            [r.get("last_next_state_step_l2_delta_mean") for r in rows]
        ),
        "last_next_state_rms": _stats([r.get("last_next_state_rms") for r in rows]),
        "last_action_per_dim_temporal_std_mean": _stats(
            [r.get("last_action_per_dim_temporal_std_mean") for r in rows]
        ),
        "last_action_constant_dim_fraction_std_lt_0p05": _stats(
            [r.get("last_action_constant_dim_fraction_std_lt_0p05") for r in rows]
        ),
    }
    for key in (
        "locomotion_witness",
        "low_energy_not_ctrl_only",
        "sign_or_phase_sensitive",
        "state_conditioned_energy_injection",
    ):
        out[f"{key}_rate"] = float(np.mean([bool(r.get(key)) for r in rows])) if rows else None
    return out


def _diversity_pass(summary: dict[str, Any], baseline_summary: dict[str, Any] | None) -> dict[str, Any]:
    def stat(path: tuple[str, str], default: float = 0.0) -> float:
        value = summary[path[0]].get(path[1])
        return float(default) if value is None else float(value)

    def base_stat(path: tuple[str, str], default: float = 0.0) -> float:
        if baseline_summary is None:
            return float(default)
        value = baseline_summary[path[0]].get(path[1])
        return float(default) if value is None else float(value)

    checks = {
        "raw_reward_std_nonzero": stat(("last_raw_reward_std", "q50")) > 1e-6,
        "training_reward_std_nonzero": stat(("last_training_reward_std", "q50")) > 1e-6,
        "state_step_nonzero": stat(("last_next_state_step_l2_delta_mean", "q50")) > 1e-6,
        "done_not_collapsed": stat(("last_done_count", "q90")) > stat(("last_done_count", "q10")),
        "action_temporal_std_nonzero": (
            stat(("last_action_per_dim_temporal_std_mean", "q50")) > 1e-6
        ),
        "constant_action_fraction_low": (
            stat(("last_action_constant_dim_fraction_std_lt_0p05", "q90"), default=1.0) < 0.5
        ),
    }
    if baseline_summary is not None:
        # Loose anti-regression checks. These are guards, not optimization
        # targets: candidate diversity can shift, but should not collapse.
        checks["raw_reward_std_not_half_baseline"] = (
            stat(("last_raw_reward_std", "q50")) >= 0.5 * base_stat(("last_raw_reward_std", "q50"))
        )
        checks["state_step_not_half_baseline"] = (
            stat(("last_next_state_step_l2_delta_mean", "q50"))
            >= 0.5 * base_stat(("last_next_state_step_l2_delta_mean", "q50"))
        )
        checks["done_spread_not_worse_than_half_baseline"] = (
            stat(("last_done_count", "q90"))
            - stat(("last_done_count", "q10"))
        ) >= 0.5 * (
            base_stat(("last_done_count", "q90"))
            - base_stat(("last_done_count", "q10"))
        )
    return {
        "checks": checks,
        "pass": bool(all(checks.values())),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run(args: argparse.Namespace) -> dict[str, Any]:
    configs = {
        "full": {
            "rule": "gym_full_obs_action_range",
            "candidate_run_dir": args.full_run_dir,
            "baseline_run_dir": args.baseline_full_run_dir,
        },
        "q90": {
            "rule": "gym_q90_obs_action_range",
            "candidate_run_dir": args.q90_run_dir,
            "baseline_run_dir": args.baseline_q90_run_dir,
        },
    }
    per_env: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "analysis_entry": "phase2_locomotion_guarded_smoke_summary",
        "contract": {
            "exploratory_only": True,
            "does_not_train": True,
            "does_not_modify_exact_scm": True,
            "does_not_modify_fit_model": True,
            "does_not_modify_ppo": True,
            "reads_semantic_sidecars_only": True,
        },
        "inputs": {
            "selected_csv": str(Path(args.selected_csv).expanduser().resolve()),
        },
        "scopes": {},
    }
    for short, cfg in configs.items():
        cand_rows, cand_meta = _per_env_rows(
            rule=cfg["rule"],
            run_dir=cfg["candidate_run_dir"],
            selected_csv=args.selected_csv,
            source_tag=f"candidate_{short}",
        )
        base_rows, base_meta = _per_env_rows(
            rule=cfg["rule"],
            run_dir=cfg["baseline_run_dir"],
            selected_csv=None,
            source_tag=f"baseline_{short}",
        )
        cand_summary = _group_summary(cand_rows)
        base_summary = _group_summary(base_rows)
        report["scopes"][short] = {
            "rule": cfg["rule"],
            "candidate_meta": cand_meta,
            "baseline_meta": base_meta,
            "candidate": cand_summary,
            "baseline": base_summary,
            "candidate_minus_baseline": {
                "delta_positive_fraction_monitor_diff": float(
                    cand_summary["raw_reward_sum_delta"].get("positive_fraction", 0.0)
                    - base_summary["raw_reward_sum_delta"].get("positive_fraction", 0.0)
                ),
                "raw_delta_mean": float(
                    cand_summary["raw_reward_sum_delta"].get("mean", 0.0)
                    - base_summary["raw_reward_sum_delta"].get("mean", 0.0)
                ),
                "locomotion_witness_rate": float(
                    cand_summary.get("locomotion_witness_rate") or 0.0
                ),
            },
            "diversity_guard": _diversity_pass(cand_summary, base_summary),
            "sufficiency_read": {
                "delta_positive_fraction_monitor": cand_summary["raw_reward_sum_delta"].get(
                    "positive_fraction"
                ),
                "delta_positive_fraction_monitor_floor": float(args.delta_positive_monitor_floor),
                "delta_positive_fraction_is_monitor_only": True,
                "raw_delta_mean_nonnegative": bool(
                    (cand_summary["raw_reward_sum_delta"].get("mean") or 0.0) >= 0.0
                ),
                "env_delta_mean_nonnegative": bool(
                    (cand_summary["reward_env_sum_delta"].get("mean") or 0.0) >= 0.0
                ),
            },
        }
        report["scopes"][short]["sufficiency_read"]["pass"] = bool(
            report["scopes"][short]["sufficiency_read"]["raw_delta_mean_nonnegative"]
            and report["scopes"][short]["sufficiency_read"]["env_delta_mean_nonnegative"]
            and report["scopes"][short]["diversity_guard"]["pass"]
        )
        per_env.extend(cand_rows)
    report["overall"] = {
        "candidate_sufficient_for_smoke": bool(
            all(scope["sufficiency_read"]["pass"] for scope in report["scopes"].values())
        ),
        "necessity_read": (
            "naive locomotion-only selection failed protected mode guards; this guarded selection "
            "is necessary as a protection wrapper if locomotion witness rate is raised by selection"
        ),
    }
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    per_env_csv = out_dir / "locomotion_guarded_smoke_per_env.csv"
    report_path = out_dir / "locomotion_guarded_smoke_summary.json"
    _write_csv(per_env_csv, per_env)
    report["outputs"] = {"per_env_csv": str(per_env_csv), "report_json": str(report_path)}
    report_path.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md = [
        "# Locomotion Guarded Smoke Summary",
        "",
        "Read-only sidecar summary. Candidate is not a milestone change.",
        "",
        "| scope | raw delta mean | delta+ monitor | env delta mean | done delta mean | diversity | delta smoke sufficient | baseline delta+ monitor |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- | ---: |",
    ]
    for short, scope in report["scopes"].items():
        cand = scope["candidate"]
        base = scope["baseline"]
        md.append(
            "| "
            + " | ".join(
                [
                    short,
                    f"{cand['raw_reward_sum_delta'].get('mean'):.6g}",
                    f"{cand['raw_reward_sum_delta'].get('positive_fraction'):.3f}",
                    f"{cand['reward_env_sum_delta'].get('mean'):.6g}",
                    f"{cand['done_count_delta'].get('mean'):.6g}",
                    "pass" if scope["diversity_guard"]["pass"] else "fail",
                    "pass" if scope["sufficiency_read"]["pass"] else "fail",
                    f"{base['raw_reward_sum_delta'].get('positive_fraction'):.3f}",
                ]
            )
            + " |"
        )
    md.extend(
        [
            "",
            "## Read",
            "",
            f"- overall smoke sufficient: `{report['overall']['candidate_sufficient_for_smoke']}`",
            f"- necessity read: {report['overall']['necessity_read']}",
            "",
            "## Files",
            "",
            f"- JSON: `{report_path}`",
            f"- per-env CSV: `{per_env_csv}`",
        ]
    )
    md_path = out_dir / "locomotion_guarded_smoke_summary.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path), "summary": str(md_path)}, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--selected-csv", default=DEFAULT_SELECTED_CSV)
    parser.add_argument("--full-run-dir", default=DEFAULT_FULL_RUN_DIR)
    parser.add_argument("--q90-run-dir", default=DEFAULT_Q90_RUN_DIR)
    parser.add_argument("--baseline-full-run-dir", default=DEFAULT_BASELINE_FULL_RUN_DIR)
    parser.add_argument("--baseline-q90-run-dir", default=DEFAULT_BASELINE_Q90_RUN_DIR)
    parser.add_argument(
        "--delta-positive-monitor-floor",
        dest="delta_positive_monitor_floor",
        type=float,
        default=0.70,
        help="Monitoring-only floor for delta>0 fraction; it does not gate smoke sufficiency.",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
