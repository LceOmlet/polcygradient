import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_DIR = "/home/chen/RLPFN/artifacts/phase2_prior_gym_hard_gate_report_0429"
DEFAULT_CAUSAL_ROWS = (
    "/home/chen/RLPFN/artifacts/"
    "phase2_prior_gym_action_causal_sensitivity_all_gain07_terminal_ablation_0428/rows.jsonl"
)
DEFAULT_CAUSAL_SUMMARY = (
    "/home/chen/RLPFN/artifacts/"
    "phase2_prior_gym_action_causal_sensitivity_all_gain07_terminal_ablation_0428/summary.json"
)
DEFAULT_NOISE_ROWS = (
    "/home/chen/RLPFN/artifacts/"
    "phase2_prior_knob_sensitivity_scan_noise_all_gain07_ref32_0428/rows.jsonl"
)
DEFAULT_STABILITY_DIR = (
    "/home/chen/RLPFN/artifacts/phase2_prior_long_rollout_stability_2048_noterm_0429"
)
DEFAULT_GYM_PROGRESS = (
    "/home/chen/RLPFN/artifacts/"
    "phase2_gym_pack_semprobe_multigym_fixedgroup_n1024_u200_0428/gym_progress.jsonl"
)
DEFAULT_GYM_OBSNORM_PROGRESS = (
    "/home/chen/RLPFN/artifacts/"
    "phase2_gym_pack_semprobe_multigym_fixedgroup_n1024_u200_obsnorm_0429/gym_progress.jsonl"
)
DEFAULT_PRIOR_PROGRESS = (
    "/home/chen/RLPFN/artifacts/"
    "phase2_prior_pack_semprobe_sampledgain_fixedgroup_n1024_u200_0428/prior_progress.jsonl"
)
DEFAULT_PRIOR_OBSNORM_PROGRESS = (
    "/home/chen/RLPFN/artifacts/"
    "phase2_prior_pack_semprobe_sampledgain_fixedgroup_n1024_u200_obsnorm_0429/prior_progress.jsonl"
)
DEFAULT_CONTROLLED_DEEP_ROWS = ""


def _finite(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _get(data: dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = data
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def _stats(values: list[Any]) -> dict[str, Any]:
    vals = sorted(v for v in (_finite(v) for v in values) if v is not None)
    if not vals:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "q10": None,
            "q25": None,
            "q50": None,
            "q75": None,
            "q90": None,
            "max": None,
        }
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)

    def q(frac: float) -> float:
        if len(vals) == 1:
            return vals[0]
        pos = frac * (len(vals) - 1)
        lo = math.floor(pos)
        hi = math.ceil(pos)
        if lo == hi:
            return vals[lo]
        return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)

    return {
        "count": len(vals),
        "mean": mean,
        "std": math.sqrt(var),
        "min": vals[0],
        "q10": q(0.10),
        "q25": q(0.25),
        "q50": q(0.50),
        "q75": q(0.75),
        "q90": q(0.90),
        "max": vals[-1],
    }


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    rows: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


def _gym_thresholds(summary: dict[str, Any]) -> dict[str, Any]:
    refs = summary.get("gym_references", {})
    metric_paths = {
        "return_std": "discounted_effective_return.std",
        "reward_action_sensitivity": "per_action_reward_sensitivity.q50",
        "state_action_sensitivity": "per_action_state_sensitivity.q50",
        "reward_action_effdim": "per_action_reward_sensitivity.effective_dim_fraction",
        "state_action_effdim": "per_action_state_sensitivity.effective_dim_fraction",
        "obs_std": "obs_value_std_all",
        "obs_rank": "obs_cov.effective_rank_fraction",
        "obs_top_eig": "obs_cov.top_eigen_share",
        "reward_std": "reward_effective.std",
        "drift_mean": "step_obs_drift_l2.mean",
    }
    gym_values: dict[str, dict[str, float]] = {}
    thresholds: dict[str, Any] = {}
    for metric, path in metric_paths.items():
        values = {}
        for env_id, ref in refs.items():
            val = _finite(_get(ref, path))
            if val is not None:
                values[str(env_id)] = val
        gym_values[metric] = values
        if values:
            if metric == "obs_top_eig":
                thresholds[f"{metric}_max"] = max(values.values())
            else:
                thresholds[f"{metric}_min"] = min(values.values())
                thresholds[f"{metric}_q10"] = sorted(values.values())[max(0, int(0.10 * (len(values) - 1)))]
    return {"gym_values": gym_values, "thresholds": thresholds}


def _case_summary(rows: list[dict[str, Any]], metrics: dict[str, str]) -> dict[str, Any]:
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_case[str(row.get("case", "unknown"))].append(row)
    out: dict[str, Any] = {}
    for case, case_rows in sorted(by_case.items()):
        cur: dict[str, Any] = {"row_count": len(case_rows), "metrics": {}}
        for name, path in metrics.items():
            cur["metrics"][name] = _stats([_get(row, path) for row in case_rows])
        out[case] = cur
    return out


def _pass_fraction(rows: list[dict[str, Any]], predicate) -> dict[str, Any]:
    total = len(rows)
    if total <= 0:
        return {"count": 0, "total": 0, "fraction": None}
    count = sum(1 for row in rows if predicate(row))
    return {"count": count, "total": total, "fraction": count / total}


def _raw_env_gate(rows: list[dict[str, Any]], thresholds: dict[str, Any]) -> dict[str, Any]:
    reward_min = thresholds.get("reward_action_sensitivity_min")
    state_min = thresholds.get("state_action_sensitivity_min")
    return_min = thresholds.get("return_std_min")
    obs_min = thresholds.get("obs_std_min")
    obs_rank_min = thresholds.get("obs_rank_min")
    obs_top_max = thresholds.get("obs_top_eig_max")
    reward_std_min = thresholds.get("reward_std_min")

    def f(path: str, row: dict[str, Any]) -> float | None:
        return _finite(_get(row, path))

    gates = {
        "action_reward_sensitivity_reaches_gym_lower_bound": _pass_fraction(
            rows,
            lambda r: reward_min is not None
            and (v := f("per_action_reward_sensitivity.q50", r)) is not None
            and v >= reward_min,
        ),
        "action_state_sensitivity_reaches_gym_lower_bound": _pass_fraction(
            rows,
            lambda r: state_min is not None
            and (v := f("per_action_state_sensitivity.q50", r)) is not None
            and v >= state_min,
        ),
        "return_scale_reaches_gym_lower_bound": _pass_fraction(
            rows,
            lambda r: return_min is not None
            and (v := f("discounted_effective_return.std", r)) is not None
            and v >= return_min,
        ),
        "reward_scale_reaches_gym_lower_bound": _pass_fraction(
            rows,
            lambda r: reward_std_min is not None
            and (v := f("reward_effective.std", r)) is not None
            and v >= reward_std_min,
        ),
        "obs_raw_scale_reaches_gym_lower_bound": _pass_fraction(
            rows,
            lambda r: obs_min is not None
            and (v := f("obs_value_std_all", r)) is not None
            and v >= obs_min,
        ),
        "obs_covariance_not_low_rank_by_gym_envelope": _pass_fraction(
            rows,
            lambda r: obs_rank_min is not None
            and obs_top_max is not None
            and (rank := f("obs_cov.effective_rank_fraction", r)) is not None
            and (top := f("obs_cov.top_eigen_share", r)) is not None
            and rank >= obs_rank_min
            and top <= obs_top_max,
        ),
        "per_action_reward_sensitivity_not_collapsed": _pass_fraction(
            rows,
            lambda r: (eff := f("per_action_reward_sensitivity.effective_dim_fraction", r)) is not None
            and eff >= 0.50
            and (dead := f("per_action_reward_sensitivity.dead_dim_fraction_abs", r)) is not None
            and dead <= 0.10,
        ),
        "per_action_state_sensitivity_not_collapsed": _pass_fraction(
            rows,
            lambda r: (eff := f("per_action_state_sensitivity.effective_dim_fraction", r)) is not None
            and eff >= 0.50
            and (dead := f("per_action_state_sensitivity.dead_dim_fraction_abs", r)) is not None
            and dead <= 0.10,
        ),
        "joint_core_raw_env_gate": _pass_fraction(
            rows,
            lambda r: reward_min is not None
            and state_min is not None
            and return_min is not None
            and obs_min is not None
            and obs_rank_min is not None
            and obs_top_max is not None
            and reward_std_min is not None
            and (f("per_action_reward_sensitivity.q50", r) or -math.inf) >= reward_min
            and (f("per_action_state_sensitivity.q50", r) or -math.inf) >= state_min
            and (f("discounted_effective_return.std", r) or -math.inf) >= return_min
            and (f("reward_effective.std", r) or -math.inf) >= reward_std_min
            and (f("obs_value_std_all", r) or -math.inf) >= obs_min
            and (f("obs_cov.effective_rank_fraction", r) or -math.inf) >= obs_rank_min
            and (f("obs_cov.top_eigen_share", r) or math.inf) <= obs_top_max,
        ),
    }
    return gates


def _causal_gate(rows: list[dict[str, Any]], thresholds: dict[str, Any]) -> dict[str, Any]:
    baseline = [row for row in rows if str(row.get("case")) == "baseline"]
    return _raw_env_gate(baseline, thresholds)


def _paired_control_summary(
    rows: list[dict[str, Any]],
    metrics: dict[str, str],
    *,
    baseline_case: str,
) -> dict[str, Any]:
    by_seed: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_seed[str(row.get("frozen_h_seed"))][str(row.get("case", "unknown"))] = row
    paired: dict[str, Any] = {}
    for case in sorted({str(row.get("case", "unknown")) for row in rows if str(row.get("case")) != baseline_case}):
        pairs = [cases for cases in by_seed.values() if baseline_case in cases and case in cases]
        payload: dict[str, Any] = {"paired_count": len(pairs), "metrics": {}}
        for name, path in metrics.items():
            deltas = []
            ratios = []
            for pair in pairs:
                base = _finite(_get(pair[baseline_case], path))
                cur = _finite(_get(pair[case], path))
                if base is None or cur is None:
                    continue
                deltas.append(cur - base)
                if abs(base) > 1e-12:
                    ratios.append(cur / base)
            payload["metrics"][name] = {"delta": _stats(deltas), "ratio": _stats(ratios)}
        paired[case] = payload
    return paired


def _controlled_deep_report(
    rows: list[dict[str, Any]],
    metrics: dict[str, str],
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_case[str(row.get("case", "unknown"))].append(row)
    baseline_case = "rms1" if "rms1" in by_case else ("baseline" if "baseline" in by_case else "")
    return {
        "path_present": bool(rows),
        "row_count": len(rows),
        "case_summary": _case_summary(rows, metrics),
        "case_gates": {case: _raw_env_gate(case_rows, thresholds) for case, case_rows in sorted(by_case.items())},
        "paired_vs_baseline_case": baseline_case or None,
        "paired_vs_baseline": (
            _paired_control_summary(rows, metrics, baseline_case=baseline_case) if baseline_case else {}
        ),
    }


def _noise_control_gate(rows: list[dict[str, Any]], baseline_case: str = "baseline") -> dict[str, Any]:
    metrics = {
        "return_std": "discounted_effective_return_std",
        "reward_action_sensitivity": "mean_abs_reward_effective_t1_action_sensitivity_per_action_unit",
        "state_action_sensitivity": "mean_l2_s_t1_action_sensitivity_per_action_unit",
        "state_action_to_drift": "state_action_sensitivity_to_step1_drift_ratio",
        "G2_identity": "G2_bias_corrected_identity",
        "G3_identity": "G3_bias_corrected_identity",
        "G4_identity": "G4_bias_corrected_identity",
        "G5_identity": "G5_bias_corrected_identity",
        "source_obs_std": "source_anchor_obs_value_std",
    }
    by_case = _case_summary(rows, metrics)
    by_seed: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_seed[str(row.get("frozen_h_seed"))][str(row.get("case"))] = row
    paired: dict[str, Any] = {}
    for case in sorted({str(row.get("case")) for row in rows if str(row.get("case")) != baseline_case}):
        pairs = [cases for cases in by_seed.values() if baseline_case in cases and case in cases]
        deltas: dict[str, Any] = {"paired_count": len(pairs), "metrics": {}}
        for name, path in metrics.items():
            vals = []
            ratios = []
            for pair in pairs:
                a = _finite(_get(pair[baseline_case], path))
                b = _finite(_get(pair[case], path))
                if a is None or b is None:
                    continue
                vals.append(b - a)
                if abs(a) > 1e-12:
                    ratios.append(b / a)
            deltas["metrics"][name] = {"delta": _stats(vals), "ratio": _stats(ratios)}
        paired[case] = deltas
    return {"case_summary": by_case, "paired_vs_baseline": paired}


def _stability_report(stability_dir: str | Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for path in sorted(Path(stability_dir).glob("*.jsonl")):
        rows = _load_jsonl(path)
        cases = []
        for row in rows:
            last = row.get("checkpoints", [])[-1] if row.get("checkpoints") else {}
            cases.append(
                {
                    "case": row.get("case"),
                    "finite_all": bool(row.get("finite_all")),
                    "n_steps": row.get("n_steps"),
                    "rollout_count": row.get("rollout_count"),
                    "max_obs_rms_seen": _finite(row.get("max_obs_rms_seen")),
                    "max_state_rms_seen": _finite(row.get("max_state_rms_seen")),
                    "max_state_abs_seen": _finite(row.get("max_state_abs_seen")),
                    "last_step": last.get("step"),
                    "last_obs_rms_mean": _get(last, "obs_rms.mean"),
                    "last_state_rms_mean": _get(last, "state_rms.mean"),
                    "sampled_env": row.get("sampled_env", {}),
                    "overrides": row.get("overrides", {}),
                }
            )
        out[path.name] = cases
    return out


def _progress_summary(path: str | Path) -> dict[str, Any]:
    rows = _load_jsonl(path)
    if not rows:
        return {"path": str(path), "exists": Path(path).exists(), "row_count": 0}

    def log(row: dict[str, Any], key: str) -> Any:
        return _get(row, f"update.logger.{key}")

    first, last = rows[0], rows[-1]
    metrics = [
        "train/legacy_pre_rollout_raw_return_mean",
        "train/legacy_pre_rollout_raw_return_std",
        "train/legacy_pre_rollout_input_observation_sample_std",
        "train/legacy_pre_rollout_input_reward_sample_std",
        "train/legacy_pre_rollout_normalized_return_std",
        "train/explained_variance",
        "train/explained_variance_normalized",
    ]
    out = {
        "path": str(path),
        "exists": True,
        "row_count": len(rows),
        "first_update": first.get("update_idx"),
        "last_update": last.get("update_idx"),
        "metrics": {},
    }
    for metric in metrics:
        vals = [log(row, metric) for row in rows]
        first_v = _finite(vals[0])
        last_v = _finite(vals[-1])
        out["metrics"][metric] = {
            "first": first_v,
            "last": last_v,
            "delta": (last_v - first_v) if first_v is not None and last_v is not None else None,
            "series": _stats(vals),
        }
    obs_norm = _get(last, "collector.obs_normalization")
    if isinstance(obs_norm, dict):
        out["obs_normalization_last"] = {
            "enabled": obs_norm.get("enabled"),
            "raw_active_obs_std": _get(obs_norm, "raw_active_obs.std"),
            "normalized_active_obs_std": _get(obs_norm, "normalized_active_obs.std"),
            "normalized_active_obs_absmax": _get(obs_norm, "normalized_active_obs.absmax"),
            "rms_count_min": obs_norm.get("rms_count_min"),
            "rms_count_max": obs_norm.get("rms_count_max"),
        }
    sidecar = _get(last, "semantic_probe_sidecar_path")
    if isinstance(sidecar, str) and Path(sidecar).exists():
        env_rows = _load_jsonl(sidecar)
        out["sidecar_last"] = {
            "path": sidecar,
            "row_count": len(env_rows),
            "obs_value_std": _stats([row.get("obs_value_std") for row in env_rows]),
            "reward_std": _stats([row.get("reward_std") for row in env_rows]),
            "next_state_step_l2_delta_to_rms_ratio": _stats(
                [row.get("next_state_step_l2_delta_to_rms_ratio") for row in env_rows]
            ),
            "action_per_dim_temporal_std_mean": _stats(
                [row.get("action_per_dim_temporal_std_mean") for row in env_rows]
            ),
            "action_constant_dim_fraction_std_lt_0p10": _stats(
                [row.get("action_constant_dim_fraction_std_lt_0p10") for row in env_rows]
            ),
        }
    return out


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    causal_summary = _load_json(args.causal_summary)
    gym = _gym_thresholds(causal_summary)
    causal_rows = _load_jsonl(args.causal_rows)
    noise_rows = _load_jsonl(args.noise_rows)
    controlled_deep_rows = _load_jsonl(args.controlled_deep_rows) if str(args.controlled_deep_rows) else []
    causal_metrics = {
        "return_std": "discounted_effective_return.std",
        "reward_action_sensitivity": "per_action_reward_sensitivity.q50",
        "state_action_sensitivity": "per_action_state_sensitivity.q50",
        "reward_action_effdim": "per_action_reward_sensitivity.effective_dim_fraction",
        "state_action_effdim": "per_action_state_sensitivity.effective_dim_fraction",
        "reward_action_dead_frac": "per_action_reward_sensitivity.dead_dim_fraction_abs",
        "state_action_dead_frac": "per_action_state_sensitivity.dead_dim_fraction_abs",
        "obs_std": "obs_value_std_all",
        "obs_rank": "obs_cov.effective_rank_fraction",
        "obs_top_eig": "obs_cov.top_eigen_share",
        "reward_std": "reward_effective.std",
        "drift_mean": "step_obs_drift_l2.mean",
        "G2_identity": "horizon_2_bias_corrected_identity_score",
        "G5_identity": "horizon_5_bias_corrected_identity_score",
    }
    report = {
        "audit_entry": "phase2_prior_gym_hard_gate_report",
        "principle": (
            "Do not stack uncontrolled fixes. Treat SB3 obs/reward normalization as input/target-scale protection, "
            "and keep raw prior dynamics gates separate."
        ),
        "gym_reference": gym,
        "causal_population_summary": _case_summary(causal_rows, causal_metrics),
        "causal_population_gates": _causal_gate(causal_rows, gym["thresholds"]),
        "controlled_deep_scan": _controlled_deep_report(controlled_deep_rows, causal_metrics, gym["thresholds"]),
        "noise_control_summary": _noise_control_gate(noise_rows),
        "long_rollout_2048_no_terminal": _stability_report(args.stability_dir),
        "fixed_group_training": {
            "gym_raw": _progress_summary(args.gym_progress),
            "gym_obsnorm": _progress_summary(args.gym_obsnorm_progress),
            "prior_raw": _progress_summary(args.prior_progress),
            "prior_obsnorm": _progress_summary(args.prior_obsnorm_progress),
        },
        "gate_interpretation": {
            "finite_2048_no_terminal": (
                "Only state_full_rms-enabled cases are acceptable. no-rms cases are a hard fail and must not be used as base."
            ),
            "obs_scale": (
                "Model-input obs scale should be judged after masked SB3 obs normalization. Raw obs scale remains a prior-structure diagnostic."
            ),
            "action_state_and_reward_sensitivity": (
                "Must be judged by paired finite-difference causal probes, not only rollout correlations."
            ),
            "per_action_sensitivity": (
                "Use effective_dim_fraction and dead_dim_fraction; a large mean can still be bad if concentrated in few action dims."
            ),
            "fixed_group_training": (
                "A pack passes only if fixed-group training shows measurable improvement under the same canonical pack/buffer/update path."
            ),
        },
    }
    return _json_safe(report)


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Phase2 Prior-vs-Gym Hard Gate Report",
        "",
        "This report separates model-input scale protection from raw prior dynamics quality.",
        "",
        "## Key Gates",
    ]
    gates = report.get("causal_population_gates", {})
    for name, value in gates.items():
        lines.append(
            f"- `{name}`: {value.get('count')}/{value.get('total')} "
            f"({value.get('fraction')})"
        )
    lines.extend(["", "## Fixed-Group Training"])
    fixed = report.get("fixed_group_training", {})
    for name, value in fixed.items():
        rows = value.get("row_count")
        last = value.get("last_update")
        obs = value.get("metrics", {}).get("train/legacy_pre_rollout_input_observation_sample_std", {})
        ret = value.get("metrics", {}).get("train/legacy_pre_rollout_raw_return_mean", {})
        lines.append(
            f"- `{name}`: rows={rows}, last_update={last}, "
            f"obs_std first->last={obs.get('first')}->{obs.get('last')}, "
            f"raw_return_mean first->last={ret.get('first')}->{ret.get('last')}"
        )
    lines.extend(["", "## Non-Negotiable Interpretation"])
    for key, text in report.get("gate_interpretation", {}).items():
        lines.append(f"- `{key}`: {text}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--causal-rows", default=DEFAULT_CAUSAL_ROWS)
    parser.add_argument("--causal-summary", default=DEFAULT_CAUSAL_SUMMARY)
    parser.add_argument("--noise-rows", default=DEFAULT_NOISE_ROWS)
    parser.add_argument("--stability-dir", default=DEFAULT_STABILITY_DIR)
    parser.add_argument("--gym-progress", default=DEFAULT_GYM_PROGRESS)
    parser.add_argument("--gym-obsnorm-progress", default=DEFAULT_GYM_OBSNORM_PROGRESS)
    parser.add_argument("--prior-progress", default=DEFAULT_PRIOR_PROGRESS)
    parser.add_argument("--prior-obsnorm-progress", default=DEFAULT_PRIOR_OBSNORM_PROGRESS)
    parser.add_argument("--controlled-deep-rows", default=DEFAULT_CONTROLLED_DEEP_ROWS)
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(args)
    report_path = out_dir / "hard_gate_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_markdown(report, out_dir / "hard_gate_report.md")
    print(json.dumps({"hard_pass": True, "report": str(report_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
