import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_OUTPUT_DIR = (
    "/home/chen/RLPFN/artifacts/"
    "phase2_gpu0_no_tail_root_cause_report_0429"
)
DEFAULT_POPULATION_FEATURES = (
    "/home/chen/RLPFN/artifacts/"
    "phase2_gpu0_population_trainability_feature_analysis_0429/"
    "population_gpu0_trainability_features.csv"
)
DEFAULT_U20_SUMMARY = (
    "/home/chen/RLPFN/artifacts/"
    "phase2_gpu0_trainability_gate_validation_groups_0429/"
    "u20_gate_validation_summary.json"
)
DEFAULT_CONTEXT_DIR = (
    "/home/chen/RLPFN/artifacts/"
    "phase2_context_sensitivity_30093_0429"
)


def _finite(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _rule_metrics(mask: pd.Series, target: pd.Series) -> dict[str, Any]:
    pred = mask.astype(bool).to_numpy()
    y = target.astype(bool).to_numpy()
    tp = int(np.logical_and(pred, y).sum())
    fp = int(np.logical_and(pred, ~y).sum())
    fn = int(np.logical_and(~pred, y).sum())
    tn = int(np.logical_and(~pred, ~y).sum())
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "predicted_count": int(pred.sum()),
        "target_count": int(y.sum()),
    }


def _feature_gaps(df: pd.DataFrame, label: str) -> list[dict[str, Any]]:
    fields = [
        c
        for c in df.columns
        if c.startswith("param_")
        or c
        in {
            "reward_spread_mean",
            "top_tail_gap_mean",
            "relative_top_tail_gap_mean",
            "action_mode_env_sum_range",
        }
    ]
    out: list[dict[str, Any]] = []
    y = df[label].astype(bool)
    for field in fields:
        values = pd.to_numeric(df[field], errors="coerce").replace([np.inf, -np.inf], np.nan)
        risk = values[y].dropna()
        other = values[~y].dropna()
        if len(risk) < 5 or len(other) < 5:
            continue
        iqr = values.quantile(0.75) - values.quantile(0.25)
        if not np.isfinite(iqr) or abs(float(iqr)) <= 1e-12:
            continue
        gap = (risk.median() - other.median()) / iqr
        out.append(
            {
                "label": label,
                "feature": field,
                "gap_iqr": float(gap),
                "abs_gap_iqr": float(abs(gap)),
                "nonrisk_median": float(other.median()),
                "risk_median": float(risk.median()),
                "risk_q10": float(risk.quantile(0.10)),
                "risk_q90": float(risk.quantile(0.90)),
            }
        )
    out.sort(key=lambda row: row["abs_gap_iqr"], reverse=True)
    return out


def _context_seed_30093(context_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    runs = {
        "gpu0_old_as_is": context_dir / "old_as_is_gpu0_run",
        "gpu1_old_as_is_repeat": context_dir / "old_as_is_gpu1_repeat_run",
    }
    for label, run_dir in runs.items():
        sidecar = run_dir / "semantic_sidecars" / "prior_update_000000_envs.jsonl"
        for row in _load_jsonl(sidecar):
            if int(row.get("env_group_env_seed", -1)) != 30093:
                continue
            rows.append(
                {
                    "context": label,
                    "env_idx": row.get("env_idx"),
                    "seed": row.get("env_group_env_seed"),
                    "reward_sum": row.get("reward_sum"),
                    "reward_env_sum": row.get("reward_env_sum"),
                    "reward_q50": row.get("reward_q50"),
                    "reward_q90": row.get("reward_q90"),
                    "reward_env_positive_fraction": row.get("reward_env_positive_fraction"),
                    "obs_value_std": row.get("obs_value_std"),
                    "next_state_step_l2_delta_to_rms_ratio": row.get(
                        "next_state_step_l2_delta_to_rms_ratio"
                    ),
                    "corr_action_norm_reward": row.get("corr_action_norm_reward"),
                }
            )
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--population-features", default=DEFAULT_POPULATION_FEATURES)
    parser.add_argument("--u20-summary", default=DEFAULT_U20_SUMMARY)
    parser.add_argument("--context-dir", default=DEFAULT_CONTEXT_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.population_features)
    # Two compact rules mined from GPU0 population evidence. They are not generator defaults.
    df["rule_amp_noise_obs"] = (
        (df["param_amp_geom_ratio"] > 0.66)
        & (df["param_reward_noise_gain"] <= 0.12)
        & (df["param_obs_std"] <= 0.99)
    )
    df["rule_obs_geometry_dim"] = (
        (df["param_obs_ratio"] > 0.59)
        & (df["param_obs_top"] <= 0.75)
        & (df["param_state_dim"] > 220)
    )
    df["rule_combined_soft"] = df["rule_amp_noise_obs"] | df["rule_obs_geometry_dim"]
    df["rule_combined_strict"] = df["rule_amp_noise_obs"] & df["rule_obs_geometry_dim"]

    targets = ["structural_no_tail_risk", "low_tail_risk", "group_drag_risk_proxy"]
    rule_rows: list[dict[str, Any]] = []
    for rule in [
        "rule_amp_noise_obs",
        "rule_obs_geometry_dim",
        "rule_combined_soft",
        "rule_combined_strict",
    ]:
        for target in targets:
            row = {"rule": rule, "target": target}
            row.update(_rule_metrics(df[rule], df[target]))
            rule_rows.append(row)

    gaps: list[dict[str, Any]] = []
    for target in targets:
        gaps.extend(_feature_gaps(df, target)[:20])

    context_rows = _context_seed_30093(Path(args.context_dir))
    u20_summary = json.loads(Path(args.u20_summary).read_text(encoding="utf-8"))

    worst = df.sort_values(
        ["structural_no_tail_risk", "low_positive_tail_fraction", "env_sum_mean"],
        ascending=[False, False, True],
    ).head(12)
    best = df.sort_values(
        ["stable_easy_positive_proxy", "bad_q90_fraction", "pos_frac_mean", "env_sum_mean"],
        ascending=[False, True, False, False],
    ).head(12)

    rule_df = pd.DataFrame(rule_rows)
    gap_df = pd.DataFrame(gaps)
    context_df = pd.DataFrame(context_rows)
    worst_df = worst[
        [
            "seed",
            "bad_q90_fraction",
            "low_positive_tail_fraction",
            "pos_frac_mean",
            "env_sum_mean",
            "param_reward_std",
            "param_return_std",
            "top_tail_gap_mean",
            "reward_spread_mean",
            "param_amp_geom_ratio",
            "param_obs_ratio",
            "param_obs_std",
            "param_obs_top",
            "param_reward_state_gain",
            "param_reward_action_gain",
            "param_reward_noise_gain",
            "param_state_dim",
            "param_noise_std",
        ]
    ]
    best_df = best[worst_df.columns]

    rule_csv = out_dir / "rule_metrics.csv"
    gap_csv = out_dir / "feature_gap_top20.csv"
    context_csv = out_dir / "context_seed30093_gpu0_gpu1.csv"
    worst_csv = out_dir / "worst_no_tail_examples.csv"
    best_csv = out_dir / "best_positive_tail_examples.csv"
    rule_df.to_csv(rule_csv, index=False)
    gap_df.to_csv(gap_csv, index=False)
    context_df.to_csv(context_csv, index=False)
    worst_df.to_csv(worst_csv, index=False)
    best_df.to_csv(best_csv, index=False)

    report = {
        "contract": (
            "GPU0-only root-cause convergence report. Rules are diagnostic candidates, "
            "not generator-default changes and not single-metric gates."
        ),
        "input_population_features": str(Path(args.population_features).resolve()),
        "input_u20_summary": str(Path(args.u20_summary).resolve()),
        "rule_metrics_csv": str(rule_csv.resolve()),
        "feature_gap_top20_csv": str(gap_csv.resolve()),
        "context_seed30093_gpu0_gpu1_csv": str(context_csv.resolve()),
        "worst_no_tail_examples_csv": str(worst_csv.resolve()),
        "best_positive_tail_examples_csv": str(best_csv.resolve()),
        "counts": {
            "records": int(len(df)),
            "structural_no_tail_risk": int(df["structural_no_tail_risk"].sum()),
            "low_tail_risk": int(df["low_tail_risk"].sum()),
            "group_drag_risk_proxy": int(df["group_drag_risk_proxy"].sum()),
            "stable_easy_positive_proxy": int(df["stable_easy_positive_proxy"].sum()),
        },
        "top_feature_gaps": gap_df.head(18).to_dict(orient="records"),
        "rule_metrics": rule_rows,
        "u20_validation_summary": u20_summary,
        "context_seed_30093": context_rows,
        "current_root_cause_hypothesis": [
            (
                "Bad prior envs are not defined by negative absolute return; they are "
                "defined by low reachable positive tail under canonical policy/action "
                "families plus high reward/return volatility."
            ),
            (
                "The strongest static structure signals are high amplitude geometry "
                "(reward/return/obs scale relative to Gym gates), high obs geometry ratio, "
                "large top-tail/spread, state-dominated reward, and weak action/noise "
                "escape."
            ),
            (
                "The group-level failure is objective concentration: a few no-tail envs "
                "dominate absolute reward mass and yield no short-run recovery under the "
                "canonical PPO runner."
            ),
            (
                "Exact-SCM numerical/context sensitivity means gates must be run on GPU0 "
                "with the target fixed group and runner; static frozen_h metadata alone "
                "is not sufficient."
            ),
        ],
    }
    (out_dir / "root_cause_report.json").write_text(
        json.dumps(_json_safe(report), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(_json_safe(report), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
