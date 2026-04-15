import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from ticl.analysis.phase3_cross_suite_regime_summary import DEFAULT_SUITES


DIST_KEYS = tuple(f"post_reset_tail_dist_{idx}" for idx in range(1, 5))


def _load_json(path: str | None) -> dict[str, Any] | None:
    if path is None:
        return None
    payload_path = Path(path).expanduser().resolve()
    if not payload_path.exists():
        return None
    return json.loads(payload_path.read_text())


def _corr(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys):
        raise ValueError("Correlation inputs must have matching lengths.")
    if len(xs) <= 1:
        return 0.0
    x = np.asarray(xs, dtype=np.float32)
    y = np.asarray(ys, dtype=np.float32)
    finite = np.isfinite(x) & np.isfinite(y)
    if int(finite.sum()) <= 1:
        return 0.0
    x = x[finite]
    y = y[finite]
    x = x - float(x.mean())
    y = y - float(y.mean())
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(x, y) / denom)


def _suite_regime(zero_payload: dict[str, Any], ppo_payload: dict[str, Any]) -> tuple[str, float]:
    ppo_minus_zero_suffix = float(
        float(ppo_payload["heldout_suite"]["metrics"]["suffix_return_mean"])
        - float(zero_payload["heldout_suite"]["metrics"]["suffix_return_mean"])
    )
    regime = "positive" if ppo_minus_zero_suffix > 0.0 else "nonpositive"
    return regime, ppo_minus_zero_suffix


def _weighted_mean(rows: list[dict[str, Any]], *, mass_key: str, value_key: str) -> float:
    masses = [float(row[mass_key]) for row in rows]
    values = [float(row[value_key]) for row in rows]
    total_mass = float(sum(masses))
    if total_mass <= 0.0:
        return 0.0
    return float(sum(float(m) * float(v) for m, v in zip(masses, values)) / total_mass)


def _group_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    masses = [float(row["dist1_to_4_mass"]) for row in rows]
    pre_gap = [float(row["pre_vs_zero_suffix_gap"]) for row in rows]
    reset_counts = [float(row["objective_terminal_reset_count"]) for row in rows]
    nonzero_rows = [row for row in rows if float(row["dist1_to_4_mass"]) > 0.0]
    return {
        "env_count": int(len(rows)),
        "nonzero_mass_env_count": int(len(nonzero_rows)),
        "nonzero_mass_env_share": float(len(nonzero_rows) / len(rows)) if rows else 0.0,
        "total_dist1_to_4_mass": float(sum(masses)),
        "mean_pre_vs_zero_suffix_gap": float(sum(pre_gap) / len(pre_gap)) if pre_gap else 0.0,
        "dist1_to_4_mass_weighted_pre_gap": _weighted_mean(rows, mass_key="dist1_to_4_mass", value_key="pre_vs_zero_suffix_gap"),
        "corr_dist1_to_4_mass_vs_pre_gap": _corr(masses, pre_gap),
        "corr_dist1_to_4_mass_vs_reset_count": _corr(masses, reset_counts),
    }


def _covariance_numerator(rows: list[dict[str, Any]], *, x_key: str, y_key: str, x_mean: float, y_mean: float) -> float:
    total = 0.0
    for row in rows:
        x = float(row[x_key])
        y = float(row[y_key])
        if np.isfinite(x) and np.isfinite(y):
            total += (x - float(x_mean)) * (y - float(y_mean))
    return float(total)


def build_regime_mixed_update_probe(suite_specs: list[dict[str, Any]]) -> dict[str, Any]:
    suite_rows: list[dict[str, Any]] = []
    suite_summaries: list[dict[str, Any]] = []
    skipped_suites: list[dict[str, Any]] = []

    for spec in suite_specs:
        zero_payload = _load_json(spec.get("zero_json"))
        ppo_payload = _load_json(spec.get("ppo_json"))
        distance_payload = _load_json(spec.get("distance_probe_json"))
        if zero_payload is None or ppo_payload is None or distance_payload is None:
            skipped_suites.append(
                {
                    "suite_name": str(spec.get("suite_name", "unknown")),
                    "missing_zero_json": zero_payload is None,
                    "missing_ppo_json": ppo_payload is None,
                    "missing_distance_probe_json": distance_payload is None,
                }
            )
            continue

        regime, ppo_minus_zero_suffix = _suite_regime(zero_payload, ppo_payload)
        rows = []
        for row in distance_payload["env_rows"]:
            dist_mass = float(sum(float(row[key]) for key in DIST_KEYS))
            rows.append(
                {
                    "suite_name": str(spec["suite_name"]),
                    "suite_regime": regime,
                    "ppo_minus_zero_suffix": float(ppo_minus_zero_suffix),
                    "env_index": int(row["env_index"]),
                    "dist1_to_4_mass": dist_mass,
                    "pre_vs_zero_suffix_gap": float(row["pre_vs_zero_suffix_gap"]),
                    "objective_terminal_reset_count": int(row["objective_terminal_reset_count"]),
                }
            )
        suite_rows.extend(rows)
        suite_summary = _group_summary(rows)
        suite_summary.update(
            {
                "suite_name": str(spec["suite_name"]),
                "suite_regime": regime,
                "ppo_minus_zero_suffix": float(ppo_minus_zero_suffix),
            }
        )
        suite_summaries.append(suite_summary)

    all_summary = _group_summary(suite_rows)
    all_masses = [float(row["dist1_to_4_mass"]) for row in suite_rows]
    all_pre = [float(row["pre_vs_zero_suffix_gap"]) for row in suite_rows]
    global_x_mean = float(np.mean(np.asarray(all_masses, dtype=np.float32))) if all_masses else 0.0
    global_y_mean = float(np.mean(np.asarray(all_pre, dtype=np.float32))) if all_pre else 0.0
    total_cov_num = _covariance_numerator(
        suite_rows,
        x_key="dist1_to_4_mass",
        y_key="pre_vs_zero_suffix_gap",
        x_mean=global_x_mean,
        y_mean=global_y_mean,
    )

    regime_groups: dict[str, list[dict[str, Any]]] = {"positive": [], "nonpositive": []}
    for row in suite_rows:
        regime_groups[str(row["suite_regime"])].append(row)

    regime_summaries: dict[str, Any] = {}
    total_mass = float(sum(float(row["dist1_to_4_mass"]) for row in suite_rows))
    for regime_name, rows in regime_groups.items():
        summary = _group_summary(rows)
        summary["suite_names"] = sorted({str(row["suite_name"]) for row in rows})
        summary["mass_share"] = (
            float(summary["total_dist1_to_4_mass"] / total_mass) if total_mass > 0.0 else 0.0
        )
        cov_num = _covariance_numerator(
            rows,
            x_key="dist1_to_4_mass",
            y_key="pre_vs_zero_suffix_gap",
            x_mean=global_x_mean,
            y_mean=global_y_mean,
        )
        summary["pooled_covariance_numerator_contribution"] = float(cov_num)
        summary["pooled_covariance_contribution_share"] = (
            float(cov_num / total_cov_num) if abs(total_cov_num) > 1e-12 else 0.0
        )
        regime_summaries[regime_name] = summary

    positive_weighted = float(regime_summaries["positive"]["dist1_to_4_mass_weighted_pre_gap"])
    nonpositive_weighted = float(regime_summaries["nonpositive"]["dist1_to_4_mass_weighted_pre_gap"])
    positive_corr = float(regime_summaries["positive"]["corr_dist1_to_4_mass_vs_pre_gap"])
    nonpositive_corr = float(regime_summaries["nonpositive"]["corr_dist1_to_4_mass_vs_pre_gap"])
    all_corr = float(all_summary["corr_dist1_to_4_mass_vs_pre_gap"])
    positive_mass_share = float(regime_summaries["positive"]["mass_share"])
    nonpositive_mass_share = float(regime_summaries["nonpositive"]["mass_share"])
    positive_cov_share = float(regime_summaries["positive"]["pooled_covariance_contribution_share"])
    nonpositive_cov_share = float(regime_summaries["nonpositive"]["pooled_covariance_contribution_share"])
    regime_split_visible = bool(positive_corr < 0.0 and nonpositive_corr > 0.0)

    return {
        "audit_entry": "phase3_regime_mixed_update_probe",
        "suite_count": int(len(suite_summaries)),
        "suite_summaries": suite_summaries,
        "skipped_suites": skipped_suites,
        "all_envs_summary": all_summary,
        "regime_groups": regime_summaries,
        "conclusions": {
            "reset_tracking_stable_across_regimes": bool(
                float(regime_summaries["positive"]["corr_dist1_to_4_mass_vs_reset_count"]) > 0.0
                and float(regime_summaries["nonpositive"]["corr_dist1_to_4_mass_vs_reset_count"]) > 0.0
            ),
            "pre_gap_alignment_flips_by_regime": bool(
                positive_weighted < 0.0 and nonpositive_weighted > 0.0
            ),
            "regime_split_visible_in_raw_correlation": regime_split_visible,
            "pooled_sign_hides_regime_split": bool(
                regime_split_visible
                and all_corr < 0.0
                and positive_corr < 0.0
                and nonpositive_corr > 0.0
            ),
            "positive_regime_dominates_pooled_update": bool(
                positive_mass_share > nonpositive_mass_share
                and abs(positive_cov_share) > abs(nonpositive_cov_share)
            ),
            "mixed_update_conflict_present": bool(
                (positive_weighted < 0.0 and nonpositive_weighted > 0.0)
                or regime_split_visible
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 3 regime-aware mixed-update probe using suite-matched baseline and distance artifacts."
    )
    parser.add_argument("--suite-specs-json", type=str, default=None)
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output_json).expanduser().resolve()
    if output_path.exists() and not bool(args.overwrite):
        print(output_path.read_text())
        return 0

    suite_specs = DEFAULT_SUITES
    if args.suite_specs_json is not None:
        suite_specs = json.loads(Path(args.suite_specs_json).expanduser().resolve().read_text())

    result = build_regime_mixed_update_probe(suite_specs)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(output_path.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
