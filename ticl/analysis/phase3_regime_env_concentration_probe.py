import argparse
import json
from pathlib import Path
import time
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


def _mass_concentration(rows: list[dict[str, Any]], mass_key: str) -> dict[str, Any]:
    masses = [max(0.0, float(row[mass_key])) for row in rows]
    total = float(sum(masses))
    if total <= 0.0:
        return {
            "count": int(len(rows)),
            "nonzero_count": 0,
            "total_mass": 0.0,
            "top1_share": 0.0,
            "top2_share": 0.0,
            "top3_share": 0.0,
            "top5_share": 0.0,
            "mass_hhi": 0.0,
        }
    shares = sorted((float(m) / total for m in masses if m > 0.0), reverse=True)
    return {
        "count": int(len(rows)),
        "nonzero_count": int(sum(1 for m in masses if m > 0.0)),
        "total_mass": total,
        "top1_share": float(sum(shares[:1])),
        "top2_share": float(sum(shares[:2])),
        "top3_share": float(sum(shares[:3])),
        "top5_share": float(sum(shares[:5])),
        "mass_hhi": float(sum(share * share for share in shares)),
    }


def _covariance_numerator(
    rows: list[dict[str, Any]],
    *,
    x_key: str,
    y_key: str,
    x_mean: float,
    y_mean: float,
) -> float:
    total = 0.0
    for row in rows:
        x = float(row[x_key])
        y = float(row[y_key])
        if np.isfinite(x) and np.isfinite(y):
            total += (x - float(x_mean)) * (y - float(y_mean))
    return float(total)


def _top_rows(
    rows: list[dict[str, Any]],
    *,
    rank_key: str,
    share_total: float,
    k: int = 5,
) -> list[dict[str, Any]]:
    ranked = sorted(rows, key=lambda row: float(row[rank_key]), reverse=True)
    return _rows_to_output(ranked[:k], share_total=share_total)


def _rows_to_output(rows: list[dict[str, Any]], *, share_total: float) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        mass = float(row["dist1_to_4_mass"])
        out.append(
            {
                "suite_name": str(row["suite_name"]),
                "env_index": int(row["env_index"]),
                "env_seed": int(row["env_seed"]),
                "rollout_seed": int(row["rollout_seed"]),
                "objective_terminal_reset_count": int(row["objective_terminal_reset_count"]),
                "objective_token_count": int(row["objective_token_count"]),
                "dist1_to_4_mass": mass,
                "dist1_to_4_mass_share": float(mass / share_total) if share_total > 0.0 else 0.0,
                "pre_vs_zero_suffix_gap": float(row["pre_vs_zero_suffix_gap"]),
                "pooled_covariance_numerator": float(row["pooled_covariance_numerator"]),
                "regime_local_covariance_numerator": float(row["regime_local_covariance_numerator"]),
            }
        )
    return out


def _suite_summary(
    rows: list[dict[str, Any]],
    *,
    regime_total_mass: float,
    global_total_mass: float,
    regime_mass_mean: float,
    regime_pre_mean: float,
) -> dict[str, Any]:
    masses = [float(row["dist1_to_4_mass"]) for row in rows]
    pre_gap = [float(row["pre_vs_zero_suffix_gap"]) for row in rows]
    pooled_cov = float(sum(float(row["pooled_covariance_numerator"]) for row in rows))
    regime_local_cov = _covariance_numerator(
        rows,
        x_key="dist1_to_4_mass",
        y_key="pre_vs_zero_suffix_gap",
        x_mean=regime_mass_mean,
        y_mean=regime_pre_mean,
    )
    total_mass = float(sum(masses))
    suite_name = str(rows[0]["suite_name"]) if rows else "unknown"
    return {
        "suite_name": suite_name,
        "env_count": int(len(rows)),
        "ppo_minus_zero_suffix": float(rows[0]["ppo_minus_zero_suffix"]) if rows else 0.0,
        "suite_regime": str(rows[0]["suite_regime"]) if rows else "unknown",
        "total_mass": total_mass,
        "mass_share_within_regime": float(total_mass / regime_total_mass) if regime_total_mass > 0.0 else 0.0,
        "mass_share_global": float(total_mass / global_total_mass) if global_total_mass > 0.0 else 0.0,
        "local_corr_mass_vs_pre_gap": _corr(masses, pre_gap),
        "pooled_covariance_numerator_contribution": pooled_cov,
        "regime_local_covariance_numerator_contribution": regime_local_cov,
        "mass_concentration": _mass_concentration(rows, "dist1_to_4_mass"),
        "top_envs_by_mass": _top_rows(rows, rank_key="dist1_to_4_mass", share_total=total_mass, k=5),
        "top_envs_by_abs_pooled_covariance": _rows_to_output(
            sorted(rows, key=lambda row: abs(float(row["pooled_covariance_numerator"])), reverse=True)[:5],
            share_total=total_mass,
        ),
    }


def _regime_summary(
    rows: list[dict[str, Any]],
    *,
    global_total_mass: float,
    global_mass_mean: float,
    global_pre_mean: float,
) -> dict[str, Any]:
    masses = [float(row["dist1_to_4_mass"]) for row in rows]
    pre_gap = [float(row["pre_vs_zero_suffix_gap"]) for row in rows]
    total_mass = float(sum(masses))
    regime_mass_mean = float(sum(masses) / len(masses)) if masses else 0.0
    regime_pre_mean = float(sum(pre_gap) / len(pre_gap)) if pre_gap else 0.0
    pooled_cov = _covariance_numerator(
        rows,
        x_key="dist1_to_4_mass",
        y_key="pre_vs_zero_suffix_gap",
        x_mean=global_mass_mean,
        y_mean=global_pre_mean,
    )
    local_cov = _covariance_numerator(
        rows,
        x_key="dist1_to_4_mass",
        y_key="pre_vs_zero_suffix_gap",
        x_mean=regime_mass_mean,
        y_mean=regime_pre_mean,
    )
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["suite_name"]), []).append(row)
    suite_breakdown = [
        _suite_summary(
            suite_rows,
            regime_total_mass=total_mass,
            global_total_mass=global_total_mass,
            regime_mass_mean=regime_mass_mean,
            regime_pre_mean=regime_pre_mean,
        )
        for _, suite_rows in sorted(
            grouped.items(),
            key=lambda item: sum(float(row["dist1_to_4_mass"]) for row in item[1]),
            reverse=True,
        )
    ]
    return {
        "env_count": int(len(rows)),
        "suite_names": [str(item["suite_name"]) for item in suite_breakdown],
        "total_mass": total_mass,
        "mass_share_global": float(total_mass / global_total_mass) if global_total_mass > 0.0 else 0.0,
        "mass_mean": regime_mass_mean,
        "pre_gap_mean": regime_pre_mean,
        "local_corr_mass_vs_pre_gap": _corr(masses, pre_gap),
        "corr_mass_vs_reset_count": _corr(
            masses,
            [float(row["objective_terminal_reset_count"]) for row in rows],
        ),
        "pooled_covariance_numerator_contribution": pooled_cov,
        "regime_local_covariance_numerator": local_cov,
        "mass_concentration": _mass_concentration(rows, "dist1_to_4_mass"),
        "top_envs_by_mass": _top_rows(rows, rank_key="dist1_to_4_mass", share_total=total_mass, k=5),
        "top_envs_by_abs_pooled_covariance": _rows_to_output(
            sorted(rows, key=lambda row: abs(float(row["pooled_covariance_numerator"])), reverse=True)[:5],
            share_total=total_mass,
        ),
        "suite_breakdown": suite_breakdown,
    }


def build_regime_env_concentration_probe(suite_specs: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
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
        for row in distance_payload["env_rows"]:
            dist_mass = float(sum(float(row[key]) for key in DIST_KEYS))
            rows.append(
                {
                    "suite_name": str(spec["suite_name"]),
                    "suite_regime": regime,
                    "ppo_minus_zero_suffix": float(ppo_minus_zero_suffix),
                    "env_index": int(row["env_index"]),
                    "env_seed": int(row["env_seed"]),
                    "rollout_seed": int(row["rollout_seed"]),
                    "objective_terminal_reset_count": int(row["objective_terminal_reset_count"]),
                    "objective_token_count": int(row["objective_token_count"]),
                    "dist1_to_4_mass": dist_mass,
                    "pre_vs_zero_suffix_gap": float(row["pre_vs_zero_suffix_gap"]),
                }
            )

    all_masses = [float(row["dist1_to_4_mass"]) for row in rows]
    all_pre_gap = [float(row["pre_vs_zero_suffix_gap"]) for row in rows]
    global_total_mass = float(sum(all_masses))
    global_mass_mean = float(sum(all_masses) / len(all_masses)) if all_masses else 0.0
    global_pre_mean = float(sum(all_pre_gap) / len(all_pre_gap)) if all_pre_gap else 0.0

    regime_groups: dict[str, list[dict[str, Any]]] = {"positive": [], "nonpositive": []}
    for row in rows:
        regime_groups[str(row["suite_regime"])].append(row)

    regime_means: dict[str, tuple[float, float]] = {}
    for regime_name, regime_rows in regime_groups.items():
        masses = [float(row["dist1_to_4_mass"]) for row in regime_rows]
        pre_gap = [float(row["pre_vs_zero_suffix_gap"]) for row in regime_rows]
        regime_means[regime_name] = (
            float(sum(masses) / len(masses)) if masses else 0.0,
            float(sum(pre_gap) / len(pre_gap)) if pre_gap else 0.0,
        )

    for row in rows:
        regime_mass_mean, regime_pre_mean = regime_means[str(row["suite_regime"])]
        row["pooled_covariance_numerator"] = float(
            (float(row["dist1_to_4_mass"]) - global_mass_mean)
            * (float(row["pre_vs_zero_suffix_gap"]) - global_pre_mean)
        )
        row["regime_local_covariance_numerator"] = float(
            (float(row["dist1_to_4_mass"]) - regime_mass_mean)
            * (float(row["pre_vs_zero_suffix_gap"]) - regime_pre_mean)
        )

    regime_summaries = {
        regime_name: _regime_summary(
            regime_rows,
            global_total_mass=global_total_mass,
            global_mass_mean=global_mass_mean,
            global_pre_mean=global_pre_mean,
        )
        for regime_name, regime_rows in regime_groups.items()
    }

    positive = regime_summaries["positive"]
    nonpositive = regime_summaries["nonpositive"]
    positive_top_suite_share = float(
        positive["suite_breakdown"][0]["mass_share_within_regime"] if positive["suite_breakdown"] else 0.0
    )
    positive_abs_pooled_cov = abs(float(positive["pooled_covariance_numerator_contribution"]))
    nonpositive_abs_pooled_cov = abs(float(nonpositive["pooled_covariance_numerator_contribution"]))

    return {
        "audit_entry": "phase3_regime_env_concentration_probe",
        "suite_count": int(len({str(row["suite_name"]) for row in rows})),
        "skipped_suites": skipped_suites,
        "all_envs_summary": {
            "env_count": int(len(rows)),
            "total_mass": global_total_mass,
            "mass_mean": global_mass_mean,
            "pre_gap_mean": global_pre_mean,
            "corr_mass_vs_pre_gap": _corr(all_masses, all_pre_gap),
            "mass_concentration": _mass_concentration(rows, "dist1_to_4_mass"),
        },
        "regime_groups": regime_summaries,
        "conclusions": {
            "nonpositive_local_signal_positive": bool(
                float(nonpositive["local_corr_mass_vs_pre_gap"]) > 0.0
            ),
            "nonpositive_signal_flips_under_global_centering": bool(
                float(nonpositive["local_corr_mass_vs_pre_gap"]) > 0.0
                and float(nonpositive["pooled_covariance_numerator_contribution"]) < 0.0
            ),
            "positive_regime_mass_share_exceeds_nonpositive": bool(
                float(positive["mass_share_global"]) > float(nonpositive["mass_share_global"])
            ),
            "positive_regime_abs_pooled_cov_dominates_nonpositive": bool(
                positive_abs_pooled_cov > nonpositive_abs_pooled_cov
            ),
            "positive_regime_few_envs_dominate_mass": bool(
                float(positive["mass_concentration"]["top3_share"]) >= 0.8
            ),
            "largest_positive_suite_dominates_regime_mass": bool(positive_top_suite_share >= 0.8),
            "nonpositive_signal_drowned_by_mass_and_centering": bool(
                float(nonpositive["local_corr_mass_vs_pre_gap"]) > 0.0
                and float(nonpositive["pooled_covariance_numerator_contribution"]) < 0.0
                and float(positive["mass_share_global"]) > float(nonpositive["mass_share_global"])
                and positive_abs_pooled_cov > nonpositive_abs_pooled_cov
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 3 regime-aware per-suite/per-env update-mass concentration probe."
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

    t0 = time.perf_counter()
    result = build_regime_env_concentration_probe(suite_specs)
    result["runtime_wall_s"] = float(time.perf_counter() - t0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(output_path.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
