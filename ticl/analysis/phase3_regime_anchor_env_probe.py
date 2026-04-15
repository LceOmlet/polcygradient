import argparse
import json
from pathlib import Path
from typing import Any


DIST_KEYS = tuple(f"post_reset_tail_dist_{idx}" for idx in range(1, 5))

PAIR1_ZERO_JSON = "/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/zero_cross_env_control.json"
PAIR1_PPO_JSON = "/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/ppo_cross_env_baseline.json"
PAIR1_DISTANCE_JSON = "/home/chen/RLPFN/artifacts/phase3_post_reset_tail_distance_probe_pair1_suite_matched.json"

PAIR2_ZERO_JSON = "/home/chen/RLPFN/artifacts/phase3_cross_env_baseline_pair2/zero_cross_env_control.json"
PAIR2_PPO_JSON = "/home/chen/RLPFN/artifacts/phase3_cross_env_baseline_pair2/ppo_cross_env_baseline.json"
PAIR2_DISTANCE_JSON = "/home/chen/RLPFN/artifacts/phase3_post_reset_tail_distance_probe_pair2.json"


def _load_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).expanduser().resolve().read_text())


def _dist_mass(row: dict[str, Any]) -> float:
    return float(sum(float(row[key]) for key in DIST_KEYS))


def _rank_desc(values: dict[int, float]) -> dict[int, int]:
    ordered = sorted(values.items(), key=lambda item: float(item[1]), reverse=True)
    return {int(env_index): int(rank) for rank, (env_index, _) in enumerate(ordered, start=1)}


def _env_rows_with_baselines(
    *,
    suite_name: str,
    distance_payload: dict[str, Any],
    zero_payload: dict[str, Any],
    ppo_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    zero_metrics = zero_payload["heldout_suite"]["metrics"]
    ppo_metrics = ppo_payload["heldout_suite"]["metrics"]
    zero_suffix = zero_metrics["suffix_return_per_env"]
    ppo_suffix = ppo_metrics["suffix_return_per_env"]
    zero_full = zero_metrics["full_return_per_env"]
    ppo_full = ppo_metrics["full_return_per_env"]
    rows = []
    for row in distance_payload["env_rows"]:
        env_index = int(row["env_index"])
        dist_mass = _dist_mass(row)
        rows.append(
            {
                "suite_name": suite_name,
                "env_index": env_index,
                "env_seed": int(row["env_seed"]),
                "rollout_seed": int(row["rollout_seed"]),
                "objective_terminal_reset_count": int(row["objective_terminal_reset_count"]),
                "objective_token_count": int(row["objective_token_count"]),
                "dist1_to_4_mass": dist_mass,
                "pre_vs_zero_suffix_gap": float(row["pre_vs_zero_suffix_gap"]),
                "zero_suffix_return": float(zero_suffix[env_index]),
                "ppo_suffix_return": float(ppo_suffix[env_index]),
                "suffix_gap": float(float(ppo_suffix[env_index]) - float(zero_suffix[env_index])),
                "zero_full_return": float(zero_full[env_index]),
                "ppo_full_return": float(ppo_full[env_index]),
                "full_gap": float(float(ppo_full[env_index]) - float(zero_full[env_index])),
            }
        )
    total_mass = float(sum(float(row["dist1_to_4_mass"]) for row in rows))
    mass_rank = _rank_desc({int(row["env_index"]): float(row["dist1_to_4_mass"]) for row in rows})
    gap_rank = _rank_desc({int(row["env_index"]): float(row["pre_vs_zero_suffix_gap"]) for row in rows})
    for row in rows:
        env_index = int(row["env_index"])
        row["dist1_to_4_mass_rank"] = int(mass_rank[env_index])
        row["pre_gap_rank"] = int(gap_rank[env_index])
        row["dist1_to_4_mass_share_within_suite"] = (
            float(float(row["dist1_to_4_mass"]) / total_mass) if total_mass > 0.0 else 0.0
        )
    return rows


def _top_rows(rows: list[dict[str, Any]], *, key: str, k: int) -> list[dict[str, Any]]:
    ranked = sorted(rows, key=lambda row: float(row[key]), reverse=True)
    keep = [
        "suite_name",
        "env_index",
        "env_seed",
        "rollout_seed",
        "objective_terminal_reset_count",
        "objective_token_count",
        "dist1_to_4_mass",
        "dist1_to_4_mass_rank",
        "dist1_to_4_mass_share_within_suite",
        "pre_vs_zero_suffix_gap",
        "pre_gap_rank",
        "zero_suffix_return",
        "ppo_suffix_return",
        "suffix_gap",
        "zero_full_return",
        "ppo_full_return",
        "full_gap",
    ]
    return [{field: row[field] for field in keep} for row in ranked[:k]]


def _suite_contract_summary(zero_payload: dict[str, Any], ppo_payload: dict[str, Any]) -> dict[str, Any]:
    zero_metrics = zero_payload["heldout_suite"]["metrics"]
    ppo_metrics = ppo_payload["heldout_suite"]["metrics"]
    zero_summary = zero_payload["heldout_suite"].get("summary", {})
    ppo_summary = ppo_payload["heldout_suite"].get("summary", {})
    checked_fields = [
        "rollout_backend",
        "batch_size",
        "n_samples",
        "single_eval_pos",
        "suffix_horizon",
    ]
    metric_matches = {
        field: zero_metrics.get(field) == ppo_metrics.get(field)
        for field in checked_fields
    }
    suite_matches = {
        "fingerprint": zero_summary.get("fingerprint") == ppo_summary.get("fingerprint"),
        "suite_seed": zero_summary.get("suite_seed") == ppo_summary.get("suite_seed"),
    }
    return {
        "metric_matches": metric_matches,
        "suite_matches": suite_matches,
        "all_checked_fields_match": bool(all(metric_matches.values()) and all(suite_matches.values())),
        "zero_nonfinite_counts": {
            "full_return_nonfinite_count": int(zero_metrics["full_return_nonfinite_count"]),
            "suffix_return_nonfinite_count": int(zero_metrics["suffix_return_nonfinite_count"]),
            "reward_step_nonfinite_count": int(zero_metrics["reward_step_nonfinite_count"]),
        },
        "ppo_nonfinite_counts": {
            "full_return_nonfinite_count": int(ppo_metrics["full_return_nonfinite_count"]),
            "suffix_return_nonfinite_count": int(ppo_metrics["suffix_return_nonfinite_count"]),
            "reward_step_nonfinite_count": int(ppo_metrics["reward_step_nonfinite_count"]),
        },
    }


def build_regime_anchor_env_probe(
    *,
    pair1_zero_json: str = PAIR1_ZERO_JSON,
    pair1_ppo_json: str = PAIR1_PPO_JSON,
    pair1_distance_json: str = PAIR1_DISTANCE_JSON,
    pair2_zero_json: str = PAIR2_ZERO_JSON,
    pair2_ppo_json: str = PAIR2_PPO_JSON,
    pair2_distance_json: str = PAIR2_DISTANCE_JSON,
) -> dict[str, Any]:
    pair1_zero = _load_json(pair1_zero_json)
    pair1_ppo = _load_json(pair1_ppo_json)
    pair1_distance = _load_json(pair1_distance_json)
    pair2_zero = _load_json(pair2_zero_json)
    pair2_ppo = _load_json(pair2_ppo_json)
    pair2_distance = _load_json(pair2_distance_json)

    pair1_rows = _env_rows_with_baselines(
        suite_name="pair1",
        distance_payload=pair1_distance,
        zero_payload=pair1_zero,
        ppo_payload=pair1_ppo,
    )
    pair2_rows = _env_rows_with_baselines(
        suite_name="pair2",
        distance_payload=pair2_distance,
        zero_payload=pair2_zero,
        ppo_payload=pair2_ppo,
    )

    pair1_total_mass = float(sum(float(row["dist1_to_4_mass"]) for row in pair1_rows))
    pair1_top_mass = _top_rows(pair1_rows, key="dist1_to_4_mass", k=5)
    pair1_top_gap = _top_rows(pair1_rows, key="pre_vs_zero_suffix_gap", k=5)
    pair1_top3_mass_share = (
        float(sum(float(row["dist1_to_4_mass"]) for row in pair1_top_mass[:3]) / pair1_total_mass)
        if pair1_total_mass > 0.0
        else 0.0
    )
    pair1_nonzero_mass_env_count = int(sum(1 for row in pair1_rows if float(row["dist1_to_4_mass"]) > 0.0))
    pair1_top3_reset_active = bool(
        pair1_top_mass
        and all(int(row["objective_terminal_reset_count"]) > 0 for row in pair1_top_mass[:3])
    )
    pair1_has_zero_mass_high_gain_anchor = bool(
        any(
            float(row["dist1_to_4_mass"]) == 0.0 and float(row["pre_vs_zero_suffix_gap"]) > 0.0
            for row in pair1_top_gap[:2]
        )
    )

    pair2_total_mass = float(sum(float(row["dist1_to_4_mass"]) for row in pair2_rows))
    pair2_top_mass = _top_rows(pair2_rows, key="dist1_to_4_mass", k=5)
    pair2_top_gap = _top_rows(pair2_rows, key="pre_vs_zero_suffix_gap", k=5)
    pair2_env12 = next(row for row in pair2_rows if int(row["env_index"]) == 12)
    pair2_contract = _suite_contract_summary(pair2_zero, pair2_ppo)
    pair2_env12_metric_outlier_unlikely = bool(
        pair2_contract["all_checked_fields_match"]
        and all(int(value) == 0 for value in pair2_contract["zero_nonfinite_counts"].values())
        and all(int(value) == 0 for value in pair2_contract["ppo_nonfinite_counts"].values())
        and float(pair2_env12["suffix_gap"]) > 0.0
        and float(pair2_env12["full_gap"]) > 0.0
    )
    pair2_env12_proxy_outlier = bool(
        int(pair2_env12["pre_gap_rank"]) == 1
        and int(pair2_env12["dist1_to_4_mass_rank"]) >= 3
        and float(pair2_env12["dist1_to_4_mass_share_within_suite"]) <= 0.02
    )

    return {
        "audit_entry": "phase3_regime_anchor_env_probe",
        "config": {
            "pair1_zero_json": str(Path(pair1_zero_json).expanduser().resolve()),
            "pair1_ppo_json": str(Path(pair1_ppo_json).expanduser().resolve()),
            "pair1_distance_json": str(Path(pair1_distance_json).expanduser().resolve()),
            "pair2_zero_json": str(Path(pair2_zero_json).expanduser().resolve()),
            "pair2_ppo_json": str(Path(pair2_ppo_json).expanduser().resolve()),
            "pair2_distance_json": str(Path(pair2_distance_json).expanduser().resolve()),
        },
        "pair1_anchor": {
            "top_mass_envs": pair1_top_mass,
            "top_pre_gap_envs": pair1_top_gap,
            "top3_mass_share_within_suite": pair1_top3_mass_share,
            "nonzero_mass_env_count": pair1_nonzero_mass_env_count,
            "env_count": int(len(pair1_rows)),
        },
        "pair2_anchor": {
            "top_mass_envs": pair2_top_mass,
            "top_pre_gap_envs": pair2_top_gap,
            "env12": pair2_env12,
            "compare_contract": pair2_contract,
            "env_count": int(len(pair2_rows)),
            "pair2_total_mass": pair2_total_mass,
        },
        "conclusions": {
            "pair1_mass_concentration_is_sparse_reset_active": bool(
                pair1_top3_mass_share >= 0.95
                and pair1_nonzero_mass_env_count <= 6
                and pair1_top3_reset_active
            ),
            "pair1_mass_proxy_not_exhaustive_for_gain": pair1_has_zero_mass_high_gain_anchor,
            "pair2_env12_metric_outlier_unlikely": pair2_env12_metric_outlier_unlikely,
            "pair2_env12_is_proxy_outlier": pair2_env12_proxy_outlier,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Anchor probe for pair1 heavy-mass envs and pair2 env12 tiny-mass / huge-gap behavior."
    )
    parser.add_argument("--pair1-zero-json", type=str, default=PAIR1_ZERO_JSON)
    parser.add_argument("--pair1-ppo-json", type=str, default=PAIR1_PPO_JSON)
    parser.add_argument("--pair1-distance-json", type=str, default=PAIR1_DISTANCE_JSON)
    parser.add_argument("--pair2-zero-json", type=str, default=PAIR2_ZERO_JSON)
    parser.add_argument("--pair2-ppo-json", type=str, default=PAIR2_PPO_JSON)
    parser.add_argument("--pair2-distance-json", type=str, default=PAIR2_DISTANCE_JSON)
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output_json).expanduser().resolve()
    if output_path.exists() and not bool(args.overwrite):
        print(output_path.read_text())
        return 0

    result = build_regime_anchor_env_probe(
        pair1_zero_json=str(args.pair1_zero_json),
        pair1_ppo_json=str(args.pair1_ppo_json),
        pair1_distance_json=str(args.pair1_distance_json),
        pair2_zero_json=str(args.pair2_zero_json),
        pair2_ppo_json=str(args.pair2_ppo_json),
        pair2_distance_json=str(args.pair2_distance_json),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(output_path.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
