import argparse
import json
from pathlib import Path
from typing import Any


TAIL_LEN_KEY = "8"
DIST_KEYS = tuple(f"post_reset_tail_dist_{idx}" for idx in range(1, 5))

DEFAULT_POSITIVE_SUITES = [
    {
        "suite_name": "pair1",
        "distance_probe_json": "/home/chen/RLPFN/artifacts/phase3_post_reset_tail_distance_probe_pair1_suite_matched.json",
        "tail_mask_probe_json": "/home/chen/RLPFN/artifacts/phase3_terminal_tail_mask_probe_pair1_suite_matched.json",
    },
    {
        "suite_name": "pair2",
        "distance_probe_json": "/home/chen/RLPFN/artifacts/phase3_post_reset_tail_distance_probe_pair2.json",
        "tail_mask_probe_json": "/home/chen/RLPFN/artifacts/phase3_terminal_tail_mask_probe_pair2.json",
    },
    {
        "suite_name": "seed13579",
        "distance_probe_json": "/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_13579/phase3_post_reset_tail_distance_probe.json",
        "tail_mask_probe_json": "/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_13579/phase3_terminal_tail_mask_probe.json",
    },
]


def _load_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).expanduser().resolve().read_text())


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(float(v) for v in values) / len(values))


def _first_episode_non_tail_mass(tail_len_row: dict[str, Any]) -> float:
    first_episode_mass = float(tail_len_row["first_episode_only"]["removed_positive_mass"])
    first_terminal_tail_mass = float(tail_len_row["first_terminal_tail"]["removed_positive_mass"])
    return float(max(0.0, first_episode_mass - first_terminal_tail_mass))


def _build_rows(specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for spec in specs:
        distance_payload = _load_json(str(spec["distance_probe_json"]))
        tail_payload = _load_json(str(spec["tail_mask_probe_json"]))
        tail_by_env = {int(row["env_index"]): row for row in tail_payload["env_rows"]}
        rows = []
        for row in distance_payload["env_rows"]:
            env_index = int(row["env_index"])
            tail_row = tail_by_env[env_index]
            tail_len_row = tail_row["tail_len_rows"][TAIL_LEN_KEY]
            rows.append(
                {
                    "suite_name": str(spec["suite_name"]),
                    "env_index": env_index,
                    "env_seed": int(row["env_seed"]),
                    "rollout_seed": int(row["rollout_seed"]),
                    "objective_terminal_reset_count": int(row["objective_terminal_reset_count"]),
                    "objective_token_count": int(row["objective_token_count"]),
                    "pre_vs_zero_suffix_gap": float(row["pre_vs_zero_suffix_gap"]),
                    "dist1_to_4_mass": float(sum(float(row[key]) for key in DIST_KEYS)),
                    "base_positive_mass": float(tail_row["base_positive_mass"]),
                    "first_episode_only_mass": float(tail_len_row["first_episode_only"]["removed_positive_mass"]),
                    "first_terminal_tail_mass": float(tail_len_row["first_terminal_tail"]["removed_positive_mass"]),
                    "first_episode_non_tail_mass": _first_episode_non_tail_mass(tail_len_row),
                }
            )
        dist_total = float(sum(float(row["dist1_to_4_mass"]) for row in rows))
        first_episode_non_tail_total = float(sum(float(row["first_episode_non_tail_mass"]) for row in rows))
        fused_total = float(
            sum(float(row["dist1_to_4_mass"]) + float(row["first_episode_non_tail_mass"]) for row in rows)
        )
        for row in rows:
            row["reset_tail_share_within_suite"] = (
                float(float(row["dist1_to_4_mass"]) / dist_total) if dist_total > 0.0 else 0.0
            )
            row["first_episode_non_tail_share_within_suite"] = (
                float(float(row["first_episode_non_tail_mass"]) / first_episode_non_tail_total)
                if first_episode_non_tail_total > 0.0
                else 0.0
            )
            row["fused_share_within_suite"] = (
                float((float(row["dist1_to_4_mass"]) + float(row["first_episode_non_tail_mass"])) / fused_total)
                if fused_total > 0.0
                else 0.0
            )
        out.extend(rows)
    return out


def _keep_fields() -> list[str]:
    return [
        "suite_name",
        "env_index",
        "env_seed",
        "rollout_seed",
        "objective_terminal_reset_count",
        "objective_token_count",
        "pre_vs_zero_suffix_gap",
        "dist1_to_4_mass",
        "reset_tail_share_within_suite",
        "first_episode_non_tail_mass",
        "first_episode_non_tail_share_within_suite",
        "fused_share_within_suite",
        "base_positive_mass",
        "first_episode_only_mass",
        "first_terminal_tail_mass",
    ]


def _group_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fields = _keep_fields()
    return {
        "count": int(len(rows)),
        "suite_counts": {
            suite_name: int(sum(1 for row in rows if str(row["suite_name"]) == suite_name))
            for suite_name in sorted({str(row["suite_name"]) for row in rows})
        },
        "mean_pre_gap": _mean([float(row["pre_vs_zero_suffix_gap"]) for row in rows]),
        "mean_reset_count": _mean([float(row["objective_terminal_reset_count"]) for row in rows]),
        "mean_base_positive_mass": _mean([float(row["base_positive_mass"]) for row in rows]),
        "mean_reset_tail_share": _mean([float(row["reset_tail_share_within_suite"]) for row in rows]),
        "mean_first_episode_non_tail_share": _mean(
            [float(row["first_episode_non_tail_share_within_suite"]) for row in rows]
        ),
        "mean_fused_share": _mean([float(row["fused_share_within_suite"]) for row in rows]),
        "env_items": [{field: row[field] for field in fields} for row in sorted(
            rows,
            key=lambda row: float(row["pre_vs_zero_suffix_gap"]),
            reverse=True,
        )],
    }


def build_dual_channel_anchor_coverage_probe(
    suite_specs: list[dict[str, Any]],
    *,
    strong_gain_min_pre_gap: float,
    coverage_share_threshold: float,
) -> dict[str, Any]:
    rows = _build_rows(suite_specs)
    strong_gain = [row for row in rows if float(row["pre_vs_zero_suffix_gap"]) > float(strong_gain_min_pre_gap)]
    baseline_captured = [
        row for row in strong_gain if float(row["reset_tail_share_within_suite"]) >= float(coverage_share_threshold)
    ]
    dual_channel_captured = [
        row
        for row in strong_gain
        if float(row["reset_tail_share_within_suite"]) >= float(coverage_share_threshold)
        or float(row["first_episode_non_tail_share_within_suite"]) >= float(coverage_share_threshold)
    ]
    fused_captured = [
        row for row in strong_gain if float(row["fused_share_within_suite"]) >= float(coverage_share_threshold)
    ]
    recovered = [
        row
        for row in dual_channel_captured
        if float(row["reset_tail_share_within_suite"]) < float(coverage_share_threshold)
    ]
    residual_missed = [
        row
        for row in strong_gain
        if float(row["reset_tail_share_within_suite"]) < float(coverage_share_threshold)
        and float(row["first_episode_non_tail_share_within_suite"]) < float(coverage_share_threshold)
    ]

    baseline_summary = _group_summary(baseline_captured)
    dual_summary = _group_summary(dual_channel_captured)
    recovered_summary = _group_summary(recovered)
    residual_summary = _group_summary(residual_missed)

    return {
        "audit_entry": "phase3_dual_channel_anchor_coverage_probe",
        "config": {
            "strong_gain_min_pre_gap": float(strong_gain_min_pre_gap),
            "coverage_share_threshold": float(coverage_share_threshold),
            "tail_len_key": TAIL_LEN_KEY,
            "suite_names": [str(spec["suite_name"]) for spec in suite_specs],
        },
        "strong_gain_anchor_count": int(len(strong_gain)),
        "baseline_reset_tail_summary": baseline_summary,
        "dual_channel_summary": dual_summary,
        "fused_sum_summary": _group_summary(fused_captured),
        "recovered_anchor_summary": recovered_summary,
        "residual_missed_summary": residual_summary,
        "deltas": {
            "dual_channel_coverage_gain": int(len(dual_channel_captured) - len(baseline_captured)),
            "fused_sum_coverage_gain": int(len(fused_captured) - len(baseline_captured)),
            "residual_missed_count": int(len(residual_missed)),
            "residual_missed_mean_base_positive_mass_minus_recovered": float(
                residual_summary["mean_base_positive_mass"] - recovered_summary["mean_base_positive_mass"]
            ),
            "residual_missed_mean_first_episode_non_tail_share_minus_recovered": float(
                residual_summary["mean_first_episode_non_tail_share"]
                - recovered_summary["mean_first_episode_non_tail_share"]
            ),
        },
        "conclusions": {
            "dual_channel_improves_anchor_coverage": bool(len(dual_channel_captured) > len(baseline_captured)),
            "dual_channel_recovers_nonreset_first_episode_anchors": bool(recovered),
            "residual_missed_anchors_remain": bool(residual_missed),
            "residual_missed_anchors_have_low_first_episode_non_tail_signal": bool(
                recovered
                and residual_missed
                and float(residual_summary["mean_first_episode_non_tail_share"])
                < float(recovered_summary["mean_first_episode_non_tail_share"])
            ),
            "residual_missed_anchors_have_low_or_zero_positive_mass": bool(
                recovered
                and residual_missed
                and float(residual_summary["mean_base_positive_mass"])
                < float(recovered_summary["mean_base_positive_mass"])
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Probe whether adding a first-episode/non-reset coverage channel reduces strong gain anchor misses."
    )
    parser.add_argument("--suite-specs-json", type=str, default=None)
    parser.add_argument("--strong-gain-min-pre-gap", type=float, default=1.0)
    parser.add_argument("--coverage-share-threshold", type=float, default=0.05)
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output_json).expanduser().resolve()
    if output_path.exists() and not bool(args.overwrite):
        print(output_path.read_text())
        return 0

    suite_specs = DEFAULT_POSITIVE_SUITES
    if args.suite_specs_json is not None:
        suite_specs = json.loads(Path(args.suite_specs_json).expanduser().resolve().read_text())

    result = build_dual_channel_anchor_coverage_probe(
        suite_specs,
        strong_gain_min_pre_gap=float(args.strong_gain_min_pre_gap),
        coverage_share_threshold=float(args.coverage_share_threshold),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(output_path.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
