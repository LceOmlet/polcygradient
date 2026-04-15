import argparse
import json
from pathlib import Path
from typing import Any


TAIL_LEN_KEY = "8"
DIST_KEYS = tuple(f"post_reset_tail_dist_{idx}" for idx in range(1, 5))

DEFAULT_POSITIVE_SUITES = [
    {
        "suite_name": "pair1",
        "zero_json": "/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/zero_cross_env_control.json",
        "ppo_json": "/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/ppo_cross_env_baseline.json",
        "distance_probe_json": "/home/chen/RLPFN/artifacts/phase3_post_reset_tail_distance_probe_pair1_suite_matched.json",
        "tail_mask_probe_json": "/home/chen/RLPFN/artifacts/phase3_terminal_tail_mask_probe_pair1_suite_matched.json",
    },
    {
        "suite_name": "pair2",
        "zero_json": "/home/chen/RLPFN/artifacts/phase3_cross_env_baseline_pair2/zero_cross_env_control.json",
        "ppo_json": "/home/chen/RLPFN/artifacts/phase3_cross_env_baseline_pair2/ppo_cross_env_baseline.json",
        "distance_probe_json": "/home/chen/RLPFN/artifacts/phase3_post_reset_tail_distance_probe_pair2.json",
        "tail_mask_probe_json": "/home/chen/RLPFN/artifacts/phase3_terminal_tail_mask_probe_pair2.json",
    },
    {
        "suite_name": "seed13579",
        "zero_json": "/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_13579/zero_cross_env_control.json",
        "ppo_json": "/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_13579/ppo_cross_env_baseline.json",
        "distance_probe_json": "/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_13579/phase3_post_reset_tail_distance_probe.json",
        "tail_mask_probe_json": "/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_13579/phase3_terminal_tail_mask_probe.json",
    },
]

REMOVED_KEYS = (
    "post_reset_terminal_tail",
    "post_reset_only",
    "first_episode_only",
    "terminal_tail",
)


def _load_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).expanduser().resolve().read_text())


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(float(v) for v in values) / len(values))


def _suite_positive(zero_payload: dict[str, Any], ppo_payload: dict[str, Any]) -> tuple[bool, float]:
    zero_suffix = float(zero_payload["heldout_suite"]["metrics"]["suffix_return_mean"])
    ppo_suffix = float(ppo_payload["heldout_suite"]["metrics"]["suffix_return_mean"])
    delta = float(ppo_suffix - zero_suffix)
    return bool(delta > 0.0), delta


def _rank_desc(rows: list[dict[str, Any]], key: str) -> dict[int, int]:
    ordered = sorted(rows, key=lambda row: float(row[key]), reverse=True)
    return {int(row["env_index"]): int(rank) for rank, row in enumerate(ordered, start=1)}


def _removed_share(base_positive_mass: float, removed_positive_mass: float) -> float:
    if float(base_positive_mass) <= 0.0:
        return 0.0
    return float(float(removed_positive_mass) / float(base_positive_mass))


def _build_suite_rows(spec: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    zero_payload = _load_json(str(spec["zero_json"]))
    ppo_payload = _load_json(str(spec["ppo_json"]))
    distance_payload = _load_json(str(spec["distance_probe_json"]))
    tail_payload = _load_json(str(spec["tail_mask_probe_json"]))

    is_positive, suffix_delta = _suite_positive(zero_payload, ppo_payload)
    if not is_positive:
        return [], {
            "suite_name": str(spec["suite_name"]),
            "available": True,
            "positive_regime": False,
            "ppo_minus_zero_suffix": float(suffix_delta),
        }

    zero_metrics = zero_payload["heldout_suite"]["metrics"]
    ppo_metrics = ppo_payload["heldout_suite"]["metrics"]
    tail_by_env = {int(row["env_index"]): row for row in tail_payload["env_rows"]}

    suite_rows: list[dict[str, Any]] = []
    for row in distance_payload["env_rows"]:
        env_index = int(row["env_index"])
        tail_row = tail_by_env[env_index]
        dist_mass = float(sum(float(row[key]) for key in DIST_KEYS))
        base_positive_mass = float(tail_row["base_positive_mass"])
        tail_len_row = tail_row["tail_len_rows"][TAIL_LEN_KEY]
        out = {
            "suite_name": str(spec["suite_name"]),
            "env_index": env_index,
            "env_seed": int(row["env_seed"]),
            "rollout_seed": int(row["rollout_seed"]),
            "objective_terminal_reset_count": int(row["objective_terminal_reset_count"]),
            "objective_token_count": int(row["objective_token_count"]),
            "pre_vs_zero_suffix_gap": float(row["pre_vs_zero_suffix_gap"]),
            "zero_suffix_return": float(zero_metrics["suffix_return_per_env"][env_index]),
            "ppo_suffix_return": float(ppo_metrics["suffix_return_per_env"][env_index]),
            "suffix_gap": float(float(ppo_metrics["suffix_return_per_env"][env_index]) - float(zero_metrics["suffix_return_per_env"][env_index])),
            "zero_full_return": float(zero_metrics["full_return_per_env"][env_index]),
            "ppo_full_return": float(ppo_metrics["full_return_per_env"][env_index]),
            "full_gap": float(float(ppo_metrics["full_return_per_env"][env_index]) - float(zero_metrics["full_return_per_env"][env_index])),
            "dist1_to_4_mass": dist_mass,
            "base_positive_mass": base_positive_mass,
        }
        for removed_key in REMOVED_KEYS:
            removed_mass = float(tail_len_row[removed_key]["removed_positive_mass"])
            out[f"{removed_key}_removed_positive_mass"] = removed_mass
            out[f"{removed_key}_removed_share"] = _removed_share(base_positive_mass, removed_mass)
        suite_rows.append(out)

    total_mass = float(sum(float(row["dist1_to_4_mass"]) for row in suite_rows))
    pre_rank = _rank_desc(suite_rows, "pre_vs_zero_suffix_gap")
    mass_rank = _rank_desc(suite_rows, "dist1_to_4_mass")
    for row in suite_rows:
        env_index = int(row["env_index"])
        row["pre_gap_rank_within_suite"] = int(pre_rank[env_index])
        row["dist_mass_rank_within_suite"] = int(mass_rank[env_index])
        row["dist_mass_share_within_suite"] = (
            float(float(row["dist1_to_4_mass"]) / total_mass) if total_mass > 0.0 else 0.0
        )

    return suite_rows, {
        "suite_name": str(spec["suite_name"]),
        "available": True,
        "positive_regime": True,
        "ppo_minus_zero_suffix": float(suffix_delta),
        "env_count": int(len(suite_rows)),
        "total_dist1_to_4_mass": total_mass,
    }


def _group_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keep_fields = [
        "suite_name",
        "env_index",
        "env_seed",
        "rollout_seed",
        "objective_terminal_reset_count",
        "objective_token_count",
        "pre_vs_zero_suffix_gap",
        "pre_gap_rank_within_suite",
        "full_gap",
        "dist1_to_4_mass",
        "dist_mass_rank_within_suite",
        "dist_mass_share_within_suite",
        "base_positive_mass",
        "post_reset_terminal_tail_removed_share",
        "post_reset_only_removed_share",
        "first_episode_only_removed_share",
        "terminal_tail_removed_share",
    ]
    return {
        "count": int(len(rows)),
        "suite_counts": {
            suite_name: int(sum(1 for row in rows if str(row["suite_name"]) == suite_name))
            for suite_name in sorted({str(row["suite_name"]) for row in rows})
        },
        "mean_pre_gap": _mean([float(row["pre_vs_zero_suffix_gap"]) for row in rows]),
        "mean_full_gap": _mean([float(row["full_gap"]) for row in rows]),
        "mean_dist1_to_4_mass": _mean([float(row["dist1_to_4_mass"]) for row in rows]),
        "mean_dist_mass_share_within_suite": _mean([float(row["dist_mass_share_within_suite"]) for row in rows]),
        "mean_reset_count": _mean([float(row["objective_terminal_reset_count"]) for row in rows]),
        "mean_base_positive_mass": _mean([float(row["base_positive_mass"]) for row in rows]),
        "mean_post_reset_terminal_tail_removed_share": _mean(
            [float(row["post_reset_terminal_tail_removed_share"]) for row in rows]
        ),
        "mean_post_reset_only_removed_share": _mean(
            [float(row["post_reset_only_removed_share"]) for row in rows]
        ),
        "mean_first_episode_only_removed_share": _mean(
            [float(row["first_episode_only_removed_share"]) for row in rows]
        ),
        "mean_terminal_tail_removed_share": _mean(
            [float(row["terminal_tail_removed_share"]) for row in rows]
        ),
        "env_items": [{field: row[field] for field in keep_fields} for row in sorted(
            rows,
            key=lambda row: (float(row["pre_vs_zero_suffix_gap"]), float(row["dist1_to_4_mass"])),
            reverse=True,
        )],
    }


def build_positive_regime_anchor_structure_probe(
    suite_specs: list[dict[str, Any]],
    *,
    anchor_min_pre_gap: float,
    captured_min_mass_share_within_suite: float,
) -> dict[str, Any]:
    suite_summaries = []
    positive_rows: list[dict[str, Any]] = []
    skipped_suites = []
    for spec in suite_specs:
        try:
            suite_rows, suite_summary = _build_suite_rows(spec)
        except FileNotFoundError:
            skipped_suites.append(
                {
                    "suite_name": str(spec.get("suite_name", "unknown")),
                    "missing": True,
                }
            )
            continue
        suite_summaries.append(suite_summary)
        if bool(suite_summary.get("positive_regime", False)):
            positive_rows.extend(suite_rows)

    strong_gain_anchors = [
        row for row in positive_rows if float(row["pre_vs_zero_suffix_gap"]) > float(anchor_min_pre_gap)
    ]
    captured = [
        row
        for row in strong_gain_anchors
        if float(row["dist_mass_share_within_suite"]) >= float(captured_min_mass_share_within_suite)
    ]
    missed = [
        row
        for row in strong_gain_anchors
        if float(row["dist_mass_share_within_suite"]) < float(captured_min_mass_share_within_suite)
    ]

    captured_summary = _group_summary(captured)
    missed_summary = _group_summary(missed)
    deltas = {
        "mean_pre_gap_delta_missed_minus_captured": float(
            missed_summary["mean_pre_gap"] - captured_summary["mean_pre_gap"]
        ),
        "mean_full_gap_delta_missed_minus_captured": float(
            missed_summary["mean_full_gap"] - captured_summary["mean_full_gap"]
        ),
        "mean_dist_mass_share_delta_missed_minus_captured": float(
            missed_summary["mean_dist_mass_share_within_suite"] - captured_summary["mean_dist_mass_share_within_suite"]
        ),
        "mean_reset_count_delta_missed_minus_captured": float(
            missed_summary["mean_reset_count"] - captured_summary["mean_reset_count"]
        ),
        "mean_base_positive_mass_delta_missed_minus_captured": float(
            missed_summary["mean_base_positive_mass"] - captured_summary["mean_base_positive_mass"]
        ),
        "mean_post_reset_tail_removed_share_delta_missed_minus_captured": float(
            missed_summary["mean_post_reset_terminal_tail_removed_share"]
            - captured_summary["mean_post_reset_terminal_tail_removed_share"]
        ),
        "mean_first_episode_only_removed_share_delta_missed_minus_captured": float(
            missed_summary["mean_first_episode_only_removed_share"]
            - captured_summary["mean_first_episode_only_removed_share"]
        ),
    }

    return {
        "audit_entry": "phase3_positive_regime_anchor_structure_probe",
        "config": {
            "anchor_min_pre_gap": float(anchor_min_pre_gap),
            "captured_min_mass_share_within_suite": float(captured_min_mass_share_within_suite),
            "tail_len_key": TAIL_LEN_KEY,
            "suite_names": [str(spec["suite_name"]) for spec in suite_specs],
        },
        "suite_summaries": suite_summaries,
        "skipped_suites": skipped_suites,
        "strong_gain_anchor_count": int(len(strong_gain_anchors)),
        "captured_summary": captured_summary,
        "missed_summary": missed_summary,
        "deltas": deltas,
        "conclusions": {
            "captured_gain_anchors_exist": bool(captured),
            "proxy_missed_gain_anchors_exist": bool(missed),
            "missed_gain_anchors_still_positive_on_full_return": bool(
                missed and float(missed_summary["mean_full_gap"]) > 0.0
            ),
            "captured_anchors_are_more_reset_heavy": bool(
                captured and missed and float(captured_summary["mean_reset_count"]) > float(missed_summary["mean_reset_count"])
            ),
            "captured_anchors_are_more_post_reset_tail_weighted": bool(
                captured
                and missed
                and float(captured_summary["mean_post_reset_terminal_tail_removed_share"])
                > float(missed_summary["mean_post_reset_terminal_tail_removed_share"])
            ),
            "missed_anchors_are_more_first_episode_skewed": bool(
                captured
                and missed
                and float(missed_summary["mean_first_episode_only_removed_share"])
                > float(captured_summary["mean_first_episode_only_removed_share"])
            ),
            "current_proxy_underweights_real_gain_anchors": bool(
                bool(missed)
                and float(missed_summary["mean_full_gap"]) > 0.0
                and float(missed_summary["mean_dist_mass_share_within_suite"])
                < float(captured_summary["mean_dist_mass_share_within_suite"])
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare proxy-captured vs proxy-missed strong gain anchors within the positive regime."
    )
    parser.add_argument("--suite-specs-json", type=str, default=None)
    parser.add_argument("--anchor-min-pre-gap", type=float, default=1.0)
    parser.add_argument("--captured-min-mass-share-within-suite", type=float, default=0.05)
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

    result = build_positive_regime_anchor_structure_probe(
        suite_specs,
        anchor_min_pre_gap=float(args.anchor_min_pre_gap),
        captured_min_mass_share_within_suite=float(args.captured_min_mass_share_within_suite),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(output_path.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
