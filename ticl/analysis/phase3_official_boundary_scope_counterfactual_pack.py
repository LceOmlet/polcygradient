import argparse
import json
from pathlib import Path
from typing import Any


OFFICIAL_SAVED_POINTS = (
    {
        "point_name": "seed13579_env12",
        "canonical_json": "/home/chen/RLPFN/artifacts/phase3_seed13579_env12_terminal_scale_chain_probe.json",
        "repeat_json": "/home/chen/RLPFN/artifacts/phase3_seed13579_env12_terminal_scale_chain_probe_repeatB.json",
    },
    {
        "point_name": "pair2_env13",
        "canonical_json": "/home/chen/RLPFN/artifacts/phase3_pair2_env13_terminal_scale_chain_probe.json",
        "repeat_json": "/home/chen/RLPFN/artifacts/phase3_pair2_env13_terminal_scale_chain_probe_repeatB.json",
    },
    {
        "point_name": "pair1_env13",
        "canonical_json": "/home/chen/RLPFN/artifacts/phase3_pair1_env13_terminal_scale_chain_probe.json",
        "repeat_json": "/home/chen/RLPFN/artifacts/phase3_pair1_env13_terminal_scale_chain_probe_repeatB.json",
    },
)


def _load_scale_chain_payload(path: str) -> dict[str, Any]:
    payload = json.loads(Path(path).expanduser().resolve().read_text())
    if str(payload.get("audit_entry", "")) != "phase3_terminal_local_scale_chain_probe":
        raise ValueError(f"Expected phase3_terminal_local_scale_chain_probe artifact: {path}")
    return payload


def _config_subset(payload: dict[str, Any]) -> dict[str, Any]:
    config = dict(payload["config"])
    return {
        "suite_name": str(config["suite_name"]),
        "env_index": int(config["env_index"]),
        "selected_target_local_index": int(config["selected_target_local_index"]),
        "single_eval_pos": int(config["single_eval_pos"]),
        "objective_terminal_reset_count_from_distance_probe": int(
            config["objective_terminal_reset_count_from_distance_probe"]
        ),
        "objective_terminal_row_count_from_token_rows": int(config["objective_terminal_row_count_from_token_rows"]),
    }


def _locked_repeat_checks(canonical_payload: dict[str, Any], repeat_payload: dict[str, Any]) -> dict[str, bool]:
    return {
        "config_subset": _config_subset(canonical_payload) == _config_subset(repeat_payload),
        "selected_target_local_readout": canonical_payload["selected_target_local_readout"]
        == repeat_payload["selected_target_local_readout"],
        "boundary_contract": canonical_payload["boundary_contract"] == repeat_payload["boundary_contract"],
        "mean_source": canonical_payload["mean_source"] == repeat_payload["mean_source"],
        "scale_collapse": canonical_payload["scale_collapse"] == repeat_payload["scale_collapse"],
        "conclusions": canonical_payload["conclusions"] == repeat_payload["conclusions"],
    }


def _scope_row_map(section: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["scope_name"]): dict(row) for row in list(section["scope_rows"])}


def _single_terminal_section(payload: dict[str, Any]) -> dict[str, Any]:
    sections = list(payload.get("terminal_local_sections", []))
    if len(sections) != 1:
        raise ValueError("Expected exactly one terminal-local section in official saved point.")
    return dict(sections[0])


def _point_summary(point_name: str, canonical_payload: dict[str, Any], repeat_payload: dict[str, Any]) -> dict[str, Any]:
    repeat_checks = _locked_repeat_checks(canonical_payload, repeat_payload)
    if not all(bool(v) for v in repeat_checks.values()):
        raise ValueError(f"{point_name}: canonical vs repeat did not match on locked fields: {repeat_checks}")

    terminal_section = _single_terminal_section(canonical_payload)
    scope_rows = _scope_row_map(terminal_section)
    current_scope = dict(scope_rows["current_full_rollout"])
    objective_scope = dict(scope_rows["objective_all_tokens"])
    scale_collapse = dict(canonical_payload["scale_collapse"])
    scale_readout = dict(scale_collapse["scale_collapse_readout"])
    full_variance = dict(scale_collapse["full_variance_decomposition"])
    objective_variance = dict(scale_collapse["objective_variance_decomposition"])
    conclusions = dict(canonical_payload["conclusions"])

    current_boundary_abs = abs(float(current_scope["local_boundary_shift_norm"]))
    objective_boundary_abs = abs(float(objective_scope["local_boundary_shift_norm"]))
    current_reward_norm_abs = abs(float(current_scope["local_reward_norm_before_value_term"]))
    objective_reward_norm_abs = abs(float(objective_scope["local_reward_norm_before_value_term"]))

    return {
        "point_name": str(point_name),
        "suite_name": str(canonical_payload["config"]["suite_name"]),
        "env_index": int(canonical_payload["config"]["env_index"]),
        "selected_target_local_index": int(canonical_payload["config"]["selected_target_local_index"]),
        "terminal_global_step": int(canonical_payload["selected_target_local_readout"]["global_step"]),
        "repeat_stability_checks": repeat_checks,
        "runtime_wall_s_total": {
            "canonical": float(canonical_payload["runtime_wall_s_total"]),
            "repeatB": float(repeat_payload["runtime_wall_s_total"]),
        },
        "scope_counterfactual": {
            "current_scope_name": "current_full_rollout",
            "counterfactual_scope_name": "objective_all_tokens",
            "current_boundary_shift_norm": float(current_scope["local_boundary_shift_norm"]),
            "counterfactual_boundary_shift_norm": float(objective_scope["local_boundary_shift_norm"]),
            "current_boundary_abs": float(current_boundary_abs),
            "counterfactual_boundary_abs": float(objective_boundary_abs),
            "boundary_abs_shrink_if_using_full_over_objective": float(objective_boundary_abs - current_boundary_abs),
            "boundary_abs_ratio_current_over_objective": float(current_boundary_abs / objective_boundary_abs),
            "current_reward_norm_before_value_term": float(current_scope["local_reward_norm_before_value_term"]),
            "counterfactual_reward_norm_before_value_term": float(
                objective_scope["local_reward_norm_before_value_term"]
            ),
            "current_reward_norm_abs": float(current_reward_norm_abs),
            "counterfactual_reward_norm_abs": float(objective_reward_norm_abs),
            "reward_norm_abs_shrink_if_using_full_over_objective": float(
                objective_reward_norm_abs - current_reward_norm_abs
            ),
            "reward_norm_abs_ratio_current_over_objective": float(
                current_reward_norm_abs / objective_reward_norm_abs
            ),
        },
        "boundary_contract_core": {
            "full_rollout_mean_raw": float(canonical_payload["boundary_contract"]["full_rollout_mean_raw"]),
            "full_rollout_std_raw": float(canonical_payload["boundary_contract"]["full_rollout_std_raw"]),
            "objective_subset_mean_raw": float(canonical_payload["boundary_contract"]["objective_subset_mean_raw"]),
            "objective_subset_std_raw": float(canonical_payload["boundary_contract"]["objective_subset_std_raw"]),
            "full_rollout_mean_over_std": float(canonical_payload["boundary_contract"]["full_rollout_mean_over_std"]),
            "objective_subset_mean_over_std": float(
                canonical_payload["boundary_contract"]["objective_subset_mean_over_std"]
            ),
            "objective_subset_gap_vs_full_mean_over_std": float(
                canonical_payload["boundary_contract"]["objective_subset_gap_vs_full_mean_over_std"]
            ),
        },
        "scale_collapse_core": {
            "objective_std_ratio_vs_full": float(scale_readout["objective_std_ratio_vs_full"]),
            "objective_variance_ratio_vs_full": float(scale_readout["objective_variance_ratio_vs_full"]),
            "full_to_objective_variance_multiple": float(scale_readout["full_to_objective_variance_multiple"]),
            "within_prefix_share": float(full_variance["within_prefix_share"]),
            "between_prefix_objective_share": float(full_variance["between_prefix_objective_share"]),
            "dominant_objective_variance_source": str(objective_variance["dominant_objective_variance_source"]),
            "between_episode_means_share": float(objective_variance["between_episode_means_share"]),
        },
        "contract_checks": {
            "current_matches_full_rollout_centering_contract": bool(
                conclusions["all_terminal_candidates_match_full_rollout_centering_contract"]
            ),
            "objective_counterfactual_worsens_boundary_penalty": bool(
                conclusions["all_terminal_candidates_worsen_under_objective_scope"]
            ),
            "objective_scope_std_collapse_is_explained_by_truncating_high_variance_prefix": bool(
                conclusions["objective_scope_std_collapse_is_explained_by_truncating_away_high_variance_prefix"]
            ),
            "objective_window_is_low_variance_slice_not_high_variance_source": bool(
                conclusions["objective_window_is_low_variance_slice_not_high_variance_source"]
            ),
        },
    }


def _aggregate(points: list[dict[str, Any]]) -> dict[str, Any]:
    boundary_shrinks = [
        float(point["scope_counterfactual"]["boundary_abs_shrink_if_using_full_over_objective"]) for point in points
    ]
    reward_shrinks = [
        float(point["scope_counterfactual"]["reward_norm_abs_shrink_if_using_full_over_objective"]) for point in points
    ]
    std_ratios = [float(point["scale_collapse_core"]["objective_std_ratio_vs_full"]) for point in points]
    prefix_shares = [float(point["scale_collapse_core"]["within_prefix_share"]) for point in points]
    return {
        "official_point_count": int(len(points)),
        "point_names": [str(point["point_name"]) for point in points],
        "all_points_replay_stable": bool(
            all(all(bool(v) for v in point["repeat_stability_checks"].values()) for point in points)
        ),
        "all_points_current_match_full_rollout_centering_contract": bool(
            all(bool(point["contract_checks"]["current_matches_full_rollout_centering_contract"]) for point in points)
        ),
        "all_points_objective_counterfactual_worsens_boundary_penalty": bool(
            all(bool(point["contract_checks"]["objective_counterfactual_worsens_boundary_penalty"]) for point in points)
        ),
        "all_points_show_std_collapse_from_truncating_high_variance_prefix": bool(
            all(
                bool(
                    point["contract_checks"][
                        "objective_scope_std_collapse_is_explained_by_truncating_high_variance_prefix"
                    ]
                )
                for point in points
            )
        ),
        "all_points_show_low_variance_objective_window": bool(
            all(bool(point["contract_checks"]["objective_window_is_low_variance_slice_not_high_variance_source"]) for point in points)
        ),
        "all_points_would_shrink_boundary_abs_under_full_vs_objective": bool(
            all(float(v) > 0.0 for v in boundary_shrinks)
        ),
        "all_points_would_shrink_reward_norm_abs_under_full_vs_objective": bool(
            all(float(v) > 0.0 for v in reward_shrinks)
        ),
        "boundary_abs_shrink_if_using_full_over_objective": {
            "min": float(min(boundary_shrinks)),
            "max": float(max(boundary_shrinks)),
            "mean": float(sum(boundary_shrinks) / len(boundary_shrinks)),
        },
        "reward_norm_abs_shrink_if_using_full_over_objective": {
            "min": float(min(reward_shrinks)),
            "max": float(max(reward_shrinks)),
            "mean": float(sum(reward_shrinks) / len(reward_shrinks)),
        },
        "objective_std_ratio_vs_full": {
            "min": float(min(std_ratios)),
            "max": float(max(std_ratios)),
            "mean": float(sum(std_ratios) / len(std_ratios)),
        },
        "within_prefix_share": {
            "min": float(min(prefix_shares)),
            "max": float(max(prefix_shares)),
            "mean": float(sum(prefix_shares) / len(prefix_shares)),
        },
    }


def _build_pack() -> dict[str, Any]:
    points = []
    for spec in OFFICIAL_SAVED_POINTS:
        canonical_payload = _load_scale_chain_payload(str(spec["canonical_json"]))
        repeat_payload = _load_scale_chain_payload(str(spec["repeat_json"]))
        points.append(
            _point_summary(
                str(spec["point_name"]),
                canonical_payload=canonical_payload,
                repeat_payload=repeat_payload,
            )
        )

    aggregate = _aggregate(points)
    return {
        "audit_entry": "phase3_official_boundary_scope_counterfactual_pack",
        "config": {
            "official_saved_points": [dict(spec) for spec in OFFICIAL_SAVED_POINTS],
        },
        "point_summaries": points,
        "aggregate": aggregate,
        "conclusions": {
            "official_saved_points_support_full_rollout_centering_as_cross_point_guardrail": bool(
                aggregate["all_points_current_match_full_rollout_centering_contract"]
                and aggregate["all_points_objective_counterfactual_worsens_boundary_penalty"]
                and aggregate["all_points_would_shrink_boundary_abs_under_full_vs_objective"]
            ),
            "counterfactual_pack_supports_small_scope_contract_lock_not_broad_objective_rewrite": bool(
                aggregate["all_points_show_std_collapse_from_truncating_high_variance_prefix"]
                and aggregate["all_points_show_low_variance_objective_window"]
                and aggregate["all_points_would_shrink_boundary_abs_under_full_vs_objective"]
            ),
            "pack_demonstrates_cross_point_coverage_for_boundary_scope_failure_mode": bool(
                aggregate["official_point_count"] >= 3
                and aggregate["all_points_replay_stable"]
                and aggregate["all_points_would_shrink_boundary_abs_under_full_vs_objective"]
            ),
            "pack_proves_current_saved_points_already_use_the_better_contract": bool(
                aggregate["all_points_current_match_full_rollout_centering_contract"]
            ),
            "pack_is_guardrail_evidence_not_yet_new_runtime_fix": bool(
                aggregate["all_points_current_match_full_rollout_centering_contract"]
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read-only counterfactual pack comparing objective-scope vs full-rollout terminal boundary centering across official saved points."
    )
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output_json).expanduser().resolve()
    if output_path.exists() and not bool(args.overwrite):
        print(output_path.read_text())
        return 0

    result = _build_pack()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(output_path.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
