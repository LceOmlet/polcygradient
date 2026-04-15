import argparse
import json
from pathlib import Path
from typing import Any


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(float(v) for v in values) / len(values))


def _load_negative_delta_payload(path: str) -> dict[str, Any]:
    payload = json.loads(Path(path).expanduser().resolve().read_text())
    if str(payload.get("audit_entry", "")) != "phase3_env12_negative_delta_probe":
        raise ValueError("Expected phase3_env12_negative_delta_probe artifact.")
    return payload


def _load_gae_payload(path: str) -> dict[str, Any]:
    payload = json.loads(Path(path).expanduser().resolve().read_text())
    if str(payload.get("audit_entry", "")) != "phase3_residual_anchor_gae_path_probe":
        raise ValueError("Expected phase3_residual_anchor_gae_path_probe artifact.")
    return payload


def _load_env_rows(gae_payload: dict[str, Any], suite_name: str, env_index: int) -> list[dict[str, Any]]:
    for bundle in list(gae_payload.get("token_row_bundles", [])):
        if str(bundle.get("suite_name", "")) == str(suite_name) and int(bundle.get("env_index", -1)) == int(env_index):
            rows = [dict(row) for row in list(bundle.get("token_rows", []))]
            rows.sort(key=lambda row: int(row["objective_local_index"]))
            return rows
    raise ValueError(f"Could not find token rows for {suite_name} env{env_index}.")


def _load_focus_rows_map(negative_payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(row["objective_local_index"]): dict(row) for row in list(negative_payload.get("focus_rows", []))}


def _build_local56_section(
    *,
    env_rows: list[dict[str, Any]],
    focus_rows: dict[int, dict[str, Any]],
    env_std: float,
    mean_over_std: float,
) -> dict[str, Any]:
    row_by_local = {int(row["objective_local_index"]): dict(row) for row in env_rows}
    row56 = dict(focus_rows[56])
    row56_env = dict(row_by_local[56])
    row55 = dict(row_by_local[55])
    row57 = dict(row_by_local[57])
    row101 = dict(row_by_local[101])

    local56_boundary_to_reward_ratio = abs(float(row56["boundary_shift_norm"])) / max(
        abs(float(row56["reward_over_std"])),
        1e-12,
    )
    local56_boundary_share_of_reward_norm_mag = abs(float(row56["boundary_shift_norm"])) / max(
        abs(float(row56["reward_norm"])),
        1e-12,
    )
    local56_boundary_share_of_delta_mag = abs(float(row56["boundary_shift_norm"])) / max(
        abs(float(row56["delta_norm"])),
        1e-12,
    )
    row101_reward_over_std = float(float(row101["reward_raw"]) / float(env_std))
    row101_boundary_shift_norm = float(float(row101["reward_norm"]) - row101_reward_over_std)
    row101_bootstrap_term_norm = float(
        row101.get("bootstrap_term_norm", float(row101["delta_norm"]) - float(row101["reward_norm"]))
    )

    window_rows = [
        {
            "objective_local_index": int(row["objective_local_index"]),
            "next_non_terminal": float(row["next_non_terminal"]),
            "reward_norm": float(row["reward_norm"]),
            "value_norm": float(row["value_norm"]),
            "next_value_norm": float(row["next_value_norm"]),
            "delta_norm": float(row["delta_norm"]),
            "flip_cause": str(row["flip_cause"]),
        }
        for row in [row55, row56_env, row57]
    ]

    return {
        "window_rows": window_rows,
        "terminal_row": {
            "objective_local_index": 56,
            "reward_over_std": float(row56["reward_over_std"]),
            "boundary_shift_norm": float(row56["boundary_shift_norm"]),
            "reward_norm": float(row56["reward_norm"]),
            "bootstrap_term_norm": float(row56["bootstrap_term_norm"]),
            "delta_norm": float(row56["delta_norm"]),
            "aggregated_negative_source_contribution": float(row56["aggregated_negative_source_contribution"]),
        },
        "nonterminal_contrast_row": {
                "objective_local_index": 101,
                "next_non_terminal": float(row101["next_non_terminal"]),
                "reward_over_std": row101_reward_over_std,
                "boundary_shift_norm": row101_boundary_shift_norm,
                "reward_norm": float(row101["reward_norm"]),
                "bootstrap_term_norm": row101_bootstrap_term_norm,
                "delta_norm": float(row101["delta_norm"]),
            },
        "env_constant_stats": {
            "env_std": float(env_std),
            "mean_over_std": float(mean_over_std),
        },
        "derived_metrics": {
            "boundary_to_immediate_reward_ratio": float(local56_boundary_to_reward_ratio),
            "boundary_share_of_reward_norm_magnitude": float(local56_boundary_share_of_reward_norm_mag),
            "boundary_share_of_delta_magnitude": float(local56_boundary_share_of_delta_mag),
            "local56_boundary_shift_present_only_on_terminal_reset": bool(
                abs(float(row55["reward_norm"]) - float(row55["reward_raw"]) / float(env_std)) < 1e-12
                and abs(float(row57["reward_norm"]) - float(row57["reward_raw"]) / float(env_std)) < 1e-12
                and abs(float(row56["boundary_shift_norm"])) > 0.0
            ),
        },
        "conclusions": {
            "local56_boundary_shift_dwarfs_immediate_reward": bool(local56_boundary_to_reward_ratio > 10.0),
            "local56_boundary_shift_is_most_of_reward_norm": bool(local56_boundary_share_of_reward_norm_mag > 0.9),
            "local56_boundary_shift_is_most_of_delta_magnitude": bool(local56_boundary_share_of_delta_mag > 0.8),
            "local101_has_no_terminal_boundary_shift": bool(abs(row101_boundary_shift_norm) < 1e-12),
        },
    }


def _build_cluster_section(
    *,
    env_rows: list[dict[str, Any]],
    focus_rows: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    row_by_local = {int(row["objective_local_index"]): dict(row) for row in env_rows}
    target_locals = [29, 38, 51]
    rows = []
    for local_idx in target_locals:
        prev_row = dict(row_by_local[local_idx - 1])
        curr_row = dict(focus_rows[local_idx])
        next_row = dict(row_by_local[local_idx + 1])
        rebound_two_steps = dict(row_by_local[local_idx + 2]) if (local_idx + 2) in row_by_local else None
        rebound_three_steps = dict(row_by_local[local_idx + 3]) if (local_idx + 3) in row_by_local else None
        prev_gap = float(float(curr_row["value_norm"]) - float(prev_row["value_norm"]))
        next_drop = float(float(curr_row["value_norm"]) - float(next_row["value_norm"]))
        rebound_within_2_steps = bool(
            rebound_two_steps is not None and float(rebound_two_steps["value_norm"]) > float(curr_row["value_norm"])
        )
        rebound_within_3_steps = bool(
            rebound_three_steps is not None and float(rebound_three_steps["value_norm"]) > float(curr_row["value_norm"])
        )
        late_preterminal = bool(int(curr_row["tokens_to_objective_episode_end"]) <= 5)
        is_local_peak = bool(
            float(curr_row["value_norm"]) > float(prev_row["value_norm"])
            and float(curr_row["value_norm"]) > float(next_row["value_norm"])
        )
        pattern = "isolated_peak"
        if is_local_peak and rebound_within_2_steps:
            pattern = "oscillatory_peak"
        elif is_local_peak and late_preterminal:
            pattern = "late_preterminal_peak"
        rows.append(
            {
                "objective_local_index": int(local_idx),
                "prev_value_norm": float(prev_row["value_norm"]),
                "value_norm": float(curr_row["value_norm"]),
                "next_value_norm": float(next_row["value_norm"]),
                "prev_gap_norm": prev_gap,
                "next_drop_norm": next_drop,
                "reward_norm": float(curr_row["reward_norm"]),
                "bootstrap_term_norm": float(curr_row["bootstrap_term_norm"]),
                "delta_norm": float(curr_row["delta_norm"]),
                "aggregated_negative_source_contribution": float(curr_row["aggregated_negative_source_contribution"]),
                "tokens_to_objective_episode_end": int(curr_row["tokens_to_objective_episode_end"]),
                "is_local_peak": is_local_peak,
                "rebound_within_2_steps_above_current": rebound_within_2_steps,
                "rebound_within_3_steps_above_current": rebound_within_3_steps,
                "late_preterminal": late_preterminal,
                "pattern_class": str(pattern),
            }
        )
    pattern_counts: dict[str, int] = {}
    for row in rows:
        pattern = str(row["pattern_class"])
        pattern_counts[pattern] = int(pattern_counts.get(pattern, 0) + 1)

    return {
        "cluster_rows": rows,
        "aggregate": {
            "token_count": int(len(rows)),
            "all_local_peaks": bool(all(bool(row["is_local_peak"]) for row in rows)),
            "mean_prev_gap_norm": _mean([float(row["prev_gap_norm"]) for row in rows]),
            "mean_next_drop_norm": _mean([float(row["next_drop_norm"]) for row in rows]),
            "mean_reward_norm": _mean([float(row["reward_norm"]) for row in rows]),
            "mean_bootstrap_term_norm": _mean([float(row["bootstrap_term_norm"]) for row in rows]),
            "mean_delta_norm": _mean([float(row["delta_norm"]) for row in rows]),
            "mean_aggregated_negative_source_contribution": _mean(
                [float(row["aggregated_negative_source_contribution"]) for row in rows]
            ),
            "positive_reward_and_negative_delta_count": int(
                sum(1 for row in rows if float(row["reward_norm"]) > 0.0 and float(row["delta_norm"]) < 0.0)
            ),
            "oscillatory_peak_count": int(sum(1 for row in rows if str(row["pattern_class"]) == "oscillatory_peak")),
            "late_preterminal_peak_count": int(
                sum(1 for row in rows if str(row["pattern_class"]) == "late_preterminal_peak")
            ),
            "pattern_counts": dict(sorted(pattern_counts.items())),
        },
        "conclusions": {
            "cluster_tokens_are_local_value_peaks": bool(all(bool(row["is_local_peak"]) for row in rows)),
            "cluster_stepdown_exceeds_reward_on_average": bool(
                abs(_mean([float(row["bootstrap_term_norm"]) for row in rows]))
                > abs(_mean([float(row["reward_norm"]) for row in rows]))
            ),
            "cluster_contains_positive_reward_peak_failures": bool(
                sum(1 for row in rows if float(row["reward_norm"]) > 0.0 and float(row["delta_norm"]) < 0.0) >= 2
            ),
            "cluster_is_not_single_pattern": bool(len(pattern_counts) > 1),
        },
    }


def _build_probe(
    negative_payload: dict[str, Any],
    gae_payload: dict[str, Any],
    *,
    suite_name: str,
    env_index: int,
) -> dict[str, Any]:
    env_rows = _load_env_rows(gae_payload, suite_name=suite_name, env_index=env_index)
    focus_rows = _load_focus_rows_map(negative_payload)
    env_std = float(negative_payload["guardrails"]["inferred_env_std"])
    mean_over_std = float(negative_payload["guardrails"]["inferred_mean_over_std_from_terminal_reset"])

    local56_section = _build_local56_section(
        env_rows=env_rows,
        focus_rows=focus_rows,
        env_std=env_std,
        mean_over_std=mean_over_std,
    )
    cluster_section = _build_cluster_section(
        env_rows=env_rows,
        focus_rows=focus_rows,
    )
    conclusions = {
        "local56_terminal_boundary_shift_confirmed": bool(
            local56_section["conclusions"]["local56_boundary_shift_dwarfs_immediate_reward"]
            and local56_section["conclusions"]["local56_boundary_shift_is_most_of_delta_magnitude"]
        ),
        "local56_terminal_shift_absent_on_nonterminal_contrast": bool(
            local56_section["conclusions"]["local101_has_no_terminal_boundary_shift"]
        ),
        "cluster_local_value_peak_stepdown_confirmed": bool(
            cluster_section["conclusions"]["cluster_tokens_are_local_value_peaks"]
            and cluster_section["conclusions"]["cluster_stepdown_exceeds_reward_on_average"]
        ),
        "cluster_contains_multiple_subpatterns": bool(cluster_section["conclusions"]["cluster_is_not_single_pattern"]),
    }
    return {
        "audit_entry": "phase3_env12_boundary_stepdown_root_probe",
        "config": {
            "source_negative_delta_json": str(
                Path("/home/chen/RLPFN/artifacts/phase3_env12_negative_delta_probe.json").expanduser().resolve()
            ),
            "source_gae_path_json": str(
                Path("/home/chen/RLPFN/artifacts/phase3_residual_anchor_gae_path_probe.json").expanduser().resolve()
            ),
            "suite_name": str(suite_name),
            "env_index": int(env_index),
        },
        "local56_boundary": local56_section,
        "cluster_stepdown": cluster_section,
        "conclusions": conclusions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Explain env12 local56 terminal boundary shift and local 29/38/51 bootstrap stepdown roots."
    )
    parser.add_argument(
        "--negative-delta-json",
        type=str,
        default="/home/chen/RLPFN/artifacts/phase3_env12_negative_delta_probe.json",
    )
    parser.add_argument(
        "--gae-path-json",
        type=str,
        default="/home/chen/RLPFN/artifacts/phase3_residual_anchor_gae_path_probe.json",
    )
    parser.add_argument("--suite-name", type=str, default="pair2")
    parser.add_argument("--env-index", type=int, default=12)
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output_json).expanduser().resolve()
    if output_path.exists() and not bool(args.overwrite):
        print(output_path.read_text())
        return 0

    negative_payload = _load_negative_delta_payload(args.negative_delta_json)
    gae_payload = _load_gae_payload(args.gae_path_json)
    result = _build_probe(
        negative_payload=negative_payload,
        gae_payload=gae_payload,
        suite_name=str(args.suite_name),
        env_index=int(args.env_index),
    )
    result["config"]["source_negative_delta_json"] = str(Path(args.negative_delta_json).expanduser().resolve())
    result["config"]["source_gae_path_json"] = str(Path(args.gae_path_json).expanduser().resolve())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(output_path.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
