import argparse
import json
from pathlib import Path
from typing import Any


CONTRACT_EPS = 1e-6


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(float(v) for v in values) / len(values))


def _max_abs(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(max(abs(float(v)) for v in values))


def _load_negative_payload(path: str) -> dict[str, Any]:
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


def _classify_value_peak_drop(*, target_step_change: float, value_stepdown: float) -> str:
    if float(target_step_change) <= 0.0 and float(value_stepdown) > 0.0:
        return "hallucinated_peak_against_target_trend"
    if float(target_step_change) > 0.0 and float(value_stepdown) > float(target_step_change):
        return "amplified_target_peak"
    return "aligned_or_other"


def _build_probe(
    negative_payload: dict[str, Any],
    gae_payload: dict[str, Any],
    *,
    suite_name: str,
    env_index: int,
    target_locals: list[int],
) -> dict[str, Any]:
    env_rows = _load_env_rows(gae_payload, suite_name=suite_name, env_index=env_index)
    row_by_local = {int(row["objective_local_index"]): dict(row) for row in env_rows}

    env_std = float(negative_payload["guardrails"]["inferred_env_std"])
    full_mean_over_std = float(negative_payload["guardrails"]["inferred_mean_over_std_from_terminal_reset"])
    full_mean_raw = float(env_std * full_mean_over_std)

    cluster_rows = []
    class_counts: dict[str, int] = {}
    for local_idx in target_locals:
        if int(local_idx) - 1 not in row_by_local or int(local_idx) not in row_by_local or int(local_idx) + 1 not in row_by_local:
            raise ValueError(f"Missing prev/curr/next rows around local {local_idx}.")
        prev_row = dict(row_by_local[int(local_idx) - 1])
        curr_row = dict(row_by_local[int(local_idx)])
        next_row = dict(row_by_local[int(local_idx) + 1])

        prev_target = float((float(prev_row["rollout_return_raw"]) - full_mean_raw) / env_std)
        curr_target = float((float(curr_row["rollout_return_raw"]) - full_mean_raw) / env_std)
        next_target = float((float(next_row["rollout_return_raw"]) - full_mean_raw) / env_std)

        target_prev_gap = float(curr_target - prev_target)
        target_step_change = float(curr_target - next_target)
        value_prev_gap = float(float(curr_row["value_norm"]) - float(prev_row["value_norm"]))
        value_stepdown = float(float(curr_row["value_norm"]) - float(curr_row["next_value_norm"]))
        curr_error = float(float(curr_row["value_norm"]) - curr_target)
        next_error = float(float(curr_row["next_value_norm"]) - next_target)
        error_drop = float(curr_error - next_error)
        stepdown_excess = float(value_stepdown - target_step_change)
        delta_reconstructed = float(float(curr_row["reward_norm"]) - value_stepdown)
        pattern_class = _classify_value_peak_drop(
            target_step_change=target_step_change,
            value_stepdown=value_stepdown,
        )
        class_counts[pattern_class] = int(class_counts.get(pattern_class, 0) + 1)

        cluster_rows.append(
            {
                "objective_local_index": int(local_idx),
                "global_step": int(curr_row["global_step"]),
                "next_non_terminal": float(curr_row["next_non_terminal"]),
                "reward_norm": float(curr_row["reward_norm"]),
                "delta_norm": float(curr_row["delta_norm"]),
                "delta_reconstructed_from_reward_minus_value_stepdown": float(delta_reconstructed),
                "delta_reconstruction_error": float(delta_reconstructed - float(curr_row["delta_norm"])),
                "rollout_return_raw": float(curr_row["rollout_return_raw"]),
                "next_rollout_return_raw": float(next_row["rollout_return_raw"]),
                "prev_target_norm": float(prev_target),
                "target_norm": float(curr_target),
                "next_target_norm": float(next_target),
                "value_norm": float(curr_row["value_norm"]),
                "next_value_norm": float(curr_row["next_value_norm"]),
                "curr_value_error_norm": float(curr_error),
                "next_value_error_norm": float(next_error),
                "target_prev_gap_norm": float(target_prev_gap),
                "target_step_change_norm": float(target_step_change),
                "reward_vs_target_step_change_error": float(target_step_change - float(curr_row["reward_norm"])),
                "value_prev_gap_norm": float(value_prev_gap),
                "value_stepdown_norm": float(value_stepdown),
                "stepdown_excess_over_target_norm": float(stepdown_excess),
                "error_drop_norm": float(error_drop),
                "stepdown_excess_minus_error_drop": float(stepdown_excess - error_drop),
                "pattern_class": str(pattern_class),
                "peak_amplification_factor": (
                    None
                    if float(target_step_change) <= CONTRACT_EPS
                    else float(value_stepdown / target_step_change)
                ),
            }
        )

    aggregate = {
        "token_count": int(len(cluster_rows)),
        "pattern_counts": dict(sorted(class_counts.items())),
        "mean_target_step_change_norm": _mean([float(row["target_step_change_norm"]) for row in cluster_rows]),
        "mean_value_stepdown_norm": _mean([float(row["value_stepdown_norm"]) for row in cluster_rows]),
        "mean_stepdown_excess_over_target_norm": _mean(
            [float(row["stepdown_excess_over_target_norm"]) for row in cluster_rows]
        ),
        "mean_curr_value_error_norm": _mean([float(row["curr_value_error_norm"]) for row in cluster_rows]),
        "mean_next_value_error_norm": _mean([float(row["next_value_error_norm"]) for row in cluster_rows]),
        "mean_raw_future_return_after_token": _mean([float(row["next_rollout_return_raw"]) for row in cluster_rows]),
        "hallucinated_peak_count": int(
            sum(1 for row in cluster_rows if str(row["pattern_class"]) == "hallucinated_peak_against_target_trend")
        ),
        "amplified_target_peak_count": int(
            sum(1 for row in cluster_rows if str(row["pattern_class"]) == "amplified_target_peak")
        ),
        "positive_target_step_change_count": int(
            sum(1 for row in cluster_rows if float(row["target_step_change_norm"]) > 0.0)
        ),
        "positive_future_raw_return_count": int(
            sum(1 for row in cluster_rows if float(row["next_rollout_return_raw"]) > 0.0)
        ),
        "max_reward_vs_target_step_change_error": _max_abs(
            [float(row["reward_vs_target_step_change_error"]) for row in cluster_rows]
        ),
        "max_stepdown_excess_minus_error_drop": _max_abs(
            [float(row["stepdown_excess_minus_error_drop"]) for row in cluster_rows]
        ),
        "max_delta_reconstruction_error": _max_abs(
            [float(row["delta_reconstruction_error"]) for row in cluster_rows]
        ),
    }

    return {
        "audit_entry": "phase3_env12_value_peak_drop_contract_probe",
        "config": {
            "source_negative_delta_json": str(
                Path("/home/chen/RLPFN/artifacts/phase3_env12_negative_delta_probe.json").expanduser().resolve()
            ),
            "source_gae_path_json": str(
                Path("/home/chen/RLPFN/artifacts/phase3_residual_anchor_gae_path_probe.json").expanduser().resolve()
            ),
            "suite_name": str(suite_name),
            "env_index": int(env_index),
            "target_locals": [int(v) for v in target_locals],
        },
        "full_rollout_contract": {
            "full_rollout_mean_raw": float(full_mean_raw),
            "full_rollout_std_raw": float(env_std),
            "full_rollout_mean_over_std": float(full_mean_over_std),
        },
        "cluster_rows": cluster_rows,
        "aggregate": aggregate,
        "conclusions": {
            "reward_matches_target_step_change_under_gamma_one_contract": bool(
                float(aggregate["max_reward_vs_target_step_change_error"]) <= 5e-5
            ),
            "delta_matches_reward_minus_value_stepdown_contract": bool(
                float(aggregate["max_delta_reconstruction_error"]) <= 5e-5
            ),
            "stepdown_excess_equals_error_drop_contract": bool(
                float(aggregate["max_stepdown_excess_minus_error_drop"]) <= 5e-5
            ),
            "all_tokens_keep_positive_future_raw_return": bool(
                int(aggregate["positive_future_raw_return_count"]) == int(aggregate["token_count"])
            ),
            "local29_is_hallucinated_peak_against_target": bool(
                any(
                    int(row["objective_local_index"]) == 29
                    and str(row["pattern_class"]) == "hallucinated_peak_against_target_trend"
                    for row in cluster_rows
                )
            ),
            "locals38_51_are_amplified_target_peaks": bool(
                all(
                    str(next(row for row in cluster_rows if int(row["objective_local_index"]) == local_idx)["pattern_class"])
                    == "amplified_target_peak"
                    for local_idx in [38, 51]
                )
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Explain why env12 locals 29/38/51 form value-head peak-then-drop under the current contract."
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

    negative_payload = _load_negative_payload(args.negative_delta_json)
    gae_payload = _load_gae_payload(args.gae_path_json)
    result = _build_probe(
        negative_payload=negative_payload,
        gae_payload=gae_payload,
        suite_name=str(args.suite_name),
        env_index=int(args.env_index),
        target_locals=[29, 38, 51],
    )
    result["config"]["source_negative_delta_json"] = str(Path(args.negative_delta_json).expanduser().resolve())
    result["config"]["source_gae_path_json"] = str(Path(args.gae_path_json).expanduser().resolve())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(output_path.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
