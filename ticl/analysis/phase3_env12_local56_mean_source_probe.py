import argparse
import json
from pathlib import Path
from typing import Any


CONTRACT_EPS = 1e-6


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(float(v) for v in values) / len(values))


def _load_boundary_contract_payload(path: str) -> dict[str, Any]:
    payload = json.loads(Path(path).expanduser().resolve().read_text())
    if str(payload.get("audit_entry", "")) != "phase3_env12_local56_boundary_contract_probe":
        raise ValueError("Expected phase3_env12_local56_boundary_contract_probe artifact.")
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


def _build_probe(
    boundary_payload: dict[str, Any],
    gae_payload: dict[str, Any],
    *,
    suite_name: str,
    env_index: int,
) -> dict[str, Any]:
    env_rows = _load_env_rows(gae_payload, suite_name=suite_name, env_index=env_index)
    if not env_rows:
        raise ValueError("Expected nonempty env rows.")

    full_mean_raw = float(boundary_payload["full_rollout_contract"]["full_rollout_mean_raw"])
    full_std_raw = float(boundary_payload["full_rollout_contract"]["full_rollout_std_raw"])
    full_mean_over_std = float(boundary_payload["full_rollout_contract"]["full_rollout_mean_over_std"])
    n_samples = int(gae_payload["config"]["n_samples"])

    objective_return_values = [float(row["rollout_return_raw"]) for row in env_rows]
    objective_mean_raw = _mean(objective_return_values)
    objective_token_count = int(len(env_rows))
    prefix_token_count = int(n_samples - objective_token_count)
    if prefix_token_count <= 0:
        raise ValueError("Expected positive omitted-prefix token count.")
    implied_prefix_mean_raw = float(
        ((float(n_samples) * full_mean_raw) - (float(objective_token_count) * objective_mean_raw)) / float(prefix_token_count)
    )

    first_row = dict(env_rows[0])
    last_row = dict(env_rows[-1])
    local56 = next(dict(row) for row in env_rows if int(row["objective_local_index"]) == 56)

    rows_by_episode: dict[int, list[dict[str, Any]]] = {}
    for row in env_rows:
        rows_by_episode.setdefault(int(row["objective_episode_index"]), []).append(dict(row))

    episode_summaries = []
    for episode_idx in sorted(rows_by_episode):
        episode_rows = rows_by_episode[episode_idx]
        episode_summaries.append(
            {
                "objective_episode_index": int(episode_idx),
                "token_count": int(len(episode_rows)),
                "mean_rollout_return_raw": float(_mean([float(row["rollout_return_raw"]) for row in episode_rows])),
                "mean_reward_raw": float(_mean([float(row["reward_raw"]) for row in episode_rows])),
                "terminal_row_count": int(sum(1 for row in episode_rows if float(row["next_non_terminal"]) <= 0.0)),
                "first_local_index": int(min(int(row["objective_local_index"]) for row in episode_rows)),
                "last_local_index": int(max(int(row["objective_local_index"]) for row in episode_rows)),
            }
        )

    ep0 = next((row for row in episode_summaries if int(row["objective_episode_index"]) == 0), None)
    ep1 = next((row for row in episode_summaries if int(row["objective_episode_index"]) == 1), None)
    if ep0 is None or ep1 is None:
        raise ValueError("Expected both objective episodes 0 and 1.")

    local56_reward_raw = float(local56["reward_raw"])
    local56_return_raw = float(local56["rollout_return_raw"])

    decomposition = {
        "full_rollout_mean_raw": float(full_mean_raw),
        "full_rollout_std_raw": float(full_std_raw),
        "full_rollout_mean_over_std": float(full_mean_over_std),
        "n_samples": int(n_samples),
        "objective_token_count": int(objective_token_count),
        "prefix_token_count_before_objective": int(prefix_token_count),
        "objective_token_share": float(objective_token_count / float(n_samples)),
        "prefix_token_share": float(prefix_token_count / float(n_samples)),
        "objective_mean_rollout_return_raw": float(objective_mean_raw),
        "implied_prefix_mean_rollout_return_raw": float(implied_prefix_mean_raw),
        "prefix_minus_objective_mean_gap_raw": float(implied_prefix_mean_raw - objective_mean_raw),
        "full_minus_objective_mean_gap_raw": float(full_mean_raw - objective_mean_raw),
    }

    timing = {
        "single_eval_pos": int(gae_payload["config"]["single_eval_pos"]),
        "objective_global_step_start": int(first_row["global_step"]),
        "objective_global_step_end": int(last_row["global_step"]),
        "objective_starts_at_single_eval_pos": bool(int(first_row["global_step"]) == int(gae_payload["config"]["single_eval_pos"])),
        "objective_is_tail_of_full_rollout": bool(int(first_row["global_step"]) > 0 and int(last_row["global_step"]) == int(n_samples - 1)),
    }

    local56_position = {
        "objective_local_index": 56,
        "global_step": int(local56["global_step"]),
        "objective_episode_index": int(local56["objective_episode_index"]),
        "rollout_return_raw": float(local56_return_raw),
        "reward_raw": float(local56_reward_raw),
        "reward_norm": float(local56["reward_norm"]),
        "next_non_terminal": float(local56["next_non_terminal"]),
        "return_minus_full_mean_raw": float(local56_return_raw - full_mean_raw),
        "return_minus_objective_mean_raw": float(local56_return_raw - objective_mean_raw),
    }

    evidence_limits = {
        "direct_terminal_bonus_trace_present_in_input_artifacts": False,
        "used_artifacts_only": True,
        "mean_source_result_is_weighted_average_inference_not_replay": True,
        "excluded_artifacts_due_to_contract_mismatch": [
            "/home/chen/RLPFN/artifacts/phase3_terminal_tail_mask_probe_pair2.json"
        ],
        "excluded_artifact_reason": (
            "pair2 env12 objective_terminal_reset_count disagrees with current trusted local56 terminal evidence, "
            "so it is not used in the main source chain."
        ),
    }

    return {
        "audit_entry": "phase3_env12_local56_mean_source_probe",
        "config": {
            "source_boundary_contract_json": str(
                Path("/home/chen/RLPFN/artifacts/phase3_env12_local56_boundary_contract_probe.json").expanduser().resolve()
            ),
            "source_gae_path_json": str(
                Path("/home/chen/RLPFN/artifacts/phase3_residual_anchor_gae_path_probe.json").expanduser().resolve()
            ),
            "suite_name": str(suite_name),
            "env_index": int(env_index),
        },
        "timing_contract": timing,
        "full_mean_decomposition": decomposition,
        "objective_episode_summaries": episode_summaries,
        "local56_position_readout": local56_position,
        "evidence_limits": evidence_limits,
        "conclusions": {
            "full_mean_is_prefix_dominated_not_objective_dominated": bool(implied_prefix_mean_raw > full_mean_raw > objective_mean_raw),
            "objective_tail_tokens_are_too_late_to_set_full_mean_constant": bool(
                bool(timing["objective_is_tail_of_full_rollout"]) and objective_mean_raw < full_mean_raw
            ),
            "post_reset_episode_lowers_objective_mean_instead_of_raising_it": bool(
                float(ep1["mean_rollout_return_raw"]) < float(ep0["mean_rollout_return_raw"])
                and float(ep1["mean_rollout_return_raw"]) < float(objective_mean_raw)
            ),
            "local56_terminal_row_is_not_a_positive_mean_source": bool(
                local56_reward_raw < 0.0 and local56_return_raw < full_mean_raw and local56_return_raw < objective_mean_raw
            ),
            "large_terminal_boundary_constant_is_explained_by_full_rollout_centering_scope": bool(
                implied_prefix_mean_raw - objective_mean_raw > 1.0
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Explain why env12 local56 inherits a large full-rollout mean source in the terminal boundary contract."
    )
    parser.add_argument(
        "--boundary-contract-json",
        type=str,
        default="/home/chen/RLPFN/artifacts/phase3_env12_local56_boundary_contract_probe.json",
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

    boundary_payload = _load_boundary_contract_payload(args.boundary_contract_json)
    gae_payload = _load_gae_payload(args.gae_path_json)
    result = _build_probe(
        boundary_payload=boundary_payload,
        gae_payload=gae_payload,
        suite_name=str(args.suite_name),
        env_index=int(args.env_index),
    )
    result["config"]["source_boundary_contract_json"] = str(Path(args.boundary_contract_json).expanduser().resolve())
    result["config"]["source_gae_path_json"] = str(Path(args.gae_path_json).expanduser().resolve())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(output_path.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
