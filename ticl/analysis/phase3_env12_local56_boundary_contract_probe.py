import argparse
import json
from pathlib import Path
from typing import Any


CONTRACT_EPS = 1e-6


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(float(v) for v in values) / len(values))


def _std(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = _mean(values)
    return float((sum((float(v) - mean) ** 2 for v in values) / len(values)) ** 0.5)


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


def _focus_row_map(negative_payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(row["objective_local_index"]): dict(row) for row in list(negative_payload.get("focus_rows", []))}


def _boundary_shift_norm(row: dict[str, Any], *, env_std: float) -> float:
    return float(float(row["reward_norm"]) - float(row["reward_raw"]) / float(env_std))


def _build_probe(
    negative_payload: dict[str, Any],
    gae_payload: dict[str, Any],
    *,
    suite_name: str,
    env_index: int,
) -> dict[str, Any]:
    env_rows = _load_env_rows(gae_payload, suite_name=suite_name, env_index=env_index)
    row_by_local = {int(row["objective_local_index"]): dict(row) for row in env_rows}
    focus_rows = _focus_row_map(negative_payload)
    if 56 not in focus_rows:
        raise ValueError("Expected local56 in focus_rows.")
    if 56 not in row_by_local:
        raise ValueError("Expected local56 in token rows.")

    env_std = float(negative_payload["guardrails"]["inferred_env_std"])
    full_mean_over_std = float(negative_payload["guardrails"]["inferred_mean_over_std_from_terminal_reset"])
    full_mean_raw = float(env_std * full_mean_over_std)

    local56 = dict(focus_rows[56])
    local56_reward_over_std = float(float(local56["reward_raw"]) / float(env_std))
    local56_boundary_shift = float(local56["reward_norm"]) - local56_reward_over_std

    contrast_locals = [55, 57, 101]
    contrast_rows = []
    gamma_inferences = []
    contrast_boundary_values = []
    for local_idx in contrast_locals:
        if local_idx not in row_by_local:
            continue
        row = dict(row_by_local[local_idx])
        boundary_shift = _boundary_shift_norm(row, env_std=env_std)
        gamma_inferred = 1.0
        if abs(float(full_mean_over_std)) > CONTRACT_EPS:
            gamma_inferred = float(1.0 + boundary_shift / float(full_mean_over_std))
        gamma_inferences.append(gamma_inferred)
        contrast_boundary_values.append(boundary_shift)
        contrast_rows.append(
            {
                "objective_local_index": int(local_idx),
                "next_non_terminal": float(row["next_non_terminal"]),
                "reward_raw": float(row["reward_raw"]),
                "reward_norm": float(row["reward_norm"]),
                "reward_over_std": float(float(row["reward_raw"]) / float(env_std)),
                "boundary_shift_norm": boundary_shift,
                "gamma_inferred_from_boundary": gamma_inferred,
            }
        )

    gamma_inferred_mean = _mean(gamma_inferences)
    expected_terminal_boundary_shift = float(((gamma_inferred_mean * float(local56["next_non_terminal"])) - 1.0) * full_mean_over_std)
    local56_delta_without_boundary_shift = float(float(local56["delta_norm"]) - local56_boundary_shift)
    local56_delta_recovery = float(local56_delta_without_boundary_shift - float(local56["delta_norm"]))

    objective_subset_raw_returns = [float(row["rollout_return_raw"]) for row in env_rows]
    objective_subset_mean_raw = _mean(objective_subset_raw_returns)
    objective_subset_std_raw = _std(objective_subset_raw_returns)
    objective_subset_mean_over_std = 0.0
    if objective_subset_std_raw > CONTRACT_EPS:
        objective_subset_mean_over_std = float(objective_subset_mean_raw / objective_subset_std_raw)

    return {
        "audit_entry": "phase3_env12_local56_boundary_contract_probe",
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
        "full_rollout_contract": {
            "full_rollout_mean_raw": float(full_mean_raw),
            "full_rollout_std_raw": float(env_std),
            "full_rollout_mean_over_std": float(full_mean_over_std),
            "gamma_inferred_from_nonterminal_boundary_mean": float(gamma_inferred_mean),
            "gamma_inferred_from_nonterminal_boundary_max_abs_error": float(
                _max_abs([float(v) - 1.0 for v in gamma_inferences])
            ),
        },
        "objective_subset_contrast": {
            "objective_subset_token_count": int(len(objective_subset_raw_returns)),
            "objective_subset_mean_raw": float(objective_subset_mean_raw),
            "objective_subset_std_raw": float(objective_subset_std_raw),
            "objective_subset_mean_over_std": float(objective_subset_mean_over_std),
            "mean_over_std_gap_vs_full_contract": float(objective_subset_mean_over_std - full_mean_over_std),
        },
        "local56_boundary": {
            "objective_local_index": 56,
            "global_step": int(local56["global_step"]),
            "next_non_terminal": float(local56["next_non_terminal"]),
            "reward_raw": float(local56["reward_raw"]),
            "reward_over_std": float(local56_reward_over_std),
            "reward_norm": float(local56["reward_norm"]),
            "value_norm": float(local56["value_norm"]),
            "delta_norm": float(local56["delta_norm"]),
            "actual_boundary_shift_norm": float(local56_boundary_shift),
            "expected_boundary_shift_norm_from_full_contract": float(expected_terminal_boundary_shift),
            "boundary_shift_contract_error": float(local56_boundary_shift - expected_terminal_boundary_shift),
            "delta_without_boundary_shift": float(local56_delta_without_boundary_shift),
            "delta_recovery_from_removing_boundary_shift": float(local56_delta_recovery),
            "boundary_shift_share_of_delta_magnitude": float(
                abs(float(local56_boundary_shift)) / max(abs(float(local56["delta_norm"])), CONTRACT_EPS)
            ),
            "full_mean_raw_to_immediate_reward_abs_ratio": float(
                abs(float(full_mean_raw)) / max(abs(float(local56["reward_raw"])), CONTRACT_EPS)
            ),
        },
        "nonterminal_contrast_rows": contrast_rows,
        "conclusions": {
            "local56_boundary_shift_matches_full_rollout_centering_contract": bool(
                abs(float(local56_boundary_shift - expected_terminal_boundary_shift)) <= CONTRACT_EPS
            ),
            "nonterminal_boundary_rows_imply_gamma_one_contract": bool(
                _max_abs([float(v) - 1.0 for v in gamma_inferences]) <= CONTRACT_EPS
            ),
            "local56_boundary_shift_recovers_most_of_negative_delta_if_removed": bool(
                float(local56_delta_recovery) > 0.8 * abs(float(local56["delta_norm"]))
            ),
            "objective_subset_stats_do_not_explain_terminal_boundary_constant": bool(
                abs(float(objective_subset_mean_over_std - full_mean_over_std)) > 1e-3
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Explain why env12 local56 gets a large negative -mean/std terminal boundary term."
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
    )
    result["config"]["source_negative_delta_json"] = str(Path(args.negative_delta_json).expanduser().resolve())
    result["config"]["source_gae_path_json"] = str(Path(args.gae_path_json).expanduser().resolve())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(output_path.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
