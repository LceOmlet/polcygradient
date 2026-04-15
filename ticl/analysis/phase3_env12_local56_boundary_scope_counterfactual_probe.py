import argparse
import json
from pathlib import Path
from typing import Any


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(float(v) for v in values) / len(values))


def _std(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = _mean(values)
    return float((sum((float(v) - mean) ** 2 for v in values) / len(values)) ** 0.5)


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


def _scope_stats(name: str, rows: list[dict[str, Any]], local56_reward_raw: float, current_boundary_abs: float) -> dict[str, Any]:
    returns = [float(row["rollout_return_raw"]) for row in rows]
    mean_raw = _mean(returns)
    std_raw = _std(returns)
    if std_raw <= 1e-12:
        mean_over_std = None
        boundary_shift = None
        reward_over_std = None
        reward_norm = None
        boundary_ratio_vs_current = None
    else:
        mean_over_std = float(mean_raw / std_raw)
        boundary_shift = float(-mean_over_std)
        reward_over_std = float(local56_reward_raw / std_raw)
        reward_norm = float(reward_over_std + boundary_shift)
        boundary_ratio_vs_current = float(abs(boundary_shift) / abs(current_boundary_abs))
    return {
        "scope_name": str(name),
        "token_count": int(len(rows)),
        "mean_raw": float(mean_raw),
        "std_raw": float(std_raw),
        "mean_over_std": mean_over_std,
        "local56_boundary_shift_norm": boundary_shift,
        "local56_reward_over_std": reward_over_std,
        "local56_reward_norm_before_value_term": reward_norm,
        "boundary_magnitude_ratio_vs_current": boundary_ratio_vs_current,
    }


def _build_probe(
    boundary_payload: dict[str, Any],
    gae_payload: dict[str, Any],
    *,
    suite_name: str,
    env_index: int,
) -> dict[str, Any]:
    env_rows = _load_env_rows(gae_payload, suite_name=suite_name, env_index=env_index)
    local56 = next(dict(row) for row in env_rows if int(row["objective_local_index"]) == 56)
    local56_reward_raw = float(local56["reward_raw"])

    current = dict(boundary_payload["local56_boundary"])
    current_boundary_shift = float(current["actual_boundary_shift_norm"])
    current_reward_over_std = float(current["reward_over_std"])
    current_reward_norm = float(current["reward_norm"])

    scopes = {
        "current_full_rollout": env_rows,
        "objective_episode0_only": [row for row in env_rows if int(row["objective_episode_index"]) == 0],
        "objective_episode1_only": [row for row in env_rows if int(row["objective_episode_index"]) == 1],
        "objective_all_tokens": env_rows,
    }

    # current_full_rollout stats come from the trusted boundary contract artifact, not objective rows.
    scope_rows = []
    scope_rows.append(
        {
            "scope_name": "current_full_rollout",
            "token_count": int(gae_payload["config"]["n_samples"]),
            "mean_raw": float(boundary_payload["full_rollout_contract"]["full_rollout_mean_raw"]),
            "std_raw": float(boundary_payload["full_rollout_contract"]["full_rollout_std_raw"]),
            "mean_over_std": float(boundary_payload["full_rollout_contract"]["full_rollout_mean_over_std"]),
            "local56_boundary_shift_norm": float(current_boundary_shift),
            "local56_reward_over_std": float(current_reward_over_std),
            "local56_reward_norm_before_value_term": float(current_reward_norm),
            "boundary_magnitude_ratio_vs_current": 1.0,
        }
    )
    for scope_name, rows in [
        ("objective_all_tokens", scopes["objective_all_tokens"]),
        ("objective_episode0_only", scopes["objective_episode0_only"]),
        ("objective_episode1_only", scopes["objective_episode1_only"]),
    ]:
        scope_rows.append(
            _scope_stats(
                scope_name,
                rows,
                local56_reward_raw=local56_reward_raw,
                current_boundary_abs=current_boundary_shift,
            )
        )

    row_by_name = {str(row["scope_name"]): row for row in scope_rows}
    objective_all = row_by_name["objective_all_tokens"]
    episode0 = row_by_name["objective_episode0_only"]
    episode1 = row_by_name["objective_episode1_only"]

    return {
        "audit_entry": "phase3_env12_local56_boundary_scope_counterfactual_probe",
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
        "local56_readout": {
            "objective_local_index": 56,
            "global_step": int(local56["global_step"]),
            "reward_raw": float(local56_reward_raw),
            "current_boundary_shift_norm": float(current_boundary_shift),
            "current_reward_over_std": float(current_reward_over_std),
            "current_reward_norm_before_value_term": float(current_reward_norm),
        },
        "scope_rows": scope_rows,
        "conclusions": {
            "objective_all_centering_worsens_boundary_penalty_vs_current": bool(
                abs(float(objective_all["local56_boundary_shift_norm"])) > abs(float(current_boundary_shift))
            ),
            "objective_episode0_centering_worsens_boundary_penalty_vs_current": bool(
                abs(float(episode0["local56_boundary_shift_norm"])) > abs(float(current_boundary_shift))
            ),
            "objective_episode1_centering_worsens_boundary_penalty_vs_current": bool(
                episode1["local56_boundary_shift_norm"] is not None
                and abs(float(episode1["local56_boundary_shift_norm"])) > abs(float(current_boundary_shift))
            ),
            "no_objective_aligned_scope_shrinks_local56_boundary_penalty": bool(
                all(
                    row["local56_boundary_shift_norm"] is not None
                    and abs(float(row["local56_boundary_shift_norm"])) >= abs(float(current_boundary_shift))
                    for row in [objective_all, episode0, episode1]
                )
            ),
            "naive_objective_aligned_scope_change_should_not_be_promoted_to_fix": bool(
                all(
                    row["local56_boundary_shift_norm"] is not None
                    and abs(float(row["local56_boundary_shift_norm"])) >= abs(float(current_boundary_shift))
                    for row in [objective_all, episode0, episode1]
                )
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Counterfactual local56 boundary penalty under alternative centering scopes, read-only."
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
