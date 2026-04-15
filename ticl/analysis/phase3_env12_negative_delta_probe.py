import argparse
import json
from pathlib import Path
from typing import Any


STD_SPREAD_EPS = 1e-6


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(float(v) for v in values) / len(values))


def _max_abs(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(max(abs(float(v)) for v in values))


def _classify_negative_delta_driver(row: dict[str, Any]) -> str:
    if float(row["next_non_terminal"]) <= 0.0 and abs(float(row["boundary_shift_norm"])) > abs(
        float(row["bootstrap_term_norm"])
    ):
        return "terminal_reset_boundary_shift_dominant"
    if float(row["reward_norm"]) > 0.0 and float(row["bootstrap_term_norm"]) < -float(row["reward_norm"]):
        return "bootstrap_stepdown_overrides_positive_reward"
    if float(row["reward_norm"]) <= 0.0 and float(row["bootstrap_term_norm"]) < 0.0:
        return "bootstrap_negative_with_nonpositive_reward"
    if float(row["reward_norm"]) < 0.0 and abs(float(row["reward_norm"])) >= abs(float(row["bootstrap_term_norm"])):
        return "negative_reward_term_dominant"
    return "mixed_negative_local_td"


def _load_env_rows(gae_payload: dict[str, Any], suite_name: str, env_index: int) -> list[dict[str, Any]]:
    for bundle in list(gae_payload.get("token_row_bundles", [])):
        if str(bundle.get("suite_name", "")) == str(suite_name) and int(bundle.get("env_index", -1)) == int(env_index):
            rows = [dict(row) for row in list(bundle.get("token_rows", []))]
            if not rows:
                raise ValueError(f"No token rows found for {suite_name} env{env_index}.")
            rows.sort(key=lambda row: int(row["objective_local_index"]))
            return rows
    raise ValueError(f"Could not find token rows for {suite_name} env{env_index}.")


def _load_source_contribution_map(source_payload: dict[str, Any]) -> dict[tuple[int, int], float]:
    out: dict[tuple[int, int], float] = {}
    for row in list(source_payload.get("top_negative_source_tokens", [])):
        out[(int(row["objective_episode_index"]), int(row["objective_local_index"]))] = float(
            row["aggregated_negative_contribution"]
        )
    return out


def _infer_env_std(rows: list[dict[str, Any]]) -> dict[str, float]:
    stds = [
        float(row["reward_raw"]) / float(row["reward_norm"])
        for row in rows
        if float(row["next_non_terminal"]) > 0.0 and abs(float(row["reward_norm"])) > 1e-12
    ]
    if not stds:
        raise RuntimeError("Could not infer env std from nonterminal rows.")
    return {
        "mean": float(_mean(stds)),
        "max_abs_deviation": float(_max_abs([float(v) - float(_mean(stds)) for v in stds])),
    }


def _infer_mean_over_std(rows: list[dict[str, Any]], env_std: float) -> dict[str, float]:
    boundary_shifts = [
        float(row["reward_norm"]) - float(row["reward_raw"]) / float(env_std)
        for row in rows
        if float(row["next_non_terminal"]) <= 0.0
    ]
    if not boundary_shifts:
        raise RuntimeError("Could not infer boundary shift from terminal-reset rows.")
    mean_over_std_values = [-float(v) for v in boundary_shifts]
    mean_over_std = float(_mean(mean_over_std_values))
    return {
        "mean_over_std": mean_over_std,
        "terminal_reset_row_count": int(len(mean_over_std_values)),
        "max_abs_deviation": float(_max_abs([float(v) - mean_over_std for v in mean_over_std_values])),
    }


def _enrich_focus_row(
    row: dict[str, Any],
    *,
    env_std: float,
    mean_over_std: float,
    source_contribution_map: dict[tuple[int, int], float],
) -> dict[str, Any]:
    reward_over_std = float(float(row["reward_raw"]) / float(env_std))
    boundary_shift_norm = float(float(row["reward_norm"]) - reward_over_std)
    next_value_step_norm = float(float(row["next_non_terminal"]) * float(row["next_value_norm"]))
    bootstrap_stepdown_norm = float(float(row["value_norm"]) - next_value_step_norm)
    raw_value_contract_recovered = float(float(mean_over_std) * float(env_std) + float(env_std) * float(row["value_norm"]))
    raw_value_contract_error = float(float(row["raw_value_raw"]) - raw_value_contract_recovered)
    negative_delta_margin_after_reward = float(-float(row["delta_norm"]) - max(0.0, -float(row["reward_norm"])))
    enriched = {
        "suite_name": str(row["suite_name"]),
        "env_index": int(row["env_index"]),
        "global_step": int(row["global_step"]),
        "objective_episode_index": int(row["objective_episode_index"]),
        "objective_local_index": int(row["objective_local_index"]),
        "token_in_objective_episode": int(row["token_in_objective_episode"]),
        "tokens_to_objective_episode_end": int(row["tokens_to_objective_episode_end"]),
        "next_non_terminal": float(row["next_non_terminal"]),
        "reward_raw": float(row["reward_raw"]),
        "reward_norm": float(row["reward_norm"]),
        "reward_over_std": reward_over_std,
        "boundary_shift_norm": boundary_shift_norm,
        "bootstrap_term_norm": float(row["bootstrap_term_norm"]),
        "bootstrap_stepdown_norm": bootstrap_stepdown_norm,
        "next_value_step_norm": next_value_step_norm,
        "value_norm": float(row["value_norm"]),
        "next_value_norm": float(row["next_value_norm"]),
        "delta_norm": float(row["delta_norm"]),
        "raw_rollout_residual_norm": float(row["raw_rollout_residual_norm"]),
        "raw_rollout_residual_raw": float(row["raw_rollout_residual_raw"]),
        "flip_cause": str(row["flip_cause"]),
        "aggregated_negative_source_contribution": float(
            source_contribution_map.get((int(row["objective_episode_index"]), int(row["objective_local_index"])), 0.0)
        ),
        "negative_delta_margin_after_reward": negative_delta_margin_after_reward,
        "raw_value_contract_error": raw_value_contract_error,
        "negative_delta_driver": _classify_negative_delta_driver(
            {
                **row,
                "reward_norm": float(row["reward_norm"]),
                "bootstrap_term_norm": float(row["bootstrap_term_norm"]),
                "boundary_shift_norm": boundary_shift_norm,
            }
        ),
    }
    return enriched


def _build_probe(
    gae_payload: dict[str, Any],
    source_payload: dict[str, Any],
    suite_name: str,
    env_index: int,
    token_local_indices: list[int],
) -> dict[str, Any]:
    rows = _load_env_rows(gae_payload, suite_name=suite_name, env_index=env_index)
    source_contribution_map = _load_source_contribution_map(source_payload)
    std_stats = _infer_env_std(rows)
    env_std = float(std_stats["mean"])
    if float(std_stats["max_abs_deviation"]) > float(STD_SPREAD_EPS):
        raise RuntimeError(f"Env std spread too large: {std_stats['max_abs_deviation']}")
    mean_stats = _infer_mean_over_std(rows, env_std=env_std)
    mean_over_std = float(mean_stats["mean_over_std"])

    row_by_local = {int(row["objective_local_index"]): row for row in rows}
    focus_rows = []
    missing = []
    for local_idx in token_local_indices:
        if int(local_idx) not in row_by_local:
            missing.append(int(local_idx))
            continue
        focus_rows.append(
            _enrich_focus_row(
                row_by_local[int(local_idx)],
                env_std=env_std,
                mean_over_std=mean_over_std,
                source_contribution_map=source_contribution_map,
            )
        )
    if missing:
        raise ValueError(f"Missing focus local indices for {suite_name} env{env_index}: {missing}")
    focus_rows.sort(key=lambda row: int(row["objective_local_index"]))

    local56 = next(row for row in focus_rows if int(row["objective_local_index"]) == 56)
    cluster_rows = [row for row in focus_rows if int(row["objective_local_index"]) in {29, 38, 51}]
    driver_counts: dict[str, int] = {}
    for row in focus_rows:
        driver = str(row["negative_delta_driver"])
        driver_counts[driver] = int(driver_counts.get(driver, 0) + 1)

    cluster_summary = {
        "token_count": int(len(cluster_rows)),
        "mean_reward_norm": _mean([float(row["reward_norm"]) for row in cluster_rows]),
        "mean_reward_over_std": _mean([float(row["reward_over_std"]) for row in cluster_rows]),
        "mean_bootstrap_term_norm": _mean([float(row["bootstrap_term_norm"]) for row in cluster_rows]),
        "mean_bootstrap_stepdown_norm": _mean([float(row["bootstrap_stepdown_norm"]) for row in cluster_rows]),
        "mean_delta_norm": _mean([float(row["delta_norm"]) for row in cluster_rows]),
        "mean_aggregated_negative_source_contribution": _mean(
            [float(row["aggregated_negative_source_contribution"]) for row in cluster_rows]
        ),
        "positive_reward_tokens": int(sum(1 for row in cluster_rows if float(row["reward_norm"]) > 0.0)),
        "positive_reward_but_negative_delta_tokens": int(
            sum(1 for row in cluster_rows if float(row["reward_norm"]) > 0.0 and float(row["delta_norm"]) < 0.0)
        ),
        "bootstrap_dominant_tokens": int(
            sum(
                1
                for row in cluster_rows
                if abs(float(row["bootstrap_term_norm"])) > abs(float(row["reward_norm"]))
            )
        ),
    }

    aggregate = {
        "focus_token_count": int(len(focus_rows)),
        "focus_driver_counts": dict(sorted(driver_counts.items())),
        "focus_local_indices": [int(row["objective_local_index"]) for row in focus_rows],
        "cluster_summary": cluster_summary,
    }
    conclusions = {
        "local56_terminal_reset_boundary_shift_dominant": bool(
            str(local56["negative_delta_driver"]) == "terminal_reset_boundary_shift_dominant"
        ),
        "local56_boundary_shift_exceeds_bootstrap_magnitude": bool(
            abs(float(local56["boundary_shift_norm"])) > abs(float(local56["bootstrap_term_norm"]))
        ),
        "local56_negative_delta_not_explained_by_immediate_reward_only": bool(
            abs(float(local56["boundary_shift_norm"])) > abs(float(local56["reward_over_std"]))
        ),
        "nonterminal_cluster_bootstrap_dominant": bool(
            int(cluster_summary["bootstrap_dominant_tokens"]) == int(cluster_summary["token_count"])
        ),
        "nonterminal_cluster_positive_reward_tokens_still_flip": bool(
            int(cluster_summary["positive_reward_but_negative_delta_tokens"]) >= 2
        ),
        "focus_rows_raw_value_contract_inconsistent": bool(
            _max_abs([float(row["raw_value_contract_error"]) for row in focus_rows]) > 1e-3
        ),
    }

    return {
        "audit_entry": "phase3_env12_negative_delta_probe",
        "config": {
            "source_gae_path_json": str(
                Path("/home/chen/RLPFN/artifacts/phase3_residual_anchor_gae_path_probe.json").expanduser().resolve()
            ),
            "source_future_carry_source_chain_json": str(
                Path(
                    "/home/chen/RLPFN/artifacts/phase3_future_carry_source_chain_probe_pair2_env12.json"
                ).expanduser().resolve()
            ),
            "suite_name": str(suite_name),
            "env_index": int(env_index),
            "token_local_indices": [int(v) for v in token_local_indices],
        },
        "guardrails": {
            "inferred_env_std": float(env_std),
            "env_std_max_abs_deviation": float(std_stats["max_abs_deviation"]),
            "inferred_mean_over_std_from_terminal_reset": float(mean_over_std),
            "terminal_reset_row_count": int(mean_stats["terminal_reset_row_count"]),
            "terminal_mean_over_std_max_abs_deviation": float(mean_stats["max_abs_deviation"]),
            "max_raw_value_contract_error_on_focus_tokens": _max_abs(
                [float(row["raw_value_contract_error"]) for row in focus_rows]
            ),
        },
        "focus_rows": focus_rows,
        "aggregate": aggregate,
        "conclusions": conclusions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Decompose why env12 key downstream tokens have negative delta_norm."
    )
    parser.add_argument(
        "--gae-path-json",
        type=str,
        default="/home/chen/RLPFN/artifacts/phase3_residual_anchor_gae_path_probe.json",
    )
    parser.add_argument(
        "--future-carry-source-chain-json",
        type=str,
        default="/home/chen/RLPFN/artifacts/phase3_future_carry_source_chain_probe_pair2_env12.json",
    )
    parser.add_argument("--suite-name", type=str, default="pair2")
    parser.add_argument("--env-index", type=int, default=12)
    parser.add_argument("--token-local-indices", type=int, nargs="*", default=[56, 29, 38, 51])
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output_json).expanduser().resolve()
    if output_path.exists() and not bool(args.overwrite):
        print(output_path.read_text())
        return 0

    gae_path = Path(args.gae_path_json).expanduser().resolve()
    source_path = Path(args.future_carry_source_chain_json).expanduser().resolve()
    gae_payload = json.loads(gae_path.read_text())
    source_payload = json.loads(source_path.read_text())
    result = _build_probe(
        gae_payload=gae_payload,
        source_payload=source_payload,
        suite_name=str(args.suite_name),
        env_index=int(args.env_index),
        token_local_indices=[int(v) for v in args.token_local_indices],
    )
    result["config"]["source_gae_path_json"] = str(gae_path)
    result["config"]["source_future_carry_source_chain_json"] = str(source_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(output_path.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
