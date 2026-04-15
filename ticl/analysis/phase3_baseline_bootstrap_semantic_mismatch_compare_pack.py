import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: str) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise ValueError(f"Missing JSON artifact: {resolved}")
    return json.loads(resolved.read_text())


def _validate_source(payload: dict[str, Any], *, expected_env_index: int) -> None:
    config = dict(payload.get("config", {}))
    if str(config.get("suite_name", "")) != "pair2":
        raise ValueError(f"Unexpected suite_name in source-chain artifact: {config.get('suite_name')!r}")
    if int(config.get("env_index", -1)) != int(expected_env_index):
        raise ValueError(
            f"Unexpected env_index in source-chain artifact: observed={config.get('env_index')!r}, "
            f"expected={expected_env_index!r}"
        )
    if str(config.get("flip_mode", "")) != "future_carry_dominant":
        raise ValueError(f"Unexpected flip_mode in source-chain artifact: {config.get('flip_mode')!r}")


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(float(v) for v in values) / len(values))


def _stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
    return {
        "count": int(len(values)),
        "mean": float(_mean(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _baseline_bootstrap_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [
        dict(row)
        for row in list(payload.get("start_rows", []))
        if str(row.get("dominant_negative_source", "")) == "baseline_bootstrap_semantic_mismatch"
    ]
    if not rows:
        raise ValueError(f"No baseline_bootstrap_semantic_mismatch rows found in {payload.get('audit_entry')!r}")
    return sorted(
        rows,
        key=lambda row: (
            int(row["start_objective_local_index"]),
            int(row.get("start_global_step", row["start_objective_local_index"])),
        ),
    )


def _summarize_rows(rows: list[dict[str, Any]], *, near_terminal_window: int) -> dict[str, Any]:
    tail_rows = [row for row in rows if int(row["start_tokens_to_objective_episode_end"]) <= int(near_terminal_window)]
    mid_rows = [row for row in rows if int(row["start_tokens_to_objective_episode_end"]) > int(near_terminal_window)]
    return {
        "row_count": int(len(rows)),
        "tail_core": {
            "window_tokens_to_objective_episode_end": int(near_terminal_window),
            "row_count": int(len(tail_rows)),
            "fraction_within_baseline_bootstrap_rows": float(len(tail_rows) / max(len(rows), 1)),
            "objective_local_indices": [int(row["start_objective_local_index"]) for row in tail_rows],
            "tokens_to_objective_episode_end": [
                int(row["start_tokens_to_objective_episode_end"]) for row in tail_rows
            ],
            "mean_tokens_to_objective_episode_end": _mean(
                [float(row["start_tokens_to_objective_episode_end"]) for row in tail_rows]
            ),
            "mean_start_raw_rollout_residual_norm": _mean(
                [float(row["start_raw_rollout_residual_norm"]) for row in tail_rows]
            ),
            "mean_start_gae_future_carry_norm": _mean(
                [float(row["start_gae_future_carry_norm"]) for row in tail_rows]
            ),
            "mean_negative_share": _mean(
                [float(row["negative_source_shares"].get("baseline_bootstrap_semantic_mismatch", 0.0)) for row in tail_rows]
            ),
        },
        "mid_episode_extension": {
            "row_count": int(len(mid_rows)),
            "fraction_within_baseline_bootstrap_rows": float(len(mid_rows) / max(len(rows), 1)),
            "objective_local_indices": [int(row["start_objective_local_index"]) for row in mid_rows],
            "tokens_to_objective_episode_end": [
                int(row["start_tokens_to_objective_episode_end"]) for row in mid_rows
            ],
            "mean_tokens_to_objective_episode_end": _mean(
                [float(row["start_tokens_to_objective_episode_end"]) for row in mid_rows]
            ),
            "mean_start_raw_rollout_residual_norm": _mean(
                [float(row["start_raw_rollout_residual_norm"]) for row in mid_rows]
            ),
            "mean_start_gae_future_carry_norm": _mean(
                [float(row["start_gae_future_carry_norm"]) for row in mid_rows]
            ),
            "mean_negative_share": _mean(
                [float(row["negative_source_shares"].get("baseline_bootstrap_semantic_mismatch", 0.0)) for row in mid_rows]
            ),
        },
        "overall": {
            "objective_local_indices": [int(row["start_objective_local_index"]) for row in rows],
            "tokens_to_objective_episode_end": [
                int(row["start_tokens_to_objective_episode_end"]) for row in rows
            ],
            "mean_tokens_to_objective_episode_end": _mean(
                [float(row["start_tokens_to_objective_episode_end"]) for row in rows]
            ),
            "mean_start_raw_rollout_residual_norm": _mean(
                [float(row["start_raw_rollout_residual_norm"]) for row in rows]
            ),
            "mean_start_gae_future_carry_norm": _mean(
                [float(row["start_gae_future_carry_norm"]) for row in rows]
            ),
            "mean_negative_share": _mean(
                [float(row["negative_source_shares"].get("baseline_bootstrap_semantic_mismatch", 0.0)) for row in rows]
            ),
        },
        "tail_core_supported": bool(tail_rows),
        "mid_episode_extension_supported": bool(mid_rows),
        "tail_core_is_pure_tail": bool(len(mid_rows) == 0),
    }


def _scope_label(near_terminal_window: int) -> str:
    if int(near_terminal_window) <= 1:
        return "terminal_adjacent_core"
    if int(near_terminal_window) <= 4:
        return "near_terminal_tail_core"
    return f"near_terminal_window_{int(near_terminal_window)}_core"


def build_baseline_bootstrap_semantic_mismatch_compare_pack(
    *,
    env12_json: str,
    env1_json: str,
    near_terminal_window: int = 4,
) -> dict[str, Any]:
    env12 = _load_json(env12_json)
    env1 = _load_json(env1_json)
    _validate_source(env12, expected_env_index=12)
    _validate_source(env1, expected_env_index=1)

    env12_rows = _baseline_bootstrap_rows(env12)
    env1_rows = _baseline_bootstrap_rows(env1)
    env12_summary = _summarize_rows(env12_rows, near_terminal_window=near_terminal_window)
    env1_summary = _summarize_rows(env1_rows, near_terminal_window=near_terminal_window)

    shared_tail_core_supported = bool(env12_summary["tail_core"]["row_count"] > 0 and env1_summary["tail_core"]["row_count"] > 0)
    env12_mid_extension_present = bool(env12_summary["mid_episode_extension"]["row_count"] > 0)
    env1_mid_extension_present = bool(env1_summary["mid_episode_extension"]["row_count"] > 0)
    branch_specific_followup_needed = bool(env12_mid_extension_present or env1_mid_extension_present)

    recommendation = {
        "candidate_small_fix_target": "baseline_bootstrap_semantic_mismatch",
        "narrow_fix_scope": _scope_label(near_terminal_window),
        "branch_specific_followup_needed": branch_specific_followup_needed,
        "reason": (
            "Both env12 and env1 share a near-terminal baseline-bootstrap core, but env12 also carries "
            "a mid-episode extension that env1 does not. The smallest shared repair axis is therefore the "
            f"{_scope_label(near_terminal_window)}, not the full pooled bootstrap bucket."
        ),
    }

    return {
        "audit_entry": "phase3_baseline_bootstrap_semantic_mismatch_compare_pack",
        "source_env12_json": str(Path(env12_json).expanduser().resolve()),
        "source_env1_json": str(Path(env1_json).expanduser().resolve()),
        "comparison": {
            "env12": {
                "env_index": 12,
                "dominant_negative_source": "baseline_bootstrap_semantic_mismatch",
                "baseline_bootstrap_rows": env12_summary,
            },
            "env1": {
                "env_index": 1,
                "dominant_negative_source": "baseline_bootstrap_semantic_mismatch",
                "baseline_bootstrap_rows": env1_summary,
            },
            "shared_tail_core_supported": shared_tail_core_supported,
            "near_terminal_window": int(near_terminal_window),
            "dominant_source_difference": True,
        },
        "conclusions": {
            "baseline_bootstrap_semantic_mismatch_is_shared_candidate": True,
            "tail_core_shared_across_env12_and_env1": shared_tail_core_supported,
            "env12_has_mid_episode_extension": env12_mid_extension_present,
            "env1_is_tail_only_for_baseline_bootstrap": bool(env1_summary["mid_episode_extension"]["row_count"] == 0),
            "branch_specific_followup_needed": branch_specific_followup_needed,
        },
        "recommendation": recommendation,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare baseline_bootstrap_semantic_mismatch rows across env12 and env1."
    )
    parser.add_argument(
        "--env12-json",
        type=str,
        default="/home/chen/RLPFN/artifacts/phase3_future_carry_source_chain_probe_pair2_env12.json",
    )
    parser.add_argument(
        "--env1-json",
        type=str,
        default="/home/chen/RLPFN/artifacts/phase3_future_carry_source_chain_probe_pair2_env1.json",
    )
    parser.add_argument(
        "--near-terminal-window",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="/home/chen/RLPFN/artifacts/phase3_baseline_bootstrap_semantic_mismatch_compare_pack.json",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output_json).expanduser().resolve()
    if output_path.exists() and not bool(args.overwrite):
        print(output_path.read_text())
        return 0

    report = build_baseline_bootstrap_semantic_mismatch_compare_pack(
        env12_json=args.env12_json,
        env1_json=args.env1_json,
        near_terminal_window=int(args.near_terminal_window),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
