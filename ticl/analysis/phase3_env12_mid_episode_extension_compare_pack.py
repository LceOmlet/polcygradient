import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: str) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise ValueError(f"Missing JSON artifact: {resolved}")
    return json.loads(resolved.read_text())


def _validate_env12_source(payload: dict[str, Any]) -> None:
    config = dict(payload.get("config", {}))
    if str(config.get("suite_name", "")) != "pair2":
        raise ValueError(f"Unexpected suite_name in source artifact: {config.get('suite_name')!r}")
    if int(config.get("env_index", -1)) != 12:
        raise ValueError(f"Unexpected env_index in source artifact: {config.get('env_index')!r}")
    if str(config.get("flip_mode", "")) != "future_carry_dominant":
        raise ValueError(f"Unexpected flip_mode in source artifact: {config.get('flip_mode')!r}")


def _rows_by_local_indices(payload: dict[str, Any], local_indices: list[int]) -> list[dict[str, Any]]:
    rows = [dict(row) for row in list(payload.get("start_rows", []))]
    index_map = {int(row.get("start_objective_local_index", -1)): row for row in rows}
    selected = []
    missing = []
    for local_idx in local_indices:
        row = index_map.get(int(local_idx))
        if row is None:
            missing.append(int(local_idx))
            continue
        selected.append(row)
    if missing:
        raise ValueError(f"Missing requested env12 branch-specific rows: {missing}")
    return sorted(selected, key=lambda row: int(row["start_objective_local_index"]))


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(float(v) for v in values) / len(values))


def _uniform_shares(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "is_uniform": False,
            "max_abs_deviation": 0.0,
            "reference": {},
        }
    reference = dict(rows[0].get("negative_source_shares", {}))
    deviations = []
    for row in rows[1:]:
        shares = dict(row.get("negative_source_shares", {}))
        keys = set(reference) | set(shares)
        for key in keys:
            deviations.append(abs(float(reference.get(key, 0.0)) - float(shares.get(key, 0.0))))
    return {
        "is_uniform": bool(max(deviations, default=0.0) <= 1e-12),
        "max_abs_deviation": float(max(deviations, default=0.0)),
        "reference": reference,
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    shares = _uniform_shares(rows)
    dominant_sources = [str(row.get("dominant_negative_source", "")) for row in rows]
    dominant_source = dominant_sources[0] if dominant_sources else "none"
    return {
        "row_count": int(len(rows)),
        "local_indices": [int(row["start_objective_local_index"]) for row in rows],
        "global_steps": [int(row.get("start_global_step", row["start_objective_local_index"])) for row in rows],
        "tokens_to_objective_episode_end": [int(row["start_tokens_to_objective_episode_end"]) for row in rows],
        "future_chain_step_count": [int(row.get("future_chain_step_count", 0)) for row in rows],
        "mean_tokens_to_objective_episode_end": _mean(
            [float(row["start_tokens_to_objective_episode_end"]) for row in rows]
        ),
        "mean_start_raw_rollout_residual_norm": _mean(
            [float(row["start_raw_rollout_residual_norm"]) for row in rows]
        ),
        "mean_start_gae_future_carry_norm": _mean([float(row["start_gae_future_carry_norm"]) for row in rows]),
        "mean_weighted_delta_sum": _mean([float(row["weighted_delta_sum"]) for row in rows]),
        "dominant_source": dominant_source,
        "dominant_source_counts": {
            str(source): int(sum(1 for item in dominant_sources if item == source))
            for source in sorted(set(dominant_sources))
        },
        "negative_source_shares_reference": shares["reference"],
        "negative_source_shares_is_uniform": bool(shares["is_uniform"]),
        "negative_source_shares_max_abs_deviation": float(shares["max_abs_deviation"]),
        "is_contiguous": bool(
            all(
                int(rows[i + 1]["start_objective_local_index"]) == int(rows[i]["start_objective_local_index"]) + 1
                for i in range(len(rows) - 1)
            )
        ),
        "monotone_tte_decreasing": bool(
            all(
                int(rows[i + 1]["start_tokens_to_objective_episode_end"]) == int(rows[i]["start_tokens_to_objective_episode_end"]) - 1
                for i in range(len(rows) - 1)
            )
        ),
        "monotone_weighted_delta_more_negative": bool(
            all(float(rows[i + 1]["weighted_delta_sum"]) < float(rows[i]["weighted_delta_sum"]) for i in range(len(rows) - 1))
        ),
    }


def build_env12_mid_episode_extension_compare_pack(
    *,
    env12_json: str,
    extension_local_indices: list[int] | None = None,
    core_local_index: int = 100,
) -> dict[str, Any]:
    payload = _load_json(env12_json)
    _validate_env12_source(payload)

    extension_local_indices = extension_local_indices or [18, 19, 20, 21]
    extension_rows = _rows_by_local_indices(payload, extension_local_indices)
    core_rows = _rows_by_local_indices(payload, [core_local_index])

    extension_summary = _summary(extension_rows)
    core_summary = _summary(core_rows)

    conclusions = {
        "env12_mid_episode_extension_is_branch_specific": True,
        "env12_mid_episode_extension_root_source_is_baseline_bootstrap_semantic_mismatch": bool(
            extension_summary["dominant_source"] == "baseline_bootstrap_semantic_mismatch"
        ),
        "env12_mid_episode_extension_is_contiguous": bool(extension_summary["is_contiguous"]),
        "env12_mid_episode_extension_has_uniform_source_shares": bool(extension_summary["negative_source_shares_is_uniform"]),
        "env12_mid_episode_extension_has_no_internal_split_supported": bool(
            extension_summary["negative_source_shares_is_uniform"] and extension_summary["is_contiguous"]
        ),
        "env12_terminal_adjacent_core_remains_shared_guardrail_anchor": bool(
            core_summary["dominant_source"] == "baseline_bootstrap_semantic_mismatch"
        ),
    }

    recommendation = {
        "candidate_small_control_target": "env12_mid_episode_extension_block",
        "branch_specific_followup_needed": True,
        "shared_guardrail_unchanged": True,
        "reason": (
            "The env12 mid-episode extension is a contiguous four-token block with uniform negative-source shares "
            "and a single dominant source, so it can be treated as one branch-specific control block. "
            "No finer split is supported by the current evidence."
        ),
    }

    return {
        "audit_entry": "phase3_env12_mid_episode_extension_compare_pack",
        "source_env12_json": str(Path(env12_json).expanduser().resolve()),
        "extension_local_indices": [int(v) for v in extension_local_indices],
        "core_local_index": int(core_local_index),
        "comparison": {
            "extension_block": extension_summary,
            "terminal_adjacent_core": core_summary,
        },
        "conclusions": conclusions,
        "recommendation": recommendation,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect the env12 mid-episode extension block as the remaining branch-specific direction."
    )
    parser.add_argument(
        "--env12-json",
        type=str,
        default="/home/chen/RLPFN/artifacts/phase3_future_carry_source_chain_probe_pair2_env12.json",
    )
    parser.add_argument(
        "--extension-local-indices",
        type=int,
        nargs="*",
        default=[18, 19, 20, 21],
    )
    parser.add_argument("--core-local-index", type=int, default=100)
    parser.add_argument(
        "--output-json",
        type=str,
        default="/home/chen/RLPFN/artifacts/phase3_env12_mid_episode_extension_compare_pack.json",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output_json).expanduser().resolve()
    if output_path.exists() and not bool(args.overwrite):
        print(output_path.read_text())
        return 0

    report = build_env12_mid_episode_extension_compare_pack(
        env12_json=args.env12_json,
        extension_local_indices=[int(v) for v in args.extension_local_indices],
        core_local_index=int(args.core_local_index),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
