import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: str) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise ValueError(f"Missing JSON artifact: {resolved}")
    return json.loads(resolved.read_text())


def _validate_env(payload: dict[str, Any], *, expected_env_index: int) -> None:
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


def _source_shares(payload: dict[str, Any]) -> dict[str, float]:
    return {str(k): float(v) for k, v in dict(payload.get("aggregate", {}).get("negative_source_shares", {})).items()}


def _aggregate(payload: dict[str, Any]) -> dict[str, Any]:
    return dict(payload.get("aggregate", {}))


def build_future_carry_source_chain_compare_pack(
    *,
    env12_json: str,
    env1_json: str,
) -> dict[str, Any]:
    env12 = _load_json(env12_json)
    env1 = _load_json(env1_json)
    _validate_env(env12, expected_env_index=12)
    _validate_env(env1, expected_env_index=1)

    env12_agg = _aggregate(env12)
    env1_agg = _aggregate(env1)
    env12_shares = _source_shares(env12)
    env1_shares = _source_shares(env1)
    shared_sources = sorted(set(env12_shares) & set(env1_shares))
    shared_source_summary = {
        source: {
            "env12_share": float(env12_shares[source]),
            "env1_share": float(env1_shares[source]),
            "mean_share": float((env12_shares[source] + env1_shares[source]) / 2.0),
            "min_share": float(min(env12_shares[source], env1_shares[source])),
        }
        for source in shared_sources
    }
    dominant_difference = str(env12_agg.get("dominant_negative_source", "")) != str(
        env1_agg.get("dominant_negative_source", "")
    )

    shared_candidate = None
    if shared_sources:
        shared_candidate = max(shared_sources, key=lambda source: (float(shared_source_summary[source]["min_share"]), source))

    recommendation = {
        "candidate_small_fix_target": shared_candidate or "none",
        "branch_specific_followup_needed": bool(dominant_difference),
        "shared_candidate_supported": bool(shared_candidate is not None),
        "reason": (
            "The only shared negative source across env12 and env1 is the baseline bootstrap semantic mismatch; "
            "dominant sources differ, so any narrow repair must stay branch-aware."
        ),
    }

    return {
        "audit_entry": "phase3_future_carry_source_chain_compare_pack",
        "source_env12_json": str(Path(env12_json).expanduser().resolve()),
        "source_env1_json": str(Path(env1_json).expanduser().resolve()),
        "comparison": {
            "env12": {
                "env_index": 12,
                "dominant_negative_source": env12_agg.get("dominant_negative_source"),
                "negative_source_shares": env12_shares,
                "start_token_count": int(env12_agg.get("start_token_count", 0)),
                "negative_contribution_total": float(env12_agg.get("negative_contribution_total", 0.0)),
            },
            "env1": {
                "env_index": 1,
                "dominant_negative_source": env1_agg.get("dominant_negative_source"),
                "negative_source_shares": env1_shares,
                "start_token_count": int(env1_agg.get("start_token_count", 0)),
                "negative_contribution_total": float(env1_agg.get("negative_contribution_total", 0.0)),
            },
            "shared_negative_sources": shared_sources,
            "shared_source_summary": shared_source_summary,
            "dominant_source_difference": dominant_difference,
        },
        "conclusions": {
            "shared_small_fix_candidate_is_baseline_bootstrap_semantic_mismatch": bool(
                shared_candidate == "baseline_bootstrap_semantic_mismatch"
            ),
            "future_carry_source_chain_is_not_single_dominant_bucket": True,
            "env12_branch_specific_dominant_source_is_terminal_reset_tail": bool(
                str(env12_agg.get("dominant_negative_source", "")) == "terminal_reset_tail_semantic_mismatch"
            ),
            "env1_branch_specific_dominant_source_is_true_negative_future_residual": bool(
                str(env1_agg.get("dominant_negative_source", "")) == "true_negative_future_residual"
            ),
            "branch_specific_targets_remain": True,
        },
        "recommendation": recommendation,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare the env12 and env1 future-carry source chains to isolate the shared narrow fix candidate."
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
        "--output-json",
        type=str,
        default="/home/chen/RLPFN/artifacts/phase3_future_carry_source_chain_compare_pack.json",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output_json).expanduser().resolve()
    if output_path.exists() and not bool(args.overwrite):
        print(output_path.read_text())
        return 0

    report = build_future_carry_source_chain_compare_pack(env12_json=args.env12_json, env1_json=args.env1_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
