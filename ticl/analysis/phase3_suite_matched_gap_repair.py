import argparse
import json
from pathlib import Path
from typing import Any

from ticl.analysis.phase3_post_reset_tail_distance_probe import (
    RESET_KS,
    TAIL_LEN,
    _summary as _distance_summary,
)
from ticl.analysis.phase3_terminal_tail_mask_probe import (
    MASK_RULES,
    TAIL_LENS,
    _summary as _tailmask_summary,
)


def _load_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).expanduser().resolve().read_text())


def _suite_matched_pre_gap_by_env(zero_json: str, ppo_json: str) -> dict[int, float]:
    zero = _load_json(zero_json)["heldout_suite"]["metrics"]["suffix_return_per_env"]
    ppo = _load_json(ppo_json)["heldout_suite"]["metrics"]["suffix_return_per_env"]
    return {int(idx): float(float(ppo[idx]) - float(zero[idx])) for idx in range(len(zero))}


def _repair_distance_probe(payload: dict[str, Any], expected_pre_gap: dict[int, float]) -> dict[str, Any]:
    env_rows = [dict(row) for row in payload["env_rows"]]
    mismatch_rows: list[dict[str, Any]] = []
    for row in env_rows:
        env_idx = int(row["env_index"])
        observed = float(row["pre_vs_zero_suffix_gap"])
        expected = float(expected_pre_gap[env_idx])
        row["pre_vs_zero_suffix_gap"] = expected
        if abs(observed - expected) > 1e-6:
            mismatch_rows.append(
                {
                    "env_index": env_idx,
                    "observed_pre_vs_zero_suffix_gap": observed,
                    "expected_pre_vs_zero_suffix_gap": expected,
                }
            )

    summaries: dict[str, Any] = {}
    for dist in range(1, int(TAIL_LEN) + 1):
        summaries[f"post_reset_tail_dist_{dist}"] = _distance_summary(
            env_rows,
            mass_key=f"post_reset_tail_dist_{dist}",
        )
    for k in RESET_KS:
        summaries[f"post_reset_early_k{k}"] = _distance_summary(
            env_rows,
            mass_key=f"post_reset_early_k{k}",
        )
        summaries[f"post_reset_late_k{k}"] = _distance_summary(
            env_rows,
            mass_key=f"post_reset_late_k{k}",
        )

    repaired = dict(payload)
    repaired["env_rows"] = env_rows
    repaired["summaries"] = summaries
    repaired["repair_metadata"] = {
        "repair_type": "suite_matched_pre_gap_repair",
        "mismatch_count": int(len(mismatch_rows)),
        "mismatch_rows": mismatch_rows,
    }
    return repaired


def _repair_tailmask_probe(payload: dict[str, Any], expected_pre_gap: dict[int, float]) -> dict[str, Any]:
    env_rows = [dict(row) for row in payload["env_rows"]]
    mismatch_rows: list[dict[str, Any]] = []
    for row in env_rows:
        env_idx = int(row["env_index"])
        observed = float(row["pre_vs_zero_suffix_gap"])
        expected = float(expected_pre_gap[env_idx])
        row["pre_vs_zero_suffix_gap"] = expected
        if abs(observed - expected) > 1e-6:
            mismatch_rows.append(
                {
                    "env_index": env_idx,
                    "observed_pre_vs_zero_suffix_gap": observed,
                    "expected_pre_vs_zero_suffix_gap": expected,
                }
            )

    summaries: dict[str, Any] = {}
    for tail_len in TAIL_LENS:
        tail_len_key = str(tail_len)
        per_variant = {}
        for variant in MASK_RULES:
            items = [
                {
                    "env_index": row["env_index"],
                    "objective_terminal_reset_count": row["objective_terminal_reset_count"],
                    "pre_vs_zero_suffix_gap": row["pre_vs_zero_suffix_gap"],
                    "raw_value_return_corr": row["raw_value_return_corr"],
                    "positive_mass": row["tail_len_rows"][tail_len_key][variant]["remaining_positive_mass"],
                }
                for row in env_rows
            ]
            per_variant[variant] = _tailmask_summary(items, mass_key="positive_mass")
        summaries[tail_len_key] = per_variant

    repaired = dict(payload)
    repaired["env_rows"] = env_rows
    repaired["summaries"] = summaries
    repaired["repair_metadata"] = {
        "repair_type": "suite_matched_pre_gap_repair",
        "mismatch_count": int(len(mismatch_rows)),
        "mismatch_rows": mismatch_rows,
    }
    return repaired


def repair_artifact(
    *,
    artifact_json: str,
    zero_json: str,
    ppo_json: str,
) -> dict[str, Any]:
    payload = _load_json(artifact_json)
    expected_pre_gap = _suite_matched_pre_gap_by_env(zero_json, ppo_json)
    audit_entry = str(payload.get("audit_entry", ""))
    if audit_entry == "phase3_post_reset_tail_distance_probe":
        return _repair_distance_probe(payload, expected_pre_gap)
    if audit_entry == "phase3_terminal_tail_mask_probe":
        return _repair_tailmask_probe(payload, expected_pre_gap)
    raise ValueError(f"Unsupported artifact for suite-matched gap repair: {audit_entry!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Repair stale per-env pre-gap fields using suite-matched zero/PPO baselines.")
    parser.add_argument("--artifact-json", type=str, required=True)
    parser.add_argument("--zero-json", type=str, required=True)
    parser.add_argument("--ppo-json", type=str, required=True)
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output_json).expanduser().resolve()
    if output_path.exists() and not bool(args.overwrite):
        print(output_path.read_text())
        return 0

    repaired = repair_artifact(
        artifact_json=str(args.artifact_json),
        zero_json=str(args.zero_json),
        ppo_json=str(args.ppo_json),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(repaired, indent=2, sort_keys=True))
    print(output_path.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
