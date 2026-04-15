import argparse
import json
import time
from pathlib import Path
from typing import Any

from ticl.analysis.phase2_guardrail import DEFAULT_PHASE2_SUMMARY, assert_phase2_green
from ticl.analysis.phase3_train_token_bucket_probe import BUCKET_NAMES


def _load_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).expanduser().resolve().read_text())


def _mean(rows: list[dict[str, Any]], getter) -> float:
    if not rows:
        return 0.0
    return float(sum(float(getter(row)) for row in rows) / len(rows))


def _positive_mass_corrs(payload: dict[str, Any]) -> dict[str, float]:
    return {
        bucket: float(payload["bucket_alignment"][bucket]["corr_delta_vs_positive_mass_share"])
        for bucket in BUCKET_NAMES
    }


def _sign(x: float) -> int:
    if x > 0.0:
        return 1
    if x < 0.0:
        return -1
    return 0


def _group_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": int(len(rows)),
        "env_indices": [int(row["env_index"]) for row in rows],
        "mean_pre_vs_zero_suffix_gap": _mean(rows, lambda row: row["pre_vs_zero_suffix_gap"]),
        "mean_suffix_gap_delta": _mean(rows, lambda row: row["suffix_gap_delta"]),
        "bucket_positive_mass_share_means": {
            bucket: _mean(rows, lambda row, b=bucket: row["bucket_metrics"][b]["positive_mass_share"])
            for bucket in BUCKET_NAMES
        },
        "bucket_value_corr_means": {
            bucket: _mean(rows, lambda row, b=bucket: row["bucket_metrics"][b]["raw_value_return_corr"])
            for bucket in BUCKET_NAMES
        },
    }


def main() -> int:
    t0 = time.perf_counter()
    parser = argparse.ArgumentParser(
        description="Phase 3 lightweight transfer bottleneck summary from canonical train/heldout token-bucket artifacts."
    )
    parser.add_argument(
        "--train-token-bucket-json",
        type=str,
        default="/home/chen/RLPFN/artifacts/phase3_train_token_bucket_probe.json",
    )
    parser.add_argument(
        "--heldout-token-bucket-json",
        type=str,
        default="/home/chen/RLPFN/artifacts/phase3_heldout_token_bucket_probe.json",
    )
    parser.add_argument("--phase2-summary-path", type=str, default=DEFAULT_PHASE2_SUMMARY)
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output_json).expanduser().resolve()
    if output_path.exists() and not bool(args.overwrite):
        print(output_path.read_text())
        return 0

    phase2_summary = assert_phase2_green(args.phase2_summary_path)
    train_payload = _load_json(args.train_token_bucket_json)
    heldout_payload = _load_json(args.heldout_token_bucket_json)

    train_rows = list(train_payload["subset_rows"])
    heldout_rows = list(heldout_payload["subset_rows"])
    train_pos = [row for row in train_rows if float(row["suffix_gap_delta"]) > 0.0]
    train_nonpos = [row for row in train_rows if float(row["suffix_gap_delta"]) <= 0.0]
    heldout_pos = [row for row in heldout_rows if float(row["suffix_gap_delta"]) > 0.0]
    heldout_nonpos = [row for row in heldout_rows if float(row["suffix_gap_delta"]) <= 0.0]

    train_corrs = _positive_mass_corrs(train_payload)
    heldout_corrs = _positive_mass_corrs(heldout_payload)
    sign_flip_buckets = [bucket for bucket in BUCKET_NAMES if _sign(train_corrs[bucket]) == -_sign(heldout_corrs[bucket])]

    train_pos_summary = _group_summary(train_pos)
    train_nonpos_summary = _group_summary(train_nonpos)
    heldout_pos_summary = _group_summary(heldout_pos)
    heldout_nonpos_summary = _group_summary(heldout_nonpos)

    train_q12 = float(
        train_pos_summary["bucket_positive_mass_share_means"]["q1"]
        + train_pos_summary["bucket_positive_mass_share_means"]["q2"]
    )
    train_q34 = float(
        train_pos_summary["bucket_positive_mass_share_means"]["q3"]
        + train_pos_summary["bucket_positive_mass_share_means"]["q4"]
    )
    heldout_q12 = float(
        heldout_pos_summary["bucket_positive_mass_share_means"]["q1"]
        + heldout_pos_summary["bucket_positive_mass_share_means"]["q2"]
    )
    heldout_q34 = float(
        heldout_pos_summary["bucket_positive_mass_share_means"]["q3"]
        + heldout_pos_summary["bucket_positive_mass_share_means"]["q4"]
    )

    result = {
        "audit_entry": "phase3_transfer_bottleneck_probe",
        "config": {
            "runtime_wall_s": float(time.perf_counter() - t0),
            "train_token_bucket_json": str(Path(args.train_token_bucket_json).expanduser().resolve()),
            "heldout_token_bucket_json": str(Path(args.heldout_token_bucket_json).expanduser().resolve()),
        },
        "phase2_preflight": {
            "required": True,
            "summary_path": str(Path(args.phase2_summary_path).expanduser().resolve()),
            "phase2_all_checks_pass": bool(phase2_summary["validation"]["all_checks_pass"]),
            "phase2_pass_flags": dict(phase2_summary["pass_flags"]),
        },
        "bucket_positive_mass_corrs": {
            "train": train_corrs,
            "heldout": heldout_corrs,
        },
        "group_summaries": {
            "train_positive": train_pos_summary,
            "train_nonpositive": train_nonpos_summary,
            "heldout_positive": heldout_pos_summary,
            "heldout_nonpositive": heldout_nonpos_summary,
        },
        "comparisons": {
            "sign_flip_buckets": sign_flip_buckets,
            "train_positive_q12_minus_q34": float(train_q12 - train_q34),
            "heldout_positive_q12_minus_q34": float(heldout_q12 - heldout_q34),
            "train_positive_pre_gap_minus_nonpositive": float(
                train_pos_summary["mean_pre_vs_zero_suffix_gap"] - train_nonpos_summary["mean_pre_vs_zero_suffix_gap"]
            ),
            "heldout_positive_pre_gap_minus_nonpositive": float(
                heldout_pos_summary["mean_pre_vs_zero_suffix_gap"] - heldout_nonpos_summary["mean_pre_vs_zero_suffix_gap"]
            ),
        },
        "conclusions": {
            "all_bucket_alignment_signs_flip_between_train_and_heldout": len(sign_flip_buckets) == len(BUCKET_NAMES),
            "train_positive_cases_are_q34_dominated": bool(train_q34 > train_q12),
            "heldout_positive_cases_are_q12_dominated": bool(heldout_q12 > heldout_q34),
            "heldout_positive_cases_have_lower_pre_gap_than_heldout_nonpositive": bool(
                heldout_pos_summary["mean_pre_vs_zero_suffix_gap"] < heldout_nonpos_summary["mean_pre_vs_zero_suffix_gap"]
            ),
            "train_positive_cases_have_higher_pre_gap_than_train_nonpositive": bool(
                train_pos_summary["mean_pre_vs_zero_suffix_gap"] > train_nonpos_summary["mean_pre_vs_zero_suffix_gap"]
            ),
            "late_bucket_success_signature_transfers_to_heldout": False,
            "phase3_transfer_pattern_is_mode_inversion_not_simple_weakening": bool(
                len(sign_flip_buckets) == len(BUCKET_NAMES)
                and train_q34 > train_q12
                and heldout_q12 > heldout_q34
            ),
            "new_phase3_trusted_baseline_established": False,
            "diagnostic_only": True,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(output_path.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
