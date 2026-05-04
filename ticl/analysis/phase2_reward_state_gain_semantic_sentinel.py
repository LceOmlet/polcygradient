import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any


DEFAULT_REJECTION_SUMMARY = (
    "/home/chen/RLPFN/artifacts/"
    "phase2_reward_state_gain_rejection_efficiency_seed30000_count2048_min058/summary.json"
)
DEFAULT_BUCKET_SUMMARY = (
    "/home/chen/RLPFN/artifacts/"
    "phase2_reward_state_gain_bucket_curve_analysis_horizon2to5_seed30000_count256_nanfix/summary.json"
)
DEFAULT_BUCKET_CSV = (
    "/home/chen/RLPFN/artifacts/"
    "phase2_reward_state_gain_bucket_curve_analysis_horizon2to5_seed30000_count256_nanfix/bucket_curves.csv"
)


def _load_json(path: str | Path) -> Any:
    resolved = Path(path).expanduser().resolve()
    with resolved.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    resolved = Path(path).expanduser().resolve()
    rows = []
    with resolved.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_csv(path: str | Path) -> list[dict[str, str]]:
    resolved = Path(path).expanduser().resolve()
    with resolved.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _check_equal(checks: list[dict[str, Any]], name: str, actual: Any, expected: Any) -> None:
    checks.append(
        {
            "name": name,
            "actual": actual,
            "expected": expected,
            "passed": bool(actual == expected),
        }
    )


def _check_close(
    checks: list[dict[str, Any]],
    name: str,
    actual: float,
    expected: float,
    *,
    atol: float = 1e-12,
) -> None:
    checks.append(
        {
            "name": name,
            "actual": float(actual),
            "expected": float(expected),
            "atol": float(atol),
            "passed": bool(math.isfinite(float(actual)) and abs(float(actual) - float(expected)) <= float(atol)),
        }
    )


def run_sentinel(args: argparse.Namespace) -> dict[str, Any]:
    rejection_summary_path = Path(args.rejection_summary).expanduser().resolve()
    bucket_summary_path = Path(args.bucket_summary).expanduser().resolve()
    bucket_csv_path = Path(args.bucket_csv).expanduser().resolve()
    rejection_summary = _load_json(rejection_summary_path)
    bucket_summary = _load_json(bucket_summary_path)
    bucket_rows = _load_csv(bucket_csv_path)
    topology_rows_path = Path(bucket_summary["rows_with_topology_jsonl"]).expanduser().resolve()
    topology_rows = _load_jsonl(topology_rows_path)

    threshold = float(args.threshold)
    topology_gains = [
        float(row["topology_reward_state_input_gain_fraction"])
        for row in topology_rows
    ]
    topology_legal_count = sum(1 for gain in topology_gains if math.isfinite(gain) and gain >= threshold)
    topology_count = len(topology_gains)
    csv_total_count = sum(int(row["row_count"]) for row in bucket_rows)
    csv_high_bucket_count = sum(
        int(row["row_count"])
        for row in bucket_rows
        if float(row["gain_q25"]) >= threshold
    )

    checks: list[dict[str, Any]] = []
    _check_equal(checks, "rejection_summary_count", int(rejection_summary["count"]), 2048)
    _check_equal(checks, "single_attempt_legal_count", int(rejection_summary["single_attempt"]["legal_count"]), 1215)
    _check_close(
        checks,
        "single_attempt_legal_rate",
        float(rejection_summary["single_attempt"]["legal_rate"]),
        1215.0 / 2048.0,
    )
    _check_equal(
        checks,
        "old_seed_only_rejection_accepted_count",
        int(rejection_summary["current_rejection_sampling"]["accepted_count"]),
        1228,
    )
    _check_close(
        checks,
        "old_seed_only_rejection_accepted_rate",
        float(rejection_summary["current_rejection_sampling"]["accepted_rate"]),
        1228.0 / 2048.0,
    )
    _check_equal(checks, "bucket_summary_row_count", int(bucket_summary["row_count"]), 256)
    _check_equal(checks, "bucket_csv_total_count", int(csv_total_count), 256)
    _check_equal(checks, "bucket_csv_row_count_list", [int(row["row_count"]) for row in bucket_rows], [51, 51, 51, 51, 52])
    _check_equal(checks, "bucket_topology_rows_count", int(topology_count), 256)
    _check_equal(checks, "bucket_topology_gain_ge_threshold_count", int(topology_legal_count), 154)
    _check_close(checks, "bucket_topology_gain_ge_threshold_rate", topology_legal_count / topology_count, 154.0 / 256.0)
    _check_equal(checks, "bucket_csv_high_bucket_count_by_gain_q25", int(csv_high_bucket_count), 154)
    _check_equal(checks, "bucket_csv_bucket_count", len(bucket_rows), 5)
    _check_equal(checks, "bucket_csv_threshold_boundary_label", bucket_rows[2]["bucket_label"].startswith("[0.580"), True)

    passed = all(bool(check["passed"]) for check in checks)
    report = {
        "audit_entry": "phase2_reward_state_gain_semantic_sentinel",
        "threshold": float(threshold),
        "passed": bool(passed),
        "rejection_summary": str(rejection_summary_path),
        "bucket_summary": str(bucket_summary_path),
        "bucket_csv": str(bucket_csv_path),
        "rows_with_topology_jsonl": str(topology_rows_path),
        "checks": checks,
    }
    if args.output_json:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Semantic sentinel for the trusted reward_state_input_gain_fraction old-contract artifacts. "
            "This verifies the original distribution facts; it is not a conditioned training sampler."
        )
    )
    parser.add_argument("--rejection-summary", default=DEFAULT_REJECTION_SUMMARY)
    parser.add_argument("--bucket-summary", default=DEFAULT_BUCKET_SUMMARY)
    parser.add_argument("--bucket-csv", default=DEFAULT_BUCKET_CSV)
    parser.add_argument("--threshold", type=float, default=0.580)
    parser.add_argument(
        "--output-json",
        default=(
            "/home/chen/RLPFN/artifacts/phase2_reward_state_gain_semantic_sentinel/"
            "summary.json"
        ),
    )
    return parser


def main() -> None:
    report = run_sentinel(build_arg_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if not bool(report["passed"]):
        sys.exit(1)


if __name__ == "__main__":
    main()
