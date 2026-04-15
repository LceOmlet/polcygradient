import argparse
import json
from pathlib import Path
from typing import Any


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return float(sum(float(row[key]) for row in rows) / float(len(rows)))


def _load_subset_rows(rollout_quality_json: str) -> list[dict[str, Any]]:
    payload = json.loads(Path(rollout_quality_json).expanduser().resolve().read_text())
    rows = [dict(row) for row in payload["env_items"] if "suffix_gap_delta" in row]
    if not rows:
        raise ValueError("No subset rows with suffix_gap_delta found in rollout quality artifact.")
    return rows


def _load_json(path: str | None) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        return None
    return json.loads(resolved.read_text())


def _median(values: list[float]) -> float:
    ordered = sorted(float(v) for v in values)
    n = len(ordered)
    if n <= 0:
        return 0.0
    mid = n // 2
    if n % 2 == 1:
        return float(ordered[mid])
    return float(0.5 * (ordered[mid - 1] + ordered[mid]))


def _group_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": int(len(rows)),
        "env_indices": [int(row["env_index"]) for row in rows],
        "mean_positive_mass": _mean(rows, "normalized_actor_adv_positive_mass"),
        "mean_suffix_gap_delta": _mean(rows, "suffix_gap_delta"),
        "mean_pre_suffix_gap": _mean(rows, "pre_vs_zero_suffix_gap"),
        "mean_value_corr": _mean(rows, "raw_value_return_corr"),
        "mean_first16_positive_share": _mean(rows, "normalized_actor_adv_first16_positive_share"),
        "mean_last16_positive_share": _mean(rows, "normalized_actor_adv_last16_positive_share"),
        "mean_episode_count": _mean(rows, "objective_episode_count"),
        "mean_terminal_reset_count": _mean(rows, "objective_terminal_reset_count"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Contrast high-positive-mass/no-gain vs low-mass/positive-gain train envs using existing Phase 3 lightweight artifacts."
    )
    parser.add_argument(
        "--rollout-quality-json",
        type=str,
        default="/home/chen/RLPFN/artifacts/phase3_train_rollout_quality_probe.json",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        required=True,
    )
    parser.add_argument("--suite-name", type=str, default=None)
    parser.add_argument("--suite-regime", type=str, default=None, choices=("positive", "nonpositive"))
    parser.add_argument("--expected-train-suite-summary-json", type=str, default=None)
    parser.add_argument("--suite-regime-summary-json", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output_json).expanduser().resolve()
    if output_path.exists() and not bool(args.overwrite):
        print(output_path.read_text())
        return 0

    rows = _load_subset_rows(args.rollout_quality_json)
    subset_mass_median = _median([float(row["normalized_actor_adv_positive_mass"]) for row in rows])

    high_mass_nonpositive = [
        row
        for row in rows
        if float(row["normalized_actor_adv_positive_mass"]) > float(subset_mass_median)
        and float(row["suffix_gap_delta"]) <= 0.0
    ]
    low_mass_positive = [
        row
        for row in rows
        if float(row["normalized_actor_adv_positive_mass"]) <= float(subset_mass_median)
        and float(row["suffix_gap_delta"]) > 0.0
    ]
    high_mass_positive = [
        row
        for row in rows
        if float(row["normalized_actor_adv_positive_mass"]) > float(subset_mass_median)
        and float(row["suffix_gap_delta"]) > 0.0
    ]

    suite_context: dict[str, Any] | None = None
    if args.suite_name is not None or args.suite_regime is not None:
        if args.suite_name is None or args.suite_regime is None:
            raise ValueError("suite-name and suite-regime must be provided together.")
        suite_context = {
            "suite_name": str(args.suite_name),
            "suite_regime": str(args.suite_regime),
            "suite_regime_summary_json": None
            if args.suite_regime_summary_json is None
            else str(Path(args.suite_regime_summary_json).expanduser().resolve()),
            "expected_train_suite_summary_json": None
            if args.expected_train_suite_summary_json is None
            else str(Path(args.expected_train_suite_summary_json).expanduser().resolve()),
        }
        current_train_fingerprint = str(
            json.loads(Path(args.rollout_quality_json).expanduser().resolve().read_text())["train_suite_summary"][
                "fingerprint"
            ]
        )
        suite_context["train_suite_fingerprint"] = current_train_fingerprint

        if args.expected_train_suite_summary_json is not None:
            expected_summary = _load_json(args.expected_train_suite_summary_json)
            if expected_summary is None:
                raise ValueError(
                    f"Missing expected train suite summary: {args.expected_train_suite_summary_json}"
                )
            expected_fingerprint = str(expected_summary.get("train_suite_summary", {}).get("fingerprint", ""))
            if not expected_fingerprint:
                expected_fingerprint = str(expected_summary.get("fingerprint", ""))
            if expected_fingerprint != current_train_fingerprint:
                raise ValueError(
                    "Train suite fingerprint mismatch for regime-labeled contrast "
                    f"(observed={current_train_fingerprint!r}, expected={expected_fingerprint!r})."
                )
            suite_context["train_suite_summary_fingerprint_match"] = True
        else:
            suite_context["train_suite_summary_fingerprint_match"] = None

        if args.suite_regime_summary_json is not None:
            regime_summary = _load_json(args.suite_regime_summary_json)
            if regime_summary is None:
                raise ValueError(f"Missing suite regime summary: {args.suite_regime_summary_json}")
            regime_groups = regime_summary.get("regime_groups", {})
            suite_names = list(regime_groups.get(str(args.suite_regime), {}).get("suite_names", []))
            if str(args.suite_name) not in {str(item) for item in suite_names}:
                raise ValueError(
                    "Suite name not present in the requested regime group "
                    f"({args.suite_name!r} not in {suite_names!r})."
                )
            suite_context["suite_regime_summary_membership_verified"] = True
        else:
            suite_context["suite_regime_summary_membership_verified"] = None

    report = {
        "audit_entry": "phase3_train_env_quality_contrast_probe",
        "source_rollout_quality_json": str(Path(args.rollout_quality_json).expanduser().resolve()),
        "subset_mass_median": float(subset_mass_median),
        "groups": {
            "high_mass_nonpositive": {
                "rows": high_mass_nonpositive,
                "summary": _group_summary(high_mass_nonpositive),
            },
            "low_mass_positive": {
                "rows": low_mass_positive,
                "summary": _group_summary(low_mass_positive),
            },
            "high_mass_positive": {
                "rows": high_mass_positive,
                "summary": _group_summary(high_mass_positive),
            },
        },
        "contrasts": {
            "low_mass_positive_minus_high_mass_nonpositive": {
                "pre_suffix_gap": _mean(low_mass_positive, "pre_vs_zero_suffix_gap")
                - _mean(high_mass_nonpositive, "pre_vs_zero_suffix_gap"),
                "raw_value_return_corr": _mean(low_mass_positive, "raw_value_return_corr")
                - _mean(high_mass_nonpositive, "raw_value_return_corr"),
                "first16_positive_share": _mean(low_mass_positive, "normalized_actor_adv_first16_positive_share")
                - _mean(high_mass_nonpositive, "normalized_actor_adv_first16_positive_share"),
                "last16_positive_share": _mean(low_mass_positive, "normalized_actor_adv_last16_positive_share")
                - _mean(high_mass_nonpositive, "normalized_actor_adv_last16_positive_share"),
                "episode_count": _mean(low_mass_positive, "objective_episode_count")
                - _mean(high_mass_nonpositive, "objective_episode_count"),
                "terminal_reset_count": _mean(low_mass_positive, "objective_terminal_reset_count")
                - _mean(high_mass_nonpositive, "objective_terminal_reset_count"),
            }
        },
        "suite_context": suite_context,
        "conclusions": {
            "new_phase3_trusted_baseline_established": False,
            "subset_only_diagnostic": True,
            "high_positive_mass_alone_is_not_sufficient_for_positive_delta": True,
            "low_mass_positive_envs_have_higher_pre_suffix_gap_than_high_mass_nonpositive": bool(
                _mean(low_mass_positive, "pre_vs_zero_suffix_gap")
                > _mean(high_mass_nonpositive, "pre_vs_zero_suffix_gap")
            ),
            "low_mass_positive_envs_have_higher_value_corr_than_high_mass_nonpositive": bool(
                _mean(low_mass_positive, "raw_value_return_corr")
                > _mean(high_mass_nonpositive, "raw_value_return_corr")
            ),
            "early_mass_distinguishes_better_than_tail_mass": bool(
                (
                    _mean(low_mass_positive, "normalized_actor_adv_first16_positive_share")
                    - _mean(high_mass_nonpositive, "normalized_actor_adv_first16_positive_share")
                )
                > abs(
                    _mean(low_mass_positive, "normalized_actor_adv_last16_positive_share")
                    - _mean(high_mass_nonpositive, "normalized_actor_adv_last16_positive_share")
                )
            ),
            "episode_segmentation_not_primary_separator": bool(
                abs(
                    _mean(low_mass_positive, "objective_episode_count")
                    - _mean(high_mass_nonpositive, "objective_episode_count")
                )
                < 0.5
                and abs(
                    _mean(low_mass_positive, "objective_terminal_reset_count")
                    - _mean(high_mass_nonpositive, "objective_terminal_reset_count")
                )
                < 0.5
            ),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
