import argparse
import json
import time
from pathlib import Path
from typing import Any

from ticl.analysis.phase2_guardrail import DEFAULT_PHASE2_SUMMARY, assert_phase2_green


def _load_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).expanduser().resolve().read_text())


def _topk_pre_gap_mass_share(payload: dict[str, Any], k: int) -> dict[str, Any]:
    items = list(payload["env_items"])
    ranked = sorted(items, key=lambda item: float(item["pre_vs_zero_suffix_gap"]), reverse=True)
    total_pos = float(sum(float(item["normalized_actor_adv_positive_mass"]) for item in items))
    top = ranked[: int(k)]
    top_mass = float(sum(float(item["normalized_actor_adv_positive_mass"]) for item in top))
    return {
        "k": int(k),
        "env_indices": [int(item["env_index"]) for item in top],
        "mean_pre_gap": float(sum(float(item["pre_vs_zero_suffix_gap"]) for item in top) / max(1, len(top))),
        "positive_mass_share": float(top_mass / total_pos) if total_pos > 0.0 else 0.0,
    }


def _corr(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys):
        raise ValueError("Correlation inputs must match in length.")
    if len(xs) <= 1:
        return 0.0
    mean_x = float(sum(xs) / len(xs))
    mean_y = float(sum(ys) / len(ys))
    cx = [float(x) - mean_x for x in xs]
    cy = [float(y) - mean_y for y in ys]
    denom_x = float(sum(x * x for x in cx)) ** 0.5
    denom_y = float(sum(y * y for y in cy)) ** 0.5
    if denom_x <= 0.0 or denom_y <= 0.0:
        return 0.0
    return float(sum(x * y for x, y in zip(cx, cy)) / (denom_x * denom_y))


def _raw_positive_mass_coupling(payload: dict[str, Any]) -> dict[str, float]:
    items = list(payload["env_items"])
    raw_pos = [float(item["raw_actor_adv_positive_mass"]) for item in items]
    resets = [float(item["objective_terminal_reset_count"]) for item in items]
    pre_gap = [float(item["pre_vs_zero_suffix_gap"]) for item in items]
    value_corr = [float(item["raw_value_return_corr"]) for item in items]
    return {
        "corr_raw_positive_mass_vs_terminal_resets": _corr(raw_pos, resets),
        "corr_raw_positive_mass_vs_pre_suffix_gap": _corr(raw_pos, pre_gap),
        "corr_raw_positive_mass_vs_value_corr": _corr(raw_pos, value_corr),
    }


def main() -> int:
    t0 = time.perf_counter()
    parser = argparse.ArgumentParser(
        description="Phase 3 root-cause synthesis over canonical lightweight diagnostics."
    )
    parser.add_argument(
        "--train-rollout-quality-json",
        type=str,
        default="/home/chen/RLPFN/artifacts/phase3_train_rollout_quality_probe.json",
    )
    parser.add_argument(
        "--heldout-rollout-quality-json",
        type=str,
        default="/home/chen/RLPFN/artifacts/phase3_heldout_rollout_quality_probe.json",
    )
    parser.add_argument(
        "--transfer-bottleneck-json",
        type=str,
        default="/home/chen/RLPFN/artifacts/phase3_transfer_bottleneck_probe.json",
    )
    parser.add_argument(
        "--sequence-layout-json",
        type=str,
        default="/home/chen/RLPFN/artifacts/phase3_shortrun_sequence_layout_source_probe.json",
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
    train_payload = _load_json(args.train_rollout_quality_json)
    heldout_payload = _load_json(args.heldout_rollout_quality_json)
    transfer_payload = _load_json(args.transfer_bottleneck_json)
    seq_payload = _load_json(args.sequence_layout_json)

    train_rollout = train_payload["rollout_summary"]
    heldout_rollout = heldout_payload["rollout_summary"]
    train_raw_coupling = _raw_positive_mass_coupling(train_payload)
    heldout_raw_coupling = _raw_positive_mass_coupling(heldout_payload)
    seq_conclusion = str(seq_payload.get("conclusion", ""))
    seq_stable = "does not reproduce" in seq_conclusion or "stable" in seq_conclusion

    train_top2 = _topk_pre_gap_mass_share(train_payload, 2)
    heldout_top2 = _topk_pre_gap_mass_share(heldout_payload, 2)
    train_top4 = _topk_pre_gap_mass_share(train_payload, 4)
    heldout_top4 = _topk_pre_gap_mass_share(heldout_payload, 4)

    result = {
        "audit_entry": "phase3_root_cause_probe",
        "config": {
            "runtime_wall_s": float(time.perf_counter() - t0),
            "train_rollout_quality_json": str(Path(args.train_rollout_quality_json).expanduser().resolve()),
            "heldout_rollout_quality_json": str(Path(args.heldout_rollout_quality_json).expanduser().resolve()),
            "transfer_bottleneck_json": str(Path(args.transfer_bottleneck_json).expanduser().resolve()),
            "sequence_layout_json": str(Path(args.sequence_layout_json).expanduser().resolve()),
        },
        "phase2_preflight": {
            "required": True,
            "summary_path": str(Path(args.phase2_summary_path).expanduser().resolve()),
            "phase2_all_checks_pass": bool(phase2_summary["validation"]["all_checks_pass"]),
            "phase2_pass_flags": dict(phase2_summary["pass_flags"]),
        },
        "comparisons": {
            "train_vs_heldout_positive_mass_coupling": {
                "raw_positive_mass": {
                    "train": train_raw_coupling,
                    "heldout": heldout_raw_coupling,
                },
                "corr_pre_suffix_gap_vs_positive_norm_mass": {
                    "train": float(train_rollout["corr_pre_suffix_gap_vs_positive_norm_mass"]),
                    "heldout": float(heldout_rollout["corr_pre_suffix_gap_vs_positive_norm_mass"]),
                },
                "corr_value_corr_vs_positive_norm_mass": {
                    "train": float(train_rollout["corr_value_corr_vs_positive_norm_mass"]),
                    "heldout": float(heldout_rollout["corr_value_corr_vs_positive_norm_mass"]),
                },
                "corr_terminal_resets_vs_positive_norm_mass": {
                    "train": float(train_rollout["corr_terminal_resets_vs_positive_norm_mass"]),
                    "heldout": float(heldout_rollout["corr_terminal_resets_vs_positive_norm_mass"]),
                },
                "corr_last16_positive_share_vs_positive_norm_mass": {
                    "train": float(train_rollout["corr_last16_positive_share_vs_positive_norm_mass"]),
                    "heldout": float(heldout_rollout["corr_last16_positive_share_vs_positive_norm_mass"]),
                },
            },
            "high_pre_gap_env_capture": {
                "train_top2": train_top2,
                "heldout_top2": heldout_top2,
                "train_top4": train_top4,
                "heldout_top4": heldout_top4,
            },
            "transfer_pattern": {
                "all_bucket_alignment_signs_flip_between_train_and_heldout": bool(
                    transfer_payload["conclusions"]["all_bucket_alignment_signs_flip_between_train_and_heldout"]
                ),
                "phase3_transfer_pattern_is_mode_inversion_not_simple_weakening": bool(
                    transfer_payload["conclusions"]["phase3_transfer_pattern_is_mode_inversion_not_simple_weakening"]
                ),
            },
            "sequence_layout_probe": {
                "collect_path_stable": bool(seq_stable),
                "conclusion": seq_conclusion,
            },
        },
        "conclusions": {
            "active_collect_layout_bug_supported": False,
            "task_unlearnable_supported": False,
            "optimization_selection_bias_supported": bool(
                transfer_payload["conclusions"]["phase3_transfer_pattern_is_mode_inversion_not_simple_weakening"]
                and float(heldout_rollout["corr_terminal_resets_vs_positive_norm_mass"]) > 0.9
                and float(heldout_top2["positive_mass_share"]) < 0.01
            ),
            "normalization_is_not_primary_driver": bool(
                float(abs(heldout_raw_coupling["corr_raw_positive_mass_vs_terminal_resets"]
                          - heldout_rollout["corr_terminal_resets_vs_positive_norm_mass"])) < 0.05
            ),
            "heldout_positive_mass_is_reset_heavy": bool(
                float(heldout_rollout["corr_terminal_resets_vs_positive_norm_mass"]) > 0.9
            ),
            "heldout_clean_high_pre_gap_envs_are_starved": bool(
                float(heldout_top2["positive_mass_share"]) < 0.01
            ),
            "train_to_heldout_transfer_is_mode_inversion": bool(
                transfer_payload["conclusions"]["phase3_transfer_pattern_is_mode_inversion_not_simple_weakening"]
            ),
            "sequence_layout_removed_as_current_root_cause": bool(seq_stable),
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
