import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_SUITES = [
    {
        "suite_name": "pair1",
        "zero_json": "/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/zero_cross_env_control.json",
        "ppo_json": "/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/ppo_cross_env_baseline.json",
        "distance_probe_json": "/home/chen/RLPFN/artifacts/phase3_post_reset_tail_distance_probe_pair1_suite_matched.json",
        "tail_ablation_json": "/home/chen/RLPFN/artifacts/phase3_post_reset_tail_ablation_verify_pair1_suite_matched.json",
        "post_reset_ablation_json": "/home/chen/RLPFN/artifacts/phase3_post_reset_ablation_verify_pair1_suite_matched.json",
    },
    {
        "suite_name": "pair2",
        "zero_json": "/home/chen/RLPFN/artifacts/phase3_cross_env_baseline_pair2/zero_cross_env_control.json",
        "ppo_json": "/home/chen/RLPFN/artifacts/phase3_cross_env_baseline_pair2/ppo_cross_env_baseline.json",
        "distance_probe_json": "/home/chen/RLPFN/artifacts/phase3_post_reset_tail_distance_probe_pair2.json",
        "tail_ablation_json": None,
        "post_reset_ablation_json": None,
    },
    {
        "suite_name": "seed24680",
        "zero_json": "/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_24680/zero_cross_env_control.json",
        "ppo_json": "/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_24680/ppo_cross_env_baseline.json",
        "distance_probe_json": "/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_24680/phase3_post_reset_tail_distance_probe.json",
        "tail_ablation_json": "/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_24680/phase3_post_reset_tail_ablation_verify.json",
        "post_reset_ablation_json": "/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_24680/phase3_post_reset_ablation_verify.json",
    },
    {
        "suite_name": "seed13579",
        "zero_json": "/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_13579/zero_cross_env_control.json",
        "ppo_json": "/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_13579/ppo_cross_env_baseline.json",
        "distance_probe_json": "/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_13579/phase3_post_reset_tail_distance_probe.json",
        "tail_ablation_json": "/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_13579/phase3_post_reset_tail_ablation_verify.json",
        "post_reset_ablation_json": "/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/seed_13579/phase3_post_reset_ablation_verify.json",
    },
]


def _load_json(path: str | None) -> dict[str, Any] | None:
    if path is None:
        return None
    payload_path = Path(path).expanduser().resolve()
    if not payload_path.exists():
        return None
    return json.loads(payload_path.read_text())


def _sign_label(value: float) -> str:
    return "positive" if float(value) > 0.0 else "nonpositive"


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(float(v) for v in values) / len(values))


def _suite_summary(spec: dict[str, Any]) -> dict[str, Any]:
    zero = _load_json(spec.get("zero_json"))
    ppo = _load_json(spec.get("ppo_json"))
    distance = _load_json(spec.get("distance_probe_json"))
    tail_ablation = _load_json(spec.get("tail_ablation_json"))
    post_reset_ablation = _load_json(spec.get("post_reset_ablation_json"))

    if zero is None or ppo is None:
        return {
            "suite_name": str(spec["suite_name"]),
            "available": False,
            "missing": {
                "zero_json": zero is None,
                "ppo_json": ppo is None,
            },
        }

    heldout_suffix_zero = float(zero["heldout_suite"]["metrics"]["suffix_return_mean"])
    heldout_suffix_ppo = float(ppo["heldout_suite"]["metrics"]["suffix_return_mean"])
    heldout_full_zero = float(zero["heldout_suite"]["metrics"]["full_return_mean"])
    heldout_full_ppo = float(ppo["heldout_suite"]["metrics"]["full_return_mean"])
    ppo_minus_zero_suffix = float(heldout_suffix_ppo - heldout_suffix_zero)
    ppo_minus_zero_full = float(heldout_full_ppo - heldout_full_zero)

    out: dict[str, Any] = {
        "suite_name": str(spec["suite_name"]),
        "available": True,
        "suite_regime": _sign_label(ppo_minus_zero_suffix),
        "baseline": {
            "heldout_suffix_zero": heldout_suffix_zero,
            "heldout_suffix_ppo": heldout_suffix_ppo,
            "heldout_full_zero": heldout_full_zero,
            "heldout_full_ppo": heldout_full_ppo,
            "ppo_minus_zero_suffix": ppo_minus_zero_suffix,
            "ppo_minus_zero_full": ppo_minus_zero_full,
        },
    }

    if distance is not None:
        dist_keys = [f"post_reset_tail_dist_{idx}" for idx in range(1, 5)]
        dist_rows = {key: distance["summaries"][key] for key in dist_keys}
        out["distance_probe"] = {
            "dist1_to_4": dist_rows,
            "dist1_to_4_mean_corr_reset": _mean(
                [float(dist_rows[key]["corr_vs_terminal_resets"]) for key in dist_keys]
            ),
            "dist1_to_4_mean_corr_pre": _mean(
                [float(dist_rows[key]["corr_vs_pre_suffix_gap"]) for key in dist_keys]
            ),
        }

    if tail_ablation is not None:
        out["post_reset_tail_ablation"] = {
            "corr_reset_delta": float(tail_ablation["deltas"]["corr_reset_delta"]),
            "corr_pre_delta": float(tail_ablation["deltas"]["corr_pre_delta"]),
        }

    if post_reset_ablation is not None:
        out["post_reset_ablation"] = {
            "corr_reset_delta": float(post_reset_ablation["deltas"]["corr_reset_delta"]),
            "corr_pre_delta": float(post_reset_ablation["deltas"]["corr_pre_delta"]),
        }

    return out


def _group_by_regime(suites: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {"positive": [], "nonpositive": []}
    for suite in suites:
        if not bool(suite.get("available", False)):
            continue
        groups[str(suite["suite_regime"])].append(suite)

    summary: dict[str, Any] = {}
    for regime, items in groups.items():
        dist_reset = []
        dist_pre = []
        tail_reset = []
        tail_pre = []
        post_reset_reset = []
        post_reset_pre = []
        suite_names = []
        for item in items:
            suite_names.append(str(item["suite_name"]))
            if "distance_probe" in item:
                dist_reset.append(float(item["distance_probe"]["dist1_to_4_mean_corr_reset"]))
                dist_pre.append(float(item["distance_probe"]["dist1_to_4_mean_corr_pre"]))
            if "post_reset_tail_ablation" in item:
                tail_reset.append(float(item["post_reset_tail_ablation"]["corr_reset_delta"]))
                tail_pre.append(float(item["post_reset_tail_ablation"]["corr_pre_delta"]))
            if "post_reset_ablation" in item:
                post_reset_reset.append(float(item["post_reset_ablation"]["corr_reset_delta"]))
                post_reset_pre.append(float(item["post_reset_ablation"]["corr_pre_delta"]))
        summary[regime] = {
            "suite_names": suite_names,
            "count": int(len(items)),
            "dist1_to_4_mean_corr_reset": _mean(dist_reset),
            "dist1_to_4_mean_corr_pre": _mean(dist_pre),
            "post_reset_tail_ablation_mean_corr_reset_delta": _mean(tail_reset),
            "post_reset_tail_ablation_mean_corr_pre_delta": _mean(tail_pre),
            "post_reset_ablation_mean_corr_reset_delta": _mean(post_reset_reset),
            "post_reset_ablation_mean_corr_pre_delta": _mean(post_reset_pre),
        }
    return summary


def build_cross_suite_regime_summary(suite_specs: list[dict[str, Any]]) -> dict[str, Any]:
    suites = [_suite_summary(spec) for spec in suite_specs]
    available = [suite for suite in suites if bool(suite.get("available", False))]
    return {
        "audit_entry": "phase3_cross_suite_regime_summary",
        "suite_summaries": suites,
        "available_suite_count": int(len(available)),
        "regime_groups": _group_by_regime(suites),
        "conclusions": {
            "dist1_to_4_reset_tracking_consistent": bool(
                available
                and all(
                    float(suite.get("distance_probe", {}).get("dist1_to_4_mean_corr_reset", 0.0)) > 0.0
                    for suite in available
                    if "distance_probe" in suite
                )
            ),
            "dist1_to_4_pre_alignment_consistent": bool(
                available
                and len(
                    {
                        _sign_label(float(suite.get("distance_probe", {}).get("dist1_to_4_mean_corr_pre", 0.0)))
                        for suite in available
                        if "distance_probe" in suite
                    }
                )
                == 1
            ),
            "post_reset_tail_ablation_pre_improvement_consistent": bool(
                available
                and all(
                    float(suite.get("post_reset_tail_ablation", {}).get("corr_pre_delta", 0.0)) > 0.0
                    for suite in available
                    if "post_reset_tail_ablation" in suite
                )
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate Phase 3 cross-suite results into regime-level summaries.")
    parser.add_argument("--suite-specs-json", type=str, default=None)
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output_json).expanduser().resolve()
    if output_path.exists() and not bool(args.overwrite):
        print(output_path.read_text())
        return 0

    suite_specs = DEFAULT_SUITES
    if args.suite_specs_json is not None:
        suite_specs = json.loads(Path(args.suite_specs_json).expanduser().resolve().read_text())

    result = build_cross_suite_regime_summary(suite_specs)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(output_path.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
