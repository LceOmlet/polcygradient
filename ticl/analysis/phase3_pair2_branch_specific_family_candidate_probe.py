import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np


DEFAULT_BASELINE_SNAPSHOT_JSON = (
    "/home/chen/RLPFN/artifacts/phase3_pair2_episode_complete_train_update_ab_baseline_outer_batch_snapshot.json"
)
DEFAULT_SCOPE_CLASS_PROBE_JSON = (
    "/home/chen/RLPFN/artifacts/phase3_pair2_branch_specific_extension_class_probe.json"
)
DEFAULT_OUTPUT_JSON = (
    "/home/chen/RLPFN/artifacts/phase3_pair2_branch_specific_family_candidate_probe.json"
)


def _load_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def _share(numerator: float, denominator: float) -> float:
    denominator = float(denominator)
    if abs(denominator) <= 1e-12:
        return 0.0
    return float(float(numerator) / denominator)


def _as_int_array(payload: dict[str, Any], key: str) -> np.ndarray:
    return np.asarray(payload[key], dtype=np.int64).reshape(-1)


def _as_float_array(payload: dict[str, Any], key: str) -> np.ndarray:
    return np.asarray(payload[key], dtype=np.float64).reshape(-1)


def _candidate_metrics(
    *,
    name: str,
    description: str,
    mask: np.ndarray,
    env_objective_mask: np.ndarray,
    objective_mask: np.ndarray,
    clip: np.ndarray,
    env_indices: np.ndarray,
    objective_episode_indices: np.ndarray,
    total_outer_batches: int,
) -> dict[str, Any]:
    token_count = int(mask.sum())
    negative_mass = float(np.abs(clip[mask & (clip < 0.0)]).sum()) if token_count > 0 else 0.0
    objective_negative_mass = float(np.abs(clip[objective_mask & (clip < 0.0)]).sum())
    env_negative_mass = float(np.abs(clip[env_objective_mask & (clip < 0.0)]).sum())
    policy_share = _share(negative_mass, objective_negative_mass)
    return {
        "name": str(name),
        "description": str(description),
        "token_count": int(token_count),
        "policy_negative_mass": float(negative_mass),
        "policy_share_of_batch_negative_mass": float(policy_share),
        "token_share_of_env12_objective": float(_share(token_count, int(env_objective_mask.sum()))),
        "policy_share_of_env12_negative_mass": float(_share(negative_mass, env_negative_mass)),
        "epoch_uniform_share_heuristic": float(_share(policy_share, float(total_outer_batches))),
        "episode_indices": sorted({int(v) for v in objective_episode_indices[mask].tolist()}),
        "env_indices": sorted({int(v) for v in env_indices[mask].tolist()}),
    }


def build_pair2_branch_specific_family_candidate_probe(
    *,
    baseline_snapshot_json: str,
    scope_class_probe_json: str,
) -> dict[str, Any]:
    baseline_snapshot = _load_json(baseline_snapshot_json)
    scope_probe = _load_json(scope_class_probe_json)
    if str(scope_probe.get("probe_entry", "")) != "phase3_pair2_branch_specific_extension_class_probe":
        raise ValueError("Expected canonical branch-specific extension class probe artifact.")

    objective_mask = _as_int_array(baseline_snapshot, "objective_mask") > 0
    env_indices = _as_int_array(baseline_snapshot, "flat_env_indices")
    objective_episode_indices = _as_int_array(baseline_snapshot, "flat_objective_episode_indices")
    objective_episode_positions = _as_int_array(baseline_snapshot, "flat_objective_episode_positions")
    clip = _as_float_array(baseline_snapshot, "clipped_objective")
    if not bool(np.any(objective_mask)):
        raise ValueError("Expected objective tokens in the baseline snapshot.")

    total_outer_batches = int(scope_probe["current_trusted_contract"]["total_outer_batches"])
    env12_objective_mask = objective_mask & (env_indices == 12)
    if not bool(np.any(env12_objective_mask)):
        raise ValueError("Expected env12 objective tokens in the baseline snapshot.")

    candidate_fns: list[tuple[str, str, Callable[[np.ndarray, np.ndarray], np.ndarray]]] = [
        (
            "selector_matched_episode_4_complete",
            "Current trusted same-class selector-matched full episode baseline.",
            lambda epi, pos: epi == 4,
        ),
        (
            "motif_positions_15_to_18_active_episodes",
            "Motif-complete family using the old local block positions only where the active episodes are long enough.",
            lambda epi, pos: ((epi == 4) & (pos >= 15) & (pos <= 18))
            | ((epi == 7) & (pos >= 15) & (pos <= 16))
            | ((epi == 11) & (pos >= 15) & (pos <= 18)),
        ),
        (
            "high_mass_complete_episodes_4_7_11",
            "Complete repeated high-mass episodes that dominate the active negative-mass path while staying well below whole-batch scope.",
            lambda epi, pos: np.isin(epi, np.asarray([4, 7, 11], dtype=np.int64)),
        ),
        (
            "pre_tail_complete_episodes_3_to_13",
            "All pre-tail objective episodes before the long low-mass tail episode 14.",
            lambda epi, pos: (epi >= 3) & (epi <= 13),
        ),
        (
            "all_env12_objective_tokens",
            "All objective-valid env12 tokens in the captured outer batch.",
            lambda epi, pos: np.ones_like(epi, dtype=bool),
        ),
    ]

    candidate_reports = []
    for name, description, fn in candidate_fns:
        local_mask = env12_objective_mask & fn(objective_episode_indices, objective_episode_positions)
        candidate_reports.append(
            _candidate_metrics(
                name=name,
                description=description,
                mask=local_mask,
                env_objective_mask=env12_objective_mask,
                objective_mask=objective_mask,
                clip=clip,
                env_indices=env_indices,
                objective_episode_indices=objective_episode_indices,
                total_outer_batches=total_outer_batches,
            )
        )

    by_name = {item["name"]: item for item in candidate_reports}
    episode_complete = dict(by_name["selector_matched_episode_4_complete"])
    motif_family = dict(by_name["motif_positions_15_to_18_active_episodes"])
    high_mass_family = dict(by_name["high_mass_complete_episodes_4_7_11"])
    pretail_family = dict(by_name["pre_tail_complete_episodes_3_to_13"])
    whole_env_family = dict(by_name["all_env12_objective_tokens"])

    return {
        "probe_entry": "phase3_pair2_branch_specific_family_candidate_probe",
        "inputs": {
            "baseline_snapshot_json": str(Path(baseline_snapshot_json).expanduser().resolve()),
            "scope_class_probe_json": str(Path(scope_class_probe_json).expanduser().resolve()),
        },
        "current_trusted_contract": dict(scope_probe["current_trusted_contract"]),
        "candidate_families": candidate_reports,
        "comparisons": {
            "motif_vs_episode_complete": {
                "policy_share_ratio": float(
                    _share(
                        motif_family["policy_share_of_batch_negative_mass"],
                        episode_complete["policy_share_of_batch_negative_mass"],
                    )
                ),
                "token_ratio": float(_share(motif_family["token_count"], episode_complete["token_count"])),
            },
            "high_mass_family_vs_episode_complete": {
                "policy_share_ratio": float(
                    _share(
                        high_mass_family["policy_share_of_batch_negative_mass"],
                        episode_complete["policy_share_of_batch_negative_mass"],
                    )
                ),
                "token_ratio": float(_share(high_mass_family["token_count"], episode_complete["token_count"])),
            },
            "pretail_vs_whole_env12": {
                "policy_share_ratio": float(
                    _share(
                        pretail_family["policy_share_of_batch_negative_mass"],
                        whole_env_family["policy_share_of_batch_negative_mass"],
                    )
                ),
                "token_ratio": float(_share(pretail_family["token_count"], whole_env_family["token_count"])),
            },
        },
        "recommended_family": {
            "candidate_small_control_target": "env12_high_mass_objective_episode_family",
            "runtime_control_shape": "repeated_high_mass_complete_episode_family_branch_specific_gate",
            "locked_outer_batch_idx": int(scope_probe["current_trusted_contract"]["target_outer_batch_idx"]),
            "locked_env_index": 12,
            "locked_objective_episode_indices": [4, 7, 11],
            "token_count": int(high_mass_family["token_count"]),
            "policy_share_of_batch_negative_mass": float(
                high_mass_family["policy_share_of_batch_negative_mass"]
            ),
            "epoch_uniform_share_heuristic": float(high_mass_family["epoch_uniform_share_heuristic"]),
            "reason": (
                "This is the next larger branch-specific family after the selector-matched full episode that still "
                "stays meaningfully below whole-batch scope, while capturing the repeated high-mass episodes that "
                "dominate the current active negative-mass path."
            ),
        },
        "conclusions": {
            "motif_family_is_not_the_next_larger_candidate": bool(
                motif_family["token_count"] < episode_complete["token_count"]
                and motif_family["policy_share_of_batch_negative_mass"]
                < episode_complete["policy_share_of_batch_negative_mass"]
            ),
            "high_mass_family_is_meaningfully_larger_than_episode_complete": bool(
                high_mass_family["policy_share_of_batch_negative_mass"]
                > episode_complete["policy_share_of_batch_negative_mass"] * 2.0
                and high_mass_family["token_count"] > episode_complete["token_count"]
            ),
            "pretail_family_is_too_close_to_whole_batch_for_next_small_control": bool(
                pretail_family["policy_share_of_batch_negative_mass"] > 0.95
                and pretail_family["token_share_of_env12_objective"] > 0.4
            ),
            "recommended_next_family_is_high_mass_complete_episodes_4_7_11": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read-only candidate probe for the next larger pair2 branch-specific family."
    )
    parser.add_argument("--baseline-snapshot-json", type=str, default=DEFAULT_BASELINE_SNAPSHOT_JSON)
    parser.add_argument("--scope-class-probe-json", type=str, default=DEFAULT_SCOPE_CLASS_PROBE_JSON)
    parser.add_argument("--output-json", type=str, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output_json).expanduser().resolve()
    if output_path.exists() and not bool(args.overwrite):
        print(output_path.read_text(encoding="utf-8"))
        return 0

    report = build_pair2_branch_specific_family_candidate_probe(
        baseline_snapshot_json=args.baseline_snapshot_json,
        scope_class_probe_json=args.scope_class_probe_json,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(output_path.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
