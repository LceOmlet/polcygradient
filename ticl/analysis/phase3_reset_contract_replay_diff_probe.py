import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ticl.analysis.phase2_guardrail import DEFAULT_PHASE2_SUMMARY, assert_phase2_green
from ticl.analysis.phase3_multi_env_optimization_audit import _default_device
from ticl.analysis.phase3_objective_reset_contract_probe import _summarize_objective_reset_contract
from ticl.analysis.phase3_residual_anchor_gae_path_probe import _collect_suite_rollout_with_debug
from ticl.analysis.phase3_terminal_local_scale_chain_probe import _load_distance_row, _suite_spec_by_name


def _mode_readout(
    *,
    mode_name: str,
    suite_payload: dict[str, Any],
    suite_summary: dict[str, Any],
    collect_contract_debug: dict[str, Any],
    env_index: int,
    distance_row: dict[str, Any],
) -> dict[str, Any]:
    objective_mask = np.asarray(suite_payload["objective_masks"][:, int(env_index)], dtype=np.float32) > 1e-8
    episode_starts = np.asarray(suite_payload["episode_starts"][:, int(env_index)], dtype=np.float32) > 1e-8
    last_done = bool(np.asarray(suite_payload["last_dones"], dtype=bool)[int(env_index)])
    fresh_summary = _summarize_objective_reset_contract(
        objective_mask=objective_mask,
        episode_starts=episode_starts,
        last_done=last_done,
    )
    reset_count = int(fresh_summary["objective_terminal_reset_count_from_segments"])
    terminal_rows = int(fresh_summary["terminal_row_count_within_objective"])
    stored_count = int(distance_row["objective_terminal_reset_count"])
    return {
        "mode_name": str(mode_name),
        "suite_summary": dict(suite_summary),
        "collect_contract_debug": dict(collect_contract_debug),
        "fresh_rollout_summary": fresh_summary,
        "matches_stored_reset_count": bool(reset_count == stored_count),
        "contains_terminal_row_within_objective": bool(terminal_rows > 0),
        "stored_probe_claims_reset_but_mode_has_none": bool(stored_count > 0 and reset_count == 0),
        "stored_probe_claims_reset_but_mode_has_no_terminal_row": bool(stored_count > 0 and terminal_rows == 0),
    }


def _mode_signature(mode_readout: dict[str, Any]) -> dict[str, Any]:
    fresh = dict(mode_readout["fresh_rollout_summary"])
    return {
        "compressed_segments": [list(seg) for seg in fresh["compressed_segments"]],
        "episode_starts_on_objective_positions": [int(v) for v in fresh["episode_starts_on_objective_positions"]],
        "objective_terminal_reset_count_from_segments": int(fresh["objective_terminal_reset_count_from_segments"]),
        "terminal_rows_within_objective_global_steps": [
            int(v) for v in fresh["terminal_rows_within_objective_global_steps"]
        ],
        "terminal_row_count_within_objective": int(fresh["terminal_row_count_within_objective"]),
    }


def _legacy_repeats_disagree(legacy_a: dict[str, Any], legacy_b: dict[str, Any]) -> bool:
    return _mode_signature(legacy_a) != _mode_signature(legacy_b)


def _resolve_likely_single_source(
    *,
    legacy_a: dict[str, Any],
    legacy_b: dict[str, Any],
    official: dict[str, Any],
) -> str:
    official_debug = dict(official["collect_contract_debug"])
    official_flags = dict(official_debug.get("algo_runtime_flags", {}))
    legacy_flags = dict(dict(legacy_a["collect_contract_debug"]).get("algo_runtime_flags", {}))

    official_seeded = bool(
        official_flags.get("strict_fixed_env_mode", False)
        and official_flags.get("env_rng_seed_spec") is not None
        and official_flags.get("rollout_rng_seed_spec") is not None
        and official_flags.get("last_collect_rollout_env_rng_seeds") is not None
        and official_flags.get("last_collect_rollout_rollout_rng_seeds") is not None
    )
    legacy_unseeded = bool(
        not legacy_flags.get("strict_fixed_env_mode", False)
        and legacy_flags.get("env_rng_seed_spec") is None
        and legacy_flags.get("rollout_rng_seed_spec") is None
        and legacy_flags.get("last_collect_rollout_env_rng_seeds") is None
        and legacy_flags.get("last_collect_rollout_rollout_rng_seeds") is None
    )
    legacy_any_match = bool(
        legacy_a.get("matches_stored_reset_count", False) or legacy_b.get("matches_stored_reset_count", False)
    )
    legacy_unstable = bool(_legacy_repeats_disagree(legacy_a, legacy_b))
    official_mismatch = bool(not official.get("matches_stored_reset_count", False))

    if official_seeded and legacy_unseeded and official_mismatch and (legacy_any_match or legacy_unstable):
        return (
            "stored distance row is tied to the legacy unseeded collect contract; "
            "current official collect explicitly wires suite env/rollout seeds and therefore resolves a different reset layout"
        )
    if official_seeded and legacy_unseeded and official_mismatch:
        return (
            "current mismatch is best explained by collect-contract drift: "
            "stored distance artifact came from a legacy unseeded path, while current official collect is strict suite-seeded"
        )
    return "unresolved: current evidence does not reduce the replay diff to a single collect-contract source"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare stored distance-probe reset contract against legacy unseeded and current official strict fresh-collect paths."
        )
    )
    parser.add_argument("checkpoint_path", type=str)
    parser.add_argument("--suite-name", type=str, required=True)
    parser.add_argument("--env-index", type=int, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--n-samples", type=int, default=2048)
    parser.add_argument("--single-eval-pos", type=int, default=1946)
    parser.add_argument(
        "--train-profile",
        type=str,
        default="trusted_sep_reset_mainline",
        choices=["trusted_sep_reset_mainline", "legacy_actor_only_probe"],
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--n-epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--target-kl", type=float, default=None)
    parser.add_argument("--phase2-summary-path", type=str, default=DEFAULT_PHASE2_SUMMARY)
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output_json).expanduser().resolve()
    if output_path.exists() and not bool(args.overwrite):
        print(output_path.read_text())
        return 0

    t0 = time.perf_counter()
    assert_phase2_green(args.phase2_summary_path)
    checkpoint_path = str(Path(args.checkpoint_path).expanduser().resolve())
    suite_spec = _suite_spec_by_name(str(args.suite_name))
    distance_row = _load_distance_row(str(suite_spec["distance_probe_json"]), int(args.env_index))
    device_obj = torch.device(str(args.device or _default_device()))

    def _run_mode(mode_name: str, collect_contract_mode: str) -> dict[str, Any]:
        payload, suite_summary, collect_contract_debug = _collect_suite_rollout_with_debug(
            checkpoint_path=checkpoint_path,
            suite_spec=suite_spec,
            device_obj=device_obj,
            n_samples=int(args.n_samples),
            single_eval_pos=int(args.single_eval_pos),
            train_profile=str(args.train_profile),
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            learning_rate=args.learning_rate,
            target_kl=args.target_kl,
            collect_contract_mode=str(collect_contract_mode),
        )
        return _mode_readout(
            mode_name=str(mode_name),
            suite_payload=payload,
            suite_summary=suite_summary,
            collect_contract_debug=collect_contract_debug,
            env_index=int(args.env_index),
            distance_row=distance_row,
        )

    legacy_a = _run_mode("legacy_distance_probe_runA", "legacy_distance_probe")
    legacy_b = _run_mode("legacy_distance_probe_runB", "legacy_distance_probe")
    official = _run_mode("official_strict", "official_strict")

    result = {
        "audit_entry": "phase3_reset_contract_replay_diff_probe",
        "config": {
            "checkpoint_path": checkpoint_path,
            "suite_name": str(args.suite_name),
            "env_index": int(args.env_index),
            "distance_probe_json": str(Path(suite_spec["distance_probe_json"]).expanduser().resolve()),
            "heldout_suite_path": str(Path(suite_spec["heldout_suite_path"]).expanduser().resolve()),
            "train_suite_path": str(Path(suite_spec["train_suite_path"]).expanduser().resolve()),
            "n_samples": int(args.n_samples),
            "single_eval_pos": int(args.single_eval_pos),
            "train_profile": str(args.train_profile),
            "phase2_summary_path": str(Path(args.phase2_summary_path).expanduser().resolve()),
        },
        "stored_distance_row": {
            "env_index": int(distance_row["env_index"]),
            "env_seed": int(distance_row["env_seed"]),
            "rollout_seed": int(distance_row["rollout_seed"]),
            "objective_terminal_reset_count": int(distance_row["objective_terminal_reset_count"]),
            "pre_vs_zero_suffix_gap": float(distance_row["pre_vs_zero_suffix_gap"]),
        },
        "modes": {
            "legacy_distance_probe_runA": legacy_a,
            "legacy_distance_probe_runB": legacy_b,
            "official_strict": official,
        },
        "conclusions": {
            "legacy_mode_repeats_disagree": bool(_legacy_repeats_disagree(legacy_a, legacy_b)),
            "legacy_mode_matches_stored_reset_count_any_run": bool(
                legacy_a["matches_stored_reset_count"] or legacy_b["matches_stored_reset_count"]
            ),
            "official_strict_matches_stored_reset_count": bool(official["matches_stored_reset_count"]),
            "legacy_mode_omits_explicit_collect_seed_wiring": bool(
                dict(legacy_a["collect_contract_debug"]).get("algo_runtime_flags", {}).get("env_rng_seed_spec") is None
                and dict(legacy_a["collect_contract_debug"]).get("algo_runtime_flags", {}).get("rollout_rng_seed_spec")
                is None
            ),
            "official_mode_wires_suite_env_and_rollout_seeds": bool(
                dict(official["collect_contract_debug"]).get("algo_runtime_flags", {}).get("env_rng_seed_spec")
                == [int(v) for v in official["collect_contract_debug"]["suite_env_seeds"]]
                and dict(official["collect_contract_debug"]).get("algo_runtime_flags", {}).get("rollout_rng_seed_spec")
                == [int(v) for v in official["collect_contract_debug"]["suite_rollout_seeds"]]
            ),
            "likely_single_source": _resolve_likely_single_source(
                legacy_a=legacy_a,
                legacy_b=legacy_b,
                official=official,
            ),
        },
        "runtime_wall_s_total": float(time.perf_counter() - t0),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(output_path.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
