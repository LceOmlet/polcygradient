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
from ticl.analysis.phase3_residual_anchor_gae_path_probe import (
    _analyze_anchor_env,
    _collect_suite_rollout_with_debug,
)
from ticl.analysis.phase3_terminal_local_scale_chain_probe import (
    _build_probe,
    _load_distance_row,
    _suite_spec_by_name,
    _terminal_local_indices,
)


def _distance_row_map(distance_probe_json: str, *, env_count: int) -> dict[int, dict[str, Any]]:
    row_map: dict[int, dict[str, Any]] = {}
    for env_index in range(int(env_count)):
        row_map[int(env_index)] = _load_distance_row(distance_probe_json, int(env_index))
    return row_map


def _scale_chain_entry_status(
    *,
    suite_name: str,
    env_index: int,
    distance_row: dict[str, Any],
    suite_payload: dict[str, Any],
    suite_summary: dict[str, Any],
    token_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    terminal_local_indices = [int(v) for v in _terminal_local_indices(token_rows)]
    if int(distance_row["objective_terminal_reset_count"]) <= 0:
        return {
            "can_enter_terminal_local_scale_chain": False,
            "entry_failure_reason": "distance_probe_has_no_objective_terminal_reset",
            "terminal_local_indices": terminal_local_indices,
            "selected_target_local_available": False,
            "selected_target_local_index": None,
            "single_terminal_local": bool(len(terminal_local_indices) == 1),
        }
    try:
        probe_payload = _build_probe(
            suite_name=str(suite_name),
            env_index=int(env_index),
            distance_row=dict(distance_row),
            suite_payload=suite_payload,
            suite_summary=suite_summary,
            token_rows=token_rows,
            explicit_target_local_index=None,
        )
    except Exception as exc:
        return {
            "can_enter_terminal_local_scale_chain": False,
            "entry_failure_reason": str(exc),
            "terminal_local_indices": terminal_local_indices,
            "selected_target_local_available": False,
            "selected_target_local_index": None,
            "single_terminal_local": bool(len(terminal_local_indices) == 1),
        }

    selected_target_local_index = probe_payload["config"].get("selected_target_local_index", None)
    return {
        "can_enter_terminal_local_scale_chain": True,
        "entry_failure_reason": None,
        "terminal_local_indices": terminal_local_indices,
        "selected_target_local_available": bool(probe_payload["conclusions"]["selected_target_local_available"]),
        "selected_target_local_index": (
            None if selected_target_local_index is None else int(selected_target_local_index)
        ),
        "single_terminal_local": bool(probe_payload["conclusions"]["single_terminal_local"]),
    }


def _suite_census(
    *,
    checkpoint_path: str,
    suite_name: str,
    device_obj: torch.device,
    n_samples: int,
    single_eval_pos: int,
    train_profile: str,
    batch_size: int | None,
    n_epochs: int | None,
    learning_rate: float | None,
    target_kl: float | None,
) -> dict[str, Any]:
    suite_spec = _suite_spec_by_name(str(suite_name))
    suite_payload, suite_summary, collect_contract_debug = _collect_suite_rollout_with_debug(
        checkpoint_path=checkpoint_path,
        suite_spec=suite_spec,
        device_obj=device_obj,
        n_samples=int(n_samples),
        single_eval_pos=int(single_eval_pos),
        train_profile=str(train_profile),
        batch_size=batch_size,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        target_kl=target_kl,
        collect_contract_mode="official_strict",
    )
    objective_masks = np.asarray(suite_payload["objective_masks"], dtype=np.float32)
    env_count = int(objective_masks.shape[1])
    distance_rows = _distance_row_map(str(suite_spec["distance_probe_json"]), env_count=env_count)

    env_rows: list[dict[str, Any]] = []
    for env_index in range(int(env_count)):
        distance_row = dict(distance_rows[int(env_index)])
        objective_mask = np.asarray(suite_payload["objective_masks"][:, env_index], dtype=np.float32) > 1e-8
        episode_starts = np.asarray(suite_payload["episode_starts"][:, env_index], dtype=np.float32) > 1e-8
        last_done = bool(np.asarray(suite_payload["last_dones"], dtype=bool)[env_index])
        reset_summary = _summarize_objective_reset_contract(
            objective_mask=objective_mask,
            episode_starts=episode_starts,
            last_done=last_done,
        )
        anchor_row = {
            "suite_name": str(suite_name),
            "env_index": int(env_index),
            "env_seed": int(distance_row["env_seed"]),
            "rollout_seed": int(distance_row["rollout_seed"]),
            "pre_vs_zero_suffix_gap": float(distance_row["pre_vs_zero_suffix_gap"]),
        }
        _env_summary, token_rows = _analyze_anchor_env(anchor_row=anchor_row, suite_payload=suite_payload)
        scale_chain_entry = _scale_chain_entry_status(
            suite_name=str(suite_name),
            env_index=int(env_index),
            distance_row=distance_row,
            suite_payload=suite_payload,
            suite_summary=suite_summary,
            token_rows=token_rows,
        )
        env_rows.append(
            {
                "env_index": int(env_index),
                "env_seed": int(distance_row["env_seed"]),
                "rollout_seed": int(distance_row["rollout_seed"]),
                "distance_probe_objective_terminal_reset_count": int(distance_row["objective_terminal_reset_count"]),
                "official_strict_objective_terminal_reset_count": int(
                    reset_summary["objective_terminal_reset_count_from_segments"]
                ),
                "terminal_row_count_within_objective": int(reset_summary["terminal_row_count_within_objective"]),
                "compressed_segments": [[int(s), int(e)] for s, e in reset_summary["compressed_segments"]],
                "terminal_rows_within_objective_global_steps": [
                    int(v) for v in reset_summary["terminal_rows_within_objective_global_steps"]
                ],
                **scale_chain_entry,
            }
        )

    reset_positive_env_indices = [
        int(row["env_index"]) for row in env_rows if int(row["official_strict_objective_terminal_reset_count"]) > 0
    ]
    terminal_row_positive_env_indices = [
        int(row["env_index"]) for row in env_rows if int(row["terminal_row_count_within_objective"]) > 0
    ]
    scale_chain_entry_env_indices = [
        int(row["env_index"]) for row in env_rows if bool(row["can_enter_terminal_local_scale_chain"])
    ]
    return {
        "suite_name": str(suite_name),
        "suite_spec": {
            "distance_probe_json": str(Path(suite_spec["distance_probe_json"]).expanduser().resolve()),
            "train_suite_path": str(Path(suite_spec["train_suite_path"]).expanduser().resolve()),
            "heldout_suite_path": str(Path(suite_spec["heldout_suite_path"]).expanduser().resolve()),
        },
        "suite_summary": dict(suite_summary),
        "collect_contract_debug": dict(collect_contract_debug),
        "env_rows": env_rows,
        "summary": {
            "env_count": int(len(env_rows)),
            "official_reset_positive_env_count": int(len(reset_positive_env_indices)),
            "official_terminal_row_positive_env_count": int(len(terminal_row_positive_env_indices)),
            "official_scale_chain_entry_env_count": int(len(scale_chain_entry_env_indices)),
            "official_reset_positive_env_indices": reset_positive_env_indices,
            "official_terminal_row_positive_env_indices": terminal_row_positive_env_indices,
            "official_scale_chain_entry_env_indices": scale_chain_entry_env_indices,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Census current official strict objective-reset / terminal-row / scale-chain-entry structure across heldout envs."
        )
    )
    parser.add_argument("checkpoint_path", type=str)
    parser.add_argument(
        "--suite-name",
        action="append",
        dest="suite_names",
        default=None,
        help="Repeat for multiple suites. Defaults to pair1 and pair2.",
    )
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
    device_obj = torch.device(str(args.device or _default_device()))
    suite_names = list(args.suite_names or ["pair1", "pair2"])

    suite_results = [
        _suite_census(
            checkpoint_path=checkpoint_path,
            suite_name=str(suite_name),
            device_obj=device_obj,
            n_samples=int(args.n_samples),
            single_eval_pos=int(args.single_eval_pos),
            train_profile=str(args.train_profile),
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            learning_rate=args.learning_rate,
            target_kl=args.target_kl,
        )
        for suite_name in suite_names
    ]

    all_entry_points = [
        {
            "suite_name": str(suite["suite_name"]),
            "env_index": int(row["env_index"]),
            "distance_probe_objective_terminal_reset_count": int(row["distance_probe_objective_terminal_reset_count"]),
            "official_strict_objective_terminal_reset_count": int(row["official_strict_objective_terminal_reset_count"]),
            "terminal_row_count_within_objective": int(row["terminal_row_count_within_objective"]),
        }
        for suite in suite_results
        for row in suite["env_rows"]
        if bool(row["can_enter_terminal_local_scale_chain"])
    ]
    result = {
        "audit_entry": "phase3_official_strict_reset_census",
        "config": {
            "checkpoint_path": checkpoint_path,
            "suite_names": [str(v) for v in suite_names],
            "n_samples": int(args.n_samples),
            "single_eval_pos": int(args.single_eval_pos),
            "train_profile": str(args.train_profile),
            "phase2_summary_path": str(Path(args.phase2_summary_path).expanduser().resolve()),
        },
        "suite_results": suite_results,
        "conclusions": {
            "any_official_strict_scale_chain_entry": bool(len(all_entry_points) > 0),
            "official_strict_scale_chain_entry_points": all_entry_points,
            "all_pair1_pair2_envs_fail_official_strict_scale_chain_entry": bool(len(all_entry_points) == 0),
        },
        "runtime_wall_s_total": float(time.perf_counter() - t0),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(output_path.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
