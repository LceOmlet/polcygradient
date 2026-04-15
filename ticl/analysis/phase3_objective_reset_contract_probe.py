import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ticl.analysis.phase2_guardrail import DEFAULT_PHASE2_SUMMARY, assert_phase2_green
from ticl.analysis.phase3_heldout_reset_semantics_probe import _objective_episode_segments
from ticl.analysis.phase3_multi_env_optimization_audit import _default_device
from ticl.analysis.phase3_residual_anchor_gae_path_probe import _collect_suite_rollout
from ticl.analysis.phase3_terminal_local_scale_chain_probe import _load_distance_row, _suite_spec_by_name


def _summarize_objective_reset_contract(
    *,
    objective_mask: np.ndarray,
    episode_starts: np.ndarray,
    last_done: bool,
) -> dict[str, Any]:
    objective_mask = np.asarray(objective_mask, dtype=bool).reshape(-1)
    episode_starts = np.asarray(episode_starts, dtype=bool).reshape(-1)
    objective_steps = np.flatnonzero(objective_mask)
    segments = _objective_episode_segments(objective_mask, episode_starts)
    terminal_rows: list[int] = []
    if int(objective_steps.size) > 0:
        last_index = int(objective_mask.shape[0] - 1)
        for global_step in objective_steps.tolist():
            if int(global_step) == int(last_index):
                next_non_terminal = 0.0 if bool(last_done) else 1.0
            else:
                next_non_terminal = 0.0 if bool(episode_starts[int(global_step) + 1]) else 1.0
            if float(next_non_terminal) <= 0.0:
                terminal_rows.append(int(global_step))
    return {
        "objective_token_count": int(objective_steps.size),
        "objective_steps_head": [int(v) for v in objective_steps[:10].tolist()],
        "objective_steps_tail": [int(v) for v in objective_steps[-10:].tolist()],
        "compressed_segment_starts": [int(s) for s, _ in segments],
        "compressed_segments": [[int(s), int(e)] for s, e in segments],
        "objective_terminal_reset_count_from_segments": max(0, int(len(segments) - 1)),
        "episode_starts_on_objective_positions": [
            int(v) for v in np.flatnonzero(np.asarray(episode_starts[objective_mask], dtype=bool)).tolist()
        ],
        "terminal_rows_within_objective_global_steps": [int(v) for v in terminal_rows],
        "terminal_row_count_within_objective": int(len(terminal_rows)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare stored distance-probe reset-count against a fresh fixed-suite rollout for one env."
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
    suite_payload, suite_summary = _collect_suite_rollout(
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
    )

    env_index = int(args.env_index)
    objective_mask = np.asarray(suite_payload["objective_masks"][:, env_index], dtype=np.float32) > 1e-8
    episode_starts = np.asarray(suite_payload["episode_starts"][:, env_index], dtype=np.float32) > 1e-8
    last_done = bool(np.asarray(suite_payload["last_dones"], dtype=bool)[env_index])
    fresh_summary = _summarize_objective_reset_contract(
        objective_mask=objective_mask,
        episode_starts=episode_starts,
        last_done=last_done,
    )

    result = {
        "audit_entry": "phase3_objective_reset_contract_probe",
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
        "suite_summary": dict(suite_summary),
        "stored_distance_row": {
            "env_index": int(distance_row["env_index"]),
            "env_seed": int(distance_row["env_seed"]),
            "rollout_seed": int(distance_row["rollout_seed"]),
            "objective_terminal_reset_count": int(distance_row["objective_terminal_reset_count"]),
            "pre_vs_zero_suffix_gap": float(distance_row["pre_vs_zero_suffix_gap"]),
        },
        "fresh_rollout_summary": fresh_summary,
        "conclusions": {
            "fresh_reset_count_matches_stored_distance_probe": bool(
                int(fresh_summary["objective_terminal_reset_count_from_segments"])
                == int(distance_row["objective_terminal_reset_count"])
            ),
            "fresh_rollout_contains_in_objective_terminal_row": bool(
                int(fresh_summary["terminal_row_count_within_objective"]) > 0
            ),
            "stored_distance_probe_claims_reset_but_fresh_rollout_has_none": bool(
                int(distance_row["objective_terminal_reset_count"]) > 0
                and int(fresh_summary["objective_terminal_reset_count_from_segments"]) == 0
            ),
            "stored_distance_probe_claims_reset_but_fresh_rollout_has_no_terminal_row": bool(
                int(distance_row["objective_terminal_reset_count"]) > 0
                and int(fresh_summary["terminal_row_count_within_objective"]) == 0
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
