import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from ticl.sb3_recurrent_ppo import (
    ENV12_MID_EPISODE_EXTENSION_BLOCK_SCALE,
    _apply_actor_objective_postprocess,
)


def _load_json(path: str, *, expected_audit_entry: str) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise ValueError(f"Missing JSON artifact: {resolved}")
    payload = json.loads(resolved.read_text())
    if str(payload.get("audit_entry", "")) != str(expected_audit_entry):
        raise ValueError(f"Expected {expected_audit_entry} artifact.")
    return payload


def _build_synthetic_runtime_payload(
    *,
    observed_abs: float,
    counterfactual_abs: float,
    block_local_indices: list[int],
    core_local_index: int,
) -> dict[str, Any]:
    max_local_index = int(max([core_local_index, *block_local_indices]))
    actor_advantages = np.zeros((max_local_index + 1, 1), dtype=np.float32)
    objective_masks = np.ones_like(actor_advantages, dtype=np.float32)
    episode_starts = np.zeros_like(actor_advantages, dtype=np.float32)
    rewards = np.zeros_like(actor_advantages, dtype=np.float32)
    episode_starts[0, 0] = 1.0
    episode_starts[int(core_local_index), 0] = 1.0
    actor_advantages[np.asarray(block_local_indices, dtype=np.int64), 0] = -float(observed_abs)
    actor_advantages[int(core_local_index), 0] = -float(counterfactual_abs)
    adjusted = _apply_actor_objective_postprocess(
        actor_advantages,
        rewards=rewards,
        objective_masks=objective_masks,
        episode_starts=episode_starts,
        actor_objective_mode="tokenwise_scale_env12_mid_episode_extension_block",
    )
    block_before = float(abs(float(actor_advantages[np.asarray(block_local_indices, dtype=np.int64), 0].mean())))
    block_after = float(abs(float(adjusted[np.asarray(block_local_indices, dtype=np.int64), 0].mean())))
    core_before = float(actor_advantages[int(core_local_index), 0])
    core_after = float(adjusted[int(core_local_index), 0])
    return {
        "scale_factor": float(ENV12_MID_EPISODE_EXTENSION_BLOCK_SCALE),
        "block_before_abs": float(block_before),
        "block_after_abs": float(block_after),
        "core_before": float(core_before),
        "core_after": float(core_after),
        "adjusted": adjusted,
    }


def build_env12_mid_episode_extension_runtime_mode_regression(
    *,
    env12_runtime_control_json: str,
    env12_counterfactual_pack_json: str,
) -> dict[str, Any]:
    control_payload = _load_json(
        env12_runtime_control_json,
        expected_audit_entry="phase3_env12_mid_episode_extension_runtime_control_probe",
    )
    counter_payload = _load_json(
        env12_counterfactual_pack_json,
        expected_audit_entry="phase3_env12_mid_episode_extension_counterfactual_pack",
    )

    control_block = dict(control_payload["control_block"])
    material_counterfactual = dict(control_payload["material_counterfactual"])
    counterfactual = dict(counter_payload["comparison"]["counterfactual"])

    observed_abs = float(material_counterfactual["mean_start_gae_future_carry_norm"]["observed_abs"])
    counterfactual_abs = float(material_counterfactual["mean_start_gae_future_carry_norm"]["counterfactual_abs"])
    block_local_indices = [int(v) for v in list(control_block["local_indices"])]
    core_local_index = 100

    runtime = _build_synthetic_runtime_payload(
        observed_abs=observed_abs,
        counterfactual_abs=counterfactual_abs,
        block_local_indices=block_local_indices,
        core_local_index=core_local_index,
    )

    scaled_matches_counterfactual = bool(
        abs(float(runtime["block_after_abs"]) - float(counterfactual_abs)) <= 1e-6
    )
    core_unchanged = bool(abs(float(runtime["core_after"]) - float(runtime["core_before"])) <= 1e-12)
    block_shrink = float(float(runtime["block_before_abs"]) - float(runtime["block_after_abs"]))

    return {
        "audit_entry": "phase3_env12_mid_episode_extension_runtime_mode_regression",
        "source_env12_runtime_control_json": str(Path(env12_runtime_control_json).expanduser().resolve()),
        "source_env12_counterfactual_pack_json": str(Path(env12_counterfactual_pack_json).expanduser().resolve()),
        "runtime_mode_name": "tokenwise_scale_env12_mid_episode_extension_block",
        "runtime_mode_scale_factor": float(runtime["scale_factor"]),
        "control_block": control_block,
        "material_counterfactual": material_counterfactual,
        "counterfactual_reference": counterfactual,
        "regression": {
            "block_local_indices": block_local_indices,
            "core_local_index": int(core_local_index),
            "block_before_abs": float(runtime["block_before_abs"]),
            "block_after_abs": float(runtime["block_after_abs"]),
            "core_before": float(runtime["core_before"]),
            "core_after": float(runtime["core_after"]),
            "abs_shrink": float(block_shrink),
            "counterfactual_block_abs": float(counterfactual_abs),
            "counterfactual_matches_scaled_block": bool(scaled_matches_counterfactual),
            "shared_core_unchanged": bool(core_unchanged),
        },
        "conclusions": {
            "branch_specific_gate_supported_in_runtime": bool(scaled_matches_counterfactual and core_unchanged),
            "shared_core_unchanged": bool(core_unchanged),
            "env12_mid_episode_extension_block_reproduces_counterfactual_carry_shrink": bool(
                scaled_matches_counterfactual
            ),
        },
        "recommendation": {
            "candidate_small_control_target": "env12_mid_episode_extension_block",
            "runtime_mode_name": "tokenwise_scale_env12_mid_episode_extension_block",
            "runtime_gate_kind": "branch_specific_actor_objective_mode",
            "shared_guardrail_unchanged": True,
            "branch_specific_followup_needed": True,
            "reason": (
                "The branch-specific env12 mid-episode extension block can be expressed by a minimal runtime "
                "actor-objective gate that scales only objective positions 18..21 in objective episode 0. "
                "It reproduces the same negative-carry shrink without touching the shared core."
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Minimal env12 runtime regression for the branch-specific mid-episode extension control block."
    )
    parser.add_argument(
        "--env12-runtime-control-json",
        type=str,
        default="/home/chen/RLPFN/artifacts/phase3_env12_mid_episode_extension_runtime_control_probe.json",
    )
    parser.add_argument(
        "--env12-counterfactual-pack-json",
        type=str,
        default="/home/chen/RLPFN/artifacts/phase3_env12_mid_episode_extension_counterfactual_pack.json",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="/home/chen/RLPFN/artifacts/phase3_env12_mid_episode_extension_runtime_mode_regression.json",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output_json).expanduser().resolve()
    if output_path.exists() and not bool(args.overwrite):
        print(output_path.read_text())
        return 0

    report = build_env12_mid_episode_extension_runtime_mode_regression(
        env12_runtime_control_json=args.env12_runtime_control_json,
        env12_counterfactual_pack_json=args.env12_counterfactual_pack_json,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(output_path.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
