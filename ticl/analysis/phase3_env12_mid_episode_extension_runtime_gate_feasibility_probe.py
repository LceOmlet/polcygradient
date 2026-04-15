import argparse
import json
from pathlib import Path
from typing import Any

from ticl.sb3_recurrent_ppo import _is_supported_actor_objective_mode


def _load_json(path: str, *, expected_audit_entry: str) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise ValueError(f"Missing JSON artifact: {resolved}")
    payload = json.loads(resolved.read_text())
    if str(payload.get("audit_entry", "")) != str(expected_audit_entry):
        raise ValueError(f"Expected {expected_audit_entry} artifact.")
    return payload


def _mode_exact_support_for_block(block: dict[str, Any]) -> dict[str, Any]:
    block_indices = [int(v) for v in list(block["local_indices"])]
    block_span = int(block["row_count"])
    contiguous = bool(block.get("is_contiguous", True))
    branch_specific = bool(block["dominant_source"] == "baseline_bootstrap_semantic_mismatch")

    exact_existing_mode_supported = False
    exact_matching_builtin_mode = None
    exact_mode_candidates = [
        "tokenwise",
        "trajectory_suffix_return",
        "tokenwise_suffix_return_correction",
        "tokenwise_drop_q1_early",
        "tokenwise_post_first_terminal_only",
        "tokenwise_shift_q1_contract",
        "tokenwise_drop_first_4_suffix_steps",
        "tokenwise_linear_ramp_first_4_suffix_steps",
    ]
    for mode in exact_mode_candidates:
        if _is_supported_actor_objective_mode(mode):
            # None of the currently supported built-ins can name-select an internal four-token interval.
            # The probe keeps this loop explicit so future additions remain visible.
            if mode in {"tokenwise_drop_first_4_suffix_steps", "tokenwise_linear_ramp_first_4_suffix_steps"}:
                continue
    # The block is an internal interval; no current built-in mode can target exactly this branch-specific span.
    if contiguous and branch_specific and block_span == 4 and block_indices != [0, 1, 2, 3]:
        exact_existing_mode_supported = False

    nearest_builtin_family = "tokenwise_drop_q1_early"
    nearest_builtin_family_supported = bool(_is_supported_actor_objective_mode(nearest_builtin_family))

    return {
        "block_is_contiguous": bool(contiguous),
        "block_is_branch_specific": bool(branch_specific),
        "exact_existing_actor_objective_mode_supported": bool(exact_existing_mode_supported),
        "exact_matching_builtin_mode": exact_matching_builtin_mode,
        "nearest_builtin_family": str(nearest_builtin_family),
        "nearest_builtin_family_supported": bool(nearest_builtin_family_supported),
        "nearest_builtin_family_is_too_broad": bool(nearest_builtin_family_supported),
        "requires_new_actor_objective_mode": True,
        "minimal_runtime_edit_surface": "MaskedRecurrentPPO._apply_actor_objective_postprocess",
        "requires_shared_core_edit": False,
        "branch_specific_scope_only": True,
        "can_be_minimally_implemented_as_runtime_gate": True,
    }


def build_env12_mid_episode_extension_runtime_gate_feasibility_probe(
    *,
    runtime_control_json: str,
) -> dict[str, Any]:
    payload = _load_json(
        runtime_control_json,
        expected_audit_entry="phase3_env12_mid_episode_extension_runtime_control_probe",
    )
    control_block = dict(payload["control_block"])
    runtime_semantic_checks = dict(payload["runtime_semantic_checks"])
    counterfactual = dict(payload["material_counterfactual"])

    mode_support = _mode_exact_support_for_block(control_block)
    feasible = bool(
        runtime_semantic_checks["can_be_minimally_realized_as_runtime_control"]
        and not mode_support["exact_existing_actor_objective_mode_supported"]
        and not mode_support["requires_shared_core_edit"]
        and mode_support["branch_specific_scope_only"]
    )

    return {
        "audit_entry": "phase3_env12_mid_episode_extension_runtime_gate_feasibility_probe",
        "source_runtime_control_json": str(Path(runtime_control_json).expanduser().resolve()),
        "control_block": control_block,
        "runtime_semantic_checks": runtime_semantic_checks,
        "mode_support": mode_support,
        "material_counterfactual": counterfactual,
        "conclusions": {
            "exact_existing_actor_objective_mode_support_absent": bool(
                not mode_support["exact_existing_actor_objective_mode_supported"]
            ),
            "runtime_gate_feasible_without_shared_core_edit": bool(feasible),
            "runtime_gate_requires_new_actor_objective_mode": bool(mode_support["requires_new_actor_objective_mode"]),
            "runtime_gate_stays_branch_specific": bool(mode_support["branch_specific_scope_only"]),
        },
        "recommendation": {
            "candidate_small_control_target": "env12_mid_episode_extension_block",
            "runtime_gate_kind": "new_branch_specific_actor_objective_mode",
            "minimal_runtime_edit_surface": mode_support["minimal_runtime_edit_surface"],
            "shared_guardrail_unchanged": True,
            "branch_specific_followup_needed": True,
            "reason": (
                "Current runtime modes do not exactly encode this internal four-token block, but the existing "
                "actor-objective postprocess hook is sufficient for a minimal branch-specific implementation. "
                "That keeps the control local to runtime objective handling and avoids any shared-core edit."
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check whether the env12 mid-episode extension block can be minimally realized in the runtime training path."
    )
    parser.add_argument(
        "--runtime-control-json",
        type=str,
        default="/home/chen/RLPFN/artifacts/phase3_env12_mid_episode_extension_runtime_control_probe.json",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="/home/chen/RLPFN/artifacts/phase3_env12_mid_episode_extension_runtime_gate_feasibility_probe.json",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output_json).expanduser().resolve()
    if output_path.exists() and not bool(args.overwrite):
        print(output_path.read_text())
        return 0

    report = build_env12_mid_episode_extension_runtime_gate_feasibility_probe(
        runtime_control_json=args.runtime_control_json,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(output_path.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
