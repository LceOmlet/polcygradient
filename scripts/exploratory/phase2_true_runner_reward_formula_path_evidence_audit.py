#!/usr/bin/env python3
"""Audit true-runner formula-path evidence for progress/potential and terminal-event reward.

This is intentionally read-only.  It does not authorize a source-label sampler unless
the formula path is present, reaches the true runner, is component-loggable, and has
base reward preservation evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _workspace_root() -> Path:
    return _repo_root().parent


def _default_artifact_root() -> Path:
    return _workspace_root() / "artifacts" / "phase2_true_runner_reward_formula_path_evidence_audit_0518"


def _default_potential_progress_component_audit() -> Path:
    return (
        _workspace_root()
        / "artifacts"
        / "phase2_potential_progress_component_reconstruction_audit_cuda_0518"
        / "potential_progress_component_reconstruction_audit.json"
    )


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"_missing": str(path)}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _line_no(text: str, needle: str) -> int | None:
    for idx, line in enumerate(text.splitlines(), start=1):
        if needle in line:
            return idx
    return None


def _contains_all(text: str, needles: list[str]) -> bool:
    return all(needle in text for needle in needles)


def _find_function_block(text: str, name: str, *, max_lines: int = 180) -> str:
    lines = text.splitlines()
    start = None
    needle = f"def {name}"
    for idx, line in enumerate(lines):
        if needle in line:
            start = idx
            break
    if start is None:
        return ""
    return "\n".join(lines[start : start + max_lines])


def _coverage_fraction(audit: dict[str, Any], key: str) -> float | None:
    for row in audit.get("coverage", []):
        if row.get("key") == key:
            try:
                return float(row.get("fraction"))
            except (TypeError, ValueError):
                return None
    return None


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    potential = report["paths"]["potential_delta_or_progress"]
    terminal = report["paths"]["terminal_event_reward_or_penalty"]
    lines = [
        "# True Runner Reward Formula Path Evidence Audit",
        "",
        "This audit is read-only and does not modify generator code.",
        "",
        "## Decision",
        "",
        f"- potential_delta/progress sampler authorized: `{potential['source_label_sampler_authorized']}`",
        f"- terminal_event sampler authorized: `{terminal['source_label_sampler_authorized']}`",
        f"- any sampler authorized: `{report['read']['any_source_label_sampler_authorized']}`",
        "",
        "## Potential Delta / Progress",
        "",
        f"- formula node present: `{potential['formula_node_present']}`",
        f"- true-runner reward-unit path present: `{potential['true_runner_reward_unit_path_present']}`",
        f"- component logging ready: `{potential['component_logging_ready']}`",
        f"- component reconstruction audit pass: `{potential['component_reconstruction_audit_pass']}`",
        f"- current explicit coverage fraction: `{potential['current_explicit_coverage_fraction']}`",
        f"- base reward preservation evidence: `{potential['base_reward_preservation_current_pass']}`",
        f"- decision reason: {potential['decision_reason']}",
        "",
        "## Terminal Event",
        "",
        f"- formula node present: `{terminal['formula_node_present']}`",
        f"- true-runner terminal path present: `{terminal['true_runner_terminal_path_present']}`",
        f"- terminal bonus single-add check: `{terminal['terminal_bonus_single_add_pass']}`",
        f"- true-runner component logging ready: `{terminal['true_runner_component_logging_ready']}`",
        f"- pack fallback component logging ready: `{terminal['pack_fallback_component_logging_ready']}`",
        f"- current explicit coverage fraction: `{terminal['current_explicit_coverage_fraction']}`",
        f"- base reward preservation under active terminal event: `{terminal['base_reward_preservation_under_active_terminal_event_pass']}`",
        f"- decision reason: {terminal['decision_reason']}",
        "",
        "## Required Next Step",
        "",
        "- `terminal_event` may enter a minimal source-label sampler only if this report authorizes it.",
        "- `potential_delta/progress` may enter a minimal source-label sampler only if this report authorizes it.",
        "- Any sampler candidate must still pass category span, independence, pack-fit parity, PPO health, and optimizability.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    repo = _repo_root()
    workspace = _workspace_root()
    env_prior_path = repo / "ticl" / "priors" / "environment_prior.py"
    sb3_path = repo / "ticl" / "sb3_recurrent_ppo.py"
    runner_path = repo / "ticl" / "analysis" / "phase2_gym_prior_pack_training_runner.py"

    env_prior = _read_text(env_prior_path)
    sb3 = _read_text(sb3_path)
    runner = _read_text(runner_path)

    env_preservation_path = Path(args.env_preservation_audit)
    base_topology_path = Path(args.base_env_topology_audit)
    terminal_micro_path = Path(args.terminal_microfamily_audit)
    potential_component_path = Path(args.potential_progress_component_audit)
    env_preservation = _read_json(env_preservation_path)
    base_topology = _read_json(base_topology_path)
    terminal_micro = _read_json(terminal_micro_path)
    potential_component = _read_json(potential_component_path)

    terminal_block = _find_function_block(env_prior, "_apply_terminal_reset_step")
    terminal_bonus_add_count = terminal_block.count("reward_next = reward_next + terminal_bonus_applied")
    terminal_single_add_pass = terminal_bonus_add_count == 1

    potential_formula_node_present = _contains_all(
        env_prior,
        [
            "reward_potential_delta_t",
            "prev_state_cost - state_cost",
            "reward_candidate",
            "obs_candidate[:, 0]",
        ],
    )
    true_runner_reward_unit_path_present = _contains_all(
        sb3,
        [
            "_unpack_transition_output",
            "reward_next_raw",
            "_compose_exact_scm_reward",
        ],
    ) and _contains_all(
        env_prior,
        [
            "reward_env_next = self._transform_exact_scm_base_reward",
            '"reward_env": reward_env_next.detach()',
        ],
    )
    potential_component_payload_key_present = _contains_all(
        env_prior + sb3 + runner,
        [
            '"reward_progress_component"',
            '"reward_potential_delta_component"',
            "_last_exact_scm_lowtail_reward_components",
        ],
    )
    potential_component_logging_ready = bool(potential_component_payload_key_present)

    terminal_formula_node_present = _contains_all(
        env_prior,
        [
            "_terminal_bonus_base_from_signal",
            "_terminal_bonus_scale_from_draw",
            "terminal_bonus_applied",
            "terminal_event_mask",
        ],
    )
    true_runner_terminal_path_present = _contains_all(
        sb3,
        [
            "terminal_bonus_base_next",
            "_apply_terminal_reset_step",
            "terminal_bonus_base=terminal_bonus_base_next",
        ],
    ) and _contains_all(
        env_prior,
        [
            "return_aux=True",
            '"reward_terminal_bonus": reward_terminal_bonus_next.detach()',
            'terminal_aux["terminal_bonus_applied"]',
        ],
    )
    true_runner_component_logging_ready = _contains_all(
        sb3 + env_prior,
        [
            '"reward_terminal_bonus"',
            "rollout_log_accumulator.observe_step",
            "reward_terminal_bonus=step_payload",
        ],
    ) or _contains_all(
        sb3 + env_prior,
        [
            '"reward_terminal_bonus": reward_terminal_bonus_next.detach()',
            "reward_terminal_bonus=step_payload",
        ],
    )
    pack_component_keys_ready = _contains_all(
        runner,
        [
            '"reward_env_components"',
            '"reward_ctrl_components"',
            '"reward_survival_components"',
            '"reward_terminal_bonus_components"',
            "trusted_pack_reward_components",
        ],
    )
    pack_fallback_component_logging_ready = pack_component_keys_ready or not (
        "terminal_bonus = np.zeros_like(raw_reward" in runner
        and "reward_terminal_bonus=terminal_bonus[step_idx]" in runner
    )

    env_preservation_read = env_preservation.get("read", {})
    base_reward_preservation_current_pass = bool(
        env_preservation_read.get("base_env_reward_present_all", False)
        and env_preservation_read.get("no_auxiliary_suppression_pass", False)
    )
    potential_current_fraction = _coverage_fraction(base_topology, "explicit_potential_delta")
    terminal_current_fraction = _coverage_fraction(base_topology, "explicit_terminal_bonus")

    potential_component_read = potential_component.get("read", {}) if isinstance(potential_component, dict) else {}
    potential_component_reconstruction_pass = bool(
        potential_component_read.get("hard_pass", False)
        and potential_component_read.get("component_logging_pass", False)
        and potential_component_read.get("reconstruction_pass", False)
        and potential_component_read.get("progress_observed", False)
        and potential_component_read.get("potential_delta_observed", False)
    )
    potential_base_preservation_under_active_path = bool(
        potential_component_read.get("base_preservation_pass", False)
    )
    terminal_micro_read = terminal_micro.get("read", {}) if isinstance(terminal_micro, dict) else {}
    terminal_base_preservation_under_active_path = bool(terminal_micro_read.get("hard_pass", False))

    potential_authorized = bool(
        potential_formula_node_present
        and true_runner_reward_unit_path_present
        and potential_component_logging_ready
        and potential_component_reconstruction_pass
        and potential_base_preservation_under_active_path
    )
    terminal_authorized = bool(
        terminal_formula_node_present
        and true_runner_terminal_path_present
        and terminal_single_add_pass
        and true_runner_component_logging_ready
        and pack_fallback_component_logging_ready
        and terminal_base_preservation_under_active_path
    )

    if potential_authorized:
        potential_reason = (
            "formula path, true-runner reward-unit path, component logging, reconstruction, "
            "and active-path base-reward preservation all pass"
        )
    else:
        potential_reason = (
            "formula path is not authorized until formula path, true-runner payload logging, "
            "component reconstruction, and active-path base-reward preservation all pass"
        )
    if terminal_base_preservation_under_active_path and pack_fallback_component_logging_ready:
        terminal_reason = (
            "formula, true-runner step payload, pack component logging, and active "
            "terminal-event base-reward preservation all pass"
        )
    else:
        terminal_reason = (
            "formula, true-runner step payload, and pack component logging paths exist, "
            "but active terminal-event base-reward preservation has not been proven"
            if pack_fallback_component_logging_ready
            else (
            "formula and true-runner step payload path exist, but pack fallback currently zero-fills "
            "terminal_bonus and active terminal-event base-reward preservation has not been proven"
            )
        )
    if not terminal_single_add_pass:
        terminal_reason = (
            "terminal bonus add count is not exactly one inside _apply_terminal_reset_step; "
            + terminal_reason
        )

    report = {
        "schema": "phase2_true_runner_reward_formula_path_evidence_audit.v1",
        "analysis_entry": "phase2_true_runner_reward_formula_path_evidence_audit",
        "contract": {
            "read_only": True,
            "does_not_modify_generator": True,
            "does_not_train": True,
            "sampler_requires_formula_path_logging_and_base_reward_preservation": True,
        },
        "inputs": {
            "environment_prior": str(env_prior_path),
            "sb3_recurrent_ppo": str(sb3_path),
            "phase2_runner": str(runner_path),
            "env_preservation_audit": str(env_preservation_path),
            "base_env_topology_audit": str(base_topology_path),
            "terminal_microfamily_audit": str(terminal_micro_path),
            "potential_progress_component_audit": str(potential_component_path),
        },
        "paths": {
            "potential_delta_or_progress": {
                "formula_node_present": potential_formula_node_present,
                "formula_evidence_lines": {
                    "reward_potential_delta_t": _line_no(env_prior, "reward_potential_delta_t"),
                    "prev_state_cost_minus_state_cost": _line_no(env_prior, "prev_state_cost - state_cost"),
                    "progress_linear_obs0": _line_no(env_prior, "obs_candidate[:, 0]"),
                },
                "true_runner_reward_unit_path_present": true_runner_reward_unit_path_present,
                "runner_evidence_lines": {
                    "sb3_unpack_transition_output": _line_no(sb3, "_unpack_transition_output"),
                    "sb3_reward_next_raw": _line_no(sb3, "reward_next_raw"),
                    "env_prior_reward_env_next": _line_no(env_prior, "reward_env_next = self._transform_exact_scm_base_reward"),
                },
                "component_logging_ready": potential_component_logging_ready,
                "component_payload_key_present": potential_component_payload_key_present,
                "component_reconstruction_audit_pass": potential_component_reconstruction_pass,
                "current_explicit_coverage_fraction": potential_current_fraction,
                "base_reward_preservation_current_pass": base_reward_preservation_current_pass,
                "base_reward_preservation_under_active_path_pass": potential_base_preservation_under_active_path,
                "source_label_sampler_authorized": potential_authorized,
                "decision_reason": potential_reason,
            },
            "terminal_event_reward_or_penalty": {
                "formula_node_present": terminal_formula_node_present,
                "formula_evidence_lines": {
                    "terminal_bonus_base_from_signal": _line_no(env_prior, "_terminal_bonus_base_from_signal"),
                    "terminal_bonus_scale_from_draw": _line_no(env_prior, "_terminal_bonus_scale_from_draw"),
                    "terminal_bonus_applied": _line_no(env_prior, "terminal_bonus_applied"),
                    "terminal_event_mask": _line_no(env_prior, "terminal_event_mask"),
                },
                "true_runner_terminal_path_present": true_runner_terminal_path_present,
                "runner_evidence_lines": {
                    "sb3_terminal_bonus_base_next": _line_no(sb3, "terminal_bonus_base_next"),
                    "env_prior_return_aux_true": _line_no(env_prior, "return_aux=True"),
                    "env_prior_reward_terminal_bonus_payload": _line_no(env_prior, '"reward_terminal_bonus": reward_terminal_bonus_next.detach()'),
                    "sb3_rollout_accumulator_terminal_bonus": _line_no(sb3, 'reward_terminal_bonus=step_payload["reward_terminal_bonus"]'),
                    "phase2_pack_fallback_zero_fill": _line_no(runner, "terminal_bonus = np.zeros_like(raw_reward"),
                },
                "terminal_bonus_add_count_in_apply_terminal_reset_step": terminal_bonus_add_count,
                "terminal_bonus_single_add_pass": terminal_single_add_pass,
                "true_runner_component_logging_ready": true_runner_component_logging_ready,
                "pack_component_keys_ready": pack_component_keys_ready,
                "pack_fallback_component_logging_ready": pack_fallback_component_logging_ready,
                "current_explicit_coverage_fraction": terminal_current_fraction,
                "base_reward_preservation_current_pass": base_reward_preservation_current_pass,
                "base_reward_preservation_under_active_terminal_event_pass": terminal_base_preservation_under_active_path,
                "source_label_sampler_authorized": terminal_authorized,
                "decision_reason": terminal_reason,
            },
        },
        "read": {
            "base_env_reward_preservation_current_pass": base_reward_preservation_current_pass,
            "potential_delta_or_progress_path_usable_but_not_loggable": bool(
                potential_formula_node_present and true_runner_reward_unit_path_present and not potential_component_logging_ready
            ),
            "terminal_event_true_runner_path_usable": bool(
                terminal_formula_node_present and true_runner_terminal_path_present and terminal_single_add_pass
            ),
            "terminal_event_pack_fallback_logging_gap": not pack_fallback_component_logging_ready,
            "any_source_label_sampler_authorized": bool(potential_authorized or terminal_authorized),
            "source_label_sampler_authorized": [
                name
                for name, row in {
                    "potential_delta_or_progress": {"ok": potential_authorized},
                    "terminal_event_reward_or_penalty": {"ok": terminal_authorized},
                }.items()
                if row["ok"]
            ],
        },
        "outputs": {},
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=_default_artifact_root())
    parser.add_argument(
        "--env-preservation-audit",
        type=Path,
        default=_workspace_root()
        / "artifacts"
        / "phase2_reward_component_env_preservation_audit_executable_formula_axis_component_hardaxis_0518"
        / "reward_component_env_preservation_audit.json",
    )
    parser.add_argument(
        "--base-env-topology-audit",
        type=Path,
        default=_workspace_root()
        / "artifacts"
        / "phase2_base_env_reward_topology_evidence_audit_0518"
        / "base_env_reward_topology_evidence_audit.json",
    )
    parser.add_argument(
        "--terminal-microfamily-audit",
        type=Path,
        default=_workspace_root()
        / "artifacts"
        / "phase2_terminal_event_true_runner_microfamily_audit_0518"
        / "terminal_event_true_runner_microfamily_audit.json",
    )
    parser.add_argument(
        "--potential-progress-component-audit",
        type=Path,
        default=_default_potential_progress_component_audit(),
    )
    args = parser.parse_args()

    report = build_report(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "true_runner_reward_formula_path_evidence_audit.json"
    md_path = out_dir / "true_runner_reward_formula_path_evidence_audit.md"
    report["outputs"] = {
        "json": str(json_path),
        "markdown": str(md_path),
    }
    _write_json(json_path, report)
    _write_markdown(md_path, report)
    print(json.dumps(report["read"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
