#!/usr/bin/env python
"""Classify natural Gym reward topologies before authorizing source samplers.

This is a read-only reduction audit.  It separates:

* reward topologies that a natural continuous-control training family should span,
* implementation details that should not become hard axes, and
* currently missing topology evidence that may justify a source-label sampler.

It intentionally does not modify the generator or materialize a new h-list.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


ART = Path("/home/chen/RLPFN/artifacts")
DEFAULT_COMPONENT_AUDIT = (
    ART
    / "phase2_reward_component_boolean_axis_audit_executable_formula_axis_component_hardaxis_0518"
    / "reward_component_boolean_axis_audit.json"
)
DEFAULT_ENV_PRESERVATION_AUDIT = (
    ART
    / "phase2_reward_component_env_preservation_audit_executable_formula_axis_component_hardaxis_0518"
    / "reward_component_env_preservation_audit.json"
)
DEFAULT_SPAN_AUDIT = (
    ART
    / "phase2_category_span_audit_executable_formula_axis_component_hardaxis_0518"
    / "category_span_audit.json"
)
DEFAULT_OUTPUT_DIR = ART / "phase2_reward_topology_requirement_audit_0518"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _coverage_status(component_audit: dict[str, Any], axis: str) -> str:
    for row in component_audit.get("coverage", []):
        if row.get("axis") == axis:
            return str(row.get("status", "missing"))
    return "missing"


def _axis_status(span_audit: dict[str, Any], axis: str) -> str:
    for row in span_audit.get("axis_coverage", []):
        if row.get("axis") == axis:
            return str(row.get("status", "missing"))
    return "missing"


def _topology_rows(
    component_audit: dict[str, Any],
    env_preservation_audit: dict[str, Any],
    span_audit: dict[str, Any],
) -> list[dict[str, Any]]:
    env_read = env_preservation_audit.get("read", {})
    no_aux_suppression = bool(env_read.get("no_auxiliary_suppression_pass", False))
    hard_component_covered = bool(env_read.get("component_hard_axis_covered", False))
    base_env_present = bool((component_audit.get("read") or {}).get("base_env_reward_present", False))
    reward_component_status = _axis_status(span_audit, "reward_component_topology")
    reward_transform_status = _axis_status(span_audit, "reward_transform")

    return [
        {
            "topology": "base_env_task_reward",
            "category": "required_natural_axis",
            "meaning": "task reward as a function of state/observation/transition; includes generic goal, distance, velocity, and dense shaping if not explicitly split",
            "gym_examples": "CartPole/Acrobot state progress, Pendulum angle cost, MuJoCo forward/progress reward, LunarLander shaping",
            "current_status": "covered_as_base_bucket" if base_env_present else "missing",
            "source_sampler_authorized_now": False,
            "reason": "base env reward is present in every current env; splitting its internal semantics requires a separate formula-evidence audit, not a blind sampler",
        },
        {
            "topology": "control_energy_cost",
            "category": "required_natural_axis",
            "meaning": "negative cost of action magnitude or motor effort",
            "gym_examples": "MuJoCo ctrl_cost, Pendulum action cost, MountainCarContinuous action cost",
            "current_status": _coverage_status(component_audit, "control_cost"),
            "source_sampler_authorized_now": False,
            "reason": "already covered as a boolean hard axis and aux does not suppress env reward",
        },
        {
            "topology": "survival_or_step_bias",
            "category": "required_natural_axis",
            "meaning": "constant alive/healthy reward, or fixed per-step bias/penalty",
            "gym_examples": "MuJoCo healthy_reward, CartPole +1 per non-terminal step, step penalties in navigation tasks",
            "current_status": _coverage_status(component_audit, "survival_bonus"),
            "source_sampler_authorized_now": False,
            "reason": "already covered as a boolean hard axis with env reward preserved",
        },
        {
            "topology": "potential_delta_or_progress",
            "category": "required_candidate_needs_evidence",
            "meaning": "reward proportional to change in potential, position, velocity, or distance-to-goal across a transition",
            "gym_examples": "BipedalWalker/LunarLander shaping deltas, MuJoCo forward progress, MountainCar goal progress",
            "current_status": _coverage_status(component_audit, "potential_delta"),
            "source_sampler_authorized_now": False,
            "reason": "important natural Gym pattern, but current maintained runner only proves a base-env bucket; lowtail potential fields are dormant/untrusted here, so sampler requires runner-image evidence first",
        },
        {
            "topology": "terminal_event_reward_or_penalty",
            "category": "required_candidate_needs_evidence",
            "meaning": "success/failure reward applied at termination or threshold crossing",
            "gym_examples": "LunarLander terminal success/failure, BipedalWalker fall penalty, MountainCar goal bonus",
            "current_status": _coverage_status(component_audit, "terminal_bonus"),
            "source_sampler_authorized_now": False,
            "reason": "broad Gym contains this pattern, but current sidecar shows terminal_bonus missing; implement only after a true runner-image component path and protection audit exist",
        },
        {
            "topology": "sparse_threshold_or_indicator_reward",
            "category": "required_candidate_needs_definition",
            "meaning": "reward from thresholded goal/contact/success predicates rather than smooth dense scalar fields",
            "gym_examples": "goal reached, legs in contact, success indicators",
            "current_status": "not_separately_logged",
            "source_sampler_authorized_now": False,
            "reason": "may be representable by base env reward plus terminal/event topology; needs a minimal predicate topology definition before sampler",
        },
        {
            "topology": "contact_or_collision_cost",
            "category": "family_specific_candidate",
            "meaning": "penalty from contacts, collisions, or constraint violations",
            "gym_examples": "BipedalWalker contact/fall, MuJoCo contact_cost in some tasks",
            "current_status": "not_separately_logged",
            "source_sampler_authorized_now": False,
            "reason": "not universal across Gym and can often be reduced to state-threshold/event penalties; do not hard-code unless contact-like modes become target family",
        },
        {
            "topology": "reward_transform_clip_tanh_dropout",
            "category": "implementation_detail_not_hard_axis",
            "meaning": "post reward transform, clipping, dropout, or wrapper normalization",
            "gym_examples": "wrappers or training pipeline transforms, not core env reward semantics",
            "current_status": reward_transform_status,
            "source_sampler_authorized_now": False,
            "reason": "sampling this as a hard env topology would mix environment semantics with preprocessing and can suppress reward",
        },
        {
            "topology": "reward_aux",
            "category": "implementation_detail_not_axis",
            "meaning": "logged aggregate reward_ctrl + reward_survival",
            "gym_examples": "none; diagnostic aggregate only",
            "current_status": "aggregate_only",
            "source_sampler_authorized_now": False,
            "reason": "not a source component; using it as a hard cell produced false gaps",
        },
        {
            "topology": "distribution_head_or_latent_reward_head",
            "category": "generator_mechanism_not_gym_topology",
            "meaning": "latent head used to shape reward distribution, not a semantic Gym reward component by itself",
            "gym_examples": "none directly",
            "current_status": _coverage_status(component_audit, "distribution_head"),
            "source_sampler_authorized_now": False,
            "reason": "can be useful implementation machinery only after a topology target is proven; not itself a Gym reward family axis",
        },
        {
            "topology": "reward_input_gain_dominance",
            "category": "diagnostic_not_semantic_topology",
            "meaning": "whether reward input gain is state/action/noise dominated",
            "gym_examples": "diagnostic of mechanism quality, not a reward formula family",
            "current_status": _axis_status(span_audit, "reward_gain_dominance"),
            "source_sampler_authorized_now": False,
            "reason": "should guard against unnatural dominance, especially noise/action dominance, rather than become a uniformly sampled Gym topology axis",
        },
    ]


def run(args: argparse.Namespace) -> dict[str, Any]:
    component_audit_path = Path(args.component_audit).expanduser().resolve()
    env_preservation_path = Path(args.env_preservation_audit).expanduser().resolve()
    span_audit_path = Path(args.span_audit).expanduser().resolve()
    component_audit = _read_json(component_audit_path)
    env_preservation = _read_json(env_preservation_path)
    span_audit = _read_json(span_audit_path)
    rows = _topology_rows(component_audit, env_preservation, span_audit)
    necessary_missing = [
        row
        for row in rows
        if str(row["category"]).startswith("required")
        and row["current_status"] in {"missing", "not_separately_logged"}
    ]
    sampler_authorized = [row for row in rows if bool(row["source_sampler_authorized_now"])]
    report = {
        "analysis_entry": "phase2_reward_topology_requirement_audit",
        "schema": "phase2_reward_topology_requirement_audit.v1",
        "contract": {
            "read_only": True,
            "does_not_modify_generator": True,
            "does_not_materialize_h_list": True,
            "separates_requirement_from_implementation": True,
        },
        "inputs": {
            "component_audit": str(component_audit_path),
            "env_preservation_audit": str(env_preservation_path),
            "span_audit": str(span_audit_path),
        },
        "read": {
            "hard_component_topology_covered": bool(
                (env_preservation.get("read") or {}).get("component_hard_axis_covered", False)
            ),
            "base_env_reward_preserved": bool(
                (env_preservation.get("read") or {}).get("base_env_reward_present_all", False)
                and (env_preservation.get("read") or {}).get("no_auxiliary_suppression_pass", False)
            ),
            "necessary_missing_or_unproven_count": len(necessary_missing),
            "source_sampler_authorized_now_count": len(sampler_authorized),
            "source_sampler_authorized_now": [row["topology"] for row in sampler_authorized],
            "next_evidence_targets": [
                "potential_delta_or_progress",
                "terminal_event_reward_or_penalty",
                "sparse_threshold_or_indicator_reward",
            ],
        },
        "topologies": rows,
    }
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "reward_topology_requirement_audit.json"
    md_path = out_dir / "reward_topology_requirement_audit.md"
    report["outputs"] = {"json": str(json_path), "markdown": str(md_path)}
    json_path.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report["outputs"], sort_keys=True))
    return report


def _markdown(report: dict[str, Any]) -> str:
    read = report["read"]
    lines = [
        "# Reward Topology Requirement Audit",
        "",
        "Read-only reduction of Gym-like reward topology requirements.",
        "",
        "## Read",
        "",
        f"- hard component topology covered: `{read['hard_component_topology_covered']}`",
        f"- base env reward preserved: `{read['base_env_reward_preserved']}`",
        f"- necessary missing or unproven count: `{read['necessary_missing_or_unproven_count']}`",
        f"- source sampler authorized now count: `{read['source_sampler_authorized_now_count']}`",
        f"- source sampler authorized now: `{read['source_sampler_authorized_now']}`",
        f"- next evidence targets: `{read['next_evidence_targets']}`",
        "",
        "## Topology Classification",
        "",
        "| topology | category | current status | sampler now | meaning | reason |",
        "|---|---|---|---:|---|---|",
    ]
    for row in report["topologies"]:
        lines.append(
            f"| `{row['topology']}` | `{row['category']}` | `{row['current_status']}` | "
            f"`{row['source_sampler_authorized_now']}` | {row['meaning']} | {row['reason']} |"
        )
    lines.extend(["", "## Outputs", ""])
    for key, value in report["outputs"].items():
        lines.append(f"- `{key}`: `{value}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component-audit", default=str(DEFAULT_COMPONENT_AUDIT))
    parser.add_argument("--env-preservation-audit", default=str(DEFAULT_ENV_PRESERVATION_AUDIT))
    parser.add_argument("--span-audit", default=str(DEFAULT_SPAN_AUDIT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
