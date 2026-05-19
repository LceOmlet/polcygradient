#!/usr/bin/env python
"""Audit what is missing before v2e can become a generator-level selector.

The fixed-list v2e candidate is paritied, but fit_model needs a generator-level
process: each environment batch should be freshly produced by the prior and then
selected by the same profile contract.  This read-only audit separates labels
that are available online from labels that currently only exist as offline probe
artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ART = Path("/home/chen/RLPFN/artifacts")
DEFAULT_SELECTED_CSV = (
    ART
    / "phase2_profile_v2e_candidate_annotated_0511"
    / "selected_profile_v2e_candidate_prior_envs.csv"
)
DEFAULT_SOURCE_POOL_CSV = (
    ART / "phase2_exploratory_reward_group_balance_gated_n128_seed6262_0510/selected_prior_envs.csv"
)
DEFAULT_SPEC_JSON = ART / "phase2_profile_v2e_mechanism_spec_0511/profile_v2e_mechanism_spec.json"
DEFAULT_OUTPUT_DIR = ART / "phase2_profile_v2e_generator_level_gap_audit_0511"


PROFILE_AXES = {
    "locomotion_witness": {
        "label_source": "locomotion temporal detector CSV",
        "online_from_h": False,
        "reason": "requires controller rollouts / temporal reward-state response, not present in frozen_h scalar fields",
    },
    "low_energy_not_ctrl_only": {
        "label_source": "prior action mode coverage probe JSON",
        "online_from_h": False,
        "reason": "requires counterfactual action-family rollouts to identify weak-action optimum",
    },
    "sign_or_phase_sensitive": {
        "label_source": "prior action mode coverage probe JSON",
        "online_from_h": False,
        "reason": "requires paired sign/phase action counterfactuals",
    },
    "state_conditioned_energy_injection": {
        "label_source": "prior action mode coverage probe JSON",
        "online_from_h": False,
        "reason": "requires state-conditioned intervention witness, not just reward topology gains",
    },
    "context_identifiable_closed_loop_settle_feedback": {
        "label_source": "strict feedback + time-profile core probes",
        "online_from_h": False,
        "reason": "requires closed-loop feedback family replay and near-best gain evaluation",
    },
    "mountaincar_upfront_investment_future_basin": {
        "label_source": "MountainCar future-basin detector",
        "online_from_h": False,
        "reason": "requires early-action/suffix-return basin counterfactuals",
    },
    "reward_path_topology_health": {
        "label_source": "gated reward-path milestone metadata",
        "online_from_h": True,
        "reason": "reward state/action/noise gains are available during gated sampling",
    },
    "terminal_target_metadata": {
        "label_source": "h fields / gated terminal metadata",
        "online_from_h": True,
        "reason": "terminal_reset_enabled and terminal_reset_count_target are h fields",
    },
}


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).expanduser().resolve().open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def _rule_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        rule = str(row.get("rule", ""))
        out[rule] = out.get(rule, 0) + 1
    return out


def _selected_columns_read(rows: list[dict[str, str]]) -> dict[str, Any]:
    columns = set(rows[0]) if rows else set()
    return {
        "has_v2e_profile_columns": {
            key: key in columns for key in PROFILE_AXES if key not in {"reward_path_topology_health", "terminal_target_metadata"}
        },
        "has_frozen_h_path": "full_frozen_h_json" in columns,
        "has_env_seed": "env_seed" in columns,
    }


def build_report(
    *,
    selected_csv: str | Path,
    source_pool_csv: str | Path,
    spec_json: str | Path,
) -> dict[str, Any]:
    selected_rows = _read_csv(selected_csv)
    source_rows = _read_csv(source_pool_csv)
    spec = _read_json(spec_json)
    offline_axes = {
        key: item for key, item in PROFILE_AXES.items() if not bool(item["online_from_h"])
    }
    online_axes = {
        key: item for key, item in PROFILE_AXES.items() if bool(item["online_from_h"])
    }
    return {
        "analysis_entry": "phase2_profile_v2e_generator_level_gap_audit",
        "contract": {
            "read_only_existing_artifacts": True,
            "does_not_sample_environments": True,
            "does_not_train": True,
            "does_not_modify_exact_scm": True,
            "does_not_modify_fit_model": True,
            "does_not_modify_ppo": True,
            "profile_quality_first": True,
        },
        "inputs": {
            "selected_csv": str(Path(selected_csv).expanduser().resolve()),
            "source_pool_csv": str(Path(source_pool_csv).expanduser().resolve()),
            "spec_json": str(Path(spec_json).expanduser().resolve()),
        },
        "current_artifacts": {
            "selected_rows": len(selected_rows),
            "selected_rule_counts": _rule_counts(selected_rows),
            "source_pool_rows": len(source_rows),
            "source_pool_rule_counts": _rule_counts(source_rows),
            "selected_columns": _selected_columns_read(selected_rows),
            "v2e_definition": spec.get("v2e_definition", {}),
        },
        "axis_availability": PROFILE_AXES,
        "generator_level_read": {
            "fixed_list_numeric_parity_is_not_generator_level": True,
            "online_label_axes": sorted(online_axes),
            "offline_label_axes": sorted(offline_axes),
            "generator_level_selector_ready": False,
            "why_not_ready": (
                "The v2e fixed-list contract depends on offline detector labels. "
                "A fit_model milestone must either port those labelers into a bounded online candidate-pool selector "
                "or train from a large labelled source-pool sampler with an explicit finite-pool contract."
            ),
        },
        "safe_implementation_options": [
            {
                "id": "online_candidate_pool_selector",
                "preferred": True,
                "definition": (
                    "For each environment batch, over-sample gated_reward_path_balance candidates, run bounded "
                    "profile labelers on the candidate pool, then apply the v2e MILP/band selector. This is the "
                    "true generator-level version but needs GPU-time profiling and parity."
                ),
                "complexity": "medium_high",
                "risk": "labeler cost and runtime variance",
            },
            {
                "id": "large_labelled_pool_sampler",
                "preferred": False,
                "definition": (
                    "Build a large labelled pool offline, then sample fresh batches from it with the v2e selector. "
                    "This gives changing batches but is finite-pool training, not fully generator-level."
                ),
                "complexity": "medium",
                "risk": "pool overfitting and weaker generator semantics",
            },
        ],
        "next_required": [
            "Do not add a fit_model v2e startup option that merely reuses the fixed 64+64 list.",
            "Choose the online candidate-pool selector as the real generator-level route.",
            "Before fit_model integration, port or wrap the profile labelers with bounded cost and produce a generator-level smoke/parity report.",
        ],
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Profile v2e Generator-Level Gap Audit",
        "",
        "This audit explains why the paritied fixed-list candidate is not yet a fit_model generator-level milestone.",
        "",
        "## Read",
        "",
    ]
    read = report["generator_level_read"]
    lines.append(f"- generator-level selector ready: `{read['generator_level_selector_ready']}`")
    lines.append(f"- fixed-list parity is not generator-level: `{read['fixed_list_numeric_parity_is_not_generator_level']}`")
    lines.append(f"- online label axes: `{read['online_label_axes']}`")
    lines.append(f"- offline label axes: `{read['offline_label_axes']}`")
    lines.append(f"- why: {read['why_not_ready']}")
    lines.extend(["", "## Axis Availability", ""])
    lines.append("| axis | online from h | label source | reason |")
    lines.append("|---|---|---|---|")
    for axis, item in report["axis_availability"].items():
        lines.append(
            f"| `{axis}` | `{item['online_from_h']}` | {item['label_source']} | {item['reason']} |"
        )
    lines.extend(["", "## Safe Implementation Options", ""])
    for item in report["safe_implementation_options"]:
        lines.append(f"- `{item['id']}` preferred=`{item['preferred']}`: {item['definition']}")
    lines.extend(["", "## Next Required", ""])
    for item in report["next_required"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Files", ""])
    for key, value in report.get("outputs", {}).items():
        lines.append(f"- `{key}`: `{value}`")
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(
        selected_csv=args.selected_csv,
        source_pool_csv=args.source_pool_csv,
        spec_json=args.spec_json,
    )
    json_path = out_dir / "profile_v2e_generator_level_gap_audit.json"
    md_path = out_dir / "profile_v2e_generator_level_gap_audit.md"
    report["outputs"] = {"json": str(json_path), "markdown": str(md_path)}
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report["outputs"], sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-csv", default=str(DEFAULT_SELECTED_CSV))
    parser.add_argument("--source-pool-csv", default=str(DEFAULT_SOURCE_POOL_CSV))
    parser.add_argument("--spec-json", default=str(DEFAULT_SPEC_JSON))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
