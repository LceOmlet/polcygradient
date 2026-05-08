#!/usr/bin/env python
"""Read-only gate for the exploratory gated reward-path candidate milestone."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_DECISION_PARITY = (
    "/home/chen/RLPFN/artifacts/phase2_exploratory_gated_wrapper_decision_parity_0506/"
    "gated_wrapper_decision_parity_report.json"
)
DEFAULT_REWARD_PATH_PARITY = (
    "/home/chen/RLPFN/artifacts/phase2_exploratory_gated_reward_path_only_parity_0506/"
    "gated_reward_path_only_parity_report.json"
)
DEFAULT_DUAL_SUPPORT = (
    "/home/chen/RLPFN/artifacts/phase2_exploratory_gated_dual_support_acceptance_0506/"
    "gated_dual_support_acceptance_report.json"
)
DEFAULT_PACK_FIT_MODEL_PARITY = (
    "/home/chen/RLPFN/artifacts/phase2_exploratory_gated_pack_fit_model_parity_0506/"
    "gated_pack_fit_model_parity_report.json"
)


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def _check_report(path: str | Path, expected_entry: str) -> dict[str, Any]:
    report = _load_json(path)
    return {
        "path": str(Path(path).expanduser().resolve()),
        "analysis_entry": report.get("analysis_entry"),
        "expected_entry": str(expected_entry),
        "entry_ok": report.get("analysis_entry") == str(expected_entry),
        "hard_pass": bool(report.get("hard_pass")),
        "exploratory_only": bool(report.get("exploratory_only", True)),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    checks = {
        "decision_table_parity": _check_report(
            args.decision_parity_report,
            "phase2_gated_wrapper_decision_parity",
        ),
        "reward_path_only_parity": _check_report(
            args.reward_path_parity_report,
            "phase2_gated_reward_path_only_parity",
        ),
        "dual_support_acceptance": _check_report(
            args.dual_support_report,
            "phase2_gated_dual_support_acceptance",
        ),
        "pack_to_fit_model_parity": _check_report(
            args.pack_fit_model_parity_report,
            "phase2_gated_pack_fit_model_parity",
        ),
    }
    hard_pass = all(
        bool(item["entry_ok"] and item["hard_pass"] and item["exploratory_only"])
        for item in checks.values()
    )
    report = {
        "analysis_entry": "phase2_gated_candidate_milestone_gate",
        "exploratory_only": True,
        "hard_pass": bool(hard_pass),
        "candidate_milestone_allowed": bool(hard_pass),
        "formal_default_generator_milestone_allowed": False,
        "formal_default_generator_blocker": (
            "This gate validates the exploratory fixed-list candidate and its pack-to-fit_model "
            "PPO bridge parity. Switching the default fit_model generator still requires an explicit "
            "online generator-level integration choice."
        ),
        "contract": {
            "does_not_modify_exact_scm": True,
            "does_not_modify_default_prior_generator": True,
            "does_not_train_ppo_longrun": True,
            "read_only_existing_artifacts": True,
        },
        "checks": checks,
    }
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "gated_candidate_milestone_gate_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_lines = [
        "# Gated Candidate Milestone Gate",
        "",
        f"hard_pass: {hard_pass}",
        f"candidate_milestone_allowed: {hard_pass}",
        "formal_default_generator_milestone_allowed: False",
        "",
        "## Checks",
    ]
    for name, item in checks.items():
        md_lines.append(
            f"- {name}: entry_ok={item['entry_ok']}, hard_pass={item['hard_pass']}, "
            f"exploratory_only={item['exploratory_only']}"
        )
    md_path = out_dir / "gated_candidate_milestone_gate_summary.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(json.dumps({"hard_pass": hard_pass, "report": str(report_path)}, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision-parity-report", type=str, default=DEFAULT_DECISION_PARITY)
    parser.add_argument("--reward-path-parity-report", type=str, default=DEFAULT_REWARD_PATH_PARITY)
    parser.add_argument("--dual-support-report", type=str, default=DEFAULT_DUAL_SUPPORT)
    parser.add_argument("--pack-fit-model-parity-report", type=str, default=DEFAULT_PACK_FIT_MODEL_PARITY)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/chen/RLPFN/artifacts/phase2_exploratory_gated_candidate_milestone_gate_0506",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
