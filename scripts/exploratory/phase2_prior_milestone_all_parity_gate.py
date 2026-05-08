#!/usr/bin/env python
"""Read-only gate for all explicit prior-generator milestones."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_SAMPLED_TOPOLOGY_PARITY = (
    "/home/chen/RLPFN/artifacts/phase2_prior_milestone_sampled_topology_pack_fit_parity_0506/"
    "pack_fit_model_parity_report.json"
)
DEFAULT_GATED_REWARD_PATH_PARITY = (
    "/home/chen/RLPFN/artifacts/phase2_prior_milestone_gated_reward_path_pack_fit_parity_0506/"
    "pack_fit_model_parity_report.json"
)


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def _check_pack_fit_report(path: str | Path, *, expected_milestone: str) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    report = _load_json(resolved)
    contract = report.get("contract", {}) or {}
    generator = report.get("prior_generator_parity", None)
    if generator is None:
        generator = report.get("sampled_topology_generator_parity", {}) or {}
    prior_update = report.get("prior_pack_ppo_update_parity", {}) or {}
    return {
        "path": str(resolved),
        "script": report.get("script"),
        "script_ok": report.get("script") == "phase2_fit_model_pack_ppo_migration_parity",
        "expected_milestone": str(expected_milestone),
        "contract_milestone": contract.get("prior_milestone"),
        "milestone_ok": contract.get("prior_milestone") == str(expected_milestone),
        "report_hard_pass": bool(report.get("hard_pass")),
        "generator_parity_hard_pass": bool(generator.get("hard_pass")),
        "prior_pack_ppo_update_hard_pass": bool(prior_update.get("hard_pass")),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    checks = {
        "sampled_topology": _check_pack_fit_report(
            args.sampled_topology_parity_report,
            expected_milestone="sampled_topology",
        ),
        "gated_reward_path_balance": _check_pack_fit_report(
            args.gated_reward_path_parity_report,
            expected_milestone="gated_reward_path_balance",
        ),
    }
    hard_pass = all(
        bool(
            item["script_ok"]
            and item["milestone_ok"]
            and item["report_hard_pass"]
            and item["generator_parity_hard_pass"]
            and item["prior_pack_ppo_update_hard_pass"]
        )
        for item in checks.values()
    )
    report = {
        "analysis_entry": "phase2_prior_milestone_all_parity_gate",
        "hard_pass": bool(hard_pass),
        "all_milestones_uncorrupted": bool(hard_pass),
        "contract": {
            "read_only_existing_artifacts": True,
            "milestones_checked": ["sampled_topology", "gated_reward_path_balance"],
            "checks": [
                "pack-vs-fit_model prior generator parity",
                "pack-vs-fit_model prior PPO update parity",
                "expected milestone name preserved in report contract",
            ],
        },
        "checks": checks,
    }
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "prior_milestone_all_parity_gate_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    md_lines = [
        "# Prior Milestone All-Parity Gate",
        "",
        f"hard_pass: {hard_pass}",
        f"all_milestones_uncorrupted: {hard_pass}",
        "",
        "## Checks",
    ]
    for name, item in checks.items():
        md_lines.append(
            f"- {name}: milestone_ok={item['milestone_ok']}, "
            f"generator_parity={item['generator_parity_hard_pass']}, "
            f"prior_pack_ppo_update={item['prior_pack_ppo_update_hard_pass']}, "
            f"report_hard_pass={item['report_hard_pass']}"
        )
    md_path = out_dir / "prior_milestone_all_parity_gate_summary.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(json.dumps({"hard_pass": hard_pass, "report": str(report_path)}, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sampled-topology-parity-report", type=str, default=DEFAULT_SAMPLED_TOPOLOGY_PARITY)
    parser.add_argument("--gated-reward-path-parity-report", type=str, default=DEFAULT_GATED_REWARD_PATH_PARITY)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/chen/RLPFN/artifacts/phase2_prior_milestone_all_parity_gate_0506",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
