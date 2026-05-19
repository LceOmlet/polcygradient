#!/usr/bin/env python
"""Materialize a raw prior source pool for low-tail support estimation.

This diagnostic samples the current maintained prior source distribution,
keeps rows that satisfy Gym validation dimension rules, and freezes the raw
sampled ``h`` values for the trusted pack runner.  It intentionally does not
apply reward balancing, topology repair, exact-SCM low-tail repair, selector
quotas, or any post-h deformation.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.exploratory.phase2_gym_near_prior_filter import (  # noqa: E402
    DEFAULT_GYM_ENV_IDS,
    _dim_row,
    _gym_validation_features,
    _json_safe,
    _make_rules,
    _parse_csv_list,
    _stats,
)
from ticl.analysis.fixed_env_h import _to_builtin, summarize_fixed_env_h  # noqa: E402
from ticl.model_configs import get_model_default_config  # noqa: E402
from ticl.priors.environment_prior import EnvironmentPrior  # noqa: E402
from ticl.rlpfn_maintained_path import validate_rlpfn_maintained_path_config  # noqa: E402
from ticl.train import _freeze_env_h_list_for_replay  # noqa: E402


def _write_frozen_h(*, prior: EnvironmentPrior, h: dict[str, Any], path: Path) -> dict[str, Any]:
    frozen_h = _freeze_env_h_list_for_replay(prior, [copy.deepcopy(h)])[0]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_builtin(frozen_h), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summarize_fixed_env_h(frozen_h)


def run(args: argparse.Namespace) -> dict[str, Any]:
    rng_seed = int(args.seed)
    random.seed(rng_seed)
    np.random.seed(rng_seed)
    torch.manual_seed(rng_seed)

    out_dir = Path(args.output_dir).expanduser().resolve()
    frozen_dir = out_dir / "frozen_h"
    out_dir.mkdir(parents=True, exist_ok=True)
    frozen_dir.mkdir(parents=True, exist_ok=True)

    gym_env_ids = _parse_csv_list(args.gym_env_ids)
    gym_rows, gym_range = _gym_validation_features(gym_env_ids)
    all_rules = _make_rules(gym_range)
    selected_rule_names = _parse_csv_list(args.rules)
    unknown_rules = sorted(set(selected_rule_names) - set(all_rules))
    if unknown_rules:
        raise ValueError(f"Unknown rule(s): {unknown_rules}; available={sorted(all_rules)}")

    cfg = copy.deepcopy(get_model_default_config("rlpfn"))
    validate_rlpfn_maintained_path_config(cfg)
    prior = EnvironmentPrior(copy.deepcopy(cfg["prior"]["environment"]))

    selected_rows_by_rule: dict[str, list[dict[str, Any]]] = {rule: [] for rule in selected_rule_names}
    selected_fingerprints_by_rule: dict[str, set[str]] = {rule: set() for rule in selected_rule_names}
    rule_dim_match_counts = {rule: 0 for rule in selected_rule_names}

    target_per_rule = int(args.target_per_rule)
    max_raw_samples = int(args.max_raw_samples)
    raw_batch_size = int(args.raw_batch_size)
    raw_seen = 0
    dim_matched = 0

    def rules_for_dim_row(row: dict[str, Any]) -> list[str]:
        return [rule for rule in selected_rule_names if all_rules[rule](row)]

    def rule_done(rule: str) -> bool:
        return len(selected_rows_by_rule[rule]) >= target_per_rule

    def all_done() -> bool:
        return all(rule_done(rule) for rule in selected_rule_names)

    while raw_seen < max_raw_samples and not all_done():
        batch_n = min(raw_batch_size, max_raw_samples - raw_seen)
        h_batch = list(prior._sample_batch_hypers(batch_n))
        for local_idx, h in enumerate(h_batch):
            candidate_idx = raw_seen + local_idx
            env_seed = int(args.env_seed_start) + int(candidate_idx)
            dim_row = _dim_row(prior, h, candidate_idx, env_seed)
            matched_rules = rules_for_dim_row(dim_row)
            if not matched_rules:
                continue
            dim_matched += 1
            for rule in matched_rules:
                rule_dim_match_counts[rule] += 1
                if rule_done(rule):
                    continue
                h_path = frozen_dir / rule / f"{rule}_{len(selected_rows_by_rule[rule]):05d}_seed{env_seed}.json"
                summary = _write_frozen_h(prior=prior, h=h, path=h_path)
                fingerprint = str(summary["frozen_h_fingerprint"])
                if fingerprint in selected_fingerprints_by_rule[rule]:
                    continue
                selected_fingerprints_by_rule[rule].add(fingerprint)
                selected_row = {
                    **dim_row,
                    "rule": str(rule),
                    "source_pool_candidate": True,
                    "screen_kind": "dimension_only_raw_source",
                    "frozen_h_fingerprint": fingerprint,
                    "full_frozen_h_json": str(h_path),
                }
                selected_rows_by_rule[rule].append(selected_row)
            if all_done():
                break
        raw_seen += batch_n
        if raw_seen % int(args.progress_every_raw) == 0 or all_done():
            print(
                json.dumps(
                    {
                        "raw_seen": int(raw_seen),
                        "dim_matched": int(dim_matched),
                        "selected": {rule: len(rows) for rule, rows in selected_rows_by_rule.items()},
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    csv_path = out_dir / "selected_prior_envs.csv"
    fieldnames = [
        "rule",
        "seed",
        "env_seed",
        "candidate_idx",
        "state_dim",
        "obs_dim",
        "action_dim",
        "noise_dim",
        "zero_pad_dim",
        "source_pool_candidate",
        "screen_kind",
        "frozen_h_fingerprint",
        "full_frozen_h_json",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rule in selected_rule_names:
            for row in selected_rows_by_rule[rule]:
                writer.writerow({name: row.get(name, "") for name in fieldnames})

    report = {
        "analysis_entry": "phase2_prior_raw_source_pool_materialize",
        "contract": {
            "read_only": True,
            "does_not_modify_generator": True,
            "does_not_apply_selector_quota_beyond_requested_dimension_rule_count": True,
            "does_not_apply_reward_balance": True,
            "does_not_apply_topology_repair": True,
            "does_not_apply_exact_scm_lowtail_repair": True,
            "output_is_fixed_frozen_list_for_pack_runner": True,
        },
        "gym_env_ids": gym_env_ids,
        "gym_validation_rows": gym_rows,
        "gym_feature_range": gym_range,
        "rules": selected_rule_names,
        "sampling": {
            "seed": int(rng_seed),
            "env_seed_start": int(args.env_seed_start),
            "raw_seen": int(raw_seen),
            "max_raw_samples": int(max_raw_samples),
            "dim_matched": int(dim_matched),
            "rule_dim_match_counts": rule_dim_match_counts,
            "target_per_rule": int(target_per_rule),
        },
        "selected_count_by_rule": {rule: int(len(rows)) for rule, rows in selected_rows_by_rule.items()},
        "selected_summary_by_rule": {
            rule: {
                "n": int(len(rows)),
                "obs_dim": _stats([float(row["obs_dim"]) for row in rows]),
                "action_dim": _stats([float(row["action_dim"]) for row in rows]),
                "state_dim": _stats([float(row["state_dim"]) for row in rows]),
                "noise_dim": _stats([float(row["noise_dim"]) for row in rows]),
                "zero_pad_dim": _stats([float(row["zero_pad_dim"]) for row in rows]),
            }
            for rule, rows in selected_rows_by_rule.items()
        },
        "files": {
            "selected_csv": str(csv_path),
            "frozen_h_dir": str(frozen_dir),
        },
    }
    report_path = out_dir / "raw_source_pool_report.json"
    report_path.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"csv": str(csv_path), "report": str(report_path), "selected_count_by_rule": report["selected_count_by_rule"]}, sort_keys=True))
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--gym-env-ids", type=str, default=DEFAULT_GYM_ENV_IDS)
    parser.add_argument("--rules", type=str, default="gym_q90_obs_action_range")
    parser.add_argument("--target-per-rule", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260517)
    parser.add_argument("--env-seed-start", type=int, default=9170000)
    parser.add_argument("--raw-batch-size", type=int, default=4096)
    parser.add_argument("--max-raw-samples", type=int, default=200000)
    parser.add_argument("--progress-every-raw", type=int, default=8192)
    return parser


def main() -> None:
    run(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
