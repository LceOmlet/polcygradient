import argparse
import csv
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ticl.analysis.phase2_prior_gym_alignment_deep_feature_audit import measure_prior_deep_features
from ticl.analysis.phase2_prior_structure_intervention_validation import (
    _get,
    _identity_curve,
    _json_safe,
    _load_thresholds,
    _run_identity,
    _summarize,
    run_no_terminal_stability,
)


DEFAULT_ENRICHED_POPULATION_CSV = (
    "/home/chen/RLPFN/artifacts/phase2_prior_generation_rule_audit_0429/enriched_population.csv"
)
DEFAULT_HARD_GATE_REPORT = (
    "/home/chen/RLPFN/artifacts/phase2_prior_gym_hard_gate_report_0429/hard_gate_report.json"
)


def _finite(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _quantile(values: list[float], frac: float) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("Cannot compute quantile on an empty list.")
    return float(np.quantile(arr, float(frac)))


def _label(value: Any, lo: float, hi: float) -> str:
    val = _finite(value)
    if val is None:
        return "missing"
    if val <= lo:
        return "low"
    if val >= hi:
        return "high"
    return "mid"


def _stats(values: list[Any]) -> dict[str, Any]:
    arr = np.asarray([v for v in (_finite(v) for v in values) if v is not None], dtype=np.float64)
    if arr.size == 0:
        return {"count": 0, "mean": None, "q50": None, "q10": None, "q90": None, "min": None, "max": None}
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "q10": float(np.quantile(arr, 0.10)),
        "q50": float(np.quantile(arr, 0.50)),
        "q90": float(np.quantile(arr, 0.90)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _load_population(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            row: dict[str, Any] = {}
            for key, value in raw.items():
                val = _finite(value)
                row[key] = val if val is not None else value
            seed = row.get("seed")
            row["frozen_h_seed"] = int(seed) if _finite(seed) is not None else seed
            row["topology_reward_state_input_gain_fraction"] = row.get("reward_state_gain")
            row["topology_reward_action_input_gain_fraction"] = row.get("reward_action_gain")
            row["topology_reward_noise_input_gain_fraction"] = row.get("reward_noise_gain")
            row["horizon_2_bias_corrected_identity_score"] = row.get("G2")
            row["horizon_5_bias_corrected_identity_score"] = row.get("G5")
            rows.append(row)
    return rows


def _add_tertile_labels(rows: list[dict[str, Any]]) -> dict[str, tuple[float, float]]:
    fields = [
        "reward_state_gain",
        "reward_action_gain",
        "reward_noise_gain",
        "reward_action_to_noise_gain_ratio",
        "reward_state_to_noise_gain_ratio",
        "_constrained_obs_u",
        "noise_budget_frac",
        "zero_budget_frac",
        "obs_frac",
        "state_dim",
        "action_dim",
        "noise_dim",
        "zero_pad_dim",
        "noise_std",
        "G5",
        "obs_rank",
        "reward_std",
        "return_std",
    ]
    for row in rows:
        noise_gain = max(float(row.get("reward_noise_gain") or 0.0), 1e-12)
        row["reward_action_to_noise_gain_ratio"] = float((row.get("reward_action_gain") or 0.0) / noise_gain)
        row["reward_state_to_noise_gain_ratio"] = float((row.get("reward_state_gain") or 0.0) / noise_gain)
    cuts: dict[str, tuple[float, float]] = {}
    for field in fields:
        vals = [_finite(row.get(field)) for row in rows]
        vals = [v for v in vals if v is not None]
        if vals:
            cuts[field] = (_quantile(vals, 1.0 / 3.0), _quantile(vals, 2.0 / 3.0))
    for row in rows:
        labels = {}
        for field, (lo, hi) in cuts.items():
            labels[field] = _label(row.get(field), lo, hi)
        row["rule_labels"] = labels
    return cuts


def _lab(row: dict[str, Any], field: str) -> str:
    labels = row.get("rule_labels") or {}
    return str(labels.get(field, "missing"))


RulePredicate = Callable[[dict[str, Any]], bool]


def _rules() -> dict[str, dict[str, Any]]:
    return {
        "population_random_control": {
            "predicate": lambda row: True,
            "rationale": "Representative baseline over the same enriched population; detects whether candidate cells beat random coverage.",
        },
        "identity_preserving_topology": {
            "predicate": lambda row: _lab(row, "reward_state_gain") == "high"
            and _lab(row, "reward_noise_gain") != "high"
            and _lab(row, "reward_action_gain") != "high",
            "rationale": "Tests the state-dominant reward topology that should preserve G2-G5 identity, while checking whether scale remains too weak.",
        },
        "balanced_topology_mid_noise": {
            "predicate": lambda row: _lab(row, "reward_noise_gain") == "mid"
            and _lab(row, "reward_state_gain") != "low"
            and _lab(row, "reward_action_gain") != "high",
            "rationale": "Tests the non-extreme reward-noise path as a possible scale helper without immediately paying the high-noise identity cost.",
        },
        "balanced_topology_obsu_nonhigh": {
            "predicate": lambda row: _lab(row, "reward_noise_gain") == "mid"
            and _lab(row, "reward_state_gain") != "low"
            and _lab(row, "reward_action_gain") != "high"
            and _lab(row, "_constrained_obs_u") != "high",
            "rationale": "Tests balanced reward topology while rejecting the high visible-obs allocation side of the geometry contrast.",
        },
        "balanced_topology_obsu_high_contrast": {
            "predicate": lambda row: _lab(row, "reward_noise_gain") == "mid"
            and _lab(row, "reward_state_gain") != "low"
            and _lab(row, "reward_action_gain") != "high"
            and _lab(row, "_constrained_obs_u") == "high",
            "rationale": "Contrast for balanced topology: isolates high obs-u while keeping reward topology family fixed.",
        },
        "balanced_topology_noisebudget_mid": {
            "predicate": lambda row: _lab(row, "reward_noise_gain") == "mid"
            and _lab(row, "reward_state_gain") != "low"
            and _lab(row, "reward_action_gain") != "high"
            and _lab(row, "noise_budget_frac") == "mid",
            "rationale": "Tests controlled noise budget under the balanced reward topology family.",
        },
        "balanced_topology_noisebudget_low_contrast": {
            "predicate": lambda row: _lab(row, "reward_noise_gain") == "mid"
            and _lab(row, "reward_state_gain") != "low"
            and _lab(row, "reward_action_gain") != "high"
            and _lab(row, "noise_budget_frac") == "low",
            "rationale": "Contrast for controlled noise budget: same reward topology family with low noise budget.",
        },
        "dim_balanced_geometry": {
            "predicate": lambda row: _lab(row, "_constrained_obs_u") == "mid"
            and _lab(row, "noise_budget_frac") == "mid",
            "rationale": "Tests constrained-dim coupling directly: visible obs allocation and noise/zero-pad budget are controlled before materialization.",
        },
        "topology_dim_coupled_candidate": {
            "predicate": lambda row: _lab(row, "reward_noise_gain") == "mid"
            and _lab(row, "noise_budget_frac") == "mid"
            and _lab(row, "_constrained_obs_u") != "high",
            "rationale": "Tests whether reward-scale and geometry controls are complementary only as a coupled generation rule.",
        },
        "strict_balanced_topology_nonextreme_dim_mid_noisebudget": {
            "predicate": lambda row: _lab(row, "reward_noise_gain") == "mid"
            and _lab(row, "reward_state_gain") != "low"
            and _lab(row, "reward_action_gain") != "high"
            and _lab(row, "_constrained_obs_u") != "high"
            and _lab(row, "noise_budget_frac") == "mid",
            "rationale": "Strict candidate: balanced reward topology with non-extreme visible geometry and mid noise budget.",
        },
        "strict_balanced_topology_mid_dim_mid_noisebudget": {
            "predicate": lambda row: _lab(row, "reward_noise_gain") == "mid"
            and _lab(row, "reward_state_gain") != "low"
            and _lab(row, "reward_action_gain") != "high"
            and _lab(row, "_constrained_obs_u") == "mid"
            and _lab(row, "noise_budget_frac") == "mid",
            "rationale": "More selective candidate: balanced topology plus mid obs allocation and mid noise budget.",
        },
        "strict_balanced_topology_mid_noisebudget_noisestd_mid": {
            "predicate": lambda row: _lab(row, "reward_noise_gain") == "mid"
            and _lab(row, "reward_state_gain") != "low"
            and _lab(row, "reward_action_gain") != "high"
            and _lab(row, "_constrained_obs_u") != "high"
            and _lab(row, "noise_budget_frac") == "mid"
            and _lab(row, "noise_std") == "mid",
            "rationale": "Adds a generation-time noise_std band to separate dimension noise budget from transition stochasticity.",
        },
        "strict_balanced_topology_mid_noisebudget_noisestd_not_high": {
            "predicate": lambda row: _lab(row, "reward_noise_gain") == "mid"
            and _lab(row, "reward_state_gain") != "low"
            and _lab(row, "reward_action_gain") != "high"
            and _lab(row, "_constrained_obs_u") != "high"
            and _lab(row, "noise_budget_frac") == "mid"
            and _lab(row, "noise_std") != "high",
            "rationale": "Less brittle sensitivity control: keeps strict topology/dim rule while excluding high transition noise.",
        },
        "strict_balanced_topology_bad_noisebudget_contrast": {
            "predicate": lambda row: _lab(row, "reward_noise_gain") == "mid"
            and _lab(row, "reward_state_gain") != "low"
            and _lab(row, "reward_action_gain") != "high"
            and _lab(row, "_constrained_obs_u") != "high"
            and _lab(row, "noise_budget_frac") != "mid",
            "rationale": "Negative contrast for the strict candidate: same topology family but unbalanced noise budget.",
        },
        "relaxed_midnoise_state_notlow_obsu_nonhigh_nb_nonhigh": {
            "predicate": lambda row: _lab(row, "reward_noise_gain") == "mid"
            and _lab(row, "reward_state_gain") != "low"
            and _lab(row, "_constrained_obs_u") != "high"
            and _lab(row, "noise_budget_frac") != "high",
            "rationale": "Larger-sample coupled rule: mid reward-noise, non-low state reward, non-high obs-u, and non-high noise budget; action gain is audited rather than pre-filtered.",
        },
        "relaxed_midnoise_state_notlow_nb_mid": {
            "predicate": lambda row: _lab(row, "reward_noise_gain") == "mid"
            and _lab(row, "reward_state_gain") != "low"
            and _lab(row, "noise_budget_frac") == "mid",
            "rationale": "Larger-sample controlled-noise-budget rule that tests whether action-gain filtering was over-restrictive.",
        },
        "action_low_obs_mid_candidate": {
            "predicate": lambda row: _lab(row, "reward_action_gain") == "low"
            and _lab(row, "_constrained_obs_u") == "mid",
            "rationale": "Re-tests the strongest prior diagnostic pair as a candidate, while explicitly checking its action-sensitivity downside.",
        },
        "bad_noise_reward_contrast": {
            "predicate": lambda row: _lab(row, "reward_noise_gain") == "high"
            and _lab(row, "reward_state_gain") == "low",
            "rationale": "Negative contrast: high reward-noise path should expose the scale-vs-identity failure mode if that hypothesis is real.",
        },
        "bad_obsu_high_geometry_contrast": {
            "predicate": lambda row: _lab(row, "_constrained_obs_u") == "high"
            and _lab(row, "noise_budget_frac") != "mid",
            "rationale": "Negative contrast: high visible-fraction with unbalanced noise budget should expose geometry/drift side effects.",
        },
    }


def _select_rule_rows(
    *,
    population: list[dict[str, Any]],
    rule_name: str,
    predicate: RulePredicate,
    count: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates = [row for row in population if bool(predicate(row))]
    if len(candidates) <= int(count):
        return list(candidates), candidates
    seed_material = f"{rule_name}:{int(seed)}".encode("utf-8")
    stable_seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], byteorder="little") % (2**32)
    rng = np.random.default_rng(stable_seed)
    idx = sorted(rng.choice(len(candidates), size=int(count), replace=False).tolist())
    return [candidates[i] for i in idx], candidates


def _summarize_source_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fields = [
        "reward_state_gain",
        "reward_action_gain",
        "reward_noise_gain",
        "_constrained_obs_u",
        "noise_budget_frac",
        "zero_budget_frac",
        "obs_std",
        "obs_rank",
        "reward_std",
        "return_std",
        "reward_sens",
        "state_sens",
        "G5",
    ]
    return {field: _stats([row.get(field) for row in rows]) for field in fields}


def _write_csv(path: Path, rows: list[dict[str, Any]], extra_fields: list[str] | None = None) -> None:
    field_names = [
        "rule",
        "selected_index",
        "candidate_count",
        "seed",
        "full_frozen_h_json",
        "reward_state_gain",
        "reward_action_gain",
        "reward_noise_gain",
        "_constrained_obs_u",
        "noise_budget_frac",
        "zero_budget_frac",
        "obs_std",
        "obs_rank",
        "reward_std",
        "return_std",
        "reward_sens",
        "state_sens",
        "G5",
        "rule_labels_json",
    ]
    if extra_fields:
        field_names.extend(extra_fields)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_names)
        writer.writeheader()
        for row in rows:
            payload = {key: row.get(key) for key in field_names if key != "rule_labels_json"}
            payload["rule_labels_json"] = json.dumps(row.get("rule_labels") or {}, sort_keys=True)
            writer.writerow(payload)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate generation-before reward-topology/dim-coupling rules with canonical full gates."
    )
    parser.add_argument("--enriched-population-csv", default=DEFAULT_ENRICHED_POPULATION_CSV)
    parser.add_argument("--hard-gate-report", default=DEFAULT_HARD_GATE_REPORT)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--envs-per-rule", type=int, default=8)
    parser.add_argument("--sample-seed", type=int, default=20260429)
    parser.add_argument("--rules", default=",".join(_rules().keys()))
    parser.add_argument("--deep-rollout-count", type=int, default=32)
    parser.add_argument("--deep-n-steps", type=int, default=128)
    parser.add_argument("--deep-per-action-stride", type=int, default=32)
    parser.add_argument("--identity-reference-state-count", type=int, default=12)
    parser.add_argument("--identity-continuation-repeats", type=int, default=12)
    parser.add_argument("--identity-anchor-step", type=int, default=128)
    parser.add_argument("--identity-n-steps", type=int, default=133)
    parser.add_argument("--stability-n-steps", type=int, default=2048)
    parser.add_argument("--stability-rollout-count", type=int, default=64)
    parser.add_argument("--dry-run-only", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    thresholds = _load_thresholds(args.hard_gate_report)
    population = _load_population(args.enriched_population_csv)
    cuts = _add_tertile_labels(population)
    rules = _rules()
    requested_rules = [piece.strip() for piece in str(args.rules).split(",") if piece.strip()]
    unknown = [rule for rule in requested_rules if rule not in rules]
    if unknown:
        raise ValueError(f"Unknown rule(s): {unknown}. Known rules: {sorted(rules)}")

    rule_manifest: dict[str, Any] = {}
    selected_all: list[dict[str, Any]] = []
    for rule_name in requested_rules:
        selected, candidates = _select_rule_rows(
            population=population,
            rule_name=rule_name,
            predicate=rules[rule_name]["predicate"],
            count=int(args.envs_per_rule),
            seed=int(args.sample_seed),
        )
        for idx, row in enumerate(selected, start=1):
            out = dict(row)
            out["rule"] = rule_name
            out["selected_index"] = idx
            out["candidate_count"] = len(candidates)
            selected_all.append(out)
        rule_manifest[rule_name] = {
            "rationale": rules[rule_name]["rationale"],
            "candidate_count": int(len(candidates)),
            "selected_count": int(len(selected)),
            "candidate_source_summary": _summarize_source_rows(candidates),
            "selected_source_summary": _summarize_source_rows(selected),
        }

    selected_path = output_dir / "selected_envs.csv"
    _write_csv(selected_path, selected_all)
    manifest_path = output_dir / "rule_manifest.json"
    manifest = {
        "audit_entry": "phase2_prior_generation_rule_full_gate_validation",
        "semantics": {
            "generator_defaults_changed": False,
            "frozen_h_topology_or_dims_mutated": False,
            "overrides": {"state_full_rms_enabled": True, "state_full_rms_target": 1.0},
            "purpose": "Validate generation-before selection/rejection hypotheses, not post-hoc parameter edits.",
        },
        "inputs": {
            "enriched_population_csv": str(Path(args.enriched_population_csv).expanduser().resolve()),
            "hard_gate_report": str(Path(args.hard_gate_report).expanduser().resolve()),
            "envs_per_rule": int(args.envs_per_rule),
            "sample_seed": int(args.sample_seed),
            "rules": requested_rules,
        },
        "tertile_cuts": cuts,
        "rules": rule_manifest,
        "selected_envs_csv": str(selected_path),
    }
    manifest_path.write_text(json.dumps(_json_safe(manifest), sort_keys=True, indent=2) + "\n", encoding="utf-8")

    if bool(args.dry_run_only):
        print(json.dumps(_json_safe(manifest), sort_keys=True, indent=2))
        return

    rows_path = output_dir / "rows.jsonl"
    out_rows: list[dict[str, Any]] = []
    gate_overrides = {"state_full_rms_enabled": True, "state_full_rms_target": 1.0}
    with rows_path.open("w", encoding="utf-8") as handle:
        for global_idx, source_row in enumerate(selected_all, start=1):
            rule_name = str(source_row["rule"])
            print(
                f"[generation-rule-full-gate] {global_idx}/{len(selected_all)} "
                f"rule={rule_name} seed={source_row.get('seed')} candidates={source_row.get('candidate_count')}",
                flush=True,
            )
            deep = measure_prior_deep_features(
                row=source_row,
                case_name=rule_name,
                frozen_h_overrides=copy.deepcopy(gate_overrides),
                device=str(args.device),
                train_env_seed=2020,
                rollout_count=int(args.deep_rollout_count),
                n_steps=int(args.deep_n_steps),
                rollout_seed_start=9300000 + global_idx * 100000,
                action_delta=0.1,
                per_action_stride=int(args.deep_per_action_stride),
            )
            identity_raw = _run_identity(
                source_row=source_row,
                overrides=gate_overrides,
                device=str(args.device),
                reference_state_count=int(args.identity_reference_state_count),
                continuation_repeats=int(args.identity_continuation_repeats),
                anchor_step=int(args.identity_anchor_step),
                n_steps=int(args.identity_n_steps),
            )
            identity = {
                "identity_score": _get(identity_raw, "aggregate.identity_score"),
                **_identity_curve(identity_raw),
            }
            stability = run_no_terminal_stability(
                source_row=source_row,
                overrides=gate_overrides,
                device=str(args.device),
                n_steps=int(args.stability_n_steps),
                rollout_count=int(args.stability_rollout_count),
                rollout_seed_start=89000000 + global_idx * 100000,
            )
            row = {
                "rule": rule_name,
                "case": rule_name,
                "selected_index": source_row.get("selected_index"),
                "candidate_count": source_row.get("candidate_count"),
                "frozen_h_seed": source_row.get("frozen_h_seed"),
                "full_frozen_h_json": source_row.get("full_frozen_h_json"),
                "rule_labels": source_row.get("rule_labels"),
                "source_features": {
                    key: source_row.get(key)
                    for key in (
                        "reward_state_gain",
                        "reward_action_gain",
                        "reward_noise_gain",
                        "_constrained_obs_u",
                        "noise_budget_frac",
                        "zero_budget_frac",
                        "obs_std",
                        "obs_rank",
                        "reward_std",
                        "return_std",
                        "reward_sens",
                        "state_sens",
                        "G5",
                    )
                },
                "overrides": gate_overrides,
                "deep": deep,
                "identity": identity,
                "stability": stability,
            }
            row = _json_safe(row)
            out_rows.append(row)
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()

    summary = {
        "audit_entry": "phase2_prior_generation_rule_full_gate_validation",
        "semantics": manifest["semantics"],
        "inputs": {
            **manifest["inputs"],
            "device": str(args.device),
            "deep_rollout_count": int(args.deep_rollout_count),
            "deep_n_steps": int(args.deep_n_steps),
            "identity_reference_state_count": int(args.identity_reference_state_count),
            "identity_continuation_repeats": int(args.identity_continuation_repeats),
            "stability_n_steps": int(args.stability_n_steps),
            "stability_rollout_count": int(args.stability_rollout_count),
        },
        "rows_jsonl": str(rows_path),
        "selected_envs_csv": str(selected_path),
        "rule_manifest_json": str(manifest_path),
        "case_summary": _summarize(out_rows, thresholds),
        "interpretation_guardrail": (
            "A rule is not acceptable unless it improves multiple gates together and is later "
            "validated by fixed-group trainability. Single-metric wins are only root-cause clues."
        ),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(_json_safe(summary), sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(_json_safe(summary), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
