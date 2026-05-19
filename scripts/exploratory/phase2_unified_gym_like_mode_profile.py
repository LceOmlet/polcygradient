#!/usr/bin/env python
"""Build a unified Gym-like mode profile for milestone2.

This is a read-only profile/guardrail builder. It consolidates existing
detectors and short-smoke sidecars into one table so future generator repairs
can be checked against protected modes instead of changing one axis blindly.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_OUTPUT_DIR = "/home/chen/RLPFN/artifacts/phase2_unified_gym_like_mode_profile_gpu0_0510"
DEFAULT_ACTION_MODE_JSON = (
    "/home/chen/RLPFN/artifacts/phase2_prior_action_mode_coverage_probe_0508/"
    "gated_full_q90_n64_s256_v2.json"
)
DEFAULT_LOCOMOTION_STRAT_JSON = (
    "/home/chen/RLPFN/artifacts/phase2_locomotion_trainability_stratification_gpu0_0510/"
    "locomotion_trainability_stratification_report.json"
)
DEFAULT_LONG_HORIZON_REPORT = (
    "/home/chen/RLPFN/artifacts/phase2_long_horizon_control_mode_extract_0509/"
    "long_horizon_control_mode_extract_report.json"
)
DEFAULT_FULL_LONG_HORIZON_CSV = (
    "/home/chen/RLPFN/artifacts/phase2_long_horizon_control_mode_extract_0509/"
    "full/long_horizon_control_labels.csv"
)
DEFAULT_Q90_LONG_HORIZON_CSV = (
    "/home/chen/RLPFN/artifacts/phase2_long_horizon_control_mode_extract_0509/"
    "q90/long_horizon_control_labels.csv"
)
DEFAULT_VALIDATION_WITNESS_JSON = (
    "/home/chen/RLPFN/artifacts/phase2_validation_mode_witness_probe_0508/witness_report.json"
)
DEFAULT_GYM_AUDIT_JSON = (
    "/home/chen/RLPFN/artifacts/phase2_gym_important_mode_prior_coverage_audit_0510/"
    "gym_important_mode_prior_coverage_audit_report.json"
)


FULL_SCOPE = "phase2_exploratory_reward_group_balance_gated_full_fixedlist_n64_s512_u4_e4_advnorm_0506"
Q90_SCOPE = "phase2_exploratory_reward_group_balance_gated_q90_fixedlist_n64_s512_u4_e4_advnorm_0506"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def _load_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _f(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _rate(count: int | float | None, n: int | float | None) -> float | None:
    if count is None or n in {None, 0}:
        return None
    return float(count) / float(n)


def _fmt_rate(value: float | None) -> str:
    return "na" if value is None else f"{100.0 * float(value):.1f}%"


def _action_mode_by_label(action_json: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for item in action_json.get("lists", []):
        out[str(item.get("label"))] = item.get("classification", {})
    return out


def _long_horizon_stats(path: str | Path) -> dict[str, Any]:
    rows = _load_csv(path)
    n = len(rows)
    primary = [r for r in rows if _bool(r.get("primary_long_horizon_target"))]
    selector_hard = [r for r in rows if _bool(r.get("selector_hard_positive"))]
    profiles = Counter(str(r.get("best_feedback_profile")) for r in primary)
    gains = Counter(str(r.get("best_feedback_gain")) for r in primary)
    return {
        "n": n,
        "primary_count": len(primary),
        "primary_rate": _rate(len(primary), n),
        "selector_hard_count": len(selector_hard),
        "selector_hard_rate": _rate(len(selector_hard), n),
        "primary_profile_counts": dict(profiles),
        "primary_gain_counts": dict(gains),
        "primary_profile_dominant": profiles.most_common(1)[0] if profiles else None,
        "primary_gain_dominant": gains.most_common(1)[0] if gains else None,
    }


def _locomotion_stats(strat: dict[str, Any], scope: str) -> dict[str, Any]:
    item = strat["scopes"][scope]
    n = int(item["n_envs"])
    witnesses = int(item["locomotion_witness_count"])
    witness_stats = item["witness"]["raw_reward_sum_delta"]
    non_stats = item["nonwitness"]["raw_reward_sum_delta"]
    return {
        "n": n,
        "witness_count": witnesses,
        "witness_rate": _rate(witnesses, n),
        "witness_raw_delta_mean": witness_stats.get("mean"),
        "nonwitness_raw_delta_mean": non_stats.get("mean"),
        "witness_positive_fraction": witness_stats.get("positive_fraction"),
        "nonwitness_positive_fraction": non_stats.get("positive_fraction"),
        "mode_rates": item.get("mode_rates", {}),
    }


def _coverage(action_by_label: dict[str, dict[str, Any]], scope: str, key: str) -> float | None:
    cov = action_by_label.get(scope, {}).get("coverage", {})
    count = cov.get(f"{key}_count")
    ratio = cov.get(f"{key}_ratio")
    if ratio is not None:
        return float(ratio)
    return _rate(count, cov.get("n_envs"))


def _build_per_env_profile(strat: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in strat.get("per_env", []):
        rows.append(
            {
                "scope": row.get("scope_short"),
                "env_idx": row.get("env_idx"),
                "locomotion_witness": row.get("locomotion_witness"),
                "low_energy_not_ctrl_only": row.get("low_energy_not_ctrl_only"),
                "sign_or_phase_sensitive": row.get("sign_or_phase_sensitive"),
                "state_conditioned_energy_injection": row.get("state_conditioned_energy_injection"),
                "pendulum_like_primary_long_horizon": row.get("pendulum_like_primary_long_horizon"),
                "pendulum_like_selector_hard_positive": row.get("pendulum_like_selector_hard_positive"),
                "terminal_metadata_enabled": row.get("terminal_metadata_enabled"),
                "terminal_activity_first_done": row.get("terminal_activity_first_done"),
                "raw_reward_sum_delta": row.get("raw_reward_sum_delta"),
                "reward_env_sum_delta": row.get("reward_env_sum_delta"),
                "training_reward_sum_delta": row.get("training_reward_sum_delta"),
                "done_count_delta": row.get("done_count_delta"),
                "action_value_absmean_delta": row.get("action_value_absmean_delta"),
                "action_saturation_delta": row.get("action_saturation_delta"),
            }
        )
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    action = _load_json(args.action_mode_json)
    action_by_label = _action_mode_by_label(action)
    strat = _load_json(args.locomotion_strat_json)
    validation = _load_json(args.validation_witness_json)
    gym_audit = _load_json(args.gym_audit_json)
    full_lh = _long_horizon_stats(args.full_long_horizon_csv)
    q90_lh = _long_horizon_stats(args.q90_long_horizon_csv)
    full_loco = _locomotion_stats(strat, FULL_SCOPE)
    q90_loco = _locomotion_stats(strat, Q90_SCOPE)

    full_cov = action_by_label[FULL_SCOPE].get("coverage", {})
    q90_cov = action_by_label[Q90_SCOPE].get("coverage", {})
    n_full = int(full_cov.get("n_envs", 64))
    n_q90 = int(q90_cov.get("n_envs", 64))

    modes: list[dict[str, Any]] = [
        {
            "mode_id": "locomotion_sustained_temporal_control",
            "gym_targets": "Ant-v5, HalfCheetah-v5, Swimmer-v5, Walker2d-v5, Hopper-v5, Humanoid-v5",
            "definition": (
                "same-env temporal counterfactual where sustained nonzero action improves env reward over zero, "
                "keeps terminal behavior approximately invariant, and increases state path length"
            ),
            "prior_full_rate": full_loco["witness_rate"],
            "prior_q90_rate": q90_loco["witness_rate"],
            "trainability_support": (
                f"yes: full witness raw delta {full_loco['witness_raw_delta_mean']:.2f} vs "
                f"{full_loco['nonwitness_raw_delta_mean']:.2f}; q90 witness "
                f"{q90_loco['witness_raw_delta_mean']:.2f} vs {q90_loco['nonwitness_raw_delta_mean']:.2f}"
            ),
            "current_read": "real coverage gap; useful guard; not enough to repair alone",
            "protection_guard": (
                "raise temporal witness rate only if low-energy and Pendulum-like coverage do not regress; "
                "q90 low-energy overlap is the main known tension"
            ),
            "repair_priority": "high_candidate_after_profile_guards",
        },
        {
            "mode_id": "low_energy_weak_action_optimum",
            "gym_targets": "Ant-v5, HalfCheetah-v5, Swimmer-v5",
            "definition": "zero/half action is globally or near-globally better than stronger random action families",
            "prior_full_rate": _coverage(action_by_label, FULL_SCOPE, "low_energy_not_ctrl_only"),
            "prior_q90_rate": _coverage(action_by_label, Q90_SCOPE, "low_energy_not_ctrl_only"),
            "trainability_support": "not isolated yet; protected because Gym locomotion includes low-energy optimum cases",
            "current_read": "covered at moderate rate; not the current scarcity",
            "protection_guard": "future locomotion repair must keep this near baseline, especially q90",
            "repair_priority": "protect_not_repair",
        },
        {
            "mode_id": "sign_phase_sensitive_action",
            "gym_targets": "Pendulum-v1, Reacher-v5, locomotion direction/phase cases",
            "definition": "action sign/phase changes return by a dynamic margin under matched state/context",
            "prior_full_rate": _coverage(action_by_label, FULL_SCOPE, "sign_or_phase_sensitive_present"),
            "prior_q90_rate": _coverage(action_by_label, Q90_SCOPE, "sign_or_phase_sensitive_present"),
            "trainability_support": "broad detector only; not sufficient for Pendulum transfer",
            "current_read": "broadly present, but can be a false comfort if not context-readable",
            "protection_guard": "do not reduce broad sign/phase coverage while repairing locomotion or strict profile/gain",
            "repair_priority": "protect_and_refine_readout",
        },
        {
            "mode_id": "state_conditioned_energy_injection",
            "gym_targets": "Pendulum-v1, MountainCarContinuous-v0, locomotion recovery/control cases",
            "definition": "state-conditioned feedback beats non-state-conditioned baselines by a margin",
            "prior_full_rate": _coverage(action_by_label, FULL_SCOPE, "state_conditioned_energy_injection_witness"),
            "prior_q90_rate": _coverage(action_by_label, Q90_SCOPE, "state_conditioned_energy_injection_witness"),
            "trainability_support": "locomotion temporal witnesses concentrate in this mode; broad mode alone is too permissive",
            "current_read": "covered broadly; needs temporal/future-basin refinements",
            "protection_guard": "keep as parent mode; repairs should target refined submodes, not just increase broad rate",
            "repair_priority": "refine_not_raise_broadly",
        },
        {
            "mode_id": "context_identifiable_closed_loop_settle_feedback",
            "gym_targets": "Pendulum-v1 primary; stabilization-style Reacher/Inverted variants secondary",
            "definition": "feedback policy is context-identifiable, phase-sensitive, and supported across suffix/settle horizon",
            "prior_full_rate": full_lh["primary_rate"],
            "prior_q90_rate": q90_lh["primary_rate"],
            "trainability_support": "strict fixed-list smoke was trainable, but Gym Pendulum transfer remains weak",
            "current_read": (
                "present but profile/gain diversity is damaged; full primary profile/gain dominance "
                f"{full_lh['primary_profile_dominant']} / {full_lh['primary_gain_dominant']}; "
                f"q90 {q90_lh['primary_profile_dominant']} / {q90_lh['primary_gain_dominant']}"
            ),
            "protection_guard": "do not blindly increase strict proportion; protect profile/gain diversity and selector hard positives",
            "repair_priority": "profile_balance_candidate_after_locomotion_guard",
        },
        {
            "mode_id": "terminal_survival_action_sensitive",
            "gym_targets": "InvertedPendulum-v5, InvertedDoublePendulum-v5, Hopper/Bipedal survival components",
            "definition": "action changes first_done/survival length and PPO advantages consume that future survival length",
            "prior_full_rate": None,
            "prior_q90_rate": None,
            "trainability_support": "same-list formal detector missing; survival-reward candidate did not cleanly improve env dynamics",
            "current_read": (
                "not profile-complete; Gym done mismatch is concentrated in InvertedPendulum/InvertedDoublePendulum, "
                f"global terminal repair recommended={gym_audit.get('done_summary', {}).get('terminal_global_repair_recommended', None)}"
            ),
            "protection_guard": "do not add global survival reward; require action-sensitive terminal counterfactual on same fixed lists",
            "repair_priority": "missing_detector_do_not_repair",
        },
    ]

    guardrails = {
        "baseline_rates": {
            "low_energy": {
                "full": modes[1]["prior_full_rate"],
                "q90": modes[1]["prior_q90_rate"],
            },
            "sign_phase": {
                "full": modes[2]["prior_full_rate"],
                "q90": modes[2]["prior_q90_rate"],
            },
            "state_conditioned_energy": {
                "full": modes[3]["prior_full_rate"],
                "q90": modes[3]["prior_q90_rate"],
            },
            "pendulum_like_primary": {
                "full": modes[4]["prior_full_rate"],
                "q90": modes[4]["prior_q90_rate"],
            },
            "locomotion_temporal": {
                "full": modes[0]["prior_full_rate"],
                "q90": modes[0]["prior_q90_rate"],
            },
        },
        "candidate_repair_acceptance": [
            "must improve the targeted scarce mode in full and/or q90 on GPU0 detector",
            "must not lower low-energy, sign/phase, state-conditioned, or Pendulum-like rates beyond small measurement noise",
            "must preserve state/reward/done diversity and sidecar short PPO raw-delta positive fraction",
            "must not introduce global survival reward or terminal shaping without same-list action-sensitive terminal proof",
            "must remain exploratory until pack-to-fit_model parity and maintained-path smoke pass",
        ],
        "current_next_best_step": (
            "design a locomotion-temporal candidate only inside this guardrail, or first add the missing same-list terminal detector "
            "if terminal-sensitive coverage becomes the next suspected bottleneck"
        ),
    }

    per_env_rows = _build_per_env_profile(strat)
    report = {
        "analysis_entry": "phase2_unified_gym_like_mode_profile_gpu0",
        "contract": {
            "exploratory_only": True,
            "reads_existing_artifacts_only": True,
            "does_not_modify_exact_scm": True,
            "does_not_modify_fit_model": True,
            "does_not_modify_ppo": True,
            "does_not_sample_environments": True,
            "does_not_train": True,
        },
        "sources": {
            "action_mode_json": str(Path(args.action_mode_json).expanduser().resolve()),
            "locomotion_strat_json": str(Path(args.locomotion_strat_json).expanduser().resolve()),
            "long_horizon_report": str(Path(args.long_horizon_report).expanduser().resolve()),
            "full_long_horizon_csv": str(Path(args.full_long_horizon_csv).expanduser().resolve()),
            "q90_long_horizon_csv": str(Path(args.q90_long_horizon_csv).expanduser().resolve()),
            "validation_witness_json": str(Path(args.validation_witness_json).expanduser().resolve()),
            "gym_audit_json": str(Path(args.gym_audit_json).expanduser().resolve()),
        },
        "modes": modes,
        "guardrails": guardrails,
        "auxiliary": {
            "validation_ant_locomotion_energy_status": validation.get("ant_locomotion_energy", {}).get(
                "sufficiency_status"
            ),
            "full_long_horizon_stats": full_lh,
            "q90_long_horizon_stats": q90_lh,
        },
        "per_env_rows": per_env_rows,
    }
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "unified_gym_like_mode_profile_report.json").write_text(
        json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (out_dir / "unified_gym_like_mode_profile.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "mode_id",
            "gym_targets",
            "prior_full_rate",
            "prior_q90_rate",
            "trainability_support",
            "current_read",
            "protection_guard",
            "repair_priority",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in modes:
            writer.writerow({key: _json_safe(row.get(key)) for key in fieldnames})
    if per_env_rows:
        with (out_dir / "unified_gym_like_per_env_profile.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(per_env_rows[0].keys()))
            writer.writeheader()
            writer.writerows(_json_safe(per_env_rows))

    lines = [
        "# Unified Gym-Like Mode Profile GPU0",
        "",
        "Read-only profile for deciding future generator repairs. It is not a milestone change.",
        "",
        "## Mode Table",
        "",
        "| mode | Gym targets | full | q90 | repair priority | guard |",
        "| --- | --- | ---: | ---: | --- | --- |",
    ]
    for row in modes:
        lines.append(
            f"| `{row['mode_id']}` | {row['gym_targets']} | {_fmt_rate(row['prior_full_rate'])} | "
            f"{_fmt_rate(row['prior_q90_rate'])} | {row['repair_priority']} | {row['protection_guard']} |"
        )
    lines.extend(
        [
            "",
            "## Key Reads",
            "",
            "- Locomotion temporal control is the strongest currently quantified scarcity: full 7.8%, q90 17.2%, and its witnesses have better short PPO raw deltas.",
            "- Low-energy is not scarce, so it should be protected rather than increased.",
            "- Pendulum-like closed-loop is present, but profile/gain/readout diversity is the risk; increasing its quantity alone is not the right repair.",
            "- Terminal-sensitive control is not profile-complete on the same fixed lists; do not repair terminal/survival globally from current evidence.",
            "",
            "## Guardrails For Any Candidate Repair",
            "",
        ]
    )
    for item in guardrails["candidate_repair_acceptance"]:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- JSON: `{out_dir / 'unified_gym_like_mode_profile_report.json'}`",
            f"- mode CSV: `{out_dir / 'unified_gym_like_mode_profile.csv'}`",
            f"- per-env CSV: `{out_dir / 'unified_gym_like_per_env_profile.csv'}`",
        ]
    )
    (out_dir / "unified_gym_like_mode_profile.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--action-mode-json", default=DEFAULT_ACTION_MODE_JSON)
    parser.add_argument("--locomotion-strat-json", default=DEFAULT_LOCOMOTION_STRAT_JSON)
    parser.add_argument("--long-horizon-report", default=DEFAULT_LONG_HORIZON_REPORT)
    parser.add_argument("--full-long-horizon-csv", default=DEFAULT_FULL_LONG_HORIZON_CSV)
    parser.add_argument("--q90-long-horizon-csv", default=DEFAULT_Q90_LONG_HORIZON_CSV)
    parser.add_argument("--validation-witness-json", default=DEFAULT_VALIDATION_WITNESS_JSON)
    parser.add_argument("--gym-audit-json", default=DEFAULT_GYM_AUDIT_JSON)
    args = parser.parse_args()
    report = run(args)
    print(
        json.dumps(
            {
                "output_dir": args.output_dir,
                "mode_count": len(report["modes"]),
                "next": report["guardrails"]["current_next_best_step"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
