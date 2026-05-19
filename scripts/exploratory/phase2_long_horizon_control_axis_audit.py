#!/usr/bin/env python
"""Read-only audit for a long-horizon closed-loop prior axis.

This script does not sample environments, train PPO, or mutate exact-SCM /
fit_model defaults.  It turns existing milestone and Pendulum-signature reports
into a concrete screening proposal for the next exploratory prior axis.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_OUTPUT_DIR = "/home/chen/RLPFN/artifacts/phase2_long_horizon_control_axis_audit_0509"
DEFAULT_PENDULUM_SIGNATURE = (
    "/home/chen/RLPFN/artifacts/phase2_prior_pendulum_like_signature_probe_0509/"
    "gated_full_q90_n64_s128_obs4_scales3_timeprofile.json"
)
DEFAULT_MILESTONE_PARITY = (
    "/home/chen/RLPFN/artifacts/phase2_prior_milestone_all_parity_gate_0506/"
    "prior_milestone_all_parity_gate_report.json"
)
DEFAULT_NEXT_MILESTONE_AUDIT = (
    "/home/chen/RLPFN/artifacts/phase2_prior_next_milestone_necessity_audit_0506/"
    "prior_next_milestone_necessity_audit_report.json"
)
DEFAULT_GYM_PENDULUM_MECHANISM = (
    "/home/chen/RLPFN/artifacts/phase2_gym_pendulum_mechanism_signature_probe_0509/"
    "fit_epoch29_vs_official_single_env_prefix_v2.json"
)
DEFAULT_HIDDEN_HEAD_ROLLOUT = (
    "/home/chen/RLPFN/artifacts/phase2_gym_pendulum_frozen_hidden_head_rollout_probe_0509/"
    "fit_epoch29_hidden_head_rollout_train32_eval16.json"
)


PROTECTED_DIMENSION_FIELDS = {"action_dim", "obs_dim", "state_dim", "noise_dim"}
PARAMETER_FIELDS = (
    "alpha",
    "lengthscale",
    "outputscale",
    "init_std",
    "init_state_std",
    "init_action_std",
    "noise_std",
    "state_noise_std",
    "ctrl_reward_weight",
    "ctrl_reward_enable_prob",
    "reward_dropout_ratio",
    "reward_scale",
    "terminal_reset_count_target",
    "reward_action_input_gain_fraction_conditioned_min",
    "reward_state_input_gain_fraction_conditioned_min",
    "reward_state_to_action_gain_ratio_conditioned_max",
    "exploratory_reward_group_state_scale",
    "exploratory_reward_group_action_scale",
    "exploratory_reward_group_noise_scale",
)


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, float):
        return None if not math.isfinite(value) else float(value)
    return value


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _stats(values: list[float]) -> dict[str, Any]:
    arr = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=np.float64)
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "q10": float(np.quantile(arr, 0.10)),
        "q50": float(np.quantile(arr, 0.50)),
        "q90": float(np.quantile(arr, 0.90)),
        "max": float(np.max(arr)),
    }


def _point_biserial(xs: list[float], labels: list[bool]) -> dict[str, Any] | None:
    x = np.asarray([float(v) for v in xs], dtype=np.float64)
    y = np.asarray([bool(v) for v in labels], dtype=bool)
    mask = np.isfinite(x)
    x = x[mask]
    y = y[mask]
    if x.size < 8 or len(set(y.tolist())) != 2:
        return None
    true = x[y]
    false = x[~y]
    if true.size == 0 or false.size == 0:
        return None
    std = float(np.std(x))
    corr = None
    if std > 0.0:
        yy = y.astype(np.float64)
        corr = float(np.corrcoef(x, yy)[0, 1])
        if not math.isfinite(corr):
            corr = None
    return {
        "n": int(x.size),
        "true_n": int(true.size),
        "false_n": int(false.size),
        "true_mean": float(np.mean(true)),
        "false_mean": float(np.mean(false)),
        "mean_diff_true_minus_false": float(np.mean(true) - np.mean(false)),
        "point_biserial_corr": corr,
    }


def _top_parameter_links(h_list: list[dict[str, Any]], labels: list[bool], *, top_k: int = 8) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in PARAMETER_FIELDS:
        xs = []
        for h in h_list:
            value = _finite_float(h.get(key))
            if value is None:
                xs.append(float("nan"))
            else:
                xs.append(value)
        item = _point_biserial(xs, labels)
        if item is None:
            continue
        item["field"] = key
        corr = item.get("point_biserial_corr")
        item["_rank"] = abs(float(corr)) if corr is not None else abs(float(item["mean_diff_true_minus_false"]))
        rows.append(item)
    rows.sort(key=lambda item: float(item["_rank"]), reverse=True)
    for item in rows:
        item.pop("_rank", None)
    return rows[: int(top_k)]


def _load_h_list(path: str | Path) -> list[dict[str, Any]]:
    value = _load_json(path)
    if not isinstance(value, list):
        raise TypeError(f"expected fixed h list at {path}, got {type(value).__name__}")
    return [dict(item) for item in value]


def _label_vector(per_env: list[dict[str, Any]], name: str) -> list[bool]:
    if name == "closed_loop_core":
        return [bool(row.get("pendulum_like_core")) for row in per_env]
    if name == "settle_strict":
        return [bool(row.get("pendulum_like_settle_strict")) for row in per_env]
    if name == "time_profile_core":
        return [bool(row.get("time_profile_core")) for row in per_env]
    if name == "early_causal_carryover":
        return [bool(row.get("early_only_causal_carryover_core")) for row in per_env]
    if name == "early_strong_carryover":
        return [bool(row.get("early_only_strong_carryover_core")) for row in per_env]
    if name == "prefix_feedback_oracle":
        return [
            bool(row.get("prefix_oracle_is_feedback")) and bool(row.get("prefix_oracle_beats_open_loop"))
            for row in per_env
        ]
    if name == "oracle_global_gap":
        return [bool(row.get("oracle_beats_global_by_margin")) for row in per_env]
    raise KeyError(name)


def _summarize_pendulum_signature(path: str | Path) -> dict[str, Any]:
    report = _load_json(path)
    out_lists = []
    for item in report.get("lists", []):
        classification = item.get("classification", {})
        coverage = classification.get("coverage", {})
        ident = classification.get("identifiability_proxy", {})
        prefix = classification.get("prefix_immediate_reward_proxy", {})
        scores = classification.get("scores", {})
        per_env = classification.get("per_env", [])
        h_path = item.get("fixed_h_list_json")
        h_list = _load_h_list(h_path) if h_path else []
        if h_list and len(h_list) != len(per_env):
            raise RuntimeError(f"h/per-env length mismatch for {item.get('label')}: {len(h_list)} vs {len(per_env)}")
        label_names = [
            "closed_loop_core",
            "settle_strict",
            "time_profile_core",
            "early_causal_carryover",
            "early_strong_carryover",
            "prefix_feedback_oracle",
            "oracle_global_gap",
        ]
        label_summaries = {}
        for label_name in label_names:
            labels = _label_vector(per_env, label_name)
            label_summaries[label_name] = {
                "count": int(sum(labels)),
                "ratio": float(sum(labels) / max(1, len(labels))),
                "top_parameter_links": _top_parameter_links(h_list, labels) if h_list else [],
            }
        out_lists.append(
            {
                "label": item.get("label"),
                "fixed_h_list_json": h_path,
                "coverage": {
                    key: coverage.get(key)
                    for key in (
                        "nonzero_required_env_ratio",
                        "feedback_beats_zero_env_ratio",
                        "feedback_beats_open_loop_env_ratio",
                        "phase_sensitive_feedback_ratio",
                        "suffix_supported_ratio",
                        "pendulum_like_core_ratio",
                        "pendulum_like_strict_ratio",
                        "pendulum_like_settle_strict_ratio",
                        "time_profile_core_ratio",
                        "early_only_causal_carryover_core_ratio",
                        "early_only_strong_carryover_core_ratio",
                        "global_feedback_matches_oracle_ratio",
                        "oracle_beats_global_by_margin_ratio",
                        "prefix_oracle_is_feedback_ratio",
                        "prefix_oracle_beats_open_loop_ratio",
                    )
                },
                "identifiability": {
                    "global_feedback_mode_by_mean_env_return": ident.get("global_feedback_mode_by_mean_env_return"),
                    "oracle_feedback_mean_env_return": ident.get("oracle_feedback_mean_env_return"),
                    "global_feedback_mean_env_return": ident.get("global_feedback_mean_env_return"),
                    "oracle_minus_global_feedback_mean_env_return": ident.get("oracle_minus_global_feedback_mean_env_return"),
                    "oracle_minus_global_feedback_core_mean_env_return": ident.get(
                        "oracle_minus_global_feedback_core_mean_env_return"
                    ),
                    "best_feedback_mode_entropy_nats": ident.get("best_feedback_mode_entropy_nats"),
                    "paired_feedback_best_sign_counts": ident.get("paired_feedback_best_sign_counts"),
                },
                "prefix_identifiability": {
                    "global_prefix_mode_by_mean_prefix_env_return": prefix.get(
                        "global_prefix_mode_by_mean_prefix_env_return"
                    ),
                    "oracle_prefix_mean_env_return": prefix.get("oracle_prefix_mean_env_return"),
                    "global_prefix_mean_env_return": prefix.get("global_prefix_mean_env_return"),
                    "oracle_minus_global_prefix_mean_env_return": prefix.get(
                        "oracle_minus_global_prefix_mean_env_return"
                    ),
                    "best_prefix_mode_entropy_nats": prefix.get("best_prefix_mode_entropy_nats"),
                },
                "score_summaries": {
                    key: scores.get(key)
                    for key in (
                        "best_feedback_minus_open_env",
                        "oracle_minus_global_feedback_env",
                        "oracle_minus_global_feedback_env_core",
                        "best_prefix_minus_open_loop_env",
                        "best_prefix_minus_global_prefix_env",
                        "best_time_profile_minus_open_env",
                        "best_early_only_suffix_gain_over_open_env",
                    )
                },
                "long_horizon_labels": label_summaries,
            }
        )
    return {
        "schema": report.get("schema"),
        "input_report": str(Path(path).expanduser().resolve()),
        "contract": report.get("contract", {}),
        "lists": out_lists,
    }


def _summarize_gym_pendulum(path: str | Path, hidden_rollout_path: str | Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if Path(path).expanduser().exists():
        report = _load_json(path)
        summaries = report.get("summaries", {})
        for source in ("fit_policy", "official_sb3_single_env_policy", "random", "zero"):
            item = summaries.get(source, {})
            out[source] = {
                "raw_episode_return_mean": (item.get("raw_episode_return") or {}).get("mean"),
                "early_abs_action_mean": (item.get("early_abs_action") or {}).get("mean"),
                "late_abs_action_mean": (item.get("late_abs_action") or {}).get("mean"),
                "late_upright_frac_abs_theta_lt_0_5_mean": (
                    item.get("late_upright_frac_abs_theta_lt_0_5") or {}
                ).get("mean"),
                "early_corr_action_theta_dot": (
                    item.get("early_action_state_correlation") or {}
                ).get("corr_action_theta_dot"),
            }
    if Path(hidden_rollout_path).expanduser().exists():
        hidden = _load_json(hidden_rollout_path)
        out["hidden_head_closed_loop"] = {
            mode: {
                "return_mean": (payload or {}).get("return_mean"),
                "len_mean": (payload or {}).get("len_mean"),
            }
            for mode, payload in (hidden.get("mode_results") or {}).items()
        }
        out["hidden_head_train_action_metrics"] = {
            key: hidden.get("train_head_action_clipped_metrics", {}).get(key)
            for key in ("corr", "r2", "mae", "sign_match_rate", "low_pred_high_teacher_rate")
        }
    return out


def _milestone_evaluation(parity_path: str | Path, next_audit_path: str | Path) -> dict[str, Any]:
    parity = _load_json(parity_path)
    next_audit = _load_json(next_audit_path)
    return {
        "input_reports": {
            "parity_gate": str(Path(parity_path).expanduser().resolve()),
            "next_milestone_necessity": str(Path(next_audit_path).expanduser().resolve()),
        },
        "sampled_topology": {
            "status": "protect_do_not_reopen_without_corruption_evidence",
            "solved": [
                "healthy sampled-topology generator contract",
                "pack-to-fit_model generator parity",
                "pack PPO update parity under the milestone contract",
            ],
            "not_claimed": [
                "Gym-near reward-path width-bias repair",
                "closed-loop long-horizon controller identifiability",
            ],
            "parity_check": (parity.get("checks") or {}).get("sampled_topology", {}),
        },
        "gated_reward_path_balance": {
            "status": "current_protected_milestone",
            "solved": [
                "very-high reward-path width-bias support failure",
                "full/q90 guarded generation yield",
                "short fixed-list optimizability non-regression",
                "pack-to-fit_model generator/PPO parity",
            ],
            "not_claimed": [
                "terminal/survival repair",
                "dimension sampler change",
                "long-horizon closed-loop feedback subtype coverage",
                "fit_model long-run validation convergence",
            ],
            "parity_check": (parity.get("checks") or {}).get("gated_reward_path_balance", {}),
        },
        "gate_status": {
            "all_milestones_uncorrupted": bool(parity.get("all_milestones_uncorrupted")),
            "all_milestone_parity_hard_pass": bool(next_audit.get("all_milestone_parity_hard_pass")),
            "dual_support_hard_pass": bool(next_audit.get("dual_support_hard_pass")),
            "previous_next_milestone_needed_now": (next_audit.get("derived_decision") or {}).get(
                "next_milestone_needed_now"
            ),
            "previous_keep_current_milestone": (next_audit.get("derived_decision") or {}).get(
                "keep_current_milestone"
            ),
        },
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    pendulum = _summarize_pendulum_signature(args.pendulum_signature_json)
    milestone = _milestone_evaluation(args.milestone_parity_json, args.next_milestone_audit_json)
    gym = _summarize_gym_pendulum(args.gym_pendulum_mechanism_json, args.hidden_head_rollout_json)
    return {
        "analysis_entry": "phase2_long_horizon_control_axis_audit",
        "exploratory_only": True,
        "does_not_modify_exact_scm": True,
        "does_not_sample_or_train": True,
        "contract": {
            "ppo_algorithm_not_under_test": True,
            "environment_side_screening_only": True,
            "uses_existing_reports_only": True,
            "protected_milestones_must_not_be_reopened_without_corruption_or_new_dominant_evidence": True,
            "protected_dimension_samplers": sorted(PROTECTED_DIMENSION_FIELDS),
        },
        "milestone_evaluation": milestone,
        "gym_pendulum_anchor": gym,
        "prior_long_horizon_signature": pendulum,
        "long_horizon_axis_definition": {
            "name": "context_identifiable_closed_loop_settle_feedback",
            "positive_labels": [
                "nonzero action is necessary",
                "state-conditioned feedback beats open-loop action",
                "feedback sign/phase is clear",
                "feedback has suffix/settle benefit, not just prefix reward",
                "a context selector can recover useful feedback mode/sign/gain on heldout envs",
            ],
            "nearest_failure_it_targets": (
                "Pendulum fit_model emits near-zero uncorrelated actions while official PPO uses "
                "early high-gain and later stabilizing feedback."
            ),
            "not_equivalent_to": [
                "large action norm by itself",
                "random/open-loop energy injection",
                "survival reward addition",
                "dimension-distribution narrowing",
            ],
        },
        "prior_screening_method": {
            "stage_1_existing_milestone_sample": (
                "Generate candidates with the protected gated_reward_path_balance milestone."
            ),
            "stage_2_behavioral_counterfactual_labels": [
                "roll out zero/open-loop/random/feedback/decay/early-then-zero controllers on same seeds",
                "label closed_loop_core, settle_strict, time_profile_core, early_carryover, oracle_global_gap",
                "do not accept a label based on parameter values alone",
            ],
            "stage_3_quota_selection": [
                "keep already healthy/full-diverse candidates",
                "target a moderate closed-loop/settle quota before fit_model long-run, e.g. 20-30%",
                "balance sign, feature index, gain scale, and full/q90 source",
                "preserve low-energy, open-loop-useful, terminal-sensitive, and broad full-distribution support",
            ],
            "stage_4_short_sufficiency_tests": [
                "context-to-controller selector heldout rollout beats global/open-loop and approaches oracle",
                "fixed-list short PPO smoke improves raw env return and late stability without action collapse",
                "only after both pass should a fit_model 1-2 day run be worth starting",
            ],
        },
        "current_decision": {
            "new_milestone_now": False,
            "next_experiment": "context_to_controller_selector_probe_on_gated_full_q90_fixed_lists",
            "why": (
                "Existing evidence shows long-horizon closed-loop signatures exist but are heterogeneous; "
                "the next falsifiable question is whether context/frozen hidden can select the right "
                "mode/sign/gain before changing generator defaults."
            ),
        },
    }


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    milestone = report["milestone_evaluation"]
    lines = [
        "# Long-Horizon Control Axis Audit",
        "",
        "Exploratory read-only report. It does not sample, train, or mutate exact SCM / fit_model defaults.",
        "",
        "## Milestone Boundary",
        "",
        f"- all_milestones_uncorrupted: {milestone['gate_status']['all_milestones_uncorrupted']}",
        f"- current protected milestone: `{milestone['gate_status']['previous_keep_current_milestone']}`",
        "- `sampled_topology` solved topology health/parity; do not reopen it for reward-path or long-horizon issues.",
        "- `gated_reward_path_balance` solved reward-path width bias and full/q90 guarded yield; do not reinterpret it as terminal/survival or long-horizon repair.",
        "",
        "## Gym Pendulum Anchor",
    ]
    gym = report.get("gym_pendulum_anchor", {})
    for source in ("fit_policy", "official_sb3_single_env_policy", "random", "zero"):
        item = gym.get(source, {})
        if not item:
            continue
        lines.append(
            f"- {source}: return={item.get('raw_episode_return_mean')}, "
            f"early_abs_action={item.get('early_abs_action_mean')}, "
            f"late_abs_action={item.get('late_abs_action_mean')}, "
            f"late_upright={item.get('late_upright_frac_abs_theta_lt_0_5_mean')}"
        )
    hidden = gym.get("hidden_head_closed_loop", {})
    if hidden:
        lines.append(
            "- hidden ridge head closed-loop did not recover Pendulum: "
            + ", ".join(f"{k}={v.get('return_mean')}" for k, v in hidden.items())
        )
    lines.extend(["", "## Prior Signature"])
    for item in report["prior_long_horizon_signature"]["lists"]:
        cov = item["coverage"]
        ident = item["identifiability"]
        lines.extend(
            [
                f"### {item['label']}",
                f"- closed-loop core: {cov.get('pendulum_like_core_ratio')}",
                f"- settle strict: {cov.get('pendulum_like_settle_strict_ratio')}",
                f"- time-profile core: {cov.get('time_profile_core_ratio')}",
                f"- early causal carryover: {cov.get('early_only_causal_carryover_core_ratio')}",
                f"- global feedback matches oracle: {cov.get('global_feedback_matches_oracle_ratio')}",
                f"- oracle beats global by margin: {cov.get('oracle_beats_global_by_margin_ratio')}",
                f"- oracle-global mean gap: {ident.get('oracle_minus_global_feedback_mean_env_return')}",
                "",
            ]
        )
        for label_name in ("settle_strict", "time_profile_core", "oracle_global_gap"):
            label = item["long_horizon_labels"].get(label_name, {})
            links = label.get("top_parameter_links", [])[:3]
            if not links:
                continue
            text = ", ".join(
                f"{link['field']} corr={link.get('point_biserial_corr')} diff={link.get('mean_diff_true_minus_false')}"
                for link in links
            )
            lines.append(f"- parameter clues for `{label_name}`: {text}")
        lines.append("")
    lines.extend(
        [
            "## Screening Method",
            "",
            "1. Sample with the protected `gated_reward_path_balance` milestone.",
            "2. Label candidates by behavior counterfactuals, not by parameter values alone.",
            "3. Select a moderate closed-loop/settle quota while balancing sign, feature, gain, and full/q90 source.",
            "4. Run a context-to-controller selector heldout rollout before any fit_model long run.",
            "",
            "Decision: no new milestone now. Next experiment is `context_to_controller_selector_probe_on_gated_full_q90_fixed_lists`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pendulum-signature-json", default=DEFAULT_PENDULUM_SIGNATURE)
    parser.add_argument("--milestone-parity-json", default=DEFAULT_MILESTONE_PARITY)
    parser.add_argument("--next-milestone-audit-json", default=DEFAULT_NEXT_MILESTONE_AUDIT)
    parser.add_argument("--gym-pendulum-mechanism-json", default=DEFAULT_GYM_PENDULUM_MECHANISM)
    parser.add_argument("--hidden-head-rollout-json", default=DEFAULT_HIDDEN_HEAD_ROLLOUT)
    args = parser.parse_args()

    report = build_report(args)
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "long_horizon_control_axis_audit_report.json"
    md_path = out_dir / "long_horizon_control_axis_audit_summary.md"
    json_path.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(report, md_path)
    print(json.dumps({"report": str(json_path), "summary": str(md_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
