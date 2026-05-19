#!/usr/bin/env python
"""Read-only audit for the Pendulum-specific long-horizon control mode.

This report intentionally does not sample, train, or mutate the prior
generator.  It separates the already-protected milestone repairs from the
new candidate mechanism needed for Pendulum-like generalization.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_DIR = "/home/chen/RLPFN/artifacts/phase2_pendulum_control_mode_sufficiency_audit_0509"
DEFAULT_REWARD_STATE_SENTINEL = (
    "/home/chen/RLPFN/artifacts/phase2_reward_state_gain_semantic_sentinel/summary.json"
)
DEFAULT_NEXT_MILESTONE_AUDIT = (
    "/home/chen/RLPFN/artifacts/phase2_prior_next_milestone_necessity_audit_0506/"
    "prior_next_milestone_necessity_audit_report.json"
)
DEFAULT_MODE_EXTRACT = (
    "/home/chen/RLPFN/artifacts/phase2_long_horizon_control_mode_extract_0509/"
    "long_horizon_control_mode_extract_report.json"
)
DEFAULT_SELECTOR_PROBE = (
    "/home/chen/RLPFN/artifacts/phase2_context_controller_selector_probe_0509/"
    "context_controller_selector_probe_report.json"
)
DEFAULT_PENDULUM_MECHANISM = (
    "/home/chen/RLPFN/artifacts/phase2_gym_pendulum_mechanism_signature_probe_0509/"
    "fit_epoch29_vs_official_single_env_prefix_v2.json"
)
DEFAULT_HIDDEN_HEAD_ROLLOUT = (
    "/home/chen/RLPFN/artifacts/phase2_gym_pendulum_frozen_hidden_head_rollout_probe_0509/"
    "fit_epoch29_hidden_head_rollout_train32_eval16.json"
)


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def _get(obj: Any, *keys: str, default: Any = None) -> Any:
    value = obj
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def _finite(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _metric_summary(block: dict[str, Any], metric: str) -> dict[str, Any]:
    item = block.get(metric, {})
    if not isinstance(item, dict):
        return {}
    fields = ("count", "mean", "std", "min", "q10", "q50", "q90", "max")
    return {key: item.get(key) for key in fields if key in item}


def _pendulum_summary(report: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for source in (
        "fit_policy",
        "official_sb3_single_env_policy",
        "random",
        "zero",
    ):
        summary = _get(report, "summaries", source, default={}) or {}
        out[source] = {
            "raw_episode_return_mean": _get(summary, "raw_episode_return", "mean"),
            "early_abs_action_mean": _get(summary, "early_abs_action", "mean"),
            "late_abs_action_mean": _get(summary, "late_abs_action", "mean"),
            "late_upright_frac_abs_theta_lt_0_5_mean": _get(
                summary, "late_upright_frac_abs_theta_lt_0_5", "mean"
            ),
            "early_action_state_correlation": summary.get("early_action_state_correlation"),
            "late_action_state_correlation": summary.get("late_action_state_correlation"),
        }
    return out


def _hidden_rollout_summary(report: dict[str, Any]) -> dict[str, Any]:
    modes = {}
    for key in ("fit", "hidden_head_eval_full", "official_eval_full"):
        block = _get(report, "mode_results", key, default={}) or {}
        modes[key] = {
            "return_mean": block.get("return_mean"),
            "len_mean": block.get("len_mean"),
        }
        detail = block.get("summary")
        if isinstance(detail, dict):
            modes[key]["eval_fit_vs_official_r2"] = _get(detail, "eval", "fit_vs_official", "r2")
            modes[key]["eval_ridge_vs_official_r2"] = _get(
                detail, "eval", "ridge_vs_official", "r2"
            )
            modes[key]["eval_ridge_abs_mean"] = _get(detail, "eval", "ridge_abs", "mean")
            modes[key]["eval_official_abs_mean"] = _get(detail, "eval", "official_abs", "mean")
    metrics = report.get("train_head_action_clipped_metrics", {}) or {}
    return {
        "mode_results": modes,
        "offline_teacher_head_metrics": {
            "r2": metrics.get("r2"),
            "corr": metrics.get("corr"),
            "mae": metrics.get("mae"),
            "sign_match_rate": metrics.get("sign_match_rate"),
            "low_pred_high_teacher_rate": metrics.get("low_pred_high_teacher_rate"),
        },
    }


def _mode_extract_summary(report: dict[str, Any]) -> dict[str, Any]:
    lists = []
    for item in report.get("lists", []) or []:
        subsets = item.get("subsets", {}) or {}
        primary = subsets.get("primary_long_horizon_target", {}) or {}
        hard = subsets.get("selector_hard_positive", {}) or {}
        easy = subsets.get("selector_easy_positive", {}) or {}
        deficit = item.get("primary_target_deficit", {}) or {}
        lists.append(
            {
                "label": item.get("label"),
                "short_label": item.get("short_label"),
                "n_envs": item.get("n_envs"),
                "primary_target_n": primary.get("n"),
                "primary_target_ratio": (
                    None
                    if primary.get("n") is None or item.get("n_envs") is None
                    else float(primary.get("n")) / float(max(1, int(item.get("n_envs"))))
                ),
                "hard_selector_n": hard.get("n"),
                "easy_selector_n": easy.get("n"),
                "primary_target_deficit": deficit,
                "primary_feedback_gain_counts": primary.get("feedback_gain_counts"),
                "primary_feedback_profile_counts": primary.get("feedback_profile_counts"),
                "primary_time_profile_counts": primary.get("time_profile_profile_counts"),
                "primary_feedback_sign_counts": primary.get("feedback_sign_counts"),
                "best_feedback_minus_open_env": _metric_summary(primary, "best_feedback_minus_open_env"),
                "oracle_minus_global_feedback_env": _metric_summary(
                    primary, "oracle_minus_global_feedback_env"
                ),
            }
        )
    return {
        "mode_definition": report.get("mode_definition"),
        "lists": lists,
    }


def _selector_task_summary(task: dict[str, Any]) -> dict[str, Any]:
    summary = _get(task, "cv", "summary", default={}) or {}
    return {
        "valid": _get(task, "cv", "valid"),
        "class_counts": task.get("class_counts"),
        "auc_mean": summary.get("auc_mean"),
        "accuracy_mean": summary.get("accuracy_mean"),
        "majority_baseline_accuracy_mean": summary.get("majority_baseline_accuracy_mean"),
        "improvement_over_majority_mean": summary.get("improvement_over_majority_mean"),
        "positive_rate_test_mean": summary.get("positive_rate_test_mean"),
    }


def _selector_summary(report: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for feature_set in (
        "visible_context_all_modes",
        "visible_context_half_random",
        "visible_context_zero",
        "h_with_dims_upper_bound",
        "h_no_dims_upper_bound",
    ):
        block = _get(report, "feature_set_results", feature_set, default={}) or {}
        tasks = block.get("tasks", {}) or {}
        out[feature_set] = {
            "n_features": block.get("n_features"),
            "n_samples": block.get("n_samples"),
            "is_primary_long_horizon_target": _selector_task_summary(
                tasks.get("is_primary_long_horizon_target", {}) or {}
            ),
            "is_selector_hard_positive": _selector_task_summary(
                tasks.get("is_selector_hard_positive", {}) or {}
            ),
            "primary_feedback_sign": _selector_task_summary(
                tasks.get("primary_feedback_sign", {}) or {}
            ),
            "primary_feedback_feature": _selector_task_summary(
                tasks.get("primary_feedback_feature", {}) or {}
            ),
            "primary_feedback_gain": _selector_task_summary(
                tasks.get("primary_feedback_gain", {}) or {}
            ),
            "primary_time_profile": _selector_task_summary(
                tasks.get("primary_time_profile", {}) or {}
            ),
        }
    return out


def _build_report(args: argparse.Namespace) -> dict[str, Any]:
    sentinel = _load_json(args.reward_state_sentinel)
    milestone = _load_json(args.next_milestone_audit)
    mode_extract = _load_json(args.mode_extract)
    selector = _load_json(args.selector_probe)
    pendulum = _load_json(args.pendulum_mechanism)
    hidden = _load_json(args.hidden_head_rollout)

    previous_repairs = {
        "state_reward_identity_and_decoherence": {
            "evidence_path": str(Path(args.reward_state_sentinel).expanduser().resolve()),
            "trusted_passed": sentinel.get("passed"),
            "threshold": sentinel.get("threshold"),
            "what_it_guards": (
                "Reward/return must keep distinguishable state identity and not collapse into "
                "noise or an action-only reward surface."
            ),
            "why_not_sufficient_for_pendulum": (
                "It does not assert that an action can move future state into a better basin, "
                "nor that the correct signed feedback controller is inferable from context."
            ),
        },
        "gated_reward_path_balance": {
            "evidence_path": str(Path(args.next_milestone_audit).expanduser().resolve()),
            "milestone_kept": _get(milestone, "derived_decision", "keep_current_milestone"),
            "next_milestone_needed_at_that_time": _get(
                milestone, "derived_decision", "next_milestone_needed_now"
            ),
            "short_optim_delta_summary": milestone.get("short_optim_delta_summary"),
            "what_it_guards": (
                "Reward path width/support and full/q90 topology health under fixed-list short training."
            ),
            "why_not_sufficient_for_pendulum": (
                "It changes only reward-path balance.  It keeps state/action/terminal paths unchanged "
                "and does not create a context-identifiable closed-loop controller."
            ),
        },
    }

    purified_mode = {
        "name": "context_identifiable_signed_closed_loop_contraction",
        "negative_boundary": [
            "not merely high action magnitude",
            "not merely early action injection",
            "not merely reward depending on state",
            "not merely a healthy reward/action topology",
            "not survival/terminal reward shaping",
        ],
        "necessary_clauses": [
            {
                "name": "state_visible_phase",
                "meaning": "The observation/context exposes which side and velocity phase the controller is in.",
                "prior_locator": "visible-context summaries must predict the relevant feedback feature/sign above majority.",
            },
            {
                "name": "signed_action_causal_path",
                "meaning": (
                    "Paired positive/negative actions from the same state/noise produce different future "
                    "states, and one sign improves future state reward."
                ),
                "prior_locator": "same-seed counterfactual +u/-u/0 action tests on suffix state/reward.",
            },
            {
                "name": "closed_loop_contraction",
                "meaning": (
                    "A state-conditioned feedback controller improves suffix reward and reduces late "
                    "state error/instability, not just prefix reward."
                ),
                "prior_locator": "feedback beats open-loop, suffix_supported, and late-contraction witness.",
            },
            {
                "name": "context_identifiable_readout",
                "meaning": (
                    "The correct sign/gain/profile is learnable from policy-visible context, not only from "
                    "hidden h fields or an oracle choosing per-env actions."
                ),
                "prior_locator": "group-split selector from visible context beats majority for sign/gain/profile.",
            },
        ],
        "why_this_is_pendulum_sufficient_if_present": (
            "Pendulum official PPO succeeds by mapping angle/velocity phase to large early torque and "
            "then smaller stabilizing feedback.  A prior family containing the four clauses above gives "
            "the meta-policy the same causal problem: infer phase and feedback sign/gain from context, "
            "apply torque to move the future state basin, then settle.  If any clause fails, the policy "
            "can still learn healthy reward identities while outputting near-zero or wrong-phase actions."
        ),
    }

    evidence = {
        "gym_pendulum_anchor": _pendulum_summary(pendulum),
        "hidden_teacher_closed_loop_anchor": _hidden_rollout_summary(hidden),
        "prior_mode_extraction": _mode_extract_summary(mode_extract),
        "context_selector_probe": _selector_summary(selector),
        "current_read": {
            "coarse_mode_exists": (
                "Primary long-horizon targets are present at roughly 26-30 percent in the audited "
                "fixed lists, so the immediate issue is not total absence."
            ),
            "mode_is_not_clean_enough": (
                "The target is dominated by gain=2.0 and constant/decay profiles, and visible-context "
                "selectors do not robustly recover sign/gain/profile above majority."
            ),
            "pendulum_failure_location": (
                "The nearest failure is initial/early policy readout: fit actions remain near zero while "
                "official PPO uses strong signed actions.  Early-only actions are still insufficient, so "
                "the missing subtype must include sustained closed-loop settling."
            ),
        },
    }

    priori_locator = {
        "screen_name": "signed_action_causal_contraction_screen",
        "contract": {
            "environment_side_only": True,
            "ppo_algorithm_not_under_test": True,
            "do_not_modify_exact_scm_or_fit_model": True,
            "same_initial_state_noise_for_counterfactuals": True,
        },
        "paired_rollout_tests": [
            "zero action",
            "open-loop random or half-random",
            "feedback +gain on visible obs feature",
            "feedback -gain on the same visible obs feature",
            "decay feedback and sustained feedback profiles",
        ],
        "labels_to_compute": [
            {
                "name": "signed_future_reward_gap",
                "definition": "max(+feedback, -feedback) suffix env reward minus open-loop suffix env reward.",
                "acceptance": "positive by an env-scale margin on heldout rollout seeds.",
            },
            {
                "name": "late_contraction_witness",
                "definition": (
                    "late state/reward proxy improves relative to prefix and to open-loop; if state proxy "
                    "is unavailable, use suffix env reward with action decay as a weaker witness."
                ),
                "acceptance": "must be paired with signed_future_reward_gap, never used alone.",
            },
            {
                "name": "controller_identifiability",
                "definition": "visible context predicts sign/gain/profile under group split by env seed.",
                "acceptance": "must beat majority baseline for the controller labels, not only the binary target.",
            },
            {
                "name": "closed_loop_smoke",
                "definition": "fixed-list short PPO improves env_ret without relying on extra reward terms.",
                "acceptance": "no degradation in health, reward/state/done diversity, or fixed-list optimizability.",
            },
        ],
    }

    repair_methods = [
        {
            "stage": "screening_only",
            "method": (
                "Keep both protected milestones unchanged; add an exploratory post-sampling screen that "
                "selects a bounded quota of context-identifiable signed contraction tasks."
            ),
            "when_allowed": (
                "Only after the new screen predicts sign/gain/profile from visible context and fixed-list "
                "short training does not regress."
            ),
        },
        {
            "stage": "minimal_generator_knob",
            "method": (
                "If healthy yield is too low, locate the generator parameter that changes action-to-future-"
                "state causal strength without changing action_dim/obs_dim/state_dim or reward identity.  "
                "Tune only that parameter and re-run the same screen."
            ),
            "when_allowed": (
                "Only after paired finite-difference probes show the knob causally changes the missing "
                "screened label, not just a correlated surface metric."
            ),
        },
        {
            "stage": "reject_or_defer",
            "method": (
                "If the mode cannot be identified from visible context, do not quota it into a milestone; "
                "a larger fraction of unidentifiable oracle-only tasks would be distribution noise."
            ),
            "when_allowed": "Default if selector remains at or below majority after label purification.",
        },
    ]

    return {
        "analysis_entry": "phase2_pendulum_control_mode_sufficiency_audit",
        "exploratory_only": True,
        "does_not_sample_or_train": True,
        "does_not_modify_exact_scm": True,
        "does_not_modify_fit_model": True,
        "input_reports": {
            "reward_state_sentinel": str(Path(args.reward_state_sentinel).expanduser().resolve()),
            "next_milestone_audit": str(Path(args.next_milestone_audit).expanduser().resolve()),
            "mode_extract": str(Path(args.mode_extract).expanduser().resolve()),
            "selector_probe": str(Path(args.selector_probe).expanduser().resolve()),
            "pendulum_mechanism": str(Path(args.pendulum_mechanism).expanduser().resolve()),
            "hidden_head_rollout": str(Path(args.hidden_head_rollout).expanduser().resolve()),
        },
        "previous_repairs_boundary": previous_repairs,
        "purified_mode_definition": purified_mode,
        "current_evidence": evidence,
        "a_priori_location_method": priori_locator,
        "repair_methods": repair_methods,
        "decision": {
            "new_milestone_now": False,
            "reason": (
                "The needed Pendulum-like mode is clearer, but current labels are not yet sufficiently "
                "context-identifiable.  The next useful action is the signed_action_causal_contraction_screen, "
                "not a generator change."
            ),
        },
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "na"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    pend = _get(report, "current_evidence", "gym_pendulum_anchor", default={}) or {}
    fit = pend.get("fit_policy", {}) or {}
    official = pend.get("official_sb3_single_env_policy", {}) or {}
    random = pend.get("random", {}) or {}
    hidden = _get(report, "current_evidence", "hidden_teacher_closed_loop_anchor", default={}) or {}
    offline = hidden.get("offline_teacher_head_metrics", {}) or {}
    hidden_modes = hidden.get("mode_results", {}) or {}
    mode_lists = _get(report, "current_evidence", "prior_mode_extraction", "lists", default=[]) or []
    selector = _get(report, "current_evidence", "context_selector_probe", default={}) or {}

    lines = [
        "# Pendulum Control Mode Sufficiency Audit",
        "",
        "This is read-only and exploratory. It does not sample, train, or mutate exact SCM / fit_model.",
        "",
        "## Boundary From Previous Repairs",
        "",
        "- `state_reward_identity_and_decoherence` guards state-distinguishable reward/return identity. It does not prove action-to-future-state controllability or context-readable controller sign.",
        "- `gated_reward_path_balance` repairs reward-path support and full/q90 health. It deliberately keeps state/action/terminal paths unchanged, so it cannot by itself add a Pendulum-like closed-loop controller.",
        "",
        "## Purified Required Mode",
        "",
        "`context_identifiable_signed_closed_loop_contraction`: visible state phase + signed action causal path + suffix closed-loop contraction + context-identifiable sign/gain/profile.",
        "",
        "This is stricter than high action, early kick, reward identity, survival reward, or generic healthy topology.",
        "",
        "## Gym Pendulum Anchor",
        "",
        "| source | return | early abs action | late abs action | late upright frac |",
        "| --- | ---: | ---: | ---: | ---: |",
        (
            f"| fit | {_fmt(fit.get('raw_episode_return_mean'))} | "
            f"{_fmt(fit.get('early_abs_action_mean'))} | {_fmt(fit.get('late_abs_action_mean'))} | "
            f"{_fmt(fit.get('late_upright_frac_abs_theta_lt_0_5_mean'))} |"
        ),
        (
            f"| official | {_fmt(official.get('raw_episode_return_mean'))} | "
            f"{_fmt(official.get('early_abs_action_mean'))} | {_fmt(official.get('late_abs_action_mean'))} | "
            f"{_fmt(official.get('late_upright_frac_abs_theta_lt_0_5_mean'))} |"
        ),
        (
            f"| random | {_fmt(random.get('raw_episode_return_mean'))} | "
            f"{_fmt(random.get('early_abs_action_mean'))} | {_fmt(random.get('late_abs_action_mean'))} | "
            f"{_fmt(random.get('late_upright_frac_abs_theta_lt_0_5_mean'))} |"
        ),
        "",
        "Offline hidden-head action prediction alone is not enough: "
        f"train clipped R2={_fmt(offline.get('r2'))}, but closed-loop hidden-head return="
        f"{_fmt(_get(hidden_modes, 'hidden_head_eval_full', 'return_mean'))}, official return="
        f"{_fmt(_get(hidden_modes, 'official_eval_full', 'return_mean'))}.",
        "",
        "## Prior Evidence",
        "",
    ]
    for item in mode_lists:
        lines.extend(
            [
                (
                    f"- {item.get('short_label')}: primary target "
                    f"{_fmt(item.get('primary_target_n'))}/{_fmt(item.get('n_envs'))} "
                    f"ratio={_fmt(item.get('primary_target_ratio'))}"
                ),
                f"  - feedback gain counts: {item.get('primary_feedback_gain_counts')}",
                f"  - feedback profile counts: {item.get('primary_feedback_profile_counts')}",
                f"  - time-profile witness counts: {item.get('primary_time_profile_counts')}",
            ]
        )
    lines.extend(
        [
            "",
            "## Selector Evidence",
            "",
            "| feature set | primary AUC | sign over majority | gain over majority | profile over majority |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for name in (
        "visible_context_all_modes",
        "visible_context_half_random",
        "visible_context_zero",
        "h_with_dims_upper_bound",
    ):
        block = selector.get(name, {}) or {}
        lines.append(
            (
                f"| {name} | "
                f"{_fmt(_get(block, 'is_primary_long_horizon_target', 'auc_mean'))} | "
                f"{_fmt(_get(block, 'primary_feedback_sign', 'improvement_over_majority_mean'))} | "
                f"{_fmt(_get(block, 'primary_feedback_gain', 'improvement_over_majority_mean'))} | "
                f"{_fmt(_get(block, 'primary_time_profile', 'improvement_over_majority_mean'))} |"
            )
        )
    lines.extend(
        [
            "",
            "## A Priori Locator",
            "",
            "Use a same-state/noise counterfactual screen: zero, open-loop, +feedback, -feedback, decay feedback, and sustained feedback. A candidate only counts if it has signed future reward gap, late contraction witness, and visible-context identifiability.",
            "",
            "## Repair Ladder",
            "",
            "1. Screening only: quota a bounded amount of context-identifiable signed contraction tasks after the screen passes.",
            "2. Minimal knob only if yield is too low: tune an action-to-future-state causal-strength knob proven by paired finite differences, without changing dims or reward identity.",
            "3. Reject/defer if sign/gain/profile remains oracle-only and not visible from context.",
            "",
            "## Decision",
            "",
            "No new milestone yet. The next useful probe is `signed_action_causal_contraction_screen`; directly increasing the current coarse Pendulum-like quota would likely add unidentifiable oracle-only tasks.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reward-state-sentinel", default=DEFAULT_REWARD_STATE_SENTINEL)
    parser.add_argument("--next-milestone-audit", default=DEFAULT_NEXT_MILESTONE_AUDIT)
    parser.add_argument("--mode-extract", default=DEFAULT_MODE_EXTRACT)
    parser.add_argument("--selector-probe", default=DEFAULT_SELECTOR_PROBE)
    parser.add_argument("--pendulum-mechanism", default=DEFAULT_PENDULUM_MECHANISM)
    parser.add_argument("--hidden-head-rollout", default=DEFAULT_HIDDEN_HEAD_ROLLOUT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report = _build_report(args)
    json_path = out_dir / "pendulum_control_mode_sufficiency_audit_report.json"
    md_path = out_dir / "pendulum_control_mode_sufficiency_audit_summary.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(report, md_path)
    print(json.dumps({"report": str(json_path), "summary": str(md_path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
