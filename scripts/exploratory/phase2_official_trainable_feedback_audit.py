#!/usr/bin/env python
"""Audit official-trainable feedback coverage and nearest failure mechanisms.

Exploratory/read-only.  Consumes previous transition-causal and GAE/lambda
decomposition reports.  It does not run training and does not mutate prior,
fit_model, or PPO code.

Definition used here:
    official_trainable_feedback := transition_and_primary environment whose
    official_actor_advantage_train_normalized alignment with the intervention-
    beneficial feedback direction is positive.

This is intentionally stricter and closer to the real failure than oracle
counterfactual labels or per-env centered returns: it asks whether the mode is
actually reinforced by the official PPO actor objective.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.exploratory.phase2_prior_action_mode_coverage_probe import _summary  # noqa: E402
from scripts.exploratory.phase2_transition_causal_context_gate import _candidate_labels  # noqa: E402
from scripts.exploratory.phase2_transition_causal_paired_intervention_gate import _binary_auc  # noqa: E402


DEFAULT_TRANSITION_REPORT = (
    "/home/chen/RLPFN/artifacts/phase2_signed_transition_causal_path_probe_0509/"
    "signed_transition_causal_path_probe_report.json"
)
DEFAULT_GAE_REPORT = (
    "/home/chen/RLPFN/artifacts/phase2_transition_causal_gae_lambda_decomposition_probe_0509/"
    "transition_causal_gae_lambda_decomposition_probe_report.json"
)
DEFAULT_OUTPUT_DIR = (
    "/home/chen/RLPFN/artifacts/phase2_official_trainable_feedback_audit_0509"
)


CORE_MECHANISM_KEYS = (
    "best_reward_pair_gap",
    "best_suffix_reward_pair_gap",
    "best_state_pair_gap",
    "best_late_state_pair_gap",
    "best_action_pair_gap",
    "best_done_pair_gap",
    "reward_gap_per_action_gap",
    "state_gap_per_action_gap",
    "suffix_reward_gap_per_late_state_gap",
)

NUISANCE_KEYS = (
    "ctrl_reward_weight",
)


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _feature_auc(y: np.ndarray, x: np.ndarray) -> float | None:
    y = np.asarray(y, dtype=bool)
    x = np.asarray(x, dtype=np.float64)
    if y.size != x.size or int(y.sum()) == 0 or int((~y).sum()) == 0:
        return None
    return _binary_auc(y, x)


def _feature_split(name: str, values: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    y = np.asarray(y, dtype=bool)
    pos = values[y]
    neg = values[~y]
    pos_mean = float(np.mean(pos)) if pos.size else None
    neg_mean = float(np.mean(neg)) if neg.size else None
    diff = None if pos_mean is None or neg_mean is None else float(pos_mean - neg_mean)
    pooled = None
    if pos.size and neg.size:
        pooled_std = float(np.std(values))
        pooled = None if pooled_std <= 0.0 else float((pos_mean - neg_mean) / pooled_std)
    auc = _feature_auc(y, values)
    strength = None if auc is None else float(max(auc, 1.0 - auc))
    return {
        "feature": str(name),
        "positive": _summary(pos),
        "negative": _summary(neg),
        "mean_diff_pos_minus_neg": diff,
        "effect_size_over_all_std": pooled,
        "auc": auc,
        "auc_strength": strength,
    }


def _load_h_list(path: str | Path) -> list[dict[str, Any]]:
    return list(_load_json(path))


def _numeric_h_features(h_list: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    keys: set[str] = set()
    for h in h_list:
        for key, value in h.items():
            if isinstance(value, (bool, int, float)) and key not in {"fixed_frozen_h_json"}:
                keys.add(str(key))
    out = {}
    for key in sorted(keys):
        arr = np.asarray([_safe_float(h.get(key, 0.0)) for h in h_list], dtype=np.float64)
        if np.isfinite(arr).all() and float(np.std(arr)) > 1e-12:
            out[key] = arr
    return out


def _source_per_env(gae_item: dict[str, Any], source: str) -> np.ndarray:
    return np.asarray(
        gae_item["alignment_by_credit_source"][source]["per_env"],
        dtype=np.float64,
    )


def _scan_one(transition_item: dict[str, Any], gae_item: dict[str, Any]) -> dict[str, Any]:
    per_env = list(transition_item["per_env"])
    labels = _candidate_labels(per_env)
    transition_and_primary = np.asarray(labels["transition_and_primary"], dtype=bool)
    transition_candidate = np.asarray(labels["transition_causal_candidate"], dtype=bool)
    primary = np.asarray(labels["primary_long_horizon_target"], dtype=bool)

    official_actor = _source_per_env(gae_item, "official_actor_advantage_train_normalized")
    official_centered = _source_per_env(gae_item, "official_value_gae_lambda_0.9")
    official_uncentered = _source_per_env(gae_item, "official_value_gae_lambda_0.9_uncentered")
    zero_centered = _source_per_env(gae_item, "zero_value_gae_lambda_0.9")
    zero_uncentered = _source_per_env(gae_item, "zero_value_gae_lambda_0.9_uncentered")
    reward_step = _source_per_env(gae_item, "reward_step_centered")
    official_values = _source_per_env(gae_item, "official_values")
    negative_values = _source_per_env(gae_item, "negative_official_values")

    official_trainable = transition_and_primary & (official_actor > 0.0)
    official_trainable_strong = transition_and_primary & (official_actor > 0.05)
    raw_supported = transition_and_primary & (zero_centered > 0.0)
    centered_supported_but_train_lost = transition_and_primary & (official_centered > 0.0) & (official_actor <= 0.0)

    rows = []
    for idx, row in enumerate(per_env):
        rows.append(
            {
                "env_index": int(row.get("env_index", idx)),
                "transition_causal_candidate": bool(transition_candidate[idx]),
                "primary_long_horizon_target": bool(primary[idx]),
                "transition_and_primary": bool(transition_and_primary[idx]),
                "official_trainable_feedback": bool(official_trainable[idx]),
                "official_trainable_feedback_strong": bool(official_trainable_strong[idx]),
                "raw_supported_transition_primary": bool(raw_supported[idx]),
                "centered_supported_but_train_lost": bool(centered_supported_but_train_lost[idx]),
                "best_pair_name": row.get("best_pair_name"),
                "best_pair_sign": row.get("best_pair_sign"),
                "official_actor_advantage_train_normalized_alignment": float(official_actor[idx]),
                "official_value_gae_lambda_0.9_centered_alignment": float(official_centered[idx]),
                "official_value_gae_lambda_0.9_uncentered_alignment": float(official_uncentered[idx]),
                "zero_value_gae_lambda_0.9_centered_alignment": float(zero_centered[idx]),
                "zero_value_gae_lambda_0.9_uncentered_alignment": float(zero_uncentered[idx]),
                "reward_step_centered_alignment": float(reward_step[idx]),
                "official_values_alignment": float(official_values[idx]),
                "negative_official_values_alignment": float(negative_values[idx]),
                "official_centering_drop": float(official_centered[idx] - official_actor[idx]),
                "zero_centering_drop": float(zero_centered[idx] - zero_uncentered[idx]),
                **{key: _safe_float(row.get(key, 0.0)) for key in CORE_MECHANISM_KEYS + NUISANCE_KEYS},
            }
        )

    y_trainable_within_tp = official_trainable[transition_and_primary]
    mechanism_features: dict[str, np.ndarray] = {
        key: np.asarray([_safe_float(row.get(key, 0.0)) for row in per_env], dtype=np.float64)
        for key in CORE_MECHANISM_KEYS + NUISANCE_KEYS
    }
    mechanism_features.update(
        {
            "official_centering_drop": official_centered - official_actor,
            "zero_centering_drop": zero_centered - zero_uncentered,
            "official_centered_minus_zero_centered": official_centered - zero_centered,
            "official_values_alignment": official_values,
            "negative_official_values_alignment": negative_values,
            "reward_step_centered_alignment": reward_step,
            "zero_value_gae_lambda_0.9_centered_alignment": zero_centered,
            "official_value_gae_lambda_0.9_centered_alignment": official_centered,
        }
    )

    split_all = [
        _feature_split(key, values, official_trainable)
        for key, values in mechanism_features.items()
        if float(np.std(values)) > 1e-12
    ]
    split_all.sort(
        key=lambda item: (
            -1.0 if item["auc_strength"] is None else -float(item["auc_strength"]),
            str(item["feature"]),
        )
    )

    split_tp = []
    if int(transition_and_primary.sum()) > 0 and len(set(y_trainable_within_tp.tolist())) >= 2:
        for key, values in mechanism_features.items():
            v = np.asarray(values, dtype=np.float64)[transition_and_primary]
            if float(np.std(v)) > 1e-12:
                split_tp.append(_feature_split(key, v, y_trainable_within_tp))
        split_tp.sort(
            key=lambda item: (
                -1.0 if item["auc_strength"] is None else -float(item["auc_strength"]),
                str(item["feature"]),
            )
        )

    h_list = _load_h_list(transition_item["fixed_h_list_json"])
    h_features = _numeric_h_features(h_list)
    h_splits = [
        _feature_split(f"h.{key}", values, official_trainable)
        for key, values in h_features.items()
        if key not in NUISANCE_KEYS
    ]
    h_splits.sort(
        key=lambda item: (
            -1.0 if item["auc_strength"] is None else -float(item["auc_strength"]),
            str(item["feature"]),
        )
    )

    return {
        "label": str(transition_item["label"]),
        "n_envs": int(len(per_env)),
        "counts": {
            "transition_causal_candidate": int(transition_candidate.sum()),
            "primary_long_horizon_target": int(primary.sum()),
            "transition_and_primary": int(transition_and_primary.sum()),
            "official_trainable_feedback": int(official_trainable.sum()),
            "official_trainable_feedback_within_transition_primary": int(
                (official_trainable & transition_and_primary).sum()
            ),
            "official_trainable_feedback_strong": int(official_trainable_strong.sum()),
            "raw_supported_transition_primary": int(raw_supported.sum()),
            "centered_supported_but_train_lost": int(centered_supported_but_train_lost.sum()),
        },
        "rates": {
            "official_trainable_feedback": float(official_trainable.mean()) if official_trainable.size else 0.0,
            "official_trainable_given_transition_primary": (
                float(official_trainable[transition_and_primary].mean())
                if int(transition_and_primary.sum()) > 0
                else 0.0
            ),
            "raw_supported_given_transition_primary": (
                float(raw_supported[transition_and_primary].mean())
                if int(transition_and_primary.sum()) > 0
                else 0.0
            ),
            "centered_supported_but_train_lost_given_transition_primary": (
                float(centered_supported_but_train_lost[transition_and_primary].mean())
                if int(transition_and_primary.sum()) > 0
                else 0.0
            ),
        },
        "alignment_summaries": {
            "official_actor_advantage_train_normalized": _summary(official_actor),
            "official_actor_on_transition_primary": _summary(official_actor[transition_and_primary]),
            "official_centered_on_transition_primary": _summary(official_centered[transition_and_primary]),
            "zero_centered_on_transition_primary": _summary(zero_centered[transition_and_primary]),
        },
        "top_mechanism_splits_all_envs": split_all[:16],
        "top_mechanism_splits_within_transition_primary": split_tp[:16],
        "top_h_scalar_splits_all_envs": h_splits[:16],
        "per_env": rows,
        "contract": {
            "official_trainable_feedback_definition": (
                "transition_and_primary AND official_actor_advantage_train_normalized_alignment > 0"
            ),
            "strong_definition": (
                "transition_and_primary AND official_actor_advantage_train_normalized_alignment > 0.05"
            ),
            "nuisance_note": "ctrl_reward_weight is reported only as a nuisance/guard, not as a proposed root repair axis.",
        },
    }


def _fmt(value: Any) -> str:
    return "na" if value is None else f"{float(value):+.4f}"


def _write_summary(report: dict[str, Any], output_dir: Path) -> None:
    lines = [
        "# Official-Trainable Feedback Audit",
        "",
        "Exploratory/read-only. It audits whether transition-causal modes are reinforced by the official PPO actor objective.",
        "",
        "| list | TP | official-trainable TP | raw-supported TP | centered-supported-but-train-lost TP | trainable given TP | raw given TP | lost given TP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report["lists"]:
        c = item["counts"]
        r = item["rates"]
        lines.append(
            "| "
            f"{item['label']} | {c['transition_and_primary']} | "
            f"{c['official_trainable_feedback_within_transition_primary']} | "
            f"{c['raw_supported_transition_primary']} | "
            f"{c['centered_supported_but_train_lost']} | "
            f"{r['official_trainable_given_transition_primary']:.3f} | "
            f"{r['raw_supported_given_transition_primary']:.3f} | "
            f"{r['centered_supported_but_train_lost_given_transition_primary']:.3f} |"
        )
    lines.append("")
    for item in report["lists"]:
        lines.extend(
            [
                f"## {item['label']}",
                "",
                "Top within-transition-primary splits, positive class = official-trainable feedback:",
                "",
                "| feature | pos mean | neg mean | diff | AUC strength |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in item["top_mechanism_splits_within_transition_primary"][:10]:
            lines.append(
                "| "
                f"{row['feature']} | "
                f"{_fmt(row['positive']['mean'])} | {_fmt(row['negative']['mean'])} | "
                f"{_fmt(row['mean_diff_pos_minus_neg'])} | {_fmt(row['auc_strength'])} |"
            )
        if not item["top_mechanism_splits_within_transition_primary"]:
            lines.append("| na | na | na | na | na |")
        lines.extend(
            [
                "",
                "Top h scalar splits across all envs, positive class = official-trainable feedback:",
                "",
                "| feature | pos mean | neg mean | diff | AUC strength |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in item["top_h_scalar_splits_all_envs"][:10]:
            lines.append(
                "| "
                f"{row['feature']} | "
                f"{_fmt(row['positive']['mean'])} | {_fmt(row['negative']['mean'])} | "
                f"{_fmt(row['mean_diff_pos_minus_neg'])} | {_fmt(row['auc_strength'])} |"
            )
        lines.append("")
    lines.extend(
        [
            "Interpretation:",
            "",
            "- If official-trainable TP is rare while raw-supported TP is common, the missing piece is not merely oracle transition causality; it is PPO-trainable credit under the real actor objective.",
            "- A split feature is a repair candidate only if it is proximal, stable, and can be controlled without collapsing full/q90 health or diversity.",
            "- h scalar splits are audit clues, not repair instructions.",
        ]
    )
    (output_dir / "official_trainable_feedback_audit_summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transition-report", type=str, default=DEFAULT_TRANSITION_REPORT)
    parser.add_argument("--gae-report", type=str, default=DEFAULT_GAE_REPORT)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    transition = _load_json(args.transition_report)
    gae = _load_json(args.gae_report)
    gae_by_label = {str(item["label"]): item for item in gae["lists"]}
    lists = []
    for item in transition["lists"]:
        label = str(item["label"])
        if label not in gae_by_label:
            raise KeyError(f"Missing GAE item for transition list {label!r}")
        lists.append(_scan_one(item, gae_by_label[label]))
    report = {
        "analysis_entry": "phase2_official_trainable_feedback_audit",
        "config": vars(args),
        "contract": {
            "exploratory_only": True,
            "consumes_existing_reports_only": True,
            "does_not_modify_prior_or_ppo": True,
        },
        "lists": lists,
    }
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "official_trainable_feedback_audit_report.json").write_text(
        json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_summary(report, out_dir)
    print(out_dir / "official_trainable_feedback_audit_summary.md")


if __name__ == "__main__":
    main()
